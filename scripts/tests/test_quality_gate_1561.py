"""The quality gate must count experiments, not how many metrics they print."""

import contextlib
import importlib.util
import io
import itertools
import math
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch


SPEC = importlib.util.spec_from_file_location(
    "quality_gate", Path(__file__).parents[2] / "bench/aggregate_quality_gate_1561.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def line(case, label="mean", metric="rmse", gam=0.5, reference=1.0, tool="mgcv"):
    return (
        f"[QUALITY_PAIR] category=families test={label} metric={metric} "
        f"gam={gam:.17e} reference={tool} reference_value={reference:.17e} "
        f"lower_is_better=true case=families::{case}\n"
    )


class QualityGate(unittest.TestCase):
    def test_multiple_channels_and_metrics_count_as_one_experiment(self):
        rows = gate._parse([
            line("a", "mean", gam=0.25),
            line("a", "scale", gam=0.5),
            line("a", "scale", metric="mae", gam=2.0),
            line("b", gam=4.0),
        ])
        summary = gate._summarize("all", rows)
        self.assertEqual((summary["n"], summary["scored"], summary["pairs"]), (2, 2, 4))
        self.assertEqual((summary["gam_wins"], summary["reference_wins"]), (1, 1))
        self.assertAlmostEqual(summary["median_log_ratio"], math.log(2.0) / 2)

    def test_repeating_a_whole_metric_panel_does_not_create_significance(self):
        panel = [line("a", "mean"), line("a", "scale", gam=0.75), line("b", gam=2.0)]
        original = gate._summarize("all", gate._parse(panel))
        repeated = gate._summarize("all", gate._parse(panel * 20))
        self.assertEqual(original, repeated)

    def test_sibling_cases_with_the_same_label_are_distinct(self):
        rows = gate._parse([line("module::first"), line("module::second")])
        self.assertEqual(len(rows), 2)
        self.assertEqual(gate._summarize("all", rows)["scored"], 2)

    def test_distinct_reference_tools_are_retained_inside_the_case(self):
        rows = gate._parse([line("a", tool="mgcv"), line("a", tool="gamlss", gam=2.0)])
        self.assertEqual(len(rows), 2)
        summary = gate._summarize("all", rows)
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["ties"], 1)

    def test_missing_case_and_conflicting_repeated_results_refuse(self):
        with self.assertRaisesRegex(ValueError, "missing libtest case"):
            gate._parse([line("a").split(" case=")[0]])
        with self.assertRaisesRegex(ValueError, "conflicting repeated"):
            gate._parse([line("a"), line("a", gam=2.0)])

    def test_invalid_channel_invalidates_the_entire_experiment(self):
        for invalid in (-1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                rows = gate._parse([line("a"), line("a", "scale", gam=invalid)])
                summary = gate._summarize("all", rows)
                self.assertEqual(summary["scored"], 0)
                self.assertEqual(summary["dropped_nonfinite"], 1)

    def test_exact_zero_errors_preserve_ties_and_boundary_dominance(self):
        rows = gate._parse([
            line("exact_tie", gam=0.0, reference=0.0),
            line("exact_win", gam=0.0),
            line("exact_loss", reference=0.0),
        ])
        summary = gate._summarize("all", rows)
        self.assertEqual(summary["scored"], 3)
        self.assertEqual(summary["dropped_nonfinite"], 0)
        self.assertEqual((summary["gam_wins"], summary["reference_wins"], summary["ties"]), (1, 1, 1))
        self.assertEqual(summary["median_log_ratio"], 0.0)

    def test_opposite_infinite_metric_endpoints_do_not_invent_a_median(self):
        rows = gate._parse([line("a", "mean", gam=0.0), line("a", "scale", reference=0.0)])
        summary = gate._summarize("all", rows)
        self.assertEqual(summary["scored"], 0)
        self.assertEqual(summary["dropped_nonfinite"], 1)

    def test_log_ratio_does_not_overflow_or_underflow(self):
        row = gate._parse([line("a", gam=1e300, reference=1e-300)])[0]
        self.assertAlmostEqual(gate._effect(row), 600 * math.log(10))
        row["lower_is_better"] = False
        self.assertAlmostEqual(gate._effect(row), -600 * math.log(10))

    def test_signed_rank_ties_use_the_same_equivalence_as_ranking(self):
        # Distinct differences separated by less than 1e-12 still have distinct
        # ranks. The variance must not treat them as a tied rank group.
        effects = [-1.0, -1.0000000000001, 2.0]
        w, p = gate._wilcoxon_less(effects)
        self.assertEqual(w, 3.0)
        self.assertEqual(p, 5 / 8)
        self.assertEqual(gate._wilcoxon_less([0.0, 0.0]), (0.0, 1.0))

    def test_exact_signed_rank_matches_exhaustive_tied_sign_assignments(self):
        magnitudes = [1, 1, 2, 4, 4]
        doubled_ranks = [3, 3, 6, 9, 9]
        assignments = list(itertools.product([-1, 1], repeat=len(magnitudes)))
        null_sums = [sum(r for r, sign in zip(doubled_ranks, signs) if sign > 0)
                    for signs in assignments]
        for signs, observed in zip(assignments, null_sums):
            effects = [m * sign for m, sign in zip(magnitudes, signs)] + [0.0]
            w, p = gate._wilcoxon_less(effects)
            self.assertEqual(w, observed / 2)
            self.assertEqual(p, sum(s <= observed for s in null_sums) / len(null_sums))

    def test_small_panels_cannot_gain_significance_from_a_normal_approximation(self):
        self.assertEqual(gate._wilcoxon_less([-1.0] * 4), (0.0, 1 / 16))
        self.assertEqual(gate._wilcoxon_less([-1.0] * 5), (0.0, 1 / 32))

    def report(self, extra="", manifest_count=40):
        body = "".join(line(f"module::case{i}") for i in range(40)) + extra
        output = io.StringIO()
        argv = ["gate", "-"]
        manifest = ""
        if manifest_count is not None:
            argv += ["--cases", "quality_cases.txt"]
            manifest = "".join(f"families::module::case{i}\n" for i in range(manifest_count))
        with patch.object(gate.sys, "argv", argv), \
                patch("builtins.open", mock_open(read_data=manifest)), \
                patch.object(gate.sys, "stdin", io.StringIO(body)), \
                contextlib.redirect_stdout(output):
            gate.main()
        return output.getvalue()

    def test_significant_survivors_do_not_close_a_failed_or_unrecorded_suite(self):
        for extra in ("", "FAIL [1.0s] gam::quality families::module::missing\n"):
            with self.subTest(extra=extra):
                report = self.report(extra)
                self.assertIn("CLOSURE", report)
                self.assertIn("): FAIL", report)
                self.assertIn("closure blocked", report)

    def test_sibling_emission_does_not_hide_a_failed_case(self):
        report = self.report("FAIL [1.0s] gam::quality families::module::missing\n")
        self.assertIn("SILENT ATTRITION: 1 test(s)", report)
        self.assertIn("families::module::missing", report)

    def test_a_passing_sibling_does_not_certify_unrecorded_emitters(self):
        report = self.report("PASS [1.0s] gam::quality families::module::unrelated\n")
        self.assertIn("): FAIL", report)
        self.assertIn("unrecorded emitting case: families::module::case0", report)

    def test_recorded_successful_case_panel_can_pass(self):
        execution = "".join(
            f"PASS [1.0s] gam::quality families::module::case{i}\n" for i in range(40)
        )
        self.assertIn("): PASS", self.report(execution))

    def test_successful_subset_cannot_certify_the_complete_suite(self):
        execution = "".join(
            f"PASS [1.0s] gam::quality families::module::case{i}\n" for i in range(40)
        )
        report = self.report(execution, manifest_count=41)
        self.assertIn("): FAIL", report)
        self.assertIn("listed case was not executed: families::module::case40", report)
        self.assertIn("): FAIL", self.report(execution, manifest_count=None))
        report = self.report(execution, manifest_count=39)
        self.assertIn("): FAIL", report)
        self.assertIn("executed case absent from manifest: families::module::case39", report)


if __name__ == "__main__":
    unittest.main()
