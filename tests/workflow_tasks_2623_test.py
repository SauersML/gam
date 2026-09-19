import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASKS_PATH = _REPO_ROOT / ".github" / "scripts" / "workflow_tasks.py"
_BENCHMARK_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "benchmark.yml"
_SPEC = importlib.util.spec_from_file_location("workflow_tasks_2623", _TASKS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load {_TASKS_PATH}")
_TASKS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TASKS)


class WorkflowTasks2623Tests(unittest.TestCase):
    def test_binary_bootstrap_loads_the_extension_without_the_unstaged_facade(self) -> None:
        workflow = _BENCHMARK_WORKFLOW_PATH.read_text()
        build_step = workflow.split(
            "      - name: Build compiled benchmark runtime", 1
        )[1].split("      - name: Save compiled benchmark runtime", 1)[0]
        self.assertIn("load_gamfit_rust_module", build_step)
        self.assertNotIn(
            "import gamfit",
            build_step,
            "the bootstrap interpreter does not own staged NumPy/Pandas; "
            "the binary smoke check must use the direct extension loader",
        )

    def test_aggregate_sparse_checkout_contains_its_repository_dependencies(self) -> None:
        workflow = _BENCHMARK_WORKFLOW_PATH.read_text()
        aggregate = workflow.split("  aggregate:", 1)[1]
        checkout = aggregate.split("- name: Download all shard artifacts", 1)[0]
        for required_path in (
            "bench",
            ".github/scripts/workflow_tasks.py",
            ".github/actions/publish-gha-results",
        ):
            self.assertIn(
                required_path,
                checkout,
                f"aggregate checkout omits repository dependency {required_path}",
            )

    def test_targeted_matrix_preserves_requested_order_and_rejects_unknown_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "output"
            with patch.dict(
                os.environ,
                {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_OUTPUT": str(output)},
                clear=False,
            ):
                _TASKS.build_matrix("wine_temp_vs_year,papuan_oce4_duchon_k6")
            values = dict(
                line.split("=", 1)
                for line in output.read_text().splitlines()
            )
            # #2737: matrix entries are `include` objects now -- each carries its
            # own wall budget and job ceiling alongside the scenario name -- so
            # order is read off the entries rather than off a bare name list.
            self.assertEqual(
                [
                    entry["scenario"]
                    for entry in json.loads(values["parallel_matrix"])["include"]
                ],
                ["wine_temp_vs_year", "papuan_oce4_duchon_k6"],
            )
            with patch.dict(
                os.environ,
                {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_OUTPUT": str(output)},
                clear=False,
            ):
                with self.assertRaisesRegex(SystemExit, "Unknown benchmark scenario"):
                    _TASKS.build_matrix("does_not_exist")

    def test_manual_all_matrix_matches_the_complete_scheduled_matrix(self) -> None:
        def matrix_for(event_name: str, requested: str | None) -> dict[str, str]:
            with tempfile.TemporaryDirectory() as td:
                output = Path(td) / "output"
                with patch.dict(
                    os.environ,
                    {"GITHUB_EVENT_NAME": event_name, "GITHUB_OUTPUT": str(output)},
                    clear=False,
                ):
                    _TASKS.build_matrix(requested)
                return dict(
                    line.split("=", 1) for line in output.read_text().splitlines()
                )

        scheduled = matrix_for("schedule", None)
        for requested in (None, "", "all", "ALL", "*"):
            manual = matrix_for("workflow_dispatch", requested)
            self.assertEqual(manual["parallel_matrix"], scheduled["parallel_matrix"])
            self.assertEqual(manual["parallel_count"], scheduled["parallel_count"])
            self.assertEqual(manual["serial_matrix"], scheduled["serial_matrix"])
            self.assertEqual(manual["serial_count"], scheduled["serial_count"])

        selected_count = int(scheduled["parallel_count"]) + int(
            scheduled["serial_count"]
        )
        configured_count = len(_TASKS._load_scenario_config()["scenarios"])
        self.assertEqual(selected_count, configured_count)

    def test_matched_verdict_checks_both_speed_measures_and_every_accuracy_direction(self) -> None:
        rows = [
            {
                "scenario_name": "wine_temp_vs_year",
                "contender": "rust_gam",
                "status": "ok",
                "fit_sec": 1.2,
                "predict_sec": 0.6,
                "rmse": 0.4,
                "r2": 0.8,
            },
            {
                "scenario_name": "wine_temp_vs_year",
                "contender": "r_mgcv",
                "status": "ok",
                "fit_sec": 1.0,
                "predict_sec": 0.5,
                "rmse": 0.4,
                "r2": 0.8,
            },
        ]
        verdict = _TASKS.matched_benchmark_verdict(rows)
        comparison = verdict["comparisons"][0]
        self.assertTrue(comparison["passed"])
        self.assertTrue(verdict["observed_scope_certified"])
        self.assertFalse(verdict["certified"], "one targeted scenario cannot certify the full suite")

        rows[0]["rmse"] = 0.4 + 0.5 * _TASKS.ACCURACY_NUMERICAL_EQUIVALENCE
        verdict = _TASKS.matched_benchmark_verdict(rows)
        comparison = verdict["comparisons"][0]
        self.assertTrue(
            comparison["passed"],
            "sub-sqrt(epsilon) cross-language differences are numerical equivalence",
        )

        rows[0]["rmse"] = 0.4000001
        verdict = _TASKS.matched_benchmark_verdict(rows)
        comparison = verdict["comparisons"][0]
        failed = [m["measure"] for m in comparison["accuracy"] if not m["passed"]]
        self.assertEqual(failed, ["rmse"])
        self.assertFalse(comparison["passed"])
        self.assertFalse(verdict["observed_scope_certified"])

        rows[0]["rmse"] = 0.4
        rows[0]["r2"] = 0.7999999
        verdict = _TASKS.matched_benchmark_verdict(rows)
        comparison = verdict["comparisons"][0]
        failed = [m["measure"] for m in comparison["accuracy"] if not m["passed"]]
        self.assertEqual(failed, ["r2"])
        self.assertFalse(comparison["passed"])

    def test_matched_verdict_does_not_invent_inapplicable_reference_pairs(self) -> None:
        rows = [
            {
                "scenario_name": "papuan_oce4_duchon_k6",
                "contender": "rust_gam",
                "status": "ok",
                "fit_sec": 1.0,
                "predict_sec": 0.1,
                "auc": 0.8,
            },
            {
                "scenario_name": "papuan_oce4_duchon_k6",
                "contender": "r_mgcv",
                "status": "ok",
                "fit_sec": 1.0,
                "predict_sec": 0.1,
                "auc": 0.8,
            },
            {
                "scenario_name": "papuan_oce4_duchon_k6",
                "contender": "rust_gamlss",
                "status": "failed",
            },
        ]
        verdict = _TASKS.matched_benchmark_verdict(rows)
        self.assertEqual(len(verdict["comparisons"]), 1)
        self.assertEqual(verdict["comparisons"][0]["gam_contender"], "rust_gam")


class BenchmarkRegressionGate2623Tests(unittest.TestCase):
    """The verdict was published and read by nobody; these pin the consumer."""

    @staticmethod
    def _matched_rows(*, gam_fit_sec):
        shared = {"predict_sec": 0.1, "auc": 0.8, "brier": 0.1}
        return [
            {
                "scenario_name": "papuan_oce4_duchon_k6",
                "contender": "rust_gam",
                "status": "ok",
                "fit_sec": gam_fit_sec,
                **shared,
            },
            {
                "scenario_name": "papuan_oce4_duchon_k6",
                "contender": "r_mgcv",
                "status": "ok",
                "fit_sec": 1.7,
                **shared,
            },
        ]

    def test_the_issue_headline_regression_fails_the_gate(self) -> None:
        # 423 s/fold against mgcv's 1.7 s/fold: the measurement that opened
        # #2623 and that no job could fail on.
        verdict = _TASKS.matched_benchmark_verdict(self._matched_rows(gam_fit_sec=423.0))
        outcome = _TASKS.benchmark_verdict_enforcement(verdict)
        self.assertFalse(outcome["passed"])
        self.assertTrue(
            any("fit_sec is 248." in failure for failure in outcome["failures"]),
            outcome["failures"],
        )

    def test_a_matched_arm_inside_the_bound_passes_the_gate(self) -> None:
        verdict = _TASKS.matched_benchmark_verdict(self._matched_rows(gam_fit_sec=1.8))
        outcome = _TASKS.benchmark_verdict_enforcement(verdict)
        self.assertTrue(outcome["passed"], outcome["failures"])

    def test_a_run_that_measured_nothing_fails_rather_than_passes_vacuously(self) -> None:
        # Every shard dying produced zero comparisons, and "nothing failed" is
        # how this regression stayed invisible. Absence must not be a pass.
        verdict = _TASKS.matched_benchmark_verdict([])
        outcome = _TASKS.benchmark_verdict_enforcement(verdict)
        self.assertFalse(outcome["passed"])
        self.assertTrue(
            any("measured nothing" in f or "no matched comparison" in f for f in outcome["failures"]),
            outcome["failures"],
        )

    def test_full_suite_requirement_is_opt_in_so_a_targeted_probe_is_graded_on_its_scope(
        self,
    ) -> None:
        verdict = _TASKS.matched_benchmark_verdict(self._matched_rows(gam_fit_sec=1.8))
        self.assertTrue(verdict["missing_scenarios"])
        self.assertTrue(_TASKS.benchmark_verdict_enforcement(verdict)["passed"])
        strict = _TASKS.benchmark_verdict_enforcement(verdict, require_full_suite=True)
        self.assertFalse(strict["passed"])

    def test_an_accuracy_regression_fails_the_gate_too(self) -> None:
        rows = self._matched_rows(gam_fit_sec=1.0)
        rows[0]["auc"] = 0.7
        outcome = _TASKS.benchmark_verdict_enforcement(_TASKS.matched_benchmark_verdict(rows))
        self.assertFalse(outcome["passed"])
        self.assertTrue(any("auc regressed" in f for f in outcome["failures"]), outcome["failures"])

    def test_the_aggregate_job_enforces_the_verdict_after_publishing_evidence(self) -> None:
        workflow = _BENCHMARK_WORKFLOW_PATH.read_text()
        aggregate = workflow.split("  aggregate:", 1)[1]
        self.assertIn("enforce_verdict", aggregate)
        # A gate that runs before publication costs us the evidence of the
        # regression it just caught.
        self.assertLess(
            aggregate.index("- name: Publish results to bench/gha_results"),
            aggregate.index("enforce_verdict"),
            "the regression gate must run after the results are published",
        )


if __name__ == "__main__":
    unittest.main()
