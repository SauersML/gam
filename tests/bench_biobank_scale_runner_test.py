import typing
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_PATH = _REPO_ROOT / "bench" / "biobank_scale" / "runner.py"
_SPEC = importlib.util.spec_from_file_location("bench_biobank_scale_runner", _RUNNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load biobank benchmark runner from {_RUNNER_PATH}")
_RUNNER: typing.Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _write_csv(path: Path, rows: typing.Sequence[typing.Mapping[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BiobankScaleRunnerTests(unittest.TestCase):
    def test_default_biobank_matrix_keeps_400k_binomial_marginal_slope_lane(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)

        self.assertEqual(int(cfg["target_n"]), 400000)

        specs = _RUNNER.build_method_specs(cfg)

        marginal_slope_disease = [
            s
            for s in specs
            if s.dataset == "disease"
            and s.backend == "rust_gam"
            and s.family == "binomial"
            and s.marginal_slope
        ]
        self.assertEqual(
            len(marginal_slope_disease),
            1,
            "expected exactly one disease + Rust + binomial + marginal-slope lane in the "
            "default biobank matrix; found "
            f"{[s.name for s in marginal_slope_disease]}",
        )
        lane = marginal_slope_disease[0]

        self.assertEqual(lane.spatial_basis, "duchon")
        self.assertEqual(lane.pc_count, 16, f"{lane.name} must run on 16 PCs")
        self.assertTrue(lane.scale_dimensions, f"{lane.name} must enable per-axis scales")
        self.assertEqual(lane.z_column, "pgs_ctn_z", f"{lane.name} must read CTN z column")

        self.assertIsNotNone(
            lane.mean_linkwiggle_knots,
            f"{lane.name} must enable mean linkwiggle (production calibration)",
        )
        self.assertGreaterEqual(int(lane.mean_linkwiggle_knots), 1)
        self.assertIsNotNone(
            lane.logslope_linkwiggle_knots,
            f"{lane.name} must enable score-warp linkwiggle on the logslope side",
        )
        self.assertGreaterEqual(int(lane.logslope_linkwiggle_knots), 1)

        self.assertIsNotNone(
            lane.max_centers,
            f"{lane.name} must declare a max-centers cap for biobank scale",
        )
        self.assertLessEqual(
            int(lane.centers),
            int(lane.max_centers),
            f"{lane.name} centers={lane.centers} exceeds its own max_centers cap "
            f"{lane.max_centers}",
        )

        capped_at_biobank_n = _RUNNER.effective_marginal_slope_centers(
            lane, train_rows=int(cfg["target_n"])
        )
        self.assertLessEqual(
            capped_at_biobank_n,
            int(lane.max_centers),
            f"{lane.name} effective centers at n={cfg['target_n']} ({capped_at_biobank_n}) "
            f"must not exceed max_centers cap {lane.max_centers}",
        )

    def test_marginal_slope_formula_supports_linkwiggle_and_scorewarp(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="margslope_variant",
            dataset="disease",
            backend="rust_gam",
            family="binomial",
            spatial_basis="duchon",
            marginal_slope=True,
            scale_dimensions=True,
            z_column="pgs_ctn_z",
            mean_linkwiggle_knots=8,
            logslope_linkwiggle_knots=7,
        )
        mean_formula, logslope_formula = _RUNNER.rust_marginal_slope_formula_classification(spec, centers=20)
        self.assertIn("duchon(pc1_std, pc2_std", mean_formula)
        self.assertIn("centers=20", mean_formula)
        self.assertIn("order=1", mean_formula)
        self.assertIn("power=8", mean_formula)
        self.assertIn("length_scale=1", mean_formula)
        self.assertNotIn("pgs_ctn_z", mean_formula)
        self.assertIn("linkwiggle(internal_knots=8)", mean_formula)
        self.assertIn("linkwiggle(internal_knots=7)", logslope_formula)

    def test_effective_marginal_slope_centers_caps_biobank_and_wiggle_modes(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="margslope_variant",
            dataset="disease",
            backend="rust_gam",
            family="binomial",
            spatial_basis="duchon",
            centers=50,
            marginal_slope=True,
            logslope_linkwiggle_knots=8,
            max_centers=30,
        )
        self.assertEqual(_RUNNER.effective_marginal_slope_centers(spec, train_rows=10000), 22)
        self.assertEqual(_RUNNER.effective_marginal_slope_centers(spec, train_rows=400000), 22)

    def test_biobank_preflight_rejects_unsafe_dense_duchon_width_before_allocation(self) -> None:
        report = _RUNNER.preflight_marginal_slope_biobank(
            n_train=400000,
            d_pc=16,
            centers=1400,
        )
        self.assertEqual(report.status, "FAIL")
        text = "\n".join(report.lines)
        self.assertIn("anisotropic derivative dense estimate", text)
        self.assertIn("status: FAIL", text)

    def test_biobank_preflight_accepts_production_marginal_slope_width(self) -> None:
        report = _RUNNER.preflight_marginal_slope_biobank(
            n_train=400000,
            d_pc=16,
            centers=24,
            linkwiggle_knots=8,
            scorewarp_knots=8,
        )
        self.assertEqual(report.status, "PASS")
        text = "\n".join(report.lines)
        self.assertIn("Duchon tuple: order=1, power=8, length_scale=1", text)
        self.assertIn("Duchon smooth: lazy chunked", text)
        self.assertIn("anisotropy derivatives: implicit streaming", text)

    def test_ctn_preflight_uses_factored_kronecker_not_dense_rowwise_product(self) -> None:
        report = _RUNNER.preflight_ctn_score_warp(
            n_train=400000,
            p_response=12,
            p_cov=50,
        )
        self.assertEqual(report.status, "PASS")
        text = "\n".join(report.lines)
        self.assertIn("CTN Kronecker: factored", text)
        self.assertIn("avoided dense rowwise Kronecker", text)
        self.assertLess(report.largest_single_allocation_bytes, 400000 * 600 * 8)

    def test_survival_prediction_preflight_chunks_large_horizon_grid(self) -> None:
        report = _RUNNER.preflight_survival_prediction(
            n_rows=400000,
            grid_points=1000,
        )
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.chunk_rows, _RUNNER.BIOBANK_SURVIVAL_PREDICTION_CHUNK_ROWS)
        self.assertLess(report.largest_single_allocation_bytes, 400000 * 1000 * 8)

    def test_marginal_slope_preflight_status_is_grep_friendly(self) -> None:
        # Routing log regression: the biobank preflight emits a `status: PASS`
        # / `status: FAIL` token that downstream log scrapers grep for. The
        # Rust planner emits a parallel `solver=...;hessian=...;matrix-free=...`
        # token via `OuterPlan::routing_log_line()`. Both contracts are pinned
        # together: a regression that drops the Python preflight token or the
        # Rust routing token will fail tests in either layer.
        report = _RUNNER.preflight_marginal_slope_biobank(
            n_train=400000,
            d_pc=16,
            centers=20,
            linkwiggle_knots=8,
            scorewarp_knots=7,
        )
        text = "\n".join(report.lines)
        self.assertIn("status: PASS", text)

    def test_run_method_subparser_exposes_emit_routing_log_flag(self) -> None:
        # The `--emit-routing-log` flag is the user-facing handle that turns
        # on routing-token capture for biobank lane runs. Removing it would
        # make the bench-level routing regressions (per the
        # OuterPlan::routing_log_line contract) silently impossible.
        parser = _RUNNER.build_parser()
        args = parser.parse_args(
            [
                "run-method",
                "--prep-dir",
                "/tmp/p",
                "--method",
                "x",
                "--out-dir",
                "/tmp/o",
                "--out-json",
                "/tmp/o.json",
                "--emit-routing-log",
            ]
        )
        self.assertTrue(getattr(args, "emit_routing_log", False))

    def test_add_standardized_columns_returns_replayable_training_statistics(self) -> None:
        train_rows: list[dict[str, object]] = []
        test_rows: list[dict[str, object]] = []
        for target, base in ((train_rows, 10.0), (test_rows, 20.0)):
            for idx in range(3):
                row: dict[str, object] = {
                    "age_entry": base + idx,
                    "lat_final": base + idx + 1.0,
                    "lon_final": base + idx + 2.0,
                    "pgs_raw": base + idx + 3.0,
                }
                row.update({f"pc{i}": base + idx + i for i in range(1, 17)})
                target.append(row)

        standardization = _RUNNER.add_standardized_columns(train_rows, test_rows)

        expected_columns = {
            "age_entry",
            "lat_final",
            "lon_final",
            "pgs_raw",
            *[f"pc{i}" for i in range(1, 17)],
        }
        self.assertEqual(set(standardization), expected_columns)
        self.assertAlmostEqual(standardization["age_entry"]["mean"], 11.0)
        self.assertGreater(standardization["age_entry"]["sd"], 0.0)
        self.assertIn("pgs_std", train_rows[0])
        self.assertIn("pc16_std", test_rows[0])

    def test_routing_log_scraper_captures_outer_plan_lines_only(self) -> None:
        # `_append_routing_lines` is the predicate that decides which captured
        # stderr lines reach the routing-log sidecar. It must accept the
        # `[OUTER]` log marker emitted by `log_plan` and reject unrelated
        # noise so test scrapers aren't fooled by a heartbeat line that
        # happens to contain `solver=`.
        with tempfile.TemporaryDirectory() as raw_dir:
            tmp = Path(raw_dir) / "lane.routing.log"
            stderr = (
                "[HEARTBEAT] elapsed=1.2s cmd='gam ...' pid=42 cpu=10% mem=2%\n"
                "[OUTER] reml outer: n_params=6, gradient=Analytic, hessian=Analytic"
                " -> solver=Arc, hessian_source=Analytic"
                " [solver=Arc;hessian=Analytic;matrix-free=true]\n"
                "some unrelated stderr noise mentioning solver=Cheese\n"
                "[OUTER] aux outer: n_params=2, gradient=Analytic, hessian=Unavailable"
                " -> solver=Bfgs, hessian_source=BfgsApprox"
                " [solver=Bfgs;hessian=BfgsApprox;matrix-free=false] [no Hessian: BFGS approximation]\n"
            )
            _RUNNER._append_routing_lines(tmp, stderr)
            captured = tmp.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(captured), 2, captured)
            self.assertIn(
                "solver=Arc;hessian=Analytic;matrix-free=true", captured[0]
            )
            self.assertIn(
                "solver=Bfgs;hessian=BfgsApprox;matrix-free=false", captured[1]
            )
            self.assertNotIn("HEARTBEAT", "\n".join(captured))
            self.assertNotIn("Cheese", "\n".join(captured))

    def test_build_method_specs_rejects_pc_count_above_generated_columns(self) -> None:
        cfg = {
            "methods": [
                {
                    "name": "too_many_pcs",
                    "dataset": "disease",
                    "backend": "rust_gam",
                    "family": "binomial",
                    "spatial_basis": "duchon",
                    "pc_count": 17,
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "pc_count in \\[1, 16\\]"):
            _RUNNER.build_method_specs(cfg)

    def test_marginal_slope_formula_supports_matern_basis(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="margslope_matern",
            dataset="disease",
            backend="rust_gam",
            family="binomial",
            spatial_basis="matern",
            marginal_slope=True,
            pc_count=4,
        )
        mean_formula, logslope_formula = _RUNNER.rust_marginal_slope_formula_classification(
            spec,
            centers=18,
        )
        self.assertIn("matern(pc1_std, pc2_std, pc3_std, pc4_std, centers=18)", mean_formula)
        self.assertIn("smooth(age_entry_std)", logslope_formula)

    def test_build_method_specs_rejects_legacy_survival_backend(self) -> None:
        cfg = {
            "methods": [
                {
                    "name": "legacy_survival",
                    "dataset": "survival",
                    "backend": "rust_gamlss_survival",
                    "family": "survival",
                    "spatial_basis": "ps",
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "legacy survival backend"):
            _RUNNER.build_method_specs(cfg)

    def _survival_contract_train_rows(self) -> list[dict[str, float]]:
        return [
            {
                "time": 4.0,
                "event": 1.0,
                "pgs_std": 0.1,
                "sex": 0.0,
                "age_entry_std": -1.0,
                "lat_final_std": 0.2,
                "lon_final_std": -0.3,
                "pc1_std": 0.1,
                "pc2_std": 0.2,
                "pc3_std": 0.3,
                "pc4_std": 0.4,
            },
            {
                "time": 10.0,
                "event": 0.0,
                "pgs_std": -0.2,
                "sex": 1.0,
                "age_entry_std": 0.5,
                "lat_final_std": -0.1,
                "lon_final_std": 0.6,
                "pc1_std": -0.1,
                "pc2_std": -0.2,
                "pc3_std": -0.3,
                "pc4_std": -0.4,
            },
        ]

    def _survival_contract_test_rows(self) -> list[dict[str, float]]:
        return [
            {
                "time": 6.0,
                "event": 1.0,
                "pgs_std": 0.4,
                "sex": 1.0,
                "age_entry_std": -0.4,
                "lat_final_std": 0.8,
                "lon_final_std": 0.1,
                "pc1_std": 0.5,
                "pc2_std": 0.4,
                "pc3_std": 0.3,
                "pc4_std": 0.2,
            }
        ]

    def test_run_rust_survival_uses_explicit_survival_contract(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="rust_gamlss_survival_ps",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="duchon",
            centers=24,
            survival_likelihood="location-scale",
            survival_distribution="probit",
        )
        train_rows = self._survival_contract_train_rows()
        test_rows = self._survival_contract_test_rows()
        snapshots: dict[str, typing.Any] = {}
        orig_load_bin = _RUNNER.load_or_build_rust_binary
        orig_run_cmd = _RUNNER.run_cmd_stream
        try:
            _RUNNER.load_or_build_rust_binary = lambda: Path("/tmp/fake-gam")

            def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any=None) -> typing.Any:
                if cmd[1] == "fit":
                    fit_input = Path(cmd[-2])
                    snapshots["fit_formula"] = cmd[-1]
                    snapshots["fit_cmd"] = list(cmd)
                    snapshots["fit_rows"] = _RUNNER.read_csv_rows(fit_input)
                    Path(cmd[cmd.index("--out") + 1]).write_text("{}", encoding="utf-8")
                    return 0, "", ""
                if cmd[1] == "predict":
                    input_path = Path(cmd[3])
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    input_rows = _RUNNER.read_csv_rows(input_path)
                    snapshots.setdefault("predict_inputs", []).append(input_rows)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    n = max(len(input_rows), 1)
                    with out_path.open("w", encoding="utf-8", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["survival_prob"])
                        writer.writeheader()
                        for idx in range(len(input_rows)):
                            # Monotone-decreasing survival in (0,1] across the grid so
                            # downstream lifted-metric computations stay well-defined.
                            writer.writerow(
                                {"survival_prob": float(0.99 - 0.05 * idx / max(n - 1, 1))}
                            )
                    return 0, "", ""
                raise AssertionError(f"unexpected command: {cmd}")

            _RUNNER.run_cmd_stream = _fake_run_cmd

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                train_csv = td_path / "train.csv"
                test_csv = td_path / "test.csv"
                _write_csv(train_csv, train_rows)
                _write_csv(test_csv, test_rows)
                result = _RUNNER.run_rust_survival(spec, train_csv, test_csv, td_path)
        finally:
            _RUNNER.load_or_build_rust_binary = orig_load_bin
            _RUNNER.run_cmd_stream = orig_run_cmd

        self.assertIn("Surv(__entry, time, event)", snapshots["fit_formula"])
        self.assertIn(
            "survmodel(spec=net, distribution=probit)",
            snapshots["fit_formula"],
        )
        self.assertIn("survival-likelihood=location-scale", result["model_spec"])
        fit_rows = snapshots["fit_rows"]
        self.assertTrue(all(float(row["__entry"]) == 0.0 for row in fit_rows))
        self.assertEqual([float(row["time"]) for row in fit_rows], [4.0, 10.0])

        predict_inputs = snapshots["predict_inputs"]
        # The runner must issue exactly two predict invocations: one for the
        # explicit horizon test set, then one for the stacked native survival
        # grid used to compute proper survival metrics.
        self.assertEqual(len(predict_inputs), 2)

        # First predict: one row per test row, all at the median-derived horizon.
        horizon = _RUNNER.survival_eval_horizon_from_rows(train_rows)
        self.assertEqual(len(predict_inputs[0]), len(test_rows))
        self.assertTrue(
            all(abs(float(row["time"]) - horizon) < 1e-12 for row in predict_inputs[0])
        )

        # Second predict: native survival grid stacks rows over the score grid.
        import numpy as np

        grid = _RUNNER._survival_score_grid(
            np.array([float(r["time"]) for r in train_rows], dtype=float)
        )
        expected_native_rows = len(test_rows) * grid.shape[0]
        self.assertEqual(
            len(predict_inputs[1]),
            expected_native_rows,
            f"native survival grid must stack {len(test_rows)} test rows × "
            f"{grid.shape[0]} grid points = {expected_native_rows}; "
            f"got {len(predict_inputs[1])}",
        )

        # Every row of every predict invocation must carry __entry left-truncation,
        # otherwise the survival likelihood degenerates silently.
        for invocation_idx, rows in enumerate(predict_inputs):
            for row_idx, row in enumerate(rows):
                self.assertIn(
                    "__entry",
                    row,
                    f"predict invocation {invocation_idx} row {row_idx} missing __entry",
                )
                self.assertEqual(
                    float(row["__entry"]),
                    0.0,
                    f"predict invocation {invocation_idx} row {row_idx} has non-zero __entry",
                )

        # The fit command must explicitly carry the survival likelihood mode,
        # so a silent backend swap is impossible.
        fit_cmd = snapshots["fit_cmd"]
        self.assertIn("--survival-likelihood", fit_cmd)
        self.assertEqual(
            fit_cmd[fit_cmd.index("--survival-likelihood") + 1],
            "location-scale",
        )

    def test_run_rust_survival_rejects_invalid_native_grid_columns(self) -> None:
        """A malformed native survival prediction must fail with a clear error,
        not silently report nonsense metrics."""
        spec = _RUNNER.MethodSpec(
            name="rust_gamlss_survival_ps",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="duchon",
            centers=24,
            survival_likelihood="location-scale",
            survival_distribution="probit",
        )
        train_rows = self._survival_contract_train_rows()
        test_rows = self._survival_contract_test_rows()

        orig_load_bin = _RUNNER.load_or_build_rust_binary
        orig_run_cmd = _RUNNER.run_cmd_stream
        try:
            _RUNNER.load_or_build_rust_binary = lambda: Path("/tmp/fake-gam")

            def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any=None) -> typing.Any:
                if cmd[1] == "fit":
                    Path(cmd[cmd.index("--out") + 1]).write_text("{}", encoding="utf-8")
                    return 0, "", ""
                if cmd[1] == "predict":
                    input_path = Path(cmd[3])
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    input_rows = _RUNNER.read_csv_rows(input_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    # Emit risk_score (legacy/wrong column name) instead of survival_prob.
                    with out_path.open("w", encoding="utf-8", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["risk_score"])
                        writer.writeheader()
                        for idx in range(len(input_rows)):
                            writer.writerow({"risk_score": float(idx)})
                    return 0, "", ""
                raise AssertionError(f"unexpected command: {cmd}")

            _RUNNER.run_cmd_stream = _fake_run_cmd

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                train_csv = td_path / "train.csv"
                test_csv = td_path / "test.csv"
                _write_csv(train_csv, train_rows)
                _write_csv(test_csv, test_rows)
                with self.assertRaises(RuntimeError) as ctx:
                    _RUNNER.run_rust_survival(spec, train_csv, test_csv, td_path)
                self.assertIn(spec.name, str(ctx.exception))
        finally:
            _RUNNER.load_or_build_rust_binary = orig_load_bin
            _RUNNER.run_cmd_stream = orig_run_cmd

    def test_survival_formula_rhs_supports_linkwiggle_and_timewiggle(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="surv_variant",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="duchon",
            centers=24,
            survival_likelihood="transformation",
            survival_distribution="gaussian",
            mean_linkwiggle_knots=8,
            timewiggle_knots=8,
        )
        rhs = _RUNNER.rust_survival_formula_rhs(spec)
        self.assertIn("linkwiggle(internal_knots=8)", rhs)
        self.assertIn("timewiggle(internal_knots=8)", rhs)

    def test_generate_raw_cohort_populates_pc_columns_from_each_row(self) -> None:
        cfg = {
            "seed": 1,
            "raw_subpop_n": 20,
            "observed_latlon_fraction": 0.5,
            "split_seed": 2,
            "target_n": 100,
            "smoke_target_n": 50,
        }
        with tempfile.TemporaryDirectory() as td:
            rows, _meta = _RUNNER.generate_raw_cohort(cfg, Path(td), smoke=False)
        self.assertGreater(len(rows), 20)
        pc1 = [float(r["pc1"]) for r in rows[:30]]
        pc2 = [float(r["pc2"]) for r in rows[:30]]
        self.assertGreater(len(set(round(v, 6) for v in pc1)), 10)
        self.assertGreater(len(set(round(v, 6) for v in pc2)), 10)


class MarkerPatternTests(unittest.TestCase):
    """Lock down the runner's regex patterns for the structured markers
    emitted by the gam binary. Each marker has had a regression in
    development (additional fields, format drift), so a direct test of
    `findall` against representative sample lines is the cheapest guard
    against future format changes silently breaking aggregation."""

    def test_ift_quality_pattern_parses_old_and_new_field_layout(self) -> None:
        # Old layout (pre-44d3c482): no drho_norm / h_pen_logdet.
        old = "[IFT-QUALITY] residual=3.456e-04 converged_norm=1.234e+00 predicted_norm=1.234e+00 iters=4"
        # New layout (44d3c482+): adds drho_norm and h_pen_logdet.
        new = "[IFT-QUALITY] residual=3.456e-04 converged_norm=1.234e+00 predicted_norm=1.234e+00 drho_norm=5.678e-01 h_pen_logdet=2.345e+01 iters=4"
        # NaN tokens (when the cache is empty / unstamped at the
        # emission point — defensive paths in runtime.rs).
        nan = "[IFT-QUALITY] residual=3.456e-04 converged_norm=1.234e+00 predicted_norm=1.234e+00 drho_norm=NaN h_pen_logdet=NaN iters=4"

        old_match = _RUNNER._IFT_QUALITY_PATTERN.findall(old)
        self.assertEqual(len(old_match), 1)
        self.assertEqual(old_match[0][0], "3.456e-04")
        self.assertEqual(old_match[0][3], "")  # drho_norm absent
        self.assertEqual(old_match[0][4], "")  # h_pen_logdet absent
        self.assertEqual(old_match[0][5], "4")

        new_match = _RUNNER._IFT_QUALITY_PATTERN.findall(new)
        self.assertEqual(len(new_match), 1)
        self.assertEqual(new_match[0][3], "5.678e-01")
        self.assertEqual(new_match[0][4], "2.345e+01")

        nan_match = _RUNNER._IFT_QUALITY_PATTERN.findall(nan)
        self.assertEqual(len(nan_match), 1)
        # NaN tokens should parse but be filtered downstream by the
        # `float(x) == float(x)` self-equality check.
        self.assertEqual(nan_match[0][3], "NaN")

    def test_ift_rejected_and_noop_patterns_capture_reason(self) -> None:
        # Each rejection reason from runtime.rs should round-trip via
        # the runner's reason-name capture. Spot-check the canonical
        # ones; the exhaustive enumeration is in the commit messages
        # for fec27c97 (initial set) and the dim-mismatch additions
        # in the current commit.
        reasons = [
            ("[IFT-REJECTED] reason=large_drho max_drho=3.456e+00 cap=2.000e+00 drho_dim=4", "large_drho"),
            ("[IFT-REJECTED] reason=hessian_factorize_failed drho_dim=4", "hessian_factorize_failed"),
            ("[IFT-REJECTED] reason=non_finite_solution max_drho=1.234e+00 drho_dim=4", "non_finite_solution"),
            ("[IFT-REJECTED] reason=qs_dim_mismatch qs_dim=10x10 expected_p=8", "qs_dim_mismatch"),
            # New dim-mismatch reasons.
            ("[IFT-REJECTED] reason=rho_dim_mismatch new_rho_dim=2 cache_rho_dim=1", "rho_dim_mismatch"),
            ("[IFT-REJECTED] reason=penalty_dim_mismatch penalties_dim=0 cache_rho_dim=1", "penalty_dim_mismatch"),
            ("[IFT-REJECTED] reason=beta_dim_mismatch cache_beta_dim=3 expected_p=4", "beta_dim_mismatch"),
        ]
        for line, expected in reasons:
            matches = _RUNNER._IFT_REJECTED_PATTERN.findall(line)
            self.assertEqual(matches, [expected], f"failed to extract reason from: {line!r}")

        noop_line = "[IFT-NOOP] reason=all_drho_below_eps max_drho=5.000e-15 drho_dim=4"
        self.assertEqual(
            _RUNNER._IFT_NOOP_PATTERN.findall(noop_line),
            ["all_drho_below_eps"],
        )

    def test_pirls_solve_end_pattern_captures_iters_elapsed_rate(self) -> None:
        sample = (
            "2026-05-05T03:14:15Z INFO  gam::solver::pirls: "
            "[PIRLS solve-end] iters=12 elapsed=0.0345s g_norm_initial=1.234e+01 "
            "g_norm_final=4.567e-08 convergence_rate=2.345e-01 status=Converged"
        )
        matches = _RUNNER._PIRLS_SOLVE_END_PATTERN.findall(sample)
        self.assertEqual(len(matches), 1)
        iters, elapsed, rate = matches[0]
        self.assertEqual(iters, "12")
        self.assertEqual(elapsed, "0.0345")
        self.assertEqual(rate, "2.345e-01")
        # NaN convergence rate (single-iter solves produce NaN: 1**(1/1)
        # is fine but degenerate cases could yield NaN). Pattern must
        # accept the token; the runner filters via `r == r`.
        nan_sample = (
            "[PIRLS solve-end] iters=1 elapsed=0.0001s g_norm_initial=NaN "
            "g_norm_final=NaN convergence_rate=NaN status=Converged"
        )
        nan_matches = _RUNNER._PIRLS_SOLVE_END_PATTERN.findall(nan_sample)
        self.assertEqual(len(nan_matches), 1)
        self.assertEqual(nan_matches[0][2], "NaN")

    def test_kappa_phase_patterns_parse_per_call_and_summary(self) -> None:
        per_call_lines = [
            "[KAPPA-PHASE] phase=cost call=12 theta_norm=3.4500e+00 log_kappa_norm=1.2000e+00 elapsed_s=0.4321",
            "[KAPPA-PHASE] phase=eval_outer call=5 order=ValueGradientHessian theta_norm=3.4500e+00 log_kappa_norm=1.2000e+00 elapsed_s=8.7654",
            "[KAPPA-PHASE] phase=efs call=2 theta_norm=3.4500e+00 log_kappa_norm=1.2000e+00 elapsed_s=2.1098",
        ]
        all_matches = []
        for line in per_call_lines:
            all_matches.extend(_RUNNER._KAPPA_PHASE_PATTERN.findall(line))
        # Three rows: (phase, call, elapsed). The `eval_outer` row has
        # an order=... field between call and theta_norm; the regex
        # makes it optional.
        self.assertEqual(len(all_matches), 3)
        self.assertEqual([m[0] for m in all_matches], ["cost", "eval_outer", "efs"])
        self.assertEqual([m[1] for m in all_matches], ["12", "5", "2"])
        self.assertEqual(
            [m[2] for m in all_matches],
            ["0.4321", "8.7654", "2.1098"],
        )

        summary = (
            "[KAPPA-PHASE-SUMMARY] log_kappa_dim=2 n_cost=12 cost_total_s=5.1840 "
            "n_eval=5 eval_total_s=43.8270 n_efs=2 efs_total_s=4.2196 optim_total_s=53.2306"
        )
        sm = _RUNNER._KAPPA_PHASE_SUMMARY_PATTERN.findall(summary)
        self.assertEqual(len(sm), 1)
        log_kappa_dim, n_cost, cost_s, n_eval, eval_s, n_efs, efs_s, optim_s = sm[0]
        self.assertEqual(log_kappa_dim, "2")
        self.assertEqual(n_cost, "12")
        self.assertAlmostEqual(float(cost_s), 5.1840)
        self.assertAlmostEqual(float(optim_s), 53.2306)


class PhaseSummaryAggregationTests(unittest.TestCase):
    """End-to-end test of `_emit_phase_summary`'s aggregation logic.
    The function is print-side-effect-only, so we capture stderr and
    assert against the emitted summary string."""

    def _run_summary(self, captured_stderr: str) -> str:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            _RUNNER._emit_phase_summary(captured_stderr, "cmd-preview", timed_out=False, rc=0)
        return buf.getvalue()

    def test_warm_start_health_verdict_classifies_tiers_correctly(self) -> None:
        # HEALTHY: coverage ≥ 0.70 AND p50_resid < 0.05
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=8, n_rejects=1, n_noops=1,
            residuals=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2],
        )
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("coverage=0.80", d)
        # MARGINAL: coverage 0.50, p50_resid moderately bad → still MARGINAL
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=4, n_rejects=2, n_noops=2,
            residuals=[0.05, 0.10, 0.15, 0.20],
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")
        # DEGRADED: low coverage AND high residual → DEGRADED
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=1, n_rejects=8, n_noops=1,
            residuals=[0.6],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        # NO-DATA: rejects-only run (predictor never accepted)
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=5, n_noops=0,
            residuals=[],
        )
        self.assertEqual(v, "NO-DATA", f"detail={d}")
        # Edge: HEALTHY threshold boundary — coverage exactly 0.70, p50 = 0.04 → HEALTHY
        v, _ = _RUNNER._warm_start_health_verdict(
            n_accepts=7, n_rejects=2, n_noops=1,
            residuals=[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        )
        self.assertEqual(v, "HEALTHY")
        # Edge: just-below HEALTHY → MARGINAL (coverage <0.70 OR p50≥0.05)
        v, _ = _RUNNER._warm_start_health_verdict(
            n_accepts=7, n_rejects=2, n_noops=1,
            residuals=[0.04, 0.04, 0.04, 0.04, 0.06, 0.06, 0.06],
        )
        # p50 of 7 sorted values is index 3 = 0.04 → still HEALTHY actually.
        # Use a clearer marginal case: coverage=0.6 → drops out of HEALTHY,
        # falls into MARGINAL because coverage ≥ 0.30 even though p50 ≥ 0.05.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=6, n_rejects=2, n_noops=2,
            residuals=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")

    def test_phase_summary_aggregates_tangent_line_predicts_and_rejects(self) -> None:
        # Pin down that [TANGENT-PREDICT] / [TANGENT-REJECTED] markers
        # roll up into the [PHASE summary] line and surface the
        # rejection-reason histogram. A "both predictors fell through
        # to flat" pattern (high IFT-reject + high tangent-reject)
        # should be visible at a glance.
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=1.234e+00 cap=1.500e+00 drho_step_norm_sq=2.345e-02 drho_prev_norm_sq=4.567e-02",
            "[TANGENT-PREDICT] alpha=8.765e-01 cap=1.500e+00 drho_step_norm_sq=2.345e-02 drho_prev_norm_sq=4.567e-02",
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=2.345e+00 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=alpha_negative alpha=-1.234e-01 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=3.000e+00 cap=1.500e+00",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=2", out)
        self.assertIn("tangent_rejects=3", out)
        # Reasons sorted alphabetically by name.
        self.assertIn("tangent_reasons=[alpha_above_cap=2,alpha_negative=1]", out)

    def test_phase_summary_aggregates_ift_accept_reject_noop_independently(self) -> None:
        stderr = "\n".join([
            # 4 accepts at varying residuals
            "[IFT-QUALITY] residual=1.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=2.000e-03 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=4",
            "[IFT-QUALITY] residual=5.000e-02 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=5",
            "[IFT-QUALITY] residual=8.000e-01 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=6",
            # 1 reject (large_drho — typical biobank case)
            "[IFT-REJECTED] reason=large_drho max_drho=3.000e+00 cap=2.000e+00 drho_dim=4",
            # 2 noops
            "[IFT-NOOP] reason=all_drho_below_eps max_drho=5.000e-15 drho_dim=4",
            "[IFT-NOOP] reason=all_drho_below_eps max_drho=4.000e-15 drho_dim=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        # Core aggregation fields surface.
        self.assertIn("ift_predicts=4", out)
        self.assertIn("ift_rejects=1", out)
        self.assertIn("ift_noops=2", out)
        self.assertIn("ift_reasons=[large_drho=1]", out)
        # The accept rate denominator is accepts + rejects + noops.
        # 4 / (4 + 1 + 2) = 0.571... → printed as 0.57.
        self.assertIn("ift_accept_rate=0.57", out)


if __name__ == "__main__":
    unittest.main()
