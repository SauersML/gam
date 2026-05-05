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


def _has_lifelines() -> bool:
    """Detect whether the optional `lifelines` dependency is installed.

    The runner's `_survival_null_curve` requires lifelines for
    Kaplan-Meier estimation (used by survival-scoring tests). Local
    dev environments often don't have it — gating the affected tests
    on this check keeps the suite green where lifelines isn't present
    while still running the test in CI where it IS installed.
    """
    return importlib.util.find_spec("lifelines") is not None


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

    @unittest.skipUnless(
        _has_lifelines(),
        "lifelines optional dep not installed; survival-scoring "
        "test path requires it for Kaplan-Meier null-curve construction",
    )
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
        iters, elapsed, rate, status = matches[0]
        self.assertEqual(iters, "12")
        self.assertEqual(elapsed, "0.0345")
        self.assertEqual(rate, "2.345e-01")
        self.assertEqual(status, "Converged")
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
        self.assertEqual(nan_matches[0][3], "Converged")
        # All five PirlsStatus enum variants must be captured by the
        # status group so the runner's per-status aggregation can
        # distinguish them. Locks the contract that adding a new
        # variant in the rust enum requires updating this regex too —
        # without that update, the new variant would parse as `None`
        # and silently disappear from the verdict.
        for status_variant in [
            "Converged",
            "MaxIterationsReached",
            "StalledAtValidMinimum",
            "LmStepSearchExhausted",
            "Unstable",
        ]:
            line = (
                f"[PIRLS solve-end] iters=5 elapsed=0.001s "
                f"g_norm_initial=1e-3 g_norm_final=1e-9 "
                f"convergence_rate=0.3 status={status_variant}"
            )
            matched = _RUNNER._PIRLS_SOLVE_END_PATTERN.findall(line)
            self.assertEqual(
                len(matched),
                1,
                f"PirlsStatus variant {status_variant!r} did not parse",
            )
            self.assertEqual(matched[0][3], status_variant)

    def test_bfgs_summary_pattern_covers_all_outcome_variants(self) -> None:
        # Locks the regex contract that all four BFGS outcomes parse
        # correctly: converged + max_iter + line-search failed (with
        # iter counts since commit b4e8b436) and generic failed
        # (no iter count). Adding a new BFGS outcome variant in the
        # opt crate / outer_strategy.rs without updating this regex
        # would silently disappear it from the verdict's status mix.
        cases = [
            (
                "[OUTER summary] BFGS converged in 12 iters elapsed=145.234s "
                "final_value=1.23e3",
                "converged", "12", "145.234",
            ),
            (
                "[OUTER summary] BFGS hit max_iter in 100 iters elapsed=2398.0s "
                "final_value=1.23e3",
                "hit max_iter", "100", "2398.0",
            ),
            (
                "[OUTER summary] BFGS line-search failed in 47 iters "
                "elapsed=87.654s final_value=1.23e3",
                "line-search failed", "47", "87.654",
            ),
            (
                "[OUTER summary] BFGS failed elapsed=12.0s err=SomeErr",
                "failed", None, "12.0",
            ),
        ]
        for line, expected_status, expected_iters, expected_elapsed in cases:
            matches = _RUNNER._BFGS_SUMMARY_PATTERN.findall(line)
            self.assertEqual(
                len(matches),
                1,
                f"BFGS outcome {expected_status!r} did not parse: {line}",
            )
            status, iters, elapsed = matches[0]
            self.assertEqual(status, expected_status)
            # The optional `(?:\s+in\s+(\d+)\s+iters)?` group yields
            # "" (empty string) on regex non-match, NOT None. Guard
            # against both for the no-iters case so the test stays
            # robust to regex-engine behavior across CPython versions.
            if expected_iters is None:
                self.assertIn(iters, ("", None))
            else:
                self.assertEqual(iters, expected_iters)
            self.assertEqual(elapsed, expected_elapsed)
        # Backward-compat: older logs (pre-b4e8b436) emitted
        # max_iter / line-search failed WITHOUT the `in N iters`
        # field. Regex's optional group must still match these.
        old_max_iter = (
            "[OUTER summary] BFGS hit max_iter elapsed=2398.0s final_value=1.23e3"
        )
        m = _RUNNER._BFGS_SUMMARY_PATTERN.findall(old_max_iter)
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0][0], "hit max_iter")
        self.assertIn(m[0][1], ("", None))

    def test_outer_hessian_route_pattern_covers_all_reasons(self) -> None:
        # The routing decision is the entry point for outer Hessian
        # cost analysis. Lock the contract that all 6 reasons + the 2
        # choice values + the `irrelevant` token (family_op branch)
        # parse correctly. A future commit that adds a new reason
        # without updating this regex would silently disappear it
        # from the runner's `outer_h_dom_reason` aggregation.
        cases = [
            # kernel-based routing — `scale_prefers_operator=true|false`
            (
                "[OUTER hessian-route] choice=operator reason=large_k "
                "n=320000 p=128 k=32 callback_kernel=false subspace_trace=false "
                "scale_prefers_operator=true",
                "operator", "large_k",
            ),
            (
                "[OUTER hessian-route] choice=operator reason=large_p "
                "n=20000 p=512 k=8 callback_kernel=false subspace_trace=false "
                "scale_prefers_operator=true",
                "operator", "large_p",
            ),
            (
                "[OUTER hessian-route] choice=dense reason=below_crossover "
                "n=1000 p=20 k=4 callback_kernel=false subspace_trace=false "
                "scale_prefers_operator=false",
                "dense", "below_crossover",
            ),
            (
                "[OUTER hessian-route] choice=dense reason=subspace_forced_dense "
                "n=320000 p=101 k=32 callback_kernel=false subspace_trace=true "
                "scale_prefers_operator=true",
                "dense", "subspace_forced_dense",
            ),
            (
                "[OUTER hessian-route] choice=dense reason=kernel_absent "
                "n=1000 p=10 k=2 callback_kernel=false subspace_trace=false "
                "scale_prefers_operator=false",
                "dense", "kernel_absent",
            ),
            # family-op branch — `scale_prefers_operator=irrelevant`
            (
                "[OUTER hessian-route] choice=operator reason=family_op "
                "n=320000 p=128 k=23 callback_kernel=false subspace_trace=false "
                "scale_prefers_operator=irrelevant",
                "operator", "family_op",
            ),
        ]
        for line, expected_choice, expected_reason in cases:
            matches = _RUNNER._OUTER_HESSIAN_ROUTE_PATTERN.findall(line)
            self.assertEqual(
                len(matches),
                1,
                f"reason {expected_reason!r} did not parse: {line}",
            )
            self.assertEqual(matches[0][0], expected_choice)
            self.assertEqual(matches[0][1], expected_reason)

    def test_outer_hessian_elapsed_pattern_extracts_timing(self) -> None:
        # The elapsed marker is paired with the route marker; together
        # they let the runner build `outer_h_total` and
        # `outer_h_subspace_total`. Lock both kernel-based and family-op
        # variants.
        cases = [
            (
                "[OUTER hessian-elapsed] choice=dense reason=subspace_forced_dense "
                "n=320000 p=101 k=32 elapsed=12.347s",
                "dense", "subspace_forced_dense", "12.347",
            ),
            (
                "[OUTER hessian-elapsed] choice=operator reason=family_op "
                "n=320000 p=128 k=23 elapsed=0.123s",
                "operator", "family_op", "0.123",
            ),
        ]
        for line, expected_choice, expected_reason, expected_elapsed in cases:
            matches = _RUNNER._OUTER_HESSIAN_ELAPSED_PATTERN.findall(line)
            self.assertEqual(len(matches), 1)
            choice, reason, _n, _p, _k, elapsed = matches[0]
            self.assertEqual(choice, expected_choice)
            self.assertEqual(reason, expected_reason)
            self.assertEqual(elapsed, expected_elapsed)

    def test_pirls_curvature_kind_pattern_captures_observed_and_fisher(self) -> None:
        # The curvature-kind log emits the ACTUAL curvature used by the
        # inner PIRLS step (after any Fisher fallback). The two enum
        # variants the rust solver emits are `Observed` and `Fisher`.
        # Lock both so a future commit that adds a third
        # `HessianCurvatureKind` variant (or renames one) without
        # updating the regex would silently disappear it from the
        # runner's `pirls_fisher_frac` diagnostic.
        cases = [
            (
                "[STAGE] PIRLS update_with_curvature iter=1 "
                "curvature=Observed elapsed=0.045s",
                "Observed",
            ),
            (
                "[STAGE] PIRLS update_with_curvature iter=2 "
                "curvature=Fisher elapsed=0.037s",
                "Fisher",
            ),
            # Edge: large iter index from the rust formatter.
            (
                "[STAGE] PIRLS update_with_curvature iter=200 "
                "curvature=Fisher elapsed=12.345s",
                "Fisher",
            ),
        ]
        for line, expected_kind in cases:
            matches = _RUNNER._PIRLS_CURVATURE_KIND_PATTERN.findall(line)
            self.assertEqual(
                len(matches),
                1,
                f"curvature kind {expected_kind!r} did not parse: {line}",
            )
            self.assertEqual(matches[0], expected_kind)

    def test_pirls_mid_iter_fisher_pattern_captures_both_reasons(self) -> None:
        # Mid-LM-loop Fisher fallback markers (commit 8ffa7225) fire
        # at TWO sites in the LM loop: one for the gain-rejection
        # branch and one for the candidate-eval-Err branch. Lock both
        # reason variants so adding a third LM-loop fallback site
        # without updating the regex (or this test) would silently
        # disappear it from the runner's
        # `pirls_mid_iter_gain_rejection` / `pirls_mid_iter_candidate_err`
        # diagnostics.
        cases = [
            (
                "[PIRLS] mid-iter Fisher fallback iter=3 reason=gain_rejection",
                "3", "gain_rejection",
            ),
            (
                "[PIRLS] mid-iter Fisher fallback iter=12 reason=candidate_err",
                "12", "candidate_err",
            ),
            # Edge: large iter (>100, the typical PIRLS max_iter cap).
            (
                "[PIRLS] mid-iter Fisher fallback iter=200 reason=candidate_err",
                "200", "candidate_err",
            ),
        ]
        for line, expected_iter, expected_reason in cases:
            matches = _RUNNER._PIRLS_MID_ITER_FISHER_PATTERN.findall(line)
            self.assertEqual(
                len(matches),
                1,
                f"reason {expected_reason!r} did not parse: {line}",
            )
            self.assertEqual(matches[0][0], expected_iter)
            self.assertEqual(matches[0][1], expected_reason)

    def test_pirls_force_fisher_pattern_captures_all_three_reasons(self) -> None:
        # `force_fisher_for_rest engaged` markers (commit dea37b05) fire
        # at AT MOST ONCE per PIRLS solve from THREE distinct sites
        # depending on which branch's increment crossed the
        # `consecutive_fisher_fallbacks > 2` threshold. Lock all three
        # reason variants:
        #   iter_start     — the original Fisher-fallback path
        #                    (Observed assembly itself failed)
        #   gain_rejection — mid-LM-loop accept-failed Fisher retry
        #   candidate_err  — mid-LM-loop candidate-eval-Err Fisher retry
        cases = [
            (
                "[PIRLS] force_fisher_for_rest engaged at iter=5 "
                "(consecutive_fisher_fallbacks=3) reason=iter_start",
                "5", "3", "iter_start",
            ),
            (
                "[PIRLS] force_fisher_for_rest engaged at iter=12 "
                "(consecutive_fisher_fallbacks=4) reason=gain_rejection",
                "12", "4", "gain_rejection",
            ),
            (
                "[PIRLS] force_fisher_for_rest engaged at iter=2 "
                "(consecutive_fisher_fallbacks=3) reason=candidate_err",
                "2", "3", "candidate_err",
            ),
        ]
        for line, expected_iter, expected_count, expected_reason in cases:
            matches = _RUNNER._PIRLS_FORCE_FISHER_PATTERN.findall(line)
            self.assertEqual(
                len(matches),
                1,
                f"reason {expected_reason!r} did not parse: {line}",
            )
            iter_str, count, reason = matches[0]
            self.assertEqual(iter_str, expected_iter)
            self.assertEqual(count, expected_count)
            self.assertEqual(reason, expected_reason)

    def test_pirls_iter_breakdown_pattern_extracts_all_seven_subphases(self) -> None:
        # The breakdown pattern captures iter + attempts + 5 wall-clock
        # sub-phases (curvature, solve, predred, candidate, other).
        # Lock the layout: a future commit that reorders or renames the
        # subphases — or adds a new one without updating the regex —
        # would silently break the runner's `pirls_dom` aggregation,
        # which is the headline diagnostic for "where is inner-Newton
        # spending wall-clock at biobank scale?".
        line = (
            "[PIRLS iter-breakdown] iter=  3 attempts=2 curvature=0.012s "
            "solve=0.003s predred=0.000s candidate=0.045s other=0.001s"
        )
        matches = _RUNNER._PIRLS_ITER_BREAKDOWN_PATTERN.findall(line)
        self.assertEqual(len(matches), 1)
        # Tuple layout: (iter, attempts, curvature, solve, predred,
        # candidate, other). Verify each field comes through correctly.
        iter_str, attempts, curv, solve, predred, candidate, other = matches[0]
        self.assertEqual(iter_str, "3")
        self.assertEqual(attempts, "2")
        self.assertEqual(curv, "0.012")
        self.assertEqual(solve, "0.003")
        self.assertEqual(predred, "0.000")
        self.assertEqual(candidate, "0.045")
        self.assertEqual(other, "0.001")
        # Edge case: zero-padded iter numbers (the rust formatter uses
        # `iter={:>3}` for alignment), high attempts count, and large
        # subphase values — make sure the regex copes with all.
        wide_line = (
            "[PIRLS iter-breakdown] iter=200 attempts=64 curvature=12.345s "
            "solve=678.901s predred=1234.567s candidate=89.012s other=0.500s"
        )
        wide_matches = _RUNNER._PIRLS_ITER_BREAKDOWN_PATTERN.findall(wide_line)
        self.assertEqual(len(wide_matches), 1)
        self.assertEqual(wide_matches[0][0], "200")
        self.assertEqual(wide_matches[0][1], "64")
        self.assertEqual(wide_matches[0][3], "678.901")

    def test_pirls_lm_trajectory_pattern_handles_finite_and_nan_rho(self) -> None:
        # The trajectory pattern captures `accept_rho` as a numeric or
        # the literal `NaN` string (rejection-exhausted iters). The
        # runner aggregator filters NaN via `r == r`; the regex must
        # accept BOTH so the iter is captured at all (otherwise
        # rejection-exhausted iters would be invisible to the
        # aggregator's `lm_attempts_max` distribution too).
        finite_line = (
            "[PIRLS lm-trajectory] iter=  3 start_lambda=1.000e-6 "
            "final_lambda=3.333e-7 log10_ratio=-0.477 accept_rho=0.985 attempts=1"
        )
        nan_line = (
            "[PIRLS lm-trajectory] iter= 10 start_lambda=1.000e-3 "
            "final_lambda=2.000e-3 log10_ratio=0.301 accept_rho=NaN attempts=8"
        )
        finite_matches = _RUNNER._PIRLS_LM_TRAJECTORY_PATTERN.findall(finite_line)
        self.assertEqual(len(finite_matches), 1)
        # Tuple layout: (iter, start, final, ratio, rho, attempts).
        _iter, _start, _final, ratio, rho, attempts = finite_matches[0]
        self.assertEqual(ratio, "-0.477")
        self.assertEqual(rho, "0.985")
        self.assertEqual(attempts, "1")
        nan_matches = _RUNNER._PIRLS_LM_TRAJECTORY_PATTERN.findall(nan_line)
        self.assertEqual(len(nan_matches), 1)
        self.assertEqual(nan_matches[0][4], "NaN")
        self.assertEqual(nan_matches[0][5], "8")

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

    def test_warm_start_health_verdict_detail_includes_tangent_stats(self) -> None:
        # Tangent-line accepts surface in the verdict's detail string
        # so reviewers see both predictor distributions in one glance.
        # The verdict tier itself is unchanged (IFT-driven), but
        # n_tangent_accepts and tangent_p50 should appear when
        # tangent-line fired at all.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=8, n_rejects=2, n_noops=0,
            residuals=[1e-3] * 8,
            n_tangent_accepts=2,
            tangent_p50=0.025,
        )
        # IFT signals dominate the tier: 8 accepts, p50 < 0.05 → HEALTHY.
        self.assertEqual(v, "HEALTHY")
        self.assertIn("n_tangent_accepts=2", d)
        self.assertIn("tangent_p50=2.50e-02", d)
        # Tangent count of 0 suppresses the tangent fields entirely
        # to keep the common-case detail string clean.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=8, n_rejects=0, n_noops=2,
            residuals=[1e-3] * 8,
            n_tangent_accepts=0,
            tangent_p50=None,
        )
        self.assertNotIn("tangent_", d)
        # When tangent fires but no finite p50 is available (e.g.
        # all NaN residuals), the count appears without a p50 field.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=4, n_rejects=2, n_noops=0,
            residuals=[1e-3] * 4,
            n_tangent_accepts=2,
            tangent_p50=None,
        )
        self.assertIn("n_tangent_accepts=2", d)
        self.assertNotIn("tangent_p50=", d)

    def test_combine_fit_verdicts_worst_wins(self) -> None:
        # DEGRADED > MARGINAL > HEALTHY > NO-DATA total ordering.
        # The combined verdict reflects the WORST tier across the
        # axes — a fit that's HEALTHY on one axis but DEGRADED
        # on another is overall DEGRADED.
        combine = _RUNNER._combine_fit_verdicts
        # Both HEALTHY → HEALTHY.
        self.assertEqual(combine("HEALTHY", "HEALTHY"), "HEALTHY")
        # One MARGINAL trumps the other HEALTHY.
        self.assertEqual(combine("HEALTHY", "MARGINAL"), "MARGINAL")
        self.assertEqual(combine("MARGINAL", "HEALTHY"), "MARGINAL")
        # One DEGRADED trumps everything else.
        self.assertEqual(combine("HEALTHY", "DEGRADED"), "DEGRADED")
        self.assertEqual(combine("DEGRADED", "HEALTHY"), "DEGRADED")
        self.assertEqual(combine("MARGINAL", "DEGRADED"), "DEGRADED")
        self.assertEqual(combine("DEGRADED", "DEGRADED"), "DEGRADED")
        # NO-DATA is the bottom of the order — any other tier wins.
        self.assertEqual(combine("HEALTHY", "NO-DATA"), "HEALTHY")
        self.assertEqual(combine("NO-DATA", "MARGINAL"), "MARGINAL")
        self.assertEqual(combine("NO-DATA", "DEGRADED"), "DEGRADED")
        # None is treated as NO-DATA (sub-verdict not emitted because
        # its source markers were absent).
        self.assertEqual(combine(None, "HEALTHY"), "HEALTHY")
        self.assertEqual(combine("MARGINAL", None), "MARGINAL")
        self.assertEqual(combine(None, None), "NO-DATA")
        # Third axis (curvature) is the new optional arg. Default None
        # is backward-compatible: existing callers that pass only 2
        # args get the same behavior as before. New callers that pass
        # 3 args get worst-of-three.
        self.assertEqual(combine("HEALTHY", "HEALTHY", "HEALTHY"), "HEALTHY")
        self.assertEqual(combine("HEALTHY", "HEALTHY", "DEGRADED"), "DEGRADED")
        self.assertEqual(combine("HEALTHY", "HEALTHY", "MARGINAL"), "MARGINAL")
        self.assertEqual(combine("DEGRADED", "HEALTHY", "MARGINAL"), "DEGRADED")
        self.assertEqual(combine("HEALTHY", "MARGINAL", "DEGRADED"), "DEGRADED")
        self.assertEqual(combine("HEALTHY", None, "MARGINAL"), "MARGINAL")
        self.assertEqual(combine(None, None, "DEGRADED"), "DEGRADED")
        self.assertEqual(combine(None, None, None), "NO-DATA")

    def test_dominant_axis_for_verdict_resolves_correctly(self) -> None:
        # Locks the contract for `_dominant_axis_for_verdict`: returns
        # the axis name (warm_start / pirls / curvature / none) that
        # drove the combined verdict via worst-of-three. Tie-breaking
        # at the same tier prefers pirls > warm_start > curvature.
        dom = _RUNNER._dominant_axis_for_verdict
        # Single-axis DEGRADED → that axis is dominant.
        self.assertEqual(
            dom("DEGRADED", warm_start="HEALTHY", pirls="HEALTHY", curvature="DEGRADED"),
            "curvature",
        )
        self.assertEqual(
            dom("DEGRADED", warm_start="DEGRADED", pirls="HEALTHY", curvature="HEALTHY"),
            "warm_start",
        )
        self.assertEqual(
            dom("DEGRADED", warm_start="HEALTHY", pirls="DEGRADED", curvature="HEALTHY"),
            "pirls",
        )
        # Tie-break: all three at DEGRADED → pirls wins.
        self.assertEqual(
            dom("DEGRADED", warm_start="DEGRADED", pirls="DEGRADED", curvature="DEGRADED"),
            "pirls",
        )
        # Tie at DEGRADED between warm_start and curvature → warm_start
        # wins (preference order pirls > warm_start > curvature).
        self.assertEqual(
            dom("DEGRADED", warm_start="DEGRADED", pirls="HEALTHY", curvature="DEGRADED"),
            "warm_start",
        )
        # MARGINAL combined: curvature is the only MARGINAL, so it's
        # dominant.
        self.assertEqual(
            dom("MARGINAL", warm_start="HEALTHY", pirls="HEALTHY", curvature="MARGINAL"),
            "curvature",
        )
        # All HEALTHY → pirls wins the tie.
        self.assertEqual(
            dom("HEALTHY", warm_start="HEALTHY", pirls="HEALTHY", curvature="HEALTHY"),
            "pirls",
        )
        # NO-DATA combined → none.
        self.assertEqual(
            dom("NO-DATA", warm_start=None, pirls=None, curvature=None),
            "none",
        )
        # None inputs are treated as NO-DATA for ranking. With combined
        # MARGINAL and only pirls at MARGINAL, the result is pirls.
        self.assertEqual(
            dom("MARGINAL", warm_start=None, pirls="MARGINAL", curvature=None),
            "pirls",
        )

    def test_curvature_health_verdict_classifies_tiers(self) -> None:
        # Tier policy from `_curvature_health_verdict`:
        #   HEALTHY    fisher_frac < 0.05 AND force_fisher_n == 0
        #   MARGINAL   0.05 ≤ fisher_frac < 0.20 AND force_fisher_n == 0
        #   DEGRADED   fisher_frac ≥ 0.20 OR force_fisher_n > 0
        #   NO-DATA    fisher_frac is None
        verdict = _RUNNER._curvature_health_verdict
        # HEALTHY tier
        self.assertEqual(
            verdict(fisher_frac=0.0, force_fisher_n=0)[0], "HEALTHY"
        )
        self.assertEqual(
            verdict(fisher_frac=0.04, force_fisher_n=0)[0], "HEALTHY"
        )
        # MARGINAL tier
        self.assertEqual(
            verdict(fisher_frac=0.05, force_fisher_n=0)[0], "MARGINAL"
        )
        self.assertEqual(
            verdict(fisher_frac=0.19, force_fisher_n=0)[0], "MARGINAL"
        )
        # DEGRADED tier — high fisher_frac
        self.assertEqual(
            verdict(fisher_frac=0.20, force_fisher_n=0)[0], "DEGRADED"
        )
        self.assertEqual(
            verdict(fisher_frac=0.50, force_fisher_n=0)[0], "DEGRADED"
        )
        # DEGRADED tier — any force_fisher engagement (even with low frac)
        self.assertEqual(
            verdict(fisher_frac=0.0, force_fisher_n=1)[0], "DEGRADED"
        )
        self.assertEqual(
            verdict(fisher_frac=0.04, force_fisher_n=1)[0], "DEGRADED"
        )
        # NO-DATA — fisher_frac=None
        self.assertEqual(
            verdict(fisher_frac=None, force_fisher_n=0)[0], "NO-DATA"
        )
        # Detail string carries both signals at the right precision.
        v, d = verdict(fisher_frac=0.123, force_fisher_n=2)
        self.assertEqual(v, "DEGRADED")
        self.assertIn("fisher_frac=0.12", d)
        self.assertIn("force_fisher_n=2", d)

    def test_phase_summary_emits_fit_health_combining_warm_start_and_pirls(self) -> None:
        # End-to-end: when both [WARM-START health] and [PIRLS health]
        # fire, the [FIT health] line combines them and shows the
        # individual sub-verdicts in its detail.
        stderr = "\n".join([
            # Healthy IFT data
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            # Degraded PIRLS rates (median 0.6 → DEGRADED tier)
            "[PIRLS solve-end] iters=8 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=5.500e-01 status=Converged",
            "[PIRLS solve-end] iters=10 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=6.500e-01 status=Converged",
            "[PIRLS solve-end] iters=12 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=7.000e-01 status=Converged",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        # All three health lines emit
        self.assertIn("[WARM-START health]", out)
        self.assertIn("[PIRLS health]", out)
        self.assertIn("[FIT health]", out)
        # Combined verdict is DEGRADED (PIRLS axis wins) and the
        # detail shows ALL THREE sub-verdicts: warm_start, pirls, and
        # curvature. The stderr above doesn't include any curvature-
        # kind markers ([STAGE] PIRLS update_with_curvature ... never
        # emitted), so curvature should be ABSENT — locking the
        # contract that absent markers map to ABSENT, not NO-DATA, in
        # the visible label.
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=DEGRADED", fit_lines[0])
        self.assertIn("warm_start=", fit_lines[0])
        self.assertIn("pirls=DEGRADED", fit_lines[0])
        self.assertIn("curvature=ABSENT", fit_lines[0])

    def test_phase_summary_tangent_accept_rate_split_matches_ift(self) -> None:
        # End-to-end symmetric to the IFT-rate split: tangent fires
        # only when IFT rejected, so a fit producing tangent markers
        # also has IFT rejects upstream. Exercise the tangent
        # aggregator: 3 tangent_predicts + 1 tangent_reject + 2
        # tangent_noops:
        #   tangent_accept_rate        = 3 / 6 = 0.50
        #   tangent_accept_rate_active = 3 / 4 = 0.75
        stderr_lines = [
            # Upstream IFT context — tangent only fires when IFT rejected.
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            # Tangent: 3 predicts (alpha values within range). The
            # regex requires alpha + cap + drho_step_norm_sq +
            # drho_prev_norm_sq fields (see _TANGENT_PREDICT_PATTERN).
            "[TANGENT-PREDICT] alpha=0.500 cap=1.500 drho_step_norm_sq=1.0e-2 drho_prev_norm_sq=4.0e-2",
            "[TANGENT-PREDICT] alpha=0.700 cap=1.500 drho_step_norm_sq=2.0e-2 drho_prev_norm_sq=4.0e-2",
            "[TANGENT-PREDICT] alpha=1.200 cap=1.500 drho_step_norm_sq=5.0e-2 drho_prev_norm_sq=4.0e-2",
            # Tangent: 1 reject.
            "[TANGENT-REJECTED] reason=alpha_above_cap",
            # Tangent: 2 noops.
            "[TANGENT-NOOP] reason=alpha_below_eps",
            "[TANGENT-NOOP] reason=alpha_below_eps",
            # Tangent quality follows from each successful predict.
            "[TANGENT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[TANGENT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[TANGENT-QUALITY] residual=3.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ]
        stderr = "\n".join(stderr_lines)
        out = self._run_summary(stderr)
        # Find the line with the tangent rate metrics.
        rate_lines = [
            line for line in out.splitlines()
            if "tangent_accept_rate=" in line
        ]
        self.assertEqual(len(rate_lines), 1, f"expected 1 rate line, got {rate_lines}")
        line = rate_lines[0]
        # accept_rate (with noops) = 3 / 6 = 0.50
        self.assertIn("tangent_accept_rate=0.50", line)
        # accept_rate_active (excluding noops) = 3 / 4 = 0.75
        self.assertIn("tangent_accept_rate_active=0.75", line)

    def test_phase_summary_distinguishes_accept_rate_from_active(self) -> None:
        # End-to-end: when the outer optimizer makes many zero-step
        # calls (noops), `ift_accept_rate` (denominator includes
        # noops) DROPS while `ift_accept_rate_active` (denominator
        # excludes noops) stays HIGH. This separates "predictor is
        # bad" from "outer is calling predictor unnecessarily".
        #
        # 4 accepts (IFT-QUALITY) + 1 reject + 5 noops:
        #   accept_rate        = 4 / 10 = 0.40
        #   accept_rate_active = 4 / 5 = 0.80
        # The 2× difference confirms the metrics are independent.
        stderr_lines = [
            # 4 accepts.
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=3.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=4.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            # 1 reject.
            "[IFT-REJECTED] reason=large_drho",
            # 5 noops.
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ]
        stderr = "\n".join(stderr_lines)
        out = self._run_summary(stderr)
        # Find the line with the IFT accept rates.
        lines_with_rate = [
            line for line in out.splitlines()
            if "ift_accept_rate=" in line
        ]
        self.assertEqual(len(lines_with_rate), 1)
        line = lines_with_rate[0]
        # Accept rate (with noops in denominator) = 4 / 10 = 0.40.
        self.assertIn("ift_accept_rate=0.40", line)
        # Active accept rate (noops EXCLUDED) = 4 / 5 = 0.80.
        self.assertIn("ift_accept_rate_active=0.80", line)
        # The order matters too: active comes after standard.
        i_accept = line.index("ift_accept_rate=")
        i_active = line.index("ift_accept_rate_active=")
        self.assertLess(i_accept, i_active)

    def test_phase_summary_curvature_healthy_when_fisher_frac_low(self) -> None:
        # End-to-end: curvature HEALTHY when fisher_frac < 0.05 AND
        # no force_fisher engagement. With all three axes HEALTHY,
        # the [FIT health] verdict should also be HEALTHY.
        # 1 Fisher / 25 Observed → fisher_frac ≈ 0.04 (HEALTHY band).
        stderr_lines = [
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.000e-01 status=Converged",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.300e-01 status=Converged",
        ]
        # 25 Observed + 1 Fisher → fisher_frac = 1/26 ≈ 0.038 < 0.05.
        for i in range(1, 26):
            stderr_lines.append(
                f"[STAGE] PIRLS update_with_curvature iter={i} curvature=Observed elapsed=0.01s"
            )
        stderr_lines.append(
            "[STAGE] PIRLS update_with_curvature iter=26 curvature=Fisher elapsed=0.01s"
        )
        stderr_lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        stderr = "\n".join(stderr_lines)
        out = self._run_summary(stderr)
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=HEALTHY", curv_lines[0])
        # All three axes HEALTHY → combined verdict HEALTHY,
        # dominant_axis=pirls (tie-break order pirls > warm_start >
        # curvature). The dominant axis being pirls in the all-HEALTHY
        # case is a documented contract — not a "winner" since
        # nothing's wrong, just the tie-break preference.
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=HEALTHY", fit_lines[0])
        self.assertIn("dominant_axis=pirls", fit_lines[0])

    def test_phase_summary_curvature_marginal_when_fisher_frac_in_band(self) -> None:
        # End-to-end: curvature verdict at MARGINAL tier (fisher_frac
        # in [0.05, 0.20) AND no force_fisher engagement). Verifies
        # the MIDDLE tier fires correctly when both other axes are
        # HEALTHY, locking the verdict combination semantics for
        # `[FIT health] verdict=MARGINAL`.
        #
        # 1 Fisher / 9 Observed → fisher_frac = 0.10 (in MARGINAL band).
        # No force_fisher_for_rest markers → no engagement.
        stderr_lines = [
            # Healthy IFT (warm_start = HEALTHY).
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            # Healthy PIRLS rates.
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.000e-01 status=Converged",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.300e-01 status=Converged",
        ]
        # 9 Observed iters
        for i in range(1, 10):
            stderr_lines.append(
                f"[STAGE] PIRLS update_with_curvature iter={i} curvature=Observed elapsed=0.01s"
            )
        # 1 Fisher iter → fisher_frac = 1/10 = 0.10
        stderr_lines.append(
            "[STAGE] PIRLS update_with_curvature iter=10 curvature=Fisher elapsed=0.01s"
        )
        stderr_lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        stderr = "\n".join(stderr_lines)
        out = self._run_summary(stderr)
        # Curvature verdict is MARGINAL (fisher_frac=0.10, in band).
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=MARGINAL", curv_lines[0])
        self.assertIn("fisher_frac=0.10", curv_lines[0])
        self.assertIn("force_fisher_n=0", curv_lines[0])
        # FIT verdict picks up MARGINAL via worst-of-three (others HEALTHY).
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=MARGINAL", fit_lines[0])
        self.assertIn("dominant_axis=curvature", fit_lines[0])
        self.assertIn("curvature=MARGINAL", fit_lines[0])

    def test_phase_summary_curvature_degraded_drives_fit_health(self) -> None:
        # End-to-end: when curvature markers fire and indicate
        # DEGRADED reliability (Fisher-fallback engagement), the
        # [CURVATURE health] line emits AND the [FIT health] line
        # picks up DEGRADED via the worst-of-three combination, even
        # when PIRLS and warm-start are both HEALTHY.
        stderr = "\n".join([
            # Healthy IFT (warm_start = HEALTHY).
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            # Healthy PIRLS rates (p95 < 0.5 → HEALTHY).
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.000e-01 status=Converged",
            "[PIRLS solve-end] iters=5 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.500e-01 status=Converged",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.300e-01 status=Converged",
            # Curvature markers: 5 Observed iters + 5 Fisher iters
            # → fisher_frac = 0.5 (≥ 0.20 threshold → DEGRADED).
            "[STAGE] PIRLS update_with_curvature iter=1 curvature=Observed elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=2 curvature=Observed elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=3 curvature=Observed elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=4 curvature=Observed elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=5 curvature=Observed elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=6 curvature=Fisher elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=7 curvature=Fisher elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=8 curvature=Fisher elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=9 curvature=Fisher elapsed=0.01s",
            "[STAGE] PIRLS update_with_curvature iter=10 curvature=Fisher elapsed=0.01s",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        # [CURVATURE health] line emits with DEGRADED.
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=DEGRADED", curv_lines[0])
        self.assertIn("fisher_frac=0.50", curv_lines[0])
        # [FIT health] line picks up DEGRADED via worst-of-three.
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=DEGRADED", fit_lines[0])
        # The OTHER two axes are HEALTHY.
        self.assertIn("warm_start=HEALTHY", fit_lines[0])
        self.assertIn("pirls=HEALTHY", fit_lines[0])
        # The curvature axis is the one driving DEGRADED.
        self.assertIn("curvature=DEGRADED", fit_lines[0])
        # The new `dominant_axis` field surfaces the offending axis
        # name directly, so CI scrapers can alert on the specific
        # failing axis without re-implementing worst-of-three.
        self.assertIn("dominant_axis=curvature", fit_lines[0])

    def test_pirls_health_verdict_classifies_tiers(self) -> None:
        # HEALTHY: 95% of solves at rate < 0.5 (each Newton iter at
        # least halved the gradient on average for the bulk of the
        # distribution). The earlier max < 0.5 rule was too strict;
        # this test exercises the new p95-based gate.
        v, d = _RUNNER._pirls_health_verdict(rates=[0.1, 0.2, 0.3, 0.4, 0.45])
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("max=0.450", d)
        # HEALTHY tolerates a few outliers: 97 clean rates + 3
        # outliers at 0.6. With n=100, p95 is sorted[95] which is
        # still a clean rate (the 3 outliers occupy indices 97-99,
        # outside the top-5% slot). So the verdict stays HEALTHY
        # despite the outliers. Earlier max-based rule would have
        # flipped to MARGINAL on a SINGLE such outlier.
        rates_with_outliers = [0.1] * 25 + [0.2] * 25 + [0.3] * 25 + [0.4] * 22 + [0.6] * 3
        v, d = _RUNNER._pirls_health_verdict(rates=rates_with_outliers)
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("max=0.600", d)
        # MARGINAL: most solves fast (p50 < 0.5) but enough struggling
        # that p95 ≥ 0.5.
        v, d = _RUNNER._pirls_health_verdict(
            rates=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")
        # DEGRADED: p50 ≥ 0.5 (median solve grinding).
        v, d = _RUNNER._pirls_health_verdict(rates=[0.5, 0.6, 0.7, 0.8])
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        # DEGRADED: max ≥ 0.85 even when p50 is fine (one solve
        # essentially failed to converge — saturation regime).
        v, d = _RUNNER._pirls_health_verdict(
            rates=[0.1, 0.2, 0.3, 0.95],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        # NO-DATA: empty rates list (no PIRLS solves emitted markers).
        v, d = _RUNNER._pirls_health_verdict(rates=[])
        self.assertEqual(v, "NO-DATA")
        self.assertIn("n_solves=0", d)

    def test_phase_summary_emits_pirls_health_verdict_alongside_warm_start(self) -> None:
        # End-to-end: stderr containing [PIRLS solve-end] markers
        # produces a [PIRLS health] line in addition to the existing
        # [WARM-START health] line.
        stderr = "\n".join([
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-04 convergence_rate=2.500e-01 status=Converged",
            "[PIRLS solve-end] iters=5 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-05 convergence_rate=1.585e-01 status=Converged",
            "[PIRLS solve-end] iters=3 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=3.0e-02 convergence_rate=4.000e-01 status=Converged",
            # Healthy IFT-QUALITY data drives the warm-start verdict
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("[WARM-START health]", out)
        self.assertIn("[PIRLS health]", out)
        # All three rates < 0.5 → HEALTHY.
        self.assertIn("verdict=HEALTHY", out.splitlines()[-1])

    def test_warm_start_health_verdict_p95_saturation_guard(self) -> None:
        # Even when p50_resid is clean (well below 0.05), a poor
        # p95_resid (≥ 0.20) drops the verdict from HEALTHY to
        # MARGINAL. Same central-tendency-safe rule as the PIRLS
        # verdict's p95 threshold (commit efc54eca): a tail of bad
        # predictions hidden behind a clean median is still a
        # degradation signal.
        #
        # 80 clean residuals (1e-3) + 20 outliers at 0.5: n=100,
        # p95 = sorted[95] = 0.5 (well above 0.20).
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=100, n_rejects=0, n_noops=0,
            residuals=[1e-3] * 80 + [0.5] * 20,
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")
        self.assertIn("p95_resid=5.00e-01", d)
        # Same fit but only 3 outliers (in 100): p95 = sorted[95] is
        # still in the clean range. Verdict goes back to HEALTHY.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=100, n_rejects=0, n_noops=0,
            residuals=[1e-3] * 97 + [0.5] * 3,
        )
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("p50_resid=1.00e-03", d)
        self.assertIn("p95_resid=1.00e-03", d)

    def test_warm_start_health_verdict_outer_nonfinite_overrides_to_degraded(self) -> None:
        # Even with HEALTHY-looking IFT signals, a single
        # [OUTER non-finite] warning during the fit must override the
        # verdict to DEGRADED. Broken geometry invalidates the
        # predictor-faithfulness measurements regardless of how clean
        # the residuals look.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=10, n_rejects=0, n_noops=0,
            residuals=[1e-5] * 10,
            n_outer_nonfinite=1,
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        self.assertIn("n_outer_nonfinite=1", d)
        # Without the override, this would have been HEALTHY.
        v_no_override, _ = _RUNNER._warm_start_health_verdict(
            n_accepts=10, n_rejects=0, n_noops=0,
            residuals=[1e-5] * 10,
            n_outer_nonfinite=0,
        )
        self.assertEqual(v_no_override, "HEALTHY")
        # NO-DATA case under override: no residuals + outer_nonfinite > 0
        # should still return DEGRADED (override fires before NO-DATA).
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=2, n_noops=0,
            residuals=[],
            n_outer_nonfinite=3,
        )
        self.assertEqual(v, "DEGRADED")
        self.assertIn("n_outer_nonfinite=3", d)

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
        # NEW SEMANTICS: rejects-only run (predictor was tried but
        # ALWAYS fell through) is DEGRADED, not NO-DATA. The
        # warm-start machinery is firing but never delivering — a
        # real degradation signal at this surface.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=5, n_noops=0,
            residuals=[],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        # NEW SEMANTICS: noops-only run is also DEGRADED — the
        # predictor was tried but only produced identity returns
        # (Δρ below floor every time), so the warm-start path
        # exercised but contributed nothing.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=0, n_noops=4,
            residuals=[],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        # NO-DATA only when the predictor was never tried at all
        # (n_accepts + n_rejects + n_noops == 0) AND no
        # outer-non-finite signals.
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=0, n_noops=0,
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

    def test_phase_summary_aggregates_ift_iters_distribution(self) -> None:
        # The [IFT-QUALITY] marker carries `iters=K` (PIRLS iters
        # consumed after the warm-start). The runner now aggregates
        # this into ift_iters_p50/p95/max so reviewers can see the
        # combined warm-start value: small residual + small iters
        # = predictor delivering correctness AND speed.
        stderr = "\n".join([
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[IFT-QUALITY] residual=3.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=5",
            "[IFT-QUALITY] residual=4.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=6",
            "[IFT-QUALITY] residual=5.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=12",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        # Sorted iters [3, 4, 5, 6, 12]; p50 (index 2) = 5,
        # p95 (index 4) = 12, max = 12.
        self.assertIn("ift_iters_p50=5", out)
        self.assertIn("ift_iters_p95=12", out)
        self.assertIn("ift_iters_max=12", out)
        # Old-format (no drho/logdet fields) lines should also
        # contribute their iters — this exercises the regex's
        # backwards compat.
        stderr_old = "\n".join([
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=2",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=3.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out_old = self._run_summary(stderr_old)
        self.assertIn("ift_iters_p50=3", out_old)

    def test_phase_summary_aggregates_tangent_noop_marker(self) -> None:
        # The tangent-line predictor's α-below-eps short-circuit
        # emits [TANGENT-NOOP], symmetric with [IFT-NOOP]. The runner
        # surfaces the count separately from predicts/rejects so a
        # reviewer can distinguish:
        #   - tangent-line accepted with real Δβ → tangent_predicts
        #   - tangent-line returned identity (Δρ collapse)  → tangent_noops
        #   - tangent-line fell through to flat (cap/dim)   → tangent_rejects
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=1.000e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.100e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-NOOP] reason=alpha_below_eps alpha=1.000e-15 eps=1.000e-12",
            "[TANGENT-NOOP] reason=alpha_below_eps alpha=5.000e-16 eps=1.000e-12",
            "[TANGENT-NOOP] reason=alpha_below_eps alpha=1.000e-14 eps=1.000e-12",
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=2.500e+00 cap=1.500e+00",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=2", out)
        self.assertIn("tangent_rejects=1", out)
        self.assertIn("tangent_noops=3", out)
        # The reasons aggregation should still include only the
        # rejected reason, not the noop reason (different bucket).
        self.assertIn("tangent_reasons=[alpha_above_cap=1]", out)

    def test_phase_summary_aggregates_tangent_iters_distribution(self) -> None:
        # Parallel to test_phase_summary_aggregates_ift_iters_distribution:
        # the tangent-line path's [TANGENT-QUALITY] markers also carry
        # `iters=K` and the runner now aggregates them into
        # tangent_iters_p50/p95/max so reviewers see whether
        # tangent-line predictions reduce the inner-Newton iter count
        # or PIRLS still has to grind through them.
        stderr = "\n".join([
            "[TANGENT-QUALITY] residual=1.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[TANGENT-QUALITY] residual=2.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=5",
            "[TANGENT-QUALITY] residual=3.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=6",
            "[TANGENT-QUALITY] residual=4.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=8",
            "[TANGENT-QUALITY] residual=5.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=15",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        # Sorted iters [4, 5, 6, 8, 15]; p50 (index 2) = 6,
        # p95 (index 4) = 15, max = 15.
        self.assertIn("tangent_iters_p50=6", out)
        self.assertIn("tangent_iters_p95=15", out)
        self.assertIn("tangent_iters_max=15", out)

    def test_phase_summary_aggregates_tangent_quality_separately_from_ift(self) -> None:
        # Pin down that [TANGENT-QUALITY] residuals roll up into a
        # SEPARATE distribution from [IFT-QUALITY] (commit 99424b47).
        # Mixing them in one percentile would skew the IFT verdict
        # whenever the tangent-line fallback fires with a different
        # residual character.
        stderr = "\n".join([
            # 4 IFT accepts at clean residuals
            "[IFT-QUALITY] residual=1.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=2.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=5.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=1.000e-03 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=4",
            # 2 tangent-line accepts at MUCH worse residuals (would
            # have skewed an unsegregated p50 from ~3e-4 to ~1e-2)
            "[TANGENT-QUALITY] residual=1.500e-02 converged_norm=1.000e+00 predicted_norm=9.985e-01 iters=5",
            "[TANGENT-QUALITY] residual=2.500e-02 converged_norm=1.000e+00 predicted_norm=9.975e-01 iters=6",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        # IFT distribution untouched by tangent-line residuals.
        self.assertIn("ift_predicts=4", out)
        # IFT p50 stays small — clean residuals only.
        self.assertIn("ift_p50=5.00e-04", out)
        # Tangent-line distribution is its own field.
        self.assertIn("tangent_quality_predicts=2", out)
        self.assertIn("tangent_p50=", out)
        # The bad tangent residuals don't bleed into ift_p50.
        self.assertNotIn("ift_p50=1.50e-02", out)
        self.assertNotIn("ift_p50=2.50e-02", out)

    def test_phase_summary_surfaces_tangent_alpha_distribution(self) -> None:
        # The [TANGENT-PREDICT] alpha values now feed a p50/max
        # distribution. Reviewers can see whether the fallback fires
        # at modest extrapolations (α ≈ 1, healthy) or pushes the
        # adaptive cap (α near 1.5+, marginal).
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=8.000e-01 cap=1.500e+00 drho_step_norm_sq=2.345e-02 drho_prev_norm_sq=4.567e-02",
            "[TANGENT-PREDICT] alpha=1.200e+00 cap=1.500e+00 drho_step_norm_sq=3.000e-02 drho_prev_norm_sq=4.000e-02",
            "[TANGENT-PREDICT] alpha=1.450e+00 cap=1.500e+00 drho_step_norm_sq=4.000e-02 drho_prev_norm_sq=3.000e-02",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=3", out)
        # Sorted [0.8, 1.2, 1.45]; p50 (index 1) = 1.20, max = 1.45.
        self.assertIn("tangent_alpha_p50=1.20", out)
        self.assertIn("tangent_alpha_max=1.45", out)

    def test_phase_summary_flags_tangent_marker_drift(self) -> None:
        # Every successful [TANGENT-PREDICT] should pair with a
        # downstream [TANGENT-QUALITY] from the post-PIRLS residual
        # block (commit 99424b47). If counts diverge by >1, the
        # instrumentation chain is silently dropping markers — a
        # regression signal worth surfacing.
        stderr = "\n".join([
            # 5 PREDICTs but only 2 QUALITY → drift signal fires.
            "[TANGENT-PREDICT] alpha=1.000e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.100e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.200e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.300e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.400e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-QUALITY] residual=1.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[TANGENT-QUALITY] residual=2.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_marker_drift=predict=5_vs_quality=2", out)

    def test_phase_summary_tolerates_off_by_one_tangent_marker_drift(self) -> None:
        # Off-by-one between PREDICT and QUALITY counts is normal at
        # command timeout (PIRLS still running, predict was logged
        # but quality block hadn't fired). Don't flag.
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=1.000e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.100e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-QUALITY] residual=1.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertNotIn("tangent_marker_drift", out)

    def test_phase_summary_kappa_complete_surfaces_per_phase_max_and_p95(self) -> None:
        # COMPLETE κ optimization: both [KAPPA-PHASE] per-call markers
        # AND [KAPPA-PHASE-SUMMARY] are present. The summary's totals
        # are authoritative, but the per-call distribution (max/p95)
        # surfaces alongside to disambiguate single-outlier vs uniform-
        # slow workload — same rationale as the INCOMPLETE branch.
        stderr_lines = []
        # 30 fast eval_outer calls + 1 slow one.
        for i in range(30):
            stderr_lines.append(
                f"[KAPPA-PHASE] phase=eval_outer call={i+1} order=ValueGradientHessian "
                f"theta_norm=1.0e+00 log_kappa_norm=1.0e+00 elapsed_s=0.3"
            )
        stderr_lines.append(
            "[KAPPA-PHASE] phase=eval_outer call=31 order=ValueGradientHessian "
            "theta_norm=1.0e+00 log_kappa_norm=1.0e+00 elapsed_s=15.0"
        )
        stderr_lines.append(
            "[KAPPA-PHASE-SUMMARY] log_kappa_dim=2 n_cost=0 cost_total_s=0.0 "
            "n_eval=31 eval_total_s=24.0 n_efs=0 efs_total_s=0.0 optim_total_s=24.0"
        )
        stderr_lines.append("[PHASE] my-fit fit end elapsed=24.0s")
        out = self._run_summary("\n".join(stderr_lines))
        # Both the summary's totals AND the per-call distribution
        # appear.
        self.assertIn("kappa_eval_calls=31", out)
        self.assertIn("kappa_eval_total=24.0s", out)
        self.assertIn("kappa_eval_outer_max=15.00s", out)
        # p95 = sorted[min(30, 29)] = sorted[29] = 0.3 (excludes the
        # single outlier at index 30 = 15.0).
        self.assertIn("kappa_eval_outer_p95=0.30s", out)

    def test_phase_summary_kappa_incomplete_surfaces_per_phase_max_and_p95(self) -> None:
        # κ-optimization interrupted by timeout: per-call [KAPPA-PHASE]
        # lines but no [KAPPA-PHASE-SUMMARY]. The runner now surfaces
        # max/p95 per phase alongside count + total, so a reviewer can
        # tell whether a slow phase had ONE big call or MANY small
        # ones.
        stderr_lines = []
        # 50 fast eval_outer calls (each 0.5s) + 1 slow one (60s)
        # would total 85s with one outlier dominating.
        for i in range(50):
            stderr_lines.append(
                f"[KAPPA-PHASE] phase=eval_outer call={i+1} order=ValueGradientHessian "
                f"theta_norm=1.0e+00 log_kappa_norm=1.0e+00 elapsed_s=0.5"
            )
        stderr_lines.append(
            "[KAPPA-PHASE] phase=eval_outer call=51 order=ValueGradientHessian "
            "theta_norm=1.0e+00 log_kappa_norm=1.0e+00 elapsed_s=60.0"
        )
        # 5 cost calls, all fast.
        for i in range(5):
            stderr_lines.append(
                f"[KAPPA-PHASE] phase=cost call={i+1} theta_norm=1.0e+00 "
                f"log_kappa_norm=1.0e+00 elapsed_s=0.1"
            )
        stderr_lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        out = self._run_summary("\n".join(stderr_lines))
        self.assertIn("kappa_optim_INCOMPLETE", out)
        # eval_outer: 51 calls, total ~85s, max 60s, p95 = sorted[48] = 0.5
        # (since 51-call list sorted has 50 values at 0.5 and 1 at 60;
        # p95 = sorted[min(50, 48)] = sorted[48] = 0.5; the outlier
        # is excluded from p95 but still flagged in max).
        self.assertIn("kappa_eval_outer_calls=51", out)
        self.assertIn("kappa_eval_outer_max=60.00s", out)
        self.assertIn("kappa_eval_outer_p95=0.50s", out)
        # cost: 5 calls, all 0.1s; max = p95 = 0.10.
        self.assertIn("kappa_cost_calls=5", out)
        self.assertIn("kappa_cost_max=0.10s", out)

    def test_phase_summary_aggregates_outer_nonfinite_warnings(self) -> None:
        # `[OUTER non-finite]` is a bug signal — the REML unified
        # evaluator hit a NaN / Inf in an intermediate. Should be 0 in
        # healthy fits; any non-zero count must surface prominently in
        # the summary so a CI reviewer sees it.
        stderr = "\n".join([
            "[OUTER non-finite] rho_a_vals[2] at (2, 2) = NaN",
            "[OUTER non-finite] penalty_a_k_betas[1] has non-finite",
            "[OUTER non-finite] penalty_a_k_betas[3] has non-finite",
            "[OUTER non-finite] leverage h^G has non-finite entries",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        # Total count surfaces.
        self.assertIn("outer_nonfinite=4", out)
        # Per-intermediate breakdown groups by name (sorted).
        self.assertIn("outer_nonfinite_at=[", out)
        self.assertIn("penalty_a_k_betas[1]=1", out)
        self.assertIn("penalty_a_k_betas[3]=1", out)
        # The first-token capture covers the field-name token before
        # the parenthesized indices.

    def test_phase_summary_omits_outer_nonfinite_when_count_is_zero(self) -> None:
        # Healthy fit: no [OUTER non-finite] markers. Aggregation should
        # not emit the field at all (rather than `outer_nonfinite=0`).
        stderr = "[PHASE] my-fit fit end elapsed=10.0s\n"
        out = self._run_summary(stderr)
        self.assertNotIn("outer_nonfinite", out)

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
            # New dim-mismatch / non-finite-α reasons.
            "[TANGENT-REJECTED] reason=rho_dim_mismatch new_rho_dim=4 cur_rho_dim=3 prev_rho_dim=3",
            "[TANGENT-REJECTED] reason=beta_dim_mismatch cur_beta_dim=10 prev_beta_dim=12",
            "[TANGENT-REJECTED] reason=degenerate_drho d_rho_norm_sq=1.000e-30",
            "[TANGENT-REJECTED] reason=nonfinite_alpha step_dot_d=NaN d_rho_norm_sq=2.345e-02",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=2", out)
        self.assertIn("tangent_rejects=7", out)
        # Reasons sorted alphabetically by name.
        self.assertIn(
            "tangent_reasons=[alpha_above_cap=2,alpha_negative=1,beta_dim_mismatch=1,degenerate_drho=1,nonfinite_alpha=1,rho_dim_mismatch=1]",
            out,
        )

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
