import typing
import csv
import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_PATH = _REPO_ROOT / "bench" / "large_scale" / "runner.py"
_SPEC = importlib.util.spec_from_file_location("bench_large_scale_runner", _RUNNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load large-scale benchmark runner from {_RUNNER_PATH}")
_RUNNER: typing.Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _write_csv(path: Path, rows: typing.Sequence[typing.Mapping[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LargeScaleRunnerTests(unittest.TestCase):
    def test_terminal_output_sanitizer_removes_cursor_controls_across_chunks(self) -> None:
        sanitizer = _RUNNER._TerminalOutputSanitizer()
        text = (
            sanitizer.feed("progress\r        [1s] ok \x1b[")
            + sanitizer.feed("2K next\x1b]0;title")
            + sanitizer.feed("\x07 done\n")
        )
        self.assertEqual(text, "progress\n[1s] ok  next done\n")

    def test_default_large_scale_matrix_keeps_both_400k_binomial_marginal_slope_lanes(self) -> None:
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
            2,
            "expected rigid and warped disease + Rust + binomial + marginal-slope lanes in the "
            "default large-scale matrix; found "
            f"{[s.name for s in marginal_slope_disease]}",
        )
        lanes = {lane.name: lane for lane in marginal_slope_disease}
        rigid = lanes["rust_margslope_aniso_duchon16d_rigid"]
        warped = lanes["rust_margslope_aniso_duchon16d_linkwiggle_scorewarp_fast"]
        for lane in (rigid, warped):
            self.assertEqual(lane.spatial_basis, "duchon")
            self.assertEqual(lane.pc_count, 16, f"{lane.name} must run on 16 PCs")
            self.assertTrue(lane.scale_dimensions, f"{lane.name} must enable per-axis scales")
            self.assertEqual(lane.z_column, "pgs_ctn_z", f"{lane.name} must read CTN z column")
            self.assertEqual(lane.centers, 24)
        self.assertIsNone(rigid.mean_linkwiggle_knots)
        self.assertIsNone(rigid.slope_linkwiggle_knots)
        self.assertEqual(warped.mean_linkwiggle_knots, 8)
        self.assertEqual(warped.slope_linkwiggle_knots, 8)

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
            slope_linkwiggle_knots=7,
        )
        mean_formula, slope_formula = _RUNNER.rust_marginal_slope_formula_classification(spec, centers=20)
        self.assertIn("duchon(pc1_std, pc2_std", mean_formula)
        self.assertIn("centers=20", mean_formula)
        self.assertIn("order=0", mean_formula)
        self.assertIn("power=9", mean_formula)
        self.assertIn("length_scale=1", mean_formula)
        self.assertNotIn("pgs_ctn_z", mean_formula)
        self.assertIn("linkwiggle(internal_knots=8)", mean_formula)
        self.assertIn("linkwiggle(internal_knots=7)", slope_formula)

    def test_large_scale_preflight_rejects_unsafe_dense_duchon_width_before_allocation(self) -> None:
        report = _RUNNER.preflight_marginal_slope_large_scale(
            n_train=400000,
            d_pc=16,
            centers=1400,
        )
        self.assertEqual(report.status, "FAIL")
        text = "\n".join(report.lines)
        self.assertIn("anisotropic derivative dense estimate", text)
        self.assertIn("status: FAIL", text)

    def test_large_scale_preflight_accepts_production_marginal_slope_width(self) -> None:
        report = _RUNNER.preflight_marginal_slope_large_scale(
            n_train=400000,
            d_pc=16,
            centers=24,
            linkwiggle_knots=8,
            scorewarp_knots=8,
        )
        self.assertEqual(report.status, "PASS")
        text = "\n".join(report.lines)
        self.assertIn("Duchon tuple: order=0, power=9, length_scale=1", text)
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
        self.assertEqual(report.chunk_rows, _RUNNER.LARGE_SCALE_SURVIVAL_PREDICTION_CHUNK_ROWS)
        self.assertLess(report.largest_single_allocation_bytes, 400000 * 1000 * 8)

    def test_marginal_slope_preflight_status_is_grep_friendly(self) -> None:
        report = _RUNNER.preflight_marginal_slope_large_scale(
            n_train=400000,
            d_pc=16,
            centers=20,
            linkwiggle_knots=8,
            scorewarp_knots=7,
        )
        text = "\n".join(report.lines)
        self.assertIn("status: PASS", text)

    def test_run_method_subparser_exposes_emit_routing_log_flag(self) -> None:
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

    def test_prepare_ctn_subparser_requires_explicit_artifact_boundaries(self) -> None:
        parser = _RUNNER.build_parser()
        args = parser.parse_args(
            [
                "prepare-ctn",
                "--prep-dir",
                "/tmp/raw-prepared",
                "--out-dir",
                "/tmp/ctn-prepared",
            ]
        )
        self.assertEqual(args.prep_dir, Path("/tmp/raw-prepared"))
        self.assertEqual(args.out_dir, Path("/tmp/ctn-prepared"))

    def test_default_marginal_slope_lanes_share_one_ctn_contract(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)
        shared = _RUNNER.shared_ctn_spec(cfg)
        consumers = [spec for spec in _RUNNER.build_method_specs(cfg) if spec.marginal_slope]
        self.assertEqual(len(consumers), 3)
        self.assertEqual(shared.pc_count, 16)
        self.assertEqual(shared.centers, 24)
        self.assertEqual(shared.z_column, "pgs_ctn_z")
        self.assertEqual(
            {(spec.pc_count, spec.centers, spec.z_column) for spec in consumers},
            {(16, 24, "pgs_ctn_z")},
        )

    def test_run_method_refuses_raw_cohort_without_shared_ctn_artifact(self) -> None:
        spec = _RUNNER.shared_ctn_spec(_RUNNER.load_config(_RUNNER.DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            train = root / "train.csv"
            heldout = root / "heldout.csv"
            _write_csv(train, [{"pgs_raw": 0.1, "pc1_std": 0.0}])
            _write_csv(heldout, [{"pgs_raw": -0.1, "pc1_std": 0.2}])
            with self.assertRaisesRegex(RuntimeError, "shared CTN preprocessing artifact"):
                _RUNNER.require_shared_ctn_columns(spec, train, heldout)

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
        with tempfile.TemporaryDirectory() as raw_dir:
            tmp = Path(raw_dir) / "lane.routing.log"
            stderr = (
                "[HEARTBEAT] elapsed=1.2s cmd='gam ...' pid=42 cpu=10% mem=2%\n"
                "[OUTER] reml outer: n_params=6, gradient=Analytic, hessian=Analytic"
                " -> solver=Arc, search_hessian_source=Analytic"
                " [solver=Arc;hessian=Analytic;matrix-free=true]\n"
                "some unrelated stderr noise mentioning solver=Cheese\n"
                "[OUTER] aux outer: n_params=2, gradient=Analytic, hessian=Unavailable"
                " -> solver=Bfgs, search_hessian_source=BfgsApprox"
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
        mean_formula, slope_formula = _RUNNER.rust_marginal_slope_formula_classification(
            spec,
            centers=18,
        )
        self.assertIn("matern(pc1_std, pc2_std, pc3_std, pc4_std, centers=18)", mean_formula)
        self.assertIn("smooth(age_entry_std)", slope_formula)

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
        orig_survival_calibration = _RUNNER._survival_calibration
        try:
            _RUNNER.load_or_build_rust_binary = lambda: Path("/tmp/fake-gam")
            _RUNNER._survival_calibration = lambda: self.fail(
                "native Rust survival scoring must not import external calibration"
            )

            def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any = None) -> typing.Any:
                self.assertIsNotNone(cwd)
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
            _RUNNER._survival_calibration = orig_survival_calibration

        self.assertIn("Surv(__entry, time, event)", snapshots["fit_formula"])
        self.assertIn(
            "survmodel(spec=net, distribution=probit)",
            snapshots["fit_formula"],
        )
        self.assertIn("survival-likelihood=location-scale", result["model_spec"])
        self.assertIsNone(
            result["metrics"]["c_index"],
            "a one-row holdout has no comparable survival pair",
        )
        fit_rows = snapshots["fit_rows"]
        self.assertTrue(all(float(row["__entry"]) == 0.0 for row in fit_rows))
        self.assertEqual([float(row["time"]) for row in fit_rows], [4.0, 10.0])

        predict_inputs = snapshots["predict_inputs"]
        self.assertEqual(len(predict_inputs), 2)

        horizon = _RUNNER.survival_eval_horizon_from_rows(train_rows)
        self.assertEqual(len(predict_inputs[0]), len(test_rows))
        self.assertTrue(
            all(abs(float(row["time"]) - horizon) < 1e-12 for row in predict_inputs[0])
        )

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

        fit_cmd = snapshots["fit_cmd"]
        self.assertIn("--survival-likelihood", fit_cmd)
        self.assertEqual(
            fit_cmd[fit_cmd.index("--survival-likelihood") + 1],
            "location-scale",
        )

    def test_run_rust_survival_rejects_invalid_native_grid_columns(self) -> None:
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

            def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any = None) -> typing.Any:
                self.assertIsNotNone(cwd)
                if cmd[1] == "fit":
                    Path(cmd[cmd.index("--out") + 1]).write_text("{}", encoding="utf-8")
                    return 0, "", ""
                if cmd[1] == "predict":
                    input_path = Path(cmd[3])
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    input_rows = _RUNNER.read_csv_rows(input_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            rows = _RUNNER.generate_raw_cohort(cfg, Path(td), smoke=False)[0]
        self.assertGreater(len(rows), 20)
        pc1 = [float(r["pc1"]) for r in rows[:30]]
        pc2 = [float(r["pc2"]) for r in rows[:30]]
        self.assertGreater(len(set(round(v, 6) for v in pc1)), 10)
        self.assertGreater(len(set(round(v, 6) for v in pc2)), 10)


def _rust_sources() -> str:
    """Concatenate every Rust source under `crates/` once.

    The marker contract has two ends: the engine's `log::` format
    strings and the runner's regexes. Holding the regexes against
    hand-written samples only checks one end — it stays green after the
    producer is deleted, which is exactly how this layer went unnoticed
    for two months. These tests search the real tree so a removed or
    renamed emission site fails here.
    """
    global _RUST_SOURCES_CACHE
    if _RUST_SOURCES_CACHE is None:
        chunks: list[str] = []
        for path in sorted((_REPO_ROOT / "crates").rglob("*.rs")):
            if "target" in path.parts:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        _RUST_SOURCES_CACHE = "\n".join(chunks)
    return _RUST_SOURCES_CACHE


_RUST_SOURCES_CACHE: str | None = None


# One row per marker family the runner parses:
#   pattern name -> (sample line in TODAY's emitted format,
#                    literal that must appear in a live Rust format string)
#
# The sample is copied from the current `log::` format string with the
# placeholders filled in, so a format change that breaks the regex shows
# up as a parse failure here rather than as silently-empty aggregations
# in a four-hour benchmark run.
_MARKER_SAMPLES: dict[str, tuple[str, str]] = {
    "_PHASE_END_PATTERN": (
        "[PHASE] CTN(transformation-normal) fit end elapsed=12.500s",
        "[PHASE] CTN(transformation-normal) fit end elapsed=",
    ),
    "_PHASE_START_PATTERN": (
        "[PHASE] survival-margslope fit start n=400000",
        "[PHASE] survival-margslope fit start n=",
    ),
    "_BFGS_SUMMARY_PATTERN": (
        "[OUTER summary] BFGS converged in 12 iters elapsed=145.234s final_value=1.234567e3",
        "[OUTER summary] BFGS converged in {} iters elapsed=",
    ),
    "_SCHEDULE_TRANSITION_PATTERN": (
        "[OUTER schedule] inner-PIRLS cap transition accepted_iter=3 eval_count=9 "
        "g_ratio=1.000e-01 last_iters=7 converged=true ift_residual=1.000e-03 "
        "accept_rho=0.985 prev=8 new=12 (margin)",
        "[OUTER schedule] inner-PIRLS cap transition accepted_iter=",
    ),
    "_SCHEDULE_QUALITY_PATTERN": (
        "[OUTER schedule] inner-PIRLS cap transition accepted_iter=3 eval_count=9 "
        "g_ratio=1.000e-01 last_iters=7 converged=false ift_residual=n/a "
        "accept_rho=0.310 prev=8 new=12 (margin)",
        "last_iters={} converged={} ift_residual={} accept_rho={}",
    ),
    "_PIRLS_ITER_END_PATTERN": (
        "[PIRLS iter-end] iter=  3 elapsed=0.0345s lm_lambda=1.00e-06 g_norm=1.234e-03 "
        "last_dev_change=5.000e-05 last_halving=0",
        "[PIRLS iter-end] iter=",
    ),
    "_PIRLS_ITER_BREAKDOWN_PATTERN": (
        "[PIRLS iter-breakdown] iter=  3 attempts=2 curvature=0.012s solve=0.003s "
        "predred=0.000s candidate=0.045s other=0.001s",
        "[PIRLS iter-breakdown] iter=",
    ),
    "_PIRLS_CURVATURE_KIND_PATTERN": (
        "[STAGE] PIRLS update_with_curvature iter=7 curvature=Fisher elapsed=0.045s "
        "source=rebuilt",
        "[STAGE] PIRLS update_with_curvature iter={} curvature={:?} elapsed=",
    ),
    "_PIRLS_MID_ITER_FISHER_PATTERN": (
        "[PIRLS] mid-iter Fisher fallback iter=12 reason=candidate_err",
        "[PIRLS] mid-iter Fisher fallback iter={} reason=candidate_err",
    ),
    "_PIRLS_FORCE_FISHER_PATTERN": (
        "[PIRLS] force_fisher_for_rest engaged at iter=5 "
        "(consecutive_fisher_fallbacks=3) reason=iter_start",
        "[PIRLS] force_fisher_for_rest engaged at iter={} "
        "(consecutive_fisher_fallbacks={}) reason=iter_start",
    ),
    "_PIRLS_LM_TRAJECTORY_PATTERN": (
        "[PIRLS lm-trajectory] iter=  3 start_lambda=1.000e-6 final_lambda=3.333e-7 "
        "log10_ratio=-0.477 accept_rho=0.985 attempts=1",
        "[PIRLS lm-trajectory] iter=",
    ),
    "_PIRLS_SOLVE_END_PATTERN": (
        "[PIRLS solve-end] iters=12 elapsed=0.0345s g_norm_initial=1.234e+01 "
        "g_norm_final=4.567e-08 convergence_rate=2.345e-01 status=Converged",
        "[PIRLS solve-end] iters={} elapsed=",
    ),
    "_OUTER_HESSIAN_ROUTE_PATTERN": (
        "[OUTER hessian-route] choice=operator reason=large_k n=320000 p=128 k=32 "
        "callback_kernel=false subspace_trace=false scale_prefers_operator=true "
        "dense_workspace_bytes=5600000000",
        "[OUTER hessian-route] choice={route_choice} reason={route_reason} ",
    ),
    "_OUTER_HESSIAN_ELAPSED_PATTERN": (
        "[OUTER hessian-elapsed] choice=dense reason=below_crossover n=1000 p=20 k=4 "
        "elapsed=12.347s",
        "[OUTER hessian-elapsed] choice={route_choice} reason={route_reason} ",
    ),
    "_OUTER_EVAL_END_PATTERN": (
        "[STAGE] outer eval end order=ValueAndGradient elapsed=2.345s cost=1.234567e3 "
        "|g|=4.500e-02 (first-order bridge, iter=3) theta=[] g=[]",
        "[STAGE] outer eval end order=ValueAndGradient elapsed=",
    ),
    "_SEED_CASCADE_PATTERN": (
        "[OUTER] large_scale_fit_001: seed screening cascade complete elapsed=12.345s "
        "stages_used=2 final_cap=uncapped ranked=8/10",
        "seed screening cascade complete elapsed=",
    ),
    "_KAPPA_PHASE_PATTERN": (
        "[KAPPA-PHASE] phase=eval_outer call=5 order=ValueGradientHessian "
        "design_revision=Some(3) theta_norm=3.4500e+00 log_kappa_norm=1.2000e+00 "
        "elapsed_s=8.7654",
        "[KAPPA-PHASE] phase=eval_outer call={} order={:?} design_revision={:?} ",
    ),
    "_KAPPA_PHASE_SUMMARY_PATTERN": (
        "[KAPPA-PHASE-SUMMARY] log_kappa_dim=2 n_cost=12 cost_total_s=5.1840 n_eval=5 "
        "eval_total_s=43.8270 n_efs=2 efs_total_s=4.2196 optim_total_s=53.2306",
        "[KAPPA-PHASE-SUMMARY] log_kappa_dim={} n_cost={} cost_total_s=",
    ),
    "_IFT_QUALITY_PATTERN": (
        "[IFT-QUALITY] quality=3.456e-04 ift=2.000e+00 pred_residual=1.234e-03 "
        "cap_predicted=1.750e+00 iters=4",
        "[IFT-QUALITY] quality={:.3e} ift={:.3e} pred_residual={:.3e} "
        "cap_predicted={:.3e} iters={}",
    ),
    "_TANGENT_QUALITY_PATTERN": (
        "[TANGENT-QUALITY] quality=1.500e-02 pred_residual=1.502e-02 iters=5",
        "[TANGENT-QUALITY] quality={:.3e} pred_residual={:.3e} iters={}",
    ),
    "_IFT_REJECTED_PATTERN": (
        "[IFT-REJECTED] reason=large_drho max_drho=3.456e+00 cap=2.000e+00 drho_dim=4",
        "[IFT-REJECTED] reason=large_drho max_drho={:.3e} cap={:.3e} drho_dim={}",
    ),
    "_IFT_NOOP_PATTERN": (
        "[IFT-NOOP] reason=all_drho_below_eps max_drho=5.000e-15 drho_dim=4",
        "[IFT-NOOP] reason=all_drho_below_eps max_drho={:.3e} drho_dim={}",
    ),
    "_IFT_CACHE_HIT_PATTERN": (
        "[IFT-CACHE] outcome=hit drho_dim=4 p=128",
        "[IFT-CACHE] outcome=hit drho_dim={} p={}",
    ),
    "_IFT_CACHE_MISS_PATTERN": (
        "[IFT-CACHE] outcome=miss drho_dim=4 p=128 elapsed=2.345s",
        "[IFT-CACHE] outcome=miss drho_dim={} p={} elapsed=",
    ),
    "_TANGENT_PREDICT_PATTERN": (
        "[TANGENT-PREDICT] alpha=1.000e+00 cap=1.500e+00 drho_step_norm_sq=2.000e-02 "
        "drho_prev_norm_sq=2.000e-02",
        "[TANGENT-PREDICT] alpha={:.3e} cap={:.3e} drho_step_norm_sq={:.3e} "
        "drho_prev_norm_sq={:.3e}",
    ),
    "_TANGENT_REJECTED_PATTERN": (
        "[TANGENT-REJECTED] reason=degenerate_drho d_rho_norm_sq=1.000e-30",
        "[TANGENT-REJECTED] reason=degenerate_drho d_rho_norm_sq={:.3e}",
    ),
    "_TANGENT_NOOP_PATTERN": (
        "[TANGENT-NOOP] reason=prediction_equals_current alpha=1.000e-15",
        "[TANGENT-NOOP] reason=prediction_equals_current alpha={:.3e}",
    ),
    "_OUTER_NONFINITE_PATTERN": (
        "[OUTER non-finite] leverage h^G has non-finite entries",
        "[OUTER non-finite] leverage h^G has non-finite entries",
    ),
}


class MarkerContractTests(unittest.TestCase):
    """The producer/consumer contract itself, checked in both directions."""

    def test_every_parsed_marker_family_still_has_a_live_emission_site(self) -> None:
        sources = _rust_sources()
        missing = [
            name
            for name, (_sample, literal) in sorted(_MARKER_SAMPLES.items())
            if literal not in sources
        ]
        self.assertEqual(
            missing,
            [],
            "the runner parses marker families the engine no longer emits; either the "
            "emission site moved (re-derive the pattern and this literal) or the "
            f"capability was removed (then say so, do not keep a dead parser): {missing}",
        )

    def test_every_parsed_marker_family_matches_its_current_format_sample(self) -> None:
        failures: list[str] = []
        for name, (sample, _literal) in sorted(_MARKER_SAMPLES.items()):
            pattern = getattr(_RUNNER, name)
            if pattern.search(sample) is None:
                failures.append(f"{name} did not match {sample!r}")
        self.assertEqual(failures, [], f"marker patterns out of date: {failures}")

    def test_capture_filter_retains_every_marker_family_the_summary_parses(self) -> None:
        """A parser whose input the capture filter drops is inert.

        `_emit_phase_summary` reads the dedicated marker buffer, not raw
        stderr, so a family missing from `_INSTRUMENTATION_MARKERS`
        aggregates nothing on every real run while still passing the
        unit tests that hand it a synthetic line.
        """
        dropped = [
            name
            for name, (sample, _literal) in sorted(_MARKER_SAMPLES.items())
            if not _RUNNER._is_instrumentation_line(sample)
        ]
        self.assertEqual(
            dropped,
            [],
            f"capture filter drops marker families the summary parses: {dropped}",
        )

    def test_capture_filter_has_no_entry_without_a_parser(self) -> None:
        """The reverse direction: no marker is buffered for nobody."""
        samples = [sample for sample, _literal in _MARKER_SAMPLES.values()]
        unused = [
            marker
            for marker in _RUNNER._INSTRUMENTATION_MARKERS
            if not any(marker in sample for sample in samples)
        ]
        self.assertEqual(
            unused,
            [],
            f"capture filter retains marker families nothing parses: {unused}",
        )

    def test_ift_rejected_reason_set_matches_the_engine(self) -> None:
        """Every reason the engine can emit must parse, and the runner must
        not claim reasons the engine cannot emit."""
        emitted = set(
            re.findall(r"\[IFT-REJECTED\] reason=(\w+)", _rust_sources())
        )
        self.assertGreater(len(emitted), 4, "no [IFT-REJECTED] emission sites found")
        for reason in sorted(emitted):
            line = f"[IFT-REJECTED] reason={reason} drho_dim=4"
            self.assertEqual(
                _RUNNER._IFT_REJECTED_PATTERN.findall(line),
                [reason],
                f"reason {reason!r} emitted by the engine does not parse",
            )

    def test_outer_hessian_route_reason_set_matches_the_engine(self) -> None:
        sources = _rust_sources()
        routing = (
            _REPO_ROOT
            / "crates"
            / "gam-solve"
            / "src"
            / "reml"
            / "reml_outer_engine"
            / "outer_derivatives"
            / "routing.rs"
        ).read_text(encoding="utf-8")
        reasons = set(re.findall(r'reason:\s*"(\w+)"', routing))
        reasons.update(
            re.findall(r"\[OUTER hessian-route\] choice=\w+ reason=(\w+)", sources)
        )
        self.assertGreater(len(reasons), 4, "no routing reasons found")
        for reason in sorted(reasons):
            line = (
                f"[OUTER hessian-route] choice=operator reason={reason} "
                "n=320000 p=128 k=32 callback_kernel=false subspace_trace=false "
                "scale_prefers_operator=true"
            )
            matches = _RUNNER._OUTER_HESSIAN_ROUTE_PATTERN.findall(line)
            self.assertEqual(len(matches), 1, f"reason {reason!r} did not parse")
            self.assertEqual(matches[0][1], reason)

    def test_pirls_status_variants_all_parse(self) -> None:
        """`status={:?}` renders whatever `PirlsStatus` currently has, so the
        variant list is read off the enum rather than hard-coded here: a
        new variant that the runner's `(\\w+)` cannot capture must fail."""
        state = (
            _REPO_ROOT / "crates" / "gam-solve" / "src" / "pirls" / "state.rs"
        ).read_text(encoding="utf-8")
        body = state.split("pub enum PirlsStatus {", 1)
        self.assertEqual(len(body), 2, "PirlsStatus enum not found")
        variants = re.findall(
            r"^\s{4}([A-Z]\w*),\s*$", body[1].split("\n}", 1)[0], re.MULTILINE
        )
        self.assertGreaterEqual(
            len(variants), 4, f"suspiciously few PirlsStatus variants: {variants}"
        )
        for status in variants:
            line = (
                "[PIRLS solve-end] iters=5 elapsed=0.0010s g_norm_initial=1.000e-03 "
                "g_norm_final=1.000e-09 convergence_rate=3.000e-01 "
                f"status={status}"
            )
            matched = _RUNNER._PIRLS_SOLVE_END_PATTERN.findall(line)
            self.assertEqual(len(matched), 1, f"status {status!r} did not parse")
            self.assertEqual(matched[0][3], status)


class MarkerPatternTests(unittest.TestCase):
    def test_ift_quality_pattern_parses_field_layout(self) -> None:
        line = (
            "[IFT-QUALITY] quality=3.456e-04 ift=2.000e+00 pred_residual=1.234e-03 "
            "cap_predicted=1.750e+00 iters=4"
        )
        matches = _RUNNER._IFT_QUALITY_PATTERN.findall(line)
        self.assertEqual(len(matches), 1)
        quality, cap, pred_residual, cap_predicted, iters = matches[0]
        self.assertEqual(quality, "3.456e-04")
        self.assertEqual(cap, "2.000e+00")
        self.assertEqual(pred_residual, "1.234e-03")
        self.assertEqual(cap_predicted, "1.750e+00")
        self.assertEqual(iters, "4")

    def test_tangent_quality_pattern_carries_no_ift_cap_fields(self) -> None:
        """The tangent-line branch does not read or write the IFT |Δρ| cap,
        so its marker must not report one — a shared field layout would
        publish a cap that does not govern that branch."""
        line = "[TANGENT-QUALITY] quality=1.500e-02 pred_residual=1.502e-02 iters=5"
        matches = _RUNNER._TANGENT_QUALITY_PATTERN.findall(line)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], ("1.500e-02", "1.502e-02", "5"))
        self.assertEqual(_RUNNER._IFT_QUALITY_PATTERN.findall(line), [])

    def test_ift_rejected_and_noop_patterns_capture_reason(self) -> None:
        reasons = [
            (
                "[IFT-REJECTED] reason=large_drho max_drho=3.456e+00 cap=2.000e+00 "
                "drho_dim=4",
                "large_drho",
            ),
            (
                "[IFT-REJECTED] reason=hessian_factorize_failed max_drho=1.234e+00 "
                "drho_dim=4",
                "hessian_factorize_failed",
            ),
            (
                "[IFT-REJECTED] reason=non_finite_solution max_drho=1.234e+00 drho_dim=4",
                "non_finite_solution",
            ),
            ("[IFT-REJECTED] reason=qs_dim_mismatch qs_dim=10x10 expected_p=8", "qs_dim_mismatch"),
            (
                "[IFT-REJECTED] reason=rho_dim_mismatch new_rho_dim=2 cache_rho_dim=1",
                "rho_dim_mismatch",
            ),
            (
                "[IFT-REJECTED] reason=penalty_dim_mismatch penalties_dim=0 cache_rho_dim=1",
                "penalty_dim_mismatch",
            ),
            (
                "[IFT-REJECTED] reason=beta_dim_mismatch cache_beta_dim=3 expected_p=4",
                "beta_dim_mismatch",
            ),
            ("[IFT-REJECTED] reason=active_constraints joint_dim=6", "active_constraints"),
        ]
        for line, expected in reasons:
            self.assertEqual(
                _RUNNER._IFT_REJECTED_PATTERN.findall(line),
                [expected],
                f"failed to extract reason from: {line!r}",
            )
        for line, expected in (
            (
                "[IFT-NOOP] reason=all_drho_below_eps max_drho=5.000e-15 drho_dim=4",
                "all_drho_below_eps",
            ),
            (
                "[IFT-NOOP] reason=all_dtheta_below_eps max_dtheta=5.000e-15 joint_dim=6",
                "all_dtheta_below_eps",
            ),
        ):
            self.assertEqual(_RUNNER._IFT_NOOP_PATTERN.findall(line), [expected])

    def test_pirls_solve_end_pattern_captures_iters_elapsed_rate(self) -> None:
        sample = (
            "2026-07-29T03:14:15Z INFO  gam::solver::pirls: "
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
        nan_sample = (
            "[PIRLS solve-end] iters=1 elapsed=0.0001s g_norm_initial=NaN "
            "g_norm_final=NaN convergence_rate=NaN status=Converged"
        )
        nan_matches = _RUNNER._PIRLS_SOLVE_END_PATTERN.findall(nan_sample)
        self.assertEqual(len(nan_matches), 1)
        self.assertEqual(nan_matches[0][2], "NaN")

    def test_outer_hessian_elapsed_pattern_extracts_timing(self) -> None:
        cases = [
            (
                "[OUTER hessian-elapsed] choice=dense reason=below_crossover "
                "n=1000 p=20 k=4 elapsed=12.347s",
                "dense", "below_crossover", "12.347",
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
            self.assertEqual(matches[0][0], expected_choice)
            self.assertEqual(matches[0][1], expected_reason)
            self.assertEqual(matches[0][5], expected_elapsed)

    def test_outer_eval_end_pattern_captures_order_and_elapsed(self) -> None:
        cases = [
            (
                "[STAGE] outer eval end order=ValueAndGradient elapsed=2.345s "
                "cost=1.234567e3 |g|=4.500e-02 (first-order bridge, iter=3)",
                "ValueAndGradient", "2.345",
            ),
            (
                "[STAGE] outer eval end order=ValueGradientHessian elapsed=12.789s "
                "cost=1.234567e3 |g|=4.500e-02",
                "ValueGradientHessian", "12.789",
            ),
            (
                "[STAGE] outer eval end order=Value elapsed=0.001s cost=1.234567e3 "
                "trial_rho_distance=1.000e-03 (first-order bridge, iter=3, cached=true)",
                "Value", "0.001",
            ),
        ]
        for line, expected_order, expected_elapsed in cases:
            matches = _RUNNER._OUTER_EVAL_END_PATTERN.findall(line)
            self.assertEqual(len(matches), 1, f"order {expected_order!r} did not parse")
            self.assertEqual(matches[0], (expected_order, expected_elapsed))

    def test_seed_cascade_pattern_captures_cascade_summary(self) -> None:
        cases = [
            (
                "[OUTER] large_scale_fit_001: seed screening cascade complete "
                "elapsed=12.345s stages_used=2 final_cap=uncapped ranked=8/10",
                ("12.345", "2", "uncapped", "8", "10"),
            ),
            (
                "[OUTER] survival-marginal-slope/large-scale-1: seed screening cascade "
                "complete elapsed=0.500s stages_used=1 final_cap=10 ranked=4/4",
                ("0.500", "1", "10", "4", "4"),
            ),
        ]
        for line, expected in cases:
            matches = _RUNNER._SEED_CASCADE_PATTERN.findall(line)
            self.assertEqual(len(matches), 1, f"cascade did not parse: {line}")
            self.assertEqual(matches[0], expected)

    def test_pirls_curvature_kind_pattern_captures_observed_and_fisher(self) -> None:
        for kind in ("Observed", "Fisher"):
            line = (
                f"[STAGE] PIRLS update_with_curvature iter=200 curvature={kind} "
                "elapsed=12.345s source=rebuilt"
            )
            self.assertEqual(
                _RUNNER._PIRLS_CURVATURE_KIND_PATTERN.findall(line),
                [kind],
                f"curvature kind {kind!r} did not parse",
            )

    def test_pirls_mid_iter_fisher_pattern_captures_both_reasons(self) -> None:
        for iter_str, reason in (("3", "gain_rejection"), ("200", "candidate_err")):
            line = f"[PIRLS] mid-iter Fisher fallback iter={iter_str} reason={reason}"
            matches = _RUNNER._PIRLS_MID_ITER_FISHER_PATTERN.findall(line)
            self.assertEqual(len(matches), 1, f"reason {reason!r} did not parse")
            self.assertEqual(matches[0], (iter_str, reason))

    def test_pirls_force_fisher_pattern_captures_all_three_reasons(self) -> None:
        for iter_str, count, reason in (
            ("5", "3", "iter_start"),
            ("12", "4", "gain_rejection"),
            ("2", "3", "candidate_err"),
        ):
            line = (
                f"[PIRLS] force_fisher_for_rest engaged at iter={iter_str} "
                f"(consecutive_fisher_fallbacks={count}) reason={reason}"
            )
            matches = _RUNNER._PIRLS_FORCE_FISHER_PATTERN.findall(line)
            self.assertEqual(len(matches), 1, f"reason {reason!r} did not parse")
            self.assertEqual(matches[0], (iter_str, count, reason))

    def test_pirls_iter_breakdown_pattern_extracts_all_seven_subphases(self) -> None:
        line = (
            "[PIRLS iter-breakdown] iter=  3 attempts=2 curvature=0.012s "
            "solve=0.003s predred=0.000s candidate=0.045s other=0.001s"
        )
        matches = _RUNNER._PIRLS_ITER_BREAKDOWN_PATTERN.findall(line)
        self.assertEqual(len(matches), 1)
        self.assertEqual(
            matches[0], ("3", "2", "0.012", "0.003", "0.000", "0.045", "0.001")
        )
        wide = (
            "[PIRLS iter-breakdown] iter=200 attempts=64 curvature=12.345s "
            "solve=678.901s predred=1234.567s candidate=89.012s other=0.500s"
        )
        wide_matches = _RUNNER._PIRLS_ITER_BREAKDOWN_PATTERN.findall(wide)
        self.assertEqual(len(wide_matches), 1)
        self.assertEqual(wide_matches[0][0], "200")
        self.assertEqual(wide_matches[0][1], "64")
        self.assertEqual(wide_matches[0][3], "678.901")

    def test_pirls_lm_trajectory_pattern_handles_finite_and_nan_rho(self) -> None:
        finite = (
            "[PIRLS lm-trajectory] iter=  3 start_lambda=1.000e-6 "
            "final_lambda=3.333e-7 log10_ratio=-0.477 accept_rho=0.985 attempts=1"
        )
        matches = _RUNNER._PIRLS_LM_TRAJECTORY_PATTERN.findall(finite)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][3:6], ("-0.477", "0.985", "1"))
        nan = (
            "[PIRLS lm-trajectory] iter= 10 start_lambda=1.000e-3 "
            "final_lambda=2.000e-3 log10_ratio=0.301 accept_rho=NaN attempts=8"
        )
        nan_matches = _RUNNER._PIRLS_LM_TRAJECTORY_PATTERN.findall(nan)
        self.assertEqual(len(nan_matches), 1)
        self.assertEqual(nan_matches[0][4], "NaN")
        self.assertEqual(nan_matches[0][5], "8")

    def test_kappa_phase_patterns_parse_per_call_and_both_summary_variants(self) -> None:
        per_call = [
            (
                "[KAPPA-PHASE] phase=cost call=12 design_revision=Some(3) "
                "theta_norm=3.4500e+00 log_kappa_norm=1.2000e+00 elapsed_s=0.4321",
                ("cost", "12", "0.4321"),
            ),
            (
                "[KAPPA-PHASE] phase=eval_outer call=5 order=ValueGradientHessian "
                "design_revision=Some(3) theta_norm=3.4500e+00 "
                "log_kappa_norm=1.2000e+00 elapsed_s=8.7654",
                ("eval_outer", "5", "8.7654"),
            ),
            (
                "[KAPPA-PHASE] phase=efs call=2 design_revision=None "
                "theta_norm=3.4500e+00 log_kappa_norm=1.2000e+00 elapsed_s=2.1098",
                ("efs", "2", "2.1098"),
            ),
        ]
        for line, expected in per_call:
            matches = _RUNNER._KAPPA_PHASE_PATTERN.findall(line)
            self.assertEqual(len(matches), 1, f"did not parse: {line}")
            self.assertEqual(matches[0], expected)

        # Both summary variants are emitted; fields are read by NAME so
        # the longer one's leading `n_rows=` cannot shift them by one.
        short = (
            "[KAPPA-PHASE-SUMMARY] log_kappa_dim=2 n_cost=12 cost_total_s=5.1840 "
            "n_eval=5 eval_total_s=43.8270 n_efs=2 efs_total_s=4.2196 "
            "optim_total_s=53.2306"
        )
        long = (
            "[KAPPA-PHASE-SUMMARY] n_rows=400000 log_kappa_dim=2 n_cost=12 "
            "cost_total_s=5.1840 n_eval=5 eval_total_s=43.8270 n_efs=2 "
            "efs_total_s=4.2196 value_realization_failures=0 "
            "value_evaluation_failures=0 slow_path_resets=0 design_revision_delta=1 "
            "nfree_skip_row_touches=0 nfree_miss_shape=0 nfree_miss_value=0 "
            "nfree_miss_gradient=0 nfree_miss_penalty=0 nfree_miss_revision=0 "
            "nfree_miss_second_order=0 nfree_miss_other=0 optim_total_s=53.2306"
        )
        for text, expected_n_rows in ((short, None), (long, "400000")):
            match = _RUNNER._KAPPA_PHASE_SUMMARY_PATTERN.search(text)
            self.assertIsNotNone(match, f"summary variant did not parse: {text}")
            assert match is not None
            self.assertEqual(match.group("n_rows"), expected_n_rows)
            self.assertEqual(match.group("log_kappa_dim"), "2")
            self.assertEqual(match.group("n_cost"), "12")
            self.assertAlmostEqual(float(match.group("cost_total_s")), 5.1840)
            self.assertEqual(match.group("n_eval"), "5")
            self.assertEqual(match.group("n_efs"), "2")
            self.assertAlmostEqual(float(match.group("optim_total_s")), 53.2306)

    def test_bfgs_summary_pattern_covers_all_outcome_variants(self) -> None:
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
            if expected_iters is None:
                self.assertIn(iters, ("", None))
            else:
                self.assertEqual(iters, expected_iters)
            self.assertEqual(elapsed, expected_elapsed)


class PhaseSummaryAggregationTests(unittest.TestCase):
    def _run_summary(self, captured_stderr: str) -> str:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            _RUNNER._emit_phase_summary(captured_stderr, "cmd-preview", timed_out=False, rc=0)
        return buf.getvalue()


    @staticmethod
    def _ift_quality(quality: str, iters: int) -> str:
        return (
            f"[IFT-QUALITY] quality={quality} ift=2.000e+00 pred_residual={quality} "
            f"cap_predicted=2.000e+00 iters={iters}"
        )

    @staticmethod
    def _tangent_quality(quality: str, iters: int) -> str:
        return (
            f"[TANGENT-QUALITY] quality={quality} pred_residual={quality} iters={iters}"
        )

    @staticmethod
    def _solve_end(rate: str, iters: int = 8, status: str = "Converged") -> str:
        return (
            f"[PIRLS solve-end] iters={iters} elapsed=0.0010s g_norm_initial=1.000e+01 "
            f"g_norm_final=1.000e-02 convergence_rate={rate} status={status}"
        )

    @staticmethod
    def _curvature(iter_no: int, kind: str) -> str:
        return (
            f"[STAGE] PIRLS update_with_curvature iter={iter_no} curvature={kind} "
            "elapsed=0.010s source=rebuilt"
        )

    @staticmethod
    def _tangent_predict(alpha: str) -> str:
        return (
            f"[TANGENT-PREDICT] alpha={alpha} cap=1.500e+00 "
            "drho_step_norm_sq=2.000e-02 drho_prev_norm_sq=2.000e-02"
        )

    def test_phase_summary_tolerates_off_by_one_tangent_marker_drift(self) -> None:
        stderr = "\n".join([
            self._tangent_predict("1.000e+00"),
            self._tangent_predict("1.100e+00"),
            self._tangent_quality("1.000e-03", 4),
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertNotIn("tangent_marker_drift", out)

    def test_phase_summary_flags_tangent_marker_drift(self) -> None:
        stderr = "\n".join(
            [self._tangent_predict(f"1.{idx}00e+00") for idx in range(5)]
            + [
                self._tangent_quality("1.000e-03", 4),
                self._tangent_quality("2.000e-03", 4),
                "[PHASE] my-fit fit end elapsed=10.0s",
            ]
        )
        out = self._run_summary(stderr)
        self.assertIn("tangent_marker_drift=predict=5_vs_quality=2", out)

    def test_phase_summary_omits_outer_nonfinite_when_count_is_zero(self) -> None:
        stderr = "[PHASE] my-fit fit end elapsed=10.0s\n"
        out = self._run_summary(stderr)
        self.assertNotIn("outer_nonfinite", out)

    def test_phase_summary_aggregates_outer_nonfinite_warnings(self) -> None:
        stderr = "\n".join([
            "[OUTER non-finite] rho_a_vals[2] at (2, 2) = NaN",
            "[OUTER non-finite] penalty_a_k_betas[1] has non-finite",
            "[OUTER non-finite] penalty_a_k_betas[3] has non-finite",
            "[OUTER non-finite] leverage h^G has non-finite entries",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("outer_nonfinite=4", out)
        self.assertIn("outer_nonfinite_at=[", out)
        self.assertIn("penalty_a_k_betas[1]=1", out)
        self.assertIn("penalty_a_k_betas[3]=1", out)

    def test_phase_summary_aggregates_ift_accept_reject_noop_independently(self) -> None:
        stderr = "\n".join([
            self._ift_quality("1.000e-04", 3),
            self._ift_quality("2.000e-03", 4),
            self._ift_quality("5.000e-02", 5),
            self._ift_quality("8.000e-01", 6),
            "[IFT-REJECTED] reason=large_drho max_drho=3.000e+00 cap=2.000e+00 drho_dim=4",
            "[IFT-NOOP] reason=all_drho_below_eps max_drho=5.000e-15 drho_dim=4",
            "[IFT-NOOP] reason=all_drho_below_eps max_drho=4.000e-15 drho_dim=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("ift_predicts=4", out)
        self.assertIn("ift_rejects=1", out)
        self.assertIn("ift_noops=2", out)
        self.assertIn("ift_reasons=[large_drho=1]", out)
        self.assertIn("ift_accept_rate=0.57", out)

    def test_phase_summary_distinguishes_accept_rate_from_active(self) -> None:
        stderr = "\n".join(
            [self._ift_quality(f"{idx + 1}.000e-04", 3) for idx in range(4)]
            + ["[IFT-REJECTED] reason=large_drho max_drho=3.000e+00 cap=2.000e+00 drho_dim=4"]
            + ["[IFT-NOOP] reason=all_drho_below_eps max_drho=5.000e-15 drho_dim=4"] * 5
            + ["[PHASE] my-fit fit end elapsed=10.0s"]
        )
        out = self._run_summary(stderr)
        rate_lines = [line for line in out.splitlines() if "ift_accept_rate=" in line]
        self.assertEqual(len(rate_lines), 1)
        line = rate_lines[0]
        self.assertIn("ift_accept_rate=0.40", line)
        self.assertIn("ift_accept_rate_active=0.80", line)
        self.assertLess(
            line.index("ift_accept_rate="), line.index("ift_accept_rate_active=")
        )

    def test_phase_summary_aggregates_ift_iters_distribution(self) -> None:
        stderr = "\n".join(
            [
                self._ift_quality("1.000e-04", 3),
                self._ift_quality("2.000e-04", 4),
                self._ift_quality("3.000e-04", 5),
                self._ift_quality("4.000e-04", 6),
                self._ift_quality("5.000e-04", 12),
                "[PHASE] my-fit fit end elapsed=10.0s",
            ]
        )
        out = self._run_summary(stderr)
        self.assertIn("ift_iters_p50=5", out)
        self.assertIn("ift_iters_p95=12", out)
        self.assertIn("ift_iters_max=12", out)

    def test_phase_summary_aggregates_tangent_quality_separately_from_ift(self) -> None:
        stderr = "\n".join([
            self._ift_quality("1.000e-04", 3),
            self._ift_quality("2.000e-04", 3),
            self._ift_quality("5.000e-04", 3),
            self._ift_quality("1.000e-03", 4),
            self._tangent_quality("1.500e-02", 5),
            self._tangent_quality("2.500e-02", 6),
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("ift_predicts=4", out)
        self.assertIn("ift_p50=5.00e-04", out)
        self.assertIn("tangent_quality_predicts=2", out)
        self.assertIn("tangent_p50=", out)
        self.assertNotIn("ift_p50=1.50e-02", out)
        self.assertNotIn("ift_p50=2.50e-02", out)

    def test_phase_summary_aggregates_tangent_iters_distribution(self) -> None:
        stderr = "\n".join([
            self._tangent_quality("1.000e-03", 4),
            self._tangent_quality("2.000e-03", 5),
            self._tangent_quality("3.000e-03", 6),
            self._tangent_quality("4.000e-03", 8),
            self._tangent_quality("5.000e-03", 15),
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_iters_p50=6", out)
        self.assertIn("tangent_iters_p95=15", out)
        self.assertIn("tangent_iters_max=15", out)

    def test_phase_summary_surfaces_tangent_alpha_distribution(self) -> None:
        stderr = "\n".join([
            self._tangent_predict("8.000e-01"),
            self._tangent_predict("1.200e+00"),
            self._tangent_predict("1.450e+00"),
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=3", out)
        self.assertIn("tangent_alpha_p50=1.20", out)
        self.assertIn("tangent_alpha_max=1.45", out)

    def test_phase_summary_aggregates_tangent_noop_marker(self) -> None:
        stderr = "\n".join([
            self._tangent_predict("1.000e+00"),
            self._tangent_predict("1.100e+00"),
            "[TANGENT-NOOP] reason=prediction_equals_current alpha=1.000e-15",
            "[TANGENT-NOOP] reason=prediction_equals_current alpha=5.000e-16",
            "[TANGENT-NOOP] reason=prediction_equals_current alpha=1.000e-14",
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=2.500e+00 cap=1.500e+00",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=2", out)
        self.assertIn("tangent_rejects=1", out)
        self.assertIn("tangent_noops=3", out)
        self.assertIn("tangent_reasons=[alpha_above_cap=1]", out)

    def test_phase_summary_aggregates_tangent_line_predicts_and_rejects(self) -> None:
        stderr = "\n".join([
            self._tangent_predict("1.234e+00"),
            self._tangent_predict("8.765e-01"),
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=2.345e+00 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=alpha_negative alpha=-1.234e-01 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=3.000e+00 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=rho_dim_mismatch new_rho_dim=4 cur_rho_dim=3 "
            "prev_rho_dim=3",
            "[TANGENT-REJECTED] reason=beta_dim_mismatch cur_beta_dim=10 prev_beta_dim=12",
            "[TANGENT-REJECTED] reason=degenerate_drho d_rho_norm_sq=1.000e-30",
            "[TANGENT-REJECTED] reason=nonfinite_alpha step_dot_d=NaN "
            "d_rho_norm_sq=2.345e-02",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=2", out)
        self.assertIn("tangent_rejects=7", out)
        self.assertIn(
            "tangent_reasons=[alpha_above_cap=2,alpha_negative=1,beta_dim_mismatch=1,"
            "degenerate_drho=1,nonfinite_alpha=1,rho_dim_mismatch=1]",
            out,
        )

    def test_phase_summary_tangent_accept_rate_split_matches_ift(self) -> None:
        stderr = "\n".join(
            ["[IFT-REJECTED] reason=large_drho max_drho=3.000e+00 cap=2.000e+00 drho_dim=4"] * 6
            + [
                self._tangent_predict("5.000e-01"),
                self._tangent_predict("7.000e-01"),
                self._tangent_predict("1.200e+00"),
                "[TANGENT-REJECTED] reason=alpha_above_cap alpha=2.500e+00 cap=1.500e+00",
                "[TANGENT-NOOP] reason=alpha_below_eps alpha=1.000e-15 eps=1.000e-12",
                "[TANGENT-NOOP] reason=alpha_below_eps alpha=2.000e-15 eps=1.000e-12",
                self._tangent_quality("1.000e-04", 3),
                self._tangent_quality("2.000e-04", 3),
                self._tangent_quality("3.000e-04", 3),
                "[PHASE] my-fit fit end elapsed=10.0s",
            ]
        )
        out = self._run_summary(stderr)
        rate_lines = [line for line in out.splitlines() if "tangent_accept_rate=" in line]
        self.assertEqual(len(rate_lines), 1, f"expected 1 rate line, got {rate_lines}")
        self.assertIn("tangent_accept_rate=0.50", rate_lines[0])
        self.assertIn("tangent_accept_rate_active=0.75", rate_lines[0])

    def test_phase_summary_kappa_complete_surfaces_per_phase_max_and_p95(self) -> None:
        lines = [
            f"[KAPPA-PHASE] phase=eval_outer call={idx + 1} "
            "order=ValueGradientHessian design_revision=Some(1) theta_norm=1.0000e+00 "
            "log_kappa_norm=1.0000e+00 elapsed_s=0.3000"
            for idx in range(30)
        ]
        lines.append(
            "[KAPPA-PHASE] phase=eval_outer call=31 order=ValueGradientHessian "
            "design_revision=Some(1) theta_norm=1.0000e+00 log_kappa_norm=1.0000e+00 "
            "elapsed_s=15.0000"
        )
        lines.append(
            "[KAPPA-PHASE-SUMMARY] log_kappa_dim=2 n_cost=0 cost_total_s=0.0000 "
            "n_eval=31 eval_total_s=24.0000 n_efs=0 efs_total_s=0.0000 "
            "optim_total_s=24.0000"
        )
        lines.append("[PHASE] my-fit fit end elapsed=24.0s")
        out = self._run_summary("\n".join(lines))
        self.assertIn("kappa_eval_calls=31", out)
        self.assertIn("kappa_eval_total=24.0s", out)
        self.assertIn("kappa_eval_outer_max=15.00s", out)
        self.assertIn("kappa_eval_outer_p95=0.30s", out)

    def test_phase_summary_kappa_incomplete_surfaces_per_phase_max_and_p95(self) -> None:
        lines = [
            f"[KAPPA-PHASE] phase=eval_outer call={idx + 1} "
            "order=ValueGradientHessian design_revision=Some(1) theta_norm=1.0000e+00 "
            "log_kappa_norm=1.0000e+00 elapsed_s=0.5000"
            for idx in range(50)
        ]
        lines.append(
            "[KAPPA-PHASE] phase=eval_outer call=51 order=ValueGradientHessian "
            "design_revision=Some(1) theta_norm=1.0000e+00 log_kappa_norm=1.0000e+00 "
            "elapsed_s=60.0000"
        )
        lines.extend(
            f"[KAPPA-PHASE] phase=cost call={idx + 1} design_revision=Some(1) "
            "theta_norm=1.0000e+00 log_kappa_norm=1.0000e+00 elapsed_s=0.1000"
            for idx in range(5)
        )
        lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        out = self._run_summary("\n".join(lines))
        self.assertIn("kappa_optim_INCOMPLETE", out)
        self.assertIn("kappa_eval_outer_calls=51", out)
        self.assertIn("kappa_eval_outer_max=60.00s", out)
        self.assertIn("kappa_eval_outer_p95=0.50s", out)
        self.assertIn("kappa_cost_calls=5", out)
        self.assertIn("kappa_cost_max=0.10s", out)

    def test_phase_summary_emits_fit_health_combining_warm_start_and_pirls(self) -> None:
        stderr = "\n".join([
            self._ift_quality("1.000e-04", 3),
            self._ift_quality("2.000e-04", 3),
            self._solve_end("5.500e-01", iters=8),
            self._solve_end("6.500e-01", iters=10),
            self._solve_end("7.000e-01", iters=12),
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("[WARM-START health]", out)
        self.assertIn("[PIRLS health]", out)
        fit_lines = [line for line in out.splitlines() if line.startswith("[FIT health]")]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=DEGRADED", fit_lines[0])
        self.assertIn("pirls=DEGRADED", fit_lines[0])
        self.assertIn("curvature=ABSENT", fit_lines[0])

    def test_phase_summary_emits_pirls_health_verdict_alongside_warm_start(self) -> None:
        stderr = "\n".join([
            self._solve_end("2.500e-01", iters=4),
            self._solve_end("1.585e-01", iters=5),
            self._solve_end("4.000e-01", iters=3),
            self._ift_quality("1.000e-04", 3),
            self._ift_quality("2.000e-04", 4),
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("[WARM-START health]", out)
        self.assertIn("[PIRLS health]", out)
        self.assertIn("verdict=HEALTHY", out.splitlines()[-1])

    def test_phase_summary_curvature_healthy_when_fisher_frac_low(self) -> None:
        lines = [
            self._ift_quality("1.000e-04", 3),
            self._solve_end("2.000e-01", iters=4),
            self._solve_end("2.300e-01", iters=4),
        ]
        lines.extend(self._curvature(idx, "Observed") for idx in range(1, 26))
        lines.append(self._curvature(26, "Fisher"))
        lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        out = self._run_summary("\n".join(lines))
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=HEALTHY", curv_lines[0])
        fit_lines = [line for line in out.splitlines() if line.startswith("[FIT health]")]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=HEALTHY", fit_lines[0])
        self.assertIn("dominant_axis=pirls", fit_lines[0])

    def test_phase_summary_curvature_marginal_when_fisher_frac_in_band(self) -> None:
        lines = [
            self._ift_quality("1.000e-04", 3),
            self._solve_end("2.000e-01", iters=4),
            self._solve_end("2.300e-01", iters=4),
        ]
        lines.extend(self._curvature(idx, "Observed") for idx in range(1, 10))
        lines.append(self._curvature(10, "Fisher"))
        lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        out = self._run_summary("\n".join(lines))
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=MARGINAL", curv_lines[0])
        self.assertIn("fisher_frac=0.10", curv_lines[0])
        self.assertIn("force_fisher_n=0", curv_lines[0])
        fit_lines = [line for line in out.splitlines() if line.startswith("[FIT health]")]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=MARGINAL", fit_lines[0])
        self.assertIn("dominant_axis=curvature", fit_lines[0])
        self.assertIn("curvature=MARGINAL", fit_lines[0])

    def test_phase_summary_curvature_degraded_drives_fit_health(self) -> None:
        lines = [
            self._ift_quality("1.000e-04", 3),
            self._ift_quality("2.000e-04", 3),
            self._solve_end("2.000e-01", iters=4),
            self._solve_end("2.500e-01", iters=5),
            self._solve_end("2.300e-01", iters=4),
        ]
        lines.extend(self._curvature(idx, "Observed") for idx in range(1, 6))
        lines.extend(self._curvature(idx, "Fisher") for idx in range(6, 11))
        lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        out = self._run_summary("\n".join(lines))
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=DEGRADED", curv_lines[0])
        self.assertIn("fisher_frac=0.50", curv_lines[0])
        fit_lines = [line for line in out.splitlines() if line.startswith("[FIT health]")]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=DEGRADED", fit_lines[0])
        self.assertIn("warm_start=HEALTHY", fit_lines[0])
        self.assertIn("pirls=HEALTHY", fit_lines[0])
        self.assertIn("curvature=DEGRADED", fit_lines[0])
        self.assertIn("dominant_axis=curvature", fit_lines[0])

    def test_phase_summary_reports_when_the_marker_buffer_rolled_over(self) -> None:
        """A truncated buffer makes every percentile a suffix statistic.
        The summary must say so rather than presenting partial
        distributions as if they covered the whole run."""
        stderr = "\n".join([
            self._ift_quality("1.000e-04", 3),
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        clean = self._run_summary(stderr)
        self.assertNotIn("marker_lines_dropped", clean)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            _RUNNER._emit_phase_summary(
                stderr, "cmd-preview", timed_out=False, rc=0, dropped_marker_lines=417
            )
        self.assertIn("marker_lines_dropped=417", buf.getvalue())


class HealthVerdictTests(unittest.TestCase):
    def test_combine_fit_verdicts_worst_wins(self) -> None:
        combine = _RUNNER._combine_fit_verdicts
        self.assertEqual(combine("HEALTHY", "HEALTHY", "HEALTHY"), "HEALTHY")
        self.assertEqual(combine("HEALTHY", "HEALTHY", "DEGRADED"), "DEGRADED")
        self.assertEqual(combine("HEALTHY", "HEALTHY", "MARGINAL"), "MARGINAL")
        self.assertEqual(combine("DEGRADED", "HEALTHY", "MARGINAL"), "DEGRADED")
        self.assertEqual(combine("HEALTHY", "MARGINAL", "DEGRADED"), "DEGRADED")
        self.assertEqual(combine("HEALTHY", None, "MARGINAL"), "MARGINAL")
        self.assertEqual(combine(None, None, "DEGRADED"), "DEGRADED")
        self.assertEqual(combine(None, None, None), "NO-DATA")

    def test_dominant_axis_for_verdict_resolves_correctly(self) -> None:
        dom = _RUNNER._dominant_axis_for_verdict
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
        self.assertEqual(
            dom("DEGRADED", warm_start="DEGRADED", pirls="DEGRADED", curvature="DEGRADED"),
            "pirls",
        )
        self.assertEqual(
            dom("DEGRADED", warm_start="DEGRADED", pirls="HEALTHY", curvature="DEGRADED"),
            "warm_start",
        )
        self.assertEqual(
            dom("MARGINAL", warm_start="HEALTHY", pirls="HEALTHY", curvature="MARGINAL"),
            "curvature",
        )
        self.assertEqual(
            dom("HEALTHY", warm_start="HEALTHY", pirls="HEALTHY", curvature="HEALTHY"),
            "pirls",
        )
        self.assertEqual(dom("NO-DATA", warm_start=None, pirls=None, curvature=None), "none")
        self.assertEqual(
            dom("MARGINAL", warm_start=None, pirls="MARGINAL", curvature=None), "pirls"
        )

    def test_curvature_health_verdict_classifies_tiers(self) -> None:
        verdict = _RUNNER._curvature_health_verdict
        self.assertEqual(verdict(fisher_frac=0.0, force_fisher_n=0)[0], "HEALTHY")
        self.assertEqual(verdict(fisher_frac=0.04, force_fisher_n=0)[0], "HEALTHY")
        self.assertEqual(verdict(fisher_frac=0.05, force_fisher_n=0)[0], "MARGINAL")
        self.assertEqual(verdict(fisher_frac=0.19, force_fisher_n=0)[0], "MARGINAL")
        self.assertEqual(verdict(fisher_frac=0.20, force_fisher_n=0)[0], "DEGRADED")
        self.assertEqual(verdict(fisher_frac=0.50, force_fisher_n=0)[0], "DEGRADED")
        self.assertEqual(verdict(fisher_frac=0.0, force_fisher_n=1)[0], "DEGRADED")
        self.assertEqual(verdict(fisher_frac=0.04, force_fisher_n=1)[0], "DEGRADED")
        self.assertEqual(verdict(fisher_frac=None, force_fisher_n=0)[0], "NO-DATA")
        tier, detail = verdict(fisher_frac=0.123, force_fisher_n=2)
        self.assertEqual(tier, "DEGRADED")
        self.assertIn("fisher_frac=0.12", detail)
        self.assertIn("force_fisher_n=2", detail)

    def test_pirls_health_verdict_classifies_tiers(self) -> None:
        verdict = _RUNNER._pirls_health_verdict
        tier, detail = verdict(rates=[0.1, 0.2, 0.3, 0.4, 0.45])
        self.assertEqual(tier, "HEALTHY", f"detail={detail}")
        self.assertIn("max=0.450", detail)
        with_outliers = [0.1] * 25 + [0.2] * 25 + [0.3] * 25 + [0.4] * 22 + [0.6] * 3
        tier, detail = verdict(rates=with_outliers)
        self.assertEqual(tier, "HEALTHY", f"detail={detail}")
        self.assertIn("max=0.600", detail)
        tier, detail = verdict(rates=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        self.assertEqual(tier, "MARGINAL", f"detail={detail}")
        tier, detail = verdict(rates=[0.5, 0.6, 0.7, 0.8])
        self.assertEqual(tier, "DEGRADED", f"detail={detail}")
        tier, detail = verdict(rates=[0.1, 0.2, 0.3, 0.95])
        self.assertEqual(tier, "DEGRADED", f"detail={detail}")
        tier, detail = verdict(rates=[])
        self.assertEqual(tier, "NO-DATA")
        self.assertIn("n_solves=0", detail)

    def test_warm_start_health_verdict_classifies_tiers_correctly(self) -> None:
        verdict = _RUNNER._warm_start_health_verdict
        tier, detail = verdict(
            n_accepts=8,
            n_rejects=1,
            n_noops=1,
            residuals=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2],
        )
        self.assertEqual(tier, "HEALTHY", f"detail={detail}")
        self.assertIn("coverage=0.80", detail)
        tier, detail = verdict(
            n_accepts=4, n_rejects=2, n_noops=2, residuals=[0.05, 0.10, 0.15, 0.20]
        )
        self.assertEqual(tier, "MARGINAL", f"detail={detail}")
        tier, detail = verdict(n_accepts=1, n_rejects=8, n_noops=1, residuals=[0.6])
        self.assertEqual(tier, "DEGRADED", f"detail={detail}")
        tier, detail = verdict(n_accepts=0, n_rejects=5, n_noops=0, residuals=[])
        self.assertEqual(tier, "DEGRADED", f"detail={detail}")
        tier, detail = verdict(n_accepts=0, n_rejects=0, n_noops=4, residuals=[])
        self.assertEqual(tier, "DEGRADED", f"detail={detail}")
        tier, detail = verdict(n_accepts=0, n_rejects=0, n_noops=0, residuals=[])
        self.assertEqual(tier, "NO-DATA", f"detail={detail}")
        self.assertEqual(
            verdict(n_accepts=7, n_rejects=2, n_noops=1, residuals=[0.04] * 7)[0],
            "HEALTHY",
        )
        tier, detail = verdict(
            n_accepts=6, n_rejects=2, n_noops=2, residuals=[0.10] * 6
        )
        self.assertEqual(tier, "MARGINAL", f"detail={detail}")

    def test_warm_start_health_verdict_p95_saturation_guard(self) -> None:
        verdict = _RUNNER._warm_start_health_verdict
        tier, detail = verdict(
            n_accepts=100, n_rejects=0, n_noops=0, residuals=[1e-3] * 80 + [0.5] * 20
        )
        self.assertEqual(tier, "MARGINAL", f"detail={detail}")
        self.assertIn("p95_resid=5.00e-01", detail)
        tier, detail = verdict(
            n_accepts=100, n_rejects=0, n_noops=0, residuals=[1e-3] * 97 + [0.5] * 3
        )
        self.assertEqual(tier, "HEALTHY", f"detail={detail}")
        self.assertIn("p50_resid=1.00e-03", detail)
        self.assertIn("p95_resid=1.00e-03", detail)

    def test_warm_start_health_verdict_outer_nonfinite_overrides_to_degraded(self) -> None:
        verdict = _RUNNER._warm_start_health_verdict
        tier, detail = verdict(
            n_accepts=10,
            n_rejects=0,
            n_noops=0,
            residuals=[1e-5] * 10,
            n_outer_nonfinite=1,
        )
        self.assertEqual(tier, "DEGRADED", f"detail={detail}")
        self.assertIn("n_outer_nonfinite=1", detail)
        self.assertEqual(
            verdict(
                n_accepts=10,
                n_rejects=0,
                n_noops=0,
                residuals=[1e-5] * 10,
                n_outer_nonfinite=0,
            )[0],
            "HEALTHY",
        )
        tier, detail = verdict(
            n_accepts=0, n_rejects=2, n_noops=0, residuals=[], n_outer_nonfinite=3
        )
        self.assertEqual(tier, "DEGRADED")
        self.assertIn("n_outer_nonfinite=3", detail)

    def test_warm_start_health_verdict_detail_includes_tangent_stats(self) -> None:
        verdict = _RUNNER._warm_start_health_verdict
        tier, detail = verdict(
            n_accepts=8,
            n_rejects=2,
            n_noops=0,
            residuals=[1e-3] * 8,
            n_tangent_accepts=2,
            tangent_p50=0.025,
        )
        self.assertEqual(tier, "HEALTHY")
        self.assertIn("n_tangent_accepts=2", detail)
        self.assertIn("tangent_p50=2.50e-02", detail)
        detail = verdict(
            n_accepts=8,
            n_rejects=0,
            n_noops=2,
            residuals=[1e-3] * 8,
            n_tangent_accepts=0,
            tangent_p50=None,
        )[1]
        self.assertNotIn("tangent_", detail)
        detail = verdict(
            n_accepts=4,
            n_rejects=2,
            n_noops=0,
            residuals=[1e-3] * 4,
            n_tangent_accepts=2,
            tangent_p50=None,
        )[1]
        self.assertIn("n_tangent_accepts=2", detail)
        self.assertNotIn("tangent_p50=", detail)


class DefaultBenchmarkConfigTests(unittest.TestCase):
    """RED tests for issue #221.

    The default config ships methods that the runner's spec validator must
    accept. Today `r_mgcv_jointpc_duchon60` is a `disease` method with
    `backend: r_mgcv`, but `validate_method_spec` rejects every disease
    backend except `rust_gam`, so the default config cannot be loaded
    end-to-end.
    """

    def test_default_config_loads_and_builds_specs(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)
        specs = _RUNNER.build_method_specs(cfg)
        self.assertGreater(len(specs), 0, "default config must define at least one method")

    def test_every_default_method_passes_spec_validation(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)
        raw_methods = list(cfg.get("methods", []))
        self.assertGreater(len(raw_methods), 0)
        failures: list[str] = []
        for entry in raw_methods:
            name = entry.get("name", "<unnamed>")
            try:
                spec = _RUNNER.MethodSpec(**entry)
                _RUNNER.validate_method_spec(spec)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        self.assertEqual(failures, [], f"default-config methods failed validation: {failures}")


if __name__ == "__main__":
    unittest.main()
