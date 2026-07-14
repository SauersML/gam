import typing
import csv
import importlib.util
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
        self.assertIsNone(rigid.logslope_linkwiggle_knots)
        self.assertEqual(warped.mean_linkwiggle_knots, 8)
        self.assertEqual(warped.logslope_linkwiggle_knots, 8)

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
        self.assertIn("order=0", mean_formula)
        self.assertIn("power=9", mean_formula)
        self.assertIn("length_scale=1", mean_formula)
        self.assertNotIn("pgs_ctn_z", mean_formula)
        self.assertIn("linkwiggle(internal_knots=8)", mean_formula)
        self.assertIn("linkwiggle(internal_knots=7)", logslope_formula)

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


class MarkerPatternTests(unittest.TestCase):
    def test_ift_quality_pattern_parses_field_layout(self) -> None:
        new = "[IFT-QUALITY] residual=3.456e-04 converged_norm=1.234e+00 predicted_norm=1.234e+00 drho_norm=5.678e-01 h_pen_logdet=2.345e+01 iters=4"
        nan = "[IFT-QUALITY] residual=3.456e-04 converged_norm=1.234e+00 predicted_norm=1.234e+00 drho_norm=NaN h_pen_logdet=NaN iters=4"

        new_match = _RUNNER._IFT_QUALITY_PATTERN.findall(new)
        self.assertEqual(len(new_match), 1)
        self.assertEqual(new_match[0][3], "5.678e-01")
        self.assertEqual(new_match[0][4], "2.345e+01")

        nan_match = _RUNNER._IFT_QUALITY_PATTERN.findall(nan)
        self.assertEqual(len(nan_match), 1)
        self.assertEqual(nan_match[0][3], "NaN")

    def test_ift_rejected_and_noop_patterns_capture_reason(self) -> None:
        reasons = [
            ("[IFT-REJECTED] reason=large_drho max_drho=3.456e+00 cap=2.000e+00 drho_dim=4", "large_drho"),
            ("[IFT-REJECTED] reason=hessian_factorize_failed drho_dim=4", "hessian_factorize_failed"),
            ("[IFT-REJECTED] reason=non_finite_solution max_drho=1.234e+00 drho_dim=4", "non_finite_solution"),
            ("[IFT-REJECTED] reason=qs_dim_mismatch qs_dim=10x10 expected_p=8", "qs_dim_mismatch"),
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
        nan_sample = (
            "[PIRLS solve-end] iters=1 elapsed=0.0001s g_norm_initial=NaN "
            "g_norm_final=NaN convergence_rate=NaN status=Converged"
        )
        nan_matches = _RUNNER._PIRLS_SOLVE_END_PATTERN.findall(nan_sample)
        self.assertEqual(len(nan_matches), 1)
        self.assertEqual(nan_matches[0][2], "NaN")
        self.assertEqual(nan_matches[0][3], "Converged")
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
    def test_outer_hessian_route_pattern_covers_all_reasons(self) -> None:
        cases = [
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
            (
                "[OUTER hessian-route] choice=operator reason=dense_memory_budget "
                "n=10000 p=10000 k=2 callback_kernel=true subspace_trace=false "
                "scale_prefers_operator=true dense_workspace_bytes=5600000000",
                "operator", "dense_memory_budget",
            ),
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
            choice, reason, elapsed = matches[0][0], matches[0][1], matches[0][5]
            self.assertEqual(choice, expected_choice)
            self.assertEqual(reason, expected_reason)
            self.assertEqual(elapsed, expected_elapsed)

    def test_outer_eval_end_pattern_captures_order_and_elapsed(self) -> None:
        cases = [
            (
                "[STAGE] outer eval end order=ValueAndGradient elapsed=2.345s "
                "cost=1.23e3 |g|=4.5e-2 (first-order bridge, iter=3)",
                "ValueAndGradient", "2.345",
            ),
            (
                "[STAGE] outer eval end order=ValueGradientHessian elapsed=12.789s "
                "cost=1.23e3 |g|=4.5e-2",
                "ValueGradientHessian", "12.789",
            ),
            (
                "[STAGE] outer eval end order=ValueAndGradient elapsed=0.001s "
                "cost=1.23e3 |g|=4.5e-2",
                "ValueAndGradient", "0.001",
            ),
        ]
        for line, expected_order, expected_elapsed in cases:
            matches = _RUNNER._OUTER_EVAL_END_PATTERN.findall(line)
            self.assertEqual(
                len(matches),
                1,
                f"order {expected_order!r} did not parse: {line}",
            )
            order, elapsed = matches[0]
            self.assertEqual(order, expected_order)
            self.assertEqual(elapsed, expected_elapsed)

    def test_seed_cascade_pattern_captures_cascade_summary(self) -> None:
        cases = [
            (
                "[OUTER] large_scale_fit_001: seed screening cascade complete "
                "elapsed=12.345s stages_used=2 final_cap=uncapped ranked=8/10",
                "12.345", "2", "uncapped", "8", "10",
            ),
            (
                "[OUTER] survival-marginal-slope/large-scale-1: seed screening cascade complete "
                "elapsed=0.500s stages_used=1 final_cap=10 ranked=4/4",
                "0.500", "1", "10", "4", "4",
            ),
        ]
        for line, exp_elapsed, exp_stages, exp_cap, exp_ranked, exp_seeds in cases:
            matches = _RUNNER._SEED_CASCADE_PATTERN.findall(line)
            self.assertEqual(
                len(matches), 1,
                f"cascade did not parse: {line}",
            )
            elapsed, stages, cap, ranked, seeds = matches[0]
            self.assertEqual(elapsed, exp_elapsed)
            self.assertEqual(stages, exp_stages)
            self.assertEqual(cap, exp_cap)
            self.assertEqual(ranked, exp_ranked)
            self.assertEqual(seeds, exp_seeds)

    def test_pirls_curvature_kind_pattern_captures_observed_and_fisher(self) -> None:
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
        cases = [
            (
                "[PIRLS] mid-iter Fisher fallback iter=3 reason=gain_rejection",
                "3", "gain_rejection",
            ),
            (
                "[PIRLS] mid-iter Fisher fallback iter=12 reason=candidate_err",
                "12", "candidate_err",
            ),
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
        line = (
            "[PIRLS iter-breakdown] iter=  3 attempts=2 curvature=0.012s "
            "solve=0.003s predred=0.000s candidate=0.045s other=0.001s"
        )
        matches = _RUNNER._PIRLS_ITER_BREAKDOWN_PATTERN.findall(line)
        self.assertEqual(len(matches), 1)
        iter_str, attempts, curv, solve, predred, candidate, other = matches[0]
        self.assertEqual(iter_str, "3")
        self.assertEqual(attempts, "2")
        self.assertEqual(curv, "0.012")
        self.assertEqual(solve, "0.003")
        self.assertEqual(predred, "0.000")
        self.assertEqual(candidate, "0.045")
        self.assertEqual(other, "0.001")
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
        ratio, rho, attempts = finite_matches[0][3:6]
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
        log_kappa_dim, n_cost, cost_s, optim_s = sm[0][0], sm[0][1], sm[0][2], sm[0][7]
        self.assertEqual(log_kappa_dim, "2")
        self.assertEqual(n_cost, "12")
        self.assertAlmostEqual(float(cost_s), 5.1840)
        self.assertAlmostEqual(float(optim_s), 53.2306)


class PhaseSummaryAggregationTests(unittest.TestCase):
    def _run_summary(self, captured_stderr: str) -> str:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            _RUNNER._emit_phase_summary(captured_stderr, "cmd-preview", timed_out=False, rc=0)
        return buf.getvalue()

    def test_warm_start_health_verdict_detail_includes_tangent_stats(self) -> None:
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=8, n_rejects=2, n_noops=0,
            residuals=[1e-3] * 8,
            n_tangent_accepts=2,
            tangent_p50=0.025,
        )
        self.assertEqual(v, "HEALTHY")
        self.assertIn("n_tangent_accepts=2", d)
        self.assertIn("tangent_p50=2.50e-02", d)
        d = _RUNNER._warm_start_health_verdict(
            n_accepts=8, n_rejects=0, n_noops=2,
            residuals=[1e-3] * 8,
            n_tangent_accepts=0,
            tangent_p50=None,
        )[1]
        self.assertNotIn("tangent_", d)
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=4, n_rejects=2, n_noops=0,
            residuals=[1e-3] * 4,
            n_tangent_accepts=2,
            tangent_p50=None,
        )
        self.assertIn("n_tangent_accepts=2", d)
        self.assertNotIn("tangent_p50=", d)

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
        self.assertEqual(
            dom("NO-DATA", warm_start=None, pirls=None, curvature=None),
            "none",
        )
        self.assertEqual(
            dom("MARGINAL", warm_start=None, pirls="MARGINAL", curvature=None),
            "pirls",
        )

    def test_curvature_health_verdict_classifies_tiers(self) -> None:
        verdict = _RUNNER._curvature_health_verdict
        self.assertEqual(
            verdict(fisher_frac=0.0, force_fisher_n=0)[0], "HEALTHY"
        )
        self.assertEqual(
            verdict(fisher_frac=0.04, force_fisher_n=0)[0], "HEALTHY"
        )
        self.assertEqual(
            verdict(fisher_frac=0.05, force_fisher_n=0)[0], "MARGINAL"
        )
        self.assertEqual(
            verdict(fisher_frac=0.19, force_fisher_n=0)[0], "MARGINAL"
        )
        self.assertEqual(
            verdict(fisher_frac=0.20, force_fisher_n=0)[0], "DEGRADED"
        )
        self.assertEqual(
            verdict(fisher_frac=0.50, force_fisher_n=0)[0], "DEGRADED"
        )
        self.assertEqual(
            verdict(fisher_frac=0.0, force_fisher_n=1)[0], "DEGRADED"
        )
        self.assertEqual(
            verdict(fisher_frac=0.04, force_fisher_n=1)[0], "DEGRADED"
        )
        self.assertEqual(
            verdict(fisher_frac=None, force_fisher_n=0)[0], "NO-DATA"
        )
        v, d = verdict(fisher_frac=0.123, force_fisher_n=2)
        self.assertEqual(v, "DEGRADED")
        self.assertIn("fisher_frac=0.12", d)
        self.assertIn("force_fisher_n=2", d)

    def test_phase_summary_emits_fit_health_combining_warm_start_and_pirls(self) -> None:
        stderr = "\n".join([
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[PIRLS solve-end] iters=8 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=5.500e-01 status=Converged",
            "[PIRLS solve-end] iters=10 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=6.500e-01 status=Converged",
            "[PIRLS solve-end] iters=12 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=7.000e-01 status=Converged",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("[WARM-START health]", out)
        self.assertIn("[PIRLS health]", out)
        self.assertIn("[FIT health]", out)
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=DEGRADED", fit_lines[0])
        self.assertIn("warm_start=", fit_lines[0])
        self.assertIn("pirls=DEGRADED", fit_lines[0])
        self.assertIn("curvature=ABSENT", fit_lines[0])

    def test_phase_summary_tangent_accept_rate_split_matches_ift(self) -> None:
        stderr_lines = [
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-REJECTED] reason=large_drho",
            "[TANGENT-PREDICT] alpha=0.500 cap=1.500 drho_step_norm_sq=1.0e-2 drho_prev_norm_sq=4.0e-2",
            "[TANGENT-PREDICT] alpha=0.700 cap=1.500 drho_step_norm_sq=2.0e-2 drho_prev_norm_sq=4.0e-2",
            "[TANGENT-PREDICT] alpha=1.200 cap=1.500 drho_step_norm_sq=5.0e-2 drho_prev_norm_sq=4.0e-2",
            "[TANGENT-REJECTED] reason=alpha_above_cap",
            "[TANGENT-NOOP] reason=alpha_below_eps",
            "[TANGENT-NOOP] reason=alpha_below_eps",
            "[TANGENT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[TANGENT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[TANGENT-QUALITY] residual=3.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ]
        stderr = "\n".join(stderr_lines)
        out = self._run_summary(stderr)
        rate_lines = [
            line for line in out.splitlines()
            if "tangent_accept_rate=" in line
        ]
        self.assertEqual(len(rate_lines), 1, f"expected 1 rate line, got {rate_lines}")
        line = rate_lines[0]
        self.assertIn("tangent_accept_rate=0.50", line)
        self.assertIn("tangent_accept_rate_active=0.75", line)

    def test_phase_summary_distinguishes_accept_rate_from_active(self) -> None:
        stderr_lines = [
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=3.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=4.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-REJECTED] reason=large_drho",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[IFT-NOOP] reason=all_drho_below_eps",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ]
        stderr = "\n".join(stderr_lines)
        out = self._run_summary(stderr)
        lines_with_rate = [
            line for line in out.splitlines()
            if "ift_accept_rate=" in line
        ]
        self.assertEqual(len(lines_with_rate), 1)
        line = lines_with_rate[0]
        self.assertIn("ift_accept_rate=0.40", line)
        self.assertIn("ift_accept_rate_active=0.80", line)
        i_accept = line.index("ift_accept_rate=")
        i_active = line.index("ift_accept_rate_active=")
        self.assertLess(i_accept, i_active)

    def test_phase_summary_curvature_healthy_when_fisher_frac_low(self) -> None:
        stderr_lines = [
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.000e-01 status=Converged",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.300e-01 status=Converged",
        ]
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
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=HEALTHY", fit_lines[0])
        self.assertIn("dominant_axis=pirls", fit_lines[0])

    def test_phase_summary_curvature_marginal_when_fisher_frac_in_band(self) -> None:
        stderr_lines = [
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.000e-01 status=Converged",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.300e-01 status=Converged",
        ]
        for i in range(1, 10):
            stderr_lines.append(
                f"[STAGE] PIRLS update_with_curvature iter={i} curvature=Observed elapsed=0.01s"
            )
        stderr_lines.append(
            "[STAGE] PIRLS update_with_curvature iter=10 curvature=Fisher elapsed=0.01s"
        )
        stderr_lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        stderr = "\n".join(stderr_lines)
        out = self._run_summary(stderr)
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=MARGINAL", curv_lines[0])
        self.assertIn("fisher_frac=0.10", curv_lines[0])
        self.assertIn("force_fisher_n=0", curv_lines[0])
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=MARGINAL", fit_lines[0])
        self.assertIn("dominant_axis=curvature", fit_lines[0])
        self.assertIn("curvature=MARGINAL", fit_lines[0])

    def test_phase_summary_curvature_degraded_drives_fit_health(self) -> None:
        stderr = "\n".join([
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.000e-01 status=Converged",
            "[PIRLS solve-end] iters=5 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.500e-01 status=Converged",
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-02 convergence_rate=2.300e-01 status=Converged",
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
        curv_lines = [
            line for line in out.splitlines() if line.startswith("[CURVATURE health]")
        ]
        self.assertEqual(len(curv_lines), 1)
        self.assertIn("verdict=DEGRADED", curv_lines[0])
        self.assertIn("fisher_frac=0.50", curv_lines[0])
        fit_lines = [
            line for line in out.splitlines() if line.startswith("[FIT health]")
        ]
        self.assertEqual(len(fit_lines), 1)
        self.assertIn("verdict=DEGRADED", fit_lines[0])
        self.assertIn("warm_start=HEALTHY", fit_lines[0])
        self.assertIn("pirls=HEALTHY", fit_lines[0])
        self.assertIn("curvature=DEGRADED", fit_lines[0])
        self.assertIn("dominant_axis=curvature", fit_lines[0])

    def test_pirls_health_verdict_classifies_tiers(self) -> None:
        v, d = _RUNNER._pirls_health_verdict(rates=[0.1, 0.2, 0.3, 0.4, 0.45])
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("max=0.450", d)
        rates_with_outliers = [0.1] * 25 + [0.2] * 25 + [0.3] * 25 + [0.4] * 22 + [0.6] * 3
        v, d = _RUNNER._pirls_health_verdict(rates=rates_with_outliers)
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("max=0.600", d)
        v, d = _RUNNER._pirls_health_verdict(
            rates=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")
        v, d = _RUNNER._pirls_health_verdict(rates=[0.5, 0.6, 0.7, 0.8])
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        v, d = _RUNNER._pirls_health_verdict(
            rates=[0.1, 0.2, 0.3, 0.95],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        v, d = _RUNNER._pirls_health_verdict(rates=[])
        self.assertEqual(v, "NO-DATA")
        self.assertIn("n_solves=0", d)

    def test_phase_summary_emits_pirls_health_verdict_alongside_warm_start(self) -> None:
        stderr = "\n".join([
            "[PIRLS solve-end] iters=4 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-04 convergence_rate=2.500e-01 status=Converged",
            "[PIRLS solve-end] iters=5 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=1.0e-05 convergence_rate=1.585e-01 status=Converged",
            "[PIRLS solve-end] iters=3 elapsed=0.001s g_norm_initial=1.0e+01 g_norm_final=3.0e-02 convergence_rate=4.000e-01 status=Converged",
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("[WARM-START health]", out)
        self.assertIn("[PIRLS health]", out)
        self.assertIn("verdict=HEALTHY", out.splitlines()[-1])

    def test_warm_start_health_verdict_p95_saturation_guard(self) -> None:
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=100, n_rejects=0, n_noops=0,
            residuals=[1e-3] * 80 + [0.5] * 20,
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")
        self.assertIn("p95_resid=5.00e-01", d)
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=100, n_rejects=0, n_noops=0,
            residuals=[1e-3] * 97 + [0.5] * 3,
        )
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("p50_resid=1.00e-03", d)
        self.assertIn("p95_resid=1.00e-03", d)

    def test_warm_start_health_verdict_outer_nonfinite_overrides_to_degraded(self) -> None:
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=10, n_rejects=0, n_noops=0,
            residuals=[1e-5] * 10,
            n_outer_nonfinite=1,
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        self.assertIn("n_outer_nonfinite=1", d)
        v_no_override = _RUNNER._warm_start_health_verdict(
            n_accepts=10, n_rejects=0, n_noops=0,
            residuals=[1e-5] * 10,
            n_outer_nonfinite=0,
        )[0]
        self.assertEqual(v_no_override, "HEALTHY")
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=2, n_noops=0,
            residuals=[],
            n_outer_nonfinite=3,
        )
        self.assertEqual(v, "DEGRADED")
        self.assertIn("n_outer_nonfinite=3", d)

    def test_warm_start_health_verdict_classifies_tiers_correctly(self) -> None:
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=8, n_rejects=1, n_noops=1,
            residuals=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2],
        )
        self.assertEqual(v, "HEALTHY", f"detail={d}")
        self.assertIn("coverage=0.80", d)
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=4, n_rejects=2, n_noops=2,
            residuals=[0.05, 0.10, 0.15, 0.20],
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=1, n_rejects=8, n_noops=1,
            residuals=[0.6],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=5, n_noops=0,
            residuals=[],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=0, n_noops=4,
            residuals=[],
        )
        self.assertEqual(v, "DEGRADED", f"detail={d}")
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=0, n_rejects=0, n_noops=0,
            residuals=[],
        )
        self.assertEqual(v, "NO-DATA", f"detail={d}")
        v = _RUNNER._warm_start_health_verdict(
            n_accepts=7, n_rejects=2, n_noops=1,
            residuals=[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        )[0]
        self.assertEqual(v, "HEALTHY")
        v, d = _RUNNER._warm_start_health_verdict(
            n_accepts=6, n_rejects=2, n_noops=2,
            residuals=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        )
        self.assertEqual(v, "MARGINAL", f"detail={d}")

    def test_phase_summary_aggregates_ift_iters_distribution(self) -> None:
        stderr = "\n".join([
            "[IFT-QUALITY] residual=1.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=3",
            "[IFT-QUALITY] residual=2.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[IFT-QUALITY] residual=3.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=5",
            "[IFT-QUALITY] residual=4.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=6",
            "[IFT-QUALITY] residual=5.0e-04 converged_norm=1.0 predicted_norm=1.0 iters=12",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("ift_iters_p50=5", out)
        self.assertIn("ift_iters_p95=12", out)
        self.assertIn("ift_iters_max=12", out)

    def test_phase_summary_aggregates_tangent_noop_marker(self) -> None:
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
        self.assertIn("tangent_reasons=[alpha_above_cap=1]", out)

    def test_phase_summary_aggregates_tangent_iters_distribution(self) -> None:
        stderr = "\n".join([
            "[TANGENT-QUALITY] residual=1.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[TANGENT-QUALITY] residual=2.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=5",
            "[TANGENT-QUALITY] residual=3.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=6",
            "[TANGENT-QUALITY] residual=4.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=8",
            "[TANGENT-QUALITY] residual=5.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=15",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_iters_p50=6", out)
        self.assertIn("tangent_iters_p95=15", out)
        self.assertIn("tangent_iters_max=15", out)

    def test_phase_summary_aggregates_tangent_quality_separately_from_ift(self) -> None:
        stderr = "\n".join([
            "[IFT-QUALITY] residual=1.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=2.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=5.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=1.000e-03 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=4",
            "[TANGENT-QUALITY] residual=1.500e-02 converged_norm=1.000e+00 predicted_norm=9.985e-01 iters=5",
            "[TANGENT-QUALITY] residual=2.500e-02 converged_norm=1.000e+00 predicted_norm=9.975e-01 iters=6",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("ift_predicts=4", out)
        self.assertIn("ift_p50=5.00e-04", out)
        self.assertIn("tangent_quality_predicts=2", out)
        self.assertIn("tangent_p50=", out)
        self.assertNotIn("ift_p50=1.50e-02", out)
        self.assertNotIn("ift_p50=2.50e-02", out)

    def test_phase_summary_surfaces_tangent_alpha_distribution(self) -> None:
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=8.000e-01 cap=1.500e+00 drho_step_norm_sq=2.345e-02 drho_prev_norm_sq=4.567e-02",
            "[TANGENT-PREDICT] alpha=1.200e+00 cap=1.500e+00 drho_step_norm_sq=3.000e-02 drho_prev_norm_sq=4.000e-02",
            "[TANGENT-PREDICT] alpha=1.450e+00 cap=1.500e+00 drho_step_norm_sq=4.000e-02 drho_prev_norm_sq=3.000e-02",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=3", out)
        self.assertIn("tangent_alpha_p50=1.20", out)
        self.assertIn("tangent_alpha_max=1.45", out)

    def test_phase_summary_flags_tangent_marker_drift(self) -> None:
        stderr = "\n".join([
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
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=1.000e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.100e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-QUALITY] residual=1.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertNotIn("tangent_marker_drift", out)

    def test_phase_summary_kappa_complete_surfaces_per_phase_max_and_p95(self) -> None:
        stderr_lines = []
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
        self.assertIn("kappa_eval_calls=31", out)
        self.assertIn("kappa_eval_total=24.0s", out)
        self.assertIn("kappa_eval_outer_max=15.00s", out)
        self.assertIn("kappa_eval_outer_p95=0.30s", out)

    def test_phase_summary_kappa_incomplete_surfaces_per_phase_max_and_p95(self) -> None:
        stderr_lines = []
        for i in range(50):
            stderr_lines.append(
                f"[KAPPA-PHASE] phase=eval_outer call={i+1} order=ValueGradientHessian "
                f"theta_norm=1.0e+00 log_kappa_norm=1.0e+00 elapsed_s=0.5"
            )
        stderr_lines.append(
            "[KAPPA-PHASE] phase=eval_outer call=51 order=ValueGradientHessian "
            "theta_norm=1.0e+00 log_kappa_norm=1.0e+00 elapsed_s=60.0"
        )
        for i in range(5):
            stderr_lines.append(
                f"[KAPPA-PHASE] phase=cost call={i+1} theta_norm=1.0e+00 "
                f"log_kappa_norm=1.0e+00 elapsed_s=0.1"
            )
        stderr_lines.append("[PHASE] my-fit fit end elapsed=10.0s")
        out = self._run_summary("\n".join(stderr_lines))
        self.assertIn("kappa_optim_INCOMPLETE", out)
        self.assertIn("kappa_eval_outer_calls=51", out)
        self.assertIn("kappa_eval_outer_max=60.00s", out)
        self.assertIn("kappa_eval_outer_p95=0.50s", out)
        self.assertIn("kappa_cost_calls=5", out)
        self.assertIn("kappa_cost_max=0.10s", out)

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

    def test_phase_summary_omits_outer_nonfinite_when_count_is_zero(self) -> None:
        stderr = "[PHASE] my-fit fit end elapsed=10.0s\n"
        out = self._run_summary(stderr)
        self.assertNotIn("outer_nonfinite", out)

    def test_phase_summary_aggregates_tangent_line_predicts_and_rejects(self) -> None:
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=1.234e+00 cap=1.500e+00 drho_step_norm_sq=2.345e-02 drho_prev_norm_sq=4.567e-02",
            "[TANGENT-PREDICT] alpha=8.765e-01 cap=1.500e+00 drho_step_norm_sq=2.345e-02 drho_prev_norm_sq=4.567e-02",
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=2.345e+00 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=alpha_negative alpha=-1.234e-01 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=alpha_above_cap alpha=3.000e+00 cap=1.500e+00",
            "[TANGENT-REJECTED] reason=rho_dim_mismatch new_rho_dim=4 cur_rho_dim=3 prev_rho_dim=3",
            "[TANGENT-REJECTED] reason=beta_dim_mismatch cur_beta_dim=10 prev_beta_dim=12",
            "[TANGENT-REJECTED] reason=degenerate_drho d_rho_norm_sq=1.000e-30",
            "[TANGENT-REJECTED] reason=nonfinite_alpha step_dot_d=NaN d_rho_norm_sq=2.345e-02",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertIn("tangent_predicts=2", out)
        self.assertIn("tangent_rejects=7", out)
        self.assertIn(
            "tangent_reasons=[alpha_above_cap=2,alpha_negative=1,beta_dim_mismatch=1,degenerate_drho=1,nonfinite_alpha=1,rho_dim_mismatch=1]",
            out,
        )

    def test_phase_summary_aggregates_ift_accept_reject_noop_independently(self) -> None:
        stderr = "\n".join([
            "[IFT-QUALITY] residual=1.000e-04 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=3",
            "[IFT-QUALITY] residual=2.000e-03 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=4",
            "[IFT-QUALITY] residual=5.000e-02 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=5",
            "[IFT-QUALITY] residual=8.000e-01 converged_norm=1.000e+00 predicted_norm=1.000e+00 iters=6",
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
