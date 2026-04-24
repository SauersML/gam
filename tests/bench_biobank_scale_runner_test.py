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
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BiobankScaleRunnerTests(unittest.TestCase):
    def test_default_biobank_matrix_keeps_400k_binomial_marginal_slope_lane(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)

        self.assertEqual(int(cfg["target_n"]), 400000)

        payload = {"include": [spec.__dict__ for spec in _RUNNER.build_method_specs(cfg)]}
        methods = {row["name"]: row for row in payload["include"]}

        self.assertIn("rust_margslope_aniso_duchon16d_50", methods)
        lane = methods["rust_margslope_aniso_duchon16d_50"]
        self.assertEqual(lane["dataset"], "disease")
        self.assertEqual(lane["family"], "binomial")
        self.assertEqual(lane["backend"], "rust_gam")
        self.assertEqual(lane["spatial_basis"], "duchon")
        self.assertEqual(lane["centers"], 24)
        self.assertTrue(lane["marginal_slope"])
        self.assertTrue(lane["scale_dimensions"])
        self.assertEqual(lane["z_column"], "pgs_ctn_z")

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

    def test_run_rust_survival_uses_explicit_survival_contract(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="rust_gamlss_survival_ps",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="ps",
            smooth_kind="separate",
            survival_likelihood="location-scale",
            survival_distribution="probit",
        )
        train_rows = [
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
        test_rows = [
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
        snapshots: dict[str, object] = {}
        orig_load_bin = _RUNNER.load_or_build_rust_binary
        orig_run_cmd = _RUNNER.run_cmd_stream
        orig_survival_metrics = _RUNNER.survival_metrics
        try:
            _RUNNER.load_or_build_rust_binary = lambda: Path("/tmp/fake-gam")

            def _fake_run_cmd(cmd, cwd=None):
                if cmd[1] == "fit":
                    fit_input = Path(cmd[-2])
                    snapshots["fit_formula"] = cmd[-1]
                    snapshots["fit_rows"] = _RUNNER.read_csv_rows(fit_input)
                    Path(cmd[cmd.index("--out") + 1]).write_text("{}", encoding="utf-8")
                    return 0, "", ""
                if cmd[1] == "predict":
                    input_path = Path(cmd[3])
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    input_rows = _RUNNER.read_csv_rows(input_path)
                    snapshots.setdefault("predict_inputs", []).append(input_rows)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w", encoding="utf-8", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["risk_score"])
                        writer.writeheader()
                        for idx, _row in enumerate(input_rows):
                            writer.writerow({"risk_score": float(idx)})
                    return 0, "", ""
                raise AssertionError(f"unexpected command: {cmd}")

            _RUNNER.run_cmd_stream = _fake_run_cmd
            _RUNNER.survival_metrics = lambda *args, **kwargs: {"c_index": 0.5}

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
            _RUNNER.survival_metrics = orig_survival_metrics

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
        for rows in predict_inputs:
            self.assertTrue(all(float(row["__entry"]) == 0.0 for row in rows))
            self.assertTrue(all(float(row["time"]) == 7.0 for row in rows))

    def test_survival_formula_rhs_supports_linkwiggle_and_timewiggle(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="surv_variant",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="ps",
            smooth_kind="separate",
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
