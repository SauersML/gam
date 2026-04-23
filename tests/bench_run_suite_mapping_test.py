import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_SUITE_PATH = _REPO_ROOT / "bench" / "run_suite.py"
_SPEC = importlib.util.spec_from_file_location("bench_run_suite", _RUN_SUITE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load benchmark runner from {_RUN_SUITE_PATH}")
_RUN_SUITE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_SUITE)
_GITIGNORE_PATH = _REPO_ROOT / ".gitignore"
_BENCH_DATASET_DIR = _REPO_ROOT / "bench" / "datasets"
_REQUIRED_BENCHMARK_DATASETS = {
    "31_day.csv",
    "bone.csv",
    "cirrhosis.csv",
    "five_day.csv",
    "haberman.csv",
    "heart_failure_clinical_records_dataset.csv",
    "horse.csv",
    "icu_survival_death.csv",
    "icu_survival_los.csv",
    "lidar.csv",
    "prostate.csv",
    "wine.csv",
}


class RunSuiteMappingTests(unittest.TestCase):
    def test_finalize_cv_result_keeps_evaluation_from_fold_count(self) -> None:
        result = _RUN_SUITE._finalize_cv_result(
            contender="rust_gam",
            scenario_name="wine_temp_vs_year",
            family="gaussian",
            cv_rows=[
                {
                    "fit_sec": 0.1,
                    "predict_sec": 0.01,
                    "logloss": 1.0,
                    "mse": 0.25,
                    "rmse": 0.5,
                    "mae": 0.4,
                    "r2": 0.2,
                    "n_test": 10,
                }
                for _ in range(5)
            ],
            plot_payload=None,
            model_spec="s_temp ~ s(year, type=ps, knots=7) via release binary [5-fold CV]",
        )
        self.assertEqual(result["evaluation"], "5-fold CV")

    def test_finalize_cv_result_rejects_reserved_metric_keys(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reserved result keys: evaluation"):
            _RUN_SUITE._finalize_cv_result(
                contender="rust_gam",
                scenario_name="wine_temp_vs_year",
                family="gaussian",
                cv_rows=[
                    {
                        "fit_sec": 0.1,
                        "predict_sec": 0.01,
                        "logloss": 1.0,
                        "mse": 0.25,
                        "rmse": 0.5,
                        "mae": 0.4,
                        "r2": 0.2,
                        "n_test": 10,
                    }
                ],
                plot_payload=None,
                model_spec="s_temp ~ s(year, type=ps, knots=7) via release binary [holdout]",
                extra_metrics={"evaluation": "broken"},
            )

    def test_validate_result_metadata_accepts_matching_cv_spec(self) -> None:
        _RUN_SUITE._validate_result_metadata(
            [
                {
                    "status": "ok",
                    "contender": "rust_gam",
                    "scenario_name": "lidar_semipar",
                    "evaluation": "5-fold CV",
                    "model_spec": "logratio ~ s(range, type=ps, knots=24) via release binary [5-fold CV]",
                }
            ]
        )

    def test_validate_result_metadata_rejects_missing_evaluation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "model result metadata/spec mismatch for rust_gam / lidar_semipar"):
            _RUN_SUITE._validate_result_metadata(
                [
                    {
                        "status": "ok",
                        "contender": "rust_gam",
                        "scenario_name": "lidar_semipar",
                        "evaluation": None,
                        "model_spec": "logratio ~ s(range, type=ps, knots=24) via release binary [5-fold CV]",
                    }
                ]
            )

    def test_r_gamlss_sigma_formula_rejects_constant_sigma(self) -> None:
        ds = {
            "rows": [{"y": 1.0}],
            "features": [],
            "target": "y",
            "family": "gaussian",
        }
        with self.assertRaisesRegex(RuntimeError, "requires a non-constant sigma model"):
            _RUN_SUITE._sigma_feature_formula(ds, scenario_name="toy", backend="r_gamlss")

    def test_run_external_r_gamlss_cv_supports_binomial_family(self) -> None:
        scenario = {"name": "small_dense"}
        ds = {
            "family": "binomial",
            "rows": [
                {"x1": 0.0, "x2": 0.0, "y": 0.0},
                {"x1": 1.0, "x2": 1.0, "y": 1.0},
            ],
            "features": ["x1", "x2"],
            "target": "y",
        }
        folds = [SimpleNamespace(train_idx=[0], test_idx=[1])]
        seen_scripts: list[str] = []
        orig_run_cmd = _RUN_SUITE.run_cmd
        orig_tempdir = _RUN_SUITE._workspace_tempdir
        try:
            def _fake_run_cmd(cmd, cwd=None):
                if cmd[0] == "Rscript":
                    script = Path(cmd[1]).read_text()
                    seen_scripts.append(script)
                    out_path = Path(cmd[3])
                    out_path.write_text(
                        json.dumps(
                            {
                                "status": "ok",
                                "fit_sec": 0.1,
                                "predict_sec": 0.01,
                                "pred": [0.8],
                                "sigma": [1.2],
                                "model_spec": "gamlss(BI; sigma.formula=~ pb(x1) + pb(x2)): y ~ pb(x1)",
                            }
                        )
                    )
                    return 0, "", ""
                return orig_run_cmd(cmd, cwd=cwd)

            _RUN_SUITE.run_cmd = _fake_run_cmd
            with tempfile.TemporaryDirectory() as td:
                temp_root = Path(td)
                _RUN_SUITE._workspace_tempdir = lambda prefix="": tempfile.TemporaryDirectory(
                    prefix=prefix, dir=temp_root
                )
                result = _RUN_SUITE.run_external_r_gamlss_cv(scenario, ds=ds, folds=folds)
        finally:
            _RUN_SUITE.run_cmd = orig_run_cmd
            _RUN_SUITE._workspace_tempdir = orig_tempdir

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["family"], "binomial")
        self.assertTrue(any("family = fit_family" in script for script in seen_scripts))
        self.assertTrue(any("BI()" in script for script in seen_scripts))

    def test_required_benchmark_datasets_exist(self) -> None:
        missing = sorted(
            dataset_name
            for dataset_name in _REQUIRED_BENCHMARK_DATASETS
            if not (_BENCH_DATASET_DIR / dataset_name).exists()
        )
        self.assertEqual(missing, [])

    def test_checked_in_benchmark_datasets_are_not_gitignored(self) -> None:
        ignored_entries = {
            line.strip()
            for line in _GITIGNORE_PATH.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        ignored_required = sorted(
            f"bench/datasets/{dataset_name}"
            for dataset_name in _REQUIRED_BENCHMARK_DATASETS
            if f"bench/datasets/{dataset_name}" in ignored_entries
        )
        self.assertEqual(ignored_required, [])

    def assert_joint_mapping(self, scenario_name: str, expected_dim: int, expected_knots: int) -> None:
        cfg = _RUN_SUITE._scenario_fit_mapping(scenario_name)
        self.assertIsNotNone(cfg, scenario_name)
        self.assertEqual(cfg["smooth_basis"], "thinplate")
        self.assertEqual(len(cfg["smooth_cols"]), expected_dim)
        self.assertEqual(cfg["knots"], expected_knots)

    def test_geo_disease_eas_tp_keeps_three_dimensional_joint_embedding(self) -> None:
        self.assert_joint_mapping("geo_disease_eas_tp_k6", expected_dim=3, expected_knots=6)
        self.assert_joint_mapping("geo_disease_eas_tp_k12", expected_dim=3, expected_knots=12)
        self.assert_joint_mapping("geo_disease_eas_tp_k24", expected_dim=3, expected_knots=24)

    def test_geo_latlon_tp_keeps_two_dimensional_joint_embedding(self) -> None:
        self.assert_joint_mapping("geo_latlon_equatornoise_tp_k12", expected_dim=2, expected_knots=12)
        self.assert_joint_mapping("geo_latlon_superpopnoise_tp_k24", expected_dim=2, expected_knots=24)

    def test_papuan_and_subpop_tp_keep_fixed_joint_embedding(self) -> None:
        self.assert_joint_mapping("papuan_oce4_tp_k12", expected_dim=3, expected_knots=12)
        self.assert_joint_mapping("geo_subpop16_tp_k24", expected_dim=3, expected_knots=24)

    def test_geo_subpop16_dataset_builds_without_external_pc_file(self) -> None:
        ds = _RUN_SUITE.dataset_for_scenario({"name": "geo_subpop16_tp_k6"})
        self.assertEqual(ds["family"], "binomial")
        self.assertEqual(ds["features"], [f"pc{i}" for i in range(1, 17)])
        self.assertGreater(len(ds["rows"]), 0)

    def test_geo_latlon_dataset_builds_without_external_pc_file(self) -> None:
        ds = _RUN_SUITE.dataset_for_scenario({"name": "geo_latlon_superpopnoise_tp_k12"})
        self.assertEqual(ds["family"], "binomial")
        self.assertEqual(ds["features"], [f"pc{i}" for i in range(1, 7)])
        self.assertGreater(len(ds["rows"]), 0)

    def test_legacy_geo_disease_tp_scenarios_use_fixed_joint_embedding(self) -> None:
        for scenario_name in ("geo_disease_tp", "geo_disease_shrinkage"):
            cfg = _RUN_SUITE._scenario_fit_mapping(scenario_name)
            self.assertEqual(cfg["smooth_basis"], "thinplate")
            self.assertEqual(cfg["smooth_cols"], ["pc1", "pc2", "pc3"])
            self.assertEqual(cfg["knots"], 12)

    def test_geo_subpop16_marginal_slope_aniso_keeps_16d_duchon_mapping(self) -> None:
        cfg = _RUN_SUITE._scenario_fit_mapping("geo_subpop16_margslope_aniso_duchon16d_k50")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["family"], "binomial-logit")
        self.assertEqual(cfg["smooth_basis"], "duchon")
        self.assertEqual(cfg["smooth_cols"], [f"pc{i}" for i in range(1, 17)])
        self.assertEqual(cfg["knots"], 50)
        self.assertTrue(cfg.get("scale_dimensions"))

    def test_geo_subpop16_marginal_slope_aniso_lane_is_present_and_enabled(self) -> None:
        scenarios = json.loads((_REPO_ROOT / "bench" / "scenarios.json").read_text())["scenarios"]
        scenario = next(
            s for s in scenarios if s["name"] == "geo_subpop16_margslope_aniso_duchon16d_k50"
        )
        self.assertTrue(
            _RUN_SUITE._is_contender_enabled(scenario, "rust_gamlss_marginal_slope")
        )

    def test_thread3_adaptive_reml_uses_current_boolean_cli(self) -> None:
        seen = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _fake_run_rust_scenario_cv(*args, **kwargs):
                if kwargs.get("contender_name") == "rust_thread3_adaptive_reml":
                    seen.append(list(kwargs.get("rust_fit_extra_args") or []))
                return {
                    "status": "ok",
                    "scenario_name": "thread3_admixture_cliff",
                    "contender": kwargs.get("contender_name", "rust_gam"),
                    "model_spec": "5-fold CV",
                    "evaluation": "cv",
                }

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                scenario_path.write_text(json.dumps({"scenarios": [{"name": "thread3_admixture_cliff"}]}))

                _RUN_SUITE.run_rust_scenario_cv = _fake_run_rust_scenario_cv
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"y": 0.0}],
                    "features": ["pc1"],
                    "target": "y",
                    "family": "binomial",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                _RUN_SUITE._is_contender_enabled = (
                    lambda _scenario, contender: contender == "rust_thread3_adaptive_reml"
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0], ["--adaptive-regularization", "true"])

    def test_thread3_adaptive_flexible_passes_flexible_formula_link(self) -> None:
        seen_formula_links = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_datapoint_figures = _RUN_SUITE.generate_scenario_datapoint_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _fake_run_rust_scenario_cv(*args, **kwargs):
                if kwargs.get("contender_name") == "rust_thread3_adaptive_reml_flexible":
                    seen_formula_links.append(kwargs.get("formula_link"))
                return {
                    "status": "ok",
                    "scenario_name": "thread3_admixture_cliff",
                    "contender": kwargs.get("contender_name", "rust_gam"),
                    "model_spec": "5-fold CV",
                    "evaluation": "5-fold CV",
                }

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                scenario_path.write_text(json.dumps({"scenarios": [{"name": "thread3_admixture_cliff"}]}))

                _RUN_SUITE.run_rust_scenario_cv = _fake_run_rust_scenario_cv
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"y": 0.0, "pc1": 0.0}],
                    "features": ["pc1"],
                    "target": "y",
                    "family": "binomial",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                _RUN_SUITE._is_contender_enabled = lambda _scenario, contender: contender in {
                    "rust_gam",
                    "rust_thread3_adaptive_reml_flexible",
                }
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.generate_scenario_datapoint_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.generate_scenario_datapoint_figures = orig_datapoint_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertEqual(
            seen_formula_links,
            [
                _RUN_SUITE._flexible_link_name(
                    _RUN_SUITE._default_rust_formula_link_for_family("binomial")
                )
            ],
        )

    def test_survival_benchmark_fit_options_require_structural_ispline_basis(self) -> None:
        expected = {
            "icu_survival_death": 10,
            "icu_survival_los": 10,
            "heart_failure_survival": 8,
            "cirrhosis_survival": 8,
        }
        for scenario_name, expected_knots in expected.items():
            with self.subTest(scenario_name=scenario_name):
                cfg = _RUN_SUITE._rust_survival_fit_options_for_scenario(scenario_name)
                self.assertEqual(cfg["time_basis"], "ispline")
                self.assertEqual(cfg["time_degree"], 3)
                self.assertEqual(cfg["time_num_internal_knots"], expected_knots)
                self.assertGreaterEqual(cfg["time_smooth_lambda"], 0.0)

    def test_survival_benchmark_cli_args_emit_ispline(self) -> None:
        args = _RUN_SUITE._rust_survival_fit_cli_args("icu_survival_death")
        self.assertIn("--time-basis", args)
        idx = args.index("--time-basis")
        self.assertEqual(args[idx + 1], "ispline")

    def test_run_rust_scenario_cv_rejects_survival_misuse(self) -> None:
        scenario = {"name": "heart_failure_survival"}
        ds = _RUN_SUITE.dataset_for_scenario(scenario)
        with self.assertRaisesRegex(RuntimeError, "run_rust_gamlss_survival_cv"):
            _RUN_SUITE.run_rust_scenario_cv(scenario, ds=ds, folds=[])

    def test_run_rust_scenario_cv_emits_evaluation_metadata(self) -> None:
        scenario = {"name": "lidar_semipar"}
        ds = {
            "family": "gaussian",
            "rows": [
                {"range": 0.0, "logratio": 0.0},
                {"range": 1.0, "logratio": 1.0},
                {"range": 2.0, "logratio": 2.0},
                {"range": 3.0, "logratio": 3.0},
            ],
            "features": ["range"],
            "target": "logratio",
        }
        folds = [
            SimpleNamespace(train_idx=[0, 1], test_idx=[2, 3]),
            SimpleNamespace(train_idx=[2, 3], test_idx=[0, 1]),
        ]
        orig_ensure_rust_binary = _RUN_SUITE._ensure_rust_binary
        orig_run_cmd = _RUN_SUITE.run_cmd
        orig_tempdir = _RUN_SUITE._workspace_tempdir
        orig_formula = _RUN_SUITE._rust_formula_for_scenario
        try:
            _RUN_SUITE._ensure_rust_binary = lambda: Path("/tmp/fake-rust-gam")
            _RUN_SUITE._rust_formula_for_scenario = lambda *_args, **_kwargs: (
                "gaussian",
                "logratio ~ s(range, type=ps, knots=24)",
            )

            def _fake_run_cmd(cmd, cwd=None):
                if len(cmd) >= 2 and cmd[1] == "fit":
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    out_path.write_text(json.dumps({"fit_result": {"standard_deviation": 1.25}}))
                    return 0, "", ""
                if len(cmd) >= 2 and cmd[1] == "predict":
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    out_path.write_text("mean\n1.5\n1.5\n")
                    return 0, "", ""
                return 1, "", f"unexpected command: {cmd}"

            _RUN_SUITE.run_cmd = _fake_run_cmd

            with tempfile.TemporaryDirectory() as td:
                temp_root = Path(td)
                _RUN_SUITE._workspace_tempdir = lambda prefix="": tempfile.TemporaryDirectory(
                    prefix=prefix, dir=temp_root
                )
                result = _RUN_SUITE.run_rust_scenario_cv(scenario, ds=ds, folds=folds)
        finally:
            _RUN_SUITE._ensure_rust_binary = orig_ensure_rust_binary
            _RUN_SUITE.run_cmd = orig_run_cmd
            _RUN_SUITE._workspace_tempdir = orig_tempdir
            _RUN_SUITE._rust_formula_for_scenario = orig_formula

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evaluation"], "2-fold CV")
        self.assertEqual(result["n_folds"], 2)
        self.assertIn("[2-fold CV]", result["model_spec"])

    def test_main_does_not_schedule_rust_gam_for_survival_scenarios(self) -> None:
        seen = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_run_gamlss_survival = _RUN_SUITE.run_rust_gamlss_survival_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _fake_run_rust_scenario_cv(*args, **kwargs):
                seen.append(("rust_gam", kwargs.get("contender_name", "rust_gam")))
                return {
                    "status": "ok",
                    "scenario_name": "heart_failure_survival",
                    "contender": "rust_gam",
                    "model_spec": "5-fold CV",
                    "evaluation": "cv",
                }

            def _fake_run_rust_gamlss_survival_cv(*args, **kwargs):
                seen.append(("survival", "rust_gamlss_survival"))
                return {
                    "status": "ok",
                    "scenario_name": "heart_failure_survival",
                    "contender": "rust_gamlss_survival",
                    "model_spec": "5-fold CV",
                    "evaluation": "cv",
                }

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                scenario_path.write_text(json.dumps({"scenarios": [{"name": "heart_failure_survival"}]}))

                _RUN_SUITE.run_rust_scenario_cv = _fake_run_rust_scenario_cv
                _RUN_SUITE.run_rust_gamlss_survival_cv = _fake_run_rust_gamlss_survival_cv
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"time": 1.0, "event": 1.0, "x": 0.0}],
                    "features": ["x"],
                    "family": "survival",
                    "time_col": "time",
                    "event_col": "event",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                _RUN_SUITE._is_contender_enabled = (
                    lambda _scenario, contender: contender == "rust_gamlss_survival"
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.run_rust_gamlss_survival_cv = orig_run_gamlss_survival
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertNotIn(("rust_gam", "rust_gam"), seen)
        self.assertIn(("survival", "rust_gamlss_survival"), seen)

    def test_main_skips_flexible_variants_for_gaussian_scenarios(self) -> None:
        seen_rust = []
        seen_gamlss = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_run_gamlss = _RUN_SUITE.run_rust_gamlss_scenario_cv
        orig_run_gamlss_survival = _RUN_SUITE.run_rust_gamlss_survival_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _fake_run_rust_scenario_cv(*args, **kwargs):
                seen_rust.append(kwargs.get("contender_name", "rust_gam"))
                return {
                    "status": "ok",
                    "scenario_name": "lidar_semipar",
                    "contender": kwargs.get("contender_name", "rust_gam"),
                    "model_spec": "cv",
                    "evaluation": "cv",
                }

            def _fake_run_rust_gamlss_scenario_cv(*args, **kwargs):
                seen_gamlss.append(kwargs.get("contender_name", "rust_gamlss"))
                return {
                    "status": "ok",
                    "scenario_name": "lidar_semipar",
                    "contender": kwargs.get("contender_name", "rust_gamlss"),
                    "model_spec": "cv",
                    "evaluation": "cv",
                }

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                scenario_path.write_text(json.dumps({"scenarios": [{"name": "lidar_semipar"}]}))

                _RUN_SUITE.run_rust_scenario_cv = _fake_run_rust_scenario_cv
                _RUN_SUITE.run_rust_gamlss_scenario_cv = _fake_run_rust_gamlss_scenario_cv
                _RUN_SUITE.run_rust_gamlss_survival_cv = lambda *args, **kwargs: None
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"y": 0.0, "range": 0.0}],
                    "features": ["range"],
                    "target": "y",
                    "family": "gaussian",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                _RUN_SUITE._is_contender_enabled = (
                    lambda _scenario, contender: contender in {"rust_gam", "rust_gamlss"}
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.run_rust_gamlss_scenario_cv = orig_run_gamlss
            _RUN_SUITE.run_rust_gamlss_survival_cv = orig_run_gamlss_survival
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertIn("rust_gam", seen_rust)
        self.assertNotIn("rust_gam_flexible", seen_rust)
        self.assertIn("rust_gamlss", seen_gamlss)
        self.assertNotIn("rust_gamlss_flexible", seen_gamlss)

    def test_main_skips_flexible_variants_for_survival_scenarios(self) -> None:
        seen_survival = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_run_gamlss = _RUN_SUITE.run_rust_gamlss_scenario_cv
        orig_run_gamlss_survival = _RUN_SUITE.run_rust_gamlss_survival_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _fake_run_rust_gamlss_survival_cv(*args, **kwargs):
                seen_survival.append(kwargs.get("contender_name", "rust_gamlss_survival"))
                return {
                    "status": "ok",
                    "scenario_name": "heart_failure_survival",
                    "contender": kwargs.get("contender_name", "rust_gamlss_survival"),
                    "model_spec": "cv",
                    "evaluation": "cv",
                }

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                scenario_path.write_text(json.dumps({"scenarios": [{"name": "heart_failure_survival"}]}))

                _RUN_SUITE.run_rust_scenario_cv = lambda *args, **kwargs: {
                    "status": "ok",
                    "scenario_name": "heart_failure_survival",
                    "contender": kwargs.get("contender_name", "rust_gam"),
                    "model_spec": "cv",
                    "evaluation": "cv",
                }
                _RUN_SUITE.run_rust_gamlss_scenario_cv = lambda *args, **kwargs: None
                _RUN_SUITE.run_rust_gamlss_survival_cv = _fake_run_rust_gamlss_survival_cv
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"time": 1.0, "event": 1.0, "x": 0.0}],
                    "features": ["x"],
                    "time_col": "time",
                    "event_col": "event",
                    "family": "survival",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                _RUN_SUITE._is_contender_enabled = (
                    lambda _scenario, contender: contender in {"rust_gam_sas", "rust_gamlss_survival"}
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.run_rust_gamlss_scenario_cv = orig_run_gamlss
            _RUN_SUITE.run_rust_gamlss_survival_cv = orig_run_gamlss_survival
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertEqual(seen_survival, ["rust_gamlss_survival"])

    def test_main_does_not_require_excluded_core_rust_contenders(self) -> None:
        seen = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_run_gamlss = _RUN_SUITE.run_rust_gamlss_scenario_cv
        orig_run_gamlss_marginal_slope = _RUN_SUITE.run_rust_gamlss_marginal_slope_cv
        orig_run_gamlss_survival = _RUN_SUITE.run_rust_gamlss_survival_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_datapoint_figures = _RUN_SUITE.generate_scenario_datapoint_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _unexpected_runner(*args, **kwargs):
                contender = kwargs.get("contender_name", "unknown")
                seen.append(contender)
                raise AssertionError(f"excluded contender scheduled: {contender}")

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                exclude_contenders = [
                    "rust_gam",
                    "rust_gam_flexible",
                    "rust_gamlss",
                    "rust_gamlss_flexible",
                    "rust_gamlss_marginal_slope",
                    "rust_gamlss_marginal_slope_aniso",
                    "r_gamlss",
                    "r_mgcv",
                    "r_mgcv_gaulss",
                    "r_gamboostlss",
                    "r_bamlss",
                    "r_brms",
                ]
                scenario_path.write_text(
                    json.dumps(
                        {
                            "scenarios": [
                                {
                                    "name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                                    "exclude_contenders": exclude_contenders,
                                }
                            ]
                        }
                    )
                )

                _RUN_SUITE.run_rust_scenario_cv = _unexpected_runner
                _RUN_SUITE.run_rust_gamlss_scenario_cv = _unexpected_runner
                _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = _unexpected_runner
                _RUN_SUITE.run_rust_gamlss_survival_cv = _unexpected_runner
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"y": 0.0, "pc1": 0.0}],
                    "features": ["pc1"],
                    "target": "y",
                    "family": "binomial",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                excluded = set(exclude_contenders)
                _RUN_SUITE._is_contender_enabled = lambda _scenario, contender: contender not in (
                    excluded
                    | {
                        "rust_gamlss_survival",
                        "rust_gamlss_survival_marginal_slope",
                        "r_mgcv_coxph",
                        "python_sksurv_rsf",
                        "python_sksurv_coxnet",
                        "python_lifelines_coxph_enet",
                        "r_glmnet_cox",
                        "python_sksurv_gb_coxph",
                        "python_sksurv_componentwise_gb_coxph",
                        "python_lifelines_weibull_aft",
                        "python_lifelines_lognormal_aft",
                        "python_xgboost_aft",
                    }
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.generate_scenario_datapoint_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
                payload = json.loads(out_path.read_text())
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.run_rust_gamlss_scenario_cv = orig_run_gamlss
            _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = orig_run_gamlss_marginal_slope
            _RUN_SUITE.run_rust_gamlss_survival_cv = orig_run_gamlss_survival
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.generate_scenario_datapoint_figures = orig_datapoint_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertEqual(seen, [])
        self.assertEqual(payload["results"], [])

    def test_main_skips_excluded_aniso_marginal_slope_contender(self) -> None:
        seen = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_run_gamlss = _RUN_SUITE.run_rust_gamlss_scenario_cv
        orig_run_gamlss_marginal_slope = _RUN_SUITE.run_rust_gamlss_marginal_slope_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_datapoint_figures = _RUN_SUITE.generate_scenario_datapoint_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _fake_run_rust_gamlss_scenario_cv(*args, **kwargs):
                return {
                    "status": "ok",
                    "scenario_name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                    "contender": kwargs.get("contender_name", "rust_gamlss"),
                    "model_spec": "5-fold CV",
                    "evaluation": "5-fold CV",
                }

            def _unexpected_marginal_slope(*args, **kwargs):
                contender = kwargs.get("contender_name", "unknown")
                seen.append(contender)
                raise AssertionError(f"excluded contender scheduled: {contender}")

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                scenario_path.write_text(
                    json.dumps(
                        {
                            "scenarios": [
                                {
                                    "name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                                    "exclude_contenders": ["rust_gamlss_marginal_slope_aniso"],
                                }
                            ]
                        }
                    )
                )

                _RUN_SUITE.run_rust_scenario_cv = lambda *args, **kwargs: {
                    "status": "ok",
                    "scenario_name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                    "contender": kwargs.get("contender_name", "rust_gam"),
                    "model_spec": "5-fold CV",
                    "evaluation": "5-fold CV",
                }
                _RUN_SUITE.run_rust_gamlss_scenario_cv = _fake_run_rust_gamlss_scenario_cv
                _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = _unexpected_marginal_slope
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"y": 0.0, "pc1": 0.0}],
                    "features": ["pc1"],
                    "target": "y",
                    "family": "binomial",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                enabled = {"rust_gam", "rust_gamlss"}
                _RUN_SUITE._is_contender_enabled = (
                    lambda scenario, contender: contender in enabled
                    and contender not in set(scenario.get("exclude_contenders", []))
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.generate_scenario_datapoint_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.run_rust_gamlss_scenario_cv = orig_run_gamlss
            _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = orig_run_gamlss_marginal_slope
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.generate_scenario_datapoint_figures = orig_datapoint_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
