import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


_RUN_SUITE_PATH = Path("/Users/user/gam/bench/run_suite.py")
_SPEC = importlib.util.spec_from_file_location("bench_run_suite", _RUN_SUITE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load benchmark runner from {_RUN_SUITE_PATH}")
_RUN_SUITE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_SUITE)
_REPO_ROOT = Path("/Users/user/gam")
_GITIGNORE_PATH = _REPO_ROOT / ".gitignore"


class RunSuiteMappingTests(unittest.TestCase):
    def test_checked_in_benchmark_datasets_are_not_gitignored(self) -> None:
        ignored_entries = {
            line.strip()
            for line in _GITIGNORE_PATH.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertNotIn("bench/datasets/icu_survival_death.csv", ignored_entries)
        self.assertNotIn("bench/datasets/icu_survival_los.csv", ignored_entries)

    def assert_joint_mapping(self, scenario_name: str, expected_dim: int, expected_knots: int) -> None:
        cfg = _RUN_SUITE._rust_fit_mapping(scenario_name)
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
            cfg = _RUN_SUITE._rust_fit_mapping(scenario_name)
            self.assertEqual(cfg["smooth_basis"], "thinplate")
            self.assertEqual(cfg["smooth_cols"], ["pc1", "pc2", "pc3"])
            self.assertEqual(cfg["knots"], 12)

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
                _RUN_SUITE._is_contender_enabled = lambda *_args, **_kwargs: False
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


if __name__ == "__main__":
    unittest.main()
