import importlib.util
import unittest
from pathlib import Path


_RUN_SUITE_PATH = Path("/Users/user/gam/bench/run_suite.py")
_SPEC = importlib.util.spec_from_file_location("bench_run_suite", _RUN_SUITE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load benchmark runner from {_RUN_SUITE_PATH}")
_RUN_SUITE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_SUITE)


class RunSuiteMappingTests(unittest.TestCase):
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

    def test_legacy_geo_disease_tp_scenarios_use_fixed_joint_embedding(self) -> None:
        for scenario_name in ("geo_disease_tp", "geo_disease_shrinkage"):
            cfg = _RUN_SUITE._rust_fit_mapping(scenario_name)
            self.assertEqual(cfg["smooth_basis"], "thinplate")
            self.assertEqual(cfg["smooth_cols"], ["pc1", "pc2", "pc3"])
            self.assertEqual(cfg["knots"], 12)


if __name__ == "__main__":
    unittest.main()
