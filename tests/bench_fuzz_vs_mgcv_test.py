import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FUZZ_PATH = _REPO_ROOT / "bench" / "fuzz_vs_mgcv.py"
_SPEC = importlib.util.spec_from_file_location("bench_fuzz_vs_mgcv", _FUZZ_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load fuzz benchmark module from {_FUZZ_PATH}")
_FUZZ = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FUZZ
_SPEC.loader.exec_module(_FUZZ)


def _scenario(**overrides):
    base = {
        "double_penalty": True,
        "basis_type": "ps",
        "knots": 8,
        "duchon_order": 0,
        "duchon_power": 1,
        "n_duchon_dims": 2,
        "model_type": "gamlss",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FuzzVsMgcvFormulaTests(unittest.TestCase):
    def test_gamlss_fit_command_uses_rhs_only_predict_noise(self) -> None:
        sc = _scenario()
        cmd = _FUZZ.build_rust_fit_cmd(sc, "train.csv", "model.json", ["x0", "x1"])

        self.assertIn("--predict-noise", cmd)
        noise_terms = cmd[cmd.index("--predict-noise") + 1]
        self.assertEqual(noise_terms, "s(x0, type=ps, knots=4, double_penalty=true)")
        self.assertNotIn("~", noise_terms)
        self.assertEqual(cmd[-1], "y ~ s(x0, type=ps, knots=8, double_penalty=true) + s(x1, type=ps, knots=8, double_penalty=true)")

    def test_duchon_noise_terms_stay_rhs_only(self) -> None:
        sc = _scenario(basis_type="duchon", knots=10, double_penalty=False, duchon_order=1, duchon_power=2)

        noise_terms = _FUZZ.rust_noise_terms(["x0", "x1", "x2"], sc)

        self.assertEqual(
            noise_terms,
            "duchon(x0, x1, centers=5, order=1, power=2, length_scale=1, double_penalty=false)",
        )
        self.assertNotIn("~", noise_terms)

    def test_gam_fit_command_omits_predict_noise(self) -> None:
        sc = _scenario(model_type="gam", basis_type="tps", knots=6)

        cmd = _FUZZ.build_rust_fit_cmd(sc, "train.csv", "model.json", ["x0", "x1"])

        self.assertNotIn("--predict-noise", cmd)
        self.assertEqual(
            cmd[-1],
            "y ~ s(x0, type=tps, centers=6, double_penalty=true) + s(x1, type=tps, centers=6, double_penalty=true)",
        )

    def test_select_scenarios_applies_cost_cap_before_sorting(self) -> None:
        scenarios, skipped = _FUZZ.select_scenarios([56, 73, 130], max_scenario_cost=75_000)

        self.assertEqual([sc.seed for sc in scenarios], [73])
        self.assertEqual([sc.seed for sc, _ in skipped], [56, 130])
        self.assertGreater(skipped[0][1], 75_000)


if __name__ == "__main__":
    unittest.main()
