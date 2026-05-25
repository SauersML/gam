import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FUZZ_PATH = _REPO_ROOT / "bench" / "fuzz_vs_mgcv.py"
_SPEC = importlib.util.spec_from_file_location("bench_fuzz_vs_mgcv", _FUZZ_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load fuzz benchmark module from {_FUZZ_PATH}")
_FUZZ: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FUZZ
_SPEC.loader.exec_module(_FUZZ)


def _scenario(**overrides: Any) -> Any:
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
        sc = _scenario(basis_type="duchon", knots=10, double_penalty=False, duchon_order=0, duchon_power=2)

        noise_terms = _FUZZ.rust_noise_terms(["x0", "x1", "x2"], sc)

        self.assertEqual(
            noise_terms,
            "duchon(x0, x1, centers=5, order=0, power=2)",
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
        seeds = list(range(1, 60))
        cap = 75_000.0
        scenarios, skipped = _FUZZ.select_scenarios(seeds, max_scenario_cost=cap)

        for sc in scenarios:
            self.assertLessEqual(_FUZZ.estimate_scenario_cost(sc), cap)
        for sc, cost in skipped:
            self.assertGreater(cost, cap)
        self.assertGreater(len(scenarios), 0)
        self.assertGreater(len(skipped), 0)

    def test_backfilled_selection_preserves_requested_ci_coverage_under_cap(self) -> None:
        scenarios, skipped = _FUZZ.select_scenarios_backfilled(
            seed_start=42,
            target_count=100,
            excluded_seeds=set(),
            max_scenario_cost=200_000.0,
        )

        self.assertEqual(len(scenarios), 100)
        self.assertGreater(len(skipped), 0)
        self.assertTrue(
            all(_FUZZ.estimate_scenario_cost(sc) <= 200_000.0 for sc in scenarios)
        )

    def test_mgcv_ps_formula_converts_rust_internal_knots_to_basis_width(self) -> None:
        sc = _scenario(model_type="gam", basis_type="ps", knots=3)

        self.assertEqual(
            _FUZZ.mgcv_formula(["x0", "x1"], sc),
            "y ~ s(x0, bs='ps', k=min(7, nrow(train_df)-1)) + s(x1, bs='ps', k=min(7, nrow(train_df)-1))",
        )
        self.assertEqual(
            _FUZZ.mgcv_sigma_formula(["x0", "x1"], sc),
            "~ s(x0, bs='ps', k=min(7, nrow(train_df)-1))",
        )

    def test_duchon_generation_disables_unmatched_mgcv_select_penalty(self) -> None:
        sc = _FUZZ.generate_scenario(126)

        self.assertEqual(sc.basis_type, "duchon")
        self.assertFalse(sc.double_penalty)

    def test_forced_duchon_filter_disables_unmatched_mgcv_select_penalty(self) -> None:
        sc = _scenario(basis_type="ps", double_penalty=True, knots=8, n_smooths=3, n_obs=40)

        _FUZZ._apply_basis_filter(sc, "duchon")

        self.assertEqual(sc.basis_type, "duchon")
        self.assertFalse(sc.double_penalty)

    def test_mgcv_select_penalty_matches_rust_double_penalty_semantics(self) -> None:
        cases = [
            ("tps", False, False),
            ("tps", True, True),
            ("duchon", True, False),
            ("ps", False, False),
            ("ps", True, True),
        ]
        for basis_type, double_penalty, expected in cases:
            with self.subTest(basis_type=basis_type, double_penalty=double_penalty):
                self.assertIs(
                    _FUZZ._mgcv_select_penalty(
                        _scenario(
                            basis_type=basis_type,
                            double_penalty=double_penalty,
                        )
                    ),
                    expected,
                )

    def test_duchon_extra_terms_raise_estimated_cost(self) -> None:
        small = SimpleNamespace(
            seed=0, family="gaussian", model_type="gam",
            n_obs=200, n_smooths=3, knots=8, double_penalty=False,
            noise_sd=1.0, noise_kind="gaussian", smooth_kinds=["sine"],
            x_distribution="uniform", basis_type="duchon",
            collinear_strength=0.0, signal_structure="additive",
            sigma_kind="constant", duchon_order=0, duchon_power=2,
            n_duchon_dims=2,
        )
        large = SimpleNamespace(
            seed=0, family="gaussian", model_type="gam",
            n_obs=5000, n_smooths=10, knots=20, double_penalty=True,
            noise_sd=1.0, noise_kind="gaussian", smooth_kinds=["sine"],
            x_distribution="uniform", basis_type="duchon",
            collinear_strength=0.0, signal_structure="additive",
            sigma_kind="constant", duchon_order=0, duchon_power=2,
            n_duchon_dims=2,
        )
        self.assertGreater(_FUZZ.estimate_scenario_cost(large), 75_000)
        self.assertLess(_FUZZ.estimate_scenario_cost(small), 75_000)


def _trial(
    *,
    family: str = "gaussian",
    model_type: str = "gam",
    basis: str = "ps",
    rust_metric: float | None = None,
    mgcv_metric: float | None = None,
    seed: int = 0,
) -> Any:
    primary_metric = "r2" if family == "gaussian" else "auc"
    scenario = {
        "seed": seed,
        "family": family,
        "model_type": model_type,
        "basis_type": basis,
        "n_obs": 200,
        "n_smooths": 2,
        "knots": 8,
    }
    fr = _FUZZ.FuzzResult(
        scenario=scenario,
        rust={primary_metric: rust_metric},
        mgcv={primary_metric: mgcv_metric},
    )
    fr.compute_gap()
    return fr


def _gate_names(gates: dict[str, Any]) -> set[str]:
    return {failure["gate"] for failure in gates["gate_failures"]}


class FuzzCiGateTests(unittest.TestCase):
    def test_clean_run_passes(self) -> None:
        results = [
            _trial(rust_metric=0.80, mgcv_metric=0.80, seed=i)
            for i in range(100)
        ]
        gates = _FUZZ.compute_ci_gates(results, requested_trials=100)
        self.assertFalse(gates["failed"], gates)

    def test_per_trial_huge_gap_fails(self) -> None:
        results = [
            _trial(rust_metric=0.80, mgcv_metric=0.81, seed=i)
            for i in range(100)
        ]
        results[7] = _trial(rust_metric=0.05, mgcv_metric=0.79, seed=7)
        gates = _FUZZ.compute_ci_gates(results, requested_trials=100)
        self.assertTrue(gates["failed"])
        self.assertIn("per_trial_gap", _gate_names(gates))

    def test_cohort_median_failure_fires(self) -> None:
        results = [
            _trial(
                family="gaussian", model_type="gam", basis="duchon",
                rust_metric=0.50, mgcv_metric=0.75, seed=1000 + i,
            )
            for i in range(20)
        ] + [
            _trial(
                family="gaussian", model_type="gam", basis="ps",
                rust_metric=0.80, mgcv_metric=0.81, seed=2000 + i,
            )
            for i in range(20)
        ]
        gates = _FUZZ.compute_ci_gates(results, requested_trials=40)
        self.assertTrue(gates["failed"])
        cohort_failures = [
            gf for gf in gates["gate_failures"]
            if gf["gate"] == "cohort_median"
        ]
        self.assertEqual(len(cohort_failures), 1)
        offenders = cohort_failures[0]["offenders"]
        self.assertEqual(len(offenders), 1)
        self.assertEqual(
            offenders[0]["cohort"], ("gaussian", "gam", "duchon")
        )

    def test_cohort_net_wins_failure_fires(self) -> None:
        results = [
            _trial(
                family="binomial", model_type="gam", basis="tps",
                rust_metric=0.60, mgcv_metric=0.66, seed=3000 + i,
            )
            for i in range(10)
        ]
        gates = _FUZZ.compute_ci_gates(results, requested_trials=10)
        self.assertTrue(gates["failed"])
        self.assertIn("cohort_net_wins", _gate_names(gates))

    def test_rust_nan_inf_failure_fires(self) -> None:
        results = [
            _trial(rust_metric=0.80, mgcv_metric=0.81, seed=i)
            for i in range(100)
        ]
        results[5] = _trial(
            rust_metric=float("nan"), mgcv_metric=0.55, seed=5,
        )
        gates = _FUZZ.compute_ci_gates(results, requested_trials=100)
        self.assertTrue(gates["failed"])
        self.assertIn("rust_nan_inf", _gate_names(gates))

    def test_coverage_failure_fires_when_too_many_skipped(self) -> None:
        results = [
            _trial(rust_metric=0.80, mgcv_metric=0.81, seed=i)
            for i in range(50)
        ]
        gates = _FUZZ.compute_ci_gates(
            results, requested_trials=100, skipped_count=50,
        )
        self.assertTrue(gates["failed"])
        self.assertIn("coverage", _gate_names(gates))

    def test_baseline_regression_fires(self) -> None:
        results = [
            _trial(
                family="gaussian", model_type="gam", basis="ps",
                rust_metric=0.70, mgcv_metric=0.80, seed=4000 + i,
            )
            for i in range(20)
        ]
        baseline = {
            "threshold": 0.05,
            "cohorts": {"gaussian/gam/ps": 0.02},
        }
        gates = _FUZZ.compute_ci_gates(
            results, requested_trials=20, baseline=baseline,
        )
        self.assertTrue(gates["failed"])
        self.assertIn("baseline_regression", _gate_names(gates))


if __name__ == "__main__":
    unittest.main()
