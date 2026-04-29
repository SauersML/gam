import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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

        # length_scale was dropped in commit 35e62344 to align rust pure
        # scale-free Duchon with mgcv bs='ds' (mgcv has no hybrid-mode analog).
        self.assertEqual(
            noise_terms,
            "duchon(x0, x1, centers=5, order=1, power=2, double_penalty=false)",
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
        # Use a wide seed sweep so we're guaranteed to land at least one
        # scenario above and one below the cost cap regardless of how the
        # scenario generator's random distributions evolve.
        seeds = list(range(1, 60))
        cap = 75_000.0
        scenarios, skipped = _FUZZ.select_scenarios(seeds, max_scenario_cost=cap)

        # Every retained scenario is at or below cap; every skipped scenario
        # is strictly above cap.
        for sc in scenarios:
            self.assertLessEqual(_FUZZ.estimate_scenario_cost(sc), cap)
        for sc, cost in skipped:
            self.assertGreater(cost, cap)
        # And the harness is producing some of each (otherwise the cap is
        # not actually doing any work in this seed range, which would be
        # a regression).
        self.assertGreater(len(scenarios), 0)
        self.assertGreater(len(skipped), 0)

    def test_duchon_extra_terms_raise_estimated_cost(self) -> None:
        # Build two synthetic Duchon scenarios at known sizes — one small
        # and cheap, one large and over-cap — instead of relying on the
        # random scenario generator producing specific shapes for fixed
        # seeds (which is fragile under generator changes).
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


def _trial(*, family="gaussian", model_type="gam", basis="ps",
           gap=None, rust_metric=None, mgcv_metric=None,
           rust=None, mgcv=None, seed=0):
    primary_metric = "r2" if family == "gaussian" else "auc"
    rv = rust_metric if rust_metric is not None else (None if gap is None else 0.5)
    mv = mgcv_metric if mgcv_metric is not None else (None if gap is None else (rv + gap if rv is not None else None))
    rust_dict = {primary_metric: rv}
    if rust:
        rust_dict.update(rust)
    mgcv_dict = {primary_metric: mv}
    if mgcv:
        mgcv_dict.update(mgcv)
    scenario = {
        "seed": seed,
        "family": family,
        "model_type": model_type,
        "basis_type": basis,
        "n_obs": 200,
        "n_smooths": 2,
        "knots": 8,
    }
    fr = _FUZZ.FuzzResult(scenario=scenario, rust=rust_dict, mgcv=mgcv_dict)
    fr.compute_gap()
    return fr


class FuzzCiGateTests(unittest.TestCase):
    def test_clean_run_passes(self) -> None:
        # True-tie metrics (gap=0) so all 100 trials land inside the
        # ±0.01 deadband regardless of FP-arithmetic noise; tests that
        # a green run does NOT fire any gate.
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
        # Inject one massive regression — gap = 0.74 like the real run.
        results[7] = _trial(rust_metric=0.05, mgcv_metric=0.79, seed=7)
        gates = _FUZZ.compute_ci_gates(results, requested_trials=100)
        self.assertTrue(gates["failed"])
        gate_names = [gf["gate"] for gf in gates["gate_failures"]]
        self.assertIn("per_trial_gap", gate_names)

    def test_cohort_median_failure_fires(self) -> None:
        # gaussian/gam/duchon cohort with median gap +0.25 like the real run.
        results = []
        for i in range(20):
            results.append(_trial(
                family="gaussian", model_type="gam", basis="duchon",
                rust_metric=0.50, mgcv_metric=0.75, seed=1000 + i,
            ))
        # Plus a clean cohort to make sure we only flag the bad one.
        for i in range(20):
            results.append(_trial(
                family="gaussian", model_type="gam", basis="ps",
                rust_metric=0.80, mgcv_metric=0.81, seed=2000 + i,
            ))
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
        # 10 mgcv wins, 0 rust wins → net = 10 > 5 in this cohort.
        results = []
        for i in range(10):
            # gap = +0.06 — too small for per-trial fail, big enough for win.
            results.append(_trial(
                family="binomial", model_type="gam", basis="tps",
                rust_metric=0.60, mgcv_metric=0.66, seed=3000 + i,
            ))
        gates = _FUZZ.compute_ci_gates(results, requested_trials=10)
        self.assertTrue(gates["failed"])
        gate_names = [gf["gate"] for gf in gates["gate_failures"]]
        self.assertIn("cohort_net_wins", gate_names)

    def test_rust_nan_inf_failure_fires(self) -> None:
        results = [
            _trial(rust_metric=0.80, mgcv_metric=0.81, seed=i)
            for i in range(100)
        ]
        # Trial 5 — rust returned NaN R² where mgcv was finite.
        results[5] = _trial(
            rust_metric=float("nan"), mgcv_metric=0.55, seed=5,
        )
        gates = _FUZZ.compute_ci_gates(results, requested_trials=100)
        self.assertTrue(gates["failed"])
        gate_names = [gf["gate"] for gf in gates["gate_failures"]]
        self.assertIn("rust_nan_inf", gate_names)

    def test_coverage_failure_fires_when_too_many_skipped(self) -> None:
        # 50 valid trials out of 100 requested ⇒ below the 80% floor.
        results = [
            _trial(rust_metric=0.80, mgcv_metric=0.81, seed=i)
            for i in range(50)
        ]
        gates = _FUZZ.compute_ci_gates(
            results, requested_trials=100, skipped_count=50,
        )
        self.assertTrue(gates["failed"])
        gate_names = [gf["gate"] for gf in gates["gate_failures"]]
        self.assertIn("coverage", gate_names)

    def test_depth_env_var_controls_default_n_trials(self) -> None:
        with mock.patch.dict(os.environ, {"FUZZ_DEPTH": "lean"}, clear=False):
            self.assertEqual(_FUZZ._default_n_trials(), 100)
        with mock.patch.dict(os.environ, {"FUZZ_DEPTH": "default"}, clear=False):
            self.assertEqual(_FUZZ._default_n_trials(), 200)
        with mock.patch.dict(os.environ, {"FUZZ_DEPTH": "deep"}, clear=False):
            self.assertEqual(_FUZZ._default_n_trials(), 500)
        with mock.patch.dict(os.environ, {"FUZZ_DEPTH": "heavy"}, clear=False):
            self.assertEqual(_FUZZ._default_n_trials(), 1000)
        with mock.patch.dict(os.environ, {"FUZZ_DEPTH": "BOGUS"}, clear=False):
            # Unknown depth tokens fall back to the safe default rather than
            # crashing — CI shouldn't bomb on a typo.
            self.assertEqual(_FUZZ._default_n_trials(), 200)

    def test_baseline_regression_fires(self) -> None:
        # Baseline says gaussian/gam/ps cohort had median gap +0.02; current
        # run has +0.10 — delta 0.08 > threshold 0.05 ⇒ fail.
        results = []
        for i in range(20):
            results.append(_trial(
                family="gaussian", model_type="gam", basis="ps",
                rust_metric=0.70, mgcv_metric=0.80, seed=4000 + i,
            ))
        baseline = {
            "threshold": 0.05,
            "cohorts": {"gaussian/gam/ps": 0.02},
        }
        gates = _FUZZ.compute_ci_gates(
            results, requested_trials=20, baseline=baseline,
        )
        self.assertTrue(gates["failed"])
        gate_names = [gf["gate"] for gf in gates["gate_failures"]]
        self.assertIn("baseline_regression", gate_names)


if __name__ == "__main__":
    unittest.main()
