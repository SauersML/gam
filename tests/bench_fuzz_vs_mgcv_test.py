import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FUZZ_PATH = _REPO_ROOT / "bench" / "fuzz_vs_mgcv.py"
_SPEC = importlib.util.spec_from_file_location("bench_fuzz_vs_mgcv", _FUZZ_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load fuzz benchmark module from {_FUZZ_PATH}")
_FUZZ: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FUZZ
_SPEC.loader.exec_module(_FUZZ)


def _fuzz_scenario(
    name: str,
    *,
    n_obs: int,
    n_features: int,
    seed: int = 0,
    family: str = "gaussian",
    model_type: str = "gam",
    basis_type: str = "ps",
) -> Any:
    return _FUZZ.FuzzScenario(
        trial_id=f"{seed}:{name}:{model_type}",
        seed=seed,
        name=name,
        family=family,
        model_type=model_type,
        basis_type=basis_type,
        n_obs=n_obs,
        n_features=n_features,
        formula="y ~ x0",
        mgcv_formula="y ~ s(x0)",
    )


class FuzzVsMgcvSelectionTests(unittest.TestCase):
    def test_cost_model_scales_with_rows_features_and_gamlss_multiplier(self) -> None:
        gam = _fuzz_scenario("gam", n_obs=10_000, n_features=4)
        gamlss = _fuzz_scenario(
            "gamlss",
            n_obs=10_000,
            n_features=4,
            model_type="gamlss",
        )

        self.assertEqual(_FUZZ._scenario_cost(gam), 40_000.0)
        self.assertEqual(_FUZZ._scenario_cost(gamlss), 72_000.0)

    def test_backfilled_selection_skips_oversized_mgcv_runner_draw(self) -> None:
        oversized = _fuzz_scenario(
            "gam_bench_mgcv_cv_skcfgtbl",
            n_obs=2_399_999,
            n_features=4,
        )
        keep_1 = _fuzz_scenario("small_1", n_obs=10_000, n_features=4, seed=1)
        keep_2 = _fuzz_scenario("small_2", n_obs=20_000, n_features=4, seed=2)
        candidates = [oversized, keep_1, keep_2]

        with patch.object(_FUZZ, "_candidate_scenarios", return_value=candidates):
            selected, skipped = _FUZZ.select_scenarios_backfilled(
                seed_start=0,
                target_count=2,
                excluded_ids=set(),
                max_scenario_cost=200_000.0,
            )

        self.assertEqual([sc.name for sc in selected], ["small_1", "small_2"])
        self.assertEqual([sc.name for sc, _ in skipped], ["gam_bench_mgcv_cv_skcfgtbl"])
        self.assertGreater(skipped[0][1], 9_000_000.0)

    def test_backfilled_selection_honors_existing_trial_ids(self) -> None:
        old = _fuzz_scenario("already_done", n_obs=10_000, n_features=4)
        new = _fuzz_scenario("new", n_obs=10_000, n_features=4, seed=1)

        with patch.object(_FUZZ, "_candidate_scenarios", return_value=[old, new]):
            selected, skipped = _FUZZ.select_scenarios_backfilled(
                seed_start=0,
                target_count=1,
                excluded_ids={old.trial_id},
                max_scenario_cost=200_000.0,
            )

        self.assertEqual([sc.name for sc in selected], ["new"])
        self.assertEqual(skipped, [])

    def test_backfilled_selection_fails_when_cap_blocks_coverage(self) -> None:
        oversized = _fuzz_scenario("too_big", n_obs=100_000, n_features=4)

        with patch.object(_FUZZ, "_candidate_scenarios", return_value=[oversized]):
            with self.assertRaisesRegex(RuntimeError, "only selected 0/1"):
                _FUZZ.select_scenarios_backfilled(
                    seed_start=0,
                    target_count=1,
                    excluded_ids=set(),
                    max_scenario_cost=200_000.0,
                )


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


def _ram_skipped_trial(*, rust_metric: float | None = 0.80, seed: int = 0) -> Any:
    """A trial whose gam arm ran but whose mgcv CV arm was RAM-skipped."""
    scenario = {
        "seed": seed,
        "family": "gaussian",
        "model_type": "gam",
        "basis_type": "ps",
        "n_obs": 2_399_999,
        "n_smooths": 1,
        "knots": 8,
    }
    fr = _FUZZ.FuzzResult(
        scenario=scenario,
        rust={"r2": rust_metric},
        mgcv={"skipped": True, "skip_reason": "projected peak exceeds runner RAM"},
    )
    fr.compute_gap()
    return fr


class FuzzRamGuardTests(unittest.TestCase):
    """gam#820: the fuzzer must not pick an (n, p) that OOM-kills the runner when
    gam and mgcv hold their peaks at once."""

    def test_oversized_np_skips_mgcv_arm_on_constrained_runner(self) -> None:
        # The OOM draw (n=2.4M, p=4) on a 15.6 GiB runner must be rejected.
        with patch.object(
            _FUZZ, "_read_meminfo", return_value={"MemTotal": 16_367_000}
        ):
            ok, reason = _FUZZ._mgcv_arm_fits_in_ram(2_399_999, 4)
        self.assertFalse(ok)
        self.assertIn("exceeds", reason)

    def test_small_fit_runs_both_arms(self) -> None:
        with patch.object(
            _FUZZ, "_read_meminfo", return_value={"MemTotal": 16_367_000}
        ):
            ok, reason = _FUZZ._mgcv_arm_fits_in_ram(150, 5)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_guard_defers_when_ram_unknown(self) -> None:
        # Non-Linux dev hosts have no /proc/meminfo: never skip blindly.
        with patch.object(_FUZZ, "_read_meminfo", return_value={}):
            ok, reason = _FUZZ._mgcv_arm_fits_in_ram(2_399_999, 4)
        self.assertTrue(ok)
        self.assertEqual(reason, "ram-unknown")

    def test_ram_skipped_trials_excluded_from_coverage_denominator(self) -> None:
        # 80 fully-comparable trials + 20 RAM-skipped (gam ran, mgcv skipped).
        # The coverage gate must measure 80/80, not 80/100, so it passes.
        results = [
            _trial(rust_metric=0.80, mgcv_metric=0.80, seed=i) for i in range(80)
        ]
        results += [_ram_skipped_trial(seed=1000 + i) for i in range(20)]
        gates = _FUZZ.compute_ci_gates(results, requested_trials=100)
        self.assertEqual(gates["ram_skipped_count"], 20)
        self.assertNotIn("coverage", _gate_names(gates))
        self.assertFalse(gates["failed"])


if __name__ == "__main__":
    unittest.main()
