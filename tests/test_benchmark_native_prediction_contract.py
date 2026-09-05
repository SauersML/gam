"""Benchmark extraction against actual fitted native prediction schemas."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

import gamfit


_PATH = Path(__file__).resolve().parents[1] / "bench/run_suite.py"
_SPEC = importlib.util.spec_from_file_location("native_contract_run_suite", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUN_SUITE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_SUITE)


class NativeBenchmarkPredictionContractTests(unittest.TestCase):
    def _check_prediction(self, model, rows, family):
        prediction = model.predict(rows, return_type="dict")
        expected = np.asarray(prediction["posterior_mean"], dtype=float)
        plugin = np.asarray(prediction["mean_plugin"], dtype=float)
        eta = np.asarray(prediction["linear_predictor_plugin"], dtype=float)
        actual, seconds = _RUN_SUITE._time_stable_mean_prediction(
            lambda: model.predict(rows, return_type="dict"))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            _RUN_SUITE._prediction_column_array(prediction, "linear_predictor_plugin"), eta)
        self.assertEqual(actual.shape, (len(rows["x"]),))
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.isfinite(seconds) and seconds >= 0)
        separation = float(np.max(np.abs(expected - plugin)))
        print(json.dumps({"family": family, "prediction_columns": sorted(prediction),
                          "posterior_plugin_separation": separation,
                          "median_prediction_seconds": seconds}))
        return expected, plugin, eta

    def test_actual_gaussian_predictions_use_current_benchmark_schema(self):
        x = np.linspace(-1.0, 1.0, 32)
        y = 0.7 + 1.3 * x + 0.1 * np.cos(7.0 * x)
        model = gamfit.fit({"x": x, "y": y}, "y ~ x", family="gaussian")
        expected, plugin, eta = self._check_prediction(
            model, {"x": np.array([-0.8, -0.1, 0.6])}, "gaussian")
        np.testing.assert_allclose(plugin, eta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(expected, plugin, rtol=0, atol=1e-12)

    def test_actual_binomial_predictions_keep_posterior_mean_distinct_from_plugin(self):
        x = np.repeat([-1.5, -0.5, 0.5, 1.5], 8)
        y = np.concatenate([np.r_[np.ones(successes), np.zeros(8 - successes)]
                            for successes in (2, 3, 5, 6)])
        model = gamfit.fit({"x": x, "y": y}, "y ~ x", family="binomial")
        expected, plugin, eta = self._check_prediction(
            model, {"x": np.array([-1.25, -0.25, 0.75, 1.25])}, "binomial")
        np.testing.assert_allclose(plugin, 1.0 / (1.0 + np.exp(-eta)), rtol=0, atol=1e-12)
        self.assertTrue(np.all((expected > 0) & (expected < 1)))
        self.assertGreater(float(np.max(np.abs(expected - plugin))), 1e-5)


if __name__ == "__main__":
    unittest.main()
