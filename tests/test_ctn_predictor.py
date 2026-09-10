"""Prediction-chain contracts. Native numerical acceptance is a separate test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import gamfit
from gamfit._ctn_model import (CtnMarginalSlopeModel, _SCORE, crossfit_assignment,
                               fit_ctn_chain)


class Transform:
    def __init__(self, center):
        self.center = center

    def transformation_score(self, data):
        return np.asarray(data["pgs"]) - self.center

    def predict(self, data, **kwargs):
        raise AssertionError("CTN predict is a conditional mean, not a score")

    def dumps(self):
        return json.dumps({"center": self.center}).encode()


class Outcome:
    def predict(self, data, **kwargs):
        return .1 + .03 * np.asarray(data[_SCORE]) + .02 * np.asarray(data["x"])

    def dumps(self):
        return b'{"outcome":true}'


class CtnPredictorContracts(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"pgs": np.arange(12.), "x": np.linspace(-1, 1, 12),
                                  "y": np.arange(12) % 2, "family_id": np.arange(12) // 2,
                                  "fold": np.repeat(np.arange(3), 4)})
        self.recipe = gamfit.CtnStage1("pgs", "x", fold_column="fold", group_column="family_id")

    def test_explicit_groups_and_permutation(self):
        recipe = gamfit.CtnStage1("pgs", "x", group_column="family_id", folds=3, seed=72)
        folds = crossfit_assignment(self.data, recipe)
        order = np.random.default_rng(2).permutation(12)
        np.testing.assert_array_equal(crossfit_assignment(self.data.iloc[order], recipe), folds[order])
        self.assertTrue(np.all(folds[::2] == folds[1::2]))
        bad = self.data.copy()
        bad.loc[0, "fold"] = 1
        with self.assertRaisesRegex(ValueError, "group crosses"):
            crossfit_assignment(bad, self.recipe)
        with self.assertRaisesRegex(ValueError, "fold_column or group_column"):
            gamfit.CtnStage1("pgs", "x")

    def test_fold_local_fits_oof_scores_and_full_training_deployment(self):
        fits = []

        def fit(data, formula, **options):
            fits.append((data.copy(), formula, options))
            return Transform(float(data.pgs.mean())) if options.get("transformation_normal") else Outcome()

        model = fit_ctn_chain(fit, self.data, "y ~ x", self.recipe,
                              {"family": "bernoulli-marginal-slope", "slope_formula": "1"})
        self.assertEqual(len(fits), 5)
        for fold, (data, _, options) in enumerate(fits[:3]):
            self.assertFalse((data.fold == fold).any())
            self.assertNotIn("ctn_stage1", options["config"])
        outcome_data, _, options = fits[3]
        expected = np.concatenate([self.data.loc[self.data.fold == f, "pgs"].to_numpy() -
                                   self.data.loc[self.data.fold != f, "pgs"].mean() for f in range(3)])
        np.testing.assert_array_equal(outcome_data[_SCORE], expected)
        self.assertEqual(options["config"], {"frozen_score": True})
        self.assertEqual(len(fits[-1][0]), len(self.data))
        self.assertEqual(model.transform.center, self.data.pgs.mean())
        self.assertNotIn(_SCORE, self.data)

    def test_failed_fold_is_not_replaced(self):
        with self.assertRaisesRegex(RuntimeError, "fold 0 failed"):
            fit_ctn_chain(lambda *a, **k: (_ for _ in ()).throw(ValueError("bad fit")),
                          self.data, "y ~ x", self.recipe,
                          {"family": "bernoulli-marginal-slope"})

    def test_raw_score_prediction_save_load_and_batch_contract(self):
        model = CtnMarginalSlopeModel(Transform(4.5), Outcome(), self.recipe, "fold-hash")
        test = self.data[["pgs", "x"]]
        expected = model.predict(test)
        np.testing.assert_array_equal(model.predict(test.iloc[::-1]), expected[::-1])
        np.testing.assert_array_equal(model.predict(test.iloc[[2]]), expected[[2]])
        explicit = model.outcome.predict(test.assign(**{_SCORE: model.transformation_score(test)}))
        np.testing.assert_array_equal(expected, explicit)
        real_loads = gamfit.loads

        def load_component(raw):
            payload = json.loads(raw)
            if "center" in payload:
                return Transform(payload["center"])
            if "outcome" in payload and "schema" not in payload:
                return Outcome()
            return real_loads(raw)

        with tempfile.TemporaryDirectory() as directory, patch("gamfit._api.loads", side_effect=load_component):
            path = Path(directory) / "model.gamfit"
            model.save(path)
            loaded = gamfit.load(path)
            np.testing.assert_array_equal(loaded.predict(test), expected)
        np.testing.assert_array_equal(model.predict(test.assign(y=1e9, fold=-1)), expected)

    def test_public_fit_dispatches_without_native_influence_payload(self):
        with patch("gamfit._api.fit_ctn_chain", return_value="chain") as chain:
            fitted = gamfit.fit(self.data, "y ~ x", family="bernoulli-marginal-slope",
                                slope_formula="1", transformation_normal_stage1=self.recipe)
        self.assertEqual(fitted, "chain")
        self.assertNotIn("transformation_normal_stage1", chain.call_args.args[-1])

    def test_requested_survival_horizons_ignore_observed_exit_and_event(self):
        class Surface:
            def cumulative_hazard_at(self, times):
                return np.tile(.2 * np.asarray(times), (12, 1))

        class SurvivalOutcome:
            formula = "Surv(entry, exit, event) ~ x"

            def predict(inner_self, data, **kwargs):
                np.testing.assert_array_equal(data["exit"], np.full(12, 5.))
                np.testing.assert_array_equal(data["entry"], np.full(12, 2.))
                np.testing.assert_array_equal(data["event"], np.zeros(12))
                return Surface()

        model = CtnMarginalSlopeModel(Transform(0), SurvivalOutcome(), self.recipe, "fold-hash")
        supplied = self.data.assign(exit=1000, event=1, entry=-20)
        result = model.survival_at(supplied, [2., 3., 5.], entry_time=2.)
        np.testing.assert_allclose(result[0], np.exp(-.2 * np.array([0, 1, 3])))


if __name__ == "__main__":
    unittest.main()
