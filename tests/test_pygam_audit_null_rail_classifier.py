"""GAMClassifier on sklearn's ``check_classifiers_train`` blobs data must fit.

The pyGAM audit (bench/pygam_audit, api.md F3 and inference.md B2) ran
sklearn's estimator checks with ``GAMClassifier(formula="s(x0)+s(x1)")`` on the
0.1.267 wheel. ``check_classifiers_train`` builds its data as below; its float32
pass raised ``IntegrationError: Outer smoothing-parameter optimization did not
certify a stationary optimum`` (``|Pg|=2.675e-4 > bound=7.178e-5``,
``hessian_psd=NO``) after about 9 s, and the three-class set was refused as
binary-only. Both now fit from a certified outer optimum.

The data are linearly well separated but not perfectly: a smooth classifier
reaches training accuracy 0.97 on the binary subset, so each fit is also held
to the accuracy a converged fit reaches on its own training rows.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from gamfit.sklearn import GAMClassifier

FORMULA = "s(x0)+s(x1)"


def _check_classifiers_train_blobs(dtype):
    """``sklearn.utils.estimator_checks.check_classifiers_train``'s data."""
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m).astype(dtype)
    return X_m, y_m, X_m[y_m != 2], y_m[y_m != 2]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_binary_blobs_classifier_fits(dtype):
    _, _, X_b, y_b = _check_classifiers_train_blobs(dtype)
    classifier = GAMClassifier(formula=FORMULA).fit(X_b, y_b)
    np.testing.assert_array_equal(classifier.classes_, [0, 1])
    assert np.mean(classifier.predict(X_b) == y_b) >= 0.95


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_multiclass_blobs_classifier_fits(dtype):
    X_m, y_m, _, _ = _check_classifiers_train_blobs(dtype)
    classifier = GAMClassifier(formula=FORMULA).fit(X_m, y_m)
    np.testing.assert_array_equal(classifier.classes_, [0, 1, 2])
    assert np.mean(classifier.predict(X_m) == y_m) >= 0.9
