"""Audit the original prostate AUC/NLL quality bars on exported held-out rows.

Usage: python binomial_reference_audit.py PROSTATE_CSV GAM_PREDICTIONS_CSV
Uses the quality fixture's deterministic splits and reference model settings.
Only EBM's worker count changes, to respect the shared MSI CPU allocation.
"""

import importlib.metadata
import json
import sys
import time

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from pygam import LogisticGAM, s
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits


def run(data_path, prediction_path):
    data = np.genfromtxt(data_path, delimiter=",", names=True)
    prediction = np.genfromtxt(prediction_path, delimiter=",", names=True)
    order = np.argsort(prediction["row"])
    prediction = prediction[order]
    assert np.array_equal(prediction["row"], np.arange(len(data))), "each observation must be held out exactly once"
    assert np.array_equal(prediction["fold"], np.arange(len(data)) % 5), "fold identities changed"
    assert np.array_equal(prediction["y"], data["y"]), "held-out labels changed"
    x = np.column_stack([data["pc1"], data["pc2"]])
    y = data["y"]
    scores = {name: [] for name in ("gam", "ebm", "pygam")}
    losses = {}
    for fold in range(5):
        started = time.monotonic()
        test = np.arange(len(data)) % 5 == fold
        train = ~test
        eta = prediction["eta"][test]
        scores["gam"].append(float(roc_auc_score(y[test], eta)))
        ebm = ExplainableBoostingClassifier(interactions=0, random_state=0, n_jobs=4)
        ebm.fit(x[train], y[train])
        p_ebm = ebm.predict_proba(x[test])[:, 1]
        lg = LogisticGAM(s(0, n_splines=5) + s(1, n_splines=5)).fit(x[train], y[train])
        p_lg = lg.predict_proba(x[test])
        scores["ebm"].append(float(roc_auc_score(y[test], p_ebm)))
        scores["pygam"].append(float(roc_auc_score(y[test], p_lg)))
        if fold == 0:
            losses = {
                "gam": float(np.mean(np.logaddexp(0, eta) - y[test] * eta)),
                "ebm": float(log_loss(y[test], np.clip(p_ebm, 1e-12, 1 - 1e-12), labels=[0, 1])),
                "pygam": float(log_loss(y[test], np.clip(p_lg, 1e-12, 1 - 1e-12), labels=[0, 1])),
            }
        print(json.dumps({"fold": fold, "auc": {k: v[-1] for k, v in scores.items()},
                          "reference_seconds": time.monotonic() - started}), flush=True)
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    verdict = {
        "auc_absolute": means["gam"] >= 0.62,
        "auc_match_or_beat": means["gam"] >= max(means["ebm"], means["pygam"]) - 0.02,
        "nll_absolute": losses["gam"] <= 0.66,
        "nll_match_or_beat": losses["gam"] <= min(losses["ebm"], losses["pygam"]) + 0.03,
    }
    print(json.dumps({"mean_auc": means, "fold_0_nll": losses, "verdict": verdict,
                      "versions": {name: importlib.metadata.version(name) for name in ("interpret-core", "pygam", "scikit-learn", "numpy")}}), flush=True)
    assert all(verdict.values()), "the original binomial quality bars did not all pass"


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run(*sys.argv[1:])
