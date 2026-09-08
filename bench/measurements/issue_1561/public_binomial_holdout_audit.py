"""Replay the original five prostate folds through the public prediction API.

From the repository root:
python -m bench.measurements.issue_1561.public_binomial_holdout_audit DATA_CSV REFERENCE_LOG OUTPUT_PREFIX
Reference scores are the retained deterministic EBM/pyGAM audit, not rerun here.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import gamfit
import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def auc(labels, scores):
    difference = scores[labels == 1, None] - scores[labels == 0]
    assert difference.size > 0, "both held-out classes are required"
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def run(data_path, reference_path, output_prefix):
    data = np.genfromtxt(data_path, delimiter=",", names=True)
    references = [
        json.loads(line) for line in Path(reference_path).read_text().splitlines()
        if line.startswith('{"mean_auc":')
    ]
    assert len(references) == 1, "exactly one retained reference summary is required"
    reference = references[0]
    artifact = Path(gamfit._rust.__file__).resolve()
    record = {
        "artifact": str(artifact), "artifact_sha256": digest(artifact),
        "data_sha256": digest(data_path), "reference_sha256": digest(reference_path),
        "script_sha256": digest(__file__), "python": sys.version,
        "numpy": np.__version__, "reference": reference,
        "reference_recomputed": False, "folds": [],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "all_folds_completed": False,
    }
    output_json = Path(output_prefix + ".json")
    output_json.write_text(json.dumps(record, indent=2))
    audit_started = time.monotonic()
    results = np.empty((len(data), 3))
    with open(output_prefix + ".csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "row", "y", "eta", "mean_plugin", "posterior_mean"])
        for fold in [3, 0, 1, 2, 4]:
            started = time.monotonic()
            record["active_fold"] = fold
            output_json.write_text(json.dumps(record, indent=2))
            test = np.arange(len(data)) % 5 == fold
            training = {name: data[name][~test] for name in data.dtype.names}
            model = gamfit.fit(training, "y ~ s(pc1, k=5) + s(pc2, k=5)", family="binomial")
            held_out = {name: data[name][test] for name in ("pc1", "pc2")}
            predicted = model.predict(held_out, return_type="dict")
            eta = np.asarray(predicted["linear_predictor_plugin"])
            plugin = np.asarray(predicted["mean_plugin"])
            posterior = np.asarray(predicted["posterior_mean"])
            assert np.isfinite(eta).all()
            assert np.all((plugin > 0) & (plugin < 1))
            assert np.all((posterior > 0) & (posterior < 1))
            results[test] = np.column_stack([eta, plugin, posterior])
            for row, values in zip(np.flatnonzero(test), results[test]):
                writer.writerow([fold, row, data["y"][row], *values])
            stream.flush()
            item = {
                "fold": fold, "seconds": time.monotonic() - started,
                "auc_plugin": auc(data["y"][test], plugin),
                "auc_posterior": auc(data["y"][test], posterior),
            }
            record["folds"].append(item)
            record["total_seconds"] = time.monotonic() - audit_started
            output_json.write_text(json.dumps(record, indent=2))
            print(json.dumps(item), flush=True)
    fold_zero = np.arange(len(data)) % 5 == 0
    labels = data["y"][fold_zero]
    eta = results[fold_zero, 0]
    means = {
        "plugin": float(np.mean([fold["auc_plugin"] for fold in record["folds"]])),
        "posterior": float(np.mean([fold["auc_posterior"] for fold in record["folds"]])),
    }
    posterior = results[fold_zero, 2]
    losses = {
        "plugin": float(np.mean(np.logaddexp(0, eta) - labels * eta)),
        "posterior": float(-np.mean(labels * np.log(posterior) + (1 - labels) * np.log1p(-posterior))),
    }
    best_auc = max(reference["mean_auc"][name] for name in ("ebm", "pygam"))
    best_nll = min(reference["fold_0_nll"][name] for name in ("ebm", "pygam"))
    verdict = {
        mode: {
            "auc_absolute": means[mode] >= 0.62,
            "auc_reference": means[mode] >= best_auc - 0.02,
            "nll_absolute": losses[mode] <= 0.66,
            "nll_reference": losses[mode] <= best_nll + 0.03,
        } for mode in means
    }
    record.update(mean_auc=means, fold_0_nll=losses, verdict=verdict,
                  all_folds_completed=True, total_seconds=time.monotonic() - audit_started)
    record.pop("active_fold")
    output_json.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    assert all(passed for checks in verdict.values() for passed in checks.values())


if __name__ == "__main__":
    run(*sys.argv[1:])
