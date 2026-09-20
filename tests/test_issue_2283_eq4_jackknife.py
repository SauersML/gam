"""The #2283 grouped jackknife against the real Eq-4 scorer.

Two properties the paired measurement rests on:

* deleting rows from a fitted featurizer (``eq4_jackknife.restrict_rows``) scores
  exactly as the same featurizer built on the kept rows, so every deleted-block
  score is the scorer's own value on that subsample;
* on a Gaussian source whose population spectrum is known, the plug-in carries
  the Wishart bias the #2283 derivation predicts, and the jackknife removes all
  of it but the n^-2 remainder, which is also predicted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "1026_close"
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

import arm_featurizers  # noqa: E402
import bits_eq4  # noqa: E402
import eq4_jackknife  # noqa: E402

HORIZON = 120_000


def _score(featurizer, rows):
    return bits_eq4.description_length(
        featurizer, np.asarray(rows, dtype=np.float64), amortization_horizon=HORIZON
    )


def _flat_featurizer(decoder, indices, codes):
    gate, contribution, code_dims, params = arm_featurizers._flat_block_from_sparse(
        decoder, indices, codes
    )
    recon = np.einsum("nk,nkp->np", codes.astype(np.float64), decoder[indices].astype(np.float64))
    return bits_eq4.make_fitted_featurizer(
        name="flat", gate=gate, atom_contribution=contribution, code_dims=code_dims,
        dictionary_params=params, recon=recon, fit_seconds=0.0,
    )


def test_a_restricted_featurizer_scores_exactly_as_one_built_on_the_kept_rows():
    rng = np.random.default_rng(2283)
    rows, p, atoms, active = 400, 16, 12, 3
    decoder = rng.standard_normal((atoms, p)).astype(np.float32)
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
    indices = np.stack([rng.choice(atoms, active, replace=False) for _ in range(rows)])
    codes = rng.standard_normal((rows, active)).astype(np.float32)
    x = np.einsum("nk,nkp->np", codes, decoder[indices]) + 0.1 * rng.standard_normal((rows, p))
    fitted = _flat_featurizer(decoder, indices, codes)

    row_ids = rng.permutation(10 * rows)[:rows]
    blocks = eq4_jackknife.corpus_blocks(row_ids)
    assert sorted(np.concatenate(blocks).tolist()) == list(range(rows))
    assert {block.size for block in blocks} == {rows // eq4_jackknife.JACKKNIFE_GROUPS}
    for block in blocks:
        ids = np.sort(row_ids[block])
        outside = np.setdiff1d(row_ids, ids)
        # Each block is one contiguous run of the scored rows' corpus ids.
        assert not np.any((outside > ids[0]) & (outside < ids[-1]))

    keep = np.setdiff1d(np.arange(rows), blocks[3])
    restricted = _score(eq4_jackknife.restrict_rows(fitted, keep), x[keep])
    rebuilt = _score(_flat_featurizer(decoder, indices[keep], codes[keep]), x[keep])
    scalars = {key for key, value in rebuilt.items() if isinstance(value, (int, float))}
    assert scalars and all(restricted[key] == rebuilt[key] for key in scalars), {
        key: (restricted[key], rebuilt[key]) for key in scalars
    }

    payload = eq4_jackknife.leave_one_block_out(fitted, x, row_ids, _score)
    assert payload["groups"] == eq4_jackknife.JACKKNIFE_GROUPS
    assert payload["block_sizes"] == [block.size for block in blocks]
    assert payload["block_positions"] == eq4_jackknife.block_positions_digest(blocks)
    assert payload["leave_one_out"][3] == {key: restricted[key] for key in payload["leave_one_out"][3]}


def _digamma(x: float) -> float:
    """psi(x) by its asymptotic series; the truncation error is below 1e-16 for x >= 100."""
    inverse = 1.0 / x
    square = inverse * inverse
    return (
        math.log(x)
        - 0.5 * inverse
        - square * (1.0 / 12 - square * (1.0 / 120 - square * (1.0 / 252 - square / 240)))
    )


def _bits(x):
    """No reconstruction and one atom that never fires: the score is the residual's.

    The scorer fetches contributions only for atoms that fire, so the callback is
    never reached.
    """
    return bits_eq4.make_fitted_featurizer(
        name="residual-only", gate=np.zeros((x.shape[0], 1)),
        atom_contribution=lambda atom: np.zeros((0, x.shape[1])),
        code_dims=np.ones(1, dtype=int), dictionary_params=0,
        recon=np.zeros_like(x), fit_seconds=0.0,
    )


def test_the_jackknife_removes_the_predicted_wishart_bias_from_a_gaussian_residual():
    """Plug-in bias -P(P+1)/(4 n ln 2) at leading order, removed to its n^-2 part.

    x ~ N(0, diag(s)) in P = 64 dimensions from n = 640 rows, the acceptance
    cell's P/n regime (0.1 here, 0.085 there). At R2 = 0.99 every residual mode is
    coded, so the score is 1 + sum_k 1/2 log2(mu_k / theta) with theta = D*/P, and
    its plug-in expectation is exact up to the delta method on theta:

      E sum ln mu_hat - sum ln s = sum_{i=1..P} psi((m-i+1)/2) + P ln 2 - P ln m,
      E ln V_hat - ln V ~= ln((m-1)/m) - sum s^2 / ((m-1) V^2)    (centred V_ref)

    at m = n rows for the plug-in and m = n(1-1/J) for each deleted block. The
    population value is the same scorer on rows whose moments ARE the population's.
    """
    p, n, replicates = 64, 640, 200
    groups = eq4_jackknife.JACKKNIFE_GROUPS
    variances = 0.5 + np.arange(p) / (p - 1)
    total = float(variances.sum())
    key = "bits_at_r2_0.99"

    population_rows = np.sqrt(p * variances)[:, None] * np.eye(p)
    population_rows = np.concatenate([population_rows, -population_rows])
    population = _score(_bits(population_rows), population_rows)[key]

    def wishart(m):
        return sum(_digamma((m - i + 1) / 2.0) for i in range(1, p + 1)) + p * math.log(2.0 / m)

    def reference(m):
        return math.log((m - 1) / m) - float((variances**2).sum()) / ((m - 1) * total**2)

    def expected_offset(m):
        return (wishart(m) - p * reference(m)) / (2.0 * math.log(2.0))

    deleted = n - n // groups
    predicted_plug_in = expected_offset(n)
    predicted_corrected = groups * expected_offset(n) - (groups - 1) * expected_offset(deleted)

    rng = np.random.default_rng(2283)
    plug_in, corrected, variance = [], [], []
    for _ in range(replicates):
        x = rng.standard_normal((n, p)) * np.sqrt(variances)
        full = _score(_bits(x), x)[key]
        payload = eq4_jackknife.leave_one_block_out(_bits(x), x, np.arange(n), _score)
        estimate = eq4_jackknife.jackknife(full, [entry[key] for entry in payload["leave_one_out"]])
        plug_in.append(full - population)
        corrected.append(estimate.corrected - population)
        variance.append(estimate.standard_error**2)
    plug_in, corrected = np.asarray(plug_in), np.asarray(corrected)

    def standard_error(values):
        return float(values.std(ddof=1)) / math.sqrt(values.size)

    # Four Monte Carlo standard errors: a correct prediction fails with
    # probability about 6e-5 per assertion.
    assert abs(plug_in.mean() - predicted_plug_in) <= 4 * standard_error(plug_in), (
        plug_in.mean(), predicted_plug_in, standard_error(plug_in),
    )
    assert abs(corrected.mean() - predicted_corrected) <= 4 * standard_error(corrected), (
        corrected.mean(), predicted_corrected, standard_error(corrected),
    )
    # The leading term is the all-active Wishart bound, and the jackknife leaves
    # only the n^-2 part of it.
    assert abs(predicted_plug_in - eq4_jackknife.wishart_all_active_bias_bits(p, n)) < 0.1 * abs(
        predicted_plug_in
    )
    assert abs(predicted_corrected) < 0.05 * abs(predicted_plug_in)
    # The jackknife variance never understates the plug-in's replicate variance
    # (Efron-Stein): it is exact for the statistic's linear part and inflates its
    # higher-order parts. Here the linear part nearly cancels, because theta
    # moves with the trace of the same moment: its variance is
    # (2/n) sum_k (1 - s_k / mean s)^2 / (2 ln 2)^2, below the replicate variance,
    # so the second-order part the jackknife inflates is material and the SE it
    # gives is conservative. The ratio's relative standard error combines the
    # replicate variance's sqrt(2/(R-1)) and the mean of R jackknife variances
    # of J-1 degrees of freedom each, sqrt(2/((J-1) R)).
    replicate_variance = float(plug_in.var(ddof=1))
    linear_variance = (
        2.0 / n * float(((1.0 - variances / variances.mean()) ** 2).sum()) / (2.0 * math.log(2.0)) ** 2
    )
    ratio = float(np.mean(variance)) / replicate_variance
    ratio_error = math.sqrt(2.0 / (replicates - 1) + 2.0 / ((groups - 1) * replicates))
    assert ratio >= 1.0 - 4 * ratio_error, (ratio, replicate_variance, linear_variance)
    assert linear_variance < replicate_variance, (linear_variance, replicate_variance)
