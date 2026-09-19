"""Grouped jackknife over the scored rows of the #2283 Eq-4 comparison.

Eq. 4 is a plug-in: every term but the dictionary penalty is an average or a
spectrum over the n scored rows, so the score of a FIXED featurizer carries an
estimation error. Its variance is O(1/n). Its bias is a/n + O(n^-2), from the
concavity of the rate in each eigenvalue and the spread of the sample
eigenvalues of a P-dimensional moment. For a Gaussian residual whose every mode
is coded, the Wishart log-determinant gives

    E sum_k 1/2 log2 mu_hat_k - sum_k 1/2 log2 mu_k ~= -P(P+1) / (4 n ln 2)

bits, -63.1 at P = 2048, n = 24000. The arms of the pair share rows but not
spectra, so the paired difference keeps the difference of their biases.

Deleting one of J equal blocks leaves n(1 - 1/J) rows, whose bias is
a / (n(1 - 1/J)). The corrected value J*B - (J-1)*mean_j B_(j) therefore has no
a/n term, and ((J-1)/J) * sum_j (B_(j) - mean)^2 estimates Var(B): exactly for
the statistic's linear part, and with the higher-order parts inflated
(Efron-Stein), so the standard error it gives is conservative. Both use the
scorer itself on row subsets of one fitted featurizer, so there is no second
implementation of the score (#2283).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Callable

import numpy as np

# The jackknife SE has relative standard error 1/sqrt(2(J-1)): 16 % at J = 20,
# for J scorings per arm beyond the full one. Every deleted sample keeps 95 % of
# the rows, so it stays in the full sample's P/n regime.
JACKKNIFE_GROUPS = 20


def corpus_blocks(row_ids: np.ndarray) -> list[np.ndarray]:
    """Scored-row positions cut into JACKKNIFE_GROUPS contiguous corpus blocks.

    Neighbouring corpus rows are correlated: consecutive creditscope L30
    residual-stream rows share about a fifth of their variance. A block therefore
    holds whole runs of corpus row ids, so a deleted block removes correlated
    rows together instead of splitting them across blocks.
    """
    row_ids = np.asarray(row_ids)
    if row_ids.ndim != 1 or row_ids.size < JACKKNIFE_GROUPS:
        raise ValueError(
            f"the jackknife needs at least {JACKKNIFE_GROUPS} scored rows, got shape {row_ids.shape}"
        )
    order = np.argsort(row_ids, kind="stable")
    return [np.sort(block) for block in np.array_split(order, JACKKNIFE_GROUPS)]


class _RestrictedContribution:
    """``atom_contribution(g)`` of a featurizer seen through a row subset."""

    __slots__ = ("_inner", "_keep")

    def __init__(self, inner: Any, keep: np.ndarray) -> None:
        self._inner = inner
        self._keep = keep

    def __getitem__(self, take: Any) -> np.ndarray:
        return self._inner[self._keep[np.asarray(take)]]


def restrict_rows(fitted: Any, keep: np.ndarray) -> Any:
    """The same fitted featurizer on the scored rows ``keep`` (sorted positions).

    Decoder, codes, curved latents and every other fitted quantity are unchanged;
    only the rows the scorer sees are removed.
    """
    keep = np.asarray(keep, dtype=np.int64)
    contribution = fitted.atom_contribution
    return dataclasses.replace(
        fitted,
        gate=np.asarray(fitted.gate)[keep],
        recon=np.asarray(fitted.recon)[keep],
        atom_contribution=lambda atom: _RestrictedContribution(contribution(atom), keep),
    )


def leave_one_block_out(
    fitted: Any,
    x_bits: np.ndarray,
    row_ids: np.ndarray,
    score: Callable[[Any, np.ndarray], dict],
) -> dict:
    """Score ``fitted`` once per deleted corpus block of the scored rows.

    ``score(featurizer, rows)`` is the Eq-4 scorer with the run's declared
    amortization horizon. Returns the block digest and one record of the scorer's
    scalar outputs per deleted block, in block order.
    """
    blocks = corpus_blocks(row_ids)
    positions = np.arange(np.asarray(x_bits).shape[0], dtype=np.int64)
    leave_one_out = []
    for block in blocks:
        keep = np.setdiff1d(positions, block, assume_unique=True)
        report = score(restrict_rows(fitted, keep), np.asarray(x_bits)[keep])
        leave_one_out.append(
            {
                key: value
                for key, value in report.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
    return {
        "groups": JACKKNIFE_GROUPS,
        "block_sizes": [int(block.size) for block in blocks],
        "block_positions": block_positions_digest(blocks),
        "leave_one_out": leave_one_out,
    }


def block_positions_digest(blocks: list[np.ndarray]) -> str:
    """A digest of the block partition, so two arms can prove they share it."""
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(np.ascontiguousarray(block, dtype=np.int64).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class JackknifeEstimate:
    """Plug-in value, first-order bias-corrected value, and standard error."""

    plug_in: float
    corrected: float
    standard_error: float

    @property
    def bias(self) -> float:
        """The jackknife estimate of the plug-in's bias, ``plug_in - corrected``."""
        return self.plug_in - self.corrected


def jackknife(full: float, leave_one_out: list[float]) -> JackknifeEstimate:
    """Grouped jackknife of one statistic from its full and deleted-block values."""
    values = np.asarray(leave_one_out, dtype=np.float64)
    groups = values.size
    if groups < 2:
        raise ValueError(f"a jackknife needs at least two deleted blocks, got {groups}")
    if not (math.isfinite(full) and np.all(np.isfinite(values))):
        raise ValueError("jackknife inputs must be finite")
    mean = float(values.mean())
    corrected = groups * float(full) - (groups - 1) * mean
    variance = (groups - 1) / groups * float(((values - mean) ** 2).sum())
    return JackknifeEstimate(float(full), corrected, math.sqrt(variance))


def paired_jackknife(
    full_a: float,
    full_b: float,
    leave_one_out_a: list[float],
    leave_one_out_b: list[float],
) -> JackknifeEstimate:
    """Grouped jackknife of ``a - b``, both scored on the same deleted blocks."""
    if len(leave_one_out_a) != len(leave_one_out_b):
        raise ValueError("paired jackknife arms must share their deleted blocks")
    return jackknife(
        float(full_a) - float(full_b),
        [float(a) - float(b) for a, b in zip(leave_one_out_a, leave_one_out_b)],
    )


def wishart_all_active_bias_bits(p: int, n: int) -> float:
    """Leading plug-in bias of sum_k 1/2 log2 mu_hat_k when all P modes are coded."""
    return -p * (p + 1) / (4.0 * n * math.log(2.0))
