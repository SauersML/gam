"""Cross-node fixed-topology deterministic reduction — bit-reproducible.

This is the cross-node lift of the intra-node *pairwise-tree determinism*
discipline. When ``r`` partial accumulators (one per rank) are reduced into a
global accumulator, the floating-point result depends on the *order* the
partials are summed (addition is not associative in floating point). A naive
"sum as messages arrive" reduction is therefore non-reproducible: rerun the job
on a flakier network and the partials arrive in a different order, the rounding
differs, and the global border / accumulator changes bit-for-bit. Two runs of
the *same* job must agree to the bit, independent of network timing.

:class:`TreeReducer` pins a **fixed binary reduction tree** over the rank ids.
The tree is a function of the rank *count* alone — never of arrival order, never
of wall-clock — so the accumulation order is fixed before any message is sent.
``reduce(partials)`` evaluates that tree in a deterministic post-order: every
internal node is ``combine(left_subtree, right_subtree)`` with the left subtree
always folded before the right. The result is identical whether the partials
were handed in sorted, reversed, or in any permutation.

Checkpoint-as-job-model
------------------------
The reduction is **restartable**. Each rank can checkpoint its own partial (its
leaf accumulator) and any internal subtree result it has finished folding via
:meth:`TreeReducer.checkpoint` → a plain, JSON-round-trippable dict. A fresh
process restores with :meth:`TreeReducer.restore` and resumes: subtrees already
recorded in the checkpoint are *not* recombined (their bit-exact value is
reused), and only the missing folds are evaluated — in the same fixed tree
order, so the resumed result equals the from-scratch result to the bit.

Framework-light
---------------
The reduction logic is numpy-level (``combine`` defaults to elementwise add of
``float64`` arrays) so the whole thing is testable without a real cluster or a
torch build. ``torch`` is optional: if partials arrive as torch tensors they are
coerced to numpy via the shared :func:`gamfit.torch._coerce.to_numpy_f64` bridge
and the reduced result is returned as numpy (the caller promotes back at the
framework boundary, mirroring the harvest-shard f32→f64 convention).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "ReductionTree",
    "TreeReducer",
    "build_reduction_tree",
]


# ---------------------------------------------------------------------------
# Fixed binary reduction tree over rank ids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReductionTree:
    """A fixed binary reduction tree over ``n_ranks`` leaves.

    A node is either a leaf ``("leaf", rank_id)`` or an internal combine node
    ``("node", left, right)`` where ``left`` / ``right`` are child
    :class:`ReductionTree` instances. The tree shape is a pure function of
    ``n_ranks`` (see :func:`build_reduction_tree`), so two builds with the same
    rank count produce structurally identical trees and therefore identical
    accumulation order.

    ``leaves()`` lists the leaf rank ids in the tree's fixed left-to-right
    order; ``signature()`` is a stable string key (used by the checkpoint to
    detect a topology mismatch on restore).
    """

    kind: str
    rank_id: int | None
    left: "ReductionTree | None"
    right: "ReductionTree | None"

    @staticmethod
    def leaf(rank_id: int) -> "ReductionTree":
        return ReductionTree(kind="leaf", rank_id=int(rank_id), left=None, right=None)

    @staticmethod
    def node(left: "ReductionTree", right: "ReductionTree") -> "ReductionTree":
        return ReductionTree(kind="node", rank_id=None, left=left, right=right)

    def is_leaf(self) -> bool:
        return self.kind == "leaf"

    def leaves(self) -> list[int]:
        if self.is_leaf():
            if self.rank_id is None:
                raise RuntimeError("rank_id cannot be None in a leaf node")
            return [self.rank_id]
        if self.left is None or self.right is None:
            raise RuntimeError("internal node must have left and right children")
        return self.left.leaves() + self.right.leaves()

    def signature(self) -> str:
        """Stable parenthesized string of leaf ids in fixed fold order."""
        if self.is_leaf():
            return str(self.rank_id)
        if self.left is None or self.right is None:
            raise RuntimeError("internal node must have left and right children")
        return f"({self.left.signature()}+{self.right.signature()})"


def build_reduction_tree(rank_ids: Sequence[int]) -> ReductionTree:
    """Build the canonical fixed binary reduction tree over ``rank_ids``.

    The construction is deterministic and depends only on the *sorted* set of
    rank ids (so callers cannot perturb the fold order by handing the ids in a
    different sequence — a defining property of the cross-node discipline). The
    sorted leaves are folded pairwise bottom-up: adjacent pairs combine into the
    next level, with an odd trailing leaf carried up unchanged. This yields a
    balanced tree whose left subtree is always folded before the right, fixing
    the accumulation order before any partial is produced.
    """
    ids = sorted({int(r) for r in rank_ids})
    if not ids:
        raise ValueError("build_reduction_tree requires at least one rank id")
    level: list[ReductionTree] = [ReductionTree.leaf(r) for r in ids]
    while len(level) > 1:
        nxt: list[ReductionTree] = []
        i = 0
        while i + 1 < len(level):
            nxt.append(ReductionTree.node(level[i], level[i + 1]))
            i += 2
        if i < len(level):
            # Odd trailing subtree carried up unchanged (still left-of-nothing,
            # so it folds last at the next level — order stays fixed).
            nxt.append(level[i])
        level = nxt
    return level[0]


# ---------------------------------------------------------------------------
# Coercion bridge (torch optional)
# ---------------------------------------------------------------------------


def _to_numpy_f64(value: Any) -> np.ndarray:
    """Coerce a partial (numpy array / torch tensor / list) to ``float64`` numpy.

    Torch is optional: only if the value is a torch tensor do we route through
    the shared ``_coerce`` bridge, so this module imports and reduces with no
    torch build present.
    """
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float64)
    mod = type(value).__module__ or ""
    if mod.startswith("torch"):
        from ._coerce import to_numpy_f64

        return np.ascontiguousarray(to_numpy_f64(value), dtype=np.float64)
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


def _default_combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise add of two ``float64`` accumulators (the default monoid)."""
    if a.shape != b.shape:
        raise ValueError(
            f"reduction combine requires matching shapes; got {a.shape} and {b.shape}"
        )
    return a + b


# ---------------------------------------------------------------------------
# The reducer
# ---------------------------------------------------------------------------


class TreeReducer:
    """Bit-reproducible cross-node reduction over a fixed binary tree.

    Parameters
    ----------
    rank_ids
        The rank ids participating in the reduction. The fixed tree is built
        from these (sorted) ids, so the fold order is pinned independent of the
        order partials arrive.
    combine
        Associative-*intent* binary combiner ``(acc, acc) -> acc``. Defaults to
        elementwise ``float64`` addition. Because float addition is not exactly
        associative, the *tree order* (not the combiner) is what makes the
        result reproducible: :meth:`reduce` always evaluates the same fixed
        post-order regardless of arrival order.

    The reducer holds no per-call mutable network state; determinism is
    structural. The only state is the optional *checkpoint cache* populated by
    :meth:`reduce` (and restorable via :meth:`restore`), which records bit-exact
    subtree results keyed by their fixed signature so a resumed reduction reuses
    finished folds instead of recomputing them.
    """

    def __init__(
        self,
        rank_ids: Sequence[int],
        combine: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.tree = build_reduction_tree(rank_ids)
        self.rank_ids: tuple[int, ...] = tuple(self.tree.leaves())
        self._combine = combine if combine is not None else _default_combine
        # Checkpoint cache: signature -> bit-exact subtree accumulator.
        self._cache: dict[str, np.ndarray] = {}

    # -- core reduction ---------------------------------------------------

    def reduce(self, partials: Mapping[int, Any]) -> np.ndarray:
        """Reduce per-rank ``partials`` into the global accumulator.

        ``partials`` maps ``rank_id -> partial accumulator`` (numpy array, torch
        tensor, or array-like). Every rank id in the tree must be present.
        Evaluation is the fixed post-order of the tree: left subtree folded
        before right at every internal node, so the result is bit-identical for
        any permutation / arrival order of ``partials``. Subtree results are
        memoized into the checkpoint cache as they are produced.
        """
        coerced: dict[int, np.ndarray] = {}
        for rank in self.rank_ids:
            if rank not in partials:
                raise KeyError(
                    f"missing partial for rank {rank}; reduction requires all of "
                    f"{self.rank_ids}"
                )
            coerced[rank] = _to_numpy_f64(partials[rank])
        return self._reduce_subtree(self.tree, coerced)

    def _reduce_subtree(
        self, tree: ReductionTree, leaves: Mapping[int, np.ndarray]
    ) -> np.ndarray:
        sig = tree.signature()
        cached = self._cache.get(sig)
        if cached is not None:
            # Restartable: a finished subtree value is reused bit-for-bit rather
            # than recombined, so resume == from-scratch.
            return cached
        if tree.is_leaf():
            if tree.rank_id is None:
                raise RuntimeError("rank_id cannot be None in a leaf node")
            value = np.array(leaves[tree.rank_id], dtype=np.float64, copy=True)
        else:
            if tree.left is None or tree.right is None:
                raise RuntimeError("internal node must have left and right children")
            # Left ALWAYS before right — the fixed accumulation order.
            left_val = self._reduce_subtree(tree.left, leaves)
            right_val = self._reduce_subtree(tree.right, leaves)
            value = np.ascontiguousarray(
                self._combine(left_val, right_val), dtype=np.float64
            )
        self._cache[sig] = value
        return value

    # -- checkpoint / restore (checkpoint-as-job-model) -------------------

    def checkpoint(self) -> dict[str, Any]:
        """Serialize the reducer's finished-subtree cache to a plain dict.

        The returned dict is JSON-round-trippable (arrays are base64-encoded
        with their shape) and carries the tree's topology signature so
        :meth:`restore` can reject a checkpoint built for a different rank set.
        Each rank can checkpoint its own partial border / accumulator and any
        subtree it has finished folding; the reduction is restartable from any
        such checkpoint.
        """
        entries: list[dict[str, Any]] = []
        for sig, arr in self._cache.items():
            a = np.ascontiguousarray(arr, dtype=np.float64)
            entries.append(
                {
                    "signature": sig,
                    "shape": list(a.shape),
                    "data": base64.b64encode(a.tobytes()).decode("ascii"),
                }
            )
        return {
            "topology": self.tree.signature(),
            "rank_ids": list(self.rank_ids),
            "entries": entries,
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore the finished-subtree cache from a :meth:`checkpoint` dict.

        Raises if the checkpoint's topology signature does not match this
        reducer's tree (a guard against resuming a reduction with a different
        rank set, which would silently change the fold order). After restore the
        next :meth:`reduce` reuses every recorded subtree bit-for-bit and only
        evaluates the missing folds.
        """
        topo = checkpoint.get("topology")
        if topo != self.tree.signature():
            raise ValueError(
                "checkpoint topology mismatch: checkpoint was built for tree "
                f"{topo!r} but this reducer's tree is {self.tree.signature()!r}; "
                "resuming would change the accumulation order"
            )
        restored: dict[str, np.ndarray] = {}
        for entry in checkpoint.get("entries", ()):
            sig = str(entry["signature"])
            shape = tuple(int(s) for s in entry["shape"])
            raw = base64.b64decode(entry["data"])
            arr = np.frombuffer(raw, dtype=np.float64).reshape(shape).copy()
            restored[sig] = arr
        self._cache = restored

    def clear_cache(self) -> None:
        """Drop the finished-subtree cache (e.g. before reducing fresh partials)."""
        self._cache = {}
