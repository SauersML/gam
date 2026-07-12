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
Plain :meth:`TreeReducer.reduce` calls are stateless. Restartability is explicit:
:meth:`TreeReducer.start_session` snapshots one job's exact rank payloads and
returns a :class:`ReductionSession`. Its checkpoint is bound to the caller's job
id, the tree topology, the combiner contract, and a SHA-256 fingerprint of every
coerced input byte. A checkpoint from another job or another set of partials is
rejected before any cached subtree can be reused.

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
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "ReductionSession",
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
    raw_ids = [
        _coerce_rank_id(rank, context="build_reduction_tree") for rank in rank_ids
    ]
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("build_reduction_tree rank ids must be unique")
    ids = sorted(raw_ids)
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
# Stateless reducer + explicitly job-bound checkpoint sessions
# ---------------------------------------------------------------------------


_CHECKPOINT_SCHEMA = "gamfit.reduction-session.v2"
_DEFAULT_COMBINE_ID = "elementwise-f64-sum.v1"
_CANONICAL_DTYPE = np.dtype("<f8")
_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "job_id",
        "topology",
        "rank_ids",
        "combine_id",
        "input_fingerprint",
        "entries",
        "cache_fingerprint",
    }
)
_ENTRY_KEYS = frozenset({"signature", "shape", "dtype", "data"})


def _coerce_rank_id(value: Any, *, context: str) -> int:
    """Return an exact integral rank id without lossy Python coercions."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(
            f"{context} rank ids must be integers (not bools or floats); "
            f"got {value!r}"
        )
    return int(value)


def _coerce_partials(
    rank_ids: Sequence[int], partials: Mapping[int, Any]
) -> dict[int, np.ndarray]:
    normalized: dict[int, Any] = {}
    for raw_rank, value in partials.items():
        rank = _coerce_rank_id(raw_rank, context="reduction partial")
        if rank in normalized:
            raise ValueError(
                f"reduction partial rank ids collide after normalization: {rank}"
            )
        normalized[rank] = value
    expected = set(rank_ids)
    actual = set(normalized)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected, key=repr)
    if missing or unexpected:
        raise ValueError(
            "reduction partial rank set mismatch: "
            f"missing={missing}, unexpected={unexpected}, expected={tuple(rank_ids)}"
        )
    # A session owns an immutable snapshot.  Copy even an already-contiguous f64
    # input so caller mutation after start_session cannot change the job identity.
    return {
        rank: np.array(
            _to_numpy_f64(normalized[rank]), dtype=np.float64, copy=True, order="C"
        )
        for rank in rank_ids
    }


def _hash_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "little", signed=False))
    digest.update(value)


def _canonical_f64_bytes(array: np.ndarray) -> bytes:
    """Encode a float64 array as canonical C-order little-endian bytes."""
    native = np.ascontiguousarray(array, dtype=np.float64)
    return native.astype(_CANONICAL_DTYPE, copy=False).tobytes(order="C")


def _arrays_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and (
        _canonical_f64_bytes(left) == _canonical_f64_bytes(right)
    )


def _input_fingerprint(
    *,
    job_id: str,
    topology: str,
    combine_id: str,
    rank_ids: Sequence[int],
    leaves: Mapping[int, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    _hash_field(digest, _CHECKPOINT_SCHEMA.encode("utf-8"))
    _hash_field(digest, job_id.encode("utf-8"))
    _hash_field(digest, topology.encode("utf-8"))
    _hash_field(digest, combine_id.encode("utf-8"))
    _hash_field(digest, _CANONICAL_DTYPE.str.encode("ascii"))
    for rank in rank_ids:
        array = np.ascontiguousarray(leaves[rank], dtype=np.float64)
        _hash_field(digest, str(rank).encode("ascii"))
        _hash_field(
            digest,
            ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        )
        _hash_field(digest, _canonical_f64_bytes(array))
    return digest.hexdigest()


def _tree_signatures(tree: ReductionTree) -> set[str]:
    signatures = {tree.signature()}
    if not tree.is_leaf():
        if tree.left is None or tree.right is None:
            raise RuntimeError("internal node must have left and right children")
        signatures.update(_tree_signatures(tree.left))
        signatures.update(_tree_signatures(tree.right))
    return signatures


def _reduce_subtree(
    tree: ReductionTree,
    leaves: Mapping[int, np.ndarray],
    combine: Callable[[np.ndarray, np.ndarray], np.ndarray],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    signature = tree.signature()
    cached = cache.get(signature)
    if cached is not None:
        return cached
    if tree.is_leaf():
        if tree.rank_id is None:
            raise RuntimeError("rank_id cannot be None in a leaf node")
        value = np.array(leaves[tree.rank_id], dtype=np.float64, copy=True, order="C")
    else:
        if tree.left is None or tree.right is None:
            raise RuntimeError("internal node must have left and right children")
        # Left ALWAYS before right — the fixed accumulation order.
        left_value = _reduce_subtree(tree.left, leaves, combine, cache)
        right_value = _reduce_subtree(tree.right, leaves, combine, cache)
        # A custom combiner never receives cache-owned storage. It may mutate
        # either argument or return one of them without corrupting a leaf or a
        # previously-computed subtree. Copy its result as well so references
        # retained by user code cannot mutate the cache after return.
        combined = combine(
            np.array(left_value, dtype=np.float64, copy=True, order="C"),
            np.array(right_value, dtype=np.float64, copy=True, order="C"),
        )
        value = np.array(combined, dtype=np.float64, copy=True, order="C")
    cache[signature] = value
    return value


def _encoded_cache_entries(cache: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for signature in sorted(cache):
        array = np.ascontiguousarray(cache[signature], dtype=np.float64)
        entries.append(
            {
                "signature": signature,
                "shape": list(array.shape),
                "dtype": _CANONICAL_DTYPE.str,
                "data": base64.b64encode(_canonical_f64_bytes(array)).decode("ascii"),
            }
        )
    return entries


def _cache_fingerprint(input_fingerprint: str, entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    _hash_field(digest, input_fingerprint.encode("ascii"))
    for entry in entries:
        _hash_field(digest, str(entry["signature"]).encode("utf-8"))
        _hash_field(
            digest,
            ",".join(str(int(size)) for size in entry["shape"]).encode("ascii"),
        )
        _hash_field(digest, entry["dtype"].encode("ascii"))
        _hash_field(digest, str(entry["data"]).encode("ascii"))
    return digest.hexdigest()


def _require_exact_keys(
    mapping: Mapping[Any, Any], expected: frozenset[str], *, context: str
) -> None:
    """Reject missing, unknown, non-string, and repeated object keys."""
    keys = list(mapping.keys())
    if any(not isinstance(key, str) for key in keys):
        raise ValueError(f"{context} keys must all be strings")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} repeats an object key")
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{context} keys mismatch: missing={missing}, unknown={unknown}"
        )


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"reduction checkpoint JSON repeats key {key!r}")
        result[key] = value
    return result


class TreeReducer:
    """Immutable topology and combiner for bit-reproducible reductions.

    :meth:`reduce` allocates a fresh per-call cache and therefore never reuses a
    value from an earlier input. For resumability, call :meth:`start_session` and
    checkpoint that explicit input-bound :class:`ReductionSession`.

    A custom ``combine`` must provide a stable ``combine_id``. The id is included
    in every session fingerprint so a checkpoint cannot be restored under a
    different reduction algebra.
    """

    def __init__(
        self,
        rank_ids: Sequence[int],
        combine: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        *,
        combine_id: str | None = None,
    ) -> None:
        self.tree = build_reduction_tree(rank_ids)
        self.rank_ids: tuple[int, ...] = tuple(self.tree.leaves())
        if combine is None:
            if combine_id is not None:
                raise ValueError("combine_id is only valid with a custom combine function")
            self._combine = _default_combine
            self.combine_id = _DEFAULT_COMBINE_ID
        else:
            if not isinstance(combine_id, str) or not combine_id.strip():
                raise ValueError(
                    "a custom combine function requires a non-empty stable combine_id"
                )
            self._combine = combine
            self.combine_id = combine_id

    def reduce(self, partials: Mapping[int, Any]) -> np.ndarray:
        """Pure reduction of one exact rank map using a fresh local cache."""
        leaves = _coerce_partials(self.rank_ids, partials)
        result = _reduce_subtree(self.tree, leaves, self._combine, {})
        return np.array(result, dtype=np.float64, copy=True, order="C")

    def start_session(
        self, partials: Mapping[int, Any], *, job_id: str
    ) -> "ReductionSession":
        """Snapshot one logical job for explicit checkpoint/resume semantics."""
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("reduction session job_id must be a non-empty string")
        return ReductionSession(self, _coerce_partials(self.rank_ids, partials), job_id)


class ReductionSession:
    """Mutable checkpoint state bound to one reducer and one immutable input.

    Construct sessions only through :meth:`TreeReducer.start_session`. The input
    arrays are copied at construction and fingerprinted with the explicit job id.
    ``restore`` accepts only the exact same job, payload bytes, topology, and
    combiner contract.
    """

    def __init__(
        self,
        reducer: TreeReducer,
        leaves: Mapping[int, np.ndarray],
        job_id: str,
    ) -> None:
        self.tree = reducer.tree
        self.rank_ids = reducer.rank_ids
        self.combine_id = reducer.combine_id
        self.job_id = job_id
        self._combine = reducer._combine
        self._leaves = {
            rank: np.array(leaves[rank], dtype=np.float64, copy=True, order="C")
            for rank in self.rank_ids
        }
        self.input_fingerprint = _input_fingerprint(
            job_id=self.job_id,
            topology=self.tree.signature(),
            combine_id=self.combine_id,
            rank_ids=self.rank_ids,
            leaves=self._leaves,
        )
        self._cache: dict[str, np.ndarray] = {}

    def reduce(self) -> np.ndarray:
        """Finish this session, reusing only cache entries bound to this input."""
        result = _reduce_subtree(
            self.tree, self._leaves, self._combine, self._cache
        )
        # Do not expose mutable cache storage to callers.
        return np.array(result, dtype=np.float64, copy=True, order="C")

    def checkpoint(self) -> dict[str, Any]:
        """Return a JSON-round-trippable, input-bound session checkpoint."""
        entries = _encoded_cache_entries(self._cache)
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "job_id": self.job_id,
            "topology": self.tree.signature(),
            "rank_ids": list(self.rank_ids),
            "combine_id": self.combine_id,
            "input_fingerprint": self.input_fingerprint,
            "entries": entries,
            "cache_fingerprint": _cache_fingerprint(
                self.input_fingerprint, entries
            ),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """Transactionally restore a checkpoint for this exact session only."""
        if not isinstance(checkpoint, Mapping):
            raise ValueError("reduction checkpoint must be a mapping")
        _require_exact_keys(
            checkpoint, _CHECKPOINT_KEYS, context="reduction checkpoint"
        )
        if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError(
                f"unsupported reduction checkpoint schema {checkpoint.get('schema')!r}; "
                f"expected {_CHECKPOINT_SCHEMA!r}"
            )
        if checkpoint.get("job_id") != self.job_id:
            raise ValueError(
                "reduction checkpoint job mismatch: "
                f"checkpoint={checkpoint.get('job_id')!r}, session={self.job_id!r}"
            )
        if checkpoint.get("topology") != self.tree.signature():
            raise ValueError("reduction checkpoint topology mismatch")
        checkpoint_rank_ids = checkpoint.get("rank_ids")
        if not isinstance(checkpoint_rank_ids, list) or any(
            isinstance(rank, bool) or type(rank) is not int
            for rank in checkpoint_rank_ids
        ):
            raise ValueError("reduction checkpoint rank_ids must be a list of integers")
        if checkpoint_rank_ids != list(self.rank_ids):
            raise ValueError("reduction checkpoint rank-id mismatch")
        if checkpoint.get("combine_id") != self.combine_id:
            raise ValueError("reduction checkpoint combiner mismatch")
        if checkpoint.get("input_fingerprint") != self.input_fingerprint:
            raise ValueError(
                "reduction checkpoint input fingerprint mismatch; refusing to "
                "reuse cached values for different partials"
            )

        raw_entries = checkpoint.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("reduction checkpoint entries must be a list")
        if not all(isinstance(entry, Mapping) for entry in raw_entries):
            raise ValueError("reduction checkpoint entry must be a mapping")

        normalized_entries: list[dict[str, Any]] = []
        decoded_entries: list[tuple[str, tuple[int, ...], bytes]] = []
        for entry in raw_entries:
            _require_exact_keys(
                entry, _ENTRY_KEYS, context="reduction checkpoint entry"
            )
            signature = entry["signature"]
            if not isinstance(signature, str):
                raise ValueError("reduction checkpoint entry signature must be a string")
            raw_shape = entry["shape"]
            if not isinstance(raw_shape, list) or any(
                isinstance(size, bool) or type(size) is not int
                for size in raw_shape
            ):
                raise ValueError(
                    f"reduction checkpoint entry {signature!r} shape must be "
                    "a list of integers"
                )
            shape = tuple(raw_shape)
            if any(size < 0 for size in shape):
                raise ValueError(
                    f"reduction checkpoint entry {signature!r} has a negative shape"
                )
            if entry["dtype"] != _CANONICAL_DTYPE.str or not isinstance(
                entry["dtype"], str
            ):
                raise ValueError(
                    f"reduction checkpoint entry {signature!r} dtype must be "
                    f"{_CANONICAL_DTYPE.str!r}"
                )
            data = entry["data"]
            if not isinstance(data, str):
                raise ValueError(
                    f"reduction checkpoint entry {signature!r} data must be a string"
                )
            try:
                data.encode("ascii")
                raw = base64.b64decode(data, validate=True)
            except (UnicodeEncodeError, ValueError) as exc:
                raise ValueError(
                    f"malformed reduction checkpoint entry {signature!r} data"
                ) from exc
            expected_bytes = math.prod(shape) * _CANONICAL_DTYPE.itemsize
            if len(raw) != expected_bytes:
                raise ValueError(
                    f"reduction checkpoint subtree {signature!r} has {len(raw)} "
                    f"bytes; expected {expected_bytes} for shape {shape}"
                )
            normalized_entries.append(
                {
                    "signature": signature,
                    "shape": list(shape),
                    "dtype": _CANONICAL_DTYPE.str,
                    "data": data,
                }
            )
            decoded_entries.append((signature, shape, raw))

        try:
            cache_fingerprint = _cache_fingerprint(
                self.input_fingerprint, normalized_entries
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed reduction checkpoint entries") from exc
        if checkpoint.get("cache_fingerprint") != cache_fingerprint:
            raise ValueError("reduction checkpoint cache fingerprint mismatch")

        allowed_signatures = _tree_signatures(self.tree)
        restored: dict[str, np.ndarray] = {}
        for signature, shape, raw in decoded_entries:
            if signature not in allowed_signatures:
                raise ValueError(
                    f"reduction checkpoint contains unknown subtree {signature!r}"
                )
            if signature in restored:
                raise ValueError(
                    f"reduction checkpoint repeats subtree {signature!r}"
                )
            restored[signature] = (
                np.frombuffer(raw, dtype=_CANONICAL_DTYPE)
                .astype(np.float64, copy=False)
                .reshape(shape)
                .copy(order="C")
            )

        # A cached leaf must be the exact session input. Internal cache entries
        # are protected by the checkpoint digest and the input fingerprint.
        for rank in self.rank_ids:
            signature = str(rank)
            cached_leaf = restored.get(signature)
            if cached_leaf is not None and not _arrays_bitwise_equal(
                cached_leaf, self._leaves[rank]
            ):
                raise ValueError(
                    f"reduction checkpoint leaf {rank} does not match session input"
                )

        # The adjacent SHA-256 detects accidental damage, but is not an
        # authenticity primitive: a producer can rewrite both an internal
        # value and its digest. Validate every restored subtree against a fresh
        # evaluation from the immutable leaves under the declared combiner.
        # `_reduce_subtree` isolates combiner mutations, so validation itself
        # cannot alter session inputs or existing cache state.
        non_leaf_signatures = set(restored) - {str(rank) for rank in self.rank_ids}
        if non_leaf_signatures:
            expected_cache: dict[str, np.ndarray] = {}
            _reduce_subtree(
                self.tree, self._leaves, self._combine, expected_cache
            )
            for signature in non_leaf_signatures:
                if not _arrays_bitwise_equal(
                    restored[signature], expected_cache[signature]
                ):
                    raise ValueError(
                        f"reduction checkpoint subtree {signature!r} does not "
                        "match deterministic reduction"
                    )

        self._cache = restored

    def restore_json(self, payload: str | bytes | bytearray) -> None:
        """Parse and restore JSON while rejecting duplicate object keys."""
        if not isinstance(payload, (str, bytes, bytearray)):
            raise TypeError("reduction checkpoint JSON must be str or bytes")
        try:
            checkpoint = json.loads(
                payload, object_pairs_hook=_json_object_without_duplicates
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("malformed reduction checkpoint JSON") from exc
        if not isinstance(checkpoint, Mapping):
            raise ValueError("reduction checkpoint JSON root must be an object")
        self.restore(checkpoint)
