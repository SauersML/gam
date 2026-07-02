"""Python facade for the Rust fixed-K sparse, minibatched SAE trainer (#1026).

This is the "collapsed linear lane": an additive path for very large
dictionaries (``K`` up to tens of thousands) where the exact-REML / Arrow-Schur
dense joint manifold solver is the wrong engine. It routes each row against the
dictionary in ``K``-tiles, keeps only the top-``active`` atoms, and returns
fixed-width **sparse** routing (``indices[N, active]`` / ``codes[N, active]``)
so the ``N x K`` assignment matrix is never materialised. All heavy state is
FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._binding import rust_module


def _as_2d_f32(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1-D or 2-D numeric array; got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} must be non-empty; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class SparseDictionaryFit:
    """Result of a collapsed-linear-lane fit.

    Attributes
    ----------
    decoder:
        ``K x P`` unit-norm decoder (one atom per row), FP32.
    indices:
        ``N x active`` active atom indices per row (``uint32``).
    codes:
        ``N x active`` sparse codes aligned with ``indices`` (FP32).
    fitted:
        ``N x P`` dense reconstruction of the training rows (FP32).
    explained_variance:
        Held-in EV (``1 - RSS/TSS``) of the fitted reconstruction.
    epochs, converged, active:
        Run metadata.
    """

    decoder: np.ndarray
    indices: np.ndarray
    codes: np.ndarray
    fitted: np.ndarray
    explained_variance: float
    epochs: int
    converged: bool
    active: int

    def reconstruct(self, indices: Any | None = None, codes: Any | None = None) -> np.ndarray:
        """Dense reconstruct from a sparse ``(indices, codes)`` routing.

        Defaults to the training routing. ``indices`` / ``codes`` must be
        ``N x active`` with matching shapes.
        """
        idx = self.indices if indices is None else np.asarray(indices, dtype=np.uint32)
        cod = self.codes if codes is None else np.asarray(codes, dtype=np.float32)
        if idx.shape != cod.shape:
            raise ValueError(f"indices {idx.shape} and codes {cod.shape} must match")
        n, s = idx.shape
        p = self.decoder.shape[1]
        out = np.zeros((n, p), dtype=np.float32)
        for j in range(s):
            out += cod[:, [j]] * self.decoder[idx[:, j]]
        return np.ascontiguousarray(out)

    def transform(self, X: Any, active: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Route held-out rows ``X`` (``M x P``) through the fitted decoder.

        Returns ``(indices, codes)`` of shape ``M x active`` (sparse routing),
        computed by the Rust core (``sparse_dictionary_transform``): the same
        tiled top-``active`` routing + active-set ridge solve the trainer uses,
        against the frozen decoder.
        """
        x = _as_2d_f32(X, "X")
        if x.shape[1] != self.decoder.shape[1]:
            raise ValueError(
                f"X must have P={self.decoder.shape[1]} columns; got {x.shape[1]}"
            )
        s = self.active if active is None else int(active)
        s = max(1, min(s, self.decoder.shape[0]))
        indices, codes = rust_module().sparse_dictionary_transform_ffi(
            np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(self.decoder, dtype=np.float32),
            int(s),
        )
        return np.ascontiguousarray(indices), np.ascontiguousarray(codes)


@dataclass(frozen=True)
class SparseDictStreamArtifact:
    """Result of a streaming (partial-fit) collapsed-linear-lane fit.

    The streaming path never materialises an ``N x s`` routing (a streamed corpus
    is re-encoded shard-by-shard through the frozen decoder), so — unlike
    :class:`SparseDictionaryFit` — this artifact carries only the trained decoder
    plus run metadata. Call :meth:`transform` to encode any held-out (or replayed
    training) rows against it.

    Attributes
    ----------
    decoder:
        ``K x P`` unit-norm decoder (one atom per row), FP32.
    explained_variance:
        EV of the final epoch's pass (the pre-refresh decoder of the last epoch);
        for a converged fit this equals the returned decoder's EV to tolerance.
    epochs, converged, active:
        Run metadata.
    """

    decoder: np.ndarray
    explained_variance: float
    epochs: int
    converged: bool
    active: int

    def transform(self, X: Any, active: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Route held-out rows ``X`` (``M x P``) through the fitted decoder.

        Returns ``(indices, codes)`` of shape ``M x active`` (sparse routing),
        computed by the Rust core against the frozen decoder — the same tiled
        top-``active`` routing + active-set ridge solve the trainer uses.
        """
        x = _as_2d_f32(X, "X")
        if x.shape[1] != self.decoder.shape[1]:
            raise ValueError(
                f"X must have P={self.decoder.shape[1]} columns; got {x.shape[1]}"
            )
        s = self.active if active is None else int(active)
        s = max(1, min(s, self.decoder.shape[0]))
        indices, codes = rust_module().sparse_dictionary_transform_ffi(
            np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(self.decoder, dtype=np.float32),
            int(s),
        )
        return np.ascontiguousarray(indices), np.ascontiguousarray(codes)


class SparseDictStream:
    """Partial-fit streaming surface for the collapsed linear lane (#1026).

    Wraps the native ``SparseDictStreamState`` handle so a Python loop can stream
    epochs over shards of a corpus that never fits in memory at once
    (``K`` up to tens of thousands over tens of millions of tokens). All heavy
    state — the warm-started decoder, the epoch's accumulated decoder normal
    equations, the dead-atom revival reservoir — lives native-side; a shard
    round-trips only its own ``shard x P`` rows through Python, so per-shard
    overhead is independent of ``K`` and of the corpus length.

    Usage mirrors an ordinary minibatch trainer::

        stream = SparseDictStream(seed_sample, K, active=8)
        for _epoch in range(max_epochs):
            for shard in shards:
                stream.partial_fit(shard)
            stats = stream.end_epoch()
            if stats["converged"]:
                break
        artifact = stream.finalize()

    Parameters
    ----------
    seed:
        A representative ``N_seed x P`` sample used to fix ``P`` and seed the
        initial atom directions (deterministic farthest-point). One shard, or the
        whole corpus for small problems.
    K, active, minibatch, max_epochs, score_tile, code_ridge, decoder_ridge, tolerance:
        Identical hyper-parameters to :func:`sparse_dictionary_fit`. ``max_epochs``
        is advisory here (the driving Python loop decides how many epochs to run);
        it is carried only so :meth:`end_epoch`'s convergence flag matches the
        one-shot stopping rule.
    """

    def __init__(
        self,
        seed: Any,
        K: int,
        *,
        active: int = 1,
        minibatch: int = 512,
        max_epochs: int = 30,
        score_tile: int = 4096,
        code_ridge: float = 1.0e-6,
        decoder_ridge: float = 1.0e-6,
        tolerance: float = 1.0e-6,
    ) -> None:
        seed_arr = _as_2d_f32(seed, "seed")
        self._handle = rust_module().SparseDictStream(
            seed_arr,
            int(K),
            active=int(active),
            minibatch=int(minibatch),
            max_epochs=int(max_epochs),
            score_tile=int(score_tile),
            code_ridge=float(code_ridge),
            decoder_ridge=float(decoder_ridge),
            tolerance=float(tolerance),
        )

    def partial_fit(self, shard: Any) -> dict[str, Any]:
        """Route + sparse-code one ``shard`` (``M x P``) against the current
        decoder and fold it into the running epoch.

        Returns per-shard stats ``{rows, rss, alive_atoms}`` (``alive_atoms`` is
        cumulative across the shards seen since the last :meth:`end_epoch`).
        """
        shard_arr = _as_2d_f32(shard, "shard")
        return dict(self._handle.partial_fit(shard_arr))

    def end_epoch(self) -> dict[str, Any]:
        """Close the current epoch: refresh the decoder from the accumulated
        normal equations, revive dead atoms onto worst-reconstructed residual
        rows, and reset the epoch accumulators.

        Returns ``{explained_variance, revived, dead, converged, epoch}`` where
        ``explained_variance`` is the EV of the decoder routed against this epoch
        (pre-refresh) and ``converged`` follows the one-shot stopping rule (an EV
        plateau with no atom revived).
        """
        return dict(self._handle.end_epoch())

    def finalize(self) -> SparseDictStreamArtifact:
        """Hand back the trained decoder plus run metadata as a
        :class:`SparseDictStreamArtifact`."""
        data = dict(self._handle.finalize())
        return SparseDictStreamArtifact(
            decoder=np.ascontiguousarray(data["decoder"], dtype=np.float32),
            explained_variance=float(data["explained_variance"]),
            epochs=int(data["epochs"]),
            converged=bool(data["converged"]),
            active=int(data["active"]),
        )

    @property
    def decoder(self) -> np.ndarray:
        """A live copy of the current warm-started decoder (``K x P``, unit-norm)."""
        return np.ascontiguousarray(self._handle.decoder(), dtype=np.float32)

    @property
    def active(self) -> int:
        """Active budget ``s`` in use (``min(active, K)``)."""
        return int(self._handle.active)

    @property
    def epochs_run(self) -> int:
        """Epochs closed so far."""
        return int(self._handle.epochs_run)


@dataclass(frozen=True)
class BlockSparseDictionaryFit:
    """Result of a block-sparse fit (#1026 block extension).

    The ``K = G*b`` atoms are grouped into ``G`` blocks of ``b`` orthonormal
    atoms. Routing selects whole blocks by their group ℓ₂ gate ``‖z_g‖₂``
    (block-TopK, signed codes, no ReLU); each block is a Stiefel-constrained
    frame. **Presence** (:attr:`gates`) and **amplitude** (:attr:`codes`) are
    deliberately separate arrays — the decoupling this lane is built around — and
    every routing decision is invariant to each block's internal ``O(b)`` gauge.

    Attributes
    ----------
    decoder:
        ``K x P`` decoder (``K = G*b``); block ``g`` occupies rows
        ``[g*b, g*b+b)`` and its ``b`` rows are orthonormal (``D_g D_gᵀ = I_b``).
    blocks:
        ``N x block_topk`` selected block indices per row (``uint32``).
    gates:
        ``N x block_topk`` per-selected-block **gate** ``‖z_g‖₂`` (presence, FP32).
    codes:
        ``N x block_topk x b`` signed **within-block code** ``z_g`` (amplitude /
        direction, FP32), aligned with :attr:`blocks`.
    gamma:
        Shared tied-encoder scalar ``γ`` (one scalar for the whole dictionary).
    block_utilization:
        Length-``G`` fraction of rows that selected each block.
    block_stable_rank:
        Length-``G`` stable rank ``trace(C_g)/λ_max(C_g)`` of each block's
        within-block code second moment — the effective dimensionality each block
        uses (for the MDL lane).
    fitted:
        ``N x P`` dense reconstruction of the training rows (FP32).
    explained_variance, epochs, converged, block_topk, block_size:
        Run metadata.
    """

    decoder: np.ndarray
    blocks: np.ndarray
    gates: np.ndarray
    codes: np.ndarray
    gamma: float
    block_utilization: np.ndarray
    block_stable_rank: np.ndarray
    fitted: np.ndarray
    explained_variance: float
    epochs: int
    converged: bool
    block_topk: int
    block_size: int

    @property
    def n_blocks(self) -> int:
        """Number of blocks ``G = K / b``."""
        return self.decoder.shape[0] // self.block_size

    def reconstruct(self) -> np.ndarray:
        """Dense ``N x P`` reconstruction from the stored block routing
        (``x̂_i = Σ_g z_{ig} D_g``)."""
        n = self.blocks.shape[0]
        p = self.decoder.shape[1]
        b = self.block_size
        out = np.zeros((n, p), dtype=np.float32)
        for j in range(self.block_topk):
            g = self.blocks[:, j].astype(np.int64)  # N
            for r in range(b):
                atoms = self.decoder[g * b + r]  # N x P
                out += self.codes[:, j, r][:, None] * atoms
        return np.ascontiguousarray(out)

    def transform(
        self, X: Any, block_topk: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Route held-out rows ``X`` (``M x P``) through the fitted block frames.

        Returns ``(blocks, gates, codes)`` of shapes ``M x k`` / ``M x k`` /
        ``M x k x b`` — the same block-TopK routing (by group ℓ₂ gate) and tied
        signed codes ``z_g = γ x D_gᵀ`` the trainer uses, against the frozen
        frames. Pure block routing (no least-squares), so it is exactly the
        encoder the fit learned. ``k`` defaults to the fitted ``block_topk``.
        """
        x = _as_2d_f32(X, "X")
        if x.shape[1] != self.decoder.shape[1]:
            raise ValueError(
                f"X must have P={self.decoder.shape[1]} columns; got {x.shape[1]}"
            )
        b = self.block_size
        g_blocks = self.n_blocks
        k = self.block_topk if block_topk is None else int(block_topk)
        k = max(1, min(k, g_blocks))
        # Raw tied projections w[m, g, r] = x_m · D_{g,r}.
        w = (x @ self.decoder.T).reshape(x.shape[0], g_blocks, b)  # M x G x b
        gate = np.linalg.norm(w, axis=2)  # M x G  (γ-free routing gate)
        # Block-TopK by gate (ties broken toward smaller block index for
        # determinism, matching the Rust online selector).
        order = np.argsort(-gate, axis=1, kind="stable")[:, :k]  # M x k
        blocks = order.astype(np.uint32)
        rows = np.arange(x.shape[0])[:, None]
        gates = float(abs(self.gamma)) * gate[rows, order]  # M x k presence ‖z_g‖
        codes = float(self.gamma) * w[rows, order, :]  # M x k x b signed amplitude
        return (
            np.ascontiguousarray(blocks),
            np.ascontiguousarray(gates.astype(np.float32)),
            np.ascontiguousarray(codes.astype(np.float32)),
        )


def block_sparse_dictionary_fit(
    X: Any,
    n_blocks: int,
    *,
    block_size: int = 2,
    block_topk: int = 1,
    grassmann: bool = True,
    max_epochs: int = 30,
    minibatch: int = 512,
    block_tile: int = 1024,
    frame_ridge: float = 1.0e-9,
    aux_k: int = 0,
    tolerance: float = 1.0e-6,
) -> BlockSparseDictionaryFit:
    """Fit a **block-sparse** dictionary to ``X`` (``N x P``): ``G = n_blocks``
    blocks of ``b = block_size`` orthonormal atoms (``K = G*b``), block-TopK
    routing by group ℓ₂ gate, tied signed codes with one shared scalar ``γ``,
    Stiefel-constrained frames refreshed by polar steps, and AuxK dead-block
    revival seeded from worst-reconstructed residual rows.

    Parameters
    ----------
    n_blocks:
        Number of blocks ``G``. The dictionary has ``K = G * block_size`` atoms.
    block_size:
        Atoms per block ``b`` (the subspace dimension). Typically 2–4; must not
        exceed ``P``.
    block_topk:
        Block routing budget ``k`` (blocks allowed to fire per row).
    grassmann:
        Block frames are Grassmann/Stiefel-constrained (column-orthonormal).
        This lane is always frame-constrained; ``grassmann=False`` is rejected
        (use :func:`sparse_dictionary_fit` for an unconstrained atom dictionary).
    max_epochs, minibatch, block_tile:
        Streaming / tiling controls (peak routing working set is
        ``minibatch x (block_tile*b)``, never ``N x K``).
    frame_ridge, aux_k, tolerance:
        Frame-refresh ridge, AuxK dead-block revival budget, and the EV stopping
        tolerance.
    """
    if not grassmann:
        raise ValueError(
            "block_sparse_dictionary_fit is always Grassmann/Stiefel-constrained; "
            "pass grassmann=True (or use sparse_dictionary_fit for an unconstrained "
            "atom dictionary)"
        )
    x = _as_2d_f32(X, "X")
    if block_size > x.shape[1]:
        raise ValueError(
            f"block_size={block_size} cannot exceed P={x.shape[1]} "
            "(a block's b orthonormal rows must fit in R^P)"
        )
    payload = rust_module().block_sparse_dictionary_fit(
        x,
        int(n_blocks),
        block_size=int(block_size),
        block_topk=int(block_topk),
        max_epochs=int(max_epochs),
        minibatch=int(minibatch),
        block_tile=int(block_tile),
        frame_ridge=float(frame_ridge),
        aux_k=int(aux_k),
        tolerance=float(tolerance),
    )
    data = dict(payload)
    return BlockSparseDictionaryFit(
        decoder=np.ascontiguousarray(data["decoder"], dtype=np.float32),
        blocks=np.ascontiguousarray(data["blocks"], dtype=np.uint32),
        gates=np.ascontiguousarray(data["gates"], dtype=np.float32),
        codes=np.ascontiguousarray(data["codes"], dtype=np.float32),
        gamma=float(data["gamma"]),
        block_utilization=np.ascontiguousarray(data["block_utilization"], dtype=np.float32),
        block_stable_rank=np.ascontiguousarray(data["block_stable_rank"], dtype=np.float32),
        fitted=np.ascontiguousarray(data["fitted"], dtype=np.float32),
        explained_variance=float(data["explained_variance"]),
        epochs=int(data["epochs"]),
        converged=bool(data["converged"]),
        block_topk=int(data["block_topk"]),
        block_size=int(data["block_size"]),
    )


def sparse_dictionary_fit_begin(
    seed: Any,
    K: int,
    *,
    active: int = 1,
    minibatch: int = 512,
    max_epochs: int = 30,
    score_tile: int = 4096,
    code_ridge: float = 1.0e-6,
    decoder_ridge: float = 1.0e-6,
    tolerance: float = 1.0e-6,
) -> SparseDictStream:
    """Begin a streaming sparse-dictionary fit and return a :class:`SparseDictStream`.

    Thin functional alias for ``SparseDictStream(seed, K, ...)`` mirroring the
    ``fit_begin`` / ``partial_fit`` / ``finalize`` surface.
    """
    return SparseDictStream(
        seed,
        K,
        active=active,
        minibatch=minibatch,
        max_epochs=max_epochs,
        score_tile=score_tile,
        code_ridge=code_ridge,
        decoder_ridge=decoder_ridge,
        tolerance=tolerance,
    )


def sparse_dictionary_fit(
    X: Any,
    K: int,
    *,
    active: int = 1,
    minibatch: int = 512,
    max_epochs: int = 30,
    score_tile: int = 4096,
    code_ridge: float = 1.0e-6,
    decoder_ridge: float = 1.0e-6,
    tolerance: float = 1.0e-6,
) -> SparseDictionaryFit:
    """Fit a fixed-``K`` sparse, minibatched linear dictionary to ``X`` (``N x P``).

    Parameters
    ----------
    K:
        Dictionary width (number of atoms). May be very large.
    active:
        Routing sparsity ``s`` (atoms allowed to fire per row). Shared, not
        per-atom.
    minibatch, max_epochs, score_tile:
        Streaming / tiling controls.
    code_ridge, decoder_ridge, tolerance:
        Shared regularisation and stopping controls.
    """
    x = _as_2d_f32(X, "X")
    payload = rust_module().sparse_dictionary_fit(
        x,
        int(K),
        active=int(active),
        minibatch=int(minibatch),
        max_epochs=int(max_epochs),
        score_tile=int(score_tile),
        code_ridge=float(code_ridge),
        decoder_ridge=float(decoder_ridge),
        tolerance=float(tolerance),
    )
    data = dict(payload)
    return SparseDictionaryFit(
        decoder=np.ascontiguousarray(data["decoder"], dtype=np.float32),
        indices=np.ascontiguousarray(data["indices"], dtype=np.uint32),
        codes=np.ascontiguousarray(data["codes"], dtype=np.float32),
        fitted=np.ascontiguousarray(data["fitted"], dtype=np.float32),
        explained_variance=float(data["explained_variance"]),
        epochs=int(data["epochs"]),
        converged=bool(data["converged"]),
        active=int(data["active"]),
    )


__all__ = [
    "BlockSparseDictionaryFit",
    "SparseDictStream",
    "SparseDictStreamArtifact",
    "SparseDictionaryFit",
    "block_sparse_dictionary_fit",
    "sparse_dictionary_fit",
    "sparse_dictionary_fit_begin",
]
