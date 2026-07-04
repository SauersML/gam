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


def _route_stats(payload: Any) -> dict[str, Any]:
    stats = dict(payload or {})
    for key in (
        "minibatches",
        "admitted_minibatches",
        "device_minibatches",
        "cpu_minibatches",
        "score_tiles",
        "peak_score_bytes",
    ):
        if key in stats:
            stats[key] = int(stats[key])
    for key in (
        "score_elements",
        "dot_flops_lower_bound",
        "device_dtoh_bytes",
        "unfused_score_dtoh_bytes_avoided",
    ):
        if key in stats:
            stats[key] = int(stats[key])
    return stats


def _block_transform(
    decoder: np.ndarray,
    gamma: float,
    block_size: int,
    block_topk: int,
    X: Any,
    *,
    block_tile: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block out-of-sample routing ``(blocks, gates, codes)`` for held-out rows.

    Delegates to the Rust core (``block_sparse_dictionary_transform_ffi``) — the same
    group-ℓ₂ gate + block-TopK + tied signed codes the trainer uses, so held-out
    encoding is bit-consistent with training (SPEC rule 8, python-thin). Falls back
    to an equivalent numpy path ONLY when the compiled extension predates the symbol
    or is unavailable (stale-build tolerance); the two agree up to f32 rounding.
    """
    x = _as_2d_f32(X, "X")
    p = decoder.shape[1]
    if x.shape[1] != p:
        raise ValueError(f"X must have P={p} columns; got {x.shape[1]}")
    b = int(block_size)
    g_blocks = decoder.shape[0] // b
    k = max(1, min(int(block_topk), g_blocks))
    fn = None
    try:
        fn = getattr(rust_module(), "block_sparse_dictionary_transform_ffi", None)
    except Exception:
        fn = None  # extension unavailable -> pure-numpy fallback below
    if fn is not None:
        blocks, gates, codes = fn(
            np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(decoder, dtype=np.float32),
            float(gamma),
            int(b),
            int(k),
            int(block_tile),
        )
        return (
            np.ascontiguousarray(blocks),
            np.ascontiguousarray(gates),
            np.ascontiguousarray(codes),
        )
    # ---- numpy fallback (documented equivalent; used only on a stale/absent ext) --
    w = (x @ decoder.T).reshape(x.shape[0], g_blocks, b)  # M x G x b tied projections
    gate = np.linalg.norm(w, axis=2)  # M x G  (γ-free routing gate)
    order = np.argsort(-gate, axis=1, kind="stable")[:, :k]  # M x k block-TopK
    rows = np.arange(x.shape[0])[:, None]
    blocks = order.astype(np.uint32)
    gates = float(abs(gamma)) * gate[rows, order]  # M x k presence ‖z_g‖
    codes = float(gamma) * w[rows, order, :]  # M x k x b signed amplitude
    return (
        np.ascontiguousarray(blocks),
        np.ascontiguousarray(gates.astype(np.float32)),
        np.ascontiguousarray(codes.astype(np.float32)),
    )


@dataclass(frozen=True)
class SparseDictionaryTransform:
    """Out-of-sample sparse routing plus CPU/GPU route telemetry."""

    indices: np.ndarray
    codes: np.ndarray
    score_route_stats: dict[str, Any]


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
    score_route_stats: dict[str, Any]

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

    def transform(
        self, X: Any, active: int | None = None, *, score_mode: str = "required"
    ) -> SparseDictionaryTransform:
        """Route held-out rows ``X`` (``M x P``) through the fitted decoder.

        Returns a :class:`SparseDictionaryTransform` with ``indices`` and
        ``codes`` of shape ``M x active`` plus ``score_route_stats``. The stats
        are part of the API: high-``K`` jobs must prove whether the route ran on
        the device or on the host. ``score_mode`` defaults to ``"required"`` so
        an admitted route that cannot run on CUDA raises instead of returning an
        all-CPU success.
        """
        x = _as_2d_f32(X, "X")
        if x.shape[1] != self.decoder.shape[1]:
            raise ValueError(
                f"X must have P={self.decoder.shape[1]} columns; got {x.shape[1]}"
            )
        s = self.active if active is None else int(active)
        s = max(1, min(s, self.decoder.shape[0]))
        payload = rust_module().sparse_dictionary_transform_ffi(
            np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(self.decoder, dtype=np.float32),
            int(s),
            score_mode=str(score_mode),
        )
        data = dict(payload)
        return SparseDictionaryTransform(
            indices=np.ascontiguousarray(data["indices"], dtype=np.uint32),
            codes=np.ascontiguousarray(data["codes"], dtype=np.float32),
            score_route_stats=_route_stats(data["score_route_stats"]),
        )


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
    score_route_stats:
        Aggregate CPU/GPU route telemetry across streamed shards.
    """

    decoder: np.ndarray
    explained_variance: float
    epochs: int
    converged: bool
    active: int
    score_route_stats: dict[str, Any]

    def transform(
        self, X: Any, active: int | None = None, *, score_mode: str = "required"
    ) -> SparseDictionaryTransform:
        """Route held-out rows ``X`` (``M x P``) through the fitted decoder.

        Returns a :class:`SparseDictionaryTransform`, including route telemetry
        proving CPU/device dispatch for the held-out encode. ``score_mode``
        defaults to ``"required"`` for fail-closed GPU routing.
        """
        x = _as_2d_f32(X, "X")
        if x.shape[1] != self.decoder.shape[1]:
            raise ValueError(
                f"X must have P={self.decoder.shape[1]} columns; got {x.shape[1]}"
            )
        s = self.active if active is None else int(active)
        s = max(1, min(s, self.decoder.shape[0]))
        payload = rust_module().sparse_dictionary_transform_ffi(
            np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(self.decoder, dtype=np.float32),
            int(s),
            score_mode=str(score_mode),
        )
        data = dict(payload)
        return SparseDictionaryTransform(
            indices=np.ascontiguousarray(data["indices"], dtype=np.uint32),
            codes=np.ascontiguousarray(data["codes"], dtype=np.float32),
            score_route_stats=_route_stats(data["score_route_stats"]),
        )


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
    K, active, minibatch, max_epochs, score_tile, code_ridge, decoder_ridge, tolerance, score_mode:
        Identical hyper-parameters to :func:`sparse_dictionary_fit`. ``max_epochs``
        is advisory here (the driving Python loop decides how many epochs to run);
        it is carried only so :meth:`end_epoch`'s convergence flag matches the
        one-shot stopping rule. ``score_mode="required"`` fails closed if an
        admitted route cannot run on the CUDA scorer; use ``"off"`` for deliberate
        CPU-only runs.
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
        score_mode: str = "required",
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
            score_mode=str(score_mode),
        )

    def partial_fit(self, shard: Any) -> dict[str, Any]:
        """Route + sparse-code one ``shard`` (``M x P``) against the current
        decoder and fold it into the running epoch.

        Returns per-shard stats ``{rows, rss, alive_atoms, score_route_stats}``
        (``alive_atoms`` is cumulative across the shards seen since the last
        :meth:`end_epoch`).
        """
        shard_arr = _as_2d_f32(shard, "shard")
        data = dict(self._handle.partial_fit(shard_arr))
        data["score_route_stats"] = _route_stats(data["score_route_stats"])
        return data

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
            score_route_stats=_route_stats(data["score_route_stats"]),
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
        Delegates to the Rust core (numpy fallback on a stale/absent extension).
        """
        k = self.block_topk if block_topk is None else int(block_topk)
        return _block_transform(self.decoder, self.gamma, self.block_size, k, X)

    # ---- block -> chart SEEDING SURFACE (Tier-1 block -> Tier-2 nursery) -------
    #
    # These turn each block into a well-seeded low-dimensional nursery for a
    # curved (K=1-per-block) Tier-2 chart, mirroring the block_nursery recipe:
    # project the (residual of the) data into a block's own b-dim coordinates,
    # fit ONE chart there, and lift back with the block frame. The frame D_g is
    # the b x P orthonormal block; its transpose Q = D_gᵀ (P x b, column-
    # orthonormal) is the ``(p, b)`` basis the nursery convention uses
    # (``Z = Xc @ Q``, lift ``Zhat @ Qᵀ``).

    def block_frame(self, g: int) -> np.ndarray:
        """The ``b x P`` orthonormal frame ``D_g`` of block ``g`` (its ``b`` rows
        are orthonormal). Use ``block_frame(g).T`` for the ``(p, b)`` column-
        orthonormal basis ``Q`` in the nursery convention."""
        b = self.block_size
        if not 0 <= g < self.n_blocks:
            raise IndexError(f"block {g} out of range [0, {self.n_blocks})")
        return np.ascontiguousarray(self.decoder[g * b : g * b + b])

    def block_coords(self, X: Any, g: int) -> np.ndarray:
        """In-block coordinates ``X D_gᵀ`` (``M x b``): the tied projection of each
        row onto block ``g``'s subspace — the direct nursery coordinate ``Z``
        (before any residual bookkeeping). Deterministic, no routing."""
        x = _as_2d_f32(X, "X")
        return np.ascontiguousarray(x @ self.block_frame(g).T)

    def lift_block(self, coords: Any, g: int) -> np.ndarray:
        """Lift ``M x b`` in-block coordinates back to ambient ``M x P`` via the
        block frame: ``coords @ D_g``. Inverse of :meth:`block_coords` on the
        block subspace (``D_g D_gᵀ = I_b``)."""
        c = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
        if c.ndim != 2 or c.shape[1] != self.block_size:
            raise ValueError(
                f"coords must be M x b (b={self.block_size}); got {c.shape}"
            )
        return np.ascontiguousarray(c @ self.block_frame(g))

    def _reconstruct_from(self, blocks: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Dense ``M x P`` reconstruction from an arbitrary ``(blocks, codes)``
        routing (``M x k`` / ``M x k x b``)."""
        n = blocks.shape[0]
        p = self.decoder.shape[1]
        b = self.block_size
        out = np.zeros((n, p), dtype=np.float32)
        for j in range(blocks.shape[1]):
            g = blocks[:, j].astype(np.int64)
            for r in range(b):
                out += codes[:, j, r][:, None] * self.decoder[g * b + r]
        return out

    def project_residual(self, X: Any, g: int) -> np.ndarray:
        """Residual-excluding-block-``g`` projected into block ``g``'s coordinates
        (``M x b``).

        Routes ``X`` through the frozen frames, reconstructs, adds block ``g``'s
        OWN contribution back (so the target is the structure every OTHER block
        leaves for ``g``), and projects that leave-one-block-out residual into
        ``g``'s frame. This is the exact per-block target a Tier-2 chart should fit
        (the same ``r_{ig}`` the Rust frame refresh forms), and — unlike the direct
        :meth:`block_coords` — it is meaningful even when blocks share span."""
        x = _as_2d_f32(X, "X")
        b = self.block_size
        dg = self.block_frame(g)  # b x P
        blocks, _gates, codes = self.transform(x)
        xhat = self._reconstruct_from(blocks, codes)
        resid = x - xhat
        # Add block g's own contribution back wherever g was selected.
        sel = blocks == np.uint32(g)  # M x k
        for j in range(blocks.shape[1]):
            mask = sel[:, j]
            if mask.any():
                resid[mask] += codes[mask, j, :] @ dg
        return np.ascontiguousarray(resid @ dg.T)

    def block_firings(self, X: Any) -> np.ndarray:
        """Per-block firing counts on ``X`` (``G``-vector): how many rows route to
        each block under the frozen frames. ``n_firings`` for the MDL scorer."""
        x = _as_2d_f32(X, "X")
        blocks, _gates, _codes = self.transform(x)
        counts = np.zeros(self.n_blocks, dtype=np.int64)
        vals, n = np.unique(blocks, return_counts=True)
        counts[vals.astype(np.int64)] = n
        return counts

    def block_seeds(
        self,
        X: Any,
        *,
        n_basis_chart: int = 4,
        residual_target: bool = True,
        name_prefix: str = "block",
    ) -> list[dict]:
        """Per-block seed records for ranking which blocks deserve a curved Tier-2
        chart, each carrying MDL-scorer featurizer rows (block rung + circle-chart
        rung) matching the mdl_ladder JSON interface.

        For every block: its in-block coordinate statistics (variance spectrum),
        firing count, utilisation, stable rank, and the linear EV it captures — plus
        two ready-to-score featurizer dicts (``mdl_block`` / ``mdl_chart``) whose
        ``block_name`` / ``chart_name`` are set so ``mdl.score_json`` returns the
        per-block chart-vs-block crossover ``f*``. ``residual_target`` chooses the
        in-block coordinates (leave-one-block-out residual vs direct projection).

        The featurizer ``total_var``/``coded_var`` live in the block's b-dim
        coordinate space (the shared lift ``Q`` cancels in the block-vs-chart
        crossover), matching how block_nursery scores each block.
        """
        x = _as_2d_f32(X, "X")
        n_tokens = x.shape[0]
        p = self.decoder.shape[1]
        b = self.block_size
        firings = self.block_firings(x)
        # Total ambient variance (denominator for each block's linear EV).
        xc = x - x.mean(axis=0, keepdims=True)
        ambient_var = float((xc**2).sum()) or 1.0
        seeds: list[dict] = []
        for g in range(self.n_blocks):
            coords = (
                self.project_residual(x, g) if residual_target else self.block_coords(x, g)
            )
            cc = coords - coords.mean(axis=0, keepdims=True)
            # Per-intrinsic-coordinate signal variances = eigenvalues of the b x b
            # coordinate covariance (a gauge invariant of the block code).
            cov = (cc.T @ cc) / max(cc.shape[0], 1)
            eig = np.linalg.eigvalsh(cov.astype(np.float64))
            coded_var = np.sort(np.clip(eig, 0.0, None))[::-1]
            total_var = float(coded_var.sum())
            # Linear EV this block captures of the ambient (its projector's energy).
            block_ev = float((cc**2).sum() / ambient_var)
            f = int(firings[g])
            base = f"{name_prefix}{g}"
            mdl_block = {
                "name": f"{base}-linear-{b}d",
                "kind": "block",
                "total_var": max(total_var, 1e-12),
                "n_tokens": int(n_tokens),
                "n_firings": max(f, 1),
                "n_params": b * p,
                "coded_var": [float(v) for v in coded_var],
                "g_dict": int(self.n_blocks),
                "k_active": int(self.block_topk),
            }
            mdl_chart = {
                "name": f"{base}-circle-chart",
                "kind": "chart",
                "total_var": max(total_var, 1e-12),
                "n_tokens": int(n_tokens),
                "n_firings": max(f, 1),
                "n_params": int(n_basis_chart) * p,
                # A single intrinsic (angular) coordinate carries the block's signal
                # variance; ev split across the 1 coord = total captured.
                "coded_var": [max(total_var, 1e-12)],
                "g_dict": int(self.n_blocks),
                "k_active": int(self.block_topk),
                "block_name": f"{base}-linear-{b}d",
                "chart_name": f"{base}-circle-chart",
            }
            seeds.append(
                {
                    "block": g,
                    "block_dim": b,
                    "n_firings": f,
                    "utilization": float(self.block_utilization[g]),
                    "stable_rank": float(self.block_stable_rank[g]),
                    "coded_var": [float(v) for v in coded_var],
                    "total_var": total_var,
                    "block_linear_ev": block_ev,
                    "mdl_block": mdl_block,
                    "mdl_chart": mdl_chart,
                }
            )
        return seeds

    def seed_manifest(
        self,
        X: Any,
        *,
        n_basis_chart: int = 4,
        residual_target: bool = True,
        include_bases: bool = True,
    ) -> dict:
        """A JSON-serialisable Tier-1 -> Tier-2 hand-off manifest: per-block basis,
        coordinate statistics, firing counts, and MDL featurizer rows, plus a flat
        ``mdl_featurizers`` list ready to pass straight to ``mdl.score_json``.

        The heavy per-row in-block COORDINATES are NOT embedded here (call
        :meth:`block_coords` / :meth:`project_residual` and save them to an ``.npz``
        alongside); this manifest carries only the (p x b) bases + scalar stats so
        it stays a compact, human-readable JSON. Mirrors the block->chart hand-off
        block_nursery consumes (basis ``Q``, in-block coords, per-block stats)."""
        seeds = self.block_seeds(
            X, n_basis_chart=n_basis_chart, residual_target=residual_target
        )
        b = self.block_size
        blocks_out = []
        featurizers = []
        for s in seeds:
            g = s["block"]
            entry = {k: v for k, v in s.items() if k not in ("mdl_block", "mdl_chart")}
            if include_bases:
                # (p, b) column-orthonormal basis Q = D_gᵀ (nursery convention).
                entry["basis"] = self.block_frame(g).T.tolist()
            blocks_out.append(entry)
            featurizers.append(s["mdl_block"])
            featurizers.append(s["mdl_chart"])
        return {
            "schema": "block_seed_manifest.v1",
            "n_blocks": int(self.n_blocks),
            "block_size": int(b),
            "block_topk": int(self.block_topk),
            "ambient_p": int(self.decoder.shape[1]),
            "gamma": float(self.gamma),
            "explained_variance": float(self.explained_variance),
            "residual_target": bool(residual_target),
            "n_basis_chart": int(n_basis_chart),
            "blocks": blocks_out,
            "mdl_featurizers": featurizers,
        }


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


@dataclass(frozen=True)
class BlockSparseStreamArtifact:
    """Result of a streaming (partial-fit) block-sparse fit.

    The streaming path never materialises an ``N x k`` routing (a streamed corpus
    is re-encoded shard-by-shard through the frozen frames), so — unlike
    :class:`BlockSparseDictionaryFit` — this artifact carries only the trained
    block frames, the shared scalar ``γ``, the per-block report, and run metadata.
    Call :meth:`to_fit` to route a representative sample back through the frames and
    obtain a full :class:`BlockSparseDictionaryFit` (with the whole block->chart
    seeding surface), or :meth:`transform` for the sparse block routing alone.

    Attributes
    ----------
    decoder:
        ``K x P`` block frames (``K = G*b``); each block's ``b`` rows orthonormal.
    gamma:
        Shared tied-encoder scalar ``γ``.
    block_utilization, block_stable_rank:
        Length-``G`` per-block report from the final epoch.
    block_topk, block_size, epochs, explained_variance, converged:
        Run metadata.
    """

    decoder: np.ndarray
    gamma: float
    block_topk: int
    block_size: int
    block_utilization: np.ndarray
    block_stable_rank: np.ndarray
    epochs: int
    explained_variance: float
    converged: bool

    @property
    def n_blocks(self) -> int:
        """Number of blocks ``G = K / b``."""
        return self.decoder.shape[0] // self.block_size

    def transform(
        self, X: Any, block_topk: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Route held-out rows ``X`` (``M x P``) through the fitted block frames,
        returning ``(blocks, gates, codes)`` — the same block-TopK routing + tied
        signed codes the trainer uses, against the frozen frames. Delegates to the
        Rust core (numpy fallback on a stale/absent extension)."""
        k = self.block_topk if block_topk is None else int(block_topk)
        return _block_transform(self.decoder, self.gamma, self.block_size, k, X)

    def to_fit(self, X: Any) -> BlockSparseDictionaryFit:
        """Route a representative sample ``X`` through the frozen frames and return
        a full :class:`BlockSparseDictionaryFit` — the streamed frames + γ + report
        plus the sample's routing/reconstruction — so the ENTIRE block->chart
        seeding surface (``block_frame`` / ``block_coords`` / ``project_residual`` /
        ``block_seeds`` / ``seed_manifest`` …) is available on a streamed fit.
        """
        x = _as_2d_f32(X, "X")
        blocks, gates, codes = self.transform(x)
        b = self.block_size
        p = self.decoder.shape[1]
        fitted = np.zeros((x.shape[0], p), dtype=np.float32)
        for j in range(self.block_topk):
            gg = blocks[:, j].astype(np.int64)
            for r in range(b):
                fitted += codes[:, j, r][:, None] * self.decoder[gg * b + r]
        return BlockSparseDictionaryFit(
            decoder=np.ascontiguousarray(self.decoder, dtype=np.float32),
            blocks=blocks,
            gates=gates,
            codes=codes,
            gamma=float(self.gamma),
            block_utilization=np.ascontiguousarray(self.block_utilization, dtype=np.float32),
            block_stable_rank=np.ascontiguousarray(self.block_stable_rank, dtype=np.float32),
            fitted=np.ascontiguousarray(fitted),
            explained_variance=float(self.explained_variance),
            epochs=int(self.epochs),
            converged=bool(self.converged),
            block_topk=int(self.block_topk),
            block_size=int(b),
        )


class BlockSparseDictStream:
    """Partial-fit streaming surface for the block-sparse lane (#1026 block ext).

    Wraps the native ``BlockSparseDictStream`` handle so a Python loop can stream
    epochs over shards of a corpus that never fits in memory at once (a sharded
    residual-stream harvest of tens of millions of tokens). All heavy state — the
    warm-started block frames, γ, the epoch's per-block cross-moments, the dead-
    block revival reservoir — lives native-side; a shard round-trips only its own
    ``shard x P`` rows through Python, so per-shard overhead is independent of ``K``
    and of the corpus length.

    Usage mirrors :class:`SparseDictStream`::

        stream = BlockSparseDictStream(seed_sample, G, block_size=2, block_topk=1)
        for _epoch in range(max_epochs):
            for shard in reader.batches(65536):    # ShardReader batches ARE arrays
                stream.partial_fit(shard)
            stats = stream.end_epoch()
            if stats["converged"]:
                break
        art = stream.finalize()

    Parameters
    ----------
    seed:
        A representative ``N_seed x P`` sample fixing ``P`` and seeding the initial
        block frames (deterministic farthest-point + orthonormalisation).
    n_blocks, block_size, block_topk, max_epochs, minibatch, block_tile, frame_ridge, aux_k, tolerance:
        Identical hyper-parameters to :func:`block_sparse_dictionary_fit`.
    """

    def __init__(
        self,
        seed: Any,
        n_blocks: int,
        *,
        block_size: int = 2,
        block_topk: int = 1,
        max_epochs: int = 30,
        minibatch: int = 512,
        block_tile: int = 1024,
        frame_ridge: float = 1.0e-9,
        aux_k: int = 0,
        tolerance: float = 1.0e-6,
    ) -> None:
        seed_arr = _as_2d_f32(seed, "seed")
        if block_size > seed_arr.shape[1]:
            raise ValueError(
                f"block_size={block_size} cannot exceed P={seed_arr.shape[1]}"
            )
        self._handle = rust_module().BlockSparseDictStream(
            seed_arr,
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

    def partial_fit(self, shard: Any) -> dict[str, Any]:
        """Route + tied-code one ``shard`` (``M x P``) against the current frames
        and fold it into the running epoch. Accepts any 2-D float array-like,
        including :class:`ShardReader` ``batches(n)`` blocks. Returns
        ``{rows, rss, alive_blocks}`` (``alive_blocks`` cumulative since the last
        :meth:`end_epoch`)."""
        shard_arr = _as_2d_f32(shard, "shard")
        return dict(self._handle.partial_fit(shard_arr))

    def end_epoch(self) -> dict[str, Any]:
        """Close the epoch: refresh γ + block frames from the accumulators, revive
        dead blocks onto worst-reconstructed residual rows, reset the accumulators.
        Returns ``{explained_variance, revived, dead, gamma, converged, epoch}``."""
        return dict(self._handle.end_epoch())

    def block_rank_charges(self, n_obs: int) -> dict[str, Any]:
        """Per-BLOCK honest-charge ledger from the last CLOSED epoch (the
        certification surface for width-capacity fits).

        For each block ``g``: ``d_eff`` is the realised rank-charge DOF of its
        orthonormal frame under the epoch's code Gram (the SAME
        ``realised_rank_charge_dof`` currency the joint PROMOTE/DEMOTE gates
        charge); ``delta_deviance`` is the deviance reduction the block's codes
        claim; ``charge = 0.5 * d_eff * ln(n_obs)``; ``kept = margin > 0``. The
        block is the certification unit — its ``b`` atoms share one jointly
        fitted frame and one Gram, so atom ids for block ``g`` are
        ``g*b .. (g+1)*b`` and inherit the block's verdict. ``margin`` doubles
        as a ``log_e_value`` for :func:`e_bh_dictionary_certificate`. Call
        after at least one :meth:`end_epoch`.

        Returns parallel lists
        ``{block, n_eff, d_eff, delta_deviance, charge, margin, kept}``.
        """
        return dict(self._handle.block_rank_charges(int(n_obs)))

    def finalize(self) -> BlockSparseStreamArtifact:
        """Hand back the trained block frames + γ + per-block report as a
        :class:`BlockSparseStreamArtifact`."""
        data = dict(self._handle.finalize())
        return BlockSparseStreamArtifact(
            decoder=np.ascontiguousarray(data["decoder"], dtype=np.float32),
            gamma=float(data["gamma"]),
            block_topk=int(data["block_topk"]),
            block_size=int(data["block_size"]),
            block_utilization=np.ascontiguousarray(data["block_utilization"], dtype=np.float32),
            block_stable_rank=np.ascontiguousarray(data["block_stable_rank"], dtype=np.float32),
            epochs=int(data["epochs"]),
            explained_variance=float(data["explained_variance"]),
            converged=bool(data["converged"]),
        )

    @property
    def decoder(self) -> np.ndarray:
        """A live copy of the current warm-started block frames (``K x P``)."""
        return np.ascontiguousarray(self._handle.decoder(), dtype=np.float32)

    @property
    def gamma(self) -> float:
        """Current shared tied scalar ``γ``."""
        return float(self._handle.gamma)

    @property
    def block_topk(self) -> int:
        """Block routing budget ``k`` in use (``min(block_topk, G)``)."""
        return int(self._handle.block_topk)

    @property
    def block_size(self) -> int:
        """Block size ``b``."""
        return int(self._handle.block_size)

    @property
    def epochs_run(self) -> int:
        """Epochs closed so far."""
        return int(self._handle.epochs_run)


def block_sparse_dictionary_fit_begin(
    seed: Any,
    n_blocks: int,
    *,
    block_size: int = 2,
    block_topk: int = 1,
    max_epochs: int = 30,
    minibatch: int = 512,
    block_tile: int = 1024,
    frame_ridge: float = 1.0e-9,
    aux_k: int = 0,
    tolerance: float = 1.0e-6,
) -> BlockSparseDictStream:
    """Begin a STREAMING block-sparse fit and return a :class:`BlockSparseDictStream`.

    Thin functional alias for ``BlockSparseDictStream(seed, n_blocks, ...)``
    mirroring the ``fit_begin`` / ``partial_fit`` / ``finalize`` surface of the
    atom lane's :func:`sparse_dictionary_fit_begin`.
    """
    return BlockSparseDictStream(
        seed,
        n_blocks,
        block_size=block_size,
        block_topk=block_topk,
        max_epochs=max_epochs,
        minibatch=minibatch,
        block_tile=block_tile,
        frame_ridge=frame_ridge,
        aux_k=aux_k,
        tolerance=tolerance,
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
    score_mode: str = "required",
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
        score_mode=score_mode,
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
    score_mode: str = "required",
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
    score_mode:
        ``"required"`` (default) fails closed when an admitted high-``K`` route
        cannot run on CUDA. Use ``"off"`` for deliberate CPU-only runs.
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
        score_mode=str(score_mode),
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
        score_route_stats=_route_stats(data["score_route_stats"]),
    )


def rank_charge_dof(
    gram: Any,
    decoder: Any,
    n_eff: float,
    p_out: float,
    dispersion: float,
) -> float:
    """Exact realised rank-charge DOF — the single evidence currency.

    Calls the SAME native ``realised_rank_charge_dof`` the joint REML PROMOTE
    gate, the hybrid-split DEMOTE gate, and the streaming block ledger charge,
    so external drivers (the Mode-A per-block chart pass, compose/certify
    reports) price candidates with the criterion itself instead of re-deriving
    the formula — re-derivations drift. ``gram`` is the candidate's ``M x M``
    weighted design Gram, ``decoder`` its ``M x p`` decoder block, ``n_eff``
    the effective sample mass, ``p_out`` the output dimension, ``dispersion``
    the reconstruction dispersion (MP floor input). The evidence price is then
    ``0.5 * d_eff * ln(n)`` against the deviance-priced loss reduction.
    """
    gram_arr = np.ascontiguousarray(gram, dtype=np.float64)
    decoder_arr = np.ascontiguousarray(decoder, dtype=np.float64)
    if gram_arr.ndim != 2 or decoder_arr.ndim != 2:
        raise ValueError(
            "rank_charge_dof: gram and decoder must be 2-D arrays; got shapes "
            f"{gram_arr.shape} and {decoder_arr.shape}"
        )
    return float(
        rust_module().rank_charge_dof(
            gram_arr, decoder_arr, float(n_eff), float(p_out), float(dispersion)
        )
    )


__all__ = [
    "BlockSparseDictStream",
    "BlockSparseDictionaryFit",
    "BlockSparseStreamArtifact",
    "SparseDictStream",
    "SparseDictStreamArtifact",
    "SparseDictionaryFit",
    "SparseDictionaryTransform",
    "block_sparse_dictionary_fit",
    "block_sparse_dictionary_fit_begin",
    "rank_charge_dof",
    "sparse_dictionary_fit",
    "sparse_dictionary_fit_begin",
]
