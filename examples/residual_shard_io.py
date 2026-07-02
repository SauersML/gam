#!/usr/bin/env python
"""Sharded bf16 memmap I/O for harvested residual-stream activations.

This is the on-disk contract between the harvester
(``harvest_residual_activations.py``) and downstream streaming consumers
(e.g. an SAE ``partial_fit`` epoch loop). A harvest lives in one directory:

    <dir>/manifest.json          format/provenance + per-dim stats
    <dir>/shard_00000.bf16       raw little-endian uint16 bf16 bit-patterns
    <dir>/shard_00001.bf16       (rows_per_shard rows each, last is short)
    ...

Why bf16-as-uint16: NumPy has no native bfloat16, so each activation scalar
is stored as the top 16 bits of its IEEE-754 float32 encoding (a uint16 bit
pattern). Conversion is lossless in the read direction and round-to-nearest-
even in the write direction; see ``float32_to_bf16_bits`` /
``bf16_bits_to_float32``. Storing raw uint16 (rather than np.save) keeps shards
append-friendly and trivially memmappable as ``(rows, d_model)``.

Reader design targets the natural epoch loop::

    reader = load_shards(dir)                 # or ShardReader(dir)
    for batch in reader.batches(4096):        # float32 (<=4096, d_model)
        model.partial_fit(batch)

plus ``stratified_subsample`` for the 0.5-2M-token curved-fit subsample.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

FORMAT_VERSION = 1
FORMAT_NAME = "residual_shard_bf16"
MANIFEST_NAME = "manifest.json"
SHARD_SUFFIX = ".bf16"
DEFAULT_SHARD_PREFIX = "shard_"
DEFAULT_ROWS_PER_SHARD = 1_000_000
# bf16 bit-patterns are stored little-endian on disk regardless of host order.
_DISK_DTYPE = np.dtype("<u2")


# --------------------------------------------------------------------------
# bf16 <-> float32 bit-pattern conversion (NumPy-only, no torch dependency)
# --------------------------------------------------------------------------
def float32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 -> bfloat16, returned as uint16 bit patterns.

    Round-to-nearest-even, matching ``torch.Tensor.bfloat16()``. bfloat16 is
    exactly the upper 16 bits of float32, so we add the round-to-even bias to
    the 32-bit encoding and take the high half. NaN inputs are mapped to a
    canonical quiet NaN (0x7FC0) since arithmetic rounding could otherwise
    collapse a NaN mantissa to infinity.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    nan_mask = np.isnan(x)
    u = x.view(np.uint32)
    # bias = round-to-nearest-even: 0x7FFF + (lsb of the surviving mantissa)
    bias = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    bits = ((u + bias) >> 16).astype(np.uint16)
    if nan_mask.any():
        bits[nan_mask] = np.uint16(0x7FC0)
    return bits


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Widen uint16 bf16 bit patterns back to float32 (exact, lossless)."""
    bits = np.ascontiguousarray(bits, dtype=np.uint16)
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)


def quantize_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> float32 that has been snapped to the bf16 grid.

    Convenience for tests / reference: ``bf16_bits_to_float32(
    float32_to_bf16_bits(x))``. The result is bit-exactly what a reader will
    observe after a write roundtrip.
    """
    return bf16_bits_to_float32(float32_to_bf16_bits(x))


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------
def tokenizer_hash(tokenizer: Any) -> str:
    """Stable content hash of a HF tokenizer's vocabulary + specials.

    Used purely as a provenance/compatibility fingerprint in the manifest so a
    downstream consumer can detect a tokenizer mismatch. Best-effort: falls
    back to hashing the tokenizer's ``name_or_path`` if the vocab is
    unavailable.
    """
    import hashlib

    h = hashlib.sha256()
    try:
        vocab = tokenizer.get_vocab()
        payload = json.dumps(vocab, sort_keys=True, ensure_ascii=False)
        h.update(payload.encode("utf-8"))
        specials = getattr(tokenizer, "all_special_tokens", None)
        if specials:
            h.update(json.dumps(sorted(specials), ensure_ascii=False).encode("utf-8"))
    except Exception:  # pragma: no cover - provenance fallback
        h.update(str(getattr(tokenizer, "name_or_path", tokenizer)).encode("utf-8"))
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------
@dataclass
class _RunningStats:
    """Per-dimension mean / RMS accumulated in float64 over the true (pre-
    quantization) activations."""

    d_model: int
    count: int = 0
    _sum: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _sumsq: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._sum = np.zeros(self.d_model, dtype=np.float64)
        self._sumsq = np.zeros(self.d_model, dtype=np.float64)

    def update(self, rows: np.ndarray) -> None:
        r = rows.astype(np.float64, copy=False)
        self._sum += r.sum(axis=0)
        self._sumsq += np.square(r).sum(axis=0)
        self.count += r.shape[0]

    def mean(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros(self.d_model, dtype=np.float64)
        return self._sum / self.count

    def rms(self) -> np.ndarray:
        """Per-dimension root-mean-square (== per-dim L2 norm / sqrt(count))."""
        if self.count == 0:
            return np.zeros(self.d_model, dtype=np.float64)
        return np.sqrt(self._sumsq / self.count)


class ShardWriter:
    """Append token-activation rows, rolling over to a new shard file every
    ``rows_per_shard`` rows, and emit a manifest on close.

    Rows are float32 ``(n, d_model)``; they are quantized to bf16 and appended
    as raw little-endian uint16. Use as a context manager, or call ``close()``
    explicitly to flush the manifest.
    """

    def __init__(
        self,
        out_dir: str,
        d_model: int,
        *,
        rows_per_shard: int = DEFAULT_ROWS_PER_SHARD,
        shard_prefix: str = DEFAULT_SHARD_PREFIX,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.out_dir = out_dir
        self.d_model = int(d_model)
        self.rows_per_shard = int(rows_per_shard)
        self.shard_prefix = shard_prefix
        self.meta = dict(meta or {})
        os.makedirs(out_dir, exist_ok=True)

        self._shards: list[dict[str, Any]] = []
        self._stats = _RunningStats(self.d_model)
        self._cur_file = None  # open binary handle
        self._cur_rows = 0
        self._cur_index = -1
        self._total_rows = 0
        self._closed = False

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:  # abandon on error, but don't leak the file handle
            if self._cur_file is not None:
                self._cur_file.close()
                self._cur_file = None

    # -- internals ---------------------------------------------------------
    def _shard_name(self, index: int) -> str:
        return f"{self.shard_prefix}{index:05d}{SHARD_SUFFIX}"

    def _open_new_shard(self) -> None:
        self._cur_index += 1
        name = self._shard_name(self._cur_index)
        self._cur_file = open(os.path.join(self.out_dir, name), "wb")
        self._cur_rows = 0
        self._shards.append({"file": name, "rows": 0})

    def _roll_if_needed(self) -> None:
        if self._cur_file is None or self._cur_rows >= self.rows_per_shard:
            if self._cur_file is not None:
                self._cur_file.close()
            self._open_new_shard()

    # -- public ------------------------------------------------------------
    def append(self, rows: np.ndarray) -> None:
        """Append a ``(n, d_model)`` float32 block (numpy array or torch
        tensor). Larger blocks are split across shard boundaries as needed."""
        if self._closed:
            raise RuntimeError("append() on a closed ShardWriter")
        arr = _to_numpy_f32(rows)
        if arr.ndim != 2 or arr.shape[1] != self.d_model:
            raise ValueError(
                f"expected rows of shape (n, {self.d_model}), got {arr.shape}"
            )
        if arr.shape[0] == 0:
            return
        self._stats.update(arr)

        offset = 0
        n = arr.shape[0]
        while offset < n:
            self._roll_if_needed()
            room = self.rows_per_shard - self._cur_rows
            take = min(room, n - offset)
            block = arr[offset : offset + take]
            bits = float32_to_bf16_bits(block).astype(_DISK_DTYPE, copy=False)
            self._cur_file.write(bits.tobytes())
            self._cur_rows += take
            self._total_rows += take
            self._shards[-1]["rows"] = self._cur_rows
            offset += take

    def close(self) -> dict[str, Any]:
        """Flush the final shard and write ``manifest.json``; returns the
        manifest dict."""
        if self._closed:
            return self._manifest_cache
        if self._cur_file is not None:
            self._cur_file.close()
            self._cur_file = None
        manifest = self._build_manifest()
        with open(os.path.join(self.out_dir, MANIFEST_NAME), "w") as f:
            json.dump(manifest, f, indent=2)
        self._closed = True
        self._manifest_cache = manifest
        return manifest

    def _build_manifest(self) -> dict[str, Any]:
        for s in self._shards:
            s["bytes"] = s["rows"] * self.d_model * _DISK_DTYPE.itemsize
        manifest: dict[str, Any] = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "dtype": "bfloat16",
            "byte_order": "little",
            "d_model": self.d_model,
            "rows_per_shard": self.rows_per_shard,
            "total_tokens": self._total_rows,
            "shards": self._shards,
            "stats": {
                "mean": self._stats.mean().astype(np.float32).tolist(),
                "norm": self._stats.rms().astype(np.float32).tolist(),
            },
        }
        manifest.update(self.meta)
        return manifest


def _to_numpy_f32(rows: Any) -> np.ndarray:
    """Accept a numpy array or a torch tensor; return contiguous float32."""
    if isinstance(rows, np.ndarray):
        return np.ascontiguousarray(rows, dtype=np.float32)
    # torch tensor (duck-typed to avoid importing torch in the reader path)
    detach = getattr(rows, "detach", None)
    if detach is not None:
        rows = detach().to("cpu").float().numpy()
        return np.ascontiguousarray(rows, dtype=np.float32)
    return np.ascontiguousarray(np.asarray(rows), dtype=np.float32)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------
class ShardReader:
    """Memmap-backed reader over a harvest directory.

    ``batches(n)`` yields contiguous float32 ``(<=n, d_model)`` blocks that
    span shard boundaries -- the natural streaming epoch loop. Memmaps are
    opened lazily per shard and cached; only the current batch is ever
    materialized as float32, so memory stays O(n * d_model).
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        with open(os.path.join(out_dir, MANIFEST_NAME)) as f:
            self.manifest: dict[str, Any] = json.load(f)
        if self.manifest.get("format") != FORMAT_NAME:
            raise ValueError(
                f"{out_dir}: not a {FORMAT_NAME} harvest "
                f"(format={self.manifest.get('format')!r})"
            )
        self.d_model: int = int(self.manifest["d_model"])
        self.total_tokens: int = int(self.manifest["total_tokens"])
        self.shards: list[dict[str, Any]] = self.manifest["shards"]
        self._mm_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.total_tokens

    def _memmap(self, shard: dict[str, Any]) -> np.ndarray:
        name = shard["file"]
        mm = self._mm_cache.get(name)
        if mm is None:
            path = os.path.join(self.out_dir, name)
            mm = np.memmap(path, dtype=_DISK_DTYPE, mode="r").reshape(
                shard["rows"], self.d_model
            )
            self._mm_cache[name] = mm
        return mm

    def batches(self, n: int, *, drop_last: bool = False) -> Iterator[np.ndarray]:
        """Yield float32 ``(<=n, d_model)`` blocks across all shards in order.

        Blocks are exactly ``n`` rows except possibly the last (unless
        ``drop_last``). This is the epoch loop::

            for batch in reader.batches(4096):
                model.partial_fit(batch)
        """
        if n <= 0:
            raise ValueError("batch size n must be positive")
        parts: list[np.ndarray] = []
        have = 0
        for shard in self.shards:
            mm = self._memmap(shard)
            rows = int(shard["rows"])
            pos = 0
            while pos < rows:
                take = min(n - have, rows - pos)
                parts.append(np.asarray(mm[pos : pos + take]))
                have += take
                pos += take
                if have == n:
                    yield self._assemble(parts)
                    parts, have = [], 0
        if have and not drop_last:
            yield self._assemble(parts)

    @staticmethod
    def _assemble(parts: list[np.ndarray]) -> np.ndarray:
        bits = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        return bf16_bits_to_float32(bits)

    def read_all(self) -> np.ndarray:
        """Materialize the whole harvest as one float32 array (test/debug)."""
        chunks = [bf16_bits_to_float32(np.asarray(self._memmap(s))) for s in self.shards]
        if not chunks:
            return np.empty((0, self.d_model), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


def load_shards(out_dir: str) -> ShardReader:
    """Open a harvest directory for reading."""
    return ShardReader(out_dir)


def stratified_subsample(
    source: "str | ShardReader",
    n_target: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic subsample of ~``n_target`` rows, stratified by shard.

    Each shard contributes rows proportional to its size, so the subsample is
    spatially representative of the full harvest even when shards correspond to
    different corpus regions. Fully deterministic given ``seed``: each shard's
    draw is seeded from ``(seed, shard_index)`` and returned in sorted row
    order, so the result is independent of iteration timing.

    Returns a float32 ``(m, d_model)`` array with ``m <= n_target`` (``m`` may
    be slightly under ``n_target`` due to per-shard rounding, and is clamped to
    the total available tokens).
    """
    reader = source if isinstance(source, ShardReader) else load_shards(source)
    total = reader.total_tokens
    if n_target <= 0 or total == 0:
        return np.empty((0, reader.d_model), dtype=np.float32)
    n_target = min(n_target, total)

    out: list[np.ndarray] = []
    for idx, shard in enumerate(reader.shards):
        rows = int(shard["rows"])
        if rows == 0:
            continue
        quota = int(round(n_target * rows / total))
        quota = min(quota, rows)
        if quota <= 0:
            continue
        rng = np.random.default_rng([seed, idx])
        sel = np.sort(rng.choice(rows, size=quota, replace=False))
        mm = reader._memmap(shard)
        out.append(bf16_bits_to_float32(np.asarray(mm[sel])))
    if not out:
        return np.empty((0, reader.d_model), dtype=np.float32)
    return np.concatenate(out, axis=0)
