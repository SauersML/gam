"""Roundtrip + manifest + subsample tests for the sharded bf16 harvest format.

Exercises examples/residual_shard_io.py directly (no model / GPU needed):
synthetic activations through ShardWriter across several shards, read back via
ShardReader, and verify bit-exact bf16 roundtrip, manifest correctness, and
deterministic stratified subsampling.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
)

import residual_shard_io as rs  # noqa: E402


def _make_data(n_rows: int, d_model: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mix of magnitudes so bf16 rounding actually bites in the mantissa.
    return (rng.standard_normal((n_rows, d_model)).astype(np.float32) * 8.0)


def _write(dir_, data, rows_per_shard, **meta):
    with rs.ShardWriter(
        str(dir_), d_model=data.shape[1], rows_per_shard=rows_per_shard, meta=meta
    ) as w:
        # Append in odd-sized blocks so shard boundaries fall mid-block.
        for i in range(0, data.shape[0], 37):
            w.append(data[i : i + 37])
    return rs.load_shards(str(dir_))


# --------------------------------------------------------------------------
# bf16 conversion
# --------------------------------------------------------------------------
def test_bf16_roundtrip_idempotent():
    x = _make_data(2000, 16)
    q = rs.quantize_bf16(x)
    # Re-quantizing an already-bf16 value is a no-op: the grid is fixed.
    assert np.array_equal(q, rs.quantize_bf16(q))
    # Widening is lossless: bits -> f32 -> bits is identity.
    bits = rs.float32_to_bf16_bits(x)
    assert np.array_equal(bits, rs.float32_to_bf16_bits(rs.bf16_bits_to_float32(bits)))


def test_bf16_matches_torch():
    torch = pytest.importorskip("torch")
    x = _make_data(4096, 8)
    ours = rs.float32_to_bf16_bits(x)
    theirs = (
        torch.from_numpy(x).bfloat16().view(torch.uint16).numpy().reshape(x.shape)
    )
    assert np.array_equal(ours, theirs), "RNE bf16 rounding must match torch"


def test_bf16_specials():
    x = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
    y = rs.quantize_bf16(x)
    assert np.isnan(y[0])
    assert y[1] == np.inf and y[2] == -np.inf
    assert y[3] == 0.0 and y[4] == 0.0


# --------------------------------------------------------------------------
# Write / read roundtrip across shards
# --------------------------------------------------------------------------
def test_multi_shard_roundtrip_bit_exact(tmp_path):
    d_model = 24
    data = _make_data(2500, d_model)
    reader = _write(tmp_path, data, rows_per_shard=800)

    assert len(reader.shards) >= 3, "expected >=3 shards"
    assert reader.total_tokens == data.shape[0]
    assert len(reader) == data.shape[0]

    expected = rs.quantize_bf16(data)  # what a lossless reader must return
    got = reader.read_all()
    assert got.shape == data.shape
    assert got.dtype == np.float32
    assert np.array_equal(got, expected), "read-back must be bit-exact bf16"


def test_batches_epoch_loop(tmp_path):
    d_model = 12
    data = _make_data(2500, d_model)
    reader = _write(tmp_path, data, rows_per_shard=800)
    expected = rs.quantize_bf16(data)

    n = 256
    batches = list(reader.batches(n))
    # All but the last batch are exactly n rows (batches span shard boundaries).
    assert all(b.shape == (n, d_model) for b in batches[:-1])
    assert batches[-1].shape[0] == data.shape[0] - n * (len(batches) - 1)
    recon = np.concatenate(batches, axis=0)
    assert np.array_equal(recon, expected)

    # drop_last yields only full batches.
    full = list(reader.batches(n, drop_last=True))
    assert all(b.shape == (n, d_model) for b in full)
    assert len(full) == data.shape[0] // n


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def test_manifest_schema_and_stats(tmp_path):
    d_model = 20
    data = _make_data(1900, d_model)
    reader = _write(
        tmp_path,
        data,
        rows_per_shard=700,
        model_name="test/model",
        layer=5,
        tokenizer_hash="sha256:deadbeef",
    )

    with open(os.path.join(str(tmp_path), rs.MANIFEST_NAME)) as f:
        man = json.load(f)

    assert man["format"] == rs.FORMAT_NAME
    assert man["format_version"] == rs.FORMAT_VERSION
    assert man["dtype"] == "bfloat16"
    assert man["byte_order"] == "little"
    assert man["d_model"] == d_model
    assert man["total_tokens"] == data.shape[0]
    assert man["model_name"] == "test/model" and man["layer"] == 5
    assert man["tokenizer_hash"] == "sha256:deadbeef"

    # Shard bookkeeping: row counts sum to the total, byte sizes match on disk.
    assert sum(s["rows"] for s in man["shards"]) == data.shape[0]
    for s in man["shards"]:
        path = os.path.join(str(tmp_path), s["file"])
        assert os.path.getsize(path) == s["bytes"]
        assert s["bytes"] == s["rows"] * d_model * 2

    # Per-dimension stats computed on true (pre-quantization) data.
    mean = np.asarray(man["stats"]["mean"], dtype=np.float64)
    norm = np.asarray(man["stats"]["norm"], dtype=np.float64)
    assert mean.shape == (d_model,) and norm.shape == (d_model,)
    np.testing.assert_allclose(mean, data.mean(axis=0), rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        norm, np.sqrt((data.astype(np.float64) ** 2).mean(axis=0)), rtol=1e-4, atol=1e-3
    )


def test_reader_rejects_foreign_dir(tmp_path):
    with open(os.path.join(str(tmp_path), rs.MANIFEST_NAME), "w") as f:
        json.dump({"format": "something-else", "d_model": 4}, f)
    with pytest.raises(ValueError):
        rs.load_shards(str(tmp_path))


# --------------------------------------------------------------------------
# Stratified subsample
# --------------------------------------------------------------------------
def test_stratified_subsample_deterministic(tmp_path):
    d_model = 16
    data = _make_data(3000, d_model)
    reader = _write(tmp_path, data, rows_per_shard=900)

    a = rs.stratified_subsample(reader, 500, seed=123)
    b = rs.stratified_subsample(str(tmp_path), 500, seed=123)
    c = rs.stratified_subsample(reader, 500, seed=999)

    assert a.dtype == np.float32 and a.shape[1] == d_model
    assert np.array_equal(a, b), "same seed must be bit-identical"
    assert not np.array_equal(a, c), "different seed must differ"
    # ~n_target rows, never more, and rows are genuine bf16-grid values.
    assert 0 < a.shape[0] <= 500
    assert abs(a.shape[0] - 500) <= len(reader.shards)
    assert np.array_equal(a, rs.quantize_bf16(a))


def test_stratified_subsample_proportional(tmp_path):
    # Uneven shard sizes: subsample should draw from every shard.
    d_model = 8
    data = _make_data(2050, d_model)
    reader = _write(tmp_path, data, rows_per_shard=1000)  # 1000/1000/50
    sub = rs.stratified_subsample(reader, 205, seed=1)
    # 10% of the tiny 50-row shard should still contribute ~5 rows.
    assert sub.shape[0] > 0
    assert sub.shape[0] <= 205


def test_subsample_caps_at_total(tmp_path):
    d_model = 4
    data = _make_data(100, d_model)
    reader = _write(tmp_path, data, rows_per_shard=60)
    sub = rs.stratified_subsample(reader, 10_000, seed=0)
    assert sub.shape[0] == 100  # cannot exceed available tokens
