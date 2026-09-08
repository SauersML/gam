"""Native block precision and measured convergence transport regressions."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

import gamfit
from gamfit._binding import rust_module
from gamfit._sparse_dictionary import _block_sparse_fit_from_payload


def _rows():
    delta = 2.0 ** -35
    return np.array([[1.0 + delta, 0.5], [-0.25, 1.0 - delta],
                     [-1.0 - delta, -0.5], [0.25, -1.0 + delta]], dtype=np.float64)


def _batch_payload(rows):
    return dict(rust_module().fixed_budget_block_sparse_dictionary_fit(
        rows, 2, active=2, block_size=2, max_epochs=16, minibatch=2,
        block_tile=1, frame_ridge=0.0, aux_k=0, matryoshka_prefix=False,
        tolerance=1e-10))


def test_block_f64_routing_preserves_resolved_values_2825():
    native = rust_module()
    sample = _rows()[:3].copy()
    gamma = 1.0 + 2.0 ** -34
    identity = np.eye(2, dtype=np.float64)
    blocks, gates, codes = native.block_sparse_dictionary_transform_ffi(
        sample, identity, gamma, 1, 2, 1)
    assert blocks.dtype == np.uint32
    assert gates.dtype == codes.dtype == np.float64
    for row in range(sample.shape[0]):
        for slot in range(2):
            expected = gamma * sample[row, blocks[row, slot]]
            assert codes[row, slot, 0] == expected
            assert gates[row, slot] == abs(expected)
    reconstruction = native.block_sparse_dictionary_reconstruct_ffi(identity, blocks, codes, 1)
    assert reconstruction.dtype == np.float64
    assert np.array_equal(reconstruction, gamma * sample)
    witness = float(np.max(np.abs(reconstruction - reconstruction.astype(np.float32))))
    assert witness > 1e-11
    print(json.dumps({"f64_rounding_witness": witness}))


def test_stream_certificate_preserves_corpus_provenance_when_routing_a_sample_2825():
    rows = _rows()
    stream = gamfit.BlockSparseDictStream(
        rows, 1, block_size=2, block_topk=1, max_epochs=16,
        minibatch=2, block_tile=1, frame_ridge=0.0, aux_k=0, tolerance=1e-10)
    terminal = None
    for _ in range(16):
        stream.partial_fit(rows[:2])
        stream.partial_fit(rows[2:])
        terminal = stream.end_epoch()
        if terminal["converged"]:
            break
    assert terminal is not None and terminal["converged"], terminal
    payload = dict(stream._handle.finalize())
    artifact = stream.finalize()
    assert asdict(artifact.convergence) == dict(payload["convergence"])
    assert artifact.convergence.corpus_rows == rows.shape[0]
    assert artifact.convergence.epoch == terminal["epoch"]
    assert artifact.convergence.gamma_residual == terminal["gamma_residual"]
    assert artifact.convergence.frame_residual == terminal["frame_residual"]
    assert artifact.convergence.accepted_births == terminal["accepted_births"] == 0
    assert artifact.convergence.tolerance == 1e-10
    assert not hasattr(artifact, "converged")
    sample = rows[:3].copy()
    routed = artifact.to_fit(sample)
    assert routed.convergence is artifact.convergence
    assert isinstance(routed.convergence, gamfit.BlockSparseStreamConvergence)
    assert routed.convergence.corpus_rows == 4 != routed.fitted.shape[0]
    assert routed.explained_variance == artifact.explained_variance
    assert routed.matryoshka_prefix_losses == ()
    assert all(array.dtype == np.float64 for array in
               [routed.decoder, routed.codes, routed.gates, routed.fitted])
    error = float(np.max(np.abs(routed.fitted - sample)))
    assert error < 1e-12, error
    assert not hasattr(routed.convergence, "routing_residual")
    assert not hasattr(routed.convergence, "reconstruction_residual")
    stream.partial_fit(rows[:1])
    try:
        stream.finalize()
    except ValueError:
        pass
    else:
        raise AssertionError("an incomplete new epoch must invalidate native finalization")
    print(json.dumps({"stream_certificate": asdict(artifact.convergence),
                      "sample_rows": sample.shape[0], "stream_max_error": error}))


def test_batch_python_certificate_retains_every_measured_native_channel_2825():
    rows = _rows()
    payload = _batch_payload(rows)
    batch = _block_sparse_fit_from_payload(payload)
    assert asdict(batch.convergence) == dict(payload["convergence"], corpus_rows=4)
    assert isinstance(batch.convergence, gamfit.BlockSparseDictionaryConvergence)
    assert batch.convergence.accepted_births == batch.convergence.polar_failures == 0
    error = float(np.max(np.abs(batch.fitted - rows)))
    assert error < 1e-12, error
    print(json.dumps({"batch_certificate": asdict(batch.convergence), "batch_max_error": error}))


def test_native_block_artifact_preserves_f64_decoder_and_codes_2825(tmp_path):
    artifact_source = Path(__file__).resolve().parents[1] / "examples/tiered_artifact.py"
    spec = importlib.util.spec_from_file_location("tiered_artifact_precision_test", artifact_source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    batch = _block_sparse_fit_from_payload(_batch_payload(_rows()))
    artifact = module.TieredArtifact(
        t1_decoder=batch.decoder, t1_indices=batch.blocks, t1_codes=batch.codes)
    saved_hash = artifact.save(str(tmp_path))
    loaded = module.TieredArtifact.load(str(tmp_path))
    for actual, expected in [(loaded.t1_decoder, batch.decoder),
                             (loaded.t1_indices, batch.blocks), (loaded.t1_codes, batch.codes)]:
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    assert loaded.t1_decoder.dtype == loaded.t1_codes.dtype == np.float64
    assert loaded.content_hash() == saved_hash and "hash_mismatch" not in loaded.provenance
    print(json.dumps({"native_artifact_hash": saved_hash}))
