"""A block-sparse fit's native convergence certificate reaches Python intact."""

from dataclasses import asdict
import json

import numpy as np

import gamfit
from gamfit._binding import rust_module
from gamfit._sparse_dictionary import _block_sparse_fit_from_payload


def _rows():
    return np.array(
        [[1.0, 0.5], [-0.25, 1.0], [-1.0, -0.5], [0.25, -1.0]], dtype=np.float32
    )


def _float32_resolution(values):
    # Frames, codes and reconstructions are stored as f32, so a reconstruction
    # agrees with its rows to a few units in the last place of the largest entry.
    return 16.0 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(values)))


def test_stream_certificate_reaches_the_artifact_and_its_routed_sample_2825():
    rows = _rows()
    stream = gamfit.sae.BlockSparseDictStream(
        rows, 1, block_size=2, block_topk=1, max_epochs=16,
        minibatch=2, block_tile=1, aux_k=0, tolerance=1e-10)
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
    assert isinstance(artifact.convergence, gamfit.sae.BlockSparseStreamConvergence)
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
    assert routed.matryoshka_prefix_losses == ()
    assert routed.fitted.shape == sample.shape
    error = float(np.max(np.abs(routed.fitted - sample)))
    assert error <= _float32_resolution(sample), error
    stream.partial_fit(rows[:1])
    try:
        stream.finalize()
    except ValueError:
        pass
    else:
        raise AssertionError("an incomplete new epoch must invalidate native finalization")
    print(json.dumps({"stream_certificate": asdict(artifact.convergence),
                      "stream_max_error": error}))


def test_batch_certificate_retains_every_native_channel_2825():
    rows = _rows()
    payload = dict(rust_module().fixed_budget_block_sparse_dictionary_fit(
        rows, 2, active=2, block_size=2, max_epochs=16, minibatch=2,
        block_tile=1, frame_ridge=0.0, aux_k=0, matryoshka_prefix=False,
        tolerance=1e-10))
    batch = _block_sparse_fit_from_payload(payload)
    assert asdict(batch.convergence) == dict(payload["convergence"])
    assert isinstance(batch.convergence, gamfit.sae.BlockSparseDictionaryConvergence)
    assert batch.convergence.accepted_births == batch.convergence.polar_failures == 0
    error = float(np.max(np.abs(batch.fitted - rows)))
    assert error <= _float32_resolution(rows), error
    print(json.dumps({"batch_certificate": asdict(batch.convergence),
                      "batch_max_error": error}))
