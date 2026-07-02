"""#1026 collapsed linear lane, streaming (partial-fit) surface.

``SparseDictStream`` gives the fixed-K sparse trainer a resumable
``fit_begin`` / ``partial_fit`` / ``finalize`` API so a Python loop can stream
epochs over shards of a corpus that never fits in memory at once. These tests
prove the three load-bearing properties:

(a) streaming the row-ordered shards of a corpus reaches the same fixed point as
    a one-shot fit on the concatenation (same final EV, identical decoder shape);
(b) the decoder warm-starts across calls (a later epoch improves on the first);
(c) dead-atom revival pulls from the worst-reconstructed *residual rows* — never
    from principal components.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamfit import SparseDictStream, sparse_dictionary_fit, sparse_dictionary_fit_begin
from gamfit._binding import rust_module


def _planted(k, p, n, second_share=0.2):
    """`n` rows, each a scaled single planted atom plus a small bleed into the
    next atom, so the fixed-K lane has real reconstructible structure."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((p, p)).astype(np.float64)
    q, _ = np.linalg.qr(a)
    atoms = q[:k].astype(np.float32)  # k x p, orthonormal rows
    x = np.zeros((n, p), dtype=np.float32)
    for row in range(n):
        primary = row % k
        secondary = (primary + 1) % k
        scale = np.float32(0.7 + 0.01 * (row // k))
        x[row] = scale * atoms[primary] + second_share * scale * atoms[secondary]
    return x


def _reconstruct(indices, codes, decoder):
    out = np.zeros((indices.shape[0], decoder.shape[1]), dtype=np.float32)
    for j in range(indices.shape[1]):
        out += codes[:, [j]] * decoder[indices[:, j]]
    return out


def _routed_ev(x, decoder, active):
    """EV of `decoder` over `x` under the trainer's own routing (tiled top-s +
    active-set ridge) — the quantity the fit reports, computed independently."""
    idx, cod = rust_module().sparse_dictionary_transform_ffi(
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(decoder, dtype=np.float32),
        int(active),
    )
    recon = _reconstruct(idx, cod, decoder)
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0 else (1.0 if rss == 0 else 0.0)


def test_streaming_over_shards_matches_one_shot_on_concatenation():
    n, k, p = 240, 6, 8
    x = _planted(k, p, n)
    kw = dict(
        active=1,
        minibatch=32,
        max_epochs=40,
        score_tile=16,
        code_ridge=1.0e-6,
        decoder_ridge=1.0e-6,
        tolerance=1.0e-9,
    )

    one_shot = sparse_dictionary_fit(x, k, **kw)

    # Four contiguous shards whose row-order concatenation is exactly `x`; seed the
    # stream from the concatenation so the initial atom directions match one-shot.
    shards = np.array_split(x, 4)
    assert np.array_equal(np.concatenate(shards, axis=0), x)

    stream = SparseDictStream(x, k, **kw)
    for _ in range(kw["max_epochs"]):
        for shard in shards:
            stream.partial_fit(shard)
        if stream.end_epoch()["converged"]:
            break
    artifact = stream.finalize()

    assert artifact.decoder.shape == one_shot.decoder.shape == (k, p)

    ev_stream = _routed_ev(x, artifact.decoder, artifact.active)
    assert ev_stream > 0.9, f"planted corpus should fit well, got EV {ev_stream}"
    assert abs(ev_stream - one_shot.explained_variance) < 1.0e-3, (
        f"streamed EV {ev_stream} must match one-shot EV "
        f"{one_shot.explained_variance} within 1e-3"
    )


def test_one_shot_fit_still_works():
    # The batch entry point must be untouched by the streaming surgery.
    x = _planted(5, 6, 150)
    fit = sparse_dictionary_fit(x, 5, active=1, minibatch=64, max_epochs=25, score_tile=16)
    assert fit.decoder.shape == (5, 6)
    assert fit.indices.shape == fit.codes.shape == (150, fit.active)
    assert fit.explained_variance > 0.9


def test_warm_start_persists_across_epochs():
    # Full-rank data with a modest, undercomplete dictionary: the seed decoder
    # reconstructs only partially, so a later epoch — which routes against the
    # decoder that earlier epochs refreshed — must post a strictly higher EV than
    # the first. If state did not persist across calls, every epoch would re-see
    # the seed and EVs would not climb.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 12)).astype(np.float32)
    stream = sparse_dictionary_fit_begin(
        x, 8, active=1, minibatch=64, max_epochs=6, score_tile=16, tolerance=1.0e-12
    )
    evs = []
    for _ in range(6):
        stream.partial_fit(x)
        evs.append(stream.end_epoch()["explained_variance"])

    assert stream.epochs_run == 6
    assert evs[1] > evs[0] + 1.0e-3, (
        f"second-epoch EV {evs[1]} must improve on first-epoch EV {evs[0]} "
        "(warm-start persisted across partial_fit/end_epoch calls)"
    )
    assert evs[-1] > evs[0], "EV must climb across the streamed epochs"


def test_revival_pulls_from_worst_reconstructed_row_not_pcs():
    # The streaming API lets us seed the decoder from a sample that does NOT span a
    # direction the corpus then visits, giving a fully controlled revival: the seed
    # covers only e0/e1 (so one seeded atom is a redundant duplicate that fires for
    # no row — a dead atom), while the streamed shard adds a lone e2 row that no
    # atom can reconstruct — the worst-reconstructed residual row. Dead-atom revival
    # must point the orphaned atom at THAT row's residual direction (e2), never at a
    # principal component (the shard's variance is entirely along e0/e1; e2 is its
    # least-variance axis).
    p = 4
    eye = np.eye(p, dtype=np.float32)
    seed = np.concatenate(
        [np.tile(3.0 * eye[0], (10, 1)), np.tile(3.0 * eye[1], (10, 1))]
    ).astype(np.float32)
    shard = np.concatenate(
        [
            np.tile(3.0 * eye[0], (10, 1)),
            np.tile(3.0 * eye[1], (10, 1)),
            (2.0 * eye[2]).reshape(1, -1),
        ]
    ).astype(np.float32)

    stream = SparseDictStream(seed, 3, active=1, minibatch=64, max_epochs=5, score_tile=16)

    # Seed decoder spans only e0/e1: nothing points at the soon-to-be-worst row.
    assert np.abs(stream.decoder @ eye[2]).max() < 1.0e-4

    # Residual of the e2 row under the pre-refresh (seed) decoder, via the trainer's
    # own routing — the exact field revival ranks. It is the worst-reconstructed row.
    d_pre = stream.decoder
    idx, cod = rust_module().sparse_dictionary_transform_ffi(
        shard, np.ascontiguousarray(d_pre, dtype=np.float32), 1
    )
    resid = shard - _reconstruct(idx, cod, d_pre)
    resid_norm = np.linalg.norm(resid, axis=1)
    worst = int(np.argmax(resid_norm))
    assert worst == shard.shape[0] - 1, "the lone e2 row must be the worst-reconstructed"
    worst_dir = resid[worst] / np.linalg.norm(resid[worst])

    stream.partial_fit(shard)
    stats = stream.end_epoch()
    assert stats["dead"] >= 1 and stats["revived"] >= 1, (
        f"expected a dead atom revived; got dead={stats['dead']} revived={stats['revived']}"
    )

    d_post = stream.decoder
    # A revived atom now equals the worst row's residual direction (sign-free)...
    assert np.abs(d_post @ worst_dir).max() > 0.999, "no atom matched the worst residual row"
    # ...and the shard's leading principal component (along e0/e1) is NOT that
    # direction — revival used a residual row, not a PC.
    sc = shard - shard.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(sc, full_matrices=False)
    assert abs(float(worst_dir @ vt[0])) < 1.0e-3, "revival source coincided with the top PC"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
