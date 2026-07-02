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
    n, k, p = 180, 5, 6
    x = _planted(k, p, n)
    stream = sparse_dictionary_fit_begin(
        x,
        k,
        active=1,
        minibatch=64,
        max_epochs=6,
        score_tile=16,
        tolerance=1.0e-12,
    )
    evs = []
    for _ in range(6):
        stream.partial_fit(x)
        evs.append(stream.end_epoch()["explained_variance"])

    assert stream.epochs_run == 6
    # A later epoch's pre-refresh EV sees the decoder refreshed by earlier epochs;
    # if state did not persist across calls every epoch would re-see the seed.
    assert evs[1] > evs[0] + 1.0e-4, (
        f"second-epoch EV {evs[1]} must improve on first-epoch EV {evs[0]} "
        "(warm-start persisted across partial_fit/end_epoch calls)"
    )


def test_revival_pulls_from_worst_reconstructed_rows_not_pcs():
    # K far larger than the data's effective rank: farthest-point seeding spreads
    # atoms across noise rows, the decoder refresh collapses many onto shared
    # cluster directions, and the orphaned (dead) atoms must be revived onto the
    # worst-reconstructed *residual rows*. We catch the first epoch that revives an
    # atom and verify the revived direction equals that worst row's residual — and
    # is NOT the data's leading principal component.
    rng = np.random.default_rng(7)
    p = 10
    centers = rng.standard_normal((5, p)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows = []
    for c in range(5):
        scale = 4.0 if c == 0 else 2.0  # cluster 0 dominates the variance (top PC)
        block = scale * centers[c] + 0.15 * rng.standard_normal((40, p)).astype(np.float32)
        rows.append(block.astype(np.float32))
    x = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)

    k, active = 24, 1
    # Top principal component of the data (the direction revival must NOT copy).
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    pc0 = vt[0]

    stream = SparseDictStream(
        x, k, active=active, minibatch=64, max_epochs=30, score_tile=64, tolerance=0.0
    )

    observed = False
    for _ in range(30):
        stream.partial_fit(x)
        # Residual under the decoder in force this epoch (the pre-refresh decoder),
        # via the trainer's own routing — this is exactly the field revival ranks.
        d_pre = stream.decoder
        idx, cod = rust_module().sparse_dictionary_transform_ffi(
            x, np.ascontiguousarray(d_pre, dtype=np.float32), active
        )
        resid = x - _reconstruct(idx, cod, d_pre)
        resid_norm = np.linalg.norm(resid, axis=1)

        stats = stream.end_epoch()
        if stats["revived"] > 0:
            d_post = stream.decoder
            worst = int(np.argmax(resid_norm))
            wdir = resid[worst] / np.linalg.norm(resid[worst])
            # Some atom now equals the worst row's residual direction (sign-free).
            cos_atoms = np.abs(d_post @ wdir)
            assert cos_atoms.max() > 0.999, (
                f"a revived atom must equal the worst residual row's direction; "
                f"best |cos|={cos_atoms.max():.4f}"
            )
            # ...and that direction is a residual row, not the leading PC.
            assert abs(float(np.dot(wdir, pc0))) < 0.5, (
                "revival source must be a residual row, not the top principal "
                f"component (|cos to PC0|={abs(float(np.dot(wdir, pc0))):.3f})"
            )
            observed = True
            break

    assert observed, "revival never fired — construction did not orphan any atom"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
