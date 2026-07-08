//! #1026 streaming-EFS-cache regression test, split out of `tests.rs` to keep
//! that tracked file under the #780 10k-line gate. Declared as a sibling
//! `#[cfg(test)] mod` in `mod.rs`; the shared `small_two_atom_periodic_term`
//! fixture is sourced from the sibling `tests` module.

use super::tests::small_two_atom_periodic_term;
use approx::assert_abs_diff_eq;

/// #1026: the massive-K EFS lane (`efs_step`) can no longer form the dense
/// evidence cache, so when `!direct_logdet_admitted` it takes its ARD-trace and
/// dispersion inputs off the cache RETURNED by
/// `reml_criterion_streaming_exact_with_cache` instead of the dense
/// `reml_criterion_with_cache`. The behavioural contract of that fix is that the
/// streaming cache is a **drop-in** for those exact EFS consumers: at the shared
/// converged optimum it must yield the SAME `ard_inverse_traces` and the SAME
/// `reconstruction_dispersion` as the dense cache. `efs_step`'s routing branch is
/// gated on a memory-budget decision that only flips at massive K (infeasible to
/// exercise in a unit test), so we instead pin the underlying equivalence on a
/// small dictionary where BOTH caches are formable — a regression that made the
/// returned streaming cache diverge from the dense one (wrong factor, stale inner
/// state, mismatched log-det) would surface here rather than only in a multi-GB
/// massive-K fit.
#[test]
fn streaming_cache_is_efs_dropin_for_dense_cache_1026() {
    let (term0, target, rho) = small_two_atom_periodic_term();
    let mut dense = term0.clone();
    let mut streaming = term0;

    let (dense_cost, dense_loss, dense_cache) = dense
        .reml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .unwrap();
    let (stream_cost, stream_loss, stream_cache) = streaming
        .reml_criterion_streaming_exact_with_cache(
            target.view(),
            &rho,
            None,
            2,
            0.25,
            1.0e-4,
            1.0e-4,
        )
        .unwrap();

    // Cost/loss equivalence, re-pinned for the cache-returning streaming variant
    // (K=2 < the Hutchinson-trace threshold, so every quantity below is exact).
    assert_abs_diff_eq!(stream_cost, dense_cost, epsilon = 1.0e-8);
    assert_abs_diff_eq!(stream_loss.total(), dense_loss.total(), epsilon = 1.0e-8);

    // The exact inputs `efs_step` reads off the returned cache must agree.
    let dense_traces = dense.ard_inverse_traces(&dense_cache).unwrap();
    let stream_traces = streaming.ard_inverse_traces(&stream_cache).unwrap();
    assert_eq!(
        dense_traces.len(),
        stream_traces.len(),
        "streaming cache yields a different ARD-trace atom count than the dense cache"
    );
    for (k, (d, s)) in dense_traces.iter().zip(stream_traces.iter()).enumerate() {
        assert_eq!(
            d.len(),
            s.len(),
            "ARD-trace latent dim mismatch at atom {k}"
        );
        for (dv, sv) in d.iter().zip(s.iter()) {
            assert_abs_diff_eq!(dv, sv, epsilon = 1.0e-8);
        }
    }

    let dense_disp = dense
        .reconstruction_dispersion(&dense_loss, &dense_cache, &rho, None)
        .unwrap();
    let stream_disp = streaming
        .reconstruction_dispersion(&stream_loss, &stream_cache, &rho, None)
        .unwrap();
    assert_abs_diff_eq!(dense_disp, stream_disp, epsilon = 1.0e-8);
}
