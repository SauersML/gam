//! Crate-wide test-only helpers for `gam-sae`.
//!
//! This module exists because the ban scanner refuses `#[cfg(test)]` on a `src/`
//! item (`build.rs`), and its exemption is a **private** `mod` literally named
//! `tests` / `test_support` / `tests_*` / `*_tests` — `pub(crate) mod` does NOT
//! qualify (`build.rs`: *"Must literally begin with `mod ` (not `pub mod`, not
//! `pub(crate) mod`)"*). A private module at the CRATE ROOT is nevertheless
//! visible crate-wide, which is what lets a test-only helper be shared across
//! files without a visibility modifier the scanner rejects.

use ndarray::ArrayView1;

use crate::encode::{AtomEncodeAtlas, select_nearest_charts_topk};

/// Top-`k` certified charts for a row, by reconstruction distance.
///
/// **This was production code until `02b27b575` (#2518)**, which deleted
/// `CERTIFIED_ROUTING_TOPK` and made the chart scan exhaustive, removing its one
/// production call (`let candidates = nearest_charts_topk(atom_atlas, x, amplitude,
/// CERTIFIED_ROUTING_TOPK);`). Three test consumers remained, so it was parked
/// under a bare `#[cfg(test)]` attribute in `encode.rs` — which the ban scanner
/// rejects, and which broke every build in the workspace (the scanner runs in the
/// ROOT crate's `build.rs`, so `gam-cli` and `gam-pyffi` failed with
/// `failed to run custom build command for gam`, naming a crate that has nothing
/// to do with it).
///
/// It is NOT collapsed into [`select_nearest_charts_topk`], because it is not a
/// thin wrapper: it owns the `certified_radius <= 0.0` gate and the
/// `recon_center` reuse. Inlining it at the call sites would copy that gate three
/// times.
///
/// If the exhaustive scan makes top-`k` routing permanently unreachable in
/// production, the honest end state is deleting this together with the tests that
/// ride on it. That is a #2518 call about routing, not something to decide inside
/// a commit whose job is to unbreak `main`.
pub(crate) fn nearest_charts_topk(
    atom_atlas: &AtomEncodeAtlas,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    k: usize,
) -> Vec<usize> {
    // `m₁(t_c) = BᵀΦ(t_c)` is an OFFLINE per-chart constant already distilled into
    // `chart.recon_center` at build time (bit-for-bit the same φ·decoder
    // accumulation this used to recompute). Reuse it instead of re-evaluating the
    // basis at a fixed center for every row — that re-eval was the encode's
    // dominant per-row cost. The amplitude-gating + tie-break comparator lives in
    // `select_nearest_charts_topk` (shared with the GPU-host path).
    select_nearest_charts_topk(atom_atlas.charts.len(), x, amplitude, k, |idx, out| {
        let chart = &atom_atlas.charts[idx];
        if chart.certified_radius <= 0.0 {
            return false;
        }
        for (o, r) in out.iter_mut().zip(chart.recon_center.iter()) {
            *o = *r;
        }
        true
    })
    .into_iter()
    .map(|(idx, _)| idx)
    .collect()
}
