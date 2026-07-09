// [#780 line-count gate] Cohesive softmax-entropy Gershgorin majorizer leaf
// helpers split out of `construction.rs` (which crossed the 10k-line gate).
// These are the #1410 per-row active-atom majorizer / dense-entropy-Hessian /
// logit-derivative entry functions: pure leaf math over a softmax row, no
// struct-private coupling. Included via `include!` from `construction.rs` so
// they keep the SAME module scope (`use super::*`), visibility, and the debug
// oracles that pin them to the dense library routines.

#[inline]
fn softmax_entropy_log_plus_one(probability: f64) -> f64 {
    if probability > 0.0 {
        probability.ln() + 1.0
    } else {
        0.0
    }
}

/// #1410 — single active-atom entry of the per-row softmax-entropy Gershgorin
/// Loewner majorizer `D_kk = Σ_j |H_kj|` (#1419), computed WITHOUT materialising
/// a full-`K` diagonal `d`.
///
/// The compact softmax assembly / θ-adjoint only ever read `D_kk` for the
/// `≤ top_k` active atoms, yet
/// [`SoftmaxAssignmentSparsityPenalty::psd_majorizer_abs_row_sums`] returns the
/// FULL-`K` `d` vector (and the SAE callers were additionally copying the
/// row's logits into a fresh length-`K` `Vec` just to feed it). At the SAE LLM
/// shape (`K ≈ 100k`) that is two `O(K)` per-row scratch allocations on the
/// compact (`O(top_k·d)`-per-token) path the whole #1408/#1409/#1450 contract
/// exists to keep `K`-free. This helper consumes the per-row softmax
/// assignments `a` (already in hand — it IS the softmax row) and an explicit
/// active atom `kk`, and returns only that atom's majorizer diagonal, allocating
/// nothing.
///
/// It reproduces `psd_majorizer_abs_row_sums` EXACTLY (same `(a, l, m)`
/// algebra and the same exact-zero continuation for underflowed probabilities), so the
/// assembly, the criterion's `log|H|`, and the #1006 θ-adjoint still
/// differentiate ONE operator. The shared `m = Σ_j a_j l_j` is the only `O(K)`
/// pass; pass it in precomputed (`softmax_majorizer_log_mean`) so a row that
/// fills several active slots pays it once. A debug oracle
/// (`active_softmax_gershgorin_matches_dense_majorizer_1410`) pins this to the
/// dense `psd_majorizer_abs_row_sums` so the two cannot drift.
#[inline]
pub(crate) fn softmax_majorizer_log_mean(a: &[f64]) -> f64 {
    a.iter()
        .map(|&a_i| a_i * softmax_entropy_log_plus_one(a_i))
        .sum()
}

/// Single `(kk, jj)` entry of the exact per-row dense softmax-entropy Hessian
/// `H_kj = scale·a_k·(δ_kj·(m−l_k−1) + a_j·(l_k+l_j+1−2m))` (mirrors
/// [`SoftmaxAssignmentSparsityPenalty::row_dense_hessian`] entry-for-entry). Used
/// by the #1418 exact-Hessian (`A = B + ΔC`) correction so the compact path can
/// read only the active `≤ top_k × top_k` sub-block of `H_entropy` without
/// materialising the full `K×K` dense block per row (#1410). `m` is the shared
/// [`softmax_majorizer_log_mean`]; `O(1)` per entry, zero allocation.
#[inline]
fn softmax_dense_entropy_hessian_entry(a: &[f64], kk: usize, jj: usize, m: f64, scale: f64) -> f64 {
    let l_kk = softmax_entropy_log_plus_one(a[kk]);
    let l_jj = softmax_entropy_log_plus_one(a[jj]);
    let indicator = if kk == jj { 1.0 } else { 0.0 };
    scale * a[kk] * (indicator * (m - l_kk - 1.0) + a[jj] * (l_kk + l_jj + 1.0 - 2.0 * m))
}

/// Active-atom diagonal `D_kk` of the softmax-entropy Gershgorin majorizer; see
/// [`softmax_majorizer_log_mean`]. `a` is the per-row softmax assignment vector,
/// `kk` the (global) atom index, `m` the precomputed `Σ_j a_j l_j`, and `scale`
/// the `λ/τ²` penalty scale. `O(K)` time, zero allocation.
#[inline]
pub(crate) fn active_softmax_gershgorin_majorizer_entry(a: &[f64], kk: usize, m: f64, scale: f64) -> f64 {
    let l_kk = softmax_entropy_log_plus_one(a[kk]);
    // Diagonal entry H_kk.
    let h_kk = scale * a[kk] * ((m - l_kk - 1.0) + a[kk] * (2.0 * l_kk + 1.0 - 2.0 * m));
    let mut acc = h_kk.abs();
    // Off-diagonal entries H_kj, j ≠ k.
    for (jj, &a_jj) in a.iter().enumerate() {
        if jj == kk {
            continue;
        }
        let l_jj = softmax_entropy_log_plus_one(a_jj);
        let h_kj = scale * a[kk] * a_jj * (l_kk + l_jj + 1.0 - 2.0 * m);
        acc += h_kj.abs();
    }
    acc
}

/// Active-atom diagonal entry `∂D_kk/∂z_w = Σ_j sign(H_kj)·∂H_kj/∂z_w` of the
/// softmax-entropy Gershgorin majorizer derivative (mirrors
/// [`SoftmaxAssignmentSparsityPenalty::row_psd_majorizer_logit_derivative`]'s
/// `out[[kk, kk]]` entry-for-entry — that operator's output is DIAGONAL, so only
/// `kk == kk` entries are nonzero). The compact #1006 θ-adjoint needs this only
/// for the row's `≤ top_k` active atoms paired with its active logits, so this
/// computes one diagonal entry directly from the softmax row `a` instead of
/// materialising the full `K×K` derivative matrix per (row, logit) (#1410).
///
/// `a` is the per-row softmax row, `kk` the (global) atom index, `w` the (global)
/// logit being differentiated, `m` the shared [`softmax_majorizer_log_mean`],
/// `scale = λ/τ²`, and `inv_tau = 1/τ`. Uses the SAME `∂a_r/∂z_w =
/// a_r(δ_rw − a_w)/τ` convention as the dense library routine, so value and
/// adjoint stay on one operator (pinned by
/// `active_softmax_majorizer_logit_derivative_matches_dense_1410`). `O(K)` time,
/// zero allocation.
#[inline]
fn active_softmax_majorizer_logit_derivative_entry(
    a: &[f64],
    kk: usize,
    w: usize,
    m: f64,
    scale: f64,
    inv_tau: f64,
) -> f64 {
    let a_w = a[w];
    // ∂a_r/∂z_w = a_r(δ_rw − a_w)/τ ; ∂L_r/∂z_w = (∂a_r/∂z_w)/a_r ;
    // dm = Σ_r (da_r·l_r + a_r·dl_r). One O(K) pass.
    let da = |r: usize| a[r] * (if r == w { 1.0 } else { 0.0 } - a_w) * inv_tau;
    let l = |r: usize| softmax_entropy_log_plus_one(a[r]);
    let dl = |r: usize| if a[r] > 0.0 { da(r) / a[r] } else { 0.0 };
    let dm: f64 = (0..a.len()).map(|r| da(r) * l(r) + a[r] * dl(r)).sum();
    let l_kk = l(kk);
    let da_kk = da(kk);
    let dl_kk = dl(kk);
    let mut acc = 0.0_f64;
    for jj in 0..a.len() {
        let indicator = if kk == jj { 1.0 } else { 0.0 };
        let l_jj = l(jj);
        // H_kj = scale·a_k·bracket ; only its SIGN is used.
        let bracket = indicator * (m - l_kk - 1.0) + a[jj] * (l_kk + l_jj + 1.0 - 2.0 * m);
        let h_kj = scale * a[kk] * bracket;
        if h_kj == 0.0 {
            continue;
        }
        let dbracket = indicator * (dm - dl_kk)
            + da(jj) * (l_kk + l_jj + 1.0 - 2.0 * m)
            + a[jj] * (dl_kk + dl(jj) - 2.0 * dm);
        let dh_kj = scale * (da_kk * bracket + a[kk] * dbracket);
        acc += h_kj.signum() * dh_kj;
    }
    acc
}
