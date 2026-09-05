// [#780 line-count gate] Cohesive softmax-entropy Gershgorin majorizer leaf
// helpers split out of `construction.rs` (which crossed the 10k-line gate).
// These are the #1410 per-row active-atom majorizer / dense-entropy-Hessian /
// logit-derivative entry functions: pure leaf math over a softmax row, no
// struct-private coupling. Included via `include!` from `construction.rs` so
// they keep the SAME module scope (`use super::*`), visibility, and the debug
// oracles that pin them to the dense library routines.

#[cfg(test)]
#[path = "softmax_entropy_majorizer_tests.rs"]
mod recovered_majorizer_tests;

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

/// Active-atom diagonal `D̃_kk` of the softmax-entropy Gershgorin majorizer; see
/// [`softmax_majorizer_log_mean`]. `a` is the per-row softmax assignment vector,
/// `kk` the (global) atom index, `m` the precomputed `Σ_j a_j l_j`, and `scale`
/// the `λ/τ²` penalty scale. `O(K)` time, zero allocation.
///
/// The `|·|` is the soft-abs envelope `σ_{ε_k}(x) = sqrt(x² + ε₀²‖H_k·‖₂²)` of
/// #2339, so this needs TWO passes over the row: one to accumulate the row's own
/// curvature scale `‖H_k·‖₂²`, one to sum the envelope at that scale. Both walk
/// the row diagonal-first, then `j ≠ k` ascending — the same order (and the same
/// arithmetic) as `psd_majorizer_abs_row_sums`, which is what keeps the
/// bit-for-bit oracle green. Still `O(K)` and still allocation-free.
#[inline]
pub(crate) fn active_softmax_gershgorin_majorizer_entry(a: &[f64], kk: usize, m: f64, scale: f64) -> f64 {
    let l_kk = softmax_entropy_log_plus_one(a[kk]);
    // Diagonal entry H_kk.
    let h_kk = scale * a[kk] * ((m - l_kk - 1.0) + a[kk] * (2.0 * l_kk + 1.0 - 2.0 * m));
    // Pass 1: the row's own squared curvature scale ‖H_k·‖₂².
    let mut sum_sq = h_kk * h_kk;
    for (jj, &a_jj) in a.iter().enumerate() {
        if jj == kk {
            continue;
        }
        let l_jj = softmax_entropy_log_plus_one(a_jj);
        let h_kj = scale * a[kk] * (a_jj * (l_kk + l_jj + 1.0 - 2.0 * m));
        sum_sq += h_kj * h_kj;
    }
    // Pass 2: the soft-abs row sum at that scale, ε_k² = ε₀²·‖H_k·‖₂².
    let eps0 =
        gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::soft_abs_temperature(
            a.len(),
        );
    let eps_sq = eps0 * eps0 * sum_sq;
    let mut acc = gam_terms::analytic_penalties::soft_abs_squared_scale(h_kk, eps_sq);
    // Off-diagonal entries H_kj, j ≠ k.
    for (jj, &a_jj) in a.iter().enumerate() {
        if jj == kk {
            continue;
        }
        let l_jj = softmax_entropy_log_plus_one(a_jj);
        let h_kj = scale * a[kk] * (a_jj * (l_kk + l_jj + 1.0 - 2.0 * m));
        acc += gam_terms::analytic_penalties::soft_abs_squared_scale(h_kj, eps_sq);
    }
    acc
}

/// Active-atom diagonal entry `∂D̃_kk/∂z_w` of the softmax-entropy Gershgorin
/// majorizer derivative (mirrors
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
///
/// Since #2339 the majorized radius is the soft-abs envelope
/// `D̃_kk = Σ_j sqrt(H_kj² + ε₀²‖H_k·‖₂²)`, so the derivative carries the exact
/// row-scale chain term (full derivation on the dense twin):
///
/// ```text
///   ∂D̃_kk/∂z_w = Σ_j (H_kj/s_kj)·Ḣ_kj + ε₀²·(Σ_l H_kl·Ḣ_kl)·Σ_j (1/s_kj),
///   s_kj = sqrt(H_kj² + ε₀²‖H_k·‖₂²).
/// ```
///
/// Two passes over the row (scale, then contraction), diagonal-first then
/// `j ≠ k` ascending — the SAME traversal and arithmetic as the value helper and
/// as the dense twin, which is what keeps the bit-for-bit oracle green.
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
    // `(H_kj, ∂H_kj/∂z_w)` for one column of the row, built from the SAME
    // `(a, l, m)` algebra the dense `row_dense_hessian` /
    // `row_dense_hessian_logit_derivative` pair uses.
    let hessian_entry = |jj: usize| -> (f64, f64) {
        let indicator = if kk == jj { 1.0 } else { 0.0 };
        let l_jj = l(jj);
        let bracket = indicator * (m - l_kk - 1.0) + a[jj] * (l_kk + l_jj + 1.0 - 2.0 * m);
        let dbracket = indicator * (dm - dl_kk)
            + da(jj) * (l_kk + l_jj + 1.0 - 2.0 * m)
            + a[jj] * (dl_kk + dl(jj) - 2.0 * dm);
        (
            scale * a[kk] * bracket,
            scale * (da_kk * bracket + a[kk] * dbracket),
        )
    };
    // Pass 1: ‖H_k·‖₂² and the cross term Σ_l H_kl·Ḣ_kl = ½ ∂‖H_k·‖₂²/∂z_w.
    let (h_kk, dh_kk) = hessian_entry(kk);
    let mut sum_sq = h_kk * h_kk;
    let mut cross = h_kk * dh_kk;
    for jj in 0..a.len() {
        if jj == kk {
            continue;
        }
        let (h_kj, dh_kj) = hessian_entry(jj);
        sum_sq += h_kj * h_kj;
        cross += h_kj * dh_kj;
    }
    // Pass 2: the soft-sign contraction and the reciprocal-scale sum.
    let eps0 =
        gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::soft_abs_temperature(
            a.len(),
        );
    let eps0_sq = eps0 * eps0;
    let eps_sq = eps0_sq * sum_sq;
    let mut acc = 0.0_f64;
    let mut inv_envelope_sum = 0.0_f64;
    let s_kk = gam_terms::analytic_penalties::soft_abs_squared_scale(h_kk, eps_sq);
    if s_kk != 0.0 {
        acc += (h_kk / s_kk) * dh_kk;
        inv_envelope_sum += 1.0 / s_kk;
    }
    for jj in 0..a.len() {
        if jj == kk {
            continue;
        }
        let (h_kj, dh_kj) = hessian_entry(jj);
        let s_kj = gam_terms::analytic_penalties::soft_abs_squared_scale(h_kj, eps_sq);
        if s_kj == 0.0 {
            continue;
        }
        acc += (h_kj / s_kj) * dh_kj;
        inv_envelope_sum += 1.0 / s_kj;
    }
    acc + eps0_sq * cross * inv_envelope_sum
}

/// #2515 — one row's softmax `∂H_tt/∂ρ_sparse` logit block, for the operator the
/// caller's inverse belongs to.
///
/// Both arms are the installed entry itself: the entropy curvature is degree-one
/// in `λ_sparse = e^ρ` (it enters only through `scale = λ·sparsity/τ²`), so
/// `∂/∂ρ` of what the assembly wrote IS what the assembly wrote. What differs is
/// WHICH operator wrote it:
///
/// * `Majorizer` — `B` installs the soft-abs Gershgorin Loewner majorizer
///   `D̃ = diag(Σ_j σ_ε(H_kj))` (#1419/#2339), which is DIAGONAL, so
///   `∂B/∂ρ_sparse` has no off-diagonal entry at all;
/// * `ExactObservedInformation` — `A = B + ΔC` carries the exact dense entropy
///   Hessian `H_kj`, off-diagonal included.
///
/// Their difference is exactly the softmax block of
/// [`SaeManifoldTerm::exact_stationarity_penalty_derivative_delta_by_flat`],
/// entry for entry: `h_entropy − gershgorin` on the diagonal and `h_entropy`
/// off it. Emitting both arms from ONE function is what keeps that identity from
/// drifting — a trace channel and the `ΔC` map that is supposed to explain its
/// gap must not re-derive the same block twice.
///
/// `slot_atoms[s]` is the global atom index occupying local logit slot `s`;
/// `weight` is the row's design weight `w_row`, which the assembly folds into
/// the installed curvature and every ρ-trace must therefore carry. `a` is the
/// row's full-`K` softmax assignment vector and `m` its shared
/// [`softmax_majorizer_log_mean`].
pub(crate) fn softmax_sparse_curvature_rho_derivative_block(
    a: &[f64],
    slot_atoms: &[usize],
    m: f64,
    scale: f64,
    weight: f64,
    operator: EvidenceOperator,
) -> Array2<f64> {
    let slots = slot_atoms.len();
    let mut out = Array2::<f64>::zeros((slots, slots));
    match operator {
        EvidenceOperator::Majorizer => {
            for (slot, &atom) in slot_atoms.iter().enumerate() {
                out[[slot, slot]] =
                    weight * active_softmax_gershgorin_majorizer_entry(a, atom, m, scale);
            }
        }
        EvidenceOperator::ExactObservedInformation => {
            for (row_slot, &row_atom) in slot_atoms.iter().enumerate() {
                for (col_slot, &col_atom) in slot_atoms.iter().enumerate() {
                    out[[row_slot, col_slot]] = weight
                        * softmax_dense_entropy_hessian_entry(a, row_atom, col_atom, m, scale);
                }
            }
        }
    }
    out
}
