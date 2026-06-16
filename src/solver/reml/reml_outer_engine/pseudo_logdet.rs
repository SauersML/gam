//! Exact pseudo-logdet eigenspectrum kernels for the REML/LAML criteria:
//! the relative positive-eigenvalue threshold and the exact pseudo-logdet on
//! the positive eigenspace. Both are pure `&[f64] -> f64` scalar kernels with
//! no dependency on the rest of `reml_outer_engine.rs`; relocated here verbatim to shrink
//! the parent module, which re-imports them so every call site is unchanged.

/// Positive-eigenvalue threshold for a given eigenspectrum.
///
/// For a p×p PSD matrix, eigendecomposition introduces errors of order
/// `p × ε_mach × ‖S‖`. True null eigenvalues sit in this noise band.
/// The threshold must be above the noise floor but well below any
/// genuinely positive eigenvalue.
///
/// Uses `p × ε_mach × max(|eigenvalues|, 1)` with a safety factor,
/// giving ~1e-13 × max_ev for typical sizes (p ≤ 1000).
///
/// Threshold is RELATIVE to `max|eigenvalue|` — never floored at an
/// absolute value. Earlier this function clamped `max_ev` to at least
/// `1.0`, which silently classified genuine positive modes of
/// small-scale penalties (Wahba pseudo-spline `m=4` Gram had
/// `max|eig| ≈ 5e-3`) as numerical zero. That corrupted the
/// pseudo-logdet and broke REML's invariance under `S → c·S`, causing
/// the m=4 smooth contribution to collapse to ~0. When `max_ev == 0`
/// (no positive modes) the threshold collapses to 0 too, which is the
/// only correct answer.
pub(crate) fn positive_eigenvalue_threshold(eigenvalues: &[f64]) -> f64 {
    let p = eigenvalues.len();
    let max_ev = eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    // Safety factor above the theoretical noise floor p × ε_mach × ‖S‖, so a
    // genuine small positive mode is never misclassified as numerical zero.
    const SAFETY_FACTOR: f64 = 100.0;
    SAFETY_FACTOR * (p as f64) * f64::EPSILON * max_ev
}

/// Exact pseudo-logdet on the positive eigenspace: L = Σ_{σ_i > threshold} log σ_i.
///
/// No δ-regularization, no nullity parameter. The structural nullspace is
/// identified directly from the eigenspectrum. For PSD penalty sums
/// S(ρ) = Σ exp(ρ_k) S_k, the positive eigenspace is structurally fixed,
/// so this function is C∞ in ρ.
pub(crate) fn exact_pseudo_logdet(eigenvalues: &[f64], threshold: f64) -> f64 {
    eigenvalues
        .iter()
        .filter(|&&s| s > threshold)
        .map(|&s| s.ln())
        .sum()
}
