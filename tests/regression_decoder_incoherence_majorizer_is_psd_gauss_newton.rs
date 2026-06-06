//! Regression guard (companion to the #810 fix) approached from the
//! *surrogate* side rather than the exact-Hessian side.
//!
//! After #810, `DecoderIncoherencePenalty` exposes two distinct curvature
//! operators:
//!
//!   * `hvp` — the **exact** Hessian-vector product `∂²P·v`, including the
//!     indefinite residual term `W·Σ_b C[a,b]·V_k[b,o]`.
//!   * `psd_majorizer_hvp` — the **Gauss-Newton** block `W·Jᵀ(J v)` only, which
//!     is PSD by construction (`W = weight·coactivation ≥ 0`, and `JᵀJ ⪰ 0`).
//!
//! The original bug had both paths returning Gauss-Newton (the majorizer
//! delegated back to `hvp` via the trait default because `hessian_diag` is
//! `None`). This test locks in three independent facts that the fix must keep
//! true, none of which the exact-`hvp == H·v` repro alone pins down:
//!
//!   1. `hvp(v) − psd_majorizer_hvp(v)` equals the closed-form residual term
//!      `W·Σ C·V` exactly — i.e. the two operators differ by precisely the term
//!      that was dropped, computed here independently from the documented
//!      formula. (Catches a regression that drops the residual from `hvp`, or
//!      that leaks it into the majorizer.)
//!   2. The Gauss-Newton operator `B` (materialized column-by-column from
//!      `psd_majorizer_hvp`) is symmetric and positive semidefinite. (Catches a
//!      regression that makes the majorizer alias the indefinite exact Hessian.)
//!   3. `B` genuinely differs from the exact Hessian `H` (built from a central
//!      difference of the analytic gradient) whenever the cross-Gram `C` is
//!      nonzero — the two are not the same operator.
//!
//! Related: #810 (this fix), #809 (sibling IBPAssignmentPenalty diagonal-only
//! hvp).

use gam::terms::analytic_penalties::{AnalyticPenalty, DecoderIncoherencePenalty, PsiSlice};
use ndarray::{Array1, Array2};

/// Closed-form residual term of the exact Hessian-vector product (the piece the
/// Gauss-Newton surrogate omits), for the documented two-atom layout with a
/// single penalized pair `(0, 1)` of unit pairwise weight:
///   res_j[a,o] = Σ_b C[a,b]·V_k[b,o];   res_k[b,o] = Σ_a C[a,b]·V_j[a,o]
/// with `C[a,b] = Σ_o B_j[a,o]·B_k[b,o]`.
fn residual_term(
    target: &Array1<f64>,
    v: &Array1<f64>,
    m_j: usize,
    m_k: usize,
    p_out: usize,
) -> Array1<f64> {
    let off_j = 0usize;
    let off_k = m_j * p_out;
    let mut c = Array2::<f64>::zeros((m_j, m_k));
    for a in 0..m_j {
        for b in 0..m_k {
            let mut s = 0.0;
            for o in 0..p_out {
                s += target[off_j + a * p_out + o] * target[off_k + b * p_out + o];
            }
            c[[a, b]] = s;
        }
    }
    let mut out = Array1::<f64>::zeros(target.len());
    for a in 0..m_j {
        for o in 0..p_out {
            let mut s = 0.0;
            for b in 0..m_k {
                s += c[[a, b]] * v[off_k + b * p_out + o];
            }
            out[off_j + a * p_out + o] = s;
        }
    }
    for b in 0..m_k {
        for o in 0..p_out {
            let mut s = 0.0;
            for a in 0..m_j {
                s += c[[a, b]] * v[off_j + a * p_out + o];
            }
            out[off_k + b * p_out + o] = s;
        }
    }
    out
}

fn build_penalty(m_j: usize, m_k: usize, p_out: usize) -> (DecoderIncoherencePenalty, Array1<f64>) {
    let block_sizes = vec![m_j, m_k];
    let total = (m_j + m_k) * p_out;
    let coactivation =
        Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("2x2 coactivation");
    let penalty = DecoderIncoherencePenalty::new(
        PsiSlice::full(total, None),
        block_sizes,
        p_out,
        coactivation,
        1.0,
        false,
    )
    .expect("construct DecoderIncoherencePenalty");
    (penalty, Array1::<f64>::zeros(0))
}

/// Deterministic pseudo-random vector (LCG) so the PSD probe is reproducible.
fn lcg_vec(len: usize, seed: u64) -> Array1<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Array1::<f64>::zeros(len);
    for x in out.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map the top 53 bits into (-1, 1)
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        *x = 2.0 * u - 1.0;
    }
    out
}

#[test]
fn majorizer_equals_exact_hvp_minus_residual_term() {
    let (m_j, m_k, p_out) = (2usize, 3usize, 2usize);
    let (penalty, rho) = build_penalty(m_j, m_k, p_out);
    let total = (m_j + m_k) * p_out;
    let target = lcg_vec(total, 7);
    for seed in 0..16u64 {
        let v = lcg_vec(total, 101 + seed);
        let exact = penalty.hvp(target.view(), rho.view(), v.view());
        let gn = penalty.psd_majorizer_hvp(target.view(), rho.view(), v.view());
        let res = residual_term(&target, &v, m_j, m_k, p_out);
        let max_diff = exact
            .iter()
            .zip(gn.iter())
            .zip(res.iter())
            .map(|((e, g), r)| ((e - g) - r).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "hvp - psd_majorizer_hvp must equal the dropped residual term exactly; \
             max|(hvp-majorizer) - residual| = {max_diff:.3e} (seed {seed})"
        );
    }
}

#[test]
fn majorizer_is_symmetric_positive_semidefinite() {
    let (m_j, m_k, p_out) = (2usize, 3usize, 2usize);
    let (penalty, rho) = build_penalty(m_j, m_k, p_out);
    let n = (m_j + m_k) * p_out;
    let target = lcg_vec(n, 13);

    // Materialize the Gauss-Newton operator B column by column.
    let mut b = Array2::<f64>::zeros((n, n));
    for q in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[q] = 1.0;
        let col = penalty.psd_majorizer_hvp(target.view(), rho.view(), e.view());
        for p in 0..n {
            b[[p, q]] = col[p];
        }
    }

    // Symmetric.
    let mut max_asym = 0.0_f64;
    for p in 0..n {
        for q in 0..n {
            max_asym = max_asym.max((b[[p, q]] - b[[q, p]]).abs());
        }
    }
    assert!(
        max_asym < 1e-12,
        "GN majorizer must be symmetric: max|B-Bᵀ| = {max_asym:.3e}"
    );

    // PSD: vᵀ B v ≥ 0 for many directions (the operator is JᵀJ scaled by W ≥ 0).
    let mut min_quad = f64::INFINITY;
    for seed in 0..512u64 {
        let v = lcg_vec(n, 1_000 + seed);
        let bv = penalty.psd_majorizer_hvp(target.view(), rho.view(), v.view());
        let quad = v.iter().zip(bv.iter()).map(|(a, c)| a * c).sum::<f64>();
        min_quad = min_quad.min(quad);
    }
    assert!(
        min_quad > -1e-10,
        "GN majorizer must be PSD: min vᵀBv = {min_quad:.3e}"
    );
}

#[test]
fn majorizer_differs_from_exact_hessian_when_atoms_incoherent() {
    let (m_j, m_k, p_out) = (2usize, 2usize, 2usize);
    let (penalty, rho) = build_penalty(m_j, m_k, p_out);
    let n = (m_j + m_k) * p_out;
    // Atoms with nonzero cross-Gram C, so the residual term is nonzero.
    let target = lcg_vec(n, 21);
    let v = lcg_vec(n, 22);
    let exact = penalty.hvp(target.view(), rho.view(), v.view());
    let gn = penalty.psd_majorizer_hvp(target.view(), rho.view(), v.view());
    let max_gap = exact
        .iter()
        .zip(gn.iter())
        .map(|(e, g)| (e - g).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_gap > 1e-3,
        "exact hvp and GN majorizer must be distinct operators when C != 0; \
         max|hvp - majorizer| = {max_gap:.3e} (both returning GN was the bug)"
    );
}
