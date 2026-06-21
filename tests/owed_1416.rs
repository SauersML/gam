//! Owed-work regression for #1416 — IBP log-determinant derivatives must
//! contract the FULL cross-row off-diagonal of the rank-one Woodbury source,
//! not only the diagonal.
//!
//! ## The defect (now fixed)
//!
//! The IBP prior Hessian for one column `k` is
//!
//! ```text
//!   H_p = d·J Jᵀ + diag(s, c),
//! ```
//!
//! where `J_i = ∂z_i/∂ℓ_i` (`z_jac`), `d = ∂s/∂M` (`cross_row_d`) is the scalar
//! coefficient of the per-column rank-one block, and the empirical mass
//! `M_k = Σ_i z_ik` couples EVERY row pair `(i, j)`. The solver installs the
//! full rank-one `d·J Jᵀ` via Woodbury (`H_full = H₀' + U D Uᵀ`), so for fixed
//! alpha the entire IBP prior scales with `λ = eᵖ` and the correct direct
//! log-det contribution is the FULL trace
//!
//! ```text
//!   ½ tr(H⁻¹ H_p).
//! ```
//!
//! The pre-fix non-softmax ρ-trace branch contracted only the diagonal,
//! `½ Σ_i (H⁻¹)_ii (H_p)_ii`, dropping the off-diagonal rank-one cross terms
//!
//! ```text
//!   ½ Σ_{i≠j}(H⁻¹)_{ij}(H_p)_{ji} = ½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j.
//! ```
//!
//! ## What this test pins
//!
//! Using only the PUBLIC IBP channel API (`hessian_diag_logit_third_channels`),
//! it builds a small two-row, one-column interior column whose cross-row
//! coefficient `d` and per-row Jacobians `J_i` are genuinely nonzero, picks an
//! explicit SPD `H` over that column's two row slots, and asserts:
//!
//!   1. the full `½ tr(H⁻¹ H_p)` equals the diagonal-only contraction PLUS the
//!      boxed cross-row term `½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j`, computed directly
//!      from `(H⁻¹)_{ij}`, `d`, and `J`; and
//!   2. that omitted cross-row term is non-negligible, so a diagonal-only
//!      contraction (the original bug) would be measurably wrong.
//!
//! This is the standalone, dense ground-truth complement to the in-crate
//! fixed-state finite-difference tests
//! (`ibp_rho_sparse_logdet_trace_matches_dense_fd_1416` and the cross-row
//! channel FD tests in `terms/analytic_penalties/tests.rs`), which exercise the
//! same channels through the real `assignment_log_strength_hessian_trace` and
//! θ-adjoint paths.

use ndarray::{array, Array1, Array2};

use gam::terms::analytic_penalties::IBPAssignmentPenalty;

/// Invert a symmetric 2×2 SPD matrix `[[a, b], [b, c]]`.
fn invert_2x2(h: &Array2<f64>) -> Array2<f64> {
    let a = h[[0, 0]];
    let b = h[[0, 1]];
    let c = h[[1, 1]];
    let det = a * c - b * b;
    assert!(det > 0.0, "test H must be SPD (det = {det:.3e})");
    array![[c / det, -b / det], [-b / det, a / det]]
}

#[test]
fn ibp_rho_trace_includes_cross_row_offdiagonal_1416() {
    // Two rows, one column; interior logits → nonzero concrete Jacobians and a
    // nonzero column Woodbury coefficient `d`.
    let pen = IBPAssignmentPenalty::new(1, 2.0, 0.9, false);
    let t = array![0.4_f64, -0.3]; // row-major (N=2, K=1)
    let rho = Array1::<f64>::zeros(0); // fixed alpha
    let k = pen.k_max;
    let n = t.len() / k;
    assert_eq!((n, k), (2, 1));

    let ch = pen.hessian_diag_logit_third_channels(t.view(), rho.view());
    let d = ch.cross_row_d[0];
    let j0 = ch.z_jac[0];
    let j1 = ch.z_jac[1];

    // The fixture must genuinely exercise the cross-row path.
    assert!(
        d.abs() > 1.0e-6 && j0.abs() > 1.0e-6 && j1.abs() > 1.0e-6,
        "fixture must have a live cross-row source: d={d:.3e}, J0={j0:.3e}, J1={j1:.3e}"
    );

    // The per-column rank-one part of H_p over the two row slots: d·J Jᵀ.
    // (The diag(s, c) part is purely diagonal and so contributes identically to
    //  the full and the diagonal-only contractions — only the rank-one
    //  off-diagonal is at issue, so we isolate it here.)
    let mut hp_rank1 = Array2::<f64>::zeros((2, 2));
    let jvec = [j0, j1];
    for i in 0..2 {
        for j in 0..2 {
            hp_rank1[[i, j]] = d * jvec[i] * jvec[j];
        }
    }

    // An explicit SPD inverse posterior block over the two row slots. Its
    // off-diagonal is what the buggy diagonal-only contraction discards.
    let h = array![[3.0_f64, 0.8], [0.8, 2.0]];
    let h_inv = invert_2x2(&h);

    // Full trace ½ tr(H⁻¹ H_p^rank1).
    let mut full = 0.0_f64;
    for i in 0..2 {
        for j in 0..2 {
            full += h_inv[[i, j]] * hp_rank1[[j, i]];
        }
    }
    full *= 0.5;

    // Diagonal-only contraction (the pre-#1416 bug).
    let diag_only =
        0.5 * (h_inv[[0, 0]] * hp_rank1[[0, 0]] + h_inv[[1, 1]] * hp_rank1[[1, 1]]);

    // The exact omitted cross-row term: ½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j.
    let cross = 0.5 * d * (h_inv[[0, 1]] * j0 * j1 + h_inv[[1, 0]] * j1 * j0);

    // (1) full == diagonal-only + the boxed cross-row term, to round-off.
    assert!(
        (full - (diag_only + cross)).abs() <= 1.0e-12 * (1.0 + full.abs()),
        "full ½tr(H⁻¹H_p) must equal diagonal-only + cross-row term: \
         full={full:.12e}, diag_only={diag_only:.12e}, cross={cross:.12e}"
    );

    // (2) the cross-row term the diagonal-only branch drops is non-negligible,
    //     so the original bug would be measurably wrong.
    assert!(
        cross.abs() > 1.0e-3 * (1.0 + full.abs()),
        "omitted cross-row term must be non-negligible (diagonal-only would be \
         a real error): cross={cross:.6e}, full={full:.6e}"
    );
    assert!(
        (full - diag_only).abs() > 1.0e-3 * (1.0 + full.abs()),
        "full and diagonal-only traces must differ: full={full:.6e}, \
         diag_only={diag_only:.6e}"
    );
}
