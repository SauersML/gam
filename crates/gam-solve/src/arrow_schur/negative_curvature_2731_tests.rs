//! #2731 — a negative-curvature mode of the reduced Schur, carried into the
//! coordinates a fit can actually step in.
//!
//! The curved SAE tier converges at a point where the reduced Schur is
//! indefinite: the run reported `pᵀ(A + σI)p = -1.502010e10` at
//! `σ = 4.207073e-15` on `dim 288`. A conjugate-gradient recurrence cannot
//! produce that on a positive-definite operator, so the point is a saddle of
//! the fit's own objective and `log|S|` is undefined there. What was missing
//! was not the mode — the CG breakdown produces one, and Lanczos produces one
//! on demand — but the map from the mode's own coordinate system into the
//! `(delta_t, delta_beta)` a displacement is expressed in. The reduced Schur
//! lives in the ELIMINATED system; a step does not.
//!
//! That map is the arrow back-substitution's linear half, and it was already
//! being computed twice over and discarded: inside every `S·v` apply
//! (`schur_matvec_row_into` solves `H_tt⁻¹ H_tβ v` and keeps only the product
//! back through `H_βt`), and inside every Newton step
//! (`back_substitute_delta_t`, which adds the affine `g_t` term a direction
//! must not carry).
//!
//! The fixture below pins the identity that makes the whole thing valid rather
//! than plausible:
//!
//! ```text
//!     [L(v); v]ᵀ H [L(v); v]  =  vᵀ S v      for every v
//! ```
//!
//! It is an equality, not a bound, because the Schur complement IS the full
//! Hessian's quadratic form restricted to the graph of `L`. So negative
//! curvature measured on the reduced operator is negative curvature of the
//! full objective — the escape is entitled to the direction it found.

use super::*;
use super::tests::dense_direct_system;

/// The dense full arrow Hessian `[[blkdiag(H_tt), H_tβ], [H_βt, H_ββ + ρ_β I]]`
/// as one `(n·d + k)` matrix, and the quadratic form of a `(t, β)` direction on
/// it. Assembled ONLY here, as the oracle: the production path never forms it,
/// which is the entire reason the mode has to be lifted rather than read off an
/// eigendecomposition.
fn full_arrow_quadratic_form(
    sys: &ArrowSchurSystem,
    ridge_beta: f64,
    eliminated: &Array1<f64>,
    border: &Array1<f64>,
) -> f64 {
    let mut total = 0.0;
    for (i, row) in sys.rows.iter().enumerate() {
        let base = sys.row_offsets[i];
        let di = sys.row_dims[i];
        for r in 0..di {
            for c in 0..di {
                total += eliminated[base + r] * row.htt[[r, c]] * eliminated[base + c];
            }
            // The cross term appears twice in the symmetric form.
            for c in 0..sys.k {
                total += 2.0 * eliminated[base + r] * row.htbeta[[r, c]] * border[c];
            }
        }
    }
    for r in 0..sys.k {
        for c in 0..sys.k {
            total += border[r] * sys.hbb[[r, c]] * border[c];
        }
        total += ridge_beta * border[r] * border[r];
    }
    total
}

/// A well-conditioned arrow system whose BORDER carries one strongly negative
/// direction, so the reduced Schur is indefinite while every per-row `H_tt`
/// block stays SPD and factors. That is the production shape: the eliminated
/// blocks are fine and the Schur complement is not, which is exactly why a
/// blockwise PSD screen (`StreamedFrameCurvature`, PSD by construction) cannot
/// see this and a Schur-level measurement must.
fn indefinite_border_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
    let mut sys = dense_direct_system(n, d, k);
    // One coordinate of `H_ββ` pushed below the mass the elimination removes.
    sys.hbb[[0, 0]] = -5.0;
    sys.refresh_row_hessian_fingerprint();
    sys
}

/// The lift is EXACT: the full-space curvature of `(L(v), v)` equals `vᵀSv` for
/// an arbitrary direction, not merely for an eigenvector.
///
/// Asserted on a direction with no special relationship to the spectrum, so a
/// regression that made the lift correct only on eigenvectors (for instance by
/// re-solving with the eigenvalue folded in) still fails here.
#[test]
fn lifting_a_border_direction_preserves_its_quadratic_form_exactly() {
    let (n, d, k) = (12usize, 3usize, 10usize);
    let sys = indefinite_border_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1.0e-6;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("per-row blocks stay SPD in this fixture");
    let op = ReducedSchurOperator::new(&sys, &htt_factors, ridge_beta, &backend, None);

    // An arbitrary direction: not an eigenvector, not sparse, not symmetric.
    let mut v = Array1::<f64>::zeros(k);
    for (j, slot) in v.iter_mut().enumerate() {
        *slot = ((j + 1) as f64).sin() + 0.3 * ((j + 1) as f64).cos();
    }
    let inv = v.dot(&v).sqrt().recip();
    v.mapv_inplace(|x| x * inv);

    let reduced = v.dot(&op.apply_owned(&v));
    let lifted = arrow_lift_border_direction(&sys, &htt_factors, v.view(), &backend);
    let full = full_arrow_quadratic_form(&sys, ridge_beta, &lifted, &v);

    let scale = reduced.abs().max(full.abs()).max(1.0);
    assert!(
        (reduced - full).abs() <= 1.0e-11 * scale,
        "the Schur complement is the full form on the graph of the lift, so these \
         are one number: reduced vᵀSv = {reduced:.17e}, full [L(v);v]ᵀH[L(v);v] = \
         {full:.17e} (gap {:.3e}, scale {scale:.3e})",
        (reduced - full).abs()
    );

    // The affine Newton back-substitution must differ from the lift by exactly
    // the row-gradient term and by nothing else — the two share one owner and
    // this is what pins that they stayed one owner.
    let newton = back_substitute_delta_t(&sys, &htt_factors, v.view(), &backend);
    let zero = Array1::<f64>::zeros(k);
    let affine = back_substitute_delta_t(&sys, &htt_factors, zero.view(), &backend);
    for i in 0..newton.len() {
        let expected = affine[i] + lifted[i];
        assert!(
            (newton[i] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "back_substitute_delta_t must be its affine part plus the linear lift \
             at index {i}: {} vs {expected}",
            newton[i]
        );
    }
}

/// The shifted solve finds the algebraically most-negative eigenpair, certifies
/// it by an APPLY rather than by the Ritz value, and hands back a full-space
/// displacement whose curvature is that same negative number.
#[test]
fn the_reduced_schur_negative_mode_lifts_to_full_space_negative_curvature() {
    let (n, d, k) = (16usize, 3usize, 12usize);
    let sys = indefinite_border_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1.0e-6;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("per-row blocks stay SPD in this fixture");

    let lambda_max = reduced_schur_lambda_max(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        200,
        0x2731_C0DE,
    )
    .expect("the border still carries positive curvature, so λ_max is positive");

    let found = reduced_schur_negative_curvature(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        lambda_max,
        64,
        0x2731_C0DE,
    )
    .expect("this fixture's reduced Schur is indefinite by construction");

    assert!(
        found.curvature < 0.0,
        "a returned mode must carry measured negative curvature, got {}",
        found.curvature
    );
    // The measured Rayleigh quotient and the certified Ritz value are the same
    // number here; a regression that returned the mode of the FOLDED operator
    // without unfolding the eigenvalue fails this and not the sign check above.
    let scale = found.curvature.abs().max(found.ritz_eigenvalue.abs()).max(1.0);
    assert!(
        (found.curvature - found.ritz_eigenvalue).abs() <= 1.0e-6 * scale,
        "the apply-measured curvature {} and the Ritz eigenvalue {} must agree",
        found.curvature,
        found.ritz_eigenvalue
    );

    // The whole point: the same negative number in the full coordinates.
    let full = full_arrow_quadratic_form(&sys, ridge_beta, &found.eliminated, &found.border);
    assert!(
        (full - found.curvature).abs() <= 1.0e-9 * found.curvature.abs().max(1.0),
        "the lifted displacement must carry the mode's curvature into the full \
         system: reduced {} vs full {full}",
        found.curvature
    );
    assert!(
        full < 0.0,
        "the fit's own objective must be shown to descend here, got {full}"
    );
    assert_eq!(found.eliminated.len(), sys.row_offsets[sys.rows.len()]);
    assert_eq!(found.border.len(), k);
}

/// The negative control. On a positive-definite reduced Schur the search must
/// report NO negative direction — a certificate that fires on a well-behaved
/// minimum would refuse every converged curved-tier fit, which is a worse
/// failure than the one this fixes.
#[test]
fn a_definite_reduced_schur_yields_no_negative_direction() {
    let (n, d, k) = (16usize, 3usize, 12usize);
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1.0e-6;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let lambda_max = reduced_schur_lambda_max(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        200,
        0x2731_C0DE,
    )
    .expect("λ_max is positive on an SPD reduced Schur");
    let found = reduced_schur_negative_curvature(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        lambda_max,
        64,
        0x2731_C0DE,
    );
    assert!(
        found.is_none(),
        "an SPD reduced Schur has no negative direction to report, got {found:?}"
    );
}
