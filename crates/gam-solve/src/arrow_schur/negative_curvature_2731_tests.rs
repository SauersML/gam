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

/// #2731 `charts = 32` — the rational exact-A lane conditions its operator from a
/// FIXED-STEP Lanczos, and a Krylov space that has not resolved the bottom of the
/// spectrum prices nothing there. The plan builder then solves shifted systems on
/// an operator that is still indefinite. Its Rademacher probes see only a positive
/// Rayleigh quotient, and the seed CG breaks down with an untyped error, which
/// aborts the fit instead of letting the outer search steer away.
///
/// `S_A = diag(-1/2, 4, 5)`: the one row eliminates `1` from the first border
/// coordinate only, so `e1` carries the k=1 #2515 fixture's geometry (majorizer
/// curvature 3/2, basin `-1/2 + clamp`) and `e2`, `e3` are resolved positive. One
/// conditioning step from any Rademacher start sees the quotient
/// `(-1/2 + 4 + 5)/3` and prices nothing, and every Rademacher probe of a diagonal
/// operator sees the same positive form. So this is exactly the production
/// shape: a missed bottom mode that no one-sided probe can see.
#[test]
fn the_rational_lane_prices_or_refuses_a_bottom_mode_its_fixed_step_conditioning_missed() {
    let exact_a_system = |first_border_curvature: f64, clamp_value: f64| {
        let mut system = ArrowSchurSystem::new(1, 1, 3);
        system.rows[0].htt[[0, 0]] = 1.0;
        system.rows[0].htbeta[[0, 0]] = 1.0;
        system.hbb[[0, 0]] = first_border_curvature;
        system.hbb[[1, 1]] = 4.0;
        system.hbb[[2, 2]] = 5.0;
        system.exact_a_classification = Some(ExactAClassificationGeometry {
            rows: vec![ExactAClassificationRow {
                delta_tt: ndarray::array![[-2.0_f64]],
                delta_tbeta: Array2::<f64>::zeros((1, 0)),
                clamp_diag: ndarray::array![clamp_value],
            }]
            .into(),
            border_indices: std::sync::Arc::from([] as [usize; 0]),
        });
        system
    };
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);
    let lane = || {
        SurrogateLaneState::new(SurrogateLaneConfig {
            num_probes: 4,
            seed: 0x2731,
            rel_tol: 1.0e-10,
            power_iters: 16,
            cg_rel_tol: 1.0e-12,
            cg_max_iters: 64,
            deflation_max_rank: 0,
            deflation_subspace_iters: 1,
            deflation_target_std_err_rel: 1.0,
        })
    };
    let conditioning_steps = 1;

    // Precondition, so a fixture that stops reproducing the defect cannot pass
    // vacuously: the fixed-step conditioning really does miss `e1`.
    let s_a = ndarray::array![[-0.5_f64, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
    let missed = exact_a_ritz_conditioning(
        3,
        |direction| s_a.dot(&direction),
        |_| Ok((1.5, 2.0)),
        conditioning_steps,
        0x2731,
    )
    .expect("one Lanczos step on a finite operator is a conditioning, not a refusal");
    assert!(
        missed.directions.is_empty(),
        "the fixture must reproduce #2731: one conditioning step has to price nothing, \
         got {} priced directions",
        missed.directions.len()
    );

    // The clamp restores a positive basin along the missed mode: priced, so the
    // ladder completes on diag(3/2, 4, 5).
    let (row_logdet, basin_schur) = matrix_free_arrow_evidence_log_det_surrogate(
        &exact_a_system(0.5, 2.0),
        0.0,
        0.0,
        &options,
        4,
        conditioning_steps,
        0x2731,
        Some(&mut lane()),
    )
    .expect("#2731: a clamp-attributable bottom mode the conditioning missed is a priced basin");
    assert!(row_logdet.abs() <= 1.0e-12, "row log|H_tt| {row_logdet:e}");
    assert!(
        (basin_schur - 30.0_f64.ln()).abs() <= 1.0e-7,
        "#2731: log|S| must be priced at the basin, log(3/2·4·5) = {:.12e}, got {basin_schur:.12e}",
        30.0_f64.ln()
    );

    // No clamp: the missed mode is a genuine saddle, refused with the typed marker
    // the outer search maps to an infeasible probe.
    let refusal = matrix_free_arrow_evidence_log_det_surrogate(
        &exact_a_system(0.5, 0.0),
        0.0,
        0.0,
        &options,
        4,
        conditioning_steps,
        0x2731,
        Some(&mut lane()),
    )
    .expect_err("#2731: a missed bottom mode beyond its clamp basin is a saddle")
    .to_string();
    assert!(
        ArrowSchurError::rendered_is_indefinite_evidence(&refusal),
        "#2731: the rational lane must refuse a missed saddle with the typed marker: {refusal}"
    );

    // Negative control: a definite operator is untouched by the missed-mode search.
    let (_, definite_schur) = matrix_free_arrow_evidence_log_det_surrogate(
        &exact_a_system(2.0, 0.0),
        0.0,
        0.0,
        &options,
        4,
        conditioning_steps,
        0x2731,
        Some(&mut lane()),
    )
    .expect("#2731: a definite reduced Schur needs no pricing");
    assert!(
        (definite_schur - 20.0_f64.ln()).abs() <= 1.0e-7,
        "#2731: log|diag(1, 4, 5)| = {:.12e}, got {definite_schur:.12e}",
        20.0_f64.ln()
    );
}
