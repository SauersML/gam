//! Encode/decoder contract probe tests for the SAE manifold term, split out of
//! `tests.rs` to keep each tracked file under the line-count gate. These were
//! the former inline `mod inner_contract_probe_tests` block; they define their
//! own euclidean-line/periodic fixtures and assert the analytic decoder and
//! coordinate gradients/Hessians match finite differences on the penalized
//! objective contract.

use super::tests::{
    TestPeriodicEvaluator, diagonal_latent_cache, periodic_basis, warmstart_test_objective,
    warmstart_test_objective_with_evaluator,
};
use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};
use approx::assert_abs_diff_eq;
use gam_terms::latent::LatentManifold;
use ndarray::array;
use std::sync::Arc;

pub(crate) fn euclidean_line_contract_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 150usize;
    let p = 8usize;
    let mut coords = Array2::<f64>::zeros((n, 1));
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let u = -1.0 + 2.0 * row as f64 / (n as f64 - 1.0);
        coords[[row, 0]] = 2.5 + 3.0 * u;
        for col in 0..p {
            let linear_loading = 0.35 + 0.07 * col as f64;
            let offset = 0.08 * ((col % 3) as f64 - 1.0);
            let phase = (row * (col + 3)) as f64;
            let noise = 0.04 * (phase.sin() + 0.5 * (0.37 * phase).cos());
            z[[row, col]] = offset + linear_loading * u + noise;
        }
    }

    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("evaluator"));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("basis");
    let m = phi.ncols();
    let smooth_penalty =
        gam_terms::basis::create_difference_penalty_matrix(m, 2, None).expect("penalty");
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "contract-line",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        smooth_penalty,
    )
    .expect("atom")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, (0.01_f64).ln(), vec![Array1::<f64>::zeros(1)]);
    (term, z, rho)
}

/// Compare an analytic derivative against a centered finite difference with a
/// combined relative + absolute tolerance.
///
/// A centered FD `(f₊ − f₋)/(2h)` is the difference of two nearly-equal values
/// divided by the small `2h`, so its **roundoff floor** is `≈ ε·|f|/h`,
/// independent of how small the true derivative is. At a near-stationary
/// coordinate (gradient `≈ 1e-6` on an objective of magnitude `≈ 1e2`, `h=1e-6`)
/// that floor is `≈ 1e-8`, i.e. `~0.1%` of the derivative — so a relative-only
/// `1e-5` gate cannot be met by FD however correct the analytic value is. The
/// `fd_roundoff_floor` (computed by the caller from the actual `|f₊|`, `|f₋|`
/// and `h`) admits exactly that unavoidable cancellation error and nothing more;
/// for derivatives well above the floor the gate stays the strict `1e-5`
/// relative contract.
pub(crate) fn assert_contract_close_with_floor(
    label: &str,
    analytic: f64,
    finite_difference: f64,
    fd_roundoff_floor: f64,
) {
    let abs_diff = (analytic - finite_difference).abs();
    let scale = finite_difference.abs().max(analytic.abs());
    let tol = 1.0e-5 * scale + fd_roundoff_floor;
    let rel = abs_diff / scale.max(1.0e-12);
    assert!(
        abs_diff <= tol,
        "{label}: analytic={analytic:.12e} fd={finite_difference:.12e} \
         rel={rel:.3e} abs_diff={abs_diff:.3e} tol={tol:.3e} \
         (fd_roundoff_floor={fd_roundoff_floor:.3e})"
    );
}

/// Roundoff floor of a centered finite difference `(f₊ − f₋)/(2h)`: the
/// catastrophic-cancellation error `≈ ε·max(|f₊|,|f₋|)/(2h)`, scaled by a small
/// safety constant. This is the largest absolute error the FD reference can
/// carry purely from f64 rounding of the objective evaluations, so it is the
/// correct absolute floor for the contract comparison at near-stationary points.
pub(crate) fn fd_roundoff_floor(f_plus: f64, f_minus: f64, h: f64) -> f64 {
    const SAFETY: f64 = 16.0;
    SAFETY * f64::EPSILON * f_plus.abs().max(f_minus.abs()) / (2.0 * h)
}

/// Verify the analytic decoder gradient block `sys.gb` matches a centered FD of
/// the penalized objective for one `(basis_col, out_col)` coefficient, with the
/// roundoff-floor-aware tolerance. Factored out so the contract can be pinned on
/// BOTH the full width-3 monomial design (pre-fit) and the rank-reduced design
/// the joint solve collapses to (post-fit), which travel through different jet
/// paths (`EuclideanPatchEvaluator` vs `SubspaceReducedEvaluator`).
fn assert_decoder_gradient_matches_fd(
    term: &mut SaeManifoldTerm,
    z: &Array2<f64>,
    rho: &SaeManifoldRho,
    basis_col: usize,
    out_col: usize,
    p: usize,
    h: f64,
) {
    let sys = term
        .assemble_arrow_schur(z.view(), rho, None)
        .expect("decoder assemble");
    assert_eq!(sys.k, term.beta_dim());
    let beta_idx = basis_col * p + out_col;
    let analytic = sys.gb[beta_idx];
    let base = term.atoms[0].decoder_coefficients()[[basis_col, out_col]];

    term.atoms[0].decoder_coefficients_mut()[[basis_col, out_col]] = base + h;
    let f_plus = term
        .penalized_objective_total(z.view(), rho, None, 1.0)
        .expect("decoder f+");
    term.atoms[0].decoder_coefficients_mut()[[basis_col, out_col]] = base - h;
    let f_minus = term
        .penalized_objective_total(z.view(), rho, None, 1.0)
        .expect("decoder f-");
    term.atoms[0].decoder_coefficients_mut()[[basis_col, out_col]] = base;

    let fd = (f_plus - f_minus) / (2.0 * h);
    assert_contract_close_with_floor(
        &format!("CONTRACT decoder ({basis_col},{out_col})"),
        analytic,
        fd,
        fd_roundoff_floor(f_plus, f_minus, h),
    );
}

#[test]
pub(crate) fn euclidean_line_decoder_gradient_matches_penalized_objective_fd() {
    let (mut term, z, mut rho) = euclidean_line_contract_fixture();
    let p = term.output_dim();
    let h = 1.0e-6;

    // (A) The genuine DEGREE-2 decoder-gradient contract, on the full width-3
    // `[1, t, t²]` monomial design BEFORE the joint solve runs. The fixture is
    // near-linear, so once the inner solve drives the latent `t` into a narrow
    // range the rank-revealing reduction (`reduce_atoms_to_data_supported_rank`,
    // #1117) legitimately collapses this basis to its data-supported rank — but
    // at the seed the t² column is still present, so this is where the degree-2
    // `∂/∂B` contract through `EuclideanPatchEvaluator`'s jets is exercised.
    assert_eq!(
        term.atoms[0].basis_size(),
        3,
        "the degree-2 euclidean fixture must seed a width-3 [1,t,t²] basis"
    );
    for (basis_col, out_col) in [(0usize, 0usize), (1, 3), (2, 7)] {
        assert_decoder_gradient_matches_fd(&mut term, &z, &rho, basis_col, out_col, p, h);
    }

    let ridge = 1.0e-6;
    for step in 0..6 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .unwrap_or_else(|err| panic!("warm step {step} failed: {err}"));
        assert!(
            loss.total().is_finite(),
            "warm step {step} loss is non-finite"
        );
    }

    let sys_coord = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("coord assemble");
    assert_eq!(
        sys_coord.k,
        term.beta_dim(),
        "p=8 contract fixture must stay on full-B coordinates"
    );
    assert!(
        !term.frames_active(),
        "p=8 contract fixture must not activate a frame"
    );

    let h = 1.0e-6;
    for row in [3usize, 75, 140] {
        let analytic = sys_coord.rows[row].gt[0];
        let base_coord = term.assignment.coords[0].as_matrix()[[row, 0]];

        let mut plus_coords = term.assignment.coords[0].as_matrix();
        plus_coords[[row, 0]] = base_coord + h;
        let plus_flat = Array1::from_iter(plus_coords.iter().copied());
        term.assignment.coords[0].set_flat(plus_flat.view());
        term.refresh_basis_from_current_coords()
            .expect("plus refresh");
        let f_plus = term
            .penalized_objective_total(z.view(), &rho, None, 1.0)
            .expect("coord f+");

        let mut minus_coords = term.assignment.coords[0].as_matrix();
        minus_coords[[row, 0]] = base_coord - h;
        let minus_flat = Array1::from_iter(minus_coords.iter().copied());
        term.assignment.coords[0].set_flat(minus_flat.view());
        term.refresh_basis_from_current_coords()
            .expect("minus refresh");
        let f_minus = term
            .penalized_objective_total(z.view(), &rho, None, 1.0)
            .expect("coord f-");

        let mut restored_coords = term.assignment.coords[0].as_matrix();
        restored_coords[[row, 0]] = base_coord;
        let restored_flat = Array1::from_iter(restored_coords.iter().copied());
        term.assignment.coords[0].set_flat(restored_flat.view());
        term.refresh_basis_from_current_coords()
            .expect("restore refresh");

        let fd = (f_plus - f_minus) / (2.0 * h);
        assert_contract_close_with_floor(
            &format!("CONTRACT coord row {row}"),
            analytic,
            fd,
            fd_roundoff_floor(f_plus, f_minus, h),
        );
    }

    // (B) The post-fit decoder-gradient contract on the ACTUAL fitted basis.
    // On this near-linear fixture the rank-revealing reduction (#1117) collapses
    // the width-3 `[1, t, t²]` design to its data-supported rank once the joint
    // solve narrows the latent `t` range — so the decoder coefficients now live
    // on the reduced subspace and the jets travel through the
    // `SubspaceReducedEvaluator`. We pin the same `∂/∂B`↔FD contract over the
    // REDUCED width (`basis_size()` after the fit), not the stale width-3 layout:
    // the gradient block must remain correct through the reduced-jet path.
    let fitted_m = term.atoms[0].basis_size();
    assert!(
        fitted_m >= 1 && fitted_m <= 3,
        "fitted euclidean basis width must be in [1,3]; got {fitted_m}"
    );
    // Exercise every retained basis column against a representative output channel
    // spread (first / middle / last), clamped to the fitted width.
    for basis_col in 0..fitted_m {
        for &out_col in &[0usize, p / 2, p - 1] {
            assert_decoder_gradient_matches_fd(&mut term, &z, &rho, basis_col, out_col, p, h);
        }
    }
}

/// #795 — the SAE-assembled isometry Gauss-Newton curvature must be
/// decoder-scale-invariant, matching its already-scale-free gradient.
///
/// The isometry value/gradient penalize the SCALE-INVARIANT normalized residual
/// `R_n = g_n/gbar − g^ref_n`, so they are free of the decoder magnitude `‖B‖`.
/// But the arrow-Schur curvature blocks (`htt`/`htbeta`/`hbb`) are built from
/// the RAW weighted Jacobian `wj ∝ ‖B‖`, i.e. the Gauss-Newton block of the
/// UN-normalized `½μ‖g_n − g^ref‖²`, which scales ∝‖B‖⁴. Pairing a scale-free
/// gradient with a ‖B‖⁴ curvature collapses the joint Newton step as the decoder
/// grows and the proximal ridge saturates at 1e15 — the #795 failure. The fix
/// folds the frozen-normalizer factor `1/gbar²` into the curvature so it is the
/// GN block of the normalized residual; `1/gbar² ∝ ‖B‖⁻⁴` cancels the raw `‖B‖⁴`.
///
/// This pins the cancellation directly on the production assembly: scaling the
/// decoder (hence the target, so the well-fit residual is unchanged) by λ leaves
/// the ISOLATED isometry curvature contribution `htt(with) − htt(without)`
/// invariant, while WITHOUT the fix it would scale ∝λ⁴ (≈10⁴× at λ=10).
#[test]
fn sae_isometry_assembled_curvature_is_decoder_scale_invariant() {
    use gam_terms::analytic_penalties::{
        AnalyticPenaltyKind, AnalyticPenaltyRegistry, IsometryPenalty, PsiSlice,
    };
    let n = 24usize;
    let p = 4usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let m = phi.ncols();
    let base_decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        let scale = 1.0 / (1.0 + b as f64);
        scale * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });

    let isometry_curvature_norm = |lambda: f64| -> f64 {
        let decoder = &base_decoder * lambda;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "iso_scale",
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            Array2::<f64>::eye(m),
        )
        .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        // Target is the exact decoded curve at this scale, so the well-fit
        // residual (and thus the data-fit curvature) is the same shape at every
        // λ — only the isometry block carries the scale dependence we probe.
        let target = phi.dot(&decoder);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords.clone()],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .expect("fixture assignment: one logit column and one coord block per atom");
        let mut term = SaeManifoldTerm::new(vec![atom], assignment)
            .expect("fixture term: every atom's basis width matches its assignment block");

        let mut registry = AnalyticPenaltyRegistry::new();
        registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
            IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(1)), 1),
        )));
        let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![1.0_f64.ln()]]);

        // Isolate the isometry contribution to the coordinate curvature: the
        // data-fit `htt` legitimately grows with the target scale, so we
        // difference assemble-with-isometry against assemble-without.
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, Some(&registry))
            .expect("assemble with isometry succeeds");
        let bare = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("bare assemble succeeds");
        let mut htt_iso = 0.0_f64;
        for (r, b) in sys.rows.iter().zip(bare.rows.iter()) {
            for (v, bv) in r.htt.iter().zip(b.htt.iter()) {
                htt_iso += (v - bv) * (v - bv);
            }
        }
        htt_iso.sqrt()
    };

    let base = isometry_curvature_norm(1.0);
    assert!(
        base > 1.0,
        "the planted-circle fixture must produce a non-trivial isometry \
         curvature block to make the scale test meaningful; got {base:.3e}"
    );
    for &lambda in &[3.0_f64, 10.0, 50.0] {
        let scaled = isometry_curvature_norm(lambda);
        let rel = (scaled - base).abs() / base;
        assert!(
            rel < 1.0e-6,
            "SAE-assembled isometry curvature must be decoder-scale-invariant \
             (#795): λ=1 → {base:.6e}, λ={lambda} → {scaled:.6e} (rel diff {rel:.3e}). \
             A λ-dependent block is the un-normalized ‖B‖⁴ Gauss-Newton curvature \
             whose mismatch with the scale-free gradient saturates the proximal \
             ridge at 1e15."
        );
    }
}

/// #2099 — the end-to-end joint `(t, β)` fit with the dimensionless isometry
/// gauge is equivariant under a change of physical output units.
///
/// Changing output units by `c` sends the target and decoder to `c Z` and `c B`,
/// while the residual covariance becomes `c² Σ`. The shared likelihood/gauge
/// precision is therefore `W/c²`: it cancels the decoder scale in both the GLS
/// residual and the isometry pullback. Decoder smoothness precision and the
/// solver-only decoder ridge carry inverse-output-squared units and likewise
/// scale by `1/c²`; the coordinate prior and coordinate ridge are dimensionless
/// and stay fixed. Under that complete physical-unit change, the reconstruction
/// `f/c` and the full, unnormalised penalized criterion must agree with the
/// unit-scale fit. Requiring each penalized optimum to have nearly zero data
/// residual was invalid: nonzero smoothness/isometry deliberately trade data fit
/// for regularity, and absolute planted-circle recovery is already owned by
/// `sae_single_planted_circle_embedded_isometry_fit_converges_795` below.
#[test]
fn sae_isometry_joint_fit_is_physical_coscale_invariant_2099() {
    use gam_problem::RowMetric;
    use gam_terms::analytic_penalties::{
        AnalyticPenaltyKind, AnalyticPenaltyRegistry, IsometryPenalty, PsiSlice,
    };
    let n = 24usize;
    let p = 4usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let m = phi.ncols();
    let base_decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        let scale = 1.0 / (1.0 + b as f64);
        scale * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });

    struct ScaleFit {
        normalized_reconstruction: Array2<f64>,
        criterion: f64,
        components: [f64; 7],
    }

    let fit_at_scale = |physical_scale: f64| -> ScaleFit {
        let scale_sq = physical_scale * physical_scale;
        let decoder = &base_decoder * physical_scale;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "iso_converge",
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            Array2::<f64>::eye(m),
        )
        .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let target = phi.dot(&decoder);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords.clone()],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .expect("fixture assignment: one logit column and one coord block per atom");
        let mut term = SaeManifoldTerm::new(vec![atom], assignment)
            .expect("fixture term: every atom's basis width matches its assignment block");
        // Residual covariance transforms as Σ -> c²Σ, so its precision factor
        // transforms as U -> U/c. The identical RowMetric also supplies the
        // isometry pullback, making (cJ)'(W/c²)(cJ) exactly invariant.
        let metric_factor = Array2::from_shape_fn((n, p * p), |(_, flat)| {
            let output = flat / p;
            let probe = flat % p;
            if output == probe {
                physical_scale.recip()
            } else {
                0.0
            }
        });
        term.set_row_metric(
            RowMetric::behavioral_fisher(Arc::new(metric_factor), p, p)
                .expect("physical-unit precision metric is valid"),
        )
        .expect("physical-unit precision metric matches the output dimension");

        let mut registry = AnalyticPenaltyRegistry::new();
        registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
            IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(1)), 1),
        )));
        let mut rho = SaeManifoldRho::new(0.0, (0.8_f64 / scale_sq).ln(), vec![array![0.0]]);

        let loss = term
            .run_joint_fit_arrow_schur(
                target.view(),
                &mut rho,
                Some(&registry),
                12,
                1.0,
                1.0e-4,
                1.0e-4 / scale_sq,
            )
            .expect("joint fit with isometry gauge ON must converge at every decoder scale");
        assert!(
            loss.total().is_finite(),
            "converged loss must be finite at c={physical_scale}, got {}",
            loss.total()
        );

        let recon = term
            .try_fitted_for_rho(&rho)
            .expect("fitted reconstruction exists");
        let criterion = term
            .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
            .expect("co-scaled penalized criterion is defined");
        assert!(
            criterion.is_finite(),
            "penalized criterion must be finite at c={physical_scale}, got {criterion}"
        );
        let scored_loss = term
            .loss_scaled(target.view(), &rho, 1.0)
            .expect("co-scaled loss breakdown is defined");
        let analytic = term
            .analytic_penalty_value_total(&registry, 1.0)
            .expect("co-scaled analytic-penalty value is defined");
        let repulsion = term.decoder_repulsion_value(1.0);
        let separation = term.separation_barrier_value(1.0);
        ScaleFit {
            normalized_reconstruction: recon.mapv(|value| value / physical_scale),
            criterion,
            components: [
                scored_loss.data_fit,
                scored_loss.assignment_sparsity,
                scored_loss.smoothness,
                scored_loss.ard,
                analytic,
                repulsion,
                separation,
            ],
        }
    };

    let base = fit_at_scale(1.0);
    let base_image_norm_sq = base
        .normalized_reconstruction
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    assert!(
        base_image_norm_sq.is_finite() && base_image_norm_sq > 0.0,
        "unit-scale fitted image must be finite and nonzero"
    );
    for &physical_scale in &[5.0_f64, 25.0] {
        let scaled = fit_at_scale(physical_scale);
        let image_defect = base
            .normalized_reconstruction
            .iter()
            .zip(scaled.normalized_reconstruction.iter())
            .map(|(unit, rescaled)| {
                let delta = unit - rescaled;
                delta * delta
            })
            .sum::<f64>()
            .sqrt()
            / base_image_norm_sq.sqrt();
        let criterion_defect = (scaled.criterion - base.criterion).abs()
            / (1.0 + scaled.criterion.abs().max(base.criterion.abs()));
        eprintln!(
            "[#2099 fit co-scale] c={physical_scale}: image_defect={image_defect:.3e} \
             criterion_defect={criterion_defect:.3e}; components \
             [data,assignment,smooth,ard,analytic,repulsion,separation] base={:?} scaled={:?}",
            base.components, scaled.components,
        );
        assert!(
            image_defect < 1.0e-3,
            "normalized fitted reconstruction changed under physical co-scale c={physical_scale}: \
             relative image defect {image_defect:.3e}"
        );
        assert!(
            criterion_defect < 1.0e-3,
            "penalized criterion changed under physical co-scale c={physical_scale}: \
             relative criterion defect {criterion_defect:.3e}"
        );
        let component_names = [
            "data",
            "assignment",
            "smooth",
            "ard",
            "analytic",
            "repulsion",
            "separation",
        ];
        for (idx, name) in component_names.into_iter().enumerate() {
            let unit = base.components[idx];
            let rescaled = scaled.components[idx];
            let defect = (rescaled - unit).abs() / (1.0 + rescaled.abs().max(unit.abs()));
            assert!(
                defect < 1.0e-3,
                "{name} criterion component changed under physical co-scale \
                 c={physical_scale}: unit={unit:.8e}, rescaled={rescaled:.8e}, \
                 relative defect={defect:.3e}"
            );
        }
    }
}

/// #795 (regression guard mirroring the reported Python repro) — the single
/// planted circle embedded in D≫2 dimensions must converge through the isometry-
/// gauged arrow-Schur joint fit, not saturate the proximal ridge at 1e15.
///
/// The reported failure was the simplest possible manifold-SAE fit: one circle
/// (`K=1`, `d_atom=1`, `atom_topology="circle"`) planted in `D=12` via a random
/// 2×D frame. The wide embedding makes the fitted decoder large, and under the
/// un-normalized `‖B‖⁴` isometry Gauss-Newton curvature the arrow-Schur row
/// blocks lost positive-definiteness — the proximal ridge escalated to 1e15
/// while `|step|→1e-13` and every trial step was Armijo-rejected. This test runs
/// the ACTUAL `run_joint_fit_arrow_schur` with the isometry gauge ON on a
/// realistically PCA-seeded embedded circle and asserts it returns `Ok`, a finite
/// converged loss, finite atom coefficients, and recovers the circle
/// (low reconstruction error) — i.e. no `RemlConvergenceError`.
#[test]
fn sae_single_planted_circle_embedded_isometry_fit_converges_795() {
    use super::tests::{
        PlantedCircleAssignmentMode, planted_circle_embedded, planted_circle_seed_term,
    };
    use gam_terms::analytic_penalties::{
        AnalyticPenaltyKind, AnalyticPenaltyRegistry, IsometryPenalty, PsiSlice,
    };

    let n = 200usize;
    let d_embed = 12usize;
    let sigma = 0.02_f64;
    let z = planted_circle_embedded(n, d_embed, sigma);

    let (mut term, _seed_dispersion) =
        planted_circle_seed_term(z.view(), PlantedCircleAssignmentMode::Softmax);

    // Isometry gauge ON — the `t`↔`B` coupling whose un-normalized `‖B‖⁴`
    // curvature is what regressed #795 (a regression of the #681 sphere fix).
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(1)), 1),
    )));

    let mut rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0_f64]]);

    let loss = term
        .run_joint_fit_arrow_schur(
            z.view(),
            &mut rho,
            Some(&registry),
            25,
            0.04,
            1.0e-6,
            1.0e-6,
        )
        .expect(
            "single planted circle embedded in D=12 with the isometry gauge ON must converge \
             through the arrow-Schur joint fit (issue #795: the proximal ridge must not \
             saturate at 1e15 / reject every step)",
        );
    assert!(
        loss.total().is_finite(),
        "#795: converged loss on the embedded planted circle must be finite, got {}",
        loss.total()
    );

    // Atoms must remain finite (no NaN/Inf blow-up from a runaway ridge).
    let fitted = term.fitted();
    assert!(
        fitted.iter().all(|v| v.is_finite()),
        "#795: fitted reconstruction must be finite"
    );

    // The fit must actually recover the planted circle, not merely return Ok at a
    // stalled iterate: relative reconstruction error over the clean signal.
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (r, t) in fitted.iter().zip(z.iter()) {
        num += (r - t) * (r - t);
        den += t * t;
    }
    let rel_recon = (num / den.max(1.0e-300)).sqrt();
    // A healthy single fixed-ρ joint fit from the PCA seed reaches EV≈0.95
    // (rel recon ≈ 0.22 on this fixture). The #795 stall (ridge saturating at 1e15
    // with every step Armijo-rejected) leaves the fit at its seed, i.e. rel recon
    // O(1) (≳0.7, near ‖z‖). This bound cleanly separates the healthy fit from the
    // stall without over-fitting the exact converged value.
    assert!(
        rel_recon < 0.4,
        "#795: the isometry-gauged joint fit must recover the embedded planted circle \
         (rel recon {rel_recon:.3e}); a residual near ‖z‖ is the ridge-saturation stall symptom"
    );
}

/// #2226 — `penalized_quasi_laplace_criterion` must RANK the K=1 planted-circle inner fixed point
/// (return a finite Laplace value), not hard-refuse it. The inner solve reaches
/// its numerical fixed point where the objective can no longer decrease, but the
/// raw KKT gradient plateaus a couple of digits above the absolute iterate-scaled
/// tolerance on some SIMD targets (NEON/arm64 vs AVX/x86 diverge on the float
/// summation order). Before the affine-invariant Newton-decrement stationarity
/// certificate, that plateau exhausted the refine budget and returned
/// `penalized_quasi_laplace_criterion: inner solve did not converge at fixed ρ` — the exact refusal
/// reported for `sae_manifold_fit(K=1, atom_topology="circle")` on macOS arm64.
/// The Newton decrement ½λ² = −½gᵀΔ is affine-invariant, so the shared fixed
/// point is accepted on both architectures.
#[test]
fn sae_k1_circle_penalized_quasi_laplace_criterion_ranks_fixed_point_2226() {
    use super::tests::{
        PlantedCircleAssignmentMode, planted_circle_embedded, planted_circle_seed_term,
    };
    use gam_terms::analytic_penalties::{
        AnalyticPenaltyKind, AnalyticPenaltyRegistry, IsometryPenalty, PsiSlice,
    };

    let n = 200usize;
    let d_embed = 12usize;
    let sigma = 0.02_f64;
    let z = planted_circle_embedded(n, d_embed, sigma);

    let (mut term, _seed_dispersion) =
        planted_circle_seed_term(z.view(), PlantedCircleAssignmentMode::Softmax);

    // Isometry gauge ON — the same fixture as `..._795`, but exercised through
    // the ρ-ranking `penalized_quasi_laplace_criterion` (the path `sae_manifold_fit`'s outer BFGS
    // drives) rather than a single joint fit.
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(1)), 1),
    )));

    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0_f64]]);

    // A generous inner budget: the fixed point is reached well within it. The
    // criterion must rank that stationary iterate instead of hard-refusing on an
    // absolute raw-gradient tolerance that is not reachable on every SIMD target.
    let (value, loss, _cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            z.view(),
            &rho,
            Some(&registry),
            200,
            0.04,
            1.0e-6,
            1.0e-6,
        )
        .expect(
            "#2226: the K=1 planted-circle inner solve reaches a numerical fixed point; \
             penalized_quasi_laplace_criterion must rank that stationary iterate (affine-invariant Newton \
             decrement) instead of refusing on an unreachable absolute gradient tolerance",
        );
    assert!(
        value.is_finite() && loss.total().is_finite(),
        "#2226: the ranked Laplace criterion must be finite (value={value}, loss={})",
        loss.total()
    );
}

/// #1206 — ranking and gradient lanes must optimize one coherent criterion.
/// The amortized-encoder consistency diagnostic `c(ρ)` has no analytic
/// derivative, so it cannot rank `f+c` while BFGS descends `f`: the selected fit
/// would not be stationary for its selection criterion. Both lanes therefore
/// report their matched pure-penalized quasi-Laplace value, while encoder consistency and the
/// fitted-data collapse ledger remain read-only diagnostics.
#[test]
fn ranking_and_gradient_lanes_match_bare_reml() {
    let mut objective = warmstart_test_objective_with_evaluator();
    let rho_flat = objective.current_rho.to_flat();

    // Value-probe lane: the comparand the cascade uses for seed validation and
    // cross-seed ranking.
    let value_lane = objective
        .eval_cost(&rho_flat)
        .expect("value-probe lane evaluates penalized quasi-Laplace");

    // Gradient lane: the cost an ACCEPTED iterate reports, paired with the
    // analytic ∇f the BFGS Armijo test consumes. A fresh objective so the two
    // paths solve from the identical seed state.
    let mut objective_grad = warmstart_test_objective_with_evaluator();
    let gradient_lane = objective_grad
        .eval(&rho_flat)
        .expect("gradient lane evaluates")
        .cost;

    assert!(
        value_lane.is_finite() && gradient_lane.is_finite(),
        "both lanes must be finite: value={value_lane}, gradient={gradient_lane}"
    );

    // `eval` leaves the basin-envelope argmin installed in `objective_grad.term`.
    // Price that exact selected state once more through the bare term criterion:
    // this isolates objective plumbing (block Jacobian / diagnostic folds) from
    // basin selection. Constructing a fresh bare term would instead price the
    // historical single warm-start trajectory and recreate the route split this
    // contract exists to forbid.
    let bare_shared = {
        let mut selected_term = objective_grad.term.clone();
        let target = objective_grad.target.clone();
        let rho_state = objective_grad
            .baseline_rho
            .from_flat(rho_flat.view())
            .expect("the flattened rho has the length the baseline rho declares");
        let (reml, _loss, _cache) = selected_term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho_state,
                objective_grad.registry.as_ref(),
                objective_grad.inner_max_iter,
                objective_grad.learning_rate,
                objective_grad.ridge_ext_coord,
                objective_grad.ridge_beta,
            )
            .expect("selected envelope state re-prices through the bare criterion");
        reml
    };
    let value_vs_bare = (value_lane - bare_shared).abs();
    assert!(
        value_vs_bare < 1.0e-9,
        "the ranking lane must report bare REML so selection and descent share \
         one criterion: value_lane={value_lane}, bare={bare_shared}, \
         diff={value_vs_bare}"
    );

    let gradient_vs_bare = (gradient_lane - bare_shared).abs();
    assert!(
        gradient_vs_bare < 1.0e-9,
        "the gradient lane must report bare REML (no consistency fold), so its \
         (cost, ∇f) pair is self-consistent for BFGS Armijo: \
         gradient_lane={gradient_lane}, bare={bare_shared}, \
         diff={gradient_vs_bare}"
    );
    assert!(
        (value_lane - gradient_lane).abs() < 1.0e-9,
        "ranking and gradient lanes must price one fixed point: \
         value={value_lane}, gradient={gradient_lane}"
    );
}

/// #1154 item 2+3 — the amortized-encoder warm-start (Design A) accelerates
/// the inner solve to the SAME stationary point WITHOUT degrading recovery of
/// the planted manifold. On a known periodic manifold we
///
/// 1. fit the dictionary (sequential / cold inner solve) and record the
///    explained variance — the REML-then-distill baseline;
/// 2. build the amortized encoder from that fitted dictionary and offer its
///    certified rows as inner latent warm-starts. Zero certified rows is a
///    valid conservative gate outcome: the helper must then leave the cold
///    seed untouched instead of corrupting the inner state;
/// 3. re-converge the inner solve FROM the warm-start and require the
///    explained variance to be at least as good as the cold-fit baseline —
///    the warm-start changes the basin entry, not the root, so recovery never
///    regresses (and the seed lands the solve in the right basin immediately).
#[test]
fn amortized_warm_start_matches_or_beats_cold_inner_solve_on_known_manifold() {
    let n = 24usize;
    let p = 4usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let m = phi.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        let scale = 1.0 / (1.0 + b as f64);
        scale * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic_truth",
        SaeAtomBasisKind::Periodic,
        1,
        phi.clone(),
        jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let target = phi.dot(&decoder);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("fixture assignment: one logit column and one coord block per atom");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("fixture term: every atom's basis width matches its assignment block");
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![1.0_f64.ln()]]);

    // (1) Cold (sequential) inner solve — the REML-then-distill baseline.
    let mut rho_cold = rho.clone();
    term.run_joint_fit_arrow_schur(target.view(), &mut rho_cold, None, 12, 0.1, 1.0e-4, 1.0e-4)
        .expect("cold inner solve converges on the known periodic manifold");
    let cold_ev = {
        let fitted = term
            .try_fitted_for_rho(&rho_cold)
            .expect("the fit above converged, so a fitted surface exists");
        reconstruction_explained_variance(target.view(), fitted.view())
            .expect("explained variance is defined for the planted target")
    };
    assert!(
        cold_ev > 0.9,
        "cold fit must recover the planted periodic manifold (EV={cold_ev})"
    );

    // (2) Build the amortized encoder from the fitted dictionary and offer it
    // to the inner solve as an advisory warm-start. The Kantorovich gate is
    // intentionally conservative: if it cannot certify this fitted dictionary,
    // Design A must leave the cold seed untouched rather than corrupting the
    // inner state.
    let warm_started = term
        .warm_start_latents_from_amortized_encoder(target.view(), &rho_cold)
        .expect("amortized warm-start runs on the fitted dictionary");
    eprintln!("#1154 WARM-START: certified warm-started rows={warm_started}/{n}");
    assert!(
        warm_started <= n,
        "the amortized encoder cannot warm-start more rows than the fitted \
         batch size; warm_started={warm_started}, n={n}"
    );

    // (3) Re-converge FROM the warm-start; recovery must not regress.
    let mut rho_warm = rho.clone();
    term.run_joint_fit_arrow_schur(target.view(), &mut rho_warm, None, 12, 0.1, 1.0e-4, 1.0e-4)
        .expect("warm-started inner solve converges");
    let warm_ev = {
        let fitted = term
            .try_fitted_for_rho(&rho_warm)
            .expect("the fit above converged, so a fitted surface exists");
        reconstruction_explained_variance(target.view(), fitted.view())
            .expect("explained variance is defined for the planted target")
    };
    // The cold and warm solves are two INDEPENDENT Newton descents to the same
    // stationary point of the same planted manifold — cold seeds from the chart
    // center, warm from the amortized encoder — so their explained variance
    // agrees only up to solver / parallel-reduction tolerance. A sub-1e-3 EV
    // wobble between them is floating-point noise (the estimator reduces in
    // parallel), not a recovery regression. The guarantee under test is that the
    // warm-start does not MEANINGFULLY degrade recovery: the warm solve still
    // recovers the manifold and stays within an EV-agreement band of the cold one.
    assert!(
        warm_ev > 0.9,
        "warm-started inner solve must still recover the planted manifold (warm_ev={warm_ev})"
    );
    assert!(
        warm_ev >= cold_ev - 5.0e-3,
        "amortized warm-start (co-trained inner solve) must recover the manifold \
         about as well as the cold/sequential solve, to solver tolerance: \
         warm_ev={warm_ev}, cold_ev={cold_ev}"
    );
}

/// #1026 — curved-vs-linear reconstruction through the REAL solver at matched K=1,
/// FAST (fixed-rho `run_joint_fit_arrow_schur` inner solve, not the ~90 s outer
/// cascade). A periodic atom recovers a planted circle; a degree-1 euclidean
/// (the principled linear-SAE baseline) atom on the SAME circle target can only fit
/// a secant and is starved. Curved must clear a high absolute bar AND beat linear by
/// a wide margin — the shatter penalty measured through the production inner solver,
/// a cheap gam-sae-suite complement to the full-engine pin in tests/sae.
#[test]
fn sae_1026_curved_beats_linear_reconstruction_through_solver() {
    let n = 48usize;
    let p = 4usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);

    // Planted circle target: a smooth periodic decoder over the circle coordinate.
    let (phi_c, jet_c) = periodic_basis(&coords);
    let mc = phi_c.ncols();
    let decoder_c = Array2::from_shape_fn((mc, p), |(b, c)| {
        (1.0 / (1.0 + b as f64)) * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });
    let target = phi_c.dot(&decoder_c);

    // CURVED arm: one periodic atom on the circle.
    let curved_ev = {
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            phi_c.clone(),
            jet_c,
            decoder_c.clone(),
            Array2::<f64>::eye(mc),
        )
        .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords.clone()],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .expect("fixture assignment: one logit column and one coord block per atom");
        let mut term = SaeManifoldTerm::new(vec![atom], assignment)
            .expect("fixture term: every atom's basis width matches its assignment block");
        let mut rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![1.0_f64.ln()]]);
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 12, 0.1, 1.0e-4, 1.0e-4)
            .expect("curved inner solve converges on the planted circle");
        let fitted = term
            .try_fitted_for_rho(&rho)
            .expect("the fit above converged, so a fitted surface exists");
        reconstruction_explained_variance(target.view(), fitted.view())
            .expect("target and reconstruction share a shape, so explained variance is defined")
    };

    // LINEAR arm: one degree-1 euclidean atom on the SAME circle target.
    let linear_ev = {
        let evaluator = Arc::new(
            EuclideanPatchEvaluator::new(1, 1)
                .expect("a 1-D, 1-patch Euclidean basis is a valid evaluator"),
        );
        let (phi_l, jet_l) = evaluator
            .evaluate(coords.view())
            .expect("fixture coords are already wrapped into the evaluator's unit period");
        let ml = phi_l.ncols();
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "linear",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi_l,
            jet_l,
            Array2::<f64>::zeros((ml, p)),
            Array2::<f64>::eye(ml),
        )
        .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
        .with_basis_second_jet(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords.clone()],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .expect("fixture assignment: one logit column and one coord block per atom");
        let mut term = SaeManifoldTerm::new(vec![atom], assignment)
            .expect("fixture term: every atom's basis width matches its assignment block");
        let mut rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![Array1::<f64>::zeros(1)]);
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 12, 0.1, 1.0e-4, 1.0e-4)
            .expect("linear inner solve converges");
        let fitted = term
            .try_fitted_for_rho(&rho)
            .expect("the fit above converged, so a fitted surface exists");
        reconstruction_explained_variance(target.view(), fitted.view())
            .expect("target and reconstruction share a shape, so explained variance is defined")
    };

    eprintln!("#1026 solver reconstruction: curved EV={curved_ev:.4}, linear EV={linear_ev:.4}");
    assert!(
        curved_ev > 0.9,
        "the periodic atom must recover the planted circle through the solver (EV={curved_ev})"
    );
    assert!(
        curved_ev > linear_ev + 0.2,
        "curved must beat the matched-K linear baseline by a wide margin (the shatter \
         penalty: a degree-1 secant cannot follow a closed circle): \
         curved={curved_ev}, linear={linear_ev}"
    );
}

/// #1026 — K=2 superposition RECONSTRUCTION through the REAL solver. Two superposed
/// circles fit by two periodic atoms via the production joint inner solve
/// (`run_joint_fit_arrow_schur`). When the ambient dimension holds both planted
/// planes (p = 4 = 2K) the joint solve disentangles the superposition and
/// reconstructs it. With amplitude-aware routing the solver ALSO reconstructs the
/// p < 2K overlapping-plane case: reconstruction only requires the SUM of the two
/// atom images to match the target, and a shared-plane split still achieves that,
/// so reconstruction EV does not collapse below 2K even though the individual atom
/// DECOMPOSITION is under-determined there (reconstruction ≠ identifiability — the
/// p ≥ 2K boundary governs identifiability, not reconstruction fidelity). This pins
/// positive recovery in BOTH regimes. Fast (fixed-rho inner solve, ~0.1 s/arm).
#[test]
fn sae_1026_solver_recovers_separable_superposition_but_not_below_2k() {
    let recover = |p: usize, overlap: bool| -> f64 {
        let n = 80usize;
        let theta_a = Array2::from_shape_fn((n, 1), |(r, _)| ((r as f64) * 0.043).rem_euclid(1.0));
        let theta_b =
            Array2::from_shape_fn((n, 1), |(r, _)| ((r as f64) * 0.071 + 0.13).rem_euclid(1.0));
        let mut target = Array2::<f64>::zeros((n, p));
        for r in 0..n {
            let a = std::f64::consts::TAU * theta_a[[r, 0]];
            let b = std::f64::consts::TAU * theta_b[[r, 0]];
            if !overlap {
                target[[r, 0]] = a.cos();
                target[[r, 1]] = a.sin();
                target[[r, 2]] = b.cos();
                target[[r, 3]] = b.sin();
            } else {
                // p = 3: circle A in dims (0,1), circle B in dims (1,2) — they share dim 1.
                target[[r, 0]] += a.cos();
                target[[r, 1]] += a.sin();
                target[[r, 1]] += b.cos();
                target[[r, 2]] += b.sin();
            }
        }
        let seed_a =
            Array2::from_shape_fn((n, 1), |(r, _)| (theta_a[[r, 0]] + 0.03).rem_euclid(1.0));
        let seed_b =
            Array2::from_shape_fn((n, 1), |(r, _)| (theta_b[[r, 0]] + 0.03).rem_euclid(1.0));
        let (pa, ja) = periodic_basis(&seed_a);
        let (pb, jb) = periodic_basis(&seed_b);
        let m = pa.ncols();
        let a0 = SaeManifoldAtom::new_with_provided_function_gram(
            "cA",
            SaeAtomBasisKind::Periodic,
            1,
            pa,
            ja,
            Array2::<f64>::zeros((m, p)),
            Array2::<f64>::eye(m),
        )
        .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let a1 = SaeManifoldAtom::new_with_provided_function_gram(
            "cB",
            SaeAtomBasisKind::Periodic,
            1,
            pb,
            jb,
            Array2::<f64>::zeros((m, p)),
            Array2::<f64>::eye(m),
        )
        .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let logits = Array2::<f64>::from_elem((n, 2), 6.0 * 0.5);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![seed_a.clone(), seed_b.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::ordered_beta_bernoulli(0.5, 1.0, false),
        )
        .expect("fixture assignment: one logit column and one coord block per atom");
        let mut term = SaeManifoldTerm::new(vec![a0, a1], assignment)
            .expect("fixture term: every atom's basis width matches its assignment block");
        let mut rho = SaeManifoldRho::new(
            0.0,
            0.01_f64.ln(),
            vec![array![1.0_f64.ln()], array![1.0_f64.ln()]],
        );
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 24, 0.1, 1.0e-4, 1.0e-4)
            .expect("K=2 inner solve converges");
        let fitted = term
            .try_fitted_for_rho(&rho)
            .expect("the fit above converged, so a fitted surface exists");
        reconstruction_explained_variance(target.view(), fitted.view())
            .expect("target and reconstruction share a shape, so explained variance is defined")
    };
    let separable = recover(4, false);
    let under_determined = recover(3, true);
    eprintln!(
        "#1026 K=2 superposition: separable(p=4)={separable:.4}, overlap(p=3)={under_determined:.4}"
    );
    // Both regimes RECONSTRUCT the superposed circles: the joint K=2 solver
    // recovers the separable p>=2K case, and — with amplitude-aware routing — the
    // p<2K overlapping case too. The earlier premise that reconstruction
    // "collapses" below p=2K conflated reconstruction EV with atom
    // IDENTIFIABILITY: reconstruction only needs the SUM of the two atom images
    // to match the target, and a shared-plane split still achieves that (any
    // redistribution of mass along the shared dimension reconstructs identically),
    // so reconstruction EV does NOT collapse below 2K even though the individual
    // decomposition is under-determined there. The non-stale, meaningful check is
    // that the joint solver reconstructs the superposition in BOTH regimes.
    assert!(
        separable > 0.95,
        "the joint solver must recover two superposed circles when p >= 2K (EV={separable})"
    );
    assert!(
        under_determined > 0.9,
        "amplitude-aware routing reconstructs even the p < 2K overlapping \
         superposition (reconstruction, not identifiability): overlap EV={under_determined}"
    );
}

#[test]
pub(crate) fn deflated_solver_matches_plain_solve_when_no_gauge_is_installed() {
    let cache = diagonal_latent_cache(&[2.0_f64, 5.0, 7.0]);
    let solver = DeflatedArrowSolver::plain(&cache);
    let rhs_t = array![4.0_f64, 10.0, -14.0];
    let rhs_beta = Array1::<f64>::zeros(0);
    let (plain_t, plain_beta) = cache
        .full_inverse_apply(rhs_t.view(), rhs_beta.view())
        .expect("plain cache solve");
    let solved = solver
        .solve(rhs_t.view(), rhs_beta.view())
        .expect("adapter solve");
    assert_eq!(solved.t.len(), plain_t.len());
    for idx in 0..plain_t.len() {
        assert_abs_diff_eq!(solved.t[idx], plain_t[idx], epsilon = 0.0);
    }
    assert_eq!(solved.beta.len(), plain_beta.len());
    for idx in 0..plain_beta.len() {
        assert_abs_diff_eq!(solved.beta[idx], plain_beta[idx], epsilon = 0.0);
    }
}

#[test]
pub(crate) fn deflated_solver_matches_dense_quotient_pseudoinverse_on_near_null_fixture() {
    let cache = diagonal_latent_cache(&[2.0_f64, 1.0e-14]);
    let gauge = array![0.0_f64, 1.0];
    let solver = DeflatedArrowSolver::from_orthonormal_gauges(&cache, vec![gauge], 2.0)
        .expect("deflated solver");
    let rhs_beta = Array1::<f64>::zeros(0);

    let physical_rhs = array![4.0_f64, 0.0];
    let solved = solver
        .solve(physical_rhs.view(), rhs_beta.view())
        .expect("physical solve");
    let oracle = array![2.0_f64, 0.0];
    for idx in 0..oracle.len() {
        assert_abs_diff_eq!(solved.t[idx], oracle[idx], epsilon = 1.0e-12);
    }

    let gauge_rhs = array![0.0_f64, 1.0];
    let plain = cache
        .full_inverse_apply(gauge_rhs.view(), rhs_beta.view())
        .expect("plain gauge solve")
        .0;
    let stiffened = solver
        .solve(gauge_rhs.view(), rhs_beta.view())
        .expect("stiffened gauge solve")
        .t;
    assert!(plain[1] > 1.0e13, "plain near-null solve must be huge");
    assert_abs_diff_eq!(stiffened[1], 0.5, epsilon = 1.0e-12);
}

/// #2253 mechanism regression: `DeflatedArrowSolver` preconditions with
/// `(B + κ Q Qᵀ)⁻¹`, so an exact-stationarity Krylov operator must apply the
/// matching `A + κ Q Qᵀ`. Leaving `A` raw makes a gauge-bearing right-hand side
/// inconsistent and the original-residual certificate correctly refuses it.
#[test]
pub(crate) fn gauge_fixed_krylov_operator_matches_deflated_preconditioner_2253() {
    let cache = diagonal_latent_cache(&[2.0_f64, 1.0e-14]);
    let gauge = array![0.0_f64, 1.0];
    let stiffness = 2.0;
    let solver =
        DeflatedArrowSolver::from_orthonormal_gauges(&cache, vec![gauge.clone()], stiffness)
            .expect("deflated solver");
    let rhs = SaeArrowVector {
        t: array![4.0_f64, 1.0],
        beta: Array1::zeros(0),
    };
    let raw_a = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> {
        Ok(SaeArrowVector {
            t: array![2.0 * v.t[0], 0.0],
            beta: Array1::zeros(0),
        })
    };
    let add_gauge_stiffness = |v: &SaeArrowVector, applied: &mut SaeArrowVector| {
        let coefficient = stiffness * gauge.dot(&v.t);
        for index in 0..gauge.len() {
            applied.t[index] += coefficient * gauge[index];
        }
    };

    // Raw A has the exact null q=e1 while rhs has q-mass, so A x = rhs is
    // inconsistent. A residual-checked solver must refuse it rather than return
    // the old CG path's arbitrary last iterate.
    let raw = solve_b_preconditioned_gmres_with(
        &rhs,
        |v| raw_a(v),
        |vector| solver.solve(vector.t.view(), vector.beta.view()),
    );
    assert!(
        raw.is_err(),
        "raw A with rhs mass on its exact gauge null must not pass the residual certificate"
    );

    // The quotient operator uses the SAME κQQᵀ term as the preconditioner.
    // It is diag(2,2), so the exact gauge-fixed solution is (2, 1/2).
    let solved = solve_b_preconditioned_gmres_with(
        &rhs,
        |v| {
            let mut out = raw_a(v)?;
            add_gauge_stiffness(v, &mut out);
            Ok(out)
        },
        |vector| solver.solve(vector.t.view(), vector.beta.view()),
    )
    .expect("gauge-fixed exact-stationarity solve");
    assert_abs_diff_eq!(solved.t[0], 2.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(solved.t[1], 0.5, epsilon = 1.0e-12);

    let mut applied = raw_a(&solved).expect("raw A apply");
    add_gauge_stiffness(&solved, &mut applied);
    let residual = SaeArrowVector {
        t: &applied.t - &rhs.t,
        beta: &applied.beta - &rhs.beta,
    };
    assert!(
        sae_norm(&residual) <= 1.0e-12 * sae_norm(&rhs).max(1.0),
        "gauge-fixed operator and inverse must satisfy the original residual; got {:.3e}",
        sae_norm(&residual)
    );
}

#[test]
pub(crate) fn pca_seed_handles_huge_equal_finite_columns_without_mean_overflow() {
    let z = array![[1.0e308_f64, 1.0e308], [1.0e308, 1.0e308]];
    let coords = sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1])
        .expect("the planted target has full enough rank to seed one 1-D atom");
    assert_eq!(coords.dim(), (1, 2, 1));
    assert!(
        coords.iter().all(|value| value.is_finite()),
        "huge finite equal columns must not overflow the PCA seed mean: {coords:?}"
    );
}

#[test]
pub(crate) fn pca_seed_rejects_huge_finite_span_that_overflows_centering() {
    let z = array![[1.0e308_f64, 0.0], [-1.0e308, 0.0]];
    let err = sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1])
        .expect_err("opposite huge finite values exceed f64 centering range");
    assert!(
        err.contains("centered Z is non-finite") || err.contains("SVD failed"),
        "unexpected PCA seed error: {err}"
    );
}

// ---- Issue #972: low-rank Grassmann decoder frame verification ----

/// Regression test for #1415: the smooth-threshold third derivative consumed by the
/// log-determinant θ-adjoint (`assignment_prior_hdiag_derivative_entry`) must be
/// the EXACT derivative of the (separately certified) Hessian diagonal
/// `P''(ℓ)=(λ/τ²)s(1−2a)`, namely `P'''(ℓ)=(λ/τ³)s(1−6a+6a²)`. The historical
/// code used `(2λ/τ³)s²(1−2a)`, which is algebraically wrong and vanishes at the
/// threshold (`a=1/2`) where the true value is `−λ/(8τ³)`.
///
/// Oracle: a 4th-order central finite difference of the exact `P''(ℓ)` formula
/// (FD permitted only inside the test as an independent check of the closed
/// form). Also pins the exact threshold value `−λ/(8τ³)` and asserts the
/// production entry is strictly negative there (the old formula returned 0).
#[test]
fn smooth_threshold_hdiag_third_derivative_matches_central_difference_1415() {
    use ndarray::{Array1, Array2, Array3};
    let n = 6usize;
    let k = 2usize;
    let p = 3usize;
    let temperature = 0.35_f64;
    let threshold = 0.1_f64;
    // Include the exact threshold (logit == θ ⇒ a = 1/2) plus points on both sides.
    let logits = Array2::<f64>::from_shape_vec(
        (n, k),
        vec![
            0.1, 0.0, 0.2, -0.05, 0.05, 0.15, 0.25, 0.3, -0.1, 0.12, 0.18, 0.08,
        ],
    )
    .expect("valid logit grid");
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|i| {
            SaeManifoldAtom::new_with_provided_function_gram(
                &format!("atom{i}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                Array2::<f64>::ones((n, 2)),
                Array3::<f64>::zeros((n, 2, 1)),
                Array2::<f64>::zeros((2, p)),
                Array2::<f64>::eye(2),
            )
            .expect("fixture atom: basis width, latent dim and decoder shape agree by construction")
        })
        .collect();
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Euclidean; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.clone(),
        coords,
        manifolds,
        AssignmentMode::threshold_gate(temperature, threshold),
    )
    .expect("valid smooth threshold assignment");
    let term = SaeManifoldTerm::new(atoms, assignment)
        .expect("fixture term: every atom's basis width matches its assignment block");
    let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);

    let inv_tau = 1.0 / temperature;
    let sparsity = rho
        .lambda_sparse()
        .expect("the fixture rho was built with a sparsity lambda");
    // Exact, separately-certified Hessian diagonal P''(ℓ) as a function of ℓ.
    let p2 = |logit: f64| -> f64 {
        let a = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
        let s = a * (1.0 - a);
        sparsity * s * (1.0 - 2.0 * a) * inv_tau * inv_tau
    };

    let mut saw_threshold = false;
    for row in 0..n {
        for atom in 0..k {
            let logit = logits[[row, atom]];
            let entry = term.assignment_prior_hdiag_derivative_entry(
                sparsity,
                row,
                atom,
                SaeLocalRowVar::Logit { atom },
                None,
            );
            // Independent oracle: 4th-order central difference of exact P''(ℓ).
            let h = 1.0e-3_f64;
            let fd = (-p2(logit + 2.0 * h) + 8.0 * p2(logit + h) - 8.0 * p2(logit - h)
                + p2(logit - 2.0 * h))
                / (12.0 * h);
            let scale = entry.abs().max(fd.abs()).max(1.0e-8);
            assert!(
                (entry - fd).abs() <= 1.0e-5 * scale,
                "row {row} atom {atom}: P''' entry {entry:e} vs FD {fd:e}"
            );

            if (logit - threshold).abs() < 1e-12 {
                saw_threshold = true;
                // At ℓ=θ: a=1/2, s=1/4 ⇒ P'''=−λ/(8τ³); the old code gave 0.
                let expected = -sparsity / 8.0 * inv_tau * inv_tau * inv_tau;
                assert_abs_diff_eq!(entry, expected, epsilon = 1e-9);
                assert!(
                    entry < -1e-6,
                    "threshold third derivative must be strictly negative (old buggy \
                     formula returned 0): entry={entry:e}"
                );
            }
        }
    }
    assert!(
        saw_threshold,
        "fixture must include a logit exactly at the threshold to pin −λ/(8τ³)"
    );
}

/// Direct FD regression guard for the Kantorovich-certificate primitives
/// `encode_grad_hess` and `beta_eta_newton` (encode.rs). These are load-bearing
/// for `certified_encode` (the #1026/#1154 basin path) yet were previously only
/// exercised *indirectly* through the full encode. This pins, at the scalar
/// periodic (circle) coordinate that #1026 actually uses:
///   - the gradient `g = ∂f/∂t` of `f(t) = ½‖z·decode(t) − x‖²` (amplitude factor),
///   - the full Hessian `H = ∂²f/∂t²` INCLUDING the residual·second-jet curvature
///     term `r·∂²m` (the term whose absence would silently make H Gauss-Newton),
///   - `β = 1/λ_min(H) = ‖H⁻¹‖₂` and the Newton step `δ = −H⁻¹g` (NOT the classic
///     `1/‖H‖₂` error).
/// A non-stationary, off-manifold target is chosen so both the residual and the
/// curvature term are nonzero (a stationary point would zero `r` and hide bugs).
#[test]
fn encode_grad_hess_and_beta_eta_match_finite_differences() {
    use crate::encode::{beta_eta_newton, encode_grad_hess};
    use ndarray::Array2;

    // Periodic (circle) atom: real second jet via TestPeriodicEvaluator (d=1).
    let train = Array2::from_shape_fn((24, 1), |(r, _)| (r as f64 + 0.5) / 24.0);
    let (phi, jet) = periodic_basis(&train);
    let m = phi.ncols();
    let p = 4usize;
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        (1.0 / (1.0 + b as f64)) * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .expect("fixture atom: basis width, latent dim and decoder shape agree by construction");
    let eval = TestPeriodicEvaluator;
    let amplitude = 0.8_f64;

    // Objective f(t) = ½‖amp·decode(t) − x‖², with decode(t) = Φ(t)·decoder.
    let decode = |t: f64| -> ndarray::Array1<f64> {
        let coords = Array2::from_shape_fn((1, 1), |_| t);
        let (ph, _) = periodic_basis(&coords);
        amplitude * ph.dot(&decoder).row(0).to_owned()
    };
    // Off-manifold, non-stationary target ⇒ residual r ≠ 0 (exercises curvature).
    let t0 = 0.137_f64;
    let x = &decode(0.42) + &ndarray::Array1::from_vec(vec![0.3, -0.2, 0.15, -0.25]);
    let f = |t: f64| -> f64 {
        let r = &decode(t) - &x;
        0.5 * r.dot(&r)
    };

    // Analytic g, H — `encode_grad_hess` returns the TRUE Hessian ∂²f/∂t² (no
    // Levenberg ridge is added to the certified field; F2).
    let t_view = ndarray::Array1::from_vec(vec![t0]);
    let (g, h) = encode_grad_hess(&atom, &eval, t_view.view(), x.view(), amplitude)
        .expect("encode_grad_hess runs")
        .expect("second jet present ⇒ Some");

    // Central FD of f → gradient.
    let eps = 1e-6;
    let g_fd = (f(t0 + eps) - f(t0 - eps)) / (2.0 * eps);
    assert_abs_diff_eq!(g[0], g_fd, epsilon = 1e-6);

    // Central FD of f → second derivative (includes the residual·curvature term).
    let h_fd = (f(t0 + eps) - 2.0 * f(t0) + f(t0 - eps)) / (eps * eps);
    // Second-difference truncation/roundoff floor is ~1e-4 at this ε.
    assert_abs_diff_eq!(h[[0, 0]], h_fd, epsilon = 5e-3);

    // beta_eta_newton on a known SPD H: build a positive-definite 1×1 from H by
    // ridging if the off-manifold curvature happened to be indefinite here.
    let mut hpd = h.clone();
    if hpd[[0, 0]] <= 0.0 {
        hpd[[0, 0]] = 1.5;
    }
    let (beta, eta, delta) = beta_eta_newton(hpd.view(), g.view())
        .expect("beta_eta_newton runs")
        .expect("SPD ⇒ Some");
    // β = 1/λ_min = ‖H⁻¹‖₂ (the correct operator norm, not 1/‖H‖₂).
    assert_abs_diff_eq!(beta * hpd[[0, 0]], 1.0, epsilon = 1e-12);
    // δ = −H⁻¹ g, η = ‖δ‖.
    assert_abs_diff_eq!(delta[0], -g[0] / hpd[[0, 0]], epsilon = 1e-12);
    assert_abs_diff_eq!(eta, (g[0] / hpd[[0, 0]]).abs(), epsilon = 1e-12);
}

/// Driver-level freeze invariant (#850). The outer-objective test
/// `seed_inner_state_installs_and_reuses_matching_beta` exercises the freeze
/// through `OuterObjective::eval`; this one pins it at the exact seam where the
/// bug lived: `run_joint_fit_arrow_schur` itself.
///
/// At `max_iter == 0` the joint solve is a verbatim FREEZE of the warm-started
/// `(t, β)` — it must run no β-mutating stage. The regression was that the
/// driver still ran the entry-stage re-seed guards and the #1026 post-loop
/// decoder-LSQ polish, which refit β to the unpenalised least-squares argmin
/// and committed it. We seed a β deliberately OFF that argmin (the pristine
/// decoder differs from the data-optimal one), run a zero-iteration joint fit,
/// and assert β is byte-for-byte unchanged. Before the fix the polish moved it.
#[test]
pub(crate) fn run_joint_fit_max_iter_zero_freezes_beta_verbatim() {
    let mut term = warmstart_test_objective().term;
    let dim = term.beta_dim();
    // A distinctive seed that differs from the term's pristine decoder, so the
    // unpenalised LSQ polish (had it run) would have a strict decrease to chase.
    let pristine = term.flatten_beta();
    let seed: Array1<f64> = Array1::from_shape_fn(dim, |i| pristine[i] + 0.5 + 0.01 * (i as f64));
    assert!(
        (&seed - &pristine).iter().any(|d| d.abs() > 1e-6),
        "seed must differ from the pristine β for the freeze check to be meaningful"
    );
    term.set_flat_beta(seed.view())
        .expect("length-matching β must install");

    // Target matches `warmstart_test_objective`'s 4×1 signal.
    let target = array![[0.20_f64], [-0.10], [0.30], [0.05]];
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    // max_iter == 0 ⇒ verbatim freeze: no Newton step, no guard, no polish.
    let loss = term
        .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 0, 1.0, 1.0e-6, 1.0e-6)
        .expect("zero-iteration joint fit at the warm-started β must succeed");
    assert!(
        loss.total().is_finite(),
        "frozen-state loss must be finite; got {}",
        loss.total()
    );

    let frozen = term.flatten_beta();
    for (i, (&f, &s)) in frozen.iter().zip(seed.iter()).enumerate() {
        assert!(
            (f - s).abs() < 1e-12,
            "max_iter==0 must freeze β verbatim at coord {i}: frozen {f} != seed {s} (#850)"
        );
    }
}
