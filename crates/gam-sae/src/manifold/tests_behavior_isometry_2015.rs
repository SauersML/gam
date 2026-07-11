//! #2015 — the per-atom representation–behavior isometry defect, read off a REAL
//! two-block joint fit.
//!
//! Both cases share ONE activation image: a clean unit circle, constant speed in
//! the shared latent `t`. What differs is HOW the behavior traverses its own
//! circle as a function of the SAME `t`:
//!
//!  * **isometric** — behavior winds at *constant* angular speed `ψ(t) = 2π t`,
//!    so its induced speed `s_y` is (nearly) constant and the ratio `r = s_x/s_y`
//!    is constant: a scaled isometry, low reported defect.
//!  * **broken**    — behavior winds at *uneven* speed `ψ(t) = 2π t + 0.8·sin 2π t`
//!    (`ψ'` sweeps from `0.2·2π` to `1.8·2π`), so `s_y` varies strongly while
//!    `s_x` stays constant: the correspondence bends along the atom, high defect.
//!
//! The activation-only geometry is IDENTICAL in the two cases (same circle), so
//! no activation-side statistic can tell them apart — the defect is a genuinely
//! cross-block quantity that only the two-block fit exposes. The test asserts the
//! reported defect separates the two, and that the isometric defect is small.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    TwoBlockRemlControls, atom_behavior_isometry, reconstruction_explained_variance,
};

/// A probability law whose square-root half-density traces an exact circle.
///
/// `SphereTangentEmbedding` fits the normalized mean half-density as its
/// basepoint and projects orthogonally to it. The four fixed vectors below are
/// orthonormal, with `e1/e2/e3` tangent to the positive basepoint `b`. Hence
///
/// `q(t) = radial(t)b + radius(cos ψ e1 + sin ψ e2) + residual(t)e3`
///
/// has unit norm and strictly positive coordinates, and `p = q⊙q` embeds back
/// to the declared tangent circle. For the REML fixture only, `residual_amp`
/// adds an order-9 orthogonal component outside the fitted harmonic basis: it
/// keeps the behavior RSS positive (so λ_y is identifiable) while the fitted
/// decoder remains the clean constant-speed circle.
fn sphere_circle_probabilities(angle: f64, base_angle: f64, residual_amp: f64) -> [f64; 4] {
    let inv_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
    let b = [0.5_f64; 4];
    let e1 = [inv_sqrt_two, -inv_sqrt_two, 0.0, 0.0];
    let e2 = [0.5, 0.5, -0.5, -0.5];
    let e3 = [0.0, 0.0, inv_sqrt_two, -inv_sqrt_two];
    let radius = 0.25_f64;
    let residual = residual_amp * (9.0 * base_angle).sin();
    let radial = (1.0 - radius * radius - residual * residual).sqrt();
    let (cos_angle, sin_angle) = (angle.cos(), angle.sin());
    std::array::from_fn(|token| {
        let q = radial * b[token]
            + radius * (cos_angle * e1[token] + sin_angle * e2[token])
            + residual * e3[token];
        q * q
    })
}

/// K=1 periodic (circle) atom at the augmented output width, cold decoders.
fn augmented_circle_atom(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> (SaeManifoldAtom, Array2<f64>) {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new(
        "b0",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    (atom, coords.clone())
}

/// K=1 softmax term (single always-on atom) at augmented width.
fn build_k1(atom: SaeManifoldAtom, coord_block: Array2<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = coord_block.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coord_block],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

fn block_ev(target: &Array2<f64>, fitted: &Array2<f64>, c0: usize, c1: usize) -> f64 {
    let t = target.slice(ndarray::s![.., c0..c1]).to_owned();
    let f = fitted.slice(ndarray::s![.., c0..c1]).to_owned();
    reconstruction_explained_variance(t.view(), f.view()).unwrap_or(0.0)
}

/// Fit a two-block circle whose behavior traverses at angular speed `ψ(t)`, and
/// return the fitted atom's reported representation–behavior isometry defect.
fn fitted_defect(uneven: bool) -> (f64, f64, f64) {
    let n = 96usize;
    let p_x = 4usize;
    let vocab = 4usize; // p_y = 3
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    for i in 0..n {
        let t = i as f64 / n as f64;
        let theta = std::f64::consts::TAU * t;
        // Activation: a clean unit circle — constant speed in t, IDENTICAL across
        // the two cases.
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos();
        z[[i, 3]] = 0.4 * (2.0 * theta).sin();
        // Behavior winds through its own circle at angular position ψ(t).
        let psi = if uneven {
            theta + 0.8 * theta.sin()
        } else {
            theta
        };
        let law = sphere_circle_probabilities(psi, theta, 0.0);
        for j in 0..vocab {
            probs[[i, j]] = law[j];
        }
    }

    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0).unwrap();
    let p_tot = p_x + block.behavior_dim();
    let augmented = block.augmented_target(z.view()).unwrap();

    let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term, mut rho) = build_k1(atom, cb);
    term.set_behavior_block(block).unwrap();
    term.set_guards_enabled(false);
    // Match the production fit entry: seed the linear decoder conditionally at
    // the planted chart before asking the joint nonlinear walk to move chart and
    // decoder together. A cold decoder makes the coordinate Jacobian vanish and
    // tests bootstrap behavior rather than the isometry statistic.
    term.refit_decoder_least_squares_at_current_state(augmented.view(), Some(&rho))
        .unwrap();
    term.run_joint_fit_arrow_schur(augmented.view(), &mut rho, None, 64, 1.0, 1e-6, 1e-6)
        .expect("two-block fit must complete");

    // Precondition: both blocks are actually reconstructed, so the induced speeds
    // read off the fitted decoders reflect the planted geometry (not fit noise).
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    let ev_act = block_ev(&augmented, &fitted, 0, p_x);
    let ev_beh = block_ev(&augmented, &fitted, p_x, p_tot);
    assert!(
        ev_act > 0.85,
        "activation EV too low ({ev_act}) to trust the speeds"
    );
    assert!(
        ev_beh > 0.85,
        "behavior EV too low ({ev_beh}) to trust the speeds"
    );

    let cert = atom_behavior_isometry(&term, 0)
        .expect("isometry certificate must compute")
        .expect("a d=1 two-block atom must yield a certificate");
    assert!(cert.behavior_engaged, "behavior must be engaged (it moves)");
    assert!(
        cert.scale.is_finite() && cert.scale > 0.0,
        "scale {}",
        cert.scale
    );
    assert!(
        cert.nats_per_unit_t.is_finite() && cert.nats_per_unit_t > 0.0,
        "nats/unit t {}",
        cert.nats_per_unit_t
    );
    let pinned = cert
        .behavior_pinned_chart
        .as_ref()
        .expect("an engaged non-degenerate behavior circle must have a pinned chart");
    assert_eq!(
        pinned.coords[pinned.anchor_row], 0.0,
        "the data-derived anchor row must be pinned exactly at the origin"
    );
    assert!(matches!(pinned.orientation, -1 | 1));
    assert_eq!(pinned.nats_per_unit_coordinate, 2.0);
    assert!(pinned.behavior_length.is_finite() && pinned.behavior_length > 0.0);
    let period = pinned.period.expect("the behavior circle must be periodic");
    assert_eq!(
        period,
        pinned.behavior_length * std::f64::consts::FRAC_1_SQRT_2
    );
    (cert.defect_cv, cert.scale, cert.nats_per_unit_t)
}

/// The reported isometry defect separates a scaled-isometric two-block atom (low
/// defect) from one whose behavior winds unevenly relative to its activation
/// (high defect) — even though the two share an IDENTICAL activation image, so no
/// activation-only statistic could tell them apart.
#[test]
fn isometry_defect_separates_isometric_from_broken() {
    let (defect_iso, _scale_iso, _nats_iso) = fitted_defect(false);
    let (defect_broken, _scale_broken, _nats_broken) = fitted_defect(true);
    eprintln!(
        "[#2015 fixed coupling] isometric_defect={defect_iso:.6e}, broken_defect={defect_broken:.6e}"
    );

    // The scaled-isometric atom reports a small defect: the two induced metrics
    // are proportional along the shared coordinate.
    assert!(
        defect_iso < 0.05,
        "scaled-isometric atom should report a low defect, got {defect_iso}"
    );
    // The broken atom's behavior winds ~9× faster/slower across the circle, so its
    // ratio-to-activation is far from constant.
    assert!(
        defect_broken > 0.30,
        "broken-isometry atom should report a high defect, got {defect_broken}"
    );
    // And the separation is wide — the statistic is decisive, not marginal.
    assert!(
        defect_broken > 2.0 * defect_iso,
        "defect must separate the two cases: isometric {defect_iso} vs broken {defect_broken}"
    );
}

/// The full #2015 pipeline, read off the CORRECT seam: the coupling weight `λ_y`
/// is selected by REML (`run_two_block_reml_fit` → `run_multiblock_reml_fit` →
/// the shared-latent arrow-Schur assembly, NOT the sparse-dict `cofit` linear
/// seam), and the isometry defect is then measured on the FITTED coordinates of
/// that REML-coupled fit.
///
/// Returns `(defect_cv, converged, identifiable, log_lambda_y)` so the caller can
/// assert both that REML actually selected the weight (the fit converged to a
/// finite, identifiable data-determined `λ_y`) and that the defect separates the
/// isometric from the broken planting on the resulting fit.
fn reml_fitted_defect(uneven: bool) -> (f64, bool, bool, f64) {
    let n = 96usize;
    let p_x = 4usize;
    let vocab = 4usize; // p_y = 3
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    // A clean unit circle is EXACTLY harmonic, so the five-column basis (the
    // constant plus two sine/cosine harmonic pairs) fits it to
    // round-off and R_x → 0 pushes λ_y toward a near-degenerate fixed point that
    // is a poor conditioning test for the REML iteration. A small deterministic
    // activation perturbation (a fixed fraction of a low harmonic the basis
    // cannot fully absorb into the planted channels) keeps R_x safely positive,
    // so the variance-ratio fixed point is well-conditioned and converges — while
    // leaving the fitted DECODER (which the induced speeds are read off) on the
    // clean circle, so the isometry geometry is unchanged.
    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    for i in 0..n {
        let t = i as f64 / n as f64;
        let theta = std::f64::consts::TAU * t;
        // Deterministic zero-mean high-harmonic wiggle (order 9 > basis order 2),
        // ~5% amplitude: the basis cannot represent it, so it is genuine residual.
        let wiggle = 0.05 * (9.0 * theta).sin();
        // Activation: a clean unit circle (IDENTICAL across the two cases) plus the
        // unrepresentable wiggle on one channel to keep R_x conditioned.
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos() + wiggle;
        z[[i, 3]] = 0.4 * (2.0 * theta).sin();
        // Behavior winds through its own circle at angular position ψ(t): evenly
        // (scaled isometry) or unevenly (bent correspondence).
        let psi = if uneven {
            theta + 0.8 * theta.sin()
        } else {
            theta
        };
        let law = sphere_circle_probabilities(psi, theta, 0.05);
        for j in 0..vocab {
            probs[[i, j]] = law[j];
        }
    }

    // Seed the coupling AWAY from where REML lands so a moved, identifiable
    // weight is evidence the data selected it rather than a held knob.
    let start_log_lambda = 0.0_f64;
    let block = BehaviorBlock::fit(probs.view(), p_x, start_log_lambda).unwrap();
    let p_tot = p_x + block.behavior_dim();

    let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term, mut rho) = build_k1(atom, cb);
    term.set_behavior_block(block).unwrap();
    term.set_guards_enabled(false);
    // Seed at the declared starting coupling; each REML sweep then rebuilds the
    // augmented target at its current λ_y through the production driver below.
    let seed_target = term
        .behavior_block()
        .unwrap()
        .augmented_target(z.view())
        .unwrap();
    term.refit_decoder_least_squares_at_current_state(seed_target.view(), Some(&rho))
        .unwrap();

    // The correct seam: two-block joint fit with λ_y REML-selected, on the raw
    // activation Z (the augmented target is stacked internally at each sweep's
    // current λ_y).
    let report = term
        .run_two_block_reml_fit(
            z.view(),
            &mut rho,
            None,
            TwoBlockRemlControls {
                max_sweeps: 20,
                inner_max_iter: 64,
                step_size: 1.0,
                ridge_ext_coord: 1e-6,
                ridge_beta: 1e-6,
                log_lambda_tol: 1e-3,
            },
        )
        .expect("two-block REML fit must complete");

    // Precondition: both blocks are actually reconstructed at the SELECTED λ_y,
    // so the induced speeds reflect the planted geometry, not fit noise. Rebuild
    // the augmented target at the fitted weight the term now carries.
    let fitted_block = term.behavior_block().unwrap().clone();
    let augmented = fitted_block.augmented_target(z.view()).unwrap();
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    let ev_act = block_ev(&augmented, &fitted, 0, p_x);
    let ev_beh = block_ev(&augmented, &fitted, p_x, p_tot);
    assert!(
        ev_act > 0.85,
        "activation EV too low ({ev_act}) to trust the speeds"
    );
    assert!(
        ev_beh > 0.85,
        "behavior EV too low ({ev_beh}) to trust the speeds"
    );

    // The isometry defect, read off the REML-coupled fit's fitted coordinates.
    let cert = atom_behavior_isometry(&term, 0)
        .expect("isometry certificate must compute")
        .expect("a d=1 two-block atom must yield a certificate");
    assert!(cert.behavior_engaged, "behavior must be engaged (it moves)");
    assert!(
        cert.scale.is_finite() && cert.scale > 0.0,
        "scale {}",
        cert.scale
    );
    assert!(
        cert.nats_per_unit_t.is_finite() && cert.nats_per_unit_t > 0.0,
        "nats/unit t {}",
        cert.nats_per_unit_t
    );
    // The λ_y that came back is what the term actually holds — the coupling the
    // defect above was measured under.
    let installed = fitted_block.log_lambda_y;
    assert!(
        (installed - report.log_lambda_y).abs() < 1e-12,
        "installed λ_y {installed} disagrees with report {}",
        report.log_lambda_y
    );
    eprintln!(
        "[#2015 REML {}] defect={:.6e}, activation_ev={ev_act:.6}, behavior_ev={ev_beh:.6}, \
         converged={}, identifiable={}, sweeps={}, log_lambda={:.6e}, trajectory={:?}",
        if uneven { "broken" } else { "isometric" },
        cert.defect_cv,
        report.converged,
        report.lambda_identifiable,
        report.sweeps,
        report.log_lambda_y,
        report.log_lambda_trajectory,
    );
    (
        cert.defect_cv,
        report.converged,
        report.lambda_identifiable,
        report.log_lambda_y,
    )
}

/// End-to-end #2015 acceptance: with the coupling weight selected by REML on the
/// correct shared-latent seam, the isometry defect measured on the resulting
/// FITTED coordinates still (a) reports the coupling was data-selected and
/// identifiable, and (b) separates a scaled-isometric atom from one whose
/// behavior winds unevenly — proving the two payoffs compose, not just coexist.
#[test]
fn reml_selected_coupling_then_isometry_defect_separates() {
    let (defect_iso, conv_iso, ident_iso, log_lambda_iso) = reml_fitted_defect(false);
    let (defect_broken, conv_broken, ident_broken, log_lambda_broken) = reml_fitted_defect(true);

    // REML actually selected the coupling in both fits: the variance-ratio fixed
    // point converged to a finite, IDENTIFIABLE weight — i.e. both blocks carried
    // residual variance and the data determined λ_y (the destination is the REML
    // stationary point, not the arbitrary 0.0 start; see the start-invariance
    // arm of `reml_selects_lambda_y_at_planted_variance_ratio`).
    for (conv, ident, log_lambda, label) in [
        (conv_iso, ident_iso, log_lambda_iso, "isometric"),
        (conv_broken, ident_broken, log_lambda_broken, "broken"),
    ] {
        assert!(conv, "{label}: λ_y fixed point did not converge");
        assert!(
            ident,
            "{label}: λ_y should be identifiable (behavior varies)"
        );
        assert!(
            log_lambda.is_finite(),
            "{label}: REML-selected log λ_y must be finite, got {log_lambda}"
        );
    }

    // The defect, measured on the REML-coupled fit, still separates the cases.
    assert!(
        defect_iso < 0.05,
        "scaled-isometric atom should report a low defect on the REML fit, got {defect_iso}"
    );
    assert!(
        defect_broken > 0.30,
        "broken-isometry atom should report a high defect on the REML fit, got {defect_broken}"
    );
    assert!(
        defect_broken > 2.0 * defect_iso,
        "defect must separate on the REML-coupled fit: isometric {defect_iso} vs broken \
         {defect_broken}"
    );
}
