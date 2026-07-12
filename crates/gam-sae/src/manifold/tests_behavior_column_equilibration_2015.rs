//! #2015 — column-equilibration primitive test, and the revived planted
//! ~1e4-spread convergence gate now driven by the SOLVER-LEVEL fix.
//!
//! HISTORY: unit-RMS DATA equilibration was wired into the crosscoder fit
//! path (`run_auto_sae_crosscoder_fit`/`run_sae_crosscoder_fit`) and REVERTED
//! the same night: for a homoscedastic reconstruction objective, dividing
//! columns by their RMS is not a reparametrization — it changes the estimand
//! (noise-dominated columns are amplified to unit RMS and the fit spends
//! capacity explaining them). Measured on MSI 13021686 at the wired commit:
//! `zz2015_tiny_inner_crawl_terminates` refused at the co-collapse floor
//! (EV 0.4566 vs null 0.4583) and the planted transport-law verdicts
//! collapsed (phase R² 0.139, smooth R² 0.837). The `equilibrate_crosscoder_columns`
//! primitive below is kept (spec'd + unit-tested); the fit path passes a unit
//! `column_scale` and no longer calls it.
//!
//! The κ≈1e8 conditioning fix instead lives in the inner SOLVER's linear
//! algebra (design: issue 2015 comment 4949898801, landed in
//! `gam_solve::arrow_schur::reduced_solve::factor_dense_reduced_schur`):
//! Jacobi/Van der Sluis diagonal equilibration of the dense reduced-Schur
//! complement BEFORE its Cholesky factorization, undone exactly (`L = D·L̃`)
//! before the factor is returned — a pure numerical-conditioning aid with no
//! signature change and no estimand change (unlike the reverted data-frame
//! attempt). Because that fix is internal to the solver, the test below
//! exercises the UNMODIFIED `run_auto_sae_behavior_fit` front door directly on
//! an ill-conditioned planted fixture — no equilibration call anywhere in
//! this test — and the fit should converge because the solver now factors the
//! ill-conditioned Schur accurately.

use ndarray::Array2;

use crate::manifold::{
    SaeBehaviorAutoFitRequest, SaeCrosscoderAutoFitConfig, equilibrate_crosscoder_columns,
    run_auto_sae_behavior_fit,
};

/// Direct unit check of the equilibration primitive itself: every non-empty
/// column of the mutated target has unit RMS, and un-scaling by the returned
/// factor exactly reproduces the original (a diagonal, lossless change of
/// variables).
#[test]
fn equilibrate_crosscoder_columns_normalizes_and_round_trips() {
    let mut target = Array2::<f64>::zeros((5, 3));
    // Column scale ratio 1e4: column 0 ~ O(1e2), column 1 ~ O(1), column 2 ~ O(1e-2).
    for i in 0..5 {
        let t = i as f64;
        target[[i, 0]] = 100.0 * (t + 1.0);
        target[[i, 1]] = 1.0 * (t + 1.0);
        target[[i, 2]] = 0.01 * (t + 1.0);
    }
    let original = target.clone();
    let scale = equilibrate_crosscoder_columns(&mut target);
    assert_eq!(scale.len(), 3);
    for col in 0..3 {
        let rms = (target.column(col).iter().map(|v| v * v).sum::<f64>() / 5.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-9,
            "column {col} must be unit-RMS after equilibration, got {rms}"
        );
    }
    // Undo: multiplying each column back by its scale must exactly reproduce
    // the original (equilibration is a lossless per-column rescale).
    for i in 0..5 {
        for j in 0..3 {
            let restored = target[[i, j]] * scale[j];
            assert!(
                (restored - original[[i, j]]).abs() <= 1e-9 * original[[i, j]].abs().max(1.0),
                "round trip mismatch at ({i},{j}): {restored} vs {}",
                original[[i, j]]
            );
        }
    }
    // The ratio of the largest to smallest scale reflects the planted 1e4 spread.
    let scale_max = scale.iter().cloned().fold(0.0_f64, f64::max);
    let scale_min = scale.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        scale_max / scale_min > 1e3,
        "the equilibration scale must reflect the planted column-scale spread, got ratio {}",
        scale_max / scale_min
    );
}

/// Row-aligned circle activation (`p_x = 4`) and behavior probabilities
/// (`vocab = 4`, so `p_y = 3`) at `n` positions. `residual_amp` sets the
/// amplitude of ONE tangent direction (`e3`) relative to the fixed
/// `radius = 0.25` circular signal carried by the other two tangent
/// directions (`e1`, `e2`) — the column-scale-spread lever. The activation
/// construction does not depend on `residual_amp` at all, so the two fixtures
/// the test below compares share a BYTE-IDENTICAL activation target; only the
/// behavior block's own column-scale spread differs.
fn planted_fixture(n: usize, residual_amp: f64) -> (Array2<f64>, Array2<f64>) {
    let inv_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
    let b = [0.5_f64; 4];
    let e1 = [inv_sqrt_two, -inv_sqrt_two, 0.0, 0.0];
    let e2 = [0.5, 0.5, -0.5, -0.5];
    let e3 = [0.0, 0.0, inv_sqrt_two, -inv_sqrt_two];
    let radius = 0.25_f64;
    let mut z = Array2::<f64>::zeros((n, 4));
    let mut probs = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let t = i as f64 / n as f64;
        let theta = std::f64::consts::TAU * t;
        // A small deterministic high-harmonic wiggle keeps the activation
        // residual positive (so the fixed-λ_y fit sees a genuine two-block
        // residual) without perturbing the circle the fitted decoder chases.
        let wiggle = 0.05 * (9.0 * theta).sin();
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos() + wiggle;
        z[[i, 3]] = 0.4 * (2.0 * theta).sin();

        // The behavior half-density: the circular signal in e1/e2 at fixed
        // `radius`, plus a residual in e3 whose amplitude is the spread lever.
        let residual = residual_amp * (3.0 * theta).sin();
        let radial = (1.0 - radius * radius - residual * residual).sqrt();
        let (cos_t, sin_t) = (theta.cos(), theta.sin());
        for token in 0..4 {
            let q = radial * b[token]
                + radius * (cos_t * e1[token] + sin_t * e2[token])
                + residual * e3[token];
            probs[[i, token]] = q * q;
        }
    }
    (z, probs)
}

fn fit_config() -> SaeCrosscoderAutoFitConfig {
    let mut config = SaeCrosscoderAutoFitConfig::standard(1, 3);
    config.max_iter = 40;
    // Fixed-rho (λ_y held at its seed value) isolates the INNER solve's
    // conditioning from the outer λ_y selection, matching the fast
    // `zz2015_tiny_inner_crawl_terminates` convergence gate this test mirrors.
    config.run_outer_rho_search = false;
    config
}

/// #2015 — a behavior block whose own tangent-coordinate columns carry a
/// genuine ~1e4 within-block scale spread must still converge through the
/// UNMODIFIED `run_auto_sae_behavior_fit` front door (no equilibration call
/// anywhere in this test — the fix lives entirely inside the solver's dense
/// Schur factorization), and must recover essentially the SAME joint
/// reconstruction as an otherwise-identical well-scaled (spread ~1) fixture —
/// proof that the solver-level Jacobi/Van der Sluis fix neutralizes the
/// spread's effect on the inner solve's conditioning without changing what
/// gets fitted (the estimand is untouched this time: nothing in the fit
/// pipeline's data frame changed, only the solver's internal factorization
/// path — see `factor_dense_reduced_schur_reconstructs_original_illconditioned_matrix_2015`
/// in `gam-solve` for the primitive-level identity this end-to-end gate
/// depends on).
#[test]
fn planted_1e4_column_spread_behavior_block_converges_and_matches_well_scaled() {
    let n = 96usize;
    // Well-scaled: the residual direction's amplitude is the same order as the
    // signal directions'.
    let (z_well, probs_well) = planted_fixture(n, 0.05);
    // Ill-conditioned: the residual direction's amplitude is 1e-4x the
    // well-scaled fixture's — a deliberate ~1e4 within-behavior-block
    // column-scale spread. The activation target is untouched (identical to
    // `z_well`), so any degradation in ITS reconstruction quality is entirely
    // a symptom of the shared joint solve's conditioning, not new activation
    // structure.
    let (z_ill, probs_ill) = planted_fixture(n, 0.05e-4);
    assert_eq!(
        z_well, z_ill,
        "the activation target must be untouched by the spread lever"
    );

    let well = run_auto_sae_behavior_fit(SaeBehaviorAutoFitRequest {
        activation: z_well,
        probabilities: probs_well,
        config: fit_config(),
        cancel: None,
    })
    .expect("well-scaled fixture must fit");

    let ill = run_auto_sae_behavior_fit(SaeBehaviorAutoFitRequest {
        activation: z_ill,
        probabilities: probs_ill,
        config: fit_config(),
        cancel: None,
    })
    .expect(
        "the solver's Jacobi-conditioned dense Schur factorization must let the ~1e4-spread \
         behavior block converge with NO data-frame equilibration in the fit path",
    );

    assert_eq!(ill.crosscoder.layers.len(), 2);
    for layer in &ill.crosscoder.layers {
        assert!(
            layer.reconstruction_r2.is_finite() && layer.reconstruction_r2 > 0.9,
            "{}: ill-conditioned fit must still reconstruct well, got {}",
            layer.label,
            layer.reconstruction_r2
        );
    }
    assert_eq!(
        ill.kl.infinite_rows, 0,
        "no fitted row may decode off-simplex"
    );

    // The recovered ACTIVATION reconstruction (gauge-invariant: it is decoded
    // output, not an internal coordinate) must match the well-scaled fixture's
    // closely — the shared latent/gate the behavior block's spread could have
    // destabilized did not, in fact, destabilize the activation block either.
    let activation_diff = &ill.crosscoder.layers[0].fitted - &well.crosscoder.layers[0].fitted;
    let activation_max_abs_diff = activation_diff
        .iter()
        .cloned()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        activation_max_abs_diff < 0.1,
        "activation reconstruction must match the well-scaled fixture's to tolerance, got max diff {activation_max_abs_diff}"
    );

    // The recovered BEHAVIOR (decoded distributions) must likewise match.
    let mut behavior_max_abs_diff = 0.0_f64;
    for (a, b) in ill
        .fitted_probabilities
        .iter()
        .zip(well.fitted_probabilities.iter())
    {
        behavior_max_abs_diff = behavior_max_abs_diff.max((a - b).abs());
    }
    assert!(
        behavior_max_abs_diff < 0.1,
        "fitted behavior probabilities must match the well-scaled fixture's to tolerance, got max diff {behavior_max_abs_diff}"
    );
}
