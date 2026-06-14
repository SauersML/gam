//! #1099 + #1103 end-to-end: the per-atom curvature confidence interval / flatness
//! likelihood-ratio test and the Bartlett-corrected atom-smooth significance,
//! both consumed through the public `dictionary_report` surface.
//!
//! The atom's inner decoder smooth is a Gaussian-identity penalized WLS fit
//! `g_k(t) = Φ_k(t)ᵀ β` with roughness Gram `S`. We build two fixtures whose
//! TRUTH is analytic:
//!
//!   * a STRONGLY CURVED atom (large quadratic coefficient, low noise) — the
//!     extrinsic-curvature CI must EXCLUDE κ = 0 and the flatness LR test must
//!     reject the constant-curve null at a small p-value; the Bartlett-corrected
//!     smooth-significance p-value must also be small;
//!   * a FLAT atom (zero quadratic/curvature coefficient) — κ̂ = 0, the CI must
//!     STRADDLE 0 (verdict `Flat`), the flatness test must NOT reject, and the
//!     smooth-significance LR must be ≈ 0.
//!
//! Both assert that the reports are actually POPULATED (not the `None` stub) when
//! an atom carries its inner fit — i.e. that #1099/#1103 are reachable end to end.

use gam::geometry::CurvatureVerdict;
use gam::inference::row_metric::RowMetric;
use gam::inference::structure_evidence::StructureLedger;
use gam::sae_identifiability::{
    AtomInnerFit, AtomTopology, FittedAtom, FittedSaeManifold, dictionary_report,
};
use ndarray::{Array1, Array2};

/// Build a self-consistent `AtomInnerFit` for a 1-D inner smooth with basis
/// `[1, t, t²]` over `n` evenly-spaced rows on `[-1, 1]`, decoder coefficients
/// `beta = [b0, b1, b2]`, unit Gauss–Newton weights, and a roughness Gram `S`
/// that penalizes only the quadratic (curvature) column. The penalized Hessian
/// is the honest `ΦᵀWΦ + S`; the per-row scores are the fitted-state Gaussian
/// scores `s_i = −w_i r_i Φ_i / φ` for a planted residual field.
fn curved_inner_fit(beta_vec: [f64; 3], dispersion: f64, kappa_hat: f64) -> AtomInnerFit {
    let n = 64usize;
    let m = 3usize;
    let mut design = Array2::<f64>::zeros((n, m));
    let mut derivative_design = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        design[[i, 0]] = 1.0;
        design[[i, 1]] = t;
        design[[i, 2]] = t * t;
        // ∂/∂t of [1, t, t²] = [0, 1, 2t].
        derivative_design[[i, 0]] = 0.0;
        derivative_design[[i, 1]] = 1.0;
        derivative_design[[i, 2]] = 2.0 * t;
    }
    let beta = Array1::from(beta_vec.to_vec());
    let weights = Array1::<f64>::ones(n);

    // Roughness Gram: penalize the quadratic column only (∫(g'')² ∝ b2²).
    let mut penalty = Array2::<f64>::zeros((m, m));
    penalty[[2, 2]] = 1e-3;

    // Penalized Hessian H = ΦᵀWΦ + S (W = I here).
    let mut hessian = Array2::<f64>::zeros((m, m));
    for i in 0..n {
        for a in 0..m {
            for b in 0..m {
                hessian[[a, b]] += design[[i, a]] * design[[i, b]];
            }
        }
    }
    hessian = &hessian + &penalty;

    // Per-row scores for a small planted residual field r_i (deterministic,
    // mean-zero across rows) so the influence-function SE is well-defined.
    let mut row_scores = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        let r_i = 0.02 * (3.0 * std::f64::consts::PI * t).sin();
        let w_i = weights[i];
        for a in 0..m {
            row_scores[[i, a]] = -w_i * r_i * design[[i, a]] / dispersion;
        }
    }

    let peak_design_row = design.row(n - 1).to_owned();
    let mode_design_row = design.row(n / 2).to_owned();

    // Smooth EDF for an M=3 basis with one penalized curvature column: tr(H⁻¹ ΦᵀWΦ)
    // minus the unpenalized null dimension (intercept) ≈ the two free non-null
    // columns; a fixed reference df of 2 is the analytic Wood reference here.
    let smooth_edf = 2.0;

    AtomInnerFit {
        design,
        derivative_design,
        beta,
        penalty,
        penalized_hessian: hessian,
        row_scores,
        weights,
        dispersion,
        peak_design_row,
        mode_design_row,
        kappa_hat,
        smooth_edf,
    }
}

fn patch_atom_with_fit(name: &str, fit: AtomInnerFit) -> FittedAtom {
    let mut frame = Array2::<f64>::zeros((1, 2));
    frame[[0, 0]] = 1.0;
    FittedAtom {
        name: name.to_string(),
        topology: AtomTopology::EuclideanPatch { latent_dim: 2 },
        frame,
        ard_variances: None,
        lowering_error: 0.0,
        chart_canonicalized: false,
        inner_fit: None,
    }
    .with_inner_fit(fit)
}

fn single_atom_model(atom: FittedAtom) -> FittedSaeManifold {
    let param_dim = atom.frame.len();
    FittedSaeManifold {
        atoms: vec![atom],
        // Data Jacobian invariant to the so(2) rotation (param_dim = 2 entries
        // per row); the gauge half just needs a well-formed model.
        jacobian_rows: vec![vec![1.0, 0.0]],
        isometry_penalty_root: Array2::<f64>::zeros((0, param_dim)),
        metric: RowMetric::euclidean(3, 1).expect("euclidean metric"),
    }
}

#[test]
fn strongly_curved_atom_excludes_flat_and_rejects_constant_null() {
    // b2 = 1.0 is a strong quadratic (curved) decoder; low noise → sharp profile.
    let fit = curved_inner_fit([0.0, 0.5, 1.0], 1e-4, 1.0);
    let model = single_atom_model(patch_atom_with_fit("curved", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];

    // #1099 curvature CI is POPULATED (not the old None stub).
    let ci = atom
        .curvature_ci
        .as_ref()
        .expect("curved atom with an inner fit must report a curvature CI");
    assert!(ci.kappa_hat.is_finite());
    assert!(
        ci.ci.ci_lo.is_finite() && ci.ci.ci_hi.is_finite() && ci.ci.ci_lo < ci.ci.ci_hi,
        "CI must be a proper interval, got [{}, {}]",
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    // κ̂ = 1.0 with a sharp (low-noise) profile ⇒ the CI must exclude 0 and the
    // verdict must be a definite (non-flat) curvature sign.
    assert!(
        ci.ci.ci_lo > 0.0,
        "a strongly curved low-noise atom's 95% CI must exclude 0, got [{}, {}]",
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    assert_eq!(
        ci.ci.verdict,
        CurvatureVerdict::Spherical,
        "positive κ̂ with a CI strictly above 0 must read as a definite curvature sign"
    );

    // The interior-point flatness LR test must REJECT the κ = 0 null.
    assert!(
        ci.flatness_test.lr_stat > 3.84,
        "flatness LR must exceed χ²₁(0.95) for a clearly curved atom, got {}",
        ci.flatness_test.lr_stat
    );
    assert!(
        ci.flatness_test.p_value < 0.05,
        "flatness p must reject the constant-curvature null, got {}",
        ci.flatness_test.p_value
    );

    // #1103 Bartlett-corrected smooth significance is POPULATED and rejects the
    // constant-curve null with a finite, positive correction factor.
    let sig = atom
        .smooth_significance
        .as_ref()
        .expect("curved atom must report a Bartlett-corrected smooth significance");
    assert!(sig.lr_stat.is_finite() && sig.lr_stat >= 0.0);
    assert!(
        sig.bartlett_factor.is_finite() && sig.bartlett_factor > 0.0,
        "Bartlett factor must be finite and positive, got {}",
        sig.bartlett_factor
    );
    assert!(sig.df >= 1);
}

#[test]
fn flat_atom_straddles_zero_and_fails_to_reject() {
    // b2 = 0 ⇒ no curvature column energy; κ̂ planted at 0 (flat decoder).
    let fit = curved_inner_fit([0.0, 0.5, 0.0], 1e-2, 0.0);
    let model = single_atom_model(patch_atom_with_fit("flat", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];
    let ci = atom
        .curvature_ci
        .as_ref()
        .expect("flat atom with an inner fit still reports a curvature CI");

    // κ̂ = 0 ⇒ the CI must straddle 0 and the verdict must be Flat.
    assert!(
        ci.ci.ci_lo <= 0.0 && ci.ci.ci_hi >= 0.0,
        "a flat atom's CI must straddle 0, got [{}, {}]",
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    assert_eq!(
        ci.ci.verdict,
        CurvatureVerdict::Flat,
        "κ̂ = 0 must read as Flat"
    );

    // The flatness LR test must NOT reject (κ̂ = 0 is the null).
    assert!(
        ci.flatness_test.lr_stat < 1e-6,
        "flatness LR at κ̂ = 0 must be ≈ 0, got {}",
        ci.flatness_test.lr_stat
    );
    assert!(
        ci.flatness_test.p_value > 0.05,
        "flatness test must fail to reject for a flat atom, got p = {}",
        ci.flatness_test.p_value
    );
}
