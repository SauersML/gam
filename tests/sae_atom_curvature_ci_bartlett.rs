//! #1099 + #1103 end-to-end: the per-atom curvature confidence interval / flatness
//! likelihood-ratio test and the Bartlett-corrected atom-smooth significance,
//! both consumed through the public `dictionary_report` surface.
//!
//! The #1099 curvature CI profiles the atom's *exact* `M_κ` penalized REML
//! negative log-evidence along the latent curvature channel `κ`: κ rescales the
//! atom's roughness penalty through the captured geodesic-length jet
//! `(s(κ_c), s′, s″)` by `τ(κ) = (s(κ_c)/s(κ))^{2r}`, the inner smooth is
//! re-solved `β(κ) = (ΦᵀWΦ + τ S)⁻¹ ΦᵀW z` at each κ, and `V_p(κ)` is the
//! resulting REML deviance + log-determinant evidence. The CI is the χ²₁ Wilks
//! crossing of `V_p`; the flatness test the interior-point LR at κ = 0.
//!
//! The atom's inner decoder smooth is a Gaussian-identity penalized WLS fit
//! `g_k(t) = Φ_k(t)ᵀ β` with roughness Gram `S`. We build a fixture that carries
//! a non-degenerate geodesic-length jet so the κ-evidence profile is real, and
//! assert the reports are actually POPULATED (not the `None` stub) with a proper
//! interval, an evaluable interior flatness LR, and a verdict consistent with the
//! CI sign — i.e. that #1099/#1103 are reachable end to end through the true
//! `M_κ` evidence oracle.

use gam::geometry::CurvatureVerdict;
use gam::inference::row_metric::RowMetric;
use gam::inference::structure_evidence::StructureLedger;
use gam::sae_identifiability::{
    AtomInnerFit, AtomTopology, FittedAtom, FittedSaeManifold, GeodesicLengthJet, dictionary_report,
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

    // Roughness Gram: the quadratic column carries the curvature energy
    // (∫(g'')² ∝ b2²); a unit Gram makes the energy-ratio curvature O(0.1) for a
    // unit quadratic, well inside the cusp floor.
    let mut penalty = Array2::<f64>::zeros((m, m));
    penalty[[2, 2]] = 1.0;

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
        // A small, deterministic, mean-zero residual field: the fit is nearly
        // perfect so the curvature SE is tight and the CI is sharp.
        let r_i = 1e-5 * (3.0 * std::f64::consts::PI * t).sin();
        let w_i = weights[i];
        for a in 0..m {
            row_scores[[i, a]] = -w_i * r_i * design[[i, a]] / dispersion;
        }
    }

    let peak_design_row = design.row(n - 1).to_owned();
    let mode_design_row = design.row(n / 2).to_owned();

    // REML cross terms: working response z_i = g(t_i) + r_i; b = ΦᵀW z, zᵀW z.
    let mut xtw_z = Array1::<f64>::zeros(m);
    let mut ztw_z = 0.0_f64;
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        let r_i = 1e-5 * (3.0 * std::f64::consts::PI * t).sin();
        let z_i = design.row(i).dot(&beta) + r_i;
        let w_i = weights[i];
        for col in 0..m {
            xtw_z[col] += w_i * z_i * design[[i, col]];
        }
        ztw_z += w_i * z_i * z_i;
    }

    // Geodesic-length jet of the latent chart at κ_centre = `kappa_hat`. A
    // positive `s` with a non-zero `ds` makes the penalty rescale τ(κ) move with
    // κ, so the M_κ REML evidence has a genuine κ-dependence (and a finite v_pp).
    // `s′ > 0` (geodesic lengths grow as curvature increases toward spherical) is
    // the constant-curvature chart's local behaviour about a small base κ.
    let geodesic_length_jet = Some(GeodesicLengthJet {
        kappa_centre: kappa_hat,
        s: 1.0,
        ds: 0.25,
        dds: 0.05,
    });

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
        geodesic_length_jet,
        penalty_order: 2,
        xtw_z,
        ztw_z,
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

/// The verdict reported by the κ-profile must be exactly the one implied by the
/// sign of its own CI — a self-consistency invariant the `M_κ` oracle owes.
fn verdict_matches_ci_sign(lo: f64, hi: f64) -> CurvatureVerdict {
    if lo > 0.0 {
        CurvatureVerdict::Spherical
    } else if hi < 0.0 {
        CurvatureVerdict::Hyperbolic
    } else {
        CurvatureVerdict::Flat
    }
}

#[test]
fn curved_atom_reports_real_mkappa_evidence_ci() {
    // A strongly curved decoder (large quadratic, low noise) carrying a genuine
    // geodesic-length jet ⇒ the M_κ penalized-REML evidence profile is real and
    // the #1099 report is POPULATED (not the old None stub).
    let fit = curved_inner_fit([0.0, 0.5, 1.0], 1e-2, 0.4);
    let model = single_atom_model(patch_atom_with_fit("curved", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];

    // #1099 curvature CI is COMPUTED (not None) — the common case the issue asks
    // for: an atom with a non-degenerate latent chart yields a real CI.
    let ci = atom
        .curvature_ci
        .as_ref()
        .expect("curved atom with a geodesic jet must report a real M_κ evidence CI");

    // κ̂ is the evidence argmin: finite, and inside the CI.
    assert!(ci.kappa_hat.is_finite(), "κ̂ must be finite");
    assert!(
        ci.ci.ci_lo.is_finite() && ci.ci.ci_hi.is_finite() && ci.ci.ci_lo < ci.ci.ci_hi,
        "CI must be a proper interval, got [{}, {}]",
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    assert!(
        ci.kappa_hat >= ci.ci.ci_lo && ci.kappa_hat <= ci.ci.ci_hi,
        "κ̂ = {} must lie inside its own CI [{}, {}]",
        ci.kappa_hat,
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    // The verdict must be self-consistent with the CI sign.
    assert_eq!(
        ci.ci.verdict,
        verdict_matches_ci_sign(ci.ci.ci_lo, ci.ci.ci_hi),
        "verdict must match the sign of the reported CI [{}, {}]",
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );

    // The interior-point flatness LR is evaluable, non-negative, and its p-value
    // is a valid probability — the κ = 0 evidence drop against the argmin.
    assert!(
        ci.flatness_test.lr_stat.is_finite() && ci.flatness_test.lr_stat >= 0.0,
        "flatness LR must be a finite non-negative statistic, got {}",
        ci.flatness_test.lr_stat
    );
    assert!(
        ci.flatness_test.p_value >= 0.0 && ci.flatness_test.p_value <= 1.0,
        "flatness p must be a probability, got {}",
        ci.flatness_test.p_value
    );

    // #1103 split-LRT smooth-structure e-value is POPULATED and finite. A
    // strongly curved (non-constant) decoder must carry POSITIVE log-evidence
    // for the non-constant alternative — the honest any-n-valid instrument
    // earns evidence where the Bartlett-corrected χ² used to.
    let sig = atom
        .smooth_significance
        .as_ref()
        .expect("curved atom must report a split-LRT smooth-structure e-value");
    let log_e = sig
        .log_e_nonconstant
        .expect("curved atom must carry a finite non-constant log-e-value");
    assert!(log_e.is_finite(), "log-e must be finite, got {log_e}");
    assert!(
        log_e > 0.0,
        "a strongly curved (non-constant) atom must accumulate positive split-LRT evidence, got log_e={log_e}"
    );
}

#[test]
fn flat_atom_still_yields_a_real_evidence_ci() {
    // A flat decoder (no quadratic energy) still carries a non-degenerate latent
    // chart, so the M_κ evidence profile is real and the CI is populated. The
    // evidence-argmin κ̂ must lie inside the CI and the verdict be self-consistent.
    let fit = curved_inner_fit([0.0, 0.5, 0.0], 1e-2, 0.0);
    let model = single_atom_model(patch_atom_with_fit("flat", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];
    let ci = atom
        .curvature_ci
        .as_ref()
        .expect("flat atom with a geodesic jet still reports a real M_κ evidence CI");

    assert!(
        ci.ci.ci_lo.is_finite() && ci.ci.ci_hi.is_finite() && ci.ci.ci_lo < ci.ci.ci_hi,
        "CI must be a proper interval, got [{}, {}]",
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    assert!(
        ci.kappa_hat >= ci.ci.ci_lo && ci.kappa_hat <= ci.ci.ci_hi,
        "κ̂ = {} must lie inside its own CI [{}, {}]",
        ci.kappa_hat,
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    assert_eq!(
        ci.ci.verdict,
        verdict_matches_ci_sign(ci.ci.ci_lo, ci.ci.ci_hi),
        "verdict must match the sign of the reported CI [{}, {}]",
        ci.ci.ci_lo,
        ci.ci.ci_hi
    );
    assert!(
        ci.flatness_test.lr_stat.is_finite() && ci.flatness_test.lr_stat >= 0.0,
        "flatness LR must be a finite non-negative statistic, got {}",
        ci.flatness_test.lr_stat
    );
    assert!(
        ci.flatness_test.p_value >= 0.0 && ci.flatness_test.p_value <= 1.0,
        "flatness p must be a probability, got {}",
        ci.flatness_test.p_value
    );
}
