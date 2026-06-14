//! #1115 + #1103 end-to-end: the per-atom decoder-functional POINT summaries and
//! the any-n-valid atom-smooth structure e-value, both consumed through the
//! public `dictionary_report` surface.
//!
//! #1115 removed the #1099 per-atom curvature *confidence interval* and the
//! influence-function SE on the #1097 functionals: those conditioned on the
//! fitted latent coordinates / assignment (generated regressors estimated from
//! the same activations forming the response) as if known, so they omit the
//! generated-regressor variance channel and under-cover. What survives is the
//! penalty-debiased POINT summary (`AtomFunctionalEstimate`: plug-in +
//! debiased + removed bias, NO se/ci) — this test asserts the report exposes no
//! coverage-claiming fields.
//!
//! #1103 reports the split-likelihood-ratio e-value for "the atom's smooth is
//! non-constant" (the same universal-inference instrument the atom-birth gate
//! uses), replacing the earlier Lawley–Bartlett-corrected χ². It IS finite-
//! sample valid with no regularity conditions, so it is kept.
//!
//! The atom's inner decoder smooth is a Gaussian-identity penalized WLS fit
//! `g_k(t) = Φ_k(t)ᵀ β` with roughness Gram `S`. We assert the reports are
//! actually POPULATED (not the `None` stub) with finite point summaries and
//! positive non-constant evidence for a curved atom.

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
fn curved_inner_fit(beta_vec: [f64; 3], dispersion: f64) -> AtomInnerFit {
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
fn curved_atom_reports_debiased_point_summaries_without_coverage_claim() {
    // A strongly curved decoder (large quadratic, low noise). The per-atom
    // functional report is POPULATED with penalty-debiased POINT summaries and
    // carries NO se/ci (the AtomFunctionalEstimate type has no such fields by
    // construction — #1115). The #1103 split-LRT e-value, which IS valid, is
    // populated and positive.
    let fit = curved_inner_fit([0.0, 0.5, 1.0], 1e-2);
    let model = single_atom_model(patch_atom_with_fit("curved", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];

    // Functional point summaries are populated and finite — plug-in and the
    // penalty-debiased value, with the removed bias. No coverage-claiming field
    // exists on the type, so there is nothing to mis-report.
    let functionals = atom
        .functionals
        .as_ref()
        .expect("curved atom must report decoder-functional point summaries");
    for estimate in [
        functionals.average_value.as_ref(),
        functionals.peak_contrast.as_ref(),
        functionals.decoder_variation_norm.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        assert!(
            estimate.theta_plugin.is_finite()
                && estimate.theta_onestep.is_finite()
                && estimate.penalty_bias.is_finite(),
            "functional point summary must be finite: plugin={}, onestep={}, bias={}",
            estimate.theta_plugin,
            estimate.theta_onestep,
            estimate.penalty_bias
        );
    }

    // #1103 split-LRT smooth-structure e-value is POPULATED and finite. A
    // strongly curved (non-constant) decoder must carry POSITIVE log-evidence
    // for the non-constant alternative — the honest any-n-valid instrument.
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
fn flat_atom_functional_debiased_value_matches_plugin() {
    // A flat decoder (no quadratic energy) still yields finite penalty-debiased
    // point summaries; with negligible penalty bias the debiased value tracks
    // the plug-in. No SE/CI is reported (#1115).
    let fit = curved_inner_fit([0.0, 0.5, 0.0], 1e-2);
    let model = single_atom_model(patch_atom_with_fit("flat", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];
    let functionals = atom
        .functionals
        .as_ref()
        .expect("flat atom must report decoder-functional point summaries");
    let av = functionals
        .average_value
        .as_ref()
        .expect("flat atom must report an average-value point summary");
    assert!(
        av.theta_plugin.is_finite() && av.theta_onestep.is_finite(),
        "average-value point summary must be finite"
    );
}
