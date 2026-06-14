//! #1099 + #1103 end-to-end: the per-atom curvature delta-method standard error
//! and the any-n-valid atom-smooth structure e-value, both consumed through the
//! public `dictionary_report` surface.
//!
//! #1099 reports the plug-in curvature bound `κ̂ = atom_curvature_bound(β̂)` and
//! its delta-method SE `sqrt((∂κ/∂β)ᵀ H⁻¹ (∂κ/∂β))` through the inner-fit
//! penalized Hessian, with a Wald normal band. κ is a sup-norm bound read off
//! the fitted decoder — nothing in any optimisation moves it, so there is no
//! profiled criterion to walk; the honest uncertainty is the delta-method band.
//!
//! #1103 reports the split-likelihood-ratio e-value for "the atom's smooth is
//! non-constant" (the same universal-inference instrument the atom-birth gate
//! uses), replacing the earlier Lawley–Bartlett-corrected χ².
//!
//! The atom's inner decoder smooth is a Gaussian-identity penalized WLS fit
//! `g_k(t) = Φ_k(t)ᵀ β` with roughness Gram `S`. We build a fixture that carries
//! a finite curvature gradient so the delta-method SE is real, and assert the
//! reports are actually POPULATED (not the `None` stub) with a finite SE, a
//! proper Wald interval, and positive non-constant evidence for a curved atom.

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

    // Curvature-functional gradient ∂κ/∂β for the #1099 delta-method SE. The
    // extrinsic-curvature bound is driven by the quadratic (curvature) column, so
    // a representative finite gradient concentrates on that column; the SE
    // sqrt((∂κ/∂β)ᵀ H⁻¹ (∂κ/∂β)) is then a real, finite delta-method band.
    let kappa_grad = Some(Array1::from(vec![0.0, 0.1, 1.0]));

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
        kappa_grad,
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
fn curved_atom_reports_real_delta_method_curvature_se() {
    // A strongly curved decoder (large quadratic, low noise) carrying a finite
    // curvature gradient ⇒ the #1099 delta-method SE is real and the report is
    // POPULATED (not the old None stub).
    let fit = curved_inner_fit([0.0, 0.5, 1.0], 1e-2, 0.4);
    let model = single_atom_model(patch_atom_with_fit("curved", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];

    // #1099 curvature delta-method band is COMPUTED (not None).
    let ci = atom
        .curvature_ci
        .as_ref()
        .expect("curved atom with a curvature gradient must report a delta-method SE");

    // κ̂ is the plug-in bound; SE is a finite non-negative delta-method SE.
    assert!(ci.kappa_hat.is_finite(), "κ̂ must be finite");
    assert!(
        ci.se.is_finite() && ci.se >= 0.0,
        "delta-method SE must be finite and non-negative, got {}",
        ci.se
    );
    // The Wald band is a proper interval centred at κ̂ with half-width z·SE.
    let (lo, hi) = ci.ci_normal;
    assert!(
        lo.is_finite() && hi.is_finite() && lo <= hi,
        "CI must be a proper interval, got [{lo}, {hi}]"
    );
    assert!(
        ci.kappa_hat >= lo && ci.kappa_hat <= hi,
        "κ̂ = {} must lie inside its own Wald band [{lo}, {hi}]",
        ci.kappa_hat
    );
    let half = 0.5 * (hi - lo);
    assert!(
        (half - 1.959_963_984_540_054 * ci.se).abs() < 1e-9,
        "Wald half-width {half} must equal z·SE for z=1.959964, SE={}",
        ci.se
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
fn flat_atom_still_yields_a_real_delta_method_se() {
    // A flat decoder (no quadratic energy) still carries a finite curvature
    // gradient and an SPD inner Hessian, so the #1099 delta-method band is real
    // and populated. κ̂ must lie inside its own Wald band.
    let fit = curved_inner_fit([0.0, 0.5, 0.0], 1e-2, 0.0);
    let model = single_atom_model(patch_atom_with_fit("flat", fit));
    let ledger = StructureLedger::new();
    let report = dictionary_report(&model, &ledger, 0.05).expect("dictionary report");

    let atom = &report.atom_inference[0];
    let ci = atom
        .curvature_ci
        .as_ref()
        .expect("flat atom with a curvature gradient still reports a delta-method SE");

    assert!(
        ci.se.is_finite() && ci.se >= 0.0,
        "delta-method SE must be finite and non-negative, got {}",
        ci.se
    );
    let (lo, hi) = ci.ci_normal;
    assert!(
        lo.is_finite() && hi.is_finite() && lo <= hi,
        "CI must be a proper interval, got [{lo}, {hi}]"
    );
    assert!(
        ci.kappa_hat >= lo && ci.kappa_hat <= hi,
        "κ̂ = {} must lie inside its own Wald band [{lo}, {hi}]",
        ci.kappa_hat
    );
}
