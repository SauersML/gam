//! Python boundary for the landed inference instruments that previously had no
//! user-facing caller: the anytime-valid structure-discovery e-process / e-BH
//! certificate (issue #984) and the Lawley likelihood-ratio Bartlett correction
//! (issue #939).
//!
//! Design discipline:
//! * **#984 is safe-by-construction.** The predictability contract that makes
//!   the e-process a supermartingale under H0 (the alternative dictionary must
//!   be fit on data strictly *before* the shard whose null sup is being
//!   evaluated) is enforced by the *shape* of [`PyAtomBirthGate::absorb_shard`]:
//!   the gate only ever consumes the two pre-computed per-shard log-likelihoods
//!   `(alternative_prefit, null_sup)` — it can never see, and so can never
//!   peek at, the current shard's refit. The class never refits anything; the
//!   caller hands in the previous dictionary's likelihood and the gate folds it
//!   in. This is the explicit research instrument the SAE structure search and
//!   any user-level atom-existence test route through.
//! * **#939 is an explicit LR instrument, not an auto-magic Wald rewrite.** The
//!   Lawley factor `c = E[W]/d` corrects the *likelihood-ratio* statistic; the
//!   summary-table smooth term test reports a *Wald* χ², a different statistic.
//!   Silently dividing a Wald statistic by an LR Bartlett factor would be
//!   unprincipled, so the correction is exposed as a clean explicit call that
//!   takes the tested block's design, family/η, optional penalty, reference
//!   d.f., and the observed LR statistic, and returns the factor plus the
//!   corrected statistic and p-value. (See #939 follow-up issue for why the
//!   summary path cannot auto-apply it without a per-term LR refit.)

use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use gam::inference::lawley::{RowExpectedJets, RowKappas, lawley_lr_bartlett_factor};
use gam::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};
use gam::inference::structure_evidence::{
    AtomBirthGate, GateVerdict, e_benjamini_hochberg, log_e_from_p_calibrator,
    split_likelihood_log_e_value,
};

use crate::py_value_error;

// ───────────────────────────────────────────────────────────────────────────
// #984 — anytime-valid structure discovery
// ───────────────────────────────────────────────────────────────────────────

/// An anytime-valid atom-birth gate (issue #984): a universal-inference
/// (split-likelihood-ratio) e-process deciding "does atom K+1 exist?" in the
/// boundary/Davies regime where the χ² gate is broken. Resumable across corpus
/// shards and immune to optional stopping (Ville's inequality).
///
/// Predictability is enforced by construction: [`absorb_shard`] takes only the
/// two pre-computed per-shard log-likelihoods, so the gate never has the
/// opportunity to peek at the current shard's refit. The caller's contract is
/// that `alternative_prefit_loglik` is the (K+1)-atom dictionary fit on shards
/// **before** this one, evaluated on this shard, and `null_sup_loglik` is the
/// honest K-atom null refit on this shard.
#[pyclass(name = "AtomBirthGate", module = "gam_pyffi._rust")]
pub(crate) struct PyAtomBirthGate {
    gate: AtomBirthGate,
}

#[pymethods]
impl PyAtomBirthGate {
    /// Open a gate at significance level `alpha` (in `(0, 1)`); the level is
    /// fixed at construction so the verdict can never be α-shopped after seeing
    /// the evidence.
    #[new]
    fn new(alpha: f64) -> PyResult<Self> {
        let gate = AtomBirthGate::new(alpha).map_err(py_value_error)?;
        Ok(Self { gate })
    }

    /// The level the certificate is claimed at.
    #[getter]
    fn alpha(&self) -> f64 {
        self.gate.alpha()
    }

    /// Absorb one shard's split-likelihood ratio. `alternative_prefit_loglik`
    /// is the eval-fold log-likelihood of this shard under the alternative
    /// dictionary fit on PRIOR shards only; `null_sup_loglik` is the honest
    /// constrained null sup (the K-atom refit) on this shard. The per-shard
    /// log e-value `alternative_prefit_loglik − null_sup_loglik` is folded into
    /// the running e-process.
    fn absorb_shard(&mut self, alternative_prefit_loglik: f64, null_sup_loglik: f64) {
        self.gate
            .absorb_shard(alternative_prefit_loglik, null_sup_loglik);
    }

    /// `True` once the running supremum of the e-process has crossed `1/alpha`
    /// — the atom is proven to exist with type-I error ≤ α, permanently (the
    /// crossing is irreversible under Ville).
    fn certified(&self) -> bool {
        matches!(self.gate.verdict(), GateVerdict::Certified { .. })
    }

    /// The current verdict: `{"verdict": "certified"|"contested", "log_e",
    /// "e_value", "alpha"}`. A contested gate has not *disproven* the atom —
    /// it has failed to prove it; its `log_e` is the value the dictionary-level
    /// e-BH certificate consumes.
    fn verdict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        let (label, log_e) = match self.gate.verdict() {
            GateVerdict::Certified { log_e } => ("certified", log_e),
            GateVerdict::Contested { log_e } => ("contested", log_e),
        };
        out.set_item("verdict", label)?;
        out.set_item("log_e", log_e)?;
        out.set_item("e_value", log_e.exp())?;
        out.set_item("alpha", self.gate.alpha())?;
        Ok(out)
    }

    /// The current log e-value — hand this (one per claimed atom) to
    /// [`e_bh_dictionary_certificate`] for the FDR-controlled dictionary list.
    fn log_e_value(&self) -> f64 {
        match self.gate.verdict() {
            GateVerdict::Certified { log_e } | GateVerdict::Contested { log_e } => log_e,
        }
    }
}

/// One universal-inference (split-likelihood-ratio) log e-value:
/// `log E = ℓ_alt(D₀) − sup_{H0} ℓ(D₀)`, finite-sample valid with NO regularity
/// conditions (issue #984). `log_lik_alternative_on_eval` is the eval-fold
/// log-likelihood under the alternative fit on the estimation fold;
/// `log_lik_null_sup_on_eval` is the supremum of the eval-fold log-likelihood
/// over the null model class. `E_{H0}[exp(log E)] ≤ 1` exactly.
#[pyfunction]
pub(crate) fn split_likelihood_log_e(
    log_lik_alternative_on_eval: f64,
    log_lik_null_sup_on_eval: f64,
) -> f64 {
    split_likelihood_log_e_value(log_lik_alternative_on_eval, log_lik_null_sup_on_eval)
}

/// e-BH dictionary certificate (Wang–Ramdas, issue #984): FDR control over the
/// claimed structures (one log e-value per claimed atom/edge) under ARBITRARY
/// dependence — exactly the regime p-value BH cannot legally handle (atoms
/// sharing every token violate PRDS). Returns the sorted indices of the
/// CONFIRMED claims with FDR ≤ `alpha`.
#[pyfunction]
pub(crate) fn e_bh_dictionary_certificate(log_e_values: Vec<f64>, alpha: f64) -> Vec<usize> {
    e_benjamini_hochberg(&log_e_values, alpha)
}

/// Calibrate a p-value into a (conservative, valid) log e-value via the
/// `e = 1/p̂` calibrator family lower bound (issue #984): lets a p-value-only
/// claim join the e-BH dictionary certificate. `p_value` must be in `(0, 1]`.
#[pyfunction]
pub(crate) fn log_e_from_p_value(p_value: f64) -> PyResult<f64> {
    log_e_from_p_calibrator(p_value).map_err(py_value_error)
}

// ───────────────────────────────────────────────────────────────────────────
// #939 — Lawley likelihood-ratio Bartlett correction
// ───────────────────────────────────────────────────────────────────────────

/// Build the per-row expected cumulants for a one-predictor-channel GLM family
/// at linear predictor `eta` (canonical closed-form jets; issue #939).
fn row_kappas_for_family(
    family: &str,
    eta: f64,
    dispersion: f64,
) -> PyResult<RowKappas> {
    let jets: RowExpectedJets = match family {
        "gaussian" => RowExpectedJets::gaussian_identity(dispersion),
        "poisson" => RowExpectedJets::poisson_log(eta),
        "binomial" => RowExpectedJets::binomial_logit(eta),
        "gamma" => RowExpectedJets::gamma_log(eta, dispersion),
        other => {
            return Err(py_value_error(format!(
                "lawley_bartlett: unknown family {other:?}; expected one of \
                 \"gaussian\" (identity), \"poisson\" (log), \"binomial\" (logit), \
                 \"gamma\" (log)"
            )));
        }
    };
    jets.kappas().map_err(py_value_error)
}

/// Lawley likelihood-ratio Bartlett correction for a smooth/parametric block
/// (issue #939). The Lawley factor `c = E[W]/d = 1 + (ε_k − ε_{k−q})/d` makes
/// the `χ²_d` reference of the LR statistic `W` second-order accurate
/// (`O(n⁻²)` size error instead of `O(n⁻¹)`).
///
/// Inputs:
/// * `design` — the `n × k` model design, the tested block being
///   `design[:, tested_start:tested_end]`.
/// * `family`, `eta` — the GLM family (`"gaussian"`/`"poisson"`/`"binomial"`/
///   `"gamma"`) and the per-row linear predictor `η` at the NULL fit (length
///   `n`); Lawley's ε is an expectation evaluated at the null.
/// * `tested_start`, `tested_end` — the column range under test (`H0`: those
///   coefficients are zero).
/// * `ref_df` — the LR reference degrees of freedom `d`.
/// * `penalty` — optional `k × k` quadratic penalty `S_λ` folded into the
///   information (valid for nulls with `S_λ β₀ = 0`).
/// * `dispersion` — the family dispersion φ (Gaussian σ², Gamma φ; 1 for
///   Poisson/Binomial).
/// * `prior_weights` — optional per-row weights (e.g. binomial trial counts).
/// * `lr_statistic` — optional observed LR statistic to correct; when given,
///   the result also carries `corrected_statistic = lr_statistic / c` and the
///   corrected `χ²_d` `p_value`.
///
/// Returns `{"bartlett_factor", "mean_shift", "ref_df", ["corrected_statistic",
/// "p_value_corrected", "p_value_uncorrected"]}`.
#[pyfunction]
#[pyo3(signature = (
    design, family, eta, tested_start, tested_end, ref_df,
    penalty = None, dispersion = 1.0, prior_weights = None, lr_statistic = None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lawley_bartlett_factor<'py>(
    py: Python<'py>,
    design: numpy::PyReadonlyArray2<'py, f64>,
    family: &str,
    eta: numpy::PyReadonlyArray1<'py, f64>,
    tested_start: usize,
    tested_end: usize,
    ref_df: f64,
    penalty: Option<numpy::PyReadonlyArray2<'py, f64>>,
    dispersion: f64,
    prior_weights: Option<numpy::PyReadonlyArray1<'py, f64>>,
    lr_statistic: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = design.as_array();
    let eta_view = eta.as_array();
    let n = x.nrows();
    if eta_view.len() != n {
        return Err(py_value_error(format!(
            "lawley_bartlett: eta has {} entries for {n} design rows",
            eta_view.len()
        )));
    }
    let weights: Option<Array1<f64>> = match prior_weights {
        Some(w) => {
            let wv = w.as_array();
            if wv.len() != n {
                return Err(py_value_error(format!(
                    "lawley_bartlett: prior_weights has {} entries for {n} design rows",
                    wv.len()
                )));
            }
            Some(wv.to_owned())
        }
        None => None,
    };
    let mut kappas: Vec<RowKappas> = Vec::with_capacity(n);
    for i in 0..n {
        let mut k = row_kappas_for_family(family, eta_view[i], dispersion)?;
        if let Some(w) = weights.as_ref() {
            k = k.weighted(w[i]);
        }
        kappas.push(k);
    }
    let penalty_owned = penalty.map(|p| p.as_array().to_owned());
    let factor = lawley_lr_bartlett_factor(
        x,
        &kappas,
        penalty_owned.as_ref().map(|p| p.view()),
        tested_start..tested_end,
        ref_df,
    )
    .map_err(py_value_error)?;

    let out = PyDict::new(py);
    out.set_item("bartlett_factor", factor)?;
    // c = 1 + Δε/d  ⟹  Δε = (c − 1)·d.
    out.set_item("mean_shift", (factor - 1.0) * ref_df)?;
    out.set_item("ref_df", ref_df)?;
    if let Some(stat) = lr_statistic {
        if !(stat.is_finite() && stat >= 0.0) {
            return Err(py_value_error(format!(
                "lawley_bartlett: lr_statistic must be finite and non-negative; got {stat}"
            )));
        }
        let corrected = stat / factor;
        let dist = ChiSquared::new(ref_df)
            .map_err(|e| py_value_error(format!("lawley_bartlett: χ²_{ref_df} invalid: {e}")))?;
        out.set_item("corrected_statistic", corrected)?;
        out.set_item("p_value_corrected", (1.0 - dist.cdf(corrected)).clamp(0.0, 1.0))?;
        out.set_item("p_value_uncorrected", (1.0 - dist.cdf(stat)).clamp(0.0, 1.0))?;
    }
    Ok(out)
}

// ───────────────────────────────────────────────────────────────────────────
// #1055 — Riesz-representer debiased functional
// ───────────────────────────────────────────────────────────────────────────

/// Resolve a named estimand + its design payload into the linear functional
/// gradient `g = dθ/dβ` consumed by the Riesz representer.
///
/// `target` selects one of the closed-form Layer-1 functionals of
/// [`SmoothFunctional`]:
/// * `"point"` — `m(x₀)`: needs `design_row` (the prediction row at `x₀`).
/// * `"contrast"` — `m(x_a) − m(x_b)`: needs `design_row` (= `x_a`) and
///   `design_row_b` (= `x_b`).
/// * `"average_derivative"` — `mean_i w_i · ∂m(x_i)/∂x_j`: needs
///   `design_matrix` (the derivative-basis rows) and optional `weights`.
/// * `"average_value"` — `mean_i w_i · m(x_i)`: needs `design_matrix` (the
///   value-basis rows) and optional `weights`.
/// * `"linear"` — a caller-supplied functional gradient directly in
///   `design_row`.
fn riesz_functional_gradient(
    target: &str,
    design_row: Option<&Array1<f64>>,
    design_row_b: Option<&Array1<f64>>,
    design_matrix: Option<&ndarray::Array2<f64>>,
    weights: Option<&Array1<f64>>,
) -> PyResult<Array1<f64>> {
    let need_row = |name: &str, value: Option<&Array1<f64>>| {
        value.ok_or_else(|| {
            py_value_error(format!(
                "debiased_functional: target {target:?} requires `{name}`"
            ))
        })
    };
    let functional = match target {
        "point" => SmoothFunctional::PointEvaluation {
            design_row: need_row("design_row", design_row)?.view(),
        },
        "linear" => SmoothFunctional::Linear {
            gradient: need_row("design_row", design_row)?.view(),
        },
        "contrast" => SmoothFunctional::Contrast {
            design_row_a: need_row("design_row", design_row)?.view(),
            design_row_b: need_row("design_row_b", design_row_b)?.view(),
        },
        "average_derivative" => {
            let rows = design_matrix.ok_or_else(|| {
                py_value_error(
                    "debiased_functional: target \"average_derivative\" requires `design_matrix`"
                        .to_string(),
                )
            })?;
            SmoothFunctional::AverageDerivative {
                derivative_design: rows.view(),
                weights: weights.map(|w| w.view()),
            }
        }
        "average_value" => {
            let rows = design_matrix.ok_or_else(|| {
                py_value_error(
                    "debiased_functional: target \"average_value\" requires `design_matrix`"
                        .to_string(),
                )
            })?;
            SmoothFunctional::AverageValue {
                value_design: rows.view(),
                weights: weights.map(|w| w.view()),
            }
        }
        other => {
            return Err(py_value_error(format!(
                "debiased_functional: unknown target {other:?}; expected one of \
                 \"point\", \"contrast\", \"average_derivative\", \"average_value\", \"linear\""
            )));
        }
    };
    functional
        .gradient()
        .map_err(|err| py_value_error(format!("debiased_functional: {err}")))
}

/// Riesz-representer debiased / Neyman-orthogonal estimate of a smooth
/// functional of a fitted model (issue #1055). This surfaces the previously
/// unreachable `src/inference/riesz.rs` engine: the orthogonal correction is
/// always on (it strictly improves coverage under regularization), so there is
/// no flag.
///
/// The debiasing solves the Riesz representer `α = H⁻¹ g` against the *penalized*
/// fitted Hessian `H`, returns the penalty-debiased one-step estimate
/// `θ̂ = gᵀβ̂ + αᵀ(S β̂)`, and the influence-function plug-in standard error
/// `SE = sd(ψ)/√n` with `ψ_i = −n·s_iᵀα` (own-observation removed analytically
/// when `leverage` is supplied).
///
/// Inputs (all in the fitted coefficient basis):
/// * `beta` — fitted coefficients `β̂` (length `p`).
/// * `penalized_hessian` — the `p × p` penalized fitted Hessian `H` (SPD).
/// * `row_scores` — per-row score contributions `s_i = ∂nll_i/∂β` (`n × p`).
/// * `penalty_beta` — penalty gradient `S β̂` (length `p`).
/// * `target` — the named estimand (see [`riesz_functional_gradient`]).
/// * `design_row`, `design_row_b`, `design_matrix`, `weights` — the design
///   payload the chosen `target` consumes.
/// * `leverage` — optional ALO leverages `h_ii` for exact own-observation
///   removal in the influence values.
///
/// Returns `{"theta_plugin", "theta_debiased", "se", "penalty_bias", "ci_lower",
/// "ci_upper"}` (95% normal CI on the debiased estimate).
#[pyfunction]
#[pyo3(signature = (
    beta, penalized_hessian, row_scores, penalty_beta, target,
    design_row = None, design_row_b = None, design_matrix = None,
    weights = None, leverage = None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn debiased_functional<'py>(
    py: Python<'py>,
    beta: numpy::PyReadonlyArray1<'py, f64>,
    penalized_hessian: numpy::PyReadonlyArray2<'py, f64>,
    row_scores: numpy::PyReadonlyArray2<'py, f64>,
    penalty_beta: numpy::PyReadonlyArray1<'py, f64>,
    target: &str,
    design_row: Option<numpy::PyReadonlyArray1<'py, f64>>,
    design_row_b: Option<numpy::PyReadonlyArray1<'py, f64>>,
    design_matrix: Option<numpy::PyReadonlyArray2<'py, f64>>,
    weights: Option<numpy::PyReadonlyArray1<'py, f64>>,
    leverage: Option<numpy::PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let beta = beta.as_array().to_owned();
    let hessian = penalized_hessian.as_array().to_owned();
    let scores = row_scores.as_array().to_owned();
    let penalty_beta = penalty_beta.as_array().to_owned();
    let design_row = design_row.map(|a| a.as_array().to_owned());
    let design_row_b = design_row_b.map(|a| a.as_array().to_owned());
    let design_matrix = design_matrix.map(|a| a.as_array().to_owned());
    let weights = weights.map(|a| a.as_array().to_owned());
    let leverage = leverage.map(|a| a.as_array().to_owned());

    let gradient = riesz_functional_gradient(
        target,
        design_row.as_ref(),
        design_row_b.as_ref(),
        design_matrix.as_ref(),
        weights.as_ref(),
    )?;

    let input = RieszInput {
        beta: beta.view(),
        functional_gradient: gradient.view(),
        row_scores: scores.view(),
        penalty_beta: penalty_beta.view(),
        leverage: leverage.as_ref().map(|l| l.view()),
    };
    let report = debias_with_dense_hessian(&input, hessian.view())
        .map_err(|err| py_value_error(format!("debiased_functional: {err}")))?;

    let half_width = 1.959_963_984_540_054 * report.se;
    let out = PyDict::new(py);
    out.set_item("theta_plugin", report.theta_plugin)?;
    out.set_item("theta_debiased", report.theta_onestep)?;
    out.set_item("se", report.se)?;
    out.set_item("penalty_bias", report.penalty_bias)?;
    out.set_item("ci_lower", report.theta_onestep - half_width)?;
    out.set_item("ci_upper", report.theta_onestep + half_width)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn split_lr_log_e_is_the_likelihood_difference() {
        // log E = ℓ_alt − sup ℓ_null; a calibrated null (alt = null sup) is 0.
        assert!((split_likelihood_log_e(-10.0, -10.0)).abs() < 1e-15);
        assert!((split_likelihood_log_e(-8.0, -10.0) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn e_bh_confirms_strong_evidence_and_drops_weak() {
        // One overwhelming claim (log e huge) clears the e-BH threshold; a
        // cluster of near-1 e-values (log e ≈ 0) does not.
        let logs = vec![20.0_f64.ln() * 5.0, 0.01, -0.2, 0.0];
        let confirmed = e_bh_dictionary_certificate(logs, 0.05);
        assert_eq!(confirmed, vec![0]);
    }

    #[test]
    fn gate_certifies_only_after_evidence_crosses_one_over_alpha() {
        // alpha = 0.05 ⟹ threshold log(1/alpha) ≈ 2.996. Two shards each with
        // log e = 2.0 compound to 4.0 > 2.996 ⟹ certified.
        let mut gate = PyAtomBirthGate::new(0.05).expect("gate");
        assert!(!gate.certified());
        gate.absorb_shard(-8.0, -10.0); // log e = 2.0
        assert!(!gate.certified());
        gate.absorb_shard(-8.0, -10.0); // cumulative 4.0
        assert!(gate.certified());
        assert!((gate.log_e_value() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn debiased_functional_seam_returns_finite_estimate_and_se_on_a_fit() {
        // A genuine penalized least-squares fit (the same regime the engine's
        // own oracle test uses), driven through the #1055 pyffi seam helper
        // `riesz_functional_gradient` + `debias_with_dense_hessian`. Asserts the
        // surfaced debiased path yields a finite estimate and SE, and that the
        // weighted-average-derivative target recovers the truth better than the
        // (oversmoothed) plug-in.
        let n = 80usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut derivative_design = Array2::<f64>::zeros((n, p));
        let mut weights = Array1::<f64>::zeros(n);
        let beta_truth = ndarray::array![0.2, -0.4, 2.5];
        for row in 0..n {
            let z = row as f64 / (n - 1) as f64;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = z;
            x[[row, 2]] = z * z;
            derivative_design[[row, 1]] = 1.0;
            derivative_design[[row, 2]] = 2.0 * z;
            weights[row] = 1.0 + 4.0 * z;
        }
        let y = x.dot(&beta_truth);
        let mut penalty = Array2::<f64>::zeros((p, p));
        penalty[[2, 2]] = 0.1;
        let h = &x.t().dot(&x) + &penalty;
        let rhs = x.t().dot(&y);
        // Solve H β = rhs via the engine-side Cholesky path used by the seam.
        let factor = {
            use gam::faer_ndarray::FaerCholesky;
            h.cholesky(faer::Side::Lower).expect("SPD")
        };
        let sensitivity = gam::solver::sensitivity::FitSensitivity::from_faer_cholesky(&factor, p);
        let beta_hat = sensitivity.apply(&rhs);
        let mu = x.dot(&beta_hat);
        let mut row_scores = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let residual = mu[row] - y[row];
            for col in 0..p {
                row_scores[[row, col]] = x[[row, col]] * residual;
            }
        }
        let penalty_beta = penalty.dot(&beta_hat);

        let gradient = riesz_functional_gradient(
            "average_derivative",
            None,
            None,
            Some(&derivative_design),
            Some(&weights),
        )
        .expect("seam builds average-derivative gradient");

        let input = RieszInput {
            beta: beta_hat.view(),
            functional_gradient: gradient.view(),
            row_scores: row_scores.view(),
            penalty_beta: penalty_beta.view(),
            leverage: None,
        };
        let report = debias_with_dense_hessian(&input, h.view()).expect("debiased report");

        assert!(report.theta_onestep.is_finite(), "debiased estimate finite");
        assert!(report.se.is_finite() && report.se > 0.0, "SE finite & positive");

        let truth = gradient.dot(&beta_truth);
        let plugin_bias = (report.theta_plugin - truth).abs();
        let debiased_bias = (report.theta_onestep - truth).abs();
        assert!(
            debiased_bias <= plugin_bias + 1e-12,
            "orthogonal correction must not worsen bias: plugin={plugin_bias:.3e}, debiased={debiased_bias:.3e}"
        );

        // The point-evaluation and contrast targets must also yield finite gradients.
        let row0 = x.row(0).to_owned();
        let row1 = x.row(1).to_owned();
        let g_point = riesz_functional_gradient("point", Some(&row0), None, None, None)
            .expect("point gradient");
        assert_eq!(g_point.len(), p);
        let g_contrast =
            riesz_functional_gradient("contrast", Some(&row0), Some(&row1), None, None)
                .expect("contrast gradient");
        assert_eq!(g_contrast.len(), p);

        // Unknown target is a clean error, not a panic.
        assert!(riesz_functional_gradient("bogus", None, None, None, None).is_err());
    }

    #[test]
    fn lawley_factor_recovers_exponential_one_over_six_n() {
        // Exponential (Gamma-log, φ=1), intercept-only, tested = the intercept:
        // the null model has no parameters so ε_0 = 0 and the factor is
        // c = 1 + ε_n/1 = 1 + 1/(6n) (the module's certified fixture).
        for &n in &[8usize, 32] {
            let eta = 0.4;
            let kappas =
                vec![RowExpectedJets::gamma_log(eta, 1.0).kappas().expect("kappas"); n];
            let x = Array2::<f64>::ones((n, 1));
            let factor = lawley_lr_bartlett_factor(x.view(), &kappas, None, 0..1, 1.0)
                .expect("factor");
            let expected = 1.0 + 1.0 / (6.0 * n as f64);
            assert!(
                (factor - expected).abs() < 1e-10,
                "n={n}: factor={factor} vs 1+1/(6n)={expected}"
            );
        }
    }
}

/// Register the inference-instrument `#[pyfunction]`s and classes on the
/// extension module. Kept here (rather than inline in `lib.rs`) so the wiring
/// is one line at the call site.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAtomBirthGate>()?;
    module.add_function(wrap_pyfunction!(split_likelihood_log_e, module)?)?;
    module.add_function(wrap_pyfunction!(e_bh_dictionary_certificate, module)?)?;
    module.add_function(wrap_pyfunction!(log_e_from_p_value, module)?)?;
    module.add_function(wrap_pyfunction!(lawley_bartlett_factor, module)?)?;
    module.add_function(wrap_pyfunction!(debiased_functional, module)?)?;
    Ok(())
}
