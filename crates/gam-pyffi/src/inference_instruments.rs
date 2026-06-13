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
#[pyclass(name = "AtomBirthGate", module = "gam._rust")]
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

/// Register the inference-instrument `#[pyfunction]`s and classes on the
/// extension module. Kept here (rather than inline in `lib.rs`) so the wiring
/// is one line at the call site.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAtomBirthGate>()?;
    module.add_function(wrap_pyfunction!(split_likelihood_log_e, module)?)?;
    module.add_function(wrap_pyfunction!(e_bh_dictionary_certificate, module)?)?;
    module.add_function(wrap_pyfunction!(log_e_from_p_value, module)?)?;
    module.add_function(wrap_pyfunction!(lawley_bartlett_factor, module)?)?;
    Ok(())
}
