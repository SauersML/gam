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

use ndarray::{Array1, Array2, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use gam::inference::full_conformal::{CanonicalGlmFamily, GlmHomotopyFullConformal};
use gam::inference::lawley::{
    RhoPenaltyComponent, RowExpectedJets, RowKappas, lawley_lr_bartlett_factor,
    lawley_lr_mean_shift_with_rho_variation,
};
use gam::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};
use gam::inference::structure_evidence::{
    AtomBirthGate, CandidateProbe, GateVerdict, ProbePlan, e_benjamini_hochberg,
    expected_resolution_budget as core_expected_resolution_budget, log_e_from_p_calibrator,
    plan_probe_for_contested_claim as core_plan_probe_for_contested_claim,
    select_probe_by_expected_evidence as core_select_probe_by_expected_evidence,
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
// #1109 — KL-optimal steering-probe design
// ───────────────────────────────────────────────────────────────────────────

fn candidate_probes_from_arrays(
    function_name: &str,
    delta: ArrayView2<'_, f64>,
    predicted_mean_null: ArrayView2<'_, f64>,
    predicted_mean_alt: ArrayView2<'_, f64>,
    fisher: ArrayView2<'_, f64>,
) -> PyResult<(Vec<CandidateProbe>, Array2<f64>)> {
    let (n_probes, p_out) = delta.dim();
    if p_out == 0 {
        return Err(py_value_error(format!(
            "{function_name}: candidate arrays must have at least one output column"
        )));
    }
    if predicted_mean_null.dim() != (n_probes, p_out) {
        return Err(py_value_error(format!(
            "{function_name}: predicted_mean_null shape {:?} must match delta shape {:?}",
            predicted_mean_null.dim(),
            delta.dim()
        )));
    }
    if predicted_mean_alt.dim() != (n_probes, p_out) {
        return Err(py_value_error(format!(
            "{function_name}: predicted_mean_alt shape {:?} must match delta shape {:?}",
            predicted_mean_alt.dim(),
            delta.dim()
        )));
    }
    if fisher.dim() != (p_out, p_out) {
        return Err(py_value_error(format!(
            "{function_name}: fisher must be square ({p_out}, {p_out}) for output dimension {p_out}; got {:?}",
            fisher.dim()
        )));
    }
    for (label, finite) in [
        ("delta", delta.iter().all(|v| v.is_finite())),
        (
            "predicted_mean_null",
            predicted_mean_null.iter().all(|v| v.is_finite()),
        ),
        (
            "predicted_mean_alt",
            predicted_mean_alt.iter().all(|v| v.is_finite()),
        ),
        ("fisher", fisher.iter().all(|v| v.is_finite())),
    ] {
        if !finite {
            return Err(py_value_error(format!(
                "{function_name}: {label} contains non-finite values"
            )));
        }
    }

    let mut probes = Vec::with_capacity(n_probes);
    for idx in 0..n_probes {
        probes.push(CandidateProbe {
            delta: delta.row(idx).to_owned(),
            predicted_mean_null: predicted_mean_null.row(idx).to_owned(),
            predicted_mean_alt: predicted_mean_alt.row(idx).to_owned(),
        });
    }
    Ok((probes, fisher.to_owned()))
}

fn probe_plan_to_pydict<'py>(
    py: Python<'py>,
    plan: ProbePlan,
    probes: &[CandidateProbe],
) -> PyResult<Bound<'py, PyDict>> {
    let probe = &probes[plan.probe];
    let response_diff = &probe.predicted_mean_alt - &probe.predicted_mean_null;
    let out = PyDict::new(py);
    out.set_item("probe", plan.probe)?;
    out.set_item("expected_log_growth", plan.expected_log_growth)?;
    out.set_item("budget_from_scratch", plan.budget_from_scratch)?;
    out.set_item("budget_remaining", plan.budget_remaining)?;
    out.set_item("delta", probe.delta.to_vec())?;
    out.set_item("predicted_mean_null", probe.predicted_mean_null.to_vec())?;
    out.set_item("predicted_mean_alt", probe.predicted_mean_alt.to_vec())?;
    out.set_item("predicted_mean_diff", response_diff.to_vec())?;
    Ok(out)
}

/// Select the steering-probe candidate whose two structural hypotheses disagree
/// most in the output-Fisher metric (issue #1109). Inputs are row-aligned
/// candidate arrays: `delta[i]` is the steering displacement, and
/// `predicted_mean_null[i]` / `predicted_mean_alt[i]` are the two hypotheses'
/// predicted output-mean responses to that same probe. The selected score is
/// `0.5 * (mu_alt - mu_null)^T fisher (mu_alt - mu_null)` in nats per
/// observation. Returns `None` when no candidate discriminates.
#[pyfunction]
pub(crate) fn select_probe_by_expected_evidence<'py>(
    py: Python<'py>,
    delta: numpy::PyReadonlyArray2<'py, f64>,
    predicted_mean_null: numpy::PyReadonlyArray2<'py, f64>,
    predicted_mean_alt: numpy::PyReadonlyArray2<'py, f64>,
    fisher: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let (probes, fisher) = candidate_probes_from_arrays(
        "select_probe_by_expected_evidence",
        delta.as_array(),
        predicted_mean_null.as_array(),
        predicted_mean_alt.as_array(),
        fisher.as_array(),
    )?;
    let Some((idx, expected_log_growth)) = core_select_probe_by_expected_evidence(&probes, &fisher)
    else {
        return Ok(None);
    };
    let probe = &probes[idx];
    let response_diff = &probe.predicted_mean_alt - &probe.predicted_mean_null;
    let out = PyDict::new(py);
    out.set_item("probe", idx)?;
    out.set_item("expected_log_growth", expected_log_growth)?;
    out.set_item("delta", probe.delta.to_vec())?;
    out.set_item("predicted_mean_null", probe.predicted_mean_null.to_vec())?;
    out.set_item("predicted_mean_alt", probe.predicted_mean_alt.to_vec())?;
    out.set_item("predicted_mean_diff", response_diff.to_vec())?;
    Ok(Some(out))
}

/// Expected observations needed for a probe with per-observation expected
/// evidence growth `growth_nats_per_obs` to cross the Ville threshold `1/alpha`.
#[pyfunction]
pub(crate) fn expected_resolution_budget(alpha: f64, growth_nats_per_obs: f64) -> Option<f64> {
    core_expected_resolution_budget(alpha, growth_nats_per_obs)
}

/// Plan the next steering probe for a contested structural claim (issue #1109):
/// choose the KL-optimal candidate, report its expected evidence growth, and
/// discount the remaining observation budget by the claim's current log
/// e-evidence. Returns `None` when no candidate discriminates.
#[pyfunction]
#[pyo3(signature = (
    delta,
    predicted_mean_null,
    predicted_mean_alt,
    fisher,
    alpha,
    current_log_e = 0.0
))]
pub(crate) fn plan_probe_for_contested_claim<'py>(
    py: Python<'py>,
    delta: numpy::PyReadonlyArray2<'py, f64>,
    predicted_mean_null: numpy::PyReadonlyArray2<'py, f64>,
    predicted_mean_alt: numpy::PyReadonlyArray2<'py, f64>,
    fisher: numpy::PyReadonlyArray2<'py, f64>,
    alpha: f64,
    current_log_e: f64,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let (probes, fisher) = candidate_probes_from_arrays(
        "plan_probe_for_contested_claim",
        delta.as_array(),
        predicted_mean_null.as_array(),
        predicted_mean_alt.as_array(),
        fisher.as_array(),
    )?;
    let Some(plan) = core_plan_probe_for_contested_claim(&probes, &fisher, alpha, current_log_e)
    else {
        return Ok(None);
    };
    probe_plan_to_pydict(py, plan, &probes).map(Some)
}

// ───────────────────────────────────────────────────────────────────────────
// #939 — Lawley likelihood-ratio Bartlett correction
// ───────────────────────────────────────────────────────────────────────────

/// Build the per-row expected cumulants for a one-predictor-channel GLM family
/// at linear predictor `eta` (canonical closed-form jets; issue #939).
fn row_kappas_for_family(family: &str, eta: f64, dispersion: f64) -> PyResult<RowKappas> {
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
        out.set_item(
            "p_value_corrected",
            (1.0 - dist.cdf(corrected)).clamp(0.0, 1.0),
        )?;
        out.set_item(
            "p_value_uncorrected",
            (1.0 - dist.cdf(stat)).clamp(0.0, 1.0),
        )?;
    }
    Ok(out)
}

/// Estimated-λ Lawley LR Bartlett correction (issue #939 deliverable 2, the
/// genuinely-new penalized theory piece): the ρ̂-**sampling-variation**
/// contribution to the penalized-null Bartlett factor.
///
/// The plain [`lawley_bartlett_factor`] folds the penalty `S_λ` into the
/// information at the **fitted** smoothing parameter — it is the *conditional*
/// mean shift `Δε(ρ̂) = E[W | λ]`. When λ is **estimated**, ρ̂ = log λ̂ carries
/// its own sampling variation and the null mean of the LR statistic picks up the
/// second-order delta-method term
///
/// ```text
/// E[W(ρ̂)] = Δε(ρ₀) + ½ Σ_{b,b'} (∂²Δε/∂ρ_b ∂ρ_{b'}) · Cov(ρ̂_b, ρ̂_{b'}) + O(·),
/// ```
///
/// assembled exactly by
/// [`gam::inference::lawley::lawley_lr_mean_shift_with_rho_variation`] from the
/// curvature of the deterministic conditional shift in the log-smoothing
/// parameters and the inverse REML/LAML **outer Hessian** `Cov(ρ̂)` (the #740
/// quantity). Returns the **total** estimated-λ factor `c = 1 + Δε(ρ̂)/d`
/// alongside the conditional one, so the caller can read the size correction
/// attributable specifically to ρ̂-variation as `c − c_conditional`.
///
/// Inputs mirror [`lawley_bartlett_factor`] plus:
/// * `penalty` — the **total** fitted `S_λ = Σ_b λ_b S_b^unit` (`k × k`), the
///   conditional anchor (required here, unlike the conditional entry point).
/// * `components` — a list of `k × k` component penalties `S_b` at their fitted
///   scale (`λ_b · S_b^unit`); `∂S_λ/∂ρ_b = S_b`. One per smoothing parameter.
/// * `rho_cov` — the `m × m` sampling covariance `Cov(ρ̂)` of the `m`
///   log-smoothing parameters (the regularized inverse REML outer Hessian).
///
/// Returns `{"bartlett_factor", "bartlett_factor_conditional",
/// "rho_variation_shift", "mean_shift", "mean_shift_conditional", "ref_df",
/// ["corrected_statistic", "p_value_corrected", "p_value_uncorrected"]}`.
#[pyfunction]
#[pyo3(signature = (
    design, family, eta, tested_start, tested_end, ref_df,
    penalty, components, rho_cov,
    dispersion = 1.0, prior_weights = None, lr_statistic = None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lawley_bartlett_factor_estimated_lambda<'py>(
    py: Python<'py>,
    design: numpy::PyReadonlyArray2<'py, f64>,
    family: &str,
    eta: numpy::PyReadonlyArray1<'py, f64>,
    tested_start: usize,
    tested_end: usize,
    ref_df: f64,
    penalty: numpy::PyReadonlyArray2<'py, f64>,
    components: Vec<numpy::PyReadonlyArray2<'py, f64>>,
    rho_cov: numpy::PyReadonlyArray2<'py, f64>,
    dispersion: f64,
    prior_weights: Option<numpy::PyReadonlyArray1<'py, f64>>,
    lr_statistic: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if !(ref_df.is_finite() && ref_df > 0.0) {
        return Err(py_value_error(format!(
            "lawley_bartlett_estimated: ref_df must be finite and positive; got {ref_df}"
        )));
    }
    let x = design.as_array();
    let eta_view = eta.as_array();
    let n = x.nrows();
    if eta_view.len() != n {
        return Err(py_value_error(format!(
            "lawley_bartlett_estimated: eta has {} entries for {n} design rows",
            eta_view.len()
        )));
    }
    let weights: Option<Array1<f64>> = match prior_weights {
        Some(w) => {
            let wv = w.as_array();
            if wv.len() != n {
                return Err(py_value_error(format!(
                    "lawley_bartlett_estimated: prior_weights has {} entries for {n} design rows",
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
    let penalty_owned = penalty.as_array().to_owned();
    let comps: Vec<RhoPenaltyComponent> = components
        .iter()
        .map(|c| RhoPenaltyComponent {
            s_component: c.as_array().to_owned(),
        })
        .collect();
    let rho_cov_owned = rho_cov.as_array().to_owned();

    // Conditional (fixed-λ) factor — the existing deterministic-penalty path.
    let factor_conditional = lawley_lr_bartlett_factor(
        x,
        &kappas,
        Some(penalty_owned.view()),
        tested_start..tested_end,
        ref_df,
    )
    .map_err(py_value_error)?;

    // Total estimated-λ mean shift = conditional + ½·tr(Hᵨᵨ Cov(ρ̂)).
    let total_shift = lawley_lr_mean_shift_with_rho_variation(
        x,
        &kappas,
        penalty_owned.view(),
        tested_start..tested_end,
        &comps,
        rho_cov_owned.view(),
    )
    .map_err(py_value_error)?;
    let mean_w = ref_df + total_shift;
    let factor = gam::inference::higher_order::bartlett_factor_from_mean(mean_w, ref_df)
        .ok_or_else(|| {
            py_value_error(format!(
                "lawley_bartlett_estimated: degenerate mean {mean_w} (Δε(ρ̂) = {total_shift}, d = {ref_df})"
            ))
        })?;
    if !(factor.is_finite() && factor > 0.0) {
        return Err(py_value_error(format!(
            "lawley_bartlett_estimated: degenerate factor {factor}"
        )));
    }

    let out = PyDict::new(py);
    out.set_item("bartlett_factor", factor)?;
    out.set_item("bartlett_factor_conditional", factor_conditional)?;
    out.set_item("mean_shift", total_shift)?;
    // Conditional shift Δε(ρ̂) = (c_cond − 1)·d; the ρ̂-variation increment is the
    // difference the estimated-λ correction adds on top.
    let mean_shift_conditional = (factor_conditional - 1.0) * ref_df;
    out.set_item("mean_shift_conditional", mean_shift_conditional)?;
    out.set_item("rho_variation_shift", total_shift - mean_shift_conditional)?;
    out.set_item("ref_df", ref_df)?;
    if let Some(stat) = lr_statistic {
        if !(stat.is_finite() && stat >= 0.0) {
            return Err(py_value_error(format!(
                "lawley_bartlett_estimated: lr_statistic must be finite and non-negative; got {stat}"
            )));
        }
        let corrected = stat / factor;
        let dist = ChiSquared::new(ref_df).map_err(|e| {
            py_value_error(format!(
                "lawley_bartlett_estimated: χ²_{ref_df} invalid: {e}"
            ))
        })?;
        out.set_item("corrected_statistic", corrected)?;
        out.set_item(
            "p_value_corrected",
            (1.0 - dist.cdf(corrected)).clamp(0.0, 1.0),
        )?;
        out.set_item(
            "p_value_uncorrected",
            (1.0 - dist.cdf(stat)).clamp(0.0, 1.0),
        )?;
    }
    Ok(out)
}

// ───────────────────────────────────────────────────────────────────────────
// #939 deliverable 3 — Skovgaard modified directed root r* for a scalar functional
// ───────────────────────────────────────────────────────────────────────────

/// Skovgaard's modified directed likelihood root `r*` for a **scalar** interest
/// parameter `ψ = cᵀβ` (issue #939, deliverable 3), assembled EXACTLY from the
/// fitted-model matrices — no approximation beyond Skovgaard's own covariance
/// identity for the sample-space derivative.
///
/// This exposes [`gam::inference::skovgaard::scalar_skovgaard_from_matrices`] on
/// the clean `gamfit` surface (the in-tree implementation was certified against
/// the Exponential / logistic-location closed forms but was previously
/// unreachable). Ingredients, all from the fitted penalized GLM:
/// * `contrast` (`c`) — the functional gradient `∂ψ/∂β` (a prediction row for a
///   point-on-curve, a row difference for a contrast, or any linear gradient).
/// * `beta` (`β̂`) — fitted coefficients; `ψ̂ = cᵀβ̂`.
/// * `penalized_hessian` (`Ĥ = X'WX + S_λ`) — the **observed** information.
/// * `fisher_information` (`Iₑ = X'WX`, optional) — the **expected** (Fisher)
///   information; omit for a canonical link, where `î = ĵ` and the curvature
///   factor is `1`.
/// * `row_scores` (`sᵢ = ∂ℓᵢ/∂β`, `n × p`) — per-row scores for the empirical
///   (sandwich) covariance companion.
/// * `lr_statistic` (`W = 2[ℓ(β̂) − ℓ(β̂₀)] ≥ 0`) — the profile LR from the
///   constrained refit at `cᵀβ = θ₀`.
/// * `theta_null` (`θ₀`) — the tested value (default `0`).
///
/// Returns `{"r", "u", "r_star", "p_value_first_order", "p_value_corrected",
/// "u_empirical", "r_star_empirical", "p_value_corrected_empirical", "material"}`.
#[pyfunction]
#[pyo3(signature = (
    contrast, beta, penalized_hessian, row_scores, lr_statistic,
    fisher_information = None, theta_null = 0.0
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn skovgaard_r_star<'py>(
    py: Python<'py>,
    contrast: numpy::PyReadonlyArray1<'py, f64>,
    beta: numpy::PyReadonlyArray1<'py, f64>,
    penalized_hessian: numpy::PyReadonlyArray2<'py, f64>,
    row_scores: numpy::PyReadonlyArray2<'py, f64>,
    lr_statistic: f64,
    fisher_information: Option<numpy::PyReadonlyArray2<'py, f64>>,
    theta_null: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use gam::inference::skovgaard::scalar_skovgaard_from_matrices;

    let c = contrast.as_array();
    let b = beta.as_array();
    let h = penalized_hessian.as_array();
    let s = row_scores.as_array();
    let p = b.len();
    if c.len() != p {
        return Err(py_value_error(format!(
            "skovgaard_r_star: contrast has {} entries for {p} coefficients",
            c.len()
        )));
    }
    if h.nrows() != p || h.ncols() != p {
        return Err(py_value_error(format!(
            "skovgaard_r_star: penalized_hessian is {}×{}, expected {p}×{p}",
            h.nrows(),
            h.ncols()
        )));
    }
    if s.ncols() != p {
        return Err(py_value_error(format!(
            "skovgaard_r_star: row_scores has {} columns, expected {p}",
            s.ncols()
        )));
    }
    if !(lr_statistic.is_finite() && lr_statistic >= 0.0) {
        return Err(py_value_error(format!(
            "skovgaard_r_star: lr_statistic must be finite and non-negative; got {lr_statistic}"
        )));
    }
    // Own the optional Fisher matrix so its view outlives the call.
    let fisher_owned: Option<Array2<f64>> = match fisher_information {
        Some(f) => {
            let fv = f.as_array();
            if fv.nrows() != p || fv.ncols() != p {
                return Err(py_value_error(format!(
                    "skovgaard_r_star: fisher_information is {}×{}, expected {p}×{p}",
                    fv.nrows(),
                    fv.ncols()
                )));
            }
            Some(fv.to_owned())
        }
        None => None,
    };

    let res = scalar_skovgaard_from_matrices(
        c,
        b,
        h,
        fisher_owned.as_ref().map(|f| f.view()),
        s,
        lr_statistic,
        theta_null,
    )
    .ok_or_else(|| {
        py_value_error(
            "skovgaard_r_star: degenerate inputs (non-positive LR/information, \
             singular Hessian, or undefined r*); the first-order root stands"
                .to_string(),
        )
    })?;

    let out = PyDict::new(py);
    out.set_item("r", res.r)?;
    out.set_item("u", res.u)?;
    out.set_item("r_star", res.r_star)?;
    out.set_item("p_value_first_order", res.p_value_first_order)?;
    out.set_item("p_value_corrected", res.p_value_corrected)?;
    out.set_item("u_empirical", res.u_empirical)?;
    out.set_item("r_star_empirical", res.r_star_empirical)?;
    out.set_item(
        "p_value_corrected_empirical",
        res.p_value_corrected_empirical,
    )?;
    out.set_item("material", res.material)?;
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
    // A local `fn` (not a closure) so the returned reference's lifetime is tied
    // explicitly to `value`; a closure capturing `target` cannot express that
    // output-borrows-input lifetime relation and trips a borrow-checker error.
    fn need_row<'a>(
        target: &str,
        name: &str,
        value: Option<&'a Array1<f64>>,
    ) -> PyResult<&'a Array1<f64>> {
        value.ok_or_else(|| {
            py_value_error(format!(
                "debiased_functional: target {target:?} requires `{name}`"
            ))
        })
    }
    let functional = match target {
        "point" => SmoothFunctional::PointEvaluation {
            design_row: need_row(target, "design_row", design_row)?.view(),
        },
        "linear" => SmoothFunctional::Linear {
            gradient: need_row(target, "design_row", design_row)?.view(),
        },
        "contrast" => SmoothFunctional::Contrast {
            design_row_a: need_row(target, "design_row", design_row)?.view(),
            design_row_b: need_row(target, "design_row_b", design_row_b)?.view(),
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

// ───────────────────────────────────────────────────────────────────────────
// #942 — exact full-conformal prediction set for canonical-link GLMs
// ───────────────────────────────────────────────────────────────────────────

/// Resolve the canonical-GLM family name shared with the homotopy engine.
fn canonical_glm_family(name: &str) -> PyResult<CanonicalGlmFamily> {
    match name {
        "bernoulli" | "binomial" | "logit" | "bernoulli_logit" => {
            Ok(CanonicalGlmFamily::BernoulliLogit)
        }
        "poisson" | "poisson_log" => Ok(CanonicalGlmFamily::PoissonLog),
        other => Err(py_value_error(format!(
            "glm_full_conformal: unknown family {other:?}; expected \"bernoulli\" or \"poisson\""
        ))),
    }
}

/// Exact (finite-sample-valid) full-conformal prediction set for a
/// canonical-link GLM (issue #942). This surfaces the previously unreachable
/// certified predictor–corrector engine in `src/inference/full_conformal.rs`:
/// for each candidate response `z` the augmented penalized fit is tracked
/// (or cold-refit when the step certificate refuses), the `n+1` absolute-
/// residual nonconformity scores are ranked, and `z` is retained iff its
/// conformal p-value exceeds `alpha`. The returned set has finite-sample
/// coverage `≥ 1 − alpha` under exchangeability — a guarantee neither split
/// conformal at small `n` nor any mature classification-conformal tool
/// (MAPIE, glmnet, mgcv) provides for a GLM with a coverage certificate.
///
/// Smoothing is FROZEN at the supplied penalty `s_lambda` (the honest ρ-re-
/// selection is the engine's Layer-3 domain), and unit prior weights are
/// required because a reweighted training row is not exchangeable with the
/// test row — the proof would not apply, so the engine refuses rather than
/// silently mis-cover.
///
/// Inputs (all in the fitted coefficient basis):
/// * `design` — training design `X` (`n × p`).
/// * `response` — training response `y` (length `n`); `{0,1}` for Bernoulli,
///   non-negative for Poisson.
/// * `s_lambda` — the `p × p` penalty matrix `Sλ` (`≥ 0`); pass zeros for an
///   unpenalized GLM.
/// * `x_star` — the test design row `x_*` (length `p`).
/// * `family` — `"bernoulli"` or `"poisson"`.
/// * `candidates` — strictly increasing response candidates to test. Defaults
///   to `[0, 1]` (the exhaustive Bernoulli support); a Poisson caller passes
///   an explicit integer window.
/// * `alpha` — target miscoverage in `(0, 1)`.
///
/// Returns `{"members", "p_values", "candidates", "alpha", "n_augmented",
/// "refit_fallbacks", "margin_refits", "ties_unresolved",
/// "max_beta_error_bound"}`. `max_beta_error_bound` is the largest certified
/// `‖β − β̂(z)‖` over the tracked candidates — the homotopy's exactness
/// witness (`0` when every candidate was cold-fit).
#[pyfunction]
#[pyo3(signature = (design, response, s_lambda, x_star, family, alpha, candidates = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_full_conformal<'py>(
    py: Python<'py>,
    design: numpy::PyReadonlyArray2<'py, f64>,
    response: numpy::PyReadonlyArray1<'py, f64>,
    s_lambda: numpy::PyReadonlyArray2<'py, f64>,
    x_star: numpy::PyReadonlyArray1<'py, f64>,
    family: &str,
    alpha: f64,
    candidates: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let fam = canonical_glm_family(family)?;
    let x = design.as_array().to_owned();
    let y = response.as_array().to_owned();
    let sl = s_lambda.as_array().to_owned();
    let star = x_star.as_array().to_owned();
    let n = x.nrows();
    let prior_weights = Array1::<f64>::ones(n);

    let engine = GlmHomotopyFullConformal::new(fam, &x, &y, &prior_weights, &sl, &star)
        .map_err(py_value_error)?;

    // Default candidate grid: the exhaustive Bernoulli support {0, 1}. A
    // Poisson caller must supply an explicit count window — there is no honest
    // unbounded enumeration.
    let grid = match candidates {
        Some(c) => c,
        None => match fam {
            CanonicalGlmFamily::BernoulliLogit => vec![0.0, 1.0],
            CanonicalGlmFamily::PoissonLog => {
                return Err(py_value_error(
                    "glm_full_conformal: Poisson requires an explicit `candidates` count window"
                        .to_string(),
                ));
            }
        },
    };

    let set = engine
        .prediction_set(&grid, alpha)
        .map_err(py_value_error)?;

    let p_values: Vec<f64> = set.candidates.iter().map(|c| c.p_value).collect();
    let candidate_zs: Vec<f64> = set.candidates.iter().map(|c| c.z).collect();

    let out = PyDict::new(py);
    out.set_item("members", set.members)?;
    out.set_item("p_values", p_values)?;
    out.set_item("candidates", candidate_zs)?;
    out.set_item("alpha", set.alpha)?;
    out.set_item("n_augmented", set.n_augmented)?;
    out.set_item("refit_fallbacks", set.refit_fallbacks)?;
    out.set_item("margin_refits", set.margin_refits)?;
    out.set_item("ties_unresolved", set.ties_unresolved)?;
    out.set_item("max_beta_error_bound", set.max_beta_error_bound)?;
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
            use gam::linalg::faer_ndarray::FaerCholesky;
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
        assert!(
            report.se.is_finite() && report.se > 0.0,
            "SE finite & positive"
        );

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
            let kappas = vec![
                RowExpectedJets::gamma_log(eta, 1.0)
                    .kappas()
                    .expect("kappas");
                n
            ];
            let x = Array2::<f64>::ones((n, 1));
            let factor =
                lawley_lr_bartlett_factor(x.view(), &kappas, None, 0..1, 1.0).expect("factor");
            let expected = 1.0 + 1.0 / (6.0 * n as f64);
            assert!(
                (factor - expected).abs() < 1e-10,
                "n={n}: factor={factor} vs 1+1/(6n)={expected}"
            );
        }
    }

    /// ORACLE FIXTURE (#939 deliverable 3): the FFI Skovgaard assembly reproduces
    /// the certified scalar Exponential-rate closed form. `yᵢ ~ Exp(θ)` is a
    /// single-coefficient (intercept-only) canonical model with `ℓ(θ)=n lnθ−θΣy`,
    /// `θ̂=n/Σy`, and `ĵ = î = Î = n/θ̂²` at the MLE. With `c=[1]`, the matrix
    /// assembler must yield exactly the in-tree `scalar_skovgaard_r_star` result:
    /// `r=sign(θ̂−θ₀)√W`, `u=(θ̂−θ₀)√ĵ`, and the Wald-root sandwich `q<r*<r` that
    /// pulls the right-skewed first-order root back. This proves the FFI passes
    /// the ingredients through exactly — no approximation layer.
    #[test]
    fn skovgaard_ffi_assembler_recovers_exponential_closed_form() {
        use gam::inference::skovgaard::scalar_skovgaard_from_matrices;
        use ndarray::array;

        let n = 25.0_f64;
        let sum_y = 20.0_f64;
        let theta_hat = n / sum_y; // 1.25
        let theta0 = 1.0_f64;
        let ll = |t: f64| n * t.ln() - t * sum_y;
        let lr = (2.0 * (ll(theta_hat) - ll(theta0))).max(0.0);
        // Observed info ĵ = n/θ̂²: a 1×1 penalized Hessian. cᵀĤ⁻¹c = 1/ĵ.
        let obs = n / (theta_hat * theta_hat);
        let beta = array![theta_hat];
        let contrast = array![1.0_f64];
        let h = Array2::from_shape_vec((1, 1), vec![obs]).unwrap();
        // Canonical family: pass Fisher = Ĥ (î = ĵ). Score covariance Î = n/θ̂²
        // is realised by rows whose Σ sᵢ² = Î·ĵ² so the sandwich (cᵀĤ⁻¹·ΣssᵀĤ⁻¹c)
        // = Σsᵢ²/ĵ² = Î/ĵ² · ĵ² ... we instead set Σsᵢ² = ĵ² · ... :
        // sandwich = Σ(sᵢ·a)² with a = 1/ĵ ⇒ score_cov = ĵ² / Σsᵢ². For Î = ĵ we
        // need Σsᵢ² = ĵ, so a single row sᵢ = √ĵ.
        let row_scores = Array2::from_shape_vec((1, 1), vec![obs.sqrt()]).unwrap();
        let fisher = Array2::from_shape_vec((1, 1), vec![obs]).unwrap();

        let res = scalar_skovgaard_from_matrices(
            contrast.view(),
            beta.view(),
            h.view(),
            Some(fisher.view()),
            row_scores.view(),
            lr,
            theta0,
        )
        .expect("ffi-shape skovgaard");

        let r_expected = (theta_hat - theta0).signum() * lr.sqrt();
        assert!((res.r - r_expected).abs() < 1e-12, "r = {}", res.r);
        // Canonical: u = (θ̂−θ₀)·î/√ĵ = (θ̂−θ₀)√ĵ, and Î = ĵ ⇒ u_emp = u.
        let u_expected = (theta_hat - theta0) * obs.sqrt();
        assert!((res.u - u_expected).abs() < 1e-12, "u = {}", res.u);
        assert!(
            (res.u_empirical - res.u).abs() < 1e-12,
            "u_emp = {}",
            res.u_empirical
        );
        // The refinement ordering q < r* < r (Wald root < modified root < directed
        // root) for the right-skewed exponential LR.
        let q = (theta_hat - theta0) * obs.sqrt();
        assert!(
            q < res.r_star && res.r_star < r_expected,
            "need q < r* < r: q={q} r*={} r={r_expected}",
            res.r_star
        );
        assert!((0.0..=1.0).contains(&res.p_value_corrected));
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
    module.add_function(wrap_pyfunction!(select_probe_by_expected_evidence, module)?)?;
    module.add_function(wrap_pyfunction!(expected_resolution_budget, module)?)?;
    module.add_function(wrap_pyfunction!(plan_probe_for_contested_claim, module)?)?;
    module.add_function(wrap_pyfunction!(lawley_bartlett_factor, module)?)?;
    module.add_function(wrap_pyfunction!(
        lawley_bartlett_factor_estimated_lambda,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(skovgaard_r_star, module)?)?;
    module.add_function(wrap_pyfunction!(debiased_functional, module)?)?;
    module.add_function(wrap_pyfunction!(glm_full_conformal, module)?)?;
    Ok(())
}
