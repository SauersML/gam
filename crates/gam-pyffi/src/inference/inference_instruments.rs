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
    AtomBirthGate, CandidateProbe, ClaimKind, GateVerdict, ProbePlan, StructureCertificate,
    e_benjamini_hochberg, expected_resolution_budget as core_expected_resolution_budget,
    log_e_from_p_calibrator, plan_probe_for_contested_claim as core_plan_probe_for_contested_claim,
    select_probe_by_expected_evidence as core_select_probe_by_expected_evidence,
    split_likelihood_log_e_value,
};
use gam::terms::sae::manifold::bessel_i0_log_minus_abs_and_ratio;

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

    /// The realized time-to-certification: the shard count at which the running
    /// supremum first crossed `1/alpha`, or `None` if it never has. This is the
    /// first-passage time the design budget predicts; it is recorded SEPARATELY
    /// from the absorbed-shard count, which keeps growing past the crossing
    /// (absorption does not stop, so the dictionary-level e-BH certificate can
    /// clear its higher multiplicity bar).
    #[getter]
    fn certified_at_step(&self) -> Option<usize> {
        self.gate.certified_at_step()
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

/// Human-readable label for one structural claim (issue #2091 — the claim-label
/// rendering that used to live in the Python facade's `_structure_claim_label`).
fn structure_claim_label(kind: &ClaimKind) -> String {
    match kind {
        ClaimKind::AtomExists { atom } => format!("atom {atom} exists"),
        ClaimKind::BindingEdge { a, b } => format!("atoms {a}-{b} bound"),
        ClaimKind::GeometryKind { atom, kind } => format!("atom {atom} geometry={kind}"),
        ClaimKind::Custom { label } => label.clone(),
    }
}

/// Materialize the anytime-valid structure-discovery certificate report from a
/// serialized [`StructureCertificate`] (issue #2091 / #1058).
///
/// Owns the whole accessor computation the `ManifoldSAE` facade used to do in
/// numpy/Python (SPEC thin-wrapper rule 8): re-run the rank/multiplicity-aware
/// e-BH confirmation at `alpha` (defaulting to the level the fit certified at)
/// over the stored per-claim log e-values, and for each claim emit the label,
/// e-value, confirmed flag, and the anytime-valid `evidence_remaining_nats`
/// budget `max(0, ln(m / (alpha·k)) − log_e)` measured against the SAME
/// descending-log_e rank `k` (out of `m` claims) the e-BH rule uses. Returns the
/// report as a JSON string (`{"alpha", "fdr_level", "n_confirmed", "claims":
/// [...]}`) for the facade to `json.loads`; this is a runtime accessor, never
/// serialized to the model artifact, so only value-equivalence is contracted.
#[pyfunction(signature = (certificate_json, alpha=None))]
pub(crate) fn sae_structure_certificate_report(
    certificate_json: &str,
    alpha: Option<f64>,
) -> PyResult<String> {
    let cert: StructureCertificate = serde_json::from_str(certificate_json)
        .map_err(|err| py_value_error(format!("invalid structure certificate json: {err}")))?;
    let level = alpha.unwrap_or(cert.alpha);
    if !(level > 0.0 && level < 1.0) {
        return Err(py_value_error(format!(
            "alpha must lie in (0, 1); got {level}"
        )));
    }
    let entries = &cert.entries;
    let m = entries.len();
    let log_e: Vec<f64> = entries.iter().map(|entry| entry.log_e).collect();
    let confirmed_idx: std::collections::HashSet<usize> =
        e_benjamini_hochberg(&log_e, level).into_iter().collect();
    // rank_of[i] = 1-based rank of claim i in descending log_e order (stable on
    // ties, so equal log_e keep ascending index order — matching the facade's
    // `sorted(..., reverse=True)`).
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        log_e[b]
            .partial_cmp(&log_e[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank_of = vec![0usize; m];
    for (rank0, &idx) in order.iter().enumerate() {
        rank_of[idx] = rank0 + 1;
    }
    let claims: Vec<serde_json::Value> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let le = entry.log_e;
            let threshold = (m as f64).ln() - level.ln() - (rank_of[i] as f64).ln();
            serde_json::json!({
                "claim_index": i,
                "claim": structure_claim_label(&entry.kind),
                "kind": serde_json::to_value(&entry.kind)
                    .unwrap_or(serde_json::Value::Null),
                "e_value": le.exp(),
                "log_e": le,
                "steps": entry.steps,
                "confirmed": confirmed_idx.contains(&i),
                "evidence_remaining_nats": (threshold - le).max(0.0),
            })
        })
        .collect();
    let payload = serde_json::json!({
        "alpha": level,
        "fdr_level": level,
        "n_confirmed": confirmed_idx.len(),
        "claims": claims,
    });
    serde_json::to_string(&payload)
        .map_err(|err| py_value_error(format!("failed to serialize certificate report: {err}")))
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

/// Skovgaard-style modified directed likelihood root `r*` for a **scalar**
/// interest parameter `ψ = cᵀβ` (issue #939, deliverable 3), assembled from the
/// fitted-model matrices. Two approximations beyond the exact theory apply —
/// see the `gam::inference::skovgaard` module accuracy contract: (1) the
/// leading-Taylor surrogate `q̃ ≈ (θ̂−θ₀)·var[U]` (exact for canonical
/// exponential families, an extra `O(n⁻¹)` otherwise, so second-order rather
/// than third-order in general), and (2) the penalized Hessian supplies MAP
/// curvature, not pure likelihood information, so with a non-negligible
/// penalty on the interest direction the p-values describe the penalized
/// surrogate.
///
/// This exposes [`gam::inference::skovgaard::scalar_skovgaard_from_matrices`] on
/// the clean `gamfit` surface (the in-tree implementation was certified against
/// the Exponential / logistic-location closed forms but was previously
/// unreachable). Ingredients, all from the fitted penalized GLM:
/// * `contrast` (`c`) — the functional gradient `∂ψ/∂β` (a prediction row for a
///   point-on-curve, a row difference for a contrast, or any linear gradient).
/// * `beta` (`β̂`) — fitted coefficients; `ψ̂ = cᵀβ̂`.
/// * `penalized_hessian` (`Ĥ = X'WX + S_λ`) — the curvature used as observed
///   information (MAP curvature when `S_λ ≠ 0`; see the accuracy note above).
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
    fn ring_of_clusters_owns_discrete_cyclic_verdict_2262() {
        let clusters = 7usize;
        let per_cluster = 40usize;
        let mut coords = Array2::<f64>::zeros((clusters * per_cluster, 2));
        for cluster in 0..clusters {
            let angle = std::f64::consts::TAU * cluster as f64 / clusters as f64;
            let (sin_angle, cos_angle) = angle.sin_cos();
            for sample in 0..per_cluster {
                let phase = std::f64::consts::TAU * sample as f64 / per_cluster as f64;
                // This is a cloud, not a constant-radius micro-circle: the
                // varying local radius keeps the Gaussian scale direction
                // identifiable in the empirical-Fisher evidence calculation.
                let local_radius = 0.04 * (1.0 + 0.3 * (3.0 * phase).cos());
                let radial_noise = local_radius * phase.cos();
                let tangent_noise = local_radius * phase.sin();
                let radius = 2.0 + radial_noise;
                let row = cluster * per_cluster + sample;
                coords[[row, 0]] = 0.5 + radius * cos_angle - tangent_noise * sin_angle;
                coords[[row, 1]] = -0.25 + radius * sin_angle + tangent_noise * cos_angle;
            }
        }
        let verdict = run_atom_shape_race(coords.view(), 5, 2262, &[5, 7, 9]).unwrap();
        assert_eq!(verdict.ring_clusters_reporting_k, 7);
        assert_eq!(verdict.winner_class, "ring_clusters");
        assert!(
            verdict.reporting_winner.starts_with("ring_clusters_k7"),
            "reporting_winner={} names={:?} weights={:?} evidence={:?}",
            verdict.reporting_winner,
            verdict.candidate_names,
            verdict.stacking_weights,
            verdict.negative_log_evidence,
        );
        assert!(verdict.circle_wins);
        assert!(verdict.circular_margin > 0.0);
        assert_eq!(verdict.candidate_names.len(), 4);
    }

    #[test]
    fn circular_gaussian_density_is_normalized_and_finite_at_center() {
        // Integrate 2 pi r p(r) dr with composite Simpson quadrature. The
        // model is a density on Cartesian R^2, so this must be one without a
        // truncated-radius correction or a singular center convention.
        const INTERVALS: usize = 200_000;
        for (radius, noise_variance) in [(0.0, 0.7), (2.0, 0.09), (20.0, 1.0e-4)] {
            let fit = CircularGaussianFit2d {
                center: [1.25, -0.75],
                radius,
                noise_variance,
            };
            assert!(fit.log_density(fit.center[0], fit.center[1]).is_finite());

            let upper = radius + 12.0 * noise_variance.sqrt();
            let step = upper / INTERVALS as f64;
            let radial_mass = |r: f64| {
                if r == 0.0 {
                    0.0
                } else {
                    std::f64::consts::TAU
                        * r
                        * fit.log_density(fit.center[0] + r, fit.center[1]).exp()
                }
            };
            let mut weighted_sum = radial_mass(0.0) + radial_mass(upper);
            for index in 1..INTERVALS {
                let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
                weighted_sum += weight * radial_mass(index as f64 * step);
            }
            let integral = step * weighted_sum / 3.0;
            assert!(
                (integral - 1.0).abs() < 3.0e-6,
                "circular Gaussian failed to normalize: R={radius} s={noise_variance} integral={integral:.16e}"
            );
        }
    }

    #[test]
    fn circular_gaussian_mle_satisfies_all_score_equations() {
        let n = 360usize;
        let mut coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let angle = std::f64::consts::TAU * row as f64 / n as f64;
            coords[[row, 0]] = 0.2 + 1.7 * angle.cos() + 0.15 * (2.3 * angle).cos();
            coords[[row, 1]] = -0.4 + 1.7 * angle.sin() + 0.12 * (4.7 * angle).sin();
        }
        let rows = (0..n).collect::<Vec<_>>();
        let fit = CircularGaussianFit2d::fit(coords.view(), &rows).unwrap();
        let mut center_score = [0.0_f64; 2];
        let mut radius_score = 0.0_f64;
        let mut variance_score = 0.0_f64;
        for row in 0..n {
            let dx = coords[[row, 0]] - fit.center[0];
            let dy = coords[[row, 1]] - fit.center[1];
            let observed_radius = dx.hypot(dy);
            let kappa = fit.radius * observed_radius / fit.noise_variance;
            let (_, bessel_ratio) = bessel_i0_log_minus_abs_and_ratio(kappa);
            let center_multiplier =
                (1.0 - fit.radius * bessel_ratio / observed_radius) / fit.noise_variance;
            center_score[0] += center_multiplier * dx;
            center_score[1] += center_multiplier * dy;
            radius_score += (observed_radius * bessel_ratio - fit.radius) / fit.noise_variance;
            let stable_expected_residual = (observed_radius - fit.radius).powi(2)
                + 2.0 * fit.radius * observed_radius * (1.0 - bessel_ratio);
            variance_score += -1.0 / fit.noise_variance
                + 0.5 * stable_expected_residual / fit.noise_variance.powi(2);
        }
        let nf = n as f64;
        let score_scale = fit.noise_variance.sqrt();
        assert!(
            (center_score[0] / nf * score_scale).abs() < 2.0e-8
                && (center_score[1] / nf * score_scale).abs() < 2.0e-8,
            "center score did not vanish: {:?}",
            center_score
        );
        assert!(
            (radius_score / nf * score_scale).abs() < 2.0e-8,
            "radius score did not vanish: {radius_score}"
        );
        assert!(
            (variance_score / nf * fit.noise_variance).abs() < 2.0e-8,
            "variance score did not vanish: {variance_score}"
        );
    }

    #[test]
    fn smooth_circle_density_and_evidence_are_translation_invariant() {
        let n = 160usize;
        let mut coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let angle = std::f64::consts::TAU * row as f64 / n as f64;
            let radius = 2.0 + 0.08 * (3.0 * angle).cos() + 0.03 * (5.0 * angle).sin();
            coords[[row, 0]] = radius * angle.cos();
            coords[[row, 1]] = radius * angle.sin();
        }
        let mut translated = coords.clone();
        for mut row in translated.rows_mut() {
            row[0] += 137.25;
            row[1] -= 91.75;
        }
        let train = (0..120).collect::<Vec<_>>();
        let eval = (120..n).collect::<Vec<_>>();
        let base_density = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
        let shifted_density = ring_provider_2d(translated.clone())(&train, &eval).unwrap();
        for (base, shifted) in base_density.iter().zip(&shifted_density) {
            assert!(
                (base - shifted).abs() < 2.0e-11,
                "translation changed held-out ring density: base={base:.16e}, shifted={shifted:.16e}"
            );
        }
        let base_evidence = ring_negative_log_evidence_2d(coords.view()).unwrap();
        let shifted_evidence = ring_negative_log_evidence_2d(translated.view()).unwrap();
        assert!(
            (base_evidence - shifted_evidence).abs() < 2.0e-9,
            "translation changed ring evidence: base={base_evidence:.16e}, shifted={shifted_evidence:.16e}"
        );
    }

    #[test]
    fn smooth_circle_density_and_evidence_obey_the_2d_scale_law() {
        let n = 160usize;
        let scale = 7.25_f64;
        let mut coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let angle = std::f64::consts::TAU * row as f64 / n as f64;
            let radius = 2.0 + 0.08 * (3.0 * angle).cos() + 0.03 * (5.0 * angle).sin();
            coords[[row, 0]] = radius * angle.cos();
            coords[[row, 1]] = radius * angle.sin();
        }
        let scaled = coords.mapv(|value| scale * value);
        let train = (0..120).collect::<Vec<_>>();
        let eval = (120..n).collect::<Vec<_>>();
        let base_density = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
        let scaled_density = ring_provider_2d(scaled.clone())(&train, &eval).unwrap();
        for (base, transformed) in base_density.iter().zip(&scaled_density) {
            let expected = base - 2.0 * scale.ln();
            assert!(
                (transformed - expected).abs() < 2.0e-10,
                "ring density violated p_a(ax)=p(x)/a^2: expected={expected:.16e}, got={transformed:.16e}"
            );
        }

        let base_evidence = ring_negative_log_evidence_2d(coords.view()).unwrap();
        let scaled_evidence = ring_negative_log_evidence_2d(scaled.view()).unwrap();
        let expected = base_evidence + 2.0 * n as f64 * scale.ln();
        assert!(
            (scaled_evidence - expected).abs() < 2.0e-8,
            "ring evidence violated the 2-D scale law: expected={expected:.16e}, got={scaled_evidence:.16e}"
        );
    }

    #[test]
    fn held_out_values_do_not_choose_smooth_candidate_gauges() {
        let train_rows = 120usize;
        let eval_rows = 24usize;
        let mut coords = Array2::<f64>::zeros((train_rows + eval_rows, 2));
        for row in 0..coords.nrows() {
            let angle = std::f64::consts::TAU * row as f64 / coords.nrows() as f64;
            let radius = 1.8 + 0.06 * (5.0 * angle).cos();
            coords[[row, 0]] = 0.4 + radius * angle.cos();
            coords[[row, 1]] = -0.7 + radius * angle.sin();
        }
        let train = (0..train_rows).collect::<Vec<_>>();
        let eval = (train_rows..train_rows + eval_rows).collect::<Vec<_>>();
        let circle_baseline = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
        let gaussian_baseline = gaussian_provider_2d(coords.clone())(&train, &eval).unwrap();

        let perturbed_slot = eval_rows - 1;
        coords[[train_rows + perturbed_slot, 0]] = 1.0e12;
        coords[[train_rows + perturbed_slot, 1]] = -1.0e12;
        let circle_perturbed = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
        let gaussian_perturbed = gaussian_provider_2d(coords)(&train, &eval).unwrap();
        assert_eq!(
            &circle_perturbed[..perturbed_slot],
            &circle_baseline[..perturbed_slot]
        );
        assert_eq!(
            &gaussian_perturbed[..perturbed_slot],
            &gaussian_baseline[..perturbed_slot]
        );
    }

    #[test]
    fn euclidean_gaussian_is_similarity_equivariant_with_a_normalized_density() {
        let n = 180usize;
        let mut coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let phase = std::f64::consts::TAU * row as f64 / n as f64;
            let x = 2.5 * phase.cos() + 0.35 * (3.0 * phase).sin();
            let y = 0.4 * x + 0.7 * phase.sin() + 0.12 * (5.0 * phase).cos();
            coords[[row, 0]] = x;
            coords[[row, 1]] = y;
        }
        let train = (0..135).collect::<Vec<_>>();
        let eval = (135..n).collect::<Vec<_>>();
        let baseline = gaussian_provider_2d(coords.clone())(&train, &eval).unwrap();

        // A 2-D similarity x' = a Q x + b changes every normalized Cartesian
        // log density by exactly -log|det(aQ)| = -2 log(a).  In particular,
        // the covariance regularizer must scale as a^2 rather than living in
        // arbitrary absolute coordinate units.
        let scale = 1.0e-4_f64;
        let angle = 0.731_f64;
        let (sin_angle, cos_angle) = angle.sin_cos();
        let mut transformed = Array2::<f64>::zeros(coords.raw_dim());
        for row in 0..n {
            let x = coords[[row, 0]];
            let y = coords[[row, 1]];
            transformed[[row, 0]] = 0.25 + scale * (cos_angle * x - sin_angle * y);
            transformed[[row, 1]] = -0.5 + scale * (sin_angle * x + cos_angle * y);
        }
        let canonical = canonical_shape_coordinates(coords.view()).unwrap();
        let transformed_canonical = canonical_shape_coordinates(transformed.view()).unwrap();
        for row in 0..n {
            let expected_x = cos_angle * canonical[[row, 0]] - sin_angle * canonical[[row, 1]];
            let expected_y = sin_angle * canonical[[row, 0]] + cos_angle * canonical[[row, 1]];
            assert!((transformed_canonical[[row, 0]] - expected_x).abs() < 2.0e-11);
            assert!((transformed_canonical[[row, 1]] - expected_y).abs() < 2.0e-11);
        }
        let changed = gaussian_provider_2d(transformed.clone())(&train, &eval).unwrap();
        let jacobian_shift = -2.0 * scale.ln();
        for (&base, &under_similarity) in baseline.iter().zip(&changed) {
            assert!(
                (under_similarity - (base + jacobian_shift)).abs() < 2.0e-7,
                "Gaussian log density violated its similarity Jacobian: base={base:.16e}, transformed={under_similarity:.16e}"
            );
        }

        let baseline_evidence = gaussian_negative_log_evidence_2d(coords.view()).unwrap();
        let transformed_evidence = gaussian_negative_log_evidence_2d(transformed.view()).unwrap();
        let expected_evidence = baseline_evidence + 2.0 * n as f64 * scale.ln();
        assert!(
            (transformed_evidence - expected_evidence).abs() < 2.0e-5,
            "Gaussian evidence violated its similarity Jacobian: expected={expected_evidence:.16e}, actual={transformed_evidence:.16e}"
        );

        let mut translated = coords.clone();
        for mut row in translated.rows_mut() {
            row[0] += 1.0e6;
            row[1] -= 2.0e6;
        }
        let translated_density = gaussian_provider_2d(translated.clone())(&train, &eval).unwrap();
        for (&base, &shifted) in baseline.iter().zip(&translated_density) {
            assert!(
                (base - shifted).abs() < 2.0e-8,
                "translation changed Euclidean Gaussian density: base={base:.16e}, shifted={shifted:.16e}"
            );
        }
    }

    #[test]
    fn euclidean_gaussian_spectral_constraint_stays_positive_on_a_line() {
        let mut line = Array2::<f64>::zeros((9, 2));
        for row in 0..line.nrows() {
            let x = row as f64 - 4.0;
            line[[row, 0]] = x;
            line[[row, 1]] = 2.0 * x;
        }
        let rows = (0..line.nrows()).collect::<Vec<_>>();
        let fit = GaussianFit2d::fit(line.view(), &rows).unwrap();
        let on_line = fit.log_density(0.0, 0.0);
        let off_line = fit.log_density(-0.02, 0.01);
        assert!(on_line.is_finite() && off_line.is_finite());
        assert!(
            off_line < on_line - 1.0e6,
            "the constrained covariance must penalize its null direction: on={on_line}, off={off_line}"
        );
        assert!(
            gaussian_negative_log_evidence_2d(line.view())
                .unwrap()
                .is_finite()
        );

        let coincident = Array2::<f64>::from_elem((5, 2), 3.0);
        let error = GaussianFit2d::fit(coincident.view(), &[0, 1, 2, 3, 4])
            .expect_err("a zero-dimensional point mass is not a 2-D Gaussian density");
        assert!(error.contains("zero"), "{error}");
    }

    #[test]
    fn circular_verdict_aggregates_within_class_stacking_mass() {
        let kinds = [
            gam::solver::PredictiveCandidateKind::Fixed(gam::solver::AutoTopologyKind::Circle),
            gam::solver::PredictiveCandidateKind::Fixed(gam::solver::AutoTopologyKind::Euclidean),
            gam::solver::PredictiveCandidateKind::MixtureClass,
            gam::solver::PredictiveCandidateKind::RingOfClustersClass,
        ];
        // No individual circular candidate beats Euclidean (0.30 < 0.40), but
        // the circular topology class owns 0.60 total mass. A max-vs-max rule
        // would make the class verdict depend on how finely that class happened
        // to be represented in the candidate list.
        let (circular, noncircular, margin, circle_wins) =
            circular_stacking_summary(&kinds, &[0.30, 0.40, 0.0, 0.30]).unwrap();
        assert!((circular - 0.60).abs() < 1e-15);
        assert!((noncircular - 0.40).abs() < 1e-15);
        assert!((margin - 0.20).abs() < 1e-15);
        assert!(circle_wins);
    }

    #[test]
    fn discrete_rung_orders_are_selected_inside_each_outer_training_fold() {
        use gam::solver::evidence::GaussianMixtureConfig;

        let train_rows = 90usize;
        let eval_rows = 18usize;
        let mut free_coords = Array2::<f64>::zeros((train_rows + eval_rows, 2));
        for row in 0..train_rows {
            let cluster = row % 2;
            let phase = std::f64::consts::TAU * (row / 2) as f64 / (train_rows / 2) as f64;
            free_coords[[row, 0]] = if cluster == 0 { -2.0 } else { 2.0 } + 0.12 * phase.cos();
            free_coords[[row, 1]] = 0.08 * phase.sin();
        }
        // These outer-held-out rows form a remote third cluster. If they leak
        // into order selection or parameter fitting, the returned density
        // cannot equal the independently fitted training-only prediction.
        for slot in 0..eval_rows {
            let row = train_rows + slot;
            let phase = std::f64::consts::TAU * slot as f64 / eval_rows as f64;
            free_coords[[row, 0]] = 40.0 + 0.2 * phase.cos();
            free_coords[[row, 1]] = -30.0 + 0.2 * phase.sin();
        }
        let train = (0..train_rows).collect::<Vec<_>>();
        let eval = (train_rows..train_rows + eval_rows).collect::<Vec<_>>();
        let config = GaussianMixtureConfig::default();
        let ladder = vec![2usize, 3];
        let (train_coords, eval_coords, log_volume_scale) =
            canonical_shape_fold(free_coords.view(), &train, &eval, "test mixture").unwrap();
        let (train_selected_k, mut expected) = free_mixture_rung_predictive_density(
            train_coords.view(),
            eval_coords.view(),
            &ladder,
            config,
        )
        .unwrap();
        for value in &mut expected {
            *value -= log_volume_scale;
        }
        let free_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let actual = free_mixture_rung_provider_2d(
            free_coords.clone(),
            ladder.clone(),
            config,
            std::rc::Rc::clone(&free_trace),
        )(&train, &eval)
        .expect("outer provider must fit and select only on training rows");
        assert!(train_selected_k >= 2);
        assert_eq!(actual, expected);
        assert_eq!(&*free_trace.borrow(), &[train_selected_k]);

        // A held-out outlier may change its own predictive density, but it must
        // not change the training gauge, selected order, or any other held-out
        // row. The former full-data canonicalization failed this contract when
        // its absolute mixture covariance floor became active.
        let mut perturbed_free_coords = free_coords;
        let perturbed_slot = eval_rows - 1;
        perturbed_free_coords[[train_rows + perturbed_slot, 0]] = 1.0e12;
        perturbed_free_coords[[train_rows + perturbed_slot, 1]] = -1.0e12;
        let perturbed_free_trace =
            std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let perturbed_actual = free_mixture_rung_provider_2d(
            perturbed_free_coords,
            ladder,
            config,
            std::rc::Rc::clone(&perturbed_free_trace),
        )(&train, &eval)
        .expect("held-out values must not alter mixture training preprocessing");
        assert_eq!(
            &perturbed_actual[..perturbed_slot],
            &actual[..perturbed_slot]
        );
        assert_eq!(&*perturbed_free_trace.borrow(), &[train_selected_k]);

        let clusters = 3usize;
        let per_cluster = 30usize;
        let mut ring_coords = Array2::<f64>::zeros((clusters * per_cluster + eval_rows, 2));
        for cluster in 0..clusters {
            let angle = std::f64::consts::TAU * cluster as f64 / clusters as f64;
            for sample in 0..per_cluster {
                let phase = std::f64::consts::TAU * sample as f64 / per_cluster as f64;
                let row = cluster * per_cluster + sample;
                ring_coords[[row, 0]] =
                    (2.0 + 0.08 * phase.cos()) * angle.cos() - 0.05 * phase.sin() * angle.sin();
                ring_coords[[row, 1]] =
                    (2.0 + 0.08 * phase.cos()) * angle.sin() + 0.05 * phase.sin() * angle.cos();
            }
        }
        for slot in 0..eval_rows {
            let row = clusters * per_cluster + slot;
            ring_coords[[row, 0]] = 25.0 + slot as f64;
            ring_coords[[row, 1]] = -20.0;
        }
        let ring_train = (0..clusters * per_cluster).collect::<Vec<_>>();
        let ring_eval =
            (clusters * per_cluster..clusters * per_cluster + eval_rows).collect::<Vec<_>>();
        let ring_ladder = vec![3usize, 4];
        let (ring_train_coords, ring_eval_coords, ring_log_volume_scale) = canonical_shape_fold(
            ring_coords.view(),
            &ring_train,
            &ring_eval,
            "test ring cluster",
        )
        .unwrap();
        let (ring_selected_k, mut ring_expected) = ring_cluster_rung_predictive_density(
            ring_train_coords.view(),
            ring_eval_coords.view(),
            &ring_ladder,
            config,
        )
        .unwrap();
        for value in &mut ring_expected {
            *value -= ring_log_volume_scale;
        }
        let ring_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let ring_actual = ring_cluster_rung_provider_2d(
            ring_coords.clone(),
            ring_ladder.clone(),
            config,
            std::rc::Rc::clone(&ring_trace),
        )(&ring_train, &ring_eval)
        .expect("ring provider must fit and select only on training rows");
        assert!(ring_selected_k >= 3);
        assert_eq!(ring_actual, ring_expected);
        assert_eq!(&*ring_trace.borrow(), &[ring_selected_k]);

        let mut perturbed_ring_coords = ring_coords;
        let perturbed_ring_slot = eval_rows - 1;
        perturbed_ring_coords[[clusters * per_cluster + perturbed_ring_slot, 0]] = -1.0e12;
        perturbed_ring_coords[[clusters * per_cluster + perturbed_ring_slot, 1]] = 1.0e12;
        let perturbed_ring_trace =
            std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let perturbed_ring_actual = ring_cluster_rung_provider_2d(
            perturbed_ring_coords,
            ring_ladder,
            config,
            std::rc::Rc::clone(&perturbed_ring_trace),
        )(&ring_train, &ring_eval)
        .expect("held-out values must not alter ring-cluster training preprocessing");
        assert_eq!(
            &perturbed_ring_actual[..perturbed_ring_slot],
            &ring_actual[..perturbed_ring_slot]
        );
        assert_eq!(&*perturbed_ring_trace.borrow(), &[ring_selected_k]);
    }

    /// #2262 structureless-null control: on pure isotropic Gaussian noise (no
    /// ring, no cluster structure whatsoever) both matched controls must still
    /// run cleanly end to end and produce a well-formed
    /// `control_false_circle_floor` in `{0.0, 0.5, 1.0}` — the exact quantity
    /// the issue's false-circle-floor claim is built from. This exercises the
    /// SAME `matched_control_verdicts` path `adjudicate_atom_shape` calls, just
    /// without the Python boundary.
    #[test]
    fn matched_controls_report_a_well_formed_false_circle_floor_on_pure_noise_2262() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, Normal};

        let n = 300usize;
        let mut rng = StdRng::seed_from_u64(2262);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut coords = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            coords[[row, 0]] = normal.sample(&mut rng);
            coords[[row, 1]] = normal.sample(&mut rng);
        }

        // Include k=1 deliberately: it is the Euclidean candidate, not a
        // second free-mixture candidate. The race must remove that duplicate
        // before fitting and stacking.
        let ladder = [1usize, 5, 7, 9];
        let (shuffle_verdict, gaussian_verdict, false_circle_floor) = matched_control_verdicts(
            coords.view(),
            5,
            2262,
            &ladder,
            Some(80.0), // plausible healthy-dictionary mean L0
        )
        .expect("matched controls must run cleanly on pure noise, not error out");

        assert!(
            shuffle_verdict.mixture_reporting_k >= 2 && gaussian_verdict.mixture_reporting_k >= 2,
            "the free-mixture candidate must not duplicate the k=1 Euclidean Gaussian"
        );

        assert!(
            (0.0..=1.0).contains(&false_circle_floor),
            "false-circle floor must be a rate in [0, 1]: {false_circle_floor}"
        );
        assert!(
            (false_circle_floor - 0.0).abs() < 1e-12
                || (false_circle_floor - 0.5).abs() < 1e-12
                || (false_circle_floor - 1.0).abs() < 1e-12,
            "false-circle floor must be exactly {{0, 1/2, 1}} for a two-control average: got {false_circle_floor}"
        );
        assert_eq!(
            false_circle_floor,
            (usize::from(shuffle_verdict.circle_wins) + usize::from(gaussian_verdict.circle_wins))
                as f64
                / 2.0,
            "reported floor must equal the mean of the two controls' own circle_wins flags"
        );

        // mean_l0 is mandatory: omitting it must be a clean error, not a panic
        // or a silent floor.
        let err = matched_control_verdicts(coords.view(), 5, 2262, &ladder, None)
            .expect_err("mean_l0 must be required for matched controls");
        assert!(err.contains("mean_l0"), "error should name mean_l0: {err}");
    }

    /// #2262 detection-reach statement: `detection_floor` surfaced by
    /// `adjudicate_atom_shape` must be exactly the same closed-form MP edge
    /// `mp_detection_floor` (and the production reconstruction-Gram pricing)
    /// compute — no independent reimplementation to drift out of sync.
    #[test]
    fn detection_floor_matches_closed_form_mp_edge_2262() {
        use gam::terms::sae::null_battery::mp_detection_floor;

        let n_eff = 84.0_f64;
        let ambient_p = 64.0_f64;
        let dispersion_r = 1.005_f64;
        let expected = dispersion_r * (1.0 + (ambient_p / n_eff).sqrt()).powi(2);
        let floor =
            mp_detection_floor(n_eff, ambient_p, dispersion_r).expect("valid inputs must succeed");
        assert!(
            (floor - expected).abs() < 1e-12,
            "floor={floor} expected={expected}"
        );
        assert_eq!(
            shape_detection_floor(Some(n_eff), Some(ambient_p), Some(dispersion_r)).unwrap(),
            Some(expected)
        );
        assert_eq!(shape_detection_floor(None, None, None).unwrap(), None);
        for partial in [
            (Some(n_eff), None, None),
            (None, Some(ambient_p), None),
            (None, None, Some(dispersion_r)),
            (Some(n_eff), Some(ambient_p), None),
        ] {
            let error = shape_detection_floor(partial.0, partial.1, partial.2)
                .expect_err("a partial detection-reach specification must be rejected");
            assert!(error.contains("supplied together"), "{error}");
        }
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

// ───────────────────────────────────────────────────────────────────────────
// #977 / #907 — cross-class shape adjudication on a recovered atom's
// intrinsic 2-D coordinates (the SAME Rust evidence code the in-tree
// `quality_llm_weekday_circle` gate drives), exposed so the real-activation
// driver computes the verdict with ONE evidence implementation, not a Python
// re-implementation. Races a smooth S¹ ring, a Euclidean Gaussian, a free
// k-cluster mixture, and a circle-constrained ring of clusters; the held-out
// predictive-stacking headline picks the winner.
// ───────────────────────────────────────────────────────────────────────────

/// Maximum-likelihood circular-Gaussian fit for the smooth-circle candidate.
///
/// The generative model is
///
/// `X = center + radius * U + epsilon`,
///
/// where `U` is uniform on the unit circle and
/// `epsilon ~ N(0, noise_variance * I_2)`. Integrating out `U` gives the proper
/// Cartesian density
///
/// `p(x) = exp(-(r^2 + R^2)/(2s)) I0(Rr/s) / (2 pi s)`.
///
/// Unlike a Gaussian density assigned directly to the nonnegative radius, this
/// is normalized on the plane without a missing truncation normalizer, stays
/// finite at the center, and has no artificial `1/r` singularity. The center
/// is fitted jointly with `(R, s)` by exact latent-angle EM updates, rather than
/// being frozen at the coordinate mean. There are still four interior model
/// parameters: center(2), radius, and isotropic noise variance.
#[derive(Debug, Clone, Copy)]
struct CircularGaussianFit2d {
    center: [f64; 2],
    radius: f64,
    noise_variance: f64,
}

impl CircularGaussianFit2d {
    fn fit(coords: ArrayView2<'_, f64>, rows: &[usize]) -> Result<Self, String> {
        let Some(&anchor_row) = rows.first() else {
            return Err("circle density requires a nonempty training set".to_string());
        };
        if coords.ncols() != 2 || rows.iter().any(|&row| row >= coords.nrows()) {
            return Err("circle density received invalid coordinates or row indices".to_string());
        }

        if rows
            .iter()
            .any(|&row| !coords[[row, 0]].is_finite() || !coords[[row, 1]].is_finite())
        {
            return Err("circle density requires finite training coordinates".to_string());
        }

        // Work in a dimensionless chart relative to one observed point. This
        // preserves the low-order bits of a small translated circle and makes
        // the EM stopping rule and variance floor exactly scale equivariant.
        let anchor = [coords[[anchor_row, 0]], coords[[anchor_row, 1]]];
        let mut scale = 0.0_f64;
        for &row in rows {
            let dx = coords[[row, 0]] - anchor[0];
            let dy = coords[[row, 1]] - anchor[1];
            if !(dx.is_finite() && dy.is_finite()) {
                return Err("circle density coordinate range exceeds f64".to_string());
            }
            scale = scale.max(dx.hypot(dy));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err("circle density requires nonzero spatial extent".to_string());
        }

        let mut points = Vec::with_capacity(rows.len());
        let mut mean = [0.0_f64; 2];
        for &row in rows {
            let point = [
                (coords[[row, 0]] - anchor[0]) / scale,
                (coords[[row, 1]] - anchor[1]) / scale,
            ];
            points.push(point);
            mean[0] += point[0];
            mean[1] += point[1];
        }
        let count = rows.len() as f64;
        mean[0] /= count;
        mean[1] /= count;

        // Moment initialization is exact at the population level for this
        // model. If q = ||X-E X||^2, then
        //   E[q] = R^2 + 2s,  Var(q) = 4s(R^2+s),
        // hence R^4 = E[q]^2-Var(q) and s=(E[q]-R^2)/2.
        let mut squared_radii = Vec::with_capacity(rows.len());
        let mut mean_squared_radius = 0.0_f64;
        for point in &points {
            let dx = point[0] - mean[0];
            let dy = point[1] - mean[1];
            let squared_radius = dx * dx + dy * dy;
            squared_radii.push(squared_radius);
            mean_squared_radius += squared_radius;
        }
        mean_squared_radius /= count;
        let mut squared_radius_variance = 0.0_f64;
        for squared_radius in squared_radii {
            squared_radius_variance += (squared_radius - mean_squared_radius).powi(2);
        }
        squared_radius_variance /= count;

        // A noiseless observed circle is an unbounded-likelihood boundary.
        // Constrain the numerical fit to a scale-relative interior whose width
        // is at roundoff, not in arbitrary data units.
        let variance_floor = (64.0 * f64::EPSILON * mean_squared_radius).max(f64::MIN_POSITIVE);
        let radius_squared = (mean_squared_radius * mean_squared_radius - squared_radius_variance)
            .max(0.0)
            .sqrt();
        let mut radius = radius_squared.sqrt();
        let mut noise_variance = (0.5 * (mean_squared_radius - radius_squared)).max(variance_floor);
        let mut center = mean;

        // Exact EM for the latent circle angle. Given current parameters, the
        // conditional mean of U is A(kappa) * (x-c)/||x-c|| with
        // A=I1/I0 and kappa=R||x-c||/s. The joint quadratic M-step has the
        // closed forms below; using the joint center/radius solution avoids the
        // biased "center = sample mean" plug-in fit.
        const MAX_EM_ITERATIONS: usize = 4096;
        const EM_TOLERANCE: f64 = 2.0e-12;
        let mut posterior_means = vec![[0.0_f64; 2]; points.len()];
        let mut converged = false;
        for _ in 0..MAX_EM_ITERATIONS {
            let mut posterior_mean = [0.0_f64; 2];
            for (point, latent_mean) in points.iter().zip(&mut posterior_means) {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let observed_radius = dx.hypot(dy);
                if observed_radius == 0.0 || radius == 0.0 {
                    *latent_mean = [0.0, 0.0];
                } else {
                    let kappa = radius * observed_radius / noise_variance;
                    let (_, bessel_ratio) = bessel_i0_log_minus_abs_and_ratio(kappa);
                    if !(bessel_ratio.is_finite() && (0.0..=1.0).contains(&bessel_ratio)) {
                        return Err("circle density Bessel ratio left [0, 1]".to_string());
                    }
                    let multiplier = bessel_ratio / observed_radius;
                    *latent_mean = [multiplier * dx, multiplier * dy];
                }
                posterior_mean[0] += latent_mean[0];
                posterior_mean[1] += latent_mean[1];
            }
            posterior_mean[0] /= count;
            posterior_mean[1] /= count;

            let denominator =
                1.0 - posterior_mean[0] * posterior_mean[0] - posterior_mean[1] * posterior_mean[1];
            if !(denominator.is_finite() && denominator > 0.0) {
                return Err("circle density EM radius update is singular".to_string());
            }
            let mut radius_numerator = 0.0_f64;
            for (point, latent_mean) in points.iter().zip(&posterior_means) {
                radius_numerator +=
                    latent_mean[0] * (point[0] - mean[0]) + latent_mean[1] * (point[1] - mean[1]);
            }
            let next_radius = (radius_numerator / (count * denominator)).max(0.0);
            let next_center = [
                mean[0] - next_radius * posterior_mean[0],
                mean[1] - next_radius * posterior_mean[1],
            ];

            // Evaluate the expected squared residual as
            // ||d-R E[U]||^2 + R^2(1-||E[U]||^2), a nonnegative form that
            // does not catastrophically cancel on a very thin ring.
            let mut residual_sum = 0.0_f64;
            for (point, latent_mean) in points.iter().zip(&posterior_means) {
                let dx = point[0] - next_center[0];
                let dy = point[1] - next_center[1];
                let ex = dx - next_radius * latent_mean[0];
                let ey = dy - next_radius * latent_mean[1];
                let latent_norm_squared =
                    latent_mean[0] * latent_mean[0] + latent_mean[1] * latent_mean[1];
                residual_sum += ex * ex
                    + ey * ey
                    + next_radius * next_radius * (1.0 - latent_norm_squared).max(0.0);
            }
            let next_noise_variance = (residual_sum / (2.0 * count)).max(variance_floor);

            let parameter_change = (next_center[0] - center[0])
                .hypot(next_center[1] - center[1])
                .max((next_radius - radius).abs())
                .max(
                    (next_noise_variance - noise_variance).abs()
                        / (next_noise_variance + noise_variance),
                );
            center = next_center;
            radius = next_radius;
            noise_variance = next_noise_variance;
            if parameter_change <= EM_TOLERANCE {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err("circle density maximum-likelihood fit did not converge".to_string());
        }

        let fitted = Self {
            center: [anchor[0] + scale * center[0], anchor[1] + scale * center[1]],
            radius: scale * radius,
            noise_variance: scale * scale * noise_variance,
        };
        if !fitted.center.iter().all(|value| value.is_finite())
            || !fitted.radius.is_finite()
            || !(fitted.noise_variance.is_finite() && fitted.noise_variance > 0.0)
        {
            return Err("circle density fit produced non-finite parameters".to_string());
        }
        Ok(fitted)
    }

    fn log_density(self, x: f64, y: f64) -> f64 {
        let observed_radius = (x - self.center[0]).hypot(y - self.center[1]);
        let kappa = self.radius * observed_radius / self.noise_variance;
        let (log_i0_minus_kappa, _) = bessel_i0_log_minus_abs_and_ratio(kappa);
        // Algebraically this is -(r^2+R^2)/(2s)+log I0(kappa), but the
        // rearrangement preserves the cancellation log I0(kappa) ~= kappa on
        // thin rings where both terms can be enormous.
        -(std::f64::consts::TAU * self.noise_variance).ln()
            - 0.5 * (observed_radius - self.radius).powi(2) / self.noise_variance
            + log_i0_minus_kappa
    }
}

/// Held-out log-density of the smooth-circle (ring) candidate on 2-D coords:
/// a uniform latent point on the fitted circle convolved with isotropic 2-D
/// Gaussian noise. This is a normalized Cartesian density, including at the
/// fitted center.
fn ring_provider_2d(coords: Array2<f64>) -> gam::solver::HeldOutDensityProvider<'static> {
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            let (train_coords, eval_coords, log_volume_scale) =
                canonical_shape_fold(coords.view(), train, eval, "circle density")?;
            let train_rows = (0..train_coords.nrows()).collect::<Vec<_>>();
            let fit = CircularGaussianFit2d::fit(train_coords.view(), &train_rows)?;
            let mut out = Vec::with_capacity(eval_coords.nrows());
            for row in eval_coords.rows() {
                out.push(fit.log_density(row[0], row[1]) - log_volume_scale);
            }
            Ok(out)
        },
    )
}

/// One full Euclidean Gaussian fit in two dimensions.
///
/// `origin + mean_offset` is the location, represented in two pieces so a
/// translated intrinsic chart does not force us to subtract two large,
/// nearly-equal absolute coordinates when accumulating or evaluating the fit.
/// The covariance is constrained spectrally, and `precision` plus `log_norm`
/// are constructed from those exact same constrained eigenvalues.  This is
/// essential: independently flooring covariance entries and its determinant
/// does not describe any normalized Gaussian density.
#[derive(Debug, Clone, Copy)]
struct GaussianFit2d {
    origin: [f64; 2],
    mean_offset: [f64; 2],
    major_direction: [f64; 2],
    inverse_eigenvalues: [f64; 2],
    log_norm: f64,
}

impl GaussianFit2d {
    fn fit(coords: ArrayView2<'_, f64>, rows: &[usize]) -> Result<Self, String> {
        if coords.ncols() != 2 || rows.iter().any(|&row| row >= coords.nrows()) {
            return Err("Gaussian density received invalid coordinates or row indices".to_string());
        }
        if rows.len() < 3 {
            return Err("Gaussian density needs at least three training rows".to_string());
        }
        let origin = [coords[[rows[0], 0]], coords[[rows[0], 1]]];
        if !origin.iter().all(|value| value.is_finite()) {
            return Err("Gaussian density requires finite coordinates".to_string());
        }

        let mut mean_offset = [0.0_f64; 2];
        for &row in rows {
            for axis in 0..2 {
                let relative = coords[[row, axis]] - origin[axis];
                if !relative.is_finite() {
                    return Err("Gaussian density requires finite coordinates".to_string());
                }
                mean_offset[axis] += relative;
            }
        }
        let count = rows.len() as f64;
        mean_offset[0] /= count;
        mean_offset[1] /= count;

        let (mut sxx, mut sxy, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &row in rows {
            let dx = (coords[[row, 0]] - origin[0]) - mean_offset[0];
            let dy = (coords[[row, 1]] - origin[1]) - mean_offset[1];
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        sxx /= count;
        sxy /= count;
        syy /= count;
        let trace = sxx + syy;
        if !(trace.is_finite() && trace > 0.0) || !sxy.is_finite() {
            return Err(
                "Gaussian density is undefined for a point cloud with zero or non-finite variance"
                    .to_string(),
            );
        }

        // For [[sxx,sxy],[sxy,syy]], theta is the major-eigenvector angle.
        // The relative spectral floor is homogeneous in the data units: under
        // x -> a*x both eigenvalues and the floor multiply by a^2.  The old
        // absolute entry/determinant floors changed the model under this
        // harmless re-expression of an intrinsic coordinate chart.
        let spectral_gap = (sxx - syy).hypot(2.0 * sxy);
        let largest = (0.5 * (trace + spectral_gap)).max(f64::MIN_POSITIVE);
        let floor = (64.0 * f64::EPSILON * largest).max(f64::MIN_POSITIVE);
        let smallest = (0.5 * (trace - spectral_gap)).max(floor);
        let largest = largest.max(floor);
        if !smallest.is_finite() || !largest.is_finite() {
            return Err("Gaussian covariance spectrum is non-finite".to_string());
        }
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let (sin_theta, cos_theta) = theta.sin_cos();
        let inverse_largest = 1.0 / largest;
        let inverse_smallest = 1.0 / smallest;
        let log_norm = -std::f64::consts::TAU.ln() - 0.5 * (largest.ln() + smallest.ln());
        if !inverse_largest.is_finite() || !inverse_smallest.is_finite() || !log_norm.is_finite() {
            return Err("Gaussian covariance factorization is non-finite".to_string());
        }
        Ok(Self {
            origin,
            mean_offset,
            major_direction: [cos_theta, sin_theta],
            inverse_eigenvalues: [inverse_largest, inverse_smallest],
            log_norm,
        })
    }

    fn log_density(self, x: f64, y: f64) -> f64 {
        let dx = (x - self.origin[0]) - self.mean_offset[0];
        let dy = (y - self.origin[1]) - self.mean_offset[1];
        // Evaluate in the covariance eigenbasis.  This sum of nonnegative
        // squares avoids the cancellation of an expanded x' Sigma^-1 x for a
        // nearly rank-one cloud.
        let major = self.major_direction[0] * dx + self.major_direction[1] * dy;
        let minor = -self.major_direction[1] * dx + self.major_direction[0] * dy;
        let quad = major * major * self.inverse_eigenvalues[0]
            + minor * minor * self.inverse_eigenvalues[1];
        self.log_norm - 0.5 * quad
    }
}

/// Held-out log-density of the Euclidean candidate: a full 2-D Gaussian (mean +
/// 2×2 covariance) refit on each fold's training rows.
fn gaussian_provider_2d(coords: Array2<f64>) -> gam::solver::HeldOutDensityProvider<'static> {
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            let (train_coords, eval_coords, log_volume_scale) =
                canonical_shape_fold(coords.view(), train, eval, "Gaussian density")?;
            let train_rows = (0..train_coords.nrows()).collect::<Vec<_>>();
            let fit = GaussianFit2d::fit(train_coords.view(), &train_rows)?;
            let mut out = Vec::with_capacity(eval_coords.nrows());
            for row in eval_coords.rows() {
                out.push(fit.log_density(row[0], row[1]) - log_volume_scale);
            }
            Ok(out)
        },
    )
}

/// BIC-form Laplace negative-log-evidence of the circular Gaussian (4 interior
/// parameters: center(2), circle radius, and isotropic noise variance).
/// Corroborates the held-out stacking headline; lower is better.
fn ring_negative_log_evidence_2d(coords: ArrayView2<'_, f64>) -> Result<f64, String> {
    let n = coords.nrows();
    let rows = (0..n).collect::<Vec<_>>();
    let fit = CircularGaussianFit2d::fit(coords, &rows)?;
    let mut log_likelihood = 0.0_f64;
    for row in 0..n {
        log_likelihood += fit.log_density(coords[[row, 0]], coords[[row, 1]]);
    }
    Ok(-log_likelihood + 0.5 * 4.0 * (n as f64).ln())
}

/// BIC-form negative-log-evidence of the full 2-D Gaussian (5 free params:
/// mean(2) + symmetric 2×2 covariance(3)), evaluated under the exact same
/// fitted density as the held-out provider.
fn gaussian_negative_log_evidence_2d(coords: ArrayView2<'_, f64>) -> Result<f64, String> {
    let n = coords.nrows();
    let rows = (0..n).collect::<Vec<_>>();
    let fit = GaussianFit2d::fit(coords, &rows)?;
    let mut loglik = 0.0_f64;
    for row in 0..n {
        loglik += fit.log_density(coords[[row, 0]], coords[[row, 1]]);
    }
    Ok(-loglik + 0.5 * 5.0 * (n as f64).ln())
}

#[derive(Debug, Clone)]
struct AtomShapeRaceVerdict {
    winner_class: String,
    reporting_winner: String,
    candidate_names: Vec<String>,
    stacking_weights: Vec<f64>,
    negative_log_evidence: Vec<f64>,
    mixture_reporting_k: usize,
    ring_clusters_reporting_k: usize,
    mixture_fold_selected_k: Vec<usize>,
    ring_clusters_fold_selected_k: Vec<usize>,
    mixture_fold_k_histogram: std::collections::BTreeMap<usize, usize>,
    ring_clusters_fold_k_histogram: std::collections::BTreeMap<usize, usize>,
    circular_stacking_weight: f64,
    noncircular_stacking_weight: f64,
    circular_margin: f64,
    circle_wins: bool,
    is_cross_class: bool,
    headline: &'static str,
}

fn circular_stacking_summary(
    candidate_kinds: &[gam::solver::PredictiveCandidateKind],
    stacking_weights: &[f64],
) -> Result<(f64, f64, f64, bool), String> {
    if candidate_kinds.len() != stacking_weights.len() || candidate_kinds.is_empty() {
        return Err(format!(
            "shape stacking result has {} candidate kinds but {} weights",
            candidate_kinds.len(),
            stacking_weights.len()
        ));
    }
    let mut circular_weight = 0.0_f64;
    let mut noncircular_weight = 0.0_f64;
    for (&kind, &weight) in candidate_kinds.iter().zip(stacking_weights) {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "shape stacking returned invalid weight {weight} for candidate {}",
                kind.display_name()
            ));
        }
        if kind.is_circular() {
            circular_weight += weight;
        } else {
            noncircular_weight += weight;
        }
    }
    let total = circular_weight + noncircular_weight;
    if !total.is_finite() || (total - 1.0).abs() > 64.0 * f64::EPSILON.sqrt() {
        return Err(format!(
            "shape stacking weights must have unit mass; got {total}"
        ));
    }
    let circular_margin = circular_weight - noncircular_weight;
    Ok((
        circular_weight,
        noncircular_weight,
        circular_margin,
        circular_margin > 0.0,
    ))
}

#[derive(Clone, Copy)]
struct ShapeCoordinateGauge {
    anchor: [f64; 2],
    mean_offset: [f64; 2],
    scale: f64,
}

/// Fit the translation and uniform-scale gauge from training rows only.
///
/// The mean is accumulated as a sequence of convex combinations, so neither
/// `n * mean` nor a large absolute chart origin can overflow. The RMS uses the
/// LAPACK `lassq` scaling recurrence rather than summing raw squares.
fn fit_shape_coordinate_gauge(
    training: ArrayView2<'_, f64>,
    context: &str,
) -> Result<ShapeCoordinateGauge, String> {
    if training.ncols() != 2
        || training.nrows() == 0
        || !training.iter().all(|value| value.is_finite())
    {
        return Err(format!(
            "{context} training coordinates must be a nonempty finite (n, 2) matrix"
        ));
    }
    let anchor = [training[[0, 0]], training[[0, 1]]];
    let mut mean_offset = [0.0_f64; 2];
    for row in 0..training.nrows() {
        let count = (row + 1) as f64;
        let previous_weight = (count - 1.0) / count;
        let new_weight = 1.0 / count;
        for axis in 0..2 {
            let relative = training[[row, axis]] - anchor[axis];
            if !relative.is_finite() {
                return Err(format!(
                    "{context} training coordinate range overflowed on axis {axis}"
                ));
            }
            mean_offset[axis] =
                previous_weight * mean_offset[axis] + new_weight * relative;
        }
    }

    let mut norm_scale = 0.0_f64;
    let mut scaled_sum_squares = 1.0_f64;
    for row in training.rows() {
        for axis in 0..2 {
            let centered = (row[axis] - anchor[axis]) - mean_offset[axis];
            if !centered.is_finite() {
                return Err(format!(
                    "{context} centered training coordinate overflowed on axis {axis}"
                ));
            }
            let magnitude = centered.abs();
            if magnitude == 0.0 {
                continue;
            }
            if norm_scale < magnitude {
                scaled_sum_squares = 1.0
                    + scaled_sum_squares * (norm_scale / magnitude) * (norm_scale / magnitude);
                norm_scale = magnitude;
            } else {
                scaled_sum_squares += (magnitude / norm_scale) * (magnitude / norm_scale);
            }
        }
    }
    let scale = norm_scale * (scaled_sum_squares / training.nrows() as f64).sqrt();
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "{context} training coordinates have zero or non-finite centered scale"
        ));
    }
    Ok(ShapeCoordinateGauge {
        anchor,
        mean_offset,
        scale,
    })
}

fn apply_shape_coordinate_gauge(
    coords: ArrayView2<'_, f64>,
    gauge: ShapeCoordinateGauge,
    context: &str,
) -> Result<Array2<f64>, String> {
    if coords.ncols() != 2 || !coords.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "{context} coordinates must be a finite (n, 2) matrix"
        ));
    }
    let mut canonical = Array2::<f64>::zeros(coords.raw_dim());
    for row in 0..coords.nrows() {
        for axis in 0..2 {
            let value = ((coords[[row, axis]] - gauge.anchor[axis])
                - gauge.mean_offset[axis])
                / gauge.scale;
            if !value.is_finite() {
                return Err(format!(
                    "{context} canonical coordinate overflowed at row {row}, axis {axis}"
                ));
            }
            canonical[[row, axis]] = value;
        }
    }
    Ok(canonical)
}

/// Canonicalize one reporting fit from all of its rows. Outer-CV providers use
/// [`canonical_shape_fold`] instead so held-out rows cannot choose their gauge.
fn canonical_shape_coordinates(coords: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let gauge = fit_shape_coordinate_gauge(coords, "shape")?;
    apply_shape_coordinate_gauge(coords, gauge, "shape")
}

fn gather_shape_rows(
    coords: ArrayView2<'_, f64>,
    rows: &[usize],
    context: &str,
) -> Result<Array2<f64>, String> {
    if rows.iter().any(|&row| row >= coords.nrows()) {
        return Err(format!(
            "{context} contains an out-of-bounds row for {} coordinates",
            coords.nrows()
        ));
    }
    let mut gathered = Array2::<f64>::zeros((rows.len(), coords.ncols()));
    for (target, &source) in rows.iter().enumerate() {
        gathered.row_mut(target).assign(&coords.row(source));
    }
    Ok(gathered)
}

/// Derive one candidate-common chart from an outer fold's training rows and
/// apply it to both train and evaluation rows. Densities are fitted in this
/// dimensionless chart; `log_volume_scale = log(scale²)` converts their scores
/// back to proper densities in the caller's original coordinate units.
fn canonical_shape_fold(
    coords: ArrayView2<'_, f64>,
    train: &[usize],
    eval: &[usize],
    context: &str,
) -> Result<(Array2<f64>, Array2<f64>, f64), String> {
    let training = gather_shape_rows(coords, train, &format!("{context} training fold"))?;
    let evaluation = gather_shape_rows(coords, eval, &format!("{context} evaluation fold"))?;
    let gauge = fit_shape_coordinate_gauge(training.view(), context)?;
    let log_volume_scale = 2.0 * gauge.scale.ln();
    if !log_volume_scale.is_finite() {
        return Err(format!("{context} coordinate Jacobian is non-finite"));
    }
    Ok((
        apply_shape_coordinate_gauge(training.view(), gauge, context)?,
        apply_shape_coordinate_gauge(evaluation.view(), gauge, context)?,
        log_volume_scale,
    ))
}

/// Select the free-cluster order using only an outer fold's training rows, then
/// score that fold's untouched evaluation rows. The full-data order remains a
/// useful final-fit/evidence summary, but it must never choose the model used to
/// construct an outer-held-out predictive density.
fn free_mixture_rung_predictive_density(
    train: ArrayView2<'_, f64>,
    eval: ArrayView2<'_, f64>,
    ladder: &[usize],
    config: gam::solver::evidence::GaussianMixtureConfig,
) -> Result<(usize, Vec<f64>), String> {
    let rung = gam::solver::fit_mixture_rung(train, ladder, config)?;
    // Refinement may probe k=1 while bracketing k=2. Euclidean already owns
    // that identical one-component Gaussian, so the free-cluster class starts
    // at k=2 in every outer training fold just as it does on the full data.
    let fit = rung
        .fits
        .iter()
        .find(|fit| fit.k >= 2)
        .ok_or_else(|| "fold-local mixture selection produced no order k >= 2".to_string())?;
    Ok((fit.k, fit.fit.per_point_log_density(eval)?.to_vec()))
}

fn ring_cluster_rung_predictive_density(
    train: ArrayView2<'_, f64>,
    eval: ArrayView2<'_, f64>,
    ladder: &[usize],
    config: gam::solver::evidence::GaussianMixtureConfig,
) -> Result<(usize, Vec<f64>), String> {
    let rung = gam::solver::fit_ring_of_clusters_rung(train, ladder, config)?;
    let fit = rung.winner();
    Ok((fit.k, fit.fit.per_point_log_density(eval)?.to_vec()))
}

fn free_mixture_rung_provider_2d(
    coords: Array2<f64>,
    ladder: Vec<usize>,
    config: gam::solver::evidence::GaussianMixtureConfig,
    selected_orders: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
) -> gam::solver::HeldOutDensityProvider<'static> {
    Box::new(move |train: &[usize], eval: &[usize]| {
        let (train_coords, eval_coords, log_volume_scale) =
            canonical_shape_fold(coords.view(), train, eval, "mixture density")?;
        let (selected_k, mut density) = free_mixture_rung_predictive_density(
            train_coords.view(),
            eval_coords.view(),
            &ladder,
            config,
        )?;
        for value in &mut density {
            *value -= log_volume_scale;
        }
        selected_orders.borrow_mut().push(selected_k);
        Ok(density)
    })
}

fn ring_cluster_rung_provider_2d(
    coords: Array2<f64>,
    ladder: Vec<usize>,
    config: gam::solver::evidence::GaussianMixtureConfig,
    selected_orders: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
) -> gam::solver::HeldOutDensityProvider<'static> {
    Box::new(move |train: &[usize], eval: &[usize]| {
        let (train_coords, eval_coords, log_volume_scale) =
            canonical_shape_fold(coords.view(), train, eval, "ring-cluster density")?;
        let (selected_k, mut density) = ring_cluster_rung_predictive_density(
            train_coords.view(),
            eval_coords.view(),
            &ladder,
            config,
        )?;
        for value in &mut density {
            *value -= log_volume_scale;
        }
        selected_orders.borrow_mut().push(selected_k);
        Ok(density)
    })
}

fn finish_fold_order_trace(
    selected_orders: &std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
    folds: usize,
    class_name: &str,
) -> Result<Vec<usize>, String> {
    let orders = selected_orders.borrow().clone();
    if orders.len() != folds {
        return Err(format!(
            "{class_name} recorded {} fold-local orders for {folds} folds",
            orders.len()
        ));
    }
    Ok(orders)
}

fn fold_order_histogram(orders: &[usize]) -> std::collections::BTreeMap<usize, usize> {
    let mut histogram = std::collections::BTreeMap::new();
    for &order in orders {
        *histogram.entry(order).or_insert(0) += 1;
    }
    histogram
}

fn run_atom_shape_race(
    coords: ArrayView2<'_, f64>,
    folds: usize,
    seed: u64,
    k_ladder: &[usize],
) -> Result<AtomShapeRaceVerdict, String> {
    use gam::solver::evidence::{GaussianMixtureConfig, StackingConfig};
    use gam::solver::topology_selector::EvidenceCertification;
    use gam::solver::{
        AutoTopologyKind, CrossClassCandidate, Headline, PredictiveCandidateKind,
        adjudicate_cross_class_race, fit_mixture_rung, fit_ring_of_clusters_rung,
    };

    if coords.ncols() != 2 {
        return Err(format!(
            "adjudicate_atom_shape: coords must be (n, 2); got {:?}",
            coords.dim()
        ));
    }
    if coords.nrows() < 4 {
        return Err("adjudicate_atom_shape: need at least 4 rows to adjudicate".to_string());
    }
    if !coords.iter().all(|value| value.is_finite()) {
        return Err("adjudicate_atom_shape: coords must be finite".to_string());
    }
    // Full-data coordinates are canonicalized only for reporting fits and
    // corroborative evidences. Every outer-CV provider below receives the raw
    // chart and derives its gauge from that fold's training rows alone.
    let reporting_coords = canonical_shape_coordinates(coords)?;
    let raw_coords = coords.to_owned();
    let n = raw_coords.nrows();
    let config = GaussianMixtureConfig::default();
    // `Euclidean` already is the one-component full Gaussian. Letting the
    // mixture rung choose k=1 inserts the identical predictive density twice,
    // so the stacking optimum is non-identifiable and the reported weights
    // depend on candidate ordering. The free *cluster* contender begins at two
    // components; the circular cluster model begins at three.
    let mixture_ladder = k_ladder
        .iter()
        .copied()
        .filter(|&k| k >= 2)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if mixture_ladder.is_empty() {
        return Err(
            "adjudicate_atom_shape: k_ladder must contain a free-mixture order k >= 2".to_string(),
        );
    }
    let ring_cluster_ladder = mixture_ladder
        .iter()
        .copied()
        .filter(|&k| k >= 3)
        .collect::<Vec<_>>();
    if ring_cluster_ladder.is_empty() {
        return Err(
            "adjudicate_atom_shape: k_ladder must contain a ring-cluster order k >= 3".to_string(),
        );
    }
    let mixture = fit_mixture_rung(reporting_coords.view(), &mixture_ladder, config)?;
    // Local order refinement deliberately probes immediate neighbours and may
    // therefore fit k=1 while bracketing a k=2 coarse rung.  That fit is useful
    // to the in-class search but is ineligible for this race: Euclidean already
    // represents the one-component full Gaussian. Select the best ranked
    // eligible fit after refinement instead of accidentally reintroducing the
    // duplicate column we removed from the input ladder.
    let mixture_winner =
        mixture.fits.iter().find(|fit| fit.k >= 2).ok_or_else(|| {
            "shape mixture refinement produced no eligible order k >= 2".to_string()
        })?;
    let mixture_reporting_k = mixture_winner.k;
    let ring_clusters =
        fit_ring_of_clusters_rung(reporting_coords.view(), &ring_cluster_ladder, config)?;
    let ring_clusters_reporting_k = ring_clusters.winner().k;
    let mixture_fold_orders = std::rc::Rc::new(std::cell::RefCell::new(Vec::with_capacity(folds)));
    let ring_cluster_fold_orders =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::with_capacity(folds)));
    let candidate_kinds = [
        PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
        PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
        PredictiveCandidateKind::MixtureClass,
        PredictiveCandidateKind::RingOfClustersClass,
    ];
    let candidates = vec![
        CrossClassCandidate {
            kind: candidate_kinds[0],
            negative_log_evidence: ring_negative_log_evidence_2d(reporting_coords.view())?,
            certification: EvidenceCertification::Exact,
            density_provider: ring_provider_2d(raw_coords.clone()),
        },
        CrossClassCandidate {
            kind: candidate_kinds[1],
            negative_log_evidence: gaussian_negative_log_evidence_2d(reporting_coords.view())?,
            certification: EvidenceCertification::Exact,
            density_provider: gaussian_provider_2d(raw_coords.clone()),
        },
        CrossClassCandidate {
            kind: candidate_kinds[2],
            negative_log_evidence: mixture_winner.negative_log_evidence,
            certification: EvidenceCertification::Exact,
            // The displayed/reported k is the full-data final fit. Its outer-CV
            // predictive column independently selects k on each training fold,
            // and derives its chart gauge from those same rows, so held-out
            // rows cannot leak into either preprocessing or model selection.
            density_provider: free_mixture_rung_provider_2d(
                raw_coords.clone(),
                mixture_ladder.clone(),
                config,
                std::rc::Rc::clone(&mixture_fold_orders),
            ),
        },
        CrossClassCandidate {
            kind: candidate_kinds[3],
            negative_log_evidence: ring_clusters.winner().negative_log_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: ring_cluster_rung_provider_2d(
                raw_coords,
                ring_cluster_ladder.clone(),
                config,
                std::rc::Rc::clone(&ring_cluster_fold_orders),
            ),
        },
    ];
    let verdict =
        adjudicate_cross_class_race(n, candidates, folds, seed, StackingConfig::default())?;
    let stacking_weights = verdict
        .stacking
        .as_ref()
        .map(|stacking| stacking.weights.to_vec())
        .ok_or_else(|| {
            "shape race mixed model classes but returned no stacking result".to_string()
        })?;
    let winner_class = candidate_kinds[verdict.winner_index]
        .family_tag()
        .to_string();
    let reporting_winner = match candidate_kinds[verdict.winner_index] {
        PredictiveCandidateKind::MixtureClass => format!("mixture_k{mixture_reporting_k}"),
        PredictiveCandidateKind::RingOfClustersClass => {
            format!("ring_clusters_k{ring_clusters_reporting_k}")
        }
        kind => kind.display_name(),
    };
    let (circular_stacking_weight, noncircular_stacking_weight, circular_margin, circle_wins) =
        circular_stacking_summary(&candidate_kinds, &stacking_weights)?;
    let mixture_fold_selected_k =
        finish_fold_order_trace(&mixture_fold_orders, folds, "mixture class")?;
    let ring_clusters_fold_selected_k =
        finish_fold_order_trace(&ring_cluster_fold_orders, folds, "ring-cluster class")?;
    let mixture_fold_k_histogram = fold_order_histogram(&mixture_fold_selected_k);
    let ring_clusters_fold_k_histogram = fold_order_histogram(&ring_clusters_fold_selected_k);
    Ok(AtomShapeRaceVerdict {
        circle_wins,
        winner_class,
        reporting_winner,
        candidate_names: verdict.candidate_names,
        stacking_weights,
        negative_log_evidence: verdict.negative_log_evidence,
        mixture_reporting_k,
        ring_clusters_reporting_k,
        mixture_fold_selected_k,
        ring_clusters_fold_selected_k,
        mixture_fold_k_histogram,
        ring_clusters_fold_k_histogram,
        circular_stacking_weight,
        noncircular_stacking_weight,
        circular_margin,
        is_cross_class: verdict.is_cross_class,
        headline: match verdict.headline {
            Headline::Stacking => "stacking",
            Headline::Evidence => "evidence",
        },
    })
}

/// Run the two matched structureless controls (#2262) for one shape
/// adjudication and return `(shuffle_verdict, gaussian_verdict,
/// false_circle_floor)`. Pulled out of the pyfunction body so it is directly
/// unit-testable on pure-noise fixtures without a Python interpreter.
fn matched_control_verdicts(
    coords_view: ArrayView2<'_, f64>,
    folds: usize,
    seed: u64,
    ladder: &[usize],
    mean_l0: Option<f64>,
) -> Result<(AtomShapeRaceVerdict, AtomShapeRaceVerdict, f64), String> {
    let mean_l0 = mean_l0.ok_or_else(|| {
        "adjudicate_atom_shape: mean_l0 is required when matched_controls=True; a shape-verdict rate without dictionary sparsity is uninterpretable"
            .to_string()
    })?;
    if !mean_l0.is_finite() || mean_l0 < 0.0 {
        return Err(format!(
            "adjudicate_atom_shape: mean_l0 must be finite and non-negative; got {mean_l0}"
        ));
    }
    use gam::terms::sae::null_battery::{
        covariance_matched_gaussian_null, per_dimension_shuffle_null,
    };
    let shuffled = per_dimension_shuffle_null(coords_view, seed ^ 0xD1AE_510F)?;
    let gaussian = covariance_matched_gaussian_null(coords_view, seed ^ 0xC0A4_71A1)?;
    let shuffle_verdict = run_atom_shape_race(shuffled.view(), folds, seed, ladder)?;
    let gaussian_verdict = run_atom_shape_race(gaussian.view(), folds, seed, ladder)?;
    let false_circle_floor = (usize::from(shuffle_verdict.circle_wins)
        + usize::from(gaussian_verdict.circle_wins)) as f64
        / 2.0;
    Ok((shuffle_verdict, gaussian_verdict, false_circle_floor))
}

fn shape_detection_floor(
    n_eff: Option<f64>,
    ambient_p: Option<f64>,
    dispersion_r: Option<f64>,
) -> Result<Option<f64>, String> {
    match (n_eff, ambient_p, dispersion_r) {
        (None, None, None) => Ok(None),
        (Some(n_eff), Some(ambient_p), Some(dispersion_r)) => {
            gam::terms::sae::null_battery::mp_detection_floor(n_eff, ambient_p, dispersion_r)
                .map(Some)
        }
        _ => Err(
            "adjudicate_atom_shape: n_eff, ambient_p, and dispersion_r must be supplied together"
                .to_string(),
        ),
    }
}

fn atom_shape_verdict_dict<'py>(
    py: Python<'py>,
    verdict: &AtomShapeRaceVerdict,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("winner_class", &verdict.winner_class)?;
    out.set_item("reporting_winner", &verdict.reporting_winner)?;
    out.set_item("candidate_names", &verdict.candidate_names)?;
    out.set_item("stacking_weights", &verdict.stacking_weights)?;
    out.set_item("negative_log_evidence", &verdict.negative_log_evidence)?;
    out.set_item("mixture_reporting_k", verdict.mixture_reporting_k)?;
    out.set_item(
        "ring_clusters_reporting_k",
        verdict.ring_clusters_reporting_k,
    )?;
    out.set_item("mixture_fold_selected_k", &verdict.mixture_fold_selected_k)?;
    out.set_item(
        "ring_clusters_fold_selected_k",
        &verdict.ring_clusters_fold_selected_k,
    )?;
    out.set_item(
        "mixture_fold_k_histogram",
        &verdict.mixture_fold_k_histogram,
    )?;
    out.set_item(
        "ring_clusters_fold_k_histogram",
        &verdict.ring_clusters_fold_k_histogram,
    )?;
    out.set_item("circular_stacking_weight", verdict.circular_stacking_weight)?;
    out.set_item(
        "noncircular_stacking_weight",
        verdict.noncircular_stacking_weight,
    )?;
    out.set_item("circular_margin", verdict.circular_margin)?;
    out.set_item("circle_wins", verdict.circle_wins)?;
    out.set_item("is_cross_class", verdict.is_cross_class)?;
    out.set_item("headline", verdict.headline)?;
    Ok(out)
}

/// Generate one seeded structureless control for a topology census (#2262).
///
/// Call this at the entry of the pipeline being audited, then rerun the same
/// SAE training, co-activation grouping, projection, and adjudication steps.
/// Returning one control per call lets callers release it before generating the
/// other control instead of materializing two corpus-sized copies at once.
#[pyfunction]
#[pyo3(signature = (data, kind, seed = 11))]
pub(crate) fn shape_matched_control<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    kind: &str,
    seed: u64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    use gam::terms::sae::null_battery::{
        covariance_exact_hadamard_null, per_dimension_shuffle_null,
    };
    use numpy::IntoPyArray;

    let data = data.as_array();
    let control = match kind {
        "per_dimension_shuffle" => per_dimension_shuffle_null(data, seed),
        "covariance_exact_hadamard" => covariance_exact_hadamard_null(data, seed),
        other => Err(format!(
            "shape_matched_control: kind must be per_dimension_shuffle or covariance_exact_hadamard; got {other:?}"
        )),
    }
    .map_err(py_value_error)?;
    Ok(control.into_pyarray(py))
}

/// Float32-preserving structureless control for full-pipeline censuses.
///
/// Unlike [`shape_matched_control`], this entry point never widens the full
/// `n × p` input or output matrix. The covariance-exact branch transforms one
/// power-of-two row block at a time in a bounded `B × p` float64 workspace,
/// with `B <= 1024`; it never forms a `p × p` covariance or eigendecomposition.
#[pyfunction]
#[pyo3(signature = (data, kind, seed = 11))]
pub(crate) fn shape_matched_control_f32<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f32>,
    kind: &str,
    seed: u64,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    use gam::terms::sae::null_battery::{
        covariance_exact_hadamard_null_f32, per_dimension_shuffle_null_f32,
    };
    use numpy::IntoPyArray;

    let data = data.as_array();
    let control = match kind {
        "per_dimension_shuffle" => per_dimension_shuffle_null_f32(data, seed),
        "covariance_exact_hadamard" => covariance_exact_hadamard_null_f32(data, seed),
        other => Err(format!(
            "shape_matched_control_f32: kind must be per_dimension_shuffle or covariance_exact_hadamard; got {other:?}"
        )),
    }
    .map_err(py_value_error)?;
    Ok(control.into_pyarray(py))
}

/// Adjudicate the representational SHAPE of a recovered atom's intrinsic 2-D
/// coordinates (issue #977 / #907 / #2262): race a smooth S¹ ring against a
/// Euclidean Gaussian, the best free k-cluster mixture, and a constrained
/// ring-of-clusters mixture whose centers share one fitted circle. The headline
/// is held-out predictive stacking through the exact production race machinery.
/// `winner_class` is the fixed class whose outer-fold predictive column receives
/// the largest stacking weight (`circle`, `euclidean`, `mixture`, or
/// `ring_clusters`). `reporting_winner` attaches the all-data reporting order to
/// a mixture-class winner (for example `ring_clusters_k7`); that reporting fit
/// is never used to score an outer evaluation fold.
///
/// Every outer training fold selects its own free-mixture and ring-cluster order
/// using only that fold's training rows before scoring its evaluation rows.
/// `mixture_fold_selected_k` / `ring_clusters_fold_selected_k` expose those
/// leakage-free choices in fold order, and the corresponding `*_fold_k_histogram`
/// mappings summarize them. `mixture_reporting_k` and
/// `ring_clusters_reporting_k` are separate all-data fits for interpretation and
/// final deployment. The result also returns per-class stacking weights and
/// full-data evidences. Aggregating the smooth-circle and ring-cluster weights
/// makes `circle_wins` invariant to an arbitrary split of predictive mass inside
/// the circular class; `circular_margin` is circular minus non-circular mass.
///
/// `coords` is the `(n, 2)` intrinsic-coordinate matrix (e.g. `fit.coords[0]`
/// from `sae_manifold_fit`). `folds`/`seed` control the deterministic CV folding
/// of the held-out density table. By default, the identical race also runs on
/// an independent per-dimension shuffle and a covariance-matched Gaussian of
/// these supplied coordinates; `mean_l0` is then required and is emitted beside
/// this adjudicator-input false-circle floor. To audit artifacts introduced by
/// earlier SAE/grouping/PCA stages, generate each control at the pipeline entry
/// with [`shape_matched_control`] and rerun every stage. Non-dictionary callers
/// can explicitly disable `matched_controls` and receive no control-rate claim.
///
/// #2262 detection-reach statement: when `n_eff`, `ambient_p`, and
/// `dispersion_r` are all supplied (the atom's occupancy-weighted effective
/// sample size, the pre-projection ambient dictionary width, and the residual
/// dispersion `R`), the returned dict also carries `detection_floor` — the
/// Marchenko–Pastur noise edge `R·(1+√(ambient_p/n_eff))²` below which no
/// reconstruction direction at this sample size and ambient dimension can be
/// distinguished from noise (the identical closed-form edge the production
/// rank charge thresholds on; see
/// [`gam::terms::sae::null_battery::mp_detection_floor`]). A masked ring whose
/// true signal energy sits below this floor is architecturally undetectable
/// by the race regardless of the verdict returned, so this is a statement
/// about detection reach, not a substitute for the verdict. Omit all three to
/// leave `detection_floor` as `None`; supplying only a subset is an error.
#[pyfunction]
#[pyo3(
    signature = (coords, folds = 5, seed = 11, k_ladder = None, mean_l0 = None, matched_controls = true, n_eff = None, ambient_p = None, dispersion_r = None)
)]
pub(crate) fn adjudicate_atom_shape<'py>(
    py: Python<'py>,
    coords: numpy::PyReadonlyArray2<'py, f64>,
    folds: usize,
    seed: u64,
    k_ladder: Option<Vec<usize>>,
    mean_l0: Option<f64>,
    matched_controls: bool,
    n_eff: Option<f64>,
    ambient_p: Option<f64>,
    dispersion_r: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let coords_view = coords.as_array();
    let ladder = k_ladder.unwrap_or_else(|| gam::solver::MIXTURE_K_LADDER.to_vec());
    let observed =
        run_atom_shape_race(coords_view, folds, seed, &ladder).map_err(py_value_error)?;
    let out = atom_shape_verdict_dict(py, &observed)?;
    out.set_item("dictionary_mean_l0", mean_l0)?;

    let detection_floor =
        shape_detection_floor(n_eff, ambient_p, dispersion_r).map_err(py_value_error)?;
    out.set_item("detection_floor", detection_floor)?;

    if matched_controls {
        let (shuffle_verdict, gaussian_verdict, false_circle_floor) =
            matched_control_verdicts(coords_view, folds, seed, &ladder, mean_l0)
                .map_err(py_value_error)?;
        let controls = PyDict::new(py);
        controls.set_item(
            "per_dimension_shuffle",
            atom_shape_verdict_dict(py, &shuffle_verdict)?,
        )?;
        controls.set_item(
            "covariance_matched_gaussian",
            atom_shape_verdict_dict(py, &gaussian_verdict)?,
        )?;
        out.set_item("matched_controls", controls)?;
        out.set_item("control_false_circle_floor", false_circle_floor)?;
    } else {
        out.set_item("matched_controls", py.None())?;
        out.set_item("control_false_circle_floor", py.None())?;
    }
    Ok(out)
}

// ───────────────────────────────────────────────────────────────────────────
// #1017 / #1026 — cross-fit GPU multiplex throughput on the OLMo battery's real
// color-arm variant matrix. Quotes the measured `cross_fit_speedup`
// (multiplexed fits/sec ÷ sequential fits/sec) for the K{1..4}×topology{4}×
// basis{periodic,linear} sweep at the color-arm shape (n=180, p=5120), with a
// bit-for-bit parity assertion against the sequential baseline. This is the
// "throughput-quote" seam (path b): it runs the variant matrix over the resident
// kernel's deterministic frames so the speedup is measured on the REAL shape
// matrix before the per-cell real-slab fits are wired through the production
// inner solve. Concurrency is automatic (per-fit CUDA streams off the shared
// context on a CUDA build; CPU concurrency otherwise) — no flag.
// ───────────────────────────────────────────────────────────────────────────

/// Measure the GPU cross-fit multiplex throughput on the OLMo battery's real
/// color-arm variant matrix (issue #1017): dispatch the full
/// `K{1..4}×topology{4}×basis{periodic,linear}` sweep (32 cells, shape n=180,
/// p=5120) concurrently on one device, assert bit-for-bit parity against the
/// sequential baseline, and return the measured throughputs and speedup. Returns
/// a dict with `cells`, `succeeded`, `multiplexed_fits_per_second`,
/// `sequential_fits_per_second`, `cross_fit_speedup`, `multiplexed_wall_seconds`,
/// `sequential_wall_seconds`, and `used_device` (whether the device path engaged
/// on this build/host). The frames are the resident kernel's deterministic
/// fixture; the SHAPE matrix is the battery's real one, so the quoted speedup is
/// the cross-fit concurrency gain at the battery's true cell shapes.
#[pyfunction]
pub(crate) fn sweep_color_arm_throughput<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    use gam::solver::gpu_kernels::sae_resident::{
        DeviceResidentInnerOptions, assert_sweep_parity_vs_sequential, color_arm_variant_matrix,
        run_variant_sweep_multiplexed,
    };

    let variants = color_arm_variant_matrix();
    let opts = DeviceResidentInnerOptions::default();
    let (results, mux) = run_variant_sweep_multiplexed(&variants, opts).map_err(py_value_error)?;
    // Parity assertion (bit-for-bit vs sequential) doubles as the sequential
    // throughput measurement.
    let seq =
        assert_sweep_parity_vs_sequential(&variants, &opts, &results).map_err(py_value_error)?;

    let used_device = results.iter().any(|r| {
        r.as_ref()
            .map(|f| f.outcome.execution_path.used_device())
            .unwrap_or(false)
    });
    let speedup = if seq.fits_per_second > 0.0 {
        mux.fits_per_second / seq.fits_per_second
    } else {
        f64::NAN
    };

    let out = PyDict::new(py);
    out.set_item("cells", mux.fits)?;
    out.set_item("succeeded", mux.succeeded)?;
    out.set_item("multiplexed_fits_per_second", mux.fits_per_second)?;
    out.set_item("sequential_fits_per_second", seq.fits_per_second)?;
    out.set_item("cross_fit_speedup", speedup)?;
    out.set_item("multiplexed_wall_seconds", mux.wall_seconds)?;
    out.set_item("sequential_wall_seconds", seq.wall_seconds)?;
    out.set_item("used_device", used_device)?;
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
    module.add_function(wrap_pyfunction!(sae_structure_certificate_report, module)?)?;
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
    module.add_function(wrap_pyfunction!(shape_matched_control, module)?)?;
    module.add_function(wrap_pyfunction!(shape_matched_control_f32, module)?)?;
    module.add_function(wrap_pyfunction!(adjudicate_atom_shape, module)?)?;
    module.add_function(wrap_pyfunction!(sweep_color_arm_throughput, module)?)?;
    Ok(())
}
