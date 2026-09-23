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

use gam::inference::full_conformal_glm::{ConformalGlmFamily, GlmFullConformalSubstrate};
use gam::inference::lawley::{
    RhoPenaltyComponent, RowExpectedJets, RowKappas, lawley_lr_bartlett_factor,
    lawley_lr_correction_estimated_lambda,
};
use gam::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};
use gam::inference::structure_evidence::{
    AtomBirthGate, CandidateProbe, ClaimKind, GateVerdict, ProbePlan, StructureCertificate,
    e_benjamini_hochberg, e_bh_claim_verdicts,
    expected_resolution_budget as core_expected_resolution_budget,
    log_e_from_p_calibrator, plan_probe_for_contested_claim as core_plan_probe_for_contested_claim,
    select_probe_by_expected_evidence as core_select_probe_by_expected_evidence,
    split_likelihood_log_e_value,
};
use gam::probability::chi_square_sf;
use gam::terms::sae::inference::atom_shape_race::{
    AtomShapeRaceVerdict, matched_control_verdicts, run_atom_shape_race,
    shape_reconstruction_rank_edge, validate_control_mean_l0,
};
use gam::terms::sae::inference::intervention_shard::{
    ExecutedParameterRead, ExecutedParameterReads, ParameterEditScope,
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
#[pyclass(name = "AtomBirthGate", module = "gamfit._rust")]
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
) -> PyResult<f64> {
    split_likelihood_log_e_value(log_lik_alternative_on_eval, log_lik_null_sup_on_eval)
        .map_err(|error| py_value_error(error.to_string()))
}

/// e-BH dictionary certificate (Wang–Ramdas, issue #984): FDR control over the
/// claimed structures (one log e-value per claimed atom/edge) under ARBITRARY
/// dependence — exactly the regime p-value BH cannot legally handle (atoms
/// sharing every token violate PRDS). Returns the sorted indices of the
/// CONFIRMED claims with FDR ≤ `alpha`.
#[pyfunction]
pub(crate) fn e_bh_dictionary_certificate(
    log_e_values: Vec<f64>,
    alpha: f64,
) -> PyResult<Vec<usize>> {
    e_benjamini_hochberg(&log_e_values, alpha).map_err(|error| py_value_error(error.to_string()))
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
    let entries = &cert.entries;
    let log_e: Vec<f64> = entries.iter().map(|entry| entry.log_e).collect();
    // The rank/threshold arithmetic is owned by the rule's own crate
    // (`e_bh_claim_verdicts`, beside `e_benjamini_hochberg`), so this report
    // cannot drift from the rule it reports on. Alpha validation happens
    // there too.
    let verdicts =
        e_bh_claim_verdicts(&log_e, level).map_err(|error| py_value_error(error.to_string()))?;
    let n_confirmed = verdicts.iter().filter(|verdict| verdict.confirmed).count();
    let claims: Vec<serde_json::Value> = entries
        .iter()
        .zip(&verdicts)
        .enumerate()
        .map(|(i, (entry, verdict))| {
            let le = entry.log_e;
            serde_json::json!({
                "claim_index": i,
                "claim": structure_claim_label(&entry.kind),
                "kind": serde_json::to_value(&entry.kind)
                    .unwrap_or(serde_json::Value::Null),
                "e_value": le.exp(),
                "log_e": le,
                "steps": entry.steps,
                "confirmed": verdict.confirmed,
                "evidence_remaining_nats": verdict.evidence_remaining_nats,
            })
        })
        .collect();
    let payload = serde_json::json!({
        "alpha": level,
        "fdr_level": level,
        "n_confirmed": n_confirmed,
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
    let Some((idx, expected_log_growth)) =
        core_select_probe_by_expected_evidence(&probes, &fisher).map_err(py_value_error)?
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
/// Raises `ValueError` for a level outside (0, 1) or a NaN growth rate;
/// returns `None` for non-positive growth.
#[pyfunction]
pub(crate) fn expected_resolution_budget(
    alpha: f64,
    growth_nats_per_obs: f64,
) -> PyResult<Option<f64>> {
    core_expected_resolution_budget(alpha, growth_nats_per_obs).map_err(py_value_error)
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
        .map_err(py_value_error)?
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
        out.set_item("corrected_statistic", corrected)?;
        out.set_item("p_value_corrected", chi_square_sf(corrected, ref_df))?;
        out.set_item("p_value_uncorrected", chi_square_sf(stat, ref_df))?;
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
/// quantity). The two pieces enter the reference differently
/// ([`gam::inference::lawley::LawleyLrCorrection`]): the conditional shift is
/// taken at the same λ as `d` and vanishes with it, so it is the scale
/// `c = 1 + Δε(ρ̂)/d`; the ρ̂-variation increment `δ_ρ` does not vanish as the
/// tested block is absorbed (`d → 0` is where `Cov(ρ̂)` is largest), so it is an
/// additive location. The corrected reference is `c·χ²_d + δ_ρ` and the
/// corrected statistic `max(W − δ_ρ, 0)/c`. `bartlett_factor` is therefore `c`
/// (equal to `bartlett_factor_conditional`) and `rho_variation_shift` is `δ_ρ`.
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

    // The fixed-λ factor is the reference's scale; the ρ̂-variation increment
    // `½·tr(Hᵨᵨ Cov(ρ̂))` is its location (it does not scale with `d`).
    let correction = lawley_lr_correction_estimated_lambda(
        x,
        &kappas,
        penalty_owned.view(),
        tested_start..tested_end,
        &comps,
        rho_cov_owned.view(),
        ref_df,
    )
    .map_err(py_value_error)?;
    let mean_shift_conditional = (correction.scale - 1.0) * ref_df;

    let out = PyDict::new(py);
    out.set_item("bartlett_factor", correction.scale)?;
    out.set_item("bartlett_factor_conditional", correction.scale)?;
    out.set_item("mean_shift", correction.mean_shift(ref_df))?;
    out.set_item("mean_shift_conditional", mean_shift_conditional)?;
    out.set_item("rho_variation_shift", correction.location)?;
    out.set_item("ref_df", ref_df)?;
    if let Some(stat) = lr_statistic {
        if !(stat.is_finite() && stat >= 0.0) {
            return Err(py_value_error(format!(
                "lawley_bartlett_estimated: lr_statistic must be finite and non-negative; got {stat}"
            )));
        }
        let corrected = correction.corrected_statistic(stat);
        out.set_item("corrected_statistic", corrected)?;
        out.set_item("p_value_corrected", chi_square_sf(corrected, ref_df))?;
        out.set_item("p_value_uncorrected", chi_square_sf(stat, ref_df))?;
    }
    Ok(out)
}

// ───────────────────────────────────────────────────────────────────────────
// #939 deliverable 3 — Skovgaard modified directed root r* for a scalar functional
// ───────────────────────────────────────────────────────────────────────────

/// Skovgaard's modified directed likelihood root `r*` for a **scalar** interest
/// parameter `ψ = cᵀβ` with the remaining coefficients as nuisance (issue #939,
/// deliverable 3; #3535). This exposes
/// [`gam::inference::skovgaard::skovgaard_r_star_with_nuisance`] on the clean
/// `gamfit` surface; the formula and its accuracy contract are in the
/// `gam::inference::skovgaard` module documentation.
///
/// Every ingredient is a likelihood quantity of the caller's model, taken from
/// the full fit `β̂` and the constrained refit `β̃` (`cᵀβ̃ = ψ₀`, the tested
/// value); every covariance is under the full fit:
/// * `contrast` (`c`, length `p`) — the functional gradient `∂ψ/∂β`.
/// * `beta_hat` (`β̂`), `beta_null` (`β̃`) — the two fits.
/// * `observed_info_hat` (`ĵ`), `observed_info_null` (`j̃`) — observed
///   information at the two fits (`p × p`, positive definite; the penalized
///   Hessian for a penalized fit).
/// * `expected_info` (`î = var[U(β̂)]`, `p × p`, positive definite).
/// * `score_covariance` (`Ŝ = cov[U(β̂), U(β̃)ᵀ]`, `p × p`, general).
/// * `loglik_covariance` (`q̂ = cov[U(β̂), ℓ(β̂) − ℓ(β̃)]`, length `p`).
/// * `row_scores_hat`, `row_scores_null` (`n × p`) — per-row scores at the two
///   fits, and `row_loglik_diff` (length `n`) — per-row `ℓᵢ(β̂) − ℓᵢ(β̃)`; these
///   feed the empirical (Severini) companion.
/// * `lr_statistic` (`W = 2[ℓ(β̂) − ℓ(β̃)] ≥ 0`).
///
/// For a canonical-link GLM, `Ŝ = î` and `q̂ = î(β̂ − β̃)` exactly.
///
/// Returns `{"r", "u", "r_star", "p_value_first_order", "p_value_corrected",
/// "u_empirical", "r_star_empirical", "p_value_corrected_empirical", "material"}`.
#[pyfunction]
#[pyo3(signature = (
    contrast, beta_hat, beta_null, observed_info_hat, observed_info_null,
    expected_info, score_covariance, loglik_covariance, row_scores_hat,
    row_scores_null, row_loglik_diff, lr_statistic
))]
pub(crate) fn skovgaard_r_star<'py>(
    py: Python<'py>,
    contrast: numpy::PyReadonlyArray1<'py, f64>,
    beta_hat: numpy::PyReadonlyArray1<'py, f64>,
    beta_null: numpy::PyReadonlyArray1<'py, f64>,
    observed_info_hat: numpy::PyReadonlyArray2<'py, f64>,
    observed_info_null: numpy::PyReadonlyArray2<'py, f64>,
    expected_info: numpy::PyReadonlyArray2<'py, f64>,
    score_covariance: numpy::PyReadonlyArray2<'py, f64>,
    loglik_covariance: numpy::PyReadonlyArray1<'py, f64>,
    row_scores_hat: numpy::PyReadonlyArray2<'py, f64>,
    row_scores_null: numpy::PyReadonlyArray2<'py, f64>,
    row_loglik_diff: numpy::PyReadonlyArray1<'py, f64>,
    lr_statistic: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use gam::inference::skovgaard::{SkovgaardNuisanceInput, skovgaard_r_star_with_nuisance};

    let input = SkovgaardNuisanceInput {
        contrast: contrast.as_array(),
        beta_hat: beta_hat.as_array(),
        beta_null: beta_null.as_array(),
        lr_statistic,
        observed_info_hat: observed_info_hat.as_array(),
        observed_info_null: observed_info_null.as_array(),
        expected_info: expected_info.as_array(),
        score_covariance: score_covariance.as_array(),
        loglik_covariance: loglik_covariance.as_array(),
        row_scores_hat: row_scores_hat.as_array(),
        row_scores_null: row_scores_null.as_array(),
        row_loglik_diff: row_loglik_diff.as_array(),
    };
    let p = input.beta_hat.len();
    let n = input.row_scores_hat.nrows();
    for (name, len) in [
        ("contrast", input.contrast.len()),
        ("beta_null", input.beta_null.len()),
        ("loglik_covariance", input.loglik_covariance.len()),
    ] {
        if len != p {
            return Err(py_value_error(format!(
                "skovgaard_r_star: {name} has {len} entries for {p} coefficients"
            )));
        }
    }
    for (name, m) in [
        ("observed_info_hat", input.observed_info_hat),
        ("observed_info_null", input.observed_info_null),
        ("expected_info", input.expected_info),
        ("score_covariance", input.score_covariance),
    ] {
        if m.nrows() != p || m.ncols() != p {
            return Err(py_value_error(format!(
                "skovgaard_r_star: {name} is {}×{}, expected {p}×{p}",
                m.nrows(),
                m.ncols()
            )));
        }
    }
    for (name, m) in [
        ("row_scores_hat", input.row_scores_hat),
        ("row_scores_null", input.row_scores_null),
    ] {
        if m.nrows() != n || m.ncols() != p {
            return Err(py_value_error(format!(
                "skovgaard_r_star: {name} is {}×{}, expected {n}×{p}",
                m.nrows(),
                m.ncols()
            )));
        }
    }
    if input.row_loglik_diff.len() != n {
        return Err(py_value_error(format!(
            "skovgaard_r_star: row_loglik_diff has {} entries for {n} rows",
            input.row_loglik_diff.len()
        )));
    }
    if !(lr_statistic.is_finite() && lr_statistic >= 0.0) {
        return Err(py_value_error(format!(
            "skovgaard_r_star: lr_statistic must be finite and non-negative; got {lr_statistic}"
        )));
    }

    let res = skovgaard_r_star_with_nuisance(&input).ok_or_else(|| {
        py_value_error(
            "skovgaard_r_star: degenerate inputs (zero LR, non-finite entries, an \
             information matrix that is not positive definite, or a singular score \
             covariance); the first-order root stands"
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
// #942 — exact full-conformal prediction set for GLM families
// ───────────────────────────────────────────────────────────────────────────

/// Resolve the family name of the certified GLM full-conformal engine.
fn conformal_glm_family(name: &str, theta: Option<f64>) -> PyResult<ConformalGlmFamily> {
    let family = match name {
        "bernoulli" | "binomial" | "logit" | "bernoulli_logit" => {
            ConformalGlmFamily::BernoulliLogit
        }
        "poisson" | "poisson_log" => ConformalGlmFamily::PoissonLog,
        "negative_binomial" | "negbin" | "nb" => match theta {
            Some(theta) => ConformalGlmFamily::NegativeBinomialLog { theta },
            None => {
                return Err(py_value_error(
                    "glm_full_conformal: the negative binomial family needs `theta`".to_string(),
                ));
            }
        },
        "gamma" | "gamma_log" => ConformalGlmFamily::GammaLog,
        other => {
            return Err(py_value_error(format!(
                "glm_full_conformal: unknown family {other:?}; expected \"bernoulli\", \
                 \"poisson\", \"negative_binomial\" or \"gamma\""
            )));
        }
    };
    if theta.is_some() && !matches!(family, ConformalGlmFamily::NegativeBinomialLog { .. }) {
        return Err(py_value_error(format!(
            "glm_full_conformal: `theta` applies only to the negative binomial family, not {name:?}"
        )));
    }
    Ok(family)
}

/// Conservative full-conformal numerical enclosure for a GLM at a
/// frozen penalty (issue #942), computed by the same certified engine the
/// predict route uses (`gam::inference::full_conformal_glm`). For each
/// candidate response the augmented penalized fit is solved by certified
/// Newton, the `n + 1` working-score nonconformity scores are ranked, and the
/// candidate is retained if its conformal p-value exceeds `alpha` or a
/// numerical comparison cannot certify exclusion. The discrete
/// families use one independent randomized smoothed p-value per inversion.
/// The enclosure has at least nominal marginal coverage under exchangeable
/// supplied rows and a fixed symmetric fitting map; it is not a conditional
/// guarantee for a training-only learned basis. The count families enumerate the
/// whole support up to a certified tail, and Gamma walks the continuum.
///
/// Smoothing is FROZEN at the supplied penalty `s_lambda`, and there are no
/// prior weights: a reweighted training row is not exchangeable with the test
/// row, so the guarantee would not apply.
///
/// Inputs (all in the fitted coefficient basis):
/// * `design` — training design `X` (`n × p`).
/// * `response` — training response `y` (length `n`): `{0,1}` for Bernoulli,
///   non-negative integers for Poisson and negative binomial, positive for
///   Gamma.
/// * `s_lambda` — the `p × p` positive semidefinite penalty `Sλ` in
///   unit-dispersion units; pass zeros for an unpenalized GLM.
/// * `x_star` — the test design row `x_*` (length `p`).
/// * `family` — `"bernoulli"`, `"poisson"`, `"negative_binomial"` or `"gamma"`.
/// * `alpha` — target miscoverage in `(0, 1)`.
/// * `theta` — the negative-binomial size (required for that family only).
/// * `offset`, `offset_star` — training offsets (default zeros) and the test
///   row's offset (default `0`).
///
/// Returns `{"intervals", "alpha", "n_augmented", "set_kind"}`: `intervals` is the sorted,
/// disjoint list of `(lo, hi)` pieces of the set (endpoints may be infinite).
/// For the discrete families each piece is the integer run `lo..=hi`.
#[pyfunction]
#[pyo3(signature = (
    design, response, s_lambda, x_star, family, alpha,
    theta = None, offset = None, offset_star = 0.0
))]
pub(crate) fn glm_full_conformal<'py>(
    py: Python<'py>,
    design: numpy::PyReadonlyArray2<'py, f64>,
    response: numpy::PyReadonlyArray1<'py, f64>,
    s_lambda: numpy::PyReadonlyArray2<'py, f64>,
    x_star: numpy::PyReadonlyArray1<'py, f64>,
    family: &str,
    alpha: f64,
    theta: Option<f64>,
    offset: Option<numpy::PyReadonlyArray1<'py, f64>>,
    offset_star: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let family = conformal_glm_family(family, theta)?;
    let x = design.as_array().to_owned();
    let y = response.as_array().to_owned();
    let sl = s_lambda.as_array().to_owned();
    let star = x_star.as_array().to_owned();
    let (n, p) = x.dim();
    let offset = match offset {
        Some(offset) => offset.as_array().to_owned(),
        None => Array1::<f64>::zeros(n),
    };

    let set = GlmFullConformalSubstrate::new(family, x, y, offset, sl, Some(0), Array1::zeros(p))
        // One test row: request position 0.
        .and_then(|substrate| substrate.prediction_set(&star, offset_star, alpha, 0))
        .map_err(py_value_error)?;

    let intervals: Vec<(f64, f64)> = set.intervals.iter().map(|i| (i.lo, i.hi)).collect();
    let out = PyDict::new(py);
    out.set_item("intervals", intervals)?;
    out.set_item("alpha", set.alpha)?;
    out.set_item("n_augmented", set.n_augmented)?;
    out.set_item("set_kind", "conservative_enclosure")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn split_lr_log_e_is_the_likelihood_difference() {
        // log E = ℓ_alt − sup ℓ_null; a calibrated null (alt = null sup) is 0.
        assert!((split_likelihood_log_e(-10.0, -10.0).unwrap()).abs() < 1e-15);
        assert!((split_likelihood_log_e(-8.0, -10.0).unwrap() - 2.0).abs() < 1e-15);
    }

    #[test]
    fn e_bh_confirms_strong_evidence_and_drops_weak() {
        // One overwhelming claim (log e huge) clears the e-BH threshold; a
        // cluster of near-1 e-values (log e ≈ 0) does not.
        let logs = vec![20.0_f64.ln() * 5.0, 0.01, -0.2, 0.0];
        let confirmed = e_bh_dictionary_certificate(logs, 0.05).unwrap();
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

    /// #2262 structureless-null control: this frozen isotropic-Gaussian fixture
    /// contains no ring or cluster structure, so neither independently seeded
    /// matched control may promote the circular model. This exercises the same
    /// `matched_control_verdicts` path `adjudicate_atom_shape` calls without
    /// recomputing the expected aggregate from production's returned flags.
    #[test]
    fn matched_controls_do_not_promote_circle_on_seeded_isotropic_noise_2262() {
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

        let (shuffle_verdict, gaussian_verdict, control_circular_win_fraction) =
            matched_control_verdicts(
                coords.view(),
                5,
                2262,
                Some(80.0), // plausible healthy-dictionary mean L0
            )
            .expect("matched controls must run cleanly on pure noise, not error out");

        assert!(
            shuffle_verdict.mixture_reporting_k >= 2 && gaussian_verdict.mixture_reporting_k >= 2,
            "the free-mixture candidate must not duplicate the k=1 Euclidean Gaussian"
        );

        assert!(
            !shuffle_verdict.circle_wins,
            "the per-dimension-shuffle null must not promote a circle on the frozen isotropic-noise fixture"
        );
        assert!(
            !gaussian_verdict.circle_wins,
            "the covariance-matched Gaussian null must not promote a circle on the frozen isotropic-noise fixture"
        );
        assert_eq!(
            control_circular_win_fraction.to_bits(),
            0.0_f64.to_bits(),
            "two independently structureless controls must produce an exact zero circular-win fraction"
        );

        // mean_l0 is mandatory: omitting it must be a clean error, not a panic
        // or a silent floor.
        let err = matched_control_verdicts(coords.view(), 5, 2262, None)
            .expect_err("mean_l0 must be required for matched controls");
        assert!(err.contains("mean_l0"), "error should name mean_l0: {err}");
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
            (report.theta_onestep - report.theta_plugin).abs() > 1e-8,
            "fixture must exercise a nonzero orthogonal correction"
        );
        assert!(
            debiased_bias < plugin_bias,
            "orthogonal correction must strictly reduce bias: plugin={plugin_bias:.3e}, debiased={debiased_bias:.3e}"
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

    /// ORACLE FIXTURE (#939 deliverable 3, #3535): the ingredient layout the FFI
    /// passes reproduces the Exponential-rate closed form. `yᵢ ~ Exp(θ)` is a
    /// single-coefficient canonical model, `ℓ(θ) = n ln θ − θΣy`, `θ̂ = n/Σy`.
    /// For `p = 1` the determinant form cancels `|Ŝ|` against `cᵀŜ⁻¹` and `|j̃|^{1/2}` against `(cᵀj̃⁻¹c)^{1/2}`, so
    /// `u = √ĵ·q̂/î`. Canonical ⇒ `Ŝ = î = ĵ = n/θ̂²` and `q̂ = (θ̂−θ₀)·î`, so `u`
    /// is the Wald root `(θ̂−θ₀)√ĵ`. One row with `s = √ĵ` at both fits and
    /// `Δℓ = (θ̂−θ₀)√ĵ` makes the empirical form equal the model form. The
    /// right-skewed exponential LR puts `r*` strictly between the Wald root and
    /// `r`.
    #[test]
    fn skovgaard_ffi_assembler_recovers_exponential_closed_form() {
        use gam::inference::skovgaard::{SkovgaardNuisanceInput, skovgaard_r_star_with_nuisance};
        use gam::linalg::roundoff::accumulation_growth;
        use ndarray::array;

        let n = 25.0_f64;
        let sum_y = 20.0_f64;
        let theta_hat = n / sum_y; // 1.25
        let theta0 = 1.0_f64;
        let ll = |t: f64| n * t.ln() - t * sum_y;
        let lr = 2.0 * (ll(theta_hat) - ll(theta0));
        let info = n / (theta_hat * theta_hat);
        let dtheta = theta_hat - theta0;
        let info_matrix = array![[info]];
        // The constrained fit's curvature n/θ₀² enters only through
        // |j̃|^{1/2}/(cᵀj̃⁻¹c)^{1/2}, which is 1 for p = 1.
        let info_null = array![[n / (theta0 * theta0)]];
        let row_score = array![[info.sqrt()]];
        let res = skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
            contrast: array![1.0_f64].view(),
            beta_hat: array![theta_hat].view(),
            beta_null: array![theta0].view(),
            lr_statistic: lr,
            observed_info_hat: info_matrix.view(),
            observed_info_null: info_null.view(),
            expected_info: info_matrix.view(),
            score_covariance: info_matrix.view(),
            loglik_covariance: array![dtheta * info].view(),
            row_scores_hat: row_score.view(),
            row_scores_null: row_score.view(),
            row_loglik_diff: array![dtheta * info.sqrt()].view(),
        })
        .expect("ffi-shape skovgaard");

        let tol = |x: f64| accumulation_growth(16) * x.abs();
        let r_expected = dtheta.signum() * lr.sqrt();
        assert!((res.r - r_expected).abs() <= tol(r_expected), "r = {}", res.r);
        let wald = dtheta * info.sqrt();
        assert!((res.u - wald).abs() <= tol(wald), "u = {} vs Wald {wald}", res.u);
        assert!(
            (res.u_empirical - res.u).abs() <= tol(res.u),
            "u_emp = {}",
            res.u_empirical
        );
        assert!(
            wald < res.r_star && res.r_star < r_expected,
            "need Wald < r* < r: Wald={wald} r*={} r={r_expected}",
            res.r_star
        );
        assert!((0.0..=1.0).contains(&res.p_value_corrected));
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
    out.set_item("bic", &verdict.bic)?;
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
/// `n × p` input or output matrix. The covariance-exact branch transforms
/// independent power-of-two row-block/column-band tiles in at most
/// `min(8, worker_threads)` bounded float64 workspaces. Their total size never
/// exceeds that many `B × p` matrices, with `B <= 1024`, or the 128-MiB active
/// budget. Column banding keeps each tile at most 32 MiB. It never forms a
/// `p × p` covariance or eigendecomposition.
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

/// Row order of label-shuffle draw `draw` over `n_rows` labels, seeded only by
/// `(seed, draw)`.
#[pyfunction]
pub(crate) fn label_shuffle_permutation<'py>(
    py: Python<'py>,
    n_rows: usize,
    seed: u64,
    draw: u64,
) -> Bound<'py, numpy::PyArray1<u64>> {
    use numpy::IntoPyArray;

    gam::terms::sae::null_battery::label_shuffle_permutation(n_rows, seed, draw)
        .into_iter()
        .map(|row| row as u64)
        .collect::<Array1<u64>>()
        .into_pyarray(py)
}

/// Plus-one-corrected larger-tail randomization p-value of `observed` against
/// `null_statistics`, with ties counted against the observation. Returns
/// `(exceedance_count, p_value)`.
#[pyfunction]
pub(crate) fn randomization_p_value(
    observed: f64,
    null_statistics: numpy::PyReadonlyArray1<'_, f64>,
) -> PyResult<(usize, f64)> {
    use gam::terms::sae::null_battery::{Tail, empirical_p_value};

    let samples = null_statistics.as_array().to_vec();
    empirical_p_value(observed, &samples, Tail::Larger)
        .map(|calibration| (calibration.extreme_draws, calibration.p_value))
        .map_err(py_value_error)
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
/// full-data BIC/2 corroborating scores. Aggregating the smooth-circle and ring-cluster weights
/// makes `circle_wins` invariant to an arbitrary split of predictive mass inside
/// the circular class; `circular_margin` is circular minus non-circular mass.
///
/// `coords` is the `(n, 2)` intrinsic-coordinate matrix (e.g. `fit.coords[0]`
/// from `sae_manifold_fit`). `folds`/`seed` control the deterministic CV folding
/// of the held-out density table and must satisfy `2 <= folds <= n`. Thus the
/// default `folds = 5` requires `n >= 5`; with an explicit smaller fold count,
/// the shape models require `n >= 4` and at least three rows in every outer
/// training fold. Each cluster class walks its order up from its minimum (two
/// free clusters, three ring clusters) until the BIC is bracketed; no ladder of
/// orders is raced (SPEC rule 18, #2902). By default, the identical race also runs on an independent per-dimension
/// shuffle and a covariance-matched Gaussian of these supplied coordinates;
/// `mean_l0` is then required and is emitted beside this adjudicator-input
/// two-control circular-win fraction. That `{0, 1/2, 1}` value is descriptive,
/// not a false-positive-rate estimate. To audit artifacts introduced by earlier
/// SAE/grouping/PCA stages, generate each control at the pipeline entry with
/// [`shape_matched_control`] and rerun every stage. Non-dictionary callers can
/// explicitly disable `matched_controls` and receive no control-rate claim.
///
/// #2262 reconstruction-rank diagnostic: when `n_eff`, `ambient_p`, and
/// `dispersion_r` are all supplied (the atom's occupancy-weighted effective
/// sample size, the ambient output width used by rank pricing, and the residual
/// dispersion `R`), the returned dict also carries `reconstruction_rank_edge` — the
/// Marchenko–Pastur edge `R·(1+√(ambient_p/n_eff))²` that the production rank
/// charge uses for its hard reconstruction-rank count (see
/// [`gam::terms::sae::null_battery::mp_reconstruction_rank_edge`]). This is a
/// rank-charge diagnostic, not an information-theoretic limit: the predictive
/// 2-D shape race does not consume it, and a below-edge direction does not
/// negate or override the returned shape verdict. Omit all three to leave
/// `reconstruction_rank_edge` as `None`; supplying only a subset is an error.
#[pyfunction]
#[pyo3(
    signature = (coords, folds = 5, seed = 11, mean_l0 = None, matched_controls = true, n_eff = None, ambient_p = None, dispersion_r = None)
)]
pub(crate) fn adjudicate_atom_shape<'py>(
    py: Python<'py>,
    coords: numpy::PyReadonlyArray2<'py, f64>,
    folds: usize,
    seed: u64,
    mean_l0: Option<f64>,
    matched_controls: bool,
    n_eff: Option<f64>,
    ambient_p: Option<f64>,
    dispersion_r: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let coords_view = coords.as_array();
    // Reject malformed optional contracts before any reporting fit or CV race.
    // These diagnostics cannot alter the model fit, so spending a full race
    // before discovering a missing field is both slow and misleading.
    if matched_controls || mean_l0.is_some() {
        validate_control_mean_l0(mean_l0).map_err(py_value_error)?;
    }
    let reconstruction_rank_edge =
        shape_reconstruction_rank_edge(n_eff, ambient_p, dispersion_r).map_err(py_value_error)?;
    let observed = run_atom_shape_race(coords_view, folds, seed).map_err(py_value_error)?;
    let out = atom_shape_verdict_dict(py, &observed)?;
    out.set_item("dictionary_mean_l0", mean_l0)?;

    out.set_item("reconstruction_rank_edge", reconstruction_rank_edge)?;

    if matched_controls {
        let (shuffle_verdict, gaussian_verdict, control_circular_win_fraction) =
            matched_control_verdicts(coords_view, folds, seed, mean_l0)
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
        out.set_item(
            "control_circular_win_fraction",
            control_circular_win_fraction,
        )?;
    } else {
        out.set_item("matched_controls", py.None())?;
        out.set_item("control_circular_win_fraction", py.None())?;
    }
    Ok(out)
}

// ───────────────────────────────────────────────────────────────────────────
// #2946 — use-site parameter edits against the reads their forward executed
// ───────────────────────────────────────────────────────────────────────────

/// Check each use-site parameter edit a torch runner executed against the parameter
/// reads its forward made (#2946 Stage D, #2951). `edits` holds one
/// `(parameter, ordinal, read_module, read_op)` per use-site edit, with the names its
/// discovery reported for that read. `reads` holds the forward's reads in execution
/// order as `(parameter, ordinal, module, op)`. The comparison is the carrier's
/// [`ParameterEditScope::check_executed_reads`], so the first refusal raises
/// `ValueError` naming the edit and the read.
#[pyfunction]
pub(crate) fn check_parameter_use_site_reads(
    edits: Vec<(String, usize, String, String)>,
    reads: Vec<(String, usize, String, String)>,
) -> PyResult<()> {
    let executed = ExecutedParameterReads::new(
        reads
            .into_iter()
            .map(
                |(parameter, ordinal, read_module, read_op)| ExecutedParameterRead {
                    parameter,
                    ordinal,
                    read_module,
                    read_op,
                },
            )
            .collect(),
    )
    .map_err(|refusal| py_value_error(format!("{refusal:?}: {refusal}")))?;
    for (index, (parameter, ordinal, read_module, read_op)) in edits.into_iter().enumerate() {
        ParameterEditScope::UseSite {
            ordinal,
            read_module,
            read_op,
            positions: None,
        }
        .check_executed_reads(&parameter, &executed)
        .map_err(|refusal| {
            py_value_error(format!("use-site edit {index}: {refusal:?}: {refusal}"))
        })?;
    }
    Ok(())
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
    module.add_function(wrap_pyfunction!(label_shuffle_permutation, module)?)?;
    module.add_function(wrap_pyfunction!(randomization_p_value, module)?)?;
    module.add_function(wrap_pyfunction!(adjudicate_atom_shape, module)?)?;
    module.add_function(wrap_pyfunction!(check_parameter_use_site_reads, module)?)?;
    Ok(())
}
