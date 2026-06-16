//! Anytime-valid structure discovery: e-process gates, universal-inference
//! atom tests, e-BH error control, and KL-optimal steering probes.
//!
//! Interpretability as sequential experimental design.
//!
//! # The thesis
//!
//! The dictionary-learning stack (#974–#981) DISCOVERS structure: atom
//! birth/death/fission/fusion (#976), geometry adjudication — circle vs
//! clusters vs line (#907), feature binding (#975). Today those decisions
//! are made by evidence heuristics (likelihood-ratio ladders, BIC-flavored
//! gates). Three facts, never previously combined, say that is not merely
//! informal but WRONG in a specific, fixable way — and that fixing it
//! upgrades the whole capstone from observational description to
//! error-controlled experimental science:
//!
//! 1. **Atom existence is a NON-REGULAR testing problem.** "Does a K+1-th
//!    dictionary atom exist?" is testing K vs K+1 mixture components — the
//!    textbook boundary/loss-of-identifiability case where the classical
//!    likelihood-ratio χ² asymptotics FAIL (the null sits on the boundary
//!    of the alternative; the nuisance parameters of the new atom vanish
//!    under the null — Davies' problem). Every SAE/dictionary paper that
//!    thresholds a likelihood improvement is running this broken test.
//!    **Universal inference** (Wasserman–Ramdas–Balakrishnan 2020) is the
//!    modern resolution: a split-likelihood-ratio e-value that is valid in
//!    finite samples with NO regularity conditions whatsoever — exactly
//!    the irregular regime atom birth lives in.
//!
//! 2. **Discovery happens on streams with optional stopping.** Dictionaries
//!    are trained until the features "look right" — data-dependent
//!    stopping that p-hacks any fixed-sample test by construction.
//!    **E-processes** (nonnegative supermartingales under the null,
//!    `E[E_τ] ≤ 1` at every stopping time) are immune: by Ville's
//!    inequality `P(sup_t E_t ≥ 1/α) ≤ α`, the guarantee survives stopping
//!    whenever you like, peeking included, streaming corpora (#973)
//!    included. Evidence is a RUNNING PRODUCT, resumable across shards.
//!
//! 3. **This laboratory is INTERVENTIONAL.** The steering primitive with
//!    output dosimetry and a validity radius is landed
//!    (`crate::inference::steering`); the per-token output-Fisher harvest
//!    (#980) gives the local information geometry of the model's output.
//!    So "which probe next?" is not a vibe — it is OPTIMAL EXPERIMENTAL
//!    DESIGN: choose the steering intervention that maximizes the expected
//!    log-growth of the e-process deciding a contested structural claim.
//!    Under the local Gaussian output-Fisher model, that growth rate IS a
//!    KL divergence with a closed form (below). The model chooses its own
//!    next experiment, optimally, inside its certified validity radius.
//!
//! The deliverable shape: **every discovered atom ships an e-value; every
//! dictionary ships an e-BH FDR certificate over its claimed structure;
//! contested claims get design-optimal probes until evidence resolves.**
//! No other interpretability stack has finite-sample, optional-stopping-
//! safe error control over its discovered structure. This module is the
//! statistical substrate; the SAE structure search plugs its gates in.
//!
//! The instruments, bottom-up: [`EProcess`] (running evidence, Ville
//! semantics) → [`PredictablePluginEProcess`] (streaming universal
//! inference) → [`AtomBirthGate`] + [`run_atom_birth_gate`] (the K vs K+1
//! gate with demote-never-reject [`GateVerdict`]s; the runner enforces
//! the predictability contract by call order) → [`StructureLedger`] (one
//! e-process per claim, serializable across #973 shards) →
//! [`StructureLedger::certify`] (the e-BH [`StructureCertificate`],
//! shipped beside the gauge report via
//! `crate::terms::sae::identifiability::dictionary_report`) →
//! [`plan_probe_for_contested_claim`] (the design loop: contested claims
//! get a [`ProbePlan`] whose δ runs through
//! `crate::inference::steering::steer_delta` and whose per-hypothesis
//! μ₀/μ₁ come from `crate::inference::steering::predicted_response`).
//!
//! # The math, fixed here so implementations cannot drift
//!
//! **E-value / e-process.** A nonnegative statistic `E` with `E_{H0}[E] ≤ 1`.
//! Calibration to tests: reject at level α when `E ≥ 1/α` (Markov). An
//! e-PROCESS compounds multiplicatively: `E_t = Π_{s≤t} e_s` where each
//! `e_s` is conditionally valid given the past (`E[e_s | F_{s−1}] ≤ 1`
//! under H0). Ville: `P_{H0}(∃t: E_t ≥ 1/α) ≤ α` — anytime validity.
//! Always accumulate in log space; evidence products underflow doubles.
//!
//! **Universal inference (the atom-birth test).** Split the data (or the
//! token stream) into D₀ (evaluation) and D₁ (estimation). Fit the
//! K+1-atom alternative on D₁ by ANY method (the production fitter, warm
//! starts, GPU, anything — no conditions). Fit the K-atom null by
//! CONSTRAINED MLE ON D₀ (this is the one side that must be honest). Then
//!
//! ```text
//!   E = L(θ̂₁ ; D₀) / sup_{θ ∈ H0} L(θ ; D₀)
//! ```
//!
//! satisfies `E_{H0}[E] ≤ 1` in finite samples, mixtures and boundaries
//! and all (the proof is three lines of Markov + tower; no asymptotics).
//! Sequential version: at each batch t, the alternative plug-in is fit on
//! data BEFORE t (predictable), the null sup is over the batch; the
//! product is an e-process. This is `SplitLikelihoodEValue` /
//! `PredictablePluginEProcess` below, generic over log-likelihood
//! closures so the SAE stack passes its own (manifold likelihoods,
//! superposition-aware residual models #974, whatever exists then).
//!
//! **Bayes factors are e-values (the #907 bridge).** A Bayes factor
//! `BF = ∫ L(θ) dΠ₁(θ) / L_{H0}` with a FIXED (data-independent) prior Π₁
//! and a SIMPLE (or sup-dominated) null has `E_{H0}[BF] ≤ 1` — the #907
//! geometry-adjudication harness (circle vs clusters vs line, with its
//! discrete-mixture null) is therefore ONE PRIOR-FREEZE away from anytime
//! validity. The integration contract: route its per-batch BFs through
//! [`EProcess::absorb`] instead of comparing a final BF to a threshold,
//! and geometry claims inherit optional-stopping safety for free.
//!
//! **e-BH (the dictionary certificate).** Given e-values e_1..e_m for m
//! structural claims (one per atom/edge/binding), sort descending and
//! reject the top k* where `k* = max{ k : e_(k) ≥ m/(α·k) }`. This
//! controls FDR ≤ α under ARBITRARY dependence between the e-values
//! (Wang–Ramdas 2022) — no independence assumptions about atoms sharing
//! tokens, which is good because they all share every token. That
//! arbitrary-dependence robustness is WHY the certificate uses e-BH and
//! not a p-value BH: the p-version needs PRDS, which atom statistics
//! flagrantly violate.
//!
//! **Design-optimal probing.** For a contested claim with competing
//! structural hypotheses H₀/H₁ (e.g. "feature f is one curved atom" vs
//! "two flat atoms"), a candidate steering intervention δ (within its
//! certified validity radius, `steering.rs`) produces predicted output
//! distributions P₀^δ, P₁^δ. The expected per-observation log-growth of
//! the likelihood-ratio e-process under H₁ is exactly `KL(P₁^δ ‖ P₀^δ)`,
//! so the optimal next experiment is
//!
//! ```text
//!   δ* = argmax_{δ : ‖δ‖ ≤ r_valid}  KL(P₁^δ ‖ P₀^δ)
//!       ≈ argmax_δ  ½ (μ₁(δ) − μ₀(δ))ᵀ F (μ₁(δ) − μ₀(δ))
//! ```
//!
//! under the local Gaussian model with output-Fisher metric F (#980
//! harvest) — the SAME quadratic form the steering dosimetry already
//! computes, repurposed: dosimetry measures nats delivered, design
//! maximizes nats of DISCRIMINATION. Probes that maximize raw effect are
//! not optimal; probes that maximize the *disagreement between the
//! hypotheses' predictions*, weighted by output information, are. (Full
//! KL-optimal design over the probe manifold is a research arc; the greedy
//! quadratic rule below is the correct first instrument and is exact in
//! the local model.)
//!
//! # What this kills
//!
//! - Birth/death gates that are threshold heuristics → replaced by
//!   anytime-valid tests with declared error rates (#976's "detectable,
//!   correctable misspecification" becomes a literal hypothesis test).
//! - "We trained until the features looked interpretable" → optional
//!   stopping is SAFE; the certificate survives it.
//! - "We found N features" → "we found N features at FDR ≤ α, certificate
//!   attached, reproducible from the e-value ledger."
//! - Probe selection by intuition → probe selection by information.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Running anytime-valid evidence against one null hypothesis, in log
/// space. Multiplicative absorption of conditionally-valid e-values;
/// Ville's inequality converts the running product into a sequential test
/// that survives optional stopping. Serializable so evidence is resumable
/// across corpus shards (#973): persist, reload, keep absorbing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EProcess {
    /// log E_t — the running log-evidence. Starts at 0 (E_0 = 1).
    log_e: f64,
    /// Number of absorbed batches (the ledger length).
    steps: usize,
    /// Running maximum of log E_t — Ville's inequality applies to the
    /// supremum, so a claim once proven at level α stays proven even if
    /// later evidence retreats (evidence is not p-hackable in reverse).
    log_e_max: f64,
}

impl EProcess {
    pub fn new() -> Self {
        Self {
            log_e: 0.0,
            steps: 0,
            log_e_max: 0.0,
        }
    }

    /// Absorb one conditionally-valid e-value (NOT in log space; must be
    /// ≥ 0; `E[e | past] ≤ 1` under H0 is the caller's contract — e.g. a
    /// universal-inference batch ratio or a fixed-prior Bayes factor).
    pub fn absorb(&mut self, e_value: f64) -> Result<(), String> {
        if e_value.is_nan() || e_value < 0.0 {
            return Err(format!("e-value must be in [0, ∞], got {e_value}"));
        }
        self.absorb_log(e_value.ln())
    }

    /// Absorb a batch e-value supplied in log space (the only numerically
    /// honest interface for long streams).
    pub fn absorb_log(&mut self, log_e_value: f64) -> Result<(), String> {
        let next_log_e = checked_log_e_sum(self.log_e, log_e_value)?;
        self.log_e = next_log_e;
        self.steps += 1;
        if self.log_e > self.log_e_max {
            self.log_e_max = self.log_e;
        }
        Ok(())
    }

    pub fn log_evidence(&self) -> f64 {
        self.log_e
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Anytime-valid rejection at level α: by Ville,
    /// `P_{H0}(sup_t E_t ≥ 1/α) ≤ α`, so crossing 1/α at ANY time —
    /// including data-dependent stopping times — proves the claim with
    /// type-I error ≤ α. Uses the running supremum: once crossed, always
    /// rejected.
    pub fn rejects_at(&self, alpha: f64) -> bool {
        alpha > 0.0 && self.log_e_max >= -(alpha.ln())
    }

    /// The e-value to hand to [`e_benjamini_hochberg`] for the
    /// dictionary-level FDR certificate (current evidence, not the sup —
    /// e-BH's guarantee is stated for e-values at the chosen stopping
    /// time).
    pub fn current_e_value_log(&self) -> f64 {
        self.log_e
    }
}

impl Default for EProcess {
    fn default() -> Self {
        Self::new()
    }
}

fn checked_log_e_sum(current: f64, increment: f64) -> Result<f64, String> {
    if current.is_nan() {
        return Err("EProcess invariant violation: current log evidence is NaN".to_string());
    }
    if increment.is_nan() {
        return Err("log e-value must not be NaN".to_string());
    }
    if current.is_infinite()
        && increment.is_infinite()
        && current.is_sign_positive() != increment.is_sign_positive()
    {
        return Err(format!(
            "cannot combine opposing infinite log e-values: current {current}, increment {increment}"
        ));
    }
    Ok(current + increment)
}

/// One universal-inference (split-likelihood-ratio) e-value: finite-sample
/// valid with NO regularity conditions — the correct instrument for atom
/// birth (K vs K+1 components, boundary/Davies regime where χ² fails).
///
/// `log_lik_alternative_on_eval`: log-likelihood of the EVALUATION fold
/// under the alternative fitted on the ESTIMATION fold (any fitter — the
/// production manifold/SAE fit, warm-started, GPU; zero conditions on it).
/// `log_lik_null_sup_on_eval`: the SUPREMUM of the evaluation-fold
/// log-likelihood over the NULL model class (the honest side: a real
/// constrained fit on the eval fold, e.g. the K-atom dictionary refit on
/// D₀). Then `log E = ℓ_alt(D₀) − sup_{H0} ℓ(D₀)` and `E_{H0}[E] ≤ 1`
/// exactly.
pub fn split_likelihood_log_e_value(
    log_lik_alternative_on_eval: f64,
    log_lik_null_sup_on_eval: f64,
) -> f64 {
    assert!(
        !log_lik_alternative_on_eval.is_nan() && !log_lik_null_sup_on_eval.is_nan(),
        "split-likelihood log-likelihoods must not be NaN"
    );
    if log_lik_alternative_on_eval.is_infinite()
        && log_lik_null_sup_on_eval.is_infinite()
        && log_lik_alternative_on_eval.is_sign_positive()
            == log_lik_null_sup_on_eval.is_sign_positive()
    {
        return 0.0;
    }
    log_lik_alternative_on_eval - log_lik_null_sup_on_eval
}

/// Sequential universal inference over a stream of batches with a
/// PREDICTABLE plug-in: at batch t the alternative parameters were fit
/// using only data before t, the null sup is taken on batch t, and the
/// per-batch ratios compound into an e-process. This is the streaming /
/// optional-stopping form the corpus-scale pipeline (#973 shards) needs —
/// evidence is resumable: serialize `EProcess`, keep absorbing on the next
/// shard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictablePluginEProcess {
    pub process: EProcess,
}

impl PredictablePluginEProcess {
    pub fn new() -> Self {
        Self {
            process: EProcess::new(),
        }
    }

    /// Absorb one batch. The caller guarantees the alternative was fit
    /// WITHOUT batch-t data (predictability — this is what makes the
    /// product a supermartingale; violating it voids the guarantee, which
    /// is why the SAE integration must hand this function the PREVIOUS
    /// shard's fitted dictionary, never the current one).
    pub fn try_absorb_batch(
        &mut self,
        log_lik_alternative_prefit: f64,
        log_lik_null_sup_on_batch: f64,
    ) -> Result<(), String> {
        self.process.absorb_log(split_likelihood_log_e_value(
            log_lik_alternative_prefit,
            log_lik_null_sup_on_batch,
        ))
    }

    pub fn absorb_batch(
        &mut self,
        log_lik_alternative_prefit: f64,
        log_lik_null_sup_on_batch: f64,
    ) {
        self.try_absorb_batch(log_lik_alternative_prefit, log_lik_null_sup_on_batch)
            .expect("PredictablePluginEProcess received invalid log evidence");
    }
}

impl Default for PredictablePluginEProcess {
    fn default() -> Self {
        Self::new()
    }
}

/// The anytime-valid verdict on one structural claim. Deliberately
/// two-valued — there is NO "rejected" arm. Demote-never-reject (#969
/// philosophy): an e-process that has not crossed 1/α has failed to prove
/// the claim, not disproven it; the claim stays contested, keeps its
/// evidence, and earns a design-optimal probe budget instead of being
/// silently dropped (or worse, silently accepted the way a threshold gate
/// accepts whatever clears it).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GateVerdict {
    /// The running supremum crossed 1/α: the claim is proven with type-I
    /// error ≤ α, permanently (Ville applies to the sup, so later evidence
    /// retreat cannot un-prove it).
    Certified { log_e: f64 },
    /// Not (yet) proven. Carries the CURRENT log-evidence — the value the
    /// dictionary certificate's e-BH consumes, and the state a probe loop
    /// resumes from.
    Contested { log_e: f64 },
}

/// The atom-birth gate (#976's threshold comparison, replaced): a
/// universal-inference e-process over corpus shards deciding "does the
/// K+1-th atom exist?", the boundary/Davies-regime question where the χ²
/// gate every dictionary paper runs is broken.
///
/// Per shard t the integration contract is exactly the work plan's:
/// - `log_lik_alternative_prefit`: the K+1-atom dictionary fit on shards
///   BEFORE t (the PREVIOUS shard's fit — predictability is the one rule;
///   handing in the current shard's fit voids the guarantee), evaluated on
///   shard t. Any fitter, warm starts, GPU — no conditions.
/// - `log_lik_null_sup_on_shard`: the K-atom dictionary REFIT on shard t
///   (the honest constrained sup on the evaluation data).
///
/// The gate never rejects: [`GateVerdict::Contested`] is the only
/// alternative to certification, and a contested atom's next move is a
/// probe plan ([`plan_probe_for_contested_claim`]), not deletion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtomBirthGate {
    pub test: PredictablePluginEProcess,
    /// The level the certificate is claimed at; fixed at construction so a
    /// verdict can never be shopped across α after seeing the evidence.
    alpha: f64,
}

impl AtomBirthGate {
    pub fn new(alpha: f64) -> Result<Self, String> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(format!(
                "AtomBirthGate: alpha must be in (0,1), got {alpha}"
            ));
        }
        Ok(Self {
            test: PredictablePluginEProcess::new(),
            alpha,
        })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Absorb one shard's split-likelihood ratio (see type-level contract).
    pub fn try_absorb_shard(
        &mut self,
        log_lik_alternative_prefit: f64,
        log_lik_null_sup_on_shard: f64,
    ) -> Result<(), String> {
        self.test
            .try_absorb_batch(log_lik_alternative_prefit, log_lik_null_sup_on_shard)
    }

    pub fn absorb_shard(
        &mut self,
        log_lik_alternative_prefit: f64,
        log_lik_null_sup_on_shard: f64,
    ) {
        self.try_absorb_shard(log_lik_alternative_prefit, log_lik_null_sup_on_shard)
            .expect("AtomBirthGate received invalid log evidence");
    }

    pub fn verdict(&self) -> GateVerdict {
        if self.test.process.rejects_at(self.alpha) {
            GateVerdict::Certified {
                log_e: self.test.process.log_evidence(),
            }
        } else {
            GateVerdict::Contested {
                log_e: self.test.process.log_evidence(),
            }
        }
    }
}

/// Run the atom-birth gate over a shard stream with the predictability
/// contract enforced BY CONSTRUCTION: on each shard the alternative is
/// evaluated strictly before it is refit with that shard, so the plug-in
/// is always predictable and the product is always a supermartingale
/// under H0. This is the orchestration the SAE structure search calls;
/// the closures are the only fitter-specific surface.
///
/// - `alternative_log_lik(alt, shard)`: evaluation-fold log-likelihood of
///   shard under the CURRENT alternative state (fit on prior shards only —
///   guaranteed here by call order).
/// - `null_sup_log_lik(shard)`: the honest constrained sup — the K-atom
///   null REFIT on this shard (any fitter; this side must genuinely
///   maximize over H0 on the shard, an under-maximized null inflates the
///   e-value and voids validity).
/// - `refit_alternative(alt, shard)`: fold the shard into the alternative
///   (warm-started production fit, GPU, anything — zero conditions).
///
/// `initial_alternative` is the K+1 fit from data BEFORE the stream (or a
/// prior-driven init; validity never depends on its quality — a bad init
/// only costs power). Stops absorbing early once certified (the crossing
/// is permanent; further shards only cost compute), but still folds the
/// remaining shards into the alternative so the returned state has seen
/// the whole stream. Returns the gate (verdict + resumable evidence) and
/// the final alternative state.
pub fn run_atom_birth_gate<S, A>(
    alpha: f64,
    initial_alternative: A,
    shards: impl IntoIterator<Item = S>,
    mut alternative_log_lik: impl FnMut(&A, &S) -> f64,
    mut null_sup_log_lik: impl FnMut(&S) -> f64,
    mut refit_alternative: impl FnMut(A, &S) -> A,
) -> Result<(AtomBirthGate, A), String> {
    let mut gate = AtomBirthGate::new(alpha)?;
    let mut alt = initial_alternative;
    for shard in shards {
        if !matches!(gate.verdict(), GateVerdict::Certified { .. }) {
            let log_lik_alt = alternative_log_lik(&alt, &shard);
            let log_lik_null = null_sup_log_lik(&shard);
            gate.try_absorb_shard(log_lik_alt, log_lik_null)?;
        }
        alt = refit_alternative(alt, &shard);
    }
    Ok((gate, alt))
}

/// e-BH: FDR control over m structural claims under ARBITRARY dependence
/// (Wang–Ramdas). Input: per-claim log-e-values. Output: indices of
/// rejected (i.e. CONFIRMED-STRUCTURE) claims, FDR ≤ α.
///
/// Sort e-values descending; reject the top k* where
/// `k* = max{ k : e_(k) ≥ m/(α·k) }`.
///
/// This is the "dictionary certificate": run one e-process per claimed
/// atom (and per claimed binding edge, #975), call this at the chosen
/// stopping time, and the dictionary ships with an FDR-controlled list of
/// which of its claimed structures are statistically real. No
/// independence assumptions — atoms sharing every token is fine; that is
/// exactly the case p-value BH cannot legally handle (PRDS violation) and
/// e-BH can.
pub fn e_benjamini_hochberg(log_e_values: &[f64], alpha: f64) -> Vec<usize> {
    let m = log_e_values.len();
    if m == 0 || !(alpha > 0.0) {
        return Vec::new();
    }
    assert!(
        log_e_values.iter().all(|value| !value.is_nan()),
        "e-BH log e-values must not be NaN"
    );
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| log_e_values[b].total_cmp(&log_e_values[a]));
    let m_f = m as f64;
    let mut k_star = 0usize;
    for (rank0, &idx) in order.iter().enumerate() {
        let k = (rank0 + 1) as f64;
        // e_(k) ≥ m / (α k)  ⟺  log e_(k) ≥ log m − log α − log k
        if log_e_values[idx] >= m_f.ln() - alpha.ln() - k.ln() {
            k_star = rank0 + 1;
        }
    }
    order.truncate(k_star);
    order
}

/// What one structural claim asserts about the dictionary. One e-process
/// runs per claim; the kinds mirror the discovery stack's claim surface:
/// atom existence (#976 birth), binding edges (#975), geometry
/// adjudication (#907). `Custom` keeps the ledger open to claim types
/// that do not exist yet without an enum churn per new discovery gate.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClaimKind {
    /// "Atom `atom` is statistically real" — the K vs K+1 birth claim.
    AtomExists { atom: usize },
    /// "Atoms `a` and `b` are bound" — a #975 binding edge.
    BindingEdge { a: usize, b: usize },
    /// "Atom `atom`'s latent geometry is `kind`" (e.g. "circle",
    /// "clusters", "line") — a #907 adjudication claim.
    GeometryKind { atom: usize, kind: String },
    /// Any other structural claim, labeled.
    Custom { label: String },
}

/// One claim plus its running evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralClaim {
    pub kind: ClaimKind,
    pub evidence: EProcess,
}

/// The dictionary's claim ledger: every structural claim the discovery
/// stack makes, each with its own e-process. Serializable — evidence
/// resumes across corpus shards (#973) by persisting the ledger, not by
/// refitting. Calling [`StructureLedger::certify`] at ANY data-dependent
/// stopping time yields a valid certificate; that is the entire point.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StructureLedger {
    claims: Vec<StructuralClaim>,
}

impl StructureLedger {
    pub fn new() -> Self {
        Self { claims: Vec::new() }
    }

    /// Register a claim and return its ledger index. Idempotent on the
    /// claim kind: re-registering an existing claim (a resumed shard loop
    /// re-announcing its claim surface) returns the existing index and
    /// PRESERVES its accumulated evidence — a fresh e-process here would
    /// silently discard the stream's history.
    pub fn register(&mut self, kind: ClaimKind) -> usize {
        if let Some(idx) = self.claims.iter().position(|c| c.kind == kind) {
            return idx;
        }
        self.claims.push(StructuralClaim {
            kind,
            evidence: EProcess::new(),
        });
        self.claims.len() - 1
    }

    /// Absorb one conditionally-valid log e-value for claim `idx` (a
    /// universal-inference shard ratio, a frozen-prior log-BF — the
    /// caller's contract is per-source, documented on the producing gate).
    pub fn absorb_log(&mut self, idx: usize, log_e_value: f64) -> Result<(), String> {
        let n = self.claims.len();
        let claim = self.claims.get_mut(idx).ok_or_else(|| {
            format!("StructureLedger: claim index {idx} out of range ({n} claims)")
        })?;
        claim.evidence.absorb_log(log_e_value)
    }

    pub fn claims(&self) -> &[StructuralClaim] {
        &self.claims
    }

    /// The likelihood half of the probe-design loop (work-plan step 4):
    /// after running a planned probe ([`ProbePlan`] →
    /// `crate::inference::steering::steer_delta`), evaluate the REALIZED
    /// outcomes under both hypotheses' predictive densities and absorb the
    /// log-ratio into the contested claim's e-process.
    ///
    /// Validity contract: both predictive densities must be FROZEN before the
    /// probe outcome is observed — which the design loop satisfies by
    /// construction, since both hypotheses' dictionaries were fitted before
    /// the probe was even chosen. For a composite null, the null density must
    /// be the honest constrained fit (the same rule as
    /// [`split_likelihood_log_e_value`], which this delegates to); for a
    /// simple null the predictive density is the sup. Probe outcomes are new
    /// data by construction (the model was steered to produce them), so they
    /// compound validly with the claim's prior shard evidence.
    pub fn absorb_probe_outcome(
        &mut self,
        idx: usize,
        log_lik_alt_on_outcome: f64,
        log_lik_null_on_outcome: f64,
    ) -> Result<(), String> {
        self.absorb_log(
            idx,
            split_likelihood_log_e_value(log_lik_alt_on_outcome, log_lik_null_on_outcome),
        )
    }

    /// The dictionary certificate: e-BH over the ledger's CURRENT
    /// e-values at level α. FDR ≤ α over the confirmed set under arbitrary
    /// dependence — atoms sharing every token is fine — and valid at any
    /// stopping time because each entry is an e-process. Claims not
    /// confirmed are CONTESTED, never rejected (demote-never-reject); they
    /// keep their evidence and are the inputs to the probe-design loop.
    pub fn certify(&self, alpha: f64) -> StructureCertificate {
        let log_e: Vec<f64> = self
            .claims
            .iter()
            .map(|c| c.evidence.current_e_value_log())
            .collect();
        let confirmed_idx = e_benjamini_hochberg(&log_e, alpha);
        let mut entries: Vec<CertificateEntry> = self
            .claims
            .iter()
            .zip(&log_e)
            .map(|(c, &le)| CertificateEntry {
                kind: c.kind.clone(),
                log_e: le,
                steps: c.evidence.steps(),
                confirmed: false,
            })
            .collect();
        for &i in &confirmed_idx {
            entries[i].confirmed = true;
        }
        StructureCertificate { alpha, entries }
    }
}

/// One line of the certificate's e-value ledger: the claim, its
/// log-evidence at the stop, how many batches produced it, and the e-BH
/// outcome. The full entry list IS the reproducibility artifact: anyone
/// holding it can re-run [`e_benjamini_hochberg`] and re-derive the
/// confirmed set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertificateEntry {
    pub kind: ClaimKind,
    pub log_e: f64,
    pub steps: usize,
    pub confirmed: bool,
}

/// The deliverable: "we found N structures at FDR ≤ α, certificate
/// attached". Ships next to the identifiability certificate
/// ([`crate::terms::sae::identifiability::residual_gauge`], #981) — that one says
/// what the GAUGE cannot distinguish, this one says what the DATA can.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructureCertificate {
    pub alpha: f64,
    pub entries: Vec<CertificateEntry>,
}

impl StructureCertificate {
    pub fn confirmed(&self) -> impl Iterator<Item = &CertificateEntry> {
        self.entries.iter().filter(|e| e.confirmed)
    }

    pub fn contested(&self) -> impl Iterator<Item = &CertificateEntry> {
        self.entries.iter().filter(|e| !e.confirmed)
    }
}

/// Calibrate one (super)uniform p-value into a single e-value, in log
/// space: `e(p) = ½ p^{−1/2}` (the κ = ½ member of the calibrator family
/// `e_κ(p) = κ p^{κ−1}`; `∫₀¹ e_κ(p) dp = 1`, so `E_{H0}[e(P)] ≤ 1` for
/// any valid p — superuniformity only, no other conditions).
///
/// This is the bridge from p-value-shaped instruments into the ledger —
/// e.g. the feature-binding Wald test (`terms::structure::anova_atom::carve`'s
/// `edge_p_value` → a [`ClaimKind::BindingEdge`] entry). It spends
/// calibration slack (a p of 0.01 becomes e = 5, not 100), which is the
/// honest price of converting a fixed-sample test into anytime-valid
/// currency; instruments that can produce e-values natively should.
/// CONTRACT: one calibrated e-value per INDEPENDENT data batch — feeding
/// repeated tests of the same accumulating data into one e-process is the
/// p-hacking this module exists to kill.
pub fn log_e_from_p_calibrator(p_value: f64) -> Result<f64, String> {
    if !(p_value > 0.0) || p_value > 1.0 {
        return Err(format!("p-value must be in (0, 1], got {p_value}"));
    }
    Ok(0.5f64.ln() - 0.5 * p_value.ln())
}

/// A candidate steering probe for resolving one contested structural
/// claim: the intervention direction (in the steering primitive's
/// coordinates), and the two hypotheses' PREDICTED output-mean responses
/// to it.
pub struct CandidateProbe {
    /// Steering displacement δ, to be applied via
    /// `crate::inference::steering` (which enforces its own validity
    /// radius and reports realized dosimetry).
    pub delta: Array1<f64>,
    /// Predicted output-mean response under the null structure, μ₀(δ).
    pub predicted_mean_null: Array1<f64>,
    /// Predicted output-mean response under the alternative, μ₁(δ).
    pub predicted_mean_alt: Array1<f64>,
}

/// Greedy KL-optimal experimental design under the local Gaussian
/// output-Fisher model: pick the probe maximizing
/// `½ (μ₁(δ) − μ₀(δ))ᵀ F (μ₁(δ) − μ₀(δ))` — the expected per-observation
/// log-growth of the deciding e-process under the alternative.
///
/// `fisher` is the output-Fisher metric at the operating point (#980
/// harvest; the same object steering dosimetry contracts against). Probes
/// whose hypotheses predict the SAME response score zero no matter how
/// large their raw effect — the design rule selects for DISCRIMINATION,
/// not impact, which is the entire point: a maximally-steered output that
/// both hypotheses predict identically teaches nothing.
///
/// Returns the index of the best probe and its expected log-growth (nats
/// per observation), or None if no probe discriminates.
pub fn select_probe_by_expected_evidence(
    probes: &[CandidateProbe],
    fisher: &Array2<f64>,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (idx, probe) in probes.iter().enumerate() {
        let diff = &probe.predicted_mean_alt - &probe.predicted_mean_null;
        if diff.len() != fisher.nrows() {
            continue;
        }
        let f_diff = fisher.dot(&diff);
        let growth = 0.5 * diff.dot(&f_diff);
        if growth.is_finite() && growth > 0.0 {
            match best {
                Some((_, g)) if g >= growth => {}
                _ => best = Some((idx, growth)),
            }
        }
    }
    best
}

/// Expected number of observations for the chosen probe to push a claim's
/// e-process across the 1/α Ville threshold, under the alternative: the
/// design-time budget `log(1/α) / growth_rate`. This is what turns the
/// abstract guarantee into an experiment plan ("this probe should resolve
/// the claim in ~N tokens; if it hasn't, the alternative is weaker than
/// hypothesized — itself evidence").
pub fn expected_resolution_budget(alpha: f64, growth_nats_per_obs: f64) -> Option<f64> {
    if alpha <= 0.0 || alpha >= 1.0 || growth_nats_per_obs <= 0.0 {
        return None;
    }
    Some(-(alpha.ln()) / growth_nats_per_obs)
}

/// The experiment plan for one contested claim: which probe to run, the
/// expected per-observation evidence growth under the alternative, and the
/// design-time resolution budget. This is the loop's actionable output —
/// hand `probes[probe]`'s δ to `crate::inference::steering::steer_delta`
/// (which enforces the validity radius and reports realized dosimetry),
/// evaluate both hypotheses' likelihoods on the realized outputs, absorb
/// the log-ratio into the claim's e-process, re-certify.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbePlan {
    /// Index into the candidate probe list.
    pub probe: usize,
    /// Expected log-growth of the deciding e-process, nats/observation,
    /// under the alternative (the KL of the hypotheses' predicted
    /// responses in the output-Fisher metric).
    pub expected_log_growth: f64,
    /// Expected observations to cross 1/α from ZERO evidence — the
    /// conservative from-scratch budget.
    pub budget_from_scratch: f64,
    /// Expected observations to cross 1/α from the claim's CURRENT
    /// log-evidence — the remaining budget; 0 when already across.
    pub budget_remaining: f64,
}

/// Close the design loop for one contested claim: pick the probe whose
/// predicted hypothesis-disagreement (not raw effect) buys evidence
/// fastest, and convert the claim's current evidence into a remaining
/// budget — "this probe should resolve the claim in ~N more observations
/// at level α; if it does not, the alternative is weaker than
/// hypothesized, which is itself evidence."
///
/// `current_log_e` is the contested claim's running log-evidence (from its
/// [`StructuralClaim`] / [`GateVerdict::Contested`]). Returns `None` when
/// no probe discriminates (all candidates score zero growth: the
/// hypotheses agree on everything reachable inside the validity radius —
/// the claim is undecidable by steering and needs a different instrument,
/// which is a finding, not a failure).
pub fn plan_probe_for_contested_claim(
    probes: &[CandidateProbe],
    fisher: &Array2<f64>,
    alpha: f64,
    current_log_e: f64,
) -> Option<ProbePlan> {
    let (probe, expected_log_growth) = select_probe_by_expected_evidence(probes, fisher)?;
    let budget_from_scratch = expected_resolution_budget(alpha, expected_log_growth)?;
    let nats_remaining = (-(alpha.ln()) - current_log_e).max(0.0);
    Some(ProbePlan {
        probe,
        expected_log_growth,
        budget_from_scratch,
        budget_remaining: nats_remaining / expected_log_growth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// e-BH on a hand-checkable configuration.
    #[test]
    fn e_bh_rejects_exactly_the_qualifying_prefix() {
        // m = 4, α = 0.1 → thresholds m/(αk) = 40, 20, 13.33, 10.
        let log_e = [45.0f64.ln(), 21.0f64.ln(), 12.0f64.ln(), 1.0f64.ln()];
        let rejected = e_benjamini_hochberg(&log_e, 0.1);
        // e_(1)=45 ≥ 40 ✓, e_(2)=21 ≥ 20 ✓, e_(3)=12 < 13.33 ✗ → k* = 2.
        assert_eq!(rejected, vec![0, 1]);

        // A weaker tail cannot drag in a stronger prefix decision.
        let log_e2 = [45.0f64.ln(), 5.0f64.ln(), 2.0f64.ln(), 1.0f64.ln()];
        assert_eq!(e_benjamini_hochberg(&log_e2, 0.1), vec![0]);
    }

    #[test]
    fn split_likelihood_equal_impossibility_is_neutral_log_evidence() {
        let log_e = split_likelihood_log_e_value(f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert_eq!(log_e, 0.0);
        assert!(log_e.is_finite());

        let mut proc = EProcess::new();
        proc.absorb_log(log_e).unwrap();
        assert_eq!(proc.log_evidence(), 0.0);
        assert_eq!(proc.steps(), 1);
    }

    #[test]
    fn e_bh_orders_infinite_log_e_values_without_comparator_panic() {
        let log_e = [f64::NEG_INFINITY, f64::INFINITY, 45.0f64.ln(), 1.0f64.ln()];
        assert_eq!(e_benjamini_hochberg(&log_e, 0.1), vec![1, 2]);
    }

    #[test]
    #[should_panic(expected = "e-BH log e-values must not be NaN")]
    fn e_bh_rejects_nan_log_e_values_before_sorting() {
        let log_e = [10.0f64.ln(), f64::NAN];
        let _ = e_benjamini_hochberg(&log_e, 0.1);
    }

    #[test]
    fn e_process_absorb_log_rejects_undefined_log_products() {
        let mut proc = EProcess::new();
        assert!(proc.absorb_log(f64::NAN).is_err());

        proc.absorb_log(f64::INFINITY).unwrap();
        assert!(proc.absorb_log(f64::NEG_INFINITY).is_err());
        assert_eq!(proc.log_evidence(), f64::INFINITY);
        assert_eq!(proc.steps(), 1);
    }

    /// Ville-style sanity: under H0 (simulated fair e-values from a
    /// likelihood ratio of identical Gaussians), the e-process crosses
    /// 1/α rarely; under a true alternative it crosses fast and the
    /// crossing is PERMANENT (running-sup semantics).
    #[test]
    fn e_process_crossing_is_permanent_and_directional() {
        // Deterministic "stream": per-batch log-LR of N(μ,1) vs N(0,1)
        // evaluated at x drawn from the alternative: log e = μ x − μ²/2.
        // Use a fixed quasi-random sequence; no RNG state needed.
        let mu = 0.6f64;
        let mut proc_alt = EProcess::new();
        let mut crossed_at: Option<usize> = None;
        for t in 0..200 {
            // x_t ~ alternative-ish deterministic surrogate around μ
            let x = mu + 0.9 * ((t as f64 * 0.7321).sin());
            proc_alt.absorb_log(mu * x - 0.5 * mu * mu).unwrap();
            if proc_alt.rejects_at(0.05) && crossed_at.is_none() {
                crossed_at = Some(t);
            }
        }
        let t_cross = crossed_at.expect("true alternative must cross 1/α");
        assert!(t_cross < 100, "evidence should accumulate quickly");
        // Permanence: rejection holds at the end even if late evidence dips.
        assert!(proc_alt.rejects_at(0.05));

        // Null stream: x centered at 0 → expected log e = −μ²/2 < 0.
        let mut proc_null = EProcess::new();
        for t in 0..200 {
            let x = 0.9 * ((t as f64 * 0.7321).sin());
            proc_null.absorb_log(mu * x - 0.5 * mu * mu).unwrap();
        }
        assert!(
            !proc_null.rejects_at(0.05),
            "null stream must not accumulate evidence (log E = {:.3})",
            proc_null.log_evidence()
        );
    }

    /// The design rule selects discrimination, not raw effect.
    #[test]
    fn probe_selection_prefers_discrimination_over_impact() {
        let fisher = array![[2.0, 0.0], [0.0, 0.5]];
        let probes = vec![
            // Huge effect, but both hypotheses predict it identically.
            CandidateProbe {
                delta: array![1.0, 0.0],
                predicted_mean_null: array![10.0, 10.0],
                predicted_mean_alt: array![10.0, 10.0],
            },
            // Modest effect, hypotheses disagree along the informative axis.
            CandidateProbe {
                delta: array![0.0, 1.0],
                predicted_mean_null: array![0.0, 0.0],
                predicted_mean_alt: array![1.0, 0.2],
            },
        ];
        let (idx, growth) =
            select_probe_by_expected_evidence(&probes, &fisher).expect("a probe discriminates");
        assert_eq!(idx, 1);
        // ½·(1,0.2)ᵀ diag(2,0.5) (1,0.2) = ½·(2 + 0.02) = 1.01 nats/obs.
        assert!((growth - 1.01).abs() < 1e-12);
        // Budget: ~3 observations to certify at α=0.05.
        let budget = expected_resolution_budget(0.05, growth).expect("budget");
        assert!(budget > 2.0 && budget < 4.0);
    }

    /// The birth gate certifies under a true alternative, stays contested
    /// under the null, and never emits anything but those two verdicts.
    #[test]
    fn birth_gate_certifies_alternative_and_demotes_never_rejects() {
        let mut gate = AtomBirthGate::new(0.05).expect("valid alpha");
        // Strong shards: alternative beats the honest null sup by 1 nat each.
        for _ in 0..5 {
            gate.absorb_shard(-100.0, -101.0);
        }
        match gate.verdict() {
            GateVerdict::Certified { log_e } => assert!((log_e - 5.0).abs() < 1e-12),
            v => panic!("5 nats must certify at α=0.05, got {v:?}"),
        }
        // Permanence: a later evidence retreat cannot un-certify.
        gate.absorb_shard(-110.0, -100.0);
        assert!(matches!(gate.verdict(), GateVerdict::Certified { .. }));

        // Null-ish stream: the prefit alternative loses to the on-shard sup
        // (it must, on average — the sup is fit on the eval shard itself).
        let mut null_gate = AtomBirthGate::new(0.05).expect("valid alpha");
        for _ in 0..50 {
            null_gate.absorb_shard(-100.3, -100.0);
        }
        match null_gate.verdict() {
            GateVerdict::Contested { log_e } => assert!(log_e < 0.0),
            v => panic!("null stream must stay contested, got {v:?}"),
        }
        assert!(AtomBirthGate::new(0.0).is_err());
        assert!(AtomBirthGate::new(1.0).is_err());
    }

    /// Ledger: idempotent registration preserves evidence; the certificate
    /// splits confirmed/contested by e-BH and the entry list reproduces it.
    #[test]
    fn ledger_certificate_splits_confirmed_and_contested() {
        let mut ledger = StructureLedger::new();
        let a0 = ledger.register(ClaimKind::AtomExists { atom: 0 });
        let a1 = ledger.register(ClaimKind::AtomExists { atom: 1 });
        let edge = ledger.register(ClaimKind::BindingEdge { a: 0, b: 1 });

        // m = 3, α = 0.1 → e-BH thresholds m/(αk) = 30, 15, 10.
        ledger.absorb_log(a0, 40.0f64.ln()).unwrap();
        ledger.absorb_log(a1, 20.0f64.ln()).unwrap();
        ledger.absorb_log(edge, 2.0f64.ln()).unwrap();

        // Re-registering must return the same slot with evidence intact.
        let a0_again = ledger.register(ClaimKind::AtomExists { atom: 0 });
        assert_eq!(a0_again, a0);
        assert_eq!(ledger.claims()[a0].evidence.steps(), 1);

        let cert = ledger.certify(0.1);
        // e_(1)=40 ≥ 30 ✓, e_(2)=20 ≥ 15 ✓, e_(3)=2 < 10 ✗ → atoms confirmed,
        // the binding edge stays contested.
        let confirmed: Vec<&ClaimKind> = cert.confirmed().map(|e| &e.kind).collect();
        assert_eq!(confirmed.len(), 2);
        assert!(confirmed.contains(&&ClaimKind::AtomExists { atom: 0 }));
        assert!(confirmed.contains(&&ClaimKind::AtomExists { atom: 1 }));
        let contested: Vec<&CertificateEntry> = cert.contested().collect();
        assert_eq!(contested.len(), 1);
        assert_eq!(contested[0].kind, ClaimKind::BindingEdge { a: 0, b: 1 });

        assert!(ledger.absorb_log(99, 0.0).is_err());
    }

    /// Resumability: a serialized ledger reloads with its evidence and
    /// keeps absorbing — the #973 shard contract.
    #[test]
    fn ledger_evidence_resumes_across_serialization() {
        let mut ledger = StructureLedger::new();
        let idx = ledger.register(ClaimKind::GeometryKind {
            atom: 3,
            kind: "circle".to_string(),
        });
        ledger.absorb_log(idx, 1.25).unwrap();

        let persisted = serde_json::to_string(&ledger).expect("serialize ledger");
        let mut resumed: StructureLedger =
            serde_json::from_str(&persisted).expect("deserialize ledger");
        assert_eq!(resumed.claims()[idx].evidence.steps(), 1);

        resumed.absorb_log(idx, 0.75).unwrap();
        let log_e = resumed.claims()[idx].evidence.log_evidence();
        assert!((log_e - 2.0).abs() < 1e-12);
    }

    /// The probe plan discounts the remaining budget by evidence already
    /// banked, and floors at zero once the claim is across the line.
    #[test]
    fn probe_plan_discounts_remaining_budget_by_current_evidence() {
        let fisher = array![[2.0, 0.0], [0.0, 0.5]];
        let probes = vec![CandidateProbe {
            delta: array![0.0, 1.0],
            predicted_mean_null: array![0.0, 0.0],
            predicted_mean_alt: array![1.0, 0.2],
        }];
        // growth = 1.01 nats/obs (checked above); α=0.05 → need ln(20) ≈ 3.0 nats.
        let from_zero = plan_probe_for_contested_claim(&probes, &fisher, 0.05, 0.0).expect("plan");
        assert_eq!(from_zero.probe, 0);
        assert!((from_zero.budget_remaining - from_zero.budget_from_scratch).abs() < 1e-12);

        let halfway = plan_probe_for_contested_claim(&probes, &fisher, 0.05, 1.5).expect("plan");
        assert!(halfway.budget_remaining < from_zero.budget_remaining);
        assert!((halfway.budget_remaining - (-(0.05f64.ln()) - 1.5) / 1.01).abs() < 1e-12);

        let across = plan_probe_for_contested_claim(&probes, &fisher, 0.05, 10.0).expect("plan");
        assert_eq!(across.budget_remaining, 0.0);

        // No discriminating probe → no plan (undecidable by steering).
        let blind = vec![CandidateProbe {
            delta: array![1.0, 0.0],
            predicted_mean_null: array![5.0, 5.0],
            predicted_mean_alt: array![5.0, 5.0],
        }];
        assert!(plan_probe_for_contested_claim(&blind, &fisher, 0.05, 0.0).is_none());
    }

    /// The p→e calibrator on hand-checkable values, including its edges.
    #[test]
    fn p_to_e_calibrator_hand_values() {
        // e(p) = ½ p^{−1/2}: p = 1 → e = 0.5; p = 0.04 → e = 2.5; p = 1e-4 → e = 50.
        assert!((log_e_from_p_calibrator(1.0).unwrap() - 0.5f64.ln()).abs() < 1e-12);
        assert!((log_e_from_p_calibrator(0.04).unwrap() - 2.5f64.ln()).abs() < 1e-12);
        assert!((log_e_from_p_calibrator(1e-4).unwrap() - 50.0f64.ln()).abs() < 1e-12);
        assert!(log_e_from_p_calibrator(0.0).is_err());
        assert!(log_e_from_p_calibrator(1.5).is_err());
        assert!(log_e_from_p_calibrator(f64::NAN).is_err());
    }

    /// POWER STUDY, null side: the heuristic gate every dictionary paper
    /// runs — "accept the K+1-th atom the first time the cumulative
    /// likelihood ratio shows improvement" — versus the e-gate, on a
    /// family of NULL streams peeked at after every shard. The per-shard
    /// log-LR is `μ x_t − μ²/2` with `x_t = A sin(ω t + φ)` (a
    /// deterministic null surrogate: mean drift −μ²/2 < 0, bounded
    /// fluctuation). The naive gate's false-accept mechanism is exactly
    /// optional stopping: any phase whose partial sums wander above zero
    /// at ANY peek accepts a nonexistent atom. The e-gate needs
    /// log(1/α) ≈ 3.0 nats, and the partial-sum fluctuation is bounded by
    /// `μ·A/sin(ω/2) ≈ 1.51` nats (Dirichlet-kernel bound) BEFORE the
    /// negative drift — so it can never certify on any phase, which is
    /// Ville's inequality made concrete.
    #[test]
    fn power_study_null_naive_peeking_gate_false_accepts_e_gate_never() {
        let mu = 0.6f64;
        let amp = 0.9f64;
        let omega = 0.7321f64;
        let n_phases = 60usize;
        let n_shards = 200usize;

        let mut naive_false_accepts = 0usize;
        let mut e_gate_false_accepts = 0usize;
        for k in 0..n_phases {
            let phase = 2.0 * std::f64::consts::PI * (k as f64) / (n_phases as f64);
            let mut gate = AtomBirthGate::new(0.05).expect("alpha");
            let mut cum_log_lr = 0.0f64;
            let mut naive_accepted = false;
            for t in 0..n_shards {
                let x = amp * ((t as f64) * omega + phase).sin();
                let log_lr = mu * x - 0.5 * mu * mu;
                cum_log_lr += log_lr;
                // The broken test: peek, accept on any improvement.
                if cum_log_lr > 0.0 {
                    naive_accepted = true;
                }
                gate.absorb_shard(log_lr, 0.0);
            }
            if naive_accepted {
                naive_false_accepts += 1;
            }
            if matches!(gate.verdict(), GateVerdict::Certified { .. }) {
                e_gate_false_accepts += 1;
            }
        }
        // The naive gate false-accepts on a large fraction of null phases
        // (any phase with early-positive partial sums); the e-gate on none.
        assert!(
            naive_false_accepts >= n_phases / 3,
            "the peeking gate should false-accept often under the null \
             (got {naive_false_accepts}/{n_phases})"
        );
        assert_eq!(
            e_gate_false_accepts, 0,
            "the e-gate must never certify under the null"
        );
    }

    /// POWER STUDY, alternative side, through the orchestration harness:
    /// a planted K+1-th atom worth 0.5 nats/shard certifies in
    /// ⌈log(1/α)/0.5⌉ = 6 shards — matching the design-time
    /// `expected_resolution_budget` — after which the gate stops absorbing
    /// (the crossing is permanent) while the alternative keeps refitting
    /// on the remaining shards.
    #[test]
    fn power_study_planted_atom_certifies_at_the_predicted_budget() {
        let growth = 0.5f64;
        let (gate, alt_state) = run_atom_birth_gate(
            0.05,
            0usize, // alt state = number of shards folded into the fit
            0..20usize,
            |_, _| -99.5, // prefit alternative log-lik on the shard
            |_| -100.0,   // honest null sup on the shard
            |folded, _| folded + 1,
        )
        .expect("valid alpha");

        match gate.verdict() {
            GateVerdict::Certified { log_e } => assert!((log_e - 3.0).abs() < 1e-12),
            v => panic!("planted atom must certify, got {v:?}"),
        }
        // Realized time-to-certification == the design-time budget, rounded up.
        let budget = expected_resolution_budget(0.05, growth).expect("budget");
        assert_eq!(gate.test.process.steps(), budget.ceil() as usize);
        assert_eq!(gate.test.process.steps(), 6);
        // The alternative state saw the whole stream despite early stopping.
        assert_eq!(alt_state, 20);
    }

    /// Work-plan step 4, closed end-to-end: a contested claim gets a probe
    /// plan, the probe's realized outcomes are scored under both FROZEN
    /// hypotheses via [`StructureLedger::absorb_probe_outcome`], and the
    /// banked evidence flips the claim to confirmed within a small multiple
    /// of the plan's predicted resolution budget. Outcome noise is a
    /// deterministic bounded surrogate (zero-mean sinusoid), so under the
    /// true alternative each probe's expected log-growth is exactly the
    /// design value.
    #[test]
    fn design_loop_resolves_contested_claim_within_predicted_budget() {
        let mut ledger = StructureLedger::new();
        let idx = ledger.register(ClaimKind::GeometryKind {
            atom: 0,
            kind: "circle".to_string(),
        });

        // Local Gaussian output model, unit-isotropic noise in the
        // Fisher-whitened coordinates: per-observation expected log-growth
        // under H1 is exactly the planned ½‖μ₁−μ₀‖²_F.
        let fisher = array![[1.0, 0.0], [0.0, 1.0]];
        let mu0 = array![0.0, 0.0];
        let mu1 = array![1.2, 0.5];
        let probes = vec![CandidateProbe {
            delta: array![0.0, 1.0],
            predicted_mean_null: mu0.clone(),
            predicted_mean_alt: mu1.clone(),
        }];
        let alpha = 0.05;
        let plan = plan_probe_for_contested_claim(&probes, &fisher, alpha, 0.0).expect("plan");
        assert_eq!(plan.probe, 0);
        // ½‖μ₁−μ₀‖² = ½(1.44 + 0.25) = 0.845 nats/obs; ln 20 ≈ 3.0 ⇒ ~3.6 obs.
        assert!((plan.expected_log_growth - 0.845).abs() < 1e-12);
        let budget = plan.budget_remaining.ceil().max(1.0) as usize;

        // Run the probe loop: outcomes realized under the TRUE alternative
        // (mean μ₁ plus bounded zero-mean fluctuation); both hypotheses'
        // densities were frozen above, before any outcome existed.
        let mut observations = 0usize;
        while !ledger.claims()[idx].evidence.rejects_at(alpha) {
            observations += 1;
            assert!(
                observations <= 4 * budget,
                "claim must resolve within a small multiple of the predicted \
                 budget {budget}; still contested after {observations} probes"
            );
            let t = observations as f64;
            let eps0 = 0.8 * (t * 0.7321).sin();
            let eps1 = 0.8 * (t * 1.1173).cos();
            let y = array![mu1[0] + eps0, mu1[1] + eps1];
            // Unit-Gaussian log-densities under each frozen hypothesis; the
            // shared normalizer cancels in the ratio.
            let d1 = &y - &mu1;
            let d0 = &y - &mu0;
            ledger
                .absorb_probe_outcome(idx, -0.5 * d1.dot(&d1), -0.5 * d0.dot(&d0))
                .expect("absorb");
        }
        let cert = ledger.certify(alpha);
        assert!(
            cert.confirmed()
                .any(|e| matches!(e.kind, ClaimKind::GeometryKind { atom: 0, .. }))
        );
    }
}
