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

/// Running anytime-valid evidence against one null hypothesis, in log
/// space. Multiplicative absorption of conditionally-valid e-values;
/// Ville's inequality converts the running product into a sequential test
/// that survives optional stopping.
#[derive(Clone, Debug)]
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
        self.log_e += e_value.ln();
        self.steps += 1;
        if self.log_e > self.log_e_max {
            self.log_e_max = self.log_e;
        }
        Ok(())
    }

    /// Absorb a batch e-value supplied in log space (the only numerically
    /// honest interface for long streams).
    pub fn absorb_log(&mut self, log_e_value: f64) {
        self.log_e += log_e_value;
        self.steps += 1;
        if self.log_e > self.log_e_max {
            self.log_e_max = self.log_e;
        }
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
    log_lik_alternative_on_eval - log_lik_null_sup_on_eval
}

/// Sequential universal inference over a stream of batches with a
/// PREDICTABLE plug-in: at batch t the alternative parameters were fit
/// using only data before t, the null sup is taken on batch t, and the
/// per-batch ratios compound into an e-process. This is the streaming /
/// optional-stopping form the corpus-scale pipeline (#973 shards) needs —
/// evidence is resumable: serialize `EProcess`, keep absorbing on the next
/// shard.
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
    pub fn absorb_batch(
        &mut self,
        log_lik_alternative_prefit: f64,
        log_lik_null_sup_on_batch: f64,
    ) {
        self.process.absorb_log(split_likelihood_log_e_value(
            log_lik_alternative_prefit,
            log_lik_null_sup_on_batch,
        ));
    }
}

impl Default for PredictablePluginEProcess {
    fn default() -> Self {
        Self::new()
    }
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
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        log_e_values[b]
            .partial_cmp(&log_e_values[a])
            .expect("finite log e-values")
    });
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
            proc_alt.absorb_log(mu * x - 0.5 * mu * mu);
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
            proc_null.absorb_log(mu * x - 0.5 * mu * mu);
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
}
