//! `ProbeRunner` — the closed loop between the steering primitive
//! ([`crate::inference::steering`]) and the anytime-valid structure-evidence
//! ledger ([`gam_terms::inference::structure_evidence`]).
//!
//! Both halves are implemented and tested in isolation; nothing wired them into
//! a runnable experiment loop. This is that bridge. The evidence module decides
//! WHICH claim to interrogate and HOW MUCH a probe should move the e-process
//! (`plan_probe_for_contested_claim`); the steering module turns a chosen latent
//! intervention into the on-manifold activation delta with its dosimetry and
//! validity radius (`steer_delta`). The runner picks the contested claim, asks
//! the planner for the experiment, realizes it through the steering primitive,
//! and feeds the realized dose back into the ledger as anytime-valid evidence.
//!
//! # The discrimination coordinate
//!
//! For a contested claim about atom `k`, the two hypotheses are "this atom
//! carries the steering move along its learned surface" (the alternative) versus
//! "it does not" (the null). The steering primitive measures, in **nats of
//! output-Fisher KL**, exactly how much behavioral effect a latent move along
//! atom `k` actually delivers — its `predicted_nats` dose. That dose IS the
//! expected per-observation log-growth of the deciding e-process under the
//! alternative (the module docs' "the SAME quadratic form the steering dosimetry
//! already computes, repurposed"). So each candidate latent move becomes a
//! one-dimensional [`CandidateProbe`] whose hypothesis disagreement, read through
//! the identity Fisher, reproduces the realized dose:
//! `½ (μ₁ − μ₀)ᵀ F (μ₁ − μ₀) = predicted_nats` with `μ₀ = 0`,
//! `μ₁ = √(2·predicted_nats)`, `F = [[1]]`. No fabricated metric — the real
//! steering dose flows through the real planner.

use ndarray::array;

use crate::inference::steering::{SteerPlan, steer_delta};
use crate::manifold::SaeManifoldTerm;
use gam_problem::RowMetric;
use gam_terms::inference::structure_evidence::{
    CandidateProbe, ClaimKind, ProbePlan, StructureLedger, plan_probe_for_contested_claim,
};

/// The level the contested-claim selection and budget are computed at. Fixed so
/// a probe can never be shopped across α after seeing the evidence (mirrors
/// [`gam_terms::inference::structure_evidence::AtomBirthGate`]'s construction-time α).
const PROBE_DESIGN_ALPHA: f64 = 0.05;

/// Latent step length each candidate probe moves the contested atom by, per
/// axis, away from its fitted representative coordinate. A modest step keeps the
/// move inside the surface's local regime; the steering primitive reports the
/// validity radius so an over-long move is flagged, never silently clipped.
const PROBE_LATENT_STEP: f64 = 0.5;

/// A planned probe carried alongside its realized steering intervention: the
/// experiment-design output (`plan`), the on-manifold activation delta and
/// dosimetry the steering primitive produced for it (`steer`), and the realized
/// behavioral dose in nats once it has been measured (`realized_nats`, `None`
/// until [`ProbeRunner::absorb`] banks it).
#[derive(Clone, Debug)]
pub struct RealizedProbe {
    /// The experiment plan for the most contested claim: which candidate probe,
    /// its expected per-observation log-growth, and the resolution budget.
    pub plan: ProbePlan,
    /// The realized steering intervention for the chosen candidate: the
    /// activation-space δ, predicted dose, validity radius, off-manifold guard.
    pub steer: SteerPlan,
    /// The realized behavioral dose in nats once observed, banked by
    /// [`ProbeRunner::absorb`] into the claim's e-process. `None` at design time.
    pub realized_nats: Option<f64>,
}

/// The closed-loop probe runner over one fitted SAE-manifold term and its
/// per-row output-Fisher metric.
pub struct ProbeRunner<'a> {
    /// The fitted term whose atoms the probes steer (read only).
    pub term: &'a SaeManifoldTerm,
    /// The per-row output-Fisher inner product the dose is measured through.
    pub metric: &'a RowMetric,
}

impl<'a> ProbeRunner<'a> {
    /// Design the next probe for the most contested claim in `ledger`.
    ///
    /// Picks the contested claim with the LEAST evidence (the one furthest from
    /// the `1/α` Ville threshold — the most in need of interrogation), reads the
    /// atom it concerns, and builds candidate steering moves along that atom's
    /// latent axes from its fitted representative coordinate. Each candidate is
    /// realized through [`steer_delta`] so its actual output-Fisher dose is
    /// known; the candidates are handed to [`plan_probe_for_contested_claim`],
    /// which selects the most discriminating one and converts the claim's
    /// current evidence into a remaining budget. The selected candidate's
    /// already-computed [`SteerPlan`] rides back in the result.
    pub fn design_next(&self, ledger: &StructureLedger) -> Result<RealizedProbe, String> {
        let (claim_idx, atom_k) = self.most_contested_atom_claim(ledger)?;
        let claim_kind = &ledger.claims()[claim_idx].kind;
        let current_log_e = ledger.claims()[claim_idx].evidence.current_e_value_log();

        let candidates = self.candidate_steers(atom_k)?;
        if candidates.is_empty() {
            return Err(format!(
                "ProbeRunner::design_next: atom {atom_k} (claim {claim_idx}) admits no steering \
                 candidate (zero latent dimension or no installed basis evaluator)"
            ));
        }

        // Convert each realized steering dose into a one-dimensional candidate
        // probe whose hypothesis disagreement, read through the identity Fisher,
        // equals that dose: ½(μ₁)² = predicted_nats ⇒ μ₁ = √(2·dose).
        let probes: Vec<CandidateProbe> = candidates
            .iter()
            .map(|steer| {
                let objective = self.probe_objective(claim_kind, steer);
                CandidateProbe {
                    delta: steer.delta.clone(),
                    predicted_mean_null: array![0.0],
                    predicted_mean_alt: array![(2.0 * objective).sqrt()],
                }
            })
            .collect();
        let fisher = array![[1.0]];

        let plan =
            plan_probe_for_contested_claim(&probes, &fisher, PROBE_DESIGN_ALPHA, current_log_e)
                .ok_or_else(|| {
                    format!(
                        "ProbeRunner::design_next: no candidate probe discriminates the hypotheses \
                     for atom {atom_k} (every reachable steering move delivers zero design \
                     objective — the claim is undecidable by steering, a finding not a failure)"
                    )
                })?;

        let steer = candidates.into_iter().nth(plan.probe).ok_or_else(|| {
            format!(
                "ProbeRunner::design_next: planner selected candidate {} of {} for atom \
                     {atom_k}",
                plan.probe,
                probes.len()
            )
        })?;

        Ok(RealizedProbe {
            plan,
            steer,
            realized_nats: None,
        })
    }

    /// Absorb a realized probe outcome, updating the ledger's evidence for the
    /// probe's claim.
    ///
    /// `realized_nats` is the dose the probe actually delivered when run (the
    /// observed output-Fisher KL of the steered response). Under the local
    /// Gaussian output model the alternative-vs-null log-likelihood ratio of one
    /// such observation is exactly that dose, so it routes straight into the
    /// claim's e-process through [`StructureLedger::absorb_probe_outcome`] as
    /// `log(alt) − log(null) = realized_nats − 0`. The contract its docstring
    /// requires — both hypotheses' densities frozen before the outcome — holds
    /// here: the steering plan (and thus both predictions) was fixed at design
    /// time, before any outcome existed.
    pub fn absorb(&self, ledger: &mut StructureLedger, probe: &RealizedProbe, realized_nats: f64) {
        let Ok((claim_idx, _)) = self.claim_for_steer(ledger, &probe.steer) else {
            return;
        };
        // The realized log-LR of one observation under the local Gaussian model
        // is the delivered dose; the null density contributes log-likelihood 0.
        ledger
            .absorb_probe_outcome(claim_idx, realized_nats, 0.0)
            .ok();
    }

    /// The ledger index and atom index of the contested claim with the LEAST
    /// accumulated evidence (the one furthest from certification). Only claims
    /// naming a concrete atom — [`ClaimKind::AtomExists`] and
    /// [`ClaimKind::GeometryKind`] — are steerable; binding-edge and custom
    /// claims have no single atom to drive and are skipped.
    fn most_contested_atom_claim(
        &self,
        ledger: &StructureLedger,
    ) -> Result<(usize, usize), String> {
        let mut best: Option<(usize, usize, f64)> = None;
        for (idx, claim) in ledger.claims().iter().enumerate() {
            let Some(atom_k) = steerable_atom(&claim.kind) else {
                continue;
            };
            if atom_k >= self.term.k_atoms() {
                continue;
            }
            let log_e = claim.evidence.current_e_value_log();
            match best {
                Some((_, _, best_log_e)) if best_log_e <= log_e => {}
                _ => best = Some((idx, atom_k, log_e)),
            }
        }
        best.map(|(idx, atom_k, _)| (idx, atom_k)).ok_or_else(|| {
            "ProbeRunner: ledger has no contested claim naming a steerable atom in this term"
                .to_string()
        })
    }

    /// Find the ledger claim a realized steer belongs to: the contested
    /// steerable claim whose atom matches the steer's atom index, least-evidence
    /// first (the same selection `design_next` used).
    fn claim_for_steer(
        &self,
        ledger: &StructureLedger,
        steer: &SteerPlan,
    ) -> Result<(usize, usize), String> {
        let mut best: Option<(usize, f64)> = None;
        for (idx, claim) in ledger.claims().iter().enumerate() {
            if steerable_atom(&claim.kind) != Some(steer.atom) {
                continue;
            }
            let log_e = claim.evidence.current_e_value_log();
            match best {
                Some((_, best_log_e)) if best_log_e <= log_e => {}
                _ => best = Some((idx, log_e)),
            }
        }
        best.map(|(idx, _)| (idx, steer.atom))
            .ok_or_else(|| format!("ProbeRunner: no claim names steered atom {}", steer.atom))
    }

    /// Design objective for a realized steering candidate. Existence claims use
    /// the behavioral KL dose directly. Geometry adjudication is different: the
    /// curvature signal scales as observed chart coverage to the fourth power,
    /// so a token spent at an already-covered coordinate buys essentially no new
    /// curvature-certification power. For geometry claims the candidate score is
    /// therefore the expected increase in `extent^4` of the observed chart, per
    /// probe token (all candidates here cost one token), rather than raw dose.
    fn probe_objective(&self, claim_kind: &ClaimKind, steer: &SteerPlan) -> f64 {
        match claim_kind {
            ClaimKind::GeometryKind { .. } => self.chart_extent_fourth_gain(steer),
            ClaimKind::AtomExists { .. }
            | ClaimKind::BindingEdge { .. }
            | ClaimKind::Custom { .. } => steer.predicted_nats.unwrap_or(0.0).max(0.0),
        }
    }

    /// Increase in the atom's observed latent chart extent to the fourth power
    /// after adding this probe endpoint. The extent is the root-mean-square axis
    /// range, which is zero only for a collapsed chart and grows when any latent
    /// direction obtains genuinely wider coverage.
    fn chart_extent_fourth_gain(&self, steer: &SteerPlan) -> f64 {
        let coords = self.term.assignment.coords[steer.atom].as_matrix();
        let d = steer.t_to.len();
        if d == 0 {
            return 0.0;
        }
        let mut extent_sq = 0.0_f64;
        let mut extended_extent_sq = 0.0_f64;
        for axis in 0..d {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for row in 0..coords.nrows() {
                let t = coords[[row, axis]];
                if t.is_finite() {
                    lo = lo.min(t);
                    hi = hi.max(t);
                }
            }
            if !lo.is_finite() || !hi.is_finite() {
                lo = steer.t_from[axis];
                hi = steer.t_from[axis];
            }
            let range = (hi - lo).max(0.0);
            extent_sq += range * range;

            let to = steer.t_to[axis];
            let extended_range = (hi.max(to) - lo.min(to)).max(0.0);
            extended_extent_sq += extended_range * extended_range;
        }
        let inv_d = 1.0 / d as f64;
        let extent_fourth = (extent_sq * inv_d).powi(2);
        let extended_extent_fourth = (extended_extent_sq * inv_d).powi(2);
        (extended_extent_fourth - extent_fourth).max(0.0)
    }

    /// Build the realized steering candidates for atom `atom_k`: a positive and
    /// negative [`PROBE_LATENT_STEP`] move along each latent axis, from the
    /// atom's fitted representative (most-active-row) coordinate. Each move is
    /// realized through [`steer_delta`] so the planner sees its true dose.
    fn candidate_steers(&self, atom_k: usize) -> Result<Vec<SteerPlan>, String> {
        let (metric_row, amplitude, t0) = self.representative_anchor(atom_k)?;
        let d = t0.len();
        let mut out = Vec::with_capacity(2 * d);
        for axis in 0..d {
            for &sign in &[1.0_f64, -1.0_f64] {
                let mut t_to = t0.clone();
                t_to[axis] += sign * PROBE_LATENT_STEP;
                out.push(steer_delta(
                    self.term,
                    self.metric,
                    atom_k,
                    metric_row,
                    amplitude,
                    &t0,
                    &t_to,
                )?);
            }
        }
        Ok(out)
    }

    /// Atom `atom_k`'s fitted latent coordinate at its most-active row — the
    /// representative operating point a probe perturbs away from, together with
    /// that exact row and its applied assignment amplitude.
    fn representative_anchor(&self, atom_k: usize) -> Result<(usize, f64, Vec<f64>), String> {
        let assignments = self.term.assignment.assignments();
        let n = self.term.n_obs();
        let mut best_row = 0usize;
        let mut best_mass = f64::NEG_INFINITY;
        for row in 0..n {
            let mass = assignments[[row, atom_k]];
            if mass > best_mass {
                best_mass = mass;
                best_row = row;
            }
        }
        if !(best_mass.is_finite() && best_mass > 0.0) {
            return Err(format!(
                "probe_runner: atom {atom_k} has no positive fitted assignment amplitude"
            ));
        }
        Ok((
            best_row,
            best_mass,
            self.term.assignment.coords[atom_k].row(best_row).to_vec(),
        ))
    }
}

/// The atom a structural claim is about, when it is one a single steering move
/// can interrogate. `None` for claims with no single steerable atom (binding
/// edges concern a pair; custom claims name no atom).
fn steerable_atom(kind: &ClaimKind) -> Option<usize> {
    match kind {
        ClaimKind::AtomExists { atom } | ClaimKind::GeometryKind { atom, .. } => Some(*atom),
        ClaimKind::BindingEdge { .. } | ClaimKind::Custom { .. } => None,
    }
}
