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

use crate::inference::steering::SteerPlan;
use crate::manifold::SaeManifoldTerm;
use gam_problem::RowMetric;
use gam_terms::inference::structure_evidence::{ClaimKind, ProbePlan, StructureLedger};

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
        if let Err(err) = ledger.absorb_probe_outcome(claim_idx, realized_nats, 0.0) {
            log::debug!("probe outcome for claim {claim_idx} was not absorbed: {err}");
        }
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
