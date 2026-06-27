//! [`Certificate`] implementations and margin-resolved [`Verdict`] mappings for
//! the gam-solve-tier certificate zoo (task #16; descended #1521).
//!
//! Two concerns live here, both gam-solve-tier: (a) the `impl Certificate for …`
//! blocks for the gam-solve-owned certificate types ([`CriterionCertificate`],
//! [`CoresetCertificate`](crate::row_sampling_measure::CoresetCertificate),
//! [`LogdetEnclosure`], [`CollapseEvent`](crate::structure_search::CollapseEvent)),
//! and (b) the two pure margin-resolution helpers, whose only inputs are
//! gam-solve-tier types ([`LogdetEnclosure`]/[`MarginVerdict`] and
//! [`CoresetMarginVerdict`](crate::row_sampling_measure::CoresetMarginVerdict))
//! plus the contracted-down [`Verdict`] ladder. Both were relocated out of the
//! monolith root (`gam::inference::certificate_impls`) to satisfy the coherence
//! orphan rule: the [`Certificate`] trait now lives in the neutral `gam-problem`
//! crate and these types are owned here in `gam-solve`, so the impls must be
//! defined in the type's home crate. The bodies are byte-identical to the
//! monolith originals, so there remains exactly one decision rule per verdict.
//! (The gam-sae-owned certificate types — `EncodeResult`, `ResidualGaugeReport`,
//! `CertificateInputs` — carry their own impls in `gam_sae::certificate_impls`.)

use crate::logdet_bounds::{LogdetEnclosure, MarginVerdict};
use crate::model_types::CriterionCertificate;
use crate::row_sampling_measure::{CoresetCertificate, CoresetMarginVerdict};
use crate::structure_search::{CollapseAction, CollapseEvent};
use gam_problem::topology_certificates::{Certificate, Claim, Evidence, Verdict};

/// Helper: insert a scalar only when finite, else record it as text "n/a" so the
/// evidence is explicit about a missing quantity (never a silent 0.0).
fn put_finite(evidence: &mut Evidence, key: &'static str, value: f64) {
    if value.is_finite() {
        evidence.insert(key, value.into());
    } else {
        evidence.insert(key, "n/a".into());
    }
}

// ── 1. Outer-optimum first-order self-audit (#931/#934) ──────────────────────

impl Certificate for CriterionCertificate {
    fn claim(&self) -> Claim {
        Claim::new(
            "outer-optimality",
            concat!(
                "the returned outer optimum is a genuine stationary point: the ", // fd-ok: FD-audit certificate, not in math path
                "analytic gradient agrees with the finite-difference of the criterion ", // fd-ok: FD-audit certificate, not in math path
                "value, the final Hessian is not indefinite, and no smoothing ",
                "coordinate is railed at a box bound",
            ),
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        put_finite(&mut e, "grad_norm", self.grad_norm);
        put_finite(&mut e, "analytic_directional", self.analytic_directional);
        put_finite(&mut e, "fd_directional", self.fd_directional); // fd-ok: FD-audit certificate, not in math path
        put_finite(&mut e, "fd_error", self.fd_error); // fd-ok: FD-audit certificate, not in math path
        put_finite(&mut e, "agreement_z", self.agreement_z);
        put_finite(&mut e, "fd_step", self.fd_step); // fd-ok: FD-audit certificate, not in math path
        e.insert(
            "hessian_pd",
            match self.hessian_pd {
                Some(pd) => pd.into(),
                None => "n/a".into(),
            },
        );
        e.insert("lambdas_railed_count", self.lambdas_railed.len().into());
        e.insert(
            "first_order_consistent",
            self.first_order_consistent().into(),
        );
        e.insert("summary", self.summary().into());
        e
    }

    fn verdict(&self) -> Verdict {
        // `is_clean()` is the unchanged decision rule: gradient↔objective
        // consistent, no definiteness failure, no railed coordinate. A desync
        // does not make the evidence absent — it is present and says "not
        // clean" — so the verdict is `Insufficient`, never `Unavailable`.
        if self.is_clean() {
            Verdict::Certified
        } else {
            Verdict::Insufficient
        }
    }
}

// ── 2. Sensitivity-coreset error budget ──────────────────────────────────────

impl Certificate for CoresetCertificate {
    fn claim(&self) -> Claim {
        Claim::new(
            "coreset-budget",
            "the selected row coreset reproduces the full-corpus evidence within \
             a certified spectral + likelihood error budget; a race decision \
             inherits the full-corpus verdict only when its margin clears this \
             budget",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        put_finite(&mut e, "eps_spectral", self.eps_spectral);
        put_finite(&mut e, "eps_likelihood", self.eps_likelihood);
        e.insert("dim_effective", self.dim_effective.into());
        e.insert("n_selected", self.n_selected.into());
        put_finite(&mut e, "logdet_error_bound", self.logdet_error_bound());
        put_finite(&mut e, "race_transfer_margin", self.race_transfer_margin());
        e
    }

    fn verdict(&self) -> Verdict {
        // A coreset certificate is a transfer BUDGET, not a standalone decision:
        // it certifies a race verdict only once a consumer supplies a decision
        // margin that clears `race_transfer_margin`. With no consumer margin in
        // hand, the conservative standalone verdict is `Insufficient` (the
        // budget is present, but nothing has been decided by it yet) when the
        // budget is finite, and `Unavailable` when it is not.
        if self.race_transfer_margin().is_finite() {
            Verdict::Insufficient
        } else {
            Verdict::Unavailable
        }
    }
}

/// Map a coreset race outcome (the certificate's own
/// [`CoresetCertificate::certify_margin`](crate::row_sampling_measure::CoresetCertificate::certify_margin)
/// rule, evaluated against a consumer's
/// `decision_margin`) onto the shared [`Verdict`] ladder. This is the
/// margin-resolved entry point a race consumer uses to obtain a unified verdict
/// without re-deriving the mapping.
pub fn coreset_race_verdict(verdict: CoresetMarginVerdict) -> Verdict {
    match verdict {
        CoresetMarginVerdict::Certified { .. } => Verdict::Certified,
        CoresetMarginVerdict::InsufficientMargin { .. } => Verdict::Insufficient,
    }
}

/// Verdict for an enclosure resolved against a concrete consumer
/// `decision_margin`, reusing [`LogdetEnclosure::decide_within_margin`].
pub fn enclosure_margin_verdict(enclosure: &LogdetEnclosure, decision_margin: f64) -> Verdict {
    match enclosure.decide_within_margin(decision_margin) {
        MarginVerdict::Decided { .. } => Verdict::Certified,
        MarginVerdict::InsufficientMargin { .. } => Verdict::Insufficient,
    }
}

// ── 3. Log-det enclosure ─────────────────────────────────────────────────────

impl Certificate for LogdetEnclosure {
    fn claim(&self) -> Claim {
        Claim::new(
            "logdet-enclosure",
            "the log-determinant is enclosed in a certified [lower, upper] \
             interval whose midpoint is interchangeable with the exact value for \
             any decision whose margin exceeds the enclosure gap",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        put_finite(&mut e, "block_diag_logdet", self.block_diag_logdet);
        put_finite(&mut e, "lower", self.lower);
        put_finite(&mut e, "upper", self.upper);
        put_finite(&mut e, "gap", self.gap());
        put_finite(&mut e, "rho", self.rho);
        put_finite(&mut e, "p2", self.p2);
        match self.p3 {
            Some(p3) => put_finite(&mut e, "p3", p3),
            None => {
                e.insert("p3", "n/a".into());
            }
        }
        e
    }

    fn verdict(&self) -> Verdict {
        // An enclosure on its own does not certify a decision — only a consumer
        // margin does (via `decide_within_margin`). The standalone verdict is
        // `Insufficient` when the enclosure is finite (evidence present, no
        // decision yet) and `Unavailable` when the bounds are non-finite.
        if self.lower.is_finite() && self.upper.is_finite() && self.gap().is_finite() {
            Verdict::Insufficient
        } else {
            Verdict::Unavailable
        }
    }
}

// ── 7. Structure-search collapse event ───────────────────────────────────────

impl Certificate for CollapseEvent {
    fn claim(&self) -> Claim {
        Claim::new(
            "structure-collapse",
            "an atom's active mass fell below the collapse floor during the joint \
             fit; the guard either reseeded it from a fresh basin or, once the \
             reseed budget was exhausted, recorded the collapse as the objective's \
             terminal verdict",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        e.insert("iteration", self.iteration.into());
        e.insert("atom", self.atom.into());
        put_finite(&mut e, "max_active_mass", self.max_active_mass);
        put_finite(&mut e, "floor", self.floor);
        e.insert(
            "action",
            match self.action {
                CollapseAction::Reseeded => "reseeded",
                CollapseAction::Terminal => "terminal",
            }
            .into(),
        );
        e
    }

    fn verdict(&self) -> Verdict {
        // A collapse event is, by definition, a guard FIRING — it never certifies
        // health. A `Reseeded` event is a recovered breach (`Insufficient`: the
        // breach happened but the fit continued); a `Terminal` event is the
        // objective's verdict that the collapse stands (`Unavailable`: the claim
        // of a healthy non-collapsed dictionary cannot be made at all).
        match self.action {
            CollapseAction::Reseeded => Verdict::Insufficient,
            CollapseAction::Terminal => Verdict::Unavailable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::topology_certificates::CertificateLedger;

    #[test]
    fn criterion_clean_certifies_desync_is_insufficient() {
        let clean = CriterionCertificate {
            grad_norm: 1e-8,
            analytic_directional: 1.0,
            fd_directional: 1.0,
            fd_error: 1e-6,
            agreement_z: 0.0,
            fd_step: 1e-4,
            hessian_pd: Some(true),
            lambdas_railed: Vec::new(),
        };
        assert_eq!(clean.verdict(), Verdict::Certified);
        assert!(clean.verdict().is_certified());

        let desync = CriterionCertificate {
            analytic_directional: 1.0,
            fd_directional: 5.0,
            fd_error: 1e-6,
            agreement_z: 4.0e6,
            ..clean
        };
        assert_eq!(desync.verdict(), Verdict::Insufficient);
        assert!(!desync.verdict().is_certified());
        // The claim id is stable and the summary rides the evidence.
        assert_eq!(desync.claim().id, "outer-optimality");
        assert!(desync.evidence().contains_key("summary"));
    }

    #[test]
    fn coreset_budget_alone_is_insufficient_but_decides_with_margin() {
        let cert = CoresetCertificate::new(0.1, 0.0, 4, 32).expect("coreset cert");
        assert_eq!(cert.verdict(), Verdict::Insufficient);
        // A margin below the budget stays insufficient; above it certifies.
        let req = cert.race_transfer_margin();
        assert_eq!(
            coreset_race_verdict(cert.certify_margin(req * 0.5)),
            Verdict::Insufficient
        );
        assert_eq!(
            coreset_race_verdict(cert.certify_margin(req * 2.0 + 1.0)),
            Verdict::Certified
        );
    }

    #[test]
    fn enclosure_certifies_only_when_margin_clears_gap() {
        let enc = LogdetEnclosure {
            block_diag_logdet: 10.0,
            lower: 9.9,
            upper: 10.1,
            rho: 0.3,
            p2: 0.01,
            p3: None,
        };
        assert_eq!(enc.verdict(), Verdict::Insufficient);
        // gap = 0.2; a margin of 0.5 > gap certifies; 0.1 < gap does not.
        assert_eq!(enclosure_margin_verdict(&enc, 0.5), Verdict::Certified);
        assert_eq!(enclosure_margin_verdict(&enc, 0.1), Verdict::Insufficient);
    }

    #[test]
    fn collapse_terminal_is_unavailable_reseeded_is_insufficient() {
        let reseeded = CollapseEvent {
            iteration: 3,
            atom: 1,
            max_active_mass: 1e-4,
            floor: 1e-3,
            action: CollapseAction::Reseeded,
        };
        assert_eq!(reseeded.verdict(), Verdict::Insufficient);
        let terminal = CollapseEvent {
            action: CollapseAction::Terminal,
            ..reseeded
        };
        assert_eq!(terminal.verdict(), Verdict::Unavailable);
    }

    #[test]
    fn ledger_rolls_up_to_weakest_member() {
        let mut ledger = CertificateLedger::new();
        let clean = CriterionCertificate {
            grad_norm: 1e-8,
            analytic_directional: 1.0,
            fd_directional: 1.0,
            fd_error: 1e-6,
            agreement_z: 0.0,
            fd_step: 1e-4,
            hessian_pd: Some(true),
            lambdas_railed: Vec::new(),
        };
        let cert = CoresetCertificate::new(0.1, 0.0, 4, 32).expect("coreset");
        ledger.record(&clean); // Certified
        ledger.record(&cert); // Insufficient
        assert_eq!(ledger.overall(), Verdict::Insufficient);
        assert_eq!(ledger.verdict_of("outer-optimality"), Verdict::Certified);
        assert_eq!(ledger.verdict_of("coreset-budget"), Verdict::Insufficient);
    }
}
