//! [`Certificate`] implementations for the existing certificate zoo (task #16).
//!
//! Each implementation exposes an existing certificate type through the shared
//! [`Certificate`] contract WITHOUT changing its math: the [`Certificate::verdict`]
//! is defined in terms of the type's own (unchanged) decision rule — `is_clean`,
//! `is_certified`, `certify_margin`, `decide_within_margin`, the per-row `h ≤ ½`
//! check — so there remains exactly one source of truth for each verdict, now
//! surfaced uniformly. The conservative ladder
//! [`Verdict::Unavailable`] < [`Verdict::Insufficient`] < [`Verdict::Certified`]
//! guarantees a missing or below-margin certificate never reads as a pass.

use crate::inference::certificates::{Certificate, Claim, Evidence, Verdict};

use crate::inference::row_measure::{CoresetCertificate, CoresetMarginVerdict};
use crate::solver::logdet_bounds::{LogdetEnclosure, MarginVerdict};
use crate::solver::rho_optimizer::CriterionCertificate;
use crate::solver::structure_search::{CollapseAction, CollapseEvent};
use gam_sae::encode::EncodeResult;
use gam_sae::identifiability::ResidualGaugeReport;
use gam_sae::manifold::{CertificateInputs, GlobalOptimalityVerdict};

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
/// [`CoresetCertificate::certify_margin`] rule, evaluated against a consumer's
/// `decision_margin`) onto the shared [`Verdict`] ladder. This is the
/// margin-resolved entry point a race consumer uses to obtain a unified verdict
/// without re-deriving the mapping.
pub fn coreset_race_verdict(verdict: CoresetMarginVerdict) -> Verdict {
    match verdict {
        CoresetMarginVerdict::Certified { .. } => Verdict::Certified,
        CoresetMarginVerdict::InsufficientMargin { .. } => Verdict::Insufficient,
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

/// Verdict for an enclosure resolved against a concrete consumer
/// `decision_margin`, reusing [`LogdetEnclosure::decide_within_margin`].
pub fn enclosure_margin_verdict(enclosure: &LogdetEnclosure, decision_margin: f64) -> Verdict {
    match enclosure.decide_within_margin(decision_margin) {
        MarginVerdict::Decided { .. } => Verdict::Certified,
        MarginVerdict::InsufficientMargin { .. } => Verdict::Insufficient,
    }
}

// ── 4. Kantorovich encode atlas (#1010) ──────────────────────────────────────

impl Certificate for EncodeResult {
    fn claim(&self) -> Claim {
        Claim::new(
            "encode-atlas",
            "each encoded row carries a per-row Newton–Kantorovich certificate \
             (h = β·η·L ≤ ½ at the start point); certified rows converge \
             quadratically into the unique root, and uncertified rows are flagged \
             for the exact multi-start fallback — never silently encoded wrong",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        let n = self.certified.len();
        let certified = n - self.encode_uncertified_count;
        e.insert("rows", n.into());
        e.insert("certified_rows", certified.into());
        e.insert(
            "encode_uncertified_count",
            self.encode_uncertified_count.into(),
        );
        let frac = if n > 0 {
            certified as f64 / n as f64
        } else {
            f64::NAN
        };
        put_finite(&mut e, "certified_fraction", frac);
        e
    }

    fn verdict(&self) -> Verdict {
        // Conservative batch roll-up: the whole encode certifies only when EVERY
        // row certified. One flagged row makes the batch `Insufficient` (the
        // flagged rows must route to the exact fallback). An empty batch
        // certifies nothing → `Unavailable`.
        if self.certified.is_empty() {
            Verdict::Unavailable
        } else if self.encode_uncertified_count == 0 {
            Verdict::Certified
        } else {
            Verdict::Insufficient
        }
    }
}

// ── 5. Exact-orbit residual-gauge report (#980/#998/#1008) ───────────────────

impl Certificate for ResidualGaugeReport {
    fn claim(&self) -> Claim {
        Claim::new(
            "residual-gauge",
            "the fit is identified up to a named residual gauge group: every \
             generator was curvature-tested in the fit's own metric, the pinning \
             span rank is reported, and any surviving (unpinned) freedom is \
             enumerated rather than silently absorbed",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        e.insert(
            "metric_provenance",
            format!("{:?}", self.metric_provenance).into(),
        );
        e.insert("group_signature", self.group_signature().into());
        e.insert("pinning_rank", self.pinning_rank.into());
        e.insert("residual_gauge_dim", self.residual_gauge_dim.into());
        e.insert(
            "diffeomorphism_unpinned",
            self.diffeomorphism_unpinned.into(),
        );
        e.insert("generator_count", self.generators.len().into());
        match self.sym_f_trivial_under_output_fisher {
            Some(t) => e.insert("sym_f_trivial_under_output_fisher", t.into()),
            None => e.insert("sym_f_trivial_under_output_fisher", "n/a".into()),
        };
        e.insert("summary", self.summary.clone().into());
        e
    }

    fn verdict(&self) -> Verdict {
        // The report ALWAYS makes a claim once computed (every generator is
        // tested), so it is never `Unavailable` here. The conservative reading:
        // the identifiability claim is `Certified` when the model is pinned down
        // to a discrete (zero-dimensional) gauge group and the diffeomorphism
        // pin is active; a positive residual gauge dimension or an inactive
        // diffeomorphism pin is the escalation flag → `Insufficient`. Under
        // OutputFisher provenance a surviving atom-permutation is a certificate
        // violation → also `Insufficient`.
        let pinned = self.residual_gauge_dim == 0
            && !self.diffeomorphism_unpinned
            && self.sym_f_trivial_under_output_fisher != Some(false);
        if pinned {
            Verdict::Certified
        } else {
            Verdict::Insufficient
        }
    }
}

// ── 6. Dictionary incoherence / global optimality (#1008) ────────────────────

impl Certificate for CertificateInputs {
    fn claim(&self) -> Claim {
        Claim::new(
            "global-optimality",
            "the fitted dictionary's basin stationary point is the unique global \
             optimum up to the residual gauge group: a conservative sufficient \
             condition on mutual coherence, per-atom curvature, activity floors, \
             and reconstruction SNR holds with positive margin",
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        put_finite(&mut e, "mu_hat", self.mu_hat);
        put_finite(&mut e, "mean_activity_floor", self.mean_activity_floor);
        put_finite(&mut e, "peak_activity_floor", self.peak_activity_floor);
        put_finite(&mut e, "snr_proxy", self.snr_proxy);
        put_finite(&mut e, "dispersion", self.dispersion);
        put_finite(
            &mut e,
            "global_optimality_margin",
            self.global_optimality.margin(),
        );
        e.insert(
            "global_optimality",
            if self.global_optimality.is_certified() {
                "certified_global"
            } else {
                "uncertified"
            }
            .into(),
        );
        e.insert("atom_count", self.per_atom_mean_activity.len().into());
        e.insert("note", self.note.clone().into());
        e
    }

    fn verdict(&self) -> Verdict {
        // The unchanged decision rule is `GlobalOptimalityVerdict::is_certified`:
        // a `CertifiedGlobal { margin > 0 }` is never wrong (conservative
        // sufficient condition), an `Uncertified` is "cannot decide" — not
        // "non-unique" — so it maps to `Insufficient`, never a false pass.
        match self.global_optimality {
            GlobalOptimalityVerdict::CertifiedGlobal { .. } => Verdict::Certified,
            GlobalOptimalityVerdict::Uncertified { .. } => Verdict::Insufficient,
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
    use crate::inference::certificates::CertificateLedger;

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
