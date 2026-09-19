//! `Certificate` implementations for the gam-solve-tier certificate zoo (task #16; descended #1521).
//!
//! This module holds the `impl Certificate for …` blocks for the gam-solve-owned
//! certificate types (`OuterCriterionCertificate`,
//! [`CoresetCertificate`](crate::row_sampling_measure::CoresetCertificate),
//! [`CollapseEvent`](crate::structure_search::CollapseEvent)).
//! They were relocated out of the
//! monolith root (`gam::inference::certificate_impls`) to satisfy the coherence
//! orphan rule: the `Certificate` trait now lives in the neutral `gam-problem`
//! crate and these types are owned here in `gam-solve`, so the impls must be
//! defined in the type's home crate. The bodies are byte-identical to the
//! monolith originals, so there remains exactly one decision rule per verdict.
//! (The gam-sae-owned certificate types — `ResidualGaugeReport`, `CertificateInputs`
//! — carry their own impls in `gam_sae::certificate_impls`.)

use crate::model_types::OuterCriterionCertificate;
use crate::row_sampling_measure::CoresetCertificate;
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

impl Certificate for OuterCriterionCertificate {
    fn claim(&self) -> Claim {
        Claim::new(
            "outer-optimality",
            concat!(
                "the returned outer optimum is analytically KKT-stationary and ",
                "its available exact curvature is not indefinite",
            ),
        )
    }

    fn evidence(&self) -> Evidence {
        let mut e = Evidence::new();
        put_finite(
            &mut e,
            "stationarity_raw_norm",
            self.stationarity.raw_norm(),
        );
        put_finite(
            &mut e,
            "stationarity_projected_norm",
            self.stationarity.projected_norm(),
        );
        put_finite(&mut e, "stationarity_bound", self.stationarity.bound());
        // The bound never appears without the standard that produced it: across
        // this subsystem the bound alone spans nine orders and names nothing
        // (#2458/#2530).
        let rung = self.stationarity.rung();
        e.insert("stationarity_rung", rung.label.clone().into());
        e.insert("stationarity_rung_derived", rung.derived_standard.into());
        // `kind_label` names all three routes, so an AsymptoteRail certificate
        // is never reported as "analytic_gradient" in the map a reader consults
        // to find out which route ran.
        e.insert("stationarity_kind", self.stationarity.kind_label().into());
        e.insert(
            "hessian_psd",
            match self.hessian_psd() {
                Some(psd) => psd.into(),
                None => "n/a".into(),
            },
        );
        // The floor's verdict rides BESIDE the raw measurement, never over it,
        // so a reader can see both which question was asked and how much
        // negative curvature the floor was able to absorb.
        if let Some(clearance) = self.curvature_floor {
            e.insert("curvature_floor_cleared", clearance.cleared.into());
            put_finite(
                &mut e,
                "curvature_interior_min_eigenvalue",
                clearance.interior_min_eigenvalue,
            );
            put_finite(&mut e, "curvature_gradient_floor", clearance.gradient_floor);
        }
        e.insert("lambdas_railed_count", self.lambdas_railed.len().into());
        // #2954: the Newton steps the mint took before judging, so a certified
        // point that is not where the search stopped says so, and by how much.
        if let Some(polish) = self.newton_polish.as_ref() {
            e.insert("newton_polish_steps", polish.decreases.len().into());
            e.insert("newton_polish_step_budget", polish.step_budget.into());
            put_finite(
                &mut e,
                "newton_polish_lambda_sq_before",
                polish.lambda_sq_before,
            );
            put_finite(
                &mut e,
                "newton_polish_lambda_sq_after",
                polish.lambda_sq_after,
            );
            e.insert(
                "newton_polish_decreases",
                format!("{:?}", polish.decreases).into(),
            );
            e.insert("newton_polish_rails", polish.rails.len().into());
            if !polish.rails.is_empty() {
                e.insert(
                    "newton_polish_railed_coordinates",
                    format!(
                        "{:?}",
                        polish
                            .rails
                            .iter()
                            .map(|rail| rail.index)
                            .collect::<Vec<_>>()
                    )
                    .into(),
                );
            }
        }
        e.insert("stationary", self.is_stationary().into());
        // Both, deliberately (#2578): the boolean is the published contract and
        // stays byte-compatible, and the verdict beside it says WHICH of the
        // three states produced it — so a reader can tell "measured and
        // admissible" from "nothing was measured", which the boolean alone
        // cannot express.
        e.insert("curvature_admissible", self.curvature_not_refused().into());
        e.insert(
            "curvature_verdict",
            self.curvature_verdict().to_string().into(),
        );
        e.insert("summary", self.summary().into());
        e
    }

    fn verdict(&self) -> Verdict {
        if self.certifies() {
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

// ── 3. Structure-search collapse event ───────────────────────────────────────

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

    #[test]
    fn criterion_stationarity_and_curvature_control_verdict() {
        let clean = OuterCriterionCertificate {
            stationarity: crate::model_types::OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 1e-8,
                projected_grad_norm: 1e-8,
                bound: 1e-6,
                rung: gam_problem::StationarityRung {
                    label: "solver-band",
                    derived_standard: false,
                }
                .into(),
            },
            curvature: crate::model_types::CurvatureEvidence::Measured { psd: true },
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
        };
        assert_eq!(clean.verdict(), Verdict::Certified);
        assert!(clean.verdict().is_certified());

        let nonstationary = OuterCriterionCertificate {
            stationarity: crate::model_types::OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 1e-2,
                projected_grad_norm: 1e-2,
                bound: 1e-6,
                rung: gam_problem::StationarityRung {
                    label: "solver-band",
                    derived_standard: false,
                }
                .into(),
            },
            ..clean
        };
        assert_eq!(nonstationary.verdict(), Verdict::Insufficient);
        assert!(!nonstationary.verdict().is_certified());
        // The claim id is stable and the summary rides the evidence.
        assert_eq!(nonstationary.claim().id, "outer-optimality");
        assert!(nonstationary.evidence().contains_key("summary"));
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

}
