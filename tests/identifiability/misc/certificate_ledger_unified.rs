//! The unified certificate ledger (task #16): every certificate states a claim,
//! evidence, and a conservative verdict on the shared ladder, and the ledger
//! rolls them up so it can never read stronger than its weakest member.

use gam::inference::certificates::{Certificate, CertificateLedger, Verdict};

use gam::inference::row_measure::CoresetCertificate;
use gam::solver::logdet_bounds::LogdetEnclosure;
use gam::solver::rho_optimizer::{OuterCriterionCertificate, OuterStationarityCertificate};
use gam::solver::structure_search::{CollapseAction, CollapseEvent};

fn clean_criterion() -> OuterCriterionCertificate {
    OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::AnalyticGradient {
            grad_norm: 1e-9,
            projected_grad_norm: 1e-9,
            bound: 1e-6,
            rung: gam::solver::rho_optimizer::CertifiedRung {
                label: "solver-band".to_string(),
                derived_standard: false,
            },
        },
        curvature: gam::solver::rho_optimizer::CurvatureEvidence::Measured { psd: true },
        lambdas_railed: Vec::new(),
        railed_facts: Vec::new(),
        curvature_floor: None,
    }
}

#[test]
fn every_certificate_states_claim_evidence_and_conservative_verdict() {
    // Each certificate exposes a stable claim id, non-empty evidence, and a
    // verdict drawn from the conservative ladder.
    let crit = clean_criterion();
    assert_eq!(crit.claim().id, "outer-optimality");
    assert!(!crit.evidence().is_empty());
    assert_eq!(crit.verdict(), Verdict::Certified);

    let coreset = CoresetCertificate::new(0.1, 0.0, 4, 32).expect("coreset");
    assert_eq!(coreset.claim().id, "coreset-budget");
    // A budget alone certifies nothing — it is Insufficient until raced.
    assert_eq!(coreset.verdict(), Verdict::Insufficient);

    let enclosure = LogdetEnclosure {
        block_diag_logdet: 5.0,
        lower: 4.95,
        upper: 5.05,
        rho: 0.2,
        p2: 0.001,
        p3: None,
    };
    assert_eq!(enclosure.claim().id, "logdet-enclosure");
    assert_eq!(enclosure.verdict(), Verdict::Insufficient);
}

#[test]
fn collapse_terminal_makes_no_health_claim() {
    let terminal = CollapseEvent {
        iteration: 2,
        atom: 0,
        max_active_mass: 1e-5,
        floor: 1e-3,
        action: CollapseAction::Terminal,
    };
    // A terminal collapse cannot certify a healthy dictionary at all.
    assert_eq!(terminal.verdict(), Verdict::Unavailable);
}

#[test]
fn all_certified_ledger_rolls_up_certified() {
    let mut ledger = CertificateLedger::new();
    ledger.record(&clean_criterion());
    assert_eq!(ledger.overall(), Verdict::Certified);
    assert!(ledger.overall().is_certified());
}
