//! The unified certificate ledger (task #16): every certificate states a claim,
//! evidence, and a conservative verdict on the shared ladder, and the ledger
//! rolls them up so it can never read stronger than its weakest member.

use gam::inference::certificates::{Certificate, CertificateLedger, Verdict};

use gam::inference::row_measure::CoresetCertificate;
use gam::solver::logdet_bounds::LogdetEnclosure;
use gam::solver::rho_optimizer::{OuterCriterionCertificate, OuterStationarityCertificate};
use gam::solver::structure_search::{CollapseAction, CollapseEvent};
use gam::terms::sae::encode::EncodeResult;

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
fn encode_result_batch_verdict_is_all_or_flagged() {
    // An all-certified batch certifies; one flagged row makes it Insufficient;
    // an empty batch certifies nothing (Unavailable).
    let all_good = EncodeResult {
        coords: ndarray::Array2::zeros((3, 1)),
        certified: vec![true, true, true],
        encode_uncertified_count: 0,
    };
    assert_eq!(all_good.verdict(), Verdict::Certified);

    let one_flagged = EncodeResult {
        coords: ndarray::Array2::zeros((3, 1)),
        certified: vec![true, false, true],
        encode_uncertified_count: 1,
    };
    assert_eq!(one_flagged.verdict(), Verdict::Insufficient);

    let empty = EncodeResult {
        coords: ndarray::Array2::zeros((0, 1)),
        certified: vec![],
        encode_uncertified_count: 0,
    };
    assert_eq!(empty.verdict(), Verdict::Unavailable);
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
    let good_encode = EncodeResult {
        coords: ndarray::Array2::zeros((2, 1)),
        certified: vec![true, true],
        encode_uncertified_count: 0,
    };
    ledger.record(&good_encode);
    assert_eq!(ledger.overall(), Verdict::Certified);
    assert!(ledger.overall().is_certified());
}
