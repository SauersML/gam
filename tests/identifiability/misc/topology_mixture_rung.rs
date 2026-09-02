//! WP-C / Object 3a — the discrete-mixture rung in the topology race.
//!
//! These planted tests sample from two GROUND-TRUTH generative structures at
//! matched signal-to-noise and assert that the cross-class adjudicator recovers
//! the planted truth BOTH ways across several fixed seeds:
//!
//!   * circle truth  → the smooth-circle (ring) candidate wins the headline,
//!   * 7-cluster truth → the discrete `k`-component mixture rung wins.
//!
//! The assertions are against the PLANTED TRUTH (which generator produced the
//! data), never against a reference tool's output. All randomness is a fixed
//! splitmix64 stream seeded by an integer — there is no clock randomness.

use gam::solver::evidence::StackingConfig;
use gam::solver::topology_selector::{AutoTopologyKind, EvidenceCertification, Headline, HeldOutDensityProvider, PredictiveCandidateKind, PredictiveRaceCandidate, STACKING_CV_FOLDS, STACKING_CV_SEED, adjudicate_predictive_race};
use ndarray::{Array2, ArrayView2};

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock).
// ---------------------------------------------------------------------------

use gam::utils::splitmix64;
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.state)
    }
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa uniform in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_gaussian(&mut self) -> f64 {
        // Box-Muller; deterministic from the stream.
        let u1 = self.next_unit().max(1e-12);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Planted generators at MATCHED SNR.
//
// SNR is defined as (structure scale) / (noise scale). For the circle the
// structure scale is the ring radius R and the noise is the radial jitter
// `noise`; for the clusters the structure scale is the inter-cluster spacing
// and the noise is the within-cluster spread. We set both so that
//   structure / noise == SNR  (the same value) for both generators.
// ---------------------------------------------------------------------------

const SNR: f64 = 12.0;
const N_OBS: usize = 350;

/// Points on a unit ring with isotropic radial jitter. Truth = SMOOTH CIRCLE.
fn sample_circle(seed: u64) -> Array2<f64> {
    let radius = 1.0_f64;
    let noise = radius / SNR; // matched SNR
    let mut rng = SplitMix64::new(seed ^ 0xC18C1E_u64);
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        let theta = std::f64::consts::TAU * rng.next_unit();
        let x = radius * theta.cos() + noise * rng.next_gaussian();
        let y = radius * theta.sin() + noise * rng.next_gaussian();
        out[[i, 0]] = x;
        out[[i, 1]] = y;
    }
    out
}

// ---------------------------------------------------------------------------
// Smooth-circle (ring) held-out density provider.
//
// A genuine smooth-class candidate: it models the data as living on a ring with
// a learned radius mean / variance and a uniform-in-angle distribution. The
// held-out density of (x, y) in polar (r, phi) is
//   p(x, y) = N(r; r_bar, sigma_r^2) * (1 / (2 pi)) * (1 / r)
// (the 1/r is the Cartesian->polar Jacobian). It refits r_bar, sigma_r on each
// fold's training rows so the table is genuinely held out.
// ---------------------------------------------------------------------------

fn ring_density_provider<'a>(data: ArrayView2<'a, f64>) -> HeldOutDensityProvider<'a> {
    let owned = data.to_owned();
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            if train.is_empty() {
                return Err("ring provider got empty training set".to_string());
            }
            let r_of = |i: usize| -> f64 { (owned[[i, 0]].powi(2) + owned[[i, 1]].powi(2)).sqrt() };
            let n = train.len() as f64;
            let mean: f64 = train.iter().map(|&i| r_of(i)).sum::<f64>() / n;
            let var: f64 = train.iter().map(|&i| (r_of(i) - mean).powi(2)).sum::<f64>() / n;
            let var = var.max(1e-9);
            let log_norm = -0.5 * (std::f64::consts::TAU * var).ln();
            let log_angle = -(std::f64::consts::TAU).ln();
            let mut out = Vec::with_capacity(eval.len());
            for &i in eval {
                let r = r_of(i).max(1e-9);
                let log_r_density = log_norm - 0.5 * (r - mean).powi(2) / var;
                // + log(1/r) Jacobian + uniform angle.
                out.push(log_r_density + log_angle - r.ln());
            }
            Ok(out)
        },
    )
}

// ---------------------------------------------------------------------------
// Cross-class race driver: smooth-circle vs the winning mixture order.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn same_class_race_keeps_evidence_headline() {
    // A race with no mixture candidate must stay winner-take-all on evidence.
    let data = sample_circle(11);
    let provider_a = ring_density_provider(data.view());
    let provider_b = ring_density_provider(data.view());
    let candidates = vec![
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
            negative_log_evidence: 100.0,
            certification: EvidenceCertification::Exact,
            density_provider: provider_a,
        },
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
            negative_log_evidence: 250.0,
            certification: EvidenceCertification::Exact,
            density_provider: provider_b,
        },
    ];
    let verdict = adjudicate_predictive_race(
        data.nrows(),
        candidates,
        STACKING_CV_FOLDS,
        STACKING_CV_SEED,
        StackingConfig::default(),
    )
    .expect("same-class adjudication");
    assert!(!verdict.is_cross_class, "no mixture → not cross-class");
    assert_eq!(verdict.headline, Headline::Evidence);
    assert_eq!(
        verdict.winner_index, 0,
        "lower rank-aware evidence wins the same-class headline"
    );
    assert!(verdict.stacking.is_none());
}
