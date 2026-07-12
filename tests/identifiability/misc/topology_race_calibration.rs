//! #907 acceptance — **selection accuracy + calibrated BF/stacking magnitudes
//! under repeated draws** for the cross-class topology race.
//!
//! The in-tree acceptance criterion from the issue's rescope: planted truth
//! recovery (circle vs `k`-cluster at matched SNR) with *calibrated* decision
//! magnitudes — a large reported log-Bayes-factor or a decisive stacking
//! weight must actually be right, replicate after replicate. Selection
//! accuracy alone can hide overconfidence; this sweep pins both:
//!
//! * **Accuracy**: across `2 × N_REPLICATES` independent draws (half circle
//!   truth, half 7-cluster truth at matched SNR), the stacking-headline winner
//!   must match the planted generator every time.
//! * **Calibration of magnitudes**: every *decisive* call — held-out stacking
//!   weight above [`DECISIVE_STACKING_WEIGHT`] or rank-aware-evidence log-BF
//!   above [`DECISIVE_LOG_BF`] nats — must be correct. An overconfident
//!   adjudicator (big BF, wrong class) fails here even if some accuracy
//!   slack were allowed.
//! * **Direction of the corroborating evidence**: the Laplace-evidence
//!   difference must agree with the planted truth in the overwhelming
//!   majority of draws (it is the corroboration channel, allowed a small
//!   minority of near-tie inversions, never a systematic flip).
//!
//! Same generators and candidates as `tests/topology_mixture_rung.rs`
//! (matched SNR 12), swept over many more seeds.

use gam::solver::evidence::{GaussianMixtureConfig, StackingConfig};
use gam::solver::topology_selector::{
    AutoTopologyKind, EvidenceCertification, Headline, HeldOutDensityProvider, MIXTURE_K_LADDER,
    PredictiveCandidateKind, PredictiveRaceCandidate, STACKING_CV_FOLDS, STACKING_CV_SEED,
    adjudicate_predictive_race, fit_mixture_rung, mixture_density_provider,
};
use ndarray::{Array2, ArrayView2};

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock).
// ---------------------------------------------------------------------------

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-12);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Planted generators at MATCHED SNR (identical to the mixture-rung tests).
// ---------------------------------------------------------------------------

const SNR: f64 = 12.0;
const N_OBS: usize = 350;
const N_CLUSTERS: usize = 7;
/// Independent draws per truth class.
const N_REPLICATES: usize = 12;
/// A stacking call is *decisive* when the winner carries at least this much
/// held-out predictive mass.
const DECISIVE_STACKING_WEIGHT: f64 = 0.9;
/// A corroborating evidence call is *decisive* at this log-Bayes-factor (nats).
const DECISIVE_LOG_BF: f64 = 20.0;

fn sample_circle(seed: u64) -> Array2<f64> {
    let radius = 1.0_f64;
    let noise = radius / SNR;
    let mut rng = SplitMix64::new(seed ^ 0xC18C1E_u64);
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        let theta = std::f64::consts::TAU * rng.next_unit();
        out[[i, 0]] = radius * theta.cos() + noise * rng.next_gaussian();
        out[[i, 1]] = radius * theta.sin() + noise * rng.next_gaussian();
    }
    out
}

fn sample_clusters(seed: u64) -> Array2<f64> {
    let r_struct = 1.0_f64;
    let spacing = 2.0 * r_struct * (std::f64::consts::PI / N_CLUSTERS as f64).sin();
    let spread = spacing / SNR;
    let mut rng = SplitMix64::new(seed ^ 0xC1057E12_u64);
    let mut centers = Vec::with_capacity(N_CLUSTERS);
    for c in 0..N_CLUSTERS {
        let phi = std::f64::consts::TAU * c as f64 / N_CLUSTERS as f64;
        centers.push((r_struct * phi.cos(), r_struct * phi.sin()));
    }
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        let c = (rng.next_u64() as usize) % N_CLUSTERS;
        let (cx, cy) = centers[c];
        out[[i, 0]] = cx + spread * rng.next_gaussian();
        out[[i, 1]] = cy + spread * rng.next_gaussian();
    }
    out
}

// ---------------------------------------------------------------------------
// Ring candidate (held-out density + rank-aware evidence), as in the planted
// races.
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
                out.push(log_norm - 0.5 * (r - mean).powi(2) / var + log_angle - r.ln());
            }
            Ok(out)
        },
    )
}

fn ring_negative_log_evidence(data: ArrayView2<'_, f64>) -> f64 {
    let n = data.nrows();
    let r: Vec<f64> = (0..n)
        .map(|i| (data[[i, 0]].powi(2) + data[[i, 1]].powi(2)).sqrt())
        .collect();
    let mean = r.iter().sum::<f64>() / n as f64;
    let var = (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).max(1e-9);
    let log_norm = -0.5 * (std::f64::consts::TAU * var).ln();
    let log_angle = -(std::f64::consts::TAU).ln();
    let mut loglik = 0.0_f64;
    for &ri in &r {
        let ri = ri.max(1e-9);
        loglik += log_norm - 0.5 * (ri - mean).powi(2) / var + log_angle - ri.ln();
    }
    -loglik + 0.5 * 2.0 * (n as f64).ln()
}

// ---------------------------------------------------------------------------
// One race; records the decision and its magnitudes.
// ---------------------------------------------------------------------------

struct Draw {
    /// `true` ⇒ the stacking-headline winner is the smooth circle.
    circle_won: bool,
    /// Held-out stacking weight of the winner.
    winner_weight: f64,
    /// Corroborating log-Bayes-factor for circle over mixture (nats):
    /// `nle_mixture − nle_circle` (positive favours the circle).
    log_bf_circle_over_mixture: f64,
}

fn run_race(data: &Array2<f64>) -> Draw {
    let cfg = GaussianMixtureConfig::default();
    let rung = fit_mixture_rung(data.view(), MIXTURE_K_LADDER, cfg)
        .expect("mixture rung must fit at least one order");
    let mix_winner = rung.winner();
    let circle_nle = ring_negative_log_evidence(data.view());

    let candidates = vec![
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
            negative_log_evidence: circle_nle,
            certification: EvidenceCertification::Exact,
            density_provider: ring_density_provider(data.view()),
        },
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Mixture { k: mix_winner.k }),
            negative_log_evidence: mix_winner.bic,
            certification: EvidenceCertification::Exact,
            density_provider: mixture_density_provider(data.view(), mix_winner.k, cfg),
        },
    ];

    let verdict = adjudicate_predictive_race(
        data.nrows(),
        candidates,
        STACKING_CV_FOLDS,
        STACKING_CV_SEED,
        StackingConfig::default(),
    )
    .expect("cross-class adjudication must succeed");
    assert_eq!(verdict.headline, Headline::Stacking);
    let stacking = verdict.stacking.as_ref().expect("stacking weights");

    let circle_won = verdict.candidate_names[verdict.winner_index].starts_with("circle");
    Draw {
        circle_won,
        winner_weight: stacking.weights[verdict.winner_index],
        log_bf_circle_over_mixture: mix_winner.bic - circle_nle,
    }
}

// ---------------------------------------------------------------------------
// The sweep.
// ---------------------------------------------------------------------------

#[test]
fn repeated_draws_are_accurate_and_decisive_calls_are_never_wrong() {
    let mut draws: Vec<(bool /* truth_is_circle */, Draw)> = Vec::with_capacity(2 * N_REPLICATES);
    for r in 0..N_REPLICATES {
        let seed = 1000 + 37 * r as u64;
        draws.push((true, run_race(&sample_circle(seed))));
        draws.push((false, run_race(&sample_clusters(seed))));
    }

    // 1. Selection accuracy: every draw must recover its planted generator.
    let mut wrong = 0usize;
    for (truth_is_circle, d) in &draws {
        if d.circle_won != *truth_is_circle {
            wrong += 1;
        }
    }
    assert_eq!(
        wrong,
        0,
        "the stacking-headline winner must match the planted truth on every \
         draw at SNR {SNR} ({wrong} of {} wrong)",
        draws.len()
    );

    // 2. Calibration of magnitudes: every DECISIVE call must be correct. With
    //    zero selection errors this is implied, but it is asserted separately
    //    so that if accuracy slack is ever introduced above, overconfident
    //    errors (decisive AND wrong) still fail loudly on their own.
    for (truth_is_circle, d) in &draws {
        let decisive_stacking = d.winner_weight >= DECISIVE_STACKING_WEIGHT;
        let decisive_bf = d.log_bf_circle_over_mixture.abs() >= DECISIVE_LOG_BF;
        if decisive_stacking || decisive_bf {
            assert_eq!(
                d.circle_won,
                *truth_is_circle,
                "a decisive call (stacking weight {:.3}, |log BF| {:.1}) must \
                 never be wrong",
                d.winner_weight,
                d.log_bf_circle_over_mixture.abs()
            );
        }
    }

    // 3. The corroborating evidence channel must point with the truth in the
    //    overwhelming majority of draws (small near-tie minority tolerated —
    //    it is corroboration, not the headline).
    let mut evidence_agrees = 0usize;
    for (truth_is_circle, d) in &draws {
        let evidence_says_circle = d.log_bf_circle_over_mixture > 0.0;
        if evidence_says_circle == *truth_is_circle {
            evidence_agrees += 1;
        }
    }
    let needed = draws.len() - draws.len() / 12; // ≥ ~92%
    assert!(
        evidence_agrees >= needed,
        "the rank-aware Laplace evidence must corroborate the planted truth \
         on at least {needed} of {} draws, got {evidence_agrees}",
        draws.len()
    );

    // 4. And the race must actually be decisive most of the time at this SNR —
    //    a calibration claim is vacuous if no call ever clears the bar.
    let decisive = draws
        .iter()
        .filter(|(_, d)| d.winner_weight >= DECISIVE_STACKING_WEIGHT)
        .count();
    assert!(
        decisive * 2 >= draws.len(),
        "at matched SNR {SNR} at least half the draws should be stacking-decisive \
         (got {decisive} of {})",
        draws.len()
    );
}
