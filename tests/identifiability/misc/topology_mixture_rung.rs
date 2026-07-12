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

use gam::solver::evidence::{GaussianMixtureConfig, StackingConfig};
use gam::solver::topology_selector::{
    AutoTopologyKind, EvidenceCertification, Headline, HeldOutDensityProvider, MIXTURE_K_LADDER,
    PredictiveCandidateKind, PredictiveRaceCandidate, STACKING_CV_FOLDS, STACKING_CV_SEED,
    adjudicate_predictive_race, fit_mixture_rung,
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
const N_CLUSTERS: usize = 7;

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

/// Seven well-separated isotropic Gaussian blobs. Truth = DISCRETE MIXTURE.
fn sample_clusters(seed: u64) -> Array2<f64> {
    // Cluster centers on a ring of radius R_struct so the inter-cluster spacing
    // is comparable to the circle's radius (matched structure scale). The
    // within-cluster spread sets the noise; spacing / spread == SNR.
    let r_struct = 1.0_f64;
    // Nearest-neighbour spacing of N points on a ring of radius R is
    // 2 R sin(pi / N); use that as the structure scale to match SNR.
    let spacing = 2.0 * r_struct * (std::f64::consts::PI / N_CLUSTERS as f64).sin();
    let spread = spacing / SNR; // matched SNR
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

struct RaceOutcome {
    winner_name: String,
    headline: Headline,
    circle_weight: f64,
    mixture_weight: f64,
    circle_evidence: f64,
    mixture_evidence: f64,
    mixture_k: usize,
}

fn run_race(data: &Array2<f64>) -> RaceOutcome {
    let cfg = GaussianMixtureConfig::default();
    // In-class mixture rung: sweep the fixed ladder, pick the rank-aware
    // Laplace-evidence winner.
    let rung = fit_mixture_rung(data.view(), MIXTURE_K_LADDER, cfg)
        .expect("mixture rung must fit at least one order");
    let mix_winner = rung.winner();
    let mixture_k = mix_winner.k;
    let mixture_evidence = mix_winner.negative_log_evidence;

    // Smooth-circle rank-aware evidence: a 2-parameter ring model (r_bar,
    // sigma_r) plus uniform angle. Score on the SAME negative-log-evidence
    // scale: -loglik + 1/2 P log(n) is the BIC-form Laplace; here we report the
    // closed-form ring evidence as corroboration only (the headline is
    // stacking).
    let circle_evidence = ring_negative_log_evidence(data.view());

    let circle_provider = ring_density_provider(data.view());
    let mixture_provider =
        gam::solver::topology_selector::mixture_density_provider(data.view(), mixture_k, cfg);

    let candidates = vec![
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
            negative_log_evidence: circle_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: circle_provider,
        },
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Mixture { k: mixture_k }),
            negative_log_evidence: mixture_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: mixture_provider,
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

    assert!(
        verdict.is_cross_class,
        "race mixing Circle + Mixture must be detected as cross-class"
    );
    assert_eq!(
        verdict.headline,
        Headline::Stacking,
        "cross-class headline must switch to stacking"
    );
    let stacking = verdict
        .stacking
        .as_ref()
        .expect("cross-class verdict must carry stacking weights");

    RaceOutcome {
        winner_name: verdict.candidate_names[verdict.winner_index].clone(),
        headline: verdict.headline,
        circle_weight: stacking.weights[0],
        mixture_weight: stacking.weights[1],
        circle_evidence,
        mixture_evidence,
        mixture_k,
    }
}

/// Closed-form ring negative-log-evidence on the rank-aware Laplace (BIC-form)
/// scale: `-loglik + 1/2 P log n` with `P = 2` (ring radius mean + variance).
/// Reported as corroboration; the cross-class headline is stacking.
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
    let p = 2.0_f64;
    -loglik + 0.5 * p * (n as f64).ln()
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn circle_data_smooth_circle_wins_across_seeds() {
    let seeds = [11_u64, 29, 47, 101];
    for &seed in &seeds {
        let data = sample_circle(seed);
        let outcome = run_race(&data);
        assert_eq!(
            outcome.headline,
            Headline::Stacking,
            "seed {seed}: cross-class headline must be stacking"
        );
        assert!(
            outcome.circle_weight > outcome.mixture_weight,
            "seed {seed}: circle data — smooth-circle must carry more stacking mass \
             (circle_w={:.4}, mixture_w={:.4}, mixture_k={})",
            outcome.circle_weight,
            outcome.mixture_weight,
            outcome.mixture_k,
        );
        assert!(
            outcome.winner_name.starts_with("circle"),
            "seed {seed}: circle data — headline winner must be the smooth circle, got {}",
            outcome.winner_name
        );
        // Corroboration: the ring model's evidence should also beat the mixture.
        assert!(
            outcome.circle_evidence < outcome.mixture_evidence,
            "seed {seed}: circle data — ring rank-aware evidence should corroborate \
             (circle_nle={:.2}, mixture_nle={:.2})",
            outcome.circle_evidence,
            outcome.mixture_evidence,
        );
    }
}

#[test]
fn clustered_data_mixture_rung_wins_across_seeds() {
    let seeds = [13_u64, 31, 53, 103];
    for &seed in &seeds {
        let data = sample_clusters(seed);
        let outcome = run_race(&data);
        assert_eq!(
            outcome.headline,
            Headline::Stacking,
            "seed {seed}: cross-class headline must be stacking"
        );
        assert!(
            outcome.mixture_weight > outcome.circle_weight,
            "seed {seed}: clustered data — mixture rung must carry more stacking mass \
             (circle_w={:.4}, mixture_w={:.4}, mixture_k={})",
            outcome.circle_weight,
            outcome.mixture_weight,
            outcome.mixture_k,
        );
        assert!(
            outcome.winner_name.starts_with("mixture"),
            "seed {seed}: clustered data — headline winner must be the mixture, got {}",
            outcome.winner_name
        );
        // The 7-cluster truth should be recovered (or closely so) by the
        // in-class rank-aware evidence winner over the ladder.
        assert!(
            outcome.mixture_k >= 3,
            "seed {seed}: clustered data — mixture rung should select a multi-component \
             order, got k={}",
            outcome.mixture_k,
        );
    }
}

#[test]
fn mixture_rung_prices_order_by_free_parameters() {
    // The ladder is fixed and each order is priced by its own free-parameter
    // count entering the rank-aware normalizer.
    let data = sample_clusters(7);
    let rung = fit_mixture_rung(
        data.view(),
        MIXTURE_K_LADDER,
        GaussianMixtureConfig::default(),
    )
    .expect("rung fit");
    for fit in &rung.fits {
        let d = 2usize;
        let cov_per = d * (d + 1) / 2;
        let expected = (fit.k - 1) + fit.k * d + fit.k * cov_per;
        assert_eq!(
            fit.num_parameters, expected,
            "mixture k={} free-parameter count must be (k-1) + k*d + k*d(d+1)/2",
            fit.k
        );
        assert!(
            fit.negative_log_evidence.is_finite(),
            "rank-aware Laplace evidence must be finite for k={}",
            fit.k
        );
    }
}

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
