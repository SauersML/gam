//! #977 capstone — the BOTTOM two rungs of the structure ladder, end-to-end.
//!
//! The capstone wager is stated so it can lose: "manifold atoms are economical
//! iff features organize into low-dimensional curved families." The instrument
//! that converts a falsified wager into a *finding* (rather than an artifact) is
//! the structure ladder
//!
//!     isotropic noise  ⊂  dense interference factor (#974)
//!                      ⊂  sparse cluster (#907)  ⊂  manifold atom.
//!
//! The two TOP rungs (smooth circle vs discrete `k`-cluster mixture) are already
//! adjudicated end-to-end in `tests/identifiability/topology_mixture_rung.rs`
//! and `tests/quality/quality_mixture_rung_vs_reference.rs`. What was missing is
//! the FALSIFICATION-SAFETY contract on the two BOTTOM rungs: when the truth is
//! NOT a curved family — isotropic point-noise, or a dense low-rank correlated
//! Gaussian (the "dark-matter" interference factor) — the cross-class
//! adjudicator must NOT hand the headline to the smooth manifold (circle)
//! candidate. If it did, the method would manufacture geometry out of noise and
//! the whole capstone claim would be unfalsifiable.
//!
//! This test plants two ground-truth generators with NO ring structure:
//!   (A) isotropic 2-D Gaussian point noise (rung 0), and
//!   (B) a rank-1 dense interference factor (rung 1): an anisotropic Gaussian
//!       whose mass lies along a single random direction (a correlated factor,
//!       NOT a 1-D ring — a chord through the origin, not a circle).
//! and asserts the adjudicator awards the headline to the Euclidean/Gaussian
//! candidate, never to the circle, BOTH on held-out predictive density and on
//! the rank-aware quasi-Laplace score — across several fixed integer seeds.
//!
//! All assertions are against the PLANTED TRUTH (which generator produced the
//! data), never against a reference tool's output. All randomness is a fixed
//! splitmix64 stream seeded by an integer — there is no clock randomness.

use gam::solver::evidence::StackingConfig;
use gam::solver::topology_selector::{
    AutoTopologyKind, PredictiveRaceCandidate, EvidenceCertification, Headline, HeldOutDensityProvider,
    PredictiveCandidateKind, STACKING_CV_FOLDS, STACKING_CV_SEED, adjudicate_predictive_race,
};
use ndarray::{Array2, ArrayView2};

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock) — mirror of the existing
// topology-rung tests so the SNR conventions line up across the ladder.
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

const N_OBS: usize = 350;

// ---------------------------------------------------------------------------
// Planted generators for the two BOTTOM rungs — NO ring structure.
// ---------------------------------------------------------------------------

/// Rung 0 — isotropic 2-D Gaussian point noise centred at the origin. There is
/// no manifold and no cluster: the truth is a single Gaussian blob.
fn sample_isotropic_noise(seed: u64) -> Array2<f64> {
    let sigma = 1.0_f64;
    let mut rng = SplitMix64::new(seed ^ 0x150_7401C_u64);
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        out[[i, 0]] = sigma * rng.next_gaussian();
        out[[i, 1]] = sigma * rng.next_gaussian();
    }
    out
}

/// Rung 1 — a rank-1 dense interference factor (#974): an anisotropic Gaussian
/// whose mass concentrates along a single random unit direction `u`, plus a
/// thin isotropic floor. This is a CORRELATED factor (a chord through the
/// origin), NOT a 1-D ring: the radius is unbounded below (points pass near the
/// origin), so the ring model is badly mis-specified while a 2-D Gaussian fits
/// it exactly. It is the "dark-matter" object the wager must not mistake for a
/// curved atom.
fn sample_dense_factor(seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64::new(seed ^ 0xDA12_FAC7_u64);
    // Fixed-but-seed-dependent factor direction.
    let phi = std::f64::consts::TAU * SplitMix64::new(seed ^ 0xD11).next_unit();
    let (ux, uy) = (phi.cos(), phi.sin());
    let along = 1.0_f64; // factor (signal) std along u
    let floor = 0.12_f64; // isotropic noise floor across u
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        // Project a strong 1-D factor along u and a weak floor everywhere.
        let s = along * rng.next_gaussian();
        let nx = floor * rng.next_gaussian();
        let ny = floor * rng.next_gaussian();
        out[[i, 0]] = s * ux + nx;
        out[[i, 1]] = s * uy + ny;
    }
    out
}

// ---------------------------------------------------------------------------
// Held-out density providers.
//
// Euclidean/Gaussian: a full 2-D Gaussian (mean + 2x2 covariance) refit on each
// fold's training rows. P = 2 (mean) + 3 (symmetric 2x2 cov) = 5 parameters.
//
// Ring (smooth circle): the SAME ring model the top-rung test uses — a learned
// radius mean/variance with a uniform-in-angle distribution and the
// Cartesian->polar 1/r Jacobian. P = 2 (radius mean + variance).
//
// Both are genuinely held out: they refit on `train` and score on `eval`.
// ---------------------------------------------------------------------------

fn gaussian2d_density_provider<'a>(data: ArrayView2<'a, f64>) -> HeldOutDensityProvider<'a> {
    let owned = data.to_owned();
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            if train.len() < 3 {
                return Err("gaussian provider needs >=3 training rows".to_string());
            }
            let n = train.len() as f64;
            let (mut mx, mut my) = (0.0_f64, 0.0_f64);
            for &i in train {
                mx += owned[[i, 0]];
                my += owned[[i, 1]];
            }
            mx /= n;
            my /= n;
            let (mut sxx, mut sxy, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
            for &i in train {
                let dx = owned[[i, 0]] - mx;
                let dy = owned[[i, 1]] - my;
                sxx += dx * dx;
                sxy += dx * dy;
                syy += dy * dy;
            }
            sxx = (sxx / n).max(1e-9);
            syy = (syy / n).max(1e-9);
            sxy /= n;
            // Keep the covariance positive-definite (ridge the off-diagonal if
            // the sample is nearly degenerate).
            let mut det = sxx * syy - sxy * sxy;
            if det <= 1e-12 {
                sxy *= 0.999;
                det = sxx * syy - sxy * sxy;
                det = det.max(1e-12);
            }
            let inv_xx = syy / det;
            let inv_yy = sxx / det;
            let inv_xy = -sxy / det;
            let log_norm = -((std::f64::consts::TAU).ln()) - 0.5 * det.ln();
            let mut out = Vec::with_capacity(eval.len());
            for &i in eval {
                let dx = owned[[i, 0]] - mx;
                let dy = owned[[i, 1]] - my;
                let quad = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
                out.push(log_norm - 0.5 * quad);
            }
            Ok(out)
        },
    )
}

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
                out.push(log_r_density + log_angle - r.ln());
            }
            Ok(out)
        },
    )
}

// ---------------------------------------------------------------------------
// Closed-form rank-aware (BIC-form Laplace) negative-log-evidences:
//   -loglik + 1/2 * P * log n.
// Lower is better; these corroborate the held-out stacking headline.
// ---------------------------------------------------------------------------

fn gaussian2d_negative_log_evidence(data: ArrayView2<'_, f64>) -> f64 {
    let n = data.nrows();
    let nf = n as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    for i in 0..n {
        mx += data[[i, 0]];
        my += data[[i, 1]];
    }
    mx /= nf;
    my /= nf;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let dx = data[[i, 0]] - mx;
        let dy = data[[i, 1]] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    sxx = (sxx / nf).max(1e-9);
    syy = (syy / nf).max(1e-9);
    sxy /= nf;
    let det = (sxx * syy - sxy * sxy).max(1e-12);
    let inv_xx = syy / det;
    let inv_yy = sxx / det;
    let inv_xy = -sxy / det;
    let log_norm = -((std::f64::consts::TAU).ln()) - 0.5 * det.ln();
    let mut loglik = 0.0_f64;
    for i in 0..n {
        let dx = data[[i, 0]] - mx;
        let dy = data[[i, 1]] - my;
        let quad = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
        loglik += log_norm - 0.5 * quad;
    }
    let p = 5.0_f64; // mean(2) + symmetric 2x2 cov(3)
    -loglik + 0.5 * p * nf.ln()
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
    let p = 2.0_f64;
    -loglik + 0.5 * p * (n as f64).ln()
}

// ---------------------------------------------------------------------------
// Adjudication driver: Euclidean/Gaussian vs smooth-circle.
//
// This is a SAME-CLASS race (no discrete mixture candidate), so the headline is
// rank-aware evidence (winner-take-all, lower NLE wins). We ALSO build the
// held-out cross-validated predictive density table by inserting a discrete
// Mixture-k1 sibling whose evidence is fixed far above the others, forcing the
// cross-class stacking path so we can read held-out predictive mass directly.
// The wager-safety contract is asserted on BOTH headlines.
// ---------------------------------------------------------------------------

struct LadderOutcome {
    evidence_winner: String,
    gaussian_evidence: f64,
    ring_evidence: f64,
    gaussian_stack_weight: f64,
    ring_stack_weight: f64,
}

fn run_bottom_rung(data: &Array2<f64>) -> LadderOutcome {
    let gaussian_evidence = gaussian2d_negative_log_evidence(data.view());
    let ring_evidence = ring_negative_log_evidence(data.view());

    // --- Same-class headline: Euclidean(Gaussian) vs Circle, winner-take-all.
    let candidates = vec![
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
            negative_log_evidence: gaussian_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: gaussian2d_density_provider(data.view()),
        },
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
            negative_log_evidence: ring_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: ring_density_provider(data.view()),
        },
    ];
    let verdict = adjudicate_predictive_race(
        data.nrows(),
        candidates,
        STACKING_CV_FOLDS,
        STACKING_CV_SEED,
        StackingConfig::default(),
    )
    .expect("same-class adjudication must succeed");
    assert!(
        !verdict.is_cross_class,
        "Euclidean + Circle (no mixture) must be a same-class race"
    );
    assert_eq!(
        verdict.headline,
        Headline::Evidence,
        "same-class headline must be rank-aware evidence"
    );
    let evidence_winner = verdict.candidate_names[verdict.winner_index].clone();

    // --- Cross-class held-out predictive density: force the stacking path by
    // adding a deliberately-poor discrete mixture sibling, then read the
    // Gaussian vs Circle predictive mass. The added sibling is a real k=1
    // Gaussian-mixture density (NOT a stub) so the held-out table is honest; we
    // only assert on the Gaussian-vs-Circle ORDERING, not the mixture's mass.
    let mixture_provider = gam::solver::topology_selector::mixture_density_provider(
        data.view(),
        1,
        gam::solver::evidence::GaussianMixtureConfig::default(),
    );
    let stack_candidates = vec![
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
            negative_log_evidence: gaussian_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: gaussian2d_density_provider(data.view()),
        },
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
            negative_log_evidence: ring_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: ring_density_provider(data.view()),
        },
        PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Mixture { k: 1 }),
            negative_log_evidence: gaussian_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: mixture_provider,
        },
    ];
    let stack_verdict = adjudicate_predictive_race(
        data.nrows(),
        stack_candidates,
        STACKING_CV_FOLDS,
        STACKING_CV_SEED,
        StackingConfig::default(),
    )
    .expect("cross-class adjudication must succeed");
    assert!(
        stack_verdict.is_cross_class,
        "adding a Mixture candidate must trigger cross-class stacking"
    );
    assert_eq!(stack_verdict.headline, Headline::Stacking);
    let stacking = stack_verdict
        .stacking
        .as_ref()
        .expect("cross-class verdict must carry stacking weights");

    LadderOutcome {
        evidence_winner,
        gaussian_evidence,
        ring_evidence,
        gaussian_stack_weight: stacking.weights[0],
        ring_stack_weight: stacking.weights[1],
    }
}

// ---------------------------------------------------------------------------
// Tests — the falsification-safety contract on the two bottom rungs.
// ---------------------------------------------------------------------------

/// Rung 0 — isotropic point noise must NOT be claimed as a manifold. The
/// Euclidean/Gaussian candidate must win the evidence headline AND carry strictly
/// more held-out predictive mass than the smooth circle, across fixed seeds.
#[test]
fn isotropic_noise_does_not_masquerade_as_a_manifold() {
    let seeds = [7_u64, 19, 41, 83];
    for &seed in &seeds {
        let data = sample_isotropic_noise(seed);
        let outcome = run_bottom_rung(&data);
        assert_eq!(
            outcome.evidence_winner, "euclidean",
            "seed {seed}: isotropic noise — Gaussian must win the evidence headline, \
             not the circle (gaussian_nle={:.2}, ring_nle={:.2})",
            outcome.gaussian_evidence, outcome.ring_evidence,
        );
        assert!(
            outcome.gaussian_evidence < outcome.ring_evidence,
            "seed {seed}: isotropic noise — Gaussian rank-aware evidence must beat the \
             ring (gaussian_nle={:.2}, ring_nle={:.2})",
            outcome.gaussian_evidence,
            outcome.ring_evidence,
        );
        assert!(
            outcome.gaussian_stack_weight > outcome.ring_stack_weight,
            "seed {seed}: isotropic noise — Gaussian must carry more held-out predictive \
             mass than the circle (gaussian_w={:.4}, ring_w={:.4})",
            outcome.gaussian_stack_weight,
            outcome.ring_stack_weight,
        );
    }
}

/// Rung 1 — a dense rank-1 interference factor (#974) must NOT be claimed as a
/// manifold. It is a correlated Gaussian (a chord through the origin), not a
/// ring: the Gaussian candidate must win both headlines.
#[test]
fn dense_interference_factor_does_not_masquerade_as_a_manifold() {
    let seeds = [5_u64, 17, 37, 79];
    for &seed in &seeds {
        let data = sample_dense_factor(seed);
        let outcome = run_bottom_rung(&data);
        assert_eq!(
            outcome.evidence_winner, "euclidean",
            "seed {seed}: dense factor — Gaussian must win the evidence headline, not the \
             circle (gaussian_nle={:.2}, ring_nle={:.2})",
            outcome.gaussian_evidence, outcome.ring_evidence,
        );
        assert!(
            outcome.gaussian_evidence < outcome.ring_evidence,
            "seed {seed}: dense factor — Gaussian rank-aware evidence must beat the ring \
             (gaussian_nle={:.2}, ring_nle={:.2})",
            outcome.gaussian_evidence,
            outcome.ring_evidence,
        );
        assert!(
            outcome.gaussian_stack_weight > outcome.ring_stack_weight,
            "seed {seed}: dense factor — Gaussian must carry more held-out predictive mass \
             than the circle (gaussian_w={:.4}, ring_w={:.4})",
            outcome.gaussian_stack_weight,
            outcome.ring_stack_weight,
        );
    }
}

/// Ladder monotonicity guard: on genuine ring data the SAME instrument flips —
/// the circle must beat the Gaussian on rank-aware evidence. This anchors the
/// bottom-rung tests above against a trivially-passing "Gaussian always wins"
/// bug (which would also pass on a ring) and ties the bottom rungs to the
/// top-rung manifold/cluster tests as one ladder.
#[test]
fn ring_truth_flips_the_verdict_to_the_manifold() {
    let seeds = [11_u64, 29, 47];
    for &seed in &seeds {
        // Genuine ring: matched-SNR radial jitter, radius 1.
        let radius = 1.0_f64;
        let noise = radius / 12.0;
        let mut rng = SplitMix64::new(seed ^ 0xC18C1E_u64);
        let mut data = Array2::<f64>::zeros((N_OBS, 2));
        for i in 0..N_OBS {
            let theta = std::f64::consts::TAU * rng.next_unit();
            data[[i, 0]] = radius * theta.cos() + noise * rng.next_gaussian();
            data[[i, 1]] = radius * theta.sin() + noise * rng.next_gaussian();
        }
        let gaussian_evidence = gaussian2d_negative_log_evidence(data.view());
        let ring_evidence = ring_negative_log_evidence(data.view());
        assert!(
            ring_evidence < gaussian_evidence,
            "seed {seed}: ring truth — the circle must beat the Gaussian on rank-aware \
             evidence (ring_nle={:.2}, gaussian_nle={:.2}); a 'Gaussian always wins' \
             instrument is broken",
            ring_evidence,
            gaussian_evidence,
        );
    }
}
