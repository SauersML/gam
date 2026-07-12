//! #980 verification §2 — the **circle-read-discretely two-verdict race**,
//! the fixture that demonstrates representational and computational geometry
//! are *different measurable objects* (consuming #907's discrete-mixture rung
//! and cross-class stacking adjudication).
//!
//! # The planted situation
//!
//! A geometric circle is planted in activation space. A synthetic downstream
//! readout consumes it **discretely**: it snaps the circular coordinate to the
//! nearest of `k = 7` arc centers (a sharp von-Mises-weighted average of the
//! arc-center directions — smooth, saturating, exactly the "circle used as 7
//! arcs" of the Engels et al. controversy). Then:
//!
//! * **Representational verdict** — the topology race run on the raw
//!   activation cloud must say **circle** (the shape of `p(x)`).
//! * **Computational verdict** — the race run on the cloud's image under the
//!   readout must say **7-cluster mixture** (the shape of what `F` reads).
//!   Racing the readout image *is* racing under the output-Fisher pullback
//!   geometry: distances in the image are `‖dF‖ = ‖J dx‖`, i.e. `M(x)`
//!   distances with `M = JᵀJ` — realized globally, so the linearization
//!   caveat does not bite.
//!
//! # The amended #980 semantics, asserted
//!
//! The two verdicts are **both reported, neither replaces the other**: "circle
//! in the representation, consumed discretely here" is the *finding*, not a
//! contradiction to resolve by pickinging a metric. The test therefore asserts
//! the two arms disagree — that disagreement is the measurable content of the
//! representational-vs-computational distinction, and it is exactly what
//! eyeballing PCA plots cannot adjudicate.
//!
//! Both arms run the same cross-class machinery: in-class mixture ladder by
//! rank-aware Laplace evidence, then the cross-class race with the **held-out
//! stacking log-density headline** (#907's discipline — Laplace evidence
//! across model classes is corroboration only).

use gam::solver::evidence::{GaussianMixtureConfig, StackingConfig};
use gam::solver::topology_selector::{
    AutoTopologyKind, CrossClassCandidate, EvidenceCertification, Headline, HeldOutDensityProvider,
    MIXTURE_K_LADDER, PredictiveCandidateKind, STACKING_CV_FOLDS, STACKING_CV_SEED,
    adjudicate_cross_class_race, fit_mixture_rung, mixture_density_provider,
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
// The plant: one circle, one quantizing readout.
// ---------------------------------------------------------------------------

const N_OBS: usize = 350;
/// Number of arcs the synthetic readout quantizes the circle into. 7 is both
/// the issue's own example ("a circle the readout consumes as 7 arcs") and a
/// rung of [`MIXTURE_K_LADDER`], so the in-class ladder can name it exactly.
const N_ARCS: usize = 7;
/// Radial activation noise (SNR 12, matching the existing planted races).
const ACTIVATION_NOISE: f64 = 1.0 / 12.0;
/// Von-Mises sharpness of the arc snap. At 150 the transition zones between
/// adjacent arcs cover only a few percent of the circle, so the readout image
/// is 7 tight blobs plus a thin smear of genuinely-in-between rows — soft
/// quantization, honestly modeled, not a hand-placed cluster sample.
const READOUT_SHARPNESS: f64 = 150.0;
/// Observation noise on the readout image. Spacing between adjacent arc
/// centers is `2 sin(π/7) ≈ 0.87`, so this is image SNR ≈ 5.8 — the blobs are
/// unambiguous, yet wide enough that the few soft-transition rows (a boundary
/// row maps to a point part-way between two arc centers) sit within ~3σ of a
/// corner blob. That keeps extra mixture components unprofitable under the
/// rank-aware pricing, so the in-class ladder recovers the PLANTED order
/// `k = 7` instead of spending components on the transition smear.
const READOUT_NOISE: f64 = 0.15;

/// Planted circle activations: unit ring with isotropic jitter.
fn sample_circle_activations(seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64::new(seed ^ 0x2C19C1E_u64);
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    for i in 0..N_OBS {
        let theta = std::f64::consts::TAU * rng.next_unit();
        out[[i, 0]] = theta.cos() + ACTIVATION_NOISE * rng.next_gaussian();
        out[[i, 1]] = theta.sin() + ACTIVATION_NOISE * rng.next_gaussian();
    }
    out
}

/// The synthetic quantizing readout `F: ℝ² → ℝ²`: a sharp von-Mises-weighted
/// average of the `n_arcs` arc-center unit vectors,
///
/// ```text
/// F(x) = Σ_j w_j(θ) (cos φ_j, sin φ_j),   w_j ∝ exp(κ cos(θ − φ_j)),
/// ```
///
/// with `θ = atan2(x_1, x_0)`. Smooth and differentiable everywhere, but with
/// `κ = READOUT_SHARPNESS` it saturates to the nearest arc center across all
/// but a thin boundary zone: the circle is *consumed as `n_arcs` arcs*. Plus
/// isotropic observation noise on the image.
fn readout_image(activations: &Array2<f64>, n_arcs: usize, seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64::new(seed ^ 0xF15_4E_AD_u64);
    let n = activations.nrows();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let theta = activations[[i, 1]].atan2(activations[[i, 0]]);
        // Log-sum-exp-stable von-Mises weights over the arc centers.
        let mut log_w = vec![0.0_f64; n_arcs];
        let mut max_log = f64::NEG_INFINITY;
        for (j, lw) in log_w.iter_mut().enumerate() {
            let phi = std::f64::consts::TAU * j as f64 / n_arcs as f64;
            *lw = READOUT_SHARPNESS * (theta - phi).cos();
            max_log = max_log.max(*lw);
        }
        let mut total = 0.0_f64;
        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        for (j, lw) in log_w.iter().enumerate() {
            let w = (lw - max_log).exp();
            let phi = std::f64::consts::TAU * j as f64 / n_arcs as f64;
            fx += w * phi.cos();
            fy += w * phi.sin();
            total += w;
        }
        out[[i, 0]] = fx / total + READOUT_NOISE * rng.next_gaussian();
        out[[i, 1]] = fy / total + READOUT_NOISE * rng.next_gaussian();
    }
    out
}

// ---------------------------------------------------------------------------
// Smooth-circle (ring) candidate: held-out density provider + rank-aware
// evidence, identical in form to the existing planted races.
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
// One arm of the race: circle candidate vs the in-class mixture winner, with
// the cross-class stacking headline.
// ---------------------------------------------------------------------------

struct Verdict {
    winner_name: String,
    circle_weight: f64,
    mixture_weight: f64,
    mixture_k: usize,
}

fn run_race(data: &Array2<f64>) -> Verdict {
    let cfg = GaussianMixtureConfig::default();
    let rung = fit_mixture_rung(data.view(), MIXTURE_K_LADDER, cfg)
        .expect("mixture rung must fit at least one order");
    let mix_winner = rung.winner();
    let mixture_k = mix_winner.k;

    let candidates = vec![
        CrossClassCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
            negative_log_evidence: ring_negative_log_evidence(data.view()),
            certification: EvidenceCertification::Exact,
            density_provider: ring_density_provider(data.view()),
        },
        CrossClassCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Mixture { k: mixture_k }),
            negative_log_evidence: mix_winner.negative_log_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: mixture_density_provider(data.view(), mixture_k, cfg),
        },
    ];

    let verdict = adjudicate_cross_class_race(
        data.nrows(),
        candidates,
        STACKING_CV_FOLDS,
        STACKING_CV_SEED,
        StackingConfig::default(),
    )
    .expect("cross-class adjudication must succeed");

    assert!(
        verdict.is_cross_class,
        "Circle + Mixture race is cross-class"
    );
    assert_eq!(
        verdict.headline,
        Headline::Stacking,
        "cross-class headline must be held-out stacking, not Laplace evidence"
    );
    let stacking = verdict
        .stacking
        .as_ref()
        .expect("cross-class verdict must carry stacking weights");

    Verdict {
        winner_name: verdict.candidate_names[verdict.winner_index].clone(),
        circle_weight: stacking.weights[0],
        mixture_weight: stacking.weights[1],
        mixture_k,
    }
}

// ---------------------------------------------------------------------------
// The two-verdict test.
// ---------------------------------------------------------------------------

#[test]
fn circle_read_discretely_yields_two_different_verdicts() {
    let seeds = [17_u64, 59, 113];
    for &seed in &seeds {
        let activations = sample_circle_activations(seed);
        let image = readout_image(&activations, N_ARCS, seed);

        // ---- representational verdict: the activation cloud is a circle ----
        let representational = run_race(&activations);
        assert!(
            representational.winner_name.starts_with("circle"),
            "seed {seed}: the representational race (raw activations) must \
             say CIRCLE, got {} (circle_w={:.4}, mixture_w={:.4}, k={})",
            representational.winner_name,
            representational.circle_weight,
            representational.mixture_weight,
            representational.mixture_k,
        );
        assert!(
            representational.circle_weight > representational.mixture_weight,
            "seed {seed}: representational arm — circle must carry the \
             stacking mass (circle_w={:.4}, mixture_w={:.4})",
            representational.circle_weight,
            representational.mixture_weight,
        );

        // ---- computational verdict: the readout image is 7 clusters --------
        let computational = run_race(&image);
        assert!(
            computational.winner_name.starts_with("mixture"),
            "seed {seed}: the computational race (readout image = pullback \
             geometry) must say DISCRETE MIXTURE, got {} (circle_w={:.4}, \
             mixture_w={:.4}, k={})",
            computational.winner_name,
            computational.circle_weight,
            computational.mixture_weight,
            computational.mixture_k,
        );
        assert!(
            computational.mixture_weight > computational.circle_weight,
            "seed {seed}: computational arm — the mixture must carry the \
             stacking mass (circle_w={:.4}, mixture_w={:.4})",
            computational.circle_weight,
            computational.mixture_weight,
        );
        assert_eq!(
            computational.mixture_k, N_ARCS,
            "seed {seed}: the computational verdict must recover the planted \
             number of arcs (got k={}, planted {N_ARCS})",
            computational.mixture_k,
        );

        // ---- the finding is the DISAGREEMENT, both verdicts standing -------
        // "Circle in the representation, consumed discretely here." Neither
        // verdict replaces the other; a harness that collapsed them to one
        // metric would erase exactly the distinction this fixture plants.
        assert_ne!(
            representational.winner_name, computational.winner_name,
            "seed {seed}: the two geometries are different measurable objects \
             — the verdicts must disagree on this plant"
        );
        println!(
            "seed {seed}: representational = {} (w={:.3}), computational = {} \
             (w={:.3}, k={}) — circle in the representation, consumed \
             discretely by the readout",
            representational.winner_name,
            representational.circle_weight,
            computational.winner_name,
            computational.mixture_weight,
            computational.mixture_k,
        );
    }
}

/// The natural first-choice readout — a 4-arc *quadrant* quantizer — which the
/// fixture originally had to route around because the coarse mixture ladder
/// could not name `k = 4`. With the #996 local refinement the ladder can, so
/// the computational verdict must now recover the planted quadrant order
/// exactly. (The representational arm on the same activations is covered by
/// the 7-arc test above; this arm pins the off-ladder order recovery.)
#[test]
fn quadrant_readout_computational_verdict_recovers_k4() {
    const QUADRANTS: usize = 4;
    for &seed in &[23_u64, 71] {
        let activations = sample_circle_activations(seed);
        let image = readout_image(&activations, QUADRANTS, seed);
        let computational = run_race(&image);
        assert!(
            computational.winner_name.starts_with("mixture"),
            "seed {seed}: quadrant readout image must adjudicate DISCRETE \
             MIXTURE, got {} (circle_w={:.4}, mixture_w={:.4}, k={})",
            computational.winner_name,
            computational.circle_weight,
            computational.mixture_weight,
            computational.mixture_k,
        );
        assert_eq!(
            computational.mixture_k, QUADRANTS,
            "seed {seed}: the refined ladder must name the planted quadrant \
             order exactly (got k={}, planted {QUADRANTS})",
            computational.mixture_k,
        );
    }
}
