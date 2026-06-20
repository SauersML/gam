//! #907 — structured-union composites in the topology race.
//!
//! A *union* candidate is a small FIXED composite of named component structures
//! ({circle+circle, circle+point-cluster, line+cluster}) joined by a hard
//! row-responsibility split. Each component is REML/Laplace-fit on its group and
//! the per-component rank-aware evidences are SUMMED, priced by the TOTAL
//! free-parameter count across components — so a union is strictly more expensive
//! than either pure rung and can only win when the structured split buys enough
//! likelihood to pay for its extra parameters.
//!
//! These planted tests sample from GROUND-TRUTH generators at fixed integer seeds
//! (no clock randomness) and assert the cross-class adjudicator recovers the
//! planted truth:
//!
//!   * two well-separated circles → a structured union BEATS the single-torus and
//!     the single-circle pure rungs;
//!   * circle + outlier cluster   → a structured union BEATS both pure rungs;
//!   * NEGATIVE CONTROL: a single circle must NOT prefer any union — the
//!     complexity pricing earns its keep and a pure rung carries the headline.
//!
//! The assertions are against the PLANTED TRUTH (which generator produced the
//! data), never against a reference tool's output.

use gam::solver::evidence::{GaussianMixtureConfig, StackingConfig, UnionStructure};
use gam::solver::topology_selector::{
    AutoTopologyKind, CrossClassCandidate, EvidenceCertification, Headline, HeldOutDensityProvider,
    STACKING_CV_FOLDS, STACKING_CV_SEED, adjudicate_cross_class_race, fit_union_candidate,
    fit_union_rung, union_density_provider,
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

const N_OBS: usize = 360;

// ---------------------------------------------------------------------------
// Planted generators.
// ---------------------------------------------------------------------------

/// One ring of radius `radius` centred at `(cx, cy)` with isotropic radial
/// jitter `noise`, writing `count` rows starting at `offset`.
fn fill_ring(
    out: &mut Array2<f64>,
    rng: &mut SplitMix64,
    offset: usize,
    count: usize,
    cx: f64,
    cy: f64,
    radius: f64,
    noise: f64,
) {
    for k in 0..count {
        let theta = std::f64::consts::TAU * rng.next_unit();
        out[[offset + k, 0]] = cx + radius * theta.cos() + noise * rng.next_gaussian();
        out[[offset + k, 1]] = cy + radius * theta.sin() + noise * rng.next_gaussian();
    }
}

/// One isotropic Gaussian blob centred at `(cx, cy)` with spread `spread`.
fn fill_blob(
    out: &mut Array2<f64>,
    rng: &mut SplitMix64,
    offset: usize,
    count: usize,
    cx: f64,
    cy: f64,
    spread: f64,
) {
    for k in 0..count {
        out[[offset + k, 0]] = cx + spread * rng.next_gaussian();
        out[[offset + k, 1]] = cy + spread * rng.next_gaussian();
    }
}

/// TWO well-separated unit circles. Truth = circle+circle UNION.
fn sample_two_circles(seed: u64) -> Array2<f64> {
    let radius = 1.0_f64;
    let noise = radius / 18.0;
    let mut rng = SplitMix64::new(seed ^ 0x2C18C1E_u64);
    let half = N_OBS / 2;
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    // Two rings separated by 8 radii along x — disjoint, no overlap.
    fill_ring(&mut out, &mut rng, 0, half, -4.0, 0.0, radius, noise);
    fill_ring(
        &mut out,
        &mut rng,
        half,
        N_OBS - half,
        4.0,
        0.0,
        radius,
        noise,
    );
    out
}

/// ONE circle plus an isolated outlier blob. Truth = circle+cluster UNION.
fn sample_circle_plus_cluster(seed: u64) -> Array2<f64> {
    let radius = 1.0_f64;
    let noise = radius / 18.0;
    let mut rng = SplitMix64::new(seed ^ 0xC1057E_u64);
    // Most rows on a ring at the origin; a minority in a tight blob far away.
    let n_blob = N_OBS / 4;
    let n_ring = N_OBS - n_blob;
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    fill_ring(&mut out, &mut rng, 0, n_ring, 0.0, 0.0, radius, noise);
    fill_blob(&mut out, &mut rng, n_ring, n_blob, 6.0, 6.0, 0.08);
    out
}

/// ONE single circle. NEGATIVE CONTROL: no union should be preferred.
fn sample_single_circle(seed: u64) -> Array2<f64> {
    let radius = 1.0_f64;
    let noise = radius / 18.0;
    let mut rng = SplitMix64::new(seed ^ 0x51A91E_u64);
    let mut out = Array2::<f64>::zeros((N_OBS, 2));
    fill_ring(&mut out, &mut rng, 0, N_OBS, 0.0, 0.0, radius, noise);
    out
}

// ---------------------------------------------------------------------------
// Smooth pure-rung held-out density providers (genuine smooth-class candidates).
//
// Single-circle (ring): the data lives on ONE ring with a learned radius
// mean/variance about the data centroid and a uniform-in-angle distribution.
//   p(x, y) = N(r; r_bar, sigma_r^2) * (1 / (2 pi)) * (1 / r)   (polar Jacobian).
//
// Single-torus: a 2-D anisotropic Gaussian (a flat "torus patch" treated as a
// single smooth chart) — the natural smooth competitor that a union of two
// circles or a circle+cluster must beat. Refits per fold.
// ---------------------------------------------------------------------------

fn ring_density_provider<'a>(data: ArrayView2<'a, f64>) -> HeldOutDensityProvider<'a> {
    let owned = data.to_owned();
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            if train.is_empty() {
                return Err("ring provider got empty training set".to_string());
            }
            let n = train.len() as f64;
            let cx = train.iter().map(|&i| owned[[i, 0]]).sum::<f64>() / n;
            let cy = train.iter().map(|&i| owned[[i, 1]]).sum::<f64>() / n;
            let r_of = |i: usize| -> f64 {
                ((owned[[i, 0]] - cx).powi(2) + (owned[[i, 1]] - cy).powi(2)).sqrt()
            };
            let mean: f64 = train.iter().map(|&i| r_of(i)).sum::<f64>() / n;
            let var: f64 =
                (train.iter().map(|&i| (r_of(i) - mean).powi(2)).sum::<f64>() / n).max(1e-9);
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

fn torus_density_provider<'a>(data: ArrayView2<'a, f64>) -> HeldOutDensityProvider<'a> {
    // A single anisotropic 2-D Gaussian chart (a flat torus patch fit as one
    // smooth manifold). Diagonal-plus-cross covariance estimated from the fold.
    let owned = data.to_owned();
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            if train.len() < 3 {
                return Err("torus provider got too few training rows".to_string());
            }
            let n = train.len() as f64;
            let mx = train.iter().map(|&i| owned[[i, 0]]).sum::<f64>() / n;
            let my = train.iter().map(|&i| owned[[i, 1]]).sum::<f64>() / n;
            let mut sxx = 0.0_f64;
            let mut syy = 0.0_f64;
            let mut sxy = 0.0_f64;
            for &i in train {
                let dx = owned[[i, 0]] - mx;
                let dy = owned[[i, 1]] - my;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
            sxx = (sxx / n).max(1e-9);
            syy = (syy / n).max(1e-9);
            sxy /= n;
            let det = (sxx * syy - sxy * sxy).max(1e-12);
            let inv_xx = syy / det;
            let inv_yy = sxx / det;
            let inv_xy = -sxy / det;
            let log_norm = -(std::f64::consts::TAU).ln() - 0.5 * det.ln();
            let mut out = Vec::with_capacity(eval.len());
            for &i in eval {
                let dx = owned[[i, 0]] - mx;
                let dy = owned[[i, 1]] - my;
                let q = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
                out.push(log_norm - 0.5 * q);
            }
            Ok(out)
        },
    )
}

// ---------------------------------------------------------------------------
// Race driver: a structured union competes cross-class against the smooth rungs.
// ---------------------------------------------------------------------------

struct RaceOutcome {
    winner_name: String,
    headline: Headline,
    /// Stacking weight carried by the structured-union candidate.
    union_weight: f64,
    /// Largest stacking weight carried by any pure (smooth) rung.
    best_pure_weight: f64,
    /// The in-class union winner's composite structure.
    union_structure: UnionStructure,
}

/// Run a cross-class race: in-class union winner (over the fixed ladder) vs the
/// single-circle ring and single-torus pure rungs. The union enters the race via
/// `fit_union_rung` (in-class winner) + `union_density_provider`; the cross-class
/// adjudicator builds the selection-time CV held-out density table and stacks.
fn run_race(data: &Array2<f64>) -> RaceOutcome {
    let cfg = GaussianMixtureConfig::default();
    // In-class union rung: fit the FIXED ladder, take the summed rank-aware
    // evidence winner. This is the composite that competes cross-class.
    let rung =
        fit_union_rung(data.view(), cfg).expect("union rung must fit at least one composite");
    let union_winner = rung.winner();
    let union_structure = union_winner.structure;
    let union_evidence = union_winner.negative_log_evidence;

    // Pure-rung evidences (corroboration only; the headline is stacking).
    let ring_evidence = ring_negative_log_evidence(data.view());
    let torus_evidence = torus_negative_log_evidence(data.view());

    let candidates = vec![
        CrossClassCandidate {
            kind: AutoTopologyKind::Circle,
            negative_log_evidence: ring_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: ring_density_provider(data.view()),
        },
        CrossClassCandidate {
            kind: AutoTopologyKind::Torus,
            negative_log_evidence: torus_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: torus_density_provider(data.view()),
        },
        CrossClassCandidate {
            kind: AutoTopologyKind::Union {
                structure: union_structure,
            },
            negative_log_evidence: union_evidence,
            certification: EvidenceCertification::Exact,
            density_provider: union_density_provider(data.view(), union_structure, cfg),
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
        "a race mixing smooth rungs with a structured union must be cross-class"
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

    // Column order matches the candidate vec: [Circle, Torus, Union].
    let union_weight = stacking.weights[2];
    let best_pure_weight = stacking.weights[0].max(stacking.weights[1]);

    RaceOutcome {
        winner_name: verdict.candidate_names[verdict.winner_index].clone(),
        headline: verdict.headline,
        union_weight,
        best_pure_weight,
        union_structure,
    }
}

/// Closed-form single-circle (ring) rank-aware negative-log-evidence: a 4-param
/// ring (centre x/y, radius mean, radial variance) plus uniform angle, on the
/// `-loglik + 1/2 P log n` BIC-form scale. Reported as corroboration.
fn ring_negative_log_evidence(data: ArrayView2<'_, f64>) -> f64 {
    let n = data.nrows();
    let cx = (0..n).map(|i| data[[i, 0]]).sum::<f64>() / n as f64;
    let cy = (0..n).map(|i| data[[i, 1]]).sum::<f64>() / n as f64;
    let r: Vec<f64> = (0..n)
        .map(|i| ((data[[i, 0]] - cx).powi(2) + (data[[i, 1]] - cy).powi(2)).sqrt())
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
    let p = 4.0_f64; // cx, cy, radius mean, radial variance
    -loglik + 0.5 * p * (n as f64).ln()
}

/// Closed-form single-torus (one anisotropic 2-D Gaussian chart) rank-aware
/// negative-log-evidence on the `-loglik + 1/2 P log n` scale with `P = 5`
/// (mean x/y + 3 covariance entries). Reported as corroboration.
fn torus_negative_log_evidence(data: ArrayView2<'_, f64>) -> f64 {
    let n = data.nrows();
    let mx = (0..n).map(|i| data[[i, 0]]).sum::<f64>() / n as f64;
    let my = (0..n).map(|i| data[[i, 1]]).sum::<f64>() / n as f64;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..n {
        let dx = data[[i, 0]] - mx;
        let dy = data[[i, 1]] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    sxx = (sxx / n as f64).max(1e-9);
    syy = (syy / n as f64).max(1e-9);
    sxy /= n as f64;
    let det = (sxx * syy - sxy * sxy).max(1e-12);
    let inv_xx = syy / det;
    let inv_yy = sxx / det;
    let inv_xy = -sxy / det;
    let log_norm = -(std::f64::consts::TAU).ln() - 0.5 * det.ln();
    let mut loglik = 0.0_f64;
    for i in 0..n {
        let dx = data[[i, 0]] - mx;
        let dy = data[[i, 1]] - my;
        let q = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
        loglik += log_norm - 0.5 * q;
    }
    let p = 5.0_f64; // mean x/y + 3 covariance entries
    -loglik + 0.5 * p * (n as f64).ln()
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn two_circles_prefer_structured_union_over_single_torus_and_circle() {
    let seeds = [7_u64, 23, 41, 89];
    for &seed in &seeds {
        let data = sample_two_circles(seed);
        let outcome = run_race(&data);
        assert_eq!(
            outcome.headline,
            Headline::Stacking,
            "seed {seed}: cross-class headline must be stacking"
        );
        assert!(
            outcome.winner_name.starts_with("union"),
            "seed {seed}: two-circles data — headline winner must be a structured union, got {}",
            outcome.winner_name
        );
        assert!(
            outcome.union_weight > outcome.best_pure_weight,
            "seed {seed}: two-circles data — the union must carry more stacking mass than any \
             pure rung (union_w={:.4}, best_pure_w={:.4}, structure={})",
            outcome.union_weight,
            outcome.best_pure_weight,
            outcome.union_structure.as_str(),
        );
        // The recovered composite should be the circle+circle one (two loops).
        assert_eq!(
            outcome.union_structure,
            UnionStructure::CircleCircle,
            "seed {seed}: two-circles data — the in-class union winner should be circle+circle, \
             got {}",
            outcome.union_structure.as_str(),
        );
    }
}

#[test]
fn circle_plus_outlier_cluster_prefers_structured_union_over_pure_rungs() {
    let seeds = [13_u64, 37, 59, 97];
    for &seed in &seeds {
        let data = sample_circle_plus_cluster(seed);
        let outcome = run_race(&data);
        assert_eq!(
            outcome.headline,
            Headline::Stacking,
            "seed {seed}: cross-class headline must be stacking"
        );
        assert!(
            outcome.winner_name.starts_with("union"),
            "seed {seed}: circle+cluster data — headline winner must be a structured union, got {}",
            outcome.winner_name
        );
        assert!(
            outcome.union_weight > outcome.best_pure_weight,
            "seed {seed}: circle+cluster data — the union must carry more stacking mass than any \
             pure rung (union_w={:.4}, best_pure_w={:.4}, structure={})",
            outcome.union_weight,
            outcome.best_pure_weight,
            outcome.union_structure.as_str(),
        );
    }
}

#[test]
fn single_circle_negative_control_does_not_prefer_any_union() {
    // NEGATIVE CONTROL: data from ONE circle. The complexity pricing (total
    // parameter count summed across components) must earn its keep — a pure rung
    // carries the headline and no union out-stacks the best pure rung.
    let seeds = [5_u64, 19, 43, 83];
    for &seed in &seeds {
        let data = sample_single_circle(seed);
        let outcome = run_race(&data);
        assert_eq!(
            outcome.headline,
            Headline::Stacking,
            "seed {seed}: cross-class headline must be stacking"
        );
        assert!(
            !outcome.winner_name.starts_with("union"),
            "seed {seed}: single-circle data — no union should win; got union winner {}",
            outcome.winner_name
        );
        assert!(
            outcome.union_weight < outcome.best_pure_weight,
            "seed {seed}: single-circle data — the over-priced union must NOT out-stack the best \
             pure rung (union_w={:.4}, best_pure_w={:.4}, structure={})",
            outcome.union_weight,
            outcome.best_pure_weight,
            outcome.union_structure.as_str(),
        );
    }
}

#[test]
fn fit_union_candidate_prices_by_total_parameter_count() {
    // The union evidence is the SUM of component rank-aware evidences and the
    // complexity price is the TOTAL free-parameter count across components.
    let data = sample_two_circles(7);
    let cfg = GaussianMixtureConfig::default();
    let fit = fit_union_candidate(data.view(), UnionStructure::CircleCircle, cfg)
        .expect("circle+circle union must fit");
    assert_eq!(
        fit.structure,
        UnionStructure::CircleCircle,
        "structure must round-trip"
    );
    // Two circle components, 4 parameters each → 8 total.
    assert_eq!(
        fit.total_parameters, 8,
        "circle+circle total parameter count must be the sum across components (4 + 4)"
    );
    assert_eq!(
        fit.total_parameters, fit.fit.total_parameters,
        "rung total_parameters must mirror the inner UnionStructureFit"
    );
    // Summed evidence equals the sum of the per-component negative-log-evidences.
    let summed: f64 = fit
        .fit
        .components
        .iter()
        .map(|c| c.negative_log_evidence)
        .sum();
    assert!(
        (fit.negative_log_evidence - summed).abs() <= 1e-9 * (1.0 + summed.abs()),
        "union negative-log-evidence must be the SUM of component evidences \
         (got {:.6}, sum {:.6})",
        fit.negative_log_evidence,
        summed,
    );
    assert!(
        fit.negative_log_evidence.is_finite(),
        "summed rank-aware Laplace evidence must be finite"
    );
    let total_component_params: usize = fit.fit.components.iter().map(|c| c.num_parameters).sum();
    assert_eq!(
        fit.total_parameters, total_component_params,
        "total_parameters must equal the sum of per-component parameter counts"
    );
}
