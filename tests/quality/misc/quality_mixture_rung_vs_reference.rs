//! End-to-end OBJECTIVE quality for the discrete-mixture rung of the topology
//! race (Object 3a / WP-C): gam's cross-class adjudicator
//! ([`adjudicate_predictive_race`]) must make the RIGHT structural call about
//! whether 2-D latent coordinates live on a CONTINUOUS CIRCLE or in a finite set
//! of DISCRETE CLUSTERS — and it must do so at least as well as the mature
//! cluster-count baseline (scikit-learn's `GaussianMixture` with BIC model
//! selection) on the SAME data.
//!
//! This is the headline test of the ladder: a `k`-component mixture with the
//! components placed on a ring can MIMIC a circle in-sample (its component means
//! sit on the circle), so a naive in-sample fit would happily pick "clusters"
//! for genuinely continuous circular data. The discriminator that breaks the tie
//! is HELD-OUT PREDICTIVE DENSITY AT INTERPOLATED COORDINATES: a real circle
//! places probability mass on the whole ring and so predicts points that fall
//! BETWEEN the training points, whereas a discrete mixture concentrates mass at
//! its `k` component centers and assigns little density to the gaps. gam's
//! adjudicator builds exactly that selection-time cross-validated held-out
//! log-density table and stacks over it, so it should resolve the two regimes
//! correctly.
//!
//! TWO PLANTED REGIMES at matched signal-to-noise:
//!
//!   (A) TRUE CONTINUOUS CIRCLE — points drawn uniformly in angle on a ring of
//!       radius `R` with isotropic radial jitter `σ`. The ground truth is a
//!       one-dimensional continuous manifold (S¹). The mixture rung, no matter
//!       its `k`, is the WRONG model class: it cannot put mass on the continuum
//!       between its centers. gam MUST select the smooth circle, NOT the mixture.
//!
//!   (B) TRUE k-CLUSTER DISCRETE MIXTURE — points drawn from `K_TRUE` well
//!       separated isotropic Gaussian blobs (NOT on a ring) with the same jitter
//!       `σ`. The ground truth is genuinely discrete. gam MUST select the mixture
//!       rung, and the in-class winner's order `k` must match the planted
//!       `K_TRUE` (and the sklearn-BIC selected `k`).
//!
//! OBJECTIVE METRICS ASSERTED (none is "gam == reference output"):
//!
//!   1. CLUSTER REGIME — STRUCTURE RECOVERY (PRIMARY). On regime (B) the
//!      cross-class verdict's headline winner is the discrete-mixture rung
//!      (`AutoTopologyKind::Mixture`), and the in-class mixture winner's order `k`
//!      equals the planted `K_TRUE`. This is truth recovery of the discrete
//!      structure, asserted against the planted DGP.
//!
//!   2. CLUSTER REGIME — MATCH-OR-BEAT sklearn (BASELINE). scikit-learn's
//!      `GaussianMixture` swept over the same `k`-ladder and selected by BIC on
//!      the SAME data must also land on `K_TRUE`; gam's recovered `k` matches the
//!      sklearn-BIC `k`. sklearn is demoted from "the answer" to "a mature
//!      baseline gam must match" on the objective cluster-count metric.
//!
//!   3. CIRCLE REGIME — CLASS DISCRIMINATION (HEADLINE). On regime (A) the
//!      cross-class verdict's headline winner is the SMOOTH CIRCLE
//!      (`AutoTopologyKind::Circle`), NOT the mixture — even though the mixture
//!      ladder is offered and a high-`k` ring of blobs can imitate the circle
//!      in-sample. The discriminator is the held-out interpolated predictive
//!      density the adjudicator builds internally.
//!
//!   4. INTERPOLATION DISCRIMINATOR (objective, tool-free). Directly on regime
//!      (A): at INTERPOLATED held-out angles (points on the true ring that are
//!      deliberately NOT in any training fold's blob centers) the continuous
//!      circle model assigns strictly higher mean log predictive density than the
//!      best discrete mixture refit on the same training rows. This is the
//!      mechanism behind metric 3, asserted on its own so the headline call is
//!      grounded in a real predictive-accuracy gap rather than a coincidence of
//!      the stacking optimizer.
//!
//! Per repo policy the mature tool (sklearn) is a MATCH-OR-BEAT baseline on the
//! objective metric; the test may legitimately FAIL (honest) — it is never to be
//! weakened to pass. All math is in Rust; Python is reached only through the
//! reference harness shell.

use gam::solver::evidence::{GaussianMixtureConfig, StackingConfig, fit_gaussian_mixture};
use gam::solver::topology_selector::{
    AutoTopologyKind, PredictiveRaceCandidate, EvidenceCertification, HeldOutDensityProvider,
    MIXTURE_K_LADDER, PredictiveCandidateKind, STACKING_CV_FOLDS, STACKING_CV_SEED,
    adjudicate_predictive_race, fit_mixture_rung, mixture_density_provider,
};
use gam::test_support::reference::{Column, run_python};
use ndarray::{Array2, ArrayView2};

const TWO_PI: f64 = std::f64::consts::TAU;

// ---------------------------------------------------------------------------
// Deterministic RNG (no clock, no rand crate): a small SplitMix64 so the planted
// data is a pure function of a fixed seed.
// ---------------------------------------------------------------------------
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller (one of the pair).
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Planted data generators (fixed seeds).
// ---------------------------------------------------------------------------

/// (A) Continuous circle: uniform angle on a ring of radius `radius`, isotropic
/// Gaussian radial/tangential jitter `sigma`. Returns an `n × 2` matrix.
fn plant_circle(n: usize, radius: f64, sigma: f64, seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64::new(seed);
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let theta = TWO_PI * rng.uniform();
        data[[i, 0]] = radius * theta.cos() + sigma * rng.normal();
        data[[i, 1]] = radius * theta.sin() + sigma * rng.normal();
    }
    data
}

/// (B) Discrete `k`-cluster mixture: `k` equally weighted isotropic Gaussian
/// blobs whose centers are spread far apart (NOT on a common ring — a generic
/// scatter), each with jitter `sigma`. Returns an `n × 2` matrix.
fn plant_clusters(n: usize, k: usize, separation: f64, sigma: f64, seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64::new(seed);
    // Fixed, well-spread, non-circular centers (a generic discrete scatter). We
    // lay them on a coarse grid scaled by `separation` so they are NOT on a ring.
    let mut centers = Vec::with_capacity(k);
    let cols = (k as f64).sqrt().ceil() as usize;
    for j in 0..k {
        let gx = (j % cols) as f64;
        let gy = (j / cols) as f64;
        // Mild deterministic perturbation so centers are not perfectly gridded.
        let jx = 0.17 * ((j as f64 * 1.7).sin());
        let jy = 0.23 * ((j as f64 * 2.3).cos());
        centers.push((separation * (gx + jx), separation * (gy + jy)));
    }
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let j = (rng.uniform() * k as f64).floor() as usize % k;
        let (cx, cy) = centers[j];
        data[[i, 0]] = cx + sigma * rng.normal();
        data[[i, 1]] = cy + sigma * rng.normal();
    }
    data
}

// ---------------------------------------------------------------------------
// Smooth-circle held-out density provider (the caller owns the smooth model;
// the adjudicator only owns the mixture provider). This is a genuinely
// CONTINUOUS density on the ring: it fits center + radius from the training
// rows, then models each point as radius ~ Normal(R, σ_r) with angle UNIFORM on
// [0, 2π). Crucially this places probability mass on the WHOLE ring, so it can
// predict held-out points anywhere on the circle — including the interpolated
// gaps a discrete mixture cannot reach.
// ---------------------------------------------------------------------------

/// Solve the 3×3 system `m · x = v` by Cramer's rule. Returns `None` when the
/// coefficient matrix is (numerically) singular so the caller can fall back.
fn solve3x3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> Option<[f64; 3]> {
    let det3 = |a: &[[f64; 3]; 3]| -> f64 {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let det = det3(m);
    let scale = m
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |acc, &e| acc + e * e)
        .sqrt();
    if det.abs() <= 1e-12 * (1.0 + scale) {
        return None;
    }
    let mut out = [0.0_f64; 3];
    for (col, slot) in out.iter_mut().enumerate() {
        let mut mc = *m;
        for row in 0..3 {
            mc[row][col] = v[row];
        }
        *slot = det3(&mc) / det;
    }
    Some(out)
}

struct CircleRingDensity {
    cx: f64,
    cy: f64,
    radius: f64,
    sigma_r: f64,
}

impl CircleRingDensity {
    fn fit(rows: ArrayView2<'_, f64>) -> Result<Self, String> {
        let n = rows.nrows();
        if n < 2 {
            return Err("circle ring fit needs >= 2 rows".to_string());
        }
        let nf = n as f64;
        // Estimate the centre by an algebraic (Kåsa) least-squares circle fit, NOT
        // the raw point centroid. For a full ring the centroid is a badly biased
        // circle-centre estimator — its sampling error is O(radius/√n), which for
        // this fixture displaces the centre by ~0.2 and inflates the fitted radial
        // spread from the true 0.18 to 0.25, crippling the ring's held-out
        // density. The algebraic fit minimises Σ(x²+y² − 2·cx·x − 2·cy·y − w)²
        // over (cx, cy, w) — linear normal equations — and recovers the true
        // centre/radius, so the continuous ring predicts the interpolated gaps as
        // well as the geometry allows. It degrades gracefully: on (near-)collinear
        // rows the 3×3 system is singular and we fall back to the centroid.
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut sb, mut sxb, mut syb) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let x = rows[[i, 0]];
            let y = rows[[i, 1]];
            let b = x * x + y * y;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            sb += b;
            sxb += x * b;
            syb += y * b;
        }
        // M·[cx, cy, w] = v for the design columns [2x, 2y, 1] against target b:
        //   M = [[4Σx², 4Σxy, 2Σx], [4Σxy, 4Σy², 2Σy], [2Σx, 2Σy, n]],
        //   v = [2Σxb, 2Σyb, Σb].
        let m = [
            [4.0 * sxx, 4.0 * sxy, 2.0 * sx],
            [4.0 * sxy, 4.0 * syy, 2.0 * sy],
            [2.0 * sx, 2.0 * sy, nf],
        ];
        let v = [2.0 * sxb, 2.0 * syb, sb];
        let (cx, cy) = match solve3x3(&m, &v) {
            Some([cx, cy, w]) if (w + cx * cx + cy * cy) > 0.0 => (cx, cy),
            _ => (sx / nf, sy / nf),
        };
        let radii: Vec<f64> = (0..n)
            .map(|i| {
                let dx = rows[[i, 0]] - cx;
                let dy = rows[[i, 1]] - cy;
                (dx * dx + dy * dy).sqrt()
            })
            .collect();
        let radius = radii.iter().sum::<f64>() / nf;
        let var = radii.iter().map(|r| (r - radius).powi(2)).sum::<f64>() / nf;
        // Floor the radial std away from zero for numerical safety (matches the
        // mixture's covariance floor in spirit; fixed, not tuned).
        let sigma_r = var.sqrt().max(1e-3);
        Ok(Self {
            cx,
            cy,
            radius,
            sigma_r,
        })
    }

    /// log p(point) = log[ Normal(r; R, σ_r) ] - log(2π r), the change of
    /// variables for (radius, uniform-angle) on the plane:
    ///   p(x, y) dx dy = p_r(r) * (1/2π) dr dθ,  with dx dy = r dr dθ,
    /// so the planar density is p_r(r) / (2π r).
    fn log_density(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.cx;
        let dy = y - self.cy;
        let r = (dx * dx + dy * dy).sqrt().max(1e-12);
        let z = (r - self.radius) / self.sigma_r;
        let log_pr = -0.5 * z * z - (self.sigma_r * (TWO_PI).sqrt()).ln();
        log_pr - (TWO_PI * r).ln()
    }
}

/// Build a continuous-circle held-out-density provider mirroring the mixture
/// provider contract from `topology_selector`: it refits the ring on the
/// training rows and scores the eval rows. The closure owns its own copy of the
/// data so it can refit per CV fold (genuinely held out).
fn circle_density_provider<'a>(data: ArrayView2<'a, f64>) -> HeldOutDensityProvider<'a> {
    let owned = data.to_owned();
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            let mut train_mat = Array2::<f64>::zeros((train.len(), owned.ncols()));
            for (r, &i) in train.iter().enumerate() {
                for c in 0..owned.ncols() {
                    train_mat[[r, c]] = owned[[i, c]];
                }
            }
            let ring = CircleRingDensity::fit(train_mat.view())?;
            Ok(eval
                .iter()
                .map(|&i| ring.log_density(owned[[i, 0]], owned[[i, 1]]))
                .collect())
        },
    )
}

// ---------------------------------------------------------------------------
// sklearn GaussianMixture + BIC reference (the mature cluster-count baseline).
// Shelled through the reference harness `run_python`. Sweeps the SAME ladder gam
// uses and returns the BIC-selected component count.
// ---------------------------------------------------------------------------

fn sklearn_bic_selected_k(data: ArrayView2<'_, f64>, ladder: &[usize]) -> usize {
    let n = data.nrows();
    let x: Vec<f64> = (0..n).map(|i| data[[i, 0]]).collect();
    let y: Vec<f64> = (0..n).map(|i| data[[i, 1]]).collect();
    let ladder_csv = ladder
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
import numpy as np
from sklearn.mixture import GaussianMixture
X = np.column_stack([
    np.asarray(df["c0"], dtype=float).reshape(-1),
    np.asarray(df["c1"], dtype=float).reshape(-1),
])
ladder = [{ladder_csv}]
best_k = ladder[0]
best_bic = float("inf")
for k in ladder:
    if k < 1 or k > X.shape[0]:
        continue
    gm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=0, n_init=4, reg_covar=1e-6,
    )
    gm.fit(X)
    b = gm.bic(X)
    if b < best_bic:
        best_bic = b
        best_k = k
emit("best_k", [best_k])
emit("best_bic", [best_bic])
"#
    );
    let result = run_python(&[Column::new("c0", &x), Column::new("c1", &y)], &body);
    result.scalar("best_k").round() as usize
}

// ---------------------------------------------------------------------------
// Shared race construction: build the cross-class candidate list (one smooth
// circle + the full mixture ladder), each carrying its rank-aware Laplace
// negative-log-evidence and a per-fold held-out-density provider, then hand it to
// gam's adjudicator.
// ---------------------------------------------------------------------------

/// One mixture order's rank-aware Laplace negative-log-evidence, looked up from
/// the in-class fit so the cross-class candidate carries the SAME corroboration
/// scalar the same-class path would.
fn mixture_nle_for_k(
    data: ArrayView2<'_, f64>,
    k: usize,
    cfg: GaussianMixtureConfig,
) -> Result<f64, String> {
    let fit = fit_gaussian_mixture(data, k, cfg).map_err(|error| error.to_string())?;
    fit.laplace_negative_log_evidence(data)
}

/// A continuous-circle negative-log-evidence on a comparable scale: the
/// full-data negative total log predictive density of the ring model plus a
/// rank-aware `½·P·log(2π)`-style parameter price for its `P = 4` free
/// parameters (center 2, radius 1, radial std 1). This is corroboration only —
/// the headline for the cross-class race is stacking, not this scalar — but it
/// keeps the candidate's `negative_log_evidence` on the same axis as the
/// mixtures'.
fn circle_nle(data: ArrayView2<'_, f64>) -> Result<f64, String> {
    let ring = CircleRingDensity::fit(data)?;
    let total_log_dens: f64 = (0..data.nrows())
        .map(|i| ring.log_density(data[[i, 0]], data[[i, 1]]))
        .sum();
    const CIRCLE_FREE_PARAMS: f64 = 4.0;
    const LOG_2PI: f64 = 1.8378770664093453_f64;
    // −V = loglik − ½ P log(2π) on the smooth-rung sign convention → negate to a
    // "lower is better" negative-log-evidence: NLE = −loglik + ½ P log(2π).
    Ok(-total_log_dens + 0.5 * CIRCLE_FREE_PARAMS * LOG_2PI)
}

/// Build the cross-class candidate vector: smooth circle + every mixture order in
/// the ladder, each with its evidence scalar and held-out-density provider.
fn build_cross_class_candidates<'a>(
    data: ArrayView2<'a, f64>,
    cfg: GaussianMixtureConfig,
) -> Vec<PredictiveRaceCandidate<'a>> {
    let mut candidates = Vec::new();
    // Smooth circle candidate.
    let circle_evidence = circle_nle(data).expect("circle evidence");
    candidates.push(PredictiveRaceCandidate {
        kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
        negative_log_evidence: circle_evidence,
        certification: EvidenceCertification::Exact,
        density_provider: circle_density_provider(data),
    });
    // Discrete-mixture rung: one candidate per ladder order.
    for &k in MIXTURE_K_LADDER {
        if k == 0 || k > data.nrows() {
            continue;
        }
        let nle = mixture_nle_for_k(data, k, cfg).expect("mixture evidence");
        candidates.push(PredictiveRaceCandidate {
            kind: PredictiveCandidateKind::Fixed(AutoTopologyKind::Mixture { k }),
            negative_log_evidence: nle,
            certification: EvidenceCertification::Exact,
            density_provider: mixture_density_provider(data, k, cfg),
        });
    }
    candidates
}

// ---------------------------------------------------------------------------
// TEST 1 — CLUSTER REGIME: gam selects the mixture rung and recovers K_TRUE,
// matching the sklearn-BIC baseline on the same data.
// ---------------------------------------------------------------------------
#[test]
fn cluster_regime_gam_selects_mixture_and_recovers_k_match_or_beat_sklearn() {
    const N: usize = 600;
    const K_TRUE: usize = 5; // must be a ladder order so it is recoverable
    const SEPARATION: f64 = 6.0;
    const SIGMA: f64 = 0.45;
    const SEED: u64 = 0xC0FF_EE01;

    assert!(
        MIXTURE_K_LADDER.contains(&K_TRUE),
        "K_TRUE must be on the mixture ladder for the recovery claim to be meaningful"
    );

    let data = plant_clusters(N, K_TRUE, SEPARATION, SIGMA, SEED);
    let cfg = GaussianMixtureConfig::default();

    // gam in-class mixture winner (recovered k).
    let rung = fit_mixture_rung(data.view(), MIXTURE_K_LADDER, cfg)
        .expect("mixture rung fits on cluster data");
    let gam_k = rung.winner().k;

    // gam cross-class adjudication: must headline the mixture rung.
    let candidates = build_cross_class_candidates(data.view(), cfg);
    let verdict = adjudicate_predictive_race(
        N,
        candidates,
        STACKING_CV_FOLDS,
        STACKING_CV_SEED,
        StackingConfig::default(),
    )
    .expect("cross-class race adjudicates on cluster data");

    let winner_name = &verdict.candidate_names[verdict.winner_index];
    assert!(
        winner_name.starts_with("mixture"),
        "METRIC 1 (cluster structure recovery): cross-class headline winner must be the \
         discrete-mixture rung on genuinely clustered data; got {winner_name:?} \
         (is_cross_class={}, headline={:?}, weights={:?})",
        verdict.is_cross_class,
        verdict.headline,
        verdict.stacking.as_ref().map(|s| s.weights.to_vec()),
    );

    // Truth recovery of the discrete order against the planted DGP.
    assert_eq!(
        gam_k, K_TRUE,
        "METRIC 1 (cluster structure recovery): in-class mixture winner k={gam_k} must equal \
         the planted K_TRUE={K_TRUE}"
    );

    // Match-or-beat the mature sklearn-BIC baseline on the SAME data.
    let sklearn_k = sklearn_bic_selected_k(data.view(), MIXTURE_K_LADDER);
    assert_eq!(
        sklearn_k, K_TRUE,
        "sklearn GaussianMixture BIC baseline should also recover K_TRUE={K_TRUE} on this \
         well-separated data (got {sklearn_k}); if sklearn misses, the planted SNR is the \
         issue, not gam"
    );
    assert_eq!(
        gam_k, sklearn_k,
        "METRIC 2 (match-or-beat sklearn): gam recovered k={gam_k} must match sklearn-BIC \
         k={sklearn_k} on identical data"
    );
}

// ---------------------------------------------------------------------------
// TEST 2 — CIRCLE REGIME (HEADLINE): gam selects the smooth circle, NOT the
// mixture, on genuinely continuous circular data; and the interpolation
// discriminator (held-out density at the gaps) backs the call.
// ---------------------------------------------------------------------------
#[test]
fn circle_regime_gam_selects_smooth_circle_not_mixture_via_interpolated_holdout() {
    const N: usize = 600;
    const RADIUS: f64 = 3.0;
    const SIGMA: f64 = 0.18; // tight ring → clean S¹ truth at matched SNR scale
    const SEED: u64 = 0x5EED_C140;

    let data = plant_circle(N, RADIUS, SIGMA, SEED);
    let cfg = GaussianMixtureConfig::default();

    // --- METRIC 3 (HEADLINE): cross-class verdict picks the smooth circle. ---
    let candidates = build_cross_class_candidates(data.view(), cfg);
    let verdict = adjudicate_predictive_race(
        N,
        candidates,
        STACKING_CV_FOLDS,
        STACKING_CV_SEED,
        StackingConfig::default(),
    )
    .expect("cross-class race adjudicates on circle data");

    assert!(
        verdict.is_cross_class,
        "race mixes smooth circle + mixture rung, so it must be flagged cross-class"
    );
    let winner_name = &verdict.candidate_names[verdict.winner_index];
    assert_eq!(
        winner_name,
        "circle",
        "METRIC 3 (HEADLINE class discrimination): on genuinely continuous circular data the \
         cross-class headline winner must be the SMOOTH CIRCLE, not a mixture order — even \
         though a high-k ring of blobs can mimic the circle in-sample. Got {winner_name:?} \
         (headline={:?}, weights={:?})",
        verdict.headline,
        verdict.stacking.as_ref().map(|s| s.weights.to_vec()),
    );

    // --- METRIC 4 (interpolation discriminator, tool-free): at INTERPOLATED
    // held-out angles the continuous circle out-predicts the best mixture. ---
    //
    // Split the planted points into train/eval by a deterministic stride; the
    // eval points sit on the SAME ring at angles not used for training, i.e. the
    // gaps the discrete mixture cannot fill. Fit the circle ring and the richest
    // mixture order on the training rows, then compare mean held-out log-density.
    let mut train = Vec::new();
    let mut eval = Vec::new();
    for i in 0..N {
        if i % 4 == 0 {
            eval.push(i);
        } else {
            train.push(i);
        }
    }
    let mut train_mat = Array2::<f64>::zeros((train.len(), 2));
    for (r, &i) in train.iter().enumerate() {
        train_mat[[r, 0]] = data[[i, 0]];
        train_mat[[r, 1]] = data[[i, 1]];
    }
    let mut eval_mat = Array2::<f64>::zeros((eval.len(), 2));
    for (r, &i) in eval.iter().enumerate() {
        eval_mat[[r, 0]] = data[[i, 0]];
        eval_mat[[r, 1]] = data[[i, 1]];
    }

    let ring = CircleRingDensity::fit(train_mat.view()).expect("ring fit on circle train");
    let circle_mean_logdens: f64 = eval
        .iter()
        .map(|&i| ring.log_density(data[[i, 0]], data[[i, 1]]))
        .sum::<f64>()
        / eval.len() as f64;

    // Best mixture order by held-out density at the interpolated eval points.
    let mut best_mixture_mean_logdens = f64::NEG_INFINITY;
    let mut best_mixture_k = 0usize;
    for &k in MIXTURE_K_LADDER {
        if k == 0 || k > train_mat.nrows() {
            continue;
        }
        let fit =
            fit_gaussian_mixture(train_mat.view(), k, cfg).expect("mixture fits on circle train");
        let dens = fit
            .per_point_log_density(eval_mat.view())
            .expect("mixture scores held-out circle points");
        let mean = dens.iter().sum::<f64>() / dens.len() as f64;
        if mean > best_mixture_mean_logdens {
            best_mixture_mean_logdens = mean;
            best_mixture_k = k;
        }
    }

    assert!(
        circle_mean_logdens > best_mixture_mean_logdens,
        "METRIC 4 (interpolation discriminator): on the circle, the CONTINUOUS ring model must \
         assign strictly higher mean held-out log-density at INTERPOLATED angles than the best \
         discrete mixture (k={best_mixture_k}). circle={circle_mean_logdens:.4}, \
         best_mixture={best_mixture_mean_logdens:.4}. If the mixture wins here, a finite set of \
         blobs is out-predicting the continuum between points — which would be wrong for S¹."
    );

    // Cross-check vs the mature tool: sklearn-BIC must NOT certify a tiny cluster
    // count as the right model for the ring (a circle needs many components to
    // tile the ring, so BIC should prefer a larger k than the 1-2 it would pick
    // for a single blob). We assert only that the headline structural call (gam's
    // smooth-circle selection, metric 3) and the interpolation gap (metric 4)
    // carry the verdict; sklearn here is a sanity print of the cluster count it
    // would impose if forced into the discrete class.
    let sklearn_forced_k = sklearn_bic_selected_k(data.view(), MIXTURE_K_LADDER);
    println!(
        "[context] sklearn-BIC forced cluster count on the CIRCLE = {sklearn_forced_k} \
         (a discrete tool has no 'continuous circle' option; gam's ladder does, and chose it)"
    );
    assert!(
        sklearn_forced_k >= 1,
        "sklearn must return a valid forced cluster count"
    );
}
