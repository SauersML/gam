//! End-to-end stress test for the closed-form Duchon pipeline at
//! large-scale-relevant scale.
//!
//! Runs in the default suite. The fit can take many minutes and use a
//! lot of memory — run under `--release` if iteration time matters:
//!
//! ```text
//! cargo test --release large_scale_reml_stress
//! ```
//!
//! It exercises the full Duchon-on-PC GAM pipeline end-to-end:
//!
//!   * Pure-Rust deterministic large-scale-style simulator producing
//!     `n` rows of `pc_dim` PC features sampled from N(0, I) and a
//!     continuous response `y = f_true(X) + ε`.
//!   * Hybrid anisotropic Duchon smooth (`length_scale = Some(...)`,
//!     `aniso_log_scales = Some(zeros)`) with `K` farthest-point centers.
//!   * REML/LAML outer loop must converge.
//!   * Held-out-grid relative L2 reconstruction error must be < 0.10.
//!   * Bias-corrected predictions must be available on `FitInference`
//!     and finite.
//!   * 95% prediction-interval coverage on held-out samples must
//!     exceed 0.85 across `N_COVERAGE_SIMS` independent simulations.
//!   * Each fit must terminate on convergence, strictly inside the outer
//!     iteration budget it was configured with.
//!
//! All randomness is seeded; failures are reproducible.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    duchon_max_active_operator_derivative_order, resolve_duchon_orders,
};
use gam::estimate::{FitOptions, UnifiedFitResult};
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec, build_term_collection_design, fit_term_collection_forspec,
    fit_term_collectionwith_spatial_length_scale_optimization,
    freeze_term_collection_from_design,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

// ─── Test scale knobs ───────────────────────────────────────────────────
//
// `N_TRAIN`, `K_CENTERS`, and `PC_DIM` are deliberately moderate so the
// test is feasible at all in a default `--release` run on a developer
// box. The team-lead spec calls out `K∈{500,1000}` (start small) and
// `n` in the 50K-300K range; `n=50_000` and `K=500` are the lower end
// of that range. Crank up by editing these constants.
const N_TRAIN: usize = 50_000;
const N_HOLDOUT: usize = 4_000;
const PC_DIM: usize = 6;
const K_CENTERS: usize = 500;
const NOISE_SD: f64 = 0.30;
const SEED_BASE: u64 = 0xB10B_0001_0001_0001;
// The coverage claim is a POOLED statistic: every per-row interval from every
// sim is aggregated into one `total_in / total_pts` fraction. With
// `N_COVERAGE_SIMS × N_COVERAGE_HOLDOUT` = 8 × 400 = 3200 pooled points, the
// empirical-coverage standard error is √(0.95·0.05/3200) ≈ 0.004, so the
// `coverage > 0.85` bound (true coverage ≈ 0.95) keeps a >20·SE margin. Reduced
// from 20 to 8 sims to keep this test under the 300s nextest SLOW budget; the
// pooled coverage estimate stays tight and the assertion is unchanged.
const N_COVERAGE_SIMS: usize = 8;
const N_COVERAGE_TRAIN: usize = 4_000;
const N_COVERAGE_HOLDOUT: usize = 400;
const K_COVERAGE: usize = 80;
const PC_DIM_COVERAGE: usize = 4;

// Work ceilings. These replace the wall-clock ceilings this file used to
// assert (1800s for the main fit, 120s per coverage fit). A wall-clock
// assertion on a shared CI runner measures the runner, not the solver, so it
// flakes in both directions; and the 1800s one could never fire at all,
// because the harness SIGKILLs the target long before half an hour elapses —
// dead code wearing the shape of a budget. What the fits are actually
// supposed to demonstrate is that the REML outer loop CONVERGES rather than
// grinding to its cap, and `max_iter` (the cap the fit is configured with)
// is the machine-independent statement of exactly that. No new magic
// constants: the ceiling is the fit's own configured budget.
const MAIN_MAX_ITER: usize = 40;
const COVERAGE_MAX_ITER: usize = 30;
const NORMAL_95_TWO_SIDED_Z: f64 = 1.959_963_984_540_054;

/// The two-sided standard-normal mass inside `NORMAL_95_TWO_SIDED_Z`. Bound to
/// a constant so the level the interval is BUILT at and the level it is SCORED
/// against cannot drift apart: change the `z` above and this is the number that
/// has to move with it.
const NOMINAL_COVERAGE: f64 = 0.95;

/// False-alarm budget for the two-sided coverage band, in standard errors of
/// the coverage estimate. It is not a tolerance on the statistics — the target
/// is `NOMINAL_COVERAGE` exactly and the standard error is computed from the
/// run — it is the rate at which a correctly calibrated fit is allowed to fail
/// this gate by chance, which at two-sided 3σ is 0.27% of runs.
const COVERAGE_BAND_SIGMAS: f64 = 3.0;

/// Held-out reconstruction bar, and gam#2735 — WHERE THIS NUMBER COMES FROM.
///
/// The value is unchanged at `0.10`. What it lacked was provenance: it entered
/// in `6a4b4728c`, the commit that created the test, and per #2709 the fixture
/// could not build until `da1228e1c`, so it had never once been evaluated. This
/// comment is the missing derivation, so it cannot be re-litigated from scratch
/// or "relaxed to wherever the code landed".
///
/// `relative_l2` normalises by the truth's SD, so this is `sqrt(1 - R²)`
/// against the NOISELESS truth; `0.10` is `R² > 0.99`. Measured references on
/// THIS design (n = `N_TRAIN`, `K_CENTERS` centres, `PC_DIM` dims,
/// `NOISE_SD` noise), 3–8 replicates each, hyper-parameters tuned on a
/// held-out split and verified to select an INTERIOR optimum:
///
/// | reference                                            | held-out `rel_l2` |
/// |------------------------------------------------------|-------------------|
/// | oracle OLS on the exact generating features           | 0.0043 ± 0.0007   |
/// | 500-centre kernel ridge, ANISOTROPIC per-axis scales  | **0.0616 ± 0.0007** |
/// | floor for any purely ADDITIVE smoother (closed form)  | 0.0821            |
/// | 500-centre kernel ridge, ISOTROPIC                    | 0.2290 ± 0.0014   |
/// | least-squares linear-only (this fixture's reference)  | 0.3521 ± 0.0042   |
///
/// The decisive row is the second: a 500-centre radial smoother with **learned
/// per-axis scales** — exactly what `aniso_log_scales` configures — reaches
/// `0.0616` on this data. So `0.10` is achievable by this fixture's own model
/// class with ~1.6x headroom, and it is NOT a bar that needs relaxing.
///
/// The isotropic row is why the diagnostic reports the fitted scales: an
/// isotropic 500-centre smoother in six dimensions can only reach `0.2290`, and
/// no bar near `0.10` is reachable from there. A fit landing near or above that
/// value has not earned its `K_CENTERS` basis functions — it has lost its
/// anisotropy.
///
/// The additive row is a floor, not an attainment: `truth()` contains
/// `exp(-|x-m|²/1.28) = Π_j exp(-(x_j-m_j)²/1.28)`, a product and therefore not
/// additive. With independent coordinates the additive projection is
/// `Σ_j E[f|x_j] - (d-1)E[f]`; the residual variance is `5.043e-3` against
/// `Var(truth) = 0.7477`, i.e. `rel_l2 ≥ 0.0821` for ANY additive smoother.
/// The Duchon smooth here is a full `PC_DIM`-dimensional smooth, so it is not
/// bound by that floor — it is quoted to show `0.10` is not below one.
const RECONSTRUCTION_REL_L2_MAX: f64 = 0.10;

/// The best measured held-out `rel_l2` from a smoother matched to this
/// fixture's OWN budget — 500 centres, learned per-axis scales — quoted in the
/// failure message so the reader sees the achievable number next to the miss
/// rather than having to trust the bar. See [`RECONSTRUCTION_REL_L2_MAX`].
const ANISO_REFERENCE_REL_L2: f64 = 0.0616;

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

// ─── Synthetic large-scale simulator ────────────────────────────────────────

/// Smooth ground-truth function on PC coordinates. Used both for
/// generating `y` and for evaluating reconstruction error.
///
/// The functional form mirrors the pipeline contract in
/// `production_pipeline_spec.md` and `large_scale_sim.py`: a sum of a
/// linear PC trend, a radial bump centered near the origin, and a
/// sinusoid on PC0. It is smooth, bounded, and not separable into
/// per-axis pieces — all properties an anisotropic Duchon smooth
/// should be able to track.
fn truth(row: &[f64]) -> f64 {
    let mut linear = 0.0;
    let coefs = [0.55, -0.40, 0.30, 0.20, -0.15, 0.10];
    for (j, &xj) in row.iter().enumerate() {
        if j < coefs.len() {
            linear += coefs[j] * xj;
        }
    }
    let mut dist2 = 0.0;
    for (j, &xj) in row.iter().enumerate() {
        let cj = match j {
            0 => 0.30,
            1 => -0.20,
            2 => 0.10,
            _ => 0.0,
        };
        let d = xj - cj;
        dist2 += d * d;
    }
    let radial_bump = 1.0 * (-dist2 / (2.0 * 0.8 * 0.8)).exp();
    let sinusoid = 0.4 * (std::f64::consts::PI * row[0]).sin();
    linear + radial_bump + sinusoid
}

/// Generate `(X, y, y_true)` with PC coordinates sampled iid from
/// the standard normal and `y = truth(X) + N(0, NOISE_SD²)`.
fn simulate(n: usize, pc_dim: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal params must be valid");
    let noise = Normal::new(0.0, NOISE_SD).expect("noise params must be valid");

    let mut x = Array2::<f64>::zeros((n, pc_dim));
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row = vec![0.0_f64; pc_dim];
        for j in 0..pc_dim {
            let v = normal.sample(&mut rng);
            x[[i, j]] = v;
            row[j] = v;
        }
        let f = truth(&row);
        y_true[i] = f;
        y[i] = f + noise.sample(&mut rng);
    }
    (x, y, y_true)
}

/// Length scale of the hybrid (Matérn-blended) Duchon kernel. Bound to a
/// constant because the SAME value has to reach both the order resolution
/// below and the spec: `resolve_duchon_orders` branches on
/// `length_scale.is_none()` (the pure-mode CPD constraint `2s < d` applies
/// only there), so resolving for one mode and building the other would resolve
/// against constraints the built kernel does not have.
const HYBRID_LENGTH_SCALE: f64 = 1.0;

/// Build the anisotropic-hybrid Duchon term spec used throughout the
/// test.
fn duchon_aniso_pc_spec(name: &str, pc_dim: usize, k_centers: usize) -> TermCollectionSpec {
    let operator_penalties = DuchonOperatorPenaltySpec::default();
    // The Duchon orders are RESOLVED from `pc_dim`, not written down (#2709).
    //
    // The pointwise kernel is the inverse Fourier of `1/|ξ|^{2(p+s)}`, finite
    // at the origin iff `2(p+s) > d` — a condition on the dimension. This
    // fixture hardcoded `power: 1.0` with a `Linear` nullspace (`p = 2`), i.e.
    // `2(p+s) = 6`, which holds at the sibling coverage test's `PC_DIM_COVERAGE
    // = 4` and fails at `PC_DIM = 6` on the equality `6 > 6`. So the main test
    // died in 0.21 s inside basis construction and never ran a single one of
    // its large-scale assertions, while the coverage test using the same
    // builder ran fine — a constant that was correct for one caller and
    // inadmissible for the other.
    //
    // `resolve_duchon_orders` is the library's own answer to that question: the
    // smallest admissible `(nullspace, s)` at this dimension, also clearing the
    // D1 collocation margin `2(p+s) > d+1` that the active tension penalty
    // needs. Deriving it here means the fixture follows `PC_DIM` instead of
    // being re-broken by the next edit to it, and it is the SAME resolution the
    // production paths use rather than a fixture-local rule that could agree
    // with nothing.
    //
    // At the two dimensions this test uses, and with the default penalties
    // (mass + tension active, stiffness disabled ⇒ max operator order 1):
    //   pc_dim = 4 → (Linear, s = 1), 2(p+s) = 6 > 5 — what the coverage test
    //                already built, so its behaviour is unchanged.
    //   pc_dim = 6 → (Linear, s = 2), 2(p+s) = 8 > 7 — the main test's repair.
    let (nullspace_order, power) = resolve_duchon_orders(
        pc_dim,
        DuchonNullspaceOrder::Linear,
        duchon_max_active_operator_derivative_order(&operator_penalties),
        Some(HYBRID_LENGTH_SCALE),
    );
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: name.to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..pc_dim).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: k_centers,
                    },
                    // Hybrid Duchon — required for aniso_log_scales.
                    length_scale: Some(HYBRID_LENGTH_SCALE),
                    power: power as f64,
                    nullspace_order,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: Some(vec![0.0; pc_dim]),
                    operator_penalties,

                    periodic: None,
                    boundary: gam::basis::OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options(max_iter: usize) -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter,
        tol: 1e-5,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

/// L2 relative error: ||pred - truth||₂ / ||truth - mean(truth)||₂.
/// Held-out `relative_l2` of the best predictor that uses NO curvature: an
/// ordinary least-squares fit of the truth on the raw coordinates plus an
/// intercept, solved on the held-out rows themselves.
///
/// Solved on the held-out rows deliberately, and it makes the reference
/// STRICTER rather than weaker: this is the best a linear model could possibly
/// do on exactly the rows being scored, with no estimation error of its own. A
/// smoother that does not clear it by a wide margin has not earned its basis.
///
/// Unlike a literal bar, this is re-derived from the data on every run, so it
/// cannot age against a changed fixture or drift out of calibration — the
/// failure mode that produced the un-evaluated `0.10` above.
fn linear_only_reference(x: ArrayView2<'_, f64>, truth: &Array1<f64>) -> f64 {
    let (n, d) = x.dim();
    // Normal equations for `[1, x]`, which is `(d+1)` square and tiny.
    let p = d + 1;
    let mut xtx = Array2::<f64>::zeros((p, p));
    let mut xty = Array1::<f64>::zeros(p);
    for row in 0..n {
        let mut z = Array1::<f64>::ones(p);
        for col in 0..d {
            z[col + 1] = x[[row, col]];
        }
        for a in 0..p {
            xty[a] += z[a] * truth[row];
            for b in 0..p {
                xtx[[a, b]] += z[a] * z[b];
            }
        }
    }
    // Gaussian elimination with partial pivoting; `p` is at most 7 here.
    let mut aug = Array2::<f64>::zeros((p, p + 1));
    for a in 0..p {
        for b in 0..p {
            aug[[a, b]] = xtx[[a, b]];
        }
        aug[[a, p]] = xty[a];
    }
    for col in 0..p {
        let mut pivot = col;
        for row in (col + 1)..p {
            if aug[[row, col]].abs() > aug[[pivot, col]].abs() {
                pivot = row;
            }
        }
        if aug[[pivot, col]].abs() < 1e-12 {
            // Degenerate design: fall back to the intercept-only predictor,
            // whose relative_l2 against a mean-centred truth is exactly 1.
            return 1.0;
        }
        if pivot != col {
            for b in col..=p {
                let tmp = aug[[col, b]];
                aug[[col, b]] = aug[[pivot, b]];
                aug[[pivot, b]] = tmp;
            }
        }
        for row in 0..p {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]] / aug[[col, col]];
            for b in col..=p {
                let v = aug[[col, b]] * factor;
                aug[[row, b]] -= v;
            }
        }
    }
    let mut beta = Array1::<f64>::zeros(p);
    for a in 0..p {
        beta[a] = aug[[a, p]] / aug[[a, a]];
    }
    let mut pred = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut acc = beta[0];
        for col in 0..d {
            acc += beta[col + 1] * x[[row, col]];
        }
        pred[row] = acc;
    }
    relative_l2(&pred, truth)
}

fn relative_l2(pred: &Array1<f64>, truth: &Array1<f64>) -> f64 {
    let mean_t = truth.mean().unwrap_or(0.0);
    let mut num = 0.0;
    let mut den = 0.0;
    for (p, t) in pred.iter().zip(truth.iter()) {
        let dp = p - t;
        let dt = t - mean_t;
        num += dp * dp;
        den += dt * dt;
    }
    (num / den.max(1e-30)).sqrt()
}

fn gaussian_identity_mean(
    design: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut mean = design.dot(&beta);
    mean += &offset;
    mean
}

fn gaussian_identity_bias_corrected_mean(
    design: ArrayView2<'_, f64>,
    fit: &UnifiedFitResult,
    offset: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let bias_correction = fit
        .inference
        .as_ref()
        .and_then(|inference| inference.bias_correction_beta.as_ref())
        .expect("FitInference must carry bias_correction_beta");
    let beta = &fit.beta + bias_correction;
    gaussian_identity_mean(design, beta.view(), offset)
}

fn gaussian_identity_bias_corrected_mean_interval(
    design: ArrayView2<'_, f64>,
    fit: &UnifiedFitResult,
    offset: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mean = gaussian_identity_bias_corrected_mean(design, fit, offset);
    let covariance = fit
        .beta_covariance_corrected()
        .expect("Gaussian identity coverage requires smoothing-corrected covariance");
    assert_eq!(covariance.nrows(), fit.beta.len());
    assert_eq!(covariance.ncols(), fit.beta.len());

    let mut eta_se = Array1::<f64>::zeros(design.nrows());
    for (i, row) in design.outer_iter().enumerate() {
        let cov_row = covariance.dot(&row);
        eta_se[i] = row.dot(&cov_row).max(0.0).sqrt();
    }
    let z_se = eta_se.mapv(|se| NORMAL_95_TWO_SIDED_Z * se);
    let lower = &mean - &z_se;
    let upper = &mean + &z_se;
    (mean, lower, upper)
}

// ─── Main stress test ───────────────────────────────────────────────────

#[test]
fn large_scale_reml_stress_main() {
    let (x_train, y_train, _y_true_train) = simulate(N_TRAIN, PC_DIM, SEED_BASE);
    let (x_holdout, _y_holdout, y_true_holdout) =
        simulate(N_HOLDOUT, PC_DIM, SEED_BASE.wrapping_add(0xDEAD));

    let spec = duchon_aniso_pc_spec("duchon_pc_main", PC_DIM, K_CENTERS);
    let weights = Array1::ones(N_TRAIN);
    let offset = Array1::<f64>::zeros(N_TRAIN);

    // gam#2735 — THE ENTRY POINT IS PART OF WHAT THIS FIXTURE CLAIMS TO TEST.
    //
    // This used to call `fit_term_collection_forspec`, the FIXED-GEOMETRY entry:
    // it builds the design once and optimizes λ. Neither the global length scale
    // (pinned at `HYBRID_LENGTH_SCALE`) nor the per-axis anisotropy ever moved,
    // so a header promising "the full Duchon-on-PC GAM pipeline end-to-end" was
    // describing a pipeline with its geometry nailed down — and the held-out
    // reconstruction it then scored was the best a smoother can do at ONE
    // arbitrary length scale with a response-blind metric.
    //
    // `fit_term_collectionwith_spatial_length_scale_optimization` is the entry
    // the production path uses (`StandardFitRequest` → `fit_standard_base`), so
    // it is the one whose reconstruction the bar is about.
    //
    // `pilot_subsample_threshold: 0` disables the large-n pilot geometry
    // initializer deliberately: the pilot exists to seed κ/η cheaply from a
    // subsample, and this fixture is scoring what the FULL-data outer solve
    // learns. Leaving the pilot on would make the measurement partly a
    // measurement of the subsample.
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let start = Instant::now();
    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        x_train.view(),
        y_train.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        gaussian_identity_likelihood(),
        &fit_options(MAIN_MAX_ITER),
        &kappa_options,
    )
    .expect("large-scale Duchon-on-PC fit should succeed");
    let elapsed = start.elapsed();

    // (1) Fit existence is the sealed convergence proof (SPEC 20).
    assert!(
        fitted.fit.beta.iter().all(|v| v.is_finite()),
        "fitted coefficients must all be finite",
    );

    // (2) Held-out-grid reconstruction error: build the held-out design
    //     using the *fitted* term collection design (so centers, scaling,
    //     etc. match), then compute relative L2 against truth.
    //     The optimizing entry already returns the frozen trained spec, which
    //     carries the LEARNED length scale and per-axis η — refreezing the
    //     caller's spec against the design would silently score the seed.
    let frozenspec = fitted.resolvedspec.clone();
    let holdout_design = build_term_collection_design(x_holdout.view(), &frozenspec)
        .expect("holdout design build must succeed");
    let holdout_dense = holdout_design.design.to_dense();
    let holdout_offset = Array1::<f64>::zeros(N_HOLDOUT);

    let pred_mean = gaussian_identity_mean(
        holdout_dense.view(),
        fitted.fit.beta.view(),
        holdout_offset.view(),
    );
    assert!(pred_mean.iter().all(|v| v.is_finite()));
    let rel_l2 = relative_l2(&pred_mean, &y_true_holdout);

    // A BAR-FREE reference, computed on the same held-out rows.
    //
    // `relative_l2` normalises by the truth's standard deviation, so this
    // metric is `sqrt(1 - R²)` against the NOISELESS truth and the `0.10` bar
    // below is really "R² > 0.99". That bar has no measured provenance in this
    // fixture — it was written in the same commit that created the test
    // (`6a4b4728c`), and per #2709 the fixture could not build until
    // `da1228e1c`, so the assertion had never once been evaluated. A number
    // nobody has ever seen a run produce is an intention, not a threshold.
    //
    // So report a reference that needs no calibration: the best predictor that
    // uses NO curvature at all, least-squares on the raw held-out coordinates
    // plus an intercept. Any smoother spending `K_CENTERS` basis functions must
    // beat it by a wide margin to have earned them, and unlike `0.10` this
    // number is re-derived from the data on every run and cannot drift or age.
    let linear_only_rel_l2 = linear_only_reference(x_holdout.view(), &y_true_holdout);

    // gam#2735 — THE FITTED ANISOTROPIC LOG-SCALES, read off the TRAINED spec.
    //
    // The spec seeds `aniso_log_scales: Some(vec![0.0; PC_DIM])` — isotropic —
    // and the fit is supposed to learn per-axis scales from there. Whether it
    // does is the single discriminator between the two explanations of a missed
    // reconstruction bar on this design, because the achievable error differs by
    // ~3.7x between them (see `RECONSTRUCTION_REL_L2_MAX`). `frozenspec` is the
    // trained spec — the same object the CLI's anisotropic report reads — so
    // this costs nothing and needs no new plumbing.
    //
    // All-zero means the scales never moved off their seed and the smooth is
    // still isotropic in six dimensions whatever the spec asked for.
    let aniso_report = match gam::smooth::get_spatial_aniso_log_scales(&frozenspec, 0) {
        Some(eta) if !eta.is_empty() => {
            let moved = eta.iter().any(|v| v.abs() > 1e-8);
            let parts: Vec<String> = eta.iter().map(|v| format!("{v:+.4}")).collect();
            format!(
                "aniso_log_scales=[{}] moved_off_seed={moved}",
                parts.join(",")
            )
        }
        _ => "aniso_log_scales=<absent>".to_string(),
    };

    // Emitted BEFORE any quality assertion, and this ordering is the point.
    // The convergence check and the summary line used to sit *after* the bar,
    // so a fit that missed the bar aborted before reporting whether it had
    // converged, how long it took, or how it compared to anything — the
    // diagnostic that explains the failure was downstream of the assertion that
    // fires. The first real run of this fixture produced `0.2944` and not one
    // other number.
    eprintln!(
        "[large_scale_reml_stress_main] n={N_TRAIN}, K={K_CENTERS}, pc_dim={PC_DIM} \
         | wall_clock={:.2}s, outer_iter={} (cap {MAIN_MAX_ITER}), \
         rel_l2_holdout={rel_l2:.4} (bar {RECONSTRUCTION_REL_L2_MAX:.2}), \
         linear_only_rel_l2={linear_only_rel_l2:.4}, \
         nonlinear_variance_captured={:.1}%, {aniso_report}",
        elapsed.as_secs_f64(),
        fitted.fit.outer_iterations,
        // What fraction of the structure a straight line CANNOT express did
        // this fit actually recover? `1 - (rel_l2 / linear_only)²` in variance
        // terms; 0% means the smoother matched a straight line, 100% means it
        // recovered everything the line missed.
        100.0 * (1.0 - (rel_l2 * rel_l2) / (linear_only_rel_l2 * linear_only_rel_l2).max(1e-30)),
    );

    assert!(
        rel_l2 < RECONSTRUCTION_REL_L2_MAX,
        "held-out relative L2 reconstruction error too high: {rel_l2:.4} \
         (>= {RECONSTRUCTION_REL_L2_MAX:.2}). Linear-only reference on the same rows: \
         {linear_only_rel_l2:.4}. The bar is NOT the suspect here — see \
         `RECONSTRUCTION_REL_L2_MAX` for the measured references that establish \
         {ANISO_REFERENCE_REL_L2:.4} as achievable on this exact design, and see the \
         `aniso_log_scales=` field of the diagnostic line above: if those are all zero \
         the smooth never left its isotropic seed, which is the only configuration in \
         which this bar is out of reach.",
    );

    // (3) Bias-corrected predictions: FitInference must carry a finite
    //     bias-correction vector after a successful REML fit, and the
    //     bias-corrected Gaussian identity prediction must stay finite.
    let inference = fitted
        .fit
        .inference
        .as_ref()
        .expect("compute_inference=true must populate FitInference");
    let bc = inference
        .bias_correction_beta
        .as_ref()
        .expect("FitInference must carry bias_correction_beta");
    assert_eq!(bc.len(), fitted.fit.beta.len());
    assert!(
        bc.iter().all(|v| v.is_finite()),
        "bias_correction_beta must be entirely finite",
    );

    let pred_unc_mean = gaussian_identity_bias_corrected_mean(
        holdout_dense.view(),
        &fitted.fit,
        holdout_offset.view(),
    );
    assert!(pred_unc_mean.iter().all(|v| v.is_finite()));

    // (4) The outer loop converged inside its configured budget rather than
    //     stopping because it ran out of iterations.
    assert!(
        fitted.fit.outer_iterations < MAIN_MAX_ITER,
        "main large-scale stress fit ran {} outer iterations, exhausting its \
         configured {MAIN_MAX_ITER}-iteration REML budget (elapsed {:.1}s): the \
         outer loop is grinding to its cap instead of converging",
        fitted.fit.outer_iterations,
        elapsed.as_secs_f64(),
    );
}

/// #2708: report WHICH of the three candidate causes the coverage number is
/// consistent with. Report-only; it asserts nothing.
///
/// The aggregate SD of the standardised error `z = (truth − mean)/SE` splits two
/// of them — `SD ≈ 1` with an off-centre mean says bias carries it, `SD ≈ 2`
/// centred says the variance is understated — but it CANNOT separate a uniform
/// scale error from a missing variance component, because both inflate the same
/// aggregate. The binned columns do: a missing smoothing-parameter-uncertainty
/// term is heteroscedastic by construction, so it lands on the high-leverage /
/// boundary rows and leaves the interior near nominal, while a scale error is
/// flat in leverage.
///
/// `SD(z)` binned by SE quantile is the leverage axis (SE² = xᵀΣx is a monotone
/// leverage proxy); `SD(z)` binned by ‖x‖ is the boundary axis; and the mean
/// RESIDUAL binned by `x₀` is the approximation-error axis, because the truth's
/// out-of-model term is `0.4·sin(π·x₀)` and an approximation error is a
/// systematic function of position while sampling error is not.
///
/// Also printed: `|z| > 3` frequency against the 0.27% a standard normal gives.
/// A heavy tail with a near-nominal centre is the outcome the two-way split does
/// not cover, and it would mean an interaction rather than either single cause.
fn report_coverage_diagnostics(z: &[f64], resid: &[f64], se: &[f64], radius: &[f64], x0: &[f64]) {
    if z.is_empty() {
        eprintln!("[cov-diag] no finite points");
        return;
    }
    let n = z.len() as f64;
    let mean_z = z.iter().sum::<f64>() / n;
    let sd_z = (z.iter().map(|v| (v - mean_z).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let tail = z.iter().filter(|v| v.abs() > 3.0).count() as f64 / n;
    let mean_abs_resid = resid.iter().map(|v| v.abs()).sum::<f64>() / n;
    let mean_se = se.iter().sum::<f64>() / n;
    eprintln!(
        "[cov-diag] n={} mean_z={mean_z:+.4} sd_z={sd_z:.4} |z|>3={:.4} (normal 0.0027) \
         mean|resid|={mean_abs_resid:.5} mean_se={mean_se:.5} ratio={:.3}",
        z.len(),
        tail,
        mean_abs_resid / mean_se.max(f64::MIN_POSITIVE),
    );

    // Binned SD(z): flat across bins => a scale error; ramping => a missing,
    // leverage-dependent variance component.
    let binned_sd = |key: &[f64], label: &str| {
        let mut idx: Vec<usize> = (0..key.len()).collect();
        idx.sort_by(|&a, &b| {
            key[a]
                .partial_cmp(&key[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        const BINS: usize = 5;
        let per = idx.len() / BINS;
        if per == 0 {
            return;
        }
        let mut out = String::new();
        for b in 0..BINS {
            let lo = b * per;
            let hi = if b + 1 == BINS {
                idx.len()
            } else {
                (b + 1) * per
            };
            let slice = &idx[lo..hi];
            let m = slice.len() as f64;
            let mu = slice.iter().map(|&i| z[i]).sum::<f64>() / m;
            let sd = (slice.iter().map(|&i| (z[i] - mu).powi(2)).sum::<f64>() / (m - 1.0).max(1.0))
                .sqrt();
            out.push_str(&format!(
                " [{:.3}..{:.3}] sd={sd:.3} mean={mu:+.3};",
                key[slice[0]],
                key[slice[slice.len() - 1]]
            ));
        }
        eprintln!("[cov-diag] SD(z) by {label}:{out}");
    };
    binned_sd(se, "SE quantile (leverage proxy)");
    binned_sd(radius, "‖x‖ (distance from design centre)");

    // Mean RESIDUAL by x0: the truth's out-of-model term is `0.4·sin(π·x₀)`, so a
    // systematic sign pattern here is approximation error, not sampling error.
    {
        let mut idx: Vec<usize> = (0..x0.len()).collect();
        idx.sort_by(|&a, &b| {
            x0[a]
                .partial_cmp(&x0[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        const BINS: usize = 8;
        let per = idx.len() / BINS;
        if per > 0 {
            let mut out = String::new();
            for b in 0..BINS {
                let lo = b * per;
                let hi = if b + 1 == BINS {
                    idx.len()
                } else {
                    (b + 1) * per
                };
                let slice = &idx[lo..hi];
                let mu = slice.iter().map(|&i| resid[i]).sum::<f64>() / slice.len() as f64;
                out.push_str(&format!(" [{:+.2}]={mu:+.4};", x0[slice[slice.len() / 2]]));
            }
            eprintln!("[cov-diag] mean(truth-mean) by x0:{out}");
        }
    }
}

// ─── Coverage simulation ────────────────────────────────────────────────

/// Empirical coverage of the shipped nominal-95% mean interval, measured
/// against a mean function the fitted model can represent (#2708).
///
/// # Why the truth is drawn from the basis and not from `truth()`
///
/// A credible interval is a statement *conditional on the model*. `Vb` and
/// `Vp` describe how far `X·β̂` moves around `X·β` under resampling; neither
/// contains, or claims to contain, the gap between `X·β` and a mean function
/// outside the column span of `X`. Scoring the interval against an out-of-span
/// truth therefore measures basis truncation and reports it as miscalibration,
/// and no correct covariance can pass such a test.
///
/// That is not hypothetical here — it is what this fixture used to do, and it
/// is what #2708 was. Measured at `b49e5a662`, 8 × 400 = 3200 pooled
/// intervals, this exact configuration (`K_COVERAGE = 80`, `PC_DIM_COVERAGE =
/// 4`, `N_COVERAGE_TRAIN = 4000`):
///
/// ```text
///   analytic `truth()`      coverage 0.1769   mean_se 0.0496   mean|resid| 0.2438
///   in-model truth          coverage 0.9463 (Vb) / 0.9678 (Vp)
/// ```
///
/// Same centre count, same `n`, same solver, same covariance code. Only the
/// representability of the target changed, and coverage moved from 0.18 to
/// nominal. The residual under the analytic truth tracks its
/// `0.4·sin(π·x₀)` term in sign at 7 of 8 `x₀` bins at 0.5–0.8 of its
/// amplitude while the linear trend is recovered cleanly: 80 centres in 4-D is
/// a mean spacing of order 1.5 against a wavelength of 2.0, i.e. under
/// Nyquist, so the sinusoid is simply not in the span.
///
/// # Why the centre count is not the lever
///
/// Raising `K` was measured and does not reach nominal. Out-of-model coverage
/// at current main, 4 sims × 400 per arm:
///
/// ```text
///   K =  80   coverage 0.1794   rel_l2 0.3074   edf ≈  64
///   K = 160   coverage 0.3113   rel_l2 0.2600   edf ≈ 134
///   K = 320   coverage 0.6200   rel_l2 0.1896   edf ≈ 263
/// ```
///
/// Truncation error falls slowly while `edf` — and with it the posterior SE —
/// climbs toward `n`. The two do not cross at 0.95 at any centre count this
/// fixture could afford, so "add centres" trades one wrong answer for a
/// slower wrong answer.
///
/// # Why the bar is two-sided
///
/// The old bar was `coverage > 0.85`, which cannot detect an interval that is
/// too WIDE. #2728 is the proof: a sigma-point node placed 3309 nats above the
/// optimum inflated `J·V_ρ·Jᵀ` until `rms(se_corr)/rms(se_cond)` reached 5.45,
/// and the resulting over-wide interval moved this fixture's number from
/// 0.1769 *up* to 0.6447 — i.e. a catastrophic covariance defect pushed the
/// gate TOWARD passing. A one-sided coverage bar rewards inflation, so this
/// one is two-sided and is accompanied by a width guard.
#[test]
fn large_scale_reml_stress_coverage() {
    // ── The in-model mean function ──────────────────────────────────────
    //
    // One pilot fit on the analytic DGP supplies the geometry: freezing its
    // design pins the centre set, the scaling, and every other data-dependent
    // choice, so every replicate below is fit in the SAME column span. Its
    // coefficient vector becomes the truth. Using a fitted `β̂` rather than a
    // synthetic draw keeps the target a realistic smooth function at a
    // realistic amplitude instead of white noise in coefficient space.
    let (x_pilot, y_pilot, _) = simulate(
        N_COVERAGE_TRAIN,
        PC_DIM_COVERAGE,
        SEED_BASE.wrapping_add(0xB1A5_0000),
    );
    let pilot_spec = duchon_aniso_pc_spec("duchon_pc_cov_pilot", PC_DIM_COVERAGE, K_COVERAGE);
    let weights = Array1::ones(N_COVERAGE_TRAIN);
    let offset_tr = Array1::<f64>::zeros(N_COVERAGE_TRAIN);
    let pilot = fit_term_collection_forspec(
        x_pilot.view(),
        y_pilot.view(),
        weights.view(),
        offset_tr.view(),
        &pilot_spec,
        gaussian_identity_likelihood(),
        &fit_options(COVERAGE_MAX_ITER),
    )
    .expect("coverage pilot Duchon-on-PC fit should succeed");
    let frozenspec = freeze_term_collection_from_design(&pilot_spec, &pilot.design)
        .expect("coverage pilot freeze must succeed");
    let beta_truth = pilot.fit.beta.clone();
    assert!(
        beta_truth.iter().all(|v| v.is_finite()),
        "the in-model truth's coefficient vector must be finite",
    );

    let mut in_conditional = 0usize;
    let mut in_corrected = 0usize;
    let mut total_pts = 0usize;
    let mut per_sim_corrected: Vec<f64> = Vec::new();
    let mut sum_sq_se_cond = 0.0_f64;
    let mut sum_sq_se_corr = 0.0_f64;
    // #2708 diagnostics, unchanged in form from the report-only version that
    // localised this defect. They now describe an in-model fit, so `sd_z ≈ 1`
    // and a flat profile across the leverage bins is the expected reading and
    // a ramp is the signature to chase.
    let mut z_all: Vec<f64> = Vec::new();
    let mut resid_all: Vec<f64> = Vec::new();
    let mut se_all: Vec<f64> = Vec::new();
    let mut radius_all: Vec<f64> = Vec::new();
    let mut x0_all: Vec<f64> = Vec::new();

    for sim_idx in 0..N_COVERAGE_SIMS {
        let train_seed = SEED_BASE.wrapping_add(0xC0DE_0000 + sim_idx as u64);
        let test_seed = SEED_BASE.wrapping_add(0xFADE_0000 + sim_idx as u64);
        let (x_tr, _, _) = simulate(N_COVERAGE_TRAIN, PC_DIM_COVERAGE, train_seed);
        let (x_te, _, _) = simulate(N_COVERAGE_HOLDOUT, PC_DIM_COVERAGE, test_seed);

        // Both designs are built from the SAME frozen spec, so the truth and
        // the fit live in one column span by construction rather than by
        // assumption.
        let train_design = build_term_collection_design(x_tr.view(), &frozenspec)
            .expect("coverage-sim train design build must succeed");
        let holdout_design = build_term_collection_design(x_te.view(), &frozenspec)
            .expect("coverage-sim holdout design build must succeed");
        let train_dense = train_design.design.to_dense();
        let holdout_dense = holdout_design.design.to_dense();
        let truth_te = holdout_dense.dot(&beta_truth);

        let mut y_tr = train_dense.dot(&beta_truth);
        let mut rng = StdRng::seed_from_u64(train_seed ^ 0x5EED_0F17);
        let noise = Normal::new(0.0, NOISE_SD).expect("noise params must be valid");
        for value in y_tr.iter_mut() {
            *value += noise.sample(&mut rng);
        }

        let start = Instant::now();
        let fitted = fit_term_collection_forspec(
            x_tr.view(),
            y_tr.view(),
            weights.view(),
            offset_tr.view(),
            &frozenspec,
            gaussian_identity_likelihood(),
            &fit_options(COVERAGE_MAX_ITER),
        )
        .expect("coverage-sim Duchon-on-PC fit should succeed");
        let elapsed = start.elapsed();
        assert!(
            fitted.fit.outer_iterations < COVERAGE_MAX_ITER,
            "coverage-sim fit {sim_idx} ran {} outer iterations, exhausting its \
             configured {COVERAGE_MAX_ITER}-iteration REML budget (elapsed \
             {:.1}s): the outer loop is grinding to its cap instead of converging",
            fitted.fit.outer_iterations,
            elapsed.as_secs_f64(),
        );
        // Fit existence is the sealed convergence proof (SPEC 20).

        let covariance_conditional = fitted
            .fit
            .beta_covariance()
            .expect("Gaussian identity coverage requires the conditional covariance")
            .clone();
        let offset_te = Array1::<f64>::zeros(N_COVERAGE_HOLDOUT);
        let (pred_mean, pred_lower, pred_upper) = gaussian_identity_bias_corrected_mean_interval(
            holdout_dense.view(),
            &fitted.fit,
            offset_te.view(),
        );
        assert!(pred_mean.iter().all(|v| v.is_finite()));

        let mut sim_in_corrected = 0usize;
        for i in 0..N_COVERAGE_HOLDOUT {
            let truth_i = truth_te[i];
            // The corrected interval is what `predict()` ships, and it is what
            // the half-width below reports.
            let se_corr = (pred_upper[i] - pred_lower[i]) / (2.0 * NORMAL_95_TWO_SIDED_Z);
            let row = holdout_dense.row(i);
            let se_cond = row.dot(&covariance_conditional.dot(&row)).max(0.0).sqrt();
            let resid = truth_i - pred_mean[i];

            if truth_i >= pred_lower[i] && truth_i <= pred_upper[i] {
                in_corrected += 1;
                sim_in_corrected += 1;
            }
            if resid.abs() <= NORMAL_95_TWO_SIDED_Z * se_cond {
                in_conditional += 1;
            }
            total_pts += 1;
            sum_sq_se_cond += se_cond * se_cond;
            sum_sq_se_corr += se_corr * se_corr;

            if se_corr > 0.0 && resid.is_finite() {
                z_all.push(resid / se_corr);
                resid_all.push(resid);
                se_all.push(se_corr);
                let raw_row = x_te.row(i);
                radius_all.push(raw_row.dot(&raw_row).sqrt());
                x0_all.push(raw_row[0]);
            }
        }
        per_sim_corrected.push(sim_in_corrected as f64 / N_COVERAGE_HOLDOUT as f64);
    }

    report_coverage_diagnostics(&z_all, &resid_all, &se_all, &radius_all, &x0_all);

    let points = total_pts.max(1) as f64;
    let coverage_conditional = in_conditional as f64 / points;
    let coverage_corrected = in_corrected as f64 / points;
    let rms_se_cond = (sum_sq_se_cond / points).sqrt();
    let rms_se_corr = (sum_sq_se_corr / points).sqrt();
    let width_ratio = rms_se_corr / rms_se_cond.max(f64::MIN_POSITIVE);

    // The two standard errors this number admits, both computed rather than
    // written down.
    //
    // The binomial one treats every interval as independent. They are not: the
    // `N_COVERAGE_HOLDOUT` points inside one replicate share a single `β̂` and
    // a single `ρ̂`, so it is optimistic by construction. The between-replicate
    // standard error of the per-replicate coverage has the replicate as its
    // unit and carries that dependence honestly. Gating on the LARGER of the
    // two never lets the optimistic one manufacture a failure.
    let binomial_se = (NOMINAL_COVERAGE * (1.0 - NOMINAL_COVERAGE) / points).sqrt();
    let replicates = per_sim_corrected.len() as f64;
    let mean_per_sim = per_sim_corrected.iter().sum::<f64>() / replicates.max(1.0);
    let between_sim_se = if replicates > 1.0 {
        (per_sim_corrected
            .iter()
            .map(|c| (c - mean_per_sim).powi(2))
            .sum::<f64>()
            / (replicates - 1.0)
            / replicates)
            .sqrt()
    } else {
        f64::INFINITY
    };
    let coverage_se = binomial_se.max(between_sim_se);

    eprintln!(
        "[large_scale_reml_stress_coverage] sims={N_COVERAGE_SIMS}, points={total_pts}, \
         in-model truth | coverage_conditional={coverage_conditional:.4} \
         coverage_corrected={coverage_corrected:.4} | binomial_se={binomial_se:.5} \
         between_sim_se={between_sim_se:.5} gate_se={coverage_se:.5} | \
         rms_se_cond={rms_se_cond:.5} rms_se_corr={rms_se_corr:.5} \
         width_ratio={width_ratio:.4} | per_sim={per_sim_corrected:?}",
    );

    // (1) CALIBRATION, two-sided. The conditional covariance is the object
    //     whose pointwise coverage is a nominal-95% claim once the target is
    //     in the span, so it is the one the equality is asserted on.
    let deviation = (coverage_conditional - NOMINAL_COVERAGE).abs();
    assert!(
        deviation <= COVERAGE_BAND_SIGMAS * coverage_se,
        "empirical coverage of the nominal-{NOMINAL_COVERAGE:.2} conditional mean \
         interval is {coverage_conditional:.4} ({in_conditional}/{total_pts}) against an \
         in-model truth: |{deviation:.4}| exceeds {COVERAGE_BAND_SIGMAS} × \
         {coverage_se:.5}. The truth here IS in the fitted span, so this is a \
         statement about the covariance path and nothing else.",
    );

    // (2) The smoothing correction must WIDEN the interval. `Vp = Vb +
    //     Cov_ρ[β̂]` is a sum of PSD terms, so a corrected interval that covers
    //     less than the conditional one is a sign error or a lost term, not
    //     conservatism.
    assert!(
        coverage_corrected >= coverage_conditional,
        "the smoothing-corrected interval covers {coverage_corrected:.4}, LESS than the \
         conditional {coverage_conditional:.4}. `Vp = Vb + Cov_ρ[β̂]` adds a PSD term, so \
         it cannot be narrower than `Vb`.",
    );

    // (3) …and must not exceed the thing it corrects. `Vp = Vb + Cov_ρ[β̂]`, so
    //     `rms(se_corr) ≤ √2 · rms(se_cond)` is exactly the statement that the
    //     smoothing-parameter term contributes no more VARIANCE than the
    //     conditional posterior it is added to. That is the dividing line at
    //     which the correction stops correcting an estimate and starts being
    //     the estimate — it is derived from which term dominates, not tuned.
    //     Measured at `b49e5a662`: 1.19 here, and 1.13–1.15 across `K ∈
    //     {80,160,320}` on the analytic DGP. #2728 shipped 5.45.
    let width_ratio_ceiling = 2.0_f64.sqrt();
    assert!(
        width_ratio <= width_ratio_ceiling,
        "the smoothing-parameter correction contributes more variance than the \
         conditional posterior it corrects: rms(se_corr)/rms(se_cond) = \
         {width_ratio:.4} > √2 = {width_ratio_ceiling:.4} (rms_se_cond={rms_se_cond:.5}, \
         rms_se_corr={rms_se_corr:.5}). This is the shape #2728 had, where the ratio \
         reached 5.45 and the resulting over-wide interval moved a ONE-SIDED coverage \
         gate toward passing.",
    );
}
