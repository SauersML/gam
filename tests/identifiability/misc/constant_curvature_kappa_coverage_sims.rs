//! #944 stage-4 validation sims — the deferred quantitative half of "curvature
//! as an estimand": across REPLICATES of data generated on a known `M_κ`, the
//! profile-likelihood machinery must (1) RECOVER the planted curvature with low
//! bias, (2) COVER the true κ⋆ with its 95% profile CI at ≈ the nominal rate,
//! and (3) hold SIZE on the interior κ=0 flatness test (flat data is not
//! spuriously rejected) while having POWER (curved data is rejected). The
//! single-dataset e2e test (`constant_curvature_kappa_inference_e2e`) asserts
//! sign-recovery and flatness DIRECTION; this test adds the replicate-level
//! calibration the issue charter names ("recovery of κ̂, CI coverage, size of
//! the κ=0 test") — the claims that make "κ̂ = … (95% CI …)" a statistically
//! honest sentence rather than a point estimate.
//!
//! Reference-as-truth: every dataset is generated on a known `ConstantCurvature`
//! geometry and every assertion is against that self-constructed truth or the
//! exact χ² calibration of gam's own profiled REML criterion — never another
//! tool's output. Bars are sized to the small replicate count `R` so they catch
//! a genuinely miscalibrated estimator/CI/test without flaking on binomial noise
//! (kept CI-cheap: small n, few centers, a handful of replicates).

use gam::estimate::FitOptions;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::SmoothBasisSpec;
use gam::smooth::{
    CurvatureInference, SpatialLengthScaleOptimizationOptions, curvature_inference_forspec,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use gam::terms::basis::build_constant_curvature_basis;
use gam::terms::term_builder::build_termspec;
use gam::types::LikelihoodSpec;
use ndarray::{Array1, Array2};

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps ------

/// The one formula the generator and the fit must share: the plant is built
/// from THIS spec's realized centers and length scale, so a change here that did
/// not reach the generator would silently re-misspecify the fixture.
const FORMULA: &str = "y ~ curv(x1, x2, centers=6)";

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Build a `TermCollectionSpec` for a `curv(...)` formula. Mirrors the e2e
/// inference test's builder: a 3-column `[y, x1, x2]` continuous schema so the
/// `curv(x1, x2)` term resolves and is not rejected as a constant-column smooth.
fn termspec_for(formula: &str, frame: &Array2<f64>) -> gam::smooth::TermCollectionSpec {
    let parsed = parse_formula(formula).expect("formula parses");
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let ds = EncodedDataset {
        headers: headers.clone(),
        values: frame.clone(),
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.clone(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; 3],
    };
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam::ResourcePolicy::default_library(),
    )
    .expect("term spec")
}

/// `n` chart points uniformly in a disk of radius `radius`, with a Gaussian
/// response drawn from the `M_{κ⋆}` model the fit will estimate.
///
/// ## The plant has to be INSIDE the κ⋆ model (gam#2687)
///
/// This generator used to plant `μ = 2·exp(−d_{κ⋆}(x,0)) − 1`: a kernel section
/// at unit length about the chart origin. That function is in **no** realized
/// span — the basis is built on a handful of centers at the smooth's own range,
/// not at length 1 — so at every κ, including κ⋆, the fit is
/// approximating rather than estimating. The κ-spans then happen to be ORDERED
/// in approximation power, and the profiled criterion faithfully follows that
/// order instead of the curvature: `V_p` descends monotonically across the whole
/// admissible interval and κ̂ rails at the box endpoint, measured at 9/9
/// replicates and confirmed unchanged at **zero noise** and across plant scales
/// {0.5, 1.0, 1.19, 2.0}. It is misspecification, not an estimator defect.
///
/// The control that separates the two is on the #2687 thread: planting `y` as an
/// exact linear combination of the **κ⋆ basis's own columns** — so the truth is
/// in the κ⋆-span by construction and in no other — recovers κ⋆ to one grid step
/// at κ⋆ ∈ {−1.0, −0.5, +0.5, +1.0}, interior on both branches. A coverage
/// fixture measures the coverage of a parameter the model HAS, so it must draw
/// from that model; this generator now does.
///
/// The coefficient profile `w_j = 1/(1+j)` is deterministic and decaying, so the
/// planted function is a genuinely smooth member of the span rather than a single
/// kernel section, and the signal is standardized to unit SD before the noise is
/// applied so `noise_sd` keeps meaning a signal-to-noise ratio across κ⋆.
///
/// ## The planted RANGE has to move too (gam#2747)
///
/// Planting inside the κ⋆ span is necessary and was not sufficient. The truth
/// was built from the fit's own spec, so it also inherited the fit's own AUTO
/// length scale — and that is precisely the one configuration in which a
/// range-blind criterion works. Measured on a 3 curvatures × 3 ranges grid: with
/// `ℓ` pinned at the auto `ℓ_ref`, the criterion recovers κ⋆ in the `1×` column
/// and nowhere else — railed at a box endpoint at `0.5×` and `2×`, sign-inverted
/// on the spherical arm, and reporting a confident interior `κ̂ = ∓0.94` on
/// genuinely flat data. A fixture that plants only at `1×` cannot see any of it.
///
/// So `range_multiplier` cycles `{0.5, 1, 2}` across the replicates. The
/// binomial bar below is unchanged: the replicates are still `R` independent
/// datasets, they now just draw from a mixture over the range as well as over
/// the noise, which makes the same coverage claim strictly stronger.
///
/// ## Flat truth (`κ⋆ = 0`) is a real signal again
///
/// This generator used to plant a CONSTANT mean for flat truth, because *"any
/// centre-peaked radial signal is fitted better by a renormalized kernel at
/// κ ≠ 0 even at matched effective degrees of freedom"* — measured as `κ̂` railed
/// at the box's upper end in 6 of 9 genuinely flat replicates. That lean was the
/// range confounding (gam#2747): a flat signal at a range the criterion could
/// not move was fitted better by buying resolution with κ. With the range
/// estimated the lean is gone, and a constant mean is no longer an honest size
/// test — it asks whether the flatness test survives a signal with no shape at
/// all, which is the easy case. Flat truth is now a genuine member of the κ = 0
/// span at the replicate's own range, exactly like the curved arms.
fn dataset_on_m_kappa(
    n: usize,
    kappa_star: f64,
    radius: f64,
    noise_sd: f64,
    range_multiplier: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut st = seed;
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut noise = Array1::<f64>::zeros(n);
    // Draw the row's chart point and its noise draw together, in that order, so
    // the RNG stream is byte-identical to the pre-#2687 generator's. The plant
    // itself needs every feature row before it can be built (the centers and the
    // realized length scale are functions of the whole cloud), but that is a
    // reordering of the ARITHMETIC, not of the stream, and a fixture whose data
    // silently moved would make every before/after on this thread unreadable.
    for i in 0..n {
        let (x1, x2) = loop {
            let a = 2.0 * next_unit(&mut st) - 1.0;
            let b = 2.0 * next_unit(&mut st) - 1.0;
            if a * a + b * b <= 1.0 {
                break (a * radius, b * radius);
            }
        };
        feats[(i, 0)] = x1;
        feats[(i, 1)] = x2;
        noise[i] = next_gauss(&mut st);
    }
    let mut y = Array1::<f64>::zeros(n);
    // A member of the κ⋆ span at the replicate's own range, built from the SAME
    // spec the fit will use — same center strategy, same auto rule for the
    // reference range — so the truth is in the model being estimated and in no
    // other member of the family. `range_multiplier` moves it OFF the auto
    // range, which is the configuration a range-blind criterion cannot handle.
    {
        let mut frame = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            frame[(i, 1)] = feats[(i, 0)];
            frame[(i, 2)] = feats[(i, 1)];
        }
        let fitspec = termspec_for(FORMULA, &frame);
        let SmoothBasisSpec::ConstantCurvature { spec: cc, .. } = &fitspec.smooth_terms[0].basis
        else {
            panic!("the fixture formula must resolve to a constant-curvature term");
        };
        let mut truth_spec = cc.clone();
        truth_spec.kappa = kappa_star;
        truth_spec.kappa_fixed = true;
        truth_spec.double_penalty = false;
        let centers = gam::basis::constant_curvature_realized_centers(feats.view(), &truth_spec)
            .expect("the fixture cloud yields a realized center set");
        let ell_ref = gam::basis::realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("the realized centers span a positive pairwise distance");
        truth_spec.length_scale = ell_ref * range_multiplier;
        truth_spec.length_scale_fixed = true;
        let basis = build_constant_curvature_basis(feats.view(), &truth_spec)
            .expect("the planted κ⋆ geometry must be inside its own chart");
        let design = basis.design.to_dense();
        for j in 0..design.ncols() {
            let w = 1.0 / (1.0 + j as f64);
            for i in 0..n {
                y[i] += w * design[(i, j)];
            }
        }
        // Standardize so `noise_sd` is a signal-to-noise ratio, not an absolute
        // scale that would drift with κ⋆ and the realized center count.
        let mean = y.iter().sum::<f64>() / n as f64;
        let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        assert!(
            sd > 0.0,
            "the planted κ⋆ = {kappa_star} signal collapsed to a constant"
        );
        for i in 0..n {
            y[i] = (y[i] - mean) / sd;
        }
    }
    for i in 0..n {
        y[i] += noise_sd * noise[i];
    }
    (feats, y)
}

/// Fit `curv(x1, x2)` with κ optimized as an outer ψ-coordinate, then run the
/// full curvature inference (κ̂ + profile CI + κ=0 LR test) off the REAL
/// profiled REML criterion. CI-cheap: small `centers`, capped outer iters.
fn fit_and_infer(feats: &Array2<f64>, y: &Array1<f64>) -> CurvatureInference {
    let n = y.len();
    let mut frame = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        frame[(i, 0)] = y[i];
        frame[(i, 1)] = feats[(i, 0)];
        frame[(i, 2)] = feats[(i, 1)];
    }
    let spec = termspec_for(FORMULA, &frame);

    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let options = FitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 8,
        rel_tol: 1e-4,
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };

    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        frame.view(),
        y.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &options,
        &kappa_options,
    )
    .expect("constant-curvature fit with κ optimization");

    curvature_inference_forspec(
        frame.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &fitted.resolvedspec,
        0,
        LikelihoodSpec::gaussian_identity(),
        &options,
        0.95,
    )
    .expect("curvature inference")
}

/// Number of replicate datasets per arm, and the miss bar that goes with it.
///
/// The count is DERIVED, not chosen (gam#2687). A "at most one miss out of `n`"
/// bar is unfalsifiable at `n = 3`: exact binomial arithmetic gives
/// `P(pass | true coverage 0.50) = 0.5000` and `P(pass | 0.83) = 0.9231`, so an
/// estimator with half the nominal coverage passes as often as it fails, and
/// tightening the bar does not help — at `n = 3` with ZERO misses allowed a
/// 0.83-coverage estimator still passes 0.5718 of the time while a correct one
/// already fails 0.1426 of the time. It is the COUNT, not the bar, that makes
/// the claim unresolvable.
///
/// Taking the bar as the smallest `k` holding false alarm ≤ 1% at true coverage
/// 0.95:
///
/// | n | bar `k` | P(pass \| 0.95) | power vs 0.50 | power vs 0.83 |
/// |---|---|---|---|---|
/// | 3 | 1 | 0.9928 | 0.5000 | 0.0769 |
/// | **9** | **2** | 0.9916 | **0.9102** | 0.1861 |
/// | 25 | 4 | 0.9928 | 0.9995 | 0.4241 |
/// | 60 | 7 | 0.9902 | 1.0000 | 0.8222 |
///
/// `n = 9, k = 2` is the smallest count that catches a GROSSLY broken estimator
/// (coverage ≈ 0.5) at ≥ 90% power while false-alarming under 1% on a correct
/// one. Catching a mildly miscalibrated one (0.83) needs `n ≈ 60`, which is a
/// cluster-scale sweep and is deliberately out of scope here — so this gate's
/// stated job is "grossly broken", and the table says so rather than leaving the
/// reader to assume more.
///
/// The bar must move with the count: at `n = 50`, "at most one miss" would
/// reject a perfectly calibrated estimator 72% of the time. Any change to
/// `REPLICATE_COUNT` has to re-derive `MAX_MISSES` from the same binomial.
const REPLICATE_COUNT: usize = 9;

/// Maximum number of missed replicates the coverage/size bars tolerate. Derived
/// with [`REPLICATE_COUNT`]; see its table.
const MAX_MISSES: usize = 2;

/// The planted RANGE multipliers, cycled across replicates (gam#2747). `1.0` is
/// the fit's own auto rule — the single configuration a range-blind criterion
/// handles — so a fixture that used only it could not see the defect. The
/// factor-of-two span on each side is where the pre-#2747 criterion railed,
/// inverted the sign, or invented curvature from flat data.
const RANGE_MULTIPLIERS: [f64; 3] = [0.5, 1.0, 2.0];

fn replicate_count() -> usize {
    REPLICATE_COUNT
}

/// How one replicate's profile CI resolved against a target κ.
///
/// The middle arm is the one this file was missing (gam#2687), and the asymmetry
/// between the two directions is the point. `KappaProfileCi` carries
/// `lo_at_bound` / `hi_at_bound` — *"CI is left/right-open at the bound"* — and
/// `kappa_hat_support`, which says whether κ̂ is itself a box endpoint. Both
/// defects make the REPORTED set a truncation of the set the data actually
/// support:
///
/// * a bound-open endpoint is where the walk ran out of box, not where the
///   profile crossed χ²₁, so the true set extends past it;
/// * a railed κ̂ anchors the Wilks drop at a boundary value of `V_p` rather than
///   at its minimum, so `2[V_p(κ) − V_p(κ̂)]` understates the drop everywhere.
///
/// Truncation only ever removes κ from the reported interval. So **containment
/// survives it and exclusion does not**: a target INSIDE the reported interval is
/// inside the true one too and is genuinely covered, while a target outside is
/// only excluded if that endpoint is a real χ²₁ crossing AND κ̂ is a genuine
/// interior optimum. Everything else is UNRESOLVED — neither coverage nor a
/// miss — and a gate that folds it into either number is measuring something
/// other than coverage. Before #2687 this file tested closed containment and
/// reported `covers=false` for a right-open interval, which reads as a
/// mis-covering CI when the true state was "no exclusion was ever claimed".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resolution {
    Covered,
    Missed,
    Unresolved,
}

/// The SPHERICAL arm of [`curved_coverage_arm`]. Seed base unchanged, so every
/// replicate is byte-identical to the pre-parameterisation fixture's and the
/// before/after on this thread stays readable.
#[test]
fn profile_ci_covers_planted_curvature_across_replicates() {
    curved_coverage_arm(1.0, 0x5EED_0944_0000_0000);
}

/// The HYPERBOLIC arm of [`curved_coverage_arm`] — the half of gam#2747's
/// “on both curvature signs” that no fixture in the tree measured at a range
/// the auto heuristic does not already supply.
///
/// A distinct seed base, so the two arms are independent datasets rather than
/// one chart cloud carrying two responses.
#[test]
fn profile_ci_covers_planted_hyperbolic_curvature_across_replicates() {
    curved_coverage_arm(-1.0, 0x5EED_0944_2747_0000);
}

