//! #944 end-to-end: curvature as an estimand — κ̂ + profile CI + κ=0 LR test
//! from a real constant-curvature (`M_κ`) fit.
//!
//! This is the inferential payoff of the whole #944 program: not "we chose
//! hyperbolic space" but "κ̂ = … (95% CI …), flat rejected at p = …". Data are
//! GENERATED on a known `ConstantCurvature` geometry (self-constructed truth —
//! never another tool's output); a `curv(x1, x2)` smooth is fitted with κ as an
//! outer ψ-coordinate; and `curvature_inference_forspec` reports κ̂, its
//! profile-likelihood CI, and the interior-point flatness test built on the
//! REAL profiled REML criterion `V_p(κ)`.
//!
//! Three arms, one fit each (bounded for CI cost):
//!   * spherical truth  (κ⋆ = +2)  ⇒ κ̂ > 0, flatness rejected, verdict ≠ Hyperbolic
//!   * flat truth       (κ⋆ =  0)  ⇒ flatness NOT rejected, CI straddles 0 (verdict Flat)
//!   * hyperbolic truth (κ⋆ = −2)  ⇒ κ̂ < 0, flatness rejected, verdict ≠ Spherical
//!
//! The assertions are truth-recovery + correct-size, not tight coverage (which
//! needs many replicates); sign-recovery and the flatness direction are the
//! issue's headline claims and are the stable single-dataset statements.

use gam::estimate::FitOptions;
use gam::geometry::curvature_estimand::CurvatureVerdict;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::{
    CurvatureInference, SpatialLengthScaleOptimizationOptions, curvature_inference_forspec,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use gam::terms::term_builder::build_termspec;
use gam::types::LikelihoodSpec;
use ndarray::{Array1, Array2};

/// Number of chart points per fitted dataset. The default is CI-affordable; the
/// signal is identified by the chart GEOMETRY (radius, noise_sd), not the row
/// count, so the strong sign-recovery / flatness asserts below hold at the
/// smaller default. A cluster-scale n=2000 sweep is a separate MSI artifact.
fn n_obs() -> usize {
    // Fixed CI-affordable size. The curvature signal is identified by the chart
    // GEOMETRY (radius, noise_sd), not the row count, so the sign-recovery /
    // flatness asserts hold at this n. A cluster-scale n=2000 sweep is a separate
    // MSI artifact, not an env/cfg branch inside the test.
    600
}

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps ------

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Build a `TermCollectionSpec` for a `curv(...)` formula (FarthestPoint
/// centers, auto length scale, κ seeded at 0 so the optimizer has to move it).
///
/// `frame` is the real `[y, x1, x2]` training matrix: the spec builder now
/// guards against degenerate (single-unique-value) smooth columns, so the
/// schema dataset must carry the actual feature values rather than a constant
/// placeholder, or the `curv(x1, x2)` term is rejected as a constant-column
/// smooth before it can ever be fitted.
/// The ONE formula the generator and the fit share: the curved plant is built
/// from THIS spec's realized centers and range, so a change here that did not
/// reach the generator would silently re-misspecify the fixture.
const FIT_FORMULA: &str = "y ~ curv(x1, x2, centers=10)";

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
    )
    .expect("term spec")
}

/// `n` chart points uniformly in a disk of radius `radius`, with a Gaussian
/// response built so that the curvature is genuinely IDENTIFIABLE (#944):
///
/// * **Curved truth (κ⋆ ≠ 0):** a member of the κ⋆ span, built from the same
///   center rule and the same auto range the fit will use, so the truth is in
///   the model being estimated and in no other member of the family. The
///   coefficient profile `w_j = 1/(1+j)` is deterministic and decaying, so the
///   planted function is a genuinely smooth member of the span rather than a
///   single kernel section, and the signal is standardized to unit SD before
///   the noise is applied so `noise_sd` keeps meaning a signal-to-noise ratio.
///
///   This generator used to plant `μ = 2·exp(−d_{κ⋆}(x,0)) − 1`, a kernel
///   section at unit length about the chart ORIGIN, and that plant is
///   **curvature-BLIND as a function class**: `d_κ(x,0) = 2·arctan(√κ r)/√κ` is
///   a strictly monotone reparametrization of the chart radius for EVERY κ, so
///   a function of the origin distance is a function of the origin distance at
///   every curvature and the geometry carries no signal at all. κ is then
///   identified only by which radial profiles the center set happens to be able
///   to make, which is a knife edge — measured (gam#2747): on that plant at
///   κ⋆ = 0 the criterion rails with an LR of 34 against a χ²₁ threshold of
///   3.84, and at κ⋆ < 0 it recovers the wrong sign. The sibling
///   `constant_curvature_kappa_coverage_sims` moved inside the span for the
///   same reason under `#2687`; this file was the last one that had not.
///
/// * **Flat truth (κ⋆ = 0):** the mean is constant (κ-NEUTRAL). A flat space has
///   no preferred geodesic-distance shape, so there is no curvature to plant,
///   and the honest realization of "flat truth" is the absence of curvature
///   structure rather than a curvature signal at κ = 0.
fn dataset_on_m_kappa(
    n: usize,
    kappa_star: f64,
    radius: f64,
    noise_sd: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut st = seed;
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut noise = Array1::<f64>::zeros(n);
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
    if kappa_star != 0.0 {
        let mut frame = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            frame[(i, 1)] = feats[(i, 0)];
            frame[(i, 2)] = feats[(i, 1)];
        }
        let fitspec = termspec_for(FIT_FORMULA, &frame);
        let gam::smooth::SmoothBasisSpec::ConstantCurvature { spec: cc, .. } =
            &fitspec.smooth_terms[0].basis
        else {
            panic!("the fixture formula must resolve to a constant-curvature term");
        };
        let mut truth_spec = cc.clone();
        truth_spec.kappa = kappa_star;
        truth_spec.kappa_fixed = true;
        truth_spec.double_penalty = false;
        let basis = gam::basis::build_constant_curvature_basis(feats.view(), &truth_spec)
            .expect("the planted κ⋆ geometry must be inside its own chart");
        let design = basis.design.to_dense();
        for j in 0..design.ncols() {
            let w = 1.0 / (1.0 + j as f64);
            for i in 0..n {
                y[i] += w * design[(i, j)];
            }
        }
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

/// Fit `curv(x1, x2)` on data built into a 3-column `[y, x1, x2]` frame with κ
/// optimized as an outer ψ-coordinate, then return the full curvature report.
fn fit_and_infer(feats: &Array2<f64>, y: &Array1<f64>) -> CurvatureInference {
    let n = y.len();
    // The design driver consumes feature columns by index; the spec built for
    // "y ~ curv(x1, x2)" references columns x1, x2 which in the encoded frame
    // are indices 1, 2 — but the spatial fit takes the FEATURE matrix directly,
    // so we hand it the [x1, x2] columns and a spec whose feature_cols are 0, 1.
    // build_termspec resolves x1->1, x2->2 against the 3-col schema; to keep the
    // column indices consistent we pass the full 3-col frame's feature view.
    let mut frame = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        frame[(i, 0)] = y[i];
        frame[(i, 1)] = feats[(i, 0)];
        frame[(i, 2)] = feats[(i, 1)];
    }
    let spec = termspec_for(FIT_FORMULA, &frame);

    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let options = FitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 24,
        rel_tol: 1e-5,
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

/// The planted curvature must lie INSIDE the box the estimator is allowed to
/// report (gam#2687's lesson, applied here by gam#2747).
///
/// The κ box is the half-margin to the antipodal fold, `±F/max‖x‖²`, so on this
/// radius-0.68 disk it is `±0.5/0.4624 = ±1.081`. The pre-#2747 fixture planted
/// `κ⋆ = ±2` — `κ⋆R² = ±0.925` against a cap of `F = 0.50`, i.e. **85% outside
/// the interval the estimator may report**. That was survivable only while the
/// criterion happened to turn over inside the box anyway; with the range
/// profiled it does not, and the CI machinery refuses by name rather than
/// manufacturing an interval: *"curvature profile is not outward-monotone at
/// chart bound −1.083"*. The refusal is correct — the profile is still
/// descending where the box ends, so the reported set would not be a truncation
/// of a connected likelihood set anchored at κ̂ — and the repair is the one
/// `#2687` applied to the sibling coverage fixture for the identical reason:
/// plant where coverage can be measured.
///
/// `0.75` is 69% of the cap, matching the `71%` that fixture settled on, so the
/// profile has room on both sides to close an interval.
const KAPPA_STAR: f64 = 0.75;

#[test]
fn spherical_truth_recovers_positive_kappa_and_rejects_flat() {
    let (feats, y) = dataset_on_m_kappa(n_obs(), KAPPA_STAR, 0.68, 0.02, 0x5151_0001);
    let inf = fit_and_infer(&feats, &y);
    log::trace!(
        "[spherical] κ̂={:.4} CI=[{:.4}, {:.4}] verdict={:?} flat_p={:.4} lr={:.4}",
        inf.kappa_hat,
        inf.ci.ci_lo,
        inf.ci.ci_hi,
        inf.ci.verdict,
        inf.flatness.p_value,
        inf.flatness.lr_stat
    );
    assert!(
        inf.kappa_hat > 0.0,
        "spherical truth κ⋆=+{KAPPA_STAR} should give κ̂ > 0, got {}",
        inf.kappa_hat
    );
    assert_ne!(
        inf.ci.verdict,
        CurvatureVerdict::Hyperbolic,
        "spherical truth must not be called hyperbolic"
    );
    assert!(
        inf.flatness.p_value < 0.05,
        "spherical truth should reject flatness (p<0.05), got p={}",
        inf.flatness.p_value
    );
}

#[test]
fn flat_truth_does_not_reject_flatness() {
    let (feats, y) = dataset_on_m_kappa(n_obs(), 0.0, 0.68, 0.02, 0x5151_0002);
    let inf = fit_and_infer(&feats, &y);
    log::trace!(
        "[flat] κ̂={:.4} CI=[{:.4}, {:.4}] verdict={:?} flat_p={:.4} lr={:.4}",
        inf.kappa_hat,
        inf.ci.ci_lo,
        inf.ci.ci_hi,
        inf.ci.verdict,
        inf.flatness.p_value,
        inf.flatness.lr_stat
    );
    // Correct size: flat data must NOT be spuriously called curved.
    assert!(
        inf.flatness.p_value > 0.05,
        "flat truth κ⋆=0 should NOT reject flatness, got p={}",
        inf.flatness.p_value
    );
    assert_eq!(
        inf.ci.verdict,
        CurvatureVerdict::Flat,
        "flat truth CI must straddle 0 (verdict Flat); CI=[{}, {}]",
        inf.ci.ci_lo,
        inf.ci.ci_hi
    );
}

#[test]
fn hyperbolic_truth_recovers_negative_kappa_and_rejects_flat() {
    let (feats, y) = dataset_on_m_kappa(n_obs(), -KAPPA_STAR, 0.68, 0.02, 0x5151_0003);
    let inf = fit_and_infer(&feats, &y);
    log::trace!(
        "[hyperbolic] κ̂={:.4} CI=[{:.4}, {:.4}] verdict={:?} flat_p={:.4} lr={:.4}",
        inf.kappa_hat,
        inf.ci.ci_lo,
        inf.ci.ci_hi,
        inf.ci.verdict,
        inf.flatness.p_value,
        inf.flatness.lr_stat
    );
    assert!(
        inf.kappa_hat < 0.0,
        "hyperbolic truth κ⋆=−2 should give κ̂ < 0, got {}",
        inf.kappa_hat
    );
    assert_ne!(
        inf.ci.verdict,
        CurvatureVerdict::Spherical,
        "hyperbolic truth must not be called spherical"
    );
    assert!(
        inf.flatness.p_value < 0.05,
        "hyperbolic truth should reject flatness (p<0.05), got p={}",
        inf.flatness.p_value
    );
}
