//! Measure-jet spline quality gates on a filament web (#904 paradigm: assert
//! against self-constructed truth; a mature in-crate smoother is the
//! match-or-beat baseline).
//!
//! Geometry: a Y-junction web in latent R² — strand A (-1,0)→(0,0), strand B
//! (0,0)→(0.8,0.6), strand C (0,0)→(0.8,-0.6) — embedded into ambient R⁸ by a
//! fixed orthonormal linear map plus small ambient coordinate noise. The
//! response is a trend in arc-length, continuous at the junction, with a
//! slope change onto strand C. Training deletes the middle third of strand B
//! (the gap). One integrated gate checks five contracts in file order, sharing
//! fitted models where the contracts intentionally use the same data-generating
//! process:
//!
//! 1. **Truth recovery off-gap**:
//!    at d = 8 ambient with 1-D intrinsic structure, the measure-learned
//!    geometry must recover the strand signal within 2.5× the observation
//!    noise, and match-or-beat the geometry-blind Duchon smoother on the same
//!    eight ambient columns (×1.10 flake guard — the jet term should
//!    genuinely win here, the guard only absorbs seed luck).
//! 2. **Support diagnostic**:
//!    the support curve must be computable from the FITTED model alone and
//!    must separate an on-web query from a far off-web query at the finest
//!    band scale.
//! 3. **Gap bridging with the trend, not the mean**: inside the deleted
//!    stretch of strand B the fit must continue the flank-attested slope
//!    (same sign, within 60% of truth) rather than collapse toward the
//!    global training mean. This is the no-mass-term/N2 contract observed
//!    end to end through REML.
//! 4. **GLM composition**: the same web with a Poisson count response — the
//!    measure-jet penalty must
//!    compose with PIRLS/REML for a non-gaussian family and recover the
//!    log-intensity off-gap.
//! 5. **Interval honesty**: 95% pointwise bands built from the fit's
//!    smoothing-corrected coefficient
//!    covariance must approximately cover the true mean at held-out
//!    on-web points.

use csv::StringRecord;
use gam::basis::{CenterStrategy, MeasureJetExtrapolationSpectrum, PenaltySource};
use gam::matrix::LinearOperator;
use gam::smooth::{SmoothBasisSpec, build_term_collection_design};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};

const AMBIENT_D: usize = 8;
const Y_NOISE_SIGMA: f64 = 0.1;
const COORD_NOISE_SIGMA: f64 = 0.02;
/// Arc-length slope of the response along A→B (truth the gap test recovers).
const TREND_SLOPE_AB: f64 = 1.5;
/// Arc-length slope along strand C after the junction (continuous level).
const TREND_SLOPE_C: f64 = -0.8;
/// Poisson arm: η = POISSON_ETA_SCALE · (arc-length truth). The truth spans
/// [0, 3] over the web, so η ∈ [0, 1.5] and μ = e^η ∈ [1, ~4.5] — counts small
/// enough that the Poisson information is genuinely sparse, large enough that
/// log-intensity recovery is well posed.
const POISSON_ETA_SCALE: f64 = 0.5;
/// Two-sided 95% normal quantile, qnorm(0.975).
const Z_95: f64 = 1.959963984540054;
/// All eight ambient columns handed to the measure-jet term.
const MJS_PREDICTIVE_FORMULA: &str =
    "y ~ mjs(x0, x1, x2, x3, x4, x5, x6, x7, centers=24, scales=3)";
const MJS_INTERVAL_FORMULA: &str = "y ~ mjs(x0, x1, x2, x3, x4, x5, x6, x7, centers=24)";
/// Geometry-blind mature in-crate baseline on the same ambient columns.
const DUCHON_FORMULA: &str = "y ~ duchon(x0, x1, x2, x3, x4, x5, x6, x7)";

// ---------------------------------------------------------------------------
// Data generation: latent web geometry, R²→R⁸ embedding, response encoders.
// ---------------------------------------------------------------------------

/// Fixed embedding R² → R⁸: two exactly orthonormal columns (Householder-free
/// hand construction; entries chosen irrational-ish so no ambient axis is
/// privileged), hardcoded for determinism.
fn embedding() -> [[f64; 2]; AMBIENT_D] {
    // Two orthogonal unit vectors in R⁸.
    let u = [0.42, -0.31, 0.18, 0.55, -0.27, 0.36, 0.22, -0.41];
    let v0 = [0.12, 0.47, -0.39, 0.08, 0.51, 0.24, -0.33, 0.17];
    let nu = u.iter().map(|a| a * a).sum::<f64>().sqrt();
    let un: Vec<f64> = u.iter().map(|a| a / nu).collect();
    let dot = un.iter().zip(v0.iter()).map(|(a, b)| a * b).sum::<f64>();
    let mut v: Vec<f64> = v0.iter().zip(un.iter()).map(|(b, a)| b - dot * a).collect();
    let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    for b in v.iter_mut() {
        *b /= nv;
    }
    let mut e = [[0.0; 2]; AMBIENT_D];
    for k in 0..AMBIENT_D {
        e[k][0] = un[k];
        e[k][1] = v[k];
    }
    e
}

/// Apply the fixed R² → R⁸ embedding to a latent point (no ambient noise).
fn embed_latent(e: &[[f64; 2]; AMBIENT_D], z: [f64; 2]) -> [f64; AMBIENT_D] {
    let mut coords = [0.0; AMBIENT_D];
    for k in 0..AMBIENT_D {
        coords[k] = e[k][0] * z[0] + e[k][1] * z[1];
    }
    coords
}

/// One sampled web point: ambient coordinates, true response value, strand
/// id (0 = A, 1 = B, 2 = C), and the strand parameter t ∈ [0, 1].
struct WebPoint {
    coords: [f64; AMBIENT_D],
    truth: f64,
    strand: usize,
    t: f64,
}

/// Latent strand geometry + arc-length response truth (continuous at the
/// junction: f = TREND_SLOPE_AB·(arc from A's far end); strand C continues
/// the junction level with its own slope).
fn latent_point(strand: usize, t: f64) -> ([f64; 2], f64) {
    match strand {
        0 => ([-1.0 + t, 0.0], TREND_SLOPE_AB * t),
        1 => (
            [0.8 * t, 0.6 * t],
            TREND_SLOPE_AB * (1.0 + t), // |B| = 1.0, slope continues
        ),
        _ => (
            [0.8 * t, -0.6 * t],
            TREND_SLOPE_AB + TREND_SLOPE_C * t, // junction level + C slope
        ),
    }
}

/// The training gap: the middle third of strand B (deleted at fit time,
/// probed at test time by the bridging gate).
fn in_b_gap(strand: usize, t: f64) -> bool {
    strand == 1 && (1.0 / 3.0..2.0 / 3.0).contains(&t)
}

fn sample_web(n_per_strand: usize, seed: u64, drop_b_gap: bool) -> Vec<WebPoint> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ut = Uniform::new(0.0, 1.0).expect("uniform");
    let cnoise = Normal::new(0.0, COORD_NOISE_SIGMA).expect("normal");
    let e = embedding();
    let mut out = Vec::new();
    for strand in 0..3usize {
        for _ in 0..n_per_strand {
            let t: f64 = ut.sample(&mut rng);
            if drop_b_gap && in_b_gap(strand, t) {
                continue;
            }
            let (z, truth) = latent_point(strand, t);
            let mut coords = embed_latent(&e, z);
            for c in coords.iter_mut() {
                *c += cnoise.sample(&mut rng);
            }
            out.push(WebPoint {
                coords,
                truth,
                strand,
                t,
            });
        }
    }
    out
}

/// Encode web points as the training dataset: ambient columns x0..x7 plus a
/// `y` column drawn per point by `response` from one seeded StdRng (a single
/// deterministic stream per encoder call, in point order).
fn encode_web(
    points: &[WebPoint],
    seed: u64,
    mut response: impl FnMut(&WebPoint, &mut StdRng) -> f64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut headers: Vec<String> = (0..AMBIENT_D).map(|k| format!("x{k}")).collect();
    headers.push("y".to_string());
    let rows: Vec<StringRecord> = points
        .iter()
        .map(|p| {
            let mut fields: Vec<String> = p.coords.iter().map(|v| v.to_string()).collect();
            fields.push(response(p, &mut rng).to_string());
            StringRecord::from(fields)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode web dataset")
}

/// Gaussian arm: y = truth + N(0, Y_NOISE_SIGMA), one draw per point.
fn encode_training(points: &[WebPoint], seed: u64) -> gam::data::EncodedDataset {
    let ynoise = Normal::new(0.0, Y_NOISE_SIGMA).expect("normal");
    encode_web(points, seed, |p, rng| p.truth + ynoise.sample(rng))
}

/// Poisson arm: y ~ Poisson(exp(η)) with η = POISSON_ETA_SCALE · truth, one
/// draw per point (mirrors the gaussian arm's stream discipline).
fn encode_poisson_training(points: &[WebPoint], seed: u64) -> gam::data::EncodedDataset {
    encode_web(points, seed, |p, rng| {
        let mu = (POISSON_ETA_SCALE * p.truth).exp();
        Poisson::new(mu).expect("poisson").sample(rng)
    })
}

// ---------------------------------------------------------------------------
// Fit + readout: formula fits, frozen-spec design replay, error metrics.
// ---------------------------------------------------------------------------

fn fit_web(
    formula: &str,
    data: &gam::data::EncodedDataset,
    family: &str,
) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some(family.to_string()),
        outer_max_iter: Some(35),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("web fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    fit
}

/// Raw ambient-coordinate matrix for the test points, laid out on the
/// training dataset's column map (the shape `build_term_collection_design`
/// replays the frozen spec against).
fn ambient_matrix(data: &gam::data::EncodedDataset, test: &[WebPoint]) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((test.len(), data.headers.len()));
    let cmap = data.column_map();
    for (i, p) in test.iter().enumerate() {
        for k in 0..AMBIENT_D {
            m[[i, cmap[format!("x{k}").as_str()]]] = p.coords[k];
        }
    }
    m
}

/// Evaluate the LINEAR PREDICTOR η = X·β at the test points (for the gaussian
/// arm η is the mean; for poisson/log it is the log-intensity — applying the
/// frozen design to β never passes through the inverse link).
fn predict_with_fit(
    fit: &gam::StandardFitResult,
    data: &gam::data::EncodedDataset,
    test: &[WebPoint],
) -> Vec<f64> {
    let m = ambient_matrix(data, test);
    let test_design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    test_design.design.apply(&fit.fit.beta).to_vec()
}

/// RMSE of predictions against `truth_scale · truth` (scale 1.0 for the
/// gaussian arms; POISSON_ETA_SCALE for the count arm, where the comparison
/// lives on the η = log-intensity scale).
fn rmse_vs_truth(pred: &[f64], test: &[WebPoint], truth_scale: f64) -> f64 {
    let n = pred.len() as f64;
    (pred
        .iter()
        .zip(test.iter())
        .map(|(p, q)| {
            let e = p - truth_scale * q.truth;
            e * e
        })
        .sum::<f64>()
        / n)
        .sqrt()
}

/// Pointwise prediction SE from a coefficient covariance: se_i = sqrt(s_i^T covariance s_i).
/// for design row sᵢ (same access pattern as the mgcv-CI quality suite).
fn pointwise_se(design: ArrayView2<'_, f64>, cov: &Array2<f64>) -> Vec<f64> {
    let p = design.ncols();
    assert_eq!(cov.nrows(), p, "covariance/design dimension mismatch");
    assert_eq!(cov.ncols(), p, "covariance must be square");
    design
        .rows()
        .into_iter()
        .map(|s| {
            let mut acc = 0.0;
            for a in 0..p {
                let sa = s[a];
                if sa == 0.0 {
                    continue;
                }
                let mut row_dot = 0.0;
                for b in 0..p {
                    row_dot += cov[[a, b]] * s[b];
                }
                acc += sa * row_dot;
            }
            acc.max(0.0).sqrt()
        })
        .collect()
}

/// Test-local mirror of the production measure-jet extrapolation variance
/// producer. The interval contract is total variance = posterior variance + extrapolation variance, where
/// `Var_extrap` prices finite-support uncertainty from the frozen measure-jet
/// spectrum; a posterior-covariance-only band is intentionally incomplete for this smooth.
fn measure_jet_extrapolation_variance_for_fit(
    fit: &gam::StandardFitResult,
    data: &gam::data::EncodedDataset,
    test: &[WebPoint],
) -> Array1<f64> {
    let raw = ambient_matrix(data, test);
    let mut total = Array1::<f64>::zeros(test.len());
    let phi_scale = fit.fit.coefficient_covariance_scale();
    for term in &fit.resolvedspec.smooth_terms {
        let SmoothBasisSpec::MeasureJet {
            feature_cols,
            spec,
            input_scales,
        } = &term.basis
        else {
            continue;
        };
        let (Some(frozen), CenterStrategy::UserProvided(centers)) =
            (spec.frozen_quadrature.as_ref(), &spec.center_strategy)
        else {
            panic!("interval gate needs frozen measure-jet geometry");
        };
        let n_levels = frozen.eps_band.len();
        let mut per_scale: Vec<(usize, f64)> = Vec::new();
        let mut fused = None;
        for info in &fit.design.penaltyinfo {
            if info.termname.as_deref() != Some(term.name.as_str()) {
                continue;
            }
            let lambda = fit.fit.lambdas[info.global_index];
            match &info.penalty.source {
                PenaltySource::Other(label) => {
                    if let Some(level_txt) = label.strip_prefix("measure_jet_scale_") {
                        let level: usize = level_txt.parse().expect("measure-jet scale label");
                        per_scale.push((level, lambda));
                    }
                }
                PenaltySource::Primary => {
                    fused = Some(lambda);
                }
                _ => {}
            }
        }
        let mut lambda_phys = Vec::with_capacity(n_levels);
        let spectrum = if per_scale.is_empty() {
            let lambda = fused.expect("fused measure-jet penalty lambda");
            let c = frozen
                .fused_penalty_normalization_scale
                .expect("fused measure-jet penalty normalization scale");
            MeasureJetExtrapolationSpectrum::Fused(lambda / c)
        } else {
            per_scale.sort_by_key(|&(level, _)| level);
            assert_eq!(
                per_scale.len(),
                n_levels,
                "measure-jet per-scale lambda count"
            );
            assert_eq!(
                frozen.penalty_normalization_scales.len(),
                n_levels,
                "measure-jet per-scale normalization count"
            );
            lambda_phys.extend(
                per_scale
                    .iter()
                    .map(|&(level, lambda)| lambda / frozen.penalty_normalization_scales[level]),
            );
            MeasureJetExtrapolationSpectrum::PerLevel(&lambda_phys)
        };
        let mut queries = Array2::<f64>::zeros((test.len(), feature_cols.len()));
        for (j, &col) in feature_cols.iter().enumerate() {
            queries.column_mut(j).assign(&raw.column(col));
        }
        if let Some(scales) = input_scales {
            for (j, &scale) in scales.iter().enumerate() {
                queries.column_mut(j).mapv_inplace(|v| v / scale);
            }
        }
        let support = gam::basis::measure_jet_support_curve(
            queries.view(),
            centers.view(),
            frozen.masses.view(),
            &frozen.eps_band,
        )
        .expect("measure-jet support curve for interval gate");
        for i in 0..test.len() {
            let v = gam::basis::measure_jet_extrapolation_variance(
                support.row(i),
                &frozen.eps_band,
                &frozen.support_means,
                spectrum,
                0.05,
            )
            .expect("measure-jet extrapolation variance for interval gate");
            total[i] += phi_scale * v;
        }
    }
    total
}

/// Integrated quality gate: one predictive gaussian measure-jet fit feeds the
/// truth/support/bridge contracts, one Duchon fit supplies the mature baseline,
/// one Poisson measure-jet fit gates PIRLS/REML composition, and one
/// interval-resolution measure-jet fit gates covariance coverage. This keeps
/// the suite honest and removes the old harness shape that launched five
/// independent threaded REML fits over the same web fixture.
#[test]
fn measure_jet_web_quality_contracts() {
    init_parallelism();
    let train_points = sample_web(400, 41, true);
    let data = encode_training(&train_points, 42);
    let jet_fit = fit_web(MJS_PREDICTIVE_FORMULA, &data, "gaussian");

    // Contract 1 — truth recovery off-gap: with d = 8 ambient columns and 1-D
    // intrinsic structure, the fit must resolve the strand signal to within
    // 2.5× the observation noise (PRIMARY), and match-or-beat the
    // geometry-blind Duchon baseline on the same columns (SECONDARY, ×1.10
    // flake guard for seed luck only).
    // Held-out web draws OUTSIDE the gap (the gap has its own gate below).
    let test: Vec<WebPoint> = sample_web(140, 43, true);

    let jet_pred = predict_with_fit(&jet_fit, &data, &test);
    let jet_rmse = rmse_vs_truth(&jet_pred, &test, 1.0);

    // PRIMARY — truth recovery: the measure-learned geometry must resolve
    // the strand signal to within 2.5× the observation noise.
    assert!(
        jet_rmse <= 2.5 * Y_NOISE_SIGMA,
        "measure-jet truth recovery too weak: RMSE {jet_rmse:.4} vs bound {:.4}",
        2.5 * Y_NOISE_SIGMA
    );

    // SECONDARY — match-or-beat the geometry-blind mature in-crate smoother
    // on the same eight ambient columns (×1.10 flake guard only).
    let duchon_fit = fit_web(DUCHON_FORMULA, &data, "gaussian");
    let duchon_pred = predict_with_fit(&duchon_fit, &data, &test);
    let duchon_rmse = rmse_vs_truth(&duchon_pred, &test, 1.0);
    assert!(
        jet_rmse <= 1.10 * duchon_rmse,
        "measure-jet must match-or-beat geometry-blind Duchon at d=8: jet {jet_rmse:.4} vs duchon {duchon_rmse:.4}"
    );

    // Contract 2 — support diagnostic: computable from the FITTED model alone
    // (frozen nodes + masses + band + support anchors ride the replay
    // contract) and must separate an on-web query from a far off-web query at
    // the finest band scale — the on-web-ness label every prediction ships
    // with.
    // Dig the frozen measure-jet term out of the resolved collection: the
    // freeze step must have pinned nodes, masses, band, and support anchors.
    let (spec, scales) = jet_fit
        .resolvedspec
        .smooth_terms
        .iter()
        .find_map(|st| match &st.basis {
            gam::smooth::SmoothBasisSpec::MeasureJet {
                spec, input_scales, ..
            } => Some((spec.clone(), input_scales.clone())),
            _ => None,
        })
        .expect("fitted model carries a measure-jet term");
    let gam::basis::CenterStrategy::UserProvided(centers) = &spec.center_strategy else {
        panic!("frozen measure-jet term must pin its quadrature nodes")
    };
    let frozen = spec
        .frozen_quadrature
        .as_ref()
        .expect("frozen measure-jet term must carry its fit-time quadrature");
    // One on-web query (mid strand A, no coordinate noise) and one query far
    // off the web, both standardized exactly as the dispatch standardizes
    // training rows.
    let (z_on, _) = latent_point(0, 0.5);
    let on_web = embed_latent(&embedding(), z_on);
    let mut queries = Array2::<f64>::zeros((2, AMBIENT_D));
    for k in 0..AMBIENT_D {
        queries[(0, k)] = on_web[k];
        queries[(1, k)] = 5.0;
    }
    if let Some(s) = &scales {
        for k in 0..AMBIENT_D {
            queries[(0, k)] /= s[k];
            queries[(1, k)] /= s[k];
        }
    }
    let curves = gam::basis::measure_jet_support_curve(
        queries.view(),
        centers.view(),
        frozen.masses.view(),
        &frozen.eps_band,
    )
    .expect("support curve from frozen model");
    assert!(
        curves[(0, 0)] > 10.0 * curves[(1, 0)],
        "fine-scale support must separate on-web from off-web: on {:.3e} vs off {:.3e}",
        curves[(0, 0)],
        curves[(1, 0)]
    );

    // Contract 3 — gap bridging: inside the deleted stretch of strand B the fit
    // must continue the flank-attested slope (same sign, within 60% of truth)
    // rather than collapse toward the global training mean — the
    // no-mass-term/N2 contract observed end to end through REML.
    // Test points INSIDE the strand-B gap (kept only at test time).
    let in_gap: Vec<WebPoint> = sample_web(400, 44, false)
        .into_iter()
        .filter(|p| in_b_gap(p.strand, p.t))
        .collect();
    assert!(in_gap.len() >= 60, "gap sample unexpectedly thin");

    let pred = predict_with_fit(&jet_fit, &data, &in_gap);

    // Fitted slope across the gap: least-squares regression of prediction on
    // arc-length (= t along B), compared to the flank-attested truth slope.
    let n = in_gap.len() as f64;
    let tbar = in_gap.iter().map(|p| p.t).sum::<f64>() / n;
    let pbar = pred.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (p, q) in pred.iter().zip(in_gap.iter()) {
        cov += (q.t - tbar) * (p - pbar);
        var += (q.t - tbar) * (q.t - tbar);
    }
    let slope_hat = cov / var;
    // d(truth)/dt on strand B = TREND_SLOPE_AB (|B| = 1).
    let slope_truth = TREND_SLOPE_AB;
    assert!(
        slope_hat * slope_truth > 0.0,
        "gap bridge has the wrong trend direction: fitted slope {slope_hat:.3} vs truth {slope_truth:.3}"
    );
    assert!(
        (slope_hat - slope_truth).abs() <= 0.6 * slope_truth.abs(),
        "gap bridge does not carry the flank trend: fitted slope {slope_hat:.3} vs truth {slope_truth:.3}"
    );

    // No collapse to the global mean: in-gap predictions must deviate from
    // the training mean at least half as strongly as the truth does (a
    // mean-reverting bridge scores near zero here).
    let y_col = data.column_map()["y"];
    let train_mean = data.values.column(y_col).sum() / data.values.nrows() as f64;
    let mad_pred = pred.iter().map(|p| (p - train_mean).abs()).sum::<f64>() / n;
    let mad_truth = in_gap
        .iter()
        .map(|p| (p.truth - train_mean).abs())
        .sum::<f64>()
        / n;
    assert!(
        mad_pred >= 0.5 * mad_truth,
        "gap predictions collapse toward the training mean: MAD {mad_pred:.3} vs truth MAD {mad_truth:.3}"
    );

    // Contract 4 — GLM composition: the measure-jet penalty must compose with
    // PIRLS/REML for a non-gaussian family. Same Y-web geometry, count response
    // y ~ Poisson(exp(η)) with η = POISSON_ETA_SCALE · (arc-length truth),
    // scored on the η (log-intensity) scale at held-out off-gap points —
    // applying the frozen design to β yields η directly, the same scale the
    // truth lives on (the convention every non-gaussian quality test in this
    // suite uses).
    // Same geometry seed as the gaussian arms; counts get their own stream.
    let poisson_data = encode_poisson_training(&train_points, 45);
    // Held-out web draws OUTSIDE the gap (gap extrapolation is gated by the
    // gaussian bridge test; this gate isolates the GLM composition).
    let poisson_test: Vec<WebPoint> = sample_web(140, 43, true);

    let poisson_fit = fit_web(MJS_PREDICTIVE_FORMULA, &poisson_data, "poisson");
    let poisson_pred = predict_with_fit(&poisson_fit, &poisson_data, &poisson_test);
    let eta_rmse = rmse_vs_truth(&poisson_pred, &poisson_test, POISSON_ETA_SCALE);

    // Bound: η ∈ [0, 1.5] ⇒ μ = e^η ∈ [1, ~4.5]; the per-observation Fisher
    // information on the η scale is μ, so the raw per-point η noise is
    // 1/√μ ≈ 0.47–1.0. With ~1.1k training counts over a 1-D intrinsic web,
    // REML smoothing concentrates that noise into a per-point estimator error
    // of roughly sd·√(edf/n) ≈ 0.1, so 0.5 is ~5× the effective per-point
    // noise — generous against seed luck, yet well below the raw noise floor
    // and far below the 1.5 signal span, so a broken PIRLS/REML composition
    // (wrong link gradient, penalty desync, working-weight drift) cannot pass.
    assert!(
        eta_rmse <= 0.5,
        "measure-jet poisson log-intensity recovery too weak: η-RMSE {eta_rmse:.4} vs bound 0.5"
    );

    // Contract 5 — interval honesty: pointwise 95% bands from the fit's PUBLIC
    // smoothing-corrected coefficient covariance (`beta_covariance_corrected`,
    // the mgcv `predict(se.fit = TRUE)` analog used across the CI quality suite)
    // must approximately cover the TRUE mean at held-out on-web points.
    // Held-out ON-WEB points outside the gap: coverage is a claim about
    // interpolation honesty; gap extrapolation has its own (bias) gate.
    let coverage_test: Vec<WebPoint> = sample_web(140, 46, true);

    let interval_fit = fit_web(MJS_INTERVAL_FORMULA, &data, "gaussian");
    let m = ambient_matrix(&data, &coverage_test);
    let design = build_term_collection_design(m.view(), &interval_fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    let pred = design.design.apply(&interval_fit.fit.beta);
    let dense = design.design.to_dense();

    // The corrected covariance propagates smoothing-parameter uncertainty into the band; it is the
    // covariance the rest of the CI quality suite gates coverage on. If the
    // measure-jet path fails to populate it, that is an honest red, not a
    // reason to fall back to a weaker object.
    let vp = interval_fit
        .fit
        .beta_covariance_corrected()
        .expect("standard gaussian fit exposes the smoothing-corrected covariance");
    let mut se = pointwise_se(dense.view(), vp);
    let extrap = measure_jet_extrapolation_variance_for_fit(&interval_fit, &data, &coverage_test);
    for (s, v) in se.iter_mut().zip(extrap.iter()) {
        *s = (*s * *s + *v).sqrt();
    }

    let mut hits = 0usize;
    for ((p, s), q) in pred.iter().zip(se.iter()).zip(coverage_test.iter()) {
        if q.truth >= p - Z_95 * s && q.truth <= p + Z_95 * s {
            hits += 1;
        }
    }
    let coverage = hits as f64 / coverage_test.len() as f64;

    // Window [0.85, 1.0] for the PRODUCTION total variance:
    // posterior variance plus `measure_jet_extrapolation_variance`. All ~390 trials share one
    // fitted curve and one noise draw, so the empirical rate is far noisier
    // than a binomial count would suggest. The 0.85 floor still rejects
    // systematically small SEs (a covariance that drops the measure-jet block,
    // smoothing-variance term, or finite-support term under-covers by far more
    // than 10 nominal points). The upper edge is inclusive by construction:
    // the measure-jet extrapolation term is a one-sided honesty add-on for
    // finite-support uncertainty, so full coverage on this small correlated
    // web probe is conservative, not a failure of the total-variance contract.
    assert!(
        (0.85..=1.0).contains(&coverage),
        "measure-jet 95% interval coverage {coverage:.4} outside the honest window [0.85, 1.0] \
         ({hits}/{} held-out on-web points)",
        coverage_test.len()
    );
}
