//! Measure-jet spline quality gates on a filament web (#904 paradigm: assert
//! against self-constructed truth; a mature in-crate smoother is the
//! match-or-beat baseline).
//!
//! Geometry: a Y-junction web in latent R² — strand A (-1,0)→(0,0), strand B
//! (0,0)→(0.8,0.6), strand C (0,0)→(0.8,-0.6) — embedded into ambient R⁸ by a
//! fixed orthonormal linear map plus small ambient coordinate noise. The
//! response is a trend in arc-length, continuous at the junction, with a
//! slope change onto strand C. Training deletes the middle third of strand B
//! (the gap), so the two gates are exactly the term's two contracts:
//!
//! 1. **Truth recovery off-gap**: at d = 8 ambient with 1-D intrinsic
//!    structure, the measure-learned geometry must recover the strand signal
//!    within 2.5× the observation noise, and match-or-beat the geometry-blind
//!    Duchon smoother on the same eight ambient columns (×1.10 flake guard —
//!    the jet term should genuinely win here, the guard only absorbs seed
//!    luck).
//! 2. **Gap bridging with the trend, not the mean**: inside the deleted
//!    stretch of strand B the fit must continue the flank-attested slope
//!    (same sign, within 60% of truth) rather than collapse toward the
//!    global training mean. This is the no-mass-term/N2 contract observed
//!    end to end through REML.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const AMBIENT_D: usize = 8;
const Y_NOISE_SIGMA: f64 = 0.1;
const COORD_NOISE_SIGMA: f64 = 0.02;
/// Arc-length slope of the response along A→B (truth the gap test recovers).
const TREND_SLOPE_AB: f64 = 1.5;
/// Arc-length slope along strand C after the junction (continuous level).
const TREND_SLOPE_C: f64 = -0.8;

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

fn sample_web(n_per_strand: usize, seed: u64, drop_b_gap: bool) -> Vec<WebPoint> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ut = Uniform::new(0.0, 1.0).expect("uniform");
    let cnoise = Normal::new(0.0, COORD_NOISE_SIGMA).expect("normal");
    let e = embedding();
    let mut out = Vec::new();
    for strand in 0..3usize {
        for _ in 0..n_per_strand {
            let t: f64 = ut.sample(&mut rng);
            // The training gap: middle third of strand B.
            if drop_b_gap && strand == 1 && (1.0 / 3.0..2.0 / 3.0).contains(&t) {
                continue;
            }
            let (z, truth) = latent_point(strand, t);
            let mut coords = [0.0; AMBIENT_D];
            for k in 0..AMBIENT_D {
                coords[k] = e[k][0] * z[0] + e[k][1] * z[1] + cnoise.sample(&mut rng);
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

fn encode_training(points: &[WebPoint], seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ynoise = Normal::new(0.0, Y_NOISE_SIGMA).expect("normal");
    let mut headers: Vec<String> = (0..AMBIENT_D).map(|k| format!("x{k}")).collect();
    headers.push("y".to_string());
    let rows: Vec<StringRecord> = points
        .iter()
        .map(|p| {
            let mut fields: Vec<String> = p.coords.iter().map(|v| v.to_string()).collect();
            fields.push((p.truth + ynoise.sample(&mut rng)).to_string());
            StringRecord::from(fields)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode web dataset")
}

fn fit_and_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    test: &[WebPoint],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("web fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut m = Array2::<f64>::zeros((test.len(), data.headers.len()));
    let cmap = data.column_map();
    for (i, p) in test.iter().enumerate() {
        for k in 0..AMBIENT_D {
            m[[i, cmap[format!("x{k}").as_str()]]] = p.coords[k];
        }
    }
    let test_design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    test_design.design.apply(&fit.fit.beta).to_vec()
}

fn rmse_vs_truth(pred: &[f64], test: &[WebPoint]) -> f64 {
    let n = pred.len() as f64;
    (pred
        .iter()
        .zip(test.iter())
        .map(|(p, q)| (p - q.truth) * (p - q.truth))
        .sum::<f64>()
        / n)
        .sqrt()
}

const MJS_FORMULA: &str = "y ~ mjs(x0, x1, x2, x3, x4, x5, x6, x7)";
const DUCHON_FORMULA: &str = "y ~ duchon(x0, x1, x2, x3, x4, x5, x6, x7)";

#[test]
fn measure_jet_recovers_web_signal_vs_truth() {
    init_parallelism();
    let train_points = sample_web(400, 41, true);
    let data = encode_training(&train_points, 42);
    // Held-out web draws OUTSIDE the gap (the gap has its own gate below).
    let test: Vec<WebPoint> = sample_web(140, 43, true);

    let jet_pred = fit_and_predict(MJS_FORMULA, &data, &test);
    let jet_rmse = rmse_vs_truth(&jet_pred, &test);

    // PRIMARY — truth recovery: the measure-learned geometry must resolve
    // the strand signal to within 2.5× the observation noise.
    assert!(
        jet_rmse <= 2.5 * Y_NOISE_SIGMA,
        "measure-jet truth recovery too weak: RMSE {jet_rmse:.4} vs bound {:.4}",
        2.5 * Y_NOISE_SIGMA
    );

    // SECONDARY — match-or-beat the geometry-blind mature in-crate smoother
    // on the same eight ambient columns (×1.10 flake guard only).
    let duchon_pred = fit_and_predict(DUCHON_FORMULA, &data, &test);
    let duchon_rmse = rmse_vs_truth(&duchon_pred, &test);
    assert!(
        jet_rmse <= 1.10 * duchon_rmse,
        "measure-jet must match-or-beat geometry-blind Duchon at d=8: jet {jet_rmse:.4} vs duchon {duchon_rmse:.4}"
    );
}

/// The support diagnostic must be computable from the FITTED model alone
/// (frozen nodes + masses + band ride the replay contract) and must separate
/// an on-web query from a far off-web query at the finest band scale — the
/// on-web-ness label every prediction ships with.
#[test]
fn measure_jet_support_curve_flags_off_web_queries() {
    init_parallelism();
    let train_points = sample_web(400, 41, true);
    let data = encode_training(&train_points, 42);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(MJS_FORMULA, &data, &cfg).expect("web fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    // Dig the frozen measure-jet term out of the resolved collection: the
    // freeze step must have pinned nodes, masses, and band.
    let (spec, scales) = fit
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
    let e = embedding();
    let (z_on, _) = latent_point(0, 0.5);
    let mut queries = Array2::<f64>::zeros((2, AMBIENT_D));
    for k in 0..AMBIENT_D {
        queries[(0, k)] = e[k][0] * z_on[0] + e[k][1] * z_on[1];
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
}

#[test]
fn measure_jet_bridges_gap_with_trend_not_mean() {
    init_parallelism();
    let train_points = sample_web(400, 41, true);
    let data = encode_training(&train_points, 42);
    // Test points INSIDE the strand-B gap (kept only at test time).
    let in_gap: Vec<WebPoint> = sample_web(400, 44, false)
        .into_iter()
        .filter(|p| p.strand == 1 && (1.0 / 3.0..2.0 / 3.0).contains(&p.t))
        .collect();
    assert!(in_gap.len() >= 60, "gap sample unexpectedly thin");

    let pred = fit_and_predict(MJS_FORMULA, &data, &in_gap);

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
    let train_mean =
        data.values.column(y_col).sum() / data.values.nrows() as f64;
    let mad_pred =
        pred.iter().map(|p| (p - train_mean).abs()).sum::<f64>() / n;
    let mad_truth = in_gap
        .iter()
        .map(|p| (p.truth - train_mean).abs())
        .sum::<f64>()
        / n;
    assert!(
        mad_pred >= 0.5 * mad_truth,
        "gap predictions collapse toward the training mean: MAD {mad_pred:.3} vs truth MAD {mad_truth:.3}"
    );
}
