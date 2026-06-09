//! By-factor smooth predict-path truth recovery (non-survival / Gaussian).
//!
//! A GAM `y ~ s(x, by=g)` for a categorical factor `g` lowers (in
//! `term_builder`) to ONE `SmoothBasisSpec::ByVariable { kind: Level, .. }`
//! block per factor level — the inner B-spline gated to that level's rows by an
//! exact `value.to_bits()` indicator — PLUS one treatment-coded factor main
//! effect carrying the per-group baseline level. At fit time each level block is
//! centered against its *gated* level indicator (so the within-level constant
//! goes to the factor main effect, not the smooth — mgcv's by-factor
//! convention), and the resulting orthogonalization transform `Z` is composed
//! into the inner B-spline metadata and frozen as `BSplineIdentifiability::
//! FrozenTransform`. At predict the frozen spec rebuilds the *same* gated,
//! transformed per-level design (`smooth_has_frozen_identifiability` skips
//! re-deriving `Z` from the fresh grid) and `remap_feature_columns` realigns the
//! `by_col` + inner `feature_col` to the prediction frame.
//!
//! This is the non-survival analogue of the survival weibull-by-factor
//! per-group-baseline drop (#900) and the linear-interaction `feature_cols`
//! predict remap drop (`82f184bb6`): if any per-level block were dropped,
//! mis-indexed, zeroed, or had its centering replayed inconsistently, a group's
//! fresh-grid prediction would lose its shape and/or its baseline.
//!
//! The test fits Gaussian data whose groups carry GENUINELY DIFFERENT smooth
//! shapes AND different baseline levels, JSON round-trips the frozen
//! `TermCollectionSpec` (exactly what model save → reload persists), rebuilds the
//! design on a FRESH per-group grid through the deserialized spec, and asserts
//! every group's prediction tracks its TRUE function (tight per-group RMSE) and
//! that the groups stay correctly separated. Pure gam-internal truth recovery —
//! no external tool.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::{TermCollectionSpec, build_term_collection_design};
use gam::test_support::reference::pearson;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_LEVEL: usize = 220;
const N_LEVELS: usize = 3;
const SIGMA: f64 = 0.06;
const SEED: u64 = 90091;

/// Per-group TRUE mean on x in [0,1]: a distinct shape PLUS a distinct baseline.
///
/// * grp0: baseline -1.5, shape sin(2 pi x)   (one full cycle)
/// * grp1: baseline  0.0, shape cos(3 pi x)   (different frequency + phase)
/// * grp2: baseline +1.5, shape 2(x-0.5)^2    (a parabola, not sinusoidal)
fn truth(level: usize, x: f64) -> f64 {
    use std::f64::consts::PI;
    match level {
        0 => -1.5 + (2.0 * PI * x).sin(),
        1 => 0.0 + (3.0 * PI * x).cos(),
        2 => 1.5 + 2.0 * (x - 0.5) * (x - 0.5),
        _ => unreachable!("only {N_LEVELS} levels generated"),
    }
}

#[test]
fn byfactor_smooth_predict_recovers_per_group_truth_on_fresh_grid() {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform support");
    let noise = Normal::new(0.0, SIGMA).expect("normal");

    let n = N_LEVELS * N_PER_LEVEL;
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g = Vec::<String>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for level in 0..N_LEVELS {
        for _ in 0..N_PER_LEVEL {
            let xi = ux.sample(&mut rng);
            x.push(xi);
            y.push(truth(level, xi) + noise.sample(&mut rng));
            g.push(format!("grp{level}"));
        }
    }

    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), g[i].clone(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode by-factor dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    // Recover each level's encoded (f64) value from a row carrying it, so the
    // fresh prediction grid uses the exact bit pattern the by-level gate expects.
    let level_values: Vec<f64> = (0..N_LEVELS)
        .map(|k| {
            let needle = format!("grp{k}");
            let row = g.iter().position(|v| v == &needle).expect("level present");
            ds.values[[row, g_idx]]
        })
        .collect();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, by=g)", &ds, &cfg)
        .unwrap_or_else(|e| panic!("by-factor Gaussian smooth must fit, got: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the by-factor Gaussian smooth");
    };
    assert!(
        fit.fit.beta.iter().all(|v| v.is_finite()),
        "fitted by-factor coefficients must be finite"
    );

    // Persist exactly what model save → reload round-trips: the frozen
    // `TermCollectionSpec` (per-level `ByVariable` blocks + their frozen inner
    // B-spline identifiability transforms + the factor main effect). Round-trip
    // through JSON so the test fails if any predict-bearing field is lost.
    let spec_json = serde_json::to_string(&fit.resolvedspec).expect("serialize frozen spec");
    let reloaded: TermCollectionSpec =
        serde_json::from_str(&spec_json).expect("deserialize frozen spec");
    let beta: Array1<f64> = fit.fit.beta.clone();

    // (a) Reload-then-replay reproduces the fitted in-sample fit EXACTLY: the
    // frozen, serialized by-factor design must rebuild bit-for-bit.
    let eta_fit: Vec<f64> = fit.design.design.apply(&beta).to_vec();
    let mut train_grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        train_grid[[i, x_idx]] = x[i];
        train_grid[[i, g_idx]] = ds.values[[i, g_idx]];
    }
    let eta_replay: Vec<f64> = build_term_collection_design(train_grid.view(), &reloaded)
        .expect("rebuild frozen by-factor design at training rows")
        .design
        .apply(&beta)
        .to_vec();
    let max_abs_dev = eta_fit
        .iter()
        .zip(&eta_replay)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_dev < 1e-8,
        "reloaded by-factor replay must reproduce the fitted eta (max|delta|={max_abs_dev:.3e})"
    );

    // (b) Predict each group on a FRESH dense grid through the reloaded spec.
    const N_GRID: usize = 120;
    let xs_grid: Vec<f64> = (0..N_GRID)
        .map(|j| j as f64 / (N_GRID as f64 - 1.0))
        .collect();
    let mut pred_curves: Vec<Vec<f64>> = Vec::with_capacity(N_LEVELS);
    let mut true_curves: Vec<Vec<f64>> = Vec::with_capacity(N_LEVELS);
    for (level, &lvl_bits) in level_values.iter().enumerate() {
        let mut grid = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
        for (j, &xj) in xs_grid.iter().enumerate() {
            grid[[j, x_idx]] = xj;
            grid[[j, g_idx]] = lvl_bits;
        }
        let eta: Vec<f64> = build_term_collection_design(grid.view(), &reloaded)
            .expect("rebuild by-factor design on a fresh single-level grid")
            .design
            .apply(&beta)
            .to_vec();
        assert!(
            eta.iter().all(|v| v.is_finite()),
            "fresh-grid by-factor predictions must be finite for grp{level}"
        );
        pred_curves.push(eta);
        true_curves.push(xs_grid.iter().map(|&xj| truth(level, xj)).collect());
    }

    // A single GLOBAL constant (the model intercept reference) is shared across
    // every group; subtract the pooled mean prediction-vs-truth gap once so we
    // assess per-group SHAPE and RELATIVE BASELINE recovery (not an arbitrary
    // overall offset). Per-group constants are NOT removed: a dropped/leaked
    // per-group baseline would survive this single global shift and fail below.
    let mut pooled_gap = 0.0_f64;
    let mut pooled_cnt = 0.0_f64;
    for level in 0..N_LEVELS {
        for j in 0..N_GRID {
            pooled_gap += pred_curves[level][j] - true_curves[level][j];
            pooled_cnt += 1.0;
        }
    }
    let global_shift = pooled_gap / pooled_cnt;

    // (c) Per-group truth recovery: each group's shifted prediction must track
    // its TRUE function (shape + its own baseline) within a tight RMSE bar.
    const RMSE_BAR: f64 = 0.12;
    for level in 0..N_LEVELS {
        let mut sse = 0.0_f64;
        for j in 0..N_GRID {
            let resid = (pred_curves[level][j] - global_shift) - true_curves[level][j];
            sse += resid * resid;
        }
        let rmse = (sse / N_GRID as f64).sqrt();
        assert!(
            rmse < RMSE_BAR,
            "grp{level}: by-factor fresh-grid prediction did not recover the true function \
             (RMSE={rmse:.4} >= {RMSE_BAR}); a dropped/zeroed/mis-centered per-level block \
             or lost baseline is the prime suspect"
        );
    }

    // (d) The groups must stay correctly SEPARATED: each predicted curve must
    // correlate most strongly with its OWN true shape, not another group's.
    for level in 0..N_LEVELS {
        let own = pearson(&pred_curves[level], &true_curves[level]).abs();
        for other in 0..N_LEVELS {
            if other == level {
                continue;
            }
            let cross = pearson(&pred_curves[level], &true_curves[other]).abs();
            assert!(
                own > cross + 0.05,
                "grp{level} predicted curve is not better aligned with its own true shape \
                 (|corr_own|={own:.3}) than with grp{other}'s (|corr_cross|={cross:.3}); \
                 the by-factor levels were swapped or collapsed at predict"
            );
        }
    }
}
