//! Regression for #700 — sum-to-zero factor smooths (`bs="sz"`) must FIT and
//! REPLAY, not crash basis generation.
//!
//! `s(g, x, bs="sz")` lowers to `SmoothBasisSpec::FactorSumToZero`: one inner
//! marginal, replicated into `L-1` sum-to-zero deviation blocks. The sz path
//! forces the inner marginal to a single penalty (free per-level null space, to
//! match mgcv), so the L-1-block penalty keeps a non-trivial null space and the
//! raw builder computes a joint-null absorption rotation `Q` in the FULL
//! `p·(L-1)`-column design space. `apply_global_smooth_identifiability` then tried
//! to fold that full-design `Q` into the term's PER-MARGINAL metadata, a hard
//! dimension mismatch that crashed every sz fit ("identifiability transform
//! mismatch: existing is pxr, extra is (p·(L-1))x(p·(L-1))").
//!
//! Fix: the per-marginal metadata is left untouched for `FactorSumToZero`; the raw
//! builder already applies `Q` to the re-stacked design and recomputes it
//! deterministically from the (frozen) penalties at predict time. This test guards
//! the fix from a moderate-`n`, multi-level, fresh-grid-prediction angle that
//! complements the small-`n` exact-replay unit in
//! `bug_hunt_factor_smooth_degree_shrink_predict_replay`.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::pearson;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_LEVEL: usize = 60;
const N_LEVELS: usize = 4;
const SIGMA: f64 = 0.08;
const SEED: u64 = 700;

/// Distinct per-level truths on [0,1]: level `k` carries `sin((k+2)·π·x)`.
fn truth(level: usize, x: f64) -> f64 {
    ((level as f64 + 2.0) * std::f64::consts::PI * x).sin()
}

#[test]
fn sz_factor_smooth_fits_replays_and_keeps_levels_distinct() {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");

    let n = N_LEVELS * N_PER_LEVEL;
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g = Vec::<String>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for level in 0..N_LEVELS {
        let mut xs: Vec<f64> = (0..N_PER_LEVEL).map(|_| ux.sample(&mut rng)).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &xi in &xs {
            x.push(xi);
            y.push(truth(level, xi) + noise.sample(&mut rng));
            g.push(format!("grp{level}"));
        }
    }

    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), g[i].clone(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode sz dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];
    // Recover each level's encoded (f64) value from the first row carrying it.
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
    // The exact form the #700 quality suite measures (factor-first), which crashed
    // before the fix for every size.
    let result = fit_from_formula("y ~ s(g, x, bs=\"sz\")", &ds, &cfg)
        .unwrap_or_else(|e| panic!("#700: sz factor smooth must fit, got: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the sz factor smooth");
    };
    assert!(
        fit.fit.beta.iter().all(|v| v.is_finite()),
        "#700: fitted sz coefficients must be finite"
    );

    // (a) Frozen replay reproduces the fitted η exactly — the chart-exactness that
    // proves the per-marginal metadata + raw-builder Q replay are consistent.
    let eta_fit: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
    let mut train_grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        train_grid[[i, x_idx]] = x[i];
        train_grid[[i, g_idx]] = ds.values[[i, g_idx]];
    }
    let eta_replay: Vec<f64> = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("#700: rebuild frozen sz design at training rows")
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    let max_abs_dev = eta_fit
        .iter()
        .zip(&eta_replay)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_dev < 1e-8,
        "#700: frozen sz replay must reproduce the fitted η (max|Δ|={max_abs_dev:.3e})"
    );

    // (b) Fresh per-level grids predict finite values and recover distinct curves
    // (the sz mechanism must not collapse the levels onto one shared smooth).
    const N_GRID: usize = 100;
    let mut curves: Vec<Vec<f64>> = Vec::with_capacity(N_LEVELS);
    for &lvl in &level_values {
        let mut grid = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
        for j in 0..N_GRID {
            grid[[j, x_idx]] = j as f64 / (N_GRID as f64 - 1.0);
            grid[[j, g_idx]] = lvl;
        }
        let eta: Vec<f64> = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("#700: rebuild sz design on a fresh single-level grid must not panic")
            .design
            .apply(&fit.fit.beta)
            .to_vec();
        assert!(
            eta.iter().all(|v| v.is_finite()),
            "#700: fresh-grid sz predictions must be finite"
        );
        curves.push(eta);
    }
    // At least one pair of level curves must be genuinely distinct.
    let mut min_abs_corr = 1.0_f64;
    for a in 0..N_LEVELS {
        for b in (a + 1)..N_LEVELS {
            min_abs_corr = min_abs_corr.min(pearson(&curves[a], &curves[b]).abs());
        }
    }
    assert!(
        min_abs_corr < 0.9,
        "#700: sz per-level curves collapsed onto one shared shape (min |pearson|={min_abs_corr:.3})"
    );
}
