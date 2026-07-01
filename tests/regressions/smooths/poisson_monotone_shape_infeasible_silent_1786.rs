//! Regression (#1786): a `monotone_increasing` (or convex/concave) shape
//! constraint must NEVER be silently violated under a non-canonical (log-link)
//! family.
//!
//! Repro from the issue: low-count Poisson (log link) `s(x)` with
//! `shape=monotone_increasing`. Because the base rate is low, many zero counts
//! collapse the log-link IRLS working weights `W ≈ μ` toward 0, which can drive
//! the constrained Newton active-set inner solve away from a stationary point.
//! The failure mode the issue reports is that the outer's keep-best /
//! best-iterate substitution path then ships the last (infeasible) β to the
//! caller as a SUCCESSFUL `Model` whose point predictions actually DECREASE — no
//! error surfaced, only a non-blocking warning.
//!
//! Contract: a returned `monotone_increasing` model MUST have non-decreasing
//! predictions, family-independent — OR `fit` must FAIL rather than silently
//! return an infeasible model. This test enforces exactly that: the Poisson
//! monotone fit must either (a) return predictions that are actually
//! non-decreasing, or (b) surface a clear error. It must NEVER be a silent
//! infeasible success.
//!
//! The load-bearing protection is the post-fit feasibility audit
//! `enforce_term_constraint_feasibility` (in
//! `fit_orchestration/drivers/design_construction.rs`), invoked unconditionally
//! on every standard fit path: it re-checks the fitted coefficient block against
//! each smooth term's shape-constraint rows (the box `γ_i ≥ 0` lower bounds for
//! reparameterized monotone/convex B-splines, or the `A·β ≥ b` grid rows for the
//! non-box path) and returns `ParameterConstraintViolation` if the returned β
//! violates them beyond a small tolerance — so a keep-best / best-iterate β that
//! the constrained inner solve could not certify is surfaced as an error rather
//! than shipped as a silent infeasible `Model`. This test is the family-
//! independent regression lock on that guarantee (the audit previously had no
//! test asserting the feasible-or-error contract for a non-canonical family).
//!
//! The Gaussian control on the identical integer response (identity link) is
//! kept as a regression anchor: the shape-constraint machinery itself is
//! correct, so the Gaussian fit must honor the constraint exactly.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Deterministic SplitMix64 → U(0,1).
fn splitmix_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state: u64 = seed;
    move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Poisson(lambda) via Knuth's product-of-uniforms method, drawing uniforms from
/// the supplied stream. Deterministic given a deterministic stream.
fn poisson_knuth(lambda: f64, next_unit: &mut impl FnMut() -> f64) -> f64 {
    let l = (-lambda).exp();
    let mut k = 0.0_f64;
    let mut p = 1.0_f64;
    loop {
        k += 1.0;
        p *= next_unit().max(1e-300);
        if p <= l {
            break;
        }
    }
    k - 1.0
}

/// The issue's DGP: n=200, x=linspace(0,1), y ~ Poisson(exp(-1 + 2x)), seed 0.
fn make_low_count_poisson_data() -> (Vec<f64>, Vec<f64>) {
    let n = 200usize;
    let mut next_unit = splitmix_stream(0);
    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let xi = i as f64 / (n as f64 - 1.0);
        let lambda = (-1.0 + 2.0 * xi).exp();
        x[i] = xi;
        y[i] = poisson_knuth(lambda, &mut next_unit);
    }
    (x, y)
}

/// A harder, constraint-BINDING low-count regime: a Poisson hump
/// `y ~ Poisson(exp(-3 + 3·sin(pi·x)))` (rises then falls) under a
/// `monotone_increasing` constraint. The unconstrained log-link fit is
/// non-monotone, so the constraint genuinely binds on the falling half — the
/// stress case for the constrained P-IRLS inner solve under the collapsed
/// low-count working weights that #1786 identifies. The base rate is low so many
/// zeros drive the ill-conditioning.
fn make_binding_hump_poisson_data() -> (Vec<f64>, Vec<f64>) {
    let n = 120usize;
    let mut next_unit = splitmix_stream(0);
    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let xi = i as f64 / (n as f64 - 1.0);
        let lambda = (-3.0 + 3.0 * (std::f64::consts::PI * xi).sin()).exp();
        x[i] = xi;
        y[i] = poisson_knuth(lambda, &mut next_unit);
    }
    (x, y)
}

/// Assert the returned/failed fit honors the #1786 contract: a returned Model's
/// predictions on a dense grid must be non-decreasing, OR `fit` must surface a
/// clear error — never a silent infeasible success.
fn assert_monotone_or_error(outcome: Result<(FitResult, usize, usize), String>) {
    match outcome {
        Ok((FitResult::Standard(fit), n_headers, x_idx)) => {
            // A returned model MUST honor the constraint: predictions on a dense
            // grid must be non-decreasing (the log link is monotone, so eta
            // non-decreasing ⇔ mu non-decreasing). The bug was that this model
            // was shipped with a genuinely non-monotone eta (a SILENT infeasible
            // success), which the contract forbids.
            let n_grid = 600usize;
            let eta =
                predict_eta_on_grid(&fit.resolvedspec, &fit.fit.beta, n_headers, x_idx, n_grid);
            assert!(
                eta.iter().all(|v| v.is_finite()),
                "poisson monotone prediction must be finite"
            );
            let range = {
                let lo = eta.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                hi - lo
            };
            let step_tol = 1e-6 * range.max(1.0);
            let mut worst_drop = 0.0_f64;
            let mut n_drops = 0usize;
            for w in eta.windows(2) {
                let drop = w[0] - w[1];
                if drop > step_tol {
                    n_drops += 1;
                    worst_drop = worst_drop.max(drop);
                }
            }
            assert_eq!(
                n_drops, 0,
                "SILENT INFEASIBLE MODEL: poisson monotone_increasing fit returned a Model \
                 whose predictions DECREASE at {n_drops}/{} grid steps (worst drop {worst_drop:.3e}, \
                 range {range:.3e}). The contract requires either non-decreasing predictions or a \
                 surfaced error — never a silent infeasible success.",
                eta.len() - 1
            );
        }
        Ok((_, _, _)) => panic!("expected a Standard GAM fit for poisson s(x)"),
        Err(e) => {
            // Acceptable outcome (b): the constrained solve could not certify a
            // feasible monotone optimum, so `fit` surfaced a clear error rather
            // than silently returning an infeasible model. This satisfies the
            // contract. Assert the error is non-empty so it is genuinely
            // actionable.
            assert!(
                !e.is_empty(),
                "poisson monotone fit failed but surfaced an empty error message"
            );
        }
    }
}

/// Build a temp CSV, fit `formula` under `cfg`, and return the fit result plus
/// the dataset header width and the covariate column index (so predictions can
/// be reconstructed on a grid). The path is unique per call so parallel #[test]
/// threads never collide.
fn fit_formula(
    formula: &str,
    x: &[f64],
    y: &[f64],
    cfg: &FitConfig,
) -> Result<(FitResult, usize, usize), String> {
    let n = x.len();
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let unique = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_poisson_mono_1786_{}_{}_{}.csv",
        std::process::id(),
        n,
        unique
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic poisson data");
    std::fs::remove_file(&tmp).ok();
    let n_headers = ds.headers.len();
    let x_idx = ds.column_map()["x"];
    let result = fit_from_formula(formula, &ds, cfg).map_err(|e| e.to_string())?;
    Ok((result, n_headers, x_idx))
}

/// Predict the linear predictor on a dense grid spanning [0, 1].
fn predict_eta_on_grid(
    resolvedspec: &gam::smooth::TermCollectionSpec,
    beta: &ndarray::Array1<f64>,
    n_headers: usize,
    x_idx: usize,
    n_grid: usize,
) -> Vec<f64> {
    let mut grid = Array2::<f64>::zeros((n_grid, n_headers));
    for j in 0..n_grid {
        grid[[j, x_idx]] = j as f64 / (n_grid as f64 - 1.0);
    }
    let design = build_term_collection_design(grid.view(), resolvedspec)
        .expect("rebuild 1-D smooth design on evaluation grid");
    design.design.apply(beta).to_vec()
}

#[test]
fn poisson_low_count_monotone_increasing_is_feasible_or_errors_1786() {
    init_parallelism();

    // The issue's exact repro: n=200, x=linspace(0,1), y ~ Poisson(exp(-1+2x)).
    let (x, y) = make_low_count_poisson_data();
    // Sanity: this really is a low-count regime with many zeros (the driver of
    // the collapsed IRLS working weights).
    let zeros = y.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros >= 20,
        "expected a genuinely low-count regime with many zeros; got {zeros} zeros of {}",
        y.len()
    );

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };

    // Contract: non-decreasing predictions OR a surfaced error — never a silent
    // infeasible success.
    assert_monotone_or_error(fit_formula(
        "y ~ s(x, shape=monotone_increasing)",
        &x,
        &y,
        &cfg,
    ));
}

#[test]
fn poisson_binding_hump_monotone_increasing_is_feasible_or_errors_1786() {
    init_parallelism();

    // The stress face: a low-count Poisson HUMP where the monotone_increasing
    // constraint genuinely binds on the falling half. This is the regime that
    // drives the constrained P-IRLS inner solve into the ill-conditioned
    // collapsed-working-weight corner #1786 identifies. The contract must still
    // hold: feasible monotone predictions, or a clear error.
    let (x, y) = make_binding_hump_poisson_data();
    let zeros = y.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros >= 20,
        "expected a low-count binding regime with many zeros; got {zeros} zeros of {}",
        y.len()
    );

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };

    assert_monotone_or_error(fit_formula(
        "y ~ s(x, k=15, shape=monotone_increasing)",
        &x,
        &y,
        &cfg,
    ));
}

#[test]
fn gaussian_control_low_count_monotone_increasing_honors_constraint_1786() {
    init_parallelism();

    // Regression anchor: the identical integer response under Gaussian /
    // identity link must honor the constraint exactly. The shape-constraint
    // machinery is correct; only the log-link inner solve was defective.
    let (x, y) = make_low_count_poisson_data();

    let cfg = FitConfig::default(); // gaussian / identity / REML
    let (result, n_headers, x_idx) =
        fit_formula("y ~ s(x, shape=monotone_increasing)", &x, &y, &cfg)
            .expect("gaussian monotone fit must succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected a Standard GAM fit for gaussian s(x)");
    };

    let n_grid = 600usize;
    let eta = predict_eta_on_grid(&fit.resolvedspec, &fit.fit.beta, n_headers, x_idx, n_grid);
    assert!(
        eta.iter().all(|v| v.is_finite()),
        "gaussian monotone prediction must be finite"
    );
    let range = {
        let lo = eta.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let step_tol = 1e-6 * range.max(1.0);
    for w in eta.windows(2) {
        assert!(
            w[1] - w[0] >= -step_tol,
            "gaussian monotone_increasing control must be non-decreasing: dropped by {:.3e}",
            w[0] - w[1]
        );
    }
}
