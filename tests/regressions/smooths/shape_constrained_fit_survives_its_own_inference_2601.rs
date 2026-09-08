//! Regression for #2601: a shape-constrained fit must survive its own
//! inference layer.
//!
//! `gamfit.fit(frame, "y ~ s(x)", constraints={"s(x)": kind})` on 300 rows of
//! clean increasing linear data — the fixture `#376`'s guard
//! `test_gaussian_reml_fit_all_shape_constraints_do_not_panic` uses — failed on
//! three of its four constraint kinds, for three unrelated reasons. Two of them
//! killed a fit whose own inner solves had all converged:
//!
//! * `[concave]` — the fit produced a model that could not be READ BACK.
//!   `ConstrainedPosteriorCorrection::normal_upper_limits` carries
//!   `f64::INFINITY` for every half-line coordinate (the common case for a
//!   one-sided shape constraint); `serde_json` writes `+inf` as `null` and
//!   `Vec<f64>` then refuses its own output with `invalid type: null, expected
//!   f64`, at load, far from the fit that produced it.
//!
//! * `[convex]` — the SIGMA-POINT CUBATURE leg of the smoothing correction,
//!   an off-trajectory refinement of the reported UNCERTAINTY, failed to
//!   converge its own inner solve and propagated, aborting the fit with
//!   `KKT residuals exceed tolerance ... stat_rel=1.000e0`.
//!
//! * `[monotone_decreasing]` — the constrained-posterior moment integral
//!   refused: `truncated moments for an 11-dimensional constraint face did not
//!   converge`. That one could not be degraded around, because the correction
//!   supplies the REPORTED COEFFICIENTS, not just their covariance.
//!
//! The assertion is deliberately end-to-end and deliberately about all four
//! kinds together: each mechanism sits in a different layer, and only the whole
//! path exercises them at once. A returned model must also honour the shape it
//! was asked for — avoiding the abort is necessary and not sufficient.

use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::io::Write;

/// 300 rows of `y = 2x + N(0, 0.1)` with `x` sorted uniform on `[0, 1]`.
///
/// Deterministic without depending on any particular RNG stream: the fixture's
/// content is "clean, well-conditioned, increasing linear data", which is what
/// makes `monotone_decreasing` / `convex` / `concave` all bind.
fn linear_fixture() -> (Vec<f64>, Vec<f64>) {
    let n = 300usize;
    let mut state: u64 = 0x2601_0000_0000_0005;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite covariate"));
    let y: Vec<f64> = x
        .iter()
        .map(|xi| {
            let u1 = next().max(1e-12);
            let u2 = next();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            2.0 * xi + 0.1 * z
        })
        .collect();
    (x, y)
}

#[test]
fn every_shape_constraint_fits_clean_linear_data_2601() {
    gam_runtime::test_support::install_diagnostic_logger();
    init_parallelism();
    let (x, y) = linear_fixture();
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("gam_2601_shapes_{}.csv", std::process::id()));
    {
        let mut file = std::fs::File::create(&tmp).expect("create fixture csv");
        file.write_all(csv.as_bytes()).expect("write fixture csv");
    }
    let dataset = load_csvwith_inferred_schema(&tmp).expect("load fixture");
    std::fs::remove_file(&tmp).expect("remove the fixture csv this test created");

    let mut failures: Vec<String> = Vec::new();
    for kind in [
        "monotone_increasing",
        "monotone_decreasing",
        "convex",
        "concave",
    ] {
        let formula = format!("y ~ s(x, shape={kind})");
        let fitted = match fit_from_formula(&formula, &dataset, &FitConfig::default()) {
            Ok(FitResult::Standard(fit)) => fit,
            Ok(_) => {
                failures.push(format!("{kind}: a 1-D gaussian smooth is not a Standard fit"));
                continue;
            }
            Err(error) => {
                failures.push(format!("{kind}: {error}"));
                continue;
            }
        };

        // Evaluate on a UNIFORM grid. Convexity is a property of the function,
        // and the plain second difference only certifies it when the abscissae
        // are evenly spaced — on the raw sorted-uniform sample the gap ratio is
        // ~1e5 and a genuinely convex function has negative plain second
        // differences across such triples.
        let n_grid = 201usize;
        let x_index = dataset.column_map()["x"];
        let mut grid = Array2::<f64>::zeros((n_grid, dataset.headers.len()));
        for j in 0..n_grid {
            grid[[j, x_index]] = j as f64 / (n_grid as f64 - 1.0);
        }
        let design = build_term_collection_design(grid.view(), &fitted.resolvedspec)
            .expect("rebuild the smooth design on the evaluation grid");
        let curve = {
            use gam::matrix::LinearOperator;
            design.design.apply(&fitted.fit.beta).to_vec()
        };
        if !curve.iter().all(|v| v.is_finite()) {
            failures.push(format!("{kind}: fitted curve has non-finite entries"));
            continue;
        }

        let scale = curve.iter().fold(0.0f64, |acc, v| acc.max(v.abs())) + 1.0;
        let tol = 1e-5 * scale;
        let first: Vec<f64> = curve.windows(2).map(|w| w[1] - w[0]).collect();
        let violation = match kind {
            "monotone_increasing" => first.iter().cloned().fold(f64::INFINITY, f64::min) < -tol,
            "monotone_decreasing" => {
                first.iter().cloned().fold(f64::NEG_INFINITY, f64::max) > tol
            }
            "convex" => first
                .windows(2)
                .map(|w| w[1] - w[0])
                .fold(f64::INFINITY, f64::min)
                < -tol,
            _ => {
                first
                    .windows(2)
                    .map(|w| w[1] - w[0])
                    .fold(f64::NEG_INFINITY, f64::max)
                    > tol
            }
        };
        if violation {
            failures.push(format!(
                "{kind}: the returned model does not honour the shape it was asked for"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "shape-constrained fits failed on clean linear data (#2601):\n  {}",
        failures.join("\n  ")
    );
}
