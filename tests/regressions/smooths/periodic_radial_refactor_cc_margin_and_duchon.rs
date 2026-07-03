//! Regression for #1776: the periodic-radial refactor (squash-merge c8c3192fa,
//! "feat(#580): periodic period derivation for radial builders") and the
//! follow-up build repair must keep BOTH periodic-smooth paths it touched alive.
//!
//! ## Background — what broke and why this test exists
//!
//! #1776 was filed as a *build* break: the refactor replaced the cyclic tensor
//! margin's data-range fallback with an unconditional `period=` requirement and
//! left the `margin_is_cc` local binding orphaned, which `[lints.rust]
//! warnings = "deny"` turned into a hard `-D unused-variables` compile error that
//! aborted `cargo build`, `cargo test`, and the `gamfit` wheel. The sanctioned
//! fix (8a8c96d83) was NOT to delete the binding but to *wire it back* so a
//! `cc`/`cp`/`cyclic` tensor margin with no explicit `period=` once again wraps
//! on the covariate's observed `[min, max]` span (the #1752 behaviour), mirroring
//! the 1-D `s(x, bs='cc')` cyclic fallback.
//!
//! The repro test the issue cited (`bug_hunt_periodic_radial_refactor_dead_
//! binding_breaks_build.rs`) never reached `main`, so this file restores
//! regression coverage and pins the *behaviour* the binding drives — from two
//! independent angles so a future refactor cannot silently regress either:
//!
//!   1. **Tensor cyclic margin, no `period=`** — `te(x, z, bs=c('cc','cc'))`.
//!      This is the exact code path `margin_is_cc` guards
//!      (`term_builder.rs`, `None if margin_is_cc => …`). If a refactor reverts
//!      to "periodic tensor margins *require* an explicit period" (the tempting
//!      but wrong fix the issue explicitly warns against — just removing the
//!      binding), this fit hard-errors and the test fails. The fitted surface
//!      must also genuinely wrap on each margin's data range.
//!   2. **1-D periodic radial Duchon** — `y ~ duchon(x, periodic=true)`. This is
//!      the very feature c8c3192fa shipped: a boolean `periodic=` on the radial
//!      builders that derives the wrap period from the closed center lattice
//!      (gam#580). The fit must succeed and produce a finite, periodic curve.
//!
//! Either assertion failing means the periodic-radial surface the breaking
//! commit introduced has regressed (or the build-breaking dead binding has been
//! re-introduced under a different guise).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TWO_PI: f64 = std::f64::consts::TAU;

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// Doubly-periodic truth on `[0, 1)^2` (period 1 on both axes).
fn doubly_periodic_surface(x: f64, z: f64) -> f64 {
    (TWO_PI * x).sin() * (TWO_PI * z).cos() - 0.3 * (TWO_PI * x).cos()
}

fn encode_xzy(rows: &[(f64, f64, f64)]) -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|(x, z, y)| StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

/// Angle 1 — `te(x, z, bs=c('cc','cc'))` with NO `period=`: the cyclic margins
/// must wrap on their own observed data range (mgcv `bs="cc"` semantics), not
/// hard-error demanding an explicit period. This is the `margin_is_cc` path.
#[test]
fn tensor_cc_margins_without_period_wrap_on_data_range() {
    init_parallelism();

    // Half-open `[0, 1)` grids on BOTH axes, so each margin's data span is the
    // natural wrap period (`bs='cc'` derives period = max - min).
    let n_x = 24usize;
    let n_z = 8usize;
    let mut training = Vec::with_capacity(n_x * n_z);
    for i in 0..n_x {
        let x = (i as f64) / (n_x as f64);
        for j in 0..n_z {
            let z = (j as f64) / (n_z as f64);
            training.push((x, z, doubly_periodic_surface(x, z)));
        }
    }
    let data = encode_xzy(&training);

    // NO `period=` here on purpose: the cc/cc margins must derive the wrap from
    // the observed [min, max] span. With the dead-binding regression present
    // (periodic margins require an explicit period) this `fit_from_formula`
    // returns Err and the `.expect` below fails — catching the exact regression
    // from a behavioural angle the compiler alone would miss.
    let result = fit_from_formula("y ~ te(x, z, bs=c('cc','cc'))", &data, &gaussian_cfg())
        .expect("te(x,z,bs=c('cc','cc')) with no period must fit by wrapping on the data range");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian cyclic tensor smooth");
    };

    let edf = fit.fit.edf_total().expect("gam reports total edf");
    assert!(
        edf.is_finite() && edf > 0.0 && edf < (n_x * n_z) as f64,
        "cyclic-margin tensor fit EDF must be finite and in (0, n); got {edf}"
    );

    // Each margin's data span (the derived wrap period). `[0,1)` grid → span is
    // `1 - 1/n`, with the endpoint at `max` wrapping back onto `min`.
    let x_min = 0.0;
    let x_max = (n_x - 1) as f64 / n_x as f64;
    let z_min = 0.0;
    let z_max = (n_z - 1) as f64 / n_z as f64;

    // Probe points spread across the interior; the wrap check compares the
    // fitted surface at one margin's endpoints with the other margin held fixed.
    let z_probes = [0.1, 0.4, 0.7];
    let x_probes = [0.15, 0.5, 0.85];

    // Build the probe design at (x_min,z) and (x_max,z) and compare: axis-0 wrap.
    let mut pts = Vec::new();
    for &z in &z_probes {
        pts.push((x_min, z));
        pts.push((x_max, z));
    }
    for &x in &x_probes {
        pts.push((x, z_min));
        pts.push((x, z_max));
    }
    let mut design_pts = Array2::<f64>::zeros((pts.len(), 2));
    for (r, &(x, z)) in pts.iter().enumerate() {
        design_pts[[r, 0]] = x;
        design_pts[[r, 1]] = z;
    }
    let design = build_term_collection_design(design_pts.view(), &fit.resolvedspec)
        .expect("rebuild cyclic-margin tensor design at wrap endpoints");
    let fitted = design.design.apply(&fit.fit.beta).to_vec();

    // The fitted surface must be non-trivial, else the wrap test is vacuous.
    let lo = fitted.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = fitted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        hi - lo > 0.1,
        "fitted cyclic surface is flat (ptp={:.3e}); wrap assertion would be vacuous",
        hi - lo
    );

    // Axis-0 wrap: fitted(x_min, z) == fitted(x_max, z) for each probed z.
    for (k, &z) in z_probes.iter().enumerate() {
        let at_min = fitted[2 * k];
        let at_max = fitted[2 * k + 1];
        let gap = (at_min - at_max).abs();
        assert!(
            gap < 1e-6,
            "axis-0 cc margin must wrap on its data range: |s({x_min}, {z}) - s({x_max}, {z})| \
             = {gap:.3e} (>= 1e-6)"
        );
    }
    // Axis-1 wrap: fitted(x, z_min) == fitted(x, z_max) for each probed x.
    let off = 2 * z_probes.len();
    for (k, &x) in x_probes.iter().enumerate() {
        let at_min = fitted[off + 2 * k];
        let at_max = fitted[off + 2 * k + 1];
        let gap = (at_min - at_max).abs();
        assert!(
            gap < 1e-6,
            "axis-1 cc margin must wrap on its data range: |s({x}, {z_min}) - s({x}, {z_max})| \
             = {gap:.3e} (>= 1e-6)"
        );
    }
}

/// Angle 2 — 1-D periodic radial Duchon via the formula API, the feature the
/// breaking commit c8c3192fa shipped. `y ~ duchon(x, periodic=true)` must fit
/// (deriving the wrap period from the center lattice, gam#580) and produce a
/// finite, genuinely periodic curve.
#[test]
fn periodic_radial_duchon_1d_fits_and_wraps() {
    init_parallelism();

    let n = 160usize;
    // Half-open `[0, 1)` grid; a clean two-harmonic periodic truth.
    let mut training = Vec::with_capacity(n);
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    for &x in &xs {
        let y = (TWO_PI * x).sin() + 0.4 * (2.0 * TWO_PI * x).cos();
        // z is an unused filler column so we can reuse the 3-column encoder;
        // keep it constant so it never enters the single-term formula.
        training.push((x, 0.0, y));
    }
    let data = encode_xzy(&training);

    let result = fit_from_formula("y ~ duchon(x, periodic=true)", &data, &gaussian_cfg())
        .expect("y ~ duchon(x, periodic=true) must fit (period derived from the center lattice)");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a 1-D periodic Duchon smooth");
    };

    // The whole coefficient vector must be finite — the issue's core assertion.
    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "periodic Duchon coefficient vector must be finite"
    );
    let edf = fit.fit.edf_total().expect("gam reports total edf");
    assert!(
        edf.is_finite() && edf > 0.0 && edf < n as f64,
        "periodic Duchon fit EDF must be finite and in (0, n); got {edf}"
    );

    // Periodic-wrap guarantee on the converged fit: fitted(min) == fitted(max),
    // the defining property of a cyclic radial basis.
    let x_min = 0.0;
    let x_max = (n - 1) as f64 / n as f64;
    let probe = [x_min, x_max];
    let mut design_pts = Array2::<f64>::zeros((probe.len(), 2));
    for (i, &x) in probe.iter().enumerate() {
        design_pts[[i, 0]] = x;
    }
    let design = build_term_collection_design(design_pts.view(), &fit.resolvedspec)
        .expect("rebuild 1-D periodic Duchon design at wrap endpoints");
    let fitted = design.design.apply(&fit.fit.beta).to_vec();
    let span = fitted.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let gap = (fitted[0] - fitted[1]).abs();
    assert!(
        gap < 1e-2 * span.max(1.0),
        "1-D periodic Duchon must wrap at the seam: |s({x_min}) - s({x_max})| = {gap:.3e} \
         (span={span:.3e})"
    );
}
