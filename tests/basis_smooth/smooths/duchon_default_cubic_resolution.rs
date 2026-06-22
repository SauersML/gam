//! Default ("magic") resolution of the redesigned Euclidean Duchon smoother,
//! plus a fractional-power fit→predict round-trip.
//!
//! CONTRACT. With no `power=`/`order=` override, `duchon(...)` resolves to the
//! STRUCTURAL CUBIC smoother in every dimension `d`:
//!
//!   * null-space order = `Linear` (affine null space, polynomial degree 1),
//!   * spectral power    `s = (d − 1) / 2`  (an `f64`, possibly fractional),
//!   * kernel radial exponent `2(p + s) − d = 3`, i.e. the cubic `r³` kernel,
//!     where `p = 2` for the `Linear` null space.
//!
//! There is NO escalation to a quadratic (`Degree(2)`) null space for any `d` —
//! the structural cubic kernel is well-defined in every dimension, so the
//! resolver must hand back `Linear` unchanged. This is the property the old
//! escalating `resolve_duchon_orders` path violated (it raised the null-space
//! order whenever pure-mode CPD demanded a richer polynomial absorption space).
//!
//! We probe the resolution through the public `materialize` entry point, which
//! parses the formula and resolves every term's basis spec WITHOUT fitting, so
//! the test is cheap and isolates the resolution logic. The fractional-power
//! round-trip then fits a real `d=2` model at `s = 0.5` and predicts, guarding
//! the `power: f64` metadata change end to end.

use gam::basis::DuchonNullspaceOrder;
use gam::matrix::LinearOperator;
use gam::smooth::SmoothBasisSpec;
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism, materialize,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Build an encoded dataset with a response `y` and `d` continuous predictors
/// `x0..x{d-1}` drawn uniformly from `[-1, 1]`, `y` a smooth-ish target plus
/// light noise. `n` rows.
fn encoded_dataset(n: usize, d: usize, seed: u64) -> gam::inference::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let xdist = Uniform::new(-1.0_f64, 1.0).expect("uniform valid");
    let noise = Normal::new(0.0, 0.05).expect("normal valid");

    let mut headers = vec!["y".to_string()];
    for j in 0..d {
        headers.push(format!("x{j}"));
    }

    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let xs: Vec<f64> = (0..d).map(|_| xdist.sample(&mut rng)).collect();
        // A smooth target: sum of low-frequency sinusoids per axis.
        let mut y = 0.0;
        for (j, &xj) in xs.iter().enumerate() {
            y += (0.6 + 0.1 * j as f64) * (std::f64::consts::PI * xj).sin();
        }
        y += noise.sample(&mut rng);
        let mut fields = vec![y.to_string()];
        fields.extend(xs.iter().map(|v| v.to_string()));
        records.push(csv::StringRecord::from(fields));
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

/// Pull the (nullspace_order, power) of the first (and only) Duchon smooth term
/// from a resolved formula, without fitting.
fn resolved_duchon(
    formula: &str,
    ds: &gam::inference::data::EncodedDataset,
) -> (DuchonNullspaceOrder, f64) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize(formula, ds, &cfg).expect("formula materializes");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected a standard fit request for a gaussian Duchon smooth");
    };
    let term = request
        .spec
        .smooth_terms
        .iter()
        .find_map(|t| match &t.basis {
            SmoothBasisSpec::Duchon { spec, .. } => Some(spec.clone()),
            _ => None,
        })
        .expect("a resolved Duchon smooth term");
    (term.nullspace_order, term.power)
}

// ── (c) default cubic resolution for d in {1, 2, 3, 8} ──────────────────────

#[test]
fn default_duchon_resolves_to_cubic_for_each_dimension() {
    init_parallelism();

    for &d in &[1usize, 2, 3, 8] {
        // Need centers/k > polynomial-nullspace columns (= d + 1 for Linear);
        // n large enough to place them. The default `duchon(...)` uses the
        // magic center count, so just supply enough data.
        let n = 64 + 24 * d;
        let ds = encoded_dataset(n, d, 1000 + d as u64);

        let cols: Vec<String> = (0..d).map(|j| format!("x{j}")).collect();
        let formula = format!("y ~ duchon({})", cols.join(", "));

        let (nullspace, power) = resolved_duchon(&formula, &ds);

        // Affine null space, no quadratic escalation.
        assert_eq!(
            nullspace,
            DuchonNullspaceOrder::Linear,
            "d={d}: default Duchon must resolve to Linear (affine) null space, \
             not escalate to a quadratic null space; got {nullspace:?}"
        );

        // Spectral power s = (d - 1)/2 exactly (as f64).
        let expected_power = (d as f64 - 1.0) / 2.0;
        assert!(
            (power - expected_power).abs() < 1e-12,
            "d={d}: default Duchon power must be (d-1)/2 = {expected_power}, got {power}"
        );

        // Derived kernel radial exponent: 2(p + s) - d with p = 2 (Linear).
        // = 2*(2 + (d-1)/2) - d = 4 + (d-1) - d = 3, the cubic r^3 kernel.
        let p = 2.0_f64; // Linear null space => m = p = 2.
        let kernel_exponent = 2.0 * (p + power) - d as f64;
        assert!(
            (kernel_exponent - 3.0).abs() < 1e-12,
            "d={d}: default Duchon kernel exponent must be 3 (cubic r^3), got {kernel_exponent}"
        );
    }
}

// ── (d) fractional-power fit→predict round-trip (d=2, power=0.5) ────────────

/// A `d=2` Duchon fit with an explicit fractional power `s = 0.5` must fit and
/// predict finite, non-degenerate values. This guards the `power: f64` change:
/// the whole pipeline (spec → basis → penalties → fit → predict) has to carry a
/// non-integer power without rounding it to an integer or panicking.
#[test]
fn fractional_power_duchon_fit_predict_round_trip_d2() {
    init_parallelism();

    let n = 200usize;
    let ds = encoded_dataset(n, 2, 77);

    // Confirm the explicit fractional power survives resolution as an exact f64.
    let (nullspace, power) = resolved_duchon("y ~ duchon(x0, x1, power=0.5)", &ds);
    assert_eq!(
        nullspace,
        DuchonNullspaceOrder::Linear,
        "explicit power keeps the requested (default Linear) null space; got {nullspace:?}"
    );
    assert!(
        (power - 0.5).abs() < 1e-12,
        "fractional power=0.5 must thread through as an exact f64, got {power}"
    );

    // Fit the model and predict on the training rows; everything must be finite.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x0, x1, power=0.5)", &ds, &cfg)
        .expect("fractional-power Duchon fit succeeds");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian fractional-power Duchon smooth");
    };

    // All fitted coefficients finite.
    assert!(
        fit.fit.beta.iter().all(|v| v.is_finite()),
        "fractional-power Duchon produced non-finite coefficients"
    );

    // Predictions at the training design must be finite and not collapse to a
    // single constant (which would mean the smooth carried no signal).
    let col = ds.column_map();
    let (x0, x1) = (col["x0"], col["x1"]);
    let m = 50usize;
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for i in 0..m {
        let t = -0.9 + 1.8 * i as f64 / (m as f64 - 1.0);
        grid[[i, x0]] = t;
        grid[[i, x1]] = -t;
    }
    let design = gam::smooth::build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild fractional-power Duchon design at grid");
    let preds = design.design.apply(&fit.fit.beta);
    assert!(
        preds.iter().all(|v| v.is_finite()),
        "fractional-power Duchon predictions contain non-finite values"
    );
    let pmin = preds.iter().cloned().fold(f64::INFINITY, f64::min);
    let pmax = preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (pmax - pmin).abs() > 1e-6,
        "fractional-power Duchon prediction is degenerate (constant {pmin:.3e}); \
         the smooth carried no signal across the grid"
    );
}
