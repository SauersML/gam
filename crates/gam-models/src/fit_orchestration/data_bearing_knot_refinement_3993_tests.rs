//! #3993: the adaptive open B-spline refines along data-bearing intervals.
//!
//! A covariate stretched by one outlier (`x ~ U(0, 1)` plus `x = 1e6`) puts
//! every uniform knot in the data-free span, and uniform `2K + 1` refinement
//! needs about `log2(1e6) = 20` levels before the bulk gains a single knot.
//! [`BSplineKnotPlacement::UniformRefined`] keeps the uniform pilot grid and
//! splits, per level, only intervals whose observations a knot can separate,
//! at their coarsest separating dyadic point. These tests pin the chain's
//! three properties (uniform on covered data, nested in `K`, the bulk resolved
//! on the outlier design) and the issue's end-to-end reproducer.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotPlacement,
    BSplineKnotSpec, BasisMetadata, OneDimensionalBoundary, build_bspline_basis_1d,
    generate_full_knot_vector,
};
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const DEGREE: usize = 3;

fn outlier_covariate(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0_f64, 1.0).unwrap();
    let mut x: Vec<f64> = (0..n).map(|_| unif.sample(&mut rng)).collect();
    x[0] = 1.0e6;
    x
}

/// The internal knots the chain realizes for `num_internal_knots` from `root`.
fn chain_internal_knots(x: &[f64], root: usize, num_internal_knots: usize) -> Vec<f64> {
    let spec = BSplineBasisSpec {
        degree: DEGREE,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots,
            placement: BSplineKnotPlacement::UniformRefined { root },
            adaptive: false,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };
    let data = Array1::from_vec(x.to_vec());
    let built = build_bspline_basis_1d(data.view(), &spec)
        .unwrap_or_else(|e| panic!("chain at K={num_internal_knots} from root {root}: {e:?}"));
    let BasisMetadata::BSpline1D { knots, .. } = built.metadata else {
        panic!("expected BSpline1D metadata");
    };
    assert_eq!(knots.len(), num_internal_knots + 2 * (DEGREE + 1));
    knots.slice(ndarray::s![DEGREE + 1..knots.len() - DEGREE - 1]).to_vec()
}

#[test]
fn chain_is_the_uniform_refinement_on_covered_data_3993() {
    // Every interval of every level holds data on both sides of its midpoint,
    // so each level is exactly the uniform `2K + 1` grid (#3078's designs).
    let x: Vec<f64> = (0..=2000).map(|i| i as f64 / 2000.0).collect();
    for &k in &[15usize, 31, 63] {
        let chain = chain_internal_knots(&x, 7, k);
        let uniform = generate_full_knot_vector((0.0, 1.0), k, DEGREE).expect("uniform grid");
        let uniform = &uniform.as_slice().unwrap()[DEGREE + 1..uniform.len() - DEGREE - 1];
        for (a, b) in chain.iter().zip(uniform) {
            // Midpoint bisection and `min + i h` round differently; both are
            // within a few ulps of the exact dyadic point on [0, 1].
            assert!((a - b).abs() <= 4.0 * f64::EPSILON, "K={k}: {a} vs {b}");
        }
    }
    // At or below its root the chain is the uniform grid itself.
    let root = chain_internal_knots(&x, 7, 7);
    assert_eq!(root.len(), 7);
    for (i, knot) in root.iter().enumerate() {
        assert!((knot - (i + 1) as f64 / 8.0).abs() <= 4.0 * f64::EPSILON);
    }
}

#[test]
fn chain_is_nested_in_the_knot_count_3993() {
    let x = outlier_covariate(300, 3993);
    let root = 7;
    let mut previous = chain_internal_knots(&x, root, root);
    for k in root + 1..=60 {
        let next = chain_internal_knots(&x, root, k);
        assert!(
            previous.iter().all(|knot| next.contains(knot)),
            "the chain at K={k} must keep every knot of K={}",
            k - 1
        );
        previous = next;
    }
}

#[test]
fn chain_spends_its_refinement_on_the_bulk_3993() {
    let x = outlier_covariate(300, 3993);
    let root = 7;
    let root_knots = chain_internal_knots(&x, root, root);
    // Uniform on [min, 1e6]: every pilot knot sits in the data-free span.
    assert!(root_knots.iter().all(|&knot| knot > 1.0));
    // One refinement level (7 -> 15): the only intervals holding two
    // separable values are in the bulk, so every new knot lands in [0, 1].
    let refined = chain_internal_knots(&x, root, 15);
    let bulk = refined.iter().filter(|&&knot| knot < 1.0).count();
    assert_eq!(bulk, 15 - root, "refined knots: {refined:?}");
    // The chain reaches the support bound K = u - degree - 1.
    let mut distinct = x.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let support = distinct.len() - DEGREE - 1;
    let saturated = chain_internal_knots(&x, root, support);
    assert!(saturated.windows(2).all(|w| w[0] < w[1]));
}

fn fit_on_bulk_grid(formula: &str, x: &[f64], y: &[f64], grid: &[f64]) -> (Vec<f64>, f64) {
    let headers: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let rows = x
        .iter()
        .zip(y)
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map()["x"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &ds, &cfg)
        .unwrap_or_else(|e| panic!("{formula} must fit the outlier design: {e:?}"));
    let StandardFitResult {
        fit, resolvedspec, ..
    } = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit"),
    };
    let edf = fit.edf_total().expect("edf_total");
    let mut mat = Array2::<f64>::zeros((grid.len(), ds.headers.len()));
    for (row, &g) in grid.iter().enumerate() {
        mat[[row, col]] = g;
    }
    let design = build_term_collection_design(mat.view(), &resolvedspec).expect("grid design");
    let predicted: Array1<f64> = design.design.matrixvectormultiply(&fit.beta);
    (predicted.to_vec(), edf)
}

#[test]
fn default_smooth_resolves_the_bulk_behind_an_outlier_3993() {
    let n = 300usize;
    let sigma = 0.2f64;
    let x = outlier_covariate(n, 0);
    let mut rng = StdRng::seed_from_u64(39930);
    let noise = Normal::new(0.0, sigma).unwrap();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (6.0 * xi).sin() + noise.sample(&mut rng))
        .collect();
    let grid: Vec<f64> = (0..50).map(|i| 0.05 + 0.9 * i as f64 / 49.0).collect();
    let rmse = |predicted: &[f64]| {
        (predicted
            .iter()
            .zip(&grid)
            .map(|(p, g)| (p - (6.0 * g).sin()).powi(2))
            .sum::<f64>()
            / grid.len() as f64)
            .sqrt()
    };
    let (default_fit, edf_default) = fit_on_bulk_grid("y ~ s(x)", &x, &y, &grid);
    let (quantile_fit, edf_quantile) =
        fit_on_bulk_grid("y ~ s(x, knot_placement=quantile)", &x, &y, &grid);
    let (rmse_default, rmse_quantile) = (rmse(&default_fit), rmse(&quantile_fit));
    // Each resolved fit's error about the truth has the variance scale
    // sigma * sqrt(edf / n) (the mean of the smoother's pointwise variance
    // sigma^2 tr(A) / n), so by the triangle inequality the two RMSEs differ
    // by at most the sum of those scales when both resolve the bulk. A basis
    // with no knot in the bulk is flat there, off by the truth's own spread
    // (about 0.6), far outside this bound.
    let bound = rmse_quantile
        + sigma * ((edf_default / n as f64).sqrt() + (edf_quantile / n as f64).sqrt());
    eprintln!(
        "#3993 outlier design: default rmse={rmse_default:.4} (edf {edf_default:.2}), \
         quantile rmse={rmse_quantile:.4} (edf {edf_quantile:.2}), bound={bound:.4}"
    );
    assert!(
        rmse_default <= bound,
        "default s(x) rmse {rmse_default:.4} exceeds the quantile fit's {rmse_quantile:.4} \
         plus the variance scale (bound {bound:.4})"
    );
}
