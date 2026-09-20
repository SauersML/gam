#![cfg(test)]
use crate::basis::{BasisError, CenterStrategy, SpatialIdentifiability, ThinPlateBasisSpec};
use crate::smooth::name_thin_plate_outlier_span;
use ndarray::Array2;

fn spec(num_centers: usize) -> ThinPlateBasisSpec {
    ThinPlateBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers },
        periodic: None,
        length_scale: 1.0,
        identifiability: SpatialIdentifiability::OrthogonalToParametric,
        double_penalty: true,
        radial_reparam: None,
    }
}

/// A 1-D bulk on `[0, 1)` with its first row replaced by `outlier`, already
/// divided by the sample SD the way the isotropic frame standardizes it.
fn standardized_bulk_with_outlier(outlier: f64) -> Array2<f64> {
    let n = 300;
    let mut x: Vec<f64> = (0..n).map(|i| ((i * 37) % n) as f64 / n as f64).collect();
    x[0] = outlier;
    let mean = x.iter().sum::<f64>() / n as f64;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    Array2::from_shape_vec((n, 1), x.into_iter().map(|v| v / sd).collect()).expect("shape")
}

#[test]
fn a_far_outlier_that_flattens_the_bulk_is_refused_not_fit_linear() {
    let data = standardized_bulk_with_outlier(1.0e6);
    let err = super::build_thin_plate_basis(data.view(), &spec(10))
        .err()
        .expect("an outlier-dominated thin-plate basis must be refused");
    let BasisError::ThinPlateBulkUnresolvable {
        term,
        axis,
        bulk_fraction,
        resolvable_fraction,
        retained,
        available,
        spans,
    } = &err
    else {
        panic!("expected the typed thin-plate bulk refusal, got {err:?}");
    };
    assert_eq!((term, *axis, spans.len()), (&None, 0, 0));
    assert!(bulk_fraction <= resolvable_fraction, "{err}");
    assert!(retained < available, "{err}");
    assert!(err.advice().is_some_and(|a| a.contains("bs='cr'")));
}

#[test]
fn an_ordinary_spread_keeps_every_bending_direction() {
    let data = standardized_bulk_with_outlier(1.0);
    let built = super::build_thin_plate_basis(data.view(), &spec(10))
        .expect("an ordinary 1-D cloud builds a full thin-plate basis");
    // 10 centres minus the {1, x} null space are all bending modes, plus the
    // linear column that survives the intercept-orthogonal chart.
    assert_eq!(built.design.ncols(), 10 - 2 + 1);
}

#[test]
fn the_term_level_refusal_names_the_outlying_span_in_original_units() {
    let n = 9;
    let mut raw = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        raw[[i, 1]] = i as f64 / n as f64;
    }
    raw[[0, 1]] = 1.0e6;
    let basis_level = BasisError::ThinPlateBulkUnresolvable {
        term: None,
        axis: 0,
        bulk_fraction: 1.0e-7,
        resolvable_fraction: 1.0e-4,
        retained: 1,
        available: 8,
        spans: Vec::new(),
    };
    let err = name_thin_plate_outlier_span(basis_level, "s(x)", raw.view(), &[1]);
    let BasisError::ThinPlateBulkUnresolvable {
        term,
        spans,
        retained,
        available,
        ..
    } = &err
    else {
        panic!("expected the typed refusal, got {err:?}");
    };
    assert_eq!(term.as_deref(), Some("s(x)"));
    assert_eq!((*retained, *available), (1, 8));
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].column, 1);
    assert_eq!(spans[0].max, 1.0e6);
    let msg = err.to_string();
    assert!(msg.contains("thin-plate smooth 's(x)'"), "{msg}");
    assert!(msg.contains("spans [1.111111e-1, 1.000000e6]"), "{msg}");
    assert!(
        msg.contains("middle half spans [3.333333e-1, 7.777778e-1]"),
        "{msg}"
    );
    assert!(msg.contains("only 1 of 8 bending directions"), "{msg}");
    assert!(msg.contains("bs='cr'"), "{msg}");

    let other = name_thin_plate_outlier_span(
        BasisError::InvalidInput("unrelated".to_string()),
        "s(x)",
        raw.view(),
        &[1],
    );
    assert!(matches!(other, BasisError::InvalidInput(m) if m == "unrelated"));
}

#[test]
fn losing_only_the_finest_directions_of_a_resolvable_bulk_is_not_refused() {
    // A moderate outlier and a dense 200-centre basis both drop their finest
    // bending directions below the rank floor, yet the bulk's middle half is
    // wider than the finest surviving scale, so both still bend across it.
    for (outlier, num_centers) in [(1.0e2, 10), (1.0, 200)] {
        let n = 2000;
        let mut x: Vec<f64> = (0..n).map(|i| ((i * 37) % n) as f64 / n as f64).collect();
        x[0] = outlier;
        let data = Array2::from_shape_vec((n, 1), x).expect("shape");
        let built = super::build_thin_plate_basis(data.view(), &spec(num_centers))
            .unwrap_or_else(|e| panic!("outlier {outlier}, k={num_centers}: {e}"));
        assert!(
            built.design.ncols() > 2,
            "outlier {outlier}, k={num_centers}"
        );
    }
}

#[test]
fn a_tied_bulk_has_no_width_to_resolve_and_is_not_refused() {
    // Most rows at 0: the middle half has zero width, so there is nothing
    // inside it for a bending direction to resolve.
    let n = 300;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            if i % 4 == 0 {
                1.0e6 * i as f64 / n as f64
            } else {
                0.0
            }
        })
        .collect();
    let data = Array2::from_shape_vec((n, 1), x).expect("shape");
    super::build_thin_plate_basis(data.view(), &spec(10)).expect("tied bulk builds");
}
