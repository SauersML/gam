//! Fast owning-crate acceptance coverage for #2297.
//!
//! A non-zero endpoint pin is an affine constraint, so a B-spline smooth is
//! represented as
//!
//!     f(x) = B_raw(x) beta_p + (B_raw(x) Z) gamma.
//!
//! The fixed particular solution `beta_p` belongs in the row offset while the
//! fitted coefficient chart remains the homogeneous null space `Z`.  These
//! tests keep the four end-to-end contracts in one leaf integration binary so
//! they do not require linking the workspace root's monolithic test harness.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::matrix::LinearOperator;
use gam_models::fit_orchestration::{FitConfig, FitResult, StandardFitResult, fit_from_formula};
use gam_terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineEndpointBoundaryCondition,
    BSplineIdentifiability, BSplineKnotSpec, OneDimensionalBoundary, build_bspline_basis_1d,
};
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2};

fn single_predictor_data() -> EncodedDataset {
    let headers = ["x", "y"].into_iter().map(str::to_string).collect();
    let rows = (0..48)
        .map(|i| {
            let x = i as f64 / 47.0;
            // Deliberately far from every tested anchor at x=0: an omitted
            // affine lift cannot accidentally look like a successful pin.
            let y = 5.0 + 3.0 * x + 0.25 * x * (1.0 - x);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode one-predictor fixture")
}

fn two_predictor_data() -> EncodedDataset {
    let headers = ["x", "z", "y"].into_iter().map(str::to_string).collect();
    let rows = (0..48)
        .map(|i| {
            let x = i as f64 / 47.0;
            let z = 1.0 - x;
            let y = 0.5 + 0.6 * x * x + 0.4 * (1.0 - z) * (1.0 - z);
            StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode two-predictor fixture")
}

fn fit_gaussian(formula: &str, data: &EncodedDataset) -> StandardFitResult {
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &config)
        .unwrap_or_else(|error| panic!("Gaussian fit `{formula}` failed: {error:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit for `{formula}`");
    };
    fit
}

fn one_predictor_probe(data: &EncodedDataset, values: &[f64]) -> Array2<f64> {
    let mut probe = Array2::zeros((values.len(), data.values.ncols()));
    let x_col = data.column_map()["x"];
    for (row, value) in values.iter().copied().enumerate() {
        probe[[row, x_col]] = value;
    }
    probe
}

#[test]
fn closed_form_nonzero_anchor_is_affine_and_the_fitted_chart_is_homogeneous() {
    let x = Array1::linspace(0.0, 1.0, 20);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 8,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions {
            left: BSplineEndpointBoundaryCondition::Anchored { value: 1.5 },
            right: BSplineEndpointBoundaryCondition::Free,
        },
    };
    let built = build_bspline_basis_1d(x.view(), &spec).expect("build affine anchored basis");
    let affine = built
        .affine_offset
        .as_ref()
        .expect("a non-zero anchor must emit a fixed affine row function");

    assert!(
        (affine[0] - 1.5).abs() < 1e-10,
        "the particular solution must equal the requested endpoint value"
    );
    let homogeneous_design = built.design.to_dense();
    assert!(
        homogeneous_design
            .row(0)
            .iter()
            .all(|value| value.abs() < 1e-10),
        "every estimated basis column must obey the homogeneous endpoint pin"
    );
}

#[test]
fn fitted_and_frozen_predictions_obey_every_nonzero_anchor_variant() {
    let data = single_predictor_data();
    let values = [0.0, 0.2, 0.5, 0.8, 1.0];
    let probe = one_predictor_probe(&data, &values);

    for (formula, left_pin, right_pin) in [
        (
            "y ~ s(x, bc=anchored, anchor_left=1, anchor_right=-1)",
            Some(1.0),
            Some(-1.0),
        ),
        ("y ~ s(x, bc_left=anchored, anchor_left=1)", Some(1.0), None),
        (
            "y ~ s(x, bc_right=anchored, anchor_right=-1)",
            None,
            Some(-1.0),
        ),
        (
            "y ~ s(x, bc_left=anchored, anchor_left=0.5)",
            Some(0.5),
            None,
        ),
    ] {
        let fit = fit_gaussian(formula, &data);
        let frozen = build_term_collection_design(probe.view(), &fit.resolvedspec)
            .unwrap_or_else(|error| panic!("frozen rebuild `{formula}` failed: {error:?}"));
        assert_eq!(
            frozen.design.ncols(),
            fit.fit.beta.len(),
            "frozen design width must match the fitted homogeneous chart for `{formula}`"
        );
        let prediction = frozen
            .apply(fit.fit.beta.view())
            .unwrap_or_else(|error| panic!("frozen prediction `{formula}` failed: {error:?}"));
        assert!(
            prediction.iter().all(|value| value.is_finite()),
            "all frozen predictions must be finite for `{formula}`"
        );
        if let Some(expected) = left_pin {
            assert!(
                (prediction[0] - expected).abs() < 1e-8,
                "left prediction for `{formula}` was {}, expected {expected}",
                prediction[0]
            );
        }
        if let Some(expected) = right_pin {
            let last = prediction.len() - 1;
            assert!(
                (prediction[last] - expected).abs() < 1e-8,
                "right prediction for `{formula}` was {}, expected {expected}",
                prediction[last]
            );
        }
    }
}

#[test]
fn multi_term_affine_offsets_sum_once_and_replay_exactly_after_freeze() {
    let data = two_predictor_data();
    let fit = fit_gaussian(
        "y ~ s(x, bc_left=anchored, anchor_left=1.25, k=8) + \
         s(z, bc_right=anchored, anchor_right=-0.75, k=8)",
        &data,
    );

    let training_replay = build_term_collection_design(data.values.view(), &fit.resolvedspec)
        .expect("replay frozen multi-term training design");
    assert_eq!(fit.design.affine_offset, training_replay.affine_offset);
    assert_eq!(
        fit.design.design.to_dense(),
        training_replay.design.to_dense(),
        "freezing must preserve the complete homogeneous training design"
    );

    let mut probe = Array2::<f64>::zeros((3, data.values.ncols()));
    let x_col = data.column_map()["x"];
    let z_col = data.column_map()["z"];
    for (row, (x, z)) in [(0.0, 1.0), (0.3, 0.8), (1.0, 0.0)].into_iter().enumerate() {
        probe[[row, x_col]] = x;
        probe[[row, z_col]] = z;
    }

    let joint = build_term_collection_design(probe.view(), &fit.resolvedspec)
        .expect("build joint frozen prediction design");
    let mut left_spec = fit.resolvedspec.clone();
    left_spec.smooth_terms = vec![fit.resolvedspec.smooth_terms[0].clone()];
    let left = build_term_collection_design(probe.view(), &left_spec)
        .expect("build left-anchor frozen design");
    let mut right_spec = fit.resolvedspec.clone();
    right_spec.smooth_terms = vec![fit.resolvedspec.smooth_terms[1].clone()];
    let right = build_term_collection_design(probe.view(), &right_spec)
        .expect("build right-anchor frozen design");

    for row in 0..probe.nrows() {
        let expected = left.affine_offset[row] + right.affine_offset[row];
        assert!(
            (joint.affine_offset[row] - expected).abs() < 1e-12,
            "row {row}: the joint affine channel must be the exact sum of its terms"
        );
    }
    assert!(
        (joint.affine_offset[0] - 0.5).abs() < 1e-10,
        "the coincident endpoint pins must contribute 1.25 + (-0.75) exactly once"
    );

    let prediction = joint
        .apply(fit.fit.beta.view())
        .expect("apply multi-term affine prediction design");
    let homogeneous = joint.design.apply(&fit.fit.beta);
    for row in 0..probe.nrows() {
        assert!(prediction[row].is_finite());
        assert!(
            (prediction[row] - homogeneous[row] - joint.affine_offset[row]).abs() < 1e-12,
            "row {row}: prediction must equal X beta plus one affine lift"
        );
    }
}

#[test]
fn one_sided_zero_anchor_survives_freeze_without_reintroducing_an_intercept() {
    let data = single_predictor_data();
    let values = [0.0, 0.25, 0.5, 0.75, 1.0];
    let probe = one_predictor_probe(&data, &values);

    for (formula, anchored_row) in [
        ("y ~ s(x, bc_left=anchored, anchor_left=0, k=10)", 0),
        (
            "y ~ s(x, bc_right=anchored, anchor_right=0, k=10)",
            values.len() - 1,
        ),
    ] {
        let fit = fit_gaussian(formula, &data);
        let frozen = build_term_collection_design(probe.view(), &fit.resolvedspec)
            .unwrap_or_else(|error| panic!("one-sided frozen rebuild `{formula}`: {error:?}"));
        assert_eq!(
            frozen.design.ncols(),
            fit.fit.beta.len(),
            "freeze must not reintroduce the globally suppressed intercept for `{formula}`"
        );
        let prediction = frozen
            .apply(fit.fit.beta.view())
            .unwrap_or_else(|error| panic!("one-sided frozen prediction `{formula}`: {error:?}"));
        assert!(prediction.iter().all(|value| value.is_finite()));
        assert!(
            prediction[anchored_row].abs() < 1e-8,
            "the frozen one-sided endpoint for `{formula}` must remain pinned, got {}",
            prediction[anchored_row]
        );
    }
}
