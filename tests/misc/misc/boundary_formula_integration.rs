use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn boundary_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..28)
        .map(|i| {
            let x = i as f64 / 27.0;
            let y = 0.5 + 2.0 * x * x * (1.0 - x) * (1.0 - x);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode boundary dataset")
}

#[test]
fn fit_from_formula_accepts_bspline_endpoint_boundary_conditions() {
    init_parallelism();
    let data = boundary_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bc_left=anchored, anchor_left=0, bc_right=clamped, k=8)",
        &data,
        &config,
    )
    .expect("boundary-conditioned formula fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert_eq!(fit.design.smooth.terms.len(), 1);
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert!(!fit.design.smooth.terms[0].coeff_range.is_empty());
}

#[test]
fn boundary_conditioned_saved_spec_rebuilds_for_prediction() {
    init_parallelism();
    let data = boundary_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bc_left=anchored, anchor_left=0, bc_right=clamped, k=8)",
        &data,
        &config,
    )
    .expect("boundary-conditioned formula fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };

    let mut new_data = Array2::<f64>::zeros((31, 2));
    for i in 0..31 {
        new_data[[i, 0]] = i as f64 / 30.0;
        new_data[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("frozen boundary-conditioned spec should rebuild for prediction");
    let pred = design
        .apply(fit.fit.beta.view())
        .expect("apply frozen boundary-conditioned design");
    assert!(pred.iter().all(|v| v.is_finite()));
}

/// #1265: a ONE-SIDED anchored B-spline (`s(x, bc_left=anchored, anchor_left=0)`)
/// suppresses the global intercept at fit time (#1238). The freeze folds the
/// boundary projection into the smooth's `FrozenTransform` and previously RESET
/// `boundary_conditions` to `Free`, so the predict-time intercept-suppression
/// decision flipped and re-added a spurious intercept column → the prediction
/// design had one more column than the fitted `beta` (the documented 21-vs-22
/// mismatch). After the fix the frozen spec keeps its anchored
/// `boundary_conditions` (the builder skips re-deriving the boundary transform
/// when identifiability is already `FrozenTransform`, so no double-projection),
/// the intercept stays suppressed at predict, the design width matches `beta`,
/// and the anchored endpoint evaluates to the anchor value (0).
#[test]
fn one_sided_anchored_bspline_is_predictable_after_freeze() {
    init_parallelism();
    // Linear-ish response anchored at 0 at x = 0, exactly the documented idiom.
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..200)
        .map(|i| {
            let x = i as f64 / 199.0;
            let y = 3.0 * x; // f(0) = 0, matches the anchor
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    for formula in [
        "y ~ s(x, bc_left=anchored, anchor_left=0, k=10)",
        "y ~ s(x, bc_right=anchored, anchor_right=0, k=10)",
    ] {
        let result = fit_from_formula(formula, &data, &config)
            .unwrap_or_else(|e| panic!("one-sided anchored fit `{formula}` must succeed: {e:?}"));
        let FitResult::Standard(fit) = result else {
            panic!("expected standard Gaussian fit for `{formula}`");
        };
        let p_beta = fit.fit.beta.len();

        // Predict on fresh data — this is where #1265 aborted with the 21-vs-22
        // design/beta column mismatch. The rebuilt design must have exactly
        // `p_beta` columns so `apply(beta)` is well-posed.
        let probe = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
        let mut new_data = Array2::<f64>::zeros((probe.len(), 2));
        for (i, &xp) in probe.iter().enumerate() {
            new_data[[i, 0]] = xp;
        }
        let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
            .unwrap_or_else(|e| {
                panic!("frozen one-sided anchored spec must rebuild `{formula}`: {e:?}")
            });
        assert_eq!(
            design.design.ncols(),
            p_beta,
            "prediction design column count must match fitted beta for `{formula}` \
             (the #1265 spurious-intercept regression)"
        );
        let pred = design
            .apply(fit.fit.beta.view())
            .unwrap_or_else(|e| panic!("apply frozen design `{formula}`: {e:?}"));
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "predictions must be finite for `{formula}`"
        );
        // The anchored endpoint is a structural value pin: f(anchor) == 0.
        let anchored_pred = if formula.contains("bc_left") {
            pred[0] // x = 0.0
        } else {
            pred[probe.len() - 1] // x = 1.0
        };
        assert!(
            anchored_pred.abs() < 1e-6,
            "anchored endpoint must evaluate to the anchor value 0 for `{formula}`, got {anchored_pred:.3e}"
        );
    }
}

#[test]
fn nonzero_anchors_sum_once_and_survive_multi_term_frozen_replay() {
    init_parallelism();
    let headers = ["x", "z", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..48)
        .map(|i| {
            let x = i as f64 / 47.0;
            let z = 1.0 - x;
            let y = 0.5 + 0.6 * x * x + 0.4 * (1.0 - z) * (1.0 - z);
            StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode affine dataset");
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bc_left=anchored, anchor_left=1.25, k=8) + \
         s(z, bc_right=anchored, anchor_right=-0.75, k=8)",
        &data,
        &config,
    )
    .expect("multi-term non-zero-anchor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };

    // Rebuilding the training design from the frozen specification must be an
    // exact replay, including the particular solutions that are not encoded in
    // the homogeneous coefficient vector.
    let replay = build_term_collection_design(data.values.view(), &fit.resolvedspec)
        .expect("replay frozen multi-term design");
    assert_eq!(fit.design.affine_offset, replay.affine_offset);
    assert_eq!(fit.design.design.to_dense(), replay.design.to_dense());

    let mut probe = Array2::<f64>::zeros((3, 3));
    probe[[0, 0]] = 0.0;
    probe[[0, 1]] = 1.0;
    probe[[1, 0]] = 0.3;
    probe[[1, 1]] = 0.8;
    probe[[2, 0]] = 1.0;
    probe[[2, 1]] = 0.0;
    let joint = build_term_collection_design(probe.view(), &fit.resolvedspec)
        .expect("joint frozen prediction design");

    let mut left_spec = fit.resolvedspec.clone();
    left_spec.smooth_terms = vec![fit.resolvedspec.smooth_terms[0].clone()];
    let left = build_term_collection_design(probe.view(), &left_spec)
        .expect("left-anchor frozen prediction design");
    let mut right_spec = fit.resolvedspec.clone();
    right_spec.smooth_terms = vec![fit.resolvedspec.smooth_terms[1].clone()];
    let right = build_term_collection_design(probe.view(), &right_spec)
        .expect("right-anchor frozen prediction design");

    for row in 0..probe.nrows() {
        let expected = left.affine_offset[row] + right.affine_offset[row];
        assert!(
            (joint.affine_offset[row] - expected).abs() < 1e-12,
            "row {row}: multi-term affine channel must be the exact sum of its terms"
        );
    }
    assert!(
        (joint.affine_offset[0] - 0.5).abs() < 1e-10,
        "the two endpoint anchors must contribute 1.25 + (-0.75) exactly once"
    );

    let prediction = joint
        .apply(fit.fit.beta.view())
        .expect("apply multi-term affine prediction design");
    let linear = joint.design.apply(&fit.fit.beta);
    for row in 0..probe.nrows() {
        assert!(prediction[row].is_finite());
        assert!(
            (prediction[row] - linear[row] - joint.affine_offset[row]).abs() < 1e-12,
            "row {row}: prediction must equal X beta plus exactly one affine lift"
        );
    }
}
