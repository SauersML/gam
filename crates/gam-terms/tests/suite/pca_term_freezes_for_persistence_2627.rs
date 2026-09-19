//! A Pca smooth term freezes for persistence (#2627).
//!
//! `freeze_term_collection_from_design` rewrites every smooth term's spec from
//! its fit-time metadata, so save → load → predict rebuilds the training design.
//! The freezer had no Pca arm, so every Pca term reached the catch-all
//! "smooth metadata/spec type mismatch while freezing term" refusal and no
//! `pca()` fit could be assembled (examples/lazy_pca_basis_demo.py, census
//! 1145256). A frozen centred eager term must carry the TRAINING mean: a rebuild
//! over a few rows then reproduces those rows of the training design instead of
//! recentring on the subset.

use gam_terms::basis::BasisMetadata;
use gam_terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, freeze_term_collection_from_design,
};
use ndarray::{Array2, s};

const N: usize = 40;

/// Three smooth covariates with distinct nonzero means, so recentring a subset
/// on its own mean moves the design by O(1), far above any rounding.
fn covariates() -> Array2<f64> {
    Array2::from_shape_fn((N, 3), |(i, j)| {
        let t = i as f64 / N as f64;
        match j {
            0 => (6.0 * t).sin() + 0.5,
            1 => t * t - 0.2,
            _ => 0.7 * (3.0 * t).cos() + 1.1,
        }
    })
}

fn eager_pca_spec() -> TermCollectionSpec {
    // Orthonormal directions over the three covariates (one row per covariate).
    let directions = Array2::from_shape_vec((3, 2), vec![0.6, 0.0, 0.8, 0.0, 0.0, 1.0])
        .expect("3x2 principal directions");
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "pca_eager".to_string(),
            basis: SmoothBasisSpec::Pca {
                feature_cols: vec![0, 1, 2],
                basis_matrix: directions,
                centered: true,
                center_mean: None,
                pca_basis_path: None,
                chunk_size: 32,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
        level: Default::default(),
    }
}

#[test]
fn an_eager_pca_term_freezes_its_training_centering_2627() {
    let data = covariates();
    let spec = eager_pca_spec();
    let training = build_term_collection_design(data.view(), &spec).expect("fit-time pca design");
    let BasisMetadata::Pca {
        center_mean: Some(training_mean),
        ..
    } = &training.smooth.terms[0].metadata
    else {
        panic!("a centred eager pca term records its training mean in its metadata");
    };

    let frozen = freeze_term_collection_from_design(&spec, &training)
        .expect("a pca term must freeze for persistence");
    let SmoothBasisSpec::Pca {
        center_mean,
        pca_basis_path,
        ..
    } = &frozen.smooth_terms[0].basis
    else {
        panic!("freezing keeps the pca basis kind");
    };
    assert_eq!(
        center_mean.as_ref(),
        Some(training_mean),
        "the frozen term carries the training mean"
    );
    assert!(
        pca_basis_path.is_none(),
        "an eager pca term has no score file to depend on"
    );

    let rows = 5;
    let subset = data.slice(s![0..rows, ..]).to_owned();
    let rebuilt = build_term_collection_design(subset.view(), &frozen)
        .expect("rebuild from the frozen spec");
    let training_dense = training.design.to_dense();
    let rebuilt_dense = rebuilt.design.to_dense();
    assert_eq!(rebuilt_dense.ncols(), training_dense.ncols());
    for row in 0..rows {
        for col in 0..training_dense.ncols() {
            let expected = training_dense[[row, col]];
            let got = rebuilt_dense[[row, col]];
            assert!(
                (got - expected).abs() <= 8.0 * f64::EPSILON * expected.abs().max(1.0),
                "rebuilt design [{row}, {col}] = {got} but the training design has {expected}: \
                 the frozen term must reuse the training centering"
            );
        }
    }
}
