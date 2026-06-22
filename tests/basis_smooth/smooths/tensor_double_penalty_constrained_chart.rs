//! Regression for the #1476-class double-penalty null-space bug, extended to
//! the TENSOR (`te`/`ti`) path.
//!
//! The Marra & Wood (2011) "double penalty" emits a second penalty — the
//! null-space shrinkage ridge — whose RANGE must be exactly the null space of
//! the bending penalty (the directions no roughness operator can see). REML
//! then drives an unsupported low-order/bilinear interaction to `EDF → 0`
//! independently of the wiggliness. For that to work the ridge has to live in
//! the SAME coefficient chart the fit uses: it must shrink the null space of
//! the *constrained* bending penalty, not the *raw* (pre-identifiability) one.
//!
//! The 1-D B-spline path already rebuilds its ridge in the constrained chart
//! (`rebuild_double_penalty_nullspace_in_constrained_chart`). The tensor path
//! used to build the ridge as `Z_null Z_nullᵀ` from the RAW Kronecker-sum
//! bending penalty and then merely congruence-restrict it through the
//! sum-to-zero transform `Z`. Because `Z` is not norm-preserving and `Zᵀ Z_null`
//! is not orthonormal, that restricted matrix is neither idempotent nor aligned
//! with `null(Zᵀ S_bend Z)`, so it shrank the wrong directions (the #1266/#1476
//! flat-collapse / EDF mis-allocation symptom under the tensor transform).
//!
//! Contract this test pins (independent of the implementation detail): in the
//! realized constrained coefficient chart, the double-penalty ridge and the
//! summed bending penalty must penalize COMPLEMENTARY subspaces. Equivalently
//! the ridge's range lies in the null space of the bending penalty, so their
//! product is (numerically) zero and the ridge's principal direction earns a
//! zero bending quadratic form. The raw-restricted ridge violates both.

use csv::StringRecord;
use gam::basis::PenaltySource;
use gam::linalg::faer_ndarray::FaerEigh;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;

fn grid_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    let grid = 18usize;
    for i in 0..grid {
        for j in 0..grid {
            let x = i as f64 / (grid as f64 - 1.0);
            let z = j as f64 / (grid as f64 - 1.0);
            let y = (2.0 * x).sin() + (3.0 * z).cos() + x * z;
            rows.push(StringRecord::from(vec![
                x.to_string(),
                z.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode tensor grid dataset")
}

fn fit(formula: &str) -> StandardFitResult {
    init_parallelism();
    let data = grid_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &config)
        .unwrap_or_else(|err| panic!("{formula} should fit: {err:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for {formula}");
    };
    fit
}

/// Frobenius norm of a matrix.
fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Assert the double-penalty ridge of a `te`/`ti` smooth lives in the
/// constrained chart: it shrinks exactly the null space of the constrained
/// bending penalty, so `ridge · bending ≈ 0` and the ridge's top direction has
/// a (near-)zero bending quadratic form.
fn assert_ridge_in_constrained_null_space(fit: &StandardFitResult, formula: &str) {
    let term = &fit.design.smooth.terms[0];
    let width = term.coeff_range.len();

    // Summed bending penalty (all TensorMarginal / TensorSeparable blocks) and
    // the single null-space ridge, both already in the realized constrained
    // coefficient chart (every penalty matrix is `width × width`).
    let mut bending = Array2::<f64>::zeros((width, width));
    let mut ridge: Option<Array2<f64>> = None;
    for (s, info) in term
        .penalties_local
        .iter()
        .zip(term.penaltyinfo_local.iter())
    {
        assert_eq!(
            (s.nrows(), s.ncols()),
            (width, width),
            "{formula}: every penalty must live in the constrained term chart"
        );
        match info.source {
            PenaltySource::TensorGlobalRidge => {
                assert!(
                    ridge.is_none(),
                    "{formula}: expected a single double-penalty ridge"
                );
                ridge = Some(s.clone());
            }
            PenaltySource::TensorMarginal { .. } | PenaltySource::TensorSeparable { .. } => {
                bending += s;
            }
            ref other => panic!("{formula}: unexpected tensor penalty source {other:?}"),
        }
    }

    let ridge = ridge.unwrap_or_else(|| {
        panic!("{formula}: double_penalty=TRUE must emit a TensorGlobalRidge null-space block")
    });

    let bending_norm = frob(&bending);
    let ridge_norm = frob(&ridge);
    assert!(
        bending_norm > 0.0 && ridge_norm > 0.0,
        "{formula}: both bending and ridge penalties must be non-trivial \
         (bending‖·‖={bending_norm:.3e}, ridge‖·‖={ridge_norm:.3e})"
    );

    // Complementary-subspace contract: the ridge range is the bending null
    // space, so `ridge · bending = 0`. Scale by the two operator norms so the
    // bound is dimensionless. The raw-chart restricted ridge leaves a large
    // residual here.
    let product = ridge.dot(&bending);
    let rel = frob(&product) / (ridge_norm * bending_norm);
    assert!(
        rel < 1e-8,
        "{formula}: double-penalty ridge does not annihilate the constrained \
         bending penalty (‖ridge·bending‖/(‖ridge‖‖bending‖) = {rel:.3e} ≥ 1e-8). \
         The ridge is built in the wrong (raw) chart and shrinks penalized \
         directions — the #1476-class tensor null-space bug."
    );

    // The ridge's principal direction must be a genuine bending null direction:
    // its bending quadratic form is ~0 (it lives in the unpenalized subspace).
    let ridge_sym = (&ridge + &ridge.t()) * 0.5;
    let (evals, evecs) = FaerEigh::eigh(&ridge_sym, faer::Side::Lower)
        .unwrap_or_else(|err| panic!("{formula}: ridge eigendecomposition failed: {err:?}"));
    let top = evals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .expect("ridge has eigenvalues");
    let v = evecs.column(top).to_owned();
    let bending_form = v.dot(&bending.dot(&v));
    // Normalize against the bending penalty's spectral scale so the bound is
    // dimensionless and independent of the (unit-Frobenius) penalty scaling.
    let bending_scale = frob(&bending).max(1e-30);
    assert!(
        bending_form.abs() / bending_scale < 1e-6,
        "{formula}: the ridge's principal shrinkage direction earns a non-zero \
         bending energy (vᵀ S_bend v = {bending_form:.3e}, scale {bending_scale:.3e}); \
         it is not a null-space direction of the constrained bending penalty \
         (#1476-class)."
    );
}

#[test]
fn te_double_penalty_ridge_lives_in_constrained_null_space() {
    let fit = fit("y ~ te(x, z, k=5, double_penalty=TRUE)");
    assert_ridge_in_constrained_null_space(&fit, "te(x, z, double_penalty)");
}

#[test]
fn ti_double_penalty_ridge_lives_in_constrained_null_space() {
    let fit = fit("y ~ ti(x, z, k=5, double_penalty=TRUE)");
    assert_ridge_in_constrained_null_space(&fit, "ti(x, z, double_penalty)");
}
