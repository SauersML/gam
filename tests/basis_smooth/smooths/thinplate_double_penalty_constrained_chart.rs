//! Regression for the #1476-class double-penalty null-space bug on the
//! thin-plate / Matérn-blended (`bs="tp"`) path.
//!
//! Like the tensor (`te`/`ti`) and 1-D B-spline paths, a thin-plate smooth's
//! Marra & Wood (2011) double-penalty null-space shrinkage ridge must shrink the
//! null space of the *constrained* bending penalty `Zᵀ S_bend Z`, in the same
//! coefficient chart the fit uses. The ridge was instead built as the projector
//! `Z_null Z_nullᵀ` onto the null space of the RAW (pre-identifiability) bending
//! penalty — its all-zero polynomial block — and then merely congruence-
//! restricted through the identifiability transform `Z`
//! (`SpatialIdentifiability::OrthogonalToParametric`, the default). That `Z`
//! DROPS the constant polynomial column and is not norm-preserving, so the
//! restricted matrix `Zᵀ Z_null Z_nullᵀ Z` is neither idempotent nor aligned
//! with `null(Zᵀ S_bend Z)` and carries shrinkage mass onto penalized (kernel)
//! directions — the #1266/#1476 flat-collapse / EDF mis-allocation class.
//!
//! Contract pinned on the realized constrained `penalties_local`: the ridge and
//! the bending penalty penalize COMPLEMENTARY subspaces, so `ridge·bending ≈ 0`
//! and the ridge's principal direction earns ~zero bending energy. The raw-
//! restricted ridge violates both.

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
    let grid = 16usize;
    for i in 0..grid {
        for j in 0..grid {
            let x = i as f64 / (grid as f64 - 1.0);
            let z = j as f64 / (grid as f64 - 1.0);
            let y = (2.0 * x).sin() + (3.0 * z).cos() + 0.5 * x * z;
            rows.push(StringRecord::from(vec![
                x.to_string(),
                z.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode thin-plate grid dataset")
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
        panic!("expected a standard GAM fit for {formula}");
    };
    fit
}

fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn thin_plate_double_penalty_ridge_lives_in_constrained_null_space() {
    // 2-D thin-plate (`s(x, z)` auto-promotes to bs="tp"); poly nullspace =
    // {const, x, z}. Default double_penalty=true emits the null-space ridge,
    // and the default OrthogonalToParametric identifiability drops the constant,
    // so the constrained null space genuinely differs from the raw one.
    let fit = fit("y ~ s(x, z, bs=\"tp\", k=20, double_penalty=TRUE)");
    let formula = "s(x, z, bs=\"tp\", double_penalty)";

    let term = &fit.design.smooth.terms[0];
    let width = term.coeff_range.len();

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
            PenaltySource::DoublePenaltyNullspace => {
                assert!(ridge.is_none(), "{formula}: expected a single ridge");
                ridge = Some(s.clone());
            }
            // The thin-plate bending penalty the ridge's null space is built
            // from. (The `bs="tp"` path emits exactly Primary + the ridge.)
            PenaltySource::Primary => bending += s,
            ref other => panic!("{formula}: unexpected thin-plate penalty source {other:?}"),
        }
    }

    let ridge = ridge.unwrap_or_else(|| {
        panic!("{formula}: double_penalty=TRUE must emit a DoublePenaltyNullspace ridge")
    });

    let bending_norm = frob(&bending);
    let ridge_norm = frob(&ridge);
    assert!(
        bending_norm > 0.0 && ridge_norm > 0.0,
        "{formula}: bending and ridge must both be non-trivial \
         (bending={bending_norm:.3e}, ridge={ridge_norm:.3e})"
    );

    // Complementary-subspace contract: ridge range = bending null space ⇒
    // ridge·bending = 0. The raw-restricted ridge leaves a large residual here
    // (numerically ≈ 0.43 of the product norm in the failing repro).
    let product = ridge.dot(&bending);
    let rel = frob(&product) / (ridge_norm * bending_norm);
    assert!(
        rel < 1e-8,
        "{formula}: double-penalty ridge does not annihilate the constrained \
         bending penalty (‖ridge·bending‖/(‖ridge‖‖bending‖) = {rel:.3e} ≥ 1e-8); \
         the ridge is built in the wrong (raw) chart and shrinks penalized \
         directions (#1476-class)."
    );

    // The ridge's principal direction must be a genuine bending null direction.
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
    assert!(
        bending_form.abs() / bending_norm < 1e-6,
        "{formula}: the ridge's principal shrinkage direction earns a non-zero \
         bending energy (vᵀ S_bend v = {bending_form:.3e}); it is not a null-space \
         direction of the constrained bending penalty (#1476-class)."
    );
}
