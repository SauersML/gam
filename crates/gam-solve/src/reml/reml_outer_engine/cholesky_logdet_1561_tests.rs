//! #1561 — a graded Hessian's `log|H|` from its LLT, not its eigenvalues.
//!
//! A large smoothing strength beside unit data curvature grades the penalized
//! Hessian by ten orders of magnitude. A backward-stable eigensolver perturbs it
//! normwise, `‖E‖₂ ≤ p·ε·‖H‖₂`, so each small eigenvalue carries an absolute
//! error of order `ε·‖H‖₂` and `Σ ln σ_i` moves by up to `p·ε·‖H‖₂·Σ 1/σ_i`.
//! Measured on the survival Weibull AFT curved arm (`quality_vs_lifelines_
//! weibull_aft_by`): the eigen-priced `½·log|H|` differed between a warm- and a
//! cold-started inner mode by `5.2·10⁻⁸` where the modes themselves moved the
//! exact value by `4.9·10⁻¹⁰`, and by `7.7·10⁻⁹` where they moved it by
//! `5·10⁻¹³`. That noise is the outer objective's value noise.
//!
//! An LLT's error is componentwise, `|δH| ≤ γ_(p+1)·|L||Lᵀ|`, so diagonal grading
//! costs it nothing. The fixture is `H = D·A·D` with `A` the Kac–Murdock–Szegő
//! matrix `A_ij = r^|i−j|` (`det A = (1 − r²)^(p−1)`) and `D` a diagonal of powers
//! of two, so `H` is exact in floating point and `log|H|` is known in closed form.

use super::*;
use ndarray::Array2;

const DIMENSION: usize = 12;
const KMS_RATIO: f64 = 0.5;

/// `D_ii = 2^e_i`: the first half at unit scale, the second half at the square
/// root of a `2^34 ≈ 1.7·10¹⁰` smoothing strength.
fn scale_exponents() -> Vec<i32> {
    (0..DIMENSION)
        .map(|i| if i < DIMENSION / 2 { 0 } else { 17 })
        .collect()
}

/// `H = D·A·D`, exact: every `A_ij = r^|i−j|` with `r = ½` is a power of two,
/// and so is every product with the powers of two in `D`.
fn graded_hessian() -> Array2<f64> {
    let exponents = scale_exponents();
    let mut h = Array2::<f64>::zeros((DIMENSION, DIMENSION));
    for i in 0..DIMENSION {
        for j in 0..DIMENSION {
            let kms = KMS_RATIO.powi(i.abs_diff(j) as i32);
            h[[i, j]] = kms * 2.0_f64.powi(exponents[i] + exponents[j]);
        }
    }
    h
}

/// `log|H| = (p − 1)·ln(1 − r²) + 2·Σ e_i·ln 2`, and the rounding band of
/// evaluating that sum.
fn exact_logdet() -> (f64, f64) {
    let kms_term = (DIMENSION - 1) as f64 * (1.0 - KMS_RATIO * KMS_RATIO).ln();
    let scale_term =
        2.0 * scale_exponents().iter().map(|&e| e as f64).sum::<f64>() * std::f64::consts::LN_2;
    let value = kms_term + scale_term;
    let band = gam_linalg::roundoff::accumulation_growth(4) * (kms_term.abs() + scale_term.abs());
    (value, band)
}

#[test]
fn graded_hessian_logdet_is_priced_inside_its_certified_band() {
    let h = graded_hessian();
    let (exact, reference_band) = exact_logdet();

    let spectral =
        DenseSpectralOperator::from_symmetric_with_mode(&h, PseudoLogdetMode::PositiveDefinite)
            .expect("the graded fixture is positive definite");
    let spectral_value = spectral.logdet();
    let spectral_band = spectral
        .logdet_forward_error()
        .expect("the Weyl bound is finite");
    let spectral_error = (spectral_value - exact).abs();

    let factored =
        DenseSpectralOperator::from_symmetric_with_mode(&h, PseudoLogdetMode::PositiveDefinite)
            .expect("the graded fixture is positive definite")
            .with_cholesky_logdet(&h);
    let factored_value = factored.logdet();
    let factored_band = factored
        .logdet_forward_error()
        .expect("the componentwise bound is finite");
    let factored_error = (factored_value - exact).abs();

    println!(
        "[#1561] exact={exact:.17e} eigh={spectral_value:.17e} |err|={spectral_error:.3e} \
         weyl={spectral_band:.3e} llt={factored_value:.17e} |err|={factored_error:.3e} \
         componentwise={factored_band:.3e} reference={reference_band:.3e}"
    );

    assert!(
        spectral_error <= spectral_band + reference_band,
        "the eigen-priced log|H| error {spectral_error:.3e} must sit inside its own Weyl \
         bound {spectral_band:.3e}"
    );
    assert!(
        factored_error <= factored_band + reference_band,
        "the LLT-priced log|H| error {factored_error:.3e} must sit inside its componentwise \
         bound {factored_band:.3e}"
    );
    assert!(
        factored_band < spectral_band,
        "the componentwise bound {factored_band:.3e} must be the tighter one on a graded \
         Hessian (Weyl {spectral_band:.3e}), or the operator does not install it"
    );
    assert!(
        spectral_error > factored_band + reference_band,
        "the fixture must exercise the defect: the eigen-priced error {spectral_error:.3e} \
         must exceed the band {factored_band:.3e} the operator now certifies"
    );
    // Traces, solves and the logdet derivatives stay on the eigenpairs.
    assert_eq!(factored.active_rank(), DIMENSION);
    assert_eq!(factored.raw_spectrum(), spectral.raw_spectrum());
}

#[test]
fn cholesky_logdet_is_not_installed_where_the_weyl_bound_is_tighter_or_the_mode_is_not_exact() {
    // Unit grading: the componentwise bound carries a `p·γ_(p+1)·‖H̃⁻¹‖_F` factor
    // the normwise one does not, and the spectral value is kept.
    let n = DIMENSION;
    let identity = Array2::<f64>::eye(n);
    let spectral = DenseSpectralOperator::from_symmetric_with_mode(
        &identity,
        PseudoLogdetMode::PositiveDefinite,
    )
    .expect("the identity is positive definite");
    let weyl = spectral
        .logdet_forward_error()
        .expect("the Weyl bound is finite");
    let kept = spectral.with_cholesky_logdet(&identity);
    assert!(kept.factored_logdet.is_none());
    assert_eq!(kept.logdet_forward_error(), Some(weyl));

    // A smooth spectral floor prices `Σ ln r_ε(σ)`, not the LLT's `Σ ln σ`: where the
    // LLT is installed there, its value carries the floor's exact share
    // `Σ ln(r_ε(σ)/σ)`, so it is the SAME smooth criterion, and it agrees with the
    // eigen-priced value inside the two bounds.
    let h = graded_hessian();
    let spectral_smooth =
        DenseSpectralOperator::from_symmetric_with_mode(&h, PseudoLogdetMode::Smooth)
            .expect("the graded fixture decomposes");
    let spectral_value = spectral_smooth.logdet();
    let spectral_band = spectral_smooth
        .logdet_forward_error()
        .expect("the Weyl bound is finite");
    let smooth = spectral_smooth.with_cholesky_logdet(&h);
    if let Some((value, band)) = smooth.factored_logdet {
        assert!(
            band < spectral_band,
            "installed only where its bound is the tighter: {band:.3e} vs {spectral_band:.3e}"
        );
        assert!(
            (value - spectral_value).abs() <= band + spectral_band,
            "the LLT-priced smooth value {value:.12e} is not the eigen-priced \
             {spectral_value:.12e} within {band:.3e} + {spectral_band:.3e}"
        );
    }
}
