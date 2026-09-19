//! Matrix-level effect and contrast inference.
//!
//! This module owns the statistical kernel shared by difference-smooth,
//! partial-dependence, and other linear-contrast reports.  Callers supply a
//! coefficient vector, its covariance, and a contrast design.  Presentation
//! layers only marshal the resulting typed report.

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::matrix::symmetrize_in_place;
use gam_math::probability::standard_normal_quantile;
use gam_math::quantile::quantile_from_sorted;
use gam_solve::estimate::UnifiedFitResult;
use gam_solve::model_types::InferenceCovarianceMode;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::error::Error;
use std::fmt;

/// Default confidence level for effect bands.
pub(crate) const DEFAULT_BAND_LEVEL: f64 = 0.95;
/// Default Monte Carlo draw count for simultaneous bands.
pub(crate) const DEFAULT_SIMULATIONS: usize = 10_000;
/// Default deterministic random seed for simultaneous bands.
pub(crate) const DEFAULT_SIMULATION_SEED: u64 = 12_345;

/// The coefficient covariance definition used by an effect report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CovarianceSource {
    /// Conditional Bayesian covariance with smoothing parameters fixed (`Vb`).
    Conditional,
    /// Bayesian covariance including smoothing-parameter uncertainty (`Vp`).
    SmoothingCorrected,
    /// Frequentist sandwich covariance (`Ve`).
    Frequentist,
}

impl fmt::Display for CovarianceSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Conditional => "conditional",
            Self::SmoothingCorrected => "smoothing-corrected",
            Self::Frequentist => "frequentist",
        })
    }
}

/// A covariance borrowed from a fit together with its exact provenance.
#[derive(Clone, Copy, Debug)]
pub struct SelectedCovariance<'a> {
    pub source: CovarianceSource,
    pub matrix: ArrayView2<'a, f64>,
}

/// Select a coefficient covariance from a unified fit under an explicit policy.
///
/// The requested source is exact: absence is an error and no other covariance
/// definition is substituted. [`SelectedCovariance::source`] records the
/// resolved provenance.
pub fn select_covariance<'a>(
    fit: &'a UnifiedFitResult,
    source: CovarianceSource,
) -> Result<SelectedCovariance<'a>, EffectError> {
    let matrix =
        covariance_by_source(fit, source).ok_or(EffectError::MissingCovariance { source })?;

    Ok(SelectedCovariance {
        source,
        matrix: matrix.view(),
    })
}

/// Select the covariance the fit PUBLISHES (gam#2779): smoothing-corrected
/// whenever the fit carries it, conditional when the correction is typed
/// unavailable (e.g. a smoothing parameter certified at its infinite rail).
/// This is the default for every band surface, so one fitted object prices
/// its `summary()` standard errors and its effect bands from the same matrix;
/// [`SelectedCovariance::source`] names which one was used.
pub fn select_published_covariance(
    fit: &UnifiedFitResult,
) -> Result<SelectedCovariance<'_>, EffectError> {
    let source = match fit.published_covariance_mode() {
        InferenceCovarianceMode::SmoothingCorrected => CovarianceSource::SmoothingCorrected,
        InferenceCovarianceMode::Conditional => CovarianceSource::Conditional,
    };
    select_covariance(fit, source)
}

fn covariance_by_source(fit: &UnifiedFitResult, source: CovarianceSource) -> Option<&Array2<f64>> {
    match source {
        CovarianceSource::Conditional => fit.beta_covariance(),
        CovarianceSource::SmoothingCorrected => fit.beta_covariance_corrected(),
        CovarianceSource::Frequentist => fit.beta_covariance_ve(),
    }
}

/// Configuration for a pointwise normal-theory confidence band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PointwiseBandOptions {
    pub level: f64,
}

impl Default for PointwiseBandOptions {
    fn default() -> Self {
        Self {
            level: DEFAULT_BAND_LEVEL,
        }
    }
}

/// Configuration for a simulated simultaneous confidence band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SimultaneousBandOptions {
    pub level: f64,
    pub simulations: usize,
    pub seed: u64,
}

impl Default for SimultaneousBandOptions {
    fn default() -> Self {
        Self {
            level: DEFAULT_BAND_LEVEL,
            simulations: DEFAULT_SIMULATIONS,
            seed: DEFAULT_SIMULATION_SEED,
        }
    }
}

/// Confidence-band procedure for a linear effect curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BandOptions {
    /// Independent marginal normal intervals at each contrast row.
    Pointwise(PointwiseBandOptions),
    /// A common critical value calibrated from the supremum of the standardized
    /// Gaussian effect curve.
    Simultaneous(SimultaneousBandOptions),
}

impl Default for BandOptions {
    fn default() -> Self {
        Self::Pointwise(PointwiseBandOptions::default())
    }
}

/// A matrix-level effect report, with one entry per contrast-design row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EffectReport {
    pub center: Array1<f64>,
    pub se: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub critical: f64,
    /// Whole-curve p-value of `H0: C beta = 0` at every contrast row, from the
    /// same simulated law of `max_i |Z_i|` that calibrates the simultaneous
    /// critical value: `(1 + #{M_s >= T}) / (S + 1)` with
    /// `T = max_i |center_i| / se_i`. It rejects at `alpha` exactly when the
    /// `1 - alpha` simultaneous band excludes zero somewhere, up to the
    /// simulation's resolution. `None` for pointwise bands, which carry no
    /// whole-curve null law.
    pub zero_curve_p_value: Option<f64>,
}

/// Typed failures from covariance selection or effect-band construction.
#[derive(Clone, Debug, PartialEq)]
pub enum EffectError {
    MissingCovariance {
        source: CovarianceSource,
    },
    EmptyCoefficients,
    EmptyContrastDesign,
    InvalidLevel {
        level: f64,
    },
    InvalidSimulationCount,
    CovarianceShape {
        rows: usize,
        columns: usize,
        expected: usize,
    },
    ContrastShape {
        columns: usize,
        expected: usize,
    },
    NonFiniteInput {
        input: &'static str,
    },
    NonSymmetricCovariance {
        row: usize,
        column: usize,
        difference: f64,
        tolerance: f64,
    },
    IndefiniteCovariance {
        matrix: &'static str,
        minimum_eigenvalue: f64,
        tolerance: f64,
    },
    Eigendecomposition {
        matrix: &'static str,
        detail: String,
    },
    NormalQuantile {
        detail: String,
    },
}

impl fmt::Display for EffectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCovariance { source } => {
                write!(formatter, "fit has no {source} coefficient covariance")
            }
            Self::EmptyCoefficients => {
                formatter.write_str("beta must contain at least one coefficient")
            }
            Self::EmptyContrastDesign => {
                formatter.write_str("contrast design must contain at least one row")
            }
            Self::InvalidLevel { level } => {
                write!(
                    formatter,
                    "confidence level must be finite and in (0, 1), got {level}"
                )
            }
            Self::InvalidSimulationCount => {
                formatter.write_str("simultaneous-band simulation count must be positive")
            }
            Self::CovarianceShape {
                rows,
                columns,
                expected,
            } => write!(
                formatter,
                "covariance must have shape {expected}x{expected}, got {rows}x{columns}"
            ),
            Self::ContrastShape { columns, expected } => write!(
                formatter,
                "contrast design must have {expected} columns, got {columns}"
            ),
            Self::NonFiniteInput { input } => {
                write!(formatter, "{input} contains a non-finite value")
            }
            Self::NonSymmetricCovariance {
                row,
                column,
                difference,
                tolerance,
            } => write!(
                formatter,
                "covariance is not symmetric at ({row}, {column}): absolute difference {difference:e} exceeds tolerance {tolerance:e}"
            ),
            Self::IndefiniteCovariance {
                matrix,
                minimum_eigenvalue,
                tolerance,
            } => write!(
                formatter,
                "{matrix} is materially indefinite: minimum eigenvalue {minimum_eigenvalue:e} is below -{tolerance:e}"
            ),
            Self::Eigendecomposition { matrix, detail } => {
                write!(formatter, "{matrix} eigendecomposition failed: {detail}")
            }
            Self::NormalQuantile { detail } => {
                write!(
                    formatter,
                    "normal critical-value calculation failed: {detail}"
                )
            }
        }
    }
}

impl Error for EffectError {}


/// Compute centers, standard errors, bounds, and the common critical value for
/// a linear effect curve.
///
/// For `m x p` contrast design `C`, coefficient vector `beta`, and covariance
/// `V`, the report center is `C beta` and its covariance is `C V C'`.
/// Pointwise bands compute only the diagonal of that covariance, with O(p)
/// working memory. Simultaneous bands calibrate `max_i |Z_i|` for the
/// standardized Gaussian curve and factor whichever covariance space is
/// smaller: coefficient space when `p <= m`, projected curve space otherwise.
/// Positive-semidefinite singular matrices are supported without a ridge.
pub(crate) fn effect_report(
    beta: ArrayView1<'_, f64>,
    covariance: ArrayView2<'_, f64>,
    contrast_design: ArrayView2<'_, f64>,
    options: BandOptions,
) -> Result<EffectReport, EffectError> {
    validate_inputs(beta, covariance, contrast_design, options)?;

    let covariance = validated_symmetric_matrix(covariance)?;
    let center = contrast_design.dot(&beta);
    let (se, critical, zero_curve_p_value) = match options {
        BandOptions::Pointwise(pointwise) => {
            let se = pointwise_standard_errors(contrast_design, covariance.view())?;
            let critical = standard_normal_quantile(0.5 * (1.0 + pointwise.level))
                .map_err(|detail| EffectError::NormalQuantile { detail })?;
            (se, critical, None)
        }
        BandOptions::Simultaneous(simultaneous) => {
            let curve_factor = simultaneous_curve_factor(contrast_design, covariance.view())?;
            let se = factor_standard_errors(&curve_factor);
            let maxima = simulated_supremum_law(
                &curve_factor,
                se.view(),
                simultaneous.simulations,
                simultaneous.seed,
            );
            let critical = quantile_from_sorted(&maxima, simultaneous.level);
            let p_value = supremum_tail_p_value(&maxima, max_standardized_magnitude(&center, &se));
            (se, critical, Some(p_value))
        }
    };

    let half_width = se.mapv(|value| critical * value);
    let lower = &center - &half_width;
    let upper = &center + &half_width;
    Ok(EffectReport {
        center,
        se,
        lower,
        upper,
        critical,
        zero_curve_p_value,
    })
}

fn pointwise_standard_errors(
    contrast_design: ArrayView2<'_, f64>,
    covariance: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, EffectError> {
    let p = covariance.nrows();
    let mut se = Array1::<f64>::zeros(contrast_design.nrows());
    let mut product = vec![0.0_f64; p];
    for (row_index, row) in contrast_design.rows().into_iter().enumerate() {
        product.fill(0.0);
        for covariance_row in 0..p {
            for column in 0..p {
                product[covariance_row] += covariance[[covariance_row, column]] * row[column];
            }
        }
        let variance = row
            .iter()
            .zip(&product)
            .map(|(&loading, &projected)| loading * projected)
            .sum::<f64>();
        let scale = row
            .iter()
            .zip(&product)
            .map(|(&loading, &projected)| (loading * projected).abs())
            .sum::<f64>();
        let tolerance = roundoff_tolerance(scale, p);
        if variance < -tolerance {
            return Err(EffectError::IndefiniteCovariance {
                matrix: "projected curve covariance",
                minimum_eigenvalue: variance,
                tolerance,
            });
        }
        se[row_index] = variance.max(0.0).sqrt();
    }
    Ok(se)
}

fn simultaneous_curve_factor(
    contrast_design: ArrayView2<'_, f64>,
    covariance: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, EffectError> {
    if covariance.nrows() <= contrast_design.nrows() {
        let coefficient_eigen = psd_eigendecomposition(covariance, "coefficient covariance")?;
        let coefficient_factor = covariance_factor(&coefficient_eigen);
        let factor = contrast_design.dot(&coefficient_factor);
        if factor.iter().any(|value| !value.is_finite()) {
            return Err(EffectError::NonFiniteInput {
                input: "projected curve factor",
            });
        }
        return Ok(factor);
    }

    let mut curve_covariance = contrast_design.dot(&covariance).dot(&contrast_design.t());
    if curve_covariance.iter().any(|value| !value.is_finite()) {
        return Err(EffectError::NonFiniteInput {
            input: "projected curve covariance",
        });
    }
    symmetrize_in_place(&mut curve_covariance);
    let curve_eigen =
        psd_eigendecomposition(curve_covariance.view(), "projected curve covariance")?;
    Ok(covariance_factor(&curve_eigen))
}

fn validate_inputs(
    beta: ArrayView1<'_, f64>,
    covariance: ArrayView2<'_, f64>,
    contrast_design: ArrayView2<'_, f64>,
    options: BandOptions,
) -> Result<(), EffectError> {
    if beta.is_empty() {
        return Err(EffectError::EmptyCoefficients);
    }
    if contrast_design.nrows() == 0 {
        return Err(EffectError::EmptyContrastDesign);
    }
    let p = beta.len();
    if covariance.dim() != (p, p) {
        return Err(EffectError::CovarianceShape {
            rows: covariance.nrows(),
            columns: covariance.ncols(),
            expected: p,
        });
    }
    if contrast_design.ncols() != p {
        return Err(EffectError::ContrastShape {
            columns: contrast_design.ncols(),
            expected: p,
        });
    }
    if beta.iter().any(|value| !value.is_finite()) {
        return Err(EffectError::NonFiniteInput { input: "beta" });
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(EffectError::NonFiniteInput {
            input: "covariance",
        });
    }
    if contrast_design.iter().any(|value| !value.is_finite()) {
        return Err(EffectError::NonFiniteInput {
            input: "contrast design",
        });
    }

    let (level, simulations) = match options {
        BandOptions::Pointwise(pointwise) => (pointwise.level, None),
        BandOptions::Simultaneous(simultaneous) => {
            (simultaneous.level, Some(simultaneous.simulations))
        }
    };
    if !level.is_finite() || !(0.0..1.0).contains(&level) || level == 0.0 {
        return Err(EffectError::InvalidLevel { level });
    }
    if simulations == Some(0) {
        return Err(EffectError::InvalidSimulationCount);
    }
    Ok(())
}

struct PsdEigen {
    vectors: Array2<f64>,
    active: Vec<(usize, f64)>,
}

fn validated_symmetric_matrix(matrix: ArrayView2<'_, f64>) -> Result<Array2<f64>, EffectError> {
    let n = matrix.nrows();
    let scale = matrix
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let symmetry_tolerance = roundoff_tolerance(scale, n);
    let mut symmetric = matrix.to_owned();
    for row in 0..n {
        for column in 0..row {
            let difference = (matrix[[row, column]] - matrix[[column, row]]).abs();
            if difference > symmetry_tolerance {
                return Err(EffectError::NonSymmetricCovariance {
                    row,
                    column,
                    difference,
                    tolerance: symmetry_tolerance,
                });
            }
            let average = 0.5 * (matrix[[row, column]] + matrix[[column, row]]);
            symmetric[[row, column]] = average;
            symmetric[[column, row]] = average;
        }
    }
    Ok(symmetric)
}

fn psd_eigendecomposition(
    matrix: ArrayView2<'_, f64>,
    label: &'static str,
) -> Result<PsdEigen, EffectError> {
    let n = matrix.nrows();
    let symmetric = validated_symmetric_matrix(matrix)?;

    let (values, vectors) =
        symmetric
            .eigh(Side::Lower)
            .map_err(|error| EffectError::Eigendecomposition {
                matrix: label,
                detail: error.to_string(),
            })?;
    let spectral_scale = values
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let tolerance = roundoff_tolerance(spectral_scale, n);
    let minimum_eigenvalue = values.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum_eigenvalue < -tolerance {
        return Err(EffectError::IndefiniteCovariance {
            matrix: label,
            minimum_eigenvalue,
            tolerance,
        });
    }
    let active = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(column, value)| (value > tolerance).then(|| (column, value.sqrt())))
        .collect();
    Ok(PsdEigen { vectors, active })
}

fn roundoff_tolerance(scale: f64, dimension: usize) -> f64 {
    scale * f64::EPSILON * dimension.max(1) as f64
}

fn covariance_factor(eigen: &PsdEigen) -> Array2<f64> {
    let mut factor = Array2::zeros((eigen.vectors.nrows(), eigen.active.len()));
    for (active_column, &(eigen_column, eigenvalue_sqrt)) in eigen.active.iter().enumerate() {
        for row in 0..eigen.vectors.nrows() {
            factor[[row, active_column]] = eigen.vectors[[row, eigen_column]] * eigenvalue_sqrt;
        }
    }
    factor
}

fn factor_standard_errors(curve_factor: &Array2<f64>) -> Array1<f64> {
    // Each row is a covariance factor, so its variance is a sum of squares
    // and has no negative roundoff to clip. A threshold derived from OTHER
    // rows incorrectly turns a small, uncertain contrast into an exact one.
    // Hypot also keeps representable standard errors whose squares would
    // underflow or overflow.
    Array1::from_iter(
        curve_factor
            .rows()
            .into_iter()
            .map(|row| row.iter().fold(0.0_f64, |norm, &value| norm.hypot(value))),
    )
}

/// Sorted draws of `max_i |Z_i|` for the standardized Gaussian curve whose
/// covariance factor is `curve_factor`. A curve with no random direction has
/// the point mass at zero as its supremum law.
fn simulated_supremum_law(
    curve_factor: &Array2<f64>,
    se: ArrayView1<'_, f64>,
    simulations: usize,
    seed: u64,
) -> Vec<f64> {
    if curve_factor.ncols() == 0 {
        return vec![0.0; simulations];
    }

    let mut standardized_factor = curve_factor.clone();
    for row in 0..standardized_factor.nrows() {
        if se[row] == 0.0 {
            standardized_factor.row_mut(row).fill(0.0);
        } else {
            standardized_factor
                .row_mut(row)
                .mapv_inplace(|value| value / se[row]);
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut normal_coordinates = vec![0.0; curve_factor.ncols()];
    let mut maxima = Vec::with_capacity(simulations);
    for _ in 0..simulations {
        fill_standard_normals(&mut rng, &mut normal_coordinates);
        let maximum = standardized_factor
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .zip(&normal_coordinates)
                    .map(|(&loading, &coordinate)| loading * coordinate)
                    .sum::<f64>()
                    .abs()
            })
            .fold(0.0_f64, f64::max);
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    maxima
}

/// `max_i |center_i| / se_i`. A row with zero variance contributes nothing
/// when its center is zero and an unbounded deviation otherwise.
fn max_standardized_magnitude(center: &Array1<f64>, se: &Array1<f64>) -> f64 {
    center
        .iter()
        .zip(se)
        .map(|(&value, &scale)| {
            if scale > 0.0 {
                value.abs() / scale
            } else if value == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0_f64, f64::max)
}

/// Monte Carlo tail probability `(1 + #{M_s >= statistic}) / (S + 1)` of the
/// sorted supremum draws; counting the observed statistic among the draws
/// keeps the test valid at every simulation count.
fn supremum_tail_p_value(sorted_maxima: &[f64], statistic: f64) -> f64 {
    let exceed = sorted_maxima.len() - sorted_maxima.partition_point(|&draw| draw < statistic);
    (1 + exceed) as f64 / (sorted_maxima.len() + 1) as f64
}

fn fill_standard_normals(rng: &mut StdRng, output: &mut [f64]) {
    for pair in output.chunks_mut(2) {
        // The complement of a [0, 1) draw lies in (0, 1], so `ln` is finite.
        let uniform_radius = 1.0 - rng.random::<f64>();
        let uniform_angle = rng.random::<f64>();
        let radius = (-2.0 * uniform_radius.ln()).sqrt();
        let angle = std::f64::consts::TAU * uniform_angle;
        pair[0] = radius * angle.cos();
        if pair.len() == 2 {
            pair[1] = radius * angle.sin();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn closed_form_centers_and_standard_errors() {
        let beta = array![2.0, -1.0];
        let covariance = array![[4.0, 1.0], [1.0, 9.0]];
        let contrast = array![[1.0, 0.0], [1.0, 2.0]];

        let report = effect_report(
            beta.view(),
            covariance.view(),
            contrast.view(),
            BandOptions::default(),
        )
        .unwrap();

        assert_abs_diff_eq!(report.center[0], 2.0, epsilon = 1e-14);
        assert_abs_diff_eq!(report.center[1], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(report.se[0], 2.0, epsilon = 1e-14);
        assert_abs_diff_eq!(report.se[1], 44.0_f64.sqrt(), epsilon = 1e-13);
    }

    #[test]
    fn singular_psd_simulation_is_reproducible() {
        let beta = array![0.5, -0.5];
        let covariance = array![[1.0, 1.0], [1.0, 1.0]];
        let contrast = array![[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]];
        let options = BandOptions::Simultaneous(SimultaneousBandOptions {
            simulations: 2_000,
            ..SimultaneousBandOptions::default()
        });

        let first =
            effect_report(beta.view(), covariance.view(), contrast.view(), options).unwrap();
        let second =
            effect_report(beta.view(), covariance.view(), contrast.view(), options).unwrap();

        assert_eq!(first, second);
        assert_abs_diff_eq!(first.se[0], 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(first.se[1], 1.0, epsilon = 1e-14);
        assert_eq!(first.se[2], 0.0);
        assert!(first.critical.is_finite());
    }

    #[test]
    fn materially_indefinite_covariance_is_rejected() {
        let error = effect_report(
            array![0.0, 0.0].view(),
            array![[1.0, 0.0], [0.0, -0.1]].view(),
            array![[0.0, 1.0]].view(),
            BandOptions::default(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EffectError::IndefiniteCovariance {
                matrix: "projected curve covariance",
                ..
            }
        ));
    }

    #[test]
    fn pointwise_band_uses_two_sided_normal_critical_value() {
        let report = effect_report(
            array![0.0].view(),
            array![[1.0]].view(),
            array![[1.0]].view(),
            BandOptions::default(),
        )
        .unwrap();

        // True two-sided 95% normal critical value z_{0.975} = 1.9599639845400545.
        // The former golden 1.959963986120195 was the RAW Acklam-approximation
        // output (absolute error ~1.6e-9); the quantile now carries a two-round
        // Halley refinement against erfc and returns the true value, so the
        // golden pins the mathematically correct quantile at a tightened
        // tolerance.
        assert_abs_diff_eq!(report.critical, 1.959_963_984_540_054, epsilon = 1e-9);
    }

    #[test]
    fn simultaneous_critical_is_beta_and_contrast_sign_invariant() {
        let covariance = array![[2.0, 0.25], [0.25, 1.0]];
        let contrast = array![[1.0, 0.5], [-0.25, 1.0]];
        let options = BandOptions::Simultaneous(SimultaneousBandOptions {
            simulations: 1_000,
            ..SimultaneousBandOptions::default()
        });
        let first = effect_report(
            array![1.0, -2.0].view(),
            covariance.view(),
            contrast.view(),
            options,
        )
        .unwrap();
        let shifted = effect_report(
            array![8.0, 3.0].view(),
            covariance.view(),
            contrast.view(),
            options,
        )
        .unwrap();
        let signed = effect_report(
            array![1.0, -2.0].view(),
            covariance.view(),
            (-&contrast).view(),
            options,
        )
        .unwrap();

        assert_eq!(first.critical, shifted.critical);
        assert_eq!(first.critical, signed.critical);
        assert_eq!(first.se, shifted.se);
        assert_eq!(first.se, signed.se);
        for row in 0..contrast.nrows() {
            assert_abs_diff_eq!(signed.center[row], -first.center[row], epsilon = 1e-14);
            assert_abs_diff_eq!(signed.lower[row], -first.upper[row], epsilon = 1e-14);
            assert_abs_diff_eq!(signed.upper[row], -first.lower[row], epsilon = 1e-14);
        }
    }

    #[test]
    fn simultaneous_bands_preserve_small_nonzero_contrast_scales() {
        let report = effect_report(
            array![0.0].view(),
            array![[1.0]].view(),
            array![[1.0], [1e-9]].view(),
            BandOptions::Simultaneous(SimultaneousBandOptions {
                simulations: 64,
                ..SimultaneousBandOptions::default()
            }),
        )
        .expect("scaled contrasts of the same Gaussian coefficient");
        assert_eq!(report.se[0], 1.0);
        assert_eq!(report.se[1], 1e-9);
        assert!(report.upper[1] > 0.0);
        assert_abs_diff_eq!(report.upper[1] / report.upper[0], 1e-9, epsilon = 1e-24);
    }

    /// A smooth-curve contrast: cosine basis on a 50-point grid with an
    /// AR(1) coefficient covariance, so neighbouring rows are strongly but
    /// not perfectly correlated, as for a fitted spline difference.
    fn correlated_curve_fixture() -> (Array2<f64>, Array2<f64>) {
        let rows = 50;
        let columns = 8;
        let contrast = Array2::from_shape_fn((rows, columns), |(row, column)| {
            let x = row as f64 / (rows - 1) as f64;
            (std::f64::consts::PI * column as f64 * x).cos()
        });
        let covariance = Array2::from_shape_fn((columns, columns), |(i, j)| {
            0.04 * 0.6_f64.powi((i as i32 - j as i32).abs())
        });
        (contrast, covariance)
    }

    /// Independent standardized curve deviations `C (beta_hat - beta) / se`
    /// drawn from the fixture's exact Gaussian law, from a stream seeded apart
    /// from the band's calibration draws.
    fn independent_standardized_maxima(
        contrast: &Array2<f64>,
        covariance: &Array2<f64>,
        se: &Array1<f64>,
        replicates: usize,
        seed: u64,
    ) -> Vec<f64> {
        let eigen = psd_eigendecomposition(covariance.view(), "fixture covariance").unwrap();
        let curve_factor = contrast.dot(&covariance_factor(&eigen));
        let mut rng = StdRng::seed_from_u64(seed);
        let mut coordinates = vec![0.0; curve_factor.ncols()];
        (0..replicates)
            .map(|_| {
                fill_standard_normals(&mut rng, &mut coordinates);
                let deviation = curve_factor.dot(&Array1::from_vec(coordinates.clone()));
                max_standardized_magnitude(&deviation, se)
            })
            .collect()
    }

    #[test]
    fn simultaneous_band_covers_the_whole_curve_at_its_nominal_rate() {
        let (contrast, covariance) = correlated_curve_fixture();
        let beta = Array1::zeros(covariance.nrows());
        let level = 0.95;
        let simultaneous = effect_report(
            beta.view(),
            covariance.view(),
            contrast.view(),
            BandOptions::Simultaneous(SimultaneousBandOptions {
                level,
                ..SimultaneousBandOptions::default()
            }),
        )
        .unwrap();
        let pointwise = effect_report(
            beta.view(),
            covariance.view(),
            contrast.view(),
            BandOptions::Pointwise(PointwiseBandOptions { level }),
        )
        .unwrap();
        for (&factor_se, &quadratic_se) in simultaneous.se.iter().zip(&pointwise.se) {
            assert_abs_diff_eq!(factor_se, quadratic_se, epsilon = 1e-12);
        }

        let replicates = 20_000;
        let maxima = independent_standardized_maxima(
            &contrast,
            &covariance,
            &simultaneous.se,
            replicates,
            20_260_919,
        );
        let coverage = |critical: f64| {
            maxima.iter().filter(|&&maximum| maximum <= critical).count() as f64
                / replicates as f64
        };
        // Coverage error from the replicate count and from the calibration's
        // own quantile estimate at the default simulation count.
        let mcse = (level * (1.0 - level) / replicates as f64
            + level * (1.0 - level) / DEFAULT_SIMULATIONS as f64)
            .sqrt();
        let whole_curve = coverage(simultaneous.critical);
        assert!(
            (whole_curve - level).abs() <= 2.0 * mcse,
            "simultaneous whole-curve coverage {whole_curve} vs {level} (2 MCSE = {})",
            2.0 * mcse
        );
        // The pointwise critical value under-covers the whole curve, which is
        // what the simultaneous calibration exists to fix.
        let pointwise_curve = coverage(pointwise.critical);
        assert!(
            pointwise_curve < level - 10.0 * mcse,
            "pointwise whole-curve coverage {pointwise_curve} should fall well short of {level}"
        );
        assert!(simultaneous.critical > pointwise.critical);
    }

    #[test]
    fn zero_curve_p_value_is_uniform_under_the_null_and_matches_the_band() {
        let (contrast, covariance) = correlated_curve_fixture();
        let beta = Array1::zeros(covariance.nrows());
        let options = SimultaneousBandOptions::default();
        let report = effect_report(
            beta.view(),
            covariance.view(),
            contrast.view(),
            BandOptions::Simultaneous(options),
        )
        .unwrap();
        // An exactly-zero estimated curve carries no evidence against zero.
        assert_eq!(report.zero_curve_p_value, Some(1.0));

        let curve_factor = simultaneous_curve_factor(contrast.view(), covariance.view()).unwrap();
        let law = simulated_supremum_law(
            &curve_factor,
            report.se.view(),
            options.simulations,
            options.seed,
        );
        let replicates = 20_000;
        let mut p_values: Vec<f64> = independent_standardized_maxima(
            &contrast,
            &covariance,
            &report.se,
            replicates,
            7_031_966,
        )
        .into_iter()
        .map(|statistic| supremum_tail_p_value(&law, statistic))
        .collect();
        for alpha in [0.10, 0.05, 0.01] {
            let size = p_values.iter().filter(|&&p| p <= alpha).count() as f64 / replicates as f64;
            let mcse = (alpha * (1.0 - alpha) / replicates as f64
                + alpha * (1.0 - alpha) / options.simulations as f64)
                .sqrt();
            assert!(
                (size - alpha).abs() <= 2.0 * mcse,
                "size at {alpha}: {size} (2 MCSE = {})",
                2.0 * mcse
            );
        }
        p_values.sort_by(f64::total_cmp);
        let ks = p_values
            .iter()
            .enumerate()
            .map(|(index, &p)| {
                let upper = (index + 1) as f64 / replicates as f64 - p;
                let lower = p - index as f64 / replicates as f64;
                upper.max(lower)
            })
            .fold(0.0_f64, f64::max);
        // Kolmogorov 1% critical value 1.628 / sqrt(R), widened by the
        // calibration law's own sup-norm error at S draws.
        let bound = 1.628 * (1.0 / replicates as f64).sqrt()
            + 1.628 * (1.0 / options.simulations as f64).sqrt();
        assert!(ks <= bound, "KS distance {ks} exceeds {bound}");

        // Duality with the band: a curve the 95% band excludes zero from has
        // p <= 0.05, and one it does not has p > 0.05.
        // The intercept column is constant one, so shifting it moves every row
        // by the same amount and the statistic is `shift / min_i se_i`.
        let minimum_se = report.se.iter().copied().fold(f64::INFINITY, f64::min);
        for (multiple, rejects) in [(1.2, true), (0.8, false)] {
            let mut shifted = beta.clone();
            shifted[0] = multiple * report.critical * minimum_se;
            let shifted_report = effect_report(
                shifted.view(),
                covariance.view(),
                contrast.view(),
                BandOptions::Simultaneous(options),
            )
            .unwrap();
            let excludes_zero = shifted_report
                .lower
                .iter()
                .zip(&shifted_report.upper)
                .any(|(&lower, &upper)| lower > 0.0 || upper < 0.0);
            let p_value = shifted_report.zero_curve_p_value.unwrap();
            assert_eq!(excludes_zero, p_value <= 0.05, "multiple {multiple}: p = {p_value}");
            assert_eq!(excludes_zero, rejects);
        }
    }

    #[test]
    fn pointwise_bands_carry_no_zero_curve_p_value() {
        let report = effect_report(
            array![1.0].view(),
            array![[1.0]].view(),
            array![[1.0]].view(),
            BandOptions::default(),
        )
        .unwrap();
        assert_eq!(report.zero_curve_p_value, None);
    }

    #[test]
    fn factor_standard_errors_do_not_square_outside_the_float_range() {
        let se = factor_standard_errors(&array![[3e200, 4e200], [3e-200, 4e-200]]);
        assert_abs_diff_eq!(se[0] / 5e200, 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(se[1] / 5e-200, 1.0, epsilon = 1e-14);
    }
}
