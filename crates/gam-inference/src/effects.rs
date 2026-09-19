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
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::error::Error;
use std::fmt;

/// Default confidence level for effect bands.
pub(crate) const DEFAULT_BAND_LEVEL: f64 = 0.95;
/// Default deterministic random seed for simultaneous bands.
pub const DEFAULT_SIMULATION_SEED: u64 = 12_345;

/// Target Monte Carlo error of a simultaneous band's attained miss rate,
/// relative to the nominal miss rate `alpha = 1 - level`.
///
/// A simulated band's critical value is the `ceil(level * N)`-th order statistic
/// of `N` simulated suprema, so its attained coverage `F(c_hat)` is exactly
/// `Beta(k, N + 1 - k)` for the supremum's continuous CDF `F`, whatever the grid
/// or covariance: mean `level` and standard deviation `sqrt(level * alpha / N)`
/// to first order. Asking that standard deviation to be this fraction of
/// `alpha` fixes `N` (see [`simultaneous_band_simulations`]); at 5%, two Monte
/// Carlo standard deviations keep the attained miss rate within a tenth of
/// nominal.
pub const SIMULTANEOUS_BAND_RELATIVE_MC_ERROR: f64 = 0.05;

/// Monte Carlo draw count whose attained-coverage standard deviation is
/// [`SIMULTANEOUS_BAND_RELATIVE_MC_ERROR`] times the nominal miss rate:
/// `N = ceil(level / (alpha * delta^2))`, which is 7,600 draws at 95% and
/// 39,600 at 99%.
pub fn simultaneous_band_simulations(level: f64) -> Result<usize, EffectError> {
    validate_level(level)?;
    let miss_rate = 1.0 - level;
    let delta = SIMULTANEOUS_BAND_RELATIVE_MC_ERROR;
    let draws = (level / (miss_rate * delta * delta)).ceil();
    if !draws.is_finite() || draws > usize::MAX as f64 {
        return Err(EffectError::InvalidLevel { level });
    }
    Ok(draws as usize)
}

fn validate_level(level: f64) -> Result<(), EffectError> {
    if !level.is_finite() || !(0.0..1.0).contains(&level) || level == 0.0 {
        return Err(EffectError::InvalidLevel { level });
    }
    Ok(())
}

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

impl SimultaneousBandOptions {
    /// A band at `level` with the Monte Carlo size its target error implies.
    pub fn at_level(level: f64) -> Result<Self, EffectError> {
        Ok(Self {
            level,
            simulations: simultaneous_band_simulations(level)?,
            seed: DEFAULT_SIMULATION_SEED,
        })
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
    let (se, critical) = match options {
        BandOptions::Pointwise(pointwise) => {
            let se = pointwise_standard_errors(contrast_design, covariance.view())?;
            let critical = standard_normal_quantile(0.5 * (1.0 + pointwise.level))
                .map_err(|detail| EffectError::NormalQuantile { detail })?;
            (se, critical)
        }
        BandOptions::Simultaneous(simultaneous) => {
            let curve_factor = simultaneous_curve_factor(contrast_design, covariance.view())?;
            let se = factor_standard_errors(&curve_factor);
            let critical = simultaneous_critical(
                &curve_factor,
                se.view(),
                simultaneous.level,
                simultaneous.simulations,
                simultaneous.seed,
            );
            (se, critical)
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
    })
}

/// Pointwise and simultaneous bands for one linear effect curve at one level.
///
/// Both bands share the center `C beta` and the standard errors of `C V C'`;
/// they differ only in the critical value. The pointwise one is the two-sided
/// normal quantile. The simultaneous one is the `level` quantile of
/// `max_i |Z_i|` for the standardized Gaussian curve `Z`, simulated with
/// [`simultaneous_band_simulations`] draws, so the band covers the whole curve
/// over the rows of `C` jointly.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectBands {
    pub fit: Array1<f64>,
    pub se: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub simultaneous_lower: Array1<f64>,
    pub simultaneous_upper: Array1<f64>,
    pub level: f64,
    pub pointwise_critical: f64,
    pub simultaneous_critical: f64,
    /// The Monte Carlo draws behind `simultaneous_critical`.
    pub simulations: usize,
    /// The random seed behind `simultaneous_critical`.
    pub seed: u64,
}

/// Compute [`EffectBands`] for contrast design `C`, coefficients `beta`, and
/// covariance `V` at `level`.
pub fn effect_bands(
    beta: ArrayView1<'_, f64>,
    covariance: ArrayView2<'_, f64>,
    contrast_design: ArrayView2<'_, f64>,
    level: f64,
) -> Result<EffectBands, EffectError> {
    let options = SimultaneousBandOptions::at_level(level)?;
    validate_inputs(
        beta,
        covariance,
        contrast_design,
        BandOptions::Simultaneous(options),
    )?;
    let covariance = validated_symmetric_matrix(covariance)?;
    let fit = contrast_design.dot(&beta);
    let curve_factor = simultaneous_curve_factor(contrast_design, covariance.view())?;
    let se = factor_standard_errors(&curve_factor);
    let pointwise_critical = standard_normal_quantile(0.5 * (1.0 + level))
        .map_err(|detail| EffectError::NormalQuantile { detail })?;
    let simultaneous_critical = simultaneous_critical(
        &curve_factor,
        se.view(),
        level,
        options.simulations,
        options.seed,
    );
    let band = |critical: f64| {
        let half_width = se.mapv(|value| critical * value);
        (&fit - &half_width, &fit + &half_width)
    };
    let (lower, upper) = band(pointwise_critical);
    let (simultaneous_lower, simultaneous_upper) = band(simultaneous_critical);
    Ok(EffectBands {
        fit,
        se,
        lower,
        upper,
        simultaneous_lower,
        simultaneous_upper,
        level,
        pointwise_critical,
        simultaneous_critical,
        simulations: options.simulations,
        seed: options.seed,
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
    validate_level(level)?;
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

fn simultaneous_critical(
    curve_factor: &Array2<f64>,
    se: ArrayView1<'_, f64>,
    level: f64,
    simulations: usize,
    seed: u64,
) -> f64 {
    if curve_factor.ncols() == 0 {
        return 0.0;
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

    // Draws are generated one at a time, so the stream (and hence the result)
    // does not depend on the block size; each block of draws is pushed through
    // the factor as one matrix product.
    let rank = curve_factor.ncols();
    let block = gam_runtime::resource::byte_balanced_row_chunk(
        standardized_factor.nrows(),
        simulations,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut draws = Array2::<f64>::zeros((block, rank));
    let mut maxima = Vec::with_capacity(simulations);
    while maxima.len() < simulations {
        let rows = block.min(simulations - maxima.len());
        for mut draw in draws.rows_mut().into_iter().take(rows) {
            let coordinates = draw
                .as_slice_mut()
                .expect("a row of a standard-layout array is contiguous");
            fill_standard_normals(&mut rng, coordinates);
        }
        let curves = draws
            .slice(ndarray::s![..rows, ..])
            .dot(&standardized_factor.t());
        maxima.extend(
            curves
                .rows()
                .into_iter()
                .map(|curve| curve.iter().fold(0.0_f64, |maximum, value| maximum.max(value.abs()))),
        );
    }
    maxima.sort_by(f64::total_cmp);
    quantile_from_sorted(&maxima, level)
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
            ..SimultaneousBandOptions::at_level(DEFAULT_BAND_LEVEL).unwrap()
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
            ..SimultaneousBandOptions::at_level(DEFAULT_BAND_LEVEL).unwrap()
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
                ..SimultaneousBandOptions::at_level(DEFAULT_BAND_LEVEL).unwrap()
            }),
        )
        .expect("scaled contrasts of the same Gaussian coefficient");
        assert_eq!(report.se[0], 1.0);
        assert_eq!(report.se[1], 1e-9);
        assert!(report.upper[1] > 0.0);
        assert_abs_diff_eq!(report.upper[1] / report.upper[0], 1e-9, epsilon = 1e-24);
    }

    #[test]
    fn simulation_count_meets_the_relative_mc_error_target() {
        assert_eq!(simultaneous_band_simulations(0.95).unwrap(), 7_600);
        assert_eq!(simultaneous_band_simulations(0.99).unwrap(), 39_600);
        for level in [0.8, 0.9, 0.95, 0.99, 0.999] {
            let n = simultaneous_band_simulations(level).unwrap() as f64;
            // Attained coverage is Beta(k, N + 1 - k) with k ~ level * N, so
            // its standard deviation relative to the miss rate is at most the target.
            let relative = (level * (1.0 - level) / n).sqrt() / (1.0 - level);
            assert!(relative <= SIMULTANEOUS_BAND_RELATIVE_MC_ERROR + 1e-12);
        }
        for level in [0.0, 1.0, -0.5, f64::NAN] {
            assert!(simultaneous_band_simulations(level).is_err());
        }
    }

    #[test]
    fn simultaneous_band_attains_its_exact_coverage_on_independent_rows() {
        // For m independent standard normal rows, P(max |Z_i| <= c) = (2 Phi(c) - 1)^m
        // exactly, so the attained coverage of the simulated critical value is
        // checkable against the level within the Monte Carlo error the draw count targets.
        let m = 12;
        let level = 0.95;
        let bands = effect_bands(
            Array1::zeros(m).view(),
            Array2::eye(m).view(),
            Array2::eye(m).view(),
            level,
        )
        .unwrap();
        let attained =
            (2.0 * gam_math::probability::normal_cdf(bands.simultaneous_critical) - 1.0)
                .powi(m as i32);
        let mc_sd = (level * (1.0 - level) / bands.simulations as f64).sqrt();
        assert!(
            (attained - level).abs() <= 4.0 * mc_sd,
            "attained {attained} at critical {}",
            bands.simultaneous_critical
        );
        assert_eq!(bands.simulations, simultaneous_band_simulations(level).unwrap());
        assert_abs_diff_eq!(bands.pointwise_critical, 1.959_963_984_540_054, epsilon = 1e-9);
        assert!(bands.simultaneous_critical > bands.pointwise_critical);
        for row in 0..m {
            assert!(bands.simultaneous_lower[row] < bands.lower[row]);
            assert!(bands.simultaneous_upper[row] > bands.upper[row]);
        }
    }

    #[test]
    fn a_perfectly_correlated_curve_needs_only_the_pointwise_critical_value() {
        // Every row is a positive multiple of one coefficient, so the maximum
        // of the standardized curve is a single |Z| and the simultaneous and
        // pointwise critical values agree up to Monte Carlo error.
        let level = 0.95;
        let bands = effect_bands(
            array![0.3].view(),
            array![[2.0]].view(),
            array![[1.0], [2.0], [0.5]].view(),
            level,
        )
        .unwrap();
        let attained = 2.0 * gam_math::probability::normal_cdf(bands.simultaneous_critical) - 1.0;
        let mc_sd = (level * (1.0 - level) / bands.simulations as f64).sqrt();
        assert!((attained - level).abs() <= 4.0 * mc_sd);
        let again = effect_bands(
            array![0.3].view(),
            array![[2.0]].view(),
            array![[1.0], [2.0], [0.5]].view(),
            level,
        )
        .unwrap();
        assert_eq!(bands, again);
    }

    #[test]
    fn factor_standard_errors_do_not_square_outside_the_float_range() {
        let se = factor_standard_errors(&array![[3e200, 4e200], [3e-200, 4e-200]]);
        assert_abs_diff_eq!(se[0] / 5e200, 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(se[1] / 5e-200, 1.0, epsilon = 1e-14);
    }
}
