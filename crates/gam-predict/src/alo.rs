use std::{ops::Range, sync::Arc};

use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix, ReparamOperator};
use gam_models::gamlss::{
    BinomialLocationScaleAloRowInput, DispersionFamilyKind, GaussianLocationScaleAloRowInput,
    binomial_location_scale_alo_row_geometry, dispersion_alo_row_geometry,
    gaussian_location_scale_alo_row_geometry,
};
use gam_models::inference::model::{
    FittedModel, PredictModelClass, binomial_location_scale_threshold_beta,
    gaussian_location_scale_mean_beta, location_scale_noise_beta,
};
use gam_models::survival::{
    CauseSpecificSurvivalAloRowInput, SurvivalLikelihoodMode,
    SurvivalLocationScaleAloRowInput, SurvivalLocationScaleAloTimeWiggleInput,
    SurvivalLocationScaleAloWiggleInput, SurvivalMarginalSlopeSavedAloReplayInput,
    cause_specific_survival_alo_row_geometry, replay_saved_survival_marginal_slope_alo,
    require_saved_survival_likelihood_mode, survival_event_code_from_value,
    survival_location_scale_alo_row_geometry,
    survival_location_scale_time_wiggle_basis_authority,
};
use gam_models::transformation_normal::{
    TRANSFORMATION_MONOTONICITY_EPS, TransformationNormalAloRowInput,
    transformation_normal_alo_row_geometry,
};
use gam_problem::{BlockRole, EstimationError, Gauge, ResponseFamily};
use gam_solve::inference::alo::{
    AloInput, MultiBlockAloDiagnostics, MultiBlockAloInput, compute_alo_from_input,
    compute_multiblock_alo,
};
use gam_spec::{GlmLikelihoodSpec, LinkFunction};
use gam_terms::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense,
};
use gam_terms::smooth::{LinearCoefficientGeometry, TermCollectionSpec};
use ndarray::{Array1, Array2, s};

use crate::{FittedModelPredictExt, PredictInput};

/// Observed row data needed to replay a saved likelihood for ALO.
pub struct SavedAloObservations<'a> {
    pub response: &'a Array1<f64>,
    pub prior_weights: &'a Array1<f64>,
}

/// Exact affine row carrier for a saved cause-specific transformation/Weibull
/// survival likelihood.
///
/// Coordinates are flattened cause-major as
/// `[eta_exit, eta_entry, derivative_exit]` for each cause. Every coordinate
/// carries its own parameter-aligned design and coefficient range; the three
/// ranges for one cause intentionally overlap because all three predictors are
/// affine functions of the same fitted endpoint block.
pub struct SavedCauseSpecificSurvivalAloInput {
    event_codes: Array1<u8>,
    entry_active: Vec<bool>,
    coordinate_designs: Vec<DesignMatrix>,
    coordinate_offsets: Vec<Array1<f64>>,
    coordinate_ranges: Vec<Range<usize>>,
    cause_count: usize,
}

/// Exact row channels needed to reconstruct a fitted survival
/// marginal-slope likelihood.  The three time designs deliberately share the
/// same coefficient block; their time-wiggle tail is the fit-time zero
/// placeholder because the nonlinear basis evaluation is replayed by the
/// family row program itself.
pub struct SavedMarginalSlopeSurvivalAloInput {
    event: Array1<f64>,
    latent_z: Array1<f64>,
    design_entry: DesignMatrix,
    design_exit: DesignMatrix,
    design_derivative_exit: DesignMatrix,
    offset_entry: Array1<f64>,
    offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
    marginal_design: DesignMatrix,
    marginal_offset: Array1<f64>,
    logslope_design: DesignMatrix,
    logslope_offset: Array1<f64>,
}

/// One fitted affine block evaluated on survival entry, exit, and exit-time
/// derivative channels.
///
/// Every design has the same raw coefficient width. A structurally static
/// block uses its exit design at entry and an exact-zero derivative design;
/// callers must make that topology explicit instead of relying on a replay
/// fallback inside the ALO likelihood.
pub struct SavedSurvivalAffineBlockAloInput {
    design_entry: DesignMatrix,
    design_exit: DesignMatrix,
    design_derivative_exit: DesignMatrix,
    offset_entry: Array1<f64>,
    offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
}

impl SavedSurvivalAffineBlockAloInput {
    pub fn new(
        design_entry: DesignMatrix,
        design_exit: DesignMatrix,
        design_derivative_exit: DesignMatrix,
        offset_entry: Array1<f64>,
        offset_exit: Array1<f64>,
        derivative_offset_exit: Array1<f64>,
    ) -> Result<Self, String> {
        let n = design_exit.nrows();
        let width = design_exit.ncols();
        if n == 0
            || design_entry.nrows() != n
            || design_derivative_exit.nrows() != n
            || offset_entry.len() != n
            || offset_exit.len() != n
            || derivative_offset_exit.len() != n
        {
            return Err(format!(
                "saved survival affine ALO block row mismatch: entry={}, exit={n}, derivative={}, offsets={}/{}/{}",
                design_entry.nrows(),
                design_derivative_exit.nrows(),
                offset_entry.len(),
                offset_exit.len(),
                derivative_offset_exit.len(),
            ));
        }
        if width == 0
            || design_entry.ncols() != width
            || design_derivative_exit.ncols() != width
        {
            return Err(format!(
                "saved survival affine ALO block width mismatch: entry={}, exit={width}, derivative={}",
                design_entry.ncols(),
                design_derivative_exit.ncols(),
            ));
        }
        for (label, values) in [
            ("entry offset", &offset_entry),
            ("exit offset", &offset_exit),
            ("derivative offset", &derivative_offset_exit),
        ] {
            if let Some((row, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "saved survival affine ALO {label}[{row}] must be finite, got {value}"
                ));
            }
        }
        Ok(Self {
            design_entry,
            design_exit,
            design_derivative_exit,
            offset_entry,
            offset_exit,
            derivative_offset_exit,
        })
    }

    fn evaluate(
        &self,
        beta: &Array1<f64>,
        label: &str,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        if self.design_exit.ncols() != beta.len() {
            return Err(invalid(format!(
                "saved survival {label} ALO design has {} columns; fitted affine block has {} coefficients",
                self.design_exit.ncols(),
                beta.len(),
            )));
        }
        Ok((
            self.design_entry.dot(beta) + &self.offset_entry,
            self.design_exit.dot(beta) + &self.offset_exit,
            self.design_derivative_exit.dot(beta) + &self.derivative_offset_exit,
        ))
    }
}

/// Exact saved-row carrier for a survival location-scale fit.
///
/// `time_base` contains only the affine pre-time-wiggle coefficients. The
/// baseline time-wiggle tail is deliberately reconstructed by the fitting row
/// program, alongside the link wiggle, so its score-times-map-Hessian term is
/// retained in observed row curvature.
pub struct SavedLocationScaleSurvivalAloInput {
    event: Array1<f64>,
    derivative_guard: f64,
    time_base: SavedSurvivalAffineBlockAloInput,
    threshold: SavedSurvivalAffineBlockAloInput,
    log_sigma: SavedSurvivalAffineBlockAloInput,
}

impl SavedLocationScaleSurvivalAloInput {
    pub fn new(
        event: Array1<f64>,
        derivative_guard: f64,
        time_base: SavedSurvivalAffineBlockAloInput,
        threshold: SavedSurvivalAffineBlockAloInput,
        log_sigma: SavedSurvivalAffineBlockAloInput,
    ) -> Result<Self, String> {
        let n = event.len();
        if n == 0
            || time_base.design_exit.nrows() != n
            || threshold.design_exit.nrows() != n
            || log_sigma.design_exit.nrows() != n
        {
            return Err(format!(
                "saved survival location-scale ALO row mismatch: event={n}, time={}, threshold={}, log_sigma={}",
                time_base.design_exit.nrows(),
                threshold.design_exit.nrows(),
                log_sigma.design_exit.nrows(),
            ));
        }
        if !derivative_guard.is_finite() || derivative_guard <= 0.0 {
            return Err(format!(
                "saved survival location-scale ALO derivative guard must be positive and finite, got {derivative_guard}"
            ));
        }
        if let Some((row, value)) = event
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value != 0.0 && *value != 1.0)
        {
            return Err(format!(
                "saved survival location-scale ALO event[{row}] must be exactly 0 or 1, got {value}"
            ));
        }
        Ok(Self {
            event,
            derivative_guard,
            time_base,
            threshold,
            log_sigma,
        })
    }
}

impl SavedMarginalSlopeSurvivalAloInput {
    pub fn new(
        event: Array1<f64>,
        latent_z: Array1<f64>,
        design_entry: DesignMatrix,
        design_exit: DesignMatrix,
        design_derivative_exit: DesignMatrix,
        offset_entry: Array1<f64>,
        offset_exit: Array1<f64>,
        derivative_offset_exit: Array1<f64>,
        marginal_design: DesignMatrix,
        marginal_offset: Array1<f64>,
        logslope_design: DesignMatrix,
        logslope_offset: Array1<f64>,
    ) -> Result<Self, String> {
        let n = event.len();
        if n == 0
            || latent_z.len() != n
            || offset_entry.len() != n
            || offset_exit.len() != n
            || derivative_offset_exit.len() != n
            || marginal_offset.len() != n
            || logslope_offset.len() != n
            || design_entry.nrows() != n
            || design_exit.nrows() != n
            || design_derivative_exit.nrows() != n
            || marginal_design.nrows() != n
            || logslope_design.nrows() != n
        {
            return Err(
                "saved survival marginal-slope ALO row channels are not aligned".to_string(),
            );
        }
        if design_entry.ncols() == 0
            || design_entry.ncols() != design_exit.ncols()
            || design_entry.ncols() != design_derivative_exit.ncols()
            || marginal_design.ncols() == 0
            || logslope_design.ncols() == 0
        {
            return Err(format!(
                "saved survival marginal-slope ALO design topology is invalid: time entry/exit/derivative={}/{}/{}, marginal={}, logslope={}",
                design_entry.ncols(),
                design_exit.ncols(),
                design_derivative_exit.ncols(),
                marginal_design.ncols(),
                logslope_design.ncols(),
            ));
        }
        if let Some((row, value)) = event
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value != 0.0 && *value != 1.0)
        {
            return Err(format!(
                "saved survival marginal-slope ALO event[{row}] must be exactly 0 or 1, got {value}"
            ));
        }
        Ok(Self {
            event,
            latent_z,
            design_entry,
            design_exit,
            design_derivative_exit,
            offset_entry,
            offset_exit,
            derivative_offset_exit,
            marginal_design,
            marginal_offset,
            logslope_design,
            logslope_offset,
        })
    }
}

impl SavedCauseSpecificSurvivalAloInput {
    pub fn new(
        event_codes: Array1<u8>,
        entry_active: Vec<bool>,
        coordinate_designs: Vec<DesignMatrix>,
        coordinate_offsets: Vec<Array1<f64>>,
        coordinate_ranges: Vec<Range<usize>>,
        cause_count: usize,
    ) -> Result<Self, String> {
        if cause_count == 0 {
            return Err("saved cause-specific ALO requires at least one cause".to_string());
        }
        let n = event_codes.len();
        if n == 0 {
            return Err("saved cause-specific ALO requires at least one row".to_string());
        }
        if entry_active.len() != n {
            return Err(format!(
                "saved cause-specific ALO entry activity has {} rows; expected {n}",
                entry_active.len()
            ));
        }
        let coordinate_count = cause_count.checked_mul(3).ok_or_else(|| {
            "saved cause-specific ALO coordinate count overflows usize".to_string()
        })?;
        if coordinate_designs.len() != coordinate_count
            || coordinate_offsets.len() != coordinate_count
            || coordinate_ranges.len() != coordinate_count
        {
            return Err(format!(
                "saved cause-specific ALO requires {coordinate_count} coordinate designs/offsets/ranges for {cause_count} causes; got designs={}, offsets={}, ranges={}",
                coordinate_designs.len(),
                coordinate_offsets.len(),
                coordinate_ranges.len()
            ));
        }
        for coordinate in 0..coordinate_count {
            let design = &coordinate_designs[coordinate];
            let offset = &coordinate_offsets[coordinate];
            let range = &coordinate_ranges[coordinate];
            if design.nrows() != n || offset.len() != n {
                return Err(format!(
                    "saved cause-specific ALO coordinate {coordinate} row mismatch: design={}, offset={}, expected={n}",
                    design.nrows(),
                    offset.len()
                ));
            }
            if range.is_empty() || design.ncols() != range.len() {
                return Err(format!(
                    "saved cause-specific ALO coordinate {coordinate} has design width {} and coefficient range {:?}",
                    design.ncols(),
                    range
                ));
            }
        }
        if let Some((row, code)) = event_codes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| usize::from(*code) > cause_count)
        {
            return Err(format!(
                "saved cause-specific ALO event code[{row}]={code} exceeds fitted cause count {cause_count}"
            ));
        }
        Ok(Self {
            event_codes,
            entry_active,
            coordinate_designs,
            coordinate_offsets,
            coordinate_ranges,
            cause_count,
        })
    }
}

/// Typed survival row carriers accepted by saved-model ALO.
pub enum SavedSurvivalAloInput {
    CauseSpecific(SavedCauseSpecificSurvivalAloInput),
    LocationScale(SavedLocationScaleSurvivalAloInput),
    MarginalSlope(SavedMarginalSlopeSurvivalAloInput),
}

/// Class-aware saved-model ALO input.
///
/// The affine predictor carrier cannot represent survival's entry/exit/time
/// derivative row map. Making that distinction explicit prevents a survival
/// model from being silently forced through a simpler prediction surface.
pub enum SavedModelAloInput {
    Affine(PredictInput),
    Survival(SavedSurvivalAloInput),
}

impl SavedModelAloInput {
    pub fn affine(input: PredictInput) -> Self {
        Self::Affine(input)
    }

    pub fn survival(input: SavedSurvivalAloInput) -> Self {
        Self::Survival(input)
    }

    pub fn require_affine(&self, class: PredictModelClass) -> Result<&PredictInput, String> {
        match self {
            Self::Affine(input) => Ok(input),
            Self::Survival(_) => Err(format!(
                "saved {} ALO requires an affine predictor carrier, not survival row geometry",
                class.name()
            )),
        }
    }
}

/// Class-neutral diagnostics returned by saved-model multi-coordinate ALO.
#[derive(Debug, Clone)]
pub struct SavedModelAloDiagnostics {
    pub model_class: PredictModelClass,
    /// Names of the affine local coordinates represented by each entry of
    /// `diagnostics.eta_tilde` / `diagnostics.alo_variance`.
    pub coordinate_names: Vec<String>,
    pub diagnostics: MultiBlockAloDiagnostics,
}

fn invalid(reason: impl Into<String>) -> EstimationError {
    EstimationError::InvalidInput(reason.into())
}

fn termspec_has_bounded_terms(spec: &TermCollectionSpec) -> bool {
    spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    })
}

fn require_location_scale_inputs<'a>(
    class: PredictModelClass,
    input: &'a PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<&'a DesignMatrix, EstimationError> {
    let n = observations.response.len();
    if n == 0 {
        return Err(invalid(format!(
            "saved {} ALO requires at least one observation",
            class.name()
        )));
    }
    if observations.prior_weights.len() != n || input.design.nrows() != n || input.offset.len() != n
    {
        return Err(invalid(format!(
            "saved {} ALO row mismatch: response={n}, weights={}, primary_design={}, primary_offset={}",
            class.name(),
            observations.prior_weights.len(),
            input.design.nrows(),
            input.offset.len(),
        )));
    }
    if let Some((row, weight)) = observations
        .prior_weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid(format!(
            "saved {} ALO prior weight[{row}] must be finite and non-negative, got {weight}",
            class.name()
        )));
    }
    let noise_design = input.design_noise.as_ref().ok_or_else(|| {
        invalid(format!(
            "saved {} ALO requires the persisted secondary design",
            class.name()
        ))
    })?;
    if noise_design.nrows() != n {
        return Err(invalid(format!(
            "saved {} ALO secondary design has {} rows; expected {n}",
            class.name(),
            noise_design.nrows()
        )));
    }
    if let Some(offset) = input.offset_noise.as_ref()
        && offset.len() != n
    {
        return Err(invalid(format!(
            "saved {} ALO secondary offset has {} rows; expected {n}",
            class.name(),
            offset.len()
        )));
    }
    Ok(noise_design)
}

fn secondary_eta(input: &PredictInput, design: &DesignMatrix, beta: &Array1<f64>) -> Array1<f64> {
    let mut eta = design.dot(beta);
    if let Some(offset) = input.offset_noise.as_ref() {
        eta += offset;
    }
    eta
}

fn score_outer_product(score: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((score.len(), score.len()), |(row, column)| {
        score[row] * score[column]
    })
}

fn constant_scalar_design(n: usize) -> DesignMatrix {
    DesignMatrix::Dense(DenseDesignMatrix::from(Array2::ones((n, 1))))
}

fn location_scale_coordinate_layout(
    primary_design: &DesignMatrix,
    secondary_design: &DesignMatrix,
    primary_dimension: usize,
    secondary_dimension: usize,
    wiggle_dimension: usize,
    primary_name: &str,
    secondary_name: &str,
) -> (Vec<DesignMatrix>, Vec<Range<usize>>, Vec<String>) {
    let n = primary_design.nrows();
    let mut designs = Vec::with_capacity(2 + wiggle_dimension);
    let mut ranges = Vec::with_capacity(2 + wiggle_dimension);
    let mut names = Vec::with_capacity(2 + wiggle_dimension);
    designs.push(primary_design.clone());
    ranges.push(0..primary_dimension);
    names.push(primary_name.to_string());
    designs.push(secondary_design.clone());
    ranges.push(primary_dimension..primary_dimension + secondary_dimension);
    names.push(secondary_name.to_string());
    for coordinate in 0..wiggle_dimension {
        let coefficient = primary_dimension + secondary_dimension + coordinate;
        designs.push(constant_scalar_design(n));
        ranges.push(coefficient..coefficient + 1);
        names.push(format!("link-wiggle[{coordinate}]"));
    }
    (designs, ranges, names)
}

struct SavedAloGeometry<'a> {
    gauge: &'a Gauge,
    penalized_hessian: &'a Array2<f64>,
}

/// Require the exact active-coordinate geometry persisted by the fit.
///
/// A saved beta vector is a reporting coordinate, not necessarily the frame
/// in which the optimizer's precision lives.  The mandatory gauge is the
/// authority connecting them: `beta_saved = T theta_active + a`.  Guessing an
/// identity map or treating a pushed-forward covariance as a precision would
/// be mathematically invalid whenever `T` is rectangular.
fn require_saved_geometry<'a>(
    model: &'a FittedModel,
    class: PredictModelClass,
    raw_parameter_dimension: usize,
) -> Result<SavedAloGeometry<'a>, EstimationError> {
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid(format!(
            "saved {} ALO requires a canonical fitted coefficient state",
            class.name()
        ))
    })?;
    let geometry = fit.geometry.as_ref().ok_or_else(|| {
        invalid(format!(
            "saved {} ALO requires exact fitted working geometry",
            class.name()
        ))
    })?;
    geometry.coefficient_gauge.validate().map_err(|reason| {
        invalid(format!(
            "saved {} ALO coefficient gauge is invalid: {reason}",
            class.name()
        ))
    })?;
    if geometry.coefficient_gauge.raw_total() != raw_parameter_dimension {
        return Err(invalid(format!(
            "saved {} ALO gauge maps {} raw coefficients; fitted row layout requires {raw_parameter_dimension}",
            class.name(),
            geometry.coefficient_gauge.raw_total(),
        )));
    }
    let active_dimension = geometry.coefficient_gauge.reduced_total();
    let hessian = geometry.penalized_hessian.as_array();
    if hessian.dim() != (active_dimension, active_dimension) || active_dimension == 0 {
        return Err(invalid(format!(
            "saved {} ALO active precision is {}x{}; coefficient gauge requires {active_dimension}x{active_dimension}",
            class.name(),
            hessian.nrows(),
            hessian.ncols(),
        )));
    }
    Ok(SavedAloGeometry {
        gauge: &geometry.coefficient_gauge,
        penalized_hessian: hessian,
    })
}

fn exact_identity(matrix: &Array2<f64>) -> bool {
    matrix.nrows() == matrix.ncols()
        && matrix
            .indexed_iter()
            .all(|((row, column), &value)| value == if row == column { 1.0 } else { 0.0 })
}

/// Pull local raw-coordinate row Jacobians into the saved Hessian's active
/// frame.  Each raw local design `X_b` covers `raw_range_b`, so its exact
/// active Jacobian is
///
/// `X_b * T[raw_range_b, active_support_b]`.
///
/// The support is trimmed only at columns that are *exactly* zero.  This keeps
/// identity/block gauges cheap while preserving arbitrary dense or triangular
/// cross-block maps.  A coordinate fixed by the gauge receives a one-column
/// exact-zero design, which represents zero influence without inventing a
/// parameter or requiring an empty range that the ALO kernel forbids.
fn pullback_saved_coordinate_designs(
    class: PredictModelClass,
    gauge: &Gauge,
    coordinate_designs: Vec<DesignMatrix>,
    raw_coordinate_ranges: Vec<Range<usize>>,
) -> Result<(Vec<DesignMatrix>, Vec<Range<usize>>), EstimationError> {
    if coordinate_designs.len() != raw_coordinate_ranges.len() {
        return Err(invalid(format!(
            "saved {} ALO has {} coordinate designs but {} raw coefficient ranges",
            class.name(),
            coordinate_designs.len(),
            raw_coordinate_ranges.len(),
        )));
    }
    let active_dimension = gauge.reduced_total();
    if active_dimension == 0 {
        return Err(invalid(format!(
            "saved {} ALO coefficient gauge has no active coordinates",
            class.name()
        )));
    }

    let mut active_designs = Vec::with_capacity(coordinate_designs.len());
    let mut active_ranges = Vec::with_capacity(raw_coordinate_ranges.len());
    for (coordinate, (design, raw_range)) in coordinate_designs
        .into_iter()
        .zip(raw_coordinate_ranges)
        .enumerate()
    {
        if raw_range.start > raw_range.end
            || raw_range.end > gauge.raw_total()
            || design.ncols() != raw_range.len()
            || raw_range.is_empty()
        {
            return Err(invalid(format!(
                "saved {} ALO coordinate {coordinate} design width {} is not aligned to raw range {:?} in a {}-coefficient gauge",
                class.name(),
                design.ncols(),
                raw_range,
                gauge.raw_total(),
            )));
        }

        let raw_to_active = gauge.t_full.slice(s![raw_range, ..]);
        let first_active = (0..active_dimension).find(|&column| {
            raw_to_active
                .column(column)
                .iter()
                .any(|&value| value != 0.0)
        });
        let active_range = match first_active {
            Some(start) => {
                let end = (start..active_dimension)
                    .rev()
                    .find(|&column| {
                        raw_to_active
                            .column(column)
                            .iter()
                            .any(|&value| value != 0.0)
                    })
                    .expect("first active gauge column proves a last column")
                    + 1;
                start..end
            }
            None => 0..1,
        };
        let transform = raw_to_active.slice(s![.., active_range.clone()]).to_owned();
        if exact_identity(&transform) {
            active_designs.push(design);
        } else {
            active_designs.push(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
                ReparamOperator::new(design, Arc::new(transform)),
            ))));
        }
        active_ranges.push(active_range);
    }
    Ok((active_designs, active_ranges))
}

fn standard_alo_dispersion(
    standard_deviation: f64,
    link: LinkFunction,
) -> Result<f64, EstimationError> {
    if link != LinkFunction::Identity {
        return Ok(1.0);
    }
    if !standard_deviation.is_finite() || standard_deviation <= 0.0 {
        return Err(invalid(format!(
            "saved standard identity-link ALO requires a positive finite fitted residual standard deviation, got {standard_deviation}"
        )));
    }
    let phi = standard_deviation * standard_deviation;
    if !phi.is_finite() {
        return Err(invalid(format!(
            "saved standard identity-link ALO residual variance is outside f64 range: sigma={standard_deviation}"
        )));
    }
    Ok(phi)
}

fn compute_saved_standard_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::Standard;
    let n = observations.response.len();
    if n == 0
        || observations.prior_weights.len() != n
        || input.design.nrows() != n
        || input.offset.len() != n
    {
        return Err(invalid(format!(
            "saved standard ALO row mismatch: response={n}, weights={}, design={}, offset={}",
            observations.prior_weights.len(),
            input.design.nrows(),
            input.offset.len(),
        )));
    }
    if input.design_noise.is_some()
        || input.offset_noise.is_some()
        || input.auxiliary_scalar.is_some()
        || input.auxiliary_matrix.is_some()
    {
        return Err(invalid(
            "saved standard ALO received non-standard secondary or auxiliary coordinates",
        ));
    }
    // `require_saved_hessian` below is the saved *unconstrained* penalized
    // Hessian: `compute_alo_from_input` feeds it straight into the classical
    // one-step leave-one-out formula, which assumes a single quadratic
    // neighborhood around beta. A `bounded()` term's active box constraint
    // breaks that assumption — the true leave-one-out geometry at an active
    // bound is piecewise, not a single quadratic — so replaying the saved
    // Hessian would silently ignore the constraint. Surface that gap
    // explicitly (this dispatcher never refits, so there is no constrained
    // alternative to fall back to) instead of reporting a leverage that
    // quietly assumes the bound away.
    if model
        .resolved_termspec
        .as_ref()
        .is_some_and(termspec_has_bounded_terms)
    {
        return Err(invalid(
            "saved standard ALO does not yet support bounded() coefficients: leave-one-out \
             geometry for a box-constrained coefficient is not a single unconstrained \
             quadratic, so replaying the saved penalized Hessian would silently ignore the \
             constraint",
        ));
    }
    if let Some((row, weight)) = observations
        .prior_weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid(format!(
            "saved standard ALO prior weight[{row}] must be finite and non-negative, got {weight}"
        )));
    }

    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved standard ALO requires a canonical fitted coefficient state")
    })?;
    if fit.blocks.len() != 1 {
        return Err(invalid(format!(
            "saved standard ALO requires exactly one affine coefficient block; got {}",
            fit.blocks.len()
        )));
    }
    let beta = &fit.blocks[0].beta;
    if input.design.ncols() != beta.len() {
        return Err(invalid(format!(
            "saved standard ALO design has {} columns; fitted affine block has {} coefficients",
            input.design.ncols(),
            beta.len()
        )));
    }
    let geometry = require_saved_geometry(model, class, beta.len())?;
    let eta = input.design.dot(beta) + &input.offset;
    let likelihood =
        GlmLikelihoodSpec::try_new(model.likelihood(), fit.likelihood_scale.clone())
            .map_err(|error| invalid(format!("saved standard ALO likelihood scale: {error}")))?;
    let mut mean = Array1::<f64>::zeros(n);
    let mut working_weights = Array1::<f64>::zeros(n);
    let mut working_response = Array1::<f64>::zeros(n);
    gam_solve::pirls::update_glmvectors_by_family(
        observations.response.view(),
        &eta,
        &likelihood,
        observations.prior_weights.view(),
        &mut mean,
        &mut working_weights,
        &mut working_response,
    )?;
    let phi = standard_alo_dispersion(fit.standard_deviation, likelihood.spec.link_function())?;
    let (mut active_designs, active_ranges) = pullback_saved_coordinate_designs(
        class,
        geometry.gauge,
        vec![input.design.clone()],
        vec![0..beta.len()],
    )?;
    if active_ranges.len() != 1 || active_ranges[0] != (0..geometry.gauge.reduced_total()) {
        return Err(invalid(
            "saved standard ALO gauge does not expose every active coordinate to its sole fitted block",
        ));
    }
    let dense_design = active_designs
        .pop()
        .expect("one standard ALO design was constructed")
        .to_dense();
    let scalar = compute_alo_from_input(&AloInput::from_penalized_hessian_with_working_state(
        geometry.penalized_hessian,
        &dense_design,
        &eta,
        &input.offset,
        phi,
        &working_weights,
        &working_response,
    ))?;

    let eta_tilde = scalar
        .eta_tilde
        .iter()
        .copied()
        .map(|value| Array1::from_vec(vec![value]))
        .collect::<Vec<_>>();
    let alo_variance = scalar
        .se_sandwich
        .iter()
        .copied()
        .map(|standard_error| Array1::from_vec(vec![standard_error * standard_error]))
        .collect::<Vec<_>>();
    let mut cook_distance = Array1::<f64>::zeros(n);
    for row in 0..n {
        let deletion = scalar.eta_tilde[row] - eta[row];
        let cook = phi * working_weights[row] * deletion * deletion;
        if !cook.is_finite() || cook < 0.0 {
            return Err(invalid(format!(
                "saved standard ALO Cook distance is invalid at row {row}: {cook}"
            )));
        }
        cook_distance[row] = cook;
    }
    Ok(SavedModelAloDiagnostics {
        model_class: class,
        coordinate_names: vec!["eta".to_string()],
        diagnostics: MultiBlockAloDiagnostics {
            eta_tilde,
            leverage: scalar.leverage,
            alo_variance,
            cook_distance,
        },
    })
}

fn compute_saved_multicoordinate_core(
    class: PredictModelClass,
    coordinate_names: Vec<String>,
    coordinate_designs: Vec<DesignMatrix>,
    coordinate_ranges: Vec<Range<usize>>,
    geometry: SavedAloGeometry<'_>,
    observed_hessians: Vec<Array2<f64>>,
    scores: Vec<Array1<f64>>,
    coordinate_values: Vec<Array1<f64>>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let (active_designs, active_ranges) = pullback_saved_coordinate_designs(
        class,
        geometry.gauge,
        coordinate_designs,
        coordinate_ranges,
    )?;
    let score_covariances = scores.iter().map(score_outer_product).collect::<Vec<_>>();
    let diagnostics = compute_multiblock_alo(&MultiBlockAloInput {
        n_obs: scores.len(),
        n_coordinates: coordinate_names.len(),
        coordinate_designs: &active_designs,
        coordinate_coefficient_ranges: &active_ranges,
        penalized_hessian: geometry.penalized_hessian,
        observed_hessians: &observed_hessians,
        score_covariances: &score_covariances,
        scores: &scores,
        coordinate_values: &coordinate_values,
    })?;
    Ok(SavedModelAloDiagnostics {
        model_class: class,
        coordinate_names,
        diagnostics,
    })
}

fn compute_gaussian_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::GaussianLocationScale;
    let secondary_design = require_location_scale_inputs(class, input, observations)?;
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved Gaussian location-scale ALO requires a canonical fit result")
    })?;
    let beta_mean = gaussian_location_scale_mean_beta(fit)
        .ok_or_else(|| invalid("saved Gaussian location-scale ALO is missing the mean block"))?;
    let beta_scale = location_scale_noise_beta(fit).ok_or_else(|| {
        invalid("saved Gaussian location-scale ALO is missing the log-scale block")
    })?;
    let runtime = model.saved_prediction_runtime().map_err(|error| {
        invalid(format!(
            "saved Gaussian location-scale ALO runtime: {error}"
        ))
    })?;
    let wiggle = runtime.link_wiggle;
    let wiggle_beta = wiggle
        .as_ref()
        .map_or(&[][..], |runtime| runtime.beta.as_slice());
    let parameter_dimension = beta_mean.len() + beta_scale.len() + wiggle_beta.len();
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    let base_mean = input.design.dot(&beta_mean) + &input.offset;
    let eta_scale = secondary_eta(input, secondary_design, &beta_scale);
    let n = observations.response.len();
    let (basis, basis_d1, basis_d2) = match wiggle.as_ref() {
        Some(runtime) => {
            runtime.derivative_q0(&base_mean).map_err(|error| {
                invalid(format!("saved Gaussian location-scale ALO warp: {error}"))
            })?;
            (
                runtime.constrained_basis(&base_mean, BasisOptions::value()),
                runtime.constrained_basis(&base_mean, BasisOptions::first_derivative()),
                runtime.constrained_basis(&base_mean, BasisOptions::second_derivative()),
            )
        }
        None => (
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
        ),
    };
    let basis =
        basis.map_err(|error| invalid(format!("saved Gaussian ALO warp basis: {error}")))?;
    let basis_d1 =
        basis_d1.map_err(|error| invalid(format!("saved Gaussian ALO warp d1: {error}")))?;
    let basis_d2 =
        basis_d2.map_err(|error| invalid(format!("saved Gaussian ALO warp d2: {error}")))?;
    let response_scale = model.payload().gaussian_response_scale.ok_or_else(|| {
        invalid("saved Gaussian location-scale ALO is missing its response standardization scale")
    })?;

    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let basis_row = basis.row(row);
        let basis_d1_row = basis_d1.row(row);
        let basis_d2_row = basis_d2.row(row);
        let geometry = gaussian_location_scale_alo_row_geometry(GaussianLocationScaleAloRowInput {
            row,
            y: observations.response[row],
            base_mean: base_mean[row],
            eta_log_sigma: eta_scale[row],
            prior_weight: observations.prior_weights[row],
            response_scale,
            wiggle_basis: basis_row.as_slice().expect("basis row contiguous"),
            wiggle_basis_d1: basis_d1_row
                .as_slice()
                .expect("basis derivative row contiguous"),
            wiggle_basis_d2: basis_d2_row
                .as_slice()
                .expect("basis second derivative row contiguous"),
            wiggle_beta,
        })
        .map_err(|reason| invalid(format!("saved Gaussian ALO row {row}: {reason}")))?;
        observed_hessians.push(geometry.observed_hessian);
        scores.push(geometry.nll_score);
        let mut values = Array1::<f64>::zeros(2 + wiggle_beta.len());
        values[0] = base_mean[row];
        values[1] = eta_scale[row];
        for (coordinate, &value) in wiggle_beta.iter().enumerate() {
            values[2 + coordinate] = value;
        }
        coordinate_values.push(values);
    }
    let (designs, ranges, names) = location_scale_coordinate_layout(
        &input.design,
        secondary_design,
        beta_mean.len(),
        beta_scale.len(),
        wiggle_beta.len(),
        "mean-base",
        "log-sigma",
    );
    compute_saved_multicoordinate_core(
        class,
        names,
        designs,
        ranges,
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

fn compute_binomial_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::BinomialLocationScale;
    let secondary_design = require_location_scale_inputs(class, input, observations)?;
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved binomial location-scale ALO requires a canonical fit result")
    })?;
    let beta_threshold = binomial_location_scale_threshold_beta(fit).ok_or_else(|| {
        invalid("saved binomial location-scale ALO is missing the threshold block")
    })?;
    let beta_scale = location_scale_noise_beta(fit).ok_or_else(|| {
        invalid("saved binomial location-scale ALO is missing the log-scale block")
    })?;
    let runtime = model.saved_prediction_runtime().map_err(|error| {
        invalid(format!(
            "saved binomial location-scale ALO runtime: {error}"
        ))
    })?;
    let inverse_link = runtime.inverse_link.ok_or_else(|| {
        invalid("saved binomial location-scale ALO is missing its resolved inverse link")
    })?;
    let wiggle = runtime.link_wiggle;
    let wiggle_beta = wiggle
        .as_ref()
        .map_or(&[][..], |runtime| runtime.beta.as_slice());
    let parameter_dimension = beta_threshold.len() + beta_scale.len() + wiggle_beta.len();
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    let threshold_eta = input.design.dot(&beta_threshold) + &input.offset;
    let eta_scale = secondary_eta(input, secondary_design, &beta_scale);
    // Replay the fitted binomial map exactly. This deliberately does not use
    // the survival prediction saturation helper: a non-representable q0 is a
    // factual saved-row geometry failure here, not a value to replace before
    // evaluating the fitted warp basis.
    let q0 = Array1::from_shape_fn(threshold_eta.len(), |row| {
        -threshold_eta[row] * (-eta_scale[row]).exp()
    });
    let n = observations.response.len();
    let (basis, basis_d1, basis_d2) = match wiggle.as_ref() {
        Some(runtime) => {
            runtime.derivative_q0(&q0).map_err(|error| {
                invalid(format!("saved binomial location-scale ALO warp: {error}"))
            })?;
            (
                runtime.constrained_basis(&q0, BasisOptions::value()),
                runtime.constrained_basis(&q0, BasisOptions::first_derivative()),
                runtime.constrained_basis(&q0, BasisOptions::second_derivative()),
            )
        }
        None => (
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
        ),
    };
    let basis =
        basis.map_err(|error| invalid(format!("saved binomial ALO warp basis: {error}")))?;
    let basis_d1 =
        basis_d1.map_err(|error| invalid(format!("saved binomial ALO warp d1: {error}")))?;
    let basis_d2 =
        basis_d2.map_err(|error| invalid(format!("saved binomial ALO warp d2: {error}")))?;

    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let basis_row = basis.row(row);
        let basis_d1_row = basis_d1.row(row);
        let basis_d2_row = basis_d2.row(row);
        let geometry = binomial_location_scale_alo_row_geometry(BinomialLocationScaleAloRowInput {
            y: observations.response[row],
            threshold_eta: threshold_eta[row],
            eta_log_sigma: eta_scale[row],
            prior_weight: observations.prior_weights[row],
            inverse_link: &inverse_link,
            wiggle_basis: basis_row.as_slice().expect("basis row contiguous"),
            wiggle_basis_d1: basis_d1_row
                .as_slice()
                .expect("basis derivative row contiguous"),
            wiggle_basis_d2: basis_d2_row
                .as_slice()
                .expect("basis second derivative row contiguous"),
            wiggle_beta,
        })
        .map_err(|reason| invalid(format!("saved binomial ALO row {row}: {reason}")))?;
        observed_hessians.push(geometry.observed_hessian);
        scores.push(geometry.nll_score);
        let mut values = Array1::<f64>::zeros(2 + wiggle_beta.len());
        values[0] = threshold_eta[row];
        values[1] = eta_scale[row];
        for (coordinate, &value) in wiggle_beta.iter().enumerate() {
            values[2 + coordinate] = value;
        }
        coordinate_values.push(values);
    }
    let (designs, ranges, names) = location_scale_coordinate_layout(
        &input.design,
        secondary_design,
        beta_threshold.len(),
        beta_scale.len(),
        wiggle_beta.len(),
        "threshold",
        "log-sigma",
    );
    compute_saved_multicoordinate_core(
        class,
        names,
        designs,
        ranges,
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

fn dispersion_kind(response: &ResponseFamily) -> Result<DispersionFamilyKind, EstimationError> {
    match response {
        ResponseFamily::NegativeBinomial { .. } => Ok(DispersionFamilyKind::NegativeBinomial),
        ResponseFamily::Gamma => Ok(DispersionFamilyKind::Gamma),
        ResponseFamily::Beta { .. } => Ok(DispersionFamilyKind::Beta),
        ResponseFamily::Tweedie { p } => Ok(DispersionFamilyKind::Tweedie { p: *p }),
        response => Err(invalid(format!(
            "saved dispersion location-scale ALO cannot replay response family {response:?}"
        ))),
    }
}

fn compute_dispersion_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::DispersionLocationScale;
    let secondary_design = require_location_scale_inputs(class, input, observations)?;
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved dispersion location-scale ALO requires a canonical fit result")
    })?;
    let beta_mean = gaussian_location_scale_mean_beta(fit)
        .ok_or_else(|| invalid("saved dispersion location-scale ALO is missing the mean block"))?;
    let beta_dispersion = location_scale_noise_beta(fit).ok_or_else(|| {
        invalid("saved dispersion location-scale ALO is missing the precision block")
    })?;
    let parameter_dimension = beta_mean.len() + beta_dispersion.len();
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    let eta_mean = input.design.dot(&beta_mean) + &input.offset;
    let eta_dispersion = secondary_eta(input, secondary_design, &beta_dispersion);
    let kind = dispersion_kind(&model.payload().family_state.likelihood().response)?;
    let n = observations.response.len();
    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let geometry = dispersion_alo_row_geometry(
            kind,
            row,
            observations.response[row],
            eta_mean[row],
            eta_dispersion[row],
            observations.prior_weights[row],
        )
        .map_err(|reason| invalid(format!("saved dispersion ALO row {row}: {reason}")))?;
        let score = Array1::from_vec(geometry.nll_score.to_vec());
        let hessian = Array2::from_shape_vec(
            (2, 2),
            geometry.observed_hessian.into_iter().flatten().collect(),
        )
        .expect("fixed 2x2 dispersion geometry");
        observed_hessians.push(hessian);
        scores.push(score);
        coordinate_values.push(Array1::from_vec(vec![eta_mean[row], eta_dispersion[row]]));
    }
    let (designs, ranges, names) = location_scale_coordinate_layout(
        &input.design,
        secondary_design,
        beta_mean.len(),
        beta_dispersion.len(),
        0,
        "mean",
        "log-precision",
    );
    compute_saved_multicoordinate_core(
        class,
        names,
        designs,
        ranges,
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

/// Dispatch exact saved-H ALO for every location-scale model class.
///
/// This path never refits and never substitutes another model. The saved
/// penalized Hessian, row likelihood and coefficient topology must all be
/// present and dimensionally aligned or the request fails factually.
fn compute_saved_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    match model.predict_model_class() {
        PredictModelClass::GaussianLocationScale => {
            compute_gaussian_location_scale_alo(model, input, &observations)
        }
        PredictModelClass::BinomialLocationScale => {
            compute_binomial_location_scale_alo(model, input, &observations)
        }
        PredictModelClass::DispersionLocationScale => {
            compute_dispersion_location_scale_alo(model, input, &observations)
        }
        class => Err(invalid(format!(
            "saved location-scale ALO dispatcher received model class {}",
            class.name()
        ))),
    }
}

/// Replay exact saved-H ALO for every fitted Bernoulli marginal-slope mode.
fn compute_saved_bernoulli_marginal_slope_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::BernoulliMarginalSlope;
    if model.predict_model_class() != class {
        return Err(invalid(format!(
            "saved marginal-slope ALO dispatcher received model class {}",
            model.predict_model_class().name()
        )));
    }
    let n = observations.response.len();
    if n == 0
        || observations.prior_weights.len() != n
        || input.design.nrows() != n
        || input
            .design_noise
            .as_ref()
            .is_none_or(|design| design.nrows() != n)
    {
        return Err(invalid(format!(
            "saved marginal-slope ALO row mismatch: response={n}, weights={}, marginal_design={}, slope_design={}",
            observations.prior_weights.len(),
            input.design.nrows(),
            input.design_noise.as_ref().map_or(0, DesignMatrix::nrows),
        )));
    }
    if let Some((row, weight)) = observations
        .prior_weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid(format!(
            "saved marginal-slope ALO prior weight[{row}] must be finite and non-negative, got {weight}"
        )));
    }
    let predictor = model
        .bernoulli_marginal_slope_predictor()
        .map_err(|reason| invalid(format!("saved marginal-slope ALO predictor: {reason}")))?;
    let replay =
        predictor.saved_alo_replay(input, observations.response, observations.prior_weights)?;
    let marginal_dimension = predictor.beta_marginal.len();
    let logslope_dimension = predictor.beta_logslope.len();
    let parameter_dimension = marginal_dimension
        + logslope_dimension
        + replay.score_warp_dimension
        + replay.link_deviation_dimension;
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    if replay.rows.len() != n {
        return Err(invalid(format!(
            "saved marginal-slope ALO replay returned {} rows; expected {n}",
            replay.rows.len(),
        )));
    }
    for geometry in replay.rows {
        scores.push(geometry.nll_score);
        observed_hessians.push(geometry.observed_hessian);
        coordinate_values.push(geometry.coordinate_values);
    }
    let secondary_design = input
        .design_noise
        .as_ref()
        .expect("validated marginal-slope design");
    let mut coordinate_designs =
        Vec::with_capacity(2 + replay.score_warp_dimension + replay.link_deviation_dimension);
    let mut coordinate_ranges = Vec::with_capacity(coordinate_designs.capacity());
    let mut coordinate_names = Vec::with_capacity(coordinate_designs.capacity());
    coordinate_designs.push(input.design.clone());
    coordinate_ranges.push(0..marginal_dimension);
    coordinate_names.push("marginal-eta".to_string());
    coordinate_designs.push(secondary_design.clone());
    coordinate_ranges.push(marginal_dimension..marginal_dimension + logslope_dimension);
    coordinate_names.push("slope".to_string());
    let mut coefficient = marginal_dimension + logslope_dimension;
    for coordinate in 0..replay.score_warp_dimension {
        coordinate_designs.push(constant_scalar_design(n));
        coordinate_ranges.push(coefficient..coefficient + 1);
        coordinate_names.push(format!("score-warp[{coordinate}]"));
        coefficient += 1;
    }
    for coordinate in 0..replay.link_deviation_dimension {
        coordinate_designs.push(constant_scalar_design(n));
        coordinate_ranges.push(coefficient..coefficient + 1);
        coordinate_names.push(format!("link-deviation[{coordinate}]"));
        coefficient += 1;
    }
    if coefficient != parameter_dimension {
        return Err(invalid(format!(
            "saved marginal-slope ALO coordinate layout ends at {coefficient}; fitted parameter dimension is {parameter_dimension}"
        )));
    }
    compute_saved_multicoordinate_core(
        class,
        coordinate_names,
        coordinate_designs,
        coordinate_ranges,
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

/// Replay exact saved-H ALO for a finite-support transformation-normal fit.
fn compute_saved_transformation_normal_alo(
    model: &FittedModel,
    covariate_design: &DesignMatrix,
    additive_offset: &Array1<f64>,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::TransformationNormal;
    if model.predict_model_class() != class {
        return Err(invalid(format!(
            "saved transformation-normal ALO dispatcher received model class {}",
            model.predict_model_class().name()
        )));
    }
    let n = observations.response.len();
    if n == 0
        || observations.prior_weights.len() != n
        || covariate_design.nrows() != n
        || additive_offset.len() != n
    {
        return Err(invalid(format!(
            "saved transformation-normal ALO row mismatch: response={n}, weights={}, design={}, offset={}",
            observations.prior_weights.len(),
            covariate_design.nrows(),
            additive_offset.len(),
        )));
    }
    let payload = model.payload();
    let knots = payload
        .transformation_response_knots
        .as_ref()
        .ok_or_else(|| invalid("saved transformation-normal ALO is missing response knots"))?;
    let transform_rows = payload
        .transformation_response_transform
        .as_ref()
        .ok_or_else(|| {
            invalid("saved transformation-normal ALO is missing its response transform")
        })?;
    let degree = payload
        .transformation_response_degree
        .ok_or_else(|| invalid("saved transformation-normal ALO is missing its response degree"))?;
    let response_median = payload
        .transformation_response_median
        .ok_or_else(|| invalid("saved transformation-normal ALO is missing its response median"))?;
    if knots.is_empty() || transform_rows.is_empty() {
        return Err(invalid(
            "saved transformation-normal ALO response basis metadata is empty",
        ));
    }
    let transform_columns = transform_rows[0].len();
    if transform_columns == 0
        || transform_rows
            .iter()
            .any(|row| row.len() != transform_columns)
    {
        return Err(invalid(
            "saved transformation-normal ALO response transform is empty or ragged",
        ));
    }
    let mut transform = Array2::<f64>::zeros((transform_rows.len(), transform_columns));
    for (row_index, row) in transform_rows.iter().enumerate() {
        for (column_index, &value) in row.iter().enumerate() {
            transform[[row_index, column_index]] = value;
        }
    }
    let knots = Array1::from_vec(knots.clone());
    let (raw_value_basis, _) = create_basis::<Dense>(
        observations.response.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .map_err(|error| {
        invalid(format!(
            "saved transformation-normal ALO value basis: {error}"
        ))
    })?;
    let raw_value_basis = raw_value_basis.as_ref();
    let raw_derivative_basis =
        create_ispline_derivative_dense(observations.response.view(), &knots, degree, 1).map_err(
            |error| {
                invalid(format!(
                    "saved transformation-normal ALO derivative basis: {error}"
                ))
            },
        )?;
    if raw_value_basis.ncols() != transform.nrows()
        || raw_derivative_basis.dim() != raw_value_basis.dim()
    {
        return Err(invalid(format!(
            "saved transformation-normal ALO raw basis/transform mismatch: value={}x{}, derivative={}x{}, transform={}x{}",
            raw_value_basis.nrows(),
            raw_value_basis.ncols(),
            raw_derivative_basis.nrows(),
            raw_derivative_basis.ncols(),
            transform.nrows(),
            transform.ncols(),
        )));
    }
    let shape_value = raw_value_basis.dot(&transform);
    let shape_derivative = raw_derivative_basis.dot(&transform);
    let response_dimension = transform_columns + 1;
    let mut response_value_basis = Array2::<f64>::zeros((n, response_dimension));
    response_value_basis.column_mut(0).fill(1.0);
    response_value_basis
        .slice_mut(s![.., 1..])
        .assign(&shape_value);
    let mut response_derivative_basis = Array2::<f64>::zeros((n, response_dimension));
    response_derivative_basis
        .slice_mut(s![.., 1..])
        .assign(&shape_derivative);

    let lower_response = knots[0];
    let upper_response = knots[knots.len() - 1];
    if !(upper_response > lower_response) {
        return Err(invalid(format!(
            "saved transformation-normal ALO support is degenerate: lower={lower_response}, upper={upper_response}"
        )));
    }
    let endpoints = Array1::from_vec(vec![lower_response, upper_response]);
    let (raw_endpoint_basis, _) = create_basis::<Dense>(
        endpoints.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .map_err(|error| {
        invalid(format!(
            "saved transformation-normal ALO endpoints: {error}"
        ))
    })?;
    if raw_endpoint_basis.ncols() != transform.nrows() {
        return Err(invalid(format!(
            "saved transformation-normal ALO endpoint basis has {} columns; transform requires {}",
            raw_endpoint_basis.ncols(),
            transform.nrows(),
        )));
    }
    let endpoint_shape = raw_endpoint_basis.as_ref().dot(&transform);
    let mut lower_basis = Array1::<f64>::zeros(response_dimension);
    let mut upper_basis = Array1::<f64>::zeros(response_dimension);
    lower_basis[0] = 1.0;
    upper_basis[0] = 1.0;
    lower_basis
        .slice_mut(s![1..])
        .assign(&endpoint_shape.row(0));
    upper_basis
        .slice_mut(s![1..])
        .assign(&endpoint_shape.row(1));

    let fit = payload.fit_result.as_ref().ok_or_else(|| {
        invalid("saved transformation-normal ALO requires a canonical fit result")
    })?;
    if fit.blocks.len() != 1 {
        return Err(invalid(format!(
            "saved transformation-normal ALO requires one coefficient block, got {}",
            fit.blocks.len()
        )));
    }
    let beta = &fit.blocks[0].beta;
    let covariate_dimension = covariate_design.ncols();
    let parameter_dimension = response_dimension * covariate_dimension;
    if beta.len() != parameter_dimension {
        return Err(invalid(format!(
            "saved transformation-normal ALO beta has {} entries; response/covariate layout requires {parameter_dimension}",
            beta.len()
        )));
    }
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    let mut gamma = Array2::<f64>::zeros((n, response_dimension));
    for component in 0..response_dimension {
        let start = component * covariate_dimension;
        let beta_component = beta
            .slice(s![start..start + covariate_dimension])
            .to_owned();
        gamma
            .column_mut(component)
            .assign(&covariate_design.dot(&beta_component));
    }

    let lower_floor = TRANSFORMATION_MONOTONICITY_EPS * (lower_response - response_median);
    let upper_floor = TRANSFORMATION_MONOTONICITY_EPS * (upper_response - response_median);
    let lower_basis_slice = lower_basis
        .as_slice()
        .expect("owned lower basis contiguous");
    let upper_basis_slice = upper_basis
        .as_slice()
        .expect("owned upper basis contiguous");
    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let response_value_row = response_value_basis.row(row);
        let response_derivative_row = response_derivative_basis.row(row);
        let gamma_row = gamma.row(row);
        let geometry = transformation_normal_alo_row_geometry(TransformationNormalAloRowInput {
            response_value_basis: response_value_row
                .as_slice()
                .expect("response value row contiguous"),
            response_derivative_basis: response_derivative_row
                .as_slice()
                .expect("response derivative row contiguous"),
            response_lower_basis: lower_basis_slice,
            response_upper_basis: upper_basis_slice,
            gamma: gamma_row.as_slice().expect("gamma row contiguous"),
            additive_offset: additive_offset[row],
            response_floor_offset: TRANSFORMATION_MONOTONICITY_EPS
                * (observations.response[row] - response_median),
            response_lower_floor_offset: lower_floor,
            response_upper_floor_offset: upper_floor,
            prior_weight: observations.prior_weights[row],
        })
        .map_err(|reason| {
            invalid(format!(
                "saved transformation-normal ALO row {row}: {reason}"
            ))
        })?;
        observed_hessians.push(geometry.observed_hessian);
        scores.push(geometry.nll_score);
        coordinate_values.push(gamma_row.to_owned());
    }
    let coordinate_designs = (0..response_dimension)
        .map(|_| covariate_design.clone())
        .collect::<Vec<_>>();
    let coordinate_ranges = (0..response_dimension)
        .map(|component| {
            let start = component * covariate_dimension;
            start..start + covariate_dimension
        })
        .collect::<Vec<_>>();
    let coordinate_names = (0..response_dimension)
        .map(|component| {
            if component == 0 {
                "location-gamma".to_string()
            } else {
                format!("shape-gamma[{component}]")
            }
        })
        .collect::<Vec<_>>();
    compute_saved_multicoordinate_core(
        class,
        coordinate_names,
        coordinate_designs,
        coordinate_ranges,
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

fn compute_saved_cause_specific_survival_alo(
    model: &FittedModel,
    input: &SavedCauseSpecificSurvivalAloInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::Survival;
    let likelihood_mode = require_saved_survival_likelihood_mode(model)
        .map_err(|error| invalid(format!("saved survival ALO likelihood mode: {error}")))?;
    if !matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        return Err(invalid(format!(
            "saved cause-specific ALO row geometry is valid only for transformation/weibull survival, got {likelihood_mode:?}"
        )));
    }
    let n = input.event_codes.len();
    if observations.response.len() != n || observations.prior_weights.len() != n {
        return Err(invalid(format!(
            "saved cause-specific ALO row mismatch: event_codes={n}, response={}, weights={}",
            observations.response.len(),
            observations.prior_weights.len()
        )));
    }
    for row in 0..n {
        let response_code = survival_event_code_from_value(observations.response[row], row)
            .map_err(|reason| invalid(format!("saved survival ALO response: {reason}")))?;
        if response_code != input.event_codes[row] {
            return Err(invalid(format!(
                "saved survival ALO event code mismatch at row {row}: typed carrier={}, observations={response_code}",
                input.event_codes[row]
            )));
        }
        let weight = observations.prior_weights[row];
        if !weight.is_finite() || weight < 0.0 {
            return Err(invalid(format!(
                "saved survival ALO prior weight[{row}] must be finite and non-negative, got {weight}"
            )));
        }
    }

    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved survival ALO requires a canonical fitted coefficient state")
    })?;
    let parameter_dimension = fit.beta.len();
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    let fitted_cause_count = model
        .payload()
        .survival_cause_count
        .unwrap_or(fit.blocks.len())
        .max(1);
    if input.cause_count != fitted_cause_count || fit.blocks.len() != fitted_cause_count {
        return Err(invalid(format!(
            "saved cause-specific ALO cause topology mismatch: carrier={}, metadata={}, fitted_blocks={}",
            input.cause_count,
            fitted_cause_count,
            fit.blocks.len()
        )));
    }

    let coordinate_count = input.cause_count * 3;
    let mut coordinate_arrays = Vec::with_capacity(coordinate_count);
    for coordinate in 0..coordinate_count {
        let range = &input.coordinate_ranges[coordinate];
        if range.end > parameter_dimension {
            return Err(invalid(format!(
                "saved cause-specific ALO coordinate {coordinate} range {range:?} exceeds fitted parameter dimension {parameter_dimension}"
            )));
        }
        let beta = fit.beta.slice(s![range.clone()]).to_owned();
        coordinate_arrays.push(
            input.coordinate_designs[coordinate].dot(&beta) + &input.coordinate_offsets[coordinate],
        );
    }

    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let mut score = Array1::<f64>::zeros(coordinate_count);
        let mut observed_hessian = Array2::<f64>::zeros((coordinate_count, coordinate_count));
        let mut values = Array1::<f64>::zeros(coordinate_count);
        for cause in 0..input.cause_count {
            let start = cause * 3;
            let eta_exit = coordinate_arrays[start][row];
            let eta_entry = coordinate_arrays[start + 1][row];
            let derivative_exit = coordinate_arrays[start + 2][row];
            values[start] = eta_exit;
            values[start + 1] = eta_entry;
            values[start + 2] = derivative_exit;
            let geometry =
                cause_specific_survival_alo_row_geometry(CauseSpecificSurvivalAloRowInput {
                    eta_exit,
                    eta_entry,
                    derivative_exit,
                    prior_weight: observations.prior_weights[row],
                    entry_active: input.entry_active[row],
                    event: usize::from(input.event_codes[row]) == cause + 1,
                })
                .map_err(|reason| {
                    invalid(format!(
                        "saved survival ALO row {row}, cause {}: {reason}",
                        cause + 1
                    ))
                })?;
            for left in 0..3 {
                score[start + left] = geometry.nll_score[left];
                for right in 0..3 {
                    observed_hessian[[start + left, start + right]] =
                        geometry.observed_hessian[left][right];
                }
            }
        }
        scores.push(score);
        observed_hessians.push(observed_hessian);
        coordinate_values.push(values);
    }
    let coordinate_names = (0..input.cause_count)
        .flat_map(|cause| {
            let prefix = if input.cause_count == 1 {
                String::new()
            } else {
                format!("cause[{}].", cause + 1)
            };
            [
                format!("{prefix}eta-exit"),
                format!("{prefix}eta-entry"),
                format!("{prefix}derivative-exit"),
            ]
        })
        .collect::<Vec<_>>();
    compute_saved_multicoordinate_core(
        class,
        coordinate_names,
        input.coordinate_designs.clone(),
        input.coordinate_ranges.clone(),
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

fn compute_saved_location_scale_survival_alo(
    model: &FittedModel,
    input: &SavedLocationScaleSurvivalAloInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::Survival;
    let likelihood_mode = require_saved_survival_likelihood_mode(model)
        .map_err(|error| invalid(format!("saved survival ALO likelihood mode: {error}")))?;
    if likelihood_mode != SurvivalLikelihoodMode::LocationScale {
        return Err(invalid(format!(
            "saved location-scale survival ALO row geometry cannot replay {likelihood_mode:?}"
        )));
    }
    let n = input.event.len();
    if observations.response.len() != n || observations.prior_weights.len() != n {
        return Err(invalid(format!(
            "saved survival location-scale ALO row mismatch: carrier={n}, response={}, weights={}",
            observations.response.len(),
            observations.prior_weights.len(),
        )));
    }
    for row in 0..n {
        if observations.response[row] != input.event[row] {
            return Err(invalid(format!(
                "saved survival location-scale ALO event mismatch at row {row}: carrier={}, observations={}",
                input.event[row], observations.response[row],
            )));
        }
        let weight = observations.prior_weights[row];
        if !weight.is_finite() || weight < 0.0 {
            return Err(invalid(format!(
                "saved survival location-scale ALO prior weight[{row}] must be finite and non-negative, got {weight}"
            )));
        }
    }

    let payload = model.payload();
    let structure = payload
        .survival_location_scale_structure
        .as_ref()
        .ok_or_else(|| {
            invalid("saved survival location-scale ALO is missing exact replay structure")
        })?;
    let fit = payload.fit_result.as_ref().ok_or_else(|| {
        invalid("saved survival location-scale ALO requires a canonical fitted coefficient state")
    })?;
    let runtime = model.saved_prediction_runtime().map_err(|error| {
        invalid(format!(
            "saved survival location-scale ALO runtime: {error}"
        ))
    })?;
    let inverse_link = runtime.inverse_link.ok_or_else(|| {
        invalid("saved survival location-scale ALO is missing its resolved inverse link")
    })?;

    let block_range = |role: BlockRole, label: &str| -> Result<Range<usize>, EstimationError> {
        let mut cursor = 0usize;
        let mut found = None;
        for block in &fit.blocks {
            let next = cursor.checked_add(block.beta.len()).ok_or_else(|| {
                invalid("saved survival location-scale ALO coefficient layout overflows usize")
            })?;
            if block.role == role {
                if found.is_some() {
                    return Err(invalid(format!(
                        "saved survival location-scale ALO has duplicate {label} blocks"
                    )));
                }
                found = Some(cursor..next);
            }
            cursor = next;
        }
        if cursor != fit.beta.len() {
            return Err(invalid(format!(
                "saved survival location-scale ALO block widths sum to {cursor}; fitted beta has {} entries",
                fit.beta.len(),
            )));
        }
        found.ok_or_else(|| {
            invalid(format!(
                "saved survival location-scale ALO is missing its {label} block"
            ))
        })
    };
    let time_range = block_range(BlockRole::Time, "time")?;
    let threshold_range = block_range(BlockRole::Threshold, "threshold")?;
    let log_sigma_range = block_range(BlockRole::Scale, "log-sigma")?;
    let link_range = match runtime.link_wiggle.as_ref() {
        Some(_) => Some(block_range(BlockRole::LinkWiggle, "link-wiggle")?),
        None => {
            if fit
                .blocks
                .iter()
                .any(|block| block.role == BlockRole::LinkWiggle)
            {
                return Err(invalid(
                    "saved survival location-scale ALO fit has a link-wiggle block without runtime authority",
                ));
            }
            None
        }
    };
    let expected_blocks = 3 + usize::from(link_range.is_some());
    if fit.blocks.len() != expected_blocks {
        return Err(invalid(format!(
            "saved survival location-scale ALO expects {expected_blocks} fitted blocks, got {}",
            fit.blocks.len(),
        )));
    }

    let time_wiggle_width = runtime
        .baseline_time_wiggle
        .as_ref()
        .map_or(0, |wiggle| wiggle.beta.len());
    if time_wiggle_width >= time_range.len() && time_wiggle_width != 0 {
        return Err(invalid(format!(
            "saved survival location-scale ALO time-wiggle width {time_wiggle_width} must leave a non-empty affine time block within {} coefficients",
            time_range.len(),
        )));
    }
    let time_base_width = time_range.len() - time_wiggle_width;
    if input.time_base.design_exit.ncols() != time_base_width {
        return Err(invalid(format!(
            "saved survival location-scale ALO affine time design has {} columns; fitted base time block has {time_base_width}",
            input.time_base.design_exit.ncols(),
        )));
    }
    let beta_time_base = fit
        .beta
        .slice(s![time_range.start..time_range.start + time_base_width])
        .to_owned();
    let (h_base_entry, h_base_exit, hdot_base_exit) = input
        .time_base
        .evaluate(&beta_time_base, "location-scale base-time")?;
    let beta_threshold = fit.beta.slice(s![threshold_range.clone()]).to_owned();
    let (eta_threshold_entry, eta_threshold_exit, eta_threshold_derivative_exit) = input
        .threshold
        .evaluate(&beta_threshold, "location-scale threshold")?;
    let beta_log_sigma = fit.beta.slice(s![log_sigma_range.clone()]).to_owned();
    let (eta_log_sigma_entry, eta_log_sigma_exit, eta_log_sigma_derivative_exit) = input
        .log_sigma
        .evaluate(&beta_log_sigma, "location-scale log-sigma")?;

    let reduced_parametric_aft = matches!(
        structure.time_parameterization,
        gam_models::survival::SurvivalLocationScaleTimeParameterization::ReducedParametricAft
    );
    if reduced_parametric_aft {
        if runtime.baseline_time_wiggle.is_some() || runtime.link_wiggle.is_some() {
            return Err(invalid(
                "saved reduced parametric-AFT location-scale ALO cannot carry time/link wiggles",
            ));
        }
        for (label, design) in [
            ("entry", &input.time_base.design_entry),
            ("exit", &input.time_base.design_exit),
            ("derivative", &input.time_base.design_derivative_exit),
        ] {
            if design.to_dense().iter().any(|value| *value != 0.0) {
                return Err(invalid(format!(
                    "saved reduced parametric-AFT location-scale ALO {label} time design must be the exact zero affine lift"
                )));
            }
        }
        if h_base_entry.iter().any(|value| *value != 0.0)
            || h_base_exit.iter().any(|value| *value != 0.0)
            || hdot_base_exit.iter().any(|value| *value != 0.0)
        {
            return Err(invalid(
                "saved reduced parametric-AFT location-scale ALO time coordinates must be exact zero",
            ));
        }
    }

    let time_wiggle_beta = fit
        .beta
        .slice(s![time_range.start + time_base_width..time_range.end]);
    let time_wiggle_basis = match runtime.baseline_time_wiggle.as_ref() {
        Some(wiggle) => {
            if wiggle.beta.as_slice() != time_wiggle_beta.as_slice().unwrap_or(&[]) {
                return Err(invalid(
                    "saved survival location-scale ALO time-wiggle runtime disagrees with the fitted time-block tail",
                ));
            }
            let knots = Array1::from_vec(wiggle.knots.clone());
            Some(
                survival_location_scale_time_wiggle_basis_authority(
                    h_base_entry.view(),
                    h_base_exit.view(),
                    &knots,
                    wiggle.degree,
                    time_wiggle_width,
                )
                .map_err(|reason| {
                    invalid(format!(
                        "saved survival location-scale ALO time-wiggle basis: {reason}"
                    ))
                })?,
            )
        }
        None => None,
    };

    let inv_sigma_entry = eta_log_sigma_entry.mapv(|value| (-value).exp());
    let inv_sigma_exit = eta_log_sigma_exit.mapv(|value| (-value).exp());
    let q_base_entry = -&eta_threshold_entry * &inv_sigma_entry;
    let q_base_exit = -&eta_threshold_exit * &inv_sigma_exit;
    let link_wiggle = runtime.link_wiggle.as_ref();
    let (link_entry, link_entry_d1, link_entry_d2, link_entry_d3) = match link_wiggle {
        Some(wiggle) => (
            wiggle.constrained_basis(&q_base_entry, BasisOptions::value()),
            wiggle.constrained_basis(&q_base_entry, BasisOptions::first_derivative()),
            wiggle.constrained_basis(&q_base_entry, BasisOptions::second_derivative()),
            wiggle.constrained_basis(&q_base_entry, BasisOptions::third_derivative()),
        ),
        None => (
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
        ),
    };
    let (link_exit, link_exit_d1, link_exit_d2, link_exit_d3) = match link_wiggle {
        Some(wiggle) => {
            wiggle.derivative_q0(&q_base_exit).map_err(|error| {
                invalid(format!(
                    "saved survival location-scale ALO link-wiggle monotonicity: {error}"
                ))
            })?;
            (
                wiggle.constrained_basis(&q_base_exit, BasisOptions::value()),
                wiggle.constrained_basis(&q_base_exit, BasisOptions::first_derivative()),
                wiggle.constrained_basis(&q_base_exit, BasisOptions::second_derivative()),
                wiggle.constrained_basis(&q_base_exit, BasisOptions::third_derivative()),
            )
        }
        None => (
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
        ),
    };
    let map_link_basis_error = |channel: &str, error| {
        invalid(format!(
            "saved survival location-scale ALO link-wiggle {channel}: {error}"
        ))
    };
    let link_entry = link_entry.map_err(|error| map_link_basis_error("entry value", error))?;
    let link_entry_d1 =
        link_entry_d1.map_err(|error| map_link_basis_error("entry d1", error))?;
    let link_entry_d2 =
        link_entry_d2.map_err(|error| map_link_basis_error("entry d2", error))?;
    let link_entry_d3 =
        link_entry_d3.map_err(|error| map_link_basis_error("entry d3", error))?;
    let link_exit = link_exit.map_err(|error| map_link_basis_error("exit value", error))?;
    let link_exit_d1 =
        link_exit_d1.map_err(|error| map_link_basis_error("exit d1", error))?;
    let link_exit_d2 =
        link_exit_d2.map_err(|error| map_link_basis_error("exit d2", error))?;
    let link_exit_d3 =
        link_exit_d3.map_err(|error| map_link_basis_error("exit d3", error))?;
    let link_wiggle_beta = link_wiggle.map_or(&[][..], |wiggle| wiggle.beta.as_slice());

    let parameter_dimension = fit.beta.len();
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let time_wiggle_row = time_wiggle_basis.as_ref().map(|basis| {
            SurvivalLocationScaleAloTimeWiggleInput {
                beta: time_wiggle_beta.as_slice().expect("fitted beta slice contiguous"),
                entry_basis: basis.entry.value.row(row).as_slice().expect("basis row contiguous"),
                entry_basis_d1: basis.entry.d1.row(row).as_slice().expect("basis row contiguous"),
                entry_basis_d2: basis.entry.d2.row(row).as_slice().expect("basis row contiguous"),
                entry_basis_d3: basis.entry.d3.row(row).as_slice().expect("basis row contiguous"),
                exit_basis: basis.exit.value.row(row).as_slice().expect("basis row contiguous"),
                exit_basis_d1: basis.exit.d1.row(row).as_slice().expect("basis row contiguous"),
                exit_basis_d2: basis.exit.d2.row(row).as_slice().expect("basis row contiguous"),
                exit_basis_d3: basis.exit.d3.row(row).as_slice().expect("basis row contiguous"),
            }
        });
        let link_wiggle_row = link_wiggle.map(|_| SurvivalLocationScaleAloWiggleInput {
            beta: link_wiggle_beta,
            entry_basis: link_entry.row(row).as_slice().expect("basis row contiguous"),
            entry_basis_d1: link_entry_d1.row(row).as_slice().expect("basis row contiguous"),
            entry_basis_d2: link_entry_d2.row(row).as_slice().expect("basis row contiguous"),
            entry_basis_d3: link_entry_d3.row(row).as_slice().expect("basis row contiguous"),
            exit_basis: link_exit.row(row).as_slice().expect("basis row contiguous"),
            exit_basis_d1: link_exit_d1.row(row).as_slice().expect("basis row contiguous"),
            exit_basis_d2: link_exit_d2.row(row).as_slice().expect("basis row contiguous"),
            exit_basis_d3: link_exit_d3.row(row).as_slice().expect("basis row contiguous"),
        });
        let row_geometry =
            survival_location_scale_alo_row_geometry(SurvivalLocationScaleAloRowInput {
                inverse_link: &inverse_link,
                prior_weight: observations.prior_weights[row],
                event: input.event[row],
                derivative_guard: input.derivative_guard,
                h_entry: h_base_entry[row],
                h_exit: h_base_exit[row],
                hdot_exit: hdot_base_exit[row],
                eta_threshold_exit: eta_threshold_exit[row],
                eta_threshold_entry: eta_threshold_entry[row],
                eta_threshold_derivative_exit: eta_threshold_derivative_exit[row],
                eta_log_sigma_exit: eta_log_sigma_exit[row],
                eta_log_sigma_entry: eta_log_sigma_entry[row],
                eta_log_sigma_derivative_exit: eta_log_sigma_derivative_exit[row],
                time_wiggle: time_wiggle_row,
                link_wiggle: link_wiggle_row,
            })
            .map_err(|reason| {
                invalid(format!(
                    "saved survival location-scale ALO row {row}: {reason}"
                ))
            })?;
        observed_hessians.push(row_geometry.observed_hessian);
        scores.push(row_geometry.nll_score);
        coordinate_values.push(row_geometry.coordinate_values);
    }

    let mut coordinate_designs = vec![
        input.time_base.design_entry.clone(),
        input.time_base.design_exit.clone(),
        input.time_base.design_derivative_exit.clone(),
        input.threshold.design_exit.clone(),
        input.threshold.design_entry.clone(),
        input.threshold.design_derivative_exit.clone(),
        input.log_sigma.design_exit.clone(),
        input.log_sigma.design_entry.clone(),
        input.log_sigma.design_derivative_exit.clone(),
    ];
    let time_base_range = time_range.start..time_range.start + time_base_width;
    let mut coordinate_ranges = vec![
        time_base_range.clone(),
        time_base_range.clone(),
        time_base_range,
        threshold_range.clone(),
        threshold_range.clone(),
        threshold_range,
        log_sigma_range.clone(),
        log_sigma_range.clone(),
        log_sigma_range,
    ];
    let mut coordinate_names = vec![
        "time-base-entry".to_string(),
        "time-base-exit".to_string(),
        "time-base-derivative-exit".to_string(),
        "threshold-exit".to_string(),
        "threshold-entry".to_string(),
        "threshold-derivative-exit".to_string(),
        "log-sigma-exit".to_string(),
        "log-sigma-entry".to_string(),
        "log-sigma-derivative-exit".to_string(),
    ];
    for coefficient in 0..time_wiggle_width {
        coordinate_designs.push(constant_scalar_design(n));
        let raw = time_range.start + time_base_width + coefficient;
        coordinate_ranges.push(raw..raw + 1);
        coordinate_names.push(format!("time-wiggle[{coefficient}]"));
    }
    if let Some(range) = link_range {
        if range.len() != link_wiggle_beta.len() {
            return Err(invalid(format!(
                "saved survival location-scale ALO link-wiggle block has {} coefficients; runtime has {}",
                range.len(),
                link_wiggle_beta.len(),
            )));
        }
        for coefficient in 0..range.len() {
            coordinate_designs.push(constant_scalar_design(n));
            let raw = range.start + coefficient;
            coordinate_ranges.push(raw..raw + 1);
            coordinate_names.push(format!("link-wiggle[{coefficient}]"));
        }
    }
    compute_saved_multicoordinate_core(
        class,
        coordinate_names,
        coordinate_designs,
        coordinate_ranges,
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

fn compute_saved_marginal_slope_survival_alo(
    model: &FittedModel,
    input: &SavedMarginalSlopeSurvivalAloInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::Survival;
    let likelihood_mode = require_saved_survival_likelihood_mode(model)
        .map_err(|error| invalid(format!("saved survival ALO likelihood mode: {error}")))?;
    if likelihood_mode != SurvivalLikelihoodMode::MarginalSlope {
        return Err(invalid(format!(
            "saved marginal-slope survival ALO row geometry cannot replay {likelihood_mode:?}"
        )));
    }
    let n = input.event.len();
    if observations.response.len() != n || observations.prior_weights.len() != n {
        return Err(invalid(format!(
            "saved survival marginal-slope ALO row mismatch: carrier={n}, response={}, weights={}",
            observations.response.len(),
            observations.prior_weights.len(),
        )));
    }
    for row in 0..n {
        if observations.response[row] != input.event[row] {
            return Err(invalid(format!(
                "saved survival marginal-slope ALO event mismatch at row {row}: carrier={}, observations={}",
                input.event[row], observations.response[row],
            )));
        }
    }

    let payload = model.payload();
    let fit = payload.fit_result.as_ref().ok_or_else(|| {
        invalid("saved survival marginal-slope ALO requires a canonical fitted coefficient state")
    })?;
    let runtime = model.saved_prediction_runtime().map_err(|error| {
        invalid(format!(
            "saved survival marginal-slope ALO runtime: {error}"
        ))
    })?;
    let expected_blocks = 3
        + usize::from(runtime.score_warp.is_some())
        + usize::from(runtime.link_deviation.is_some())
        + usize::from(payload.influence_absorber_width.is_some());
    if fit.blocks.len() != expected_blocks {
        return Err(invalid(format!(
            "saved survival marginal-slope ALO expects {expected_blocks} coefficient blocks, got {}",
            fit.blocks.len(),
        )));
    }
    let time_beta = &fit.blocks[0].beta;
    let marginal_beta = &fit.blocks[1].beta;
    let logslope_beta = &fit.blocks[2].beta;
    if input.design_entry.ncols() != time_beta.len()
        || input.marginal_design.ncols() != marginal_beta.len()
        || input.logslope_design.ncols() != logslope_beta.len()
    {
        return Err(invalid(format!(
            "saved survival marginal-slope ALO raw design/beta mismatch: time={}/{}, marginal={}/{}, logslope={}/{}",
            input.design_entry.ncols(),
            time_beta.len(),
            input.marginal_design.ncols(),
            marginal_beta.len(),
            input.logslope_design.ncols(),
            logslope_beta.len(),
        )));
    }
    let mut cursor = 3usize;
    let score_warp_beta = runtime.score_warp.as_ref().map(|_| {
        let beta = &fit.blocks[cursor].beta;
        cursor += 1;
        beta
    });
    let link_deviation_beta = runtime.link_deviation.as_ref().map(|_| {
        let beta = &fit.blocks[cursor].beta;
        cursor += 1;
        beta
    });
    let influence_beta = payload.influence_absorber_width.map(|_| {
        let beta = &fit.blocks[cursor].beta;
        cursor += 1;
        beta
    });
    if cursor != fit.blocks.len() {
        return Err(invalid(
            "saved survival marginal-slope ALO coefficient topology did not consume every fitted block",
        ));
    }

    let normalized_z = payload
        .latent_z_normalization
        .ok_or_else(|| {
            invalid("saved survival marginal-slope ALO is missing latent-z normalization")
        })?
        .apply(&input.latent_z, "saved survival marginal-slope ALO")
        .map_err(|error| invalid(error.to_string()))?;
    let time_wiggle_knots = payload
        .baseline_timewiggle_knots
        .as_ref()
        .map(|knots| Array1::from_vec(knots.clone()));
    let time_wiggle_ncols = payload
        .beta_baseline_timewiggle
        .as_ref()
        .map_or(0, Vec::len);
    let saved_score_covariance = payload
        .survival_marginal_slope_score_covariance
        .as_ref()
        .ok_or_else(|| {
            invalid(
                "saved survival marginal-slope ALO is missing its exact scalar score covariance",
            )
        })?;
    if saved_score_covariance.len() != 1 || saved_score_covariance[0].len() != 1 {
        return Err(invalid(format!(
            "saved survival marginal-slope ALO scalar score covariance must be 1x1, got {} rows with widths {:?}",
            saved_score_covariance.len(),
            saved_score_covariance
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
        )));
    }
    let score_variance = saved_score_covariance[0][0];
    let influence_design = match payload.influence_absorber_design.as_ref() {
        None => None,
        Some(rows) => {
            let width = payload.influence_absorber_width.ok_or_else(|| {
                invalid("saved survival marginal-slope ALO influence design has no persisted width")
            })?;
            if rows.len() != n || rows.iter().any(|row| row.len() != width) {
                return Err(invalid(format!(
                    "saved survival marginal-slope ALO influence design is not {n}x{width}"
                )));
            }
            Some(
                Array2::from_shape_vec(
                    (n, width),
                    rows.iter().flat_map(|row| row.iter().copied()).collect(),
                )
                .map_err(|error| invalid(format!("saved influence design shape: {error}")))?,
            )
        }
    };
    let gaussian_frailty_sd = match payload.family_state.frailty() {
        Some(gam_models::survival::lognormal_kernel::FrailtySpec::None) => None,
        Some(gam_models::survival::lognormal_kernel::FrailtySpec::GaussianShift {
            sigma_fixed: Some(sigma),
        }) => Some(*sigma),
        Some(frailty) => {
            return Err(invalid(format!(
                "saved survival marginal-slope ALO has a fitted frailty state forbidden by the marginal-slope schema: {frailty:?}"
            )));
        }
        None => {
            return Err(invalid(
                "saved survival marginal-slope ALO is missing its fitted frailty state",
            ));
        }
    };

    let replay =
        replay_saved_survival_marginal_slope_alo(SurvivalMarginalSlopeSavedAloReplayInput {
            design_entry: &input.design_entry,
            design_exit: &input.design_exit,
            design_derivative_exit: &input.design_derivative_exit,
            offset_entry: &input.offset_entry,
            offset_exit: &input.offset_exit,
            derivative_offset_exit: &input.derivative_offset_exit,
            marginal_design: &input.marginal_design,
            marginal_offset: &input.marginal_offset,
            logslope_design: &input.logslope_design,
            logslope_offset: &input.logslope_offset,
            latent_z: &normalized_z,
            event: &input.event,
            prior_weights: observations.prior_weights,
            score_variance,
            derivative_guard: gam_models::survival::survival_derivative_guard_for_likelihood(
                likelihood_mode,
            ),
            time_wiggle_knots: time_wiggle_knots.as_ref(),
            time_wiggle_degree: payload.baseline_timewiggle_degree,
            time_wiggle_ncols,
            time_beta,
            marginal_beta,
            logslope_beta,
            score_warp_beta,
            link_deviation_beta,
            influence_beta,
            score_warp_runtime: runtime.score_warp.as_ref(),
            link_deviation_runtime: runtime.link_deviation.as_ref(),
            influence_design: influence_design.as_ref(),
            gaussian_frailty_sd,
        })
        .map_err(|reason| {
            invalid(format!(
                "saved survival marginal-slope ALO replay: {reason}"
            ))
        })?;
    let parameter_dimension = fit.beta.len();
    let geometry = require_saved_geometry(model, class, parameter_dimension)?;
    if replay.block_dimensions.iter().sum::<usize>() != parameter_dimension {
        return Err(invalid(format!(
            "saved survival marginal-slope ALO row program has {} raw coefficients; fit has {parameter_dimension}",
            replay.block_dimensions.iter().sum::<usize>(),
        )));
    }
    let coordinate_names = replay
        .block_dimensions
        .iter()
        .enumerate()
        .flat_map(|(block, &width)| {
            let label = match block {
                0 => "time",
                1 => "marginal",
                2 => "logslope",
                _ if runtime.score_warp.is_some() && block == 3 => "score-warp",
                _ if runtime.link_deviation.is_some()
                    && block == 3 + usize::from(runtime.score_warp.is_some()) =>
                {
                    "link-deviation"
                }
                _ => "influence-absorber",
            };
            (0..width)
                .map(move |coefficient| format!("{label}[{coefficient}]"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let coordinate_designs = (0..parameter_dimension)
        .map(|_| constant_scalar_design(n))
        .collect::<Vec<_>>();
    let coordinate_ranges = (0..parameter_dimension)
        .map(|coefficient| coefficient..coefficient + 1)
        .collect::<Vec<_>>();
    let observed_hessians = replay
        .rows
        .iter()
        .map(|row| row.observed_hessian.clone())
        .collect::<Vec<_>>();
    let scores = replay
        .rows
        .iter()
        .map(|row| row.nll_score.clone())
        .collect::<Vec<_>>();
    let coordinate_values = replay
        .rows
        .into_iter()
        .map(|row| row.coordinate_values)
        .collect::<Vec<_>>();
    compute_saved_multicoordinate_core(
        class,
        coordinate_names,
        coordinate_designs,
        coordinate_ranges,
        geometry,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

/// Replay exact ALO from one saved-model authority for every fitted class.
///
/// The dispatcher never refits, substitutes a simpler model, or reconstructs
/// a penalized Hessian from covariance. `input` must contain the affine row
/// designs consumed by the fitted likelihood. For transformation-normal fits,
/// that means the persisted covariate design rather than the response-scale
/// prediction quadrature carrier.
pub fn compute_saved_model_alo(
    model: &FittedModel,
    input: &SavedModelAloInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    match model.predict_model_class() {
        PredictModelClass::Standard => compute_saved_standard_alo(
            model,
            input
                .require_affine(PredictModelClass::Standard)
                .map_err(invalid)?,
            &observations,
        ),
        PredictModelClass::GaussianLocationScale
        | PredictModelClass::BinomialLocationScale
        | PredictModelClass::DispersionLocationScale => {
            let class = model.predict_model_class();
            compute_saved_location_scale_alo(
                model,
                input.require_affine(class).map_err(invalid)?,
                observations,
            )
        }
        PredictModelClass::BernoulliMarginalSlope => compute_saved_bernoulli_marginal_slope_alo(
            model,
            input
                .require_affine(PredictModelClass::BernoulliMarginalSlope)
                .map_err(invalid)?,
            observations,
        ),
        PredictModelClass::TransformationNormal => compute_saved_transformation_normal_alo(
            model,
            &input
                .require_affine(PredictModelClass::TransformationNormal)
                .map_err(invalid)?
                .design,
            &input
                .require_affine(PredictModelClass::TransformationNormal)
                .map_err(invalid)?
                .offset,
            observations,
        ),
        PredictModelClass::Survival => match input {
            SavedModelAloInput::Survival(SavedSurvivalAloInput::CauseSpecific(input)) => {
                compute_saved_cause_specific_survival_alo(model, input, observations)
            }
            SavedModelAloInput::Survival(SavedSurvivalAloInput::MarginalSlope(input)) => {
                compute_saved_marginal_slope_survival_alo(model, input, observations)
            }
            SavedModelAloInput::Affine(_) => Err(invalid(
                "saved survival ALO requires typed entry/exit/derivative row geometry",
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_alo_pulls_raw_rows_through_rectangular_gauge() {
        // beta_raw = T theta_active + a with three reported coefficients but
        // only two fitted directions.  The local coordinate touches raw rows
        // 1..3, so its active Jacobian must be X_local * T[1..3, :].
        let gauge = Gauge {
            t_full: Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.5, 2.0, -1.0, 3.0])
                .expect("3x2 gauge"),
            affine_shift: Array1::from_vec(vec![4.0, -2.0, 1.0]),
            block_starts_raw: vec![0, 3],
            block_starts_reduced: vec![0, 2],
        };
        gauge.validate().expect("rectangular gauge is valid");
        let raw_design =
            Array2::from_shape_vec((2, 2), vec![2.0, 4.0, -1.0, 3.0]).expect("2x2 raw row design");
        let expected = raw_design.dot(&gauge.t_full.slice(s![1..3, ..]));

        let (active_designs, active_ranges) = pullback_saved_coordinate_designs(
            PredictModelClass::Standard,
            &gauge,
            vec![DesignMatrix::from(raw_design)],
            vec![1..3],
        )
        .expect("exact active-row pullback");

        assert_eq!(active_ranges, vec![0..2]);
        assert_eq!(active_designs[0].to_dense(), expected);
    }
}
