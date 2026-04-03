//! Jointly learned latent-frailty survival and binary deployment families with
//! a live time/baseline block.
//!
//! Model:
//!   H_0(a) = exp(q(a)),
//!   h_0(a) = dq(a)/da,
//!   H(a | U) = H_0(a) * exp(U),
//!   U ~ N(mu, sigma^2),
//!   mu = X beta + offset.
//!
//! Unlike the old compiled-row path, the cumulative masses and baseline hazard
//! are rebuilt inside the optimizer from the current time-basis coefficients.
//! The current family-level fit surface uses exact events and right censoring;
//! interval-censored rows exist at the kernel layer but are not exposed here.

use crate::estimate::UnifiedFitResult;
use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, fit_custom_family,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::lognormal_kernel::{
    FrailtySpec, HazardLoading, LatentSurvivalEventType, LatentSurvivalRow, LatentSurvivalRowJet,
    log_kernel_bundle,
};
use crate::families::sigma_link::{exp_sigma_eta_for_sigma_scalar, exp_sigma_from_eta_scalar};
use crate::families::survival_location_scale::{
    TimeBlockInput, project_onto_linear_constraints, structural_time_coefficient_constraints,
};
use crate::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::quadrature::{IntegratedExpectationMode, QuadratureContext};
use crate::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use std::sync::Arc;

const MIN_WEIGHT: f64 = 1e-12;

#[derive(Clone)]
pub struct LatentSurvivalTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
}

pub struct LatentSurvivalTermFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub latent_sd: f64,
}

#[derive(Clone)]
pub struct LatentBinaryTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
}

pub struct LatentBinaryTermFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
}

#[derive(Clone)]
struct PreparedLatentTimeBlock {
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    linear_constraints: Option<LinearInequalityConstraints>,
    penalties: Vec<Array2<f64>>,
    initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct LatentSurvivalFamily {
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub latent_sd_fixed: Option<f64>,
    pub hazard_loading: HazardLoading,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub x_time_entry: Array2<f64>,
    pub x_time_exit: Array2<f64>,
    pub x_time_derivative_exit: Array2<f64>,
    pub x_mean: DesignMatrix,
    pub time_linear_constraints: Option<LinearInequalityConstraints>,
    pub quadctx: Arc<QuadratureContext>,
}

#[derive(Clone)]
pub struct LatentBinaryFamily {
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub latent_sd: f64,
    pub hazard_loading: HazardLoading,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub x_time_entry: Array2<f64>,
    pub x_time_exit: Array2<f64>,
    pub x_mean: DesignMatrix,
    pub time_linear_constraints: Option<LinearInequalityConstraints>,
    pub quadctx: Arc<QuadratureContext>,
}

impl LatentSurvivalFamily {
    pub const BLOCK_TIME: usize = 0;
    pub const BLOCK_MEAN: usize = 1;
    pub const BLOCK_LOG_SIGMA: usize = 2;

    pub fn parameter_names() -> &'static [&'static str] {
        &["time_transform", "mean"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Identity]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "latent_survival",
            parameternames: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn split_time_eta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<
        (
            ArrayView1<'a, f64>,
            ArrayView1<'a, f64>,
            ArrayView1<'a, f64>,
            &'a Array1<f64>,
        ),
        String,
    > {
        let expected_blocks = if self.latent_sd_fixed.is_some() { 2 } else { 3 };
        if block_states.len() != expected_blocks {
            return Err(format!(
                "LatentSurvivalFamily expects {expected_blocks} blocks, got {}",
                block_states.len(),
            ));
        }
        let n = self.event_target.len();
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_mean = &block_states[Self::BLOCK_MEAN].eta;
        if eta_time.len() != 3 * n {
            return Err(format!(
                "latent survival time eta length mismatch: got {}, expected {}",
                eta_time.len(),
                3 * n
            ));
        }
        if eta_mean.len() != n || self.weights.len() != n {
            return Err("latent survival mean eta dimension mismatch".to_string());
        }
        Ok((
            eta_time.slice(s![0..n]),
            eta_time.slice(s![n..2 * n]),
            eta_time.slice(s![2 * n..3 * n]),
            eta_mean,
        ))
    }

    fn latent_sd(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if let Some(sigma) = self.latent_sd_fixed {
            return Ok(sigma);
        }
        let eta = *block_states
            .get(Self::BLOCK_LOG_SIGMA)
            .and_then(|state| state.eta.get(0))
            .ok_or_else(|| "latent survival learnable log_sigma block is missing".to_string())?;
        let sigma = exp_sigma_from_eta_scalar(eta);
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(format!(
                "latent survival learnable sigma became invalid: log_sigma={eta}, sigma={sigma}"
            ));
        }
        Ok(sigma)
    }
}

impl LatentBinaryFamily {
    pub const BLOCK_TIME: usize = 0;
    pub const BLOCK_MEAN: usize = 1;

    fn split_time_eta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<(ArrayView1<'a, f64>, ArrayView1<'a, f64>, &'a Array1<f64>), String> {
        if block_states.len() != 2 {
            return Err(format!(
                "LatentBinaryFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.event_target.len();
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_mean = &block_states[Self::BLOCK_MEAN].eta;
        if eta_time.len() != 3 * n {
            return Err(format!(
                "latent binary time eta length mismatch: got {}, expected {}",
                eta_time.len(),
                3 * n
            ));
        }
        if eta_mean.len() != n || self.weights.len() != n {
            return Err("latent binary mean eta dimension mismatch".to_string());
        }
        Ok((
            eta_time.slice(s![0..n]),
            eta_time.slice(s![n..2 * n]),
            eta_mean,
        ))
    }
}

pub fn fixed_latent_hazard_frailty(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<(f64, HazardLoading), String> {
    match frailty {
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading,
        } if sigma.is_finite() && *sigma >= 0.0 => Ok((*sigma, *loading)),
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            ..
        } => Err(format!(
            "{context} requires a finite fixed hazard-multiplier sigma >= 0, got {sigma}"
        )),
        FrailtySpec::HazardMultiplier {
            sigma_fixed: None, ..
        } => Err(format!(
            "{context} requires a fixed hazard-multiplier sigma; learnable sigma is not implemented for this family"
        )),
        FrailtySpec::GaussianShift { .. } => Err(format!(
            "{context} requires HazardMultiplier frailty, not GaussianShift"
        )),
        FrailtySpec::None => Err(format!(
            "{context} requires a fixed HazardMultiplier frailty specification"
        )),
    }
}

pub fn latent_hazard_loading(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<HazardLoading, String> {
    match frailty {
        FrailtySpec::HazardMultiplier { loading, .. } => Ok(*loading),
        FrailtySpec::GaussianShift { .. } => Err(format!(
            "{context} requires HazardMultiplier frailty, not GaussianShift"
        )),
        FrailtySpec::None => Err(format!(
            "{context} requires a HazardMultiplier frailty specification"
        )),
    }
}

#[derive(Clone, Copy)]
struct LatentSurvivalTimeJet {
    grad_entry: f64,
    grad_exit: f64,
    grad_qdot: f64,
    neg_hess_entry: f64,
    neg_hess_exit: f64,
    neg_hess_qdot: f64,
    neg_hess_exit_qdot: f64,
}

pub fn fit_latent_survival_terms(
    data: ArrayView2<'_, f64>,
    spec: LatentSurvivalTermSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentSurvivalTermFitResult, String> {
    let latent_sd = validate_latent_survival_inputs(data, &spec, &frailty)?;
    let hazard_loading = latent_hazard_loading(&frailty, "latent-survival")?;
    let mean_design =
        build_term_collection_design(data, &spec.meanspec).map_err(|e| e.to_string())?;
    let resolvedspec = freeze_term_collection_from_design(&spec.meanspec, &mean_design)
        .map_err(|e| e.to_string())?;
    let time_prepared = prepare_latent_time_block(&spec.time_block, spec.derivative_guard)?;

    let family = LatentSurvivalFamily {
        event_target: spec.event_target.clone(),
        weights: spec.weights.clone(),
        latent_sd_fixed: latent_sd,
        hazard_loading,
        unloaded_mass_entry: spec.unloaded_mass_entry.clone(),
        unloaded_mass_exit: spec.unloaded_mass_exit.clone(),
        unloaded_hazard_exit: spec.unloaded_hazard_exit.clone(),
        x_time_entry: time_prepared.design_entry.clone(),
        x_time_exit: time_prepared.design_exit.clone(),
        x_time_derivative_exit: time_prepared.design_derivative_exit.clone(),
        x_mean: mean_design.design.clone(),
        time_linear_constraints: time_prepared.linear_constraints.clone(),
        quadctx: Arc::new(QuadratureContext::new()),
    };

    let mut blocks = vec![
        build_time_blockspec(&time_prepared, &spec.time_block),
        build_mean_blockspec(&mean_design, spec.mean_offset.clone()),
    ];
    if latent_sd.is_none() {
        blocks.push(build_log_sigma_blockspec(0.5));
    }
    let fit = fit_custom_family(&family, &blocks, options).map_err(|e| e.to_string())?;
    let latent_sd = family.latent_sd(&fit.block_states)?;
    Ok(LatentSurvivalTermFitResult {
        fit,
        design: mean_design,
        resolvedspec,
        latent_sd,
    })
}

pub fn fit_latent_binary_terms(
    data: ArrayView2<'_, f64>,
    spec: LatentBinaryTermSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentBinaryTermFitResult, String> {
    let latent_sd = validate_latent_binary_inputs(data, &spec, &frailty)?;
    let (_, hazard_loading) = fixed_latent_hazard_frailty(&frailty, "latent-binary")?;
    let mean_design =
        build_term_collection_design(data, &spec.meanspec).map_err(|e| e.to_string())?;
    let resolvedspec = freeze_term_collection_from_design(&spec.meanspec, &mean_design)
        .map_err(|e| e.to_string())?;
    let time_prepared = prepare_latent_time_block(&spec.time_block, spec.derivative_guard)?;

    let family = LatentBinaryFamily {
        event_target: spec.event_target.clone(),
        weights: spec.weights.clone(),
        latent_sd,
        hazard_loading,
        unloaded_mass_entry: spec.unloaded_mass_entry.clone(),
        unloaded_mass_exit: spec.unloaded_mass_exit.clone(),
        x_time_entry: time_prepared.design_entry.clone(),
        x_time_exit: time_prepared.design_exit.clone(),
        x_mean: mean_design.design.clone(),
        time_linear_constraints: time_prepared.linear_constraints.clone(),
        quadctx: Arc::new(QuadratureContext::new()),
    };

    let blocks = vec![
        build_time_blockspec(&time_prepared, &spec.time_block),
        build_mean_blockspec(&mean_design, spec.mean_offset.clone()),
    ];
    let fit = fit_custom_family(&family, &blocks, options).map_err(|e| e.to_string())?;
    Ok(LatentBinaryTermFitResult {
        fit,
        design: mean_design,
        resolvedspec,
    })
}

fn validate_latent_survival_inputs(
    data: ArrayView2<'_, f64>,
    spec: &LatentSurvivalTermSpec,
    frailty: &FrailtySpec,
) -> Result<Option<f64>, String> {
    let (sigma, hazard_loading) = match frailty {
        FrailtySpec::HazardMultiplier {
            sigma_fixed,
            loading,
        } => {
            if let Some(sigma) = sigma_fixed
                && (!sigma.is_finite() || *sigma < 0.0)
            {
                return Err(format!(
                    "latent-survival requires a finite hazard-multiplier sigma >= 0, got {sigma}"
                ));
            }
            (*sigma_fixed, *loading)
        }
        FrailtySpec::GaussianShift { .. } => {
            return Err(
                "latent-survival requires HazardMultiplier frailty, not GaussianShift".to_string(),
            );
        }
        FrailtySpec::None => {
            return Err(
                "latent-survival requires a HazardMultiplier frailty specification".to_string(),
            );
        }
    };
    let n = data.nrows();
    if n == 0 {
        return Err("latent-survival requires a non-empty dataset".to_string());
    }
    if spec.age_entry.len() != n
        || spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.unloaded_mass_entry.len() != n
        || spec.unloaded_mass_exit.len() != n
        || spec.unloaded_hazard_exit.len() != n
        || spec.mean_offset.len() != n
    {
        return Err(format!(
            "latent-survival size mismatch: data has {n} rows, entry={}, exit={}, event={}, weights={}, unloaded_entry={}, unloaded_exit={}, unloaded_hazard={}, offset={}",
            spec.age_entry.len(),
            spec.age_exit.len(),
            spec.event_target.len(),
            spec.weights.len(),
            spec.unloaded_mass_entry.len(),
            spec.unloaded_mass_exit.len(),
            spec.unloaded_hazard_exit.len(),
            spec.mean_offset.len()
        ));
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard < 0.0 {
        return Err(format!(
            "latent-survival derivative_guard must be finite and >= 0, got {}",
            spec.derivative_guard
        ));
    }
    for i in 0..n {
        let entry = spec.age_entry[i];
        let exit = spec.age_exit[i];
        let event = spec.event_target[i];
        let weight = spec.weights[i];
        let unloaded_entry = spec.unloaded_mass_entry[i];
        let unloaded_exit = spec.unloaded_mass_exit[i];
        let unloaded_hazard = spec.unloaded_hazard_exit[i];
        if !entry.is_finite() || !exit.is_finite() {
            return Err(format!(
                "latent-survival row {} has non-finite entry/exit ages: entry={}, exit={}",
                i + 1,
                entry,
                exit
            ));
        }
        if entry < 0.0 || exit < entry {
            return Err(format!(
                "latent-survival row {} has invalid delayed-entry bounds: entry={}, exit={}",
                i + 1,
                entry,
                exit
            ));
        }
        if event > 1 {
            return Err(format!(
                "latent-survival row {} has invalid event target {}; expected 0 or 1",
                i + 1,
                event
            ));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "latent-survival row {} has invalid weight {}; expected a finite non-negative weight",
                i + 1,
                weight
            ));
        }
        if !unloaded_entry.is_finite()
            || !unloaded_exit.is_finite()
            || !unloaded_hazard.is_finite()
            || unloaded_entry < 0.0
            || unloaded_exit < unloaded_entry
            || unloaded_hazard < 0.0
        {
            return Err(format!(
                "latent-survival row {} has invalid unloaded hazard decomposition: entry_mass={}, exit_mass={}, exit_hazard={}",
                i + 1,
                unloaded_entry,
                unloaded_exit,
                unloaded_hazard
            ));
        }
        validate_unloaded_components_for_loading(
            "latent-survival",
            i,
            hazard_loading,
            unloaded_entry,
            unloaded_exit,
            Some(unloaded_hazard),
        )?;
    }
    let time_block = &spec.time_block;
    let p_time = time_block.design_exit.ncols();
    if time_block.design_entry.nrows() != n
        || time_block.design_exit.nrows() != n
        || time_block.design_derivative_exit.nrows() != n
    {
        return Err(format!(
            "latent-survival time block row mismatch: n={}, entry_rows={}, exit_rows={}, derivative_rows={}",
            n,
            time_block.design_entry.nrows(),
            time_block.design_exit.nrows(),
            time_block.design_derivative_exit.nrows()
        ));
    }
    if time_block.design_entry.ncols() != p_time
        || time_block.design_derivative_exit.ncols() != p_time
    {
        return Err(format!(
            "latent-survival time block column mismatch: entry_cols={}, exit_cols={}, derivative_cols={}",
            time_block.design_entry.ncols(),
            time_block.design_exit.ncols(),
            time_block.design_derivative_exit.ncols()
        ));
    }
    if time_block.offset_entry.len() != n
        || time_block.offset_exit.len() != n
        || time_block.derivative_offset_exit.len() != n
    {
        return Err(format!(
            "latent-survival time block offset mismatch: n={}, entry_offset={}, exit_offset={}, derivative_offset={}",
            n,
            time_block.offset_entry.len(),
            time_block.offset_exit.len(),
            time_block.derivative_offset_exit.len()
        ));
    }
    Ok(sigma)
}

fn validate_unloaded_components_for_loading(
    context: &str,
    row_index: usize,
    loading: HazardLoading,
    unloaded_entry: f64,
    unloaded_exit: f64,
    unloaded_hazard: Option<f64>,
) -> Result<(), String> {
    match loading {
        HazardLoading::Full => {
            if unloaded_entry != 0.0
                || unloaded_exit != 0.0
                || unloaded_hazard.is_some_and(|hazard| hazard != 0.0)
            {
                return Err(format!(
                    "{context} row {} uses full hazard loading, so unloaded components must be exactly zero; got entry_mass={}, exit_mass={}, exit_hazard={}",
                    row_index + 1,
                    unloaded_entry,
                    unloaded_exit,
                    unloaded_hazard.unwrap_or(0.0)
                ));
            }
        }
        HazardLoading::LoadedVsUnloaded => {}
    }
    Ok(())
}

fn validate_latent_binary_inputs(
    data: ArrayView2<'_, f64>,
    spec: &LatentBinaryTermSpec,
    frailty: &FrailtySpec,
) -> Result<f64, String> {
    let (sigma, hazard_loading) = fixed_latent_hazard_frailty(frailty, "latent-binary")?;
    let n = data.nrows();
    if n == 0 {
        return Err("latent-binary requires a non-empty dataset".to_string());
    }
    if spec.age_entry.len() != n
        || spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.unloaded_mass_entry.len() != n
        || spec.unloaded_mass_exit.len() != n
        || spec.mean_offset.len() != n
    {
        return Err(format!(
            "latent-binary size mismatch: data has {n} rows, entry={}, exit={}, event={}, weights={}, unloaded_entry={}, unloaded_exit={}, offset={}",
            spec.age_entry.len(),
            spec.age_exit.len(),
            spec.event_target.len(),
            spec.weights.len(),
            spec.unloaded_mass_entry.len(),
            spec.unloaded_mass_exit.len(),
            spec.mean_offset.len()
        ));
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard < 0.0 {
        return Err(format!(
            "latent-binary derivative_guard must be finite and >= 0, got {}",
            spec.derivative_guard
        ));
    }
    for i in 0..n {
        let entry = spec.age_entry[i];
        let exit = spec.age_exit[i];
        let weight = spec.weights[i];
        let unloaded_entry = spec.unloaded_mass_entry[i];
        let unloaded_exit = spec.unloaded_mass_exit[i];
        let event = spec.event_target[i];
        if !entry.is_finite() || !exit.is_finite() {
            return Err(format!(
                "latent-binary row {} has non-finite entry/exit ages: entry={}, exit={}",
                i + 1,
                entry,
                exit
            ));
        }
        if entry < 0.0 || exit < entry {
            return Err(format!(
                "latent-binary row {} has invalid delayed-entry bounds: entry={}, exit={}",
                i + 1,
                entry,
                exit
            ));
        }
        if event > 1 {
            return Err(format!(
                "latent-binary row {} has invalid event target {}; expected 0 or 1",
                i + 1,
                event
            ));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "latent-binary row {} has invalid weight {}; expected a finite non-negative weight",
                i + 1,
                weight
            ));
        }
        if !unloaded_entry.is_finite()
            || !unloaded_exit.is_finite()
            || unloaded_entry < 0.0
            || unloaded_exit < unloaded_entry
        {
            return Err(format!(
                "latent-binary row {} has invalid unloaded mass decomposition: entry_mass={}, exit_mass={}",
                i + 1,
                unloaded_entry,
                unloaded_exit,
            ));
        }
        validate_unloaded_components_for_loading(
            "latent-binary",
            i,
            hazard_loading,
            unloaded_entry,
            unloaded_exit,
            None,
        )?;
    }
    let time_block = &spec.time_block;
    let p_time = time_block.design_exit.ncols();
    if time_block.design_entry.nrows() != n
        || time_block.design_exit.nrows() != n
        || time_block.design_derivative_exit.nrows() != n
    {
        return Err(format!(
            "latent-binary time block row mismatch: n={}, entry_rows={}, exit_rows={}, derivative_rows={}",
            n,
            time_block.design_entry.nrows(),
            time_block.design_exit.nrows(),
            time_block.design_derivative_exit.nrows()
        ));
    }
    if time_block.design_entry.ncols() != p_time
        || time_block.design_derivative_exit.ncols() != p_time
    {
        return Err(format!(
            "latent-binary time block column mismatch: entry_cols={}, exit_cols={}, derivative_cols={}",
            time_block.design_entry.ncols(),
            time_block.design_exit.ncols(),
            time_block.design_derivative_exit.ncols()
        ));
    }
    if time_block.offset_entry.len() != n
        || time_block.offset_exit.len() != n
        || time_block.derivative_offset_exit.len() != n
    {
        return Err(format!(
            "latent-binary time block offset mismatch: n={}, entry_offset={}, exit_offset={}, derivative_offset={}",
            n,
            time_block.offset_entry.len(),
            time_block.offset_exit.len(),
            time_block.derivative_offset_exit.len()
        ));
    }
    Ok(sigma)
}

fn prepare_latent_time_block(
    input: &TimeBlockInput,
    derivative_guard: f64,
) -> Result<PreparedLatentTimeBlock, String> {
    if !input.structural_monotonicity {
        return Err(
            "latent survival requires a structurally monotone time block; non-structural time transforms are unsupported"
                .to_string(),
        );
    }
    let design_entry = input.design_entry.to_dense();
    let design_exit = input.design_exit.to_dense();
    let design_derivative_exit = input.design_derivative_exit.to_dense();
    let linear_constraints = structural_time_coefficient_constraints(
        &design_derivative_exit,
        &input.derivative_offset_exit,
        derivative_guard,
    )?;
    let initial_beta = linear_constraints.as_ref().map(|constraints| {
        project_onto_linear_constraints(
            design_exit.ncols(),
            constraints,
            input.initial_beta.as_ref(),
        )
    });
    Ok(PreparedLatentTimeBlock {
        design_entry,
        design_exit,
        design_derivative_exit,
        linear_constraints,
        penalties: input.penalties.clone(),
        initial_beta,
    })
}

fn stack_rows(blocks: &[&Array2<f64>]) -> Array2<f64> {
    let ncols = blocks.first().map_or(0, |m| m.ncols());
    let nrows = blocks.iter().map(|m| m.nrows()).sum();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    let mut row = 0usize;
    for block in blocks {
        let end = row + block.nrows();
        out.slice_mut(s![row..end, ..]).assign(block);
        row = end;
    }
    out
}

fn stack_offsets(blocks: &[&Array1<f64>]) -> Array1<f64> {
    let n: usize = blocks.iter().map(|v| v.len()).sum();
    let mut out = Array1::<f64>::zeros(n);
    let mut row = 0usize;
    for block in blocks {
        let end = row + block.len();
        out.slice_mut(s![row..end]).assign(block);
        row = end;
    }
    out
}

fn build_time_blockspec(
    prepared: &PreparedLatentTimeBlock,
    input: &TimeBlockInput,
) -> ParameterBlockSpec {
    let stacked_design = stack_rows(&[
        &prepared.design_entry,
        &prepared.design_exit,
        &prepared.design_derivative_exit,
    ]);
    ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(stacked_design))),
        offset: stack_offsets(&[
            &input.offset_entry,
            &input.offset_exit,
            &input.derivative_offset_exit,
        ]),
        penalties: prepared
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: input.nullspace_dims.clone(),
        initial_log_lambdas: input
            .initial_log_lambdas
            .clone()
            .unwrap_or_else(|| Array1::zeros(prepared.penalties.len())),
        initial_beta: prepared.initial_beta.clone(),
    }
}

fn build_mean_blockspec(design: &TermCollectionDesign, offset: Array1<f64>) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "mean".to_string(),
        design: design.design.clone(),
        offset,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: Array1::zeros(design.penalties.len()),
        initial_beta: None,
    }
}

fn build_log_sigma_blockspec(initial_sigma: f64) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(Array2::from_elem(
            (1, 1),
            1.0,
        )))),
        offset: Array1::zeros(1),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::from_elem(
            1,
            exp_sigma_eta_for_sigma_scalar(initial_sigma),
        )),
    }
}

fn log_kernel_ratio(
    bundle: &crate::families::lognormal_kernel::LogLognormalKernelBundle,
    num: usize,
    den: usize,
) -> f64 {
    let delta = bundle.get(num) - bundle.get(den);
    if delta.is_finite() {
        delta.exp()
    } else if delta > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

fn logk_q_derivatives(
    quadctx: &QuadratureContext,
    k: usize,
    mass: f64,
    mu: f64,
    sigma: f64,
) -> Result<(f64, f64, IntegratedExpectationMode), String> {
    if mass <= 0.0 {
        return Ok((0.0, 0.0, IntegratedExpectationMode::ExactClosedForm));
    }
    let bundle = log_kernel_bundle(quadctx, mass, mu, sigma, k + 2)
        .map_err(|e| format!("latent survival kernel evaluation failed: {e}"))?;
    let r1 = log_kernel_ratio(&bundle, k + 1, k);
    let r2 = log_kernel_ratio(&bundle, k + 2, k);
    let d1 = -mass * r1;
    let d2 = d1 + mass * mass * (r2 - r1 * r1);
    Ok((d1, d2, bundle.mode))
}

fn latent_survival_time_jet(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    qdot_exit: f64,
    mu: f64,
    sigma: f64,
) -> Result<LatentSurvivalTimeJet, String> {
    let (entry_d1, entry_d2, _) = logk_q_derivatives(quadctx, 0, row.mass_entry, mu, sigma)?;
    match row.event_type {
        LatentSurvivalEventType::RightCensored => {
            let (exit_d1, exit_d2, _) = logk_q_derivatives(quadctx, 0, row.mass_exit, mu, sigma)?;
            Ok(LatentSurvivalTimeJet {
                grad_entry: -entry_d1,
                grad_exit: exit_d1,
                grad_qdot: 0.0,
                neg_hess_entry: entry_d2,
                neg_hess_exit: -exit_d2,
                neg_hess_qdot: 0.0,
                neg_hess_exit_qdot: 0.0,
            })
        }
        LatentSurvivalEventType::ExactEvent => {
            if !(qdot_exit.is_finite() && qdot_exit > 0.0) {
                return Err(format!(
                    "latent survival requires positive finite baseline hazard derivative, got {qdot_exit}"
                ));
            }
            if row.hazard_unloaded > 0.0 {
                let bundle = log_kernel_bundle(quadctx, row.mass_exit, mu, sigma, 3)
                    .map_err(|e| format!("latent survival kernel evaluation failed: {e}"))?;
                let (unloaded_d1, unloaded_d2, _) =
                    logk_q_derivatives(quadctx, 0, row.mass_exit, mu, sigma)?;
                let (loaded_log_d1, loaded_d2, _) =
                    logk_q_derivatives(quadctx, 1, row.mass_exit, mu, sigma)?;
                let loaded_d1 = 1.0 + loaded_log_d1;
                let log_loaded = row.hazard_loaded.ln() + bundle.get(1);
                let log_unloaded = row.hazard_unloaded.ln() + bundle.get(0);
                let shift = log_loaded.max(log_unloaded);
                let loaded_weight = (log_loaded - shift).exp();
                let unloaded_weight = (log_unloaded - shift).exp();
                let normalizer = loaded_weight + unloaded_weight;
                if !(normalizer.is_finite() && normalizer > 0.0) {
                    return Err(
                        "latent survival exact-event numerator became non-finite under loaded/unloaded hazard decomposition"
                            .to_string(),
                    );
                }
                let w_loaded = loaded_weight / normalizer;
                let w_unloaded = unloaded_weight / normalizer;
                let grad_exit = w_loaded * loaded_d1 + w_unloaded * unloaded_d1;
                let grad_qdot = w_loaded / qdot_exit;
                let d2_exit = w_loaded * (loaded_d2 + loaded_d1 * loaded_d1)
                    + w_unloaded * (unloaded_d2 + unloaded_d1 * unloaded_d1)
                    - grad_exit * grad_exit;
                let d2_qdot = -grad_qdot * grad_qdot;
                let d2_exit_qdot = grad_qdot * (loaded_d1 - grad_exit);
                Ok(LatentSurvivalTimeJet {
                    grad_entry: -entry_d1,
                    grad_exit,
                    grad_qdot,
                    neg_hess_entry: entry_d2,
                    neg_hess_exit: -d2_exit,
                    neg_hess_qdot: -d2_qdot,
                    neg_hess_exit_qdot: -d2_exit_qdot,
                })
            } else {
                let (exit_d1, exit_d2, _) =
                    logk_q_derivatives(quadctx, 1, row.mass_exit, mu, sigma)?;
                Ok(LatentSurvivalTimeJet {
                    grad_entry: -entry_d1,
                    grad_exit: 1.0 + exit_d1,
                    grad_qdot: qdot_exit.recip(),
                    neg_hess_entry: entry_d2,
                    neg_hess_exit: -exit_d2,
                    neg_hess_qdot: qdot_exit.recip().powi(2),
                    neg_hess_exit_qdot: 0.0,
                })
            }
        }
        LatentSurvivalEventType::IntervalCensored => Err(
            "latent survival dynamic time derivatives do not implement interval censoring"
                .to_string(),
        ),
    }
}

fn dense_outer_accumulate(target: &mut Array2<f64>, weight: f64, x: ArrayView1<'_, f64>) {
    for a in 0..x.len() {
        let xa = x[a];
        if xa == 0.0 {
            continue;
        }
        for b in 0..x.len() {
            let xb = x[b];
            if xb == 0.0 {
                continue;
            }
            target[[a, b]] += weight * xa * xb;
        }
    }
}

fn dense_symmetric_cross_accumulate(
    target: &mut Array2<f64>,
    weight: f64,
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) {
    for a in 0..x.len() {
        let xa = x[a];
        let ya = y[a];
        if xa == 0.0 && ya == 0.0 {
            continue;
        }
        for b in 0..x.len() {
            let xb = x[b];
            let yb = y[b];
            let contribution = xa * yb + ya * xb;
            if contribution == 0.0 {
                continue;
            }
            target[[a, b]] += weight * contribution;
        }
    }
}

fn build_latent_survival_row(
    row_index: usize,
    hazard_loading: HazardLoading,
    event_type: LatentSurvivalEventType,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    unloaded_mass_entry: f64,
    unloaded_mass_exit: f64,
    unloaded_hazard_exit: f64,
) -> Result<LatentSurvivalRow, String> {
    if !(q_entry.is_finite() && q_exit.is_finite()) {
        return Err(format!(
            "latent survival requires finite q_entry and q_exit, got q_entry={q_entry}, q_exit={q_exit}"
        ));
    }
    if q_exit < q_entry {
        return Err(format!(
            "latent survival requires q_exit >= q_entry so cumulative mass is monotone, got q_entry={q_entry}, q_exit={q_exit}"
        ));
    }
    if !(unloaded_mass_entry.is_finite()
        && unloaded_mass_exit.is_finite()
        && unloaded_hazard_exit.is_finite())
    {
        return Err(format!(
            "latent survival requires finite unloaded components, got entry_mass={unloaded_mass_entry}, exit_mass={unloaded_mass_exit}, exit_hazard={unloaded_hazard_exit}"
        ));
    }
    if unloaded_mass_entry < 0.0
        || unloaded_mass_exit < unloaded_mass_entry
        || unloaded_hazard_exit < 0.0
    {
        return Err(format!(
            "latent survival requires unloaded masses/hazard to be non-negative and monotone, got entry_mass={unloaded_mass_entry}, exit_mass={unloaded_mass_exit}, exit_hazard={unloaded_hazard_exit}"
        ));
    }
    validate_unloaded_components_for_loading(
        "latent-survival",
        row_index,
        hazard_loading,
        unloaded_mass_entry,
        unloaded_mass_exit,
        Some(unloaded_hazard_exit),
    )?;
    let mass_entry = q_entry.exp();
    let mass_exit = q_exit.exp();
    let row = match event_type {
        LatentSurvivalEventType::RightCensored => LatentSurvivalRow::right_censored(
            mass_entry,
            mass_exit,
            unloaded_mass_entry,
            unloaded_mass_exit,
        ),
        LatentSurvivalEventType::ExactEvent => LatentSurvivalRow::exact_event(
            mass_entry,
            mass_exit,
            unloaded_mass_entry,
            unloaded_mass_exit,
            mass_exit
                * if qdot_exit.is_finite() && qdot_exit > 0.0 {
                    qdot_exit
                } else {
                    return Err(format!(
                        "latent survival exact event requires positive finite baseline hazard derivative, got {qdot_exit}"
                    ));
                },
            unloaded_hazard_exit,
        ),
        LatentSurvivalEventType::IntervalCensored => unreachable!(
            "latent survival fit path currently exposes only exact events and right censoring"
        ),
    };
    row.validate().map_err(|e| e.to_string())?;
    Ok(row)
}

#[derive(Clone, Copy)]
struct BinaryFromLogSurvival {
    log_lik: f64,
    grad_scale: f64,
    neg_hess_scale: f64,
    outer_scale: f64,
}

fn binary_from_log_survival(log_survival: f64, event: u8) -> Result<BinaryFromLogSurvival, String> {
    if event == 0 {
        return Ok(BinaryFromLogSurvival {
            log_lik: log_survival,
            grad_scale: 1.0,
            neg_hess_scale: 1.0,
            outer_scale: 0.0,
        });
    }
    if event != 1 {
        return Err(format!(
            "latent-binary requires event targets in {{0,1}}, got {event}"
        ));
    }
    let log_survival = log_survival.min(-1e-15);
    let survival = log_survival.exp();
    let event_prob = 1.0 - survival;
    if !(event_prob.is_finite() && event_prob > 0.0) {
        return Err(format!(
            "latent-binary encountered non-positive event probability from log survival {log_survival}"
        ));
    }
    Ok(BinaryFromLogSurvival {
        log_lik: event_prob.ln(),
        grad_scale: -survival / event_prob,
        neg_hess_scale: survival / event_prob,
        outer_scale: survival / (event_prob * event_prob),
    })
}

impl CustomFamily for LatentSurvivalFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let latent_sd = self.latent_sd(block_states)?;
        let n = self.event_target.len();
        let p_time = self.x_time_exit.ncols();
        let p_mean = self.x_mean.ncols();

        let mut ll = 0.0;
        let mut grad_time = Array1::<f64>::zeros(p_time);
        let mut hess_time = Array2::<f64>::zeros((p_time, p_time));
        let mut grad_mean = Array1::<f64>::zeros(p_mean);
        let mut hess_mean = Array2::<f64>::zeros((p_mean, p_mean));
        let mut grad_log_sigma = 0.0_f64;
        let mut hess_log_sigma = 0.0_f64;

        for i in 0..n {
            let wi = self.weights[i];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let event_type = if self.event_target[i] >= 1 {
                LatentSurvivalEventType::ExactEvent
            } else {
                LatentSurvivalEventType::RightCensored
            };
            if !(q_entry[i].is_finite() && q_exit[i].is_finite() && mu[i].is_finite()) {
                return Err(format!(
                    "latent survival row {i} contains non-finite predictors: q_entry={}, q_exit={}, mu={}",
                    q_entry[i], q_exit[i], mu[i]
                ));
            }
            if matches!(event_type, LatentSurvivalEventType::ExactEvent)
                && !(qdot_exit[i].is_finite() && qdot_exit[i] > 0.0)
            {
                return Err(format!(
                    "latent survival row {i} has non-positive baseline derivative {}",
                    qdot_exit[i]
                ));
            }

            let row = build_latent_survival_row(
                i,
                self.hazard_loading,
                event_type,
                q_entry[i],
                q_exit[i],
                qdot_exit[i],
                self.unloaded_mass_entry[i],
                self.unloaded_mass_exit[i],
                self.unloaded_hazard_exit[i],
            )?;
            let row_jet = LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], latent_sd)
                .map_err(|e| format!("LatentSurvivalFamily row {i}: {e}"))?;
            ll += wi * row_jet.log_lik;

            let mean_row = self.x_mean.row_chunk(i..i + 1);
            let mean_vec = mean_row.row(0);
            for j in 0..p_mean {
                grad_mean[j] += wi * row_jet.score * mean_vec[j];
            }
            dense_outer_accumulate(&mut hess_mean, wi * row_jet.neg_hessian, mean_vec);

            grad_log_sigma += wi * row_jet.score_log_sigma;
            hess_log_sigma += wi * row_jet.neg_hessian_log_sigma;

            let time_jet =
                latent_survival_time_jet(&self.quadctx, &row, qdot_exit[i], mu[i], latent_sd)?;
            let t_entry = self.x_time_entry.row(i);
            let t_exit = self.x_time_exit.row(i);
            let t_deriv = self.x_time_derivative_exit.row(i);
            for j in 0..p_time {
                grad_time[j] += wi
                    * (time_jet.grad_entry * t_entry[j]
                        + time_jet.grad_exit * t_exit[j]
                        + time_jet.grad_qdot * t_deriv[j]);
            }
            dense_outer_accumulate(&mut hess_time, wi * time_jet.neg_hess_entry, t_entry);
            dense_outer_accumulate(&mut hess_time, wi * time_jet.neg_hess_exit, t_exit);
            if time_jet.neg_hess_qdot != 0.0 {
                dense_outer_accumulate(&mut hess_time, wi * time_jet.neg_hess_qdot, t_deriv);
            }
            if time_jet.neg_hess_exit_qdot != 0.0 {
                dense_symmetric_cross_accumulate(
                    &mut hess_time,
                    wi * time_jet.neg_hess_exit_qdot,
                    t_exit,
                    t_deriv,
                );
            }
        }

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: grad_time,
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_mean,
                hessian: SymmetricMatrix::Dense(hess_mean),
            },
        ];
        if self.latent_sd_fixed.is_none() {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: Array1::from_elem(1, grad_log_sigma),
                hessian: SymmetricMatrix::Dense(Array2::from_elem((1, 1), hess_log_sigma)),
            });
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let latent_sd = self.latent_sd(block_states)?;
        let n = self.event_target.len();
        let mut ll = 0.0;
        for i in 0..n {
            let wi = self.weights[i];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let event_type = if self.event_target[i] >= 1 {
                LatentSurvivalEventType::ExactEvent
            } else {
                LatentSurvivalEventType::RightCensored
            };
            let row = build_latent_survival_row(
                i,
                self.hazard_loading,
                event_type,
                q_entry[i],
                q_exit[i],
                qdot_exit[i],
                self.unloaded_mass_entry[i],
                self.unloaded_mass_exit[i],
                self.unloaded_hazard_exit[i],
            )?;
            let jet = LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], latent_sd)
                .map_err(|e| format!("LatentSurvivalFamily row {i}: {e}"))?;
            ll += wi * jet.log_lik;
        }
        Ok(ll)
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == Self::BLOCK_TIME {
            Ok(self.time_linear_constraints.clone())
        } else {
            Ok(None)
        }
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
}

impl CustomFamily for LatentBinaryFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let n = self.event_target.len();
        let p_time = self.x_time_exit.ncols();
        let p_mean = self.x_mean.ncols();

        let mut ll = 0.0;
        let mut grad_time = Array1::<f64>::zeros(p_time);
        let mut hess_time = Array2::<f64>::zeros((p_time, p_time));
        let mut grad_mean = Array1::<f64>::zeros(p_mean);
        let mut hess_mean = Array2::<f64>::zeros((p_mean, p_mean));

        for i in 0..n {
            let wi = self.weights[i];
            if wi <= MIN_WEIGHT {
                continue;
            }
            if !(q_entry[i].is_finite() && q_exit[i].is_finite() && mu[i].is_finite()) {
                return Err(format!(
                    "latent-binary row {i} contains non-finite predictors: q_entry={}, q_exit={}, mu={}",
                    q_entry[i], q_exit[i], mu[i]
                ));
            }
            let row = build_latent_survival_row(
                i,
                self.hazard_loading,
                LatentSurvivalEventType::RightCensored,
                q_entry[i],
                q_exit[i],
                1.0,
                self.unloaded_mass_entry[i],
                self.unloaded_mass_exit[i],
                0.0,
            )?;
            let survival_jet =
                LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], self.latent_sd)
                    .map_err(|e| format!("LatentBinaryFamily row {i}: {e}"))?;
            let binary = binary_from_log_survival(survival_jet.log_lik, self.event_target[i])?;
            ll += wi * binary.log_lik;

            let mean_row = self.x_mean.row_chunk(i..i + 1);
            let mean_vec = mean_row.row(0);
            let mean_grad_scale = wi * binary.grad_scale * survival_jet.score;
            for j in 0..p_mean {
                grad_mean[j] += mean_grad_scale * mean_vec[j];
            }
            let mean_neg_hess = wi
                * (binary.neg_hess_scale * survival_jet.neg_hessian
                    + binary.outer_scale * survival_jet.score * survival_jet.score);
            dense_outer_accumulate(&mut hess_mean, mean_neg_hess, mean_vec);

            let time_jet =
                latent_survival_time_jet(&self.quadctx, &row, 0.0, mu[i], self.latent_sd)?;
            let t_entry = self.x_time_entry.row(i);
            let t_exit = self.x_time_exit.row(i);
            for j in 0..p_time {
                grad_time[j] += wi
                    * binary.grad_scale
                    * (time_jet.grad_entry * t_entry[j] + time_jet.grad_exit * t_exit[j]);
            }
            dense_outer_accumulate(
                &mut hess_time,
                wi * binary.neg_hess_scale * time_jet.neg_hess_entry,
                t_entry,
            );
            dense_outer_accumulate(
                &mut hess_time,
                wi * binary.neg_hess_scale * time_jet.neg_hess_exit,
                t_exit,
            );
            if binary.outer_scale != 0.0 {
                dense_outer_accumulate(
                    &mut hess_time,
                    wi * binary.outer_scale * time_jet.grad_entry * time_jet.grad_entry,
                    t_entry,
                );
                dense_outer_accumulate(
                    &mut hess_time,
                    wi * binary.outer_scale * time_jet.grad_exit * time_jet.grad_exit,
                    t_exit,
                );
                dense_symmetric_cross_accumulate(
                    &mut hess_time,
                    wi * binary.outer_scale * time_jet.grad_entry * time_jet.grad_exit,
                    t_entry,
                    t_exit,
                );
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: SymmetricMatrix::Dense(hess_time),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_mean,
                    hessian: SymmetricMatrix::Dense(hess_mean),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let mut ll = 0.0;
        for i in 0..self.event_target.len() {
            let wi = self.weights[i];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row = build_latent_survival_row(
                i,
                self.hazard_loading,
                LatentSurvivalEventType::RightCensored,
                q_entry[i],
                q_exit[i],
                1.0,
                self.unloaded_mass_entry[i],
                self.unloaded_mass_exit[i],
                0.0,
            )?;
            let survival_jet =
                LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], self.latent_sd)
                    .map_err(|e| format!("LatentBinaryFamily row {i}: {e}"))?;
            ll +=
                wi * binary_from_log_survival(survival_jet.log_lik, self.event_target[i])?.log_lik;
        }
        Ok(ll)
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == Self::BLOCK_TIME {
            Ok(self.time_linear_constraints.clone())
        } else {
            Ok(None)
        }
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
}
