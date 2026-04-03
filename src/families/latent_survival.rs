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
use std::collections::BTreeMap;
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
            "{context} currently requires a fixed hazard-multiplier sigma"
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

const LATENT_SURVIVAL_PRIMARY_Q_ENTRY: usize = 0;
const LATENT_SURVIVAL_PRIMARY_Q_EXIT: usize = 1;
const LATENT_SURVIVAL_PRIMARY_QDOT_EXIT: usize = 2;
const LATENT_SURVIVAL_PRIMARY_MU: usize = 3;
const LATENT_SURVIVAL_PRIMARY_LOG_SIGMA: usize = 4;
const LATENT_SURVIVAL_PRIMARY_DIM: usize = 5;

fn latent_jet_subset_partitions(mask: usize) -> Vec<Vec<usize>> {
    if mask == 0 {
        return vec![Vec::new()];
    }
    let first = mask & mask.wrapping_neg();
    let rest = mask ^ first;
    let mut out = Vec::new();
    let mut subset = rest;
    loop {
        let block = first | subset;
        for mut remainder in latent_jet_subset_partitions(rest ^ subset) {
            remainder.push(block);
            out.push(remainder);
        }
        if subset == 0 {
            break;
        }
        subset = (subset - 1) & rest;
    }
    out
}

#[derive(Clone)]
struct LatentMultiDirJet {
    coeffs: Vec<f64>,
}

impl LatentMultiDirJet {
    fn zero(n_dirs: usize) -> Self {
        Self {
            coeffs: vec![0.0; 1usize << n_dirs],
        }
    }

    fn constant(n_dirs: usize, value: f64) -> Self {
        let mut out = Self::zero(n_dirs);
        out.coeffs[0] = value;
        out
    }

    fn coeff(&self, mask: usize) -> f64 {
        self.coeffs[mask]
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(other.coeffs.iter())
                .map(|(l, r)| l + r)
                .collect(),
        }
    }

    fn scale(&self, scalar: f64) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|v| scalar * v).collect(),
        }
    }

    fn compose_unary(&self, derivs: [f64; 5]) -> Self {
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        out[0] = derivs[0];
        for (mask, value) in out.iter_mut().enumerate().skip(1) {
            let mut total = 0.0;
            for partition in latent_jet_subset_partitions(mask) {
                let order = partition.len();
                if order == 0 || order >= derivs.len() {
                    continue;
                }
                let mut prod = 1.0;
                for block in partition {
                    prod *= self.coeffs[block];
                }
                total += derivs[order] * prod;
            }
            *value = total;
        }
        Self { coeffs: out }
    }
}

#[inline]
fn latent_unary_derivatives_log(x: f64) -> [f64; 5] {
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x3 * x1;
    [x1.ln(), 1.0 / x1, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

#[inline]
fn latent_log1mexp(a: f64) -> f64 {
    assert!(a >= 0.0);
    if a > core::f64::consts::LN_2 {
        (-(-a).exp()).ln_1p()
    } else if a > 0.0 {
        (-(-a).exp_m1()).ln()
    } else {
        f64::NEG_INFINITY
    }
}

fn latent_signed_log_sum_exp(log_mags: &[f64], signs: &[f64]) -> (f64, f64) {
    let mut pos_max = f64::NEG_INFINITY;
    let mut neg_max = f64::NEG_INFINITY;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if signs[idx] > 0.0 {
            pos_max = pos_max.max(lm);
        } else if signs[idx] < 0.0 {
            neg_max = neg_max.max(lm);
        }
    }

    let mut pos_sum = 0.0_f64;
    let mut neg_sum = 0.0_f64;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if !lm.is_finite() {
            continue;
        }
        if signs[idx] > 0.0 {
            pos_sum += (lm - pos_max).exp();
        } else if signs[idx] < 0.0 {
            neg_sum += (lm - neg_max).exp();
        }
    }

    let log_pos = if pos_sum > 0.0 {
        pos_max + pos_sum.ln()
    } else {
        f64::NEG_INFINITY
    };
    let log_neg = if neg_sum > 0.0 {
        neg_max + neg_sum.ln()
    } else {
        f64::NEG_INFINITY
    };

    if log_neg == f64::NEG_INFINITY {
        return (log_pos, 1.0);
    }
    if log_pos == f64::NEG_INFINITY {
        return (log_neg, -1.0);
    }
    if log_pos > log_neg {
        let gap = log_pos - log_neg;
        (log_pos + latent_log1mexp(gap), 1.0)
    } else if log_neg > log_pos {
        let gap = log_neg - log_pos;
        (log_neg + latent_log1mexp(gap), -1.0)
    } else {
        (f64::NEG_INFINITY, 0.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryTerm {
    coeff: f64,
    q_exp: usize,
    qdot_power: usize,
    tau_exp: usize,
    k: usize,
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryDirection {
    dq: f64,
    dqd: f64,
    dmu: f64,
    dtau: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentSurvivalPrimaryDirection {
    dq_entry: f64,
    dq_exit: f64,
    dqdot_exit: f64,
    dmu: f64,
    dlog_sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryState {
    q: f64,
    qdot: f64,
    mu: f64,
    sigma: f64,
    log_sigma_factor: f64,
}

fn latent_kernel_accumulate_term(
    terms: &mut BTreeMap<(usize, usize, usize, usize), f64>,
    term: LatentKernelPrimaryTerm,
    scale: f64,
) {
    if scale == 0.0 || term.coeff == 0.0 {
        return;
    }
    *terms
        .entry((term.q_exp, term.qdot_power, term.tau_exp, term.k))
        .or_insert(0.0) += scale * term.coeff;
}

fn latent_kernel_differentiate_terms(
    terms: &[LatentKernelPrimaryTerm],
    dir: LatentKernelPrimaryDirection,
) -> Vec<LatentKernelPrimaryTerm> {
    let mut out = BTreeMap::<(usize, usize, usize, usize), f64>::new();
    for term in terms {
        if dir.dq != 0.0 {
            if term.q_exp > 0 {
                latent_kernel_accumulate_term(&mut out, *term, dir.dq * term.q_exp as f64);
            }
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dq,
            );
        }
        if dir.dmu != 0.0 {
            if term.k > 0 {
                latent_kernel_accumulate_term(&mut out, *term, dir.dmu * term.k as f64);
            }
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dmu,
            );
        }
        if dir.dtau != 0.0 {
            if term.tau_exp > 0 {
                latent_kernel_accumulate_term(&mut out, *term, dir.dtau * term.tau_exp as f64);
            }
            let kf = term.k as f64;
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    tau_exp: term.tau_exp + 2,
                    ..*term
                },
                dir.dtau * kf * kf,
            );
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    tau_exp: term.tau_exp + 2,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dtau * (2.0 * kf + 1.0),
            );
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 2,
                    tau_exp: term.tau_exp + 2,
                    k: term.k + 2,
                    ..*term
                },
                dir.dtau,
            );
        }
        if dir.dqd != 0.0 && term.qdot_power > 0 {
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    qdot_power: term.qdot_power - 1,
                    ..*term
                },
                dir.dqd * term.qdot_power as f64,
            );
        }
    }
    out.into_iter()
        .filter_map(|((q_exp, qdot_power, tau_exp, k), coeff)| {
            (coeff != 0.0).then_some(LatentKernelPrimaryTerm {
                coeff,
                q_exp,
                qdot_power,
                tau_exp,
                k,
            })
        })
        .collect()
}

fn latent_kernel_term_lists_for_directions(
    base_terms: &[LatentKernelPrimaryTerm],
    directions: &[LatentKernelPrimaryDirection],
) -> Vec<Vec<LatentKernelPrimaryTerm>> {
    fn build_mask(
        mask: usize,
        base_terms: &[LatentKernelPrimaryTerm],
        directions: &[LatentKernelPrimaryDirection],
        cache: &mut [Option<Vec<LatentKernelPrimaryTerm>>],
    ) -> Vec<LatentKernelPrimaryTerm> {
        if let Some(existing) = &cache[mask] {
            return existing.clone();
        }
        let built = if mask == 0 {
            base_terms.to_vec()
        } else {
            let bit = 1usize << mask.trailing_zeros();
            let prev = build_mask(mask ^ bit, base_terms, directions, cache);
            latent_kernel_differentiate_terms(&prev, directions[bit.trailing_zeros() as usize])
        };
        cache[mask] = Some(built.clone());
        built
    }

    let mut cache = vec![None; 1usize << directions.len()];
    (0..cache.len())
        .map(|mask| build_mask(mask, base_terms, directions, &mut cache))
        .collect()
}

fn latent_kernel_sum_log_jet(
    quadctx: &QuadratureContext,
    base_terms: &[LatentKernelPrimaryTerm],
    state: LatentKernelPrimaryState,
    directions: &[LatentKernelPrimaryDirection],
    context: &str,
) -> Result<LatentMultiDirJet, String> {
    let term_lists = latent_kernel_term_lists_for_directions(base_terms, directions);
    let max_k = term_lists
        .iter()
        .flat_map(|terms| terms.iter().map(|term| term.k))
        .max()
        .unwrap_or(0);
    let bundle = log_kernel_bundle(quadctx, state.q.exp(), state.mu, state.sigma, max_k)
        .map_err(|e| format!("{context} kernel evaluation failed: {e}"))?;

    let evaluate_terms = |terms: &[LatentKernelPrimaryTerm]| -> Result<(f64, f64), String> {
        let mut log_mags = Vec::new();
        let mut signs = Vec::new();
        for term in terms {
            if term.coeff == 0.0 {
                continue;
            }
            if term.qdot_power > 0 && !(state.qdot.is_finite() && state.qdot > 0.0) {
                return Err(format!(
                    "{context} requires positive finite qdot for exact-event directional terms, got {}",
                    state.qdot
                ));
            }
            let log_qdot = if term.qdot_power > 0 {
                state.qdot.ln()
            } else {
                0.0
            };
            let log_mag = term.coeff.abs().ln()
                + term.q_exp as f64 * state.q
                + term.tau_exp as f64 * state.log_sigma_factor
                + term.qdot_power as f64 * log_qdot
                + bundle.get(term.k);
            log_mags.push(log_mag);
            signs.push(term.coeff.signum());
        }
        if log_mags.is_empty() {
            return Ok((f64::NEG_INFINITY, 0.0));
        }
        Ok(latent_signed_log_sum_exp(&log_mags, &signs))
    };

    let (base_log_sum, base_sign) = evaluate_terms(&term_lists[0])?;
    if !(base_log_sum.is_finite() && base_sign > 0.0) {
        return Err(format!(
            "{context} produced a non-positive signed kernel sum"
        ));
    }

    let mut normalized = LatentMultiDirJet::constant(directions.len(), 1.0);
    for mask in 1..term_lists.len() {
        let (log_abs, sign) = evaluate_terms(&term_lists[mask])?;
        normalized.coeffs[mask] = if !log_abs.is_finite() || sign == 0.0 {
            0.0
        } else {
            sign * (log_abs - base_log_sum).exp()
        };
    }

    let mut out = normalized.compose_unary(latent_unary_derivatives_log(1.0));
    out.coeffs[0] += base_log_sum;
    Ok(out)
}

fn latent_survival_basis_direction(primary_idx: usize) -> LatentSurvivalPrimaryDirection {
    match primary_idx {
        LATENT_SURVIVAL_PRIMARY_Q_ENTRY => LatentSurvivalPrimaryDirection {
            dq_entry: 1.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_Q_EXIT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 1.0,
            dqdot_exit: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_QDOT_EXIT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 1.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_MU => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dmu: 1.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dmu: 0.0,
            dlog_sigma: 1.0,
        },
        _ => panic!("invalid latent survival primary index {primary_idx}"),
    }
}

fn latent_survival_map_entry_direction(
    direction: LatentSurvivalPrimaryDirection,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_entry,
        dqd: 0.0,
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

fn latent_survival_map_exit_direction(
    direction: LatentSurvivalPrimaryDirection,
    event_type: LatentSurvivalEventType,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_exit,
        dqd: if matches!(event_type, LatentSurvivalEventType::ExactEvent) {
            direction.dqdot_exit
        } else {
            0.0
        },
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

fn latent_survival_row_primary_log_jet(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    mu: f64,
    sigma: f64,
    log_sigma_factor: f64,
    directions: &[LatentSurvivalPrimaryDirection],
) -> Result<LatentMultiDirJet, String> {
    let entry_state = LatentKernelPrimaryState {
        q: q_entry,
        qdot: 1.0,
        mu,
        sigma,
        log_sigma_factor,
    };
    let exit_state = LatentKernelPrimaryState {
        q: q_exit,
        qdot: qdot_exit,
        mu,
        sigma,
        log_sigma_factor,
    };
    let entry_directions = directions
        .iter()
        .copied()
        .map(latent_survival_map_entry_direction)
        .collect::<Vec<_>>();
    let exit_directions = directions
        .iter()
        .copied()
        .map(|dir| latent_survival_map_exit_direction(dir, row.event_type))
        .collect::<Vec<_>>();

    let denominator = latent_kernel_sum_log_jet(
        quadctx,
        &[LatentKernelPrimaryTerm {
            coeff: 1.0,
            q_exp: 0,
            qdot_power: 0,
            tau_exp: 0,
            k: 0,
        }],
        entry_state,
        &entry_directions,
        "latent survival denominator",
    )?;

    let numerator_terms = match row.event_type {
        LatentSurvivalEventType::RightCensored => vec![LatentKernelPrimaryTerm {
            coeff: 1.0,
            q_exp: 0,
            qdot_power: 0,
            tau_exp: 0,
            k: 0,
        }],
        LatentSurvivalEventType::ExactEvent => {
            let mut terms = Vec::new();
            if row.hazard_unloaded > 0.0 {
                terms.push(LatentKernelPrimaryTerm {
                    coeff: row.hazard_unloaded,
                    q_exp: 0,
                    qdot_power: 0,
                    tau_exp: 0,
                    k: 0,
                });
            }
            terms.push(LatentKernelPrimaryTerm {
                coeff: 1.0,
                q_exp: 1,
                qdot_power: 1,
                tau_exp: 0,
                k: 1,
            });
            terms
        }
        LatentSurvivalEventType::IntervalCensored => {
            return Err(
                "latent survival dynamic time derivatives do not implement interval censoring"
                    .to_string(),
            );
        }
    };
    let numerator = latent_kernel_sum_log_jet(
        quadctx,
        &numerator_terms,
        exit_state,
        &exit_directions,
        "latent survival numerator",
    )?;

    let mut total = numerator.add(&denominator.scale(-1.0));
    total.coeffs[0] += -row.mass_unloaded_exit + row.mass_unloaded_entry;
    Ok(total)
}

fn latent_survival_row_primary_gradient_hessian(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    mu: f64,
    sigma: f64,
    include_log_sigma: bool,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    let log_sigma_factor = if sigma > 0.0 { sigma.ln() } else { 0.0 };
    let mut gradient = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
    let mut neg_hessian =
        Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
    let active_primary = if include_log_sigma {
        LATENT_SURVIVAL_PRIMARY_DIM
    } else {
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
    };
    let log_lik = latent_survival_row_primary_log_jet(
        quadctx,
        row,
        q_entry,
        q_exit,
        qdot_exit,
        mu,
        sigma,
        log_sigma_factor,
        &[],
    )?
    .coeff(0);
    for a in 0..active_primary {
        let dir_a = latent_survival_basis_direction(a);
        gradient[a] = latent_survival_row_primary_log_jet(
            quadctx,
            row,
            q_entry,
            q_exit,
            qdot_exit,
            mu,
            sigma,
            log_sigma_factor,
            &[dir_a],
        )?
        .coeff(1);
        for b in a..active_primary {
            let coeff = latent_survival_row_primary_log_jet(
                quadctx,
                row,
                q_entry,
                q_exit,
                qdot_exit,
                mu,
                sigma,
                log_sigma_factor,
                &[dir_a, latent_survival_basis_direction(b)],
            )?
            .coeff(3);
            neg_hessian[[a, b]] = -coeff;
            neg_hessian[[b, a]] = -coeff;
        }
    }
    Ok((log_lik, gradient, neg_hessian))
}

fn latent_survival_row_primary_third_contracted(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    mu: f64,
    sigma: f64,
    direction: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, String> {
    let log_sigma_factor = if sigma > 0.0 { sigma.ln() } else { 0.0 };
    let active_primary = if include_log_sigma {
        LATENT_SURVIVAL_PRIMARY_DIM
    } else {
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
    };
    let dir = LatentSurvivalPrimaryDirection {
        dq_entry: direction[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
        dq_exit: direction[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
        dqdot_exit: direction[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
        dmu: direction[LATENT_SURVIVAL_PRIMARY_MU],
        dlog_sigma: direction[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA],
    };
    let mut out = Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
    for a in 0..active_primary {
        let dir_a = latent_survival_basis_direction(a);
        for b in a..active_primary {
            let coeff = latent_survival_row_primary_log_jet(
                quadctx,
                row,
                q_entry,
                q_exit,
                qdot_exit,
                mu,
                sigma,
                log_sigma_factor,
                &[dir_a, latent_survival_basis_direction(b), dir],
            )?
            .coeff(7);
            out[[a, b]] = -coeff;
            out[[b, a]] = -coeff;
        }
    }
    Ok(out)
}

#[derive(Clone)]
struct LatentSurvivalJointSlices {
    time: std::ops::Range<usize>,
    mean: std::ops::Range<usize>,
    log_sigma: Option<std::ops::Range<usize>>,
    total: usize,
}

impl LatentSurvivalFamily {
    fn joint_slices(&self) -> LatentSurvivalJointSlices {
        let p_time = self.x_time_exit.ncols();
        let p_mean = self.x_mean.ncols();
        let time = 0..p_time;
        let mean = p_time..p_time + p_mean;
        let log_sigma = self
            .latent_sd_fixed
            .is_none()
            .then_some((p_time + p_mean)..(p_time + p_mean + 1));
        LatentSurvivalJointSlices {
            total: log_sigma
                .as_ref()
                .map_or(p_time + p_mean, |range| range.end),
            time,
            mean,
            log_sigma,
        }
    }

    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        out[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] = self.x_time_entry.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_Q_EXIT] = self.x_time_exit.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT] = self.x_time_derivative_exit.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_MU] = self
            .x_mean
            .dot_row_view(row, d_beta_flat.slice(s![slices.mean.clone()]));
        if let Some(range) = &slices.log_sigma {
            out[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA] = d_beta_flat[range.start];
        }
        out
    }

    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let time_weights = [
            primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
            ]],
            primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
            ]],
            primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
            ]],
        ];
        let time_cross_weights = [
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                &self.x_time_entry,
                &self.x_time_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                &self.x_time_entry,
                &self.x_time_derivative_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                &self.x_time_exit,
                &self.x_time_derivative_exit,
            ),
        ];
        {
            let time_target = &mut target.slice_mut(s![slices.time.clone(), slices.time.clone()]);
            dense_outer_accumulate(time_target, time_weights[0], self.x_time_entry.row(row));
            dense_outer_accumulate(time_target, time_weights[1], self.x_time_exit.row(row));
            dense_outer_accumulate(
                time_target,
                time_weights[2],
                self.x_time_derivative_exit.row(row),
            );
            for (a, b, lhs, rhs) in time_cross_weights {
                let weight = primary_hessian[[a, b]];
                if weight == 0.0 {
                    continue;
                }
                dense_symmetric_cross_accumulate(time_target, weight, lhs.row(row), rhs.row(row));
            }
        }

        let mean_weight = primary_hessian[[LATENT_SURVIVAL_PRIMARY_MU, LATENT_SURVIVAL_PRIMARY_MU]];
        self.x_mean
            .syr_row_into_view(
                row,
                mean_weight,
                target.slice_mut(s![slices.mean.clone(), slices.mean.clone()]),
            )
            .expect("latent survival mean pullback dimension mismatch");

        let mean_row = self.x_mean.row_chunk(row..row + 1);
        let mean_vec = mean_row.row(0);
        let time_mean_weights = [
            (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
            (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
            (
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                self.x_time_derivative_exit.row(row),
            ),
        ];
        for (primary_idx, time_vec) in time_mean_weights {
            let weight = primary_hessian[[primary_idx, LATENT_SURVIVAL_PRIMARY_MU]];
            if weight == 0.0 {
                continue;
            }
            for i in 0..time_vec.len() {
                let xi = time_vec[i];
                if xi == 0.0 {
                    continue;
                }
                for j in 0..mean_vec.len() {
                    let xj = mean_vec[j];
                    if xj == 0.0 {
                        continue;
                    }
                    target[[slices.time.start + i, slices.mean.start + j]] += weight * xi * xj;
                    target[[slices.mean.start + j, slices.time.start + i]] += weight * xj * xi;
                }
            }
        }

        if let Some(log_sigma) = &slices.log_sigma {
            let sigma_idx = log_sigma.start;
            target[[sigma_idx, sigma_idx]] += primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            ]];

            for (primary_idx, time_vec) in [
                (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
                (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
                (
                    LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                    self.x_time_derivative_exit.row(row),
                ),
            ] {
                let weight = primary_hessian[[primary_idx, LATENT_SURVIVAL_PRIMARY_LOG_SIGMA]];
                if weight == 0.0 {
                    continue;
                }
                for i in 0..time_vec.len() {
                    let xi = time_vec[i];
                    if xi == 0.0 {
                        continue;
                    }
                    target[[slices.time.start + i, sigma_idx]] += weight * xi;
                    target[[sigma_idx, slices.time.start + i]] += weight * xi;
                }
            }

            let mean_sigma_weight = primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_MU,
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            ]];
            if mean_sigma_weight != 0.0 {
                for j in 0..mean_vec.len() {
                    let xj = mean_vec[j];
                    if xj == 0.0 {
                        continue;
                    }
                    target[[slices.mean.start + j, sigma_idx]] += mean_sigma_weight * xj;
                    target[[sigma_idx, slices.mean.start + j]] += mean_sigma_weight * xj;
                }
            }
        }
    }

    fn joint_outer_hyper_surrogate_hessian_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Array2<f64>, String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let event_type = if self.event_target[row_idx] >= 1 {
                LatentSurvivalEventType::ExactEvent
            } else {
                LatentSurvivalEventType::RightCensored
            };
            let row = build_latent_survival_row(
                row_idx,
                self.hazard_loading,
                event_type,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                self.unloaded_mass_entry[row_idx],
                self.unloaded_mass_exit[row_idx],
                self.unloaded_hazard_exit[row_idx],
            )?;
            let (_, _, primary_hessian) = latent_survival_row_primary_gradient_hessian(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                mu[row_idx],
                sigma,
                include_log_sigma,
            )?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &(wi * primary_hessian));
        }
        Ok(out)
    }

    fn joint_outer_hyper_surrogate_dh_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "latent survival joint dH direction length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let event_type = if self.event_target[row_idx] >= 1 {
                LatentSurvivalEventType::ExactEvent
            } else {
                LatentSurvivalEventType::RightCensored
            };
            let row = build_latent_survival_row(
                row_idx,
                self.hazard_loading,
                event_type,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                self.unloaded_mass_entry[row_idx],
                self.unloaded_mass_exit[row_idx],
                self.unloaded_hazard_exit[row_idx],
            )?;
            let direction = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
            let third = latent_survival_row_primary_third_contracted(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                mu[row_idx],
                sigma,
                &direction,
                include_log_sigma,
            )?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &(wi * third));
        }
        Ok(out)
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

fn dense_outer_accumulate<S>(
    target: &mut ndarray::ArrayBase<S, ndarray::Ix2>,
    weight: f64,
    x: ArrayView1<'_, f64>,
) where
    S: ndarray::DataMut<Elem = f64>,
{
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

fn dense_symmetric_cross_accumulate<S>(
    target: &mut ndarray::ArrayBase<S, ndarray::Ix2>,
    weight: f64,
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) where
    S: ndarray::DataMut<Elem = f64>,
{
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

    fn joint_outer_hyper_surrogate_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_outer_hyper_surrogate_hessian_dense(block_states)
            .map(Some)
    }

    fn joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_outer_hyper_surrogate_dh_dense(block_states, d_beta_flat)
            .map(Some)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::custom_family::BlockWorkingSet;
    use crate::matrix::DenseDesignMatrix;
    use ndarray::array;

    #[test]
    fn latent_survival_learnable_sigma_block_matches_family_fd() {
        let sigma = 0.35;
        let log_sigma = sigma.ln();
        let h = 1e-5;

        let learnable_family = LatentSurvivalFamily {
            event_target: array![1u8],
            weights: array![1.0],
            latent_sd_fixed: None,
            hazard_loading: HazardLoading::Full,
            unloaded_mass_entry: array![0.0],
            unloaded_mass_exit: array![0.0],
            unloaded_hazard_exit: array![0.0],
            x_time_entry: array![[1.0]],
            x_time_exit: array![[1.0]],
            x_time_derivative_exit: array![[1.0]],
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0]])),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        };
        let fixed_family = LatentSurvivalFamily {
            latent_sd_fixed: Some(sigma),
            ..learnable_family.clone()
        };

        let time_eta = array![0.2, 0.6, 0.8];
        let mean_eta = array![0.15];
        let fixed_states = vec![
            ParameterBlockState {
                beta: time_eta.clone(),
                eta: time_eta.clone(),
            },
            ParameterBlockState {
                beta: mean_eta.clone(),
                eta: mean_eta.clone(),
            },
        ];
        let states = vec![
            ParameterBlockState {
                beta: time_eta.clone(),
                eta: time_eta,
            },
            ParameterBlockState {
                beta: mean_eta.clone(),
                eta: mean_eta,
            },
            ParameterBlockState {
                beta: array![log_sigma],
                eta: array![log_sigma],
            },
        ];

        let eval = learnable_family
            .evaluate(&states)
            .expect("learnable latent survival evaluation");
        let fixed_eval = fixed_family
            .evaluate(&fixed_states)
            .expect("fixed latent survival evaluation");
        assert!(
            (eval.log_likelihood - fixed_eval.log_likelihood).abs() < 1e-12,
            "learnable/fixed ll mismatch: {} vs {}",
            eval.log_likelihood,
            fixed_eval.log_likelihood
        );
        assert_eq!(eval.blockworking_sets.len(), 3);
        assert_eq!(fixed_eval.blockworking_sets.len(), 2);

        let (grad, neg_hess) = match &eval.blockworking_sets[LatentSurvivalFamily::BLOCK_LOG_SIGMA]
        {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                let neg_hess = match hessian {
                    SymmetricMatrix::Dense(mat) => mat[[0, 0]],
                    _ => panic!("log_sigma block should use a dense exact-Newton Hessian"),
                };
                (gradient[0], neg_hess)
            }
            _ => panic!("log_sigma block should use ExactNewton"),
        };
        assert!(grad.is_finite());
        assert!(neg_hess.is_finite());

        let mut states_plus = states.clone();
        states_plus[LatentSurvivalFamily::BLOCK_LOG_SIGMA].beta[0] += h;
        states_plus[LatentSurvivalFamily::BLOCK_LOG_SIGMA].eta[0] += h;
        let ll_plus = learnable_family
            .log_likelihood_only(&states_plus)
            .expect("ll plus");

        let ll_0 = learnable_family
            .log_likelihood_only(&states)
            .expect("ll base");

        let mut states_minus = states.clone();
        states_minus[LatentSurvivalFamily::BLOCK_LOG_SIGMA].beta[0] -= h;
        states_minus[LatentSurvivalFamily::BLOCK_LOG_SIGMA].eta[0] -= h;
        let ll_minus = learnable_family
            .log_likelihood_only(&states_minus)
            .expect("ll minus");

        let fd_grad = (ll_plus - ll_minus) / (2.0 * h);
        let fd_neg_hess = -(ll_plus - 2.0 * ll_0 + ll_minus) / (h * h);

        assert!(
            (grad - fd_grad).abs() / fd_grad.abs().max(1e-15) < 5e-3,
            "family log_sigma grad={}, fd={fd_grad}",
            grad
        );
        assert!(
            (neg_hess - fd_neg_hess).abs() / fd_neg_hess.abs().max(1e-12) < 2e-2,
            "family log_sigma neg_hess={}, fd={fd_neg_hess}",
            neg_hess
        );
    }
}
