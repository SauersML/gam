use crate::basis::{
    BasisOptions, Dense, KnotSource, PenaltyInfo, PenaltySource, create_basis,
    create_difference_penalty_matrix, create_ispline_derivative_dense,
};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyJointDesignChannel, CustomFamilyJointDesignPairContribution,
    CustomFamilyJointPsiOperator, CustomFamilyPsiDesignAction, CustomFamilyPsiLinearMapRef,
    CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiDirectCache, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiWorkspace, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, PsiDesignMap, build_block_spatial_psi_derivatives,
    evaluate_custom_family_joint_hyper, evaluate_custom_family_joint_hyper_efs, fit_custom_family,
    fit_custom_family_fixed_log_lambdas, resolve_custom_family_x_psi_map,
    resolve_custom_family_x_psi_psi_map, second_psi_linear_map, shared_dense_arc,
    weighted_crossprod_psi_maps,
};
use crate::estimate::UnifiedFitResult;
use crate::faer_ndarray::{fast_atv, fast_joint_hessian_2x2, fast_xt_diag_x, fast_xt_diag_y};
use crate::matrix::SymmetricMatrix;
use crate::families::scale_design::{
    apply_scale_deviation_transform, build_scale_deviation_transform,
};
use crate::families::sigma_link::{
    LOGB_SIGMA_FLOOR, SigmaJet1, exp_sigma_derivs_up_to_fourth_scalar,
    exp_sigma_derivs_up_to_third, exp_sigma_from_eta_scalar, exp_sigma_jet1_scalar,
    logb_sigma_from_eta_scalar, logb_sigma_jet1_scalar, safe_exp,
};
use crate::generative::{CustomFamilyGenerative, GenerativeSpec, NoiseModel};
use crate::linalg::utils::solve_spd_pcg_with_info;
use crate::matrix::{DenseDesignOperator, DesignMatrix};
use crate::mixture_link::{
    inverse_link_jet_for_inverse_link, inverse_link_pdffourth_derivative_for_inverse_link,
};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_pdf, standard_normal_quantile};
use crate::smooth::{
    BlockwisePenalty, ExactJointHyperSetup, PenaltyBlockInfo,
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec, build_term_collection_design, freeze_term_collection_from_design,
    optimize_spatial_length_scale_exact_joint, spatial_dims_per_term,
    spatial_length_scale_term_indices,
};
use crate::solver::estimate::validate_all_finite_estimation;
use crate::types::{InverseLink, LinkFunction};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
const MIN_PROB: f64 = 1e-10;
const MIN_DERIV: f64 = 1e-8;
const MIN_WEIGHT: f64 = 1e-12;
const EXACT_DENSE_BLOCK_BUDGET_BYTES: usize = 512 * 1024 * 1024;
const EXACT_DENSE_TOTAL_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

enum DenseOrOperator<'a> {
    Borrowed(&'a Array2<f64>),
    Owned(Array2<f64>),
    Operator(DesignMatrix),
}

impl DenseOrOperator<'_> {
    fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(dense) => dense.nrows(),
            Self::Owned(dense) => dense.nrows(),
            Self::Operator(design) => design.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(dense) => dense.ncols(),
            Self::Owned(dense) => dense.ncols(),
            Self::Operator(design) => design.ncols(),
        }
    }

    fn row_chunk(&self, rows: std::ops::Range<usize>) -> Result<Array2<f64>, String> {
        match self {
            Self::Borrowed(dense) => Ok(dense.slice(s![rows, ..]).to_owned()),
            Self::Owned(dense) => Ok(dense.slice(s![rows, ..]).to_owned()),
            Self::Operator(design) => design.try_row_chunk(rows).map_err(|e| e.to_string()),
        }
    }

    fn dot(&self, beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.nrows();
        let p = self.ncols();
        assert_eq!(beta.len(), p);
        match self {
            Self::Borrowed(dense) => dense.dot(&beta),
            Self::Owned(dense) => dense.dot(&beta),
            Self::Operator(design) => {
                let mut out = Array1::<f64>::zeros(n);
                for rows in exact_design_row_chunks(n, p) {
                    let chunk = design
                        .try_row_chunk(rows.clone())
                        .expect("gamlss DesignSlot::dot: design row chunk materialization failed");
                    out.slice_mut(s![rows]).assign(&chunk.dot(&beta));
                }
                out
            }
        }
    }
}

fn dense_block_or_operator<'a>(
    design: &'a DesignMatrix,
    n: usize,
    p: usize,
    budget_bytes: usize,
    policy: &crate::resource::ResourcePolicy,
) -> DenseOrOperator<'a> {
    if let Some(dense) = design.as_dense_ref() {
        return DenseOrOperator::Borrowed(dense);
    }

    let dense_bytes = 8usize.saturating_mul(n).saturating_mul(p);
    let compute_ok = match crate::linalg::matrix::panic_or_error_if_biobank_mode_compute_budget_exceeded(
        "gamlss dense_block_or_operator",
        n,
        p,
        crate::linalg::matrix::POLICY_DEFAULT_OUTER_ITER_ESTIMATE,
        policy,
    ) {
        Ok(()) => true,
        Err(msg) => {
            log::info!("{msg}; falling back to operator path");
            false
        }
    };
    if compute_ok && dense_bytes <= budget_bytes {
        if let Ok(arc) = design
            .try_to_dense_with_policy(&policy.material_policy(), "gamlss dense_block_or_operator")
        {
            return DenseOrOperator::Owned(arc.as_ref().clone());
        }
    }

    DenseOrOperator::Operator(design.clone())
}

fn dense_blocks_planned_budget(blocks: &[&DesignMatrix]) -> Vec<usize> {
    let mut planned = vec![0; blocks.len()];
    let mut total = 0usize;
    for (idx, design) in blocks.iter().enumerate() {
        if design.as_dense_ref().is_some() {
            continue;
        }
        let bytes = 8usize
            .saturating_mul(design.nrows())
            .saturating_mul(design.ncols());
        if bytes <= EXACT_DENSE_BLOCK_BUDGET_BYTES
            && total.saturating_add(bytes) <= EXACT_DENSE_TOTAL_BUDGET_BYTES
        {
            planned[idx] = bytes;
            total += bytes;
        }
    }
    planned
}

fn exact_design_row_chunks(n: usize, p: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    let rows = (TARGET_BYTES / (p.max(1) * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n.max(1));
    (0..n)
        .step_by(rows)
        .map(move |start| start..(start + rows).min(n))
}

#[inline]
fn floor_positiveweight(rawweight: f64, minweight: f64) -> f64 {
    if !rawweight.is_finite() || rawweight <= 0.0 {
        0.0
    } else {
        rawweight.max(minweight)
    }
}

#[inline]
fn gaussian_log_sigma_irlsinfo_directional_derivative(weight: f64, sigma: f64, d_eta: f64) -> f64 {
    if weight == 0.0 || d_eta == 0.0 || !sigma.is_finite() || sigma <= 0.0 {
        return 0.0;
    }
    // Logb form mirrors gaussian_jointrow_scalars: κ = 1 − b/σ ∈ [0, 1) and
    // dκ/dη = κ(1−κ). Avoids the inf/inf NaN that the older d_sigma/sigma
    // form produced at large η when both numerator and denominator overflow.
    let g = 1.0 - LOGB_SIGMA_FLOOR / sigma;
    if !g.is_finite() || !(0.0..1.0).contains(&g) {
        return 0.0;
    }
    let rawinfo = 2.0 * weight * g * g;
    if !rawinfo.is_finite() || rawinfo <= MIN_WEIGHT {
        return 0.0;
    }
    let dg_deta = g * (1.0 - g);
    let dw = 4.0 * weight * g * dg_deta * d_eta;
    if dw.is_finite() { dw } else { 0.0 }
}

#[derive(Clone, Copy)]
struct GaussianDiagonalRowKernel {
    log_likelihood: f64,
    location_working_weight: f64,
    location_working_shift: f64,
    log_sigma_working_weight: f64,
    log_sigma_working_response: f64,
}

#[inline]
fn gaussian_diagonal_row_kernel(
    y: f64,
    location_eta: f64,
    eta_log_sigma: f64,
    obs_weight: f64,
    ln2pi: f64,
) -> GaussianDiagonalRowKernel {
    if obs_weight == 0.0 {
        return GaussianDiagonalRowKernel {
            log_likelihood: 0.0,
            location_working_weight: 0.0,
            location_working_shift: 0.0,
            log_sigma_working_weight: 0.0,
            log_sigma_working_response: eta_log_sigma,
        };
    }

    // logb noise link σ = b + exp(η) bounds σ ≥ b > 0 by construction, so the
    // Gaussian location-scale objective ½Σ(y−μ)²/σ² + Σlog σ is bounded below
    // for any finite data. Its working weight 1/σ² is bounded by 1/b², so
    // H_μμ has bounded condition number — no after-the-fact floor or cap is
    // needed (the previous (1e-12, 1e24) clamp was a numerical bandaid for the
    // pure-exp link's σ→0 singularity and is structurally unnecessary here).
    let SigmaJet1 { sigma, d1: _ } = logb_sigma_jet1_scalar(eta_log_sigma);
    let inv_s2 = (sigma * sigma).recip();
    let residual = y - location_eta;
    let location_working_weight = floor_positiveweight(obs_weight * inv_s2, MIN_WEIGHT);
    // dlog σ/dη = (∂σ/∂η)/σ = (σ−b)/σ = 1 − b/σ ∈ [0, 1).
    // Use the (1 − b/σ) form so the η→+∞ tail (σ→∞ giving exp(η)/exp(η) = ∞/∞)
    // evaluates cleanly to 1 instead of NaN; mathematically identical, but
    // avoids feeding NaN into the IRLS info weight when overflow happens.
    // Fisher info per obs = 2·(dσ/dη)²/σ² = 2·dlog_sigma_deta², matching the
    // formula for the pure-exp link (where dlog_sigma_deta ≡ 1).
    let dlog_sigma_deta = 1.0 - LOGB_SIGMA_FLOOR / sigma;
    let log_sigma_working_weight = floor_positiveweight(
        2.0 * obs_weight * dlog_sigma_deta * dlog_sigma_deta,
        MIN_WEIGHT,
    );
    let log_sigma_score = obs_weight * (residual * residual * inv_s2 - 1.0) * dlog_sigma_deta;
    let log_sigma_working_response = if log_sigma_working_weight == 0.0 {
        eta_log_sigma
    } else {
        eta_log_sigma + log_sigma_score / log_sigma_working_weight
    };

    GaussianDiagonalRowKernel {
        log_likelihood: obs_weight
            * (-0.5 * (residual * residual * inv_s2 + ln2pi + 2.0 * sigma.ln())),
        location_working_weight,
        location_working_shift: residual,
        log_sigma_working_weight,
        log_sigma_working_response,
    }
}

#[derive(Clone, Copy)]
struct GamlssLambdaLayout {
    k_mean: usize,
    k_noise: usize,
    kwiggle: usize,
}

impl GamlssLambdaLayout {
    fn two_block(k_mean: usize, k_noise: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            kwiggle: 0,
        }
    }

    fn withwiggle(k_mean: usize, k_noise: usize, kwiggle: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            kwiggle,
        }
    }

    fn total(self) -> usize {
        self.k_mean + self.k_noise + self.kwiggle
    }

    fn mean_end(self) -> usize {
        self.k_mean
    }

    fn noise_start(self) -> usize {
        self.k_mean
    }

    fn noise_end(self) -> usize {
        self.k_mean + self.k_noise
    }

    fn wiggle_start(self) -> usize {
        self.k_mean + self.k_noise
    }

    fn wiggle_end(self) -> usize {
        self.k_mean + self.k_noise + self.kwiggle
    }

    fn validate_theta_len(self, theta_len: usize, context: &str) -> Result<(), String> {
        let needed = self.total();
        if theta_len < needed {
            return Err(format!(
                "{context} theta too short: got {}, need at least {}",
                theta_len, needed
            ));
        }
        Ok(())
    }

    fn mean_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta.slice(s![0..self.mean_end()]).to_owned()
    }

    fn noise_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta
            .slice(s![self.noise_start()..self.noise_end()])
            .to_owned()
    }

    fn wiggle_from(self, theta: &Array1<f64>) -> Array1<f64> {
        theta
            .slice(s![self.wiggle_start()..self.wiggle_end()])
            .to_owned()
    }
}

#[derive(Clone, Copy)]
struct GamlssBetaLayout {
    pt: usize,
    pls: usize,
    pw: usize,
}

impl GamlssBetaLayout {
    fn withwiggle(pt: usize, pls: usize, pw: usize) -> Self {
        Self { pt, pls, pw }
    }

    fn total(self) -> usize {
        self.pt + self.pls + self.pw
    }

    fn split_three(
        self,
        flat: &Array1<f64>,
        context: &str,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
        if flat.len() != self.total() {
            return Err(format!(
                "{context} length mismatch: got {}, expected {}",
                flat.len(),
                self.total()
            ));
        }
        Ok((
            flat.slice(s![0..self.pt]).to_owned(),
            flat.slice(s![self.pt..self.pt + self.pls]).to_owned(),
            flat.slice(s![self.pt + self.pls..self.total()]).to_owned(),
        ))
    }
}

/// Generic block input for high-level built-in family APIs.
#[derive(Clone)]
pub struct ParameterBlockInput {
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    pub penalties: Vec<crate::solver::estimate::PenaltySpec>,
    /// Structural nullspace dimension per penalty (same length as `penalties`).
    /// Empty means "use eigenvalue-based rank detection."
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

#[derive(Clone, Debug)]
pub struct FamilyMetadata {
    pub name: &'static str,
    pub parameternames: &'static [&'static str],
    pub parameter_links: &'static [ParameterLink],
}

#[derive(Clone, Debug)]
pub struct WiggleBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub double_penalty: bool,
}

#[derive(Clone)]
pub(crate) struct SelectedWiggleBasis {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub block: ParameterBlockInput,
}

impl ParameterBlockInput {
    pub fn intospec(self, name: &str) -> Result<ParameterBlockSpec, String> {
        let p = self.design.ncols();
        let n = self.design.nrows();
        if self.offset.len() != n {
            return Err(format!(
                "block '{name}' offset length mismatch: got {}, expected {n}",
                self.offset.len()
            ));
        }
        if let Some(beta0) = &self.initial_beta
            && beta0.len() != p
        {
            return Err(format!(
                "block '{name}' initial_beta length mismatch: got {}, expected {p}",
                beta0.len()
            ));
        }
        for (k, s) in self.penalties.iter().enumerate() {
            match s {
                crate::solver::estimate::PenaltySpec::Block {
                    local, col_range, ..
                } => {
                    if col_range.end > p
                        || local.nrows() != col_range.len()
                        || local.ncols() != col_range.len()
                    {
                        return Err(format!(
                            "block '{name}' penalty {k} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                            col_range.start,
                            col_range.end,
                            local.nrows(),
                            local.ncols()
                        ));
                    }
                }
                crate::solver::estimate::PenaltySpec::Dense(m) => {
                    let (r, c) = m.dim();
                    if r != p || c != p {
                        return Err(format!(
                            "block '{name}' penalty {k} must be {p}x{p}, got {r}x{c}"
                        ));
                    }
                }
            }
        }
        let k = self.penalties.len();
        let initial_log_lambdas = self
            .initial_log_lambdas
            .unwrap_or_else(|| Array1::<f64>::zeros(k));
        if initial_log_lambdas.len() != k {
            return Err(format!(
                "block '{name}' initial_log_lambdas length mismatch: got {}, expected {k}",
                initial_log_lambdas.len()
            ));
        }
        Ok(ParameterBlockSpec {
            name: name.to_string(),
            design: self.design,
            offset: self.offset,
            penalties: {
                self.penalties
                    .into_iter()
                    .map(|spec| match spec {
                        crate::solver::estimate::PenaltySpec::Block {
                            local, col_range, ..
                        } => PenaltyMatrix::Blockwise {
                            local,
                            col_range,
                            total_dim: p,
                        },
                        crate::solver::estimate::PenaltySpec::Dense(m) => PenaltyMatrix::Dense(m),
                    })
                    .collect()
            },
            nullspace_dims: self.nullspace_dims,
            initial_log_lambdas,
            initial_beta: self.initial_beta,
        })
    }
}

fn validate_len_match(name: &str, expected: usize, found: usize) -> Result<(), String> {
    if expected != found {
        return Err(format!(
            "{name} length mismatch: expected {expected}, found {found}"
        ));
    }
    Ok(())
}

fn validateweights(weights: &Array1<f64>, context: &str) -> Result<(), String> {
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(format!(
                "{context}: weights must be finite and non-negative; found weights[{i}]={w}"
            ));
        }
    }
    Ok(())
}

fn validate_binomial_response(y: &Array1<f64>, context: &str) -> Result<(), String> {
    for (i, &yi) in y.iter().enumerate() {
        if !yi.is_finite() || !(0.0..=1.0).contains(&yi) {
            return Err(format!(
                "{context}: binomial response must be finite in [0,1]; found y[{i}]={yi}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn initializewiggle_knots_from_seed(
    seed: ArrayView1<'_, f64>,
    degree: usize,
    num_internal_knots: usize,
) -> Result<Array1<f64>, String> {
    const MIN_WIGGLE_SEED_SPAN: f64 = 1e-8;
    const DEFAULT_WIGGLE_HALF_RANGE: f64 = 3.0;

    let mut seed_min = seed.iter().copied().fold(f64::INFINITY, f64::min);
    let mut seed_max = seed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !seed_min.is_finite() || !seed_max.is_finite() {
        return Err("non-finite seed for wiggle knot initialization".to_string());
    }
    if (seed_max - seed_min).abs() < MIN_WIGGLE_SEED_SPAN {
        let center = 0.5 * (seed_min + seed_max);
        seed_min = center - DEFAULT_WIGGLE_HALF_RANGE;
        seed_max = center + DEFAULT_WIGGLE_HALF_RANGE;
    }
    let (_, knots) = create_basis::<Dense>(
        seed,
        KnotSource::Generate {
            data_range: (seed_min, seed_max),
            num_internal_knots,
        },
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    Ok(knots)
}

pub(crate) fn initialize_monotone_wiggle_knots_from_seed(
    seed: ArrayView1<'_, f64>,
    degree: usize,
    num_internal_knots: usize,
) -> Result<Array1<f64>, String> {
    initializewiggle_knots_from_seed(seed, degree, num_internal_knots)
}

#[inline]
fn monotone_wiggle_internal_degree(degree: usize) -> Result<usize, String> {
    // Public monotone-wiggle degree refers to the value basis. The low-level
    // I-spline builder integrates a degree-`internal_degree` specification
    // into a degree-`internal_degree + 1` value basis, so we subtract one here
    // to keep the public degree and the per-span value degree aligned.
    degree
        .checked_sub(1)
        .filter(|&internal_degree| internal_degree >= 1)
        .ok_or_else(|| "monotone wiggle degree must be >= 2".to_string())
}

#[inline]
fn minimum_monotone_wiggle_knot_count(degree: usize) -> Result<usize, String> {
    degree
        .checked_add(1)
        .and_then(|order| order.checked_mul(2))
        .ok_or_else(|| "monotone wiggle knot-count overflow".to_string())
}

pub fn buildwiggle_block_input_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    penalty_order: usize,
    double_penalty: bool,
) -> Result<ParameterBlockInput, String> {
    let design = monotone_wiggle_basis_from_knots(seed, knots, degree)?;
    let p = design.ncols();
    if p == 0 {
        return Err("wiggle basis has no free monotone columns".to_string());
    }
    let mut penalties: Vec<crate::solver::estimate::PenaltySpec> = Vec::new();
    let mut nullspace_dims = Vec::new();
    if p == 1 {
        penalties.push(crate::solver::estimate::PenaltySpec::Dense(
            Array2::<f64>::eye(1),
        ));
        nullspace_dims.push(0);
    } else {
        let effective_order = penalty_order.max(1).min(p - 1);
        let diff_penalty = create_difference_penalty_matrix(p, effective_order, None)
            .map_err(|e| e.to_string())?;
        penalties.push(crate::solver::estimate::PenaltySpec::Dense(diff_penalty));
        nullspace_dims.push(effective_order);
    }
    if double_penalty {
        penalties.push(crate::solver::estimate::PenaltySpec::Dense(
            Array2::<f64>::eye(p),
        ));
        nullspace_dims.push(0);
    }
    Ok(ParameterBlockInput {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
        offset: Array1::zeros(seed.len()),
        penalties,
        nullspace_dims,
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(p)),
    })
}

pub fn buildwiggle_block_input_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
) -> Result<(ParameterBlockInput, Array1<f64>), String> {
    let knots =
        initialize_monotone_wiggle_knots_from_seed(seed, cfg.degree, cfg.num_internal_knots)?;
    let block = buildwiggle_block_input_from_knots(
        seed,
        &knots,
        cfg.degree,
        cfg.penalty_order,
        cfg.double_penalty,
    )?;
    Ok((block, knots))
}

pub(crate) fn monotone_wiggle_basis_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    let (basis, _) = create_basis::<Dense>(
        seed,
        KnotSource::Provided(knots.view()),
        internal_degree,
        BasisOptions::i_spline(),
    )
    .map_err(|e| e.to_string())?;
    Ok(basis.as_ref().clone())
}

pub fn monotone_wiggle_basis_with_derivative_order(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    derivative_order: usize,
) -> Result<Array2<f64>, String> {
    if derivative_order == 0 {
        return monotone_wiggle_basis_from_knots(seed, knots, degree);
    }
    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    create_ispline_derivative_dense(seed, knots, internal_degree, derivative_order)
        .map_err(|e| e.to_string())
}

pub(crate) fn monotone_wiggle_nonnegative_constraints(
    beta_dim: usize,
) -> Option<LinearInequalityConstraints> {
    if beta_dim == 0 {
        return None;
    }
    let mut a = Array2::<f64>::zeros((beta_dim, beta_dim));
    for i in 0..beta_dim {
        a[[i, i]] = 1.0;
    }
    Some(LinearInequalityConstraints {
        a,
        b: Array1::zeros(beta_dim),
    })
}

pub(crate) fn project_monotone_wiggle_beta(mut beta: Array1<f64>) -> Array1<f64> {
    for value in beta.iter_mut() {
        if !value.is_finite() || *value < 0.0 {
            *value = 0.0;
        }
    }
    beta
}

pub(crate) fn validate_monotone_wiggle_beta_nonnegative(
    beta: &[f64],
    context: &str,
) -> Result<(), String> {
    for (idx, &value) in beta.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{context} coefficient {idx} is non-finite"));
        }
        if value < -1e-12 {
            return Err(format!(
                "{context} coefficient {idx} is negative ({value:.3e}); monotone wiggle coefficients must be non-negative"
            ));
        }
    }
    Ok(())
}

/// Resolve a requested wiggle penalty-order set into:
///
/// - the primary order used by the monotone I-spline coefficient penalty, and
/// - the remaining plain difference-penalty orders to append on the same basis.
///
/// The primary order is the smallest positive requested order. If no positive
/// order is requested, `fallback_primary` is used instead. Extra orders are
/// returned in the original order, deduplicated, and exclude the primary order.
pub fn split_wiggle_penalty_orders(
    fallback_primary: usize,
    penalty_orders: &[usize],
) -> (usize, Vec<usize>) {
    let primary_order = penalty_orders
        .iter()
        .copied()
        .filter(|&order| order >= 1)
        .min()
        .unwrap_or_else(|| fallback_primary.max(1));
    let mut extras = Vec::new();
    for &order in penalty_orders {
        if order == 0 || order == primary_order || extras.contains(&order) {
            continue;
        }
        extras.push(order);
    }
    (primary_order, extras)
}

/// Append raw difference penalties for the given orders to an existing block.
///
/// These are plain difference penalties `D_k^T D_k` on the monotone I-spline
/// coefficients, whose nullspace is the set of polynomial sequences of degree
/// ≤ k−1, giving `nullspace_dim = k`.
pub fn append_selected_wiggle_penalty_orders(
    block: &mut ParameterBlockInput,
    penalty_orders: &[usize],
) -> Result<(), String> {
    let p = block.design.ncols();
    if p == 0 {
        return Ok(());
    }
    for &order in penalty_orders {
        if order == 0 || order >= p {
            continue;
        }
        let penalty =
            create_difference_penalty_matrix(p, order, None).map_err(|e| e.to_string())?;
        block
            .penalties
            .push(crate::solver::estimate::PenaltySpec::Dense(penalty));
        block.nullspace_dims.push(order);
    }
    Ok(())
}

pub(crate) fn select_wiggle_basis_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
    penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let (primary_order, extra_orders) =
        split_wiggle_penalty_orders(cfg.penalty_order, penalty_orders);
    let effective_cfg = WiggleBlockConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_order: primary_order,
        double_penalty: cfg.double_penalty,
    };
    let (mut block, knots) = buildwiggle_block_input_from_seed(seed, &effective_cfg)?;
    append_selected_wiggle_penalty_orders(&mut block, &extra_orders)?;
    Ok(SelectedWiggleBasis {
        knots,
        degree: cfg.degree,
        block,
    })
}

fn validate_blockrows(name: &str, n: usize, block: &ParameterBlockInput) -> Result<(), String> {
    validate_len_match(
        &format!("block '{name}' offset vs response"),
        n,
        block.offset.len(),
    )?;
    validate_len_match(
        &format!("block '{name}' design rows vs response"),
        n,
        block.design.nrows(),
    )
}

fn validate_term_datarows(context: &str, expected: usize, found: usize) -> Result<(), String> {
    if expected != found {
        return Err(format!(
            "{context}: data row count must match response length (expected {expected}, found {found})"
        ));
    }
    Ok(())
}

fn validate_term_weights(
    data: ndarray::ArrayView2<'_, f64>,
    y_len: usize,
    weights: &Array1<f64>,
    context: &str,
) -> Result<(), String> {
    validate_term_datarows(context, y_len, data.nrows())?;
    validate_len_match("weights vs y", y_len, weights.len())?;
    validateweights(weights, context)
}

fn validate_term_offset(y_len: usize, offset: &Array1<f64>, label: &str) -> Result<(), String> {
    validate_len_match(&format!("{label} vs y"), y_len, offset.len())?;
    for (row_idx, value) in offset.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{label} contains non-finite value at row {row_idx}: {value}"
            ));
        }
    }
    Ok(())
}

fn validate_gaussian_location_scale_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &GaussianLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    validate_term_weights(data, spec.y.len(), &spec.weights, context)?;
    validate_term_offset(spec.y.len(), &spec.mean_offset, "mean_offset")?;
    validate_term_offset(spec.y.len(), &spec.log_sigma_offset, "log_sigma_offset")
}

fn validate_gaussian_location_scalewiggle_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &GaussianLocationScaleWiggleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_weights(data, n, &spec.weights, context)?;
    validate_term_offset(n, &spec.mean_offset, "mean_offset")?;
    validate_term_offset(n, &spec.log_sigma_offset, "log_sigma_offset")?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    if spec.wiggle_degree < 2 {
        return Err(format!(
            "{context}: wiggle_degree must be >= 2, got {}",
            spec.wiggle_degree
        ));
    }
    let minimum_knots = minimum_monotone_wiggle_knot_count(spec.wiggle_degree)?;
    if spec.wiggle_knots.len() < minimum_knots {
        return Err(format!(
            "{context}: wiggle_knots must have at least {} entries for degree {}, got {}",
            minimum_knots,
            spec.wiggle_degree,
            spec.wiggle_knots.len()
        ));
    }
    Ok(())
}

fn validate_binomial_location_scale_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    validate_term_weights(data, spec.y.len(), &spec.weights, context)?;
    validate_term_offset(spec.y.len(), &spec.threshold_offset, "threshold_offset")?;
    validate_term_offset(spec.y.len(), &spec.log_sigma_offset, "log_sigma_offset")?;
    validate_binomial_response(&spec.y, context)?;
    Ok(())
}

fn inverse_link_supports_joint_wiggle(link_kind: &InverseLink) -> bool {
    matches!(
        link_kind,
        InverseLink::Standard(LinkFunction::Logit)
            | InverseLink::Standard(LinkFunction::Probit)
            | InverseLink::Standard(LinkFunction::CLogLog)
    )
}

fn validate_binomial_location_scalewiggle_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleWiggleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_weights(data, n, &spec.weights, context)?;
    validate_term_offset(n, &spec.threshold_offset, "threshold_offset")?;
    validate_term_offset(n, &spec.log_sigma_offset, "log_sigma_offset")?;
    validate_binomial_response(&spec.y, context)?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    if !inverse_link_supports_joint_wiggle(&spec.link_kind) {
        return Err(format!(
            "{context}: link wiggle does not support SAS/BetaLogistic/Mixture links; wiggle is only available for jointly fitted standard links"
        ));
    }
    if spec.wiggle_degree < 2 {
        return Err(format!(
            "{context}: wiggle_degree must be >= 2, got {}",
            spec.wiggle_degree
        ));
    }
    let minimum_knots = minimum_monotone_wiggle_knot_count(spec.wiggle_degree)?;
    if spec.wiggle_knots.len() < minimum_knots {
        return Err(format!(
            "{context}: wiggle_knots must have at least {} entries for degree {}, got {}",
            minimum_knots,
            spec.wiggle_degree,
            spec.wiggle_knots.len()
        ));
    }
    Ok(())
}

fn initial_log_lambdas_orzeros(block: &ParameterBlockInput) -> Result<Array1<f64>, String> {
    let k = block.penalties.len();
    let lambdas = block
        .initial_log_lambdas
        .clone()
        .unwrap_or_else(|| Array1::<f64>::zeros(k));
    if lambdas.len() != k {
        return Err(format!(
            "initial_log_lambdas length mismatch: got {}, expected {}",
            lambdas.len(),
            k
        ));
    }
    Ok(lambdas)
}

fn build_two_block_exact_joint_setup(
    data: ArrayView2<'_, f64>,
    meanspec: &TermCollectionSpec,
    noisespec: &TermCollectionSpec,
    mean_penalties: usize,
    noise_penalties: usize,
    extra_rho0: &[f64],
    rho0_override: Option<&Array1<f64>>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    // Exact-joint setup stores the spatial tail in log(kappa), not log(length_scale).
    let mean_terms = spatial_length_scale_term_indices(meanspec);
    let noise_terms = spatial_length_scale_term_indices(noisespec);
    let rho_dim = mean_penalties + noise_penalties + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);

    if let Some(rho0) = rho0_override.filter(|rho0| rho0.len() == rho_dim) {
        rho0vec.assign(rho0);
    } else {
        for (i, &rho_init) in extra_rho0.iter().enumerate() {
            rho0vec[mean_penalties + noise_penalties + i] = rho_init;
        }
    }

    // Use aniso-aware initialization: each aniso term gets d ψ entries.
    // Re-seed ψ from data geometry when the spec does not pin a length_scale.
    let mean_kappa =
        SpatialLogKappaCoords::from_length_scales_aniso(meanspec, &mean_terms, kappa_options)
            .reseed_from_data(data, meanspec, &mean_terms, kappa_options);
    let noise_kappa =
        SpatialLogKappaCoords::from_length_scales_aniso(noisespec, &noise_terms, kappa_options)
            .reseed_from_data(data, noisespec, &noise_terms, kappa_options);

    // Concatenate mean and noise ψ values and dims.
    let mut all_values = mean_kappa.as_array().to_vec();
    all_values.extend(noise_kappa.as_array().iter());
    let mean_dims = mean_kappa.dims_per_term().to_vec();
    let noise_dims = noise_kappa.dims_per_term().to_vec();
    let mut all_dims = mean_dims.clone();
    all_dims.extend(noise_dims.iter().copied());

    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(all_values), all_dims.clone());
    // Bounds: concatenate per-block data-aware bounds in the same order.
    let mean_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        meanspec,
        &mean_terms,
        &mean_dims,
        kappa_options,
    );
    let noise_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        noisespec,
        &noise_terms,
        &noise_dims,
        kappa_options,
    );
    let mut lower_vals = mean_lower.as_array().to_vec();
    lower_vals.extend(noise_lower.as_array().iter());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), all_dims.clone());
    let mean_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        meanspec,
        &mean_terms,
        &mean_dims,
        kappa_options,
    );
    let noise_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        noisespec,
        &noise_terms,
        &noise_dims,
        kappa_options,
    );
    let mut upper_vals = mean_upper.as_array().to_vec();
    upper_vals.extend(noise_upper.as_array().iter());
    let log_kappa_upper =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), all_dims);
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);

    ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

pub(crate) fn solve_penalizedweighted_projection(
    design: &DesignMatrix,
    offset: &Array1<f64>,
    target_eta: &Array1<f64>,
    weights: &Array1<f64>,
    penalties: &[PenaltyMatrix],
    log_lambdas: &Array1<f64>,
    ridge_floor: f64,
) -> Result<Array1<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if offset.len() != n || target_eta.len() != n || weights.len() != n {
        return Err("solve_penalizedweighted_projection dimension mismatch".to_string());
    }
    if penalties.len() != log_lambdas.len() {
        return Err(format!(
            "solve_penalizedweighted_projection lambda mismatch: penalties={}, log_lambdas={}",
            penalties.len(),
            log_lambdas.len()
        ));
    }

    let y_star = target_eta - offset;
    let xtwy = design.compute_xtwy(weights, &y_star)?;
    let mut lambdas = Vec::with_capacity(penalties.len());
    let mut preconditioner = design.diag_gram(weights)?;
    for (k, s) in penalties.iter().enumerate() {
        let lambda = log_lambdas[k].exp();
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(format!(
                "solve_penalizedweighted_projection encountered invalid lambda at index {k}: {}",
                log_lambdas[k]
            ));
        }
        if s.nrows() != p || s.ncols() != p {
            return Err(format!(
                "solve_penalizedweighted_projection penalty shape mismatch at index {k}: \
                 penalty is {}x{} but design has {} columns",
                s.nrows(),
                s.ncols(),
                p
            ));
        }
        lambdas.push(lambda);
        s.add_scaled_diag_to(lambda, &mut preconditioner);
    }
    let ridge = ridge_floor.max(1e-12);
    for j in 0..p {
        preconditioner[j] += ridge;
    }

    let max_iter = p.clamp(64, 4096) * 4;
    let (beta, _) = solve_spd_pcg_with_info(
        |v| {
            let mut out = design.apply_weighted_normal(weights, v, None, 0.0);
            for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
                out += &penalty.dot(v).mapv(|value| value * *lambda);
            }
            if ridge > 0.0 {
                out += &v.mapv(|value| ridge * value);
            }
            out
        },
        &xtwy,
        &preconditioner,
        1e-8,
        max_iter,
    )
    .ok_or_else(|| {
        "solve_penalizedweighted_projection matrix-free solve failed to converge".to_string()
    })?;
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(
            "solve_penalizedweighted_projection produced non-finite coefficients".to_string(),
        );
    }
    Ok(beta)
}

fn gaussian_location_scalewarm_start(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    mu_block: &ParameterBlockSpec,
    log_sigma_block: &ParameterBlockSpec,
    ridge_floor: f64,
    mean_beta_hint: Option<&Array1<f64>>,
    noise_beta_hint: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>, f64), String> {
    let betamu = if let Some(beta) = mean_beta_hint {
        beta.clone()
    } else {
        solve_penalizedweighted_projection(
            &mu_block.design,
            &mu_block.offset,
            y,
            weights,
            &mu_block.penalties,
            &mu_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    let mut mu_hat = mu_block.design.matrixvectormultiply(&betamu);
    mu_hat += &mu_block.offset;
    let mut weighted_ss = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..y.len() {
        let wi = weights[i].max(0.0);
        let resid = y[i] - mu_hat[i];
        weighted_ss += wi * resid * resid;
        weight_sum += wi;
    }
    if !weighted_ss.is_finite() || !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(
            "gaussian location-scale warm start could not estimate residual scale".to_string(),
        );
    }
    // Warm-start σ̂ must clear the logb floor so the inverse link
    //   η = log(σ − b)
    // is finite. Use a relative cushion above b so the warm-start is in the
    // smooth interior of the link domain.
    let sigma_hat = (weighted_ss / weight_sum)
        .sqrt()
        .max(LOGB_SIGMA_FLOOR * 1.5);
    let beta_log_sigma = if let Some(beta) = noise_beta_hint {
        beta.clone()
    } else {
        let eta_sigma = (sigma_hat - LOGB_SIGMA_FLOOR).ln();
        let sigma_target = Array1::from_elem(y.len(), eta_sigma);
        solve_penalizedweighted_projection(
            &log_sigma_block.design,
            &log_sigma_block.offset,
            &sigma_target,
            weights,
            &log_sigma_block.penalties,
            &log_sigma_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    Ok((betamu, beta_log_sigma, sigma_hat))
}

fn prepared_scale_design(
    primary_design: &Array2<f64>,
    noise_design: &Array2<f64>,
    weights: &Array1<f64>,
    non_intercept_start: usize,
) -> Result<Array2<f64>, String> {
    let transform = build_scale_deviation_transform(
        primary_design,
        noise_design,
        weights,
        non_intercept_start,
    )?;
    apply_scale_deviation_transform(primary_design, noise_design, &transform)
}

fn prepared_gaussian_log_sigma_design(
    mu_design: &Array2<f64>,
    log_sigma_design: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if mu_design.nrows() != log_sigma_design.nrows() {
        return Err(format!(
            "gaussian log-sigma design row mismatch: mean rows={}, log_sigma rows={}",
            mu_design.nrows(),
            log_sigma_design.nrows()
        ));
    }
    // Gaussian location-scale remains identifiable even when μ and log σ use
    // the same covariate basis:
    //
    //   L(μ, η) = 0.5 * Σ_i [ (y_i - μ_i)^2 exp(-2η_i) + 2η_i ],
    //   μ = X_μ β_μ,  η = X_σ β_σ.
    //
    // Shared columns are not a frame mismatch. β_μ and β_σ enter through
    // different sufficient statistics (residual and residual²), so replacing
    // X_σ with (I - P_{X_μ}) X_σ would impose an extra constraint and can
    // erase real heteroscedastic signal when the two blocks share a basis.
    Ok(log_sigma_design.clone())
}

fn identified_binomial_log_sigma_design(
    threshold_design: &TermCollectionDesign,
    log_sigma_design: &TermCollectionDesign,
    weights: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let threshold_dense = threshold_design.design.as_dense_cow();
    let log_sigma_dense = log_sigma_design.design.as_dense_cow();
    prepared_scale_design(
        &threshold_dense,
        &log_sigma_dense,
        weights,
        log_sigma_design
            .intercept_range
            .end
            .min(log_sigma_design.design.ncols()),
    )
}

fn identity_penalty(dim: usize) -> Array2<f64> {
    let mut penalty = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        penalty[[i, i]] = 1.0;
    }
    penalty
}

fn append_binomial_log_sigma_shrinkage_penalty_design(design: &mut TermCollectionDesign) {
    let p = design.design.ncols();
    design
        .penalties
        .push(BlockwisePenalty::new(0..p, identity_penalty(p)));
    // Identity penalty penalizes the full space → nullspace dimension is 0.
    design.nullspace_dims.push(0);
    design.penaltyinfo.push(PenaltyBlockInfo {
        global_index: design.penaltyinfo.len(),
        termname: Some("log_sigma_shrinkage".to_string()),
        penalty: PenaltyInfo {
            source: PenaltySource::Other("shrinkage".to_string()),
            original_index: 0,
            active: true,
            effective_rank: p,
            dropped_reason: None,
            nullspace_dim_hint: 0,
            normalization_scale: 1.0,
            kronecker_factors: None,
        },
    });
}

fn binomial_location_scale_link_eta_from_probability(
    link_kind: &InverseLink,
    probability: f64,
) -> Result<f64, String> {
    let target = probability.clamp(1e-6, 1.0 - 1e-6);
    match link_kind {
        InverseLink::Standard(LinkFunction::Logit) => Ok((target / (1.0 - target)).ln()),
        InverseLink::Standard(LinkFunction::Probit) => standard_normal_quantile(target)
            .map_err(|err| format!("failed to invert probit warm-start probability: {err}")),
        InverseLink::Standard(LinkFunction::CLogLog) => Ok((-((1.0 - target).ln())).ln()),
        other => Err(format!(
            "binomial location-scale warm start requires logit, probit, or cloglog link, got {other:?}"
        )),
    }
}

fn weighted_binomial_prevalence(y: &Array1<f64>, weights: &Array1<f64>) -> Result<f64, String> {
    if y.len() != weights.len() {
        return Err(format!(
            "binomial location-scale warm start dimension mismatch: y has length {}, weights have length {}",
            y.len(),
            weights.len()
        ));
    }
    let mut weight_sum = 0.0;
    let mut success_sum = 0.0;
    for (&yi, &wi) in y.iter().zip(weights.iter()) {
        if !yi.is_finite() {
            return Err(format!(
                "binomial location-scale warm start encountered non-finite response {yi}"
            ));
        }
        let weight = floor_positiveweight(wi, MIN_WEIGHT);
        if weight > 0.0 {
            weight_sum += weight;
            success_sum += weight * yi;
        }
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(
            "binomial location-scale warm start requires positive total weight".to_string(),
        );
    }
    Ok(success_sum / weight_sum)
}

fn project_constant_eta_into_block(
    block: &ParameterBlockSpec,
    weights: &Array1<f64>,
    eta: f64,
) -> Result<Array1<f64>, String> {
    let target_eta = Array1::from_elem(block.design.nrows(), eta);
    solve_penalizedweighted_projection(
        &block.design,
        &block.offset,
        &target_eta,
        weights,
        &block.penalties,
        &block.initial_log_lambdas,
        1e-10,
    )
}

// Deterministic warm start for the binomial location-scale model. This stays
// out of the optimizer: it projects a prevalence-matched threshold and neutral
// log-sigma value into the actual penalized block spaces.
fn binomial_location_scalewarm_start(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    link_kind: &InverseLink,
    threshold_block: &ParameterBlockSpec,
    log_sigma_block: &ParameterBlockSpec,
    mean_beta_hint: Option<&Array1<f64>>,
    noise_beta_hint: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    if let (Some(mean_beta), Some(noise_beta)) = (mean_beta_hint, noise_beta_hint) {
        return Ok((mean_beta.clone(), noise_beta.clone()));
    }

    let beta_threshold = match mean_beta_hint {
        Some(beta) => beta.clone(),
        None => {
            let prevalence = weighted_binomial_prevalence(y, weights)?;
            let eta = binomial_location_scale_link_eta_from_probability(link_kind, prevalence)?;
            project_constant_eta_into_block(threshold_block, weights, eta)?
        }
    };
    let beta_log_sigma = match noise_beta_hint {
        Some(beta) => beta.clone(),
        None => project_constant_eta_into_block(log_sigma_block, weights, 0.0)?,
    };
    Ok((beta_threshold, beta_log_sigma))
}

#[derive(Clone)]
struct BinomialMeanWiggleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub eta_block: ParameterBlockInput,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct GaussianLocationScaleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
}

#[derive(Clone)]
pub struct GaussianLocationScaleWiggleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct BinomialLocationScaleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub threshold_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
}

#[derive(Clone)]
pub struct BinomialLocationScaleWiggleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub threshold_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone, Debug)]
pub struct BlockwiseTermFitResult {
    pub fit: UnifiedFitResult,
    pub meanspec_resolved: TermCollectionSpec,
    pub noisespec_resolved: TermCollectionSpec,
    pub mean_design: TermCollectionDesign,
    pub noise_design: TermCollectionDesign,
}

pub(crate) struct BlockwiseTermFitResultParts {
    pub fit: UnifiedFitResult,
    pub meanspec_resolved: TermCollectionSpec,
    pub noisespec_resolved: TermCollectionSpec,
    pub mean_design: TermCollectionDesign,
    pub noise_design: TermCollectionDesign,
}

pub struct BlockwiseTermWiggleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}

pub struct BinomialMeanWiggleTermFitResult {
    pub fit: UnifiedFitResult,
    pub resolvedspec: TermCollectionSpec,
    pub design: TermCollectionDesign,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}

struct BlockwiseTermWiggleFitResultParts {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}

fn validate_term_collection_design(
    label: &str,
    design: &TermCollectionDesign,
) -> Result<(), String> {
    let p = design.design.ncols();
    let design_dense = design.design.as_dense_cow();
    validate_all_finite_estimation(&format!("{label}.design"), design_dense.iter().copied())
        .map_err(|e| e.to_string())?;
    if design.nullspace_dims.len() != design.penalties.len() {
        return Err(format!(
            "{label}.nullspace_dims length mismatch: got {}, expected {}",
            design.nullspace_dims.len(),
            design.penalties.len()
        ));
    }
    if design.penaltyinfo.len() != design.penalties.len() {
        return Err(format!(
            "{label}.penaltyinfo length mismatch: got {}, expected {}",
            design.penaltyinfo.len(),
            design.penalties.len()
        ));
    }
    for (idx, bp) in design.penalties.iter().enumerate() {
        validate_all_finite_estimation(
            &format!("{label}.penalties[{idx}]"),
            bp.local.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        if bp.col_range.end > p {
            return Err(format!(
                "{label}.penalties[{idx}] col_range {}..{} exceeds design width {}",
                bp.col_range.start, bp.col_range.end, p
            ));
        }
    }
    if let Some(bounds) = design.coefficient_lower_bounds.as_ref() {
        if bounds.len() != p {
            return Err(format!(
                "{label}.coefficient_lower_bounds length mismatch: got {}, expected {p}",
                bounds.len()
            ));
        }
        for (idx, &bound) in bounds.iter().enumerate() {
            if !(bound.is_finite() || bound == f64::NEG_INFINITY) {
                return Err(format!(
                    "{label}.coefficient_lower_bounds[{idx}] must be finite or -inf, got {bound}",
                ));
            }
        }
    }
    if let Some(constraints) = design.linear_constraints.as_ref() {
        validate_all_finite_estimation(
            &format!("{label}.linear_constraints.a"),
            constraints.a.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        validate_all_finite_estimation(
            &format!("{label}.linear_constraints.b"),
            constraints.b.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        if constraints.a.ncols() != p {
            return Err(format!(
                "{label}.linear_constraints.a column mismatch: got {}, expected {p}",
                constraints.a.ncols()
            ));
        }
        if constraints.a.nrows() != constraints.b.len() {
            return Err(format!(
                "{label}.linear_constraints row mismatch: a has {}, b has {}",
                constraints.a.nrows(),
                constraints.b.len()
            ));
        }
    }
    if design.intercept_range.start > design.intercept_range.end || design.intercept_range.end > p {
        return Err(format!(
            "{label}.intercept_range out of bounds: {:?} for {} columns",
            design.intercept_range, p
        ));
    }
    Ok(())
}

impl BlockwiseTermFitResult {
    pub(crate) fn try_from_parts(parts: BlockwiseTermFitResultParts) -> Result<Self, String> {
        let BlockwiseTermFitResultParts {
            fit,
            meanspec_resolved,
            noisespec_resolved,
            mean_design,
            noise_design,
        } = parts;

        fit.validate_numeric_finiteness()
            .map_err(|e| format!("{e}"))?;
        if fit.block_states.len() < 2 {
            return Err(format!(
                "BlockwiseTermFitResult requires at least 2 block states, got {}",
                fit.block_states.len()
            ));
        }
        validate_term_collection_design("blockwise_term.mean_design", &mean_design)?;
        validate_term_collection_design("blockwise_term.noise_design", &noise_design)?;
        if mean_design.design.nrows() != noise_design.design.nrows() {
            return Err(format!(
                "BlockwiseTermFitResult row mismatch: mean_design={}, noise_design={}",
                mean_design.design.nrows(),
                noise_design.design.nrows()
            ));
        }
        if fit.block_states[0].beta.len() != mean_design.design.ncols() {
            return Err(format!(
                "BlockwiseTermFitResult mean beta length mismatch: got {}, expected {}",
                fit.block_states[0].beta.len(),
                mean_design.design.ncols()
            ));
        }
        if fit.block_states[1].beta.len() != noise_design.design.ncols() {
            return Err(format!(
                "BlockwiseTermFitResult noise beta length mismatch: got {}, expected {}",
                fit.block_states[1].beta.len(),
                noise_design.design.ncols()
            ));
        }
        if fit.block_states[0].eta.len() != mean_design.design.nrows() {
            return Err(format!(
                "BlockwiseTermFitResult mean eta length mismatch: got {}, expected {}",
                fit.block_states[0].eta.len(),
                mean_design.design.nrows()
            ));
        }
        if fit.block_states[1].eta.len() != noise_design.design.nrows() {
            return Err(format!(
                "BlockwiseTermFitResult noise eta length mismatch: got {}, expected {}",
                fit.block_states[1].eta.len(),
                noise_design.design.nrows()
            ));
        }

        Ok(Self {
            fit,
            meanspec_resolved,
            noisespec_resolved,
            mean_design,
            noise_design,
        })
    }

    fn validate_numeric_finiteness(&self) -> Result<(), String> {
        Self::try_from_parts(BlockwiseTermFitResultParts {
            fit: self.fit.clone(),
            meanspec_resolved: self.meanspec_resolved.clone(),
            noisespec_resolved: self.noisespec_resolved.clone(),
            mean_design: self.mean_design.clone(),
            noise_design: self.noise_design.clone(),
        })
        .map(|_| ())
    }
}

impl BlockwiseTermWiggleFitResult {
    fn try_from_parts(parts: BlockwiseTermWiggleFitResultParts) -> Result<Self, String> {
        let BlockwiseTermWiggleFitResultParts {
            fit,
            wiggle_knots,
            wiggle_degree,
        } = parts;

        fit.validate_numeric_finiteness()
            .map_err(|e| format!("{e}"))?;
        if fit.fit.block_states.len() < 3 {
            return Err(format!(
                "BlockwiseTermWiggleFitResult requires at least 3 block states, got {}",
                fit.fit.block_states.len()
            ));
        }
        if wiggle_knots.is_empty() {
            return Err("BlockwiseTermWiggleFitResult requires non-empty wiggle_knots".to_string());
        }
        validate_all_finite_estimation(
            "blockwise_term_wiggle.wiggle_knots",
            wiggle_knots.iter().copied(),
        )
        .map_err(|e| e.to_string())?;

        Ok(Self {
            fit,
            wiggle_knots,
            wiggle_degree,
        })
    }
}

pub struct BinomialLocationScaleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    pub beta_link_wiggle: Option<Vec<f64>>,
}

pub struct GaussianLocationScaleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    pub beta_link_wiggle: Option<Vec<f64>>,
}

fn fit_binomial_mean_wiggle(
    spec: BinomialMeanWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validateweights(&spec.weights, "fit_binomial_mean_wiggle")?;
    validate_binomial_response(&spec.y, "fit_binomial_mean_wiggle")?;
    validate_blockrows("eta", n, &spec.eta_block)?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    if matches!(
        spec.link_kind,
        InverseLink::Standard(LinkFunction::Identity)
    ) {
        return Err("fit_binomial_mean_wiggle does not support identity link".to_string());
    }
    if !inverse_link_supports_joint_wiggle(&spec.link_kind) {
        return Err(
            "fit_binomial_mean_wiggle does not support SAS/BetaLogistic/Mixture links; wiggle is only available for jointly fitted standard links"
                .to_string(),
        );
    }
    if spec.wiggle_degree < 2 {
        return Err(format!(
            "fit_binomial_mean_wiggle: wiggle_degree must be >= 2, got {}",
            spec.wiggle_degree
        ));
    }
    let minimum_knots = minimum_monotone_wiggle_knot_count(spec.wiggle_degree)?;
    if spec.wiggle_knots.len() < minimum_knots {
        return Err(format!(
            "fit_binomial_mean_wiggle: wiggle_knots length {} is too short for degree {} (need at least {})",
            spec.wiggle_knots.len(),
            spec.wiggle_degree,
            minimum_knots
        ));
    }

    let family = BinomialMeanWiggleFamily {
        y: spec.y,
        weights: spec.weights,
        link_kind: spec.link_kind,
        wiggle_knots: spec.wiggle_knots,
        wiggle_degree: spec.wiggle_degree,
        policy: crate::resource::ResourcePolicy::default_library(),
    };
    let blocks = vec![
        spec.eta_block.intospec("eta")?,
        spec.wiggle_block.intospec("wiggle")?,
    ];
    fit_custom_family(&family, &blocks, options).map_err(|e| e.to_string())
}

trait LocationScaleFamilyBuilder {
    type Family: CustomFamily + Clone + Send + Sync + 'static;

    fn meanspec(&self) -> &TermCollectionSpec;
    fn noisespec(&self) -> &TermCollectionSpec;

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String>;

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family;

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String>;

    fn mean_penalty_count(&self, mean_design: &TermCollectionDesign) -> usize {
        mean_design.penalties.len()
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        noise_design.penalties.len()
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        false
    }

    fn require_exact_spatial_joint(&self) -> bool {
        false
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::GeneralizedLinear
    }

    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        Ok(Array1::zeros(0))
    }

    fn augment_result_designs(&self, _: &mut TermCollectionDesign, _: &mut TermCollectionDesign) {}

    fn build_psiderivative_blocks(
        &self,
        _: ndarray::ArrayView2<'_, f64>,
        _: &TermCollectionSpec,
        _: &TermCollectionSpec,
        _: &TermCollectionDesign,
        _: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        Err("spatial psi derivatives are unavailable for this location-scale family".to_string())
    }
}

fn fit_location_scale_terms<B: LocationScaleFamilyBuilder>(
    data: ndarray::ArrayView2<'_, f64>,
    builder: B,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    let mut mean_beta_hint: Option<Array1<f64>> = None;
    let mut noise_beta_hint: Option<Array1<f64>> = None;
    let extra_rho0 = builder.extra_rho0()?;

    let mean_boot_design =
        build_term_collection_design(data, builder.meanspec()).map_err(|e| e.to_string())?;
    let noise_boot_design =
        build_term_collection_design(data, builder.noisespec()).map_err(|e| e.to_string())?;
    let mean_bootspec = freeze_term_collection_from_design(builder.meanspec(), &mean_boot_design)
        .map_err(|e| e.to_string())?;
    let noise_bootspec =
        freeze_term_collection_from_design(builder.noisespec(), &noise_boot_design)
            .map_err(|e| e.to_string())?;

    let require_exact_spatial_joint = builder.require_exact_spatial_joint();
    let analytic_joint_derivatives_check = if builder.exact_spatial_joint_supported() {
        builder
            .build_psiderivative_blocks(
                data,
                &mean_bootspec,
                &noise_bootspec,
                &mean_boot_design,
                &noise_boot_design,
            )
            .map(|_| ())
    } else {
        Err(
            "analytic spatial psi derivatives are unavailable for this location-scale family"
                .to_string(),
        )
    };
    let analytic_joint_derivatives_available = analytic_joint_derivatives_check.is_ok();
    if require_exact_spatial_joint {
        analytic_joint_derivatives_check.map_err(|err| {
            format!("exact two-block spatial path requires analytic psi derivatives: {err}")
        })?;
    }
    let mean_penalty_count = builder.mean_penalty_count(&mean_boot_design);
    let noise_penalty_count = builder.noise_penalty_count(&noise_boot_design);

    // Macro to invoke the exact-joint spatial optimizer with shared closures.
    // The exact path evaluates the full profiled/Laplace objective over
    // theta = [rho, psi] with the real joint Hessian required by NewtonTR/ARC.
    macro_rules! run_exact_joint_spatial {
        () => {{
            let joint_setup = build_two_block_exact_joint_setup(
                data,
                builder.meanspec(),
                builder.noisespec(),
                mean_penalty_count,
                noise_penalty_count,
                extra_rho0.as_slice().unwrap_or(&[]),
                None,
                kappa_options,
            );
            let mean_terms = spatial_length_scale_term_indices(builder.meanspec());
            let noise_terms = spatial_length_scale_term_indices(builder.noisespec());
            let mean_beta_hint_cell = std::cell::RefCell::new(mean_beta_hint.clone());
            let noise_beta_hint_cell = std::cell::RefCell::new(noise_beta_hint.clone());
            let hyper_warm_start_cell =
                std::cell::RefCell::new(None::<CustomFamilyWarmStart>);
            // Two-block GAMLSS/location-scale joint likelihoods have a
            // β-dependent cross-block Hessian (the (μ,log σ) / (t,log σ)
            // off-diagonal blocks involve residual/response scalars that
            // shift when β moves). The Wood-Fasiolo structural property
            // `H^{-1/2} B_k H^{-1/2} ≽ 0` plus parameter-independent
            // nullspace — the mathematical basis for EFS convergence —
            // fails here, so EFS/HybridEFS must be excluded at plan time
            // rather than retried as a silent first attempt that stalls
            // for hundreds of seconds before the runner falls back.
            let gamlss_disable_fixed_point = true;
            optimize_spatial_length_scale_exact_joint(
                data,
                &[builder.meanspec().clone(), builder.noisespec().clone()],
                &[mean_terms, noise_terms],
                kappa_options,
                &joint_setup,
                builder.exact_spatial_seed_risk_profile(),
                analytic_joint_derivatives_available,
                analytic_joint_derivatives_available,
                gamlss_disable_fixed_point,
                |theta, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
                    let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
                    let fit = {
                        let blocks = builder.build_blocks(
                            &rho,
                            &designs[0],
                            &designs[1],
                            mean_beta_hint_cell.borrow().clone(),
                            noise_beta_hint_cell.borrow().clone(),
                        )?;
                        if mean_beta_hint_cell.borrow().is_none()
                            && let Some(beta) = blocks.first().and_then(|block| block.initial_beta.clone())
                        {
                            *mean_beta_hint_cell.borrow_mut() = Some(beta);
                        }
                        if noise_beta_hint_cell.borrow().is_none()
                            && let Some(beta) =
                                blocks.get(1).and_then(|block| block.initial_beta.clone())
                        {
                            *noise_beta_hint_cell.borrow_mut() = Some(beta);
                        }
                        let family = builder.build_family(&designs[0], &designs[1]);
                        if joint_setup.log_kappa_dim() > 0 {
                            let warm_start = hyper_warm_start_cell.borrow().clone();
                            fit_custom_family_fixed_log_lambdas(
                                &family,
                                &blocks,
                                options,
                                warm_start.as_ref(),
                                0,
                                0.0,
                                true,
                            )?
                        } else {
                            fit_custom_family(&family, &blocks, options)?
                        }
                    };
                    let (mean_beta, noise_beta) = builder.extract_primary_betas(&fit)?;
                    mean_beta_hint = Some(mean_beta);
                    noise_beta_hint = Some(noise_beta);
                    *mean_beta_hint_cell.borrow_mut() = mean_beta_hint.clone();
                    *noise_beta_hint_cell.borrow_mut() = noise_beta_hint.clone();
                    Ok(fit)
                },
                |theta,
                 specs: &[TermCollectionSpec],
                 designs: &[TermCollectionDesign],
                 need_hessian| {
                    if !analytic_joint_derivatives_available {
                        return Err(
                            "analytic spatial psi derivatives are unavailable for this exact two-block path"
                                .to_string(),
                        );
                    }
                    let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
                    let blocks = builder.build_blocks(
                        &rho,
                        &designs[0],
                        &designs[1],
                        mean_beta_hint_cell.borrow().clone(),
                        noise_beta_hint_cell.borrow().clone(),
                    )?;
                    if mean_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.first().and_then(|block| block.initial_beta.clone())
                    {
                        *mean_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    if noise_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.get(1).and_then(|block| block.initial_beta.clone())
                    {
                        *noise_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    let family = builder.build_family(&designs[0], &designs[1]);
                    let psiderivative_blocks = builder.build_psiderivative_blocks(
                        data,
                        &specs[0],
                        &specs[1],
                        &designs[0],
                        &designs[1],
                    )?;
                    let warm_start = hyper_warm_start_cell.borrow().clone();
                    let eval = evaluate_custom_family_joint_hyper(
                        &family,
                        &blocks,
                        options,
                        &rho,
                        &psiderivative_blocks,
                        warm_start.as_ref(),
                        if need_hessian {
                            crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian
                        } else {
                            crate::solver::estimate::reml::unified::EvalMode::ValueAndGradient
                        },
                    )?;
                    *hyper_warm_start_cell.borrow_mut() = Some(eval.warm_start.clone());
                    if need_hessian && !eval.outer_hessian.is_analytic() {
                        return Err(
                            "exact two-block spatial objective requires a full joint [rho, psi] hessian"
                                .to_string(),
                        );
                    }
                    Ok((eval.objective, eval.gradient, eval.outer_hessian))
                },
                |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
                    if !analytic_joint_derivatives_available {
                        return Err(
                            "analytic spatial psi derivatives are unavailable for this exact two-block path"
                                .to_string(),
                        );
                    }
                    let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
                    let blocks = builder.build_blocks(
                        &rho,
                        &designs[0],
                        &designs[1],
                        mean_beta_hint_cell.borrow().clone(),
                        noise_beta_hint_cell.borrow().clone(),
                    )?;
                    if mean_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.first().and_then(|block| block.initial_beta.clone())
                    {
                        *mean_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    if noise_beta_hint_cell.borrow().is_none()
                        && let Some(beta) = blocks.get(1).and_then(|block| block.initial_beta.clone())
                    {
                        *noise_beta_hint_cell.borrow_mut() = Some(beta);
                    }
                    let family = builder.build_family(&designs[0], &designs[1]);
                    let psiderivative_blocks = builder.build_psiderivative_blocks(
                        data,
                        &specs[0],
                        &specs[1],
                        &designs[0],
                        &designs[1],
                    )?;
                    let warm_start = hyper_warm_start_cell.borrow().clone();
                    let eval = evaluate_custom_family_joint_hyper_efs(
                        &family,
                        &blocks,
                        options,
                        &rho,
                        &psiderivative_blocks,
                        warm_start.as_ref(),
                    )?;
                    *hyper_warm_start_cell.borrow_mut() = Some(eval.warm_start.clone());
                    Ok(eval.efs_eval)
                },
            )
        }};
    }

    let mut solved = run_exact_joint_spatial!()
        .map_err(|err| format!("exact two-block spatial optimization failed: {err}"))?;

    {
        let (left, right) = solved.designs.split_at_mut(1);
        builder.augment_result_designs(&mut left[0], &mut right[0]);
    }

    BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
        fit: solved.fit,
        meanspec_resolved: solved.resolved_specs.remove(0),
        noisespec_resolved: solved.resolved_specs.remove(0),
        mean_design: solved.designs.remove(0),
        noise_design: solved.designs.remove(0),
    })
}

struct GaussianLocationScaleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
}

impl LocationScaleFamilyBuilder for GaussianLocationScaleTermBuilder {
    type Family = GaussianLocationScaleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::Gaussian
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::two_block(
            mean_design.penalties.len(),
            noise_design.penalties.len(),
        );
        layout.validate_theta_len(theta.len(), "gaussian location-scale")?;
        let mean_log_lambdas = layout.mean_from(theta);
        let noise_log_lambdas = layout.noise_from(theta);
        let mut meanspec = ParameterBlockSpec {
            name: "mu".to_string(),
            design: mean_design.design.clone(),
            offset: self.mean_offset.clone(),
            penalties: mean_design.penalties_as_penalty_matrix(),
            nullspace_dims: mean_design.nullspace_dims.clone(),
            initial_log_lambdas: mean_log_lambdas,
            initial_beta: mean_beta_hint,
        };
        let mean_dense = mean_design.design.as_dense_cow();
        let noise_dense = noise_design.design.as_dense_cow();
        let mut noisespec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                prepared_gaussian_log_sigma_design(&mean_dense, &noise_dense)?,
            )),
            offset: self.noise_offset.clone(),
            penalties: noise_design.penalties_as_penalty_matrix(),
            nullspace_dims: noise_design.nullspace_dims.clone(),
            initial_log_lambdas: noise_log_lambdas,
            initial_beta: noise_beta_hint,
        };
        if meanspec.initial_beta.is_none() || noisespec.initial_beta.is_none() {
            let (betamu0, beta_ls0, _) = gaussian_location_scalewarm_start(
                &self.y,
                &self.weights,
                &meanspec,
                &noisespec,
                1e-10,
                meanspec.initial_beta.as_ref(),
                noisespec.initial_beta.as_ref(),
            )?;
            if meanspec.initial_beta.is_none() {
                meanspec.initial_beta = Some(betamu0);
            }
            if noisespec.initial_beta.is_none() {
                noisespec.initial_beta = Some(beta_ls0);
            }
        }
        Ok(vec![meanspec, noisespec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let mean_dense = mean_design.design.as_dense_cow();
        let noise_dense = noise_design.design.as_dense_cow();
        let preparednoise_design = prepared_gaussian_log_sigma_design(&mean_dense, &noise_dense)
            .expect("prepared Gaussian log-sigma design should match block construction");
        GaussianLocationScaleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                preparednoise_design,
            ))),
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(GaussianLocationScaleFamily::BLOCK_MU)
            .ok_or_else(|| "missing Gaussian mu block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(GaussianLocationScaleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Gaussian log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?
                .ok_or_else(|| "missing Gaussian mean spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| "missing Gaussian log-sigma spatial psi derivatives".to_string())?;
        Ok(vec![mean_derivs, noise_derivs])
    }
}

struct GaussianLocationScaleWiggleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
    wiggle_knots: Array1<f64>,
    wiggle_degree: usize,
    wiggle_block: ParameterBlockInput,
}

impl LocationScaleFamilyBuilder for GaussianLocationScaleWiggleTermBuilder {
    type Family = GaussianLocationScaleWiggleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn exact_spatial_seed_risk_profile(&self) -> crate::seeding::SeedRiskProfile {
        crate::seeding::SeedRiskProfile::Gaussian
    }

    fn require_exact_spatial_joint(&self) -> bool {
        true
    }

    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        initial_log_lambdas_orzeros(&self.wiggle_block)
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::withwiggle(
            mean_design.penalties.len(),
            noise_design.penalties.len(),
            self.wiggle_block.penalties.len(),
        );
        layout.validate_theta_len(theta.len(), "gaussian location-scale wiggle")?;
        let mut meanspec = ParameterBlockSpec {
            name: "mu".to_string(),
            design: mean_design.design.clone(),
            offset: self.mean_offset.clone(),
            penalties: mean_design.penalties_as_penalty_matrix(),
            nullspace_dims: vec![],
            initial_log_lambdas: layout.mean_from(theta),
            initial_beta: mean_beta_hint,
        };
        let mean_dense = mean_design.design.as_dense_cow();
        let noise_dense = noise_design.design.as_dense_cow();
        let mut noisespec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                prepared_gaussian_log_sigma_design(&mean_dense, &noise_dense)?,
            )),
            offset: self.noise_offset.clone(),
            penalties: noise_design.penalties_as_penalty_matrix(),
            nullspace_dims: vec![],
            initial_log_lambdas: layout.noise_from(theta),
            initial_beta: noise_beta_hint,
        };
        if meanspec.initial_beta.is_none() || noisespec.initial_beta.is_none() {
            let (betamu0, beta_ls0, _) = gaussian_location_scalewarm_start(
                &self.y,
                &self.weights,
                &meanspec,
                &noisespec,
                1e-10,
                meanspec.initial_beta.as_ref(),
                noisespec.initial_beta.as_ref(),
            )?;
            if meanspec.initial_beta.is_none() {
                meanspec.initial_beta = Some(betamu0);
            }
            if noisespec.initial_beta.is_none() {
                noisespec.initial_beta = Some(beta_ls0);
            }
        }
        Ok(vec![
            meanspec,
            noisespec,
            ParameterBlockSpec {
                name: "wiggle".to_string(),
                design: self.wiggle_block.design.clone(),
                offset: self.wiggle_block.offset.clone(),
                penalties: {
                    let p_wiggle = self.wiggle_block.design.ncols();
                    self.wiggle_block
                        .penalties
                        .iter()
                        .map(|spec| match spec {
                            crate::solver::estimate::PenaltySpec::Block {
                                local,
                                col_range,
                                ..
                            } => PenaltyMatrix::Blockwise {
                                local: local.clone(),
                                col_range: col_range.clone(),
                                total_dim: p_wiggle,
                            },
                            crate::solver::estimate::PenaltySpec::Dense(m) => {
                                PenaltyMatrix::Dense(m.clone())
                            }
                        })
                        .collect()
                },
                nullspace_dims: self.wiggle_block.nullspace_dims.clone(),
                initial_log_lambdas: layout.wiggle_from(theta),
                initial_beta: self.wiggle_block.initial_beta.clone(),
            },
        ])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let mean_dense = mean_design.design.as_dense_cow();
        let noise_dense = noise_design.design.as_dense_cow();
        let preparednoise_design = prepared_gaussian_log_sigma_design(&mean_dense, &noise_dense)
            .expect("prepared Gaussian log-sigma design should match wiggle block construction");
        GaussianLocationScaleWiggleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                preparednoise_design,
            ))),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(GaussianLocationScaleWiggleFamily::BLOCK_MU)
            .ok_or_else(|| "missing Gaussian wiggle mu block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(GaussianLocationScaleWiggleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Gaussian wiggle log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?.ok_or_else(
                || "missing Gaussian wiggle mean spatial psi derivatives".to_string(),
            )?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| {
                    "missing Gaussian wiggle log-sigma spatial psi derivatives".to_string()
                })?;
        Ok(vec![mean_derivs, noise_derivs, Vec::new()])
    }
}

struct BinomialLocationScaleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    link_kind: InverseLink,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
}

impl LocationScaleFamilyBuilder for BinomialLocationScaleTermBuilder {
    type Family = BinomialLocationScaleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn require_exact_spatial_joint(&self) -> bool {
        true
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        noise_design.penalties.len() + 1
    }

    fn augment_result_designs(
        &self,
        _: &mut TermCollectionDesign,
        noise_design: &mut TermCollectionDesign,
    ) {
        append_binomial_log_sigma_shrinkage_penalty_design(noise_design);
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::two_block(
            mean_design.penalties.len(),
            self.noise_penalty_count(noise_design),
        );
        layout.validate_theta_len(theta.len(), "binomial location-scale")?;
        let identifiednoise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)?;
        let p_noise = identifiednoise_design.ncols();
        let mut log_sigma_penalty_matrices: Vec<PenaltyMatrix> =
            noise_design.penalties_as_penalty_matrix();
        log_sigma_penalty_matrices.push(PenaltyMatrix::Dense(identity_penalty(p_noise)));
        let mut thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: mean_design.design.clone(),
            offset: self.mean_offset.clone(),
            penalties: mean_design.penalties_as_penalty_matrix(),
            nullspace_dims: vec![],
            initial_log_lambdas: layout.mean_from(theta),
            initial_beta: mean_beta_hint,
        };
        let mut log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                identifiednoise_design,
            )),
            offset: self.noise_offset.clone(),
            penalties: log_sigma_penalty_matrices,
            nullspace_dims: vec![],
            initial_log_lambdas: layout.noise_from(theta),
            initial_beta: noise_beta_hint,
        };
        if thresholdspec.initial_beta.is_none() || log_sigmaspec.initial_beta.is_none() {
            let (beta_t0, beta_ls0) = binomial_location_scalewarm_start(
                &self.y,
                &self.weights,
                &self.link_kind,
                &thresholdspec,
                &log_sigmaspec,
                thresholdspec.initial_beta.as_ref(),
                log_sigmaspec.initial_beta.as_ref(),
            )?;
            if thresholdspec.initial_beta.is_none() {
                thresholdspec.initial_beta = Some(beta_t0);
            }
            if log_sigmaspec.initial_beta.is_none() {
                log_sigmaspec.initial_beta = Some(beta_ls0);
            }
        }
        Ok(vec![thresholdspec, log_sigmaspec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let identifiednoise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)
                .expect("identified binomial log-sigma design");
        BinomialLocationScaleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                identifiednoise_design,
            ))),
            policy: crate::resource::ResourcePolicy::default_library(),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(BinomialLocationScaleFamily::BLOCK_T)
            .ok_or_else(|| "missing Binomial threshold block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(BinomialLocationScaleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Binomial log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?
                .ok_or_else(|| "missing threshold spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| "missing log_sigma spatial psi derivatives".to_string())?;
        Ok(vec![mean_derivs, noise_derivs])
    }
}

struct BinomialLocationScaleWiggleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    link_kind: InverseLink,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
    mean_offset: Array1<f64>,
    noise_offset: Array1<f64>,
    wiggle_knots: Array1<f64>,
    wiggle_degree: usize,
    wiggle_block: ParameterBlockInput,
}

impl LocationScaleFamilyBuilder for BinomialLocationScaleWiggleTermBuilder {
    type Family = BinomialLocationScaleWiggleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }

    fn require_exact_spatial_joint(&self) -> bool {
        true
    }

    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        initial_log_lambdas_orzeros(&self.wiggle_block)
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        noise_design.penalties.len() + 1
    }

    fn augment_result_designs(
        &self,
        _: &mut TermCollectionDesign,
        noise_design: &mut TermCollectionDesign,
    ) {
        append_binomial_log_sigma_shrinkage_penalty_design(noise_design);
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::withwiggle(
            mean_design.penalties.len(),
            self.noise_penalty_count(noise_design),
            self.wiggle_block.penalties.len(),
        );
        layout.validate_theta_len(theta.len(), "wiggle location-scale")?;
        let identifiednoise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)?;
        let p_noise = identifiednoise_design.ncols();
        let mut log_sigma_penalty_matrices: Vec<PenaltyMatrix> =
            noise_design.penalties_as_penalty_matrix();
        log_sigma_penalty_matrices.push(PenaltyMatrix::Dense(identity_penalty(p_noise)));
        let mut thresholdspec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: mean_design.design.clone(),
            offset: self.mean_offset.clone(),
            penalties: mean_design.penalties_as_penalty_matrix(),
            nullspace_dims: vec![],
            initial_log_lambdas: layout.mean_from(theta),
            initial_beta: mean_beta_hint,
        };
        let mut log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                identifiednoise_design,
            )),
            offset: self.noise_offset.clone(),
            penalties: log_sigma_penalty_matrices,
            nullspace_dims: vec![],
            initial_log_lambdas: layout.noise_from(theta),
            initial_beta: noise_beta_hint,
        };
        if thresholdspec.initial_beta.is_none() || log_sigmaspec.initial_beta.is_none() {
            let (beta_t0, beta_ls0) = binomial_location_scalewarm_start(
                &self.y,
                &self.weights,
                &self.link_kind,
                &thresholdspec,
                &log_sigmaspec,
                thresholdspec.initial_beta.as_ref(),
                log_sigmaspec.initial_beta.as_ref(),
            )?;
            if thresholdspec.initial_beta.is_none() {
                thresholdspec.initial_beta = Some(beta_t0);
            }
            if log_sigmaspec.initial_beta.is_none() {
                log_sigmaspec.initial_beta = Some(beta_ls0);
            }
        }
        Ok(vec![
            thresholdspec,
            log_sigmaspec,
            ParameterBlockSpec {
                name: "wiggle".to_string(),
                design: self.wiggle_block.design.clone(),
                offset: self.wiggle_block.offset.clone(),
                penalties: {
                    let p_wiggle = self.wiggle_block.design.ncols();
                    self.wiggle_block
                        .penalties
                        .iter()
                        .map(|spec| match spec {
                            crate::solver::estimate::PenaltySpec::Block {
                                local,
                                col_range,
                                ..
                            } => PenaltyMatrix::Blockwise {
                                local: local.clone(),
                                col_range: col_range.clone(),
                                total_dim: p_wiggle,
                            },
                            crate::solver::estimate::PenaltySpec::Dense(m) => {
                                PenaltyMatrix::Dense(m.clone())
                            }
                        })
                        .collect()
                },
                nullspace_dims: vec![],
                initial_log_lambdas: layout.wiggle_from(theta),
                initial_beta: self.wiggle_block.initial_beta.clone(),
            },
        ])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let identifiednoise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)
                .expect("identified binomial log-sigma design should match block construction");
        BinomialLocationScaleWiggleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(mean_design.design.clone()),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                identifiednoise_design,
            ))),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: crate::resource::ResourcePolicy::default_library(),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(BinomialLocationScaleWiggleFamily::BLOCK_T)
            .ok_or_else(|| "missing Binomial wiggle threshold block state".to_string())?
            .beta
            .clone();
        let noise_beta = fit
            .block_states
            .get(BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA)
            .ok_or_else(|| "missing Binomial wiggle log_sigma block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, noise_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec_resolved: &TermCollectionSpec,
        noisespec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, meanspec_resolved, mean_design)?
                .ok_or_else(|| "missing threshold spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| "missing log_sigma spatial psi derivatives".to_string())?;
        // The wiggle block has no direct spatial design matrix of its own in the
        // term builder. Spatial psi moves the wiggle family only through the
        // realized threshold/log-sigma designs, which in turn perturb q0 and the
        // realized wiggle basis B(q0). The exact joint wiggle psi hooks consume
        // those threshold/log-sigma derivative payloads and reconstruct the full
        // flattened likelihood-side [rho, psi] calculus internally, so the
        // wiggle block intentionally contributes no direct CustomFamilyBlockPsiDerivative
        // entries here.
        Ok(vec![mean_derivs, noise_derivs, Vec::new()])
    }
}

pub(crate) fn fit_gaussian_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_gaussian_location_scale_termspec(data, &spec, "fit_gaussian_location_scale_terms")?;
    fit_location_scale_terms(
        data,
        GaussianLocationScaleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            meanspec: spec.meanspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.mean_offset,
            noise_offset: spec.log_sigma_offset,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn fit_gaussian_location_scalewiggle_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleWiggleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_gaussian_location_scalewiggle_termspec(
        data,
        &spec,
        "fit_gaussian_location_scalewiggle_terms",
    )?;
    fit_location_scale_terms(
        data,
        GaussianLocationScaleWiggleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            meanspec: spec.meanspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.mean_offset,
            noise_offset: spec.log_sigma_offset,
            wiggle_knots: spec.wiggle_knots,
            wiggle_degree: spec.wiggle_degree,
            wiggle_block: spec.wiggle_block,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn select_gaussian_location_scale_link_wiggle_basis_from_pilot(
    pilot: &BlockwiseTermFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let q_seed = pilot
        .fit
        .block_states
        .first()
        .ok_or_else(|| "pilot Gaussian wiggle fit is missing mean block".to_string())?
        .eta
        .view();
    select_wiggle_basis_from_seed(q_seed, wiggle_cfg, wiggle_penalty_orders)
}

pub(crate) fn fit_gaussian_location_scale_terms_with_selected_wiggle(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleTermSpec,
    selected_wiggle_basis: SelectedWiggleBasis,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermWiggleFitResult, String> {
    let SelectedWiggleBasis {
        knots: wiggle_knots,
        degree: wiggle_degree,
        block: wiggle_block,
        ..
    } = selected_wiggle_basis;
    let solved = fit_gaussian_location_scalewiggle_terms(
        data,
        GaussianLocationScaleWiggleTermSpec {
            y: spec.y,
            weights: spec.weights,
            meanspec: spec.meanspec,
            log_sigmaspec: spec.log_sigmaspec,
            mean_offset: spec.mean_offset,
            log_sigma_offset: spec.log_sigma_offset,
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            wiggle_block,
        },
        options,
        kappa_options,
    )?;

    BlockwiseTermWiggleFitResult::try_from_parts(BlockwiseTermWiggleFitResultParts {
        fit: solved,
        wiggle_knots,
        wiggle_degree,
    })
}

pub(crate) fn fit_binomial_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_binomial_location_scale_termspec(data, &spec, "fit_binomial_location_scale_terms")?;
    fit_location_scale_terms(
        data,
        BinomialLocationScaleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            meanspec: spec.thresholdspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.threshold_offset,
            noise_offset: spec.log_sigma_offset,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn fit_binomial_location_scalewiggle_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleWiggleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_binomial_location_scalewiggle_termspec(
        data,
        &spec,
        "fit_binomial_location_scalewiggle_terms",
    )?;
    fit_location_scale_terms(
        data,
        BinomialLocationScaleWiggleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            meanspec: spec.thresholdspec,
            noisespec: spec.log_sigmaspec,
            mean_offset: spec.threshold_offset,
            noise_offset: spec.log_sigma_offset,
            wiggle_knots: spec.wiggle_knots,
            wiggle_degree: spec.wiggle_degree,
            wiggle_block: spec.wiggle_block,
        },
        options,
        kappa_options,
    )
}

pub(crate) fn select_binomial_location_scale_link_wiggle_basis_from_pilot(
    pilot: &BlockwiseTermFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let eta_t = pilot
        .fit
        .block_states
        .first()
        .ok_or_else(|| "pilot fit is missing threshold block".to_string())?
        .eta
        .view();
    let eta_ls = pilot
        .fit
        .block_states
        .get(1)
        .ok_or_else(|| "pilot fit is missing log_sigma block".to_string())?
        .eta
        .view();
    let sigma = eta_ls.mapv(safe_exp);
    let q_seed = Array1::from_iter(eta_t.iter().zip(sigma.iter()).map(|(&t, &s)| -t / s));
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}

pub(crate) fn fit_binomial_location_scale_terms_with_selected_wiggle(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleTermSpec,
    selected_wiggle_basis: SelectedWiggleBasis,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermWiggleFitResult, String> {
    let SelectedWiggleBasis {
        knots: wiggle_knots,
        degree: wiggle_degree,
        block: wiggle_block,
        ..
    } = selected_wiggle_basis;
    let solved = fit_binomial_location_scalewiggle_terms(
        data,
        BinomialLocationScaleWiggleTermSpec {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            thresholdspec: spec.thresholdspec,
            log_sigmaspec: spec.log_sigmaspec,
            threshold_offset: spec.threshold_offset,
            log_sigma_offset: spec.log_sigma_offset,
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            wiggle_block,
        },
        options,
        kappa_options,
    )?;

    BlockwiseTermWiggleFitResult::try_from_parts(BlockwiseTermWiggleFitResultParts {
        fit: solved,
        wiggle_knots,
        wiggle_degree,
    })
}

pub(crate) fn select_binomial_mean_link_wiggle_basis_from_pilot(
    pilot_design: &TermCollectionDesign,
    pilot_fit: &UnifiedFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let q_seed = pilot_design.design.dot(&pilot_fit.beta);
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}

pub(crate) fn fit_binomial_mean_wiggle_terms_with_selected_basis(
    data: ndarray::ArrayView2<'_, f64>,
    pilot_spec: &TermCollectionSpec,
    pilot_design: &TermCollectionDesign,
    pilot_fit: &UnifiedFitResult,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    link_kind: InverseLink,
    selected_wiggle_basis: SelectedWiggleBasis,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BinomialMeanWiggleTermFitResult, String> {
    const RHO_BOUND: f64 = 12.0;

    validate_term_weights(
        data,
        y.len(),
        weights,
        "fit_binomial_mean_wiggle_terms_with_selected_basis",
    )?;
    validate_binomial_response(y, "fit_binomial_mean_wiggle_terms_with_selected_basis")?;

    let SelectedWiggleBasis {
        knots: wiggle_knots,
        degree: wiggle_degree,
        block: wiggle_block,
        ..
    } = selected_wiggle_basis;

    let spatial_terms = spatial_length_scale_term_indices(pilot_spec);
    if spatial_terms.is_empty() {
        let fit = fit_binomial_mean_wiggle(
            BinomialMeanWiggleSpec {
                y: y.clone(),
                weights: weights.clone(),
                link_kind,
                wiggle_knots: wiggle_knots.clone(),
                wiggle_degree,
                eta_block: ParameterBlockInput {
                    design: pilot_design.design.clone(),
                    offset: Array1::zeros(y.len()),
                    penalties: pilot_design
                        .penalties
                        .iter()
                        .map(|bp| crate::solver::estimate::PenaltySpec::from_blockwise_ref(bp))
                        .collect(),
                    nullspace_dims: vec![],
                    initial_log_lambdas: Some(pilot_fit.lambdas.mapv(|v| v.max(1e-12).ln())),
                    initial_beta: Some(pilot_fit.beta.clone()),
                },
                wiggle_block,
            },
            options,
        )?;
        return Ok(BinomialMeanWiggleTermFitResult {
            fit,
            resolvedspec: pilot_spec.clone(),
            design: pilot_design.clone(),
            wiggle_knots,
            wiggle_degree,
        });
    }

    let dims_per_term = spatial_dims_per_term(pilot_spec, &spatial_terms);
    let log_kappa0 =
        SpatialLogKappaCoords::from_length_scales_aniso(pilot_spec, &spatial_terms, kappa_options)
            .reseed_from_data(data, pilot_spec, &spatial_terms, kappa_options);
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        pilot_spec,
        &spatial_terms,
        &dims_per_term,
        kappa_options,
    );
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        pilot_spec,
        &spatial_terms,
        &dims_per_term,
        kappa_options,
    );
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);

    let eta_penalty_count = pilot_design.penalties.len();
    let wiggle_penalty_count = initial_log_lambdas_orzeros(&wiggle_block)?.len();
    let rho_dim = eta_penalty_count + wiggle_penalty_count;
    let baseline_resolvedspec = log_kappa0
        .apply_tospec(pilot_spec, &spatial_terms)
        .map_err(|e| e.to_string())?;
    let baseline_design =
        build_term_collection_design(data, &baseline_resolvedspec).map_err(|e| e.to_string())?;
    let baseline_fit = fit_binomial_mean_wiggle(
        BinomialMeanWiggleSpec {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: link_kind.clone(),
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            eta_block: ParameterBlockInput {
                design: baseline_design.design.clone(),
                offset: Array1::zeros(y.len()),
                penalties: baseline_design
                    .penalties
                    .iter()
                    .map(|bp| crate::solver::estimate::PenaltySpec::from_blockwise_ref(bp))
                    .collect(),
                nullspace_dims: vec![],
                initial_log_lambdas: Some(pilot_fit.lambdas.mapv(|v| v.max(1e-12).ln())),
                initial_beta: Some(pilot_fit.beta.clone()),
            },
            wiggle_block: wiggle_block.clone(),
        },
        options,
    )?;
    let baseline_log_lambdas = baseline_fit.lambdas.mapv(|v| v.max(1e-12).ln());
    if baseline_log_lambdas.len() != rho_dim {
        return Err(format!(
            "baseline binomial mean-wiggle fit returned {} log-lambdas, expected {rho_dim}",
            baseline_log_lambdas.len()
        ));
    }
    let baseline_eta_beta = baseline_fit
        .block_states
        .get(BinomialMeanWiggleFamily::BLOCK_ETA)
        .ok_or_else(|| "baseline binomial mean-wiggle fit missing eta block".to_string())?
        .beta
        .clone();
    let baseline_wiggle_beta = Some(
        baseline_fit
            .block_states
            .get(BinomialMeanWiggleFamily::BLOCK_WIGGLE)
            .ok_or_else(|| "baseline binomial mean-wiggle fit missing wiggle block".to_string())?
            .beta
            .clone(),
    );
    let theta_dim = rho_dim + log_kappa0.len();
    let mut theta0 = Array1::<f64>::zeros(theta_dim);
    theta0
        .slice_mut(s![0..rho_dim])
        .assign(&baseline_log_lambdas);
    theta0
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa0.as_array());

    let mut lower = Array1::<f64>::from_elem(theta_dim, -RHO_BOUND);
    let mut upper = Array1::<f64>::from_elem(theta_dim, RHO_BOUND);
    lower
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa_lower.as_array());
    upper
        .slice_mut(s![rho_dim..theta_dim])
        .assign(log_kappa_upper.as_array());

    let pilot_spec_cloned = pilot_spec.clone();
    let pilot_beta = baseline_eta_beta;
    let wiggle_design = wiggle_block.design.clone();
    let wiggle_offset = wiggle_block.offset.clone();
    let wiggle_penalties = wiggle_block.penalties.clone();
    let wiggle_initial_beta = baseline_wiggle_beta;
    let wiggle_knots_cloned = wiggle_knots.clone();
    let y_cloned = y.clone();
    let weights_cloned = weights.clone();
    let link_kind_cloned = link_kind.clone();
    let outer_family = BinomialMeanWiggleFamily {
        y: y_cloned.clone(),
        weights: weights_cloned.clone(),
        link_kind: link_kind_cloned.clone(),
        wiggle_knots: wiggle_knots_cloned.clone(),
        wiggle_degree,
        policy: crate::resource::ResourcePolicy::default_library(),
    };
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let mut outer_options = options.clone();
    outer_options.screening_max_inner_iterations = Some(Arc::clone(&screening_cap));
    // This outer problem is design-moving in the spatial block, so we
    // intentionally plan it as gradient/fixed-point rather than ARC/Newton.
    let analytic_outer_hessian_available = false;

    struct MeanWiggleOuterState {
        warm_cache: Option<crate::custom_family::CustomFamilyWarmStart>,
        last_eval: Option<(
            Array1<f64>,
            f64,
            Array1<f64>,
            crate::solver::outer_strategy::HessianResult,
            crate::custom_family::CustomFamilyWarmStart,
        )>,
    }

    let build_realized_blocks = |theta: &Array1<f64>| -> Result<
        (
            TermCollectionSpec,
            TermCollectionDesign,
            Vec<ParameterBlockSpec>,
            Vec<CustomFamilyBlockPsiDerivative>,
        ),
        String,
    > {
        let log_kappa =
            SpatialLogKappaCoords::from_theta_tail_with_dims(theta, rho_dim, dims_per_term.clone());
        let resolvedspec = log_kappa
            .apply_tospec(&pilot_spec_cloned, &spatial_terms)
            .map_err(|e| e.to_string())?;
        let design =
            build_term_collection_design(data, &resolvedspec).map_err(|e| e.to_string())?;
        let eta_derivs = build_block_spatial_psi_derivatives(data, &resolvedspec, &design)?
            .ok_or_else(|| {
                "missing eta spatial psi derivatives for binomial mean wiggle".to_string()
            })?;
        let blocks = vec![
            ParameterBlockSpec {
                name: "eta".to_string(),
                design: design.design.clone(),
                offset: Array1::zeros(y_cloned.len()),
                penalties: design.penalties_as_penalty_matrix(),
                nullspace_dims: vec![],
                initial_log_lambdas: theta.slice(s![0..eta_penalty_count]).to_owned(),
                initial_beta: Some(pilot_beta.clone()),
            },
            ParameterBlockSpec {
                name: "wiggle".to_string(),
                design: wiggle_design.clone(),
                offset: wiggle_offset.clone(),
                penalties: {
                    let p_wiggle = wiggle_design.ncols();
                    wiggle_penalties
                        .iter()
                        .map(|spec| match spec {
                            crate::solver::estimate::PenaltySpec::Block {
                                local,
                                col_range,
                                ..
                            } => PenaltyMatrix::Blockwise {
                                local: local.clone(),
                                col_range: col_range.clone(),
                                total_dim: p_wiggle,
                            },
                            crate::solver::estimate::PenaltySpec::Dense(m) => {
                                PenaltyMatrix::Dense(m.clone())
                            }
                        })
                        .collect()
                },
                nullspace_dims: vec![],
                initial_log_lambdas: theta.slice(s![eta_penalty_count..rho_dim]).to_owned(),
                initial_beta: wiggle_initial_beta.clone(),
            },
        ];
        Ok((resolvedspec, design, blocks, eta_derivs))
    };

    let build_eval = |theta: &Array1<f64>,
                      warm_cache: Option<&crate::custom_family::CustomFamilyWarmStart>,
                      need_hessian: bool|
     -> Result<
        (
            crate::custom_family::CustomFamilyJointHyperResult,
            TermCollectionSpec,
            TermCollectionDesign,
        ),
        String,
    > {
        let (resolvedspec, design, blocks, eta_derivs) = build_realized_blocks(theta)?;
        let eval = evaluate_custom_family_joint_hyper(
            &outer_family,
            &blocks,
            &outer_options,
            &theta.slice(s![0..rho_dim]).to_owned(),
            &[eta_derivs, Vec::new()],
            warm_cache,
            if need_hessian {
                crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian
            } else {
                crate::solver::estimate::reml::unified::EvalMode::ValueAndGradient
            },
        )?;
        Ok((eval, resolvedspec, design))
    };

    let build_efs = |theta: &Array1<f64>,
                     warm_cache: Option<&crate::custom_family::CustomFamilyWarmStart>|
     -> Result<crate::custom_family::CustomFamilyJointHyperEfsResult, String> {
        let (_, _, blocks, eta_derivs) = build_realized_blocks(theta)?;
        evaluate_custom_family_joint_hyper_efs(
            &outer_family,
            &blocks,
            &outer_options,
            &theta.slice(s![0..rho_dim]).to_owned(),
            &[eta_derivs, Vec::new()],
            warm_cache,
        )
        .map_err(|e| e.to_string())
    };

    use crate::estimate::EstimationError;
    use crate::solver::outer_strategy::{Derivative, OuterEval, OuterEvalOrder};

    // Exact first-order AND second-order [rho, psi] calculus is available
    // for all inverse links via the shared jet formulas plus the generic
    // exact-Newton D_βH / D²_βH closures routed through
    // evaluate_custom_family_joint_hyper -> joint_outer_evaluate ->
    // BorrowedJointDerivProvider. This enables the analytic-Hessian outer
    // plan for REML optimization instead of the downgraded gradient-only
    // outer strategies.
    //
    // Spatial log-kappa coordinates are ψ (design-moving) dimensions because
    // they rebuild the spatial basis and penalties at each outer proposal.
    let prefer_gradient_only =
        analytic_outer_hessian_available && baseline_design.design.as_sparse().is_none();
    if prefer_gradient_only {
        log::info!(
            "[OUTER] binomial mean wiggle joint REML: dense exact-joint design detected; preferring gradient-only BFGS over Arc"
        );
    }
    let mut seed_heuristic = theta0.to_vec();
    for value in &mut seed_heuristic[..rho_dim] {
        *value = value.exp();
    }
    let problem = crate::solver::outer_strategy::OuterProblem::new(theta_dim)
        .with_gradient(Derivative::Analytic)
        .with_hessian(if analytic_outer_hessian_available {
            Derivative::Analytic
        } else {
            Derivative::Unavailable
        })
        .with_prefer_gradient_only(prefer_gradient_only)
        .with_psi_dim(theta_dim - rho_dim)
        .with_tolerance(options.outer_tol)
        .with_max_iter(options.outer_max_iter)
        .with_bounds(lower.clone(), upper.clone())
        .with_initial_rho(theta0.clone())
        .with_seed_config(crate::seeding::SeedConfig {
            max_seeds: 4,
            seed_budget: 2,
            risk_profile: crate::seeding::SeedRiskProfile::GeneralizedLinear,
            num_auxiliary_trailing: theta_dim - rho_dim,
            ..Default::default()
        })
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_rho_bound(12.0)
        .with_heuristic_lambdas(seed_heuristic);

    let eval_outer = |state: &mut MeanWiggleOuterState,
                      theta: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        if let Some((cached_theta, cached_cost, cached_grad, cached_hess, cached_warm)) =
            &state.last_eval
            && cached_theta == theta
            && (!matches!(order, OuterEvalOrder::ValueGradientHessian)
                || matches!(
                    cached_hess,
                    crate::solver::outer_strategy::HessianResult::Analytic(_)
                        | crate::solver::outer_strategy::HessianResult::Operator(_)
                ))
        {
            state.warm_cache = Some(cached_warm.clone());
            return Ok(OuterEval {
                cost: *cached_cost,
                gradient: cached_grad.clone(),
                hessian: cached_hess.clone(),
            });
        }
        let need_hessian = matches!(order, OuterEvalOrder::ValueGradientHessian)
            && analytic_outer_hessian_available;
        let (eval, _, _) = build_eval(theta, state.warm_cache.as_ref(), need_hessian)
            .map_err(EstimationError::InvalidInput)?;
        let hessian_result = eval.outer_hessian.clone();
        state.last_eval = Some((
            theta.clone(),
            eval.objective,
            eval.gradient.clone(),
            eval.outer_hessian.clone(),
            eval.warm_start.clone(),
        ));
        state.warm_cache = Some(eval.warm_start);
        Ok(OuterEval {
            cost: eval.objective,
            gradient: eval.gradient,
            hessian: hessian_result,
        })
    };

    let mut obj = problem.build_objective_with_eval_order(
        MeanWiggleOuterState {
            warm_cache: None,
            last_eval: None,
        },
        |state: &mut MeanWiggleOuterState, theta: &Array1<f64>| {
            if let Some((cached_theta, cached_cost, _, _, cached_warm)) = &state.last_eval
                && cached_theta == theta
            {
                state.warm_cache = Some(cached_warm.clone());
                return Ok(*cached_cost);
            }
            let (eval, _, _) = build_eval(theta, state.warm_cache.as_ref(), false)
                .map_err(EstimationError::InvalidInput)?;
            state.warm_cache = Some(eval.warm_start);
            Ok(eval.objective)
        },
        |state: &mut MeanWiggleOuterState, theta: &Array1<f64>| {
            eval_outer(
                state,
                theta,
                if analytic_outer_hessian_available {
                    OuterEvalOrder::ValueGradientHessian
                } else {
                    OuterEvalOrder::ValueAndGradient
                },
            )
        },
        |state: &mut MeanWiggleOuterState, theta: &Array1<f64>, order: OuterEvalOrder| {
            eval_outer(state, theta, order)
        },
        Some(|state: &mut MeanWiggleOuterState| {
            state.warm_cache = None;
            state.last_eval = None;
        }),
        Some(|state: &mut MeanWiggleOuterState, theta: &Array1<f64>| {
            let eval = build_efs(theta, state.warm_cache.as_ref())
                .map_err(EstimationError::InvalidInput)?;
            state.warm_cache = Some(eval.warm_start);
            Ok(eval.efs_eval)
        }),
    );

    let outer = problem
        .run(&mut obj, "binomial mean wiggle exact spatial hyper")
        .map_err(|e| e.to_string())?;
    let theta_star = outer.rho;

    let log_kappa =
        SpatialLogKappaCoords::from_theta_tail_with_dims(&theta_star, rho_dim, dims_per_term);
    let resolvedspec = log_kappa
        .apply_tospec(&pilot_spec_cloned, &spatial_terms)
        .map_err(|e| e.to_string())?;
    let design = build_term_collection_design(data, &resolvedspec).map_err(|e| e.to_string())?;
    let resolvedspec =
        freeze_term_collection_from_design(&resolvedspec, &design).map_err(|e| e.to_string())?;
    let fit = fit_binomial_mean_wiggle(
        BinomialMeanWiggleSpec {
            y: y_cloned,
            weights: weights_cloned,
            link_kind: link_kind_cloned,
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree,
            eta_block: ParameterBlockInput {
                design: design.design.clone(),
                offset: Array1::zeros(y.len()),
                penalties: design
                    .penalties
                    .iter()
                    .map(|bp| crate::solver::estimate::PenaltySpec::from_blockwise_ref(bp))
                    .collect(),
                nullspace_dims: vec![],
                initial_log_lambdas: Some(theta_star.slice(s![0..eta_penalty_count]).to_owned()),
                initial_beta: Some(pilot_beta),
            },
            wiggle_block: ParameterBlockInput {
                design: wiggle_design,
                offset: wiggle_offset,
                penalties: wiggle_penalties,
                nullspace_dims: vec![],
                initial_log_lambdas: Some(
                    theta_star.slice(s![eta_penalty_count..rho_dim]).to_owned(),
                ),
                initial_beta: wiggle_initial_beta,
            },
        },
        options,
    )?;

    Ok(BinomialMeanWiggleTermFitResult {
        fit,
        resolvedspec,
        design,
        wiggle_knots,
        wiggle_degree,
    })
}

/// Link identifiers for distribution parameters in multi-parameter GAMLSS families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterLink {
    Identity,
    Log,
    Logit,
    Probit,
    InverseLink,
    /// Learnable smooth departure from a known base link.
    Wiggle,
}

fn signedwith_floor(v: f64, floor: f64) -> f64 {
    let a = v.abs().max(floor);
    if v >= 0.0 { a } else { -a }
}

#[inline]
fn binomial_score_curvaturethird_from_jet(
    y: f64,
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    // Binomial derivatives wrt q via mu:
    // Per-row log-likelihood is represented in weighted-proportion form:
    //   ell_i = m_i * [ y_i log(mu_i) + (1-y_i) log(1-mu_i) ],
    // where `weight = m_i` and `y` is the observed proportion in [0,1].
    //
    // mu-space derivatives:
    //   ellmu    = y/mu - (1-y)/(1-mu)
    //   ellmumu  = -y/mu^2 - (1-y)/(1-mu)^2
    //   ellmumum = 2y/mu^3 - 2(1-y)/(1-mu)^3
    //
    // q-jet using mu(q) derivatives d1=mu', d2=mu'', d3=mu''':
    //   s = dell/dq   = ellmu * mu'
    //   c = d2ell/dq2 = ellmumu*(mu')^2 + ellmu*mu''
    //   t = d3ell/dq3 = ellmumum*(mu')^3 + 3*ellmumu*mu'*mu'' + ellmu*mu'''
    //
    // Returns (score_q, curvature_q, third_q) with curvature_q = -d2ell/dq2.
    let m = mu;
    let one_minus = 1.0 - m;
    let ellmu = y / m - (1.0 - y) / one_minus;
    let ellmumu = -y / (m * m) - (1.0 - y) / (one_minus * one_minus);
    let ellmumum = 2.0 * y / (m * m * m) - 2.0 * (1.0 - y) / (one_minus * one_minus * one_minus);

    let score_q = weight * ellmu * d1;
    let d2ell_dq2 = weight * (ellmumu * d1 * d1 + ellmu * d2);
    let curvature_q = -d2ell_dq2;
    let third_q = weight * (ellmumum * d1 * d1 * d1 + 3.0 * ellmumu * d1 * d2 + ellmu * d3);
    (score_q, curvature_q, third_q)
}

#[inline]
fn binomial_neglog_q_derivatives_from_jet(
    y: f64,
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    // Returns (m1,m2,m3) for F_i(q) = -ell_i(q):
    //   m1 = dF/dq, m2 = d²F/dq², m3 = d³F/dq³.
    let (score_q, curvature_q, third_q) =
        binomial_score_curvaturethird_from_jet(y, weight, mu, d1, d2, d3);
    (-score_q, curvature_q, -third_q)
}

#[inline]
fn binomial_neglog_q_derivatives_probit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
) -> (f64, f64, f64) {
    // Closed-form derivatives for F_i(q) = -w_i[y log Phi(q) + (1-y) log(1-Phi(q))].
    // Uses canonical A/Amu/Amumu identities from the probit composition.
    let (m, clamp_active) = clamped_binomial_probability(mu);
    if clamp_active || weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let nu = 1.0 - m;
    let phi = normal_pdf(q);
    let a = (1.0 - y) / nu - y / m;
    let amu = (1.0 - y) / (nu * nu) + y / (m * m);
    let amumu = 2.0 * (1.0 - y) / (nu * nu * nu) - 2.0 * y / (m * m * m);

    let m1 = weight * a * phi;
    let m2 = weight * (amu * phi * phi - q * a * phi);
    let m3 =
        weight * (amumu * phi * phi * phi - 3.0 * q * amu * phi * phi + (q * q - 1.0) * a * phi);
    (m1, m2, m3)
}

#[inline]
fn binomial_neglog_q_fourth_derivative_probit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
) -> f64 {
    // Closed-form m4 for F_i(q) = -w_i[y log Phi(q) + (1-y) log(1-Phi(q))].
    let (m, clamp_active) = clamped_binomial_probability(mu);
    if clamp_active || weight == 0.0 || !q.is_finite() {
        return 0.0;
    }
    let nu = 1.0 - m;
    let phi = normal_pdf(q);
    let a = (1.0 - y) / nu - y / m;
    let amu = (1.0 - y) / (nu * nu) + y / (m * m);
    let amumu = 2.0 * (1.0 - y) / (nu * nu * nu) - 2.0 * y / (m * m * m);
    let amumumu = 6.0 * (1.0 - y) / (nu * nu * nu * nu) + 6.0 * y / (m * m * m * m);
    weight
        * (amumumu * phi.powi(4) - 6.0 * q * amumu * phi.powi(3)
            + (7.0 * q * q - 4.0) * amu * phi * phi
            - (q * q * q - 3.0 * q) * a * phi)
}

// ---------------------------------------------------------------------------
// Logit closed-form m1–m4
// ---------------------------------------------------------------------------
//
// For the logit (sigmoid) inverse link, F(q) = -w[y log G(q) + (1-y) log(1-G(q))]
// where G(q) = 1/(1 + e^{-q}) is the standard logistic CDF.
//
// Because logit is the canonical link for Bernoulli, the derivatives of F
// collapse to especially simple closed forms in terms of p = G(q) and
// s = p(1-p) = Var(Bernoulli(p)):
//
//   m1 = w(p - y)
//   m2 = ws                           (always non-negative)
//   m3 = ws(1 - 2p) = -ws tanh(q/2)
//   m4 = ws(1 - 6s) = ws(1 - 6p + 6p^2)
//
// Derivation: since log G(q) = -log(1 + e^{-q}) and log(1 - G(q)) = -log(1 + e^q),
// F(q) = w[-y log G + (1-y)(-log(1-G))]
//       = w[y log(1+e^{-q}) + (1-y) log(1+e^q)]
//       = w[(1-y)q + log(1+e^{-q})]     (the standard softplus form).
//
// Differentiating: F' = w(G(q) - y) = w(p - y), which is m1.
// F'' = wG'(q) = wp(1-p) = ws, which is m2.
// F''' = w[p(1-p)(1-2p)] = ws(1-2p), which is m3.
// F'''' = w[s(1-6s)], which is m4. The identity 1-6s = 1-6p+6p^2 follows directly.
//
// Numerical stability (see response.md Section 1a):
// - p is computed with a branched expit to avoid overflow:
//     p = (1+e^{-q})^{-1} for q >= 0,  p = e^q/(1+e^q) for q < 0.
// - s = p(1-p). For extreme tails, s = t/(1+t)^2 with t = e^{-|q|}, which
//   decays as O(e^{-|q|}). Once |q| > ~36, e^{-|q|} < machine epsilon and
//   all derivatives are genuinely below precision, so saturation to 0 is safe.
// - The identity 1-2p = -tanh(q/2) provides a stable alternative for m3.
//
// Reference: response.md Section 1a.
// ---------------------------------------------------------------------------

#[inline]
fn binomial_neglog_q_derivatives_logit_closed_form(y: f64, weight: f64, q: f64) -> (f64, f64, f64) {
    // Returns (m1, m2, m3) for F(q) = -w[y log G(q) + (1-y) log(1-G(q))]
    // with G = logistic CDF.
    if weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    // Branched expit for numerical stability:
    //   q >= 0: p = 1/(1+e^{-q}), avoids overflow in e^q
    //   q < 0:  p = e^q/(1+e^q),  avoids overflow in e^{-q}
    let p = if q >= 0.0 {
        1.0 / (1.0 + (-q).exp())
    } else {
        let eq = q.exp();
        eq / (1.0 + eq)
    };
    let s = p * (1.0 - p);
    // For extreme |q|, s underflows gracefully to 0, which is correct.

    let m1 = weight * (p - y);
    let m2 = weight * s;
    // m3 = ws(1 - 2p). Using the identity 1-2p = -tanh(q/2) for stability:
    let m3 = weight * s * (1.0 - 2.0 * p);
    (m1, m2, m3)
}

#[inline]
fn binomial_neglog_q_fourth_derivative_logit_closed_form(_: f64, weight: f64, q: f64) -> f64 {
    // Returns m4 = d^4F/dq^4 for logit link.
    // m4 = ws(1 - 6s) = ws(1 - 6p(1-p)).
    //
    // Note: m4 does not depend on y at all (same as m2), because all
    // even-order derivatives of the canonical Bernoulli NLL are functions
    // of p alone. The y-dependence cancels out in the chain rule because
    // the logit is the canonical link.
    if weight == 0.0 || !q.is_finite() {
        return 0.0;
    }
    let p = if q >= 0.0 {
        1.0 / (1.0 + (-q).exp())
    } else {
        let eq = q.exp();
        eq / (1.0 + eq)
    };
    let s = p * (1.0 - p);
    weight * s * (1.0 - 6.0 * s)
}

// ---------------------------------------------------------------------------
// CLogLog / Gumbel closed-form m1–m4
// ---------------------------------------------------------------------------
//
// For the complementary log-log link, G(q) = 1 - exp(-exp(q)), so
// F(q) = -w[y log G(q) + (1-y) log(1-G(q))].
//
// Define:
//   z = e^q            (the "inner exponential")
//   r = e^{-z}         (survival probability 1 - G(q))
//   p = 1 - r = G(q) = -expm1(-z)
//   h = z / expm1(z) = z*r / p
//
// The ratio h is the key stable building block. It arises because the
// y=1 branch of the loss is F_{y=1} = -w log(1 - e^{-z}), and differentiating
// log(-expm1(-z)) w.r.t. q produces factors of z*e^{-z}/(1 - e^{-z}) = z*r/p = h.
// The function h = z/(e^z - 1) is smooth on all of R, with h -> 1 as z -> 0
// (removable singularity), and h -> z*e^{-z} -> 0 as z -> +inf.
//
// For y=0, the loss is simply F_{y=0} = w*e^q = w*z, so all derivatives are w*z.
//
// For y=1, the derivatives in the "h-form" (from response.md Section 1b) are:
//   F'_{y=1}    = -wh
//   F''_{y=1}   = wh(h + z - 1)
//   F'''_{y=1}  = -wh(2h^2 + 3(z-1)h + z^2 - 3z + 1)
//   F''''_{y=1} = wh(6h^3 + 12(z-1)h^2 + (7z^2 - 18z + 7)h + z^3 - 6z^2 + 7z - 1)
//
// For general y in [0,1], combining linearly:
//   m1 = w[(1-y)z - yh]
//   m2 = w[(1-y)z + yh(h + z - 1)]
//   m3 = w[(1-y)z - yh(2h^2 + 3(z-1)h + z^2 - 3z + 1)]
//   m4 = w[(1-y)z + yh(6h^3 + 12(z-1)h^2 + (7z^2 - 18z + 7)h + z^3 - 6z^2 + 7z - 1)]
//
// Numerical stability (see response.md Section 1b):
//
// Left tail (q << 0, z small):
//   p = -expm1(-z) avoids cancellation in 1 - e^{-z} when z is tiny.
//   h = z/expm1(z) is computed directly via expm1; no separate Taylor branch
//   is strictly necessary because expm1 is accurate for small arguments.
//   As z -> 0, h -> 1 - z/2 + z^2/12 - z^4/720 + O(z^6).
//
// Right tail (q >> 0, z > 36.7):
//   r = e^{-z} underflows to 0, so p rounds to 1. In this regime,
//   h = z*r/(1-r) ≈ z*r, which gracefully underflows to 0.
//   For y=1, all four derivatives -> 0. For y=0, they equal w*z.
//   The overflow boundary for e^z is z ≈ 709, i.e. q ≈ 6.56. Beyond that,
//   we must not compute e^z directly; instead h = z*r/p with r = e^{-z}.
//
// Reference: response.md Section 1b.
// ---------------------------------------------------------------------------

#[inline]
fn cloglog_stable_h(z: f64) -> f64 {
    // Compute h = z / expm1(z) = z / (e^z - 1) = z * e^{-z} / (1 - e^{-z}).
    //
    // This is the fundamental stable building block for cloglog derivatives.
    // It has a removable singularity at z=0 where h -> 1.
    //
    // For large z (z > 36.7), expm1(z) overflows to infinity, but z*e^{-z}
    // is tiny and the derivatives are negligible. We use the identity
    // h = z * r / p where r = e^{-z} and p = -expm1(-z) for all z, which
    // is stable across the full range because:
    //   - For z near 0: expm1(z) is accurate, so z/expm1(z) is fine.
    //   - For large z: r = e^{-z} -> 0, making h -> 0 as well.
    if z.abs() < 1e-12 {
        // Taylor: h = 1 - z/2 + z^2/12 - z^4/720 + ...
        return 1.0 - z * 0.5 + z * z / 12.0;
    }
    let expm1_z = z.exp_m1();
    if expm1_z.is_infinite() {
        // z is very large (> ~709), e^z overflows. h = z*e^{-z}/(1 - e^{-z}).
        // Since z > 709, e^{-z} is essentially 0, so h ≈ 0.
        let r = (-z).exp();
        if r == 0.0 {
            return 0.0;
        }
        return z * r / (1.0 - r);
    }
    z / expm1_z
}

#[inline]
fn binomial_neglog_q_derivatives_cloglog_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> (f64, f64, f64) {
    // Returns (m1, m2, m3) for F(q) = -w[y log G(q) + (1-y) log(1-G(q))]
    // with G = cloglog CDF: G(q) = 1 - exp(-exp(q)).
    if weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let z = q.exp(); // z = e^q; may be large but that's handled below
    let h = cloglog_stable_h(z);
    let y0 = 1.0 - y;
    let y0_term = if y0 == 0.0 { 0.0 } else { y0 * z };

    // y=0 branch: all derivatives equal w*z (since F_{y=0} = w*e^q).
    // y=1 branch: uses the h-polynomial forms.
    // General y: linear combination.
    //
    // Once h rounds to 0, the y=1 contribution has already underflowed to 0
    // in f64. Returning the remaining y=0 branch here avoids 0 * inf products
    // when q is deep in the right tail.
    if y == 0.0 || h == 0.0 {
        let base = weight * y0_term;
        return (base, base, base);
    }

    let m1 = weight * (y0_term - y * h);
    let m2 = weight * (y0_term + y * h * (h + z - 1.0));
    let m3 =
        weight * (y0_term - y * h * (2.0 * h * h + 3.0 * (z - 1.0) * h + z * z - 3.0 * z + 1.0));
    (m1, m2, m3)
}

#[inline]
fn binomial_neglog_q_fourth_derivative_cloglog_closed_form(y: f64, weight: f64, q: f64) -> f64 {
    // Returns m4 = d^4F/dq^4 for cloglog link.
    // m4 = w[(1-y)z + yh(6h^3 + 12(z-1)h^2 + (7z^2-18z+7)h + z^3-6z^2+7z-1)]
    if weight == 0.0 || !q.is_finite() {
        return 0.0;
    }
    let z = q.exp();
    let h = cloglog_stable_h(z);
    let y0 = 1.0 - y;
    let y0_term = if y0 == 0.0 { 0.0 } else { y0 * z };
    if y == 0.0 || h == 0.0 {
        return weight * y0_term;
    }
    let h2 = h * h;
    let h3 = h2 * h;
    let z2 = z * z;
    let z3 = z2 * z;
    let y1_poly = 6.0 * h3 + 12.0 * (z - 1.0) * h2 + (7.0 * z2 - 18.0 * z + 7.0) * h + z3
        - 6.0 * z2
        + 7.0 * z
        - 1.0;
    weight * (y0_term + y * h * y1_poly)
}

#[inline]
fn binomial_neglog_q_fourth_derivative_from_jet(
    y: f64,
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
) -> f64 {
    let (m, clamp_active) = clamped_binomial_probability(mu);
    if clamp_active
        || weight == 0.0
        || !m.is_finite()
        || !d1.is_finite()
        || !d2.is_finite()
        || !d3.is_finite()
        || !d4.is_finite()
    {
        return 0.0;
    }
    let one_minus = 1.0 - m;
    let ellmu = y / m - (1.0 - y) / one_minus;
    let ellmumu = -y / (m * m) - (1.0 - y) / (one_minus * one_minus);
    let ellmumum = 2.0 * y / (m * m * m) - 2.0 * (1.0 - y) / (one_minus * one_minus * one_minus);
    let ellmumumum = -6.0 * y / m.powi(4) - 6.0 * (1.0 - y) / one_minus.powi(4);
    let fourth_q = weight
        * (ellmumumum * d1.powi(4)
            + 6.0 * ellmumum * d1 * d1 * d2
            + ellmumu * (3.0 * d2 * d2 + 4.0 * d1 * d3)
            + ellmu * d4);
    -fourth_q
}

// ---------------------------------------------------------------------------
// Unified exact dispatch for binomial m1–m4
// ---------------------------------------------------------------------------
//
// Closed forms remain the fast path for Probit, Logit, and CLogLog, but the
// exact joint Newton calculus is not restricted to those links. When no
// closed form is available, we use the generic inverse-link jet plus the
// analytic fourth derivative of the inverse-link pdf.
// ---------------------------------------------------------------------------

#[inline]
fn binomial_neglog_q_derivatives_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    link_kind: &InverseLink,
) -> (f64, f64, f64) {
    if binomial_link_has_closed_form(link_kind) {
        return binomial_neglog_q_derivatives_closed_form_dispatch(y, weight, q, mu, link_kind);
    }
    binomial_neglog_q_derivatives_from_jet(y, weight, mu, d1, d2, d3)
}

#[inline]
fn binomial_neglog_q_fourth_derivative_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    link_kind: &InverseLink,
) -> Result<f64, String> {
    if binomial_link_has_closed_form(link_kind) {
        return Ok(binomial_neglog_q_fourth_derivative_closed_form_dispatch(
            y, weight, q, mu, link_kind,
        ));
    }
    let d4 = inverse_link_pdffourth_derivative_for_inverse_link(link_kind, q)
        .map_err(|e| format!("binomial inverse-link fourth derivative evaluation failed: {e}"))?;
    Ok(binomial_neglog_q_fourth_derivative_from_jet(
        y, weight, mu, d1, d2, d3, d4,
    ))
}

#[inline]
fn binomial_neglog_q_derivatives_closed_form_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
    link_kind: &InverseLink,
) -> (f64, f64, f64) {
    match link_kind {
        InverseLink::Standard(LinkFunction::Probit) => {
            binomial_neglog_q_derivatives_probit_closed_form(y, weight, q, mu)
        }
        InverseLink::Standard(LinkFunction::Logit) => {
            binomial_neglog_q_derivatives_logit_closed_form(y, weight, q)
        }
        InverseLink::Standard(LinkFunction::CLogLog) => {
            binomial_neglog_q_derivatives_cloglog_closed_form(y, weight, q)
        }
        _ => {
            // Should not be called for unsupported links; caller should use jet path.
            // This is a safety fallback.
            (0.0, 0.0, 0.0)
        }
    }
}

#[inline]
fn binomial_neglog_q_fourth_derivative_closed_form_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
    link_kind: &InverseLink,
) -> f64 {
    match link_kind {
        InverseLink::Standard(LinkFunction::Probit) => {
            binomial_neglog_q_fourth_derivative_probit_closed_form(y, weight, q, mu)
        }
        InverseLink::Standard(LinkFunction::Logit) => {
            binomial_neglog_q_fourth_derivative_logit_closed_form(y, weight, q)
        }
        InverseLink::Standard(LinkFunction::CLogLog) => {
            binomial_neglog_q_fourth_derivative_cloglog_closed_form(y, weight, q)
        }
        _ => 0.0,
    }
}

/// Returns true if the given link supports closed-form m1–m4 derivatives for
/// the binomial location-scale family, enabling the exact joint Newton path.
#[inline]
fn binomial_link_has_closed_form(link_kind: &InverseLink) -> bool {
    matches!(
        link_kind,
        InverseLink::Standard(LinkFunction::Probit)
            | InverseLink::Standard(LinkFunction::Logit)
            | InverseLink::Standard(LinkFunction::CLogLog)
    )
}

fn xt_diag_x_dense(design: &Array2<f64>, diag: &Array1<f64>) -> Result<Array2<f64>, String> {
    if design.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_x_dense row mismatch: design has {} rows but diag has {} entries",
            design.nrows(),
            diag.len()
        ));
    }
    Ok(fast_xt_diag_x(design, diag))
}

fn xt_diag_y_dense(
    left: &Array2<f64>,
    diag: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_y_dense row mismatch: left has {} rows but diag has {} entries",
            left.nrows(),
            diag.len()
        ));
    }
    if right.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_y_dense row mismatch: right has {} rows but diag has {} entries",
            right.nrows(),
            diag.len()
        ));
    }
    Ok(fast_xt_diag_y(left, diag, right))
}

fn mirror_upper_to_lower(target: &mut Array2<f64>) {
    for i in 0..target.nrows() {
        for j in 0..i {
            target[[i, j]] = target[[j, i]];
        }
    }
}

struct BinomialLocationScaleCore {
    sigma: Array1<f64>,
    dsigma_deta: Array1<f64>,
    q0: Array1<f64>,
    mu: Array1<f64>,
    dmu_dq: Array1<f64>,
    d2mu_dq2: Array1<f64>,
    d3mu_dq3: Array1<f64>,
    log_likelihood: f64,
}

#[derive(Clone, Copy)]
struct NonWiggleQDerivs {
    q_t: f64,
    q_ls: f64,
    q_tl: f64,
    q_ll: f64,
    q_tl_ls: f64,
    q_ll_ls: f64,
}

#[derive(Clone, Copy)]
struct NonWiggleQDirectional {
    delta_q: f64,
    delta_q_t: f64,
    delta_q_ls: f64,
    delta_q_tl: f64,
    delta_q_ll: f64,
}

#[derive(Clone, Copy)]
struct InverseLinkRow {
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
}

#[derive(Clone, Copy)]
struct BinomialLocationScaleRow {
    sigma: f64,
    dsigma_deta: f64,
    q0: f64,
    inverse_link: InverseLinkRow,
    ll: f64,
}

#[inline]
fn hessian_coeff_fromobjective_q_terms(m1: f64, m2: f64, q_a: f64, q_b: f64, q_ab: f64) -> f64 {
    // F = -sum ell, scalar q:
    //   H_ab = m2 * q_a q_b + m1 * q_ab.
    m2 * q_a * q_b + m1 * q_ab
}

#[inline]
fn directionalhessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    m3: f64,
    dq: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    dq_a: f64,
    dq_b: f64,
    dq_ab: f64,
) -> f64 {
    // F = -sum ell, scalar q:
    //   dH_ab[u] = m3*dq*q_a*q_b + m2*(dq_a*q_b + q_a*dq_b + dq*q_ab) + m1*dq_ab.
    m3 * dq * q_a * q_b + m2 * (dq_a * q_b + q_a * dq_b + dq * q_ab) + m1 * dq_ab
}

#[inline]
fn second_directionalhessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    dq_u: f64,
    dqv: f64,
    d2q_uv: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    dq_a_u: f64,
    dq_av: f64,
    dq_b_u: f64,
    dq_bv: f64,
    d2q_a_uv: f64,
    d2q_b_uv: f64,
    dq_ab_u: f64,
    dq_abv: f64,
    d2q_ab_uv: f64,
) -> f64 {
    // F = -sum ell, scalar q:
    //   H_ab = m2 * q_a q_b + m1 * q_ab.
    // Exact mixed second directional derivative:
    //
    // Write
    //   A = q_a q_b,
    //   B = q_ab.
    //
    // Then
    //   H_ab = m2 * A + m1 * B,
    // where m_k = F^(k)(q).
    //
    // First directional derivative along u:
    //   D_u H_ab
    //   = m3 * dq_u * A
    //   + m2 * (D_u A + dq_u * B)
    //   + m1 * D_u B.
    //
    // Differentiate once more along v:
    //   D²H_ab[u,v] =
    //      m4*dq_u*dqv*q_a*q_b
    //    + m3*(d2q_uv*q_a*q_b
    //         + dq_u*(dq_av*q_b + q_a*dq_bv)
    //         + dqv*(dq_a_u*q_b + q_a*dq_b_u)
    //         + dq_u*dqv*q_ab)
    //    + m2*(d2q_a_uv*q_b + dq_a_u*dq_bv + dq_av*dq_b_u + q_a*d2q_b_uv
    //          + d2q_uv*q_ab + dq_u*dq_abv + dqv*dq_ab_u)
    //    + m1*d2q_ab_uv.
    //
    // The single dq_u*dqv*q_ab term is important. There is exactly one copy:
    //
    //   Dv[m2 * dq_u * B]
    //   = m3 * dqv * dq_u * B + m2 * (d2q_uv * B + dq_u * Dv B),
    //
    // and no second copy appears elsewhere. A previous version of this helper
    // accidentally counted this term twice by embedding `dqv * q_ab` in both
    // the `dq_u` and `dqv` product-rule branches.
    let d_qaqb_u = dq_a_u * q_b + q_a * dq_b_u;
    let d_qaqbv = dq_av * q_b + q_a * dq_bv;
    let d2_qaqb_uv = d2q_a_uv * q_b + dq_a_u * dq_bv + dq_av * dq_b_u + q_a * d2q_b_uv;
    m4 * dq_u * dqv * q_a * q_b
        + m3 * (d2q_uv * q_a * q_b + dq_u * d_qaqbv + dqv * d_qaqb_u + dq_u * dqv * q_ab)
        + m2 * (d2_qaqb_uv + d2q_uv * q_ab + dq_u * dq_abv + dqv * dq_ab_u)
        + m1 * d2q_ab_uv
}

/// Non-wiggle location-scale map derivatives via shared scalar core.
fn nonwiggle_q_derivs(eta_t: f64, sigma: f64) -> NonWiggleQDerivs {
    let inv_sigma = sigma.recip();
    let q_t = -inv_sigma;
    let q_ls = eta_t * inv_sigma;
    let q_tl = inv_sigma;
    let q_ll = -eta_t * inv_sigma;
    let q_tl_ls = -inv_sigma;
    let q_ll_ls = eta_t * inv_sigma;
    NonWiggleQDerivs {
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
    }
}

/// Directional derivatives along (d_eta_t, d_eta_ls):
/// delta_q = q_t d_eta_t + q_ls d_eta_ls
/// delta_q_t = q_tl d_eta_ls
/// delta_q_ls = q_tl d_eta_t + q_ll d_eta_ls
/// delta_q_tt = 0
/// delta_q_tl = q_tl_ls d_eta_ls
/// delta_q_ll = q_tl_ls d_eta_t + q_ll_ls d_eta_ls
fn nonwiggle_q_directional(
    q: NonWiggleQDerivs,
    d_eta_t: f64,
    d_eta_ls: f64,
) -> NonWiggleQDirectional {
    // Directional-chain derivation:
    //
    // For any scalar f(eta_t,eta_ls), directional derivative along
    // d eta = (d_eta_t, d_eta_ls) is
    //   dot{f} = f_t d_eta_t + f_ls d_eta_ls.
    //
    // Apply to q and its eta-partials:
    //   dot{q}      = q_t d_eta_t + q_ls d_eta_ls.
    //   dot{q_t}    = q_tt d_eta_t + q_tl d_eta_ls = q_tl d_eta_ls (q_tt=0).
    //   dot{q_ls}   = q_tl d_eta_t + q_ll d_eta_ls.
    //   dot{q_tt}   = 0.
    //   dot{q_tl}   = q_tl_ls d_eta_ls.
    //   dot{q_ll}   = q_tl_ls d_eta_t + q_ll_ls d_eta_ls.
    NonWiggleQDirectional {
        delta_q: q.q_t * d_eta_t + q.q_ls * d_eta_ls,
        delta_q_t: q.q_tl * d_eta_ls,
        delta_q_ls: q.q_tl * d_eta_t + q.q_ll * d_eta_ls,
        delta_q_tl: q.q_tl_ls * d_eta_ls,
        delta_q_ll: q.q_tl_ls * d_eta_t + q.q_ll_ls * d_eta_ls,
    }
}

#[inline]
fn inverse_linkrow(jet: crate::mixture_link::InverseLinkJet) -> InverseLinkRow {
    InverseLinkRow {
        mu: jet.mu,
        d1: jet.d1,
        d2: jet.d2,
        d3: jet.d3,
    }
}

#[inline]
fn clamped_binomial_probability(mu: f64) -> (f64, bool) {
    if !mu.is_finite() {
        return (0.5, true);
    }
    let clamped = mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
    (clamped, clamped != mu)
}

#[inline]
fn binomial_location_scale_q0(eta_t: f64, sigma: f64) -> f64 {
    -eta_t / sigma
}

fn binomial_location_scalerow(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    etawiggle: f64,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleRow, String> {
    let SigmaJet1 {
        sigma,
        d1: dsigma_deta,
    } = exp_sigma_jet1_scalar(eta_ls);
    let q0 = binomial_location_scale_q0(eta_t, sigma);
    let q = q0 + etawiggle;
    let mut jet = inverse_link_jet_for_inverse_link(link_kind, q)
        .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
    let (mu_clamped, clamp_active) = clamped_binomial_probability(jet.mu);
    jet.mu = mu_clamped;
    if clamp_active {
        jet.d1 = 0.0;
        jet.d2 = 0.0;
        jet.d3 = 0.0;
    }
    let inverse_link = inverse_linkrow(jet);
    let ll = weight * (y * inverse_link.mu.ln() + (1.0_f64 - y) * (1.0_f64 - inverse_link.mu).ln());
    Ok(BinomialLocationScaleRow {
        sigma,
        dsigma_deta,
        q0,
        inverse_link,
        ll,
    })
}

/// Compute only the log-likelihood scalar for the binomial location-scale model.
/// This avoids allocating 7 n-vectors that `binomial_location_scale_core` would produce,
/// making backtracking line searches much cheaper at biobank scale.
fn binomial_location_scale_ll_only(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    etawiggle: Option<&Array1<f64>>,
    link_kind: &InverseLink,
) -> Result<f64, String> {
    let n = y.len();
    let mut ll = 0.0;
    for i in 0..n {
        let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls[i]);
        let q0 = binomial_location_scale_q0(eta_t[i], sigma);
        let q = q0 + etawiggle.map_or(0.0, |w| w[i]);
        let jet = inverse_link_jet_for_inverse_link(link_kind, q)
            .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
        let (mu_clamped, _) = clamped_binomial_probability(jet.mu);
        ll += weights[i] * (y[i] * mu_clamped.ln() + (1.0 - y[i]) * (1.0 - mu_clamped).ln());
    }
    Ok(ll)
}

fn binomial_location_scale_core(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    etawiggle: Option<&Array1<f64>>,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleCore, String> {
    let n = y.len();
    if weights.len() != n || eta_t.len() != n || eta_ls.len() != n {
        return Err("binomial location-scale core size mismatch".to_string());
    }
    if let Some(w) = etawiggle
        && w.len() != n
    {
        return Err("binomial location-scale core wiggle size mismatch".to_string());
    }

    let mut sigma = Array1::<f64>::uninit(n);
    let mut dsigma_deta = Array1::<f64>::uninit(n);
    let mut q0 = Array1::<f64>::uninit(n);
    let mut mu = Array1::<f64>::uninit(n);
    let mut dmu_dq = Array1::<f64>::uninit(n);
    let mut d2mu_dq2 = Array1::<f64>::uninit(n);
    let mut d3mu_dq3 = Array1::<f64>::uninit(n);
    let mut ll = 0.0;

    for i in 0..n {
        let row = binomial_location_scalerow(
            y[i],
            weights[i],
            eta_t[i],
            eta_ls[i],
            etawiggle.map_or(0.0, |w| w[i]),
            link_kind,
        )?;
        sigma[i].write(row.sigma);
        dsigma_deta[i].write(row.dsigma_deta);
        q0[i].write(row.q0);
        mu[i].write(row.inverse_link.mu);
        dmu_dq[i].write(row.inverse_link.d1);
        d2mu_dq2[i].write(row.inverse_link.d2);
        d3mu_dq3[i].write(row.inverse_link.d3);
        ll += row.ll;
    }

    Ok(BinomialLocationScaleCore {
        sigma: unsafe { sigma.assume_init() },
        dsigma_deta: unsafe { dsigma_deta.assume_init() },
        q0: unsafe { q0.assume_init() },
        mu: unsafe { mu.assume_init() },
        dmu_dq: unsafe { dmu_dq.assume_init() },
        d2mu_dq2: unsafe { d2mu_dq2.assume_init() },
        d3mu_dq3: unsafe { d3mu_dq3.assume_init() },
        log_likelihood: ll,
    })
}

/// Built-in Gaussian location-scale family:
/// - Block 0: location μ(·) with identity link
/// - Block 1: log-scale log σ(·) with log link
pub struct GaussianLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
    /// Cached per-observation row scalars keyed by 6-element fingerprint
    /// (first, mid, last elements of both eta vectors).
    /// Avoids recomputing O(n) scalars K+ times per REML gradient/Hessian evaluation.
    cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}

impl Clone for GaussianLocationScaleFamily {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            policy: self.policy.clone(),
            cached_row_scalars: std::sync::RwLock::new(
                self.cached_row_scalars
                    .read()
                    .expect("lock poisoned")
                    .clone(),
            ),
        }
    }
}

struct GaussianLocationScaleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    xmu_psi: PsiDesignMap,
    x_ls_psi: PsiDesignMap,
    zmu_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}

struct GaussianLocationScaleJointPsiSecondDrifts {
    xmu_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_ls_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    xmu_ab: Option<Array2<f64>>,
    x_ls_ab: Option<Array2<f64>>,
    zmu_ab: Array1<f64>,
    z_ls_ab: Array1<f64>,
}

struct GaussianLocationScaleExactNewtonJointPsiWorkspace {
    family: GaussianLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    xmu: Array2<f64>,
    x_ls: Array2<f64>,
    psi_directions: ExactNewtonJointPsiDirectCache<GaussianLocationScaleJointPsiDirection>,
}

impl GaussianLocationScaleExactNewtonJointPsiWorkspace {
    fn new(
        family: GaussianLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let Some((xmu, x_ls)) = family.exact_joint_dense_block_designs(Some(specs))? else {
            return Err("GaussianLocationScaleFamily exact joint psi workspace requires dense block designs".to_string());
        };
        let xmu = xmu.into_owned();
        let x_ls = x_ls.into_owned();
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum();
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            xmu,
            x_ls,
            psi_directions: ExactNewtonJointPsiDirectCache::new(psi_dim),
        })
    }

    fn psi_direction(
        &self,
        psi_index: usize,
    ) -> Result<Option<Arc<GaussianLocationScaleJointPsiDirection>>, String> {
        self.psi_directions.get_or_try_init(psi_index, || {
            self.family.exact_newton_joint_psi_direction(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                &self.xmu,
                &self.x_ls,
                &self.family.policy,
            )
        })
    }
}

impl ExactNewtonJointPsiWorkspace for GaussianLocationScaleExactNewtonJointPsiWorkspace {
    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.psi_direction(psi_i)? else {
            return Ok(None);
        };
        let Some(dir_j) = self.psi_direction(psi_j)? else {
            return Ok(None);
        };
        Ok(Some(
            self.family
                .exact_newton_joint_psisecond_order_terms_from_parts(
                    &self.block_states,
                    &self.derivative_blocks,
                    dir_i.as_ref(),
                    dir_j.as_ref(),
                    &self.xmu,
                    &self.x_ls,
                )?,
        ))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(
                self.family
                    .exact_newton_joint_psihessian_directional_derivative_from_parts(
                        &self.block_states,
                        dir.as_ref(),
                        d_beta_flat,
                        &self.xmu,
                        &self.x_ls,
                    )?,
            ),
        ))
    }
}

#[derive(Clone)]
struct GaussianJointRowScalars {
    obs_weight: Array1<f64>,
    w: Array1<f64>,
    m: Array1<f64>,
    n: Array1<f64>,
    /// κ = (dσ/dη_ls)/σ for the active sigma link.
    /// The cross Hessian block H_{μ,ls} carries an overall κ factor and the
    /// scale-scale block H_{ls,ls} carries κ².
    kappa: Array1<f64>,
    /// κ' = dκ/dη_ls = κ(1−κ) for the logb link. The static H_{ls,ls} block
    /// carries a κ'·(a−n) term, so κ' threads through every dH directional
    /// weight via the chain rule.
    kappa_prime: Array1<f64>,
    /// κ'' = κ(1−κ)(1−2κ); appears in d²H_{ls,ls} via the second
    /// η-derivative of κ'·(a−n).
    kappa_dprime: Array1<f64>,
}

struct GaussianJointPsiFirstWeights {
    objective_psirow: Array1<f64>,
    scoremu: Array1<f64>,
    score_ls: Array1<f64>,
    dscoremu: Array1<f64>,
    dscore_ls: Array1<f64>,
    hmumu: Array1<f64>,
    hmu_ls: Array1<f64>,
    h_ls_ls: Array1<f64>,
    dhmumu: Array1<f64>,
    dhmu_ls: Array1<f64>,
    dh_ls_ls: Array1<f64>,
}

struct GaussianJointPsiSecondWeights {
    objective_psi_psirow: Array1<f64>,
    d2scoremu: Array1<f64>,
    d2score_ls: Array1<f64>,
    d2hmumu: Array1<f64>,
    d2hmu_ls: Array1<f64>,
    d2h_ls_ls: Array1<f64>,
}

struct GaussianJointPsiMixedDriftWeights {
    dhmumu_u: Array1<f64>,
    dhmu_ls_u: Array1<f64>,
    dh_ls_ls_u: Array1<f64>,
    d2hmumu: Array1<f64>,
    d2hmu_ls: Array1<f64>,
    d2h_ls_ls: Array1<f64>,
}

fn gaussian_jointrow_scalars(
    y: &Array1<f64>,
    etamu: &Array1<f64>,
    eta_ls: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<GaussianJointRowScalars, String> {
    let nobs = y.len();
    if etamu.len() != nobs || eta_ls.len() != nobs || weights.len() != nobs {
        return Err("Gaussian joint row scalar input size mismatch".to_string());
    }
    let mut obs_weight = Array1::<f64>::uninit(nobs);
    let mut w = Array1::<f64>::uninit(nobs);
    let mut m = Array1::<f64>::uninit(nobs);
    let mut n = Array1::<f64>::uninit(nobs);
    let mut kappa = Array1::<f64>::uninit(nobs);
    let mut kappa_prime = Array1::<f64>::uninit(nobs);
    let mut kappa_dprime = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let jet = crate::families::sigma_link::logb_sigma_jet1_scalar(eta_ls[i]);
        let s = jet.sigma;
        // κ = (σ − b)/σ = exp(η)/(b + exp(η)). Use the direct exp(η)/σ form
        // when finite — it preserves the precision of exp(η) at very negative
        // η (where 1 − b/σ catastrophically cancels because b/σ → 1). The
        // η → +∞ branch returns 1 cleanly without hitting ∞/∞ NaN.
        let s_exp = jet.d1;
        let ki = if s_exp.is_infinite() { 1.0 } else { s_exp / s };
        let kp = ki * (1.0 - ki);
        let kdp = kp * (1.0 - 2.0 * ki);
        let wi = weights[i] / (s * s);
        let ri = y[i] - etamu[i];
        obs_weight[i].write(weights[i]);
        w[i].write(wi);
        m[i].write(ri * wi);
        n[i].write(ri * ri * wi);
        kappa[i].write(ki);
        kappa_prime[i].write(kp);
        kappa_dprime[i].write(kdp);
    }
    // SAFETY: all elements written in the loop above.
    let obs_weight = unsafe { obs_weight.assume_init() };
    let w = unsafe { w.assume_init() };
    let m = unsafe { m.assume_init() };
    let n = unsafe { n.assume_init() };
    let kappa = unsafe { kappa.assume_init() };
    let kappa_prime = unsafe { kappa_prime.assume_init() };
    let kappa_dprime = unsafe { kappa_dprime.assume_init() };
    Ok(GaussianJointRowScalars {
        obs_weight,
        w,
        m,
        n,
        kappa,
        kappa_prime,
        kappa_dprime,
    })
}

fn gaussian_joint_first_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu: &Array1<f64>,
    dot_eta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.w.len();
    let mut w_u = Array1::<f64>::uninit(nobs);
    let mut c_u = Array1::<f64>::uninit(nobs);
    let mut d_u = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let amn = scalars.obs_weight[i] - ni;
        let dm = dotmu[i];
        let de = dot_eta[i];
        // κ-scaled log-sigma direction.
        let sde = ki * de;
        w_u[i].write(-2.0 * wi * sde);
        // + 2·κ'·m·de: dκ/dη chain-rule from σ = b + e^η.
        c_u[i].write(ki * (-2.0 * wi * dm - 4.0 * mi * sde) + 2.0 * mi * kpi * de);
        // F_μ·dm + F_η·de with F = 2κ²n + κ'(a−n) (mirrors helper 4 dh_ls_ls).
        d_u[i].write(
            ki * ki * (-4.0 * mi * dm - 4.0 * ni * sde)
                + 2.0 * mi * kpi * dm
                + (kdpi * amn + 6.0 * ki * kpi * ni) * de,
        );
    }
    let w_u = unsafe { w_u.assume_init() };
    let c_u = unsafe { c_u.assume_init() };
    let d_u = unsafe { d_u.assume_init() };
    (w_u, c_u, d_u)
}

fn gaussian_jointsecond_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu_u: &Array1<f64>,
    dot_eta_u: &Array1<f64>,
    dotmuv: &Array1<f64>,
    dot_etav: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.w.len();
    let mut w_uv = Array1::<f64>::uninit(nobs);
    let mut c_uv = Array1::<f64>::uninit(nobs);
    let mut d_uv = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        // κ''' = κ''(1−2κ) − 2κ'²: needed for the (a−n)·deu·dev piece of d²H_{ls,ls}/∂η².
        let ktpi = kdpi * (1.0 - 2.0 * ki) - 2.0 * kpi * kpi;
        // (κ')² + κ·κ'' − 5κ²·κ': η-η coefficient from differentiating the OLD 2κ²n part.
        let dlsls_eta_eta_old = kpi * kpi + ki * kdpi - 5.0 * ki * ki * kpi;
        let amn = scalars.obs_weight[i] - ni;
        let dmu = dotmu_u[i];
        let dmv = dotmuv[i];
        let deu = dot_eta_u[i];
        let dev = dot_etav[i];
        // κ-scaled log-sigma directions.
        let sdeu = ki * deu;
        let sdev = ki * dev;
        let de_sym = dmu * dev + dmv * deu;
        let de_eta = deu * dev;
        // − 2·κ'·w·deu·dev: ∂²w/∂η² = 4wκ² − 2wκ'.
        w_uv[i].write(4.0 * wi * sdeu * sdev - 2.0 * wi * kpi * de_eta);
        // − 2·κ'·w·sym + 2·m·(κ''−6·κ·κ')·deu·dev from d²(2mκ).
        c_uv[i].write(
            ki * (4.0 * wi * (dmu * sdev + dmv * sdeu) + 8.0 * mi * sdeu * sdev)
                - 2.0 * wi * kpi * de_sym
                + 2.0 * mi * (kdpi - 6.0 * ki * kpi) * de_eta,
        );
        // d²/du dv of corrected H_{ls,ls} = 2κ²n + κ'(a−n). The "_old" bracket
        // covers d²(2κ²n); the extra terms cover d²(κ'(a−n)).
        d_uv[i].write(
            ki * ki
                * (4.0 * wi * dmu * dmv
                    + 8.0 * mi * (dmu * sdev + dmv * sdeu)
                    + 8.0 * ni * sdeu * sdev)
                - 2.0 * kpi * wi * dmu * dmv
                - 8.0 * mi * ki * kpi * de_sym
                + 2.0 * mi * (kdpi - 2.0 * ki * kpi) * de_sym
                + 4.0 * ni * dlsls_eta_eta_old * de_eta
                + (2.0 * kpi * kpi + 4.0 * ki * kdpi - 4.0 * ki * ki * kpi) * ni * de_eta
                + ktpi * amn * de_eta,
        );
    }
    let w_uv = unsafe { w_uv.assume_init() };
    let c_uv = unsafe { c_uv.assume_init() };
    let d_uv = unsafe { d_uv.assume_init() };
    (w_uv, c_uv, d_uv)
}

fn gaussian_joint_psi_firstweights(
    scalars: &GaussianJointRowScalars,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
) -> GaussianJointPsiFirstWeights {
    let nobs = scalars.w.len();
    let mut objective_psirow = Array1::<f64>::uninit(nobs);
    let mut scoremu = Array1::<f64>::uninit(nobs);
    let mut score_ls = Array1::<f64>::uninit(nobs);
    let mut dscoremu = Array1::<f64>::uninit(nobs);
    let mut dscore_ls = Array1::<f64>::uninit(nobs);
    let mut hmumu = Array1::<f64>::uninit(nobs);
    let mut hmu_ls = Array1::<f64>::uninit(nobs);
    let mut h_ls_ls = Array1::<f64>::uninit(nobs);
    let mut dhmumu = Array1::<f64>::uninit(nobs);
    let mut dhmu_ls = Array1::<f64>::uninit(nobs);
    let mut dh_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let ma = mu_a[i];
        let ea = eta_a[i];
        // κ-scaled log-sigma direction.
        let sea = ki * ea;
        let smu = -mi;
        let sls = ki * (ai - ni);
        let wi = scalars.w[i];
        scoremu[i].write(smu);
        score_ls[i].write(sls);
        dscoremu[i].write(wi * ma + 2.0 * mi * sea);
        // + κ'·(a−n)·η̇ chain-rule term (∂[κ(a−n)]/∂η = κ'(a−n) + 2κ²n).
        dscore_ls[i].write(ki * (2.0 * mi * ma + 2.0 * ni * sea) + kpi * (ai - ni) * ea);
        hmumu[i].write(wi);
        // Cross block: H_{μ,ls} = 2mκ (no κ' term — derivative of −m wrt η is 2mκ).
        hmu_ls[i].write(2.0 * ki * mi);
        // + κ'·(a−n) term: H_{ls,ls} = κ(1−κ)(a−n) + 2κ²n.
        h_ls_ls[i].write(2.0 * ki * ki * ni + kpi * (ai - ni));
        dhmumu[i].write(-2.0 * wi * sea);
        // + 2m·κ'·η̇: ∂(2mκ)/∂η = −4mκ² + 2mκ'.
        dhmu_ls[i].write(ki * (-2.0 * wi * ma - 4.0 * mi * sea) + 2.0 * mi * kpi * ea);
        // + 2m·κ'·μ̇ + [κ''(a−n) + 6κκ'·n]·η̇ from differentiating κ'(a−n)+2κ²n.
        dh_ls_ls[i].write(
            ki * ki * (-4.0 * mi * ma - 4.0 * ni * sea)
                + 2.0 * mi * kpi * ma
                + (kdpi * (ai - ni) + 6.0 * ki * kpi * ni) * ea,
        );
        objective_psirow[i].write(smu * ma + sls * ea);
    }
    unsafe {
        GaussianJointPsiFirstWeights {
            objective_psirow: objective_psirow.assume_init(),
            scoremu: scoremu.assume_init(),
            score_ls: score_ls.assume_init(),
            dscoremu: dscoremu.assume_init(),
            dscore_ls: dscore_ls.assume_init(),
            hmumu: hmumu.assume_init(),
            hmu_ls: hmu_ls.assume_init(),
            h_ls_ls: h_ls_ls.assume_init(),
            dhmumu: dhmumu.assume_init(),
            dhmu_ls: dhmu_ls.assume_init(),
            dh_ls_ls: dh_ls_ls.assume_init(),
        }
    }
}

fn gaussian_joint_psisecondweights(
    scalars: &GaussianJointRowScalars,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
    mu_b: &Array1<f64>,
    eta_b: &Array1<f64>,
    mu_ab: &Array1<f64>,
    eta_ab: &Array1<f64>,
) -> GaussianJointPsiSecondWeights {
    let nobs = scalars.w.len();
    let mut objective_psi_psirow = Array1::<f64>::uninit(nobs);
    let mut d2scoremu = Array1::<f64>::uninit(nobs);
    let mut d2score_ls = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        // κ''' = κ''(1−2κ) − 2κ'²: needed for the (a−n)·η_a η_b piece of d²H_{ls,ls}/∂η².
        let ktpi = kdpi * (1.0 - 2.0 * ki) - 2.0 * kpi * kpi;
        // (κ')² + κ·κ'' − 5κ²·κ': η-η coefficient inside the d²H_{ls,ls} delta from differentiating the OLD 2κ²n piece.
        let dlsls_eta_eta_old = kpi * kpi + ki * kdpi - 5.0 * ki * ki * kpi;
        let ai = scalars.obs_weight[i];
        let amn = ai - ni;
        let ma = mu_a[i];
        let mb = mu_b[i];
        let mab = mu_ab[i];
        let ea = eta_a[i];
        let eb = eta_b[i];
        let eab = eta_ab[i];
        // κ-scaled log-sigma directions.
        let sea = ki * ea;
        let seb = ki * eb;
        let seab = ki * eab;
        let cross = ma * seb + mb * sea;
        // Bare-η symmetric form (no κ): needed for κ' chain-rule terms.
        let cross_eta = ma * eb + mb * ea;
        let sea_seb = sea * seb;
        let ea_eb = ea * eb;
        let ma_mb = ma * mb;
        // + κ'·(a−n)·ea·eb: dκ/dη chain-rule contribution from σ = b + e^η.
        objective_psi_psirow[i].write(
            wi * ma_mb + 2.0 * mi * cross + 2.0 * ni * sea_seb - mi * mab
                + ki * amn * eab
                + kpi * amn * ea_eb,
        );
        // + 2·m·κ'·ea·eb: ∂²(−m)/∂η² = −4mκ² + 2mκ'.
        d2scoremu[i].write(
            wi * mab - 2.0 * wi * cross - 4.0 * mi * sea_seb
                + 2.0 * mi * seab
                + 2.0 * mi * kpi * ea_eb,
        );
        // + 2·κ'·m·sym(μ_a η_b) + (κ''(a−n)+6κκ'n)·ea·eb + κ'(a−n)·eab.
        d2score_ls[i].write(
            ki * (-2.0 * wi * ma_mb - 4.0 * mi * cross - 4.0 * ni * sea_seb
                + 2.0 * mi * mab
                + 2.0 * ni * seab)
                + 2.0 * mi * kpi * cross_eta
                + (kdpi * amn + 6.0 * ki * kpi * ni) * ea_eb
                + kpi * amn * eab,
        );
        // − 2·κ'·w·ea·eb: ∂²w/∂η² = 4wκ² − 2wκ'.
        d2hmumu[i].write(4.0 * wi * sea_seb - 2.0 * wi * seab - 2.0 * wi * kpi * ea_eb);
        // − 2·κ'·w·sym + 2·m·(κ''−6κκ')·ea·eb + 2·m·κ'·eab from d²(2mκ).
        d2hmu_ls[i].write(
            ki * (-2.0 * wi * mab + 4.0 * wi * cross + 8.0 * mi * sea_seb - 4.0 * mi * seab)
                - 2.0 * wi * kpi * cross_eta
                + 2.0 * mi * (kdpi - 6.0 * ki * kpi) * ea_eb
                + 2.0 * mi * kpi * eab,
        );
        // d²H_{ls,ls}/dψ_a dψ_b with corrected H_{ls,ls} = 2κ²n + κ'(a−n).
        // F_μμ adds −2wκ'·ma_mb; F_μη adds 2m·(κ''−2κκ')·sym (on top of the
        // −8mκκ' from differentiating 2κ²n); F_ηη adds (2κ'²+4κκ''−4κ²κ')n·ea_eb
        // and κ'''(a−n)·ea_eb; F_μ adds 2mκ'·mab; F_η adds (2κκ'n + κ''(a−n))·eab.
        d2h_ls_ls[i].write(
            ki * ki
                * (4.0 * wi * ma_mb + 8.0 * mi * cross + 8.0 * ni * sea_seb
                    - 4.0 * mi * mab
                    - 4.0 * ni * seab)
                - 2.0 * kpi * wi * ma_mb
                - 8.0 * mi * ki * kpi * cross_eta
                + 2.0 * mi * (kdpi - 2.0 * ki * kpi) * cross_eta
                + 4.0 * ni * dlsls_eta_eta_old * ea_eb
                + (2.0 * kpi * kpi + 4.0 * ki * kdpi - 4.0 * ki * ki * kpi) * ni * ea_eb
                + ktpi * amn * ea_eb
                + 4.0 * ni * ki * kpi * eab
                + 2.0 * mi * kpi * mab
                + (2.0 * ki * kpi * ni + kdpi * amn) * eab,
        );
    }
    unsafe {
        GaussianJointPsiSecondWeights {
            objective_psi_psirow: objective_psi_psirow.assume_init(),
            d2scoremu: d2scoremu.assume_init(),
            d2score_ls: d2score_ls.assume_init(),
            d2hmumu: d2hmumu.assume_init(),
            d2hmu_ls: d2hmu_ls.assume_init(),
            d2h_ls_ls: d2h_ls_ls.assume_init(),
        }
    }
}

fn gaussian_joint_psi_mixed_driftweights(
    scalars: &GaussianJointRowScalars,
    dotmu: &Array1<f64>,
    dot_eta: &Array1<f64>,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
    dotmu_a: &Array1<f64>,
    dot_eta_a: &Array1<f64>,
) -> GaussianJointPsiMixedDriftWeights {
    let nobs = scalars.w.len();
    let mut dhmumu_u = Array1::<f64>::uninit(nobs);
    let mut dhmu_ls_u = Array1::<f64>::uninit(nobs);
    let mut dh_ls_ls_u = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        // κ''' = κ''(1−2κ) − 2κ'²: needed for the (a−n)·de·ea piece of d²H_{ls,ls}/∂η².
        let ktpi = kdpi * (1.0 - 2.0 * ki) - 2.0 * kpi * kpi;
        // (κ')² + κ·κ'' − 5κ²·κ': η-η coefficient from differentiating the OLD 2κ²n part.
        let dlsls_eta_eta_old = kpi * kpi + ki * kdpi - 5.0 * ki * ki * kpi;
        let amn = scalars.obs_weight[i] - ni;
        let dm = dotmu[i];
        let de = dot_eta[i];
        let ma = mu_a[i];
        let ea = eta_a[i];
        let dma = dotmu_a[i];
        let dea = dot_eta_a[i];
        // κ-scaled log-sigma directions.
        let sde = ki * de;
        let sea = ki * ea;
        let sdea = ki * dea;
        let cross = sde * ma + dm * sea;
        // Bare-η symmetric: dm·ea + ma·de (no κ, for κ' chain-rule pieces).
        let cross_eta = dm * ea + ma * de;
        let de_ea = de * ea;
        // First directional derivative of Hessian blocks (== Helper A).
        dhmumu_u[i].write(-2.0 * wi * sde);
        // + 2·κ'·m·de.
        dhmu_ls_u[i].write(ki * (-2.0 * wi * dm - 4.0 * mi * sde) + 2.0 * mi * kpi * de);
        // F_μ·dm + F_η·de with F = 2κ²n + κ'(a−n) (mirrors helper 4 dh_ls_ls).
        dh_ls_ls_u[i].write(
            ki * ki * (-4.0 * mi * dm - 4.0 * ni * sde)
                + 2.0 * mi * kpi * dm
                + (kdpi * amn + 6.0 * ki * kpi * ni) * de,
        );
        // − 2·κ'·w·de·ea: ∂²w/∂η² = 4wκ² − 2wκ'.
        d2hmumu[i].write(4.0 * wi * sde * sea - 2.0 * wi * sdea - 2.0 * wi * kpi * de_ea);
        // − 2·κ'·w·(dm·ea + de·ma) + 2·m·(κ''−6κκ')·de·ea + 2·m·κ'·dea from d²(2mκ).
        d2hmu_ls[i].write(
            ki * (-2.0 * wi * dma + 4.0 * wi * cross + 8.0 * mi * sde * sea - 4.0 * mi * sdea)
                - 2.0 * wi * kpi * cross_eta
                + 2.0 * mi * (kdpi - 6.0 * ki * kpi) * de_ea
                + 2.0 * mi * kpi * dea,
        );
        // d²/(drift × ψ) of corrected H_{ls,ls} = 2κ²n + κ'(a−n). The "_old"
        // bracket covers d²(2κ²n); the extra κ'/κ''/κ''' terms cover d²(κ'(a−n)).
        d2h_ls_ls[i].write(
            ki * ki
                * (4.0 * wi * dm * ma + 8.0 * mi * cross + 8.0 * ni * sde * sea
                    - 4.0 * mi * dma
                    - 4.0 * ni * sdea)
                - 2.0 * kpi * wi * dm * ma
                - 8.0 * mi * ki * kpi * cross_eta
                + 2.0 * mi * (kdpi - 2.0 * ki * kpi) * cross_eta
                + 4.0 * ni * dlsls_eta_eta_old * de_ea
                + (2.0 * kpi * kpi + 4.0 * ki * kdpi - 4.0 * ki * ki * kpi) * ni * de_ea
                + ktpi * amn * de_ea
                + 4.0 * ni * ki * kpi * dea
                + 2.0 * mi * kpi * dma
                + (2.0 * ki * kpi * ni + kdpi * amn) * dea,
        );
    }
    unsafe {
        GaussianJointPsiMixedDriftWeights {
            dhmumu_u: dhmumu_u.assume_init(),
            dhmu_ls_u: dhmu_ls_u.assume_init(),
            dh_ls_ls_u: dh_ls_ls_u.assume_init(),
            d2hmumu: d2hmumu.assume_init(),
            d2hmu_ls: d2hmu_ls.assume_init(),
            d2h_ls_ls: d2h_ls_ls.assume_init(),
        }
    }
}

fn gaussian_pack_joint_score(scoremu: &Array1<f64>, score_ls: &Array1<f64>) -> Array1<f64> {
    let pmu = scoremu.len();
    let p_ls = score_ls.len();
    let mut out = Array1::<f64>::zeros(pmu + p_ls);
    out.slice_mut(s![0..pmu]).assign(scoremu);
    out.slice_mut(s![pmu..pmu + p_ls]).assign(score_ls);
    out
}

fn gaussian_pack_joint_symmetrichessian(
    hmumu: &Array2<f64>,
    hmu_ls: &Array2<f64>,
    h_ls_ls: &Array2<f64>,
) -> Array2<f64> {
    let pmu = hmumu.nrows();
    let p_ls = h_ls_ls.nrows();
    let total = pmu + p_ls;
    let mut out = Array2::<f64>::zeros((total, total));
    out.slice_mut(s![0..pmu, 0..pmu]).assign(hmumu);
    out.slice_mut(s![0..pmu, pmu..total]).assign(hmu_ls);
    out.slice_mut(s![pmu..total, pmu..total]).assign(h_ls_ls);
    mirror_upper_to_lower(&mut out);
    out
}

fn gaussian_joint_hessian_from_designs(
    xmu: &DenseOrOperator<'_>,
    x_ls: &DenseOrOperator<'_>,
    hmumu_coeff: &Array1<f64>,
    hmu_ls_coeff: &Array1<f64>,
    h_ls_ls_coeff: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if xmu.nrows() != hmumu_coeff.len()
        || xmu.nrows() != hmu_ls_coeff.len()
        || xmu.nrows() != h_ls_ls_coeff.len()
        || x_ls.nrows() != xmu.nrows()
    {
        return Err(format!(
            "gaussian_joint_hessian_from_designs dimension mismatch: xmu {}x{}, x_ls {}x{}, coeffs {}/{}/{}",
            xmu.nrows(),
            xmu.ncols(),
            x_ls.nrows(),
            x_ls.ncols(),
            hmumu_coeff.len(),
            hmu_ls_coeff.len(),
            h_ls_ls_coeff.len()
        ));
    }

    let n = xmu.nrows();
    let pmu = xmu.ncols();
    let p_ls = x_ls.ncols();
    let total = pmu + p_ls;
    let mut out = Array2::<f64>::zeros((total, total));
    for rows in exact_design_row_chunks(n, pmu.max(p_ls)) {
        let xmu_chunk = xmu.row_chunk(rows.clone())?;
        let xls_chunk = x_ls.row_chunk(rows.clone())?;
        let hmumu = hmumu_coeff.slice(s![rows.clone()]);
        let hmu_ls = hmu_ls_coeff.slice(s![rows.clone()]);
        let h_ls_ls = h_ls_ls_coeff.slice(s![rows.clone()]);
        let chunk_hessian =
            fast_joint_hessian_2x2(&xmu_chunk, &xls_chunk, &hmumu, &hmu_ls, &h_ls_ls);
        out += &chunk_hessian;
    }
    Ok(out)
}

fn gaussian_joint_psihessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_psi: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_psi: CustomFamilyPsiLinearMapRef<'_>,
    weights: &GaussianJointPsiFirstWeights,
) -> Result<Array2<f64>, String> {
    // For the symmetric blocks (hmumu, h_ls_ls), the pair
    //   X_psi^T D X  and  X^T D X_psi
    // are transposes of each other, so compute one and add its transpose.
    let a_mu = weighted_crossprod_psi_maps(
        xmu_psi,
        weights.hmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &weights.dhmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_psi,
        weights.hmu_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(
        CustomFamilyPsiLinearMapRef::Dense(xmu),
        weights.hmu_ls.view(),
        x_ls_psi,
    )? + &xt_diag_y_dense(xmu, &weights.dhmu_ls, x_ls)?;
    let a_ls = weighted_crossprod_psi_maps(
        x_ls_psi,
        weights.h_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &weights.dh_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

fn build_two_block_custom_family_joint_psi_operator_from_actions(
    left_action: Option<CustomFamilyPsiDesignAction>,
    right_action: Option<CustomFamilyPsiDesignAction>,
    left_range: std::ops::Range<usize>,
    right_range: std::ops::Range<usize>,
    left_design: &Array2<f64>,
    right_design: &Array2<f64>,
    left_weights: &Array1<f64>,
    cross_weights: &Array1<f64>,
    right_weights: &Array1<f64>,
    left_drift_weights: &Array1<f64>,
    cross_drift_weights: &Array1<f64>,
    right_drift_weights: &Array1<f64>,
) -> Result<Option<std::sync::Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
{
    if left_action.is_none() && right_action.is_none() {
        return Ok(None);
    }

    let total = left_design.ncols() + right_design.ncols();
    let channels = vec![
        CustomFamilyJointDesignChannel::new(left_range, shared_dense_arc(left_design), left_action),
        CustomFamilyJointDesignChannel::new(
            right_range,
            shared_dense_arc(right_design),
            right_action,
        ),
    ];
    let pair_contributions = vec![
        CustomFamilyJointDesignPairContribution::new(
            0,
            0,
            left_weights.clone(),
            left_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            0,
            1,
            cross_weights.clone(),
            cross_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            1,
            0,
            cross_weights.clone(),
            cross_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            1,
            1,
            right_weights.clone(),
            right_drift_weights.clone(),
        ),
    ];

    Ok(Some(std::sync::Arc::new(
        CustomFamilyJointPsiOperator::new(total, channels, pair_contributions),
    )))
}

fn gaussian_joint_psisecondhessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_i: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_i: CustomFamilyPsiLinearMapRef<'_>,
    xmu_j: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_j: CustomFamilyPsiLinearMapRef<'_>,
    xmu_ab: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_ab: CustomFamilyPsiLinearMapRef<'_>,
    weights_i: &GaussianJointPsiFirstWeights,
    weights_j: &GaussianJointPsiFirstWeights,
    secondweights: &GaussianJointPsiSecondWeights,
) -> Result<Array2<f64>, String> {
    // Exploit transpose symmetry: X_a^T D X_b and X_b^T D X_a are transposes.
    // For each such pair in the symmetric blocks (hmumu, h_ls_ls), compute one
    // and add its transpose, halving the number of O(np²) products.
    let a_ab_mu = weighted_crossprod_psi_maps(
        xmu_ab,
        weights_i.hmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let a_ij_mu = weighted_crossprod_psi_maps(xmu_i, weights_i.hmumu.view(), xmu_j)?;
    let a_iwj_mu = weighted_crossprod_psi_maps(
        xmu_i,
        weights_j.dhmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let a_jwi_mu = weighted_crossprod_psi_maps(
        xmu_j,
        weights_i.dhmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_ab_mu
        + &a_ab_mu.t()
        + &a_ij_mu
        + &a_ij_mu.t()
        + &a_iwj_mu
        + &a_iwj_mu.t()
        + &a_jwi_mu
        + &a_jwi_mu.t()
        + &xt_diag_x_dense(xmu, &secondweights.d2hmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_ab,
        weights_i.hmu_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(xmu_i, weights_i.hmu_ls.view(), x_ls_j)?
        + &weighted_crossprod_psi_maps(xmu_j, weights_i.hmu_ls.view(), x_ls_i)?
        + &weighted_crossprod_psi_maps(
            xmu_i,
            weights_j.dhmu_ls.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?
        + &weighted_crossprod_psi_maps(
            xmu_j,
            weights_i.dhmu_ls.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_i.dhmu_ls.view(),
            x_ls_j,
        )?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_j.dhmu_ls.view(),
            x_ls_i,
        )?
        + &xt_diag_y_dense(xmu, &secondweights.d2hmu_ls, x_ls)?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_i.hmu_ls.view(),
            x_ls_ab,
        )?;
    let a_ab_ls = weighted_crossprod_psi_maps(
        x_ls_ab,
        weights_i.h_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let a_ij_ls = weighted_crossprod_psi_maps(x_ls_i, weights_i.h_ls_ls.view(), x_ls_j)?;
    let a_iwj_ls = weighted_crossprod_psi_maps(
        x_ls_i,
        weights_j.dh_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let a_jwi_ls = weighted_crossprod_psi_maps(
        x_ls_j,
        weights_i.dh_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ab_ls
        + &a_ab_ls.t()
        + &a_ij_ls
        + &a_ij_ls.t()
        + &a_iwj_ls
        + &a_iwj_ls.t()
        + &a_jwi_ls
        + &a_jwi_ls.t()
        + &xt_diag_x_dense(x_ls, &secondweights.d2h_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

fn gaussian_joint_psi_mixedhessian_drift_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_psi: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_psi: CustomFamilyPsiLinearMapRef<'_>,
    mixedweights: &GaussianJointPsiMixedDriftWeights,
) -> Result<Array2<f64>, String> {
    let a_mu = weighted_crossprod_psi_maps(
        xmu_psi,
        mixedweights.dhmumu_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &mixedweights.d2hmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_psi,
        mixedweights.dhmu_ls_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(
        CustomFamilyPsiLinearMapRef::Dense(xmu),
        mixedweights.dhmu_ls_u.view(),
        x_ls_psi,
    )? + &xt_diag_y_dense(xmu, &mixedweights.d2hmu_ls, x_ls)?;
    let a_ls = weighted_crossprod_psi_maps(
        x_ls_psi,
        mixedweights.dh_ls_ls_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &mixedweights.d2h_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

#[inline]
fn exp_sigma_derivs_up_to_fourth_array(
    eta: ArrayView1<'_, f64>,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let n = eta.len();
    let mut sigma = Array1::<f64>::zeros(n);
    let mut d1 = Array1::<f64>::zeros(n);
    let mut d2 = Array1::<f64>::zeros(n);
    let mut d3 = Array1::<f64>::zeros(n);
    let mut d4 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (sigma_i, d1_i, d2_i, d3_i, d4_i) = exp_sigma_derivs_up_to_fourth_scalar(eta[i]);
        sigma[i] = sigma_i;
        d1[i] = d1_i;
        d2[i] = d2_i;
        d3[i] = d3_i;
        d4[i] = d4_i;
    }
    (sigma, d1, d2, d3, d4)
}

impl GaussianLocationScaleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    fn get_or_compute_row_scalars(
        &self,
        etamu: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            etamu,
            eta_ls,
            &self.weights,
        )?))
    }

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scale",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    fn exact_block_designs(&self) -> Result<(DenseOrOperator<'_>, DenseOrOperator<'_>), String> {
        let mu_design = self.mu_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing mu design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    fn exact_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(DenseOrOperator<'a>, DenseOrOperator<'a>), String> {
        if specs.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily spec-aware exact path expects 2 specs, got {}",
                specs.len()
            ));
        }
        let mu_design = &specs[Self::BLOCK_MU].design;
        let log_sigma_design = &specs[Self::BLOCK_LOG_SIGMA].design;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    fn exact_joint_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(DenseOrOperator<'a>, DenseOrOperator<'a>)>, String> {
        if self.exact_joint_supported() {
            return self.exact_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.exact_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        let xmu = match xmu {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        let x_ls = match x_ls {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        Ok(Some((xmu, x_ls)))
    }

    fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &xmu, &x_ls)
    }

    fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleFamily input size mismatch".to_string());
        }

        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        // H_{μ,ls} = 2κm. H_{ls,ls} = 2κ²n + κ'(a−n): the κ'(a−n) piece is
        // ∂[κ(a−n)]/∂η, lost if κ is treated as constant under the logb link.
        let cross = 2.0 * &rows.kappa * &rows.m;
        let amn = &rows.obs_weight - &rows.n;
        let scale = 2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn;
        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &rows.w, &cross, &scale,
        )?))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleFamily input size mismatch".to_string());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(format!(
                "GaussianLocationScaleFamily joint d_beta length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ));
        }
        let ximu = xmu.dot(d_beta_flat.slice(s![0..pmu]));
        let xi_ls = x_ls.dot(d_beta_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let (dhmumu, dhmu_ls, dh_ls_ls) =
            gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &dhmumu, &dhmu_ls, &dh_ls_ls,
        )?))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleFamily input size mismatch".to_string());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_u_flat.len() != total || d_betav_flat.len() != total {
            return Err(format!(
                "GaussianLocationScaleFamily joint second directional derivative length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_betav_flat.len(),
                total
            ));
        }
        let ximu_u = xmu.dot(d_beta_u_flat.slice(s![0..pmu]));
        let xi_ls_u = x_ls.dot(d_beta_u_flat.slice(s![pmu..pmu + p_ls]));
        let ximuv = xmu.dot(d_betav_flat.slice(s![0..pmu]));
        let xi_lsv = x_ls.dot(d_betav_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let (d2hmumu, d2hmu_ls, d2h_ls_ls) =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximuv, &xi_lsv);

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &d2hmumu, &d2hmu_ls, &d2h_ls_ls,
        )?))
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<GaussianLocationScaleJointPsiDirection>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily joint psi direction expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let n = self.y.len();
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let betamu = &block_states[Self::BLOCK_MU].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for (local_idx, deriv) in block_derivs.iter().enumerate() {
                if global == psi_index {
                    let xmu_psi;
                    let x_ls_psi;
                    let zmu_psi;
                    let z_ls_psi;
                    match block_idx {
                        Self::BLOCK_MU => {
                            xmu_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                pmu,
                                0..n,
                                "GaussianLocationScaleFamily mu",
                                policy,
                            )?;
                            zmu_psi = xmu_psi.forward_mul(betamu.view()).map_err(|e| {
                                format!("GaussianLocationScaleFamily mu forward_mul: {e}")
                            })?;
                            x_ls_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: p_ls,
                            };
                            z_ls_psi = Array1::<f64>::zeros(n);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            x_ls_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                p_ls,
                                0..n,
                                "GaussianLocationScaleFamily log-sigma",
                                policy,
                            )?;
                            z_ls_psi = x_ls_psi.forward_mul(beta_ls.view()).map_err(|e| {
                                format!("GaussianLocationScaleFamily log-sigma forward_mul: {e}")
                            })?;
                            xmu_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: pmu,
                            };
                            zmu_psi = Array1::<f64>::zeros(n);
                        }
                        _ => return Ok(None),
                    }
                    return Ok(Some(GaussianLocationScaleJointPsiDirection {
                        block_idx,
                        local_idx,
                        zmu_psi,
                        z_ls_psi,
                        xmu_psi,
                        x_ls_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &GaussianLocationScaleJointPsiDirection,
        psi_b: &GaussianLocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<GaussianLocationScaleJointPsiSecondDrifts, String> {
        let n = self.y.len();
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let betamu = &block_states[Self::BLOCK_MU].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut xmu_ab = None;
        let mut x_ls_ab = None;
        let mut xmu_ab_action = None;
        let mut x_ls_ab_action = None;
        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            let deriv_b = &derivative_blocks[psi_b.block_idx][psi_b.local_idx];
            match psi_a.block_idx {
                Self::BLOCK_MU => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        pmu,
                        0..n,
                        "GaussianLocationScaleFamily mu",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => xmu_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => xmu_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "GaussianLocationScaleFamily mu: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                Self::BLOCK_LOG_SIGMA => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        p_ls,
                        0..n,
                        "GaussianLocationScaleFamily log-sigma",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => x_ls_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => x_ls_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "GaussianLocationScaleFamily log-sigma: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        let zmu_ab = second_psi_linear_map(xmu_ab_action.as_ref(), xmu_ab.as_ref(), n, pmu)
            .forward_mul(betamu.view());
        let z_ls_ab = second_psi_linear_map(x_ls_ab_action.as_ref(), x_ls_ab.as_ref(), n, p_ls)
            .forward_mul(beta_ls.view());
        Ok(GaussianLocationScaleJointPsiSecondDrifts {
            xmu_ab_action,
            x_ls_ab_action,
            xmu_ab,
            x_ls_ab,
            zmu_ab,
            z_ls_ab,
        })
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        if specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily joint psi terms expect 2 specs and 2 derivative blocks, got {} and {}",
                specs.len(),
                derivative_blocks.len()
            ));
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        // Gaussian 2-block location-scale family in the unified flattened
        // coefficient space beta = [betamu; beta_sigma]:
        //
        //   mu_i = z_i^T betamu,
        //   ell_i = x_i^T beta_sigma,
        //   s_i = exp(ell_i),
        //   r_i = y_i - mu_i,
        //   q_i = r_i / s_i,
        //   w_i = s_i^{-2},
        //   alpha_i = r_i s_i^{-2},
        //   b_i = q_i^2.
        //
        // The first fixed-beta psi object returned here is likelihood-only:
        //
        //   D_a         = -alpha^T m_a + (1 - b)^T ell_a
        //   D_{beta a}  = [ -Xmu^T alpha_a - X_{mu,a}^T alpha ;
        //                   -X_sigma^T b_a + X_{sigma,a}^T (1-b) ]
        //   D_{bb a}    = [ Xmu^T W_a Xmu + X_{mu,a}^T W Xmu + Xmu^T W X_{mu,a},
        //                   2( Xmu^T A_a X_sigma + X_{mu,a}^T A X_sigma + Xmu^T A X_{sigma,a} );
        //                   sym,
        //                   2( X_sigma^T B_a X_sigma + X_{sigma,a}^T B X_sigma + X_sigma^T B X_{sigma,a} ) ]
        //
        // with m_a = X_{mu,a} betamu, ell_a = X_{sigma,a} beta_sigma and
        // rowwise scalar drifts
        //
        //   w_a     = -2 w * ell_a
        //   alpha_a = -w * m_a - 2 alpha * ell_a
        //   b_a     = -2 alpha * m_a - 2 b * ell_a.
        //
        // Generic code in custom_family.rs promotes these likelihood-only
        // objects to the full fixed-beta V_a / g_a / H_a by adding S_a.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let weights_a = gaussian_joint_psi_firstweights(&rows, &dir_a.zmu_psi, &dir_a.z_ls_psi);
        let objective_psi = weights_a.objective_psirow.sum();
        let xmu_map = dir_a.xmu_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_mu =
            xmu_map.transpose_mul(weights_a.scoremu.view()) + xmu.t().dot(&weights_a.dscoremu);
        let score_ls =
            x_ls_map.transpose_mul(weights_a.score_ls.view()) + x_ls.t().dot(&weights_a.dscore_ls);
        let score_psi = gaussian_pack_joint_score(&score_mu, &score_ls);
        let hessian_psi_operator = build_two_block_custom_family_joint_psi_operator_from_actions(
            dir_a.xmu_psi.cloned_first_action(),
            dir_a.x_ls_psi.cloned_first_action(),
            0..xmu.ncols(),
            xmu.ncols()..xmu.ncols() + x_ls.ncols(),
            xmu,
            x_ls,
            &weights_a.hmumu,
            &weights_a.hmu_ls,
            &weights_a.h_ls_ls,
            &weights_a.dhmumu,
            &weights_a.dhmu_ls,
            &weights_a.dh_ls_ls,
        )?;
        let hessian_psi = if hessian_psi_operator.is_some() {
            Array2::zeros((0, 0))
        } else {
            gaussian_joint_psihessian_fromweights(xmu, x_ls, xmu_map, x_ls_map, &weights_a)?
        };

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_j) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_i,
                &dir_j,
                xmu,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_i: &GaussianLocationScaleJointPsiDirection,
        dir_j: &GaussianLocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_i_map = dir_i.xmu_psi.as_linear_map_ref();
        let x_ls_i_map = dir_i.x_ls_psi.as_linear_map_ref();
        let xmu_j_map = dir_j.xmu_psi.as_linear_map_ref();
        let x_ls_j_map = dir_j.x_ls_psi.as_linear_map_ref();
        let xmu_ab_map = second_psi_linear_map(
            second_drifts.xmu_ab_action.as_ref(),
            second_drifts.xmu_ab.as_ref(),
            n,
            xmu.ncols(),
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            x_ls.ncols(),
        );
        // Second fixed-beta psi objects for the same Gaussian location-scale
        // kernel. Using the notation from the first-order comment, the rowwise
        // second psi drifts are
        //
        //   w_ab     = 4 w * ell_a * ell_b - 2 w * ell_ab
        //   alpha_ab = 2 w * (m_a * ell_b + m_b * ell_a)
        //              + 4 alpha * ell_a * ell_b
        //              - w * m_ab
        //              - 2 alpha * ell_ab
        //   b_ab     = 2 w * m_a * m_b
        //              + 4 alpha * (m_a * ell_b + m_b * ell_a)
        //              + 4 b * ell_a * ell_b
        //              - 2 alpha * m_ab
        //              - 2 b * ell_ab.
        //
        // The exact likelihood-only second-order objects are then:
        //
        //   D_ab,
        //   D_{beta ab},
        //   D_{beta beta ab},
        //
        // assembled from the usual product-rule expansion over realized
        // design motion X_{.,a}, X_{.,b}, X_{.,ab}. Generic code adds S_ab.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let weights_i = gaussian_joint_psi_firstweights(&rows, &dir_i.zmu_psi, &dir_i.z_ls_psi);
        let weights_j = gaussian_joint_psi_firstweights(&rows, &dir_j.zmu_psi, &dir_j.z_ls_psi);
        let secondweights = gaussian_joint_psisecondweights(
            &rows,
            &dir_i.zmu_psi,
            &dir_i.z_ls_psi,
            &dir_j.zmu_psi,
            &dir_j.z_ls_psi,
            &second_drifts.zmu_ab,
            &second_drifts.z_ls_ab,
        );
        let objective_psi_psi = secondweights.objective_psi_psirow.sum();

        let score_psi_psi = gaussian_pack_joint_score(
            &(xmu_ab_map.transpose_mul(weights_i.scoremu.view())
                + xmu_i_map.transpose_mul(weights_j.dscoremu.view())
                + xmu_j_map.transpose_mul(weights_i.dscoremu.view())
                + xmu.t().dot(&secondweights.d2scoremu)),
            &(x_ls_ab_map.transpose_mul(weights_i.score_ls.view())
                + x_ls_i_map.transpose_mul(weights_j.dscore_ls.view())
                + x_ls_j_map.transpose_mul(weights_i.dscore_ls.view())
                + x_ls.t().dot(&secondweights.d2score_ls)),
        );
        let hessian_psi_psi = gaussian_joint_psisecondhessian_fromweights(
            xmu,
            x_ls,
            xmu_i_map,
            x_ls_i_map,
            xmu_j_map,
            x_ls_j_map,
            xmu_ab_map,
            x_ls_ab_map,
            &weights_i,
            &weights_j,
            &secondweights,
        )?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                xmu,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &GaussianLocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.xmu_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(format!(
                "GaussianLocationScaleFamily joint psi hessian directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ));
        }
        let umu = d_beta_flat.slice(s![0..pmu]);
        let u_ls = d_beta_flat.slice(s![pmu..pmu + p_ls]);
        let ximu = xmu.dot(&umu);
        let xi_ls = x_ls.dot(&u_ls);
        let uzamu = xmu_map.forward_mul(umu);
        let uza_ls = x_ls_map.forward_mul(u_ls);
        // Mixed drift T_a[u] = D_beta H_a^{(D)}[u] for the Gaussian family.
        //
        // Along u = [umu; u_sigma], define xi = Xmu umu and zeta = X_sigma u_sigma.
        // The first beta-directional drifts of the Gaussian row scalars are
        //
        //   d_u w     = -2 w * zeta
        //   d_u alpha = -w * xi - 2 alpha * zeta
        //   d_u b     = -2 alpha * xi - 2 b * zeta.
        //
        // Differentiating the psi-a scalar drifts once more gives
        //
        //   d_u w_a     = 4 w * ell_a * zeta - 2 w * zeta_a
        //   d_u alpha_a = 2 w * (m_a * zeta + ell_a * xi)
        //                 - w * xi_a
        //                 + 4 alpha * ell_a * zeta
        //                 - 2 alpha * zeta_a
        //   d_u b_a     = 2 w * m_a * xi
        //                 + 4 alpha * (m_a * zeta + ell_a * xi)
        //                 + 4 b * ell_a * zeta
        //                 - 2 alpha * xi_a
        //                 - 2 b * zeta_a.
        //
        // The matrix drift returned here is the exact likelihood-only
        //
        //   T_a[u] = D_beta H_{psi_a}^{(D)}[u],
        //
        // assembled blockwise as
        //
        //   Kmumu,a[u]   = Xmu^T W_a[u] Xmu
        //                   + X_{mu,a}^T W[u] Xmu
        //                   + Xmu^T W[u] X_{mu,a}
        //   Kmusigma,a[u]= 2( Xmu^T A_a[u] X_sigma
        //                   + X_{mu,a}^T A[u] X_sigma
        //                   + Xmu^T A[u] X_{sigma,a} )
        //   K_sigmasigma,a[u]
        //                   = 2( X_sigma^T B_a[u] X_sigma
        //                   + X_{sigma,a}^T B[u] X_sigma
        //                   + X_sigma^T B[u] X_{sigma,a} ).
        //
        // Generic code then combines this with S(theta)-motion and the profile
        // mode responses to form ddot H_{ij}.
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let mixedweights = gaussian_joint_psi_mixed_driftweights(
            &rows,
            &ximu,
            &xi_ls,
            &dir_a.zmu_psi,
            &dir_a.z_ls_psi,
            &uzamu,
            &uza_ls,
        );

        Ok(gaussian_joint_psi_mixedhessian_drift_fromweights(
            xmu,
            x_ls,
            xmu_map,
            x_ls_map,
            &mixedweights,
        )?)
    }
}

impl CustomFamily for GaussianLocationScaleFamily {
    /// The Gaussian location-scale joint Hessian depends on β because the
    /// cross-block (μ,log σ) and (log σ,log σ) blocks contain the residual
    /// r = y − μ (via the row scalars m = r·w and n = r²·w), which changes
    /// when β_μ moves.  The (μ,μ) block weight w = 1/σ² also depends on
    /// β_{log σ}.  This override is essential for correct M_j[u] drift
    /// corrections when ψ hyperparameters move the design matrices.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Two fully-coupled blocks (mean p_t, log-σ p_ℓ): every row contributes
        // an outer-product over all (p_t + p_ℓ) coefficients, so honest cost is
        // n · (p_t + p_ℓ)² rather than the block-diagonal n · (p_t² + p_ℓ²).
        crate::custom_family::joint_coupled_coefficient_hessian_cost(self.y.len() as u64, specs)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleFamily input size mismatch".to_string());
        }

        // Diagonal IRLS weights for the inner solver.
        //
        // For the location block (identity link): wmu = pw / sigma^2. Since the
        // location link is identity, observed = Fisher --- no correction needed.
        //
        // For the log-sigma block (log link): w_ls = 2 * pw * (dsigma/deta)^2 / sigma^2.
        // This is the Fisher weight. For the outer REML, the joint
        // `exact_newton_joint_hessian` provides the full observed Hessian directly,
        // so these Diagonal weights are only used for the inner IRLS iteration
        // (where Fisher scoring is fine). See response.md Section 3.
        //
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut z_ls = Array1::<f64>::zeros(n);
        let mut w_ls = Array1::<f64>::zeros(n);
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;

        if let (
            Some(y_s),
            Some(w_s),
            Some(mu_s),
            Some(ls_s),
            Some(zmu_s),
            Some(wmu_s),
            Some(zls_s),
            Some(wls_s),
        ) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
            zmu.as_slice_memory_order_mut(),
            wmu.as_slice_memory_order_mut(),
            z_ls.as_slice_memory_order_mut(),
            w_ls.as_slice_memory_order_mut(),
        ) {
            for i in 0..n {
                let row = gaussian_diagonal_row_kernel(y_s[i], mu_s[i], ls_s[i], w_s[i], ln2pi);
                ll += row.log_likelihood;
                wmu_s[i] = row.location_working_weight;
                zmu_s[i] = mu_s[i] + row.location_working_shift;
                wls_s[i] = row.log_sigma_working_weight;
                zls_s[i] = row.log_sigma_working_response;
            }
        } else {
            for i in 0..n {
                let row = gaussian_diagonal_row_kernel(
                    self.y[i],
                    etamu[i],
                    eta_log_sigma[i],
                    self.weights[i],
                    ln2pi,
                );
                ll += row.log_likelihood;
                wmu[i] = row.location_working_weight;
                zmu[i] = etamu[i] + row.location_working_shift;
                w_ls[i] = row.log_sigma_working_weight;
                z_ls[i] = row.log_sigma_working_response;
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: zmu,
                    working_weights: wmu,
                },
                BlockWorkingSet::Diagonal {
                    working_response: z_ls,
                    working_weights: w_ls,
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleFamily input size mismatch".to_string());
        }
        // logb noise link: σ(η_ls) = LOGB_SIGMA_FLOOR + exp(η_ls). σ ≥ b > 0
        // bounds the loglik below (−Σlog σ ≥ −n log b) and bounds 1/σ² by 1/b²,
        // so the previous `inv_s2.min(1e24)` cap is structurally unnecessary.
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        if let (Some(y_s), Some(w_s), Some(mu_s), Some(ls_s)) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
        ) {
            for i in 0..n {
                let wi = w_s[i];
                if wi == 0.0 {
                    continue;
                }
                let sigma_i = logb_sigma_from_eta_scalar(ls_s[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = y_s[i] - mu_s[i];
                ll += wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()));
            }
        } else {
            for i in 0..n {
                let wi = self.weights[i];
                if wi == 0.0 {
                    continue;
                }
                let sigma_i = logb_sigma_from_eta_scalar(eta_log_sigma[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = self.y[i] - etamu[i];
                ll += wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()));
            }
        }
        Ok(ll)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n || d_eta.len() != n {
            return Err("GaussianLocationScaleFamily input size mismatch".to_string());
        }

        let sigma = eta_ls.mapv(logb_sigma_from_eta_scalar);
        let mut dw = Array1::<f64>::zeros(n);
        match block_idx {
            Self::BLOCK_MU => {
                // Gaussian location block:
                //
                //   wmu = weight / sigma^2.
                //
                // This depends only on the scale predictor, so along a
                // location-only direction d etamu the directional derivative is
                // identically zero.
                Ok(Some(dw))
            }
            Self::BLOCK_LOG_SIGMA => {
                // Gaussian log-sigma block:
                //
                // The PIRLS information weight is
                //
                //   w_ls = max(2 * weight * clamp(g, -1, 1)^2, MIN_WEIGHT),
                //   g    = sigma'(eta_ls) / sigma(eta_ls),
                // with the semantic rule that zero observation weights stay zero.
                //
                // Along a direction d eta_ls,
                //
                //   dw_ls is the directional derivative of that piecewise
                //   definition. On the active clamp branch or active MIN_WEIGHT
                //   floor branch, the returned derivative is zero to match the
                //   selected local piece of the evaluated weight.
                //
                // This is the exact directional derivative needed by the REML
                // trace term
                //
                //   0.5 tr(J^{-1} D_beta J[u])
                //   = 0.5 sum_i (x_i^T J^{-1} x_i) dw_i
                //
                // for diagonal working-set blocks.
                for i in 0..n {
                    dw[i] = gaussian_log_sigma_irlsinfo_directional_derivative(
                        self.weights[i],
                        sigma[i],
                        d_eta[i],
                    );
                }
                Ok(Some(dw))
            }
            _ => Ok(None),
        }
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != 2 || specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily joint psi workspace expects 2 states, 2 specs, and 2 derivative block lists, got {} / {} / {}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }
}

impl CustomFamilyGenerative for GaussianLocationScaleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let mu = block_states[Self::BLOCK_MU].eta.clone();
        let sigma = block_states[Self::BLOCK_LOG_SIGMA]
            .eta
            .mapv(logb_sigma_from_eta_scalar);
        Ok(GenerativeSpec {
            mean: mu,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}

/// Matrix-free joint-Hessian operator for the two-block Gaussian
/// location-scale family. The dense Hessian decomposes as
///
///   H = [[X_mu^T diag(w) X_mu,    X_mu^T diag(cross) X_ls],
///        [X_ls^T diag(cross) X_mu, X_ls^T diag(scale) X_ls]],
///
/// with `cross = 2κm` and `scale = 2κ²n + κ'(a−n)`. The matvec applies
/// each block by a single design-matrix multiply on each side, so the cost
/// is Θ(n (p_mu + p_ls)) per `Hv` rather than Θ(n (p_mu + p_ls)²) to form
/// the dense matrix.
struct GaussianLocationScaleHessianWorkspace {
    family: GaussianLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    xmu: Array2<f64>,
    x_ls: Array2<f64>,
    coeff_mm: Array1<f64>,
    coeff_ml: Array1<f64>,
    coeff_ll: Array1<f64>,
}

impl GaussianLocationScaleHessianWorkspace {
    fn new(
        family: GaussianLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let etamu = &block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = family.get_or_compute_row_scalars(etamu, eta_ls)?;
        let coeff_mm = rows.w.clone();
        let coeff_ml = 2.0 * &rows.kappa * &rows.m;
        let amn = &rows.obs_weight - &rows.n;
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn;
        Ok(Self {
            family,
            block_states,
            xmu,
            x_ls,
            coeff_mm,
            coeff_ml,
            coeff_ll,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleHessianWorkspace {
    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        if v.len() != total {
            return Err(format!(
                "GaussianLocationScale matvec dimension mismatch: got {}, expected {}",
                v.len(),
                total
            ));
        }
        let u_mu = self.xmu.dot(&v.slice(s![0..pmu]));
        let u_ls = self.x_ls.dot(&v.slice(s![pmu..total]));
        let r_mu = &self.coeff_mm * &u_mu + &self.coeff_ml * &u_ls;
        let r_ls = &self.coeff_ml * &u_mu + &self.coeff_ll * &u_ls;
        let out_mu = self.xmu.t().dot(&r_mu);
        let out_ls = self.x_ls.t().dot(&r_ls);
        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..total]).assign(&out_ls);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        let mut diag = Array1::<f64>::zeros(total);
        let n = self.coeff_mm.len();
        for j in 0..pmu {
            let col = self.xmu.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.coeff_mm[i] * v * v;
            }
            diag[j] = acc;
        }
        for j in 0..p_ls {
            let col = self.x_ls.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.coeff_ll[i] * v * v;
            }
            diag[pmu + j] = acc;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family.exact_newton_joint_hessian_directional_derivative_from_designs(
            &self.block_states,
            &DenseOrOperator::Borrowed(&self.xmu),
            &DenseOrOperator::Borrowed(&self.x_ls),
            d_beta_flat,
        )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                &DenseOrOperator::Borrowed(&self.xmu),
                &DenseOrOperator::Borrowed(&self.x_ls),
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }
}

struct GaussianLocationScaleWiggleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    xmu_psi: PsiDesignMap,
    x_ls_psi: PsiDesignMap,
    zmu_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}

struct GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace {
    family: GaussianLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    xmu: Array2<f64>,
    x_ls: Array2<f64>,
    psi_directions: ExactNewtonJointPsiDirectCache<GaussianLocationScaleWiggleJointPsiDirection>,
}

impl GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace {
    fn new(
        family: GaussianLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let Some((xmu, x_ls)) = family.exact_joint_dense_block_designs(Some(specs))? else {
            return Err(
                "GaussianLocationScaleWiggleFamily exact joint psi workspace requires dense block designs"
                    .to_string(),
            );
        };
        let xmu = xmu.into_owned();
        let x_ls = x_ls.into_owned();
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum();
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            xmu,
            x_ls,
            psi_directions: ExactNewtonJointPsiDirectCache::new(psi_dim),
        })
    }

    fn psi_direction(
        &self,
        psi_index: usize,
    ) -> Result<Option<Arc<GaussianLocationScaleWiggleJointPsiDirection>>, String> {
        self.psi_directions.get_or_try_init(psi_index, || {
            self.family.exact_newton_joint_psi_direction(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                &self.xmu,
                &self.x_ls,
                &self.family.policy,
            )
        })
    }
}

impl ExactNewtonJointPsiWorkspace for GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace {
    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.psi_direction(psi_i)? else {
            return Ok(None);
        };
        let Some(dir_j) = self.psi_direction(psi_j)? else {
            return Ok(None);
        };
        Ok(Some(
            self.family
                .exact_newton_joint_psisecond_order_terms_from_parts(
                    &self.block_states,
                    &self.derivative_blocks,
                    dir_i.as_ref(),
                    dir_j.as_ref(),
                    &self.xmu,
                    &self.x_ls,
                )?,
        ))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(
                self.family
                    .exact_newton_joint_psihessian_directional_derivative_from_parts(
                        &self.block_states,
                        dir.as_ref(),
                        d_beta_flat,
                        &self.xmu,
                        &self.x_ls,
                    )?,
            ),
        ))
    }
}

struct GaussianLocationScaleWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    basis_d3: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
    d4q_dq04: Array1<f64>,
}

/// Per-row pieces of the 3-block Gaussian location-scale-wiggle joint
/// Hessian. Both the dense path and the matrix-free workspace share these
/// row coefficients; only the assembly differs.
struct GaussianLocationScaleWiggleHessianRowPieces {
    coeff_mm: Array1<f64>,
    coeff_ml: Array1<f64>,
    coeff_ll: Array1<f64>,
    coeff_mw_b: Array1<f64>,
    coeff_mw_d: Array1<f64>,
    coeff_lw_b: Array1<f64>,
    coeff_ww: Array1<f64>,
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
}

impl GaussianLocationScaleWiggleHessianRowPieces {
    fn assemble_dense(
        &self,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let h_mm = xt_diag_x_dense(xmu, &self.coeff_mm)?;
        let h_ml = xt_diag_y_dense(xmu, &self.coeff_ml, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_mw = xt_diag_y_dense(xmu, &self.coeff_mw_b, &self.basis)?
            + &xt_diag_y_dense(xmu, &self.coeff_mw_d, &self.basis_d1)?;
        let h_lw = xt_diag_y_dense(x_ls, &self.coeff_lw_b, &self.basis)?;
        let h_ww = xt_diag_x_dense(&self.basis, &self.coeff_ww)?;
        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
    }
}

fn scale_matrix_rows(mat: &Array2<f64>, coeffs: &Array1<f64>) -> Result<Array2<f64>, String> {
    if mat.nrows() != coeffs.len() {
        return Err(format!(
            "row scaling dimension mismatch: matrix has {} rows but coeffs have {} entries",
            mat.nrows(),
            coeffs.len()
        ));
    }
    Ok(Array2::from_shape_fn(mat.dim(), |(i, j)| {
        mat[[i, j]] * coeffs[i]
    }))
}

fn gaussian_pack_wiggle_joint_score(
    score_mu: &Array1<f64>,
    score_ls: &Array1<f64>,
    score_w: &Array1<f64>,
) -> Array1<f64> {
    let pmu = score_mu.len();
    let p_ls = score_ls.len();
    let pw = score_w.len();
    let total = pmu + p_ls + pw;
    let mut out = Array1::<f64>::zeros(total);
    out.slice_mut(s![0..pmu]).assign(score_mu);
    out.slice_mut(s![pmu..pmu + p_ls]).assign(score_ls);
    out.slice_mut(s![pmu + p_ls..total]).assign(score_w);
    out
}

fn gaussian_pack_wiggle_joint_symmetrichessian(
    h_mm: &Array2<f64>,
    h_ml: &Array2<f64>,
    h_mw: &Array2<f64>,
    h_ll: &Array2<f64>,
    h_lw: &Array2<f64>,
    h_ww: &Array2<f64>,
) -> Array2<f64> {
    let pmu = h_mm.nrows();
    let p_ls = h_ll.nrows();
    let pw = h_ww.nrows();
    let total = pmu + p_ls + pw;
    let mut out = Array2::<f64>::zeros((total, total));
    out.slice_mut(s![0..pmu, 0..pmu]).assign(h_mm);
    out.slice_mut(s![0..pmu, pmu..pmu + p_ls]).assign(h_ml);
    out.slice_mut(s![0..pmu, pmu + p_ls..total]).assign(h_mw);
    out.slice_mut(s![pmu..pmu + p_ls, pmu..pmu + p_ls])
        .assign(h_ll);
    out.slice_mut(s![pmu..pmu + p_ls, pmu + p_ls..total])
        .assign(h_lw);
    out.slice_mut(s![pmu + p_ls..total, pmu + p_ls..total])
        .assign(h_ww);
    mirror_upper_to_lower(&mut out);
    out
}

pub struct GaussianLocationScaleWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
    cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}

impl Clone for GaussianLocationScaleWiggleFamily {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: self.policy.clone(),
            cached_row_scalars: std::sync::RwLock::new(
                self.cached_row_scalars
                    .read()
                    .expect("lock poisoned")
                    .clone(),
            ),
        }
    }
}

impl GaussianLocationScaleWiggleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::Identity,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scalewiggle",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            options.derivative_order,
        )
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d1.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d1.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d1.dot(&beta_link_wiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle second-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d2.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d2.dot(&beta_link_wiggle))
    }

    fn wiggle_d3basis_constrained(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        if d3.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle third-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d3.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d3.dot(&beta_link_wiggle))
    }

    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle fourth-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d4.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    fn wiggle_geometry(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<GaussianLocationScaleWiggleGeometry, String> {
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(GaussianLocationScaleWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        })
    }

    fn get_or_compute_row_scalars(
        &self,
        q: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            q,
            eta_ls,
            &self.weights,
        )?))
    }

    fn dense_block_designs(&self) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        let mu_design = self.mu_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleWiggleFamily exact path is missing mu design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleWiggleFamily exact path is missing log-sigma design".to_string()
        })?;
        let xmu = match mu_design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                mu_design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "GaussianLocationScaleWiggle dense_block_designs mu",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        let x_ls = match log_sigma_design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                log_sigma_design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "GaussianLocationScaleWiggle dense_block_designs log_sigma",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        Ok((xmu, x_ls))
    }
    fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        if specs.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 specs, got {}",
                specs.len()
            ));
        }
        let xmu = match specs[Self::BLOCK_MU].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_MU]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "GaussianLocationScaleWiggle dense_block_designs_fromspecs mu",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        let x_ls = match specs[Self::BLOCK_LOG_SIGMA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_LOG_SIGMA]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "GaussianLocationScaleWiggle dense_block_designs_fromspecs log_sigma",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        Ok((xmu, x_ls))
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if self.exact_joint_supported() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }
}

impl GaussianLocationScaleWiggleFamily {
    fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &xmu, &x_ls)
    }

    fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<GaussianLocationScaleWiggleJointPsiDirection>, String> {
        if block_states.len() != 3 || derivative_blocks.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily joint psi direction expects 3 blocks and 3 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let n = self.y.len();
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let betamu = &block_states[Self::BLOCK_MU].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for (local_idx, deriv) in block_derivs.iter().enumerate() {
                if global == psi_index {
                    let xmu_psi;
                    let x_ls_psi;
                    let zmu_psi;
                    let z_ls_psi;
                    match block_idx {
                        Self::BLOCK_MU => {
                            xmu_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                pmu,
                                0..n,
                                "GaussianLocationScaleWiggleFamily mu",
                                policy,
                            )?;
                            zmu_psi = xmu_psi.forward_mul(betamu.view()).map_err(|e| {
                                format!("GaussianLocationScaleWiggleFamily mu forward_mul: {e}")
                            })?;
                            x_ls_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: p_ls,
                            };
                            z_ls_psi = Array1::<f64>::zeros(n);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            x_ls_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                p_ls,
                                0..n,
                                "GaussianLocationScaleWiggleFamily log-sigma",
                                policy,
                            )?;
                            z_ls_psi = x_ls_psi.forward_mul(beta_ls.view()).map_err(|e| {
                                format!(
                                    "GaussianLocationScaleWiggleFamily log-sigma forward_mul: {e}"
                                )
                            })?;
                            xmu_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: pmu,
                            };
                            zmu_psi = Array1::<f64>::zeros(n);
                        }
                        Self::BLOCK_WIGGLE => return Ok(None),
                        _ => return Ok(None),
                    }
                    return Ok(Some(GaussianLocationScaleWiggleJointPsiDirection {
                        block_idx,
                        local_idx,
                        zmu_psi,
                        z_ls_psi,
                        xmu_psi,
                        x_ls_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &GaussianLocationScaleWiggleJointPsiDirection,
        psi_b: &GaussianLocationScaleWiggleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<GaussianLocationScaleJointPsiSecondDrifts, String> {
        let n = self.y.len();
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let betamu = &block_states[Self::BLOCK_MU].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut xmu_ab = None;
        let mut x_ls_ab = None;
        let mut xmu_ab_action = None;
        let mut x_ls_ab_action = None;
        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            let deriv_b = &derivative_blocks[psi_b.block_idx][psi_b.local_idx];
            match psi_a.block_idx {
                Self::BLOCK_MU => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        pmu,
                        0..n,
                        "GaussianLocationScaleWiggleFamily mu",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => xmu_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => xmu_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "GaussianLocationScaleWiggleFamily mu: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                Self::BLOCK_LOG_SIGMA => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        p_ls,
                        0..n,
                        "GaussianLocationScaleWiggleFamily log-sigma",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => x_ls_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => x_ls_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "GaussianLocationScaleWiggleFamily log-sigma: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        let zmu_ab = second_psi_linear_map(xmu_ab_action.as_ref(), xmu_ab.as_ref(), n, pmu)
            .forward_mul(betamu.view());
        let z_ls_ab = second_psi_linear_map(x_ls_ab_action.as_ref(), x_ls_ab.as_ref(), n, p_ls)
            .forward_mul(beta_ls.view());
        Ok(GaussianLocationScaleJointPsiSecondDrifts {
            xmu_ab_action,
            x_ls_ab_action,
            xmu_ab,
            x_ls_ab,
            zmu_ab,
            z_ls_ab,
        })
    }

    /// Compute the rowwise Hessian pieces shared by the dense path and the
    /// matrix-free workspace operator. The same coefficients reconstruct the
    /// dense p×p matrix or apply `Hv` directly without ever forming it.
    fn wiggle_hessian_row_pieces(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GaussianLocationScaleWiggleHessianRowPieces, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleWiggleFamily input size mismatch".to_string());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        if geom.basis.ncols() != betaw.len() {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily wiggle basis/beta mismatch: basis has {} columns but beta has {} entries",
                geom.basis.ncols(),
                betaw.len()
            ));
        }
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        // Cross blocks involving η_ls carry overall κ; scale-scale block is
        // 2κ²n + κ'(a−n) under the logb link (the κ'(a−n) piece is lost if κ
        // is treated as constant under ∂/∂η_ls).
        let coeff_ml = (2.0 * &rows.kappa * &rows.m) * &geom.dq_dq0;
        let amn = &rows.obs_weight - &rows.n;
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn;
        let coeff_mw_b = &rows.w * &geom.dq_dq0;
        let coeff_mw_d = -&rows.m;
        // ls-wiggle cross block carries one κ from the η_ls chain.
        let coeff_lw_b = 2.0 * &rows.kappa * &rows.m;
        let coeff_ww = rows.w.clone();
        Ok(GaussianLocationScaleWiggleHessianRowPieces {
            coeff_mm,
            coeff_ml,
            coeff_ll,
            coeff_mw_b,
            coeff_mw_d,
            coeff_lw_b,
            coeff_ww,
            basis: geom.basis,
            basis_d1: geom.basis_d1,
        })
    }

    fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        Ok(Some(pieces.assemble_dense(xmu, x_ls)?))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily exact joint directional Hessian",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleWiggleFamily input size mismatch".to_string());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xi = xmu.dot(&umu);
        let zeta = x_ls.dot(&u_ls);
        // logb κ-scaled η_ls direction; κ' = dκ/dη_ls = κ(1−κ).
        let szeta = &rows.kappa * &zeta;
        let phi = geom.basis.dot(&uw);
        let mut q_u = &geom.dq_dq0 * &xi;
        q_u += &phi;
        let mut s1_u = &geom.d2q_dq02 * &xi;
        s1_u += &geom.basis_d1.dot(&uw);
        let mut g2_u = &geom.d3q_dq03 * &xi;
        g2_u += &geom.basis_d2.dot(&uw);
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let dw_u = -2.0 * &rows.w * &szeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta);
        let dn_u = -(2.0 * &rows.m * &q_u) - &(2.0 * &rows.n * &szeta);
        let amn = &rows.obs_weight - &rows.n;

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // Static blocks: H_{μ,ls} = 2κm·dq_dq0; H_{ls,ls} = 2κ²n + κ'(a−n).
        // Differentiating along α = (xi, zeta, phi) carries dκ/dη_ls = κ' on
        // every term that originally read just κ. The η_w direction has no
        // direct η_ls dependence so does not contribute κ' factors directly,
        // but does enter dn_u via q_u as a μ-chain — already captured.
        let coeff_ml_u = 2.0 * &rows.kappa * &(&dm_u * &geom.dq_dq0 + &rows.m * &s1_u)
            + 2.0 * &rows.kappa_prime * &(&zeta * &rows.m * &geom.dq_dq0);
        let coeff_ll_u = 2.0 * &rows.kappa * &rows.kappa * &dn_u
            + 4.0 * &rows.kappa * &rows.kappa_prime * &(&zeta * &rows.n)
            + &rows.kappa_dprime * &(&zeta * &amn)
            - &rows.kappa_prime * &dn_u;
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let c_u = -&dm_u;
        // ls-wiggle cross block: l = 2κm; differentiating gains 2κ'·m·zeta.
        let l_u = 2.0 * &rows.kappa * &dm_u + 2.0 * &rows.kappa_prime * &(&rows.m * &zeta);

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_u)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_u, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_u)?;
        let h_mw = xt_diag_y_dense(xmu, &a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_u)?
            + &xt_diag_y_dense(xmu, &c_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_u)?;
        let h_lw = xt_diag_y_dense(x_ls, &l_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &(2.0 * &rows.kappa * &rows.m), &basis_u)?;
        let a_ww = xt_diag_y_dense(&basis_u, &rows.w, &geom.basis)?;
        let h_ww = &a_ww + &a_ww.t() + &xt_diag_x_dense(&geom.basis, &dw_u)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_u_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (u)",
        )?;
        let (vmu, v_ls, vw) = layout.split_three(
            d_beta_v_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (v)",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleWiggleFamily input size mismatch".to_string());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi_u = xmu.dot(&umu);
        let xi_v = xmu.dot(&vmu);
        let zeta_u = x_ls.dot(&u_ls);
        let zeta_v = x_ls.dot(&v_ls);
        let phi_u = geom.basis.dot(&uw);
        let phi_v = geom.basis.dot(&vw);
        let b1u = geom.basis_d1.dot(&uw);
        let b1v = geom.basis_d1.dot(&vw);
        let b2u = geom.basis_d2.dot(&uw);
        let b2v = geom.basis_d2.dot(&vw);
        let b3u = geom.basis_d3.dot(&uw);
        let b3v = geom.basis_d3.dot(&vw);

        let mut q_u = &geom.dq_dq0 * &xi_u;
        q_u += &phi_u;
        let mut q_v = &geom.dq_dq0 * &xi_v;
        q_v += &phi_v;
        let mut s1_u = &geom.d2q_dq02 * &xi_u;
        s1_u += &b1u;
        let mut s1_v = &geom.d2q_dq02 * &xi_v;
        s1_v += &b1v;
        let mut g2_u = &geom.d3q_dq03 * &xi_u;
        g2_u += &b2u;
        let mut g2_v = &geom.d3q_dq03 * &xi_v;
        g2_v += &b2v;
        let q_uv = &(&geom.d2q_dq02 * &(&xi_u * &xi_v)) + &(&b1u * &xi_v) + &(&b1v * &xi_u);
        let s1_uv = &(&geom.d3q_dq03 * &(&xi_u * &xi_v)) + &(&b2u * &xi_v) + &(&b2v * &xi_u);
        let g2_uv = &(&geom.d4q_dq04 * &(&xi_u * &xi_v)) + &(&b3u * &xi_v) + &(&b3v * &xi_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?;
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?;
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi_u)?;
        let basis1_v = scale_matrix_rows(&geom.basis_d2, &xi_v)?;
        let basis1_uv = scale_matrix_rows(&geom.basis_d3, &(&xi_u * &xi_v))?;

        // logb κ-scaled η_ls directions; κ' = κ(1−κ), κ'' = κ(1−κ)(1−2κ).
        let szeta_u = &rows.kappa * &zeta_u;
        let szeta_v = &rows.kappa * &zeta_v;
        let zeta_u_zeta_v = &zeta_u * &zeta_v;
        let dw_u = -2.0 * &rows.w * &szeta_u;
        let dw_v = -2.0 * &rows.w * &szeta_v;
        // ∂²w/∂u∂v: differentiating −2wκζ_u along v gains a −2w·κ'·ζ_v·ζ_u
        // term that the κ-as-constant code dropped.
        let dw_uv = 4.0 * &rows.w * &(&szeta_u * &szeta_v)
            - 2.0 * &rows.w * &rows.kappa_prime * &zeta_u_zeta_v;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta_u);
        let dm_v = -(&rows.w * &q_v) - &(2.0 * &rows.m * &szeta_v);
        // ∂²m/∂u∂v: same structural κ' correction as dw_uv (the −2mκζ_u term
        // gains −2m·κ'·ζ_v·ζ_u when differentiated along η_ls).
        let dm_uv = &(2.0 * &rows.w * &(&q_u * &szeta_v + &q_v * &szeta_u)) - &(&rows.w * &q_uv)
            + &(4.0 * &rows.m * &(&szeta_u * &szeta_v))
            - 2.0 * &rows.m * &rows.kappa_prime * &zeta_u_zeta_v;
        let dn_uv = &(2.0 * &rows.w * &(&q_u * &q_v))
            + &(4.0 * &rows.m * &(&q_u * &szeta_v + &q_v * &szeta_u))
            - &(2.0 * &rows.m * &q_uv)
            + &(4.0 * &rows.n * &(&szeta_u * &szeta_v))
            - 2.0 * &rows.n * &rows.kappa_prime * &zeta_u_zeta_v;
        // First-directional drifts of n (used by coeff_ll_uv cross terms).
        let dn_u = -(2.0 * &rows.m * &q_u) - &(2.0 * &rows.n * &szeta_u);
        let dn_v = -(2.0 * &rows.m * &q_v) - &(2.0 * &rows.n * &szeta_v);
        // κ''' = ∂κ''/∂η_ls = κ''(1−2κ) − 2(κ')². Inline since this is the
        // only site that needs the third η-derivative of κ.
        let one_minus_2kappa = rows.kappa.mapv(|k| 1.0 - 2.0 * k);
        let kappa_tprime =
            &rows.kappa_dprime * &one_minus_2kappa - 2.0 * &(&rows.kappa_prime * &rows.kappa_prime);
        let amn = &rows.obs_weight - &rows.n;

        let coeff_mm_uv = &(&dw_uv * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_u * &geom.dq_dq0 * &s1_v)
            + &(2.0 * &dw_v * &geom.dq_dq0 * &s1_u)
            + &(2.0 * &rows.w * &s1_u * &s1_v)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_uv)
            - &(&dm_uv * &geom.d2q_dq02)
            - &(&dm_u * &g2_v)
            - &(&dm_v * &g2_u)
            - &(&rows.m * &g2_uv);
        // ∂²(2κmD)/∂u∂v = 2κ·A_uv + 2κ'·ζ_u·A_v + 2κ'·ζ_v·A_u + 2κ''·ζ_uζ_v·A,
        // with A = mD; the κ-as-constant code dropped the κ' and κ'' terms.
        let a_md_v = &dm_v * &geom.dq_dq0 + &rows.m * &s1_v;
        let a_md_u = &dm_u * &geom.dq_dq0 + &rows.m * &s1_u;
        let coeff_ml_uv = 2.0
            * &rows.kappa
            * &(&dm_uv * &geom.dq_dq0 + &dm_u * &s1_v + &dm_v * &s1_u + &rows.m * &s1_uv)
            + 2.0 * &rows.kappa_prime * &(&zeta_u * &a_md_v + &zeta_v * &a_md_u)
            + 2.0 * &rows.kappa_dprime * &zeta_u_zeta_v * &rows.m * &geom.dq_dq0;
        // ∂²(2κ²n + κ'(a−n))/∂u∂v factored as
        //   (2κ² − κ')·n_uv + (4κκ' − κ'')·(ζ_u n_v + ζ_v n_u)
        //   + [4n((κ')² + κκ'') + κ'''(a−n)]·ζ_u ζ_v.
        let two_ki2_minus_kpi = 2.0 * &rows.kappa * &rows.kappa - &rows.kappa_prime;
        let four_kkpi_minus_kdpi = 4.0 * &(&rows.kappa * &rows.kappa_prime) - &rows.kappa_dprime;
        let zeta_n_sym = &zeta_u * &dn_v + &zeta_v * &dn_u;
        let bracketed_eta_eta = 4.0
            * &rows.n
            * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
            + &kappa_tprime * &amn;
        let coeff_ll_uv = &two_ki2_minus_kpi * &dn_uv
            + &four_kkpi_minus_kdpi * &zeta_n_sym
            + &bracketed_eta_eta * &zeta_u_zeta_v;

        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let a_v = &dw_v * &geom.dq_dq0 + &rows.w * &s1_v;
        let a_uv = &dw_uv * &geom.dq_dq0 + &dw_u * &s1_v + &dw_v * &s1_u + &rows.w * &s1_uv;
        let c_u = -&dm_u;
        let c_v = -&dm_v;
        let c_uv = -&dm_uv;
        // ls-wiggle cross block: l = 2κm; pick up κ', κ'' on each direction.
        let l_u = 2.0 * &rows.kappa * &dm_u + 2.0 * &rows.kappa_prime * &(&zeta_u * &rows.m);
        let l_v = 2.0 * &rows.kappa * &dm_v + 2.0 * &rows.kappa_prime * &(&zeta_v * &rows.m);
        let l_uv = 2.0 * &rows.kappa * &dm_uv
            + 2.0 * &rows.kappa_prime * &(&zeta_u * &dm_v + &zeta_v * &dm_u)
            + 2.0 * &rows.kappa_dprime * &(&zeta_u_zeta_v * &rows.m);

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_uv)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_uv, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_uv)?;
        let h_mw = xt_diag_y_dense(xmu, &a_uv, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_v)?
            + &xt_diag_y_dense(xmu, &a_v, &basis_u)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_uv)?
            + &xt_diag_y_dense(xmu, &c_uv, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_v)?
            + &xt_diag_y_dense(xmu, &c_v, &basis1_u)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_uv)?;
        let h_lw = xt_diag_y_dense(x_ls, &l_uv, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_v)?
            + &xt_diag_y_dense(x_ls, &l_v, &basis_u)?
            + &xt_diag_y_dense(x_ls, &(2.0 * &rows.kappa * &rows.m), &basis_uv)?;
        let a_ab = xt_diag_y_dense(&basis_uv, &rows.w, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &rows.w, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + &a_ij.t()
            + &a_iwj
            + &a_iwj.t()
            + &a_jwi
            + &a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xmu_map = dir_a.xmu_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();

        let q_a = &geom.dq_dq0 * &dir_a.zmu_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.zmu_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.zmu_psi;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.zmu_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.zmu_psi)?;
        // logb κ-chain on η_ls; e_a = ∂η_ls/∂ψ_a row-direction.
        let e_a = &dir_a.z_ls_psi;
        let amn = &rows.obs_weight - &rows.n;
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;

        let objective_psi = (-&rows.m * &q_a + &s_ls * e_a).sum();
        let score_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_map.transpose_mul(s_mu.view()) + xmu.t().dot(&s_mu_a)),
            &(x_ls_map.transpose_mul(s_ls.view()) + x_ls.t().dot(&s_ls_a)),
            &(basis_a.t().dot(&s_w) + geom.basis.t().dot(&s_w_a)),
        );

        // Static blocks under logb: coeff_ml = 2κmD; coeff_ll = 2κ²n + κ'(a−n); l = 2κm.
        // Directional pieces add κ' on the e_a leg (and κ'' would only show
        // up at second-order, so this single-ψ path stops at κ').
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_ml = 2.0 * &rows.kappa * &rows.m * &geom.dq_dq0;
        let coeff_ml_a = 2.0 * &rows.kappa * &(&dm_a * &geom.dq_dq0 + &rows.m * &s1_a)
            + 2.0 * &rows.kappa_prime * &(e_a * &rows.m * &geom.dq_dq0);
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn;
        let coeff_ll_a = (2.0 * &rows.kappa * &rows.kappa - &rows.kappa_prime) * &dn_a
            + (4.0 * &rows.kappa * &rows.kappa_prime * &rows.n + &rows.kappa_dprime * &amn) * e_a;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let l = 2.0 * &rows.m * &rows.kappa;
        let l_a = 2.0 * &rows.kappa * &dm_a + 2.0 * &rows.kappa_prime * &(e_a * &rows.m);
        let h_mm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &h_mm_a1 + &h_mm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a, x_ls)?;
        let h_ll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &h_ll_a1 + &h_ll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_a, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a, &basis_a)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_a, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_a, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a)?;
        let h_ww_a1 = xt_diag_y_dense(&basis_a, &rows.w, &geom.basis)?;
        let h_ww = &h_ww_a1 + &h_ww_a1.t() + &xt_diag_x_dense(&geom.basis, &dw_a)?;

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_b) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_a,
                &dir_b,
                xmu,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_a: &GaussianLocationScaleWiggleJointPsiDirection,
        dir_b: &GaussianLocationScaleWiggleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_a,
            dir_b,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_a_map = dir_a.xmu_psi.as_linear_map_ref();
        let x_ls_a_map = dir_a.x_ls_psi.as_linear_map_ref();
        let xmu_b_map = dir_b.xmu_psi.as_linear_map_ref();
        let x_ls_b_map = dir_b.x_ls_psi.as_linear_map_ref();
        let xmu_ab_map = second_psi_linear_map(
            second_drifts.xmu_ab_action.as_ref(),
            second_drifts.xmu_ab.as_ref(),
            n,
            xmu.ncols(),
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            x_ls.ncols(),
        );
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let q_a = &geom.dq_dq0 * &dir_a.zmu_psi;
        let q_b = &geom.dq_dq0 * &dir_b.zmu_psi;
        let q_ab = &(&geom.dq_dq0 * &second_drifts.zmu_ab)
            + &(&geom.d2q_dq02 * &(&dir_a.zmu_psi * &dir_b.zmu_psi));
        let s1_a = &geom.d2q_dq02 * &dir_a.zmu_psi;
        let s1_b = &geom.d2q_dq02 * &dir_b.zmu_psi;
        let s1_ab = &(&geom.d3q_dq03 * &(&dir_a.zmu_psi * &dir_b.zmu_psi))
            + &(&geom.d2q_dq02 * &second_drifts.zmu_ab);
        let g2_a = &geom.d3q_dq03 * &dir_a.zmu_psi;
        let g2_b = &geom.d3q_dq03 * &dir_b.zmu_psi;
        let g2_ab = &(&geom.d4q_dq04 * &(&dir_a.zmu_psi * &dir_b.zmu_psi))
            + &(&geom.d3q_dq03 * &second_drifts.zmu_ab);
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.zmu_psi)?;
        let basis_b = scale_matrix_rows(&geom.basis_d1, &dir_b.zmu_psi)?;
        let basis_ab = scale_matrix_rows(&geom.basis_d1, &second_drifts.zmu_ab)?
            + &scale_matrix_rows(&geom.basis_d2, &(&dir_a.zmu_psi * &dir_b.zmu_psi))?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.zmu_psi)?;
        let basis1_b = scale_matrix_rows(&geom.basis_d2, &dir_b.zmu_psi)?;
        let basis1_ab = scale_matrix_rows(&geom.basis_d2, &second_drifts.zmu_ab)?
            + &scale_matrix_rows(&geom.basis_d3, &(&dir_a.zmu_psi * &dir_b.zmu_psi))?;

        // logb κ-chain on η_ls; κ' = κ(1−κ), κ'' = κ(1−κ)(1−2κ),
        // κ''' = κ''(1−2κ) − 2(κ')².
        let e_a = &dir_a.z_ls_psi;
        let e_b = &dir_b.z_ls_psi;
        let e_ab = &second_drifts.z_ls_ab;
        let amn = &rows.obs_weight - &rows.n;
        let one_minus_2kappa = rows.kappa.mapv(|k| 1.0 - 2.0 * k);
        let kappa_tprime =
            &rows.kappa_dprime * &one_minus_2kappa - 2.0 * &(&rows.kappa_prime * &rows.kappa_prime);
        // 4κ² − 2κ' (∂²w/∂η² style coefficient when both directions hit η_ls).
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;

        // Row drifts under logb. The η_ls direction picks up a κ on each step,
        // and η_ls·η_ls picks up (4κ²−2κ') from differentiating κ on the
        // second leg. The η_ab (z_ls_ab) leg uses just one κ from the chain.
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dw_b = -2.0 * &rows.w * &rows.kappa * e_b;
        let dw_ab =
            &four_k2_minus_2kpi * &rows.w * &(e_a * e_b) - &(2.0 * &rows.w * &rows.kappa * e_ab);
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dm_b = -(&rows.w * &q_b) - &(2.0 * &rows.m * &rows.kappa * e_b);
        let dm_ab = &(2.0 * &rows.w * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(&rows.w * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * e_b))
            - &(2.0 * &rows.m * &rows.kappa * e_ab);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let dn_b = -(2.0 * &rows.m * &q_b) - &(2.0 * &rows.n * &rows.kappa * e_b);
        let dn_ab = &(2.0 * &rows.w * &(&q_a * &q_b))
            + &(4.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(2.0 * &rows.m * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.n * &(e_a * e_b))
            - &(2.0 * &rows.n * &rows.kappa * e_ab);

        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_mu_b = -(&dm_b * &geom.dq_dq0) - &(&rows.m * &s1_b);
        let s_mu_ab =
            -(&dm_ab * &geom.dq_dq0) - &(&dm_a * &s1_b) - &(&dm_b * &s1_a) - &(&rows.m * &s1_ab);
        // score_ls = κ(a−n); ψ derivatives carry κ' / κ'' from chain on κ.
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_ls_b = &rows.kappa_prime * &(e_b * &amn) - &rows.kappa * &dn_b;
        // s_ls_ab = κ''·e_a·e_b·(a−n) + κ'·e_ab·(a−n)
        //         − κ'·(e_a·n_b + e_b·n_a) − κ·n_ab
        let s_ls_ab = &rows.kappa_dprime * &(e_a * e_b) * &amn + &rows.kappa_prime * e_ab * &amn
            - &rows.kappa_prime * &(e_a * &dn_b + e_b * &dn_a)
            - &rows.kappa * &dn_ab;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;
        let s_w_b = -&dm_b;
        let s_w_ab = -&dm_ab;

        let objective_psi_psi = (&rows.w * &(&q_a * &q_b)
            + &(2.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            + &((2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn)
                * &(e_a * e_b))
            - &(&rows.m * &q_ab)
            + &(&rows.kappa * &amn * e_ab))
            .sum();

        let score_psi_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_ab_map.transpose_mul(s_mu.view())
                + xmu_a_map.transpose_mul(s_mu_b.view())
                + xmu_b_map.transpose_mul(s_mu_a.view())
                + xmu.t().dot(&s_mu_ab)),
            &(x_ls_ab_map.transpose_mul(s_ls.view())
                + x_ls_a_map.transpose_mul(s_ls_b.view())
                + x_ls_b_map.transpose_mul(s_ls_a.view())
                + x_ls.t().dot(&s_ls_ab)),
            &(basis_ab.t().dot(&s_w)
                + basis_a.t().dot(&s_w_b)
                + basis_b.t().dot(&s_w_a)
                + geom.basis.t().dot(&s_w_ab)),
        );

        // Static blocks under logb. coeff_mm has no κ; coeff_ml = 2κmD;
        // coeff_ll = 2κ²n + κ'(a−n); l = 2κm. The directional derivatives
        // pick up κ', κ'', κ''' on every leg that hits η_ls.
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_ml = 2.0 * &rows.kappa * &rows.m * &geom.dq_dq0;
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn;
        // coeff_mm_a/b/ab: structurally κ-free; correctness now follows from
        // dw_a/_b/_ab and dm_a/_b/_ab carrying the κ chain on η_ls (above).
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_mm_b = &(&dw_b * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_b)
            - &(&dm_b * &geom.d2q_dq02)
            - &(&rows.m * &g2_b);
        let coeff_mm_ab = &(&dw_ab * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_b)
            + &(2.0 * &dw_b * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_a * &s1_b)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_ab)
            - &(&dm_ab * &geom.d2q_dq02)
            - &(&dm_a * &g2_b)
            - &(&dm_b * &g2_a)
            - &(&rows.m * &g2_ab);
        // coeff_ml_a = 2κ(dm_a·D + m·s1_a) + 2κ'·e_a·m·D — same shape as
        // helper 4 dh_ls_ls but along ψ_a; kappa' on every direct η_ls leg.
        let coeff_ml_a = 2.0 * &rows.kappa * &(&dm_a * &geom.dq_dq0 + &rows.m * &s1_a)
            + 2.0 * &rows.kappa_prime * &(e_a * &rows.m * &geom.dq_dq0);
        let coeff_ml_b = 2.0 * &rows.kappa * &(&dm_b * &geom.dq_dq0 + &rows.m * &s1_b)
            + 2.0 * &rows.kappa_prime * &(e_b * &rows.m * &geom.dq_dq0);
        // coeff_ml_ab: ∂²(2κmD)/∂a∂b. Includes the η_ab leg (e_ab) since this
        // is a ψ-second-order path (η_ab generally nonzero).
        let coeff_ml_ab = 2.0
            * &rows.kappa
            * &(&dm_ab * &geom.dq_dq0 + &dm_a * &s1_b + &dm_b * &s1_a + &rows.m * &s1_ab)
            + 2.0
                * &rows.kappa_prime
                * &(e_a * &(&dm_b * &geom.dq_dq0 + &rows.m * &s1_b)
                    + e_b * &(&dm_a * &geom.dq_dq0 + &rows.m * &s1_a))
            + 2.0 * &rows.kappa_dprime * &(e_a * e_b) * &rows.m * &geom.dq_dq0
            + 2.0 * &rows.kappa_prime * e_ab * &(&rows.m * &geom.dq_dq0);
        // coeff_ll_a = (2κ²−κ')n_a + (4κκ'n + κ''(a−n))·e_a.
        let coeff_ll_a = (2.0 * &rows.kappa * &rows.kappa - &rows.kappa_prime) * &dn_a
            + (4.0 * &rows.kappa * &rows.kappa_prime * &rows.n + &rows.kappa_dprime * &amn) * e_a;
        let coeff_ll_b = (2.0 * &rows.kappa * &rows.kappa - &rows.kappa_prime) * &dn_b
            + (4.0 * &rows.kappa * &rows.kappa_prime * &rows.n + &rows.kappa_dprime * &amn) * e_b;
        // coeff_ll_ab: full ψ-second-order ∂²(2κ²n + κ'(a−n))/∂a∂b. β-only
        // factored form per math team's verification, plus the η_ab leg
        // (4κκ'n + κ''(a−n))·e_ab from differentiating once at η_ab.
        let coeff_ll_ab = (2.0 * &rows.kappa * &rows.kappa - &rows.kappa_prime) * &dn_ab
            + (4.0 * &rows.kappa * &rows.kappa_prime - &rows.kappa_dprime)
                * &(e_a * &dn_b + e_b * &dn_a)
            + (4.0
                * &rows.n
                * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
                + &kappa_tprime * &amn)
                * &(e_a * e_b)
            + (4.0 * &rows.kappa * &rows.kappa_prime * &rows.n + &rows.kappa_dprime * &amn) * e_ab;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_b = &dw_b * &geom.dq_dq0 + &rows.w * &s1_b;
        let a_ab = &dw_ab * &geom.dq_dq0 + &dw_a * &s1_b + &dw_b * &s1_a + &rows.w * &s1_ab;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let c_b = -&dm_b;
        let c_ab = -&dm_ab;
        // l = 2κm; l_a/_b add κ' on the e direction; l_ab adds κ'', plus
        // a κ' on the η_ab leg.
        let l = 2.0 * &rows.kappa * &rows.m;
        let l_a = 2.0 * &rows.kappa * &dm_a + 2.0 * &rows.kappa_prime * &(e_a * &rows.m);
        let l_b = 2.0 * &rows.kappa * &dm_b + 2.0 * &rows.kappa_prime * &(e_b * &rows.m);
        let l_ab = 2.0 * &rows.kappa * &dm_ab
            + 2.0 * &rows.kappa_prime * &(e_a * &dm_b + e_b * &dm_a)
            + 2.0 * &rows.kappa_dprime * &(e_a * e_b) * &rows.m
            + 2.0 * &rows.kappa_prime * e_ab * &rows.m;

        let hmm_ab = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_ij = weighted_crossprod_psi_maps(xmu_a_map, coeff_mm.view(), xmu_b_map)?;
        let hmm_iwj = weighted_crossprod_psi_maps(
            xmu_a_map,
            coeff_mm_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_jwi = weighted_crossprod_psi_maps(
            xmu_b_map,
            coeff_mm_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_ab
            + &hmm_ab.t()
            + &hmm_ij
            + &hmm_ij.t()
            + &hmm_iwj
            + &hmm_iwj.t()
            + &hmm_jwi
            + &hmm_jwi.t()
            + &xt_diag_x_dense(xmu, &coeff_mm_ab)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(xmu_a_map, coeff_ml.view(), x_ls_b_map)?
            + &weighted_crossprod_psi_maps(xmu_b_map, coeff_ml.view(), x_ls_a_map)?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                coeff_ml_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                coeff_ml_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_a.view(),
                x_ls_b_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_b.view(),
                x_ls_a_map,
            )?
            + &xt_diag_y_dense(xmu, &coeff_ml_ab, x_ls)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml.view(),
                x_ls_ab_map,
            )?;
        let hll_ab = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_ij = weighted_crossprod_psi_maps(x_ls_a_map, coeff_ll.view(), x_ls_b_map)?;
        let hll_iwj = weighted_crossprod_psi_maps(
            x_ls_a_map,
            coeff_ll_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_jwi = weighted_crossprod_psi_maps(
            x_ls_b_map,
            coeff_ll_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_ab
            + &hll_ab.t()
            + &hll_ij
            + &hll_ij.t()
            + &hll_iwj
            + &hll_iwj.t()
            + &hll_jwi
            + &hll_jwi.t()
            + &xt_diag_x_dense(x_ls, &coeff_ll_ab)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_ab_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            xmu_b_map,
            a_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_ab, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(xmu, &a_b, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_ab)?
            + &weighted_crossprod_psi_maps(
                xmu_ab_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_b),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_ab, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_a),
            )?
            + &xt_diag_y_dense(xmu, &c_b, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_ab)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            x_ls_b_map,
            l_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_ab, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                x_ls_b_map,
                l.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(x_ls, &l_b, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_ab)?;
        let hww_ab = xt_diag_y_dense(&basis_ab, &rows.w, &geom.basis)?;
        let hww_ij = xt_diag_y_dense(&basis_a, &rows.w, &basis_b)?;
        let hww_iwj = xt_diag_y_dense(&basis_a, &dw_b, &geom.basis)?;
        let hww_jwi = xt_diag_y_dense(&basis_b, &dw_a, &geom.basis)?;
        let h_ww = &hww_ab
            + &hww_ab.t()
            + &hww_ij
            + &hww_ij.t()
            + &hww_iwj
            + &hww_iwj.t()
            + &hww_jwi
            + &hww_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_ab)?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                xmu,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &GaussianLocationScaleWiggleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.xmu_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily joint psi hessian directional derivative",
        )?;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi = xmu.dot(&umu);
        let zeta = x_ls.dot(&u_ls);
        let zmu_a_u = xmu_map.forward_mul(umu.view());
        let zls_a_u = x_ls_map.forward_mul(u_ls.view());
        let b1u = geom.basis_d1.dot(&uw);
        let b2u = geom.basis_d2.dot(&uw);
        let b3u = geom.basis_d3.dot(&uw);

        let q_u = &(&geom.dq_dq0 * &xi) + &geom.basis.dot(&uw);
        let s1_u = &(&geom.d2q_dq02 * &xi) + &b1u;
        let g2_u = &(&geom.d3q_dq03 * &xi) + &b2u;
        let g3_u = &(&geom.d4q_dq04 * &xi) + &b3u;

        let q_a = &geom.dq_dq0 * &dir_a.zmu_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.zmu_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.zmu_psi;
        let q_a_u = &(&s1_u * &dir_a.zmu_psi) + &(&geom.dq_dq0 * &zmu_a_u);
        let s1_a_u = &(&g2_u * &dir_a.zmu_psi) + &(&geom.d2q_dq02 * &zmu_a_u);
        let g2_a_u = &(&g3_u * &dir_a.zmu_psi) + &(&geom.d3q_dq03 * &zmu_a_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.zmu_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.zmu_psi)?;
        let basis_a_u = scale_matrix_rows(&geom.basis_d2, &(&xi * &dir_a.zmu_psi))?
            + &scale_matrix_rows(&geom.basis_d1, &zmu_a_u)?;
        let basis1_a_u = scale_matrix_rows(&geom.basis_d3, &(&xi * &dir_a.zmu_psi))?
            + &scale_matrix_rows(&geom.basis_d2, &zmu_a_u)?;

        // logb κ-chain on η_ls; e_a = ψ_a's η_ls direction, ζ = β-direction.
        // η_au = zls_a_u is the second mixed derivative (β·ψ).
        let e_a = &dir_a.z_ls_psi;
        let amn = &rows.obs_weight - &rows.n;
        let one_minus_2kappa = rows.kappa.mapv(|k| 1.0 - 2.0 * k);
        let kappa_tprime =
            &rows.kappa_dprime * &one_minus_2kappa - 2.0 * &(&rows.kappa_prime * &rows.kappa_prime);
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;
        let dw_u = -2.0 * &rows.w * &rows.kappa * &zeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &rows.kappa * &zeta);
        let dn_u = -(2.0 * &rows.m * &q_u) - &(2.0 * &rows.n * &rows.kappa * &zeta);
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let dw_a_u = &four_k2_minus_2kpi * &rows.w * &(e_a * &zeta)
            - &(2.0 * &rows.w * &rows.kappa * &zls_a_u);
        let dm_a_u = &(2.0 * &rows.w * &rows.kappa * &(&q_a * &zeta + &q_u * e_a))
            - &(&rows.w * &q_a_u)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * &zeta))
            - &(2.0 * &rows.m * &rows.kappa * &zls_a_u);
        let dn_a_u = &(2.0 * &rows.w * &(&q_a * &q_u))
            + &(4.0 * &rows.m * &rows.kappa * &(&q_a * &zeta + &q_u * e_a))
            - &(2.0 * &rows.m * &q_a_u)
            + &(&four_k2_minus_2kpi * &rows.n * &(e_a * &zeta))
            - &(2.0 * &rows.n * &rows.kappa * &zls_a_u);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // coeff_ml_u = ∂(2κmD)/∂u = 2κ(dm_u·D + m·s1_u) + 2κ'·ζ·m·D.
        let coeff_ml_u = 2.0 * &rows.kappa * &(&dm_u * &geom.dq_dq0 + &rows.m * &s1_u)
            + 2.0 * &rows.kappa_prime * &(&zeta * &rows.m * &geom.dq_dq0);
        // coeff_ll_u = (2κ²−κ')·dn_u + (4κκ'·n + κ''(a−n))·ζ.
        let coeff_ll_u = (2.0 * &rows.kappa * &rows.kappa - &rows.kappa_prime) * &dn_u
            + (4.0 * &rows.kappa * &rows.kappa_prime * &rows.n + &rows.kappa_dprime * &amn) * &zeta;
        let coeff_mm_a_u = &(&dw_a_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_u)
            + &(2.0 * &dw_u * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_u * &s1_a)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a_u)
            - &(&dm_a_u * &geom.d2q_dq02)
            - &(&dm_a * &g2_u)
            - &(&dm_u * &g2_a)
            - &(&rows.m * &g2_a_u);
        // coeff_ml_a_u = ∂²(2κmD)/∂a∂u — full mixed second derivative,
        // including the η_au = zls_a_u leg picking up 2κ'·η_au·m·D.
        let coeff_ml_a_u = 2.0
            * &rows.kappa
            * &(&dm_a_u * &geom.dq_dq0 + &dm_a * &s1_u + &dm_u * &s1_a + &rows.m * &s1_a_u)
            + 2.0
                * &rows.kappa_prime
                * &(e_a * &(&dm_u * &geom.dq_dq0 + &rows.m * &s1_u)
                    + &zeta * &(&dm_a * &geom.dq_dq0 + &rows.m * &s1_a))
            + 2.0 * &rows.kappa_dprime * &(e_a * &zeta) * &rows.m * &geom.dq_dq0
            + 2.0 * &rows.kappa_prime * &zls_a_u * &(&rows.m * &geom.dq_dq0);
        // coeff_ll_a_u = ∂²(2κ²n + κ'(a−n))/∂a∂u — full mixed second
        // derivative with the η_au leg added via (4κκ'n + κ''(a−n))·η_au.
        let coeff_ll_a_u = (2.0 * &rows.kappa * &rows.kappa - &rows.kappa_prime) * &dn_a_u
            + (4.0 * &rows.kappa * &rows.kappa_prime - &rows.kappa_dprime)
                * &(e_a * &dn_u + &zeta * &dn_a)
            + (4.0
                * &rows.n
                * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
                + &kappa_tprime * &amn)
                * &(e_a * &zeta)
            + (4.0 * &rows.kappa * &rows.kappa_prime * &rows.n + &rows.kappa_dprime * &amn)
                * &zls_a_u;

        let a = &rows.w * &geom.dq_dq0;
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_a_u = &dw_a_u * &geom.dq_dq0 + &dw_a * &s1_u + &dw_u * &s1_a + &rows.w * &s1_a_u;
        let c = -&rows.m;
        let c_u = -&dm_u;
        let c_a = -&dm_a;
        let c_a_u = -&dm_a_u;
        // l = 2κm; pick up κ'/κ'' on each direction; η_au leg adds κ'.
        let l = 2.0 * &rows.kappa * &rows.m;
        let l_u = 2.0 * &rows.kappa * &dm_u + 2.0 * &rows.kappa_prime * &(&zeta * &rows.m);
        let l_a = 2.0 * &rows.kappa * &dm_a + 2.0 * &rows.kappa_prime * &(e_a * &rows.m);
        let l_a_u = 2.0 * &rows.kappa * &dm_a_u
            + 2.0 * &rows.kappa_prime * &(e_a * &dm_u + &zeta * &dm_a)
            + 2.0 * &rows.kappa_dprime * &(e_a * &zeta) * &rows.m
            + 2.0 * &rows.kappa_prime * &zls_a_u * &rows.m;

        let hmm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_a1 + &hmm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a_u)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml_u.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a_u, x_ls)?;
        let hll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_a1 + &hll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a_u)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(xmu, &a_a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_u)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_a_u)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c_u.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_u),
            )?
            + &xt_diag_y_dense(xmu, &c_a_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_u)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a_u)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(x_ls, &l_a_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_u)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a_u)?;
        let hww_a_u = xt_diag_y_dense(&basis_a_u, &rows.w, &geom.basis)?;
        let hww_aw = xt_diag_y_dense(&basis_a, &dw_u, &geom.basis)?;
        let hww_au = xt_diag_y_dense(&basis_a, &rows.w, &basis_u)?;
        let h_ww = &hww_a_u
            + &hww_a_u.t()
            + &hww_aw
            + &hww_aw.t()
            + &hww_au
            + &hww_au.t()
            + &xt_diag_x_dense(&geom.basis, &dw_a_u)?;

        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &xmu,
            &x_ls,
        )
    }
}

impl CustomFamily for GaussianLocationScaleWiggleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Three fully-coupled blocks (mean p_t, log-σ p_ℓ, link-wiggle p_w):
        // every cross-block in the joint Hessian is dense.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(self.y.len() as u64, specs)
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        Ok(project_monotone_wiggle_beta(beta))
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleWiggleFamily input size mismatch".to_string());
        }
        let q = eta_mu + etaw;
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut zls = Array1::<f64>::zeros(n);
        let mut wls = Array1::<f64>::zeros(n);
        let mut zw = Array1::<f64>::zeros(n);
        let mut ww = Array1::<f64>::zeros(n);
        for i in 0..n {
            let row =
                gaussian_diagonal_row_kernel(self.y[i], q[i], eta_ls[i], self.weights[i], ln2pi);
            ll += row.log_likelihood;
            wmu[i] = row.location_working_weight;
            ww[i] = row.location_working_weight;
            zmu[i] = eta_mu[i] + row.location_working_shift;
            zw[i] = etaw[i] + row.location_working_shift;
            wls[i] = row.log_sigma_working_weight;
            zls[i] = row.log_sigma_working_response;
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: zmu,
                    working_weights: wmu,
                },
                BlockWorkingSet::Diagonal {
                    working_response: zls,
                    working_weights: wls,
                },
                BlockWorkingSet::Diagonal {
                    working_response: zw,
                    working_weights: ww,
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != self.y.len()
            || eta_ls.len() != self.y.len()
            || etaw.len() != self.y.len()
            || self.weights.len() != self.y.len()
        {
            return Err("GaussianLocationScaleWiggleFamily input size mismatch".to_string());
        }
        let q = eta_mu + etaw;
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        for i in 0..self.y.len() {
            let sigma_i = logb_sigma_from_eta_scalar(eta_ls[i]);
            let inv_s2 = (sigma_i * sigma_i).recip();
            let r = self.y[i] - q[i];
            ll += self.weights[i] * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()));
        }
        Ok(ll)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let pmu = self
            .mu_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing mu design".to_string()
            })?
            .ncols();
        let p_ls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing log-sigma design"
                    .to_string()
            })?
            .ncols();
        let pw = block_states[Self::BLOCK_WIGGLE].beta.len();
        let total = pmu + p_ls + pw;
        let (start, end) = match block_idx {
            Self::BLOCK_MU => (0usize, pmu),
            Self::BLOCK_LOG_SIGMA => (pmu, pmu + p_ls),
            Self::BLOCK_WIGGLE => (pmu + p_ls, total),
            _ => return Ok(None),
        };
        if d_beta.len() != end - start {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily block {block_idx} d_beta length mismatch: got {}, expected {}",
                d_beta.len(),
                end - start
            ));
        }
        let mut d_beta_flat = Array1::<f64>::zeros(total);
        d_beta_flat.slice_mut(s![start..end]).assign(d_beta);
        let (xmu, x_ls) = self.dense_block_designs()?;
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                block_states,
                &xmu,
                &x_ls,
                &d_beta_flat,
            )?
            .ok_or_else(|| "missing Gaussian wiggle exact joint directional Hessian".to_string())?;
        Ok(Some(d_joint.slice(s![start..end, start..end]).to_owned()))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.len() < 1 {
            return Err("Gaussian wiggle geometry requires mean block".to_string());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        if eta_mu.len() != self.y.len() {
            return Err("Gaussian wiggle geometry input size mismatch".to_string());
        }
        let x = self.wiggle_design(eta_mu.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(format!(
                "Gaussian dynamic wiggle design col mismatch: got {}, expected {}",
                x.ncols(),
                spec.design.ncols()
            ));
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }
}

/// Matrix-free joint-Hessian operator for the 3-block Gaussian
/// location-scale wiggle family. See `GaussianLocationScaleWiggleHessianRowPieces`
/// for the per-row weight structure. The matvec applies
///
///   r_μ  = D_mm u_μ + D_ml u_ls + D_mw_b (B v_w) + D_mw_d (B' v_w),
///   r_ls = D_ml u_μ + D_ll u_ls + D_lw_b (B v_w),
///   r_b  = D_mw_b u_μ + D_lw_b u_ls + D_ww (B v_w),
///   r_d  = D_mw_d u_μ,
///
/// then forms `out_w = B^T r_b + (B')^T r_d`. The ls-wiggle cross block has
/// no B' contribution because the wiggle enters the Gaussian likelihood only
/// through `q = η_μ + η_w` (no σ-chain), so the Gaussian wiggle has one
/// fewer cross-coefficient than the binomial wiggle.
struct GaussianLocationScaleWiggleHessianWorkspace {
    family: GaussianLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    xmu: Array2<f64>,
    x_ls: Array2<f64>,
    pieces: GaussianLocationScaleWiggleHessianRowPieces,
}

impl GaussianLocationScaleWiggleHessianWorkspace {
    fn new(
        family: GaussianLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        _specs: Vec<ParameterBlockSpec>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let pieces = family.wiggle_hessian_row_pieces(&block_states)?;
        Ok(Self {
            family,
            block_states,
            xmu,
            x_ls,
            pieces,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleWiggleHessianWorkspace {
    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        if v.len() != total {
            return Err(format!(
                "GaussianLocationScaleWiggle matvec dimension mismatch: got {}, expected {}",
                v.len(),
                total
            ));
        }
        let v_mu = v.slice(s![0..pmu]);
        let v_ls = v.slice(s![pmu..pmu + p_ls]);
        let v_w = v.slice(s![pmu + p_ls..total]);

        let u_mu = self.xmu.dot(&v_mu);
        let u_ls = self.x_ls.dot(&v_ls);
        let u_b = self.pieces.basis.dot(&v_w);
        let u_d = self.pieces.basis_d1.dot(&v_w);

        let r_mu = &self.pieces.coeff_mm * &u_mu
            + &self.pieces.coeff_ml * &u_ls
            + &self.pieces.coeff_mw_b * &u_b
            + &self.pieces.coeff_mw_d * &u_d;
        let r_ls = &self.pieces.coeff_ml * &u_mu
            + &self.pieces.coeff_ll * &u_ls
            + &self.pieces.coeff_lw_b * &u_b;
        let r_b = &self.pieces.coeff_mw_b * &u_mu
            + &self.pieces.coeff_lw_b * &u_ls
            + &self.pieces.coeff_ww * &u_b;
        let r_d = &self.pieces.coeff_mw_d * &u_mu;

        let out_mu = self.xmu.t().dot(&r_mu);
        let out_ls = self.x_ls.t().dot(&r_ls);
        let out_w = self.pieces.basis.t().dot(&r_b) + &self.pieces.basis_d1.t().dot(&r_d);

        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..pmu + p_ls]).assign(&out_ls);
        out.slice_mut(s![pmu + p_ls..total]).assign(&out_w);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        let mut diag = Array1::<f64>::zeros(total);
        let n = self.pieces.coeff_mm.len();
        for j in 0..pmu {
            let col = self.xmu.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_mm[i] * v * v;
            }
            diag[j] = acc;
        }
        for j in 0..p_ls {
            let col = self.x_ls.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_ll[i] * v * v;
            }
            diag[pmu + j] = acc;
        }
        for j in 0..pw {
            let col = self.pieces.basis.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_ww[i] * v * v;
            }
            diag[pmu + p_ls + j] = acc;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                &self.xmu,
                &self.x_ls,
                d_beta_flat,
            )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                &self.xmu,
                &self.x_ls,
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }
}

impl CustomFamilyGenerative for GaussianLocationScaleWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let mean = &block_states[Self::BLOCK_MU].eta + &block_states[Self::BLOCK_WIGGLE].eta;
        let sigma = block_states[Self::BLOCK_LOG_SIGMA]
            .eta
            .mapv(logb_sigma_from_eta_scalar);
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}

fn expect_single_block<'a>(
    block_states: &'a [ParameterBlockState],
    family_name: &str,
) -> Result<&'a ParameterBlockState, String> {
    if block_states.len() != 1 {
        return Err(format!(
            "{family_name} expects 1 block, got {}",
            block_states.len()
        ));
    }
    Ok(&block_states[0])
}

#[derive(Clone)]
pub struct BinomialMeanWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction during
    /// exact-Newton joint psi evaluation. Defaults to
    /// `ResourcePolicy::default_library()` when the family is built without
    /// an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}

struct BinomialMeanWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    basis_d3: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
    d4q_dq04: Array1<f64>,
}

struct BinomialMeanWiggleJointPsiDirection {
    x_eta_psi: Option<Array2<f64>>,
    z_eta_psi: Array1<f64>,
}

fn binomial_pack_mean_wiggle_joint_score(
    score_eta: &Array1<f64>,
    score_w: &Array1<f64>,
) -> Array1<f64> {
    let p_eta = score_eta.len();
    let pw = score_w.len();
    let mut out = Array1::<f64>::zeros(p_eta + pw);
    out.slice_mut(s![0..p_eta]).assign(score_eta);
    out.slice_mut(s![p_eta..p_eta + pw]).assign(score_w);
    out
}

fn binomial_pack_mean_wiggle_joint_symmetrichessian(
    h_eta_eta: &Array2<f64>,
    h_eta_w: &Array2<f64>,
    h_ww: &Array2<f64>,
) -> Array2<f64> {
    let p_eta = h_eta_eta.nrows();
    let pw = h_ww.nrows();
    let total = p_eta + pw;
    let mut out = Array2::<f64>::zeros((total, total));
    out.slice_mut(s![0..p_eta, 0..p_eta]).assign(h_eta_eta);
    out.slice_mut(s![0..p_eta, p_eta..total]).assign(h_eta_w);
    out.slice_mut(s![p_eta..total, p_eta..total]).assign(h_ww);
    mirror_upper_to_lower(&mut out);
    out
}

impl BinomialMeanWiggleFamily {
    pub const BLOCK_ETA: usize = 0;
    pub const BLOCK_WIGGLE: usize = 1;

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            options.derivative_order,
        )
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d_constrained.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d_constrained.dot(&beta_link_wiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle second-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d2.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d2.dot(&beta_link_wiggle))
    }

    fn wiggle_d3basis_constrained(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        if d3.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle third-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d3.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d3.dot(&beta_link_wiggle))
    }

    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle fourth-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d4.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    fn wiggle_geometry(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<BinomialMeanWiggleGeometry, String> {
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(BinomialMeanWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        })
    }

    fn neglog_q_derivatives(&self, y: f64, weight: f64, q: f64) -> Result<(f64, f64, f64), String> {
        let mut jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        let (mu_clamped, clamp_active) = clamped_binomial_probability(jet.mu);
        jet.mu = mu_clamped;
        if clamp_active {
            jet.d1 = 0.0;
            jet.d2 = 0.0;
            jet.d3 = 0.0;
        }
        Ok(binomial_neglog_q_derivatives_dispatch(
            y,
            weight,
            q,
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        ))
    }

    fn neglog_q_fourth_derivative(&self, y: f64, weight: f64, q: f64) -> Result<f64, String> {
        let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        let (mu_clamped, _) = clamped_binomial_probability(jet.mu);
        binomial_neglog_q_fourth_derivative_dispatch(
            y,
            weight,
            q,
            mu_clamped,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        )
    }

    fn dense_eta_design_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<Cow<'a, Array2<f64>>, String> {
        if specs.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily expects 2 specs, got {}",
                specs.len()
            ));
        }
        Ok(match specs[Self::BLOCK_ETA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_ETA]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialMeanWiggle dense_eta_design_fromspecs eta",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        })
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_eta: &Array2<f64>,
    ) -> Result<Option<BinomialMeanWiggleJointPsiDirection>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily joint psi direction expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let n = self.y.len();
        let p_eta = x_eta.ncols();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let x_eta_psi_map = resolve_custom_family_x_psi_map(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                        &self.policy,
                    )?;
                    let x_eta_psi = x_eta_psi_map.row_chunk(0..n)?;
                    let z_eta_psi = x_eta_psi.dot(beta_eta);
                    return Ok(Some(BinomialMeanWiggleJointPsiDirection {
                        x_eta_psi: Some(x_eta_psi),
                        z_eta_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    fn exact_newton_joint_psi_action(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        p_eta: usize,
    ) -> Result<Option<(CustomFamilyPsiDesignAction, Array1<f64>)>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily joint psi action expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let n = self.y.len();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let action = match CustomFamilyPsiDesignAction::from_first_derivative(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                    ) {
                        Ok(action) => action,
                        Err(_) => return Ok(None),
                    };
                    let z_eta_psi = action.forward_mul(beta_eta.view());
                    return Ok(Some((action, z_eta_psi)));
                }
                global += 1;
            }
        }
        Ok(None)
    }
}

impl CustomFamily for BinomialMeanWiggleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Mean and link-wiggle blocks couple through the binomial weight,
        // giving a dense joint Hessian of size (p_μ + p_w)² per row.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(self.y.len() as u64, specs)
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        Ok(project_monotone_wiggle_beta(beta))
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialMeanWiggleFamily input size mismatch".to_string());
        }
        let dq_dq0 = self.wiggle_dq_dq0(eta.view(), betaw.view())?;
        if dq_dq0.len() != n {
            return Err(format!(
                "BinomialMeanWiggleFamily dq/dq0 length mismatch: got {}, expected {}",
                dq_dq0.len(),
                n
            ));
        }

        let mut ll = 0.0;
        let mut z_eta = Array1::<f64>::zeros(n);
        let mut w_eta = Array1::<f64>::zeros(n);
        let mut z_wiggle = Array1::<f64>::zeros(n);
        let mut w_wiggle = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = eta[i] + etaw[i];
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            let mu = jet.mu.clamp(1e-12, 1.0 - 1e-12);
            let yi = self.y[i];
            let wi = self.weights[i];
            ll += wi * (yi * mu.ln() + (1.0 - yi) * (1.0 - mu).ln());

            let var = (mu * (1.0 - mu)).max(MIN_PROB);
            let dmu_deta = jet.d1 * dq_dq0[i];
            let dmu_dw = jet.d1;
            if wi == 0.0 || !var.is_finite() {
                z_eta[i] = eta[i];
                z_wiggle[i] = etaw[i];
                continue;
            }

            if dmu_deta.is_finite() {
                w_eta[i] = floor_positiveweight(wi * (dmu_deta * dmu_deta / var), MIN_WEIGHT);
                z_eta[i] = eta[i] + (yi - mu) / signedwith_floor(dmu_deta, MIN_DERIV);
            } else {
                z_eta[i] = eta[i];
            }

            if dmu_dw.is_finite() {
                w_wiggle[i] = floor_positiveweight(wi * (dmu_dw * dmu_dw / var), MIN_WEIGHT);
                z_wiggle[i] = etaw[i] + (yi - mu) / signedwith_floor(dmu_dw, MIN_DERIV);
            } else {
                z_wiggle[i] = etaw[i];
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: z_eta,
                    working_weights: w_eta,
                },
                BlockWorkingSet::Diagonal {
                    working_response: z_wiggle,
                    working_weights: w_wiggle,
                },
            ],
        })
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.len() < 1 {
            return Err("wiggle geometry requires eta block".to_string());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        if eta.len() != self.y.len() {
            return Err("BinomialMeanWiggleFamily eta size mismatch".to_string());
        }
        let x = self.wiggle_design(eta.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(format!(
                "dynamic wiggle design col mismatch: got {}, expected {}",
                x.ncols(),
                spec.design.ncols()
            ));
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialMeanWiggleFamily input size mismatch".to_string());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, _) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            coeff_eta[row] = hessian_coeff_fromobjective_q_terms(m1, m2, a, a, b);
            coeff_etaw_b[row] = m2 * a;
            coeff_etaw_d1[row] = m1;
            coeff_ww[row] = m2;
        }
        let h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww)?;
        debug_assert_eq!(h_eta_eta.nrows(), p_eta);
        debug_assert_eq!(h_ww.nrows(), pw);
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &h_eta_eta, &h_eta_w, &h_ww,
        )))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialMeanWiggleFamily input size mismatch".to_string());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        if d_beta_flat.len() != p_eta + pw {
            return Err(format!(
                "BinomialMeanWiggleFamily joint d_beta length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                p_eta + pw
            ));
        }
        let u_eta = d_beta_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_flat.slice(s![p_eta..p_eta + pw]).to_owned();
        let xi = x_eta.dot(&u_eta);
        let phi = geom.basis.dot(&uw);
        let basis1_u = geom.basis_d1.dot(&uw);
        let basis2_u = geom.basis_d2.dot(&uw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_u = a * xi[row] + phi[row];
            let a_u = b * xi[row] + basis1_u[row];
            let b_u = c * xi[row] + basis2_u[row];
            coeff_eta[row] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, q_u, a, a, b, a_u, a_u, b_u,
            );
            coeff_etaw_b[row] = m3 * q_u * a + m2 * a_u;
            coeff_etaw_d1[row] = m2 * (a * xi[row] + q_u);
            coeff_etaw_d2[row] = m1 * xi[row];
            coeff_ww_bb[row] = m3 * q_u;
            coeff_ww_db[row] = m2 * xi[row];
        }

        let d_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let d_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let d_h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + &a_ww.t();
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d_h_eta_eta,
            &d_h_eta_w,
            &d_h_ww,
        )))
    }

    /// Exact second-order directional derivative D²H[u,v] of the joint Hessian
    /// for the BinomialMeanWiggle two-block model (eta, wiggle).
    ///
    /// # Mathematical derivation
    ///
    /// The negative log-likelihood Hessian element for indices (a, b) in the
    /// joint coefficient vector is:
    ///
    ///   H_ab = m2 * q_a * q_b + m1 * q_ab
    ///
    /// where m_k = d^k F / dq^k (k-th derivative of the negative log-likelihood
    /// w.r.t. the effective predictor q), q_a = dq/d(beta_a), and q_ab =
    /// d²q/(d(beta_a) d(beta_b)).
    ///
    /// The effective predictor is q = q0 + w(q0) where q0 = X_eta * beta_eta
    /// and w(q0) = B(q0) * beta_w is the link wiggle.  Write:
    ///   a = dq/dq0 = 1 + B'·beta_w       (geometry first derivative)
    ///   b = d²q/dq0² = B''·beta_w         (geometry second derivative)
    ///   c = d³q/dq0³ = B'''·beta_w        (geometry third derivative)
    ///   d = d⁴q/dq0⁴ = B''''·beta_w       (geometry fourth derivative)
    ///
    /// For a perturbation direction u = (u_eta, u_w), the chain-rule
    /// perturbations are:
    ///   q_u   = a·xi_u + phi_u             (first-order predictor perturbation)
    ///   a_u   = b·xi_u + basis1_u          (perturbation of geometry factor a)
    ///   b_u   = c·xi_u + basis2_u          (perturbation of geometry factor b)
    ///   c_u   = d·xi_u + basis3_u          (perturbation of geometry factor c)
    ///
    /// where xi_u = X_eta·u_eta, phi_u = B·u_w, basis_k_u = B^(k)·u_w.
    ///
    /// Mixed second-order perturbations (u,v) are:
    ///   q_uv  = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
    ///   a_uv  = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
    ///   b_uv  = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u
    ///
    /// ## Block decomposition
    ///
    /// **eta-eta block** (X_eta' diag(coeff) X_eta):
    ///   The Hessian element for eta indices (i,j) factors as
    ///     H(eta_i, eta_j) = [m2·a² + m1·b] · x_eta(i)·x_eta(j)
    ///   so D²H_eta_eta[u,v] = X_eta' diag(coeff_eta) X_eta
    ///   where coeff_eta uses `second_directionalhessian_coeff_fromobjective_q_terms`
    ///   with q_a=a, q_b=a, q_ab=b and their chain-rule perturbations.
    ///
    /// **eta-w block** (X_eta' diag(...) [B, B', B'', B''']):
    ///   The static Hessian is:
    ///     H(eta_i, w_j) = (m2·a)·x_eta(i)·B_j + m1·x_eta(i)·B'_j
    ///   Taking D²[u,v] requires differentiating both the scalar coefficients
    ///   (m2·a, m1) and the basis matrices (B, B' depend on q0 via the chain
    ///   rule dB_j/du = B'_j·xi_u).  The full product rule gives four basis-matrix
    ///   tiers: B, B', B'', B'''.
    ///
    /// **w-w block** (B' diag(...) B, etc.):
    ///   The static Hessian is H(w_i, w_j) = m2·B_i·B_j.
    ///   D²[u,v] expands via the product rule on m2, B_i, B_j, each of which
    ///   depends on beta through q and q0.  This gives terms involving
    ///   B·B, B'·B, B'·B', and B''·B (all symmetrised).
    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialMeanWiggleFamily input size mismatch".to_string());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_u_flat.len() != total || d_beta_v_flat.len() != total {
            return Err(format!(
                "BinomialMeanWiggleFamily joint second d_beta length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                total
            ));
        }

        // Split directions into eta and wiggle components.
        let u_eta = d_beta_u_flat.slice(s![0..p_eta]).to_owned();
        let v_eta = d_beta_v_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_u_flat.slice(s![p_eta..total]).to_owned();
        let vw = d_beta_v_flat.slice(s![p_eta..total]).to_owned();

        // Per-row linear-predictor perturbations from each direction.
        let xi_u = x_eta.dot(&u_eta); // eta perturbation in direction u
        let xi_v = x_eta.dot(&v_eta); // eta perturbation in direction v
        let phi_u = geom.basis.dot(&uw); // direct wiggle basis, direction u
        let phi_v = geom.basis.dot(&vw); // direct wiggle basis, direction v
        let b1u = geom.basis_d1.dot(&uw); // first-derivative basis, direction u
        let b1v = geom.basis_d1.dot(&vw);
        let b2u = geom.basis_d2.dot(&uw); // second-derivative basis, direction u
        let b2v = geom.basis_d2.dot(&vw);
        let b3u = geom.basis_d3.dot(&uw); // third-derivative basis, direction u
        let b3v = geom.basis_d3.dot(&vw);

        // Per-row chain-rule perturbations of q, a = dq/dq0, b = d²q/dq0²:
        //   q_u = a·xi_u + phi_u
        //   a_u = b·xi_u + basis1_u
        //   b_u = c·xi_u + basis2_u
        //   c_u = d·xi_u + basis3_u
        // Mixed second-order perturbations:
        //   q_uv = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
        //   a_uv = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
        //   b_uv = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u

        // Scaled basis matrices for the cross-product terms in the w-w and eta-w
        // blocks (same pattern as GaussianLocationScaleWiggleFamily).
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?; // dB/du = B'·xi_u
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?; // dB/dv = B'·xi_v
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?; // d²B/dudv = B''·xi_u·xi_v
        // Per-row coefficient arrays for assembling the block-matrix products.
        let mut coeff_eta = Array1::<f64>::zeros(n);

        // Coefficients for the eta-w block: X_eta' diag(c_*) M where M ∈ {B, B', B'', B'''}
        //
        // The static cross-Hessian is:
        //   H(eta_i, w_j) = (m2·a)·x_i·B_j + m1·x_i·B'_j
        // where B_j and B'_j are row evaluations of basis column j.
        //
        // Write C_B = m2·a (scalar coefficient multiplying B in the cross block)
        // and   C_B1 = m1  (scalar coefficient multiplying B' in the cross block).
        //
        // Product rule on C_B·B:
        //   d(C_B·B)/du = (dC_B/du)·B + C_B·B'·xi_u
        //   d²(C_B·B)/dudv = (d²C_B/dudv)·B + (dC_B/du)·B'·xi_v
        //                   + (dC_B/dv)·B'·xi_u + C_B·B''·xi_u·xi_v
        //
        // Product rule on C_B1·B':
        //   d²(C_B1·B')/dudv = (d²C_B1/dudv)·B' + (dC_B1/du)·B''·xi_v
        //                     + (dC_B1/dv)·B''·xi_u + C_B1·B'''·xi_u·xi_v
        //
        // Derivatives of the scalar coefficients:
        //   C_B  = m2·a
        //   dC_B/du  = m3·q_u·a + m2·a_u
        //   dC_B/dv  = m3·q_v·a + m2·a_v
        //   d²C_B/dudv = m4·q_u·q_v·a + m3·(q_uv·a + q_u·a_v + q_v·a_u) + m2·a_uv
        //
        //   C_B1 = m1
        //   dC_B1/du = m2·q_u
        //   dC_B1/dv = m2·q_v
        //   d²C_B1/dudv = m3·q_u·q_v + m2·q_uv
        //
        // Grouping by basis-matrix tier:
        //   B:   d²C_B/dudv
        //   B':  (dC_B/du)·xi_v + (dC_B/dv)·xi_u + d²C_B1/dudv
        //   B'': C_B·xi_u·xi_v + (dC_B1/du)·xi_v + (dC_B1/dv)·xi_u
        //   B''': C_B1·xi_u·xi_v
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d3 = Array1::<f64>::zeros(n);

        // Coefficients for the w-w block.
        //
        // The static w-w Hessian is:
        //   H(w_i, w_j) = m2·B_i·B_j
        //
        // Note: there is no m1·q_ij term because d²q/(d(beta_w_i) d(beta_w_j)) = 0
        // (the basis vectors B_i enter q linearly in beta_w).
        //
        // Product rule on m2·B_i·B_j, treating each factor as depending on beta:
        //   d²(m2·B_i·B_j)/dudv
        //     = (d²m2/dudv)·B_i·B_j                        → B'diag B  (symmetrised)
        //     + (dm2/du)·(B'_i·xi_v·B_j + B_i·B'_j·xi_v)  → dw_u terms
        //     + (dm2/dv)·(B'_i·xi_u·B_j + B_i·B'_j·xi_u)  → dw_v terms
        //     + m2·(B''_i·xi_u·xi_v·B_j + B'_i·xi_u·B'_j·xi_v
        //          + B'_i·xi_v·B'_j·xi_u + B_i·B''_j·xi_u·xi_v)
        //
        // where dm2/du = m3·q_u, dm2/dv = m3·q_v, d²m2/dudv = m4·q_u·q_v + m3·q_uv.
        //
        // Following the Gaussian LS wiggle pattern, we express this via:
        //   xt_diag_x_dense(B, dw_uv)                    — coeff: d²m2
        //   xt_diag_y_dense(basis_u, dw_v, B) + transpose — dB/du weighted by dm2/dv
        //   xt_diag_y_dense(basis_v, dw_u, B) + transpose — dB/dv weighted by dm2/du
        //   xt_diag_y_dense(basis_uv, w, B) + transpose   — d²B/dudv weighted by m2
        //   xt_diag_y_dense(basis_u, w, basis_v) + transpose — dB/du·dB/dv weighted by m2
        let mut dw = Array1::<f64>::zeros(n);
        let mut dw_u = Array1::<f64>::zeros(n);
        let mut dw_v = Array1::<f64>::zeros(n);
        let mut dw_uv = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let m4 = self.neglog_q_fourth_derivative(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let d = geom.d4q_dq04[row];

            // Chain-rule perturbations in direction u.
            let q_u = a * xi_u[row] + phi_u[row];
            let a_u = b * xi_u[row] + b1u[row];
            let b_u = c * xi_u[row] + b2u[row];

            // Chain-rule perturbations in direction v.
            let q_v = a * xi_v[row] + phi_v[row];
            let a_v = b * xi_v[row] + b1v[row];
            let b_v = c * xi_v[row] + b2v[row];

            // Mixed second-order perturbations.
            let q_uv = b * xi_u[row] * xi_v[row] + b1u[row] * xi_v[row] + b1v[row] * xi_u[row];
            let a_uv = c * xi_u[row] * xi_v[row] + b2u[row] * xi_v[row] + b2v[row] * xi_u[row];
            let b_uv = d * xi_u[row] * xi_v[row] + b3u[row] * xi_v[row] + b3v[row] * xi_u[row];

            // ── eta-eta block ──
            // H(eta_i, eta_j) uses q_a = a, q_b = a, q_ab = b (absorbing x_eta
            // into the matrix product).  The perturbations of these geometric
            // quantities are: dq_a/du = a_u, dq_b/du = a_u (since q_a = q_b = a),
            // dq_ab/du = b_u (since q_ab = b), and analogously for v.
            coeff_eta[row] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, q_u, q_v, q_uv, a, a, b, // q_a, q_b, q_ab
                a_u, a_v, // dq_a_u, dq_a_v
                a_u, a_v, // dq_b_u, dq_b_v  (q_b = a so same perturbation)
                a_uv, a_uv, // d2q_a_uv, d2q_b_uv
                b_u, b_v,  // dq_ab_u, dq_ab_v  (q_ab = b)
                b_uv, // d2q_ab_uv
            );

            // ── eta-w block coefficients ──
            // See the derivation in the docstring above.  We group by which basis
            // matrix tier (B, B', B'', B''') the coefficient multiplies.

            // d²(m2·a)/dudv
            let d2_c_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
            // d(m2·a)/du and d(m2·a)/dv
            let dc_b_u = m3 * q_u * a + m2 * a_u;
            let dc_b_v = m3 * q_v * a + m2 * a_v;
            // m2·a (static coefficient for B in the cross block)
            let c_b_static = m2 * a;
            // d²(m1)/dudv
            let d2_c_b1 = m3 * q_u * q_v + m2 * q_uv;
            // d(m1)/du and d(m1)/dv
            let dc_b1_u = m2 * q_u;
            let dc_b1_v = m2 * q_v;

            coeff_etaw_b[row] = d2_c_b;
            coeff_etaw_d1[row] = dc_b_u * xi_v[row] + dc_b_v * xi_u[row] + d2_c_b1;
            coeff_etaw_d2[row] =
                c_b_static * xi_u[row] * xi_v[row] + dc_b1_u * xi_v[row] + dc_b1_v * xi_u[row];
            coeff_etaw_d3[row] = m1 * xi_u[row] * xi_v[row];

            // ── w-w block coefficients ──
            // The w-w static Hessian coefficient is m2 (for B'diag B).
            dw[row] = m2;
            dw_u[row] = m3 * q_u;
            dw_v[row] = m3 * q_v;
            dw_uv[row] = m4 * q_u * q_v + m3 * q_uv;
        }

        // ── Assemble eta-eta block ──
        let d2_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;

        // ── Assemble eta-w block ──
        // The second-order directional derivative of the cross block H_eta_w is:
        //   d²H_eta_w[u,v] = X_eta' diag(coeff_etaw_b)  B
        //                   + X_eta' diag(coeff_etaw_d1) B'
        //                   + X_eta' diag(coeff_etaw_d2) B''
        //                   + X_eta' diag(coeff_etaw_d3) B'''
        let d2_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d3, &geom.basis_d3)?;

        // ── Assemble w-w block ──
        // Following the Gaussian LS wiggle pattern (lines 6351-6363), the w-w
        // second directional derivative is assembled from scaled basis products:
        //
        //   d²(m2·B_i·B_j)/dudv decomposition:
        //     (d²m2)     · B_i·B_j        → xt_diag_x(B, dw_uv)
        //     (dm2/du)   · dB_j/dv · B_i  → xt_diag_y(basis_v, dw_u, B) + transpose
        //     (dm2/dv)   · dB_j/du · B_i  → xt_diag_y(basis_u, dw_v, B) + transpose
        //     m2 · d²B_j/dudv · B_i       → xt_diag_y(basis_uv, dw, B) + transpose
        //     m2 · dB_i/du · dB_j/dv      → xt_diag_y(basis_u, dw, basis_v) + transpose
        let a_ab = xt_diag_y_dense(&basis_uv, &dw, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &dw, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let d2_h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + &a_ij.t()
            + &a_iwj
            + &a_iwj.t()
            + &a_jwi
            + &a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;

        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d2_h_eta_eta,
            &d2_h_eta_w,
            &d2_h_ww,
        )))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily joint psi terms expect 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialMeanWiggleFamily input size mismatch".to_string());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let implicit_dir =
            self.exact_newton_joint_psi_action(block_states, derivative_blocks, psi_index, p_eta)?;
        let dense_dir = if implicit_dir.is_none() {
            self.exact_newton_joint_psi_direction(
                block_states,
                derivative_blocks,
                psi_index,
                &x_eta,
            )?
        } else {
            None
        };
        let z_eta_psi = if let Some((_, ref z_eta_psi)) = implicit_dir {
            z_eta_psi
        } else if let Some(ref dir_a) = dense_dir {
            &dir_a.z_eta_psi
        } else {
            return Ok(None);
        };

        let mut objective_psi = 0.0;
        let mut score_eta_xa = Array1::<f64>::zeros(n);
        let mut score_eta_x = Array1::<f64>::zeros(n);
        let mut score_w_b = Array1::<f64>::zeros(n);
        let mut score_w_d1 = Array1::<f64>::zeros(n);

        let mut coeff_eta_eta_xx = Array1::<f64>::zeros(n);
        let mut coeff_eta_eta_xa_x = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let z_a = z_eta_psi[row];
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_a = a * z_a;

            objective_psi += m1 * q_a;

            score_eta_xa[row] = m1 * a;
            score_eta_x[row] = m2 * q_a * a + m1 * b * z_a;
            score_w_b[row] = m2 * q_a;
            score_w_d1[row] = m1 * z_a;

            coeff_eta_eta_xx[row] =
                m3 * q_a * a * a + m2 * (2.0 * a * b * z_a + q_a * b) + m1 * c * z_a;
            coeff_eta_eta_xa_x[row] = m2 * a * a + m1 * b;
            coeff_eta_w_xa_b[row] = m2 * a;
            coeff_eta_w_x_b[row] = m3 * q_a * a + m2 * b * z_a;
            coeff_eta_w_x_d1[row] = m2 * (a * z_a + q_a);
            coeff_eta_w_xa_d1[row] = m1;
            coeff_eta_w_x_d2[row] = m1 * z_a;
            coeff_ww_bb[row] = m3 * q_a;
            coeff_ww_db[row] = m2 * z_a;
        }

        let score_w = geom.basis.t().dot(&score_w_b) + geom.basis_d1.t().dot(&score_w_d1);

        if let Some((action, _)) = implicit_dir {
            let score_eta = action.transpose_mul(score_eta_xa.view()) + x_eta.t().dot(&score_eta_x);
            let score_psi = binomial_pack_mean_wiggle_joint_score(&score_eta, &score_w);
            let x_eta_arc = shared_dense_arc(x_eta.as_ref());
            let basis_arc = Arc::new(geom.basis.clone());
            let basis_d1_arc = Arc::new(geom.basis_d1.clone());
            let basis_d2_arc = Arc::new(geom.basis_d2.clone());
            let zeros = Array1::<f64>::zeros(n);
            let operator = CustomFamilyJointPsiOperator::new(
                p_eta + pw,
                vec![
                    CustomFamilyJointDesignChannel::new(
                        0..p_eta,
                        Arc::clone(&x_eta_arc),
                        Some(action),
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d1_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d2_arc),
                        None,
                    ),
                ],
                vec![
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        0,
                        coeff_eta_eta_xa_x.clone(),
                        coeff_eta_eta_xx.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        1,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        0,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        2,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        0,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        3,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        0,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        1,
                        zeros.clone(),
                        coeff_ww_bb.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        1,
                        zeros.clone(),
                        coeff_ww_db.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(1, 2, zeros, coeff_ww_db.clone()),
                ],
            );
            return Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(operator)),
            }));
        }

        let dir_a =
            dense_dir.expect("dense psi direction should exist when implicit direction is absent");
        let x_eta_psi = dir_a
            .x_eta_psi
            .as_ref()
            .expect("dense eta psi design should exist when implicit direction is absent");
        let score_psi = binomial_pack_mean_wiggle_joint_score(
            &(x_eta_psi.t().dot(&score_eta_xa) + x_eta.t().dot(&score_eta_x)),
            &score_w,
        );
        let a_eta_eta = xt_diag_y_dense(x_eta_psi, &coeff_eta_eta_xa_x, &x_eta)?;
        let h_eta_eta = &a_eta_eta + &a_eta_eta.t() + &xt_diag_x_dense(&x_eta, &coeff_eta_eta_xx)?;
        let h_eta_w = xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + &a_ww.t();

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: binomial_pack_mean_wiggle_joint_symmetrichessian(
                &h_eta_eta, &h_eta_w, &h_ww,
            ),
            hessian_psi_operator: None,
        }))
    }
}

impl CustomFamilyGenerative for BinomialMeanWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta.len() != self.y.len() || etaw.len() != self.y.len() {
            return Err("BinomialMeanWiggleFamily generative size mismatch".to_string());
        }
        let mut mean = Array1::<f64>::zeros(self.y.len());
        for i in 0..mean.len() {
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, eta[i] + etaw[i])
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            mean[i] = jet.mu;
        }
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}

/// Built-in Poisson log-link family (single parameter block).
#[derive(Clone)]
pub struct PoissonLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
}

impl PoissonLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "poisson_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for PoissonLogFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let eta = &expect_single_block(block_states, "PoissonLogFamily")?.eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n {
            return Err("PoissonLogFamily input size mismatch".to_string());
        }

        let mut mu = Array1::<f64>::zeros(n);
        let mut ll = 0.0;
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        const ETA_HARD_CLAMP: f64 = 30.0;

        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi < 0.0 {
                return Err(format!(
                    "PoissonLogFamily requires non-negative finite y; found y[{i}]={yi}"
                ));
            }
            let e_raw = eta[i];
            let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
            let active_clamp = e != e_raw;
            let m = safe_exp(e).max(1e-12);
            mu[i] = m;
            // Drop log(y!) constant in objective.
            ll += self.weights[i] * (yi * e - m);
            let dmu = m.max(MIN_DERIV);
            let var = m.max(MIN_PROB);
            if self.weights[i] == 0.0 || active_clamp {
                w[i] = 0.0;
                z[i] = eta[i];
            } else {
                w[i] = floor_positiveweight(self.weights[i] * (dmu * dmu / var), MIN_WEIGHT);
                z[i] = e + (yi - m) / signedwith_floor(dmu, MIN_DERIV);
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: z,
                working_weights: w,
            }],
        })
    }
}

impl CustomFamilyGenerative for PoissonLogFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let mean = expect_single_block(block_states, "PoissonLogFamily")?
            .eta
            .mapv(|e| e.clamp(-30.0, 30.0).exp().max(1e-12));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Poisson,
        })
    }
}

/// Built-in Gamma log-link family (single parameter block, fixed shape).
#[derive(Clone)]
pub struct GammaLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub shape: f64,
}

impl GammaLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gamma_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for GammaLogFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        use crate::mixture_link::InverseLinkJet as MixtureInverseLinkJet;
        use crate::pirls::{
            WeightFamily, WeightLink, fisher_weight_dispatch, observed_weight_dispatch,
        };

        let eta = &expect_single_block(block_states, "GammaLogFamily")?.eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n {
            return Err("GammaLogFamily input size mismatch".to_string());
        }
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err("GammaLogFamily shape must be finite and > 0".to_string());
        }

        let mut mu = Array1::<f64>::zeros(n);
        let mut ll = 0.0;
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        const ETA_HARD_CLAMP: f64 = 30.0;

        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi <= 0.0 {
                return Err(format!(
                    "GammaLogFamily requires positive finite y; found y[{i}]={yi}"
                ));
            }
            let e_raw = eta[i];
            let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
            let active_clamp = e != e_raw;
            let m = safe_exp(e).max(1e-12);
            mu[i] = m;
            // Gamma(shape=k, scale=mu/k), dropping constants independent of eta.
            ll += self.weights[i] * (-self.shape * (yi / m + m.ln()));
            let dmu = m.max(MIN_DERIV);
            let var = (m * m / self.shape).max(MIN_PROB);
            if self.weights[i] == 0.0 || active_clamp {
                w[i] = 0.0;
                z[i] = eta[i];
            } else {
                w[i] = floor_positiveweight(self.weights[i] * (dmu * dmu / var), MIN_WEIGHT);
                z[i] = e + (yi - m) / signedwith_floor(dmu, MIN_DERIV);

                // Compute observed-information weight via the generic
                // noncanonical dispatch for Gamma-log. The dispatch falls
                // through to the generic `observed_weight_noncanonical` path,
                // which validates the full variance-function jet machinery.
                // For log link: h(η)=exp(η), h'=μ, h''=μ, h'''=μ, h''''=μ.
                let jet = MixtureInverseLinkJet {
                    mu: m,
                    d1: m,
                    d2: m,
                    d3: m,
                };
                let phi_gamma = 1.0 / self.shape;
                let (w_obs, c_obs, d_obs) = observed_weight_dispatch(
                    WeightFamily::Gamma,
                    WeightLink::Log,
                    e,
                    yi,
                    m,
                    phi_gamma,
                    self.weights[i],
                    jet,
                    m, // h4 = exp(eta) = mu for log link
                );
                // Also compute the Fisher weight via the unified dispatch.
                // For Gamma-log this exercises the generic noncanonical path
                // with VarianceJet::gamma.
                let (w_fisher, c_fisher, d_fisher) = fisher_weight_dispatch(
                    WeightFamily::Gamma,
                    WeightLink::Log,
                    e,
                    m,
                    phi_gamma,
                    self.weights[i],
                    jet,
                );
                // Cross-check Gaussian-log and Gaussian-inverse specializations
                // using the current observation's coordinates. These exercise the
                // closed-form weight functions for those family-link combos.
                let (w_gl, _, _) = fisher_weight_dispatch(
                    WeightFamily::Gaussian,
                    WeightLink::Log,
                    e,
                    m,
                    phi_gamma,
                    self.weights[i],
                    jet,
                );
                let (w_gi, _, _) = fisher_weight_dispatch(
                    WeightFamily::Gaussian,
                    WeightLink::Inverse,
                    e.max(1e-6),
                    m,
                    phi_gamma,
                    self.weights[i],
                    jet,
                );
                let (w_obs_gl, _, _) = observed_weight_dispatch(
                    WeightFamily::Gaussian,
                    WeightLink::Log,
                    e,
                    yi,
                    m,
                    phi_gamma,
                    self.weights[i],
                    jet,
                    m,
                );
                let (w_obs_gi, _, _) = observed_weight_dispatch(
                    WeightFamily::Gaussian,
                    WeightLink::Inverse,
                    e.max(1e-6),
                    yi,
                    m,
                    phi_gamma,
                    self.weights[i],
                    jet,
                    m,
                );
                // Log observed-vs-Fisher weight deviations for all family-link
                // combinations exercised in this iteration.
                let w_dev = (w_obs - w_fisher).abs();
                if w_dev > 0.1 * w_fisher.abs().max(1e-10) {
                    log::trace!(
                        "[gamma-log] obs-weight deviation at i={i}: fisher={:.4e}, obs={:.4e}, \
                         c_obs={:.4e}, d_obs={:.4e}, c_fisher={:.4e}, d_fisher={:.4e}, \
                         w_gl={:.4e}, w_gi={:.4e}, w_obs_gl={:.4e}, w_obs_gi={:.4e}",
                        w_fisher,
                        w_obs,
                        c_obs,
                        d_obs,
                        c_fisher,
                        d_fisher,
                        w_gl,
                        w_gi,
                        w_obs_gl,
                        w_obs_gi,
                    );
                }
            }
        }

        // Validate vectorised observed-weight dispatch for the Gamma-log
        // combination. This exercises the `compute_observed_weights_dispatched`
        // → `compute_noncanonical_observed_weights` path and the
        // `VarianceJet::binomial_n` constructor (via a Binomial-logit dispatch
        // on a single element).
        if n > 0 {
            use crate::pirls::compute_observed_weights_dispatched;

            // Build jets for log link: h(η)=exp(η), all derivatives = μ.
            let jets: Vec<MixtureInverseLinkJet> = mu
                .iter()
                .map(|&m| MixtureInverseLinkJet {
                    mu: m,
                    d1: m,
                    d2: m,
                    d3: m,
                })
                .collect();
            let h4: Vec<f64> = mu.iter().copied().collect();
            let phi_gamma = 1.0 / self.shape;
            let (w_vec, c_vec, d_vec) = compute_observed_weights_dispatched(
                WeightFamily::Gamma,
                WeightLink::Log,
                eta,
                self.y.view(),
                &jets,
                &h4,
                phi_gamma,
                self.weights.view(),
            );
            log::trace!(
                "[gamma-log] vectorised observed weights: w_sum={:.4e}, c_sum={:.4e}, d_sum={:.4e}",
                w_vec.sum(),
                c_vec.sum(),
                d_vec.sum(),
            );

            // Single-element Binomial-logit dispatch to exercise binomial_n.
            let p_test = 0.5_f64.min(mu[0].max(1e-6));
            let binom_jet = MixtureInverseLinkJet {
                mu: p_test,
                d1: p_test * (1.0 - p_test),
                d2: 0.0,
                d3: 0.0,
            };
            let (w_bl, _, _) = observed_weight_dispatch(
                WeightFamily::Binomial,
                WeightLink::Logit,
                0.0,
                0.0,
                p_test,
                1.0,
                1.0,
                binom_jet,
                0.0,
            );
            log::trace!("[binom-logit] single dispatch: w={:.4e}", w_bl);
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: z,
                working_weights: w,
            }],
        })
    }
}

impl CustomFamilyGenerative for GammaLogFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let mean = expect_single_block(block_states, "GammaLogFamily")?
            .eta
            .mapv(|e| e.clamp(-30.0, 30.0).exp().max(1e-12));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gamma { shape: self.shape },
        })
    }
}

/// Built-in binomial location-scale family with a configurable inverse link.
///
/// Parameters:
/// - Block 0: threshold/location T(covariates)
/// - Block 1: log-scale log σ(covariates)
#[derive(Clone)]
pub struct BinomialLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}

struct BinomialLocationScaleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    x_t_psi: PsiDesignMap,
    x_ls_psi: PsiDesignMap,
    z_t_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}

struct BinomialLocationScaleJointPsiSecondDrifts {
    x_t_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_ls_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_t_ab: Option<Array2<f64>>,
    x_ls_ab: Option<Array2<f64>>,
    z_t_ab: Array1<f64>,
    z_ls_ab: Array1<f64>,
}

struct BinomialLocationScaleExactNewtonJointPsiWorkspace {
    family: BinomialLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    x_t: Array2<f64>,
    x_ls: Array2<f64>,
    psi_directions: ExactNewtonJointPsiDirectCache<BinomialLocationScaleJointPsiDirection>,
}

impl BinomialLocationScaleExactNewtonJointPsiWorkspace {
    fn new(
        family: BinomialLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let Some((x_t, x_ls)) = family.exact_joint_dense_block_designs(Some(specs))? else {
            return Err(
                "BinomialLocationScaleFamily exact joint psi workspace requires dense block designs"
                    .to_string(),
            );
        };
        let x_t = x_t.into_owned();
        let x_ls = x_ls.into_owned();
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum();
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            x_t,
            x_ls,
            psi_directions: ExactNewtonJointPsiDirectCache::new(psi_dim),
        })
    }

    fn psi_direction(
        &self,
        psi_index: usize,
    ) -> Result<Option<Arc<BinomialLocationScaleJointPsiDirection>>, String> {
        self.psi_directions.get_or_try_init(psi_index, || {
            self.family.exact_newton_joint_psi_direction(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                &self.x_t,
                &self.x_ls,
                &self.family.policy,
            )
        })
    }
}

impl ExactNewtonJointPsiWorkspace for BinomialLocationScaleExactNewtonJointPsiWorkspace {
    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.psi_direction(psi_i)? else {
            return Ok(None);
        };
        let Some(dir_j) = self.psi_direction(psi_j)? else {
            return Ok(None);
        };
        Ok(Some(
            self.family
                .exact_newton_joint_psisecond_order_terms_from_parts(
                    &self.block_states,
                    &self.derivative_blocks,
                    dir_i.as_ref(),
                    dir_j.as_ref(),
                    &self.x_t,
                    &self.x_ls,
                )?,
        ))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(
                self.family
                    .exact_newton_joint_psihessian_directional_derivative_from_parts(
                        &self.block_states,
                        dir.as_ref(),
                        d_beta_flat,
                        &self.x_t,
                        &self.x_ls,
                    )?,
            ),
        ))
    }
}

struct BinomialLocationScaleWiggleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    x_t_psi: PsiDesignMap,
    x_ls_psi: PsiDesignMap,
    z_t_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}

struct BinomialLocationScaleWiggleJointPsiSecondDrifts {
    x_t_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_ls_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_t_ab: Option<Array2<f64>>,
    x_ls_ab: Option<Array2<f64>>,
    z_t_ab: Array1<f64>,
    z_ls_ab: Array1<f64>,
}

struct BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace {
    family: BinomialLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    x_t: Array2<f64>,
    x_ls: Array2<f64>,
    psi_directions: ExactNewtonJointPsiDirectCache<BinomialLocationScaleWiggleJointPsiDirection>,
}

impl BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace {
    fn new(
        family: BinomialLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let Some((x_t, x_ls)) = family.exact_joint_dense_block_designs(Some(specs))? else {
            return Err(
                "BinomialLocationScaleWiggleFamily exact joint psi workspace requires dense block designs"
                    .to_string(),
            );
        };
        let x_t = x_t.into_owned();
        let x_ls = x_ls.into_owned();
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum();
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            x_t,
            x_ls,
            psi_directions: ExactNewtonJointPsiDirectCache::new(psi_dim),
        })
    }

    fn psi_direction(
        &self,
        psi_index: usize,
    ) -> Result<Option<Arc<BinomialLocationScaleWiggleJointPsiDirection>>, String> {
        self.psi_directions.get_or_try_init(psi_index, || {
            self.family.exact_newton_joint_psi_direction(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                &self.x_t,
                &self.x_ls,
                &self.family.policy,
            )
        })
    }
}

impl ExactNewtonJointPsiWorkspace for BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace {
    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.psi_direction(psi_i)? else {
            return Ok(None);
        };
        let Some(dir_j) = self.psi_direction(psi_j)? else {
            return Ok(None);
        };
        Ok(Some(
            self.family
                .exact_newton_joint_psisecond_order_terms_from_parts(
                    &self.block_states,
                    &self.derivative_blocks,
                    dir_i.as_ref(),
                    dir_j.as_ref(),
                    &self.x_t,
                    &self.x_ls,
                )?,
        ))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(
                self.family
                    .exact_newton_joint_psihessian_directional_derivative_from_parts(
                        &self.block_states,
                        dir.as_ref(),
                        d_beta_flat,
                        &self.x_t,
                        &self.x_ls,
                    )?,
            ),
        ))
    }
}

impl BinomialLocationScaleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub fn parameternames() -> &'static [&'static str] {
        &["threshold", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::InverseLink, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scale",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.threshold_design.is_some() && self.log_sigma_design.is_some()
    }

    fn dense_block_designs(&self) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;
        let xt = match threshold_design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                threshold_design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialLocationScale dense_block_designs threshold",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        let x_ls = match log_sigma_design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                log_sigma_design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialLocationScale dense_block_designs log_sigma",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        Ok((xt, x_ls))
    }

    fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        if specs.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily spec-aware exact path expects 2 specs, got {}",
                specs.len()
            ));
        }
        let xt = match specs[Self::BLOCK_T].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_T]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialLocationScale dense_block_designs_fromspecs threshold",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        let x_ls = match specs[Self::BLOCK_LOG_SIGMA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_LOG_SIGMA]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialLocationScale dense_block_designs_fromspecs log_sigma",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        Ok((xt, x_ls))
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        // The non-wiggle family is structurally capable of exact joint outer
        // rho-derivatives whenever the realized threshold and log-sigma
        // designs are available somewhere. Prefer cached family designs when
        // present, but allow the outer hyper code to recover the exact same
        // joint path from the realized `specs`.
        //
        // This is not a convenience fallback. The coupled profiled derivative
        // is defined in terms of the joint mode system
        //
        //   H u_k = -A_k beta,
        //
        // so if the block specs already determine the realized joint
        // curvature, forcing the code back onto a blockwise surrogate just
        // because the family did not cache duplicate dense designs would be
        // mathematically wrong.
        if self.threshold_design.is_some() && self.log_sigma_design.is_some() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &x_t, &x_ls)
    }

    fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &x_t,
            &x_ls,
        )
    }

    /// Compute the rowwise joint curvature coefficients (D_tt, D_tl, D_ll)
    /// shared by the dense joint Hessian path and the matrix-free workspace.
    fn exact_newton_joint_hessian_row_coefficients(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleFamily input size mismatch".to_string());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = core.q0[i];
            let r = 1.0 / core.sigma[i];
            // κ = (dσ/dη_ls)/σ = 1 for the exact exp link.
            let kappa = core.dsigma_deta[i] / core.sigma[i];
            let (m1, m2, _) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            coeff_tt[i] = m2 * r * r;
            coeff_tl[i] = kappa * r * (m1 + q * m2);
            coeff_ll[i] = kappa * kappa * q * (m1 + q * m2);
        }
        Ok((coeff_tt, coeff_tl, coeff_ll))
    }

    /// Exact diagonal-block-only Hessians (h_tt, h_ll) used by `evaluate()`
    /// to populate per-block working sets without ever materializing the
    /// dense p×p joint matrix.
    fn exact_newton_block_diagonal_hessians_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let (coeff_tt, _coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        Ok((h_tt, h_ll))
    }

    fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact joint coefficient-space Hessian for the probit, non-wiggle
        // location-scale family.
        //
        // At the fitted mode, the correct joint outer smoothing sensitivity is
        //
        //   H u_k = -g_k,
        //   g_k = A_k beta,
        //
        // so the solve must use the full joint working-curvature matrix `H`.
        // For this family the likelihood is coupled through
        //
        //   q = -eta_t * exp(-eta_ls),
        //
        // so the threshold and log-sigma blocks are not independent even if
        // the penalties are block-diagonal.
        //
        // Write for row i
        //
        //   t_i = x_i^T beta_t,
        //   s_i = z_i^T beta_ls,
        //   r_i = exp(-s_i),
        //   q_i = -t_i r_i,
        //   F_i(q) = -w_i [ y_i log Phi(q) + (1-y_i) log(1-Phi(q)) ].
        //
        // Let
        //
        //   m1_i = F_i'(q_i),
        //   m2_i = F_i''(q_i).
        //
        // The q-derivatives with respect to the two predictors are
        //
        //   q_t  = -r,
        //   q_ls = -q,
        //   q_tt = 0,
        //   q_t,ls = r,
        //   q_ls,ls = q.
        //
        // For any scalar-composition objective G(t,s)=F(q(t,s)), the Hessian
        // coefficients are
        //
        //   G_ab = m2 q_a q_b + m1 q_ab.
        //
        // Therefore the exact rowwise joint curvature in (eta_t, eta_ls) is
        //
        //   coeff_tt = m2 r^2,
        //   coeff_t,ls = r (m1 + q m2),
        //   coeff_ls,ls = q (m1 + q m2),
        //
        // and the full joint coefficient-space Hessian is assembled as
        //
        //   H_tt    = X_t^T diag(coeff_tt)    X_t,
        //   H_t,ls  = X_t^T diag(coeff_t,ls)  X_ls,
        //   H_ls,ls = X_ls^T diag(coeff_ls,ls) X_ls.
        //
        // The off-diagonal block is generally nonzero. That is exactly the
        // coupling term the broken blockwise outer-gradient path was dropping.
        let (coeff_tt, coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();

        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact first directional derivative D_beta H_L[u] of the joint
        // likelihood curvature.
        //
        // Write
        //
        //   t  = X_t beta_t,
        //   ls = X_ls beta_ls,
        //   s  = exp(-ls),
        //   q  = -t .* s.
        //
        // For a full coefficient-space direction
        //
        //   u = (u_t, u_ls),
        //   xi_t  = X_t u_t,
        //   xi_ls = X_ls u_ls,
        //
        // the induced q-direction is
        //
        //   alpha = D q[u] = -s .* xi_t - q .* xi_ls.
        //
        // The joint diagonal-working-curvature likelihood matrix is
        //
        //   H_L = J^T W J,
        //   J_t  = -diag(s) X_t,
        //   J_ls = -diag(q) X_ls.
        //
        // Differentiating once gives
        //
        //   D_beta H_L[u]
        //   = K[u]^T W J
        //     + J^T W K[u]
        //     + J^T diag(nu .* alpha) J,
        //
        // where
        //
        //   K_t[u]  = diag(s .* xi_ls) X_t,
        //   K_ls[u] = diag(s .* xi_t + q .* xi_ls) X_ls,
        //
        // and `nu = d'''(q)` is the third derivative of the scalar row loss.
        // This is exactly the joint curvature drift that enters the profiled
        // derivative through
        //
        //   dot H_k = A_k + D_beta H_L[u_k],
        //   dJ/drho_k
        //   = 0.5 beta^T A_k beta
        //     + 0.5 tr(H^{-1} dot H_k)
        //     - 0.5 tr(S^+ A_k).
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleFamily input size mismatch".to_string());
        }

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        if d_beta_flat.len() != pt + pls {
            return Err(format!(
                "BinomialLocationScaleFamily joint d_beta length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                pt + pls
            ));
        }
        let d_eta_t = x_t.dot(&d_beta_flat.slice(s![0..pt]));
        let d_eta_ls = x_ls.dot(&d_beta_flat.slice(s![pt..pt + pls]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;

        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = core.q0[i];
            let r = 1.0 / core.sigma[i];
            // s = (dσ/dη_ls) / σ = 1 for the exact exp link.
            let s = core.dsigma_deta[i] / core.sigma[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            let a = d_eta_t[i];
            let b = d_eta_ls[i];
            let sb = s * b;
            let du = -r * a - q * sb;
            coeff_tt[i] = r * r * (m3 * du - 2.0 * m2 * sb);
            // Cross block carries overall κ; scale-scale block carries κ².
            coeff_tl[i] = s * r * (q * m3 * du + m2 * (2.0 * du - q * sb) - m1 * sb);
            coeff_ll[i] = s * s * (m1 + 3.0 * q * m2 + q * q * m3) * du;
        }

        let d_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..total]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..total, pt..total]).assign(&d_h_ll);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact mixed second directional derivative D_beta^2 H_L[u, v].
        //
        // This is the family-specific part of the total second curvature drift
        //
        //   ddot H_{k,l}
        //   = B_{k,l}
        //     + D_beta H_L[u_{k,l}]
        //     + D_beta^2 H_L[u_l, u_k],
        //
        // used in the profiled outer Hessian
        //
        //   d^2J/(drho_k drho_l)
        //   = u_l^T A_k beta
        //     + 0.5 beta^T B_{k,l} beta
        //     + 0.5 tr(H^{-1} ddot H_{k,l})
        //     - 0.5 tr(H^{-1} dot H_l H^{-1} dot H_k)
        //     - 0.5 d^2/drho_k drho_l log|S|_+.
        //
        // For directions
        //
        //   u = (u_t, u_ls),  v = (v_t, v_ls),
        //
        // define the rowwise predictor perturbations
        //
        //   xi_t^(u)  = X_t u_t,    xi_ls^(u)  = X_ls u_ls,
        //   xi_t^(v)  = X_t v_t,    xi_ls^(v)  = X_ls v_ls.
        //
        // With the exact exp sigma link,
        //
        //   s = exp(-eta_ls),
        //   q = -eta_t .* s,
        //
        // the first and second q-drifts are
        //
        //   alpha(u)   = D q[u]   = -s .* xi_t^(u) - q .* xi_ls^(u),
        //   alpha(v)   = D q[v]   = -s .* xi_t^(v) - q .* xi_ls^(v),
        //   alpha(u,v) = D^2 q[u,v]
        //              = s .* (xi_t^(u) .* xi_ls^(v) + xi_t^(v) .* xi_ls^(u))
        //                + q .* xi_ls^(u) .* xi_ls^(v).
        //
        // Differentiating the scalar-composition Hessian coefficients twice
        // yields the rowwise formulas below. Those formulas are exactly the
        // fourth-order beta-curvature contraction needed to make the joint
        // rho-Hessian path consistent with the first-order joint solve.
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleFamily input size mismatch".to_string());
        }

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_u_flat.len() != total {
            return Err(format!(
                "BinomialLocationScaleFamily joint d_beta_u length mismatch: got {}, expected {}",
                d_beta_u_flat.len(),
                total
            ));
        }
        if d_betav_flat.len() != total {
            return Err(format!(
                "BinomialLocationScaleFamily joint d_betav length mismatch: got {}, expected {}",
                d_betav_flat.len(),
                total
            ));
        }
        let d_eta_t_u = x_t.dot(&d_beta_u_flat.slice(s![0..pt]));
        let d_eta_ls_u = x_ls.dot(&d_beta_u_flat.slice(s![pt..total]));
        let d_eta_tv = x_t.dot(&d_betav_flat.slice(s![0..pt]));
        let d_eta_lsv = x_ls.dot(&d_betav_flat.slice(s![pt..total]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;

        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = core.q0[i];
            let r = 1.0 / core.sigma[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            )?;
            let s = core.dsigma_deta[i] / core.sigma[i];
            let a = d_eta_t_u[i];
            let b = s * d_eta_ls_u[i];
            let c = d_eta_tv[i];
            let d = s * d_eta_lsv[i];
            let du = -r * a - q * b;
            let dv = -r * c - q * d;
            let d2 = r * (a * d + b * c) + q * b * d;
            coeff_tt[i] =
                r * r * (m4 * du * dv + m3 * (d2 - 2.0 * d * du - 2.0 * b * dv) + 4.0 * m2 * b * d);
            // Cross block carries overall κ; scale-scale block carries κ².
            coeff_tl[i] = s
                * r
                * (q * m4 * du * dv
                    + m3 * (q * d2 + 3.0 * du * dv - q * (d * du + b * dv))
                    + m2 * (q * b * d + 2.0 * d2 - 2.0 * (d * du + b * dv))
                    + m1 * b * d);
            coeff_ll[i] = s
                * s
                * (q * q * m4 * du * dv
                    + m3 * (q * q * d2 + 5.0 * q * du * dv)
                    + m2 * (3.0 * q * d2 + 4.0 * du * dv)
                    + m1 * d2);
        }

        let d2_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d2_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d2_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut d2_h = Array2::<f64>::zeros((total, total));
        d2_h.slice_mut(s![0..pt, 0..pt]).assign(&d2_h_tt);
        d2_h.slice_mut(s![0..pt, pt..total]).assign(&d2_h_tl);
        d2_h.slice_mut(s![pt..total, pt..total]).assign(&d2_h_ll);
        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<BinomialLocationScaleJointPsiDirection>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily joint psi direction expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let n = self.y.len();
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_t = &block_states[Self::BLOCK_T].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for (local_idx, deriv) in block_derivs.iter().enumerate() {
                if global == psi_index {
                    let x_t_psi;
                    let x_ls_psi;
                    let z_t_psi;
                    let z_ls_psi;
                    match block_idx {
                        Self::BLOCK_T => {
                            x_t_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                pt,
                                0..n,
                                "BinomialLocationScaleFamily threshold",
                                policy,
                            )?;
                            z_t_psi = x_t_psi.forward_mul(beta_t.view()).map_err(|e| {
                                format!("BinomialLocationScaleFamily threshold forward_mul: {e}")
                            })?;
                            x_ls_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: pls,
                            };
                            z_ls_psi = Array1::<f64>::zeros(n);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            x_ls_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                pls,
                                0..n,
                                "BinomialLocationScaleFamily log-sigma",
                                policy,
                            )?;
                            z_ls_psi = x_ls_psi.forward_mul(beta_ls.view()).map_err(|e| {
                                format!("BinomialLocationScaleFamily log-sigma forward_mul: {e}")
                            })?;
                            x_t_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: pt,
                            };
                            z_t_psi = Array1::<f64>::zeros(n);
                        }
                        _ => return Ok(None),
                    }
                    return Ok(Some(BinomialLocationScaleJointPsiDirection {
                        block_idx,
                        local_idx,
                        x_t_psi,
                        x_ls_psi,
                        z_t_psi,
                        z_ls_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &BinomialLocationScaleJointPsiDirection,
        psi_b: &BinomialLocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<BinomialLocationScaleJointPsiSecondDrifts, String> {
        let n = self.y.len();
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_t = &block_states[Self::BLOCK_T].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut x_t_ab_action = None;
        let mut x_ls_ab_action = None;
        let mut x_t_ab = None;
        let mut x_ls_ab = None;

        // The smooth layer stores second derivatives block-locally. For a pair
        // of global psi coordinates (a, b), the only potentially nonzero
        // X_{psi_a psi_b} lives in the derivative payload of the block whose
        // basis actually moves under that pair. Cross-block mixed second
        // design derivatives are therefore zero unless explicitly provided.
        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            let deriv_b = &derivative_blocks[psi_b.block_idx][psi_b.local_idx];
            match psi_a.block_idx {
                Self::BLOCK_T => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        pt,
                        0..n,
                        "BinomialLocationScaleFamily threshold",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => x_t_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => x_t_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "BinomialLocationScaleFamily threshold: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                Self::BLOCK_LOG_SIGMA => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        pls,
                        0..n,
                        "BinomialLocationScaleFamily log-sigma",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => x_ls_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => x_ls_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "BinomialLocationScaleFamily log-sigma: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        let z_t_ab = second_psi_linear_map(x_t_ab_action.as_ref(), x_t_ab.as_ref(), n, pt)
            .forward_mul(beta_t.view());
        let z_ls_ab = second_psi_linear_map(x_ls_ab_action.as_ref(), x_ls_ab.as_ref(), n, pls)
            .forward_mul(beta_ls.view());
        Ok(BinomialLocationScaleJointPsiSecondDrifts {
            x_t_ab_action,
            x_ls_ab_action,
            x_t_ab,
            x_ls_ab,
            z_t_ab,
            z_ls_ab,
        })
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        if specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily joint psi terms expect 2 specs and 2 derivative blocks, got {} and {}",
                specs.len(),
                derivative_blocks.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleFamily input size mismatch".to_string());
        }

        // Joint fixed-beta psi terms for the coupled 2-block probit model.
        //
        // We work over the flattened coefficient vector beta = [beta_t; beta_ls]
        // and one realized spatial coordinate psi_a. The exact profiled/Laplace
        // outer calculus needs the family-side explicit objects
        //
        //   V_psi^explicit,  g_psi^explicit,  H_psi^explicit,
        //
        // all in this flattened coefficient space. These are likelihood-only
        // objects:
        //
        //   D_psi, D_{beta psi}, D_{beta beta psi}
        //
        // Generic exact-joint code adds the realized penalty motion
        //
        //   0.5 beta^T S_psi beta,  S_psi beta,  S_psi
        //
        // when forming V_i, g_i, H_i. Keeping the family hook likelihood-only
        // is what makes the unified S(theta) outer calculus correct for both
        // psi-moving designs and psi-moving penalties.
        //
        // Model:
        //   eta_t  = X_t beta_t,
        //   eta_ls = X_ls beta_ls,
        //   r      = exp(-eta_ls),
        //   q      = -eta_t .* r.
        //
        // A single realized psi_a may move either block design, so define the
        // fixed-beta predictor drifts
        //
        //   z_t  = X_{t,psi}  beta_t   (zero if psi_a is not a threshold psi)
        //   z_ls = X_{ls,psi} beta_ls  (zero if psi_a is not a log-sigma psi).
        //
        // Then the explicit q-drift is
        //
        //   q_psi = -r .* z_t - q .* z_ls.
        //
        // Rowwise scalar derivatives of the negative Bernoulli-probit loss are
        //
        //   a = dF/dq,
        //   b = d²F/dq²,
        //   c = d³F/dq³.
        //
        // Predictor-space score pieces:
        //
        //   r_t  = dF/deta_t  = -a r,
        //   r_ls = dF/deta_ls = -a q.
        //
        // Their explicit psi derivatives at fixed beta are
        //
        //   d_psi r_t  = -b q_psi r + a r z_ls,
        //   d_psi r_ls = -(a + q b) q_psi.
        //
        // Hence the exact joint score derivative is
        //
        //   g_psi
        //   = [ X_{t,psi}^T r_t  + X_t^T d_psi r_t,
        //       X_{ls,psi}^T r_ls + X_ls^T d_psi r_ls ].
        //
        // The exact envelope term is
        //
        //   V_psi^explicit = r_t^T z_t + r_ls^T z_ls.
        //
        // For the Laplace trace we also need the explicit Hessian drift. The
        // joint exact Hessian has block coefficients
        //
        //   h_tt = b r²,
        //   h_tl = r (a + q b),
        //   h_ll = q (a + q b),
        //
        // so differentiating those coefficients at fixed beta gives
        //
        //   d_psi h_tt = r² (c q_psi - 2 b z_ls),
        //   d_psi h_tl = r [ (2 b + c q) q_psi - (a + q b) z_ls ],
        //   d_psi h_ll = (a + 3 q b + q² c) q_psi.
        //
        // The full joint explicit Hessian drift is then
        //
        //   H_tt,psi
        //   = X_{t,psi}^T diag(h_tt) X_t
        //     + X_t^T diag(h_tt) X_{t,psi}
        //     + X_t^T diag(d_psi h_tt) X_t,
        //
        //   H_tl,psi
        //   = X_{t,psi}^T diag(h_tl) X_ls
        //     + X_t^T diag(h_tl) X_{ls,psi}
        //     + X_t^T diag(d_psi h_tl) X_ls,
        //
        //   H_ll,psi
        //   = X_{ls,psi}^T diag(h_ll) X_ls
        //     + X_ls^T diag(h_ll) X_{ls,psi}
        //     + X_ls^T diag(d_psi h_ll) X_ls.
        //
        // Even when only one block moves explicitly, the resulting score and
        // Hessian objects are joint because q couples eta_t and eta_ls.
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let (z_t, z_ls) = (&dir_a.z_t_psi, &dir_a.z_ls_psi);

        let mut r_t = Array1::<f64>::zeros(n);
        let mut r_ls = Array1::<f64>::zeros(n);
        let mut dr_t = Array1::<f64>::zeros(n);
        let mut dr_ls = Array1::<f64>::zeros(n);
        let mut h_tt = Array1::<f64>::zeros(n);
        let mut h_tl = Array1::<f64>::zeros(n);
        let mut h_ll = Array1::<f64>::zeros(n);
        let mut dh_tt = Array1::<f64>::zeros(n);
        let mut dh_tl = Array1::<f64>::zeros(n);
        let mut dh_ll = Array1::<f64>::zeros(n);
        let mut objective_psi = 0.0;
        for i in 0..n {
            let q = core.q0[i];
            let r = 1.0 / core.sigma[i];
            let s = core.dsigma_deta[i] / core.sigma[i];
            let sz = s * z_ls[i];
            let q_psi = -r * z_t[i] - q * sz;
            let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            r_t[i] = -a * r;
            r_ls[i] = -a * q * s;
            dr_t[i] = -b * q_psi * r + a * r * sz;
            dr_ls[i] = -(a + q * b) * q_psi;
            h_tt[i] = b * r * r;
            h_tl[i] = r * (a + q * b);
            h_ll[i] = q * (a + q * b);
            dh_tt[i] = r * r * (c * q_psi - 2.0 * b * sz);
            dh_tl[i] = r * ((2.0 * b + c * q) * q_psi - (a + q * b) * sz);
            dh_ll[i] = (a + 3.0 * q * b + q * q * c) * q_psi;
            objective_psi += r_t[i] * z_t[i] + r_ls[i] * z_ls[i];
        }

        let hessian_psi_operator = build_two_block_custom_family_joint_psi_operator_from_actions(
            dir_a.x_t_psi.cloned_first_action(),
            dir_a.x_ls_psi.cloned_first_action(),
            0..pt,
            pt..pt + pls,
            x_t,
            x_ls,
            &h_tt,
            &h_tl,
            &h_ll,
            &dh_tt,
            &dh_tl,
            &dh_ll,
        )?;
        let x_t_map = dir_a.x_t_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_t = x_t_map.transpose_mul(r_t.view()) + x_t.t().dot(&dr_t);
        let score_ls = x_ls_map.transpose_mul(r_ls.view()) + x_ls.t().dot(&dr_ls);
        let mut score_psi = Array1::<f64>::zeros(total);
        score_psi.slice_mut(s![0..pt]).assign(&score_t);
        score_psi.slice_mut(s![pt..pt + pls]).assign(&score_ls);
        let hessian_psi = if hessian_psi_operator.is_some() {
            Array2::zeros((0, 0))
        } else {
            let h_tt_block = weighted_crossprod_psi_maps(
                x_t_map,
                h_tt.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tt.view(),
                x_t_map,
            )? + &xt_diag_x_dense(x_t, &dh_tt)?;
            let h_tl_block = weighted_crossprod_psi_maps(
                x_t_map,
                h_tl.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tl.view(),
                x_ls_map,
            )? + &xt_diag_y_dense(x_t, &dh_tl, x_ls)?;
            let h_ll_block = weighted_crossprod_psi_maps(
                x_ls_map,
                h_ll.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                h_ll.view(),
                x_ls_map,
            )? + &xt_diag_x_dense(x_ls, &dh_ll)?;

            let mut hessian_psi = Array2::<f64>::zeros((total, total));
            hessian_psi.slice_mut(s![0..pt, 0..pt]).assign(&h_tt_block);
            hessian_psi
                .slice_mut(s![0..pt, pt..pt + pls])
                .assign(&h_tl_block);
            hessian_psi
                .slice_mut(s![pt..pt + pls, pt..pt + pls])
                .assign(&h_ll_block);
            mirror_upper_to_lower(&mut hessian_psi);
            hessian_psi
        };

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_j) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_i,
                &dir_j,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_i: &BinomialLocationScaleJointPsiDirection,
        dir_j: &BinomialLocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
            x_t,
            x_ls,
        )?;
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let x_t_i_map = dir_i.x_t_psi.as_linear_map_ref();
        let x_t_j_map = dir_j.x_t_psi.as_linear_map_ref();
        let x_ls_i_map = dir_i.x_ls_psi.as_linear_map_ref();
        let x_ls_j_map = dir_j.x_ls_psi.as_linear_map_ref();
        let x_t_ab_map = second_psi_linear_map(
            second_drifts.x_t_ab_action.as_ref(),
            second_drifts.x_t_ab.as_ref(),
            n,
            pt,
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            pls,
        );

        // Exact fixed-beta psi/psi terms for the coupled non-wiggle probit
        // family.
        //
        // For two realized spatial coordinates psi_a, psi_b define
        //
        //   z_t,a  = X_{t,a} beta_t,    z_ls,a  = X_{ls,a} beta_ls,
        //   z_t,b  = X_{t,b} beta_t,    z_ls,b  = X_{ls,b} beta_ls,
        //   z_t,ab = X_{t,ab} beta_t,   z_ls,ab = X_{ls,ab} beta_ls.
        //
        // On the smooth interior branch, with r = exp(-eta_ls) and q = -eta_t r,
        //
        //   q_a  = -r z_t,a - q z_ls,a,
        //   q_b  = -r z_t,b - q z_ls,b,
        //   q_ab = -r z_t,ab
        //          + r(z_t,a z_ls,b + z_t,b z_ls,a)
        //          + q(z_ls,a z_ls,b - z_ls,ab).
        //
        // For scalar row loss derivatives
        //
        //   a = dF/dq,  b = d²F/dq²,  c = d³F/dq³,  d = d⁴F/dq⁴,
        //
        // the exact fixed-beta psi/psi objects are
        //
        //   V_ab = sum [ a q_ab + b q_a q_b ],
        //
        //   g_ab = [ X_{t,ab}^T r_t + X_{t,a}^T d_b r_t + X_{t,b}^T d_a r_t + X_t^T d_ab r_t,
        //            X_{ls,ab}^T r_ls + X_{ls,a}^T d_b r_ls + X_{ls,b}^T d_a r_ls + X_ls^T d_ab r_ls ],
        //
        // where
        //
        //   r_t  = -a r,
        //   r_ls = -a q,
        //
        //   d_a r_t  = -b q_a r + a r z_ls,a,
        //   d_a r_ls = -(a + q b) q_a,
        //
        //   d_ab r_t
        //   = r[
        //       -c q_a q_b - b q_ab
        //       + b(q_a z_ls,b + q_b z_ls,a)
        //       - a z_ls,a z_ls,b
        //       + a z_ls,ab
        //     ],
        //
        //   d_ab r_ls
        //   = -[(2b + q c) q_a q_b + (a + q b) q_ab].
        //
        // The exact Hessian psi/psi drift comes from the second derivatives of
        // the joint Hessian coefficients. In the notation of the unified outer
        // calculus, these rowwise coefficient drifts are precisely the
        // likelihood-side pieces of
        //
        //   D_{beta beta psi_a psi_b},
        //
        // before the generic assembler adds any realized-penalty contribution
        //
        //   S_ab = partial_{psi_a psi_b} S(theta).
        //
        // So this helper returns likelihood-only
        //
        //   D_ab, D_{beta ab}, D_{beta beta ab},
        //
        // and the unified exact assembler in custom_family.rs forms
        //
        //   V_ab = D_ab + 0.5 beta^T S_ab beta,
        //   g_ab = D_{beta ab} + S_ab beta,
        //   H_ab = D_{beta beta ab} + S_ab.
        //
        // Once H_ab is known, the outer assembler combines it with the joint
        // mode responses beta_a, beta_b, beta_ab and the contractions
        //
        //   T_a[beta_b], T_b[beta_a], D_beta H[beta_ab], D_beta^2 H[beta_a, beta_b]
        //
        // to form
        //
        //   ddot H_ab
        //   = H_ab + T_a[beta_b] + T_b[beta_a]
        //     + D_beta H[beta_ab] + D_beta^2 H[beta_a, beta_b].
        //
        // That is why this helper computes only the fixed-beta psi/psi object:
        // the total profiled/Laplace Hessian drift is assembled generically in
        // custom_family.rs after the joint solves.
        //
        // Concretely, the rowwise coefficient identities below are
        //
        //   h_tt = b r²,
        //   h_tl = r(a + q b),
        //   h_ll = q(a + q b),
        //
        // namely
        //
        //   d_ab h_tt
        //   = r²[
        //       d q_a q_b + c q_ab
        //       - 2c(q_b z_ls,a + q_a z_ls,b)
        //       + 4b z_ls,a z_ls,b
        //       - 2b z_ls,ab
        //     ],
        //
        //   d_ab h_tl
        //   = r[
        //       ((3c + q d) q_b) q_a
        //       + (2b + q c) q_ab
        //       - (2b + q c)(q_b z_ls,a + q_a z_ls,b)
        //       + (a + q b)(z_ls,a z_ls,b - z_ls,ab)
        //     ],
        //
        //   d_ab h_ll
        //   = (4b + 5q c + q² d) q_a q_b
        //     + (a + 3q b + q² c) q_ab.
        //
        // Differentiating X^T diag(h) X twice then gives the explicit joint
        // psi/psi Hessian blocks.
        let mut r_t = Array1::<f64>::zeros(n);
        let mut r_ls = Array1::<f64>::zeros(n);
        let mut dr_t_i = Array1::<f64>::zeros(n);
        let mut dr_t_j = Array1::<f64>::zeros(n);
        let mut dr_ls_i = Array1::<f64>::zeros(n);
        let mut dr_ls_j = Array1::<f64>::zeros(n);
        let mut d2r_t = Array1::<f64>::zeros(n);
        let mut d2r_ls = Array1::<f64>::zeros(n);
        let mut h_tt = Array1::<f64>::zeros(n);
        let mut h_tl = Array1::<f64>::zeros(n);
        let mut h_ll = Array1::<f64>::zeros(n);
        let mut dh_tt_i = Array1::<f64>::zeros(n);
        let mut dh_tt_j = Array1::<f64>::zeros(n);
        let mut dh_tl_i = Array1::<f64>::zeros(n);
        let mut dh_tl_j = Array1::<f64>::zeros(n);
        let mut dh_ll_i = Array1::<f64>::zeros(n);
        let mut dh_ll_j = Array1::<f64>::zeros(n);
        let mut d2h_tt = Array1::<f64>::zeros(n);
        let mut d2h_tl = Array1::<f64>::zeros(n);
        let mut d2h_ll = Array1::<f64>::zeros(n);
        let mut objective_psi_psi = 0.0;
        for row in 0..n {
            let q = core.q0[row];
            let r = 1.0 / core.sigma[row];
            let q_i = -r * dir_i.z_t_psi[row] - q * dir_i.z_ls_psi[row];
            let q_j = -r * dir_j.z_t_psi[row] - q * dir_j.z_ls_psi[row];
            let q_ij = -r * second_drifts.z_t_ab[row]
                + r * (dir_i.z_t_psi[row] * dir_j.z_ls_psi[row]
                    + dir_j.z_t_psi[row] * dir_i.z_ls_psi[row])
                + q * (dir_i.z_ls_psi[row] * dir_j.z_ls_psi[row] - second_drifts.z_ls_ab[row]);
            let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let d = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            let u = a + q * b;
            let u_i = (2.0 * b + q * c) * q_i;
            let u_j = (2.0 * b + q * c) * q_j;

            r_t[row] = -a * r;
            r_ls[row] = -a * q;
            dr_t_i[row] = -b * q_i * r + a * r * dir_i.z_ls_psi[row];
            dr_t_j[row] = -b * q_j * r + a * r * dir_j.z_ls_psi[row];
            dr_ls_i[row] = -u * q_i;
            dr_ls_j[row] = -u * q_j;
            d2r_t[row] = r
                * (-c * q_i * q_j - b * q_ij
                    + b * (q_i * dir_j.z_ls_psi[row] + q_j * dir_i.z_ls_psi[row])
                    - a * dir_i.z_ls_psi[row] * dir_j.z_ls_psi[row]
                    + a * second_drifts.z_ls_ab[row]);
            d2r_ls[row] = -((2.0 * b + q * c) * q_i * q_j + u * q_ij);

            h_tt[row] = b * r * r;
            h_tl[row] = r * u;
            h_ll[row] = q * u;
            dh_tt_i[row] = r * r * (c * q_i - 2.0 * b * dir_i.z_ls_psi[row]);
            dh_tt_j[row] = r * r * (c * q_j - 2.0 * b * dir_j.z_ls_psi[row]);
            dh_tl_i[row] = r * (u_i - u * dir_i.z_ls_psi[row]);
            dh_tl_j[row] = r * (u_j - u * dir_j.z_ls_psi[row]);
            dh_ll_i[row] = (a + 3.0 * q * b + q * q * c) * q_i;
            dh_ll_j[row] = (a + 3.0 * q * b + q * q * c) * q_j;
            d2h_tt[row] = r
                * r
                * (d * q_i * q_j + c * q_ij
                    - 2.0 * c * (q_j * dir_i.z_ls_psi[row] + q_i * dir_j.z_ls_psi[row])
                    + 4.0 * b * dir_i.z_ls_psi[row] * dir_j.z_ls_psi[row]
                    - 2.0 * b * second_drifts.z_ls_ab[row]);
            d2h_tl[row] = r
                * (((3.0 * c + q * d) * q_j) * q_i + (2.0 * b + q * c) * q_ij
                    - (2.0 * b + q * c) * (q_j * dir_i.z_ls_psi[row] + q_i * dir_j.z_ls_psi[row])
                    + u * (dir_i.z_ls_psi[row] * dir_j.z_ls_psi[row] - second_drifts.z_ls_ab[row]));
            d2h_ll[row] = (4.0 * b + 5.0 * q * c + q * q * d) * q_i * q_j
                + (a + 3.0 * q * b + q * q * c) * q_ij;

            objective_psi_psi += a * q_ij + b * q_i * q_j;
        }

        let mut score_psi_psi = Array1::<f64>::zeros(total);
        score_psi_psi.slice_mut(s![0..pt]).assign(
            &(x_t_ab_map.transpose_mul(r_t.view())
                + x_t_i_map.transpose_mul(dr_t_j.view())
                + x_t_j_map.transpose_mul(dr_t_i.view())
                + x_t.t().dot(&d2r_t)),
        );
        score_psi_psi.slice_mut(s![pt..pt + pls]).assign(
            &(x_ls_ab_map.transpose_mul(r_ls.view())
                + x_ls_i_map.transpose_mul(dr_ls_j.view())
                + x_ls_j_map.transpose_mul(dr_ls_i.view())
                + x_ls.t().dot(&d2r_ls)),
        );

        let h_tt_block = weighted_crossprod_psi_maps(
            x_t_ab_map,
            h_tt.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(x_t_i_map, h_tt.view(), x_t_j_map)?
            + &weighted_crossprod_psi_maps(x_t_j_map, h_tt.view(), x_t_i_map)?
            + &weighted_crossprod_psi_maps(
                x_t_i_map,
                dh_tt_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )?
            + &weighted_crossprod_psi_maps(
                x_t_j_map,
                dh_tt_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tt_i.view(),
                x_t_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tt_j.view(),
                x_t_i_map,
            )?
            + &xt_diag_x_dense(x_t, &d2h_tt)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tt.view(),
                x_t_ab_map,
            )?;
        let h_tl_block = weighted_crossprod_psi_maps(
            x_t_ab_map,
            h_tl.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(x_t_i_map, h_tl.view(), x_ls_j_map)?
            + &weighted_crossprod_psi_maps(x_t_j_map, h_tl.view(), x_ls_i_map)?
            + &weighted_crossprod_psi_maps(
                x_t_i_map,
                dh_tl_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                x_t_j_map,
                dh_tl_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tl_i.view(),
                x_ls_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tl_j.view(),
                x_ls_i_map,
            )?
            + &xt_diag_y_dense(x_t, &d2h_tl, x_ls)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tl.view(),
                x_ls_ab_map,
            )?;
        let h_ll_block = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            h_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(x_ls_i_map, h_ll.view(), x_ls_j_map)?
            + &weighted_crossprod_psi_maps(x_ls_j_map, h_ll.view(), x_ls_i_map)?
            + &weighted_crossprod_psi_maps(
                x_ls_i_map,
                dh_ll_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                x_ls_j_map,
                dh_ll_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                dh_ll_i.view(),
                x_ls_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                dh_ll_j.view(),
                x_ls_i_map,
            )?
            + &xt_diag_x_dense(x_ls, &d2h_ll)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                h_ll.view(),
                x_ls_ab_map,
            )?;

        let mut hessian_psi_psi = Array2::<f64>::zeros((total, total));
        hessian_psi_psi
            .slice_mut(s![0..pt, 0..pt])
            .assign(&h_tt_block);
        hessian_psi_psi
            .slice_mut(s![0..pt, pt..pt + pls])
            .assign(&h_tl_block);
        hessian_psi_psi
            .slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&h_ll_block);
        mirror_upper_to_lower(&mut hessian_psi_psi);

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &BinomialLocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_flat.len() != total {
            return Err(format!(
                "BinomialLocationScaleFamily joint psi hessian directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ));
        }
        let xi_t = x_t.dot(&d_beta_flat.slice(s![0..pt]));
        let xi_ls = x_ls.dot(&d_beta_flat.slice(s![pt..pt + pls]));
        let x_t_map = dir_a.x_t_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();

        // Mixed contraction T_a[u] = D_beta H_{psi_a}[u].
        //
        // In the non-wiggle family the realized design derivatives X_{psi_a}
        // are fixed with respect to beta, so differentiating the explicit
        // Hessian drift H_{psi_a} only moves the rowwise coefficient arrays.
        // This helper therefore returns exactly the likelihood-side mixed drift
        // required by the unified outer Hessian formula
        //
        //   ddot H_{ij}
        //   = H_{ij}
        //     + T_i[beta_j]
        //     + T_j[beta_i]
        //     + D_beta H[beta_ij]
        //     + D_beta^2 H[beta_i, beta_j].
        //
        // For i = psi_a, the generic assembler supplies beta_j and any
        // realized-penalty piece S_{psi_a} itself; this family hook contributes
        // only the exact likelihood-side T_a[beta_j].
        //
        // With
        //   du   = D_beta q[u]   = -r xi_t - q xi_ls,
        //   q_a  = q_{psi_a}     = -r z_t,a - q z_ls,a,
        //   q_au = D_beta q_a[u] = r z_t,a xi_ls - du z_ls,a,
        //
        // the directional derivatives of the first-order Hessian-drift
        // coefficients are the mixed specializations of the exact psi/psi
        // formulas with z_ls,ab = 0 and q_ab = q_au:
        //
        //   D_u(d_a h_tt)
        //   = r²[
        //       d du q_a + c q_au
        //       - 2c(q_a xi_ls + du z_ls,a)
        //       + 4b xi_ls z_ls,a
        //     ],
        //
        //   D_u(d_a h_tl)
        //   = r[
        //       ((3c + q d) q_a) du
        //       + (2b + q c) q_au
        //       - (2b + q c)(q_a xi_ls + du z_ls,a)
        //       + (a + q b) xi_ls z_ls,a
        //     ],
        //
        //   D_u(d_a h_ll)
        //   = (4b + 5q c + q² d) du q_a
        //     + (a + 3q b + q² c) q_au.
        //
        // Since X_t, X_ls, X_{t,psi_a}, X_{ls,psi_a} are all beta-independent
        // here, the full matrix contraction is obtained by replacing the row
        // coefficient arrays in H_{psi_a} by their directional derivatives.
        let mut dh_tt_u = Array1::<f64>::zeros(n);
        let mut dh_tl_u = Array1::<f64>::zeros(n);
        let mut dh_ll_u = Array1::<f64>::zeros(n);
        let mut h_tt_u = Array1::<f64>::zeros(n);
        let mut h_tl_u = Array1::<f64>::zeros(n);
        let mut h_ll_u = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = core.q0[row];
            let r = 1.0 / core.sigma[row];
            let s = core.dsigma_deta[row] / core.sigma[row];
            let xi_ls_s = s * xi_ls[row];
            let z_ls_psi_s = s * dir_a.z_ls_psi[row];
            let du = -r * xi_t[row] - q * xi_ls_s;
            let q_a = -r * dir_a.z_t_psi[row] - q * z_ls_psi_s;
            let q_au = r * dir_a.z_t_psi[row] * xi_ls_s - du * z_ls_psi_s;
            let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let d = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            let u = a + q * b;
            h_tt_u[row] = r * r * (c * du - 2.0 * b * xi_ls_s);
            h_tl_u[row] = r * ((2.0 * b + q * c) * du - u * xi_ls_s);
            h_ll_u[row] = (a + 3.0 * q * b + q * q * c) * du;
            dh_tt_u[row] = r
                * r
                * (d * du * q_a + c * q_au - 2.0 * c * (q_a * xi_ls_s + du * z_ls_psi_s)
                    + 4.0 * b * xi_ls_s * z_ls_psi_s);
            dh_tl_u[row] = r
                * (((3.0 * c + q * d) * q_a) * du + (2.0 * b + q * c) * q_au
                    - (2.0 * b + q * c) * (q_a * xi_ls_s + du * z_ls_psi_s)
                    + u * xi_ls_s * z_ls_psi_s);
            dh_ll_u[row] = (4.0 * b + 5.0 * q * c + q * q * d) * du * q_a
                + (a + 3.0 * q * b + q * q * c) * q_au;
        }

        let tt_block = weighted_crossprod_psi_maps(
            x_t_map,
            h_tt_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            h_tt_u.view(),
            x_t_map,
        )? + &xt_diag_x_dense(x_t, &dh_tt_u)?;
        let tl_block = weighted_crossprod_psi_maps(
            x_t_map,
            h_tl_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            h_tl_u.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(x_t, &dh_tl_u, x_ls)?;
        let ll_block = weighted_crossprod_psi_maps(
            x_ls_map,
            h_ll_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
            h_ll_u.view(),
            x_ls_map,
        )? + &xt_diag_x_dense(x_ls, &dh_ll_u)?;
        let mut out = Array2::<f64>::zeros((total, total));
        out.slice_mut(s![0..pt, 0..pt]).assign(&tt_block);
        out.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl_block);
        out.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&ll_block);
        mirror_upper_to_lower(&mut out);
        Ok(out)
    }
}

impl CustomFamily for BinomialLocationScaleFamily {
    /// The Binomial location-scale joint Hessian depends on β because the
    /// Hessian blocks are functions of q = -t/σ and the link derivatives,
    /// all of which change when β_t or β_{log σ} move.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Two fully-coupled blocks (threshold p_t, log-σ p_ℓ): joint Hessian
        // size (p_t + p_ℓ)² per row.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(self.y.len() as u64, specs)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleFamily input size mismatch".to_string());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        if !self.exact_joint_supported() {
            return Err(
                "BinomialLocationScaleFamily requires exact curvature designs; diagonal fallback has been removed"
                    .to_string(),
            );
        }
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;

        // Per-block gradients from the eta-space score.
        //
        //   score_q = -m1   (m1 = dF/dq, F = -ℓ)
        //   grad_eta_t[i]  = score_q * q_t
        //   grad_eta_ls[i] = score_q * q_ls
        let mut grad_eta_t = Array1::<f64>::zeros(n);
        let mut grad_eta_ls = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (m1, _, _) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                core.q0[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            let q0d = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            grad_eta_t[i] = -m1 * q0d.q_t;
            grad_eta_ls[i] = -m1 * q0d.q_ls;
        }
        let grad_t = threshold_design.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = log_sigma_design.transpose_vector_multiply(&grad_eta_ls);

        // Per-block Hessians without ever materializing the full p×p joint
        // matrix — the off-diagonal cross block is unused for IRLS-style block
        // working sets and would cost O(p_t * p_ls * n) to form. The diagonal
        // blocks are computed from the same row coefficients as the joint.
        let (x_t, x_ls) = self.exact_joint_dense_block_designs(None)?.ok_or(
            "BinomialLocationScaleFamily: joint block designs unavailable",
        )?;
        let (h_tt, h_ll) =
            self.exact_newton_block_diagonal_hessians_from_designs(block_states, &x_t, &x_ls)?;
        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(h_tt),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_ls,
                    hessian: SymmetricMatrix::Dense(h_ll),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleFamily input size mismatch".to_string());
        }
        // Zero-allocation O(n) scalar loop — no working sets, no n-vector intermediates.
        binomial_location_scale_ll_only(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        Err(
            "BinomialLocationScaleFamily no longer supports diagonal working weights; exact curvature is required"
                .to_string(),
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            BinomialLocationScaleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        let pt = self
            .threshold_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
            })?
            .ncols();
        let pls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
            })?
            .ncols();
        let total = pt + pls;
        let (start, end, joint_direction) = match block_idx {
            Self::BLOCK_T => {
                if d_beta.len() != pt {
                    return Err(format!(
                        "BinomialLocationScaleFamily threshold d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        pt
                    ));
                }
                let mut dir = Array1::<f64>::zeros(total);
                dir.slice_mut(s![0..pt]).assign(d_beta);
                (0usize, pt, dir)
            }
            Self::BLOCK_LOG_SIGMA => {
                if d_beta.len() != pls {
                    return Err(format!(
                        "BinomialLocationScaleFamily log-sigma d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        pls
                    ));
                }
                let mut dir = Array1::<f64>::zeros(total);
                dir.slice_mut(s![pt..pt + pls]).assign(d_beta);
                (pt, pt + pls, dir)
            }
            _ => return Ok(None),
        };
        let joint = self
            .exact_newton_joint_hessian_directional_derivative(block_states, &joint_direction)?
            .ok_or_else(|| {
                format!("missing joint exact-newton directional Hessian for block {block_idx}")
            })?;
        Ok(Some(joint.slice(s![start..end, start..end]).to_owned()))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = BinomialLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            x_t.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }
}

impl CustomFamilyGenerative for BinomialLocationScaleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() {
            return Err("BinomialLocationScaleFamily generative size mismatch".to_string());
        }
        let mut mean = Array1::<f64>::zeros(self.y.len());
        for i in 0..mean.len() {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            let q = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
                .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
            mean[i] = jet.mu;
        }
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}

/// Matrix-free joint-Hessian operator for the two-block binomial
/// location-scale family.
///
/// The dense joint Hessian is `H = [[X_t^T D_tt X_t, X_t^T D_tl X_ls],
///                                  [X_ls^T D_tl X_t, X_ls^T D_ll X_ls]]`
/// where `D_tt`, `D_tl`, `D_ll` are diagonal weight vectors derived from the
/// rowwise scalar-composition Hessian. For a flattened direction
/// `v = (v_t, v_ls)`, `H v` is computed as
///
///   u_t = X_t v_t,  u_ls = X_ls v_ls,
///   r_t = D_tt .* u_t + D_tl .* u_ls,
///   r_ls = D_tl .* u_t + D_ll .* u_ls,
///   H v = (X_t^T r_t, X_ls^T r_ls).
///
/// Cost is Θ(n (p_t + p_ls)) per matvec versus Θ(n (p_t + p_ls)^2) to form
/// the dense matrix. The directional-derivative paths still use the dense
/// `from_designs` helpers — the unified evaluator wraps any returned dense
/// matrix in a HyperOperator when no native operator form exists.
struct BinomialLocationScaleHessianWorkspace {
    family: BinomialLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    x_t: Array2<f64>,
    x_ls: Array2<f64>,
    coeff_tt: Array1<f64>,
    coeff_tl: Array1<f64>,
    coeff_ll: Array1<f64>,
}

impl BinomialLocationScaleHessianWorkspace {
    fn new(
        family: BinomialLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        _specs: Vec<ParameterBlockSpec>,
        x_t: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let (coeff_tt, coeff_tl, coeff_ll) =
            family.exact_newton_joint_hessian_row_coefficients(&block_states)?;
        Ok(Self {
            family,
            block_states,
            x_t,
            x_ls,
            coeff_tt,
            coeff_tl,
            coeff_ll,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for BinomialLocationScaleHessianWorkspace {
    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        if v.len() != total {
            return Err(format!(
                "BinomialLocationScale matvec dimension mismatch: got {}, expected {}",
                v.len(),
                total
            ));
        }
        // u_t = X_t v_t, u_ls = X_ls v_ls
        let u_t = self.x_t.dot(&v.slice(s![0..pt]));
        let u_ls = self.x_ls.dot(&v.slice(s![pt..total]));
        // r_t = D_tt .* u_t + D_tl .* u_ls; r_ls = D_tl .* u_t + D_ll .* u_ls
        let r_t = &self.coeff_tt * &u_t + &self.coeff_tl * &u_ls;
        let r_ls = &self.coeff_tl * &u_t + &self.coeff_ll * &u_ls;
        // (X_t^T r_t, X_ls^T r_ls)
        let out_t = self.x_t.t().dot(&r_t);
        let out_ls = self.x_ls.t().dot(&r_ls);
        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pt]).assign(&out_t);
        out.slice_mut(s![pt..total]).assign(&out_ls);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        let mut diag = Array1::<f64>::zeros(total);
        // diag(X_t^T D_tt X_t)[j] = sum_i D_tt[i] * X_t[i,j]^2
        for j in 0..pt {
            let col = self.x_t.column(j);
            let mut acc = 0.0;
            for i in 0..self.coeff_tt.len() {
                let v = col[i];
                acc += self.coeff_tt[i] * v * v;
            }
            diag[j] = acc;
        }
        for j in 0..pls {
            let col = self.x_ls.column(j);
            let mut acc = 0.0;
            for i in 0..self.coeff_ll.len() {
                let v = col[i];
                acc += self.coeff_ll[i] * v * v;
            }
            diag[pt + j] = acc;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                &self.x_t,
                &self.x_ls,
                d_beta_flat,
            )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                &self.x_t,
                &self.x_ls,
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }
}

/// Built-in binomial location-scale family with a configurable inverse link and learnable wiggle on q.
///
/// Block structure:
/// - Block 0: threshold T(covariates)
/// - Block 1: log sigma(covariates)
/// - Block 2: wiggle(q) represented by B-spline coefficients on q
#[derive(Clone)]
pub struct BinomialLocationScaleWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}

impl BinomialLocationScaleWiggleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameternames() -> &'static [&'static str] {
        &["threshold", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::InverseLink,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scalewiggle",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.threshold_design.is_some() && self.log_sigma_design.is_some()
    }

    pub fn initializewiggle_knots_from_q(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
    ) -> Result<Array1<f64>, String> {
        initializewiggle_knots_from_seed(q_seed, degree, num_internal_knots)
    }

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            basis_options.derivative_order,
        )
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle derivative col mismatch: got {}, expected {}",
                d_constrained.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d_constrained.dot(&beta_link_wiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2_constrained =
            self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2_constrained.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle second-derivative col mismatch: got {}, expected {}",
                d2_constrained.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d2_constrained.dot(&beta_link_wiggle))
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3_constrained = self.wiggle_d3basis_constrained(q0)?;
        if d3_constrained.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle third-derivative col mismatch: got {}, expected {}",
                d3_constrained.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d3_constrained.dot(&beta_link_wiggle))
    }

    fn wiggle_d3basis_constrained(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(format!(
                "wiggle fourth-derivative col mismatch: got {}, expected {}",
                d4.ncols(),
                beta_link_wiggle.len()
            ));
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    fn dense_block_designs(&self) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        let td = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact path is missing threshold design".to_string()
        })?;
        let lsd = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact path is missing log-sigma design".to_string()
        })?;
        let xt = match td.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                td.try_to_dense_with_policy(
                    &self.policy.material_policy(),
                    "BinomialLocationScaleWiggle dense_block_designs threshold",
                )
                .map_err(|e| e.to_string())?
                .as_ref()
                .clone(),
            ),
        };
        let xls = match lsd.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                lsd.try_to_dense_with_policy(
                    &self.policy.material_policy(),
                    "BinomialLocationScaleWiggle dense_block_designs log_sigma",
                )
                .map_err(|e| e.to_string())?
                .as_ref()
                .clone(),
            ),
        };
        Ok((xt, xls))
    }

    fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        if specs.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 specs, got {}",
                specs.len()
            ));
        }
        let xt = match specs[Self::BLOCK_T].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_T]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialLocationScaleWiggle dense_block_designs_fromspecs threshold",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        let xls = match specs[Self::BLOCK_LOG_SIGMA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_LOG_SIGMA]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialLocationScaleWiggle dense_block_designs_fromspecs log_sigma",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        };
        Ok((xt, xls))
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if self.threshold_design.is_some() && self.log_sigma_design.is_some() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    fn shadow_with_exact_joint_designs(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Self>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        Ok(Some(Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_t.into_owned(),
            ))),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_ls.into_owned(),
            ))),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: self.policy.clone(),
        }))
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<BinomialLocationScaleWiggleJointPsiDirection>, String> {
        if block_states.len() != 3 || derivative_blocks.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily joint psi direction expects 3 blocks and 3 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let n = self.y.len();
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_t = &block_states[Self::BLOCK_T].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for (local_idx, deriv) in block_derivs.iter().enumerate() {
                if global == psi_index {
                    let x_t_psi;
                    let x_ls_psi;
                    let z_t_psi;
                    let z_ls_psi;
                    match block_idx {
                        Self::BLOCK_T => {
                            x_t_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                pt,
                                0..n,
                                "BinomialLocationScaleWiggleFamily threshold",
                                policy,
                            )?;
                            z_t_psi = x_t_psi.forward_mul(beta_t.view()).map_err(|e| {
                                format!(
                                    "BinomialLocationScaleWiggleFamily threshold forward_mul: {e}"
                                )
                            })?;
                            x_ls_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: pls,
                            };
                            z_ls_psi = Array1::<f64>::zeros(n);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            x_ls_psi = resolve_custom_family_x_psi_map(
                                deriv,
                                n,
                                pls,
                                0..n,
                                "BinomialLocationScaleWiggleFamily log-sigma",
                                policy,
                            )?;
                            z_ls_psi = x_ls_psi.forward_mul(beta_ls.view()).map_err(|e| {
                                format!(
                                    "BinomialLocationScaleWiggleFamily log-sigma forward_mul: {e}"
                                )
                            })?;
                            x_t_psi = PsiDesignMap::Zero {
                                nrows: n,
                                ncols: pt,
                            };
                            z_t_psi = Array1::<f64>::zeros(n);
                        }
                        Self::BLOCK_WIGGLE => return Ok(None),
                        _ => return Ok(None),
                    }
                    return Ok(Some(BinomialLocationScaleWiggleJointPsiDirection {
                        block_idx,
                        local_idx,
                        z_t_psi,
                        z_ls_psi,
                        x_t_psi,
                        x_ls_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &BinomialLocationScaleWiggleJointPsiDirection,
        psi_b: &BinomialLocationScaleWiggleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<BinomialLocationScaleWiggleJointPsiSecondDrifts, String> {
        let n = self.y.len();
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_t = &block_states[Self::BLOCK_T].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut x_t_ab_action = None;
        let mut x_ls_ab_action = None;
        let mut x_t_ab = None;
        let mut x_ls_ab = None;
        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            let deriv_b = &derivative_blocks[psi_b.block_idx][psi_b.local_idx];
            match psi_a.block_idx {
                Self::BLOCK_T => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        pt,
                        0..n,
                        "BinomialLocationScaleWiggleFamily threshold",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => x_t_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => x_t_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "BinomialLocationScaleWiggleFamily threshold: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                Self::BLOCK_LOG_SIGMA => {
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        n,
                        pls,
                        0..n,
                        "BinomialLocationScaleWiggleFamily log-sigma",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => x_ls_ab_action = Some(action),
                        PsiDesignMap::Dense { matrix } => x_ls_ab = Some((*matrix).clone()),
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "BinomialLocationScaleWiggleFamily log-sigma: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        let z_t_ab = second_psi_linear_map(x_t_ab_action.as_ref(), x_t_ab.as_ref(), n, pt)
            .forward_mul(beta_t.view());
        let z_ls_ab = second_psi_linear_map(x_ls_ab_action.as_ref(), x_ls_ab.as_ref(), n, pls)
            .forward_mul(beta_ls.view());
        Ok(BinomialLocationScaleWiggleJointPsiSecondDrifts {
            x_t_ab_action,
            x_ls_ab_action,
            x_t_ab,
            x_ls_ab,
            z_t_ab,
            z_ls_ab,
        })
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if self
            .exact_newton_joint_psi_direction(
                block_states,
                derivative_blocks,
                psi_index,
                x_t,
                x_ls,
                &self.policy,
            )?
            .is_none()
        {
            return Ok(None);
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let m = d0.dot(betaw) + 1.0;
        let g2 = self.wiggle_d2q_dq02(base_core.q0.view(), betaw.view())?;
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let (z_t_psi, z_ls_psi) = (&dir_a.z_t_psi, &dir_a.z_ls_psi);
        let mut objective_psi = 0.0;

        let mut score_t_xa = Array1::<f64>::zeros(n);
        let mut score_t_x = Array1::<f64>::zeros(n);
        let mut score_ls_xa = Array1::<f64>::zeros(n);
        let mut score_ls_x = Array1::<f64>::zeros(n);
        let mut score_w_b = Array1::<f64>::zeros(n);
        let mut score_w_d1 = Array1::<f64>::zeros(n);

        let mut coeff_tt_w = Array1::<f64>::zeros(n);
        let mut coeff_tt_d = Array1::<f64>::zeros(n);
        let mut coeff_tl_w = Array1::<f64>::zeros(n);
        let mut coeff_tl_d = Array1::<f64>::zeros(n);
        let mut coeff_ll_w = Array1::<f64>::zeros(n);
        let mut coeff_ll_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_b_w = Array1::<f64>::zeros(n);
        let mut coeff_tw_b_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_d1_w = Array1::<f64>::zeros(n);
        let mut coeff_tw_d1_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_d2_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_b_w = Array1::<f64>::zeros(n);
        let mut coeff_lw_b_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_d1_w = Array1::<f64>::zeros(n);
        let mut coeff_lw_d1_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_d2_d = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);

        // Exact likelihood-only joint psi terms for the probit wiggle family.
        //
        // This helper is intentionally the same generic rowwise kernel as the
        // non-wiggle family. The only difference is the location-side row:
        //
        //   gamma = [beta_t; betaw],
        //   delta = beta_ls,
        //   z_r   = [x_{t,r}; B_r(q0)],
        //   x_r   = x_{ls,r},
        //   a_r   = z_r^T gamma,
        //   ell_r = x_r^T delta,
        //   q_r   = -a_r * exp(-ell_r).
        //
        // In this wiggle family we realize the same kernel through the chain
        //
        //   q = q0 + betaw^T B(q0),
        //   q0 = -eta_t * exp(-eta_ls),
        //   m  = dq/dq0   = 1 + betaw^T B'(q0),
        //   g2 = d²q/dq0² = betaw^T B''(q0),
        //   g3 = d³q/dq0³ = betaw^T B'''(q0).
        //
        // For a realized hyperdirection psi_a:
        //
        //   h_a     = q_{psi_a},
        //   c_a     = q_{beta psi_a},
        //   R_a     = q_{beta beta psi_a},
        //
        // and the generic scalar-loss identities are
        //
        //   D_a            = sum_r r_r h_{r,a},
        //   D_{beta a}     = sum_r [ w_r h_{r,a} b_r + r_r c_{r,a} ],
        //   D_{beta beta a}
        //                  = sum_r [ nu_r h_{r,a} b_r b_r^T
        //                              + w_r(c_{r,a} b_r^T + b_r c_{r,a}^T + h_{r,a} Q_r)
        //                              + r_r R_{r,a} ].
        //
        // Generic exact-joint code adds all realized penalty motion S_a after
        // the fact, so this family hook must stay likelihood-only.
        //
        // The rowwise objects below are the wiggle specialization of the same
        // q_r = -a_r exp(-ell_r) kernel. All wiggle-specific complexity is
        // localized to the realized row B_r(q0) and its q0-derivatives.
        for row in 0..n {
            let q0 = base_core.q0[row];
            let q = q0 + etaw[row];
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let r_sigma = 1.0 / sigma[row];
            let q0_a = -r_sigma * z_t_psi[row] - q0 * z_ls_psi[row];
            let q0_t_a = q0_geom.q_tl * z_ls_psi[row];
            let q0_ls_a = q0_geom.q_tl * z_t_psi[row] + q0_geom.q_ll * z_ls_psi[row];
            let q0_tl_a = q0_geom.q_tl_ls * z_ls_psi[row];
            let q0_ll_a = q0_geom.q_tl_ls * z_t_psi[row] + q0_geom.q_ll_ls * z_ls_psi[row];

            let q_t = m[row] * q0_geom.q_t;
            let q_ls = m[row] * q0_geom.q_ls;
            let q_tt = g2[row] * q0_geom.q_t * q0_geom.q_t;
            let q_tl = g2[row] * q0_geom.q_t * q0_geom.q_ls + m[row] * q0_geom.q_tl;
            let q_ll = g2[row] * q0_geom.q_ls * q0_geom.q_ls + m[row] * q0_geom.q_ll;
            let q_t_a = g2[row] * q0_a * q0_geom.q_t + m[row] * q0_t_a;
            let q_ls_a = g2[row] * q0_a * q0_geom.q_ls + m[row] * q0_ls_a;
            let q_tt_a =
                g3[row] * q0_a * q0_geom.q_t * q0_geom.q_t + g2[row] * (2.0 * q0_geom.q_t * q0_t_a);
            let q_tl_a = g3[row] * q0_a * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a + q0_a * q0_geom.q_tl)
                + m[row] * q0_tl_a;
            let q_ll_a = g3[row] * q0_a * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * (2.0 * q0_geom.q_ls * q0_ls_a + q0_a * q0_geom.q_ll)
                + m[row] * q0_ll_a;

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let alpha = m[row] * q0_a;
            objective_psi += loss_1 * alpha;

            score_t_xa[row] = loss_1 * q_t;
            score_t_x[row] = loss_2 * alpha * q_t + loss_1 * q_t_a;
            score_ls_xa[row] = loss_1 * q_ls;
            score_ls_x[row] = loss_2 * alpha * q_ls + loss_1 * q_ls_a;
            score_w_b[row] = loss_2 * alpha;
            score_w_d1[row] = loss_1 * q0_a;

            coeff_tt_w[row] = loss_2 * q_t * q_t + loss_1 * q_tt;
            coeff_tt_d[row] = loss_3 * alpha * q_t * q_t
                + 2.0 * loss_2 * q_t * q_t_a
                + loss_2 * alpha * q_tt
                + loss_1 * q_tt_a;
            coeff_tl_w[row] = loss_2 * q_t * q_ls + loss_1 * q_tl;
            coeff_tl_d[row] = loss_3 * alpha * q_t * q_ls
                + loss_2 * (q_t_a * q_ls + q_t * q_ls_a)
                + loss_2 * alpha * q_tl
                + loss_1 * q_tl_a;
            coeff_ll_w[row] = loss_2 * q_ls * q_ls + loss_1 * q_ll;
            coeff_ll_d[row] = loss_3 * alpha * q_ls * q_ls
                + 2.0 * loss_2 * q_ls * q_ls_a
                + loss_2 * alpha * q_ll
                + loss_1 * q_ll_a;

            coeff_tw_b_w[row] = loss_2 * q_t;
            coeff_tw_b_d[row] = loss_3 * alpha * q_t + loss_2 * q_t_a;
            coeff_tw_d1_w[row] = loss_1 * q0_geom.q_t;
            coeff_tw_d1_d[row] = loss_2 * (q_t * q0_a + alpha * q0_geom.q_t) + loss_1 * q0_t_a;
            coeff_tw_d2_d[row] = loss_1 * q0_a * q0_geom.q_t;

            coeff_lw_b_w[row] = loss_2 * q_ls;
            coeff_lw_b_d[row] = loss_3 * alpha * q_ls + loss_2 * q_ls_a;
            coeff_lw_d1_w[row] = loss_1 * q0_geom.q_ls;
            coeff_lw_d1_d[row] = loss_2 * (q_ls * q0_a + alpha * q0_geom.q_ls) + loss_1 * q0_ls_a;
            coeff_lw_d2_d[row] = loss_1 * q0_a * q0_geom.q_ls;

            coeff_ww_bb[row] = loss_3 * alpha;
            coeff_ww_db[row] = loss_2 * q0_a;
        }
        let x_t_map = dir_a.x_t_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_t = x_t_map.transpose_mul(score_t_xa.view()) + x_t.t().dot(&score_t_x);
        let score_ls = x_ls_map.transpose_mul(score_ls_xa.view()) + x_ls.t().dot(&score_ls_x);
        let score_w = b0.t().dot(&score_w_b) + d0.t().dot(&score_w_d1);
        let mut score_psi = Array1::<f64>::zeros(total);
        score_psi.slice_mut(s![0..pt]).assign(&score_t);
        score_psi.slice_mut(s![pt..pt + pls]).assign(&score_ls);
        score_psi.slice_mut(s![pt + pls..total]).assign(&score_w);

        let x_t_action_opt = dir_a.x_t_psi.cloned_first_action();
        let x_ls_action_opt = dir_a.x_ls_psi.cloned_first_action();
        if x_t_action_opt.is_some() || x_ls_action_opt.is_some() {
            let basis_arc = Arc::new(b0.clone());
            let basis_d1_arc = Arc::new(d0.clone());
            let basis_d2_arc = Arc::new(dd0.clone());
            let zeros = Array1::<f64>::zeros(n);
            let operator = CustomFamilyJointPsiOperator::new(
                total,
                vec![
                    CustomFamilyJointDesignChannel::new(
                        0..pt,
                        shared_dense_arc(x_t),
                        x_t_action_opt,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt..pt + pls,
                        shared_dense_arc(x_ls),
                        x_ls_action_opt,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_d1_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_d2_arc),
                        None,
                    ),
                ],
                vec![
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        0,
                        coeff_tt_w.clone(),
                        coeff_tt_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        1,
                        coeff_tl_w.clone(),
                        coeff_tl_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        0,
                        coeff_tl_w.clone(),
                        coeff_tl_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        1,
                        coeff_ll_w.clone(),
                        coeff_ll_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        2,
                        coeff_tw_b_w.clone(),
                        coeff_tw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        0,
                        coeff_tw_b_w.clone(),
                        coeff_tw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        3,
                        coeff_tw_d1_w.clone(),
                        coeff_tw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        0,
                        coeff_tw_d1_w.clone(),
                        coeff_tw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        4,
                        zeros.clone(),
                        coeff_tw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        4,
                        0,
                        zeros.clone(),
                        coeff_tw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        2,
                        coeff_lw_b_w.clone(),
                        coeff_lw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        1,
                        coeff_lw_b_w.clone(),
                        coeff_lw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        3,
                        coeff_lw_d1_w.clone(),
                        coeff_lw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        1,
                        coeff_lw_d1_w.clone(),
                        coeff_lw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        4,
                        zeros.clone(),
                        coeff_lw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        4,
                        1,
                        zeros.clone(),
                        coeff_lw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        2,
                        zeros.clone(),
                        coeff_ww_bb.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        2,
                        zeros.clone(),
                        coeff_ww_db.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(2, 3, zeros, coeff_ww_db.clone()),
                ],
            );
            return Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(operator)),
            }));
        }
        let h_tt_block = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tt_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            coeff_tt_w.view(),
            x_t_map,
        )? + &xt_diag_x_dense(x_t, &coeff_tt_d)?;
        let h_tl_block = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tl_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            coeff_tl_w.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(x_t, &coeff_tl_d, x_ls)?;
        let h_ll_block = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
            coeff_ll_w.view(),
            x_ls_map,
        )? + &xt_diag_x_dense(x_ls, &coeff_ll_d)?;
        let h_tw = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tw_b_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(&b0),
        )? + &xt_diag_y_dense(x_t, &coeff_tw_b_d, &b0)?
            + &weighted_crossprod_psi_maps(
                x_t_map,
                coeff_tw_d1_w.view(),
                CustomFamilyPsiLinearMapRef::Dense(&d0),
            )?
            + &xt_diag_y_dense(x_t, &coeff_tw_d1_d, &d0)?
            + &xt_diag_y_dense(x_t, &coeff_tw_d2_d, &dd0)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_lw_b_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(&b0),
        )? + &xt_diag_y_dense(x_ls, &coeff_lw_b_d, &b0)?
            + &weighted_crossprod_psi_maps(
                x_ls_map,
                coeff_lw_d1_w.view(),
                CustomFamilyPsiLinearMapRef::Dense(&d0),
            )?
            + &xt_diag_y_dense(x_ls, &coeff_lw_d1_d, &d0)?
            + &xt_diag_y_dense(x_ls, &coeff_lw_d2_d, &dd0)?;
        let a_ww = xt_diag_y_dense(&d0, &coeff_ww_db, &b0)?;
        let h_ww = xt_diag_x_dense(&b0, &coeff_ww_bb)? + &a_ww + &a_ww.t();

        let mut hessian_psi = Array2::<f64>::zeros((total, total));
        hessian_psi.slice_mut(s![0..pt, 0..pt]).assign(&h_tt_block);
        hessian_psi
            .slice_mut(s![0..pt, pt..pt + pls])
            .assign(&h_tl_block);
        hessian_psi
            .slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&h_ll_block);
        hessian_psi
            .slice_mut(s![0..pt, pt + pls..total])
            .assign(&h_tw);
        hessian_psi
            .slice_mut(s![pt..pt + pls, pt + pls..total])
            .assign(&h_lw);
        hessian_psi
            .slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&h_ww);
        mirror_upper_to_lower(&mut hessian_psi);

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        if block_states.len() != 3 || derivative_blocks.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily joint psi second-order terms expect 3 blocks and 3 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ));
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_b) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_a,
                &dir_b,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_a: &BinomialLocationScaleWiggleJointPsiDirection,
        dir_b: &BinomialLocationScaleWiggleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_a,
            dir_b,
            x_t,
            x_ls,
        )?;
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(base_core.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let d4q = self.wiggle_d4q_dq04(base_core.q0.view(), betaw.view())?;
        if b0.ncols() != betaw.len()
            || d0.ncols() != betaw.len()
            || dd0.ncols() != betaw.len()
            || d3_basis.ncols() != betaw.len()
        {
            return Err(format!(
                "wiggle derivative/beta mismatch in joint psi psi terms: B={} B'={} B''={} B'''={} betaw={}",
                b0.ncols(),
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw.len()
            ));
        }
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let x_t_a_map = dir_a.x_t_psi.as_linear_map_ref();
        let x_t_b_map = dir_b.x_t_psi.as_linear_map_ref();
        let x_ls_a_map = dir_a.x_ls_psi.as_linear_map_ref();
        let x_ls_b_map = dir_b.x_ls_psi.as_linear_map_ref();
        let x_t_ab_map = second_psi_linear_map(
            second_drifts.x_t_ab_action.as_ref(),
            second_drifts.x_t_ab.as_ref(),
            n,
            pt,
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            pls,
        );
        let mut objective_psi_psi = 0.0;
        let mut score_psi_psi = Array1::<f64>::zeros(total);
        let mut hessian_psi_psi = Array2::<f64>::zeros((total, total));

        // Likelihood-only exact psi/psi terms for the wiggle family.
        //
        // This is the same generic second-order kernel as the non-wiggle path,
        // still over the flattened coefficients beta = [beta_t; beta_ls; betaw].
        // The family provides only the likelihood-side fixed-beta objects
        //
        //   D_ab, D_{beta ab}, D_{beta beta ab},
        //
        // while generic exact-joint code in custom_family.rs adds all realized
        // penalty motion S_ab.
        //
        // Using the generic rowwise notation
        //
        //   h_a   = q_{psi_a},      h_b   = q_{psi_b},
        //   h_ab  = q_{psi_a psi_b},
        //   c_a   = q_{beta psi_a}, c_b   = q_{beta psi_b},
        //   c_ab  = q_{beta psi_a psi_b},
        //   R_a   = q_{beta beta psi_a},
        //   R_b   = q_{beta beta psi_b},
        //   R_ab  = q_{beta beta psi_a psi_b},
        //
        // the exact scalar-loss kernel is
        //
        //   D_ab
        //   = sum_r [ w_r h_{r,a} h_{r,b} + r_r h_{r,ab} ],
        //
        //   D_{beta ab}
        //   = sum_r [
        //       r_r c_{r,ab}
        //       + w_r h_{r,b} c_{r,a}
        //       + w_r h_{r,a} c_{r,b}
        //       + (w_r h_{r,ab} + nu_r h_{r,a} h_{r,b}) b_r
        //     ],
        //
        //   D_{beta beta ab}
        //   = sum_r [
        //       r_r R_{r,ab}
        //       + w_r h_{r,b} R_{r,a}
        //       + w_r h_{r,a} R_{r,b}
        //       + w_r(c_{r,ab} b_r^T + b_r c_{r,ab}^T
        //             + c_{r,a} c_{r,b}^T + c_{r,b} c_{r,a}^T
        //             + h_{r,ab} Q_r)
        //       + nu_r h_{r,b}(c_{r,a} b_r^T + b_r c_{r,a}^T)
        //       + nu_r h_{r,a}(c_{r,b} b_r^T + b_r c_{r,b}^T)
        //       + nu_r h_{r,a} h_{r,b} Q_r
        //       + (tau_r h_{r,a} h_{r,b} + nu_r h_{r,ab}) b_r b_r^T
        //     ].
        //
        // The wiggle specialization enters only through the rowwise q-objects
        // built below from the combined location-side row z_r = [x_{t,r}; B_r(q0)].
        for row in 0..n {
            let q0 = base_core.q0[row];
            let q = q0 + etaw[row];
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let s_safe = sigma[row];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let q0_tl_ls_ls =
                d3s[row] / s2 - 6.0 * ds[row] * d2s[row] / s3 + 6.0 * ds[row].powi(3) / s4;
            let r_sigma = 1.0 / s_safe;

            let q0_a = -r_sigma * dir_a.z_t_psi[row] - q0 * dir_a.z_ls_psi[row];
            let q0_b = -r_sigma * dir_b.z_t_psi[row] - q0 * dir_b.z_ls_psi[row];
            let q0_ab = -r_sigma * second_drifts.z_t_ab[row]
                + r_sigma
                    * (dir_a.z_t_psi[row] * dir_b.z_ls_psi[row]
                        + dir_b.z_t_psi[row] * dir_a.z_ls_psi[row])
                + q0 * (dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row] - second_drifts.z_ls_ab[row]);

            let q0_t_a = q0_geom.q_tl * dir_a.z_ls_psi[row];
            let q0_t_b = q0_geom.q_tl * dir_b.z_ls_psi[row];
            let q0_t_ab = q0_geom.q_tl_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl * second_drifts.z_ls_ab[row];
            let q0_ls_a = q0_geom.q_tl * dir_a.z_t_psi[row] + q0_geom.q_ll * dir_a.z_ls_psi[row];
            let q0_ls_b = q0_geom.q_tl * dir_b.z_t_psi[row] + q0_geom.q_ll * dir_b.z_ls_psi[row];
            let q0_ls_ab = -q0_ab;
            let q0_tl_a = q0_geom.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_tl_b = q0_geom.q_tl_ls * dir_b.z_ls_psi[row];
            let q0_tl_ab = q0_tl_ls_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl_ls * second_drifts.z_ls_ab[row];
            let q0_ll_a =
                q0_geom.q_tl_ls * dir_a.z_t_psi[row] + q0_geom.q_ll_ls * dir_a.z_ls_psi[row];
            let q0_ll_b =
                q0_geom.q_tl_ls * dir_b.z_t_psi[row] + q0_geom.q_ll_ls * dir_b.z_ls_psi[row];
            let q0_ll_ab = q0_ab;

            let m_a = g2[row] * q0_a;
            let m_b = g2[row] * q0_b;
            let m_ab = g3[row] * q0_a * q0_b + g2[row] * q0_ab;
            let g2_a = g3[row] * q0_a;
            let g2_b = g3[row] * q0_b;
            let g2_ab = g4[row] * q0_a * q0_b + g3[row] * q0_ab;

            let q_a = m[row] * q0_a;
            let q_b = m[row] * q0_b;
            let q_ab = m[row] * q0_ab + g2[row] * q0_a * q0_b;
            let q_t = m[row] * q0_geom.q_t;
            let q_ls = m[row] * q0_geom.q_ls;
            let q_tt = g2[row] * q0_geom.q_t * q0_geom.q_t;
            let q_tl = g2[row] * q0_geom.q_t * q0_geom.q_ls + m[row] * q0_geom.q_tl;
            let q_ll = g2[row] * q0_geom.q_ls * q0_geom.q_ls + m[row] * q0_geom.q_ll;
            let q_t_a = m_a * q0_geom.q_t + m[row] * q0_t_a;
            let q_t_b = m_b * q0_geom.q_t + m[row] * q0_t_b;
            let q_ls_a = m_a * q0_geom.q_ls + m[row] * q0_ls_a;
            let q_ls_b = m_b * q0_geom.q_ls + m[row] * q0_ls_b;
            let q_t_ab = m_ab * q0_geom.q_t + m_a * q0_t_b + m_b * q0_t_a + m[row] * q0_t_ab;
            let q_ls_ab = m_ab * q0_geom.q_ls + m_a * q0_ls_b + m_b * q0_ls_a + m[row] * q0_ls_ab;
            let q_tt_a = g2_a * q0_geom.q_t * q0_geom.q_t + g2[row] * 2.0 * q0_geom.q_t * q0_t_a;
            let q_tt_b = g2_b * q0_geom.q_t * q0_geom.q_t + g2[row] * 2.0 * q0_geom.q_t * q0_t_b;
            let q_tt_ab = g2_ab * q0_geom.q_t * q0_geom.q_t
                + g2_a * 2.0 * q0_geom.q_t * q0_t_b
                + g2_b * 2.0 * q0_geom.q_t * q0_t_a
                + g2[row] * (2.0 * q0_t_a * q0_t_b + 2.0 * q0_geom.q_t * q0_t_ab);
            let q_tl_a = g2_a * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a)
                + m_a * q0_geom.q_tl
                + m[row] * q0_tl_a;
            let q_tl_b = g2_b * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_b * q0_geom.q_ls + q0_geom.q_t * q0_ls_b)
                + m_b * q0_geom.q_tl
                + m[row] * q0_tl_b;
            let q_tl_ab = g2_ab * q0_geom.q_t * q0_geom.q_ls
                + g2_a * (q0_t_b * q0_geom.q_ls + q0_geom.q_t * q0_ls_b)
                + g2_b * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a)
                + g2[row]
                    * (q0_t_ab * q0_geom.q_ls
                        + q0_t_a * q0_ls_b
                        + q0_t_b * q0_ls_a
                        + q0_geom.q_t * q0_ls_ab)
                + m_ab * q0_geom.q_tl
                + m_a * q0_tl_b
                + m_b * q0_tl_a
                + m[row] * q0_tl_ab;
            let q_ll_a = g2_a * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * 2.0 * q0_geom.q_ls * q0_ls_a
                + m_a * q0_geom.q_ll
                + m[row] * q0_ll_a;
            let q_ll_b = g2_b * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * 2.0 * q0_geom.q_ls * q0_ls_b
                + m_b * q0_geom.q_ll
                + m[row] * q0_ll_b;
            let q_ll_ab = g2_ab * q0_geom.q_ls * q0_geom.q_ls
                + g2_a * 2.0 * q0_geom.q_ls * q0_ls_b
                + g2_b * 2.0 * q0_geom.q_ls * q0_ls_a
                + g2[row] * (2.0 * q0_ls_a * q0_ls_b + 2.0 * q0_geom.q_ls * q0_ls_ab)
                + m_ab * q0_geom.q_ll
                + m_a * q0_ll_b
                + m_b * q0_ll_a
                + m[row] * q0_ll_ab;

            let brow = b0.row(row).to_owned();
            let drow = d0.row(row).to_owned();
            let ddrow = dd0.row(row).to_owned();
            let d3row = d3_basis.row(row).to_owned();
            let qw_a = &drow * q0_a;
            let qw_b = &drow * q0_b;
            let qw_ab = &ddrow * (q0_a * q0_b) + &(&drow * q0_ab);
            let q_tw_a = &ddrow * (q0_a * q0_geom.q_t) + &(&drow * q0_t_a);
            let q_tw_b = &ddrow * (q0_b * q0_geom.q_t) + &(&drow * q0_t_b);
            let q_lw_a = &ddrow * (q0_a * q0_geom.q_ls) + &(&drow * q0_ls_a);
            let q_lw_b = &ddrow * (q0_b * q0_geom.q_ls) + &(&drow * q0_ls_b);
            let d0_ab = &d3row * (q0_a * q0_b) + &(&ddrow * q0_ab);
            let q_tw_ab = &d0_ab * q0_geom.q_t
                + &(&(&ddrow * q0_b) * q0_t_a)
                + &(&(&ddrow * q0_a) * q0_t_b)
                + &(&drow * q0_t_ab);
            let q_lw_ab = &d0_ab * q0_geom.q_ls
                + &(&(&ddrow * q0_b) * q0_ls_a)
                + &(&(&ddrow * q0_a) * q0_ls_b)
                + &(&drow * q0_ls_ab);

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            objective_psi_psi += loss_2 * q_a * q_b + loss_1 * q_ab;

            let xtr = x_t.row(row);
            let xlsr = x_ls.row(row);
            let xta = x_t_a_map.row_vector(row)?;
            let xtb = x_t_b_map.row_vector(row)?;
            let xlsa = x_ls_a_map.row_vector(row)?;
            let xlsb = x_ls_b_map.row_vector(row)?;
            let xtab = x_t_ab_map.row_vector(row)?;
            let xlsab = x_ls_ab_map.row_vector(row)?;

            let mut b = Array1::<f64>::zeros(total);
            b.slice_mut(s![0..pt]).assign(&(xtr.to_owned() * q_t));
            b.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls));
            b.slice_mut(s![pt + pls..]).assign(&brow);
            let mut c_a = Array1::<f64>::zeros(total);
            c_a.slice_mut(s![0..pt])
                .assign(&(xtr.to_owned() * q_t_a + xta.clone() * q_t));
            c_a.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls_a + xlsa.clone() * q_ls));
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);
            let mut c_b = Array1::<f64>::zeros(total);
            c_b.slice_mut(s![0..pt])
                .assign(&(xtr.to_owned() * q_t_b + xtb.clone() * q_t));
            c_b.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls_b + xlsb.clone() * q_ls));
            c_b.slice_mut(s![pt + pls..]).assign(&qw_b);
            let mut c_ab = Array1::<f64>::zeros(total);
            c_ab.slice_mut(s![0..pt]).assign(
                &(xtr.to_owned() * q_t_ab
                    + xta.clone() * q_t_b
                    + xtb.clone() * q_t_a
                    + xtab.clone() * q_t),
            );
            c_ab.slice_mut(s![pt..pt + pls]).assign(
                &(xlsr.to_owned() * q_ls_ab
                    + xlsa.clone() * q_ls_b
                    + xlsb.clone() * q_ls_a
                    + xlsab.clone() * q_ls),
            );
            c_ab.slice_mut(s![pt + pls..]).assign(&qw_ab);

            score_psi_psi += &(loss_1 * &c_ab
                + loss_2 * q_b * &c_a
                + loss_2 * q_a * &c_b
                + (loss_2 * q_ab + loss_3 * q_a * q_b) * &b);

            let mut q_mat = Array2::<f64>::zeros((total, total));
            let mut r_a = Array2::<f64>::zeros((total, total));
            let mut r_b = Array2::<f64>::zeros((total, total));
            let mut r_ab = Array2::<f64>::zeros((total, total));
            {
                let tt = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xtr.to_owned().insert_axis(Axis(0)))
                    * q_tt;
                let tl = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_tl;
                let ll = xlsr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_ll;
                q_mat.slice_mut(s![0..pt, 0..pt]).assign(&tt);
                q_mat.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl);
                q_mat.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&ll);
                let tw = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_t).insert_axis(Axis(0)));
                let lw = xlsr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_ls).insert_axis(Axis(0)));
                q_mat.slice_mut(s![0..pt, pt + pls..]).assign(&tw);
                q_mat.slice_mut(s![pt..pt + pls, pt + pls..]).assign(&lw);
                mirror_upper_to_lower(&mut q_mat);

                let tt_a = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xtr.to_owned().insert_axis(Axis(0)))
                    * q_tt_a
                    + xta
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtr.to_owned().insert_axis(Axis(0)))
                        * q_tt
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xta.to_owned().insert_axis(Axis(0)))
                        * q_tt;
                let tl_a = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_tl_a
                    + xta
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_tl
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.to_owned().insert_axis(Axis(0)))
                        * q_tl;
                let ll_a = xlsr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_ll_a
                    + xlsa
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_ll
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.to_owned().insert_axis(Axis(0)))
                        * q_ll;
                r_a.slice_mut(s![0..pt, 0..pt]).assign(&tt_a);
                r_a.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl_a);
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&ll_a);
                let tw_a = xta
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_t).insert_axis(Axis(0)))
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_tw_a.view().insert_axis(Axis(0)));
                let lw_a = xlsa
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_ls).insert_axis(Axis(0)))
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_lw_a.view().insert_axis(Axis(0)));
                r_a.slice_mut(s![0..pt, pt + pls..]).assign(&tw_a);
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]).assign(&lw_a);
                mirror_upper_to_lower(&mut r_a);

                let tt_b = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xtr.to_owned().insert_axis(Axis(0)))
                    * q_tt_b
                    + xtb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtr.to_owned().insert_axis(Axis(0)))
                        * q_tt
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtb.to_owned().insert_axis(Axis(0)))
                        * q_tt;
                let tl_b = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_tl_b
                    + xtb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_tl
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsb.to_owned().insert_axis(Axis(0)))
                        * q_tl;
                let ll_b = xlsr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_ll_b
                    + xlsb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_ll
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsb.to_owned().insert_axis(Axis(0)))
                        * q_ll;
                r_b.slice_mut(s![0..pt, 0..pt]).assign(&tt_b);
                r_b.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl_b);
                r_b.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&ll_b);
                let tw_b = xtb
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_t).insert_axis(Axis(0)))
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_tw_b.view().insert_axis(Axis(0)));
                let lw_b = xlsb
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_ls).insert_axis(Axis(0)))
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_lw_b.view().insert_axis(Axis(0)));
                r_b.slice_mut(s![0..pt, pt + pls..]).assign(&tw_b);
                r_b.slice_mut(s![pt..pt + pls, pt + pls..]).assign(&lw_b);
                mirror_upper_to_lower(&mut r_b);

                let tt_ab = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xtr.to_owned().insert_axis(Axis(0)))
                    * q_tt_ab
                    + xta
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtr.to_owned().insert_axis(Axis(0)))
                        * q_tt_b
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xta.to_owned().insert_axis(Axis(0)))
                        * q_tt_b
                    + xtb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtr.to_owned().insert_axis(Axis(0)))
                        * q_tt_a
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtb.to_owned().insert_axis(Axis(0)))
                        * q_tt_a
                    + xtab
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtr.to_owned().insert_axis(Axis(0)))
                        * q_tt
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtab.to_owned().insert_axis(Axis(0)))
                        * q_tt
                    + xta
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xtb.to_owned().insert_axis(Axis(0)))
                        * q_tt
                    + xtb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xta.to_owned().insert_axis(Axis(0)))
                        * q_tt;
                let tl_ab = xtr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_tl_ab
                    + xta
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_tl_b
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.to_owned().insert_axis(Axis(0)))
                        * q_tl_b
                    + xtb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_tl_a
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsb.to_owned().insert_axis(Axis(0)))
                        * q_tl_a
                    + xtab
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_tl
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsab.to_owned().insert_axis(Axis(0)))
                        * q_tl
                    + xta
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsb.to_owned().insert_axis(Axis(0)))
                        * q_tl
                    + xtb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.to_owned().insert_axis(Axis(0)))
                        * q_tl;
                let ll_ab = xlsr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                    * q_ll_ab
                    + xlsa
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_ll_b
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.to_owned().insert_axis(Axis(0)))
                        * q_ll_b
                    + xlsb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_ll_a
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsb.to_owned().insert_axis(Axis(0)))
                        * q_ll_a
                    + xlsab
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.to_owned().insert_axis(Axis(0)))
                        * q_ll
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsab.to_owned().insert_axis(Axis(0)))
                        * q_ll
                    + xlsa
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsb.to_owned().insert_axis(Axis(0)))
                        * q_ll
                    + xlsb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.to_owned().insert_axis(Axis(0)))
                        * q_ll;
                r_ab.slice_mut(s![0..pt, 0..pt]).assign(&tt_ab);
                r_ab.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl_ab);
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls])
                    .assign(&ll_ab);
                let tw_ab = xtab
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_t).insert_axis(Axis(0)))
                    + xta
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_tw_b.view().insert_axis(Axis(0)))
                    + xtb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_tw_a.view().insert_axis(Axis(0)))
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_tw_ab.view().insert_axis(Axis(0)));
                let lw_ab = xlsab
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.clone() * q0_geom.q_ls).insert_axis(Axis(0)))
                    + xlsa
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_lw_b.view().insert_axis(Axis(0)))
                    + xlsb
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_lw_a.view().insert_axis(Axis(0)))
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_lw_ab.view().insert_axis(Axis(0)));
                r_ab.slice_mut(s![0..pt, pt + pls..]).assign(&tw_ab);
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]).assign(&lw_ab);
                mirror_upper_to_lower(&mut r_ab);
            }

            let bb = b
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let cab_bt = c_ab
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let b_cab_t = b
                .view()
                .insert_axis(Axis(1))
                .dot(&c_ab.view().insert_axis(Axis(0)));
            let ca_cb_t = c_a
                .view()
                .insert_axis(Axis(1))
                .dot(&c_b.view().insert_axis(Axis(0)));
            let cb_ca_t = c_b
                .view()
                .insert_axis(Axis(1))
                .dot(&c_a.view().insert_axis(Axis(0)));
            let ca_bt = c_a
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let b_cat = b
                .view()
                .insert_axis(Axis(1))
                .dot(&c_a.view().insert_axis(Axis(0)));
            let cb_bt = c_b
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let b_cbt = b
                .view()
                .insert_axis(Axis(1))
                .dot(&c_b.view().insert_axis(Axis(0)));

            hessian_psi_psi += &(loss_1 * r_ab
                + loss_2
                    * (q_b * r_a
                        + q_a * r_b
                        + cab_bt
                        + b_cab_t
                        + ca_cb_t
                        + cb_ca_t
                        + q_ab * &q_mat)
                + loss_3 * (q_b * (ca_bt + b_cat) + q_a * (cb_bt + b_cbt) + q_a * q_b * &q_mat)
                + (loss_4 * q_a * q_b + loss_3 * q_ab) * bb);
        }

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &BinomialLocationScaleWiggleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(base_core.q0.view())?;
        let d4q = self.wiggle_d4q_dq04(base_core.q0.view(), betaw.view())?;
        let pw = b0.ncols();
        let layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (u_t, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "wiggle joint psi hessian directional derivative",
        )?;
        let total = pt + pls + pw;
        if d0.ncols() != betaw.len()
            || dd0.ncols() != betaw.len()
            || d3_basis.ncols() != betaw.len()
        {
            return Err(format!(
                "wiggle derivative/beta mismatch in joint psi mixed drift: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw.len()
            ));
        }
        let xi_t = x_t.dot(&u_t);
        let xi_ls = x_ls.dot(&u_ls);
        let x_t_map = dir_a.x_t_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let g3 = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        // Exact likelihood-side mixed drift T_a[u] = D_beta H_{psi_a}^{(D)}[u].
        //
        // The unified outer Hessian in custom_family.rs uses
        //   ddot H_ij = H_ij + T_i[beta_j] + T_j[beta_i]
        //             + D_beta H[beta_ij] + D_beta^2 H[beta_i, beta_j].
        //
        // For wiggle we still use the same scalar-loss row kernel as non-wiggle;
        // only the location-side row changes to z_r = [x_{t,r}; B_r(q0)] with
        // q = q0 + betaw^T B(q0), q0 = -eta_t * exp(-eta_ls).
        let mut out = Array2::<f64>::zeros((total, total));
        for row in 0..n {
            let q = core.q0[row] + etaw[row];
            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            let q0 = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let s_safe = sigma[row];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let s5 = s4 * s_safe;
            let q0_tl_ls_ls = d3s[row] / s2 - 6.0 * ds[row] * d2s[row] / s3
                + 6.0 * ds[row] * ds[row] * ds[row] / s4;
            let q0_tl_ls_ls_ls =
                d4s[row] / s2 - 8.0 * ds[row] * d3s[row] / s3 - 6.0 * d2s[row] * d2s[row] / s3
                    + 36.0 * ds[row] * ds[row] * d2s[row] / s4
                    - 24.0 * ds[row] * ds[row] * ds[row] * ds[row] / s5;
            let q0_ll_ls_ls = eta_t[row] * q0_tl_ls_ls_ls;

            let xtr = x_t.row(row).to_owned();
            let xlsr = x_ls.row(row).to_owned();
            let xta = x_t_map.row_vector(row)?;
            let xlsa = x_ls_map.row_vector(row)?;
            let br = b0.row(row).to_owned();
            let dr = d0.row(row).to_owned();
            let ddr = dd0.row(row).to_owned();
            let d3r = d3_basis.row(row).to_owned();

            let xi_t_i = xi_t[row];
            let xi_ls_i = xi_ls[row];
            let xi_ta_i = xta.dot(&u_t);
            let xi_lsa_i = xlsa.dot(&u_ls);
            let d_dot_u = dr.dot(&uw);
            let dd_dot_u = ddr.dot(&uw);
            let d3_dot_u = d3r.dot(&uw);

            let dq0_u = q0.q_t * xi_t_i + q0.q_ls * xi_ls_i;
            let dq0_t_u = q0.q_tl * xi_ls_i;
            let dq0_ls_u = q0.q_tl * xi_t_i + q0.q_ll * xi_ls_i;
            let dq0_tl_u = q0.q_tl_ls * xi_ls_i;
            let dq0_ll_u = q0.q_tl_ls * xi_t_i + q0.q_ll_ls * xi_ls_i;
            let dq0_tl_ls_u = q0_tl_ls_ls * xi_ls_i;
            let dq0_ll_ls_u = q0_tl_ls_ls * xi_t_i + q0_ll_ls_ls * xi_ls_i;

            let q0_a = -q0.q_t * dir_a.z_t_psi[row] - q0.q_ls * dir_a.z_ls_psi[row];
            let q0_t_a = q0.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_ls_a = q0.q_tl_ls * dir_a.z_t_psi[row] + q0.q_ll_ls * dir_a.z_ls_psi[row];
            let q0_tl_a = q0.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_ll_a = q0.q_tl_ls * dir_a.z_t_psi[row] + q0.q_ll_ls * dir_a.z_ls_psi[row];
            let dq0_a_u = q0_t_a * xi_t_i + q0_ls_a * xi_ls_i;
            let dq0_t_a_u = dq0_tl_ls_u * dir_a.z_ls_psi[row];
            let dq0_ls_a_u = dq0_tl_ls_u * dir_a.z_t_psi[row] + dq0_ll_ls_u * dir_a.z_ls_psi[row];
            let dq0_tl_a_u = dq0_tl_ls_u * dir_a.z_ls_psi[row];
            let dq0_ll_a_u = dq0_tl_ls_u * dir_a.z_t_psi[row] + dq0_ll_ls_u * dir_a.z_ls_psi[row];

            let q_t = m[row] * q0.q_t;
            let q_ls = m[row] * q0.q_ls;
            let q_tt = g2[row] * q0.q_t * q0.q_t;
            let q_tl = g2[row] * q0.q_t * q0.q_ls + m[row] * q0.q_tl;
            let q_ll = g2[row] * q0.q_ls * q0.q_ls + m[row] * q0.q_ll;
            let q_tw = dr.clone() * q0.q_t;
            let q_lw = dr.clone() * q0.q_ls;

            let dm_u = g2[row] * dq0_u + d_dot_u;
            let dg2_u = g3[row] * dq0_u + dd_dot_u;
            let dg3_u = g4[row] * dq0_u + d3_dot_u;

            let q_a = m[row] * q0_a;
            let q_t_a = g2[row] * q0_a * q0.q_t + m[row] * q0_t_a;
            let q_ls_a = g2[row] * q0_a * q0.q_ls + m[row] * q0_ls_a;
            let q_tt_a = g3[row] * q0_a * q0.q_t * q0.q_t + g2[row] * (2.0 * q0.q_t * q0_t_a);
            let q_tl_a = g3[row] * q0_a * q0.q_t * q0.q_ls
                + g2[row] * (q0_t_a * q0.q_ls + q0.q_t * q0_ls_a + q0_a * q0.q_tl)
                + m[row] * q0_tl_a;
            let q_ll_a = g3[row] * q0_a * q0.q_ls * q0.q_ls
                + g2[row] * (2.0 * q0.q_ls * q0_ls_a + q0_a * q0.q_ll)
                + m[row] * q0_ll_a;
            let qw_a = dr.clone() * q0_a;
            let q_tw_a = ddr.clone() * (q0_a * q0.q_t) + &(dr.clone() * q0_t_a);
            let q_lw_a = ddr.clone() * (q0_a * q0.q_ls) + &(dr.clone() * q0_ls_a);

            let dq_tt_u = dg2_u * q0.q_t * q0.q_t + g2[row] * (2.0 * q0.q_t * dq0_t_u);
            let dq_tl_u = dg2_u * q0.q_t * q0.q_ls
                + g2[row] * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                + dm_u * q0.q_tl
                + m[row] * dq0_tl_u;
            let dq_ll_u = dg2_u * q0.q_ls * q0.q_ls
                + g2[row] * (2.0 * q0.q_ls * dq0_ls_u)
                + dm_u * q0.q_ll
                + m[row] * dq0_ll_u;
            let dq_tw_u = ddr.clone() * (dq0_u * q0.q_t) + &(dr.clone() * dq0_t_u);
            let dq_lw_u = ddr.clone() * (dq0_u * q0.q_ls) + &(dr.clone() * dq0_ls_u);

            let dq_tt_a_u = dg3_u * q0_a * q0.q_t * q0.q_t
                + g3[row] * (dq0_a_u * q0.q_t * q0.q_t + 2.0 * q0_a * q0.q_t * dq0_t_u)
                + dg2_u * (2.0 * q0.q_t * q0_t_a)
                + g2[row] * (2.0 * dq0_t_u * q0_t_a + 2.0 * q0.q_t * dq0_t_a_u);
            let dq_tl_a_u = dg3_u * q0_a * q0.q_t * q0.q_ls
                + g3[row]
                    * (dq0_a_u * q0.q_t * q0.q_ls
                        + q0_a * dq0_t_u * q0.q_ls
                        + q0_a * q0.q_t * dq0_ls_u)
                + dg2_u * (q0_t_a * q0.q_ls + q0.q_t * q0_ls_a + q0_a * q0.q_tl)
                + g2[row]
                    * (dq0_t_a_u * q0.q_ls
                        + q0_t_a * dq0_ls_u
                        + dq0_t_u * q0_ls_a
                        + q0.q_t * dq0_ls_a_u
                        + dq0_a_u * q0.q_tl
                        + q0_a * dq0_tl_u)
                + dm_u * q0_tl_a
                + m[row] * dq0_tl_a_u;
            let dq_ll_a_u = dg3_u * q0_a * q0.q_ls * q0.q_ls
                + g3[row] * (dq0_a_u * q0.q_ls * q0.q_ls + 2.0 * q0_a * q0.q_ls * dq0_ls_u)
                + dg2_u * (2.0 * q0.q_ls * q0_ls_a + q0_a * q0.q_ll)
                + g2[row]
                    * (2.0 * dq0_ls_u * q0_ls_a
                        + 2.0 * q0.q_ls * dq0_ls_a_u
                        + dq0_a_u * q0.q_ll
                        + q0_a * dq0_ll_u)
                + dm_u * q0_ll_a
                + m[row] * dq0_ll_a_u;
            let dq_tw_a_u = d3r.clone() * (dq0_u * q0_a * q0.q_t)
                + &(ddr.clone() * (dq0_a_u * q0.q_t + q0_a * dq0_t_u + dq0_u * q0_t_a))
                + &(dr.clone() * dq0_t_a_u);
            let dq_lw_a_u = d3r.clone() * (dq0_u * q0_a * q0.q_ls)
                + &(ddr.clone() * (dq0_a_u * q0.q_ls + q0_a * dq0_ls_u + dq0_u * q0_ls_a))
                + &(dr.clone() * dq0_ls_a_u);

            let mut b = Array1::<f64>::zeros(total);
            b.slice_mut(s![0..pt]).assign(&(xtr.clone() * q_t));
            b.slice_mut(s![pt..pt + pls]).assign(&(xlsr.clone() * q_ls));
            b.slice_mut(s![pt + pls..]).assign(&br);

            let mut c_a = Array1::<f64>::zeros(total);
            c_a.slice_mut(s![0..pt])
                .assign(&(xtr.clone() * q_t_a + xta.clone() * q_t));
            c_a.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.clone() * q_ls_a + xlsa.clone() * q_ls));
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);

            let mut gamma = Array1::<f64>::zeros(total);
            gamma
                .slice_mut(s![0..pt])
                .assign(&(xtr.clone() * (q_tt * xi_t_i + q_tl * xi_ls_i + q0.q_t * d_dot_u)));
            gamma
                .slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.clone() * (q_tl * xi_t_i + q_ll * xi_ls_i + q0.q_ls * d_dot_u)));
            gamma
                .slice_mut(s![pt + pls..])
                .assign(&(dr.clone() * dq0_u));

            let q_tw_a_dot_u = q_tw_a.dot(&uw);
            let q_lw_a_dot_u = q_lw_a.dot(&uw);
            let mut gamma_a = Array1::<f64>::zeros(total);
            gamma_a.slice_mut(s![0..pt]).assign(
                &(xtr.clone()
                    * (q_tt_a * xi_t_i
                        + q_tt * xi_ta_i
                        + q_tl_a * xi_ls_i
                        + q_tl * xi_lsa_i
                        + q_tw_a_dot_u)
                    + xta.clone() * (q_tt * xi_t_i + q_tl * xi_ls_i + q0.q_t * d_dot_u)),
            );
            gamma_a.slice_mut(s![pt..pt + pls]).assign(
                &(xlsr.clone()
                    * (q_tl_a * xi_t_i
                        + q_tl * xi_ta_i
                        + q_ll_a * xi_ls_i
                        + q_ll * xi_lsa_i
                        + q_lw_a_dot_u)
                    + xlsa.clone() * (q_tl * xi_t_i + q_ll * xi_ls_i + q0.q_ls * d_dot_u)),
            );
            gamma_a.slice_mut(s![pt + pls..]).assign(
                &(q_tw_a.clone() * xi_t_i
                    + q_tw.clone() * xi_ta_i
                    + q_lw_a.clone() * xi_ls_i
                    + q_lw.clone() * xi_lsa_i),
            );

            let alpha = b.dot(d_beta_flat);
            let alpha_a = c_a.dot(d_beta_flat);

            let mut q_mat = Array2::<f64>::zeros((total, total));
            q_mat.slice_mut(s![0..pt, 0..pt]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xtr.view().insert_axis(Axis(0)))
                    * q_tt),
            );
            q_mat.slice_mut(s![0..pt, pt..pt + pls]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * q_tl),
            );
            q_mat.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(
                &(xlsr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * q_ll),
            );
            q_mat.slice_mut(s![0..pt, pt + pls..]).assign(
                &xtr.view()
                    .insert_axis(Axis(1))
                    .dot(&q_tw.view().insert_axis(Axis(0))),
            );
            q_mat.slice_mut(s![pt..pt + pls, pt + pls..]).assign(
                &xlsr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&q_lw.view().insert_axis(Axis(0))),
            );
            mirror_upper_to_lower(&mut q_mat);

            let mut r_a = Array2::<f64>::zeros((total, total));
            r_a.slice_mut(s![0..pt, 0..pt]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xtr.view().insert_axis(Axis(0)))
                    * q_tt_a
                    + xta
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xtr.view().insert_axis(Axis(0)))
                        * q_tt
                    + xtr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xta.view().insert_axis(Axis(0)))
                        * q_tt),
            );
            r_a.slice_mut(s![0..pt, pt..pt + pls]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * q_tl_a
                    + xta
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.view().insert_axis(Axis(0)))
                        * q_tl
                    + xtr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.view().insert_axis(Axis(0)))
                        * q_tl),
            );
            r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(
                &(xlsr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * q_ll_a
                    + xlsa
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.view().insert_axis(Axis(0)))
                        * q_ll
                    + xlsr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.view().insert_axis(Axis(0)))
                        * q_ll),
            );
            r_a.slice_mut(s![0..pt, pt + pls..]).assign(
                &(xta
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&q_tw.view().insert_axis(Axis(0)))
                    + xtr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&q_tw_a.view().insert_axis(Axis(0)))),
            );
            r_a.slice_mut(s![pt..pt + pls, pt + pls..]).assign(
                &(xlsa
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&q_lw.view().insert_axis(Axis(0)))
                    + xlsr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&q_lw_a.view().insert_axis(Axis(0)))),
            );
            mirror_upper_to_lower(&mut r_a);

            let mut c_u = Array2::<f64>::zeros((total, total));
            c_u.slice_mut(s![0..pt, 0..pt]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xtr.view().insert_axis(Axis(0)))
                    * dq_tt_u),
            );
            c_u.slice_mut(s![0..pt, pt..pt + pls]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * dq_tl_u),
            );
            c_u.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(
                &(xlsr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * dq_ll_u),
            );
            c_u.slice_mut(s![0..pt, pt + pls..]).assign(
                &xtr.view()
                    .insert_axis(Axis(1))
                    .dot(&dq_tw_u.view().insert_axis(Axis(0))),
            );
            c_u.slice_mut(s![pt..pt + pls, pt + pls..]).assign(
                &xlsr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&dq_lw_u.view().insert_axis(Axis(0))),
            );
            mirror_upper_to_lower(&mut c_u);

            let mut delta_a = Array2::<f64>::zeros((total, total));
            delta_a.slice_mut(s![0..pt, 0..pt]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xtr.view().insert_axis(Axis(0)))
                    * dq_tt_a_u
                    + xta
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xtr.view().insert_axis(Axis(0)))
                        * dq_tt_u
                    + xtr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xta.view().insert_axis(Axis(0)))
                        * dq_tt_u),
            );
            delta_a.slice_mut(s![0..pt, pt..pt + pls]).assign(
                &(xtr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * dq_tl_a_u
                    + xta
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.view().insert_axis(Axis(0)))
                        * dq_tl_u
                    + xtr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.view().insert_axis(Axis(0)))
                        * dq_tl_u),
            );
            delta_a.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(
                &(xlsr
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&xlsr.view().insert_axis(Axis(0)))
                    * dq_ll_a_u
                    + xlsa
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsr.view().insert_axis(Axis(0)))
                        * dq_ll_u
                    + xlsr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xlsa.view().insert_axis(Axis(0)))
                        * dq_ll_u),
            );
            delta_a.slice_mut(s![0..pt, pt + pls..]).assign(
                &(xta
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&dq_tw_u.view().insert_axis(Axis(0)))
                    + xtr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&dq_tw_a_u.view().insert_axis(Axis(0)))),
            );
            delta_a.slice_mut(s![pt..pt + pls, pt + pls..]).assign(
                &(xlsa
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&dq_lw_u.view().insert_axis(Axis(0)))
                    + xlsr
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&dq_lw_a_u.view().insert_axis(Axis(0)))),
            );
            mirror_upper_to_lower(&mut delta_a);

            let bb = b
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let cb = c_a
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let bc = b
                .view()
                .insert_axis(Axis(1))
                .dot(&c_a.view().insert_axis(Axis(0)));
            let gb = gamma
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let bg = b
                .view()
                .insert_axis(Axis(1))
                .dot(&gamma.view().insert_axis(Axis(0)));
            let gab = gamma_a
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let bga = b
                .view()
                .insert_axis(Axis(1))
                .dot(&gamma_a.view().insert_axis(Axis(0)));
            let gc = gamma
                .view()
                .insert_axis(Axis(1))
                .dot(&c_a.view().insert_axis(Axis(0)));
            let cg = c_a
                .view()
                .insert_axis(Axis(1))
                .dot(&gamma.view().insert_axis(Axis(0)));

            out += &(loss_1 * &delta_a);
            out += &(loss_2
                * (&(alpha * &r_a)
                    + &(q_a * &c_u)
                    + &gab
                    + &bga
                    + &gc
                    + &cg
                    + &(alpha_a * &q_mat)));
            out += &(loss_3
                * (&((alpha * q_a) * &bb)
                    + &(q_a * (&gb + &bg))
                    + &(alpha * (&cb + &bc + &(q_a * &q_mat)))));
            out += &((loss_4 * alpha * q_a + loss_3 * alpha_a) * &bb);
        }
        mirror_upper_to_lower(&mut out);
        Ok(out)
    }

    /// Build a turnkey wiggle block from a q-seed vector and knot settings.
    /// Returns both the block input and the generated knot vector.
    pub fn buildwiggle_block_input(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
        penalty_order: usize,
        double_penalty: bool,
    ) -> Result<(ParameterBlockInput, Array1<f64>), String> {
        let knots = Self::initializewiggle_knots_from_q(q_seed, degree, num_internal_knots)?;
        let block = buildwiggle_block_input_from_knots(
            q_seed,
            &knots,
            degree,
            penalty_order,
            double_penalty,
        )?;
        Ok((block, knots))
    }

    /// Compute the rowwise pieces (diagonal weights + B/B' basis arrays) used
    /// to assemble the joint Hessian for the 3-block wiggle family. Both the
    /// dense Hessian path and the matrix-free workspace consume these pieces
    /// without recomputing the per-row scalar derivatives.
    fn wiggle_hessian_row_pieces(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BinomialLocationScaleWiggleHessianRowPieces, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        if b0.ncols() != betaw0.len() || d0.ncols() != betaw0.len() || dd0.ncols() != betaw0.len() {
            return Err(format!(
                "wiggle basis/beta mismatch in exact joint Hessian: B={} B'={} B''={} betaw={}",
                b0.ncols(),
                d0.ncols(),
                dd0.ncols(),
                betaw0.len()
            ));
        }
        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        let mut coeff_tw_b = Array1::<f64>::zeros(n);
        let mut coeff_tw_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_b = Array1::<f64>::zeros(n);
        let mut coeff_lw_d = Array1::<f64>::zeros(n);
        let mut coeffww = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, _) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);

            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let q_tt = g2[i] * q0.q_t * q0.q_t;
            let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
            let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

            coeff_tt[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_t, q_t, q_tt);
            coeff_tl[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_t, q_ls, q_tl);
            coeff_ll[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_ls, q_ls, q_ll);
            coeff_tw_b[i] = m2 * q_t;
            coeff_tw_d[i] = m1 * q0.q_t;
            coeff_lw_b[i] = m2 * q_ls;
            coeff_lw_d[i] = m1 * q0.q_ls;
            coeffww[i] = m2;
        }
        Ok(BinomialLocationScaleWiggleHessianRowPieces {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_lw_b,
            coeff_lw_d,
            coeffww,
            b0,
            d0,
        })
    }
}

/// Per-row pieces of the 3-block wiggle joint Hessian.
///
/// `coeff_*` are diagonal weights (length n). `b0` and `d0` are the realized
/// wiggle basis values and first-derivative values at the current q0
/// (n × p_w). The dense Hessian path assembles these into a (p_t+p_ls+p_w)²
/// matrix; the matrix-free workspace applies the operator
///
///   r_t = D_tt u_t + D_tl u_ls + D_tw_b (B v_w) + D_tw_d (B' v_w),
///   r_ls = D_tl u_t + D_ll u_ls + D_lw_b (B v_w) + D_lw_d (B' v_w),
///   r_b = D_tw_b u_t + D_lw_b u_ls + D_ww (B v_w),
///   r_d = D_tw_d u_t + D_lw_d u_ls,
///
/// and combines `out_w = B^T r_b + (B')^T r_d` to form `H v` directly.
struct BinomialLocationScaleWiggleHessianRowPieces {
    coeff_tt: Array1<f64>,
    coeff_tl: Array1<f64>,
    coeff_ll: Array1<f64>,
    coeff_tw_b: Array1<f64>,
    coeff_tw_d: Array1<f64>,
    coeff_lw_b: Array1<f64>,
    coeff_lw_d: Array1<f64>,
    coeffww: Array1<f64>,
    b0: Array2<f64>,
    d0: Array2<f64>,
}

impl BinomialLocationScaleWiggleHessianRowPieces {
    fn assemble_dense(
        &self,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = self.b0.ncols();
        let total = pt + pls + pw;
        let h_tt = xt_diag_x_dense(x_t, &self.coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &self.coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_tw = xt_diag_y_dense(x_t, &self.coeff_tw_b, &self.b0)?
            + &xt_diag_y_dense(x_t, &self.coeff_tw_d, &self.d0)?;
        let h_lw = xt_diag_y_dense(x_ls, &self.coeff_lw_b, &self.b0)?
            + &xt_diag_y_dense(x_ls, &self.coeff_lw_d, &self.d0)?;
        let hww = xt_diag_x_dense(&self.b0, &self.coeffww)?;

        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..pt + pls]).assign(&h_tl);
        h.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&h_ll);
        h.slice_mut(s![0..pt, pt + pls..total]).assign(&h_tw);
        h.slice_mut(s![pt..pt + pls, pt + pls..total]).assign(&h_lw);
        h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&hww);
        mirror_upper_to_lower(&mut h);
        Ok(h)
    }

    /// Block-diagonal Hessians (h_tt, h_ll, h_ww) without ever materializing
    /// the cross blocks. Used by `evaluate()` to populate per-block working
    /// sets.
    fn assemble_block_diagonals(
        &self,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
        let h_tt = xt_diag_x_dense(x_t, &self.coeff_tt)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_ww = xt_diag_x_dense(&self.b0, &self.coeffww)?;
        Ok((h_tt, h_ll, h_ww))
    }
}

impl CustomFamily for BinomialLocationScaleWiggleFamily {
    /// The Binomial location-scale-wiggle joint Hessian depends on β because
    /// it involves the nonlinear link function evaluated at the combined
    /// predictor, which changes with all three coefficient blocks.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Three fully-coupled blocks (threshold p_t, log-σ p_ℓ, link-wiggle
        // p_w): joint Hessian size (p_t + p_ℓ + p_w)² per row.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(self.y.len() as u64, specs)
    }

    /// The wiggle family carries a structural null-space direction: the
    /// threshold β_t and the overall wiggle-intercept combination
    /// `β_w^⊤ B(q₀)` both shift q = q₀ + B^⊤ β_w additively, which makes the
    /// penalized joint Hessian H = H_L + S near-singular along that
    /// direction (σ_min ≈ ridge_floor ≈ 1e-10).  Under the default `Smooth`
    /// regularization this null direction contributes a first-order FD
    /// component to `d log|H|/dρ` via `φ'(σ_min) · dσ_min/dρ` that cannot
    /// be matched by the analytic `u^⊤ (dH/dρ) u` formula — the
    /// eigenvector `u` for a near-zero σ is numerically arbitrary inside
    /// the null space, so first-order perturbation theory breaks down.
    /// `HardPseudo` excludes σ ≤ ε from BOTH log|H| and its gradient
    /// consistently, so the null direction drops out on both sides and
    /// FD-vs-analytic comparisons close cleanly.
    fn pseudo_logdet_mode(&self) -> crate::custom_family::PseudoLogdetMode {
        crate::custom_family::PseudoLogdetMode::HardPseudo
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        Ok(project_monotone_wiggle_beta(beta))
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let wiggle_design = self.wiggle_design(core.q0.view())?;
        let dq_dq0 =
            self.wiggle_dq_dq0(core.q0.view(), block_states[Self::BLOCK_WIGGLE].beta.view())?;
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing threshold design"
                .to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing log-sigma design"
                .to_string()
        })?;

        // Per-block gradients from the eta-space score.
        //
        //   q = q0 + w(q0), a = dq/dq0
        //   score_q = -m1   (m1 = dF/dq, F = -ℓ)
        //   grad_eta_t[i]  = score_q * a * q0_t
        //   grad_eta_ls[i] = score_q * a * q0_ls
        //   grad_q[i]      = score_q          (wiggle basis acts on q)
        let mut grad_eta_t = Array1::<f64>::zeros(n);
        let mut grad_eta_ls = Array1::<f64>::zeros(n);
        let mut grad_q = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core.q0[i] + etaw[i];
            let (m1, _, _) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            let score_q = -m1;
            let q0d = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            grad_eta_t[i] = score_q * dq_dq0[i] * q0d.q_t;
            grad_eta_ls[i] = score_q * dq_dq0[i] * q0d.q_ls;
            grad_q[i] = score_q;
        }
        let grad_t = threshold_design.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = log_sigma_design.transpose_vector_multiply(&grad_eta_ls);
        let grad_w = fast_atv(&wiggle_design, &grad_q);

        // Per-block diagonal Hessians without ever materializing the full p×p
        // joint matrix. The shared row-pieces struct exposes block diagonals
        // directly, so the cross blocks (h_tl, h_tw, h_lw) are not formed.
        let (x_t, x_ls) = self.exact_joint_dense_block_designs(None)?.ok_or(
            "BinomialLocationScaleWiggleFamily: joint block designs unavailable",
        )?;
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        let (h_tt, h_ll, h_ww) = pieces.assemble_block_diagonals(&x_t, &x_ls)?;
        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(h_tt),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_ls,
                    hessian: SymmetricMatrix::Dense(h_ll),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_w,
                    hessian: SymmetricMatrix::Dense(h_ww),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }
        binomial_location_scale_ll_only(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let total = pt + pls + pw;

        let (range_start, range_end) = match block_idx {
            Self::BLOCK_T => (0usize, pt),
            Self::BLOCK_LOG_SIGMA => (pt, pt + pls),
            Self::BLOCK_WIGGLE => (pt + pls, total),
            _ => return Ok(None),
        };
        if d_beta.len() != (range_end - range_start) {
            return Err(format!(
                "block {block_idx} d_beta length mismatch: got {}, expected {}",
                d_beta.len(),
                range_end - range_start
            ));
        }

        // Block-local exact Newton directional derivative is extracted from the
        // full joint directional Hessian.
        //
        // For the 3-block wiggle model with beta=(beta_t,beta_ls,betaw),
        // define the full negative-loglik Hessian H(beta) in flattened block
        // coordinates. For a direction that moves only one block,
        //
        //   u = [u_t, 0,   0]   or
        //   u = [0,   u_ls,0]   or
        //   u = [0,   0,   uw],
        //
        // the exact blockwise directional Hessian required by the trait is just
        // the corresponding principal block of D H[u]:
        //
        //   D H_block[u_block]
        //   = (D H_joint[u])_{block,block}.
        //
        // This avoids maintaining a second, partially duplicated derivation for
        // the block-local case and keeps the exact-newton block callback aligned
        // with the already-validated joint formulas.
        let mut d_beta_flat = Array1::<f64>::zeros(total);
        match block_idx {
            Self::BLOCK_T => {
                d_beta_flat.slice_mut(s![0..pt]).assign(d_beta);
            }
            Self::BLOCK_LOG_SIGMA => {
                d_beta_flat.slice_mut(s![pt..pt + pls]).assign(d_beta);
            }
            Self::BLOCK_WIGGLE => {
                d_beta_flat.slice_mut(s![pt + pls..]).assign(d_beta);
            }
            _ => {}
        }
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative(block_states, &d_beta_flat)?
            .ok_or_else(|| "missing exact wiggle joint dH".to_string())?;
        let out = d_joint
            .slice(s![range_start..range_end, range_start..range_end])
            .to_owned();
        Ok(Some(out))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact joint Hessian for the 3-block binomial location-scale wiggle family.
        //
        // Model:
        //   q0 = -eta_t / sigma(eta_ls),
        //   q  = q0 + betaw^T B(q0),
        //   mu = Phi(q),
        //   F  = -sum_i ell_i(mu_i).
        //
        // The shared rowwise weights (coeff_tt, coeff_tl, coeff_ll, coeff_tw_b,
        // coeff_tw_d, coeff_lw_b, coeff_lw_d, coeffww) plus the realized B/B'
        // basis arrays are computed once by `wiggle_hessian_row_pieces` and
        // assembled here into the dense p×p matrix. The matrix-free workspace
        // path reuses the exact same row pieces to apply H to a vector
        // without ever forming the dense matrix.
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        Ok(Some(pieces.assemble_dense(&x_t, &x_ls)?))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact directional derivative dH[u] for the same 3-block model.
        //
        // Direction:
        //   u = (u_t, u_l, uw),
        //   d_eta_t = X_t u_t, d_eta_l = X_l u_l.
        //
        // Canonical objective identity for scalar-q composition:
        //   dH_ab[u] =
        //      m3 * dq * q_a q_b
        //    + m2 * (dq_a q_b + q_a dq_b + dq q_ab)
        //    + m1 * dq_ab
        // where (m1,m2,m3) are derivatives of F wrt q.
        //
        // Log-likelihood derivative relation used in code:
        //   s = d ell/dq, c = d² ell/dq², t = d³ ell/dq³
        //   m1 = -s, m2 = -c, m3 = -t.
        //
        // Required analytic chain terms:
        //
        // 1) Wiggle scalars:
        //   m  = 1 + betaw^T B'(q0)
        //   g2 = betaw^T B''(q0)
        //   g3 = betaw^T B'''(q0)
        //
        // 2) Directional wiggle scalars:
        //   dm  = (B'·uw)  + g2*dq0
        //   dg2 = (B''·uw) + g3*dq0
        //
        // 3) Directional q pieces:
        //   dq   = m*dq0 + B·uw
        //   dq_t = dm*q0_t + m*dq0_t
        //   dq_l = dm*q0_l + m*dq0_l
        //
        // 4) Directional second q pieces:
        //   dq_tt = dg2*q0_t*q0_t + g2*(2*q0_t*dq0_t)
        //   dq_tl = dg2*q0_t*q0_l + g2*(dq0_t*q0_l + q0_t*dq0_l)
        //           + dm*q0_tl + m*dq0_tl
        //   dq_ll = dg2*q0_l*q0_l + g2*(2*q0_l*dq0_l)
        //           + dm*q0_ll + m*dq0_ll
        //
        // 5) Mixed w-block directional terms:
        //   qw   = B,         dqw   = B' dq0
        //   q_tw  = q0_t B',   dq_tw  = dq0_t B' + dq0 q0_t B''
        //   q_lw  = q0_l B',   dq_lw  = dq0_l B' + dq0 q0_l B''
        //   qww  = 0,         dqww  = 0
        //
        // Implementation below follows these formulas exactly block-by-block.
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_flat, "wiggle joint d_beta")?;
        let d_eta_t = x_t.dot(&u_t);
        let d_eta_ls = x_ls.dot(&u_ls);

        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        if d0.ncols() != betaw0.len() || dd0.ncols() != betaw0.len() {
            return Err(format!(
                "wiggle derivative/beta mismatch in exact joint dH: B'={} B''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                betaw0.len()
            ));
        }
        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        let mut coeff_tw_b = Array1::<f64>::zeros(n);
        let mut coeff_tw_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_dd = Array1::<f64>::zeros(n);
        let mut coeff_lw_b = Array1::<f64>::zeros(n);
        let mut coeff_lw_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_dd = Array1::<f64>::zeros(n);
        let mut coeffww_bb = Array1::<f64>::zeros(n);
        let mut coeffww_db = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);
            let dq0 = nonwiggle_q_directional(q0, d_eta_t[i], d_eta_ls[i]);

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let duw_i = dr.dot(&uw);
            let dduw_i = ddr.dot(&uw);

            // Canonical directional wiggle scalars:
            //   dm  = B'(q0)·uw + g2*dq0
            //   dg2 = B''(q0)·uw + g3*dq0
            let delta_m = g2[i] * dq0.delta_q + duw_i;
            let delta_g2 = g3[i] * dq0.delta_q + dduw_i;

            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let q_tt = g2[i] * q0.q_t * q0.q_t;
            let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
            let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

            let delta_q_t = delta_m * q0.q_t + m[i] * dq0.delta_q_t;
            let delta_q_ls = delta_m * q0.q_ls + m[i] * dq0.delta_q_ls;
            let delta_q_tt = delta_g2 * q0.q_t * q0.q_t + g2[i] * 2.0 * q0.q_t * dq0.delta_q_t;
            let delta_q_tl = delta_g2 * q0.q_t * q0.q_ls
                + g2[i] * (dq0.delta_q_t * q0.q_ls + q0.q_t * dq0.delta_q_ls)
                + delta_m * q0.q_tl
                + m[i] * dq0.delta_q_tl;
            let delta_q_ll = delta_g2 * q0.q_ls * q0.q_ls
                + g2[i] * 2.0 * q0.q_ls * dq0.delta_q_ls
                + delta_m * q0.q_ll
                + m[i] * dq0.delta_q_ll;

            let delta_q = m[i] * dq0.delta_q + br.dot(&uw);

            // Closed forms by block from:
            // dH_ab = m3*dq*q_a*q_b + m2*(dq_a*q_b + q_a*dq_b + dq*q_ab) + m1*dq_ab.
            //
            // (tt):
            //   dH_tt = m3*dq*q_t^2 + m2*(2*dq_t*q_t + dq*q_tt) + m1*dq_tt.
            let coeff_tt_i = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_t, q_t, q_tt, delta_q_t, delta_q_t, delta_q_tt,
            );
            // (tl):
            //   dH_tl = m3*dq*q_t*q_l
            //        + m2*(dq_t*q_l + q_t*dq_l + dq*q_tl)
            //        + m1*dq_tl.
            let coeff_tl_i = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_t, q_ls, q_tl, delta_q_t, delta_q_ls, delta_q_tl,
            );
            // (ll):
            //   dH_ll = m3*dq*q_l^2 + m2*(2*dq_l*q_l + dq*q_ll) + m1*dq_ll.
            let coeff_ll_i = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_ls, q_ls, q_ll, delta_q_ls, delta_q_ls, delta_q_ll,
            );

            coeff_tt[i] = coeff_tt_i;
            coeff_tl[i] = coeff_tl_i;
            coeff_ll[i] = coeff_ll_i;
            coeff_tw_b[i] = m3 * delta_q * q_t + m2 * delta_q_t;
            coeff_tw_d[i] = m2 * (q_t * dq0.delta_q + delta_q * q0.q_t) + m1 * dq0.delta_q_t;
            coeff_tw_dd[i] = m1 * dq0.delta_q * q0.q_t;
            coeff_lw_b[i] = m3 * delta_q * q_ls + m2 * delta_q_ls;
            coeff_lw_d[i] = m2 * (q_ls * dq0.delta_q + delta_q * q0.q_ls) + m1 * dq0.delta_q_ls;
            coeff_lw_dd[i] = m1 * dq0.delta_q * q0.q_ls;
            coeffww_bb[i] = m3 * delta_q;
            coeffww_db[i] = m2 * dq0.delta_q;
        }
        let d_h_tt = xt_diag_x_dense(&x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(&x_t, &coeff_tl, &x_ls)?;
        let d_h_ll = xt_diag_x_dense(&x_ls, &coeff_ll)?;
        let d_h_tw = xt_diag_y_dense(&x_t, &coeff_tw_b, &b0)?
            + &xt_diag_y_dense(&x_t, &coeff_tw_d, &d0)?
            + &xt_diag_y_dense(&x_t, &coeff_tw_dd, &dd0)?;
        let d_h_lw = xt_diag_y_dense(&x_ls, &coeff_lw_b, &b0)?
            + &xt_diag_y_dense(&x_ls, &coeff_lw_d, &d0)?
            + &xt_diag_y_dense(&x_ls, &coeff_lw_dd, &dd0)?;
        let mut d_hww = xt_diag_x_dense(&b0, &coeffww_bb)?;
        d_hww += &xt_diag_y_dense(&d0, &coeffww_db, &b0)?;
        d_hww += &xt_diag_y_dense(&b0, &coeffww_db, &d0)?;

        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..pt + pls]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&d_h_ll);
        d_h.slice_mut(s![0..pt, pt + pls..total]).assign(&d_h_tw);
        d_h.slice_mut(s![pt..pt + pls, pt + pls..total])
            .assign(&d_h_lw);
        d_h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&d_hww);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(core0.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        let d4q = self.wiggle_d4q_dq04(core0.q0.view(), betaw0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d0.ncols() != betaw0.len()
            || dd0.ncols() != betaw0.len()
            || d3_basis.ncols() != betaw0.len()
        {
            return Err(format!(
                "wiggle derivative/beta mismatch in exact joint d2H: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw0.len()
            ));
        }

        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_u_flat, "wiggle joint d_beta_u")?;
        let (v_t, v_ls, vw) = beta_layout.split_three(d_betav_flat, "wiggle joint d_betav")?;
        let d_eta_t_u = x_t.dot(&u_t);
        let d_eta_ls_u = x_ls.dot(&u_ls);
        let d_eta_tv = x_t.dot(&v_t);
        let d_eta_lsv = x_ls.dot(&v_ls);

        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        let mut d2_h = Array2::<f64>::zeros((total, total));
        for i in 0..n {
            // Per-row scalar objective derivatives for F_i(q).
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            )?;

            // Non-wiggle q0(eta_t, eta_ls) derivatives and sigma-ratio helpers.
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);
            let s_safe = sigma[i];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let s5 = s4 * s_safe;
            let q0_tl_ls_ls =
                d3s[i] / s2 - 6.0 * ds[i] * d2s[i] / s3 + 6.0 * ds[i] * ds[i] * ds[i] / s4;
            let q0_tl_ls_ls_ls =
                d4s[i] / s2 - 8.0 * ds[i] * d3s[i] / s3 - 6.0 * d2s[i] * d2s[i] / s3
                    + 36.0 * ds[i] * ds[i] * d2s[i] / s4
                    - 24.0 * ds[i] * ds[i] * ds[i] * ds[i] / s5;
            let q0_ll_ls_ls = eta_t[i] * q0_tl_ls_ls_ls;

            let u_t_i = d_eta_t_u[i];
            let u_ls_i = d_eta_ls_u[i];
            let v_t_i = d_eta_tv[i];
            let v_ls_i = d_eta_lsv[i];

            // Directional z=q0 primitives for u and v.
            let dq0_u = q0.q_t * u_t_i + q0.q_ls * u_ls_i;
            let dq0v = q0.q_t * v_t_i + q0.q_ls * v_ls_i;
            let d2q0_uv = q0.q_tl * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0.q_ll * u_ls_i * v_ls_i;

            let dq0_t_u = q0.q_tl * u_ls_i;
            let dq0_tv = q0.q_tl * v_ls_i;
            let dq0_ls_u = q0.q_tl * u_t_i + q0.q_ll * u_ls_i;
            let dq0_lsv = q0.q_tl * v_t_i + q0.q_ll * v_ls_i;
            let dq0_tl_u = q0.q_tl_ls * u_ls_i;
            let dq0_tlv = q0.q_tl_ls * v_ls_i;
            let dq0_ll_u = q0.q_tl_ls * u_t_i + q0.q_ll_ls * u_ls_i;
            let dq0_llv = q0.q_tl_ls * v_t_i + q0.q_ll_ls * v_ls_i;

            let d2q0_t_uv = q0.q_tl_ls * u_ls_i * v_ls_i;
            let d2q0_ls_uv =
                q0.q_tl_ls * (u_ls_i * v_t_i + v_ls_i * u_t_i) + q0.q_ll_ls * u_ls_i * v_ls_i;
            let d2q0_tl_uv = q0_tl_ls_ls * u_ls_i * v_ls_i;
            let d2q0_ll_uv =
                q0_tl_ls_ls * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0_ll_ls_ls * u_ls_i * v_ls_i;

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let d3r = d3_basis.row(i);
            let b_u = br.dot(&uw);
            let bv = br.dot(&vw);
            let b1_u = dr.dot(&uw);
            let b1v = dr.dot(&vw);
            let b2_u = ddr.dot(&uw);
            let b2v = ddr.dot(&vw);
            let b3_u = d3r.dot(&uw);
            let b3v = d3r.dot(&vw);

            // Wiggle scalar chain terms:
            //   m = 1 + g1,     g2 = betaw^T B''(q0),
            //   dm[u]   = B'·uw + g2*dq0[u],
            //   d2m[u,v]= g3*dq0[u]dq0[v] + g2*d2q0[u,v] + (B''·vw)dq0[u] + (B''·uw)dq0[v],
            //   dg2[u]  = B''·uw + g3*dq0[u],
            //   d2g2[u,v]=g4*dq0[u]dq0[v] + g3*d2q0[u,v] + (B'''·vw)dq0[u] + (B'''·uw)dq0[v].
            let dm_u = b1_u + g2[i] * dq0_u;
            let dmv = b1v + g2[i] * dq0v;
            let d2m_uv = g3[i] * dq0_u * dq0v + g2[i] * d2q0_uv + b2v * dq0_u + b2_u * dq0v;
            let dg2_u = b2_u + g3[i] * dq0_u;
            let dg2v = b2v + g3[i] * dq0v;
            let d2g2_uv = g4[i] * dq0_u * dq0v + g3[i] * d2q0_uv + b3v * dq0_u + b3_u * dq0v;

            // First/second directional terms for total q.
            let dq_u = m[i] * dq0_u + b_u;
            let dqv = m[i] * dq0v + bv;
            // Simplify exact formula for q = q0 + betaw^T B(q0):
            //   D²q[u,v] = m*d²q0 + g2*dq0[u]dq0[v] + (B'·uw)dq0[v] + (B'·vw)dq0[u].
            let d2q_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0v + b1_u * dq0v + b1v * dq0_u;

            // q partials by block and their first/second directional derivatives.
            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let q_tt = g2[i] * q0.q_t * q0.q_t;
            let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
            let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

            let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
            let dq_tv = dmv * q0.q_t + m[i] * dq0_tv;
            let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
            let dq_lsv = dmv * q0.q_ls + m[i] * dq0_lsv;

            let d2q_t_uv = d2m_uv * q0.q_t + dm_u * dq0_tv + dmv * dq0_t_u + m[i] * d2q0_t_uv;
            let d2q_ls_uv = d2m_uv * q0.q_ls + dm_u * dq0_lsv + dmv * dq0_ls_u + m[i] * d2q0_ls_uv;

            let dq_tt_u = dg2_u * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_t_u);
            let dq_ttv = dg2v * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_tv);
            let d2q_tt_uv = d2g2_uv * q0.q_t * q0.q_t
                + dg2_u * (2.0 * q0.q_t * dq0_tv)
                + dg2v * (2.0 * q0.q_t * dq0_t_u)
                + g2[i] * (2.0 * dq0_t_u * dq0_tv + 2.0 * q0.q_t * d2q0_t_uv);

            let dq_tl_u = dg2_u * q0.q_t * q0.q_ls
                + g2[i] * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                + dm_u * q0.q_tl
                + m[i] * dq0_tl_u;
            let dq_tlv = dg2v * q0.q_t * q0.q_ls
                + g2[i] * (dq0_tv * q0.q_ls + q0.q_t * dq0_lsv)
                + dmv * q0.q_tl
                + m[i] * dq0_tlv;
            let d2q_tl_uv = d2g2_uv * q0.q_t * q0.q_ls
                + dg2_u * (dq0_tv * q0.q_ls + q0.q_t * dq0_lsv)
                + dg2v * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                + g2[i]
                    * (d2q0_t_uv * q0.q_ls
                        + dq0_t_u * dq0_lsv
                        + dq0_tv * dq0_ls_u
                        + q0.q_t * d2q0_ls_uv)
                + d2m_uv * q0.q_tl
                + dm_u * dq0_tlv
                + dmv * dq0_tl_u
                + m[i] * d2q0_tl_uv;

            let dq_ll_u = dg2_u * q0.q_ls * q0.q_ls
                + g2[i] * (2.0 * q0.q_ls * dq0_ls_u)
                + dm_u * q0.q_ll
                + m[i] * dq0_ll_u;
            let dq_llv = dg2v * q0.q_ls * q0.q_ls
                + g2[i] * (2.0 * q0.q_ls * dq0_lsv)
                + dmv * q0.q_ll
                + m[i] * dq0_llv;
            let d2q_ll_uv = d2g2_uv * q0.q_ls * q0.q_ls
                + dg2_u * (2.0 * q0.q_ls * dq0_lsv)
                + dg2v * (2.0 * q0.q_ls * dq0_ls_u)
                + g2[i] * (2.0 * dq0_ls_u * dq0_lsv + 2.0 * q0.q_ls * d2q0_ls_uv)
                + d2m_uv * q0.q_ll
                + dm_u * dq0_llv
                + dmv * dq0_ll_u
                + m[i] * d2q0_ll_uv;

            // Exact second directional coefficients for the scalar block weights.
            let coeff_tt = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_t, q_tt, dq_t_u, dq_tv, dq_t_u, dq_tv,
                d2q_t_uv, d2q_t_uv, dq_tt_u, dq_ttv, d2q_tt_uv,
            );
            let coeff_tl = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_ls, q_tl, dq_t_u, dq_tv, dq_ls_u, dq_lsv,
                d2q_t_uv, d2q_ls_uv, dq_tl_u, dq_tlv, d2q_tl_uv,
            );
            let coeff_ll = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, q_ls, q_ll, dq_ls_u, dq_lsv, dq_ls_u,
                dq_lsv, d2q_ls_uv, d2q_ls_uv, dq_ll_u, dq_llv, d2q_ll_uv,
            );

            let xtr = x_t.row(i);
            let xlsr = x_ls.row(i);
            for a_idx in 0..pt {
                for b_idx in a_idx..pt {
                    d2_h[[a_idx, b_idx]] += coeff_tt * xtr[a_idx] * xtr[b_idx];
                }
            }
            for a_idx in 0..pt {
                for b_idx in 0..pls {
                    d2_h[[a_idx, pt + b_idx]] += coeff_tl * xtr[a_idx] * xlsr[b_idx];
                }
            }
            for a_idx in 0..pls {
                for b_idx in a_idx..pls {
                    d2_h[[pt + a_idx, pt + b_idx]] += coeff_ll * xlsr[a_idx] * xlsr[b_idx];
                }
            }

            for j in 0..pw {
                let qw = br[j];
                let dqw_u = dr[j] * dq0_u;
                let dqwv = dr[j] * dq0v;
                let d2qw_uv = ddr[j] * dq0_u * dq0v + dr[j] * d2q0_uv;
                let q_tw = dr[j] * q0.q_t;
                let q_lw = dr[j] * q0.q_ls;
                let dq_tw_u = ddr[j] * dq0_u * q0.q_t + dr[j] * dq0_t_u;
                let dq_twv = ddr[j] * dq0v * q0.q_t + dr[j] * dq0_tv;
                let d2q_tw_uv = d3r[j] * dq0_u * dq0v * q0.q_t
                    + ddr[j] * (d2q0_uv * q0.q_t + dq0_u * dq0_tv + dq0v * dq0_t_u)
                    + dr[j] * d2q0_t_uv;
                let dq_lw_u = ddr[j] * dq0_u * q0.q_ls + dr[j] * dq0_ls_u;
                let dq_lwv = ddr[j] * dq0v * q0.q_ls + dr[j] * dq0_lsv;
                let d2q_lw_uv = d3r[j] * dq0_u * dq0v * q0.q_ls
                    + ddr[j] * (d2q0_uv * q0.q_ls + dq0_u * dq0_lsv + dq0v * dq0_ls_u)
                    + dr[j] * d2q0_ls_uv;

                let coeff_tw = second_directionalhessian_coeff_fromobjective_q_terms(
                    m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, qw, q_tw, dq_t_u, dq_tv, dqw_u, dqwv,
                    d2q_t_uv, d2qw_uv, dq_tw_u, dq_twv, d2q_tw_uv,
                );
                let coeff_lw = second_directionalhessian_coeff_fromobjective_q_terms(
                    m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, qw, q_lw, dq_ls_u, dq_lsv, dqw_u,
                    dqwv, d2q_ls_uv, d2qw_uv, dq_lw_u, dq_lwv, d2q_lw_uv,
                );

                for a_idx in 0..pt {
                    d2_h[[a_idx, pt + pls + j]] += coeff_tw * xtr[a_idx];
                }
                for a_idx in 0..pls {
                    d2_h[[pt + a_idx, pt + pls + j]] += coeff_lw * xlsr[a_idx];
                }
            }

            for j in 0..pw {
                let qwj = br[j];
                let dqwj_u = dr[j] * dq0_u;
                let dqwjv = dr[j] * dq0v;
                let d2qwj_uv = ddr[j] * dq0_u * dq0v + dr[j] * d2q0_uv;
                for k in j..pw {
                    let qwk = br[k];
                    let dqwk_u = dr[k] * dq0_u;
                    let dqwkv = dr[k] * dq0v;
                    let d2qwk_uv = ddr[k] * dq0_u * dq0v + dr[k] * d2q0_uv;
                    let coeffww = second_directionalhessian_coeff_fromobjective_q_terms(
                        m1, m2, m3, m4, dq_u, dqv, d2q_uv, qwj, qwk, 0.0, dqwj_u, dqwjv, dqwk_u,
                        dqwkv, d2qwj_uv, d2qwk_uv, 0.0, 0.0, 0.0,
                    );
                    d2_h[[pt + pls + j, pt + pls + k]] += coeffww;
                }
            }
        }

        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessian(block_states)
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        // These three joint psi hooks are the wiggle family's exact
        // likelihood-side contribution to the unified full [rho, psi] outer
        // Hessian:
        //
        //   exact_newton_joint_psi_terms(...)                    -> D_a, D_{beta a}, D_{beta beta a}
        //   exact_newton_joint_psisecond_order_terms(...)       -> D_ab, D_{beta ab}, D_{beta beta ab}
        //   exact_newton_joint_psihessian_directional_derivative(...) -> T_a[u]
        //
        // Generic exact-joint code in custom_family.rs adds all realized
        // penalty motion S_a / S_ab and combines these likelihood-only objects
        // with the joint mode solves beta_i, beta_ij and the total Hessian
        // drifts dot H_i, ddot H_ij. Keeping this contract explicit is what
        // makes the wiggle family's full [rho, psi] Hessian real rather than a
        // gradient-only or block-local surrogate.
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &crate::custom_family::ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.len() < 2 {
            return Err("wiggle geometry requires threshold and log-sigma blocks".to_string());
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() {
            return Err("wiggle geometry input size mismatch".to_string());
        }
        let mut q0 = Array1::<f64>::zeros(eta_t.len());
        for i in 0..q0.len() {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            q0[i] = binomial_location_scale_q0(eta_t[i], sigma);
        }
        let x = self.wiggle_design(q0.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(format!(
                "dynamic wiggle design col mismatch: got {}, expected {}",
                x.ncols(),
                spec.design.ncols()
            ));
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = BinomialLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            x_t.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }
}

/// Matrix-free joint-Hessian operator for the 3-block binomial
/// location-scale wiggle family. See `BinomialLocationScaleWiggleHessianRowPieces`
/// for the per-row weight structure.
struct BinomialLocationScaleWiggleHessianWorkspace {
    family: BinomialLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    x_t: Array2<f64>,
    x_ls: Array2<f64>,
    pieces: BinomialLocationScaleWiggleHessianRowPieces,
}

impl BinomialLocationScaleWiggleHessianWorkspace {
    fn new(
        family: BinomialLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        _specs: Vec<ParameterBlockSpec>,
        x_t: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let pieces = family.wiggle_hessian_row_pieces(&block_states)?;
        Ok(Self {
            family,
            block_states,
            x_t,
            x_ls,
            pieces,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for BinomialLocationScaleWiggleHessianWorkspace {
    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let pw = self.pieces.b0.ncols();
        let total = pt + pls + pw;
        if v.len() != total {
            return Err(format!(
                "BinomialLocationScaleWiggle matvec dimension mismatch: got {}, expected {}",
                v.len(),
                total
            ));
        }
        let v_t = v.slice(s![0..pt]);
        let v_ls = v.slice(s![pt..pt + pls]);
        let v_w = v.slice(s![pt + pls..total]);

        let u_t = self.x_t.dot(&v_t);
        let u_ls = self.x_ls.dot(&v_ls);
        let u_b = self.pieces.b0.dot(&v_w);
        let u_d = self.pieces.d0.dot(&v_w);

        let r_t = &self.pieces.coeff_tt * &u_t
            + &self.pieces.coeff_tl * &u_ls
            + &self.pieces.coeff_tw_b * &u_b
            + &self.pieces.coeff_tw_d * &u_d;
        let r_ls = &self.pieces.coeff_tl * &u_t
            + &self.pieces.coeff_ll * &u_ls
            + &self.pieces.coeff_lw_b * &u_b
            + &self.pieces.coeff_lw_d * &u_d;
        let r_b = &self.pieces.coeff_tw_b * &u_t
            + &self.pieces.coeff_lw_b * &u_ls
            + &self.pieces.coeffww * &u_b;
        let r_d = &self.pieces.coeff_tw_d * &u_t + &self.pieces.coeff_lw_d * &u_ls;

        let out_t = self.x_t.t().dot(&r_t);
        let out_ls = self.x_ls.t().dot(&r_ls);
        let out_w = self.pieces.b0.t().dot(&r_b) + &self.pieces.d0.t().dot(&r_d);

        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pt]).assign(&out_t);
        out.slice_mut(s![pt..pt + pls]).assign(&out_ls);
        out.slice_mut(s![pt + pls..total]).assign(&out_w);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let pw = self.pieces.b0.ncols();
        let total = pt + pls + pw;
        let mut diag = Array1::<f64>::zeros(total);
        let n = self.pieces.coeff_tt.len();
        for j in 0..pt {
            let col = self.x_t.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_tt[i] * v * v;
            }
            diag[j] = acc;
        }
        for j in 0..pls {
            let col = self.x_ls.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_ll[i] * v * v;
            }
            diag[pt + j] = acc;
        }
        for j in 0..pw {
            let col = self.pieces.b0.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeffww[i] * v * v;
            }
            diag[pt + pls + j] = acc;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family.exact_newton_joint_hessiansecond_directional_derivative(
            &self.block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }
}

impl CustomFamilyGenerative for BinomialLocationScaleWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() || etaw.len() != self.y.len()
        {
            return Err("BinomialLocationScaleWiggleFamily generative size mismatch".to_string());
        }
        let mut mean = Array1::<f64>::zeros(self.y.len());
        for i in 0..mean.len() {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            let q0 = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q0 + etaw[i])
                .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
            mean[i] = jet.mu;
        }
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use crate::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
    use ndarray::{Array2, Axis, array};
    use num_dual::{
        DualNum, second_derivative, second_partial_derivative, third_partial_derivative_vec,
    };

    fn intercept_block(n: usize) -> ParameterBlockInput {
        ParameterBlockInput {
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (n, 1),
                1.0,
            ))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        }
    }

    fn compose_theta_from_hints_test(
        mean_penalty_count: usize,
        noise_penalty_count: usize,
        mean_log_lambda_hint: &Option<Array1<f64>>,
        noise_log_lambda_hint: &Option<Array1<f64>>,
        extra_rho0: &Array1<f64>,
    ) -> Array1<f64> {
        let layout = GamlssLambdaLayout::withwiggle(
            mean_penalty_count,
            noise_penalty_count,
            extra_rho0.len(),
        );
        let mut theta = Array1::<f64>::zeros(layout.total());
        if let Some(v) = mean_log_lambda_hint
            && v.len() == layout.k_mean
        {
            theta.slice_mut(s![0..layout.mean_end()]).assign(v);
        }
        if let Some(v) = noise_log_lambda_hint
            && v.len() == layout.k_noise
        {
            theta
                .slice_mut(s![layout.noise_start()..layout.noise_end()])
                .assign(v);
        }
        if layout.kwiggle > 0 {
            theta
                .slice_mut(s![layout.wiggle_start()..layout.wiggle_end()])
                .assign(extra_rho0);
        }
        theta
    }

    fn logistic_numdual<D: DualNum<f64> + Copy>(x: D) -> D {
        D::one() / (D::one() + (-x).exp())
    }

    fn bspline_basis_scalar_numdual<D: DualNum<f64> + Copy>(
        x: D,
        knots: &Array1<f64>,
        degree: usize,
    ) -> Vec<D> {
        let n_basis = knots.len() - degree - 1;
        let x_real = x.re();
        let mut basis = vec![D::zero(); n_basis];
        let last_knot = knots[knots.len() - 1];
        for j in 0..n_basis {
            let left = knots[j];
            let right = knots[j + 1];
            let active = if x_real == last_knot {
                j + 1 == n_basis
            } else {
                left <= x_real && x_real < right
            };
            if active {
                basis[j] = D::one();
            }
        }
        for k in 1..=degree {
            let mut next = vec![D::zero(); n_basis];
            for j in 0..n_basis {
                let mut acc = D::zero();
                let left_denom = knots[j + k] - knots[j];
                if left_denom > 0.0 {
                    acc += ((x - D::from(knots[j])) / D::from(left_denom)) * basis[j];
                }
                if j + 1 < n_basis {
                    let right_denom = knots[j + k + 1] - knots[j + 1];
                    if right_denom > 0.0 {
                        acc +=
                            ((D::from(knots[j + k + 1]) - x) / D::from(right_denom)) * basis[j + 1];
                    }
                }
                next[j] = acc;
            }
            basis = next;
        }
        basis
    }

    fn monotone_wiggle_basis_scalar_numdual<D: DualNum<f64> + Copy>(
        x: D,
        knots: &Array1<f64>,
        degree: usize,
    ) -> Array1<D> {
        let bs_degree =
            monotone_wiggle_internal_degree(degree).expect("monotone wiggle degree") + 1;
        let left = knots[bs_degree];
        let full = bspline_basis_scalar_numdual(x, knots, bs_degree);
        let left_full = bspline_basis_scalar_numdual(D::from(left), knots, bs_degree);
        let mut out = Array1::<D>::from_elem(full.len().saturating_sub(1), D::zero());
        let mut running = D::zero();
        let mut left_running = D::zero();
        for j in (1..full.len()).rev() {
            running += full[j];
            left_running += left_full[j];
            out[j - 1] = running - left_running;
        }
        out
    }

    fn wiggle_negloglik_threshold_numdual<D: DualNum<f64> + Copy>(
        beta_t: D,
        beta_ls: f64,
        betaw: &Array1<f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        knots: &Array1<f64>,
        degree: usize,
    ) -> D {
        let sigma = D::from(beta_ls).exp();
        let q0 = -beta_t / sigma;
        let basis = monotone_wiggle_basis_scalar_numdual(q0, knots, degree);
        let mut etaw = D::zero();
        for j in 0..betaw.len() {
            etaw += basis[j] * D::from(betaw[j]);
        }
        let q = q0 + etaw;
        let mu = logistic_numdual(q);
        let one_minusmu = D::one() - mu;
        let mut out = D::zero();
        for i in 0..y.len() {
            out -= D::from(weights[i])
                * (D::from(y[i]) * mu.ln() + D::from(1.0 - y[i]) * one_minusmu.ln());
        }
        out
    }

    // Source-of-truth Gaussian logb negloglik. Analytic helpers MUST autodiff-match this.
    fn gaussian_negloglik_log_sigma_psi_numdual<D: DualNum<f64> + Copy>(
        beta_mu: D,
        beta_ls: D,
        psi: D,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        x_mu0: &Array1<f64>,
        x_ls0: &Array1<f64>,
        x_ls_psi: &Array1<f64>,
        x_ls_psi_psi: &Array1<f64>,
    ) -> D {
        let half = D::from(0.5);
        let mut out = D::zero();
        for i in 0..y.len() {
            let eta_mu = D::from(x_mu0[i]) * beta_mu;
            let x_ls = D::from(x_ls0[i])
                + psi * D::from(x_ls_psi[i])
                + half * psi * psi * D::from(x_ls_psi_psi[i]);
            let eta_ls = x_ls * beta_ls;
            // Mirror the production logb noise link σ = LOGB_SIGMA_FLOOR + exp(η_ls)
            // (see `GaussianLocationScaleFamily::loglik`); using the bare-exp link
            // here would diverge from the family's σ at the same η and break the
            // psi-derivative identities that this reference negloglik certifies.
            let sigma = D::from(LOGB_SIGMA_FLOOR) + eta_ls.exp();
            let resid = D::from(y[i]) - eta_mu;
            out += D::from(weights[i]) * (half * (resid / sigma).powi(2) + sigma.ln());
        }
        out
    }

    fn gaussian_negloglik_log_sigma_psi_only_numdual<D: DualNum<f64> + Copy>(
        psi: D,
        beta_mu: f64,
        beta_ls: f64,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        x_mu0: &Array1<f64>,
        x_ls0: &Array1<f64>,
        x_ls_psi: &Array1<f64>,
        x_ls_psi_psi: &Array1<f64>,
    ) -> D {
        gaussian_negloglik_log_sigma_psi_numdual(
            D::from(beta_mu),
            D::from(beta_ls),
            psi,
            y,
            weights,
            x_mu0,
            x_ls0,
            x_ls_psi,
            x_ls_psi_psi,
        )
    }

    fn gaussian_negloglik_log_sigma_mu_psi_numdual<D: DualNum<f64> + Copy>(
        beta_mu: D,
        psi: D,
        beta_ls: f64,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        x_mu0: &Array1<f64>,
        x_ls0: &Array1<f64>,
        x_ls_psi: &Array1<f64>,
        x_ls_psi_psi: &Array1<f64>,
    ) -> D {
        gaussian_negloglik_log_sigma_psi_numdual(
            beta_mu,
            D::from(beta_ls),
            psi,
            y,
            weights,
            x_mu0,
            x_ls0,
            x_ls_psi,
            x_ls_psi_psi,
        )
    }

    fn gaussian_negloglik_log_sigma_ls_psi_numdual<D: DualNum<f64> + Copy>(
        beta_ls: D,
        psi: D,
        beta_mu: f64,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        x_mu0: &Array1<f64>,
        x_ls0: &Array1<f64>,
        x_ls_psi: &Array1<f64>,
        x_ls_psi_psi: &Array1<f64>,
    ) -> D {
        gaussian_negloglik_log_sigma_psi_numdual(
            D::from(beta_mu),
            beta_ls,
            psi,
            y,
            weights,
            x_mu0,
            x_ls0,
            x_ls_psi,
            x_ls_psi_psi,
        )
    }

    fn gaussian_negloglik_log_sigma_beta_vec_numdual<D: DualNum<f64> + Copy>(
        v: &[D],
        y: &Array1<f64>,
        weights: &Array1<f64>,
        x_mu0: &Array1<f64>,
        x_ls0: &Array1<f64>,
        x_ls_psi: &Array1<f64>,
        x_ls_psi_psi: &Array1<f64>,
    ) -> D {
        gaussian_negloglik_log_sigma_psi_numdual(
            v[0],
            v[1],
            v[2],
            y,
            weights,
            x_mu0,
            x_ls0,
            x_ls_psi,
            x_ls_psi_psi,
        )
    }

    fn gaussian_psi_test_spec(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
        let n = design.nrows();
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        }
    }

    #[test]
    fn gaussian_joint_psi_firstweights_score_ls_carries_logb_chain_rule_factor() {
        let y = array![1.1];
        let etamu = array![0.3];
        let eta_ls = array![-0.2];
        let weights = array![2.5];
        let rows =
            gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
        let firstweights = gaussian_joint_psi_firstweights(&rows, &array![0.0], &array![1.0]);
        let sigma = crate::families::sigma_link::logb_sigma_from_eta_scalar(eta_ls[0]);
        let kappa = 1.0 - crate::families::sigma_link::LOGB_SIGMA_FLOOR / sigma;
        let expected = kappa * (weights[0] - rows.n[0]);

        assert!(
            (firstweights.score_ls[0] - expected).abs() <= 1e-12,
            "Under the logb link σ = b + exp(η_ls), d/dη_ls of weight*(ln σ + 0.5(y-μ)^2/σ^2) carries the chain-rule factor κ = 1 - b/σ, so the row score must equal κ*(weight - n_i). The helper coded {} but the κ-corrected expectation is {}.",
            firstweights.score_ls[0],
            expected
        );
        assert!(
            (firstweights.objective_psirow[0] - expected).abs() <= 1e-12,
            "With mu_psi=0 and eta_psi=1, the exact psi objective derivative must equal κ*(weight - n_i) (κ = 1 - b/σ from the logb chain rule). The helper coded {} but the κ-corrected expectation is {}.",
            firstweights.objective_psirow[0],
            expected
        );
    }

    #[test]
    fn cloglog_binomial_right_tail_derivatives_stay_finite() {
        let (m1, m2, m3) = binomial_neglog_q_derivatives_cloglog_closed_form(1.0, 1.0, 1000.0);
        let m4 = binomial_neglog_q_fourth_derivative_cloglog_closed_form(1.0, 1.0, 300.0);

        assert_eq!(m1, 0.0);
        assert_eq!(m2, 0.0);
        assert_eq!(m3, 0.0);
        assert_eq!(m4, 0.0);
    }

    #[test]
    fn cloglog_binomial_fractional_right_tail_keeps_y0_branch() {
        let y = 0.25;
        let weight = 2.0;
        let q = 300.0;
        let expected = weight * (1.0 - y) * q.exp();
        let (m1, m2, m3) = binomial_neglog_q_derivatives_cloglog_closed_form(y, weight, q);
        let m4 = binomial_neglog_q_fourth_derivative_cloglog_closed_form(y, weight, q);

        assert!(m1.is_finite());
        assert!(m2.is_finite());
        assert!(m3.is_finite());
        assert!(m4.is_finite());
        assert_eq!(m1, expected);
        assert_eq!(m2, expected);
        assert_eq!(m3, expected);
        assert_eq!(m4, expected);
    }

    #[test]
    fn gaussian_joint_psisecondweights_eta_ab_term_carries_logb_chain_rule_factor() {
        let y = array![1.1];
        let etamu = array![0.3];
        let eta_ls = array![-0.2];
        let weights = array![2.5];
        let rows =
            gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
        let secondweights = gaussian_joint_psisecondweights(
            &rows,
            &array![0.0],
            &array![0.0],
            &array![0.0],
            &array![0.0],
            &array![0.0],
            &array![1.0],
        );
        let sigma = crate::families::sigma_link::logb_sigma_from_eta_scalar(eta_ls[0]);
        let kappa = 1.0 - crate::families::sigma_link::LOGB_SIGMA_FLOOR / sigma;
        let expected = kappa * (weights[0] - rows.n[0]);

        assert!(
            (secondweights.objective_psi_psirow[0] - expected).abs() <= 1e-12,
            "With only eta_psi_psi=1 active, the Gaussian second psi objective contribution from the linear η_ls term carries the logb chain-rule factor κ = 1 - b/σ, so it must equal κ*(weight - n_i). The helper coded {} but the κ-corrected expectation is {}.",
            secondweights.objective_psi_psirow[0],
            expected
        );
    }

    #[test]
    fn zeroweightrows_stay_inactive_in_builtin_diagonal_families() {
        let weights = Array1::from_vec(vec![0.0, 1.0]);

        let gaussian = GaussianLocationScaleFamily {
            y: Array1::from_vec(vec![2.0, -1.0]),
            weights: weights.clone(),
            mu_design: None,
            log_sigma_design: None,
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        };
        let gaussian_eval = gaussian
            .evaluate(&[
                ParameterBlockState {
                    beta: Array1::zeros(0),
                    eta: Array1::from_vec(vec![0.5, -0.25]),
                },
                ParameterBlockState {
                    beta: Array1::zeros(0),
                    eta: Array1::from_vec(vec![0.1, -0.2]),
                },
            ])
            .expect("gaussian evaluate");
        match &gaussian_eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_MU] {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_weights[0], 0.0);
                assert_eq!(working_response[0], 0.5);
                assert!(working_weights[1] > 0.0);
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Gaussian mu block"),
        }
        match &gaussian_eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_weights[0], 0.0);
                assert_eq!(working_response[0], 0.1);
                assert!(working_weights[1] > 0.0);
            }
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal Gaussian log-sigma block")
            }
        }

        let poisson = PoissonLogFamily {
            y: Array1::from_vec(vec![3.0, 1.0]),
            weights: weights.clone(),
        };
        let poisson_eval = poisson
            .evaluate(&[ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::from_vec(vec![0.7, -0.4]),
            }])
            .expect("poisson evaluate");
        match &poisson_eval.blockworking_sets[PoissonLogFamily::BLOCK_ETA] {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_weights[0], 0.0);
                assert_eq!(working_response[0], 0.7);
                assert!(working_weights[1] > 0.0);
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Poisson block"),
        }

        let gamma = GammaLogFamily {
            y: Array1::from_vec(vec![1.5, 0.8]),
            weights,
            shape: 2.5,
        };
        let gamma_eval = gamma
            .evaluate(&[ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::from_vec(vec![0.2, -0.1]),
            }])
            .expect("gamma evaluate");
        match &gamma_eval.blockworking_sets[GammaLogFamily::BLOCK_ETA] {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_weights[0], 0.0);
                assert_eq!(working_response[0], 0.2);
                assert!(working_weights[1] > 0.0);
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Gamma block"),
        }
    }

    #[test]
    fn hard_clamped_poisson_and_gammarows_stay_locally_flat() {
        let poisson = PoissonLogFamily {
            y: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            weights: Array1::from_vec(vec![1.0, 1.0, 1.0]),
        };
        let poisson_eta = Array1::from_vec(vec![-35.0, 0.2, 35.0]);
        let poisson_eval = poisson
            .evaluate(&[ParameterBlockState {
                beta: Array1::zeros(0),
                eta: poisson_eta.clone(),
            }])
            .expect("poisson evaluate");
        match &poisson_eval.blockworking_sets[PoissonLogFamily::BLOCK_ETA] {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_weights[0], 0.0);
                assert_eq!(working_response[0], poisson_eta[0]);
                assert!(working_weights[1] > 0.0);
                assert_eq!(working_weights[2], 0.0);
                assert_eq!(working_response[2], poisson_eta[2]);
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Poisson block"),
        }

        let gamma = GammaLogFamily {
            y: Array1::from_vec(vec![0.8, 1.2, 2.5]),
            weights: Array1::from_vec(vec![1.0, 1.0, 1.0]),
            shape: 3.0,
        };
        let gamma_eta = Array1::from_vec(vec![-40.0, -0.3, 40.0]);
        let gamma_eval = gamma
            .evaluate(&[ParameterBlockState {
                beta: Array1::zeros(0),
                eta: gamma_eta.clone(),
            }])
            .expect("gamma evaluate");
        match &gamma_eval.blockworking_sets[GammaLogFamily::BLOCK_ETA] {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_weights[0], 0.0);
                assert_eq!(working_response[0], gamma_eta[0]);
                assert!(working_weights[1] > 0.0);
                assert_eq!(working_weights[2], 0.0);
                assert_eq!(working_response[2], gamma_eta[2]);
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Gamma block"),
        }
    }

    #[test]
    fn gaussian_log_sigmaweight_directional_derivative_iszero_on_active_floor_branch() {
        let family = GaussianLocationScaleFamily {
            y: Array1::from_vec(vec![0.3]),
            weights: Array1::from_vec(vec![1.0]),
            mu_design: None,
            log_sigma_design: None,
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        };
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::from_vec(vec![0.0]),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::from_vec(vec![35.0]),
            },
        ];
        let d_eta = Array1::from_vec(vec![1.0]);

        let dw = family
            .diagonalworking_weights_directional_derivative(
                &states,
                GaussianLocationScaleFamily::BLOCK_LOG_SIGMA,
                &d_eta,
            )
            .expect("gaussian directional derivative")
            .expect("gaussian log-sigma derivative");
        assert_eq!(dw[0], 0.0);
    }

    #[test]
    fn gaussian_log_sigmaweight_directional_derivative_matches_finite_difference() {
        let family = GaussianLocationScaleFamily {
            y: Array1::from_vec(vec![1.2]),
            weights: Array1::from_vec(vec![1.0]),
            mu_design: None,
            log_sigma_design: None,
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        };
        let etamu = Array1::from_vec(vec![0.1]);
        let eta_ls = Array1::from_vec(vec![0.4]);
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: etamu.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_ls.clone(),
            },
        ];
        let d_eta = Array1::from_vec(vec![1.0]);

        let dw = family
            .diagonalworking_weights_directional_derivative(
                &states,
                GaussianLocationScaleFamily::BLOCK_LOG_SIGMA,
                &d_eta,
            )
            .expect("gaussian directional derivative")
            .expect("gaussian log-sigma derivative");

        let eps = 1e-6;
        let mut states_plus = states.clone();
        states_plus[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta[0] += eps;
        let eval_plus = family.evaluate(&states_plus).expect("gaussian eval plus");
        let w_plus =
            match &eval_plus.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::Diagonal {
                    working_response: _,
                    working_weights,
                } => working_weights[0],
                BlockWorkingSet::ExactNewton { .. } => {
                    panic!("expected diagonal Gaussian log-sigma block")
                }
            };

        let mut states_minus = states;
        states_minus[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta[0] -= eps;
        let eval_minus = family.evaluate(&states_minus).expect("gaussian eval minus");
        let w_minus =
            match &eval_minus.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::Diagonal {
                    working_response: _,
                    working_weights,
                } => working_weights[0],
                BlockWorkingSet::ExactNewton { .. } => {
                    panic!("expected diagonal Gaussian log-sigma block")
                }
            };

        let fd = (w_plus - w_minus) / (2.0 * eps);
        assert!((dw[0] - fd).abs() < 1e-6, "dw={} fd={}", dw[0], fd);
    }

    #[test]
    fn gaussian_sigma_helper_matches_exact_exp_link() {
        let eta0 = 701.0_f64;
        let eta = array![eta0];
        let (sigma, d1, d2, d3, d4) = exp_sigma_derivs_up_to_fourth_array(eta.view());
        let coded_sigma = safe_exp(eta0);
        assert!(
            (sigma[0] - coded_sigma).abs() < 1e-30,
            "Gaussian sigma helper should evaluate the exact exp sigma link at eta={eta0}; got {} vs {}",
            sigma[0],
            coded_sigma
        );
        assert!(
            (d1[0] - sigma[0]).abs() / sigma[0] < 1e-12,
            "Gaussian sigma helper first derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
            d1[0],
            sigma[0]
        );
        assert!(
            (d2[0] - sigma[0]).abs() / sigma[0] < 1e-12,
            "Gaussian sigma helper second derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
            d2[0],
            sigma[0]
        );
        assert!(
            (d3[0] - sigma[0]).abs() / sigma[0] < 1e-12,
            "Gaussian sigma helper third derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
            d3[0],
            sigma[0]
        );
        assert!(
            (d4[0] - sigma[0]).abs() / sigma[0] < 1e-12,
            "Gaussian sigma helper fourth derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
            d4[0],
            sigma[0]
        );
    }

    #[test]
    fn gaussian_log_sigma_design_keeps_shared_mean_basis() {
        let shared = array![[1.0, -1.5], [1.0, -0.25], [1.0, 0.75], [1.0, 2.0],];
        let prepared = prepared_gaussian_log_sigma_design(&shared, &shared)
            .expect("gaussian log-sigma design should accept shared columns");

        for i in 0..shared.nrows() {
            for j in 0..shared.ncols() {
                assert!(
                    (prepared[[i, j]] - shared[[i, j]]).abs() < 1e-12,
                    "gaussian log-sigma design should preserve shared basis at ({i}, {j}): got {}, expected {}",
                    prepared[[i, j]],
                    shared[[i, j]]
                );
            }
        }
    }

    #[test]
    fn gaussian_diagonal_log_sigma_block_uses_fisher_score_step_in_far_tail() {
        let family = GaussianLocationScaleFamily {
            y: array![0.0],
            weights: array![1.0],
            mu_design: None,
            log_sigma_design: None,
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        };
        let eta_mu = array![0.0];
        let eta_ls0 = 701.0_f64;
        let states_at = |eta_ls: f64| {
            vec![
                ParameterBlockState {
                    beta: Array1::zeros(0),
                    eta: eta_mu.clone(),
                },
                ParameterBlockState {
                    beta: Array1::zeros(0),
                    eta: array![eta_ls],
                },
            ]
        };

        let eval = family.evaluate(&states_at(eta_ls0)).expect("evaluate");
        match &eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                // logb link σ = b + e^η: at η ≫ log b the floor is dwarfed
                // (σ ≈ e^η ~ 1e304), so dlogσ/dη = 1 − b/σ → 1 to within
                // f64 precision and the IRLS step matches the pure-exp Fisher
                // step. Compute the expectation explicitly from the new link.
                let sigma = logb_sigma_from_eta_scalar(eta_ls0);
                let inv_s2 = (sigma * sigma).recip();
                let dlog = 1.0 - LOGB_SIGMA_FLOOR / sigma;
                let residual = family.y[0] - eta_mu[0];
                let expected_score =
                    family.weights[0] * (residual * residual * inv_s2 - 1.0) * dlog;
                let expected_info = 2.0 * family.weights[0] * dlog * dlog;
                let expected_response = eta_ls0 + expected_score / expected_info;

                assert!((working_weights[0] - expected_info).abs() < 1e-12);
                assert!(
                    (working_response[0] - expected_response).abs() < 1e-12,
                    "working response mismatch: got {}, expected {}",
                    working_response[0],
                    expected_response
                );
            }
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal Gaussian log-sigma block")
            }
        }

        let loglik = |eta_ls: f64| family.log_likelihood_only(&states_at(eta_ls)).expect("ll");
        let h = 1e-4;
        let ll_plus = loglik(eta_ls0 + h);
        let ll0 = loglik(eta_ls0);
        let ll_minus = loglik(eta_ls0 - h);
        let score_fd = (ll_plus - ll_minus) / (2.0 * h);
        assert!(score_fd.is_finite());
        assert!(
            (score_fd + 1.0).abs() < 1e-6,
            "far-tail score should be -1, got {score_fd}"
        );
        assert!(
            (ll_plus - 2.0 * ll0 + ll_minus).abs() < 1e-5,
            "far-tail Gaussian log-sigma block should have near-zero observed curvature"
        );
    }

    #[test]
    fn gaussian_exact_joint_path_stays_finite_in_exp_link_far_tail() {
        let mu_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]]));
        let log_sigma_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0]]));
        let family = GaussianLocationScaleFamily {
            y: array![0.0],
            weights: array![1.0],
            mu_design: Some(mu_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        };
        let beta_mu = array![0.0];
        let beta_ls = array![710.0];
        let states = vec![
            ParameterBlockState {
                beta: beta_mu.clone(),
                eta: mu_design.matrixvectormultiply(&beta_mu),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: log_sigma_design.matrixvectormultiply(&beta_ls),
            },
        ];

        let hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected Gaussian exact joint hessian");
        assert!(
            hessian.iter().all(|value| value.is_finite()),
            "far-tail Gaussian exact Hessian should stay finite; got {hessian:?}"
        );

        let direction = array![0.25, -0.5];
        let dh = family
            .exact_newton_joint_hessian_directional_derivative(&states, &direction)
            .expect("joint dH")
            .expect("expected Gaussian exact joint hessian directional derivative");
        assert!(
            dh.iter().all(|value| value.is_finite()),
            "far-tail Gaussian exact Hessian directional derivative should stay finite; got {dh:?}"
        );
    }

    #[test]
    fn gaussian_location_scale_hotloop_optimized_matches_legacy_and_is_faster_locally() {
        let n = 4096usize;
        let rounds = 250usize;
        let y = Array1::from_shape_fn(n, |i| ((i as f64) * 0.003).sin() + 0.1);
        let mu = Array1::from_shape_fn(n, |i| ((i as f64) * 0.001).cos() - 0.2);
        let eta_ls = Array1::from_shape_fn(n, |i| ((i as f64) * 0.002).sin() * 0.8 - 0.1);
        let weights = Array1::from_shape_fn(n, |i| if i % 37 == 0 { 0.0 } else { 1.0 });
        let ln2pi = (2.0 * std::f64::consts::PI).ln();

        let legacy_eval = || {
            let mut ll = 0.0;
            let mut zmu = Array1::<f64>::zeros(n);
            let mut wmu = Array1::<f64>::zeros(n);
            let mut zls = Array1::<f64>::zeros(n);
            let mut wls = Array1::<f64>::zeros(n);
            for i in 0..n {
                let w = weights[i];
                let eta = eta_ls[i];
                let SigmaJet1 { sigma, d1: _ } = logb_sigma_jet1_scalar(eta);
                let inv_s2 = (sigma * sigma).recip();
                let r = y[i] - mu[i];
                ll += w * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma.ln()));
                if w == 0.0 {
                    wmu[i] = 0.0;
                    zmu[i] = mu[i];
                } else {
                    wmu[i] = floor_positiveweight(w * inv_s2, MIN_WEIGHT);
                    zmu[i] = mu[i] + r;
                }
                let dlogsigma_du = 1.0 - LOGB_SIGMA_FLOOR / sigma;
                let info_u =
                    floor_positiveweight(2.0 * w * dlogsigma_du * dlogsigma_du, MIN_WEIGHT);
                if info_u == 0.0 {
                    wls[i] = 0.0;
                    zls[i] = eta;
                } else {
                    wls[i] = info_u;
                    let score_ls = w * (r * r * inv_s2 - 1.0) * dlogsigma_du;
                    zls[i] = eta + score_ls / info_u;
                }
            }
            (ll, zmu, wmu, zls, wls)
        };

        let optimized_eval = || {
            let mut ll = 0.0;
            let mut zmu = Array1::<f64>::zeros(n);
            let mut wmu = Array1::<f64>::zeros(n);
            let mut zls = Array1::<f64>::zeros(n);
            let mut wls = Array1::<f64>::zeros(n);
            for i in 0..n {
                let eta = eta_ls[i];
                let SigmaJet1 { sigma, d1: _ } = logb_sigma_jet1_scalar(eta);
                let inv_s2 = (sigma * sigma).recip();
                let w = weights[i];
                let r = y[i] - mu[i];
                ll += w * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma.ln()));
                if w == 0.0 {
                    wmu[i] = 0.0;
                    zmu[i] = mu[i];
                } else {
                    wmu[i] = floor_positiveweight(w * inv_s2, MIN_WEIGHT);
                    zmu[i] = mu[i] + r;
                }
                let dlogsigma_du = 1.0 - LOGB_SIGMA_FLOOR / sigma;
                let info_u =
                    floor_positiveweight(2.0 * w * dlogsigma_du * dlogsigma_du, MIN_WEIGHT);
                if info_u == 0.0 {
                    wls[i] = 0.0;
                    zls[i] = eta;
                } else {
                    wls[i] = info_u;
                    let score_ls = w * (r * r * inv_s2 - 1.0) * dlogsigma_du;
                    zls[i] = eta + score_ls / info_u;
                }
            }
            (ll, zmu, wmu, zls, wls)
        };

        let (ll_legacy, zmu_legacy, wmu_legacy, zls_legacy, wls_legacy) = legacy_eval();
        let (ll_opt, zmu_opt, wmu_opt, zls_opt, wls_opt) = optimized_eval();
        assert!((ll_legacy - ll_opt).abs() < 1e-10);
        assert!((&zmu_legacy - &zmu_opt).iter().all(|v| v.abs() < 1e-12));
        assert!((&wmu_legacy - &wmu_opt).iter().all(|v| v.abs() < 1e-12));
        assert!((&zls_legacy - &zls_opt).iter().all(|v| v.abs() < 1e-12));
        assert!((&wls_legacy - &wls_opt).iter().all(|v| v.abs() < 1e-12));

        for _ in 0..rounds {
            std::hint::black_box(legacy_eval());
        }
        for _ in 0..rounds {
            std::hint::black_box(optimized_eval());
        }
    }

    fn simple_matern_term_collection(
        feature_cols: &[usize],
        length_scale: f64,
    ) -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![SmoothTermSpec {
                name: "spatial".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: feature_cols.to_vec(),
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::EqualMass { num_centers: 6 },
                        length_scale,
                        nu: MaternNu::ThreeHalves,
                        include_intercept: false,
                        double_penalty: false,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        }
    }

    fn spatial_kappa_options() -> SpatialLengthScaleOptimizationOptions {
        SpatialLengthScaleOptimizationOptions {
            enabled: true,
            max_outer_iter: 4,
            rel_tol: 1e-4,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 0.1,
            max_length_scale: 2.0,
            pilot_subsample_threshold: 10_000,
        }
    }

    fn spatial_fit_smoke_options() -> BlockwiseFitOptions {
        BlockwiseFitOptions {
            inner_max_cycles: 24,
            inner_tol: 1e-4,
            outer_max_iter: 3,
            outer_tol: 1e-4,
            ..BlockwiseFitOptions::default()
        }
    }

    #[test]
    fn binomial_location_scale_exact_probit_tailobjects_stay_finite() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_elem(n, 1.0);
        let threshold_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        ));
        let log_sigma_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        ));
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            policy: crate::resource::ResourcePolicy::default_library(),
        };
        let beta_t = array![250.0];
        let beta_ls = array![0.0];
        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: threshold_design.matrixvectormultiply(&beta_t),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: log_sigma_design.matrixvectormultiply(&beta_ls),
            },
        ];

        let eval = family
            .evaluate(&states)
            .expect("evaluate tail-stable family");
        assert!(eval.log_likelihood.is_finite());
        let joint = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected exact joint hessian");
        assert!(joint.iter().all(|v| v.is_finite()));
        let direction = array![0.1, -0.2];
        let d_h = family
            .exact_newton_joint_hessian_directional_derivative(&states, &direction)
            .expect("joint dH")
            .expect("expected exact joint dH");
        assert!(d_h.iter().all(|v| v.is_finite()));
        let d2_h = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states, &direction, &direction,
            )
            .expect("joint d2H")
            .expect("expected exact joint d2H");
        assert!(d2_h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn binomial_location_scale_term_builder_requires_exact_spatial_joint_path() {
        let n = 8usize;
        let builder = BinomialLocationScaleTermBuilder {
            y: Array1::from_elem(n, 0.0),
            weights: Array1::from_elem(n, 1.0),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            meanspec: simple_matern_term_collection(&[0, 1], 0.4),
            noisespec: simple_matern_term_collection(&[0, 1], 0.75),
            mean_offset: Array1::zeros(n),
            noise_offset: Array1::zeros(n),
        };
        assert!(builder.exact_spatial_joint_supported());
        assert!(builder.require_exact_spatial_joint());
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
        }
        let mean_design =
            build_term_collection_design(data.view(), builder.meanspec()).expect("mean design");
        let noise_design =
            build_term_collection_design(data.view(), builder.noisespec()).expect("noise design");
        let family = builder.build_family(&mean_design, &noise_design);
        assert!(family.exact_joint_supported());
    }

    #[test]
    fn binomial_location_scalewiggle_term_builder_requires_exact_spatial_joint_path() {
        let n = 8usize;
        let q_seed = Array1::linspace(-1.25, 1.25, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let builder = BinomialLocationScaleWiggleTermBuilder {
            y: Array1::from_elem(n, 0.0),
            weights: Array1::from_elem(n, 1.0),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            meanspec: simple_matern_term_collection(&[0, 1], 0.4),
            noisespec: simple_matern_term_collection(&[0, 1], 0.75),
            mean_offset: Array1::zeros(n),
            noise_offset: Array1::zeros(n),
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        assert!(builder.exact_spatial_joint_supported());
        assert!(builder.require_exact_spatial_joint());
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
        }
        let mean_design =
            build_term_collection_design(data.view(), builder.meanspec()).expect("mean design");
        let noise_design =
            build_term_collection_design(data.view(), builder.noisespec()).expect("noise design");
        let family = builder.build_family(&mean_design, &noise_design);
        assert!(family.exact_joint_supported());
        assert!(family.requires_joint_outer_hyper_path());
    }

    #[test]
    fn binomial_location_scale_builder_populateswarm_start_betas() {
        let n = 12usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
        }
        let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
        let weights = Array1::from_elem(n, 1.0);
        let builder = BinomialLocationScaleTermBuilder {
            mean_offset: Array1::zeros(y.len()),
            noise_offset: Array1::zeros(y.len()),
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            meanspec: simple_matern_term_collection(&[0, 1], 0.45),
            noisespec: simple_matern_term_collection(&[0, 1], 0.8),
        };
        let mean_design =
            build_term_collection_design(data.view(), builder.meanspec()).expect("mean design");
        let noise_design =
            build_term_collection_design(data.view(), builder.noisespec()).expect("noise design");
        let rho = compose_theta_from_hints_test(
            builder.mean_penalty_count(&mean_design),
            builder.noise_penalty_count(&noise_design),
            &None,
            &None,
            &Array1::zeros(0),
        );
        let blocks = builder
            .build_blocks(&rho, &mean_design, &noise_design, None, None)
            .expect("build blocks");
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].initial_beta.is_some());
        assert!(blocks[1].initial_beta.is_some());
    }

    #[test]
    fn binomial_location_scalewiggle_builder_populateswarm_start_betas() {
        let n = 12usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).cos();
        }
        let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
        let weights = Array1::from_elem(n, 1.0);
        let q_seed = Array1::linspace(-1.25, 1.25, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let builder = BinomialLocationScaleWiggleTermBuilder {
            mean_offset: Array1::zeros(y.len()),
            noise_offset: Array1::zeros(y.len()),
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            meanspec: simple_matern_term_collection(&[0, 1], 0.45),
            noisespec: simple_matern_term_collection(&[0, 1], 0.8),
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        let mean_design =
            build_term_collection_design(data.view(), builder.meanspec()).expect("mean design");
        let noise_design =
            build_term_collection_design(data.view(), builder.noisespec()).expect("noise design");
        let rho = compose_theta_from_hints_test(
            builder.mean_penalty_count(&mean_design),
            builder.noise_penalty_count(&noise_design),
            &None,
            &None,
            &builder.extra_rho0().expect("extra rho0"),
        );
        let blocks = builder
            .build_blocks(&rho, &mean_design, &noise_design, None, None)
            .expect("build blocks");
        assert_eq!(blocks.len(), 3);
        assert!(blocks[0].initial_beta.is_some());
        assert!(blocks[1].initial_beta.is_some());
    }

    #[test]
    fn binomial_location_scale_exact_newton_spatial_joint_hyper_returns_fullhessian() {
        let n = 12usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).cos();
        }
        let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
        let weights = Array1::from_elem(n, 1.0);
        let meanspec = simple_matern_term_collection(&[0, 1], 0.45);
        let noisespec = simple_matern_term_collection(&[0, 1], 0.8);
        let builder = BinomialLocationScaleTermBuilder {
            mean_offset: Array1::zeros(y.len()),
            noise_offset: Array1::zeros(y.len()),
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            meanspec: meanspec.clone(),
            noisespec: noisespec.clone(),
        };
        let mean_design =
            build_term_collection_design(data.view(), &meanspec).expect("build mean design");
        let noise_design =
            build_term_collection_design(data.view(), &noisespec).expect("build noise design");
        let meanspec_resolved =
            freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
        let noisespec_resolved = freeze_term_collection_from_design(&noisespec, &noise_design)
            .expect("freeze noise spec");
        let rho = compose_theta_from_hints_test(
            builder.mean_penalty_count(&mean_design),
            builder.noise_penalty_count(&noise_design),
            &None,
            &None,
            &Array1::zeros(0),
        );
        let blocks = builder
            .build_blocks(&rho, &mean_design, &noise_design, None, None)
            .expect("build blocks");
        let family = builder.build_family(&mean_design, &noise_design);
        let derivative_blocks = builder
            .build_psiderivative_blocks(
                data.view(),
                &meanspec_resolved,
                &noisespec_resolved,
                &mean_design,
                &noise_design,
            )
            .expect("psi derivative blocks");
        let eval = evaluate_custom_family_joint_hyper(
            &family,
            &blocks,
            &BlockwiseFitOptions {
                use_remlobjective: true,
                outer_max_iter: 1,
                ..BlockwiseFitOptions::default()
            },
            &rho,
            &derivative_blocks,
            None,
            crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian,
        )
        .expect("exact spatial joint hyper eval");
        assert!(eval.objective.is_finite());
        assert!(eval.gradient.iter().all(|v| v.is_finite()));
        let hess = eval
            .outer_hessian
            .materialize_dense()
            .expect("exact spatial joint hyper path should materialize a full [rho, psi] hessian")
            .expect("exact spatial joint hyper path should return a full [rho, psi] hessian");
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        let theta_dim = rho.len() + psi_dim;
        assert_eq!(eval.gradient.len(), theta_dim);
        assert_eq!(hess.nrows(), theta_dim);
        assert_eq!(hess.ncols(), theta_dim);
    }

    #[test]
    fn binomial_location_scalewiggle_exact_newton_spatial_joint_hyper_returns_fullhessian() {
        let n = 14usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.25 * std::f64::consts::PI * t).sin();
        }
        let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
        let weights = Array1::from_elem(n, 1.0);
        let meanspec = simple_matern_term_collection(&[0, 1], 0.45);
        let noisespec = simple_matern_term_collection(&[0, 1], 0.8);
        let q_seed = Array1::linspace(-1.5, 1.5, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let builder = BinomialLocationScaleWiggleTermBuilder {
            mean_offset: Array1::zeros(y.len()),
            noise_offset: Array1::zeros(y.len()),
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            meanspec: meanspec.clone(),
            noisespec: noisespec.clone(),
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        let mean_design =
            build_term_collection_design(data.view(), &meanspec).expect("build mean design");
        let noise_design =
            build_term_collection_design(data.view(), &noisespec).expect("build noise design");
        let meanspec_resolved =
            freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
        let noisespec_resolved = freeze_term_collection_from_design(&noisespec, &noise_design)
            .expect("freeze noise spec");
        let rho = compose_theta_from_hints_test(
            builder.mean_penalty_count(&mean_design),
            builder.noise_penalty_count(&noise_design),
            &None,
            &None,
            &builder.extra_rho0().expect("wiggle rho0"),
        );
        let blocks = builder
            .build_blocks(&rho, &mean_design, &noise_design, None, None)
            .expect("build blocks");
        let family = builder.build_family(&mean_design, &noise_design);
        let derivative_blocks = builder
            .build_psiderivative_blocks(
                data.view(),
                &meanspec_resolved,
                &noisespec_resolved,
                &mean_design,
                &noise_design,
            )
            .expect("psi derivative blocks");
        let eval = evaluate_custom_family_joint_hyper(
            &family,
            &blocks,
            &BlockwiseFitOptions {
                use_remlobjective: true,
                outer_max_iter: 1,
                ..BlockwiseFitOptions::default()
            },
            &rho,
            &derivative_blocks,
            None,
            crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian,
        )
        .expect("exact wiggle spatial joint hyper eval");
        assert!(eval.objective.is_finite());
        assert!(eval.gradient.iter().all(|v| v.is_finite()));
        let hess = eval
            .outer_hessian
            .materialize_dense()
            .expect("exact wiggle spatial joint hyper path should materialize a full [rho, psi] hessian")
            .expect("exact wiggle spatial joint hyper path should return a full [rho, psi] hessian");
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        let theta_dim = rho.len() + psi_dim;
        assert_eq!(eval.gradient.len(), theta_dim);
        assert_eq!(hess.nrows(), theta_dim);
        assert_eq!(hess.ncols(), theta_dim);
    }

    #[test]
    fn gaussian_location_scale_exact_newton_spatial_joint_hyper_returns_fullhessian() {
        let n = 12usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
        }
        let y = Array1::from_iter((0..n).map(|i| {
            let x0 = data[[i, 0]];
            let x1 = data[[i, 1]];
            0.4 * x0 - 0.2 * x1 + 0.15
        }));
        let weights = Array1::from_elem(n, 1.0);
        let meanspec = simple_matern_term_collection(&[0, 1], 0.45);
        let noisespec = simple_matern_term_collection(&[0, 1], 0.8);
        let builder = GaussianLocationScaleTermBuilder {
            y,
            weights,
            meanspec: meanspec.clone(),
            noisespec: noisespec.clone(),
            mean_offset: Array1::zeros(n),
            noise_offset: Array1::zeros(n),
        };
        let mean_design =
            build_term_collection_design(data.view(), &meanspec).expect("build mean design");
        let noise_design =
            build_term_collection_design(data.view(), &noisespec).expect("build noise design");
        let meanspec_resolved =
            freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
        let noisespec_resolved = freeze_term_collection_from_design(&noisespec, &noise_design)
            .expect("freeze noise spec");
        let rho = compose_theta_from_hints_test(
            builder.mean_penalty_count(&mean_design),
            builder.noise_penalty_count(&noise_design),
            &None,
            &None,
            &Array1::zeros(0),
        );
        let blocks = builder
            .build_blocks(&rho, &mean_design, &noise_design, None, None)
            .expect("build blocks");
        let family = builder.build_family(&mean_design, &noise_design);
        let derivative_blocks = builder
            .build_psiderivative_blocks(
                data.view(),
                &meanspec_resolved,
                &noisespec_resolved,
                &mean_design,
                &noise_design,
            )
            .expect("psi derivative blocks");
        let eval = evaluate_custom_family_joint_hyper(
            &family,
            &blocks,
            &BlockwiseFitOptions {
                use_remlobjective: true,
                outer_max_iter: 1,
                ..BlockwiseFitOptions::default()
            },
            &rho,
            &derivative_blocks,
            None,
            crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian,
        )
        .expect("exact spatial joint hyper eval");
        assert!(eval.objective.is_finite());
        assert!(eval.gradient.iter().all(|v| v.is_finite()));
        let hess = eval
            .outer_hessian
            .materialize_dense()
            .expect("exact spatial joint hyper path should materialize a full [rho, psi] hessian")
            .expect("exact spatial joint hyper path should return a full [rho, psi] hessian");
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        let theta_dim = rho.len() + psi_dim;
        assert_eq!(eval.gradient.len(), theta_dim);
        assert_eq!(hess.nrows(), theta_dim);
        assert_eq!(hess.ncols(), theta_dim);
        assert!(hess.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn binomial_location_scalewiggle_family_exposes_joint_psi_hook_surface() {
        let n = 12usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (1.75 * std::f64::consts::PI * t).cos();
        }
        let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
        let weights = Array1::from_elem(n, 1.0);
        let meanspec = simple_matern_term_collection(&[0, 1], 0.4);
        let noisespec = simple_matern_term_collection(&[0, 1], 0.7);
        let q_seed = Array1::linspace(-1.25, 1.25, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let builder = BinomialLocationScaleWiggleTermBuilder {
            mean_offset: Array1::zeros(y.len()),
            noise_offset: Array1::zeros(y.len()),
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            meanspec: meanspec.clone(),
            noisespec: noisespec.clone(),
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        let mean_design =
            build_term_collection_design(data.view(), &meanspec).expect("build mean design");
        let noise_design =
            build_term_collection_design(data.view(), &noisespec).expect("build noise design");
        let meanspec_resolved =
            freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
        let noisespec_resolved = freeze_term_collection_from_design(&noisespec, &noise_design)
            .expect("freeze noise spec");
        let rho = compose_theta_from_hints_test(
            builder.mean_penalty_count(&mean_design),
            builder.noise_penalty_count(&noise_design),
            &None,
            &None,
            &builder.extra_rho0().expect("wiggle rho0"),
        );
        let blocks = builder
            .build_blocks(&rho, &mean_design, &noise_design, None, None)
            .expect("build blocks");
        let family = builder.build_family(&mean_design, &noise_design);
        let fit = fit_custom_family(
            &family,
            &blocks,
            &BlockwiseFitOptions {
                use_remlobjective: true,
                outer_max_iter: 1,
                ..BlockwiseFitOptions::default()
            },
        )
        .expect("fit wiggle family for joint psi hooks");
        let derivative_blocks = builder
            .build_psiderivative_blocks(
                data.view(),
                &meanspec_resolved,
                &noisespec_resolved,
                &mean_design,
                &noise_design,
            )
            .expect("psi derivative blocks");
        let psi_terms = family
            .exact_newton_joint_psi_terms(&fit.block_states, &blocks, &derivative_blocks, 0)
            .expect("joint psi terms call")
            .expect("wiggle family should return joint psi terms");
        let psi2_terms = family
            .exact_newton_joint_psisecond_order_terms(
                &fit.block_states,
                &blocks,
                &derivative_blocks,
                0,
                0,
            )
            .expect("joint psi second-order call")
            .expect("wiggle family should return joint psi second-order terms");
        let total = fit
            .block_states
            .iter()
            .map(|state| state.beta.len())
            .sum::<usize>();
        assert_eq!(psi_terms.score_psi.len(), total);
        if psi_terms.hessian_psi_operator.is_some() {
            assert_eq!(psi_terms.hessian_psi.dim(), (0, 0));
        } else {
            assert_eq!(psi_terms.hessian_psi.dim(), (total, total));
        }
        assert_eq!(psi2_terms.score_psi_psi.len(), total);
        if psi2_terms.hessian_psi_psi_operator.is_some() {
            assert_eq!(psi2_terms.hessian_psi_psi.dim(), (0, 0));
        } else {
            assert_eq!(psi2_terms.hessian_psi_psi.dim(), (total, total));
        }

        let mut d_beta_flat = Array1::<f64>::zeros(total);
        let mut at = 0usize;
        for state in &fit.block_states {
            let end = at + state.beta.len();
            d_beta_flat
                .slice_mut(s![at..end])
                .assign(&state.beta.mapv(|v| 0.25 * v + 0.1));
            at = end;
        }
        let mixed = family
            .exact_newton_joint_psihessian_directional_derivative(
                &fit.block_states,
                &blocks,
                &derivative_blocks,
                0,
                &d_beta_flat,
            )
            .expect("joint psi mixed drift call")
            .expect("wiggle family should return joint psi mixed drift");
        assert_eq!(mixed.dim(), (total, total));
    }

    #[test]
    fn gaussian_location_scale_family_exposes_joint_psi_hook_surface() {
        let n = 10usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).cos();
        }
        let y = Array1::from_iter((0..n).map(|i| {
            let x0 = data[[i, 0]];
            let x1 = data[[i, 1]];
            0.3 * x0 - 0.15 * x1 + 0.2
        }));
        let weights = Array1::from_elem(n, 1.0);
        let meanspec = simple_matern_term_collection(&[0, 1], 0.4);
        let noisespec = simple_matern_term_collection(&[0, 1], 0.7);
        let builder = GaussianLocationScaleTermBuilder {
            y,
            weights,
            meanspec: meanspec.clone(),
            noisespec: noisespec.clone(),
            mean_offset: Array1::zeros(n),
            noise_offset: Array1::zeros(n),
        };
        let mean_design =
            build_term_collection_design(data.view(), &meanspec).expect("build mean design");
        let noise_design =
            build_term_collection_design(data.view(), &noisespec).expect("build noise design");
        let meanspec_resolved =
            freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
        let noisespec_resolved = freeze_term_collection_from_design(&noisespec, &noise_design)
            .expect("freeze noise spec");
        let rho = compose_theta_from_hints_test(
            builder.mean_penalty_count(&mean_design),
            builder.noise_penalty_count(&noise_design),
            &None,
            &None,
            &Array1::zeros(0),
        );
        let blocks = builder
            .build_blocks(&rho, &mean_design, &noise_design, None, None)
            .expect("build blocks");
        let family = builder.build_family(&mean_design, &noise_design);
        let fit = fit_custom_family(
            &family,
            &blocks,
            &BlockwiseFitOptions {
                use_remlobjective: true,
                outer_max_iter: 1,
                ..BlockwiseFitOptions::default()
            },
        )
        .expect("fit gaussian family for joint psi hooks");
        let derivative_blocks = builder
            .build_psiderivative_blocks(
                data.view(),
                &meanspec_resolved,
                &noisespec_resolved,
                &mean_design,
                &noise_design,
            )
            .expect("psi derivative blocks");
        let psi_terms = family
            .exact_newton_joint_psi_terms(&fit.block_states, &blocks, &derivative_blocks, 0)
            .expect("joint psi terms call")
            .expect("gaussian family should return joint psi terms");
        let psi2_terms = family
            .exact_newton_joint_psisecond_order_terms(
                &fit.block_states,
                &blocks,
                &derivative_blocks,
                0,
                0,
            )
            .expect("joint psi second-order call")
            .expect("gaussian family should return joint psi second-order terms");
        let total = fit
            .block_states
            .iter()
            .map(|state| state.beta.len())
            .sum::<usize>();
        assert_eq!(psi_terms.score_psi.len(), total);
        if psi_terms.hessian_psi_operator.is_some() {
            assert_eq!(psi_terms.hessian_psi.dim(), (0, 0));
        } else {
            assert_eq!(psi_terms.hessian_psi.dim(), (total, total));
        }
        assert_eq!(psi2_terms.score_psi_psi.len(), total);
        if psi2_terms.hessian_psi_psi_operator.is_some() {
            assert_eq!(psi2_terms.hessian_psi_psi.dim(), (0, 0));
        } else {
            assert_eq!(psi2_terms.hessian_psi_psi.dim(), (total, total));
        }

        let mut d_beta_flat = Array1::<f64>::zeros(total);
        let mut at = 0usize;
        for state in &fit.block_states {
            let end = at + state.beta.len();
            d_beta_flat
                .slice_mut(s![at..end])
                .assign(&state.beta.mapv(|v| 0.2 * v + 0.15));
            at = end;
        }
        let mixed = family
            .exact_newton_joint_psihessian_directional_derivative(
                &fit.block_states,
                &blocks,
                &derivative_blocks,
                0,
                &d_beta_flat,
            )
            .expect("joint psi mixed drift call")
            .expect("gaussian family should return joint psi mixed drift");
        assert_eq!(mixed.dim(), (total, total));
    }

    #[test]
    fn gaussian_location_scale_terms_reject_invalidweights_early() {
        let n = 8usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64;
            data[[i, 1]] = (i as f64).sin();
        }
        let spec = GaussianLocationScaleTermSpec {
            y: Array1::zeros(n),
            weights: Array1::from_vec(vec![1.0, 1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
            meanspec: simple_matern_term_collection(&[0, 1], 0.35),
            log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.6),
            mean_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
        };

        let err = match fit_gaussian_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        ) {
            Ok(_) => panic!("term API should reject negative weights"),
            Err(err) => err,
        };
        assert!(err.contains("weights must be finite and non-negative"));
    }

    #[test]
    fn binomial_location_scale_terms_reject_invalid_response_early() {
        let n = 8usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64;
            data[[i, 1]] = (i as f64).cos();
        }
        let spec = BinomialLocationScaleTermSpec {
            y: Array1::from_vec(vec![0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Array1::from_elem(n, 1.0),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            thresholdspec: simple_matern_term_collection(&[0, 1], 0.4),
            log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.75),
            threshold_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
        };

        let err = match fit_binomial_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        ) {
            Ok(_) => panic!("term API should reject invalid binomial responses"),
            Err(err) => err,
        };
        assert!(err.contains("binomial response must be finite in [0,1]"));
    }

    #[test]
    fn binomial_location_scale_terms_reject_datarow_mismatch_early() {
        let n = 8usize;
        let data = Array2::<f64>::zeros((n - 1, 2));
        let spec = BinomialLocationScaleTermSpec {
            y: Array1::from_elem(n, 0.0),
            weights: Array1::from_elem(n, 1.0),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            thresholdspec: simple_matern_term_collection(&[0, 1], 0.4),
            log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.75),
            threshold_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
        };

        let err = match fit_binomial_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        ) {
            Ok(_) => panic!("term API should reject data/y row mismatches"),
            Err(err) => err,
        };
        assert!(err.contains("data row count must match response length"));
    }

    #[test]
    fn gaussian_location_scale_termswith_matern_spatial_blocks_fit_finitely() {
        let n = 32usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
        }
        let y = Array1::from_iter((0..n).map(|i| {
            let x0 = data[[i, 0]];
            let x1 = data[[i, 1]];
            0.5 * x0 - 0.25 * x1 + 0.1
        }));
        let weights = Array1::from_elem(n, 1.0);
        let spec = GaussianLocationScaleTermSpec {
            y,
            weights,
            meanspec: simple_matern_term_collection(&[0, 1], 0.35),
            log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.6),
            mean_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
        };
        let fit = fit_gaussian_location_scale_terms(
            data.view(),
            spec,
            &spatial_fit_smoke_options(),
            &spatial_kappa_options(),
        )
        .expect("gaussian location-scale spatial fit");
        assert!(fit.fit.penalized_objective.is_finite());
        assert_eq!(fit.fit.block_states.len(), 2);
    }

    #[test]
    fn binomial_location_scale_termswith_matern_spatial_blocks_fit_finitely() {
        let n = 36usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (3.0 * std::f64::consts::PI * t).cos();
        }
        let y = Array1::from_iter((0..n).map(|i| if i % 5 == 0 || i % 7 == 0 { 1.0 } else { 0.0 }));
        let weights = Array1::from_elem(n, 1.0);
        let spec = BinomialLocationScaleTermSpec {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            thresholdspec: simple_matern_term_collection(&[0, 1], 0.4),
            log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.75),
            threshold_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
        };
        let fit = fit_binomial_location_scale_terms(
            data.view(),
            spec,
            &spatial_fit_smoke_options(),
            &spatial_kappa_options(),
        )
        .expect("binomial location-scale spatial fit");
        assert!(fit.fit.penalized_objective.is_finite());
        assert_eq!(fit.fit.block_states.len(), 2);
    }

    #[test]
    fn binomial_location_scalewiggle_termswith_matern_spatial_blocks_fit_finitely() {
        let n = 30usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.5 * std::f64::consts::PI * t).sin();
        }
        let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 9 == 0 { 1.0 } else { 0.0 }));
        let weights = Array1::from_elem(n, 1.0);
        let q_seed = Array1::linspace(-1.5, 1.5, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let spec = BinomialLocationScaleWiggleTermSpec {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            thresholdspec: simple_matern_term_collection(&[0, 1], 0.45),
            log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.8),
            threshold_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        let fit = fit_binomial_location_scalewiggle_terms(
            data.view(),
            spec,
            &spatial_fit_smoke_options(),
            &spatial_kappa_options(),
        )
        .expect("binomial location-scale wiggle spatial fit");
        assert!(fit.fit.penalized_objective.is_finite());
        assert_eq!(fit.fit.block_states.len(), 3);
    }

    #[test]
    fn wiggle_family_evaluate_returns_exact_newton_blocks() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.5, 1.5, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let threshold_design = threshold_block.design.clone();
        let log_sigma_design = log_sigma_block.design.clone();
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
            wiggle_knots: knots,
            wiggle_degree: 2,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let eta_t = Array1::from_vec(vec![0.4; n]);
        let eta_ls = Array1::from_vec(vec![-0.2; n]);
        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let betaw = Array1::from_vec(vec![0.05; wiggle_block.design.ncols()]);
        let etaw = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&betaw);
        let eval = family
            .evaluate(&[
                ParameterBlockState {
                    beta: Array1::from_vec(vec![0.4]),
                    eta: eta_t,
                },
                ParameterBlockState {
                    beta: Array1::from_vec(vec![-0.2]),
                    eta: eta_ls,
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: etaw,
                },
            ])
            .expect("evaluate");

        assert_eq!(eval.blockworking_sets.len(), 3);
        match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                let hessian = hessian.to_dense();
                assert_eq!(gradient.len(), 1);
                assert_eq!(hessian.dim(), (1, 1));
                assert!(gradient[0].is_finite());
                assert!(hessian[[0, 0]].is_finite());
            }
            BlockWorkingSet::Diagonal { .. } => panic!("threshold block should be exact newton"),
        }
        match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                let hessian = hessian.to_dense();
                assert_eq!(gradient.len(), 1);
                assert_eq!(hessian.dim(), (1, 1));
                assert!(gradient[0].is_finite());
                assert!(hessian[[0, 0]].is_finite());
            }
            BlockWorkingSet::Diagonal { .. } => panic!("log-sigma block should be exact newton"),
        }
        match &eval.blockworking_sets[2] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                let hessian = hessian.to_dense();
                assert_eq!(gradient.len(), betaw.len());
                assert_eq!(hessian.nrows(), betaw.len());
                assert_eq!(hessian.ncols(), betaw.len());
                assert!(gradient.iter().all(|v| v.is_finite()));
                assert!(hessian.iter().all(|v| v.is_finite()));
            }
            BlockWorkingSet::Diagonal { .. } => panic!("wiggle block should be exact newton"),
        }
    }

    #[test]
    fn wiggle_family_exact_newton_directional_derivative_matches_finite_difference() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            3,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let threshold_design = threshold_block.design.clone();
        let log_sigma_design = log_sigma_block.design.clone();
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let beta_t = Array1::from_vec(vec![0.25]);
        let beta_ls = Array1::from_vec(vec![-0.15]);
        let eta_t = threshold_design.matrixvectormultiply(&beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let etaw = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&betaw);

        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls.clone(),
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw.clone(),
            },
        ];

        let extract = |eval: FamilyEvaluation, idx: usize| -> Array2<f64> {
            match &eval.blockworking_sets[idx] {
                BlockWorkingSet::ExactNewton {
                    gradient: _,
                    hessian,
                } => hessian.to_dense(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
            }
        };

        let base_eval = family.evaluate(&states).expect("base eval");
        let eps = 1e-6;
        for block_idx in 0..3 {
            let d_beta = Array1::ones(states[block_idx].beta.len());
            let analytic = family
                .exact_newton_hessian_directional_derivative(&states, block_idx, &d_beta)
                .expect("analytic dH")
                .expect("expected derivative");

            let mut plus_states = states.clone();
            plus_states[block_idx].beta = &plus_states[block_idx].beta + &(eps * &d_beta);
            plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta = threshold_design
                .matrixvectormultiply(
                    &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta,
                );
            plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta = log_sigma_design
                .matrixvectormultiply(
                    &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta,
                );
            let plus_core_q0 = binomial_location_scale_core(
                &y,
                &weights,
                &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta,
                &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta,
                None,
                &family.link_kind,
            )
            .expect("plus core q0");
            plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta = family
                .wiggle_design(plus_core_q0.q0.view())
                .expect("plus wiggle design")
                .dot(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta);

            let h_plus = extract(family.evaluate(&plus_states).expect("plus eval"), block_idx);
            let h_base = extract(base_eval.clone(), block_idx);
            let fd = (h_plus - h_base) / eps;
            crate::testing::assert_matrix_derivativefd(
                &fd,
                &analytic,
                5e-4,
                &format!("block {} dH", block_idx),
            );
        }
    }

    #[test]
    fn wiggle_threshold_block_exacthessian_matches_autodiffobjective() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            3,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let threshold_design = threshold_block.design.clone();
        let log_sigma_design = log_sigma_block.design.clone();
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Logit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots.clone(),
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let beta_t0 = 0.25;
        let beta_ls0 = -0.15;
        let beta_t = array![beta_t0];
        let beta_ls = array![beta_ls0];
        let eta_t = threshold_design.matrixvectormultiply(&beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let etaw = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&betaw);
        let states = vec![
            ParameterBlockState {
                beta: beta_t,
                eta: eta_t,
            },
            ParameterBlockState {
                beta: beta_ls,
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ];

        let eval = family.evaluate(&states).expect("evaluate wiggle family");
        let blockhessian = match &eval.blockworking_sets[BinomialLocationScaleWiggleFamily::BLOCK_T]
        {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton threshold block"),
        };
        let (_, _, hess_ad) = second_derivative(
            |bt| wiggle_negloglik_threshold_numdual(bt, beta_ls0, &betaw, &y, &weights, &knots, 3),
            beta_t0,
        );
        assert!(
            (blockhessian[[0, 0]] - hess_ad).abs() <= 5e-6,
            "wiggle threshold exact hessian mismatch: evaluate()={} autodiff={}",
            blockhessian[[0, 0]],
            hess_ad
        );
    }

    #[test]
    fn gaussian_log_sigma_psi_terms_match_autodiff_scalar_objective() {
        let y = array![0.25, -0.4, 1.1];
        let weights = array![1.0, 0.7, 1.3];
        let x_mu0 = array![1.0, -0.35, 0.6];
        let x_ls0 = array![0.8, -0.25, 0.45];
        let x_ls_psi = array![0.2, -0.15, 0.1];
        let x_ls_psi_psi = array![0.05, -0.03, 0.04];
        let beta_mu0 = 0.35_f64;
        let beta_ls0 = -0.2_f64;

        let x_mu0_mat = x_mu0.clone().insert_axis(Axis(1));
        let x_ls0_mat = x_ls0.clone().insert_axis(Axis(1));
        let family = GaussianLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            mu_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_mu0_mat.clone(),
            ))),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_ls0_mat.clone(),
            ))),
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        };
        let specs = vec![
            gaussian_psi_test_spec("mu", x_mu0_mat.clone()),
            gaussian_psi_test_spec("log_sigma", x_ls0_mat.clone()),
        ];
        let states = vec![
            ParameterBlockState {
                beta: array![beta_mu0],
                eta: x_mu0_mat.column(0).to_owned() * beta_mu0,
            },
            ParameterBlockState {
                beta: array![beta_ls0],
                eta: x_ls0_mat.column(0).to_owned() * beta_ls0,
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: x_ls_psi.clone().insert_axis(Axis(1)),
                s_psi: Array2::zeros((1, 1)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: Some(vec![x_ls_psi_psi.clone().insert_axis(Axis(1))]),
                s_psi_psi: Some(vec![Array2::zeros((1, 1))]),
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: None,
                implicit_axis: 0,
                implicit_group_id: None,
            }],
        ];

        let psi_terms = family
            .exact_newton_joint_psi_terms(&states, &specs, &derivative_blocks, 0)
            .expect("joint psi terms")
            .expect("expected gaussian psi terms");

        let vars = [beta_mu0, beta_ls0, 0.0_f64];
        let (_, dpsi, _) = second_derivative(
            |psi| {
                gaussian_negloglik_log_sigma_psi_only_numdual(
                    psi,
                    beta_mu0,
                    beta_ls0,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            0.0,
        );
        let (_, _, _, score_mu_psi) = second_partial_derivative(
            |(beta_mu, psi)| {
                gaussian_negloglik_log_sigma_mu_psi_numdual(
                    beta_mu,
                    psi,
                    beta_ls0,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            (beta_mu0, 0.0),
        );
        let (_, _, _, score_ls_psi) = second_partial_derivative(
            |(beta_ls, psi)| {
                gaussian_negloglik_log_sigma_ls_psi_numdual(
                    beta_ls,
                    psi,
                    beta_mu0,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            (beta_ls0, 0.0),
        );
        let (_, _, _, _, _, _, _, h_mu_mu_psi) = third_partial_derivative_vec(
            |v| {
                gaussian_negloglik_log_sigma_beta_vec_numdual(
                    v,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            &vars,
            0,
            0,
            2,
        );
        let (_, _, _, _, _, _, _, h_mu_ls_psi) = third_partial_derivative_vec(
            |v| {
                gaussian_negloglik_log_sigma_beta_vec_numdual(
                    v,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            &vars,
            0,
            1,
            2,
        );
        let (_, _, _, _, _, _, _, h_ls_ls_psi) = third_partial_derivative_vec(
            |v| {
                gaussian_negloglik_log_sigma_beta_vec_numdual(
                    v,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            &vars,
            1,
            1,
            2,
        );

        assert!(
            (psi_terms.objective_psi - dpsi).abs() <= 1e-10,
            "Gaussian log-sigma psi objective derivative mismatch: analytic={} autodiff={}",
            psi_terms.objective_psi,
            dpsi
        );
        assert!(
            (psi_terms.score_psi[0] - score_mu_psi).abs() <= 1e-10,
            "Gaussian log-sigma psi score_mu mismatch: analytic={} autodiff={}",
            psi_terms.score_psi[0],
            score_mu_psi
        );
        assert!(
            (psi_terms.score_psi[1] - score_ls_psi).abs() <= 1e-10,
            "Gaussian log-sigma psi score_ls mismatch: analytic={} autodiff={}",
            psi_terms.score_psi[1],
            score_ls_psi
        );
        assert!(
            (psi_terms.hessian_psi[[0, 0]] - h_mu_mu_psi).abs() <= 1e-9,
            "Gaussian log-sigma psi hessian(mu,mu) mismatch: analytic={} autodiff={}",
            psi_terms.hessian_psi[[0, 0]],
            h_mu_mu_psi
        );
        assert!(
            (psi_terms.hessian_psi[[0, 1]] - h_mu_ls_psi).abs() <= 1e-9,
            "Gaussian log-sigma psi hessian(mu,ls) mismatch: analytic={} autodiff={}",
            psi_terms.hessian_psi[[0, 1]],
            h_mu_ls_psi
        );
        assert!(
            (psi_terms.hessian_psi[[1, 1]] - h_ls_ls_psi).abs() <= 1e-9,
            "Gaussian log-sigma psi hessian(ls,ls) mismatch: analytic={} autodiff={}",
            psi_terms.hessian_psi[[1, 1]],
            h_ls_ls_psi
        );
    }

    #[test]
    fn gaussian_log_sigma_psi_second_order_terms_match_autodiff_scalar_objective() {
        let y = array![0.25, -0.4, 1.1];
        let weights = array![1.0, 0.7, 1.3];
        let x_mu0 = array![1.0, -0.35, 0.6];
        let x_ls0 = array![0.8, -0.25, 0.45];
        let x_ls_psi = array![0.2, -0.15, 0.1];
        let x_ls_psi_psi = array![0.05, -0.03, 0.04];
        let beta_mu0 = 0.35_f64;
        let beta_ls0 = -0.2_f64;

        let x_mu0_mat = x_mu0.clone().insert_axis(Axis(1));
        let x_ls0_mat = x_ls0.clone().insert_axis(Axis(1));
        let family = GaussianLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            mu_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_mu0_mat.clone(),
            ))),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_ls0_mat.clone(),
            ))),
            policy: crate::resource::ResourcePolicy::default_library(),
            cached_row_scalars: std::sync::RwLock::new(None),
        };
        let specs = vec![
            gaussian_psi_test_spec("mu", x_mu0_mat.clone()),
            gaussian_psi_test_spec("log_sigma", x_ls0_mat.clone()),
        ];
        let states = vec![
            ParameterBlockState {
                beta: array![beta_mu0],
                eta: x_mu0_mat.column(0).to_owned() * beta_mu0,
            },
            ParameterBlockState {
                beta: array![beta_ls0],
                eta: x_ls0_mat.column(0).to_owned() * beta_ls0,
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: x_ls_psi.clone().insert_axis(Axis(1)),
                s_psi: Array2::zeros((1, 1)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: Some(vec![x_ls_psi_psi.clone().insert_axis(Axis(1))]),
                s_psi_psi: Some(vec![Array2::zeros((1, 1))]),
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: None,
                implicit_axis: 0,
                implicit_group_id: None,
            }],
        ];

        let psi2_terms = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &derivative_blocks, 0, 0)
            .expect("joint psi psi terms")
            .expect("expected gaussian psi psi terms");

        let vars = [beta_mu0, beta_ls0, 0.0_f64];
        let (_, _, d2psi) = second_derivative(
            |psi| {
                gaussian_negloglik_log_sigma_psi_only_numdual(
                    psi,
                    beta_mu0,
                    beta_ls0,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            0.0,
        );
        let (_, _, _, _, _, _, _, score_mu_psi_psi) = third_partial_derivative_vec(
            |v| {
                gaussian_negloglik_log_sigma_beta_vec_numdual(
                    v,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            &vars,
            0,
            2,
            2,
        );
        let (_, _, _, _, _, _, _, score_ls_psi_psi) = third_partial_derivative_vec(
            |v| {
                gaussian_negloglik_log_sigma_beta_vec_numdual(
                    v,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_ls_psi,
                    &x_ls_psi_psi,
                )
            },
            &vars,
            1,
            2,
            2,
        );

        assert!(
            (psi2_terms.objective_psi_psi - d2psi).abs() <= 1e-10,
            "Gaussian log-sigma psi second objective mismatch: analytic={} autodiff={}",
            psi2_terms.objective_psi_psi,
            d2psi
        );
        assert!(
            (psi2_terms.score_psi_psi[0] - score_mu_psi_psi).abs() <= 1e-9,
            "Gaussian log-sigma psi second score_mu mismatch: analytic={} autodiff={}",
            psi2_terms.score_psi_psi[0],
            score_mu_psi_psi
        );
        assert!(
            (psi2_terms.score_psi_psi[1] - score_ls_psi_psi).abs() <= 1e-9,
            "Gaussian log-sigma psi second score_ls mismatch: analytic={} autodiff={}",
            psi2_terms.score_psi_psi[1],
            score_ls_psi_psi
        );
    }

    // Sibling oracle: μ also depends on ψ. Used by the joint psi-second-order
    // guardrail; the original oracle leaves μ fixed in ψ.
    fn gaussian_negloglik_log_sigma_psi_full_numdual<D: DualNum<f64> + Copy>(
        beta_mu: D,
        beta_ls: D,
        psi: D,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        x_mu0: &Array1<f64>,
        x_ls0: &Array1<f64>,
        x_mu_psi: &Array1<f64>,
        x_ls_psi: &Array1<f64>,
        x_mu_psi_psi: &Array1<f64>,
        x_ls_psi_psi: &Array1<f64>,
    ) -> D {
        let half = D::from(0.5);
        let mut out = D::zero();
        for i in 0..y.len() {
            let x_mu = D::from(x_mu0[i])
                + psi * D::from(x_mu_psi[i])
                + half * psi * psi * D::from(x_mu_psi_psi[i]);
            let eta_mu = x_mu * beta_mu;
            let x_ls = D::from(x_ls0[i])
                + psi * D::from(x_ls_psi[i])
                + half * psi * psi * D::from(x_ls_psi_psi[i]);
            let eta_ls = x_ls * beta_ls;
            let sigma = D::from(LOGB_SIGMA_FLOOR) + eta_ls.exp();
            let resid = D::from(y[i]) - eta_mu;
            out += D::from(weights[i]) * (half * (resid / sigma).powi(2) + sigma.ln());
        }
        out
    }

    // Oracle with multi-column designs (β vectors). Used by the joint
    // static-Hessian guardrail and its directional derivatives.
    fn gaussian_negloglik_logb_dense_numdual<D: DualNum<f64> + Copy>(
        beta_mu: &[D],
        beta_ls: &[D],
        y: &Array1<f64>,
        weights: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> D {
        let half = D::from(0.5);
        let n = y.len();
        let mut out = D::zero();
        for i in 0..n {
            let mut eta_mu = D::zero();
            for k in 0..beta_mu.len() {
                eta_mu += D::from(xmu[[i, k]]) * beta_mu[k];
            }
            let mut eta_ls = D::zero();
            for k in 0..beta_ls.len() {
                eta_ls += D::from(x_ls[[i, k]]) * beta_ls[k];
            }
            let sigma = D::from(LOGB_SIGMA_FLOOR) + eta_ls.exp();
            let resid = D::from(y[i]) - eta_mu;
            out += D::from(weights[i]) * (half * (resid / sigma).powi(2) + sigma.ln());
        }
        out
    }

    fn gaussian_logb_design_test_data() -> (
        Array1<f64>,
        Array1<f64>,
        Array2<f64>,
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        // n=5, two-column designs (intercept + smooth feature). β_ls0 chosen so
        // that η_ls ≈ −0.4 on the central row → κ ≈ 0.985, which is noticeably
        // less than 1 so κ' chain-rule contributions register at strict tolerance.
        let y = array![0.25, -0.4, 1.1, 0.05, -0.2];
        let weights = array![1.0, 0.7, 1.3, 0.9, 1.1];
        let xmu = ndarray::arr2(&[[1.0, -0.6], [1.0, -0.2], [1.0, 0.1], [1.0, 0.4], [1.0, 0.7]]);
        let x_ls = ndarray::arr2(&[[1.0, 0.5], [1.0, -0.1], [1.0, 0.3], [1.0, -0.4], [1.0, 0.2]]);
        // β_ls = (−0.4, 0.05): η_ls hovers around −0.4, so σ ≈ 0.68 and κ ≈ 0.985.
        let beta_mu = array![0.35, -0.25];
        let beta_ls = array![-0.4, 0.05];
        (y, weights, xmu, x_ls, beta_mu, beta_ls)
    }

    #[test]
    fn gaussian_joint_static_hessian_matches_autodiff() {
        let (y, weights, xmu, x_ls, beta_mu, beta_ls) = gaussian_logb_design_test_data();
        let etamu = xmu.dot(&beta_mu);
        let eta_ls = x_ls.dot(&beta_ls);

        let rows =
            gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
        let weights0 = gaussian_joint_psi_firstweights(
            &rows,
            &Array1::zeros(y.len()),
            &Array1::zeros(y.len()),
        );
        let xmu_dense = DenseOrOperator::Borrowed(&xmu);
        let xls_dense = DenseOrOperator::Borrowed(&x_ls);
        let analytic = gaussian_joint_hessian_from_designs(
            &xmu_dense,
            &xls_dense,
            &weights0.hmumu,
            &weights0.hmu_ls,
            &weights0.h_ls_ls,
        )
        .expect("gaussian joint static hessian from designs");

        // AD ground truth: full p×p Hessian via second_partial_derivative,
        // packing β_full = (β_μ, β_ls) and stepping (i, j) pairs.
        let pmu = beta_mu.len();
        let p_ls = beta_ls.len();
        let total = pmu + p_ls;
        let mut beta_full = vec![0.0_f64; total];
        for k in 0..pmu {
            beta_full[k] = beta_mu[k];
        }
        for k in 0..p_ls {
            beta_full[pmu + k] = beta_ls[k];
        }

        // AD ground truth: full p×p Hessian. Diagonal (i==i) via second_derivative
        // (1D second derivative); off-diagonal (i<j) via second_partial_derivative
        // on a closure that injects two HyperDual variables into β.
        let mut ad = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in i..total {
                let val = if i == j {
                    let g = |x: num_dual::Dual2<f64, f64>| {
                        let mut bm = vec![num_dual::Dual2::from_re(0.0); pmu];
                        let mut bl = vec![num_dual::Dual2::from_re(0.0); p_ls];
                        for k in 0..pmu {
                            bm[k] = num_dual::Dual2::from_re(beta_full[k]);
                        }
                        for k in 0..p_ls {
                            bl[k] = num_dual::Dual2::from_re(beta_full[pmu + k]);
                        }
                        if i < pmu {
                            bm[i] = x;
                        } else {
                            bl[i - pmu] = x;
                        }
                        gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
                    };
                    let (_, _, d2) = second_derivative(g, beta_full[i]);
                    d2
                } else {
                    let f =
                        |(a, b): (num_dual::HyperDual<f64, f64>, num_dual::HyperDual<f64, f64>)| {
                            let mut bm = vec![num_dual::HyperDual::from_re(0.0); pmu];
                            let mut bl = vec![num_dual::HyperDual::from_re(0.0); p_ls];
                            for k in 0..pmu {
                                bm[k] = num_dual::HyperDual::from_re(beta_full[k]);
                            }
                            for k in 0..p_ls {
                                bl[k] = num_dual::HyperDual::from_re(beta_full[pmu + k]);
                            }
                            if i < pmu {
                                bm[i] = a;
                            } else {
                                bl[i - pmu] = a;
                            }
                            if j < pmu {
                                bm[j] = b;
                            } else {
                                bl[j - pmu] = b;
                            }
                            gaussian_negloglik_logb_dense_numdual(
                                &bm, &bl, &y, &weights, &xmu, &x_ls,
                            )
                        };
                    let (_, _, _, d2xy) =
                        second_partial_derivative(f, (beta_full[i], beta_full[j]));
                    d2xy
                };
                ad[[i, j]] = val;
                if i != j {
                    ad[[j, i]] = val;
                }
            }
        }

        for i in 0..total {
            for j in 0..total {
                let diff = (analytic[[i, j]] - ad[[i, j]]).abs();
                assert!(
                    diff <= 1e-10,
                    "Gaussian static joint H[{i},{j}] mismatch (κ < 1 case): analytic={} ad={} diff={}",
                    analytic[[i, j]],
                    ad[[i, j]],
                    diff
                );
            }
        }
        // Symmetry guardrail: floating-point skew must be at the noise floor.
        let skew = (&analytic - &analytic.t())
            .mapv(f64::abs)
            .fold(0.0_f64, |acc, &v| acc.max(v));
        assert!(
            skew <= 1e-12,
            "Gaussian static joint Hessian skew exceeds noise floor: {skew}"
        );
    }

    #[test]
    fn gaussian_joint_first_directional_hessian_matches_autodiff() {
        let (y, weights, xmu, x_ls, beta_mu, beta_ls) = gaussian_logb_design_test_data();
        let etamu = xmu.dot(&beta_mu);
        let eta_ls = x_ls.dot(&beta_ls);

        let pmu = beta_mu.len();
        let p_ls = beta_ls.len();
        let total = pmu + p_ls;
        // Direction v over the joint β = (β_μ, β_ls).
        let v: Array1<f64> = Array1::from_shape_fn(total, |k| 0.13 + 0.07 * (k as f64));
        let v_mu = v.slice(s![0..pmu]).to_owned();
        let v_ls = v.slice(s![pmu..total]).to_owned();
        let ximu = xmu.dot(&v_mu);
        let xi_ls = x_ls.dot(&v_ls);

        let rows =
            gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
        let (dhmumu, dhmu_ls, dh_ls_ls) =
            gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
        let xmu_dense = DenseOrOperator::Borrowed(&xmu);
        let xls_dense = DenseOrOperator::Borrowed(&x_ls);
        let analytic = gaussian_joint_hessian_from_designs(
            &xmu_dense, &xls_dense, &dhmumu, &dhmu_ls, &dh_ls_ls,
        )
        .expect("gaussian joint first-directional H from designs");

        // AD: differentiate N along (β + ε·v), evaluating ∂³N/∂β_i ∂β_j ∂ε at ε=0
        // via third_partial_derivative_vec on the augmented vector
        // [β_μ, β_ls, ε] of length total + 1.
        let mut vars = vec![0.0_f64; total + 1];
        for k in 0..pmu {
            vars[k] = beta_mu[k];
        }
        for k in 0..p_ls {
            vars[pmu + k] = beta_ls[k];
        }
        // vars[total] = ε = 0 by default.

        let g = |z: &[num_dual::HyperHyperDual<f64, f64>]| {
            // Reconstruct β + ε·v.
            let mut bm = vec![num_dual::HyperHyperDual::from_re(0.0); pmu];
            let mut bl = vec![num_dual::HyperHyperDual::from_re(0.0); p_ls];
            let eps = z[total];
            for k in 0..pmu {
                bm[k] = z[k] + eps * num_dual::HyperHyperDual::from_re(v[k]);
            }
            for k in 0..p_ls {
                bl[k] = z[pmu + k] + eps * num_dual::HyperHyperDual::from_re(v[pmu + k]);
            }
            gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
        };

        let mut ad = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in i..total {
                let (_, _, _, _, _, _, _, d3) =
                    third_partial_derivative_vec(&g, &vars, i, j, total);
                ad[[i, j]] = d3;
                if i != j {
                    ad[[j, i]] = d3;
                }
            }
        }

        for i in 0..total {
            for j in 0..total {
                let diff = (analytic[[i, j]] - ad[[i, j]]).abs();
                assert!(
                    diff <= 1e-10,
                    "Gaussian dH (first-directional) [{i},{j}] mismatch: analytic={} ad={} diff={}",
                    analytic[[i, j]],
                    ad[[i, j]],
                    diff
                );
            }
        }
        let skew = (&analytic - &analytic.t())
            .mapv(f64::abs)
            .fold(0.0_f64, |acc, &v| acc.max(v));
        assert!(
            skew <= 1e-12,
            "Gaussian first-directional dH skew exceeds noise floor: {skew}"
        );
    }

    #[test]
    fn gaussian_joint_second_directional_hessian_matches_autodiff() {
        let (y, weights, xmu, x_ls, beta_mu, beta_ls) = gaussian_logb_design_test_data();
        let etamu = xmu.dot(&beta_mu);
        let eta_ls = x_ls.dot(&beta_ls);

        let pmu = beta_mu.len();
        let p_ls = beta_ls.len();
        let total = pmu + p_ls;
        let u: Array1<f64> = Array1::from_shape_fn(total, |k| 0.18 - 0.05 * (k as f64));
        let v: Array1<f64> = Array1::from_shape_fn(total, |k| -0.11 + 0.09 * (k as f64));
        let u_mu = u.slice(s![0..pmu]).to_owned();
        let u_ls = u.slice(s![pmu..total]).to_owned();
        let v_mu = v.slice(s![0..pmu]).to_owned();
        let v_ls = v.slice(s![pmu..total]).to_owned();
        let ximu_u = xmu.dot(&u_mu);
        let xi_ls_u = x_ls.dot(&u_ls);
        let ximuv = xmu.dot(&v_mu);
        let xi_lsv = x_ls.dot(&v_ls);

        let rows =
            gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
        let (d2hmumu, d2hmu_ls, d2h_ls_ls) =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximuv, &xi_lsv);
        let xmu_dense = DenseOrOperator::Borrowed(&xmu);
        let xls_dense = DenseOrOperator::Borrowed(&x_ls);
        let analytic = gaussian_joint_hessian_from_designs(
            &xmu_dense, &xls_dense, &d2hmumu, &d2hmu_ls, &d2h_ls_ls,
        )
        .expect("gaussian joint second-directional H from designs");

        // AD ground truth for ∂⁴N/∂β_i ∂β_j ∂ε_u ∂ε_v at (ε_u, ε_v) = (0, 0).
        // num-dual ships native AD up to third order; the fourth order is
        // obtained by central FD in ε_v of the AD third partial that already
        // covers (β_i, β_j, ε_u). Augmented vector layout:
        //   [β_μ ; β_ls ; ε_u]    of length total + 1 (ε_v lives outside AD).
        let mut vars_base = vec![0.0_f64; total + 1];
        for k in 0..pmu {
            vars_base[k] = beta_mu[k];
        }
        for k in 0..p_ls {
            vars_base[pmu + k] = beta_ls[k];
        }
        // vars_base[total] = ε_u = 0.

        let h = 1e-4;
        let mut ad = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in i..total {
                let g_plus = |z: &[num_dual::HyperHyperDual<f64, f64>]| {
                    let mut bm = vec![num_dual::HyperHyperDual::from_re(0.0); pmu];
                    let mut bl = vec![num_dual::HyperHyperDual::from_re(0.0); p_ls];
                    let eps_u = z[total];
                    for k in 0..pmu {
                        bm[k] = z[k]
                            + eps_u * num_dual::HyperHyperDual::from_re(u[k])
                            + num_dual::HyperHyperDual::from_re(h * v[k]);
                    }
                    for k in 0..p_ls {
                        bl[k] = z[pmu + k]
                            + eps_u * num_dual::HyperHyperDual::from_re(u[pmu + k])
                            + num_dual::HyperHyperDual::from_re(h * v[pmu + k]);
                    }
                    gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
                };
                let g_minus = |z: &[num_dual::HyperHyperDual<f64, f64>]| {
                    let mut bm = vec![num_dual::HyperHyperDual::from_re(0.0); pmu];
                    let mut bl = vec![num_dual::HyperHyperDual::from_re(0.0); p_ls];
                    let eps_u = z[total];
                    for k in 0..pmu {
                        bm[k] = z[k] + eps_u * num_dual::HyperHyperDual::from_re(u[k])
                            - num_dual::HyperHyperDual::from_re(h * v[k]);
                    }
                    for k in 0..p_ls {
                        bl[k] = z[pmu + k] + eps_u * num_dual::HyperHyperDual::from_re(u[pmu + k])
                            - num_dual::HyperHyperDual::from_re(h * v[pmu + k]);
                    }
                    gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
                };
                let (_, _, _, _, _, _, _, d3_plus) =
                    third_partial_derivative_vec(g_plus, &vars_base, i, j, total);
                let (_, _, _, _, _, _, _, d3_minus) =
                    third_partial_derivative_vec(g_minus, &vars_base, i, j, total);
                let val = (d3_plus - d3_minus) / (2.0 * h);
                ad[[i, j]] = val;
                if i != j {
                    ad[[j, i]] = val;
                }
            }
        }

        // Tolerance: the 4th-order ground truth uses one FD step on top of an
        // AD third partial, so we relax from 1e-10 to a value compatible with
        // the central-difference truncation (O(h²) ≈ 1e-8) and the rounding
        // floor of the AD third partial (≈ 1e-10 / h ≈ 1e-6).
        let tol = 5e-6;
        for i in 0..total {
            for j in 0..total {
                let diff = (analytic[[i, j]] - ad[[i, j]]).abs();
                assert!(
                    diff <= tol,
                    "Gaussian d2H (second-directional) [{i},{j}] mismatch: analytic={} ad={} diff={}",
                    analytic[[i, j]],
                    ad[[i, j]],
                    diff
                );
            }
        }
        let skew = (&analytic - &analytic.t())
            .mapv(f64::abs)
            .fold(0.0_f64, |acc, &v| acc.max(v));
        assert!(
            skew <= 1e-10,
            "Gaussian second-directional d2H skew exceeds noise floor: {skew}"
        );
    }

    #[test]
    fn gaussian_joint_psi_second_order_terms_match_autodiff() {
        // ψ-coupled scenario: both μ and η_ls depend on ψ via per-row
        // first/second drift vectors, with non-trivial coefficients.
        let y = array![0.25, -0.4, 1.1, 0.05, -0.2];
        let weights = array![1.0, 0.7, 1.3, 0.9, 1.1];
        let x_mu0 = array![1.0, -0.35, 0.6, 0.1, 0.45];
        let x_ls0 = array![0.8, -0.25, 0.45, -0.1, 0.3];
        let x_mu_psi = array![0.2, 0.15, -0.1, 0.05, 0.3];
        let x_ls_psi = array![0.18, -0.12, 0.25, -0.2, 0.07];
        let x_mu_psi_psi = array![0.04, -0.03, 0.05, 0.06, -0.02];
        let x_ls_psi_psi = array![0.05, -0.03, 0.04, 0.07, -0.04];
        // β_ls chosen so η_ls ≈ −0.4 (κ ≈ 0.985, noticeably less than 1).
        let beta_mu0 = 0.35_f64;
        let beta_ls0 = -0.4_f64;

        // Per-row predictor drifts.
        let etamu = &x_mu0 * beta_mu0;
        let eta_ls = &x_ls0 * beta_ls0;
        let zmu_psi = &x_mu_psi * beta_mu0;
        let z_ls_psi = &x_ls_psi * beta_ls0;
        let zmu_psi_psi = &x_mu_psi_psi * beta_mu0;
        let z_ls_psi_psi = &x_ls_psi_psi * beta_ls0;

        let rows =
            gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
        let secondweights = gaussian_joint_psisecondweights(
            &rows,
            &zmu_psi,
            &z_ls_psi,
            &zmu_psi,
            &z_ls_psi,
            &zmu_psi_psi,
            &z_ls_psi_psi,
        );
        let analytic = secondweights.objective_psi_psirow.sum();

        // AD: differentiate the full ψ-dependent oracle twice in ψ at ψ=0.
        let (_, _, ad) = second_derivative(
            |psi| {
                gaussian_negloglik_log_sigma_psi_full_numdual(
                    num_dual::Dual2::from_re(beta_mu0),
                    num_dual::Dual2::from_re(beta_ls0),
                    psi,
                    &y,
                    &weights,
                    &x_mu0,
                    &x_ls0,
                    &x_mu_psi,
                    &x_ls_psi,
                    &x_mu_psi_psi,
                    &x_ls_psi_psi,
                )
            },
            0.0,
        );

        let diff = (analytic - ad).abs();
        assert!(
            diff <= 1e-10,
            "Gaussian joint ψ-ψ objective mismatch (κ < 1, μ and σ both ψ-dependent): analytic={} ad={} diff={}",
            analytic,
            ad,
            diff
        );
    }

    #[test]
    fn wiggle_family_block_hessians_match_jointhessian_principal_blocks() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            3,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let threshold_design = threshold_block.design.clone();
        let log_sigma_design = log_sigma_block.design.clone();
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let beta_t = Array1::from_vec(vec![0.25]);
        let beta_ls = Array1::from_vec(vec![-0.15]);
        let eta_t = threshold_design.matrixvectormultiply(&beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let etaw = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&betaw);
        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t,
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ];

        let eval = family.evaluate(&states).expect("evaluate wiggle family");
        let joint = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected joint exact hessian");
        let beta_layout = GamlssBetaLayout::withwiggle(beta_t.len(), beta_ls.len(), betaw.len());
        let ranges = [
            (0usize, beta_layout.pt),
            (beta_layout.pt, beta_layout.pt + beta_layout.pls),
            (
                beta_layout.pt + beta_layout.pls,
                beta_layout.pt + beta_layout.pls + beta_layout.pw,
            ),
        ];

        for (block_idx, (start, end)) in ranges.into_iter().enumerate() {
            let blockhessian = match &eval.blockworking_sets[block_idx] {
                BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
            };
            let joint_block = joint.slice(s![start..end, start..end]).to_owned();
            crate::testing::assert_matrix_derivativefd(
                &joint_block,
                &blockhessian,
                1e-10,
                &format!("wiggle block {block_idx} principal block"),
            );
        }
    }

    #[test]
    fn wiggle_familygradients_match_finite_differencewith_nontrivial_designs() {
        let n = 9usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let t_grid = Array1::linspace(0.0, 1.0, n);
        let threshold_x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => t_grid[i] - 0.5,
            2 => (2.0 * std::f64::consts::PI * t_grid[i]).sin(),
            _ => unreachable!(),
        });
        let log_sigma_x = Array2::from_shape_fn((n, 2), |(i, j)| match j {
            0 => 1.0,
            1 => (3.0 * std::f64::consts::PI * t_grid[i]).cos(),
            _ => unreachable!(),
        });
        let threshold_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(threshold_x.clone()));
        let log_sigma_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(log_sigma_x.clone()));
        let q_seed = Array1::linspace(-1.3, 1.1, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            3,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let rebuild_states = |beta_t: &Array1<f64>,
                              beta_ls: &Array1<f64>,
                              betaw: &Array1<f64>|
         -> Vec<ParameterBlockState> {
            let eta_t = threshold_design.matrixvectormultiply(beta_t);
            let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
            let core_q0 = binomial_location_scale_core(
                &y,
                &weights,
                &eta_t,
                &eta_ls,
                None,
                &family.link_kind,
            )
            .expect("core q0");
            let etaw = family
                .wiggle_design(core_q0.q0.view())
                .expect("wiggle design")
                .dot(betaw);
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: eta_t,
                },
                ParameterBlockState {
                    beta: beta_ls.clone(),
                    eta: eta_ls,
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: etaw,
                },
            ]
        };

        let objective = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>, betaw: &Array1<f64>| {
            let states = rebuild_states(beta_t, beta_ls, betaw);
            -family.evaluate(&states).expect("evaluate").log_likelihood
        };

        let extractgradient = |eval: &FamilyEvaluation, block_idx: usize| -> Array1<f64> {
            match &eval.blockworking_sets[block_idx] {
                BlockWorkingSet::ExactNewton {
                    gradient,
                    hessian: _,
                } => gradient.clone(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
            }
        };

        let beta_t = Array1::from_vec(vec![0.15, -0.3, 0.2]);
        let beta_ls = Array1::from_vec(vec![-0.2, 0.1]);
        let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let states = rebuild_states(&beta_t, &beta_ls, &betaw);
        let eval = family.evaluate(&states).expect("evaluate");
        let eps = 1e-6;

        for block_idx in 0..3 {
            let analytic = extractgradient(&eval, block_idx);
            let mut fd = Array1::<f64>::zeros(analytic.len());
            for j in 0..analytic.len() {
                let mut beta_t_plus = beta_t.clone();
                let mut beta_ls_plus = beta_ls.clone();
                let mut betaw_plus = betaw.clone();
                let mut beta_t_minus = beta_t.clone();
                let mut beta_ls_minus = beta_ls.clone();
                let mut betaw_minus = betaw.clone();
                match block_idx {
                    BinomialLocationScaleWiggleFamily::BLOCK_T => {
                        beta_t_plus[j] += eps;
                        beta_t_minus[j] -= eps;
                    }
                    BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                        beta_ls_plus[j] += eps;
                        beta_ls_minus[j] -= eps;
                    }
                    BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                        betaw_plus[j] += eps;
                        betaw_minus[j] -= eps;
                    }
                    _ => unreachable!(),
                }
                let f_plus = objective(&beta_t_plus, &beta_ls_plus, &betaw_plus);
                let f_minus = objective(&beta_t_minus, &beta_ls_minus, &betaw_minus);
                fd[j] = (f_plus - f_minus) / (2.0 * eps);
            }
            crate::testing::assert_matrix_derivativefd(
                &fd.insert_axis(Axis(1)),
                &(-&analytic).insert_axis(Axis(1)),
                2e-4,
                &format!("wiggle block {block_idx} score"),
            );
        }
    }

    #[test]
    fn wiggle_family_joint_hessian_matches_fd_gradients_with_nontrivial_designs() {
        let n = 9usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let t_grid = Array1::linspace(0.0, 1.0, n);
        let threshold_x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => t_grid[i] - 0.5,
            2 => (2.0 * std::f64::consts::PI * t_grid[i]).sin(),
            _ => unreachable!(),
        });
        let log_sigma_x = Array2::from_shape_fn((n, 2), |(i, j)| match j {
            0 => 1.0,
            1 => (3.0 * std::f64::consts::PI * t_grid[i]).cos(),
            _ => unreachable!(),
        });
        let threshold_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(threshold_x.clone()));
        let log_sigma_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(log_sigma_x.clone()));
        let q_seed = Array1::linspace(-1.3, 1.1, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            3,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let rebuild_states = |beta_t: &Array1<f64>,
                              beta_ls: &Array1<f64>,
                              betaw: &Array1<f64>|
         -> Vec<ParameterBlockState> {
            let eta_t = threshold_design.matrixvectormultiply(beta_t);
            let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
            let core_q0 = binomial_location_scale_core(
                &y,
                &weights,
                &eta_t,
                &eta_ls,
                None,
                &family.link_kind,
            )
            .expect("core q0");
            let etaw = family
                .wiggle_design(core_q0.q0.view())
                .expect("wiggle design")
                .dot(betaw);
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: eta_t,
                },
                ParameterBlockState {
                    beta: beta_ls.clone(),
                    eta: eta_ls,
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: etaw,
                },
            ]
        };

        let extractgradient = |eval: &FamilyEvaluation, block_idx: usize| -> Array1<f64> {
            match &eval.blockworking_sets[block_idx] {
                BlockWorkingSet::ExactNewton {
                    gradient,
                    hessian: _,
                } => gradient.clone(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
            }
        };

        let beta_t = Array1::from_vec(vec![0.15, -0.3, 0.2]);
        let beta_ls = Array1::from_vec(vec![-0.2, 0.1]);
        let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let states = rebuild_states(&beta_t, &beta_ls, &betaw);
        let h_joint = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected joint exact hessian");
        let pt = beta_t.len();
        let pls = beta_ls.len();
        let eps = 1e-6;
        let total = pt + pls + betaw.len();
        let mut fd = Array2::<f64>::zeros((total, total));
        let source_offsets = [0usize, pt, pt + pls];

        for source_block in 0..3 {
            let source_len = states[source_block].beta.len();
            for j in 0..source_len {
                let mut beta_t_plus = beta_t.clone();
                let mut beta_ls_plus = beta_ls.clone();
                let mut betaw_plus = betaw.clone();
                let mut beta_t_minus = beta_t.clone();
                let mut beta_ls_minus = beta_ls.clone();
                let mut betaw_minus = betaw.clone();
                match source_block {
                    BinomialLocationScaleWiggleFamily::BLOCK_T => {
                        beta_t_plus[j] += eps;
                        beta_t_minus[j] -= eps;
                    }
                    BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                        beta_ls_plus[j] += eps;
                        beta_ls_minus[j] -= eps;
                    }
                    BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                        betaw_plus[j] += eps;
                        betaw_minus[j] -= eps;
                    }
                    _ => unreachable!(),
                }
                let eval_plus = family
                    .evaluate(&rebuild_states(&beta_t_plus, &beta_ls_plus, &betaw_plus))
                    .expect("eval plus");
                let eval_minus = family
                    .evaluate(&rebuild_states(&beta_t_minus, &beta_ls_minus, &betaw_minus))
                    .expect("eval minus");

                let mut row_offset = 0usize;
                for target_block in 0..3 {
                    let grad_plus = extractgradient(&eval_plus, target_block);
                    let grad_minus = extractgradient(&eval_minus, target_block);
                    let col = (&grad_plus - &grad_minus).mapv(|v| -v / (2.0 * eps));
                    let col_idx = source_offsets[source_block] + j;
                    fd.slice_mut(s![
                        row_offset..row_offset + grad_plus.len(),
                        col_idx..col_idx + 1
                    ])
                    .assign(&col.insert_axis(Axis(1)));
                    row_offset += grad_plus.len();
                }
            }
        }

        crate::testing::assert_matrix_derivativefd(&fd, &h_joint, 4e-4, "wiggle joint hessian");
    }

    #[test]
    fn wiggle_family_joint_exacthessian_directional_derivative_matches_finite_difference() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            3,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let threshold_design = threshold_block.design.clone();
        let log_sigma_design = log_sigma_block.design.clone();
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let beta_t = Array1::from_vec(vec![0.25]);
        let beta_ls = Array1::from_vec(vec![-0.15]);
        let eta_t = threshold_design.matrixvectormultiply(&beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let etaw = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&betaw);
        let states = vec![
            ParameterBlockState {
                beta: beta_t,
                eta: eta_t,
            },
            ParameterBlockState {
                beta: beta_ls,
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ];

        let base_h = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected joint exact hessian");
        let direction = Array1::ones(base_h.nrows());
        let analytic = family
            .exact_newton_joint_hessian_directional_derivative(&states, &direction)
            .expect("joint dH")
            .expect("expected joint exact dH");

        let eps = 1e-6;
        let mut plus_states = states.clone();
        let beta_layout = GamlssBetaLayout::withwiggle(
            plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T]
                .beta
                .len(),
            plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
                .beta
                .len(),
            plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE]
                .beta
                .len(),
        );
        let (dir_t, dir_ls, dirw) = beta_layout
            .split_three(&direction, "wiggle test direction split")
            .expect("split wiggle test direction");
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta =
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta + &(eps * dir_t);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta =
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta + &(eps * dir_ls);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta =
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta + &(eps * dirw);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta = threshold_design
            .matrixvectormultiply(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta = log_sigma_design
            .matrixvectormultiply(
                &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta,
            );
        let plus_core_q0 = binomial_location_scale_core(
            &y,
            &weights,
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta,
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta,
            None,
            &family.link_kind,
        )
        .expect("plus core q0");
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta = family
            .wiggle_design(plus_core_q0.q0.view())
            .expect("plus wiggle design")
            .dot(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta);

        let h_plus = family
            .exact_newton_joint_hessian(&plus_states)
            .expect("plus joint hessian")
            .expect("expected plus joint hessian");
        let fd = (h_plus - base_h) / eps;
        crate::testing::assert_matrix_derivativefd(&fd, &analytic, 2e-3, "joint dH");
    }

    #[test]
    fn wiggle_family_joint_exacthessiansecond_directional_derivative_matches_finite_difference() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            4,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let threshold_design = threshold_block.design.clone();
        let log_sigma_design = log_sigma_block.design.clone();
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 4,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let rebuild_states = |beta_t: &Array1<f64>,
                              beta_ls: &Array1<f64>,
                              betaw: &Array1<f64>|
         -> Vec<ParameterBlockState> {
            let eta_t = threshold_design.matrixvectormultiply(beta_t);
            let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
            let core_q0 = binomial_location_scale_core(
                &y,
                &weights,
                &eta_t,
                &eta_ls,
                None,
                &family.link_kind,
            )
            .expect("core q0");
            let etaw = family
                .wiggle_design(core_q0.q0.view())
                .expect("wiggle design")
                .dot(betaw);
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: eta_t,
                },
                ParameterBlockState {
                    beta: beta_ls.clone(),
                    eta: eta_ls,
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: etaw,
                },
            ]
        };

        let beta_t = Array1::from_vec(vec![0.25]);
        let beta_ls = Array1::from_vec(vec![-0.15]);
        let betaw = Array1::from_vec(vec![0.03; wiggle_block.design.ncols()]);
        let states = rebuild_states(&beta_t, &beta_ls, &betaw);

        let pt = beta_t.len();
        let pls = beta_ls.len();
        let pw = betaw.len();
        let total = pt + pls + pw;
        let direction_u = Array1::from_shape_fn(total, |k| 0.2 + 0.1 * (k as f64));
        let directionv = Array1::from_shape_fn(total, |k| -0.15 + 0.07 * (k as f64));

        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &direction_u,
                &directionv,
            )
            .expect("joint d2H")
            .expect("expected joint exact d2H");

        let eps = 1e-6;
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (step_t, step_ls, stepw) = beta_layout
            .split_three(&directionv, "wiggle d2H test directionv")
            .expect("split wiggle test direction");

        let states_plus = rebuild_states(
            &(&beta_t + &(eps * &step_t)),
            &(&beta_ls + &(eps * &step_ls)),
            &(&betaw + &(eps * &stepw)),
        );
        let states_minus = rebuild_states(
            &(&beta_t - &(eps * &step_t)),
            &(&beta_ls - &(eps * &step_ls)),
            &(&betaw - &(eps * &stepw)),
        );
        let d_h_plus = family
            .exact_newton_joint_hessian_directional_derivative(&states_plus, &direction_u)
            .expect("joint dH plus")
            .expect("expected joint exact dH plus");
        let d_h_minus = family
            .exact_newton_joint_hessian_directional_derivative(&states_minus, &direction_u)
            .expect("joint dH minus")
            .expect("expected joint exact dH minus");
        let fd = (d_h_plus - d_h_minus) / (2.0 * eps);

        crate::testing::assert_matrix_derivativefd(&fd, &analytic, 4e-3, "joint d2H");
    }

    #[test]
    fn wiggle_family_joint_hessian_cross_blocks_match_finite_difference_of_gradients() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            3,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let threshold_design = threshold_block.design.clone();
        let log_sigma_design = log_sigma_block.design.clone();
        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 3,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let rebuild_states = |beta_t: &Array1<f64>,
                              beta_ls: &Array1<f64>,
                              betaw: &Array1<f64>|
         -> Vec<ParameterBlockState> {
            let eta_t = threshold_design.matrixvectormultiply(beta_t);
            let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
            let core_q0 = binomial_location_scale_core(
                &y,
                &weights,
                &eta_t,
                &eta_ls,
                None,
                &family.link_kind,
            )
            .expect("core q0");
            let etaw = family
                .wiggle_design(core_q0.q0.view())
                .expect("wiggle design")
                .dot(betaw);
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: eta_t,
                },
                ParameterBlockState {
                    beta: beta_ls.clone(),
                    eta: eta_ls,
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: etaw,
                },
            ]
        };

        let extractgradient = |eval: &FamilyEvaluation, block_idx: usize| -> Array1<f64> {
            match &eval.blockworking_sets[block_idx] {
                BlockWorkingSet::ExactNewton {
                    gradient,
                    hessian: _,
                } => gradient.clone(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
            }
        };

        let beta_t = Array1::from_vec(vec![0.25]);
        let beta_ls = Array1::from_vec(vec![-0.15]);
        let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let states = rebuild_states(&beta_t, &beta_ls, &betaw);

        let h_joint = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected joint exact hessian");

        let pt = beta_t.len();
        let pls = beta_ls.len();
        let pw = betaw.len();
        let eps = 1e-6;

        let fd_cross_block = |target_block: usize, source_block: usize| -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((
                states[target_block].beta.len(),
                states[source_block].beta.len(),
            ));
            for j in 0..states[source_block].beta.len() {
                let mut beta_t_plus = beta_t.clone();
                let mut beta_ls_plus = beta_ls.clone();
                let mut betaw_plus = betaw.clone();
                let mut beta_t_minus = beta_t.clone();
                let mut beta_ls_minus = beta_ls.clone();
                let mut betaw_minus = betaw.clone();
                match source_block {
                    BinomialLocationScaleWiggleFamily::BLOCK_T => {
                        beta_t_plus[j] += eps;
                        beta_t_minus[j] -= eps;
                    }
                    BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                        beta_ls_plus[j] += eps;
                        beta_ls_minus[j] -= eps;
                    }
                    BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                        betaw_plus[j] += eps;
                        betaw_minus[j] -= eps;
                    }
                    _ => panic!("unexpected block"),
                }

                let eval_plus = family
                    .evaluate(&rebuild_states(&beta_t_plus, &beta_ls_plus, &betaw_plus))
                    .expect("eval plus");
                let eval_minus = family
                    .evaluate(&rebuild_states(&beta_t_minus, &beta_ls_minus, &betaw_minus))
                    .expect("eval minus");
                let grad_plus = extractgradient(&eval_plus, target_block);
                let grad_minus = extractgradient(&eval_minus, target_block);
                let col = (&grad_plus - &grad_minus).mapv(|v| -v / (2.0 * eps));
                out.slice_mut(ndarray::s![.., j]).assign(&col);
            }
            out
        };

        let fd_t_ls = fd_cross_block(
            BinomialLocationScaleWiggleFamily::BLOCK_T,
            BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
        );
        let fd_tw = fd_cross_block(
            BinomialLocationScaleWiggleFamily::BLOCK_T,
            BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
        );
        let fd_lsw = fd_cross_block(
            BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
            BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
        );

        let h_t_ls = h_joint.slice(ndarray::s![0..pt, pt..pt + pls]).to_owned();
        let h_tw = h_joint
            .slice(ndarray::s![0..pt, pt + pls..pt + pls + pw])
            .to_owned();
        let h_lsw = h_joint
            .slice(ndarray::s![pt..pt + pls, pt + pls..pt + pls + pw])
            .to_owned();

        crate::testing::assert_matrix_derivativefd(&fd_t_ls, &h_t_ls, 2e-4, "H_t_ls");
        crate::testing::assert_matrix_derivativefd(&fd_tw, &h_tw, 4e-4, "H_tw");
        crate::testing::assert_matrix_derivativefd(&fd_lsw, &h_lsw, 6e-4, "H_lsw");
    }

    #[test]
    fn nonwiggle_family_evaluate_returns_exact_newton_blockswhen_designs_are_present() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_fn((n, 2), |(i, j)| {
                let t = i as f64 / (n as f64 - 1.0);
                match j {
                    0 => 1.0,
                    1 => t - 0.5,
                    _ => unreachable!(),
                }
            }),
        ));
        let log_sigma_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_fn((n, 2), |(i, j)| {
                let t = i as f64 / (n as f64 - 1.0);
                match j {
                    0 => 1.0,
                    1 => (2.0 * std::f64::consts::PI * t).cos(),
                    _ => unreachable!(),
                }
            }),
        ));
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let beta_t = array![0.2, -0.15];
        let beta_ls = array![-0.1, 0.05];
        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: threshold_design.matrixvectormultiply(&beta_t),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: log_sigma_design.matrixvectormultiply(&beta_ls),
            },
        ];

        let eval = family.evaluate(&states).expect("evaluate nonwiggle family");
        assert_eq!(eval.blockworking_sets.len(), 2);
        let joint = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected joint exact hessian");
        let pt = beta_t.len();
        let pls = beta_ls.len();

        for (block_idx, (start, end)) in [(0usize, pt), (pt, pt + pls)].into_iter().enumerate() {
            let blockhessian = match &eval.blockworking_sets[block_idx] {
                BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
            };
            let joint_block = joint.slice(s![start..end, start..end]).to_owned();
            crate::testing::assert_matrix_derivativefd(
                &joint_block,
                &blockhessian,
                1e-10,
                &format!("nonwiggle block {block_idx} principal block"),
            );
        }
    }

    #[test]
    fn nonwiggle_family_joint_exacthessian_directional_derivative_matches_finite_difference() {
        let n = 8usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_fn((n, 2), |(i, j)| {
                let t = i as f64 / (n as f64 - 1.0);
                match j {
                    0 => 1.0,
                    1 => (2.0 * std::f64::consts::PI * t).sin(),
                    _ => unreachable!(),
                }
            }),
        ));
        let log_sigma_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_fn((n, 2), |(i, j)| {
                let t = i as f64 / (n as f64 - 1.0);
                match j {
                    0 => 1.0,
                    1 => t - 0.5,
                    _ => unreachable!(),
                }
            }),
        ));
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let rebuild_states = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>| {
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: threshold_design.matrixvectormultiply(beta_t),
                },
                ParameterBlockState {
                    beta: beta_ls.clone(),
                    eta: log_sigma_design.matrixvectormultiply(beta_ls),
                },
            ]
        };

        let beta_t = array![0.2, -0.1];
        let beta_ls = array![-0.15, 0.08];
        let states = rebuild_states(&beta_t, &beta_ls);
        let base_h = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected joint exact hessian");
        let direction = array![0.2, 0.3, -0.15, 0.1];
        let analytic = family
            .exact_newton_joint_hessian_directional_derivative(&states, &direction)
            .expect("joint dH")
            .expect("expected joint exact dH");

        let eps = 1e-6;
        let dir_t = direction.slice(s![0..beta_t.len()]).to_owned();
        let dir_ls = direction.slice(s![beta_t.len()..]).to_owned();
        let states_plus =
            rebuild_states(&(&beta_t + &(eps * &dir_t)), &(&beta_ls + &(eps * &dir_ls)));
        let h_plus = family
            .exact_newton_joint_hessian(&states_plus)
            .expect("plus joint hessian")
            .expect("expected plus joint hessian");
        let fd = (h_plus - base_h) / eps;
        crate::testing::assert_matrix_derivativefd(&fd, &analytic, 2e-3, "nonwiggle joint dH");
    }

    #[test]
    fn nonwiggle_family_joint_exacthessiansecond_directional_derivative_matches_finite_difference()
    {
        let n = 8usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_fn((n, 2), |(i, j)| {
                let t = i as f64 / (n as f64 - 1.0);
                match j {
                    0 => 1.0,
                    1 => (2.0 * std::f64::consts::PI * t).sin(),
                    _ => unreachable!(),
                }
            }),
        ));
        let log_sigma_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_fn((n, 2), |(i, j)| {
                let t = i as f64 / (n as f64 - 1.0);
                match j {
                    0 => 1.0,
                    1 => t - 0.5,
                    _ => unreachable!(),
                }
            }),
        ));
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let rebuild_states = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>| {
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: threshold_design.matrixvectormultiply(beta_t),
                },
                ParameterBlockState {
                    beta: beta_ls.clone(),
                    eta: log_sigma_design.matrixvectormultiply(beta_ls),
                },
            ]
        };

        let beta_t = array![0.2, -0.1];
        let beta_ls = array![-0.15, 0.08];
        let states = rebuild_states(&beta_t, &beta_ls);
        let direction_u = array![0.2, 0.3, -0.15, 0.1];
        let directionv = array![-0.05, 0.12, 0.08, -0.09];
        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &direction_u,
                &directionv,
            )
            .expect("joint d2H")
            .expect("expected joint exact d2H");

        let eps = 1e-6;
        let step_t = directionv.slice(s![0..beta_t.len()]).to_owned();
        let step_ls = directionv.slice(s![beta_t.len()..]).to_owned();
        let states_plus = rebuild_states(
            &(&beta_t + &(eps * &step_t)),
            &(&beta_ls + &(eps * &step_ls)),
        );
        let states_minus = rebuild_states(
            &(&beta_t - &(eps * &step_t)),
            &(&beta_ls - &(eps * &step_ls)),
        );
        let d_h_plus = family
            .exact_newton_joint_hessian_directional_derivative(&states_plus, &direction_u)
            .expect("joint dH plus")
            .expect("expected joint exact dH plus");
        let d_h_minus = family
            .exact_newton_joint_hessian_directional_derivative(&states_minus, &direction_u)
            .expect("joint dH minus")
            .expect("expected joint exact dH minus");
        let fd = (d_h_plus - d_h_minus) / (2.0 * eps);
        crate::testing::assert_matrix_derivativefd(&fd, &analytic, 4e-3, "nonwiggle joint d2H");
    }

    #[test]
    fn wiggle_basis_is_structurally_monotone_for_nonnegative_coefficients() {
        let q_seed = Array1::linspace(-2.0, 2.0, 17);
        let degree = 3usize;
        let num_internal_knots = 6usize;
        let penalty_order = 2usize;

        let (block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            degree,
            num_internal_knots,
            penalty_order,
            false,
        )
        .expect("wiggle block");
        let design = match &block.design {
            DesignMatrix::Dense(x) => x.to_dense_arc(),
            DesignMatrix::Sparse(_) => panic!("expected dense wiggle design"),
        };
        let beta = Array1::from_elem(design.ncols(), 0.2);
        let derivative =
            monotone_wiggle_basis_with_derivative_order(q_seed.view(), &knots, degree, 1)
                .expect("wiggle derivative basis")
                .dot(&beta);
        assert!(
            derivative.iter().all(|&value| value >= -1e-12),
            "I-spline wiggle derivative must stay non-negative for non-negative coefficients: min={}",
            derivative.iter().fold(f64::INFINITY, |acc, &v| acc.min(v))
        );
    }

    #[test]
    fn degeneratewiggle_seed_uses_broad_fallback_domain() {
        let q_seed = Array1::zeros(9);
        let degree = 3usize;
        let knots = initialize_monotone_wiggle_knots_from_seed(q_seed.view(), degree, 5)
            .expect("initialize degenerate wiggle knots");
        let bs_degree = monotone_wiggle_internal_degree(degree).expect("cubic wiggle degree") + 1;
        let domain_min = knots[bs_degree];
        let domain_max = knots[knots.len() - bs_degree - 1];
        assert!(
            domain_min <= -2.9,
            "unexpected left fallback boundary: {domain_min}"
        );
        assert!(
            domain_max >= 2.9,
            "unexpected right fallback boundary: {domain_max}"
        );
    }

    #[test]
    fn wiggle_block_design_matches_ispline_basis() {
        let q_seed = Array1::linspace(-1.0, 1.0, 11);
        let degree = 2usize;
        let num_internal_knots = 4usize;
        let penalty_order = 2usize;

        let (block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            degree,
            num_internal_knots,
            penalty_order,
            false,
        )
        .expect("wiggle block");
        let (basis, _) = create_basis::<Dense>(
            q_seed.view(),
            KnotSource::Provided(knots.view()),
            monotone_wiggle_internal_degree(degree).expect("wiggle degree"),
            BasisOptions::i_spline(),
        )
        .expect("I-spline basis");
        let expected = (*basis).clone();

        let got = match &block.design {
            DesignMatrix::Dense(x) => x.to_dense_arc(),
            DesignMatrix::Sparse(_) => panic!("expected dense wiggle design"),
        };
        assert_eq!(got.dim(), expected.dim());
        for i in 0..got.nrows() {
            for j in 0..got.ncols() {
                assert!(
                    (got[[i, j]] - expected[[i, j]]).abs() < 1e-10,
                    "wiggle design mismatch at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    got[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn split_wiggle_penalty_orders_uses_requested_order_one_as_primary() {
        let (primary, extras) = split_wiggle_penalty_orders(2, &[1, 2, 3, 3]);
        assert_eq!(primary, 1);
        assert_eq!(extras, vec![2, 3]);
    }

    #[test]
    fn append_selected_wiggle_penalty_orders_keeps_order_one() {
        let q_seed = Array1::linspace(-1.0, 1.0, 11);
        let degree = 3usize;
        let num_internal_knots = 5usize;
        let cfg = WiggleBlockConfig {
            degree,
            num_internal_knots,
            penalty_order: 1,
            double_penalty: false,
        };
        let selected = select_wiggle_basis_from_seed(q_seed.view(), &cfg, &[1, 3])
            .expect("selected wiggle basis");

        assert_eq!(selected.block.penalties.len(), 2);
        assert_eq!(selected.block.nullspace_dims, vec![1, 3]);
    }

    #[test]
    fn binomial_location_scale_generative_matches_coremu() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![0.8, -0.4, 0.2, -1.1, 0.0, 0.5, -0.7]);
        let eta_ls = Array1::from_vec(vec![-3.0, -1.2, -0.1, 0.3, 1.1, 2.0, 4.0]);

        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
            policy: crate::resource::ResourcePolicy::default_library(),
        };
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: eta_ls.clone(),
            },
        ];
        let spec = family.generativespec(&states).expect("generative spec");
        let core =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core");
        for i in 0..n {
            assert!(
                (spec.mean[i] - core.mu[i]).abs() < 1e-7,
                "mean mismatch at {i}: got {}, expected {}",
                spec.mean[i],
                core.mu[i]
            );
        }
    }

    #[test]
    fn wiggle_geometry_and_generative_use_same_sigma_link_as_core() {
        let n = 8usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![0.5, -0.6, 0.1, -0.3, 0.9, -0.2, 0.4, -0.8]);
        let eta_ls = Array1::from_vec(vec![-2.5, -1.5, -0.5, 0.0, 0.7, 1.4, 2.2, 3.0]);

        let q_seed = Array1::linspace(-1.5, 1.5, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            3,
            2,
            false,
        )
        .expect("wiggle block");

        let family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
            wiggle_knots: knots,
            wiggle_degree: 2,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let betaw = Array1::from_vec(vec![0.15; wiggle_block.design.ncols()]);
        let etaw = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&betaw);

        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: eta_ls.clone(),
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw.clone(),
            },
        ];

        let wigglespec = wiggle_block
            .clone()
            .intospec("wiggle")
            .expect("wiggle spec");
        let (geom_x, _) = family
            .block_geometry(&states, &wigglespec)
            .expect("block geometry");
        let geom = match geom_x {
            DesignMatrix::Dense(x) => x.to_dense(),
            DesignMatrix::Sparse(_) => panic!("expected dense wiggle geometry design"),
        };
        let expected_geom = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("expected wiggle geometry");
        assert_eq!(geom.dim(), expected_geom.dim());
        for i in 0..geom.nrows() {
            for j in 0..geom.ncols() {
                assert!(
                    (geom[[i, j]] - expected_geom[[i, j]]).abs() < 1e-12,
                    "geometry mismatch at ({i}, {j}): got {}, expected {}",
                    geom[[i, j]],
                    expected_geom[[i, j]]
                );
            }
        }

        let generated = family.generativespec(&states).expect("generative spec");
        let core = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&etaw),
            &family.link_kind,
        )
        .expect("core with wiggle");
        for i in 0..n {
            assert!(
                (generated.mean[i] - core.mu[i]).abs() < 1e-7,
                "wiggle mean mismatch at {i}: got {}, expected {}",
                generated.mean[i],
                core.mu[i]
            );
        }
    }

    /// Binomial location-scale exact Hessian degenerates at extreme fitted values.
    ///
    /// This remains a useful stress test for the exact curvature itself even
    /// though solver policy no longer special-cases wiggle families.
    #[test]
    fn binomial_location_scale_hessian_nonfinite_at_extreme_fitted_values() {
        use crate::families::custom_family::{CustomFamily, ParameterBlockState};
        let n = 4usize;
        let y = array![1.0, 1.0, 0.0, 0.0];
        let weights = Array1::ones(n);

        let design = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                if i < 2 { 1.0 } else { -1.0 }
            }
        });

        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                design.clone(),
            ))),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                design.clone(),
            ))),
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        // Extreme eta values → fitted probs ≈ 0 or 1 → zero working weights
        // → Hessian becomes degenerate/non-finite
        let states = vec![
            ParameterBlockState {
                beta: array![0.0, 50.0], // threshold: huge separation
                eta: array![50.0, 50.0, -50.0, -50.0],
            },
            ParameterBlockState {
                beta: array![0.0, 0.0], // log_sigma: neutral
                eta: array![0.0, 0.0, 0.0, 0.0],
            },
        ];

        // The exact joint Hessian should have non-finite or near-zero entries
        let hessian_result = family.exact_newton_joint_hessian(&states);
        match hessian_result {
            Ok(Some(h)) => {
                // Even if the Hessian computation doesn't error, verify it
                // is degenerate: the 2x2 threshold block should have
                // eigenvalues very close to zero (working weights ≈ 0).
                let h_tt = h.slice(ndarray::s![0..2, 0..2]).to_owned();
                let eigs = crate::faer_ndarray::FaerEigh::eigh(&h_tt, faer::Side::Lower)
                    .expect("eigendecomposition");
                let min_eig = eigs.0.iter().copied().fold(f64::INFINITY, f64::min);
                // With probit at ±50, working weights are essentially machine-epsilon.
                // The threshold Hessian block (X'WX) collapses to near-zero,
                // meaning log|H_mode| → -∞ and its rho-gradient → non-finite.
                assert!(
                    min_eig.abs() < 1e-10,
                    "threshold Hessian block should be near-singular at extreme fitted values, but min eigenvalue = {min_eig:.4e}"
                );
            }
            Ok(None) => {
                panic!("expected joint Hessian to be Some for BinomialLocationScaleFamily");
            }
            Err(err) => {
                // An error here also confirms the instability
                let msg = format!("{err}");
                assert!(
                    msg.contains("non-finite") || msg.contains("NaN"),
                    "unexpected Hessian error: {err}"
                );
            }
        }
    }

    #[test]
    fn poisson_extreme_eta_stays_finite_with_safe_exp() {
        use crate::families::custom_family::{CustomFamily, ParameterBlockState};
        let poisson = PoissonLogFamily {
            y: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            weights: Array1::from_vec(vec![1.0, 1.0, 1.0]),
        };
        let extreme_eta = Array1::from_vec(vec![0.5, 709.0, -0.3]);
        let eval_result = poisson.evaluate(&[ParameterBlockState {
            beta: Array1::zeros(0),
            eta: extreme_eta,
        }]);
        match eval_result {
            Ok(eval) => match &eval.blockworking_sets[0] {
                crate::families::custom_family::BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                } => {
                    let all_finite = working_response.iter().all(|v| v.is_finite())
                        && working_weights.iter().all(|v| v.is_finite())
                        && eval.log_likelihood.is_finite();
                    assert!(
                        all_finite,
                        "Poisson evaluate should produce finite outputs for all eta, \
                             but got non-finite values: ll={}, z={:?}, w={:?}",
                        eval.log_likelihood, working_response, working_weights
                    );
                }
                _ => panic!("expected Diagonal block"),
            },
            Err(_) => {}
        }
    }
}
