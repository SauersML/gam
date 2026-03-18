use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    evaluate_bspline_fourth_derivative_scalar, evaluate_bsplinethird_derivative_scalar,
};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyPsiDesignAction,
    CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    build_block_spatial_psi_derivatives, custom_family_outer_capability,
    evaluate_custom_family_joint_hyper, first_psi_linear_map, fit_custom_family,
    second_psi_linear_map,
};
use crate::estimate::{FitOptions, UnifiedFitResult, fit_gam};
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::gamlss::{ParameterBlockInput, initializewiggle_knots_from_seed};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf};
use crate::quadrature::compute_gauss_hermite_n;
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::function::erf::erfc;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct DeviationBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub monotonicity_eps: f64,
}

impl Default for DeviationBlockConfig {
    fn default() -> Self {
        Self {
            degree: 3,
            num_internal_knots: 8,
            penalty_order: 2,
            penalty_orders: vec![1, 2],
            double_penalty: true,
            monotonicity_eps: 1e-4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviationRuntime {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub transform: Array2<f64>,
    pub monotonicity_eps: f64,
}

#[derive(Clone)]
struct DeviationPrepared {
    block: ParameterBlockInput,
    runtime: DeviationRuntime,
}

#[derive(Clone)]
pub struct BernoulliMarginalSlopeTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub marginalspec: TermCollectionSpec,
    pub logslopespec: TermCollectionSpec,
    pub score_warp: Option<DeviationBlockConfig>,
    pub link_dev: Option<DeviationBlockConfig>,
    pub quadrature_points: usize,
}

pub struct BernoulliMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub logslopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    pub logslope_design: TermCollectionDesign,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
}

#[derive(Clone)]
struct BernoulliMarginalSlopeFamily {
    y: Array1<f64>,
    weights: Array1<f64>,
    z: Array1<f64>,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    quadrature_nodes: Array1<f64>,
    quadrature_weights: Array1<f64>,
    score_warp: Option<DeviationRuntime>,
    score_warp_obs_design: Option<Array2<f64>>,
    link_dev: Option<DeviationRuntime>,
}

#[derive(Clone, Default)]
struct ThetaHints {
    marginal_beta: Option<Array1<f64>>,
    logslope_beta: Option<Array1<f64>>,
    score_warp_beta: Option<Array1<f64>>,
    link_dev_beta: Option<Array1<f64>>,
}

impl DeviationRuntime {
    fn constrained_basis(
        &self,
        values: &Array1<f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        let (basis, _) = create_basis::<Dense>(
            values.view(),
            KnotSource::Provided(self.knots.view()),
            self.degree,
            basis_options,
        )
        .map_err(|e| e.to_string())?;
        let full = basis.as_ref().clone();
        if full.ncols() != self.transform.nrows() {
            return Err(format!(
                "deviation basis/transform mismatch: basis has {} columns but transform has {} rows",
                full.ncols(),
                self.transform.nrows()
            ));
        }
        Ok(full.dot(&self.transform))
    }

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::value())
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::first_derivative())
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::second_derivative())
    }

    pub fn third_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.higher_derivative_design(values, 3)
    }

    pub fn fourth_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.higher_derivative_design(values, 4)
    }

    fn support_interval(&self) -> Result<(f64, f64), String> {
        if self.knots.len() < self.degree + 2 {
            return Err(format!(
                "deviation support needs at least {} knots for degree {}, got {}",
                self.degree + 2,
                self.degree,
                self.knots.len()
            ));
        }
        Ok((
            self.knots[self.degree],
            self.knots[self.knots.len() - self.degree - 1],
        ))
    }

    fn supported_unique_values_from_iter<I>(&self, values: I) -> Result<Array1<f64>, String>
    where
        I: IntoIterator<Item = f64>,
    {
        let (left, right) = self.support_interval()?;
        let mut filtered = values
            .into_iter()
            .filter(|value| value.is_finite() && *value >= left && *value <= right)
            .collect::<Vec<_>>();
        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        filtered.dedup_by(|a, b| (*a - *b).abs() <= 1e-10);
        Ok(Array1::from_vec(filtered))
    }

    fn higher_derivative_design(
        &self,
        values: &Array1<f64>,
        order: usize,
    ) -> Result<Array2<f64>, String> {
        let raw_dim = self.transform.nrows();
        let mut raw = Array2::<f64>::zeros((values.len(), raw_dim));
        if order > self.degree {
            return Ok(raw.dot(&self.transform));
        }
        for (row_idx, &x) in values.iter().enumerate() {
            let row = raw
                .slice_mut(s![row_idx, ..])
                .into_slice()
                .ok_or_else(|| "higher derivative basis row is not contiguous".to_string())?;
            match order {
                3 => {
                    evaluate_bsplinethird_derivative_scalar(x, self.knots.view(), self.degree, row)
                        .map_err(|e| e.to_string())?
                }
                4 => evaluate_bspline_fourth_derivative_scalar(
                    x,
                    self.knots.view(),
                    self.degree,
                    row,
                )
                .map_err(|e| e.to_string())?,
                _ => return Err(format!("unsupported higher derivative order {order}")),
            }
        }
        Ok(raw.dot(&self.transform))
    }
}

fn design_row_owned(design: &DesignMatrix, row: usize) -> Array1<f64> {
    design.row_chunk(row..row + 1).row(0).to_owned()
}

fn deviation_transform(
    knots: &Array1<f64>,
    degree: usize,
    seed: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let mut sorted = seed.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    };
    let anchor = Array1::from_vec(vec![median]);
    let (value_basis, _) = create_basis::<Dense>(
        anchor.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    let (d1_basis, _) = create_basis::<Dense>(
        anchor.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .map_err(|e| e.to_string())?;
    let k = value_basis.ncols();
    let mut c = Array2::<f64>::zeros((2, k));
    c.row_mut(0).assign(&value_basis.row(0));
    c.row_mut(1).assign(&d1_basis.row(0));
    let (z, rank) = rrqr_nullspace_basis(&c.t(), default_rrqr_rank_alpha())
        .map_err(|e| format!("deviation RRQR failed: {e}"))?;
    if rank >= k || z.ncols() == 0 {
        return Err(
            "deviation anchor constraints removed all columns; increase basis richness".to_string(),
        );
    }
    Ok(z)
}

fn build_deviation_block_from_seed(
    seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    let knots = initializewiggle_knots_from_seed(seed.view(), cfg.degree, cfg.num_internal_knots)?;
    let runtime = DeviationRuntime {
        knots: knots.clone(),
        degree: cfg.degree,
        transform: deviation_transform(&knots, cfg.degree, seed)?,
        monotonicity_eps: cfg.monotonicity_eps,
    };
    let design = runtime.design(seed)?;
    let raw_dim = runtime.transform.nrows();
    let dim = runtime.transform.ncols();
    let mut penalties: Vec<crate::solver::estimate::PenaltySpec> = Vec::new();
    // The deviation transform T projects out 2 anchor constraints (value + derivative
    // at the seed median).  For a difference penalty of order k on the raw basis,
    // ker(D^k) has dimension k (polynomials of degree ≤ k-1).  After the projection
    // T'S T, the anchor constraints absorb 2 of those null directions, giving
    // transformed nullspace dimension = max(0, k - 2).
    let anchor_rank = raw_dim - dim; // number of constraints removed by T (typically 2)
    let mut nullspace_dims: Vec<usize> = Vec::new();
    let base_penalty = create_difference_penalty_matrix(raw_dim, cfg.penalty_order, None)
        .map_err(|e| e.to_string())?;
    penalties.push(crate::solver::estimate::PenaltySpec::Dense(fast_ab(
        &fast_atb(&runtime.transform, &base_penalty),
        &runtime.transform,
    )));
    nullspace_dims.push(cfg.penalty_order.saturating_sub(anchor_rank));
    for &order in &cfg.penalty_orders {
        if order == cfg.penalty_order || order == 0 || order >= raw_dim {
            continue;
        }
        let raw =
            create_difference_penalty_matrix(raw_dim, order, None).map_err(|e| e.to_string())?;
        penalties.push(crate::solver::estimate::PenaltySpec::Dense(fast_ab(
            &fast_atb(&runtime.transform, &raw),
            &runtime.transform,
        )));
        nullspace_dims.push(order.saturating_sub(anchor_rank));
    }
    if cfg.double_penalty {
        penalties.push(crate::solver::estimate::PenaltySpec::Dense(
            Array2::<f64>::eye(dim),
        ));
        nullspace_dims.push(0); // identity has full rank, no nullspace
    }
    Ok(DeviationPrepared {
        block: ParameterBlockInput {
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
            offset: Array1::zeros(seed.len()),
            penalties,
            nullspace_dims,
            initial_log_lambdas: None,
            initial_beta: Some(Array1::zeros(dim)),
        },
        runtime,
    })
}

fn validate_spec(
    data: ArrayView2<'_, f64>,
    spec: &BernoulliMarginalSlopeTermSpec,
) -> Result<(), String> {
    let n = data.nrows();
    if spec.y.len() != n || spec.weights.len() != n || spec.z.len() != n {
        return Err(format!(
            "bernoulli-marginal-slope row mismatch: data={}, y={}, weights={}, z={}",
            n,
            spec.y.len(),
            spec.weights.len(),
            spec.z.len()
        ));
    }
    if spec
        .y
        .iter()
        .any(|&yi| !yi.is_finite() || ((yi - 0.0).abs() > 1e-9 && (yi - 1.0).abs() > 1e-9))
    {
        return Err("bernoulli-marginal-slope requires binary y in {0,1}".to_string());
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err("bernoulli-marginal-slope requires finite non-negative weights".to_string());
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err("bernoulli-marginal-slope requires finite z values".to_string());
    }
    Ok(())
}

fn pooled_probit_baseline(
    y: &Array1<f64>,
    z: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<(f64, f64), String> {
    let n = y.len();
    let mut x = Array2::<f64>::zeros((n, 2));
    x.column_mut(0).fill(1.0);
    x.column_mut(1).assign(z);
    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        Array1::zeros(n).view(),
        &[],
        LikelihoodFamily::BinomialProbit,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: Vec::new(),
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    )
    .map_err(|e| format!("failed to fit pooled bernoulli-marginal-slope pilot probit: {e}"))?;
    let a = fit.beta.get(0).copied().unwrap_or(0.0);
    let b = fit.beta.get(1).copied().unwrap_or(0.0).abs().max(1e-6);
    Ok((a / (1.0 + b * b).sqrt(), b.ln()))
}

fn joint_setup(
    marginalspec: &TermCollectionSpec,
    logslopespec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslope_penalties: usize,
    extra_rho0: &[f64],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let rho_dim = marginal_penalties + logslope_penalties + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    for (idx, &value) in extra_rho0.iter().enumerate() {
        rho0vec[marginal_penalties + logslope_penalties + idx] = value;
    }
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    );
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    );
    let mut values = marginal_kappa.as_array().to_vec();
    values.extend(logslope_kappa.as_array().iter());
    let mut dims = marginal_kappa.dims_per_term().to_vec();
    dims.extend(logslope_kappa.dims_per_term());
    let log_kappa0 = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values), dims.clone());
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso(&dims, kappa_options);
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso(&dims, kappa_options);
    ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

fn normal_expectation_nodes(n: usize) -> (Array1<f64>, Array1<f64>) {
    let gh = compute_gauss_hermite_n(n);
    let scale = 2.0_f64.sqrt();
    let norm = std::f64::consts::PI.sqrt();
    (
        Array1::from_iter(gh.nodes.into_iter().map(|x| scale * x)),
        Array1::from_iter(gh.weights.into_iter().map(|w| w / norm)),
    )
}

#[inline]
pub(crate) fn erfcx_nonnegative(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < 26.0 {
        ((x * x).min(700.0)).exp() * erfc(x)
    } else {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let poly = 1.0
            + 0.5 * inv2
            + 0.75 * inv2 * inv2
            + 1.875 * inv2 * inv2 * inv2
            + 6.5625 * inv2 * inv2 * inv2 * inv2;
        inv * poly / std::f64::consts::PI.sqrt()
    }
}

#[inline]
pub(crate) fn signed_probit_logcdf_and_mills_ratio(x: f64) -> (f64, f64) {
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        let ex = erfcx_nonnegative(u).max(1e-300);
        let log_cdf = -u * u + (0.5 * ex).ln();
        let lambda = (2.0 / std::f64::consts::PI).sqrt() / ex;
        (log_cdf, lambda)
    } else {
        let cdf = normal_cdf(x).clamp(1e-300, 1.0);
        let lambda = normal_pdf(x) / cdf;
        (cdf.ln(), lambda)
    }
}

pub(crate) fn signed_probit_neglog_derivatives_up_to_fourth(
    signed_margin: f64,
    weight: f64,
) -> (f64, f64, f64, f64) {
    if weight == 0.0 || !signed_margin.is_finite() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let (_, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
    let k1 = -lambda;
    let k2 = lambda * (signed_margin + lambda);
    let k3 = lambda
        * (1.0
            - signed_margin * signed_margin
            - 3.0 * signed_margin * lambda
            - 2.0 * lambda * lambda);
    let k4 = lambda
        * ((signed_margin.powi(3) - 3.0 * signed_margin)
            + (7.0 * signed_margin * signed_margin - 4.0) * lambda
            + 12.0 * signed_margin * lambda * lambda
            + 6.0 * lambda.powi(3));
    (weight * k1, weight * k2, weight * k3, weight * k4)
}

pub(crate) fn unary_derivatives_exp(x: f64) -> [f64; 5] {
    let ex = x.exp();
    [ex, ex, ex, ex, ex]
}

pub(crate) fn unary_derivatives_sqrt(x: f64) -> [f64; 5] {
    let s = x.max(1e-300).sqrt();
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    [
        s,
        0.5 / s,
        -0.25 / (x1 * s),
        3.0 / (8.0 * x2 * s),
        -15.0 / (16.0 * x3 * s),
    ]
}

pub(crate) fn unary_derivatives_normal_cdf(x: f64) -> [f64; 5] {
    let phi = normal_pdf(x);
    [
        normal_cdf(x),
        phi,
        -x * phi,
        (x * x - 1.0) * phi,
        (-x.powi(3) + 3.0 * x) * phi,
    ]
}

pub(crate) fn unary_derivatives_neglog_phi(x: f64, weight: f64) -> [f64; 5] {
    let (d1, d2, d3, d4) = signed_probit_neglog_derivatives_up_to_fourth(x, weight);
    let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(x);
    [-weight * log_cdf, d1, d2, d3, d4]
}

/// Derivatives of log(x) through 4th order.
pub(crate) fn unary_derivatives_log(x: f64) -> [f64; 5] {
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x3 * x1;
    [x1.ln(), 1.0 / x1, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

/// Derivatives of log φ(x) = -½x² - ½ln(2π) through 4th order.
pub(crate) fn unary_derivatives_log_normal_pdf(x: f64) -> [f64; 5] {
    let c = 0.5 * (2.0 * std::f64::consts::PI).ln();
    [-0.5 * x * x - c, -x, -1.0, 0.0, 0.0]
}

fn subset_partitions(mask: usize) -> Vec<Vec<usize>> {
    if mask == 0 {
        return vec![Vec::new()];
    }
    let first = mask & mask.wrapping_neg();
    let rest = mask ^ first;
    let mut out = Vec::new();
    let mut subset = rest;
    loop {
        let block = first | subset;
        for mut remainder in subset_partitions(rest ^ subset) {
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

#[derive(Clone, Debug)]
pub(crate) struct MultiDirJet {
    pub n_dirs: usize,
    pub coeffs: Vec<f64>,
}

impl MultiDirJet {
    pub fn zero(n_dirs: usize) -> Self {
        Self {
            n_dirs,
            coeffs: vec![0.0; 1usize << n_dirs],
        }
    }

    pub fn constant(n_dirs: usize, value: f64) -> Self {
        let mut out = Self::zero(n_dirs);
        out.coeffs[0] = value;
        out
    }

    pub fn linear(n_dirs: usize, base: f64, first: &[f64]) -> Self {
        assert!(
            first.len() <= n_dirs,
            "MultiDirJet::linear: first.len()={} exceeds n_dirs={}",
            first.len(),
            n_dirs,
        );
        let mut out = Self::constant(n_dirs, base);
        for (idx, &value) in first.iter().take(n_dirs).enumerate() {
            out.coeffs[1usize << idx] = value;
        }
        out
    }

    pub fn full_mask(&self) -> usize {
        (1usize << self.n_dirs) - 1
    }

    pub fn coeff(&self, mask: usize) -> f64 {
        self.coeffs.get(mask).copied().unwrap_or(0.0)
    }

    pub fn set_coeff(&mut self, mask: usize, value: f64) {
        assert!(
            mask < self.coeffs.len(),
            "MultiDirJet::set_coeff: mask={mask} exceeds capacity={}",
            self.coeffs.len(),
        );
        self.coeffs[mask] = value;
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            out.coeffs[mask] = self.coeff(mask) + other.coeff(mask);
        }
        out
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            out.coeffs[mask] = self.coeff(mask) - other.coeff(mask);
        }
        out
    }

    pub fn scale(&self, scalar: f64) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            out.coeffs[mask] = self.coeffs[mask] * scalar;
        }
        out
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut out = Self::zero(self.n_dirs);
        for mask in 0..=self.full_mask() {
            let mut total = 0.0;
            let mut submask = mask;
            loop {
                total += self.coeff(submask) * other.coeff(mask ^ submask);
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & mask;
            }
            out.coeffs[mask] = total;
        }
        out
    }

    pub fn compose_unary(&self, derivs: [f64; 5]) -> Self {
        let mut out = Self::constant(self.n_dirs, derivs[0]);
        for mask in 1..=self.full_mask() {
            let mut total = 0.0;
            for partition in subset_partitions(mask) {
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
            out.coeffs[mask] = total;
        }
        out
    }
}

#[derive(Clone)]
struct BlockSlices {
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    h: Option<std::ops::Range<usize>>,
    w: Option<std::ops::Range<usize>>,
    total: usize,
}

fn block_slices(
    states: &[ParameterBlockState],
    has_score_warp: bool,
    has_link_dev: bool,
) -> BlockSlices {
    let mut cursor = 0usize;
    let mut block_idx = 0usize;
    let marginal = cursor..cursor + states[block_idx].beta.len();
    cursor = marginal.end;
    block_idx += 1;
    let logslope = cursor..cursor + states[block_idx].beta.len();
    cursor = logslope.end;
    block_idx += 1;
    let h = if has_score_warp {
        let range = cursor..cursor + states[block_idx].beta.len();
        cursor = range.end;
        block_idx += 1;
        Some(range)
    } else {
        None
    };
    let w = if has_link_dev {
        let range = cursor..cursor + states[block_idx].beta.len();
        cursor = range.end;
        Some(range)
    } else {
        None
    };
    BlockSlices {
        marginal,
        logslope,
        h,
        w,
        total: cursor,
    }
}

#[derive(Clone)]
struct PrimarySlices {
    h: Option<std::ops::Range<usize>>,
    w: Option<std::ops::Range<usize>>,
    total: usize,
}

fn primary_slices(slices: &BlockSlices) -> PrimarySlices {
    let mut cursor = 2usize;
    let h = slices.h.as_ref().map(|range| {
        let out = cursor..cursor + range.len();
        cursor = out.end;
        out
    });
    let w = slices.w.as_ref().map(|range| {
        let out = cursor..cursor + range.len();
        cursor = out.end;
        out
    });
    PrimarySlices {
        h,
        w,
        total: cursor,
    }
}

fn unit_primary_direction(primary: &PrimarySlices, idx: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(primary.total);
    out[idx] = 1.0;
    out
}

#[derive(Clone)]
struct LinkDerivativeStack {
    basis: Array2<f64>,
    d1: Array2<f64>,
    d2: Array2<f64>,
    d3: Array2<f64>,
    d4: Array2<f64>,
}

#[derive(Clone)]
struct BernoulliMarginalSlopeRowExactContext {
    intercept: f64,
    m_a: f64,
    h_obs_base: f64,
    node_link: Option<LinkDerivativeStack>,
    obs_link: Option<LinkDerivativeStack>,
}

/// Chunk size for parallel row accumulation.  Rows within a chunk are
/// processed sequentially and the per-row context / gradient / Hessian is
/// computed on the fly and discarded after accumulation — no per-row state
/// is persisted.
const ROW_CHUNK_SIZE: usize = 1024;

/// Lightweight cache that stores ONLY the shared precomputed state needed to
/// recompute any individual row's context on demand.  Unlike the previous
/// design this stores O(1) data with respect to `n` (the per-row gradient
/// and Hessian vectors are never materialised across the full dataset).
#[derive(Clone)]
struct BernoulliMarginalSlopeExactEvalCache {
    slices: BlockSlices,
    primary: PrimarySlices,
    h_nodes: Array1<f64>,
    h_node_design: Option<Array2<f64>>,
    score_warp_obs: Option<(Array2<f64>, Array1<f64>)>,
}

const EXACT_OUTER_HESSIAN_MAX_ROW_PAIR_WORK: usize = 2_000_000;

fn bernoulli_exact_outer_derivative_order(
    n_rows: usize,
    score_warp_dim: usize,
    link_dev_dim: usize,
) -> ExactOuterDerivativeOrder {
    // The exact outer Hessian repeatedly executes row-local primary×primary
    // contractions, so `n * primary_total^2` is the right first-order scale
    // signal for when second-order exact calculus stops being practical.
    let primary_total = 2usize
        .saturating_add(score_warp_dim)
        .saturating_add(link_dev_dim);
    let row_pair_work = n_rows.saturating_mul(primary_total.saturating_mul(primary_total));
    if row_pair_work > EXACT_OUTER_HESSIAN_MAX_ROW_PAIR_WORK {
        ExactOuterDerivativeOrder::First
    } else {
        ExactOuterDerivativeOrder::Second
    }
}

struct BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    cache: BernoulliMarginalSlopeExactEvalCache,
}

struct BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: BernoulliMarginalSlopeExactEvalCache,
}

impl BernoulliMarginalSlopeFamily {
    fn flex_active(&self) -> bool {
        self.score_warp.is_some() || self.link_dev.is_some()
    }

    fn link_terms(
        &self,
        eta0: &Array1<f64>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
        if let (Some(runtime), Some(beta)) = (&self.link_dev, beta_w) {
            let basis = runtime.design(eta0)?;
            let d1 = runtime.first_derivative_design(eta0)?;
            let d2 = runtime.second_derivative_design(eta0)?;
            Ok((eta0 + &basis.dot(beta), d1.dot(beta) + 1.0, d2.dot(beta)))
        } else {
            Ok((
                eta0.clone(),
                Array1::ones(eta0.len()),
                Array1::zeros(eta0.len()),
            ))
        }
    }

    fn quadrature_h(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), String> {
        if let Some(runtime) = &self.score_warp {
            let beta_h = &block_states[2].beta;
            let design = runtime.design(&self.quadrature_nodes)?;
            Ok((&self.quadrature_nodes + &design.dot(beta_h), Some(design)))
        } else {
            Ok((self.quadrature_nodes.clone(), None))
        }
    }

    fn score_warp_obs(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<(Array2<f64>, Array1<f64>)>, String> {
        let Some(obs_design) = self.score_warp_obs_design.as_ref() else {
            return Ok(None);
        };
        if self.score_warp.is_none() {
            return Ok(None);
        }
        let beta_h = &block_states[2].beta;
        Ok(Some((obs_design.clone(), obs_design.dot(beta_h))))
    }

    fn deviation_constraints_from_points(
        runtime: &DeviationRuntime,
        points: Array1<f64>,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if points.is_empty() {
            return Ok(None);
        }
        let derivative_design = runtime.first_derivative_design(&points)?;
        Ok(Some(LinearInequalityConstraints {
            a: derivative_design,
            b: Array1::from_elem(points.len(), runtime.monotonicity_eps - 1.0),
        }))
    }

    fn score_warp_constraint_points(&self) -> Result<Option<Array1<f64>>, String> {
        let Some(runtime) = self.score_warp.as_ref() else {
            return Ok(None);
        };
        let points = runtime.supported_unique_values_from_iter(
            self.z
                .iter()
                .copied()
                .chain(self.quadrature_nodes.iter().copied()),
        )?;
        if points.is_empty() {
            return Ok(None);
        }
        Ok(Some(points))
    }

    fn link_dev_constraint_points(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        let Some(runtime) = self.link_dev.as_ref() else {
            return Ok(None);
        };
        let beta_w = block_states.last().map(|state| &state.beta);
        let (h_nodes, _) = self.quadrature_h(block_states)?;
        let score_warp_obs = self.score_warp_obs(block_states)?;
        let mut values = Vec::with_capacity(self.z.len() * (h_nodes.len() + 1));
        for row in 0..self.z.len() {
            let marginal_eta = block_states[0].eta[row];
            let slope = block_states[1].eta[row].exp();
            let h_obs_base = if let Some((_, dev_obs)) = score_warp_obs.as_ref() {
                self.z[row] + dev_obs[row]
            } else {
                self.z[row]
            };
            let (intercept, _) =
                self.solve_row_intercept_base(marginal_eta, slope, &h_nodes, beta_w)?;
            values.extend(h_nodes.iter().map(|&h| intercept + slope * h));
            values.push(intercept + slope * h_obs_base);
        }
        let points = runtime.supported_unique_values_from_iter(values)?;
        if points.is_empty() {
            return Ok(None);
        }
        Ok(Some(points))
    }

    fn solve_row_intercept_base(
        &self,
        marginal_eta: f64,
        slope: f64,
        h_nodes: &Array1<f64>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let target = normal_cdf(marginal_eta);

        // Evaluate the constraint function F(a) and its derivative F_a at a given intercept.
        // F(a) = ∫ Φ(L(a + b·h)) w(h) dh − Φ(q)
        // F_a(a) = ∫ φ(L(a + b·h)) · L'(a + b·h) · w(h) dh
        let eval = |a: f64| -> Result<(f64, f64), String> {
            let v = h_nodes.mapv(|h| a + slope * h);
            let (t, t1, _) = self.link_terms(&v, beta_w)?;
            let f_val = self
                .quadrature_weights
                .iter()
                .zip(t.iter())
                .map(|(&w, &tt)| w * normal_cdf(tt))
                .sum::<f64>()
                - target;
            let f_deriv = self
                .quadrature_weights
                .iter()
                .zip(t.iter().zip(t1.iter()))
                .map(|(&w, (&tt, &tt1))| w * normal_pdf(tt) * tt1)
                .sum::<f64>();
            Ok((f_val, f_deriv))
        };

        // Initial guess: closed-form solution for the rigid (no link deviation) case.
        let mut a = marginal_eta * (1.0 + slope * slope).sqrt();

        // Track bracket [lo, hi] such that F(lo) < 0 < F(hi).
        // F is monotone increasing in a when the link is monotone.
        let mut lo = f64::NEG_INFINITY;
        let mut hi = f64::INFINITY;

        for _ in 0..40 {
            let (f_val, f_deriv) = eval(a)?;

            // Update bracket.
            if f_val < 0.0 {
                lo = a;
            } else if f_val > 0.0 {
                hi = a;
            } else {
                // Exact root.
                return Ok((a, f_deriv));
            }

            // Check convergence on bracket width.
            if lo.is_finite() && hi.is_finite() && (hi - lo) < 1e-12 {
                break;
            }

            // Try Newton step if derivative is well-behaved.
            let use_newton = f_deriv.is_finite() && f_deriv > 1e-30;
            if use_newton {
                let a_newton = a - f_val / f_deriv;
                // Accept Newton step only if it stays within the bracket.
                if lo.is_finite() && hi.is_finite() {
                    if a_newton > lo && a_newton < hi {
                        a = a_newton;
                    } else {
                        a = 0.5 * (lo + hi);
                    }
                } else {
                    a = a_newton;
                }
            } else if lo.is_finite() && hi.is_finite() {
                // Bisect within bracket.
                a = 0.5 * (lo + hi);
            } else {
                // No bracket yet, no usable Newton step: expand search.
                if f_val < 0.0 {
                    a += 4.0;
                } else {
                    a -= 4.0;
                }
            }
        }

        // Final evaluation and checks.
        let (f_val, f_deriv) = eval(a)?;
        if f_val.abs() > 1e-6 {
            return Err(format!(
                "bernoulli marginal-slope intercept solve failed: \
                 residual={f_val:.3e} at a={a:.6}, target Φ(q)={target:.6}"
            ));
        }
        if !f_deriv.is_finite() || f_deriv <= 0.0 {
            return Err(format!(
                "bernoulli marginal-slope intercept solve: \
                 link monotonicity violated, F_a={f_deriv:.3e} at a={a:.6}"
            ));
        }
        Ok((a, f_deriv))
    }

    fn link_basis_stack(
        &self,
        runtime: &DeviationRuntime,
        values: &Array1<f64>,
    ) -> Result<LinkDerivativeStack, String> {
        Ok(LinkDerivativeStack {
            basis: runtime.design(values)?,
            d1: runtime.first_derivative_design(values)?,
            d2: runtime.second_derivative_design(values)?,
            d3: runtime.third_derivative_design(values)?,
            d4: runtime.fourth_derivative_design(values)?,
        })
    }

    fn build_row_exact_context(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        h_nodes: &Array1<f64>,
        score_warp_obs: Option<&(Array2<f64>, Array1<f64>)>,
    ) -> Result<BernoulliMarginalSlopeRowExactContext, String> {
        let marginal_eta = block_states[0].eta[row];
        let slope = block_states[1].eta[row].exp();
        let beta_w = if self.link_dev.is_some() {
            block_states.last().map(|state| &state.beta)
        } else {
            None
        };
        let h_obs_base = if let Some((_, dev_obs)) = score_warp_obs {
            self.z[row] + dev_obs[row]
        } else {
            self.z[row]
        };
        let (intercept, m_a) = if self.flex_active() {
            self.solve_row_intercept_base(marginal_eta, slope, h_nodes, beta_w)?
        } else {
            (marginal_eta * (1.0 + slope * slope).sqrt(), f64::NAN)
        };
        let node_link = if let Some(runtime) = &self.link_dev {
            let u_base = h_nodes.mapv(|h| intercept + slope * h);
            Some(self.link_basis_stack(runtime, &u_base)?)
        } else {
            None
        };
        let obs_link = if let Some(runtime) = &self.link_dev {
            let eta_base = Array1::from_vec(vec![intercept + slope * h_obs_base]);
            Some(self.link_basis_stack(runtime, &eta_base)?)
        } else {
            None
        };
        Ok(BernoulliMarginalSlopeRowExactContext {
            intercept,
            m_a,
            h_obs_base,
            node_link,
            obs_link,
        })
    }

    fn build_exact_eval_cache(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
        let primary = primary_slices(&slices);
        let (h_nodes, h_node_design) = self.quadrature_h(block_states)?;
        let score_warp_obs = self.score_warp_obs(block_states)?;
        Ok(BernoulliMarginalSlopeExactEvalCache {
            slices,
            primary,
            h_nodes,
            h_node_design,
            score_warp_obs,
        })
    }

    fn build_gamma_jets(
        &self,
        primary: &PrimarySlices,
        beta_w: Option<&Array1<f64>>,
        dirs: &[Array1<f64>],
    ) -> Vec<MultiDirJet> {
        let Some(range) = primary.w.as_ref() else {
            return Vec::new();
        };
        let Some(beta) = beta_w else {
            return Vec::new();
        };
        (0..range.len())
            .map(|idx| {
                let first = dirs
                    .iter()
                    .map(|dir| dir[range.start + idx])
                    .collect::<Vec<_>>();
                MultiDirJet::linear(dirs.len(), beta[idx], &first)
            })
            .collect()
    }

    fn apply_link_jet_from_rows(
        &self,
        eta_jet: &MultiDirJet,
        gamma_jets: &[MultiDirJet],
        basis: &[f64],
        d1: &[f64],
        d2: &[f64],
        d3: &[f64],
        d4: &[f64],
    ) -> MultiDirJet {
        let mut out = eta_jet.clone();
        for idx in 0..basis.len() {
            let basis_jet = eta_jet.compose_unary([basis[idx], d1[idx], d2[idx], d3[idx], d4[idx]]);
            out = out.add(&basis_jet.mul(&gamma_jets[idx]));
        }
        out
    }

    fn intercept_equation_jet(
        &self,
        primary: &PrimarySlices,
        dirs: &[Array1<f64>],
        q_jet: &MultiDirJet,
        b_jet: &MultiDirJet,
        a_jet: &MultiDirJet,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        h_nodes: &Array1<f64>,
        h_node_design: Option<&Array2<f64>>,
        gamma_jets: &[MultiDirJet],
    ) -> Result<MultiDirJet, String> {
        let k = dirs.len();
        let mut out = MultiDirJet::zero(k);
        if let Some(link_stack) = row_ctx.node_link.as_ref() {
            for node in 0..h_nodes.len() {
                let h_first =
                    if let (Some(h_range), Some(design)) = (primary.h.as_ref(), h_node_design) {
                        dirs.iter()
                            .map(|dir| {
                                design
                                    .row(node)
                                    .dot(&dir.slice(s![h_range.start..h_range.end]).to_owned())
                            })
                            .collect::<Vec<_>>()
                    } else {
                        vec![0.0; k]
                    };
                let h_jet = MultiDirJet::linear(k, h_nodes[node], &h_first);
                let u_jet = a_jet.add(&b_jet.mul(&h_jet));
                let s_jet = self.apply_link_jet_from_rows(
                    &u_jet,
                    gamma_jets,
                    link_stack.basis.row(node).as_slice().unwrap_or(&[]),
                    link_stack.d1.row(node).as_slice().unwrap_or(&[]),
                    link_stack.d2.row(node).as_slice().unwrap_or(&[]),
                    link_stack.d3.row(node).as_slice().unwrap_or(&[]),
                    link_stack.d4.row(node).as_slice().unwrap_or(&[]),
                );
                out = out.add(
                    &s_jet
                        .compose_unary(unary_derivatives_normal_cdf(s_jet.coeff(0)))
                        .scale(self.quadrature_weights[node]),
                );
            }
        } else {
            for node in 0..h_nodes.len() {
                let h_first =
                    if let (Some(h_range), Some(design)) = (primary.h.as_ref(), h_node_design) {
                        dirs.iter()
                            .map(|dir| {
                                design
                                    .row(node)
                                    .dot(&dir.slice(s![h_range.start..h_range.end]).to_owned())
                            })
                            .collect::<Vec<_>>()
                    } else {
                        vec![0.0; k]
                    };
                let h_jet = MultiDirJet::linear(k, h_nodes[node], &h_first);
                let u_jet = a_jet.add(&b_jet.mul(&h_jet));
                out = out.add(
                    &u_jet
                        .compose_unary(unary_derivatives_normal_cdf(u_jet.coeff(0)))
                        .scale(self.quadrature_weights[node]),
                );
            }
        }
        Ok(out.sub(&q_jet.compose_unary(unary_derivatives_normal_cdf(q_jet.coeff(0)))))
    }

    fn row_neglog_directional_from_primary(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        dirs: &[Array1<f64>],
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        h_nodes: &Array1<f64>,
        h_node_design: Option<&Array2<f64>>,
        score_warp_obs: Option<&(Array2<f64>, Array1<f64>)>,
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!(
                "row directional derivative expects 0..=4 directions, got {k}"
            ));
        }
        let q_first = dirs.iter().map(|dir| dir[0]).collect::<Vec<_>>();
        let g_first = dirs.iter().map(|dir| dir[1]).collect::<Vec<_>>();
        let q_jet = MultiDirJet::linear(k, block_states[0].eta[row], &q_first);
        let g_jet = MultiDirJet::linear(k, block_states[1].eta[row], &g_first);
        let b_jet = g_jet.compose_unary(unary_derivatives_exp(g_jet.coeff(0)));
        let beta_w = if self.link_dev.is_some() {
            block_states.last().map(|state| &state.beta)
        } else {
            None
        };
        let gamma_jets = self.build_gamma_jets(primary, beta_w, dirs);

        let a_jet = if self.flex_active() {
            let mut jet = MultiDirJet::constant(k, row_ctx.intercept);
            let m_a = row_ctx.m_a;
            for order in 1..=k {
                for mask in 1..=jet.full_mask() {
                    if mask.count_ones() as usize != order {
                        continue;
                    }
                    jet.set_coeff(mask, 0.0);
                    let m_jet = self.intercept_equation_jet(
                        primary,
                        dirs,
                        &q_jet,
                        &b_jet,
                        &jet,
                        row_ctx,
                        h_nodes,
                        h_node_design,
                        &gamma_jets,
                    )?;
                    jet.set_coeff(mask, -m_jet.coeff(mask) / m_a);
                }
            }
            jet
        } else {
            let one_plus_b2 = MultiDirJet::constant(k, 1.0).add(&b_jet.mul(&b_jet));
            let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));
            q_jet.mul(&c_jet)
        };

        let h_obs_first =
            if let (Some(h_range), Some((obs_design, _))) = (primary.h.as_ref(), score_warp_obs) {
                dirs.iter()
                    .map(|dir| {
                        obs_design
                            .row(row)
                            .dot(&dir.slice(s![h_range.start..h_range.end]).to_owned())
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![0.0; k]
            };
        let h_obs_jet = MultiDirJet::linear(k, row_ctx.h_obs_base, &h_obs_first);
        let eta_jet = a_jet.add(&b_jet.mul(&h_obs_jet));

        let s_jet = if let Some(link_stack) = row_ctx.obs_link.as_ref() {
            self.apply_link_jet_from_rows(
                &eta_jet,
                &gamma_jets,
                link_stack.basis.row(0).as_slice().unwrap_or(&[]),
                link_stack.d1.row(0).as_slice().unwrap_or(&[]),
                link_stack.d2.row(0).as_slice().unwrap_or(&[]),
                link_stack.d3.row(0).as_slice().unwrap_or(&[]),
                link_stack.d4.row(0).as_slice().unwrap_or(&[]),
            )
        } else {
            eta_jet
        };

        let signed_margin = s_jet.scale(2.0 * self.y[row] - 1.0);
        let f_jet = signed_margin.compose_unary(unary_derivatives_neglog_phi(
            signed_margin.coeff(0),
            self.weights[row],
        ));
        Ok(f_jet.coeff(f_jet.full_mask()))
    }

    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope d_beta length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let mut out = Array1::<f64>::zeros(primary.total);
        out[0] = design_row_owned(&self.marginal_design, row)
            .dot(&d_beta_flat.slice(s![slices.marginal.clone()]).to_owned());
        out[1] = design_row_owned(&self.logslope_design, row)
            .dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned());
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        Ok(out)
    }

    fn resolve_psi_location(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Option<(usize, usize)> {
        let mut cursor = 0usize;
        for (block_idx, block) in derivative_blocks.iter().enumerate() {
            if psi_index < cursor + block.len() {
                return Some((block_idx, psi_index - cursor));
            }
            cursor += block.len();
        }
        None
    }

    fn psi_design_row_vector(
        &self,
        row: usize,
        deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiDesignAction::from_first_derivative(
            deriv,
            total_rows,
            p,
            0..total_rows,
            label,
        )
        .ok();
        first_psi_linear_map(action.as_ref(), &deriv.x_psi, total_rows, p).row_vector(row)
    }

    fn psi_second_design_row_vector(
        &self,
        row: usize,
        deriv_i: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        deriv_j: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        local_j: usize,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiSecondDesignAction::from_second_derivative(
            deriv_i,
            deriv_j,
            total_rows,
            p,
            0..total_rows,
            label,
        )?;
        let dense = deriv_i
            .x_psi_psi
            .as_ref()
            .and_then(|rows| rows.get(local_j));
        second_psi_linear_map(action.as_ref(), dense, total_rows, p).row_vector(row)
    }

    fn row_primary_psi_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        primary: &PrimarySlices,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?;
                out[0] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?;
                out[1] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi direction only supports spatial marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_action_on_direction(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?;
                out[0] = x_row.dot(&d_beta_flat.slice(s![slices.marginal.clone()]).to_owned())
            }
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?;
                out[1] = x_row.dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned())
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi action only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_second_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        primary: &PrimarySlices,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(Some(Array1::<f64>::zeros(primary.total)));
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_i {
            0 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?;
                out[0] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?;
                out[1] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second direction only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn pullback_primary_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_vec: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(slices.total);
        out.slice_mut(s![slices.marginal.clone()])
            .assign(&(&design_row_owned(&self.marginal_design, row) * primary_vec[0]));
        out.slice_mut(s![slices.logslope.clone()])
            .assign(&(&design_row_owned(&self.logslope_design, row) * primary_vec[1]));
        if let Some(primary_h) = primary.h.as_ref() {
            if let Some(block_h) = slices.h.as_ref() {
                out.slice_mut(s![block_h.clone()]).assign(
                    &primary_vec
                        .slice(s![primary_h.start..primary_h.end])
                        .to_owned(),
                );
            }
        }
        if let Some(primary_w) = primary.w.as_ref() {
            if let Some(block_w) = slices.w.as_ref() {
                out.slice_mut(s![block_w.clone()]).assign(
                    &primary_vec
                        .slice(s![primary_w.start..primary_w.end])
                        .to_owned(),
                );
            }
        }
        out
    }

    fn embedded_psi_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, Array1<f64>)>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(slices.total);
        match block_idx {
            0 => out
                .slice_mut(s![slices.marginal.clone()])
                .assign(&self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?),
            1 => out
                .slice_mut(s![slices.logslope.clone()])
                .assign(&self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi embedding only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some((block_idx, out)))
    }

    fn embedded_psi_second_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<(usize, Array1<f64>)>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(Some((block_i, Array1::<f64>::zeros(slices.total))));
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let mut out = Array1::<f64>::zeros(slices.total);
        match block_i {
            0 => out.slice_mut(s![slices.marginal.clone()]).assign(
                &self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?,
            ),
            1 => out.slice_mut(s![slices.logslope.clone()]).assign(
                &self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?,
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second embedding only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        }
        Ok(Some((block_i, out)))
    }

    fn compute_row_primary_gradient_hessian(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        h_nodes: &Array1<f64>,
        h_node_design: Option<&Array2<f64>>,
        score_warp_obs: Option<&(Array2<f64>, Array1<f64>)>,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        let mut grad = Array1::<f64>::zeros(primary.total);
        let mut hess = Array2::<f64>::zeros((primary.total, primary.total));
        for a in 0..primary.total {
            let da = unit_primary_direction(primary, a);
            grad[a] = self.row_neglog_directional_from_primary(
                row,
                block_states,
                primary,
                &[da.clone()],
                row_ctx,
                h_nodes,
                h_node_design,
                score_warp_obs,
            )?;
            for b in a..primary.total {
                let db = unit_primary_direction(primary, b);
                let value = self.row_neglog_directional_from_primary(
                    row,
                    block_states,
                    primary,
                    &[da.clone(), db.clone()],
                    row_ctx,
                    h_nodes,
                    h_node_design,
                    score_warp_obs,
                )?;
                hess[[a, b]] = value;
                hess[[b, a]] = value;
            }
        }
        Ok((grad, hess))
    }

    fn row_primary_third_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let primary = &cache.primary;
        let mut out = Array2::<f64>::zeros((primary.total, primary.total));
        for a in 0..primary.total {
            let da = unit_primary_direction(primary, a);
            for b in a..primary.total {
                let db = unit_primary_direction(primary, b);
                let value = self.row_neglog_directional_from_primary(
                    row,
                    block_states,
                    primary,
                    &[da.clone(), db.clone(), dir.clone()],
                    row_ctx,
                    &cache.h_nodes,
                    cache.h_node_design.as_ref(),
                    cache.score_warp_obs.as_ref(),
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    fn row_primary_fourth_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let primary = &cache.primary;
        let mut out = Array2::<f64>::zeros((primary.total, primary.total));
        for a in 0..primary.total {
            let da = unit_primary_direction(primary, a);
            for b in a..primary.total {
                let db = unit_primary_direction(primary, b);
                let value = self.row_neglog_directional_from_primary(
                    row,
                    block_states,
                    primary,
                    &[da.clone(), db.clone(), dir_u.clone(), dir_v.clone()],
                    row_ctx,
                    &cache.h_nodes,
                    cache.h_node_design.as_ref(),
                    cache.score_warp_obs.as_ref(),
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: &Array2<f64>,
    ) {
        let x_row = design_row_owned(&self.marginal_design, row);
        let g_row = design_row_owned(&self.logslope_design, row);
        let xx = x_row
            .view()
            .insert_axis(Axis(1))
            .dot(&x_row.view().insert_axis(Axis(0)))
            * primary_hessian[[0, 0]];
        target
            .slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
            .scaled_add(1.0, &xx);

        let xg = x_row
            .view()
            .insert_axis(Axis(1))
            .dot(&g_row.view().insert_axis(Axis(0)))
            * primary_hessian[[0, 1]];
        target
            .slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
            .scaled_add(1.0, &xg);
        target
            .slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
            .scaled_add(1.0, &xg.t().to_owned());

        let gg = g_row
            .view()
            .insert_axis(Axis(1))
            .dot(&g_row.view().insert_axis(Axis(0)))
            * primary_hessian[[1, 1]];
        target
            .slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .scaled_add(1.0, &gg);

        if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
            let h_row0 = primary_hessian
                .slice(s![0, primary_h.start..primary_h.end])
                .to_owned();
            let h_len = h_row0.len();
            let h_row0_2d = h_row0.into_shape_with_order((1, h_len)).unwrap();
            let qh = x_row.view().insert_axis(Axis(1)).dot(&h_row0_2d);
            target
                .slice_mut(s![slices.marginal.clone(), block_h.clone()])
                .scaled_add(1.0, &qh);
            target
                .slice_mut(s![block_h.clone(), slices.marginal.clone()])
                .scaled_add(1.0, &qh.t().to_owned());

            let h_row1 = primary_hessian
                .slice(s![1, primary_h.start..primary_h.end])
                .to_owned();
            let h_row1_2d = h_row1.into_shape_with_order((1, h_len)).unwrap();
            let gh = g_row.view().insert_axis(Axis(1)).dot(&h_row1_2d);
            target
                .slice_mut(s![slices.logslope.clone(), block_h.clone()])
                .scaled_add(1.0, &gh);
            target
                .slice_mut(s![block_h.clone(), slices.logslope.clone()])
                .scaled_add(1.0, &gh.t().to_owned());

            target
                .slice_mut(s![block_h.clone(), block_h.clone()])
                .scaled_add(
                    1.0,
                    &primary_hessian
                        .slice(s![
                            primary_h.start..primary_h.end,
                            primary_h.start..primary_h.end
                        ])
                        .to_owned(),
                );
        }

        if let (Some(primary_w), Some(block_w)) = (primary.w.as_ref(), slices.w.as_ref()) {
            let w_row0 = primary_hessian
                .slice(s![0, primary_w.start..primary_w.end])
                .to_owned();
            let w_len = w_row0.len();
            let w_row0_2d = w_row0.into_shape_with_order((1, w_len)).unwrap();
            let qw = x_row.view().insert_axis(Axis(1)).dot(&w_row0_2d);
            target
                .slice_mut(s![slices.marginal.clone(), block_w.clone()])
                .scaled_add(1.0, &qw);
            target
                .slice_mut(s![block_w.clone(), slices.marginal.clone()])
                .scaled_add(1.0, &qw.t().to_owned());

            let w_row1 = primary_hessian
                .slice(s![1, primary_w.start..primary_w.end])
                .to_owned();
            let w_row1_2d = w_row1.into_shape_with_order((1, w_len)).unwrap();
            let gw = g_row.view().insert_axis(Axis(1)).dot(&w_row1_2d);
            target
                .slice_mut(s![slices.logslope.clone(), block_w.clone()])
                .scaled_add(1.0, &gw);
            target
                .slice_mut(s![block_w.clone(), slices.logslope.clone()])
                .scaled_add(1.0, &gw.t().to_owned());

            if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
                target
                    .slice_mut(s![block_h.clone(), block_w.clone()])
                    .scaled_add(
                        1.0,
                        &primary_hessian
                            .slice(s![
                                primary_h.start..primary_h.end,
                                primary_w.start..primary_w.end
                            ])
                            .to_owned(),
                    );
                target
                    .slice_mut(s![block_w.clone(), block_h.clone()])
                    .scaled_add(
                        1.0,
                        &primary_hessian
                            .slice(s![
                                primary_w.start..primary_w.end,
                                primary_h.start..primary_h.end
                            ])
                            .to_owned(),
                    );
            }

            target
                .slice_mut(s![block_w.clone(), block_w.clone()])
                .scaled_add(
                    1.0,
                    &primary_hessian
                        .slice(s![
                            primary_w.start..primary_w.end,
                            primary_w.start..primary_w.end
                        ])
                        .to_owned(),
                );
        }
    }

    fn joint_gradient_hessian(
        &self,
        block_states: &[ParameterBlockState],
        need_hessian: bool,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String> {
        let cache = self.build_exact_eval_cache(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        // Parallel chunked accumulation: each chunk computes row contexts on
        // the fly, accumulates gradient/hessian/ll, then discards row state.
        let chunk_results: Vec<Result<(f64, Array1<f64>, Option<Array2<f64>>), String>> =
            (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let mut chunk_ll = 0.0;
                    let mut chunk_grad = Array1::<f64>::zeros(slices.total);
                    let mut chunk_hess =
                        need_hessian.then(|| Array2::<f64>::zeros((slices.total, slices.total)));
                    for i in start..end {
                        let row_ctx = self.build_row_exact_context(
                            i,
                            block_states,
                            &cache.h_nodes,
                            cache.score_warp_obs.as_ref(),
                        )?;
                        let row_neglog = self.row_neglog_directional_from_primary(
                            i,
                            block_states,
                            primary,
                            &[],
                            &row_ctx,
                            &cache.h_nodes,
                            cache.h_node_design.as_ref(),
                            cache.score_warp_obs.as_ref(),
                        )?;
                        chunk_ll -= row_neglog;

                        let (f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                            i,
                            block_states,
                            primary,
                            &row_ctx,
                            &cache.h_nodes,
                            cache.h_node_design.as_ref(),
                            cache.score_warp_obs.as_ref(),
                        )?;
                        chunk_grad -= &self.pullback_primary_vector(i, slices, primary, &f_pi);
                        if let Some(ref mut hmat) = chunk_hess {
                            self.add_pullback_primary_hessian(hmat, i, slices, primary, &f_pipi);
                        }
                        // row_ctx and f_pi/f_pipi are dropped here — no per-row storage
                    }
                    Ok((chunk_ll, chunk_grad, chunk_hess))
                })
                .collect();

        // Reduce chunk results
        let mut ll = 0.0;
        let mut gradient = Array1::<f64>::zeros(slices.total);
        let mut hessian = need_hessian.then(|| Array2::<f64>::zeros((slices.total, slices.total)));
        for result in chunk_results {
            let (chunk_ll, chunk_grad, chunk_hess) = result?;
            ll += chunk_ll;
            gradient += &chunk_grad;
            if let (Some(hmat), Some(chunk_h)) = (&mut hessian, chunk_hess) {
                *hmat += &chunk_h;
            }
        }
        Ok((ll, gradient, hessian))
    }

    fn exact_newton_joint_hessian_matvec_from_cache(
        &self,
        direction: &Array1<f64>,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let chunk_results: Vec<Result<Array1<f64>, String>> = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut chunk_out = Array1::<f64>::zeros(slices.total);
                for row in start..end {
                    let row_ctx = self.build_row_exact_context(
                        row,
                        block_states,
                        &cache.h_nodes,
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, direction)?;
                    let (_, row_hessian) = self.compute_row_primary_gradient_hessian(
                        row,
                        block_states,
                        primary,
                        &row_ctx,
                        &cache.h_nodes,
                        cache.h_node_design.as_ref(),
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let row_action = row_hessian.dot(&row_dir);
                    chunk_out += &self.pullback_primary_vector(row, slices, primary, &row_action);
                }
                Ok(chunk_out)
            })
            .collect();
        let mut out = Array1::<f64>::zeros(slices.total);
        for result in chunk_results {
            out += &result?;
        }
        Ok(out)
    }

    fn exact_newton_joint_hessian_diagonal_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let chunk_results: Vec<Result<Array1<f64>, String>> = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut chunk_diag = Array1::<f64>::zeros(slices.total);
                for row in start..end {
                    let row_ctx = self.build_row_exact_context(
                        row,
                        block_states,
                        &cache.h_nodes,
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let (_, row_hessian) = self.compute_row_primary_gradient_hessian(
                        row,
                        block_states,
                        primary,
                        &row_ctx,
                        &cache.h_nodes,
                        cache.h_node_design.as_ref(),
                        cache.score_warp_obs.as_ref(),
                    )?;

                    let marginal_row = design_row_owned(&self.marginal_design, row);
                    for (local_idx, value) in marginal_row.iter().enumerate() {
                        chunk_diag[slices.marginal.start + local_idx] +=
                            value * value * row_hessian[[0, 0]];
                    }
                    let logslope_row = design_row_owned(&self.logslope_design, row);
                    for (local_idx, value) in logslope_row.iter().enumerate() {
                        chunk_diag[slices.logslope.start + local_idx] +=
                            value * value * row_hessian[[1, 1]];
                    }

                    if let (Some(primary_h), Some(block_h)) =
                        (primary.h.as_ref(), slices.h.as_ref())
                    {
                        for (local_idx, global_idx) in block_h.clone().enumerate() {
                            chunk_diag[global_idx] += row_hessian
                                [[primary_h.start + local_idx, primary_h.start + local_idx]];
                        }
                    }
                    if let (Some(primary_w), Some(block_w)) =
                        (primary.w.as_ref(), slices.w.as_ref())
                    {
                        for (local_idx, global_idx) in block_w.clone().enumerate() {
                            chunk_diag[global_idx] += row_hessian
                                [[primary_w.start + local_idx, primary_w.start + local_idx]];
                        }
                    }
                }
                Ok(chunk_diag)
            })
            .collect();
        let mut diagonal = Array1::<f64>::zeros(slices.total);
        for result in chunk_results {
            diagonal += &result?;
        }
        Ok(diagonal)
    }

    fn exact_newton_joint_psi_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, _)) =
            self.embedded_psi_vector(0, slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let chunk_results: Vec<Result<(f64, Array1<f64>, Array2<f64>), String>> = (0..((n
            + ROW_CHUNK_SIZE
            - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut c_obj = 0.0;
                let mut c_score = Array1::<f64>::zeros(slices.total);
                let mut c_hess = Array2::<f64>::zeros((slices.total, slices.total));
                for row in start..end {
                    let Some(dir) = self.row_primary_psi_direction(
                        row,
                        block_states,
                        derivative_blocks,
                        psi_index,
                        primary,
                    )?
                    else {
                        continue;
                    };
                    let row_ctx = self.build_row_exact_context(
                        row,
                        block_states,
                        &cache.h_nodes,
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let (f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                        row,
                        block_states,
                        primary,
                        &row_ctx,
                        &cache.h_nodes,
                        cache.h_node_design.as_ref(),
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &dir,
                    )?;
                    let (_, left_vec) = self
                        .embedded_psi_vector(row, slices, derivative_blocks, psi_index)?
                        .ok_or_else(|| "missing bernoulli marginal-slope psi vector".to_string())?;
                    c_obj += f_pi.dot(&dir);
                    c_score += &(left_vec.clone() * f_pi[idx_primary]);
                    c_score +=
                        &self.pullback_primary_vector(row, slices, primary, &f_pipi.dot(&dir));

                    let right_vec = self.pullback_primary_vector(
                        row,
                        slices,
                        primary,
                        &f_pipi.row(idx_primary).to_owned(),
                    );
                    c_hess += &left_vec
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&right_vec.view().insert_axis(Axis(0)));
                    c_hess += &right_vec
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&left_vec.view().insert_axis(Axis(0)));
                    self.add_pullback_primary_hessian(&mut c_hess, row, slices, primary, &third);
                }
                Ok((c_obj, c_score, c_hess))
            })
            .collect();
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(slices.total);
        let mut hessian_psi = Array2::<f64>::zeros((slices.total, slices.total));
        for result in chunk_results {
            let (c_obj, c_score, c_hess) = result?;
            objective_psi += c_obj;
            score_psi += &c_score;
            hessian_psi += &c_hess;
        }
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_i, _)) = self.embedded_psi_vector(0, slices, derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_j, _)) = self.embedded_psi_vector(0, slices, derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let idx_i = if block_i == 0 { 0 } else { 1 };
        let idx_j = if block_j == 0 { 0 } else { 1 };
        let n = self.y.len();
        let chunk_results: Vec<Result<(f64, Array1<f64>, Array2<f64>), String>> = (0..((n
            + ROW_CHUNK_SIZE
            - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut c_obj = 0.0;
                let mut c_score = Array1::<f64>::zeros(slices.total);
                let mut c_hess = Array2::<f64>::zeros((slices.total, slices.total));
                for row in start..end {
                    let Some(dir_i) = self.row_primary_psi_direction(
                        row,
                        block_states,
                        derivative_blocks,
                        psi_i,
                        primary,
                    )?
                    else {
                        continue;
                    };
                    let Some(dir_j) = self.row_primary_psi_direction(
                        row,
                        block_states,
                        derivative_blocks,
                        psi_j,
                        primary,
                    )?
                    else {
                        continue;
                    };
                    let dir_ij = self
                        .row_primary_psi_second_direction(
                            row,
                            block_states,
                            derivative_blocks,
                            psi_i,
                            psi_j,
                            primary,
                        )?
                        .unwrap_or_else(|| Array1::<f64>::zeros(primary.total));
                    let row_ctx = self.build_row_exact_context(
                        row,
                        block_states,
                        &cache.h_nodes,
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let (f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                        row,
                        block_states,
                        primary,
                        &row_ctx,
                        &cache.h_nodes,
                        cache.h_node_design.as_ref(),
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let third_i = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &dir_i,
                    )?;
                    let third_j = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &dir_j,
                    )?;
                    let fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &dir_i,
                        &dir_j,
                    )?;
                    let (_, left_i) = self
                        .embedded_psi_vector(row, slices, derivative_blocks, psi_i)?
                        .ok_or_else(|| {
                            "missing bernoulli marginal-slope psi_i vector".to_string()
                        })?;
                    let (_, left_j) = self
                        .embedded_psi_vector(row, slices, derivative_blocks, psi_j)?
                        .ok_or_else(|| {
                            "missing bernoulli marginal-slope psi_j vector".to_string()
                        })?;
                    let left_ij = self
                        .embedded_psi_second_vector(row, slices, derivative_blocks, psi_i, psi_j)?
                        .map(|(_, v)| v)
                        .unwrap_or_else(|| Array1::<f64>::zeros(slices.total));

                    c_obj += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);
                    if left_ij.iter().any(|v| v.abs() > 0.0) {
                        let idx_ij = if left_ij
                            .slice(s![slices.marginal.clone()])
                            .iter()
                            .any(|v| v.abs() > 0.0)
                        {
                            0
                        } else {
                            1
                        };
                        c_score += &(left_ij.clone() * f_pi[idx_ij]);
                    }
                    c_score += &(left_i.clone() * f_pipi.row(idx_i).dot(&dir_j));
                    c_score += &(left_j.clone() * f_pipi.row(idx_j).dot(&dir_i));
                    c_score +=
                        &self.pullback_primary_vector(row, slices, primary, &f_pipi.dot(&dir_ij));
                    c_score +=
                        &self.pullback_primary_vector(row, slices, primary, &third_i.dot(&dir_j));

                    if left_ij.iter().any(|v| v.abs() > 0.0) {
                        let idx_ij = if left_ij
                            .slice(s![slices.marginal.clone()])
                            .iter()
                            .any(|v| v.abs() > 0.0)
                        {
                            0
                        } else {
                            1
                        };
                        let right_ij = self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &f_pipi.row(idx_ij).to_owned(),
                        );
                        c_hess += &left_ij
                            .view()
                            .insert_axis(Axis(1))
                            .dot(&right_ij.view().insert_axis(Axis(0)));
                        c_hess += &right_ij
                            .view()
                            .insert_axis(Axis(1))
                            .dot(&left_ij.view().insert_axis(Axis(0)));
                    }

                    let scalar_ij = f_pipi[[idx_i, idx_j]];
                    c_hess += &(left_i
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&left_j.view().insert_axis(Axis(0)))
                        * scalar_ij);
                    c_hess += &(left_j
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&left_i.view().insert_axis(Axis(0)))
                        * scalar_ij);

                    let right_i = self.pullback_primary_vector(
                        row,
                        slices,
                        primary,
                        &third_j.row(idx_i).to_owned(),
                    );
                    c_hess += &left_i
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&right_i.view().insert_axis(Axis(0)));
                    c_hess += &right_i
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&left_i.view().insert_axis(Axis(0)));

                    let right_j = self.pullback_primary_vector(
                        row,
                        slices,
                        primary,
                        &third_i.row(idx_j).to_owned(),
                    );
                    c_hess += &left_j
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&right_j.view().insert_axis(Axis(0)));
                    c_hess += &right_j
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&left_j.view().insert_axis(Axis(0)));

                    self.add_pullback_primary_hessian(&mut c_hess, row, slices, primary, &fourth);
                    let third_ij = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &dir_ij,
                    )?;
                    self.add_pullback_primary_hessian(&mut c_hess, row, slices, primary, &third_ij);
                }
                Ok((c_obj, c_score, c_hess))
            })
            .collect();
        let mut objective_psi_psi = 0.0;
        let mut score_psi_psi = Array1::<f64>::zeros(slices.total);
        let mut hessian_psi_psi = Array2::<f64>::zeros((slices.total, slices.total));
        for result in chunk_results {
            let (c_obj, c_score, c_hess) = result?;
            objective_psi_psi += c_obj;
            score_psi_psi += &c_score;
            hessian_psi_psi += &c_hess;
        }
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
        }))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, _)) =
            self.embedded_psi_vector(0, slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let chunk_results: Vec<Result<Array2<f64>, String>> = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut c_out = Array2::<f64>::zeros((slices.total, slices.total));
                for row in start..end {
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let Some(psi_dir) = self.row_primary_psi_direction(
                        row,
                        block_states,
                        derivative_blocks,
                        psi_index,
                        primary,
                    )?
                    else {
                        continue;
                    };
                    let psi_action = self
                        .row_primary_psi_action_on_direction(
                            row,
                            slices,
                            derivative_blocks,
                            psi_index,
                            d_beta_flat,
                            primary,
                        )?
                        .unwrap_or_else(|| Array1::<f64>::zeros(primary.total));
                    let row_ctx = self.build_row_exact_context(
                        row,
                        block_states,
                        &cache.h_nodes,
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let third_beta = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &row_dir,
                    )?;
                    let fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &row_dir,
                        &psi_dir,
                    )?;
                    let (_, left_vec) = self
                        .embedded_psi_vector(row, slices, derivative_blocks, psi_index)?
                        .ok_or_else(|| "missing bernoulli marginal-slope psi vector".to_string())?;
                    let right_vec = self.pullback_primary_vector(
                        row,
                        slices,
                        primary,
                        &third_beta.row(idx_primary).to_owned(),
                    );
                    c_out += &left_vec
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&right_vec.view().insert_axis(Axis(0)));
                    c_out += &right_vec
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&left_vec.view().insert_axis(Axis(0)));
                    self.add_pullback_primary_hessian(&mut c_out, row, slices, primary, &fourth);
                    let third_action = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &psi_action,
                    )?;
                    self.add_pullback_primary_hessian(
                        &mut c_out,
                        row,
                        slices,
                        primary,
                        &third_action,
                    );
                }
                Ok(c_out)
            })
            .collect();
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for result in chunk_results {
            out += &result?;
        }
        Ok(Some(out))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let chunk_results: Vec<Result<Array2<f64>, String>> = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut c_out = Array2::<f64>::zeros((slices.total, slices.total));
                for row in start..end {
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let row_ctx = self.build_row_exact_context(
                        row,
                        block_states,
                        &cache.h_nodes,
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &row_dir,
                    )?;
                    self.add_pullback_primary_hessian(&mut c_out, row, slices, primary, &third);
                }
                Ok(c_out)
            })
            .collect();
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for result in chunk_results {
            out += &result?;
        }
        Ok(Some(out))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let chunk_results: Vec<Result<Array2<f64>, String>> = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut c_out = Array2::<f64>::zeros((slices.total, slices.total));
                for row in start..end {
                    let row_u =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                    let row_v =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                    let row_ctx = self.build_row_exact_context(
                        row,
                        block_states,
                        &cache.h_nodes,
                        cache.score_warp_obs.as_ref(),
                    )?;
                    let fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        &row_ctx,
                        &row_u,
                        &row_v,
                    )?;
                    self.add_pullback_primary_hessian(&mut c_out, row, slices, primary, &fourth);
                }
                Ok(c_out)
            })
            .collect();
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for result in chunk_results {
            out += &result?;
        }
        Ok(Some(out))
    }
}

impl CustomFamily for BernoulliMarginalSlopeFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_outer_derivative_order(
        &self,
        _: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        bernoulli_exact_outer_derivative_order(
            self.y.len(),
            self.score_warp
                .as_ref()
                .map(|runtime| runtime.transform.ncols())
                .unwrap_or(0),
            self.link_dev
                .as_ref()
                .map(|runtime| runtime.transform.ncols())
                .unwrap_or(0),
        )
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, gradient, hessian) = self.joint_gradient_hessian(block_states, true)?;
        let hessian = hessian.ok_or_else(|| "joint hessian unavailable".to_string())?;
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![slices.marginal.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian
                        .slice(s![slices.marginal.clone(), slices.marginal.clone()])
                        .to_owned(),
                ),
            },
            BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![slices.logslope.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian
                        .slice(s![slices.logslope.clone(), slices.logslope.clone()])
                        .to_owned(),
                ),
            },
        ];
        if let Some(h_range) = slices.h {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![h_range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian.slice(s![h_range.clone(), h_range]).to_owned(),
                ),
            });
        }
        if let Some(w_range) = slices.w {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![w_range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian.slice(s![w_range.clone(), w_range]).to_owned(),
                ),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.joint_gradient_hessian(block_states, false)
            .map(|r| r.0)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_gradient_hessian(block_states, true)
            .map(|(_, _, h)| h)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        Ok(Some(Arc::new(
            BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
                self.clone(),
                block_states.to_vec(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_hessian_directional_derivative_from_cache(
            block_states,
            d_beta_flat,
            &cache,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_hessiansecond_directional_derivative_from_cache(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
            &cache,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psi_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            &cache,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psisecond_order_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &cache,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psihessian_directional_derivative_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &cache,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(
            BernoulliMarginalSlopeExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
        if slices.h.as_ref().is_some_and(|_| block_idx == 2) {
            return match self.score_warp_constraint_points()? {
                Some(points) => Self::deviation_constraints_from_points(
                    self.score_warp
                        .as_ref()
                        .ok_or_else(|| "score-warp runtime missing".to_string())?,
                    points,
                ),
                None => Ok(None),
            };
        }
        let link_block_idx = if slices.h.is_some() { 3 } else { 2 };
        if slices
            .w
            .as_ref()
            .is_some_and(|_| block_idx == link_block_idx)
        {
            return match self.link_dev_constraint_points(block_states)? {
                Some(points) => Self::deviation_constraints_from_points(
                    self.link_dev
                        .as_ref()
                        .ok_or_else(|| "link-deviation runtime missing".to_string())?,
                    points,
                ),
                None => Ok(None),
            };
        }
        Ok(None)
    }
}

impl BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let cache = family.build_exact_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            cache,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_matvec_from_cache(
                beta_flat,
                &self.block_states,
                &self.cache,
            )
            .map(Some)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_diagonal_from_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_cache(
                &self.block_states,
                d_beta_flat,
                &self.cache,
            )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_cache(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &self.cache,
            )
    }
}

impl BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let cache = family.build_exact_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            cache,
        })
    }
}

impl ExactNewtonJointPsiWorkspace for BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family.exact_newton_joint_psi_terms_from_cache(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            &self.cache,
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family
            .exact_newton_joint_psisecond_order_terms_from_cache(
                &self.block_states,
                &self.derivative_blocks,
                psi_i,
                psi_j,
                &self.cache,
            )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_psihessian_directional_derivative_from_cache(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
                &self.cache,
            )
    }
}

fn build_blockspec(
    name: &str,
    design: &TermCollectionDesign,
    baseline: f64,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: design.design.clone(),
        offset: Array1::from_elem(design.design.nrows(), baseline),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_aux_blockspec(
    name: &str,
    prepared: &DeviationPrepared,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> Result<ParameterBlockSpec, String> {
    let mut block = prepared.block.clone();
    block.initial_log_lambdas = Some(rho);
    block.initial_beta = beta_hint.or_else(|| block.initial_beta.clone());
    block.intospec(name)
}

fn inner_fit(
    family: &BernoulliMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}

pub fn fit_bernoulli_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: BernoulliMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    validate_spec(data, &spec)?;
    let baseline = pooled_probit_baseline(&spec.y, &spec.z, &spec.weights)?;
    let marginal_design =
        build_term_collection_design(data, &spec.marginalspec).map_err(|e| e.to_string())?;
    let logslope_design =
        build_term_collection_design(data, &spec.logslopespec).map_err(|e| e.to_string())?;
    let marginalspec_boot =
        freeze_spatial_length_scale_terms_from_design(&spec.marginalspec, &marginal_design)
            .map_err(|e| e.to_string())?;
    let logslopespec_boot =
        freeze_spatial_length_scale_terms_from_design(&spec.logslopespec, &logslope_design)
            .map_err(|e| e.to_string())?;

    let (quad_nodes, quad_weights) = normal_expectation_nodes(spec.quadrature_points);

    let score_warp_prepared = spec
        .score_warp
        .as_ref()
        .map(|cfg| build_deviation_block_from_seed(&spec.z, cfg))
        .transpose()?;
    let rigid_family = BernoulliMarginalSlopeFamily {
        y: spec.y.clone(),
        weights: spec.weights.clone(),
        z: spec.z.clone(),
        marginal_design: marginal_design.design.clone(),
        logslope_design: logslope_design.design.clone(),
        quadrature_nodes: quad_nodes.clone(),
        quadrature_weights: quad_weights.clone(),
        score_warp: None,
        score_warp_obs_design: None,
        link_dev: None,
    };
    let rigid_blocks = vec![
        build_blockspec(
            "marginal_surface",
            &marginal_design,
            baseline.0,
            Array1::zeros(marginal_design.penalties.len()),
            None,
        ),
        build_blockspec(
            "logslope_surface",
            &logslope_design,
            baseline.1,
            Array1::zeros(logslope_design.penalties.len()),
            None,
        ),
    ];
    let rigid_fit = inner_fit(&rigid_family, &rigid_blocks, options)?;
    let q0_seed = {
        let marginal_eta = &rigid_fit.block_states[0].eta;
        let logslope_eta = &rigid_fit.block_states[1].eta;
        Array1::from_iter((0..marginal_eta.len()).map(|i| {
            let b = logslope_eta[i].exp();
            marginal_eta[i] * (1.0 + b * b).sqrt() + b * spec.z[i]
        }))
    };
    let link_dev_prepared = spec
        .link_dev
        .as_ref()
        .map(|cfg| build_deviation_block_from_seed(&q0_seed, cfg))
        .transpose()?;

    let extra_rho0 = {
        let mut out = Vec::new();
        if let Some(ref prepared) = score_warp_prepared {
            out.extend(std::iter::repeat(0.0).take(prepared.block.penalties.len()));
        }
        if let Some(ref prepared) = link_dev_prepared {
            out.extend(std::iter::repeat(0.0).take(prepared.block.penalties.len()));
        }
        out
    };
    let setup = joint_setup(
        &marginalspec_boot,
        &logslopespec_boot,
        marginal_design.penalties.len(),
        logslope_design.penalties.len(),
        &extra_rho0,
        kappa_options,
    );
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);
    let hints = RefCell::new(ThetaHints::default());
    let y = spec.y.clone();
    let weights = spec.weights.clone();
    let z = spec.z.clone();
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let score_warp_obs_design = score_warp_prepared
        .as_ref()
        .and_then(|p| match &p.block.design {
            DesignMatrix::Dense(x) => Some(x.to_dense()),
            DesignMatrix::Sparse(_) => None,
        });
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        cursor += logslope_design.penalties.len();
        let mut blocks = vec![
            build_blockspec(
                "marginal_surface",
                marginal_design,
                baseline.0,
                rho_marginal,
                hints.marginal_beta.clone(),
            ),
            build_blockspec(
                "logslope_surface",
                logslope_design,
                baseline.1,
                rho_logslope,
                hints.logslope_beta.clone(),
            ),
        ];
        if let Some(ref prepared) = score_warp_prepared {
            let rho_h = rho
                .slice(s![cursor..cursor + prepared.block.penalties.len()])
                .to_owned();
            cursor += prepared.block.penalties.len();
            blocks.push(build_aux_blockspec(
                "score_warp_dev",
                prepared,
                rho_h,
                hints.score_warp_beta.clone(),
            )?);
        }
        if let Some(ref prepared) = link_dev_prepared {
            let rho_w = rho
                .slice(s![cursor..cursor + prepared.block.penalties.len()])
                .to_owned();
            blocks.push(build_aux_blockspec(
                "link_dev",
                prepared,
                rho_w,
                hints.link_dev_beta.clone(),
            )?);
        }
        Ok(blocks)
    };

    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign|
     -> BernoulliMarginalSlopeFamily {
        BernoulliMarginalSlopeFamily {
            y: y.clone(),
            weights: weights.clone(),
            z: z.clone(),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            quadrature_nodes: quad_nodes.clone(),
            quadrature_weights: quad_weights.clone(),
            score_warp: score_warp_runtime.clone(),
            score_warp_obs_design: score_warp_obs_design.clone(),
            link_dev: link_dev_runtime.clone(),
        }
    };

    let marginal_psi_result =
        build_block_spatial_psi_derivatives(data, &marginalspec_boot, &marginal_design);
    let logslope_psi_result =
        build_block_spatial_psi_derivatives(data, &logslopespec_boot, &logslope_design);
    // Track which blocks actually have spatial psi-dependence.  A block that is
    // entirely non-spatial returns Ok(None), which is fine — its psi contribution
    // is identically zero.  We only require that at least one block has spatial
    // derivatives when the model has kappa dimensions to optimize.
    let marginal_has_spatial = marginal_psi_result
        .as_ref()
        .map(|r| r.is_some())
        .unwrap_or(false);
    let logslope_has_spatial = logslope_psi_result
        .as_ref()
        .map(|r| r.is_some())
        .unwrap_or(false);
    let analytic_joint_derivatives_available = marginal_psi_result.is_ok()
        && logslope_psi_result.is_ok()
        && (marginal_has_spatial || logslope_has_spatial);
    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err(
            "exact bernoulli marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }
    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design);
    let joint_cap = custom_family_outer_capability(
        &initial_family,
        &initial_blocks,
        options,
        setup.theta0().len(),
        setup.log_kappa_dim() > 0,
    );
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_cap.gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let analytic_joint_hessian_available = analytic_joint_derivatives_available
        && matches!(
            joint_cap.hessian,
            crate::solver::outer_strategy::Derivative::Analytic
        );

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms, logslope_terms],
        kappa_options,
        &setup,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        |rho, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            let mut bidx = 0usize;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.marginal_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.logslope_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if score_warp_prepared.is_some() {
                if let Some(block) = fit.block_states.get(bidx) {
                    hints_mut.score_warp_beta = Some(block.beta.clone());
                }
                bidx += 1;
            }
            if link_dev_prepared.is_some() {
                if let Some(block) = fit.block_states.get(bidx) {
                    hints_mut.link_dev_beta = Some(block.beta.clone());
                }
            }
            Ok(fit)
        },
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
            // For blocks with spatial terms, require derivatives (error if missing).
            // For non-spatial blocks, use an empty vec (zero psi contribution).
            let marginal_psi_derivs = if marginal_has_spatial {
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: marginal block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let logslope_psi_derivs = if logslope_has_spatial {
                build_block_spatial_psi_derivatives(data, &specs[1], &designs[1])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: logslope block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let mut derivative_blocks = vec![marginal_psi_derivs, logslope_psi_derivs];
            if family.score_warp.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if family.link_dev.is_some() {
                derivative_blocks.push(Vec::new());
            }
            let eval = evaluate_custom_family_joint_hyper(
                &family,
                &blocks,
                options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                need_hessian,
            )?;
            exact_warm_start.replace(Some(eval.warm_start));
            if need_hessian && eval.outer_hessian.is_none() {
                return Err(
                    "exact bernoulli marginal-slope joint [rho, psi] objective did not return an outer Hessian"
                        .to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    Ok(BernoulliMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs.remove(0),
        logslope_design: designs.remove(0),
        baseline_marginal: baseline.0,
        baseline_logslope: baseline.1,
        score_warp_runtime,
        link_dev_runtime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::CustomFamily;
    use ndarray::array;

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn dummy_blockspec(p: usize, n_rows: usize) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: "dummy".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::<f64>::zeros((n_rows, p)),
            )),
            offset: Array1::zeros(n_rows),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::zeros(p)),
        }
    }

    fn dummy_block_state(beta: Array1<f64>, n_rows: usize) -> ParameterBlockState {
        ParameterBlockState {
            beta,
            eta: Array1::zeros(n_rows),
        }
    }

    fn base_spec(
        y: Array1<f64>,
        weights: Array1<f64>,
        z: Array1<f64>,
    ) -> BernoulliMarginalSlopeTermSpec {
        BernoulliMarginalSlopeTermSpec {
            y,
            weights,
            z,
            marginalspec: empty_termspec(),
            logslopespec: empty_termspec(),
            score_warp: None,
            link_dev: None,
            quadrature_points: 7,
        }
    }

    fn pair_distance(lhs: (f64, f64), rhs: (f64, f64)) -> f64 {
        (lhs.0 - rhs.0).abs() + (lhs.1 - rhs.1).abs()
    }

    fn max_abs(matrix: &Array2<f64>) -> f64 {
        matrix.iter().fold(0.0, |acc, value| acc.max(value.abs()))
    }

    fn expand_integer_weight_rows(
        y: &Array1<f64>,
        z: &Array1<f64>,
        weights: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut y_expanded = Vec::new();
        let mut z_expanded = Vec::new();
        for i in 0..y.len() {
            let reps = weights[i] as usize;
            assert!(
                (weights[i] - reps as f64).abs() < 1e-12,
                "test helper expects integer weights, got {}",
                weights[i]
            );
            for _ in 0..reps {
                y_expanded.push(y[i]);
                z_expanded.push(z[i]);
            }
        }
        (Array1::from_vec(y_expanded), Array1::from_vec(z_expanded))
    }

    #[test]
    fn link_dev_without_score_warp_uses_w_block_and_constraints() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link block initial beta")
            .len();
        let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.1 * (idx as f64 + 1.0)));
        let family = BernoulliMarginalSlopeFamily {
            y: Array1::zeros(seed.len()),
            weights: Array1::ones(seed.len()),
            z: seed.clone(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            quadrature_nodes: array![0.0],
            quadrature_weights: array![1.0],
            score_warp: None,
            score_warp_obs_design: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_link.clone(), seed.len()),
        ];

        let slices = block_slices(&block_states, false, true);
        assert!(slices.h.is_none(), "score-warp slice should be absent");
        let link_slice = slices.w.as_ref().expect("link slice");
        assert_eq!(link_slice.start, 2, "link-only block should occupy block 2");
        assert_eq!(link_slice.len(), link_dim);

        let primary = primary_slices(&slices);
        assert!(primary.h.is_none(), "primary h slice should be absent");
        let primary_w = primary.w.as_ref().expect("primary link slice");
        assert_eq!(primary_w.start, 2, "primary link slice should start at 2");
        assert_eq!(primary.total, 2 + link_dim);

        let dirs = (0..primary.total)
            .map(|idx| unit_primary_direction(&primary, idx))
            .collect::<Vec<_>>();
        let gamma_jets = family.build_gamma_jets(&primary, Some(&beta_link), &dirs);
        assert_eq!(
            gamma_jets.len(),
            link_dim,
            "link-only layout must produce one gamma jet per link coefficient"
        );

        let basis_row = prepared
            .runtime
            .design(&array![0.25])
            .expect("link basis row")
            .row(0)
            .to_vec();
        let zeros = vec![0.0; link_dim];
        let eta_jet = MultiDirJet::constant(dirs.len(), 0.2);
        let link_jet = family.apply_link_jet_from_rows(
            &eta_jet,
            &gamma_jets,
            &basis_row,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
        );
        assert!(
            link_jet.coeff(0).is_finite(),
            "link jet evaluation should remain finite with link_dev only"
        );

        let dummy_spec = dummy_blockspec(link_dim, seed.len());
        let constraints = family
            .block_linear_constraints(&block_states, 2, &dummy_spec)
            .expect("constraint lookup")
            .expect("link block constraints");
        let exact_points = family
            .link_dev_constraint_points(&block_states)
            .expect("exact link constraint points")
            .expect("non-empty link constraint points");
        assert_eq!(constraints.a.ncols(), link_dim);
        assert_eq!(constraints.a.nrows(), exact_points.len());
        assert_eq!(
            constraints.a,
            prepared
                .runtime
                .first_derivative_design(&exact_points)
                .expect("link derivative design"),
        );
        assert_eq!(
            constraints.b,
            Array1::from_elem(exact_points.len(), prepared.runtime.monotonicity_eps - 1.0),
        );
        assert!(
            family
                .block_linear_constraints(&block_states, 1, &dummy_spec)
                .expect("non-link constraint lookup")
                .is_none(),
            "only block 2 should expose link constraints when score_warp is absent"
        );
    }

    #[test]
    fn score_warp_constraints_use_observed_and_quadrature_points() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let family = BernoulliMarginalSlopeFamily {
            y: Array1::zeros(seed.len()),
            weights: Array1::ones(seed.len()),
            z: seed.clone(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            quadrature_nodes: array![-0.25, 0.25],
            quadrature_weights: array![0.5, 0.5],
            score_warp: Some(prepared.runtime.clone()),
            score_warp_obs_design: Some(prepared.block.design.to_dense().as_ref().clone()),
            link_dev: None,
        };
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(Array1::zeros(score_dim), seed.len()),
        ];

        let points = family
            .score_warp_constraint_points()
            .expect("score-warp points")
            .expect("non-empty score-warp points");
        assert!(
            points.iter().any(|value| (*value - 0.25).abs() < 1e-12),
            "quadrature nodes should be included in the exact score-warp constraint domain"
        );

        let dummy_spec = dummy_blockspec(score_dim, seed.len());
        let constraints = family
            .block_linear_constraints(&block_states, 2, &dummy_spec)
            .expect("constraint lookup")
            .expect("score-warp constraints");
        assert_eq!(constraints.a.ncols(), score_dim);
        assert_eq!(constraints.a.nrows(), points.len());
        assert_eq!(
            constraints.a,
            prepared
                .runtime
                .first_derivative_design(&points)
                .expect("score-warp derivative design"),
        );
    }

    #[test]
    fn pooled_probit_baseline_matches_expanded_integer_weight_fit() {
        let y = array![0.0, 1.0, 0.0, 1.0];
        let z = array![-1.5, -0.2, 0.4, 1.4];
        let weights = array![25.0, 2.0, 1.0, 20.0];
        let weighted = pooled_probit_baseline(&y, &z, &weights).expect("weighted baseline");
        let unweighted =
            pooled_probit_baseline(&y, &z, &Array1::ones(y.len())).expect("unweighted baseline");
        let (y_expanded, z_expanded) = expand_integer_weight_rows(&y, &z, &weights);
        let expanded =
            pooled_probit_baseline(&y_expanded, &z_expanded, &Array1::ones(y_expanded.len()))
                .expect("expanded baseline");

        assert!(
            pair_distance(expanded, unweighted) > 1e-2,
            "test data should distinguish weighted from unweighted seeding"
        );
        assert!(
            pair_distance(weighted, expanded) < 1e-8,
            "weighted pilot baseline should match the expanded integer-weight fit"
        );
    }

    #[test]
    fn deviation_transform_enforces_constraints_at_seed_median_anchor() {
        let seed = array![4.5, 5.0, 6.0, 7.5, 8.0];
        let degree = 3usize;
        let knots = initializewiggle_knots_from_seed(seed.view(), degree, 4)
            .expect("initialize deviation knots");
        let transform = deviation_transform(&knots, degree, &seed).expect("deviation transform");
        let anchor = array![6.0];
        let (value_basis, _) = create_basis::<Dense>(
            anchor.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .expect("median value basis");
        let (d1_basis, _) = create_basis::<Dense>(
            anchor.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .expect("median derivative basis");

        let anchored_value = value_basis.dot(&transform);
        let anchored_d1 = d1_basis.dot(&transform);
        assert!(
            max_abs(&anchored_value) < 1e-8,
            "value constraint should be imposed at the seed median anchor"
        );
        assert!(
            max_abs(&anchored_d1) < 1e-8,
            "derivative constraint should be imposed at the seed median anchor"
        );
    }

    #[test]
    fn validate_spec_rejects_nonfinite_or_negative_weights() {
        let data = Array2::<f64>::zeros((3, 0));
        let y = array![0.0, 1.0, 0.0];
        let z = array![-1.0, 0.0, 1.0];

        let err = validate_spec(
            data.view(),
            &base_spec(y.clone(), array![1.0, f64::NAN, 1.0], z.clone()),
        )
        .expect_err("non-finite weights should be rejected");
        assert!(err.contains("finite non-negative weights"));

        let err = validate_spec(data.view(), &base_spec(y, array![1.0, -0.5, 1.0], z))
            .expect_err("negative weights should be rejected");
        assert!(err.contains("finite non-negative weights"));
    }

    #[test]
    fn validate_spec_rejects_nonfinite_z_values() {
        let data = Array2::<f64>::zeros((3, 0));
        let err = validate_spec(
            data.view(),
            &base_spec(
                array![0.0, 1.0, 0.0],
                array![1.0, 1.0, 1.0],
                array![-1.0, f64::INFINITY, 1.0],
            ),
        )
        .expect_err("non-finite z should be rejected");
        assert!(err.contains("finite z values"));
    }

    #[test]
    fn exact_outer_derivative_order_keeps_small_primary_models_second_order() {
        assert_eq!(
            bernoulli_exact_outer_derivative_order(100_000, 0, 0),
            ExactOuterDerivativeOrder::Second
        );
        assert_eq!(
            bernoulli_exact_outer_derivative_order(4_000, 8, 6),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn exact_outer_derivative_order_downgrades_large_rich_models_to_first_order() {
        assert_eq!(
            bernoulli_exact_outer_derivative_order(10_000, 8, 6),
            ExactOuterDerivativeOrder::First
        );
    }

    /// Exercises the w-only (link_dev without score_warp) layout through the
    /// full gradient + Hessian path, verifying that:
    ///   (a) no index-out-of-bounds panic occurs,
    ///   (b) all outputs are finite,
    ///   (c) the Hessian is symmetric.
    ///
    /// This guards against arity bookkeeping bugs where the directional-jet or
    /// block-slice code assumes both h and w blocks are present.
    #[test]
    fn w_only_gradient_hessian_finite_and_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        // Non-trivial link coefficients to exercise all jet branches.
        let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0)));

        let family = BernoulliMarginalSlopeFamily {
            y: array![0.0, 1.0, 0.0, 1.0, 0.0],
            weights: Array1::ones(seed.len()),
            z: seed.clone(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            quadrature_nodes: array![0.0],
            quadrature_weights: array![1.0],
            score_warp: None,
            score_warp_obs_design: None,
            link_dev: Some(prepared.runtime.clone()),
        };

        // Three blocks: marginal (dim 0), logslope (dim 0), link_dev.
        // eta is irrelevant for zero-column designs; use zeros.
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_link.clone(), seed.len()),
        ];

        let slices = block_slices(&block_states, false, true);
        assert!(slices.h.is_none(), "score-warp absent → no h slice");
        let primary = primary_slices(&slices);
        assert!(primary.h.is_none(), "primary h absent");
        assert_eq!(primary.total, 2 + link_dim);

        let h_nodes = &family.quadrature_nodes;

        // Exercise every row — different z values exercise different link
        // regimes (negative tail, near zero, positive tail).
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states, h_nodes, None)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));

            let (grad, hess) = family
                .compute_row_primary_gradient_hessian(
                    row,
                    &block_states,
                    &primary,
                    &row_ctx,
                    h_nodes,
                    None, // h_node_design: absent when no score_warp
                    None, // score_warp_obs: absent
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: compute_row_primary_gradient_hessian failed: {e}")
                });

            assert_eq!(
                grad.len(),
                primary.total,
                "row {row}: gradient length mismatch"
            );
            assert_eq!(
                hess.dim(),
                (primary.total, primary.total),
                "row {row}: hessian shape mismatch"
            );
            assert!(
                grad.iter().all(|v| v.is_finite()),
                "row {row}: non-finite gradient entry: {grad:?}"
            );
            assert!(
                hess.iter().all(|v| v.is_finite()),
                "row {row}: non-finite hessian entry"
            );

            // Symmetry check.
            for a in 0..primary.total {
                for b in 0..a {
                    let diff = (hess[[a, b]] - hess[[b, a]]).abs();
                    assert!(
                        diff < 1e-10,
                        "row {row}: hessian asymmetry at ({a},{b}): \
                         H[{a},{b}]={:.6e} vs H[{b},{a}]={:.6e}, diff={diff:.3e}",
                        hess[[a, b]],
                        hess[[b, a]]
                    );
                }
            }
        }
    }
}
