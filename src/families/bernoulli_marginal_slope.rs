use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    evaluate_bspline_fourth_derivative_scalar, evaluate_bsplinethird_derivative_scalar,
};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyJointDesignChannel,
    CustomFamilyJointDesignPairContribution, CustomFamilyJointPsiOperator,
    CustomFamilyPsiDesignAction, CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart,
    ExactNewtonJointHessianWorkspace, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace, ExactOuterDerivativeOrder,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, build_block_spatial_psi_derivatives,
    cost_gated_outer_order, custom_family_outer_capability, evaluate_custom_family_joint_hyper,
    first_psi_linear_map, fit_custom_family, second_psi_linear_map,
};
use crate::estimate::reml::unified::HyperOperator;
use crate::estimate::{FitOptions, UnifiedFitResult, fit_gam};
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::gamlss::{ParameterBlockInput, initializewiggle_knots_from_seed};
use crate::families::row_kernel::{RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf};
use crate::quadrature::compute_gauss_hermite_n;
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
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
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    z: Arc<Array1<f64>>,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    quadrature_nodes: Array1<f64>,
    quadrature_weights: Array1<f64>,
    score_warp: Option<DeviationRuntime>,
    score_warp_obs_design: Option<DesignMatrix>,
    /// Precomputed score-warp monotonicity constraints (state-independent).
    /// Built once at family construction from z, quadrature nodes, and the
    /// dense support grid.
    cached_score_warp_constraints: Option<LinearInequalityConstraints>,
    link_dev: Option<DeviationRuntime>,
    /// Precomputed link-deviation monotonicity constraints (state-independent).
    /// Built once at family construction from the dense knot-span grid.
    cached_link_dev_constraints: Option<LinearInequalityConstraints>,
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

    /// Return a knot-span-aware grid over the support interval, suitable for
    /// derivative-based monotonicity constraints.  Every active spline span
    /// contributes both endpoints and evenly spaced interior points, which is
    /// materially stronger than a purely uniform support sweep because it
    /// forces explicit coverage on each piecewise-polynomial interval.
    fn dense_constraint_grid(&self, per_span: usize) -> Result<Array1<f64>, String> {
        let (left, right) = self.support_interval()?;
        let samples_per_span = per_span.max(3);
        let mut pts = Vec::new();
        let active_knots = &self
            .knots
            .slice(s![self.degree..=self.knots.len() - self.degree - 1]);
        let mut span_added = false;
        for window in active_knots.windows(2) {
            let lo = window[0].max(left);
            let hi = window[1].min(right);
            if !lo.is_finite() || !hi.is_finite() || hi < lo {
                continue;
            }
            if (hi - lo).abs() <= 1e-12 {
                pts.push(lo);
                continue;
            }
            span_added = true;
            for idx in 0..=samples_per_span {
                let t = idx as f64 / samples_per_span as f64;
                pts.push(lo + (hi - lo) * t);
            }
        }
        if !span_added {
            pts.push(left);
            pts.push(right);
        }
        self.supported_unique_values_from_iter(pts)
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
    // Enforce z is approximately standard normal.
    // The Gaussian decoupling identity E[Φ(a + β Z)] = Φ(a / √(1+β²))
    // holds only when Z ~ N(0, 1).  Checking mean ≈ 0, sd ≈ 1 is necessary
    // but NOT sufficient: a standardized non-normal distribution would pass
    // those checks but invalidate the closed form.  We additionally check
    // weighted skewness ≈ 0 and excess kurtosis ≈ 0 to catch the most
    // common violations (e.g. heavy tails, asymmetry, discrete scores).
    let weight_sum = spec.weights.iter().copied().sum::<f64>();
    if weight_sum.is_finite() && weight_sum > 0.0 {
        let mean = spec
            .z
            .iter()
            .zip(spec.weights.iter())
            .map(|(&zi, &wi)| wi * zi)
            .sum::<f64>()
            / weight_sum;
        let var = spec
            .z
            .iter()
            .zip(spec.weights.iter())
            .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
            .sum::<f64>()
            / weight_sum;
        let sd = var.sqrt();
        if mean.abs() > 1e-6 || (sd - 1.0).abs() > 1e-6 {
            return Err(format!(
                "bernoulli-marginal-slope requires z to already represent a latent N(0,1) score; \
                 weighted mean 0 and weighted sd 1 are necessary sanity checks. got mean={mean:.6e}, sd={sd:.6e}"
            ));
        }
        // Weighted skewness and excess kurtosis as normality gates.
        // For Z ~ N(0,1): skewness = 0, excess kurtosis = 0.
        if sd > 1e-12 {
            let skew = spec
                .z
                .iter()
                .zip(spec.weights.iter())
                .map(|(&zi, &wi)| wi * ((zi - mean) / sd).powi(3))
                .sum::<f64>()
                / weight_sum;
            let kurt = spec
                .z
                .iter()
                .zip(spec.weights.iter())
                .map(|(&zi, &wi)| wi * ((zi - mean) / sd).powi(4))
                .sum::<f64>()
                / weight_sum
                - 3.0;
            // Tolerances are generous for finite-sample noise but catch
            // obviously non-Gaussian shapes (e.g. uniform: kurt ≈ -1.2).
            if skew.abs() > 0.5 || kurt.abs() > 1.5 {
                log::warn!(
                    "bernoulli-marginal-slope: z has skewness={skew:.3} and \
                     excess kurtosis={kurt:.3}; the Gaussian marginalization \
                     identity E[Φ(a+βZ)]=Φ(a/√(1+β²)) is only exact for \
                     Z~N(0,1). Results may be biased if z is not approximately \
                     normal."
                );
            }
        }
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
    // Signed slope: preserve direction from pilot probit.
    let b = fit.beta.get(1).copied().unwrap_or(0.0);
    let b = if b.abs() < 1e-6 { 1e-6 } else { b };
    Ok((a / (1.0 + b * b).sqrt(), b))
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

/// Rigid probit scalar kernel: closed-form derivatives up to 4th order.
///
/// η = q·c(g) + g·z,  c(g) = √(1+g²),  s = 2y−1,  m = s·η.
/// u_k absorb weight and sign: u1=w·s·κ₁, u2=w·κ₂, u3=w·s·κ₃, u4=w·κ₄.
struct RigidProbitKernel {
    logcdf: f64,
    u1: f64,
    u2: f64,
    u3: f64,
    u4: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    eta_q: f64,
    eta_g: f64,
}

impl RigidProbitKernel {
    #[inline]
    fn new(q: f64, g: f64, z: f64, y: f64, w: f64) -> Self {
        let s = 2.0 * y - 1.0;
        let g2 = g * g;
        let c = (1.0 + g2).sqrt();
        let c1 = g / c;
        let c_inv3 = 1.0 / (c * c * c);
        let c_inv5 = c_inv3 / (c * c);
        let c_inv7 = c_inv5 / (c * c);
        let eta = q * c + g * z;
        let m = s * eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(m);
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w);
        Self {
            logcdf,
            u1: s * k1,
            u2: k2,
            u3: s * k3,
            u4: k4,
            c1,
            c2: c_inv3,
            c3: -3.0 * g * c_inv5,
            c4: (12.0 * g2 - 3.0) * c_inv7,
            eta_q: c,
            eta_g: q * c1 + z,
        }
    }

    #[inline]
    fn primary_hessian(&self, q: f64) -> [[f64; 2]; 2] {
        let h00 = self.u2 * self.eta_q * self.eta_q;
        let h01 = self.u2 * self.eta_q * self.eta_g + self.u1 * self.c1;
        let h11 = self.u2 * self.eta_g * self.eta_g + self.u1 * q * self.c2;
        [[h00, h01], [h01, h11]]
    }

    #[inline]
    fn third_contracted(&self, q: f64, dq: f64, dg: f64) -> [[f64; 2]; 2] {
        let dd = self.eta_q * dq + self.eta_g * dg;
        let dd_q = self.c1 * dg;
        let dd_g = self.c1 * dq + q * self.c2 * dg;
        let dd_qg = self.c2 * dg;
        let dd_gg = self.c2 * dq + q * self.c3 * dg;
        let t00 = self.u3 * self.eta_q * self.eta_q * dd + self.u2 * 2.0 * self.eta_q * dd_q;
        let t01 = self.u3 * self.eta_q * self.eta_g * dd
            + self.u2 * (self.c1 * dd + self.eta_q * dd_g + self.eta_g * dd_q)
            + self.u1 * dd_qg;
        let t11 = self.u3 * self.eta_g * self.eta_g * dd
            + self.u2 * (q * self.c2 * dd + 2.0 * self.eta_g * dd_g)
            + self.u1 * dd_gg;
        [[t00, t01], [t01, t11]]
    }

    #[inline]
    fn fourth_contracted(&self, q: f64, uq: f64, ug: f64, vq: f64, vg: f64) -> [[f64; 2]; 2] {
        let du = self.eta_q * uq + self.eta_g * ug;
        let dv = self.eta_q * vq + self.eta_g * vg;
        let du_a = [self.c1 * ug, self.c1 * uq + q * self.c2 * ug];
        let dv_a = [self.c1 * vg, self.c1 * vq + q * self.c2 * vg];
        let du_ab = [
            [0.0, self.c2 * ug],
            [self.c2 * ug, self.c2 * uq + q * self.c3 * ug],
        ];
        let dv_ab = [
            [0.0, self.c2 * vg],
            [self.c2 * vg, self.c2 * vq + q * self.c3 * vg],
        ];
        let dduv = self.c1 * (uq * vg + ug * vq) + q * self.c2 * ug * vg;
        let dduv_a = [
            self.c2 * ug * vg,
            self.c2 * (uq * vg + ug * vq) + q * self.c3 * ug * vg,
        ];
        let dduv_ab = [
            [0.0, self.c3 * ug * vg],
            [
                self.c3 * ug * vg,
                self.c3 * (uq * vg + ug * vq) + q * self.c4 * ug * vg,
            ],
        ];
        let eta_a = [self.eta_q, self.eta_g];
        let eta_ab = [[0.0, self.c1], [self.c1, q * self.c2]];
        let mut f = [[0.0f64; 2]; 2];
        for a in 0..2 {
            for b in a..2 {
                let val = self.u4 * eta_a[a] * eta_a[b] * du * dv
                    + self.u3
                        * (eta_ab[a][b] * du * dv
                            + du_a[a] * eta_a[b] * dv
                            + dv_a[a] * eta_a[b] * du
                            + du_a[b] * eta_a[a] * dv
                            + dv_a[b] * eta_a[a] * du
                            + dduv * eta_a[a] * eta_a[b])
                    + self.u2
                        * (eta_ab[a][b] * dduv
                            + du_a[a] * dv_a[b]
                            + dv_a[a] * du_a[b]
                            + du_ab[a][b] * dv
                            + dv_ab[a][b] * du
                            + eta_a[b] * dduv_a[a]
                            + eta_a[a] * dduv_a[b])
                    + self.u1 * dduv_ab[a][b];
                f[a][b] = val;
                f[b][a] = val;
            }
        }
        f
    }
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
/// Block-local psi derivative row: avoids allocating a full p-vector
/// when the psi derivative lives in a single channel (marginal or logslope).
struct BlockPsiRow {
    /// Which parameter block (0 = marginal, 1 = logslope).
    block_idx: usize,
    /// Coefficient range in global (flat) space for this block.
    range: std::ops::Range<usize>,
    /// The p_block-length psi design derivative row.
    local_vec: Array1<f64>,
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
// ── Block-local Hessian accumulator for Bernoulli marginal-slope ─────
//
// The two large blocks are marginal (p_m) and logslope (p_g).
// Optional h/w blocks are tiny (1-5 params each), so their contributions
// go into a dense p_total x p_total correction matrix.  The main savings
// is avoiding O(n * (p_m^2 + p_g^2)) dense accumulation into a full p*p target.

struct BernoulliBlockHessianAccumulator {
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_mg: Array2<f64>,
    dense_correction: Option<Array2<f64>>,
}

impl BernoulliBlockHessianAccumulator {
    fn new(slices: &BlockSlices) -> Self {
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let has_hw = slices.h.is_some() || slices.w.is_some();
        Self {
            h_mm: Array2::zeros((p_m, p_m)),
            h_gg: Array2::zeros((p_g, p_g)),
            h_mg: Array2::zeros((p_m, p_g)),
            dense_correction: if has_hw {
                Some(Array2::zeros((slices.total, slices.total)))
            } else {
                None
            },
        }
    }

    /// Accumulate a primary-space Hessian into block-local matrices.
    /// The marginal block uses H[0,0], logslope uses H[1,1],
    /// cross uses H[0,1].  All h/w cross-blocks go to dense_correction.
    fn add_pullback(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;

        // marginal x marginal: H[0,0] * x_row outer x_row
        family
            .marginal_design
            .syr_row_into(row, h[[0, 0]], &mut self.h_mm)
            .expect("marginal syr_row_into dimension mismatch");

        // logslope x logslope: H[1,1] * g_row outer g_row
        family
            .logslope_design
            .syr_row_into(row, h[[1, 1]], &mut self.h_gg)
            .expect("logslope syr_row_into dimension mismatch");

        // marginal x logslope: H[0,1] * x_row outer g_row
        if h[[0, 1]] != 0.0 {
            family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &family.logslope_design,
                    h[[0, 1]],
                    self.h_mg.view_mut(),
                )
                .expect("marginal-logslope row_outer_into dimension mismatch");
        }

        // h/w cross-blocks -> dense_correction
        if let Some(ref mut dc) = self.dense_correction {
            family.add_pullback_primary_hessian_hw_only(dc, row, slices, primary, h);
        }
    }

    /// Add a rank-1 update from psi_row (in the psi block) crossed with the
    /// pullback of a primary-space vector.  Adds both left outer right and right outer left.
    ///
    /// psi_row lives in block `psi_block_idx` (0=marginal, 1=logslope).
    /// right_primary is a primary-space vector; its [0] component maps to marginal,
    /// [1] to logslope, and the rest to h/w (dense correction).
    fn add_rank1_psi_cross(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        psi_block_idx: usize,
        psi_row: &Array1<f64>,
        right_primary: &Array1<f64>,
    ) {
        // Marginal component of right_primary
        if right_primary[0] != 0.0 {
            match psi_block_idx {
                0 => {
                    // psi=marginal, right=marginal -> h_mm, symmetric rank-2
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let scale = right_primary[0] * value;
                        {
                            let mut col = self.h_mm.column_mut(idx);
                            family
                                .marginal_design
                                .axpy_row_into(row, scale, &mut col)
                                .expect("marginal axpy column mismatch");
                        }
                        {
                            let mut row_view = self.h_mm.row_mut(idx);
                            family
                                .marginal_design
                                .axpy_row_into(row, scale, &mut row_view)
                                .expect("marginal axpy row mismatch");
                        }
                    }
                }
                1 => {
                    // psi=logslope, right=marginal -> h_mg (marginal x logslope)
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let mut col = self.h_mg.column_mut(idx);
                        family
                            .marginal_design
                            .axpy_row_into(row, right_primary[0] * value, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                }
                _ => {}
            }
        }

        // Logslope component of right_primary
        if right_primary[1] != 0.0 {
            match psi_block_idx {
                0 => {
                    // psi=marginal, right=logslope -> h_mg
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let mut row_view = self.h_mg.row_mut(idx);
                        family
                            .logslope_design
                            .axpy_row_into(row, right_primary[1] * value, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
                1 => {
                    // psi=logslope, right=logslope -> h_gg, symmetric rank-2
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let scale = right_primary[1] * value;
                        {
                            let mut col = self.h_gg.column_mut(idx);
                            family
                                .logslope_design
                                .axpy_row_into(row, scale, &mut col)
                                .expect("logslope axpy column mismatch");
                        }
                        {
                            let mut row_view = self.h_gg.row_mut(idx);
                            family
                                .logslope_design
                                .axpy_row_into(row, scale, &mut row_view)
                                .expect("logslope axpy row mismatch");
                        }
                    }
                }
                _ => {}
            }
        }

        // h/w components -> dense_correction
        if let Some(ref mut dc) = self.dense_correction {
            let psi_range = if psi_block_idx == 0 {
                slices.marginal.clone()
            } else {
                slices.logslope.clone()
            };
            if let (Some(ph), Some(bh)) = (primary.h.as_ref(), slices.h.as_ref()) {
                let h_part = right_primary.slice(ndarray::s![ph.start..ph.end]);
                for (li, gi) in psi_range.clone().enumerate() {
                    for (lj, gj) in bh.clone().enumerate() {
                        let val = psi_row[li] * h_part[lj];
                        dc[[gi, gj]] += val;
                        dc[[gj, gi]] += val;
                    }
                }
            }
            if let (Some(pw), Some(bw)) = (primary.w.as_ref(), slices.w.as_ref()) {
                let w_part = right_primary.slice(ndarray::s![pw.start..pw.end]);
                for (li, gi) in psi_range.enumerate() {
                    for (lj, gj) in bw.clone().enumerate() {
                        let val = psi_row[li] * w_part[lj];
                        dc[[gi, gj]] += val;
                        dc[[gj, gi]] += val;
                    }
                }
            }
        }
    }

    /// Add outer product of two psi block-local rows (possibly in different blocks).
    /// Adds both alpha * (a outer b) and alpha * (b outer a) to maintain symmetry.
    fn add_psi_psi_outer(
        &mut self,
        block_i: usize,
        psi_row_i: &Array1<f64>,
        block_j: usize,
        psi_row_j: &Array1<f64>,
        alpha: f64,
    ) {
        if alpha == 0.0 {
            return;
        }
        let col_i = psi_row_i.view().insert_axis(Axis(1));
        let row_j = psi_row_j.view().insert_axis(Axis(0));

        if block_i == block_j {
            // Same block: symmetric rank-2 update to diagonal block
            let col_j = psi_row_j.view().insert_axis(Axis(1));
            let row_i = psi_row_i.view().insert_axis(Axis(0));
            let target = match block_i {
                0 => &mut self.h_mm,
                1 => &mut self.h_gg,
                _ => return,
            };
            ndarray::linalg::general_mat_mul(alpha, &col_i, &row_j, 1.0, target);
            ndarray::linalg::general_mat_mul(alpha, &col_j, &row_i, 1.0, target);
        } else {
            // Different blocks: one rank-1 update to h_mg.
            // h_mg = marginal x logslope; the transpose is assembled automatically.
            let (marginal_row, logslope_row) = if block_i == 0 {
                (psi_row_i, psi_row_j)
            } else {
                (psi_row_j, psi_row_i)
            };
            let m_col = marginal_row.view().insert_axis(Axis(1));
            let g_row = logslope_row.view().insert_axis(Axis(0));
            ndarray::linalg::general_mat_mul(alpha, &m_col, &g_row, 1.0, &mut self.h_mg);
        }
    }

    /// Merge another accumulator into this one (for parallel reduce).
    fn add(&mut self, other: &BernoulliBlockHessianAccumulator) {
        self.h_mm += &other.h_mm;
        self.h_gg += &other.h_gg;
        self.h_mg += &other.h_mg;
        if let (Some(ref mut dc), Some(ref odc)) = (
            self.dense_correction.as_mut(),
            other.dense_correction.as_ref(),
        ) {
            dc.scaled_add(1.0, odc);
        }
    }

    fn to_dense(&self, slices: &BlockSlices) -> Array2<f64> {
        let mut out = Array2::zeros((slices.total, slices.total));
        out.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .assign(&self.h_gg);
        out.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
            .assign(&self.h_mg.t());
        if let Some(ref dc) = self.dense_correction {
            out += dc;
        }
        out
    }

    fn into_operator(self, slices: &BlockSlices) -> BernoulliBlockHessianOperator {
        BernoulliBlockHessianOperator {
            h_mm: self.h_mm,
            h_gg: self.h_gg,
            h_mg: self.h_mg,
            dense_correction: self.dense_correction,
            marginal: slices.marginal.clone(),
            logslope: slices.logslope.clone(),
            total: slices.total,
        }
    }
}

/// Block-structured HyperOperator for Bernoulli marginal-slope psi Hessians.
/// Stores 3 block matrices (h_mm, h_gg, h_mg) plus an optional dense correction
/// for h/w cross-blocks.  Matvec is O(p_m^2 + p_g^2 + p_m*p_g) for the block part,
/// plus O(p_total^2) only if h/w blocks exist (which is rare and tiny).
struct BernoulliBlockHessianOperator {
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_mg: Array2<f64>,
    dense_correction: Option<Array2<f64>>,
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    total: usize,
}

impl HyperOperator for BernoulliBlockHessianOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let v_m = v.slice(s![self.marginal.clone()]);
        let v_g = v.slice(s![self.logslope.clone()]);
        let mut out = Array1::zeros(self.total);
        {
            let mut o_m = out.slice_mut(s![self.marginal.clone()]);
            o_m += &self.h_mm.dot(&v_m);
            o_m += &self.h_mg.dot(&v_g);
        }
        {
            let mut o_g = out.slice_mut(s![self.logslope.clone()]);
            o_g += &self.h_mg.t().dot(&v_m);
            o_g += &self.h_gg.dot(&v_g);
        }
        if let Some(ref dc) = self.dense_correction {
            out += &dc.dot(v);
        }
        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_m = v.slice(s![self.marginal.clone()]);
        let v_g = v.slice(s![self.logslope.clone()]);
        let u_m = u.slice(s![self.marginal.clone()]);
        let u_g = u.slice(s![self.logslope.clone()]);
        // Diagonal blocks
        let mut total = v_m.dot(&self.h_mm.dot(&u_m));
        total += v_g.dot(&self.h_gg.dot(&u_g));
        // Off-diagonal blocks (symmetric)
        total += v_m.dot(&self.h_mg.dot(&u_g));
        total += v_g.dot(&self.h_mg.t().dot(&u_m));
        // Dense correction
        if let Some(ref dc) = self.dense_correction {
            total += v.dot(&dc.dot(u));
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.total, self.total));
        out.slice_mut(s![self.marginal.clone(), self.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![self.logslope.clone(), self.logslope.clone()])
            .assign(&self.h_gg);
        out.slice_mut(s![self.marginal.clone(), self.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![self.logslope.clone(), self.marginal.clone()])
            .assign(&self.h_mg.t());
        if let Some(ref dc) = self.dense_correction {
            out += dc;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

/// How many link-deviation derivative orders the caller actually needs.
/// Higher tiers are strict supersets of lower ones.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LinkOrder {
    /// Basis + first derivative only (intercept solve, log-likelihood value).
    ValueD1,
    /// Through second derivative (inner loop: gradient + Hessian, k ≤ 2).
    Hessian,
    /// All five (third + fourth for exact outer higher-order paths, k ≤ 4).
    Full,
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
/// processed sequentially; per-row gradient / Hessian vectors are computed
/// on the fly and discarded after accumulation.  The per-row *context*
/// (intercept, M_a, link stacks) is pre-solved once in the eval cache.
const ROW_CHUNK_SIZE: usize = 1024;

/// Shared precomputed state plus pre-solved per-row contexts.  All row
/// intercepts (and link stacks for flex models) are solved once during cache
/// construction so that workspace calls (matvec, diagonal, psi, directional
/// derivatives) never redundantly re-solve the Newton intercept equation.
/// For rigid models the per-row context is just three cheap scalars with no
/// link stacks.
#[derive(Clone)]
struct BernoulliMarginalSlopeExactEvalCache {
    slices: BlockSlices,
    primary: PrimarySlices,
    h_nodes: Array1<f64>,
    h_node_design: Option<Array2<f64>>,
    score_warp_obs: Option<(DesignMatrix, Array1<f64>)>,
    /// Pre-solved row contexts (intercept, M_a, link stacks).
    row_contexts: Vec<BernoulliMarginalSlopeRowExactContext>,
}

/// Maximum row-loop work proxy before downgrading to first-order:
///   n × K_pairs × primary²
const EXACT_OUTER_MAX_ROW_WORK: u64 = 2_000_000;

/// Row-loop gate for Bernoulli marginal-slope.  The memory gate is handled
/// by [`cost_gated_outer_order`] at the call site.
fn bernoulli_row_work_order(
    n_rows: usize,
    score_warp_dim: usize,
    link_dev_dim: usize,
    k_smoothing: usize,
) -> ExactOuterDerivativeOrder {
    let n = n_rows as u64;
    let k = k_smoothing as u64;
    let k_pairs = k.saturating_mul(k.saturating_add(1)) / 2;
    let primary_total = 2u64
        .saturating_add(score_warp_dim as u64)
        .saturating_add(link_dev_dim as u64);
    let row_work = n
        .saturating_mul(k_pairs)
        .saturating_mul(primary_total.saturating_mul(primary_total));
    if row_work > EXACT_OUTER_MAX_ROW_WORK {
        ExactOuterDerivativeOrder::First
    } else {
        ExactOuterDerivativeOrder::Second
    }
}

// ── RowKernel<2> implementation (rigid path only) ────────────────────

struct BernoulliRigidRowKernel {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
}

impl BernoulliRigidRowKernel {
    fn new(family: BernoulliMarginalSlopeFamily, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = block_slices(
            &block_states,
            family.score_warp.is_some(),
            family.link_dev.is_some(),
        );
        Self {
            family,
            block_states,
            slices,
        }
    }
}

impl RowKernel<2> for BernoulliRigidRowKernel {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
        let q = self.block_states[0].eta[row];
        let g = self.block_states[1].eta[row];
        let k = RigidProbitKernel::new(
            q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
        );
        let nll = -self.family.weights[row] * k.logcdf;
        let grad = [-k.u1 * k.eta_q, -k.u1 * k.eta_g];
        let h = k.primary_hessian(q);
        Ok((nll, grad, h))
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 2] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        [
            self.family
                .marginal_design
                .dot_row_view(row, d_beta.slice(s![self.slices.marginal.clone()])),
            self.family
                .logslope_design
                .dot_row_view(row, d_beta.slice(s![self.slices.logslope.clone()])),
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 2], out: &mut [f64]) {
        {
            let mut m = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0], &mut m)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut g = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[1], &mut g)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 2]; 2], target: &mut Array2<f64>) {
        self.family
            .marginal_design
            .syr_row_into_view(
                row,
                h[0][0],
                target.slice_mut(s![
                    self.slices.marginal.clone(),
                    self.slices.marginal.clone()
                ]),
            )
            .expect("marginal syr dim mismatch");
        if h[0][1] != 0.0 {
            self.family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &self.family.logslope_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.marginal.clone(),
                        self.slices.logslope.clone()
                    ]),
                )
                .expect("marginal-logslope outer dim mismatch");
            self.family
                .logslope_design
                .row_outer_into_view(
                    row,
                    &self.family.marginal_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.logslope.clone(),
                        self.slices.marginal.clone()
                    ]),
                )
                .expect("logslope-marginal outer dim mismatch");
        }
        self.family
            .logslope_design
            .syr_row_into_view(
                row,
                h[1][1],
                target.slice_mut(s![
                    self.slices.logslope.clone(),
                    self.slices.logslope.clone()
                ]),
            )
            .expect("logslope syr dim mismatch");
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 2]; 2], diag: &mut [f64]) {
        {
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, h[0][0], &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[1][1], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 2]) -> Result<[[f64; 2]; 2], String> {
        let q = self.block_states[0].eta[row];
        let g = self.block_states[1].eta[row];
        let k = RigidProbitKernel::new(
            q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
        );
        Ok(k.third_contracted(q, dir[0], dir[1]))
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 2],
        dir_v: &[f64; 2],
    ) -> Result<[[f64; 2]; 2], String> {
        let q = self.block_states[0].eta[row];
        let g = self.block_states[1].eta[row];
        let k = RigidProbitKernel::new(
            q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
        );
        Ok(k.fourth_contracted(q, dir_u[0], dir_u[1], dir_v[0], dir_v[1]))
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

    /// Fast path: value + first derivative only, skips second derivative design.
    fn link_terms_value_d1(
        &self,
        eta0: &Array1<f64>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        if let (Some(runtime), Some(beta)) = (&self.link_dev, beta_w) {
            let basis = runtime.design(eta0)?;
            let d1 = runtime.first_derivative_design(eta0)?;
            Ok((eta0 + &basis.dot(beta), d1.dot(beta) + 1.0))
        } else {
            Ok((eta0.clone(), Array1::ones(eta0.len())))
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
    ) -> Result<Option<(DesignMatrix, Array1<f64>)>, String> {
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

    /// Build score-warp constraints once from state-independent inputs.
    /// Called at family construction time and cached on the struct.
    fn build_score_warp_constraints(
        runtime: &DeviationRuntime,
        z: &Array1<f64>,
        quadrature_nodes: &Array1<f64>,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        let dense = runtime.dense_constraint_grid(4)?;
        let points = runtime.supported_unique_values_from_iter(
            z.iter()
                .copied()
                .chain(quadrature_nodes.iter().copied())
                .chain(dense.iter().copied()),
        )?;
        if points.is_empty() {
            return Ok(None);
        }
        Self::deviation_constraints_from_points(runtime, points)
    }

    /// Build link-deviation constraints once from the static knot-span grid.
    /// Monotonicity is a function-space property: the dense grid covers every
    /// knot span with enough points to pin a bounded-degree polynomial,
    /// so no O(nQ) state-dependent evaluation-point scan is needed.
    fn build_link_dev_constraints(
        runtime: &DeviationRuntime,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        let points = runtime.dense_constraint_grid(4)?;
        Self::deviation_constraints_from_points(runtime, points)
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
            let (t, t1) = self.link_terms_value_d1(&v, beta_w)?;
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

        // Initial guess: closed-form for rigid probit: a₀ = q·√(1+b²).
        // When link deviation is active, upgrade to affine-link warm start:
        //   L(u) ≈ ℓ₀ + ℓ₁·u  ⟹  a = (q·√(1+ℓ₁²b²) − ℓ₀) / ℓ₁
        let a_rigid = marginal_eta * (1.0 + slope * slope).sqrt();
        let mut a = if beta_w.is_some() {
            let v = Array1::from_vec(vec![a_rigid]);
            let (l_val, l_d1) = self.link_terms_value_d1(&v, beta_w)?;
            let ell1 = l_d1[0];
            if ell1 > 1e-8 {
                let ell0 = l_val[0] - ell1 * a_rigid;
                (marginal_eta * (1.0 + ell1 * ell1 * slope * slope).sqrt() - ell0) / ell1
            } else {
                a_rigid
            }
        } else {
            a_rigid
        };

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
        order: LinkOrder,
    ) -> Result<LinkDerivativeStack, String> {
        let basis = runtime.design(values)?;
        let nrows = values.len();
        let ncols = basis.ncols();
        let d1 = runtime.first_derivative_design(values)?;
        let d2 = if order >= LinkOrder::Hessian {
            runtime.second_derivative_design(values)?
        } else {
            Array2::zeros((nrows, ncols))
        };
        let d3 = if order >= LinkOrder::Full {
            runtime.third_derivative_design(values)?
        } else {
            Array2::zeros((nrows, ncols))
        };
        let d4 = if order >= LinkOrder::Full {
            runtime.fourth_derivative_design(values)?
        } else {
            Array2::zeros((nrows, ncols))
        };
        Ok(LinkDerivativeStack {
            basis,
            d1,
            d2,
            d3,
            d4,
        })
    }

    fn build_row_exact_context(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        h_nodes: &Array1<f64>,
        score_warp_obs: Option<&(DesignMatrix, Array1<f64>)>,
        link_order: LinkOrder,
    ) -> Result<BernoulliMarginalSlopeRowExactContext, String> {
        let marginal_eta = block_states[0].eta[row];
        // The log-slope block now parameterizes the signed slope directly.
        let slope = block_states[1].eta[row];
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
            Some(self.link_basis_stack(runtime, &u_base, link_order)?)
        } else {
            None
        };
        let obs_link = if let Some(runtime) = &self.link_dev {
            let eta_base = Array1::from_vec(vec![intercept + slope * h_obs_base]);
            Some(self.link_basis_stack(runtime, &eta_base, link_order)?)
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

    /// Look up the pre-solved row context from the cache.  All row contexts
    /// are built at `LinkOrder::Full` during cache construction, so this is
    /// valid for every call path (ll/grad/Hessian, third/fourth derivatives,
    /// psi terms, directional derivatives).
    #[inline]
    fn row_ctx(
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> &BernoulliMarginalSlopeRowExactContext {
        &cache.row_contexts[row]
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
        let n = self.y.len();
        let link_order = if self.link_dev.is_some() {
            LinkOrder::Full
        } else {
            LinkOrder::ValueD1
        };
        let row_contexts: Result<Vec<_>, String> = (0..n)
            .into_par_iter()
            .map(|row| {
                self.build_row_exact_context(
                    row,
                    block_states,
                    &h_nodes,
                    score_warp_obs.as_ref(),
                    link_order,
                )
            })
            .collect();
        let row_contexts = row_contexts?;
        Ok(BernoulliMarginalSlopeExactEvalCache {
            slices,
            primary,
            h_nodes,
            h_node_design,
            score_warp_obs,
            row_contexts,
        })
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
        out[0] = self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[1] = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
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
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(slices.total);
        {
            let mut marginal = out.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_vec[0], &mut marginal)?;
        }
        {
            let mut logslope = out.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design
                .axpy_row_into(row, primary_vec[1], &mut logslope)?;
        }
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
        Ok(out)
    }

    fn block_psi_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<BlockPsiRow>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (local_vec, range) = match block_idx {
            0 => (
                self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?,
                slices.marginal.clone(),
            ),
            1 => (
                self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?,
                slices.logslope.clone(),
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi embedding only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };
        Ok(Some(BlockPsiRow {
            block_idx,
            range,
            local_vec,
        }))
    }

    fn block_psi_second_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<BlockPsiRow>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(None);
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let (local_vec, range) = match block_i {
            0 => (
                self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?,
                slices.marginal.clone(),
            ),
            1 => (
                self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?,
                slices.logslope.clone(),
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second embedding only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        };
        Ok(Some(BlockPsiRow {
            block_idx: block_i,
            range,
            local_vec,
        }))
    }

    /// Returns (neg_log_lik, gradient, Hessian) in primary coordinates.
    /// Fully analytic for both flex and non-flex paths — no AD jets.
    fn compute_row_primary_gradient_hessian(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        h_nodes: &Array1<f64>,
        h_node_design: Option<&Array2<f64>>,
        score_warp_obs: Option<&(DesignMatrix, Array1<f64>)>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Flex path: full IFT analytic kernel.
        if self.flex_active() {
            return self.compute_row_analytic_flex(
                row,
                block_states,
                primary,
                row_ctx,
                h_nodes,
                h_node_design,
                score_warp_obs,
                true,
            );
        }
        // Rigid path: closed-form eta = q*c(g) + g*z.
        // primary.total == 2 (q at 0, g at 1), no h/w blocks.
        let q = block_states[0].eta[row];
        let g = block_states[1].eta[row];
        let yi = self.y[row];
        let wi = self.weights[row];
        let zi = self.z[row];
        let s = 2.0 * yi - 1.0;

        let g2 = g * g;
        let c = (1.0 + g2).sqrt();
        let c1 = g / c;
        let c2 = 1.0 / (c * c * c);
        let eta = q * c + g * zi;
        let m = s * eta;

        let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(m);
        let neglog = -wi * logcdf;

        let u1 = -wi * s * lambda;
        let u2 = wi * lambda * (m + lambda);

        let deta_dq = c;
        let deta_dg = q * c1 + zi;

        let mut grad = Array1::<f64>::zeros(2);
        grad[0] = u1 * deta_dq;
        grad[1] = u1 * deta_dg;

        let mut hess = Array2::<f64>::zeros((2, 2));
        hess[[0, 0]] = u2 * deta_dq * deta_dq;
        hess[[0, 1]] = u2 * deta_dq * deta_dg;
        hess[[1, 0]] = hess[[0, 1]];
        hess[[1, 1]] = u2 * deta_dg * deta_dg + u1 * q * c2;

        Ok((neglog, grad, hess))
    }

    /// Analytic IFT-based value / gradient / Hessian for the flexible
    /// (score-warp and/or link-deviation) path.
    ///
    /// Replaces the O(r² Q) generic `MultiDirJet` tower with a single
    /// O(Q r + r²) pass.  One sweep over quadrature nodes accumulates
    /// intercept-equation partial derivatives, then the implicit-function
    /// theorem gives a_u, a_{uv} in closed form.
    fn compute_row_analytic_flex(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        h_nodes: &Array1<f64>,
        h_node_design: Option<&Array2<f64>>,
        score_warp_obs: Option<&(DesignMatrix, Array1<f64>)>,
        need_hessian: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let r = primary.total;
        let q = block_states[0].eta[row];
        let b = block_states[1].eta[row];
        let a = row_ctx.intercept;
        let m_a = row_ctx.m_a;
        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let beta_w: Option<&Array1<f64>> = if self.link_dev.is_some() {
            block_states.last().map(|st| &st.beta)
        } else {
            None
        };
        let n_h = primary.h.as_ref().map_or(0, |rng| rng.len());
        let n_w = primary.w.as_ref().map_or(0, |rng| rng.len());
        let obs_row: Option<Array1<f64>> =
            if let (Some(_), Some((obs_design, _))) = (primary.h.as_ref(), score_warp_obs) {
                Some(obs_design.row_chunk(row..row + 1).row(0).to_owned())
            } else {
                None
            };
        let phi_q = normal_pdf(q);
        let inv_ma = 1.0 / m_a;

        // ── Phase 1: intercept-equation derivatives over quadrature nodes ──
        let mut m_aa = 0.0f64;
        let mut m_u = Array1::<f64>::zeros(r);
        let mut m_au = Array1::<f64>::zeros(r);
        let mut m_uv = Array2::<f64>::zeros((r, r));
        let nq = h_nodes.len();
        for j in 0..nq {
            let h_j = h_nodes[j];
            let omega_j = self.quadrature_weights[j];
            let (s_j, c_j, d_j) = if let Some(ls) = row_ctx.node_link.as_ref() {
                let bw = beta_w.unwrap();
                let u_j = a + b * h_j;
                (
                    u_j + ls.basis.row(j).dot(bw),
                    1.0 + ls.d1.row(j).dot(bw),
                    ls.d2.row(j).dot(bw),
                )
            } else {
                (a + b * h_j, 1.0, 0.0)
            };
            let r_j = omega_j * normal_pdf(s_j);
            let t_j = -s_j * r_j;
            let st_b = c_j * h_j;
            m_u[1] += r_j * st_b;
            if need_hessian {
                m_aa += t_j * c_j * c_j + r_j * d_j;
                m_au[1] += t_j * c_j * st_b + r_j * d_j * h_j;
                m_uv[[1, 1]] += t_j * st_b * st_b + r_j * d_j * h_j * h_j;
            }
            if let (Some(h_range), Some(design)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    let dk = design[[j, k]];
                    let st_hk = c_j * b * dk;
                    let idx_k = h_range.start + k;
                    m_u[idx_k] += r_j * st_hk;
                    if need_hessian {
                        m_au[idx_k] += t_j * c_j * st_hk + r_j * d_j * b * dk;
                        m_uv[[1, idx_k]] += t_j * st_b * st_hk + r_j * dk * (d_j * b * h_j + c_j);
                        let d_b2 = d_j * b * b;
                        for ell in k..n_h {
                            let dl = design[[j, ell]];
                            m_uv[[idx_k, h_range.start + ell]] +=
                                t_j * st_hk * (c_j * b * dl) + r_j * dk * dl * d_b2;
                        }
                    }
                }
            }
            if let (Some(w_range), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    let st_wl = ls.basis[[j, ell]];
                    let idx_l = w_range.start + ell;
                    m_u[idx_l] += r_j * st_wl;
                    if need_hessian {
                        let ct_wl = ls.d1[[j, ell]];
                        m_au[idx_l] += t_j * c_j * st_wl + r_j * ct_wl;
                        m_uv[[1, idx_l]] += t_j * st_b * st_wl + r_j * h_j * ct_wl;
                        if let (Some(h_range), Some(design)) = (primary.h.as_ref(), h_node_design) {
                            for k in 0..n_h {
                                m_uv[[h_range.start + k, idx_l]] +=
                                    t_j * (c_j * b * design[[j, k]]) * st_wl
                                        + r_j * b * design[[j, k]] * ct_wl;
                            }
                        }
                        for m_idx in ell..n_w {
                            m_uv[[idx_l, w_range.start + m_idx]] +=
                                t_j * st_wl * ls.basis[[j, m_idx]];
                        }
                    }
                }
            }
        }
        if need_hessian {
            for u in 0..r {
                for v in (u + 1)..r {
                    m_uv[[v, u]] = m_uv[[u, v]];
                }
            }
        }
        m_u[0] -= phi_q;
        if need_hessian {
            m_uv[[0, 0]] += q * phi_q;
        }

        // ── Phase 2: IFT ──
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -m_u[u] * inv_ma;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        if need_hessian {
            for u in 0..r {
                for v in u..r {
                    let val = -(m_uv[[u, v]]
                        + m_au[u] * a_u[v]
                        + m_au[v] * a_u[u]
                        + m_aa * a_u[u] * a_u[v])
                        * inv_ma;
                    a_uv[[u, v]] = val;
                    a_uv[[v, u]] = val;
                }
            }
        }

        // ── Phase 3: observed predictor ──
        let h_obs = row_ctx.h_obs_base;
        let (c_o, d_o) = if let Some(ls) = row_ctx.obs_link.as_ref() {
            let bw = beta_w.unwrap();
            (1.0 + ls.d1.row(0).dot(bw), ls.d2.row(0).dot(bw))
        } else {
            (1.0, 0.0)
        };
        let mut uo_u = a_u.clone();
        uo_u[1] += h_obs;
        if let (Some(h_range), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
            for k in 0..n_h {
                uo_u[h_range.start + k] += b * obs[k];
            }
        }
        let mut eta_u = uo_u.mapv(|x| c_o * x);
        if let (Some(w_range), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
            for ell in 0..n_w {
                eta_u[w_range.start + ell] += ls.basis[[0, ell]];
            }
        }
        let u_o_val = a + b * h_obs;
        let eta_val = if let Some(ls) = row_ctx.obs_link.as_ref() {
            u_o_val + ls.basis.row(0).dot(&beta_w.unwrap().view())
        } else {
            u_o_val
        };
        let signed_margin = s_y * eta_val;
        let (log_cdf, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
        let neglog_val = -w_i * log_cdf;
        let d1_m = -w_i * lambda;
        let d2_m = w_i * lambda * (signed_margin + lambda);
        let grad = eta_u.mapv(|eu| d1_m * s_y * eu);
        let mut hess = Array2::<f64>::zeros((r, r));
        if need_hessian {
            for u in 0..r {
                for v in u..r {
                    let mut uo_uv = a_uv[[u, v]];
                    if let (Some(h_range), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
                        if u == 1 && h_range.contains(&v) {
                            uo_uv += obs[v - h_range.start];
                        } else if v == 1 && h_range.contains(&u) {
                            uo_uv += obs[u - h_range.start];
                        }
                    }
                    let mut eta_uv = d_o * uo_u[u] * uo_u[v] + c_o * uo_uv;
                    if let (Some(w_range), Some(ls)) =
                        (primary.w.as_ref(), row_ctx.obs_link.as_ref())
                    {
                        if w_range.contains(&u) {
                            eta_uv += ls.d1[[0, u - w_range.start]] * uo_u[v];
                        }
                        if w_range.contains(&v) {
                            eta_uv += ls.d1[[0, v - w_range.start]] * uo_u[u];
                        }
                    }
                    let val = d2_m * eta_u[u] * eta_u[v] + d1_m * s_y * eta_uv;
                    hess[[u, v]] = val;
                    hess[[v, u]] = val;
                }
            }
        }
        Ok((neglog_val, grad, hess))
    }

    /// Third-derivative tensor contracted with direction `dir`:
    ///   out[k,l] = sum_m f_{klm} dir[m]
    /// Fully analytic for both rigid and flex paths -- no AD jets.
    fn row_primary_third_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let primary = &cache.primary;
        // Rigid fast path
        if !self.flex_active() {
            let q = block_states[0].eta[row];
            let g = block_states[1].eta[row];
            let kern = RigidProbitKernel::new(q, g, self.z[row], self.y[row], self.weights[row]);
            let t = kern.third_contracted(q, dir[0], dir[1]);
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = t[0][0];
            out[[0, 1]] = t[0][1];
            out[[1, 0]] = t[1][0];
            out[[1, 1]] = t[1][1];
            return Ok(out);
        }
        // Flex: analytic IFT 3rd order
        let r = primary.total;
        let q = block_states[0].eta[row];
        let b = block_states[1].eta[row];
        let a = row_ctx.intercept;
        let m_a = row_ctx.m_a;
        let inv_ma = 1.0 / m_a;
        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let beta_w: Option<&Array1<f64>> = if self.link_dev.is_some() {
            block_states.last().map(|st| &st.beta)
        } else {
            None
        };
        let n_h = primary.h.as_ref().map_or(0, |rng| rng.len());
        let n_w = primary.w.as_ref().map_or(0, |rng| rng.len());
        let obs_row: Option<Array1<f64>> = if let (Some(_), Some((obs_design, _))) =
            (primary.h.as_ref(), cache.score_warp_obs.as_ref())
        {
            Some(obs_design.row_chunk(row..row + 1).row(0).to_owned())
        } else {
            None
        };
        let phi_q = normal_pdf(q);
        let h_nodes = &cache.h_nodes;
        let h_node_design = cache.h_node_design.as_ref();
        let nq = h_nodes.len();

        // Phase 1: M derivatives through 3rd order, contracted with dir
        let mut m_aa = 0.0f64;
        let mut m_u = Array1::<f64>::zeros(r);
        let mut m_au = Array1::<f64>::zeros(r);
        let mut m_uv = Array2::<f64>::zeros((r, r));
        let mut m_aaa = 0.0f64;
        let mut m_aau_k = Array1::<f64>::zeros(r);
        let mut m_auv_v = Array1::<f64>::zeros(r);
        let mut m_uvw_v = Array2::<f64>::zeros((r, r));
        let mut m_a_kl = Array2::<f64>::zeros((r, r)); // M_{a,kl} (uncontracted)

        for j in 0..nq {
            let h_j = h_nodes[j];
            let omega_j = self.quadrature_weights[j];
            let (s_j, c_j, d_j, e_j) = if let Some(ls) = row_ctx.node_link.as_ref() {
                let bw = beta_w.unwrap();
                let u_j = a + b * h_j;
                (
                    u_j + ls.basis.row(j).dot(bw),
                    1.0 + ls.d1.row(j).dot(bw),
                    ls.d2.row(j).dot(bw),
                    ls.d3.row(j).dot(bw),
                )
            } else {
                (a + b * h_j, 1.0, 0.0, 0.0)
            };
            let r_j = omega_j * normal_pdf(s_j);
            let t_j = -s_j * r_j;
            let u3_j = (s_j * s_j - 1.0) * r_j;

            // Per-node xi_jk, dc_jk, du_jk
            let mut xi = Array1::<f64>::zeros(r);
            let mut dc_arr = Array1::<f64>::zeros(r);
            let mut du_arr = Array1::<f64>::zeros(r);
            xi[1] = c_j * h_j;
            dc_arr[1] = d_j * h_j;
            du_arr[1] = h_j;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    let idx = hr.start + k;
                    xi[idx] = c_j * b * des[[j, k]];
                    dc_arr[idx] = d_j * b * des[[j, k]];
                    du_arr[idx] = b * des[[j, k]];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    xi[wr.start + ell] = ls.basis[[j, ell]];
                    dc_arr[wr.start + ell] = ls.d1[[j, ell]];
                    // du_arr[w] = 0 (u_j doesn't depend on w)
                }
            }

            // Contracted with dir
            let xi_v = xi.dot(dir);
            let du_v = du_arr.dot(dir);
            let mut dc_v = d_j * du_v;
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    dc_v += ls.d1[[j, ell]] * dir[wr.start + ell];
                }
            }
            let mut dh_v = 0.0;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    dh_v += des[[j, k]] * dir[hr.start + k];
                }
            }

            // dxi_jk/dv
            let mut dxi_v = Array1::<f64>::zeros(r);
            dxi_v[1] = dc_v * h_j + c_j * dh_v;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    dxi_v[hr.start + k] = dc_v * b * des[[j, k]] + c_j * des[[j, k]] * dir[1];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    dxi_v[wr.start + ell] = ls.d1[[j, ell]] * du_v;
                }
            }

            // ddc_jk/dv (d(dc_jk)/dv)
            let mut ddc_v_arr = Array1::<f64>::zeros(r);
            ddc_v_arr[1] = e_j * h_j * du_v + d_j * dh_v;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    ddc_v_arr[hr.start + k] =
                        e_j * b * des[[j, k]] * du_v + d_j * des[[j, k]] * dir[1];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    ddc_v_arr[wr.start + ell] = ls.d2[[j, ell]] * du_v;
                }
            }

            // dd_jk (partial of d_j w.r.t. theta_k)
            let mut dd_arr = Array1::<f64>::zeros(r);
            dd_arr[1] = e_j * h_j;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    dd_arr[hr.start + k] = e_j * b * des[[j, k]];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    dd_arr[wr.start + ell] = ls.d2[[j, ell]];
                }
            }

            // 1st order
            for k in 0..r {
                m_u[k] += r_j * xi[k];
            }
            // 2nd order
            m_aa += t_j * c_j * c_j + r_j * d_j;
            for k in 0..r {
                m_au[k] += t_j * c_j * xi[k] + r_j * dc_arr[k];
                for l in k..r {
                    // rank-1 + correction
                    let mut val = t_j * xi[k] * xi[l];
                    // corrections from dxi/dtheta
                    if k == 1 && l == 1 {
                        val += r_j * d_j * h_j * h_j;
                    } else if k == 1 {
                        if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                            if l >= hr.start && l < hr.end {
                                val += r_j
                                    * (d_j * b * des[[j, l - hr.start]] * h_j
                                        + c_j * des[[j, l - hr.start]]);
                            }
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if l >= wr.start && l < wr.end {
                                val += r_j * ls.d1[[j, l - wr.start]] * h_j;
                            }
                        }
                    } else if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                        if k >= hr.start && k < hr.end && l >= hr.start && l < hr.end {
                            val +=
                                r_j * d_j * b * b * des[[j, k - hr.start]] * des[[j, l - hr.start]];
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if k >= hr.start && k < hr.end && l >= wr.start && l < wr.end {
                                val += r_j * ls.d1[[j, l - wr.start]] * b * des[[j, k - hr.start]];
                            }
                        }
                    }
                    m_uv[[k, l]] += val;
                    if l != k {
                        m_uv[[l, k]] += val;
                    }
                }
            }

            // 3rd order
            m_aaa += u3_j * c_j * c_j * c_j + 3.0 * t_j * c_j * d_j + r_j * e_j;
            for k in 0..r {
                m_aau_k[k] += u3_j * c_j * c_j * xi[k]
                    + t_j * (2.0 * c_j * dc_arr[k] + d_j * xi[k])
                    + r_j * dd_arr[k];
                m_auv_v[k] += u3_j * c_j * xi[k] * xi_v
                    + t_j * (dc_v * xi[k] + c_j * dxi_v[k] + xi_v * dc_arr[k])
                    + r_j * ddc_v_arr[k];
            }
            for k in 0..r {
                if xi[k] == 0.0 && dxi_v[k] == 0.0 {
                    continue;
                }
                for l in k..r {
                    if xi[l] == 0.0 && dxi_v[l] == 0.0 {
                        continue;
                    }
                    let val =
                        u3_j * xi[k] * xi[l] * xi_v + t_j * (dxi_v[k] * xi[l] + xi[k] * dxi_v[l]);
                    m_uvw_v[[k, l]] += val;
                    if l != k {
                        m_uvw_v[[l, k]] += val;
                    }
                }
            }
            // M_{a,kl}: rank-1 + corrections (same structure as m_uv but with extra c_j factor)
            for k in 0..r {
                if xi[k] == 0.0 && dc_arr[k] == 0.0 {
                    continue;
                }
                for l in k..r {
                    if xi[l] == 0.0 && dc_arr[l] == 0.0 {
                        continue;
                    }
                    let mut val =
                        u3_j * c_j * xi[k] * xi[l] + t_j * (dc_arr[k] * xi[l] + xi[k] * dc_arr[l]);
                    // dxi/dtheta + d^2c/dtheta^2 corrections (same pattern as m_uv but with Phi'' c factor)
                    if k == 1 && l == 1 {
                        val += t_j * c_j * d_j * h_j * h_j + r_j * e_j * h_j * h_j;
                    } else if k == 1 {
                        if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                            if l >= hr.start && l < hr.end {
                                val += t_j
                                    * c_j
                                    * (d_j * b * des[[j, l - hr.start]] * h_j
                                        + c_j * des[[j, l - hr.start]])
                                    + r_j
                                        * (e_j * h_j * b * des[[j, l - hr.start]]
                                            + d_j * des[[j, l - hr.start]]);
                            }
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if l >= wr.start && l < wr.end {
                                val += t_j * c_j * ls.d1[[j, l - wr.start]] * h_j
                                    + r_j * ls.d2[[j, l - wr.start]] * h_j;
                            }
                        }
                    } else if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                        if k >= hr.start && k < hr.end && l >= hr.start && l < hr.end {
                            val += t_j
                                * c_j
                                * d_j
                                * b
                                * b
                                * des[[j, k - hr.start]]
                                * des[[j, l - hr.start]]
                                + r_j
                                    * e_j
                                    * b
                                    * b
                                    * des[[j, k - hr.start]]
                                    * des[[j, l - hr.start]];
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if k >= hr.start && k < hr.end && l >= wr.start && l < wr.end {
                                val += t_j
                                    * c_j
                                    * ls.d1[[j, l - wr.start]]
                                    * b
                                    * des[[j, k - hr.start]]
                                    + r_j * ls.d2[[j, l - wr.start]] * b * des[[j, k - hr.start]];
                            }
                        }
                    }
                    m_a_kl[[k, l]] += val;
                    if l != k {
                        m_a_kl[[l, k]] += val;
                    }
                }
            }
        } // end node loop

        // Finalize
        m_u[0] -= phi_q;
        m_uv[[0, 0]] += q * phi_q;
        m_uvw_v[[0, 0]] += dir[0] * (1.0 - q * q) * phi_q;

        // IFT 1st + 2nd order
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -m_u[u] * inv_ma;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(m_uv[[u, v]] + m_au[u] * a_u[v] + m_au[v] * a_u[u] + m_aa * a_u[u] * a_u[v])
                        * inv_ma;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }

        // IFT 3rd order contracted with dir
        let a_v = a_u.dot(dir);
        let a_kv = a_uv.dot(dir);
        let dm_a_v = m_au.dot(dir) + m_aa * a_v;
        let dm_aa_v = m_aau_k.dot(dir) + m_aaa * a_v;
        let dm_au_v: Array1<f64> = &m_auv_v + &(&m_aau_k * a_v);
        let dm_kl_v: Array2<f64> = &m_uvw_v + &(&m_a_kl * a_v);
        let mut a_klv = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let dpsi = dm_kl_v[[k, l]]
                    + dm_au_v[k] * a_u[l]
                    + m_au[k] * a_kv[l]
                    + dm_au_v[l] * a_u[k]
                    + m_au[l] * a_kv[k]
                    + dm_aa_v * a_u[k] * a_u[l]
                    + m_aa * (a_kv[k] * a_u[l] + a_u[k] * a_kv[l]);
                let val = -dpsi * inv_ma - a_uv[[k, l]] * dm_a_v * inv_ma;
                a_klv[[k, l]] = val;
                a_klv[[l, k]] = val;
            }
        }

        // Phase 3: observation-point derivatives
        let h_obs = row_ctx.h_obs_base;
        let (c_o, d_o, e_o) = if let Some(ls) = row_ctx.obs_link.as_ref() {
            let bw = beta_w.unwrap();
            (
                1.0 + ls.d1.row(0).dot(bw),
                ls.d2.row(0).dot(bw),
                ls.d3.row(0).dot(bw),
            )
        } else {
            (1.0, 0.0, 0.0)
        };
        let mut uo_u = a_u.clone();
        uo_u[1] += h_obs;
        if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
            for k in 0..n_h {
                uo_u[hr.start + k] += b * obs[k];
            }
        }
        let mut eta_u = uo_u.mapv(|x| c_o * x);
        if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
            for ell in 0..n_w {
                eta_u[wr.start + ell] += ls.basis[[0, ell]];
            }
        }
        let eta_val = if let Some(ls) = row_ctx.obs_link.as_ref() {
            (a + b * h_obs) + ls.basis.row(0).dot(&beta_w.unwrap().view())
        } else {
            a + b * h_obs
        };
        let signed_margin = s_y * eta_val;
        let (k1_s, k2_s, k3_s, _) =
            signed_probit_neglog_derivatives_up_to_fourth(signed_margin, w_i);
        let uo_v = uo_u.dot(dir);
        let eta_v = eta_u.dot(dir);
        let mut uo_kv = a_uv.dot(dir);
        if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
            for k in 0..n_h {
                uo_kv[hr.start + k] += obs[k] * dir[1];
            }
            for k in 0..n_h {
                uo_kv[1] += obs[k] * dir[hr.start + k];
            }
        }
        let mut bp_v = 0.0;
        if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
            for ell in 0..n_w {
                bp_v += ls.d1[[0, ell]] * dir[wr.start + ell];
            }
        }
        let mut eta_kv = Array1::<f64>::zeros(r);
        for k in 0..r {
            eta_kv[k] = d_o * uo_u[k] * uo_v + c_o * uo_kv[k] + bp_v * uo_u[k];
        }
        if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
            for ell in 0..n_w {
                eta_kv[wr.start + ell] += ls.d1[[0, ell]] * uo_v;
            }
        }
        let mut eta_kl = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let mut uo_kl = a_uv[[k, l]];
                if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
                    if k == 1 && hr.contains(&l) {
                        uo_kl += obs[l - hr.start];
                    } else if l == 1 && hr.contains(&k) {
                        uo_kl += obs[k - hr.start];
                    }
                }
                let mut val = d_o * uo_u[k] * uo_u[l] + c_o * uo_kl;
                if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
                    if wr.contains(&k) {
                        val += ls.d1[[0, k - wr.start]] * uo_u[l];
                    }
                    if wr.contains(&l) {
                        val += ls.d1[[0, l - wr.start]] * uo_u[k];
                    }
                }
                eta_kl[[k, l]] = val;
                eta_kl[[l, k]] = val;
            }
        }
        // eta_{klv}
        let mut eta_klv = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let mut uo_kl = a_uv[[k, l]];
                if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
                    if k == 1 && hr.contains(&l) {
                        uo_kl += obs[l - hr.start];
                    } else if l == 1 && hr.contains(&k) {
                        uo_kl += obs[k - hr.start];
                    }
                }
                let mut val = e_o * uo_u[k] * uo_u[l] * uo_v
                    + d_o * (uo_kv[k] * uo_u[l] + uo_u[k] * uo_kv[l] + uo_kl * uo_v)
                    + c_o * a_klv[[k, l]];
                if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
                    let mut bpp_v = 0.0;
                    for ell in 0..n_w {
                        bpp_v += ls.d2[[0, ell]] * dir[wr.start + ell];
                    }
                    val += bpp_v * uo_u[k] * uo_u[l];
                    if wr.contains(&k) {
                        val += ls.d2[[0, k - wr.start]] * uo_u[l] * uo_v;
                        val += ls.d1[[0, k - wr.start]] * (d_o * uo_u[l] * uo_v + c_o * uo_kv[l]);
                    }
                    if wr.contains(&l) {
                        val += ls.d2[[0, l - wr.start]] * uo_u[k] * uo_v;
                        val += ls.d1[[0, l - wr.start]] * (d_o * uo_u[k] * uo_v + c_o * uo_kv[k]);
                    }
                    val += bp_v * (d_o * uo_u[k] * uo_u[l] + c_o * uo_kl);
                }
                eta_klv[[k, l]] = val;
                eta_klv[[l, k]] = val;
            }
        }

        // f_{klv} = k3 s_y eta_k eta_l eta_v + k2 (eta_{kl} eta_v + eta_{kv} eta_l + eta_{lv} eta_k) + k1 s_y eta_{klv}
        let mut out = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let val = k3_s * s_y * eta_u[k] * eta_u[l] * eta_v
                    + k2_s * (eta_kl[[k, l]] * eta_v + eta_kv[k] * eta_u[l] + eta_kv[l] * eta_u[k])
                    + k1_s * s_y * eta_klv[[k, l]];
                out[[k, l]] = val;
                out[[l, k]] = val;
            }
        }
        Ok(out)
    }

    /// Fourth-derivative tensor contracted with two directions dir_u, dir_v:
    ///   out[k,l] = sum_{m,n} f_{klmn} dir_u[m] dir_v[n]
    /// Fully analytic for both rigid and flex paths -- no AD jets.
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
        // Rigid fast path
        if !self.flex_active() {
            let q = block_states[0].eta[row];
            let g = block_states[1].eta[row];
            let kern = RigidProbitKernel::new(q, g, self.z[row], self.y[row], self.weights[row]);
            let f = kern.fourth_contracted(q, dir_u[0], dir_u[1], dir_v[0], dir_v[1]);
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = f[0][0];
            out[[0, 1]] = f[0][1];
            out[[1, 0]] = f[1][0];
            out[[1, 1]] = f[1][1];
            return Ok(out);
        }
        // Flex: analytic IFT through 4th order.
        // Compute a_{klv}, a_{klu} (3rd IFT), then a_{kluv} (4th IFT),
        // push through observation-point chain rule and Faa di Bruno.
        //
        // The third-order helper returns f_{klw}, but the 4th-order path needs
        // the IFT intermediates a_{klv} and a_{klu}, so recompute those
        // intermediates locally in one more node sweep.

        let r = primary.total;
        let q_val = block_states[0].eta[row];
        let b = block_states[1].eta[row];
        let a = row_ctx.intercept;
        let m_a = row_ctx.m_a;
        let inv_ma = 1.0 / m_a;
        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let beta_w: Option<&Array1<f64>> = if self.link_dev.is_some() {
            block_states.last().map(|st| &st.beta)
        } else {
            None
        };
        let n_h = primary.h.as_ref().map_or(0, |rng| rng.len());
        let n_w = primary.w.as_ref().map_or(0, |rng| rng.len());
        let obs_row: Option<Array1<f64>> = if let (Some(_), Some((obs_design, _))) =
            (primary.h.as_ref(), cache.score_warp_obs.as_ref())
        {
            Some(obs_design.row_chunk(row..row + 1).row(0).to_owned())
        } else {
            None
        };
        let phi_q = normal_pdf(q_val);
        let h_nodes = &cache.h_nodes;
        let h_node_design = cache.h_node_design.as_ref();
        let nq = h_nodes.len();

        // Phase 1+2: Recompute 1st/2nd order IFT (needed for a_{klv}, a_{klu})
        let mut m_aa = 0.0f64;
        let mut m_u = Array1::<f64>::zeros(r);
        let mut m_au = Array1::<f64>::zeros(r);
        let mut m_uv = Array2::<f64>::zeros((r, r));
        // 3rd order accumulators for BOTH directions
        let mut m_aaa = 0.0f64;
        let mut m_aau_k = Array1::<f64>::zeros(r);
        let mut m_a_kl = Array2::<f64>::zeros((r, r));
        // Contracted with v
        let mut m_auv_v = Array1::<f64>::zeros(r);
        let mut m_uvw_v = Array2::<f64>::zeros((r, r));
        // Contracted with u
        let mut m_auv_u = Array1::<f64>::zeros(r);
        let mut m_uvw_u = Array2::<f64>::zeros((r, r));
        // 4th order cross-contracted D_uv quantities
        let mut d2m_a_uv = 0.0f64;
        let mut d2m_aa_uv = 0.0f64;
        let mut d2m_au_uv = Array1::<f64>::zeros(r);
        let mut d2m_kl_uv = Array2::<f64>::zeros((r, r));

        for j in 0..nq {
            let h_j = h_nodes[j];
            let omega_j = self.quadrature_weights[j];
            let (s_j, c_j, d_j, e_j) = if let Some(ls) = row_ctx.node_link.as_ref() {
                let bw = beta_w.unwrap();
                let u_j = a + b * h_j;
                (
                    u_j + ls.basis.row(j).dot(bw),
                    1.0 + ls.d1.row(j).dot(bw),
                    ls.d2.row(j).dot(bw),
                    ls.d3.row(j).dot(bw),
                )
            } else {
                (a + b * h_j, 1.0, 0.0, 0.0)
            };

            let r_j = omega_j * normal_pdf(s_j);
            let t_j = -s_j * r_j;
            let u3_j = (s_j * s_j - 1.0) * r_j;

            // Per-k arrays
            let mut xi = Array1::<f64>::zeros(r);
            let mut dc_arr = Array1::<f64>::zeros(r);
            let mut du_arr = Array1::<f64>::zeros(r);
            let mut dd_arr = Array1::<f64>::zeros(r);
            xi[1] = c_j * h_j;
            dc_arr[1] = d_j * h_j;
            du_arr[1] = h_j;
            dd_arr[1] = e_j * h_j;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    let idx = hr.start + k;
                    xi[idx] = c_j * b * des[[j, k]];
                    dc_arr[idx] = d_j * b * des[[j, k]];
                    du_arr[idx] = b * des[[j, k]];
                    dd_arr[idx] = e_j * b * des[[j, k]];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    xi[wr.start + ell] = ls.basis[[j, ell]];
                    dc_arr[wr.start + ell] = ls.d1[[j, ell]];
                    dd_arr[wr.start + ell] = ls.d2[[j, ell]];
                }
            }

            // Directional contracted scalars for v
            let du_v = du_arr.dot(dir_v);
            let mut dh_v = 0.0;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    dh_v += des[[j, k]] * dir_v[hr.start + k];
                }
            }

            // Same for u
            let du_u = du_arr.dot(dir_u);
            let mut dh_u = 0.0;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    dh_u += des[[j, k]] * dir_u[hr.start + k];
                }
            }

            // 1st order M
            for k in 0..r {
                m_u[k] += r_j * xi[k];
            }
            // 2nd order M
            m_aa += t_j * c_j * c_j + r_j * d_j;
            for k in 0..r {
                m_au[k] += t_j * c_j * xi[k] + r_j * dc_arr[k];
            }
            // m_uv: rank-1 + corrections (same as third function)
            for k in 0..r {
                for l in k..r {
                    let mut val = t_j * xi[k] * xi[l];
                    // dxi/dtheta corrections
                    if k == 1 && l == 1 {
                        val += r_j * d_j * h_j * h_j;
                    } else if k == 1 {
                        if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                            if l >= hr.start && l < hr.end {
                                val += r_j
                                    * (d_j * b * des[[j, l - hr.start]] * h_j
                                        + c_j * des[[j, l - hr.start]]);
                            }
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if l >= wr.start && l < wr.end {
                                val += r_j * ls.d1[[j, l - wr.start]] * h_j;
                            }
                        }
                    } else if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                        if k >= hr.start && k < hr.end && l >= hr.start && l < hr.end {
                            val +=
                                r_j * d_j * b * b * des[[j, k - hr.start]] * des[[j, l - hr.start]];
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if k >= hr.start && k < hr.end && l >= wr.start && l < wr.end {
                                val += r_j * ls.d1[[j, l - wr.start]] * b * des[[j, k - hr.start]];
                            }
                        }
                    }
                    m_uv[[k, l]] += val;
                    if l != k {
                        m_uv[[l, k]] += val;
                    }
                }
            }

            // 3rd order M (needed for BOTH v and u directions)
            m_aaa += u3_j * c_j * c_j * c_j + 3.0 * t_j * c_j * d_j + r_j * e_j;
            for k in 0..r {
                m_aau_k[k] += u3_j * c_j * c_j * xi[k]
                    + t_j * (2.0 * c_j * dc_arr[k] + d_j * xi[k])
                    + r_j * dd_arr[k];
            }

            // Build dxi_v, dxi_u, ddc_v, ddc_u per-k
            let mut dc_v = d_j * du_v;
            let mut dc_u = d_j * du_u;
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    dc_v += ls.d1[[j, ell]] * dir_v[wr.start + ell];
                    dc_u += ls.d1[[j, ell]] * dir_u[wr.start + ell];
                }
            }
            let xi_v = xi.dot(dir_v);
            let xi_u = xi.dot(dir_u);

            let mut dxi_v = Array1::<f64>::zeros(r);
            let mut dxi_u = Array1::<f64>::zeros(r);
            let mut ddc_v_arr = Array1::<f64>::zeros(r);
            let mut ddc_u_arr = Array1::<f64>::zeros(r);
            dxi_v[1] = dc_v * h_j + c_j * dh_v;
            dxi_u[1] = dc_u * h_j + c_j * dh_u;
            ddc_v_arr[1] = e_j * h_j * du_v + d_j * dh_v;
            ddc_u_arr[1] = e_j * h_j * du_u + d_j * dh_u;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    dxi_v[hr.start + k] = dc_v * b * des[[j, k]] + c_j * des[[j, k]] * dir_v[1];
                    dxi_u[hr.start + k] = dc_u * b * des[[j, k]] + c_j * des[[j, k]] * dir_u[1];
                    ddc_v_arr[hr.start + k] =
                        e_j * b * des[[j, k]] * du_v + d_j * des[[j, k]] * dir_v[1];
                    ddc_u_arr[hr.start + k] =
                        e_j * b * des[[j, k]] * du_u + d_j * des[[j, k]] * dir_u[1];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    dxi_v[wr.start + ell] = ls.d1[[j, ell]] * du_v;
                    dxi_u[wr.start + ell] = ls.d1[[j, ell]] * du_u;
                    ddc_v_arr[wr.start + ell] = ls.d2[[j, ell]] * du_v;
                    ddc_u_arr[wr.start + ell] = ls.d2[[j, ell]] * du_u;
                }
            }

            // Accumulate 3rd-order contracted with v
            for k in 0..r {
                m_auv_v[k] += u3_j * c_j * xi[k] * xi_v
                    + t_j * (dc_v * xi[k] + c_j * dxi_v[k] + xi_v * dc_arr[k])
                    + r_j * ddc_v_arr[k];
            }
            for k in 0..r {
                if xi[k] == 0.0 && dxi_v[k] == 0.0 {
                    continue;
                }
                for l in k..r {
                    if xi[l] == 0.0 && dxi_v[l] == 0.0 {
                        continue;
                    }
                    let val =
                        u3_j * xi[k] * xi[l] * xi_v + t_j * (dxi_v[k] * xi[l] + xi[k] * dxi_v[l]);
                    m_uvw_v[[k, l]] += val;
                    if l != k {
                        m_uvw_v[[l, k]] += val;
                    }
                }
            }
            // Same for u
            for k in 0..r {
                m_auv_u[k] += u3_j * c_j * xi[k] * xi_u
                    + t_j * (dc_u * xi[k] + c_j * dxi_u[k] + xi_u * dc_arr[k])
                    + r_j * ddc_u_arr[k];
            }
            for k in 0..r {
                if xi[k] == 0.0 && dxi_u[k] == 0.0 {
                    continue;
                }
                for l in k..r {
                    if xi[l] == 0.0 && dxi_u[l] == 0.0 {
                        continue;
                    }
                    let val =
                        u3_j * xi[k] * xi[l] * xi_u + t_j * (dxi_u[k] * xi[l] + xi[k] * dxi_u[l]);
                    m_uvw_u[[k, l]] += val;
                    if l != k {
                        m_uvw_u[[l, k]] += val;
                    }
                }
            }

            // M_{a,kl} (needed uncontracted for both directions)
            for k in 0..r {
                if xi[k] == 0.0 && dc_arr[k] == 0.0 {
                    continue;
                }
                for l in k..r {
                    if xi[l] == 0.0 && dc_arr[l] == 0.0 {
                        continue;
                    }
                    let mut val =
                        u3_j * c_j * xi[k] * xi[l] + t_j * (dc_arr[k] * xi[l] + xi[k] * dc_arr[l]);
                    // dxi/dtheta + d^2c/dtheta^2 corrections
                    if k == 1 && l == 1 {
                        val += t_j * c_j * d_j * h_j * h_j + r_j * e_j * h_j * h_j;
                    } else if k == 1 {
                        if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                            if l >= hr.start && l < hr.end {
                                val += t_j
                                    * c_j
                                    * (d_j * b * des[[j, l - hr.start]] * h_j
                                        + c_j * des[[j, l - hr.start]])
                                    + r_j
                                        * (e_j * h_j * b * des[[j, l - hr.start]]
                                            + d_j * des[[j, l - hr.start]]);
                            }
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if l >= wr.start && l < wr.end {
                                val += t_j * c_j * ls.d1[[j, l - wr.start]] * h_j
                                    + r_j * ls.d2[[j, l - wr.start]] * h_j;
                            }
                        }
                    } else if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                        if k >= hr.start && k < hr.end && l >= hr.start && l < hr.end {
                            val += t_j
                                * c_j
                                * d_j
                                * b
                                * b
                                * des[[j, k - hr.start]]
                                * des[[j, l - hr.start]]
                                + r_j
                                    * e_j
                                    * b
                                    * b
                                    * des[[j, k - hr.start]]
                                    * des[[j, l - hr.start]];
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if k >= hr.start && k < hr.end && l >= wr.start && l < wr.end {
                                val += t_j
                                    * c_j
                                    * ls.d1[[j, l - wr.start]]
                                    * b
                                    * des[[j, k - hr.start]]
                                    + r_j * ls.d2[[j, l - wr.start]] * b * des[[j, k - hr.start]];
                            }
                        }
                    }
                    m_a_kl[[k, l]] += val;
                    if l != k {
                        m_a_kl[[l, k]] += val;
                    }
                }
            }

            // 4th order: D_uv cross quantities using total-derivative scalars.
            // p_jv = a_v + du_jv, p_ju = a_u + du_ju (computed after IFT).
            // sigma_jv = c_j * p_jv + B_jv = c_j * (a_v + du_v) + B_v_contracted = xi_v + c_j * a_v
            // Similarly sigma_ju = xi_u + c_j * a_u
            // These will be computed AFTER the node loop once a_u, a_v are known.
            // So we defer the D_uv accumulations to a second pass.
            // Store per-node base quantities for the second pass.
            // (We'll do this after computing IFT 1st/2nd order.)
        } // end first node loop

        // Finalize 1st/2nd order
        m_u[0] -= phi_q;
        m_uv[[0, 0]] += q_val * phi_q;
        m_uvw_v[[0, 0]] += dir_v[0] * (1.0 - q_val * q_val) * phi_q;
        m_uvw_u[[0, 0]] += dir_u[0] * (1.0 - q_val * q_val) * phi_q;

        // IFT 1st order
        let mut a_u_vec = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u_vec[u] = -m_u[u] * inv_ma;
        }
        // IFT 2nd order
        let mut a_uv_mat = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = -(m_uv[[u, v]]
                    + m_au[u] * a_u_vec[v]
                    + m_au[v] * a_u_vec[u]
                    + m_aa * a_u_vec[u] * a_u_vec[v])
                    * inv_ma;
                a_uv_mat[[u, v]] = val;
                a_uv_mat[[v, u]] = val;
            }
        }

        // IFT 3rd order for v
        let a_v_s = a_u_vec.dot(dir_v);
        let a_kv = a_uv_mat.dot(dir_v);
        let dm_a_v = m_au.dot(dir_v) + m_aa * a_v_s;
        let dm_aa_v = m_aau_k.dot(dir_v) + m_aaa * a_v_s;
        let dm_au_v: Array1<f64> = &m_auv_v + &(&m_aau_k * a_v_s);
        let dm_kl_v: Array2<f64> = &m_uvw_v + &(&m_a_kl * a_v_s);
        let mut a_klv = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let dpsi = dm_kl_v[[k, l]]
                    + dm_au_v[k] * a_u_vec[l]
                    + m_au[k] * a_kv[l]
                    + dm_au_v[l] * a_u_vec[k]
                    + m_au[l] * a_kv[k]
                    + dm_aa_v * a_u_vec[k] * a_u_vec[l]
                    + m_aa * (a_kv[k] * a_u_vec[l] + a_u_vec[k] * a_kv[l]);
                let val = -dpsi * inv_ma - a_uv_mat[[k, l]] * dm_a_v * inv_ma;
                a_klv[[k, l]] = val;
                a_klv[[l, k]] = val;
            }
        }

        // IFT 3rd order for u (same structure)
        let a_u_s = a_u_vec.dot(dir_u);
        let a_ku = a_uv_mat.dot(dir_u);
        let dm_a_u = m_au.dot(dir_u) + m_aa * a_u_s;
        let dm_aa_u = m_aau_k.dot(dir_u) + m_aaa * a_u_s;
        let dm_au_u: Array1<f64> = &m_auv_u + &(&m_aau_k * a_u_s);
        let dm_kl_u: Array2<f64> = &m_uvw_u + &(&m_a_kl * a_u_s);
        let mut a_klu = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let dpsi = dm_kl_u[[k, l]]
                    + dm_au_u[k] * a_u_vec[l]
                    + m_au[k] * a_ku[l]
                    + dm_au_u[l] * a_u_vec[k]
                    + m_au[l] * a_ku[k]
                    + dm_aa_u * a_u_vec[k] * a_u_vec[l]
                    + m_aa * (a_ku[k] * a_u_vec[l] + a_u_vec[k] * a_ku[l]);
                let val = -dpsi * inv_ma - a_uv_mat[[k, l]] * dm_a_u * inv_ma;
                a_klu[[k, l]] = val;
                a_klu[[l, k]] = val;
            }
        }

        // Contracted derived quantities
        let a_uv_c = dir_u.dot(&a_uv_mat.dot(dir_v)); // scalar u^T a_{kl} v
        let a_kuv = a_klv.dot(dir_u); // r-vector: eta_{kuv}

        // ── Second node pass for D_uv cross quantities ────────────────
        // Now that a_u, a_v, a_uv_c are known, compute per-node total-
        // derivative cross scalars and accumulate D_uv[M] quantities.
        for j in 0..nq {
            let h_j = h_nodes[j];
            let omega_j = self.quadrature_weights[j];
            let (s_j, c_j, d_j, e_j, f_j_link) = if let Some(ls) = row_ctx.node_link.as_ref() {
                let bw = beta_w.unwrap();
                let u_j = a + b * h_j;
                (
                    u_j + ls.basis.row(j).dot(bw),
                    1.0 + ls.d1.row(j).dot(bw),
                    ls.d2.row(j).dot(bw),
                    ls.d3.row(j).dot(bw),
                    ls.d4.row(j).dot(bw),
                )
            } else {
                (a + b * h_j, 1.0, 0.0, 0.0, 0.0)
            };

            let r_j = omega_j * normal_pdf(s_j);
            let t_j = -s_j * r_j;
            let u3_j = (s_j * s_j - 1.0) * r_j;
            let u4_j = -(s_j * s_j * s_j - 3.0 * s_j) * r_j;

            // du_jv, du_ju from du_arr . dir
            let mut du_jv = h_j * dir_v[1];
            let mut du_ju = h_j * dir_u[1];
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    du_jv += b * des[[j, k]] * dir_v[hr.start + k];
                    du_ju += b * des[[j, k]] * dir_u[hr.start + k];
                }
            }
            let p_jv = a_v_s + du_jv;
            let p_ju = a_u_s + du_ju;

            // dh contracted
            let mut dh_jv = 0.0;
            let mut dh_ju = 0.0;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    dh_jv += des[[j, k]] * dir_v[hr.start + k];
                    dh_ju += des[[j, k]] * dir_u[hr.start + k];
                }
            }
            let d2u_juv = dh_ju * dir_v[1] + dir_u[1] * dh_jv;
            let p_juv = a_uv_c + d2u_juv;

            // Link basis contracted with v and u
            let mut bp_jv = 0.0;
            let mut bp_ju = 0.0;
            let mut bpp_jv = 0.0;
            let mut bpp_ju = 0.0;
            let mut bppp_jv = 0.0;
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    bp_jv += ls.d1[[j, ell]] * dir_v[wr.start + ell];
                    bp_ju += ls.d1[[j, ell]] * dir_u[wr.start + ell];
                    bpp_jv += ls.d2[[j, ell]] * dir_v[wr.start + ell];
                    bpp_ju += ls.d2[[j, ell]] * dir_u[wr.start + ell];
                    bppp_jv += ls.d3[[j, ell]] * dir_v[wr.start + ell];
                }
            }

            // Total-derivative scalars
            let mut b_jv = 0.0;
            let mut b_ju = 0.0;
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    b_jv += ls.basis[[j, ell]] * dir_v[wr.start + ell];
                    b_ju += ls.basis[[j, ell]] * dir_u[wr.start + ell];
                }
            }
            let sigma_jv = c_j * p_jv + b_jv;
            let sigma_ju = c_j * p_ju + b_ju;
            let tau_jv = d_j * p_jv + bp_jv;
            let tau_ju = d_j * p_ju + bp_ju;
            let rho_jv = e_j * p_jv + bpp_jv;
            let rho_ju = e_j * p_ju + bpp_ju;

            // Cross scalars
            let sigma_juv = tau_ju * p_jv + c_j * p_juv + bp_jv * p_ju;
            // = tau_ju * p_jv + c_j * p_juv + B'_jv * p_ju
            // Wait no: B_jv = sum B_l(u_j) * v_{w_l}. D_u[B_jv] = sum B'_l(u_j) * D_u[u_j] * v_{w_l} = sum B'_l * p_ju * v_{w_l}.
            // Actually no: B_jv = Σ B_l(u_j) v_{w_l}. D_u[u_j] = p_ju.
            // D_u[B_l(u_j)] = B'_l(u_j) p_ju. So D_u[B_jv] = (Σ B'_l v_{w_l}) p_ju = bp_jv * p_ju.
            // But wait: B_jv is NOT in sigma_jv! Let me recheck.
            // sigma_jv = total ds_j/dv. s_j = u_j + Σ B_l(u_j) w_l.
            // ds_j/dv_total = (∂s_j/∂u_j)(du_j/dv_total) + Σ B_l(u_j) (dw_l/dv)
            // So sigma_juv = tau_ju * p_jv + c_j * p_juv + bp_jv * p_ju. ✓

            let tau_juv = rho_ju * p_jv + d_j * p_juv + bpp_jv * p_ju;
            let rho_juv = (f_j_link * p_ju) * p_jv + e_j * p_juv + bppp_jv * p_ju;

            // D_uv[M_a] = Σ w_j [Φ''' c σ_u σ_v + Φ''(σ_uv c + σ_v τ_u + σ_u τ_v) + Φ' τ_uv]
            d2m_a_uv += u3_j * c_j * sigma_ju * sigma_jv
                + t_j * (sigma_juv * c_j + sigma_jv * tau_ju + sigma_ju * tau_jv)
                + r_j * tau_juv;

            // D_uv[M_{aa}] using the chain rule on G = Φ'' c² + Φ' d
            // ∂G/∂s = Φ''' c² + Φ'' d,  ∂G/∂c = 2Φ'' c,  ∂G/∂d = Φ'
            // D_uv[G] = (Φ'''' c² + Φ''' d) σ_u σ_v + 2Φ''' c (σ_u τ_v + σ_v τ_u)
            //          + Φ''(σ_u ρ_v + σ_v ρ_u) + 2Φ'' τ_u τ_v
            //          + (Φ''' c² + Φ'' d) σ_uv + 2Φ'' c τ_uv + Φ' ρ_uv
            d2m_aa_uv += (u4_j * c_j * c_j + u3_j * d_j) * sigma_ju * sigma_jv
                + 2.0 * u3_j * c_j * (sigma_ju * tau_jv + sigma_jv * tau_ju)
                + t_j * (sigma_ju * rho_jv + sigma_jv * rho_ju)
                + 2.0 * t_j * tau_ju * tau_jv
                + (u3_j * c_j * c_j + t_j * d_j) * sigma_juv
                + 2.0 * t_j * c_j * tau_juv
                + r_j * rho_juv;

            // D_uv[M_{ak}] for each k: per-k accumulation
            // H_jk = Φ'' c ξ_k + Φ' dc_k. Treat as h(s, c, ξ_k, dc_k).
            // D_uv[h] involves 9 terms (see derivation).
            // We use total-derivative scalars: D_w[ξ_k] = dxi_w[k], D_w[dc_k] = ddc_w[k].
            // D_uv[ξ_k] and D_uv[dc_k] need to be computed.

            // Rebuild dxi_v, dxi_u, ddc_v, ddc_u for this node
            let mut dxi_v_arr = Array1::<f64>::zeros(r);
            let mut dxi_u_arr = Array1::<f64>::zeros(r);
            let mut ddc_v_a = Array1::<f64>::zeros(r);
            let mut ddc_u_a = Array1::<f64>::zeros(r);
            // These are TOTAL derivatives D_w[ξ_k] = τ_w du_k + c_j D_w[du_k] + δ(k∈w) B'_k p_w
            dxi_v_arr[1] = tau_jv * h_j + c_j * dh_jv;
            dxi_u_arr[1] = tau_ju * h_j + c_j * dh_ju;
            ddc_v_a[1] = rho_jv * h_j + d_j * dh_jv;
            ddc_u_a[1] = rho_ju * h_j + d_j * dh_ju;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    let idx = hr.start + k;
                    dxi_v_arr[idx] = tau_jv * b * des[[j, k]] + c_j * des[[j, k]] * dir_v[1];
                    dxi_u_arr[idx] = tau_ju * b * des[[j, k]] + c_j * des[[j, k]] * dir_u[1];
                    ddc_v_a[idx] = rho_jv * b * des[[j, k]] + d_j * des[[j, k]] * dir_v[1];
                    ddc_u_a[idx] = rho_ju * b * des[[j, k]] + d_j * des[[j, k]] * dir_u[1];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    let idx = wr.start + ell;
                    dxi_v_arr[idx] = ls.d1[[j, ell]] * p_jv; // B'_l * p_jv
                    dxi_u_arr[idx] = ls.d1[[j, ell]] * p_ju;
                    ddc_v_a[idx] = ls.d2[[j, ell]] * p_jv; // B''_l * p_jv
                    ddc_u_a[idx] = ls.d2[[j, ell]] * p_ju;
                }
            }

            // D_uv[ξ_k] = tau_juv du_k + tau_jv D_u[du_k] + tau_ju D_v[du_k] + δ(k∈w)(B''_k p_ju p_jv + B'_k p_juv)
            let mut dxi_uv = Array1::<f64>::zeros(r);
            dxi_uv[1] = tau_juv * h_j + tau_jv * dh_ju + tau_ju * dh_jv;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    let idx = hr.start + k;
                    dxi_uv[idx] = tau_juv * b * des[[j, k]]
                        + tau_jv * des[[j, k]] * dir_u[1]
                        + tau_ju * des[[j, k]] * dir_v[1];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    dxi_uv[wr.start + ell] =
                        ls.d2[[j, ell]] * p_ju * p_jv + ls.d1[[j, ell]] * p_juv;
                }
            }

            // D_uv[dc_k] = rho_juv du_k + rho_jv D_u[du_k] + rho_ju D_v[du_k] + δ(k∈w)(B'''_k p_ju p_jv + B''_k p_juv)
            let mut ddc_uv = Array1::<f64>::zeros(r);
            ddc_uv[1] = rho_juv * h_j + rho_jv * dh_ju + rho_ju * dh_jv;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    ddc_uv[hr.start + k] = rho_juv * b * des[[j, k]]
                        + rho_jv * des[[j, k]] * dir_u[1]
                        + rho_ju * des[[j, k]] * dir_v[1];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    ddc_uv[wr.start + ell] =
                        ls.d3[[j, ell]] * p_ju * p_jv + ls.d2[[j, ell]] * p_juv;
                }
            }

            // Rebuild xi per k (again, for this node)
            let mut xi = Array1::<f64>::zeros(r);
            let mut dc_a = Array1::<f64>::zeros(r);
            xi[1] = c_j * h_j;
            dc_a[1] = d_j * h_j;
            if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                for k in 0..n_h {
                    xi[hr.start + k] = c_j * b * des[[j, k]];
                    dc_a[hr.start + k] = d_j * b * des[[j, k]];
                }
            }
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.node_link.as_ref()) {
                for ell in 0..n_w {
                    xi[wr.start + ell] = ls.basis[[j, ell]];
                    dc_a[wr.start + ell] = ls.d1[[j, ell]];
                }
            }

            // D_uv[M_{ak}] per k using chain rule on H(s, c, xi_k, dc_k)
            for k in 0..r {
                let val = (u4_j * c_j * xi[k] + u3_j * dc_a[k]) * sigma_ju * sigma_jv
                    + u3_j * xi[k] * (sigma_ju * tau_jv + sigma_jv * tau_ju)
                    + u3_j * c_j * (sigma_ju * dxi_v_arr[k] + sigma_jv * dxi_u_arr[k])
                    + t_j * (sigma_ju * ddc_v_a[k] + sigma_jv * ddc_u_a[k])
                    + t_j * (tau_ju * dxi_v_arr[k] + tau_jv * dxi_u_arr[k])
                    + (u3_j * c_j * xi[k] + t_j * dc_a[k]) * sigma_juv
                    + t_j * xi[k] * tau_juv
                    + t_j * c_j * dxi_uv[k]
                    + r_j * ddc_uv[k];
                d2m_au_uv[k] += val;
            }

            // D_uv[M_{kl}] per (k,l) using chain rule on F(s, xi_k, xi_l, Delta_kl)
            // F = Phi'' xi_k xi_l + Phi' Delta_kl
            // The dominant terms: Phi'''' xi_k xi_l sigma_u sigma_v + ...
            // For simplicity, use the "symmetric total derivative" approach:
            // D_uv[Phi'' xi_k xi_l] = u4 xi_k xi_l sigma_u sigma_v
            //   + u3(xi_k xi_l sigma_uv + xi_k dxi_lv sigma_u + xi_k dxi_lu sigma_v + dxi_kv xi_l sigma_u + dxi_ku xi_l sigma_v + xi_k xi_l ... hmm
            // Actually this is getting very complex for the full (k,l) case.
            // Let me use a simpler decomposition:
            // M_{kl} per node = t_j * xi_k * xi_l + r_j * Delta_kl
            // where Delta_kl = dxi_jkl (structure derivative corrections)
            // D_uv of this = D_uv[t_j * xi_k * xi_l] + D_uv[r_j * Delta_kl]

            // D_uv[t_j xi_k xi_l] where t_j = Phi''(s_j) = -s_j Phi'(s_j)
            // = D_uv[Phi''(s)] xi_k xi_l + D_u[Phi''(s)](D_v[xi_k] xi_l + xi_k D_v[xi_l])
            //   + D_v[Phi''(s)](D_u[xi_k] xi_l + xi_k D_u[xi_l])
            //   + Phi''(s)(D_uv[xi_k] xi_l + D_u[xi_k] D_v[xi_l] + D_v[xi_k] D_u[xi_l] + xi_k D_uv[xi_l])

            // D_w[Phi''(s)] = Phi'''(s) sigma_w = u3_j sigma_w / (omega_j phi(s_j))... hmm
            // Actually: t_j = omega_j Phi''(s_j). D_w[t_j] = omega_j Phi'''(s_j) sigma_jw = u3_j sigma_jw.
            // D_uv[t_j] = omega_j Phi''''(s_j) sigma_ju sigma_jv + omega_j Phi'''(s_j) sigma_juv
            //            = u4_j sigma_ju sigma_jv + u3_j sigma_juv.

            for k in 0..r {
                if xi[k] == 0.0 && dxi_v_arr[k] == 0.0 && dxi_u_arr[k] == 0.0 && dxi_uv[k] == 0.0 {
                    continue;
                }
                for l in k..r {
                    if xi[l] == 0.0
                        && dxi_v_arr[l] == 0.0
                        && dxi_u_arr[l] == 0.0
                        && dxi_uv[l] == 0.0
                    {
                        continue;
                    }
                    // D_uv[t_j xi_k xi_l]
                    let d2t = u4_j * sigma_ju * sigma_jv + u3_j * sigma_juv;
                    let dt_u = u3_j * sigma_ju;
                    let dt_v = u3_j * sigma_jv;
                    let val_main = d2t * xi[k] * xi[l]
                        + dt_u * (dxi_v_arr[k] * xi[l] + xi[k] * dxi_v_arr[l])
                        + dt_v * (dxi_u_arr[k] * xi[l] + xi[k] * dxi_u_arr[l])
                        + t_j
                            * (dxi_uv[k] * xi[l]
                                + dxi_u_arr[k] * dxi_v_arr[l]
                                + dxi_v_arr[k] * dxi_u_arr[l]
                                + xi[k] * dxi_uv[l]);

                    // D_uv[r_j Delta_kl]: Delta_kl = structure derivative corrections.
                    // For simplicity and correctness, compute Delta_kl from the 2nd-order
                    // m_uv accumulation pattern. Delta_kl per node is what gets added beyond
                    // the t_j * xi_k * xi_l rank-1 term.
                    // From the m_uv accumulation: the "correction" terms are r_j * something.
                    // D_uv[r_j * Delta_kl]:
                    //   = D_uv[r_j]*Delta + D_u[r_j]*D_v[Delta] + D_v[r_j]*D_u[Delta] + r_j*D_uv[Delta]
                    // where D_w[r_j] = t_j*sigma_w, D_uv[r_j] = u3_j*sigma_u*sigma_v + t_j*sigma_uv.
                    // Delta_kl = dxi_jk/dtheta_l (structure derivative correction).
                    // D_w[Delta] depends on (k,l) type and involves link d/e derivatives.

                    let mut delta_kl = 0.0;
                    let mut d_delta_u = 0.0;
                    let mut d_delta_v = 0.0;
                    let mut d2_delta_uv = 0.0;

                    // Compute delta_kl and its total derivatives along u and v.
                    // delta_kl = dxi_jk/dtheta_l.  For k=g: dxi_jg/dtheta_l = dc_jl*h_j + c_j*(dh_j/dtheta_l).
                    // For k=h_k': dxi_j,h_k'/dtheta_l = dc_jl*b*D_{jk'} + c_j*D_{jk'}*delta(l=g).
                    // For k=w_m: dxi_j,w_m/dtheta_l = B'_m(u_j)*(du_j/dtheta_l).
                    if k == 1 && l == 1 {
                        // delta = d_j*h_j^2 (from dc_jg*h_j where dc_jg=d_j*h_j)
                        delta_kl = d_j * h_j * h_j;
                        d_delta_u = rho_ju * h_j * h_j + 2.0 * d_j * h_j * dh_ju;
                        d_delta_v = rho_jv * h_j * h_j + 2.0 * d_j * h_j * dh_jv;
                        d2_delta_uv = rho_juv * h_j * h_j
                            + rho_ju * 2.0 * h_j * dh_jv
                            + rho_jv * 2.0 * h_j * dh_ju
                            + 2.0 * d_j * (dh_ju * dh_jv);
                    } else if k == 1 {
                        if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                            if l >= hr.start && l < hr.end {
                                let dl = des[[j, l - hr.start]];
                                // delta = d_j*b*dl*h_j + c_j*dl
                                delta_kl = d_j * b * dl * h_j + c_j * dl;
                                d_delta_u = rho_ju * b * dl * h_j
                                    + d_j * dl * (dir_u[1] * h_j + b * dh_ju)
                                    + tau_ju * dl;
                                d_delta_v = rho_jv * b * dl * h_j
                                    + d_j * dl * (dir_v[1] * h_j + b * dh_jv)
                                    + tau_jv * dl;
                                d2_delta_uv = rho_juv * b * dl * h_j
                                    + rho_ju * dl * (dir_v[1] * h_j + b * dh_jv)
                                    + rho_jv * dl * (dir_u[1] * h_j + b * dh_ju)
                                    + d_j * dl * (dir_u[1] * dh_jv + dir_v[1] * dh_ju)
                                    + tau_juv * dl;
                            }
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if l >= wr.start && l < wr.end {
                                let li = l - wr.start;
                                // delta = B'_l*h_j (from dc_j,w_l = B'_l, times h_j from k=g geometry)
                                delta_kl = ls.d1[[j, li]] * h_j;
                                d_delta_u = ls.d2[[j, li]] * p_ju * h_j + ls.d1[[j, li]] * dh_ju;
                                d_delta_v = ls.d2[[j, li]] * p_jv * h_j + ls.d1[[j, li]] * dh_jv;
                                d2_delta_uv = ls.d3[[j, li]] * p_ju * p_jv * h_j
                                    + ls.d2[[j, li]] * p_juv * h_j
                                    + ls.d2[[j, li]] * (p_ju * dh_jv + p_jv * dh_ju)
                                    + ls.d1[[j, li]] * 0.0; // dh is constant w.r.t. theta
                            }
                        }
                    } else if let (Some(hr), Some(des)) = (primary.h.as_ref(), h_node_design) {
                        if k >= hr.start && k < hr.end && l >= hr.start && l < hr.end {
                            let dk_val = des[[j, k - hr.start]];
                            let dl_val = des[[j, l - hr.start]];
                            // delta = d_j*b^2*dk*dl
                            delta_kl = d_j * b * b * dk_val * dl_val;
                            d_delta_u = rho_ju * b * b * dk_val * dl_val
                                + 2.0 * d_j * b * dk_val * dl_val * dir_u[1];
                            d_delta_v = rho_jv * b * b * dk_val * dl_val
                                + 2.0 * d_j * b * dk_val * dl_val * dir_v[1];
                            d2_delta_uv = rho_juv * b * b * dk_val * dl_val
                                + rho_ju * 2.0 * b * dk_val * dl_val * dir_v[1]
                                + rho_jv * 2.0 * b * dk_val * dl_val * dir_u[1]
                                + 2.0 * d_j * dk_val * dl_val * dir_u[1] * dir_v[1];
                        }
                        if let (Some(wr), Some(ls)) =
                            (primary.w.as_ref(), row_ctx.node_link.as_ref())
                        {
                            if k >= hr.start && k < hr.end && l >= wr.start && l < wr.end {
                                let dk_val = des[[j, k - hr.start]];
                                let li = l - wr.start;
                                // delta = B'_l*b*dk
                                delta_kl = ls.d1[[j, li]] * b * dk_val;
                                d_delta_u = ls.d2[[j, li]] * p_ju * b * dk_val
                                    + ls.d1[[j, li]] * dk_val * dir_u[1];
                                d_delta_v = ls.d2[[j, li]] * p_jv * b * dk_val
                                    + ls.d1[[j, li]] * dk_val * dir_v[1];
                                d2_delta_uv = ls.d3[[j, li]] * p_ju * p_jv * b * dk_val
                                    + ls.d2[[j, li]] * p_juv * b * dk_val
                                    + ls.d2[[j, li]] * dk_val * (p_ju * dir_v[1] + p_jv * dir_u[1])
                                    + ls.d1[[j, li]] * dk_val * 0.0;
                            }
                        }
                    }
                    // D_uv[r_j*Delta] = D_uv[r_j]*Delta + D_u[r_j]*D_v[Delta] + D_v[r_j]*D_u[Delta] + r_j*D_uv[Delta]
                    let d2r = u3_j * sigma_ju * sigma_jv + t_j * sigma_juv;
                    let dr_u = t_j * sigma_ju;
                    let dr_v = t_j * sigma_jv;
                    let val_delta =
                        d2r * delta_kl + dr_u * d_delta_v + dr_v * d_delta_u + r_j * d2_delta_uv;

                    let total = val_main + val_delta;
                    d2m_kl_uv[[k, l]] += total;
                    if l != k {
                        d2m_kl_uv[[l, k]] += total;
                    }
                }
            }
        } // end second node loop

        // Add q-dependent terms for d2m_kl_uv[0,0]
        // D_uv[-Phi(q)] at (0,0): d^4(-Phi(q))/dq^4 contracted with dir_u[0]*dir_v[0]
        // d^3(-Phi)/dq^3 = -(q^2-1)phi(q), d^4(-Phi)/dq^4 = (q^3-3q)phi(q)
        d2m_kl_uv[[0, 0]] +=
            dir_u[0] * dir_v[0] * (q_val.powi(3) - 3.0 * q_val) * phi_q + dir_u[0] * dir_v[0] * 0.0; // (already included in node terms? no, q term is separate)
        // Actually: d^2/du dv [q phi(q)] at (0,0) has already been accumulated from nodes? No.
        // The m_uv[0,0] += q*phi(q) was for the second derivative.
        // The d2m_kl_uv needs the FOURTH partial of -Phi(q)/dq^4 contracted.
        // But wait: D_uv[M_{kl}] for (0,0) involves D_uv of (q*phi(q)):
        // d/dq[q phi(q)] = phi(q) - q^2 phi(q) = (1-q^2)phi(q).
        // d^2/dq^2[q phi(q)] = (-2q - (1-q^2)q)phi(q) = (-3q + q^3)phi(q) = (q^3-3q)phi(q).
        // So D_uv[q phi(q)] = (q^3-3q)phi(q) * dir_u[0] * dir_v[0].
        // This is what I wrote. But it should be:
        // Actually M_{kl} at (0,0) includes q*phi(q). D_uv of this:
        // this is a function of q only. D_v[q phi(q)] = (1-q^2)phi(q) * v_q.
        // D_uv[q phi(q)] = (q^3-3q)phi(q) * u_q * v_q.
        // Already correct.

        // Add M_{a,kl} * a_uv_c terms to complete D_uv[M_{kl}]
        // D_uv[M_{kl}] = d2m_kl_uv + D_uv[M_{a,kl}*...] terms
        // Actually: D_uv[M_{kl}] as total derivative already includes the a contribution
        // through the sigma terms. So d2m_kl_uv IS the full D_uv[M_{kl}]. No extra m_a_kl term needed.
        // So it IS the full second total derivative. ✓

        // 4th order IFT:
        // a_{kluv} = -[D_u[Psi_3v] + D_uv[M_a]*a_{kl} + D_v[M_a]*a_{klu} + D_u[M_a]*a_{klv}] / M_a
        //
        // D_u[Psi_3v] expands to:
        // D_uv[M_{kl}] + D_uv[M_{ak}]*a_l + D_v[M_{ak}]*a_{lu} + D_u[M_{ak}]*a_{lv} + M_{ak}*a_{luv}
        // + D_uv[M_{al}]*a_k + D_v[M_{al}]*a_{ku} + D_u[M_{al}]*a_{kv} + M_{al}*a_{kuv}
        // + D_uv[M_{aa}]*a_k*a_l + D_v[M_{aa}]*(a_{ku}*a_l + a_k*a_{lu})
        // + D_u[M_{aa}]*(a_{kv}*a_l + a_k*a_{lv})
        // + M_{aa}*(a_{kuv}*a_l + a_{kv}*a_{lu} + a_{ku}*a_{lv} + a_k*a_{luv})

        let a_luv = a_kuv.clone(); // same vector (symmetric in klu indices)
        let mut a_kluv = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let du_psi3v = d2m_kl_uv[[k, l]]
                    + d2m_au_uv[k] * a_u_vec[l]
                    + dm_au_v[k] * a_ku[l]
                    + dm_au_u[k] * a_kv[l]
                    + m_au[k] * a_luv[l]
                    + d2m_au_uv[l] * a_u_vec[k]
                    + dm_au_v[l] * a_ku[k]
                    + dm_au_u[l] * a_kv[k]
                    + m_au[l] * a_kuv[k]
                    + d2m_aa_uv * a_u_vec[k] * a_u_vec[l]
                    + dm_aa_v * (a_ku[k] * a_u_vec[l] + a_u_vec[k] * a_ku[l])
                    + dm_aa_u * (a_kv[k] * a_u_vec[l] + a_u_vec[k] * a_kv[l])
                    + m_aa
                        * (a_kuv[k] * a_u_vec[l]
                            + a_kv[k] * a_ku[l]
                            + a_ku[k] * a_kv[l]
                            + a_u_vec[k] * a_luv[l]);
                let val = -(du_psi3v
                    + d2m_a_uv * a_uv_mat[[k, l]]
                    + dm_a_v * a_klu[[k, l]]
                    + dm_a_u * a_klv[[k, l]])
                    * inv_ma;
                a_kluv[[k, l]] = val;
                a_kluv[[l, k]] = val;
            }
        }

        // Phase 5: Observation-point chain rule
        let h_obs = row_ctx.h_obs_base;
        let (c_o, d_o, e_o, f_o) = if let Some(ls) = row_ctx.obs_link.as_ref() {
            let bw = beta_w.unwrap();
            (
                1.0 + ls.d1.row(0).dot(bw),
                ls.d2.row(0).dot(bw),
                ls.d3.row(0).dot(bw),
                ls.d4.row(0).dot(bw),
            )
        } else {
            (1.0, 0.0, 0.0, 0.0)
        };

        let mut uo_u_vec = a_u_vec.clone();
        uo_u_vec[1] += h_obs;
        if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
            for k in 0..n_h {
                uo_u_vec[hr.start + k] += b * obs[k];
            }
        }
        let mut eta_u_vec = uo_u_vec.mapv(|x| c_o * x);
        if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
            for ell in 0..n_w {
                eta_u_vec[wr.start + ell] += ls.basis[[0, ell]];
            }
        }

        let eta_val = if let Some(ls) = row_ctx.obs_link.as_ref() {
            (a + b * h_obs) + ls.basis.row(0).dot(&beta_w.unwrap().view())
        } else {
            a + b * h_obs
        };
        let signed_margin = s_y * eta_val;
        let (k1_s, k2_s, k3_s, k4_s) =
            signed_probit_neglog_derivatives_up_to_fourth(signed_margin, w_i);

        // Contracted scalars
        let eta_v = eta_u_vec.dot(dir_v);
        let eta_u = eta_u_vec.dot(dir_u);

        // eta_{kl}
        let mut eta_kl = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let mut uo_kl = a_uv_mat[[k, l]];
                if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
                    if k == 1 && hr.contains(&l) {
                        uo_kl += obs[l - hr.start];
                    } else if l == 1 && hr.contains(&k) {
                        uo_kl += obs[k - hr.start];
                    }
                }
                let mut val = d_o * uo_u_vec[k] * uo_u_vec[l] + c_o * uo_kl;
                if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
                    if wr.contains(&k) {
                        val += ls.d1[[0, k - wr.start]] * uo_u_vec[l];
                    }
                    if wr.contains(&l) {
                        val += ls.d1[[0, l - wr.start]] * uo_u_vec[k];
                    }
                }
                eta_kl[[k, l]] = val;
                eta_kl[[l, k]] = val;
            }
        }

        // eta_{ku}, eta_{kv}
        let eta_ku = eta_kl.dot(dir_u);
        let eta_kv_vec = eta_kl.dot(dir_v);
        let eta_uv_s = dir_u.dot(&eta_kl.dot(dir_v));

        // For eta_{klu}, eta_{klv}: use the observation-point chain rule from third.
        // eta_{klw} = e_o uo_k uo_l uo_w + d_o(uo_{kw}uo_l + uo_k uo_{lw} + uo_{kl}uo_w) + c_o a_{klw}
        //           + link B' and B'' cross terms
        let compute_eta_klw = |a_klw: &Array2<f64>, dir_w: &Array1<f64>| -> Array2<f64> {
            let uo_w = uo_u_vec.dot(dir_w);
            let mut uo_kw = a_uv_mat.dot(dir_w);
            if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
                for k in 0..n_h {
                    uo_kw[hr.start + k] += obs[k] * dir_w[1];
                }
                for k in 0..n_h {
                    uo_kw[1] += obs[k] * dir_w[hr.start + k];
                }
            }
            let mut bp_w = 0.0;
            if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
                for ell in 0..n_w {
                    bp_w += ls.d1[[0, ell]] * dir_w[wr.start + ell];
                }
            }
            let mut result = Array2::<f64>::zeros((r, r));
            for k in 0..r {
                for l in k..r {
                    let mut uo_kl = a_uv_mat[[k, l]];
                    if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
                        if k == 1 && hr.contains(&l) {
                            uo_kl += obs[l - hr.start];
                        } else if l == 1 && hr.contains(&k) {
                            uo_kl += obs[k - hr.start];
                        }
                    }
                    let mut val = e_o * uo_u_vec[k] * uo_u_vec[l] * uo_w
                        + d_o * (uo_kw[k] * uo_u_vec[l] + uo_u_vec[k] * uo_kw[l] + uo_kl * uo_w)
                        + c_o * a_klw[[k, l]];
                    if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
                        let mut bpp_w = 0.0;
                        for ell in 0..n_w {
                            bpp_w += ls.d2[[0, ell]] * dir_w[wr.start + ell];
                        }
                        val += bpp_w * uo_u_vec[k] * uo_u_vec[l];
                        if wr.contains(&k) {
                            val += ls.d2[[0, k - wr.start]] * uo_u_vec[l] * uo_w;
                            val += ls.d1[[0, k - wr.start]]
                                * (d_o * uo_u_vec[l] * uo_w + c_o * uo_kw[l]);
                        }
                        if wr.contains(&l) {
                            val += ls.d2[[0, l - wr.start]] * uo_u_vec[k] * uo_w;
                            val += ls.d1[[0, l - wr.start]]
                                * (d_o * uo_u_vec[k] * uo_w + c_o * uo_kw[k]);
                        }
                        val += bp_w * (d_o * uo_u_vec[k] * uo_u_vec[l] + c_o * uo_kl);
                    }
                    result[[k, l]] = val;
                    result[[l, k]] = val;
                }
            }
            result
        };
        let eta_klv = compute_eta_klw(&a_klv, dir_v);
        let eta_klu = compute_eta_klw(&a_klu, dir_u);
        let eta_kuv_vec = eta_klv.dot(dir_u); // r-vector

        // eta_{kluv}: 4th derivative of the linked predictor at the observation point.
        // The observation-point link is eta = uo + sum_m B_m(uo) * w_m, so the
        // exact 4th derivative is the partition expansion through the scalar
        // uo map plus the single-slot explicit w contributions.
        let uo_v = uo_u_vec.dot(dir_v);
        let uo_u_s = uo_u_vec.dot(dir_u);
        let mut uo_kv = a_uv_mat.dot(dir_v);
        let mut uo_ku = a_uv_mat.dot(dir_u);
        if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
            for k in 0..n_h {
                uo_kv[hr.start + k] += obs[k] * dir_v[1];
                uo_kv[1] += obs[k] * dir_v[hr.start + k];
                uo_ku[hr.start + k] += obs[k] * dir_u[1];
                uo_ku[1] += obs[k] * dir_u[hr.start + k];
            }
        }
        let uo_uv_s = dir_u.dot(&a_uv_mat.dot(dir_v));
        // B-contracted scalars at observation point for the w-direction components.
        let mut bp_u_obs = 0.0;
        let mut bp_v_obs = 0.0;
        let mut bpp_u_obs = 0.0;
        let mut bpp_v_obs = 0.0;
        let mut bppp_u_obs = 0.0;
        let mut bppp_v_obs = 0.0;
        if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
            for ell in 0..n_w {
                let dv_w = dir_v[wr.start + ell];
                let du_w = dir_u[wr.start + ell];
                bp_u_obs += ls.d1[[0, ell]] * du_w;
                bp_v_obs += ls.d1[[0, ell]] * dv_w;
                bpp_u_obs += ls.d2[[0, ell]] * du_w;
                bpp_v_obs += ls.d2[[0, ell]] * dv_w;
                bppp_u_obs += ls.d3[[0, ell]] * du_w;
                bppp_v_obs += ls.d3[[0, ell]] * dv_w;
            }
        }
        let mut eta_kluv = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let mut uo_kl = a_uv_mat[[k, l]];
                if let (Some(hr), Some(obs)) = (primary.h.as_ref(), obs_row.as_ref()) {
                    if k == 1 && hr.contains(&l) {
                        uo_kl += obs[l - hr.start];
                    } else if l == 1 && hr.contains(&k) {
                        uo_kl += obs[k - hr.start];
                    }
                }
                let x_k = uo_u_vec[k];
                let x_l = uo_u_vec[l];
                let x_u = uo_u_s;
                let x_v = uo_v;
                let x_ku = uo_ku[k];
                let x_lu = uo_ku[l];
                let x_kv = uo_kv[k];
                let x_lv = uo_kv[l];
                let x_uv = uo_uv_s;
                let x_klu = a_klu[[k, l]];
                let x_klv = a_klv[[k, l]];
                let x_kuv = a_kuv[k];
                let x_luv = a_kuv[l];
                let x_kluv = a_kluv[[k, l]];

                let mut val = c_o * x_kluv
                    + d_o
                        * (x_klu * x_v
                            + x_klv * x_u
                            + x_kuv * x_l
                            + x_luv * x_k
                            + uo_kl * x_uv
                            + x_ku * x_lv
                            + x_kv * x_lu)
                    + e_o
                        * (uo_kl * x_u * x_v
                            + x_ku * x_l * x_v
                            + x_kv * x_l * x_u
                            + x_lu * x_k * x_v
                            + x_lv * x_k * x_u
                            + x_uv * x_k * x_l)
                    + f_o * x_k * x_l * x_u * x_v
                    + bp_u_obs * x_klv
                    + bpp_u_obs * (uo_kl * x_v + x_kv * x_l + x_k * x_lv)
                    + bppp_u_obs * x_k * x_l * x_v
                    + bp_v_obs * x_klu
                    + bpp_v_obs * (uo_kl * x_u + x_ku * x_l + x_k * x_lu)
                    + bppp_v_obs * x_k * x_l * x_u;

                if let (Some(wr), Some(ls)) = (primary.w.as_ref(), row_ctx.obs_link.as_ref()) {
                    if wr.contains(&k) {
                        let ki = k - wr.start;
                        val += ls.d1[[0, ki]] * x_luv;
                        val += ls.d2[[0, ki]] * (x_lu * x_v + x_lv * x_u + x_l * x_uv);
                        val += ls.d3[[0, ki]] * x_l * x_u * x_v;
                    }
                    if wr.contains(&l) {
                        let li = l - wr.start;
                        val += ls.d1[[0, li]] * x_kuv;
                        val += ls.d2[[0, li]] * (x_ku * x_v + x_kv * x_u + x_k * x_uv);
                        val += ls.d3[[0, li]] * x_k * x_u * x_v;
                    }
                }
                eta_kluv[[k, l]] = val;
                eta_kluv[[l, k]] = val;
            }
        }

        // Phase 6: Assemble F[k,l] via Arbogast 4th-order chain rule
        let mut out = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            for l in k..r {
                let val = k4_s * eta_u_vec[k] * eta_u_vec[l] * eta_u * eta_v
                    + k3_s
                        * s_y
                        * (eta_kl[[k, l]] * eta_u * eta_v
                            + eta_ku[k] * eta_u_vec[l] * eta_v
                            + eta_kv_vec[k] * eta_u_vec[l] * eta_u
                            + eta_ku[l] * eta_u_vec[k] * eta_v
                            + eta_kv_vec[l] * eta_u_vec[k] * eta_u
                            + eta_uv_s * eta_u_vec[k] * eta_u_vec[l])
                    + k2_s
                        * (eta_kl[[k, l]] * eta_uv_s
                            + eta_ku[k] * eta_kv_vec[l]
                            + eta_kv_vec[k] * eta_ku[l]
                            + eta_klu[[k, l]] * eta_v
                            + eta_klv[[k, l]] * eta_u
                            + eta_kuv_vec[l] * eta_u_vec[k]
                            + eta_kuv_vec[k] * eta_u_vec[l])
                    + k1_s * s_y * eta_kluv[[k, l]];
                out[[k, l]] = val;
                out[[l, k]] = val;
            }
        }
        Ok(out)
    }
    /// Like `add_pullback_primary_hessian` but only accumulates the h/w
    /// cross-block contributions. The marginal-marginal, marginal-logslope,
    /// and logslope-logslope blocks are handled by the weighted-Gram operator.
    fn add_pullback_primary_hessian_hw_only(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;
        if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
            for (local_idx, global_idx) in block_h.clone().enumerate() {
                let h_q = h[[0, primary_h.start + local_idx]];
                if h_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let h_g = h[[1, primary_h.start + local_idx]];
                if h_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            target
                .slice_mut(s![block_h.clone(), block_h.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_h.start..primary_h.end,
                        primary_h.start..primary_h.end
                    ]),
                );
        }

        if let (Some(primary_w), Some(block_w)) = (primary.w.as_ref(), slices.w.as_ref()) {
            for (local_idx, global_idx) in block_w.clone().enumerate() {
                let w_q = h[[0, primary_w.start + local_idx]];
                if w_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let w_g = h[[1, primary_w.start + local_idx]];
                if w_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
                target
                    .slice_mut(s![block_h.clone(), block_w.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_h.start..primary_h.end,
                            primary_w.start..primary_w.end
                        ]),
                    );
                target
                    .slice_mut(s![block_w.clone(), block_h.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_w.start..primary_w.end,
                            primary_h.start..primary_h.end
                        ]),
                    );
            }

            target
                .slice_mut(s![block_w.clone(), block_w.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_w.start..primary_w.end,
                        primary_w.start..primary_w.end
                    ]),
                );
        }
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

        // ── Rigid closed-form: scalar kernel + design row ops ────────
        if !self.flex_active() {
            let out = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_out, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let q = block_states[0].eta[row];
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                            );
                            let h = k.primary_hessian(q);
                            let v_q = self
                                .marginal_design
                                .dot_row_view(row, direction.slice(s![slices.marginal.clone()]));
                            let v_g = self
                                .logslope_design
                                .dot_row_view(row, direction.slice(s![slices.logslope.clone()]));
                            let a_q = h[0][0] * v_q + h[0][1] * v_g;
                            let a_g = h[1][0] * v_q + h[1][1] * v_g;
                            {
                                let mut m = chunk_out.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design.axpy_row_into(row, a_q, &mut m)?;
                            }
                            {
                                let mut l = chunk_out.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design.axpy_row_into(row, a_g, &mut l)?;
                            }
                        }
                        Ok(chunk_out)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(out);
        }

        let out = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_out, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        let row_dir =
                            self.row_primary_direction_from_flat(row, slices, primary, direction)?;
                        let (_, _, row_hessian) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                            &cache.h_nodes,
                            cache.h_node_design.as_ref(),
                            cache.score_warp_obs.as_ref(),
                        )?;
                        let row_action = row_hessian.dot(&row_dir);
                        chunk_out +=
                            &self.pullback_primary_vector(row, slices, primary, &row_action)?;
                    }
                    Ok(chunk_out)
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
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

        // ── Rigid closed-form: no jets, no row contexts ──────────────
        if !self.flex_active() {
            let diagonal = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_diag, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let q = block_states[0].eta[row];
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                            );
                            let h = k.primary_hessian(q);
                            {
                                let mut m = chunk_diag.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design
                                    .squared_axpy_row_into(row, h[0][0], &mut m)?;
                            }
                            {
                                let mut l = chunk_diag.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design
                                    .squared_axpy_row_into(row, h[1][1], &mut l)?;
                            }
                        }
                        Ok(chunk_diag)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(diagonal);
        }

        let diagonal = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_diag, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        let (_, _, row_hessian) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                            &cache.h_nodes,
                            cache.h_node_design.as_ref(),
                            cache.score_warp_obs.as_ref(),
                        )?;

                        {
                            let mut marginal_diag =
                                chunk_diag.slice_mut(s![slices.marginal.clone()]);
                            self.marginal_design.squared_axpy_row_into(
                                row,
                                row_hessian[[0, 0]],
                                &mut marginal_diag,
                            )?;
                        }
                        {
                            let mut logslope_diag =
                                chunk_diag.slice_mut(s![slices.logslope.clone()]);
                            self.logslope_design.squared_axpy_row_into(
                                row,
                                row_hessian[[1, 1]],
                                &mut logslope_diag,
                            )?;
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
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
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
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let psi_label = if block_idx == 0 {
            "BernoulliMarginalSlopeFamily marginal"
        } else {
            "BernoulliMarginalSlopeFamily log-slope"
        };
        let psi_p = if block_idx == 0 {
            self.marginal_design.ncols()
        } else {
            self.logslope_design.ncols()
        };
        let psi_action = CustomFamilyPsiDesignAction::from_first_derivative(
            deriv,
            self.y.len(),
            psi_p,
            0..self.y.len(),
            psi_label,
        )
        .ok();
        let can_operatorize = psi_action.is_some();
        let has_hw = primary.h.is_some() || primary.w.is_some();

        if can_operatorize {
            let (objective_psi, score_psi, w_mm, w_mg, w_gm, w_gg, d_mm, d_mg, d_gg, dense_corr) =
                (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                    .into_par_iter()
                    .try_fold(
                        || {
                            (
                                0.0,
                                Array1::<f64>::zeros(slices.total),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array2::<f64>::zeros((slices.total, slices.total)),
                            )
                        },
                        |mut acc, chunk_idx| -> Result<_, String> {
                            let start = chunk_idx * ROW_CHUNK_SIZE;
                            let end = (start + ROW_CHUNK_SIZE).min(n);
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
                                let row_ctx = Self::row_ctx(cache, row);
                                let (_, f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                                    row,
                                    block_states,
                                    primary,
                                    row_ctx,
                                    &cache.h_nodes,
                                    cache.h_node_design.as_ref(),
                                    cache.score_warp_obs.as_ref(),
                                )?;
                                let third = self.row_primary_third_contracted_recompute(
                                    row,
                                    block_states,
                                    cache,
                                    row_ctx,
                                    &dir,
                                )?;
                                let psi_row = self.psi_design_row_vector(
                                    row,
                                    deriv,
                                    self.y.len(),
                                    psi_p,
                                    psi_label,
                                )?;

                                acc.0 += f_pi.dot(&dir);
                                match block_idx {
                                    0 => {
                                        let mut score_m =
                                            acc.1.slice_mut(s![slices.marginal.clone()]);
                                        score_m.scaled_add(f_pi[idx_primary], &psi_row);
                                    }
                                    1 => {
                                        let mut score_g =
                                            acc.1.slice_mut(s![slices.logslope.clone()]);
                                        score_g.scaled_add(f_pi[idx_primary], &psi_row);
                                    }
                                    _ => unreachable!(),
                                }
                                acc.1 += &self.pullback_primary_vector(
                                    row,
                                    slices,
                                    primary,
                                    &f_pipi.dot(&dir),
                                )?;

                                if block_idx == 0 {
                                    acc.2[row] = f_pipi[[0, 0]];
                                    acc.3[row] = f_pipi[[0, 1]];
                                    acc.4[row] = f_pipi[[1, 0]];
                                } else {
                                    acc.3[row] = f_pipi[[0, 1]];
                                    acc.4[row] = f_pipi[[1, 0]];
                                    acc.5[row] = f_pipi[[1, 1]];
                                }
                                acc.6[row] = third[[0, 0]];
                                acc.7[row] = third[[0, 1]];
                                acc.8[row] = third[[1, 1]];

                                // h/w cross-block contributions -> dense correction
                                if has_hw {
                                    let right_primary = f_pipi.row(idx_primary).to_owned();
                                    let psi_range = if block_idx == 0 {
                                        slices.marginal.clone()
                                    } else {
                                        slices.logslope.clone()
                                    };
                                    if let (Some(ph), Some(bh)) =
                                        (primary.h.as_ref(), slices.h.as_ref())
                                    {
                                        let h_part = right_primary.slice(s![ph.start..ph.end]);
                                        for (li, gi) in psi_range.clone().enumerate() {
                                            for (lj, gj) in bh.clone().enumerate() {
                                                let val = psi_row[li] * h_part[lj];
                                                acc.9[[gi, gj]] += val;
                                                acc.9[[gj, gi]] += val;
                                            }
                                        }
                                    }
                                    if let (Some(pw), Some(bw)) =
                                        (primary.w.as_ref(), slices.w.as_ref())
                                    {
                                        let w_part = right_primary.slice(s![pw.start..pw.end]);
                                        for (li, gi) in psi_range.enumerate() {
                                            for (lj, gj) in bw.clone().enumerate() {
                                                let val = psi_row[li] * w_part[lj];
                                                acc.9[[gi, gj]] += val;
                                                acc.9[[gj, gi]] += val;
                                            }
                                        }
                                    }
                                    self.add_pullback_primary_hessian_hw_only(
                                        &mut acc.9, row, slices, primary, &third,
                                    );
                                }
                            }
                            Ok(acc)
                        },
                    )
                    .try_reduce(
                        || {
                            (
                                0.0,
                                Array1::<f64>::zeros(slices.total),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array2::<f64>::zeros((slices.total, slices.total)),
                            )
                        },
                        |mut left, right| -> Result<_, String> {
                            left.0 += right.0;
                            left.1 += &right.1;
                            left.2 += &right.2;
                            left.3 += &right.3;
                            left.4 += &right.4;
                            left.5 += &right.5;
                            left.6 += &right.6;
                            left.7 += &right.7;
                            left.8 += &right.8;
                            left.9 += &right.9;
                            Ok(left)
                        },
                    )?;

            let channels = vec![
                CustomFamilyJointDesignChannel::new(
                    slices.marginal.clone(),
                    self.marginal_design.to_dense_arc(),
                    if block_idx == 0 {
                        psi_action.clone()
                    } else {
                        None
                    },
                ),
                CustomFamilyJointDesignChannel::new(
                    slices.logslope.clone(),
                    self.logslope_design.to_dense_arc(),
                    if block_idx == 1 { psi_action } else { None },
                ),
            ];
            let pair_contributions = vec![
                CustomFamilyJointDesignPairContribution::new(0, 0, w_mm, d_mm),
                CustomFamilyJointDesignPairContribution::new(0, 1, w_mg.clone(), d_mg.clone()),
                CustomFamilyJointDesignPairContribution::new(1, 0, w_gm, d_mg),
                CustomFamilyJointDesignPairContribution::new(1, 1, w_gg, d_gg),
            ];

            let dense_correction = if has_hw { Some(dense_corr) } else { None };

            return Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(Box::new(
                    CustomFamilyJointPsiOperator::with_dense_correction(
                        slices.total,
                        channels,
                        pair_contributions,
                        dense_correction,
                    ),
                )),
            }));
        }

        // Block-local accumulator path: avoids O(n p^2) dense Hessian
        let (objective_psi, score_psi, block_acc) = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
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
                        let row_ctx = Self::row_ctx(cache, row);
                        let (_, f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                            &cache.h_nodes,
                            cache.h_node_design.as_ref(),
                            cache.score_warp_obs.as_ref(),
                        )?;
                        let third = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir,
                        )?;
                        let psi_row = self
                            .block_psi_row(row, slices, derivative_blocks, psi_index)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi vector".to_string()
                            })?;
                        acc.0 += f_pi.dot(&dir);
                        acc.1
                            .slice_mut(s![psi_row.range.clone()])
                            .scaled_add(f_pi[idx_primary], &psi_row.local_vec);
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &f_pipi.dot(&dir),
                        )?;

                        // psi_row outer pullback(f_pipi[idx_primary,:]) + transpose
                        let right_primary = f_pipi.row(idx_primary).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_idx,
                            &psi_row.local_vec,
                            &right_primary,
                        );
                        // third tensor pullback
                        acc.2.add_pullback(self, row, slices, primary, &third);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(Box::new(block_acc.into_operator(slices))),
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
        let Some((block_i, _)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, _)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        let idx_i = if block_i == 0 { 0 } else { 1 };
        let idx_j = if block_j == 0 { 0 } else { 1 };
        let n = self.y.len();
        // Block-local accumulator path for second-order psi terms
        let (objective_psi_psi, score_psi_psi, block_acc) = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
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
                        let row_ctx = Self::row_ctx(cache, row);
                        let (_, f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                            &cache.h_nodes,
                            cache.h_node_design.as_ref(),
                            cache.score_warp_obs.as_ref(),
                        )?;
                        let third_i = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                        )?;
                        let third_j = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_j,
                        )?;
                        let fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                            &dir_j,
                        )?;
                        let br_i = self
                            .block_psi_row(row, slices, derivative_blocks, psi_i)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi_i vector".to_string()
                            })?;
                        let br_j = self
                            .block_psi_row(row, slices, derivative_blocks, psi_j)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi_j vector".to_string()
                            })?;
                        let br_ij = self.block_psi_second_row(
                            row,
                            slices,
                            derivative_blocks,
                            psi_i,
                            psi_j,
                        )?;

                        // --- scalar and score accumulation (unchanged) ---
                        acc.0 += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            acc.1
                                .slice_mut(s![bij.range.clone()])
                                .scaled_add(f_pi[idx_ij], &bij.local_vec);
                        }
                        acc.1
                            .slice_mut(s![br_i.range.clone()])
                            .scaled_add(f_pipi.row(idx_i).dot(&dir_j), &br_i.local_vec);
                        acc.1
                            .slice_mut(s![br_j.range.clone()])
                            .scaled_add(f_pipi.row(idx_j).dot(&dir_i), &br_j.local_vec);
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &f_pipi.dot(&dir_ij),
                        )?;
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &third_i.dot(&dir_j),
                        )?;

                        // --- Hessian: bij outer pullback(f_pipi[idx_ij,:]) + transpose ---
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            let right_primary_ij = f_pipi.row(idx_ij).to_owned();
                            acc.2.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                bij.block_idx,
                                &bij.local_vec,
                                &right_primary_ij,
                            );
                        }

                        // --- br_i outer br_j * f_pipi[[idx_i, idx_j]] + transpose ---
                        let scalar_ij = f_pipi[[idx_i, idx_j]];
                        acc.2.add_psi_psi_outer(
                            block_i,
                            &br_i.local_vec,
                            block_j,
                            &br_j.local_vec,
                            scalar_ij,
                        );

                        // --- br_i outer pullback(third_j[idx_i,:]) + transpose ---
                        let right_primary_i = third_j.row(idx_i).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_i,
                            &br_i.local_vec,
                            &right_primary_i,
                        );

                        // --- br_j outer pullback(third_i[idx_j,:]) + transpose ---
                        let right_primary_j = third_i.row(idx_j).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_j,
                            &br_j.local_vec,
                            &right_primary_j,
                        );

                        // --- fourth tensor pullback ---
                        acc.2.add_pullback(self, row, slices, primary, &fourth);

                        // --- third_ij tensor pullback ---
                        let third_ij = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_ij,
                        )?;
                        acc.2.add_pullback(self, row, slices, primary, &third_ij);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(block_acc.into_operator(slices))),
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
        let Some((block_idx, _)) = self.resolve_psi_location(derivative_blocks, psi_index) else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_dir = self.row_primary_direction_from_flat(
                            row,
                            slices,
                            primary,
                            d_beta_flat,
                        )?;
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
                        let row_ctx = Self::row_ctx(cache, row);
                        let third_beta = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                        )?;
                        let fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                            &psi_dir,
                        )?;
                        let psi_row = self
                            .block_psi_row(row, slices, derivative_blocks, psi_index)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi vector".to_string()
                            })?;
                        let right_primary = third_beta.row(idx_primary).to_owned();
                        acc.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            psi_row.block_idx,
                            &psi_row.local_vec,
                            &right_primary,
                        );
                        acc.add_pullback(self, row, slices, primary, &fourth);
                        let third_action = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &psi_action,
                        )?;
                        acc.add_pullback(self, row, slices, primary, &third_action);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
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

        // ── Rigid closed-form: 3rd-order scalar kernel ───────────────
        if !self.flex_active() {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let q = block_states[0].eta[row];
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                            );
                            let dq = self
                                .marginal_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                            let dg = self
                                .logslope_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                            let t = k.third_contracted(q, dq, dg);
                            let t_arr = Array2::from_shape_fn((2, 2), |(a, b)| t[a][b]);
                            acc.add_pullback(self, row, slices, primary, &t_arr);
                        }
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut left, right| {
                        left.add(&right);
                        Ok(left)
                    },
                )?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_dir = self.row_primary_direction_from_flat(
                            row,
                            slices,
                            primary,
                            d_beta_flat,
                        )?;
                        let row_ctx = Self::row_ctx(cache, row);
                        let third = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                        )?;
                        acc.add_pullback(self, row, slices, primary, &third);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
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
        let make_acc = || BernoulliBlockHessianAccumulator::new(slices);

        // ── Rigid closed-form: 4th-order scalar kernel ───────────────
        if !self.flex_active() {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let q = block_states[0].eta[row];
                        let g = block_states[1].eta[row];
                        let k = RigidProbitKernel::new(
                            q,
                            g,
                            self.z[row],
                            self.y[row],
                            self.weights[row],
                        );
                        let uq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_u_flat.slice(s![slices.marginal.clone()]));
                        let ug = self
                            .logslope_design
                            .dot_row_view(row, d_beta_u_flat.slice(s![slices.logslope.clone()]));
                        let vq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_v_flat.slice(s![slices.marginal.clone()]));
                        let vg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_v_flat.slice(s![slices.logslope.clone()]));
                        let f = k.fourth_contracted(q, uq, ug, vq, vg);
                        let f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                        acc.add_pullback(self, row, slices, primary, &f_arr);
                    }
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                })?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                for row in start..end {
                    let row_u =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                    let row_v =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_u,
                        &row_v,
                    )?;
                    acc.add_pullback(self, row, slices, primary, &fourth);
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let cache = self.build_exact_eval_cache(block_states)?;
        let slices = cache.slices.clone();
        let primary = cache.primary.clone();
        let mut ll = 0.0;
        let mut grad_marginal = Array1::<f64>::zeros(slices.marginal.len());
        let mut grad_logslope = Array1::<f64>::zeros(slices.logslope.len());
        let mut hess_marginal =
            Array2::<f64>::zeros((slices.marginal.len(), slices.marginal.len()));
        let mut hess_logslope =
            Array2::<f64>::zeros((slices.logslope.len(), slices.logslope.len()));
        let mut grad_h = slices.h.as_ref().map(|r| Array1::<f64>::zeros(r.len()));
        let mut grad_w = slices.w.as_ref().map(|r| Array1::<f64>::zeros(r.len()));
        let mut hess_h = slices
            .h
            .as_ref()
            .map(|r| Array2::<f64>::zeros((r.len(), r.len())));
        let mut hess_w = slices
            .w
            .as_ref()
            .map(|r| Array2::<f64>::zeros((r.len(), r.len())));

        // Fast path: rigid model (no score-warp, no link-deviation).
        // Use closed-form scalar kernel: η = q·c + g·z, ℓ = -w·logΦ(s·η).
        // No jets, no intercept solve, no quadrature.
        if !self.flex_active() {
            for row in 0..self.y.len() {
                let q = block_states[0].eta[row];
                let g = block_states[1].eta[row];
                let yi = self.y[row];
                let wi = self.weights[row];
                let zi = self.z[row];
                let s = 2.0 * yi - 1.0;

                let g2 = g * g;
                let c = (1.0 + g2).sqrt();
                // c(g) = sqrt(1+g²), c'(g) = g/c, c''(g) = 1/c³
                let c1 = g / c;
                let c2 = 1.0 / (c * c * c);
                let eta = q * c + g * zi;
                let m = s * eta;

                let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(m);
                ll += wi * logcdf;

                let u1 = -wi * s * lambda;
                let u2 = wi * lambda * (m + lambda);

                let deta_dq = c;
                let deta_dg = q * c1 + zi;

                {
                    let mut marginal = grad_marginal.view_mut();
                    self.marginal_design
                        .axpy_row_into(row, -(u1 * deta_dq), &mut marginal)?;
                }
                {
                    let mut logslope = grad_logslope.view_mut();
                    self.logslope_design
                        .axpy_row_into(row, -(u1 * deta_dg), &mut logslope)?;
                }
                self.marginal_design.syr_row_into(
                    row,
                    u2 * deta_dq * deta_dq,
                    &mut hess_marginal,
                )?;
                self.logslope_design.syr_row_into(
                    row,
                    u2 * deta_dg * deta_dg + u1 * q * c2,
                    &mut hess_logslope,
                )?;
            }

            return Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: SymmetricMatrix::Dense(hess_marginal),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: SymmetricMatrix::Dense(hess_logslope),
                    },
                ],
            });
        }

        for row in 0..self.y.len() {
            let row_ctx = Self::row_ctx(&cache, row);
            let (row_neglog, f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                row,
                block_states,
                &primary,
                row_ctx,
                &cache.h_nodes,
                cache.h_node_design.as_ref(),
                cache.score_warp_obs.as_ref(),
            )?;
            ll -= row_neglog;
            {
                let mut marginal = grad_marginal.view_mut();
                self.marginal_design
                    .axpy_row_into(row, -f_pi[0], &mut marginal)?;
            }
            {
                let mut logslope = grad_logslope.view_mut();
                self.logslope_design
                    .axpy_row_into(row, -f_pi[1], &mut logslope)?;
            }
            self.marginal_design
                .syr_row_into(row, f_pipi[[0, 0]], &mut hess_marginal)?;
            self.logslope_design
                .syr_row_into(row, f_pipi[[1, 1]], &mut hess_logslope)?;

            if let (Some(primary_h), Some(grad_h), Some(hess_h)) =
                (primary.h.as_ref(), grad_h.as_mut(), hess_h.as_mut())
            {
                *grad_h -= &f_pi.slice(s![primary_h.start..primary_h.end]).to_owned();
                *hess_h += &f_pipi
                    .slice(s![
                        primary_h.start..primary_h.end,
                        primary_h.start..primary_h.end
                    ])
                    .to_owned();
            }
            if let (Some(primary_w), Some(grad_w), Some(hess_w)) =
                (primary.w.as_ref(), grad_w.as_mut(), hess_w.as_mut())
            {
                *grad_w -= &f_pi.slice(s![primary_w.start..primary_w.end]).to_owned();
                *hess_w += &f_pipi
                    .slice(s![
                        primary_w.start..primary_w.end,
                        primary_w.start..primary_w.end
                    ])
                    .to_owned();
            }
        }

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: grad_marginal,
                hessian: SymmetricMatrix::Dense(hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_logslope,
                hessian: SymmetricMatrix::Dense(hess_logslope),
            },
        ];
        if let (Some(gradient), Some(hessian)) = (grad_h, hess_h) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        if let (Some(gradient), Some(hessian)) = (grad_w, hess_w) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }
}

impl CustomFamily for BernoulliMarginalSlopeFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        // Shared memory gate: K(K+1)/2 × p² dense psi Hessians.
        if cost_gated_outer_order(specs) == ExactOuterDerivativeOrder::First {
            return ExactOuterDerivativeOrder::First;
        }
        // Family-specific row-loop gate.
        let k_smoothing: usize = specs.iter().map(|s| s.penalties.len()).sum();
        bernoulli_row_work_order(
            self.y.len(),
            self.score_warp
                .as_ref()
                .map(|runtime| runtime.transform.ncols())
                .unwrap_or(0),
            self.link_dev
                .as_ref()
                .map(|runtime| runtime.transform.ncols())
                .unwrap_or(0),
            k_smoothing,
        )
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if !self.flex_active() {
            // Rigid probit: vectorized closed-form.
            // η_i = q_i·c_i + b_i·z_i  where c_i = √(1+b_i²)
            // ll = Σ w_i · log Φ((2y_i−1)·η_i)
            let q = &block_states[0].eta;
            let b = &block_states[1].eta;
            let n = self.y.len();
            let mut ll = 0.0;
            for i in 0..n {
                let c_i = (1.0 + b[i] * b[i]).sqrt();
                let eta_i = q[i] * c_i + b[i] * self.z[i];
                let signed = (2.0 * self.y[i] - 1.0) * eta_i;
                let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
                ll += self.weights[i] * log_cdf;
            }
            return Ok(ll);
        }
        // Flexible path: per-row intercept solve + link value evaluation.
        // No MultiDirJet, no d2/d3/d4 derivative stacks.
        let (h_nodes, _) = self.quadrature_h(block_states)?;
        let score_warp_obs = self.score_warp_obs(block_states)?;
        let beta_w = if self.link_dev.is_some() {
            block_states.last().map(|state| &state.beta)
        } else {
            None
        };
        let n = self.y.len();
        let mut ll = 0.0;
        for row in 0..n {
            let q_i = block_states[0].eta[row];
            let b_i = block_states[1].eta[row];
            // Solve intercept (reuses link_terms which needs basis + d1 only).
            let (a_i, _) = self.solve_row_intercept_base(q_i, b_i, &h_nodes, beta_w)?;
            // Observation heterogeneity offset.
            let h_obs = if let Some((_, ref dev_obs)) = score_warp_obs {
                self.z[row] + dev_obs[row]
            } else {
                self.z[row]
            };
            // Apply link at the observation point (basis only, no derivatives).
            let eta_obs = a_i + b_i * h_obs;
            let s_i = if let (Some(runtime), Some(beta)) = (&self.link_dev, beta_w) {
                let v = Array1::from_vec(vec![eta_obs]);
                let basis = runtime.design(&v)?;
                eta_obs + basis.row(0).dot(beta)
            } else {
                eta_obs
            };
            let signed = (2.0 * self.y[row] - 1.0) * s_i;
            let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
            ll += self.weights[row] * log_cdf;
        }
        Ok(ll)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
        if slices.total >= 512 {
            return Ok(None);
        }
        if !self.flex_active() {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern)?;
            return Ok(Some(crate::families::row_kernel::row_kernel_hessian_dense(
                &kern, &cache,
            )));
        }

        let cache = self.build_exact_eval_cache(block_states)?;
        let mut dense = Array2::<f64>::zeros((slices.total, slices.total));
        let mut basis = Array1::<f64>::zeros(slices.total);
        for col in 0..slices.total {
            basis[col] = 1.0;
            let applied =
                self.exact_newton_joint_hessian_matvec_from_cache(&basis, block_states, &cache)?;
            basis[col] = 0.0;
            dense.column_mut(col).assign(&applied);
        }
        Ok(Some(dense))
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if !self.flex_active() {
            // Rigid path: use generic RowKernel<2> operator
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            Ok(Some(Arc::new(RowKernelHessianWorkspace::new(kern)?)))
        } else {
            // Flex path: keep existing workspace for variable-K primary space
            Ok(Some(Arc::new(
                BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                )?,
            )))
        }
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.flex_active() {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let sl = d_beta_flat.as_slice().ok_or("non-contiguous d_beta")?;
            crate::families::row_kernel::row_kernel_directional_derivative(&kern, sl).map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessian_directional_derivative_from_cache(
                block_states,
                d_beta_flat,
                &cache,
            )
        }
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.flex_active() {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let su = d_beta_u_flat.as_slice().ok_or("non-contiguous d_beta_u")?;
            let sv = d_beta_v_flat.as_slice().ok_or("non-contiguous d_beta_v")?;
            crate::families::row_kernel::row_kernel_second_directional_derivative(&kern, su, sv)
                .map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessiansecond_directional_derivative_from_cache(
                block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &cache,
            )
        }
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
            return Ok(self.cached_score_warp_constraints.clone());
        }
        let link_block_idx = if slices.h.is_some() { 3 } else { 2 };
        if slices
            .w
            .as_ref()
            .is_some_and(|_| block_idx == link_block_idx)
        {
            return Ok(self.cached_link_dev_constraints.clone());
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
        freeze_term_collection_from_design(&spec.marginalspec, &marginal_design)
            .map_err(|e| e.to_string())?;
    let logslopespec_boot =
        freeze_term_collection_from_design(&spec.logslopespec, &logslope_design)
            .map_err(|e| e.to_string())?;

    let (quad_nodes, quad_weights) = normal_expectation_nodes(spec.quadrature_points);
    let y = Arc::new(spec.y.clone());
    let weights = Arc::new(spec.weights.clone());
    let z = Arc::new(spec.z.clone());

    let score_warp_prepared = spec
        .score_warp
        .as_ref()
        .map(|cfg| build_deviation_block_from_seed(&spec.z, cfg))
        .transpose()?;
    // Build the link-deviation block if requested.  The seed only determines
    // knot placement for the deviation basis, so we use the closed-form
    // pooled-intercept probit solution instead of a full rigid pilot solve
    // (which would double total work at biobank scale):
    //   q0 ≈ a0 · √(1 + b0²) + b0 · z[i]
    // where (a0, b0) = pooled_probit_baseline.
    let link_dev_prepared = spec
        .link_dev
        .as_ref()
        .map(|cfg| {
            let a0 = baseline.0;
            let b0 = baseline.1;
            let scale = (1.0 + b0 * b0).sqrt();
            let q0_seed = Array1::from_iter(spec.z.iter().map(|&zi| a0 * scale + b0 * zi));
            // Pad with a conservative envelope so the basis support covers
            // the domain reached during flexible joint optimization, not just
            // the baseline range.  Padding is 50 % of the IQR on each side.
            let link_dev_seed = {
                let mut sorted = q0_seed.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n_s = sorted.len();
                if n_s >= 4 {
                    let q1 = sorted[n_s / 4];
                    let q3 = sorted[3 * n_s / 4];
                    let iqr = (q3 - q1).max(1.0);
                    let pad = 0.5 * iqr;
                    let lo = sorted[0] - pad;
                    let hi = sorted[n_s - 1] + pad;
                    let mut padded = q0_seed.to_vec();
                    padded.push(lo);
                    padded.push(hi);
                    Array1::from_vec(padded)
                } else {
                    q0_seed
                }
            };
            build_deviation_block_from_seed(&link_dev_seed, cfg)
        })
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
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let score_warp_obs_design = score_warp_prepared.as_ref().map(|p| p.block.design.clone());
    let cached_score_warp_constraints = score_warp_runtime
        .as_ref()
        .map(|runtime| {
            BernoulliMarginalSlopeFamily::build_score_warp_constraints(
                runtime,
                z.as_ref(),
                &quad_nodes,
            )
        })
        .transpose()?
        .flatten();
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());
    let cached_link_dev_constraints = link_dev_runtime
        .as_ref()
        .map(BernoulliMarginalSlopeFamily::build_link_dev_constraints)
        .transpose()?
        .flatten();

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
            y: Arc::clone(&y),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            quadrature_nodes: quad_nodes.clone(),
            quadrature_weights: quad_weights.clone(),
            score_warp: score_warp_runtime.clone(),
            score_warp_obs_design: score_warp_obs_design.clone(),
            cached_score_warp_constraints: cached_score_warp_constraints.clone(),
            link_dev: link_dev_runtime.clone(),
            cached_link_dev_constraints: cached_link_dev_constraints.clone(),
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
        crate::seeding::SeedRiskProfile::GeneralizedLinear,
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
            y: Arc::new(Array1::zeros(seed.len())),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
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
            cached_score_warp_constraints: None,
            link_dev: Some(prepared.runtime.clone()),
            cached_link_dev_constraints: BernoulliMarginalSlopeFamily::build_link_dev_constraints(
                &prepared.runtime,
            )
            .expect("build link dev constraints"),
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

        // Verify that the analytic IFT path produces finite gradient/Hessian
        // for a link-dev-only family.
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("eval cache");
        let row_ctx = family
            .build_row_exact_context(
                0,
                &block_states,
                &cache.h_nodes,
                cache.score_warp_obs.as_ref(),
                LinkOrder::Full,
            )
            .expect("row context");
        let (nll, grad, hess) = family
            .compute_row_analytic_flex(
                0,
                &block_states,
                &primary,
                &row_ctx,
                &cache.h_nodes,
                cache.h_node_design.as_ref(),
                cache.score_warp_obs.as_ref(),
                true,
            )
            .expect("analytic flex eval");
        assert!(nll.is_finite(), "neglog should be finite for link-dev-only");
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "gradient should be finite"
        );
        assert!(
            hess.iter().all(|v| v.is_finite()),
            "Hessian should be finite"
        );

        let dummy_spec = dummy_blockspec(link_dim, seed.len());
        let constraints = family
            .block_linear_constraints(&block_states, 2, &dummy_spec)
            .expect("constraint lookup")
            .expect("link block constraints");
        // Constraints come from the static dense knot-span grid, not an O(nQ)
        // state-dependent scan.
        let grid_points = prepared
            .runtime
            .dense_constraint_grid(4)
            .expect("dense constraint grid");
        assert_eq!(constraints.a.ncols(), link_dim);
        assert_eq!(constraints.a.nrows(), grid_points.len());
        assert_eq!(
            constraints.a,
            prepared
                .runtime
                .first_derivative_design(&grid_points)
                .expect("link derivative design"),
        );
        assert_eq!(
            constraints.b,
            Array1::from_elem(grid_points.len(), prepared.runtime.monotonicity_eps - 1.0),
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
            y: Arc::new(Array1::zeros(seed.len())),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            quadrature_nodes: array![-0.25, 0.25],
            quadrature_weights: array![0.5, 0.5],
            score_warp: Some(prepared.runtime.clone()),
            score_warp_obs_design: Some(prepared.block.design.clone()),
            cached_score_warp_constraints:
                BernoulliMarginalSlopeFamily::build_score_warp_constraints(
                    &prepared.runtime,
                    &seed,
                    &array![-0.25, 0.25],
                )
                .expect("cached score-warp constraints"),
            link_dev: None,
            cached_link_dev_constraints: None,
        };
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(Array1::zeros(score_dim), seed.len()),
        ];

        let dense = prepared
            .runtime
            .dense_constraint_grid(4)
            .expect("dense score-warp grid");
        let points = prepared
            .runtime
            .supported_unique_values_from_iter(
                seed.iter()
                    .copied()
                    .chain(family.quadrature_nodes.iter().copied())
                    .chain(dense.iter().copied()),
            )
            .expect("score-warp points");
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
    fn row_work_order_keeps_small_primary_models_second_order() {
        // Small n, no auxiliary primaries, K=2.
        // n=100_000 × k_pairs=3 × primary²=4 = 1_200_000 < 2M.
        assert_eq!(
            bernoulli_row_work_order(100_000, 0, 0, 2),
            ExactOuterDerivativeOrder::Second
        );
        // Moderate n with rich primaries, K=1.
        // n=4000 × k_pairs=1 × primary²=256 = 1_024_000 < 2M.
        assert_eq!(
            bernoulli_row_work_order(4_000, 8, 6, 1),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn row_work_order_downgrades_large_rich_models_to_first_order() {
        // n=10_000 × k_pairs=6 × primary²=256 = 15_360_000 > 2M.
        assert_eq!(
            bernoulli_row_work_order(10_000, 8, 6, 3),
            ExactOuterDerivativeOrder::First
        );
    }

    #[test]
    fn cost_gated_outer_order_downgrades_large_p() {
        use crate::custom_family::cost_gated_outer_order;
        use crate::matrix::DesignMatrix;
        use ndarray::Array2;

        // 2 blocks × 500 cols = p=1000, each block has 10 penalties → K=20,
        // K_pairs=210, psi_elements = 210 × 1000² = 210M > 200M limit.
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
            10, 500,
        ))));
        let specs: Vec<ParameterBlockSpec> = (0..2)
            .map(|i| ParameterBlockSpec {
                name: format!("block_{i}"),
                design: design.clone(),
                offset: ndarray::Array1::zeros(10),
                penalties: (0..10)
                    .map(|_| crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((500, 500))))
                    .collect(),
                nullspace_dims: vec![0; 10],
                initial_log_lambdas: ndarray::Array1::zeros(10),
                initial_beta: None,
            })
            .collect();
        assert_eq!(
            cost_gated_outer_order(&specs),
            ExactOuterDerivativeOrder::First
        );
    }

    #[test]
    fn rigid_fast_path_matches_loglik_finite_differences() {
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![1.0]),
            weights: Arc::new(array![1.2]),
            z: Arc::new(array![0.3]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            quadrature_nodes: array![0.0],
            quadrature_weights: array![1.0],
            score_warp: None,
            score_warp_obs_design: None,
            cached_score_warp_constraints: None,
            link_dev: None,
            cached_link_dev_constraints: None,
        };
        let states_at = |q: f64, g: f64| {
            vec![
                ParameterBlockState {
                    beta: array![q],
                    eta: array![q],
                },
                ParameterBlockState {
                    beta: array![g],
                    eta: array![g],
                },
            ]
        };
        let q = 0.4;
        let g = 0.7;
        let block_states = states_at(q, g);
        let eval = family
            .evaluate(&block_states)
            .expect("rigid family evaluation");
        let grad_q = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton marginal block")
            }
        };
        let grad_g = match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton log-slope block")
            }
        };
        let hess_qq = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { hessian, .. } => match hessian {
                SymmetricMatrix::Dense(h) => h[[0, 0]],
                _ => panic!("expected dense marginal Hessian"),
            },
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton marginal block")
            }
        };
        let hess_gg = match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { hessian, .. } => match hessian {
                SymmetricMatrix::Dense(h) => h[[0, 0]],
                _ => panic!("expected dense log-slope Hessian"),
            },
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton log-slope block")
            }
        };

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("rigid exact eval cache");
        let row_ctx = family
            .build_row_exact_context(
                0,
                &block_states,
                &cache.h_nodes,
                cache.score_warp_obs.as_ref(),
                LinkOrder::Full,
            )
            .expect("rigid row context");
        let (_, primary_grad, primary_hess) = family
            .compute_row_primary_gradient_hessian(
                0,
                &block_states,
                &cache.primary,
                &row_ctx,
                &cache.h_nodes,
                cache.h_node_design.as_ref(),
                cache.score_warp_obs.as_ref(),
            )
            .expect("rigid exact row derivatives");

        assert!(
            (grad_q + primary_grad[0]).abs() < 1e-10,
            "marginal gradient mismatch: fast={grad_q:.12e}, exact={:.12e}",
            -primary_grad[0]
        );
        assert!(
            (grad_g + primary_grad[1]).abs() < 1e-10,
            "logslope gradient mismatch: fast={grad_g:.12e}, exact={:.12e}",
            -primary_grad[1]
        );
        assert!(
            (hess_qq - primary_hess[[0, 0]]).abs() < 1e-10,
            "marginal Hessian mismatch: fast={hess_qq:.12e}, exact={:.12e}",
            primary_hess[[0, 0]]
        );
        assert!(
            (hess_gg - primary_hess[[1, 1]]).abs() < 1e-10,
            "logslope Hessian mismatch: fast={hess_gg:.12e}, exact={:.12e}",
            primary_hess[[1, 1]]
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
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
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
            cached_score_warp_constraints: None,
            link_dev: Some(prepared.runtime.clone()),
            cached_link_dev_constraints: BernoulliMarginalSlopeFamily::build_link_dev_constraints(
                &prepared.runtime,
            )
            .expect("build link dev constraints"),
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
                .build_row_exact_context(row, &block_states, h_nodes, None, LinkOrder::Full)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));

            let (_, grad, hess) = family
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
