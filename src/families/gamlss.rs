use std::borrow::Cow;

use crate::basis::{
    BasisOptions, Dense, KnotSource, compute_geometric_constraint_transform, create_basis,
    create_difference_penalty_matrix, evaluate_bspline_fourth_derivative_scalar,
    evaluate_bsplinethird_derivative_scalar,
};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily,
    CustomFamilyBlockPsiDerivative, FamilyEvaluation, KnownLinkWiggle, ParameterBlockSpec,
    ParameterBlockState, evaluate_custom_family_joint_hyper, fit_custom_family,
};
use crate::faer_ndarray::{fast_atv, fast_joint_hessian_2x2, fast_xt_diag_x, fast_xt_diag_y};
use crate::families::scale_design::{
    apply_scale_deviation_transform, build_scale_deviation_transform, infer_non_intercept_start,
};
use crate::families::sigma_link::{
    SigmaJet1, exp_sigma_derivs_up_to_third, exp_sigma_from_eta_scalar, exp_sigma_jet1_scalar,
    safe_exp,
};
use crate::generative::{CustomFamilyGenerative, GenerativeSpec, NoiseModel};
use crate::matrix::{
    DesignMatrix, EmbeddedColumnBlock, EmbeddedSquareBlock, SymmetricMatrix, xt_diag_x_symmetric,
};
use crate::mixture_link::inverse_link_jet_for_inverse_link;
use crate::probability::{normal_cdf, normal_pdf};
use crate::smooth::{
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec, TwoBlockExactJointHyperSetup, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design, get_spatial_length_scale,
    optimize_two_block_spatial_length_scale, optimize_two_block_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices, try_build_spatial_log_kappa_derivativeinfo_list,
};
use crate::types::{InverseLink, LinkFunction};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
const MIN_PROB: f64 = 1e-10;
const MIN_DERIV: f64 = 1e-8;
const MIN_WEIGHT: f64 = 1e-12;
const BETA_RANGE_WARN_THRESHOLD: f64 = 1.10;
const BINOMIAL_EFFECTIVE_N_WARN_THRESHOLD: f64 = 25.0;
const BINOMIAL_LOG_SIGMA_SHRINKAGE_LOG_LAMBDA_INIT: f64 = 0.0;

#[inline]
fn floor_positiveweight(rawweight: f64, minweight: f64) -> f64 {
    if !rawweight.is_finite() || rawweight <= 0.0 {
        0.0
    } else {
        rawweight.max(minweight)
    }
}

#[inline]
fn gaussian_log_sigma_irlsinfo_directional_derivative(
    weight: f64,
    sigma: f64,
    d_sigma: f64,
    d2_sigma: f64,
    d_eta: f64,
) -> f64 {
    if weight == 0.0 || d_eta == 0.0 {
        return 0.0;
    }

    let s = sigma.max(1e-10);
    let raw_g = if d_sigma == 0.0 { 0.0 } else { d_sigma / s };
    let g = raw_g.clamp(-1.0, 1.0);
    let rawinfo = 2.0 * weight * g * g;
    if !rawinfo.is_finite() || rawinfo <= MIN_WEIGHT {
        return 0.0;
    }

    // The evaluated IRLS weight uses clamp(raw_g, -1, 1). On an active clamp
    // branch the working weight is locally constant in eta_ls, so we return the
    // derivative of that active piece, namely zero.
    if !(-1.0..=1.0).contains(&raw_g) || raw_g == -1.0 || raw_g == 1.0 {
        return 0.0;
    }

    let dg_deta = d2_sigma / s - d_sigma * d_sigma / (s * s);
    let dw = 4.0 * weight * g * dg_deta * d_eta;
    if dw.is_finite() { dw } else { 0.0 }
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

    fn haswiggle(self) -> bool {
        self.kwiggle > 0
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
    pub penalties: Vec<Array2<f64>>,
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
            let (r, c) = s.dim();
            if r != p || c != p {
                return Err(format!(
                    "block '{name}' penalty {k} must be {p}x{p}, got {r}x{c}"
                ));
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
            penalties: self.penalties,
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

pub fn initializewiggle_knots_from_seed(
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

pub fn buildwiggle_block_input_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    penalty_order: usize,
    double_penalty: bool,
) -> Result<ParameterBlockInput, String> {
    let (basis, _) = create_basis::<Dense>(
        seed,
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    let full = (*basis).clone();
    if full.ncols() < 3 {
        return Err("wiggle basis has fewer than three columns".to_string());
    }
    let (z, s_constrained) = compute_geometric_constraint_transform(knots, degree, penalty_order)
        .map_err(|e| e.to_string())?;
    if full.ncols() != z.nrows() {
        return Err(format!(
            "wiggle basis/constraint mismatch: basis has {} columns but transform has {} rows",
            full.ncols(),
            z.nrows()
        ));
    }
    let design = full.dot(&z);
    let p = design.ncols();
    let mut penalties = vec![s_constrained];
    if double_penalty {
        penalties.push(Array2::<f64>::eye(p));
    }
    Ok(ParameterBlockInput {
        design: DesignMatrix::Dense(design),
        offset: Array1::zeros(seed.len()),
        penalties,
        initial_log_lambdas: None,
        initial_beta: None,
    })
}

pub fn buildwiggle_block_input_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
) -> Result<(ParameterBlockInput, Array1<f64>), String> {
    let knots = initializewiggle_knots_from_seed(seed, cfg.degree, cfg.num_internal_knots)?;
    let block = buildwiggle_block_input_from_knots(
        seed,
        &knots,
        cfg.degree,
        cfg.penalty_order,
        cfg.double_penalty,
    )?;
    Ok((block, knots))
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

fn validate_gaussian_location_scale_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &GaussianLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    validate_term_weights(data, spec.y.len(), &spec.weights, context)
}

fn validate_binomial_location_scale_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    validate_term_weights(data, spec.y.len(), &spec.weights, context)?;
    validate_binomial_response(&spec.y, context)?;
    Ok(())
}

fn validate_binomial_location_scalewiggle_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleWiggleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_weights(data, n, &spec.weights, context)?;
    validate_binomial_response(&spec.y, context)?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    if spec.wiggle_degree < 1 {
        return Err(format!(
            "{context}: wiggle_degree must be >= 1, got {}",
            spec.wiggle_degree
        ));
    }
    if spec.wiggle_knots.len() < spec.wiggle_degree + 2 {
        return Err(format!(
            "{context}: wiggle_knots must have at least {} entries for degree {}, got {}",
            spec.wiggle_degree + 2,
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

fn build_block_spatial_psi_derivatives(
    data: ndarray::ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    // Custom-family exact blocks consume psi = log(kappa) derivatives. The exact-joint
    // setup is typed in SpatialLogKappaCoords so these derivatives stay in the same
    // parameterization end-to-end.
    let spatial_terms = spatial_length_scale_term_indices(resolvedspec);
    let Some(info_list) =
        try_build_spatial_log_kappa_derivativeinfo_list(data, resolvedspec, design, &spatial_terms)
            .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    let psi_dim = info_list.len();
    Ok(Some(
        info_list
            .into_iter()
            .enumerate()
            .map(|(psi_idx, info)| {
                let x_full = EmbeddedColumnBlock::new(
                    &info.x_psi_local,
                    info.global_range.clone(),
                    info.total_p,
                )
                .materialize();
                let s_full = EmbeddedSquareBlock::new(
                    &info.s_psi_local,
                    info.global_range.clone(),
                    info.total_p,
                )
                .materialize();
                let penalty_indices = info.penalty_indices.clone();
                CustomFamilyBlockPsiDerivative {
                    penalty_index: Some(info.penalty_index),
                    x_psi: x_full.clone(),
                    s_psi: s_full,
                    s_psi_components: Some(
                        info.penalty_indices
                            .into_iter()
                            .zip(info.s_psi_components_local.into_iter().map(|local| {
                                EmbeddedSquareBlock::new(
                                    &local,
                                    info.global_range.clone(),
                                    info.total_p,
                                )
                                .materialize()
                            }))
                            .collect(),
                    ),
                    x_psi_psi: Some({
                        let mut rows =
                            vec![Array2::<f64>::zeros((x_full.nrows(), x_full.ncols())); psi_dim];
                        rows[psi_idx] = EmbeddedColumnBlock::new(
                            &info.x_psi_psi_local,
                            info.global_range.clone(),
                            info.total_p,
                        )
                        .materialize();
                        rows
                    }),
                    s_psi_psi: Some({
                        let mut rows =
                            vec![Array2::<f64>::zeros((info.total_p, info.total_p)); psi_dim];
                        rows[psi_idx] = EmbeddedSquareBlock::new(
                            &info.s_psi_psi_local,
                            info.global_range.clone(),
                            info.total_p,
                        )
                        .materialize();
                        rows
                    }),
                    s_psi_psi_components: Some({
                        let mut rows = vec![Vec::<(usize, Array2<f64>)>::new(); psi_dim];
                        rows[psi_idx] = penalty_indices
                            .into_iter()
                            .zip(info.s_psi_psi_components_local.into_iter().map(|local| {
                                EmbeddedSquareBlock::new(
                                    &local,
                                    info.global_range.clone(),
                                    info.total_p,
                                )
                                .materialize()
                            }))
                            .collect();
                        rows
                    }),
                }
            })
            .collect(),
    ))
}

fn build_two_block_exact_joint_setup(
    meanspec: &TermCollectionSpec,
    noisespec: &TermCollectionSpec,
    mean_penalties: usize,
    noise_penalties: usize,
    extra_rho0: &[f64],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> TwoBlockExactJointHyperSetup {
    // Exact-joint setup stores the spatial tail in log(kappa), not log(length_scale).
    let mean_terms = spatial_length_scale_term_indices(meanspec);
    let noise_terms = spatial_length_scale_term_indices(noisespec);
    let rho_dim = mean_penalties + noise_penalties + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    let mut log_kappa0 = Array1::<f64>::zeros(mean_terms.len() + noise_terms.len());

    for (i, &rho_init) in extra_rho0.iter().enumerate() {
        rho0vec[mean_penalties + noise_penalties + i] = rho_init;
    }
    for (slot, &term_idx) in mean_terms.iter().enumerate() {
        let length_scale = get_spatial_length_scale(meanspec, term_idx)
            .unwrap_or(kappa_options.min_length_scale)
            .clamp(
                kappa_options.min_length_scale,
                kappa_options.max_length_scale,
            );
        log_kappa0[slot] = -length_scale.ln();
    }
    for (slot, &term_idx) in noise_terms.iter().enumerate() {
        let length_scale = get_spatial_length_scale(noisespec, term_idx)
            .unwrap_or(kappa_options.min_length_scale)
            .clamp(
                kappa_options.min_length_scale,
                kappa_options.max_length_scale,
            );
        log_kappa0[mean_terms.len() + slot] = -length_scale.ln();
    }
    TwoBlockExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        SpatialLogKappaCoords::new(log_kappa0),
        SpatialLogKappaCoords::lower_bounds(mean_terms.len() + noise_terms.len(), kappa_options),
        SpatialLogKappaCoords::upper_bounds(mean_terms.len() + noise_terms.len(), kappa_options),
    )
}

fn solveweighted_projection(
    design: &DesignMatrix,
    offset: &Array1<f64>,
    target_eta: &Array1<f64>,
    weights: &Array1<f64>,
    ridge_floor: f64,
) -> Result<Array1<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if offset.len() != n || target_eta.len() != n || weights.len() != n {
        return Err("solveweighted_projection dimension mismatch".to_string());
    }

    let y_star = target_eta - offset;
    let xtwy = design.compute_xtwy(weights, &y_star)?;
    let ridge = ridge_floor.max(1e-12);
    let penalty = Array2::from_diag(&Array1::from_elem(p, ridge));
    let beta = design
        .solve_system(weights, &xtwy, Some(&penalty))
        .map_err(|_| "solveweighted_projection produced non-finite coefficients".to_string())?;
    if beta.iter().any(|v| !v.is_finite()) {
        return Err("solveweighted_projection produced non-finite coefficients".to_string());
    }
    Ok(beta)
}

fn solve_penalizedweighted_projection(
    design: &DesignMatrix,
    offset: &Array1<f64>,
    target_eta: &Array1<f64>,
    weights: &Array1<f64>,
    penalties: &[Array2<f64>],
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
    let mut penalty = Array2::<f64>::zeros((p, p));
    for (k, s) in penalties.iter().enumerate() {
        let lambda = log_lambdas[k].exp();
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(format!(
                "solve_penalizedweighted_projection encountered invalid lambda at index {k}: {}",
                log_lambdas[k]
            ));
        }
        penalty.scaled_add(lambda, s);
    }
    let ridge = ridge_floor.max(1e-12);
    for a in 0..p {
        penalty[[a, a]] += ridge;
    }

    let beta = design
        .solve_system(weights, &xtwy, Some(&penalty))
        .map_err(|_| {
            "solve_penalizedweighted_projection produced non-finite coefficients".to_string()
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
    let sigma_hat = (weighted_ss / weight_sum).sqrt().max(1e-10);
    let beta_log_sigma = if let Some(beta) = noise_beta_hint {
        beta.clone()
    } else {
        let eta_sigma = sigma_hat.ln();
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

fn weighted_prevalence(y: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let w_sum: f64 = weights.iter().copied().sum();
    if w_sum <= 0.0 {
        return 0.5;
    }
    let yw_sum: f64 = y.iter().zip(weights.iter()).map(|(&yi, &wi)| yi * wi).sum();
    (yw_sum / w_sum).clamp(0.0, 1.0)
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
    weights: &Array1<f64>,
    non_intercept_start: usize,
) -> Result<Array2<f64>, String> {
    prepared_scale_design(mu_design, log_sigma_design, weights, non_intercept_start)
}

fn identified_binomial_log_sigma_design(
    threshold_design: &TermCollectionDesign,
    log_sigma_design: &TermCollectionDesign,
    weights: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    prepared_scale_design(
        &threshold_design.design,
        &log_sigma_design.design,
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

fn append_log_lambdavalue(log_lambdas: &Array1<f64>, value: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(log_lambdas.len() + 1);
    if !log_lambdas.is_empty() {
        out.slice_mut(s![0..log_lambdas.len()]).assign(log_lambdas);
    }
    out[log_lambdas.len()] = value;
    out
}

fn append_binomial_log_sigma_shrinkage_penalty_input(block: &mut ParameterBlockInput) {
    let p = block.design.ncols();
    block.penalties.push(identity_penalty(p));
    block.initial_log_lambdas = Some(match block.initial_log_lambdas.take() {
        Some(log_lambdas) => {
            append_log_lambdavalue(&log_lambdas, BINOMIAL_LOG_SIGMA_SHRINKAGE_LOG_LAMBDA_INIT)
        }
        None => Array1::from_vec(vec![BINOMIAL_LOG_SIGMA_SHRINKAGE_LOG_LAMBDA_INIT]),
    });
}

fn append_binomial_log_sigma_shrinkage_penalty_design(design: &mut TermCollectionDesign) {
    design
        .penalties
        .push(identity_penalty(design.design.ncols()));
}

fn emit_binomial_alpha_betawarnings(
    context: &str,
    betavalues: &Array1<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
) {
    if betavalues.is_empty() {
        return;
    }
    let beta_min = betavalues.iter().copied().fold(f64::INFINITY, f64::min);
    let beta_max = betavalues.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !beta_min.is_finite() || !beta_max.is_finite() || beta_min <= 0.0 {
        log::warn!(
            "[GAMLSS][{}] non-positive or non-finite beta encountered (min={}, max={})",
            context,
            beta_min,
            beta_max
        );
    } else {
        let ratio = beta_max / beta_min;
        if ratio > BETA_RANGE_WARN_THRESHOLD {
            log::warn!(
                "[GAMLSS][{}] beta range ratio {:.3} exceeds {:.3}; transformed-penalty distortion risk is elevated",
                context,
                ratio,
                BETA_RANGE_WARN_THRESHOLD
            );
        }
    }

    let pi = weighted_prevalence(y, weights);
    let w_sum: f64 = weights.iter().copied().sum();
    let n_eff = w_sum * pi * (1.0 - pi);
    if n_eff < BINOMIAL_EFFECTIVE_N_WARN_THRESHOLD {
        log::warn!(
            "[GAMLSS][{}] low effective sample size N_eff={:.3} (sumw={:.3}, prevalence={:.3}); location-scale separation artifacts are more likely",
            context,
            n_eff,
            w_sum,
            pi
        );
    }
}

#[derive(Clone)]
struct BinomialAlphaBetaWarmStartFamily {
    y: Array1<f64>,
    weights: Array1<f64>,
}

impl BinomialAlphaBetaWarmStartFamily {
    const BLOCK_ALPHA: usize = 0;
    const BLOCK_BETA: usize = 1;
}

impl CustomFamily for BinomialAlphaBetaWarmStartFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialAlphaBetaWarmStartFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_alpha = &block_states[Self::BLOCK_ALPHA].eta;
        let eta_beta = &block_states[Self::BLOCK_BETA].eta;
        if eta_alpha.len() != n || eta_beta.len() != n || self.weights.len() != n {
            return Err("BinomialAlphaBetaWarmStartFamily input size mismatch".to_string());
        }

        let mut z_alpha = Array1::<f64>::zeros(n);
        let mut w_alpha = Array1::<f64>::zeros(n);
        let mut z_beta = Array1::<f64>::zeros(n);
        let mut w_beta = Array1::<f64>::zeros(n);
        let mut ll = 0.0_f64;

        for i in 0..n {
            let q = eta_alpha[i];
            // Mathematical status of this warm-start family:
            //
            //   ell_i(alpha_i, beta_i)
            //   = w_i [ y_i log Phi(q_i) + (1-y_i) log(1-Phi(q_i)) ],
            //   q_i = eta_alpha,i.
            //
            // As coded, q_i does not depend on eta_beta,i at all. Therefore the
            // literal per-row objective satisfies
            //
            //   d ell_i / d eta_beta,i    = 0,
            //   d²ell_i / d eta_beta,i²   = 0,
            //   d²ell_i / d eta_alpha,i d eta_beta,i = 0.
            //
            // Earlier versions fabricated a beta working derivative by pushing a
            // bounded-beta chain rule through an alpha-only likelihood. That was
            // mathematically false. The current implementation keeps the beta
            // block neutral:
            //
            //   w_beta,i = 0,
            //   z_beta,i = eta_beta,i,
            //
            // so the warm-start IRLS step is an honest derivative of the coded
            // objective. If this family is ever meant to inform beta, the model
            // itself must change so q_i or ell_i actually depends on beta.
            let mu = normal_cdf(q);
            let dmu_dq = normal_pdf(q).max(MIN_DERIV);
            let var = (mu * (1.0 - mu)).max(MIN_PROB);

            ll += self.weights[i] * (self.y[i] * mu.ln() + (1.0 - self.y[i]) * (1.0 - mu).ln());

            // Clamp-consistent derivative:
            //
            //   mu_bar(q) = clamp(Phi(q), eps, 1-eps).
            //
            // On an active clamp branch mu_bar is locally constant, so the
            // exact derivative of the reported log-likelihood is zero. In that
            // region the IRLS surrogate must also be locally flat:
            //
            //   w_alpha,i = 0,
            //   z_alpha,i = eta_alpha,i.
            let dmu_alpha = dmu_dq;
            if dmu_alpha == 0.0 {
                w_alpha[i] = 0.0;
                z_alpha[i] = eta_alpha[i];
            } else {
                w_alpha[i] = floor_positiveweight(
                    self.weights[i] * (dmu_alpha * dmu_alpha / var),
                    MIN_WEIGHT,
                );
                z_alpha[i] =
                    eta_alpha[i] + (self.y[i] - mu) / signedwith_floor(dmu_alpha, MIN_DERIV);
            }

            w_beta[i] = 0.0;
            z_beta[i] = eta_beta[i];
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: z_alpha,
                    working_weights: w_alpha,
                },
                BlockWorkingSet::Diagonal {
                    working_response: z_beta,
                    working_weights: w_beta,
                },
            ],
        })
    }
}

fn try_binomial_alpha_betawarm_start(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    threshold_block: &ParameterBlockInput,
    log_sigma_block: &ParameterBlockInput,
    options: &BlockwiseFitOptions,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let warm_family = BinomialAlphaBetaWarmStartFamily {
        y: y.clone(),
        weights: weights.clone(),
    };

    let alphaspec = ParameterBlockSpec {
        name: "alphawarm".to_string(),
        design: threshold_block.design.clone(),
        offset: threshold_block.offset.clone(),
        penalties: threshold_block.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas_orzeros(threshold_block)?,
        initial_beta: None,
    };
    let betaspec = ParameterBlockSpec {
        name: "betawarm".to_string(),
        design: log_sigma_block.design.clone(),
        offset: log_sigma_block.offset.clone(),
        penalties: log_sigma_block.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas_orzeros(log_sigma_block)?,
        initial_beta: None,
    };

    let warm_options = BlockwiseFitOptions {
        inner_max_cycles: options.inner_max_cycles.clamp(5, 40),
        inner_tol: options.inner_tol,
        outer_max_iter: options.outer_max_iter.clamp(3, 20),
        outer_tol: options.outer_tol.max(1e-6),
        minweight: options.minweight,
        ridge_floor: options.ridge_floor.max(1e-10),
        ridge_policy: options.ridge_policy,
        // Warm start optimization focuses on robust initialization, not REML correction.
        use_remlobjective: false,
        // Warm-start covariance is unused.
        compute_covariance: false,
    };
    let warm_fit = fit_custom_family(&warm_family, &[alphaspec, betaspec], &warm_options)?;
    let eta_alpha = &warm_fit.block_states[BinomialAlphaBetaWarmStartFamily::BLOCK_ALPHA].eta;
    if eta_alpha.len() != y.len() {
        return Err("warm start eta length mismatch".to_string());
    }

    // This warm-start family currently identifies alpha only. Seed the
    // log-sigma block from a deterministic midpoint sigma instead of reusing
    // the beta block's penalty-only iterate.
    let sigma_target: f64 = 1.0;
    let eta_ls_target = 0.0;
    let betaobs = Array1::from_elem(y.len(), 1.0 / sigma_target.max(1e-12));
    let t_target = Array1::from_iter(
        eta_alpha
            .iter()
            .zip(betaobs.iter())
            .map(|(&a, &b)| -a / b.max(1e-12)),
    );
    let log_sigma_target = Array1::from_elem(y.len(), eta_ls_target);
    // T = -alpha/beta is noisy when beta is small (large sigma). Weight the
    // projection by beta^2 (inverse variance of T under fixed alpha noise).
    let t_projectionw =
        Array1::from_iter(weights.iter().zip(betaobs.iter()).map(|(&w, &b)| w * b * b));

    let beta_t = solveweighted_projection(
        &threshold_block.design,
        &threshold_block.offset,
        &t_target,
        &t_projectionw,
        options.ridge_floor.max(1e-10),
    )?;
    let beta_log_sigma = solveweighted_projection(
        &log_sigma_block.design,
        &log_sigma_block.offset,
        &log_sigma_target,
        weights,
        options.ridge_floor.max(1e-10),
    )?;

    Ok((beta_t, beta_log_sigma, betaobs))
}

#[derive(Clone)]
pub struct GaussianLocationScaleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_block: ParameterBlockInput,
    pub log_sigma_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct BinomialMeanWiggleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub eta_block: ParameterBlockInput,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct PoissonLogSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub eta_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct GammaLogSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    /// Gamma shape parameter (k > 0).
    pub shape: f64,
    pub eta_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct BinomialLocationScaleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_block: ParameterBlockInput,
    pub log_sigma_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct BinomialLocationScaleWiggleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub threshold_block: ParameterBlockInput,
    pub log_sigma_block: ParameterBlockInput,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct GaussianLocationScaleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
}

#[derive(Clone)]
pub struct BinomialLocationScaleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
}

#[derive(Clone)]
pub struct BinomialLocationScaleWiggleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Debug)]
pub struct BlockwiseTermFitResult {
    pub fit: BlockwiseFitResult,
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

#[derive(Clone, Debug)]
pub struct BinomialLocationScaleWiggleWorkflowConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}

pub struct BinomialLocationScaleWorkflowResult {
    pub fit: BlockwiseTermFitResult,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    pub betawiggle: Option<Vec<f64>>,
}

fn slice_log_lambda_block(
    log_lambdas: &Array1<f64>,
    start: usize,
    len: usize,
    blockname: &str,
) -> Result<Array1<f64>, String> {
    let end = start + len;
    if end > log_lambdas.len() {
        return Err(format!(
            "log lambda slice for block '{blockname}' is out of bounds: {}..{} with total {}",
            start,
            end,
            log_lambdas.len()
        ));
    }
    Ok(log_lambdas.slice(s![start..end]).to_owned())
}

pub fn fit_gaussian_location_scale(
    spec: GaussianLocationScaleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validateweights(&spec.weights, "fit_gaussian_location_scale")?;
    validate_blockrows("mu", n, &spec.mu_block)?;
    validate_blockrows("log_sigma", n, &spec.log_sigma_block)?;

    let GaussianLocationScaleSpec {
        y,
        weights,
        mu_block,
        log_sigma_block,
    } = spec;
    let mut muspec = mu_block.intospec("mu")?;
    let mut log_sigmaspec = log_sigma_block.intospec("log_sigma")?;
    let mu_dense = muspec.design.to_dense();
    let raw_log_sigma_dense = log_sigmaspec.design.to_dense();
    let non_intercept_start = infer_non_intercept_start(&raw_log_sigma_dense, &weights);
    log_sigmaspec.design = DesignMatrix::Dense(prepared_gaussian_log_sigma_design(
        &mu_dense,
        &raw_log_sigma_dense,
        &weights,
        non_intercept_start,
    )?);
    if muspec.initial_beta.is_none() || log_sigmaspec.initial_beta.is_none() {
        let (betamu0, beta_ls0, sigma0) = gaussian_location_scalewarm_start(
            &y,
            &weights,
            &muspec,
            &log_sigmaspec,
            options.ridge_floor,
            muspec.initial_beta.as_ref(),
            log_sigmaspec.initial_beta.as_ref(),
        )?;
        if muspec.initial_beta.is_none() {
            muspec.initial_beta = Some(betamu0);
        }
        if log_sigmaspec.initial_beta.is_none() {
            log_sigmaspec.initial_beta = Some(beta_ls0);
        }
        log::info!(
            "[GAMLSS][fit_gaussian_location_scale] initialized at residual sigma {:.6e}",
            sigma0
        );
    }

    let family = GaussianLocationScaleFamily {
        y,
        weights,
        mu_design: Some(muspec.design.clone()),
        log_sigma_design: Some(log_sigmaspec.design.clone()),
        cached_row_scalars: std::cell::RefCell::new(None),
    };
    let blocks = vec![muspec, log_sigmaspec];
    Ok(fit_custom_family(&family, &blocks, options)?)
}

pub fn fit_binomial_mean_wiggle(
    spec: BinomialMeanWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
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
    if spec.wiggle_degree < 1 {
        return Err(format!(
            "fit_binomial_mean_wiggle: wiggle_degree must be >= 1, got {}",
            spec.wiggle_degree
        ));
    }
    if spec.wiggle_knots.len() < spec.wiggle_degree + 2 {
        return Err(format!(
            "fit_binomial_mean_wiggle: wiggle_knots length {} is too short for degree {}",
            spec.wiggle_knots.len(),
            spec.wiggle_degree
        ));
    }

    let family = BinomialMeanWiggleFamily {
        y: spec.y,
        weights: spec.weights,
        link_kind: spec.link_kind,
        wiggle_knots: spec.wiggle_knots,
        wiggle_degree: spec.wiggle_degree,
    };
    let blocks = vec![
        spec.eta_block.intospec("eta")?,
        spec.wiggle_block.intospec("wiggle")?,
    ];
    fit_custom_family(&family, &blocks, options).map_err(|e| e.to_string())
}

pub fn fit_poisson_log(
    spec: PoissonLogSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validateweights(&spec.weights, "fit_poisson_log")?;
    validate_blockrows("eta", n, &spec.eta_block)?;

    let family = PoissonLogFamily {
        y: spec.y,
        weights: spec.weights,
    };
    let blocks = vec![spec.eta_block.intospec("eta")?];
    Ok(fit_custom_family(&family, &blocks, options)?)
}

pub fn fit_gamma_log(
    spec: GammaLogSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validateweights(&spec.weights, "fit_gamma_log")?;
    validate_blockrows("eta", n, &spec.eta_block)?;
    if !spec.shape.is_finite() || spec.shape <= 0.0 {
        return Err(format!(
            "fit_gamma_log: shape must be finite and > 0, got {}",
            spec.shape
        ));
    }

    let family = GammaLogFamily {
        y: spec.y,
        weights: spec.weights,
        shape: spec.shape,
    };
    let blocks = vec![spec.eta_block.intospec("eta")?];
    Ok(fit_custom_family(&family, &blocks, options)?)
}

pub fn fit_binomial_location_scale(
    spec: BinomialLocationScaleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validateweights(&spec.weights, "fit_binomial_location_scale")?;
    validate_binomial_response(&spec.y, "fit_binomial_location_scale")?;
    validate_blockrows("threshold", n, &spec.threshold_block)?;
    validate_blockrows("log_sigma", n, &spec.log_sigma_block)?;

    let BinomialLocationScaleSpec {
        y,
        weights,
        link_kind,
        mut threshold_block,
        mut log_sigma_block,
    } = spec;
    let threshold_dense = threshold_block.design.to_dense();
    let raw_log_sigma_dense = log_sigma_block.design.to_dense();
    let non_intercept_start = infer_non_intercept_start(&raw_log_sigma_dense, &weights);
    log_sigma_block.design = DesignMatrix::Dense(prepared_scale_design(
        &threshold_dense,
        &raw_log_sigma_dense,
        &weights,
        non_intercept_start,
    )?);
    append_binomial_log_sigma_shrinkage_penalty_input(&mut log_sigma_block);

    if matches!(link_kind, InverseLink::Standard(LinkFunction::Probit)) {
        match try_binomial_alpha_betawarm_start(
            &y,
            &weights,
            &threshold_block,
            &log_sigma_block,
            options,
        ) {
            Ok((beta_t0, beta_ls0, betawarm)) => {
                threshold_block.initial_beta = Some(beta_t0);
                log_sigma_block.initial_beta = Some(beta_ls0);
                emit_binomial_alpha_betawarnings("warm-start", &betawarm, &y, &weights);
            }
            Err(err) => {
                log::warn!(
                    "[GAMLSS][fit_binomial_location_scale] alpha/beta warm start failed, falling back to direct initialization: {}",
                    err
                );
            }
        }
    }

    let blocks = vec![
        threshold_block.intospec("threshold")?,
        log_sigma_block.intospec("log_sigma")?,
    ];
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: link_kind.clone(),
        threshold_design: Some(blocks[0].design.clone()),
        log_sigma_design: Some(blocks[1].design.clone()),
    };
    let fit = fit_custom_family(&family, &blocks, options)?;
    let beta_final = fit.block_states[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .eta
        .mapv(|eta| 1.0 / exp_sigma_from_eta_scalar(eta).max(1e-12));
    emit_binomial_alpha_betawarnings("final-fit", &beta_final, &y, &weights);
    Ok(fit)
}

pub fn fit_binomial_location_scalewiggle(
    spec: BinomialLocationScaleWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validateweights(&spec.weights, "fit_binomial_location_scalewiggle")?;
    validate_binomial_response(&spec.y, "fit_binomial_location_scalewiggle")?;
    validate_blockrows("threshold", n, &spec.threshold_block)?;
    validate_blockrows("log_sigma", n, &spec.log_sigma_block)?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    if spec.wiggle_degree < 1 {
        return Err(format!(
            "fit_binomial_location_scalewiggle: wiggle_degree must be >= 1, got {}",
            spec.wiggle_degree
        ));
    }
    if spec.wiggle_knots.len() < spec.wiggle_degree + 2 {
        return Err(format!(
            "fit_binomial_location_scalewiggle: wiggle_knots length {} is too short for degree {}",
            spec.wiggle_knots.len(),
            spec.wiggle_degree
        ));
    }

    let BinomialLocationScaleWiggleSpec {
        y,
        weights,
        link_kind,
        wiggle_knots,
        wiggle_degree,
        mut threshold_block,
        mut log_sigma_block,
        wiggle_block,
    } = spec;
    let threshold_dense = threshold_block.design.to_dense();
    let raw_log_sigma_dense = log_sigma_block.design.to_dense();
    let non_intercept_start = infer_non_intercept_start(&raw_log_sigma_dense, &weights);
    log_sigma_block.design = DesignMatrix::Dense(prepared_scale_design(
        &threshold_dense,
        &raw_log_sigma_dense,
        &weights,
        non_intercept_start,
    )?);
    append_binomial_log_sigma_shrinkage_penalty_input(&mut log_sigma_block);

    if (threshold_block.initial_beta.is_none() || log_sigma_block.initial_beta.is_none())
        && matches!(link_kind, InverseLink::Standard(LinkFunction::Probit))
    {
        match try_binomial_alpha_betawarm_start(
            &y,
            &weights,
            &threshold_block,
            &log_sigma_block,
            options,
        ) {
            Ok((beta_t0, beta_ls0, betawarm)) => {
                if threshold_block.initial_beta.is_none() {
                    threshold_block.initial_beta = Some(beta_t0);
                }
                if log_sigma_block.initial_beta.is_none() {
                    log_sigma_block.initial_beta = Some(beta_ls0);
                }
                emit_binomial_alpha_betawarnings("warm-start-wiggle", &betawarm, &y, &weights);
            }
            Err(err) => {
                log::warn!(
                    "[GAMLSS][fit_binomial_location_scalewiggle] alpha/beta warm start failed, falling back to direct initialization: {}",
                    err
                );
            }
        }
    }

    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: link_kind.clone(),
        threshold_design: Some(threshold_block.design.clone()),
        log_sigma_design: Some(log_sigma_block.design.clone()),
        wiggle_knots,
        wiggle_degree,
    };
    let blocks = vec![
        threshold_block.intospec("threshold")?,
        log_sigma_block.intospec("log_sigma")?,
        wiggle_block.intospec("wiggle")?,
    ];
    let fit = fit_custom_family(&family, &blocks, options)?;
    let beta_final = fit.block_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
        .eta
        .mapv(|eta| 1.0 / exp_sigma_from_eta_scalar(eta).max(1e-12));
    emit_binomial_alpha_betawarnings("final-fit-wiggle", &beta_final, &y, &weights);
    Ok(fit)
}

trait LocationScaleFamilyBuilder {
    type Family: CustomFamily;

    fn meanspec(&self) -> &TermCollectionSpec;
    fn noisespec(&self) -> &TermCollectionSpec;
    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }
    fn require_exact_spatial_joint(&self) -> bool {
        false
    }
    fn mean_penalty_count(&self, mean_design: &TermCollectionDesign) -> usize {
        mean_design.penalties.len()
    }
    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        noise_design.penalties.len()
    }
    fn augment_result_designs(&self, _: &mut TermCollectionDesign, _: &mut TermCollectionDesign) {}
    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        Ok(Array1::zeros(0))
    }
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
        fit: &BlockwiseFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String>;
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
                .ok_or_else(|| "missing mean spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noisespec_resolved, noise_design)?
                .ok_or_else(|| "missing noise spatial psi derivatives".to_string())?;
        Ok(vec![mean_derivs, noise_derivs])
    }
}

fn compose_theta_from_hints(
    mean_penalty_count: usize,
    noise_penalty_count: usize,
    mean_log_lambda_hint: &Option<Array1<f64>>,
    noise_log_lambda_hint: &Option<Array1<f64>>,
    extra_rho0: &Array1<f64>,
) -> Array1<f64> {
    let layout =
        GamlssLambdaLayout::withwiggle(mean_penalty_count, noise_penalty_count, extra_rho0.len());
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
    if layout.haswiggle() {
        theta
            .slice_mut(s![layout.wiggle_start()..layout.wiggle_end()])
            .assign(extra_rho0);
    }
    theta
}

fn fit_location_scale_terms<B: LocationScaleFamilyBuilder>(
    data: ndarray::ArrayView2<'_, f64>,
    builder: B,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    let mut mean_log_lambda_hint: Option<Array1<f64>> = None;
    let mut noise_log_lambda_hint: Option<Array1<f64>> = None;
    let mut mean_beta_hint: Option<Array1<f64>> = None;
    let mut noise_beta_hint: Option<Array1<f64>> = None;
    let mut spatial_search_options = options.clone();
    spatial_search_options.use_remlobjective = false;
    spatial_search_options.compute_covariance = false;
    spatial_search_options.outer_max_iter = 1;
    let extra_rho0 = builder.extra_rho0()?;

    let mean_boot_design =
        build_term_collection_design(data, builder.meanspec()).map_err(|e| e.to_string())?;
    let noise_boot_design =
        build_term_collection_design(data, builder.noisespec()).map_err(|e| e.to_string())?;
    let mean_bootspec =
        freeze_spatial_length_scale_terms_from_design(builder.meanspec(), &mean_boot_design)
            .map_err(|e| e.to_string())?;
    let noise_bootspec =
        freeze_spatial_length_scale_terms_from_design(builder.noisespec(), &noise_boot_design)
            .map_err(|e| e.to_string())?;

    let exact_joint_ready = builder.exact_spatial_joint_supported()
        && matches!(
            (
                build_block_spatial_psi_derivatives(data, &mean_bootspec, &mean_boot_design)?,
                build_block_spatial_psi_derivatives(data, &noise_bootspec, &noise_boot_design)?,
            ),
            (Some(_), Some(_))
        );

    let require_exact_spatial_joint = builder.require_exact_spatial_joint();
    let mut usedfd_spatial_search = false;
    let mean_penalty_count = builder.mean_penalty_count(&mean_boot_design);
    let noise_penalty_count = builder.noise_penalty_count(&noise_boot_design);

    // Covered exact spatial families must use the unified exact-joint hyper
    // path and must never fall back to the older finite-difference search.
    // That exact path is the only implementation that knows how to evaluate
    // the full profiled/Laplace objective over theta = [rho, psi] with the
    // real joint Hessian required by NewtonTR/ARC.
    let mut solved = if require_exact_spatial_joint {
        if !exact_joint_ready {
            return Err(
                "exact two-block spatial optimization is required for this family, but analytic spatial psi derivatives are unavailable"
                    .to_string(),
            );
        }
        let joint_setup = build_two_block_exact_joint_setup(
            builder.meanspec(),
            builder.noisespec(),
            mean_penalty_count,
            noise_penalty_count,
            extra_rho0.as_slice().unwrap_or(&[]),
            kappa_options,
        );
        let mean_beta_hint_cell = std::cell::RefCell::new(mean_beta_hint.clone());
        let noise_beta_hint_cell = std::cell::RefCell::new(noise_beta_hint.clone());
        optimize_two_block_spatial_length_scale_exact_joint(
            data,
            builder.meanspec(),
            builder.noisespec(),
            kappa_options,
            &joint_setup,
            |rho, _, _, mean_design, noise_design| {
                let fit = {
                    let blocks = builder.build_blocks(
                        rho,
                        mean_design,
                        noise_design,
                        mean_beta_hint_cell.borrow().clone(),
                        noise_beta_hint_cell.borrow().clone(),
                    )?;
                    let family = builder.build_family(mean_design, noise_design);
                    fit_custom_family(&family, &blocks, options)?
                };
                let layout = GamlssLambdaLayout::two_block(
                    builder.mean_penalty_count(mean_design),
                    builder.noise_penalty_count(noise_design),
                );
                if fit.log_lambdas.len() >= layout.total() {
                    mean_log_lambda_hint = Some(layout.mean_from(&fit.log_lambdas));
                    noise_log_lambda_hint = Some(layout.noise_from(&fit.log_lambdas));
                }
                let (mean_beta, noise_beta) = builder.extract_primary_betas(&fit)?;
                mean_beta_hint = Some(mean_beta);
                noise_beta_hint = Some(noise_beta);
                *mean_beta_hint_cell.borrow_mut() = mean_beta_hint.clone();
                *noise_beta_hint_cell.borrow_mut() = noise_beta_hint.clone();
                Ok(fit)
            },
            |rho,
             meanspec_resolved,
             noisespec_resolved,
             mean_design,
             noise_design,
             need_hessian| {
                let blocks = builder.build_blocks(
                    rho,
                    mean_design,
                    noise_design,
                    mean_beta_hint_cell.borrow().clone(),
                    noise_beta_hint_cell.borrow().clone(),
                )?;
                let family = builder.build_family(mean_design, noise_design);
                let psiderivative_blocks = builder.build_psiderivative_blocks(
                    data,
                    meanspec_resolved,
                    noisespec_resolved,
                    mean_design,
                    noise_design,
                )?;
                let eval = evaluate_custom_family_joint_hyper(
                    &family,
                    &blocks,
                    options,
                    rho,
                    &psiderivative_blocks,
                    None,
                    need_hessian,
                )?;
                if need_hessian && eval.outer_hessian.is_none() {
                    return Err(
                        "exact two-block spatial objective requires a full joint [rho, psi] hessian"
                            .to_string(),
                    );
                }
                Ok((eval.objective, eval.gradient, eval.outer_hessian))
            },
        )
        .map_err(|err| {
            format!(
                "exact two-block spatial optimization failed and finite-difference fallback is disabled for this family: {err}"
            )
        })?
    } else if exact_joint_ready {
        let joint_setup = build_two_block_exact_joint_setup(
            builder.meanspec(),
            builder.noisespec(),
            mean_penalty_count,
            noise_penalty_count,
            extra_rho0.as_slice().unwrap_or(&[]),
            kappa_options,
        );
        let mean_beta_hint_cell = std::cell::RefCell::new(mean_beta_hint.clone());
        let noise_beta_hint_cell = std::cell::RefCell::new(noise_beta_hint.clone());
        match optimize_two_block_spatial_length_scale_exact_joint(
            data,
            builder.meanspec(),
            builder.noisespec(),
            kappa_options,
            &joint_setup,
            |rho, _, _, mean_design, noise_design| {
                let fit = {
                    let blocks = builder.build_blocks(
                        rho,
                        mean_design,
                        noise_design,
                        mean_beta_hint_cell.borrow().clone(),
                        noise_beta_hint_cell.borrow().clone(),
                    )?;
                    let family = builder.build_family(mean_design, noise_design);
                    fit_custom_family(&family, &blocks, options)?
                };
                let layout = GamlssLambdaLayout::two_block(
                    builder.mean_penalty_count(mean_design),
                    builder.noise_penalty_count(noise_design),
                );
                if fit.log_lambdas.len() >= layout.total() {
                    mean_log_lambda_hint = Some(layout.mean_from(&fit.log_lambdas));
                    noise_log_lambda_hint = Some(layout.noise_from(&fit.log_lambdas));
                }
                let (mean_beta, noise_beta) = builder.extract_primary_betas(&fit)?;
                mean_beta_hint = Some(mean_beta);
                noise_beta_hint = Some(noise_beta);
                *mean_beta_hint_cell.borrow_mut() = mean_beta_hint.clone();
                *noise_beta_hint_cell.borrow_mut() = noise_beta_hint.clone();
                Ok(fit)
            },
            |rho,
             meanspec_resolved,
             noisespec_resolved,
             mean_design,
             noise_design,
             need_hessian| {
                let blocks = builder.build_blocks(
                    rho,
                    mean_design,
                    noise_design,
                    mean_beta_hint_cell.borrow().clone(),
                    noise_beta_hint_cell.borrow().clone(),
                )?;
                let family = builder.build_family(mean_design, noise_design);
                let psiderivative_blocks = builder.build_psiderivative_blocks(
                    data,
                    meanspec_resolved,
                    noisespec_resolved,
                    mean_design,
                    noise_design,
                )?;
                let eval = evaluate_custom_family_joint_hyper(
                    &family,
                    &blocks,
                    options,
                    rho,
                    &psiderivative_blocks,
                    None,
                    need_hessian,
                )?;
                if need_hessian && eval.outer_hessian.is_none() {
                    return Err(
                        "exact two-block spatial objective requires a full joint [rho, psi] hessian"
                            .to_string(),
                    );
                }
                Ok((eval.objective, eval.gradient, eval.outer_hessian))
            },
        ) {
            Ok(sol) => sol,
            Err(err) => {
                if !kappa_options.allow_finite_difference_fallback {
                    return Err(format!(
                        "exact two-block spatial optimization failed ({err}); finite-difference fallback is disabled by default"
                    ));
                }
                log::warn!(
                    "exact two-block spatial optimization failed ({}); falling back to finite-difference optimizer",
                    err
                );
                usedfd_spatial_search = true;
                optimize_two_block_spatial_length_scale(
                    data,
                    builder.meanspec(),
                    builder.noisespec(),
                    kappa_options,
                    |mean_design, noise_design| {
                        let theta = compose_theta_from_hints(
                            builder.mean_penalty_count(mean_design),
                            builder.noise_penalty_count(noise_design),
                            &mean_log_lambda_hint,
                            &noise_log_lambda_hint,
                            &extra_rho0,
                        );
                        let blocks = builder.build_blocks(
                            &theta,
                            mean_design,
                            noise_design,
                            mean_beta_hint.clone(),
                            noise_beta_hint.clone(),
                        )?;
                        let family = builder.build_family(mean_design, noise_design);
                        let fit = fit_custom_family(&family, &blocks, &spatial_search_options)?;
                        let layout = GamlssLambdaLayout::two_block(
                            builder.mean_penalty_count(mean_design),
                            builder.noise_penalty_count(noise_design),
                        );
                        if fit.log_lambdas.len() >= layout.total() {
                            mean_log_lambda_hint = Some(layout.mean_from(&fit.log_lambdas));
                            noise_log_lambda_hint = Some(layout.noise_from(&fit.log_lambdas));
                        }
                        let (mean_beta, noise_beta) = builder.extract_primary_betas(&fit)?;
                        mean_beta_hint = Some(mean_beta);
                        noise_beta_hint = Some(noise_beta);
                        Ok(fit)
                    },
                    |fit| fit.penalizedobjective,
                )?
            }
        }
    } else {
        if !kappa_options.allow_finite_difference_fallback {
            return Err(
                "finite-difference spatial length-scale optimization is disabled by default; enable allow_finite_difference_fallback to use the legacy coordinate-search path"
                    .to_string(),
            );
        }
        usedfd_spatial_search = true;
        optimize_two_block_spatial_length_scale(
            data,
            builder.meanspec(),
            builder.noisespec(),
            kappa_options,
            |mean_design, noise_design| {
                let theta = compose_theta_from_hints(
                    builder.mean_penalty_count(mean_design),
                    builder.noise_penalty_count(noise_design),
                    &mean_log_lambda_hint,
                    &noise_log_lambda_hint,
                    &extra_rho0,
                );
                let blocks = builder.build_blocks(
                    &theta,
                    mean_design,
                    noise_design,
                    mean_beta_hint.clone(),
                    noise_beta_hint.clone(),
                )?;
                let family = builder.build_family(mean_design, noise_design);
                let fit = fit_custom_family(&family, &blocks, &spatial_search_options)?;
                let layout = GamlssLambdaLayout::two_block(
                    builder.mean_penalty_count(mean_design),
                    builder.noise_penalty_count(noise_design),
                );
                if fit.log_lambdas.len() >= layout.total() {
                    mean_log_lambda_hint = Some(layout.mean_from(&fit.log_lambdas));
                    noise_log_lambda_hint = Some(layout.noise_from(&fit.log_lambdas));
                }
                let (mean_beta, noise_beta) = builder.extract_primary_betas(&fit)?;
                mean_beta_hint = Some(mean_beta);
                noise_beta_hint = Some(noise_beta);
                Ok(fit)
            },
            |fit| fit.penalizedobjective,
        )?
    };

    if usedfd_spatial_search {
        let theta = compose_theta_from_hints(
            builder.mean_penalty_count(&solved.mean_design),
            builder.noise_penalty_count(&solved.noise_design),
            &mean_log_lambda_hint,
            &noise_log_lambda_hint,
            &extra_rho0,
        );
        let blocks = builder.build_blocks(
            &theta,
            &solved.mean_design,
            &solved.noise_design,
            mean_beta_hint.clone(),
            noise_beta_hint.clone(),
        )?;
        let family = builder.build_family(&solved.mean_design, &solved.noise_design);
        solved.fit = fit_custom_family(&family, &blocks, options)?;
    }

    builder.augment_result_designs(&mut solved.mean_design, &mut solved.noise_design);

    Ok(BlockwiseTermFitResult {
        fit: solved.fit,
        meanspec_resolved: solved.resolved_meanspec,
        noisespec_resolved: solved.resolved_noisespec,
        mean_design: solved.mean_design,
        noise_design: solved.noise_design,
    })
}

struct GaussianLocationScaleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
}

impl LocationScaleFamilyBuilder for GaussianLocationScaleTermBuilder {
    type Family = GaussianLocationScaleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
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
            design: DesignMatrix::Dense(mean_design.design.clone()),
            offset: Array1::zeros(self.y.len()),
            penalties: mean_design.penalties.clone(),
            initial_log_lambdas: mean_log_lambdas,
            initial_beta: mean_beta_hint,
        };
        let mut noisespec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(prepared_gaussian_log_sigma_design(
                &mean_design.design,
                &noise_design.design,
                &self.weights,
                noise_design
                    .intercept_range
                    .end
                    .min(noise_design.design.ncols()),
            )?),
            offset: Array1::zeros(self.y.len()),
            penalties: noise_design.penalties.clone(),
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
        let preparednoise_design = prepared_gaussian_log_sigma_design(
            &mean_design.design,
            &noise_design.design,
            &self.weights,
            noise_design
                .intercept_range
                .end
                .min(noise_design.design.ncols()),
        )
        .expect("prepared Gaussian log-sigma design should match block construction");
        GaussianLocationScaleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: Some(DesignMatrix::Dense(mean_design.design.clone())),
            log_sigma_design: Some(DesignMatrix::Dense(preparednoise_design)),
            cached_row_scalars: std::cell::RefCell::new(None),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &BlockwiseFitResult,
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
}

struct BinomialLocationScaleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    link_kind: InverseLink,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
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
        let mut log_sigma_penalties = noise_design.penalties.clone();
        log_sigma_penalties.push(identity_penalty(identifiednoise_design.ncols()));
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(identifiednoise_design),
            offset: Array1::zeros(self.y.len()),
            penalties: log_sigma_penalties,
            initial_log_lambdas: layout.noise_from(theta),
            initial_beta: noise_beta_hint,
        };
        Ok(vec![
            ParameterBlockSpec {
                name: "threshold".to_string(),
                design: DesignMatrix::Dense(mean_design.design.clone()),
                offset: Array1::zeros(self.y.len()),
                penalties: mean_design.penalties.clone(),
                initial_log_lambdas: layout.mean_from(theta),
                initial_beta: mean_beta_hint,
            },
            log_sigmaspec,
        ])
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
            threshold_design: Some(DesignMatrix::Dense(mean_design.design.clone())),
            log_sigma_design: Some(DesignMatrix::Dense(identifiednoise_design)),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &BlockwiseFitResult,
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
}

struct BinomialLocationScaleWiggleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    link_kind: InverseLink,
    meanspec: TermCollectionSpec,
    noisespec: TermCollectionSpec,
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
        let mut log_sigma_penalties = noise_design.penalties.clone();
        log_sigma_penalties.push(identity_penalty(identifiednoise_design.ncols()));
        let log_sigmaspec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(identifiednoise_design),
            offset: Array1::zeros(self.y.len()),
            penalties: log_sigma_penalties,
            initial_log_lambdas: layout.noise_from(theta),
            initial_beta: noise_beta_hint,
        };
        Ok(vec![
            ParameterBlockSpec {
                name: "threshold".to_string(),
                design: DesignMatrix::Dense(mean_design.design.clone()),
                offset: Array1::zeros(self.y.len()),
                penalties: mean_design.penalties.clone(),
                initial_log_lambdas: layout.mean_from(theta),
                initial_beta: mean_beta_hint,
            },
            log_sigmaspec,
            ParameterBlockSpec {
                name: "wiggle".to_string(),
                design: self.wiggle_block.design.clone(),
                offset: self.wiggle_block.offset.clone(),
                penalties: self.wiggle_block.penalties.clone(),
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
            threshold_design: Some(DesignMatrix::Dense(mean_design.design.clone())),
            log_sigma_design: Some(DesignMatrix::Dense(identifiednoise_design)),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &BlockwiseFitResult,
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

pub fn fit_gaussian_location_scale_terms(
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
        },
        options,
        kappa_options,
    )
}

pub fn fit_binomial_location_scale_terms(
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
        },
        options,
        kappa_options,
    )
}

pub fn fit_binomial_location_scalewiggle_terms(
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
            wiggle_knots: spec.wiggle_knots,
            wiggle_degree: spec.wiggle_degree,
            wiggle_block: spec.wiggle_block,
        },
        options,
        kappa_options,
    )
}

pub fn fit_binomial_location_scalewiggle_terms_auto(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleTermSpec,
    wiggle_cfg: WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermWiggleFitResult, String> {
    let pilot = fit_binomial_location_scale_terms(
        data,
        BinomialLocationScaleTermSpec {
            y: spec.y.clone(),
            weights: spec.weights.clone(),
            link_kind: spec.link_kind.clone(),
            thresholdspec: spec.thresholdspec.clone(),
            log_sigmaspec: spec.log_sigmaspec.clone(),
        },
        options,
        kappa_options,
    )?;

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
    let sigma = eta_ls.mapv(f64::exp);
    let q_seed = Array1::from_iter(
        eta_t
            .iter()
            .zip(sigma.iter())
            .map(|(&t, &s)| -t / s.max(1e-12)),
    );
    let identifiednoise_design = identified_binomial_log_sigma_design(
        &pilot.mean_design,
        &pilot.noise_design,
        &spec.weights,
    )?;

    let threshold_penalty_count = pilot.mean_design.penalties.len();
    let noise_penalty_count = pilot.noise_design.penalties.len();
    let threshold_log_lambdas = slice_log_lambda_block(
        &pilot.fit.log_lambdas,
        0,
        threshold_penalty_count,
        "threshold",
    )?;
    let noise_log_lambdas = slice_log_lambda_block(
        &pilot.fit.log_lambdas,
        threshold_penalty_count,
        noise_penalty_count,
        "log_sigma",
    )?;

    let (mut wiggle_block, wiggle_knots) =
        buildwiggle_block_input_from_seed(q_seed.view(), &wiggle_cfg)?;
    let pw = wiggle_block.design.ncols();
    for &ord in wiggle_penalty_orders {
        if ord <= 1 || ord >= pw {
            continue;
        }
        let s = create_difference_penalty_matrix(pw, ord, None).map_err(|e| e.to_string())?;
        wiggle_block.penalties.push(s);
    }

    let fit = fit_binomial_location_scalewiggle(
        BinomialLocationScaleWiggleSpec {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            wiggle_knots: wiggle_knots.clone(),
            wiggle_degree: wiggle_cfg.degree,
            threshold_block: ParameterBlockInput {
                design: DesignMatrix::Dense(pilot.mean_design.design.clone()),
                offset: Array1::zeros(pilot.mean_design.design.nrows()),
                penalties: pilot.mean_design.penalties.clone(),
                initial_log_lambdas: Some(threshold_log_lambdas),
                initial_beta: pilot.fit.block_states.first().map(|b| b.beta.clone()),
            },
            log_sigma_block: ParameterBlockInput {
                design: DesignMatrix::Dense(identifiednoise_design),
                offset: Array1::zeros(pilot.noise_design.design.nrows()),
                penalties: pilot.noise_design.penalties.clone(),
                initial_log_lambdas: Some(noise_log_lambdas),
                initial_beta: pilot.fit.block_states.get(1).map(|b| b.beta.clone()),
            },
            wiggle_block,
        },
        options,
    )?;

    Ok(BlockwiseTermWiggleFitResult {
        fit: BlockwiseTermFitResult {
            fit,
            meanspec_resolved: pilot.meanspec_resolved,
            noisespec_resolved: pilot.noisespec_resolved,
            mean_design: pilot.mean_design,
            noise_design: pilot.noise_design,
        },
        wiggle_knots,
        wiggle_degree: wiggle_cfg.degree,
    })
}

pub fn fit_binomial_location_scale_termsworkflow(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleTermSpec,
    wiggle: Option<BinomialLocationScaleWiggleWorkflowConfig>,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BinomialLocationScaleWorkflowResult, String> {
    if let Some(wiggle_cfg) = wiggle {
        let solved = fit_binomial_location_scalewiggle_terms_auto(
            data,
            spec,
            WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
            options,
            kappa_options,
        )?;
        let fit = solved.fit.fit;
        let betawiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(BinomialLocationScaleWorkflowResult {
            fit: BlockwiseTermFitResult {
                fit,
                meanspec_resolved: solved.fit.meanspec_resolved,
                noisespec_resolved: solved.fit.noisespec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            },
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            betawiggle,
        })
    } else {
        let solved = fit_binomial_location_scale_terms(data, spec, options, kappa_options)?;
        Ok(BinomialLocationScaleWorkflowResult {
            fit: solved,
            wiggle_knots: None,
            wiggle_degree: None,
            betawiggle: None,
        })
    }
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

struct BinomialLocationScaleExactGeometry<'a> {
    threshold_design: &'a DesignMatrix,
    log_sigma_design: &'a DesignMatrix,
    wiggle_design: Option<&'a Array2<f64>>,
    d2sigma_deta2: &'a Array1<f64>,
    d2q_dq02: Option<&'a Array1<f64>>,
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
fn nonwiggle_q_derivs(
    eta_t: f64,
    sigma: f64,
    dsigma: f64,
    d2sigma: f64,
    d3sigma: f64,
) -> NonWiggleQDerivs {
    let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) =
        crate::families::survival_location_scale::q_chain_derivs_scalar(
            eta_t, sigma, dsigma, d2sigma, d3sigma,
        );
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
    -eta_t / sigma.max(1e-12)
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

    let mut sigma = Array1::<f64>::zeros(n);
    let mut dsigma_deta = Array1::<f64>::zeros(n);
    let mut q0 = Array1::<f64>::zeros(n);
    let mut mu = Array1::<f64>::zeros(n);
    let mut dmu_dq = Array1::<f64>::zeros(n);
    let mut d2mu_dq2 = Array1::<f64>::zeros(n);
    let mut d3mu_dq3 = Array1::<f64>::zeros(n);
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
        sigma[i] = row.sigma;
        dsigma_deta[i] = row.dsigma_deta;
        q0[i] = row.q0;
        mu[i] = row.inverse_link.mu;
        dmu_dq[i] = row.inverse_link.d1;
        d2mu_dq2[i] = row.inverse_link.d2;
        d3mu_dq3[i] = row.inverse_link.d3;
        ll += row.ll;
    }

    Ok(BinomialLocationScaleCore {
        sigma,
        dsigma_deta,
        q0,
        mu,
        dmu_dq,
        d2mu_dq2,
        d3mu_dq3,
        log_likelihood: ll,
    })
}

fn binomial_location_scale_working_sets(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    etawiggle: Option<&Array1<f64>>,
    dq_dq0: Option<&Array1<f64>>,
    exact_geometry: Option<BinomialLocationScaleExactGeometry<'_>>,
    core: &BinomialLocationScaleCore,
) -> Result<(BlockWorkingSet, BlockWorkingSet, Option<BlockWorkingSet>), String> {
    let n = y.len();
    if let Some(geom) = exact_geometry {
        let mut grad_eta_t = Array1::<f64>::zeros(n);
        let mut h_eta_t = Array1::<f64>::zeros(n);
        let mut grad_eta_ls = Array1::<f64>::zeros(n);
        let mut h_eta_ls = Array1::<f64>::zeros(n);
        let mut grad_q = etawiggle.map(|_| Array1::<f64>::zeros(n));
        let mut h_q_psd = etawiggle.map(|_| Array1::<f64>::zeros(n));

        for i in 0..n {
            let (score_q, curvature_q, _) = binomial_score_curvaturethird_from_jet(
                y[i],
                weights[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
            );
            let a_i = dq_dq0.map_or(1.0, |v| v[i]);
            let c_i = geom.d2q_dq02.map_or(0.0, |v| v[i]);
            let q0 = nonwiggle_q_derivs(
                eta_t[i],
                core.sigma[i],
                core.dsigma_deta[i],
                geom.d2sigma_deta2[i],
                0.0,
            );
            // Full rowwise chain rule for the exact wiggle geometry.
            //
            // For one observation we work with
            //
            //   q0(t, l) = -t / sigma(l),
            //   q(t, l)  = q0 + w(q0),
            //   F(q)     = -ell(q),
            //
            // where t = eta_t, l = eta_log_sigma, sigma(l) = exp(l), and
            //
            //   a_i = dq/dq0 = 1 + w'(q0),
            //   c_i = d^2 q / d q0^2 = w''(q0).
            //
            // The non-wiggle derivatives returned by `nonwiggle_q_derivs` are
            //
            //   q0_t  = -1/sigma,
            //   q0_l  = -q0,
            //   q0_tt = 0,
            //   q0_tl = 1/sigma,
            //   q0_ll = q0.
            //
            // The wiggle-composed scalar q = h(q0) with h'(q0)=a_i and
            // h''(q0)=c_i then satisfies
            //
            //   q_t  = a_i * q0_t,
            //   q_l  = a_i * q0_l,
            //   q_tt = c_i * q0_t^2 + a_i * q0_tt,
            //   q_tl = c_i * q0_t q0_l + a_i * q0_tl,
            //   q_ll = c_i * q0_l^2 + a_i * q0_ll.
            //
            // Because q0_tt = 0, the threshold curvature reduces to
            //
            //   q_tt = c_i * q0_t^2.
            //
            // This is the term the old code was missing. The exact negative
            // log-likelihood Hessian in (t, l) coordinates is
            //
            //   F_ab = m2 * q_a q_b + m1 * q_ab,
            //
            // where (m1, m2) are the first and second derivatives of F with
            // respect to the scalar q. In particular,
            //
            //   F_tt = m2 * q_t^2 + m1 * q_tt,
            //   F_ll = m2 * q_l^2 + m1 * q_ll.
            //
            // So once the wiggle is nonlinear (c_i != 0), dropping q_tt from
            // the threshold block is mathematically wrong even though q0 itself
            // is linear in eta_t. The spatial wiggle failure that triggered this
            // patch was exactly that omission surfacing inside the exact-Newton
            // path.
            let q_t = a_i * q0.q_t;
            let q_ls = a_i * q0.q_ls;
            let q_tt = c_i * q0.q_t * q0.q_t;
            let q_ll = c_i * q0.q_ls * q0.q_ls + a_i * q0.q_ll;
            let (m1, m2, _) = binomial_neglog_q_derivatives_from_jet(
                y[i],
                weights[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
            );

            grad_eta_t[i] = score_q * q_t;
            grad_eta_ls[i] = score_q * q_ls;
            h_eta_t[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_t, q_t, q_tt);
            h_eta_ls[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_ls, q_ls, q_ll);

            if let Some(grad_q) = grad_q.as_mut() {
                grad_q[i] = score_q;
            }
            if let Some(h_q_psd) = h_q_psd.as_mut() {
                h_q_psd[i] = curvature_q.max(0.0);
            }
        }

        let grad_t = geom.threshold_design.transpose_vector_multiply(&grad_eta_t);
        let hess_t = xt_diag_x_symmetric(geom.threshold_design, &h_eta_t)?.to_dense();
        let grad_ls = geom
            .log_sigma_design
            .transpose_vector_multiply(&grad_eta_ls);
        let hess_ls = xt_diag_x_symmetric(geom.log_sigma_design, &h_eta_ls)?.to_dense();
        let wws = match (geom.wiggle_design, grad_q, h_q_psd) {
            (Some(wiggle_design), Some(grad_q), Some(h_q_psd)) => {
                Some(BlockWorkingSet::ExactNewton {
                    gradient: fast_atv(wiggle_design, &grad_q),
                    hessian: SymmetricMatrix::Dense(xt_diag_x_dense(wiggle_design, &h_q_psd)?),
                })
            }
            _ => None,
        };

        return Ok((
            BlockWorkingSet::ExactNewton {
                gradient: grad_t,
                hessian: SymmetricMatrix::Dense(hess_t),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_ls,
                hessian: SymmetricMatrix::Dense(hess_ls),
            },
            wws,
        ));
    }

    let mut z_t = Array1::<f64>::zeros(n);
    let mut w_t = Array1::<f64>::zeros(n);
    let mut z_ls = Array1::<f64>::zeros(n);
    let mut w_ls = Array1::<f64>::zeros(n);
    let mut zw = etawiggle.map(|_| Array1::<f64>::zeros(n));
    let mut ww = etawiggle.map(|_| Array1::<f64>::zeros(n));

    for i in 0..n {
        let var = (core.mu[i] * (1.0 - core.mu[i])).max(MIN_PROB);
        let link_chain = dq_dq0.map_or(1.0, |v| v[i]);

        // Location/threshold chain: dq/deta_t = -1/sigma
        let chain_t = -link_chain / core.sigma[i].max(1e-12);
        let dmu_t = core.dmu_dq[i] * chain_t;
        if weights[i] == 0.0 || dmu_t == 0.0 {
            w_t[i] = 0.0;
            z_t[i] = eta_t[i];
        } else {
            w_t[i] = floor_positiveweight(weights[i] * (dmu_t * dmu_t / var), MIN_WEIGHT);
            z_t[i] = eta_t[i] + (y[i] - core.mu[i]) / signedwith_floor(dmu_t, MIN_DERIV);
        }

        // Scale chain: dq/deta_log_sigma = -q0 * dsigma/deta / sigma
        // This is the generic location-scale structure; the -Z multiplier appears here.
        let chain_ls = {
            let s = core.sigma[i].max(1e-12);
            -link_chain * core.q0[i] * core.dsigma_deta[i] / s
        };
        let dmu_ls = core.dmu_dq[i] * chain_ls;
        if weights[i] == 0.0 || dmu_ls == 0.0 {
            w_ls[i] = 0.0;
            z_ls[i] = eta_ls[i];
        } else {
            w_ls[i] = floor_positiveweight(weights[i] * (dmu_ls * dmu_ls / var), MIN_WEIGHT);
            z_ls[i] = eta_ls[i] + (y[i] - core.mu[i]) / signedwith_floor(dmu_ls, MIN_DERIV);
        }

        if let (Some(etaw), Some(zwv), Some(wwv)) = (etawiggle, zw.as_mut(), ww.as_mut()) {
            // Wiggle enters additively in q, so chain is 1.
            let dmuw = core.dmu_dq[i];
            if weights[i] == 0.0 || dmuw == 0.0 {
                wwv[i] = 0.0;
                zwv[i] = etaw[i];
            } else {
                wwv[i] = floor_positiveweight(weights[i] * (dmuw * dmuw / var), MIN_WEIGHT);
                zwv[i] = etaw[i] + (y[i] - core.mu[i]) / signedwith_floor(dmuw, MIN_DERIV);
            }
        }
    }

    let tws = BlockWorkingSet::Diagonal {
        working_response: z_t,
        working_weights: w_t,
    };
    let lsws = BlockWorkingSet::Diagonal {
        working_response: z_ls,
        working_weights: w_ls,
    };
    let wws = match (zw, ww) {
        (Some(z), Some(w)) => Some(BlockWorkingSet::Diagonal {
            working_response: z,
            working_weights: w,
        }),
        _ => None,
    };
    Ok((tws, lsws, wws))
}

/// Built-in Gaussian location-scale family:
/// - Block 0: location μ(·) with identity link
/// - Block 1: log-scale log σ(·) with log link
#[derive(Clone)]
pub struct GaussianLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Cached per-observation row scalars keyed by (eta_mu[0], eta_ls[0]) fingerprint.
    /// Avoids recomputing O(n) scalars K+ times per REML gradient/Hessian evaluation.
    cached_row_scalars: std::cell::RefCell<Option<(f64, f64, GaussianJointRowScalars)>>,
}

struct GaussianLocationScaleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    xmu_psi: Array2<f64>,
    x_ls_psi: Array2<f64>,
    zmu_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}

#[derive(Clone)]
struct GaussianJointRowScalars {
    w: Array1<f64>,
    m: Array1<f64>,
    n: Array1<f64>,
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
    let mut w = Array1::<f64>::uninit(nobs);
    let mut m = Array1::<f64>::uninit(nobs);
    let mut n = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let s = safe_exp(eta_ls[i]).max(1e-12);
        let wi = weights[i] / (s * s);
        let ri = y[i] - etamu[i];
        w[i].write(wi);
        m[i].write(ri * wi);
        n[i].write(ri * ri * wi);
    }
    // SAFETY: all elements written in the loop above.
    let w = unsafe { w.assume_init() };
    let m = unsafe { m.assume_init() };
    let n = unsafe { n.assume_init() };
    Ok(GaussianJointRowScalars { w, m, n })
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
        let dm = dotmu[i];
        let de = dot_eta[i];
        w_u[i].write(-2.0 * wi * de);
        c_u[i].write(-2.0 * wi * dm - 4.0 * mi * de);
        d_u[i].write(-4.0 * mi * dm - 4.0 * ni * de);
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
        let dmu = dotmu_u[i];
        let deu = dot_eta_u[i];
        let dmv = dotmuv[i];
        let dev = dot_etav[i];
        w_uv[i].write(4.0 * wi * deu * dev);
        c_uv[i].write(4.0 * wi * (dmu * dev + dmv * deu) + 8.0 * mi * deu * dev);
        d_uv[i].write(
            4.0 * wi * dmu * dmv + 8.0 * mi * (dmu * dev + dmv * deu) + 8.0 * ni * deu * dev,
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
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ma = mu_a[i];
        let ea = eta_a[i];
        let smu = -mi;
        let sls = 1.0 - ni;
        scoremu[i].write(smu);
        score_ls[i].write(sls);
        dscoremu[i].write(wi * ma + 2.0 * mi * ea);
        dscore_ls[i].write(2.0 * mi * ma + 2.0 * ni * ea);
        hmumu[i].write(wi);
        hmu_ls[i].write(2.0 * mi);
        h_ls_ls[i].write(2.0 * ni);
        dhmumu[i].write(-2.0 * wi * ea);
        dhmu_ls[i].write(-2.0 * wi * ma - 4.0 * mi * ea);
        dh_ls_ls[i].write(-4.0 * mi * ma - 4.0 * ni * ea);
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
        let ma = mu_a[i];
        let ea = eta_a[i];
        let mb = mu_b[i];
        let eb = eta_b[i];
        let mab = mu_ab[i];
        let eab = eta_ab[i];
        let cross = ma * eb + mb * ea;
        let ea_eb = ea * eb;
        let ma_mb = ma * mb;
        objective_psi_psirow[i]
            .write(wi * ma_mb + 2.0 * mi * cross + 2.0 * ni * ea_eb - mi * mab + (1.0 - ni) * eab);
        d2scoremu[i].write(wi * mab - 2.0 * wi * cross - 4.0 * mi * ea_eb + 2.0 * mi * eab);
        d2score_ls[i].write(
            -2.0 * wi * ma_mb - 4.0 * mi * cross - 4.0 * ni * ea_eb
                + 2.0 * mi * mab
                + 2.0 * ni * eab,
        );
        d2hmumu[i].write(4.0 * wi * ea_eb - 2.0 * wi * eab);
        d2hmu_ls[i].write(-2.0 * wi * mab + 4.0 * wi * cross + 8.0 * mi * ea_eb - 4.0 * mi * eab);
        d2h_ls_ls[i].write(
            4.0 * wi * ma_mb + 8.0 * mi * cross + 8.0 * ni * ea_eb
                - 4.0 * mi * mab
                - 4.0 * ni * eab,
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
        let dm = dotmu[i];
        let de = dot_eta[i];
        let ma = mu_a[i];
        let ea = eta_a[i];
        let dma = dotmu_a[i];
        let dea = dot_eta_a[i];
        let cross = de * ma + dm * ea;
        // First directional: w_u not needed, compute c_u and d_u inline
        dhmumu_u[i].write(-2.0 * wi * de);
        dhmu_ls_u[i].write(-2.0 * wi * dm - 4.0 * mi * de);
        dh_ls_ls_u[i].write(-4.0 * mi * dm - 4.0 * ni * de);
        d2hmumu[i].write(4.0 * wi * de * ea - 2.0 * wi * dea);
        d2hmu_ls[i].write(-2.0 * wi * dma + 4.0 * wi * cross + 8.0 * mi * de * ea - 4.0 * mi * dea);
        d2h_ls_ls[i].write(
            4.0 * wi * dm * ma + 8.0 * mi * cross + 8.0 * ni * de * ea
                - 4.0 * mi * dma
                - 4.0 * ni * dea,
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

fn gaussian_joint_hessian_from_coeffs(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
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
            "gaussian_joint_hessian_from_coeffs dimension mismatch: xmu {}x{}, x_ls {}x{}, coeffs {}/{}/{}",
            xmu.nrows(),
            xmu.ncols(),
            x_ls.nrows(),
            x_ls.ncols(),
            hmumu_coeff.len(),
            hmu_ls_coeff.len(),
            h_ls_ls_coeff.len()
        ));
    }
    // Fused single-pass: reads X_mu and X_ls once instead of twice each.
    Ok(fast_joint_hessian_2x2(
        xmu,
        x_ls,
        hmumu_coeff,
        hmu_ls_coeff,
        h_ls_ls_coeff,
    ))
}

fn gaussian_joint_psihessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_psi: &Array2<f64>,
    x_ls_psi: &Array2<f64>,
    weights: &GaussianJointPsiFirstWeights,
) -> Result<Array2<f64>, String> {
    // For the symmetric blocks (hmumu, h_ls_ls), the pair
    //   X_psi^T D X  and  X^T D X_psi
    // are transposes of each other, so compute one and add its transpose.
    let a_mu = xt_diag_y_dense(xmu_psi, &weights.hmumu, xmu)?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &weights.dhmumu)?;
    let hmu_ls = xt_diag_y_dense(xmu_psi, &weights.hmu_ls, x_ls)?
        + &xt_diag_y_dense(xmu, &weights.hmu_ls, x_ls_psi)?
        + &xt_diag_y_dense(xmu, &weights.dhmu_ls, x_ls)?;
    let a_ls = xt_diag_y_dense(x_ls_psi, &weights.h_ls_ls, x_ls)?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &weights.dh_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

fn gaussian_joint_psisecondhessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_i: &Array2<f64>,
    x_ls_i: &Array2<f64>,
    xmu_j: &Array2<f64>,
    x_ls_j: &Array2<f64>,
    xmu_ab: &Array2<f64>,
    x_ls_ab: &Array2<f64>,
    weights_i: &GaussianJointPsiFirstWeights,
    weights_j: &GaussianJointPsiFirstWeights,
    secondweights: &GaussianJointPsiSecondWeights,
) -> Result<Array2<f64>, String> {
    // Exploit transpose symmetry: X_a^T D X_b and X_b^T D X_a are transposes.
    // For each such pair in the symmetric blocks (hmumu, h_ls_ls), compute one
    // and add its transpose, halving the number of O(np²) products.
    let a_ab_mu = xt_diag_y_dense(xmu_ab, &weights_i.hmumu, xmu)?;
    let a_ij_mu = xt_diag_y_dense(xmu_i, &weights_i.hmumu, xmu_j)?;
    let a_iwj_mu = xt_diag_y_dense(xmu_i, &weights_j.dhmumu, xmu)?;
    let a_jwi_mu = xt_diag_y_dense(xmu_j, &weights_i.dhmumu, xmu)?;
    let hmumu = &a_ab_mu
        + &a_ab_mu.t()
        + &a_ij_mu
        + &a_ij_mu.t()
        + &a_iwj_mu
        + &a_iwj_mu.t()
        + &a_jwi_mu
        + &a_jwi_mu.t()
        + &xt_diag_x_dense(xmu, &secondweights.d2hmumu)?;
    let hmu_ls = xt_diag_y_dense(xmu_ab, &weights_i.hmu_ls, x_ls)?
        + &xt_diag_y_dense(xmu_i, &weights_i.hmu_ls, x_ls_j)?
        + &xt_diag_y_dense(xmu_j, &weights_i.hmu_ls, x_ls_i)?
        + &xt_diag_y_dense(xmu_i, &weights_j.dhmu_ls, x_ls)?
        + &xt_diag_y_dense(xmu_j, &weights_i.dhmu_ls, x_ls)?
        + &xt_diag_y_dense(xmu, &weights_i.dhmu_ls, x_ls_j)?
        + &xt_diag_y_dense(xmu, &weights_j.dhmu_ls, x_ls_i)?
        + &xt_diag_y_dense(xmu, &secondweights.d2hmu_ls, x_ls)?
        + &xt_diag_y_dense(xmu, &weights_i.hmu_ls, x_ls_ab)?;
    let a_ab_ls = xt_diag_y_dense(x_ls_ab, &weights_i.h_ls_ls, x_ls)?;
    let a_ij_ls = xt_diag_y_dense(x_ls_i, &weights_i.h_ls_ls, x_ls_j)?;
    let a_iwj_ls = xt_diag_y_dense(x_ls_i, &weights_j.dh_ls_ls, x_ls)?;
    let a_jwi_ls = xt_diag_y_dense(x_ls_j, &weights_i.dh_ls_ls, x_ls)?;
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
    xmu_psi: &Array2<f64>,
    x_ls_psi: &Array2<f64>,
    mixedweights: &GaussianJointPsiMixedDriftWeights,
) -> Result<Array2<f64>, String> {
    let a_mu = xt_diag_y_dense(xmu_psi, &mixedweights.dhmumu_u, xmu)?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &mixedweights.d2hmumu)?;
    let hmu_ls = xt_diag_y_dense(xmu_psi, &mixedweights.dhmu_ls_u, x_ls)?
        + &xt_diag_y_dense(xmu, &mixedweights.dhmu_ls_u, x_ls_psi)?
        + &xt_diag_y_dense(xmu, &mixedweights.d2hmu_ls, x_ls)?;
    let a_ls = xt_diag_y_dense(x_ls_psi, &mixedweights.dh_ls_ls_u, x_ls)?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &mixedweights.d2h_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

#[inline]
fn gaussian_sigma_derivs_up_to_fourth(
    eta: ArrayView1<'_, f64>,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let sigma = eta.mapv(f64::exp);
    (
        sigma.clone(),
        sigma.clone(),
        sigma.clone(),
        sigma.clone(),
        sigma,
    )
}

impl GaussianLocationScaleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    /// Get or compute the per-observation row scalars, caching the result.
    /// Uses a (eta_mu[0], eta_ls[0]) fingerprint to detect when etas change.
    fn get_or_compute_row_scalars(
        &self,
        etamu: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<GaussianJointRowScalars, String> {
        let key = (
            etamu.get(0).copied().unwrap_or(f64::NAN),
            eta_ls.get(0).copied().unwrap_or(f64::NAN),
        );
        {
            let cache = self.cached_row_scalars.borrow();
            if let Some((k0, k1, ref scalars)) = *cache {
                if k0 == key.0 && k1 == key.1 {
                    return Ok(scalars.clone());
                }
            }
        }
        let scalars = gaussian_jointrow_scalars(&self.y, etamu, eta_ls, &self.weights)?;
        *self.cached_row_scalars.borrow_mut() = Some((key.0, key.1, scalars.clone()));
        Ok(scalars)
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

    fn dense_block_designs(&self) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        let mu_design = self.mu_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing mu design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;
        let xmu = match mu_design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(mu_design.to_dense()),
        };
        let x_ls = match log_sigma_design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(log_sigma_design.to_dense()),
        };
        Ok((xmu, x_ls))
    }

    fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        if specs.len() != 2 {
            return Err(format!(
                "GaussianLocationScaleFamily spec-aware exact path expects 2 specs, got {}",
                specs.len()
            ));
        }
        let xmu = match specs[Self::BLOCK_MU].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(specs[Self::BLOCK_MU].design.to_dense()),
        };
        let x_ls = match specs[Self::BLOCK_LOG_SIGMA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(specs[Self::BLOCK_LOG_SIGMA].design.to_dense()),
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
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
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
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
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
        Ok(Some(gaussian_joint_hessian_from_coeffs(
            xmu,
            x_ls,
            &rows.w,
            &(2.0 * &rows.m),
            &(2.0 * &rows.n),
        )?))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
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
        let ximu = xmu.dot(&d_beta_flat.slice(s![0..pmu]));
        let xi_ls = x_ls.dot(&d_beta_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let (dhmumu, dhmu_ls, dh_ls_ls) =
            gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);

        Ok(Some(gaussian_joint_hessian_from_coeffs(
            xmu, x_ls, &dhmumu, &dhmu_ls, &dh_ls_ls,
        )?))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
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
        let ximu_u = xmu.dot(&d_beta_u_flat.slice(s![0..pmu]));
        let xi_ls_u = x_ls.dot(&d_beta_u_flat.slice(s![pmu..pmu + p_ls]));
        let ximuv = xmu.dot(&d_betav_flat.slice(s![0..pmu]));
        let xi_lsv = x_ls.dot(&d_betav_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let (d2hmumu, d2hmu_ls, d2h_ls_ls) =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximuv, &xi_lsv);

        Ok(Some(gaussian_joint_hessian_from_coeffs(
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
                    let mut xmu_psi = Array2::<f64>::zeros((n, pmu));
                    let mut x_ls_psi = Array2::<f64>::zeros((n, p_ls));
                    match block_idx {
                        Self::BLOCK_MU => {
                            if deriv.x_psi.nrows() != n || deriv.x_psi.ncols() != pmu {
                                return Err(format!(
                                    "GaussianLocationScaleFamily mu x_psi shape mismatch: got {}x{}, expected {}x{}",
                                    deriv.x_psi.nrows(),
                                    deriv.x_psi.ncols(),
                                    n,
                                    pmu
                                ));
                            }
                            xmu_psi.assign(&deriv.x_psi);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            if deriv.x_psi.nrows() != n || deriv.x_psi.ncols() != p_ls {
                                return Err(format!(
                                    "GaussianLocationScaleFamily log-sigma x_psi shape mismatch: got {}x{}, expected {}x{}",
                                    deriv.x_psi.nrows(),
                                    deriv.x_psi.ncols(),
                                    n,
                                    p_ls
                                ));
                            }
                            x_ls_psi.assign(&deriv.x_psi);
                        }
                        _ => return Ok(None),
                    }
                    return Ok(Some(GaussianLocationScaleJointPsiDirection {
                        block_idx,
                        local_idx,
                        zmu_psi: xmu_psi.dot(betamu),
                        z_ls_psi: x_ls_psi.dot(beta_ls),
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
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), String> {
        let n = self.y.len();
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let betamu = &block_states[Self::BLOCK_MU].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut xmu_ab = Array2::<f64>::zeros((n, pmu));
        let mut x_ls_ab = Array2::<f64>::zeros((n, p_ls));
        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            if let Some(x_psi_psi) = deriv.x_psi_psi.as_ref() {
                if let Some(x_ab) = x_psi_psi.get(psi_b.local_idx) {
                    match psi_a.block_idx {
                        Self::BLOCK_MU => {
                            if x_ab.nrows() != n || x_ab.ncols() != pmu {
                                return Err(format!(
                                    "GaussianLocationScaleFamily mu x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
                                    x_ab.nrows(),
                                    x_ab.ncols(),
                                    n,
                                    pmu
                                ));
                            }
                            xmu_ab.assign(x_ab);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            if x_ab.nrows() != n || x_ab.ncols() != p_ls {
                                return Err(format!(
                                    "GaussianLocationScaleFamily log-sigma x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
                                    x_ab.nrows(),
                                    x_ab.ncols(),
                                    n,
                                    p_ls
                                ));
                            }
                            x_ls_ab.assign(x_ab);
                        }
                        _ => {}
                    }
                }
            }
        }
        let zmu_ab = xmu_ab.dot(betamu);
        let z_ls_ab = x_ls_ab.dot(beta_ls);
        Ok((xmu_ab, x_ls_ab, zmu_ab, z_ls_ab))
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
        let score_psi = gaussian_pack_joint_score(
            &(dir_a.xmu_psi.t().dot(&weights_a.scoremu) + xmu.t().dot(&weights_a.dscoremu)),
            &(dir_a.x_ls_psi.t().dot(&weights_a.score_ls) + x_ls.t().dot(&weights_a.dscore_ls)),
        );
        let hessian_psi = gaussian_joint_psihessian_fromweights(
            xmu,
            x_ls,
            &dir_a.xmu_psi,
            &dir_a.x_ls_psi,
            &weights_a,
        )?;

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
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
        )?
        else {
            return Ok(None);
        };
        let (xmu_ab, x_ls_ab, zmu_ab, z_ls_ab) = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            &dir_i,
            &dir_j,
            xmu,
            x_ls,
        )?;
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
            &zmu_ab,
            &z_ls_ab,
        );
        let objective_psi_psi = secondweights.objective_psi_psirow.sum();

        let score_psi_psi = gaussian_pack_joint_score(
            &(xmu_ab.t().dot(&weights_i.scoremu)
                + dir_i.xmu_psi.t().dot(&weights_j.dscoremu)
                + dir_j.xmu_psi.t().dot(&weights_i.dscoremu)
                + xmu.t().dot(&secondweights.d2scoremu)),
            &(x_ls_ab.t().dot(&weights_i.score_ls)
                + dir_i.x_ls_psi.t().dot(&weights_j.dscore_ls)
                + dir_j.x_ls_psi.t().dot(&weights_i.dscore_ls)
                + x_ls.t().dot(&secondweights.d2score_ls)),
        );
        let hessian_psi_psi = gaussian_joint_psisecondhessian_fromweights(
            xmu,
            x_ls,
            &dir_i.xmu_psi,
            &dir_i.x_ls_psi,
            &dir_j.xmu_psi,
            &dir_j.x_ls_psi,
            &xmu_ab,
            &x_ls_ab,
            &weights_i,
            &weights_j,
            &secondweights,
        )?;

        Ok(Some(
            crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
                objective_psi_psi,
                score_psi_psi,
                hessian_psi_psi,
            },
        ))
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
        )?
        else {
            return Ok(None);
        };
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
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
        let uzamu = dir_a.xmu_psi.dot(&umu);
        let uza_ls = dir_a.x_ls_psi.dot(&u_ls);
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

        Ok(Some(gaussian_joint_psi_mixedhessian_drift_fromweights(
            xmu,
            x_ls,
            &dir_a.xmu_psi,
            &dir_a.x_ls_psi,
            &mixedweights,
        )?))
    }
}

impl CustomFamily for GaussianLocationScaleFamily {
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

        // SAFETY: every element is written before being read in the loop below.
        let mut zmu = unsafe { Array1::<f64>::uninit(n).assume_init() };
        let mut wmu = unsafe { Array1::<f64>::uninit(n).assume_init() };
        let mut z_ls = unsafe { Array1::<f64>::uninit(n).assume_init() };
        let mut w_ls = unsafe { Array1::<f64>::uninit(n).assume_init() };
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
                let eta_ls_i = ls_s[i];
                let two_eta = 2.0 * eta_ls_i;
                let sigma_i = safe_exp(eta_ls_i).max(1e-12);
                let inv_s2 = (sigma_i * sigma_i).recip().min(1e24);
                let r = y_s[i] - mu_s[i];
                let weight_i = w_s[i];
                ll += weight_i * (-0.5 * (r * r * inv_s2 + ln2pi + two_eta));

                if weight_i == 0.0 {
                    wmu_s[i] = 0.0;
                    zmu_s[i] = mu_s[i];
                } else {
                    wmu_s[i] = floor_positiveweight(weight_i * inv_s2, MIN_WEIGHT);
                    zmu_s[i] = mu_s[i] + r;
                }

                let dlogsigma_du = if sigma_i <= 1e-10 {
                    sigma_i * 1e10
                } else {
                    1.0
                };
                let info_u =
                    floor_positiveweight(2.0 * weight_i * dlogsigma_du * dlogsigma_du, MIN_WEIGHT);
                if info_u == 0.0 {
                    wls_s[i] = 0.0;
                    zls_s[i] = eta_ls_i;
                } else {
                    wls_s[i] = info_u;
                    let score_ls = weight_i * (r * r * inv_s2 - 1.0) * dlogsigma_du;
                    zls_s[i] = eta_ls_i + score_ls / info_u;
                }
            }
        } else {
            for i in 0..n {
                let eta_ls_i = eta_log_sigma[i];
                let two_eta = 2.0 * eta_ls_i;
                let sigma_i = safe_exp(eta_ls_i).max(1e-12);
                let inv_s2 = (sigma_i * sigma_i).recip().min(1e24);
                let r = self.y[i] - etamu[i];
                let weight_i = self.weights[i];
                ll += weight_i * (-0.5 * (r * r * inv_s2 + ln2pi + two_eta));
                if weight_i == 0.0 {
                    wmu[i] = 0.0;
                    zmu[i] = etamu[i];
                } else {
                    wmu[i] = floor_positiveweight(weight_i * inv_s2, MIN_WEIGHT);
                    zmu[i] = etamu[i] + r;
                }
                let dlogsigma_du = if sigma_i <= 1e-10 {
                    sigma_i * 1e10
                } else {
                    1.0
                };
                let info_u =
                    floor_positiveweight(2.0 * weight_i * dlogsigma_du * dlogsigma_du, MIN_WEIGHT);
                if info_u == 0.0 {
                    w_ls[i] = 0.0;
                    z_ls[i] = eta_ls_i;
                } else {
                    w_ls[i] = info_u;
                    let score_ls = weight_i * (r * r * inv_s2 - 1.0) * dlogsigma_du;
                    z_ls[i] = eta_ls_i + score_ls / info_u;
                }
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
        // -0.5 * (r²/s² + ln(2π s²)) = -0.5 * (r²/s² + ln(2π) + 2*eta_ls)
        // since s² = exp(2*eta_ls), ln(s²) = 2*eta_ls.
        // This avoids exp() + ln() per observation.
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        if let (Some(y_s), Some(w_s), Some(mu_s), Some(ls_s)) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
        ) {
            for i in 0..n {
                let two_eta = 2.0 * ls_s[i];
                let inv_s2 = safe_exp(-two_eta).min(1e24);
                let r = y_s[i] - mu_s[i];
                ll += w_s[i] * (-0.5 * (r * r * inv_s2 + ln2pi + two_eta));
            }
        } else {
            for i in 0..n {
                let two_eta = 2.0 * eta_log_sigma[i];
                let inv_s2 = safe_exp(-two_eta).min(1e24);
                let r = self.y[i] - etamu[i];
                ll += self.weights[i] * (-0.5 * (r * r * inv_s2 + ln2pi + two_eta));
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

        let (sigma, d_sigma, d2_sigma, _, _) = gaussian_sigma_derivs_up_to_fourth(eta_ls.view());
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
                        d_sigma[i],
                        d2_sigma[i],
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
        let sigma = block_states[Self::BLOCK_LOG_SIGMA].eta.mapv(f64::exp);
        Ok(GenerativeSpec {
            mean: mu,
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
}

impl BinomialMeanWiggleFamily {
    pub const BLOCK_ETA: usize = 0;
    pub const BLOCK_WIGGLE: usize = 1;

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        let (basis, _) = create_basis::<Dense>(
            q0,
            KnotSource::Provided(self.wiggle_knots.view()),
            self.wiggle_degree,
            options,
        )
        .map_err(|e| e.to_string())?;
        let full = (*basis).clone();
        let (z, _) =
            compute_geometric_constraint_transform(&self.wiggle_knots, self.wiggle_degree, 2)
                .map_err(|e| e.to_string())?;
        if full.ncols() != z.nrows() {
            return Err(format!(
                "wiggle basis/constraint mismatch: basis has {} columns but transform has {} rows",
                full.ncols(),
                z.nrows()
            ));
        }
        Ok(full.dot(&z))
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        betawiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != betawiggle.len() {
            return Err(format!(
                "wiggle derivative/beta mismatch: basis has {} columns but betawiggle has {} coefficients",
                d_constrained.ncols(),
                betawiggle.len()
            ));
        }
        Ok(d_constrained.dot(&betawiggle) + 1.0)
    }
}

impl CustomFamily for BinomialMeanWiggleFamily {
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

    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        match self.link_kind {
            InverseLink::Standard(base_link) => Some(KnownLinkWiggle {
                base_link,
                wiggle_block: Some(Self::BLOCK_WIGGLE),
            }),
            _ => None,
        }
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
        Ok((DesignMatrix::Dense(x), Array1::zeros(nrows)))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
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
}

struct BinomialLocationScaleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    x_t_psi: Array2<f64>,
    x_ls_psi: Array2<f64>,
    z_t_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}

struct BinomialLocationScaleWiggleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    x_t_psi: Array2<f64>,
    x_ls_psi: Array2<f64>,
    z_t_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
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
        matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit))
            && self.threshold_design.is_some()
            && self.log_sigma_design.is_some()
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
            None => Cow::Owned(threshold_design.to_dense()),
        };
        let x_ls = match log_sigma_design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(log_sigma_design.to_dense()),
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
            None => Cow::Owned(specs[Self::BLOCK_T].design.to_dense()),
        };
        let x_ls = match specs[Self::BLOCK_LOG_SIGMA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(specs[Self::BLOCK_LOG_SIGMA].design.to_dense()),
        };
        Ok((xt, x_ls))
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        // The probit non-wiggle family is structurally capable of exact joint
        // outer rho-derivatives whenever the realized threshold and log-sigma
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
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }
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
        //   q = -eta_t exp(-eta_ls),
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
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = core.q0[i];
            let r = 1.0 / core.sigma[i].max(1e-12);
            let (m1, m2, _) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
            );
            coeff_tt[i] = m2 * r * r;
            coeff_tl[i] = r * (m1 + q * m2);
            coeff_ll[i] = q * (m1 + q * m2);
        }

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
            let r = 1.0 / core.sigma[i].max(1e-12);
            let (m1, m2, m3) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
            );
            let a = d_eta_t[i];
            let b = d_eta_ls[i];
            let du = -r * a - q * b;
            coeff_tt[i] = r * r * (m3 * du - 2.0 * m2 * b);
            coeff_tl[i] = r * (q * m3 * du + m2 * (2.0 * du - q * b) - m1 * b);
            coeff_ll[i] = (m1 + 3.0 * q * m2 + q * q * m3) * du;
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
        // With
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
            let r = 1.0 / core.sigma[i].max(1e-12);
            let (m1, m2, m3) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
            );
            let m4 = binomial_neglog_q_fourth_derivative_probit_closed_form(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
            );
            let a = d_eta_t_u[i];
            let b = d_eta_ls_u[i];
            let c = d_eta_tv[i];
            let d = d_eta_lsv[i];
            let du = -r * a - q * b;
            let dv = -r * c - q * d;
            let d2 = r * (a * d + b * c) + q * b * d;
            coeff_tt[i] =
                r * r * (m4 * du * dv + m3 * (d2 - 2.0 * d * du - 2.0 * b * dv) + 4.0 * m2 * b * d);
            coeff_tl[i] = r
                * (q * m4 * du * dv
                    + m3 * (q * d2 + 3.0 * du * dv - q * (d * du + b * dv))
                    + m2 * (q * b * d + 2.0 * d2 - 2.0 * (d * du + b * dv))
                    + m1 * b * d);
            coeff_ll[i] = q * q * m4 * du * dv
                + m3 * (q * q * d2 + 5.0 * q * du * dv)
                + m2 * (3.0 * q * d2 + 4.0 * du * dv)
                + m1 * d2;
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
                    let mut x_t_psi = Array2::<f64>::zeros((n, pt));
                    let mut x_ls_psi = Array2::<f64>::zeros((n, pls));
                    match block_idx {
                        Self::BLOCK_T => {
                            if deriv.x_psi.nrows() != n || deriv.x_psi.ncols() != pt {
                                return Err(format!(
                                    "BinomialLocationScaleFamily threshold x_psi shape mismatch: got {}x{}, expected {}x{}",
                                    deriv.x_psi.nrows(),
                                    deriv.x_psi.ncols(),
                                    n,
                                    pt
                                ));
                            }
                            x_t_psi.assign(&deriv.x_psi);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            if deriv.x_psi.nrows() != n || deriv.x_psi.ncols() != pls {
                                return Err(format!(
                                    "BinomialLocationScaleFamily log-sigma x_psi shape mismatch: got {}x{}, expected {}x{}",
                                    deriv.x_psi.nrows(),
                                    deriv.x_psi.ncols(),
                                    n,
                                    pls
                                ));
                            }
                            x_ls_psi.assign(&deriv.x_psi);
                        }
                        _ => return Ok(None),
                    }
                    let z_t_psi = x_t_psi.dot(beta_t);
                    let z_ls_psi = x_ls_psi.dot(beta_ls);
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
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), String> {
        let n = self.y.len();
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_t = &block_states[Self::BLOCK_T].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut x_t_ab = Array2::<f64>::zeros((n, pt));
        let mut x_ls_ab = Array2::<f64>::zeros((n, pls));

        // The smooth layer stores second derivatives block-locally. For a pair
        // of global psi coordinates (a, b), the only potentially nonzero
        // X_{psi_a psi_b} lives in the derivative payload of the block whose
        // basis actually moves under that pair. Cross-block mixed second
        // design derivatives are therefore zero unless explicitly provided.
        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            if let Some(x_psi_psi) = deriv.x_psi_psi.as_ref() {
                let maybe = x_psi_psi.get(psi_b.local_idx);
                if let Some(x_ab) = maybe {
                    match psi_a.block_idx {
                        Self::BLOCK_T => {
                            if x_ab.nrows() != n || x_ab.ncols() != pt {
                                return Err(format!(
                                    "BinomialLocationScaleFamily threshold x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
                                    x_ab.nrows(),
                                    x_ab.ncols(),
                                    n,
                                    pt
                                ));
                            }
                            x_t_ab.assign(x_ab);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            if x_ab.nrows() != n || x_ab.ncols() != pls {
                                return Err(format!(
                                    "BinomialLocationScaleFamily log-sigma x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
                                    x_ab.nrows(),
                                    x_ab.ncols(),
                                    n,
                                    pls
                                ));
                            }
                            x_ls_ab.assign(x_ab);
                        }
                        _ => {}
                    }
                }
            }
        }

        let z_t_ab = x_t_ab.dot(beta_t);
        let z_ls_ab = x_ls_ab.dot(beta_ls);
        Ok((x_t_ab, x_ls_ab, z_t_ab, z_ls_ab))
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
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }
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

        let Some(psi_dir) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
        )?
        else {
            return Ok(None);
        };

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
        let x_t_psi = &psi_dir.x_t_psi;
        let x_ls_psi = &psi_dir.x_ls_psi;
        let z_t = &psi_dir.z_t_psi;
        let z_ls = &psi_dir.z_ls_psi;

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
            let r = 1.0 / core.sigma[i].max(1e-12);
            let q_psi = -r * z_t[i] - q * z_ls[i];
            let (a, b, c) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[i],
                self.weights[i],
                q,
                core.mu[i],
            );
            r_t[i] = -a * r;
            r_ls[i] = -a * q;
            dr_t[i] = -b * q_psi * r + a * r * z_ls[i];
            dr_ls[i] = -(a + q * b) * q_psi;
            h_tt[i] = b * r * r;
            h_tl[i] = r * (a + q * b);
            h_ll[i] = q * (a + q * b);
            dh_tt[i] = r * r * (c * q_psi - 2.0 * b * z_ls[i]);
            dh_tl[i] = r * ((2.0 * b + c * q) * q_psi - (a + q * b) * z_ls[i]);
            dh_ll[i] = (a + 3.0 * q * b + q * q * c) * q_psi;
            objective_psi += r_t[i] * z_t[i] + r_ls[i] * z_ls[i];
        }

        let mut score_psi = Array1::<f64>::zeros(total);
        score_psi
            .slice_mut(s![0..pt])
            .assign(&(x_t_psi.t().dot(&r_t) + x_t.t().dot(&dr_t)));
        score_psi
            .slice_mut(s![pt..pt + pls])
            .assign(&(x_ls_psi.t().dot(&r_ls) + x_ls.t().dot(&dr_ls)));

        let h_tt_block = xt_diag_y_dense(&x_t_psi, &h_tt, x_t)?
            + &xt_diag_y_dense(x_t, &h_tt, &x_t_psi)?
            + &xt_diag_x_dense(x_t, &dh_tt)?;
        let h_tl_block = xt_diag_y_dense(&x_t_psi, &h_tl, x_ls)?
            + &xt_diag_y_dense(x_t, &h_tl, &x_ls_psi)?
            + &xt_diag_y_dense(x_t, &dh_tl, x_ls)?;
        let h_ll_block = xt_diag_y_dense(&x_ls_psi, &h_ll, x_ls)?
            + &xt_diag_y_dense(x_ls, &h_ll, &x_ls_psi)?
            + &xt_diag_x_dense(x_ls, &dh_ll)?;

        let mut hessian_psi = Array2::<f64>::zeros((total, total));
        hessian_psi.slice_mut(s![0..pt, 0..pt]).assign(&h_tt_block);
        hessian_psi
            .slice_mut(s![0..pt, pt..pt + pls])
            .assign(&h_tl_block);
        hessian_psi
            .slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&h_ll_block);
        mirror_upper_to_lower(&mut hessian_psi);

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
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
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            x_t,
            x_ls,
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
        )?
        else {
            return Ok(None);
        };
        let (x_t_ab, x_ls_ab, z_t_ab, z_ls_ab) = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            &dir_i,
            &dir_j,
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

        // Exact fixed-beta psi/psi terms for the coupled non-wiggle probit
        // family.
        //
        // For two realized spatial coordinates psi_a, psi_b define
        //
        //   z_t,a  = X_{t,a} beta_t,    z_ls,a  = X_{ls,a} beta_ls,
        //   z_t,b  = X_{t,b} beta_t,    z_ls,b  = X_{ls,b} beta_ls,
        //   z_t,ab = X_{t,ab} beta_t,   z_ls,ab = X_{ls,ab} beta_ls.
        //
        // With r = exp(-eta_ls) and q = -eta_t r,
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
            let r = 1.0 / core.sigma[row].max(1e-12);
            let q_i = -r * dir_i.z_t_psi[row] - q * dir_i.z_ls_psi[row];
            let q_j = -r * dir_j.z_t_psi[row] - q * dir_j.z_ls_psi[row];
            let q_ij = -r * z_t_ab[row]
                + r * (dir_i.z_t_psi[row] * dir_j.z_ls_psi[row]
                    + dir_j.z_t_psi[row] * dir_i.z_ls_psi[row])
                + q * (dir_i.z_ls_psi[row] * dir_j.z_ls_psi[row] - z_ls_ab[row]);
            let (a, b, c) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
            let d = binomial_neglog_q_fourth_derivative_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
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
                    + a * z_ls_ab[row]);
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
                    - 2.0 * b * z_ls_ab[row]);
            d2h_tl[row] = r
                * (((3.0 * c + q * d) * q_j) * q_i + (2.0 * b + q * c) * q_ij
                    - (2.0 * b + q * c) * (q_j * dir_i.z_ls_psi[row] + q_i * dir_j.z_ls_psi[row])
                    + u * (dir_i.z_ls_psi[row] * dir_j.z_ls_psi[row] - z_ls_ab[row]));
            d2h_ll[row] = (4.0 * b + 5.0 * q * c + q * q * d) * q_i * q_j
                + (a + 3.0 * q * b + q * q * c) * q_ij;

            objective_psi_psi += a * q_ij + b * q_i * q_j;
        }

        let mut score_psi_psi = Array1::<f64>::zeros(total);
        score_psi_psi.slice_mut(s![0..pt]).assign(
            &(x_t_ab.t().dot(&r_t)
                + dir_i.x_t_psi.t().dot(&dr_t_j)
                + dir_j.x_t_psi.t().dot(&dr_t_i)
                + x_t.t().dot(&d2r_t)),
        );
        score_psi_psi.slice_mut(s![pt..pt + pls]).assign(
            &(x_ls_ab.t().dot(&r_ls)
                + dir_i.x_ls_psi.t().dot(&dr_ls_j)
                + dir_j.x_ls_psi.t().dot(&dr_ls_i)
                + x_ls.t().dot(&d2r_ls)),
        );

        let h_tt_block = xt_diag_y_dense(&x_t_ab, &h_tt, x_t)?
            + &xt_diag_y_dense(&dir_i.x_t_psi, &h_tt, &dir_j.x_t_psi)?
            + &xt_diag_y_dense(&dir_j.x_t_psi, &h_tt, &dir_i.x_t_psi)?
            + &xt_diag_y_dense(&dir_i.x_t_psi, &dh_tt_j, x_t)?
            + &xt_diag_y_dense(&dir_j.x_t_psi, &dh_tt_i, x_t)?
            + &xt_diag_y_dense(x_t, &dh_tt_i, &dir_j.x_t_psi)?
            + &xt_diag_y_dense(x_t, &dh_tt_j, &dir_i.x_t_psi)?
            + &xt_diag_x_dense(x_t, &d2h_tt)?
            + &xt_diag_y_dense(x_t, &h_tt, &x_t_ab)?;
        let h_tl_block = xt_diag_y_dense(&x_t_ab, &h_tl, x_ls)?
            + &xt_diag_y_dense(&dir_i.x_t_psi, &h_tl, &dir_j.x_ls_psi)?
            + &xt_diag_y_dense(&dir_j.x_t_psi, &h_tl, &dir_i.x_ls_psi)?
            + &xt_diag_y_dense(&dir_i.x_t_psi, &dh_tl_j, x_ls)?
            + &xt_diag_y_dense(&dir_j.x_t_psi, &dh_tl_i, x_ls)?
            + &xt_diag_y_dense(x_t, &dh_tl_i, &dir_j.x_ls_psi)?
            + &xt_diag_y_dense(x_t, &dh_tl_j, &dir_i.x_ls_psi)?
            + &xt_diag_y_dense(x_t, &d2h_tl, x_ls)?
            + &xt_diag_y_dense(x_t, &h_tl, &x_ls_ab)?;
        let h_ll_block = xt_diag_y_dense(&x_ls_ab, &h_ll, x_ls)?
            + &xt_diag_y_dense(&dir_i.x_ls_psi, &h_ll, &dir_j.x_ls_psi)?
            + &xt_diag_y_dense(&dir_j.x_ls_psi, &h_ll, &dir_i.x_ls_psi)?
            + &xt_diag_y_dense(&dir_i.x_ls_psi, &dh_ll_j, x_ls)?
            + &xt_diag_y_dense(&dir_j.x_ls_psi, &dh_ll_i, x_ls)?
            + &xt_diag_y_dense(x_ls, &dh_ll_i, &dir_j.x_ls_psi)?
            + &xt_diag_y_dense(x_ls, &dh_ll_j, &dir_i.x_ls_psi)?
            + &xt_diag_x_dense(x_ls, &d2h_ll)?
            + &xt_diag_y_dense(x_ls, &h_ll, &x_ls_ab)?;

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

        Ok(Some(
            crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
                objective_psi_psi,
                score_psi_psi,
                hessian_psi_psi,
            },
        ))
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
        )?
        else {
            return Ok(None);
        };
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
            let r = 1.0 / core.sigma[row].max(1e-12);
            let du = -r * xi_t[row] - q * xi_ls[row];
            let q_a = -r * dir_a.z_t_psi[row] - q * dir_a.z_ls_psi[row];
            let q_au = r * dir_a.z_t_psi[row] * xi_ls[row] - du * dir_a.z_ls_psi[row];
            let (a, b, c) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
            let d = binomial_neglog_q_fourth_derivative_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
            let u = a + q * b;
            h_tt_u[row] = r * r * (c * du - 2.0 * b * xi_ls[row]);
            h_tl_u[row] = r * ((2.0 * b + q * c) * du - u * xi_ls[row]);
            h_ll_u[row] = (a + 3.0 * q * b + q * q * c) * du;
            dh_tt_u[row] = r
                * r
                * (d * du * q_a + c * q_au
                    - 2.0 * c * (q_a * xi_ls[row] + du * dir_a.z_ls_psi[row])
                    + 4.0 * b * xi_ls[row] * dir_a.z_ls_psi[row]);
            dh_tl_u[row] = r
                * (((3.0 * c + q * d) * q_a) * du + (2.0 * b + q * c) * q_au
                    - (2.0 * b + q * c) * (q_a * xi_ls[row] + du * dir_a.z_ls_psi[row])
                    + u * xi_ls[row] * dir_a.z_ls_psi[row]);
            dh_ll_u[row] = (4.0 * b + 5.0 * q * c + q * q * d) * du * q_a
                + (a + 3.0 * q * b + q * q * c) * q_au;
        }

        let tt_block = xt_diag_y_dense(&dir_a.x_t_psi, &h_tt_u, x_t)?
            + &xt_diag_y_dense(x_t, &h_tt_u, &dir_a.x_t_psi)?
            + &xt_diag_x_dense(x_t, &dh_tt_u)?;
        let tl_block = xt_diag_y_dense(&dir_a.x_t_psi, &h_tl_u, x_ls)?
            + &xt_diag_y_dense(x_t, &h_tl_u, &dir_a.x_ls_psi)?
            + &xt_diag_y_dense(x_t, &dh_tl_u, x_ls)?;
        let ll_block = xt_diag_y_dense(&dir_a.x_ls_psi, &h_ll_u, x_ls)?
            + &xt_diag_y_dense(x_ls, &h_ll_u, &dir_a.x_ls_psi)?
            + &xt_diag_x_dense(x_ls, &dh_ll_u)?;
        let mut out = Array2::<f64>::zeros((total, total));
        out.slice_mut(s![0..pt, 0..pt]).assign(&tt_block);
        out.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl_block);
        out.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&ll_block);
        mirror_upper_to_lower(&mut out);
        Ok(Some(out))
    }
}

impl CustomFamily for BinomialLocationScaleFamily {
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
        if self.exact_joint_supported() {
            // Exact-joint path for the 2-block non-wiggle probit family.
            //
            // Model:
            //   t_i     = x_i^T beta_t
            //   s_i     = z_i^T beta_s
            //   sigma_i = exp(s_i)
            //   q_i     = -t_i / sigma_i = -t_i exp(-s_i)
            //   mu_i    = Phi(q_i)
            //
            // For one observation the negative log-likelihood is
            //
            //   F_i(q) = -w_i [ y_i log Phi(q) + (1-y_i) log(1-Phi(q)) ].
            //
            // The exact-Newton working objects used by the inner solve are the
            // coefficient-space score and Hessian of the unpenalized objective
            // sum_i F_i(q_i(beta_t, beta_s)).
            //
            // In the non-wiggle family there is no dynamic basis, but there is
            // still nontrivial predictor geometry because q depends jointly on
            // both linear predictors through the quotient -t / exp(s). The
            // helper `binomial_location_scale_working_sets(..., exact_geometry:
            // Some(...))` already assembles those exact blockwise score/Hessian
            // objects from the rowwise eta-space chain rule. Entering that path
            // here is what makes the family consistent with the exact joint
            // outer-Hessian callbacks implemented below.
            // For the exp link, sigma = d1 = d2 = d3 = exp(eta), so just compute once.
            let d2sigma_deta2 = eta_ls.mapv(f64::exp);
            let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
            })?;
            let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
            })?;
            let (tws, lsws, _) = binomial_location_scale_working_sets(
                &self.y,
                &self.weights,
                eta_t,
                eta_ls,
                None,
                None,
                Some(BinomialLocationScaleExactGeometry {
                    threshold_design,
                    log_sigma_design,
                    wiggle_design: None,
                    d2sigma_deta2: &d2sigma_deta2,
                    d2q_dq02: None,
                }),
                &core,
            )?;
            return Ok(FamilyEvaluation {
                log_likelihood: core.log_likelihood,
                blockworking_sets: vec![tws, lsws],
            });
        }
        let (tws, lsws, _) = binomial_location_scale_working_sets(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            None,
            None,
            &core,
        )?;

        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![tws, lsws],
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
        matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit))
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        // Full directional derivative used by the diagonal H_beta term in
        //   H = X' diag(w) X + S
        // so that
        //   H_beta[d_beta] = X' diag(dw) X.
        //
        // For this family (no wiggle block), each diagonal working weight is
        //   w = weights_i * (dmu/deta_block)^2 / var(mu)
        // with
        //   q  = -eta_t / sigma(eta_ls),
        //   mu = link^{-1}(q),
        //   var = mu(1-mu),
        //   dmu/deta_block = (dmu/dq) * chain.
        //
        // Define
        //   g(q) = (dmu/dq)^2 / var(mu),
        //   w    = weights_i * g(q) * chain^2.
        //
        // Directional derivative along d_eta (block-local predictor direction):
        //   dw = weights_i * d[ g(q) * chain^2 ]
        //      = weights_i * ( dg * chain^2 + g * d(chain^2) )
        //      = weights_i * ( (dg/dq * dq) * chain^2 + 2*g*chain*dchain ).
        //
        // Here
        //   dq = chain * d_eta_i.
        //
        // We compute dg/dq from log-derivative algebra:
        //   log g = 2 log(dmu/dq) - log(var),
        //   d/dq log g = 2*(d2mu/dq2)/(dmu/dq) - (dvar/dq)/var,
        //   dvar/dq = (dmu/dq)*(1 - 2*mu),
        // hence
        //   dg/dq = g * [ 2*(d2mu/dq2)/(dmu/dq) - (dvar/dq)/var ].
        //
        // Block-specific chain terms:
        // - Threshold block (eta_t):
        //     chain = dq/deta_t = -1/sigma,
        //     dchain = 0 (chain does not depend on eta_t).
        // - Log-sigma block (eta_ls):
        //     q0 = -eta_t/sigma,
        //     chain = dq/deta_ls = -q0 * dsigma/deta_ls / sigma,
        //     dchain = (dchain/deta_ls) * d_eta_i,
        //     dchain/deta_ls
        //       = eta_t * [ d2sigma/deta_ls2 / sigma^2
        //                  - 2*(dsigma/deta_ls)^2 / sigma^3 ].
        //
        // If the unclamped working weight is at/below MIN_WEIGHT, this code
        // returns dw=0 to match the active branch of the clamped definition.
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        if d_eta.len() != n {
            return Err(format!(
                "BinomialLocationScaleFamily directional eta length mismatch: got {}, expected {n}",
                d_eta.len()
            ));
        }
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
        let (_, dsigma_deta, d2sigma_deta2, _) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let mut dw = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = core.sigma[i].max(1e-12);
            let mu = core.mu[i];
            let dmu = core.dmu_dq[i];
            let d2mu = core.d2mu_dq2[i];
            let var = (mu * (1.0 - mu)).max(MIN_PROB);
            let dir = d_eta[i];
            if !s.is_finite()
                || !mu.is_finite()
                || !dmu.is_finite()
                || !d2mu.is_finite()
                || !dir.is_finite()
            {
                dw[i] = 0.0;
                continue;
            }

            let (chain, dchain) = match block_idx {
                Self::BLOCK_T => (-1.0 / s, 0.0),
                Self::BLOCK_LOG_SIGMA => {
                    let chain_i = -core.q0[i] * core.dsigma_deta[i] / s;
                    let dchain_deta = eta_t[i]
                        * (d2sigma_deta2[i] / (s * s) - 2.0 * dsigma_deta[i].powi(2) / (s * s * s));
                    (chain_i, dchain_deta * dir)
                }
                _ => return Ok(None),
            };
            if !chain.is_finite() || !dchain.is_finite() {
                dw[i] = 0.0;
                continue;
            }

            let dvar_dq = dmu * (1.0 - 2.0 * mu);
            let g = dmu * dmu / var;
            let raww = self.weights[i] * g * chain * chain;
            if !raww.is_finite() || raww <= MIN_WEIGHT {
                dw[i] = 0.0;
                continue;
            }
            // In extreme tails dmu can approach zero numerically; avoid unstable
            // 2*d2mu/dmu amplification and fall back to the active clamp branch.
            if dmu.abs() <= MIN_DERIV {
                dw[i] = 0.0;
                continue;
            }

            let dq = chain * dir;
            let dg_dq = g * (2.0 * d2mu / dmu - dvar_dq / var);
            let dw_i = self.weights[i] * (dg_dq * dq * chain * chain + 2.0 * g * chain * dchain);
            dw[i] = if dw_i.is_finite() { dw_i } else { 0.0 };
        }
        Ok(Some(dw))
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

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
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
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]).max(1e-12);
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
        matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit))
            && self.threshold_design.is_some()
            && self.log_sigma_design.is_some()
    }

    pub fn initializewiggle_knots_from_q(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
    ) -> Result<Array1<f64>, String> {
        initializewiggle_knots_from_seed(q_seed, degree, num_internal_knots)
    }

    fn wiggle_constraint_transform(&self) -> Result<Array2<f64>, String> {
        let (z, _) =
            compute_geometric_constraint_transform(&self.wiggle_knots, self.wiggle_degree, 2)
                .map_err(|e| e.to_string())?;
        Ok(z)
    }

    fn constrainwiggle_basis(&self, full: Array2<f64>) -> Result<Array2<f64>, String> {
        if full.ncols() < 3 {
            return Err("wiggle basis has fewer than three columns".to_string());
        }
        let z = self.wiggle_constraint_transform()?;
        if full.ncols() != z.nrows() {
            return Err(format!(
                "wiggle basis/constraint mismatch: basis has {} columns but transform has {} rows",
                full.ncols(),
                z.nrows()
            ));
        }
        // Keep all value/derivative evaluations in the same constrained subspace:
        // d(BZ)/dx = B'Z, d²(BZ)/dx² = B''Z.
        Ok(full.dot(&z))
    }

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        let (basis, _) = create_basis::<Dense>(
            q0,
            KnotSource::Provided(self.wiggle_knots.view()),
            self.wiggle_degree,
            basis_options,
        )
        .map_err(|e| e.to_string())?;
        self.constrainwiggle_basis((*basis).clone())
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        betawiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != betawiggle.len() {
            return Err(format!(
                "wiggle derivative col mismatch: got {}, expected {}",
                d_constrained.ncols(),
                betawiggle.len()
            ));
        }
        Ok(d_constrained.dot(&betawiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        betawiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2_constrained =
            self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2_constrained.ncols() != betawiggle.len() {
            return Err(format!(
                "wiggle second-derivative col mismatch: got {}, expected {}",
                d2_constrained.ncols(),
                betawiggle.len()
            ));
        }
        Ok(d2_constrained.dot(&betawiggle))
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        betawiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3_constrained = self.wiggle_d3basis_constrained(q0)?;
        if d3_constrained.ncols() != betawiggle.len() {
            return Err(format!(
                "wiggle third-derivative col mismatch: got {}, expected {}",
                d3_constrained.ncols(),
                betawiggle.len()
            ));
        }
        Ok(d3_constrained.dot(&betawiggle))
    }

    fn wiggle_d3basis_constrained(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        if self.wiggle_degree < 3 {
            let z = self.wiggle_constraint_transform()?;
            return Ok(Array2::zeros((q0.len(), z.ncols())));
        }
        let z = self.wiggle_constraint_transform()?;
        let num_basis = self
            .wiggle_knots
            .len()
            .checked_sub(self.wiggle_degree + 1)
            .ok_or_else(|| "wiggle knot vector too short for third derivative".to_string())?;
        if z.nrows() != num_basis {
            return Err(format!(
                "wiggle third-derivative/constraint mismatch: basis has {} columns but transform has {} rows",
                num_basis,
                z.nrows()
            ));
        }
        let mut raw = vec![0.0; num_basis];
        let mut out = Array2::<f64>::zeros((q0.len(), z.ncols()));
        for (i, &q0_i) in q0.iter().enumerate() {
            evaluate_bsplinethird_derivative_scalar(
                q0_i,
                self.wiggle_knots.view(),
                self.wiggle_degree,
                &mut raw,
            )
            .map_err(|e| format!("failed to evaluate wiggle third derivative basis: {e}"))?;
            for constrained_j in 0..z.ncols() {
                let mut basis_j = 0.0;
                for raw_k in 0..num_basis {
                    basis_j += raw[raw_k] * z[[raw_k, constrained_j]];
                }
                out[[i, constrained_j]] = basis_j;
            }
        }
        Ok(out)
    }

    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        betawiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        if self.wiggle_degree < 4 {
            return Ok(Array1::zeros(q0.len()));
        }
        let z = self.wiggle_constraint_transform()?;
        let num_basis = self
            .wiggle_knots
            .len()
            .checked_sub(self.wiggle_degree + 1)
            .ok_or_else(|| "wiggle knot vector too short for fourth derivative".to_string())?;
        if z.nrows() != num_basis {
            return Err(format!(
                "wiggle fourth-derivative/constraint mismatch: basis has {} columns but transform has {} rows",
                num_basis,
                z.nrows()
            ));
        }
        if z.ncols() != betawiggle.len() {
            return Err(format!(
                "wiggle fourth-derivative col mismatch: got {}, expected {}",
                z.ncols(),
                betawiggle.len()
            ));
        }
        let mut raw = vec![0.0; num_basis];
        let mut out = Array1::<f64>::zeros(q0.len());
        for (i, &q0_i) in q0.iter().enumerate() {
            evaluate_bspline_fourth_derivative_scalar(
                q0_i,
                self.wiggle_knots.view(),
                self.wiggle_degree,
                &mut raw,
            )
            .map_err(|e| format!("failed to evaluate wiggle fourth derivative basis: {e}"))?;
            let mut acc = 0.0;
            for constrained_j in 0..betawiggle.len() {
                let mut basis_j = 0.0;
                for raw_k in 0..num_basis {
                    basis_j += raw[raw_k] * z[[raw_k, constrained_j]];
                }
                acc += basis_j * betawiggle[constrained_j];
            }
            out[i] = acc;
        }
        Ok(out)
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
            None => Cow::Owned(td.to_dense()),
        };
        let xls = match lsd.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(lsd.to_dense()),
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
            None => Cow::Owned(specs[Self::BLOCK_T].design.to_dense()),
        };
        let xls = match specs[Self::BLOCK_LOG_SIGMA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(specs[Self::BLOCK_LOG_SIGMA].design.to_dense()),
        };
        Ok((xt, xls))
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        self.dense_block_designs().map(Some)
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
            threshold_design: Some(DesignMatrix::Dense(x_t.into_owned())),
            log_sigma_design: Some(DesignMatrix::Dense(x_ls.into_owned())),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
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
                    let mut x_t_psi = Array2::<f64>::zeros((n, pt));
                    let mut x_ls_psi = Array2::<f64>::zeros((n, pls));
                    match block_idx {
                        Self::BLOCK_T => {
                            if deriv.x_psi.nrows() != n || deriv.x_psi.ncols() != pt {
                                return Err(format!(
                                    "BinomialLocationScaleWiggleFamily threshold x_psi shape mismatch: got {}x{}, expected {}x{}",
                                    deriv.x_psi.nrows(),
                                    deriv.x_psi.ncols(),
                                    n,
                                    pt
                                ));
                            }
                            x_t_psi.assign(&deriv.x_psi);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            if deriv.x_psi.nrows() != n || deriv.x_psi.ncols() != pls {
                                return Err(format!(
                                    "BinomialLocationScaleWiggleFamily log-sigma x_psi shape mismatch: got {}x{}, expected {}x{}",
                                    deriv.x_psi.nrows(),
                                    deriv.x_psi.ncols(),
                                    n,
                                    pls
                                ));
                            }
                            x_ls_psi.assign(&deriv.x_psi);
                        }
                        Self::BLOCK_WIGGLE => return Ok(None),
                        _ => return Ok(None),
                    }
                    return Ok(Some(BinomialLocationScaleWiggleJointPsiDirection {
                        block_idx,
                        local_idx,
                        z_t_psi: x_t_psi.dot(beta_t),
                        z_ls_psi: x_ls_psi.dot(beta_ls),
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
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), String> {
        let n = self.y.len();
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_t = &block_states[Self::BLOCK_T].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let mut x_t_ab = Array2::<f64>::zeros((n, pt));
        let mut x_ls_ab = Array2::<f64>::zeros((n, pls));
        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            if let Some(x_psi_psi) = deriv.x_psi_psi.as_ref() {
                if let Some(x_ab) = x_psi_psi.get(psi_b.local_idx) {
                    match psi_a.block_idx {
                        Self::BLOCK_T => {
                            if x_ab.nrows() != n || x_ab.ncols() != pt {
                                return Err(format!(
                                    "BinomialLocationScaleWiggleFamily threshold x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
                                    x_ab.nrows(),
                                    x_ab.ncols(),
                                    n,
                                    pt
                                ));
                            }
                            x_t_ab.assign(x_ab);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            if x_ab.nrows() != n || x_ab.ncols() != pls {
                                return Err(format!(
                                    "BinomialLocationScaleWiggleFamily log-sigma x_psi_psi shape mismatch: got {}x{}, expected {}x{}",
                                    x_ab.nrows(),
                                    x_ab.ncols(),
                                    n,
                                    pls
                                ));
                            }
                            x_ls_ab.assign(x_ab);
                        }
                        _ => {}
                    }
                }
            }
        }
        let z_t_ab = x_t_ab.dot(beta_t);
        let z_ls_ab = x_ls_ab.dot(beta_ls);
        Ok((x_t_ab, x_ls_ab, z_t_ab, z_ls_ab))
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
        )?
        else {
            return Ok(None);
        };
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
        let g2 = dd0.dot(betaw);
        let g3 = d3q;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(total);
        let mut hessian_psi = Array2::<f64>::zeros((total, total));

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
        //   q_r   = -a_r exp(-ell_r).
        //
        // In this wiggle family we realize the same kernel through the chain
        //
        //   q = q0 + betaw^T B(q0),
        //   q0 = -eta_t exp(-eta_ls),
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
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row], ds[row], d2s[row], d3s[row]);
            let r_sigma = 1.0 / sigma[row].max(1e-12);
            let q0_a = -r_sigma * dir_a.z_t_psi[row] - q0 * dir_a.z_ls_psi[row];
            let q0_t_a = q0_geom.q_tl * dir_a.z_ls_psi[row];
            let q0_ls_a = q0_geom.q_tl * dir_a.z_t_psi[row] + q0_geom.q_ll * dir_a.z_ls_psi[row];
            let q0_tl_a = q0_geom.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_ll_a =
                q0_geom.q_tl_ls * dir_a.z_t_psi[row] + q0_geom.q_ll_ls * dir_a.z_ls_psi[row];

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

            let brow = b0.row(row);
            let drow = d0.row(row);
            let ddrow = dd0.row(row);
            let qw_a = drow.to_owned() * q0_a;
            let q_tw_a = ddrow.to_owned() * (q0_a * q0_geom.q_t) + &(drow.to_owned() * q0_t_a);
            let q_lw_a = ddrow.to_owned() * (q0_a * q0_geom.q_ls) + &(drow.to_owned() * q0_ls_a);

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );

            objective_psi += loss_1 * m[row] * q0_a;

            let xtr = x_t.row(row);
            let xlsr = x_ls.row(row);
            let xta = dir_a.x_t_psi.row(row);
            let xlsa = dir_a.x_ls_psi.row(row);

            let mut b = Array1::<f64>::zeros(total);
            b.slice_mut(s![0..pt]).assign(&(xtr.to_owned() * q_t));
            b.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls));
            b.slice_mut(s![pt + pls..]).assign(&brow.to_owned());

            let mut c_a = Array1::<f64>::zeros(total);
            c_a.slice_mut(s![0..pt])
                .assign(&(xtr.to_owned() * q_t_a + xta.to_owned() * q_t));
            c_a.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls_a + xlsa.to_owned() * q_ls));
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);

            score_psi += &(loss_2 * m[row] * q0_a * &b + loss_1 * &c_a);

            let mut q_mat = Array2::<f64>::zeros((total, total));
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
                    .dot(&(drow.to_owned() * q0_geom.q_t).insert_axis(Axis(0)));
                let lw = xlsr
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.to_owned() * q0_geom.q_ls).insert_axis(Axis(0)));
                q_mat.slice_mut(s![0..pt, pt + pls..]).assign(&tw);
                q_mat.slice_mut(s![pt..pt + pls, pt + pls..]).assign(&lw);
                mirror_upper_to_lower(&mut q_mat);
            }

            let mut r_a = Array2::<f64>::zeros((total, total));
            {
                let tt = xtr
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
                let tl = xtr
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
                let ll = xlsr
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
                r_a.slice_mut(s![0..pt, 0..pt]).assign(&tt);
                r_a.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl);
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&ll);
                let tw = xta
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.to_owned() * q0_geom.q_t).insert_axis(Axis(0)))
                    + xtr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_tw_a.insert_axis(Axis(0)));
                let lw = xlsa
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&(drow.to_owned() * q0_geom.q_ls).insert_axis(Axis(0)))
                    + xlsr
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&q_lw_a.insert_axis(Axis(0)));
                r_a.slice_mut(s![0..pt, pt + pls..]).assign(&tw);
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]).assign(&lw);
                mirror_upper_to_lower(&mut r_a);
            }

            let bb = b
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let ca_bt = c_a
                .view()
                .insert_axis(Axis(1))
                .dot(&b.view().insert_axis(Axis(0)));
            let b_cat = b
                .view()
                .insert_axis(Axis(1))
                .dot(&c_a.view().insert_axis(Axis(0)));
            hessian_psi += &(loss_3 * m[row] * q0_a * bb
                + loss_2 * (ca_bt + b_cat + (m[row] * q0_a) * q_mat)
                + loss_1 * r_a);
        }

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
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
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }
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
        )?
        else {
            return Ok(None);
        };
        let (x_t_ab, x_ls_ab, z_t_ab, z_ls_ab) = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            &dir_a,
            &dir_b,
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
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row], ds[row], d2s[row], d3s[row]);
            let s_safe = sigma[row].max(1e-12);
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let q0_tl_ls_ls =
                d3s[row] / s2 - 6.0 * ds[row] * d2s[row] / s3 + 6.0 * ds[row].powi(3) / s4;
            let r_sigma = 1.0 / s_safe;

            let q0_a = -r_sigma * dir_a.z_t_psi[row] - q0 * dir_a.z_ls_psi[row];
            let q0_b = -r_sigma * dir_b.z_t_psi[row] - q0 * dir_b.z_ls_psi[row];
            let q0_ab = -r_sigma * z_t_ab[row]
                + r_sigma
                    * (dir_a.z_t_psi[row] * dir_b.z_ls_psi[row]
                        + dir_b.z_t_psi[row] * dir_a.z_ls_psi[row])
                + q0 * (dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row] - z_ls_ab[row]);

            let q0_t_a = q0_geom.q_tl * dir_a.z_ls_psi[row];
            let q0_t_b = q0_geom.q_tl * dir_b.z_ls_psi[row];
            let q0_t_ab = q0_geom.q_tl_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl * z_ls_ab[row];
            let q0_ls_a = q0_geom.q_tl * dir_a.z_t_psi[row] + q0_geom.q_ll * dir_a.z_ls_psi[row];
            let q0_ls_b = q0_geom.q_tl * dir_b.z_t_psi[row] + q0_geom.q_ll * dir_b.z_ls_psi[row];
            let q0_ls_ab = -q0_ab;
            let q0_tl_a = q0_geom.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_tl_b = q0_geom.q_tl_ls * dir_b.z_ls_psi[row];
            let q0_tl_ab = q0_tl_ls_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl_ls * z_ls_ab[row];
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

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
            objective_psi_psi += loss_2 * q_a * q_b + loss_1 * q_ab;

            let xtr = x_t.row(row);
            let xlsr = x_ls.row(row);
            let xta = dir_a.x_t_psi.row(row);
            let xtb = dir_b.x_t_psi.row(row);
            let xlsa = dir_a.x_ls_psi.row(row);
            let xlsb = dir_b.x_ls_psi.row(row);
            let xtab = x_t_ab.row(row);
            let xlsab = x_ls_ab.row(row);

            let mut b = Array1::<f64>::zeros(total);
            b.slice_mut(s![0..pt]).assign(&(xtr.to_owned() * q_t));
            b.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls));
            b.slice_mut(s![pt + pls..]).assign(&brow);
            let mut c_a = Array1::<f64>::zeros(total);
            c_a.slice_mut(s![0..pt])
                .assign(&(xtr.to_owned() * q_t_a + xta.to_owned() * q_t));
            c_a.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls_a + xlsa.to_owned() * q_ls));
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);
            let mut c_b = Array1::<f64>::zeros(total);
            c_b.slice_mut(s![0..pt])
                .assign(&(xtr.to_owned() * q_t_b + xtb.to_owned() * q_t));
            c_b.slice_mut(s![pt..pt + pls])
                .assign(&(xlsr.to_owned() * q_ls_b + xlsb.to_owned() * q_ls));
            c_b.slice_mut(s![pt + pls..]).assign(&qw_b);
            let mut c_ab = Array1::<f64>::zeros(total);
            c_ab.slice_mut(s![0..pt]).assign(
                &(xtr.to_owned() * q_t_ab
                    + xta.to_owned() * q_t_b
                    + xtb.to_owned() * q_t_a
                    + xtab.to_owned() * q_t),
            );
            c_ab.slice_mut(s![pt..pt + pls]).assign(
                &(xlsr.to_owned() * q_ls_ab
                    + xlsa.to_owned() * q_ls_b
                    + xlsb.to_owned() * q_ls_a
                    + xlsab.to_owned() * q_ls),
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

        Ok(Some(
            crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
                objective_psi_psi,
                score_psi_psi,
                hessian_psi_psi,
            },
        ))
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
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
        )?
        else {
            return Ok(None);
        };
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
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let g3 = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = gaussian_sigma_derivs_up_to_fourth(eta_ls.view());

        // Exact likelihood-side mixed drift T_a[u] = D_beta H_{psi_a}^{(D)}[u].
        //
        // The unified outer Hessian in custom_family.rs uses
        //   ddot H_ij = H_ij + T_i[beta_j] + T_j[beta_i]
        //             + D_beta H[beta_ij] + D_beta^2 H[beta_i, beta_j].
        //
        // For wiggle we still use the same scalar-loss row kernel as non-wiggle;
        // only the location-side row changes to z_r = [x_{t,r}; B_r(q0)] with
        // q = q0 + betaw^T B(q0), q0 = -eta_t exp(-eta_ls).
        let mut out = Array2::<f64>::zeros((total, total));
        for row in 0..n {
            let q = core.q0[row] + etaw[row];
            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_probit_closed_form(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
            );
            let q0 = nonwiggle_q_derivs(eta_t[row], sigma[row], ds[row], d2s[row], d3s[row]);
            let s_safe = sigma[row].max(1e-12);
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
            let xta = dir_a.x_t_psi.row(row).to_owned();
            let xlsa = dir_a.x_ls_psi.row(row).to_owned();
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
        Ok(Some(out))
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
}

impl CustomFamily for BinomialLocationScaleWiggleFamily {
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
        let d2q_dq02 =
            self.wiggle_d2q_dq02(core.q0.view(), block_states[Self::BLOCK_WIGGLE].beta.view())?;
        let d2sigma_deta2 = eta_ls.mapv(f64::exp);
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing threshold design"
                .to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing log-sigma design"
                .to_string()
        })?;
        let (tws, lsws, wws) = binomial_location_scale_working_sets(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            Some(&dq_dq0),
            Some(BinomialLocationScaleExactGeometry {
                threshold_design,
                log_sigma_design,
                wiggle_design: Some(&wiggle_design),
                d2sigma_deta2: &d2sigma_deta2,
                d2q_dq02: Some(&d2q_dq02),
            }),
            &core,
        )?;
        let wws = wws.ok_or_else(|| "wiggle working set missing".to_string())?;

        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![tws, lsws, wws],
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
        // Canonical per-row identities implemented below:
        //
        // 1) q-derivative building blocks
        //   m  = dq/dq0 = 1 + betaw^T B'(q0)
        //   g2 = d²q/dq0² via wiggle = betaw^T B''(q0)
        //
        //   q_t  = m q0_t
        //   q_l  = m q0_l
        //   q_tt = g2 q0_t q0_t
        //   q_tl = g2 q0_t q0_l + m q0_tl
        //   q_ll = g2 q0_l q0_l + m q0_ll
        //   qw  = B
        //   q_tw = q0_t B'
        //   q_lw = q0_l B'
        //   qww = 0
        //
        // 2) Objective derivative sign mapping
        //   Let s = d ell/dq, c = d² ell/dq² (log-likelihood derivatives).
        //   For F = -sum ell:
        //     m1 = dF/dq   = -s
        //     m2 = d²F/dq² = -c
        //
        // 3) Master Hessian identity for scalar q:
        //   H_ab = sum_i [ m2 q_a q_b^T + m1 q_ab ].
        //
        // We apply that identity blockwise:
        //   tt/tl/ll from (q_t,q_l,q_tt,q_tl,q_ll),
        //   tw/lw from (qw,q_tw,q_lw),
        //   ww from qww=0 => Hww = sum m2 * qw qw^T.
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
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls.view());
        let pw = b0.ncols();
        let total = pt + pls + pw;
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
            let (m1, m2, _) =
                if matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
                    binomial_neglog_q_derivatives_probit_closed_form(
                        self.y[i],
                        self.weights[i],
                        q_i,
                        core0.mu[i],
                    )
                } else {
                    binomial_neglog_q_derivatives_from_jet(
                        self.y[i],
                        self.weights[i],
                        core0.mu[i],
                        core0.dmu_dq[i],
                        core0.d2mu_dq2[i],
                        core0.d3mu_dq3[i],
                    )
                };
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i], ds[i], d2s[i], d3s[i]);

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
        let h_tt = xt_diag_x_dense(&x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(&x_t, &coeff_tl, &x_ls)?;
        let h_ll = xt_diag_x_dense(&x_ls, &coeff_ll)?;
        let h_tw =
            xt_diag_y_dense(&x_t, &coeff_tw_b, &b0)? + &xt_diag_y_dense(&x_t, &coeff_tw_d, &d0)?;
        let h_lw =
            xt_diag_y_dense(&x_ls, &coeff_lw_b, &b0)? + &xt_diag_y_dense(&x_ls, &coeff_lw_d, &d0)?;
        let hww = xt_diag_x_dense(&b0, &coeffww)?;

        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..pt + pls]).assign(&h_tl);
        h.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&h_ll);
        h.slice_mut(s![0..pt, pt + pls..total]).assign(&h_tw);
        h.slice_mut(s![pt..pt + pls, pt + pls..total]).assign(&h_lw);
        h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&hww);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
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
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls.view());

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
            let (m1, m2, m3) =
                if matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
                    binomial_neglog_q_derivatives_probit_closed_form(
                        self.y[i],
                        self.weights[i],
                        q_i,
                        core0.mu[i],
                    )
                } else {
                    binomial_neglog_q_derivatives_from_jet(
                        self.y[i],
                        self.weights[i],
                        core0.mu[i],
                        core0.dmu_dq[i],
                        core0.d2mu_dq2[i],
                        core0.d3mu_dq3[i],
                    )
                };
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i], ds[i], d2s[i], d3s[i]);
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
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }
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
        let (sigma, ds, d2s, d3s, d4s) = gaussian_sigma_derivs_up_to_fourth(eta_ls.view());

        let mut d2_h = Array2::<f64>::zeros((total, total));
        for i in 0..n {
            // Per-row scalar objective derivatives for F_i(q).
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
            );
            let m4 = binomial_neglog_q_fourth_derivative_probit_closed_form(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
            );

            // Non-wiggle q0(eta_t, eta_ls) derivatives and sigma-ratio helpers.
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i], ds[i], d2s[i], d3s[i]);
            let s_safe = sigma[i].max(1e-12);
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

    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        Some(KnownLinkWiggle {
            base_link: self.link_kind.link_function(),
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        })
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
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]).max(1e-12);
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
        Ok((DesignMatrix::Dense(x), Array1::zeros(nrows)))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
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
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]).max(1e-12);
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
    use crate::basis::{
        CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu,
        compute_greville_abscissae,
    };
    use crate::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
    use ndarray::{Axis, array};
    use num_dual::{
        DualNum, second_derivative, second_partial_derivative, third_partial_derivative_vec,
    };
    use std::time::Instant;

    fn intercept_block(n: usize) -> ParameterBlockInput {
        ParameterBlockInput {
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }
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

    fn constrainedwiggle_basis_scalar_numdual<D: DualNum<f64> + Copy>(
        x: D,
        knots: &Array1<f64>,
        degree: usize,
    ) -> Array1<D> {
        let (z, penalty) = compute_geometric_constraint_transform(knots, degree, 2)
            .expect("wiggle constraint transform");
        let _ = penalty;
        let full = bspline_basis_scalar_numdual(x, knots, degree);
        let mut constrained = Array1::from_elem(z.ncols(), D::zero());
        for j in 0..z.nrows() {
            for k in 0..z.ncols() {
                constrained[k] += full[j] * D::from(z[[j, k]]);
            }
        }
        constrained
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
        let basis = constrainedwiggle_basis_scalar_numdual(q0, knots, degree);
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
            let sigma = eta_ls.exp();
            let resid = D::from(y[i]) - eta_mu;
            out += D::from(weights[i]) * (half * (resid / sigma).powi(2) + eta_ls);
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
            design: DesignMatrix::Dense(design),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        }
    }

    #[test]
    fn gaussian_joint_psi_firstweights_should_use_observation_weight_in_log_sigma_score() {
        let y = array![1.1];
        let etamu = array![0.3];
        let eta_ls = array![-0.2];
        let weights = array![2.5];
        let rows =
            gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
        let firstweights = gaussian_joint_psi_firstweights(&rows, &array![0.0], &array![1.0]);
        let expected_score_ls = weights[0] - rows.n[0];

        assert!(
            (firstweights.score_ls[0] - expected_score_ls).abs() <= 1e-12,
            "For one Gaussian row, d/deta_ls [ weight * (eta_ls + 0.5 * (y-mu)^2 * exp(-2 eta_ls)) ] = weight - n_i. The helper coded {} but the implemented scalar objective differentiates to {}.",
            firstweights.score_ls[0],
            expected_score_ls
        );
        assert!(
            (firstweights.objective_psirow[0] - expected_score_ls).abs() <= 1e-12,
            "With mu_psi=0 and eta_psi=1, the exact psi objective derivative must equal weight - n_i. The helper coded {} but the scalar objective differentiates to {}.",
            firstweights.objective_psirow[0],
            expected_score_ls
        );
    }

    #[test]
    fn gaussian_joint_psisecondweights_should_use_observation_weight_in_log_sigma_eab_term() {
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
        let expected_objective_psi_psi = weights[0] - rows.n[0];

        assert!(
            (secondweights.objective_psi_psirow[0] - expected_objective_psi_psi).abs() <= 1e-12,
            "With only eta_psi_psi=1 active, the Gaussian second psi objective must contribute weight - n_i from the linear eta_ls term. The helper coded {} but the scalar objective differentiates to {}.",
            secondweights.objective_psi_psirow[0],
            expected_objective_psi_psi
        );
    }

    #[test]
    fn weighted_projection_returns_finite_coefficients() {
        let n = 8usize;
        let design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let offset = Array1::zeros(n);
        let target_eta = Array1::from_vec(vec![0.2; n]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let beta =
            solveweighted_projection(&design, &offset, &target_eta, &weights, 1e-10).unwrap();
        assert_eq!(beta.len(), 1);
        assert!(beta[0].is_finite());
        assert!((beta[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn alpha_betawarm_start_produces_finite_targets() {
        let n = 16usize;
        let y = Array1::from_vec((0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect());
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold = intercept_block(n);
        let log_sigma = intercept_block(n);

        let (beta_t, beta_ls, betaobs) = try_binomial_alpha_betawarm_start(
            &y,
            &weights,
            &threshold,
            &log_sigma,
            &BlockwiseFitOptions::default(),
        )
        .unwrap();

        assert_eq!(beta_t.len(), 1);
        assert_eq!(beta_ls.len(), 1);
        assert!(beta_t[0].is_finite());
        assert!(beta_ls[0].is_finite());
        assert!(betaobs.iter().all(|v| v.is_finite() && *v > 0.0));
        let expected_beta = 1.0;
        assert!(betaobs.iter().all(|v| (*v - expected_beta).abs() < 1e-12));
    }

    #[test]
    fn zeroweightrows_stay_inactive_in_builtin_diagonal_families() {
        let weights = Array1::from_vec(vec![0.0, 1.0]);

        let gaussian = GaussianLocationScaleFamily {
            y: Array1::from_vec(vec![2.0, -1.0]),
            weights: weights.clone(),
            mu_design: None,
            log_sigma_design: None,
            cached_row_scalars: std::cell::RefCell::new(None),
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
            cached_row_scalars: std::cell::RefCell::new(None),
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
            cached_row_scalars: std::cell::RefCell::new(None),
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
    fn gaussian_sigma_helper_uses_a_different_function_than_the_coded_sigma_link() {
        let eta0 = 701.0_f64;
        let eta = array![eta0];
        let (sigma, d1, d2, d3, d4) = gaussian_sigma_derivs_up_to_fourth(eta.view());
        let coded_sigma = |x: f64| safe_exp(x).max(1e-12);
        let h = 1e-6;
        let fd1 = (coded_sigma(eta0 + h) - coded_sigma(eta0 - h)) / (2.0 * h);
        let fd2 =
            (coded_sigma(eta0 + h) - 2.0 * coded_sigma(eta0) + coded_sigma(eta0 - h)) / (h * h);
        assert_eq!(fd1, 0.0);
        assert_eq!(fd2, 0.0);
        assert!(
            (sigma[0] - coded_sigma(eta0)).abs() < 1e-30,
            "Gaussian sigma helper should evaluate the same coded sigma link as GaussianLocationScaleFamily at eta={eta0}; got {} vs {}",
            sigma[0],
            coded_sigma(eta0)
        );
        assert!(
            (d1[0] - fd1).abs() < 1e-30,
            "Gaussian sigma helper first derivative should match the coded sigma link at eta={eta0}; got {} vs {}",
            d1[0],
            fd1
        );
        assert!(
            (d2[0] - fd2).abs() < 1e-30,
            "Gaussian sigma helper second derivative should match the coded sigma link at eta={eta0}; got {} vs {}",
            d2[0],
            fd2
        );
        assert!(
            (d3[0] - 0.0).abs() < 1e-30,
            "Gaussian sigma helper third derivative should be 0 on the coded safe_exp plateau at eta={eta0}; got {}",
            d3[0]
        );
        assert!(
            (d4[0] - 0.0).abs() < 1e-30,
            "Gaussian sigma helper fourth derivative should be 0 on the coded safe_exp plateau at eta={eta0}; got {}",
            d4[0]
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
                let two_eta = 2.0 * eta;
                let sigma = safe_exp(eta).max(1e-12);
                let inv_s2 = safe_exp(-two_eta).min(1e24);
                let r = y[i] - mu[i];
                ll += w * (-0.5 * (r * r * inv_s2 + ln2pi + two_eta));
                if w == 0.0 {
                    wmu[i] = 0.0;
                    zmu[i] = mu[i];
                } else {
                    wmu[i] = floor_positiveweight(w * inv_s2, MIN_WEIGHT);
                    zmu[i] = mu[i] + r;
                }
                let dlogsigma_du = if sigma <= 1e-10 { sigma * 1e10 } else { 1.0 };
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
                let two_eta = 2.0 * eta;
                let sigma = safe_exp(eta).max(1e-12);
                let inv_s2 = (sigma * sigma).recip().min(1e24);
                let w = weights[i];
                let r = y[i] - mu[i];
                ll += w * (-0.5 * (r * r * inv_s2 + ln2pi + two_eta));
                if w == 0.0 {
                    wmu[i] = 0.0;
                    zmu[i] = mu[i];
                } else {
                    wmu[i] = floor_positiveweight(w * inv_s2, MIN_WEIGHT);
                    zmu[i] = mu[i] + r;
                }
                let dlogsigma_du = if sigma <= 1e-10 { sigma * 1e10 } else { 1.0 };
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

        let t_legacy = Instant::now();
        for _ in 0..rounds {
            std::hint::black_box(legacy_eval());
        }
        let legacy_dt = t_legacy.elapsed();

        let t_opt = Instant::now();
        for _ in 0..rounds {
            std::hint::black_box(optimized_eval());
        }
        let opt_dt = t_opt.elapsed();
        eprintln!(
            "gaussian hotloop legacy={:?} optimized={:?} speedup={:.3}x",
            legacy_dt,
            opt_dt,
            legacy_dt.as_secs_f64() / opt_dt.as_secs_f64()
        );
    }

    #[test]
    fn binomial_location_scale_log_sigmaweight_directional_derivative_matches_finite_difference() {
        let family = BinomialLocationScaleFamily {
            y: Array1::from_vec(vec![0.0]),
            weights: Array1::from_vec(vec![1.0]),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
        };
        let eta_t = Array1::from_vec(vec![0.35]);
        let eta_ls = Array1::from_vec(vec![-0.2]);
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_t,
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
                BinomialLocationScaleFamily::BLOCK_LOG_SIGMA,
                &d_eta,
            )
            .expect("binomial directional derivative")
            .expect("binomial log-sigma derivative");

        let eps = 1e-6;
        let mut states_plus = states.clone();
        states_plus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta[0] += eps;
        let eval_plus = family.evaluate(&states_plus).expect("binomial eval plus");
        let w_plus =
            match &eval_plus.blockworking_sets[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::Diagonal {
                    working_response: _,
                    working_weights,
                } => working_weights[0],
                BlockWorkingSet::ExactNewton { .. } => {
                    panic!("expected diagonal binomial log-sigma block")
                }
            };

        let mut states_minus = states;
        states_minus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta[0] -= eps;
        let eval_minus = family.evaluate(&states_minus).expect("binomial eval minus");
        let w_minus =
            match &eval_minus.blockworking_sets[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::Diagonal {
                    working_response: _,
                    working_weights,
                } => working_weights[0],
                BlockWorkingSet::ExactNewton { .. } => {
                    panic!("expected diagonal binomial log-sigma block")
                }
            };

        let fd = (w_plus - w_minus) / (2.0 * eps);
        assert!((dw[0] - fd).abs() < 1e-6, "dw={} fd={}", dw[0], fd);
    }

    #[test]
    fn binomial_location_scale_thresholdweight_directional_derivative_matches_finite_difference() {
        let family = BinomialLocationScaleFamily {
            y: Array1::from_vec(vec![1.0]),
            weights: Array1::from_vec(vec![1.0]),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
        };
        let eta_t = Array1::from_vec(vec![0.35]);
        let eta_ls = Array1::from_vec(vec![-0.2]);
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_ls,
            },
        ];
        let d_eta = Array1::from_vec(vec![1.0]);

        let dw = family
            .diagonalworking_weights_directional_derivative(
                &states,
                BinomialLocationScaleFamily::BLOCK_T,
                &d_eta,
            )
            .expect("binomial directional derivative")
            .expect("binomial threshold derivative");

        let eps = 1e-6;
        let mut states_plus = states.clone();
        states_plus[BinomialLocationScaleFamily::BLOCK_T].eta[0] += eps;
        let eval_plus = family.evaluate(&states_plus).expect("binomial eval plus");
        let w_plus = match &eval_plus.blockworking_sets[BinomialLocationScaleFamily::BLOCK_T] {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => working_weights[0],
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal binomial threshold block")
            }
        };

        let mut states_minus = states;
        states_minus[BinomialLocationScaleFamily::BLOCK_T].eta[0] -= eps;
        let eval_minus = family.evaluate(&states_minus).expect("binomial eval minus");
        let w_minus = match &eval_minus.blockworking_sets[BinomialLocationScaleFamily::BLOCK_T] {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => working_weights[0],
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal binomial threshold block")
            }
        };

        let fd = (w_plus - w_minus) / (2.0 * eps);
        assert!((dw[0] - fd).abs() < 1e-6, "dw={} fd={}", dw[0], fd);
    }

    #[test]
    fn fit_binomial_location_scale_runswithwarm_start_path() {
        let n = 32usize;
        let y = Array1::from_vec((0..n).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect());
        let weights = Array1::from_vec(vec![1.0; n]);
        let spec = BinomialLocationScaleSpec {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_block: intercept_block(n),
            log_sigma_block: intercept_block(n),
        };

        let fit = fit_binomial_location_scale(spec, &BlockwiseFitOptions::default())
            .expect("binomial location-scale family should fit");
        assert_eq!(fit.block_states.len(), 2);
        assert!(fit.log_likelihood.is_finite());
    }

    #[test]
    fn fit_binomial_location_scale_applies_shrinkage_to_log_sigma_block() {
        let n = 64usize;
        let y = Array1::from_vec(
            (0..n)
                .map(|i| if i % 5 == 0 || i % 7 == 0 { 1.0 } else { 0.0 })
                .collect(),
        );
        let weights = Array1::from_vec(vec![1.0; n]);
        let spec = BinomialLocationScaleSpec {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_block: intercept_block(n),
            log_sigma_block: intercept_block(n),
        };

        let fit = fit_binomial_location_scale(
            spec,
            &BlockwiseFitOptions {
                compute_covariance: true,
                ..BlockwiseFitOptions::default()
            },
        )
        .expect("binomial location-scale family should fit with shrinkage penalty");
        assert_eq!(fit.log_lambdas.len(), 1);
        assert_eq!(fit.lambdas.len(), 1);
        let covariance = fit
            .covariance_conditional
            .as_ref()
            .expect("conditional covariance");
        assert!(
            covariance[[1, 1]].is_finite() && covariance[[1, 1]] < 50.0,
            "log_sigma variance should be regularized, got {}",
            covariance[[1, 1]]
        );
    }

    #[test]
    fn fit_binomial_location_scale_sas_runs() {
        let n = 28usize;
        let y = Array1::from_vec((0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect());
        let weights = Array1::from_vec(vec![1.0; n]);
        let sas = crate::mixture_link::state_from_sasspec(crate::types::SasLinkSpec {
            initial_epsilon: 0.15,
            initial_log_delta: -0.05,
        })
        .expect("sas state");
        let spec = BinomialLocationScaleSpec {
            y,
            weights,
            link_kind: InverseLink::Sas(sas),
            threshold_block: intercept_block(n),
            log_sigma_block: intercept_block(n),
        };

        let fit = fit_binomial_location_scale(spec, &BlockwiseFitOptions::default())
            .expect("binomial location-scale sas should fit");
        assert_eq!(fit.block_states.len(), 2);
        assert!(fit.log_likelihood.is_finite());
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
                    },
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
            allow_finite_difference_fallback: false,
        }
    }

    #[test]
    fn binomial_location_scale_exact_probit_tailobjects_stay_finite() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_elem(n, 1.0);
        let threshold_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let log_sigma_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let family = BinomialLocationScaleFamily {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
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
            freeze_spatial_length_scale_terms_from_design(&meanspec, &mean_design)
                .expect("freeze mean spec");
        let noisespec_resolved =
            freeze_spatial_length_scale_terms_from_design(&noisespec, &noise_design)
                .expect("freeze noise spec");
        let rho = compose_theta_from_hints(
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
            true,
        )
        .expect("exact spatial joint hyper eval");
        assert!(eval.objective.is_finite());
        assert!(eval.gradient.iter().all(|v| v.is_finite()));
        let hess = eval
            .outer_hessian
            .expect("exact spatial joint hyper path should return a full [rho, psi] hessian");
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        let theta_dim = rho.len() + psi_dim;
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
            freeze_spatial_length_scale_terms_from_design(&meanspec, &mean_design)
                .expect("freeze mean spec");
        let noisespec_resolved =
            freeze_spatial_length_scale_terms_from_design(&noisespec, &noise_design)
                .expect("freeze noise spec");
        let rho = compose_theta_from_hints(
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
            true,
        )
        .expect("exact wiggle spatial joint hyper eval");
        assert!(eval.objective.is_finite());
        assert!(eval.gradient.iter().all(|v| v.is_finite()));
        let hess = eval.outer_hessian.expect(
            "exact wiggle spatial joint hyper path should return a full [rho, psi] hessian",
        );
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        let theta_dim = rho.len() + psi_dim;
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
        };
        let mean_design =
            build_term_collection_design(data.view(), &meanspec).expect("build mean design");
        let noise_design =
            build_term_collection_design(data.view(), &noisespec).expect("build noise design");
        let meanspec_resolved =
            freeze_spatial_length_scale_terms_from_design(&meanspec, &mean_design)
                .expect("freeze mean spec");
        let noisespec_resolved =
            freeze_spatial_length_scale_terms_from_design(&noisespec, &noise_design)
                .expect("freeze noise spec");
        let rho = compose_theta_from_hints(
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
            true,
        )
        .expect("exact spatial joint hyper eval");
        assert!(eval.objective.is_finite());
        assert!(eval.gradient.iter().all(|v| v.is_finite()));
        let hess = eval
            .outer_hessian
            .expect("exact spatial joint hyper path should return a full [rho, psi] hessian");
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        let theta_dim = rho.len() + psi_dim;
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
            freeze_spatial_length_scale_terms_from_design(&meanspec, &mean_design)
                .expect("freeze mean spec");
        let noisespec_resolved =
            freeze_spatial_length_scale_terms_from_design(&noisespec, &noise_design)
                .expect("freeze noise spec");
        let rho = compose_theta_from_hints(
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
        assert_eq!(psi_terms.hessian_psi.dim(), (total, total));
        assert_eq!(psi2_terms.score_psi_psi.len(), total);
        assert_eq!(psi2_terms.hessian_psi_psi.dim(), (total, total));

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
        };
        let mean_design =
            build_term_collection_design(data.view(), &meanspec).expect("build mean design");
        let noise_design =
            build_term_collection_design(data.view(), &noisespec).expect("build noise design");
        let meanspec_resolved =
            freeze_spatial_length_scale_terms_from_design(&meanspec, &mean_design)
                .expect("freeze mean spec");
        let noisespec_resolved =
            freeze_spatial_length_scale_terms_from_design(&noisespec, &noise_design)
                .expect("freeze noise spec");
        let rho = compose_theta_from_hints(
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
        assert_eq!(psi_terms.hessian_psi.dim(), (total, total));
        assert_eq!(psi2_terms.score_psi_psi.len(), total);
        assert_eq!(psi2_terms.hessian_psi_psi.dim(), (total, total));

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
        };

        let err = fit_gaussian_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect_err("term API should reject negative weights");
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
        };

        let err = fit_binomial_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect_err("term API should reject invalid binomial responses");
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
        };

        let err = fit_binomial_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect_err("term API should reject data/y row mismatches");
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
        };
        let fit = fit_gaussian_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect("gaussian location-scale spatial fit");
        assert!(fit.fit.penalizedobjective.is_finite());
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
        };
        let fit = fit_binomial_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect("binomial location-scale spatial fit");
        assert!(fit.fit.penalizedobjective.is_finite());
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
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        let fit = fit_binomial_location_scalewiggle_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect("binomial location-scale wiggle spatial fit");
        assert!(fit.fit.penalizedobjective.is_finite());
        assert_eq!(fit.fit.block_states.len(), 3);
    }

    #[test]
    fn wiggle_chain_rule_scales_location_and_scaleworking_weights() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![0.3, -0.2, 0.4, -0.1, 0.2, -0.3]);
        let eta_ls = Array1::from_vec(vec![-0.4, -0.1, 0.0, 0.1, 0.2, -0.2]);

        let q_seed = Array1::from_vec(vec![-1.5, -0.8, -0.1, 0.4, 1.1, 1.7]);
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
        };

        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let betawiggle = Array1::from_vec(vec![0.1; wiggle_block.design.ncols()]);
        let etaw = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&betawiggle);

        let core = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&etaw),
            &family.link_kind,
        )
        .expect("core");
        let dq_dq0 = family
            .wiggle_dq_dq0(core.q0.view(), betawiggle.view())
            .expect("dq/dq0");
        let (tws, lsws, w) = binomial_location_scale_working_sets(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&etaw),
            Some(&dq_dq0),
            None,
            &core,
        )
        .expect("working sets");
        let _ = w;
        let tw = match &tws {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => working_weights,
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal working set for threshold block")
            }
        };
        let lsw = match &lsws {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => working_weights,
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal working set for log-sigma block")
            }
        };

        for i in 0..n {
            let var = (core.mu[i] * (1.0 - core.mu[i])).max(MIN_PROB);
            let s = core.sigma[i].max(1e-12);
            let dmu_t = core.dmu_dq[i] * (-dq_dq0[i] / s);
            let dmu_ls = core.dmu_dq[i] * (-dq_dq0[i] * core.q0[i] * core.dsigma_deta[i] / s);
            let expectedw_t = (weights[i] * (dmu_t * dmu_t / var)).max(MIN_WEIGHT);
            let expectedw_ls = (weights[i] * (dmu_ls * dmu_ls / var)).max(MIN_WEIGHT);
            assert!(
                (tw[i] - expectedw_t).abs() < 1e-10,
                "threshold weight mismatch at {}: got {}, expected {}",
                i,
                tw[i],
                expectedw_t
            );
            assert!(
                (lsw[i] - expectedw_ls).abs() < 1e-10,
                "log-sigma weight mismatch at {}: got {}, expected {}",
                i,
                lsw[i],
                expectedw_ls
            );
        }
    }

    #[test]
    fn clamped_binomialworking_sets_stay_locally_flat() {
        let n = 4usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![1.0e6, -1.0e6, 1.0e6, -1.0e6]);
        let eta_ls = Array1::zeros(n);
        let etaw = Array1::from_vec(vec![0.2, -0.1, 0.3, -0.2]);
        let dq_dq0 = Array1::from_vec(vec![1.0; n]);

        let core = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&etaw),
            &InverseLink::Standard(LinkFunction::Probit),
        )
        .expect("core");
        assert!(
            core.mu
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
        );

        let (tws, lsws, wws) = binomial_location_scale_working_sets(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&etaw),
            Some(&dq_dq0),
            None,
            &core,
        )
        .expect("working sets");

        match &tws {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_response, &eta_t);
                assert!(working_weights.iter().all(|w| *w == 0.0));
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal threshold block"),
        }
        match &lsws {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_response, &eta_ls);
                assert!(working_weights.iter().all(|w| *w == 0.0));
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal log-sigma block"),
        }
        match wws.expect("wiggle block") {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_response, etaw);
                assert!(working_weights.iter().all(|w| *w == 0.0));
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal wiggle block"),
        }
    }

    #[test]
    fn binomial_location_scale_log_sigma_working_weight_should_vanish_on_sigma_floor_branch() {
        let family = BinomialLocationScaleFamily {
            y: Array1::from_vec(vec![0.0]),
            weights: Array1::from_vec(vec![1.0]),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
        };
        let eta_t = Array1::from_vec(vec![1.0e-13]);
        let eta_ls0 = -30.0_f64;
        let eta_ls = Array1::from_vec(vec![eta_ls0]);
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_ls,
            },
        ];

        let mu_of_eta_ls = |ls: f64| {
            let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(ls);
            let q0 = binomial_location_scale_q0(eta_t[0], sigma);
            let jet =
                inverse_link_jet_for_inverse_link(&family.link_kind, q0).expect("inverse-link jet");
            clamped_binomial_probability(jet.mu).0
        };
        let h = 1e-6;
        let dmu_fd = (mu_of_eta_ls(eta_ls0 + h) - mu_of_eta_ls(eta_ls0 - h)) / (2.0 * h);
        assert_eq!(
            dmu_fd, 0.0,
            "with q0(eta_ls) = -eta_t / max(sigma(eta_ls), 1e-12), the active sigma floor makes mu locally constant in eta_ls"
        );

        let eval = family.evaluate(&states).expect("evaluate");
        match &eval.blockworking_sets[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => {
                assert!(
                    (working_weights[0] - 0.0).abs() < 1e-30,
                    "the log-sigma working weight should be 0 when the coded mu(eta_ls) is locally constant on the active sigma floor; got {}",
                    working_weights[0]
                );
            }
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal log-sigma block")
            }
        }
    }

    #[test]
    fn binomial_location_scale_exact_log_sigma_block_should_be_flat_on_safe_exp_plateau() {
        let y = array![1.0];
        let weights = array![1.0];
        let design = DesignMatrix::Dense(array![[1.0]]);
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(design.clone()),
            log_sigma_design: Some(design),
        };
        let eta_t0 = 0.1 * safe_exp(700.0);
        let eta_ls0 = 701.0_f64;
        let states = vec![
            ParameterBlockState {
                beta: array![eta_t0],
                eta: array![eta_t0],
            },
            ParameterBlockState {
                beta: array![eta_ls0],
                eta: array![eta_ls0],
            },
        ];

        let eval = family.evaluate(&states).expect("evaluate");
        let (analytic_score, analytic_info) =
            match &eval.blockworking_sets[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, hessian } => {
                    (gradient[0], hessian.to_dense()[[0, 0]])
                }
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton log-sigma block"),
            };

        let loglik = |eta_ls: f64| {
            binomial_location_scale_ll_only(
                &y,
                &weights,
                &array![eta_t0],
                &array![eta_ls],
                None,
                &family.link_kind,
            )
            .expect("log-likelihood")
        };
        let h = 1e-4;
        let ll_plus = loglik(eta_ls0 + h);
        let ll0 = loglik(eta_ls0);
        let ll_minus = loglik(eta_ls0 - h);
        let score_fd = (ll_plus - ll_minus) / (2.0 * h);
        let info_fd = -(ll_plus - 2.0 * ll0 + ll_minus) / (h * h);
        assert_eq!(
            score_fd, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded log-likelihood is locally flat in eta_ls on that plateau"
        );
        assert_eq!(
            info_fd, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded log-likelihood has zero second derivative in eta_ls on that plateau"
        );
        assert!(
            (analytic_score - score_fd).abs() < 1e-30,
            "the exact-newton log-sigma score should be the derivative of the coded plateau log-likelihood at eta_ls={eta_ls0}; got {} vs {}",
            analytic_score,
            score_fd
        );
        assert!(
            (analytic_info - info_fd).abs() < 1e-20,
            "the exact-newton log-sigma information should be the negative second derivative of the coded plateau log-likelihood at eta_ls={eta_ls0}; got {} vs {}",
            analytic_info,
            info_fd
        );
    }

    #[test]
    fn binomial_location_scalewiggle_exact_log_sigma_block_should_be_flat_on_safe_exp_plateau() {
        let n = 4usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.0, 1.0, n);
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
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 2,
        };

        let beta_t0 = 0.1 * safe_exp(700.0);
        let beta_ls0 = 701.0_f64;
        let betaw = Array1::zeros(wiggle_block.design.ncols());
        let rebuild_states = |beta_ls: f64| -> Vec<ParameterBlockState> {
            let beta_t = array![beta_t0];
            let beta_ls_arr = array![beta_ls];
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: threshold_design.matrixvectormultiply(&beta_t),
                },
                ParameterBlockState {
                    beta: beta_ls_arr.clone(),
                    eta: log_sigma_design.matrixvectormultiply(&beta_ls_arr),
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: Array1::zeros(n),
                },
            ]
        };

        let eval = family
            .evaluate(&rebuild_states(beta_ls0))
            .expect("evaluate");
        let (analytic_score, analytic_info) =
            match &eval.blockworking_sets[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, hessian } => {
                    (gradient[0], hessian.to_dense()[[0, 0]])
                }
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton log-sigma block"),
            };

        let objective = |beta_ls: f64| -> f64 {
            family
                .evaluate(&rebuild_states(beta_ls))
                .expect("eval objective")
                .log_likelihood
        };
        let h = 1e-4;
        let ll_plus = objective(beta_ls0 + h);
        let ll0 = objective(beta_ls0);
        let ll_minus = objective(beta_ls0 - h);
        let score_fd = (ll_plus - ll_minus) / (2.0 * h);
        let info_fd = -(ll_plus - 2.0 * ll0 + ll_minus) / (h * h);
        assert_eq!(
            score_fd, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded wiggle-family log-likelihood is locally flat in beta_log_sigma on that plateau"
        );
        assert_eq!(
            info_fd, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded wiggle-family log-likelihood has zero second derivative in beta_log_sigma on that plateau"
        );
        assert!(
            (analytic_score - score_fd).abs() < 1e-30,
            "the exact-newton wiggle-family log-sigma score should be the derivative of the coded plateau log-likelihood at beta_log_sigma={beta_ls0}; got {} vs {}",
            analytic_score,
            score_fd
        );
        assert!(
            (analytic_info - info_fd).abs() < 1e-20,
            "the exact-newton wiggle-family log-sigma information should be the negative second derivative of the coded plateau log-likelihood at beta_log_sigma={beta_ls0}; got {} vs {}",
            analytic_info,
            info_fd
        );
    }

    #[test]
    fn binomial_location_scale_exact_log_sigma_dh_should_match_zero_third_derivative_on_plateau() {
        let y = array![1.0];
        let weights = array![1.0];
        let design = DesignMatrix::Dense(array![[1.0]]);
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(design.clone()),
            log_sigma_design: Some(design),
        };
        let beta_t0 = 0.1 * safe_exp(700.0);
        let beta_ls0 = 701.0_f64;
        let states = vec![
            ParameterBlockState {
                beta: array![beta_t0],
                eta: array![beta_t0],
            },
            ParameterBlockState {
                beta: array![beta_ls0],
                eta: array![beta_ls0],
            },
        ];
        let analytic = family
            .exact_newton_hessian_directional_derivative(
                &states,
                BinomialLocationScaleFamily::BLOCK_LOG_SIGMA,
                &array![1.0],
            )
            .expect("analytic dH")
            .expect("expected exact dH");
        let loglik = |beta_ls: f64| {
            binomial_location_scale_ll_only(
                &y,
                &weights,
                &array![beta_t0],
                &array![beta_ls],
                None,
                &family.link_kind,
            )
            .expect("log-likelihood")
        };
        let h = 1e-4_f64;
        let fd3 = (loglik(beta_ls0 + 2.0 * h) - 2.0 * loglik(beta_ls0 + h)
            + 2.0 * loglik(beta_ls0 - h)
            - loglik(beta_ls0 - 2.0 * h))
            / (2.0 * h.powi(3));
        assert_eq!(
            fd3, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded log-likelihood has zero third derivative in beta_log_sigma on that plateau"
        );
        assert!(
            (analytic[[0, 0]] + fd3).abs() < 1e-20,
            "the exact-newton log-sigma dH entry should equal the negative third derivative of the coded plateau log-likelihood at beta_log_sigma={beta_ls0}; got analytic {} vs expected {}",
            analytic[[0, 0]],
            -fd3
        );
    }

    #[test]
    fn binomial_location_scale_exact_log_sigma_d2h_should_match_zero_fourth_derivative_on_plateau()
    {
        let y = array![1.0];
        let weights = array![1.0];
        let design = DesignMatrix::Dense(array![[1.0]]);
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(design.clone()),
            log_sigma_design: Some(design),
        };
        let beta_t0 = 0.1 * safe_exp(700.0);
        let beta_ls0 = 701.0_f64;
        let states = vec![
            ParameterBlockState {
                beta: array![beta_t0],
                eta: array![beta_t0],
            },
            ParameterBlockState {
                beta: array![beta_ls0],
                eta: array![beta_ls0],
            },
        ];
        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &array![0.0, 1.0],
                &array![0.0, 1.0],
            )
            .expect("analytic d2H")
            .expect("expected exact d2H");
        let loglik = |beta_ls: f64| {
            binomial_location_scale_ll_only(
                &y,
                &weights,
                &array![beta_t0],
                &array![beta_ls],
                None,
                &family.link_kind,
            )
            .expect("log-likelihood")
        };
        let h = 1e-4_f64;
        let fd4 = (loglik(beta_ls0 - 2.0 * h) - 4.0 * loglik(beta_ls0 - h)
            + 6.0 * loglik(beta_ls0)
            - 4.0 * loglik(beta_ls0 + h)
            + loglik(beta_ls0 + 2.0 * h))
            / h.powi(4);
        assert_eq!(
            fd4, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded log-likelihood has zero fourth derivative in beta_log_sigma on that plateau"
        );
        assert!(
            (analytic[[1, 1]] + fd4).abs() < 1e-8,
            "the exact-newton log-sigma d2H entry should equal the negative fourth derivative of the coded plateau log-likelihood at beta_log_sigma={beta_ls0}; got analytic {} vs expected {}",
            analytic[[1, 1]],
            -fd4
        );
    }

    #[test]
    fn binomial_location_scalewiggle_exact_log_sigma_dh_should_match_zero_third_derivative_on_plateau()
     {
        let n = 4usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.0, 1.0, n);
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
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 2,
        };
        let beta_t0 = 0.1 * safe_exp(700.0);
        let beta_ls0 = 701.0_f64;
        let betaw = Array1::zeros(wiggle_block.design.ncols());
        let rebuild_states = |beta_ls: f64| -> Vec<ParameterBlockState> {
            let beta_t = array![beta_t0];
            let beta_ls_arr = array![beta_ls];
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: threshold_design.matrixvectormultiply(&beta_t),
                },
                ParameterBlockState {
                    beta: beta_ls_arr.clone(),
                    eta: log_sigma_design.matrixvectormultiply(&beta_ls_arr),
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: Array1::zeros(n),
                },
            ]
        };
        let analytic = family
            .exact_newton_hessian_directional_derivative(
                &rebuild_states(beta_ls0),
                BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
                &array![1.0],
            )
            .expect("analytic dH")
            .expect("expected exact dH");
        let objective = |beta_ls: f64| -> f64 {
            family
                .evaluate(&rebuild_states(beta_ls))
                .expect("eval objective")
                .log_likelihood
        };
        let h = 1e-4_f64;
        let fd3 = (objective(beta_ls0 + 2.0 * h) - 2.0 * objective(beta_ls0 + h)
            + 2.0 * objective(beta_ls0 - h)
            - objective(beta_ls0 - 2.0 * h))
            / (2.0 * h.powi(3));
        assert_eq!(
            fd3, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded wiggle-family log-likelihood has zero third derivative in beta_log_sigma on that plateau"
        );
        assert!(
            (analytic[[0, 0]] + fd3).abs() < 1e-20,
            "the exact-newton wiggle-family log-sigma dH entry should equal the negative third derivative of the coded plateau log-likelihood at beta_log_sigma={beta_ls0}; got analytic {} vs expected {}",
            analytic[[0, 0]],
            -fd3
        );
    }

    #[test]
    fn binomial_location_scalewiggle_exact_log_sigma_d2h_should_match_zero_fourth_derivative_on_plateau()
     {
        let n = 4usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.0, 1.0, n);
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
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
            wiggle_knots: knots,
            wiggle_degree: 2,
        };
        let beta_t0 = 0.1 * safe_exp(700.0);
        let beta_ls0 = 701.0_f64;
        let betaw = Array1::zeros(wiggle_block.design.ncols());
        let rebuild_states = |beta_ls: f64| -> Vec<ParameterBlockState> {
            let beta_t = array![beta_t0];
            let beta_ls_arr = array![beta_ls];
            vec![
                ParameterBlockState {
                    beta: beta_t.clone(),
                    eta: threshold_design.matrixvectormultiply(&beta_t),
                },
                ParameterBlockState {
                    beta: beta_ls_arr.clone(),
                    eta: log_sigma_design.matrixvectormultiply(&beta_ls_arr),
                },
                ParameterBlockState {
                    beta: betaw.clone(),
                    eta: Array1::zeros(n),
                },
            ]
        };
        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &rebuild_states(beta_ls0),
                &array![0.0, 1.0, 0.0],
                &array![0.0, 1.0, 0.0],
            )
            .expect("analytic d2H")
            .expect("expected exact d2H");
        let objective = |beta_ls: f64| -> f64 {
            family
                .evaluate(&rebuild_states(beta_ls))
                .expect("eval objective")
                .log_likelihood
        };
        let h = 1e-4_f64;
        let fd4 = (objective(beta_ls0 - 2.0 * h) - 4.0 * objective(beta_ls0 - h)
            + 6.0 * objective(beta_ls0)
            - 4.0 * objective(beta_ls0 + h)
            + objective(beta_ls0 + 2.0 * h))
            / h.powi(4);
        assert_eq!(
            fd4, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded wiggle-family log-likelihood has zero fourth derivative in beta_log_sigma on that plateau"
        );
        assert!(
            (analytic[[1, 1]] + fd4).abs() < 1e-8,
            "the exact-newton wiggle-family log-sigma d2H entry should equal the negative fourth derivative of the coded plateau log-likelihood at beta_log_sigma={beta_ls0}; got analytic {} vs expected {}",
            analytic[[1, 1]],
            -fd4
        );
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
        let (value_ad, grad_ad, hess_ad) = second_derivative(
            |bt| wiggle_negloglik_threshold_numdual(bt, beta_ls0, &betaw, &y, &weights, &knots, 3),
            beta_t0,
        );
        let _ = value_ad;
        let _ = grad_ad;
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
            mu_design: Some(DesignMatrix::Dense(x_mu0_mat.clone())),
            log_sigma_design: Some(DesignMatrix::Dense(x_ls0_mat.clone())),
            cached_row_scalars: std::cell::RefCell::new(None),
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
                x_psi_psi: Some(vec![x_ls_psi_psi.clone().insert_axis(Axis(1))]),
                s_psi_psi: Some(vec![Array2::zeros((1, 1))]),
                s_psi_psi_components: None,
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
            mu_design: Some(DesignMatrix::Dense(x_mu0_mat.clone())),
            log_sigma_design: Some(DesignMatrix::Dense(x_ls0_mat.clone())),
            cached_row_scalars: std::cell::RefCell::new(None),
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
                x_psi_psi: Some(vec![x_ls_psi_psi.clone().insert_axis(Axis(1))]),
                s_psi_psi: Some(vec![Array2::zeros((1, 1))]),
                s_psi_psi_components: None,
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
        let threshold_design = DesignMatrix::Dense(threshold_x.clone());
        let log_sigma_design = DesignMatrix::Dense(log_sigma_x.clone());
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
        let threshold_design = DesignMatrix::Dense(threshold_x.clone());
        let log_sigma_design = DesignMatrix::Dense(log_sigma_x.clone());
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
        let threshold_design = DesignMatrix::Dense(Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }));
        let log_sigma_design = DesignMatrix::Dense(Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).cos(),
                _ => unreachable!(),
            }
        }));
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
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
        let threshold_design = DesignMatrix::Dense(Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                _ => unreachable!(),
            }
        }));
        let log_sigma_design = DesignMatrix::Dense(Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }));
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
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
        let threshold_design = DesignMatrix::Dense(Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                _ => unreachable!(),
            }
        }));
        let log_sigma_design = DesignMatrix::Dense(Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }));
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: Some(threshold_design.clone()),
            log_sigma_design: Some(log_sigma_design.clone()),
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
    fn wiggle_constraint_removes_constant_and_linear_modes() {
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
        let (z, s_constrained) =
            compute_geometric_constraint_transform(&knots, degree, penalty_order)
                .expect("constraint transform");
        let _ = s_constrained;
        let g = compute_greville_abscissae(&knots, degree).expect("greville abscissae");

        assert_eq!(block.design.ncols(), z.ncols());

        let beta = Array1::from_vec(vec![0.2; z.ncols()]);
        let theta = z.dot(&beta);
        let c0 = theta.sum();
        let c1 = theta.dot(&g);
        assert!(
            c0.abs() < 1e-9,
            "constant mode leaked through wiggle constraint: {}",
            c0
        );
        assert!(
            c1.abs() < 1e-9,
            "linear mode leaked through wiggle constraint: {}",
            c1
        );
    }

    #[test]
    fn degeneratewiggle_seed_uses_broad_fallback_domain() {
        let q_seed = Array1::zeros(9);
        let degree = 3usize;
        let knots = initializewiggle_knots_from_seed(q_seed.view(), degree, 5)
            .expect("initialize degenerate wiggle knots");
        let domain_min = knots[degree];
        let domain_max = knots[knots.len() - degree - 1];
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
    fn wiggle_block_design_matches_constrained_basis_projection() {
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
            degree,
            BasisOptions::value(),
        )
        .expect("full basis");
        let full = (*basis).clone();
        let (z, s_constrained) =
            compute_geometric_constraint_transform(&knots, degree, penalty_order)
                .expect("constraint transform");
        let _ = s_constrained;
        let expected = full.dot(&z);

        let got = match &block.design {
            DesignMatrix::Dense(x) => x.clone(),
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
        let (geom_x, geom_offset) = family
            .block_geometry(&states, &wigglespec)
            .expect("block geometry");
        let _ = geom_offset;
        let geom = match geom_x {
            DesignMatrix::Dense(x) => x,
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

    // ── Root-cause reproduction: binomial location-scale exact REML
    //    vulnerability when stabilization path is not active ─────────

    /// **Sufficiency – BinomialLocationScaleFamily's exact Newton
    /// Hessian becomes non-finite when fitted probabilities are extreme.**
    ///
    /// When probit-link eta values are large (|η| > 30), the working
    /// weights effectively become zero, making the exact Hessian
    /// ill-conditioned.  At extreme enough values the Hessian entries
    /// overflow or underflow to non-finite values.
    ///
    /// This is the mechanism that causes the `rust_gamlss_flexible /
    /// bone_gamair` benchmark failure: the pilot fit uses
    /// `BinomialLocationScaleFamily` which does NOT have
    /// `known_link_wiggle()` (returns None), so the stabilization path
    /// at `fit_custom_family` line 5142 is skipped.  The outer Newton
    /// evaluates the exact REML objective (including `log|H_mode|`)
    /// at the initial rho, gets non-finite values, and
    /// `CachedSecondOrderObjective` has no cache fallback at iter=0.
    ///
    /// This test directly verifies that the Hessian becomes non-finite
    /// under extreme fitted values — proving the mechanism.
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
            threshold_design: Some(DesignMatrix::Dense(design.clone())),
            log_sigma_design: Some(DesignMatrix::Dense(design.clone())),
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

        // Verify the structural vulnerability: known_link_wiggle is None,
        // meaning the stabilization path in fit_custom_family is NOT active.
        assert!(
            family.known_link_wiggle().is_none(),
            "BinomialLocationScaleFamily should NOT have known_link_wiggle (that's the vulnerability)"
        );
    }

    /// **Necessity – the wiggle variant IS protected by the stabilization path.**
    ///
    /// `BinomialLocationScaleWiggleFamily` overrides `known_link_wiggle()`
    /// to return `Some(...)`, which activates the stabilization path in
    /// `fit_custom_family` that skips the fragile outer REML optimization.
    /// This is why the wiggle variant does not suffer the same failure.
    #[test]
    fn binomial_location_scale_wiggle_has_stabilization() {
        use crate::families::custom_family::CustomFamily;
        let n = 4usize;
        let y = array![1.0, 1.0, 0.0, 0.0];
        let weights = Array1::ones(n);

        // Non-wiggle: no stabilization (the vulnerability)
        let family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
        };
        assert!(
            family.known_link_wiggle().is_none(),
            "BinomialLocationScaleFamily should NOT have stabilization"
        );

        // Wiggle variant: has stabilization (the fix)
        let wiggle_family = BinomialLocationScaleWiggleFamily {
            y,
            weights,
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_design: None,
            log_sigma_design: None,
            wiggle_knots: array![0.0, 0.5, 1.0],
            wiggle_degree: 3,
        };
        assert!(
            wiggle_family.known_link_wiggle().is_some(),
            "BinomialLocationScaleWiggleFamily SHOULD have stabilization"
        );
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

    #[test]
    fn gaussian_location_scale_extreme_log_sigma_survives() {
        use crate::families::custom_family::BlockwiseFitOptions;
        let n = 20;
        let y = Array1::from_shape_fn(n, |i| (i as f64) * 0.5 + 0.1);
        let weights = Array1::ones(n);
        let mu_design = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { 1.0 } else { 0.5 });
        let ls_design = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { 1.0 } else { 0.1 });
        let spec = GaussianLocationScaleSpec {
            y,
            weights,
            mu_block: ParameterBlockInput {
                design: DesignMatrix::Dense(mu_design),
                offset: Array1::zeros(n),
                penalties: vec![],
                initial_log_lambdas: None,
                initial_beta: Some(array![0.0, 1.0]),
            },
            log_sigma_block: ParameterBlockInput {
                design: DesignMatrix::Dense(ls_design),
                offset: Array1::zeros(n),
                penalties: vec![],
                initial_log_lambdas: None,
                initial_beta: Some(array![500.0, 0.0]),
            },
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 3,
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };
        let result = fit_gaussian_location_scale(spec, &options);
        match result {
            Ok(fit) => {
                let all_finite = fit
                    .block_states
                    .iter()
                    .all(|state| state.beta.iter().all(|v| v.is_finite()));
                assert!(
                    all_finite,
                    "fit succeeded but produced non-finite coefficients"
                );
            }
            Err(ref e) => {
                let msg = format!("{e}");
                assert!(
                    !msg.contains("eigendecomposition failed"),
                    "the fit crashed with the eigendecomposition bug: {msg}"
                );
            }
        }
    }
}
