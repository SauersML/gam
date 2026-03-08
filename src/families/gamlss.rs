use crate::basis::{
    BasisOptions, Dense, KnotSource, compute_geometric_constraint_transform, create_basis,
    create_difference_penalty_matrix, evaluate_bspline_fourth_derivative_scalar,
    evaluate_bspline_third_derivative_scalar,
};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily,
    CustomFamilyBlockPsiDerivative, FamilyEvaluation, KnownLinkWiggle, ParameterBlockSpec,
    ParameterBlockState, evaluate_custom_family_joint_hyper, fit_custom_family,
};
use crate::faer_ndarray::{fast_atv, fast_xt_diag_x, fast_xt_diag_y};
use crate::families::scale_design::{
    apply_scale_deviation_transform, build_scale_deviation_transform, infer_non_intercept_start,
};
use crate::families::sigma_link::SigmaJet1;
use crate::generative::{CustomFamilyGenerative, GenerativeSpec, NoiseModel};
use crate::matrix::{DesignMatrix, SymmetricMatrix, xt_diag_x_symmetric};
use crate::mixture_link::inverse_link_jet_for_inverse_link;
use crate::probability::{normal_cdf, normal_pdf};
use crate::smooth::{
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec, TwoBlockExactJointHyperSetup, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design, get_spatial_length_scale,
    optimize_two_block_spatial_length_scale, optimize_two_block_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices, try_build_spatial_log_kappa_derivative_info_list,
};
use crate::solver::pirls::WorkingLikelihood;
use crate::types::{GlmLikelihoodFamily, InverseLink, LinkFunction};
use ndarray::{Array1, Array2, ArrayView1, s};
const MIN_PROB: f64 = 1e-10;
const MIN_DERIV: f64 = 1e-8;
const MIN_WEIGHT: f64 = 1e-12;
const BETA_RANGE_WARN_THRESHOLD: f64 = 1.10;
const BINOMIAL_EFFECTIVE_N_WARN_THRESHOLD: f64 = 25.0;

#[inline]
fn floor_positive_weight(raw_weight: f64, min_weight: f64) -> f64 {
    if raw_weight <= 0.0 {
        0.0
    } else {
        raw_weight.max(min_weight)
    }
}

#[inline]
fn gaussian_log_sigma_irls_info(weight: f64, sigma: f64, d_sigma: f64) -> (f64, f64) {
    let s = sigma.max(1e-10);
    let dlogsigma_du = if d_sigma == 0.0 {
        0.0
    } else {
        (d_sigma / s).clamp(-1.0, 1.0)
    };
    let info_u = floor_positive_weight(2.0 * weight * dlogsigma_du * dlogsigma_du, MIN_WEIGHT);
    (dlogsigma_du, info_u)
}

#[inline]
fn gaussian_log_sigma_irls_info_directional_derivative(
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
    let raw_info = 2.0 * weight * g * g;
    if !raw_info.is_finite() || raw_info <= MIN_WEIGHT {
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

#[inline]
fn hard_clamped_eta_row(eta: f64, min_eta: f64, max_eta: f64) -> (f64, bool) {
    let eta_used = eta.clamp(min_eta, max_eta);
    let clamp_active = eta != eta_used;
    (eta_used, clamp_active)
}

#[derive(Clone, Copy)]
struct GamlssLambdaLayout {
    k_mean: usize,
    k_noise: usize,
    k_wiggle: usize,
}

impl GamlssLambdaLayout {
    fn two_block(k_mean: usize, k_noise: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            k_wiggle: 0,
        }
    }

    fn with_wiggle(k_mean: usize, k_noise: usize, k_wiggle: usize) -> Self {
        Self {
            k_mean,
            k_noise,
            k_wiggle,
        }
    }

    fn total(self) -> usize {
        self.k_mean + self.k_noise + self.k_wiggle
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
        self.k_mean + self.k_noise + self.k_wiggle
    }

    fn has_wiggle(self) -> bool {
        self.k_wiggle > 0
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
    fn two_block(pt: usize, pls: usize) -> Self {
        Self { pt, pls, pw: 0 }
    }

    fn with_wiggle(pt: usize, pls: usize, pw: usize) -> Self {
        Self { pt, pls, pw }
    }

    fn total(self) -> usize {
        self.pt + self.pls + self.pw
    }

    fn split_two(
        self,
        flat: &Array1<f64>,
        context: &str,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
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
        ))
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
    pub parameter_names: &'static [&'static str],
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
    pub fn into_spec(self, name: &str) -> Result<ParameterBlockSpec, String> {
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

fn validate_weights(weights: &Array1<f64>, context: &str) -> Result<(), String> {
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

pub fn initialize_wiggle_knots_from_seed(
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

pub fn build_wiggle_block_input_from_knots(
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

pub fn build_wiggle_block_input_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
) -> Result<(ParameterBlockInput, Array1<f64>), String> {
    let knots = initialize_wiggle_knots_from_seed(seed, cfg.degree, cfg.num_internal_knots)?;
    let block = build_wiggle_block_input_from_knots(
        seed,
        &knots,
        cfg.degree,
        cfg.penalty_order,
        cfg.double_penalty,
    )?;
    Ok((block, knots))
}

fn validate_block_rows(name: &str, n: usize, block: &ParameterBlockInput) -> Result<(), String> {
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

fn validate_term_data_rows(context: &str, expected: usize, found: usize) -> Result<(), String> {
    if expected != found {
        return Err(format!(
            "{context}: data row count must match response length (expected {expected}, found {found})"
        ));
    }
    Ok(())
}

fn validate_gaussian_location_scale_term_spec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &GaussianLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_data_rows(context, n, data.nrows())?;
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, context)?;
    Ok(())
}

fn validate_binomial_location_scale_term_spec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_data_rows(context, n, data.nrows())?;
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, context)?;
    validate_binomial_response(&spec.y, context)?;
    Ok(())
}

fn validate_binomial_location_scale_wiggle_term_spec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleWiggleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_data_rows(context, n, data.nrows())?;
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, context)?;
    validate_binomial_response(&spec.y, context)?;
    validate_block_rows("wiggle", n, &spec.wiggle_block)?;
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

/// Shared single-block GLM evaluation adapter backed by the engine-level
/// `WorkingLikelihood` implementation used by PIRLS.
fn evaluate_single_block_glm(
    family: GlmLikelihoodFamily,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta: &Array1<f64>,
) -> Result<FamilyEvaluation, String> {
    let n = y.len();
    if eta.len() != n || weights.len() != n {
        return Err("single-block GLM input size mismatch".to_string());
    }
    let mut mu = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);
    family
        .irls_update(
            y.view(),
            eta,
            weights.view(),
            &mut mu,
            &mut w,
            &mut z,
            None,
            None,
        )
        .map_err(|e| e.to_string())?;
    let ll = family
        .log_likelihood(y.view(), eta, &mu, weights.view())
        .map_err(|e| e.to_string())?;
    Ok(FamilyEvaluation {
        log_likelihood: ll,
        block_working_sets: vec![BlockWorkingSet::Diagonal {
            working_response: z,
            working_weights: w,
        }],
    })
}

fn initial_log_lambdas_or_zeros(block: &ParameterBlockInput) -> Result<Array1<f64>, String> {
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
    resolved_spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    // Custom-family exact blocks consume psi = log(kappa) derivatives. The exact-joint
    // setup is typed in SpatialLogKappaCoords so these derivatives stay in the same
    // parameterization end-to-end.
    let spatial_terms = spatial_length_scale_term_indices(resolved_spec);
    let Some(info_list) = try_build_spatial_log_kappa_derivative_info_list(
        data,
        resolved_spec,
        design,
        &spatial_terms,
    )
    .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    Ok(Some(
        info_list
            .into_iter()
            .map(|info| CustomFamilyBlockPsiDerivative {
                penalty_index: info.penalty_index,
                x_psi: info.x_psi,
                s_psi: info.s_psi,
                s_psi_components: Some(
                    info.penalty_indices
                        .into_iter()
                        .zip(info.s_psi_components.into_iter())
                        .collect(),
                ),
            })
            .collect(),
    ))
}

fn build_two_block_exact_joint_setup(
    mean_spec: &TermCollectionSpec,
    noise_spec: &TermCollectionSpec,
    mean_penalties: usize,
    noise_penalties: usize,
    extra_rho0: &[f64],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> TwoBlockExactJointHyperSetup {
    // Exact-joint setup stores the spatial tail in log(kappa), not log(length_scale).
    let mean_terms = spatial_length_scale_term_indices(mean_spec);
    let noise_terms = spatial_length_scale_term_indices(noise_spec);
    let rho_dim = mean_penalties + noise_penalties + extra_rho0.len();
    let mut rho0_vec = Array1::<f64>::zeros(rho_dim);
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    let mut log_kappa0 = Array1::<f64>::zeros(mean_terms.len() + noise_terms.len());

    for (i, &rho_init) in extra_rho0.iter().enumerate() {
        rho0_vec[mean_penalties + noise_penalties + i] = rho_init;
    }
    for (slot, &term_idx) in mean_terms.iter().enumerate() {
        let length_scale = get_spatial_length_scale(mean_spec, term_idx)
            .unwrap_or(kappa_options.min_length_scale)
            .clamp(
                kappa_options.min_length_scale,
                kappa_options.max_length_scale,
            );
        log_kappa0[slot] = -length_scale.ln();
    }
    for (slot, &term_idx) in noise_terms.iter().enumerate() {
        let length_scale = get_spatial_length_scale(noise_spec, term_idx)
            .unwrap_or(kappa_options.min_length_scale)
            .clamp(
                kappa_options.min_length_scale,
                kappa_options.max_length_scale,
            );
        log_kappa0[mean_terms.len() + slot] = -length_scale.ln();
    }
    TwoBlockExactJointHyperSetup::new(
        rho0_vec,
        rho_lower,
        rho_upper,
        SpatialLogKappaCoords::new(log_kappa0),
        SpatialLogKappaCoords::lower_bounds(mean_terms.len() + noise_terms.len(), kappa_options),
        SpatialLogKappaCoords::upper_bounds(mean_terms.len() + noise_terms.len(), kappa_options),
    )
}

fn solve_weighted_projection(
    design: &DesignMatrix,
    offset: &Array1<f64>,
    target_eta: &Array1<f64>,
    weights: &Array1<f64>,
    ridge_floor: f64,
) -> Result<Array1<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if offset.len() != n || target_eta.len() != n || weights.len() != n {
        return Err("solve_weighted_projection dimension mismatch".to_string());
    }

    let y_star = target_eta - offset;
    let xtwy = design.compute_xtwy(weights, &y_star)?;
    let ridge = ridge_floor.max(1e-12);
    let penalty = Array2::from_diag(&Array1::from_elem(p, ridge));
    let beta = design
        .solve_system(weights, &xtwy, Some(&penalty))
        .map_err(|_| "solve_weighted_projection produced non-finite coefficients".to_string())?;
    if beta.iter().any(|v| !v.is_finite()) {
        return Err("solve_weighted_projection produced non-finite coefficients".to_string());
    }
    Ok(beta)
}

fn solve_penalized_weighted_projection(
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
        return Err("solve_penalized_weighted_projection dimension mismatch".to_string());
    }
    if penalties.len() != log_lambdas.len() {
        return Err(format!(
            "solve_penalized_weighted_projection lambda mismatch: penalties={}, log_lambdas={}",
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
                "solve_penalized_weighted_projection encountered invalid lambda at index {k}: {}",
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
            "solve_penalized_weighted_projection produced non-finite coefficients".to_string()
        })?;
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(
            "solve_penalized_weighted_projection produced non-finite coefficients".to_string(),
        );
    }
    Ok(beta)
}

fn gaussian_location_scale_warm_start(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    mu_block: &ParameterBlockSpec,
    log_sigma_block: &ParameterBlockSpec,
    ridge_floor: f64,
    mean_beta_hint: Option<&Array1<f64>>,
    noise_beta_hint: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>, f64), String> {
    let beta_mu = if let Some(beta) = mean_beta_hint {
        beta.clone()
    } else {
        solve_penalized_weighted_projection(
            &mu_block.design,
            &mu_block.offset,
            y,
            weights,
            &mu_block.penalties,
            &mu_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    let mut mu_hat = mu_block.design.matrix_vector_multiply(&beta_mu);
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
        solve_penalized_weighted_projection(
            &log_sigma_block.design,
            &log_sigma_block.offset,
            &sigma_target,
            weights,
            &log_sigma_block.penalties,
            &log_sigma_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    Ok((beta_mu, beta_log_sigma, sigma_hat))
}

fn weighted_prevalence(y: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let w_sum: f64 = weights.iter().copied().sum();
    if w_sum <= 0.0 {
        return 0.5;
    }
    let y_w_sum: f64 = y.iter().zip(weights.iter()).map(|(&yi, &wi)| yi * wi).sum();
    (y_w_sum / w_sum).clamp(0.0, 1.0)
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

fn emit_binomial_alpha_beta_warnings(
    context: &str,
    beta_values: &Array1<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
) {
    if beta_values.is_empty() {
        return;
    }
    let beta_min = beta_values.iter().copied().fold(f64::INFINITY, f64::min);
    let beta_max = beta_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

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
            "[GAMLSS][{}] low effective sample size N_eff={:.3} (sum_w={:.3}, prevalence={:.3}); location-scale separation artifacts are more likely",
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
            let raw_mu = normal_cdf(q);
            let clamp_active = raw_mu <= MIN_PROB || raw_mu >= 1.0 - MIN_PROB;
            let mu = raw_mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
            let dmu_dq = if clamp_active {
                0.0
            } else {
                normal_pdf(q).max(MIN_DERIV)
            };
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
                w_alpha[i] = floor_positive_weight(
                    self.weights[i] * (dmu_alpha * dmu_alpha / var),
                    MIN_WEIGHT,
                );
                z_alpha[i] =
                    eta_alpha[i] + (self.y[i] - mu) / signed_with_floor(dmu_alpha, MIN_DERIV);
            }

            w_beta[i] = 0.0;
            z_beta[i] = eta_beta[i];
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            block_working_sets: vec![
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

fn try_binomial_alpha_beta_warm_start(
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

    let alpha_spec = ParameterBlockSpec {
        name: "alpha_warm".to_string(),
        design: threshold_block.design.clone(),
        offset: threshold_block.offset.clone(),
        penalties: threshold_block.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas_or_zeros(threshold_block)?,
        initial_beta: None,
    };
    let beta_spec = ParameterBlockSpec {
        name: "beta_warm".to_string(),
        design: log_sigma_block.design.clone(),
        offset: log_sigma_block.offset.clone(),
        penalties: log_sigma_block.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas_or_zeros(log_sigma_block)?,
        initial_beta: None,
    };

    let warm_options = BlockwiseFitOptions {
        inner_max_cycles: options.inner_max_cycles.min(40).max(5),
        inner_tol: options.inner_tol,
        outer_max_iter: options.outer_max_iter.min(20).max(3),
        outer_tol: options.outer_tol.max(1e-6),
        min_weight: options.min_weight,
        ridge_floor: options.ridge_floor.max(1e-10),
        ridge_policy: options.ridge_policy,
        // Warm start optimization focuses on robust initialization, not REML correction.
        use_reml_objective: false,
        // Warm-start covariance is unused.
        compute_covariance: false,
    };
    let warm_fit = fit_custom_family(&warm_family, &[alpha_spec, beta_spec], &warm_options)?;
    let eta_alpha = &warm_fit.block_states[BinomialAlphaBetaWarmStartFamily::BLOCK_ALPHA].eta;
    if eta_alpha.len() != y.len() {
        return Err("warm start eta length mismatch".to_string());
    }

    // This warm-start family currently identifies alpha only. Seed the
    // log-sigma block from a deterministic midpoint sigma instead of reusing
    // the beta block's penalty-only iterate.
    let sigma_target: f64 = 1.0;
    let eta_ls_target = 0.0;
    let beta_obs = Array1::from_elem(y.len(), 1.0 / sigma_target.max(1e-12));
    let t_target = Array1::from_iter(
        eta_alpha
            .iter()
            .zip(beta_obs.iter())
            .map(|(&a, &b)| -a / b.max(1e-12)),
    );
    let log_sigma_target = Array1::from_elem(y.len(), eta_ls_target);
    // T = -alpha/beta is noisy when beta is small (large sigma). Weight the
    // projection by beta^2 (inverse variance of T under fixed alpha noise).
    let t_projection_w = Array1::from_iter(
        weights
            .iter()
            .zip(beta_obs.iter())
            .map(|(&w, &b)| w * b * b),
    );

    let beta_t = solve_weighted_projection(
        &threshold_block.design,
        &threshold_block.offset,
        &t_target,
        &t_projection_w,
        options.ridge_floor.max(1e-10),
    )?;
    let beta_log_sigma = solve_weighted_projection(
        &log_sigma_block.design,
        &log_sigma_block.offset,
        &log_sigma_target,
        weights,
        options.ridge_floor.max(1e-10),
    )?;

    Ok((beta_t, beta_log_sigma, beta_obs))
}

#[derive(Clone)]
pub struct GaussianLocationScaleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_block: ParameterBlockInput,
    pub log_sigma_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct BinomialLogitSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub eta_block: ParameterBlockInput,
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
    pub mean_spec: TermCollectionSpec,
    pub log_sigma_spec: TermCollectionSpec,
}

#[derive(Clone)]
pub struct BinomialLocationScaleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_spec: TermCollectionSpec,
    pub log_sigma_spec: TermCollectionSpec,
}

#[derive(Clone)]
pub struct BinomialLocationScaleWiggleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_spec: TermCollectionSpec,
    pub log_sigma_spec: TermCollectionSpec,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub wiggle_block: ParameterBlockInput,
}

#[derive(Debug)]
pub struct BlockwiseTermFitResult {
    pub fit: BlockwiseFitResult,
    pub mean_spec_resolved: TermCollectionSpec,
    pub noise_spec_resolved: TermCollectionSpec,
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
    pub beta_wiggle: Option<Vec<f64>>,
}

pub type BinomialLocationScaleProbitSpec = BinomialLocationScaleSpec;
pub type BinomialLocationScaleProbitWiggleSpec = BinomialLocationScaleWiggleSpec;
pub type BinomialLocationScaleProbitTermSpec = BinomialLocationScaleTermSpec;
pub type BinomialLocationScaleProbitWiggleTermSpec = BinomialLocationScaleWiggleTermSpec;
pub type BinomialLocationScaleProbitFamily = BinomialLocationScaleFamily;
pub type BinomialLocationScaleProbitWiggleFamily = BinomialLocationScaleWiggleFamily;

pub fn fit_binomial_location_scale_probit(
    spec: BinomialLocationScaleProbitSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    fit_binomial_location_scale(spec, options)
}

pub fn fit_binomial_location_scale_probit_wiggle(
    spec: BinomialLocationScaleProbitWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    fit_binomial_location_scale_wiggle(spec, options)
}

pub fn fit_binomial_location_scale_probit_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleProbitTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    fit_binomial_location_scale_terms(data, spec, options, kappa_options)
}

pub fn fit_binomial_location_scale_probit_wiggle_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleProbitWiggleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    fit_binomial_location_scale_wiggle_terms(data, spec, options, kappa_options)
}

pub fn fit_binomial_location_scale_probit_wiggle_terms_auto(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleProbitTermSpec,
    wiggle_cfg: WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermWiggleFitResult, String> {
    fit_binomial_location_scale_wiggle_terms_auto(
        data,
        spec,
        wiggle_cfg,
        wiggle_penalty_orders,
        options,
        kappa_options,
    )
}

pub fn fit_binomial_location_scale_probit_terms_workflow(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleProbitTermSpec,
    wiggle: Option<BinomialLocationScaleWiggleWorkflowConfig>,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BinomialLocationScaleWorkflowResult, String> {
    fit_binomial_location_scale_terms_workflow(data, spec, wiggle, options, kappa_options)
}

fn slice_log_lambda_block(
    log_lambdas: &Array1<f64>,
    start: usize,
    len: usize,
    block_name: &str,
) -> Result<Array1<f64>, String> {
    let end = start + len;
    if end > log_lambdas.len() {
        return Err(format!(
            "log lambda slice for block '{block_name}' is out of bounds: {}..{} with total {}",
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
    validate_weights(&spec.weights, "fit_gaussian_location_scale")?;
    validate_block_rows("mu", n, &spec.mu_block)?;
    validate_block_rows("log_sigma", n, &spec.log_sigma_block)?;

    let GaussianLocationScaleSpec {
        y,
        weights,
        mu_block,
        log_sigma_block,
    } = spec;
    let mut mu_spec = mu_block.into_spec("mu")?;
    let mut log_sigma_spec = log_sigma_block.into_spec("log_sigma")?;
    let mu_dense = mu_spec.design.to_dense();
    let raw_log_sigma_dense = log_sigma_spec.design.to_dense();
    let non_intercept_start = infer_non_intercept_start(&raw_log_sigma_dense, &weights);
    log_sigma_spec.design = DesignMatrix::Dense(prepared_gaussian_log_sigma_design(
        &mu_dense,
        &raw_log_sigma_dense,
        &weights,
        non_intercept_start,
    )?);
    if mu_spec.initial_beta.is_none() || log_sigma_spec.initial_beta.is_none() {
        let (beta_mu0, beta_ls0, sigma0) = gaussian_location_scale_warm_start(
            &y,
            &weights,
            &mu_spec,
            &log_sigma_spec,
            options.ridge_floor,
            mu_spec.initial_beta.as_ref(),
            log_sigma_spec.initial_beta.as_ref(),
        )?;
        if mu_spec.initial_beta.is_none() {
            mu_spec.initial_beta = Some(beta_mu0);
        }
        if log_sigma_spec.initial_beta.is_none() {
            log_sigma_spec.initial_beta = Some(beta_ls0);
        }
        log::info!(
            "[GAMLSS][fit_gaussian_location_scale] initialized at residual sigma {:.6e}",
            sigma0
        );
    }

    let family = GaussianLocationScaleFamily {
        y,
        weights,
        mu_design: Some(mu_spec.design.clone()),
        log_sigma_design: Some(log_sigma_spec.design.clone()),
    };
    let blocks = vec![mu_spec, log_sigma_spec];
    Ok(fit_custom_family(&family, &blocks, options)?)
}

pub fn fit_binomial_logit(
    spec: BinomialLogitSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_binomial_logit")?;
    validate_binomial_response(&spec.y, "fit_binomial_logit")?;
    validate_block_rows("eta", n, &spec.eta_block)?;

    let family = BinomialLogitFamily {
        y: spec.y,
        weights: spec.weights,
    };
    let blocks = vec![spec.eta_block.into_spec("eta")?];
    Ok(fit_custom_family(&family, &blocks, options)?)
}

pub fn fit_poisson_log(
    spec: PoissonLogSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_poisson_log")?;
    validate_block_rows("eta", n, &spec.eta_block)?;

    let family = PoissonLogFamily {
        y: spec.y,
        weights: spec.weights,
    };
    let blocks = vec![spec.eta_block.into_spec("eta")?];
    Ok(fit_custom_family(&family, &blocks, options)?)
}

pub fn fit_gamma_log(
    spec: GammaLogSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_gamma_log")?;
    validate_block_rows("eta", n, &spec.eta_block)?;
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
    let blocks = vec![spec.eta_block.into_spec("eta")?];
    Ok(fit_custom_family(&family, &blocks, options)?)
}

pub fn fit_binomial_location_scale(
    spec: BinomialLocationScaleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_binomial_location_scale")?;
    validate_binomial_response(&spec.y, "fit_binomial_location_scale")?;
    validate_block_rows("threshold", n, &spec.threshold_block)?;
    validate_block_rows("log_sigma", n, &spec.log_sigma_block)?;

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

    if matches!(link_kind, InverseLink::Standard(LinkFunction::Probit)) {
        match try_binomial_alpha_beta_warm_start(
            &y,
            &weights,
            &threshold_block,
            &log_sigma_block,
            options,
        ) {
            Ok((beta_t0, beta_ls0, beta_warm)) => {
                threshold_block.initial_beta = Some(beta_t0);
                log_sigma_block.initial_beta = Some(beta_ls0);
                emit_binomial_alpha_beta_warnings("warm-start", &beta_warm, &y, &weights);
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
        threshold_block.into_spec("threshold")?,
        log_sigma_block.into_spec("log_sigma")?,
    ];
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: link_kind.clone(),
    };
    let fit = fit_custom_family(&family, &blocks, options)?;
    let beta_final = fit.block_states[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .eta
        .mapv(|eta| 1.0 / exp_sigma_from_eta_scalar(eta).max(1e-12));
    emit_binomial_alpha_beta_warnings("final-fit", &beta_final, &y, &weights);
    Ok(fit)
}

pub fn fit_binomial_location_scale_wiggle(
    spec: BinomialLocationScaleWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_binomial_location_scale_wiggle")?;
    validate_binomial_response(&spec.y, "fit_binomial_location_scale_wiggle")?;
    validate_block_rows("threshold", n, &spec.threshold_block)?;
    validate_block_rows("log_sigma", n, &spec.log_sigma_block)?;
    validate_block_rows("wiggle", n, &spec.wiggle_block)?;
    if spec.wiggle_degree < 1 {
        return Err(format!(
            "fit_binomial_location_scale_wiggle: wiggle_degree must be >= 1, got {}",
            spec.wiggle_degree
        ));
    }
    if spec.wiggle_knots.len() < spec.wiggle_degree + 2 {
        return Err(format!(
            "fit_binomial_location_scale_wiggle: wiggle_knots length {} is too short for degree {}",
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

    if (threshold_block.initial_beta.is_none() || log_sigma_block.initial_beta.is_none())
        && matches!(link_kind, InverseLink::Standard(LinkFunction::Probit))
    {
        match try_binomial_alpha_beta_warm_start(
            &y,
            &weights,
            &threshold_block,
            &log_sigma_block,
            options,
        ) {
            Ok((beta_t0, beta_ls0, beta_warm)) => {
                if threshold_block.initial_beta.is_none() {
                    threshold_block.initial_beta = Some(beta_t0);
                }
                if log_sigma_block.initial_beta.is_none() {
                    log_sigma_block.initial_beta = Some(beta_ls0);
                }
                emit_binomial_alpha_beta_warnings("warm-start-wiggle", &beta_warm, &y, &weights);
            }
            Err(err) => {
                log::warn!(
                    "[GAMLSS][fit_binomial_location_scale_wiggle] alpha/beta warm start failed, falling back to direct initialization: {}",
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
        threshold_block.into_spec("threshold")?,
        log_sigma_block.into_spec("log_sigma")?,
        wiggle_block.into_spec("wiggle")?,
    ];
    let fit = fit_custom_family(&family, &blocks, options)?;
    let beta_final = fit.block_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
        .eta
        .mapv(|eta| 1.0 / exp_sigma_from_eta_scalar(eta).max(1e-12));
    emit_binomial_alpha_beta_warnings("final-fit-wiggle", &beta_final, &y, &weights);
    Ok(fit)
}

trait LocationScaleFamilyBuilder {
    type Family: CustomFamily;

    fn mean_spec(&self) -> &TermCollectionSpec;
    fn noise_spec(&self) -> &TermCollectionSpec;
    fn exact_spatial_joint_supported(&self) -> bool {
        true
    }
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
    fn build_psi_derivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        mean_spec_resolved: &TermCollectionSpec,
        noise_spec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, mean_spec_resolved, mean_design)?
                .ok_or_else(|| "missing mean spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noise_spec_resolved, noise_design)?
                .ok_or_else(|| "missing noise spatial psi derivatives".to_string())?;
        Ok(vec![mean_derivs, noise_derivs])
    }
}

fn compose_theta_from_hints(
    mean_design: &TermCollectionDesign,
    noise_design: &TermCollectionDesign,
    mean_log_lambda_hint: &Option<Array1<f64>>,
    noise_log_lambda_hint: &Option<Array1<f64>>,
    extra_rho0: &Array1<f64>,
) -> Array1<f64> {
    let layout = GamlssLambdaLayout::with_wiggle(
        mean_design.penalties.len(),
        noise_design.penalties.len(),
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
    if layout.has_wiggle() {
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
    let extra_rho0 = builder.extra_rho0()?;

    let mean_boot_design =
        build_term_collection_design(data, builder.mean_spec()).map_err(|e| e.to_string())?;
    let noise_boot_design =
        build_term_collection_design(data, builder.noise_spec()).map_err(|e| e.to_string())?;
    let mean_boot_spec =
        freeze_spatial_length_scale_terms_from_design(builder.mean_spec(), &mean_boot_design)
            .map_err(|e| e.to_string())?;
    let noise_boot_spec =
        freeze_spatial_length_scale_terms_from_design(builder.noise_spec(), &noise_boot_design)
            .map_err(|e| e.to_string())?;

    let exact_joint_ready = builder.exact_spatial_joint_supported()
        && matches!(
            (
                build_block_spatial_psi_derivatives(data, &mean_boot_spec, &mean_boot_design)?,
                build_block_spatial_psi_derivatives(data, &noise_boot_spec, &noise_boot_design)?,
            ),
            (Some(_), Some(_))
        );

    let solved = if exact_joint_ready {
        let joint_setup = build_two_block_exact_joint_setup(
            builder.mean_spec(),
            builder.noise_spec(),
            mean_boot_design.penalties.len(),
            noise_boot_design.penalties.len(),
            extra_rho0.as_slice().unwrap_or(&[]),
            kappa_options,
        );
        let mean_beta_hint_cell = std::cell::RefCell::new(mean_beta_hint.clone());
        let noise_beta_hint_cell = std::cell::RefCell::new(noise_beta_hint.clone());
        match optimize_two_block_spatial_length_scale_exact_joint(
            data,
            builder.mean_spec(),
            builder.noise_spec(),
            kappa_options,
            &joint_setup,
            |rho, _mean_spec_resolved, _noise_spec_resolved, mean_design, noise_design| {
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
                    mean_design.penalties.len(),
                    noise_design.penalties.len(),
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
             mean_spec_resolved,
             noise_spec_resolved,
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
                let psi_derivative_blocks = builder.build_psi_derivative_blocks(
                    data,
                    mean_spec_resolved,
                    noise_spec_resolved,
                    mean_design,
                    noise_design,
                )?;
                let eval = evaluate_custom_family_joint_hyper(
                    &family,
                    &blocks,
                    options,
                    rho,
                    &psi_derivative_blocks,
                    None,
                    need_hessian,
                )?;
                Ok((eval.objective, eval.gradient, eval.outer_hessian))
            },
        ) {
            Ok(sol) => sol,
            Err(err) => {
                log::warn!(
                    "exact two-block spatial optimization failed ({}); falling back to finite-difference optimizer",
                    err
                );
                optimize_two_block_spatial_length_scale(
                    data,
                    builder.mean_spec(),
                    builder.noise_spec(),
                    kappa_options,
                    |mean_design, noise_design| {
                        let theta = compose_theta_from_hints(
                            mean_design,
                            noise_design,
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
                        let fit = fit_custom_family(&family, &blocks, options)?;
                        let layout = GamlssLambdaLayout::two_block(
                            mean_design.penalties.len(),
                            noise_design.penalties.len(),
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
                    |fit| fit.penalized_objective,
                )?
            }
        }
    } else {
        optimize_two_block_spatial_length_scale(
            data,
            builder.mean_spec(),
            builder.noise_spec(),
            kappa_options,
            |mean_design, noise_design| {
                let theta = compose_theta_from_hints(
                    mean_design,
                    noise_design,
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
                let fit = fit_custom_family(&family, &blocks, options)?;
                let layout = GamlssLambdaLayout::two_block(
                    mean_design.penalties.len(),
                    noise_design.penalties.len(),
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
            |fit| fit.penalized_objective,
        )?
    };

    Ok(BlockwiseTermFitResult {
        fit: solved.fit,
        mean_spec_resolved: solved.resolved_mean_spec,
        noise_spec_resolved: solved.resolved_noise_spec,
        mean_design: solved.mean_design,
        noise_design: solved.noise_design,
    })
}

struct GaussianLocationScaleTermBuilder {
    y: Array1<f64>,
    weights: Array1<f64>,
    mean_spec: TermCollectionSpec,
    noise_spec: TermCollectionSpec,
}

impl LocationScaleFamilyBuilder for GaussianLocationScaleTermBuilder {
    type Family = GaussianLocationScaleFamily;

    fn mean_spec(&self) -> &TermCollectionSpec {
        &self.mean_spec
    }

    fn noise_spec(&self) -> &TermCollectionSpec {
        &self.noise_spec
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
        let mut mean_spec = ParameterBlockSpec {
            name: "mu".to_string(),
            design: DesignMatrix::Dense(mean_design.design.clone()),
            offset: Array1::zeros(self.y.len()),
            penalties: mean_design.penalties.clone(),
            initial_log_lambdas: mean_log_lambdas,
            initial_beta: mean_beta_hint,
        };
        let mut noise_spec = ParameterBlockSpec {
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
        if mean_spec.initial_beta.is_none() || noise_spec.initial_beta.is_none() {
            let (beta_mu0, beta_ls0, _) = gaussian_location_scale_warm_start(
                &self.y,
                &self.weights,
                &mean_spec,
                &noise_spec,
                1e-10,
                mean_spec.initial_beta.as_ref(),
                noise_spec.initial_beta.as_ref(),
            )?;
            if mean_spec.initial_beta.is_none() {
                mean_spec.initial_beta = Some(beta_mu0);
            }
            if noise_spec.initial_beta.is_none() {
                noise_spec.initial_beta = Some(beta_ls0);
            }
        }
        Ok(vec![mean_spec, noise_spec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        let prepared_noise_design = prepared_gaussian_log_sigma_design(
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
            log_sigma_design: Some(DesignMatrix::Dense(prepared_noise_design)),
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
    mean_spec: TermCollectionSpec,
    noise_spec: TermCollectionSpec,
}

impl LocationScaleFamilyBuilder for BinomialLocationScaleTermBuilder {
    type Family = BinomialLocationScaleFamily;

    fn mean_spec(&self) -> &TermCollectionSpec {
        &self.mean_spec
    }

    fn noise_spec(&self) -> &TermCollectionSpec {
        &self.noise_spec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        false
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
        layout.validate_theta_len(theta.len(), "binomial location-scale")?;
        let identified_noise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)?;
        Ok(vec![
            ParameterBlockSpec {
                name: "threshold".to_string(),
                design: DesignMatrix::Dense(mean_design.design.clone()),
                offset: Array1::zeros(self.y.len()),
                penalties: mean_design.penalties.clone(),
                initial_log_lambdas: layout.mean_from(theta),
                initial_beta: mean_beta_hint,
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(identified_noise_design),
                offset: Array1::zeros(self.y.len()),
                penalties: noise_design.penalties.clone(),
                initial_log_lambdas: layout.noise_from(theta),
                initial_beta: noise_beta_hint,
            },
        ])
    }

    fn build_family(
        &self,
        _mean_design: &TermCollectionDesign,
        _noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        BinomialLocationScaleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
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
    mean_spec: TermCollectionSpec,
    noise_spec: TermCollectionSpec,
    wiggle_knots: Array1<f64>,
    wiggle_degree: usize,
    wiggle_block: ParameterBlockInput,
}

impl LocationScaleFamilyBuilder for BinomialLocationScaleWiggleTermBuilder {
    type Family = BinomialLocationScaleWiggleFamily;

    fn mean_spec(&self) -> &TermCollectionSpec {
        &self.mean_spec
    }

    fn noise_spec(&self) -> &TermCollectionSpec {
        &self.noise_spec
    }

    fn exact_spatial_joint_supported(&self) -> bool {
        false
    }

    fn extra_rho0(&self) -> Result<Array1<f64>, String> {
        initial_log_lambdas_or_zeros(&self.wiggle_block)
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::with_wiggle(
            mean_design.penalties.len(),
            noise_design.penalties.len(),
            self.wiggle_block.penalties.len(),
        );
        layout.validate_theta_len(theta.len(), "wiggle location-scale")?;
        let identified_noise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)?;
        Ok(vec![
            ParameterBlockSpec {
                name: "threshold".to_string(),
                design: DesignMatrix::Dense(mean_design.design.clone()),
                offset: Array1::zeros(self.y.len()),
                penalties: mean_design.penalties.clone(),
                initial_log_lambdas: layout.mean_from(theta),
                initial_beta: mean_beta_hint,
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(identified_noise_design),
                offset: Array1::zeros(self.y.len()),
                penalties: noise_design.penalties.clone(),
                initial_log_lambdas: layout.noise_from(theta),
                initial_beta: noise_beta_hint,
            },
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
        let identified_noise_design =
            identified_binomial_log_sigma_design(mean_design, noise_design, &self.weights)
                .expect("identified binomial log-sigma design should match block construction");
        BinomialLocationScaleWiggleFamily {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(DesignMatrix::Dense(mean_design.design.clone())),
            log_sigma_design: Some(DesignMatrix::Dense(identified_noise_design)),
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

    fn build_psi_derivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        mean_spec_resolved: &TermCollectionSpec,
        noise_spec_resolved: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        let mean_derivs =
            build_block_spatial_psi_derivatives(data, mean_spec_resolved, mean_design)?
                .ok_or_else(|| "missing threshold spatial psi derivatives".to_string())?;
        let noise_derivs =
            build_block_spatial_psi_derivatives(data, noise_spec_resolved, noise_design)?
                .ok_or_else(|| "missing log_sigma spatial psi derivatives".to_string())?;
        Ok(vec![mean_derivs, noise_derivs, Vec::new()])
    }
}

pub fn fit_gaussian_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_gaussian_location_scale_term_spec(data, &spec, "fit_gaussian_location_scale_terms")?;
    fit_location_scale_terms(
        data,
        GaussianLocationScaleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            mean_spec: spec.mean_spec,
            noise_spec: spec.log_sigma_spec,
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
    validate_binomial_location_scale_term_spec(data, &spec, "fit_binomial_location_scale_terms")?;
    fit_location_scale_terms(
        data,
        BinomialLocationScaleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            mean_spec: spec.threshold_spec,
            noise_spec: spec.log_sigma_spec,
        },
        options,
        kappa_options,
    )
}

pub fn fit_binomial_location_scale_wiggle_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleWiggleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    validate_binomial_location_scale_wiggle_term_spec(
        data,
        &spec,
        "fit_binomial_location_scale_wiggle_terms",
    )?;
    fit_location_scale_terms(
        data,
        BinomialLocationScaleWiggleTermBuilder {
            y: spec.y,
            weights: spec.weights,
            link_kind: spec.link_kind,
            mean_spec: spec.threshold_spec,
            noise_spec: spec.log_sigma_spec,
            wiggle_knots: spec.wiggle_knots,
            wiggle_degree: spec.wiggle_degree,
            wiggle_block: spec.wiggle_block,
        },
        options,
        kappa_options,
    )
}

pub fn fit_binomial_location_scale_wiggle_terms_auto(
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
            threshold_spec: spec.threshold_spec.clone(),
            log_sigma_spec: spec.log_sigma_spec.clone(),
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
    let identified_noise_design = identified_binomial_log_sigma_design(
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
        build_wiggle_block_input_from_seed(q_seed.view(), &wiggle_cfg)?;
    let p_w = wiggle_block.design.ncols();
    for &ord in wiggle_penalty_orders {
        if ord <= 1 || ord >= p_w {
            continue;
        }
        let s = create_difference_penalty_matrix(p_w, ord, None).map_err(|e| e.to_string())?;
        wiggle_block.penalties.push(s);
    }

    let fit = fit_binomial_location_scale_wiggle(
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
                design: DesignMatrix::Dense(identified_noise_design),
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
            mean_spec_resolved: pilot.mean_spec_resolved,
            noise_spec_resolved: pilot.noise_spec_resolved,
            mean_design: pilot.mean_design,
            noise_design: pilot.noise_design,
        },
        wiggle_knots,
        wiggle_degree: wiggle_cfg.degree,
    })
}

pub fn fit_binomial_location_scale_terms_workflow(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleTermSpec,
    wiggle: Option<BinomialLocationScaleWiggleWorkflowConfig>,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BinomialLocationScaleWorkflowResult, String> {
    if let Some(wiggle_cfg) = wiggle {
        let solved = fit_binomial_location_scale_wiggle_terms_auto(
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
        let beta_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(BinomialLocationScaleWorkflowResult {
            fit: BlockwiseTermFitResult {
                fit,
                mean_spec_resolved: solved.fit.mean_spec_resolved,
                noise_spec_resolved: solved.fit.noise_spec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            },
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            beta_wiggle,
        })
    } else {
        let solved = fit_binomial_location_scale_terms(data, spec, options, kappa_options)?;
        Ok(BinomialLocationScaleWorkflowResult {
            fit: solved,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_wiggle: None,
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

fn signed_with_floor(v: f64, floor: f64) -> f64 {
    let a = v.abs().max(floor);
    if v >= 0.0 { a } else { -a }
}

#[inline]
fn binomial_score_curvature_third_from_jet(
    y: f64,
    weight: f64,
    clamp_active: bool,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    if clamp_active {
        return (0.0, 0.0, 0.0);
    }
    // Binomial derivatives wrt q via mu:
    // Per-row log-likelihood is represented in weighted-proportion form:
    //   ell_i = m_i * [ y_i log(mu_i) + (1-y_i) log(1-mu_i) ],
    // where `weight = m_i` and `y` is the observed proportion in [0,1].
    //
    // mu-space derivatives:
    //   ell_mu    = y/mu - (1-y)/(1-mu)
    //   ell_mumu  = -y/mu^2 - (1-y)/(1-mu)^2
    //   ell_mumum = 2y/mu^3 - 2(1-y)/(1-mu)^3
    //
    // q-jet using mu(q) derivatives d1=mu', d2=mu'', d3=mu''':
    //   s = dell/dq   = ell_mu * mu'
    //   c = d2ell/dq2 = ell_mumu*(mu')^2 + ell_mu*mu''
    //   t = d3ell/dq3 = ell_mumum*(mu')^3 + 3*ell_mumu*mu'*mu'' + ell_mu*mu'''
    //
    // Returns (score_q, curvature_q, third_q) with curvature_q = -d2ell/dq2.
    let m = mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let one_minus = (1.0 - m).max(MIN_PROB);
    let ell_mu = y / m - (1.0 - y) / one_minus;
    let ell_mumu = -y / (m * m) - (1.0 - y) / (one_minus * one_minus);
    let ell_mumum = 2.0 * y / (m * m * m) - 2.0 * (1.0 - y) / (one_minus * one_minus * one_minus);

    let score_q = weight * ell_mu * d1;
    let d2ell_dq2 = weight * (ell_mumu * d1 * d1 + ell_mu * d2);
    let curvature_q = -d2ell_dq2;
    let third_q = weight * (ell_mumum * d1 * d1 * d1 + 3.0 * ell_mumu * d1 * d2 + ell_mu * d3);
    (score_q, curvature_q, third_q)
}

#[inline]
fn binomial_neglog_q_derivatives_from_jet(
    y: f64,
    weight: f64,
    clamp_active: bool,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    // Returns (m1,m2,m3) for F_i(q) = -ell_i(q):
    //   m1 = dF/dq, m2 = d²F/dq², m3 = d³F/dq³.
    let (score_q, curvature_q, third_q) =
        binomial_score_curvature_third_from_jet(y, weight, clamp_active, mu, d1, d2, d3);
    (-score_q, curvature_q, -third_q)
}

#[inline]
fn binomial_neglog_q_derivatives_probit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
    clamp_active: bool,
    mu: f64,
) -> (f64, f64, f64) {
    if clamp_active {
        return (0.0, 0.0, 0.0);
    }
    // Closed-form derivatives for F_i(q) = -w_i[y log Phi(q) + (1-y) log(1-Phi(q))].
    // Uses canonical A/A_mu/A_mumu identities from the probit composition.
    let m = mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let nu = (1.0 - m).max(MIN_PROB);
    let phi = normal_pdf(q);
    let a = (1.0 - y) / nu - y / m;
    let a_mu = (1.0 - y) / (nu * nu) + y / (m * m);
    let a_mumu = 2.0 * (1.0 - y) / (nu * nu * nu) - 2.0 * y / (m * m * m);

    let m1 = weight * a * phi;
    let m2 = weight * (a_mu * phi * phi - q * a * phi);
    let m3 =
        weight * (a_mumu * phi * phi * phi - 3.0 * q * a_mu * phi * phi + (q * q - 1.0) * a * phi);
    (m1, m2, m3)
}

#[inline]
fn binomial_neglog_q_fourth_derivative_probit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
    clamp_active: bool,
    mu: f64,
) -> f64 {
    if clamp_active {
        return 0.0;
    }
    // Closed-form m4 for F_i(q) = -w_i[y log Phi(q) + (1-y) log(1-Phi(q))].
    let m = mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let nu = (1.0 - m).max(MIN_PROB);
    let phi = normal_pdf(q);
    let a = (1.0 - y) / nu - y / m;
    let a_mu = (1.0 - y) / (nu * nu) + y / (m * m);
    let a_mumu = 2.0 * (1.0 - y) / (nu * nu * nu) - 2.0 * y / (m * m * m);
    let a_mumumu = 6.0 * (1.0 - y) / (nu * nu * nu * nu) + 6.0 * y / (m * m * m * m);
    weight
        * (a_mumumu * phi.powi(4) - 6.0 * q * a_mumu * phi.powi(3)
            + (7.0 * q * q - 4.0) * a_mu * phi * phi
            - (q * q * q - 3.0 * q) * a * phi)
}

#[inline]
fn binomial_fourth_from_jet(
    y: f64,
    weight: f64,
    clamp_active: bool,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
) -> f64 {
    if clamp_active {
        return 0.0;
    }
    let m = mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let one_minus = (1.0 - m).max(MIN_PROB);
    let ell_mu = y / m - (1.0 - y) / one_minus;
    let ell_mumu = -y / (m * m) - (1.0 - y) / (one_minus * one_minus);
    let ell_mumum = 2.0 * y / (m * m * m) - 2.0 * (1.0 - y) / (one_minus * one_minus * one_minus);
    let ell_mumumum = -6.0 * y / (m * m * m * m)
        - 6.0 * (1.0 - y) / (one_minus * one_minus * one_minus * one_minus);
    weight
        * (ell_mumumum * d1.powi(4)
            + 6.0 * ell_mumum * d1 * d1 * d2
            + 3.0 * ell_mumu * d2 * d2
            + 4.0 * ell_mumu * d1 * d3
            + ell_mu * d4)
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

fn assemble_two_block_symmetric(
    upper_left: &Array2<f64>,
    upper_right: &Array2<f64>,
    lower_right: &Array2<f64>,
) -> Array2<f64> {
    let pt = upper_left.nrows();
    let pls = lower_right.nrows();
    let total = pt + pls;
    let mut out = Array2::<f64>::zeros((total, total));
    out.slice_mut(s![0..pt, 0..pt]).assign(upper_left);
    out.slice_mut(s![0..pt, pt..total]).assign(upper_right);
    out.slice_mut(s![pt..total, pt..total]).assign(lower_right);
    mirror_upper_to_lower(&mut out);
    out
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
    clamp_active: Vec<bool>,
    dmu_dq: Array1<f64>,
    d2mu_dq2: Array1<f64>,
    d3mu_dq3: Array1<f64>,
    log_likelihood: f64,
}

#[derive(Clone, Copy)]
struct NonWiggleQDerivs {
    q_t: f64,
    q_ls: f64,
    q_tt: f64,
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
    delta_q_tt: f64,
    delta_q_tl: f64,
    delta_q_ll: f64,
}

#[derive(Clone, Copy)]
struct EtaTwoBlockJet {
    grad_t: f64,
    grad_ls: f64,
    w_tt: f64,
    w_tl: f64,
    w_ll: f64,
}

#[derive(Clone, Copy)]
struct ClampedInverseLinkRow {
    mu: f64,
    clamp_active: bool,
    d1: f64,
    d2: f64,
    d3: f64,
}

#[derive(Clone, Copy)]
struct BinomialLocationScaleRow {
    sigma: f64,
    dsigma_deta: f64,
    q0: f64,
    clamped_mu: ClampedInverseLinkRow,
    ll: f64,
}

/// Chain rule in (eta_t, eta_ls) space for two coupled blocks.
///
/// With q = q(eta_t, eta_ls), score s = d ell / dq, and
/// c = d2 ell / dq2, the eta-space derivatives are:
///   d ell / d eta_t  = s * q_t
///   d ell / d eta_ls = s * q_ls
///   d2 ell / d eta_a d eta_b = c * q_a q_b + s * q_ab.
///
/// In this project, `curvature_q` stores -d2 ell / dq2, so c = -curvature_q.
/// `ExactNewton` requires negative log-likelihood Hessian contributions:
///   w_ab = - d2 ell / d eta_a d eta_b.
fn eta_two_block_jet_from_q(
    score_q: f64,
    curvature_q: f64,
    q_t: f64,
    q_ls: f64,
    q_tt: f64,
    q_tl: f64,
    q_ll: f64,
) -> EtaTwoBlockJet {
    // Full rowwise calculus for eta=(eta_t,eta_ls):
    //
    // Let ell(q) be per-row log-likelihood, with
    //   s := d ell/dq,   c := d² ell/dq².
    // Define objective piece as negative log-likelihood in eta:
    //   F(eta_t,eta_ls) = -ell(q(eta_t,eta_ls)).
    //
    // First derivatives:
    //   F_t  = -(d ell/dq) q_t  = -s q_t,
    //   F_ls = -(d ell/dq) q_ls = -s q_ls.
    //
    // Code stores gradient entries with sign convention consistent with
    // `score_q = d ell/dq`, hence `grad_* = score_q * q_*` here.
    //
    // Second derivatives (Hessian of -ell wrt eta):
    //   F_ab
    //   = - d/d eta_b (s q_a)
    //   = -( (d s/d eta_b) q_a + s q_ab )
    //   = -( c q_b q_a + s q_ab ).
    //
    // Therefore
    //   w_tt = -(c q_t q_t + s q_tt),
    //   w_tl = -(c q_t q_ls + s q_tl),
    //   w_ll = -(c q_ls q_ls + s q_ll).
    //
    // In this module `curvature_q` is ell''(q), usually non-positive near the
    // mode for binomial/probit; we write c_q = -curvature_q to match the
    // positive-curvature convention used in these Newton weights.
    let c_q = -curvature_q;
    EtaTwoBlockJet {
        grad_t: score_q * q_t,
        grad_ls: score_q * q_ls,
        w_tt: -(c_q * q_t * q_t + score_q * q_tt),
        w_tl: -(c_q * q_t * q_ls + score_q * q_tl),
        w_ll: -(c_q * q_ls * q_ls + score_q * q_ll),
    }
}

/// Generic directional derivative of Newton weight:
///   w_ab = -(c q_a q_b + s q_ab), with c = d2 ell / dq2.
/// Along direction u:
///   delta w_ab =
///     -[ t delta_q q_a q_b
///        + c(delta q_a q_b + q_a delta q_b)
///        + c delta_q q_ab
///        + s delta q_ab ].
///
/// Inputs use project convention `curvature_q = -c`.
fn delta_newton_weight_from_q_terms(
    score_q: f64,
    curvature_q: f64,
    third_q: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    delta_q: f64,
    delta_q_a: f64,
    delta_q_b: f64,
    delta_q_ab: f64,
) -> f64 {
    // Start from
    //   w_ab = -(c q_a q_b + s q_ab).
    //
    // Directional derivative along u:
    //   dot{w}_ab
    //   = -dot(c q_a q_b + s q_ab)
    //   = -( dot{c} q_a q_b + c dot{q}_a q_b + c q_a dot{q}_b
    //       + dot{s} q_ab + s dot{q}_{ab} ).
    //
    // With chain identities wrt q:
    //   dot{s} = c dot{q},   dot{c} = t dot{q},
    // this becomes
    //   dot{w}_ab
    //   = -( t dot{q} q_a q_b
    //       + c(dot{q}_a q_b + q_a dot{q}_b)
    //       + c dot{q} q_ab
    //       + s dot{q}_{ab} ).
    //
    // Parameter mapping here:
    //   delta_q    -> dot{q},
    //   delta_q_a  -> dot{q}_a,
    //   delta_q_b  -> dot{q}_b,
    //   delta_q_ab -> dot{q}_{ab}.
    let c_q = -curvature_q;
    -(third_q * delta_q * q_a * q_b
        + c_q * (delta_q_a * q_b + q_a * delta_q_b)
        + c_q * delta_q * q_ab
        + score_q * delta_q_ab)
}

#[inline]
fn hessian_coeff_from_objective_q_terms(m1: f64, m2: f64, q_a: f64, q_b: f64, q_ab: f64) -> f64 {
    // F = -sum ell, scalar q:
    //   H_ab = m2 * q_a q_b + m1 * q_ab.
    m2 * q_a * q_b + m1 * q_ab
}

#[inline]
fn directional_hessian_coeff_from_objective_q_terms(
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
fn second_directional_hessian_coeff_from_objective_q_terms(
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    dq_u: f64,
    dq_v: f64,
    d2q_uv: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    dq_a_u: f64,
    dq_a_v: f64,
    dq_b_u: f64,
    dq_b_v: f64,
    d2q_a_uv: f64,
    d2q_b_uv: f64,
    dq_ab_u: f64,
    dq_ab_v: f64,
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
    //      m4*dq_u*dq_v*q_a*q_b
    //    + m3*(d2q_uv*q_a*q_b
    //         + dq_u*(dq_a_v*q_b + q_a*dq_b_v)
    //         + dq_v*(dq_a_u*q_b + q_a*dq_b_u)
    //         + dq_u*dq_v*q_ab)
    //    + m2*(d2q_a_uv*q_b + dq_a_u*dq_b_v + dq_a_v*dq_b_u + q_a*d2q_b_uv
    //          + d2q_uv*q_ab + dq_u*dq_ab_v + dq_v*dq_ab_u)
    //    + m1*d2q_ab_uv.
    //
    // The single dq_u*dq_v*q_ab term is important. There is exactly one copy:
    //
    //   D_v[m2 * dq_u * B]
    //   = m3 * dq_v * dq_u * B + m2 * (d2q_uv * B + dq_u * D_v B),
    //
    // and no second copy appears elsewhere. A previous version of this helper
    // accidentally counted this term twice by embedding `dq_v * q_ab` in both
    // the `dq_u` and `dq_v` product-rule branches.
    let d_qaqb_u = dq_a_u * q_b + q_a * dq_b_u;
    let d_qaqb_v = dq_a_v * q_b + q_a * dq_b_v;
    let d2_qaqb_uv = d2q_a_uv * q_b + dq_a_u * dq_b_v + dq_a_v * dq_b_u + q_a * d2q_b_uv;
    m4 * dq_u * dq_v * q_a * q_b
        + m3 * (d2q_uv * q_a * q_b + dq_u * d_qaqb_v + dq_v * d_qaqb_u + dq_u * dq_v * q_ab)
        + m2 * (d2_qaqb_uv + d2q_uv * q_ab + dq_u * dq_ab_v + dq_v * dq_ab_u)
        + m1 * d2q_ab_uv
}

#[inline]
fn second_delta_newton_weight_from_q_terms(
    score_q: f64,
    curvature_q: f64,
    third_q: f64,
    fourth_q: f64,
    q_ab_term: f64,
    d_q_u: f64,
    d_q_v: f64,
    d2_q_uv: f64,
    a0: f64,
    a_u: f64,
    a_v: f64,
    a_uv: f64,
    b_u: f64,
    b_v: f64,
    b_uv: f64,
) -> f64 {
    // Exact symmetric bilinear second variation for the generic weight
    //   w = -(c a + s b),
    // where, in eta-space:
    //   a  is one of (q_t^2, q_t q_ls, q_ls^2),
    //   b  is one of (q_tt, q_tl, q_ll),
    // and wrt q:
    //   s = l'(q), c = l''(q), t = l'''(q), f = l''''(q).
    //
    // Directional chain identities along u and v:
    //   Ds[u] = c dq_u,    Dc[u] = t dq_u,
    //   D²s[u,v] = t dq_u dq_v + c d²q_uv,
    //   D²c[u,v] = f dq_u dq_v + t d²q_uv.
    //
    // Product-rule expansion:
    //   D²w[u,v] = -(
    //      D²c[u,v] a
    //    + Dc[u] Da[v] + Dc[v] Da[u] + c D²a[u,v]
    //    + D²s[u,v] b
    //    + Ds[u] Db[v] + Ds[v] Db[u] + s D²b[u,v] ).
    //
    // Argument mapping in this helper:
    //   a0   = a,      a_u = Da[u],   a_v = Da[v],   a_uv = D²a[u,v]
    //   q_ab_term = b, b_u = Db[u],   b_v = Db[v],   b_uv = D²b[u,v].
    let c_q = -curvature_q;
    let ds_u = c_q * d_q_u;
    let ds_v = c_q * d_q_v;
    let dc_u = third_q * d_q_u;
    let dc_v = third_q * d_q_v;
    let ddc_uv = fourth_q * d_q_v * d_q_u + third_q * d2_q_uv;
    let dds_uv = third_q * d_q_v * d_q_u + c_q * d2_q_uv;
    -(ddc_uv * a0
        + dc_u * a_v
        + dc_v * a_u
        + c_q * a_uv
        + dds_uv * q_ab_term
        + ds_u * b_v
        + ds_v * b_u
        + score_q * b_uv)
}

/// Non-wiggle location-scale map derivatives:
/// q(eta_t, eta_ls) = -eta_t / sigma(eta_ls), with:
/// q_t=-1/sigma, q_ls=eta_t*sigma'/sigma^2, q_tt=0, q_tl=sigma'/sigma^2,
/// q_ll=eta_t*(sigma''/sigma^2 - 2*(sigma')^2/sigma^3),
/// q_tl_ls=sigma''/sigma^2 - 2*(sigma')^2/sigma^3,
/// q_ll_ls=eta_t*(sigma'''/sigma^2 - 6*sigma'*sigma''/sigma^3 + 6*(sigma')^3/sigma^4).
fn nonwiggle_q_derivs(
    eta_t: f64,
    sigma: f64,
    dsigma: f64,
    d2sigma: f64,
    d3sigma: f64,
) -> NonWiggleQDerivs {
    // Full quotient-rule derivation for q(eta_t,eta_ls) = -eta_t / sigma(eta_ls):
    //
    // Write s = sigma(eta_ls), s' = dsigma/d eta_ls, s'' = d²sigma/d eta_ls².
    //
    // 1) First partials
    //   q_t  = ∂q/∂eta_t  = -1/s.
    //   q_ls = ∂q/∂eta_ls = -eta_t * ∂(1/s)/∂eta_ls
    //        = -eta_t * (-(s'/s²)) = eta_t s'/s².
    //
    // 2) Second partials
    //   q_tt = ∂/∂eta_t (-1/s) = 0.
    //   q_tl = ∂/∂eta_ls (-1/s) = s'/s².
    //   q_ll = ∂/∂eta_ls (eta_t s'/s²)
    //        = eta_t * ( s''/s² - 2(s')²/s³ ).
    //
    // 3) Third eta_ls derivatives needed by directional Hessian calculus
    //   q_tl_ls = ∂/∂eta_ls (q_tl)
    //           = s''/s² - 2(s')²/s³.
    //   q_ll_ls = ∂/∂eta_ls (q_ll)
    //           = eta_t * [ s'''/s² - 6 s' s''/s³ + 6 (s')³/s⁴ ].
    //
    // The last line comes from differentiating each product/quotient term:
    //   d/dx [s'' s^{-2}]            = s''' s^{-2} - 2 s' s'' s^{-3},
    //   d/dx [-2 (s')² s^{-3}]       = -4 s' s'' s^{-3} + 6 (s')³ s^{-4},
    // and summing.
    let s = sigma.max(1e-12);
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    let q_tl_ls = d2sigma / s2 - 2.0 * dsigma * dsigma / s3;
    NonWiggleQDerivs {
        q_t: -1.0 / s,
        q_ls: eta_t * dsigma / s2,
        q_tt: 0.0,
        q_tl: dsigma / s2,
        q_ll: eta_t * q_tl_ls,
        q_tl_ls,
        q_ll_ls: eta_t
            * (d3sigma / s2 - 6.0 * dsigma * d2sigma / s3 + 6.0 * dsigma * dsigma * dsigma / s4),
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
        delta_q_tt: 0.0,
        delta_q_tl: q.q_tl_ls * d_eta_ls,
        delta_q_ll: q.q_tl_ls * d_eta_t + q.q_ll_ls * d_eta_ls,
    }
}

#[inline]
fn clamped_inverse_link_row(jet: crate::mixture_link::InverseLinkJet) -> ClampedInverseLinkRow {
    let raw_mu = jet.mu;
    let mu = raw_mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let clamp_active = raw_mu <= MIN_PROB || raw_mu >= 1.0 - MIN_PROB;
    if clamp_active {
        ClampedInverseLinkRow {
            mu,
            clamp_active,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        }
    } else {
        ClampedInverseLinkRow {
            mu,
            clamp_active,
            d1: jet.d1,
            d2: jet.d2,
            d3: jet.d3,
        }
    }
}

#[inline]
fn binomial_location_scale_q0(eta_t: f64, sigma: f64) -> f64 {
    -eta_t / sigma.max(1e-12)
}

fn binomial_location_scale_row(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    eta_wiggle: f64,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleRow, String> {
    let SigmaJet1 {
        sigma,
        d1: dsigma_deta,
    } = exp_sigma_jet1_scalar(eta_ls);
    let q0 = binomial_location_scale_q0(eta_t, sigma);
    let q = q0 + eta_wiggle;
    let jet = inverse_link_jet_for_inverse_link(link_kind, q)
        .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
    let clamped_mu = clamped_inverse_link_row(jet);
    let ll = weight * (y * clamped_mu.mu.ln() + (1.0_f64 - y) * (1.0_f64 - clamped_mu.mu).ln());
    Ok(BinomialLocationScaleRow {
        sigma,
        dsigma_deta,
        q0,
        clamped_mu,
        ll,
    })
}

fn binomial_location_scale_core(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    eta_wiggle: Option<&Array1<f64>>,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleCore, String> {
    let n = y.len();
    if weights.len() != n || eta_t.len() != n || eta_ls.len() != n {
        return Err("binomial location-scale core size mismatch".to_string());
    }
    if let Some(w) = eta_wiggle
        && w.len() != n
    {
        return Err("binomial location-scale core wiggle size mismatch".to_string());
    }

    let mut sigma = Array1::<f64>::zeros(n);
    let mut dsigma_deta = Array1::<f64>::zeros(n);
    let mut q0 = Array1::<f64>::zeros(n);
    let mut mu = Array1::<f64>::zeros(n);
    let mut clamp_active = vec![false; n];
    let mut dmu_dq = Array1::<f64>::zeros(n);
    let mut d2mu_dq2 = Array1::<f64>::zeros(n);
    let mut d3mu_dq3 = Array1::<f64>::zeros(n);
    let mut ll = 0.0;

    for i in 0..n {
        let row = binomial_location_scale_row(
            y[i],
            weights[i],
            eta_t[i],
            eta_ls[i],
            eta_wiggle.map_or(0.0, |w| w[i]),
            link_kind,
        )?;
        sigma[i] = row.sigma;
        dsigma_deta[i] = row.dsigma_deta;
        q0[i] = row.q0;
        mu[i] = row.clamped_mu.mu;
        clamp_active[i] = row.clamped_mu.clamp_active;
        dmu_dq[i] = row.clamped_mu.d1;
        d2mu_dq2[i] = row.clamped_mu.d2;
        d3mu_dq3[i] = row.clamped_mu.d3;
        ll += row.ll;
    }

    Ok(BinomialLocationScaleCore {
        sigma,
        dsigma_deta,
        q0,
        mu,
        clamp_active,
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
    eta_wiggle: Option<&Array1<f64>>,
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
        let mut grad_q = eta_wiggle.map(|_| Array1::<f64>::zeros(n));
        let mut h_q_psd = eta_wiggle.map(|_| Array1::<f64>::zeros(n));

        for i in 0..n {
            let (score_q, curvature_q, _third_q) = binomial_score_curvature_third_from_jet(
                y[i],
                weights[i],
                core.clamp_active[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
            );
            let a_i = dq_dq0.map_or(1.0, |v| v[i]);
            let c_i = geom.d2q_dq02.map_or(0.0, |v| v[i]);
            let s = core.sigma[i].max(1e-12);

            let dq_t = -a_i / s;
            let d2q_t = c_i / (s * s);

            let dq0_ls = -core.q0[i] * core.dsigma_deta[i] / s;
            let d2q0_ls = eta_t[i]
                * (geom.d2sigma_deta2[i] / (s * s)
                    - 2.0 * core.dsigma_deta[i] * core.dsigma_deta[i] / (s * s * s));
            let dq_ls = a_i * dq0_ls;
            let d2q_ls = a_i * d2q0_ls + c_i * dq0_ls * dq0_ls;
            let eta_jet =
                eta_two_block_jet_from_q(score_q, curvature_q, dq_t, dq_ls, d2q_t, 0.0, d2q_ls);
            grad_eta_t[i] = eta_jet.grad_t;
            grad_eta_ls[i] = eta_jet.grad_ls;
            h_eta_t[i] = eta_jet.w_tt;
            h_eta_ls[i] = eta_jet.w_ll;

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
        let w_ws = match (geom.wiggle_design, grad_q, h_q_psd) {
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
            w_ws,
        ));
    }

    let mut z_t = Array1::<f64>::zeros(n);
    let mut w_t = Array1::<f64>::zeros(n);
    let mut z_ls = Array1::<f64>::zeros(n);
    let mut w_ls = Array1::<f64>::zeros(n);
    let mut z_w = eta_wiggle.map(|_| Array1::<f64>::zeros(n));
    let mut w_w = eta_wiggle.map(|_| Array1::<f64>::zeros(n));

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
            w_t[i] = floor_positive_weight(weights[i] * (dmu_t * dmu_t / var), MIN_WEIGHT);
            z_t[i] = eta_t[i] + (y[i] - core.mu[i]) / signed_with_floor(dmu_t, MIN_DERIV);
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
            w_ls[i] = floor_positive_weight(weights[i] * (dmu_ls * dmu_ls / var), MIN_WEIGHT);
            z_ls[i] = eta_ls[i] + (y[i] - core.mu[i]) / signed_with_floor(dmu_ls, MIN_DERIV);
        }

        if let (Some(eta_w), Some(z_wv), Some(w_wv)) = (eta_wiggle, z_w.as_mut(), w_w.as_mut()) {
            // Wiggle enters additively in q, so chain is 1.
            let dmu_w = core.dmu_dq[i];
            if weights[i] == 0.0 || dmu_w == 0.0 {
                w_wv[i] = 0.0;
                z_wv[i] = eta_w[i];
            } else {
                w_wv[i] = floor_positive_weight(weights[i] * (dmu_w * dmu_w / var), MIN_WEIGHT);
                z_wv[i] = eta_w[i] + (y[i] - core.mu[i]) / signed_with_floor(dmu_w, MIN_DERIV);
            }
        }
    }

    let t_ws = BlockWorkingSet::Diagonal {
        working_response: z_t,
        working_weights: w_t,
    };
    let ls_ws = BlockWorkingSet::Diagonal {
        working_response: z_ls,
        working_weights: w_ls,
    };
    let w_ws = match (z_w, w_w) {
        (Some(z), Some(w)) => Some(BlockWorkingSet::Diagonal {
            working_response: z,
            working_weights: w,
        }),
        _ => None,
    };
    Ok((t_ws, ls_ws, w_ws))
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

#[inline]
fn exp_sigma_jet1_scalar(eta: f64) -> SigmaJet1 {
    let sigma = eta.exp();
    SigmaJet1 { sigma, d1: sigma }
}

#[inline]
fn exp_sigma_from_eta_scalar(eta: f64) -> f64 {
    eta.exp()
}

#[inline]
fn exp_sigma_derivs_up_to_third(
    eta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let sigma = eta.mapv(f64::exp);
    (sigma.clone(), sigma.clone(), sigma.clone(), sigma)
}

impl GaussianLocationScaleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub fn parameter_names() -> &'static [&'static str] {
        &["mu", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scale",
            parameter_names: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
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
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_mu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err("GaussianLocationScaleFamily input size mismatch".to_string());
        }

        let mut sigma = Array1::<f64>::zeros(n);
        let mut dsigma_deta = Array1::<f64>::zeros(n);
        let mut ll = 0.0;

        for i in 0..n {
            let sigma_i = eta_log_sigma[i].exp();
            let dsigma_deta_i = sigma_i;
            sigma[i] = sigma_i;
            dsigma_deta[i] = dsigma_deta_i;
            let r = self.y[i] - eta_mu[i];
            let s2 = (sigma[i] * sigma[i]).max(1e-20);
            ll += self.weights[i] * (-0.5 * (r * r / s2 + (2.0 * std::f64::consts::PI * s2).ln()));
        }

        let mut z_mu = Array1::<f64>::zeros(n);
        let mut w_mu = Array1::<f64>::zeros(n);
        let mut z_ls = Array1::<f64>::zeros(n);
        let mut w_ls = Array1::<f64>::zeros(n);

        for i in 0..n {
            let r = self.y[i] - eta_mu[i];
            let s = sigma[i].max(1e-10);
            let s2 = (s * s).max(1e-20);

            // mu block (identity): canonical WLS
            if self.weights[i] == 0.0 {
                w_mu[i] = 0.0;
                z_mu[i] = eta_mu[i];
            } else {
                w_mu[i] = floor_positive_weight(self.weights[i] / s2, MIN_WEIGHT);
                z_mu[i] = eta_mu[i] + r;
            }

            // log-sigma block: IRLS working response and weights.
            // The score for log-sigma is d(ll)/d(eta_ls) = (r²/s² - 1) * dsigma/(sigma * deta).
            // Working response: z_ls = eta_ls + score / info.
            let (dlogsigma_du, info_u) =
                gaussian_log_sigma_irls_info(self.weights[i], s, dsigma_deta[i]);
            if info_u == 0.0 {
                w_ls[i] = 0.0;
                z_ls[i] = eta_log_sigma[i];
            } else {
                w_ls[i] = info_u;
                // Score: d(ll)/d(eta_ls) = w_i * (r²/s² - 1) * dsigma/sigma
                let score_ls = self.weights[i] * (r * r / s2 - 1.0) * dlogsigma_du;
                z_ls[i] = eta_log_sigma[i] + score_ls / info_u;
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            block_working_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: z_mu,
                    working_weights: w_mu,
                },
                BlockWorkingSet::Diagonal {
                    working_response: z_ls,
                    working_weights: w_ls,
                },
            ],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        _block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn exact_newton_joint_hessian_second_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _d_beta_u_flat: &Array1<f64>,
        _d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn diagonal_working_weights_directional_derivative(
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
                //   w_mu = weight / sigma^2.
                //
                // This depends only on the scale predictor, so along a
                // location-only direction d eta_mu the directional derivative is
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
                    dw[i] = gaussian_log_sigma_irls_info_directional_derivative(
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
}

impl CustomFamilyGenerative for GaussianLocationScaleFamily {
    fn generative_spec(
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

/// Built-in binomial logit family (single parameter block).
#[derive(Clone)]
pub struct BinomialLogitFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
}

impl BinomialLogitFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameter_names() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Logit]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_logit",
            parameter_names: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for BinomialLogitFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "BinomialLogitFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n {
            return Err("BinomialLogitFamily input size mismatch".to_string());
        }
        evaluate_single_block_glm(
            GlmLikelihoodFamily::BinomialLogit,
            &self.y,
            &self.weights,
            eta,
        )
    }
}

impl CustomFamilyGenerative for BinomialLogitFamily {
    fn generative_spec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "BinomialLogitFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let mean = block_states[Self::BLOCK_ETA].eta.mapv(|e| {
            (1.0 / (1.0 + (-e.clamp(-30.0, 30.0)).exp())).clamp(MIN_PROB, 1.0 - MIN_PROB)
        });
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

    pub fn parameter_names() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "poisson_log",
            parameter_names: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for PoissonLogFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "PoissonLogFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n {
            return Err("PoissonLogFamily input size mismatch".to_string());
        }

        let mut mu = Array1::<f64>::zeros(n);
        let mut ll = 0.0;
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);

        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi < 0.0 {
                return Err(format!(
                    "PoissonLogFamily requires non-negative finite y; found y[{i}]={yi}"
                ));
            }
            let (e, clamp_active) = hard_clamped_eta_row(eta[i], -30.0, 30.0);
            let m = e.exp().max(1e-12);
            mu[i] = m;
            // Drop log(y!) constant in objective.
            ll += self.weights[i] * (yi * e - m);
            let dmu = m.max(MIN_DERIV);
            let var = m.max(MIN_PROB);
            if self.weights[i] == 0.0 || clamp_active {
                w[i] = 0.0;
                z[i] = eta[i];
            } else {
                w[i] = floor_positive_weight(self.weights[i] * (dmu * dmu / var), MIN_WEIGHT);
                z[i] = e + (yi - m) / signed_with_floor(dmu, MIN_DERIV);
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            block_working_sets: vec![BlockWorkingSet::Diagonal {
                working_response: z,
                working_weights: w,
            }],
        })
    }
}

impl CustomFamilyGenerative for PoissonLogFamily {
    fn generative_spec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "PoissonLogFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let mean = block_states[Self::BLOCK_ETA]
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

    pub fn parameter_names() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gamma_log",
            parameter_names: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for GammaLogFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "GammaLogFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
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

        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi <= 0.0 {
                return Err(format!(
                    "GammaLogFamily requires positive finite y; found y[{i}]={yi}"
                ));
            }
            let (e, clamp_active) = hard_clamped_eta_row(eta[i], -30.0, 30.0);
            let m = e.exp().max(1e-12);
            mu[i] = m;
            // Gamma(shape=k, scale=mu/k), dropping constants independent of eta.
            ll += self.weights[i] * (-self.shape * (yi / m + m.ln()));
            let dmu = m.max(MIN_DERIV);
            let var = (m * m / self.shape).max(MIN_PROB);
            if self.weights[i] == 0.0 || clamp_active {
                w[i] = 0.0;
                z[i] = eta[i];
            } else {
                w[i] = floor_positive_weight(self.weights[i] * (dmu * dmu / var), MIN_WEIGHT);
                z[i] = e + (yi - m) / signed_with_floor(dmu, MIN_DERIV);
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            block_working_sets: vec![BlockWorkingSet::Diagonal {
                working_response: z,
                working_weights: w,
            }],
        })
    }
}

impl CustomFamilyGenerative for GammaLogFamily {
    fn generative_spec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 1 {
            return Err(format!(
                "GammaLogFamily expects 1 block, got {}",
                block_states.len()
            ));
        }
        let mean = block_states[Self::BLOCK_ETA]
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
}

impl BinomialLocationScaleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub fn parameter_names() -> &'static [&'static str] {
        &["threshold", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::InverseLink, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scale",
            parameter_names: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
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
        let (t_ws, ls_ws, _none) = binomial_location_scale_working_sets(
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
            block_working_sets: vec![t_ws, ls_ws],
        })
    }

    fn diagonal_working_weights_directional_derivative(
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
            let raw_w = self.weights[i] * g * chain * chain;
            if !raw_w.is_finite() || raw_w <= MIN_WEIGHT {
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

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Derivation ledger:
        //
        // 1) Model blocks:
        //      beta=[beta_t;beta_ls], eta_t=X_t beta_t, eta_ls=X_ls beta_ls,
        //      q=-eta_t/sigma(eta_ls), mu=Phi(q), l(beta)=sum_i l_i(mu_i).
        //
        // 2) Inner curvature in eta-space:
        //      H = -∇²_beta l(beta)
        //        = block(X_t, X_ls; w_tt,w_tl,w_ll),
        //      w_ab = -(c q_a q_b + s q_ab), a,b in {t,ls},
        //      s=l'(q), c=l''(q).
        //
        // 3) Non-wiggle q-partials used rowwise:
        //      q_t=-1/sigma, q_ls=eta_t sigma'/sigma², q_tt=0,
        //      q_tl=sigma'/sigma²,
        //      q_ll=eta_t( sigma''/sigma² - 2(sigma')²/sigma³ ).
        //
        // 4) Row contribution to H:
        //      H_i =
        //      [ w_tt x_ti x_ti'            w_tl x_ti x_lsi' ]
        //      [ w_tl x_lsi x_ti'           w_ll x_lsi x_lsi' ].
        //
        // This function evaluates exactly that formula with
        // eta_two_block_jet_from_q(...) -> (w_tt,w_tl,w_ll), then accumulates
        // the three block products and mirrors symmetry.
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
        let (x_t, x_ls) = self.dense_block_designs()?;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let (sigma, ds, d2s, _) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let mut w_tt_diag = Array1::<f64>::zeros(n);
        let mut w_tl_diag = Array1::<f64>::zeros(n);
        let mut w_ll_diag = Array1::<f64>::zeros(n);

        for i in 0..n {
            let (score_q, curvature_q, _third_q) = binomial_score_curvature_third_from_jet(
                self.y[i],
                self.weights[i],
                core.clamp_active[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
            );
            let q = nonwiggle_q_derivs(eta_t[i], sigma[i], ds[i], d2s[i], 0.0);

            let eta_jet = eta_two_block_jet_from_q(
                score_q,
                curvature_q,
                q.q_t,
                q.q_ls,
                q.q_tt,
                q.q_tl,
                q.q_ll,
            );
            w_tt_diag[i] = eta_jet.w_tt;
            w_tl_diag[i] = eta_jet.w_tl;
            w_ll_diag[i] = eta_jet.w_ll;
        }

        let h_tt = xt_diag_x_dense(&x_t, &w_tt_diag)?;
        let h_tl = xt_diag_y_dense(&x_t, &w_tl_diag, &x_ls)?;
        let h_ll = xt_diag_x_dense(&x_ls, &w_ll_diag)?;
        Ok(Some(assemble_two_block_symmetric(&h_tt, &h_tl, &h_ll)))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }

        // Directional derivative of the joint Hessian H(beta).
        //
        // We need dH[u] for outer REML/LAML gradients because:
        //   dJ/drho_k = S_k + (∂H/∂beta) * (d beta / d rho_k),
        // where J = H(beta) + P(rho) and u_k = d beta / d rho_k.
        //
        // ---------------------------------------------------------------------
        // Full two-block probit location-scale derivation map used here.
        //
        // Model blocks:
        //   beta = [beta_t; beta_ls], eta_t = X_t beta_t, eta_ls = X_ls beta_ls.
        //   sigma = sigma(eta_ls), q = -eta_t / sigma, mu = Phi(q).
        //
        // Rowwise negative log-likelihood Hessian wrt eta=(eta_t,eta_ls):
        //   H_i(eta) =
        //     [ w_tt(i)  w_tl(i) ]
        //     [ w_tl(i)  w_ll(i) ],
        // where each weight has generic shape
        //   w = -(c a + s b),
        // with s = d ell/dq, c = d² ell/dq², and (a,b) determined by which
        // eta-partial is being assembled.
        //
        // For non-wiggle q=-eta_t/sigma(eta_ls):
        //   w_tt = -(c q_t²       + s q_tt),   (q_tt = 0)
        //   w_tl = -(c q_t q_ls   + s q_tl),
        //   w_ll = -(c q_ls²      + s q_ll).
        //
        // Along direction u=[u_t;u_ls], with
        //   d eta_t = X_t u_t, d eta_ls = X_ls u_ls,
        //   delta q = q_t d eta_t + q_ls d eta_ls,
        // and chain derivatives ds = c delta q, dc = t delta q
        // (t = d³ ell/dq³), the directional derivative of any weight is
        //
        //   delta w
        //   = -delta(c a + s b)
        //   = -( dc a + c delta a + ds b + s delta b ).
        //
        // This is exactly what `delta_newton_weight_from_q_terms` evaluates:
        //   inputs (q_a,q_b,q_ab,delta_q,delta_q_a,delta_q_b,delta_q_ab)
        //   produce delta w_ab.
        //
        // Block assembly then follows:
        //   dH[u] =
        //     X_t'  diag(delta w_tt) X_t
        //   + X_t'  diag(delta w_tl) X_ls
        //   + X_ls' diag(delta w_tl) X_t
        //   + X_ls' diag(delta w_ll) X_ls.
        //
        // NOTE on exact outer Hessian (second derivatives wrt rho):
        //   The full exact outer Hessian additionally needs
        //     J_{k,l} = dH[u_{k,l}] + d²H[u_l,u_k],
        //   where d²H[u,v] requires fourth-order likelihood derivatives wrt q
        //   (ell^(4)) and fourth-order q-map terms (equivalently sigma^(4) or AD).
        // ---------------------------------------------------------------------
        //
        // For any direction u = [u_t; u_ls]:
        //   d eta_t  = X_t  u_t
        //   d eta_ls = X_ls u_ls
        //
        // With q = q(eta_t, eta_ls), define first-order variations:
        //   delta q      = q_t d eta_t + q_ls d eta_ls
        //   delta q_t    = q_tl d eta_ls                       (q_tt = 0)
        //   delta q_ls   = q_tl d eta_t + q_ll d eta_ls
        //   delta q_tl   = q_tl_ls d eta_ls
        //   delta q_ll   = q_tl_ls d eta_t + q_ll_ls d eta_ls
        //
        // Per-row negative-loglik Hessian weights are:
        //   w_ab = -(c q_a q_b + s q_ab),  a,b in {t,ls},
        // where:
        //   s = d ell / dq     (score_q),
        //   c = d2 ell / dq2   (= -curvature_q),
        //   t = d3 ell / dq3   (third_q).
        //
        // Directional derivative:
        //   delta w_ab = -delta(c q_a q_b + s q_ab)
        //              = -[ t delta_q q_a q_b
        //                   + c(delta q_a q_b + q_a delta q_b)
        //                   + c delta_q q_ab
        //                   + s delta q_ab ].
        //
        // Specialized forms used below:
        //   delta w_tt = -[ t delta_q q_t^2 + 2c(delta q_t)q_t ]
        //   delta w_tl = -[ t delta_q q_t q_ls
        //                   + c((delta q_t)q_ls + q_t(delta q_ls))
        //                   + c delta_q q_tl + s delta q_tl ]
        //   delta w_ll = -[ t delta_q q_ls^2
        //                   + 2c(delta q_ls)q_ls
        //                   + c delta_q q_ll + s delta q_ll ].
        //
        // Then assemble:
        //   dH[u] =
        //     [ X_t^T  diag(delta w_tt) X_t     X_t^T  diag(delta w_tl) X_ls ]
        //     [ X_ls^T diag(delta w_tl) X_t     X_ls^T diag(delta w_ll) X_ls ].
        //
        // Local variable map:
        //   q.*  -> q-derivative tuple from `nonwiggle_q_derivs`
        //   dq.* -> directional q-derivative tuple from `nonwiggle_q_directional`
        //   coeff_tt/tl/ll -> delta w_tt / delta w_tl / delta w_ll
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
        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_layout = GamlssBetaLayout::two_block(pt, pls);
        let (u_t, u_ls) = beta_layout.split_two(d_beta_flat, "binomial joint d_beta")?;
        let d_eta_t = x_t.dot(&u_t);
        let d_eta_ls = x_ls.dot(&u_ls);

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let mut w_tt_diag = Array1::<f64>::zeros(n);
        let mut w_tl_diag = Array1::<f64>::zeros(n);
        let mut w_ll_diag = Array1::<f64>::zeros(n);

        for i in 0..n {
            let (score_q, curvature_q, third_q) = binomial_score_curvature_third_from_jet(
                self.y[i],
                self.weights[i],
                core.clamp_active[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
            );
            let q = nonwiggle_q_derivs(eta_t[i], sigma[i], ds[i], d2s[i], d3s[i]);
            let dq = nonwiggle_q_directional(q, d_eta_t[i], d_eta_ls[i]);

            w_tt_diag[i] = delta_newton_weight_from_q_terms(
                score_q,
                curvature_q,
                third_q,
                q.q_t,
                q.q_t,
                q.q_tt,
                dq.delta_q,
                dq.delta_q_t,
                dq.delta_q_t,
                dq.delta_q_tt,
            );
            w_tl_diag[i] = delta_newton_weight_from_q_terms(
                score_q,
                curvature_q,
                third_q,
                q.q_t,
                q.q_ls,
                q.q_tl,
                dq.delta_q,
                dq.delta_q_t,
                dq.delta_q_ls,
                dq.delta_q_tl,
            );
            w_ll_diag[i] = delta_newton_weight_from_q_terms(
                score_q,
                curvature_q,
                third_q,
                q.q_ls,
                q.q_ls,
                q.q_ll,
                dq.delta_q,
                dq.delta_q_ls,
                dq.delta_q_ls,
                dq.delta_q_ll,
            );
        }

        let d_h_tt = xt_diag_x_dense(&x_t, &w_tt_diag)?;
        let d_h_tl = xt_diag_y_dense(&x_t, &w_tl_diag, &x_ls)?;
        let d_h_ll = xt_diag_x_dense(&x_ls, &w_ll_diag)?;
        Ok(Some(assemble_two_block_symmetric(
            &d_h_tt, &d_h_tl, &d_h_ll,
        )))
    }

    fn exact_newton_joint_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
            return Ok(None);
        }

        // Exact D²H[u,v] for the two-block location-scale probit model.
        //
        // Notation (per observation i):
        //   eta_t = (X_t beta_t)_i,   eta_ls = (X_ls beta_ls)_i,
        //   q = -eta_t / sigma(eta_ls),   mu = Phi(q),
        //   s = ∂l/∂q, c = ∂²l/∂q², t = ∂³l/∂q³, r = ∂⁴l/∂q⁴.
        //
        // Eta-space Hessian weights:
        //   w_ab = -(c q_a q_b + s q_ab), a,b in {t,ls}.
        //
        // For directions u,v define z_t(u)=(X_t u_t)_i, z_ls(u)=(X_ls u_ls)_i
        // and similarly for v. Then:
        //   Dq[u]      = q_t z_t(u) + q_ls z_ls(u),
        //   D²q[u,v]   = q_tl(z_t(u)z_ls(v)+z_ls(u)z_t(v)) + q_ll z_ls(u)z_ls(v),
        // with corresponding Dq_a[·], D²q_a[·,·], Dq_ab[·], D²q_ab[·,·].
        //
        // Generic exact second variation for w = -(c a + s b):
        //   D²w[u,v] = -(
        //      D²c[u,v] a
        //    + Dc[u] Da[v] + Dc[v] Da[u] + c D²a[u,v]
        //    + D²s[u,v] b
        //    + Ds[u] Db[v] + Ds[v] Db[u] + s D²b[u,v] ),
        // where
        //   Ds[u]=c Dq[u], Dc[u]=t Dq[u],
        //   D²s[u,v]=t Dq[u]Dq[v] + c D²q[u,v],
        //   D²c[u,v]=r Dq[u]Dq[v] + t D²q[u,v].
        //
        // Instantiations used below:
        //   (tt): a=q_t²,     b=q_tt(=0),
        //   (tl): a=q_t q_ls, b=q_tl,
        //   (ll): a=q_ls²,    b=q_ll.
        //
        // The map q=-eta_t/sigma(eta_ls) needs q-partials up to 4th ls-order:
        // q_tl_ls_ls_ls and q_ll_ls_ls (equivalently sigma'''').
        //
        // Final assembly:
        //   D²H[u,v]
        //   = X_t'  diag(D²w_tt[u,v]) X_t
        //   + X_t'  diag(D²w_tl[u,v]) X_ls
        //   + X_ls' diag(D²w_tl[u,v]) X_t
        //   + X_ls' diag(D²w_ll[u,v]) X_ls.
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

        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_layout = GamlssBetaLayout::two_block(pt, pls);
        let (u_t, u_ls) = beta_layout.split_two(d_beta_u_flat, "binomial joint d_beta_u")?;
        let (v_t, v_ls) = beta_layout.split_two(d_beta_v_flat, "binomial joint d_beta_v")?;
        let d_eta_t_u = x_t.dot(&u_t);
        let d_eta_ls_u = x_ls.dot(&u_ls);
        let d_eta_t_v = x_t.dot(&v_t);
        let d_eta_ls_v = x_ls.dot(&v_ls);

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let (sigma, ds, d2s, d3s, d4s) = gaussian_sigma_derivs_up_to_fourth(eta_ls.view());

        let mut w_tt_diag = Array1::<f64>::zeros(n);
        let mut w_tl_diag = Array1::<f64>::zeros(n);
        let mut w_ll_diag = Array1::<f64>::zeros(n);

        for i in 0..n {
            // Per-observation 4th-order ingredients for l(Phi(q)):
            //   s = l'(q), c = l''(q), t = l'''(q), r = l''''(q).
            let q0 = core.q0[i];
            let d4mu = (-q0 * q0 * q0 + 3.0 * q0) * core.dmu_dq[i];
            let (score_q, curvature_q, third_q) = binomial_score_curvature_third_from_jet(
                self.y[i],
                self.weights[i],
                core.clamp_active[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
            );
            let fourth_q = binomial_fourth_from_jet(
                self.y[i],
                self.weights[i],
                core.clamp_active[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                d4mu,
            );

            // q-map partials for q = -eta_t/sigma(eta_ls):
            //   q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls.
            let q = nonwiggle_q_derivs(eta_t[i], sigma[i], ds[i], d2s[i], d3s[i]);
            let u_t_i = d_eta_t_u[i];
            let u_ls_i = d_eta_ls_u[i];
            let v_t_i = d_eta_t_v[i];
            let v_ls_i = d_eta_ls_v[i];

            // First directional variations:
            //   dq_u = Dq[u], dq_v = Dq[v].
            let dq_u = q.q_t * u_t_i + q.q_ls * u_ls_i;
            let dq_v = q.q_t * v_t_i + q.q_ls * v_ls_i;
            // Second directional variation:
            //   d2q_uv = D²q[u,v]
            //          = q_tl (u_t v_ls + v_t u_ls) + q_ll u_ls v_ls.
            let d2q_uv = q.q_tl * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q.q_ll * u_ls_i * v_ls_i;

            // Directional variations of q-partials:
            //   Dq_t[u], Dq_t[v], Dq_ls[u], Dq_ls[v], Dq_tl[u], Dq_tl[v], Dq_ll[u], Dq_ll[v].
            let dq_t_u = q.q_tl * u_ls_i;
            let dq_t_v = q.q_tl * v_ls_i;
            let dq_ls_u = q.q_tl * u_t_i + q.q_ll * u_ls_i;
            let dq_ls_v = q.q_tl * v_t_i + q.q_ll * v_ls_i;
            let dq_tl_u = q.q_tl_ls * u_ls_i;
            let dq_tl_v = q.q_tl_ls * v_ls_i;
            let dq_ll_u = q.q_tl_ls * u_t_i + q.q_ll_ls * u_ls_i;
            let dq_ll_v = q.q_tl_ls * v_t_i + q.q_ll_ls * v_ls_i;

            let s_safe = sigma[i].max(1e-12);
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let s5 = s4 * s_safe;
            let q_tl_ls_ls =
                d3s[i] / s2 - 6.0 * ds[i] * d2s[i] / s3 + 6.0 * ds[i] * ds[i] * ds[i] / s4;
            // q_tl_ls_ls_ls is ∂^4 q /(∂eta_t ∂eta_ls^3) = -a''''
            // when q=-eta_t a(eta_ls), written directly via sigma derivatives.
            let q_tl_ls_ls_ls =
                d4s[i] / s2 - 8.0 * ds[i] * d3s[i] / s3 - 6.0 * d2s[i] * d2s[i] / s3
                    + 36.0 * ds[i] * ds[i] * d2s[i] / s4
                    - 24.0 * ds[i] * ds[i] * ds[i] * ds[i] / s5;
            // q_ll_ls_ls = ∂^4 q / ∂eta_ls^4.
            let q_ll_ls_ls = eta_t[i] * q_tl_ls_ls_ls;

            // Second directional variations of q-partials:
            //   D²q_t[u,v], D²q_ls[u,v], D²q_tl[u,v], D²q_ll[u,v].
            let d2q_t_uv = q.q_tl_ls * u_ls_i * v_ls_i;
            let d2q_ls_uv =
                q.q_tl_ls * (u_ls_i * v_t_i + v_ls_i * u_t_i) + q.q_ll_ls * u_ls_i * v_ls_i;
            let d2q_tl_uv = q_tl_ls_ls * u_ls_i * v_ls_i;
            let d2q_ll_uv =
                q_tl_ls_ls * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q_ll_ls_ls * u_ls_i * v_ls_i;

            // Build a=q_a q_b and its directional variations for (a,b)=(t,t):
            //   a_tt    = q_t^2,
            //   a_tt_u  = D(a_tt)[u] = 2 q_t Dq_t[u],
            //   a_tt_uv = D²(a_tt)[u,v] = 2(Dq_t[u]Dq_t[v] + q_t D²q_t[u,v]).
            let a_tt = q.q_t * q.q_t;
            let a_tt_u = 2.0 * q.q_t * dq_t_u;
            let a_tt_v = 2.0 * q.q_t * dq_t_v;
            let a_tt_uv = 2.0 * (dq_t_u * dq_t_v + q.q_t * d2q_t_uv);
            // For (tt), b=q_tt=0 so Db and D²b vanish.
            w_tt_diag[i] = second_delta_newton_weight_from_q_terms(
                score_q,
                curvature_q,
                third_q,
                fourth_q,
                q.q_tt,
                dq_u,
                dq_v,
                d2q_uv,
                a_tt,
                a_tt_u,
                a_tt_v,
                a_tt_uv,
                0.0,
                0.0,
                0.0,
            );

            // (t,ls) block:
            //   a_tl = q_t q_ls,
            //   a_tl_uv = D²(q_t q_ls)[u,v]
            //           = D²q_t[u,v] q_ls + Dq_t[u]Dq_ls[v] + Dq_t[v]Dq_ls[u] + q_t D²q_ls[u,v].
            //   b_tl = q_tl, with Db_tl and D²b_tl supplied below.
            let a_tl = q.q_t * q.q_ls;
            let a_tl_u = dq_t_u * q.q_ls + q.q_t * dq_ls_u;
            let a_tl_v = dq_t_v * q.q_ls + q.q_t * dq_ls_v;
            let a_tl_uv =
                d2q_t_uv * q.q_ls + dq_t_u * dq_ls_v + dq_t_v * dq_ls_u + q.q_t * d2q_ls_uv;
            w_tl_diag[i] = second_delta_newton_weight_from_q_terms(
                score_q,
                curvature_q,
                third_q,
                fourth_q,
                q.q_tl,
                dq_u,
                dq_v,
                d2q_uv,
                a_tl,
                a_tl_u,
                a_tl_v,
                a_tl_uv,
                dq_tl_u,
                dq_tl_v,
                d2q_tl_uv,
            );

            // (ls,ls) block:
            //   a_ll = q_ls^2,
            //   a_ll_uv = 2(Dq_ls[u]Dq_ls[v] + q_ls D²q_ls[u,v]).
            //   b_ll = q_ll, with Db_ll and D²b_ll provided below.
            let a_ll = q.q_ls * q.q_ls;
            let a_ll_u = 2.0 * q.q_ls * dq_ls_u;
            let a_ll_v = 2.0 * q.q_ls * dq_ls_v;
            let a_ll_uv = 2.0 * (dq_ls_u * dq_ls_v + q.q_ls * d2q_ls_uv);
            w_ll_diag[i] = second_delta_newton_weight_from_q_terms(
                score_q,
                curvature_q,
                third_q,
                fourth_q,
                q.q_ll,
                dq_u,
                dq_v,
                d2q_uv,
                a_ll,
                a_ll_u,
                a_ll_v,
                a_ll_uv,
                dq_ll_u,
                dq_ll_v,
                d2q_ll_uv,
            );
        }

        let d2_h_tt = xt_diag_x_dense(&x_t, &w_tt_diag)?;
        let d2_h_tl = xt_diag_y_dense(&x_t, &w_tl_diag, &x_ls)?;
        let d2_h_ll = xt_diag_x_dense(&x_ls, &w_ll_diag)?;
        Ok(Some(assemble_two_block_symmetric(
            &d2_h_tt, &d2_h_tl, &d2_h_ll,
        )))
    }
}

impl CustomFamilyGenerative for BinomialLocationScaleFamily {
    fn generative_spec(
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
            mean[i] = jet.mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
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

    pub fn parameter_names() -> &'static [&'static str] {
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
            name: "binomial_location_scale_wiggle",
            parameter_names: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }

    pub fn initialize_wiggle_knots_from_q(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
    ) -> Result<Array1<f64>, String> {
        initialize_wiggle_knots_from_seed(q_seed, degree, num_internal_knots)
    }

    fn wiggle_constraint_transform(&self) -> Result<Array2<f64>, String> {
        let (z, _s_constrained) =
            compute_geometric_constraint_transform(&self.wiggle_knots, self.wiggle_degree, 2)
                .map_err(|e| e.to_string())?;
        Ok(z)
    }

    fn constrain_wiggle_basis(&self, full: Array2<f64>) -> Result<Array2<f64>, String> {
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

    fn wiggle_basis_with_options(
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
        self.constrain_wiggle_basis((*basis).clone())
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basis_with_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basis_with_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != beta_wiggle.len() {
            return Err(format!(
                "wiggle derivative col mismatch: got {}, expected {}",
                d_constrained.ncols(),
                beta_wiggle.len()
            ));
        }
        Ok(d_constrained.dot(&beta_wiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2_constrained =
            self.wiggle_basis_with_options(q0, BasisOptions::second_derivative())?;
        if d2_constrained.ncols() != beta_wiggle.len() {
            return Err(format!(
                "wiggle second-derivative col mismatch: got {}, expected {}",
                d2_constrained.ncols(),
                beta_wiggle.len()
            ));
        }
        Ok(d2_constrained.dot(&beta_wiggle))
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3_constrained = self.wiggle_d3basis_constrained(q0)?;
        if d3_constrained.ncols() != beta_wiggle.len() {
            return Err(format!(
                "wiggle third-derivative col mismatch: got {}, expected {}",
                d3_constrained.ncols(),
                beta_wiggle.len()
            ));
        }
        Ok(d3_constrained.dot(&beta_wiggle))
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
            evaluate_bspline_third_derivative_scalar(
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
        beta_wiggle: ArrayView1<'_, f64>,
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
        if z.ncols() != beta_wiggle.len() {
            return Err(format!(
                "wiggle fourth-derivative col mismatch: got {}, expected {}",
                z.ncols(),
                beta_wiggle.len()
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
            for constrained_j in 0..beta_wiggle.len() {
                let mut basis_j = 0.0;
                for raw_k in 0..num_basis {
                    basis_j += raw[raw_k] * z[[raw_k, constrained_j]];
                }
                acc += basis_j * beta_wiggle[constrained_j];
            }
            out[i] = acc;
        }
        Ok(out)
    }

    #[allow(dead_code)]
    fn dense_block_designs(&self) -> Result<(Array2<f64>, Array2<f64>), String> {
        let xt = self
            .threshold_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleWiggleFamily exact path is missing threshold design"
                    .to_string()
            })?
            .to_dense();
        let xls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleWiggleFamily exact path is missing log-sigma design"
                    .to_string()
            })?
            .to_dense();
        Ok((xt, xls))
    }

    /// Build a turnkey wiggle block from a q-seed vector and knot settings.
    /// Returns both the block input and the generated knot vector.
    pub fn build_wiggle_block_input(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
        penalty_order: usize,
        double_penalty: bool,
    ) -> Result<(ParameterBlockInput, Array1<f64>), String> {
        let knots = Self::initialize_wiggle_knots_from_q(q_seed, degree, num_internal_knots)?;
        let block = build_wiggle_block_input_from_knots(
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
        let eta_w = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || eta_w.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(eta_w),
            &self.link_kind,
        )?;
        let wiggle_design = self.wiggle_design(core.q0.view())?;
        let dq_dq0 =
            self.wiggle_dq_dq0(core.q0.view(), block_states[Self::BLOCK_WIGGLE].beta.view())?;
        let d2q_dq02 =
            self.wiggle_d2q_dq02(core.q0.view(), block_states[Self::BLOCK_WIGGLE].beta.view())?;
        let (_, _, d2sigma_deta2, _) = exp_sigma_derivs_up_to_third(eta_ls.view());
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing threshold design"
                .to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing log-sigma design"
                .to_string()
        })?;
        let (t_ws, ls_ws, w_ws) = binomial_location_scale_working_sets(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(eta_w),
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
        let w_ws = w_ws.ok_or_else(|| "wiggle working set missing".to_string())?;

        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            block_working_sets: vec![t_ws, ls_ws, w_ws],
        })
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
        // For the 3-block wiggle model with beta=(beta_t,beta_ls,beta_w),
        // define the full negative-loglik Hessian H(beta) in flattened block
        // coordinates. For a direction that moves only one block,
        //
        //   u = [u_t, 0,   0]   or
        //   u = [0,   u_ls,0]   or
        //   u = [0,   0,   u_w],
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
        // ---------------------------------------------------------------------
        // Exact joint Hessian for the 3-block binomial location-scale wiggle family.
        //
        // Model:
        //   q0 = -eta_t / sigma(eta_ls),
        //   q  = q0 + beta_w^T B(q0),
        //   mu = Phi(q),
        //   F  = -sum_i ell_i(mu_i).
        //
        // Canonical per-row identities implemented below:
        //
        // 1) q-derivative building blocks
        //   m  = dq/dq0 = 1 + beta_w^T B'(q0)
        //   g2 = d²q/dq0² via wiggle = beta_w^T B''(q0)
        //
        //   q_t  = m q0_t
        //   q_l  = m q0_l
        //   q_tt = g2 q0_t q0_t
        //   q_tl = g2 q0_t q0_l + m q0_tl
        //   q_ll = g2 q0_l q0_l + m q0_ll
        //   q_w  = B
        //   q_tw = q0_t B'
        //   q_lw = q0_l B'
        //   q_ww = 0
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
        //   tw/lw from (q_w,q_tw,q_lw),
        //   ww from q_ww=0 => H_ww = sum m2 * q_w q_w^T.
        // ---------------------------------------------------------------------
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let eta_w = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || eta_w.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_w0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(eta_w),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basis_with_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basis_with_options(core0.q0.view(), BasisOptions::second_derivative())?;
        if b0.ncols() != beta_w0.len()
            || d0.ncols() != beta_w0.len()
            || dd0.ncols() != beta_w0.len()
        {
            return Err(format!(
                "wiggle basis/beta mismatch in exact joint Hessian: B={} B'={} B''={} beta_w={}",
                b0.ncols(),
                d0.ncols(),
                dd0.ncols(),
                beta_w0.len()
            ));
        }
        let m = d0.dot(&beta_w0) + 1.0;
        let g2 = dd0.dot(&beta_w0);
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
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core0.q0[i] + eta_w[i];
            let (m1, m2, _m3) =
                if matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
                    binomial_neglog_q_derivatives_probit_closed_form(
                        self.y[i],
                        self.weights[i],
                        q_i,
                        core0.clamp_active[i],
                        core0.mu[i],
                    )
                } else {
                    binomial_neglog_q_derivatives_from_jet(
                        self.y[i],
                        self.weights[i],
                        core0.clamp_active[i],
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

            coeff_tt[i] = hessian_coeff_from_objective_q_terms(m1, m2, q_t, q_t, q_tt);
            coeff_tl[i] = hessian_coeff_from_objective_q_terms(m1, m2, q_t, q_ls, q_tl);
            coeff_ll[i] = hessian_coeff_from_objective_q_terms(m1, m2, q_ls, q_ls, q_ll);
            coeff_tw_b[i] = m2 * q_t;
            coeff_tw_d[i] = m1 * q0.q_t;
            coeff_lw_b[i] = m2 * q_ls;
            coeff_lw_d[i] = m1 * q0.q_ls;
            coeff_ww[i] = m2;
        }
        let h_tt = xt_diag_x_dense(&x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(&x_t, &coeff_tl, &x_ls)?;
        let h_ll = xt_diag_x_dense(&x_ls, &coeff_ll)?;
        let h_tw =
            xt_diag_y_dense(&x_t, &coeff_tw_b, &b0)? + &xt_diag_y_dense(&x_t, &coeff_tw_d, &d0)?;
        let h_lw =
            xt_diag_y_dense(&x_ls, &coeff_lw_b, &b0)? + &xt_diag_y_dense(&x_ls, &coeff_lw_d, &d0)?;
        let h_ww = xt_diag_x_dense(&b0, &coeff_ww)?;

        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..pt + pls]).assign(&h_tl);
        h.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&h_ll);
        h.slice_mut(s![0..pt, pt + pls..total]).assign(&h_tw);
        h.slice_mut(s![pt..pt + pls, pt + pls..total]).assign(&h_lw);
        h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&h_ww);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // ---------------------------------------------------------------------
        // Exact directional derivative dH[u] for the same 3-block model.
        //
        // Direction:
        //   u = (u_t, u_l, u_w),
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
        //   m  = 1 + beta_w^T B'(q0)
        //   g2 = beta_w^T B''(q0)
        //   g3 = beta_w^T B'''(q0)
        //
        // 2) Directional wiggle scalars:
        //   dm  = (B'·u_w)  + g2*dq0
        //   dg2 = (B''·u_w) + g3*dq0
        //
        // 3) Directional q pieces:
        //   dq   = m*dq0 + B·u_w
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
        //   q_w   = B,         dq_w   = B' dq0
        //   q_tw  = q0_t B',   dq_tw  = dq0_t B' + dq0 q0_t B''
        //   q_lw  = q0_l B',   dq_lw  = dq0_l B' + dq0 q0_l B''
        //   q_ww  = 0,         dq_ww  = 0
        //
        // Implementation below follows these formulas exactly block-by-block.
        // ---------------------------------------------------------------------
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let eta_w = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || eta_w.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_w0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(eta_w),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::with_wiggle(pt, pls, pw);
        let total = beta_layout.total();
        let (u_t, u_ls, u_w) = beta_layout.split_three(d_beta_flat, "wiggle joint d_beta")?;
        let d_eta_t = x_t.dot(&u_t);
        let d_eta_ls = x_ls.dot(&u_ls);

        let d0 =
            self.wiggle_basis_with_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basis_with_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), beta_w0.view())?;
        if d0.ncols() != beta_w0.len() || dd0.ncols() != beta_w0.len() {
            return Err(format!(
                "wiggle derivative/beta mismatch in exact joint dH: B'={} B''={} beta_w={}",
                d0.ncols(),
                dd0.ncols(),
                beta_w0.len()
            ));
        }
        let m = d0.dot(&beta_w0) + 1.0;
        let g2 = dd0.dot(&beta_w0);
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
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core0.q0[i] + eta_w[i];
            let (m1, m2, m3) =
                if matches!(self.link_kind, InverseLink::Standard(LinkFunction::Probit)) {
                    binomial_neglog_q_derivatives_probit_closed_form(
                        self.y[i],
                        self.weights[i],
                        q_i,
                        core0.clamp_active[i],
                        core0.mu[i],
                    )
                } else {
                    binomial_neglog_q_derivatives_from_jet(
                        self.y[i],
                        self.weights[i],
                        core0.clamp_active[i],
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
            let duw_i = dr.dot(&u_w);
            let dduw_i = ddr.dot(&u_w);

            // Canonical directional wiggle scalars:
            //   dm  = B'(q0)·u_w + g2*dq0
            //   dg2 = B''(q0)·u_w + g3*dq0
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

            let delta_q = m[i] * dq0.delta_q + br.dot(&u_w);

            // Closed forms by block from:
            // dH_ab = m3*dq*q_a*q_b + m2*(dq_a*q_b + q_a*dq_b + dq*q_ab) + m1*dq_ab.
            //
            // (tt):
            //   dH_tt = m3*dq*q_t^2 + m2*(2*dq_t*q_t + dq*q_tt) + m1*dq_tt.
            let coeff_tt_i = directional_hessian_coeff_from_objective_q_terms(
                m1, m2, m3, delta_q, q_t, q_t, q_tt, delta_q_t, delta_q_t, delta_q_tt,
            );
            // (tl):
            //   dH_tl = m3*dq*q_t*q_l
            //        + m2*(dq_t*q_l + q_t*dq_l + dq*q_tl)
            //        + m1*dq_tl.
            let coeff_tl_i = directional_hessian_coeff_from_objective_q_terms(
                m1, m2, m3, delta_q, q_t, q_ls, q_tl, delta_q_t, delta_q_ls, delta_q_tl,
            );
            // (ll):
            //   dH_ll = m3*dq*q_l^2 + m2*(2*dq_l*q_l + dq*q_ll) + m1*dq_ll.
            let coeff_ll_i = directional_hessian_coeff_from_objective_q_terms(
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
            coeff_ww_bb[i] = m3 * delta_q;
            coeff_ww_db[i] = m2 * dq0.delta_q;
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
        let mut d_h_ww = xt_diag_x_dense(&b0, &coeff_ww_bb)?;
        d_h_ww += &xt_diag_y_dense(&d0, &coeff_ww_db, &b0)?;
        d_h_ww += &xt_diag_y_dense(&b0, &coeff_ww_db, &d0)?;

        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..pt + pls]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&d_h_ll);
        d_h.slice_mut(s![0..pt, pt + pls..total]).assign(&d_h_tw);
        d_h.slice_mut(s![pt..pt + pls, pt + pls..total])
            .assign(&d_h_lw);
        d_h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&d_h_ww);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    fn exact_newton_joint_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
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
        let eta_w = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || eta_w.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleWiggleFamily input size mismatch".to_string());
        }

        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let beta_w0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(eta_w),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basis_with_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basis_with_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(core0.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), beta_w0.view())?;
        let d4q = self.wiggle_d4q_dq04(core0.q0.view(), beta_w0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::with_wiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d0.ncols() != beta_w0.len()
            || dd0.ncols() != beta_w0.len()
            || d3_basis.ncols() != beta_w0.len()
        {
            return Err(format!(
                "wiggle derivative/beta mismatch in exact joint d2H: B'={} B''={} B'''={} beta_w={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                beta_w0.len()
            ));
        }

        let (u_t, u_ls, u_w) = beta_layout.split_three(d_beta_u_flat, "wiggle joint d_beta_u")?;
        let (v_t, v_ls, v_w) = beta_layout.split_three(d_beta_v_flat, "wiggle joint d_beta_v")?;
        let d_eta_t_u = x_t.dot(&u_t);
        let d_eta_ls_u = x_ls.dot(&u_ls);
        let d_eta_t_v = x_t.dot(&v_t);
        let d_eta_ls_v = x_ls.dot(&v_ls);

        let m = d0.dot(&beta_w0) + 1.0;
        let g2 = dd0.dot(&beta_w0);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = gaussian_sigma_derivs_up_to_fourth(eta_ls.view());

        let mut d2_h = Array2::<f64>::zeros((total, total));
        for i in 0..n {
            // Per-row scalar objective derivatives for F_i(q).
            let q_i = core0.q0[i] + eta_w[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_probit_closed_form(
                self.y[i],
                self.weights[i],
                q_i,
                core0.clamp_active[i],
                core0.mu[i],
            );
            let m4 = binomial_neglog_q_fourth_derivative_probit_closed_form(
                self.y[i],
                self.weights[i],
                q_i,
                core0.clamp_active[i],
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
            let v_t_i = d_eta_t_v[i];
            let v_ls_i = d_eta_ls_v[i];

            // Directional z=q0 primitives for u and v.
            let dq0_u = q0.q_t * u_t_i + q0.q_ls * u_ls_i;
            let dq0_v = q0.q_t * v_t_i + q0.q_ls * v_ls_i;
            let d2q0_uv = q0.q_tl * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0.q_ll * u_ls_i * v_ls_i;

            let dq0_t_u = q0.q_tl * u_ls_i;
            let dq0_t_v = q0.q_tl * v_ls_i;
            let dq0_ls_u = q0.q_tl * u_t_i + q0.q_ll * u_ls_i;
            let dq0_ls_v = q0.q_tl * v_t_i + q0.q_ll * v_ls_i;
            let dq0_tl_u = q0.q_tl_ls * u_ls_i;
            let dq0_tl_v = q0.q_tl_ls * v_ls_i;
            let dq0_ll_u = q0.q_tl_ls * u_t_i + q0.q_ll_ls * u_ls_i;
            let dq0_ll_v = q0.q_tl_ls * v_t_i + q0.q_ll_ls * v_ls_i;

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
            let b_u = br.dot(&u_w);
            let b_v = br.dot(&v_w);
            let b1_u = dr.dot(&u_w);
            let b1_v = dr.dot(&v_w);
            let b2_u = ddr.dot(&u_w);
            let b2_v = ddr.dot(&v_w);
            let b3_u = d3r.dot(&u_w);
            let b3_v = d3r.dot(&v_w);

            // Wiggle scalar chain terms:
            //   m = 1 + g1,     g2 = beta_w^T B''(q0),
            //   dm[u]   = B'·u_w + g2*dq0[u],
            //   d2m[u,v]= g3*dq0[u]dq0[v] + g2*d2q0[u,v] + (B''·v_w)dq0[u] + (B''·u_w)dq0[v],
            //   dg2[u]  = B''·u_w + g3*dq0[u],
            //   d2g2[u,v]=g4*dq0[u]dq0[v] + g3*d2q0[u,v] + (B'''·v_w)dq0[u] + (B'''·u_w)dq0[v].
            let dm_u = b1_u + g2[i] * dq0_u;
            let dm_v = b1_v + g2[i] * dq0_v;
            let d2m_uv = g3[i] * dq0_u * dq0_v + g2[i] * d2q0_uv + b2_v * dq0_u + b2_u * dq0_v;
            let dg2_u = b2_u + g3[i] * dq0_u;
            let dg2_v = b2_v + g3[i] * dq0_v;
            let d2g2_uv = g4[i] * dq0_u * dq0_v + g3[i] * d2q0_uv + b3_v * dq0_u + b3_u * dq0_v;

            // First/second directional terms for total q.
            let dq_u = m[i] * dq0_u + b_u;
            let dq_v = m[i] * dq0_v + b_v;
            // Simplify exact formula for q = q0 + beta_w^T B(q0):
            //   D²q[u,v] = m*d²q0 + g2*dq0[u]dq0[v] + (B'·u_w)dq0[v] + (B'·v_w)dq0[u].
            let d2q_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0_v + b1_u * dq0_v + b1_v * dq0_u;

            // q partials by block and their first/second directional derivatives.
            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let q_tt = g2[i] * q0.q_t * q0.q_t;
            let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
            let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

            let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
            let dq_t_v = dm_v * q0.q_t + m[i] * dq0_t_v;
            let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
            let dq_ls_v = dm_v * q0.q_ls + m[i] * dq0_ls_v;

            let d2q_t_uv = d2m_uv * q0.q_t + dm_u * dq0_t_v + dm_v * dq0_t_u + m[i] * d2q0_t_uv;
            let d2q_ls_uv =
                d2m_uv * q0.q_ls + dm_u * dq0_ls_v + dm_v * dq0_ls_u + m[i] * d2q0_ls_uv;

            let dq_tt_u = dg2_u * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_t_u);
            let dq_tt_v = dg2_v * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_t_v);
            let d2q_tt_uv = d2g2_uv * q0.q_t * q0.q_t
                + dg2_u * (2.0 * q0.q_t * dq0_t_v)
                + dg2_v * (2.0 * q0.q_t * dq0_t_u)
                + g2[i] * (2.0 * dq0_t_u * dq0_t_v + 2.0 * q0.q_t * d2q0_t_uv);

            let dq_tl_u = dg2_u * q0.q_t * q0.q_ls
                + g2[i] * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                + dm_u * q0.q_tl
                + m[i] * dq0_tl_u;
            let dq_tl_v = dg2_v * q0.q_t * q0.q_ls
                + g2[i] * (dq0_t_v * q0.q_ls + q0.q_t * dq0_ls_v)
                + dm_v * q0.q_tl
                + m[i] * dq0_tl_v;
            let d2q_tl_uv = d2g2_uv * q0.q_t * q0.q_ls
                + dg2_u * (dq0_t_v * q0.q_ls + q0.q_t * dq0_ls_v)
                + dg2_v * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                + g2[i]
                    * (d2q0_t_uv * q0.q_ls
                        + dq0_t_u * dq0_ls_v
                        + dq0_t_v * dq0_ls_u
                        + q0.q_t * d2q0_ls_uv)
                + d2m_uv * q0.q_tl
                + dm_u * dq0_tl_v
                + dm_v * dq0_tl_u
                + m[i] * d2q0_tl_uv;

            let dq_ll_u = dg2_u * q0.q_ls * q0.q_ls
                + g2[i] * (2.0 * q0.q_ls * dq0_ls_u)
                + dm_u * q0.q_ll
                + m[i] * dq0_ll_u;
            let dq_ll_v = dg2_v * q0.q_ls * q0.q_ls
                + g2[i] * (2.0 * q0.q_ls * dq0_ls_v)
                + dm_v * q0.q_ll
                + m[i] * dq0_ll_v;
            let d2q_ll_uv = d2g2_uv * q0.q_ls * q0.q_ls
                + dg2_u * (2.0 * q0.q_ls * dq0_ls_v)
                + dg2_v * (2.0 * q0.q_ls * dq0_ls_u)
                + g2[i] * (2.0 * dq0_ls_u * dq0_ls_v + 2.0 * q0.q_ls * d2q0_ls_uv)
                + d2m_uv * q0.q_ll
                + dm_u * dq0_ll_v
                + dm_v * dq0_ll_u
                + m[i] * d2q0_ll_uv;

            // Exact second directional coefficients for the scalar block weights.
            let coeff_tt = second_directional_hessian_coeff_from_objective_q_terms(
                m1, m2, m3, m4, dq_u, dq_v, d2q_uv, q_t, q_t, q_tt, dq_t_u, dq_t_v, dq_t_u, dq_t_v,
                d2q_t_uv, d2q_t_uv, dq_tt_u, dq_tt_v, d2q_tt_uv,
            );
            let coeff_tl = second_directional_hessian_coeff_from_objective_q_terms(
                m1, m2, m3, m4, dq_u, dq_v, d2q_uv, q_t, q_ls, q_tl, dq_t_u, dq_t_v, dq_ls_u,
                dq_ls_v, d2q_t_uv, d2q_ls_uv, dq_tl_u, dq_tl_v, d2q_tl_uv,
            );
            let coeff_ll = second_directional_hessian_coeff_from_objective_q_terms(
                m1, m2, m3, m4, dq_u, dq_v, d2q_uv, q_ls, q_ls, q_ll, dq_ls_u, dq_ls_v, dq_ls_u,
                dq_ls_v, d2q_ls_uv, d2q_ls_uv, dq_ll_u, dq_ll_v, d2q_ll_uv,
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
                let q_w = br[j];
                let dq_w_u = dr[j] * dq0_u;
                let dq_w_v = dr[j] * dq0_v;
                let d2q_w_uv = ddr[j] * dq0_u * dq0_v + dr[j] * d2q0_uv;
                let q_tw = dr[j] * q0.q_t;
                let q_lw = dr[j] * q0.q_ls;
                let dq_tw_u = ddr[j] * dq0_u * q0.q_t + dr[j] * dq0_t_u;
                let dq_tw_v = ddr[j] * dq0_v * q0.q_t + dr[j] * dq0_t_v;
                let d2q_tw_uv = d3r[j] * dq0_u * dq0_v * q0.q_t
                    + ddr[j] * (d2q0_uv * q0.q_t + dq0_u * dq0_t_v + dq0_v * dq0_t_u)
                    + dr[j] * d2q0_t_uv;
                let dq_lw_u = ddr[j] * dq0_u * q0.q_ls + dr[j] * dq0_ls_u;
                let dq_lw_v = ddr[j] * dq0_v * q0.q_ls + dr[j] * dq0_ls_v;
                let d2q_lw_uv = d3r[j] * dq0_u * dq0_v * q0.q_ls
                    + ddr[j] * (d2q0_uv * q0.q_ls + dq0_u * dq0_ls_v + dq0_v * dq0_ls_u)
                    + dr[j] * d2q0_ls_uv;

                let coeff_tw = second_directional_hessian_coeff_from_objective_q_terms(
                    m1, m2, m3, m4, dq_u, dq_v, d2q_uv, q_t, q_w, q_tw, dq_t_u, dq_t_v, dq_w_u,
                    dq_w_v, d2q_t_uv, d2q_w_uv, dq_tw_u, dq_tw_v, d2q_tw_uv,
                );
                let coeff_lw = second_directional_hessian_coeff_from_objective_q_terms(
                    m1, m2, m3, m4, dq_u, dq_v, d2q_uv, q_ls, q_w, q_lw, dq_ls_u, dq_ls_v, dq_w_u,
                    dq_w_v, d2q_ls_uv, d2q_w_uv, dq_lw_u, dq_lw_v, d2q_lw_uv,
                );

                for a_idx in 0..pt {
                    d2_h[[a_idx, pt + pls + j]] += coeff_tw * xtr[a_idx];
                }
                for a_idx in 0..pls {
                    d2_h[[pt + a_idx, pt + pls + j]] += coeff_lw * xlsr[a_idx];
                }
            }

            for j in 0..pw {
                let q_wj = br[j];
                let dq_wj_u = dr[j] * dq0_u;
                let dq_wj_v = dr[j] * dq0_v;
                let d2q_wj_uv = ddr[j] * dq0_u * dq0_v + dr[j] * d2q0_uv;
                for k in j..pw {
                    let q_wk = br[k];
                    let dq_wk_u = dr[k] * dq0_u;
                    let dq_wk_v = dr[k] * dq0_v;
                    let d2q_wk_uv = ddr[k] * dq0_u * dq0_v + dr[k] * d2q0_uv;
                    let coeff_ww = second_directional_hessian_coeff_from_objective_q_terms(
                        m1, m2, m3, m4, dq_u, dq_v, d2q_uv, q_wj, q_wk, 0.0, dq_wj_u, dq_wj_v,
                        dq_wk_u, dq_wk_v, d2q_wj_uv, d2q_wk_uv, 0.0, 0.0, 0.0,
                    );
                    d2_h[[pt + pls + j, pt + pls + k]] += coeff_ww;
                }
            }
        }

        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
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
    fn generative_spec(
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
        let eta_w = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != self.y.len()
            || eta_ls.len() != self.y.len()
            || eta_w.len() != self.y.len()
        {
            return Err("BinomialLocationScaleWiggleFamily generative size mismatch".to_string());
        }
        let mut mean = Array1::<f64>::zeros(self.y.len());
        for i in 0..mean.len() {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]).max(1e-12);
            let q0 = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q0 + eta_w[i])
                .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
            mean[i] = jet.mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
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

    fn intercept_block(n: usize) -> ParameterBlockInput {
        ParameterBlockInput {
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }
    }

    #[test]
    fn weighted_projection_returns_finite_coefficients() {
        let n = 8usize;
        let design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let offset = Array1::zeros(n);
        let target_eta = Array1::from_vec(vec![0.2; n]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let beta =
            solve_weighted_projection(&design, &offset, &target_eta, &weights, 1e-10).unwrap();
        assert_eq!(beta.len(), 1);
        assert!(beta[0].is_finite());
        assert!((beta[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn alpha_beta_warm_start_produces_finite_targets() {
        let n = 16usize;
        let y = Array1::from_vec((0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect());
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold = intercept_block(n);
        let log_sigma = intercept_block(n);

        let (beta_t, beta_ls, beta_obs) = try_binomial_alpha_beta_warm_start(
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
        assert!(beta_obs.iter().all(|v| v.is_finite() && *v > 0.0));
        let expected_beta = 1.0;
        assert!(beta_obs.iter().all(|v| (*v - expected_beta).abs() < 1e-12));
    }

    #[test]
    fn zero_weight_rows_stay_inactive_in_builtin_diagonal_families() {
        let weights = Array1::from_vec(vec![0.0, 1.0]);

        let gaussian = GaussianLocationScaleFamily {
            y: Array1::from_vec(vec![2.0, -1.0]),
            weights: weights.clone(),
            mu_design: None,
            log_sigma_design: None,
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
        match &gaussian_eval.block_working_sets[GaussianLocationScaleFamily::BLOCK_MU] {
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
        match &gaussian_eval.block_working_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
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
        match &poisson_eval.block_working_sets[PoissonLogFamily::BLOCK_ETA] {
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
        match &gamma_eval.block_working_sets[GammaLogFamily::BLOCK_ETA] {
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
    fn hard_clamped_poisson_and_gamma_rows_stay_locally_flat() {
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
        match &poisson_eval.block_working_sets[PoissonLogFamily::BLOCK_ETA] {
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
        match &gamma_eval.block_working_sets[GammaLogFamily::BLOCK_ETA] {
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
    fn gaussian_log_sigma_weight_directional_derivative_is_zero_on_active_floor_branch() {
        let family = GaussianLocationScaleFamily {
            y: Array1::from_vec(vec![0.3]),
            weights: Array1::from_vec(vec![1.0]),
            mu_design: None,
            log_sigma_design: None,
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
            .diagonal_working_weights_directional_derivative(
                &states,
                GaussianLocationScaleFamily::BLOCK_LOG_SIGMA,
                &d_eta,
            )
            .expect("gaussian directional derivative")
            .expect("gaussian log-sigma derivative");
        assert_eq!(dw[0], 0.0);
    }

    #[test]
    fn gaussian_log_sigma_weight_directional_derivative_matches_finite_difference() {
        let family = GaussianLocationScaleFamily {
            y: Array1::from_vec(vec![1.2]),
            weights: Array1::from_vec(vec![1.0]),
            mu_design: None,
            log_sigma_design: None,
        };
        let eta_mu = Array1::from_vec(vec![0.1]);
        let eta_ls = Array1::from_vec(vec![0.4]);
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_mu.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_ls.clone(),
            },
        ];
        let d_eta = Array1::from_vec(vec![1.0]);

        let dw = family
            .diagonal_working_weights_directional_derivative(
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
            match &eval_plus.block_working_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
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
            match &eval_minus.block_working_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
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
    fn fit_binomial_location_scale_runs_with_warm_start_path() {
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
    fn fit_binomial_location_scale_sas_runs() {
        let n = 28usize;
        let y = Array1::from_vec((0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect());
        let weights = Array1::from_vec(vec![1.0; n]);
        let sas = crate::mixture_link::state_from_sas_spec(crate::types::SasLinkSpec {
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
        }
    }

    #[test]
    fn gaussian_location_scale_terms_reject_invalid_weights_early() {
        let n = 8usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64;
            data[[i, 1]] = (i as f64).sin();
        }
        let spec = GaussianLocationScaleTermSpec {
            y: Array1::zeros(n),
            weights: Array1::from_vec(vec![1.0, 1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
            mean_spec: simple_matern_term_collection(&[0, 1], 0.35),
            log_sigma_spec: simple_matern_term_collection(&[0, 1], 0.6),
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
            threshold_spec: simple_matern_term_collection(&[0, 1], 0.4),
            log_sigma_spec: simple_matern_term_collection(&[0, 1], 0.75),
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
    fn binomial_location_scale_terms_reject_data_row_mismatch_early() {
        let n = 8usize;
        let data = Array2::<f64>::zeros((n - 1, 2));
        let spec = BinomialLocationScaleTermSpec {
            y: Array1::from_elem(n, 0.0),
            weights: Array1::from_elem(n, 1.0),
            link_kind: InverseLink::Standard(LinkFunction::Probit),
            threshold_spec: simple_matern_term_collection(&[0, 1], 0.4),
            log_sigma_spec: simple_matern_term_collection(&[0, 1], 0.75),
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
    fn gaussian_location_scale_terms_with_matern_spatial_blocks_fit_finitely() {
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
            mean_spec: simple_matern_term_collection(&[0, 1], 0.35),
            log_sigma_spec: simple_matern_term_collection(&[0, 1], 0.6),
        };
        let fit = fit_gaussian_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect("gaussian location-scale spatial fit");
        assert!(fit.fit.penalized_objective.is_finite());
        assert_eq!(fit.fit.block_states.len(), 2);
    }

    #[test]
    fn binomial_location_scale_terms_with_matern_spatial_blocks_fit_finitely() {
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
            threshold_spec: simple_matern_term_collection(&[0, 1], 0.4),
            log_sigma_spec: simple_matern_term_collection(&[0, 1], 0.75),
        };
        let fit = fit_binomial_location_scale_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect("binomial location-scale spatial fit");
        assert!(fit.fit.penalized_objective.is_finite());
        assert_eq!(fit.fit.block_states.len(), 2);
    }

    #[test]
    fn binomial_location_scale_wiggle_terms_with_matern_spatial_blocks_fit_finitely() {
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
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
            threshold_spec: simple_matern_term_collection(&[0, 1], 0.45),
            log_sigma_spec: simple_matern_term_collection(&[0, 1], 0.8),
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        let fit = fit_binomial_location_scale_wiggle_terms(
            data.view(),
            spec,
            &BlockwiseFitOptions::default(),
            &spatial_kappa_options(),
        )
        .expect("binomial location-scale wiggle spatial fit");
        assert!(fit.fit.penalized_objective.is_finite());
        assert_eq!(fit.fit.block_states.len(), 3);
    }

    #[test]
    fn wiggle_chain_rule_scales_location_and_scale_working_weights() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![0.3, -0.2, 0.4, -0.1, 0.2, -0.3]);
        let eta_ls = Array1::from_vec(vec![-0.4, -0.1, 0.0, 0.1, 0.2, -0.2]);

        let q_seed = Array1::from_vec(vec![-1.5, -0.8, -0.1, 0.4, 1.1, 1.7]);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
        let beta_wiggle = Array1::from_vec(vec![0.1; wiggle_block.design.ncols()]);
        let eta_w = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&beta_wiggle);

        let core = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&eta_w),
            &family.link_kind,
        )
        .expect("core");
        let dq_dq0 = family
            .wiggle_dq_dq0(core.q0.view(), beta_wiggle.view())
            .expect("dq/dq0");
        let (t_ws, ls_ws, _w) = binomial_location_scale_working_sets(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&eta_w),
            Some(&dq_dq0),
            None,
            &core,
        )
        .expect("working sets");
        let t_w = match &t_ws {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => working_weights,
            BlockWorkingSet::ExactNewton { .. } => {
                panic!("expected diagonal working set for threshold block")
            }
        };
        let ls_w = match &ls_ws {
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
            let expected_w_t = (weights[i] * (dmu_t * dmu_t / var)).max(MIN_WEIGHT);
            let expected_w_ls = (weights[i] * (dmu_ls * dmu_ls / var)).max(MIN_WEIGHT);
            assert!(
                (t_w[i] - expected_w_t).abs() < 1e-10,
                "threshold weight mismatch at {}: got {}, expected {}",
                i,
                t_w[i],
                expected_w_t
            );
            assert!(
                (ls_w[i] - expected_w_ls).abs() < 1e-10,
                "log-sigma weight mismatch at {}: got {}, expected {}",
                i,
                ls_w[i],
                expected_w_ls
            );
        }
    }

    #[test]
    fn clamped_binomial_working_sets_stay_locally_flat() {
        let n = 4usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![1.0e6, -1.0e6, 1.0e6, -1.0e6]);
        let eta_ls = Array1::zeros(n);
        let eta_w = Array1::from_vec(vec![0.2, -0.1, 0.3, -0.2]);
        let dq_dq0 = Array1::from_vec(vec![1.0; n]);

        let core = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&eta_w),
            &InverseLink::Standard(LinkFunction::Probit),
        )
        .expect("core");
        assert!(core.clamp_active.iter().all(|v| *v));

        let (t_ws, ls_ws, w_ws) = binomial_location_scale_working_sets(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&eta_w),
            Some(&dq_dq0),
            None,
            &core,
        )
        .expect("working sets");

        match &t_ws {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_response, &eta_t);
                assert!(working_weights.iter().all(|w| *w == 0.0));
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal threshold block"),
        }
        match &ls_ws {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_response, &eta_ls);
                assert!(working_weights.iter().all(|w| *w == 0.0));
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal log-sigma block"),
        }
        match w_ws.expect("wiggle block") {
            BlockWorkingSet::Diagonal {
                working_response,
                working_weights,
            } => {
                assert_eq!(working_response, eta_w);
                assert!(working_weights.iter().all(|w| *w == 0.0));
            }
            BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal wiggle block"),
        }
    }

    #[test]
    fn wiggle_family_evaluate_returns_exact_newton_blocks() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.5, 1.5, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
        let beta_w = Array1::from_vec(vec![0.05; wiggle_block.design.ncols()]);
        let eta_w = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&beta_w);
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
                    beta: beta_w.clone(),
                    eta: eta_w,
                },
            ])
            .expect("evaluate");

        assert_eq!(eval.block_working_sets.len(), 3);
        match &eval.block_working_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                let hessian = hessian.to_dense();
                assert_eq!(gradient.len(), 1);
                assert_eq!(hessian.dim(), (1, 1));
                assert!(gradient[0].is_finite());
                assert!(hessian[[0, 0]].is_finite());
            }
            BlockWorkingSet::Diagonal { .. } => panic!("threshold block should be exact newton"),
        }
        match &eval.block_working_sets[1] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                let hessian = hessian.to_dense();
                assert_eq!(gradient.len(), 1);
                assert_eq!(hessian.dim(), (1, 1));
                assert!(gradient[0].is_finite());
                assert!(hessian[[0, 0]].is_finite());
            }
            BlockWorkingSet::Diagonal { .. } => panic!("log-sigma block should be exact newton"),
        }
        match &eval.block_working_sets[2] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                let hessian = hessian.to_dense();
                assert_eq!(gradient.len(), beta_w.len());
                assert_eq!(hessian.nrows(), beta_w.len());
                assert_eq!(hessian.ncols(), beta_w.len());
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
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
        let eta_t = threshold_design.matrix_vector_multiply(&beta_t);
        let eta_ls = log_sigma_design.matrix_vector_multiply(&beta_ls);
        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let beta_w = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let eta_w = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&beta_w);

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
                beta: beta_w.clone(),
                eta: eta_w.clone(),
            },
        ];

        let extract = |eval: FamilyEvaluation, idx: usize| -> Array2<f64> {
            match &eval.block_working_sets[idx] {
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
                .matrix_vector_multiply(
                    &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta,
                );
            plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta = log_sigma_design
                .matrix_vector_multiply(
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
            crate::testing::assert_matrix_derivative_fd(
                &fd,
                &analytic,
                5e-4,
                &format!("block {} dH", block_idx),
            );
        }
    }

    #[test]
    fn wiggle_family_joint_exact_hessian_directional_derivative_matches_finite_difference() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
        let eta_t = threshold_design.matrix_vector_multiply(&beta_t);
        let eta_ls = log_sigma_design.matrix_vector_multiply(&beta_ls);
        let core_for_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let beta_w = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let eta_w = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&beta_w);
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
                beta: beta_w.clone(),
                eta: eta_w,
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
        let beta_layout = GamlssBetaLayout::with_wiggle(
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
        let (dir_t, dir_ls, dir_w) = beta_layout
            .split_three(&direction, "wiggle test direction split")
            .expect("split wiggle test direction");
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta =
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta + &(eps * dir_t);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta =
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta + &(eps * dir_ls);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta =
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta + &(eps * dir_w);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta = threshold_design
            .matrix_vector_multiply(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta = log_sigma_design
            .matrix_vector_multiply(
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
        crate::testing::assert_matrix_derivative_fd(&fd, &analytic, 2e-3, "joint dH");
    }

    #[test]
    fn wiggle_family_joint_exact_hessian_second_directional_derivative_matches_finite_difference() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
                              beta_w: &Array1<f64>|
         -> Vec<ParameterBlockState> {
            let eta_t = threshold_design.matrix_vector_multiply(beta_t);
            let eta_ls = log_sigma_design.matrix_vector_multiply(beta_ls);
            let core_q0 = binomial_location_scale_core(
                &y,
                &weights,
                &eta_t,
                &eta_ls,
                None,
                &family.link_kind,
            )
            .expect("core q0");
            let eta_w = family
                .wiggle_design(core_q0.q0.view())
                .expect("wiggle design")
                .dot(beta_w);
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
                    beta: beta_w.clone(),
                    eta: eta_w,
                },
            ]
        };

        let beta_t = Array1::from_vec(vec![0.25]);
        let beta_ls = Array1::from_vec(vec![-0.15]);
        let beta_w = Array1::from_vec(vec![0.03; wiggle_block.design.ncols()]);
        let states = rebuild_states(&beta_t, &beta_ls, &beta_w);

        let pt = beta_t.len();
        let pls = beta_ls.len();
        let pw = beta_w.len();
        let total = pt + pls + pw;
        let direction_u = Array1::from_shape_fn(total, |k| 0.2 + 0.1 * (k as f64));
        let direction_v = Array1::from_shape_fn(total, |k| -0.15 + 0.07 * (k as f64));

        let analytic = family
            .exact_newton_joint_hessian_second_directional_derivative(
                &states,
                &direction_u,
                &direction_v,
            )
            .expect("joint d2H")
            .expect("expected joint exact d2H");

        let eps = 1e-6;
        let beta_layout = GamlssBetaLayout::with_wiggle(pt, pls, pw);
        let (step_t, step_ls, step_w) = beta_layout
            .split_three(&direction_v, "wiggle d2H test direction_v")
            .expect("split wiggle test direction");

        let states_plus = rebuild_states(
            &(&beta_t + &(eps * &step_t)),
            &(&beta_ls + &(eps * &step_ls)),
            &(&beta_w + &(eps * &step_w)),
        );
        let states_minus = rebuild_states(
            &(&beta_t - &(eps * &step_t)),
            &(&beta_ls - &(eps * &step_ls)),
            &(&beta_w - &(eps * &step_w)),
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

        crate::testing::assert_matrix_derivative_fd(&fd, &analytic, 4e-3, "joint d2H");
    }

    #[test]
    fn wiggle_family_joint_hessian_cross_blocks_match_finite_difference_of_gradients() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_block = intercept_block(n);
        let log_sigma_block = intercept_block(n);
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
                              beta_w: &Array1<f64>|
         -> Vec<ParameterBlockState> {
            let eta_t = threshold_design.matrix_vector_multiply(beta_t);
            let eta_ls = log_sigma_design.matrix_vector_multiply(beta_ls);
            let core_q0 = binomial_location_scale_core(
                &y,
                &weights,
                &eta_t,
                &eta_ls,
                None,
                &family.link_kind,
            )
            .expect("core q0");
            let eta_w = family
                .wiggle_design(core_q0.q0.view())
                .expect("wiggle design")
                .dot(beta_w);
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
                    beta: beta_w.clone(),
                    eta: eta_w,
                },
            ]
        };

        let extract_gradient = |eval: &FamilyEvaluation, block_idx: usize| -> Array1<f64> {
            match &eval.block_working_sets[block_idx] {
                BlockWorkingSet::ExactNewton {
                    gradient,
                    hessian: _,
                } => gradient.clone(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
            }
        };

        let beta_t = Array1::from_vec(vec![0.25]);
        let beta_ls = Array1::from_vec(vec![-0.15]);
        let beta_w = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
        let states = rebuild_states(&beta_t, &beta_ls, &beta_w);

        let h_joint = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian")
            .expect("expected joint exact hessian");

        let pt = beta_t.len();
        let pls = beta_ls.len();
        let pw = beta_w.len();
        let eps = 1e-6;

        let fd_cross_block = |target_block: usize, source_block: usize| -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((
                states[target_block].beta.len(),
                states[source_block].beta.len(),
            ));
            for j in 0..states[source_block].beta.len() {
                let mut beta_t_plus = beta_t.clone();
                let mut beta_ls_plus = beta_ls.clone();
                let mut beta_w_plus = beta_w.clone();
                let mut beta_t_minus = beta_t.clone();
                let mut beta_ls_minus = beta_ls.clone();
                let mut beta_w_minus = beta_w.clone();
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
                        beta_w_plus[j] += eps;
                        beta_w_minus[j] -= eps;
                    }
                    _ => panic!("unexpected block"),
                }

                let eval_plus = family
                    .evaluate(&rebuild_states(&beta_t_plus, &beta_ls_plus, &beta_w_plus))
                    .expect("eval plus");
                let eval_minus = family
                    .evaluate(&rebuild_states(
                        &beta_t_minus,
                        &beta_ls_minus,
                        &beta_w_minus,
                    ))
                    .expect("eval minus");
                let grad_plus = extract_gradient(&eval_plus, target_block);
                let grad_minus = extract_gradient(&eval_minus, target_block);
                let col = (&grad_plus - &grad_minus).mapv(|v| -v / (2.0 * eps));
                out.slice_mut(ndarray::s![.., j]).assign(&col);
            }
            out
        };

        let fd_t_ls = fd_cross_block(
            BinomialLocationScaleWiggleFamily::BLOCK_T,
            BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
        );
        let fd_t_w = fd_cross_block(
            BinomialLocationScaleWiggleFamily::BLOCK_T,
            BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
        );
        let fd_ls_w = fd_cross_block(
            BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
            BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
        );

        let h_t_ls = h_joint.slice(ndarray::s![0..pt, pt..pt + pls]).to_owned();
        let h_t_w = h_joint
            .slice(ndarray::s![0..pt, pt + pls..pt + pls + pw])
            .to_owned();
        let h_ls_w = h_joint
            .slice(ndarray::s![pt..pt + pls, pt + pls..pt + pls + pw])
            .to_owned();

        crate::testing::assert_matrix_derivative_fd(&fd_t_ls, &h_t_ls, 2e-4, "H_t_ls");
        crate::testing::assert_matrix_derivative_fd(&fd_t_w, &h_t_w, 4e-4, "H_t_w");
        crate::testing::assert_matrix_derivative_fd(&fd_ls_w, &h_ls_w, 6e-4, "H_ls_w");
    }

    #[test]
    fn wiggle_constraint_removes_constant_and_linear_modes() {
        let q_seed = Array1::linspace(-2.0, 2.0, 17);
        let degree = 3usize;
        let num_internal_knots = 6usize;
        let penalty_order = 2usize;

        let (block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
            q_seed.view(),
            degree,
            num_internal_knots,
            penalty_order,
            false,
        )
        .expect("wiggle block");
        let (z, _s_constrained) =
            compute_geometric_constraint_transform(&knots, degree, penalty_order)
                .expect("constraint transform");
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
    fn degenerate_wiggle_seed_uses_broad_fallback_domain() {
        let q_seed = Array1::zeros(9);
        let degree = 3usize;
        let knots = initialize_wiggle_knots_from_seed(q_seed.view(), degree, 5)
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

        let (block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
        let (z, _s_constrained) =
            compute_geometric_constraint_transform(&knots, degree, penalty_order)
                .expect("constraint transform");
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
    fn binomial_location_scale_generative_matches_core_mu() {
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
        let spec = family.generative_spec(&states).expect("generative spec");
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
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::build_wiggle_block_input(
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
        let beta_w = Array1::from_vec(vec![0.15; wiggle_block.design.ncols()]);
        let eta_w = family
            .wiggle_design(core_for_q0.q0.view())
            .expect("wiggle design")
            .dot(&beta_w);

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
                beta: beta_w.clone(),
                eta: eta_w.clone(),
            },
        ];

        let wiggle_spec = wiggle_block
            .clone()
            .into_spec("wiggle")
            .expect("wiggle spec");
        let (geom_x, _geom_offset) = family
            .block_geometry(&states, &wiggle_spec)
            .expect("block geometry");
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

        let generated = family.generative_spec(&states).expect("generative spec");
        let core = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            Some(&eta_w),
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
}
