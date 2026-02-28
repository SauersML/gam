use crate::basis::{
    BasisOptions, Dense, KnotSource, compute_geometric_constraint_transform, create_basis,
};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    KnownLinkWiggle, ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
use crate::faer_ndarray::{fast_ata, fast_atv};
use crate::generative::{CustomFamilyGenerative, GenerativeSpec, NoiseModel};
use crate::matrix::DesignMatrix;
use crate::pirls::WorkingLikelihood as EngineWorkingLikelihood;
use crate::probability::{normal_cdf_approx, normal_pdf};
use crate::families::sigma_link::{
    bounded_sigma_and_deriv_from_eta_scalar, bounded_sigma_from_eta_scalar,
};
use crate::smooth::{
    MaternKappaOptimizationOptions, TermCollectionDesign, TermCollectionSpec,
    optimize_two_block_matern_kappa,
};
use crate::types::{LikelihoodFamily, LinkFunction};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use ndarray::{Array1, Array2, ArrayView1, s};
const MIN_PROB: f64 = 1e-10;
const MIN_DERIV: f64 = 1e-8;
const MIN_WEIGHT: f64 = 1e-12;
const BETA_RANGE_WARN_THRESHOLD: f64 = 1.10;
const BINOMIAL_EFFECTIVE_N_WARN_THRESHOLD: f64 = 25.0;

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

fn validate_sigma_bounds(sigma_min: f64, sigma_max: f64, context: &str) -> Result<(), String> {
    if !sigma_min.is_finite() || !sigma_max.is_finite() {
        return Err(format!("{context}: sigma bounds must be finite"));
    }
    if sigma_min <= 0.0 || sigma_max <= 0.0 {
        return Err(format!(
            "{context}: sigma bounds must be strictly positive (got min={sigma_min}, max={sigma_max})"
        ));
    }
    if sigma_min > sigma_max {
        return Err(format!(
            "{context}: sigma_min ({sigma_min}) must be <= sigma_max ({sigma_max})"
        ));
    }
    Ok(())
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
    let seed_min = seed.iter().copied().fold(f64::INFINITY, f64::min);
    let mut seed_max = seed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !seed_min.is_finite() || !seed_max.is_finite() {
        return Err("non-finite seed for wiggle knot initialization".to_string());
    }
    if (seed_max - seed_min).abs() < 1e-12 {
        seed_max = seed_min + 1e-6;
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

/// Shared single-block GLM evaluation adapter backed by the engine-level
/// `WorkingLikelihood` implementation used by PIRLS.
fn evaluate_single_block_glm(
    family: LikelihoodFamily,
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
        .irls_update(y.view(), eta, weights.view(), &mut mu, &mut w, &mut z, None)
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

    let (mut xtwx, xtwy) = match design {
        DesignMatrix::Dense(x) => {
            let mut xw = x.clone();
            for i in 0..n {
                let sw = weights[i].max(0.0).sqrt();
                if sw != 1.0 {
                    let mut row = xw.row_mut(i);
                    row *= sw;
                }
            }
            let xtwx = fast_ata(&xw);
            let mut y_w = target_eta - offset;
            for i in 0..n {
                y_w[i] *= weights[i].max(0.0).sqrt();
            }
            let xtwy = fast_atv(&xw, &y_w);
            (xtwx, xtwy)
        }
        DesignMatrix::Sparse(xs) => {
            let csr = xs
                .as_ref()
                .to_row_major()
                .map_err(|_| "failed to obtain CSR view for weighted projection".to_string())?;
            let sym = csr.symbolic();
            let row_ptr = sym.row_ptr();
            let col_idx = sym.col_idx();
            let vals = csr.val();
            let mut xtwx = Array2::<f64>::zeros((p, p));
            let mut xtwy = Array1::<f64>::zeros(p);

            for i in 0..n {
                let wi = weights[i].max(0.0);
                if wi == 0.0 {
                    continue;
                }
                let y_star = target_eta[i] - offset[i];
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                for a_ptr in start..end {
                    let a = col_idx[a_ptr];
                    let xa = vals[a_ptr];
                    xtwy[a] += wi * xa * y_star;
                    for b_ptr in a_ptr..end {
                        let b = col_idx[b_ptr];
                        let xb = vals[b_ptr];
                        let v = wi * xa * xb;
                        xtwx[[a, b]] += v;
                        if a != b {
                            xtwx[[b, a]] += v;
                        }
                    }
                }
            }
            (xtwx, xtwy)
        }
    };
    for a in 0..p {
        xtwx[[a, a]] += ridge_floor.max(1e-12);
    }

    let h = crate::faer_ndarray::FaerArrayView::new(&xtwx);
    let mut rhs_mat = FaerMat::zeros(p, 1);
    for i in 0..p {
        rhs_mat[(i, 0)] = xtwy[i];
    }

    if let Ok(ch) = FaerLlt::new(h.as_ref(), Side::Lower) {
        ch.solve_in_place(rhs_mat.as_mut());
    } else if let Ok(ld) = FaerLdlt::new(h.as_ref(), Side::Lower) {
        ld.solve_in_place(rhs_mat.as_mut());
    } else {
        let lb = FaerLblt::new(h.as_ref(), Side::Lower);
        lb.solve_in_place(rhs_mat.as_mut());
    }

    let mut beta = Array1::<f64>::zeros(p);
    for i in 0..p {
        beta[i] = rhs_mat[(i, 0)];
    }
    if beta.iter().any(|v| !v.is_finite()) {
        return Err("solve_weighted_projection produced non-finite coefficients".to_string());
    }
    Ok(beta)
}

fn weighted_prevalence(y: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let w_sum: f64 = weights.iter().copied().sum();
    if w_sum <= 0.0 {
        return 0.5;
    }
    let y_w_sum: f64 = y.iter().zip(weights.iter()).map(|(&yi, &wi)| yi * wi).sum();
    (y_w_sum / w_sum).clamp(0.0, 1.0)
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
    beta_min: f64,
    beta_max: f64,
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
            let raw_beta = eta_beta[i];
            let beta = raw_beta.clamp(self.beta_min, self.beta_max);
            let dbeta_deta = if raw_beta >= self.beta_min && raw_beta <= self.beta_max {
                1.0
            } else {
                0.0
            };
            let q = eta_alpha[i];
            let chain_beta = dbeta_deta * beta;
            let mu = normal_cdf_approx(q).clamp(MIN_PROB, 1.0 - MIN_PROB);
            let dmu_dq = normal_pdf(q).max(MIN_DERIV);
            let var = (mu * (1.0 - mu)).max(MIN_PROB);

            ll += self.weights[i] * (self.y[i] * mu.ln() + (1.0 - self.y[i]) * (1.0 - mu).ln());

            let dmu_alpha = dmu_dq;
            w_alpha[i] = (self.weights[i] * (dmu_alpha * dmu_alpha / var)).max(MIN_WEIGHT);
            z_alpha[i] = eta_alpha[i] + (self.y[i] - mu) / signed_with_floor(dmu_alpha, MIN_DERIV);

            let dmu_beta = dmu_dq * chain_beta;
            w_beta[i] = (self.weights[i] * (dmu_beta * dmu_beta / var)).max(MIN_WEIGHT);
            z_beta[i] = eta_beta[i] + (self.y[i] - mu) / signed_with_floor(dmu_beta, MIN_DERIV);
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

    fn post_update_beta(&self, beta: Array1<f64>) -> Result<Array1<f64>, String> {
        Ok(beta.mapv(|v| v.clamp(self.beta_min, self.beta_max)))
    }
}

fn try_binomial_alpha_beta_warm_start(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    sigma_min: f64,
    sigma_max: f64,
    threshold_block: &ParameterBlockInput,
    log_sigma_block: &ParameterBlockInput,
    options: &BlockwiseFitOptions,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let beta_min = (1.0 / sigma_max.max(1e-12)).max(1e-12);
    let beta_max = (1.0 / sigma_min.max(1e-12)).max(beta_min + 1e-12);
    let warm_family = BinomialAlphaBetaWarmStartFamily {
        y: y.clone(),
        weights: weights.clone(),
        beta_min,
        beta_max,
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
    };
    let warm_fit = fit_custom_family(&warm_family, &[alpha_spec, beta_spec], &warm_options)?;
    let eta_alpha = &warm_fit.block_states[BinomialAlphaBetaWarmStartFamily::BLOCK_ALPHA].eta;
    let eta_beta = &warm_fit.block_states[BinomialAlphaBetaWarmStartFamily::BLOCK_BETA].eta;
    if eta_alpha.len() != y.len() || eta_beta.len() != y.len() {
        return Err("warm start eta length mismatch".to_string());
    }

    let beta_obs = eta_beta.mapv(|v| v.clamp(beta_min, beta_max));
    let t_target = Array1::from_iter(
        eta_alpha
            .iter()
            .zip(beta_obs.iter())
            .map(|(&a, &b)| -a / b.max(1e-12)),
    );
    let log_sigma_target = beta_obs.mapv(|b| -b.max(1e-12).ln());
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
    pub sigma_min: f64,
    pub sigma_max: f64,
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
pub struct BinomialLocationScaleProbitSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub threshold_block: ParameterBlockInput,
    pub log_sigma_block: ParameterBlockInput,
}

#[derive(Clone)]
pub struct BinomialLocationScaleProbitWiggleSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
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
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub mean_spec: TermCollectionSpec,
    pub log_sigma_spec: TermCollectionSpec,
}

#[derive(Clone)]
pub struct BinomialLocationScaleProbitTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub threshold_spec: TermCollectionSpec,
    pub log_sigma_spec: TermCollectionSpec,
}

#[derive(Clone)]
pub struct BinomialLocationScaleProbitWiggleTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub threshold_spec: TermCollectionSpec,
    pub log_sigma_spec: TermCollectionSpec,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    pub wiggle_block: ParameterBlockInput,
}

pub struct BlockwiseTermFitResult {
    pub fit: BlockwiseFitResult,
    pub mean_spec_resolved: TermCollectionSpec,
    pub noise_spec_resolved: TermCollectionSpec,
    pub mean_design: TermCollectionDesign,
    pub noise_design: TermCollectionDesign,
}

pub fn fit_gaussian_location_scale(
    spec: GaussianLocationScaleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_gaussian_location_scale")?;
    validate_sigma_bounds(
        spec.sigma_min,
        spec.sigma_max,
        "fit_gaussian_location_scale",
    )?;
    validate_block_rows("mu", n, &spec.mu_block)?;
    validate_block_rows("log_sigma", n, &spec.log_sigma_block)?;

    let family = GaussianLocationScaleFamily {
        y: spec.y,
        weights: spec.weights,
        sigma_min: spec.sigma_min,
        sigma_max: spec.sigma_max,
    };
    let blocks = vec![
        spec.mu_block.into_spec("mu")?,
        spec.log_sigma_block.into_spec("log_sigma")?,
    ];
    fit_custom_family(&family, &blocks, options)
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
    fit_custom_family(&family, &blocks, options)
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
    fit_custom_family(&family, &blocks, options)
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
    fit_custom_family(&family, &blocks, options)
}

pub fn fit_binomial_location_scale_probit(
    spec: BinomialLocationScaleProbitSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_binomial_location_scale_probit")?;
    validate_binomial_response(&spec.y, "fit_binomial_location_scale_probit")?;
    validate_sigma_bounds(
        spec.sigma_min,
        spec.sigma_max,
        "fit_binomial_location_scale_probit",
    )?;
    validate_block_rows("threshold", n, &spec.threshold_block)?;
    validate_block_rows("log_sigma", n, &spec.log_sigma_block)?;

    let BinomialLocationScaleProbitSpec {
        y,
        weights,
        sigma_min,
        sigma_max,
        mut threshold_block,
        mut log_sigma_block,
    } = spec;

    match try_binomial_alpha_beta_warm_start(
        &y,
        &weights,
        sigma_min,
        sigma_max,
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
                "[GAMLSS][fit_binomial_location_scale_probit] alpha/beta warm start failed, falling back to direct initialization: {}",
                err
            );
        }
    }

    let family = BinomialLocationScaleProbitFamily {
        y: y.clone(),
        weights: weights.clone(),
        sigma_min,
        sigma_max,
    };
    let blocks = vec![
        threshold_block.into_spec("threshold")?,
        log_sigma_block.into_spec("log_sigma")?,
    ];
    let fit = fit_custom_family(&family, &blocks, options)?;
    let beta_final = fit.block_states[BinomialLocationScaleProbitFamily::BLOCK_LOG_SIGMA]
        .eta
        .mapv(|eta| 1.0 / bounded_sigma_from_eta_scalar(eta, sigma_min, sigma_max).max(1e-12));
    emit_binomial_alpha_beta_warnings("final-fit", &beta_final, &y, &weights);
    Ok(fit)
}

pub fn fit_binomial_location_scale_probit_wiggle(
    spec: BinomialLocationScaleProbitWiggleSpec,
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let n = spec.y.len();
    validate_len_match("weights vs y", n, spec.weights.len())?;
    validate_weights(&spec.weights, "fit_binomial_location_scale_probit_wiggle")?;
    validate_binomial_response(&spec.y, "fit_binomial_location_scale_probit_wiggle")?;
    validate_sigma_bounds(
        spec.sigma_min,
        spec.sigma_max,
        "fit_binomial_location_scale_probit_wiggle",
    )?;
    validate_block_rows("threshold", n, &spec.threshold_block)?;
    validate_block_rows("log_sigma", n, &spec.log_sigma_block)?;
    validate_block_rows("wiggle", n, &spec.wiggle_block)?;
    if spec.wiggle_degree < 1 {
        return Err(format!(
            "fit_binomial_location_scale_probit_wiggle: wiggle_degree must be >= 1, got {}",
            spec.wiggle_degree
        ));
    }
    if spec.wiggle_knots.len() < spec.wiggle_degree + 2 {
        return Err(format!(
            "fit_binomial_location_scale_probit_wiggle: wiggle_knots length {} is too short for degree {}",
            spec.wiggle_knots.len(),
            spec.wiggle_degree
        ));
    }

    let BinomialLocationScaleProbitWiggleSpec {
        y,
        weights,
        sigma_min,
        sigma_max,
        wiggle_knots,
        wiggle_degree,
        mut threshold_block,
        mut log_sigma_block,
        wiggle_block,
    } = spec;

    match try_binomial_alpha_beta_warm_start(
        &y,
        &weights,
        sigma_min,
        sigma_max,
        &threshold_block,
        &log_sigma_block,
        options,
    ) {
        Ok((beta_t0, beta_ls0, beta_warm)) => {
            threshold_block.initial_beta = Some(beta_t0);
            log_sigma_block.initial_beta = Some(beta_ls0);
            emit_binomial_alpha_beta_warnings("warm-start-wiggle", &beta_warm, &y, &weights);
        }
        Err(err) => {
            log::warn!(
                "[GAMLSS][fit_binomial_location_scale_probit_wiggle] alpha/beta warm start failed, falling back to direct initialization: {}",
                err
            );
        }
    }

    let family = BinomialLocationScaleProbitWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        sigma_min,
        sigma_max,
        wiggle_knots,
        wiggle_degree,
    };
    let blocks = vec![
        threshold_block.into_spec("threshold")?,
        log_sigma_block.into_spec("log_sigma")?,
        wiggle_block.into_spec("wiggle")?,
    ];
    let fit = fit_custom_family(&family, &blocks, options)?;
    let beta_final = fit.block_states[BinomialLocationScaleProbitWiggleFamily::BLOCK_LOG_SIGMA]
        .eta
        .mapv(|eta| 1.0 / bounded_sigma_from_eta_scalar(eta, sigma_min, sigma_max).max(1e-12));
    emit_binomial_alpha_beta_warnings("final-fit-wiggle", &beta_final, &y, &weights);
    Ok(fit)
}

pub fn fit_gaussian_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: GaussianLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &MaternKappaOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    let y = spec.y;
    let weights = spec.weights;
    let sigma_min = spec.sigma_min;
    let sigma_max = spec.sigma_max;
    let mut mean_log_lambda_hint: Option<Array1<f64>> = None;
    let mut noise_log_lambda_hint: Option<Array1<f64>> = None;
    let solved = optimize_two_block_matern_kappa(
        data,
        &spec.mean_spec,
        &spec.log_sigma_spec,
        kappa_options,
        |mean_design, noise_design| {
            let fit = fit_gaussian_location_scale(
                GaussianLocationScaleSpec {
                    y: y.clone(),
                    weights: weights.clone(),
                    sigma_min,
                    sigma_max,
                    mu_block: ParameterBlockInput {
                        design: DesignMatrix::Dense(mean_design.design.clone()),
                        offset: Array1::zeros(y.len()),
                        penalties: mean_design.penalties.clone(),
                        initial_log_lambdas: mean_log_lambda_hint.clone(),
                        initial_beta: None,
                    },
                    log_sigma_block: ParameterBlockInput {
                        design: DesignMatrix::Dense(noise_design.design.clone()),
                        offset: Array1::zeros(y.len()),
                        penalties: noise_design.penalties.clone(),
                        initial_log_lambdas: noise_log_lambda_hint.clone(),
                        initial_beta: None,
                    },
                },
                options,
            )?;
            let k_mean = mean_design.penalties.len();
            let k_noise = noise_design.penalties.len();
            if fit.log_lambdas.len() >= k_mean + k_noise {
                mean_log_lambda_hint = Some(fit.log_lambdas.slice(s![0..k_mean]).to_owned());
                noise_log_lambda_hint = Some(
                    fit.log_lambdas
                        .slice(s![k_mean..(k_mean + k_noise)])
                        .to_owned(),
                );
            }
            Ok(fit)
        },
        |fit| fit.penalized_objective,
    )?;
    Ok(BlockwiseTermFitResult {
        fit: solved.fit,
        mean_spec_resolved: solved.resolved_mean_spec,
        noise_spec_resolved: solved.resolved_noise_spec,
        mean_design: solved.mean_design,
        noise_design: solved.noise_design,
    })
}

pub fn fit_binomial_location_scale_probit_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleProbitTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &MaternKappaOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    let y = spec.y;
    let weights = spec.weights;
    let sigma_min = spec.sigma_min;
    let sigma_max = spec.sigma_max;
    let mut threshold_log_lambda_hint: Option<Array1<f64>> = None;
    let mut noise_log_lambda_hint: Option<Array1<f64>> = None;
    let solved = optimize_two_block_matern_kappa(
        data,
        &spec.threshold_spec,
        &spec.log_sigma_spec,
        kappa_options,
        |threshold_design, noise_design| {
            let fit = fit_binomial_location_scale_probit(
                BinomialLocationScaleProbitSpec {
                    y: y.clone(),
                    weights: weights.clone(),
                    sigma_min,
                    sigma_max,
                    threshold_block: ParameterBlockInput {
                        design: DesignMatrix::Dense(threshold_design.design.clone()),
                        offset: Array1::zeros(y.len()),
                        penalties: threshold_design.penalties.clone(),
                        initial_log_lambdas: threshold_log_lambda_hint.clone(),
                        initial_beta: None,
                    },
                    log_sigma_block: ParameterBlockInput {
                        design: DesignMatrix::Dense(noise_design.design.clone()),
                        offset: Array1::zeros(y.len()),
                        penalties: noise_design.penalties.clone(),
                        initial_log_lambdas: noise_log_lambda_hint.clone(),
                        initial_beta: None,
                    },
                },
                options,
            )?;
            let k_threshold = threshold_design.penalties.len();
            let k_noise = noise_design.penalties.len();
            if fit.log_lambdas.len() >= k_threshold + k_noise {
                threshold_log_lambda_hint =
                    Some(fit.log_lambdas.slice(s![0..k_threshold]).to_owned());
                noise_log_lambda_hint = Some(
                    fit.log_lambdas
                        .slice(s![k_threshold..(k_threshold + k_noise)])
                        .to_owned(),
                );
            }
            Ok(fit)
        },
        |fit| fit.penalized_objective,
    )?;
    Ok(BlockwiseTermFitResult {
        fit: solved.fit,
        mean_spec_resolved: solved.resolved_mean_spec,
        noise_spec_resolved: solved.resolved_noise_spec,
        mean_design: solved.mean_design,
        noise_design: solved.noise_design,
    })
}

pub fn fit_binomial_location_scale_probit_wiggle_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: BinomialLocationScaleProbitWiggleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &MaternKappaOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    let y = spec.y;
    let weights = spec.weights;
    let sigma_min = spec.sigma_min;
    let sigma_max = spec.sigma_max;
    let wiggle_knots = spec.wiggle_knots;
    let wiggle_degree = spec.wiggle_degree;
    let wiggle_block = spec.wiggle_block;
    let mut threshold_log_lambda_hint: Option<Array1<f64>> = None;
    let mut noise_log_lambda_hint: Option<Array1<f64>> = None;
    let solved = optimize_two_block_matern_kappa(
        data,
        &spec.threshold_spec,
        &spec.log_sigma_spec,
        kappa_options,
        |threshold_design, noise_design| {
            let fit = fit_binomial_location_scale_probit_wiggle(
                BinomialLocationScaleProbitWiggleSpec {
                    y: y.clone(),
                    weights: weights.clone(),
                    sigma_min,
                    sigma_max,
                    wiggle_knots: wiggle_knots.clone(),
                    wiggle_degree,
                    threshold_block: ParameterBlockInput {
                        design: DesignMatrix::Dense(threshold_design.design.clone()),
                        offset: Array1::zeros(y.len()),
                        penalties: threshold_design.penalties.clone(),
                        initial_log_lambdas: threshold_log_lambda_hint.clone(),
                        initial_beta: None,
                    },
                    log_sigma_block: ParameterBlockInput {
                        design: DesignMatrix::Dense(noise_design.design.clone()),
                        offset: Array1::zeros(y.len()),
                        penalties: noise_design.penalties.clone(),
                        initial_log_lambdas: noise_log_lambda_hint.clone(),
                        initial_beta: None,
                    },
                    wiggle_block: wiggle_block.clone(),
                },
                options,
            )?;
            let k_threshold = threshold_design.penalties.len();
            let k_noise = noise_design.penalties.len();
            if fit.log_lambdas.len() >= k_threshold + k_noise {
                threshold_log_lambda_hint =
                    Some(fit.log_lambdas.slice(s![0..k_threshold]).to_owned());
                noise_log_lambda_hint = Some(
                    fit.log_lambdas
                        .slice(s![k_threshold..(k_threshold + k_noise)])
                        .to_owned(),
                );
            }
            Ok(fit)
        },
        |fit| fit.penalized_objective,
    )?;
    Ok(BlockwiseTermFitResult {
        fit: solved.fit,
        mean_spec_resolved: solved.resolved_mean_spec,
        noise_spec_resolved: solved.resolved_noise_spec,
        mean_design: solved.mean_design,
        noise_design: solved.noise_design,
    })
}

/// Link identifiers for distribution parameters in multi-parameter GAMLSS families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterLink {
    Identity,
    Log,
    Logit,
    Probit,
    /// Learnable smooth departure from a known base link.
    Wiggle,
}

fn signed_with_floor(v: f64, floor: f64) -> f64 {
    let a = v.abs().max(floor);
    if v >= 0.0 { a } else { -a }
}

struct BinomialLocationScaleCore {
    sigma: Array1<f64>,
    dsigma_deta: Array1<f64>,
    q0: Array1<f64>,
    mu: Array1<f64>,
    dmu_dq: Array1<f64>,
    log_likelihood: f64,
}

fn binomial_location_scale_core(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    eta_wiggle: Option<&Array1<f64>>,
    sigma_min: f64,
    sigma_max: f64,
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
    let mut dmu_dq = Array1::<f64>::zeros(n);
    let mut ll = 0.0;

    for i in 0..n {
        let (sigma_i, dsigma_deta_i) =
            bounded_sigma_and_deriv_from_eta_scalar(eta_ls[i], sigma_min, sigma_max);
        sigma[i] = sigma_i;
        dsigma_deta[i] = dsigma_deta_i;
        q0[i] = -eta_t[i] / sigma[i].max(1e-12);
        let q = q0[i] + eta_wiggle.map_or(0.0, |w| w[i]);
        mu[i] = normal_cdf_approx(q).clamp(MIN_PROB, 1.0 - MIN_PROB);
        dmu_dq[i] = normal_pdf(q).max(MIN_DERIV);
        ll += weights[i] * (y[i] * mu[i].ln() + (1.0 - y[i]) * (1.0 - mu[i]).ln());
    }

    Ok(BinomialLocationScaleCore {
        sigma,
        dsigma_deta,
        q0,
        mu,
        dmu_dq,
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
    core: &BinomialLocationScaleCore,
) -> (BlockWorkingSet, BlockWorkingSet, Option<BlockWorkingSet>) {
    let n = y.len();
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
        w_t[i] = (weights[i] * (dmu_t * dmu_t / var)).max(MIN_WEIGHT);
        z_t[i] = eta_t[i] + (y[i] - core.mu[i]) / signed_with_floor(dmu_t, MIN_DERIV);

        // Scale chain: dq/deta_log_sigma = -q0 * dsigma/deta / sigma
        // This is the generic location-scale structure; the -Z multiplier appears here.
        let chain_ls = {
            let s = core.sigma[i].max(1e-12);
            -link_chain * core.q0[i] * core.dsigma_deta[i] / s
        };
        let dmu_ls = core.dmu_dq[i] * chain_ls;
        w_ls[i] = (weights[i] * (dmu_ls * dmu_ls / var)).max(MIN_WEIGHT);
        z_ls[i] = eta_ls[i] + (y[i] - core.mu[i]) / signed_with_floor(dmu_ls, MIN_DERIV);

        if let (Some(eta_w), Some(z_wv), Some(w_wv)) = (eta_wiggle, z_w.as_mut(), w_w.as_mut()) {
            // Wiggle enters additively in q, so chain is 1.
            let dmu_w = core.dmu_dq[i];
            w_wv[i] = (weights[i] * (dmu_w * dmu_w / var)).max(MIN_WEIGHT);
            z_wv[i] = eta_w[i] + (y[i] - core.mu[i]) / signed_with_floor(dmu_w, MIN_DERIV);
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
    (t_ws, ls_ws, w_ws)
}

/// Built-in Gaussian location-scale family:
/// - Block 0: location μ(·) with identity link
/// - Block 1: log-scale log σ(·) with log link
#[derive(Clone)]
pub struct GaussianLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
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
            let (sigma_i, dsigma_deta_i) = bounded_sigma_and_deriv_from_eta_scalar(
                eta_log_sigma[i],
                self.sigma_min,
                self.sigma_max,
            );
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
            w_mu[i] = (self.weights[i] / s2).max(MIN_WEIGHT);
            z_mu[i] = eta_mu[i] + r;

            // log-sigma block: IRLS working response and weights.
            // The score for log-sigma is d(ll)/d(eta_ls) = (r²/s² - 1) * dsigma/(sigma * deta).
            // Working response: z_ls = eta_ls + score / info.
            let dlogsigma_du = if dsigma_deta[i] == 0.0 {
                0.0
            } else {
                (dsigma_deta[i] / s).clamp(-1.0, 1.0)
            };
            let info_u = (2.0 * self.weights[i] * dlogsigma_du * dlogsigma_du).max(MIN_WEIGHT);
            w_ls[i] = info_u;
            // Score: d(ll)/d(eta_ls) = w_i * (r²/s² - 1) * dsigma/sigma
            let score_ls = self.weights[i] * (r * r / s2 - 1.0) * dlogsigma_du;
            z_ls[i] = eta_log_sigma[i] + score_ls / info_u;
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
        let sigma = block_states[Self::BLOCK_LOG_SIGMA]
            .eta
            .mapv(|eta| bounded_sigma_from_eta_scalar(eta, self.sigma_min, self.sigma_max));
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
        evaluate_single_block_glm(LikelihoodFamily::BinomialLogit, &self.y, &self.weights, eta)
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
            let e = eta[i].clamp(-30.0, 30.0);
            let m = e.exp().max(1e-12);
            mu[i] = m;
            // Drop log(y!) constant in objective.
            ll += self.weights[i] * (yi * e - m);
            let dmu = m.max(MIN_DERIV);
            let var = m.max(MIN_PROB);
            w[i] = (self.weights[i] * (dmu * dmu / var)).max(MIN_WEIGHT);
            z[i] = e + (yi - m) / signed_with_floor(dmu, MIN_DERIV);
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
            let e = eta[i].clamp(-30.0, 30.0);
            let m = e.exp().max(1e-12);
            mu[i] = m;
            // Gamma(shape=k, scale=mu/k), dropping constants independent of eta.
            ll += self.weights[i] * (-self.shape * (yi / m + m.ln()));
            let dmu = m.max(MIN_DERIV);
            let var = (m * m / self.shape).max(MIN_PROB);
            w[i] = (self.weights[i] * (dmu * dmu / var)).max(MIN_WEIGHT);
            z[i] = e + (yi - m) / signed_with_floor(dmu, MIN_DERIV);
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

/// Built-in binomial location-scale probit family.
///
/// Parameters:
/// - Block 0: threshold/location T(covariates)
/// - Block 1: log-scale log σ(covariates)
#[derive(Clone)]
pub struct BinomialLocationScaleProbitFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
}

impl BinomialLocationScaleProbitFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub fn parameter_names() -> &'static [&'static str] {
        &["threshold", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Probit, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scale_probit",
            parameter_names: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for BinomialLocationScaleProbitFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleProbitFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleProbitFamily input size mismatch".to_string());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            self.sigma_min,
            self.sigma_max,
        )?;
        let (t_ws, ls_ws, _none) = binomial_location_scale_working_sets(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            None,
            &core,
        );

        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            block_working_sets: vec![t_ws, ls_ws],
        })
    }
}

impl CustomFamilyGenerative for BinomialLocationScaleProbitFamily {
    fn generative_spec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BinomialLocationScaleProbitFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() {
            return Err("BinomialLocationScaleProbitFamily generative size mismatch".to_string());
        }
        let mut mean = Array1::<f64>::zeros(self.y.len());
        for i in 0..mean.len() {
            let sigma =
                bounded_sigma_from_eta_scalar(eta_ls[i], self.sigma_min, self.sigma_max).max(1e-12);
            let q = -eta_t[i] / sigma;
            mean[i] = normal_cdf_approx(q).clamp(MIN_PROB, 1.0 - MIN_PROB);
        }
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}

/// Built-in binomial location-scale probit with learnable wiggle on q.
///
/// Block structure:
/// - Block 0: threshold T(covariates)
/// - Block 1: log sigma(covariates)
/// - Block 2: wiggle(q) represented by B-spline coefficients on q
#[derive(Clone)]
pub struct BinomialLocationScaleProbitWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
}

impl BinomialLocationScaleProbitWiggleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameter_names() -> &'static [&'static str] {
        &["threshold", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::Probit,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scale_probit_wiggle",
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

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        let (basis, _) = create_basis::<Dense>(
            q0,
            KnotSource::Provided(self.wiggle_knots.view()),
            self.wiggle_degree,
            BasisOptions::value(),
        )
        .map_err(|e| e.to_string())?;
        let full = (*basis).clone();
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
        Ok(full.dot(&z))
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let (dbasis, _) = create_basis::<Dense>(
            q0,
            KnotSource::Provided(self.wiggle_knots.view()),
            self.wiggle_degree,
            BasisOptions::first_derivative(),
        )
        .map_err(|e| e.to_string())?;
        let full = (*dbasis).clone();
        if full.ncols() < 3 {
            return Err("wiggle derivative basis has fewer than three columns".to_string());
        }
        let z = self.wiggle_constraint_transform()?;
        if full.ncols() != z.nrows() {
            return Err(format!(
                "wiggle derivative/constraint mismatch: basis has {} columns but transform has {} rows",
                full.ncols(),
                z.nrows()
            ));
        }
        let d_constrained = full.dot(&z);
        if d_constrained.ncols() != beta_wiggle.len() {
            return Err(format!(
                "wiggle derivative col mismatch: got {}, expected {}",
                d_constrained.ncols(),
                beta_wiggle.len()
            ));
        }
        Ok(d_constrained.dot(&beta_wiggle) + 1.0)
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

impl CustomFamily for BinomialLocationScaleProbitWiggleFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleProbitWiggleFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let eta_w = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || eta_w.len() != n || self.weights.len() != n {
            return Err("BinomialLocationScaleProbitWiggleFamily input size mismatch".to_string());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(eta_w),
            self.sigma_min,
            self.sigma_max,
        )?;
        let dq_dq0 =
            self.wiggle_dq_dq0(core.q0.view(), block_states[Self::BLOCK_WIGGLE].beta.view())?;
        let (t_ws, ls_ws, w_ws) = binomial_location_scale_working_sets(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(eta_w),
            Some(&dq_dq0),
            &core,
        );
        let w_ws = w_ws.ok_or_else(|| "wiggle working set missing".to_string())?;

        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            block_working_sets: vec![t_ws, ls_ws, w_ws],
        })
    }

    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        Some(KnownLinkWiggle {
            base_link: LinkFunction::Probit,
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
            let sigma =
                bounded_sigma_from_eta_scalar(eta_ls[i], self.sigma_min, self.sigma_max).max(1e-12);
            q0[i] = -eta_t[i] / sigma;
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
}

impl CustomFamilyGenerative for BinomialLocationScaleProbitWiggleFamily {
    fn generative_spec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "BinomialLocationScaleProbitWiggleFamily expects 3 blocks, got {}",
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
            return Err(
                "BinomialLocationScaleProbitWiggleFamily generative size mismatch".to_string(),
            );
        }
        let mut mean = Array1::<f64>::zeros(self.y.len());
        for i in 0..mean.len() {
            let sigma =
                bounded_sigma_from_eta_scalar(eta_ls[i], self.sigma_min, self.sigma_max).max(1e-12);
            let q0 = -eta_t[i] / sigma;
            mean[i] = normal_cdf_approx(q0 + eta_w[i]).clamp(MIN_PROB, 1.0 - MIN_PROB);
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
    use crate::basis::compute_greville_abscissae;

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
            0.25,
            4.0,
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
    }

    #[test]
    fn fit_binomial_location_scale_probit_runs_with_warm_start_path() {
        let n = 32usize;
        let y = Array1::from_vec((0..n).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect());
        let weights = Array1::from_vec(vec![1.0; n]);
        let spec = BinomialLocationScaleProbitSpec {
            y,
            weights,
            sigma_min: 0.3,
            sigma_max: 3.0,
            threshold_block: intercept_block(n),
            log_sigma_block: intercept_block(n),
        };

        let fit = fit_binomial_location_scale_probit(spec, &BlockwiseFitOptions::default())
            .expect("binomial location-scale probit should fit");
        assert_eq!(fit.block_states.len(), 2);
        assert!(fit.log_likelihood.is_finite());
    }

    #[test]
    fn wiggle_chain_rule_scales_location_and_scale_working_weights() {
        let n = 6usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![0.3, -0.2, 0.4, -0.1, 0.2, -0.3]);
        let eta_ls = Array1::from_vec(vec![-0.4, -0.1, 0.0, 0.1, 0.2, -0.2]);

        let q_seed = Array1::from_vec(vec![-1.5, -0.8, -0.1, 0.4, 1.1, 1.7]);
        let (wiggle_block, knots) =
            BinomialLocationScaleProbitWiggleFamily::build_wiggle_block_input(
                q_seed.view(),
                2,
                3,
                2,
                false,
            )
            .expect("wiggle block");
        let family = BinomialLocationScaleProbitWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            sigma_min: 0.05,
            sigma_max: 20.0,
            wiggle_knots: knots,
            wiggle_degree: 2,
        };

        let core_for_q0 = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            None,
            family.sigma_min,
            family.sigma_max,
        )
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
            family.sigma_min,
            family.sigma_max,
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
            &core,
        );
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
    fn wiggle_constraint_removes_constant_and_linear_modes() {
        let q_seed = Array1::linspace(-2.0, 2.0, 17);
        let degree = 3usize;
        let num_internal_knots = 6usize;
        let penalty_order = 2usize;

        let (block, knots) = BinomialLocationScaleProbitWiggleFamily::build_wiggle_block_input(
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
    fn wiggle_block_design_matches_constrained_basis_projection() {
        let q_seed = Array1::linspace(-1.0, 1.0, 11);
        let degree = 2usize;
        let num_internal_knots = 4usize;
        let penalty_order = 2usize;

        let (block, knots) = BinomialLocationScaleProbitWiggleFamily::build_wiggle_block_input(
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
    fn binomial_location_scale_probit_generative_matches_core_mu() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let eta_t = Array1::from_vec(vec![0.8, -0.4, 0.2, -1.1, 0.0, 0.5, -0.7]);
        let eta_ls = Array1::from_vec(vec![-3.0, -1.2, -0.1, 0.3, 1.1, 2.0, 4.0]);

        let family = BinomialLocationScaleProbitFamily {
            y: y.clone(),
            weights: weights.clone(),
            sigma_min: 0.05,
            sigma_max: 10.0,
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
        let core = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            None,
            family.sigma_min,
            family.sigma_max,
        )
        .expect("core");
        for i in 0..n {
            assert!(
                (spec.mean[i] - core.mu[i]).abs() < 1e-12,
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
        let (wiggle_block, knots) =
            BinomialLocationScaleProbitWiggleFamily::build_wiggle_block_input(
                q_seed.view(),
                2,
                3,
                2,
                false,
            )
            .expect("wiggle block");

        let family = BinomialLocationScaleProbitWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            sigma_min: 0.1,
            sigma_max: 8.0,
            wiggle_knots: knots,
            wiggle_degree: 2,
        };

        let core_for_q0 = binomial_location_scale_core(
            &y,
            &weights,
            &eta_t,
            &eta_ls,
            None,
            family.sigma_min,
            family.sigma_max,
        )
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
            family.sigma_min,
            family.sigma_max,
        )
        .expect("core with wiggle");
        for i in 0..n {
            assert!(
                (generated.mean[i] - core.mu[i]).abs() < 1e-12,
                "wiggle mean mismatch at {i}: got {}, expected {}",
                generated.mean[i],
                core.mu[i]
            );
        }
    }
}
