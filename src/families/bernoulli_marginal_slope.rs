use crate::basis::{
    create_basis, create_difference_penalty_matrix, BasisOptions, Dense, KnotSource,
};
use crate::custom_family::{
    fit_custom_family, BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState,
};
use crate::estimate::{fit_gam, FitOptions, UnifiedFitResult};
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::gamlss::{initializewiggle_knots_from_seed, ParameterBlockInput};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf};
use crate::quadrature::compute_gauss_hermite_n;
use crate::smooth::{
    build_term_collection_design, freeze_spatial_length_scale_terms_from_design,
    optimize_two_block_spatial_length_scale_exact_joint, spatial_length_scale_term_indices,
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec, TwoBlockExactJointHyperSetup,
};
use crate::types::LikelihoodFamily;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct DeviationBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub derivative_grid_size: usize,
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
            derivative_grid_size: 64,
            monotonicity_eps: 1e-4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviationRuntime {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub transform: Array2<f64>,
}

#[derive(Clone)]
struct DeviationPrepared {
    block: ParameterBlockInput,
    runtime: DeviationRuntime,
    constraints: LinearInequalityConstraints,
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
    quadrature_nodes: Array1<f64>,
    quadrature_weights: Array1<f64>,
    score_warp: Option<DeviationRuntime>,
    score_warp_obs_design: Option<Array2<f64>>,
    link_dev: Option<DeviationRuntime>,
    score_warp_constraints: Option<LinearInequalityConstraints>,
    link_dev_constraints: Option<LinearInequalityConstraints>,
}

#[derive(Clone, Default)]
struct ThetaHints {
    marginal_beta: Option<Array1<f64>>,
    logslope_beta: Option<Array1<f64>>,
    score_warp_beta: Option<Array1<f64>>,
    link_dev_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
struct ImplicitRowDerivatives {
    intercept: f64,
    a_q: f64,
    a_b: f64,
    a_qq: f64,
    a_qb: f64,
    a_bb: f64,
    a_h: Option<Array1<f64>>,
    a_qh: Option<Array1<f64>>,
    a_bh: Option<Array1<f64>>,
    a_hh: Option<Array2<f64>>,
    a_w: Option<Array1<f64>>,
    a_qw: Option<Array1<f64>>,
    a_bw: Option<Array1<f64>>,
    a_hw: Option<Array2<f64>>,
    a_ww: Option<Array2<f64>>,
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
}

fn deviation_grid_from_knots(
    knots: &Array1<f64>,
    degree: usize,
    grid_size: usize,
) -> Result<Array1<f64>, String> {
    if knots.len() < degree + 2 {
        return Err(format!(
            "deviation derivative grid needs at least {} knots for degree {}, got {}",
            degree + 2,
            degree,
            knots.len()
        ));
    }
    let left = knots[degree];
    let right = knots[knots.len() - degree - 1];
    let n = grid_size.max(2);
    if (right - left).abs() < 1e-12 {
        return Ok(Array1::from_vec(vec![left; n]));
    }
    Ok(Array1::from_iter((0..n).map(|i| {
        let t = i as f64 / ((n - 1) as f64);
        left + (right - left) * t
    })))
}

fn deviation_transform(knots: &Array1<f64>, degree: usize) -> Result<Array2<f64>, String> {
    let anchor = Array1::from_vec(vec![0.0]);
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
        transform: deviation_transform(&knots, cfg.degree)?,
    };
    let design = runtime.design(seed)?;
    let raw_dim = runtime.transform.nrows();
    let dim = runtime.transform.ncols();
    let mut penalties = Vec::new();
    let base_penalty = create_difference_penalty_matrix(raw_dim, cfg.penalty_order, None)
        .map_err(|e| e.to_string())?;
    penalties.push(fast_ab(
        &fast_atb(&runtime.transform, &base_penalty),
        &runtime.transform,
    ));
    for &order in &cfg.penalty_orders {
        if order == cfg.penalty_order || order == 0 || order >= raw_dim {
            continue;
        }
        let raw =
            create_difference_penalty_matrix(raw_dim, order, None).map_err(|e| e.to_string())?;
        penalties.push(fast_ab(
            &fast_atb(&runtime.transform, &raw),
            &runtime.transform,
        ));
    }
    if cfg.double_penalty {
        penalties.push(Array2::<f64>::eye(dim));
    }
    let derivative_grid = deviation_grid_from_knots(&knots, cfg.degree, cfg.derivative_grid_size)?;
    let derivative_design = runtime.first_derivative_design(&derivative_grid)?;
    let constraints = LinearInequalityConstraints {
        a: derivative_design,
        b: Array1::from_elem(derivative_grid.len(), cfg.monotonicity_eps - 1.0),
    };
    Ok(DeviationPrepared {
        block: ParameterBlockInput {
            design: DesignMatrix::Dense(design),
            offset: Array1::zeros(seed.len()),
            penalties,
            initial_log_lambdas: None,
            initial_beta: Some(Array1::zeros(dim)),
        },
        runtime,
        constraints,
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
    Ok(())
}

fn pooled_probit_baseline(y: &Array1<f64>, z: &Array1<f64>) -> Result<(f64, f64), String> {
    let n = y.len();
    let mut x = Array2::<f64>::zeros((n, 2));
    x.column_mut(0).fill(1.0);
    x.column_mut(1).assign(z);
    let fit = fit_gam(
        x.view(),
        y.view(),
        Array1::ones(n).view(),
        Array1::zeros(n).view(),
        &[],
        LikelihoodFamily::BinomialProbit,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: Vec::new(),
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
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
) -> TwoBlockExactJointHyperSetup {
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
    TwoBlockExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

fn fit_score(fit: &UnifiedFitResult) -> f64 {
    if fit.reml_score.is_finite() {
        fit.reml_score
    } else {
        let score = 0.5 * fit.deviance + 0.5 * fit.stable_penalty_term;
        if score.is_finite() {
            score
        } else {
            f64::INFINITY
        }
    }
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

fn probit_neglog_derivatives(y: f64, weight: f64, q: f64) -> (f64, f64) {
    if weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0);
    }
    let m = normal_cdf(q).clamp(1e-12, 1.0 - 1e-12);
    let nu = 1.0 - m;
    let phi = normal_pdf(q);
    let a = (1.0 - y) / nu - y / m;
    let amu = (1.0 - y) / (nu * nu) + y / (m * m);
    let m1 = weight * a * phi;
    let m2 = weight * (amu * phi * phi - q * a * phi);
    (m1, m2)
}

struct BlockSlices {
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    h: Option<std::ops::Range<usize>>,
    w: Option<std::ops::Range<usize>>,
    total: usize,
}

fn block_slices(states: &[ParameterBlockState]) -> BlockSlices {
    let mut cursor = 0usize;
    let marginal = cursor..cursor + states[0].beta.len();
    cursor = marginal.end;
    let logslope = cursor..cursor + states[1].beta.len();
    cursor = logslope.end;
    let h = if states.len() > 2 {
        let range = cursor..cursor + states[2].beta.len();
        cursor = range.end;
        Some(range)
    } else {
        None
    };
    let w = if states.len() > 3 {
        let range = cursor..cursor + states[3].beta.len();
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

    fn root_solve_row(
        &self,
        marginal_eta: f64,
        slope: f64,
        h_nodes: &Array1<f64>,
        h_node_design: Option<&Array2<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ImplicitRowDerivatives, String> {
        let ph = h_node_design.map(|d| d.ncols()).unwrap_or(0);
        let pw = beta_w.map(|b| b.len()).unwrap_or(0);
        let target = normal_cdf(marginal_eta);
        let mut intercept = marginal_eta * (1.0 + slope * slope).sqrt();
        for _ in 0..20 {
            let v = h_nodes.mapv(|h| intercept + slope * h);
            let (t, t1, _) = self.link_terms(&v, beta_w)?;
            let f = self
                .quadrature_weights
                .iter()
                .zip(t.iter())
                .map(|(&w, &tt)| w * normal_cdf(tt))
                .sum::<f64>()
                - target;
            let fp = self
                .quadrature_weights
                .iter()
                .zip(t.iter().zip(t1.iter()))
                .map(|(&w, (&tt, &tt1))| w * normal_pdf(tt) * tt1)
                .sum::<f64>();
            if !fp.is_finite() || fp.abs() < 1e-12 {
                break;
            }
            let step = f / fp;
            intercept -= step;
            if step.abs() < 1e-10 {
                break;
            }
        }

        let v = h_nodes.mapv(|h| intercept + slope * h);
        let (t, t1, t2) = self.link_terms(&v, beta_w)?;
        let phi_t = t.mapv(normal_pdf);
        let c1 = &phi_t * &t1;
        let c2 = Array1::from_iter(
            phi_t
                .iter()
                .zip(t.iter().zip(t1.iter().zip(t2.iter())))
                .map(|(&phi, (&tt, (&tt1, &tt2)))| phi * (tt2 - tt * tt1 * tt1)),
        );
        let ma = self.quadrature_weights.dot(&c1);
        let mb = self
            .quadrature_weights
            .iter()
            .zip(h_nodes.iter().zip(c1.iter()))
            .map(|(&w, (&h, &c))| w * h * c)
            .sum::<f64>();
        let maa = self.quadrature_weights.dot(&c2);
        let mab = self
            .quadrature_weights
            .iter()
            .zip(h_nodes.iter().zip(c2.iter()))
            .map(|(&w, (&h, &c))| w * h * c)
            .sum::<f64>();
        let mbb = self
            .quadrature_weights
            .iter()
            .zip(h_nodes.iter().zip(c2.iter()))
            .map(|(&w, (&h, &c))| w * h * h * c)
            .sum::<f64>();
        let a_q = normal_pdf(marginal_eta) / ma.max(1e-12);
        let a_b = -mb / ma.max(1e-12);
        let a_qq = -(maa * a_q * a_q + marginal_eta * normal_pdf(marginal_eta)) / ma.max(1e-12);
        let a_qb = -(a_q * (maa * a_b + mab)) / ma.max(1e-12);
        let a_bb = -(maa * a_b * a_b + 2.0 * mab * a_b + mbb) / ma.max(1e-12);

        let (a_h, a_qh, a_bh, a_hh) = if let Some(h_design) = h_node_design {
            let mut mh = Array1::<f64>::zeros(ph);
            let mut mah = Array1::<f64>::zeros(ph);
            let mut mbh = Array1::<f64>::zeros(ph);
            let mut mhh = Array2::<f64>::zeros((ph, ph));
            for m in 0..ph {
                for k in 0..h_nodes.len() {
                    let bhm = h_design[[k, m]];
                    let w = self.quadrature_weights[k];
                    mh[m] += w * slope * bhm * c1[k];
                    mah[m] += w * slope * bhm * c2[k];
                    mbh[m] += w * (bhm * c1[k] + slope * h_nodes[k] * bhm * c2[k]);
                    for n in 0..ph {
                        mhh[[m, n]] += w * slope * slope * bhm * h_design[[k, n]] * c2[k];
                    }
                }
            }
            let ah = mh.mapv(|v| -v / ma.max(1e-12));
            let aqh =
                Array1::from_iter((0..ph).map(|m| -(a_q * (maa * ah[m] + mah[m])) / ma.max(1e-12)));
            let abh = Array1::from_iter((0..ph).map(|m| {
                -(maa * a_b * ah[m] + mab * ah[m] + mah[m] * a_b + mbh[m]) / ma.max(1e-12)
            }));
            let mut ahh = Array2::<f64>::zeros((ph, ph));
            for m in 0..ph {
                for n in 0..ph {
                    ahh[[m, n]] =
                        -(maa * ah[m] * ah[n] + mah[m] * ah[n] + mah[n] * ah[m] + mhh[[m, n]])
                            / ma.max(1e-12);
                }
            }
            (Some(ah), Some(aqh), Some(abh), Some(ahh))
        } else {
            (None, None, None, None)
        };

        let (a_w, a_qw, a_bw, a_hw, a_ww) = if let (Some(_), Some(runtime)) =
            (beta_w, &self.link_dev)
        {
            let basis = runtime.design(&v)?;
            let basis_d1 = runtime.first_derivative_design(&v)?;
            let mut mw = Array1::<f64>::zeros(pw);
            let mut maw = Array1::<f64>::zeros(pw);
            let mut mbw = Array1::<f64>::zeros(pw);
            let mut mww = Array2::<f64>::zeros((pw, pw));
            let mut mhw = if ph > 0 {
                Some(Array2::<f64>::zeros((ph, pw)))
            } else {
                None
            };
            for r in 0..pw {
                for k in 0..v.len() {
                    let wr = basis[[k, r]];
                    let wr1 = basis_d1[[k, r]];
                    let common = phi_t[k] * (wr1 - t[k] * t1[k] * wr);
                    let wq = self.quadrature_weights[k];
                    mw[r] += wq * phi_t[k] * wr;
                    maw[r] += wq * common;
                    mbw[r] += wq * h_nodes[k] * common;
                    for s in 0..pw {
                        mww[[r, s]] += wq * (-t[k] * phi_t[k] * wr * basis[[k, s]]);
                    }
                    if let (Some(h_design), Some(ref mut mhw_mat)) = (h_node_design, mhw.as_mut()) {
                        for m in 0..ph {
                            mhw_mat[[m, r]] += wq * slope * h_design[[k, m]] * common;
                        }
                    }
                }
            }
            let aw = mw.mapv(|v| -v / ma.max(1e-12));
            let aqw =
                Array1::from_iter((0..pw).map(|r| -(a_q * (maa * aw[r] + maw[r])) / ma.max(1e-12)));
            let abw = Array1::from_iter((0..pw).map(|r| {
                -(maa * a_b * aw[r] + mab * aw[r] + maw[r] * a_b + mbw[r]) / ma.max(1e-12)
            }));
            let mut aww = Array2::<f64>::zeros((pw, pw));
            for r in 0..pw {
                for s in 0..pw {
                    aww[[r, s]] =
                        -(maa * aw[r] * aw[s] + maw[r] * aw[s] + maw[s] * aw[r] + mww[[r, s]])
                            / ma.max(1e-12);
                }
            }
            let ahw = if let (Some(ah), Some(mah), Some(mhw_mat)) = (&a_h, &a_qh, mhw) {
                let _ = mah;
                let mut out = Array2::<f64>::zeros((ph, pw));
                for m in 0..ph {
                    for r in 0..pw {
                        out[[m, r]] = -(maa * ah[m] * aw[r]
                            + mah[m] * aw[r]
                            + maw[r] * ah[m]
                            + mhw_mat[[m, r]])
                            / ma.max(1e-12);
                    }
                }
                Some(out)
            } else {
                None
            };
            (Some(aw), Some(aqw), Some(abw), ahw, Some(aww))
        } else {
            (None, None, None, None, None)
        };

        Ok(ImplicitRowDerivatives {
            intercept,
            a_q,
            a_b,
            a_qq,
            a_qb,
            a_bb,
            a_h,
            a_qh,
            a_bh,
            a_hh,
            a_w,
            a_qw,
            a_bw,
            a_hw,
            a_ww,
        })
    }

    fn joint_gradient_hessian(
        &self,
        block_states: &[ParameterBlockState],
        need_hessian: bool,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String> {
        let slices = block_slices(block_states);
        let n = self.y.len();
        let mut gradient = Array1::<f64>::zeros(slices.total);
        let mut hessian = need_hessian.then(|| Array2::<f64>::zeros((slices.total, slices.total)));
        let prev_eta = &block_states[0].eta;
        let logslope_eta = &block_states[1].eta;
        let (h_nodes, h_node_design) = self.quadrature_h(block_states)?;
        let score_warp_obs = self.score_warp_obs(block_states)?;
        let beta_w = if slices.w.is_some() {
            Some(&block_states[block_states.len() - 1].beta)
        } else {
            None
        };
        let mut ll = 0.0;
        for i in 0..n {
            let marginal_eta = prev_eta[i];
            let g = logslope_eta[i];
            let slope = g.exp();
            let warped_obs = if let Some((_, ref dev_obs)) = score_warp_obs {
                self.z[i] + dev_obs[i]
            } else {
                self.z[i]
            };
            let (
                eta0,
                eta_q,
                eta_g,
                eta_h,
                eta_w,
                eta_qq,
                eta_qg,
                eta_gg,
                eta_qh,
                eta_gh,
                eta_hh,
                eta_qw,
                eta_gw,
                eta_hw,
                eta_ww,
            ) = if !self.flex_active() {
                let c = (1.0 + slope * slope).sqrt();
                (
                    marginal_eta * c + slope * warped_obs,
                    c,
                    marginal_eta * (slope * slope / c) + slope * warped_obs,
                    None,
                    None,
                    0.0,
                    slope * slope / c,
                    marginal_eta * slope * slope * (2.0 + slope * slope) / c.powi(3)
                        + slope * warped_obs,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            } else {
                let implicit = self.root_solve_row(
                    marginal_eta,
                    slope,
                    &h_nodes,
                    h_node_design.as_ref(),
                    beta_w,
                )?;
                let eta_h = implicit.a_h.as_ref().map(|a| {
                    let row = score_warp_obs
                        .as_ref()
                        .expect("score_warp_obs exists when h active")
                        .0
                        .row(i)
                        .to_owned();
                    a + &(row * slope)
                });
                let eta_gh = implicit.a_bh.as_ref().map(|v| {
                    let row = score_warp_obs
                        .as_ref()
                        .expect("score_warp_obs exists when h active")
                        .0
                        .row(i)
                        .to_owned();
                    (v + &row) * slope
                });
                (
                    implicit.intercept + slope * warped_obs,
                    implicit.a_q,
                    slope * (implicit.a_b + warped_obs),
                    eta_h,
                    implicit.a_w.clone(),
                    implicit.a_qq,
                    slope * implicit.a_qb,
                    slope * (implicit.a_b + warped_obs) + slope * slope * implicit.a_bb,
                    implicit.a_qh.clone(),
                    eta_gh,
                    implicit.a_hh.clone(),
                    implicit.a_qw.clone(),
                    implicit.a_bw.as_ref().map(|v| v * slope),
                    implicit.a_hw.clone(),
                    implicit.a_ww.clone(),
                )
            };

            let (q, q1, q2, basis_w, basis_w_d1) =
                if let (Some(runtime), Some(beta), Some(eta_w_vec)) =
                    (&self.link_dev, beta_w, eta_w.as_ref())
                {
                    let eta_arr = Array1::from_vec(vec![eta0]);
                    let basis = runtime.design(&eta_arr)?;
                    let d1 = runtime.first_derivative_design(&eta_arr)?;
                    let d2 = runtime.second_derivative_design(&eta_arr)?;
                    let q = eta0 + basis.row(0).dot(beta);
                    let q1 = 1.0 + d1.row(0).dot(beta);
                    let q2 = d2.row(0).dot(beta);
                    let _ = eta_w_vec;
                    (
                        q,
                        q1,
                        q2,
                        Some(basis.row(0).to_owned()),
                        Some(d1.row(0).to_owned()),
                    )
                } else {
                    (eta0, 1.0, 0.0, None, None)
                };

            let mu = normal_cdf(q).clamp(1e-12, 1.0 - 1e-12);
            ll += self.weights[i] * (self.y[i] * mu.ln() + (1.0 - self.y[i]) * (1.0 - mu).ln());
            let (m1, m2) = probit_neglog_derivatives(self.y[i], self.weights[i], q);

            let mut q_beta = Array1::<f64>::zeros(slices.total);
            q_beta
                .slice_mut(s![slices.marginal.clone()])
                .fill(q1 * eta_q);
            q_beta
                .slice_mut(s![slices.logslope.clone()])
                .fill(q1 * eta_g);
            if let Some(h_range) = slices.h.clone() {
                if let Some(eta_h_vec) = eta_h.as_ref() {
                    q_beta.slice_mut(s![h_range]).assign(&(eta_h_vec * q1));
                }
            }
            if let Some(w_range) = slices.w.clone() {
                if let (Some(eta_w_vec), Some(bw)) = (eta_w.as_ref(), basis_w.as_ref()) {
                    q_beta.slice_mut(s![w_range]).assign(&(eta_w_vec * q1 + bw));
                }
            }
            gradient -= &(q_beta.mapv(|v| m1 * v));

            if let Some(ref mut hmat) = hessian {
                let outer = q_beta
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&q_beta.view().insert_axis(Axis(0)));
                hmat.scaled_add(m2, &outer);

                let pp = Array2::from_elem(
                    (slices.marginal.len(), slices.marginal.len()),
                    q1 * eta_qq + q2 * eta_q * eta_q,
                );
                hmat.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
                    .scaled_add(m1, &pp);
                let pg = Array2::from_elem(
                    (slices.marginal.len(), slices.logslope.len()),
                    q1 * eta_qg + q2 * eta_q * eta_g,
                );
                hmat.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
                    .scaled_add(m1, &pg);
                hmat.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
                    .scaled_add(m1, &pg.t().to_owned());
                let gg = Array2::from_elem(
                    (slices.logslope.len(), slices.logslope.len()),
                    q1 * eta_gg + q2 * eta_g * eta_g,
                );
                hmat.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
                    .scaled_add(m1, &gg);

                if let Some(h_range) = slices.h.clone() {
                    if let (Some(eta_h_vec), Some(eta_qh_vec), Some(eta_gh_vec)) =
                        (eta_h.as_ref(), eta_qh.as_ref(), eta_gh.as_ref())
                    {
                        let qh = Array2::from_shape_fn(
                            (slices.marginal.len(), h_range.len()),
                            |(_, j)| q1 * eta_qh_vec[j] + q2 * eta_q * eta_h_vec[j],
                        );
                        hmat.slice_mut(s![slices.marginal.clone(), h_range.clone()])
                            .scaled_add(m1, &qh);
                        hmat.slice_mut(s![h_range.clone(), slices.marginal.clone()])
                            .scaled_add(m1, &qh.t().to_owned());
                        let gh = Array2::from_shape_fn(
                            (slices.logslope.len(), h_range.len()),
                            |(_, j)| q1 * eta_gh_vec[j] + q2 * eta_g * eta_h_vec[j],
                        );
                        hmat.slice_mut(s![slices.logslope.clone(), h_range.clone()])
                            .scaled_add(m1, &gh);
                        hmat.slice_mut(s![h_range.clone(), slices.logslope.clone()])
                            .scaled_add(m1, &gh.t().to_owned());
                        if let Some(eta_hh_mat) = eta_hh.as_ref() {
                            let mut hh = eta_hh_mat.clone();
                            hh.mapv_inplace(|v| v * q1);
                            hh.scaled_add(
                                q2,
                                &eta_h_vec
                                    .view()
                                    .insert_axis(Axis(1))
                                    .dot(&eta_h_vec.view().insert_axis(Axis(0))),
                            );
                            hmat.slice_mut(s![h_range.clone(), h_range.clone()])
                                .scaled_add(m1, &hh);
                        }
                    }
                }
                if let Some(w_range) = slices.w.clone() {
                    if let (Some(eta_w_vec), Some(bw_d1)) = (eta_w.as_ref(), basis_w_d1.as_ref()) {
                        if let Some(eta_qw_vec) = eta_qw.as_ref() {
                            let qw = Array2::from_shape_fn(
                                (slices.marginal.len(), w_range.len()),
                                |(_, j)| {
                                    q1 * eta_qw_vec[j]
                                        + q2 * eta_q * eta_w_vec[j]
                                        + bw_d1[j] * eta_q
                                },
                            );
                            hmat.slice_mut(s![slices.marginal.clone(), w_range.clone()])
                                .scaled_add(m1, &qw);
                            hmat.slice_mut(s![w_range.clone(), slices.marginal.clone()])
                                .scaled_add(m1, &qw.t().to_owned());
                        }
                        if let Some(eta_gw_vec) = eta_gw.as_ref() {
                            let gw = Array2::from_shape_fn(
                                (slices.logslope.len(), w_range.len()),
                                |(_, j)| {
                                    q1 * eta_gw_vec[j]
                                        + q2 * eta_g * eta_w_vec[j]
                                        + bw_d1[j] * eta_g
                                },
                            );
                            hmat.slice_mut(s![slices.logslope.clone(), w_range.clone()])
                                .scaled_add(m1, &gw);
                            hmat.slice_mut(s![w_range.clone(), slices.logslope.clone()])
                                .scaled_add(m1, &gw.t().to_owned());
                        }
                        if let Some(h_range) = slices.h.clone() {
                            if let (Some(eta_h_vec), Some(eta_hw_mat)) =
                                (eta_h.as_ref(), eta_hw.as_ref())
                            {
                                let hw = Array2::from_shape_fn(
                                    (h_range.len(), w_range.len()),
                                    |(r, c)| {
                                        q1 * eta_hw_mat[[r, c]]
                                            + q2 * eta_h_vec[r] * eta_w_vec[c]
                                            + bw_d1[c] * eta_h_vec[r]
                                    },
                                );
                                hmat.slice_mut(s![h_range.clone(), w_range.clone()])
                                    .scaled_add(m1, &hw);
                                hmat.slice_mut(s![w_range.clone(), h_range.clone()])
                                    .scaled_add(m1, &hw.t().to_owned());
                            }
                        }
                        if let Some(eta_ww_mat) = eta_ww.as_ref() {
                            let mut ww =
                                Array2::from_shape_fn((w_range.len(), w_range.len()), |(r, c)| {
                                    q2 * eta_w_vec[r] * eta_w_vec[c]
                                        + bw_d1[r] * eta_w_vec[c]
                                        + bw_d1[c] * eta_w_vec[r]
                                });
                            ww.scaled_add(q1, eta_ww_mat);
                            hmat.slice_mut(s![w_range.clone(), w_range.clone()])
                                .scaled_add(m1, &ww);
                        }
                    }
                }
            }
        }
        Ok((ll, gradient, hessian))
    }
}

impl CustomFamily for BernoulliMarginalSlopeFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, gradient, hessian) = self.joint_gradient_hessian(block_states, true)?;
        let hessian = hessian.ok_or_else(|| "joint hessian unavailable".to_string())?;
        let slices = block_slices(block_states);
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

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        Ok(match block_idx {
            2 => self.score_warp_constraints.clone(),
            3 => self.link_dev_constraints.clone(),
            _ => None,
        })
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
        design: DesignMatrix::Dense(design.design.clone()),
        offset: Array1::from_elem(design.design.nrows(), baseline),
        penalties: design.penalties.clone(),
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
    let baseline = pooled_probit_baseline(&spec.y, &spec.z)?;
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
    let score_warp_prepared = spec
        .score_warp
        .as_ref()
        .map(|cfg| build_deviation_block_from_seed(&spec.z, cfg))
        .transpose()?;

    let (quad_nodes, quad_weights) = normal_expectation_nodes(spec.quadrature_points);
    let rigid_family = BernoulliMarginalSlopeFamily {
        y: spec.y.clone(),
        weights: spec.weights.clone(),
        z: spec.z.clone(),
        quadrature_nodes: quad_nodes.clone(),
        quadrature_weights: quad_weights.clone(),
        score_warp: None,
        score_warp_obs_design: None,
        link_dev: None,
        score_warp_constraints: None,
        link_dev_constraints: None,
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
    let hints = RefCell::new(ThetaHints::default());
    let y = spec.y.clone();
    let weights = spec.weights.clone();
    let z = spec.z.clone();
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let score_warp_obs_design = score_warp_prepared
        .as_ref()
        .and_then(|p| match &p.block.design {
            DesignMatrix::Dense(x) => Some(x.clone()),
            DesignMatrix::Sparse(_) => None,
        });
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());
    let score_warp_constraints = score_warp_prepared.as_ref().map(|p| p.constraints.clone());
    let link_dev_constraints = link_dev_prepared.as_ref().map(|p| p.constraints.clone());

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

    let make_family = || BernoulliMarginalSlopeFamily {
        y: y.clone(),
        weights: weights.clone(),
        z: z.clone(),
        quadrature_nodes: quad_nodes.clone(),
        quadrature_weights: quad_weights.clone(),
        score_warp: score_warp_runtime.clone(),
        score_warp_obs_design: score_warp_obs_design.clone(),
        link_dev: link_dev_runtime.clone(),
        score_warp_constraints: score_warp_constraints.clone(),
        link_dev_constraints: link_dev_constraints.clone(),
    };

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let all_dims = setup.log_kappa_dims_per_term();
    let rho_dim = setup.rho_dim();

    let solved = optimize_two_block_spatial_length_scale_exact_joint(
        data,
        &marginalspec_boot,
        &logslopespec_boot,
        kappa_options,
        &setup,
        true,
        false,
        |rho, _, _, marginal_design, logslope_design| {
            let blocks = build_blocks(rho, marginal_design, logslope_design)?;
            let family = make_family();
            let fit = inner_fit(&family, &blocks, options)?;
            Ok(fit_score(&fit))
        },
        |rho, _, _, marginal_design, logslope_design| {
            let blocks = build_blocks(rho, marginal_design, logslope_design)?;
            let family = make_family();
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = fit.block_states.first() {
                hints_mut.marginal_beta = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(1) {
                hints_mut.logslope_beta = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(2) {
                hints_mut.score_warp_beta = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(3) {
                hints_mut.link_dev_beta = Some(block.beta.clone());
            }
            Ok(fit)
        },
        |rho, marginal_resolved, logslope_resolved, _, _, need_hessian| {
            let _ = need_hessian;
            let objective = {
                let marginal_design = build_term_collection_design(data, marginal_resolved)
                    .map_err(|e| e.to_string())?;
                let logslope_design = build_term_collection_design(data, logslope_resolved)
                    .map_err(|e| e.to_string())?;
                let blocks = build_blocks(rho, &marginal_design, &logslope_design)?;
                let family = make_family();
                fit_score(&inner_fit(&family, &blocks, options)?)
            };
            let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
                marginal_resolved,
                &marginal_terms,
                kappa_options,
            );
            let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
                logslope_resolved,
                &logslope_terms,
                kappa_options,
            );
            let mut theta = Array1::<f64>::zeros(rho_dim + all_dims.iter().sum::<usize>());
            theta.slice_mut(s![..rho.len()]).assign(rho);
            let mut cursor = rho.len();
            theta
                .slice_mut(s![cursor..cursor + marginal_kappa.as_array().len()])
                .assign(marginal_kappa.as_array());
            cursor += marginal_kappa.as_array().len();
            theta
                .slice_mut(s![cursor..cursor + logslope_kappa.as_array().len()])
                .assign(logslope_kappa.as_array());

            let mut gradient = Array1::<f64>::zeros(theta.len());
            for j in 0..theta.len() {
                let h = 1e-3 * (1.0 + theta[j].abs());
                let mut plus = theta.clone();
                plus[j] += h;
                let mut minus = theta.clone();
                minus[j] -= h;
                let plus_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
                    &plus,
                    rho_dim,
                    all_dims.clone(),
                );
                let minus_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
                    &minus,
                    rho_dim,
                    all_dims.clone(),
                );
                let (plus_marginal, plus_logslope) = plus_kappa.split_at(marginal_terms.len());
                let (minus_marginal, minus_logslope) = minus_kappa.split_at(marginal_terms.len());
                let plus_marginal_spec = plus_marginal
                    .apply_tospec(&marginalspec_boot, &marginal_terms)
                    .map_err(|e| e.to_string())?;
                let plus_logslope_spec = plus_logslope
                    .apply_tospec(&logslopespec_boot, &logslope_terms)
                    .map_err(|e| e.to_string())?;
                let minus_marginal_spec = minus_marginal
                    .apply_tospec(&marginalspec_boot, &marginal_terms)
                    .map_err(|e| e.to_string())?;
                let minus_logslope_spec = minus_logslope
                    .apply_tospec(&logslopespec_boot, &logslope_terms)
                    .map_err(|e| e.to_string())?;
                let plus_marginal_design = build_term_collection_design(data, &plus_marginal_spec)
                    .map_err(|e| e.to_string())?;
                let plus_logslope_design = build_term_collection_design(data, &plus_logslope_spec)
                    .map_err(|e| e.to_string())?;
                let minus_marginal_design =
                    build_term_collection_design(data, &minus_marginal_spec)
                        .map_err(|e| e.to_string())?;
                let minus_logslope_design =
                    build_term_collection_design(data, &minus_logslope_spec)
                        .map_err(|e| e.to_string())?;
                let plus_fit = {
                    let blocks = build_blocks(
                        &plus.slice(s![..rho_dim]).to_owned(),
                        &plus_marginal_design,
                        &plus_logslope_design,
                    )?;
                    inner_fit(&make_family(), &blocks, options)?
                };
                let minus_fit = {
                    let blocks = build_blocks(
                        &minus.slice(s![..rho_dim]).to_owned(),
                        &minus_marginal_design,
                        &minus_logslope_design,
                    )?;
                    inner_fit(&make_family(), &blocks, options)?
                };
                gradient[j] = (fit_score(&plus_fit) - fit_score(&minus_fit)) / (2.0 * h);
            }
            Ok((objective, gradient, None))
        },
    )?;

    Ok(BernoulliMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: solved.resolved_meanspec,
        logslopespec_resolved: solved.resolved_noisespec,
        marginal_design: solved.mean_design,
        logslope_design: solved.noise_design,
        baseline_marginal: baseline.0,
        baseline_logslope: baseline.1,
        score_warp_runtime,
        link_dev_runtime,
    })
}
