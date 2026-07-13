//! Grouped LAML smoothing selection for the support-sparse TopK term.
//!
//! A separate smoothing coordinate for every atom makes the outer problem grow
//! as `O(K)` even when thousands of atoms share one declared function space.
//! The sparse lane instead shares a variance component by exact
//! `(basis kind, latent dimension)` family. The generic optimizer therefore
//! sees the number of heterogeneous families, while the inner model still has
//! distinct decoder functions and coordinates for every occupied atom.

use std::collections::BTreeMap;

use gam_problem::{DeclaredHessianForm, Derivative, EstimationError, HessianValue, OuterEval};
use gam_solve::rho_optimizer::{
    OuterCapability, OuterCriterionCertificate, OuterObjective, OuterProblem, SeedOutcome,
};
use ndarray::{Array1, Array2, ArrayView1};

use super::*;

const SUPPORT_LAML_CONTEXT: &str = "support-sparse TopK grouped LAML";
const SUPPORT_LAML_TRACE_PROBES: usize = 16;
const SUPPORT_LAML_CG_REL_TOL: f64 = 1.0e-8;

fn outer_error(message: impl Into<String>) -> EstimationError {
    EstimationError::RemlOptimizationFailed(message.into())
}

#[derive(Clone, Debug)]
pub struct SaeSupportSmoothingLayout {
    pub atom_group: Vec<usize>,
    pub group_keys: Vec<String>,
}

impl SaeSupportSmoothingLayout {
    pub fn from_term(term: &SaeSupportSparseTerm) -> Self {
        let mut keys = BTreeMap::<String, ()>::new();
        let atom_keys = term
            .atoms
            .iter()
            .map(|atom| {
                format!(
                    "{}:d{}",
                    sae_atom_basis_kind_name(&atom.basis_kind),
                    atom.latent_dim
                )
            })
            .collect::<Vec<_>>();
        for key in &atom_keys {
            keys.insert(key.clone(), ());
        }
        let group_keys = keys.into_keys().collect::<Vec<_>>();
        let index = group_keys
            .iter()
            .enumerate()
            .map(|(group, key)| (key.clone(), group))
            .collect::<BTreeMap<_, _>>();
        let atom_group = atom_keys.iter().map(|key| index[key]).collect();
        Self {
            atom_group,
            group_keys,
        }
    }

    pub fn expand(&self, rho: &Array1<f64>) -> Result<Vec<f64>, String> {
        if rho.len() != self.group_keys.len() {
            return Err(format!(
                "SaeSupportSmoothingLayout::expand: rho length {} != groups {}",
                rho.len(),
                self.group_keys.len()
            ));
        }
        let lambdas =
            gam_problem::checked_exp_log_strengths(rho.iter().copied()).map_err(|error| {
                format!("SaeSupportSmoothingLayout::expand: invalid log strength: {error}")
            })?;
        Ok(self
            .atom_group
            .iter()
            .map(|&group| lambdas[group])
            .collect())
    }
}

pub struct SaeSupportOuterRequest {
    pub term: SaeSupportSparseTerm,
    pub target: Array2<f64>,
    pub initial_smoothness: f64,
    pub ard_precisions: Vec<Vec<f64>>,
    pub max_outer_iter: usize,
    pub max_inner_iter: usize,
    pub inner_tolerance: f64,
    pub trust_radius: f64,
    pub random_state: u64,
}

pub struct SaeSupportOuterReport {
    pub term: SaeSupportSparseTerm,
    pub smoothing_layout: SaeSupportSmoothingLayout,
    pub log_lambda_groups: Array1<f64>,
    pub lambda_smooth: Vec<f64>,
    pub ard_precisions: Vec<Vec<f64>>,
    pub criterion: f64,
    pub fixed_point: SaeSupportFixedPointReport,
    pub outer_iterations: usize,
    pub outer_certificate: OuterCriterionCertificate,
}

struct PenaltySpectrum {
    rank_by_group: Vec<usize>,
    log_pdet_base_by_group: Vec<f64>,
    total_rank: usize,
}

struct SupportOuterEvaluation {
    cost: f64,
    gradient: Array1<f64>,
    lambda_smooth: Vec<f64>,
    fixed_point: SaeSupportFixedPointReport,
}

struct SaeSupportOuterObjective {
    term: SaeSupportSparseTerm,
    initial_term: SaeSupportSparseTerm,
    target: Array2<f64>,
    layout: SaeSupportSmoothingLayout,
    spectrum: PenaltySpectrum,
    ard_precisions: Vec<Vec<f64>>,
    max_inner_iter: usize,
    inner_tolerance: f64,
    trust_radius: f64,
    random_state: u64,
    last_evaluation: Option<SupportOuterEvaluation>,
}

fn penalty_spectrum(
    term: &SaeSupportSparseTerm,
    layout: &SaeSupportSmoothingLayout,
) -> Result<PenaltySpectrum, String> {
    let groups = layout.group_keys.len();
    let mut rank_by_group = vec![0usize; groups];
    let mut log_pdet_base_by_group = vec![0.0; groups];
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let symmetric = (&atom.smooth_penalty + &atom.smooth_penalty.t()) * 0.5;
        let (values, _) = symmetric
            .eigh(Side::Lower)
            .map_err(|error| format!("support smooth-penalty eigendecomposition: {error}"))?;
        let scale = values.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let tolerance = f64::EPSILON.sqrt() * scale * atom.basis_size().max(1) as f64;
        if values.iter().any(|value| *value < -tolerance) {
            return Err(format!(
                "support smooth penalty for atom {atom_idx} is not positive semidefinite"
            ));
        }
        let group = layout.atom_group[atom_idx];
        for value in values.iter().copied().filter(|value| *value > tolerance) {
            rank_by_group[group] = rank_by_group[group]
                .checked_add(term.output_dim())
                .ok_or_else(|| "support penalty rank overflow".to_string())?;
            log_pdet_base_by_group[group] += term.output_dim() as f64 * value.ln();
        }
    }
    let total_rank = rank_by_group.iter().sum();
    Ok(PenaltySpectrum {
        rank_by_group,
        log_pdet_base_by_group,
        total_rank,
    })
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl SaeSupportOuterObjective {
    fn beta_layout(&self) -> Result<(Vec<usize>, usize), EstimationError> {
        self.term.beta_layout().map_err(outer_error)
    }

    fn penalty_apply_group(
        &self,
        group: usize,
        lambda_smooth: &[f64],
        vector: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let (offsets, beta_dim) = self.beta_layout()?;
        if vector.len() != beta_dim {
            return Err(outer_error("support penalty vector width mismatch"));
        }
        let mut out = Array1::<f64>::zeros(beta_dim);
        for atom in 0..self.term.k_atoms() {
            if self.layout.atom_group[atom] != group {
                continue;
            }
            let m = self.term.atoms[atom].basis_size();
            let offset = offsets[atom];
            let lambda = lambda_smooth[atom];
            for left in 0..m {
                for right in 0..m {
                    let weight = lambda * self.term.atoms[atom].smooth_penalty[[left, right]];
                    for channel in 0..self.term.output_dim() {
                        out[offset + left * self.term.output_dim() + channel] +=
                            weight * vector[offset + right * self.term.output_dim() + channel];
                    }
                }
            }
        }
        Ok(out)
    }

    fn penalty_energy_by_group(&self, lambda_smooth: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.layout.group_keys.len()];
        for atom in 0..self.term.k_atoms() {
            let sb = self.term.atoms[atom]
                .smooth_penalty
                .dot(&self.term.atoms[atom].decoder_coefficients);
            let energy = self.term.atoms[atom]
                .decoder_coefficients
                .iter()
                .zip(sb.iter())
                .map(|(left, right)| left * right)
                .sum::<f64>();
            out[self.layout.atom_group[atom]] += lambda_smooth[atom] * energy;
        }
        out
    }

    fn trace_by_group(
        &self,
        system: &ArrowSchurSystem,
        cache: &ArrowFactorCache,
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, EstimationError> {
        let (_, beta_dim) = self.beta_layout()?;
        let latent_dim = cache.delta_t_len();
        let rhs_t = Array1::<f64>::zeros(latent_dim);
        let mut traces = vec![0.0; self.layout.group_keys.len()];
        let max_iters = beta_dim.saturating_mul(2).clamp(128, 4096);
        for probe in 0..SUPPORT_LAML_TRACE_PROBES {
            let mut z = Array1::<f64>::zeros(beta_dim);
            for index in 0..beta_dim {
                let hash = splitmix64(
                    self.random_state
                        ^ (probe as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ index as u64,
                );
                z[index] = if hash >> 63 == 0 { -1.0 } else { 1.0 };
            }
            for (group, trace) in traces.iter_mut().enumerate() {
                let rhs_beta = self.penalty_apply_group(group, lambda_smooth, z.view())?;
                let (_, solved_beta) = matrix_free_arrow_inverse_apply(
                    system,
                    cache,
                    rhs_t.view(),
                    rhs_beta.view(),
                    SUPPORT_LAML_CG_REL_TOL,
                    max_iters,
                )
                .map_err(|error| outer_error(format!("support LAML trace solve: {error}")))?;
                *trace += z.dot(&solved_beta) / SUPPORT_LAML_TRACE_PROBES as f64;
            }
        }
        Ok(traces)
    }

    fn evaluate(&mut self, rho: &Array1<f64>) -> Result<SupportOuterEvaluation, EstimationError> {
        let lambda_smooth = self.layout.expand(rho).map_err(outer_error)?;
        let fixed_point = self
            .term
            .solve_fixed_point(
                self.target.view(),
                &lambda_smooth,
                &self.ard_precisions,
                self.max_inner_iter,
                self.inner_tolerance,
                self.trust_radius,
            )
            .map_err(outer_error)?;
        let system = self
            .term
            .assemble_arrow_schur(self.target.view(), &lambda_smooth, &self.ard_precisions)
            .map_err(outer_error)?;
        let options = ArrowSolveOptions::inexact_pcg().with_positive_definite_evidence();
        let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .map_err(|error| outer_error(format!("support LAML Arrow factorization: {error}")))?;
        let joint_logdet = cache
            .arrow_log_det()
            .ok_or_else(|| outer_error("support LAML factor cache has no joint log determinant"))?;
        let residual = self
            .term
            .raw_residual(self.target.view())
            .map_err(outer_error)?;
        let rss = residual.iter().map(|value| value * value).sum::<f64>();
        if !(rss.is_finite() && rss > 0.0) {
            return Err(outer_error(format!(
                "support LAML requires positive finite raw residual deviance; got {rss}"
            )));
        }
        let (_, beta_dim) = self.beta_layout()?;
        let beta_nullity = beta_dim
            .checked_sub(self.spectrum.total_rank)
            .ok_or_else(|| outer_error("support smooth penalty rank exceeds beta dimension"))?;
        let data_dim = self
            .term
            .n_obs()
            .checked_mul(self.term.output_dim())
            .ok_or_else(|| outer_error("support LAML data dimension overflow"))?;
        if data_dim <= beta_nullity {
            return Err(outer_error(format!(
                "support LAML requires more response cells than unpenalized decoder coefficients; got {data_dim} <= {beta_nullity}"
            )));
        }
        let residual_df = (data_dim - beta_nullity) as f64;
        let mut penalty_logdet = 0.0;
        for group in 0..self.layout.group_keys.len() {
            penalty_logdet += self.spectrum.log_pdet_base_by_group[group]
                + self.spectrum.rank_by_group[group] as f64 * rho[group];
        }
        let cost = 0.5
            * (joint_logdet - penalty_logdet
                + residual_df * (1.0 + (std::f64::consts::TAU * rss / residual_df).ln()));
        let traces = self.trace_by_group(&system, &cache, &lambda_smooth)?;
        let energy = self.penalty_energy_by_group(&lambda_smooth);
        let mut gradient = Array1::<f64>::zeros(self.layout.group_keys.len());
        for group in 0..gradient.len() {
            gradient[group] = 0.5
                * (traces[group] - self.spectrum.rank_by_group[group] as f64
                    + residual_df * energy[group] / rss);
        }
        if !cost.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(outer_error(
                "support LAML produced a non-finite value or gradient",
            ));
        }
        Ok(SupportOuterEvaluation {
            cost,
            gradient,
            lambda_smooth,
            fixed_point,
        })
    }
}

impl OuterObjective for SaeSupportOuterObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.layout.group_keys.len(),
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: true,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.evaluate(rho).map(|evaluation| evaluation.cost)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let evaluation = self.evaluate(rho)?;
        let out = OuterEval {
            cost: evaluation.cost,
            gradient: evaluation.gradient.clone(),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        };
        self.last_evaluation = Some(evaluation);
        Ok(out)
    }

    fn reset(&mut self) {
        self.term = self.initial_term.clone();
        self.last_evaluation = None;
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(outer_error(
                "support outer seed contains a non-finite value",
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }

    fn allow_continuation_prewarm(&self) -> bool {
        false
    }
}

/// Select topology-grouped smoothing strengths through the shared generic
/// outer optimizer. Only a terminal point with an analytic stationarity
/// certificate and a recurring raw inner fixed point is returned.
pub fn run_sae_support_outer(
    request: SaeSupportOuterRequest,
) -> Result<SaeSupportOuterReport, EstimationError> {
    if !(request.initial_smoothness.is_finite() && request.initial_smoothness > 0.0) {
        return Err(outer_error(format!(
            "support outer initial_smoothness must be finite and positive; got {}",
            request.initial_smoothness
        )));
    }
    let layout = SaeSupportSmoothingLayout::from_term(&request.term);
    if layout.group_keys.is_empty() {
        return Err(outer_error(
            "support outer requires at least one smoothing group",
        ));
    }
    let spectrum = penalty_spectrum(&request.term, &layout).map_err(outer_error)?;
    let initial_term = request.term.clone();
    let mut objective = SaeSupportOuterObjective {
        term: request.term,
        initial_term,
        target: request.target,
        layout: layout.clone(),
        spectrum,
        ard_precisions: request.ard_precisions.clone(),
        max_inner_iter: request.max_inner_iter,
        inner_tolerance: request.inner_tolerance,
        trust_radius: request.trust_radius,
        random_state: request.random_state,
        last_evaluation: None,
    };
    let initial_rho = Array1::from_elem(layout.group_keys.len(), request.initial_smoothness.ln());
    let problem = OuterProblem::new(layout.group_keys.len())
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_prefer_gradient_only(true)
        .with_disable_fixed_point(true)
        .with_continuation_prewarm(false)
        .with_initial_rho(initial_rho)
        .with_max_iter(request.max_outer_iter.max(1));
    let outer = problem.run(&mut objective, SUPPORT_LAML_CONTEXT)?;
    let certificate = outer
        .criterion_certificate
        .clone()
        .filter(OuterCriterionCertificate::certifies)
        .ok_or_else(|| {
            outer_error(format!(
                "support outer returned without an analytic stationarity certificate after {} iterations",
                outer.iterations
            ))
        })?;
    let terminal = objective.evaluate(&outer.rho)?;
    if !terminal.fixed_point.recurred {
        return Err(outer_error(
            "support outer terminal inner state did not recur",
        ));
    }
    Ok(SaeSupportOuterReport {
        term: objective.term,
        smoothing_layout: layout,
        log_lambda_groups: outer.rho,
        lambda_smooth: terminal.lambda_smooth,
        ard_precisions: request.ard_precisions,
        criterion: terminal.cost,
        fixed_point: terminal.fixed_point,
        outer_iterations: outer.iterations,
        outer_certificate: certificate,
    })
}
