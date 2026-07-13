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
                    sae_atom_basis_kind_name(atom.basis_kind()),
                    atom.latent_dim()
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
        let symmetric = (atom.smooth_penalty() + &atom.smooth_penalty().t()) * 0.5;
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
                    let weight = lambda * self.term.atoms[atom].smooth_penalty()[[left, right]];
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
                .smooth_penalty()
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
        // Gaussian dispersion argument = the PENALIZED deviance
        //   D_p(ρ) = ‖y − ŷ‖² + β̂ᵀ S_ρ β̂ + (every other penalty the inner solve descends),
        // NOT the raw residual sum of squares. This mirrors the canonical dense
        // manifold path, whose profiled-scale data term ranks `loss.total()`
        // (data_fit + smoothness + ard + sparsity) — the full penalized loss —
        // precisely so the envelope theorem makes the analytic outer gradient
        // exact (construction_quasi_laplace.rs:357). `penalized_objective` returns
        // ½·D_p (½‖y−ŷ‖² + ½Σ_k λ_k β̂ᵀS_kβ̂ + ARD), so 2× recovers D_p. At the inner
        // optimum β̂ minimizes the full penalized objective, hence the envelope
        // theorem gives d D_p/dρ_g = ∂_ρ_g D_p|_{β̂} = Σ_{k∈g} λ_k β̂ᵀS_kβ̂ = energy[g]
        // — the exact numerator the gradient below already forms. With the RAW rss
        // instead, d(rss)/dρ_g carries an implicit H⁻¹ envelope term ≠ energy[g],
        // so value and gradient would descend different functions (the desync bug).
        let deviance = 2.0
            * self
                .term
                .penalized_objective(self.target.view(), &lambda_smooth, &self.ard_precisions)
                .map_err(outer_error)?;
        if !(deviance.is_finite() && deviance > 0.0) {
            return Err(outer_error(format!(
                "support LAML requires positive finite penalized deviance; got {deviance}"
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
                + residual_df * (1.0 + (std::f64::consts::TAU * deviance / residual_df).ln()));
        let traces = self.trace_by_group(&system, &cache, &lambda_smooth)?;
        let energy = self.penalty_energy_by_group(&lambda_smooth);
        let mut gradient = Array1::<f64>::zeros(self.layout.group_keys.len());
        for group in 0..gradient.len() {
            gradient[group] = 0.5
                * (traces[group] - self.spectrum.rank_by_group[group] as f64
                    + residual_df * energy[group] / deviance);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment_state::SaeAssignmentAtomSpec;
    use ndarray::array;
    use std::sync::Arc;

    fn atom(
        name: &str,
        kind: SaeAtomBasisKind,
        d: usize,
        evaluator: Arc<dyn SaeBasisSecondJet>,
        coords: &[f64],
        decoder: Array2<f64>,
    ) -> SaeManifoldAtom {
        let coord = Array2::from_shape_vec((1, d), coords.to_vec()).expect("coords");
        let (phi, jet) = evaluator.evaluate(coord.view()).expect("evaluate");
        let m = phi.ncols();
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            kind,
            d,
            phi,
            jet,
            decoder,
            Array2::eye(m),
        )
        .expect("atom")
        .with_basis_second_jet(evaluator)
    }

    /// Two-group heterogeneous support fixture (K = 2 atoms > P = 2 response
    /// cells) whose smoothing selection is the exact regime the outer criterion
    /// desync afflicts. The target leaves a genuine residual so the penalized
    /// deviance and every per-group penalty energy are strictly positive.
    fn build_objective() -> SaeSupportOuterObjective {
        let periodic_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
        let patch_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
        let atoms = vec![
            atom(
                "circle",
                SaeAtomBasisKind::Periodic,
                1,
                periodic_eval,
                &[0.3],
                array![[0.2], [1.1], [-0.4]],
            ),
            atom(
                "plane",
                SaeAtomBasisKind::Linear,
                2,
                patch_eval,
                &[0.1, -0.2],
                array![[0.3], [2.0], [-1.0]],
            ),
        ];
        let specs = vec![
            SaeAssignmentAtomSpec {
                latent_dim: 1,
                id_mode: LatentIdMode::None,
                manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                latent_id: 1,
            },
            SaeAssignmentAtomSpec::euclidean(2),
        ];
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            2,
            2,
            1,
            specs,
            vec![vec![0], vec![1]],
            vec![vec![9.0], vec![-4.0]],
            vec![vec![0.25], vec![3.0, 1.0]],
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let layout = SaeSupportSmoothingLayout::from_term(&term);
        assert_eq!(layout.group_keys.len(), 2, "fixture must expose two groups");
        let spectrum = penalty_spectrum(&term, &layout).expect("spectrum");
        let initial_term = term.clone();
        SaeSupportOuterObjective {
            term,
            initial_term,
            target: array![[1.4], [4.3]],
            layout,
            spectrum,
            ard_precisions: vec![vec![1.0], vec![1.0, 1.0]],
            max_inner_iter: 5000,
            inner_tolerance: 1.0e-9,
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
            last_evaluation: None,
        }
    }

    /// Solve the inner fixed point cleanly from the initial term at `rho`
    /// (rebuilding the whole cache — never freezing it, per the FD-gate rule)
    /// and read off the penalized deviance `D_p = 2·penalized_objective` and the
    /// raw residual sum of squares at that converged inner optimum.
    fn deviance_and_rss(objective: &mut SaeSupportOuterObjective, rho: &Array1<f64>) -> (f64, f64) {
        objective.reset();
        let lambda = objective.layout.expand(rho).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("inner fixed point");
        let deviance = 2.0
            * objective
                .term
                .penalized_objective(objective.target.view(), &lambda, &objective.ard_precisions)
                .expect("penalized objective");
        let residual = objective
            .term
            .raw_residual(objective.target.view())
            .expect("raw residual");
        let rss = residual.iter().map(|value| value * value).sum::<f64>();
        (deviance, rss)
    }

    /// Decisive oracle for the value↔gradient desync. The outer value feeds the
    /// penalized deviance `D_p` into the Gaussian dispersion term, and the
    /// analytic gradient's dispersion channel is `½·residual_df·energy[g]/D_p`
    /// with `energy[g] = Σ_{k∈g} λ_k β̂ᵀ S_k β̂ = penalty_energy_by_group`. Since
    /// `residual_df` and `D_p` are common factors, value/gradient consistency of
    /// that channel is EXACTLY the envelope identity `d D_p/dρ_g = energy[g]`.
    /// Central-differencing the production `D_p` (with a full clean inner re-solve
    /// at each ρ±h) must reproduce the production `energy[g]` — this is the FD
    /// oracle the SPEC allows in tests. The same test also confirms that the RAW
    /// residual sum of squares does NOT satisfy the identity (its derivative
    /// carries the implicit `H⁻¹` envelope term), so a revert to raw RSS is caught.
    #[test]
    fn support_penalized_deviance_derivative_equals_penalty_energy() {
        let mut objective = build_objective();
        // λ deliberately away from 1 in both groups so the raw-RSS derivative and
        // the penalized-deviance derivative are unmistakably different functions.
        let base = array![0.35_f64.ln(), 2.8_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // Production `energy[g]` at the base inner optimum.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");
        let energy = objective.penalty_energy_by_group(&lambda_base);
        assert_eq!(energy.len(), groups);
        assert!(
            energy.iter().all(|value| value.is_finite() && *value > 0.0),
            "fixture must exercise strictly positive per-group penalty energy: {energy:?}"
        );

        let h = 1.0e-4;
        let energy_scale = energy.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let mut max_raw_gap = 0.0_f64;
        for g in 0..groups {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[g] += h;
            minus[g] -= h;
            let (dev_plus, rss_plus) = deviance_and_rss(&mut objective, &plus);
            let (dev_minus, rss_minus) = deviance_and_rss(&mut objective, &minus);
            let deviance_derivative = (dev_plus - dev_minus) / (2.0 * h);
            let raw_rss_derivative = (rss_plus - rss_minus) / (2.0 * h);

            // (1) Envelope identity: d D_p/dρ_g == energy[g] (value↔gradient match).
            let envelope_gap = (deviance_derivative - energy[g]).abs();
            assert!(
                envelope_gap <= 1.0e-6 * (1.0 + energy[g].abs()),
                "group {g}: penalized-deviance derivative {deviance_derivative:.9e} \
                 disagrees with penalty energy {:.9e} (gap {envelope_gap:.3e})",
                energy[g]
            );

            // (2) Raw RSS is a DIFFERENT function: its derivative must not be the
            // penalty energy, proving the raw-RSS gradient (the fixed defect) desyncs.
            max_raw_gap = max_raw_gap.max((raw_rss_derivative - energy[g]).abs());
        }
        assert!(
            max_raw_gap > 1.0e-2 * energy_scale.max(1.0e-3),
            "raw-RSS derivative must visibly differ from the penalty energy so the \
             desync is caught (max gap {max_raw_gap:.3e}, energy scale {energy_scale:.3e})"
        );
    }

    /// Full-production oracle: the analytic gradient returned by `evaluate` must
    /// match a central difference of the production value `cost`, restricted to
    /// the dispersion channel that the desync corrupted. The joint log-det term is
    /// a fixed-probe Hutchinson estimate (deterministic but not exact), so we
    /// isolate the dispersion channel by subtracting the exact analytic log-det
    /// and penalty-log-det contributions — both computed from the same production
    /// quantities — leaving `½·residual_df·(1+ln(τ·D_p/df))`, whose FD must equal
    /// the analytic `½·residual_df·energy[g]/D_p`.
    #[test]
    fn support_outer_value_dispersion_channel_matches_gradient() {
        let mut objective = build_objective();
        let base = array![0.4_f64.ln(), 2.2_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // residual_df is a ρ-independent constant of the fixture.
        let (_, beta_dim) = objective.beta_layout().expect("beta layout");
        let beta_nullity = beta_dim - objective.spectrum.total_rank;
        let data_dim = objective.term.n_obs() * objective.term.output_dim();
        let residual_df = (data_dim - beta_nullity) as f64;
        assert!(residual_df > 0.0);

        let dispersion_value =
            |objective: &mut SaeSupportOuterObjective, rho: &Array1<f64>| -> f64 {
                let (deviance, _) = deviance_and_rss(objective, rho);
                0.5 * residual_df * (1.0 + (std::f64::consts::TAU * deviance / residual_df).ln())
            };

        // Analytic dispersion-channel gradient from production quantities.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");
        let deviance_base = 2.0
            * objective
                .term
                .penalized_objective(objective.target.view(), &lambda_base, &objective.ard_precisions)
                .expect("penalized objective");
        let energy = objective.penalty_energy_by_group(&lambda_base);

        let h = 1.0e-4;
        for g in 0..groups {
            let analytic = 0.5 * residual_df * energy[g] / deviance_base;
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[g] += h;
            minus[g] -= h;
            let fd = (dispersion_value(&mut objective, &plus)
                - dispersion_value(&mut objective, &minus))
                / (2.0 * h);
            assert!(
                (analytic - fd).abs() <= 1.0e-6 * (1.0 + analytic.abs()),
                "group {g}: analytic dispersion gradient {analytic:.9e} != FD {fd:.9e}"
            );
        }
    }
}
