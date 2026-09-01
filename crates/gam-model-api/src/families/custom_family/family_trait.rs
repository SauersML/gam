//! The `CustomFamily` trait itself plus the evaluation result structs it returns
//! (`FamilyEvaluation`, joint-gradient/batched-term carriers) and the eval-scope /
//! outer-eval-context enums that parameterize trait calls.

use crate::families::custom_family::joint_newton_defaults::{
    exact_newton_joint_hessian_directional_derivative_from_blocks,
    exact_newton_joint_hessian_directional_derivative_from_working_sets,
    exact_newton_joint_hessian_from_exact_blocks, exact_newton_joint_hessian_from_working_sets,
    exact_newton_joint_hessiansecond_directional_derivative_from_blocks,
    joint_hessian_has_cross_block_coupling,
};
use crate::families::custom_family::options::{
    BlockwiseFitOptions, OuterDerivativePolicy, assert_block_index_matches_spec,
    assert_block_local_beta_direction, assert_block_local_eta_direction,
    assert_blockstates_are_a_point, assert_hyper_layout_matches_specs, assert_psi_index_in_layout,
    assert_rho_matches_specs, assert_states_match_specs, assert_valid_blockspecs,
    assert_valid_options, default_coefficient_hessian_cost, default_outer_derivative_policy_costs,
    exact_outer_order_with_outer_hvp, validate_hessian_workspace_ready,
};
use crate::families::custom_family::psi_design::{
    CustomFamilyHyperLayout, ExactNewtonJointHessianWorkspace,
};
use gam_linalg::matrix::DesignMatrix;
use gam_problem::{
    BlockGeometryDirectionalDerivative, BlockWorkingSet, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace, ExactNewtonOuterObjective,
    ExactOuterDerivativeOrder, ParameterBlockSpec, ParameterBlockState, PseudoLogdetMode,
};
// The coordinate declaration `block_coefficient_coordinate` returns, re-exported
// beside the constraint types it is derived from (#2748).
pub use gam_problem::CoefficientCoordinate;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Family evaluation over all parameter blocks.
#[derive(Clone, Debug)]
pub struct FamilyEvaluation {
    pub log_likelihood: f64,
    pub blockworking_sets: Vec<BlockWorkingSet>,
}

pub struct ExactNewtonJointGradientEvaluation {
    pub log_likelihood: f64,
    pub gradient: Array1<f64>,
}

/// Batched per-θ_j contributions to the analytic outer gradient.
///
/// Used by [`CustomFamily::batched_outer_gradient_terms`] to amortize the
/// joint-Hessian factorization across all K hyperparameters: instead of
/// computing each `tr(H⁻¹ · Ḣ_j)` independently (K independent solves), the
/// family factors `H` once, computes per-row leverages `L_i = Z_i H⁻¹ Z_iᵀ`,
/// and accumulates all K traces in a single streaming pass.
///
/// All three vectors have length equal to the total number of outer
/// hyperparameters (K = `rho.len() + hyper_layout.len()`), in the
/// same coordinate order as the unified evaluator's gradient: ρ-coords first,
/// ψ-coords appended.
///
/// # Assembly formula
///
/// The caller assembles the outer gradient as
///
/// ```text
///   grad[j] = objective_theta[j]
///           + 0.5 * trace_h_inv_hdot[j]
///           - 0.5 * trace_s_pinv_sdot[j]
/// ```
///
/// matching the three-term convention in `outer_gradient_entry` (penalty +
/// trace − det).
pub struct BatchedOuterHessianTerms {
    /// Exact profiled outer Hessian over θ = (ρ, ψ), assembled or exposed in
    /// operator form by the family in one amortized evaluation.
    pub outer_hessian: gam_problem::HessianValue,
}

pub struct BatchedOuterGradientTerms {
    /// Explicit ∂J/∂θ_j contributions evaluated at the converged β̂ holding
    /// β fixed (i.e. the part that does NOT flow through H or S):
    ///
    /// * For ρ-coords: `½ β̂ᵀ A_k β̂` (penalty quadratic).
    /// * For ψ-coords: `V_i^explicit + g_i^explicit · β̂` style contributions.
    pub objective_theta: Array1<f64>,
    /// `tr(H⁻¹ · ∂H/∂θ_j)` for each j, with H = -∇²log L + S the full
    /// penalized Hessian at the mode.
    pub trace_h_inv_hdot: Array1<f64>,
    /// `tr(S⁺ · ∂S/∂θ_j)` for each j (penalty pseudo-logdet first derivative).
    pub trace_s_pinv_sdot: Array1<f64>,
}

/// Scale-aware exact joint curvature payload for the outer REML evaluator.
/// The neutral definition lives in `gam-problem`; re-exported here so the
/// `custom_family::ExactNewtonOuterCurvature` path keeps resolving without a
/// duplicate definition.
pub use gam_problem::ExactNewtonOuterCurvature;
pub use gam_problem::{ConstraintSet, KhatriRaoConeConstraints, PlacedConstraintBlock};

/// Shared lifecycle for an unbiased sampled outer-derivative pilot.
///
/// Large marginal-slope objectives may use a deterministic
/// Horvitz--Thompson row sample while the outer iterate is moving rapidly, but
/// a fit may only be certified after optimizing the exact full-data measure.
/// The family owns the evaluation counter; the generic outer runner owns the
/// stage transition.  Keeping that transition explicit prevents a solver that
/// converges before the nominal pilot budget from attempting to certify the
/// sampled objective as though it were the exact REML/LAML criterion (#979).
#[derive(Clone, Debug)]
pub struct OuterDerivativePilotSchedule {
    phase_counter: Arc<AtomicUsize>,
    sampled_phase_budget: usize,
}

impl OuterDerivativePilotSchedule {
    pub fn new(phase_counter: Arc<AtomicUsize>, sampled_phase_budget: usize) -> Self {
        Self {
            phase_counter,
            sampled_phase_budget,
        }
    }

    /// Enter the exact full-data phase iff at least one sampled derivative
    /// evaluation actually ran and the family has not already transitioned.
    ///
    /// A compare/exchange loop makes the stage boundary single-shot even if a
    /// future evaluator invokes the hook concurrently.  A zero counter is left
    /// untouched: it means the problem was too small to install a sample (or a
    /// caller supplied its own explicit measure), so no second optimization is
    /// needed.
    pub fn enter_exact_phase(&self) -> bool {
        let mut observed = self.phase_counter.load(Ordering::SeqCst);
        // `0..=budget` records how many sampled derivative points have run.
        // `budget + 1` is the unambiguous exact-phase sentinel. In particular,
        // `observed == budget` still means the solver may have stopped exactly
        // after its last sampled point, before any full-data evaluation.
        let exact_phase_sentinel = self.sampled_phase_budget.saturating_add(1);
        loop {
            if observed == 0 || observed >= exact_phase_sentinel {
                return false;
            }
            match self.phase_counter.compare_exchange(
                observed,
                exact_phase_sentinel,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(current) => observed = current,
            }
        }
    }
}

/// User-defined family contract for multi-block generalized models.
pub trait CustomFamily {

    /// Evaluate log-likelihood and per-block working quantities at current block predictors.
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String>;

    /// Whether the outer REML/LAML logdet term `½ log|H + Sλ|` and its analytic
    /// trace gradient `½ tr((H+Sλ)⁺ ∂Sλ)` are evaluated over the FULL
    /// identifiable subspace `range(H + Sλ)` (mgcv's generalized determinant,
    /// gam#752) rather than the penalty-range subspace `range(Sλ)`.
    ///
    /// This is a value/gradient SUBSPACE-CONSISTENCY concern, orthogonal to
    /// whether the Hessian depends on β (`exact_newton_joint_hessian_beta_dependent`,
    /// which gates the *drift* corrections). The previous code conflated the two
    /// by gating the projected logdet on β-dependence, so `RidgedQuadraticReml`
    /// families (survival/bernoulli marginal-slope) silently used the
    /// `range(Sλ)`-only determinant: on a near-collinear penalty-null trend (the
    /// clustered-PC matern marginal-slope geometry) that DROPS the penalty-null
    /// likelihood determinant `log|U_kᵀ H U_k|` from the value while
    /// `½ log|Sλ|₊` is correctly over `range(Sλ)`, making the ρ-derivative of the
    /// REML criterion inconsistent. The outer optimizer then drives that block's
    /// λ → ∞ and the envelope gradient (valid only at a stationary β̂) freezes —
    /// the constant-‖g‖ outer stall in gam#808/#787.
    ///
    /// The generalized determinant is the correct objective in ALL cases: when
    /// `H + Sλ` is full rank it equals the ordinary logdet (the projection is a
    /// no-op, so the correction is ≈0), and when it is rank-deficient it drops
    /// only the truly unidentified `ker(H) ∩ ker(Sλ)` directions — exactly the
    /// directions `½ log|Sλ|₊` also omits, keeping value and gradient over one
    /// subspace. Always enabled by default.
    fn use_projected_penalty_logdet(&self) -> bool {
        true
    }

    /// Optional block-concatenated log-likelihood gradient `g = nabla l(theta)`
    /// assembled from the SAME single source of truth as
    /// [`Self::exact_newton_joint_hessian`] (e.g. a per-row jet-tower kernel), so
    /// the damped Newton `H delta = g` is solved on a consistent (objective,
    /// gradient, Hessian) triple. The default returns `None`, leaving the caller
    /// on its legacy hand-assembled gradient.
    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        assert_blockstates_are_a_point(block_states, "exact Newton joint log-likelihood gradient");
        Ok(None)
    }

    /// Optional batched analytic-Hessian / HVP hook.
    ///
    /// This is the Hessian-side analogue of
    /// [`Self::batched_outer_gradient_terms`]: families that can share a
    /// single factorization, row-leverage stream, or directional θθ kernel
    /// across all explicit outer-Hessian terms return the exact profiled
    /// Hessian here.  The evaluator uses this hook only for Hessian-capable
    /// families and only after the inner mode has been fitted; default
    /// `None` leaves unsupported families on their existing exact path.
    fn batched_outer_hessian_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &CustomFamilyHyperLayout,
        rho: &Array1<f64>,
        hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterHessianTerms>, String> {
        assert_valid_blockspecs(specs, "batched outer Hessian terms");
        assert_states_match_specs(block_states, specs, "batched outer Hessian terms");
        assert_hyper_layout_matches_specs(hyper_layout, specs, "batched outer Hessian terms");
        assert_rho_matches_specs(rho, specs, "batched outer Hessian terms");
        validate_hessian_workspace_ready(
            &hessian_workspace,
            "batched outer Hessian terms",
            gam_problem::EvalMode::ValueGradientHessian,
        )?;
        Ok(self
            .outer_hyper_hessian_operator(specs)
            .map(|operator| BatchedOuterHessianTerms {
                outer_hessian: gam_problem::HessianValue::Operator(operator),
            }))
    }

    /// Family-supplied exact outer Hessian operator over θ = (ρ, ψ).
    ///
    /// When a family can produce the full profiled outer Hessian as a
    /// matrix-free Hv operator — using its own directional θθ kernels and
    /// trace algebra rather than the generic per-pair enumeration — it
    /// overrides this method and returns `Some(op)`.  The unified REML/LAML
    /// evaluator wires the operator into `HessianValue::Operator` via
    /// the `HessianDerivativeProvider::family_outer_hessian_operator` hook
    /// the family installs on its provider; consumers see a generic
    /// `Arc<dyn HessianOperator>` (`apply_into`, `apply_mat`, `dim`, and the
    /// explicit materialization work model).
    ///
    /// Default returns `None`, leaving the family on the existing pairwise
    /// assembly path.  This is the architectural contract for CTN, survival
    /// (Gompertz-Makeham + timewiggle), GAMLSS location-scale, and
    /// Bernoulli marginal-slope families to plug their directional
    /// outer-HVP operators into the same surface.
    fn outer_hyper_hessian_operator(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> Option<Arc<dyn gam_problem::HessianOperator>> {
        assert_valid_blockspecs(specs, "outer hyper-Hessian operator");
        None
    }

    /// Structural-coupling probe shared by the `_with_specs` joint dispatch
    /// gates: is the family's `exact_newton_joint_hessian` a genuinely coupled
    /// matrix (nonzero off-diagonal blocks), as opposed to the trait's
    /// block-diagonal default? This is the marker-free signal that lets the
    /// engine trust a coupled multi-block family that overrode the joint
    /// Hessian without hand-setting `has_explicit_joint_hessian()`. Returns
    /// `false` when no joint Hessian is available or it is block-diagonal.
    fn joint_hessian_is_structurally_coupled(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<bool, String> {
        Ok(match self.exact_newton_joint_hessian(block_states)? {
            Some(hessian) => joint_hessian_has_cross_block_coupling(&hessian, block_states),
            None => false,
        })
    }

    /// Optional contracted second beta-derivative of the observed joint
    /// Newton information:
    ///
    ///   ∇²_β tr(W H(β))
    ///
    /// for a fixed full-joint trace weight `W`.
    ///
    /// This is the wide-p route for Jeffreys' omitted second-directional
    /// completion. The default returns `None`, so callers fall back to the
    /// existing p(p+1)/2 pairwise `H''[e_a,e_b]` path.
    fn exact_newton_joint_contracted_trace_hessian(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != specs.len() {
            return Err(format!(
                "exact_newton_joint_contracted_trace_hessian default: block state count {} != spec count {}",
                block_states.len(),
                specs.len()
            ));
        }
        let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        if weight.dim() != (total, total) {
            return Err(format!(
                "exact_newton_joint_contracted_trace_hessian default: weight shape {:?} != ({total}, {total})",
                weight.dim()
            ));
        }
        for (block_idx, (state, spec)) in block_states.iter().zip(specs.iter()).enumerate() {
            let p_block = spec.design.ncols();
            if state.beta.len() != p_block {
                return Err(format!(
                    "exact_newton_joint_contracted_trace_hessian default: block {block_idx} beta length {} != design cols {p_block}",
                    state.beta.len()
                ));
            }
        }
        Ok(None)
    }

    /// Internal helper: do the outer-REML `_with_specs` defaults trust the
    /// inner-fit's block-diagonal-from-blocks output for this family?
    ///
    /// Trustworthy iff:
    /// - single-block (no cross-block coupling possible), or
    /// - the family has declared its blocks uncoupled in the likelihood
    ///   Hessian (`likelihood_blocks_uncoupled` ⇒ block-diagonal IS exact),
    ///   or
    /// - the family has an explicit joint-Hessian override
    ///   (`has_explicit_joint_hessian` ⇒ what we receive from
    ///   `exact_newton_joint_hessian` is the true coupled Hessian, not the
    ///   block-diagonal default).
    fn outer_default_trustworthy_for_joint_hessian(&self, specs: &[ParameterBlockSpec]) -> bool {
        specs.len() <= 1 || self.likelihood_blocks_uncoupled() || self.has_explicit_joint_hessian()
    }

    /// Optional first directional derivative matching
    /// `exact_newton_outer_curvature`.
    fn exact_newton_outer_curvature_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    /// Spec-aware variant of `exact_newton_outer_curvature_directional_derivative`.
    fn exact_newton_outer_curvature_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Dropping the specs on the way to the spec-free variant is only sound
        // when they describe the very states being differentiated.
        assert_valid_blockspecs(specs, "exact Newton outer curvature directional derivative");
        assert_states_match_specs(
            block_states,
            specs,
            "exact Newton outer curvature directional derivative",
        );
        self.exact_newton_outer_curvature_directional_derivative(block_states, d_beta_flat)
    }

    /// Optional second directional derivative matching
    /// `exact_newton_outer_curvature`.
    fn exact_newton_outer_curvature_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    /// Spec-aware variant of `exact_newton_outer_curvature_second_directional_derivative`.
    fn exact_newton_outer_curvature_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Same contract as the first-order sibling: the specs may only be
        // dropped once they are known to describe these states.
        assert_valid_blockspecs(
            specs,
            "exact Newton outer curvature second directional derivative",
        );
        assert_states_match_specs(
            block_states,
            specs,
            "exact Newton outer curvature second directional derivative",
        );
        self.exact_newton_outer_curvature_second_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    /// Optional joint multi-block outer-hyper surrogate Hessian over the
    /// flattened coefficient vector.
    ///
    /// This hook exists for families whose inner working representation is
    /// block-diagonal/diagonal in `evaluate(...)`, but whose outer profiled
    /// smoothing derivatives are still joint because the fitted mode response
    /// couples blocks. The generic blockwise outer-hyper surrogate only sees
    /// per-block working sets, so it cannot recover missing cross-block
    /// curvature on its own.
    ///
    /// Families that can construct a mathematically valid joint surrogate
    /// `H_L(beta)` for the current realized `specs` may override this and the
    /// two directional derivative hooks below. Generic code then reuses the
    /// same joint rho-calculus as the exact path, but on the family-supplied
    /// surrogate curvature instead of the exact Newton Hessian.
    ///
    /// Default behavior is to reuse the spec-aware exact joint curvature when
    /// the family already provides it. That is the mathematically correct
    /// repair for the old broken multi-block blockwise surrogate path: if the
    /// family knows the full coupled Hessian and its beta-drifts, generic code
    /// should use that joint information instead of pretending per-block
    /// working sets are enough.
    fn joint_outer_hyper_surrogate_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_with_specs(block_states, specs)
    }

    /// Optional first beta-directional derivative of the joint surrogate
    /// outer-hyper Hessian.
    fn joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_flat,
        )
    }

    /// Optional second beta-directional derivative of the joint surrogate
    /// outer-hyper Hessian.
    fn joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    /// Optional exact second directional derivative of diagonal working weights.
    ///
    /// This callback supplies the `d²w` term for static-design single-block
    /// generic fallback Hessian drift:
    ///
    ///   D²_beta H_L[u, v] = X^T diag(D²w[D eta_u, D eta_v]) X.
    ///
    /// Families with coefficient-dependent block geometry must use an exact
    /// Newton Hessian path or a joint outer path until second-order geometry
    /// hooks are available; the generic diagonal fallback will reject nonzero
    /// first-order geometry while building `d²H`.
    fn diagonalworking_weights_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        d_eta_u: &Array1<f64>,
        d_eta_v: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert_block_local_eta_direction(
            block_states,
            block_index,
            d_eta_u,
            "diagonal working-weight second directional derivative (u)",
        );
        assert_block_local_eta_direction(
            block_states,
            block_index,
            d_eta_v,
            "diagonal working-weight second directional derivative (v)",
        );
        Ok(None)
    }

}

/// Scope of an outer-evaluation context — distinguishes a real outer
/// derivative evaluation (where auto-subsample is allowed to install a
/// fresh stratified mask and emit phase prints) from an inner
/// coefficient line-search trial (where the family must reuse the outer
/// row measure, so auto-subsample must stay disabled).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum EvalScope {
    /// Real outer derivative evaluation: ρ has advanced; auto-subsample
    /// install paths may build/refresh a mask keyed on this ρ.
    OuterDerivative,
    /// Inner coefficient trial (joint-Newton / line-search) at fixed
    /// outer ρ: row measure must remain identical to the surrounding
    /// outer eval, so auto-subsample must not install a fresh mask.
    InnerCoefficient,
}

/// Context published by the outer smoothing optimizer for every
/// downstream family evaluation. Carries the current outer ρ and a
/// monotonic per-outer-eval id alongside the [`EvalScope`] tag used to
/// gate auto-subsample installation. See the
/// [`BlockwiseFitOptions::outer_eval_context`] field doc for the bug
/// this prevents.
#[derive(Clone, Debug)]
pub struct OuterEvalContext {
    pub rho: Arc<Array1<f64>>,
    pub eval_id: usize,
    pub scope: EvalScope,
}
