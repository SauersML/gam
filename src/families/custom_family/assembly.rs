//! Inner-assembly construction, the unified joint cost/gradient/EFS
//! evaluators, the assembled-operator cache, the joint outer-evaluate
//! entry points, and block-local penalty assembly, split out of
//! `outer_objective.rs` by concern (#1145). Re-exported via `custom_family`.

use super::*;

/// Build the canonical unified REML/LAML assembly for a custom-family outer
/// evaluation.
pub(crate) fn build_custom_family_inner_assembly<'dp>(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn HessianOperator>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    rho_prior: crate::types::RhoPrior,
    deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,
    ext_bundle: Option<ExtCoordBundle>,
    firth_value: Option<f64>,
) -> Result<(crate::estimate::reml::assembly::InnerAssembly<'dp>, usize), String> {
    use crate::estimate::reml::assembly::{
        InnerAssembly, PenaltyBlockDesc, penalty_coords_from_blocks,
    };

    // Collect dense penalty matrices so references stay valid for the assembler.
    let per_block_penalties_dense: Vec<Vec<Array2<f64>>> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        (0..specs.len())
            .into_par_iter()
            .map(|b| specs[b].penalties.iter().map(|p| p.to_dense()).collect())
            .collect()
    };
    let block_descs: Vec<PenaltyBlockDesc> = (0..specs.len())
        .flat_map(|b| {
            let (start, end) = ranges[b];
            per_block_penalties_dense[b]
                .iter()
                .map(move |dense| PenaltyBlockDesc {
                    matrix: dense,
                    range_start: start,
                    range_end: end,
                })
        })
        .collect();
    let penalty_coords = penalty_coords_from_blocks(&block_descs, total)?;

    // Compute penalty logdet derivatives.
    let per_block_penalties: Vec<&[Array2<f64>]> = per_block_penalties_dense
        .iter()
        .map(|v| v.as_slice())
        .collect();
    let penalty_logdet_ridge = if options.ridge_policy.include_penalty_logdet {
        ridge
    } else {
        0.0
    };
    let penalty_logdet =
        compute_block_penalty_logdet_derivs(per_block, &per_block_penalties, penalty_logdet_ridge)?;

    let n_observations = inner.block_states.first().map(|s| s.eta.len()).unwrap_or(0);

    // Unpack optional ext-coord bundle.
    let (ext_coords, ext_coord_pair_fn, rho_ext_pair_fn, fixed_drift_deriv, contracted_psi_fn) =
        if let Some(bundle) = ext_bundle {
            (
                bundle.coords,
                bundle.ext_ext_fn,
                bundle.rho_ext_fn,
                bundle.drift_fn,
                bundle.contracted_psi_fn,
            )
        } else {
            (Vec::new(), None, None, None, None)
        };

    let ext_dim = ext_coords.len();

    let evaluator = InnerAssembly {
        log_likelihood: inner.log_likelihood,
        // inner.penalty_value includes the 0.5 factor (= 0.5 β̂ᵀSβ̂), but the
        // unified evaluator convention expects the FULL quadratic β̂ᵀSβ̂ and
        // applies 0.5 itself. Double to match the convention.
        penalty_quadratic: 2.0 * inner.penalty_value,
        beta: beta_flat.clone(),
        n_observations,
        hessian_op,
        penalty_coords,
        penalty_logdet,
        dispersion: DispersionHandling::Fixed {
            phi: 1.0,
            include_logdet_h,
            include_logdet_s,
        },
        rho_curvature_scale,
        rho_prior,
        hessian_logdet_correction,
        penalty_subspace_trace,
        deriv_provider: Some(deriv_provider),
        // Tier-B Firth fold (gam#979): the inner mode minimizes
        // `−ℓ + ½βᵀSβ − Φ`, so the LAML cost must subtract the same gated
        // `Φ(β̂)` or the envelope-based analytic outer gradient and the value
        // describe different criteria at every Firth-active mode.
        firth: firth_value.map(ExactJeffreysTerm::value_only),
        nullspace_dim: None,
        barrier_config: None,
        ext_coords,
        ext_coord_pair_fn,
        rho_ext_pair_fn,
        fixed_drift_deriv,
        contracted_psi_second_order: contracted_psi_fn,
        kkt_residual: inner.kkt_residual.clone(),
        active_constraints: inner.active_constraints.clone(),
    };

    Ok((evaluator, ext_dim))
}

pub(crate) struct FirstOrderTraceSkipOperator {
    pub(crate) inner: Arc<dyn HessianOperator>,
    pub(crate) remaining_first_order_traces: AtomicUsize,
}

impl FirstOrderTraceSkipOperator {
    pub(crate) fn new(inner: Arc<dyn HessianOperator>, skip_count: usize) -> Self {
        Self {
            inner,
            remaining_first_order_traces: AtomicUsize::new(skip_count),
        }
    }

    pub(crate) fn first_order_skip_active(&self) -> bool {
        self.remaining_first_order_traces.load(Ordering::Acquire) > 0
    }

    pub(crate) fn consume_first_order_trace(&self) -> bool {
        let mut current = self.remaining_first_order_traces.load(Ordering::Acquire);
        while current > 0 {
            match self.remaining_first_order_traces.compare_exchange(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
        false
    }
}

impl HessianOperator for FirstOrderTraceSkipOperator {
    fn logdet(&self) -> f64 {
        self.inner.logdet()
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        self.inner.trace_hinv_product(a)
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        if self.first_order_skip_active() {
            None
        } else {
            self.inner.as_exact_dense_spectral()
        }
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        if self.first_order_skip_active() {
            Err("backend does not support tangent projection".to_string())
        } else {
            self.inner.assemble_h_dense_for_tangent_projection()
        }
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.inner.trace_hinv_operator(op)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.inner.solve(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.inner.solve_multi(rhs)
    }

    fn stochastic_trace_solve(&self, rhs: &Array1<f64>, rel_tol: f64) -> Array1<f64> {
        self.inner.stochastic_trace_solve(rhs, rel_tol)
    }

    fn stochastic_trace_solve_for_probe(
        &self,
        rhs: &Array1<f64>,
        rel_tol: f64,
        probe_id: u64,
        trace_state: Option<&Arc<Mutex<StochasticTraceState>>>,
    ) -> Array1<f64> {
        self.inner
            .stochastic_trace_solve_for_probe(rhs, rel_tol, probe_id, trace_state)
    }

    fn stochastic_trace_solve_multi(&self, rhs: &Array2<f64>, rel_tol: f64) -> Array2<f64> {
        self.inner.stochastic_trace_solve_multi(rhs, rel_tol)
    }

    fn has_matrix_free_trace_cg_operator(&self) -> bool {
        self.inner.has_matrix_free_trace_cg_operator()
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.inner.trace_hinv_product_cross(a, b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        self.inner.trace_hinv_matrix_operator_cross(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        self.inner.trace_hinv_operator_cross(left, right)
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        if self.consume_first_order_trace() {
            0.0
        } else {
            self.inner.trace_logdet_gradient(a)
        }
    }

    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        self.inner.xt_logdet_kernel_x_diagonal(x)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        if self.consume_first_order_trace() {
            0.0
        } else {
            self.inner.trace_logdet_operator(op)
        }
    }

    fn trace_logdet_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        if self.consume_first_order_trace() {
            0.0
        } else {
            self.inner.trace_logdet_h_k(a_k, third_deriv_correction)
        }
    }

    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        if self.consume_first_order_trace() {
            0.0
        } else {
            self.inner
                .trace_logdet_block_local(block, scale, start, end)
        }
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        self.inner.trace_logdet_hessian_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.inner
            .trace_logdet_hessian_cross_matrix_operator(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.inner.trace_logdet_hessian_cross_operator(h_i, h_j)
    }

    fn active_rank(&self) -> usize {
        self.inner.active_rank()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn is_dense(&self) -> bool {
        self.inner.is_dense()
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        if self.first_order_skip_active() {
            false
        } else {
            self.inner.prefers_stochastic_trace_estimation()
        }
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        self.inner.logdet_traces_match_hinv_kernel()
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        if self.first_order_skip_active() {
            None
        } else {
            self.inner.as_dense_spectral()
        }
    }
}

/// Build an `InnerSolution` from joint Hessian data and call the unified evaluator.
///
/// Bridge between the custom family's joint Hessian infrastructure and the
/// unified REML/LAML evaluator, routed through the canonical assembly module.
pub(crate) fn unified_joint_cost_gradient(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn HessianOperator>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    rho_prior: crate::types::RhoPrior,
    deriv_provider: Box<dyn HessianDerivativeProvider + '_>,
    eval_mode: EvalMode,
    ext_bundle: Option<ExtCoordBundle>,
    first_order_trace_skip: Option<Array1<f64>>,
    // Gated Tier-B Jeffreys value `Φ(β̂)`, folded into the LAML cost
    // (`cost −= Φ`) so the outer criterion matches the Φ-augmented inner
    // objective (gam#979). `None` when the term is unavailable/gated to zero.
    firth_value: Option<f64>,
) -> Result<
    (
        f64,
        Array1<f64>,
        crate::solver::rho_optimizer::HessianResult,
    ),
    String,
> {
    let hessian_op: Arc<dyn HessianOperator> = match first_order_trace_skip.as_ref() {
        Some(trace_values) if !trace_values.is_empty() => Arc::new(
            FirstOrderTraceSkipOperator::new(hessian_op, trace_values.len()),
        ),
        _ => hessian_op,
    };
    let (evaluator, ext_dim) = build_custom_family_inner_assembly(
        inner,
        specs,
        per_block,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        penalty_subspace_trace,
        include_logdet_h,
        include_logdet_s,
        options,
        rho_prior,
        deriv_provider,
        ext_bundle,
        firth_value,
    )?;
    let rho_slice = rho
        .as_slice()
        .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
    let first_order_trace_correction = first_order_trace_skip.map(|trace_values| {
        let gradient_correction = trace_values.mapv(|trace| 0.5 * trace);
        (0.0, gradient_correction, None)
    });
    let result = evaluator.evaluate(rho_slice, eval_mode, first_order_trace_correction)?;

    let cost = result.cost;
    let gradient = result
        .gradient
        .unwrap_or_else(|| Array1::zeros(rho.len() + ext_dim));

    let hessian = result.hessian;

    Ok((cost, gradient, hessian))
}

pub(crate) fn unified_joint_efs_eval(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn HessianOperator>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    rho_prior: crate::types::RhoPrior,
    deriv_provider: Box<dyn HessianDerivativeProvider + '_>,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<crate::solver::rho_optimizer::EfsEval, String> {
    let (assembly, _) = build_custom_family_inner_assembly(
        inner,
        specs,
        per_block,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        penalty_subspace_trace,
        include_logdet_h,
        include_logdet_s,
        options,
        rho_prior,
        deriv_provider,
        ext_bundle,
        // The EFS screening path evaluates the Φ-less criterion with an
        // unaugmented operator throughout; it stays self-consistent without
        // the Tier-B Firth fold.
        None,
    )?;
    let rho_slice = rho
        .as_slice()
        .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
    let inner_solution = assembly.build();
    let has_psi = inner_solution
        .ext_coords
        .iter()
        .any(|coord| !coord.is_penalty_like);
    // Always evaluate gradient: the universal-form EFS step
    // `Δρ = log(1 − 2·g_full / q_eff)` reads it directly from the cost
    // gradient slot, so out-of-band cost terms (TK, prior, Firth,
    // barrier, SAS log-δ ridge) shift the multiplicative target through
    // their gradient contribution without needing per-augmentation
    // post-corrections.
    let eval_mode = EvalMode::ValueAndGradient;
    let result = crate::estimate::reml::assembly::evaluate_solution(
        &inner_solution,
        rho_slice,
        eval_mode,
        None,
    )?;

    let gradient = result
        .gradient
        .as_ref()
        .ok_or_else(|| "EFS evaluation did not return the required gradient".to_string())?;
    let gradient_slice = gradient
        .as_slice()
        .ok_or_else(|| "outer gradient must be contiguous for EFS".to_string())?;

    if has_psi {
        let inner_hessian_scale =
            hessian_operator_geometric_scale(inner_solution.hessian_op.as_ref());
        let hybrid = compute_hybrid_efs_update(&inner_solution, rho_slice, gradient_slice);
        Ok(crate::solver::rho_optimizer::EfsEval {
            cost: result.cost,
            steps: hybrid.steps,
            beta: Some(inner_solution.beta.clone()),
            psi_gradient: if hybrid.psi_gradient.is_empty() {
                None
            } else {
                Some(Array1::from_vec(hybrid.psi_gradient))
            },
            psi_indices: if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(hybrid.psi_indices)
            },
            inner_hessian_scale,
            logdet_enclosure_gap: None,
        })
    } else {
        let inner_hessian_scale =
            hessian_operator_geometric_scale(inner_solution.hessian_op.as_ref());
        Ok(crate::solver::rho_optimizer::EfsEval {
            cost: result.cost,
            steps: compute_efs_update(&inner_solution, rho_slice, gradient_slice),
            beta: Some(inner_solution.beta.clone()),
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale,
            logdet_enclosure_gap: None,
        })
    }
}

/// Same-ρ assembled-operator reuse for the custom-family outer driver.
///
/// The outer Hessian operator the cost-side `hop.logdet()` and the gradient
/// traces consume is `H(β̂, ρ) + S_λ(ρ) + H_Φ(β̂, ρ)`. Its expensive part is the
/// dense spectral factorization (`g_factor` / `projected_factor_cache`), built
/// lazily inside `MatrixFreeSpdOperator` / `BlockCoupledOperator` the first time
/// a logdet/trace method is touched (≈14–19 s at biobank scale). BFGS issues a
/// `Value` eval immediately followed by a `ValueAndGradient` eval at the SAME ρ
/// (and the line search re-probes ρ), so the released path rebuilt + refactorized
/// that operator 2–4× per fit at identical ρ.
///
/// The operator is a deterministic function of `(β̂, ρ)` for a fixed
/// family/data plus the scalar assembly knobs (ridges, curvature scale, logdet
/// flags, pseudo-logdet mode, Jeffreys curvature). We therefore cache the
/// assembled `Arc<dyn HessianOperator>` keyed by a content fingerprint of ALL of
/// those inputs and reuse it on a bit-identical hit. Because the reuse condition
/// is exact byte-equality of the build inputs, the reused operator — and so the
/// LAML cost and its analytic gradient — is bit-identical to a fresh build.
/// On a miss we build + store, evicting the older of the (at most) two retained
/// entries, which bounds memory to the last two distinct ρ assemblies.
struct AssembledOperatorCache {
    /// `(fingerprint, operator)` for at most the last two distinct assemblies.
    entries: Vec<(u64, Arc<dyn HessianOperator>)>,
}

impl AssembledOperatorCache {
    const CAPACITY: usize = 2;

    fn get(&self, fingerprint: u64) -> Option<Arc<dyn HessianOperator>> {
        self.entries
            .iter()
            .find(|(key, _)| *key == fingerprint)
            .map(|(_, op)| Arc::clone(op))
    }

    fn insert(&mut self, fingerprint: u64, op: Arc<dyn HessianOperator>) {
        if self.entries.iter().any(|(key, _)| *key == fingerprint) {
            return;
        }
        if self.entries.len() >= Self::CAPACITY {
            // Evict the oldest entry (front); newest assemblies stay resident so
            // the immediate Value→ValueAndGradient pair at one ρ always hits.
            self.entries.remove(0);
        }
        self.entries.push((fingerprint, op));
    }
}

fn assembled_operator_cache() -> &'static Mutex<AssembledOperatorCache> {
    static CACHE: OnceLock<Mutex<AssembledOperatorCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(AssembledOperatorCache {
            entries: Vec::with_capacity(AssembledOperatorCache::CAPACITY),
        })
    })
}

/// Fold a finite `f64`'s canonical bit pattern into a hasher (±0.0 → +0.0,
/// mirroring `solver::reml::rho_key::sanitized_rhokey`). Non-finite values poison
/// the fingerprint with a distinguished sentinel so a NaN/∞ assembly never
/// aliases a finite one (it will simply never hit, which is the safe outcome).
fn hash_f64<H: std::hash::Hasher>(value: f64, hasher: &mut H) {
    use std::hash::Hash;
    let bits = if value == 0.0 {
        0.0f64.to_bits()
    } else if value.is_finite() {
        value.to_bits()
    } else {
        // Distinct sentinel for any non-finite component.
        0xFFFF_FFFF_FFFF_FFFFu64
    };
    bits.hash(hasher);
}

/// Content fingerprint of every input that determines the assembled outer
/// Hessian operator. Reuse is gated on exact equality of this fingerprint, so a
/// hit means a bit-identical operator. `None` (here: never) would disable
/// caching; we always produce a fingerprint and let mismatches simply miss.
fn assembled_operator_fingerprint(
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    h_joint_unpen: &JointHessianSource,
    scaled_s_lambdas: &[Array2<f64>],
    robust_jeffreys_hphi_for_operator: Option<&Array2<f64>>,
    ranges: &[(usize, usize)],
    total: usize,
    scaled_joint_trace_diagonal_ridge: f64,
    rho_curvature_scale: f64,
    pseudo_logdet_mode: PseudoLogdetMode,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Structural discriminants.
    total.hash(&mut hasher);
    ranges.hash(&mut hasher);
    (pseudo_logdet_mode == PseudoLogdetMode::Smooth).hash(&mut hasher);
    hash_f64(scaled_joint_trace_diagonal_ridge, &mut hasher);
    hash_f64(rho_curvature_scale, &mut hasher);
    // ρ and β̂ together pin the family/data state at this evaluation: for a fixed
    // family/data the operator is `H(β̂, ρ) + S_λ(ρ) + H_Φ(β̂, ρ)`, so identical
    // (ρ, β̂) ⇒ identical operator, and distinct fits at the same ρ differ in β̂.
    rho.len().hash(&mut hasher);
    for &v in rho {
        hash_f64(v, &mut hasher);
    }
    beta_flat.len().hash(&mut hasher);
    for &v in beta_flat {
        hash_f64(v, &mut hasher);
    }
    // Scaled penalty blocks (O(Σ p_b²); cheap vs the O(total³) factorization).
    scaled_s_lambdas.len().hash(&mut hasher);
    for matrix in scaled_s_lambdas {
        matrix.dim().hash(&mut hasher);
        for &v in matrix.iter() {
            hash_f64(v, &mut hasher);
        }
    }
    // Jeffreys curvature: presence + content.
    match robust_jeffreys_hphi_for_operator {
        Some(hphi) => {
            1u8.hash(&mut hasher);
            hphi.dim().hash(&mut hasher);
            for &v in hphi.iter() {
                hash_f64(v, &mut hasher);
            }
        }
        None => 0u8.hash(&mut hasher),
    }
    // Dense joint Hessian content is part of the operator only on the dense
    // path; the operator (matrix-free) path's curvature is pinned by (ρ, β̂)
    // through the family workspace and is fingerprinted above. Including the
    // dense source closes the only remaining content channel.
    match h_joint_unpen {
        JointHessianSource::Dense(matrix) => {
            2u8.hash(&mut hasher);
            matrix.dim().hash(&mut hasher);
            for &v in matrix.iter() {
                hash_f64(v, &mut hasher);
            }
        }
        JointHessianSource::Operator { diagonal, .. } => {
            3u8.hash(&mut hasher);
            diagonal.len().hash(&mut hasher);
            for &v in diagonal.iter() {
                hash_f64(v, &mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Shared implementation for the joint exact-Newton and surrogate outer paths.
///
/// Both paths differ only in:
/// - how the joint Hessian source is obtained (exact vs surrogate family methods)
/// - the closure for computing D_β H_L[v] (`compute_dh`)
/// - the closure for computing D²_β H_L[u, v] (`compute_d2h`)
/// - whether a tangent-basis projection is applied to the mode inverse
///
/// This function encapsulates all shared logic: penalty assembly, mode inverse
/// computation, precomputation of joint corrections + second-order traces, and
/// routing through `unified_joint_cost_gradient`.
pub(crate) fn joint_outer_evaluate(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    h_joint_unpen: JointHessianSource,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    moderidge: f64,
    extra_logdet_ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    include_logdet_h: bool,
    include_logdet_s: bool,
    strict_spd: bool,
    project_hessian_logdet: bool,
    eval_mode: EvalMode,
    options: &BlockwiseFitOptions,
    rho_prior: crate::types::RhoPrior,
    pseudo_logdet_mode: PseudoLogdetMode,
    compute_dh: &DriftDerivFn<'_>,
    compute_dh_many: Option<&DriftDerivManyFn<'_>>,
    compute_d2h: &DriftSecondDerivFn<'_>,
    compute_d2h_many: Option<&DriftSecondDerivManyFn<'_>>,
    owned_compute_dh: Option<
        Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
    >,
    owned_compute_dh_many: Option<
        Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync>,
    >,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    >,
    owned_compute_d2h_many: Option<
        Arc<
            dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
                + Send
                + Sync,
        >,
    >,
    ext_bundle: Option<ExtCoordBundle>,
    first_order_trace_skip: Option<Array1<f64>>,
    batched_outer_hessian_operator: Option<
        Arc<dyn crate::solver::rho_optimizer::OuterHessianOperator>,
    >,
    // Universal under-identification robustness (always armed when the family can
    // expose an exact joint Hessian). The
    // outer REML logdet AND its trace derivatives must run on the same
    // Jeffreys-augmented Hessian `H + S_λ + H_Φ` the inner Newton converged on,
    // or the LAML value and its analytic gradient describe different objectives.
    // Folding `H_Φ` into the operator's matvec augments the inverse/logdet, but is
    // NOT by itself sufficient: `H_Φ` depends on ρ THROUGH β̂, so the trace
    // contraction also needs its mode-response drift `D_β H_Φ[v_k]` — supplied
    // separately via `jeffreys_hphi_drift` and folded into the first-order trace
    // by `JeffreysHphiAwareJointDerivatives`. `None` means this evaluation has
    // no active Jeffreys curvature (empty system, unavailable exact derivatives,
    // or the conditioning gate proved the term zero), not a user-selected
    // robustness-off mode.
    // Gated Jeffreys VALUE `Φ(β̂)` paired with the divided-difference curvature
    // `H_Φ` and its (optional) second-order completion, all from the same term
    // evaluation. The value is folded into the LAML cost (`cost −= Φ`) so the
    // outer criterion is the Laplace approximation of the SAME Firth-augmented
    // objective the inner Newton converged on; the completion is folded into
    // the mode-response OPERATOR only (see
    // `custom_family_outer_jeffreys_hphi` for the chain-rule split) (gam#979).
    robust_jeffreys_phi_hphi: Option<(f64, Array2<f64>, Option<Array2<f64>>)>,
    // Companion mode-response drift `D_β H_Φ[δβ]` for the outer gradient's trace
    // identity. `Some` exactly when `robust_jeffreys_phi_hphi` is `Some` (same
    // under-identified span); installing it wraps the derivative provider so the
    // first-order trace gains the `½ tr[(H+S_λ+H_Φ)⁻¹ D_β H_Φ[v_k]]` term that
    // makes the analytic gradient match the augmented objective. `None` ⇒ the
    // provider is used unwrapped.
    jeffreys_hphi_drift: Option<JeffreysHphiDriftBatchFn>,
) -> Result<OuterObjectiveEvalResult, String> {
    let joint_trace_diagonal_ridge = moderidge + if !strict_spd { extra_logdet_ridge } else { 0.0 };
    let scaled_joint_trace_diagonal_ridge = rho_curvature_scale * joint_trace_diagonal_ridge;

    let (robust_jeffreys_phi, robust_jeffreys_hphi, robust_jeffreys_completion): (
        Option<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    ) = match robust_jeffreys_phi_hphi {
        Some((phi, hphi, completion)) => (Some(phi), Some(hphi), completion),
        None => (None, None, None),
    };
    // Mode-response operator curvature: the divided-difference `H_Φ` PLUS its
    // second-order completion when available — the TRUE Hessian of the
    // Φ-augmented inner objective, which is what `v_k = ∂β̂/∂ρ_k` solves
    // against. The logdet VALUE and its trace kernel keep the bare `H_Φ`
    // (value↔drift consistency); see `custom_family_outer_jeffreys_hphi`.
    // Folded ONLY when the projected kernel will own the value and the
    // first-order traces (the same precondition as the kernel install below);
    // on the unprojected route the operator IS the value/trace object and
    // must stay on the divided-difference pair.
    let completion_in_operator = project_hessian_logdet
        && include_logdet_h
        && include_logdet_s
        && pseudo_logdet_mode == PseudoLogdetMode::Smooth;
    let robust_jeffreys_hphi_for_operator: Option<Array2<f64>> = match (
        robust_jeffreys_hphi.as_ref(),
        robust_jeffreys_completion.filter(|_| completion_in_operator),
    ) {
        (Some(hphi), Some(completion)) => Some(hphi + &completion),
        (Some(hphi), None) => Some(hphi.clone()),
        (None, _) => None,
    };
    // Pre-scale the outer-REML Jeffreys curvature into the same rescaled space as
    // the penalties so the projected-logdet path and the operator agree. `None`
    // (flag OFF / no under-identified span) keeps the released outer REML exact.
    let scaled_robust_jeffreys_hphi: Option<Array2<f64>> = robust_jeffreys_hphi
        .as_ref()
        .map(|hphi| hphi.mapv(|value| rho_curvature_scale * value));

    // Build derivative provider from the caller-supplied closures.
    let base_provider_box: Box<dyn HessianDerivativeProvider + '_> =
        if let (Some(owned_dh), Some(owned_d2h)) = (owned_compute_dh, owned_compute_d2h) {
            Box::new(OwnedJointDerivProvider {
                compute_dh: owned_dh,
                compute_dh_many: owned_compute_dh_many,
                compute_d2h: owned_d2h,
                compute_d2h_many: owned_compute_d2h_many,
                family_outer_hessian_operator: batched_outer_hessian_operator.clone(),
            })
        } else {
            Box::new(BorrowedJointDerivProvider {
                compute_dh,
                compute_dh_many,
                compute_d2h,
                compute_d2h_many,
                family_outer_hessian_operator: batched_outer_hessian_operator.clone(),
            })
        };

    // Install the Jeffreys-`H_Φ` mode-response drift on top of the likelihood
    // drift whenever the Jeffreys term is active. This is the term that makes the
    // analytic outer gradient match the augmented objective `½ log|H+S_λ+H_Φ|`;
    // without it the gradient omits `D_β H_Φ[v_k]` and the line search / KKT
    // certification drifts in exactly the near-separating regime this machinery
    // exists for. `None` ⇒ provider used unwrapped (byte-identical released path).
    let provider_box: Box<dyn HessianDerivativeProvider + '_> = match jeffreys_hphi_drift {
        Some(drift) => Box::new(JeffreysHphiAwareJointDerivatives::new(
            base_provider_box,
            drift,
            total,
        )),
        None => base_provider_box,
    };

    let scaled_s_lambdas: Vec<Array2<f64>> = inner
        .s_lambdas
        .iter()
        .map(|matrix| {
            if rho_curvature_scale == 1.0 {
                matrix.clone()
            } else {
                matrix.mapv(|value| rho_curvature_scale * value)
            }
        })
        .collect();

    // Reuse the assembled outer Hessian operator (and its lazily-built spectral
    // factorization) when an immediately-prior eval at the SAME ρ/β̂/curvature
    // assembled the bit-identical operator (the BFGS Value→ValueAndGradient
    // pair and repeated line-search probes). The fingerprint pins every operator
    // input, so a hit is bit-identical to a fresh build — cost and analytic
    // gradient are unchanged. See `AssembledOperatorCache`.
    let operator_fingerprint = assembled_operator_fingerprint(
        rho,
        beta_flat,
        &h_joint_unpen,
        &scaled_s_lambdas,
        robust_jeffreys_hphi_for_operator.as_ref(),
        ranges,
        total,
        scaled_joint_trace_diagonal_ridge,
        rho_curvature_scale,
        pseudo_logdet_mode,
    );
    let cached_operator = assembled_operator_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(operator_fingerprint));

    let hessian_op: Arc<dyn HessianOperator> = if let Some(cached) = cached_operator {
        log::debug!(
            "[OUTER hessian-route] reusing cached same-ρ assembled operator (fingerprint hit)"
        );
        cached
    } else {
        let built: Arc<dyn HessianOperator> = if use_joint_matrix_free_path(
            total,
            joint_observation_count(&inner.block_states),
        ) {
            let ranges_vec = ranges.to_vec();
            let s_lambdas = Arc::new(scaled_s_lambdas.clone());
            let trace_diagonal_ridge = scaled_joint_trace_diagonal_ridge
                + rho_curvature_scale * JOINT_TRACE_STABILITY_RIDGE;
            match &h_joint_unpen {
                JointHessianSource::Dense(h_joint) => {
                    let h_joint = Arc::new(h_joint.clone());
                    let apply_h = Arc::clone(&h_joint);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    let apply_hphi = robust_jeffreys_hphi_for_operator.clone();
                    let hphi_scale = rho_curvature_scale;
                    Arc::new(MatrixFreeSpdOperator::new_with_mode(
                        total,
                        move |v| {
                            let mut out = apply_h.dot(v);
                            let penalty = apply_joint_block_penalty(
                                &apply_ranges,
                                apply_s.as_ref(),
                                v,
                                trace_diagonal_ridge,
                                None,
                            );
                            out += &penalty;
                            if let Some(hphi) = apply_hphi.as_ref() {
                                let jeffreys = hphi.dot(v);
                                out.scaled_add(hphi_scale, &jeffreys);
                            }
                            out
                        },
                        pseudo_logdet_mode,
                    ))
                }
                JointHessianSource::Operator {
                    apply,
                    dense_forced,
                    ..
                } => {
                    let apply_h = Arc::clone(apply);
                    let apply_ranges = ranges_vec.clone();
                    let apply_s = Arc::clone(&s_lambdas);
                    let apply_hphi = robust_jeffreys_hphi_for_operator.clone();
                    let hphi_scale = rho_curvature_scale;
                    // Single-pass dense assembly of the SAME penalized
                    // operator `H_unpen + S_λ + scale·H_Φ`. When the
                    // operator source can structurally build its full dense
                    // `H_unpen` in one chunked BLAS-3 `XᵀWX` row pass
                    // (`dense_forced`), the LAML logdet factorization assembles
                    // it once here and adds the penalty/Jeffreys terms in
                    // O(p²) — instead of `total` canonical-basis matvecs, each
                    // a full n-row pass through `apply_h`. The matvec closure
                    // below is the exact same algebra column-for-column, so the
                    // materialized dense operator (and its logdet) are
                    // numerically identical; the direct build is preferred only
                    // when `dense_forced` actually yields a matrix.
                    let dense_forced = Arc::clone(dense_forced);
                    let dense_ranges = ranges_vec.clone();
                    let dense_s = Arc::clone(&s_lambdas);
                    let dense_hphi = robust_jeffreys_hphi_for_operator.clone();
                    let dense_assemble: Arc<dyn Fn() -> Option<Array2<f64>> + Send + Sync> =
                        Arc::new(move || {
                            let mut matrix = match dense_forced() {
                                Ok(Some(matrix)) => matrix,
                                Ok(None) => return None,
                                Err(error) => {
                                    log::warn!(
                                        "joint exact-newton dense_forced failed during outer logdet materialization: {error}"
                                    );
                                    return None;
                                }
                            };
                            if matrix.nrows() != total || matrix.ncols() != total {
                                return None;
                            }
                            add_joint_penalty_to_matrix(
                                &mut matrix,
                                &dense_ranges,
                                dense_s.as_ref(),
                                trace_diagonal_ridge,
                                None,
                            );
                            if let Some(hphi) = dense_hphi.as_ref() {
                                matrix.scaled_add(hphi_scale, hphi);
                            }
                            Some(matrix)
                        });
                    Arc::new(MatrixFreeSpdOperator::new_with_mode_and_dense_assemble(
                        total,
                        move |v| {
                            let mut out = match apply_h(v) {
                                Ok(out) => out,
                                Err(error) => {
                                    log::warn!(
                                        "joint exact-newton operator matvec failed during outer trace construction: {error}"
                                    );
                                    Array1::<f64>::from_elem(total, f64::NAN)
                                }
                            };
                            let penalty = apply_joint_block_penalty(
                                &apply_ranges,
                                apply_s.as_ref(),
                                v,
                                trace_diagonal_ridge,
                                None,
                            );
                            out += &penalty;
                            if let Some(hphi) = apply_hphi.as_ref() {
                                let jeffreys = hphi.dot(v);
                                out.scaled_add(hphi_scale, &jeffreys);
                            }
                            out
                        },
                        pseudo_logdet_mode,
                        Some(dense_assemble),
                    ))
                }
            }
        } else {
            let mut j_for_traces = materialize_joint_hessian_source(
                &h_joint_unpen,
                total,
                "joint exact-newton Hessian materialization",
            )?;
            add_joint_penalty_to_matrix(
                &mut j_for_traces,
                ranges,
                &scaled_s_lambdas,
                scaled_joint_trace_diagonal_ridge,
                None,
            );
            if let Some(hphi) = robust_jeffreys_hphi_for_operator.as_ref() {
                j_for_traces.scaled_add(rho_curvature_scale, hphi);
            }
            Arc::new(
                BlockCoupledOperator::from_joint_hessian_with_mode(
                    &j_for_traces,
                    pseudo_logdet_mode,
                )
                .map_err(|e| format!("BlockCoupledOperator from joint Hessian: {e}"))?,
            )
        };
        if let Ok(mut cache) = assembled_operator_cache().lock() {
            cache.insert(operator_fingerprint, Arc::clone(&built));
        }
        built
    };

    let (projected_logdet_correction, penalty_subspace_trace) = if project_hessian_logdet
        && include_logdet_h
        && include_logdet_s
        && pseudo_logdet_mode == PseudoLogdetMode::Smooth
    {
        let (projected_logdet, kernel) = joint_penalty_subspace_trace_parts(
            &h_joint_unpen,
            ranges,
            &scaled_s_lambdas,
            total,
            scaled_joint_trace_diagonal_ridge,
            scaled_robust_jeffreys_hphi.as_ref(),
        )?;
        let correction = projected_logdet - hessian_op.logdet();
        if kernel.is_some() {
            log::debug!(
                "[OUTER hessian-route] joint penalty subspace trace installed correction={:.6e}",
                correction
            );
        }
        (correction, kernel.map(Arc::new))
    } else {
        (0.0, None)
    };
    let hessian_logdet_correction = hessian_logdet_correction + projected_logdet_correction;

    let expected_theta_dim = rho.len()
        + ext_bundle
            .as_ref()
            .map(|bundle| bundle.coords.len())
            .unwrap_or(0);
    let has_penalty_subspace_trace = penalty_subspace_trace.is_some();

    // Option C: when the caller already has the batched first-order
    // logdet traces, let the unified VGH path keep all mode-response,
    // second-order, and Hessian work, but short-circuit only the
    // soon-discarded first-order trace calls. The projected-subspace
    // trace path is left untouched because the Hessian shares that
    // kernel and it is not routed through HessianOperator trace methods.
    // Bind the gating flag before `penalty_subspace_trace` is consumed by
    // the call below so the trace-skip choice does not depend on a moved
    // value (was: `if penalty_subspace_trace.is_none()` evaluated AFTER
    // the trace had already been forwarded to `unified_joint_cost_gradient`).
    let first_order_trace_skip = if penalty_subspace_trace.is_none() {
        first_order_trace_skip
    } else {
        None
    };
    let (objective, grad, outer_hessian) = unified_joint_cost_gradient(
        inner,
        specs,
        per_block,
        rho,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        penalty_subspace_trace,
        include_logdet_h,
        include_logdet_s,
        options,
        rho_prior,
        provider_box,
        eval_mode,
        ext_bundle.map(|bundle| bundle.scaled(rho_curvature_scale)),
        // Option C: when the caller already has the batched first-order
        // logdet traces, let the unified VGH path keep all mode-response,
        // second-order, and Hessian work, but short-circuit only the
        // soon-discarded first-order trace calls. The projected-subspace
        // trace path is left untouched because the Hessian shares that
        // kernel and it is not routed through HessianOperator trace methods.
        if has_penalty_subspace_trace {
            None
        } else {
            first_order_trace_skip
        },
        robust_jeffreys_phi,
    )?;
    if !objective.is_finite() {
        log::warn!(
            "joint outer evaluation produced non-finite objective: log_likelihood={} penalty_value={} block_logdet_h={} block_logdet_s={} include_logdet_h={} include_logdet_s={} rho_curvature_scale={}",
            inner.log_likelihood,
            inner.penalty_value,
            inner.block_logdet_h,
            inner.block_logdet_s,
            include_logdet_h,
            include_logdet_s,
            rho_curvature_scale,
        );
        return Err(CustomFamilyError::NumericalFailure {
            reason: "joint outer evaluation produced a non-finite objective".to_string(),
        }
        .into());
    }
    if grad.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: "joint outer evaluation produced a non-finite gradient".to_string(),
        }
        .into());
    }
    if grad.len() != expected_theta_dim {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "joint outer evaluation returned gradient length {}, expected {}",
                grad.len(),
                expected_theta_dim
            ),
        }
        .into());
    }
    match &outer_hessian {
        crate::solver::rho_optimizer::HessianResult::Analytic(hessian) => {
            if hessian.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: "joint outer evaluation produced a non-finite Hessian".to_string(),
                }
                .into());
            }
            if hessian.nrows() != expected_theta_dim || hessian.ncols() != expected_theta_dim {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "joint outer evaluation returned Hessian shape {}x{}, expected {}x{}",
                        hessian.nrows(),
                        hessian.ncols(),
                        expected_theta_dim,
                        expected_theta_dim
                    ),
                }
                .into());
            }
        }
        crate::solver::rho_optimizer::HessianResult::Operator(op) => {
            if op.dim() != expected_theta_dim {
                return Err(format!(
                    "joint outer evaluation returned operator Hessian dim {}, expected {}",
                    op.dim(),
                    expected_theta_dim
                ));
            }
        }
        crate::solver::rho_optimizer::HessianResult::Unavailable => {}
    }

    let warm = ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|st| st.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
        cached_inner: Some(cached_inner_mode_from_result(inner)),
    };

    Ok(OuterObjectiveEvalResult {
        objective,
        gradient: grad,
        outer_hessian,
        warm_start: warm,
        inner_converged: inner.converged,
    })
}

pub(crate) fn joint_outer_evaluate_efs(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    h_joint_unpen: JointHessianSource,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    moderidge: f64,
    extra_logdet_ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    include_logdet_h: bool,
    include_logdet_s: bool,
    strict_spd: bool,
    project_hessian_logdet: bool,
    options: &BlockwiseFitOptions,
    rho_prior: crate::types::RhoPrior,
    pseudo_logdet_mode: PseudoLogdetMode,
    compute_dh: &DriftDerivFn<'_>,
    compute_dh_many: Option<&DriftDerivManyFn<'_>>,
    compute_d2h: &DriftSecondDerivFn<'_>,
    compute_d2h_many: Option<&DriftSecondDerivManyFn<'_>>,
    owned_compute_dh: Option<
        Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>,
    >,
    owned_compute_dh_many: Option<
        Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, String> + Send + Sync>,
    >,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, String>
                + Send
                + Sync,
        >,
    >,
    owned_compute_d2h_many: Option<
        Arc<
            dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, String>
                + Send
                + Sync,
        >,
    >,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<crate::solver::rho_optimizer::EfsEval, String> {
    let joint_trace_diagonal_ridge = moderidge + if !strict_spd { extra_logdet_ridge } else { 0.0 };
    let scaled_joint_trace_diagonal_ridge = rho_curvature_scale * joint_trace_diagonal_ridge;

    let provider_box: Box<dyn HessianDerivativeProvider + '_> =
        if let (Some(owned_dh), Some(owned_d2h)) = (owned_compute_dh, owned_compute_d2h) {
            Box::new(OwnedJointDerivProvider {
                compute_dh: owned_dh,
                compute_dh_many: owned_compute_dh_many,
                compute_d2h: owned_d2h,
                compute_d2h_many: owned_compute_d2h_many,
                family_outer_hessian_operator: None,
            })
        } else {
            Box::new(BorrowedJointDerivProvider {
                compute_dh,
                compute_dh_many,
                compute_d2h,
                compute_d2h_many,
                family_outer_hessian_operator: None,
            })
        };

    let scaled_s_lambdas: Vec<Array2<f64>> = inner
        .s_lambdas
        .iter()
        .map(|matrix| {
            if rho_curvature_scale == 1.0 {
                matrix.clone()
            } else {
                matrix.mapv(|value| rho_curvature_scale * value)
            }
        })
        .collect();

    let hessian_op: Arc<dyn HessianOperator> = if use_joint_matrix_free_path(
        total,
        joint_observation_count(&inner.block_states),
    ) {
        let ranges_vec = ranges.to_vec();
        let s_lambdas = Arc::new(scaled_s_lambdas.clone());
        let trace_diagonal_ridge =
            scaled_joint_trace_diagonal_ridge + rho_curvature_scale * JOINT_TRACE_STABILITY_RIDGE;
        match &h_joint_unpen {
            JointHessianSource::Dense(h_joint) => {
                let h_joint = Arc::new(h_joint.clone());
                let apply_h = Arc::clone(&h_joint);
                let apply_ranges = ranges_vec.clone();
                let apply_s = Arc::clone(&s_lambdas);
                Arc::new(MatrixFreeSpdOperator::new_with_mode(
                    total,
                    move |v| {
                        let mut out = apply_h.dot(v);
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                            None,
                        );
                        out += &penalty;
                        out
                    },
                    pseudo_logdet_mode,
                ))
            }
            JointHessianSource::Operator {
                apply,
                dense_forced,
                ..
            } => {
                let apply_h = Arc::clone(apply);
                let apply_ranges = ranges_vec.clone();
                let apply_s = Arc::clone(&s_lambdas);
                // Single-pass dense assembly of the SAME penalized operator
                // `H_unpen + S_λ` (this fixed-point path carries no Jeffreys
                // term). One chunked BLAS-3 `XᵀWX` row pass via `dense_forced`
                // replaces `total` full-n canonical-basis matvecs for the LAML
                // logdet factorization; numerically identical to the matvec
                // reconstruction below.
                let dense_forced = Arc::clone(dense_forced);
                let dense_ranges = ranges_vec.clone();
                let dense_s = Arc::clone(&s_lambdas);
                let dense_assemble: Arc<dyn Fn() -> Option<Array2<f64>> + Send + Sync> = Arc::new(
                    move || {
                        let mut matrix = match dense_forced() {
                            Ok(Some(matrix)) => matrix,
                            Ok(None) => return None,
                            Err(error) => {
                                log::warn!(
                                    "joint exact-newton dense_forced failed during fixed-point logdet materialization: {error}"
                                );
                                return None;
                            }
                        };
                        if matrix.nrows() != total || matrix.ncols() != total {
                            return None;
                        }
                        add_joint_penalty_to_matrix(
                            &mut matrix,
                            &dense_ranges,
                            dense_s.as_ref(),
                            trace_diagonal_ridge,
                            None,
                        );
                        Some(matrix)
                    },
                );
                Arc::new(MatrixFreeSpdOperator::new_with_mode_and_dense_assemble(
                    total,
                    move |v| {
                        let mut out = match apply_h(v) {
                            Ok(out) => out,
                            Err(error) => {
                                log::warn!(
                                    "joint exact-newton operator matvec failed during fixed-point trace construction: {error}"
                                );
                                Array1::<f64>::from_elem(total, f64::NAN)
                            }
                        };
                        let penalty = apply_joint_block_penalty(
                            &apply_ranges,
                            apply_s.as_ref(),
                            v,
                            trace_diagonal_ridge,
                            None,
                        );
                        out += &penalty;
                        out
                    },
                    pseudo_logdet_mode,
                    Some(dense_assemble),
                ))
            }
        }
    } else {
        let mut j_for_traces = materialize_joint_hessian_source(
            &h_joint_unpen,
            total,
            "joint exact-newton Hessian materialization for fixed-point evaluation",
        )?;
        add_joint_penalty_to_matrix(
            &mut j_for_traces,
            ranges,
            &scaled_s_lambdas,
            scaled_joint_trace_diagonal_ridge,
            None,
        );
        Arc::new(
            BlockCoupledOperator::from_joint_hessian_with_mode(&j_for_traces, pseudo_logdet_mode)
                .map_err(|e| format!("BlockCoupledOperator from joint Hessian: {e}"))?,
        )
    };

    let (projected_logdet_correction, penalty_subspace_trace) = if project_hessian_logdet
        && include_logdet_h
        && include_logdet_s
        && pseudo_logdet_mode == PseudoLogdetMode::Smooth
    {
        let (projected_logdet, kernel) = joint_penalty_subspace_trace_parts(
            &h_joint_unpen,
            ranges,
            &scaled_s_lambdas,
            total,
            scaled_joint_trace_diagonal_ridge,
            None,
        )?;
        let correction = projected_logdet - hessian_op.logdet();
        if kernel.is_some() {
            log::debug!(
                "[OUTER hessian-route] joint EFS penalty subspace trace installed correction={:.6e}",
                correction
            );
        }
        (correction, kernel.map(Arc::new))
    } else {
        (0.0, None)
    };
    let hessian_logdet_correction = hessian_logdet_correction + projected_logdet_correction;

    unified_joint_efs_eval(
        inner,
        specs,
        per_block,
        rho,
        beta_flat,
        hessian_op,
        ranges,
        total,
        ridge,
        rho_curvature_scale,
        hessian_logdet_correction,
        penalty_subspace_trace,
        include_logdet_h,
        include_logdet_s,
        options,
        rho_prior,
        provider_box,
        ext_bundle.map(|bundle| bundle.scaled(rho_curvature_scale)),
    )
}

/// Evaluate the rho-only custom-family outer objective through the unified
/// joint hyperpath with no external ψ coordinates attached.
pub(crate) fn outerobjectivegradienthessian_internal<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: crate::types::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, String> {
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
    evaluate_custom_family_hyper_internal(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        &derivative_blocks,
        warm_start,
        rho_prior,
        eval_mode,
    )
    .map_err(String::from)
}

pub(crate) fn outerobjectiveefs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: crate::types::RhoPrior,
) -> Result<
    (
        crate::solver::rho_optimizer::EfsEval,
        ConstrainedWarmStart,
        bool,
    ),
    String,
> {
    let include_logdet_h = include_exact_newton_logdet_h(family, options);
    let include_logdet_s = include_exact_newton_logdet_s(family, options);
    let strict_spd = use_exact_newton_strict_spd(family);
    let per_block = split_log_lambdas(rho, penalty_counts)?;
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, warm_start)?;
    if !inner.converged {
        log::warn!(
            "[OUTER] custom-family EFS inner solve did not converge after {} cycle(s); \
             skipping EFS derivative assembly for theta_dim={}",
            inner.cycles,
            rho.len(),
        );
        return nonconverged_outer_efs_result(
            &inner,
            rho,
            rho.len(),
            include_logdet_h,
            include_logdet_s,
            "custom-family EFS non-converged inner solve",
        );
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.include_quadratic_penalty {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = if options.ridge_policy.include_penalty_logdet
        && !options.ridge_policy.include_quadratic_penalty
    {
        ridge
    } else {
        0.0
    };

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, end)| *end).unwrap_or(0);

    let efs_eval = {
        if let Some(joint_bundle) = build_joint_hessian_closures(
            family,
            &inner.block_states,
            specs,
            total,
            options,
            inner.joint_workspace.clone(),
        )? {
            let JointHessianBundle {
                source: h_joint_unpen,
                beta_flat,
                compute_dh,
                compute_dh_many,
                compute_d2h,
                compute_d2h_many,
                owned_compute_dh,
                owned_compute_dh_many,
                owned_compute_d2h,
                owned_compute_d2h_many,
                rho_curvature_scale,
                hessian_logdet_correction,
            } = joint_bundle;
            joint_outer_evaluate_efs(
                &inner,
                specs,
                &per_block,
                rho,
                &beta_flat,
                h_joint_unpen,
                &ranges,
                total,
                ridge,
                moderidge,
                extra_logdet_ridge,
                rho_curvature_scale,
                hessian_logdet_correction,
                include_logdet_h,
                include_logdet_s,
                strict_spd,
                family.use_projected_penalty_logdet(),
                options,
                rho_prior.clone(),
                family.pseudo_logdet_mode(),
                compute_dh.as_ref(),
                compute_dh_many.as_deref(),
                compute_d2h.as_ref(),
                compute_d2h_many.as_deref(),
                owned_compute_dh,
                owned_compute_dh_many,
                owned_compute_d2h,
                owned_compute_d2h_many,
                None,
            )
        } else {
            if family.requires_joint_outer_hyper_path() {
                return Err(
                        "outer hyper fixed-point evaluation requires a joint exact path for this family"
                            .to_string(),
                    );
            }
            if specs.len() != 1 {
                return Err(
                        "generic fixed-point outer fallback is only valid for single-block families; multi-block families must provide a joint outer path"
                            .to_string(),
                    );
            }

            let eval = family.evaluate(&inner.block_states)?;
            let block_idx = 0;
            let spec = &specs[block_idx];
            let work = &eval.blockworking_sets[block_idx];
            let p = spec.design.ncols();
            let mut diagonal_design = None::<DesignMatrix>;
            let h_joint_unpen = match work {
                BlockWorkingSet::Diagonal {
                    working_response: _,
                    working_weights,
                } => with_block_geometry(
                    family,
                    &inner.block_states,
                    spec,
                    block_idx,
                    |x_dyn, _| {
                        let w = floor_positiveworking_weights(working_weights, options.minweight);
                        let (xtwx, _) = weighted_normal_equations(x_dyn, &w, None)?;
                        diagonal_design = Some(x_dyn.clone());
                        Ok(xtwx)
                    },
                )?,
                BlockWorkingSet::ExactNewton {
                    gradient: _,
                    hessian,
                } => {
                    if hessian.nrows() != p || hessian.ncols() != p {
                        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                            "block {block_idx} exact-newton Hessian shape mismatch in fixed-point outer evaluation: got {}x{}, expected {}x{}",
                            hessian.nrows(),
                            hessian.ncols(),
                            p,
                            p
                        ) }.into());
                    }
                    hessian.to_dense()
                }
            };
            let beta_flat = inner.block_states[block_idx].beta.clone();
            let compute_dh = |direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
                if !include_logdet_h {
                    return Ok(None);
                }
                match work {
                    BlockWorkingSet::ExactNewton { .. } => {
                        match family.exact_newton_hessian_directional_derivative(
                            &inner.block_states,
                            block_idx,
                            direction,
                        )? {
                            Some(h_exact) => {
                                Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                                    h_exact,
                                    p,
                                    &format!(
                                        "block {block_idx} exact-newton dH shape mismatch in fixed-point outer evaluation"
                                    ),
                                )?)))
                            }
                            None => Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                                "missing exact-newton dH callback for block {block_idx} while fixed-point evaluation requires H_beta term"
                            ) }.into()),
                        }
                    }
                    BlockWorkingSet::Diagonal {
                        working_response: _,
                        working_weights,
                    } => {
                        let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                                    format!(
                                        "missing dynamic design for block {block_idx} diagonal fixed-point correction"
                                    )
                                })?;
                        let wwork =
                            floor_positiveworking_weights(working_weights, options.minweight);
                        let x_dense = x_dyn.to_dense();
                        let n = x_dense.nrows();

                        let mut d_eta = x_dyn.matrixvectormultiply(direction);
                        let geom = family.block_geometry_directional_derivative(
                            &inner.block_states,
                            block_idx,
                            spec,
                            direction,
                        )?;
                        let mut correction_mat = Array2::<f64>::zeros((p, p));

                        if let Some(geom_dir) = geom {
                            d_eta += &geom_dir.d_offset;
                            if let Some(dx) = geom_dir.d_design {
                                d_eta += &fast_av(&dx, &beta_flat);
                                let mut wx = x_dense.clone();
                                let mut wdx = dx.clone();
                                ndarray::Zip::from(wx.rows_mut())
                                    .and(wdx.rows_mut())
                                    .and(wwork.view())
                                    .par_for_each(|mut wxr, mut wdxr, &wi| {
                                        if wi != 1.0 {
                                            wxr.mapv_inplace(|v| v * wi);
                                            wdxr.mapv_inplace(|v| v * wi);
                                        }
                                    });
                                correction_mat += &fast_atb(&dx, &wx);
                                correction_mat += &fast_atb(&x_dense, &wdx);
                            }
                        }

                        let dw = family
                                    .diagonalworking_weights_directional_derivative(
                                        &inner.block_states,
                                        block_idx,
                                        &d_eta,
                                    )?
                                    .ok_or_else(|| {
                                        format!(
                                            "missing diagonal dW callback for block {block_idx} while fixed-point evaluation requires H_beta term"
                                        )
                                    })?;
                        if dw.len() != n {
                            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                                "block {block_idx} diagonal dW length mismatch in fixed-point outer evaluation: got {}, expected {}",
                                dw.len(),
                                n
                            ) }.into());
                        }
                        let mut scaled_x = x_dense.clone();
                        ndarray::Zip::from(scaled_x.rows_mut())
                            .and(&dw)
                            .par_for_each(|mut sr, &dwi| sr.mapv_inplace(|v| v * dwi));
                        correction_mat += &fast_atb(&x_dense, &scaled_x);

                        Ok(Some(DriftDerivResult::Dense(correction_mat)))
                    }
                }
            };
            let compute_d2h = |u: &Array1<f64>,
                               v: &Array1<f64>|
             -> Result<Option<DriftDerivResult>, String> {
                if !include_logdet_h {
                    return Ok(None);
                }
                match work {
                    BlockWorkingSet::ExactNewton { .. } => {
                        match family.exact_newton_hessian_second_directional_derivative(
                            &inner.block_states,
                            block_idx,
                            u,
                            v,
                        )? {
                            Some(h_exact) => {
                                Ok(Some(DriftDerivResult::Dense(symmetrized_square_matrix(
                                    h_exact,
                                    p,
                                    &format!(
                                        "block {block_idx} exact-newton d2H shape mismatch in fixed-point outer evaluation"
                                    ),
                                )?)))
                            }
                            None => Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                                "missing exact-newton d2H callback for block {block_idx} while fixed-point evaluation requires H_beta_beta term"
                            ) }.into()),
                        }
                    }
                    BlockWorkingSet::Diagonal { .. } => {
                        let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                            format!(
                                "missing dynamic design for block {block_idx} diagonal fixed-point second correction"
                            )
                        })?;
                        let x_dense = x_dyn.to_dense();
                        let n = x_dense.nrows();
                        let reject_second_order_geometry =
                            |label: &str,
                             geom: Option<BlockGeometryDirectionalDerivative>|
                             -> Result<(), String> {
                                if let Some(geom_dir) = geom {
                                    let has_offset =
                                        geom_dir.d_offset.iter().any(|value| *value != 0.0);
                                    if geom_dir.d_design.is_some() || has_offset {
                                        return Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                                            "block {block_idx} diagonal d2H requires second-order block-geometry derivatives for {label}; use an exact-newton or joint outer path"
                                        ) }.into());
                                    }
                                }
                                Ok(())
                            };
                        reject_second_order_geometry(
                            "first direction",
                            family.block_geometry_directional_derivative(
                                &inner.block_states,
                                block_idx,
                                spec,
                                u,
                            )?,
                        )?;
                        reject_second_order_geometry(
                            "second direction",
                            family.block_geometry_directional_derivative(
                                &inner.block_states,
                                block_idx,
                                spec,
                                v,
                            )?,
                        )?;
                        let d_eta_u = x_dyn.matrixvectormultiply(u);
                        let d_eta_v = x_dyn.matrixvectormultiply(v);
                        let d2w = family
                            .diagonalworking_weights_second_directional_derivative(
                                &inner.block_states,
                                block_idx,
                                &d_eta_u,
                                &d_eta_v,
                            )?
                            .ok_or_else(|| {
                                format!(
                                    "missing diagonal d2W callback for block {block_idx} while fixed-point evaluation requires H_beta_beta term"
                                )
                            })?;
                        if d2w.len() != n {
                            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                                "block {block_idx} diagonal d2W length mismatch in fixed-point outer evaluation: got {}, expected {}",
                                d2w.len(),
                                n
                            ) }.into());
                        }
                        let mut scaled_x = x_dense.clone();
                        ndarray::Zip::from(scaled_x.rows_mut())
                            .and(&d2w)
                            .par_for_each(|mut sr, &d2wi| sr.mapv_inplace(|value| value * d2wi));
                        Ok(Some(DriftDerivResult::Dense(fast_atb(&x_dense, &scaled_x))))
                    }
                }
            };
            joint_outer_evaluate_efs(
                &inner,
                specs,
                &per_block,
                rho,
                &beta_flat,
                JointHessianSource::Dense(h_joint_unpen),
                &ranges,
                total,
                ridge,
                moderidge,
                extra_logdet_ridge,
                1.0,
                0.0,
                include_logdet_h,
                include_logdet_s,
                strict_spd,
                family.use_projected_penalty_logdet(),
                options,
                rho_prior.clone(),
                family.pseudo_logdet_mode(),
                &compute_dh,
                None,
                &compute_d2h,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        }
    }?;

    let warm = ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|state| state.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
        cached_inner: Some(cached_inner_mode_from_result(&inner)),
    };

    Ok((efs_eval, warm, inner.converged))
}

pub(crate) fn normalize_outer_eval_error_detail(error: &str) -> &str {
    // Any `String` round-tripped through `CustomFamilyError::From<String>`
    // gets re-wrapped as `InvalidInput { context: "custom-family string
    // boundary", … }`, which `Display`s as `custom-family invalid input
    // in custom-family string boundary: <reason>`. Strip that "boundary"
    // wrapper first, then the historical bare `custom-family invalid
    // input: ` form, so the `last objective error: …` summary surfaces
    // the inner reason root cause once — not the doubly-wrapped form
    // that masked the synthetic-failure marker the outer-objective error
    // contract pins.
    let stripped = error
        .strip_prefix("custom-family invalid input in custom-family string boundary: ")
        .unwrap_or(error);
    stripped
        .strip_prefix("custom-family invalid input: ")
        .unwrap_or(stripped)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section: joint outer hyper surface — unified calculus for [rho, psi]
// ═══════════════════════════════════════════════════════════════════════════
//
// The callers have already applied the current spatial coordinates `psi` when
// constructing `family`, `specs`, and `derivative_blocks`, so the explicit
// input into the section below is still only the smoothing vector
// `rho_current`. Mathematically, however, the surface being differentiated
// is the full joint profiled/Laplace objective in
//
//     theta = [rho, psi].
//
// The exact outer calculus is unified across all hypercoordinates:
//
//     J(theta)
//     = V(beta^(theta), theta)
//       + 0.5 log|H(beta^(theta), theta)|
//       - 0.5 log|S(theta)|_+,
//
// with stationarity and joint curvature
//
//     F(beta, theta) := V_beta(beta, theta) = 0,
//     H(beta, theta) := V_beta_beta(beta, theta).
//
// For each theta_i we need the fixed-beta objects
//
//     V_i, g_i := F_i, H_i,
//
// and for each pair (i, j)
//
//     V_ij, g_ij, H_ij,
//
// together with the beta-curvature contractions
//
//     D_beta H[u], D_beta^2 H[u, v], T_i[u] := D_beta H_i[u].
//
// These determine the exact joint mode responses
//
//     beta_i  = -H^{-1} g_i,
//     beta_ij = -H^{-1}(g_ij + H_i beta_j + H_j beta_i + D_beta H[beta_i] beta_j),
//
// and the total Hessian drifts
//
//     dot H_i
//     = H_i + D_beta H[beta_i],
//
//     ddot H_ij
//     = H_ij
//       + T_i[beta_j]
//       + T_j[beta_i]
//       + D_beta H[beta_ij]
//       + D_beta^2 H[beta_i, beta_j].
//
// Therefore the exact joint outer derivatives are
//
//     J_i
//     = V_i
//       + 0.5 tr(H^{-1} dot H_i)
//       - 0.5 partial_i log|S(theta)|_+,
//
//     J_ij
//     = (V_ij - g_i^T H^{-1} g_j)
//       + 0.5 [ tr(H^{-1} ddot H_ij)
//               - tr(H^{-1} dot H_j H^{-1} dot H_i) ]
//       - 0.5 partial^2_{ij} log|S(theta)|_+.
//
// In this unified view rho and psi differ only in the likelihood-side
// fixed-beta derivative objects contributed by the family. The generic exact
// assembler always adds realized penalty motion through `S(theta)` for every
// hypercoordinate:
//
// - `rho` coordinates usually have zero likelihood-side objects and pick up
//   their fixed-beta derivatives entirely from `S_rho` / `S_{rho rho}`
// - `psi` coordinates contribute likelihood-side objects from the family's
//   joint exact psi hooks and may also pick up extra penalty terms through
//   `S_psi`, `S_{rho psi}`, and `S_{psi psi}` when realized penalties move
//   with `psi`
//
// The implementation below follows this unified calculus directly. Once a
// family supplies the joint fixed-beta psi objects and the mixed
// `D_beta H_psi[u]` contraction, exact joint hyper evaluation treats `rho`
// and `psi` identically and returns the full profiled/Laplace Hessian over
// `theta = [rho, psi]`.
//
// ═══════════════════════════════════════════════════════════════════════════
//  Unified HyperCoord builders for ψ coordinates
// ═══════════════════════════════════════════════════════════════════════════

/// Assemble the penalty derivative matrix S_ψ = Σ_k exp(ρ_k) ∂S_k/∂ψ
/// in the *block-local* coefficient space (p_block × p_block).
///
/// When the derivative carries multi-penalty components the sum iterates
/// over all `(penalty_idx, s_part)` pairs.  When only a single
/// `penalty_index` is stored the derivative `s_psi` is scaled by that
/// penalty's current lambda.  If neither is present, the derivative is
/// zero (the ψ coordinate does not move any realized penalty).
pub(crate) fn assemble_block_local_s_psi(
    deriv: &CustomFamilyBlockPsiDerivative,
    per_block_rho: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref components) = deriv.s_psi_penalty_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s_part.add_scaled_to(per_block_rho[*penalty_idx].exp(), &mut s);
        }
        return s;
    }
    if let Some(ref components) = deriv.s_psi_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s.scaled_add(per_block_rho[*penalty_idx].exp(), s_part);
        }
        s
    } else if let Some(penalty_idx) = deriv.penalty_index {
        deriv.s_psi.mapv(|v| per_block_rho[penalty_idx].exp() * v)
    } else {
        Array2::<f64>::zeros((p_block, p_block))
    }
}

/// Assemble the second penalty derivative matrix S_{ψ_i ψ_j} in block-local
/// coefficient space.
///
/// This mirrors the psi/psi branch of `joint_theta_penaltysecond_matrix` but
/// returns the block-local matrix directly instead of embedding it into the
/// full flattened coefficient space.
pub(crate) fn assemble_block_local_s_psi_psi(
    deriv_i: &CustomFamilyBlockPsiDerivative,
    local_j: usize,
    per_block_rho: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref parts) = deriv_i.s_psi_psi_penalty_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        if let Some(pair_parts) = parts.get(local_j) {
            for (penalty_idx, s_part) in pair_parts {
                s_part.add_scaled_to(per_block_rho[*penalty_idx].exp(), &mut s);
            }
        }
        return s;
    }
    if let Some(ref parts) = deriv_i.s_psi_psi_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        if let Some(pair_parts) = parts.get(local_j) {
            for (penalty_idx, s_part) in pair_parts {
                s.scaled_add(per_block_rho[*penalty_idx].exp(), s_part);
            }
        }
        s
    } else if let Some(ref parts) = deriv_i.s_psi_psi {
        if let Some(s_part) = parts.get(local_j) {
            if let Some(penalty_index) = deriv_i.penalty_index {
                s_part.mapv(|v| per_block_rho[penalty_index].exp() * v)
            } else {
                Array2::<f64>::zeros((p_block, p_block))
            }
        } else {
            Array2::<f64>::zeros((p_block, p_block))
        }
    } else {
        Array2::<f64>::zeros((p_block, p_block))
    }
}

#[derive(Clone)]
pub struct BlockwiseInnerResult {
    pub block_states: Vec<ParameterBlockState>,
    pub active_sets: Vec<Option<Vec<usize>>>,
    pub log_likelihood: f64,
    pub penalty_value: f64,
    pub cycles: usize,
    pub converged: bool,
    pub block_logdet_h: f64,
    pub block_logdet_s: f64,
    /// Cached assembled penalty matrices S(ρ) = Σ_k exp(ρ_k) S_k per block.
    /// Avoids redundant re-assembly in the outer objective evaluation.
    pub s_lambdas: Vec<Array2<f64>>,
    pub joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    /// Projected KKT residual at the converged inner iterate, propagated to
    /// the unified evaluator's `InnerAssembly::kkt_residual` for the
    /// outer REML/LAML scoring path. `None` when the solver path doesn't
    /// produce a typed KKT diagnostic (blockwise NR fallback, eager-stop).
    pub kkt_residual: Option<ProjectedKktResidual>,
    /// Active linear-inequality constraint rows at the converged inner
    /// iterate. When `Some`, the unified evaluator builds the
    /// constraint-aware kernel `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`
    /// for per-coordinate mode responses `v_k = ∂β/∂ρ_k`.
    pub active_constraints: Option<Arc<ActiveLinearConstraintBlock>>,
}

impl std::fmt::Debug for BlockwiseInnerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockwiseInnerResult")
            .field("block_states", &self.block_states)
            .field("active_sets", &self.active_sets)
            .field("log_likelihood", &self.log_likelihood)
            .field("penalty_value", &self.penalty_value)
            .field("cycles", &self.cycles)
            .field("converged", &self.converged)
            .field("block_logdet_h", &self.block_logdet_h)
            .field("block_logdet_s", &self.block_logdet_s)
            .field("s_lambdas", &self.s_lambdas)
            .field(
                "joint_workspace",
                &self.joint_workspace.as_ref().map(|_| "<workspace>"),
            )
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct ConstrainedWarmStart {
    pub(crate) rho: Array1<f64>,
    pub(crate) block_beta: Vec<Array1<f64>>,
    pub(crate) active_sets: Vec<Option<Vec<usize>>>,
    pub(crate) cached_inner: Option<CachedInnerMode>,
}

#[derive(Clone)]
pub(crate) struct CachedInnerMode {
    pub(crate) log_likelihood: f64,
    pub(crate) penalty_value: f64,
    pub(crate) cycles: usize,
    pub(crate) converged: bool,
    pub(crate) block_logdet_h: f64,
    pub(crate) block_logdet_s: f64,
    pub(crate) joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    pub(crate) kkt_residual: Option<ProjectedKktResidual>,
    pub(crate) active_constraints: Option<Arc<ActiveLinearConstraintBlock>>,
}
