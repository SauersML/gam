//! Inner-assembly construction, the unified joint cost/gradient/EFS
//! evaluators, the assembled-operator cache, the joint outer-evaluate
//! entry points, and block-local penalty assembly, split out of
//! `outer_objective.rs` by concern (#1145). Re-exported via `custom_family`.

use super::*;

/// Maximum joint dimension for which the gam#1395 logdet collapse guard rebuilds
/// the ground-truth penalized Hessian and runs an O(p³) reference
/// eigendecomposition to cross-check the assembled operator's `log|H|`. Above
/// this the dense reference would dominate the per-outer-eval cost, so the guard
/// is skipped (the matrix-free regime is the one whose logdet already routes
/// through the same exact dense materialization when the byte budget allows).
const JOINT_LOGDET_GUARD_MAX_DIM: usize = 64;

/// Build the canonical unified REML/LAML assembly for a custom-family outer
/// evaluation.
pub(crate) fn build_custom_family_inner_assembly<'dp>(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn HessianFactorization>,
    // #2612: the operator the IFT mode response `v_k = ∂β̂/∂ρ_k` is solved
    // against, when that is not `hessian_op`. `None` on every lane whose logdet
    // is priced on the inner objective's own Hessian.
    mode_response_op: Option<Arc<dyn HessianFactorization>>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    rho_prior: gam_problem::RhoPrior,
    deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,
    ext_bundle: Option<ExtCoordBundle>,
    firth_value: Option<f64>,
) -> Result<
    (
        gam_solve::estimate::reml::assembly::InnerAssembly<'dp>,
        usize,
        Vec<f64>,
    ),
    CustomFamilyError,
> {
    use gam_problem::PenaltyCoordinate;
    use gam_solve::estimate::reml::assembly::{
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
    let mut penalty_coords = penalty_coords_from_blocks(&block_descs, total)?;

    // Compute penalty logdet derivatives.
    let mut per_block_penalties: Vec<&[Array2<f64>]> = per_block_penalties_dense
        .iter()
        .map(|v| v.as_slice())
        .collect();
    let penalty_logdet_ridge = if options.ridge_policy.accounts_for_objective() {
        ridge
    } else {
        0.0
    };

    // gam#1587: append the full-width joint penalties as one extra pseudo-block.
    // Each `M⊗S_t` becomes a `PenaltyCoordinate::DenseRoot` (dim == total) at the
    // outer ρ slot following the per-block coords, and the same matrices form one
    // coupled logdet block `log|Σ_t λ_t M⊗S_t|₊`. The evaluator's positional
    // contract (penalty_coords.len() == rho_slice.len() == penalty_logdet.first
    // .len()) is preserved by extending all three in lock-step. Empty for every
    // family without joint penalties — `joint_log_lambdas` is then empty and the
    // assembly is byte-identical to the per-block-only path.
    let joint_bundle = options.joint_penalties.as_deref();
    let joint_log_lambdas: Vec<f64> = joint_bundle
        .map(|b| b.log_lambdas().to_vec())
        .unwrap_or_default();
    let joint_penalty_matrices: Vec<Array2<f64>> = joint_bundle
        .map(|b| b.specs().iter().map(|s| s.matrix.clone()).collect())
        .unwrap_or_default();
    let joint_penalty_roots = joint_bundle.map(|bundle| bundle.roots()).unwrap_or(&[]);
    if joint_penalty_roots.len() != joint_penalty_matrices.len() {
        return Err(CustomFamilyError::trial_point(format!(
            "joint penalty geometry mismatch: {} matrices vs {} cached roots",
            joint_penalty_matrices.len(),
            joint_penalty_roots.len(),
        )));
    }
    for root in joint_penalty_roots {
        penalty_coords.push(PenaltyCoordinate::from_dense_root(root.clone()));
    }
    // Hierarchical coefficient-group penalties are INDEPENDENT Gaussian prior
    // factors, not additive pieces of one smooth prior: their evidence
    // normalizer is per-factor `Σ_k(rank Sₖ·ρₖ + log|Sₖ|₊)` rather than the
    // coalesced `log|Σ λₖSₖ|₊` of a multi-penalty smooth. The two differ
    // exactly when factors overlap (each shared dimension loses ½·log λ from
    // the coalesced form), so penalties whose precision label is declared in
    // `options.independent_prior_factor_labels` are routed to their own
    // singleton pseudo-logdet block. `None` for every ordinary fit.
    let prior_factor_masks: Option<Vec<Vec<bool>>> =
        if options.independent_prior_factor_labels.is_empty() {
            None
        } else {
            Some(
                specs
                    .iter()
                    .map(|spec| {
                        spec.penalties
                            .iter()
                            .map(|penalty| {
                                penalty.precision_label().is_some_and(|label| {
                                    options
                                        .independent_prior_factor_labels
                                        .iter()
                                        .any(|factor| factor == label)
                                })
                            })
                            .collect()
                    })
                    .collect(),
            )
        };
    let per_block_with_joint: Vec<Array1<f64>>;
    let masks_with_joint: Option<Vec<Vec<bool>>>;
    let penalty_logdet = if joint_penalty_matrices.is_empty() {
        compute_block_penalty_logdet_derivs_with_prior_factors(
            per_block,
            &per_block_penalties,
            prior_factor_masks.as_deref(),
            penalty_logdet_ridge,
        )?
    } else {
        // Append the joint pseudo-block to the per-block rho list and penalty list
        // so its logdet value / ρ-derivatives slot in after the per-block coords.
        // Joint penalties are one coupled Gaussian prior — never prior factors.
        per_block_with_joint = per_block
            .iter()
            .cloned()
            .chain(std::iter::once(Array1::from(joint_log_lambdas.clone())))
            .collect();
        per_block_penalties.push(joint_penalty_matrices.as_slice());
        masks_with_joint = prior_factor_masks.map(|mut masks| {
            masks.push(vec![false; joint_penalty_matrices.len()]);
            masks
        });
        compute_block_penalty_logdet_derivs_with_prior_factors(
            &per_block_with_joint,
            &per_block_penalties,
            masks_with_joint.as_deref(),
            penalty_logdet_ridge,
        )?
    };

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
        mode_response_op,
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
        // Display boundary (gam#2689): `InnerAssembly` is a gam-solve struct
        // whose hyper-pair callbacks are declared over `Result<_, String>`, so
        // the typed error is rendered HERE, explicitly, rather than by a silent
        // blanket `From` at every `?` inside the closures.
        ext_coord_pair_fn: ext_coord_pair_fn.map(|f| {
            Box::new(move |i: usize, j: usize| f(i, j).map_err(|error| error.to_string()))
                as Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, String> + Send + Sync>
        }),
        rho_ext_pair_fn: rho_ext_pair_fn.map(|f| {
            Box::new(move |i: usize, j: usize| f(i, j).map_err(|error| error.to_string()))
                as Box<dyn Fn(usize, usize) -> Result<HyperCoordPair, String> + Send + Sync>
        }),
        fixed_drift_deriv,
        contracted_psi_second_order: contracted_psi_fn,
        kkt_residual: inner.kkt_residual.clone(),
        active_constraints: inner.active_constraints.clone(),
    };

    Ok((evaluator, ext_dim, joint_log_lambdas))
}

pub(crate) struct FirstOrderTraceSkipOperator {
    pub(crate) inner: Arc<dyn HessianFactorization>,
    pub(crate) remaining_first_order_traces: AtomicUsize,
}

impl FirstOrderTraceSkipOperator {
    pub(crate) fn new(inner: Arc<dyn HessianFactorization>, skip_count: usize) -> Self {
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

impl HessianFactorization for FirstOrderTraceSkipOperator {
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

    /// Always the inner backend's, skip list or not (gam#979).
    ///
    /// This wrapper exists to serve a precomputed list of FIRST-ORDER TRACES,
    /// and `as_exact_dense_spectral` above is gated because the eigenbasis
    /// batch route computes traces and would walk straight past that list. A
    /// dense stationarity curvature computes no trace and cannot bypass
    /// anything: its one consumer is `try_tangent_projected_evaluate`, which
    /// needs `Z' M Z` for the mode response at an ACTIVE-CONSTRAINT iterate.
    ///
    /// Refusing it there does not fall back to something slower — it refuses
    /// the whole trial point as infeasible. On the large-scale CTN
    /// preprocessor the cone constraints are active at nearly every trial
    /// point, so the outer κ search burned entire BFGS restarts on probes
    /// declined for a capability the backend underneath this wrapper has.
    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        self.inner.assemble_h_dense_for_tangent_projection()
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
    hessian_op: Arc<dyn HessianFactorization>,
    // #2612: see `build_custom_family_inner_assembly`.
    mode_response_op: Option<Arc<dyn HessianFactorization>>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    rho_prior: gam_problem::RhoPrior,
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
        gam_problem::HessianValue,
        [f64; 4],
        Option<Array2<f64>>,
    ),
    CustomFamilyError,
> {
    let trace_skip = first_order_trace_skip
        .as_ref()
        .map(|v| v.len() as i64)
        .unwrap_or(-1);
    let hessian_op: Arc<dyn HessianFactorization> = match first_order_trace_skip.as_ref() {
        Some(trace_values) if !trace_values.is_empty() => Arc::new(
            FirstOrderTraceSkipOperator::new(hessian_op, trace_values.len()),
        ),
        _ => hessian_op,
    };
    let (evaluator, ext_dim, joint_log_lambdas) = build_custom_family_inner_assembly(
        inner,
        specs,
        per_block,
        beta_flat,
        hessian_op,
        mode_response_op,
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
    // gam#1587: the evaluator's penalty coords are per-block coords followed by
    // the appended joint coords, so the rho slice must carry the joint λ tail
    // (`[per_block_rho ; joint_log_lambdas]`). Empty joint tail ⇒ unchanged.
    let n_joint = joint_log_lambdas.len();
    let rho_with_joint: Array1<f64> = if n_joint == 0 {
        rho.clone()
    } else {
        rho.iter()
            .copied()
            .chain(joint_log_lambdas.iter().copied())
            .collect()
    };
    let rho_slice = rho_with_joint
        .as_slice()
        .ok_or_else(|| "outer rho vector must be contiguous".to_string())?;
    let first_order_trace_correction = first_order_trace_skip.map(|trace_values| {
        // Append zero corrections for the joint coords so the correction vector
        // aligns with the extended penalty-coordinate / rho length.
        let mut gradient_correction = trace_values.mapv(|trace| 0.5 * trace);
        if n_joint > 0 {
            let mut extended = Array1::<f64>::zeros(gradient_correction.len() + n_joint);
            extended
                .slice_mut(ndarray::s![..gradient_correction.len()])
                .assign(&gradient_correction);
            gradient_correction = extended;
        }
        (0.0, gradient_correction, None)
    });
    let result = evaluator.evaluate(rho_slice, eval_mode, first_order_trace_correction)?;

    let cost = result.cost;
    let criterion_components = [
        result.criterion_components.fixed_beta,
        result.criterion_components.logdet_h,
        result.criterion_components.logdet_s,
        result.criterion_components.kkt,
    ];
    let gradient_present = result.gradient.is_some();
    let gradient = result
        .gradient
        .unwrap_or_else(|| Array1::zeros(rho.len() + n_joint + ext_dim));
    log::debug!(
        "[UNIFIED-GRAD] mode={eval_mode:?} present={gradient_present} trace_skip={trace_skip} \
         |g|={:.6e} len={} n_joint={n_joint} rho_len={} ext={ext_dim}",
        gradient.iter().map(|g| g * g).sum::<f64>().sqrt(),
        gradient.len(),
        rho.len(),
    );

    let hessian = result.hessian;
    let ext_mode_response_cols = result.ext_mode_response_cols;

    Ok((
        cost,
        gradient,
        hessian,
        criterion_components,
        ext_mode_response_cols,
    ))
}

pub(crate) fn unified_joint_efs_eval(
    inner: &BlockwiseInnerResult,
    specs: &[ParameterBlockSpec],
    per_block: &[Array1<f64>],
    rho: &Array1<f64>,
    beta_flat: &Array1<f64>,
    hessian_op: Arc<dyn HessianFactorization>,
    ranges: &[(usize, usize)],
    total: usize,
    ridge: f64,
    rho_curvature_scale: f64,
    hessian_logdet_correction: f64,
    penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,
    include_logdet_h: bool,
    include_logdet_s: bool,
    options: &BlockwiseFitOptions,
    rho_prior: gam_problem::RhoPrior,
    deriv_provider: Box<dyn HessianDerivativeProvider + '_>,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<gam_problem::EfsEval, CustomFamilyError> {
    let (assembly, _, joint_log_lambdas) = build_custom_family_inner_assembly(
        inner,
        specs,
        per_block,
        beta_flat,
        hessian_op,
        // The EFS screening path evaluates the Φ-less criterion, so its logdet
        // operator IS its inner objective's Hessian and the mode response has
        // nothing to separate from (#2612).
        None,
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
    // gam#1587: extend the rho slice with the joint λ tail to match the appended
    // joint penalty coords (empty for every non-joint family ⇒ unchanged).
    let rho_with_joint: Array1<f64> = if joint_log_lambdas.is_empty() {
        rho.clone()
    } else {
        rho.iter()
            .copied()
            .chain(joint_log_lambdas.iter().copied())
            .collect()
    };
    let rho_slice = rho_with_joint
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
    let result = gam_solve::estimate::reml::assembly::evaluate_solution(
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
            hessian_factorization_geometric_scale(inner_solution.hessian_op.as_ref());
        let hybrid = compute_hybrid_efs_update(&inner_solution, rho_slice, gradient_slice)?;
        Ok(gam_problem::EfsEval {
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
            consecutive_restored_incumbents: None,
        })
    } else {
        let inner_hessian_scale =
            hessian_factorization_geometric_scale(inner_solution.hessian_op.as_ref());
        Ok(gam_problem::EfsEval {
            cost: result.cost,
            steps: compute_efs_update(&inner_solution, rho_slice, gradient_slice)?,
            beta: Some(inner_solution.beta.clone()),
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale,
            logdet_enclosure_gap: None,
            consecutive_restored_incumbents: None,
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
/// assembled `Arc<dyn HessianFactorization>` keyed by a content fingerprint of ALL of
/// those inputs and reuse it on a bit-identical hit. Because the reuse condition
/// is exact byte-equality of the build inputs, the reused operator — and so the
/// LAML cost and its analytic gradient — is bit-identical to a fresh build.
/// On a miss we build + store, evicting the older of the (at most) two retained
/// entries, which bounds memory to the last two distinct ρ assemblies.
struct AssembledOperatorCache {
    /// `(fingerprint, operator)` for at most the last two distinct assemblies.
    entries: Vec<(u64, Arc<dyn HessianFactorization>)>,
}

impl AssembledOperatorCache {
    const CAPACITY: usize = 2;

    fn get(&self, fingerprint: u64) -> Option<Arc<dyn HessianFactorization>> {
        self.entries
            .iter()
            .find(|(key, _)| *key == fingerprint)
            .map(|(_, op)| Arc::clone(op))
    }

    fn insert(&mut self, fingerprint: u64, op: Arc<dyn HessianFactorization>) {
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
    scaled_joint_penalty: Option<&Array2<f64>>,
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
    // gam#1587/#561: the full-width joint penalty `Σ_t λ_t (M⊗S_t)` is part of
    // the assembled operator `H + S_λ`, but its λ live in the OUTER ρ
    // coordinates, NOT in the physical `rho` hashed above (which is empty when
    // the family rides entirely on joint penalties, e.g. multinomial). Without
    // hashing the joint penalty's content, two outer evaluations at different
    // joint λ but a coincident β̂ collide on the same fingerprint and the second
    // reuses the first's STALE operator — the silent gam#1395 logdet divergence.
    // Hash presence + content so the cache key tracks the joint λ exactly.
    match scaled_joint_penalty {
        Some(joint) => {
            1u8.hash(&mut hasher);
            joint.dim().hash(&mut hasher);
            for &v in joint.iter() {
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
    rho_prior: gam_problem::RhoPrior,
    pseudo_logdet_mode: PseudoLogdetMode,
    compute_dh: &DriftDerivFn<'_>,
    compute_dh_many: Option<&DriftDerivManyFn<'_>>,
    compute_d2h: &DriftSecondDerivFn<'_>,
    compute_d2h_many: Option<&DriftSecondDerivManyFn<'_>>,
    owned_compute_dh: Option<
        Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, CustomFamilyError> + Send + Sync>,
    >,
    owned_compute_dh_many: Option<
        Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, CustomFamilyError> + Send + Sync>,
    >,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, CustomFamilyError>
                + Send
                + Sync,
        >,
    >,
    owned_compute_d2h_many: Option<
        Arc<
            dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, CustomFamilyError>
                + Send
                + Sync,
        >,
    >,
    ext_bundle: Option<ExtCoordBundle>,
    first_order_trace_skip: Option<Array1<f64>>,
    batched_outer_hessian_operator: Option<Arc<dyn gam_problem::HessianOperator>>,
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
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
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
    //
    // #2765: the log-determinant operator must NEVER carry a term whose drift
    // the gradient cannot produce — on ANY route, not merely on the routes that
    // happen to hand the scalar to a different object.
    //
    // The completion was folded into `hessian_op` whenever the projected-logdet
    // route was going to own the value and the traces, on the reasoning that
    // the operator is then used only for solves and its `logdet()` cancels
    // exactly against `projected_logdet − hessian_op.logdet()`. That reasoning
    // is sound and it is also a PRECONDITION a downstream lane can invalidate,
    // which is what happens on an active inequality face: the tangent-projected
    // evaluator drops the projected kernel (correctly — a p-space subspace
    // kernel does not act on an m-dimensional face) and takes its determinant
    // from `hessian_op`'s dense assembly directly. The completion is then in
    // the VALUE with nothing to differentiate it, and the outer gradient is
    // short by `½tr(K · D_β[completion][v])` — a third directional derivative
    // no family exposes, so the term is not merely missing but unobtainable.
    // Measured on #2765 as the residual `logdet_h` gap after the correction
    // pairing was fixed: 4.4% on both ρ coordinates, identical to four digits,
    // and linear in `v` exactly as `D_β[·][v]` must be.
    //
    // So the completion goes to the IFT operator unconditionally, which is the
    // #2612 separation applied as an invariant instead of as a route-dependent
    // convenience. `hessian_op` is the value/trace object everywhere, and no
    // later lane can make it one by surprise.

    // The IFT operator differentiates the SELECTED INNER STATIONARITY system.
    // Its curvature is therefore the complete coefficient Hessian
    //
    //     M_true = H + S_λ + H_Φ + completion,
    //
    // not the divided-difference `H_Φ` used by the deliberately distinct
    // projected-logdet value/trace kernel. Whether `H_Φ + completion` is PSD
    // in isolation is irrelevant: an additive component of a valid Hessian may
    // be indefinite, and replacing it because of that changes `dβ̂/dθ`.
    // gam#2460 measured exactly that desynchronization on the selected Matérn
    // profile (6.0% / 22.1% coefficient-response error). The full penalized
    // operator below is the object whose solve decides the mode response.
    //
    // When no projected logdet owns the scalar criterion, `hessian_op` remains
    // the value/trace object and completion stays out: introducing it there
    // without its third-order drift would define a different scalar. Under the
    // projected route, however, the scalar and IFT kernels are already separate,
    // so every available completion belongs in the IFT operator unconditionally.
    let robust_jeffreys_hphi_for_operator: Option<Array2<f64>> = robust_jeffreys_hphi.clone();
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

    // gam#1587: the reconstructed outer `H_pen = H + S_λ` operator below adds
    // only the per-block `scaled_s_lambdas`. When a full-width joint penalty is
    // active (the centered `M⊗S_t` multinomial smoothing), its
    // `rho_curvature_scale · Σ_t λ_t S_joint_t` must ALSO be folded in, or the
    // LAML `log|H_pen|` and the trace kernel `K = H_pen⁻¹` disagree with the
    // inner-converged penalized Hessian (which DID include it). Precompute the
    // dense scaled joint penalty once (`total × total`); `None` for every family
    // without joint penalties keeps every operator path byte-identical.
    let scaled_joint_penalty: Option<Array2<f64>> =
        options.joint_penalties.as_deref().and_then(|bundle| {
            if bundle.is_empty() {
                return None;
            }
            let mut matrix = Array2::<f64>::zeros((total, total));
            bundle.add_to_matrix(&mut matrix);
            if rho_curvature_scale != 1.0 {
                matrix.mapv_inplace(|value| rho_curvature_scale * value);
            }
            Some(matrix)
        });

    // Assemble `H_unpen + S_λ + scale · J` for a caller-chosen Jeffreys curvature
    // `J`. Everything except `J` is fixed for this evaluation point, so the two
    // objects the outer criterion needs — the logdet/trace object (`J = H_Φ`) and
    // the IFT mode-response object (`J = H_Φ + completion`, the TRUE Hessian of
    // the Φ-augmented inner objective) — differ only in that argument.
    //
    // Reuse the assembled operator (and its lazily-built spectral factorization)
    // when an immediately-prior eval at the SAME ρ/β̂/curvature assembled the
    // bit-identical operator (the BFGS Value→ValueAndGradient pair and repeated
    // line-search probes). The fingerprint pins every operator input INCLUDING
    // `J`, so the two objects never collide in the cache and a hit is
    // bit-identical to a fresh build. See `AssembledOperatorCache`.
    let assemble_operator = |jeffreys_for_operator: Option<&Array2<f64>>| -> Result<
        Arc<dyn HessianFactorization>,
        CustomFamilyError,
    > {
        let operator_fingerprint = assembled_operator_fingerprint(
            rho,
            beta_flat,
            &h_joint_unpen,
            &scaled_s_lambdas,
            scaled_joint_penalty.as_ref(),
            jeffreys_for_operator,
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

        let assembled: Arc<dyn HessianFactorization> = if let Some(cached) = cached_operator {
            log::debug!(
                "[OUTER hessian-route] reusing cached same-ρ assembled operator (fingerprint hit)"
            );
            cached
        } else {
            let built: Arc<dyn HessianFactorization> = if use_joint_matrix_free_path(
                total,
                joint_observation_count(&inner.block_states),
            ) {
                let ranges_vec = ranges.to_vec();
                let s_lambdas = Arc::new(scaled_s_lambdas.clone());
                // gam#1587: full-width joint penalty (already scaled by
                // `rho_curvature_scale`), folded into every operator path below.
                let joint_penalty_arc: Option<Arc<Array2<f64>>> =
                    scaled_joint_penalty.clone().map(Arc::new);
                let trace_diagonal_ridge = scaled_joint_trace_diagonal_ridge
                    + rho_curvature_scale * JOINT_TRACE_STABILITY_RIDGE;
                match &h_joint_unpen {
                    JointHessianSource::Dense(h_joint) => {
                        let h_joint = Arc::new(h_joint.clone());
                        let apply_h = Arc::clone(&h_joint);
                        let apply_ranges = ranges_vec.clone();
                        let apply_s = Arc::clone(&s_lambdas);
                        let apply_hphi = jeffreys_for_operator.cloned();
                        let apply_joint = joint_penalty_arc.clone();
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
                                if let Some(joint) = apply_joint.as_ref() {
                                    out += &joint.dot(v);
                                }
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
                        let apply_hphi = jeffreys_for_operator.cloned();
                        let apply_joint = joint_penalty_arc.clone();
                        let dense_joint = joint_penalty_arc.clone();
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
                        let dense_hphi = jeffreys_for_operator.cloned();
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
                                if let Some(joint) = dense_joint.as_ref() {
                                    matrix += joint.as_ref();
                                }
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
                                if let Some(joint) = apply_joint.as_ref() {
                                    out += &joint.dot(v);
                                }
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
                if let Some(joint) = scaled_joint_penalty.as_ref() {
                    j_for_traces += joint;
                }
                if let Some(hphi) = jeffreys_for_operator {
                    j_for_traces.scaled_add(rho_curvature_scale, hphi);
                }
                // gam#1395/#1854: `BlockCoupledOperator::from_joint_hessian_with_mode`
                // eigendecomposes via `eigh(Side::Lower)`, which reads ONLY the lower
                // triangle and assumes the input is already symmetric. The assembled
                // `H_unpen + S_λ + scale·H_Φ` is symmetric in exact arithmetic, but
                // reduction-order f.p. noise desyncs mirror entries — and on the
                // multinomial Firth/Jeffreys path the divided-difference `H_Φ` (plus
                // its second-order completion) carries an `O(1e10)` curvature scale in
                // the near-separation regime, so that asymmetry is large enough that
                // reading the raw lower triangle yields a materially different spectrum
                // (and logdet) than the symmetrized matrix. That is exactly the gam#1395
                // logdet-collapse the ground-truth guard below detects, because the
                // guard reconstructs the SAME matrix but symmetrizes it first (as does
                // the matrix-free dense-assemble path). Symmetrize here too so every
                // route feeds `from_symmetric_with_mode` the identical symmetric matrix
                // and the operator realizes the penalized joint Hessian consistently.
                symmetrize_dense_in_place(&mut j_for_traces);
                Arc::new(
                    BlockCoupledOperator::from_joint_hessian_with_mode(
                        &j_for_traces,
                        pseudo_logdet_mode,
                    )
                    .map_err(|e| CustomFamilyError::trial_point(format!("BlockCoupledOperator from joint Hessian: {e}")))?,
                )
            };
            if let Ok(mut cache) = assembled_operator_cache().lock() {
                cache.insert(operator_fingerprint, Arc::clone(&built));
            }
            built
        };
        Ok(assembled)
    };

    let hessian_op = assemble_operator(robust_jeffreys_hphi_for_operator.as_ref())?;

    // THE IFT OPERATOR IS NOT THE LOGDET OPERATOR (#2612).
    //
    // `v_k = ∂β̂/∂ρ_k` is a property of the inner stationarity system, so it
    // solves against `M_true = H + S_λ + H_Φ + completion` — the exact Hessian
    // of the Φ-augmented objective the inner Newton converged on. The logdet
    // VALUE and its trace kernel must instead share ONE object, and that object
    // is the divided-difference `M_DD = H + S_λ + H_Φ`: folding the completion
    // into the scalar would require the completion's own β-drift (third
    // directional derivatives no family exposes) to keep the trace consistent
    // with it.
    //
    // Under the projected/`Smooth` route those two roles are ALREADY separate —
    // the projected kernel owns the value and the traces, so `hessian_op` is
    // free to carry the completion, which is what `completion_in_operator`
    // arranges. Under every other route (`PseudoLogdetMode::PositiveDefinite`,
    // which is what the multinomial declares) `hessian_op` is the value/trace
    // object, so the completion cannot go into it — and before this the mode
    // response silently borrowed that same object and solved on `M_DD`.
    //
    // `v_k` reaches the outer gradient through the drift trace
    // `½ tr(M_DD⁻¹ D_β M_DD[v_k])`, so a `v_k` from the wrong operator makes the
    // analytic gradient the derivative of a DIFFERENT function than the value —
    // measured on the penguins arm as up to 150% relative error with the sign
    // wrong on three coordinates, which is what dropped the Jeffreys-armed refit
    // into `line_search=StepSizeTooSmall` and refused the fit.
    //
    // So build the second object when, and only when, the completion exists and
    // is not already in `hessian_op`. `None` everywhere else keeps every other
    // family and every clean fit byte-identical: no completion (or no Jeffreys
    // term at all) means `M_true == M_DD` and there is nothing to separate.
    let mode_response_op: Option<Arc<dyn HessianFactorization>> = match (
        robust_jeffreys_hphi.as_ref(),
        robust_jeffreys_completion.as_ref(),
    ) {
        (Some(hphi), Some(completion)) => Some(assemble_operator(Some(&(hphi + completion)))?),
        _ => None,
    };

    // Structural guard against the gam#1395 `0.5·log|H|` collapse.
    //
    // The LAML/pseudo-Laplace objective adds `0.5·hessian_op.logdet()` (the
    // `0.5·log|H|` Laplace term). `hessian_op` is assembled by one of several
    // structurally-independent routes — the `MatrixFreeSpdOperator` matvec
    // closure, its single-pass `dense_assemble` BLAS-3 build, the
    // `BlockCoupledOperator` dense eigendecomposition, or a fingerprint cache
    // hit. Each is *asserted* (see the assembly comments above) to realize the
    // exact penalized joint Hessian `H_unpen + S_λ + scale·H_Φ`, but nothing
    // *checks* it. gam#1395 is precisely the failure where the operator's
    // effective spectrum diverges from that matrix (the reported symptom: an
    // effective eigenvalue ~1.0 instead of 2.0, so `0.5·log|H|` collapses from
    // `0.5·ln2 = 0.3466` to ~0.0044). A halved curvature, a dropped penalty/
    // Jeffreys term, or a stale cache entry would all enter the objective
    // SILENTLY through `logdet()`.
    //
    // So when the logdet term is actually consumed (`include_logdet_h`), and the
    // dimension is small enough to afford a dense ground-truth eigendecomposition,
    // rebuild the SAME matrix directly from `h_joint_unpen` + penalty + Jeffreys,
    // run it through the SAME `pseudo_logdet_mode` spectral operator, and compare
    // its logdet to the assembled operator's. An apples-to-apples match proves no
    // collapse entered the assembly; a divergence is a true defect. We
    // `debug_assert!` (panicking the test/debug builds that would otherwise ship
    // a wrong value) and `log::error!` in release so the regression is never
    // silent. This makes the gam#1395 collapse structurally observable at its
    // source rather than only at the far-downstream objective value.
    if include_logdet_h && total > 0 && total <= JOINT_LOGDET_GUARD_MAX_DIM {
        if let Ok(mut ground_truth) =
            materialize_joint_hessian_source(&h_joint_unpen, total, "gam#1395 logdet guard")
        {
            add_joint_penalty_to_matrix(
                &mut ground_truth,
                ranges,
                &scaled_s_lambdas,
                scaled_joint_trace_diagonal_ridge,
                None,
            );
            // gam#1587: the assembled operator includes the full-width joint
            // penalty, so the ground-truth reference must too — otherwise this
            // guard false-positives a logdet divergence.
            if let Some(joint) = scaled_joint_penalty.as_ref() {
                ground_truth += joint;
            }
            // Mirror the operator's `else`-branch Jeffreys term EXACTLY
            // (`robust_jeffreys_hphi_for_operator` scaled by `rho_curvature_scale`,
            // including any completion span) rather than the pre-scaled
            // `scaled_robust_jeffreys_hphi` used by the projected-logdet path, so
            // the comparison stays apples-to-apples and never false-positives.
            if let Some(hphi) = robust_jeffreys_hphi_for_operator.as_ref() {
                ground_truth.scaled_add(rho_curvature_scale, hphi);
            }
            symmetrize_dense_in_place(&mut ground_truth);
            match DenseSpectralOperator::from_symmetric_with_mode(&ground_truth, pseudo_logdet_mode)
            {
                Ok(reference) => {
                    let reference_logdet = reference.logdet();
                    let assembled_logdet = hessian_op.logdet();
                    if reference_logdet.is_finite() && assembled_logdet.is_finite() {
                        // Relative tolerance scaled by the dimension (each of the
                        // `total` eigenvalues contributes one `ln σ`), with a small
                        // absolute floor for the all-near-unit-spectrum case.
                        let tol = 1e-7 * (total as f64) * (1.0 + reference_logdet.abs());
                        let gap = (assembled_logdet - reference_logdet).abs();
                        if gap > tol {
                            log::error!(
                                "[gam#1395] assembled joint-Hessian logdet diverges from the \
                                 ground-truth penalized Hessian: assembled={assembled_logdet:.9e} \
                                 reference={reference_logdet:.9e} gap={gap:.3e} tol={tol:.3e} \
                                 total={total}. The 0.5*log|H| Laplace term is being computed from \
                                 an operator that does not realize H_unpen + S_lambda + \
                                 scale*H_Phi (collapse / dropped-term / stale-cache class)."
                            );
                            assert!(
                                gap <= tol,
                                "gam#1395 logdet collapse guard: assembled joint-Hessian \
                                 logdet={assembled_logdet:.9e} != ground-truth {reference_logdet:.9e} \
                                 (gap={gap:.3e} > tol={tol:.3e}, total={total})"
                            );
                        }
                    }
                }
                Err(error) => {
                    log::debug!(
                        "[gam#1395] logdet guard skipped: ground-truth eigendecomposition failed: {error}"
                    );
                }
            }
        }
    }

    // The pseudo-log-determinant this route installs and the kernel that
    // differentiates it are ONE object (#2765): the correction rides on the
    // kernel, so a consumer that declines the kernel cannot inherit a corrected
    // value it has no derivative for. That also removes the collapse the
    // `joint_penalty` doc above warns about — when the route yields no kernel
    // (`rank == 0`, or every eigenpair below threshold) the old code still
    // applied `0 − hop.logdet()` and silently deleted `½log|H_pen|` from the
    // cost while the gradient kept its `½tr(H⁻¹∂H)` derivative.
    let penalty_subspace_trace = if project_hessian_logdet
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
            scaled_joint_penalty.as_ref(),
        )?;
        kernel.map(|mut kernel| {
            kernel.logdet_correction = projected_logdet - hessian_op.logdet();
            log::debug!(
                "[OUTER hessian-route] joint penalty subspace trace installed correction={:.6e}",
                kernel.logdet_correction
            );
            Arc::new(kernel)
        })
    } else {
        None
    };

    // gam#1587/#561: `unified_joint_cost_gradient` appends one coordinate per
    // full-width joint penalty (the centered `M⊗S_t` multinomial penalty) AFTER
    // the per-block ρ coordinates — the returned gradient/Hessian have length
    // `rho.len() + n_joint + ext`. The dimension contract validated below was
    // written before #1587 and omitted `n_joint`, so for any family whose
    // smoothing rides ENTIRELY on joint penalties (multinomial: per-block
    // penalties are emptied, so `rho.len() == 0`) every outer ρ-evaluation
    // returned a length-`n_joint` gradient against an `expected == 0` and was
    // rejected at startup validation — silently killing the entire REML/LAML
    // smoothing-parameter search (λ pinned at its seed, EDF near-unpenalized).
    // Count the joint coordinates so the contract matches the gradient the
    // evaluator actually produces.
    let n_joint = options
        .joint_penalties
        .as_deref()
        .map(|bundle| bundle.len())
        .unwrap_or(0);
    let expected_theta_dim = rho.len()
        + n_joint
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
    // kernel and it is not routed through HessianFactorization trace methods.
    // Bind the gating flag before `penalty_subspace_trace` is consumed by
    // the call below so the trace-skip choice does not depend on a moved
    // value (was: `if penalty_subspace_trace.is_none()` evaluated AFTER
    // the trace had already been forwarded to `unified_joint_cost_gradient`).
    let first_order_trace_skip = if penalty_subspace_trace.is_none() {
        first_order_trace_skip
    } else {
        None
    };
    let (objective, grad, outer_hessian, criterion_components, ext_mode_response_cols) =
        unified_joint_cost_gradient(
            inner,
            specs,
            per_block,
            rho,
            beta_flat,
            hessian_op,
            mode_response_op,
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
            // kernel and it is not routed through HessianFactorization trace methods.
            if has_penalty_subspace_trace {
                None
            } else {
                first_order_trace_skip
            },
            robust_jeffreys_phi,
        )?;
    if !objective.is_finite() {
        log::warn!(
            "joint outer evaluation produced non-finite objective: log_likelihood={} penalty_value={} block_logdet_h={:?} block_logdet_s={:?} include_logdet_h={} include_logdet_s={} rho_curvature_scale={}",
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
        });
    }
    if grad.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::NumericalFailure {
            reason: "joint outer evaluation produced a non-finite gradient".to_string(),
        });
    }
    if grad.len() != expected_theta_dim {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "joint outer evaluation returned gradient length {}, expected {}",
                grad.len(),
                expected_theta_dim
            ),
        });
    }
    match &outer_hessian {
        gam_problem::HessianValue::Dense(hessian) => {
            if hessian.iter().any(|value| !value.is_finite()) {
                return Err(CustomFamilyError::NumericalFailure {
                    reason: "joint outer evaluation produced a non-finite Hessian".to_string(),
                });
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
                });
            }
        }
        gam_problem::HessianValue::Operator(op) => {
            if op.dim() != expected_theta_dim {
                return Err(CustomFamilyError::trial_point(format!(
                    "joint outer evaluation returned operator Hessian dim {}, expected {}",
                    op.dim(),
                    expected_theta_dim
                )));
            }
        }
        gam_problem::HessianValue::Unavailable => {}
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
        criterion_components,
        gradient: grad,
        outer_hessian,
        warm_start: warm,
        inner_converged: inner.converged,
        hyper_values: Array1::zeros(0),
        ext_mode_response_cols,
        inner: inner.clone(),
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
    rho_prior: gam_problem::RhoPrior,
    pseudo_logdet_mode: PseudoLogdetMode,
    compute_dh: &DriftDerivFn<'_>,
    compute_dh_many: Option<&DriftDerivManyFn<'_>>,
    compute_d2h: &DriftSecondDerivFn<'_>,
    compute_d2h_many: Option<&DriftSecondDerivManyFn<'_>>,
    owned_compute_dh: Option<
        Arc<dyn Fn(&Array1<f64>) -> Result<Option<DriftDerivResult>, CustomFamilyError> + Send + Sync>,
    >,
    owned_compute_dh_many: Option<
        Arc<dyn Fn(&[Array1<f64>]) -> Result<Vec<Option<DriftDerivResult>>, CustomFamilyError> + Send + Sync>,
    >,
    owned_compute_d2h: Option<
        Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<DriftDerivResult>, CustomFamilyError>
                + Send
                + Sync,
        >,
    >,
    owned_compute_d2h_many: Option<
        Arc<
            dyn Fn(&[(Array1<f64>, Array1<f64>)]) -> Result<Vec<Option<DriftDerivResult>>, CustomFamilyError>
                + Send
                + Sync,
        >,
    >,
    ext_bundle: Option<ExtCoordBundle>,
) -> Result<gam_problem::EfsEval, CustomFamilyError> {
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

    // gam#1587/#561: the full-width centered joint penalty must enter the EFS
    // path's operator AND its projected-logdet kernel, identically to the ARC
    // path above — otherwise the EFS step optimizes a criterion missing
    // `½log|H_pen|` (see `joint_penalty_subspace_trace_parts`). `None` for
    // every per-block-only family keeps this byte-identical.
    let scaled_joint_penalty: Option<Array2<f64>> =
        options.joint_penalties.as_deref().and_then(|bundle| {
            if bundle.is_empty() {
                return None;
            }
            let mut matrix = Array2::<f64>::zeros((total, total));
            bundle.add_to_matrix(&mut matrix);
            if rho_curvature_scale != 1.0 {
                matrix.mapv_inplace(|value| rho_curvature_scale * value);
            }
            Some(matrix)
        });

    let hessian_op: Arc<dyn HessianFactorization> = if use_joint_matrix_free_path(
        total,
        joint_observation_count(&inner.block_states),
    ) {
        let ranges_vec = ranges.to_vec();
        let s_lambdas = Arc::new(scaled_s_lambdas.clone());
        let joint_penalty_arc: Option<Arc<Array2<f64>>> =
            scaled_joint_penalty.clone().map(Arc::new);
        let trace_diagonal_ridge =
            scaled_joint_trace_diagonal_ridge + rho_curvature_scale * JOINT_TRACE_STABILITY_RIDGE;
        match &h_joint_unpen {
            JointHessianSource::Dense(h_joint) => {
                let h_joint = Arc::new(h_joint.clone());
                let apply_h = Arc::clone(&h_joint);
                let apply_ranges = ranges_vec.clone();
                let apply_s = Arc::clone(&s_lambdas);
                let apply_joint = joint_penalty_arc.clone();
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
                        if let Some(joint) = apply_joint.as_ref() {
                            out += &joint.dot(v);
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
                let apply_joint = joint_penalty_arc.clone();
                let dense_joint = joint_penalty_arc.clone();
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
                        if let Some(joint) = dense_joint.as_ref() {
                            matrix += joint.as_ref();
                        }
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
                        if let Some(joint) = apply_joint.as_ref() {
                            out += &joint.dot(v);
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
            "joint exact-newton Hessian materialization for fixed-point evaluation",
        )?;
        add_joint_penalty_to_matrix(
            &mut j_for_traces,
            ranges,
            &scaled_s_lambdas,
            scaled_joint_trace_diagonal_ridge,
            None,
        );
        if let Some(joint) = scaled_joint_penalty.as_ref() {
            j_for_traces += joint;
        }
        Arc::new(
            BlockCoupledOperator::from_joint_hessian_with_mode(&j_for_traces, pseudo_logdet_mode)
                .map_err(|e| CustomFamilyError::trial_point(format!("BlockCoupledOperator from joint Hessian: {e}")))?,
        )
    };

    // One object, as above (#2765): the corrected value rides on the kernel that
    // differentiates it.
    let penalty_subspace_trace = if project_hessian_logdet
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
            scaled_joint_penalty.as_ref(),
        )?;
        kernel.map(|mut kernel| {
            kernel.logdet_correction = projected_logdet - hessian_op.logdet();
            log::debug!(
                "[OUTER hessian-route] joint EFS penalty subspace trace installed correction={:.6e}",
                kernel.logdet_correction
            );
            Arc::new(kernel)
        })
    } else {
        None
    };

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
    rho_prior: gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    let hyper_layout = CustomFamilyHyperLayout::new(
        vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()],
        Vec::new(),
        Array1::zeros(0),
    )?;
    evaluate_custom_family_hyper_internal(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        &hyper_layout,
        warm_start,
        rho_prior,
        eval_mode,
    )
}

pub(crate) fn outerobjectiveefs<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
    rho_prior: gam_problem::RhoPrior,
) -> Result<
    (
        gam_problem::EfsEval,
        ConstrainedWarmStart,
        bool,
        BlockwiseInnerResult,
    ),
    CustomFamilyError,
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
        let (eval, warm, converged) = nonconverged_outer_efs_result(
            &inner,
            rho,
            rho.len(),
            "custom-family EFS non-converged inner solve",
        )?;
        return Ok((eval, warm, converged, inner));
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let moderidge = if options.ridge_policy.accounts_for_objective() {
        ridge
    } else {
        0.0
    };
    let extra_logdet_ridge = 0.0;

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
            // The EFS evaluator always assembles the first-order (gradient)
            // fixed-point terms; the third-derivative directional cache is the
            // one it consumes (gam#979).
            EvalMode::ValueAndGradient,
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
                return Err(CustomFamilyError::trial_point(
                    "outer hyper fixed-point evaluation requires a joint exact path for this family",
                ));
            }
            if specs.len() != 1 {
                return Err(CustomFamilyError::trial_point(
                    "generic fixed-point outer fallback is only valid for single-block families; multi-block families must provide a joint outer path",
                ));
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
                }
                | BlockWorkingSet::NaturalDiagonal {
                    observed_curvature: working_weights,
                    ..
                } => with_block_geometry(
                    family,
                    &inner.block_states,
                    spec,
                    block_idx,
                    |x_dyn, _| {
                        let w = certify_finite_working_weights(working_weights)?;
                        let (xtwx, _) = weighted_normal_equations(x_dyn, w, None)?;
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
                        ) });
                    }
                    hessian.to_dense()
                }
            };
            let beta_flat = inner.block_states[block_idx].beta.clone();
            let compute_dh = |direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, CustomFamilyError> {
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
                            ) }),
                        }
                    }
                    BlockWorkingSet::Diagonal {
                        working_response: _,
                        working_weights,
                    }
                    | BlockWorkingSet::NaturalDiagonal {
                        observed_curvature: working_weights,
                        ..
                    } => {
                        let x_dyn = diagonal_design.as_ref().ok_or_else(|| {
                                    format!(
                                        "missing dynamic design for block {block_idx} diagonal fixed-point correction"
                                    )
                                })?;
                        let wwork = certify_finite_working_weights(working_weights)?;
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
                            ) });
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
             -> Result<Option<DriftDerivResult>, CustomFamilyError> {
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
                            ) }),
                        }
                    }
                    BlockWorkingSet::Diagonal {
                        working_response: _,
                        // The diagonal d2H correction re-derives curvature through
                        // `diagonalworking_weights_second_directional_derivative`
                        // (`d2w` below), so the base working weights are unused here.
                        working_weights: _,
                    }
                    | BlockWorkingSet::NaturalDiagonal { .. } => {
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
                             -> Result<(), CustomFamilyError> {
                                if let Some(geom_dir) = geom {
                                    let has_offset =
                                        geom_dir.d_offset.iter().any(|value| *value != 0.0);
                                    if geom_dir.d_design.is_some() || has_offset {
                                        return Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
                                            "block {block_idx} diagonal d2H requires second-order block-geometry derivatives for {label}; use an exact-newton or joint outer path"
                                        ) });
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
                            ) });
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

    Ok((efs_eval, warm, inner.converged, inner))
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
    per_block_lambdas: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref components) = deriv.s_psi_penalty_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s_part.add_scaled_to(per_block_lambdas[*penalty_idx], &mut s);
        }
        return s;
    }
    if let Some(ref components) = deriv.s_psi_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        for (penalty_idx, s_part) in components {
            s.scaled_add(per_block_lambdas[*penalty_idx], s_part);
        }
        s
    } else if let Some(penalty_idx) = deriv.penalty_index {
        deriv.s_psi.mapv(|v| per_block_lambdas[penalty_idx] * v)
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
    per_block_lambdas: &Array1<f64>,
    p_block: usize,
) -> Array2<f64> {
    if let Some(ref parts) = deriv_i.s_psi_psi_penalty_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        if let Some(pair_parts) = parts.get(local_j) {
            for (penalty_idx, s_part) in pair_parts {
                s_part.add_scaled_to(per_block_lambdas[*penalty_idx], &mut s);
            }
        }
        return s;
    }
    if let Some(ref parts) = deriv_i.s_psi_psi_components {
        let mut s = Array2::<f64>::zeros((p_block, p_block));
        if let Some(pair_parts) = parts.get(local_j) {
            for (penalty_idx, s_part) in pair_parts {
                s.scaled_add(per_block_lambdas[*penalty_idx], s_part);
            }
        }
        s
    } else if let Some(ref parts) = deriv_i.s_psi_psi {
        if let Some(s_part) = parts.get(local_j) {
            if let Some(penalty_index) = deriv_i.penalty_index {
                s_part.mapv(|v| per_block_lambdas[penalty_index] * v)
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

/// A joint likelihood score together with the operating point it was
/// evaluated at.
///
/// The inner solve reloads `∇ℓ` immediately after every accepted step and
/// never moves β afterwards without reloading, so at a convergence exit the
/// score belongs to the returned `block_states`. Carrying `beta` alongside it
/// makes that an assertion the consumer checks rather than a property of how
/// the loop happens to be ordered.
#[derive(Clone)]
pub(crate) struct TerminalLikelihoodScore {
    /// Concatenated block coefficients, block-major.
    pub(crate) beta: Array1<f64>,
    /// `∇ℓ(beta)` in the same layout.
    pub(crate) score: Array1<f64>,
}

impl TerminalLikelihoodScore {
    /// Concatenate block coefficients in the layout the joint score uses.
    pub(crate) fn joint_beta(states: &[ParameterBlockState]) -> Array1<f64> {
        let total = states.iter().map(|state| state.beta.len()).sum::<usize>();
        let mut beta = Array1::<f64>::zeros(total);
        let mut offset = 0usize;
        for state in states {
            let end = offset + state.beta.len();
            beta.slice_mut(s![offset..end]).assign(&state.beta);
            offset = end;
        }
        beta
    }

    /// Whether this score was evaluated at exactly `states`.
    ///
    /// Bitwise: the states are carried out of the solve unmodified after the
    /// reload, so any difference at all means the score is stale.
    pub(crate) fn evaluated_at(&self, states: &[ParameterBlockState]) -> bool {
        self.beta == Self::joint_beta(states)
    }
}

/// The complete state an inner Laplace mode is a function of — the objective's
/// state, not merely the penalty's.
///
/// The mode `β̂` minimises the inner coefficient objective
///
/// ```text
///     ℓ_p(β)  =  −ℓ(β)  +  ½ βᵀ S_λ β  −  τ · Φ(β)
/// ```
///
/// so it is determined by the TOTAL penalty
///
/// ```text
///     S_λ = Σ_b Σ_k e^{ρ_{bk}} S_{bk}   +   Σ_j e^{ρ_j} S_j
///            ╰──── per-block ────╯          ╰─── joint ───╯
/// ```
///
/// AND by the family's Jeffreys/Firth augmentation strength `τ`. A cached mode
/// is reusable exactly when every one of those is unchanged. This records them
/// at the state the mode was actually solved at, and the cache lookup compares
/// against that record — the mode carries its own provenance instead of the
/// consumer reconstructing a key from whatever coordinates it happens to hold.
///
/// Equality is the reuse test, and it is BITWISE on purpose: a cached mode is
/// reusable only when the objective it minimises is the identical function. A
/// tolerance here would reintroduce exactly the failure this type exists to
/// remove — a criterion evaluated at one state and reported at another.
///
/// # Both halves of the state have failed, for the same reason
///
/// **The ρ half (#2615).** `physical_warm_start_for_labeled` rewrites a warm
/// start's `rho` to the PHYSICAL per-block coordinates before the inner solve
/// sees it, and a family whose smoothing lives entirely in the joint bundle has
/// none: since #1587 emptied the multinomial's `spec.penalties` into a centered
/// `M ⊗ S_t` bundle, `physical_count() == 0`. The old guard compared an empty
/// vector against an empty vector, answered `true` at every ρ, and returned the
/// mode solved at the FIRST ρ for every subsequent one — freezing `log|H|` at a
/// constant while `½log|S_λ|` kept growing, which leaves the profiled criterion
/// monotone in ρ and rails every coordinate at its upper bound.
///
/// **The τ half (#2612).** The same key, complete in ρ, was still the WHOLE
/// reuse test — so it answered `true` at every `τ`. That is not a corner case
/// for the one path that varies `τ`: `coefficient_mode_homotopy_member` defines
/// the armed coefficient mode as the endpoint of a continuation in `τ` **at
/// fixed ρ** (#2366), so on that path the ρ half is constant by construction and
/// the key matched at every waypoint. The only thing that kept the corrector
/// running was the fresh curvature certificate refusing a non-PSD incoming
/// point — and the finer the discretization, the less the mode moves per
/// waypoint and the more often that certificate passes. Measured on the
/// penguins stride-4 refit, 8-step sweep, `λ_min` at each incoming mode:
///
/// ```text
///   wp1 −7.88e-2   wp2 −8.74e-6   wp3 −7.26e-6   wp4 −4.82e-6
///   wp5 −2.46e-6   wp6 −7.38e-7   wp7 −5.16e-8   wp8 +2.78e-8  <- reused
/// ```
///
/// The endpoint of that sweep — the point the whole continuation exists to
/// produce — was the `τ = 0.875` mode relabelled as the `τ = 1` mode, and the
/// refinement that was supposed to make the path converge is precisely what
/// disabled its corrector.
///
/// # What belongs here
///
/// Anything the inner coefficient objective depends on that is not the data.
/// Today that is the two above: the family is immutable for the duration of a
/// fit, and the ONE mechanism that produces a different family value at the same
/// ρ is [`CustomFamily::coefficient_mode_homotopy_member`], whose contract
/// restricts it to the augmentation strength. A future homotopy over any other
/// family-carried quantity must extend this type in the same commit, because
/// this equality is the only thing standing between it and a mode served for the
/// wrong function.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InnerObjectiveState {
    /// Per-block `log λ`, in block order.
    block: Vec<Array1<f64>>,
    /// Joint-bundle `log λ`, parallel to the bundle's specs; empty when the
    /// family declares no joint penalties.
    joint: Vec<f64>,
    /// The family's Jeffreys/Firth augmentation strength `τ`, which multiplies
    /// every Jeffreys value, score and curvature the inner objective carries.
    /// `0` on the unbiased objective, `1` on the fully armed one, and the
    /// homotopy coordinate in between.
    jeffreys_strength: f64,
}

impl InnerObjectiveState {
    pub(crate) fn new<F: CustomFamily + ?Sized>(
        family: &F,
        block_log_lambdas: &[Array1<f64>],
        joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    ) -> Self {
        Self {
            block: block_log_lambdas.to_vec(),
            joint: joint_bundle
                .map(|bundle| bundle.log_lambdas().to_vec())
                .unwrap_or_default(),
            jeffreys_strength: family.joint_jeffreys_term_strength(),
        }
    }

    /// Round-trip form for the on-disk warm-start record. A persisted mode is
    /// only reusable if the state it was solved at travels with it, so both
    /// smoothing halves are serialized rather than re-derived from the record's
    /// ρ (which carries no block/joint split).
    ///
    /// `τ` is deliberately NOT part of this pair: it reaches the persisted
    /// record through the cache KEY
    /// (`persistent_custom_family_key`), so a record that loads at all was
    /// written by a family with the identical strength, and `from_parts` takes
    /// that strength from the loading family. Serializing it as well would give
    /// the same fact two homes that could disagree.
    pub(crate) fn to_parts(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        (
            self.block.iter().map(|b| b.to_vec()).collect(),
            self.joint.clone(),
        )
    }

    pub(crate) fn from_parts<F: CustomFamily + ?Sized>(
        family: &F,
        block: Vec<Vec<f64>>,
        joint: Vec<f64>,
    ) -> Self {
        Self {
            block: block.into_iter().map(Array1::from_vec).collect(),
            joint,
            jeffreys_strength: family.joint_jeffreys_term_strength(),
        }
    }

    /// The augmentation strength this state was recorded at. Read by the reuse
    /// gate's own diagnostics so a refusal can say WHICH half of the state
    /// differed.
    pub(crate) fn jeffreys_strength(&self) -> f64 {
        self.jeffreys_strength
    }
}

/// Constructors that exist only for fixtures, kept out of the production impl
/// so no production site can reach one.
#[cfg(test)]
mod test_support {
    use super::{Array1, InnerObjectiveState};

    impl InnerObjectiveState {
        /// The state of an objective carrying NO Jeffreys augmentation.
        ///
        /// For fixtures that assemble a [`super::BlockwiseInnerResult`]
        /// directly and have no family to read the strength from. Every
        /// production site has one and must use [`InnerObjectiveState::new`],
        /// which cannot be handed the wrong number.
        pub(crate) fn unaugmented(
            block_log_lambdas: &[Array1<f64>],
            joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
        ) -> Self {
            Self {
                block: block_log_lambdas.to_vec(),
                joint: joint_bundle
                    .map(|bundle| bundle.log_lambdas().to_vec())
                    .unwrap_or_default(),
                jeffreys_strength: 0.0,
            }
        }
    }
}

#[derive(Clone)]
pub struct BlockwiseInnerResult {
    pub block_states: Vec<ParameterBlockState>,
    /// Exact working-set evidence evaluated at `block_states` by the inner
    /// solve. `None` only when an owned exact-Hessian workspace was the sole
    /// family evaluation source.
    pub(crate) terminal_working_sets: Option<Vec<BlockWorkingSet>>,
    /// Joint likelihood score the inner solve last loaded, carried with the
    /// coefficient vector it was evaluated at.
    ///
    /// This is the same vector the projected KKT residual is computed from, so
    /// wherever `kkt_residual` certifies the iterate this score is the gradient
    /// that certified it. It exists because a family with an analytic joint
    /// gradient never produces working sets — `load_joint_gradient_evaluation`
    /// returns the gradient and no `FamilyEvaluation` — so reconstructing the
    /// score downstream from `terminal_working_sets` is impossible for exactly
    /// the families that compute it most directly.
    pub(crate) terminal_likelihood_score: Option<TerminalLikelihoodScore>,
    pub active_sets: Vec<Option<Vec<usize>>>,
    pub log_likelihood: f64,
    pub penalty_value: f64,
    pub cycles: usize,
    pub converged: bool,
    /// The blockwise loop's terminal decision variables. Unlike `kkt_residual`
    /// below, this is populated whether or not the solve converged — it is the
    /// diagnostic a non-convergence refusal needs, and it is NEVER an IFT input.
    pub terminal_convergence_state: Option<gam_problem::InnerConvergenceTerminalState>,
    /// Laplace Hessian log-determinant, defined only at a certified inner mode.
    pub block_logdet_h: Option<f64>,
    /// Penalty pseudo-logdeterminant, defined only at a certified inner mode.
    pub block_logdet_s: Option<f64>,
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
    /// The smoothing state this mode was solved at — the cache key a reuse
    /// decision must be taken against (#2615).
    pub(crate) objective_state: InnerObjectiveState,
}

impl std::fmt::Debug for BlockwiseInnerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockwiseInnerResult")
            .field("block_states", &self.block_states)
            .field(
                "terminal_working_sets",
                &self.terminal_working_sets.as_ref().map(Vec::len),
            )
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
    pub(crate) block_logdet_h: Option<f64>,
    pub(crate) block_logdet_s: Option<f64>,
    pub(crate) joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    pub(crate) kkt_residual: Option<ProjectedKktResidual>,
    pub(crate) active_constraints: Option<Arc<ActiveLinearConstraintBlock>>,
    pub(crate) terminal_working_sets: Option<Vec<BlockWorkingSet>>,
    pub(crate) terminal_likelihood_score: Option<TerminalLikelihoodScore>,
    /// The smoothing state this cached mode was solved at (#2615). A lookup
    /// compares against this, not against a key rebuilt from the caller's
    /// coordinates.
    pub(crate) objective_state: InnerObjectiveState,
}

#[cfg(test)]
mod first_order_trace_skip_tests {
    use super::{Arc, DenseSpectralOperator, FirstOrderTraceSkipOperator, HessianFactorization};
    use ndarray::array;

    /// gam#979: the trace-skip wrapper must not refuse a capability its inner
    /// backend has. `assemble_h_dense_for_tangent_projection` has one
    /// consumer — the active-constraint mode response — and its refusal is not
    /// a slow path but a REFUSED TRIAL POINT, which is what made the CTN
    /// preprocessor's outer search spend whole restarts on infeasible probes.
    ///
    /// The skip list itself still has to work, and `as_exact_dense_spectral`
    /// still has to be gated: that route batches traces in the eigenbasis and
    /// would walk past the list this wrapper is serving.
    #[test]
    fn the_trace_skip_still_hands_back_the_dense_stationarity_curvature_979() {
        let hessian = array![[4.0, 1.0], [1.0, 3.0]];
        let inner: Arc<dyn HessianFactorization> =
            Arc::new(DenseSpectralOperator::from_symmetric(&hessian).expect("SPD fixture"));
        let want = inner
            .assemble_h_dense_for_tangent_projection()
            .expect("the inner backend assembles its own stationarity curvature");
        // `DenseSpectralOperator` reconstructs `H` from its eigendecomposition,
        // so the round trip is exact only to accumulation order.
        for row in 0..hessian.nrows() {
            for column in 0..hessian.ncols() {
                let (got, expected) = (want[[row, column]], hessian[[row, column]]);
                assert!(
                    (got - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
                    "the fixture's own curvature is what comes back: [{row}, {column}] \
                     = {got:.17e} against {expected:.17e}"
                );
            }
        }

        let skipping = FirstOrderTraceSkipOperator::new(Arc::clone(&inner), 2);
        assert!(
            skipping.first_order_skip_active(),
            "the fixture must start with the skip list live, or the gate below is vacuous"
        );
        let got = skipping
            .assemble_h_dense_for_tangent_projection()
            .expect("a live skip list must not refuse the inner backend's curvature");
        assert_eq!(got, want);

        assert!(
            skipping.as_exact_dense_spectral().is_none(),
            "the eigenbasis batch route stays closed while the skip list is live"
        );
        assert!(skipping.consume_first_order_trace());
        assert!(skipping.consume_first_order_trace());
        assert!(!skipping.consume_first_order_trace());
        assert!(!skipping.first_order_skip_active());
        assert!(
            skipping.as_exact_dense_spectral().is_some(),
            "with the list exhausted the eigenbasis route reopens"
        );
        assert_eq!(
            skipping
                .assemble_h_dense_for_tangent_projection()
                .expect("still available once the list is exhausted"),
            want
        );
    }
}
