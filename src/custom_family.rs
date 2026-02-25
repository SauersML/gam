use crate::matrix::DesignMatrix;
use crate::types::{LinkFunction, RidgeDeterminantMode, RidgePolicy};
use crate::faer_ndarray::FaerCholesky;
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use ndarray::{Array1, Array2};
use wolfe_bfgs::Bfgs;

/// Optional known link metadata when a family uses a learnable wiggle correction.
#[derive(Debug, Clone, Copy)]
pub struct KnownLinkWiggle {
    pub base_link: LinkFunction,
    /// Index of the block that parameterizes the wiggle term, if any.
    pub wiggle_block: Option<usize>,
}

/// Static specification for one parameter block in a custom family.
#[derive(Clone)]
pub struct ParameterBlockSpec {
    pub name: String,
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    /// Block-local penalty matrices (all p_block x p_block).
    pub penalties: Vec<Array2<f64>>,
    /// Initial log-smoothing parameters for this block (same length as `penalties`).
    pub initial_log_lambdas: Array1<f64>,
    /// Optional initial coefficients (defaults to zeros if omitted).
    pub initial_beta: Option<Array1<f64>>,
}

/// Current state for a parameter block.
#[derive(Clone)]
pub struct ParameterBlockState {
    pub beta: Array1<f64>,
    pub eta: Array1<f64>,
}

/// Working quantities supplied by a custom family for one block.
#[derive(Clone)]
pub struct BlockWorkingSet {
    /// IRLS pseudo-response for this block's linear predictor.
    pub working_response: Array1<f64>,
    /// IRLS working weights for this block (non-negative, length n).
    pub working_weights: Array1<f64>,
    /// Optional score wrt this block's linear predictor (diagnostics / custom control).
    pub gradient_eta: Option<Array1<f64>>,
}

/// Family evaluation over all parameter blocks.
#[derive(Clone)]
pub struct FamilyEvaluation {
    pub log_likelihood: f64,
    pub block_working_sets: Vec<BlockWorkingSet>,
}

/// User-defined family contract for multi-block generalized models.
pub trait CustomFamily {
    /// Evaluate log-likelihood and per-block working quantities at current block predictors.
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String>;

    /// Optional metadata describing a known link with learnable wiggle.
    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        None
    }

    /// Optional dynamic geometry hook for blocks whose design/offset depend on
    /// current values of other blocks.
    fn block_geometry(
        &self,
        _block_index: usize,
        _block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        Ok((spec.design.clone(), spec.offset.clone()))
    }

    /// Optional per-block coefficient projection applied after each block update.
    fn post_update_beta(
        &self,
        _block_index: usize,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        Ok(beta)
    }
}

#[derive(Clone)]
pub struct BlockwiseFitOptions {
    pub inner_max_cycles: usize,
    pub inner_tol: f64,
    pub outer_max_iter: usize,
    pub outer_tol: f64,
    pub min_weight: f64,
    pub ridge_floor: f64,
    /// Shared ridge semantics used by solve/quadratic/logdet terms.
    pub ridge_policy: RidgePolicy,
    /// If true, outer smoothing optimization uses a Laplace/REML-style objective:
    ///   -loglik + penalty + 0.5(log|H| - log|S|_+)
    /// where H is blockwise working curvature and S is blockwise penalty.
    pub use_reml_objective: bool,
}

impl Default for BlockwiseFitOptions {
    fn default() -> Self {
        Self {
            inner_max_cycles: 100,
            inner_tol: 1e-6,
            outer_max_iter: 60,
            outer_tol: 1e-5,
            min_weight: 1e-12,
            ridge_floor: 1e-12,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: true,
        }
    }
}

#[derive(Clone)]
pub struct BlockwiseInnerResult {
    pub block_states: Vec<ParameterBlockState>,
    pub log_likelihood: f64,
    pub penalty_value: f64,
    pub cycles: usize,
    pub converged: bool,
    pub block_logdet_h: f64,
    pub block_logdet_s: f64,
}

#[derive(Clone)]
pub struct BlockwiseFitResult {
    pub block_states: Vec<ParameterBlockState>,
    pub log_likelihood: f64,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    pub inner_cycles: usize,
    pub converged: bool,
}

fn validate_block_specs(specs: &[ParameterBlockSpec]) -> Result<(usize, Vec<usize>), String> {
    if specs.is_empty() {
        return Err("fit_custom_family requires at least one parameter block".to_string());
    }
    let n = specs[0].design.nrows();
    let mut penalty_counts = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        if spec.design.nrows() != n {
            return Err(format!(
                "block {b} row mismatch: got {}, expected {n}",
                spec.design.nrows()
            ));
        }
        if spec.offset.len() != n {
            return Err(format!(
                "block {b} offset length mismatch: got {}, expected {n}",
                spec.offset.len()
            ));
        }
        let p = spec.design.ncols();
        if let Some(beta0) = &spec.initial_beta {
            if beta0.len() != p {
                return Err(format!(
                    "block {b} initial_beta length mismatch: got {}, expected {p}",
                    beta0.len()
                ));
            }
        }
        if spec.initial_log_lambdas.len() != spec.penalties.len() {
            return Err(format!(
                "block {b} initial_log_lambdas length {} does not match penalties {}",
                spec.initial_log_lambdas.len(),
                spec.penalties.len()
            ));
        }
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.dim();
            if r != p || c != p {
                return Err(format!(
                    "block {b} penalty {k} must be {p}x{p}, got {r}x{c}"
                ));
            }
        }
        penalty_counts.push(spec.penalties.len());
    }
    Ok((n, penalty_counts))
}

fn flatten_log_lambdas(specs: &[ParameterBlockSpec]) -> Array1<f64> {
    let total = specs
        .iter()
        .map(|s| s.initial_log_lambdas.len())
        .sum::<usize>();
    let mut out = Array1::<f64>::zeros(total);
    let mut at = 0usize;
    for spec in specs {
        let len = spec.initial_log_lambdas.len();
        if len > 0 {
            out.slice_mut(ndarray::s![at..at + len])
                .assign(&spec.initial_log_lambdas);
        }
        at += len;
    }
    out
}

fn split_log_lambdas(
    flat: &Array1<f64>,
    penalty_counts: &[usize],
) -> Result<Vec<Array1<f64>>, String> {
    let expected: usize = penalty_counts.iter().sum();
    if flat.len() != expected {
        return Err(format!(
            "log-lambda length mismatch: got {}, expected {expected}",
            flat.len()
        ));
    }
    let mut out = Vec::with_capacity(penalty_counts.len());
    let mut at = 0usize;
    for &k in penalty_counts {
        out.push(flat.slice(ndarray::s![at..at + k]).to_owned());
        at += k;
    }
    Ok(out)
}

fn build_block_states<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> Result<Vec<ParameterBlockState>, String> {
    let mut states = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let p = spec.design.ncols();
        let beta = spec
            .initial_beta
            .clone()
            .unwrap_or_else(|| Array1::<f64>::zeros(p));
        let (x_dyn, off_dyn) = family.block_geometry(b, &states, spec)?;
        if x_dyn.nrows() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic design row mismatch: got {}, expected {}",
                x_dyn.nrows(),
                spec.design.nrows()
            ));
        }
        if x_dyn.ncols() != p {
            return Err(format!(
                "block {b} dynamic design col mismatch: got {}, expected {p}",
                x_dyn.ncols()
            ));
        }
        if off_dyn.len() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic offset length mismatch: got {}, expected {}",
                off_dyn.len(),
                spec.design.nrows()
            ));
        }
        let mut eta = x_dyn.matrix_vector_multiply(&beta);
        eta += &off_dyn;
        states.push(ParameterBlockState { beta, eta });
    }
    Ok(states)
}

fn refresh_all_block_etas<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
) -> Result<(), String> {
    for b in 0..specs.len() {
        let spec = &specs[b];
        let p = states[b].beta.len();
        let (x_dyn, off_dyn) = family.block_geometry(b, states, spec)?;
        if x_dyn.nrows() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic design row mismatch: got {}, expected {}",
                x_dyn.nrows(),
                spec.design.nrows()
            ));
        }
        if x_dyn.ncols() != p {
            return Err(format!(
                "block {b} dynamic design col mismatch: got {}, expected {p}",
                x_dyn.ncols()
            ));
        }
        if off_dyn.len() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic offset length mismatch: got {}, expected {}",
                off_dyn.len(),
                spec.design.nrows()
            ));
        }
        states[b].eta = x_dyn.matrix_vector_multiply(&states[b].beta) + &off_dyn;
    }
    Ok(())
}

fn solve_block_weighted_system(
    x: &Array2<f64>,
    y_star: &Array1<f64>,
    w: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    let p = x.ncols();
    if y_star.len() != n || w.len() != n {
        return Err("weighted-system dimension mismatch".to_string());
    }

    let mut xtwx = Array2::<f64>::zeros((p, p));
    let mut xtwy = Array1::<f64>::zeros(p);

    for i in 0..n {
        let wi = w[i].max(0.0);
        if wi == 0.0 {
            continue;
        }
        let row = x.row(i);
        for a in 0..p {
            let xa = row[a];
            xtwy[a] += wi * xa * y_star[i];
            for b in a..p {
                xtwx[[a, b]] += wi * xa * row[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            xtwx[[a, b]] = xtwx[[b, a]];
        }
    }

    xtwx += s_lambda;

    let ridge = if ridge_policy.include_laplace_hessian {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };
    for d in 0..p {
        xtwx[[d, d]] += ridge;
    }

    let h = crate::faer_ndarray::FaerArrayView::new(&xtwx);
    let mut rhs = xtwy.clone();
    let mut rhs_mat = FaerMat::zeros(p, 1);
    for i in 0..p {
        rhs_mat[(i, 0)] = rhs[i];
    }

    if let Ok(ch) = FaerLlt::new(h.as_ref(), Side::Lower) {
        ch.solve_in_place(rhs_mat.as_mut());
    } else if let Ok(ld) = FaerLdlt::new(h.as_ref(), Side::Lower) {
        ld.solve_in_place(rhs_mat.as_mut());
    } else {
        let lb = FaerLblt::new(h.as_ref(), Side::Lower);
        lb.solve_in_place(rhs_mat.as_mut());
    }

    for i in 0..p {
        rhs[i] = rhs_mat[(i, 0)];
    }
    if rhs.iter().any(|v| !v.is_finite()) {
        return Err("block solve produced non-finite coefficients".to_string());
    }
    Ok(rhs)
}

#[inline]
fn effective_solver_ridge(ridge_floor: f64) -> f64 {
    ridge_floor.max(1e-15)
}

fn stable_logdet_with_ridge_policy(
    matrix: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<f64, String> {
    let mut a = matrix.clone();
    let p = a.nrows();
    for i in 0..p {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
    let ridge = if ridge_policy.include_penalty_logdet {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    match ridge_policy.determinant_mode {
        RidgeDeterminantMode::Full => {
            let chol = a.clone().cholesky(Side::Lower).map_err(|_| {
                "cholesky failed while computing full ridge-aware logdet".to_string()
            })?;
            Ok(2.0 * chol.diag().mapv(f64::ln).sum())
        }
        RidgeDeterminantMode::PositivePart => {
            let (evals, _) = crate::faer_ndarray::FaerEigh::eigh(&a, Side::Lower)
                .map_err(|e| format!("eigh failed while computing logdet: {e}"))?;
            let floor = ridge.max(1e-14);
            let mut logdet = 0.0;
            for &ev in &evals {
                if ev > floor {
                    logdet += ev.ln();
                }
            }
            Ok(logdet)
        }
    }
}

fn blockwise_logdet_terms<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<(f64, f64), String> {
    refresh_all_block_etas(family, specs, states)?;
    let eval = family.evaluate(states)?;
    if eval.block_working_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.block_working_sets.len(),
            specs.len()
        ));
    }

    let mut logdet_h_total = 0.0;
    let mut logdet_s_total = 0.0;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.block_working_sets[b];
        let p = spec.design.ncols();
        let (x_dyn, _) = family.block_geometry(b, states, spec)?;
        if x_dyn.ncols() != p {
            return Err(format!(
                "block {b} dynamic design col mismatch: got {}, expected {p}",
                x_dyn.ncols()
            ));
        }
        let x = x_dyn.to_dense();
        let w = work.working_weights.mapv(|wi| wi.max(options.min_weight));

        let mut xtwx = Array2::<f64>::zeros((p, p));
        for i in 0..x.nrows() {
            let wi = w[i];
            if wi == 0.0 {
                continue;
            }
            let row = x.row(i);
            for a in 0..p {
                let xa = row[a];
                for c in a..p {
                    xtwx[[a, c]] += wi * xa * row[c];
                }
            }
        }
        for a in 0..p {
            for c in 0..a {
                xtwx[[a, c]] = xtwx[[c, a]];
            }
        }

        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }

        let mut h = xtwx;
        h += &s_lambda;
        logdet_h_total +=
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?;
        logdet_s_total += stable_logdet_with_ridge_policy(
            &s_lambda,
            options.ridge_floor,
            options.ridge_policy,
        )?;
    }
    Ok((logdet_h_total, logdet_s_total))
}

fn inner_blockwise_fit<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseInnerResult, String> {
    let mut states = build_block_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let mut last_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut cycles_done = 0usize;

    for cycle in 0..options.inner_max_cycles {
        let mut max_beta_step = 0.0_f64;

        for b in 0..specs.len() {
            // Keep all blocks synchronized with any dynamic geometry.
            refresh_all_block_etas(family, specs, &mut states)?;
            let eval = family.evaluate(&states)?;
            if eval.block_working_sets.len() != specs.len() {
                return Err(format!(
                    "family returned {} block working sets, expected {}",
                    eval.block_working_sets.len(),
                    specs.len()
                ));
            }

            let spec = &specs[b];
            let work = &eval.block_working_sets[b];
            let p = spec.design.ncols();
            if work.working_response.len() != spec.design.nrows()
                || work.working_weights.len() != spec.design.nrows()
            {
                return Err(format!(
                    "family working-set size mismatch on block {b} ({})",
                    spec.name
                ));
            }

            let (x_dyn, off_dyn) = family.block_geometry(b, &states, spec)?;
            if x_dyn.nrows() != spec.design.nrows() {
                return Err(format!(
                    "block {b} dynamic design row mismatch: got {}, expected {}",
                    x_dyn.nrows(),
                    spec.design.nrows()
                ));
            }
            if x_dyn.ncols() != p {
                return Err(format!(
                    "block {b} dynamic design col mismatch: got {}, expected {p}",
                    x_dyn.ncols()
                ));
            }
            if off_dyn.len() != spec.design.nrows() {
                return Err(format!(
                    "block {b} dynamic offset length mismatch: got {}, expected {}",
                    off_dyn.len(),
                    spec.design.nrows()
                ));
            }

            let x = x_dyn.to_dense();
            let mut y_star = work.working_response.clone();
            y_star -= &off_dyn;

            let lambdas = block_log_lambdas[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((p, p));
            for (k, s) in spec.penalties.iter().enumerate() {
                s_lambda.scaled_add(lambdas[k], s);
            }

            let beta_new_raw = solve_block_weighted_system(
                &x,
                &y_star,
                &work.working_weights.mapv(|wi| wi.max(options.min_weight)),
                &s_lambda,
                options.ridge_floor,
                options.ridge_policy,
            )?;
            let beta_new = family.post_update_beta(b, beta_new_raw)?;

            let step = (&beta_new - &states[b].beta)
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max);
            max_beta_step = max_beta_step.max(step);

            states[b].beta = beta_new;
            refresh_all_block_etas(family, specs, &mut states)?;
        }

        refresh_all_block_etas(family, specs, &mut states)?;
        let eval = family.evaluate(&states)?;
        let ll = eval.log_likelihood;
        let ll_change = (ll - last_ll).abs();
        last_ll = ll;
        cycles_done = cycle + 1;

        if max_beta_step <= options.inner_tol && ll_change <= options.inner_tol {
            converged = true;
            break;
        }
    }

    let final_eval = family.evaluate(&states)?;
    let mut penalty_value = 0.0;
    // Keep the objective coherent with the stabilized block solves/log-dets:
    // Single policy contract: if solve path includes ridge in curvature, include
    // the same ridge in quadratic penalty iff policy demands it.
    let ridge = effective_solver_ridge(options.ridge_floor);
    for (b, spec) in specs.iter().enumerate() {
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        for (k, s) in spec.penalties.iter().enumerate() {
            let sb = s.dot(&states[b].beta);
            penalty_value += 0.5 * lambdas[k] * states[b].beta.dot(&sb);
        }
        if options.ridge_policy.include_quadratic_penalty {
            penalty_value += 0.5 * ridge * states[b].beta.dot(&states[b].beta);
        }
    }

    let (block_logdet_h, block_logdet_s) =
        blockwise_logdet_terms(family, specs, &mut states, block_log_lambdas, options)?;

    Ok(BlockwiseInnerResult {
        block_states: states,
        log_likelihood: final_eval.log_likelihood,
        penalty_value,
        cycles: cycles_done,
        converged,
        block_logdet_h,
        block_logdet_s,
    })
}

/// Fit a custom multi-block family.
///
/// Inner loop: cyclic blockwise penalized weighted regressions.
/// Outer loop: joint optimization of all log-smoothing parameters.
pub fn fit_custom_family<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let (_n, penalty_counts) = validate_block_specs(specs)?;
    let rho0 = flatten_log_lambdas(specs);

    if rho0.is_empty() {
        let inner =
            inner_blockwise_fit(family, specs, &vec![Array1::zeros(0); specs.len()], options)?;
        let reml_term = if options.use_reml_objective {
            0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            0.0
        };
        return Ok(BlockwiseFitResult {
            block_states: inner.block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            penalized_objective: -inner.log_likelihood + inner.penalty_value + reml_term,
            outer_iterations: 0,
            inner_cycles: inner.cycles,
            converged: inner.converged,
        });
    }

    let objective = |rho: &Array1<f64>| -> Result<f64, String> {
        let per_block = split_log_lambdas(rho, &penalty_counts)?;
        let inner = inner_blockwise_fit(family, specs, &per_block, options)?;
        let reml_term = if options.use_reml_objective {
            0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            0.0
        };
        Ok(-inner.log_likelihood + inner.penalty_value + reml_term)
    };

    let cost_grad = |rho: &Array1<f64>| -> Result<(f64, Array1<f64>), String> {
        let f0 = objective(rho)?;
        let mut g = Array1::<f64>::zeros(rho.len());
        for j in 0..rho.len() {
            let h = 1e-4_f64 * (1.0 + rho[j].abs());
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus[j] += h;
            minus[j] -= h;
            let fp = objective(&plus)?;
            let fm = objective(&minus)?;
            g[j] = (fp - fm) / (2.0 * h);
        }
        Ok((f0, g))
    };

    let mut solver = Bfgs::new(rho0.clone(), |x| match cost_grad(x) {
        Ok(pair) => pair,
        Err(_) => (f64::INFINITY, Array1::<f64>::zeros(x.len())),
    })
    .with_tolerance(options.outer_tol)
    .with_max_iterations(options.outer_max_iter);
    let sol = solver
        .run()
        .map_err(|e| format!("outer smoothing optimization failed: {e:?}"))?;

    let rho_star = sol.final_point;
    let per_block = split_log_lambdas(&rho_star, &penalty_counts)?;
    let inner = inner_blockwise_fit(family, specs, &per_block, options)?;

    Ok(BlockwiseFitResult {
        block_states: inner.block_states,
        log_likelihood: inner.log_likelihood,
        log_lambdas: rho_star.clone(),
        lambdas: rho_star.mapv(f64::exp),
        penalized_objective: if options.use_reml_objective {
            -inner.log_likelihood
                + inner.penalty_value
                + 0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            -inner.log_likelihood + inner.penalty_value
        },
        outer_iterations: sol.iterations,
        inner_cycles: inner.cycles,
        converged: inner.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DesignMatrix;
    use ndarray::{Array1, array};

    #[derive(Clone)]
    struct OneBlockIdentityFamily;

    impl CustomFamily for OneBlockIdentityFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                block_working_sets: vec![BlockWorkingSet {
                    working_response: Array1::ones(n),
                    working_weights: Array1::ones(n),
                    gradient_eta: None,
                }],
            })
        }
    }

    #[test]
    fn effective_ridge_is_never_below_solver_floor() {
        assert!((effective_solver_ridge(0.0) - 1e-15).abs() < 1e-30);
        assert!((effective_solver_ridge(1e-8) - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn objective_includes_solver_ridge_quadratic_term() {
        // One-parameter block with X=1, y*=1, w=1, no explicit penalties.
        // Inner solve gives beta = 1 / (1 + ridge), so objective should include
        // 0.5 * ridge * beta^2 even when no smoothing penalties are present.
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            inner_tol: 0.0,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            min_weight: 1e-12,
            ridge_floor: 1e-4,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: false,
        };

        let result = fit_custom_family(&OneBlockIdentityFamily, &[spec], &options)
            .expect("custom family fit should succeed");
        let ridge = effective_solver_ridge(options.ridge_floor);
        let beta = result.block_states[0].beta[0];
        let expected_penalty = 0.5 * ridge * beta * beta;
        assert!(
            (result.penalized_objective - expected_penalty).abs() < 1e-12,
            "penalized objective should equal ridge quadratic term when ll=0 and S=0; got {}, expected {}",
            result.penalized_objective,
            expected_penalty
        );
    }
}
