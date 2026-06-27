//! Outer-Hessian representation routing and scale decisions.
//!
//! Cost — never capability — selects between the dense `K × K` assembly and
//! the matrix-free Hv-operator: the operator path delivers the same math while
//! avoiding large dense `p × p` drift storage and pairwise row assembly when
//! the model's `(n, p, K)` shape says those dominate.

pub(crate) const HESSIAN_UNAVAILABLE_PREFIX: &str = "outer Hessian unavailable:";

/// Minimum coefficient dimension at which the matrix-free operator path is
/// selected unconditionally — once `p` is this large the dense `p × p`
/// assembly itself dominates and operator HVPs win regardless of `n` or `K`.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD: usize = 512;

/// Sample-count threshold for the (`n`, `p`) crossover branch: when `n` is
/// large enough that per-row work dominates, the operator path wins even
/// at moderate `p`.
pub const MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD: usize = 50_000;

/// Coefficient dimension paired with [`MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD`]
/// in the (`n`, `p`) crossover branch.
pub const MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N: usize = 32;

/// `n · p` linear-work cutoff: per-eval `O(K · n · p²)` dense assembly
/// dominates once `n · p` crosses this threshold even when both `n` and `p`
/// are individually below the per-axis thresholds.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD: usize = 4_000_000;

/// Smoothing-parameter count above which the operator path wins regardless
/// of `n` and `p`: the per-outer-eval Hessian-assembly cost is
/// `O(K · n · p²)`, so `K` itself drives the crossover.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD: usize = 32;

/// Row-pair work cutoff for callback-backed outer Hessians.
///
/// Callback kernels expose exact row-local first/second Hessian drifts. Dense
/// `K x K` assembly can still be expensive at tiny coefficient dimension
/// because the dominant work is not `p x p` algebra; it is repeated row-kernel
/// contractions over the upper-triangular coordinate pairs.
pub(crate) const CALLBACK_OUTER_HESSIAN_ROW_PAIR_WORK_THRESHOLD: usize = 25_000_000;

/// Coefficient-dimension threshold above which a stochastic (Hutch++) trace
/// kernel is preferred over the exact dense trace for the logdet-H⁻¹ and ψ-Gram
/// paths. Below this the exact dense O(p³) work is cheap enough that the
/// estimator's variance is not worth trading for; above it the stochastic
/// estimator's O(p²·m) cost wins.
pub(crate) const STOCHASTIC_TRACE_DIM_THRESHOLD: usize = 500;

/// Elapsed-time (ms) above which a sparse-Cholesky trace path emits a timing
/// diagnostic. Purely observational — surfaces slow per-eval trace solves to the
/// bench runner without affecting the fit.
pub(crate) const REML_TRACE_SLOW_LOG_MS: f64 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterHessianRoutePlan {
    pub use_operator: bool,
    pub reason: &'static str,
    pub scale_prefers_operator: bool,
    pub dense_workspace_bytes: usize,
}

impl OuterHessianRoutePlan {
    pub(crate) fn choice(self) -> &'static str {
        if self.use_operator {
            "operator"
        } else {
            "dense"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OuterHessianScaleDecision {
    pub(crate) prefers_operator: bool,
    pub(crate) reason: &'static str,
}

pub(crate) fn saturating_f64_matrix_bytes(rows: usize, cols: usize) -> usize {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<f64>())
}

pub(crate) fn outer_hessian_dense_workspace_bytes(p: usize, k: usize) -> usize {
    // Dense assembly keeps first-order drifts for each coordinate and uses at
    // least one transient second-order drift while filling the K x K Hessian.
    // Charge a small safety multiple so the route never depends on fitting a
    // single p x p matrix while the actual dense path holds several.
    let drift_count = k.saturating_mul(2).saturating_add(3).max(1);
    saturating_f64_matrix_bytes(p, p).saturating_mul(drift_count)
}

pub(crate) fn outer_hessian_dense_workspace_budget_bytes() -> usize {
    gam_runtime::resource::ResourcePolicy::default_library().max_single_materialization_bytes
}

pub(crate) fn dense_outer_hessian_workspace_fits(p: usize, k: usize) -> bool {
    outer_hessian_dense_workspace_bytes(p, k) <= outer_hessian_dense_workspace_budget_bytes()
}

pub(crate) fn generic_outer_hessian_scale_decision(
    n: usize,
    p: usize,
    k: usize,
) -> OuterHessianScaleDecision {
    if !dense_outer_hessian_workspace_fits(p, k) {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "dense_memory_budget",
        };
    }
    if k >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_k",
        };
    }
    if p >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_p",
        };
    }
    if n >= MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD
        && p >= MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N
    {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_n_moderate_p",
        };
    }
    if n.saturating_mul(p) >= MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_linear_work",
        };
    }
    OuterHessianScaleDecision {
        prefers_operator: false,
        reason: "below_crossover",
    }
}

pub(crate) fn callback_outer_hessian_scale_decision(
    n: usize,
    p: usize,
    k: usize,
) -> OuterHessianScaleDecision {
    if !dense_outer_hessian_workspace_fits(p, k) {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "dense_memory_budget",
        };
    }
    if k >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_k",
        };
    }
    if p >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_p",
        };
    }
    if n.saturating_mul(k).saturating_mul(k) >= CALLBACK_OUTER_HESSIAN_ROW_PAIR_WORK_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "callback_row_pair_work",
        };
    }
    OuterHessianScaleDecision {
        prefers_operator: false,
        reason: "below_crossover",
    }
}

pub fn outer_hessian_route_plan(
    n: usize,
    p: usize,
    k: usize,
    kernel_available: bool,
    callback_kernel: bool,
    subspace_trace: bool,
) -> OuterHessianRoutePlan {
    let dense_workspace_bytes = outer_hessian_dense_workspace_bytes(p, k);
    if !kernel_available {
        return OuterHessianRoutePlan {
            use_operator: false,
            reason: "kernel_absent",
            scale_prefers_operator: false,
            dense_workspace_bytes,
        };
    }

    let scale = if callback_kernel {
        callback_outer_hessian_scale_decision(n, p, k)
    } else {
        generic_outer_hessian_scale_decision(n, p, k)
    };
    let reason = if subspace_trace && scale.prefers_operator {
        "subspace_projected_operator"
    } else {
        scale.reason
    };
    OuterHessianRoutePlan {
        use_operator: scale.prefers_operator,
        reason,
        scale_prefers_operator: scale.prefers_operator,
        dense_workspace_bytes,
    }
}

/// Predicate for selecting the matrix-free Hv-operator outer-Hessian
/// representation over the dense `K × K` assembly.  Cost selects
/// representation, never capability — the operator path delivers the same
/// math as the dense path while avoiding large dense `p × p` drift storage
/// and pairwise row assembly when the model says those dominate.
pub fn prefer_outer_hessian_operator(n: usize, p: usize, k: usize) -> bool {
    generic_outer_hessian_scale_decision(n, p, k).prefers_operator
}

pub(crate) fn is_hessian_unavailable(error: &str) -> bool {
    error.starts_with(HESSIAN_UNAVAILABLE_PREFIX)
}
