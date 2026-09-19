//! Laplace-correction / mode-posterior sampler contract (trait-inversion #1521).
//!
//! gam-solve's REML inner loop (`#784` block-local quadrature correction)
//! and the custom-family never-fail covariance path call into the
//! gam-inference-tier NUTS / importance-sampling engine (`inference::hmc_io`,
//! ~8k lines) — an UP-edge that keeps gam-solve in the inference SCC.
//!
//! The COMPUTATION (NUTS, importance sampling, the directional-cubic eigen
//! diagnostic) is irreducibly above gam-solve and STAYS UP in `hmc_io`. Only
//! the neutral surface is contract-downed here, mirroring the `rho_posterior`
//! data-down (#1521):
//!
//! * the plain-DATA result carriers gam-solve reads
//!   ([`BlockQuadratureMarginal`], [`BlockQuadratureMoments`],
//!   [`LaplaceTrustworthiness`]);
//! * the caller-supplied [`BlockExcessTarget`] evaluator gam-solve IMPLEMENTS
//!   (its `Gam784BlockTarget`), so the trait must live below both;
//! * the CORRECTOR TRAIT [`LaplaceMarginalCorrector`] gam-solve calls THROUGH; the
//!   monolith / gam-inference implements it over `hmc_io` and injects the impl
//!   via the process-level registry below.
//!
//! The pure threshold math ([`laplace_skewness_threshold`],
//! [`laplace_trustworthiness_from_skewness`]) has no sampler dependency, so it is
//! moved down outright (gam-solve calls it directly), and so does the admission's
//! order selection ([`select_block_quadrature_orders`]), which only calls the
//! corrector.
//!
//! When no impl is registered (e.g. a build that never links the sampler tier)
//! the sampler getters return `None` and gam-solve degrades to its existing
//! decline paths — the `#784` correction returns zero (already a frequent
//! decline outcome) and the never-fail covariance path keeps the
//! optimizer-conditional covariance (already the `Err(reason)` fallback). The
//! contract therefore introduces no behavioral cliff and no stub.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use gam_linalg::matrix::DesignMatrix;
use ndarray::{Array1, Array2};

// ───────────────────────── data carriers (contract-down) ─────────────────────

/// Adaptive, block-local Laplace-trustworthiness verdict (issue #784): which
/// curvature directions are too non-Gaussian for the plain Laplace summary.
///
/// Field-for-field the monolith `hmc_io` type; that module re-exports this so
/// its construction sites name it unchanged.
#[derive(Clone, Debug)]
pub struct LaplaceTrustworthiness {
    /// Per-direction standardized skewness `γ_r`.
    pub directional_skewness: Array1<f64>,
    /// Indices of the directions whose skewness exceeds the auto-derived
    /// validity threshold (the curvature-heavy, non-Gaussian block).
    pub untrustworthy_directions: Vec<usize>,
    /// The auto-derived per-direction skewness threshold `τ(n)` actually used.
    pub threshold: f64,
    /// `max_r |γ_r|` across all directions (the global non-Gaussianity scale).
    pub max_abs_skewness: f64,
}

impl LaplaceTrustworthiness {
    /// Whether any curvature direction is too non-Gaussian for the plain
    /// Laplace summary, i.e. whether the higher-order correction / directional
    /// sampling fallback should engage at all.
    pub fn fallback_required(&self) -> bool {
        !self.untrustworthy_directions.is_empty()
    }
}

/// Quadrature-weighted moments of the per-node gradient channels — the
/// integration-side half of the #784 exact-gradient seam. All expectations are
/// under `p ∝ q·e^{−ΔF}` over the SAME deterministic nodes that produced the
/// value, so the spliced value and its assembled gradient cannot desync (#901).
#[derive(Clone, Debug)]
pub struct BlockQuadratureMoments {
    /// `E_p[t]`, length `m`.
    pub e_t: Array1<f64>,
    /// `E_p[t tᵀ]`, shape `m × m`.
    pub e_tt: Array2<f64>,
    /// `E_p[ngs(η̂+s)]`, length n — the displaced per-row score moment.
    pub e_neg_score: Array1<f64>,
    /// Column `r` = `E_p[t_r · ngs(η̂+s)]`, shape `n × m`.
    pub e_t_neg_score: Array2<f64>,
}

/// Block-local deterministic quadrature correction (issue #784).
///
/// `value` is `Δ_b` (added to the block marginal log-likelihood, subtracted from
/// the REML/LAML cost); `rho_gradient` is the explicit penalty-score channel (a)
/// of the gradient exactness contract; `moments` carries the channels (b)–(d) the
/// gam-solve assembly contracts against fields it already owns.
#[derive(Clone, Debug)]
pub struct BlockQuadratureMarginal {
    /// `Δ_b`: additive correction to the block marginal log-likelihood.
    pub value: f64,
    /// `∂Δ_b/∂ρ`, length `rho_dim()` — explicit channel (a) ONLY.
    pub rho_gradient: Array1<f64>,
    /// Per-axis Gauss–Hermite orders of the product rule that produced `value`.
    pub axis_orders: Vec<usize>,
    /// For each axis `r`, the larger of `|Δ_b − Δ_b⁽ʳ⁾|` over the integrals
    /// repeated with axis `r` one and two orders lower (the same-parity rule
    /// tracks a sequence the next lower rule can cross), in the same
    /// log-likelihood units as `value`. `+∞` on an axis at order one, which has no
    /// lower rule.
    pub axis_quadrature_errors: Vec<f64>,
    /// The largest entry of `axis_quadrature_errors`, `0` for an empty block.
    pub quadrature_error: f64,
    /// Number of nodes in the product rule, `Π_r axis_orders[r]`.
    pub node_count: usize,
    /// Nodes per target batch while the rule was evaluated: the largest chunk the
    /// memory governor admitted, capped at `node_count`. `0` for an empty block.
    pub chunk_nodes: usize,
    /// Bytes held on the memory governor's ledger while the rule was evaluated: one
    /// chunk's target working set plus the accumulators and rule tables. `0` for an
    /// empty block.
    pub reservation_bytes: usize,
    /// Gradient-channel moments for the exact (b)–(d) assembly; `None` only when
    /// the block is empty (`m == 0`, where the correction is zero).
    pub moments: Option<BlockQuadratureMoments>,
}

/// Why the deterministic block quadrature could not be evaluated at the orders
/// it was asked for.
#[derive(Clone, Debug)]
pub enum BlockQuadratureRefusal {
    /// The process memory governor did not admit the working set of a one-node chunk
    /// of the product rule (its accumulators, rule tables and one node's target
    /// transients), or that working set overflows `usize`. The rule is streamed, so
    /// the node count itself never meets the budget.
    WorkingMemory {
        axis_orders: Vec<usize>,
        /// `None` when the node count itself overflows `usize`.
        node_count: Option<usize>,
        reason: String,
    },
    /// Axis `axis` asked for an order whose Gauss–Hermite rule carries a weight
    /// that underflows to zero, so the rule has passed the largest order whose
    /// nodes all carry representable mass.
    UnrepresentableOrder { axis: usize, order: usize },
    /// The axis was evaluated at every order through `max_representable_order` and is
    /// still unresolved there, so no representable order is left to raise it to (#784).
    /// `running_minimum` is the smallest paired difference the axis showed at any of
    /// those orders. The refusal is measured at the ceiling, never projected from a rate.
    UnresolvableAtRepresentableOrders {
        order: usize,
        running_minimum: f64,
        max_representable_order: usize,
    },
    /// Any other failure of the integration itself (non-positive curvature,
    /// infeasible nodes, non-finite output, a malformed order list).
    Integration(String),
}

impl std::fmt::Display for BlockQuadratureRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkingMemory {
                axis_orders,
                node_count,
                reason,
            } => match node_count {
                Some(nodes) => write!(
                    f,
                    "the {nodes}-node product rule at axis orders {axis_orders:?} is not admitted by \
                     the memory budget: {reason}"
                ),
                None => write!(
                    f,
                    "the product rule at axis orders {axis_orders:?} has a node count that \
                     overflows usize: {reason}"
                ),
            },
            Self::UnrepresentableOrder { axis, order } => write!(
                f,
                "axis {axis} at Gauss–Hermite order {order} carries a weight that underflows to \
                 zero"
            ),
            Self::UnresolvableAtRepresentableOrders {
                order,
                running_minimum,
                max_representable_order,
            } => write!(
                f,
                "unresolved through the largest representable Gauss–Hermite order: the axis \
                 reached order {order} (of {max_representable_order}), and its smallest paired \
                 difference at any order was {running_minimum:.4e}"
            ),
            Self::Integration(reason) => f.write_str(reason),
        }
    }
}

/// An admission whose order search stopped with an axis still unresolved.
#[derive(Clone, Debug)]
pub struct BlockQuadratureOrderRefusal {
    /// The unresolved axis the search was raising when it stopped.
    pub axis: usize,
    /// The orders in force when the search stopped.
    pub axis_orders: Vec<usize>,
    /// That axis's paired difference at `axis_orders` (`+∞` when no rule was
    /// ever evaluated).
    pub paired_error: f64,
    /// `min(|Δ_b|, next-order remainder)` at `axis_orders`.
    pub resolution_target: f64,
    /// Why the next orders could not be evaluated.
    pub cause: BlockQuadratureRefusal,
}

impl std::fmt::Display for BlockQuadratureOrderRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "axis {} stayed unresolved at axis orders {:?} (paired difference {:.4e} against \
             resolution target {:.4e}): {}",
            self.axis, self.axis_orders, self.paired_error, self.resolution_target, self.cause
        )
    }
}

// ───────────────────────── pure threshold math (moved down) ──────────────────

/// Auto-derive the per-direction skewness threshold `τ(n)` separating
/// Laplace-trustworthy directions from those that need the higher-order
/// correction / sampling fallback. Derived purely from the effective sample
/// size, no tunable flag: `(5/24)γ_r² > 1/n_eff ⇔ |γ_r| > sqrt((24/5)/n_eff)`.
pub fn laplace_skewness_threshold(n_eff: f64) -> f64 {
    if !(n_eff > 0.0) {
        return f64::INFINITY;
    }
    ((24.0 / 5.0) / n_eff).sqrt()
}

/// Adaptive, block-local Laplace-trustworthiness verdict (issue #784): flag the
/// directions whose standardized skewness exceeds [`laplace_skewness_threshold`].
/// No linear algebra of its own — consumes the directional cubic diagnostic.
pub fn laplace_trustworthiness_from_skewness(
    directional_skewness: &Array1<f64>,
    n_eff: f64,
) -> LaplaceTrustworthiness {
    let threshold = laplace_skewness_threshold(n_eff);
    let mut untrustworthy_directions = Vec::new();
    let mut max_abs_skewness = 0.0_f64;
    for (r, &gamma) in directional_skewness.iter().enumerate() {
        let abs_gamma = if gamma.is_finite() { gamma.abs() } else { 0.0 };
        max_abs_skewness = max_abs_skewness.max(abs_gamma);
        if abs_gamma > threshold {
            untrustworthy_directions.push(r);
        }
    }
    LaplaceTrustworthiness {
        directional_skewness: directional_skewness.clone(),
        untrustworthy_directions,
        threshold,
        max_abs_skewness,
    }
}

/// Whether one axis's paired difference resolves `resolution_target`. An exactly
/// zero difference is an exact rule on that axis, which resolves any target.
fn axis_resolved(paired_error: f64, resolution_target: f64) -> bool {
    paired_error == 0.0 || paired_error < resolution_target
}

/// Select each axis's Gauss–Hermite order once, at admission (#2623 ruling A).
///
/// The correction exists to remove the `O(1/n_eff)` Laplace term, so its
/// quadrature error must sit below the next-order remainder: an axis is resolved
/// when its paired difference with the two next lower rules is below
/// `min(|Δ_b|, next_order_remainder)`, with `next_order_remainder = 1/n_eff²`.
///
/// Every axis starts at order four. At orders two and three one of the lower
/// rules is the single node at the mode, where `ΔF = 0`, so that axis's paired
/// difference is `|Δ_b|` itself and cannot resolve `min(|Δ_b|, ·)`.
///
/// Each step raises ONE unresolved axis by one order (#784): the axis with the most
/// raises still projected from its own measured contraction rate
/// (`projected_remaining_raises`), ties going to the smaller node growth `(o+1)/o`,
/// i.e. the higher order, and then to the lower index. Raising every unresolved axis
/// together multiplied the node count by every axis's factor per step; one axis at a
/// time grows it by one factor and stops each axis at the first order that resolves
/// it. Every step is published through
/// [`LaplaceMarginalCorrector::publish_order_search_step`], with the node count
/// projected at the resolving orders.
///
/// The search ends when every axis is resolved, or with a typed refusal naming the
/// unresolved axis:
/// - when the axis to raise already sits at the largest representable order and is
///   still unresolved there ([`BlockQuadratureRefusal::UnresolvableAtRepresentableOrders`]).
///   Each step raises one axis by one order, so the search makes at most
///   `m·(max_representable_order − 3)` requests before every axis is resolved or one is
///   refused;
/// - or when the corrector refuses the next orders (a one-node chunk the memory budget
///   does not admit, or a rule past the representable order).
///
/// No refusal is projected from a measured rate. The paired difference at order `o` is
/// `max(|Q_o − Q_{o−1}|, |Q_o − Q_{o−2}|)`, and a Gauss–Hermite rule of order `k` is
/// exact only through degree `2k − 1`. So through the first orders the paired difference
/// has the size of the lower rules' own errors, and a ratio of two of them measures which
/// degrees rules two orders apart integrate, not how fast the axis contracts. On the q5
/// fixture's eight axes the first judged ratio, 4 → 5, was ×0.45 to ×0.94, and the next,
/// 5 → 6, was ×0.044 to ×0.14 (job 1215221). A stop that projected from the first ratio
/// refused the q6 and q8 blocks at order 5, where each axis's own one-dimensional ladder
/// at the mode resolves at order 9 or 11 (job 1232529).
pub fn select_block_quadrature_orders(
    corrector: &dyn LaplaceMarginalCorrector,
    target: &dyn BlockExcessTarget,
    next_order_remainder: f64,
) -> Result<BlockQuadratureMarginal, BlockQuadratureOrderRefusal> {
    let m = target.block_dim();
    let max_representable_order = corrector.max_representable_order();
    let mut axis_orders = vec![4usize; m];
    // Each axis's latest paired difference at every order it has been evaluated at,
    // from which its contraction rate is measured.
    let mut errors_by_order = vec![BTreeMap::<usize, f64>::new(); m];
    // The unresolved axis raised on the last step, with its paired difference and
    // target, so a refusal of the raised rule names what it was raising.
    let mut raising: Option<(usize, f64, f64)> = None;
    loop {
        let marginal = match corrector.block_quadrature_marginal_correction(target, &axis_orders)
        {
            Ok(marginal) => marginal,
            Err(cause) => {
                let (axis, paired_error, resolution_target) =
                    raising.unwrap_or((0, f64::INFINITY, next_order_remainder));
                return Err(BlockQuadratureOrderRefusal {
                    axis,
                    axis_orders,
                    paired_error,
                    resolution_target,
                    cause,
                });
            }
        };
        let resolution_target = marginal.value.abs().min(next_order_remainder);
        for (axis, &error) in marginal.axis_quadrature_errors.iter().enumerate() {
            errors_by_order[axis].insert(axis_orders[axis], error);
        }
        let mut next: Option<(usize, f64)> = None;
        let mut projected_nodes = Some(1usize);
        let mut unmeasured = Vec::new();
        for (axis, &error) in marginal.axis_quadrature_errors.iter().enumerate() {
            let order = axis_orders[axis];
            if axis_resolved(error, resolution_target) {
                projected_nodes = projected_nodes.and_then(|nodes| nodes.checked_mul(order));
                continue;
            }
            let remaining =
                projected_remaining_raises(&errors_by_order[axis], order, error, resolution_target);
            if remaining.is_finite() {
                // `as` saturates a projection past `usize`; the checked sum reports it.
                projected_nodes = projected_nodes.and_then(|nodes| {
                    order
                        .checked_add(remaining.ceil() as usize)
                        .and_then(|projected_order| nodes.checked_mul(projected_order))
                });
            } else {
                unmeasured.push(axis);
            }
            let raises_first = match next {
                None => true,
                Some((incumbent, incumbent_remaining)) => {
                    remaining > incumbent_remaining
                        || (remaining == incumbent_remaining && order > axis_orders[incumbent])
                }
            };
            if raises_first {
                next = Some((axis, remaining));
            }
        }
        let Some((axis, remaining)) = next else {
            return Ok(marginal);
        };
        let projected_node_count = if unmeasured.is_empty() {
            projected_nodes.map_or(ProjectedNodeCount::Overflow, ProjectedNodeCount::Nodes)
        } else {
            ProjectedNodeCount::Unmeasured { axes: unmeasured }
        };
        corrector.publish_order_search_step(&BlockQuadratureOrderStep {
            axis_orders: axis_orders.clone(),
            node_count: marginal.node_count,
            chunk_nodes: marginal.chunk_nodes,
            reservation_bytes: marginal.reservation_bytes,
            axis_quadrature_errors: marginal.axis_quadrature_errors.clone(),
            abs_value: marginal.value.abs(),
            resolution_target,
            raised_axis: axis,
            projected_remaining_raises: remaining,
            projected_node_count,
        });
        // The axis to raise already sits at the largest representable order, unresolved
        // there, so no representable order is left to raise it to (#784).
        if axis_orders[axis] >= max_representable_order {
            let running_minimum = errors_by_order[axis]
                .values()
                .copied()
                .fold(f64::INFINITY, f64::min);
            return Err(BlockQuadratureOrderRefusal {
                axis,
                paired_error: marginal.axis_quadrature_errors[axis],
                resolution_target,
                cause: BlockQuadratureRefusal::UnresolvableAtRepresentableOrders {
                    order: axis_orders[axis],
                    running_minimum,
                    max_representable_order,
                },
                axis_orders,
            });
        }
        raising = Some((axis, marginal.axis_quadrature_errors[axis], resolution_target));
        axis_orders[axis] += 1;
    }
}

/// How many raises an unresolved axis still needs, projected from its own measured
/// contraction rate (#784): `ln(e/τ) / ln(1/q̂)`, where `q̂` is the slower of the
/// axis's last two measured ratios `e(o)/e(o−1)` and `e(o−1)/e(o−2)`. An axis with no
/// measured ratio yet, or with a measured ratio that does not contract, projects `+∞`.
/// The projection only orders the raises and is published as evidence. It never refuses
/// an axis, since its early ratios are not the axis's contraction (see
/// [`select_block_quadrature_orders`]).
fn projected_remaining_raises(
    errors_by_order: &BTreeMap<usize, f64>,
    order: usize,
    error: f64,
    resolution_target: f64,
) -> f64 {
    let ratio_at = |upper: usize| -> Option<f64> {
        let lower = upper.checked_sub(1)?;
        Some(errors_by_order.get(&upper)? / errors_by_order.get(&lower)?)
    };
    let ratios: Vec<f64> = [Some(order), order.checked_sub(1)]
        .into_iter()
        .flatten()
        .filter_map(ratio_at)
        .collect();
    let Some(rate) = ratios.iter().copied().reduce(f64::max) else {
        return f64::INFINITY;
    };
    if ratios.iter().all(|&ratio| ratio > 0.0 && ratio < 1.0) {
        (error / resolution_target).ln() / rate.recip().ln()
    } else {
        f64::INFINITY
    }
}

/// The node count of the product rule at the orders every unresolved axis is
/// projected to resolve at (#784).
#[derive(Clone, Debug)]
pub enum ProjectedNodeCount {
    Nodes(usize),
    /// The projected node count overflows `usize`.
    Overflow,
    /// These unresolved axes have no measured contracting rate yet, so no resolving
    /// order is projected for them.
    Unmeasured { axes: Vec<usize> },
}

impl std::fmt::Display for ProjectedNodeCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nodes(nodes) => write!(f, "{nodes}"),
            Self::Overflow => f.write_str("overflows usize"),
            Self::Unmeasured { axes } => {
                write!(f, "unprojected (axes {axes:?} have no contracting measured rate)")
            }
        }
    }
}

/// One step of [`select_block_quadrature_orders`]: the rule it evaluated and the
/// unresolved axis it raises next.
#[derive(Clone, Debug)]
pub struct BlockQuadratureOrderStep {
    /// The orders of the evaluated rule.
    pub axis_orders: Vec<usize>,
    pub node_count: usize,
    pub chunk_nodes: usize,
    pub reservation_bytes: usize,
    pub axis_quadrature_errors: Vec<f64>,
    /// `|Δ_b|` at `axis_orders`.
    pub abs_value: f64,
    /// `min(|Δ_b|, next-order remainder)` at `axis_orders`.
    pub resolution_target: f64,
    /// The unresolved axis the search raises by one order next.
    pub raised_axis: usize,
    /// That axis's projected raises left, from its measured contraction rate.
    pub projected_remaining_raises: f64,
    pub projected_node_count: ProjectedNodeCount,
}

impl std::fmt::Display for BlockQuadratureOrderStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "orders {:?}: {} nodes in chunks of {} ({} bytes reserved), paired errors {:?} \
             against target {:.4e} (|Δ_b| {:.4e}); raising axis {} with {:.3} projected raises \
             left; nodes at the projected resolving orders: {}",
            self.axis_orders,
            self.node_count,
            self.chunk_nodes,
            self.reservation_bytes,
            self.axis_quadrature_errors,
            self.resolution_target,
            self.abs_value,
            self.raised_axis,
            self.projected_remaining_raises,
            self.projected_node_count,
        )
    }
}

// ───────────────────────── caller-supplied excess evaluator ──────────────────

/// Caller-supplied evaluator for the non-Gaussian remainder `ΔF(t)` of the local
/// log-posterior, restricted to the curvature-heavy block subspace (issue #784).
///
/// Implemented by gam-solve's `Gam784BlockTarget`; consumed by
/// [`LaplaceMarginalCorrector::block_quadrature_marginal_correction`]. Lives in this
/// neutral crate so both the implementor (gam-solve) and the sampler impl (the
/// gam-inference monolith) name the same trait without an SCC edge.
pub trait BlockExcessTarget {
    /// Dimension `m` of the block subspace (number of untrustworthy directions
    /// being integrated).
    fn block_dim(&self) -> usize;
    /// Number of outer ρ coordinates the gradient is reported against.
    fn rho_dim(&self) -> usize;
    /// Block curvatures `λ_r` (the H-eigenvalues of the integrated directions),
    /// length `block_dim()`.
    fn block_curvatures(&self) -> &Array1<f64>;
    /// Non-Gaussian remainder `ΔF(t)` at whitened block displacement `t`
    /// (length `block_dim()`).
    fn excess(&self, t: &Array1<f64>) -> f64;
    /// The rounding band of [`Self::excess`] at `t`: how far the computed `ΔF(t)`
    /// can sit from the exact one, from the magnitudes its sums accumulate.
    /// `ΔF` is a cancellation of like-sized terms at small `t`, so a quantity
    /// formed from differences of `ΔF` is resolved only above this band (#784).
    fn excess_rounding_band(&self, t: &Array1<f64>) -> f64;
    /// ρ-gradient `∂ΔF/∂ρ_k` at the same `t`, length `rho_dim()` — the explicit
    /// penalty-score channel (a).
    fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64>;
    /// Per-row displaced score `∂(D(η̂+s(t))/2φ)/∂η` evaluated at `η̂ + s(t)`
    /// (length = number of observation rows): the only per-draw ingredient of
    /// the exact-gradient channels (b)–(d) the assembly side cannot reconstruct.
    /// A row-domain failure rejects the complete score atomically.
    fn displaced_neg_score(&self, t: &Array1<f64>) -> Result<Array1<f64>, String>;
    /// The same per-row score channel at the undisplaced mode `η̂`.
    fn base_neg_score(&self) -> Result<Array1<f64>, String>;

    /// The most bytes one node holds live while
    /// [`Self::excess_with_displaced_neg_score_batch`] or [`Self::excess_batch`]
    /// evaluates it inside a batch: everything the implementor allocates per node
    /// (displacements, the returned score and its entry, row transients). A batch of
    /// `B` nodes holds at most `B` times this, and the corrector reserves exactly that
    /// on the memory governor before it evaluates a chunk, so an implementor that
    /// changes what a batch allocates changes this with it. `None` when the count
    /// overflows `usize`.
    fn node_working_bytes(&self) -> Option<usize>;

    /// Fused `(excess(t), displaced_neg_score(t))`. The returned score is `None`
    /// exactly when the excess is non-finite (an infeasible draw the sampler
    /// discards before reading the score). The default preserves the two-call
    /// behavior; implementors override to share the displacement + jet.
    fn excess_with_displaced_neg_score(&self, t: &Array1<f64>) -> (f64, Option<Array1<f64>>) {
        let excess = self.excess(t);
        if excess.is_finite() {
            match self.displaced_neg_score(t) {
                Ok(score) => (excess, Some(score)),
                Err(_) => (f64::INFINITY, None),
            }
        } else {
            (excess, None)
        }
    }

    /// Batched [`Self::excess_with_displaced_neg_score`] over many whitened draws
    /// (one draw per COLUMN, shape `block_dim() × n_draws`). Batching may only
    /// change HOW the shared linear algebra is computed (one BLAS-3 product over
    /// all columns), never WHAT is computed. The default preserves the per-column
    /// behavior exactly; the GLM implementor overrides it.
    fn excess_with_displaced_neg_score_batch(
        &self,
        draws: &Array2<f64>,
    ) -> Vec<(f64, Option<Array1<f64>>)> {
        let n_draws = draws.ncols();
        let mut out = Vec::with_capacity(n_draws);
        let mut t = Array1::<f64>::zeros(draws.nrows());
        for s in 0..n_draws {
            t.assign(&draws.column(s));
            out.push(self.excess_with_displaced_neg_score(&t));
        }
        out
    }

    /// Batched excess-only evaluation for a matrix of quadrature nodes. The
    /// coarse error rule does not consume score moments, so requiring them
    /// would duplicate the expensive row-score work solely to discard it.
    fn excess_batch(&self, nodes: &Array2<f64>) -> Vec<f64> {
        let mut out = Vec::with_capacity(nodes.ncols());
        let mut t = Array1::<f64>::zeros(nodes.nrows());
        for column in nodes.columns() {
            t.assign(&column);
            out.push(self.excess(&t));
        }
        out
    }
}

// ───────────────────────── injected sampler traits ───────────────────────────

/// The gam-inference-tier sampler for the #784 block-local Laplace correction.
///
/// Implementable UP in an inference tier over
/// (`laplace_directional_cubic_diagnostic` + `block_quadrature_marginal_correction`)
/// and injected DOWN via [`set_laplace_marginal_corrector`]. The standard
/// estimator installs the deterministic quadrature implementation; alternate
/// embeddings may install another implementation before process initialization.
pub trait LaplaceMarginalCorrector: Send + Sync {
    /// Per-direction standardized cubic skewness `γ_r` of the local posterior:
    /// returns `(max_r |γ_r|, γ)`. Pure eigen-diagnostic (no sampling), but kept
    /// behind the trait because it lives in the sampler module up-tier.
    fn directional_cubic_diagnostic(
        &self,
        hessian: &Array2<f64>,
        design: &DesignMatrix,
        c_weights: &Array1<f64>,
        refine_supremum: bool,
    ) -> Result<(f64, Array1<f64>), String>;

    /// Integrate `Δ_b` and its ρ-gradient against the local Laplace Gaussian with
    /// a product Gauss–Hermite rule of the given per-axis orders, contracting the
    /// caller-supplied [`BlockExcessTarget`].
    fn block_quadrature_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
        axis_orders: &[usize],
    ) -> Result<BlockQuadratureMarginal, BlockQuadratureRefusal>;

    /// Publish one step of [`select_block_quadrature_orders`]: the rule it evaluated,
    /// the unresolved axis it raises next, and the node count projected at the
    /// resolving orders. The standard corrector logs it at info, so a search that
    /// grinds names what it is grinding on (#784).
    fn publish_order_search_step(&self, step: &BlockQuadratureOrderStep);

    /// The largest Gauss–Hermite order this corrector's rule builder represents: every
    /// order up to it builds a rule whose weights are all positive, and the next order
    /// does not, so a search that raises one order at a time is refused there. The order
    /// search compares its projected resolving orders against it (#784).
    fn max_representable_order(&self) -> usize;
}

// ───────────────────────── process-level injection registry ──────────────────

static LAPLACE_MARGINAL_CORRECTOR: OnceLock<Box<dyn LaplaceMarginalCorrector>> = OnceLock::new();

/// Register the #784 block-local Laplace corrector. First writer wins; a later
/// call is ignored so a re-init can never swap a live criterion mid-run.
pub fn set_laplace_marginal_corrector(
    corrector: Box<dyn LaplaceMarginalCorrector>,
) -> Result<(), Box<dyn LaplaceMarginalCorrector>> {
    LAPLACE_MARGINAL_CORRECTOR.set(corrector)
}

/// The registered #784 block-local Laplace corrector, or `None` when the
/// embedding has not initialized the inference tier.
pub fn laplace_marginal_corrector() -> Option<&'static dyn LaplaceMarginalCorrector> {
    LAPLACE_MARGINAL_CORRECTOR.get().map(|b| b.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── laplace_skewness_threshold ────────────────────────────────────────────

    #[test]
    fn threshold_is_infinity_for_zero_n_eff() {
        assert_eq!(laplace_skewness_threshold(0.0), f64::INFINITY);
    }

    #[test]
    fn threshold_is_infinity_for_negative_n_eff() {
        assert_eq!(laplace_skewness_threshold(-5.0), f64::INFINITY);
    }

    #[test]
    fn threshold_known_value() {
        // n_eff = 24/5 → sqrt((24/5) / (24/5)) = 1.0
        let n_eff = 24.0 / 5.0;
        let t = laplace_skewness_threshold(n_eff);
        assert!((t - 1.0).abs() < 1e-14, "threshold={t}");
    }

    #[test]
    fn threshold_decreases_as_n_eff_increases() {
        let t_small = laplace_skewness_threshold(10.0);
        let t_large = laplace_skewness_threshold(1000.0);
        assert!(
            t_large < t_small,
            "threshold should decrease with more data"
        );
    }

    // ── laplace_trustworthiness_from_skewness ─────────────────────────────────

    #[test]
    fn all_small_skewness_gives_no_untrustworthy_directions() {
        // With n_eff=1000, threshold ≈ 0.069; all |γ| < that
        let skewness = array![0.01_f64, -0.02, 0.005];
        let result = laplace_trustworthiness_from_skewness(&skewness, 1000.0);
        assert!(result.untrustworthy_directions.is_empty());
        assert!(!result.fallback_required());
    }

    #[test]
    fn large_skewness_flagged_as_untrustworthy() {
        // With n_eff=10, threshold ≈ 0.693; γ=2.0 exceeds it
        let skewness = array![0.1_f64, 2.0];
        let result = laplace_trustworthiness_from_skewness(&skewness, 10.0);
        assert!(result.untrustworthy_directions.contains(&1));
        assert!(!result.untrustworthy_directions.contains(&0));
        assert!(result.fallback_required());
    }

    #[test]
    fn max_abs_skewness_is_largest_abs_value() {
        let skewness = array![1.5_f64, -3.0, 2.0];
        let result = laplace_trustworthiness_from_skewness(&skewness, 1.0);
        assert!((result.max_abs_skewness - 3.0).abs() < 1e-14);
    }

    #[test]
    fn non_finite_skewness_treated_as_zero_for_max_abs() {
        let skewness = array![f64::NAN, 1.0];
        let result = laplace_trustworthiness_from_skewness(&skewness, 1.0);
        // NaN is treated as 0 in the loop; max_abs comes from 1.0
        assert!((result.max_abs_skewness - 1.0).abs() < 1e-14);
    }

    // ── LaplaceTrustworthiness::fallback_required ─────────────────────────────

    #[test]
    fn fallback_required_true_when_directions_nonempty() {
        let lt = LaplaceTrustworthiness {
            directional_skewness: array![1.0_f64],
            untrustworthy_directions: vec![0],
            threshold: 0.5,
            max_abs_skewness: 1.0,
        };
        assert!(lt.fallback_required());
    }

    #[test]
    fn fallback_required_false_when_directions_empty() {
        let lt = LaplaceTrustworthiness {
            directional_skewness: array![0.1_f64],
            untrustworthy_directions: vec![],
            threshold: 0.5,
            max_abs_skewness: 0.1,
        };
        assert!(!lt.fallback_required());
    }
}
