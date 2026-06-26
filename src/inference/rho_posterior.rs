//! Exact marginal smoothing inference over the smoothing parameters `ρ`
//! (issue #938): the Tier-0 **PSIS certificate**, plus the auto-selected
//! escalation tiers — Tier-1 **Gauss-Hermite quadrature** over `ρ` (`K ≤ 4`,
//! [`rho_posterior_quadrature`]) and Tier-2 **NUTS over `ρ`** with the exact
//! profiled gradient (`K ≤ 16`, [`rho_posterior_nuts`]), routed by
//! [`escalate_rho_posterior`] when the certificate refuses to certify the
//! plug-in.
//!
//! Every GAM ecosystem conditions inference on the estimated smoothing
//! parameters `ρ̂`; intervals from `V(β̂|ρ̂)` undercover because they ignore
//! `ρ`-uncertainty. The honest marginal posterior factorizes as
//! `π(β, ρ | y) = π(β | ρ, y) · π(ρ | y)`, where `π(ρ|y) ∝ exp(−criterion(ρ))`
//! is exactly the LAML/REML objective the outer optimizer already minimizes
//! with exact gradients.
//!
//! Tier 0 turns "should I worry about `ρ`-uncertainty?" — currently folklore —
//! into a *computed* diagnostic on every fit, with no MCMC:
//!
//! 1. Treat the Laplace approximation at `ρ̂`, `N(ρ̂, H_ρ⁻¹)` (with `H_ρ` the
//!    exact outer Hessian), as the importance proposal.
//! 2. Draw `M` whitened samples `z_m ~ N(0, I)`, map them to
//!    `ρ_m = ρ̂ + L z_m` where `L Lᵀ = H_ρ⁻¹`.
//! 3. The importance weight is `w_m = π(ρ_m|y) / proposal(ρ_m)`; in log-space the
//!    Gaussian proposal's quadratic cancels to the whitened norm, giving
//!    `log w_m = −criterion(ρ_m) + criterion(ρ̂) + ½‖z_m‖²` (the `criterion(ρ̂)`
//!    shift makes the weights self-normalized and finite).
//! 4. Pareto-smooth the weights ([`crate::psis`]) and read the
//!    Zhang–Stephens tail shape `k̂`. `k̂ < 0.5` ⇒ the plug-in + first-order
//!    correction answer is **certified** adequate; `0.5 ≤ k̂ ≤ 0.7` ⇒ usable as
//!    a self-normalized importance correction; `k̂ > 0.7` ⇒ the Laplace proposal
//!    is a poor fit and the honest path is a full quadrature/NUTS escalation.
//!
//! The certificate is deterministic: the whitened draws come from a fixed-seed
//! splitmix64 + Box–Muller stream, so the same fit yields the same `k̂` every
//! run.

use crate::estimate::EstimationError;
use crate::psis::pareto_smooth_weights;
use crate::solver::rho_optimizer::OuterObjective;
use ndarray::{Array1, Array2};

// The `ρ`-posterior certificate/escalation DATA types were contract-downed to
// the neutral `gam-problem` crate (#1521) so gam-solve can store/return them
// without a back-edge into gam-inference. The COMPUTATION below (PSIS
// certificate, Tier-1 quadrature, Tier-2 NUTS via `hmc_io`) stays here and
// constructs these types under their original names via this re-export.
pub use gam_problem::rho_posterior::{
    RhoCertificate, RhoMixtureNode, RhoPosteriorCertificate, RhoPosteriorEscalation,
    RhoPosteriorMixture, RhoPosteriorSamples,
};

/// One deterministic Tier-1 quadrature node for `π(ρ|y)`.
#[derive(Debug, Clone)]
pub struct RhoQuadratureNode {
    /// Smoothing parameters at this node.
    pub rho: Array1<f64>,
    /// Normalized node probability after reweighting the Gaussian proposal by the
    /// exact profiled criterion.
    pub weight: f64,
    /// Normalized log node probability.
    pub log_weight: f64,
    /// Exact profiled criterion value at the node.
    pub cost: f64,
    /// Exact profiled outer gradient at the node.
    pub gradient: Array1<f64>,
}

/// Tier-1 higher-order Laplace/quadrature approximation to `π(ρ|y)`.
#[derive(Debug, Clone)]
pub struct RhoQuadratureMixture {
    pub nodes: Vec<RhoQuadratureNode>,
    pub effective_sample_size: f64,
    pub max_gradient_norm: f64,
}

/// Mixture-corrected coefficient posterior moments (law of total
/// variance/expectation over the `ρ` mixture). See
/// [`mixture_coefficient_covariance`].
#[derive(Debug, Clone)]
pub struct MixtureCoefficientCovariance {
    /// Mixture posterior mean `β̄ = Σ_m w_m β̂(ρ_m)`.
    pub beta_bar: Array1<f64>,
    /// `Vβ_marginal = Σ_m w_m [Vb(ρ_m) + (β̂(ρ_m)−β̄)(β̂(ρ_m)−β̄)ᵀ]`.
    pub covariance: Array2<f64>,
}

/// Largest `K` for which the Tier-1 Gauss-Hermite product grid is affordable
/// (3–5 nodes per axis ⇒ at most 81–125 criterion evaluations).
pub const TIER1_MAX_DIM: usize = 4;
/// Largest `K` for which the Tier-2 NUTS escalation runs; beyond this the fit
/// honestly reports that escalation is unavailable.
pub const TIER2_MAX_DIM: usize = 16;
/// Post-warmup draw budget for the auto-selected Tier-2 escalation. Each
/// leapfrog step is one warm inner profile solve, so the budget is deliberately
/// modest: the whitened `ρ`-posterior is a smooth, near-Gaussian, low-dim
/// target where a few hundred draws already pin the first two moments.
const ESCALATION_NUTS_SAMPLES: usize = 256;
/// Deterministic seed for the auto-selected Tier-2 escalation (no clock).
const ESCALATION_NUTS_SEED: u64 = 0x938_5EED_0938_5EED;
const RHO_POSTERIOR_NUTS_WARMUP_FLOOR: usize = 32;

const DEFAULT_M: usize = 64;
const CERTIFICATE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Deterministic standard-normal stream (splitmix64 + Box–Muller). No RNG / env
/// dependency: the same seed yields the same draws every run.
struct DetNormal {
    state: u64,
}
impl DetNormal {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn uniform(&mut self) -> f64 {
        let z = crate::linalg::utils::splitmix64(&mut self.state);
        (((z >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Lower-triangular Cholesky `L` (`L Lᵀ = a`) with diagonal jitter for safety.
/// Returns `None` when `a` is not positive definite even after jitter.
fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n {
        return None;
    }
    let scale = (0..n)
        .map(|i| a[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let jitter = 1e-10 * scale;
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut d = a[[j, j]] + jitter;
        for k in 0..j {
            d -= l[[j, k]] * l[[j, k]];
        }
        if !(d.is_finite() && d > 0.0) {
            return None;
        }
        let ljj = d.sqrt();
        l[[j, j]] = ljj;
        for i in (j + 1)..n {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = s / ljj;
        }
    }
    Some(l)
}

/// Solve `H_ρ⁻¹`'s Cholesky `L_inv` (with `L_inv L_invᵀ = H_ρ⁻¹`) from the outer
/// Hessian `H_ρ`. We factor `H_ρ = R Rᵀ` (lower `R`) and use `L_inv = R⁻ᵀ`: then
/// `L_inv L_invᵀ = R⁻ᵀ R⁻¹ = (R Rᵀ)⁻¹ = H_ρ⁻¹`. Mapping `ρ_m = ρ̂ + L_inv z_m`
/// gives draws with covariance `H_ρ⁻¹`, and `‖z_m‖² = (ρ_m−ρ̂)ᵀ H_ρ (ρ_m−ρ̂)`.
fn whitening_factor_from_outer_hessian(outer_hessian: &Array2<f64>) -> Option<Array2<f64>> {
    let r = cholesky_lower(outer_hessian)?;
    let n = r.nrows();
    // Invert-transpose: solve R z = e_i columns to build R⁻¹, then transpose.
    // L_inv = R⁻ᵀ, so column j of L_inv is row j of R⁻¹. Build R⁻¹ by forward
    // substitution against identity columns.
    let mut r_inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        // Solve R x = e_col (lower triangular forward substitution).
        let mut x = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = if i == col { 1.0 } else { 0.0 };
            for k in 0..i {
                acc -= r[[i, k]] * x[k];
            }
            let rii = r[[i, i]];
            if !(rii.is_finite() && rii.abs() > 0.0) {
                return None;
            }
            x[i] = acc / rii;
        }
        for i in 0..n {
            r_inv[[i, col]] = x[i];
        }
    }
    // L_inv = R⁻ᵀ.
    let mut l_inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            l_inv[[i, j]] = r_inv[[j, i]];
        }
    }
    Some(l_inv)
}

/// Estimate the local outer Hessian by central differencing the profiled-exact
/// `OuterObjective::eval` gradient.
///
/// This is for objectives such as the SAE REML surface that expose exact
/// gradients but not a dense Hessian. The criterion values are never
/// finite-differenced; only the exact profiled gradient is sampled.
pub fn rho_hessian_from_profiled_exact_gradient(
    objective: &mut dyn OuterObjective,
    rho_hat: &Array1<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let k = rho_hat.len();
    let mut hessian = Array2::<f64>::zeros((k, k));
    if k == 0 {
        return Ok(hessian);
    }
    let base_step = 1.0e-4;
    for j in 0..k {
        let step = base_step * rho_hat[j].abs().max(1.0);
        let mut plus = rho_hat.clone();
        let mut minus = rho_hat.clone();
        plus[j] += step;
        minus[j] -= step;
        let gp = objective.eval(&plus)?.gradient;
        let gm = objective.eval(&minus)?.gradient;
        if gp.len() != k || gm.len() != k {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "rho_hessian_from_profiled_exact_gradient: gradient length mismatch: expected {k}, got {} and {}",
                gp.len(),
                gm.len()
            )));
        }
        for i in 0..k {
            hessian[[i, j]] = (gp[i] - gm[i]) / (2.0 * step);
        }
    }
    crate::matrix::symmetrize_in_place(&mut hessian);
    Ok(hessian)
}

/// Gauss-Hermite rules for the STANDARD NORMAL weight (probabilists'
/// convention: nodes are `√2·x_i`, weights `w_i/√π` of the physicists' rule, so
/// the weights sum to 1 and the rule integrates polynomials of degree
/// `2n−1` exactly against `N(0,1)`).
fn standard_normal_gh_rule(nodes_per_axis: usize) -> Option<&'static [(f64, f64)]> {
    match nodes_per_axis {
        3 => Some(&[
            (-1.732_050_807_568_877_2, 1.0 / 6.0),
            (0.0, 2.0 / 3.0),
            (1.732_050_807_568_877_2, 1.0 / 6.0),
        ]),
        5 => Some(&[
            (-2.856_970_013_872_805_6, 1.125_741_132_772_071_8e-2),
            (-1.355_626_179_974_265_9, 2.220_759_220_056_126_4e-1),
            (0.0, 5.333_333_333_333_333e-1),
            (1.355_626_179_974_265_9, 2.220_759_220_056_126_4e-1),
            (2.856_970_013_872_805_6, 1.125_741_132_772_071_8e-2),
        ]),
        _ => None,
    }
}

fn enumerate_gh_product(
    dim: usize,
    rule: &[(f64, f64)],
    axis: usize,
    z: &mut Array1<f64>,
    log_w: f64,
    out: &mut Vec<(Array1<f64>, f64)>,
) {
    if axis == dim {
        out.push((z.clone(), log_w));
        return;
    }
    for &(node, weight) in rule {
        z[axis] = node;
        enumerate_gh_product(dim, rule, axis + 1, z, log_w + weight.ln(), out);
    }
}

/// One normalized node from the shared quadrature core: `(ρ, cost, optional
/// exact gradient, normalized weight, normalized log-weight)`. Infeasible nodes
/// carry `cost = +∞` and zero weight.
struct NormalizedQuadratureNode {
    rho: Array1<f64>,
    cost: f64,
    gradient: Option<Array1<f64>>,
    weight: f64,
    log_weight: f64,
}

/// Shared Tier-1 quadrature core (#938): whiten by the exact outer Hessian,
/// enumerate the Gauss-Hermite product grid, reweight each node by the exact
/// profiled criterion `exp(−V(ρ_m) + V(ρ̂) + ½‖z_m‖²) × GH-weight`, and
/// normalize. Both public entry points (the `OuterObjective` form that carries
/// exact gradients and the criterion-closure form) are thin adapters over this
/// single implementation.
fn quadrature_nodes_core<E>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    nodes_per_axis: usize,
    cost_hat: f64,
    mut eval_node: E,
) -> Result<(Vec<NormalizedQuadratureNode>, f64), EstimationError>
where
    E: FnMut(&Array1<f64>) -> Result<Option<(f64, Option<Array1<f64>>)>, EstimationError>,
{
    let k = rho_hat.len();
    if k == 0 || outer_hessian.nrows() != k || outer_hessian.ncols() != k {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: rho/Hessian shape mismatch".to_string(),
        ));
    }
    if k > TIER1_MAX_DIM {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "rho_posterior_quadrature: product quadrature is capped at K<={TIER1_MAX_DIM}, got {k}"
        )));
    }
    let rule = standard_normal_gh_rule(nodes_per_axis).ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(format!(
            "rho_posterior_quadrature: unsupported nodes_per_axis {nodes_per_axis}"
        ))
    })?;
    let l_inv = whitening_factor_from_outer_hessian(outer_hessian).ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: outer Hessian is not positive definite".to_string(),
        )
    })?;
    if !cost_hat.is_finite() {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: non-finite criterion at rho_hat".to_string(),
        ));
    }

    let mut product_nodes = Vec::new();
    enumerate_gh_product(
        k,
        rule,
        0,
        &mut Array1::<f64>::zeros(k),
        0.0,
        &mut product_nodes,
    );
    let mut raw_nodes = Vec::with_capacity(product_nodes.len());
    let mut max_log_weight = f64::NEG_INFINITY;
    for (z, log_base_weight) in product_nodes {
        let mut rho = rho_hat.clone();
        for i in 0..k {
            let mut acc = 0.0;
            for j in 0..k {
                acc += l_inv[[i, j]] * z[j];
            }
            rho[i] += acc;
        }
        let (cost, gradient, log_weight) = match eval_node(&rho)? {
            Some((cost, gradient)) if cost.is_finite() => {
                let half_norm_sq = 0.5 * z.iter().map(|&v| v * v).sum::<f64>();
                (
                    cost,
                    gradient,
                    log_base_weight - cost + cost_hat + half_norm_sq,
                )
            }
            // Infeasible node: zero importance weight, never fatal.
            _ => (f64::INFINITY, None, f64::NEG_INFINITY),
        };
        if log_weight.is_finite() {
            max_log_weight = max_log_weight.max(log_weight);
        }
        raw_nodes.push((rho, cost, gradient, log_weight));
    }
    if !max_log_weight.is_finite() {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: all quadrature nodes were non-finite".to_string(),
        ));
    }
    let mut total = 0.0;
    let mut scaled = Vec::with_capacity(raw_nodes.len());
    for (_, _, _, log_weight) in &raw_nodes {
        let w = if log_weight.is_finite() {
            (*log_weight - max_log_weight).exp()
        } else {
            0.0
        };
        total += w;
        scaled.push(w);
    }
    if !(total.is_finite() && total > 0.0) {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: non-positive normalized mass".to_string(),
        ));
    }

    let mut nodes = Vec::with_capacity(raw_nodes.len());
    let mut sum_sq = 0.0;
    for ((rho, cost, gradient, log_weight), scaled_weight) in raw_nodes.into_iter().zip(scaled) {
        let weight = scaled_weight / total;
        sum_sq += weight * weight;
        nodes.push(NormalizedQuadratureNode {
            rho,
            cost,
            gradient,
            weight,
            log_weight: log_weight - max_log_weight - total.ln(),
        });
    }
    let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
    Ok((nodes, ess))
}

/// Tier-1 deterministic higher-order Laplace quadrature over `ρ`, the
/// `OuterObjective` form.
///
/// The input Hessian defines the local Gaussian proposal around `rho_hat`; each
/// Gauss-Hermite node is reweighted by the exact profiled criterion,
/// `exp(-V(ρ_node) + V(ρ_hat) + 0.5 ||z||²)`, and stores the exact profiled
/// gradient from `OuterObjective::eval`.
pub fn rho_posterior_tier1_quadrature(
    objective: &mut dyn OuterObjective,
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    nodes_per_axis: usize,
) -> Result<RhoQuadratureMixture, EstimationError> {
    let cost_hat = objective.eval_cost(rho_hat)?;
    let (core_nodes, effective_sample_size) =
        quadrature_nodes_core(rho_hat, outer_hessian, nodes_per_axis, cost_hat, |rho| {
            let eval = objective.eval(rho)?;
            Ok(Some((eval.cost, Some(eval.gradient))))
        })?;
    let k = rho_hat.len();
    let mut nodes = Vec::with_capacity(core_nodes.len());
    let mut max_gradient_norm = 0.0_f64;
    for node in core_nodes {
        let gradient = node.gradient.unwrap_or_else(|| Array1::zeros(k));
        let grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        max_gradient_norm = max_gradient_norm.max(grad_norm);
        nodes.push(RhoQuadratureNode {
            rho: node.rho,
            weight: node.weight,
            log_weight: node.log_weight,
            cost: node.cost,
            gradient,
        });
    }
    Ok(RhoQuadratureMixture {
        nodes,
        effective_sample_size,
        max_gradient_norm,
    })
}

/// Posterior moments of a normalized discrete mixture over `ρ`.
fn mixture_moments(nodes: &[RhoMixtureNode], k: usize) -> (Array1<f64>, Array2<f64>) {
    let mut mean = Array1::<f64>::zeros(k);
    for node in nodes {
        for i in 0..k {
            mean[i] += node.weight * node.rho[i];
        }
    }
    let mut covariance = Array2::<f64>::zeros((k, k));
    for node in nodes {
        for i in 0..k {
            let di = node.rho[i] - mean[i];
            for j in 0..k {
                covariance[[i, j]] += node.weight * di * (node.rho[j] - mean[j]);
            }
        }
    }
    (mean, covariance)
}

/// Tier-1 of the exact marginal-smoothing inference stack (#938): adaptive
/// Gauss-Hermite quadrature over `ρ` (`K ≤ 4`), criterion-closure form.
///
/// The exact outer Hessian at `ρ̂` whitens/scales the grid; each node of the
/// product rule is reweighted by the exact profiled criterion,
/// `w_m ∝ exp(−criterion(ρ_m) + criterion(ρ̂)) × GH-weight × exp(½‖z_m‖²)`,
/// then normalized. The result is `π(ρ|y)` as a discrete mixture of conditional
/// Gaussians with its moment summary.
///
/// * `criterion` — the `OuterObjective::eval_cost` contract (`None` for
///   infeasible `ρ`, which gets zero weight). Each call is one warm inner
///   profile solve.
/// * `nodes_per_axis` — 3 or 5; pass `None` to auto-select (5 for `K ≤ 2`,
///   3 for `K ≤ 4` — at most 125 criterion evaluations either way).
pub fn rho_posterior_quadrature<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    mut criterion: F,
    nodes_per_axis: Option<usize>,
) -> Result<RhoPosteriorMixture, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Option<f64>,
{
    let k = rho_hat.len();
    let nodes_per_axis = nodes_per_axis.unwrap_or(if k <= 2 { 5 } else { 3 });
    let cost_hat = criterion(rho_hat).ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: criterion is infeasible at rho_hat itself".to_string(),
        )
    })?;
    let (core_nodes, effective_sample_size) =
        quadrature_nodes_core(rho_hat, outer_hessian, nodes_per_axis, cost_hat, |rho| {
            Ok(criterion(rho).map(|cost| (cost, None)))
        })?;
    let nodes: Vec<RhoMixtureNode> = core_nodes
        .into_iter()
        .map(|node| RhoMixtureNode {
            rho: node.rho,
            weight: node.weight,
            log_weight: node.log_weight,
            cost: node.cost,
        })
        .collect();
    let (mean, covariance) = mixture_moments(&nodes, k);
    Ok(RhoPosteriorMixture {
        nodes,
        mean,
        covariance,
        effective_sample_size,
    })
}

/// Assemble the mixture-corrected coefficient covariance from a Tier-1 mixture:
///
/// `Vβ_marginal = Σ_m w_m [Vb(ρ_m) + (β̂(ρ_m)−β̄)(β̂(ρ_m)−β̄)ᵀ]`,
/// `β̄ = Σ_m w_m β̂(ρ_m)`
///
/// — the law of total expectation/variance over the discrete `π(ρ|y)` mixture,
/// where each node's conditional is the Gaussian `N(β̂(ρ_m), Vb(ρ_m))` the
/// engine already produces at fixed `ρ`.
///
/// * `conditional` — supplies `(β̂(ρ_m), Vb(ρ_m))` for a node's `ρ_m`. The
///   caller holding the fit implements this as ONE WARM INNER PROFILE SOLVE per
///   node (the same inner solve the node's criterion evaluation already ran,
///   re-run warm-started, plus the conditional covariance assembly); evaluate
///   nodes in the order given (nearest-to-`ρ̂` first in the GH enumeration) to
///   keep warm starts effective. Nodes with zero weight are skipped and never
///   passed to the closure.
///
/// When all mixture weight concentrates at `ρ̂`, the spread term vanishes and
/// the result reduces to `Vb(ρ̂)` exactly.
pub fn mixture_coefficient_covariance<G>(
    mixture: &RhoPosteriorMixture,
    mut conditional: G,
) -> Result<MixtureCoefficientCovariance, EstimationError>
where
    G: FnMut(&Array1<f64>) -> Result<(Array1<f64>, Array2<f64>), EstimationError>,
{
    let mut conditionals: Vec<(f64, Array1<f64>, Array2<f64>)> = Vec::new();
    let mut total_weight = 0.0;
    for node in &mixture.nodes {
        if node.weight <= 0.0 {
            continue;
        }
        let (beta, vb) = conditional(&node.rho)?;
        if vb.nrows() != beta.len() || vb.ncols() != beta.len() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "mixture_coefficient_covariance: Vb shape {:?} does not match beta length {}",
                vb.dim(),
                beta.len()
            )));
        }
        if let Some((_, first_beta, _)) = conditionals.first()
            && first_beta.len() != beta.len()
        {
            return Err(EstimationError::RemlOptimizationFailed(
                "mixture_coefficient_covariance: inconsistent beta length across nodes".to_string(),
            ));
        }
        total_weight += node.weight;
        conditionals.push((node.weight, beta, vb));
    }
    if conditionals.is_empty() || !(total_weight.is_finite() && total_weight > 0.0) {
        return Err(EstimationError::RemlOptimizationFailed(
            "mixture_coefficient_covariance: no positive-weight nodes".to_string(),
        ));
    }
    let p = conditionals[0].1.len();
    let mut beta_bar = Array1::<f64>::zeros(p);
    for (weight, beta, _) in &conditionals {
        for i in 0..p {
            beta_bar[i] += (weight / total_weight) * beta[i];
        }
    }
    let mut covariance = Array2::<f64>::zeros((p, p));
    for (weight, beta, vb) in &conditionals {
        let w = weight / total_weight;
        for i in 0..p {
            let di = beta[i] - beta_bar[i];
            for j in 0..p {
                covariance[[i, j]] += w * (vb[[i, j]] + di * (beta[j] - beta_bar[j]));
            }
        }
    }
    Ok(MixtureCoefficientCovariance {
        beta_bar,
        covariance,
    })
}

/// Tier-2 of the exact marginal-smoothing inference stack (#938): NUTS over `ρ`
/// with the exact profiled gradient, whitened by the exact outer Hessian at
/// `ρ̂` (the `hmc` module's whitening design reused one level up).
///
/// * `criterion_and_grad` — `ρ ↦ (criterion(ρ), ∇_ρ criterion(ρ))`, both EXACT
///   (the engine's LAML value and ρ-gradient); `None` for infeasible `ρ`. Each
///   call is one warm inner profile solve + IFT gradient.
/// * `n_samples` — post-warmup draws per chain (2 chains; warmup is
///   `max(n_samples/2, 32)`).
/// * `seed` — deterministic seeding: the seed feeds the same splitmix64 chain /
///   transition streams as every other NUTS entry point. No clock, no global
///   RNG: the same `(fit, seed)` yields the same draws every run.
pub fn rho_posterior_nuts<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    criterion_and_grad: F,
    n_samples: usize,
    seed: u64,
) -> Result<RhoPosteriorSamples, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Option<(f64, Array1<f64>)> + Send,
{
    let k = rho_hat.len();
    let config = crate::inference::hmc_io::NutsConfig {
        n_samples: n_samples.max(4),
        nwarmup: (n_samples / 2).max(RHO_POSTERIOR_NUTS_WARMUP_FLOOR),
        n_chains: 2,
        target_accept: 0.9,
        seed,
    };
    let result = crate::inference::hmc_io::run_rho_criterion_nuts(
        rho_hat.view(),
        outer_hessian.view(),
        criterion_and_grad,
        &config,
    )
    .map_err(EstimationError::RemlOptimizationFailed)?;

    let n_draws = result.samples.nrows();
    if n_draws == 0 {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_nuts: sampler returned no draws".to_string(),
        ));
    }
    let mean = result.posterior_mean.clone();
    let mut covariance = Array2::<f64>::zeros((k, k));
    for row in result.samples.rows() {
        for i in 0..k {
            let di = row[i] - mean[i];
            for j in 0..k {
                covariance[[i, j]] += di * (row[j] - mean[j]);
            }
        }
    }
    covariance.mapv_inplace(|v| v / n_draws as f64);

    Ok(RhoPosteriorSamples {
        samples: result.samples,
        mean,
        covariance,
        rhat: result.rhat,
        ess: result.ess,
        converged: result.converged,
    })
}

/// The auto-selection seam (#938): given an [`RhoCertificate::Escalate`]
/// verdict from the Tier-0 certificate, pick and run the escalation tier by
/// dimension — Tier 1 (deterministic quadrature) for `K ≤ 4`, Tier 2 (NUTS
/// over `ρ` with the exact profiled gradient) for `K ≤ 16`, and an honest
/// [`RhoPosteriorEscalation::Unavailable`] beyond that. Magic by default: no
/// flags, the tier is chosen from the problem.
///
/// Both closures evaluate the SAME live objective the fit converged on
/// (`criterion` = `OuterObjective::eval_cost`, `criterion_and_grad` = value +
/// exact LAML ρ-gradient); run this while that objective is still alive.
pub fn escalate_rho_posterior<F, G>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    criterion: F,
    criterion_and_grad: G,
) -> RhoPosteriorEscalation
where
    F: FnMut(&Array1<f64>) -> Option<f64>,
    G: FnMut(&Array1<f64>) -> Option<(f64, Array1<f64>)> + Send,
{
    let k = rho_hat.len();
    if k == 0 {
        return RhoPosteriorEscalation::Unavailable {
            n_params: 0,
            reason: "no smoothing parameters to marginalize".to_string(),
        };
    }
    if k <= TIER1_MAX_DIM {
        match rho_posterior_quadrature(rho_hat, outer_hessian, criterion, None) {
            Ok(mixture) => RhoPosteriorEscalation::Quadrature(mixture),
            Err(e) => RhoPosteriorEscalation::Unavailable {
                n_params: k,
                reason: format!("tier-1 quadrature failed: {e}"),
            },
        }
    } else if k <= TIER2_MAX_DIM {
        match rho_posterior_nuts(
            rho_hat,
            outer_hessian,
            criterion_and_grad,
            ESCALATION_NUTS_SAMPLES,
            ESCALATION_NUTS_SEED,
        ) {
            Ok(samples) => RhoPosteriorEscalation::Nuts(samples),
            Err(e) => RhoPosteriorEscalation::Unavailable {
                n_params: k,
                reason: format!("tier-2 NUTS failed: {e}"),
            },
        }
    } else {
        RhoPosteriorEscalation::Unavailable {
            n_params: k,
            reason: format!(
                "rho-posterior escalation is unavailable for K={k} > {TIER2_MAX_DIM} smoothing \
                 parameters; intervals remain plug-in with the first-order V_rho correction"
            ),
        }
    }
}

/// Compute the Tier-0 PSIS `ρ`-certificate.
///
/// * `rho_hat` — the converged smoothing parameters `ρ̂` (length `K`).
/// * `outer_hessian` — the exact outer Hessian `H_ρ` of the criterion at `ρ̂`
///   (`K × K`, SPD). The proposal covariance is `H_ρ⁻¹`.
/// * `criterion` — evaluates the outer criterion `−log π(ρ|y)` (the LAML/REML
///   objective) at a trial `ρ`; returns `None` for infeasible `ρ`. This is the
///   `OuterObjective::eval_cost` contract, supplied by the caller that retains
///   (or rebuilds) the objective.
/// * `n_samples` — proposal draw count `M` (defaults to 64 when `None`).
///
/// Returns `None` when `K = 0`, the outer Hessian is not usable, the criterion
/// at `ρ̂` is infeasible, or too few finite weights survive for a tail fit.
pub fn rho_posterior_certificate<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    criterion: F,
    n_samples: Option<usize>,
) -> Option<RhoPosteriorCertificate>
where
    F: Fn(&Array1<f64>) -> Option<f64>,
{
    let k = rho_hat.len();
    if k == 0 || outer_hessian.nrows() != k || outer_hessian.ncols() != k {
        return None;
    }
    let cost_hat = criterion(rho_hat)?;
    if !cost_hat.is_finite() {
        return None;
    }
    let l_inv = whitening_factor_from_outer_hessian(outer_hessian)?;
    let m = n_samples
        .unwrap_or(DEFAULT_M)
        .max(2 * crate::psis::MIN_TAIL_COUNT);

    let mut rng = DetNormal::new(CERTIFICATE_SEED);
    let mut raw_weights: Vec<f64> = Vec::with_capacity(m);
    for _ in 0..m {
        let z: Array1<f64> = Array1::from_iter((0..k).map(|_| rng.normal()));
        // ρ_m = ρ̂ + L_inv z.
        let mut rho_m = rho_hat.clone();
        for i in 0..k {
            let mut acc = 0.0;
            for j in 0..k {
                acc += l_inv[[i, j]] * z[j];
            }
            rho_m[i] += acc;
        }
        let half_norm_sq = 0.5 * z.iter().map(|&v| v * v).sum::<f64>();
        // log w_m = −criterion(ρ_m) + criterion(ρ̂) + ½‖z_m‖².
        let log_w = match criterion(&rho_m) {
            Some(c) if c.is_finite() => -c + cost_hat + half_norm_sq,
            // Infeasible / non-finite criterion ⇒ zero importance weight.
            _ => f64::NEG_INFINITY,
        };
        raw_weights.push(log_w);
    }

    // Stabilize and exponentiate: subtract the max log-weight (cancels in the
    // self-normalized weights and the Pareto fit).
    let max_lw = raw_weights
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return None;
    }
    let weights: Vec<f64> = raw_weights
        .iter()
        .map(|&lw| {
            if lw.is_finite() {
                (lw - max_lw).exp()
            } else {
                0.0
            }
        })
        .collect();

    let psis = pareto_smooth_weights(&weights)?;
    let k_hat = psis.k_hat;

    // Self-normalize the smoothed weights.
    let total: f64 = psis.smoothed.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let normalized: Array1<f64> = Array1::from_iter(psis.smoothed.iter().map(|&w| w / total));
    let sum_sq: f64 = normalized.iter().map(|&w| w * w).sum();
    let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

    Some(RhoPosteriorCertificate {
        k_hat,
        certificate: RhoCertificate::from_k_hat(k_hat),
        n_samples: m,
        weights: normalized,
        effective_sample_size: ess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// CLOSED-FORM FIXTURE: when the criterion IS exactly the Gaussian
    /// `−log π(ρ|y) = ½(ρ−ρ̂)ᵀ H_ρ (ρ−ρ̂)` that the Laplace proposal assumes,
    /// the importance weights are all identically 1 — the proposal is the
    /// target. PSIS must then report a tiny `k̂` and certify the plug-in.
    #[test]
    fn exact_gaussian_target_certifies_plug_in() {
        let rho_hat = array![0.3, -0.7];
        let h = array![[2.0, 0.5], [0.5, 1.5]];
        // criterion(ρ) = ½ (ρ−ρ̂)ᵀ H (ρ−ρ̂): exactly the proposal's negative log
        // density up to the constant that cancels in the self-normalized weight.
        let crit = |rho: &Array1<f64>| {
            let d = rho - &rho_hat;
            let mut q = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    q += d[i] * h[[i, j]] * d[j];
                }
            }
            Some(0.5 * q)
        };
        let cert = rho_posterior_certificate(&rho_hat, &h, crit, Some(256)).expect("certificate");
        // All weights equal ⇒ ESS == M and k̂ small ⇒ plug-in certified.
        assert!(
            (cert.effective_sample_size - cert.n_samples as f64).abs() < 1e-6,
            "uniform weights must give ESS == M: ess={} M={}",
            cert.effective_sample_size,
            cert.n_samples
        );
        assert!(
            cert.k_hat < 0.5,
            "exact-Gaussian target must yield small k̂, got {}",
            cert.k_hat
        );
        assert_eq!(cert.certificate, RhoCertificate::PlugInCertified);
    }

    /// When the true `π(ρ|y)` is much HEAVIER-tailed than the Gaussian Laplace
    /// proposal (a criterion far flatter than the proposal quadratic in the
    /// tails), the importance weights blow up and PSIS must refuse to certify
    /// the plug-in — `k̂` rises and the tier escalates.
    #[test]
    fn heavy_tailed_target_refuses_to_certify() {
        let rho_hat = array![0.0];
        let h = array![[4.0]]; // tight proposal (variance 0.25).
        // Target ∝ a heavy Student-like tail: criterion grows only
        // logarithmically, so far in the tail π(ρ)/proposal(ρ) → ∞.
        let crit = |rho: &Array1<f64>| {
            let r = rho[0];
            Some((1.0 + r * r).ln())
        };
        let cert = rho_posterior_certificate(&rho_hat, &h, crit, Some(512)).expect("certificate");
        assert!(
            cert.k_hat > 0.5,
            "heavy-tailed target must raise k̂ above 0.5, got {}",
            cert.k_hat
        );
        assert_ne!(cert.certificate, RhoCertificate::PlugInCertified);
    }

    #[test]
    fn weights_are_normalized_and_deterministic() {
        let rho_hat = array![1.0];
        let h = array![[1.0]];
        let crit = |rho: &Array1<f64>| {
            let d = rho[0] - 1.0;
            Some(0.5 * d * d)
        };
        let a = rho_posterior_certificate(&rho_hat, &h, crit, Some(64)).expect("a");
        let b = rho_posterior_certificate(&rho_hat, &h, crit, Some(64)).expect("b");
        // Self-normalized weights sum to 1.
        let s: f64 = a.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "weights must sum to 1, got {s}");
        // Deterministic: identical k̂ across runs (fixed-seed stream).
        assert_eq!(a.k_hat.to_bits(), b.k_hat.to_bits());
    }

    #[test]
    fn empty_rho_returns_none() {
        let rho_hat: Array1<f64> = array![];
        let h = Array2::<f64>::zeros((0, 0));
        assert!(rho_posterior_certificate(&rho_hat, &h, |_| Some(0.0), None).is_none());
    }
}
