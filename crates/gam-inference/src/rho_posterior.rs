//! Marginal smoothing inference over the smoothing parameters `ρ`
//! (issue #938): the Tier-0 **PSIS adequacy diagnostic**, plus the auto-selected
//! escalation tiers — Tier-1 **Gauss-Hermite quadrature** over `ρ`
//! (`rho_posterior_quadrature`) and Tier-2 **NUTS over `ρ`** with the exact
//! profiled gradient (`rho_posterior_nuts`), routed by
//! [`escalate_rho_posterior`] to whichever needs fewer criterion evaluations
//! when the diagnostic grades the plug-in [`RhoProposalAdequacy::Escalate`].
//!
//! Every GAM ecosystem conditions inference on the estimated smoothing
//! parameters `ρ̂`; intervals from `V(β̂|ρ̂)` undercover because they ignore
//! `ρ`-uncertainty. The honest marginal posterior factorizes as
//! `π(β, ρ | y) = π(β | ρ, y) · π(ρ | y)`, where `π(ρ|y) ∝ exp(−criterion(ρ))`
//! is the LAML/REML objective the outer optimizer already minimizes with exact
//! gradients. For Gaussian REML that criterion is the restricted likelihood
//! itself; for other families it is the Laplace (LAML) or PQL approximation to
//! it, so the tiers below integrate that approximate `π(ρ|y)`, not the exact one
//! (#2946 T2).
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
//! 4. Pareto-smooth the weights ([`gam_solve::psis`]) and read the
//!    Zhang–Stephens tail shape `k̂`. `k̂ < 0.5` ⇒ the plug-in + first-order
//!    correction answer is adequate by this diagnostic; `0.5 ≤ k̂ ≤ 0.7` ⇒ usable as
//!    a self-normalized importance correction; `k̂ > 0.7` ⇒ the Laplace proposal
//!    is a poor fit and the honest path is a full quadrature/NUTS escalation.
//!
//! The diagnostic is deterministic: the whitened draws come from a fixed-seed
//! splitmix64 + Box–Muller stream, so the same fit yields the same `k̂` every
//! run.
//!
//! **`k̂` has a resolution, and it is coarse.** The GPD is fitted to
//! [`gam_solve::psis::tail_count`]`(M) = ⌈√M⌉` excesses only, and the reported
//! shape is that fit shrunk toward `0.5` by ten pseudo-observations. So the
//! reported value has standard error
//! `√n(1+k)/(n+10)` (with `n = ⌈√M⌉`; [`gam_solve::psis::shape_standard_error`],
//! and per fit [`k_hat_standard_error`]) around the shrunk shape
//! `(n·k + 10·0.5)/(n + 10)`, NOT around `k`:
//! at `M = 100` the tail sample is `10` and the standard error at the `0.7`
//! boundary is `≈ 0.27`; at the diagnostic's `M = 2155` it is `47` and `≈ 0.20`.
//! Reaching a standard error of `0.05` takes a tail of `≈ 10³`, i.e. `M ≈ 10⁶`. A single
//! `k̂` near a cutoff is therefore not evidence about which side of the cutoff
//! the truth lies on: separating a true shape from the `0.7` boundary needs
//! `⌈√M⌉` large enough that several standard errors fit in the gap. Anything
//! asserting a verdict (rather than reading a diagnostic) must size `M` from
//! `tail_count` and `shape_standard_error`.

use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_solve::estimate::EstimationError;
use gam_solve::psis::{pareto_smooth_weights, reliable_sample_size};
use ndarray::{Array1, Array2};

// The `ρ`-posterior adequacy/escalation DATA types were contract-downed to
// the neutral `gam-problem` crate (#1521) so gam-solve can store/return them
// without a back-edge into gam-inference. The COMPUTATION below (PSIS
// adequacy diagnostic, Tier-1 quadrature, Tier-2 NUTS via `hmc_io`) stays here and
// constructs these types under the names re-exported here.
pub use gam_problem::rho_posterior::{
    ESCALATE_K_HAT, PLUG_IN_ADEQUATE_K_HAT, RhoMixtureNode, RhoPosteriorAdequacy,
    RhoPosteriorEscalation, RhoPosteriorMixture, RhoPosteriorNotComputed, RhoPosteriorOutcome,
    RhoPosteriorRefusal, RhoPosteriorSamples, RhoProposalAdequacy, WeightTailShape,
};

/// Monolith (gam-inference-tier) implementor of the contract-downed
/// [`RhoPosteriorEscalator`](gam_problem::rho_posterior::RhoPosteriorEscalator)
/// (#1521): wraps the real `hmc_io`-backed Tier-0 PSIS adequacy diagnostic
/// ([`rho_posterior_adequacy`]) and the auto-selected Tier-1/Tier-2
/// escalation ([`escalate_rho_posterior`], whose Tier-2 NUTS pulls the
/// gam-inference sampler). Injected at process init via
/// `gam_problem::rho_posterior::set_rho_posterior_escalator`; gam-solve's REML
/// evaluator calls through `gam_problem::rho_posterior::rho_posterior_escalator`.
pub struct HmcIoRhoPosteriorEscalator;

impl gam_problem::rho_posterior::RhoPosteriorEscalator for HmcIoRhoPosteriorEscalator {
    fn rho_posterior_adequacy(
        &self,
        rho_hat: &Array1<f64>,
        outer_hessian: &Array2<f64>,
        criterion: &dyn Fn(&Array1<f64>) -> Result<f64, String>,
    ) -> Result<Option<RhoPosteriorAdequacy>, RhoPosteriorRefusal> {
        rho_posterior_adequacy(rho_hat, outer_hessian, criterion)
    }

    fn escalate_rho_posterior(
        &self,
        mode: &Array1<f64>,
        hessian: &Array2<f64>,
        criterion: &mut dyn FnMut(&Array1<f64>) -> Result<f64, String>,
        criterion_and_grad: &mut (dyn FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), String>
                  + Send),
    ) -> RhoPosteriorEscalation {
        escalate_rho_posterior(mode, hessian, criterion, criterion_and_grad)
    }
}

/// Deterministic seed for the auto-selected Tier-2 escalation (no clock).
const ESCALATION_NUTS_SEED: u64 = 0x938_5EED_0938_5EED;

const ADEQUACY_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

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
        let z = gam_linalg::utils::splitmix64(&mut self.state);
        (((z >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        // `uniform` returns `(k + ½)/2⁵³`, strictly inside (0, 1): `ln(u1)` is
        // finite by construction.
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Solve `H_ρ⁻¹`'s Cholesky `L_inv` (with `L_inv L_invᵀ = H_ρ⁻¹`) from the outer
/// Hessian `H_ρ`. We factor `H_ρ = R Rᵀ` (lower `R`) and use `L_inv = R⁻ᵀ`: then
/// `L_inv L_invᵀ = R⁻ᵀ R⁻¹ = (R Rᵀ)⁻¹ = H_ρ⁻¹`. Mapping `ρ_m = ρ̂ + L_inv z_m`
/// gives draws with covariance `H_ρ⁻¹`, and `‖z_m‖² = (ρ_m−ρ̂)ᵀ H_ρ (ρ_m−ρ̂)`.
///
/// `R` is the strict Cholesky factor of `H_ρ` itself. An outer Hessian that is
/// not positive definite has no Gaussian proposal, so it is refused with the
/// factorization's reason rather than ridged into one whose covariance is not
/// `H_ρ⁻¹`.
fn whitening_factor_from_outer_hessian(
    outer_hessian: &Array2<f64>,
) -> Result<Array2<f64>, RhoPosteriorRefusal> {
    let r = outer_hessian
        .cholesky(Side::Lower)
        .map_err(|error| RhoPosteriorRefusal::HessianNotPositiveDefinite {
            detail: format!("{error:?}"),
        })?
        .lower_triangular();
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
                return Err(RhoPosteriorRefusal::HessianNotPositiveDefinite {
                    detail: format!("Cholesky pivot {i} is {rii}"),
                });
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
    Ok(l_inv)
}

/// Enumerate the product rule over `rules`, one rule per axis, appending every
/// node with the log of its product weight.
pub(crate) fn enumerate_gh_product(
    rules: &[Vec<(f64, f64)>],
    axis: usize,
    z: &mut Array1<f64>,
    log_w: f64,
    out: &mut Vec<(Array1<f64>, f64)>,
) {
    if axis == rules.len() {
        out.push((z.clone(), log_w));
        return;
    }
    for &(node, weight) in &rules[axis] {
        z[axis] = node;
        enumerate_gh_product(rules, axis + 1, z, log_w + weight.ln(), out);
    }
}

/// One normalized node from the quadrature core: `(ρ, cost, normalized weight,
/// normalized log-weight)`.
struct NormalizedQuadratureNode {
    rho: Array1<f64>,
    cost: f64,
    weight: f64,
    log_weight: f64,
}

/// Tier-1 quadrature core (#938): whiten by the exact outer Hessian, enumerate
/// the Gauss-Hermite product grid, reweight each node by the exact profiled
/// criterion `exp(−V(ρ_m) + V(ρ̂) + ½‖z_m‖²) × GH-weight`, and normalize.
/// `rho_posterior_quadrature` is its criterion-closure adapter. Every node
/// carries rule mass, so a node the criterion cannot value, or values as
/// non-finite, fails the rule rather than being dropped from it.
fn quadrature_nodes_core<E>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    nodes_per_axis: usize,
    cost_hat: f64,
    mut eval_node: E,
) -> Result<(Vec<NormalizedQuadratureNode>, f64), EstimationError>
where
    E: FnMut(&Array1<f64>) -> Result<f64, String>,
{
    let k = rho_hat.len();
    if k == 0 || outer_hessian.nrows() != k || outer_hessian.ncols() != k {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: rho/Hessian shape mismatch".to_string(),
        ));
    }
    let rule = gam_math::quadrature::standard_normal_gauss_hermite_rule(nodes_per_axis).map_err(
        |error| {
            EstimationError::RemlOptimizationFailed(format!(
                "rho_posterior_quadrature: standard-normal Gauss–Hermite rule of order \
                 {nodes_per_axis}: {error}"
            ))
        },
    )?;
    let l_inv = whitening_factor_from_outer_hessian(outer_hessian).map_err(|reason| {
        EstimationError::RemlOptimizationFailed(format!("rho_posterior_quadrature: {reason}"))
    })?;
    if !cost_hat.is_finite() {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: non-finite criterion at rho_hat".to_string(),
        ));
    }

    let mut product_nodes = Vec::new();
    enumerate_gh_product(
        &vec![rule; k],
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
        let cost = eval_node(&rho).map_err(|detail| {
            EstimationError::RemlOptimizationFailed(format!(
                "rho_posterior_quadrature: criterion unavailable at node {rho:?}: {detail}"
            ))
        })?;
        if !cost.is_finite() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "rho_posterior_quadrature: criterion at node {rho:?} is {cost}"
            )));
        }
        let half_norm_sq = 0.5 * z.iter().map(|&v| v * v).sum::<f64>();
        let log_weight = log_base_weight - cost + cost_hat + half_norm_sq;
        max_log_weight = max_log_weight.max(log_weight);
        raw_nodes.push((rho, cost, log_weight));
    }
    let mut total = 0.0;
    let mut scaled = Vec::with_capacity(raw_nodes.len());
    for (_, _, log_weight) in &raw_nodes {
        let w = (*log_weight - max_log_weight).exp();
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
    for ((rho, cost, log_weight), scaled_weight) in raw_nodes.into_iter().zip(scaled) {
        let weight = scaled_weight / total;
        sum_sq += weight * weight;
        nodes.push(NormalizedQuadratureNode {
            rho,
            cost,
            weight,
            log_weight: log_weight - max_log_weight - total.ln(),
        });
    }
    let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
    Ok((nodes, ess))
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

/// Gauss-Hermite nodes per axis the auto-selected Tier-1 rule uses: five while
/// `K ≤ 2`, three beyond.
fn auto_nodes_per_axis(k: usize) -> usize {
    if k <= 2 { 5 } else { 3 }
}

/// Tier-1 of the marginal-smoothing inference stack (#938): adaptive
/// Gauss-Hermite quadrature over `ρ`, criterion-closure form.
///
/// The exact outer Hessian at `ρ̂` whitens/scales the grid; each node of the
/// product rule is reweighted by the exact profiled criterion,
/// `w_m ∝ exp(−criterion(ρ_m) + criterion(ρ̂)) × GH-weight × exp(½‖z_m‖²)`,
/// then normalized. The result is `π(ρ|y)` as a discrete mixture of conditional
/// Gaussians with its moment summary.
///
/// * `criterion` — the `OuterObjective::eval_cost` contract, or the reason it
///   cannot value a node, which fails the rule. Each call is one warm inner
///   profile solve.
/// * `nodes_per_axis` — 3 or 5; pass `None` to auto-select (5 for `K ≤ 2`,
///   3 beyond), so the grid costs `5^K` or `3^K` criterion evaluations.
pub(crate) fn rho_posterior_quadrature<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    mut criterion: F,
    nodes_per_axis: Option<usize>,
) -> Result<RhoPosteriorMixture, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<f64, String>,
{
    let k = rho_hat.len();
    let nodes_per_axis = nodes_per_axis.unwrap_or_else(|| auto_nodes_per_axis(k));
    let cost_hat = criterion(rho_hat).map_err(|detail| {
        EstimationError::RemlOptimizationFailed(format!(
            "rho_posterior_quadrature: criterion is unavailable at rho_hat itself: {detail}"
        ))
    })?;
    let (core_nodes, effective_sample_size) =
        quadrature_nodes_core(rho_hat, outer_hessian, nodes_per_axis, cost_hat, criterion)?;
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

/// Tier-2 of the marginal-smoothing inference stack (#938): NUTS over `ρ`
/// with the exact profiled gradient, centred at `rho_hat` and whitened by
/// `outer_hessian` (the `hmc` module's whitening design reused one level up).
/// The escalation passes the sampled density's own mode and Hessian (#3293).
///
/// * `criterion_and_grad` — `ρ ↦ (criterion(ρ), ∇_ρ criterion(ρ))`, both EXACT
///   (the engine's LAML value and ρ-gradient), or the reason it cannot value a
///   position, which fails the run. Each call is one warm inner profile solve +
///   IFT gradient.
/// * `seed` — deterministic seeding: the seed feeds the same splitmix64 chain /
///   transition streams as every other NUTS entry point. No clock, no global
///   RNG: the same `(fit, seed)` yields the same draws every run.
pub(crate) fn rho_posterior_nuts<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    criterion_and_grad: F,
    seed: u64,
) -> Result<RhoPosteriorSamples, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), String> + Send,
{
    let k = rho_hat.len();
    let result = crate::hmc_io::run_rho_criterion_nuts(
        rho_hat.view(),
        outer_hessian.view(),
        criterion_and_grad,
        seed,
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

/// The auto-selection seam (#938): given an [`RhoProposalAdequacy::Escalate`]
/// grade from the Tier-0 adequacy diagnostic, pick and run the escalation tier
/// that needs fewer criterion evaluations (#3187). Tier 1 (deterministic
/// quadrature) costs its product grid, `5^K` or `3^K` nodes; Tier 2 (NUTS over
/// `ρ` with the exact profiled gradient) costs more than
/// [`RHO_NUTS_MIN_EVALUATIONS`](crate::hmc_io::RHO_NUTS_MIN_EVALUATIONS)
/// value+gradient evaluations whenever it converges. Quadrature runs while its
/// grid is no larger than that floor (`K ≤ 4`), NUTS beyond, at any `K`. A tier
/// that fails reports an honest [`RhoPosteriorEscalation::Unavailable`]. Magic
/// by default: no flags, the tier is chosen from the problem.
///
/// Both closures evaluate the SAME sampled density (`criterion` its negative
/// log, `criterion_and_grad` that value plus its exact ρ-gradient); run this
/// while the objective behind them is still alive. `mode` and `hessian` are
/// that density's Laplace geometry, which both tiers centre and whiten by
/// (#3293): a geometry taken from a different density leaves the quadrature
/// proposal and the NUTS metric mis-scaled exactly where the two differ.
pub fn escalate_rho_posterior<F, G>(
    mode: &Array1<f64>,
    hessian: &Array2<f64>,
    criterion: F,
    criterion_and_grad: G,
) -> RhoPosteriorEscalation
where
    F: FnMut(&Array1<f64>) -> Result<f64, String>,
    G: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), String> + Send,
{
    let k = mode.len();
    if k == 0 {
        return RhoPosteriorEscalation::Unavailable {
            n_params: 0,
            reason: "no smoothing parameters to marginalize".to_string(),
        };
    }
    let grid_nodes = u32::try_from(k)
        .ok()
        .and_then(|exponent| auto_nodes_per_axis(k).checked_pow(exponent));
    if grid_nodes.is_some_and(|nodes| nodes <= crate::hmc_io::RHO_NUTS_MIN_EVALUATIONS) {
        match rho_posterior_quadrature(mode, hessian, criterion, None) {
            Ok(mixture) => RhoPosteriorEscalation::Quadrature(mixture),
            Err(e) => RhoPosteriorEscalation::Unavailable {
                n_params: k,
                reason: format!("tier-1 quadrature failed: {e}"),
            },
        }
    } else {
        match rho_posterior_nuts(mode, hessian, criterion_and_grad, ESCALATION_NUTS_SEED) {
            Ok(samples) => RhoPosteriorEscalation::Nuts(samples),
            Err(e) => RhoPosteriorEscalation::Unavailable {
                n_params: k,
                reason: format!("tier-2 NUTS failed: {e}"),
            },
        }
    }
}

/// Compute the Tier-0 PSIS `ρ`-adequacy diagnostic.
///
/// * `rho_hat` — the converged smoothing parameters `ρ̂` (length `K`).
/// * `outer_hessian` — the exact outer Hessian `H_ρ` of the criterion at `ρ̂`
///   (`K × K`, SPD). The proposal covariance is `H_ρ⁻¹`.
/// * `criterion` — evaluates the outer criterion `−log π(ρ|y)` (the LAML/REML
///   objective) at a trial `ρ`, or says why it cannot. This is the
///   `OuterObjective::eval_cost` contract, supplied by the caller that retains
///   (or rebuilds) the objective. Every draw carries proposal mass, so the
///   diagnostic is refused at the first draw it cannot value, never formed
///   from the rest.
///
/// The proposal draw count `M` is not an option: it is the `2155` draws at which
/// PSIS is reliable for a shape at [`ESCALATE_K_HAT`],
/// [`gam_solve::psis::reliable_sample_size`]`(ESCALATE_K_HAT)`, so every shape
/// the grade calls usable is reliably estimated (#3187).
///
/// Returns `Ok(None)` when `K = 0`: there is nothing to grade. Returns the typed
/// [`RhoPosteriorRefusal`] naming the site when the diagnostic cannot be formed —
/// an outer Hessian whose shape does not match `ρ̂` or that is not positive
/// definite, an unavailable or non-finite criterion at `ρ̂` or at a draw, a failed
/// Pareto tail fit, a non-finite tail shape, or smoothed weights that do not
/// normalize.
pub fn rho_posterior_adequacy<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    criterion: F,
) -> Result<Option<RhoPosteriorAdequacy>, RhoPosteriorRefusal>
where
    F: Fn(&Array1<f64>) -> Result<f64, String>,
{
    let k = rho_hat.len();
    if k == 0 {
        return Ok(None);
    }
    if outer_hessian.nrows() != k || outer_hessian.ncols() != k {
        return Err(RhoPosteriorRefusal::HessianShape {
            rows: outer_hessian.nrows(),
            cols: outer_hessian.ncols(),
            k,
        });
    }
    let cost_hat = criterion(rho_hat)
        .map_err(|detail| RhoPosteriorRefusal::CriterionUnavailableAtRhoHat { detail })?;
    if !cost_hat.is_finite() {
        return Err(RhoPosteriorRefusal::CriterionNotFiniteAtRhoHat);
    }
    let l_inv = whitening_factor_from_outer_hessian(outer_hessian)?;

    // The draw count is the smallest at which PSIS is reliable for every shape
    // the grade calls usable: `reliable_sample_size` is `S(k) = 10^{1/(1-k)}`,
    // and `S(ESCALATE_K_HAT) = 2155` is where the sample-size threshold
    // `1 - 1/log10(M)` reaches the escalation cutoff, so the grade's cutoffs
    // are the operative ones at this `M`. Growing `M` from a smaller start
    // while `k̂` asks for more stops on the draws' noise instead: from `100`
    // draws (a tail of `10`) an infinite-variance Cauchy-over-Gaussian target
    // reads `k̂ < 0.5` in 43.5% of 200 seeds, against 13.5% at `2155` (#3187).
    let m =
        reliable_sample_size(ESCALATE_K_HAT).expect("the escalation cutoff is a shape below 1");
    let mut rng = DetNormal::new(ADEQUACY_SEED);
    let mut raw_weights: Vec<f64> = Vec::with_capacity(m);
    for draw in 0..m {
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
        // The proposal's −log density at the draw it actually made,
        // ½(ρ_m − ρ̂)ᵀ H_ρ (ρ_m − ρ̂). It equals ½‖z_m‖² in exact arithmetic,
        // but ρ_m is rounded, and the weight is target over proposal at ONE
        // point: valuing the proposal at the unrounded draw puts rounding noise
        // in every weight of an exact proposal, which then has a spurious
        // tail instead of the flat one it has (#3202).
        let mut quad = 0.0;
        for i in 0..k {
            let di = rho_m[i] - rho_hat[i];
            for j in 0..k {
                quad += di * outer_hessian[[i, j]] * (rho_m[j] - rho_hat[j]);
            }
        }
        let half_quad = 0.5 * quad;
        // log w_m = −criterion(ρ_m) + criterion(ρ̂) + ½(ρ_m − ρ̂)ᵀ H_ρ (ρ_m − ρ̂).
        let cost = criterion(&rho_m)
            .map_err(|detail| RhoPosteriorRefusal::CriterionUnavailableAtDraw { draw, detail })?;
        if !cost.is_finite() {
            return Err(RhoPosteriorRefusal::CriterionNotFiniteAtDraw { draw });
        }
        raw_weights.push(-cost + cost_hat + half_quad);
    }

    // Stabilize and exponentiate: subtract the max log-weight (cancels in the
    // self-normalized weights and the Pareto fit).
    let max_lw = raw_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = raw_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();

    let psis = pareto_smooth_weights(&weights).ok_or(RhoPosteriorRefusal::TailFitUnavailable)?;
    if let WeightTailShape::Pareto(k_hat) = psis.shape
        && !k_hat.is_finite()
    {
        return Err(RhoPosteriorRefusal::TailShapeNotFinite);
    }

    // Kish's ESS `(Σw)²/Σw²` of the smoothed weights. It is scale-free, so the
    // weights are scaled by their largest, not their total: each scaled weight
    // is in `(0, 1]`, so `Σs ≤ M` and `1 ≤ Σs² ≤ M` neither overflow nor
    // underflow, and an exact proposal's unit weights stay exactly 1, where
    // dividing by the total rounds `1/M` at every `M` that is not a power of
    // two. Cauchy–Schwarz gives `(Σs)² ≤ M Σs²`, so the ESS lies in `[1, M]`.
    let largest = psis.smoothed.iter().copied().fold(0.0, f64::max);
    if !(largest.is_finite() && largest > 0.0) {
        return Err(RhoPosteriorRefusal::SmoothedWeightsNotNormalizable);
    }
    let (total, sum_sq) = psis.smoothed.iter().fold((0.0, 0.0), |(sum, sum_sq), &w| {
        let scaled = w / largest;
        (sum + scaled, sum_sq + scaled * scaled)
    });

    Ok(Some(RhoPosteriorAdequacy {
        tail_shape: psis.shape,
        adequacy: RhoProposalAdequacy::from_tail_shape(psis.shape),
        n_samples: m,
        effective_sample_size: total * total / sum_sq,
    }))
}

/// Standard error of the fitted `k̂` in `adequacy.tail_shape`, the resolution
/// of its grade (#2946 T2).
///
/// It is [`gam_solve::psis::shape_standard_error`] at the tail sample
/// [`gam_solve::psis::tail_count`]`(n_samples)` that the Pareto fit used,
/// evaluated at the reported shape as a plug-in for the true one. A grade whose
/// `k̂` lies within a few of these of [`PLUG_IN_ADEQUATE_K_HAT`] or
/// [`ESCALATE_K_HAT`] does not say which side of that cutoff the truth is on.
/// A flat tail fitted no shape, so it has none (`None`): its grade is read off
/// the weights exactly.
pub fn k_hat_standard_error(adequacy: &RhoPosteriorAdequacy) -> Option<f64> {
    match adequacy.tail_shape {
        WeightTailShape::Pareto(k_hat) => Some(gam_solve::psis::shape_standard_error(
            gam_solve::psis::tail_count(adequacy.n_samples),
            k_hat,
        )),
        WeightTailShape::Flat => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// CLOSED-FORM FIXTURE: when the criterion IS exactly the Gaussian
    /// `−log π(ρ|y) = ½(ρ−ρ̂)ᵀ H_ρ (ρ−ρ̂)` that the Laplace proposal assumes,
    /// the importance weights are all identically 1 — the proposal is the
    /// target. The weights then have no tail at all, so PSIS reports the flat
    /// shape, ESS is `M` exactly, and the plug-in grades adequate (#3202).
    #[test]
    fn exact_gaussian_target_grades_plug_in_adequate() {
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
            Ok(0.5 * q)
        };
        let graded = rho_posterior_adequacy(&rho_hat, &h, crit)
            .expect("diagnostic formed")
            .expect("diagnostic present");
        // The criterion and the proposal value each draw by the same
        // arithmetic, so every log-weight is exactly 0 and every weight 1:
        // M = 2155 sums, normalizes and squares without rounding.
        assert_eq!(graded.n_samples, 2155);
        assert_eq!(graded.tail_shape, WeightTailShape::Flat);
        assert_eq!(graded.effective_sample_size, graded.n_samples as f64);
        assert_eq!(graded.adequacy, RhoProposalAdequacy::PlugInAdequate);
        assert_eq!(k_hat_standard_error(&graded), None);
    }

    /// #3202 repro: 1-D, `ρ̂ = 1`, `H = [[1]]`, `c(ρ) = ½(ρ − 1)²`. The
    /// proposal is exact. Valued at the unrounded draw `z` rather than at
    /// `ρ_m = ρ̂ + z`, the log-weights were rounding noise with only a few
    /// distinct values in the tail, too few positive excesses to fit, and the
    /// best possible proposal was refused as `TailFitUnavailable`.
    #[test]
    fn exact_one_dimensional_proposal_is_flat_not_refused_3202() {
        let rho_hat = array![1.0];
        let h = array![[1.0]];
        let crit = |rho: &Array1<f64>| {
            let d = rho[0] - 1.0;
            Ok(0.5 * d * d)
        };
        let graded = rho_posterior_adequacy(&rho_hat, &h, crit)
            .expect("an exact proposal is graded, not refused")
            .expect("diagnostic present");
        assert_eq!(graded.tail_shape, WeightTailShape::Flat);
        assert_eq!(graded.effective_sample_size, graded.n_samples as f64);
        assert_eq!(graded.adequacy, RhoProposalAdequacy::PlugInAdequate);
    }

    /// When the true `π(ρ|y)` is much HEAVIER-tailed than the Gaussian Laplace
    /// proposal (a criterion far flatter than the proposal quadratic in the
    /// tails), the importance weights blow up and PSIS must not grade the
    /// plug-in adequate — `k̂` rises and the tier escalates.
    #[test]
    fn heavy_tailed_target_is_not_graded_plug_in_adequate() {
        let rho_hat = array![0.0];
        let h = array![[4.0]]; // tight proposal (variance 0.25).
        // Target ∝ a heavy Student-like tail: criterion grows only
        // logarithmically, so far in the tail π(ρ)/proposal(ρ) → ∞.
        let crit = |rho: &Array1<f64>| {
            let r = rho[0];
            Ok((1.0 + r * r).ln())
        };
        let graded = rho_posterior_adequacy(&rho_hat, &h, crit)
            .expect("diagnostic formed")
            .expect("diagnostic present");
        let WeightTailShape::Pareto(k_hat) = graded.tail_shape else {
            panic!("a heavy-tailed target has a tail to fit, got {:?}", graded.tail_shape);
        };
        assert!(k_hat > 0.5, "heavy-tailed target must raise k̂ above 0.5, got {k_hat}");
        // #3187: the draw count is the one at which PSIS is reliable for a shape
        // at the escalation cutoff.
        assert_eq!(graded.n_samples, 2155);
        // This used to be `assert_ne!(.., PlugInAdequate)`, which is ENTAILED
        // by the `k̂ > 0.5` assertion five lines up: `PlugInAdequate` is
        // DEFINED as `k̂ < 0.5`. It could not distinguish `ImportanceCorrect`
        // from `Escalate`, and it would pass for any future fourth variant.
        //
        // Pin the classifier against its own documented thresholds instead, at
        // the k̂ this fixture actually produces. That discriminates all three
        // tiers, and unlike a hard-coded expected tier it cannot go stale if
        // the fixture's k̂ drifts within a band -- while still failing loudly if
        // the thresholds are ever rewired.
        let expected = RhoProposalAdequacy::from_k_hat(k_hat);
        assert_eq!(
            graded.adequacy, expected,
            "the adequacy grade must follow from k̂ = {k_hat} by the documented \
             thresholds (k̂ < 0.5 PlugInAdequate, ≤ 0.7 ImportanceCorrect, \
             else Escalate)"
        );
        assert!(
            matches!(
                graded.adequacy,
                RhoProposalAdequacy::ImportanceCorrect | RhoProposalAdequacy::Escalate
            ),
            "a heavy-tailed target must not grade the plug-in adequate, got {:?}",
            graded.adequacy
        );
    }

    #[test]
    fn effective_sample_size_is_bounded_and_deterministic() {
        let rho_hat = array![1.0];
        let h = array![[1.0]];
        // A quartic excess over the proposal quadratic, so the weights carry a
        // genuine tail and the determinism check covers a fitted `k̂`.
        let crit = |rho: &Array1<f64>| {
            let d = rho[0] - 1.0;
            Ok(0.5 * d * d + d * d * d * d / 24.0)
        };
        let a = rho_posterior_adequacy(&rho_hat, &h, crit)
            .expect("a formed")
            .expect("a present");
        let b = rho_posterior_adequacy(&rho_hat, &h, crit)
            .expect("b formed")
            .expect("b present");
        // Kish's (Σw)²/Σw² lies in [1, M]: Cauchy–Schwarz gives (Σw)² ≤ M Σw²,
        // and (Σw)² ≥ Σw² for non-negative weights. Both edges carry the
        // M-term summations' relative rounding M·ε (Σw and Σw²).
        let ess = a.effective_sample_size;
        let m = a.n_samples as f64;
        let rounding = 1.0 + m * f64::EPSILON;
        assert!(
            ess * rounding >= 1.0 && ess <= m * rounding,
            "ESS must lie in [1, M = {m}], got {ess}"
        );
        // Deterministic: identical k̂ across runs (fixed-seed stream).
        let (WeightTailShape::Pareto(k_a), WeightTailShape::Pareto(k_b)) =
            (a.tail_shape, b.tail_shape)
        else {
            panic!("a quartic target has a tail to fit: {:?}, {:?}", a.tail_shape, b.tail_shape);
        };
        assert_eq!(k_a.to_bits(), k_b.to_bits());
    }

    /// #2946 T2: a fit's `k̂` resolution is the Pareto shape error at the tail
    /// the fit used, `tail_count(M)`, at its own `k̂`. At `M = 100` a `k̂` at
    /// the `0.7` cutoff is resolved only to `≈ 0.27`, the value the module docs
    /// state.
    #[test]
    fn k_hat_standard_error_reads_the_fits_own_tail_2946() {
        let at_cutoff = RhoPosteriorAdequacy {
            tail_shape: WeightTailShape::Pareto(ESCALATE_K_HAT),
            adequacy: RhoProposalAdequacy::from_k_hat(ESCALATE_K_HAT),
            n_samples: 100,
            effective_sample_size: 10.0,
        };
        let se = k_hat_standard_error(&at_cutoff).expect("a fitted shape has a resolution");
        assert_eq!(
            se.to_bits(),
            gam_solve::psis::shape_standard_error(10, ESCALATE_K_HAT).to_bits()
        );
        assert!((se - 0.27).abs() < 0.005, "{se}");
        let larger = RhoPosteriorAdequacy { n_samples: 2155, ..at_cutoff.clone() };
        assert!(k_hat_standard_error(&larger).expect("a fitted shape has a resolution") < se);
    }

    #[test]
    fn empty_rho_returns_none() {
        let rho_hat: Array1<f64> = array![];
        let h = Array2::<f64>::zeros((0, 0));
        assert!(matches!(
            rho_posterior_adequacy(&rho_hat, &h, |_| Ok(0.0)),
            Ok(None)
        ));
    }

    /// The whitening factor is `R⁻ᵀ` of the Hessian itself, at any curvature
    /// scale. `2⁻⁶⁰` and `2⁻⁵⁸` have exact square roots, so the factor is
    /// `diag(2³⁰, 2²⁹)` bit for bit; a ridge sized to `max(1, diag)` swamped
    /// both curvatures and returned a factor near `1e5`.
    #[test]
    fn whitening_factor_is_exact_at_any_curvature_scale() {
        let tiny = 2.0_f64.powi(-60);
        let h = array![[tiny, 0.0], [0.0, 4.0 * tiny]];
        let l_inv = whitening_factor_from_outer_hessian(&h).expect("positive definite");
        assert_eq!(l_inv[[0, 0]], 2.0_f64.powi(30));
        assert_eq!(l_inv[[1, 1]], 2.0_f64.powi(29));
        assert_eq!(l_inv[[0, 1]], 0.0);
        assert_eq!(l_inv[[1, 0]], 0.0);
    }

    /// A singular outer Hessian has no Gaussian proposal. The ridge used to turn
    /// `[[1, 1], [1, 1]]` into a proposal with variance `≈ 1e10` along its null
    /// direction; the diagnostic refuses it instead.
    #[test]
    fn singular_outer_hessian_is_refused() {
        let rho_hat = array![0.0, 0.0];
        let h = array![[1.0, 1.0], [1.0, 1.0]];
        assert!(whitening_factor_from_outer_hessian(&h).is_err());
        let refusal = rho_posterior_adequacy(&rho_hat, &h, |_| Ok(0.0))
            .expect_err("a singular outer Hessian must be refused");
        assert!(
            matches!(refusal, RhoPosteriorRefusal::HessianNotPositiveDefinite { .. }),
            "the refusal names its reason, got {refusal}"
        );
    }
}

#[cfg(test)]
mod rho_posterior_escalation_tiers_tests;
