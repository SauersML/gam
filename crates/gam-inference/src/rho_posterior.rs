//! Marginal smoothing inference over the smoothing parameters `ρ`
//! (issue #938): the Tier-0 **PSIS adequacy diagnostic**, plus the auto-selected
//! escalation tiers — Tier-1 **Gauss-Hermite quadrature** over `ρ` (`K ≤ 4`,
//! `rho_posterior_quadrature`) and Tier-2 **NUTS over `ρ`** with the exact
//! profiled gradient (`K ≤ 16`, `rho_posterior_nuts`), routed by
//! [`escalate_rho_posterior`] when the diagnostic grades the plug-in
//! [`RhoProposalAdequacy::Escalate`].
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
//! at the default `M = 64` the tail sample is `8` and the standard error at the
//! `0.7` boundary is `≈ 0.27`; at `M = 512` it is `23` and `≈ 0.25`. Reaching a
//! standard error of `0.05` takes a tail of `≈ 10³`, i.e. `M ≈ 10⁶`. A single
//! `k̂` near a cutoff is therefore not evidence about which side of the cutoff
//! the truth lies on: separating a true shape from the `0.7` boundary needs
//! `⌈√M⌉` large enough that several standard errors fit in the gap. Anything
//! asserting a verdict (rather than reading a diagnostic) must size `M` from
//! `tail_count` and `shape_standard_error`.

use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_solve::estimate::EstimationError;
use gam_solve::psis::pareto_smooth_weights;
use ndarray::{Array1, Array2};

// The `ρ`-posterior adequacy/escalation DATA types were contract-downed to
// the neutral `gam-problem` crate (#1521) so gam-solve can store/return them
// without a back-edge into gam-inference. The COMPUTATION below (PSIS
// adequacy diagnostic, Tier-1 quadrature, Tier-2 NUTS via `hmc_io`) stays here and
// constructs these types under the names re-exported here.
pub use gam_problem::rho_posterior::{
    ESCALATE_K_HAT, PLUG_IN_ADEQUATE_K_HAT, RhoMixtureNode, RhoPosteriorAdequacy,
    RhoPosteriorEscalation, RhoPosteriorMixture, RhoPosteriorNotComputed, RhoPosteriorOutcome,
    RhoPosteriorRefusal, RhoPosteriorSamples, RhoProposalAdequacy,
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
        criterion: &dyn Fn(&Array1<f64>) -> Option<f64>,
        n_samples: Option<usize>,
    ) -> Result<Option<RhoPosteriorAdequacy>, RhoPosteriorRefusal> {
        rho_posterior_adequacy(rho_hat, outer_hessian, criterion, n_samples)
    }

    fn escalate_rho_posterior(
        &self,
        rho_hat: &Array1<f64>,
        outer_hessian: &Array2<f64>,
        criterion: &mut dyn FnMut(&Array1<f64>) -> Option<f64>,
        criterion_and_grad: &mut (dyn FnMut(&Array1<f64>) -> Option<(f64, Array1<f64>)> + Send),
    ) -> RhoPosteriorEscalation {
        escalate_rho_posterior(rho_hat, outer_hessian, criterion, criterion_and_grad)
    }
}

/// Largest `K` for which the Tier-1 Gauss-Hermite product grid is affordable
/// (3–5 nodes per axis ⇒ at most 81–125 criterion evaluations).
pub(crate) const TIER1_MAX_DIM: usize = 4;
/// Largest `K` for which the Tier-2 NUTS escalation runs; beyond this the fit
/// honestly reports that escalation is unavailable.
pub(crate) const TIER2_MAX_DIM: usize = 16;
/// Post-warmup draw budget for the auto-selected Tier-2 escalation. Each
/// leapfrog step is one warm inner profile solve, so the budget is deliberately
/// modest: the whitened `ρ`-posterior is a smooth, near-Gaussian, low-dim
/// target where a few hundred draws already pin the first two moments.
const ESCALATION_NUTS_SAMPLES: usize = 256;
/// Deterministic seed for the auto-selected Tier-2 escalation (no clock).
const ESCALATION_NUTS_SEED: u64 = 0x938_5EED_0938_5EED;

const DEFAULT_M: usize = 64;
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
/// normalized log-weight)`. Infeasible nodes carry `cost = +∞` and zero weight.
struct NormalizedQuadratureNode {
    rho: Array1<f64>,
    cost: f64,
    weight: f64,
    log_weight: f64,
}

/// Tier-1 quadrature core (#938): whiten by the exact outer Hessian, enumerate
/// the Gauss-Hermite product grid, reweight each node by the exact profiled
/// criterion `exp(−V(ρ_m) + V(ρ̂) + ½‖z_m‖²) × GH-weight`, and normalize.
/// `rho_posterior_quadrature` is its criterion-closure adapter.
fn quadrature_nodes_core<E>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    nodes_per_axis: usize,
    cost_hat: f64,
    mut eval_node: E,
) -> Result<(Vec<NormalizedQuadratureNode>, f64), EstimationError>
where
    E: FnMut(&Array1<f64>) -> Result<Option<f64>, EstimationError>,
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
        let (cost, log_weight) = match eval_node(&rho)? {
            Some(cost) if cost.is_finite() => {
                let half_norm_sq = 0.5 * z.iter().map(|&v| v * v).sum::<f64>();
                (cost, log_base_weight - cost + cost_hat + half_norm_sq)
            }
            // Infeasible node: zero importance weight, never fatal.
            _ => (f64::INFINITY, f64::NEG_INFINITY),
        };
        if log_weight.is_finite() {
            max_log_weight = max_log_weight.max(log_weight);
        }
        raw_nodes.push((rho, cost, log_weight));
    }
    if !max_log_weight.is_finite() {
        return Err(EstimationError::RemlOptimizationFailed(
            "rho_posterior_quadrature: all quadrature nodes were non-finite".to_string(),
        ));
    }
    let mut total = 0.0;
    let mut scaled = Vec::with_capacity(raw_nodes.len());
    for (_, _, log_weight) in &raw_nodes {
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

/// Tier-1 of the marginal-smoothing inference stack (#938): adaptive
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
pub(crate) fn rho_posterior_quadrature<F>(
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
            Ok(criterion(rho))
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

/// Tier-2 of the marginal-smoothing inference stack (#938): NUTS over `ρ`
/// with the exact profiled gradient, whitened by the exact outer Hessian at
/// `ρ̂` (the `hmc` module's whitening design reused one level up).
///
/// * `criterion_and_grad` — `ρ ↦ (criterion(ρ), ∇_ρ criterion(ρ))`, both EXACT
///   (the engine's LAML value and ρ-gradient); `None` for infeasible `ρ`. Each
///   call is one warm inner profile solve + IFT gradient.
/// * `n_samples` — post-warmup draws per chain. Warmup ends when adaptation has
///   stabilized and the chains agree.
/// * `seed` — deterministic seeding: the seed feeds the same splitmix64 chain /
///   transition streams as every other NUTS entry point. No clock, no global
///   RNG: the same `(fit, seed)` yields the same draws every run.
pub(crate) fn rho_posterior_nuts<F>(
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
    let config = crate::hmc_io::NutsConfig {
        n_samples: n_samples.max(4),
        target_accept: 0.9,
        seed,
    };
    let result = crate::hmc_io::run_rho_criterion_nuts(
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

/// The auto-selection seam (#938): given an [`RhoProposalAdequacy::Escalate`]
/// grade from the Tier-0 adequacy diagnostic, pick and run the escalation tier by
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

/// Compute the Tier-0 PSIS `ρ`-adequacy diagnostic.
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
/// Returns `Ok(None)` when `K = 0`: there is nothing to grade. Returns the typed
/// [`RhoPosteriorRefusal`] naming the site when the diagnostic cannot be formed —
/// an outer Hessian whose shape does not match `ρ̂` or that is not positive
/// definite, an infeasible or non-finite criterion at `ρ̂`, no proposal draw with a
/// finite criterion, too few finite weights for the Pareto tail fit, a non-finite
/// tail shape, or smoothed weights that do not normalize.
pub fn rho_posterior_adequacy<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    criterion: F,
    n_samples: Option<usize>,
) -> Result<Option<RhoPosteriorAdequacy>, RhoPosteriorRefusal>
where
    F: Fn(&Array1<f64>) -> Option<f64>,
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
    let cost_hat = criterion(rho_hat).ok_or(RhoPosteriorRefusal::CriterionInfeasibleAtRhoHat)?;
    if !cost_hat.is_finite() {
        return Err(RhoPosteriorRefusal::CriterionNotFiniteAtRhoHat);
    }
    let l_inv = whitening_factor_from_outer_hessian(outer_hessian)?;
    let m = n_samples
        .unwrap_or(DEFAULT_M)
        .max(2 * gam_solve::psis::MIN_TAIL_COUNT);

    let mut rng = DetNormal::new(ADEQUACY_SEED);
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
        return Err(RhoPosteriorRefusal::NoFiniteProposal);
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

    let psis = pareto_smooth_weights(&weights).ok_or(RhoPosteriorRefusal::TooFewFiniteWeights)?;
    let k_hat = psis.k_hat;
    if !k_hat.is_finite() {
        return Err(RhoPosteriorRefusal::TailShapeNotFinite);
    }

    // Self-normalize the smoothed weights. Normalized weights sum to 1, so
    // Cauchy–Schwarz gives Σw² ≥ 1/M > 0 and the Kish ESS lies in [1, M].
    let total: f64 = psis.smoothed.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return Err(RhoPosteriorRefusal::SmoothedWeightsNotNormalizable);
    }
    let sum_sq: f64 = psis
        .smoothed
        .iter()
        .map(|&w| {
            let normalized = w / total;
            normalized * normalized
        })
        .sum();

    Ok(Some(RhoPosteriorAdequacy {
        k_hat,
        adequacy: RhoProposalAdequacy::from_k_hat(k_hat),
        n_samples: m,
        effective_sample_size: 1.0 / sum_sq,
    }))
}

/// Standard error of `adequacy.k_hat`, the resolution of its grade (#2946 T2).
///
/// It is [`gam_solve::psis::shape_standard_error`] at the tail sample
/// [`gam_solve::psis::tail_count`]`(n_samples)` that the Pareto fit used,
/// evaluated at the reported shape as a plug-in for the true one. A grade whose
/// `k_hat` lies within a few of these of [`PLUG_IN_ADEQUATE_K_HAT`] or
/// [`ESCALATE_K_HAT`] does not say which side of that cutoff the truth is on.
pub fn k_hat_standard_error(adequacy: &RhoPosteriorAdequacy) -> f64 {
    gam_solve::psis::shape_standard_error(
        gam_solve::psis::tail_count(adequacy.n_samples),
        adequacy.k_hat,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// CLOSED-FORM FIXTURE: when the criterion IS exactly the Gaussian
    /// `−log π(ρ|y) = ½(ρ−ρ̂)ᵀ H_ρ (ρ−ρ̂)` that the Laplace proposal assumes,
    /// the importance weights are all identically 1 — the proposal is the
    /// target. PSIS must then report a tiny `k̂` and grade the plug-in adequate.
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
            Some(0.5 * q)
        };
        let graded = rho_posterior_adequacy(&rho_hat, &h, crit, Some(256))
            .expect("diagnostic formed")
            .expect("diagnostic present");
        // All weights equal ⇒ ESS == M and k̂ small ⇒ plug-in adequate.
        assert!(
            (graded.effective_sample_size - graded.n_samples as f64).abs() < 1e-6,
            "uniform weights must give ESS == M: ess={} M={}",
            graded.effective_sample_size,
            graded.n_samples
        );
        assert!(
            graded.k_hat < 0.5,
            "exact-Gaussian target must yield small k̂, got {}",
            graded.k_hat
        );
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
            Some((1.0 + r * r).ln())
        };
        let graded = rho_posterior_adequacy(&rho_hat, &h, crit, Some(512))
            .expect("diagnostic formed")
            .expect("diagnostic present");
        assert!(
            graded.k_hat > 0.5,
            "heavy-tailed target must raise k̂ above 0.5, got {}",
            graded.k_hat
        );
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
        let expected = RhoProposalAdequacy::from_k_hat(graded.k_hat);
        assert_eq!(
            graded.adequacy, expected,
            "the adequacy grade must follow from k̂ = {} by the documented \
             thresholds (k̂ < 0.5 PlugInAdequate, ≤ 0.7 ImportanceCorrect, \
             else Escalate)",
            graded.k_hat
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
        let crit = |rho: &Array1<f64>| {
            let d = rho[0] - 1.0;
            Some(0.5 * d * d)
        };
        let a = rho_posterior_adequacy(&rho_hat, &h, crit, Some(64))
            .expect("a formed")
            .expect("a present");
        let b = rho_posterior_adequacy(&rho_hat, &h, crit, Some(64))
            .expect("b formed")
            .expect("b present");
        // Kish's (Σw)²/Σw² of self-normalized weights lies in [1, M]: Σw = 1 and
        // Cauchy–Schwarz give 1/M ≤ Σw² ≤ 1. Both edges carry the M-term
        // summations' relative rounding M·ε (the normalizing total and Σw²).
        let ess = a.effective_sample_size;
        let m = a.n_samples as f64;
        let rounding = 1.0 + m * f64::EPSILON;
        assert!(
            ess * rounding >= 1.0 && ess <= m * rounding,
            "ESS must lie in [1, M = {m}], got {ess}"
        );
        // Deterministic: identical k̂ across runs (fixed-seed stream).
        assert_eq!(a.k_hat.to_bits(), b.k_hat.to_bits());
    }

    /// #2946 T2: a fit's `k̂` resolution is the Pareto shape error at the tail
    /// the fit used, `tail_count(M)`, at its own `k̂`. At the default `M = 64` a
    /// `k̂` at the `0.7` cutoff is resolved only to `≈ 0.27`, the value the
    /// module docs state.
    #[test]
    fn k_hat_standard_error_reads_the_fits_own_tail_2946() {
        let at_cutoff = RhoPosteriorAdequacy {
            k_hat: ESCALATE_K_HAT,
            adequacy: RhoProposalAdequacy::from_k_hat(ESCALATE_K_HAT),
            n_samples: DEFAULT_M,
            effective_sample_size: 10.0,
        };
        let se = k_hat_standard_error(&at_cutoff);
        assert_eq!(
            se.to_bits(),
            gam_solve::psis::shape_standard_error(8, ESCALATE_K_HAT).to_bits()
        );
        assert!((se - 0.27).abs() < 0.005, "{se}");
        let larger = RhoPosteriorAdequacy { n_samples: 512, ..at_cutoff.clone() };
        assert!(k_hat_standard_error(&larger) < se);
    }

    #[test]
    fn empty_rho_returns_none() {
        let rho_hat: Array1<f64> = array![];
        let h = Array2::<f64>::zeros((0, 0));
        assert!(matches!(
            rho_posterior_adequacy(&rho_hat, &h, |_| Some(0.0), None),
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
        let refusal = rho_posterior_adequacy(&rho_hat, &h, |_| Some(0.0), Some(64))
            .expect_err("a singular outer Hessian must be refused");
        assert!(
            matches!(refusal, RhoPosteriorRefusal::HessianNotPositiveDefinite { .. }),
            "the refusal names its reason, got {refusal}"
        );
    }
}

#[cfg(test)]
mod rho_posterior_escalation_tiers_tests;
