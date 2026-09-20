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
use gam_math::probability::{
    log1mexp_positive, normal_logcdf, normal_logsf, standard_normal_quantile_from_log_cdf,
};
use gam_math::special::logaddexp;
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
        support: &(Array1<f64>, Array1<f64>),
        held: &[usize],
        criterion: &dyn Fn(&Array1<f64>) -> Result<f64, String>,
        n_samples: Option<usize>,
    ) -> Result<Option<RhoPosteriorAdequacy>, RhoPosteriorRefusal> {
        rho_posterior_adequacy(rho_hat, outer_hessian, support, held, criterion, n_samples)
    }

    fn escalate_rho_posterior(
        &self,
        rho_hat: &Array1<f64>,
        outer_hessian: &Array2<f64>,
        criterion: &mut dyn FnMut(&Array1<f64>) -> Result<f64, String>,
        criterion_and_grad: &mut (dyn FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), String>
                  + Send),
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

/// Deterministic uniform stream (splitmix64) driving the proposal's inverse-CDF
/// draws. No RNG / env dependency: the same seed yields the same draws every run.
pub(crate) struct DetNormal {
    state: u64,
}
impl DetNormal {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    /// `(k + ½)/2⁵³`, strictly inside (0, 1): `ln u` and `ln(1 − u)` are finite.
    fn uniform(&mut self) -> f64 {
        let z = gam_linalg::utils::splitmix64(&mut self.state);
        (((z >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
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

/// Tier-1 of the marginal-smoothing inference stack (#938): adaptive
/// Gauss-Hermite quadrature over `ρ` (`K ≤ 4`), criterion-closure form.
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
///   3 for `K ≤ 4` — at most 125 criterion evaluations either way).
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
    let nodes_per_axis = nodes_per_axis.unwrap_or(if k <= 2 { 5 } else { 3 });
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
/// with the exact profiled gradient, whitened by the exact outer Hessian at
/// `ρ̂` (the `hmc` module's whitening design reused one level up).
///
/// * `criterion_and_grad` — `ρ ↦ (criterion(ρ), ∇_ρ criterion(ρ))`, both EXACT
///   (the engine's LAML value and ρ-gradient), or the reason it cannot value a
///   position, which fails the run. Each call is one warm inner profile solve +
///   IFT gradient.
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
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), String> + Send,
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
    F: FnMut(&Array1<f64>) -> Result<f64, String>,
    G: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), String> + Send,
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

/// `ln(Φ(b) − Φ(a))` for `a < b`, formed where no term is close to one: the
/// upper tail `Q(a) − Q(b)` when `a ≥ 0`, the lower tail `Φ(b) − Φ(a)` when
/// `b ≤ 0`, and for an interval straddling 0 the sum of the two positive
/// half-masses `½erf(b/√2) + ½erf(−a/√2)`. `−∞` only when a one-sided
/// interval's two log-probabilities round to the same value, which is the
/// interval's own rounding, not a threshold.
fn log_standard_normal_interval_mass(a: f64, b: f64) -> f64 {
    if a >= 0.0 {
        let (log_sf_a, log_sf_b) = (normal_logsf(a), normal_logsf(b));
        log_sf_a + log1mexp_positive(log_sf_a - log_sf_b)
    } else if b <= 0.0 {
        let (log_cdf_a, log_cdf_b) = (normal_logcdf(a), normal_logcdf(b));
        log_cdf_b + log1mexp_positive(log_cdf_b - log_cdf_a)
    } else {
        // An interval around 0 holds `½(erf(b/√2) + erf(−a/√2))`: a sum of two
        // positive terms, exact down to the narrowest representable interval.
        let half_erf = |x: f64| 0.5 * libm::erf(x / std::f64::consts::SQRT_2);
        (half_erf(b) + half_erf(-a)).ln()
    }
}

/// The standard normal truncated to `[a, b]`, drawn by inverting its CDF at
/// `u ∈ (0, 1)`: `Φ(x) = Φ(a) + u·p` with `p = Φ(b) − Φ(a)`, `log_mass = ln p`.
///
/// Both `ln Φ(x) = ln(Φ(a) + u p)` and `ln Q(x) = ln(Q(b) + (1 − u) p)` are
/// formed in log space, and the smaller of the two, which is at most `ln ½`,
/// is inverted, so neither tail forms a probability-space subtraction. `Φ⁻¹`
/// is monotone, so the exact `x` lies in `[a, b]`; the final projection onto
/// `[a, b]` removes only the quantile's rounding.
fn truncated_standard_normal(a: f64, b: f64, log_mass: f64, u: f64) -> f64 {
    let log_cdf = logaddexp(normal_logcdf(a), u.ln() + log_mass);
    let log_sf = logaddexp(normal_logsf(b), (-u).ln_1p() + log_mass);
    let x = if log_cdf <= log_sf {
        standard_normal_quantile_from_log_cdf(log_cdf)
    } else {
        standard_normal_quantile_from_log_cdf(log_sf).map(|x| -x)
    }
    .expect("the smaller of ln Φ(x) and ln Q(x) is finite and at most ln ½");
    x.clamp(a, b)
}

/// The Laplace proposal of `π(ρ|y)` restricted to its support (#3010).
///
/// The support is the box on which the criterion has a value. It is finite
/// only at a literal face, past which the criterion has none; past a saturated
/// face the criterion continues, so that side is unbounded. A held coordinate (railed by the certificate or found on a
/// face, and any with `ρ̂` on a finite face of the support) is the face-reduced
/// model's, not a direction to sample, so it stays at `ρ̂`. The free coordinates
/// `f` are drawn from the Laplace approximation conditioned on the held ones at
/// `ρ̂`, mean `ρ̂_f` and precision the free block `H_ff` of the outer Hessian,
/// restricted to the support.
///
/// The restriction is drawn exactly, with no rejection: `ρ_f = ρ̂_f + U z` with
/// `U = R⁻ᵀ` upper triangular, so coordinate `a` depends on `z_a, …, z_{k−1}`
/// only. Sweeping `a` from the last coordinate to the first, the support
/// confines `z_a` to an interval given the `z` already drawn, and `z_a` is drawn
/// from the standard normal truncated to it (Geweke–Hajivassiliou–Keane). Every
/// draw lies in the support, whatever share of the Gaussian it holds. The draw's
/// density is `∏_a φ(z_a) / p_a` with `p_a` the mass of `z_a`'s interval, so
/// its negative log-density is `½(ρ − ρ̂)ᵀ H_ρ (ρ − ρ̂) + Σ_a ln p_a` up to a
/// constant that is the same for every draw and cancels from self-normalized
/// importance weights and from the scale-free Pareto tail fit. The quadratic
/// equals `½‖z‖²` in exact arithmetic; it is taken at the rounded draw `ρ`,
/// where the criterion is valued, so an exact proposal has exactly flat
/// weights (#3202). A held coordinate has `ρ_i = ρ̂_i`, so the full `H_ρ` and
/// its free block `H_ff` give the same quadratic.
///
/// This is the one proposal the Tier-0 diagnostic draws from.
pub(crate) struct DomainLaplaceProposal {
    rho_hat: Array1<f64>,
    lower: Array1<f64>,
    upper: Array1<f64>,
    free: Vec<usize>,
    /// The outer Hessian `H_ρ`, whose quadratic values each draw.
    outer_hessian: Array2<f64>,
    /// `R_ff⁻ᵀ` for `H_ff = R_ff R_ffᵀ` (upper triangular): `ρ_f = ρ̂_f + L_inv z`
    /// has covariance `H_ff⁻¹` and `‖z‖² = (ρ_f − ρ̂_f)ᵀ H_ff (ρ_f − ρ̂_f)`.
    l_inv: Array2<f64>,
}

impl DomainLaplaceProposal {
    /// `support` bounds each coordinate it has an entry for; a coordinate
    /// past the end of either bound is unbounded on that side. `held` names the
    /// coordinates the certificate railed.
    pub(crate) fn new(
        rho_hat: &Array1<f64>,
        outer_hessian: &Array2<f64>,
        support: &(Array1<f64>, Array1<f64>),
        held: &[usize],
    ) -> Result<Self, RhoPosteriorRefusal> {
        let k = rho_hat.len();
        if outer_hessian.nrows() != k || outer_hessian.ncols() != k {
            return Err(RhoPosteriorRefusal::HessianShape {
                rows: outer_hessian.nrows(),
                cols: outer_hessian.ncols(),
                k,
            });
        }
        let lower = Array1::from_iter(
            (0..k).map(|i| support.0.get(i).copied().unwrap_or(f64::NEG_INFINITY)),
        );
        let upper =
            Array1::from_iter((0..k).map(|i| support.1.get(i).copied().unwrap_or(f64::INFINITY)));
        let free: Vec<usize> = (0..k)
            .filter(|&i| !held.contains(&i) && rho_hat[i] > lower[i] && rho_hat[i] < upper[i])
            .collect();
        let free_hessian = Array2::from_shape_fn((free.len(), free.len()), |(a, b)| {
            outer_hessian[[free[a], free[b]]]
        });
        let l_inv = if free.is_empty() {
            free_hessian
        } else {
            whitening_factor_from_outer_hessian(&free_hessian)?
        };
        Ok(Self {
            rho_hat: rho_hat.clone(),
            lower,
            upper,
            free,
            outer_hessian: outer_hessian.clone(),
            l_inv,
        })
    }

    /// Number of free coordinates.
    pub(crate) fn free_dim(&self) -> usize {
        self.free.len()
    }

    /// One draw `(ρ, ½(ρ − ρ̂)ᵀ H_ρ (ρ − ρ̂) + Σ_a ln p_a)`: the point in the
    /// support and its negative log proposal density up to the shared constant.
    ///
    /// Refused as [`RhoPosteriorRefusal::DegenerateProposalInterval`] only when
    /// an interval's two log-probabilities round to the same value, so its mass
    /// is not representable and the density of the draw is not defined.
    fn draw(&self, rng: &mut DetNormal) -> Result<(Array1<f64>, f64), RhoPosteriorRefusal> {
        let kf = self.free.len();
        let mut z = Array1::<f64>::zeros(kf);
        let mut rho = self.rho_hat.clone();
        let mut log_interval_mass = 0.0;
        for a in (0..kf).rev() {
            let i = self.free[a];
            let shift: f64 = (a + 1..kf).map(|b| self.l_inv[[a, b]] * z[b]).sum();
            let scale = self.l_inv[[a, a]];
            let lo = (self.lower[i] - self.rho_hat[i] - shift) / scale;
            let hi = (self.upper[i] - self.rho_hat[i] - shift) / scale;
            let log_mass = log_standard_normal_interval_mass(lo, hi);
            if !log_mass.is_finite() {
                return Err(RhoPosteriorRefusal::DegenerateProposalInterval { coordinate: i });
            }
            z[a] = truncated_standard_normal(lo, hi, log_mass, rng.uniform());
            // `z_a ∈ [lo, hi]` puts `ρ_i` in its box exactly; the projection
            // removes only the rounding of the affine map back to `ρ`.
            rho[i] = (self.rho_hat[i] + shift + scale * z[a]).clamp(self.lower[i], self.upper[i]);
            log_interval_mass += log_mass;
        }
        let k = rho.len();
        let mut quad = 0.0;
        for i in 0..k {
            let di = rho[i] - self.rho_hat[i];
            for j in 0..k {
                quad += di * self.outer_hessian[[i, j]] * (rho[j] - self.rho_hat[j]);
            }
        }
        Ok((rho, 0.5 * quad + log_interval_mass))
    }

    /// `m` draws of the proposal, each `(ρ, ½(ρ − ρ̂)ᵀ H_ρ (ρ − ρ̂) + Σ_a ln p_a)`. Every draw is
    /// in the support, and the criterion is evaluated for none of them here.
    pub(crate) fn sample(
        &self,
        m: usize,
        rng: &mut DetNormal,
    ) -> Result<Vec<(Array1<f64>, f64)>, RhoPosteriorRefusal> {
        (0..m).map(|_| self.draw(rng)).collect()
    }
}

/// Compute the Tier-0 PSIS `ρ`-adequacy diagnostic.
///
/// * `rho_hat` — the converged smoothing parameters `ρ̂` (length `K`).
/// * `outer_hessian` — the exact outer Hessian `H_ρ` of the criterion at `ρ̂`
///   (`K × K`). The proposal precision is its free block `H_ff`, which must be
///   positive definite.
/// * `support` — the `(lower, upper)` box of `π(ρ|y)`: finite only at a
///   literal face, infinite past a saturated one.
/// * `held` — the coordinates the outer certificate railed or found on a face.
///   They, and every coordinate with `ρ̂` on a finite face of `support`, stay
///   at `ρ̂`
///   ([`DomainLaplaceProposal`]).
/// * `criterion` — evaluates the outer criterion `−log π(ρ|y)` (the LAML/REML
///   objective) at a trial `ρ`, or says why it cannot. This is the
///   `OuterObjective::eval_cost` contract, supplied by the caller that retains
///   (or rebuilds) the objective. It is only ever called inside `support`.
///   Every draw carries proposal mass, so the diagnostic is refused at the
///   first draw it cannot value, never formed from the rest.
/// * `n_samples` — proposal draw count `M` (defaults to 64 when `None`).
///
/// Returns `Ok(None)` when no coordinate is free: there is nothing to grade.
/// Returns the typed [`RhoPosteriorRefusal`] naming the site when the diagnostic
/// cannot be formed — an outer Hessian whose shape does not match `ρ̂` or whose
/// free block is not positive definite, an unavailable or non-finite criterion at
/// `ρ̂` or at a draw, a proposal interval whose mass rounds to zero, a failed
/// Pareto tail fit, a non-finite tail shape, or smoothed weights that do not
/// normalize.
pub fn rho_posterior_adequacy<F>(
    rho_hat: &Array1<f64>,
    outer_hessian: &Array2<f64>,
    support: &(Array1<f64>, Array1<f64>),
    held: &[usize],
    criterion: F,
    n_samples: Option<usize>,
) -> Result<Option<RhoPosteriorAdequacy>, RhoPosteriorRefusal>
where
    F: Fn(&Array1<f64>) -> Result<f64, String>,
{
    let proposal = DomainLaplaceProposal::new(rho_hat, outer_hessian, support, held)?;
    if proposal.free_dim() == 0 {
        return Ok(None);
    }
    let cost_hat = criterion(rho_hat)
        .map_err(|detail| RhoPosteriorRefusal::CriterionUnavailableAtRhoHat { detail })?;
    if !cost_hat.is_finite() {
        return Err(RhoPosteriorRefusal::CriterionNotFiniteAtRhoHat);
    }
    let m = n_samples
        .unwrap_or(DEFAULT_M)
        .max(2 * gam_solve::psis::MIN_TAIL_COUNT);

    let mut rng = DetNormal::new(ADEQUACY_SEED);
    let mut raw_weights: Vec<f64> = Vec::with_capacity(m);
    for (draw, (rho_m, neg_log_q)) in proposal.sample(m, &mut rng)?.iter().enumerate() {
        // log w_m = −criterion(ρ_m) + criterion(ρ̂) − ln q(ρ_m), with q valued
        // at the draw ρ_m itself (#3202).
        let cost = criterion(rho_m)
            .map_err(|detail| RhoPosteriorRefusal::CriterionUnavailableAtDraw { draw, detail })?;
        if !cost.is_finite() {
            return Err(RhoPosteriorRefusal::CriterionNotFiniteAtDraw { draw });
        }
        raw_weights.push(-cost + cost_hat + neg_log_q);
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
        tail_shape: psis.shape,
        adequacy: RhoProposalAdequacy::from_tail_shape(psis.shape),
        n_samples: m,
        effective_sample_size: 1.0 / sum_sq,
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

    /// A domain with no entry bounds nothing.
    fn unbounded() -> (Array1<f64>, Array1<f64>) {
        (Array1::zeros(0), Array1::zeros(0))
    }

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
        let graded = rho_posterior_adequacy(&rho_hat, &h, &unbounded(), &[], crit, Some(256))
            .expect("diagnostic formed")
            .expect("diagnostic present");
        // The criterion and the proposal value each draw by the same
        // arithmetic, so every log-weight is exactly 0 and every weight 1:
        // M = 256 sums, normalizes and squares without rounding.
        assert_eq!(graded.tail_shape, WeightTailShape::Flat);
        assert_eq!(graded.effective_sample_size, graded.n_samples as f64);
        assert_eq!(graded.adequacy, RhoProposalAdequacy::PlugInAdequate);
        assert_eq!(k_hat_standard_error(&graded), None);
    }

    /// #3202 repro: 1-D, `ρ̂ = 1`, `H = [[1]]`, `c(ρ) = ½(ρ − 1)²`, `M = 64`.
    /// The proposal is exact. Valued at the unrounded draw `z` rather than at
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
        let graded = rho_posterior_adequacy(&rho_hat, &h, &unbounded(), &[], crit, Some(64))
            .expect("an exact proposal is graded, not refused")
            .expect("diagnostic present");
        assert_eq!(graded.tail_shape, WeightTailShape::Flat);
        assert_eq!(graded.n_samples, 64);
        assert_eq!(graded.effective_sample_size, 64.0);
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
        let graded = rho_posterior_adequacy(&rho_hat, &h, &unbounded(), &[], crit, Some(512))
            .expect("diagnostic formed")
            .expect("diagnostic present");
        let WeightTailShape::Pareto(k_hat) = graded.tail_shape else {
            panic!("a heavy-tailed target has a tail to fit, got {:?}", graded.tail_shape);
        };
        assert!(k_hat > 0.5, "heavy-tailed target must raise k̂ above 0.5, got {k_hat}");
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
        let a = rho_posterior_adequacy(&rho_hat, &h, &unbounded(), &[], crit, Some(64))
            .expect("a formed")
            .expect("a present");
        let b = rho_posterior_adequacy(&rho_hat, &h, &unbounded(), &[], crit, Some(64))
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
        let (WeightTailShape::Pareto(k_a), WeightTailShape::Pareto(k_b)) =
            (a.tail_shape, b.tail_shape)
        else {
            panic!("a quartic target has a tail to fit: {:?}, {:?}", a.tail_shape, b.tail_shape);
        };
        assert_eq!(k_a.to_bits(), k_b.to_bits());
    }

    /// #2946 T2: a fit's `k̂` resolution is the Pareto shape error at the tail
    /// the fit used, `tail_count(M)`, at its own `k̂`. At the default `M = 64` a
    /// `k̂` at the `0.7` cutoff is resolved only to `≈ 0.27`, the value the
    /// module docs state.
    #[test]
    fn k_hat_standard_error_reads_the_fits_own_tail_2946() {
        let at_cutoff = RhoPosteriorAdequacy {
            tail_shape: WeightTailShape::Pareto(ESCALATE_K_HAT),
            adequacy: RhoProposalAdequacy::from_k_hat(ESCALATE_K_HAT),
            n_samples: DEFAULT_M,
            effective_sample_size: 10.0,
        };
        let se = k_hat_standard_error(&at_cutoff).expect("a fitted shape has a resolution");
        assert_eq!(
            se.to_bits(),
            gam_solve::psis::shape_standard_error(8, ESCALATE_K_HAT).to_bits()
        );
        assert!((se - 0.27).abs() < 0.005, "{se}");
        let larger = RhoPosteriorAdequacy { n_samples: 512, ..at_cutoff.clone() };
        assert!(k_hat_standard_error(&larger).expect("a fitted shape has a resolution") < se);
    }

    #[test]
    fn empty_rho_returns_none() {
        let rho_hat: Array1<f64> = array![];
        let h = Array2::<f64>::zeros((0, 0));
        assert!(matches!(
            rho_posterior_adequacy(&rho_hat, &h, &unbounded(), &[], |_| Ok(0.0), None),
            Ok(None)
        ));
    }

    /// Counts every criterion evaluation and records where it happened. The
    /// criterion is the Laplace quadratic plus a quartic `Σ d_i⁴ / 24`, so the
    /// importance weights vary from draw to draw and the Pareto tail has
    /// excesses to fit.
    fn recording_quartic<'a>(
        rho_hat: &'a Array1<f64>,
        h: &'a Array2<f64>,
        seen: &'a std::cell::RefCell<Vec<Array1<f64>>>,
    ) -> impl Fn(&Array1<f64>) -> Result<f64, String> + 'a {
        move |rho: &Array1<f64>| {
            seen.borrow_mut().push(rho.clone());
            let d = rho - rho_hat;
            Ok(0.5 * d.dot(&h.dot(&d)) + d.mapv(|v| v.powi(4)).sum() / 24.0)
        }
    }

    /// #3010: every criterion evaluation lies in the support, and a coordinate
    /// held on a face (here `ρ̂_0` at its lower face, and `ρ_3`, which the
    /// certificate railed) stays at `ρ̂` bit for bit. The free block has
    /// proposal standard deviation 10 on `ρ_1` and `ρ_2` against boxes of
    /// half-width 1 and 3, so an untruncated proposal lands outside the box on
    /// more than nine draws in ten.
    ///
    /// The proposal is the conditional Laplace `N(ρ̂_f, H_ff⁻¹)`: with the held
    /// coordinates at `ρ̂`, its reported `−ln q` minus `½ d_fᵀ H_ff d_f` is
    /// `Σ_a ln p_a`, and with a diagonal `H_ff` every sequential interval is the
    /// same for every draw, so that difference is one constant. The held
    /// coordinates couple to the free ones, so a proposal on the marginal
    /// covariance `(H⁻¹)_ff` instead would not reproduce it.
    #[test]
    fn draws_stay_in_the_domain_and_held_coordinates_stay_at_rho_hat_3010() {
        let rho_hat = array![-2.0, 0.2, 1.0, 0.5];
        let lower = array![-2.0, -1.0, -3.0, -3.0];
        let upper = array![2.0, 1.0, 3.0, 3.0];
        let h = array![
            [1.0, 0.05, 0.05, 0.0],
            [0.05, 1.0e-2, 0.0, -0.05],
            [0.05, 0.0, 1.0e-2, -0.05],
            [0.0, -0.05, -0.05, 1.0]
        ];
        let seen = std::cell::RefCell::new(Vec::new());
        let crit = recording_quartic(&rho_hat, &h, &seen);
        let graded = rho_posterior_adequacy(
            &rho_hat,
            &h,
            &(lower.clone(), upper.clone()),
            &[3],
            crit,
            Some(64),
        )
        .expect("diagnostic formed")
        .expect("diagnostic present");
        let seen = seen.into_inner();
        assert_eq!(seen.len(), 1 + graded.n_samples, "rho_hat plus one call per draw");
        for rho in &seen {
            for i in 0..4 {
                assert!(
                    rho[i] >= lower[i] && rho[i] <= upper[i],
                    "criterion evaluated outside the support at {rho}"
                );
            }
            assert_eq!(rho[0].to_bits(), rho_hat[0].to_bits(), "the face coordinate moved");
            assert_eq!(rho[3].to_bits(), rho_hat[3].to_bits(), "the railed coordinate moved");
        }
        let proposal =
            DomainLaplaceProposal::new(&rho_hat, &h, &(lower.clone(), upper.clone()), &[3])
                .expect("proposal formed");
        let mut rng = DetNormal::new(ADEQUACY_SEED);
        let draws = proposal.sample(graded.n_samples, &mut rng).expect("draws");
        let log_box_mass: Vec<f64> = draws
            .iter()
            .map(|(rho, neg_log_q)| {
                let d = rho - &rho_hat;
                neg_log_q - 0.5 * d.dot(&h.dot(&d))
            })
            .collect();
        for v in &log_box_mass {
            assert!(
                (v - log_box_mass[0]).abs() <= 1e-9,
                "the conditional H_ff proposal has one interval mass per draw: {v} vs {}",
                log_box_mass[0]
            );
        }
    }

    /// #3010: a box holding a sliver of the Gaussian is sampled exactly, with
    /// no rejection. Proposal standard deviation 100 against a box of
    /// half-width `10⁻³` holds about `8·10⁻⁶` of the Gaussian, which a
    /// rejection sampler would need `10⁵` draws per accepted point to reach.
    /// Every one of the `M` draws is in the box, the criterion is evaluated
    /// once per draw, and the importance weights are finite, so the ESS lies in
    /// `[1, M]`. The criterion's curvature `10⁶` is the one the box's width
    /// resolves: across the sliver it moves by `½`, so the weights are not
    /// flat to rounding.
    #[test]
    fn a_domain_holding_a_sliver_of_the_proposal_is_sampled_exactly_3010() {
        let rho_hat = array![0.0];
        let h = array![[1.0e-4]];
        let (lower, upper) = (array![-1.0e-3], array![1.0e-3]);
        let seen = std::cell::RefCell::new(Vec::new());
        let target_h = array![[1.0e6]];
        let crit = recording_quartic(&rho_hat, &target_h, &seen);
        let graded =
            rho_posterior_adequacy(&rho_hat, &h, &(lower.clone(), upper.clone()), &[], crit, Some(64))
                .expect("diagnostic formed")
                .expect("diagnostic present");
        let seen = seen.into_inner();
        assert_eq!(seen.len(), 1 + graded.n_samples);
        assert!(seen.iter().all(|rho| rho[0] >= lower[0] && rho[0] <= upper[0]));
        let m = graded.n_samples as f64;
        assert!(graded.effective_sample_size >= 1.0 && graded.effective_sample_size <= m);
    }

    /// The proposal's density is the one it reports. For a draw `ρ` with
    /// `z = U⁻¹(ρ_f − ρ̂_f)`, the importance weight of the untruncated Gaussian
    /// target `exp(−½‖z‖²)` against the proposal is `∏_a p_a`, so its mean over
    /// draws is the Gaussian probability of the box. With a correlation of
    /// `−0.8` the sequential intervals move with the earlier draws, and the
    /// probability is the bivariate normal rectangle. The band is four
    /// standard errors of the sample mean, read from the sample itself.
    #[test]
    fn the_sequential_proposal_weights_integrate_to_the_box_probability_3010() {
        let rho_hat = array![0.3, -0.4];
        let (s0, s1, r) = (2.0_f64, 0.5_f64, -0.8_f64);
        let cov = array![[s0 * s0, r * s0 * s1], [r * s0 * s1, s1 * s1]];
        let det = cov[[0, 0]] * cov[[1, 1]] - cov[[0, 1]] * cov[[1, 0]];
        let h = array![
            [cov[[1, 1]] / det, -cov[[0, 1]] / det],
            [-cov[[1, 0]] / det, cov[[0, 0]] / det]
        ];
        let (lower, upper) = (array![-1.0, -0.6], array![2.5, 0.2]);
        let proposal =
            DomainLaplaceProposal::new(&rho_hat, &h, &(lower.clone(), upper.clone()), &[])
                .expect("proposal");
        let n = 20_000;
        let mut rng = DetNormal::new(ADEQUACY_SEED);
        let weights: Vec<f64> = proposal
            .sample(n, &mut rng)
            .expect("draws")
            .into_iter()
            .map(|(rho, neg_log_q)| {
                assert!((0..2).all(|i| rho[i] >= lower[i] && rho[i] <= upper[i]));
                let d = &rho - &rho_hat;
                (neg_log_q - 0.5 * d.dot(&h.dot(&d))).exp()
            })
            .collect();
        let mean = weights.iter().sum::<f64>() / n as f64;
        let var = weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_err = (var / n as f64).sqrt();
        let std = |v: f64, i: usize| (v - rho_hat[i]) / [s0, s1][i];
        let cdf = |a: f64, b: f64| {
            gam_math::bivariate_normal::bivariate_normal_cdf(a, b, r).expect("bivariate CDF").value
        };
        let (a0, b0, a1, b1) =
            (std(lower[0], 0), std(upper[0], 0), std(lower[1], 1), std(upper[1], 1));
        let exact = cdf(b0, b1) - cdf(a0, b1) - cdf(b0, a1) + cdf(a0, a1);
        assert!(
            (mean - exact).abs() <= 4.0 * std_err,
            "mean weight {mean} vs box probability {exact} (s.e. {std_err})"
        );
    }

    /// The truncated standard normal inverts its own CDF: at `u` the draw
    /// sits at `Φ(a) + u(Φ(b) − Φ(a))`, and intervals deep in either tail,
    /// where `Φ(b) − Φ(a)` underflows in probability space, still return a
    /// point inside the interval with a finite log mass.
    #[test]
    fn truncated_standard_normal_inverts_its_cdf_in_both_tails() {
        let (a, b) = (-1.0, 2.0);
        let log_mass = log_standard_normal_interval_mass(a, b);
        let mass = gam_math::probability::normal_cdf(b) - gam_math::probability::normal_cdf(a);
        assert!((log_mass - mass.ln()).abs() <= 16.0 * f64::EPSILON);
        for u in [1.0e-9, 0.25, 0.5, 0.75, 1.0 - 1.0e-9] {
            let x = truncated_standard_normal(a, b, log_mass, u);
            let target = gam_math::probability::normal_cdf(a) + u * mass;
            assert!((gam_math::probability::normal_cdf(x) - target).abs() <= 64.0 * f64::EPSILON);
        }
        for (a, b) in [(40.0, 40.5), (-41.0, -40.0), (-1.0e-300, 1.0e-300), (38.0, f64::INFINITY)] {
            let log_mass = log_standard_normal_interval_mass(a, b);
            assert!(log_mass.is_finite(), "[{a}, {b}] has log mass {log_mass}");
            for u in [1.0e-12, 0.5, 1.0 - 1.0e-12] {
                let x = truncated_standard_normal(a, b, log_mass, u);
                assert!(x >= a && x <= b, "draw {x} outside [{a}, {b}]");
            }
        }
    }

    /// #3010: with every coordinate on a face of its domain there is no free
    /// direction to sample and nothing to grade; the criterion is never called.
    #[test]
    fn every_coordinate_on_a_face_has_nothing_to_grade_3010() {
        let rho_hat = array![-1.0, 4.0];
        let h = array![[1.0e-6, 0.0], [0.0, 1.0e-6]];
        let seen = std::cell::RefCell::new(Vec::new());
        let crit = recording_quartic(&rho_hat, &h, &seen);
        let graded =
            rho_posterior_adequacy(&rho_hat, &h, &(array![-1.0, -4.0], array![1.0, 4.0]), &[], crit, None);
        assert!(matches!(graded, Ok(None)));
        assert!(seen.into_inner().is_empty());
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
        let refusal = rho_posterior_adequacy(&rho_hat, &h, &unbounded(), &[], |_| Ok(0.0), Some(64))
            .expect_err("a singular outer Hessian must be refused");
        assert!(
            matches!(refusal, RhoPosteriorRefusal::HessianNotPositiveDefinite { .. }),
            "the refusal names its reason, got {refusal}"
        );
    }
}

#[cfg(test)]
mod rho_posterior_escalation_tiers_tests;
