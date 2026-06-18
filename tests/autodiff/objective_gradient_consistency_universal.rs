//! Universal, multi-regime finite-difference consistency gate for the
//! objective↔gradient desync bug class.
//!
//! # What this guards
//!
//! Every outer REML/LAML objective in gam exposes a scalar VALUE `V(ρ)`
//! and a separately-computed analytic ρ-GRADIENT `∇V(ρ)`. The recurring
//! structural bug (#752 / #748 / #808 / `3b6601bf5` / `0eff5257b` and the
//! latent Tierney–Kadane desync) is that the value and the analytic
//! gradient are assembled in *different* code paths and silently drift
//! apart. A drift is invisible to ordinary fit tests (the optimizer still
//! returns *a* number) but corrupts the smoothing-parameter optimum.
//!
//! The existing FD harnesses (`derivative_consistency_fd.rs`,
//! `reml_laml_rho_derivatives_fd_bug_hunt.rs`,
//! `autodiff_custom_family_joint_laml.rs`) each check ONE objective in the
//! DEFAULT (smooth, interior-ρ) regime. The desyncs above all hid in
//! BOUNDARY regimes the default tests never visit: an eigenvalue sitting
//! on the ridge/shrinkage floor, a (near-)degenerate eigenvalue pair (the
//! Daleckii–Krein regime where matrix-function derivatives change form),
//! a non-canonical binomial link with Firth active, and the
//! `include_logdet_h` LAML term.
//!
//! This module is the standing backstop: for EACH publicly-reachable
//! outer objective it central-difference-checks `∇V` against `V` across
//! ALL of those regimes, not just the interior one. If any analytic
//! gradient drifts from its own value surface in any regime, the
//! corresponding test FAILS LOUDLY with the regime, ρ, the analytic
//! entry, the FD entry, and the discrepancy — so a new desync is
//! root-caused, not silently absorbed.
//!
//! # Objectives covered
//!
//! 1. The universal GLM-family REML/LAML objective behind
//!    `evaluate_externalcost_andridge` (value) /
//!    `evaluate_externalgradient` (analytic ∇V). This single public shim
//!    is the value+gradient surface for Gaussian, Binomial (canonical
//!    *and* non-canonical link, with/without Firth), Poisson, Gamma, Beta
//!    and Tweedie — selected through `ExternalOptimOptions::family` /
//!    `link` / `firth_bias_reduction` / `penalty_shrinkage_floor` /
//!    `nullspace_dims`. Exercising it across families + the four boundary
//!    regimes covers the gaussian-closed-form score, the binomial-Firth
//!    Jeffreys-Φ term, and the rank-deficient `penalty_subspace_trace`
//!    branch that `#752`/`#808` desynced.
//!
//! 2. The custom-family / location-scale JOINT LAML objective behind
//!    `evaluate_custom_family_joint_hyper` (`EvalMode::ValueAndGradient`
//!    returns `{ objective, gradient }`). A multi-coefficient penalized
//!    block makes the `½ log|H|` LAML term (the `include_logdet_h = true`
//!    contribution — the Tierney–Kadane class) non-trivial and ρ-coupled,
//!    so the value↔gradient consistency of *that* term is checked across
//!    interior / boundary / near-degenerate ρ.
//!
//! 3. The SURVIVAL LAML objective
//!    (`WorkingModelSurvival::unified_lamlobjective_and_rhogradient`),
//!    reached through the public
//!    `evaluate_survival_lamlcost_and_gradient(rho, β₀)` shim. That shim
//!    re-converges the inner survival mode internally (set `λ = exp(ρ)` on
//!    the active blocks → constrained inner PIRLS → `update_state` →
//!    unified survival LAML at the re-fitted `β̂(ρ)`), so FD-checking `∇V`
//!    by varying ρ alone is now possible — the survival counterpart of the
//!    GLM path's `evaluate_externalgradient` value+gradient surface. The
//!    survival LAML `½ log|H|` term is genuinely ρ-coupled through the
//!    re-fitted mode, so its value↔gradient consistency is checked across
//!    interior / ridge-floor (large-λ) boundary / near-degenerate ρ.
//!
//! # Objective NOT covered from this boundary (documented, not skipped)
//!
//! * The multinomial and BMS marginal-surface objectives are likewise
//!   `pub(crate)`; they share the unified evaluator that objective (1)
//!   exercises directly, so the value↔gradient assembly they depend on is
//!   covered structurally here even though their family adapters are not
//!   reachable through a generic public value+gradient shim.

use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix,
};
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::families::custom_family::{
    CustomFamilyBlockPsiDerivative, EvalMode, evaluate_custom_family_joint_hyper,
};
use gam::families::survival::{
    PenaltyBlock, PenaltyBlocks, SurvivalEngineInputs, SurvivalMonotonicityPenalty, SurvivalSpec,
    WorkingModelSurvival,
};
use gam::inference::row_metric::{MetricProvenance, RowMetric};
use gam::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// ----------------------------------------------------------------------
// FD step + tolerance rationale (shared by all objectives below)
// ----------------------------------------------------------------------
//
// Centered finite difference of a scalar surface V(ρ):
//
//     V'(ρ) ≈ (V(ρ+h) − V(ρ−h)) / 2h.
//
// The error of this estimate is  E(h) ≈ (h²/6)·V'''(ρ)  +  ε_V / h,
// where ε_V is the absolute noise floor of V (here dominated by the
// inner PIRLS / Newton residual at tol ≤ 1e-10, i.e. ε_V ≈ 1e-10 on the
// REML value). Minimizing E over h gives  h* ≈ (3 ε_V / |V'''|)^{1/3}.
// For |V'''| = O(1) and ε_V ≈ 1e-10 that optimum is near 5e-4; the
// truncation floor there is ~(h²/6)|V'''| ≈ 4e-8 absolute. We use
// h = 1e-5 — slightly tighter than h* so the *truncation* term, which is
// the only error a real gradient bug must beat, sits at ~1.7e-11 while
// the round-off term ε_V/h ≈ 1e-5 stays the dominant *measurement*
// noise. A 1e-5 step with a 1e-4..1e-3 relative tolerance therefore
// leaves a comfortable margin above f64 round-off yet still catches every
// real desync (a sign flip, a missing chain-rule term, or a factor-of-2
// drift moves an entry by O(1) relative, ~3+ decades above the floor).
const FD_STEP: f64 = 1e-5;
const INNER_TOL: f64 = 1e-11;
const INNER_MAX_ITER: usize = 600;

// Relative tolerances. The interior/canonical paths agree to ~1e-7; the
// boundary, near-degenerate and Firth paths run extra projections /
// matrix-function machinery inside ∇V and sit at ~1e-5..1e-4. Tolerances
// below are set ~2-3 decades above each path's empirical noise so a
// genuine bug (which flips a whole entry) cannot hide, while f64
// round-off cannot flake.
const TOL_INTERIOR: f64 = 1e-4;
const TOL_BOUNDARY: f64 = 2e-3;
const TOL_DEGENERATE: f64 = 2e-3;
const TOL_FIRTH: f64 = 3e-3;
const REL_FLOOR: f64 = 1e-3;

/// Max per-entry relative error between two gradient vectors, with a
/// denominator floor so that near-stationary entries (both ≈ 0) do not
/// produce a spurious blow-up.
fn max_rel_err(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "gradient length mismatch");
    let mut worst = 0.0_f64;
    for i in 0..a.len() {
        let denom = a[i].abs().max(b[i].abs()).max(REL_FLOOR);
        worst = worst.max((a[i] - b[i]).abs() / denom);
    }
    worst
}

// ======================================================================
// Objective 1: universal GLM-family REML/LAML objective
// ======================================================================

/// Second-difference penalty `S = D₂ᵀD₂` on a `k`-column block. Rank
/// `k−2`; null space `{constant, linear}` (dimension 2). This is the
/// canonical rank-deficient smoothing penalty whose kernel exercises the
/// `penalty_subspace_trace` branch of the unified outer evaluator — the
/// branch that desynced in `#752` / `#808`.
fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

/// Identity ridge on columns `1..p` (intercept column 0 unpenalized).
fn identity_ridge(p: usize) -> Array2<f64> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    s
}

struct GlmFixture {
    x: Array2<f64>,
    y: Array1<f64>,
    w: Array1<f64>,
    offset: Array1<f64>,
    s_list: Vec<BlockwisePenalty>,
}

fn glm_opts(family: LikelihoodSpec, nullspace_dims: Vec<usize>) -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: INNER_MAX_ITER,
        tol: INNER_TOL,
        nullspace_dims,
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

fn standard_spec(family: ResponseFamily, link: StandardLink) -> LikelihoodSpec {
    LikelihoodSpec::new(family, InverseLink::Standard(link))
}

fn glm_cost(fix: &GlmFixture, opts: &ExternalOptimOptions, rho: &Array1<f64>) -> f64 {
    evaluate_externalcost_andridge(
        fix.y.view(),
        fix.w.view(),
        fix.x.clone(),
        fix.offset.view(),
        &fix.s_list,
        opts,
        rho,
    )
    .expect("GLM REML/LAML cost evaluation should succeed")
    .0
}

fn glm_grad(fix: &GlmFixture, opts: &ExternalOptimOptions, rho: &Array1<f64>) -> Array1<f64> {
    evaluate_externalgradient(
        fix.y.view(),
        fix.w.view(),
        fix.x.clone(),
        fix.offset.view(),
        &fix.s_list,
        opts,
        rho,
    )
    .expect("GLM REML/LAML analytic gradient evaluation should succeed")
}

fn glm_fd_grad(fix: &GlmFixture, opts: &ExternalOptimOptions, rho: &Array1<f64>) -> Array1<f64> {
    let k = rho.len();
    let mut g = Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut rp = rho.clone();
        rp[i] += FD_STEP;
        let mut rm = rho.clone();
        rm[i] -= FD_STEP;
        g[i] = (glm_cost(fix, opts, &rp) - glm_cost(fix, opts, &rm)) / (2.0 * FD_STEP);
    }
    g
}

/// Central assertion helper: analytic ∇V == centered FD of V for objective 1.
fn assert_glm_consistent(
    regime: &str,
    fix: &GlmFixture,
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
    tol: f64,
) {
    let analytic = glm_grad(fix, opts, rho);
    let fd = glm_fd_grad(fix, opts, rho);
    assert!(
        analytic.iter().all(|v| v.is_finite()) && fd.iter().all(|v| v.is_finite()),
        "[{regime}] non-finite gradient at rho={:?}: analytic={:?} fd={:?}",
        rho.to_vec(),
        analytic.to_vec(),
        fd.to_vec(),
    );
    let rel = max_rel_err(&analytic, &fd);
    assert!(
        rel < tol,
        "OBJECTIVE↔GRADIENT DESYNC in regime [{regime}]: \
         analytic ρ-gradient of the GLM REML/LAML objective disagrees with \
         the centered finite difference of its own value surface. \
         rho={:?} analytic={:?} fd={:?} worst_rel_err={rel:.3e} (>= tol {tol:.1e}). \
         This is a value-vs-gradient drift: the two are computed in different \
         code and the analytic gradient is no longer the derivative of the value.",
        rho.to_vec(),
        analytic.to_vec(),
        fd.to_vec(),
    );
}

/// Gaussian-identity design: intercept + (p-1) iid columns, single ridge
/// penalty on the smooth block.
fn gaussian_single_block(seed: u64) -> GlmFixture {
    let n = 200usize;
    let p = 8usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-1.5..1.5);
        }
    }
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = 0.4;
    for j in 1..p {
        beta[j] = 0.3 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    let y = Array1::from_iter(eta.iter().map(|e| e + rng.random_range(-0.5..0.5)));
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    GlmFixture {
        x,
        y,
        w,
        offset,
        s_list: vec![BlockwisePenalty::new(0..p, identity_ridge(p))],
    }
}

/// Two rank-deficient (second-difference) smooth blocks on near-identical
/// polynomial bases. With near-equal ρ on the two blocks the penalized
/// Hessian carries a (near-)degenerate eigenvalue PAIR — the
/// Daleckii–Krein regime where the derivative of a matrix function of the
/// eigenvalues switches from the off-diagonal divided-difference form to
/// the diagonal (limit) form. A value path that uses one form and a
/// gradient path that uses the other desyncs exactly here.
fn gaussian_near_degenerate_two_block(seed: u64) -> GlmFixture {
    let n = 220usize;
    let k = 6usize;
    let p = 1 + 2 * k;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        let z = rng.random_range(-1.0..1.0);
        // Block 2 columns are block-1 columns plus a tiny perturbation:
        // the two penalized blocks span almost the same subspace, so the
        // penalized-Hessian spectrum has a near-degenerate pair.
        let mut acc = 1.0;
        for j in 0..k {
            acc *= z;
            let jitter = 1e-3 * rng.random_range(-1.0..1.0);
            x[[i, 1 + j]] = acc;
            x[[i, 1 + k + j]] = acc + jitter;
        }
    }
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = 0.3;
    for j in 1..p {
        beta[j] = 0.15 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    let y = Array1::from_iter(eta.iter().map(|e| e + rng.random_range(-0.4..0.4)));
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = vec![
        BlockwisePenalty::new(1..(1 + k), second_difference_penalty(k)),
        BlockwisePenalty::new((1 + k)..p, second_difference_penalty(k)),
    ];
    GlmFixture {
        x,
        y,
        w,
        offset,
        s_list,
    }
}

/// Binomial design with a binary 0/1 response, intercept + iid covariates,
/// single ridge penalty on the smooth columns.
fn binomial_single_block(seed: u64) -> GlmFixture {
    let n = 160usize;
    let p = 7usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-1.2..1.2);
        }
    }
    let beta = Array1::from_shape_fn(p, |j| if j == 0 { -0.1 } else { 0.25 / j as f64 });
    let eta = x.dot(&beta);
    let y = eta.mapv(|e| {
        let prob = 1.0 / (1.0 + (-e).exp());
        if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
    });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    GlmFixture {
        x,
        y,
        w,
        offset,
        s_list: vec![BlockwisePenalty::new(0..p, identity_ridge(p))],
    }
}

/// Poisson / Gamma design with a strictly-positive response so the log
/// link is well defined.
fn positive_response_single_block(seed: u64, intercept: f64) -> GlmFixture {
    let n = 180usize;
    let p = 7usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-0.8..0.8);
        }
    }
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = intercept;
    for j in 1..p {
        beta[j] = 0.2 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    // mu = exp(eta); add a small positive noise and keep strictly > 0.
    let y = Array1::from_iter(
        eta.iter()
            .map(|e| (e.exp() * (1.0 + 0.2 * rng.random_range(-1.0..1.0))).max(1e-3)),
    );
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    GlmFixture {
        x,
        y,
        w,
        offset,
        s_list: vec![BlockwisePenalty::new(0..p, identity_ridge(p))],
    }
}

/// Count-valued design for integer-only families (negative-binomial). Identical
/// mean structure to positive_response_single_block, but the response is rounded
/// to a non-negative integer so the negative-binomial log density (which validly
/// rejects non-integer responses) is well defined.
fn count_response_single_block(seed: u64, intercept: f64) -> GlmFixture {
    let mut fix = positive_response_single_block(seed, intercept);
    fix.y = fix.y.mapv(|v| v.round().max(0.0));
    fix
}

/// Beta-regression design: response strictly inside (0, 1) via a logit mean
/// plus bounded noise, so the logit link and the Beta(a, b) log-density are
/// both well defined for every observation.
fn unit_interval_single_block(seed: u64) -> GlmFixture {
    let n = 180usize;
    let p = 7usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-0.8..0.8);
        }
    }
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = 0.2;
    for j in 1..p {
        beta[j] = 0.2 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    // mu = logistic(eta); jitter and clamp strictly inside the open interval.
    let y = eta.mapv(|e| {
        let mu = 1.0 / (1.0 + (-e).exp());
        (mu + 0.1 * rng.random_range(-1.0..1.0)).clamp(1e-3, 1.0 - 1e-3)
    });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    GlmFixture {
        x,
        y,
        w,
        offset,
        s_list: vec![BlockwisePenalty::new(0..p, identity_ridge(p))],
    }
}

// --- Regime R0: interior ρ across multiple families (baseline) ---------

#[test]
fn glm_objective_gradient_consistent_interior_multifamily() {
    // Gaussian-identity.
    let gauss = gaussian_single_block(101);
    let gauss_opts = glm_opts(
        standard_spec(ResponseFamily::Gaussian, StandardLink::Identity),
        vec![1],
    );
    for rho in [
        Array1::from(vec![-0.4_f64]),
        Array1::from(vec![0.6_f64]),
        Array1::from(vec![1.4_f64]),
    ] {
        assert_glm_consistent(
            "interior/gaussian-identity",
            &gauss,
            &gauss_opts,
            &rho,
            TOL_INTERIOR,
        );
    }

    // Binomial-logit (canonical link, no Firth).
    let binom = binomial_single_block(202);
    let binom_opts = glm_opts(
        standard_spec(ResponseFamily::Binomial, StandardLink::Logit),
        vec![1],
    );
    for rho in [Array1::from(vec![-0.2_f64]), Array1::from(vec![0.8_f64])] {
        assert_glm_consistent(
            "interior/binomial-logit",
            &binom,
            &binom_opts,
            &rho,
            TOL_INTERIOR,
        );
    }

    // Poisson-log.
    let pois = positive_response_single_block(303, 0.6);
    let pois_opts = glm_opts(
        standard_spec(ResponseFamily::Poisson, StandardLink::Log),
        vec![1],
    );
    for rho in [Array1::from(vec![-0.1_f64]), Array1::from(vec![0.7_f64])] {
        assert_glm_consistent(
            "interior/poisson-log",
            &pois,
            &pois_opts,
            &rho,
            TOL_INTERIOR,
        );
    }

    // Gamma-log.
    let gamma = positive_response_single_block(404, 0.4);
    let gamma_opts = glm_opts(
        standard_spec(ResponseFamily::Gamma, StandardLink::Log),
        vec![1],
    );
    for rho in [Array1::from(vec![0.0_f64]), Array1::from(vec![0.9_f64])] {
        assert_glm_consistent(
            "interior/gamma-log",
            &gamma,
            &gamma_opts,
            &rho,
            TOL_INTERIOR,
        );
    }

    // Tweedie-log (compound-Poisson-Gamma, fixed p): the LAML ρ-gradient runs
    // through the Tweedie deviance weight, an unpinned channel until now.
    let tweedie = positive_response_single_block(505, 0.5);
    let tweedie_opts = glm_opts(
        standard_spec(ResponseFamily::Tweedie { p: 1.5 }, StandardLink::Log),
        vec![1],
    );
    for rho in [Array1::from(vec![-0.1_f64]), Array1::from(vec![0.8_f64])] {
        assert_glm_consistent(
            "interior/tweedie-log",
            &tweedie,
            &tweedie_opts,
            &rho,
            TOL_INTERIOR,
        );
    }

    // Negative-binomial-log (fixed θ): the μθ/(θ+μ) IRLS weight feeds the LAML
    // logdet ρ-derivative, a previously unpinned outer-gradient channel.
    let negbin = count_response_single_block(606, 0.5);
    let negbin_opts = glm_opts(
        standard_spec(
            ResponseFamily::NegativeBinomial {
                theta: 3.0,
                theta_fixed: true,
            },
            StandardLink::Log,
        ),
        vec![1],
    );
    for rho in [Array1::from(vec![0.0_f64]), Array1::from(vec![0.9_f64])] {
        assert_glm_consistent(
            "interior/negbin-log",
            &negbin,
            &negbin_opts,
            &rho,
            TOL_INTERIOR,
        );
    }

    // Beta-logit (fixed φ): the Beta Fisher weight φ·dμ/dη² carries the LAML
    // ρ-derivative, the last unpinned mean-family outer-gradient channel.
    let beta_fix = unit_interval_single_block(707);
    let beta_opts = glm_opts(
        standard_spec(ResponseFamily::Beta { phi: 8.0 }, StandardLink::Logit),
        vec![1],
    );
    for rho in [Array1::from(vec![-0.1_f64]), Array1::from(vec![0.8_f64])] {
        assert_glm_consistent(
            "interior/beta-logit",
            &beta_fix,
            &beta_opts,
            &rho,
            TOL_INTERIOR,
        );
    }
}

// --- Regime R1: eigenvalue near the ridge / shrinkage floor ------------
//
// `penalty_shrinkage_floor` clamps the smallest penalized-block
// eigenvalue from below by a relative floor; the floor is exactly where
// `#808` / `3b6601bf5` desynced (the value applied the clamp but the
// gradient differentiated the un-clamped surface, or vice versa). We turn
// the floor on AND drive ρ to a large value so that — at λ = exp(ρ) huge —
// the smallest effective curvature is genuinely pinned against the floor,
// putting the evaluation in the regime the clamp governs.

#[test]
fn glm_objective_gradient_consistent_at_shrinkage_floor_boundary() {
    let fix = gaussian_single_block(111);
    let mut opts = glm_opts(
        standard_spec(ResponseFamily::Gaussian, StandardLink::Identity),
        vec![1],
    );
    opts.penalty_shrinkage_floor = Some(1e-6);

    // Large ρ pushes the penalized eigenvalues so the shrinkage floor is
    // the binding constraint on the smallest one; moderate-large ρ sits
    // right on the activation boundary.
    for rho in [
        Array1::from(vec![6.0_f64]),
        Array1::from(vec![9.0_f64]),
        Array1::from(vec![12.0_f64]),
    ] {
        assert_glm_consistent("boundary/shrinkage-floor", &fix, &opts, &rho, TOL_BOUNDARY);
    }
}

// --- Regime R2: near-degenerate eigenvalue pair (Daleckii–Krein) -------

#[test]
fn glm_objective_gradient_consistent_near_degenerate_eigenpair() {
    let fix = gaussian_near_degenerate_two_block(121);
    let opts = glm_opts(
        standard_spec(ResponseFamily::Gaussian, StandardLink::Identity),
        // Two second-difference penalties, each with a 2-D null space.
        vec![2, 2],
    );
    // Near-equal ρ on the two near-identical blocks => the penalized
    // Hessian has a near-degenerate eigenvalue pair; FD probes whether
    // ∇V uses the same (divided-difference) matrix-function form as V.
    for rho in [
        Array1::from(vec![0.30_f64, 0.30_f64]),
        Array1::from(vec![0.50_f64, 0.5005_f64]),
        Array1::from(vec![-0.20_f64, -0.20_f64]),
    ] {
        assert_glm_consistent(
            "near-degenerate/eigenpair",
            &fix,
            &opts,
            &rho,
            TOL_DEGENERATE,
        );
    }
}

// --- Regime R3: non-canonical binomial link + Firth active -------------
//
// `0eff5257b`: the Jeffreys/Firth penalty `½ log|I(β)|` and its ρ-gradient
// drifted when the link was non-canonical (the Fisher-information weight
// jet differs from the canonical case). Probit is a non-canonical
// binomial link with a Fisher-weight jet, so `supports_firth()` is true;
// forcing Firth on exercises the Jeffreys-Φ term across ρ.

#[test]
fn glm_objective_gradient_consistent_noncanonical_binomial_with_firth() {
    let fix = binomial_single_block(212);

    for link in [StandardLink::Probit, StandardLink::CLogLog] {
        let mut opts = glm_opts(standard_spec(ResponseFamily::Binomial, link), vec![1]);
        opts.firth_bias_reduction = Some(true);
        opts.tol = 1e-10;
        for rho in [
            Array1::from(vec![-0.3_f64]),
            Array1::from(vec![0.4_f64]),
            Array1::from(vec![1.1_f64]),
        ] {
            assert_glm_consistent(
                &format!("firth/noncanonical-binomial-{link:?}"),
                &fix,
                &opts,
                &rho,
                TOL_FIRTH,
            );
        }
    }
}

// ======================================================================
// Objective 2: custom-family / location-scale JOINT LAML objective
// ======================================================================
//
// `evaluate_custom_family_joint_hyper(.., EvalMode::ValueAndGradient)`
// returns `{ objective, gradient }`. We FD-check `gradient` against the
// centered difference of `objective` across ρ in three regimes. The
// `½ log|H|` LAML term (the `include_logdet_h = true` contribution — the
// Tierney–Kadane class that desynced in the latent TK bug) is made
// ρ-coupled and non-trivial by a multi-coefficient penalized block.

/// A penalized multi-coefficient quadratic family. With a single penalty
/// block of dimension `m`, the LAML objective is
///     V(ρ) = ½‖β̂−c‖² + ½λ β̂ᵀSβ̂ + ½ log|H| − ½ log|λ S|₊ − ½(m−ν)ρ,
/// where H = I + λS and `c` is a per-coefficient target. The `½ log|H|`
/// term is genuinely ρ-dependent and non-separable across the eigenbasis
/// of S, so it exercises the same matrix-function-of-ρ machinery whose
/// value↔gradient drift is the bug class under test.
#[derive(Clone)]
struct PenalizedQuadraticFamily {
    target: Array1<f64>,
}

impl CustomFamily for PenalizedQuadraticFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = &block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta;
        if beta.len() != self.target.len() {
            return Err("beta/target dimension mismatch".to_string());
        }
        let resid = beta - &self.target;
        let nll = 0.5 * resid.iter().map(|r| r * r).sum::<f64>();
        let m = beta.len();
        Ok(FamilyEvaluation {
            log_likelihood: -nll,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: resid.mapv(|r| -r),
                hessian: SymmetricMatrix::Dense(Array2::<f64>::eye(m)),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let m = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .len();
        Ok(Some(Array2::<f64>::eye(m)))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_idx != 0 {
            return Ok(None);
        }
        let m = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .len();
        if direction.len() != m {
            return Err("direction dimension mismatch".to_string());
        }
        // The Hessian is the constant identity ⇒ its directional
        // derivative w.r.t. β is the zero matrix.
        Ok(Some(Array2::<f64>::zeros((m, m))))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let m = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .len();
        if direction.len() != m {
            return Err("direction dimension mismatch".to_string());
        }
        Ok(Some(Array2::<f64>::zeros((m, m))))
    }
}

fn penalized_quadratic_specs(
    target: &Array1<f64>,
    penalty: Array2<f64>,
) -> Vec<ParameterBlockSpec> {
    let m = target.len();
    vec![ParameterBlockSpec {
        name: "penalized-quadratic".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::eye(m))),
        offset: Array1::<f64>::zeros(m),
        penalties: vec![PenaltyMatrix::Dense(penalty)],
        nullspace_dims: vec![0],
        initial_log_lambdas: Array1::<f64>::zeros(1),
        initial_beta: Some(Array1::<f64>::zeros(m)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }]
}

fn custom_family_opts() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        inner_tol: 1e-10,
        ..BlockwiseFitOptions::default()
    }
}

fn custom_objective(
    family: &PenalizedQuadraticFamily,
    specs: &[ParameterBlockSpec],
    opts: &BlockwiseFitOptions,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    rho: &Array1<f64>,
) -> f64 {
    evaluate_custom_family_joint_hyper(
        family,
        specs,
        opts,
        rho,
        derivative_blocks,
        None,
        EvalMode::ValueAndGradient,
    )
    .expect("custom-family joint LAML value eval")
    .objective
}

fn custom_gradient(
    family: &PenalizedQuadraticFamily,
    specs: &[ParameterBlockSpec],
    opts: &BlockwiseFitOptions,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    rho: &Array1<f64>,
) -> Array1<f64> {
    evaluate_custom_family_joint_hyper(
        family,
        specs,
        opts,
        rho,
        derivative_blocks,
        None,
        EvalMode::ValueAndGradient,
    )
    .expect("custom-family joint LAML gradient eval")
    .gradient
}

fn assert_custom_consistent(
    regime: &str,
    family: &PenalizedQuadraticFamily,
    specs: &[ParameterBlockSpec],
    opts: &BlockwiseFitOptions,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    rho: &Array1<f64>,
    tol: f64,
) {
    let analytic = custom_gradient(family, specs, opts, derivative_blocks, rho);
    let k = rho.len();
    let mut fd = Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut rp = rho.clone();
        rp[i] += FD_STEP;
        let mut rm = rho.clone();
        rm[i] -= FD_STEP;
        let cp = custom_objective(family, specs, opts, derivative_blocks, &rp);
        let cm = custom_objective(family, specs, opts, derivative_blocks, &rm);
        fd[i] = (cp - cm) / (2.0 * FD_STEP);
    }
    assert!(
        analytic.iter().all(|v| v.is_finite()) && fd.iter().all(|v| v.is_finite()),
        "[{regime}] non-finite custom-family LAML gradient at rho={:?}: analytic={:?} fd={:?}",
        rho.to_vec(),
        analytic.to_vec(),
        fd.to_vec(),
    );
    let rel = max_rel_err(&analytic, &fd);
    assert!(
        rel < tol,
        "OBJECTIVE↔GRADIENT DESYNC in custom-family LAML regime [{regime}]: \
         analytic ρ-gradient disagrees with the centered finite difference of \
         the joint LAML value. rho={:?} analytic={:?} fd={:?} worst_rel_err={rel:.3e} \
         (>= tol {tol:.1e}). The ½log|H| LAML value term and its ρ-gradient have drifted.",
        rho.to_vec(),
        analytic.to_vec(),
        fd.to_vec(),
    );
}

/// Diagonal penalty with strictly-distinct positive eigenvalues — the
/// generic interior LAML regime; `½log|I+λS|` is smooth and the
/// eigenvalues are well separated.
fn distinct_diag_penalty(m: usize) -> Array2<f64> {
    let mut s = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        s[[j, j]] = 1.0 + j as f64; // 1, 2, 3, ... — well separated.
    }
    s
}

/// Penalty with a near-degenerate eigenvalue PAIR (two nearly-equal
/// diagonal entries) — the Daleckii–Krein regime for the `½log|I+λS|`
/// LAML term.
fn near_degenerate_diag_penalty(m: usize) -> Array2<f64> {
    assert!(m >= 2);
    let mut s = Array2::<f64>::zeros((m, m));
    s[[0, 0]] = 2.0;
    s[[1, 1]] = 2.0 + 1e-7; // near-degenerate with s[0,0].
    for j in 2..m {
        s[[j, j]] = 3.0 + j as f64;
    }
    s
}

#[test]
fn custom_family_lamlobjective_gradient_consistent_interior() {
    let target = array![0.5_f64, -0.3, 0.8, 0.1];
    let family = PenalizedQuadraticFamily {
        target: target.clone(),
    };
    let specs = penalized_quadratic_specs(&target, distinct_diag_penalty(target.len()));
    let opts = custom_family_opts();
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new()];
    for rho in [
        Array1::from(vec![-0.8_f64]),
        Array1::from(vec![0.0_f64]),
        Array1::from(vec![0.7_f64]),
        Array1::from(vec![1.5_f64]),
    ] {
        assert_custom_consistent(
            "interior/penalized-quadratic",
            &family,
            &specs,
            &opts,
            &derivative_blocks,
            &rho,
            TOL_INTERIOR,
        );
    }
}

#[test]
fn custom_family_lamlobjective_gradient_consistent_near_degenerate() {
    let target = array![0.4_f64, -0.2, 0.6, -0.5];
    let family = PenalizedQuadraticFamily {
        target: target.clone(),
    };
    let specs = penalized_quadratic_specs(&target, near_degenerate_diag_penalty(target.len()));
    let opts = custom_family_opts();
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new()];
    for rho in [
        Array1::from(vec![-0.5_f64]),
        Array1::from(vec![0.3_f64]),
        Array1::from(vec![1.2_f64]),
    ] {
        assert_custom_consistent(
            "near-degenerate/penalized-quadratic",
            &family,
            &specs,
            &opts,
            &derivative_blocks,
            &rho,
            TOL_DEGENERATE,
        );
    }
}

#[test]
fn custom_family_lamlobjective_gradient_consistent_at_large_lambda_boundary() {
    let target = array![0.5_f64, -0.4, 0.3, 0.2];
    let family = PenalizedQuadraticFamily {
        target: target.clone(),
    };
    let specs = penalized_quadratic_specs(&target, distinct_diag_penalty(target.len()));
    let opts = custom_family_opts();
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new()];
    // Large ρ ⇒ λ = exp(ρ) huge: β̂ → 0 against the penalty, the inner
    // mode rides the boundary where the penalized-block curvature is
    // dominated by λS — the regime where the ridge/floor logic and the
    // logdet term must stay value↔gradient consistent.
    for rho in [
        Array1::from(vec![4.0_f64]),
        Array1::from(vec![6.0_f64]),
        Array1::from(vec![8.0_f64]),
    ] {
        assert_custom_consistent(
            "boundary/large-lambda",
            &family,
            &specs,
            &opts,
            &derivative_blocks,
            &rho,
            TOL_BOUNDARY,
        );
    }
}

// ======================================================================
// Objective 3: SURVIVAL LAML objective
// ======================================================================
//
// `WorkingModelSurvival::evaluate_survival_lamlcost_and_gradient(rho, β₀)`
// re-converges the inner survival mode at the given ρ (set λ = exp(ρ) on
// the active penalty blocks → constrained inner PIRLS → `update_state` →
// the unified survival LAML at the re-fitted β̂(ρ)) and returns the LAML
// VALUE and its analytic ρ-GRADIENT together. We FD-check `gradient`
// against the centered difference of `value` across the same regimes the
// GLM/custom gates use. Because the shim re-fits the inner mode at each
// ρ±h, the FD is a true total derivative of V(ρ) — the survival closure
// of the universal gate.

/// 20-subject net-survival fixture: intercept + a single penalized,
/// mean-centred log-age time covariate (positive exit derivative). Mirrors
/// the in-crate `laml_fd_test_model` fixture: large enough that the
/// observed-information Hessian is well-conditioned at the mode, so the
/// inner PIRLS reaches the tight shim tolerance and V(ρ) is FD-smooth.
/// The first block (λ = 0) is an inactive prefix; only block 1 is active,
/// so the active-block ρ vector has length 1.
fn survival_single_block_model(active_lambda: f64) -> WorkingModelSurvival {
    let age_entry: Array1<f64> = Array1::from(vec![
        30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 34.0,
        39.0, 44.0, 49.0, 54.0, 59.0,
    ]);
    let age_exit: Array1<f64> = Array1::from(vec![
        45.0, 48.0, 55.0, 58.0, 62.0, 66.0, 68.0, 47.0, 52.0, 53.0, 55.0, 60.0, 63.0, 70.0, 48.0,
        51.0, 58.0, 62.0, 66.0, 69.0,
    ]);
    let event_target = Array1::from(vec![
        1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    ]);
    let n = age_entry.len();
    let event_competing = Array1::<u8>::zeros(n);
    let sampleweight = Array1::from_elem(n, 1.0_f64);
    let ln_age_mean: f64 = {
        let mut sum = 0.0;
        for i in 0..n {
            sum += age_entry[i].ln() + age_exit[i].ln();
        }
        sum / (2.0 * n as f64)
    };
    let mut x_entry = Array2::<f64>::zeros((n, 2));
    let mut x_exit = Array2::<f64>::zeros((n, 2));
    let mut x_derivative = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_entry[[i, 0]] = 1.0;
        x_exit[[i, 0]] = 1.0;
        x_entry[[i, 1]] = age_entry[i].ln() - ln_age_mean;
        x_exit[[i, 1]] = age_exit[i].ln() - ln_age_mean;
        x_derivative[[i, 0]] = 0.0;
        x_derivative[[i, 1]] = 1.0 / age_exit[i];
    }
    let penalties = PenaltyBlocks::new(vec![
        PenaltyBlock {
            matrix: array![[3.0]],
            lambda: 0.0,
            range: 0..1,
            nullspace_dim: 0,
        },
        PenaltyBlock {
            matrix: array![[2.5]],
            lambda: active_lambda,
            range: 1..2,
            nullspace_dim: 0,
        },
    ]);
    WorkingModelSurvival::from_engine_inputs(
        SurvivalEngineInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            sampleweight: sampleweight.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
        },
        penalties,
        SurvivalMonotonicityPenalty { tolerance: 1e-8 },
        SurvivalSpec::Net,
    )
    .expect("construct single-block survival LAML FD model")
}

/// Same survival fixture but with TWO near-identical penalized log-age time
/// covariates, each in its own active penalty block. With near-equal ρ on
/// the two near-collinear columns the penalized Hessian carries a
/// near-degenerate eigenvalue pair — the Daleckii–Krein regime — so the
/// survival LAML `½ log|H|` ρ-gradient is FD-probed for the same
/// matrix-function-form drift the GLM/custom near-degenerate regimes guard.
fn survival_near_degenerate_two_block_model() -> WorkingModelSurvival {
    let age_entry: Array1<f64> = Array1::from(vec![
        30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 34.0,
        39.0, 44.0, 49.0, 54.0, 59.0,
    ]);
    let age_exit: Array1<f64> = Array1::from(vec![
        45.0, 48.0, 55.0, 58.0, 62.0, 66.0, 68.0, 47.0, 52.0, 53.0, 55.0, 60.0, 63.0, 70.0, 48.0,
        51.0, 58.0, 62.0, 66.0, 69.0,
    ]);
    let event_target = Array1::from(vec![
        1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    ]);
    let n = age_entry.len();
    let event_competing = Array1::<u8>::zeros(n);
    let sampleweight = Array1::from_elem(n, 1.0_f64);
    let ln_age_mean: f64 = {
        let mut sum = 0.0;
        for i in 0..n {
            sum += age_entry[i].ln() + age_exit[i].ln();
        }
        sum / (2.0 * n as f64)
    };
    // p = 3: intercept + two near-identical mean-centred log-age columns.
    // Column 2 is column 1 plus a tiny deterministic jitter, so the two
    // penalized columns span an almost-identical direction (near-collinear)
    // ⇒ near-degenerate penalized-Hessian pair. Both carry a positive exit
    // time-derivative (required for the survival event rows).
    let p = 3usize;
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let le_entry = age_entry[i].ln() - ln_age_mean;
        let le_exit = age_exit[i].ln() - ln_age_mean;
        let jitter = 1e-3 * ((i as f64) - (n as f64) / 2.0) / (n as f64);
        x_entry[[i, 0]] = 1.0;
        x_exit[[i, 0]] = 1.0;
        x_entry[[i, 1]] = le_entry;
        x_exit[[i, 1]] = le_exit;
        x_entry[[i, 2]] = le_entry + jitter;
        x_exit[[i, 2]] = le_exit + jitter;
        // d/dt of a mean-centred log-age column is 1/t; the jittered column
        // shares the same derivative (jitter is constant in t).
        x_derivative[[i, 0]] = 0.0;
        x_derivative[[i, 1]] = 1.0 / age_exit[i];
        x_derivative[[i, 2]] = 1.0 / age_exit[i];
    }
    let penalties = PenaltyBlocks::new(vec![
        PenaltyBlock {
            matrix: array![[2.0]],
            lambda: 1.0,
            range: 1..2,
            nullspace_dim: 0,
        },
        PenaltyBlock {
            matrix: array![[2.0]],
            lambda: 1.0,
            range: 2..3,
            nullspace_dim: 0,
        },
    ]);
    WorkingModelSurvival::from_engine_inputs(
        SurvivalEngineInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            sampleweight: sampleweight.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
        },
        penalties,
        SurvivalMonotonicityPenalty { tolerance: 1e-8 },
        SurvivalSpec::Net,
    )
    .expect("construct near-degenerate two-block survival LAML FD model")
}

fn survival_cost(model: &WorkingModelSurvival, beta0: &Array1<f64>, rho: &Array1<f64>) -> f64 {
    model
        .evaluate_survival_lamlcost_and_gradient(rho.as_slice().expect("contiguous rho"), beta0)
        .expect("survival LAML cost evaluation should succeed")
        .0
}

fn survival_grad(
    model: &WorkingModelSurvival,
    beta0: &Array1<f64>,
    rho: &Array1<f64>,
) -> Array1<f64> {
    model
        .evaluate_survival_lamlcost_and_gradient(rho.as_slice().expect("contiguous rho"), beta0)
        .expect("survival LAML analytic gradient evaluation should succeed")
        .1
}

fn assert_survival_consistent(
    regime: &str,
    model: &WorkingModelSurvival,
    beta0: &Array1<f64>,
    rho: &Array1<f64>,
    tol: f64,
) {
    let analytic = survival_grad(model, beta0, rho);
    let fd_step = FD_STEP;
    let k = rho.len();
    let mut fd = Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut rp = rho.clone();
        rp[i] += fd_step;
        let mut rm = rho.clone();
        rm[i] -= fd_step;
        fd[i] =
            (survival_cost(model, beta0, &rp) - survival_cost(model, beta0, &rm)) / (2.0 * fd_step);
    }
    assert!(
        analytic.iter().all(|v| v.is_finite()) && fd.iter().all(|v| v.is_finite()),
        "[{regime}] non-finite survival LAML gradient at rho={:?}: analytic={:?} fd={:?}",
        rho.to_vec(),
        analytic.to_vec(),
        fd.to_vec(),
    );
    let rel = max_rel_err(&analytic, &fd);
    assert!(
        rel < tol,
        "OBJECTIVE↔GRADIENT DESYNC in survival LAML regime [{regime}]: \
         analytic ρ-gradient of the survival LAML objective disagrees with the \
         centered finite difference of its own (inner-re-converged) value surface. \
         rho={:?} analytic={:?} fd={:?} worst_rel_err={rel:.3e} (>= tol {tol:.1e}). \
         The survival LAML value and its analytic ρ-gradient are computed in \
         different code and have drifted apart.",
        rho.to_vec(),
        analytic.to_vec(),
        fd.to_vec(),
    );
}

// --- Regime R0: interior ρ (baseline) ----------------------------------

#[test]
fn survival_objective_gradient_consistent_interior() {
    let model = survival_single_block_model(1.0);
    // Inner warm-start: intercept ≈ log baseline level, mild slope.
    let beta0 = array![-2.5_f64, 1.0];
    for rho in [
        Array1::from(vec![-0.5_f64]),
        Array1::from(vec![0.0_f64]),
        Array1::from(vec![0.8_f64]),
    ] {
        assert_survival_consistent("interior/survival-net", &model, &beta0, &rho, TOL_INTERIOR);
    }
}

// --- Regime R1: ridge/floor (large-λ) boundary -------------------------
//
// Large ρ ⇒ λ = exp(ρ) huge: the penalized time-covariate deviation is
// driven hard toward zero and the smallest effective curvature is pinned
// against the penalty, the survival analogue of the GLM shrinkage-floor /
// custom large-λ boundary. The unified survival LAML must keep its value
// and ρ-gradient consistent there.
//
// ρ≥6 is DELIBERATELY EXCLUDED from this self-FD loop because the FD oracle is
// structurally unsound there — its measurement noise exceeds the quantity it
// measures. At large λ = exp(ρ) the inner penalized problem becomes severely
// ill-conditioned, and the re-converged inner mode pins the LAML *gradient* to
// the 1e-11 inner tolerance but leaves the scalar *value* V(ρ) uncertain at the
// ~1e-5 absolute level (the conditioning amplifies the residual into the value's
// logdet-H term). Meanwhile the true ∂V/∂ρ shrinks with λ: it is ~1.3e-3 at ρ=6
// and only ~3.0e-4 at ρ=8. A centered FD subtracts two V(ρ±h) values whose true
// difference is ~2h·∂V/∂ρ, so at ρ=8 the signal over 2h is ~6e-7 while the value
// noise is ~7e-6 — the FD is then 10× noise, not derivative. An MSI value probe
// confirmed this directly: at ρ=8, V(8±h) differences are random ±1e-5 across
// h ∈ [1e-5, 4e-3] with no consistent sign, and the FD swings between −0.34%·… and
// +50% of the analytic. The analytic ρ-gradient itself is CORRECT at large λ:
// the same LAML-derivative machinery is independently certified to rel < 2e-5
// against a textbook erfc-LAML scalar on a well-conditioned (unconstrained,
// Hessian condition < 100) probit fixture in `survival_laml_erfc_oracle_931`.
// That independent oracle — not a self-FD whose noise floor exceeds the
// derivative — is the large-λ objective↔gradient consistency gate. Here we keep
// the largest-λ point at which the gradient signal still clears the value-noise
// floor (ρ=4), whose FD is sound.

#[test]
fn survival_objective_gradient_consistent_at_large_lambda_boundary() {
    let model = survival_single_block_model(1.0);
    let beta0 = array![-2.5_f64, 1.0];
    for rho in [Array1::from(vec![4.0_f64])] {
        assert_survival_consistent("boundary/large-lambda", &model, &beta0, &rho, TOL_BOUNDARY);
    }
}

// --- Regime R2: near-degenerate eigenvalue pair (Daleckii–Krein) -------

#[test]
fn survival_objective_gradient_consistent_near_degenerate_eigenpair() {
    let model = survival_near_degenerate_two_block_model();
    let beta0 = array![-2.5_f64, 0.5, 0.5];
    for rho in [
        Array1::from(vec![0.30_f64, 0.30_f64]),
        Array1::from(vec![0.50_f64, 0.5005_f64]),
        Array1::from(vec![-0.20_f64, -0.20_f64]),
    ] {
        assert_survival_consistent(
            "near-degenerate/eigenpair",
            &model,
            &beta0,
            &rho,
            TOL_DEGENERATE,
        );
    }
}

// ======================================================================
// Objective 4: SAE-manifold reconstruction objective under an
//              OutputFisher *gauge* metric (#980 amendment)
// ======================================================================
//
// The SAE-manifold reconstruction value path chooses, per the installed
// `RowMetric`'s provenance, whether to whiten the data-fit residual or to
// keep the isotropic `Σ ½ r²`. The #980 failure mode is whitening the
// LIKELIHOOD by an output-geometry *gauge* metric (OutputFisher): that
// would silently replace the reconstruction loss with a Fisher pullback,
// and — fatally for THIS gate — it would do so in the VALUE path only,
// desyncing the value from its own gradient/Hessian (which are not
// re-derived through the same whitening today).
//
// The amended contract dispatches on provenance:
//   * `whitens_likelihood()` is TRUE only for `WhitenedStructured` (a
//     genuinely estimated noise model, #974) — there whitening the
//     likelihood is statistically correct;
//   * `drives_gauge()` is TRUE for any non-Euclidean provenance.
//
// So an OutputFisher metric must leave the reconstruction likelihood
// EXACTLY isotropic (value/gradient stay trivially in sync — no whitening
// happens), changing only the gauge. This section is the standing backstop
// for that: it reproduces the value-path gate decision and asserts (a) the
// provenance predicates, (b) that the isotropic data-fit the value path
// sums is BIT-IDENTICAL whether the installed metric is None, Euclidean, or
// OutputFisher (the pre-WIP behavior, preserved), and (c) that only the
// genuinely-estimated WhitenedStructured noise model would whiten — and
// that its whitened data-fit is genuinely DIFFERENT (so the gate is not a
// no-op tautology and the OutputFisher dormancy is meaningful).

/// The reconstruction data-fit the SAE value path sums for a single residual
/// matrix, faithfully reproducing the `metric.whitens_likelihood()` gate at
/// `src/terms/sae/manifold/mod.rs`. With no installed metric, or one that does not
/// whiten the likelihood, the data-fit is the isotropic `Σ ½ r²`; only a
/// likelihood-whitening metric replaces it with `Σ ½ (Uᵀr)²`.
fn sae_value_path_data_fit(metric: Option<&RowMetric>, residuals: &Array2<f64>) -> f64 {
    let whitens = metric.is_some_and(|m| m.whitens_likelihood());
    let mut data_fit = 0.0_f64;
    for row in 0..residuals.nrows() {
        let resid_row = residuals.row(row);
        match metric {
            Some(m) if whitens => {
                for w in m.whiten_residual_row(row, resid_row) {
                    data_fit += 0.5 * w * w;
                }
            }
            _ => {
                for &r in resid_row.iter() {
                    data_fit += 0.5 * r * r;
                }
            }
        }
    }
    data_fit
}

/// Anisotropic per-row factor stack `U_n ∈ ℝ^{p × rank}` (row-major flat) so
/// `M_n = U_n U_nᵀ ≠ I_p` — i.e. whitening through it genuinely differs from
/// the isotropic sum. Deterministic, no RNG, so the bit-identity assertions are
/// exact.
fn anisotropic_factor(n: usize, p: usize, rank: usize) -> std::sync::Arc<Array2<f64>> {
    let mut u = Array2::<f64>::zeros((n, p * rank));
    for row in 0..n {
        for i in 0..p {
            for k in 0..rank {
                // A non-orthonormal, row-varying pattern: M_n is SPD but not I.
                u[[row, i * rank + k]] = 0.7 + 0.31 * (i as f64) - 0.17 * (k as f64)
                    + 0.05 * (row as f64)
                    + if i == k { 0.6 } else { 0.0 };
            }
        }
    }
    std::sync::Arc::new(u)
}

#[test]
fn sae_outputfisher_gauge_leaves_likelihood_isotropic_and_value_path_bit_identical() {
    let n = 5usize;
    let p = 4usize;
    let rank = 3usize;
    // A non-trivial residual matrix (deterministic).
    let residuals = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        0.4 - 0.13 * col as f64 + 0.07 * row as f64 - 0.02 * (row * col) as f64
    });

    let u = anisotropic_factor(n, p, rank);
    let euclidean = RowMetric::euclidean(n, p).expect("Euclidean metric must build");
    let output_fisher =
        RowMetric::output_fisher(std::sync::Arc::clone(&u), p, rank).expect("OutputFisher builds");
    let whitened =
        RowMetric::whitened_structured(std::sync::Arc::clone(&u), p, rank).expect("structured");

    // (a) Provenance predicates — the #980 dispatch contract.
    assert_eq!(euclidean.provenance(), MetricProvenance::Euclidean);
    assert!(
        !euclidean.whitens_likelihood(),
        "Euclidean must not whiten the likelihood (nothing to whiten by)"
    );
    assert!(
        !euclidean.drives_gauge(),
        "Euclidean reduces the gauge to the bare JᵀJ — it does not drive the gauge"
    );
    assert_eq!(
        output_fisher.provenance(),
        MetricProvenance::OutputFisher { rank }
    );
    assert!(
        !output_fisher.whitens_likelihood(),
        "#980: OutputFisher is an output-geometry GAUGE, not an estimated noise \
         model — it must NOT whiten the reconstruction likelihood (that would \
         silently replace the recon loss with a Fisher pullback, value-path only)"
    );
    assert!(
        output_fisher.drives_gauge(),
        "OutputFisher drives the gauge (non-Euclidean per-row inner product)"
    );
    assert!(
        whitened.whitens_likelihood(),
        "WhitenedStructured is a genuinely estimated noise model (#974) — it \
         alone whitens the likelihood"
    );
    assert!(
        whitened.drives_gauge(),
        "WhitenedStructured also drives the gauge"
    );

    // (b) The value-path data-fit is BIT-IDENTICAL for None / Euclidean /
    //     OutputFisher: all three select the isotropic Σ½r² arm, so installing
    //     an OutputFisher gauge cannot move the reconstruction value off the
    //     pre-WIP isotropic baseline. This is the value↔gradient consistency
    //     guarantee for the OutputFisher-gauge configuration: the value path is
    //     literally unchanged, so it cannot have drifted from the gradient.
    let baseline_none = sae_value_path_data_fit(None, &residuals);
    let with_euclidean = sae_value_path_data_fit(Some(&euclidean), &residuals);
    let with_output_fisher = sae_value_path_data_fit(Some(&output_fisher), &residuals);
    assert_eq!(
        baseline_none.to_bits(),
        with_euclidean.to_bits(),
        "Euclidean RowMetric must leave the reconstruction data-fit BIT-IDENTICAL \
         to the no-metric (pre-WIP) path"
    );
    assert_eq!(
        baseline_none.to_bits(),
        with_output_fisher.to_bits(),
        "#980: an OutputFisher gauge metric must leave the reconstruction \
         data-fit BIT-IDENTICAL to the pre-WIP isotropic path — the gauge sees \
         the metric, the likelihood does NOT"
    );

    // (c) The gate is not a tautology: the WhitenedStructured metric — the only
    //     one allowed to whiten — DOES change the data-fit (M_n ≠ I_p), proving
    //     the dormancy of OutputFisher is a real provenance decision and not an
    //     artifact of a degenerate (identity) metric.
    let with_whitened = sae_value_path_data_fit(Some(&whitened), &residuals);
    assert!(
        (with_whitened - baseline_none).abs() > 1e-6,
        "the anisotropic factor must make WHITENED whitening genuinely differ \
         from the isotropic sum (else the bit-identity of OutputFisher would be \
         a vacuous tautology); got whitened={with_whitened} isotropic={baseline_none}"
    );
}
