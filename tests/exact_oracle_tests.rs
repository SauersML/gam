use ndarray::{Array1, Array2, array};

use gam::estimate::{ExternalOptimOptions, evaluate_externalgradient};
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

#[derive(Clone, Copy, Debug)]
struct OracleConfig {
    grid_bound: f64,
    steps: usize,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            grid_bound: 8.0,
            steps: 241,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OracleEval {
    grad_rho: f64,
}

/// Deterministic 2D grid oracle for tiny Bernoulli-logit model.
///
/// Evidence:
///   L(ρ) = ∫ exp( l(β) - 0.5 λ ||β||² ) dβ,  λ=exp(ρ), S=I₂.
///
/// Exact gradient identity:
///   d/dρ log L(ρ) = -0.5 λ E_post[ ||β||² ].
///
/// We evaluate both by direct 2D quadrature over β=(β₁,β₂), using a stabilized
/// log-sum-exp style integral to avoid underflow.
fn exact_logit_oracle_eval_rho_2d(
    y: &Array1<f64>,
    x: &Array2<f64>,
    rho: f64,
    cfg: OracleConfig,
) -> OracleEval {
    assert_eq!(x.ncols(), 2);
    assert_eq!(x.nrows(), y.len());
    assert!(cfg.steps >= 5);

    let lambda = rho.exp();
    let h = (2.0 * cfg.grid_bound) / ((cfg.steps - 1) as f64);

    // First pass: locate max log-kernel for stable log-sum-exp integration.
    let mut max_logk = f64::NEG_INFINITY;
    for i in 0..cfg.steps {
        let b1 = -cfg.grid_bound + (i as f64) * h;
        for j in 0..cfg.steps {
            let b2 = -cfg.grid_bound + (j as f64) * h;
            let beta = array![b1, b2];
            let eta = x.dot(&beta);
            let ll = eta
                .iter()
                .zip(y.iter())
                .map(|(&e, &yy)| yy * e - softplus(e))
                .sum::<f64>();
            let pen = 0.5 * lambda * (b1 * b1 + b2 * b2);
            max_logk = max_logk.max(ll - pen);
        }
    }

    // Second pass: integrate denominator and quadratic numerator.
    let mut z = 0.0_f64;
    let mut q = 0.0_f64;
    for i in 0..cfg.steps {
        let b1 = -cfg.grid_bound + (i as f64) * h;
        let wi = if i == 0 || i + 1 == cfg.steps {
            0.5
        } else {
            1.0
        };
        for j in 0..cfg.steps {
            let b2 = -cfg.grid_bound + (j as f64) * h;
            let wj = if j == 0 || j + 1 == cfg.steps {
                0.5
            } else {
                1.0
            };
            let beta = array![b1, b2];
            let eta = x.dot(&beta);
            let ll = eta
                .iter()
                .zip(y.iter())
                .map(|(&e, &yy)| yy * e - softplus(e))
                .sum::<f64>();
            let pen = 0.5 * lambda * (b1 * b1 + b2 * b2);
            let w = wi * wj * (ll - pen - max_logk).exp();
            z += w;
            q += w * (b1 * b1 + b2 * b2);
        }
    }
    let post_q = q / z.max(1e-300);
    let grad_rho = -0.5 * lambda * post_q;
    OracleEval { grad_rho }
}

/// Deterministic 2D Bernoulli-logit fixture for the regular-regime LAML
/// vs oracle tests.
///
/// Fixture bumped to n=80 because at n=3 the LAML Laplace approximation has
/// ~25% inherent bias and the prior-Jacobian offset (-0.5·rank(S_+) = -1)
/// is the same magnitude as the un-normalized oracle gradient; at n=80 both
/// pathologies vanish (likelihood gradient is O(n), rank shift is O(1) →
/// sub-2% relative). Deterministic quasi-Halton x₁ ∈ [-1.5, 1.5], y from a
/// fixed-β Bernoulli draw with pseudo-random thresholds; design is
/// well-conditioned and non-separating.
fn regular_regime_logit_fixture() -> (Array2<f64>, Array1<f64>) {
    const N: usize = 80;
    let mut x = Array2::<f64>::zeros((N, 2));
    let mut y = Array1::<f64>::zeros(N);
    // Deterministic, reproducible pattern: x₁ swept on a stretched 1D
    // van der Corput sequence in base 2, mapped to [-1.5, 1.5]; y assigned
    // by a fixed logit threshold against a deterministic per-row pseudo-
    // probability so the y vector contains a balanced mix of 0s and 1s
    // and the design is well-conditioned (no separation).
    let beta_true = [-0.2_f64, 0.6_f64];
    for i in 0..N {
        // Van der Corput base 2 in [0, 1).
        let mut v = 0.0_f64;
        let mut denom = 0.5_f64;
        let mut k = i + 1;
        while k > 0 {
            if k & 1 == 1 {
                v += denom;
            }
            denom *= 0.5;
            k >>= 1;
        }
        let x1 = -1.5 + 3.0 * v;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        let eta = beta_true[0] + beta_true[1] * x1;
        let p = 1.0 / (1.0 + (-eta).exp());
        // Deterministic threshold: van der Corput base 3 in [0, 1).
        let mut u = 0.0_f64;
        let mut den3 = 1.0_f64 / 3.0;
        let mut q = i + 1;
        while q > 0 {
            u += ((q % 3) as f64) * den3;
            den3 /= 3.0;
            q /= 3;
        }
        y[i] = if u < p { 1.0 } else { 0.0 };
    }
    (x, y)
}

fn lamlgradient_external_logit(y: &Array1<f64>, x: &Array2<f64>, rho: f64) -> f64 {
    let w = Array1::ones(y.len());
    let offset = Array1::zeros(y.len());
    let s_list = vec![BlockwisePenalty::new(0..2, Array2::eye(2))];
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 60,
        tol: 1e-8,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho_arr = array![rho];
    let analytic_grad = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        &s_list,
        &opts,
        &rho_arr,
    )
    .expect("external gradient evaluation should succeed");
    analytic_grad[0]
}

#[inline]
fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-8)
}

#[test]
fn tiny_logit_lamlvs_exact_oracle_regular_regime() {
    let (x, y) = regular_regime_logit_fixture();
    let cfg = OracleConfig::default();

    for rho in [-0.4, 0.0, 0.4] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = lamlgradient_external_logit(&y, &x, rho);
        let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
        eprintln!(
            "[regular_regime] rho={:.3} g_exact={:.6e} g_laml={:.6e} rel_err={:.4e}",
            rho, g_exact, g_laml, rel_err
        );
        assert!(g_laml.is_finite() && g_exact.is_finite());
        assert_eq!(
            g_laml.signum(),
            g_exact.signum(),
            "gradient direction mismatch at rho={:.3}: g_laml={:.6e}, g_exact={:.6e}",
            rho,
            g_laml,
            g_exact
        );
        // Hardened from 1.25 -> 0.50: a relative error above 50% on a 3x2
        // logit problem in the regular regime is not "approximately right"
        // — it indicates the LAML gradient is materially miscalibrated
        // against the closed-form oracle. The previous bound permitted
        // bugs that doubled the gradient magnitude.
        assert!(
            rel_err < 0.50,
            "regular-regime discrepancy too large at rho={:.3}: g_laml={:.6e}, g_exact={:.6e}, rel={:.3e}",
            rho,
            g_laml,
            g_exact,
            rel_err
        );
    }
}

#[test]
fn tiny_logit_lamlvs_exact_oracle_regular_regime_sweep_is_stable() {
    let (x, y) = regular_regime_logit_fixture();
    let cfg = OracleConfig::default();

    let mut g_exacts = Vec::new();
    let mut g_lamls = Vec::new();
    let mut rels = Vec::new();
    for rho in [-0.6, -0.3, 0.0, 0.3, 0.6] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = lamlgradient_external_logit(&y, &x, rho);
        let rel = rel_err(g_laml, g_exact);
        eprintln!(
            "[regular_sweep]  rho={:.3} g_exact={:.6e} g_laml={:.6e} rel_err={:.4e}",
            rho, g_exact, g_laml, rel
        );
        assert!(g_exact.is_finite() && g_laml.is_finite());
        g_exacts.push(g_exact);
        g_lamls.push(g_laml);
        rels.push(rel);
    }

    let a = Array1::from_vec(g_lamls);
    let b = Array1::from_vec(g_exacts);
    let dot = a.dot(&b);
    let na = a.dot(&a).sqrt();
    let nb = b.dot(&b).sqrt();
    let cosine = if na * nb > 1e-12 {
        dot / (na * nb)
    } else {
        1.0
    };
    rels.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median_rel = rels[rels.len() / 2];
    let max_rel = rels.iter().copied().fold(0.0_f64, f64::max);

    assert!(
        cosine > 0.85,
        "regular sweep cosine too low: cosine={cosine:.6}"
    );
    // Hardened: median 1.0 -> 0.30 and max 1.25 -> 0.50. A regular-regime
    // sweep should track the oracle to within ~30% on average; the previous
    // bounds let bugs that doubled the gradient magnitude pass.
    assert!(
        median_rel < 0.30,
        "regular sweep median relative error too high: median_rel={median_rel:.3e}"
    );
    assert!(
        max_rel < 0.50,
        "regular sweep max relative error too high: max_rel={max_rel:.3e}"
    );
}

#[test]
fn tiny_logit_lamlvs_exact_oracle_near_separation_stress() {
    // Deliberately near-separable toy design. The point is to track discrepancy
    // behavior under stress, not to enforce tight approximation.
    let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
    let y = array![0.0, 0.0, 1.0];
    let cfg = OracleConfig {
        grid_bound: 9.0,
        steps: 261,
    };

    let mut worst_rel = 0.0_f64;
    for rho in [-0.6, 0.0, 0.8] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = lamlgradient_external_logit(&y, &x, rho);
        let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
        eprintln!(
            "[near_separation] rho={:.3} g_exact={:.6e} g_laml={:.6e} rel_err={:.4e}",
            rho, g_exact, g_laml, rel_err
        );
        worst_rel = worst_rel.max(rel_err);
        assert!(g_laml.is_finite() && g_exact.is_finite());
        assert_eq!(
            g_laml.signum(),
            g_exact.signum(),
            "stress direction mismatch at rho={:.3}: g_laml={:.6e}, g_exact={:.6e}",
            rho,
            g_laml,
            g_exact
        );
    }

    // Sanity cap: we expect larger error than regular regime, but still bounded.
    // Hardened 2.5 -> 1.25. Even in the near-separation stress regime the LAML
    // gradient should not differ from the exact oracle by more than ~125%; a
    // looser bound let cosine-only bugs sneak through.
    assert!(
        worst_rel < 1.25,
        "stress-case discrepancy exploded: worst_rel={:.3e}",
        worst_rel
    );
}

#[test]
fn tiny_logit_lamlvs_exact_oracle_stress_sweep_direction_consistent() {
    let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
    let y = array![0.0, 0.0, 1.0];
    let cfg = OracleConfig {
        grid_bound: 9.0,
        steps: 261,
    };

    let mut mismatches = 0usize;
    let mut worst_rel = 0.0_f64;
    for rho in [-0.8, -0.3, 0.2, 0.7, 1.2] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = lamlgradient_external_logit(&y, &x, rho);
        if g_exact.is_finite() && g_laml.is_finite() {
            let rel = rel_err(g_laml, g_exact);
            eprintln!(
                "[stress_sweep]    rho={:.3} g_exact={:.6e} g_laml={:.6e} rel_err={:.4e}",
                rho, g_exact, g_laml, rel
            );
            if g_exact.abs() > 1e-8 && g_laml.abs() > 1e-8 && g_exact.signum() != g_laml.signum() {
                mismatches += 1;
            }
            worst_rel = worst_rel.max(rel);
        }
    }

    assert!(
        mismatches == 0,
        "stress sweep produced too many sign mismatches: {mismatches}"
    );
    // Hardened 2.5 -> 1.25; matches the near-separation single-test bound.
    assert!(
        worst_rel < 1.25,
        "stress sweep relative error exploded: worst_rel={worst_rel:.3e}"
    );
}

#[test]
fn test_lamlgradient_exact_formula_ground_truth() {
    let (x, y) = regular_regime_logit_fixture();
    let cfg = OracleConfig::default();
    let rho = 0.0;
    let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
    let g_laml = lamlgradient_external_logit(&y, &x, rho);
    let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
    eprintln!(
        "[ground_truth]    rho={:.3} g_exact={:.6e} g_laml={:.6e} rel_err={:.4e}",
        rho, g_exact, g_laml, rel_err
    );
    assert_eq!(g_laml.signum(), g_exact.signum());
    // Hardened 1.25 -> 0.50: the closed-form ground-truth path should agree
    // with the exact quadrature oracle to within 50%.
    assert!(
        rel_err < 0.50,
        "ground-truth mismatch at rho={rho:.3}: g_laml={g_laml:.6e}, g_exact={g_exact:.6e}, rel={rel_err:.3e}",
    );
}

#[test]
fn test_lamlgradient_firth_exact_formula_ground_truth() {
    let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
    let y = array![0.0, 0.0, 1.0];
    let cfg = OracleConfig {
        grid_bound: 9.0,
        steps: 261,
    };
    let rho = 0.0;
    let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
    let g_laml = lamlgradient_external_logit(&y, &x, rho);
    let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
    eprintln!(
        "[firth_gt]        rho={:.3} g_exact={:.6e} g_laml={:.6e} rel_err={:.4e}",
        rho, g_exact, g_laml, rel_err
    );
    assert_eq!(g_laml.signum(), g_exact.signum());
    // Hardened 2.5 -> 1.25: near-separation stress design, but still bounded.
    assert!(
        rel_err < 1.25,
        "firth-path ground-truth mismatch at rho={rho:.3}: g_laml={g_laml:.6e}, g_exact={g_exact:.6e}, rel={rel_err:.3e}",
    );
}
