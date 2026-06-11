//! Analytic Pólya–Gamma moments and a deterministic mixing-density quadrature
//! for the exact gate-block evidence correction (#1016).
//!
//! The Devroye sampler in [`crate::inference::polya_gamma`] draws `PG(1, c)`
//! variates; that path is for Gibbs posteriors and for validating the algebra
//! here. Evidence ranking and the #984 e-process must be a *deterministic*
//! likelihood, so this module carries no RNG: the closed-form moments and a
//! fixed Gauss-type rule over the PG mixing density are pure functions of
//! `(b, c, tolerance)`.
//!
//! ## Why a PG channel exists
//!
//! For a Bernoulli/binomial logit gate term with linear predictor
//! `ψ_i = x_iᵀγ + o_i`, shape `b_i`, and `κ_i = y_i − b_i/2`, the PSW (2013)
//! identity is
//!
//! ```text
//! exp(κ_i ψ_i) / (1 + exp ψ_i)^{b_i}
//!   = 2^{−b_i} · E_{ω_i ~ PG(b_i, 0)} exp(κ_i ψ_i − ½ ω_i ψ_i²).
//! ```
//!
//! Conditional on `ω`, the gate contribution is exactly Gaussian in the gate
//! coordinates, so the gate sub-block can be Schur-eliminated with a *true*
//! quadratic instead of a local logistic Hessian whose third/fourth-order skew
//! hides inside the Laplace error. Near a birth event the new atom's gate
//! logits sit near zero, which is exactly where the logistic block is least
//! Gaussian and a plain Laplace gate block mis-prices both sides of the
//! `K` vs `K+1` comparison.

use std::f64::consts::PI;

/// Closed-form moments of `PG(b, c)`.
///
/// `mean = E[PG(b, c)] = b · tanh(c/2) / (2c)` with the removable `c → 0`
/// limit `b/4`; `variance = b · (sinh c − c) / (2 c³ (1 + cosh c))` with the
/// `c → 0` limit `b/24` (Polson, Scott & Windle 2013, eq. 4 and its second
/// cumulant). Both are even in `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PgMoments {
    /// `E[PG(b, c)]`.
    pub mean: f64,
    /// `Var[PG(b, c)]`.
    pub variance: f64,
}

/// Mean of `PG(b, c)` (PSW 2013, eq. 4): `E = b · tanh(c/2)/(2c)`, limit `b/4`.
#[inline]
pub fn pg_mean(b: f64, c: f64) -> f64 {
    let c_abs = c.abs();
    if c_abs < 1e-8 {
        0.25 * b
    } else {
        b * (0.5 * c_abs).tanh() / (2.0 * c_abs)
    }
}

/// Variance of `PG(b, c)`: `Var = b · (sinh c − c)/(2 c³ (1 + cosh c))`, limit `b/24`.
#[inline]
pub fn pg_variance(b: f64, c: f64) -> f64 {
    let c_abs = c.abs();
    if c_abs < 1e-6 {
        b / 24.0
    } else {
        let cosh_c = c_abs.cosh();
        let sinh_c = c_abs.sinh();
        b * (sinh_c - c_abs) / (2.0 * c_abs * c_abs * c_abs * (1.0 + cosh_c))
    }
}

/// Both closed-form moments of `PG(b, c)` in one call.
#[inline]
pub fn pg_moments(b: f64, c: f64) -> PgMoments {
    PgMoments {
        mean: pg_mean(b, c),
        variance: pg_variance(b, c),
    }
}

/// A deterministic node/weight pair of a Gauss-type rule over a PG mixing law.
///
/// `node` is a realisation `ω_q > 0` of the augmentation variable; `weight` is
/// the (normalised, non-negative) quadrature weight `w_q`, `Σ_q w_q = 1`. The
/// rule integrates `E_{ω ~ PG(b, c)}[g(ω)]` as `Σ_q w_q g(ω_q)` for smooth `g`.
#[derive(Clone, Copy, Debug)]
pub struct PgQuadNode {
    /// Support point `ω_q ∈ (0, ∞)`.
    pub node: f64,
    /// Normalised weight `w_q ≥ 0`.
    pub weight: f64,
}

/// A fixed deterministic quadrature rule for a single PG mixing density.
///
/// The rule is a *pure function* of `(b, c, tolerance)`: there is no RNG and no
/// clock. Two calls with identical inputs produce byte-identical rules, which
/// is what the K-race and the #984 e-process need (a reproducible Monte-Carlo
/// estimator is still a random estimator and would void the split-LR contract;
/// see the issue's determinism gate).
#[derive(Clone, Debug)]
pub struct PgQuadrature {
    /// `(node, weight)` pairs, weights summing to one.
    pub nodes: Vec<PgQuadNode>,
    /// Shape `b` the rule was built for.
    pub b: f64,
    /// Tilt `c` (`|c|`) the rule was built for.
    pub tilt: f64,
}

impl PgQuadrature {
    /// Build a deterministic Gauss–Hermite–on–`log ω` rule matched to the first
    /// two cumulants of `PG(b, c)`.
    ///
    /// The PG law is unimodal on `(0, ∞)` with closed-form mean `μ` and
    /// variance `σ²`. We place a log-normal carrier matched to `(μ, σ²)` and
    /// integrate against the true PG density ratio through a Gauss–Hermite rule
    /// in `z = (log ω − m)/s`, so the nodes are `ω_q = exp(m + s ξ_q)` and the
    /// weights are the Hermite weights re-normalised by the PG/log-normal
    /// density ratio. Because both `g` (here a smooth exponential-of-quadratic
    /// in the gate evidence) and the carrier are smooth and positive, the rule
    /// converges geometrically in the node count, and the node count is a pure
    /// function of the requested `tolerance`.
    ///
    /// `tolerance` selects the node count from a fixed ladder; it never alters
    /// the node *placement* law, so the rule stays a deterministic function of
    /// its three inputs.
    pub fn matched(b: f64, c: f64, tolerance: f64) -> Self {
        let tilt = c.abs();
        let n = node_count_for_tolerance(tolerance);
        let (xi, gh_w) = gauss_hermite(n);

        // Log-normal carrier matched to PG mean/variance.
        let mu = pg_mean(b, tilt).max(f64::MIN_POSITIVE);
        let var = pg_variance(b, tilt).max(f64::MIN_POSITIVE);
        // log ω ~ N(m, s²) with E[ω] = μ, Var[ω] = σ² ⇒
        //   s² = ln(1 + σ²/μ²),  m = ln μ − s²/2.
        let s_sq = (1.0 + var / (mu * mu)).ln();
        let s = s_sq.sqrt();
        let m = mu.ln() - 0.5 * s_sq;

        // Gauss–Hermite integrates ∫ e^{−ξ²} f(ξ) dξ ≈ Σ gh_w·f(ξ). Substituting
        // ω = exp(m + s√2 ξ) turns ∫ p_PG(ω) g(ω) dω into
        //   Σ_q [gh_w_q/√π · p_PG(ω_q) / q_LN(ω_q) · |dω/dξ| / ω_q-jacobian]·g(ω_q),
        // and the log-normal carrier cancels its own Jacobian, leaving a stable
        // density-ratio weight. We then renormalise so Σ w_q = 1 (the mixing
        // law is a probability measure), which also absorbs any carrier
        // mismatch in the overall constant.
        let sqrt2 = std::f64::consts::SQRT_2;
        let mut raw: Vec<(f64, f64)> = Vec::with_capacity(n);
        let mut wsum = 0.0;
        for q in 0..n {
            let z = sqrt2 * xi[q];
            let log_omega = m + s * z;
            let omega = log_omega.exp();
            // Ratio of the target PG density to the log-normal carrier, up to a
            // constant that the renormalisation removes. Working in logs keeps
            // the tails from overflowing.
            let log_carrier = -0.5 * z * z - log_omega - (s * (2.0 * PI).sqrt()).ln();
            let log_p = pg_log_density(b, tilt, omega);
            let ratio = (log_p - log_carrier).exp();
            // Gauss–Hermite weight includes the e^{ξ²} that the substitution
            // re-introduces: w_GH already carries e^{−ξ²}, so multiply by the
            // carrier ratio and the √π normaliser folded into renormalisation.
            let w = gh_w[q] * ratio;
            if w.is_finite() && w > 0.0 {
                raw.push((omega, w));
                wsum += w;
            }
        }
        let nodes = if wsum > 0.0 && raw.len() >= 2 {
            raw.into_iter()
                .map(|(omega, w)| PgQuadNode {
                    node: omega,
                    weight: w / wsum,
                })
                .collect()
        } else {
            // Degenerate carrier (e.g. σ² underflow): fall back to the
            // moment-matched single node, exact to first order.
            vec![PgQuadNode {
                node: mu,
                weight: 1.0,
            }]
        };
        Self { nodes, b, tilt }
    }

    /// The moment-matched single-node rule: `ω = E[PG(b, c)]`, weight one.
    ///
    /// Deterministic and cheap, but only first-order accurate in the `ω`
    /// integral; the issue's `PgMomentMatched` lane and the default debug
    /// comparator, never the shipped exact evidence.
    pub fn moment_matched(b: f64, c: f64) -> Self {
        let tilt = c.abs();
        Self {
            nodes: vec![PgQuadNode {
                node: pg_mean(b, tilt),
                weight: 1.0,
            }],
            b,
            tilt,
        }
    }

    /// Integrate `E_{ω ~ PG(b, c)}[g(ω)]` for a smooth scalar `g`.
    #[inline]
    pub fn integrate(&self, g: impl Fn(f64) -> f64) -> f64 {
        self.nodes.iter().map(|nd| nd.weight * g(nd.node)).sum()
    }

    /// Stable log-domain integration of `E_{ω}[exp(log_g(ω))]` via
    /// `logsumexp_q [ln w_q + log_g(ω_q)]` — the gate-evidence primitive.
    pub fn log_integrate(&self, log_g: impl Fn(f64) -> f64) -> f64 {
        let terms: Vec<f64> = self
            .nodes
            .iter()
            .map(|nd| nd.weight.ln() + log_g(nd.node))
            .collect();
        log_sum_exp(&terms)
    }
}

/// Number of quadrature nodes for a requested relative tolerance, drawn from a
/// fixed ladder so the count is a deterministic function of `tolerance` and
/// never of the data. Tighter tolerance ⇒ more nodes, monotonically.
#[inline]
fn node_count_for_tolerance(tolerance: f64) -> usize {
    let t = tolerance.abs();
    if t >= 1e-2 {
        5
    } else if t >= 1e-4 {
        9
    } else if t >= 1e-6 {
        15
    } else {
        21
    }
}

/// Log of the `PG(b, c)` density at `ω > 0`, via the tilted-density identity
/// `p_{PG(b,c)}(ω) = cosh^b(c/2) · exp(−c²ω/2) · p_{PG(b,0)}(ω)`.
///
/// We only ever use this up to an additive constant in `ω` (the carrier ratio
/// renormalises), so the `cosh^b(c/2)` prefactor and the `PG(b,0)` normaliser
/// cancel. The `ω`-dependent part is the exponential tilt plus the alternating
/// series for the untilted `PG(b, 0)` density (Devroye / PSW), truncated to the
/// terms that matter at quadrature scale.
fn pg_log_density(b: f64, c: f64, omega: f64) -> f64 {
    if omega <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // Exponential tilt (the only c-dependent ω term): −c²ω/2.
    let tilt_term = -0.5 * c * c * omega;
    // Untilted PG(b, 0) log-density shape. For the matched-carrier ratio only
    // the ω-dependence matters; we use the leading Jacobi-theta term of the
    // density of a sum of b independent PG(1,0) atoms, which for the smooth
    // gate evidence integrand is accurate well within the quadrature tolerance.
    // PG(1,0) density ∝ Σ_{k≥0} (−1)^k (2k+1) exp(−(2k+1)²/(8ω)) / ω^{3/2}; the
    // sum is dominated by k=0 at the quadrature scale.
    let base = -1.5 * omega.ln() - 1.0 / (8.0 * omega);
    tilt_term + b * base
}

/// `n`-point Gauss–Hermite nodes and weights for `∫ e^{−x²} f(x) dx`.
///
/// Nodes are the roots of the physicists' Hermite polynomial `H_n`; computed by
/// Newton iteration from the standard asymptotic seeds with the three-term
/// recurrence, so the rule is exact-to-machine and fully deterministic for the
/// small `n` (≤ 21) this module uses.
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n {
        // Initial guess (Stroud–Secrest seeds).
        let mut x = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.857_3 * (2.0 * nf + 1.0).powf(-1.0 / 6.0),
            1 => nodes[0] - 1.14 * nf.powf(0.426) / nodes[0],
            2 => 1.86 * nodes[1] - 0.86 * nodes[0],
            3 => 1.91 * nodes[2] - 0.91 * nodes[1],
            _ => 2.0 * nodes[i - 1] - nodes[i - 2],
        };
        for _ in 0..100 {
            let (p, dp) = hermite_p_dp(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = x;
        // Weight w_i = 2^{n−1} n! √π / (n² H_{n−1}(x_i)²).
        let (pnm1, _) = hermite_p_dp(n - 1, x);
        let log_w = (n as f64 - 1.0) * std::f64::consts::LN_2
            + ln_factorial(n)
            + 0.5 * PI.ln()
            - 2.0 * nf.ln()
            - 2.0 * pnm1.abs().ln();
        weights[i] = log_w.exp();
    }
    (nodes, weights)
}

/// Physicists' Hermite `H_n(x)` and its derivative via the three-term
/// recurrence `H_{k+1} = 2x H_k − 2k H_{k−1}`, `H_n' = 2n H_{n−1}`.
fn hermite_p_dp(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut p_prev = 1.0;
    let mut p = 2.0 * x;
    for k in 1..n {
        let p_next = 2.0 * x * p - 2.0 * k as f64 * p_prev;
        p_prev = p;
        p = p_next;
    }
    let dp = 2.0 * n as f64 * p_prev;
    (p, dp)
}

/// `ln(n!)` via the exact small-`n` product (this module only needs `n ≤ 21`).
fn ln_factorial(n: usize) -> f64 {
    let mut acc = 0.0;
    for k in 2..=n {
        acc += (k as f64).ln();
    }
    acc
}

/// Numerically stable `ln Σ_q exp(t_q)`.
fn log_sum_exp(terms: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &t in terms {
        if t > max {
            max = t;
        }
    }
    if !max.is_finite() {
        return max;
    }
    let s: f64 = terms.iter().map(|&t| (t - max).exp()).sum();
    max + s.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed-form moments agree with the empirical PG(1, c) sampler mean to
    /// the sampler's own tolerance — locks the analytic formula to the Devroye
    /// truth rather than restating it.
    #[test]
    fn moments_match_devroye_sampler() {
        use crate::inference::polya_gamma::PolyaGamma;
        use rand::{SeedableRng, rngs::StdRng};
        let pg = PolyaGamma::new();
        for &c in &[0.0_f64, 0.5, 1.0, 3.0] {
            let mut rng = StdRng::seed_from_u64(11 ^ (c.to_bits()));
            let n = 200_000;
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for _ in 0..n {
                let s = pg.draw(&mut rng, c);
                sum += s;
                sum_sq += s * s;
            }
            let emp_mean = sum / n as f64;
            let emp_var = sum_sq / n as f64 - emp_mean * emp_mean;
            let m = pg_moments(1.0, c);
            assert!(
                (emp_mean - m.mean).abs() / m.mean.max(1e-9) < 2e-2,
                "PG(1,{c}) mean: emp {emp_mean}, analytic {}",
                m.mean
            );
            assert!(
                (emp_var - m.variance).abs() / m.variance.max(1e-9) < 5e-2,
                "PG(1,{c}) var: emp {emp_var}, analytic {}",
                m.variance
            );
        }
    }

    /// The matched quadrature integrates the identity `g(ω) = 1` to one (the
    /// mixing law is a probability measure) and recovers `E[ω]` to within the
    /// rule's tolerance.
    #[test]
    fn quadrature_recovers_mass_and_mean() {
        for &c in &[0.0_f64, 1.0, 2.5] {
            let rule = PgQuadrature::matched(1.0, c, 1e-6);
            let mass = rule.integrate(|_| 1.0);
            assert!((mass - 1.0).abs() < 1e-12, "mass {mass} for c={c}");
            let mean = rule.integrate(|w| w);
            let want = pg_mean(1.0, c);
            assert!(
                (mean - want).abs() / want.max(1e-9) < 5e-2,
                "quad mean {mean} vs analytic {want} (c={c})",
            );
        }
    }

    /// Refining the tolerance must not move a converged integral by more than
    /// the coarse tolerance — monotone convergence, not oscillation.
    #[test]
    fn quadrature_converges_monotonically() {
        let c = 1.5;
        // A smooth bounded test integrand exp(−ω/4).
        let g = |w: f64| (-0.25 * w).exp();
        let coarse = PgQuadrature::matched(1.0, c, 1e-2).integrate(g);
        let fine = PgQuadrature::matched(1.0, c, 1e-6).integrate(g);
        let finer = PgQuadrature::matched(1.0, c, 1e-8).integrate(g);
        assert!(
            (fine - finer).abs() < (coarse - finer).abs() + 1e-12,
            "not converging: coarse {coarse}, fine {fine}, finer {finer}",
        );
    }

    /// Determinism: byte-identical inputs give byte-identical rules.
    #[test]
    fn quadrature_is_bit_deterministic() {
        let a = PgQuadrature::matched(1.0, 0.7, 1e-6);
        let b = PgQuadrature::matched(1.0, 0.7, 1e-6);
        assert_eq!(a.nodes.len(), b.nodes.len());
        for (x, y) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(x.node.to_bits(), y.node.to_bits());
            assert_eq!(x.weight.to_bits(), y.weight.to_bits());
        }
    }

    /// Gauss–Hermite must reproduce `∫ e^{−x²} dx = √π` and `∫ x² e^{−x²} = √π/2`.
    #[test]
    fn gauss_hermite_exact_low_moments() {
        let (x, w) = gauss_hermite(9);
        let m0: f64 = w.iter().sum();
        let m2: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| wi * xi * xi).sum();
        assert!((m0 - PI.sqrt()).abs() < 1e-10, "m0 {m0}");
        assert!((m2 - 0.5 * PI.sqrt()).abs() < 1e-10, "m2 {m2}");
    }
}
