//! #1690 measurement harness — 1-D P-spline Poisson/Gamma REML repro.
//!
//! Builds a single-smooth `y ~ s(x)` GAM with a cubic B-spline (P-spline) basis
//! + 2nd-order difference penalty (mgcv `bs="ps"`), under Poisson (log link) and
//! Gamma (log link) families at REML, and reports the outer REML cost-eval count,
//! inner PIRLS solve count, and wall time. Rust-level analogue of the issue's
//! `y ~ s(x)` count/positive-continuous repro.
//!
//! Run: `cargo run --release --example poisson_gamma_perf_1690`

use gam::estimate::{FitOptions, fit_gam};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use std::time::Instant;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut z = self.state;
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Poisson sample via Knuth for small means.
    fn poisson(&mut self, lambda: f64) -> f64 {
        let l = (-lambda).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            k += 1.0;
            p *= self.unit();
            if p <= l {
                return k - 1.0;
            }
        }
    }
    /// Gamma(shape,scale) via a simple Marsaglia–Tsang generator (shape >= 1).
    fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }
}

/// Cox–de Boor cubic B-spline basis on `[0,1]` with `n_basis` functions over
/// uniform knots. Returns an `n × n_basis` design block.
fn cubic_bspline_design(xs: &[f64], n_basis: usize) -> Array2<f64> {
    let degree = 3usize;
    let n_internal = n_basis - (degree + 1);
    let n_knots = n_basis + degree + 1;
    let mut knots = vec![0.0f64; n_knots];
    for (k, slot) in knots.iter_mut().enumerate() {
        *slot = if k <= degree {
            0.0
        } else if k >= n_knots - degree - 1 {
            1.0
        } else {
            (k - degree) as f64 / (n_internal as f64 + 1.0)
        };
    }
    fn bspline(i: usize, p: usize, x: f64, knots: &[f64]) -> f64 {
        if p == 0 {
            return if (knots[i] <= x && x < knots[i + 1])
                || (x == 1.0 && knots[i + 1] == 1.0 && knots[i] < knots[i + 1])
            {
                1.0
            } else {
                0.0
            };
        }
        let mut left = 0.0;
        let d1 = knots[i + p] - knots[i];
        if d1 > 0.0 {
            left = (x - knots[i]) / d1 * bspline(i, p - 1, x, knots);
        }
        let mut right = 0.0;
        let d2 = knots[i + p + 1] - knots[i + 1];
        if d2 > 0.0 {
            right = (knots[i + p + 1] - x) / d2 * bspline(i + 1, p - 1, x, knots);
        }
        left + right
    }
    let mut b = Array2::<f64>::zeros((xs.len(), n_basis));
    for (r, &x) in xs.iter().enumerate() {
        for i in 0..n_basis {
            b[[r, i]] = bspline(i, degree, x, &knots);
        }
    }
    b
}

/// 2nd-order difference penalty S = Dᵀ D for `k` coefficients (nullspace dim 2).
fn second_difference_penalty(k: usize) -> Array2<f64> {
    let m = k - 2;
    let mut d = Array2::<f64>::zeros((m, k));
    for r in 0..m {
        d[[r, r]] = 1.0;
        d[[r, r + 1]] = -2.0;
        d[[r, r + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn fit_options(nullspace_dims: Vec<usize>) -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims,
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

fn build_and_fit(family: ResponseFamily, fam_label: &str, n: usize, k: usize) {
    let p = 1 + k;
    let mut rng = Lcg::new(100 + n as u64);

    let mut xcov = vec![0.0f64; n];
    for v in xcov.iter_mut() {
        *v = rng.unit();
    }

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
    }
    let block = cubic_bspline_design(&xcov, k);
    for i in 0..n {
        for c in 0..k {
            x[[i, 1 + c]] = block[[i, c]];
        }
    }
    let s_list = vec![BlockwisePenalty::new(1..(1 + k), second_difference_penalty(k))];

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mu = (1.0 + (5.0 * xcov[i]).sin()).exp();
        y[i] = match family {
            ResponseFamily::Poisson => rng.poisson(mu),
            ResponseFamily::Gamma => rng.gamma(4.0, mu / 4.0),
            _ => unreachable!(),
        };
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let nullspace = vec![2usize];

    let t0 = Instant::now();
    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodSpec::new(family, InverseLink::Standard(StandardLink::Log)),
        &fit_options(nullspace),
    )
    .expect("1-D p-spline REML fit should succeed");
    let dt = t0.elapsed();

    println!(
        "PERF_1690 fam={fam_label:8} n={n:6} k={k} p={p}  time={:8.3}s  outer_cost_evals={:5}  inner_pirls_solves={:5}  reml_score={:.6}  edf={:.4}  grad={:?}  conv={}  lambdas={:?}",
        dt.as_secs_f64(),
        fit.outer_cost_evals,
        fit.inner_pirls_solves,
        fit.reml_score,
        fit.edf_total().unwrap_or(f64::NAN),
        fit.outer_gradient_norm,
        fit.outer_converged,
        fit.lambdas.iter().map(|l| format!("{l:.3e}")).collect::<Vec<_>>(),
    );
}

fn main() {
    let ns: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let ns = if ns.is_empty() { vec![600usize] } else { ns };
    for n in ns {
        build_and_fit(ResponseFamily::Poisson, "poisson", n, 12);
        build_and_fit(ResponseFamily::Gamma, "gamma", n, 12);
    }
}
