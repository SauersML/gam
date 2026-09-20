//! Laplace evidence of a `k`-component full-covariance Gaussian mixture.
//!
//! The rows `y_i ∈ ℝ^d` are modelled as draws from
//! `Σ_j π_j N(μ_j, Ω_j⁻¹)`, and the order `k` is priced by the marginal
//! likelihood
//!
//! ```text
//!     Z_k = ∫ Π_i Σ_j π_j N(y_i | μ_j, Ω_j⁻¹) · π(π) Π_j π(μ_j) π(Ω_j) dπ dμ dΩ
//! ```
//!
//! under proper priors, so that `Z_k` is comparable across orders:
//!
//! * `π ~ Dirichlet(1)`, flat on the simplex;
//! * `μ_j ~ N(ȳ, S)`, centred on the data mean with the data covariance
//!   `S = (1/n) Σ_i (y_i − ȳ)(y_i − ȳ)ᵀ`;
//! * `Ω_j ~ Wishart(d + 1, S⁻¹)`, the least-informative Wishart with a
//!   finite mean, centred on the data precision.
//!
//! The priors are the one-component model's own scale, so they carry no
//! constant of their own and the evidence is affine-equivariant: replacing
//! `y` by `Ay + b` shifts `ln Z_k` by exactly `−n ln|det A|` for every `k`.
//! The solve therefore runs on the whitened rows `x_i = L⁻¹(y_i − ȳ)`,
//! `S = LLᵀ`, where the priors are `N(0, I)` and `Wishart(d + 1, I)`, and the
//! raw-data evidence is the whitened one less `(n/2) ln|S|`.
//!
//! Chart. The weights are `π = softmax([z, 0])` (`z ∈ ℝ^{k−1}`, reference
//! component last), and each precision is `Ω_j = R_j R_jᵀ` with `R_j` lower
//! triangular, its diagonal stored as `ρ_a = ln R_aa`. The integrand in
//! `θ = (z, μ_1, R_1, …, μ_k, R_k)` carries every Jacobian of the chart:
//!
//! * `dπ = Π_j π_j dz` (additive log-ratio chart of the simplex);
//! * `dΩ = 2^d Π_a R_aa^{d−a} dR` (Bartlett, 0-indexed `a`);
//! * `dR_aa = R_aa dρ_a`.
//!
//! It is smooth and unconstrained, and its maximum is interior for every `k`:
//! the Wishart prior keeps each precision finite even for a component that
//! holds a single row or none, so no covariance floor is needed. The Laplace
//! approximation is taken at the mode certified by `opt::Arc` with the exact
//! analytic Hessian. The `k!` relabellings of the mode carry identical mass,
//! which contributes `ln k!`.

use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_linalg::pairwise_reduce::par_deterministic_block_fold;
use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, ArrayView2};
use opt::{
    Arc, DecrementBands, FallbackPolicy, FirstOrderObjective, FirstOrderSample, ObjectiveEvalError,
    SecondOrderObjective, SecondOrderSample, TerminationReason, ZerothOrderObjective,
};
use statrs::function::gamma::ln_gamma;

/// `ln Γ_d(a) = d(d−1)/4 · ln π + Σ_{j=0}^{d−1} ln Γ(a − j/2)`.
fn ln_multivariate_gamma(d: usize, a: f64) -> f64 {
    let df = d as f64;
    let mut out = df * (df - 1.0) / 4.0 * std::f64::consts::PI.ln();
    for j in 0..d {
        out += ln_gamma(a - j as f64 / 2.0);
    }
    out
}

/// Index of the lower-triangular entry `(a, b)`, `b ≤ a`, within a component's
/// `R` block.
#[inline]
fn tri(a: usize, b: usize) -> usize {
    a * (a + 1) / 2 + b
}

/// The whitened problem: rows, order, and the chart's layout.
pub(super) struct MixtureLaplaceProblem {
    /// Whitened rows `x_i = L⁻¹(y_i − ȳ)`, `n × d`.
    x: Array2<f64>,
    k: usize,
    d: usize,
    /// Parameters per component: `d` mean entries then `d(d+1)/2` entries of
    /// `R`, row-major over the lower triangle.
    m: usize,
    dim: usize,
    /// Every constant of the log prior: `ln Γ(k)` of the Dirichlet, and per
    /// component the Gaussian `−(d/2) ln 2π`, the Bartlett `d ln 2` and the
    /// Wishart normalizer `−(d(d+1)/2) ln 2 − ln Γ_d((d+1)/2)`.
    log_prior_norm: f64,
    /// Wilkinson growth of one evaluation's accumulations: the `n`-term row
    /// sums, each row's `k` quadratic forms, and the chart assembly.
    growth: f64,
}

/// Unpacked chart point.
struct Params {
    pi: Vec<f64>,
    log_pi: Vec<f64>,
    mu: Vec<Array1<f64>>,
    /// Lower-triangular `R_j` with positive diagonal.
    r: Vec<Array2<f64>>,
    log_det_r: Vec<f64>,
}

/// One evaluation of the log joint density `F(θ)` in the chart.
pub(super) struct MixtureEval {
    pub(super) value: f64,
    pub(super) gradient: Array1<f64>,
    pub(super) hessian: Option<Array2<f64>>,
    bands: Option<DecrementBands>,
}

/// Per-block row sums: the value, the responsibility-weighted moments of every
/// component, and (with the Hessian) the score outer products.
struct RowSums {
    value: f64,
    magnitude: f64,
    /// `N_j = Σ_i r_ij`.
    counts: Vec<f64>,
    /// `Σ_i r_ij e_ij`, `e_ij = x_i − μ_j`.
    first: Vec<Array1<f64>>,
    /// `Σ_i r_ij e_ij e_ijᵀ`.
    second: Vec<Array2<f64>>,
    /// `Σ_i [Σ_j r_ij g_ij g_ijᵀ − ḡ_i ḡ_iᵀ]` in chart coordinates.
    outer: Option<Array2<f64>>,
    outer_magnitude: f64,
    /// `Σ_i Σ_j r_ij |g_ij|`, the gradient's accumulation magnitude.
    gradient_magnitude: Option<Array1<f64>>,
}

impl RowSums {
    fn zeros(k: usize, d: usize, dim: usize, want_hessian: bool) -> Self {
        Self {
            value: 0.0,
            magnitude: 0.0,
            counts: vec![0.0; k],
            first: vec![Array1::zeros(d); k],
            second: vec![Array2::zeros((d, d)); k],
            outer: want_hessian.then(|| Array2::zeros((dim, dim))),
            outer_magnitude: 0.0,
            gradient_magnitude: want_hessian.then(|| Array1::zeros(dim)),
        }
    }

    fn add(mut self, other: Self) -> Self {
        self.value += other.value;
        self.magnitude += other.magnitude;
        for j in 0..self.counts.len() {
            self.counts[j] += other.counts[j];
            self.first[j] += &other.first[j];
            self.second[j] += &other.second[j];
        }
        if let (Some(a), Some(b)) = (self.outer.as_mut(), other.outer.as_ref()) {
            *a += b;
        }
        self.outer_magnitude += other.outer_magnitude;
        if let (Some(a), Some(b)) = (
            self.gradient_magnitude.as_mut(),
            other.gradient_magnitude.as_ref(),
        ) {
            *a += b;
        }
        self
    }
}

impl MixtureLaplaceProblem {
    pub(super) fn new(x: Array2<f64>, k: usize) -> Self {
        let n = x.nrows();
        let d = x.ncols();
        let m = d + d * (d + 1) / 2;
        let dim = (k - 1) + k * m;
        let df = d as f64;
        let per_component = -0.5 * df * (2.0 * std::f64::consts::PI).ln()
            + df * std::f64::consts::LN_2
            - 0.5 * df * (df + 1.0) * std::f64::consts::LN_2
            - ln_multivariate_gamma(d, 0.5 * (df + 1.0));
        let log_prior_norm = ln_gamma(k as f64) + k as f64 * per_component;
        let growth = accumulation_growth(n + k * (d * d + m) + dim);
        Self {
            x,
            k,
            d,
            m,
            dim,
            log_prior_norm,
            growth,
        }
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn block(&self, j: usize) -> usize {
        self.k - 1 + j * self.m
    }

    fn unpack(&self, theta: &Array1<f64>) -> Params {
        let (k, d) = (self.k, self.d);
        let mut logits = vec![0.0_f64; k];
        for l in 0..k - 1 {
            logits[l] = theta[l];
        }
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = logits.iter().map(|&l| (l - top).exp()).sum();
        let log_total = top + total.ln();
        let log_pi: Vec<f64> = logits.iter().map(|&l| l - log_total).collect();
        let pi: Vec<f64> = log_pi.iter().map(|&l| l.exp()).collect();
        let mut mu = Vec::with_capacity(k);
        let mut r = Vec::with_capacity(k);
        let mut log_det_r = Vec::with_capacity(k);
        for j in 0..k {
            let base = self.block(j);
            mu.push(Array1::from_iter((0..d).map(|c| theta[base + c])));
            let mut rj = Array2::<f64>::zeros((d, d));
            let mut log_det = 0.0;
            for a in 0..d {
                for b in 0..=a {
                    let v = theta[base + d + tri(a, b)];
                    if a == b {
                        rj[[a, a]] = v.exp();
                        log_det += v;
                    } else {
                        rj[[a, b]] = v;
                    }
                }
            }
            r.push(rj);
            log_det_r.push(log_det);
        }
        Params {
            pi,
            log_pi,
            mu,
            r,
            log_det_r,
        }
    }

    /// Accumulate the rows `range`.
    fn row_sums(
        &self,
        params: &Params,
        range: core::ops::Range<usize>,
        want_hessian: bool,
    ) -> RowSums {
        let (k, d, m, dim) = (self.k, self.d, self.m, self.dim);
        let half_d_ln_2pi = 0.5 * d as f64 * (2.0 * std::f64::consts::PI).ln();
        let mut sums = RowSums::zeros(k, d, dim, want_hessian);
        let mut e = vec![Array1::<f64>::zeros(d); k];
        let mut w = vec![Array1::<f64>::zeros(d); k];
        let mut log_terms = vec![0.0_f64; k];
        let mut quad = vec![0.0_f64; k];
        let mut resp = vec![0.0_f64; k];
        // One component's chart score over its support: `k − 1` weight logits
        // followed by its `m` block entries.
        let support = (k - 1) + m;
        let mut score = vec![0.0_f64; support];
        let mut mean_score = vec![0.0_f64; dim];
        for i in range {
            let row = self.x.row(i);
            let mut top = f64::NEG_INFINITY;
            for j in 0..k {
                let rj = &params.r[j];
                for a in 0..d {
                    e[j][a] = row[a] - params.mu[j][a];
                }
                let mut q = 0.0;
                for b in 0..d {
                    let mut wb = 0.0;
                    for a in b..d {
                        wb += rj[[a, b]] * e[j][a];
                    }
                    w[j][b] = wb;
                    q += wb * wb;
                }
                quad[j] = 0.5 * q;
                log_terms[j] = params.log_pi[j] + params.log_det_r[j] - quad[j] - half_d_ln_2pi;
                top = top.max(log_terms[j]);
            }
            let mass: f64 = log_terms.iter().map(|&t| (t - top).exp()).sum();
            let lse = top + mass.ln();
            sums.value += lse;
            sums.magnitude += lse.abs();
            for j in 0..k {
                let rij = (log_terms[j] - lse).exp();
                resp[j] = rij;
                sums.magnitude += rij
                    * (params.log_pi[j].abs()
                        + params.log_det_r[j].abs()
                        + quad[j]
                        + half_d_ln_2pi);
                sums.counts[j] += rij;
                for a in 0..d {
                    let ea = e[j][a];
                    sums.first[j][a] += rij * ea;
                    for b in 0..=a {
                        sums.second[j][[a, b]] += rij * ea * e[j][b];
                    }
                }
            }
            if !want_hessian {
                continue;
            }
            let outer = sums.outer.as_mut().expect("outer accumulator");
            let gmag = sums
                .gradient_magnitude
                .as_mut()
                .expect("gradient magnitude accumulator");
            mean_score.iter_mut().for_each(|v| *v = 0.0);
            for l in 0..k - 1 {
                mean_score[l] = resp[l] - params.pi[l];
            }
            for j in 0..k {
                let rij = resp[j];
                let rj = &params.r[j];
                for l in 0..k - 1 {
                    score[l] = if l == j { 1.0 } else { 0.0 } - params.pi[l];
                }
                let off = k - 1;
                for c in 0..d {
                    let mut v = 0.0;
                    for b in 0..=c {
                        v += rj[[c, b]] * w[j][b];
                    }
                    score[off + c] = v;
                }
                for a in 0..d {
                    for b in 0..=a {
                        let raw = -w[j][b] * e[j][a];
                        score[off + d + tri(a, b)] = if a == b {
                            // Chain to `ρ_a = ln R_aa`: `∂/∂ρ = R_aa ∂/∂R_aa`.
                            raw * rj[[a, a]] + 1.0
                        } else {
                            raw
                        };
                    }
                }
                let base = self.block(j);
                let index = |u: usize| if u < k - 1 { u } else { base + (u - (k - 1)) };
                let mut norm_sq = 0.0;
                for u in 0..support {
                    let su = score[u];
                    norm_sq += su * su;
                    let iu = index(u);
                    gmag[iu] += rij * su.abs();
                    if u >= k - 1 {
                        mean_score[iu] = rij * su;
                    }
                    for v in 0..=u {
                        let iv = index(v);
                        let add = rij * su * score[v];
                        outer[[iu, iv]] += add;
                    }
                }
                sums.outer_magnitude += rij * norm_sq;
            }
            let mut mean_norm_sq = 0.0;
            for u in 0..dim {
                let gu = mean_score[u];
                if gu == 0.0 {
                    continue;
                }
                mean_norm_sq += gu * gu;
                for v in 0..=u {
                    outer[[u, v]] -= gu * mean_score[v];
                }
            }
            sums.outer_magnitude += mean_norm_sq;
        }
        sums
    }

    /// `F(θ)`, the log joint density of the rows and the chart parameters,
    /// with its exact gradient and (when `want_hessian`) Hessian and rounding
    /// bands.
    pub(super) fn evaluate(
        &self,
        theta: &Array1<f64>,
        want_hessian: bool,
    ) -> Result<MixtureEval, String> {
        if theta.len() != self.dim || theta.iter().any(|v| !v.is_finite()) {
            return Err("Gaussian-mixture evidence: non-finite chart point".to_string());
        }
        let (k, d, dim) = (self.k, self.d, self.dim);
        let n = self.x.nrows();
        let params = self.unpack(theta);
        let sums = par_deterministic_block_fold(
            n,
            |range| self.row_sums(&params, range, want_hessian),
            RowSums::add,
        )
        .ok_or_else(|| "Gaussian-mixture evidence: no rows".to_string())?;
        let nf = n as f64;
        let mut value = sums.value + self.log_prior_norm;
        let mut magnitude = sums.magnitude + self.log_prior_norm.abs();
        // Raw-coordinate gradient and the non-outer part of the raw Hessian
        // (`R_aa` in place of `ρ_a`); the chart chain is applied below.
        let mut g = Array1::<f64>::zeros(dim);
        let mut h = want_hessian.then(|| Array2::<f64>::zeros((dim, dim)));
        let mut prior_grad_magnitude = Array1::<f64>::zeros(dim);

        // Weights: likelihood `Σ_i (r_i − π')`, Dirichlet(1) in the chart
        // `Σ_j ln π_j`, whose gradient is `1 − kπ'`; both Hessians are
        // multiples of `−(diag π' − π'π'ᵀ)`, with total multiplier `n + k`.
        let kf = k as f64;
        for j in 0..k {
            value += params.log_pi[j];
            magnitude += params.log_pi[j].abs();
        }
        for l in 0..k - 1 {
            g[l] = sums.counts[l] - nf * params.pi[l] + 1.0 - kf * params.pi[l];
            prior_grad_magnitude[l] = 1.0 + kf * params.pi[l];
            if let Some(h) = h.as_mut() {
                for l2 in 0..k - 1 {
                    let cov =
                        if l == l2 { params.pi[l] } else { 0.0 } - params.pi[l] * params.pi[l2];
                    h[[l, l2]] = -(nf + kf) * cov;
                }
            }
        }

        for j in 0..k {
            let base = self.block(j);
            let rj = &params.r[j];
            let nj = sums.counts[j];
            let m1 = &sums.first[j];
            let mut m2 = sums.second[j].clone();
            for a in 0..d {
                for b in 0..a {
                    m2[[b, a]] = m2[[a, b]];
                }
            }
            let omega = rj.dot(&rj.t());
            let omega_m1 = omega.dot(m1);
            let rt_m1 = rj.t().dot(m1);
            let m2_r = m2.dot(rj);
            // Means: likelihood `Ω_j M1_j`, prior `N(0, I)`.
            for c in 0..d {
                let mu_c = params.mu[j][c];
                g[base + c] = omega_m1[c] - mu_c;
                value -= 0.5 * mu_c * mu_c;
                magnitude += 0.5 * mu_c * mu_c;
                prior_grad_magnitude[base + c] = mu_c.abs();
            }
            // Precision factor: likelihood `−(M2 R)_ab + δ_ab N_j / R_aa`,
            // Wishart(d+1, I) with the Bartlett and log-chart Jacobians
            // `−½ Σ R_ab² + Σ_a (d − a + 1) ln R_aa`.
            for a in 0..d {
                let power = (d - a + 1) as f64;
                for b in 0..=a {
                    let u = base + d + tri(a, b);
                    let rab = rj[[a, b]];
                    value -= 0.5 * rab * rab;
                    magnitude += 0.5 * rab * rab;
                    let mut gu = -m2_r[[a, b]] - rab;
                    prior_grad_magnitude[u] = rab.abs();
                    if a == b {
                        gu += (nj + power) / rab;
                        value += power * rab.ln();
                        magnitude += (power * rab.ln()).abs();
                        prior_grad_magnitude[u] += power / rab;
                    }
                    g[u] = gu;
                }
            }
            let Some(h) = h.as_mut() else {
                continue;
            };
            for c in 0..d {
                for c2 in 0..d {
                    h[[base + c, base + c2]] = -nj * omega[[c, c2]];
                }
                h[[base + c, base + c]] -= 1.0;
            }
            for c in 0..d {
                for a in 0..d {
                    for b in 0..=a {
                        let u = base + d + tri(a, b);
                        let mut v = if c >= b { rj[[c, b]] * m1[a] } else { 0.0 };
                        if c == a {
                            v += rt_m1[b];
                        }
                        h[[base + c, u]] = v;
                        h[[u, base + c]] = v;
                    }
                }
            }
            for a in 0..d {
                for b in 0..=a {
                    let u = base + d + tri(a, b);
                    for a2 in 0..d {
                        // `−δ_bb' M2_aa'`: only entries sharing the column `b`.
                        if b > a2 {
                            continue;
                        }
                        let v = base + d + tri(a2, b);
                        h[[u, v]] = -m2[[a, a2]];
                    }
                    h[[u, u]] -= 1.0;
                    if a == b {
                        let power = (d - a + 1) as f64;
                        let raa = rj[[a, a]];
                        h[[u, u]] -= (nj + power) / (raa * raa);
                    }
                }
            }
        }

        // Chart chain `ρ_a = ln R_aa`: `g_ρ = R_aa g_R`,
        // `H_ρρ' = R_aa R_a'a' H_RR' + δ R_aa g_R`.
        let mut jac = Array1::<f64>::ones(dim);
        for j in 0..k {
            let base = self.block(j);
            for a in 0..d {
                jac[base + d + tri(a, a)] = params.r[j][[a, a]];
            }
        }
        let raw_g = g.clone();
        let gradient = &raw_g * &jac;
        if gradient.iter().any(|v| !v.is_finite()) || !value.is_finite() {
            return Err("Gaussian-mixture evidence: non-finite log joint density".to_string());
        }
        let (hessian, bands) = match h {
            None => (None, None),
            Some(mut hr) => {
                for u in 0..dim {
                    for v in 0..dim {
                        hr[[u, v]] *= jac[u] * jac[v];
                    }
                }
                // Only the diagonal chart entries have a second derivative,
                // `d²R_aa/dρ_a² = R_aa`.
                for j in 0..k {
                    let base = self.block(j);
                    for a in 0..d {
                        let u = base + d + tri(a, a);
                        hr[[u, u]] += raw_g[u] * jac[u];
                    }
                }
                let chained_band: f64 = hr.iter().map(|v| v * v).sum::<f64>().sqrt();
                let outer = sums.outer.expect("outer accumulator");
                for u in 0..dim {
                    for v in 0..=u {
                        let o = outer[[u, v]];
                        hr[[u, v]] += o;
                        if v != u {
                            hr[[v, u]] += o;
                        }
                    }
                }
                if hr.iter().any(|v| !v.is_finite()) {
                    return Err("Gaussian-mixture evidence: non-finite Hessian".to_string());
                }
                let gradient_band = (sums.gradient_magnitude.expect("gradient magnitude")
                    + &(&prior_grad_magnitude * &jac))
                    * self.growth;
                let bands = DecrementBands {
                    objective: self.growth * magnitude,
                    gradient: gradient_band,
                    hessian: self.growth * (chained_band + sums.outer_magnitude),
                };
                (Some(hr), Some(bands))
            }
        };
        Ok(MixtureEval {
            value,
            gradient,
            hessian,
            bands,
        })
    }
}

impl MixtureEval {
    fn into_cost_sample(self) -> SecondOrderSample {
        SecondOrderSample {
            value: -self.value,
            gradient: -self.gradient,
            hessian: self.hessian.map(|h| -h),
            decrement_bands: self.bands,
        }
    }
}

/// The solver's view: it minimizes the negated log joint density.
struct MixtureObjective<'p> {
    problem: &'p MixtureLaplaceProblem,
}

impl ZerothOrderObjective for MixtureObjective<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.problem
            .evaluate(x, false)
            .map(|e| -e.value)
            .map_err(ObjectiveEvalError::recoverable)
    }
}

impl FirstOrderObjective for MixtureObjective<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.problem
            .evaluate(x, false)
            .map(|e| FirstOrderSample {
                value: -e.value,
                gradient: -e.gradient,
            })
            .map_err(ObjectiveEvalError::recoverable)
    }
}

impl SecondOrderObjective for MixtureObjective<'_> {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        self.problem
            .evaluate(x, true)
            .map(MixtureEval::into_cost_sample)
            .map_err(ObjectiveEvalError::recoverable)
    }
}

/// The affine whitening `x = L⁻¹(y − ȳ)`, `S = LLᵀ`.
pub(super) struct Whitening {
    pub(super) mean: Array1<f64>,
    /// Lower Cholesky factor `L` of the data covariance.
    pub(super) factor: Array2<f64>,
    /// `ln|S|`.
    pub(super) log_det_covariance: f64,
}

impl Whitening {
    pub(super) fn of(data: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>), String> {
        let n = data.nrows();
        let d = data.ncols();
        let nf = n as f64;
        let mean = data.sum_axis(ndarray::Axis(0)) / nf;
        let mut covariance = Array2::<f64>::zeros((d, d));
        for row in data.rows() {
            for a in 0..d {
                let ea = row[a] - mean[a];
                for b in 0..=a {
                    covariance[[a, b]] += ea * (row[b] - mean[b]);
                }
            }
        }
        for a in 0..d {
            for b in 0..=a {
                covariance[[a, b]] /= nf;
                covariance[[b, a]] = covariance[[a, b]];
            }
        }
        let factor = covariance
            .cholesky(Side::Lower)
            .map_err(|e| {
                format!(
                    "Gaussian-mixture evidence: the data covariance is singular ({e:?}); the \
                     columns are affinely dependent, so no {d}-dimensional density exists"
                )
            })?
            .lower_triangular();
        let log_det_covariance: f64 = (0..d).map(|a| 2.0 * factor[[a, a]].ln()).sum();
        let mut x = Array2::<f64>::zeros((n, d));
        for (i, row) in data.rows().into_iter().enumerate() {
            for a in 0..d {
                let mut v = row[a] - mean[a];
                for b in 0..a {
                    v -= factor[[a, b]] * x[[i, b]];
                }
                x[[i, a]] = v / factor[[a, a]];
            }
        }
        Ok((
            Self {
                mean,
                factor,
                log_det_covariance,
            },
            x,
        ))
    }
}

/// A certified mixture mode and its Laplace evidence, in raw data units.
pub(super) struct MixtureLaplaceFit {
    pub(super) weights: Array1<f64>,
    pub(super) means: Array2<f64>,
    pub(super) covariances: Vec<Array2<f64>>,
    pub(super) log_evidence: f64,
}

/// The chart point of a hard partition: every row assigned to its nearest
/// seed center, each component's covariance the conjugate combination
/// `(scatter_j + I)/(n_j + 1)` of its rows' scatter about the center with one
/// pseudo-row at the unit whitened covariance (SPD even for an empty
/// component), and weights `(n_j + 1)/(n + k)`. It only starts the solve; the
/// certified mode does not depend on it beyond which basin it lies in.
fn start_point(
    problem: &MixtureLaplaceProblem,
    centers: &Array2<f64>,
) -> Result<Array1<f64>, String> {
    let (k, d) = (problem.k, problem.d);
    let x = &problem.x;
    let n = x.nrows();
    let mut counts = vec![0.0_f64; k];
    let mut scatter = vec![Array2::<f64>::eye(d); k];
    for row in x.rows() {
        let mut best = 0usize;
        let mut best_dist = f64::INFINITY;
        for j in 0..k {
            let dist: f64 = (0..d).map(|a| (row[a] - centers[[j, a]]).powi(2)).sum();
            if dist < best_dist {
                best_dist = dist;
                best = j;
            }
        }
        counts[best] += 1.0;
        for a in 0..d {
            let ea = row[a] - centers[[best, a]];
            for b in 0..d {
                scatter[best][[a, b]] += ea * (row[b] - centers[[best, b]]);
            }
        }
    }
    let mut theta = Array1::<f64>::zeros(problem.dim);
    let reference = (counts[k - 1] + 1.0) / (n + k) as f64;
    for l in 0..k - 1 {
        theta[l] = ((counts[l] + 1.0) / (n + k) as f64 / reference).ln();
    }
    for j in 0..k {
        let covariance = &scatter[j] / (counts[j] + 1.0);
        let precision = covariance
            .cholesky(Side::Lower)
            .map_err(|e| format!("Gaussian-mixture evidence: seed covariance: {e:?}"))?
            .solve_mat(&Array2::eye(d));
        let r = precision
            .cholesky(Side::Lower)
            .map_err(|e| format!("Gaussian-mixture evidence: seed precision: {e:?}"))?
            .lower_triangular();
        let base = problem.block(j);
        for c in 0..d {
            theta[base + c] = centers[[j, c]];
        }
        for a in 0..d {
            for b in 0..=a {
                theta[base + d + tri(a, b)] = if a == b { r[[a, a]].ln() } else { r[[a, b]] };
            }
        }
    }
    Ok(theta)
}

/// Fit a `k`-component mixture to its certified posterior mode and return its
/// Laplace evidence. `centers` (whitened) seed the hard partition the solve
/// starts from. A mode the solver cannot certify, or one whose negated Hessian
/// is not positive definite, is refused: there is no evidence to report for it.
pub(super) fn fit_mixture_laplace(
    data: ArrayView2<'_, f64>,
    k: usize,
    seed_centers: impl FnOnce(ArrayView2<'_, f64>) -> Result<Array2<f64>, String>,
) -> Result<MixtureLaplaceFit, String> {
    let n = data.nrows();
    let d = data.ncols();
    let (whitening, x) = Whitening::of(data)?;
    let centers = seed_centers(x.view())?;
    if centers.dim() != (k, d) {
        return Err(format!(
            "Gaussian-mixture evidence: seeding returned {}x{} centers, expected {k}x{d}",
            centers.nrows(),
            centers.ncols()
        ));
    }
    let problem = MixtureLaplaceProblem::new(x, k);
    let x0 = start_point(&problem, &centers)?;
    let seed = problem.evaluate(&x0, true)?.into_cost_sample();
    let solution = Arc::new(x0.clone(), MixtureObjective { problem: &problem })
        .with_fallback_policy(FallbackPolicy::Never)
        .with_initial_sample(x0, seed)
        .run()
        .map_err(|e| format!("Gaussian-mixture evidence: k={k} mode search: {e}"))?;
    if !matches!(
        solution.termination,
        TerminationReason::NewtonDecrementCertified { .. }
    ) {
        return Err(format!(
            "Gaussian-mixture evidence: k={k} mode search stopped without a certified mode \
             ({:?})",
            solution.termination
        ));
    }
    let theta = solution.final_point;
    let at = problem.evaluate(&theta, true)?;
    let hessian = at
        .hessian
        .as_ref()
        .ok_or_else(|| "Gaussian-mixture evidence: Hessian missing".to_string())?;
    let chol = hessian.mapv(|v| -v).cholesky(Side::Lower).map_err(|e| {
        format!(
            "Gaussian-mixture evidence: k={k} mode is not a strict maximum (negated Hessian not \
             positive definite: {e:?})"
        )
    })?;
    let log_det: f64 = chol.diag().iter().map(|&l| 2.0 * l.ln()).sum();
    let log_evidence = at.value
        + ln_gamma(k as f64 + 1.0)
        + 0.5 * problem.dim() as f64 * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * log_det
        - 0.5 * n as f64 * whitening.log_det_covariance;

    let params = problem.unpack(&theta);
    let l = &whitening.factor;
    let mut means = Array2::<f64>::zeros((k, d));
    let mut covariances = Vec::with_capacity(k);
    for j in 0..k {
        let raw_mean = &whitening.mean + &l.dot(&params.mu[j]);
        means.row_mut(j).assign(&raw_mean);
        let omega = params.r[j].dot(&params.r[j].t());
        let sigma = omega
            .cholesky(Side::Lower)
            .map_err(|e| format!("Gaussian-mixture evidence: component precision: {e:?}"))?
            .solve_mat(&Array2::eye(d));
        let mut raw = l.dot(&sigma).dot(&l.t());
        for a in 0..d {
            for b in 0..a {
                let s = 0.5 * (raw[[a, b]] + raw[[b, a]]);
                raw[[a, b]] = s;
                raw[[b, a]] = s;
            }
        }
        covariances.push(raw);
    }
    Ok(MixtureLaplaceFit {
        weights: Array1::from_vec(params.pi),
        means,
        covariances,
        log_evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_normals(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut unif = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (unif(), unif());
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect()
    }

    fn test_point(problem: &MixtureLaplaceProblem, seed: u64) -> Array1<f64> {
        Array1::from_vec(
            lcg_normals(problem.dim(), seed)
                .into_iter()
                .map(|v| 0.4 * v)
                .collect(),
        )
    }

    /// Test-only oracle: central differences of the analytic value and
    /// gradient against the analytic gradient and Hessian.
    #[test]
    fn analytic_derivatives_match_central_differences() {
        for (k, d) in [(1usize, 1usize), (2, 2), (3, 2), (2, 3)] {
            let n = 37;
            let raw = Array2::from_shape_vec((n, d), lcg_normals(n * d, 7 + k as u64)).unwrap();
            let problem = MixtureLaplaceProblem::new(raw, k);
            let theta = test_point(&problem, 11 * k as u64 + d as u64);
            let at = problem.evaluate(&theta, true).unwrap();
            let hessian = at.hessian.as_ref().unwrap();
            let step = 1e-5;
            for u in 0..problem.dim() {
                let mut plus = theta.clone();
                let mut minus = theta.clone();
                plus[u] += step;
                minus[u] -= step;
                let ep = problem.evaluate(&plus, false).unwrap();
                let em = problem.evaluate(&minus, false).unwrap();
                let fd_g = (ep.value - em.value) / (2.0 * step);
                assert!(
                    (fd_g - at.gradient[u]).abs() <= 1e-5 * (1.0 + at.gradient[u].abs()),
                    "k={k} d={d} gradient[{u}]: analytic {} vs central {fd_g}",
                    at.gradient[u]
                );
                for v in 0..problem.dim() {
                    let fd_h = (ep.gradient[v] - em.gradient[v]) / (2.0 * step);
                    assert!(
                        (fd_h - hessian[[v, u]]).abs() <= 1e-5 * (1.0 + hessian[[v, u]].abs()),
                        "k={k} d={d} hessian[{v},{u}]: analytic {} vs central {fd_h}",
                        hessian[[v, u]]
                    );
                }
            }
        }
    }

    /// The prior's constants make it a probability density in the chart: with
    /// no rows the log joint density is the log prior, and for `k = 2`, `d = 1`
    /// it integrates to one. Each factor integrates separately, so the check
    /// runs one-dimensional trapezoid sums (a test-only oracle) over the weight
    /// logit, a mean, and a log precision factor.
    #[test]
    fn prior_normalizers_integrate_to_one() {
        let problem = MixtureLaplaceProblem::new(Array2::zeros((0, 1)), 2);
        // Layout: [z, μ_1, ρ_1, μ_2, ρ_2].
        let integrate = |f: &dyn Fn(f64) -> f64, lo: f64, hi: f64| {
            let steps = 200_000;
            let h = (hi - lo) / steps as f64;
            (0..=steps)
                .map(|s| {
                    let w = if s == 0 || s == steps { 0.5 } else { 1.0 };
                    w * f(lo + s as f64 * h)
                })
                .sum::<f64>()
                * h
        };
        let params = |z: f64, mu: f64, rho: f64| Array1::from_vec(vec![z, mu, rho, 0.0, 0.0]);
        let log_prior = |theta: &Array1<f64>| -> f64 {
            // With no rows, `F` is the log prior alone.
            let p = problem.unpack(theta);
            let mut v = problem.log_prior_norm;
            for j in 0..2 {
                v += p.log_pi[j];
                v -= 0.5 * p.mu[j][0].powi(2);
                let r = p.r[j][[0, 0]];
                v += -0.5 * r * r + 2.0 * r.ln();
            }
            v
        };
        let base = log_prior(&params(0.0, 0.0, 0.0));
        let z_mass = integrate(
            &|z| (log_prior(&params(z, 0.0, 0.0)) - base).exp(),
            -40.0,
            40.0,
        );
        let mu_mass = integrate(
            &|m| (log_prior(&params(0.0, m, 0.0)) - base).exp(),
            -40.0,
            40.0,
        );
        let rho_mass = integrate(
            &|r| (log_prior(&params(0.0, 0.0, r)) - base).exp(),
            -30.0,
            5.0,
        );
        // Normalized total: `e^{base}` times each factor's mass relative to
        // its value at the origin, and the second component's `μ`, `ρ` factors
        // repeat the first's.
        let log_total = base + z_mass.ln() + 2.0 * mu_mass.ln() + 2.0 * rho_mass.ln();
        assert!(log_total.abs() < 1e-9, "log prior mass {log_total}");
    }

    /// The evidence is affine-equivariant: `y → Ay + b` shifts it by exactly
    /// `−n ln|det A|`, whatever the scale of `A`.
    #[test]
    fn evidence_is_affine_equivariant() {
        let n = 150;
        let noise = lcg_normals(2 * n, 3);
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let c = if i % 3 == 0 { -2.5 } else { 2.0 };
            data[[i, 0]] = c + 0.4 * noise[2 * i];
            data[[i, 1]] = 0.3 * noise[2 * i + 1] + 0.2 * c;
        }
        let seed = |x: ArrayView2<'_, f64>| -> Result<Array2<f64>, String> {
            gam_terms::basis::select_centers_by_strategy(
                x,
                &gam_terms::basis::CenterStrategy::KMeans {
                    num_centers: 2,
                    max_iter: 25,
                },
            )
            .map_err(|e| e.to_string())
        };
        let base = fit_mixture_laplace(data.view(), 2, seed).unwrap();
        let a = ndarray::array![[3.0e-4, 1.0e-4], [-2.0e-4, 5.0e-4]];
        let det: f64 = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
        let moved = data.dot(&a.t()) + &ndarray::array![7.0, -11.0];
        let shifted = fit_mixture_laplace(moved.view(), 2, seed).unwrap();
        let expected = base.log_evidence - n as f64 * det.abs().ln();
        assert!(
            (shifted.log_evidence - expected).abs() < 1e-6 * expected.abs().max(1.0),
            "affine image evidence {} vs expected {expected}",
            shifted.log_evidence
        );
    }
}
