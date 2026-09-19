//! Certified full conformal for the non-Gaussian families of the predict route
//! (`model.predict(interval="conformal", training_data=...)`,
//! `gam predict --conformal`).
//!
//! # The fitting map
//!
//! At the fitted penalty `Ŝ` (frozen at the training fit) the augmented fit for
//! a candidate response `z` at the test row `(x_*, o_*)` is
//!
//! ```text
//!   β̂(z) = argmin_β  Σ_{i≤n} ℓ(x_iβ + o_i; y_i) + ℓ(x_*β + o_*; z) + ½ βᵀŜβ
//! ```
//!
//! with `ℓ` the family's negative log-likelihood at unit dispersion. A fixed
//! dispersion `φ` only rescales the likelihood, so it is folded into `Ŝ`
//! (`Ŝ = φ·Sλ`) and the minimiser is the fit's own. Every `ℓ` here is convex in
//! `η` and `Ŝ ⪰ 0`, so `β̂(z)` is unique whenever it exists.
//!
//! # The score, and why it is symmetric
//!
//! The conformity score of an augmented point `j` is `|u_j|` with
//! `u_j = −∂ℓ/∂η (x_jβ̂ + o_j; y_j)`, the working score of that point at the
//! augmented fit:
//!
//! | family            | `u`                    | as a residual                  |
//! |-------------------|------------------------|--------------------------------|
//! | Bernoulli (logit) | `y − μ`                | raw residual                   |
//! | Poisson (log)     | `y − μ`                | `(y − μ)·μ/V(μ)`               |
//! | NB(θ) (log)       | `θ(y − μ)/(θ + μ)`     | `(y − μ)·μ/V(μ)`               |
//! | Gamma (log)       | `y/μ − 1`              | Pearson `(y − μ)/√V(μ)`, `V = μ²` |
//!
//! The augmented fit is the minimiser of a sum over the `n + 1` points plus a
//! penalty that does not look at them, so it is a symmetric function of the
//! multiset of points; each point's score is the same fixed function of that
//! point and the fit. Permuting the points permutes the scores, which is the
//! exchangeability the conformal rank argument needs. The working score is
//! also strictly increasing in the candidate (`du_*/dz > 0`), which is what
//! turns the candidate line into a monotone walk and makes the tails provable.
//!
//! The penalty is frozen at the training fit, whose smoothing parameters saw
//! the n training responses and not the test response. Every row therefore
//! reports `frozen_rho_certified = 0`: the set is exact for the frozen-penalty
//! map, and the honest ρ-re-selecting map is not certified here.
//!
//! # Certified solves
//!
//! Each augmented problem is solved by damped Newton, and the answer carries a
//! proof of how far it can be from the true minimiser. At an iterate `β̃` with
//! Hessian `H̃`, gradient `g`, `ν = ‖g‖_{H̃⁻¹}` and per-row leverage
//! `l_i = ‖x_i‖_{H̃⁻¹}`: every `w = ∂²ℓ/∂η²` here satisfies
//! `|d ln w/dη| ≤ 1`, so on the `H̃`-ball of radius `R` each weight stays within
//! a factor `e^{l_i R}` of its value at `β̃`. With `R = 4ν` and
//! `max_i l_i · R ≤ ln 2` the Hessian is `⪰ ½H̃` on the ball, and then the
//! minimiser exists and lies within `‖β̂ − β̃‖_{H̃} ≤ 2ν =: e`. It follows that
//! `|Δη_i| ≤ l_i e` and `|Δu_i| ≤ 2 w̃_i l_i e` for every row. A rank decision
//! is taken only when these bounds cannot flip it; otherwise the candidate
//! stays in the set (it can only be conservative, never wrong).
//!
//! # The candidate support
//!
//! * Bernoulli: both levels `{0, 1}` are tested.
//! * Poisson and NB: the counts `0, 1, …` are tested up to a tail point derived
//!   from the data, beyond which no count can be in the set:
//!   - NB: `u ∈ (−θ, y)`, so once the test score reaches `max(max y, θ)` no
//!     training score can match it;
//!   - Poisson and NB (with an unpenalised intercept): the intercept KKT row
//!     `Σ u_i + u_* = 0` and `u_i < y_i` bound the number of training scores at
//!     least `s` by `1 + Σy/s`, which drops below the rank the set needs once
//!     `s ≥ max(max y, Σy/K)`. NB takes the smaller of its two tails, which
//!     keeps the enumeration short when a near-Poisson fit estimates a large
//!     `θ`.
//!   The tail score is turned into a count by one certified solve of the
//!   problem with the test score fixed at that value. Without a provable tail
//!   the set is honestly `[0, ∞)`.
//! * Gamma: the candidate line is walked in the test-score coordinate
//!   `s = u_* ∈ (−1, ∞)`, where the augmented problem is the training fit tilted
//!   by `−s·η_*` (same KKT system as the response problem at
//!   `z = μ_*(1 + s)`). The intercept KKT row and `u_i > −1` exclude every
//!   `s ≥ max(1, (n − r)/(r + 1))`. The interval `[−1, s_top]` is bisected,
//!   certifying a whole sub-interval per solve, until each piece is decided or
//!   its width is below the anchor's own error. `z` is increasing in `s`, so a
//!   run of included pieces maps to `[z(s_lo), z(s_hi)]`; each endpoint is
//!   mapped by its own certified solve at that `s`, so the set's edges are
//!   exact boundaries to solver accuracy.
//!
//! # Ties in the discrete families
//!
//! The discrete families use the smoothed (randomised) conformal p-value
//! `π = (#{s_i > s_*} + U·(1 + T))/(n + 1)`, with `T` the training rows tied
//! with the test point (same covariates, offset and response), so coverage is
//! exactly `1 − α` rather than conservative. `U` is a uniform seeded from a
//! hash of the labeled responses and the test row: the same inputs give the
//! same set in every front end. Gamma scores are continuous and use the plain
//! p-value `(1 + #{s_i ≥ s_*})/(n + 1)`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use faer::Side;
use ndarray::{Array1, Array2, Axis};
use rand::{RngExt, SeedableRng};

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_atv, fast_av, fast_xt_diag_x};
use gam_linalg::utils::{stable_logistic as sigmoid, stable_softplus as softplus};
use gam_problem::types::LikelihoodSpec;
use gam_spec::FamilySpecKind;
use opt::{BacktrackConfig, backtracking_line_search};

use super::full_conformal::{
    ConformalInterval, GLM_ARMIJO_C1, GLM_CONVERGENCE_RTOL, GLM_NEWTON_MAX_BACKTRACKS,
    GLM_NEWTON_MAX_ITERS, conformal_rank_threshold, vec_norm,
};

/// A non-Gaussian family the certified full-conformal set supports. Each has a
/// negative log-likelihood convex in `η` whose curvature satisfies
/// `|d ln w/dη| ≤ 1`, which the solve certificate relies on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConformalGlmFamily {
    BernoulliLogit,
    PoissonLog,
    NegativeBinomialLog { theta: f64 },
    GammaLog,
}

impl ConformalGlmFamily {
    /// The conformal family of a fitted likelihood; `None` for every family
    /// this arm does not support (the negative log-likelihood of the
    /// inverse-Gaussian log link, for one, is not convex in `η`).
    pub fn from_likelihood(spec: &LikelihoodSpec) -> Option<Self> {
        match spec.kind() {
            FamilySpecKind::BinomialLogit => Some(Self::BernoulliLogit),
            FamilySpecKind::PoissonLog => Some(Self::PoissonLog),
            FamilySpecKind::NegativeBinomialLog { theta } => {
                Some(Self::NegativeBinomialLog { theta })
            }
            FamilySpecKind::GammaLog => Some(Self::GammaLog),
            _ => None,
        }
    }

    fn is_discrete(self) -> bool {
        !matches!(self, Self::GammaLog)
    }

    fn name(self) -> &'static str {
        match self {
            Self::BernoulliLogit => "binomial-logit",
            Self::PoissonLog => "poisson-log",
            Self::NegativeBinomialLog { .. } => "negative-binomial-log",
            Self::GammaLog => "gamma-log",
        }
    }

    fn validate_response(self, y: f64) -> Result<(), String> {
        let ok = match self {
            Self::BernoulliLogit => y == 0.0 || y == 1.0,
            Self::PoissonLog | Self::NegativeBinomialLog { .. } => {
                y.is_finite() && y >= 0.0 && y.fract() == 0.0
            }
            Self::GammaLog => y.is_finite() && y > 0.0,
        };
        if ok {
            Ok(())
        } else {
            let support = match self {
                Self::BernoulliLogit => "0 or 1",
                Self::PoissonLog | Self::NegativeBinomialLog { .. } => "a non-negative integer",
                Self::GammaLog => "positive and finite",
            };
            Err(format!(
                "{} full conformal: labeled response {y} is not {support}",
                self.name()
            ))
        }
    }

    /// `ℓ(η; y)` up to terms free of `η`.
    fn nll(self, eta: f64, y: f64) -> f64 {
        match self {
            Self::BernoulliLogit => softplus(eta) - y * eta,
            Self::PoissonLog => eta.exp() - y * eta,
            Self::NegativeBinomialLog { theta } => {
                (y + theta) * softplus(eta - theta.ln()) - y * eta
            }
            Self::GammaLog => y * (-eta).exp() + eta,
        }
    }

    /// Working score `u = −∂ℓ/∂η` and curvature `w = ∂²ℓ/∂η²`.
    fn score_weight(self, eta: f64, y: f64) -> (f64, f64) {
        match self {
            Self::BernoulliLogit => {
                let p = sigmoid(eta);
                (y - p, p * sigmoid(-eta))
            }
            Self::PoissonLog => {
                let mu = eta.exp();
                (y - mu, mu)
            }
            Self::NegativeBinomialLog { theta } => {
                let t = eta - theta.ln();
                let p = sigmoid(t);
                let q = sigmoid(-t);
                (y * q - theta * p, (y + theta) * p * q)
            }
            Self::GammaLog => {
                let r = y * (-eta).exp();
                (r - 1.0, r)
            }
        }
    }

    /// Response-scale mean `μ(η)`.
    pub fn mean(self, eta: f64) -> f64 {
        match self {
            Self::BernoulliLogit => sigmoid(eta),
            _ => eta.exp(),
        }
    }

    /// The response whose working score at `η` is `s` (the inverse of
    /// `z ↦ u(η; z)`). Increasing in `η` wherever `s` is a reachable score.
    fn response_of_score(self, eta: f64, s: f64) -> f64 {
        match self {
            Self::BernoulliLogit => sigmoid(eta) + s,
            Self::PoissonLog => eta.exp() + s,
            Self::NegativeBinomialLog { theta } => {
                let mu = eta.exp();
                mu + s * (theta + mu) / theta
            }
            Self::GammaLog => eta.exp() * (1.0 + s),
        }
    }

    /// The whole response support, returned when no candidate can be excluded.
    fn whole_support(self) -> Vec<ConformalInterval> {
        let hi = match self {
            Self::BernoulliLogit => 1.0,
            _ => f64::INFINITY,
        };
        vec![ConformalInterval { lo: 0.0, hi }]
    }
}

/// How the test point enters the augmented problem.
#[derive(Clone, Copy, Debug)]
enum Augmentation {
    /// The test row carries the candidate response `z`.
    Response(f64),
    /// The test row's score is fixed at `s`: the objective gains `−s·η_*` and
    /// the test row leaves the Hessian. Its minimiser is the `Response` fit at
    /// the `z` whose score there is `s`.
    Tilt(f64),
}

/// The test row of one prediction.
struct TestRow<'a> {
    x: &'a Array1<f64>,
    offset: f64,
}

/// A certified solve of one augmented problem at the iterate `β̃`.
struct Node {
    beta: Array1<f64>,
    eta_star: f64,
    /// Training working scores `ũ_i`, curvatures `w̃_i` and leverages `l_i`.
    score: Array1<f64>,
    weight: Array1<f64>,
    lever: Array1<f64>,
    lever_star: f64,
    /// `ũ_*` for a response solve, the fixed `s` for a tilt.
    score_star: f64,
    /// `w̃_*` for a response solve; zero for a tilt (the row is not in `H̃`).
    weight_star: f64,
    /// `e = 2ν`, the certified `H̃`-norm distance to the minimiser.
    error: f64,
    /// Largest leverage over the rows in the Hessian.
    max_lever_hessian: f64,
}

impl Node {
    /// Whether a `H̃`-norm error of `e` keeps every Hessian row's curvature
    /// within a factor two on the ball of radius `2e` (see the module doc).
    fn certifies(&self, e: f64) -> bool {
        e.is_finite() && self.max_lever_hessian * 2.0 * e <= std::f64::consts::LN_2
    }

    /// Largest training-score error the anchor's own residual allows.
    fn score_floor(&self) -> f64 {
        let mut floor = 0.0_f64;
        for i in 0..self.score.len() {
            floor = floor.max(2.0 * self.weight[i] * self.lever[i] * self.error);
        }
        floor
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Verdict {
    Member,
    NonMember,
    Undecided,
}

/// Gradient-side quantities at one iterate.
struct State {
    score: Array1<f64>,
    weight: Array1<f64>,
    eta_star: f64,
    score_star: f64,
    weight_star: f64,
    grad: Array1<f64>,
    natural_scale: f64,
}

/// Full-conformal prediction set of one test row.
#[derive(Clone, Debug)]
pub struct GlmFullConformalSet {
    /// Maximal pieces of the set, sorted and disjoint. For the discrete
    /// families each piece is a run of consecutive integers `lo..=hi`.
    pub intervals: Vec<ConformalInterval>,
    pub alpha: f64,
    /// `n + 1`.
    pub n_augmented: usize,
}

/// Labeled rows, frozen penalty and warm start of a non-Gaussian full-conformal
/// set. Built per prediction call from the rows the caller supplies; never
/// persisted.
#[derive(Clone, Debug)]
pub struct GlmFullConformalSubstrate {
    family: ConformalGlmFamily,
    x: Array2<f64>,
    /// `Xᵀ` (p × n), the right-hand side of the leverage solve.
    xt: Array2<f64>,
    y: Array1<f64>,
    offset: Array1<f64>,
    s_lambda: Array2<f64>,
    warm_start: Array1<f64>,
    /// A column that is identically one on the labeled rows with an all-zero
    /// penalty row and column: the unpenalised intercept the tail bounds use.
    intercept: Option<usize>,
}

impl GlmFullConformalSubstrate {
    /// `x` and `offset` are the labeled rows' design and offsets, `y` their
    /// responses, `s_lambda` the frozen penalty in unit-dispersion units (it
    /// must be positive semidefinite) and `warm_start` the fitted
    /// coefficients, the Newton starting point of every augmented solve.
    pub fn new(
        family: ConformalGlmFamily,
        x: Array2<f64>,
        y: Array1<f64>,
        offset: Array1<f64>,
        s_lambda: Array2<f64>,
        warm_start: Array1<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n || offset.len() != n {
            return Err(format!(
                "{} full conformal: {n} labeled design rows but {} responses and {} offsets",
                family.name(),
                y.len(),
                offset.len()
            ));
        }
        if n < 2 {
            return Err(format!(
                "{} full conformal needs at least two labeled rows, got {n}",
                family.name()
            ));
        }
        if s_lambda.nrows() != p || s_lambda.ncols() != p || warm_start.len() != p {
            return Err(format!(
                "{} full conformal: design has p={p} but the penalty is {}×{} and the warm \
                 start has {} entries",
                family.name(),
                s_lambda.nrows(),
                s_lambda.ncols(),
                warm_start.len()
            ));
        }
        if let ConformalGlmFamily::NegativeBinomialLog { theta } = family
            && !(theta.is_finite() && theta > 0.0)
        {
            return Err(format!(
                "negative-binomial full conformal: theta must be positive and finite, got {theta}"
            ));
        }
        if x.iter().chain(offset.iter()).chain(s_lambda.iter()).any(|v| !v.is_finite()) {
            return Err(format!(
                "{} full conformal: labeled design, offsets and penalty must be finite",
                family.name()
            ));
        }
        for &yi in &y {
            family.validate_response(yi)?;
        }
        let intercept = (0..p).find(|&c| {
            x.column(c).iter().all(|&v| v == 1.0)
                && s_lambda.row(c).iter().all(|&v| v == 0.0)
                && s_lambda.column(c).iter().all(|&v| v == 0.0)
        });
        let xt = x.t().to_owned();
        let warm_start = if warm_start.iter().all(|v| v.is_finite()) {
            warm_start
        } else {
            Array1::zeros(p)
        };
        Ok(Self {
            family,
            x,
            xt,
            y,
            offset,
            s_lambda,
            warm_start,
            intercept,
        })
    }

    fn n(&self) -> usize {
        self.x.nrows()
    }

    fn p(&self) -> usize {
        self.x.ncols()
    }

    /// The minimiser over the labeled rows alone (the tilt at `s = 0`), for
    /// checking that the substrate reproduces the fit it was built from.
    pub fn refit_labeled_rows(&self) -> Result<Array1<f64>, String> {
        let zero = Array1::<f64>::zeros(self.p());
        let row = TestRow {
            x: &zero,
            offset: 0.0,
        };
        let node = self.solve(&row, Augmentation::Tilt(0.0), &self.warm_start)?;
        if !node.certifies(node.error) {
            return Err(format!(
                "{} full conformal: the labeled-row refit did not certify",
                self.family.name()
            ));
        }
        Ok(node.beta)
    }

    /// The full-conformal set at level `1 − alpha` for the test row
    /// `(x_star, offset_star)`.
    pub fn prediction_set(
        &self,
        x_star: &Array1<f64>,
        offset_star: f64,
        alpha: f64,
    ) -> Result<GlmFullConformalSet, String> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(format!("full conformal: alpha must be in (0, 1), got {alpha}"));
        }
        if x_star.len() != self.p() {
            return Err(format!(
                "full conformal: test row has {} entries but the design has p={}",
                x_star.len(),
                self.p()
            ));
        }
        if !offset_star.is_finite() || x_star.iter().any(|v| !v.is_finite()) {
            return Err("full conformal: test row and offset must be finite".to_string());
        }
        let row = TestRow {
            x: x_star,
            offset: offset_star,
        };
        let tau = conformal_rank_threshold(alpha, self.n() + 1);
        let intervals = if self.family.is_discrete() {
            self.discrete_set(&row, tau)
        } else {
            self.continuous_set(&row, tau)
        };
        Ok(GlmFullConformalSet {
            intervals,
            alpha,
            n_augmented: self.n() + 1,
        })
    }

    fn objective(&self, beta: &Array1<f64>, row: &TestRow<'_>, aug: Augmentation) -> f64 {
        let eta = fast_av(&self.x, beta);
        let mut value = 0.5 * beta.dot(&self.s_lambda.dot(beta));
        for i in 0..self.n() {
            value += self.family.nll(eta[i] + self.offset[i], self.y[i]);
        }
        let eta_star = row.x.dot(beta) + row.offset;
        match aug {
            Augmentation::Response(z) => value + self.family.nll(eta_star, z),
            Augmentation::Tilt(s) => value - s * eta_star,
        }
    }

    fn state(&self, beta: &Array1<f64>, row: &TestRow<'_>, aug: Augmentation) -> State {
        let n = self.n();
        let eta = fast_av(&self.x, beta);
        let mut score = Array1::<f64>::zeros(n);
        let mut weight = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (u, w) = self.family.score_weight(eta[i] + self.offset[i], self.y[i]);
            score[i] = u;
            weight[i] = w;
        }
        let eta_star = row.x.dot(beta) + row.offset;
        let (score_star, weight_star) = match aug {
            Augmentation::Response(z) => self.family.score_weight(eta_star, z),
            Augmentation::Tilt(s) => (s, 0.0),
        };
        let xtu = fast_atv(&self.x, &score);
        let s_beta = self.s_lambda.dot(beta);
        let mut grad = &s_beta - &xtu;
        grad.scaled_add(-score_star, row.x);
        let natural_scale = 1.0
            + vec_norm(&xtu)
            + vec_norm(&s_beta)
            + vec_norm(row.x) * score_star.abs();
        State {
            score,
            weight,
            eta_star,
            score_star,
            weight_star,
            grad,
            natural_scale,
        }
    }

    fn hessian(&self, state: &State, row: &TestRow<'_>) -> Array2<f64> {
        let mut h = fast_xt_diag_x(&self.x, &state.weight) + &self.s_lambda;
        if state.weight_star > 0.0 {
            let p = self.p();
            for a in 0..p {
                for b in 0..p {
                    h[[a, b]] += state.weight_star * row.x[a] * row.x[b];
                }
            }
        }
        h
    }

    /// Same scale-invariant KKT acceptance as the Layer-2 homotopy's cold fit.
    fn kkt_converged(&self, state: &State) -> bool {
        let g_norm = vec_norm(&state.grad);
        let dimension_scale = ((self.n() + 1) as f64).sqrt() * (self.p() as f64).sqrt();
        g_norm < GLM_CONVERGENCE_RTOL * dimension_scale
            || g_norm / state.natural_scale < GLM_CONVERGENCE_RTOL
    }

    /// Damped Newton on the augmented problem from `init`, then the solve
    /// certificate at the final iterate. Acceptance is the caller's, through
    /// [`Node::certifies`].
    fn solve(
        &self,
        row: &TestRow<'_>,
        aug: Augmentation,
        init: &Array1<f64>,
    ) -> Result<Node, String> {
        let mut beta = init.clone();
        let mut value = self.objective(&beta, row, aug);
        if !value.is_finite() {
            beta = Array1::zeros(self.p());
            value = self.objective(&beta, row, aug);
        }
        if !value.is_finite() {
            return Err("full conformal: augmented objective is not finite".to_string());
        }
        for _ in 0..GLM_NEWTON_MAX_ITERS {
            let state = self.state(&beta, row, aug);
            if self.kkt_converged(&state) {
                break;
            }
            let chol = self
                .hessian(&state, row)
                .cholesky(Side::Lower)
                .map_err(|e| format!("full conformal: augmented Hessian not SPD: {e:?}"))?;
            let step = chol.solvevec(&state.grad);
            let decrease = state.grad.dot(&step);
            let accepted = if decrease > f64::EPSILON * value.abs() {
                let search = backtracking_line_search::<_, std::convert::Infallible>(
                    BacktrackConfig {
                        initial_step: 1.0,
                        contraction: 0.5,
                        max_steps: GLM_NEWTON_MAX_BACKTRACKS,
                    },
                    |t| {
                        let mut cand = beta.clone();
                        cand.scaled_add(-t, &step);
                        let cand_value = self.objective(&cand, row, aug);
                        Ok(if cand_value.is_finite() {
                            Some((cand_value, cand))
                        } else {
                            None
                        })
                    },
                    |t, cand_value| {
                        cand_value < value && cand_value <= value - GLM_ARMIJO_C1 * t * decrease
                    },
                );
                match search {
                    Ok(accepted) => accepted,
                    Err(never) => match never {},
                }
            } else {
                None
            };
            match accepted {
                Some(accepted) => {
                    beta = accepted.payload;
                    value = accepted.value;
                }
                // No step resolves a decrease of the objective above its
                // round-off, so the objective can no longer rank iterates: take
                // the full Newton step while it shrinks the gradient, which is
                // still resolved, and stop at the gradient's own floor.
                None => {
                    let mut cand = beta.clone();
                    cand.scaled_add(-1.0, &step);
                    let cand_value = self.objective(&cand, row, aug);
                    if !(cand_value.is_finite()
                        && vec_norm(&self.state(&cand, row, aug).grad) < vec_norm(&state.grad))
                    {
                        break;
                    }
                    beta = cand;
                    value = cand_value;
                }
            }
        }
        let state = self.state(&beta, row, aug);
        let chol = self
            .hessian(&state, row)
            .cholesky(Side::Lower)
            .map_err(|e| format!("full conformal: augmented Hessian not SPD: {e:?}"))?;
        let nu = state.grad.dot(&chol.solvevec(&state.grad)).max(0.0).sqrt();
        let h_inv_xt = chol.solve_mat(&self.xt);
        let lever = (&self.x * &h_inv_xt.t())
            .sum_axis(Axis(1))
            .mapv(|v| v.max(0.0).sqrt());
        let lever_star = row.x.dot(&chol.solvevec(row.x)).max(0.0).sqrt();
        let mut max_lever_hessian = lever.iter().fold(0.0_f64, |m, &v| m.max(v));
        if state.weight_star > 0.0 {
            max_lever_hessian = max_lever_hessian.max(lever_star);
        }
        let error = 2.0 * nu;
        if !(error.is_finite() && max_lever_hessian.is_finite() && state.eta_star.is_finite()) {
            return Err("full conformal: augmented solve is not finite".to_string());
        }
        Ok(Node {
            beta,
            eta_star: state.eta_star,
            score: state.score,
            weight: state.weight,
            lever,
            lever_star,
            score_star: state.score_star,
            weight_star: state.weight_star,
            error,
            max_lever_hessian,
        })
    }

    /// Rank decision for every candidate whose true test score lies in
    /// `[t_lo, t_hi]` and whose fit lies within `H̃`-distance `e` of `node`.
    /// `self_weight` is the p-value numerator's own term (`1`, or `U(1 + T)`
    /// when ties are randomised) and `tied` marks rows excluded as ties.
    fn verdict(
        &self,
        node: &Node,
        e: f64,
        t_lo: f64,
        t_hi: f64,
        self_weight: f64,
        tau: f64,
        tied: Option<&[bool]>,
    ) -> Verdict {
        let mut certain = 0usize;
        let mut possible = 0usize;
        for i in 0..self.n() {
            if tied.is_some_and(|t| t[i]) {
                continue;
            }
            let a = node.score[i].abs();
            let err = 2.0 * node.weight[i] * node.lever[i] * e;
            if a - err > t_hi {
                certain += 1;
            }
            if a + err >= t_lo {
                possible += 1;
            }
        }
        if certain as f64 + self_weight > tau {
            Verdict::Member
        } else if possible as f64 + self_weight <= tau {
            Verdict::NonMember
        } else {
            Verdict::Undecided
        }
    }

    /// The seeded tie-break uniform of one test row.
    fn tie_break_uniform(&self, row: &TestRow<'_>) -> f64 {
        let mut hasher = DefaultHasher::new();
        for v in self.y.iter().chain(row.x.iter()) {
            v.to_bits().hash(&mut hasher);
        }
        row.offset.to_bits().hash(&mut hasher);
        rand::rngs::StdRng::seed_from_u64(hasher.finish()).random::<f64>()
    }

    /// Enumeration over the discrete support with randomised ties.
    fn discrete_set(&self, row: &TestRow<'_>, tau: f64) -> Vec<ConformalInterval> {
        let n = self.n();
        let u_tie = self.tie_break_uniform(row);
        // `K`: the most training rows with score at least the test score that
        // still leave a candidate outside the set, `k + U ≤ τ`.
        if tau < u_tie {
            return self.family.whole_support();
        }
        let k_max = (tau - u_tie).floor() as usize;
        let z_max = match self.family {
            ConformalGlmFamily::BernoulliLogit => 1.0,
            _ => match self.count_tail(row, k_max) {
                Some(z) => z,
                None => return self.family.whole_support(),
            },
        };
        let twin: Vec<bool> = (0..n)
            .map(|i| {
                self.offset[i] == row.offset
                    && self.x.row(i).iter().zip(row.x.iter()).all(|(a, b)| a == b)
            })
            .collect();
        let mut tied = vec![false; n];
        let mut kept = Vec::<f64>::new();
        let mut warm = self.warm_start.clone();
        let mut z = 0.0_f64;
        while z <= z_max {
            let keep = match self.solve(row, Augmentation::Response(z), &warm) {
                Ok(node) => {
                    let keep = if node.certifies(node.error) {
                        let t = node.score_star.abs();
                        let err = 2.0 * node.weight_star * node.lever_star * node.error;
                        let mut ties = 0usize;
                        for i in 0..n {
                            tied[i] = twin[i] && self.y[i] == z;
                            ties += usize::from(tied[i]);
                        }
                        let self_weight = u_tie * (1 + ties) as f64;
                        self.verdict(
                            &node,
                            node.error,
                            (t - err).max(0.0),
                            t + err,
                            self_weight,
                            tau,
                            Some(&tied),
                        ) != Verdict::NonMember
                    } else {
                        true
                    };
                    warm = node.beta;
                    keep
                }
                Err(_) => true,
            };
            if keep {
                kept.push(z);
            }
            z += 1.0;
        }
        let mut runs = Vec::<ConformalInterval>::new();
        for z in kept {
            match runs.last_mut() {
                Some(last) if last.hi + 1.0 == z => last.hi = z,
                _ => runs.push(ConformalInterval { lo: z, hi: z }),
            }
        }
        runs
    }

    /// The largest count that can be in the set, from the provable score tail
    /// of the module doc; `None` when no tail is provable.
    fn count_tail(&self, row: &TestRow<'_>, k_max: usize) -> Option<f64> {
        let max_y = self.y.iter().fold(0.0_f64, |m, &v| m.max(v));
        // The intercept-KKT bound holds for both count families (`u_i < y_i`
        // for each); NB also has its score bound `θ`, and the smaller wins.
        let kkt_score = self
            .intercept
            .filter(|&c| row.x[c] == 1.0 && k_max > 0)
            .map(|_| max_y.max(self.y.sum() / k_max as f64));
        let tail_score = match self.family {
            ConformalGlmFamily::NegativeBinomialLog { theta } => {
                kkt_score.map_or(max_y.max(theta), |s| s.min(max_y.max(theta)))
            }
            ConformalGlmFamily::PoissonLog => kkt_score?,
            _ => return None,
        };
        if !(tail_score > 0.0 && tail_score.is_finite()) {
            return None;
        }
        let node = self
            .solve(row, Augmentation::Tilt(tail_score), &self.warm_start)
            .ok()?;
        if !node.certifies(node.error) {
            return None;
        }
        let z_tail = self
            .family
            .response_of_score(node.eta_star + node.lever_star * node.error, tail_score);
        z_tail.is_finite().then(|| z_tail.floor())
    }

    /// Certified walk over the test-score coordinate for the continuous
    /// (Gamma) family.
    fn continuous_set(&self, row: &TestRow<'_>, tau: f64) -> Vec<ConformalInterval> {
        let whole = self.family.whole_support();
        if tau < 1.0 {
            return whole;
        }
        // r: the fewest dominating training rows that keep a candidate in.
        let r = (tau - 1.0).floor() + 1.0;
        let Some(c) = self.intercept else {
            return whole;
        };
        if row.x[c] != 1.0 {
            return whole;
        }
        let n = self.n() as f64;
        let s_top = 1.0_f64.max((n - r) / (r + 1.0));
        // A tilt at s ≥ n has no minimiser (Σu_i = −s with every u_i > −1).
        if !(s_top < n) {
            return whole;
        }
        struct Leaf {
            included: bool,
            s_lo: f64,
            s_hi: f64,
            z_lo: f64,
            z_hi: f64,
            warm: Array1<f64>,
        }
        let mut leaves = Vec::<Leaf>::new();
        let mut stack = vec![(-1.0_f64, s_top, self.warm_start.clone())];
        while let Some((a, b, warm)) = stack.pop() {
            let m = 0.5 * (a + b);
            let half = 0.5 * (b - a);
            let splittable = m > a && m < b;
            let node = match self.solve(row, Augmentation::Tilt(m), &warm) {
                Ok(node) => node,
                Err(_) => {
                    leaves.push(Leaf {
                        included: true,
                        s_lo: a,
                        s_hi: b,
                        z_lo: 0.0,
                        z_hi: f64::INFINITY,
                        warm,
                    });
                    continue;
                }
            };
            let e = node.error + 2.0 * half * node.lever_star;
            if !node.certifies(e) {
                if splittable {
                    stack.push((m, b, node.beta.clone()));
                    stack.push((a, m, node.beta));
                } else {
                    leaves.push(Leaf {
                        included: true,
                        s_lo: a,
                        s_hi: b,
                        z_lo: 0.0,
                        z_hi: f64::INFINITY,
                        warm: node.beta,
                    });
                }
                continue;
            }
            let (t_lo, t_hi) = if a <= 0.0 && 0.0 <= b {
                (0.0, a.abs().max(b.abs()))
            } else {
                (a.abs().min(b.abs()), a.abs().max(b.abs()))
            };
            let included = match self.verdict(&node, e, t_lo, t_hi, 1.0, tau, None) {
                Verdict::Member => true,
                Verdict::NonMember => false,
                Verdict::Undecided => {
                    if splittable && half > node.score_floor() {
                        stack.push((m, b, node.beta.clone()));
                        stack.push((a, m, node.beta));
                        continue;
                    }
                    true
                }
            };
            let z_lo = (node.eta_star - node.lever_star * e).exp() * (1.0 + a);
            let z_hi = (node.eta_star + node.lever_star * e).exp() * (1.0 + b);
            leaves.push(Leaf {
                included,
                s_lo: a,
                s_hi: b,
                z_lo,
                z_hi,
                warm: node.beta,
            });
        }
        // Leaves come off the stack left to right; consecutive included leaves
        // form one run in s, hence one piece in z (z is increasing in s).
        struct Run {
            s_lo: f64,
            s_hi: f64,
            piece: ConformalInterval,
            warm_lo: Array1<f64>,
            warm_hi: Array1<f64>,
        }
        let mut runs = Vec::<Run>::new();
        let mut in_run = false;
        for leaf in leaves {
            if leaf.included {
                match runs.last_mut() {
                    Some(last) if in_run => {
                        last.s_hi = leaf.s_hi;
                        last.piece.lo = last.piece.lo.min(leaf.z_lo);
                        last.piece.hi = last.piece.hi.max(leaf.z_hi);
                        last.warm_hi = leaf.warm;
                    }
                    _ => runs.push(Run {
                        s_lo: leaf.s_lo,
                        s_hi: leaf.s_hi,
                        piece: ConformalInterval {
                            lo: leaf.z_lo,
                            hi: leaf.z_hi,
                        },
                        warm_lo: leaf.warm.clone(),
                        warm_hi: leaf.warm,
                    }),
                }
            }
            in_run = leaf.included;
        }
        // A leaf's z bounds carry its whole width's slack in `η_*`, and a
        // decided leaf next to an excluded one ends on the exact boundary in
        // `s`. The run's image is `[z(s_lo), z(s_hi)]`, so each endpoint is
        // mapped by its own solve at that `s`, whose slack is the solve error
        // alone; the leaf bounds stay when that solve does not certify.
        let endpoint = |s: f64, warm: &Array1<f64>| -> Option<(f64, f64)> {
            let node = self.solve(row, Augmentation::Tilt(s), warm).ok()?;
            if !node.certifies(node.error) {
                return None;
            }
            let slack = node.lever_star * node.error;
            Some((
                (node.eta_star - slack).exp() * (1.0 + s),
                (node.eta_star + slack).exp() * (1.0 + s),
            ))
        };
        let mut pieces = Vec::<ConformalInterval>::new();
        for run in runs {
            let mut piece = run.piece;
            if run.s_lo > -1.0 && piece.lo > 0.0 {
                if let Some((lo, _)) = endpoint(run.s_lo, &run.warm_lo) {
                    piece.lo = piece.lo.max(lo);
                }
            }
            if piece.hi.is_finite() {
                if let Some((_, hi)) = endpoint(run.s_hi, &run.warm_hi) {
                    piece.hi = piece.hi.min(hi);
                }
            }
            pieces.push(piece);
        }
        pieces.sort_by(|p, q| p.lo.total_cmp(&q.lo));
        let mut merged = Vec::<ConformalInterval>::new();
        for piece in pieces {
            match merged.last_mut() {
                Some(last) if piece.lo <= last.hi => last.hi = last.hi.max(piece.hi),
                _ => merged.push(piece),
            }
        }
        merged
    }
}

/// The frozen penalty of a converged GLM fit, recovered from its penalized
/// Hessian `H = XᵀWX + Sλ` and weighted Gram `XᵀWX` (same weights) as
/// `scale·(H − XᵀWX)`. Coefficients outside every `penalized` column range carry
/// no penalty and get exact zeros; the penalized block is projected onto the
/// positive semidefinite cone, which removes the round-off of the difference
/// (the solve certificate needs `Ŝ ⪰ 0`). `scale` is the fit's dispersion,
/// which converts the penalty to the unit-dispersion likelihood.
pub fn penalty_from_normal_and_gram(
    normal: &Array2<f64>,
    gram: &Array2<f64>,
    penalized: &[Range<usize>],
    scale: f64,
) -> Result<Array2<f64>, String> {
    let p = normal.nrows();
    if normal.ncols() != p || gram.nrows() != p || gram.ncols() != p {
        return Err(format!(
            "full conformal penalty: penalized Hessian is {}×{} but the weighted Gram is {}×{}",
            normal.nrows(),
            normal.ncols(),
            gram.nrows(),
            gram.ncols()
        ));
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "full conformal penalty: dispersion must be positive and finite, got {scale}"
        ));
    }
    let mut covered = vec![false; p];
    for range in penalized {
        if range.end > p {
            return Err(format!(
                "full conformal penalty: penalty block {range:?} exceeds p={p}"
            ));
        }
        for j in range.clone() {
            covered[j] = true;
        }
    }
    let idx: Vec<usize> = (0..p).filter(|&j| covered[j]).collect();
    let mut s = Array2::<f64>::zeros((p, p));
    if idx.is_empty() {
        return Ok(s);
    }
    let k = idx.len();
    let mut block = Array2::<f64>::zeros((k, k));
    for (a, &ia) in idx.iter().enumerate() {
        for (b, &ib) in idx.iter().enumerate() {
            let d_ab = normal[[ia, ib]] - gram[[ia, ib]];
            let d_ba = normal[[ib, ia]] - gram[[ib, ia]];
            block[[a, b]] = 0.5 * (d_ab + d_ba);
        }
    }
    if block.iter().any(|v| !v.is_finite()) {
        return Err("full conformal penalty: penalized Hessian is not finite".to_string());
    }
    let (evals, evecs) = block
        .eigh(Side::Lower)
        .map_err(|e| format!("full conformal penalty: eigendecomposition failed: {e:?}"))?;
    let mut scaled = evecs.clone();
    for (j, &lambda) in evals.iter().enumerate() {
        let keep = lambda.max(0.0) * scale;
        scaled.column_mut(j).mapv_inplace(|v| v * keep);
    }
    let psd = scaled.dot(&evecs.t());
    for (a, &ia) in idx.iter().enumerate() {
        for (b, &ib) in idx.iter().enumerate() {
            s[[ia, ib]] = psd[[a, b]];
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Gamma as GammaDist, Poisson as PoissonDist};

    const ALPHA: f64 = 0.1;

    fn design(xs: &[f64]) -> Array2<f64> {
        Array2::from_shape_fn((xs.len(), 3), |(i, j)| match j {
            0 => 1.0,
            1 => xs[i],
            _ => xs[i] * xs[i],
        })
    }

    fn row(x: f64) -> Array1<f64> {
        Array1::from(vec![1.0, x, x * x])
    }

    /// A fixed ridge on the non-intercept columns: independent of the data, so
    /// the augmented points are exactly exchangeable in the Monte Carlo checks.
    fn penalty() -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((3, 3));
        s[[1, 1]] = 0.5;
        s[[2, 2]] = 0.5;
        s
    }

    fn eta_true(x: f64) -> f64 {
        0.3 + 0.8 * x - 0.4 * x * x
    }

    fn draw(family: ConformalGlmFamily, eta: f64, rng: &mut StdRng) -> f64 {
        let mu = family.mean(eta);
        match family {
            ConformalGlmFamily::BernoulliLogit => f64::from(rng.random::<f64>() < mu),
            ConformalGlmFamily::PoissonLog => PoissonDist::new(mu).unwrap().sample(rng),
            ConformalGlmFamily::NegativeBinomialLog { theta } => {
                let lambda = GammaDist::new(theta, mu / theta).unwrap().sample(rng);
                if lambda > 0.0 {
                    PoissonDist::new(lambda).unwrap().sample(rng)
                } else {
                    0.0
                }
            }
            ConformalGlmFamily::GammaLog => GammaDist::new(2.0, mu / 2.0).unwrap().sample(rng),
        }
    }

    struct Data {
        x: Array2<f64>,
        y: Array1<f64>,
        offset: Array1<f64>,
    }

    fn data(family: ConformalGlmFamily, n: usize, rng: &mut StdRng) -> Data {
        let xs: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let offset: Array1<f64> = (0..n).map(|_| rng.random::<f64>() * 0.4 - 0.2).collect();
        let y = xs
            .iter()
            .zip(offset.iter())
            .map(|(&x, &o)| draw(family, eta_true(x) + o, rng))
            .collect();
        Data {
            x: design(&xs),
            y,
            offset,
        }
    }

    fn substrate(family: ConformalGlmFamily, d: &Data) -> GlmFullConformalSubstrate {
        GlmFullConformalSubstrate::new(
            family,
            d.x.clone(),
            d.y.clone(),
            d.offset.clone(),
            penalty(),
            Array1::zeros(3),
        )
        .unwrap()
    }

    fn contains(set: &GlmFullConformalSet, z: f64) -> bool {
        set.intervals.iter().any(|iv| iv.lo <= z && z <= iv.hi)
    }

    const FAMILIES: [ConformalGlmFamily; 4] = [
        ConformalGlmFamily::BernoulliLogit,
        ConformalGlmFamily::PoissonLog,
        ConformalGlmFamily::NegativeBinomialLog { theta: 3.0 },
        ConformalGlmFamily::GammaLog,
    ];

    #[test]
    fn score_and_weight_are_the_derivatives_of_the_nll() {
        for family in FAMILIES {
            let y = if family == ConformalGlmFamily::GammaLog { 1.7 } else { 1.0 };
            for eta in [-2.0, -0.3, 0.0, 0.8, 2.5] {
                let h = 1e-5;
                let d1 = (family.nll(eta + h, y) - family.nll(eta - h, y)) / (2.0 * h);
                let (u, w) = family.score_weight(eta, y);
                assert!((u + d1).abs() < 1e-6, "{family:?} score at {eta}");
                let (u_hi, _) = family.score_weight(eta + h, y);
                let (u_lo, _) = family.score_weight(eta - h, y);
                assert!((w + (u_hi - u_lo) / (2.0 * h)).abs() < 1e-6, "{family:?} weight");
                let (_, w_hi) = family.score_weight(eta + h, y);
                let (_, w_lo) = family.score_weight(eta - h, y);
                let dlogw = (w_hi.ln() - w_lo.ln()) / (2.0 * h);
                assert!(dlogw.abs() <= 1.0 + 1e-6, "{family:?} curvature log-derivative");
                let z = family.response_of_score(eta, u);
                assert!((z - y).abs() < 1e-9, "{family:?} response_of_score inverts the score");
            }
        }
    }

    /// Exact membership of candidate `z` by a direct solve, independent of the
    /// set's pruning and certificates.
    fn brute_force_member(sub: &GlmFullConformalSubstrate, x_star: &Array1<f64>, o: f64, z: f64, alpha: f64) -> bool {
        let row = TestRow { x: x_star, offset: o };
        let node = sub.solve(&row, Augmentation::Response(z), &sub.warm_start).unwrap();
        assert!(node.certifies(node.error));
        let n = sub.n();
        let tau = conformal_rank_threshold(alpha, n + 1);
        let s_star = node.score_star.abs();
        if sub.family.is_discrete() {
            let u = sub.tie_break_uniform(&row);
            let greater = (0..n).filter(|&i| node.score[i].abs() > s_star).count();
            greater as f64 + u > tau
        } else {
            let geq = (0..n).filter(|&i| node.score[i].abs() >= s_star).count();
            geq as f64 + 1.0 > tau
        }
    }

    #[test]
    fn discrete_sets_match_brute_force_enumeration() {
        let mut rng = StdRng::seed_from_u64(20260919);
        for family in &FAMILIES[..3] {
            for _ in 0..8 {
                let d = data(*family, 60, &mut rng);
                let sub = substrate(*family, &d);
                let x = rng.random::<f64>() * 2.0 - 1.0;
                let o = 0.1;
                let set = sub.prediction_set(&row(x), o, ALPHA).unwrap();
                let top = match family {
                    ConformalGlmFamily::BernoulliLogit => 1,
                    _ => 80,
                };
                for z in 0..=top {
                    let z = z as f64;
                    assert_eq!(
                        contains(&set, z),
                        brute_force_member(&sub, &row(x), o, z, ALPHA),
                        "{family:?} at z={z}: set {:?}",
                        set.intervals
                    );
                }
                if *family != ConformalGlmFamily::BernoulliLogit {
                    let last = set.intervals.last().unwrap();
                    assert!(last.hi.is_finite(), "{family:?}: the count tail must bound the set");
                }
            }
        }
    }

    #[test]
    fn near_poisson_negative_binomial_tail_is_on_the_data_scale() {
        // A near-Poisson NB fit estimates a huge θ; the score bound θ alone
        // would put the tail near a million counts.
        let mut rng = StdRng::seed_from_u64(31);
        let family = ConformalGlmFamily::NegativeBinomialLog { theta: 1e6 };
        let d = data(ConformalGlmFamily::PoissonLog, 60, &mut rng);
        let sub = substrate(family, &d);
        let x_star = row(0.3);
        let test = TestRow { x: &x_star, offset: 0.0 };
        let tau = conformal_rank_threshold(ALPHA, sub.n() + 1);
        let k_max = (tau - sub.tie_break_uniform(&test)).floor() as usize;
        let tail = sub.count_tail(&test, k_max).unwrap();
        assert!(tail <= d.y.sum(), "tail {tail} is not on the data scale");
        let set = sub.prediction_set(&x_star, 0.0, ALPHA).unwrap();
        for z in 0..=(tail as usize + 5) {
            let z = z as f64;
            assert_eq!(contains(&set, z), brute_force_member(&sub, &x_star, 0.0, z, ALPHA), "z={z}");
        }
    }

    #[test]
    fn rank_threshold_at_a_decimal_level_is_the_intended_integer() {
        // 1 − 0.9 is 0.0999…98 in binary, so the raw product is below 6 and
        // would admit the sixth-smallest rank as well.
        assert!((1.0 - 0.9) * 60.0 < 6.0);
        assert_eq!(conformal_rank_threshold(1.0 - 0.9, 60), 6.0);
        assert_eq!(conformal_rank_threshold(1.0 - 0.95, 100), 5.0);
        assert_eq!(conformal_rank_threshold(0.1, 51), 0.1 * 51.0);
    }

    #[test]
    fn gamma_set_is_a_tight_superset_of_the_exact_set() {
        let mut rng = StdRng::seed_from_u64(7011);
        let family = ConformalGlmFamily::GammaLog;
        // n = 50 leaves α(n+1) fractional; n = 59 at the level-derived
        // α = 1 − 0.9 puts it on the integer 6.
        for (n, alpha) in [(50, ALPHA), (50, ALPHA), (59, 1.0 - 0.9), (59, 1.0 - 0.9)] {
            let d = data(family, n, &mut rng);
            let sub = substrate(family, &d);
            let x = rng.random::<f64>() * 2.0 - 1.0;
            let set = sub.prediction_set(&row(x), 0.0, alpha).unwrap();
            let hi = set.intervals.last().unwrap().hi;
            assert!(hi.is_finite(), "gamma set must be bounded: {:?}", set.intervals);
            for k in 1..400 {
                let z = hi * 1.5 * k as f64 / 400.0;
                let exact = brute_force_member(&sub, &row(x), 0.0, z, alpha);
                if exact {
                    assert!(contains(&set, z), "exact member {z} missing from {:?}", set.intervals);
                }
            }
            // Tightness: every endpoint is an exact boundary to solver accuracy.
            for iv in &set.intervals {
                for (edge, inside) in [(iv.lo, iv.lo * (1.0 + 1e-6)), (iv.hi, iv.hi * (1.0 - 1e-6))] {
                    if edge > 0.0 {
                        assert!(
                            brute_force_member(&sub, &row(x), 0.0, inside, alpha),
                            "gamma endpoint {edge} is not an exact boundary"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn labeled_refit_reproduces_the_penalized_fit() {
        let mut rng = StdRng::seed_from_u64(99);
        for family in FAMILIES {
            let d = data(family, 80, &mut rng);
            let sub = substrate(family, &d);
            let beta = sub.refit_labeled_rows().unwrap();
            let zero = Array1::zeros(3);
            let st = sub.state(&beta, &TestRow { x: &zero, offset: 0.0 }, Augmentation::Tilt(0.0));
            assert!(vec_norm(&st.grad) < 1e-8 * st.natural_scale, "{family:?}");
        }
    }

    #[test]
    fn penalty_recovery_zeros_unpenalized_columns_and_is_psd() {
        let gram = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]).unwrap();
        let mut normal = gram.clone();
        normal[[1, 1]] += 2.0;
        normal[[2, 2]] += 1.0;
        normal[[1, 2]] += 0.5;
        normal[[2, 1]] += 0.5;
        normal[[0, 0]] += 1e-15;
        let s = penalty_from_normal_and_gram(&normal, &gram, &[1..3], 2.0).unwrap();
        assert!(s.row(0).iter().all(|&v| v == 0.0) && s.column(0).iter().all(|&v| v == 0.0));
        assert!((s[[1, 1]] - 4.0).abs() < 1e-12 && (s[[1, 2]] - 1.0).abs() < 1e-12);
        let mut indefinite = gram.clone();
        indefinite[[1, 1]] -= 1e-13;
        let s = penalty_from_normal_and_gram(&indefinite, &gram, &[1..3], 1.0).unwrap();
        let (evals, _) = s.eigh(Side::Lower).unwrap();
        assert!(evals.iter().all(|&v| v >= -1e-15));
    }

    /// Seeded Monte Carlo: marginal coverage of the certified set at n = 99,
    /// α = 0.1 (so α(n+1) is an integer and the target is exactly 0.9) is
    /// within two Monte Carlo standard errors of 1 − α, two-sided.
    #[test]
    fn monte_carlo_coverage_is_nominal_for_every_family() {
        let reps = 1000;
        let n = 99;
        for (k, family) in FAMILIES.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(4242 + k as u64);
            let mut covered = 0usize;
            for _ in 0..reps {
                let d = data(family, n, &mut rng);
                let x = rng.random::<f64>() * 2.0 - 1.0;
                let o = rng.random::<f64>() * 0.4 - 0.2;
                let y_star = draw(family, eta_true(x) + o, &mut rng);
                let set = substrate(family, &d).prediction_set(&row(x), o, ALPHA).unwrap();
                covered += usize::from(contains(&set, y_star));
            }
            let cov = covered as f64 / reps as f64;
            let target = 1.0 - ALPHA;
            let mcse = (target * ALPHA / reps as f64).sqrt();
            assert!(
                (cov - target).abs() <= 2.0 * mcse,
                "{family:?}: coverage {cov} vs {target} ± {}",
                2.0 * mcse
            );
        }
    }
}
