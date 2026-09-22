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
//! the n training responses and not the test response. A row of a fit that
//! selected a smoothing parameter (except the one-strength map below) or a
//! negative-binomial θ therefore reports
//! a typed refusal certificate: the numerical enclosure uses a frozen map
//! without a guarantee for training-selected strengths. The one-strength map
//! below re-selects its strength; see
//! [`ConformalGlmFamily::certificate`].
//!
//! # Strength re-selection, and the level walk it forces
//!
//! With one selected smoothing strength, each candidate level uses the
//! augmented-data LAML objective at a unit-Frobenius penalty shape. Scaling
//! the stored penalty changes neither that shape nor the deterministic zero
//! coefficient start. The shared outer solver certifies a local solution in
//! the augmented Gram's resolvability domain; this is not a proof of global
//! LAML minimization. Selection failures propagate as errors, without a
//! frozen-fit replacement. The score comparison remains a conservative
//! numerical enclosure, now for the re-selecting fitting map.
//!
//! Re-selection applies to the families whose likelihood carries no nuisance
//! parameter beside the strength — Bernoulli-logit and Poisson-log. The Gamma
//! dispersion and the negative-binomial θ are themselves estimated from the
//! responses, so re-selecting the strength alone would move the asymmetry
//! rather than remove it, and those rows keep their frozen-penalty refusal.
//!
//! One substrate per level is what the honest map means, and it is why the
//! candidate support below is walked level by level for the counts rather than
//! bisected in the score coordinate. A frozen penalty makes `z ↦ u_*` monotone,
//! which is what lets one tilt solve decide a whole run of counts; give each
//! level its own `ρ̂(z)` and the counts no longer share a score map, so neither
//! the bracket nor the monotonicity survives. The walk that replaces it, and
//! the ρ-free tail that closes it, are in
//! [`GlmFullConformalSubstrate::honest_count_levels`]. Scaling the penalty
//! leaves its zero rows and columns zero, so the unpenalised intercept that
//! tail argument needs is the same column at every strength.
//!
//! Symmetry requires a fixed basis and penalty shape that treat all augmented
//! rows symmetrically. Removing the fitted strength cannot make a learned,
//! training-only basis or penalty shape symmetric. Coverage statements are
//! marginal under exchangeability, not conditional on the test features.
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
//! * Bernoulli: both levels `{0, 1}` are tested, each at its own `ρ̂(z)` when
//!   the map re-selects.
//! * Poisson under the honest map: the levels `0, 1, 2, …` are walked directly,
//!   each at its own `ρ̂(z)`, and the walk closes on the ρ-free score tail
//!   below (see [`GlmFullConformalSubstrate::honest_count_levels`]).
//! * Poisson (frozen) and NB: the counts are walked in the test-score coordinate, from
//!   the score of the count `0` up to a tail score derived from the data,
//!   beyond which no count can be in the set:
//!   - NB: `u ∈ (−θ, y)`, so once the test score reaches `max(max y, θ)` no
//!     training score can match it;
//!   - Poisson and NB (with an unpenalised intercept): the intercept KKT row
//!     `Σ u_i + u_* = 0` and `u_i < y_i` bound the number of training scores at
//!     least `s` by `1 + Σy/s`, which drops below the rank the set needs once
//!     `s ≥ max(max y, Σy/K)`. NB takes the smaller of its two tails, which
//!     keeps the enumeration short when a near-Poisson fit estimates a large
//!     `θ`.
//!   The score interval is bisected as for Gamma below, one certified solve
//!   deciding a whole sub-interval, and a run of member pieces covers the
//!   counts strictly inside the images of its end scores. The counts inside an
//!   end image's slack, inside an undecided sliver, or tied with a training
//!   row are each decided by their own solve, so the work follows the set's
//!   boundaries rather than the width of the count range. Without a provable
//!   tail the set is honestly `[0, ∞)`.
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
//! # Independent randomization and numerical ties
//!
//! All families use an independent `U ~ Uniform[0, 1)` drawn once per
//! inversion and shared by all candidate labels. Exact smoothed ranks have
//! marginal coverage `1 − α` for exchangeable supplied rows and a fitting map
//! symmetric in all augmented rows. This is not a conditional-on-features
//! guarantee; a training-only learned basis or penalty need not be symmetric.
//! Numerical uncertainty is retained as a conservative enclosure, so this
//! implementation does not claim exact coverage. Gamma uses the same independent-U
//! threshold with conservative tie bounds. Fixed-U entry points support reproducible
//! tests without deriving randomization from the observations.

use std::ops::Range;

use faer::Side;
use ndarray::{Array1, Array2, Axis};
use rand::RngExt;

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_atv, fast_av, fast_xt_diag_x};
use gam_math::special::{logistic as sigmoid, softplus};
use gam_problem::types::LikelihoodSpec;
use gam_spec::FamilySpecKind;
use opt::{BacktrackConfig, backtracking_line_search};

use super::full_conformal::{
    ConformalCertificate, ConformalInterval, ConformalRefusal, conformal_rank_threshold,
    validate_tie_uniform,
};

/// Maximum damped-Newton iterations for a cold augmented GLM fit.
const GLM_NEWTON_MAX_ITERS: usize = 200;

/// Maximum Armijo backtracking halvings per cold Newton iteration.
const GLM_NEWTON_MAX_BACKTRACKS: usize = 60;

/// Strict scale-invariant KKT tolerance declaring convergence, applied to the
/// RAW penalized gradient (dimension-scaled OR natural-scale relative — the
/// same certificate the main P-IRLS solver uses). NOT a tolerance on the
/// preconditioned Newton step.
const GLM_CONVERGENCE_RTOL: f64 = 1e-12;

/// Armijo sufficient-decrease constant for the cold-fit line search —
/// sourced from the shared optimizer constants so the workspace has exactly
/// one `c₁`.
const GLM_ARMIJO_C1: f64 = opt::constants::ARMIJO_C1;

#[inline]
fn vec_norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

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

    /// What this arm's set guarantees for a fit that selected `penalty_count`
    /// smoothing parameters (`None` for a payload that did not record it).
    ///
    /// With no smoothing parameter the frozen-penalty fitting map selects
    /// nothing from the responses (the Gamma dispersion does not enter an
    /// unpenalized fit, and the score is dispersion-free), so the numerical set is
    /// a conservative enclosure. One selected strength is re-selected on the
    /// augmented rows for the two families whose likelihood has no nuisance
    /// parameter beside it — Bernoulli and Poisson. The Gamma dispersion and the
    /// negative-binomial θ are themselves estimated from the responses, so
    /// re-selecting the strength alone would move the asymmetry rather than
    /// remove it; those rows keep
    /// [`ConformalRefusal::GlmFrozenPenalty`], and more than one selected
    /// strength keeps [`ConformalRefusal::MultiPenalty`].
    pub fn certificate(self, penalty_count: Option<usize>) -> ConformalCertificate {
        match (self, penalty_count) {
            (_, None) => ConformalCertificate::Refused(ConformalRefusal::UnknownPenaltyStructure),
            (Self::NegativeBinomialLog { .. }, _) => {
                ConformalCertificate::Refused(ConformalRefusal::GlmFrozenPenalty)
            }
            (_, Some(0)) => ConformalCertificate::ConservativeFrozen,
            (Self::BernoulliLogit | Self::PoissonLog, Some(1)) => ConformalCertificate::HonestRefit,
            (Self::BernoulliLogit | Self::PoissonLog, Some(_)) => {
                ConformalCertificate::Refused(ConformalRefusal::MultiPenalty)
            }
            (_, Some(_)) => ConformalCertificate::Refused(ConformalRefusal::GlmFrozenPenalty),
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

    /// Sum of the nonnegative operands of the score, before cancellation.
    fn score_operand_scale(self, eta: f64, y: f64) -> f64 {
        match self {
            Self::BernoulliLogit => y + sigmoid(eta),
            Self::PoissonLog => y + eta.exp(),
            Self::NegativeBinomialLog { theta } => {
                let t = eta - theta.ln();
                y * sigmoid(-t) + theta * sigmoid(t)
            }
            Self::GammaLog => y * (-eta).exp() + 1.0,
        }
    }

    /// Response-scale mean `μ(η)`.
    pub fn mean(self, eta: f64) -> f64 {
        match self {
            Self::BernoulliLogit => sigmoid(eta),
            _ => eta.exp(),
        }
    }

    /// The IRLS curvature `w = ∂²ℓ/∂η²` and its first two `η`-derivatives —
    /// the only family-specific content of the honest map's Laplace-REML jet
    /// ([`GlmFullConformalSubstrate::laml_jet`]).
    ///
    /// Bernoulli-logit: `w = σ(η)σ(−η)`, `w′ = w(1 − 2μ)`, `w″ = w(1 − 6w)`.
    /// Poisson-log: `w = μ = e^η`, which is its own derivative, so
    /// `w = w′ = w″ = μ`.
    ///
    /// `None` for the families the honest map does not build
    /// ([`Self::certificate`] refuses them before a jet is ever asked for), so
    /// this returns a curvature only where one is used rather than carrying an
    /// unreachable arm.
    fn weight_jet(self, eta: f64) -> Option<(f64, f64, f64)> {
        match self {
            Self::BernoulliLogit => {
                let w = sigmoid(eta) * sigmoid(-eta);
                let mu = sigmoid(eta);
                Some((w, w * (1.0 - 2.0 * mu), w * (1.0 - 6.0 * w)))
            }
            Self::PoissonLog => {
                let mu = eta.exp();
                Some((mu, mu, mu))
            }
            Self::NegativeBinomialLog { .. } | Self::GammaLog => None,
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

    /// An enclosure of the responses at `η ∈ [eta_lo, eta_hi]` and `s ∈
    /// [s_lo, s_hi]` that also holds in floating point: the computed ends are
    /// widened by a rounding bound on `exp`, the products and the sums, so a
    /// count sitting exactly on an end (the count `0` at the score of `0`)
    /// is never rounded out of it.
    fn response_enclosure(self, eta_lo: f64, eta_hi: f64, s_lo: f64, s_hi: f64) -> (f64, f64) {
        let pad = |eta: f64, s: f64| {
            let mu = eta.exp();
            let scale = match self {
                Self::NegativeBinomialLog { theta } => mu + s.abs() * (theta + mu) / theta,
                Self::GammaLog => mu * (1.0 + s.abs()),
                _ => mu + s.abs(),
            };
            16.0 * f64::EPSILON * scale * (1.0 + eta.abs())
        };
        (
            self.response_of_score(eta_lo, s_lo) - pad(eta_lo, s_lo),
            self.response_of_score(eta_hi, s_hi) + pad(eta_hi, s_hi),
        )
    }

    /// The whole response support, returned when no candidate can be excluded.
    fn whole_support(self) -> Vec<ConformalInterval> {
        let hi = match self {
            Self::BernoulliLogit => 1.0,
            _ => f64::INFINITY,
        };
        vec![ConformalInterval::closed(0.0, hi)]
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
    /// The validated fitting-map certificate for this inversion.
    pub certificate: ConformalCertificate,
}

/// The single smoothing strength the honest map re-selects per candidate level.
#[derive(Clone, Debug)]
struct Reselection {
    /// The fitted penalty at unit Frobenius norm, `S = Sλ/‖Sλ‖_F`; the map
    /// fits at `e^ρ S`.
    unit: Array2<f64>,
    /// `rank(S)`, counted at the REML engine's positive-eigenvalue threshold.
    rank: usize,
    /// `XᵀX` of the labeled rows; the test row adds `x_* x_*ᵀ`.
    gram: Array2<f64>,
}

/// One Laplace-REML evaluation of the honest map at `ρ`.
struct LamlJet {
    value: f64,
    gradient: f64,
    hessian: f64,
    beta: Array1<f64>,
}

/// Maximal runs of consecutive integers among the increasing `levels`.
fn level_runs(levels: impl IntoIterator<Item = f64>) -> Vec<ConformalInterval> {
    let mut runs = Vec::<ConformalInterval>::new();
    for z in levels {
        match runs.last_mut() {
            Some(last) if last.hi + 1.0 == z => last.hi = z,
            _ => runs.push(ConformalInterval::closed(z, z)),
        }
    }
    runs
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
    certificate: ConformalCertificate,
    reselection: Option<Reselection>,
}

impl GlmFullConformalSubstrate {
    /// `x` and `offset` are the labeled rows' design and offsets, `y` their
    /// responses, `s_lambda` the frozen penalty in unit-dispersion units (it
    /// must be positive semidefinite), `penalty_count` declares how many strengths
    /// were selected from the training responses, and `warm_start` gives the fitted
    /// coefficients, the Newton starting point of every augmented solve.
    pub fn new(
        family: ConformalGlmFamily,
        x: Array2<f64>,
        y: Array1<f64>,
        offset: Array1<f64>,
        s_lambda: Array2<f64>,
        penalty_count: Option<usize>,
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
        if x.iter()
            .chain(offset.iter())
            .chain(s_lambda.iter())
            .any(|v| !v.is_finite())
        {
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
        if warm_start.iter().any(|v| !v.is_finite()) {
            return Err("full conformal: warm-start coefficients must be finite".into());
        }
        let mut certificate = family.certificate(penalty_count);
        let mut reselection = None;
        if certificate == ConformalCertificate::HonestRefit {
            let largest = s_lambda.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            if largest == 0.0 {
                certificate = ConformalCertificate::ConservativeFrozen;
            } else {
                // Scale before squaring: a finite penalty of any magnitude
                // must not become an infinite norm or a false zero penalty.
                let scaled = s_lambda.mapv(|v| v / largest);
                if s_lambda
                    .iter()
                    .zip(scaled.iter())
                    .any(|(&original, &value)| original != 0.0 && value == 0.0)
                {
                    return Err(
                        "honest conformal: penalty normalization loses a nonzero entry".into(),
                    );
                }
                let norm = scaled.iter().map(|v| v * v).sum::<f64>().sqrt();
                let unit = scaled.mapv(|v| v / norm);
                if scaled
                    .iter()
                    .zip(unit.iter())
                    .any(|(&scaled, &value)| scaled != 0.0 && value == 0.0)
                {
                    return Err(
                        "honest conformal: unit penalty normalization loses a nonzero entry".into(),
                    );
                }
                for a in 0..p {
                    for b in 0..a {
                        if s_lambda[[a, b]] != s_lambda[[b, a]] {
                            return Err("honest conformal: penalty must be symmetric".into());
                        }
                    }
                }
                let (evals, _) = unit.eigh(Side::Lower).map_err(|e| {
                    format!("honest conformal: penalty eigendecomposition failed: {e:?}")
                })?;
                let evals = evals.to_vec();
                let threshold =
                    gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(
                        &evals,
                    );
                if evals.iter().any(|&v| !v.is_finite() || v < -threshold) {
                    return Err(
                        "honest conformal: penalty is not resolved positive semidefinite".into(),
                    );
                }
                let rank = evals.iter().filter(|&&v| v > threshold).count();
                if rank == 0 {
                    return Err("honest conformal: nonzero penalty rank is unresolved".into());
                }
                reselection = Some(Reselection {
                    unit,
                    rank,
                    gram: x.t().dot(&x),
                });
            }
        }
        Ok(Self {
            family,
            x,
            xt,
            y,
            offset,
            s_lambda,
            warm_start,
            intercept,
            certificate,
            reselection,
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
        self.prediction_set_with_uniform(x_star, offset_star, alpha, rand::rng().random())
    }

    /// Invert with one externally supplied independent uniform variate.
    /// Reusing this value across candidates defines a coherent randomized set.
    pub fn prediction_set_with_uniform(
        &self,
        x_star: &Array1<f64>,
        offset_star: f64,
        alpha: f64,
        tie_uniform: f64,
    ) -> Result<GlmFullConformalSet, String> {
        validate_tie_uniform(tie_uniform)?;
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(format!(
                "full conformal: alpha must be in (0, 1), got {alpha}"
            ));
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
        let intervals = if let Some(reselection) = &self.reselection {
            self.honest_discrete_set(reselection, &row, tau, tie_uniform)?
        } else if self.family.is_discrete() {
            self.discrete_set(&row, tau, tie_uniform)
        } else {
            self.continuous_set(&row, tau, tie_uniform)
        };
        Ok(GlmFullConformalSet {
            intervals,
            alpha,
            n_augmented: self.n() + 1,
            certificate: self.certificate,
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
        let training_scale: f64 = (0..n)
            .map(|i| {
                self.x
                    .row(i)
                    .iter()
                    .fold(0.0_f64, |norm, value| norm.hypot(*value))
                    * self
                        .family
                        .score_operand_scale(eta[i] + self.offset[i], self.y[i])
            })
            .sum();
        let test_scale = match aug {
            Augmentation::Response(z) => self.family.score_operand_scale(eta_star, z),
            Augmentation::Tilt(s) => s.abs(),
        };
        let natural_scale = 1.0 + training_scale + vec_norm(&s_beta) + vec_norm(row.x) * test_scale;
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
        g_norm.is_finite()
            && state.natural_scale.is_finite()
            && (g_norm < GLM_CONVERGENCE_RTOL * dimension_scale
                || g_norm / state.natural_scale < GLM_CONVERGENCE_RTOL)
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

    /// The discrete set with randomised ties: both Bernoulli levels by their
    /// own solves, the counts by [`Self::count_set`].
    fn discrete_set(&self, row: &TestRow<'_>, tau: f64, u_tie: f64) -> Vec<ConformalInterval> {
        if tau < u_tie {
            return self.family.whole_support();
        }
        let k_max = (tau - u_tie).floor() as usize;
        let twin = self.twin_rows(row);
        if self.family != ConformalGlmFamily::BernoulliLogit {
            return self.count_set(row, tau, u_tie, k_max, &twin);
        }
        level_runs(
            [0.0, 1.0]
                .into_iter()
                .filter(|&z| self.count_member(row, z, u_tie, &twin, tau)),
        )
    }

    /// The training rows that share the test row's covariates and offset.
    fn twin_rows(&self, row: &TestRow<'_>) -> Vec<bool> {
        (0..self.n())
            .map(|i| {
                self.offset[i] == row.offset
                    && self.x.row(i).iter().zip(row.x.iter()).all(|(a, b)| a == b)
            })
            .collect()
    }

    /// The discrete set of the honest map: each candidate level `z` is fitted at
    /// the strength `ρ̂(z)` the augmented data select, and its rank is decided by
    /// the certified solve at `e^{ρ̂(z)} S`. Returns the reason when a selection
    /// does not complete; there is no frozen-fit substitution.
    ///
    /// Bernoulli has two levels and walks both. The count families walk
    /// `z = 0, 1, 2, …`; [`Self::honest_count_levels`] says why a level walk
    /// replaces the frozen arm's score-space bisection and what closes it.
    fn honest_discrete_set(
        &self,
        reselection: &Reselection,
        row: &TestRow<'_>,
        tau: f64,
        u_tie: f64,
    ) -> Result<Vec<ConformalInterval>, String> {
        if tau < u_tie {
            return Ok(self.family.whole_support());
        }
        let twin = self.twin_rows(row);
        if self.family == ConformalGlmFamily::BernoulliLogit {
            let mut kept = Vec::with_capacity(2);
            for z in [0.0, 1.0] {
                if self.honest_level(reselection, row, z, tau, u_tie, &twin)?.0 {
                    kept.push(z);
                }
            }
            return Ok(level_runs(kept));
        }
        self.honest_count_levels(reselection, row, tau, u_tie, &twin)
    }

    /// One level of the honest walk: the strength `ρ̂(z)` the augmented rows
    /// select, the membership the certified solve at `e^{ρ̂(z)} S` decides, and
    /// that solve's certified lower bound on the test score `|u_*|`.
    fn honest_level(
        &self,
        reselection: &Reselection,
        row: &TestRow<'_>,
        z: f64,
        tau: f64,
        u_tie: f64,
        twin: &[bool],
    ) -> Result<(bool, f64), String> {
        let (rho, beta) = self.select_strength(reselection, row, z)?;
        let refit = self.at_strength(reselection, rho, beta)?;
        Ok(refit.count_member_scored(row, z, u_tie, twin, tau))
    }

    /// The honest count set: one substrate per level.
    ///
    /// # Why the frozen arm's walk cannot carry re-selection
    ///
    /// [`Self::count_set`] bisects the test-SCORE coordinate with
    /// `Augmentation::Tilt(s)` and maps score intervals back to count runs. That
    /// construction is a property of ONE substrate: a single frozen penalty is
    /// what makes `z ↦ u_*` monotone, which is what gives a score interval a
    /// count image and lets one solve decide a whole run. Re-selecting `ρ̂(z)`
    /// gives every level its own penalty and its own score map, so neither the
    /// bracket nor the monotonicity survives. The honest set is walked level by
    /// level instead, one certified outer selection and one certified solve each.
    ///
    /// # What closes the walk
    ///
    /// At an augmented fit whose intercept column is unpenalised the intercept
    /// KKT row reads `Σ_i u_i + u_* = 0`, and for the count families
    /// `u_i = y_i − μ_i < y_i` because `μ_i > 0`. Neither statement mentions the
    /// penalty, so [`Self::count_score_tail`]'s bound — the number of training
    /// rows whose score reaches `s` is at most `1 + Σy/s`, which falls below the
    /// rank the set needs once `s ≥ s_top` — holds at EVERY strength. A level
    /// whose own certified fit puts `|u_*|` at or past `s_top` is therefore a
    /// non-member whatever strength that fit selected, and the walk stops there.
    ///
    /// # What the walk does not prove, and what it does instead of guessing
    ///
    /// The frozen arm also knows that no LARGER level returns, because
    /// `du_*/dz > 0` at a fixed penalty. That is exactly what re-selection
    /// removes: levels above the stop choose their own `ρ̂`, and this walk proves
    /// nothing about them. So it never truncates. It is bounded by `z_cap`, the
    /// first level the FROZEN substrate's own score carries to `s_top` — one
    /// tilt solve, since `z(s)` is increasing there — and a walk that reaches
    /// `z_cap` with the set still open returns the whole support `[0, ∞)`, the
    /// same conservative answer this module already gives where no tail is
    /// provable. A too-wide enclosure keeps coverage at or above `1 − α`; a
    /// truncated one would not, which is the defect being repaired.
    fn honest_count_levels(
        &self,
        reselection: &Reselection,
        row: &TestRow<'_>,
        tau: f64,
        u_tie: f64,
        twin: &[bool],
    ) -> Result<Vec<ConformalInterval>, String> {
        let whole = || self.family.whole_support();
        let k_max = (tau - u_tie).floor() as usize;
        let Some(s_top) = self.count_score_tail(row, k_max) else {
            return Ok(whole());
        };
        let Ok(cap_node) = self.solve(row, Augmentation::Tilt(s_top), &self.warm_start) else {
            return Ok(whole());
        };
        if !cap_node.certifies(cap_node.error) {
            return Ok(whole());
        }
        let slack = cap_node.lever_star * cap_node.error;
        let (_, z_cap) = self.family.response_enclosure(
            cap_node.eta_star - slack,
            cap_node.eta_star + slack,
            s_top,
            s_top,
        );
        if !(z_cap.is_finite() && z_cap >= 0.0) {
            return Ok(whole());
        }
        let z_cap = z_cap.ceil();
        let mut kept = Vec::new();
        let mut z = 0.0_f64;
        loop {
            let (member, score) = self.honest_level(reselection, row, z, tau, u_tie, twin)?;
            if member {
                kept.push(z);
            } else if score >= s_top {
                return Ok(level_runs(kept));
            }
            if z >= z_cap {
                return Ok(whole());
            }
            z += 1.0;
        }
    }

    /// This substrate at the penalty `e^ρ S`, warm-started at `beta`.
    fn at_strength(
        &self,
        reselection: &Reselection,
        rho: f64,
        beta: Array1<f64>,
    ) -> Result<Self, String> {
        let mut refit = self.clone();
        let strength = gam_problem::checked_exp_log_strength(rho)
            .map_err(|e| format!("{} honest conformal strength: {e}", self.family.name()))?;
        refit.s_lambda = reselection.unit.mapv(|v| strength * v);
        refit.warm_start = beta;
        refit.reselection = None;
        Ok(refit)
    }

    /// `ρ̂(z)`: a certified local Laplace-REML solution for the augmented rows in the
    /// #2812 resolvability domain of their Gram against `S`, through the outer
    /// engine from its own start, with the fitted coefficients at that optimum.
    ///
    /// The domain is a property of the augmented Gram and the penalty shape, so
    /// it is the same interval at every candidate level; only the criterion
    /// moves with `z`.
    fn select_strength(
        &self,
        reselection: &Reselection,
        row: &TestRow<'_>,
        z: f64,
    ) -> Result<(f64, Array1<f64>), String> {
        use gam_problem::{Derivative, HessianValue, OuterEval};
        use gam_solve::estimate::EstimationError;
        use gam_solve::rho_optimizer::OuterProblem;

        let p = self.p();
        let mut gram = reselection.gram.clone();
        for a in 0..p {
            for b in 0..p {
                gram[[a, b]] += row.x[a] * row.x[b];
            }
        }
        let spectrum = gam_solve::estimate::rho_domain::penalty_range_gammas_from_gram(
            &gram,
            &reselection.unit,
        )
        .ok_or_else(|| {
            format!(
                "{} honest conformal: augmented penalty spectrum is unresolved",
                self.family.name()
            )
        })?;
        let interval = gam_solve::estimate::rho_domain::resolvability_interval(&spectrum)
            .ok_or_else(|| {
                format!(
                    "{} honest conformal: smoothing domain is unresolved",
                    self.family.name()
                )
            })?;
        let (lower, upper) =
            gam_solve::estimate::rho_domain::coordinate_domain(Some(interval), None);
        let context = format!("{} honest full conformal at z={z}", self.family.name());
        let refuse = |reason: String| EstimationError::TrialPointRefused { reason };
        let problem = OuterProblem::new(1)
            .with_problem_size(self.n() + 1, p)
            .with_gradient(Derivative::Analytic)
            .with_hessian(gam_problem::DeclaredHessianForm::Dense)
            .with_bounds(Array1::from_elem(1, lower), Array1::from_elem(1, upper));
        let mut objective = problem.build_objective(
            Array1::<f64>::zeros(p),
            |warm: &mut Array1<f64>, rho: &Array1<f64>| {
                let jet = self
                    .laml_jet(reselection, row, z, rho[0], warm)
                    .map_err(refuse)?;
                *warm = jet.beta;
                Ok(jet.value)
            },
            |warm: &mut Array1<f64>, rho: &Array1<f64>| {
                let jet = self
                    .laml_jet(reselection, row, z, rho[0], warm)
                    .map_err(refuse)?;
                *warm = jet.beta;
                Ok(OuterEval {
                    cost: jet.value,
                    gradient: Array1::from_vec(vec![jet.gradient]),
                    hessian: HessianValue::Dense(Array2::from_elem((1, 1), jet.hessian)),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut Array1<f64>)>,
            None::<
                fn(&mut Array1<f64>, &Array1<f64>) -> Result<gam_problem::EfsEval, EstimationError>,
            >,
        );
        let result = problem
            .run_certified(&mut objective, &context)
            .map_err(|e| format!("{context}: {e}"))?;
        let rho = result.rho()[0];
        let jet = self.laml_jet(reselection, row, z, rho, &Array1::zeros(p))?;
        Ok((rho, jet.beta))
    }

    /// The Laplace-REML criterion of the `n + 1` augmented rows at
    /// `Sρ = e^ρ S`, with its first two ρ-derivatives in closed form:
    ///
    /// ```text
    ///   V(ρ) = Σ_j ℓ(η_j; y_j) + ½ β̂ᵀSρβ̂ + ½ ln|H| − ½ rank(S)·ρ,
    ///   H    = X_aᵀ W X_a + Sρ,
    /// ```
    ///
    /// `β̂` the certified augmented fit. With `β̇ = −H⁻¹Sρβ̂`, `η̇ = X_aβ̇`,
    /// `Ḣ = Sρ + X_aᵀ diag(w′η̇) X_a`, `l_j² = x_jᵀH⁻¹x_j` and the family's own
    /// curvature derivatives `w′`, `w″` from
    /// [`ConformalGlmFamily::weight_jet`] (the only family-specific content
    /// here; `ℓ` is already generic):
    ///
    /// ```text
    ///   V′ = ½ β̂ᵀSρβ̂ + ½ tr(H⁻¹Ḣ) − ½ rank,
    ///   V″ = ½ β̂ᵀSρβ̂ + β̂ᵀSρβ̇
    ///        + ½ [tr(H⁻¹Ḧ) − tr((H⁻¹Ḣ)²)],
    ///   Ḧ  = Sρ + X_aᵀ diag(w″η̇² + w′η̈) X_a,   η̈ = X_aβ̈,
    ///   β̈  = −H⁻¹(Ḣβ̇ + Sρβ̂ + Sρβ̇).
    /// ```
    fn laml_jet(
        &self,
        reselection: &Reselection,
        row: &TestRow<'_>,
        z: f64,
        rho: f64,
        warm: &Array1<f64>,
    ) -> Result<LamlJet, String> {
        let n = self.n();
        let p = self.p();
        let family = self.family.name();
        let lambda = gam_problem::checked_exp_log_strength(rho)
            .map_err(|error| format!("{family} honest full conformal: {error}"))?;
        let s_rho = reselection.unit.mapv(|v| lambda * v);
        let mut refit = self.clone();
        refit.s_lambda = s_rho.clone();
        refit.reselection = None;
        let node = refit.solve(row, Augmentation::Response(z), warm)?;
        if !node.certifies(node.error) {
            return Err(format!(
                "{family} honest full conformal: the augmented fit at ρ={rho} did not certify"
            ));
        }
        let beta = node.beta;
        let mut x_aug = Array2::<f64>::zeros((n + 1, p));
        x_aug.slice_mut(ndarray::s![..n, ..]).assign(&self.x);
        x_aug.row_mut(n).assign(row.x);
        let mut offset_aug = Array1::<f64>::zeros(n + 1);
        offset_aug.slice_mut(ndarray::s![..n]).assign(&self.offset);
        offset_aug[n] = row.offset;
        let mut y_aug = Array1::<f64>::zeros(n + 1);
        y_aug.slice_mut(ndarray::s![..n]).assign(&self.y);
        y_aug[n] = z;
        let eta = fast_av(&x_aug, &beta) + &offset_aug;
        let mut w = Array1::<f64>::zeros(n + 1);
        let mut w1 = Array1::<f64>::zeros(n + 1);
        let mut w2 = Array1::<f64>::zeros(n + 1);
        for j in 0..=n {
            let (curvature, first, second) = self.family.weight_jet(eta[j]).ok_or_else(|| {
                format!("{family} honest full conformal: this family has no re-selecting map")
            })?;
            w[j] = curvature;
            w1[j] = first;
            w2[j] = second;
        }

        let h = fast_xt_diag_x(&x_aug, &w) + &s_rho;
        let chol = h.cholesky(Side::Lower).map_err(|e| {
            format!("{family} honest full conformal: penalized Hessian not SPD: {e:?}")
        })?;
        let s_beta = s_rho.dot(&beta);
        let penalty = beta.dot(&s_beta);
        let mut value = 0.5 * penalty + chol.diag().iter().map(|d| d.ln()).sum::<f64>()
            - 0.5 * reselection.rank as f64 * rho;
        for j in 0..=n {
            value += self.family.nll(eta[j], y_aug[j]);
        }

        let d_beta = chol.solvevec(&s_beta).mapv(|v| -v);
        let d_eta = fast_av(&x_aug, &d_beta);
        let h_dot = fast_xt_diag_x(&x_aug, &(&w1 * &d_eta)) + &s_rho;
        let m = chol.solve_mat(&h_dot);
        let trace_m: f64 = (0..p).map(|a| m[[a, a]]).sum();
        let trace_m2: f64 = (0..p)
            .flat_map(|a| (0..p).map(move |b| (a, b)))
            .map(|(a, b)| m[[a, b]] * m[[b, a]])
            .sum();
        let gradient = 0.5 * penalty + 0.5 * trace_m - 0.5 * reselection.rank as f64;

        let rhs = h_dot.dot(&d_beta) + &s_beta + s_rho.dot(&d_beta);
        let dd_beta = chol.solvevec(&rhs).mapv(|v| -v);
        let dd_eta = fast_av(&x_aug, &dd_beta);
        let h_ddot = fast_xt_diag_x(&x_aug, &(&w2 * &d_eta * &d_eta + &w1 * &dd_eta)) + &s_rho;
        let trace_ddot: f64 = {
            let q = chol.solve_mat(&h_ddot);
            (0..p).map(|a| q[[a, a]]).sum()
        };
        let hessian = 0.5 * penalty + d_beta.dot(&s_beta) + 0.5 * (trace_ddot - trace_m2);
        if !(value.is_finite() && gradient.is_finite() && hessian.is_finite()) {
            return Err(format!(
                "{family} honest full conformal: criterion is not finite at ρ={rho}"
            ));
        }
        Ok(LamlJet {
            value,
            gradient,
            hessian,
            beta,
        })
    }

    /// Exact membership of the response level `z` by its own certified solve,
    /// with the training rows that share the test row's covariates, offset and
    /// response counted as ties. A solve that fails or does not certify keeps
    /// `z`.
    fn count_member(&self, row: &TestRow<'_>, z: f64, u_tie: f64, twin: &[bool], tau: f64) -> bool {
        self.count_member_scored(row, z, u_tie, twin, tau).0
    }

    /// [`Self::count_member`] together with a certified LOWER bound on the test
    /// score `|u_*|` that the same solve reports, which the honest walk's tail
    /// test reads. The bound is `0` whenever the solve fails or does not
    /// certify — the value that closes nothing, matching the membership this
    /// returns in the same case.
    fn count_member_scored(
        &self,
        row: &TestRow<'_>,
        z: f64,
        u_tie: f64,
        twin: &[bool],
        tau: f64,
    ) -> (bool, f64) {
        let Ok(node) = self.solve(row, Augmentation::Response(z), &self.warm_start) else {
            return (true, 0.0);
        };
        if !node.certifies(node.error) {
            return (true, 0.0);
        }
        let t = node.score_star.abs();
        let err = 2.0 * node.weight_star * node.lever_star * node.error;
        let tied: Vec<bool> = (0..self.n()).map(|i| twin[i] && self.y[i] == z).collect();
        let ties = tied.iter().filter(|&&is_tied| is_tied).count();
        let self_weight = u_tie * (1 + ties) as f64;
        let member = self.verdict(
            &node,
            node.error,
            (t - err).max(0.0),
            t + err,
            self_weight,
            tau,
            Some(&tied),
        ) != Verdict::NonMember;
        (member, (t - err).max(0.0))
    }

    /// The test score past which no count can be in the set, from the
    /// provable tails of the module doc; `None` when no tail is provable.
    fn count_score_tail(&self, row: &TestRow<'_>, k_max: usize) -> Option<f64> {
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
        (tail_score > 0.0 && tail_score.is_finite()).then_some(tail_score)
    }

    /// Certified walk over the test-score coordinate for the count families.
    ///
    /// The candidate counts map increasingly onto test scores `s ∈ [s(0),
    /// s_tail]`, and the response fit at `z` is the tilt fit at `s = u_*(z)`,
    /// so the score interval is bisected as in [`Self::continuous_set`], one
    /// solve deciding a whole sub-interval. A run of member pieces covers the
    /// counts strictly inside the images of its end scores, each mapped by its
    /// own certified solve; the counts inside those images' slack, inside an
    /// undecided sliver, or tied with a training twin are decided by
    /// [`Self::count_member`]. The work follows the set's boundaries, not the
    /// width of the count range.
    fn count_set(
        &self,
        row: &TestRow<'_>,
        tau: f64,
        u_tie: f64,
        k_max: usize,
        twin: &[bool],
    ) -> Vec<ConformalInterval> {
        let whole = self.family.whole_support();
        let Some(s_top) = self.count_score_tail(row, k_max) else {
            return whole;
        };
        let Ok(base) = self.solve(row, Augmentation::Response(0.0), &self.warm_start) else {
            return whole;
        };
        if !base.certifies(base.error) {
            return whole;
        }
        let mut s_bot = base.score_star - 2.0 * base.weight_star * base.lever_star * base.error;
        // NB scores exceed `−θ`, where `z(s)` stops increasing in `η_*`.
        if let ConformalGlmFamily::NegativeBinomialLog { theta } = self.family {
            s_bot = s_bot.max(-theta);
        }
        if !(s_bot < s_top) {
            // Even the smallest count scores past the tail: the set is empty.
            return Vec::new();
        }
        #[derive(Clone, Copy, PartialEq)]
        enum Kind {
            Member,
            NonMember,
            Undecided,
            Unknown,
        }
        struct Leaf {
            kind: Kind,
            s_lo: f64,
            s_hi: f64,
            z_lo: f64,
            z_hi: f64,
            warm: Array1<f64>,
        }
        let mut leaves = Vec::<Leaf>::new();
        let mut stack = vec![(s_bot, s_top, base.beta)];
        while let Some((a, b, warm)) = stack.pop() {
            let m = 0.5 * (a + b);
            let half = 0.5 * (b - a);
            let splittable = m > a && m < b;
            let unknown = |warm| Leaf {
                kind: Kind::Unknown,
                s_lo: a,
                s_hi: b,
                z_lo: 0.0,
                z_hi: f64::INFINITY,
                warm,
            };
            let node = match self.solve(row, Augmentation::Tilt(m), &warm) {
                Ok(node) => node,
                Err(_) => {
                    leaves.push(unknown(warm));
                    continue;
                }
            };
            let e = node.error + 2.0 * half * node.lever_star;
            if !node.certifies(e) {
                // Narrower pieces shrink only the width term; a solve whose
                // own error does not certify gains nothing from a split.
                if splittable && node.certifies(node.error) {
                    stack.push((m, b, node.beta.clone()));
                    stack.push((a, m, node.beta));
                } else {
                    leaves.push(unknown(node.beta));
                }
                continue;
            }
            let (t_lo, t_hi) = if a <= 0.0 && 0.0 <= b {
                (0.0, a.abs().max(b.abs()))
            } else {
                (a.abs().min(b.abs()), a.abs().max(b.abs()))
            };
            let kind = match self.verdict(&node, e, t_lo, t_hi, u_tie, tau, None) {
                Verdict::Member => Kind::Member,
                Verdict::NonMember => Kind::NonMember,
                Verdict::Undecided => {
                    if splittable && half > node.score_floor() {
                        stack.push((m, b, node.beta.clone()));
                        stack.push((a, m, node.beta));
                        continue;
                    }
                    Kind::Undecided
                }
            };
            let slack = node.lever_star * e;
            let (z_lo, z_hi) =
                self.family
                    .response_enclosure(node.eta_star - slack, node.eta_star + slack, a, b);
            leaves.push(Leaf {
                kind,
                s_lo: a,
                s_hi: b,
                z_lo,
                z_hi,
                warm: node.beta,
            });
        }
        // The image `[z(s) lower, z(s) upper]` of one score by its own solve,
        // whose slack is the solve error alone.
        let image = |s: f64, warm: &Array1<f64>| -> Option<(f64, f64)> {
            let node = self.solve(row, Augmentation::Tilt(s), warm).ok()?;
            if !node.certifies(node.error) {
                return None;
            }
            let slack = node.lever_star * node.error;
            Some(
                self.family
                    .response_enclosure(node.eta_star - slack, node.eta_star + slack, s, s),
            )
        };
        // Counts `lo..=hi` of a real range, clipped to the support.
        let counts = |lo: f64, hi: f64| (lo.max(0.0).ceil(), hi.floor());
        let mut ranges = Vec::<ConformalInterval>::new();
        let mut exact = Vec::<f64>::new();
        let push_exact = |lo: f64, hi: f64, exact: &mut Vec<f64>| {
            let (lo, hi) = counts(lo, hi);
            let mut z = lo;
            while z <= hi {
                exact.push(z);
                z += 1.0;
            }
        };
        // Leaves come off the stack left to right.
        let mut i = 0;
        while i < leaves.len() {
            let kind = leaves[i].kind;
            let mut j = i;
            while j + 1 < leaves.len() && leaves[j + 1].kind == kind {
                j += 1;
            }
            let (first, last) = (&leaves[i], &leaves[j]);
            match kind {
                Kind::NonMember => {}
                Kind::Unknown => {
                    let (lo, hi) = counts(first.z_lo, last.z_hi);
                    ranges.push(ConformalInterval::closed(lo, hi));
                }
                Kind::Undecided => push_exact(first.z_lo, last.z_hi, &mut exact),
                Kind::Member => {
                    // Counts strictly above the low end's image and strictly
                    // below the high end's are members; the counts inside
                    // each image are decided alone. An end whose solve does
                    // not certify keeps its leaf's wider bound.
                    let lo = match image(first.s_lo, &first.warm) {
                        Some((z_min, z_max)) => {
                            push_exact(z_min, z_max, &mut exact);
                            z_max.floor() + 1.0
                        }
                        None => first.z_lo.max(0.0).ceil(),
                    };
                    let hi = match image(last.s_hi, &last.warm) {
                        Some((z_min, z_max)) => {
                            push_exact(z_min, z_max, &mut exact);
                            z_min.ceil() - 1.0
                        }
                        None => last.z_hi.floor(),
                    };
                    ranges.push(ConformalInterval::closed(lo.max(0.0), hi));
                }
            }
            i = j + 1;
        }
        for (i, &is_twin) in twin.iter().enumerate() {
            if is_twin {
                exact.push(self.y[i]);
            }
        }
        exact.sort_by(f64::total_cmp);
        exact.dedup();
        let (kept, dropped): (Vec<f64>, Vec<f64>) = exact
            .into_iter()
            .partition(|&z| self.count_member(row, z, u_tie, twin, tau));
        ranges.extend(kept.into_iter().map(|z| ConformalInterval::closed(z, z)));
        ranges.retain(|r| r.lo <= r.hi);
        ranges.sort_by(|p, q| p.lo.total_cmp(&q.lo));
        let mut merged = Vec::<ConformalInterval>::new();
        for piece in ranges {
            match merged.last_mut() {
                Some(last) if piece.lo <= last.hi + 1.0 => last.hi = last.hi.max(piece.hi),
                _ => merged.push(piece),
            }
        }
        // A count decided out by its own solve leaves every range it is in.
        for z in dropped {
            merged = merged
                .into_iter()
                .flat_map(|r| {
                    if r.lo <= z && z <= r.hi {
                        [
                            ConformalInterval::closed(r.lo, z - 1.0),
                            ConformalInterval::closed(z + 1.0, r.hi),
                        ]
                        .into_iter()
                        .filter(|p| p.lo <= p.hi)
                        .collect::<Vec<_>>()
                    } else {
                        vec![r]
                    }
                })
                .collect();
        }
        merged
    }

    /// Certified walk over the test-score coordinate for the continuous
    /// (Gamma) family.
    fn continuous_set(
        &self,
        row: &TestRow<'_>,
        tau: f64,
        tie_uniform: f64,
    ) -> Vec<ConformalInterval> {
        let whole = self.family.whole_support();
        if tau < tie_uniform {
            return whole;
        }
        // r: the fewest dominating training rows that keep a candidate in.
        let r = (tau - tie_uniform).floor() + 1.0;
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
            let included = match self.verdict(&node, e, t_lo, t_hi, tie_uniform, tau, None) {
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
                        piece: ConformalInterval::closed(leaf.z_lo, leaf.z_hi),
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
        for (b, &ib) in idx.iter().enumerate().take(a + 1) {
            let value = 0.5 * psd[[a, b]] + 0.5 * psd[[b, a]];
            s[[ia, ib]] = value;
            s[[ib, ia]] = value;
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};
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
            Some(0),
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
            let y = if family == ConformalGlmFamily::GammaLog {
                1.7
            } else {
                1.0
            };
            for eta in [-2.0, -0.3, 0.0, 0.8, 2.5] {
                let h = 1e-5;
                let d1 = (family.nll(eta + h, y) - family.nll(eta - h, y)) / (2.0 * h);
                let (u, w) = family.score_weight(eta, y);
                assert!((u + d1).abs() < 1e-6, "{family:?} score at {eta}");
                let (u_hi, _) = family.score_weight(eta + h, y);
                let (u_lo, _) = family.score_weight(eta - h, y);
                assert!(
                    (w + (u_hi - u_lo) / (2.0 * h)).abs() < 1e-6,
                    "{family:?} weight"
                );
                let (_, w_hi) = family.score_weight(eta + h, y);
                let (_, w_lo) = family.score_weight(eta - h, y);
                let dlogw = (w_hi.ln() - w_lo.ln()) / (2.0 * h);
                assert!(
                    dlogw.abs() <= 1.0 + 1e-6,
                    "{family:?} curvature log-derivative"
                );
                let z = family.response_of_score(eta, u);
                assert!(
                    (z - y).abs() < 1e-9,
                    "{family:?} response_of_score inverts the score"
                );
            }
        }
    }

    /// Exact membership of candidate `z` by a direct solve, independent of the
    /// set's pruning and certificates.
    fn brute_force_member(
        sub: &GlmFullConformalSubstrate,
        x_star: &Array1<f64>,
        o: f64,
        z: f64,
        alpha: f64,
    ) -> bool {
        let row = TestRow {
            x: x_star,
            offset: o,
        };
        let node = sub
            .solve(&row, Augmentation::Response(z), &sub.warm_start)
            .unwrap();
        assert!(node.certifies(node.error));
        let n = sub.n();
        let tau = conformal_rank_threshold(alpha, n + 1);
        let s_star = node.score_star.abs();
        let greater = (0..n).filter(|&i| node.score[i].abs() > s_star).count();
        let tied = (0..n).filter(|&i| node.score[i].abs() == s_star).count();
        greater as f64 + 0.5 * (1 + tied) as f64 > tau
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
                let set = sub
                    .prediction_set_with_uniform(&row(x), o, ALPHA, 0.5)
                    .unwrap();
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
                    assert!(
                        last.hi.is_finite(),
                        "{family:?}: the count tail must bound the set"
                    );
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
        let test = TestRow {
            x: &x_star,
            offset: 0.0,
        };
        let tau = conformal_rank_threshold(ALPHA, sub.n() + 1);
        let k_max = (tau - 0.5).floor() as usize;
        let tail = sub.count_score_tail(&test, k_max).unwrap();
        assert!(
            tail <= d.y.sum(),
            "tail score {tail} is not on the data scale"
        );
        let set = sub
            .prediction_set_with_uniform(&x_star, 0.0, ALPHA, 0.5)
            .unwrap();
        let hi = set.intervals.last().unwrap().hi;
        assert!(hi <= d.y.sum(), "set edge {hi} is not on the data scale");
        for z in 0..=(hi as usize + 5) {
            let z = z as f64;
            assert_eq!(
                contains(&set, z),
                brute_force_member(&sub, &x_star, 0.0, z, ALPHA),
                "z={z}"
            );
        }
    }

    #[test]
    fn large_mean_poisson_set_costs_its_boundaries_not_its_width() {
        // Means near e^11 put the set tens of thousands of counts wide, and the
        // tail thousands beyond it; one solve per count would take minutes.
        let mut rng = StdRng::seed_from_u64(4411);
        let family = ConformalGlmFamily::PoissonLog;
        let n = 60;
        let xs: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let y: Array1<f64> = xs
            .iter()
            .map(|&x| {
                PoissonDist::new((11.0 + 0.3 * x).exp())
                    .unwrap()
                    .sample(&mut rng)
            })
            .collect();
        let d = Data {
            x: design(&xs),
            y,
            offset: Array1::zeros(n),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let sub = substrate(family, &d);
            let set = sub
                .prediction_set_with_uniform(&row(0.2), 0.0, ALPHA, 0.5)
                .unwrap();
            tx.send((sub, set)).unwrap();
        });
        let (sub, set) = rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("the large-mean Poisson set must not enumerate its counts");
        let (lo, hi) = (set.intervals[0].lo, set.intervals.last().unwrap().hi);
        assert!(lo > 0.0 && hi.is_finite(), "set {:?}", set.intervals);
        // The edges are exact boundaries: brute force agrees on either side.
        for z in [
            lo - 3.0,
            lo - 1.0,
            lo,
            lo + 1.0,
            hi - 1.0,
            hi,
            hi + 1.0,
            hi + 3.0,
        ] {
            assert_eq!(
                contains(&set, z),
                brute_force_member(&sub, &row(0.2), 0.0, z, ALPHA),
                "z={z}"
            );
        }
    }

    #[test]
    fn certificate_distinguishes_conservative_frozen_from_refused() {
        let refused = ConformalCertificate::Refused(ConformalRefusal::GlmFrozenPenalty);
        for family in [
            ConformalGlmFamily::BernoulliLogit,
            ConformalGlmFamily::PoissonLog,
            ConformalGlmFamily::GammaLog,
        ] {
            assert_eq!(
                family.certificate(Some(0)),
                ConformalCertificate::ConservativeFrozen
            );
            // Bernoulli and Poisson carry no nuisance parameter beside the
            // strength, so one strength is re-selected and several are refused
            // by name; the Gamma dispersion is itself estimated, so its rows
            // keep the frozen-penalty refusal at every count.
            let honest = matches!(
                family,
                ConformalGlmFamily::BernoulliLogit | ConformalGlmFamily::PoissonLog
            );
            assert_eq!(
                family.certificate(Some(1)),
                if honest {
                    ConformalCertificate::HonestRefit
                } else {
                    refused
                }
            );
            assert_eq!(
                family.certificate(Some(3)),
                if honest {
                    ConformalCertificate::Refused(ConformalRefusal::MultiPenalty)
                } else {
                    refused
                }
            );
        }
        let nb = ConformalGlmFamily::NegativeBinomialLog { theta: 2.0 };
        assert_eq!(
            nb.certificate(Some(0)),
            refused,
            "θ is selected on the responses"
        );
        assert_eq!(
            ConformalGlmFamily::PoissonLog.certificate(None),
            ConformalCertificate::Refused(ConformalRefusal::UnknownPenaltyStructure)
        );
        assert_eq!(refused.code(), -7);
        assert_eq!(refused.label(), "refused:glm_frozen_penalty");
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
            let set = sub
                .prediction_set_with_uniform(&row(x), 0.0, alpha, 0.5)
                .unwrap();
            let hi = set.intervals.last().unwrap().hi;
            assert!(
                hi.is_finite(),
                "gamma set must be bounded: {:?}",
                set.intervals
            );
            for k in 1..400 {
                let z = hi * 1.5 * k as f64 / 400.0;
                let exact = brute_force_member(&sub, &row(x), 0.0, z, alpha);
                if exact {
                    assert!(
                        contains(&set, z),
                        "exact member {z} missing from {:?}",
                        set.intervals
                    );
                }
            }
            // Tightness: every endpoint is an exact boundary to solver accuracy.
            for iv in &set.intervals {
                for (edge, inside) in [(iv.lo, iv.lo * (1.0 + 1e-6)), (iv.hi, iv.hi * (1.0 - 1e-6))]
                {
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
            let st = sub.state(
                &beta,
                &TestRow {
                    x: &zero,
                    offset: 0.0,
                },
                Augmentation::Tilt(0.0),
            );
            assert!(vec_norm(&st.grad) < 1e-8 * st.natural_scale, "{family:?}");
        }
    }

    #[test]
    fn penalty_recovery_zeros_unpenalized_columns_and_is_psd() {
        let gram =
            Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0])
                .unwrap();
        let mut normal = gram.clone();
        normal[[1, 1]] += 2.0;
        normal[[2, 2]] += 1.0;
        normal[[1, 2]] += 0.5;
        normal[[2, 1]] += 0.5;
        normal[[0, 0]] += 1e-15;
        let s = penalty_from_normal_and_gram(&normal, &gram, &[1..3], 2.0).unwrap();
        assert_eq!(s, s.t(), "the recovered shape must be symmetric in storage");
        assert!(s.row(0).iter().all(|&v| v == 0.0) && s.column(0).iter().all(|&v| v == 0.0));
        assert!((s[[1, 1]] - 4.0).abs() < 1e-12 && (s[[1, 2]] - 1.0).abs() < 1e-12);
        let mut indefinite = gram.clone();
        indefinite[[1, 1]] -= 1e-13;
        let s = penalty_from_normal_and_gram(&indefinite, &gram, &[1..3], 1.0).unwrap();
        let (evals, _) = s.eigh(Side::Lower).unwrap();
        assert!(evals.iter().all(|&v| v >= -1e-15));
    }

    /// The count `0` sits exactly on the low end of the score walk, where
    /// rounding in `z(s)` once pushed its image just above `0` and dropped it
    /// from a member run. These replayed draws had `0` in by its own solve.
    #[test]
    fn count_zero_on_the_walk_edge_is_not_rounded_out() {
        let reps = [169usize, 249, 630, 637, 645, 717, 722, 802];
        let n = 99;
        let family = FAMILIES[2];
        let mut rng = StdRng::seed_from_u64(4242 + 2);
        for rep in 0..=802 {
            let d = data(family, n, &mut rng);
            let x = rng.random::<f64>() * 2.0 - 1.0;
            let o = rng.random::<f64>() * 0.4 - 0.2;
            let y_star = draw(family, eta_true(x) + o, &mut rng);
            if !reps.contains(&rep) {
                continue;
            }
            let sub = substrate(family, &d);
            let xs = row(x);
            let test = TestRow { x: &xs, offset: o };
            let tau = conformal_rank_threshold(ALPHA, n + 1);
            let u = 0.5;
            let twin = vec![false; n];
            assert!(sub.count_member(&test, 0.0, u, &twin, tau), "rep {rep}");
            let set = sub.prediction_set_with_uniform(&xs, o, ALPHA, 0.5).unwrap();
            assert_eq!(
                set.intervals.first().map(|r| r.lo),
                Some(0.0),
                "rep {rep}: {:?}",
                set.intervals
            );
            let covered = set
                .intervals
                .iter()
                .any(|r| r.lo <= y_star && y_star <= r.hi);
            assert_eq!(
                covered,
                sub.count_member(&test, y_star, u, &twin, tau),
                "rep {rep}: y* {y_star}"
            );
        }
    }
}

#[cfg(test)]
mod consolidation_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn uncancelled_poisson_stationarity_preserves_3451() {
        let n = 16;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (std::f64::consts::PI * i as f64 / 15.0).cos()
            }
        });
        let y = Array1::from_shape_fn(n, |i| {
            (1.0 + (2.0 * std::f64::consts::PI * i as f64 / 15.0).sin())
                .exp()
                .round()
        });
        let total = y.sum();
        let sub = GlmFullConformalSubstrate::new(
            ConformalGlmFamily::PoissonLog,
            x,
            y,
            Array1::zeros(n),
            Array2::zeros((2, 2)),
            Some(0),
            Array1::zeros(2),
        )
        .unwrap();
        let beta = sub.refit_labeled_rows().unwrap();
        let x_star = array![1.0, (std::f64::consts::PI * 0.37).cos()];
        let row = TestRow {
            x: &x_star,
            offset: 0.0,
        };
        let z = x_star.dot(&beta).exp();
        let state = sub.state(&beta, &row, Augmentation::Response(z));
        assert!(state.natural_scale >= total);
        assert!(sub.kkt_converged(&state));
        let cold = sub
            .solve(&row, Augmentation::Response(z), &Array1::zeros(2))
            .unwrap();
        assert!(cold.certifies(cold.error));
        assert!(vec_norm(&(&cold.beta - &beta)) <= 1e-9 * (1.0 + vec_norm(&beta)));
        let nonstationary = State {
            natural_scale: f64::INFINITY,
            grad: array![1.0, 1.0],
            ..state
        };
        assert!(!sub.kkt_converged(&nonstationary));
    }

    #[test]
    fn fixed_uniform_validates_endpoints_and_is_shared_by_labels() {
        // Identical zero rows produce exactly tied scores for each Bernoulli
        // candidate. For candidate 0 every score is 1/2, hence its p-value is U.
        let sub = GlmFullConformalSubstrate::new(
            ConformalGlmFamily::BernoulliLogit,
            Array2::zeros((4, 1)),
            Array1::zeros(4),
            Array1::zeros(4),
            array![[1.0]],
            Some(0),
            array![0.0],
        )
        .unwrap();
        let row = array![0.0];
        for u in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.0] {
            assert!(sub.prediction_set_with_uniform(&row, 0.0, 0.2, u).is_err());
        }
        for u in [0.0, 0.1, 0.2, 0.3, 1.0 - f64::EPSILON] {
            let set = sub.prediction_set_with_uniform(&row, 0.0, 0.2, u).unwrap();
            let member = set
                .intervals
                .iter()
                .any(|piece| piece.lo <= 0.0 && piece.hi >= 0.0);
            assert_eq!(member, u > 0.2, "tied candidate, U={u}");
        }
    }

    #[test]
    fn supplied_uniform_preserves_row_permutation() {
        let x = array![
            [1.0, -1.0],
            [1.0, -0.4],
            [1.0, 0.2],
            [1.0, 0.8],
            [1.0, 1.2],
            [1.0, -0.8]
        ];
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let offset = array![0.1, -0.2, 0.3, 0.0, -0.1, 0.2];
        let order = [3, 0, 5, 1, 4, 2];
        let make = |x, y, offset| {
            GlmFullConformalSubstrate::new(
                ConformalGlmFamily::BernoulliLogit,
                x,
                y,
                offset,
                array![[0.5, 0.0], [0.0, 0.7]],
                Some(0),
                Array1::zeros(2),
            )
            .unwrap()
        };
        let ordinary = make(x.clone(), y.clone(), offset.clone());
        let permuted = make(
            x.select(Axis(0), &order),
            y.select(Axis(0), &order),
            offset.select(Axis(0), &order),
        );
        let row = array![1.0, 0.35];
        for u in [0.0, 0.17, 0.5, 0.93] {
            for alpha in [0.1, 0.3, 0.7] {
                let first = ordinary
                    .prediction_set_with_uniform(&row, 0.15, alpha, u)
                    .unwrap();
                let second = permuted
                    .prediction_set_with_uniform(&row, 0.15, alpha, u)
                    .unwrap();
                assert_eq!(first.intervals, second.intervals, "U={u}, alpha={alpha}");
            }
        }
    }
    #[test]
    fn finite_exchangeability_and_uniform_grid_give_nominal_coverage() {
        // Uniformly hold out one of five exchangeable rows, independently of
        // U. The augmented fit is always the same, and its fitted p exceeds
        // 1/2: two zero-label scores dominate the three one-label scores.
        // The exact ranks are 2U/5 for zeros, (2+3U)/5 for ones.
        let labels = array![0.0, 0.0, 1.0, 1.0, 1.0];
        let mut covered = 0;
        for held_out in 0..5 {
            let train = Array1::from_iter((0..5).filter(|&i| i != held_out).map(|i| labels[i]));
            let sub = GlmFullConformalSubstrate::new(
                ConformalGlmFamily::BernoulliLogit,
                Array2::ones((4, 1)),
                train,
                Array1::zeros(4),
                array![[0.5]],
                Some(0),
                array![0.0],
            )
            .unwrap();
            for k in 0..20 {
                let u = (k as f64 + 0.5) / 20.0;
                let z = labels[held_out];
                let expected = if z == 0.0 {
                    2.0 * u > 1.5
                } else {
                    2.0 + 3.0 * u > 1.5
                };
                let set = sub
                    .prediction_set_with_uniform(&array![1.0], 0.0, 0.3, u)
                    .unwrap();
                let member = set
                    .intervals
                    .iter()
                    .any(|piece| piece.lo <= z && z <= piece.hi);
                assert_eq!(member, expected, "held out {held_out}, U={u}");
                covered += usize::from(member);
            }
        }
        assert_eq!(covered, 70);
        let certificate = ConformalGlmFamily::BernoulliLogit.certificate(Some(0));
        assert_eq!(certificate.label(), "conservative_frozen");
        assert_eq!(certificate.code(), 2);
    }
}

#[cfg(test)]
mod reselection_tests {
    use super::*;
    use ndarray::{Axis, array};

    fn fixture(scale: f64, warm: Array1<f64>) -> GlmFullConformalSubstrate {
        let x = array![
            [1., -1.4],
            [1., -1.1],
            [1., -0.8],
            [1., -0.5],
            [1., -0.2],
            [1., 0.1],
            [1., 0.4],
            [1., 0.7],
            [1., 1.0],
            [1., 1.3],
            [1., 1.6],
            [1., 1.9]
        ];
        let y = array![0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1.];
        GlmFullConformalSubstrate::new(
            ConformalGlmFamily::BernoulliLogit,
            x,
            y,
            Array1::zeros(12),
            array![[0., 0.], [0., scale]],
            Some(1),
            warm,
        )
        .unwrap()
    }

    #[test]
    /// The Poisson analogue of [`fixture`]: the same design and one selected
    /// strength on the non-intercept column, with counts on a rising trend.
    fn poisson_fixture(scale: f64, warm: Array1<f64>) -> GlmFullConformalSubstrate {
        let x = array![
            [1., -1.4],
            [1., -1.1],
            [1., -0.8],
            [1., -0.5],
            [1., -0.2],
            [1., 0.1],
            [1., 0.4],
            [1., 0.7],
            [1., 1.0],
            [1., 1.3],
            [1., 1.6],
            [1., 1.9]
        ];
        let y = array![1., 0., 2., 1., 1., 2., 3., 2., 4., 3., 5., 6.];
        GlmFullConformalSubstrate::new(
            ConformalGlmFamily::PoissonLog,
            x,
            y,
            Array1::zeros(12),
            array![[0., 0.], [0., scale]],
            Some(1),
            warm,
        )
        .unwrap()
    }

    /// `weight_jet` is the only family-specific content of the honest
    /// criterion, checked against the negative log-likelihood it claims to
    /// differentiate rather than against the criterion that consumes it: `w` is
    /// the second η-difference of `ℓ`, and `w′`, `w″` are differences of `w`.
    /// The curvature of both links is free of `y`, so one response witnesses it.
    #[test]
    fn weight_jet_is_the_curvature_of_the_nll_and_its_derivatives() {
        for family in [
            ConformalGlmFamily::BernoulliLogit,
            ConformalGlmFamily::PoissonLog,
        ] {
            let y = 1.0;
            for &eta in &[-1.7, -0.6, 0.0, 0.4, 1.3] {
                // Central differences at step `h` carry truncation plus
                // cancellation: `h²·|f‴|/6 + ε·|f|/h` for the first difference,
                // `h²·|f⁗|/12 + 4ε·|f|/h²` for the second. Every η-derivative of
                // the Poisson curvature IS `μ`, and the logistic's are bounded by
                // its own curvature, so `scale` bounds each `|f|` here. The factor
                // two is the next term of each series, which has the same shape one
                // order down in `h`.
                let h = 1e-3;
                let (w, w1, w2) = family.weight_jet(eta).unwrap();
                let scale = w.abs().max(1.0);
                let first_band = 2.0 * scale * (h * h / 6.0 + f64::EPSILON / h);
                let second_band = 2.0 * scale * (h * h / 12.0 + 4.0 * f64::EPSILON / (h * h));
                let curvature = |e: f64| family.weight_jet(e).unwrap().0;
                let slope = |e: f64| family.weight_jet(e).unwrap().1;
                let fd_w = (family.nll(eta + h, y) - 2.0 * family.nll(eta, y)
                    + family.nll(eta - h, y))
                    / (h * h);
                let fd_w1 = (curvature(eta + h) - curvature(eta - h)) / (2.0 * h);
                let fd_w2 = (slope(eta + h) - slope(eta - h)) / (2.0 * h);
                assert!(
                    (w - fd_w).abs() < second_band,
                    "{family:?} eta={eta}: w={w} second difference={fd_w} band={second_band:e}"
                );
                assert!(
                    (w1 - fd_w1).abs() < first_band,
                    "{family:?} eta={eta}: w1={w1} difference={fd_w1} band={first_band:e}"
                );
                assert!(
                    (w2 - fd_w2).abs() < first_band,
                    "{family:?} eta={eta}: w2={w2} difference={fd_w2} band={first_band:e}"
                );
            }
        }
        for family in [
            ConformalGlmFamily::NegativeBinomialLog { theta: 2.0 },
            ConformalGlmFamily::GammaLog,
        ] {
            assert!(
                family.weight_jet(0.3).is_none(),
                "{family:?} has no re-selecting map and must not offer a curvature"
            );
        }
    }

    #[test]
    fn criterion_gradient_and_hessian_match_independent_differences() {
        let sub = fixture(1.0, Array1::zeros(2));
        let selected = sub.reselection.as_ref().unwrap();
        let star = array![1., 0.35];
        let row = TestRow {
            x: &star,
            offset: 0.1,
        };
        for z in [0., 1.] {
            for rho in [-3., -0.5, 1., 3.] {
                let h = 1e-4;
                let evaluate = |r| {
                    sub.laml_jet(selected, &row, z, r, &Array1::zeros(2))
                        .unwrap()
                };
                let left = evaluate(rho - h);
                let mid = evaluate(rho);
                let right = evaluate(rho + h);
                let fd_gradient = (right.value - left.value) / (2. * h);
                let fd_hessian = (right.gradient - left.gradient) / (2. * h);
                assert!(
                    (mid.gradient - fd_gradient).abs() < 2e-6 * (1. + fd_gradient.abs()),
                    "rho={rho} z={z}: gradient={} fd={fd_gradient}",
                    mid.gradient
                );
                assert!(
                    (mid.hessian - fd_hessian).abs() < 2e-6 * (1. + fd_hessian.abs()),
                    "rho={rho} z={z}: hessian={} fd={fd_hessian}",
                    mid.hessian
                );
            }
        }
    }

    #[test]
    fn penalty_units_and_training_warm_start_do_not_change_selection() {
        let baseline = fixture(1., Array1::zeros(2));
        let star = array![1., 0.35];
        let row = TestRow {
            x: &star,
            offset: 0.1,
        };
        for scale in [1e-200, 1., 1e200] {
            let sub = fixture(scale, array![30., -25.]);
            assert_eq!(sub.certificate, ConformalCertificate::HonestRefit);
            assert_eq!(
                sub.reselection.as_ref().unwrap().unit,
                baseline.reselection.as_ref().unwrap().unit
            );
            for z in [0., 1.] {
                let (rho, beta) = sub
                    .select_strength(sub.reselection.as_ref().unwrap(), &row, z)
                    .unwrap();
                let (expected, expected_beta) = baseline
                    .select_strength(baseline.reselection.as_ref().unwrap(), &row, z)
                    .unwrap();
                assert_eq!(rho, expected);
                assert_eq!(beta, expected_beta);
            }
            let set = sub
                .prediction_set_with_uniform(&star, 0.1, 0.3, 0.6)
                .unwrap();
            let expected = baseline
                .prediction_set_with_uniform(&star, 0.1, 0.3, 0.6)
                .unwrap();
            assert_eq!(set.intervals, expected.intervals);
        }
    }

    #[test]
    fn selected_membership_matches_direct_refit_and_row_permutations() {
        let sub = fixture(1., Array1::zeros(2));
        let star = array![1., 0.35];
        let row = TestRow {
            x: &star,
            offset: 0.1,
        };
        let order = [7, 2, 11, 0, 9, 1, 10, 3, 8, 4, 6, 5];
        let reordered = GlmFullConformalSubstrate::new(
            sub.family,
            sub.x.select(Axis(0), &order),
            sub.y.select(Axis(0), &order),
            sub.offset.select(Axis(0), &order),
            sub.s_lambda.clone(),
            Some(1),
            array![20., -30.],
        )
        .unwrap();
        for uniform in [0.1, 0.6, 0.9] {
            let set = sub
                .prediction_set_with_uniform(&star, 0.1, 0.3, uniform)
                .unwrap();
            let permuted = reordered
                .prediction_set_with_uniform(&star, 0.1, 0.3, uniform)
                .unwrap();
            assert_eq!(set.intervals, permuted.intervals);
            for z in [0., 1.] {
                let selected = sub.reselection.as_ref().unwrap();
                let (rho, beta) = sub.select_strength(selected, &row, z).unwrap();
                let refit = sub.at_strength(selected, rho, beta.clone()).unwrap();
                let node = refit
                    .solve(&row, Augmentation::Response(z), &Array1::zeros(2))
                    .unwrap();
                assert!(node.certifies(node.error));
                let test_score = sub
                    .family
                    .score_weight(star.dot(&node.beta) + 0.1, z)
                    .0
                    .abs();
                let scores = fast_av(&sub.x, &node.beta) + &sub.offset;
                let greater = scores
                    .iter()
                    .zip(sub.y.iter())
                    .filter(|(eta, y)| sub.family.score_weight(**eta, **y).0.abs() > test_score)
                    .count();
                let tied = scores
                    .iter()
                    .zip(sub.y.iter())
                    .filter(|(eta, y)| sub.family.score_weight(**eta, **y).0.abs() == test_score)
                    .count();
                let expected = greater as f64 + uniform * (1 + tied) as f64 > 0.3 * 13.;
                assert_eq!(
                    set.intervals.iter().any(|piece| piece.contains(z)),
                    expected
                );
            }
        }
    }

    #[test]
    fn selection_failures_and_invalid_strength_are_not_frozen_answers() {
        let sub = fixture(1., Array1::zeros(2));
        let selected = sub.reselection.as_ref().unwrap();
        for rho in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e6] {
            assert!(sub.at_strength(selected, rho, Array1::zeros(2)).is_err());
        }
        let mut invalid = sub.clone();
        invalid.reselection.as_mut().unwrap().gram.fill(f64::NAN);
        let error = invalid
            .prediction_set_with_uniform(&array![1., 0.35], 0.1, 0.3, 0.1)
            .err()
            .expect("invalid domain must propagate");
        assert!(
            error.contains("spectrum") || error.contains("domain"),
            "{error}"
        );
        let zero = fixture(0., Array1::zeros(2));
        assert_eq!(zero.certificate, ConformalCertificate::ConservativeFrozen);
        assert!(zero.reselection.is_none());
        assert!(
            GlmFullConformalSubstrate::new(
                sub.family,
                sub.x.clone(),
                sub.y.clone(),
                sub.offset.clone(),
                sub.s_lambda.clone(),
                Some(1),
                array![f64::NAN, 0.]
            )
            .is_err()
        );
    }
    #[test]
    fn augmented_row_roles_preserve_the_selected_fitting_map() {
        let x = array![
            [1., -1.2],
            [1., -0.83],
            [1., -0.19],
            [1., 0.36],
            [1., 0.77],
            [1., 1.41]
        ];
        let y = array![0., 0., 1., 0., 1., 1.];
        let mut reference: Option<(f64, Array1<f64>)> = None;
        let mut accepted = 0usize;
        for held in 0..6 {
            let retained: Vec<usize> = (0..6).filter(|&i| i != held).collect();
            let sub = GlmFullConformalSubstrate::new(
                ConformalGlmFamily::BernoulliLogit,
                x.select(Axis(0), &retained),
                y.select(Axis(0), &retained),
                Array1::zeros(5),
                array![[0., 0.], [0., 1.]],
                Some(1),
                array![held as f64, -10.],
            )
            .unwrap();
            let star = x.row(held).to_owned();
            let row = TestRow {
                x: &star,
                offset: 0.,
            };
            let (rho, beta) = sub
                .select_strength(sub.reselection.as_ref().unwrap(), &row, y[held])
                .unwrap();
            if let Some((expected, coefficients)) = &reference {
                assert!(
                    (rho - expected).abs() < 1e-8,
                    "held={held}, rho={rho}, reference={expected}"
                );
                assert!(
                    beta.iter()
                        .zip(coefficients.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-8)
                );
            } else {
                reference = Some((rho, beta));
            }
            for uniform in [0.125, 0.375, 0.625, 0.875] {
                let set = sub
                    .prediction_set_with_uniform(&star, 0., 0.5, uniform)
                    .unwrap();
                accepted += usize::from(set.intervals.iter().any(|piece| piece.contains(y[held])));
            }
        }
        // A finite rank/permutation regression, not a general coverage proof.
        assert_eq!(accepted, 12);
    }
    #[test]
    fn unit_normalization_refuses_subnormal_rank_loss() {
        let tiny = f64::from_bits(1);
        let mut penalty = Array2::<f64>::eye(5);
        penalty[[4, 4]] = tiny;
        // Max scaling leaves tiny intact; the following norm=2 division
        // rounds tiny/2 to zero and would erase a real penalized direction.
        assert_eq!(tiny / 1.0, tiny);
        assert_eq!(tiny / 2.0, 0.0);
        let error = GlmFullConformalSubstrate::new(
            ConformalGlmFamily::BernoulliLogit,
            Array2::eye(5),
            array![0., 1., 0., 1., 0.],
            Array1::zeros(5),
            penalty,
            Some(1),
            Array1::zeros(5),
        )
        .err()
        .expect("unit normalization must preserve each nonzero entry");
        assert!(error.contains("unit penalty normalization"), "{error}");
    }

    /// The generalized criterion at the Poisson weights: the same independent
    /// central differences the Bernoulli arm is held to. A wrong `w′` or `w″`
    /// in [`ConformalGlmFamily::weight_jet`] survives the value but not these.
    #[test]
    fn poisson_criterion_gradient_and_hessian_match_independent_differences() {
        let sub = poisson_fixture(1.0, Array1::zeros(2));
        let selected = sub.reselection.as_ref().unwrap();
        let star = array![1., 0.35];
        let row = TestRow {
            x: &star,
            offset: 0.1,
        };
        for z in [0., 2., 5.] {
            for rho in [-3., -0.5, 1., 3.] {
                let h = 1e-4;
                let evaluate = |r| {
                    sub.laml_jet(selected, &row, z, r, &Array1::zeros(2))
                        .unwrap()
                };
                let left = evaluate(rho - h);
                let mid = evaluate(rho);
                let right = evaluate(rho + h);
                let fd_gradient = (right.value - left.value) / (2. * h);
                let fd_hessian = (right.gradient - left.gradient) / (2. * h);
                assert!(
                    (mid.gradient - fd_gradient).abs() < 2e-6 * (1. + fd_gradient.abs()),
                    "rho={rho} z={z}: gradient={} fd={fd_gradient}",
                    mid.gradient
                );
                assert!(
                    (mid.hessian - fd_hessian).abs() < 2e-6 * (1. + fd_hessian.abs()),
                    "rho={rho} z={z}: hessian={} fd={fd_hessian}",
                    mid.hessian
                );
            }
        }
    }

    /// The walk is one substrate per level, not one substrate: the levels of a
    /// single test row select strengths that are not all equal. Without this the
    /// level walk would be an expensive way to reproduce the frozen arm.
    #[test]
    fn poisson_levels_select_their_own_strengths() {
        let sub = poisson_fixture(1., Array1::zeros(2));
        let selected = sub.reselection.as_ref().unwrap();
        let star = array![1., 0.35];
        let row = TestRow {
            x: &star,
            offset: 0.1,
        };
        let rhos: Vec<f64> = [0., 3., 8.]
            .into_iter()
            .map(|z| sub.select_strength(selected, &row, z).unwrap().0)
            .collect();
        assert!(
            rhos.windows(2).any(|pair| pair[0] != pair[1]),
            "every level selected the same strength, so this is one substrate: {rhos:?}"
        );
    }

    /// The honest count set closes, and every level it reports agrees with that
    /// level's own selected refit — the definition of the map, evaluated here
    /// without the walk's bracket or stopping rule.
    #[test]
    fn poisson_level_walk_closes_and_matches_each_levels_own_refit() {
        let sub = poisson_fixture(1., Array1::zeros(2));
        let selected = sub.reselection.as_ref().unwrap();
        let star = array![1., 0.35];
        let offset = 0.1;
        let alpha = 0.3;
        let u_tie = 0.6;
        let row = TestRow { x: &star, offset };
        let set = sub
            .prediction_set_with_uniform(&star, offset, alpha, u_tie)
            .unwrap();
        assert_eq!(set.certificate, ConformalCertificate::HonestRefit);
        let top = set
            .intervals
            .iter()
            .fold(0.0_f64, |acc, piece| acc.max(piece.hi));
        assert!(
            top.is_finite(),
            "the walk did not close: {:?}",
            set.intervals
        );
        let tau = conformal_rank_threshold(alpha, sub.n() + 1);
        let twin = sub.twin_rows(&row);
        let mut z = 0.0_f64;
        while z <= top + 2.0 {
            let direct = sub
                .honest_level(selected, &row, z, tau, u_tie, &twin)
                .unwrap()
                .0;
            let walked = set.intervals.iter().any(|piece| piece.contains(z));
            assert_eq!(
                walked, direct,
                "level {z}: walk={walked} own refit={direct}"
            );
            z += 1.0;
        }
    }
}
