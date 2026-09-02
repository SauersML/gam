//! Exact O(n) state-space polynomial smoothing spline ("the scan").
//!
//! The order-`m` intrinsic Gaussian prior whose penalized posterior mean is the
//! degree-`(2m−1)` smoothing spline (penalty `λ∫(f^{(m)})²`) is a Markov process
//! in the state `α(x) = (f, f′, …, f^{(m−1)})`: an `m`-fold integrated Wiener
//! process. The Kalman filter + RTS smoother over the x-sorted observations
//! therefore computes the EXACT smoothing-spline posterior — mean, derivatives,
//! pointwise variance — and the diffuse innovations decomposition computes the
//! EXACT restricted (REML) likelihood, all in O(n) work per smoothing-parameter
//! trial instead of the dense O(n·k²) design/Gram + O(k³) solve per trial
//! (Wahba 1978; Kohn & Ansley 1987; Durbin & Koopman exact diffuse init).
//!
//! Supported orders are `m ∈ {1, 2, 3}` (`MAX_ORDER`): `m = 1` is the
//! random-walk / linear smoother (penalty `λ∫f′²`), `m = 2` the cubic smoother
//! (`λ∫f″²`), `m = 3` the quintic smoother (`λ∫(f‴)²`, natural spline degree
//! `2m−1 = 5`). The diffuse prior carries `m` improper dimensions consumed by
//! the first `m` distinct abscissae, leaving `m − 1` *partially-diffuse leading
//! nodes* whose smoothed moments the ordinary RTS recursion cannot reach (its
//! predicted covariance is rank-deficient there). For `m = 2` that is the
//! single node 0; for `m = 3` the pair {0, 1}. These are recovered exactly by a
//! joint Gaussian conditioning of the whole leading block on the first proper
//! smoothed node (see the smoother pass) — the exact diffuse analog of RTS, and
//! the multi-node generalization of the `m = 2` reverse-Markov closure.
//!
//! Model, after sorting and pooling tied abscissae (precision-weighted):
//!   α_{t+1} = F_t α_t + η_t,   η_t ~ N(0, q·Q(δ_t)),   q = σ_w²/σ² = 1/λ,
//!   y_t     = H α_t + ε_t,     ε_t ~ N(0, σ²/w_t),     H = [1 0 … 0],
//!   F(δ) = exp(δA) (nilpotent shift A),   Q(δ) the m-fold IWP noise,
//! with a diffuse (improper, flat) prior on the first `m` states carrying the
//! unpenalized degree-`<m` polynomial null space the spline leaves unshrunk.
//! (`m = 2`: `F = [[1,δ],[0,1]]`, `Q = [[δ³/3,δ²/2],[δ²/2,δ]]`.)
//!
//! Exactness boundaries, by construction:
//! - the diffuse dimension is `m` and is consumed by the first `m` distinct
//!   abscissae, after which the filter is an ordinary proper Kalman filter;
//! - the `m − 1` partially-diffuse leading nodes are recovered by exact Markov
//!   conditioning of the whole leading block on the first proper smoothed node,
//!   `p(α_{0..m−2} | y) = ∫ p(α_{0..m−2} | α_{m−1}, y_{0..m−2}) p(α_{m−1} | y)`
//!   — an affine `((m−1)m)×m` Bayes update built from the flat leading prior,
//!   the Markov increments, and the leading observations; it reduces to the
//!   single-node reverse-Markov closure at `m = 2` and needs no diffuse RTS
//!   recursion;
//! - off-knot prediction is the Gaussian bridge conditional on the two
//!   flanking smoothed states (using the exact lag-one smoothed
//!   cross-covariance `G_t · P^s_{t+1}`), or boundary extrapolation from the
//!   end states, which reproduces the spline's polynomial extrapolation with
//!   growing variance — bridge-don't-sag is a theorem here.
//!
//! The smoothing parameter is selected by isolating every stationary interval
//! of the concentrated diffuse restricted log-likelihood over log λ. Exact
//! analytic score sensitivities are propagated through the filter, and global
//! curvature bounds drive certified adaptive subdivision; both finite-domain
//! boundaries compete exactly. σ² is profiled in closed form from the proper
//! innovations plus the within-tie residual sum.

use std::cell::RefCell;
use std::collections::HashMap;

use gam_math::score_opt::{
    ClosedInterval, DerivativeEnclosure, ScoreJet, ScoreOptimumLocation, ScoreSample,
    ScoreSearchResult, ScoreValueEnclosure, maximize_score_1d,
};

/// One pooled (distinct-abscissa) observation node.
#[derive(Clone, Copy, Debug)]
struct PooledNode {
    x: f64,
    /// Precision-weighted mean of the tied responses.
    y: f64,
    /// Total weight of the pooled ties (observation variance is `σ²/w`).
    w: f64,
}

/// Search interval for log λ (natural log), generous on both sides.
const LOG_LAMBDA_LO: f64 = -18.0;
const LOG_LAMBDA_HI: f64 = 18.0;
/// Maximum supported smoothing-spline order handled by the fixed-capacity
/// small-matrix layer. Order `m` penalizes `∫(f^{(m)})²`; the state dimension
/// is `m`. The exact diffuse leading-block smoother (see the smoother pass)
/// recovers the `m − 1` partially-diffuse leading nodes for any `m`: `m = 1`
/// has none, `m = 2` has node 0, `m = 3` has {0, 1}. Order 3 (the quintic
/// smoothing spline, #1044) is the current cap; bumping it further only needs a
/// wider `mat_inv` branch and the (already order-general) leading-block solve.
const MAX_ORDER: usize = 3;

/// Row-major `m × m` matrix stored in a fixed `MAX_ORDER`-capacity buffer; only
/// the top-left `m × m` block is meaningful. Generalizing the order-2 cubic
/// scan to order `m ∈ {1, 2, 3}` (#1034 item 2, #1044) keeps the
/// allocation-free fixed storage of the hot filter loop while letting `m` vary
/// at runtime.
type Mat2 = [[f64; MAX_ORDER]; MAX_ORDER];
type Vec2 = [f64; MAX_ORDER];

/// A nearest-rounded representative carried beside an outward interval for the
/// exact-real result of the same arithmetic expression.
///
/// Every elementary operation rounds both interval endpoints away from the
/// result. `exp` and `ln` use `gam-math`'s range-reduced, directed Taylor
/// enclosures; all other operations are IEEE basic operations. No platform
/// libm accuracy contract is assumed. This is interval arithmetic, not a
/// tolerance: widening is entirely source-derived from the operations the
/// filter actually performs.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ball {
    value: f64,
    lo: f64,
    hi: f64,
}

impl Ball {
    const ZERO: Self = Self {
        value: 0.0,
        lo: 0.0,
        hi: 0.0,
    };
    const ONE: Self = Self {
        value: 1.0,
        lo: 1.0,
        hi: 1.0,
    };

    #[inline]
    fn exact(value: f64) -> Self {
        Self {
            value,
            lo: value,
            hi: value,
        }
    }

    /// Attach an independently certified exact-real enclosure to a rounded
    /// representative.
    #[inline]
    fn certified(value: f64, enclosure: ClosedInterval) -> Self {
        Self {
            value,
            lo: enclosure.lo,
            hi: enclosure.hi,
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        let enclosure = self.interval().add(other.interval());
        Self {
            value: self.value + other.value,
            lo: enclosure.lo,
            hi: enclosure.hi,
        }
    }

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        let enclosure = self.interval().mul(other.interval());
        Self {
            value: self.value * other.value,
            lo: enclosure.lo,
            hi: enclosure.hi,
        }
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        self.mul(Self::exact(factor))
    }

    /// Division after the caller has proved the denominator interval positive.
    #[inline]
    fn div_positive(self, denominator: Self) -> Self {
        assert!(
            denominator.is_finite() && denominator.lo > 0.0,
            "Ball::div_positive requires a finite, strictly positive denominator interval, got \
             value={} lo={} hi={}",
            denominator.value,
            denominator.lo,
            denominator.hi
        );
        let reciprocal = Self {
            value: 1.0 / denominator.value,
            lo: if denominator.hi == 1.0 {
                1.0
            } else {
                next_down_ball(1.0 / denominator.hi)
            },
            hi: if denominator.lo == 1.0 {
                1.0
            } else {
                next_up_ball(1.0 / denominator.lo)
            },
        };
        self.mul(reciprocal)
    }

    #[inline]
    fn ln_positive(self) -> Self {
        assert!(
            self.is_finite() && self.lo > 0.0,
            "Ball::ln_positive requires a finite, strictly positive interval, got value={} lo={} \
             hi={}",
            self.value,
            self.lo,
            self.hi
        );
        let lo = gam_math::score_opt::certified_ln_positive(self.lo)
            .expect("positive finite interval lower endpoint");
        let hi = gam_math::score_opt::certified_ln_positive(self.hi)
            .expect("positive finite interval upper endpoint");
        Self {
            value: self.value.ln(),
            lo: lo.lo,
            hi: hi.hi,
        }
    }

    #[inline]
    fn square(self) -> Self {
        let hi_abs = self.lo.abs().max(self.hi.abs());
        let lo_abs = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            self.lo.abs().min(self.hi.abs())
        };
        Self {
            value: self.value * self.value,
            lo: if lo_abs == 0.0 {
                0.0
            } else if lo_abs == 1.0 {
                1.0
            } else {
                next_down_ball(lo_abs * lo_abs)
            },
            hi: if hi_abs == 0.0 || hi_abs == 1.0 {
                hi_abs
            } else {
                next_up_ball(hi_abs * hi_abs)
            },
        }
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.value.is_finite() && self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }

    #[inline]
    fn interval(self) -> ClosedInterval {
        ClosedInterval::new(self.lo, self.hi)
    }

    #[inline]
    fn forward_error(self) -> f64 {
        // One outward successor covers the subtraction roundoff even when the
        // exact distance is subnormal and the rounded subtraction is zero.
        // Every Ball primitive already preserves exact structural zero or
        // moves each inexact endpoint by one representable value.
        next_up_ball(
            (self.value - self.lo)
                .abs()
                .max((self.hi - self.value).abs()),
        )
    }
}

type BallMat = [[Ball; MAX_ORDER]; MAX_ORDER];
type BallVec = [Ball; MAX_ORDER];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplineInnovationKind {
    Diffuse,
    Proper,
}

/// Failure to construct a numerical proof for one spline-score evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum SplineScoreProofError {
    /// Directed roundoff made an innovation interval include zero, so division
    /// by that innovation cannot be certified.
    InnovationContainsZero {
        node: usize,
        kind: SplineInnovationKind,
        enclosure: ClosedInterval,
    },
    /// An innovation interval was entirely nonpositive. This indicates a
    /// violated covariance invariant rather than loss of numerical resolution.
    NonPositiveInnovation {
        node: usize,
        kind: SplineInnovationKind,
        enclosure: ClosedInterval,
    },
    NonPositiveProfileResidual {
        enclosure: ClosedInterval,
    },
    InvalidArithmetic {
        context: &'static str,
    },
    /// A certified filter accumulator left the finite range, reported with the
    /// node it happened at and the enclosure that did it.
    ///
    /// The `InvalidArithmetic{"diffuse filter accumulator"}` refusal above named
    /// a PHASE and nothing else: not which of the eight accumulators diverged,
    /// not at which of the ~180 nodes, and not how wide it was when it went.
    /// Diagnosing #2614 from it required adding a print, and the two repairs
    /// attempted before that print existed were both aimed at the wrong term —
    /// each exact, each landing a byte-identical failure. A verdict has to
    /// carry the quantity it was decided against (#2465), and this is that
    /// quantity: the first accumulator to leave the finite range, where, and
    /// the `q = e^{−ρ}` it was evaluated at.
    AccumulatorDiverged {
        node: usize,
        n_proper: usize,
        accumulator: &'static str,
        value: f64,
        lo: f64,
        hi: f64,
        q_value: f64,
        /// This node's own contribution to that accumulator. A finite running
        /// sum plus an infinite total means the CONTRIBUTION diverged, so this
        /// is the term to look at, not the sum.
        contribution_lo: f64,
        contribution_hi: f64,
        /// The third covariance-derivative entry `F'''` every third-order chain
        /// rule on this path divides by `F`. Reported alongside so a wide
        /// contribution can be told from a wide INPUT: if `F'''` is already
        /// unbounded the covariance jet is at fault, and if it is tight while
        /// the contribution is not, the cancellation in the chain rule is.
        f_star_d3_lo: f64,
        f_star_d3_hi: f64,
        /// The same entry AFTER this node's measurement update. `F'''` above is
        /// the PREDICTED value, so the pair localises the growth to one of the
        /// filter's two steps: an updated entry much narrower than the
        /// predicted one means the update contracts and the PREDICTION is
        /// growing it, and the reverse means the update is.
        updated_d3_lo: f64,
        updated_d3_hi: f64,
    },
    InvalidInput(String),
    MissingEndpointCertificate {
        log_lambda: f64,
    },
    GlobalValueOrderingUnresolved {
        maximum_excess: f64,
        comparison_resolution: f64,
    },
    OptimumKktUncertified {
        location: ScoreOptimumLocation,
        bracket: ClosedInterval,
        derivative: ClosedInterval,
        curvature: ClosedInterval,
    },
    Search(String),
    Computation(String),
}

impl std::fmt::Display for SplineScoreProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InnovationContainsZero {
                node,
                kind,
                enclosure,
            } => write!(
                f,
                "spline scan: {kind:?} innovation ball at node {node} contains zero: {enclosure:?}"
            ),
            Self::NonPositiveInnovation {
                node,
                kind,
                enclosure,
            } => write!(
                f,
                "spline scan: {kind:?} innovation ball at node {node} is nonpositive: {enclosure:?}"
            ),
            Self::NonPositiveProfileResidual { enclosure } => write!(
                f,
                "spline scan: profiled residual ball is not strictly positive: {enclosure:?}"
            ),
            Self::InvalidArithmetic { context } => {
                write!(
                    f,
                    "spline scan: non-finite interval arithmetic in {context}"
                )
            }
            Self::AccumulatorDiverged {
                node,
                n_proper,
                accumulator,
                value,
                lo,
                hi,
                q_value,
                contribution_lo,
                contribution_hi,
                f_star_d3_lo,
                f_star_d3_hi,
                updated_d3_lo,
                updated_d3_hi,
            } => {
                write!(
                    f,
                    "spline scan: certified accumulator `{accumulator}` left the finite range at \
                     node {node} (proper innovations so far {n_proper}, q = {q_value:.6e}): \
                     value {value:.9e} in [{lo:.9e}, {hi:.9e}]; this node's contribution was \
                     [{contribution_lo:.9e}, {contribution_hi:.9e}], predicted F''' was \
                     [{f_star_d3_lo:.9e}, {f_star_d3_hi:.9e}] and updated F''' was \
                     [{updated_d3_lo:.9e}, {updated_d3_hi:.9e}]"
                )
            }
            Self::InvalidInput(reason) => f.write_str(reason),
            Self::MissingEndpointCertificate { log_lambda } => write!(
                f,
                "spline scan: certified search requested an uncached endpoint {log_lambda}"
            ),
            Self::GlobalValueOrderingUnresolved {
                maximum_excess,
                comparison_resolution,
            } => write!(
                f,
                "spline scan: the selected REML representative can trail another exact \
                 candidate by {maximum_excess}, beyond the certified comparison resolution \
                 {comparison_resolution}"
            ),
            Self::OptimumKktUncertified {
                location,
                bracket,
                derivative,
                curvature,
            } => write!(
                f,
                "spline scan: exact-real REML KKT condition is uncertified for {location:?} \
                 on {bracket:?} (derivative {derivative:?}, curvature {curvature:?})"
            ),
            Self::Search(reason) => {
                write!(f, "spline scan: REML stationary isolation failed: {reason}")
            }
            Self::Computation(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for SplineScoreProofError {}

impl From<String> for SplineScoreProofError {
    fn from(reason: String) -> Self {
        Self::Computation(reason)
    }
}

#[inline]
fn require_positive_innovation(
    node: usize,
    kind: SplineInnovationKind,
    innovation: Ball,
) -> Result<(), SplineScoreProofError> {
    if !innovation.is_finite() {
        return Err(SplineScoreProofError::InvalidArithmetic {
            context: "innovation recurrence",
        });
    }
    let enclosure = innovation.interval();
    if innovation.lo <= 0.0 && innovation.hi >= 0.0 {
        Err(SplineScoreProofError::InnovationContainsZero {
            node,
            kind,
            enclosure,
        })
    } else if innovation.hi < 0.0 {
        Err(SplineScoreProofError::NonPositiveInnovation {
            node,
            kind,
            enclosure,
        })
    } else {
        Ok(())
    }
}

#[inline]
fn next_down_ball(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

#[inline]
fn next_up_ball(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

#[inline]
fn mat_mul(a: &Mat2, b: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..m {
                acc += a[i][k] * b[k][j];
            }
            c[i][j] = acc;
        }
    }
    c
}

#[inline]
fn mat_t(a: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[j][i];
        }
    }
    c
}

#[inline]
fn mat_vec(a: &Mat2, v: &Vec2, m: usize) -> Vec2 {
    let mut out = [0.0; MAX_ORDER];
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..m {
            acc += a[i][j] * v[j];
        }
        out[i] = acc;
    }
    out
}

#[inline]
fn mat_add(a: &Mat2, b: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

#[inline]
fn mat_sub(a: &Mat2, b: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// Inverse of an `m × m` (`m ∈ {1, 2, 3}`) with a hard singularity error.
/// Closed-form cofactor inverses keep the hot-loop arithmetic exact and
/// branch-free; order 3 is the quintic smoother's state dimension (#1044).
fn mat_inv(a: &Mat2, m: usize, what: &str) -> Result<Mat2, String> {
    let mut out = [[0.0; MAX_ORDER]; MAX_ORDER];
    match m {
        1 => {
            let d = a[0][0];
            if !(d.is_finite() && d.abs() > 0.0) {
                return Err(format!("spline scan: singular 1x1 in {what} (a00={d})"));
            }
            out[0][0] = 1.0 / d;
        }
        2 => {
            let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            if !(det.is_finite() && det.abs() > 0.0) {
                return Err(format!("spline scan: singular 2x2 in {what} (det={det})"));
            }
            out[0][0] = a[1][1] / det;
            out[0][1] = -a[0][1] / det;
            out[1][0] = -a[1][0] / det;
            out[1][1] = a[0][0] / det;
        }
        3 => {
            // Cofactor / adjugate inverse. Cofactors of the 2×2 minors:
            let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
            let c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
            let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
            let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
            if !(det.is_finite() && det.abs() > 0.0) {
                return Err(format!("spline scan: singular 3x3 in {what} (det={det})"));
            }
            let inv_det = 1.0 / det;
            // inv = adj/det = (cofactor matrix)ᵀ / det.
            out[0][0] = c00 * inv_det;
            out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
            out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
            out[1][0] = c01 * inv_det;
            out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
            out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
            out[2][0] = c02 * inv_det;
            out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
            out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
        }
        _ => return Err(format!("spline scan: unsupported order {m} in {what}")),
    }
    Ok(out)
}

/// Inverse of a general dense `d × d` SPD matrix via Gauss–Jordan elimination
/// with partial pivoting, symmetric diagonal (Jacobi) equilibration, and one
/// iterative-refinement step. Used once per fit by the leading-block diffuse
/// smoother (dimension `(order−1)·order ≤ 6`), so clarity over speed — it is
/// NOT on the hot REML grid path (that runs only `run_filter`).
///
/// Equilibration matters at order `m ≥ 3`: the IWP process noise `Q(δ)` scales
/// the `f^{(k)}` state components by `δ^{2m−1}` down to `δ`, so its inverse
/// `(qQ)⁻¹` — and hence the leading-block precision `Λ` — spans many orders of
/// magnitude (the f-component carries the `O(w)` observation term, the
/// high-derivative components carry `O(1/(qδ^{2m−1}))` penalty mass). A bare
/// Gauss–Jordan inverse of such a `Λ` loses `≈ ε·κ(Λ)` digits, which at heavy
/// smoothing (small `q`) would corrupt the quintic's leading smoothed nodes.
/// Rescaling to unit diagonal (`Λ̃ = SΛS`, `s_i = 1/√Λ_ii`) collapses that
/// scale disparity before the elimination, then `Λ⁻¹ = S Λ̃⁻¹ S`.
fn dense_spd_inverse(a: &[Vec<f64>], what: &str) -> Result<Vec<Vec<f64>>, String> {
    let d = a.len();
    // Jacobi equilibration scale s_i = 1/√Λ_ii (Λ SPD ⇒ Λ_ii > 0).
    let s: Vec<f64> = (0..d)
        .map(|i| {
            let dii = a[i][i];
            if dii.is_finite() && dii > 0.0 {
                1.0 / dii.sqrt()
            } else {
                1.0
            }
        })
        .collect();
    let a_s: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..d).map(|j| s[i] * a[i][j] * s[j]).collect())
        .collect();
    // Gauss–Jordan inverse of the equilibrated matrix.
    let mut inv_s = gauss_jordan_inverse(&a_s, what)?;
    // One iterative-refinement step against the equilibrated system:
    // X ← X + X·(I − Λ̃·X), reducing the residual to near machine precision.
    let mut resid = vec![vec![0.0_f64; d]; d]; // R = I − Λ̃·X
    for i in 0..d {
        for j in 0..d {
            let mut ax = 0.0;
            for k in 0..d {
                ax += a_s[i][k] * inv_s[k][j];
            }
            resid[i][j] = f64::from(u8::from(i == j)) - ax;
        }
    }
    let mut delta = vec![vec![0.0_f64; d]; d]; // ΔX = X·R
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0;
            for k in 0..d {
                acc += inv_s[i][k] * resid[k][j];
            }
            delta[i][j] = acc;
        }
    }
    for i in 0..d {
        for j in 0..d {
            inv_s[i][j] += delta[i][j];
        }
    }
    // Un-equilibrate: Λ⁻¹ = S·Λ̃⁻¹·S.
    Ok((0..d)
        .map(|i| (0..d).map(|j| s[i] * inv_s[i][j] * s[j]).collect())
        .collect())
}

/// Gauss–Jordan inverse with partial pivoting (helper for `dense_spd_inverse`).
fn gauss_jordan_inverse(a: &[Vec<f64>], what: &str) -> Result<Vec<Vec<f64>>, String> {
    let d = a.len();
    let mut aug = a.to_vec();
    let mut inv = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        inv[i][i] = 1.0;
    }
    for col in 0..d {
        let piv = (col..d)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .ok_or_else(|| {
                format!("spline scan: no pivot candidate in column {col} of {d} in {what}")
            })?;
        let p = aug[piv][col];
        if !(p.is_finite() && p.abs() > 0.0) {
            return Err(format!(
                "spline scan: singular {d}x{d} in {what} (pivot={p})"
            ));
        }
        aug.swap(col, piv);
        inv.swap(col, piv);
        let d_piv = aug[col][col];
        for k in 0..d {
            aug[col][k] /= d_piv;
            inv[col][k] /= d_piv;
        }
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = aug[r][col];
            if f == 0.0 {
                continue;
            }
            for k in 0..d {
                aug[r][k] -= f * aug[col][k];
                inv[r][k] -= f * inv[col][k];
            }
        }
    }
    Ok(inv)
}

/// Factorials `k!` for `k ≤ 2·MAX_ORDER` — the only ones the order-`m`
/// transition and process-noise formulas reference.
#[inline]
fn factorial(k: usize) -> f64 {
    (1..=k).map(|v| v as f64).product::<f64>().max(1.0)
}

/// Transition `F(δ) = exp(δ·A)` of the `m`-th order integrated Wiener process,
/// `A` the nilpotent shift: `F[i][j] = δ^{j−i}/(j−i)!` for `j ≥ i`, else 0.
/// `m = 1 ⇒ [[1]]`; `m = 2 ⇒ [[1, δ], [0, 1]]` (the cubic case, unchanged).
#[inline]
fn transition(delta: f64, m: usize) -> Mat2 {
    let mut f = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in i..m {
            f[i][j] = delta.powi((j - i) as i32) / factorial(j - i);
        }
    }
    f
}

/// Process noise `Q(δ) = ∫₀^δ e^{As} b bᵀ e^{Aᵀs} ds` (`b = e_{m−1}`) of the
/// `m`-th order IWP at unit `q`, scaled by `q`:
/// `Q[i][j] = q · δ^{2m−1−i−j} / ((m−1−i)! (m−1−j)! (2m−1−i−j))`.
/// `m = 1 ⇒ [[q·δ]]`; `m = 2 ⇒ [[q·δ³/3, q·δ²/2], [q·δ²/2, q·δ]]` (unchanged).
#[inline]
fn process_noise(delta: f64, q: f64, m: usize) -> Mat2 {
    let mut out = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            let p = 2 * m - 1 - i - j;
            out[i][j] = q * delta.powi(p as i32)
                / (factorial(m - 1 - i) * factorial(m - 1 - j) * (p as f64));
        }
    }
    out
}

/// Symmetrize in place against drift from the rank-one update arithmetic.
#[inline]
fn symmetrize(a: &mut Mat2, m: usize) {
    for i in 0..m {
        for j in (i + 1)..m {
            let off = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = off;
            a[j][i] = off;
        }
    }
}

#[inline]
fn ball_mat_mul(a: &BallMat, b: &BallMat, m: usize) -> BallMat {
    let mut c = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            let mut acc = Ball::ZERO;
            for k in 0..m {
                acc = acc.add(a[i][k].mul(b[k][j]));
            }
            c[i][j] = acc;
        }
    }
    c
}

#[inline]
fn ball_mat_t(a: &BallMat, m: usize) -> BallMat {
    let mut c = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[j][i];
        }
    }
    c
}

#[inline]
fn ball_mat_vec(a: &BallMat, v: &BallVec, m: usize) -> BallVec {
    let mut out = [Ball::ZERO; MAX_ORDER];
    for i in 0..m {
        let mut acc = Ball::ZERO;
        for j in 0..m {
            acc = acc.add(a[i][j].mul(v[j]));
        }
        out[i] = acc;
    }
    out
}

#[inline]
fn ball_mat_add(a: &BallMat, b: &BallMat, m: usize) -> BallMat {
    let mut c = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[i][j].add(b[i][j]);
        }
    }
    c
}

#[inline]
fn ball_mat_sub(a: &BallMat, b: &BallMat, m: usize) -> BallMat {
    let mut c = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[i][j].sub(b[i][j]);
        }
    }
    c
}

#[inline]
fn ball_transition(delta: Ball, m: usize) -> BallMat {
    let mut f = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        let mut power = Ball::ONE;
        for j in i..m {
            if j > i {
                power = power.mul(delta);
            }
            f[i][j] = power.div_positive(Ball::exact(factorial(j - i)));
        }
    }
    f
}

#[inline]
fn ball_unit_process_noise(delta: Ball, m: usize) -> BallMat {
    let mut powers = [Ball::ONE; 2 * MAX_ORDER];
    for exponent in 1..powers.len() {
        powers[exponent] = powers[exponent - 1].mul(delta);
    }
    let mut out = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            let exponent = 2 * m - 1 - i - j;
            let denominator = factorial(m - 1 - i) * factorial(m - 1 - j) * exponent as f64;
            out[i][j] = powers[exponent].div_positive(Ball::exact(denominator));
        }
    }
    out
}

/// Taylor-model decomposition of one process-noise injection.
///
/// At a fixed certified score evaluation, `q` is one exact real number in its
/// ball, shared by every node. Write
///
/// ```text
/// q = q₀ + r_q θ_q,       θ_q ∈ [-1, 1],
/// Q_ij = c₀,ij + ε_ij,
/// q Q_ij = q₀ c₀,ij + (r_q c₀,ij) θ_q + q ε_ij.
/// ```
///
/// The first term is the zonotope centre, the second is one distinguished
/// generator accumulated across the whole recursion, and only `q ε` plus the
/// floating-point error in forming the first two coefficients is an
/// independent remainder. Treating the full interval `qQ` as a fresh constant
/// at every node discards the identity of `θ_q`; after enough Riccati steps its
/// box hull is wider than the score resolution even though the signed
/// recursion is contracting.
struct ProcessNoiseTaylor {
    /// Ordinary interval enclosure used by the independent componentwise path.
    enclosure: BallMat,
    /// Centre plus deterministic-arithmetic/nonlinear remainder, with the
    /// first-order common-`q` uncertainty removed.
    constant: [Ball; COVARIANCE_D1_DIM],
    /// Coefficient added to the one common normalized `q` generator.
    shared_q: [f64; COVARIANCE_D1_DIM],
}

#[inline]
fn ball_process_noise_taylor(delta: Ball, q: Ball, m: usize) -> ProcessNoiseTaylor {
    let unit = ball_unit_process_noise(delta, m);
    let q_radius = ball_radius_about_value(q);
    let mut enclosure = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    let mut constant = [Ball::ZERO; COVARIANCE_D1_DIM];
    let mut shared_q = [0.0_f64; COVARIANCE_D1_DIM];

    for i in 0..m {
        for j in 0..m {
            let index = i * m + j;
            let coefficient = unit[i][j];
            enclosure[i][j] = q.mul(coefficient);

            let coefficient_center = Ball::exact(coefficient.value);
            let center_product = Ball::exact(q.value).mul(coefficient_center);
            let shared_product = Ball::exact(q_radius).mul(coefficient_center);

            // `q·(c-c₀)` is the nonlinear/deterministic interval remainder.
            // The other two terms charge rounding of the stored centre and
            // shared-generator coefficients. The latter is multiplied by an
            // unknown `θ_q`, so its signed error is symmetrized.
            let coefficient_error = coefficient.sub(coefficient_center);
            let center_error = center_product.sub(Ball::exact(center_product.value));
            let shared_error = shared_product.sub(Ball::exact(shared_product.value));
            let shared_error_radius = ball_radius_about_value(shared_error);
            let shared_error_symmetric = Ball {
                value: 0.0,
                lo: -shared_error_radius,
                hi: shared_error_radius,
            };
            let remainder = q
                .mul(coefficient_error)
                .add(center_error)
                .add(shared_error_symmetric);

            constant[index] = Ball::exact(center_product.value).add(remainder);
            shared_q[index] = shared_product.value;
        }
    }

    ProcessNoiseTaylor {
        enclosure,
        constant,
        shared_q,
    }
}

#[inline]
fn ball_symmetrize(a: &mut BallMat, m: usize) {
    for i in 0..m {
        for j in (i + 1)..m {
            let off = a[i][j].add(a[j][i]).scale(0.5);
            a[i][j] = off;
            a[j][i] = off;
        }
    }
}

/// `A = I − K e₀ᵀ`, built so its `(0,0)` entry is never a subtraction (#2614).
///
/// The expanded per-entry form of the congruence `A X Aᵀ` evaluates
/// `X[0][0]·(1 − 2K₀ + K₀²)` — that is `X[0][0]·(1 − K₀)²` written as three
/// terms. In the saturated regime `K₀ = P₀₀/F → 1`, so it is `1 − 1` twice
/// over: the value collapses and the interval width does not. Measured
/// consequence: the third covariance-derivative entry reached an enclosure of
/// `+/-4.1e247` by node 62, growing about `10^4` per node, while `d1` and `d2`
/// — which carry fewer such terms — stayed clean.
///
/// The identity that removes it is exact and involves no subtraction:
///
/// ```text
///   A[0][0] = 1 − K₀ = 1 − P₀₀/F = (F − P₀₀)/F = R/F
/// ```
///
/// since `F = P₀₀ + R` by construction. So that entry is `r·inv_f` directly.
/// Away from the first column `A` is the identity; below the diagonal in the
/// first column it is `−K_i`. Neither cancels, so building `A` and multiplying
/// is ordinary arithmetic on well-conditioned entries.
#[inline]
fn ball_update_operator(gain: &BallVec, a_diag_zero: Ball, order: usize) -> BallMat {
    let mut a = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for (i, row) in a.iter_mut().enumerate().take(order) {
        row[i] = Ball::ONE;
    }
    a[0][0] = a_diag_zero;
    for i in 1..order {
        a[i][0] = gain[i].neg();
    }
    a
}

/// `A⁽ᵏ⁾ = −K⁽ᵏ⁾ e₀ᵀ`, the ρ-derivative of [`ball_update_operator`].
///
/// The identity part differentiates away and only the first column survives, so
/// there is no cancelling entry here at all.
#[inline]
fn ball_update_operator_derivative(gain_jet: &BallVec, order: usize) -> BallMat {
    let mut a = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        a[i][0] = gain_jet[i].neg();
    }
    a
}

/// `L · X · Rᵀ` for `Ball` matrices.
#[inline]
fn ball_congruence(l: &BallMat, x: &BallMat, r_side: &BallMat, order: usize) -> BallMat {
    ball_mat_mul(
        &ball_mat_mul(l, x, order),
        &ball_mat_t(r_side, order),
        order,
    )
}

/// Intersect the proper innovation variance with its exact lower bound `R_t`.
///
/// `F̃_t = H P*_t Hᵀ + R_t`, and `P*` is a covariance, so `H P* Hᵀ ≥ 0` exactly
/// and therefore `F̃_t ≥ R_t > 0`. The componentwise interval evaluation of `P*`
/// forgets that: after enough rank-one updates its diagonal enclosure widens,
/// and once `P*[0][0].lo` drops below zero the enclosure of `F̃` reaches down
/// toward zero even though the true value cannot. `inv_f = 1/F̃` then has an
/// enclosure reaching `+∞` — and EVERY gain, every `log F̃`, and every one of
/// the three derivative recursions multiplies by it.
///
/// That is the observed failure, on three surfaces: `gam-solve`'s own
/// `spline_scan` tests refuse with `InvalidArithmetic{"diffuse filter
/// accumulator"}` carrying `[-inf, +inf]` derivative enclosures around exact
/// centre values; `gam-models`' `spline_scan_payload_round_trips_and_validates`
/// dies on its first line; and the Python surface reports `IntegrationError:
/// spline scan: non-finite interval arithmetic in proper covariance PSD
/// intersection` (#2614, #2616).
///
/// Restoring this bound is the same move [`intersect_proper_covariance_psd`]
/// already makes for the covariance diagonal, and its own comment states the
/// principle: "This intersection restores proof information supplied by the
/// statistical model; it is not a numerical tolerance." Here the information is
/// stronger, because `R_t = 1/w_t` is an exact input rather than a computed
/// quantity — the observation variance is data, not arithmetic.
///
/// The floor is capped at the ball's own `value` and `hi` so the result stays a
/// well-formed enclosure (`lo ≤ value ≤ hi`) even if the centre has itself gone
/// non-positive; an inconsistent enclosure is left inconsistent for the finite
/// checks downstream to reject, rather than being papered into consistency here.
#[inline]
fn intersect_innovation_above_observation_variance(
    innovation: &mut Ball,
    observation_variance: Ball,
) {
    let floor = observation_variance
        .lo
        .min(innovation.value)
        .min(innovation.hi);
    if floor.is_finite() && innovation.lo < floor {
        innovation.lo = floor;
    }
}

/// Directed square root of a nonnegative enclosure.
#[inline]
fn ball_sqrt(value: Ball) -> Option<Ball> {
    if !(value.lo >= 0.0 && value.hi.is_finite() && value.lo <= value.hi) {
        return None;
    }
    Some(Ball {
        value: value.value.max(0.0).sqrt(),
        lo: next_down_ball(value.lo.sqrt()).max(0.0),
        hi: next_up_ball(value.hi.sqrt()),
    })
}

/// Lower-triangular Cholesky factor `L` with `P = L Lᵀ`, in directed arithmetic.
///
/// `None` when a pivot enclosure fails to be strictly positive, which is a
/// statement about the enclosure and not about the matrix — the caller treats a
/// missing factor as an absence of evidence, never as a refusal.
fn ball_cholesky(covariance: &BallMat, order: usize) -> Option<BallMat> {
    let mut factor = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        for j in 0..=i {
            let mut accumulator = covariance[i][j];
            for k in 0..j {
                accumulator = accumulator.sub(factor[i][k].mul(factor[j][k]));
            }
            if i == j {
                factor[i][j] = ball_sqrt(accumulator)?;
                if !(factor[i][j].lo > 0.0) {
                    return None;
                }
            } else {
                if !(factor[j][j].lo > 0.0) {
                    return None;
                }
                factor[i][j] = accumulator.div_positive(factor[j][j]);
            }
        }
    }
    Some(factor)
}

/// How many accumulators the per-node divergence check scans.
///
/// The second- and third-order accumulators are ordered last and left OUT of
/// the scan: each has a closed-form global bound the certificate substitutes
/// (see [`BoundSource`]), so neither can justify discarding a value and slope
/// that are finite. Divergence in the VALUE or in the FIRST derivative still
/// refuses, at the node it happened — those have no substitute.
const GLOBALLY_BOUNDED_FROM: usize = 4;

/// Number of columns in the prediction prearray `[F·L, L_Q]`.
const PREARRAY_COLUMNS: usize = 2 * MAX_ORDER;

/// The measurement update, performed on a CARRIED FACTOR of the covariance
/// instead of on the covariance.
///
/// With `P⁻ = L Lᵀ` and `L` lower triangular, `Lᵀe₀` has a single nonzero, so
/// the whole Kalman update is one column scaling:
///
/// ```text
/// L⁺ = L · diag(β, 1, …, 1),      β = √(R/F),
/// ```
///
/// and `P⁺ = L⁺L⁺ᵀ` is EXACTLY `P⁻ − M Mᵀ/F`. The check: `P[i][0] = L[i][0]L₀₀`
/// because `L[0][k] = 0` for `k ≥ 1`, so
///
/// ```text
/// (L⁺L⁺ᵀ)[i][j] = P[i][j] − (1 − β²)L[i][0]L[j][0]
///               = P[i][j] − (P₀₀/F)·M[i]M[j]/P₀₀
///               = P[i][j] − M[i]M[j]/F.
/// ```
///
/// CARRIED is the load-bearing word. Recomputing the factorization per node
/// from the componentwise covariance was measured and is INERT — bit-identical
/// divergence nodes `44/44/44/45/50/88/164` at order 2 — because the Cholesky's
/// own Schur complement `L₁₁² = P₁₁ − P₀₁²/P₀₀` IS the cancelling subtraction it
/// was meant to avoid, so factoring and immediately reconstructing recomputes
/// exactly the quantity whose width is the problem. Carried across nodes, that
/// difference is never re-formed: `L₁₁` is scaled and rotated, never subtracted,
/// so its enclosure tracks the size of the RESULT instead of the size of the
/// operands it would have been differenced out of.
fn ball_factor_update(factor: &BallMat, beta: Ball, order: usize) -> BallMat {
    let mut updated = *factor;
    for row in updated.iter_mut().take(order) {
        row[0] = row[0].mul(beta);
    }
    updated
}

/// `L Lᵀ` for a lower-triangular factor.
fn ball_factor_gram(factor: &BallMat, order: usize) -> BallMat {
    let mut gram = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        for j in 0..order {
            let mut accumulator = Ball::ZERO;
            for k in 0..=i.min(j) {
                accumulator = accumulator.add(factor[i][k].mul(factor[j][k]));
            }
            gram[i][j] = accumulator;
        }
    }
    gram
}

/// Re-triangularize the prediction prearray `A = [F·L⁺, L_Q]` so that the
/// carried factor stays `order`-wide, WITHOUT assuming exact orthogonality.
///
/// Givens rotations are applied to pairs of COLUMNS with `c` and `s` taken as
/// floating-point points, never intervals. That is what makes the result sound:
/// a 2×2 `Θ = [[c, −s], [s, c]]` built from points satisfies `ΘΘᵀ = (c²+s²)I`
/// EXACTLY, so applying it to every row rescales the Gram by the scalar
/// `c²+s²` and introduces no other error — a computed rotation that is not
/// quite orthogonal is a similarity scaling, not a general perturbation. The
/// product of those scalars is returned so the caller can divide it back out.
/// Had `c` and `s` been intervals, the enclosure would have ranged over
/// non-orthogonal `Θ`s and `L Lᵀ` would no longer have enclosed `A Aᵀ`.
///
/// The trailing columns are not exactly zeroed either, so their outer product
/// `D = Σ_{k ≥ order} A[:,k]A[:,k]ᵀ` is returned as `trace(D)`, which bounds
/// every entry of `D` because `|D[i][j]| ≤ √(D_ii·D_jj) ≤ trace(D)`.
fn ball_retriangularize(
    prearray: &mut [[Ball; PREARRAY_COLUMNS]; MAX_ORDER],
    order: usize,
    columns: usize,
) -> (BallMat, f64, f64) {
    let mut gram_scale = 1.0_f64;
    for i in 0..order {
        for k in (i + 1)..columns {
            let a = prearray[i][i].value;
            let b = prearray[i][k].value;
            let radius = (a * a + b * b).sqrt();
            if !(radius.is_finite() && radius > 0.0) {
                continue;
            }
            let cosine = a / radius;
            let sine = b / radius;
            gram_scale = next_up_ball(gram_scale * (cosine * cosine + sine * sine));
            for row in prearray.iter_mut().take(order) {
                let x = row[i];
                let y = row[k];
                row[i] = x.scale(cosine).add(y.scale(sine));
                row[k] = y.scale(cosine).sub(x.scale(sine));
            }
        }
    }
    let mut factor = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        for j in 0..=i {
            factor[i][j] = prearray[i][j];
        }
    }
    let mut trailing = 0.0_f64;
    for row in prearray.iter().take(order) {
        for entry in row.iter().take(columns).skip(order) {
            let magnitude = entry.lo.abs().max(entry.hi.abs());
            trailing = next_up_ball(trailing + next_up_ball(magnitude * magnitude));
        }
    }
    (factor, trailing, gram_scale)
}

/// Intersect an enclosure with an independently derived enclosure of the same/// Intersect an enclosure with an independently derived enclosure of the same
/// quantity.
///
/// Both bound the same real number, so their intersection does too, and it is
/// no wider than either. A bound is never moved past the nearest-rounded
/// `value`, which keeps `lo <= value <= hi` an invariant of the type rather
/// than something each caller has to re-establish.
#[inline]
fn intersect_with_independent_enclosure(entry: &mut Ball, evidence: Ball) {
    let floor = evidence.lo.min(entry.value);
    if floor.is_finite() && entry.lo < floor {
        entry.lo = floor;
    }
    let ceiling = evidence.hi.max(entry.value);
    if ceiling.is_finite() && entry.hi > ceiling {
        entry.hi = ceiling;
    }
}

/// Intersect the FIRST-derivative covariance with the two-sided range the
/// covariance itself gives it: `0 ⪯ −dP/dρ ⪯ P`.
///
/// The file already uses the left half — "`P` is operator monotone increasing in
/// the process-noise scale `q = e^{−ρ}`, so `dP/dρ ⪯ 0`" — to carry a Cholesky
/// factor of `−dP/dρ`. The right half comes from the same map being operator
/// CONCAVE in `q`, which is what actually bounds the derivative:
///
/// * the measurement update `P ↦ P − PHᵀ(HPHᵀ+R)⁻¹HP = (P⁻¹ + HᵀR⁻¹H)⁻¹` is the
///   parallel sum, which is operator concave and operator monotone;
/// * the prediction `P ↦ FPFᵀ + qQ` is affine and jointly monotone in `(P, q)`;
/// * the seed is affine in `q`.
///
/// A composition of operator-concave monotone maps with affine inner maps is
/// operator concave, so `q ↦ P_t(q)` is. Concavity at `q` against `0` gives
/// `P_t(0) ⪰ P_t(q) − q·dP_t/dq`, i.e.
///
/// ```text
///   0 ⪯ −dP/dρ = q·dP/dq ⪯ P(q) − P(0) ⪯ P(q),
/// ```
///
/// the last step because `P_t(0) ⪰ 0` is a covariance. Only the DIAGONAL of a
/// semidefinite ordering transfers entrywise, so that is what is intersected
/// here; [`intersect_covariance_minors`] then carries it to the off-diagonals
/// through `|X_ij| ≤ √(X_ii X_jj)`, applied to `−dP/dρ`, which is the PSD
/// matrix of the pair.
///
/// Measured need (#2614, order 3, ρ = −16.6135, dgp_2300): at node 40 the value
/// covariance `P⁺₀₀` holds width `8.4e−3` while `dP⁺₀₀/dρ` — a quantity this
/// bound pins into `[−0.74, 0]` — carries width `4.97e5`, and by node 80 it is
/// `2.5e149`. Every gain jet is a column of that matrix divided by `R`, so the
/// mean jet, `v′`, and `Σ v²/F̃`'s derivative inherit it directly.
fn intersect_derivative_covariance_below_its_own_covariance(
    derivative: &mut BallMat,
    covariance: &BallMat,
    order: usize,
) {
    for i in 0..order {
        intersect_with_exact_range(&mut derivative[i][i], -covariance[i][i].hi, 0.0);
    }
    let mut negated = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        for j in 0..order {
            negated[i][j] = derivative[i][j].neg();
        }
    }
    intersect_covariance_minors(&mut negated, order);
    for i in 0..order {
        for j in 0..order {
            intersect_with_independent_enclosure(&mut derivative[i][j], negated[i][j].neg());
        }
    }
}

/// Intersect an enclosure with an exact two-sided range of the same quantity.
///
/// Same rule as [`intersect_with_independent_enclosure`], for evidence that is a
/// pair of bounds rather than a computed ball: bounds move only inward and never
/// past the nearest-rounded `value`, so `lo <= value <= hi` stays an invariant of
/// the type.
#[inline]
fn intersect_with_exact_range(entry: &mut Ball, lo: f64, hi: f64) {
    let floor = lo.min(entry.value);
    if floor.is_finite() && entry.lo < floor {
        entry.lo = floor;
    }
    let ceiling = hi.max(entry.value);
    if ceiling.is_finite() && entry.hi > ceiling {
        entry.hi = ceiling;
    }
}

/// Intersect the two ORDER-ONE accumulators with the exact ranges the Gaussian
/// model gives them, at every node.
///
/// Once the diffuse rank is consumed, the accumulated pair over the data prefix
/// `y` seen so far IS the restricted Gaussian pair for that prefix: with
/// `V = R + qG` (`R = diag(1/w)`, `G ⪰ 0` the process-noise Gram, `q = e^{−ρ}`),
/// `X` the diffuse polynomial design and
/// `P = V⁻¹ − V⁻¹X(XᵀV⁻¹X)⁻¹XᵀV⁻¹` its restricted inverse,
///
/// ```text
///   Σ v²/F̃ = yᵀPy,     Σ log F̃ = log|V| + log|XᵀV⁻¹X|.
/// ```
///
/// `dV/dρ = −qG = −(V − R)` and `dP/dρ = −P(dV/dρ)P = P − PRP`, so
///
/// ```text
///   d/dρ Σ v²/F̃  = yᵀPy − yᵀPRPy,
///   d/dρ Σ log F̃ = tr(P dV/dρ) = −(tr(PV) − tr(PR)).
/// ```
///
/// Every piece of that is signed by a semidefinite fact and nothing else:
/// `R ⪰ 0` gives `yᵀPRPy ≥ 0`; `V − R = qG ⪰ 0` with `PVP = P` gives
/// `yᵀPRPy ≤ yᵀPVPy = yᵀPy` and `tr(PR) ≤ tr(PV)`; and `tr(PV) = n_proper`
/// exactly. Hence, EXACTLY,
///
/// ```text
///   0 ≤ Σ v²/F̃ ≤ yᵀR⁻¹y = Σ w y²,     0 ≤ d/dρ Σ v²/F̃ ≤ Σ v²/F̃,
///   −n_proper ≤ d/dρ Σ log F̃ ≤ 0,
/// ```
///
/// the first ceiling because `P ⪯ V⁻¹ ⪯ R⁻¹`.
///
/// Why this is the intersection that matters (#2614, measured): at smoothing
/// order 2 on the #2300 nodes the search refuses at `ρ = −18` with a certified
/// criterion DERIVATIVE ball of `±1.95e91` and a cell value enclosure of
/// `±1.58e83` — both finite, so no accumulator check fires, and both useless:
/// `strict_sign` cannot sign a `±1e91` interval, so the search can neither
/// bracket a stationary point nor exclude one, and reports `Unresolved`. The
/// true derivative is bounded by `½(n_proper + ν)`, i.e. by `178` on that
/// fixture. The entire `1e89` excess is dependency loss in the jet recursions,
/// and it is the running SUM that carries it forward, so the bound is applied
/// per node rather than once at the end: a partial sum is the same quantity for
/// its own prefix, which is what makes that legitimate.
///
/// This restores proof information supplied by the statistical model; it is not
/// a numerical tolerance, and it cannot make a false statement true, because
/// each bound is a property of the exact real number the enclosure encloses.
#[inline]
fn intersect_first_order_accumulator_exact_ranges(
    quadratic: &mut Ball,
    quadratic_d1: &mut Ball,
    log_determinant_d1: &mut Ball,
    weighted_energy: Ball,
    n_proper: usize,
) {
    intersect_with_exact_range(quadratic, 0.0, weighted_energy.hi);
    intersect_with_exact_range(quadratic_d1, 0.0, quadratic.hi);
    intersect_with_exact_range(log_determinant_d1, -(n_proper as f64), 0.0);
}

/// Intersect the observed coordinate of the UPDATED covariance with the exact
/// range of its own defining map.
///
/// At the observed coordinate the Kalman update is the scalar map
///
/// ```text
/// P⁺₀₀ = P₀₀·R/(P₀₀ + R),
/// ```
///
/// whose partial derivatives `(R/F)²` and `(P₀₀/F)²` are both strictly positive
/// on `P₀₀ ≥ 0 < R`. A function monotone in each argument attains its range over
/// a box at the corners, so `[g(P.lo, R.lo), g(P.hi, R.hi)]` is the EXACT range
/// of this update over the input enclosure. Anything wider is dependency loss,
/// not information about the filter.
///
/// The componentwise Joseph evaluation loses exactly that dependency: `A = R/F`
/// and `K = P₀₀/F` are functions of the same `P₀₀` that appears in the middle
/// factor, and interval arithmetic ranges the three independently. Measured on
/// the #2300 nodes (order 1, ρ = 0, node 136, where `R = 1/9`, `P₀₀ = 0.062`,
/// `F = 0.173`): the map's true width factor is `(R/F)² = 0.41`, while the
/// componentwise width factor is
///
/// ```text
/// (R/F)² + 4·P₀₀·R²/F³ = 0.41 + 0.59 = 1.00,
/// ```
///
/// so the contraction is cancelled EXACTLY and the per-node rounding then
/// accumulates without ever being pulled back — the measured 1.28× per node
/// that carries `P⁺₀₀`'s width from `1.6e-14` at node 8 to `2.4e2` at node 140
/// on a value of `0.04`. With the range form the width contracts by `0.41` per
/// node against an additive `Q` term, which has a bounded fixed point.
///
/// Only `lo`/`hi` move, and only inward, and never past `value`: this is an
/// intersection of two valid enclosures of one quantity, so it cannot make a
/// false statement true.
#[inline]
fn intersect_observed_covariance_exact_range(
    updated: &mut Ball,
    predicted: Ball,
    observation_variance: Ball,
) {
    let corner = |p: f64, r: f64| -> Option<Ball> {
        let p = Ball::exact(p.max(0.0));
        let r = Ball::exact(r);
        let f = p.add(r);
        (f.lo > 0.0).then(|| p.mul(r).div_positive(f))
    };
    if let Some(low) = corner(predicted.lo, observation_variance.lo) {
        let floor = low.lo.min(updated.value);
        if floor.is_finite() && updated.lo < floor {
            updated.lo = floor;
        }
    }
    if let Some(high) = corner(predicted.hi, observation_variance.hi) {
        let ceiling = high.hi.max(updated.value);
        if ceiling.is_finite() && updated.hi > ceiling {
            updated.hi = ceiling;
        }
    }
}

/// Intersect a general entry of the UPDATED covariance with the exact range of
/// the Schur complement that defines it.
///
/// [`intersect_observed_covariance_exact_range`] is this argument at the
/// observed coordinate, where the map collapses to one scalar variable. The
/// general entry is
///
/// ```text
/// P⁺[i][j] = P[i][j] − P[i][0]·P[0][j] / (P[0][0] + R),
/// ```
///
/// whose partials with respect to `(a, b, c, d) = (P[i][j], P[i][0], P[0][j],
/// P[0][0])` are `1`, `−c/F`, `−b/F` and `bc/F²`. Whenever `b` and `c` have
/// DEFINITE sign the sign of every partial is fixed over the whole box, so the
/// range is attained at two corners and two evaluations bound it exactly. `R`
/// enters like `d` and moves with it.
///
/// Treating the four as independent is conservative — they are entries of one
/// PSD matrix — so this is a valid enclosure, and it is the tightest available
/// without carrying that correlation. What it removes is the evaluation-order
/// dependency: `F` appears in three factors of the componentwise form and
/// `P[0][0]` in four, and interval arithmetic ranges each occurrence
/// separately. When `b` or `c` straddles zero the monotonicity argument does
/// not hold and nothing is claimed.
fn intersect_updated_covariance_exact_range(
    updated: &mut Ball,
    entry: Ball,
    row: Ball,
    column: Ball,
    observed: Ball,
    observation_variance: Ball,
) {
    // `Some(true)` = nonnegative throughout, `Some(false)` = nonpositive
    // throughout, `None` = straddles zero and the partials change sign.
    let definite_sign = |ball: Ball| -> Option<bool> {
        if ball.lo >= 0.0 {
            Some(true)
        } else if ball.hi <= 0.0 {
            Some(false)
        } else {
            None
        }
    };
    let (Some(row_nonnegative), Some(column_nonnegative)) =
        (definite_sign(row), definite_sign(column))
    else {
        return;
    };
    // `∂/∂d = bc/F²` is nonnegative exactly when `b` and `c` agree in sign.
    let product_nonnegative = row_nonnegative == column_nonnegative;
    let corner = |minimizing: bool| -> Option<Ball> {
        // `∂/∂a = 1`.
        let a = Ball::exact(if minimizing { entry.lo } else { entry.hi });
        // `∂/∂b = −c/F`: decreasing in `b` when `c ≥ 0`.
        let b = Ball::exact(if minimizing == column_nonnegative {
            row.hi
        } else {
            row.lo
        });
        // `∂/∂c = −b/F`, symmetrically.
        let c = Ball::exact(if minimizing == row_nonnegative {
            column.hi
        } else {
            column.lo
        });
        let take_low = minimizing == product_nonnegative;
        let d = Ball::exact(if take_low { observed.lo } else { observed.hi }.max(0.0));
        let variance = Ball::exact(if take_low {
            observation_variance.lo
        } else {
            observation_variance.hi
        });
        let f = d.add(variance);
        (f.lo > 0.0).then(|| a.sub(b.mul(c).div_positive(f)))
    };
    if let Some(low) = corner(true) {
        let floor = low.lo.min(updated.value);
        if floor.is_finite() && updated.lo < floor {
            updated.lo = floor;
        }
    }
    if let Some(high) = corner(false) {
        let ceiling = high.hi.max(updated.value);
        if ceiling.is_finite() && updated.hi > ceiling {
            updated.hi = ceiling;
        }
    }
}

/// Intersect a covariance enclosure with the exact 2×2 minors of the PSD
/// constraint it satisfies.
///
/// [`intersect_proper_covariance_psd`] uses the 1×1 minors — `P[i][i] ≥ 0` —
/// and stops there. The 2×2 minors are equally exact and two-sided:
///
/// ```text
/// P[i][i]·P[j][j] − P[i][j]² ≥ 0   ⇒   |P[i][j]| ≤ √(P[i][i]·P[j][j]),
/// ```
///
/// which bounds every off-diagonal by the diagonals rather than letting it
/// drift on its own. This matters because only the OBSERVED direction is
/// contracted by an update: the componentwise recursion has nothing that pulls
/// an unobserved covariance back, and the transition then mixes that drift into
/// the observed entry as `P₀₀ + 2δP₀₁ + δ²P₁₁`.
///
/// The bound is rounded up by one ulp so that a rounded square root can never
/// claim more than the minor supports.
fn intersect_covariance_minors(covariance: &mut BallMat, order: usize) {
    let sqrt_upper = |value: f64| -> f64 {
        if !(value.is_finite() && value > 0.0) {
            return value;
        }
        let root = value.sqrt();
        if root * root >= value {
            root
        } else {
            f64::from_bits(root.to_bits() + 1)
        }
    };
    for i in 0..order {
        for j in 0..order {
            if i == j {
                continue;
            }
            let diagonal_product = Ball::exact(covariance[i][i].hi.max(0.0))
                .mul(Ball::exact(covariance[j][j].hi.max(0.0)));
            if !diagonal_product.is_finite() {
                continue;
            }
            let bound = sqrt_upper(diagonal_product.hi);
            if !bound.is_finite() {
                continue;
            }
            let entry = &mut covariance[i][j];
            let floor = (-bound).min(entry.value);
            if entry.lo < floor {
                entry.lo = floor;
            }
            let ceiling = bound.max(entry.value);
            if entry.hi > ceiling {
                entry.hi = ceiling;
            }
        }
    }
}

/// Intersect a proper covariance enclosure with its exact PSD invariant.
///
/// Once the diffuse rank is exhausted, `P*` is the conditional covariance of
/// the state. Its diagonal is therefore nonnegative at every measurement
/// update and prediction. A componentwise interval evaluation of
/// `P - PH'(HPH' + R)⁻¹HP` forgets that dependency and can widen a diagonal
/// through zero after repeated rank-one subtractions, even though the exact
/// innovation is bounded below by the positive observation variance `R`.
///
/// This intersection restores proof information supplied by the statistical
/// model; it is not a numerical tolerance. A wholly negative or non-finite
/// diagonal still signals an inconsistent enclosure and fails closed.
#[inline]
fn intersect_proper_covariance_psd(
    covariance: &mut BallMat,
    order: usize,
) -> Result<(), SplineScoreProofError> {
    for (index, row) in covariance.iter_mut().enumerate().take(order) {
        let diagonal = &mut row[index];
        if !diagonal.is_finite() || diagonal.hi < 0.0 {
            return Err(SplineScoreProofError::InvalidArithmetic {
                context: "proper covariance PSD intersection",
            });
        }
        diagonal.lo = diagonal.lo.max(0.0);
    }
    Ok(())
}

/// Per-node filter storage needed by the RTS backward pass.
struct FilterStep {
    /// Filtered mean `a_{t|t}` and proper covariance `P*_{t|t}`.
    a_filt: Vec2,
    p_filt: Mat2,
    /// One-step prediction `a_{t|t-1}`, proper covariance `P*_{t|t-1}` (for t ≥ 1).
    a_pred: Vec2,
    p_pred: Mat2,
}

/// Output of one full filter pass at a fixed `q = 1/λ` (run at unit σ²).
struct FilterPass {
    steps: Vec<FilterStep>,
    /// Σ over proper steps of `log F̃_t` (innovation variances at σ²=1).
    sum_log_f: f64,
    /// First three analytic derivatives of `sum_log_f` with respect to
    /// `rho = log lambda` (`q = exp(-rho)`). Endpoint pairs linearly
    /// interpolate the third order under a global `L5` bound, certifying the
    /// λ→∞ tail at fourth-order width `(|V′|/L₅)^{1/4}` (#2300/#2614).
    sum_log_f_d1: f64,
    sum_log_f_d2: f64,
    sum_log_f_d3: f64,
    /// Σ over proper steps of `v_t² / F̃_t`.
    sum_v2_over_f: f64,
    /// First three analytic `rho` derivatives of `sum_v2_over_f`.
    sum_v2_over_f_d1: f64,
    sum_v2_over_f_d2: f64,
    sum_v2_over_f_d3: f64,
    /// Number of proper (non-diffuse) innovations.
    n_proper: usize,
}

/// The scalar criterion accumulators with directed-rounding enclosures.
#[derive(Debug)]
struct BallFilterPass {
    sum_log_f: Ball,
    sum_log_f_d1: Ball,
    sum_log_f_d2: Ball,
    sum_log_f_d3: Ball,
    sum_v2_over_f: Ball,
    sum_v2_over_f_d1: Ball,
    sum_v2_over_f_d2: Ball,
    sum_v2_over_f_d3: Ball,
    n_proper: usize,
}

/// One forward pass of the exact diffuse filter.
///
/// `RECORD_STEPS` selects whether the per-node filtered/predicted states are
/// retained. They are needed ONLY by the RTS backward smoother in
/// [`fit_spline_scan_at`]; the profiled REML criterion
/// ([`concentrated_criterion_jet`]) reads nothing but the scalar accumulators.
/// Recording them unconditionally made every criterion evaluation of the
/// certified log-lambda search allocate, fill, and immediately drop
/// `n * size_of::<FilterStep>()` bytes — at the biobank scale this fast path
/// exists for (n = 1e6, 192 B per node) that is 192 MB of write traffic per
/// evaluation, thrown away.
fn run_filter<const RECORD_STEPS: bool>(
    nodes: &[PooledNode],
    q: f64,
    order: usize,
) -> Result<FilterPass, String> {
    let n = nodes.len();
    let mut steps = Vec::with_capacity(if RECORD_STEPS { n } else { 0 });
    // Exact diffuse initialization (Durbin–Koopman): P = P* + κ·P_∞, κ → ∞.
    // The order-`m` polynomial null space (degree < m) is fully diffuse: the
    // diffuse rank starts at `order`, consumed by the first `order` distinct
    // abscissae.
    let mut a: Vec2 = [0.0; MAX_ORDER];
    let mut a_d1: Vec2 = [0.0; MAX_ORDER];
    let mut a_d2: Vec2 = [0.0; MAX_ORDER];
    let mut a_d3: Vec2 = [0.0; MAX_ORDER];
    let mut p_star: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut p_star_d1: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut p_star_d2: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut p_star_d3: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut p_inf: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        p_inf[i][i] = 1.0;
    }
    let mut diffuse_rank = order;
    let mut sum_log_f = 0.0;
    let mut sum_log_f_d1 = 0.0;
    let mut sum_log_f_d2 = 0.0;
    let mut sum_log_f_d3 = 0.0;
    let mut sum_v2_over_f = 0.0;
    let mut sum_v2_over_f_d1 = 0.0;
    let mut sum_v2_over_f_d2 = 0.0;
    let mut sum_v2_over_f_d3 = 0.0;
    let mut n_proper = 0usize;
    for t in 0..n {
        let a_pred = a;
        let p_pred = p_star;
        let r = 1.0 / nodes[t].w;
        let v = nodes[t].y - a[0];
        let v_d1 = -a_d1[0];
        let v_d2 = -a_d2[0];
        let v_d3 = -a_d3[0];
        // H = [1 0 … 0] ⇒ M = P·H' is the first column, F = M[0] (+ r).
        let mut m_star: Vec2 = [0.0; MAX_ORDER];
        let mut m_star_d1: Vec2 = [0.0; MAX_ORDER];
        let mut m_star_d2: Vec2 = [0.0; MAX_ORDER];
        let mut m_star_d3: Vec2 = [0.0; MAX_ORDER];
        for i in 0..order {
            m_star[i] = p_star[i][0];
            m_star_d1[i] = p_star_d1[i][0];
            m_star_d2[i] = p_star_d2[i][0];
            m_star_d3[i] = p_star_d3[i][0];
        }
        let f_star = m_star[0] + r;
        let f_star_d1 = m_star_d1[0];
        let f_star_d2 = m_star_d2[0];
        let f_star_d3 = m_star_d3[0];
        let mut proper_update = diffuse_rank == 0;
        if diffuse_rank > 0 {
            let mut m_inf: Vec2 = [0.0; MAX_ORDER];
            for i in 0..order {
                m_inf[i] = p_inf[i][0];
            }
            let f_inf = m_inf[0];
            if !f_inf.is_finite() {
                return Err(format!(
                    "spline scan: non-finite diffuse innovation variance at node {t}: {f_inf}"
                ));
            } else if f_inf > 0.0 {
                // Exact diffuse update (Koopman 1997): the κ→∞ limit of the
                // standard update; the diffuse step contributes −½·log F_∞ to
                // the restricted likelihood and consumes one diffuse dimension.
                for i in 0..order {
                    let k_inf = m_inf[i] / f_inf;
                    a[i] += k_inf * v;
                    a_d1[i] += k_inf * v_d1;
                    a_d2[i] += k_inf * v_d2;
                    a_d3[i] += k_inf * v_d3;
                }
                let mut p_new = p_star;
                let mut p_new_d1 = p_star_d1;
                let mut p_new_d2 = p_star_d2;
                let mut p_new_d3 = p_star_d3;
                for i in 0..order {
                    for j in 0..order {
                        p_new[i][j] += -m_inf[i] * m_star[j] / f_inf - m_star[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star / (f_inf * f_inf);
                        p_new_d1[i][j] += -m_inf[i] * m_star_d1[j] / f_inf
                            - m_star_d1[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star_d1 / (f_inf * f_inf);
                        p_new_d2[i][j] += -m_inf[i] * m_star_d2[j] / f_inf
                            - m_star_d2[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star_d2 / (f_inf * f_inf);
                        p_new_d3[i][j] += -m_inf[i] * m_star_d3[j] / f_inf
                            - m_star_d3[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star_d3 / (f_inf * f_inf);
                    }
                }
                p_star = p_new;
                p_star_d1 = p_new_d1;
                p_star_d2 = p_new_d2;
                p_star_d3 = p_new_d3;
                symmetrize(&mut p_star, order);
                symmetrize(&mut p_star_d1, order);
                symmetrize(&mut p_star_d2, order);
                symmetrize(&mut p_star_d3, order);
                for i in 0..order {
                    for j in 0..order {
                        p_inf[i][j] -= m_inf[i] * m_inf[j] / f_inf;
                    }
                }
                symmetrize(&mut p_inf, order);
                diffuse_rank -= 1;
                if diffuse_rank == 0 {
                    p_inf = [[0.0; MAX_ORDER]; MAX_ORDER];
                }
            } else if f_inf == 0.0 {
                // Diffuse direction orthogonal to H: this observation is an
                // ordinary proper update of P* even though diffuse rank remains.
                proper_update = true;
            } else {
                return Err(format!(
                    "spline scan: non-positive diffuse innovation variance at node {t}: {f_inf}"
                ));
            }
        }
        if proper_update {
            if !(f_star.is_finite() && f_star > 0.0) {
                return Err(format!(
                    "spline scan: non-positive or non-finite proper innovation variance \
                     at node {t}: {f_star}"
                ));
            }
            let inv_f = 1.0 / f_star;
            // Quotient jets in the recursive Leibniz form: for s = num/f,
            //   s_k = (num_k − Σ_{j=1..k} C(k,j)·s_{k−j}·f_j) / f,
            // which is exactly the closed inv_f² / inv_f³ expansion used
            // before, extended to third order.
            let mut gain = [0.0; MAX_ORDER];
            let mut gain_d1 = [0.0; MAX_ORDER];
            let mut gain_d2 = [0.0; MAX_ORDER];
            let mut gain_d3 = [0.0; MAX_ORDER];
            for i in 0..order {
                gain[i] = m_star[i] * inv_f;
                gain_d1[i] = (m_star_d1[i] - gain[i] * f_star_d1) * inv_f;
                gain_d2[i] =
                    (m_star_d2[i] - 2.0 * gain_d1[i] * f_star_d1 - gain[i] * f_star_d2) * inv_f;
                gain_d3[i] = (m_star_d3[i]
                    - 3.0 * gain_d2[i] * f_star_d1
                    - 3.0 * gain_d1[i] * f_star_d2
                    - gain[i] * f_star_d3)
                    * inv_f;
            }
            let a_old_d1 = a_d1;
            let a_old_d2 = a_d2;
            let a_old_d3 = a_d3;
            for i in 0..order {
                a[i] += gain[i] * v;
                a_d1[i] = a_old_d1[i] + gain_d1[i] * v + gain[i] * v_d1;
                a_d2[i] = a_old_d2[i] + gain_d2[i] * v + 2.0 * gain_d1[i] * v_d1 + gain[i] * v_d2;
                a_d3[i] = a_old_d3[i]
                    + gain_d3[i] * v
                    + 3.0 * gain_d2[i] * v_d1
                    + 3.0 * gain_d1[i] * v_d2
                    + gain[i] * v_d3;
            }
            // The VALUE covariance through the Joseph form, the same expression
            // the certified pass runs (#2614).
            //
            // `P − M Mᵀ/F` subtracts near-equal quantities: on the #2300 nodes
            // at `ρ = −24` both terms are `1.84e5` and their difference is `1`,
            // so the result is accurate to `F/R` times the rounding of its own
            // size. `P⁺ = A P Aᵀ + R K Kᵀ` with `A = I − K e₀ᵀ` is the same
            // quantity as a sum of positive contributions at the observed
            // coordinate. This is not only the certified pass's concern: the two
            // passes must run the same expression, or the enclosure would be
            // certifying arithmetic the scalar path does not perform.
            let mut a_operator = [[0.0; MAX_ORDER]; MAX_ORDER];
            for (i, row) in a_operator.iter_mut().enumerate().take(order) {
                row[i] = 1.0;
            }
            a_operator[0][0] = r * inv_f;
            for i in 1..order {
                a_operator[i][0] = -gain[i];
            }
            let a_operator_t = mat_t(&a_operator, order);
            let mut p_new = mat_mul(&mat_mul(&a_operator, &p_star, order), &a_operator_t, order);
            for i in 0..order {
                for j in 0..order {
                    p_new[i][j] += gain[i] * gain[j] * r;
                }
            }
            let mut p_new_d1 = p_star_d1;
            let mut p_new_d2 = p_star_d2;
            let mut p_new_d3 = p_star_d3;
            for i in 0..order {
                for j in 0..order {
                    let mm = m_star[i] * m_star[j];
                    let mm_d1 = m_star_d1[i] * m_star[j] + m_star[i] * m_star_d1[j];
                    let mm_d2 = m_star_d2[i] * m_star[j]
                        + 2.0 * m_star_d1[i] * m_star_d1[j]
                        + m_star[i] * m_star_d2[j];
                    let mm_d3 = m_star_d3[i] * m_star[j]
                        + 3.0 * m_star_d2[i] * m_star_d1[j]
                        + 3.0 * m_star_d1[i] * m_star_d2[j]
                        + m_star[i] * m_star_d3[j];
                    let s0 = mm * inv_f;
                    let s1 = (mm_d1 - s0 * f_star_d1) * inv_f;
                    let s2 = (mm_d2 - 2.0 * s1 * f_star_d1 - s0 * f_star_d2) * inv_f;
                    let s3 = (mm_d3 - 3.0 * s2 * f_star_d1 - 3.0 * s1 * f_star_d2 - s0 * f_star_d3)
                        * inv_f;
                    p_new_d1[i][j] -= s1;
                    p_new_d2[i][j] -= s2;
                    p_new_d3[i][j] -= s3;
                }
            }
            p_star = p_new;
            p_star_d1 = p_new_d1;
            p_star_d2 = p_new_d2;
            p_star_d3 = p_new_d3;
            symmetrize(&mut p_star, order);
            symmetrize(&mut p_star_d1, order);
            symmetrize(&mut p_star_d2, order);
            symmetrize(&mut p_star_d3, order);

            let vv = v * v;
            let vv_d1 = 2.0 * v * v_d1;
            let vv_d2 = 2.0 * (v_d1 * v_d1 + v * v_d2);
            let vv_d3 = 2.0 * (v * v_d3 + 3.0 * v_d1 * v_d2);
            let logf_d1 = f_star_d1 * inv_f;
            let logf_d2 = f_star_d2 * inv_f - logf_d1 * logf_d1;
            let logf_d3 = f_star_d3 * inv_f - 3.0 * (f_star_d2 * inv_f) * logf_d1
                + 2.0 * logf_d1 * logf_d1 * logf_d1;
            sum_log_f += f_star.ln();
            sum_log_f_d1 += logf_d1;
            sum_log_f_d2 += logf_d2;
            sum_log_f_d3 += logf_d3;
            let t0 = vv * inv_f;
            let t1 = (vv_d1 - t0 * f_star_d1) * inv_f;
            let t2 = (vv_d2 - 2.0 * t1 * f_star_d1 - t0 * f_star_d2) * inv_f;
            let t3 = (vv_d3 - 3.0 * t2 * f_star_d1 - 3.0 * t1 * f_star_d2 - t0 * f_star_d3) * inv_f;
            sum_v2_over_f += t0;
            sum_v2_over_f_d1 += t1;
            sum_v2_over_f_d2 += t2;
            sum_v2_over_f_d3 += t3;
            n_proper += 1;
        }
        if RECORD_STEPS {
            steps.push(FilterStep {
                a_filt: a,
                p_filt: p_star,
                a_pred,
                p_pred,
            });
        }
        // Predict to the next node.
        if t + 1 < n {
            let delta = nodes[t + 1].x - nodes[t].x;
            let f_t = transition(delta, order);
            a = mat_vec(&f_t, &a, order);
            a_d1 = mat_vec(&f_t, &a_d1, order);
            a_d2 = mat_vec(&f_t, &a_d2, order);
            a_d3 = mat_vec(&f_t, &a_d3, order);
            let f_t_t = mat_t(&f_t, order);
            let q_noise = process_noise(delta, q, order);
            let mut p_next = mat_add(
                &mat_mul(&mat_mul(&f_t, &p_star, order), &f_t_t, order),
                &q_noise,
                order,
            );
            let mut p_next_d1 = mat_sub(
                &mat_mul(&mat_mul(&f_t, &p_star_d1, order), &f_t_t, order),
                &q_noise,
                order,
            );
            let mut p_next_d2 = mat_add(
                &mat_mul(&mat_mul(&f_t, &p_star_d2, order), &f_t_t, order),
                &q_noise,
                order,
            );
            // d^k q / d rho^k = (−1)^k q, so the noise term alternates sign.
            let mut p_next_d3 = mat_sub(
                &mat_mul(&mat_mul(&f_t, &p_star_d3, order), &f_t_t, order),
                &q_noise,
                order,
            );
            symmetrize(&mut p_next, order);
            symmetrize(&mut p_next_d1, order);
            symmetrize(&mut p_next_d2, order);
            symmetrize(&mut p_next_d3, order);
            p_star = p_next;
            p_star_d1 = p_next_d1;
            p_star_d2 = p_next_d2;
            p_star_d3 = p_next_d3;
            if diffuse_rank > 0 {
                let mut pi_next =
                    mat_mul(&mat_mul(&f_t, &p_inf, order), &mat_t(&f_t, order), order);
                symmetrize(&mut pi_next, order);
                p_inf = pi_next;
            }
        }
    }
    Ok(FilterPass {
        steps,
        sum_log_f,
        sum_log_f_d1,
        sum_log_f_d2,
        sum_log_f_d3,
        sum_v2_over_f,
        sum_v2_over_f_d1,
        sum_v2_over_f_d2,
        sum_v2_over_f_d3,
        n_proper,
    })
}

/// Directed-rounding twin of [`run_filter`] used by automatic REML selection.
///
/// The recurrence follows the production diffuse filter operation for
/// operation, but every scalar carries an outward interval. Its cost is
/// `O(n·order³)` (the same fixed-size covariance propagations as the scalar
/// pass), with no data-dependent refinement knobs. A denominator is used only
/// after its innovation ball is proved strictly positive; if the ball contains
/// zero, the typed proof refusal is returned at that exact node.
fn run_filter_ball(
    nodes: &[PooledNode],
    q: Ball,
    order: usize,
) -> Result<BallFilterPass, SplineScoreProofError> {
    run_filter_ball_traced(nodes, q, order, None)
}

/// One `(node, quantity, ball)` record of the `d3` recursion, for the caller
/// that asked to see it.
///
/// A refusal names the accumulator that left the finite range; it cannot show
/// how the width GOT there, and that is the difference between "the recursion
/// grows" and "the interval evaluation cannot see a cancellation the exact
/// arithmetic has". Only a caller that passes a sink pays for this.
type BallTraceRecord = (usize, &'static str, Ball);

/// Stacked mean blocks carried as ONE zonotope: `(a, a′)`.
const MEAN_BLOCKS: usize = 2;
/// Capacity of that stacked state; the ACTIVE dimension is `2·order`.
const MEAN_DIM: usize = MEAN_BLOCKS * MAX_ORDER;
/// Capacity of the `vec(dP/dρ)` state; the ACTIVE dimension is `order²`.
const COVARIANCE_D1_DIM: usize = MAX_ORDER * MAX_ORDER;
/// Generators retained before the lowest-correlation directions are folded
/// into an axis-aligned set.
///
/// Folding is a sound outer reduction: the axis box contains the discarded
/// generators and is itself a zonotope, so it keeps being transformed by the
/// true map rather than by its absolute value. The cap only bounds the work:
/// `dim²·CAP` per node, `O(n)` overall. Reduction chooses the generators for
/// which a box loses the least signed correlation; it never assumes that age
/// alone implies contraction.
const ZONOTOPE_GENERATOR_CAP: usize = 240;
/// `γ_{dim+2}` with room to spare: `2·(d+2)·u` with `u = ε/2` is `11ε` at
/// `d = 9`, and this charges `32ε` for every floating-point dot product a
/// zonotope forms.
const ZONOTOPE_ROUNDOFF: f64 = 32.0 * f64::EPSILON;

/// Radius of a ball ABOUT ITS REPRESENTATIVE, which is what a zonotope centred
/// on that representative must absorb. Not `(hi−lo)/2`: the representative is
/// not required to be the midpoint.
#[inline]
fn ball_radius_about_value(ball: Ball) -> f64 {
    let above = ball.hi - ball.value;
    let below = ball.value - ball.lo;
    next_up_ball(above.max(below).max(0.0))
}

/// A linear recursion's state as a ZONOTOPE — a centre plus a list of error
/// GENERATORS — rather than as componentwise intervals.
///
/// This is not a tightening heuristic. Two of this filter's recursions cannot
/// be carried in a box AT ANY WIDTH, and the reason is arithmetic rather than
/// numerical. Per node the mean does update-then-predict, `a ← T(A a + K y)`
/// with `A = I − K e₀ᵀ`, so the fused per-node map at order 2 is
///
/// ```text
///   B = T A = [[R/F − δ·K₁ ,  δ ],
///              [   −K₁     ,  1 ]]
/// ```
///
/// Measured on the #2300 nodes at the ρ the certified search refused at
/// (`−13.841116908`): `R/F = 0.0754`, `K₁ ≈ 41.5`, `δ = 4/179`, so `δ·K₁ = 0.930`
/// and
///
/// ```text
///   true B : trace 0.145 , det 0.0754  ⇒ |eigenvalues| = √0.0754 = 0.2746
///   |B|    : trace 1.855 , det −0.0754 ⇒ ρ(|B|)        = 1.894
/// ```
///
/// **The map contracts at 0.27 per node and its entrywise absolute value
/// expands at 1.894.** A componentwise interval carries widths through `|B|`,
/// because `a₀` reaches the next `a₀` by two paths — the row-0 update, and row 1
/// followed by the transition — whose widths ADD as `0.0754 + 0.930` where the
/// exact map SUBTRACTS to `−0.855`. Over 180 nodes that is `1.894¹⁸⁰ ≈ 10⁵⁰`,
/// and the traced widths reproduced it exactly: `w(a₀)` ran `1e−11 → 1e20` from
/// node 10 to node 113 at a clean factor of 2.0 per node, while every
/// covariance quantity in the same pass stayed flat (`w(P₀₀) = 5.6e−13`,
/// `w(F) = 1e−10`, `w(Σ log F) = 7.4e−10`).
///
/// The covariance's FIRST DERIVATIVE has the same defect one tensor rank up.
/// `dP⁺ = A·dP·Aᵀ` and `dP⁻ = F·dP·Fᵀ − Q` make `vec(dP)` a linear recursion
/// with map `B ⊗ B`, whose true spectral radius is `0.27² = 0.075` and whose
/// componentwise companion `|B| ⊗ |B|` is `1.894² = 3.59`. Measured at
/// `ρ = −12.5466`, order 2: `w(dP₁₁)` runs `1.4e−2 → 7.5e2` over nodes 40..80
/// while `w(P₀₀)` holds at `7e−12`.
///
/// Two rearrangements do NOT help and are recorded so they are not retried:
/// fusing `T` and `A` first leaves `ρ(|TA|)` at the same 1.894 — the
/// cancellation is between entries of the product, not between the factors —
/// and rescaling the state cannot help either, since `ρ(|D⁻¹BD|) ≥ ρ(|B|)` for
/// every diagonal `D` and the natural step scaling `D = diag(1, δ)` attains it
/// exactly.
///
/// A zonotope keeps the cancellation because it transforms each GENERATOR by
/// the true map, so generator norms follow `0.27` per node and a generator is
/// below `ε` relative after ~35 nodes. The value covariance escaped the same
/// problem by being carried as a Cholesky factor; neither the mean nor `dP`
/// has PSD structure enough to exploit that way, but both are exactly LINEAR
/// with coefficients this filter already encloses tightly, which is the
/// hypothesis a zonotope needs and the Riccati update itself does not satisfy.
///
/// `N` is the capacity; `dim` is how much of it the current smoothing order
/// uses, and every loop stops there, so an order-1 scan pays `1`-dimensional
/// work out of a `9`-wide array.
#[derive(Clone, Debug)]
struct Zonotope<const N: usize> {
    center: [f64; N],
    /// Coefficient of the one normalized uncertainty variable shared by every
    /// occurrence of the certified process-noise scale `q`.
    ///
    /// This generator is structural, not part of the disposable remainder
    /// basis: compaction may fold independent roundoff generators, but it must
    /// never turn one common scalar into independent per-node errors.
    shared_q: [f64; N],
    generators: Vec<[f64; N]>,
    dim: usize,
}

impl<const N: usize> Zonotope<N> {
    fn zeroed(dim: usize) -> Self {
        // `N` is the array capacity and `dim` indexes into it, so a `dim > N`
        // is an out-of-bounds every loop in this file would then commit. A
        // `debug_assert!` compiles to nothing in the release profile the
        // scan actually ships in, which is precisely where the bound stops
        // being checked by anything else -- hence the workspace ban.
        assert!(dim <= N, "zonotope dim {dim} exceeds its capacity {N}");
        Self {
            center: [0.0; N],
            shared_q: [0.0; N],
            generators: Vec::new(),
            dim,
        }
    }

    /// One coordinate as an ordinary ball, for the consumers that need a scalar.
    fn coordinate(&self, index: usize) -> Ball {
        let mut radius = self.shared_q[index].abs();
        for generator in &self.generators {
            radius = next_up_ball(radius + generator[index].abs());
        }
        let value = self.center[index];
        Ball {
            value,
            lo: next_down_ball(value - radius),
            hi: next_up_ball(value + radius),
        }
    }

    /// `x ← M x + b`, exactly on the generators, with every floating-point and
    /// interval radius charged into FRESH axis-aligned generators.
    ///
    /// The fresh radii are appended as `radius·eᵢ` rather than held in a
    /// separate box field, because a box propagated as `|M|·box` would grow at
    /// `ρ(|M|)` per node — reintroducing the exact defect this type exists to
    /// remove, on a quantity too small to notice until it is `1e11`.
    fn apply(&mut self, map: &[[Ball; N]; N], constant: &[Ball; N]) -> bool {
        self.apply_with_shared_q(map, constant, &[0.0; N])
    }

    /// `x ← Mx + b + g_q θ_q`, where the SAME `θ_q ∈ [-1, 1]` is carried by
    /// every process-noise injection in the complete filter pass.
    ///
    /// The interval part of `b` contains only nonlinear and floating-point
    /// remainder. It therefore enters as fresh independent generators, while
    /// `g_q` is accumulated onto the distinguished generator after that
    /// generator has followed the signed map `M`.
    fn apply_with_shared_q(
        &mut self,
        map: &[[Ball; N]; N],
        constant: &[Ball; N],
        shared_q_constant: &[f64; N],
    ) -> bool {
        let dim = self.dim;
        let mut generator_column_sum = [0.0f64; N];
        for (sum, &coordinate) in generator_column_sum
            .iter_mut()
            .zip(self.shared_q.iter())
            .take(dim)
        {
            *sum = coordinate.abs();
        }
        for generator in &self.generators {
            for j in 0..dim {
                generator_column_sum[j] =
                    next_up_ball(generator_column_sum[j] + generator[j].abs());
            }
        }

        let mut next_center = [0.0f64; N];
        let mut next_shared_q = [0.0f64; N];
        let mut fresh_radius = [0.0f64; N];
        for i in 0..dim {
            let mut center = constant[i].value;
            let mut shared_q = shared_q_constant[i];
            // Everything the roundoff of the dot products — the centre's and
            // every generator's — is charged against, summed once.
            let mut magnitude = constant[i].value.abs() + shared_q_constant[i].abs();
            let mut radius = ball_radius_about_value(constant[i]);
            for j in 0..dim {
                let coefficient = map[i][j].value;
                center += coefficient * self.center[j];
                shared_q += coefficient * self.shared_q[j];
                magnitude = next_up_ball(
                    magnitude
                        + (coefficient * self.center[j]).abs()
                        + coefficient.abs() * generator_column_sum[j],
                );
                radius = next_up_ball(
                    radius
                        + ball_radius_about_value(map[i][j])
                            * (self.center[j].abs() + generator_column_sum[j]),
                );
            }
            next_center[i] = center;
            next_shared_q[i] = shared_q;
            fresh_radius[i] = next_up_ball(
                (radius + ZONOTOPE_ROUNDOFF * magnitude) * (1.0 + 64.0 * f64::EPSILON),
            );
        }

        for generator in self.generators.iter_mut() {
            let previous = *generator;
            for i in 0..dim {
                let mut coordinate = 0.0f64;
                for j in 0..dim {
                    coordinate += map[i][j].value * previous[j];
                }
                generator[i] = coordinate;
            }
        }

        self.center = next_center;
        self.shared_q = next_shared_q;
        for i in 0..dim {
            if fresh_radius[i] > 0.0 {
                let mut axis = [0.0f64; N];
                axis[i] = fresh_radius[i];
                self.generators.push(axis);
            }
        }
        self.compact();
        self.center[..dim].iter().all(|value| value.is_finite())
            && self.shared_q[..dim].iter().all(|value| value.is_finite())
            && self
                .generators
                .iter()
                .all(|generator| generator[..dim].iter().all(|value| value.is_finite()))
    }

    /// Reduce the lowest-correlation generators to one axis-aligned set once
    /// the list is over the cap.
    ///
    /// Age is not a sound proxy for dispensability here.  The order-3 closed
    /// loop contracts through a rotation of its dominant directions; after ten
    /// nodes an old generator can still be large and strongly non-axis-aligned.
    /// Folding it merely because it is old discards precisely that correlation
    /// and sends its width through `|M|`, recreating the wrapping effect this
    /// zonotope exists to avoid.
    ///
    /// The reduction score `||g||₁ - ||g||∞` is zero for an axis generator and
    /// grows with the correlation that an axis box would discard.  Therefore
    /// the lowest-scoring generators are the loss-minimizing ones to fold, and
    /// the correlation-bearing generators remain explicit.  The folded box is
    /// still an outer zonotope: each coordinate radius is the outward-rounded
    /// sum of the folded generators' absolute coordinates.
    fn compact(&mut self) {
        if self.generators.len() <= ZONOTOPE_GENERATOR_CAP {
            return;
        }
        let dim = self.dim;
        let reduction_score = |generator: &[f64; N]| {
            let mut l1 = 0.0_f64;
            let mut linf = 0.0_f64;
            for &coordinate in generator.iter().take(dim) {
                let magnitude = coordinate.abs();
                l1 += magnitude;
                linf = linf.max(magnitude);
            }
            (l1 - linf).max(0.0)
        };
        self.generators
            .sort_by(|left, right| reduction_score(left).total_cmp(&reduction_score(right)));
        let fold = self.generators.len() - ZONOTOPE_GENERATOR_CAP / 2;
        let retained = self.generators.split_off(fold);
        let mut folded = [0.0f64; N];
        for generator in &self.generators {
            for i in 0..dim {
                folded[i] = next_up_ball(folded[i] + generator[i].abs());
            }
        }
        let mut next = Vec::with_capacity(retained.len() + dim);
        for i in 0..dim {
            if folded[i] > 0.0 {
                let mut axis = [0.0f64; N];
                axis[i] = folded[i];
                next.push(axis);
            }
        }
        next.extend(retained);
        self.generators = next;
    }
}

/// The identity map over the first `dim` coordinates.
fn zonotope_identity_map<const N: usize>(dim: usize) -> [[Ball; N]; N] {
    let mut map = [[Ball::ZERO; N]; N];
    for (i, row) in map.iter_mut().enumerate().take(dim) {
        row[i] = Ball::ONE;
    }
    map
}

/// Write an `order × order` block of the stacked MEAN map at block row/column
/// `(block_row, block_column)`. The mean layout is `block·order + i`.
fn mean_set_block(
    map: &mut [[Ball; MEAN_DIM]; MEAN_DIM],
    block_row: usize,
    block_column: usize,
    block: &BallMat,
    order: usize,
) {
    for i in 0..order {
        for j in 0..order {
            map[block_row * order + i][block_column * order + j] = block[i][j];
        }
    }
}

/// The congruence `X ↦ L·X·Rᵀ` as a linear map on `vec(X)`, i.e. `L ⊗ R`.
///
/// This is what makes `dP` carryable: the derivative's update and prediction
/// are congruences by matrices built from the VALUE covariance, which this
/// filter already encloses to `1e−12`, so the map's own entries are tight and
/// only the state needs the generators. The `vec` layout is `i·order + j`.
fn zonotope_congruence_map(
    left: &BallMat,
    right: &BallMat,
    order: usize,
) -> [[Ball; COVARIANCE_D1_DIM]; COVARIANCE_D1_DIM] {
    let mut map = [[Ball::ZERO; COVARIANCE_D1_DIM]; COVARIANCE_D1_DIM];
    for i in 0..order {
        for j in 0..order {
            for k in 0..order {
                for l in 0..order {
                    map[i * order + j][k * order + l] = left[i][k].mul(right[j][l]);
                }
            }
        }
    }
    map
}

/// Project a matrix-valued zonotope onto the exact symmetric subspace.
///
/// Covariances and every one of their parameter derivatives are symmetric
/// exact-real matrices. If `x` is the witness point in the incoming zonotope,
/// then `Sx = x` for the symmetrizer `S`; applying `S` to every generator
/// therefore preserves that witness while deleting enclosure directions that
/// violate a model identity.
fn project_symmetric_zonotope(state: &mut Zonotope<COVARIANCE_D1_DIM>, order: usize) -> bool {
    if order == 1 {
        return true;
    }
    let mut projection = [[Ball::ZERO; COVARIANCE_D1_DIM]; COVARIANCE_D1_DIM];
    for i in 0..order {
        for j in 0..order {
            let row = i * order + j;
            if i == j {
                projection[row][row] = Ball::ONE;
            } else {
                projection[row][i * order + j] = Ball::exact(0.5);
                projection[row][j * order + i] = Ball::exact(0.5);
            }
        }
    }
    state.apply(&projection, &[Ball::ZERO; COVARIANCE_D1_DIM])
}

/// Read `vec(X)` back out as a matrix.
fn zonotope_to_matrix(state: &Zonotope<COVARIANCE_D1_DIM>, order: usize) -> BallMat {
    let mut out = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        for j in 0..order {
            out[i][j] = state.coordinate(i * order + j);
        }
    }
    out
}

/// Apply one scalar-observation Riccati update to a covariance zonotope.
///
/// Applying the Joseph map with an interval-valued gain is sound but useless:
/// the gain is a function of the SAME covariance, while a generic interval
/// matrix application ranges the two independently.  Its resulting first-order
/// error is fed back into the next gain and recreates componentwise Riccati
/// wrapping.
///
/// Instead linearize the Riccati map at the zonotope centre and enclose its
/// exact, signed second-order remainder.  With centre column `c`, centre
/// innovation `f`, `k = c/f`, covariance perturbation `E`, observation-variance
/// perturbation `dr`, `d = E₀₀ + dr`, and
///
/// ```text
/// u = E e₀ - k d,
/// ```
///
/// direct expansion gives the IDENTITY
///
/// ```text
/// U(C + E, r + dr)
///   = A (C + E) Aᵀ + (r + dr) k kᵀ - u uᵀ / (f + d),
///   A = I - k e₀ᵀ.
/// ```
///
/// Thus every existing generator follows the contracting derivative map
/// `E ↦ AEAᵀ`; only the genuinely quadratic remainder becomes fresh interval
/// error.  This preserves the covariance/gain dependency rather than treating
/// it as uncertainty.
fn covariance_zonotope_measurement_update(
    state: &mut Zonotope<COVARIANCE_D1_DIM>,
    r: Ball,
    order: usize,
) -> bool {
    let centre_r = Ball::exact(r.value);
    let centre_f = Ball::exact(state.center[0]).add(centre_r);
    if !(centre_f.is_finite() && centre_f.lo > 0.0) {
        return false;
    }

    let mut centre_gain = [Ball::ZERO; MAX_ORDER];
    let mut column_error = [Ball::ZERO; MAX_ORDER];
    for i in 0..order {
        centre_gain[i] = Ball::exact(state.center[i * order]).div_positive(centre_f);
        let coordinate = state.coordinate(i * order);
        let radius = ball_radius_about_value(coordinate);
        column_error[i] = Ball {
            value: 0.0,
            lo: -radius,
            hi: radius,
        };
    }
    let r_error = Ball {
        value: 0.0,
        lo: next_down_ball(r.lo - r.value),
        hi: next_up_ball(r.hi - r.value),
    };
    let denominator_error = column_error[0].add(r_error);
    let denominator = centre_f.add(denominator_error);
    if !(denominator.is_finite() && denominator.lo > 0.0) {
        return false;
    }

    let mut remainder_vector = [Ball::ZERO; MAX_ORDER];
    for i in 0..order {
        remainder_vector[i] = column_error[i].sub(centre_gain[i].mul(denominator_error));
    }
    let operator = ball_update_operator(&centre_gain, centre_r.div_positive(centre_f), order);
    let mut constant = [Ball::ZERO; COVARIANCE_D1_DIM];
    for i in 0..order {
        for j in 0..order {
            let quadratic_remainder = remainder_vector[i]
                .mul(remainder_vector[j])
                .div_positive(denominator)
                .neg();
            constant[i * order + j] = r
                .mul(centre_gain[i])
                .mul(centre_gain[j])
                .add(quadratic_remainder);
        }
    }
    state.apply(
        &zonotope_congruence_map(&operator, &operator, order),
        &constant,
    )
}

/// Diagonal names for the traced covariance jets, indexed by state coordinate.
const D3_DIAGONAL_NAMES: [&str; MAX_ORDER] = ["d3_upd_00", "d3_upd_11", "d3_upd_22"];
const D2_DIAGONAL_NAMES: [&str; MAX_ORDER] = ["d2_upd_00", "d2_upd_11", "d2_upd_22"];
const D1_DIAGONAL_NAMES: [&str; MAX_ORDER] = ["d1_upd_00", "d1_upd_11", "d1_upd_22"];
const P_DIAGONAL_NAMES: [&str; MAX_ORDER] = ["p_upd_00", "p_upd_11", "p_upd_22"];
/// Kalman gain coordinates, so a caller can rebuild the closed-loop map.
const GAIN_NAMES: [&str; MAX_ORDER] = ["gain_0", "gain_1", "gain_2"];
/// Full PREDICTED covariance, so a caller can weigh that map by the matrix the
/// Riccati recursion makes a Lyapunov matrix for it.
const P_NEXT_ENTRY_NAMES: [[&str; MAX_ORDER]; MAX_ORDER] = [
    ["p_next_00", "p_next_01", "p_next_02"],
    ["p_next_10", "p_next_11", "p_next_12"],
    ["p_next_20", "p_next_21", "p_next_22"],
];

fn run_filter_ball_traced(
    nodes: &[PooledNode],
    q: Ball,
    order: usize,
    mut trace: Option<&mut Vec<BallTraceRecord>>,
) -> Result<BallFilterPass, SplineScoreProofError> {
    // `(a, a′)` together, `vec(P)` once the covariance becomes proper, and
    // `vec(dP/dρ)`, each as a zonotope; see `Zonotope` for why a componentwise
    // enclosure of these contracting signed recursions is impossible at any
    // width.
    let mut mean = Zonotope::<MEAN_DIM>::zeroed(MEAN_BLOCKS * order);
    // Start at the exact zero proper covariance and carry it even while the
    // diffuse rank is being consumed. Seeding only after the diffuse phase
    // would already have boxed the first `order - 1` occurrences of the common
    // process-noise scale, so their correlation could never be recovered.
    let mut covariance = Zonotope::<COVARIANCE_D1_DIM>::zeroed(order * order);
    let mut covariance_d1 = Zonotope::<COVARIANCE_D1_DIM>::zeroed(order * order);
    let mut a_d2: BallVec = [Ball::ZERO; MAX_ORDER];
    let mut a_d3: BallVec = [Ball::ZERO; MAX_ORDER];
    let mut p_star: BallMat = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    let mut p_star_d2: BallMat = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    let mut p_star_d3: BallMat = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    let mut p_inf: BallMat = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        p_inf[i][i] = Ball::ONE;
    }
    let mut diffuse_rank = order;
    // A CARRIED Cholesky factor of the proper covariance, once the diffuse rank
    // is consumed and there is a proper covariance to factor. The componentwise
    // recursion still runs; this is a second, independent enclosure of the same
    // matrix whose update and prediction contain no cancelling subtraction, and
    // the two are intersected. `None` means no evidence, never a refusal.
    let mut carried_factor: Option<BallMat> = None;
    let mut sum_log_f = Ball::ZERO;
    let mut sum_log_f_d1 = Ball::ZERO;
    let mut sum_log_f_d2 = Ball::ZERO;
    let mut sum_log_f_d3 = Ball::ZERO;
    let mut sum_v2_over_f = Ball::ZERO;
    let mut sum_v2_over_f_d1 = Ball::ZERO;
    let mut sum_v2_over_f_d2 = Ball::ZERO;
    let mut sum_v2_over_f_d3 = Ball::ZERO;
    let mut n_proper = 0usize;
    // `Σ w y²` over the prefix — the exact ceiling on the accumulated
    // innovations quadratic (see
    // [`intersect_first_order_accumulator_exact_ranges`]). Accumulated at EVERY
    // node, including the ones that consume diffuse rank, because the quadratic
    // it bounds is the restricted form for the whole prefix and not only for the
    // nodes that contributed a proper innovation.
    let mut weighted_energy = Ball::ZERO;

    for t in 0..nodes.len() {
        let r = Ball::ONE.div_positive(Ball::exact(nodes[t].w));
        weighted_energy = weighted_energy.add(
            Ball::exact(nodes[t].y)
                .square()
                .mul(Ball::exact(nodes[t].w)),
        );
        let v = Ball::exact(nodes[t].y).sub(mean.coordinate(0));
        let v_d1 = mean.coordinate(order).neg();
        let v_d2 = a_d2[0].neg();
        let v_d3 = a_d3[0].neg();
        // `dP/dρ` materialized from its zonotope for this node's consumers.
        let p_star_d1 = zonotope_to_matrix(&covariance_d1, order);
        let mut m_star: BallVec = [Ball::ZERO; MAX_ORDER];
        let mut m_star_d1: BallVec = [Ball::ZERO; MAX_ORDER];
        let mut m_star_d2: BallVec = [Ball::ZERO; MAX_ORDER];
        let mut m_star_d3: BallVec = [Ball::ZERO; MAX_ORDER];
        for i in 0..order {
            m_star[i] = p_star[i][0];
            m_star_d1[i] = p_star_d1[i][0];
            m_star_d2[i] = p_star_d2[i][0];
            m_star_d3[i] = p_star_d3[i][0];
        }
        let mut f_star = m_star[0].add(r);
        intersect_innovation_above_observation_variance(&mut f_star, r);
        let f_star_d1 = m_star_d1[0];
        let f_star_d2 = m_star_d2[0];
        let f_star_d3 = m_star_d3[0];

        let mut proper_update = diffuse_rank == 0;
        if diffuse_rank > 0 {
            let mut m_inf: BallVec = [Ball::ZERO; MAX_ORDER];
            for i in 0..order {
                m_inf[i] = p_inf[i][0];
            }
            let f_inf = m_inf[0];
            if f_inf.lo == 0.0 && f_inf.hi == 0.0 {
                // The observation is exactly orthogonal to the remaining
                // diffuse subspace. It receives an ordinary proper update
                // without consuming diffuse rank.
                proper_update = true;
            } else {
                require_positive_innovation(t, SplineInnovationKind::Diffuse, f_inf)?;
            }
            if !proper_update {
                let inv_f_inf = Ball::ONE.div_positive(f_inf);
                let inv_f_inf_sq = inv_f_inf.square();
                let mut gain_inf: BallVec = [Ball::ZERO; MAX_ORDER];
                for i in 0..order {
                    gain_inf[i] = m_inf[i].mul(inv_f_inf);
                    a_d2[i] = a_d2[i].add(gain_inf[i].mul(v_d2));
                    a_d3[i] = a_d3[i].add(gain_inf[i].mul(v_d3));
                }
                // `a⁺ = A_inf·a + K_inf·y` and `a′⁺ = A_inf·a′`, with
                // `A_inf = I − K_inf e₀ᵀ`: the diffuse update is the same
                // operator on both blocks and the jet block carries no `y`,
                // because `v′ = −a′₀` has no data term to pick up.
                // `K_inf[0] = M_inf[0]/F_inf = 1` identically, since `F_inf` IS
                // `M_inf[0]`, so the operator's own diagonal entry is an exact
                // zero rather than a `1 − K₀` subtraction.
                let a_inf = ball_update_operator(&gain_inf, Ball::ZERO, order);
                let mut diffuse_map = zonotope_identity_map::<MEAN_DIM>(MEAN_BLOCKS * order);
                mean_set_block(&mut diffuse_map, 0, 0, &a_inf, order);
                mean_set_block(&mut diffuse_map, 1, 1, &a_inf, order);
                let mut diffuse_constant = [Ball::ZERO; MEAN_DIM];
                let y_node = Ball::exact(nodes[t].y);
                for i in 0..order {
                    diffuse_constant[i] = gain_inf[i].mul(y_node);
                }
                if !mean.apply(&diffuse_map, &diffuse_constant) {
                    return Err(SplineScoreProofError::InvalidArithmetic {
                        context: "diffuse mean zonotope",
                    });
                }
                let mut p_new = p_star;
                let mut p_new_d2 = p_star_d2;
                let mut p_new_d3 = p_star_d3;
                for i in 0..order {
                    for j in 0..order {
                        let inf_product = m_inf[i].mul(m_inf[j]);
                        let subtract_left = m_inf[i].mul(m_star[j]).mul(inv_f_inf);
                        let subtract_right = m_star[i].mul(m_inf[j]).mul(inv_f_inf);
                        let add_star = inf_product.mul(f_star).mul(inv_f_inf_sq);
                        p_new[i][j] = p_new[i][j]
                            .sub(subtract_left)
                            .sub(subtract_right)
                            .add(add_star);

                        let subtract_left_d2 = m_inf[i].mul(m_star_d2[j]).mul(inv_f_inf);
                        let subtract_right_d2 = m_star_d2[i].mul(m_inf[j]).mul(inv_f_inf);
                        p_new_d2[i][j] = p_new_d2[i][j]
                            .sub(subtract_left_d2)
                            .sub(subtract_right_d2)
                            .add(inf_product.mul(f_star_d2).mul(inv_f_inf_sq));

                        let subtract_left_d3 = m_inf[i].mul(m_star_d3[j]).mul(inv_f_inf);
                        let subtract_right_d3 = m_star_d3[i].mul(m_inf[j]).mul(inv_f_inf);
                        p_new_d3[i][j] = p_new_d3[i][j]
                            .sub(subtract_left_d3)
                            .sub(subtract_right_d3)
                            .add(inf_product.mul(f_star_d3).mul(inv_f_inf_sq));
                    }
                }
                // The diffuse covariance update is the fixed-gain Joseph map
                //
                //     P⁺ = A_inf P A_infᵀ + r K_inf K_infᵀ.
                //
                // It is affine in the proper covariance, so the value
                // zonotope can and must follow it from the exact zero state.
                // Waiting until diffuse rank reaches zero would box the first
                // process-noise injections before the shared-q generator even
                // existed.
                let mut covariance_constant = [Ball::ZERO; COVARIANCE_D1_DIM];
                for i in 0..order {
                    for j in 0..order {
                        covariance_constant[i * order + j] = r.mul(gain_inf[i]).mul(gain_inf[j]);
                    }
                }
                if !covariance.apply(
                    &zonotope_congruence_map(&a_inf, &a_inf, order),
                    &covariance_constant,
                ) || !project_symmetric_zonotope(&mut covariance, order)
                {
                    return Err(SplineScoreProofError::InvalidArithmetic {
                        context: "diffuse covariance zonotope",
                    });
                }
                let covariance_p_new = zonotope_to_matrix(&covariance, order);
                for i in 0..order {
                    for j in 0..order {
                        intersect_with_independent_enclosure(
                            &mut p_new[i][j],
                            covariance_p_new[i][j],
                        );
                    }
                }
                // `D1+ = A_inf*D1*A_inf'` EXACTLY: expanding that congruence
                // gives `D1[i][j] - c_j D1[i][0] - c_i D1[0][j] + c_i c_j D1[0][0]`
                // with `c = M_inf/F_inf`, which is the diffuse update term for
                // term. The other jets keep the expanded form because they are
                // not carried as zonotopes.
                if !covariance_d1.apply(
                    &zonotope_congruence_map(&a_inf, &a_inf, order),
                    &[Ball::ZERO; COVARIANCE_D1_DIM],
                ) || !project_symmetric_zonotope(&mut covariance_d1, order)
                {
                    return Err(SplineScoreProofError::InvalidArithmetic {
                        context: "diffuse covariance-derivative zonotope",
                    });
                }
                p_star = p_new;
                p_star_d2 = p_new_d2;
                p_star_d3 = p_new_d3;
                ball_symmetrize(&mut p_star, order);
                ball_symmetrize(&mut p_star_d2, order);
                ball_symmetrize(&mut p_star_d3, order);
                for i in 0..order {
                    for j in 0..order {
                        p_inf[i][j] = p_inf[i][j].sub(m_inf[i].mul(m_inf[j]).mul(inv_f_inf));
                    }
                }
                ball_symmetrize(&mut p_inf, order);
                diffuse_rank -= 1;
                if diffuse_rank == 0 {
                    p_inf = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
                    intersect_proper_covariance_psd(&mut p_star, order)?;
                    carried_factor = ball_cholesky(&p_star, order);
                }
            }
        }

        if proper_update {
            require_positive_innovation(t, SplineInnovationKind::Proper, f_star)?;
            let inv_f = Ball::ONE.div_positive(f_star);
            let mut gain = [Ball::ZERO; MAX_ORDER];
            for i in 0..order {
                gain[i] = m_star[i].mul(inv_f);
            }
            // The VALUE covariance through the same Joseph form as its jets.
            //
            // `P⁺ = P − M Mᵀ/F` is the textbook update and it is a subtraction
            // of near-equal quantities exactly where this filter lives. At
            // `ρ = −24` on the #2300 nodes the predicted `P₀₀` is `1.84e5` and
            // `M₀²/F` is `1.84e5`; their difference is `1.0`, so the enclosure
            // of a quantity of size one is charged the rounding of quantities
            // `1.8e5` times larger, and the ratio is `F/R = P₀₀/R + 1`, which
            // GROWS with the process noise.
            //
            // The Joseph form `P⁺ = A P Aᵀ + R K Kᵀ` (`A = I − K e₀ᵀ`) is the
            // same quantity, and at the observed coordinate it is
            //
            //     (A P Aᵀ)₀₀ + R K₀² = (R/F)²P₀₀ + R P₀₀²/F²
            //                        = P₀₀ R (R + P₀₀)/F² = P₀₀ R/F,
            //
            // a sum of two POSITIVE terms — the exact `P⁺₀₀`, with no
            // subtraction anywhere in it. `A[0][0] = R/F` is itself formed
            // exactly rather than as `1 − K₀` (see `ball_update_operator`).
            //
            // The measurement that forced this (#2614, per-node trace at
            // `ρ = −24`, order 2). `P⁺₀₀` holds the VALUE `0.99999456` at every
            // node while its enclosure WIDTH runs `1.6e-1, 2.3e0, 3.1e1, 3.8e2,
            // 4.7e3, 5.9e4, 9.2e5` over nodes 6..12 — a factor of ~13 per node,
            // on a quantity of size one. `F` inherits it, so `inv_f`'s enclosure
            // width reaches `1.0` at node 13 against a value of `5.4e-6`; at
            // that node `A[0][0]` stops being a contraction and EVERY derivative
            // jet leaves the finite range in a single step (`d3⁺₀₀` width
            // `1.7e-3 → 4.5e19`). The `d3` recursion was never the defect: its
            // own congruence contracts by `(R/F)² ≈ 3e-11` per node exactly as
            // intended, and its VALUES are bit-stable throughout. It diverges
            // because the value covariance it multiplies by had already lost its
            // enclosure.
            let a_operator = ball_update_operator(&gain, r.mul(inv_f), order);
            let mut component_p_new = ball_congruence(&a_operator, &p_star, &a_operator, order);
            for i in 0..order {
                for j in 0..order {
                    component_p_new[i][j] = component_p_new[i][j].add(gain[i].mul(gain[j]).mul(r));
                }
            }
            // Carry the covariance through the Riccati map's exact centred
            // Taylor form. Its linear part is the signed Joseph congruence and
            // its remainder is quadratic in the covariance radius; see
            // `covariance_zonotope_measurement_update`.
            if !covariance_zonotope_measurement_update(&mut covariance, r, order)
                || !project_symmetric_zonotope(&mut covariance, order)
            {
                return Err(SplineScoreProofError::InvalidArithmetic {
                    context: "proper covariance zonotope update",
                });
            }
            let mut p_new = zonotope_to_matrix(&covariance, order);
            for i in 0..order {
                for j in 0..order {
                    intersect_with_independent_enclosure(&mut p_new[i][j], component_p_new[i][j]);
                }
            }
            // ... and then tightened to the EXACT range of the map each entry
            // is, which the componentwise evaluation above cannot see because
            // `A`, `K` and the middle factor all carry the same `P₀₀`.
            for i in 0..order {
                for j in 0..order {
                    if i == j {
                        // On the DIAGONAL the subtracted term is `M[i]²/F`, a
                        // SQUARE. The general corner form ranges `P[i][0]` and
                        // `P[0][j]` independently, so when that entry straddles
                        // zero it not only doubles the width but admits a
                        // NEGATIVE product — an upper bound larger than
                        // `P[i][i]` itself — and its monotonicity argument then
                        // bails out entirely. `Ball::square` returns `[0, max²]`
                        // for a straddling interval, which is the exact range.
                        intersect_with_independent_enclosure(
                            &mut p_new[i][i],
                            p_star[i][i].sub(m_star[i].square().div_positive(f_star)),
                        );
                    }
                    intersect_updated_covariance_exact_range(
                        &mut p_new[i][j],
                        p_star[i][j],
                        m_star[i],
                        m_star[j],
                        m_star[0],
                        r,
                    );
                }
            }
            // The first column of the UPDATED covariance is EXACTLY `R·K`, and
            // the file already relies on that identity for every derivative
            // gain: `M⁺ = P⁺e₀ = M − M·P₀₀/F = M·(F − P₀₀)/F = M·R/F = R·K`.
            // The VALUE covariance was not using it. `P⁺[0][j] = P[0][j]·R/F`
            // is a PRODUCT — no subtraction, and no shared variable appearing
            // twice, which is what the corner form above cannot exploit: it
            // ranges `P[0][j]` as both `a` and `c` independently even though
            // they are one entry, and so cannot see the factorization at all.
            //
            // Measured at order 2, ρ = −13.8411 (the log-lambda the #2300
            // search refuses at), node 31: `P⁺₀₁` carries value `41.56` with
            // width `1.06e3`, and `P⁺₁₁` value `1.14e4` with width `3.55e8`,
            // while `P⁺₀₀` — the one entry that already had its exact range —
            // is `0.92 ± 0.99`. The unobserved entries are the runaway, and the
            // transition feeds them straight back into the observed one as
            // `P₀₀ + 2δP₀₁ + δ²P₁₁`.
            for i in 0..order {
                let exact = gain[i].mul(r);
                intersect_with_independent_enclosure(&mut p_new[i][0], exact);
                if i != 0 {
                    intersect_with_independent_enclosure(&mut p_new[0][i], exact);
                }
            }
            intersect_observed_covariance_exact_range(&mut p_new[0][0], m_star[0], r);
            // An update can only remove variance: `P⁺ = P⁻ − M Mᵀ/F ⪯ P⁻`, so
            // every diagonal is bounded above by the one it came from. The
            // corner form above cannot see this, because it ranges `P[i][0]`
            // and `P[0][j]` independently even when they are the same entry.
            for i in 0..order {
                let ceiling = p_star[i][i].hi.max(p_new[i][i].value);
                if p_new[i][i].hi > ceiling {
                    p_new[i][i].hi = ceiling;
                }
            }
            intersect_covariance_minors(&mut p_new, order);
            // The same update through the CARRIED factor, where it is one
            // column scaling and contains no subtraction at all.
            if carried_factor.is_none() {
                carried_factor = ball_cholesky(&p_star, order);
            }
            if let (Some(factor), Some(beta)) = (carried_factor, ball_sqrt(r.mul(inv_f))) {
                let updated_factor = ball_factor_update(&factor, beta, order);
                let gram = ball_factor_gram(&updated_factor, order);
                for i in 0..order {
                    for j in 0..order {
                        intersect_with_independent_enclosure(&mut p_new[i][j], gram[i][j]);
                    }
                }
                carried_factor = Some(updated_factor);
            } else {
                carried_factor = None;
            }

            // Derivative covariances through the JOSEPH form, which for the
            // derivative jets is not a reformulation but an exact cancellation
            // (#2614).
            //
            // Writing `A = I − K e₀ᵀ`, the update is `P⁺ = A P` and the Joseph
            // form `P⁺ = A P Aᵀ + R K Kᵀ` is algebraically identical. `R` is
            // `1/wₜ` — data, not a function of ρ — so differentiating the
            // Joseph form gives
            //
            //     dP⁺ = A dP Aᵀ + dK·(R Kᵀ − M⁺ᵀ) + (R K − M⁺)·dKᵀ
            //
            // and the first column of the UPDATED covariance is
            //
            //     M⁺ = P⁺e₀ = M − M·P₀₀/F = M·(F − P₀₀)/F = M·R/F = R·K
            //
            // so `R K − M⁺ = 0` EXACTLY and both `dK` terms vanish:
            //
            //     dP⁺ = A · dP · Aᵀ
            //
            // The first derivative is a pure congruence of the derivative, and
            // `dK` does not enter it at all. That matters here because the
            // subtractive form this replaces built each `s_k` from products of
            // derivative quantities divided by `F`, so an interval evaluation
            // widened multiplicatively at every one of the ~180 updates until
            // the accumulators saturated at `[−∞, +∞]` while their centres
            // stayed exact. The congruence enters the derivative LINEARLY with
            // value-side coefficients, and in the scalar case reads
            // `dP⁺/dρ = (R/F)²·dP⁻/dρ` with `0 < R/F < 1` — a contraction where
            // the old form accumulated a cancelling difference.
            //
            // Nothing here is a bound, a tolerance or a restored invariant: it
            // is the same quantity through an expression that does not cancel,
            // so it cannot make a non-stationary point certify.
            let d1_pred = p_star_d1;
            let d2_pred = p_star_d2;
            let d3_pred = p_star_d3;
            // Gain jets from the UPDATED covariance, not from a subtractive
            // recursion (#2614, second measurement).
            //
            // The identity that made `dP+ = A dP A^T` exact gives the gain
            // derivatives for free. `M+ = R*K` with `R = 1/w_t` constant in rho,
            // so `K = M+/R` and
            //
            //     d^k K / drho^k  =  (d^k M+ / drho^k) / R
            //
            // Every gain jet is a column of the corresponding UPDATED covariance
            // jet divided by a constant: no subtraction, no division by `F`, no
            // accumulation of derivative-times-derivative products.
            //
            // The measurement that forced this. After the covariance jets became
            // congruences, the refusal -- now carrying its own evidence -- named
            // the first accumulator to leave the finite range as
            // `sum_v2_over_f_d3`, at node 63 of 179, q = 1.641e7, value
            // 6.766e-5, enclosure [-inf, +inf]. That accumulator is fed by the
            // MEAN derivative chain `a_d3 -> v_d3 -> vv_d3`, which consumes
            // `gain_d3` -- the one recursion the congruence rewrite left alone,
            // still built as three subtractions of derivative products divided
            // by `F`, once per node.
            //
            // The staging is well founded, not circular: `d1` needs only `gain`;
            // `gain_d1` is then `p_new_d1`'s first column; `d2` needs `gain_d1`;
            // and so on. Each jet exists exactly when the next one needs it,
            // which is also why the mean update now follows this block.
            // `A[0][0] = 1 − K₀ = R/F` EXACTLY — never `1 − K₀` as a
            // subtraction. See `ball_update_operator`: that single entry is
            // where the `d3` jet was losing everything, and the measurement
            // that found it is on #2614 (`F'''` at `+/-4.1e247` by node 62,
            // ~`10^4` per node). The same operator carries the VALUE covariance
            // above, for the same reason.
            //
            // That measurement also reported `d1` and `d2` as clean, and this
            // comment used to say so without qualification. THAT IS FALSE as a
            // general statement, and it stood long enough to send a later lane
            // looking inside the `d3` recursion for a defect that was never
            // there. It was a reading at ONE rho. Measured at `rho = -13.8411`
            // on the same nodes at order 2, the per-node enclosure WIDTHS over
            // nodes 40..43 are
            //
            //     d1: 1.36e46  -> 2.34e49  -> 4.43e52   -> 9.18e55    (x1.7e3)
            //     d2: 1.06e93  -> 3.12e99  -> 1.11e106  -> 4.80e112   (x3e6)
            //     d3: 1.24e140 -> 6.23e149 -> 4.21e159  -> 3.76e169   (x6e9)
            //
            // — exponents exactly 1 : 2 : 3, so `d3` is `d1`'s growth CUBED and
            // not a defect of its own. At node 40 the congruence `A D3 Aᵀ` is
            // 5.17e133 against a predicted `d3` of 1.68e135, i.e. it CONTRACTS
            // by 32x, while the two terms carrying `D1` are the large ones.
            // Every value is stable to 15 digits throughout. The repair that
            // followed is the carried factor below, not anything in this block.

            // dP⁺ = A · dP · Aᵀ  (the `dK` terms cancel exactly, since M⁺ = R·K)
            //
            // Applied to the ZONOTOPE, not to a matrix of intervals. This is
            // the same recursion the carried `−dP` Cholesky factor used to
            // guard, and the factor is gone with it: a factor can only be
            // re-seeded from an enclosure that still PROVES `−dP ⪰ 0`, so once
            // the componentwise widths grew past that it could not come back,
            // which is exactly the window the trace shows (`w(dP₁₁)` 1.4e−2 at
            // node 40, 7.5e2 at node 80, back to 4e−9 at node 160 when a
            // re-seed finally succeeded). A zonotope needs no re-seeding
            // because it never loses the structure in the first place.
            if !covariance_d1.apply(
                &zonotope_congruence_map(&a_operator, &a_operator, order),
                &[Ball::ZERO; COVARIANCE_D1_DIM],
            ) || !project_symmetric_zonotope(&mut covariance_d1, order)
            {
                return Err(SplineScoreProofError::InvalidArithmetic {
                    context: "covariance-derivative zonotope update",
                });
            }
            let mut p_new_d1 = zonotope_to_matrix(&covariance_d1, order);
            // `0 ⪯ −dP/dρ ⪯ P`, applied to the MATERIALIZED view the consumers
            // below read rather than to the zonotope that carries it: the bound
            // is a fact about the matrix, the zonotope is a representation of
            // it, and every consumer here — `gain_d1`, and through it every
            // higher jet — reads this matrix.
            //
            // Only once the diffuse rank is consumed: until then `P*` is the
            // proper PART of a two-matrix decomposition and not the filtered
            // covariance the concavity argument is about.
            if diffuse_rank == 0 {
                intersect_derivative_covariance_below_its_own_covariance(
                    &mut p_new_d1,
                    &p_new,
                    order,
                );
            }
            let mut gain_d1 = [Ball::ZERO; MAX_ORDER];
            for i in 0..order {
                gain_d1[i] = p_new_d1[i][0].div_positive(r);
            }
            let a_d1_operator = ball_update_operator_derivative(&gain_d1, order);

            // d²P⁺ = A D₂ Aᵀ + A′D₁Aᵀ + A D₁A′ᵀ
            let mut p_new_d2 = ball_congruence(&a_operator, &d2_pred, &a_operator, order);
            let d2_cross = ball_congruence(&a_d1_operator, &d1_pred, &a_operator, order);
            for i in 0..order {
                for j in 0..order {
                    p_new_d2[i][j] = p_new_d2[i][j].add(d2_cross[i][j]).add(d2_cross[j][i]);
                }
            }
            let mut gain_d2 = [Ball::ZERO; MAX_ORDER];
            for i in 0..order {
                gain_d2[i] = p_new_d2[i][0].div_positive(r);
            }
            let a_d2_operator = ball_update_operator_derivative(&gain_d2, order);

            // d³P⁺ = A D₃ Aᵀ + A″D₁Aᵀ + A D₁A″ᵀ + 2A′D₂Aᵀ + 2A D₂A′ᵀ + 2A′D₁A′ᵀ
            let d3_congruence = ball_congruence(&a_operator, &d3_pred, &a_operator, order);
            let mut p_new_d3 = d3_congruence;
            let d3_second = ball_congruence(&a_d2_operator, &d1_pred, &a_operator, order);
            let d3_first = ball_congruence(&a_d1_operator, &d2_pred, &a_operator, order);
            let d3_both = ball_congruence(&a_d1_operator, &d1_pred, &a_d1_operator, order);
            for i in 0..order {
                for j in 0..order {
                    p_new_d3[i][j] = p_new_d3[i][j]
                        .add(d3_second[i][j])
                        .add(d3_second[j][i])
                        .add(d3_first[i][j].scale(2.0))
                        .add(d3_first[j][i].scale(2.0))
                        .add(d3_both[i][j].scale(2.0));
                }
            }
            let mut gain_d3 = [Ball::ZERO; MAX_ORDER];
            for i in 0..order {
                gain_d3[i] = p_new_d3[i][0].div_positive(r);
            }

            if let Some(sink) = trace.as_mut() {
                for record in [
                    ("f_star", f_star),
                    ("inv_f", inv_f),
                    ("a_operator_00", a_operator[0][0]),
                    ("gain_d1_0", gain_d1[0]),
                    ("gain_d2_0", gain_d2[0]),
                    ("gain_d3_0", gain_d3[0]),
                    ("d3_pred_00", d3_pred[0][0]),
                    ("d3_congruence_00", d3_congruence[0][0]),
                    ("d3_term_a2_d1_at", d3_second[0][0]),
                    ("d3_term_a1_d2_at", d3_first[0][0]),
                    ("d3_term_a1_d1_a1t", d3_both[0][0]),
                ] {
                    sink.push((t, record.0, record.1));
                }
                if order > 1 {
                    sink.push((t, "p_upd_01", p_new[0][1]));
                    sink.push((t, "d1_upd_01", p_new_d1[0][1]));
                    sink.push((t, "d2_upd_01", p_new_d2[0][1]));
                    sink.push((t, "d3_upd_01", p_new_d3[0][1]));
                }
                for i in 0..order {
                    sink.push((t, GAIN_NAMES[i], gain[i]));
                    sink.push((t, P_DIAGONAL_NAMES[i], p_new[i][i]));
                    sink.push((t, D1_DIAGONAL_NAMES[i], p_new_d1[i][i]));
                    sink.push((t, D2_DIAGONAL_NAMES[i], p_new_d2[i][i]));
                    sink.push((t, D3_DIAGONAL_NAMES[i], p_new_d3[i][i]));
                }
            }

            // Mean update as ONE linear map on the stacked `(a, a′)`.
            //
            // The jets follow from `a⁺⁽ᵏ⁾ = Σ_j C(k,j)·A⁽ʲ⁾a⁽ᵏ⁻ʲ⁾ + K⁽ᵏ⁾y` with
            // `A⁽ʲ⁾ = −K⁽ʲ⁾e₀ᵀ` for `j ≥ 1` and `v⁽ᵐ⁾ = −a⁽ᵐ⁾₀`:
            //
            //     a⁺    = A a    + K y
            //     a⁺′   = A a′   + K′v  = A a′ − K′e₀ᵀa + K′y
            //     a⁺″   = A a″   + 2K′v′  + K″v
            //     a⁺‴   = A a‴   + 3K′v″  + 3K″v′ + K‴v
            //
            // The first two are BLOCK LOWER-TRIANGULAR in `(a, a′)` — `a` never
            // reads its own jet — which is what lets one zonotope carry both.
            //
            // `64778c4e0` wrote the first line for every row and was corrected
            // by a row split, because under BOX arithmetic `A a + K y` costs
            // `(|a₀| + |y|)·w(K_i)` on rows `i ≥ 1` where the innovation form
            // `a_i + K_i v` costs only `|v|·w(K_i)`. Under affine arithmetic
            // that distinction is gone: the two are the same linear map, and a
            // zonotope applies a linear map exactly, so writing it as a matrix
            // costs nothing and is what the generators need. This is NOT a
            // reinstatement of that commit's claim — its form is used because
            // the enclosure it was wrong about is no longer a box.
            let y_node = Ball::exact(nodes[t].y);
            let mut update_map = zonotope_identity_map::<MEAN_DIM>(MEAN_BLOCKS * order);
            mean_set_block(&mut update_map, 0, 0, &a_operator, order);
            mean_set_block(&mut update_map, 1, 1, &a_operator, order);
            mean_set_block(&mut update_map, 1, 0, &a_d1_operator, order);
            let mut update_constant = [Ball::ZERO; MEAN_DIM];
            for i in 0..order {
                update_constant[i] = gain[i].mul(y_node);
                update_constant[order + i] = gain_d1[i].mul(y_node);
            }
            if !mean.apply(&update_map, &update_constant) {
                return Err(SplineScoreProofError::InvalidArithmetic {
                    context: "proper mean zonotope",
                });
            }
            let a_d2_contracted = ball_mat_vec(&a_operator, &a_d2, order);
            let a_d3_contracted = ball_mat_vec(&a_operator, &a_d3, order);
            for i in 0..order {
                a_d2[i] = a_d2_contracted[i]
                    .add(gain_d1[i].mul(v_d1).scale(2.0))
                    .add(gain_d2[i].mul(v));
                a_d3[i] = a_d3_contracted[i]
                    .add(gain_d1[i].mul(v_d2).scale(3.0))
                    .add(gain_d2[i].mul(v_d1).scale(3.0))
                    .add(gain_d3[i].mul(v));
            }

            p_star = p_new;
            p_star_d2 = p_new_d2;
            p_star_d3 = p_new_d3;
            ball_symmetrize(&mut p_star, order);
            ball_symmetrize(&mut p_star_d2, order);
            ball_symmetrize(&mut p_star_d3, order);
            intersect_proper_covariance_psd(&mut p_star, order)?;

            let vv = v.square();
            let vv_d1 = v.mul(v_d1).scale(2.0);
            let vv_d2 = v_d1.square().add(v.mul(v_d2)).scale(2.0);
            let vv_d3 = v.mul(v_d3).add(v_d1.mul(v_d2).scale(3.0)).scale(2.0);
            let logf_d1 = f_star_d1.mul(inv_f);
            let logf_d2 = f_star_d2.mul(inv_f).sub(logf_d1.square());
            let logf_d3 = f_star_d3
                .mul(inv_f)
                .sub(f_star_d2.mul(inv_f).mul(logf_d1).scale(3.0))
                .add(logf_d1.square().mul(logf_d1).scale(2.0));
            sum_log_f = sum_log_f.add(f_star.ln_positive());
            sum_log_f_d1 = sum_log_f_d1.add(logf_d1);
            sum_log_f_d2 = sum_log_f_d2.add(logf_d2);
            sum_log_f_d3 = sum_log_f_d3.add(logf_d3);
            let t0 = vv.mul(inv_f);
            let t1 = vv_d1.sub(t0.mul(f_star_d1)).mul(inv_f);
            let t2 = vv_d2
                .sub(t1.mul(f_star_d1).scale(2.0))
                .sub(t0.mul(f_star_d2))
                .mul(inv_f);
            let t3 = vv_d3
                .sub(t2.mul(f_star_d1).scale(3.0))
                .sub(t1.mul(f_star_d2).scale(3.0))
                .sub(t0.mul(f_star_d3))
                .mul(inv_f);
            sum_v2_over_f = sum_v2_over_f.add(t0);
            sum_v2_over_f_d1 = sum_v2_over_f_d1.add(t1);
            sum_v2_over_f_d2 = sum_v2_over_f_d2.add(t2);
            sum_v2_over_f_d3 = sum_v2_over_f_d3.add(t3);
            n_proper += 1;
            if diffuse_rank == 0 {
                intersect_first_order_accumulator_exact_ranges(
                    &mut sum_v2_over_f,
                    &mut sum_v2_over_f_d1,
                    &mut sum_log_f_d1,
                    weighted_energy,
                    n_proper,
                );
            }
            if let Some(sink) = trace.as_mut() {
                for i in 0..order {
                    sink.push((t, GAIN_NAMES[i], gain[i]));
                }
                for record in [
                    ("mean_a0", mean.coordinate(0)),
                    ("mean_a0_d1", mean.coordinate(MAX_ORDER)),
                    ("innovation_v", v),
                    ("innovation_v_d1", v_d1),
                    ("logf_d1", logf_d1),
                    ("term_t0", t0),
                    ("term_t1", t1),
                    ("acc_sum_log_f", sum_log_f),
                    ("acc_sum_log_f_d1", sum_log_f_d1),
                    ("acc_sum_v2", sum_v2_over_f),
                    ("acc_sum_v2_d1", sum_v2_over_f_d1),
                ] {
                    sink.push((t, record.0, record.1));
                }
            }
            // Refuse AT the node that diverged, not at the end of the pass.
            //
            // The end-of-pass check below reports that some accumulator is
            // non-finite and nothing more, which is what made #2614 expensive:
            // two exact repairs were aimed at the wrong term because the
            // refusal could not say WHICH accumulator went, WHERE, or how wide
            // it was. Checking here costs eight `is_finite` calls per proper
            // node and turns the refusal into the measurement.
            if let Some((accumulator, ball, contribution)) = [
                ("sum_log_f", sum_log_f, f_star.ln_positive()),
                ("sum_log_f_d1", sum_log_f_d1, logf_d1),
                ("sum_v2_over_f", sum_v2_over_f, t0),
                ("sum_v2_over_f_d1", sum_v2_over_f_d1, t1),
                // Second and third order last, and deliberately outside the
                // scan below: each has a closed-form global bound the
                // certificate substitutes, so neither can justify discarding a
                // value and slope that are finite.
                ("sum_log_f_d2", sum_log_f_d2, logf_d2),
                ("sum_v2_over_f_d2", sum_v2_over_f_d2, t2),
                ("sum_log_f_d3", sum_log_f_d3, logf_d3),
                ("sum_v2_over_f_d3", sum_v2_over_f_d3, t3),
            ]
            .into_iter()
            .take(GLOBALLY_BOUNDED_FROM)
            .find(|(_, ball, _)| !ball.is_finite())
            {
                return Err(SplineScoreProofError::AccumulatorDiverged {
                    node: t,
                    n_proper,
                    accumulator,
                    value: ball.value,
                    lo: ball.lo,
                    hi: ball.hi,
                    q_value: q.value,
                    contribution_lo: contribution.lo,
                    contribution_hi: contribution.hi,
                    f_star_d3_lo: f_star_d3.lo,
                    f_star_d3_hi: f_star_d3.hi,
                    updated_d3_lo: p_star_d3[0][0].lo,
                    updated_d3_hi: p_star_d3[0][0].hi,
                });
            }
        }

        if t + 1 < nodes.len() {
            let delta = Ball::exact(nodes[t + 1].x).sub(Ball::exact(nodes[t].x));
            let f_t = ball_transition(delta, order);
            // The transition is block diagonal on the stacked mean: every jet
            // is transported by the same `T`, since `T` does not depend on ρ.
            let mut transition_map = zonotope_identity_map::<MEAN_DIM>(MEAN_BLOCKS * order);
            mean_set_block(&mut transition_map, 0, 0, &f_t, order);
            mean_set_block(&mut transition_map, 1, 1, &f_t, order);
            if !mean.apply(&transition_map, &[Ball::ZERO; MEAN_DIM]) {
                return Err(SplineScoreProofError::InvalidArithmetic {
                    context: "mean zonotope transition",
                });
            }
            a_d2 = ball_mat_vec(&f_t, &a_d2, order);
            a_d3 = ball_mat_vec(&f_t, &a_d3, order);
            let f_t_t = ball_mat_t(&f_t, order);
            let ProcessNoiseTaylor {
                enclosure: q_noise,
                constant: q_noise_constant,
                shared_q: q_noise_shared_q,
            } = ball_process_noise_taylor(delta, q, order);
            let component_p_next = ball_mat_add(
                &ball_mat_mul(&ball_mat_mul(&f_t, &p_star, order), &f_t_t, order),
                &q_noise,
                order,
            );
            if !covariance.apply_with_shared_q(
                &zonotope_congruence_map(&f_t, &f_t, order),
                &q_noise_constant,
                &q_noise_shared_q,
            ) || !project_symmetric_zonotope(&mut covariance, order)
            {
                return Err(SplineScoreProofError::InvalidArithmetic {
                    context: "proper covariance zonotope transition",
                });
            }
            let mut p_next = zonotope_to_matrix(&covariance, order);
            for i in 0..order {
                for j in 0..order {
                    intersect_with_independent_enclosure(&mut p_next[i][j], component_p_next[i][j]);
                }
            }
            let mut p_next_d2 = ball_mat_add(
                &ball_mat_mul(&ball_mat_mul(&f_t, &p_star_d2, order), &f_t_t, order),
                &q_noise,
                order,
            );
            let mut p_next_d3 = ball_mat_sub(
                &ball_mat_mul(&ball_mat_mul(&f_t, &p_star_d3, order), &f_t_t, order),
                &q_noise,
                order,
            );
            // `dP⁻ = F·dP·Fᵀ − Q`, since `dQ/dρ = −Q`: a congruence plus an
            // EXACT constant, so the whole `dP` recursion is affine in `dP`
            // with coefficients built from the value covariance.
            let mut prediction_constant = [Ball::ZERO; COVARIANCE_D1_DIM];
            let mut prediction_shared_q = [0.0_f64; COVARIANCE_D1_DIM];
            for i in 0..order {
                for j in 0..order {
                    let index = i * order + j;
                    prediction_constant[index] = q_noise_constant[index].neg();
                    prediction_shared_q[index] = -q_noise_shared_q[index];
                }
            }
            if !covariance_d1.apply_with_shared_q(
                &zonotope_congruence_map(&f_t, &f_t, order),
                &prediction_constant,
                &prediction_shared_q,
            ) || !project_symmetric_zonotope(&mut covariance_d1, order)
            {
                return Err(SplineScoreProofError::InvalidArithmetic {
                    context: "covariance-derivative zonotope transition",
                });
            }
            ball_symmetrize(&mut p_next, order);
            ball_symmetrize(&mut p_next_d2, order);
            ball_symmetrize(&mut p_next_d3, order);
            // NOTE: `0 ⪯ −dP/dρ ⪯ P` is applied at the UPDATE, where the
            // derivative covariance is materialized from its zonotope, and not
            // here. Across the prediction the zonotope carries `dP/dρ` in
            // factored form and never forms the matrix, which is the whole
            // point of carrying it that way; the bound is a fact about the
            // matrix, so it belongs at the materialization its consumers read.
            p_star = p_next;
            p_star_d2 = p_next_d2;
            p_star_d3 = p_next_d3;
            if let Some(sink) = trace.as_mut() {
                for i in 0..order {
                    for j in 0..order {
                        sink.push((t, P_NEXT_ENTRY_NAMES[i][j], p_star[i][j]));
                    }
                }
                sink.push((t, "d1_next_00", covariance_d1.coordinate(0)));
                sink.push((t, "d2_next_00", p_star_d2[0][0]));
                sink.push((t, "d3_next_00", p_star_d3[0][0]));
            }
            if diffuse_rank > 0 {
                let mut pi_next = ball_mat_mul(&ball_mat_mul(&f_t, &p_inf, order), &f_t_t, order);
                ball_symmetrize(&mut pi_next, order);
                p_inf = pi_next;
            } else {
                intersect_proper_covariance_psd(&mut p_star, order)?;
            }
            // Carry the factor across the transition: `P⁻ = (F L)(F L)ᵀ + Q`,
            // so the prearray is `[F·L, L_Q]` and re-triangularizing it keeps
            // the factor `order`-wide. Nothing here subtracts.
            carried_factor = carried_factor.and_then(|factor| {
                let transported = ball_mat_mul(&f_t, &factor, order);
                let noise_factor = ball_cholesky(&q_noise, order)?;
                let mut prearray = [[Ball::ZERO; PREARRAY_COLUMNS]; MAX_ORDER];
                for i in 0..order {
                    for j in 0..order {
                        prearray[i][j] = transported[i][j];
                        prearray[i][order + j] = noise_factor[i][j];
                    }
                }
                let (next_factor, trailing, gram_scale) =
                    ball_retriangularize(&mut prearray, order, 2 * order);
                let gram = ball_factor_gram(&next_factor, order);
                // `P⁻ = (L Lᵀ + D)/gram_scale` with `0 ⪯ D` and every entry of
                // `D` bounded by `trace(D)`; `gram_scale` is `1 + O(eps)`.
                let slack = Ball {
                    value: 0.0,
                    lo: -trailing,
                    hi: trailing,
                };
                let scale = Ball {
                    value: 1.0,
                    lo: next_down_ball(1.0 / gram_scale),
                    hi: next_up_ball(gram_scale),
                };
                for i in 0..order {
                    for j in 0..order {
                        let evidence = gram[i][j].add(slack).mul(scale);
                        intersect_with_independent_enclosure(&mut p_star[i][j], evidence);
                    }
                }
                Some(next_factor)
            });
        }
    }

    let pass = BallFilterPass {
        sum_log_f,
        sum_log_f_d1,
        sum_log_f_d2,
        sum_log_f_d3,
        sum_v2_over_f,
        sum_v2_over_f_d1,
        sum_v2_over_f_d2,
        sum_v2_over_f_d3,
        n_proper,
    };
    if [
        pass.sum_log_f,
        pass.sum_log_f_d1,
        pass.sum_v2_over_f,
        pass.sum_v2_over_f_d1,
    ]
    .into_iter()
    .any(|ball| !ball.is_finite())
    {
        return Err(SplineScoreProofError::InvalidArithmetic {
            context: "diffuse filter accumulator",
        });
    }
    Ok(pass)
}

/// Fitted exact smoothing-spline posterior on the pooled knots.
#[derive(Clone, Debug)]
pub struct SplineScanFit {
    /// Smoothing-spline order `m` (penalize `∫(f^{(m)})²`); state dimension.
    /// `m = 1` is the random-walk/linear smoother, `m = 2` the cubic smoother,
    /// `m = 3` the quintic smoother.
    pub order: usize,
    /// Distinct sorted abscissae (pooled knots).
    pub knots: Vec<f64>,
    /// Smoothed posterior mean of `f` at each knot.
    pub mean: Vec<f64>,
    /// Smoothed posterior mean of `f′` at each knot, present only for order
    /// `m ≥ 2`. At `m = 1` the latent process is Brownian motion, which has NO
    /// pointwise derivative state (it is a.s. nondifferentiable), so this is
    /// `None` rather than a fabricated zero.
    pub deriv: Option<Vec<f64>>,
    /// Posterior variance of `f` at each knot (scaled by `sigma2`).
    pub var: Vec<f64>,
    /// Selected (or supplied) log smoothing parameter `log λ`.
    log_lambda: f64,
    /// Profiled (or supplied) observation variance σ².
    pub sigma2: f64,
    /// Concentrated diffuse restricted log-likelihood at the optimum, up to a
    /// λ- and data-independent additive constant. Differences across λ are
    /// exact REML criterion differences.
    pub restricted_loglik: f64,
    /// Ordinary fully normalized Gaussian log-likelihood at the fitted mean
    /// and profiled observation variance. Unlike `restricted_loglik`, this is
    /// on the common data-likelihood scale consumed by conditional AIC.
    pub log_likelihood: f64,
    /// Original training row count (pre-pooling; ties collapse to fewer
    /// knots), retained for every sample-size-based post-fit calculation.
    training_sample_size: std::num::NonZeroUsize,
    /// Weighted DATA residual sum of squares `Σ wᵢ (yᵢ − f̂(xᵢ))²` at the
    /// smoothed posterior mean. Stored explicitly because the profiled
    /// innovations quadratic `σ̂²·(n − order)` is the REML objective's
    /// quadratic — data residual energy PLUS process/roughness energy at the
    /// posterior mode — and is therefore NOT the Gaussian deviance.
    pub data_sse: f64,
    /// Smoothed full states `(f, f′)` per knot.
    smoothed_state: Vec<Vec2>,
    /// Smoothed full state covariances per knot (unit-σ² scale).
    smoothed_cov: Vec<Mat2>,
    /// RTS backward gains `G_t` (lag-one cross-covariance is `G_t · P^s_{t+1}`).
    rts_gain: Vec<Mat2>,
    /// q = 1/λ used by the pass (unit-σ² scale).
    q: f64,
    /// Pooled observation weight per knot (sum of tied raw weights).
    node_weight: Vec<f64>,
}

/// Move the response into its constant-null-space chart, pool tied abscissae,
/// and validate inputs. Returns nodes plus the within-tie weighted residual
/// sum, raw observation count, and chart origin.
///
/// Centering must precede pooling. A weighted tied-row mean formed from the
/// absolute response level leaks that level through floating-point products
/// and sums before the innovation recurrence ever sees the data. Computing
/// both the pooled mean and within-tie residual energy from the same centered
/// rows makes the translation-free chart the sole arithmetic authority.
fn pool_nodes(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    order: usize,
) -> Result<(Vec<PooledNode>, f64, usize, f64), String> {
    let n = x.len();
    if y.len() != n || w.len() != n {
        return Err(format!(
            "spline scan: length mismatch x={n}, y={}, w={}",
            y.len(),
            w.len()
        ));
    }
    for i in 0..n {
        if !(x[i].is_finite() && y[i].is_finite() && w[i].is_finite() && w[i] > 0.0) {
            return Err(format!(
                "spline scan: non-finite or non-positive input at row {i} (x={}, y={}, w={})",
                x[i], y[i], w[i]
            ));
        }
    }
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let response_origin = perm
        .first()
        .map(|&index| y[index])
        .ok_or_else(|| "spline scan: cannot pool an empty response".to_string())?;
    let centered_y = y
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let centered = value - response_origin;
            centered.is_finite().then_some(centered).ok_or_else(|| {
                format!("spline scan: centered response is non-finite at row {index}")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut nodes: Vec<PooledNode> = Vec::new();
    for &i in &perm {
        match nodes.last_mut() {
            Some(last) if last.x == x[i] => {
                let w_new = last.w + w[i];
                last.y = (last.y * last.w + centered_y[i] * w[i]) / w_new;
                last.w = w_new;
            }
            _ => nodes.push(PooledNode {
                x: x[i],
                y: centered_y[i],
                w: w[i],
            }),
        }
    }
    // Need the `order` diffuse dimensions plus at least one proper innovation.
    if nodes.len() < order + 1 {
        return Err(format!(
            "spline scan: order {order} needs at least {} distinct abscissae, got {}",
            order + 1,
            nodes.len()
        ));
    }
    // Within-tie residual sum Σ w_i (y_i − ȳ_group)², part of the profiled σ².
    let mut ssr_within = 0.0;
    let mut k = 0usize;
    for &i in &perm {
        while nodes[k].x != x[i] {
            k += 1;
        }
        let d = centered_y[i] - nodes[k].y;
        ssr_within += w[i] * d * d;
    }
    Ok((nodes, ssr_within, n, response_origin))
}

/// Concentrated diffuse restricted log-likelihood and its exact first three
/// derivatives with respect to `log λ` (σ² profiled). The derivatives are
/// propagated through the same diffuse Kalman recursion as the value; no
/// finite differencing or surrogate objective is involved. The third order
/// exists solely to anchor the certified-search enclosure on endpoint pairs
/// (#2300/#2614 fourth-order tail).
fn concentrated_criterion_jet(
    nodes: &[PooledNode],
    ssr_within: f64,
    n_obs: usize,
    log_lambda: f64,
    order: usize,
) -> Result<(f64, f64, f64, f64), String> {
    let q = gam_problem::checked_exp_log_strength(-log_lambda)
        .map_err(|error| format!("spline scan inverse log strength: {error}"))?;
    let pass = run_filter::<false>(nodes, q, order)?;
    // Profiled σ̂² over the proper innovations plus within-tie residuals;
    // the restricted degrees of freedom subtract the diffuse dimension `order`.
    let dof = (n_obs - order) as f64;
    let rss = pass.sum_v2_over_f + ssr_within;
    if rss <= 0.0 {
        return Err("spline scan: degenerate zero residual sum".to_string());
    }
    let sigma2 = rss / dof;
    if pass.n_proper != nodes.len() - order {
        return Err(format!(
            "spline scan: expected {} proper innovations, got {} (diffuse rank not consumed)",
            nodes.len() - order,
            pass.n_proper
        ));
    }
    let rss_d1 = pass.sum_v2_over_f_d1;
    let rss_d2 = pass.sum_v2_over_f_d2;
    let rss_d3 = pass.sum_v2_over_f_d3;
    let rss_log_d1 = rss_d1 / rss;
    let rss_log_d2 = rss_d2 / rss - rss_log_d1 * rss_log_d1;
    let rss_log_d3 = rss_d3 / rss - 3.0 * (rss_d2 / rss) * rss_log_d1
        + 2.0 * rss_log_d1 * rss_log_d1 * rss_log_d1;
    Ok((
        -0.5 * (pass.sum_log_f + dof * sigma2.ln()),
        -0.5 * (pass.sum_log_f_d1 + dof * rss_log_d1),
        -0.5 * (pass.sum_log_f_d2 + dof * rss_log_d2),
        -0.5 * (pass.sum_log_f_d3 + dof * rss_log_d3),
    ))
}

#[derive(Clone, Copy, Debug)]
struct CertifiedCriterionJet {
    jet: ScoreJet,
    value: Ball,
    derivative: Ball,
    curvature: Ball,
    third: Ball,
    /// Where the curvature and the third order came from. A search that quietly
    /// got weaker is the same defect class as a criterion that quietly drifted,
    /// so a certificate that fell back to a global constant names itself.
    curvature_source: BoundSource,
    third_source: BoundSource,
}

impl CertifiedCriterionJet {
    /// Which bounds anchored this endpoint, when either is not the exact jet.
    ///
    /// `curvature_source` / `third_source` exist so a certificate that fell back
    /// to a closed-form global constant NAMES ITSELF instead of quietly getting
    /// wider. A field nobody reads cannot do that: written-and-never-read IS the
    /// silent degradation these fields were added to prevent, and it is also a
    /// hard `-D dead-code` failure in any build of this crate as a plain library
    /// rather than a test target -- which `-p gam-solve --lib` never exercises,
    /// so it broke every integration binary while the usual measurement stayed
    /// green.
    ///
    /// `None` on the common path, so a reader only hears about a weakened anchor.
    fn weakened_anchor(self) -> Option<(BoundSource, BoundSource)> {
        if matches!(
            (self.curvature_source, self.third_source),
            (BoundSource::EndpointJet, BoundSource::EndpointJet)
        ) {
            None
        } else {
            Some((self.curvature_source, self.third_source))
        }
    }
}

/// Which bound anchored a derivative at this endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundSource {
    /// The exact endpoint `V‴` jet — the fourth-order tail of #2300/#2614.
    EndpointJet,
    /// The closed-form global bound `½(r/4 + 6ν)`, taken because the endpoint
    /// jet's enclosure left the finite range. The search stays CERTIFIED and
    /// its tail cells are merely wider: `(|V′|/L₃)^{1/2}` in place of
    /// the endpoint-pair `(|V′|/L₅)^{1/4}` rate. The third order exists SOLELY
    /// to anchor that radius, so losing it costs cells, never soundness.
    ///
    /// The measured boundary that makes this reachable, recorded where a reader
    /// meets it rather than left to be rediscovered: before #2614's centred
    /// Riccati/shared-`q` repair, smoothing order 3 refused throughout
    /// `-20 <= ρ <= -10` when the covariance-derivative zonotope overflowed.
    /// The repaired representation reaches the exact endpoint jets throughout
    /// that measured domain. This fallback remains the sound terminal bound for
    /// other inputs whose endpoint jet genuinely carries less information.
    AnalyticGlobalBound,
}

/// `|V″| ≤ ½(r/4 + 2ν)` and `|V‴| ≤ ½(r/4 + 6ν)`, the closed-form derivative
/// bounds derived in [`concentrated_criterion_enclosure`]'s own documentation,
/// in the same family as the fourth-order bound the radius already uses.
fn curvature_global_bound(proper_modes: f64, residual_dof: f64) -> f64 {
    0.5 * (0.25 * proper_modes + 2.0 * residual_dof)
}

fn third_derivative_global_bound(proper_modes: f64, residual_dof: f64) -> f64 {
    0.5 * (0.25 * proper_modes + 6.0 * residual_dof)
}

/// Conservative closed-form bound on the concentrated criterion's fifth
/// derivative with respect to `rho = log(lambda)`.
///
/// For a determinant mode `u in [0,1]`, the fifth derivative is
///
/// `u(1-u)(1 - 14u + 36u² - 24u³)`,
///
/// up to sign. Absolute coefficient summation and `u(1-u) <= 1/4` bound it by
/// `(1+14+36+24)/4 = 18.75`. For one normalized residual kernel,
///
/// `t⁽⁵⁾/t = u(1 - 30u + 150u² - 240u³ + 120u⁴)`,
///
/// up to sign, hence `|t⁽⁵⁾/t| <= 541`; the ratios through order four are each
/// bounded by one as documented on [`concentrated_criterion_enclosure`].
/// Faa di Bruno for `(log R)⁽⁵⁾` adds absolute coefficients
/// `5+10+20+30+60+24 = 149` from those lower ratios, for `541+149 = 690`.
///
/// These deliberately elementary coefficient bounds are wider than the exact
/// polynomial ranges but need no spectral decomposition or data-dependent
/// tail assumption. Their consumer integrates the bound four times, so the
/// resulting remainder is still far below endpoint evaluator error.
fn fifth_derivative_global_bound(proper_modes: Ball, residual_dof: Ball) -> Ball {
    proper_modes
        .scale(18.75)
        .add(residual_dof.scale(690.0))
        .scale(0.5)
}

/// Intersect a derivative enclosure with its closed-form global bound.
///
/// ONE rule at every order, not a branch taken only on failure: the minimum of
/// two valid upper bounds on the same quantity is a valid upper bound, so this
/// is sound wherever it applies and strictly tighter than the endpoint jet
/// whenever the jet is the wider of the two. The fallback is then the special
/// case where the jet carries no information at all.
///
/// The thing being replaced in that case is `[-inf, +inf]`, which is not a
/// stronger object than a finite global bound — the search cannot bracket on
/// it. A certificate that took the global bound says so through its
/// [`BoundSource`], because a search that silently got weaker is the defect
/// class this whole issue is about.
fn intersect_with_global_bound(ball: Ball, bound: f64) -> (Ball, BoundSource) {
    if ball.is_finite() {
        let lo = ball.lo.max(-bound);
        let hi = ball.hi.min(bound);
        if lo <= hi {
            return (
                Ball {
                    value: ball.value.clamp(lo, hi),
                    lo,
                    hi,
                },
                BoundSource::EndpointJet,
            );
        }
    }
    (
        Ball {
            value: ball.value.clamp(-bound, bound),
            lo: -bound,
            hi: bound,
        },
        BoundSource::AnalyticGlobalBound,
    )
}

/// The concentrated criterion evaluated once with a simultaneous
/// directed-rounding proof of all four returned components.
fn certified_concentrated_criterion_jet(
    nodes: &[PooledNode],
    ssr_within: f64,
    n_obs: usize,
    log_lambda: f64,
    order: usize,
) -> Result<CertifiedCriterionJet, SplineScoreProofError> {
    let q_value = gam_problem::checked_exp_log_strength(-log_lambda).map_err(|error| {
        SplineScoreProofError::InvalidInput(format!("spline scan inverse log strength: {error}"))
    })?;
    let q_enclosure = gam_math::score_opt::certified_exp(-log_lambda).ok_or(
        SplineScoreProofError::InvalidArithmetic {
            context: "inverse log-strength exponential",
        },
    )?;
    let q = Ball::certified(q_value, q_enclosure);
    let pass = run_filter_ball(nodes, q, order)?;
    if pass.n_proper != nodes.len() - order {
        return Err(SplineScoreProofError::InvalidInput(format!(
            "spline scan: expected {} proper innovations, got {} (diffuse rank not consumed)",
            nodes.len() - order,
            pass.n_proper
        )));
    }

    let dof = Ball::exact((n_obs - order) as f64);
    let rss = pass.sum_v2_over_f.add(Ball::exact(ssr_within));
    if !(rss.lo > 0.0) {
        return Err(SplineScoreProofError::NonPositiveProfileResidual {
            enclosure: rss.interval(),
        });
    }
    let sigma2 = rss.div_positive(dof);
    let rss_d1 = pass.sum_v2_over_f_d1;
    let rss_d2 = pass.sum_v2_over_f_d2;
    let rss_d3 = pass.sum_v2_over_f_d3;
    let mut rss_log_d1 = rss_d1.div_positive(rss);
    // `0 ≤ (Σ v²/F̃)′ ≤ Σ v²/F̃ ≤ rss` bounds this RATIO by one, and the division
    // above cannot see that: it ranges numerator and denominator independently,
    // so the same dependency loss the accumulator ranges just removed reappears
    // one line later. Measured at order 3, ρ = −16.6135: the accumulator pair is
    // `(21.2 ± 34.8, 0.494 ± 38.6)` — both inside their ranges — and the
    // quotient still reaches `2.8e2`, carrying the certified derivative to
    // `[−2.51e4, 88.5]` when the model fixes it at `[−ν/2, r/2]`. The upper end
    // is already exactly `r/2 = 88.5`; only the quotient's lower end was loose.
    intersect_with_exact_range(&mut rss_log_d1, 0.0, 1.0);
    let rss_log_d2 = rss_d2.div_positive(rss).sub(rss_log_d1.square());
    let rss_log_d3 = rss_d3
        .div_positive(rss)
        .sub(rss_d2.div_positive(rss).mul(rss_log_d1).scale(3.0))
        .add(rss_log_d1.square().mul(rss_log_d1).scale(2.0));
    let value = pass
        .sum_log_f
        .add(dof.mul(sigma2.ln_positive()))
        .scale(-0.5);
    let derivative = pass.sum_log_f_d1.add(dof.mul(rss_log_d1)).scale(-0.5);
    let curvature = pass.sum_log_f_d2.add(dof.mul(rss_log_d2)).scale(-0.5);
    let third = pass.sum_log_f_d3.add(dof.mul(rss_log_d3)).scale(-0.5);
    if [value, derivative]
        .into_iter()
        .any(|ball| !ball.is_finite())
    {
        return Err(SplineScoreProofError::InvalidArithmetic {
            context: "concentrated criterion",
        });
    }
    // The third order is an OPTIMISATION, not a requirement: endpoint pairs
    // linearly interpolate it so the #2300/#2614 tail remainder is fourth
    // order. When its enclosure leaves the finite range, the closed-form global
    // bound keeps the certificate valid and costs only a wider tail cell.
    // Refusing the whole jet instead discards a value, slope and curvature that
    // are all finite, which is what the divergence refusal used to do at every
    // order-3 rho below -6.
    let proper_modes = (nodes.len() - order) as f64;
    let residual_dof = (n_obs - order) as f64;
    let (curvature, curvature_source) = intersect_with_global_bound(
        curvature,
        curvature_global_bound(proper_modes, residual_dof),
    );
    let (third, third_source) = intersect_with_global_bound(
        third,
        third_derivative_global_bound(proper_modes, residual_dof),
    );
    Ok(CertifiedCriterionJet {
        jet: ScoreJet {
            value: value.value,
            derivative: derivative.value,
            curvature: curvature.value,
            third: third.value,
        },
        value,
        derivative,
        curvature,
        third,
        curvature_source,
        third_source,
    })
}

/// Rigorous interval enclosure of the score's first two derivatives.
///
/// After eliminating the diffuse polynomial null space, the Gaussian profile
/// is an affine covariance pencil. Every determinant mode has response
/// `u in [0,1]`; every normalized profiled-residual derivative is a convex
/// average of the same kernels. Consequently
///
/// `|L'| <= 1/2 (r/4 + nu)`, `|L''| <= 1/2 (r/4 + 2 nu)`,
/// `|L'''| <= 1/2 (r/4 + 6 nu)`, `|L''''| <= 1/2 (r/4 + 26 nu)`, and
/// `|L'''''| <= 1/2 (18.75 r + 690 nu)`,
///
/// where `r` is the number of proper innovation modes and `nu=n-order` is the
/// residual d.f. For one normalized determinant contribution, the fourth
/// derivative is `u(1-u)(1-6u+6u^2)`, whose magnitude is at most `1/4` on
/// `u in [0,1]`; so is the FIRST derivative `u(1-u)`, which is why every order
/// carries the same `r/4` term. For each residual kernel `t = z^2 (1-u)`,
/// every ratio `|t^{(k)}/t| <= 1` for `k <= 4`, so Faa di Bruno on `log R`
/// gives `1 = 1` at first order, `1+1 = 2` at second, `1+3+2 = 6` at third and
/// `1+4+3+12+6 = 26` at fourth. Within-tie residual energy is
/// lambda-independent and only tightens these bounds.
/// Endpoint jets plus these analytic Lipschitz bounds therefore enclose the
/// entire interval without a sampling lattice.
///
/// The cell is the union of its two half-cells. Every point is within
/// `h=(hi-lo)/2` of its nearest endpoint, so the left endpoint anchors signed
/// displacements `[0,h]` and the right endpoint anchors `[-h,0]`. The two
/// certified endpoint `V'''` balls define a linear interpolant. The standard
/// interpolation error
///
/// `|V'''(x) - linear(V'''(lo), V'''(hi))| <= L5 (x-lo)(hi-x)/2`
///
/// integrates from either endpoint to give maximum half-cell remainders
/// `L5*w^5/960`, `L5*w^4/128`, and `L5*w^3/24` for `V`, `V'`, and `V''`.
/// Evaluating the resulting quartic polynomial over each signed half-cell and
/// hulling the two results gives one theorem uniformly for all three channels.
/// Independently, integrating over the full cell from EACH endpoint gives
/// value remainders `L5*w^5/80`. Both full-cell Taylor ranges contain every
/// score in the cell, so their intersection with the half-cell hull removes
/// endpoint roundoff asymmetry without weakening the theorem.
///
/// Nearest-endpoint geometry first removes factors 16, 8, and 4 of false
/// fourth-derivative uncertainty. Even then, a data-independent global `L4`
/// dominates an exponentially saturated endpoint jet. Interpolating the
/// endpoint third derivatives removes that constant floor without a fourth
/// filter jet: only the globally bounded interpolation error remains, one
/// asymptotic order smaller.
///
/// A second independent curvature theorem protects stationary isolation from
/// a loose covariance second-derivative recurrence. The derivative endpoint
/// balls give a secant `s=(V'(hi)-V'(lo))/w`; the mean-value theorem supplies
/// `xi` in the cell with `V''(xi)=s`, and the global `|V'''|` bound then gives
/// `V''(x) in s ± L3*w` everywhere in the cell. Intersecting this range with
/// the endpoint-third range can only tighten a valid outer enclosure. It is
/// especially decisive near a root, where endpoint `V'` remains sharp even
/// when direct interval propagation has lost the sign of `V''`.
///
/// Once that whole-cell curvature range `C` is known, integrating it from both
/// endpoints gives two further derivative theorems:
/// `V'(x) in V'(lo)+C*[0,w]` and `V'(x) in V'(hi)+C*[-w,0]`. Their intersection
/// with the endpoint-third derivative range preserves the same exact-real
/// derivative while recovering local root information from tight endpoint
/// balls. A disjoint intersection is an internal certificate contradiction,
/// never a reason to widen or fall back.
///
/// All polynomial operations below use endpoint BALLS and outward interval
/// arithmetic. The search caches those balls alongside each endpoint jet, so
/// this function performs no filter pass of its own and includes the endpoint
/// evaluator's directed-rounding error in every returned channel.
fn concentrated_criterion_enclosure(
    n_nodes: usize,
    n_obs: usize,
    left: ScoreSample,
    right: ScoreSample,
    left_certificate: CertifiedCriterionJet,
    right_certificate: CertifiedCriterionJet,
    order: usize,
) -> Result<DerivativeEnclosure, SplineScoreProofError> {
    let (lo, hi) = (left.x, right.x);
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return Err(SplineScoreProofError::InvalidInput(format!(
            "spline scan: invalid score-enclosure interval [{lo}, {hi}]"
        )));
    }
    if lo == hi {
        return Ok(DerivativeEnclosure {
            score: ScoreValueEnclosure {
                value: ClosedInterval::new(
                    left_certificate.value.lo.min(right_certificate.value.lo),
                    left_certificate.value.hi.max(right_certificate.value.hi),
                ),
                evaluation_error: left_certificate
                    .value
                    .forward_error()
                    .max(right_certificate.value.forward_error()),
            },
            derivative: ClosedInterval::new(
                left_certificate
                    .derivative
                    .lo
                    .min(right_certificate.derivative.lo),
                left_certificate
                    .derivative
                    .hi
                    .max(right_certificate.derivative.hi),
            ),
            curvature: ClosedInterval::new(
                left_certificate
                    .curvature
                    .lo
                    .min(right_certificate.curvature.lo),
                left_certificate
                    .curvature
                    .hi
                    .max(right_certificate.curvature.hi),
            ),
        });
    }
    let width = Ball::exact(hi).sub(Ball::exact(lo));
    if !(width.lo > 0.0) {
        return Err(SplineScoreProofError::InvalidArithmetic {
            context: "positive score-enclosure width",
        });
    }
    let proper_modes = Ball::exact((n_nodes - order) as f64);
    let residual_dof = Ball::exact((n_obs - order) as f64);
    let fifth_abs_bound = fifth_derivative_global_bound(proper_modes, residual_dof);
    let third_abs_bound = proper_modes
        .scale(0.25)
        .add(residual_dof.scale(6.0))
        .scale(0.5);
    // Announce a weakened anchor at the point it anchors.
    //
    // Both endpoints feed their nearest half-cell, so either one falling back
    // to a global constant widens that half. Reported here rather than at
    // construction so the message names the consequence, not just the fact.
    for (side, weakened) in [
        ("left", left_certificate.weakened_anchor()),
        ("right", right_certificate.weakened_anchor()),
    ] {
        if let Some((curvature_source, third_source)) = weakened {
            log::debug!(
                "spline scan enclosure: {side} endpoint curvature anchored by \
                 {curvature_source:?}, third order by {third_source:?}. A global-bound \
                 anchor keeps the search CERTIFIED and widens its tail cells -- half rate \
                 in place of fourth-order rate -- so it costs cells, never soundness."
            );
        }
    }
    let half_width = width.scale(0.5);
    let width2 = width.square();
    let width3 = width2.mul(width);
    let width4 = width2.square();
    let width5 = width4.mul(width);
    let value_remainder = fifth_abs_bound
        .mul(width5)
        .div_positive(Ball::exact(960.0))
        .hi;
    let derivative_remainder = fifth_abs_bound
        .mul(width4)
        .div_positive(Ball::exact(128.0))
        .hi;
    let curvature_remainder = fifth_abs_bound
        .mul(width3)
        .div_positive(Ball::exact(24.0))
        .hi;
    let third_slope = right_certificate
        .third
        .sub(left_certificate.third)
        .div_positive(width);

    // Integrate the endpoint-pair linear interpolant of V''' from either
    // endpoint over a signed displacement. Keeping the sign of `d` is materially
    // tighter than replacing every term by an absolute-value radius, while
    // ordinary interval arithmetic still gives an outer range despite
    // dependencies among powers of `d`.
    let endpoint_enclosure = |certificate: CertifiedCriterionJet,
                              displacement: ClosedInterval,
                              value_remainder: f64,
                              derivative_remainder: f64,
                              curvature_remainder: f64| {
        let d = Ball::certified(0.0, displacement);
        let d2 = d.square();
        let d3 = d2.mul(d);
        let d4 = d2.square();
        let value = certificate
            .value
            .add(certificate.derivative.mul(d))
            .add(certificate.curvature.mul(d2).scale(0.5))
            .add(certificate.third.mul(d3).div_positive(Ball::exact(6.0)))
            .add(third_slope.mul(d4).div_positive(Ball::exact(24.0)))
            .interval()
            .add(ClosedInterval::new(-value_remainder, value_remainder));
        let derivative = certificate
            .derivative
            .add(certificate.curvature.mul(d))
            .add(certificate.third.mul(d2).scale(0.5))
            .add(third_slope.mul(d3).div_positive(Ball::exact(6.0)))
            .interval()
            .add(ClosedInterval::new(
                -derivative_remainder,
                derivative_remainder,
            ));
        let curvature = certificate
            .curvature
            .add(certificate.third.mul(d))
            .add(third_slope.mul(d2).scale(0.5))
            .interval()
            .add(ClosedInterval::new(
                -curvature_remainder,
                curvature_remainder,
            ));
        (value, derivative, curvature)
    };

    let (left_value, left_derivative, left_curvature) = endpoint_enclosure(
        left_certificate,
        ClosedInterval::new(0.0, half_width.hi),
        value_remainder,
        derivative_remainder,
        curvature_remainder,
    );
    let (right_value, right_derivative, right_curvature) = endpoint_enclosure(
        right_certificate,
        ClosedInterval::new(-half_width.hi, 0.0),
        value_remainder,
        derivative_remainder,
        curvature_remainder,
    );
    let half_cell_score = ClosedInterval::new(
        left_value.lo.min(right_value.lo),
        left_value.hi.max(right_value.hi),
    );
    let full_value_remainder = fifth_abs_bound
        .mul(width5)
        .div_positive(Ball::exact(80.0))
        .hi;
    let full_derivative_remainder = fifth_abs_bound
        .mul(width4)
        .div_positive(Ball::exact(24.0))
        .hi;
    let full_curvature_remainder = fifth_abs_bound
        .mul(width3)
        .div_positive(Ball::exact(12.0))
        .hi;
    let (full_left_value, _, _) = endpoint_enclosure(
        left_certificate,
        ClosedInterval::new(0.0, width.hi),
        full_value_remainder,
        full_derivative_remainder,
        full_curvature_remainder,
    );
    let (full_right_value, _, _) = endpoint_enclosure(
        right_certificate,
        ClosedInterval::new(-width.hi, 0.0),
        full_value_remainder,
        full_derivative_remainder,
        full_curvature_remainder,
    );
    let score_value = ClosedInterval::new(
        half_cell_score
            .lo
            .max(full_left_value.lo)
            .max(full_right_value.lo),
        half_cell_score
            .hi
            .min(full_left_value.hi)
            .min(full_right_value.hi),
    );
    if !(score_value.lo <= score_value.hi) {
        return Err(SplineScoreProofError::InvalidArithmetic {
            context: "endpoint score-enclosure intersection",
        });
    }
    let endpoint_third_derivative = ClosedInterval::new(
        left_derivative.lo.min(right_derivative.lo),
        left_derivative.hi.max(right_derivative.hi),
    );
    let endpoint_third_curvature = ClosedInterval::new(
        left_curvature.lo.min(right_curvature.lo),
        left_curvature.hi.max(right_curvature.hi),
    );
    let derivative_secant = right_certificate
        .derivative
        .sub(left_certificate.derivative)
        .div_positive(width);
    let secant_radius = third_abs_bound.mul(width).hi;
    let secant_curvature = derivative_secant
        .interval()
        .add(ClosedInterval::new(-secant_radius, secant_radius));
    let curvature = ClosedInterval::new(
        endpoint_third_curvature.lo.max(secant_curvature.lo),
        endpoint_third_curvature.hi.min(secant_curvature.hi),
    );
    if !(curvature.lo <= curvature.hi) {
        return Err(SplineScoreProofError::InvalidArithmetic {
            context: "curvature secant intersection",
        });
    }
    let curvature_ball = Ball::certified(0.0, curvature);
    let derivative_from_left = left_certificate
        .derivative
        .add(curvature_ball.mul(Ball::certified(0.0, ClosedInterval::new(0.0, width.hi))))
        .interval();
    let derivative_from_right = right_certificate
        .derivative
        .add(curvature_ball.mul(Ball::certified(0.0, ClosedInterval::new(-width.hi, 0.0))))
        .interval();
    let derivative_from_curvature = ClosedInterval::new(
        derivative_from_left.lo.max(derivative_from_right.lo),
        derivative_from_left.hi.min(derivative_from_right.hi),
    );
    let derivative = ClosedInterval::new(
        endpoint_third_derivative
            .lo
            .max(derivative_from_curvature.lo),
        endpoint_third_derivative
            .hi
            .min(derivative_from_curvature.hi),
    );
    if !(derivative.lo <= derivative.hi) {
        return Err(SplineScoreProofError::InvalidArithmetic {
            context: "derivative curvature-integral intersection",
        });
    }
    let evaluation_error = left_certificate
        .value
        .forward_error()
        .max(right_certificate.value.forward_error());
    Ok(DerivativeEnclosure {
        score: ScoreValueEnclosure {
            value: score_value,
            evaluation_error,
        },
        derivative,
        curvature,
    })
}

/// Exact diffuse smoother for the `order−1` partially-diffuse leading nodes
/// (#1044 — the multi-node generalization of the `m = 2` reverse-Markov
/// closure).
///
/// Ordinary RTS recovers every node `t ≥ order−1` (where the filtered
/// distribution is proper). The first `order−1` nodes are partially diffuse:
/// their filtered covariance still carries unresolved diffuse mass, so RTS —
/// which needs the predicted covariance `P_{t+1|t}` to be invertible — cannot
/// reach them. By the Markov property the leading block depends on all future
/// data ONLY through the first proper smoothed node `α_{order−1}`:
///
///   p(α_{0..order−2} | y) = ∫ p(α_{0..order−2} | α_{order−1}, y_{0..order−2})
///                             · p(α_{order−1} | y) dα_{order−1}.
///
/// The inner conditional is a proper Gaussian: it is the flat (improper)
/// leading prior tightened by the Markov increments `(α_{t+1} − Fα_t)ᵀ(qQ)⁻¹(·)`
/// and the leading observations `w_t (y_t − f_t)²`, with `α_{order−1}` entering
/// linearly through the last increment. Writing `u = (α_0, …, α_{order−2})`,
///
///   u | α_{order−1} ~ N(C·α_{order−1} + d,  Σ),   Σ = Λ⁻¹,
///   Λ  = increments(F'(qQ)⁻¹F …) + leading obs,
///   d  = Σ·b_const,   C = Σ·B   (B = the pinned-node coupling F'(qQ)⁻¹),
///
/// and pushing the smoothed `α_{order−1} ~ N(α̂_p, V_p)` through the affine map
/// gives the EXACT smoothed leading block, its covariances, and the lag-one
/// cross-covariances `Cov(α_j, α_{j+1} | y)` the bridge `predict` needs:
///
///   mean(u) = C·α̂_p + d,   Cov(u) = C V_p Cᵀ + Σ,   Cov(u, α_p) = C V_p.
///
/// This is exact Gaussian conditioning — no diffuse RTS recursion, no
/// sign-convention-laden `r/N` adjoint. At `order = 2` (one leading node) it is
/// algebraically the existing single-node closure.
fn leading_block_smooth(
    sm_state: &mut [Vec2],
    sm_cov: &mut [Mat2],
    gains: &mut [Mat2],
    nodes: &[PooledNode],
    q: f64,
    order: usize,
) -> Result<(), String> {
    let nb = order - 1; // leading nodes 0..nb-1 (the partially-diffuse ones)
    let pin = order - 1; // first proper smoothed node (conditioning anchor)
    let d = nb * order; // joint dimension of the leading block
    let mut lambda = vec![vec![0.0_f64; d]; d];
    let mut b_const = vec![0.0_f64; d];
    let mut bmat = vec![vec![0.0_f64; order]; d]; // coupling to the pinned node

    // Markov increments t = 0..order-2, each connecting node t and node t+1.
    for t in 0..order - 1 {
        let delta = nodes[t + 1].x - nodes[t].x;
        let f = transition(delta, order);
        let qn = process_noise(delta, q, order);
        let a = mat_inv(&qn, order, "leading-block increment noise")?; // (qQ)⁻¹ (symmetric)
        let ft = mat_t(&f, order);
        let fta = mat_mul(&ft, &a, order); // F'A
        let ftaf = mat_mul(&fta, &f, order); // F'A F
        let af = mat_mul(&a, &f, order); // A F = (F'A)'
        // Node t diagonal block (node t is always in the block): += F'A F.
        for i in 0..order {
            for j in 0..order {
                lambda[t * order + i][t * order + j] += ftaf[i][j];
            }
        }
        if t + 1 <= nb - 1 {
            // Both nodes are in the block: fill node t+1's diagonal and the
            // symmetric cross blocks.
            for i in 0..order {
                for j in 0..order {
                    lambda[(t + 1) * order + i][(t + 1) * order + j] += a[i][j];
                    lambda[t * order + i][(t + 1) * order + j] -= fta[i][j];
                    lambda[(t + 1) * order + i][t * order + j] -= af[i][j];
                }
            }
        } else {
            // t+1 is the pinned node: it enters the conditional only linearly,
            // through B (its coupling into node t's score is F'A·α_pin).
            for i in 0..order {
                for j in 0..order {
                    bmat[t * order + i][j] += fta[i][j];
                }
            }
        }
    }
    // Leading observations: y_t informs the f-component (local index 0) of node t.
    for t in 0..nb {
        let w = nodes[t].w;
        lambda[t * order][t * order] += w;
        b_const[t * order] += w * nodes[t].y;
    }

    // Conditional covariance Σ = Λ⁻¹, intercept d = Σ·b_const, coupling C = Σ·B.
    let sigma = dense_spd_inverse(&lambda, "leading-block precision")?;
    let dvec: Vec<f64> = (0..d)
        .map(|i| (0..d).map(|k| sigma[i][k] * b_const[k]).sum())
        .collect();
    let cmat: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..order)
                .map(|j| (0..d).map(|k| sigma[i][k] * bmat[k][j]).sum())
                .collect()
        })
        .collect();

    // Pinned smoothed moments (from the ordinary RTS pass).
    let ahat_p = sm_state[pin];
    let vp = sm_cov[pin];
    // cvp = C·V_p  (= Cov(u, α_pin)), D×order.
    let cvp: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..order)
                .map(|j| (0..order).map(|k| cmat[i][k] * vp[k][j]).sum())
                .collect()
        })
        .collect();
    // mean(u) = C·α̂_p + d.
    let mean_u: Vec<f64> = (0..d)
        .map(|i| (0..order).map(|j| cmat[i][j] * ahat_p[j]).sum::<f64>() + dvec[i])
        .collect();
    // Cov(u) = cvp·Cᵀ + Σ.
    let cov_u: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..d)
                .map(|k| (0..order).map(|j| cvp[i][j] * cmat[k][j]).sum::<f64>() + sigma[i][k])
                .collect()
        })
        .collect();

    // Scatter the smoothed leading states and covariances.
    for j in 0..nb {
        for i in 0..order {
            sm_state[j][i] = mean_u[j * order + i];
        }
        let mut cov = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
        for i in 0..order {
            for k in 0..order {
                cov[i][k] = cov_u[j * order + i][j * order + k];
            }
        }
        symmetrize(&mut cov, order);
        sm_cov[j] = cov;
    }
    // Lag-one bridge gains for the leading intervals [j, j+1], j = 0..order-2.
    // gain_j = Cov(α_j, α_{j+1} | y) · Cov(α_{j+1} | y)⁻¹, so that the bridge's
    // `gain_j · P^s_{j+1}` reproduces the exact lag-one smoothed cross-cov.
    for j in 0..nb {
        let mut cross = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
        if j + 1 <= nb - 1 {
            // Both in the block: read the (j, j+1) sub-block of Cov(u).
            for i in 0..order {
                for k in 0..order {
                    cross[i][k] = cov_u[j * order + i][(j + 1) * order + k];
                }
            }
        } else {
            // j+1 is the pinned node: read node j's rows of Cov(u, α_pin) = cvp.
            for i in 0..order {
                for k in 0..order {
                    cross[i][k] = cvp[j * order + i][k];
                }
            }
        }
        let denom_inv = mat_inv(&sm_cov[j + 1], order, "leading-block gain denominator")?;
        gains[j] = mat_mul(&cross, &denom_inv, order);
    }
    Ok(())
}

/// Fit at a FIXED `log λ` and order `m ∈ {1, 2, 3}`, σ² either supplied or
/// profiled.
pub fn fit_spline_scan_at(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    log_lambda: f64,
    sigma2: Option<f64>,
    order: usize,
) -> Result<SplineScanFit, String> {
    if order == 0 || order > MAX_ORDER {
        return Err(format!(
            "spline scan: order must be in 1..={MAX_ORDER}, got {order}"
        ));
    }
    let (nodes, ssr_within, n_obs, response_origin) = pool_nodes(x, y, w, order)?;
    let q = gam_problem::checked_exp_log_strength(-log_lambda)
        .map_err(|error| format!("spline scan inverse log strength: {error}"))?;
    let pass = run_filter::<true>(&nodes, q, order)?;
    let n = nodes.len();
    let dof = (n_obs - order) as f64;
    let sigma2 = match sigma2 {
        Some(s) => {
            if !(s.is_finite() && s > 0.0) {
                return Err(format!("spline scan: invalid sigma2 {s}"));
            }
            s
        }
        None => (pass.sum_v2_over_f + ssr_within) / dof,
    };
    // Full diffuse restricted log-likelihood at this (λ, σ²), up to λ- and
    // σ-free additive constants: −½[Σ log F̃ + dof·ln σ² + RSS/σ²]. At the
    // profiled σ̂² the quadratic term collapses to the λ-free constant `dof`,
    // matching `concentrated_criterion` up to that constant.
    let rss = pass.sum_v2_over_f + ssr_within;
    let restricted_loglik = -0.5 * (pass.sum_log_f + dof * sigma2.ln() + rss / sigma2);

    // ── Smoother: ordinary RTS for the proper nodes (t ≥ order−1) plus an
    // exact diffuse conditioning of the `order−1` leading nodes. ──
    // The filtered distribution is fully proper from node order−1 onward (the
    // diffuse rank, = order, is consumed by node order−1), so ordinary RTS is
    // valid for t ≥ order−1. The first order−1 nodes are partially diffuse —
    // their filtered covariance still carries unresolved diffuse mass and the
    // RTS predicted-covariance inverse is singular there — and are recovered
    // exactly, jointly, by `leading_block_smooth` (conditioning the whole
    // leading block on the first proper smoothed node). For order = 1 there is
    // no leading node and RTS covers every node down to t = 0.
    let mut sm_state = vec![[0.0_f64; MAX_ORDER]; n];
    let mut sm_cov = vec![[[0.0_f64; MAX_ORDER]; MAX_ORDER]; n];
    let mut gains = vec![[[0.0_f64; MAX_ORDER]; MAX_ORDER]; n];
    sm_state[n - 1] = pass.steps[n - 1].a_filt;
    sm_cov[n - 1] = pass.steps[n - 1].p_filt;
    for t in (order - 1..n - 1).rev() {
        let p_next_pred = &pass.steps[t + 1].p_pred;
        let delta = nodes[t + 1].x - nodes[t].x;
        let f_t = transition(delta, order);
        let p_inv = mat_inv(p_next_pred, order, "RTS predicted covariance")?;
        let g = mat_mul(
            &mat_mul(&pass.steps[t].p_filt, &mat_t(&f_t, order), order),
            &p_inv,
            order,
        );
        let mut dm: Vec2 = [0.0; MAX_ORDER];
        for i in 0..order {
            dm[i] = sm_state[t + 1][i] - pass.steps[t + 1].a_pred[i];
        }
        let corr = mat_vec(&g, &dm, order);
        for i in 0..order {
            sm_state[t][i] = pass.steps[t].a_filt[i] + corr[i];
        }
        let dp = mat_sub(&sm_cov[t + 1], p_next_pred, order);
        let mut cov = mat_add(
            &pass.steps[t].p_filt,
            &mat_mul(&mat_mul(&g, &dp, order), &mat_t(&g, order), order),
            order,
        );
        symmetrize(&mut cov, order);
        sm_cov[t] = cov;
        gains[t] = g;
    }
    // The order−1 partially-diffuse leading nodes by exact joint conditioning
    // (the multi-node generalization of the m=2 reverse-Markov closure).
    if order >= 2 {
        leading_block_smooth(&mut sm_state, &mut sm_cov, &mut gains, &nodes, q, order)?;
    }

    let knots: Vec<f64> = nodes.iter().map(|n| n.x).collect();
    let centered_mean: Vec<f64> = sm_state.iter().map(|s| s[0]).collect();
    // Weighted DATA residual sum of squares at the smoothed mean. Tied rows
    // pool exactly: Σᵢ wᵢ(yᵢ − f̂ₖ)² = Σᵢ wᵢ(yᵢ − ȳₖ)² + Σₖ Wₖ(ȳₖ − f̂ₖ)²
    // (within-tie scatter plus pooled-node misfit), so the raw rows the scan
    // does not retain are not needed.
    let data_sse = ssr_within
        + nodes
            .iter()
            .zip(centered_mean.iter())
            .map(|(node, &fhat)| {
                let r = node.y - fhat;
                node.w * r * r
            })
            .sum::<f64>();
    // Evaluate the ordinary weighted Gaussian likelihood while the original
    // row weights still exist. Pooled node weights preserve Σw but not Σln(w):
    // for tied rows, ln(w₁ + w₂) != ln(w₁) + ln(w₂), so this normalizer cannot
    // be reconstructed from the persisted state after pooling.
    let sum_log_weights = w.iter().map(|weight| weight.ln()).sum::<f64>();
    let log_likelihood = -0.5
        * (data_sse / sigma2
            + n_obs as f64 * (std::f64::consts::TAU.ln() + sigma2.ln())
            - sum_log_weights);
    if !log_likelihood.is_finite() {
        return Err(format!(
            "spline scan: weighted Gaussian log-likelihood is non-finite \
             (data_sse={data_sse}, sigma2={sigma2}, n={n_obs}, \
             sum_log_weights={sum_log_weights})"
        ));
    }
    // Cross back from the centered numerical chart only after every
    // translation-invariant quantity has been formed. Derivative states and
    // covariances are unchanged by a constant shift.
    for (index, state) in sm_state.iter_mut().enumerate() {
        state[0] += response_origin;
        if !state[0].is_finite() {
            return Err(format!(
                "spline scan: restored fitted response is non-finite at node {index}"
            ));
        }
    }
    let mean: Vec<f64> = sm_state.iter().map(|state| state[0]).collect();
    // f′ lives at state index 1 — present for order ≥ 2 only; the m = 1 latent
    // process (Brownian motion) has no derivative state to expose.
    let deriv: Option<Vec<f64>> =
        (order >= 2).then(|| sm_state.iter().map(|state| state[1]).collect());
    let var: Vec<f64> = sm_cov.iter().map(|p| p[0][0] * sigma2).collect();
    Ok(SplineScanFit {
        order,
        knots,
        mean,
        deriv,
        var,
        log_lambda,
        sigma2,
        restricted_loglik,
        log_likelihood,
        training_sample_size: std::num::NonZeroUsize::new(n_obs)
            .expect("pool_nodes requires at least one training row"),
        data_sse,
        smoothed_state: sm_state,
        smoothed_cov: sm_cov,
        rts_gain: gains,
        q,
        node_weight: nodes.iter().map(|n| n.w).collect(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SplineKktKind {
    LowerBoundary,
    UpperBoundary,
    Stationary { curvature: ClosedInterval },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SplineOptimumProof {
    Kkt {
        bracket: ClosedInterval,
        kind: SplineKktKind,
    },
    /// The producer proved the region's exact maximum indistinguishable from
    /// the representative at the point evaluator's certified comparison
    /// resolution. This is a successful typed optimum, not a failed
    /// stationary-point certificate.
    ResolutionFlat {
        bracket: ClosedInterval,
        max_score_gap: f64,
        score_resolution: f64,
    },
}

/// Preserve the certified optimizer's proof category at the spline consumer
/// seam.
///
/// Boundary and stationary selections require their exact-real KKT proof
/// below. A [`ScoreOptimumLocation::ResolutionFlat`] selection instead carries
/// the producer's successful value-resolution theorem. Requiring a stationary
/// KKT certificate from that category contradicts its contract: its whole
/// purpose is that unresolved stationary structure is immaterial because the
/// maximum's excess over the representative does not exceed comparison
/// resolution.
fn spline_optimum_proof(
    search: &ScoreSearchResult,
) -> Result<SplineOptimumProof, SplineScoreProofError> {
    match search.location {
        ScoreOptimumLocation::LowerBoundary => Ok(SplineOptimumProof::Kkt {
            bracket: ClosedInterval::point(search.lower_boundary.x),
            kind: SplineKktKind::LowerBoundary,
        }),
        ScoreOptimumLocation::UpperBoundary => Ok(SplineOptimumProof::Kkt {
            bracket: ClosedInterval::point(search.upper_boundary.x),
            kind: SplineKktKind::UpperBoundary,
        }),
        ScoreOptimumLocation::Stationary(index) => {
            let stationary = search.stationary_points.get(index).ok_or_else(|| {
                SplineScoreProofError::Search(
                    "optimizer returned an invalid stationary-point index".to_string(),
                )
            })?;
            Ok(SplineOptimumProof::Kkt {
                bracket: stationary.bracket,
                kind: SplineKktKind::Stationary {
                    curvature: stationary.curvature,
                },
            })
        }
        ScoreOptimumLocation::ResolutionFlat(index) => {
            let flat = search.resolution_flat_regions.get(index).ok_or_else(|| {
                SplineScoreProofError::Search(
                    "optimizer returned an invalid resolution-flat index".to_string(),
                )
            })?;
            if !(flat.max_score_gap.is_finite()
                && flat.max_score_gap >= 0.0
                && flat.score_resolution.is_finite()
                && flat.score_resolution >= 0.0
                && flat.max_score_gap <= flat.score_resolution
                && flat.bracket.contains(search.optimum.x)
                && flat.sample.x.to_bits() == search.optimum.x.to_bits())
            {
                return Err(SplineScoreProofError::Search(format!(
                    "optimizer returned an invalid resolution-flat certificate: selected {}, \
                     representative {}, bracket {:?}, maximum score gap {}, score resolution {}",
                    search.optimum.x,
                    flat.sample.x,
                    flat.bracket,
                    flat.max_score_gap,
                    flat.score_resolution
                )));
            }
            Ok(SplineOptimumProof::ResolutionFlat {
                bracket: flat.bracket,
                max_score_gap: flat.max_score_gap,
                score_resolution: flat.score_resolution,
            })
        }
    }
}

fn spline_kkt_holds(
    kind: SplineKktKind,
    final_enclosure: DerivativeEnclosure,
) -> (bool, ClosedInterval) {
    match kind {
        SplineKktKind::LowerBoundary => (
            final_enclosure.derivative.hi <= 0.0,
            final_enclosure.curvature,
        ),
        SplineKktKind::UpperBoundary => (
            final_enclosure.derivative.lo >= 0.0,
            final_enclosure.curvature,
        ),
        SplineKktKind::Stationary { curvature } => (
            // Recompute the final bracket's derivative containment, which
            // depends on its endpoint certificates. Preserve the producer's
            // strict curvature enclosure: it proved this root unique on a
            // parent cell and therefore remains valid on every contracted
            // subset, even if a fresh tiny-cell formula loses the sign to
            // cancellation.
            final_enclosure.derivative.contains_zero() && curvature.hi < 0.0,
            curvature,
        ),
    }
}

/// Fit with `log λ` selected by the concentrated diffuse REML criterion.
/// Every stationary interval in the bounded, scale-equivariant log-λ domain
/// is isolated using analytic derivatives and rigorous interval bounds; the
/// two boundary/null-recovery candidates are evaluated exactly.
pub fn fit_spline_scan(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    order: usize,
) -> Result<SplineScanFit, SplineScoreProofError> {
    if order == 0 || order > MAX_ORDER {
        return Err(SplineScoreProofError::InvalidInput(format!(
            "spline scan: order must be in 1..={MAX_ORDER}, got {order}"
        )));
    }
    let (nodes, ssr_within, n_obs, _response_origin) = pool_nodes(x, y, w, order)?;
    // Covariate-rescaling equivariance (#1214). The order-`m` IWP process noise
    // is `Q(δ) ∝ q · δ^{2m−1}`, so under an affine covariate rescale `x → a·x`
    // (all abscissa gaps `δ → a·δ`) the posterior `f(x)` is *exactly* invariant
    // iff the smoothing parameter co-transforms as `q → q / a^{2m−1}`, i.e.
    // `log λ → log λ + (2m−1)·log a` (λ = 1/q). The whole smoother — criterion,
    // fit, and the Gaussian-bridge `predict` — runs self-consistently in the raw
    // covariate units, so the *only* place covariate scale leaks in is this
    // outer `log λ` search: a fixed absolute bracket `[LOG_LAMBDA_LO,
    // LOG_LAMBDA_HI]` does not track the data span, so at small/large covariate
    // scale the equivariant optimum rails out of the bracket and the fit drifts.
    // Anchor the bracket to the data's own length scale: search `log λ` around
    // `(2m−1)·log L` where `L` is the abscissa span (which scales linearly with
    // the covariate), so the search is performed in scale-free units and the
    // selected `q · L^{2m−1}` — hence the posterior `f(x)` — is invariant.
    let first_x = nodes
        .first()
        .ok_or_else(|| {
            SplineScoreProofError::InvalidInput(
                "spline scan: pooled data unexpectedly contain no nodes".to_string(),
            )
        })?
        .x;
    let last_x = nodes
        .last()
        .ok_or_else(|| {
            SplineScoreProofError::InvalidInput(
                "spline scan: pooled data unexpectedly contain no nodes".to_string(),
            )
        })?
        .x;
    let span = last_x - first_x;
    if !(span.is_finite() && span > 0.0) {
        return Err(SplineScoreProofError::InvalidInput(format!(
            "spline scan: pooled covariate span must be finite and positive, got {span}"
        )));
    }
    let log_span = gam_math::score_opt::certified_ln_positive(span).ok_or(
        SplineScoreProofError::InvalidArithmetic {
            context: "covariate-span logarithm",
        },
    )?;
    let log_span_representative = log_span.lo + 0.5 * (log_span.hi - log_span.lo);
    let scale_shift = (2 * order - 1) as f64 * log_span_representative;
    let lo_anchor = LOG_LAMBDA_LO + scale_shift;
    let hi_anchor = LOG_LAMBDA_HI + scale_shift;
    let n_nodes = nodes.len();
    let endpoint_certificates = RefCell::new(HashMap::<u64, CertifiedCriterionJet>::new());
    let search = maximize_score_1d(
        lo_anchor,
        hi_anchor,
        f64::EPSILON.sqrt(),
        |ll| {
            let certificate =
                certified_concentrated_criterion_jet(&nodes, ssr_within, n_obs, ll, order)?;
            endpoint_certificates
                .borrow_mut()
                .insert(ll.to_bits(), certificate);
            Ok(certificate.jet)
        },
        |left, right| {
            let certificates = endpoint_certificates.borrow();
            let left_certificate = certificates
                .get(&left.x.to_bits())
                .copied()
                .ok_or(SplineScoreProofError::MissingEndpointCertificate { log_lambda: left.x })?;
            let right_certificate = certificates.get(&right.x.to_bits()).copied().ok_or(
                SplineScoreProofError::MissingEndpointCertificate {
                    log_lambda: right.x,
                },
            )?;
            concentrated_criterion_enclosure(
                n_nodes,
                n_obs,
                left,
                right,
                left_certificate,
                right_certificate,
                order,
            )
        },
    )
    .map_err(|error| match error {
        gam_math::score_opt::ScoreSearchError::PointEvaluation { source, .. }
        | gam_math::score_opt::ScoreSearchError::EnclosureEvaluation { source, .. } => source,
        other => SplineScoreProofError::Search(other.to_string()),
    })?;
    if search.value_certificate.maximum_excess > search.value_certificate.comparison_resolution {
        return Err(SplineScoreProofError::GlobalValueOrderingUnresolved {
            maximum_excess: search.value_certificate.maximum_excess,
            comparison_resolution: search.value_certificate.comparison_resolution,
        });
    }
    match spline_optimum_proof(&search)? {
        SplineOptimumProof::Kkt {
            bracket: kkt_bracket,
            kind: kkt_kind,
        } => {
            let kkt_enclosure = {
                let certificates = endpoint_certificates.borrow();
                let left_certificate = certificates.get(&kkt_bracket.lo.to_bits()).copied().ok_or(
                    SplineScoreProofError::MissingEndpointCertificate {
                        log_lambda: kkt_bracket.lo,
                    },
                )?;
                let right_certificate = certificates
                    .get(&kkt_bracket.hi.to_bits())
                    .copied()
                    .ok_or(SplineScoreProofError::MissingEndpointCertificate {
                        log_lambda: kkt_bracket.hi,
                    })?;
                let sample = |log_lambda: f64, certificate: CertifiedCriterionJet| ScoreSample {
                    x: log_lambda,
                    value: certificate.jet.value,
                    derivative: certificate.jet.derivative,
                    curvature: certificate.jet.curvature,
                    third: certificate.jet.third,
                };
                concentrated_criterion_enclosure(
                    n_nodes,
                    n_obs,
                    sample(kkt_bracket.lo, left_certificate),
                    sample(kkt_bracket.hi, right_certificate),
                    left_certificate,
                    right_certificate,
                    order,
                )?
            };
            let (kkt_holds, kkt_curvature) = spline_kkt_holds(kkt_kind, kkt_enclosure);
            if !kkt_holds {
                return Err(SplineScoreProofError::OptimumKktUncertified {
                    location: search.location,
                    bracket: kkt_bracket,
                    derivative: kkt_enclosure.derivative,
                    curvature: kkt_curvature,
                });
            }
        }
        SplineOptimumProof::ResolutionFlat {
            bracket,
            max_score_gap,
            score_resolution,
        } => {
            log::debug!(
                "spline scan: accepting certified resolution-flat REML optimum on \
                 {bracket:?}; maximum score excess {max_score_gap:e} <= comparison \
                 resolution {score_resolution:e}"
            );
        }
    }
    // The fixed-λ fitter below consumes the historical scalar recurrence.
    // Before crossing that seam, independently re-evaluate the selected point
    // and require every scalar component to lie in the directed ball that won
    // the search. This is one O(n) pass per completed fit, not per search cell.
    let selected_certificate = endpoint_certificates
        .borrow()
        .get(&search.optimum.x.to_bits())
        .copied()
        .ok_or_else(|| {
            SplineScoreProofError::Search(format!(
                "spline scan: selected log lambda {} has no cached score certificate",
                search.optimum.x
            ))
        })?;
    let independent =
        concentrated_criterion_jet(&nodes, ssr_within, n_obs, search.optimum.x, order)
            .map_err(SplineScoreProofError::Computation)?;
    for (name, ball, scalar) in [
        ("value", selected_certificate.value, independent.0),
        ("derivative", selected_certificate.derivative, independent.1),
        ("curvature", selected_certificate.curvature, independent.2),
        ("third", selected_certificate.third, independent.3),
    ] {
        if !ball.interval().contains(scalar) {
            return Err(SplineScoreProofError::Computation(format!(
                "spline scan: selected {name} scalar {scalar} escapes its directed score ball {:?}",
                ball.interval()
            )));
        }
    }
    fit_spline_scan_at(x, y, w, search.optimum.x, None, order)
        .map_err(SplineScoreProofError::Computation)
}

/// Lossless serializable snapshot of a [`SplineScanFit`] (#1034).
///
/// Carries exactly the smoother state the Gaussian-bridge `predict` replays:
/// pooled knots, smoothed `(f, f′, …, f^{(m−1)})` states (`m` per knot),
/// smoothed state covariances (unit-σ² scale, symmetric — stored as the
/// upper triangle row-major, `m(m+1)/2` per knot), RTS backward gains (full
/// `m×m` row-major — gains are NOT symmetric), pooled node weights, and the
/// required fit scalars. `q = e^{−log λ}` and the public `mean`/`deriv`/`var`
/// views are derived on restore rather than stored, so a snapshot cannot go
/// internally inconsistent. Every field is required: a snapshot that predates
/// the current statistical contract must be regenerated rather than guessed.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SplineScanState {
    /// Smoothing-spline order `m ∈ {1, 2, 3}`.
    pub order: usize,
    pub knots: Vec<f64>,
    /// Smoothed `(f, f′, …, f^{(m−1)})` per knot, row-major (`m` per knot).
    pub state: Vec<f64>,
    /// Smoothed covariance per knot at unit-σ² scale, upper triangle row-major
    /// (`m(m+1)/2` per knot): `[c00, c01, …, c0,m−1, c11, …, c_{m−1,m−1}]`.
    pub cov: Vec<f64>,
    /// RTS backward gain per knot, full `m×m` row-major (`m²` per knot); the
    /// last knot's gain is structurally unused and stored as written.
    pub gain: Vec<f64>,
    /// Pooled (tied-abscissa summed) observation weight per knot.
    pub node_weight: Vec<f64>,
    pub log_lambda: f64,
    pub sigma2: f64,
    pub restricted_loglik: f64,
    /// Ordinary fully normalized weighted Gaussian log-likelihood. Required on
    /// the wire because the raw per-row weights needed for its normalizer are
    /// deliberately not retained after tied abscissae are pooled.
    pub log_likelihood: f64,
    /// Original training row count. Required on the wire.
    pub training_sample_size: std::num::NonZeroU64,
    /// Weighted data residual sum of squares `Σ wᵢ (yᵢ − f̂(xᵢ))²` at the
    /// smoothed mean — the Gaussian deviance. Stored because it cannot be
    /// recovered from the profiled σ² (whose quadratic also carries
    /// process/roughness energy) and the raw rows are not retained.
    pub data_sse: f64,
}

impl SplineScanFit {
    /// Snapshot the full smoother state for persistence (#1034).
    pub fn to_state(&self) -> SplineScanState {
        let order = self.order;
        let tri = order * (order + 1) / 2;
        let nk = self.knots.len();
        let mut state = Vec::with_capacity(order * nk);
        for s in &self.smoothed_state {
            state.extend_from_slice(&s[..order]);
        }
        let mut cov = Vec::with_capacity(tri * nk);
        for c in &self.smoothed_cov {
            for i in 0..order {
                for j in i..order {
                    cov.push(c[i][j]);
                }
            }
        }
        let mut gain = Vec::with_capacity(order * order * nk);
        for g in &self.rts_gain {
            for i in 0..order {
                for j in 0..order {
                    gain.push(g[i][j]);
                }
            }
        }
        SplineScanState {
            order: self.order,
            knots: self.knots.clone(),
            state,
            cov,
            gain,
            node_weight: self.node_weight.clone(),
            log_lambda: self.log_lambda,
            sigma2: self.sigma2,
            restricted_loglik: self.restricted_loglik,
            log_likelihood: self.log_likelihood,
            training_sample_size: std::num::NonZeroU64::new(
                u64::try_from(self.training_sample_size.get())
                    .expect("SplineScanFit row count exceeds the persistence format"),
            )
            .expect("SplineScanFit construction requires training rows"),
            data_sse: self.data_sse,
        }
    }

    /// Rebuild the exact in-memory fit from a persisted snapshot (#1034).
    ///
    /// Validates shape, finiteness, strict knot ordering, positive weights and
    /// σ², so a corrupt payload fails loudly here instead of inside a later
    /// `predict`. The restored fit replays the Gaussian bridge bit-for-bit:
    /// every field `predict`/`edf`/`deriv_at_knot` reads is either stored
    /// verbatim or derived by the same expressions the fitter uses.
    pub fn from_state(state: &SplineScanState) -> Result<Self, String> {
        let order = state.order;
        if order == 0 || order > MAX_ORDER {
            return Err(format!(
                "spline scan state: order must be in 1..={MAX_ORDER}, got {order}"
            ));
        }
        let m = state.knots.len();
        if m < order + 1 {
            return Err(format!(
                "spline scan state: order {order} needs at least {} knots, got {m}",
                order + 1
            ));
        }
        let tri = order * (order + 1) / 2;
        if state.state.len() != order * m
            || state.cov.len() != tri * m
            || state.gain.len() != order * order * m
            || state.node_weight.len() != m
        {
            return Err(format!(
                "spline scan state: inconsistent lengths (order={order}, m={m}, state={}, cov={}, gain={}, weights={})",
                state.state.len(),
                state.cov.len(),
                state.gain.len(),
                state.node_weight.len()
            ));
        }
        let all = state
            .state
            .iter()
            .chain(&state.cov)
            .chain(&state.gain)
            .chain(&state.knots)
            .chain(&state.node_weight);
        for (i, v) in all.enumerate() {
            if !v.is_finite() {
                return Err(format!("spline scan state: non-finite entry at {i}"));
            }
        }
        gam_problem::validate_log_strength(state.log_lambda)
            .map_err(|error| format!("spline scan state: {error}"))?;
        if !(state.restricted_loglik.is_finite()
            && state.log_likelihood.is_finite()
            && state.sigma2.is_finite()
            && state.sigma2 > 0.0)
        {
            return Err(format!(
                "spline scan state: invalid scalars (log_lambda={}, sigma2={}, \
                 restricted_loglik={}, log_likelihood={})",
                state.log_lambda, state.sigma2, state.restricted_loglik, state.log_likelihood
            ));
        }
        if !(state.data_sse.is_finite() && state.data_sse >= 0.0) {
            return Err(format!(
                "spline scan state: invalid data_sse {}",
                state.data_sse
            ));
        }
        if state.knots.windows(2).any(|kk| !(kk[0] < kk[1])) {
            return Err("spline scan state: knots must be strictly increasing".to_string());
        }
        if state.node_weight.iter().any(|&w| w <= 0.0) {
            return Err("spline scan state: node weights must be positive".to_string());
        }
        let smoothed_state: Vec<Vec2> = state
            .state
            .chunks_exact(order)
            .map(|s| {
                let mut v = [0.0_f64; MAX_ORDER];
                v[..order].copy_from_slice(s);
                v
            })
            .collect();
        let smoothed_cov: Vec<Mat2> = state
            .cov
            .chunks_exact(tri)
            .map(|c| {
                let mut mm = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
                let mut idx = 0;
                for i in 0..order {
                    for j in i..order {
                        mm[i][j] = c[idx];
                        mm[j][i] = c[idx];
                        idx += 1;
                    }
                }
                mm
            })
            .collect();
        let rts_gain: Vec<Mat2> = state
            .gain
            .chunks_exact(order * order)
            .map(|g| {
                let mut mm = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
                for i in 0..order {
                    for j in 0..order {
                        mm[i][j] = g[i * order + j];
                    }
                }
                mm
            })
            .collect();
        let sigma2 = state.sigma2;
        let training_sample_size =
            usize::try_from(state.training_sample_size.get()).map_err(|_| {
                format!(
                    "spline scan state: training_sample_size {} exceeds this platform's usize",
                    state.training_sample_size
                )
            })?;
        Ok(Self {
            order,
            knots: state.knots.clone(),
            mean: smoothed_state.iter().map(|s| s[0]).collect(),
            deriv: (order >= 2).then(|| smoothed_state.iter().map(|s| s[1]).collect()),
            var: smoothed_cov.iter().map(|c| c[0][0] * sigma2).collect(),
            log_lambda: state.log_lambda,
            sigma2,
            restricted_loglik: state.restricted_loglik,
            log_likelihood: state.log_likelihood,
            training_sample_size: std::num::NonZeroUsize::new(training_sample_size)
                .expect("nonzero wire count remains nonzero after conversion"),
            data_sse: state.data_sse,
            smoothed_state,
            smoothed_cov,
            rts_gain,
            q: gam_problem::checked_exp_log_strength(-state.log_lambda)
                .map_err(|error| format!("spline scan inverse log strength: {error}"))?,
            node_weight: state.node_weight.clone(),
        })
    }

    /// Exact posterior `(mean, variance)` of `f` at an arbitrary abscissa.
    ///
    /// Interior points use the Gaussian bridge conditional on the two flanking
    /// smoothed states with the exact lag-one smoothed cross-covariance
    /// `Cov(α_t, α_{t+1} | y) = G_t · P^s_{t+1}`; exterior points extrapolate
    /// from the boundary state (linear mean, cubically growing variance).
    pub fn predict(&self, x_new: f64) -> Result<(f64, f64), String> {
        if !x_new.is_finite() {
            return Err("spline scan: non-finite prediction abscissa".to_string());
        }
        let n = self.knots.len();
        let order = self.order;
        let first = self.knots[0];
        let last = self.knots[n - 1];
        if x_new <= first {
            let delta = first - x_new;
            // Backward extrapolation through the reverse map α(x) = F⁻¹(α₁ − η).
            let f_t = transition(delta, order);
            let f_inv = mat_inv(&f_t, order, "backward extrapolation transition")?;
            let mean_s = mat_vec(&f_inv, &self.smoothed_state[0], order);
            let qm = process_noise(delta, self.q, order);
            let cov = mat_add(
                &mat_mul(
                    &mat_mul(&f_inv, &self.smoothed_cov[0], order),
                    &mat_t(&f_inv, order),
                    order,
                ),
                &mat_mul(&mat_mul(&f_inv, &qm, order), &mat_t(&f_inv, order), order),
                order,
            );
            return Ok((mean_s[0], cov[0][0] * self.sigma2));
        }
        if x_new >= last {
            let delta = x_new - last;
            let f_t = transition(delta, order);
            let mean_s = mat_vec(&f_t, &self.smoothed_state[n - 1], order);
            let cov = mat_add(
                &mat_mul(
                    &mat_mul(&f_t, &self.smoothed_cov[n - 1], order),
                    &mat_t(&f_t, order),
                    order,
                ),
                &process_noise(delta, self.q, order),
                order,
            );
            return Ok((mean_s[0], cov[0][0] * self.sigma2));
        }
        // Flanking knot interval via binary search.
        let t = match self.knots.binary_search_by(|k| k.total_cmp(&x_new)) {
            Ok(idx) => return Ok((self.mean[idx], self.var[idx])),
            Err(idx) => idx - 1,
        };
        let (xa, xb) = (self.knots[t], self.knots[t + 1]);
        let (d1, d2) = (x_new - xa, xb - x_new);
        let (f1m, f2m) = (transition(d1, order), transition(d2, order));
        let (q1, q2) = (
            process_noise(d1, self.q, order),
            process_noise(d2, self.q, order),
        );
        let q1_inv = mat_inv(&q1, order, "bridge left noise")?;
        let q2_inv = mat_inv(&q2, order, "bridge right noise")?;
        // p(α* | α_t, α_{t+1}) ∝ N(α*; F₁α_t, Q₁)·N(α_{t+1}; F₂α*, Q₂):
        //   Λ = Q₁⁻¹ + F₂ᵀQ₂⁻¹F₂,  mean = Λ⁻¹(Q₁⁻¹F₁ α_t + F₂ᵀQ₂⁻¹ α_{t+1}).
        let lambda = mat_add(
            &q1_inv,
            &mat_mul(&mat_mul(&mat_t(&f2m, order), &q2_inv, order), &f2m, order),
            order,
        );
        let lam_inv = mat_inv(&lambda, order, "bridge precision")?;
        let ca = mat_mul(&lam_inv, &mat_mul(&q1_inv, &f1m, order), order);
        let cb = mat_mul(
            &lam_inv,
            &mat_mul(&mat_t(&f2m, order), &q2_inv, order),
            order,
        );
        let ma = mat_vec(&ca, &self.smoothed_state[t], order);
        let mb = mat_vec(&cb, &self.smoothed_state[t + 1], order);
        let mut mean_s = [0.0_f64; MAX_ORDER];
        for i in 0..order {
            mean_s[i] = ma[i] + mb[i];
        }
        // Push the joint smoothed covariance of (α_t, α_{t+1}) through the
        // affine map: cross term uses Cov(α_t, α_{t+1}|y) = G_t · P^s_{t+1}.
        let cross = mat_mul(&self.rts_gain[t], &self.smoothed_cov[t + 1], order);
        let mut cov = mat_add(
            &mat_add(
                &mat_mul(
                    &mat_mul(&ca, &self.smoothed_cov[t], order),
                    &mat_t(&ca, order),
                    order,
                ),
                &mat_mul(
                    &mat_mul(&cb, &self.smoothed_cov[t + 1], order),
                    &mat_t(&cb, order),
                    order,
                ),
                order,
            ),
            &lam_inv,
            order,
        );
        let cab = mat_mul(&mat_mul(&ca, &cross, order), &mat_t(&cb, order), order);
        cov = mat_add(&cov, &mat_add(&cab, &mat_t(&cab, order), order), order);
        symmetrize(&mut cov, order);
        Ok((mean_s[0], cov[0][0] * self.sigma2))
    }

    /// Exact effective degrees of freedom of the fitted smoother.
    ///
    /// For a Gaussian smoother the influence (hat) matrix is
    /// `S = Cov_post · W / σ²` (posterior mean is linear in `y` with that
    /// exact coefficient matrix), so
    /// `EDF = tr(S) = tr(W · Cov_post) / σ² = Σ_t w_t · Var_smoothed(f_t) / σ²`.
    /// This is the standard Gaussian-process identity — no second smoother
    /// pass and no approximation. Tied abscissae pool exactly: each raw row
    /// `i` in tie-group `k` contributes `∂f̂(x_k)/∂y_i = C̃_kk · w_i` (the
    /// pooled mean `ȳ_k` is precision-weighted), so the raw-row trace
    /// `Σ_i w_i · C̃_{k(i),k(i)}` collapses to `Σ_k W_k · C̃_kk` with the
    /// pooled weights `W_k`. `smoothed_cov` is stored at unit-σ² scale
    /// (`C̃ = Cov_post / σ²`), so the σ² factors cancel exactly.
    pub fn edf(&self) -> f64 {
        self.node_weight
            .iter()
            .zip(self.smoothed_cov.iter())
            .map(|(w, c)| w * c[0][0])
            .sum()
    }

    /// Selected smoothing parameter `λ = e^{log λ}` (#1046).
    pub fn lambda(&self) -> f64 {
        gam_problem::checked_exp_log_strength(self.log_lambda)
            .expect("SplineScanFit construction validates its private log strength")
    }

    pub fn log_lambda(&self) -> f64 {
        self.log_lambda
    }

    /// Number of original training rows / experimental units.
    pub fn training_sample_size(&self) -> usize {
        self.training_sample_size.get()
    }

    /// Gaussian deviance — the weighted DATA residual sum of squares
    /// `Σ wᵢ(yᵢ − f̂ᵢ)²` at the smoothed mean (#1046). This is the stored
    /// `data_sse`, computed against the fitted values at fit time. It is NOT
    /// `σ̂²·(n − order)`: the profiled σ² divides the REML innovations
    /// quadratic, which is data residual energy PLUS process/roughness energy
    /// at the posterior mode (for order 1 on `x = (0,1)`, `y = (0,1)`, unit
    /// weights and λ = 1 the posterior mean is `(1/3, 2/3)`; the data SSE is
    /// 2/9 while `σ̂²·(n − order) = 1/3`, the extra 1/9 being penalty energy).
    pub fn deviance(&self) -> f64 {
        self.data_sse
    }
}

#[cfg(test)]
mod tests {
    /// Seed a covariance zonotope without throwing away exact symmetry.
    ///
    /// The two off-diagonal storage locations denote one real covariance entry.
    /// A shared generator therefore encloses their common error while preserving
    /// that identity; two independent axis generators would immediately forget it
    /// and recreate the componentwise wrapping effect on the first congruence.
    fn covariance_zonotope_from_symmetric_matrix(
        matrix: &BallMat,
        order: usize,
    ) -> Zonotope<COVARIANCE_D1_DIM> {
        let mut state = Zonotope::<COVARIANCE_D1_DIM>::zeroed(order * order);
        for i in 0..order {
            for j in i..order {
                let value = matrix[i][j].value;
                state.center[i * order + j] = value;
                state.center[j * order + i] = value;
                let radius = [
                    (value - matrix[i][j].lo).abs(),
                    (matrix[i][j].hi - value).abs(),
                    (value - matrix[j][i].lo).abs(),
                    (matrix[j][i].hi - value).abs(),
                ]
                .into_iter()
                .fold(0.0_f64, f64::max);
                if radius > 0.0 {
                    let mut generator = [0.0_f64; COVARIANCE_D1_DIM];
                    let radius = next_up_ball(radius);
                    generator[i * order + j] = radius;
                    generator[j * order + i] = radius;
                    state.generators.push(generator);
                }
            }
        }
        state
    }

    /// Compaction must preserve the signed directions that make a contracting
    /// recursion contract.  Fresh roundoff enters as axis generators, so age
    /// based reduction used to fold this old `[1, -1]` direction first and
    /// replace it by the expanding box `[±1] × [±1]`.
    #[test]
    fn zonotope_compaction_retains_correlation_before_axis_roundoff() {
        let mut state = Zonotope::<2>::zeroed(2);
        state.generators.push([1.0, -1.0]);
        for i in 0..ZONOTOPE_GENERATOR_CAP {
            state
                .generators
                .push(if i % 2 == 0 { [0.25, 0.0] } else { [0.0, 0.25] });
        }

        state.compact();

        assert!(state.generators.len() <= ZONOTOPE_GENERATOR_CAP);
        assert!(
            state
                .generators
                .iter()
                .any(|generator| *generator == [1.0, -1.0]),
            "compaction discarded the only signed correlation direction"
        );
    }

    /// Two occurrences of `qQ` contain ONE uncertain `q`, not two independent
    /// interval choices. The distinguished coefficient must therefore add
    /// under the identity and cancel under an opposing signed map. Turning
    /// each occurrence into a fresh axis generator leaves radius `2|g|` in the
    /// cancellation arm and cannot prove the exact identity.
    #[test]
    fn shared_q_process_noise_injections_accumulate_and_cancel_as_one_generator() {
        let q = Ball {
            value: 10.0,
            lo: 9.0,
            hi: 11.0,
        };
        let noise = ball_process_noise_taylor(Ball::exact(2.0), q, 1);
        let g = noise.shared_q[0];
        assert!(g > 0.0);

        let identity = zonotope_identity_map::<COVARIANCE_D1_DIM>(1);
        let mut accumulated = Zonotope::<COVARIANCE_D1_DIM>::zeroed(1);
        assert!(accumulated.apply_with_shared_q(&identity, &noise.constant, &noise.shared_q,));
        assert!(accumulated.apply_with_shared_q(&identity, &noise.constant, &noise.shared_q,));
        assert_eq!(accumulated.shared_q[0], 2.0 * g);

        let mut negative_identity = [[Ball::ZERO; COVARIANCE_D1_DIM]; COVARIANCE_D1_DIM];
        negative_identity[0][0] = Ball::exact(-1.0);
        let mut cancelled = Zonotope::<COVARIANCE_D1_DIM>::zeroed(1);
        assert!(cancelled.apply_with_shared_q(&identity, &noise.constant, &noise.shared_q,));
        assert!(cancelled.apply_with_shared_q(
            &negative_identity,
            &noise.constant,
            &noise.shared_q,
        ));
        assert_eq!(cancelled.shared_q[0], 0.0);

        let old_independent_radius = 2.0 * g.abs();
        assert!(
            ball_radius_about_value(cancelled.coordinate(0)) < old_independent_radius * 1.0e-10,
            "independent qQ axes would retain radius {old_independent_radius:e}, \
             but the shared-q cancellation left {:?}",
            cancelled.coordinate(0),
        );
    }

    #[test]
    fn centred_riccati_zonotope_contains_an_off_centre_covariance_and_noise() {
        let centres = [[4.0, 1.0, 0.3], [1.0, 3.0, 0.2], [0.3, 0.2, 2.0]];
        let radii = [[0.2, 0.1, 0.08], [0.1, 0.2, 0.07], [0.08, 0.07, 0.2]];
        let mut enclosure = [[Ball::ZERO; MAX_ORDER]; MAX_ORDER];
        for i in 0..MAX_ORDER {
            for j in 0..MAX_ORDER {
                enclosure[i][j] = Ball {
                    value: centres[i][j],
                    lo: centres[i][j] - radii[i][j],
                    hi: centres[i][j] + radii[i][j],
                };
            }
        }
        let mut state = covariance_zonotope_from_symmetric_matrix(&enclosure, MAX_ORDER);
        let observation_variance = Ball {
            value: 1.2,
            lo: 1.1,
            hi: 1.3,
        };
        assert!(covariance_zonotope_measurement_update(
            &mut state,
            observation_variance,
            MAX_ORDER,
        ));

        let actual = [[4.1, 0.95, 0.35], [0.95, 3.1, 0.15], [0.35, 0.15, 1.9]];
        let actual_r = 1.25;
        let innovation = actual[0][0] + actual_r;
        for i in 0..MAX_ORDER {
            for j in 0..MAX_ORDER {
                let updated = actual[i][j] - actual[i][0] * actual[0][j] / innovation;
                assert!(
                    state
                        .coordinate(i * MAX_ORDER + j)
                        .interval()
                        .contains(updated),
                    "updated covariance ({i},{j})={updated} escaped {:?}",
                    state.coordinate(i * MAX_ORDER + j).interval()
                );
            }
        }
    }

    /// Diagnostic reproduction of the #2300 weighted-scan non-termination:
    /// the exact acceptance DGP (n=180, step weights 1/9), with the SAME
    /// certified search `fit_spline_scan` runs — but through a counting
    /// wrapper that bails out with the evaluation count and the stuck
    /// abscissa once the search exceeds a budget no terminating search on a
    /// 36-wide bracket can legitimately need. A pass proves termination in
    /// bounded work; the panic message is the diagnosis.
    #[test]
    fn weighted_scan_dgp_2300_search_terminates_in_bounded_evaluations() {
        // Deterministic stand-in for the acceptance DGP (xorshift Box-Muller;
        // the hang class is structural, not noise-realization-specific). Shared
        // with the `d3` enclosure diagnostic so the two cannot drift apart.
        let (x, y, w) = dgp_2300();
        // Every smoothing order, not just the cubic: the order-3 (quintic)
        // search has a deeper λ→∞ tail walk (scale shift (2m−1)·log L) and a
        // larger residual-d.f. Lipschitz constant, and was the remaining
        // effective hang after the order-2 fix (#2300 — the degree-5
        // observation-interval node timed out at 1500s). Endpoint-pair V‴
        // interpolation certifies its tail at fourth-order rate, so a uniform
        // budget far below the pre-fix eval counts must hold at all orders.
        //
        // The three orders are mathematically independent. Run them as three
        // scoped, single-core lanes so this regression's wall time is the
        // maximum order cost instead of their sum; three workers are negligible
        // on the remote validation nodes and avoid turning a performance test
        // into its own serial bottleneck.
        std::thread::scope(|scope| {
            for order in 1..=MAX_ORDER {
                let (x, y, w) = (&x, &y, &w);
                scope.spawn(move || {
                    let (nodes, ssr_within, n_obs, _response_origin) =
                        pool_nodes(x, y, w, order).expect("pool");
                    let span = nodes.last().unwrap().x - nodes.first().unwrap().x;
                    let scale_shift = (2 * order - 1) as f64 * span.ln();
                    let lo = LOG_LAMBDA_LO + scale_shift;
                    let hi = LOG_LAMBDA_HI + scale_shift;

                    let n_nodes = nodes.len();
                    let evals = std::cell::Cell::new(0u64);
                    let last_x = std::cell::Cell::new(f64::NAN);
                    let endpoint_certificates =
                        RefCell::new(HashMap::<u64, CertifiedCriterionJet>::new());
                    let budget = 4_096u64;
                    let result = gam_math::score_opt::maximize_score_1d(
                        lo,
                        hi,
                        f64::EPSILON.sqrt(),
                        |ll| {
                            let count = evals.get() + 1;
                            evals.set(count);
                            last_x.set(ll);
                            assert!(
                                count <= budget,
                                "order-{order} certified scan search exceeded {budget} criterion \
                                 evaluations (last log-lambda sample {ll:.9}; bracket \
                                 [{lo:.3}, {hi:.3}]) — non-terminating subdivision reproduced"
                            );
                            let certificate = certified_concentrated_criterion_jet(
                                &nodes, ssr_within, n_obs, ll, order,
                            )?;
                            endpoint_certificates
                                .borrow_mut()
                                .insert(ll.to_bits(), certificate);
                            Ok(certificate.jet)
                        },
                        |a, b| {
                            let certificates = endpoint_certificates.borrow();
                            let left = certificates.get(&a.x.to_bits()).copied().ok_or(
                                SplineScoreProofError::MissingEndpointCertificate {
                                    log_lambda: a.x,
                                },
                            )?;
                            let right = certificates.get(&b.x.to_bits()).copied().ok_or(
                                SplineScoreProofError::MissingEndpointCertificate {
                                    log_lambda: b.x,
                                },
                            )?;
                            concentrated_criterion_enclosure(
                                n_nodes, n_obs, a, b, left, right, order,
                            )
                        },
                    );
                    match result {
                        Ok(search) => assert!(
                            search.optimum.x.is_finite(),
                            "order-{order} search must return a finite optimum"
                        ),
                        Err(error) => panic!(
                            "order-{order} weighted scan search failed after {} evaluations \
                             (last x {:.9}): {error:?}",
                            evals.get(),
                            last_x.get()
                        ),
                    }
                });
            }
        });
    }

    /// The #2300 weighted-scan DGP, as its own function so the certified-search
    /// test and the `d3` enclosure diagnostics below read the SAME data rather
    /// than two copies that can drift apart.
    fn dgp_2300() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 180usize;
        let mut state: u64 = 0x2300_2300_2300_2300;
        let mut next_unit = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        for i in 0..n {
            let xi = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
            let wi: f64 = if xi < 0.0 { 1.0 } else { 9.0 };
            let u1 = next_unit().max(1e-12);
            let u2 = next_unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x.push(xi);
            w.push(wi);
            y.push(0.4 + (1.3 * xi).sin() + (0.45 / wi.sqrt()) * z);
        }
        (x, y, w)
    }

    /// The certified derivative ladder reaches exact endpoint jets throughout
    /// the search domain that exposed #2614.
    ///
    /// This fixture used to refuse at smoothing order 3 throughout
    /// `-20 <= rho <= -10` when its covariance-derivative zonotope overflowed.
    /// Merely accepting a global analytic fallback there would be sound but
    /// would reintroduce the loose cells that exhausted the subdivision budget.
    /// The repaired centred Riccati/shared-`q` representation must instead
    /// preserve enough dependence for BOTH curvature and third derivative to
    /// come from their endpoint jets at every measured point.
    #[test]
    fn certified_ladder_reaches_endpoint_jets_across_the_search_domain() {
        let (x, y, w) = dgp_2300();
        let visited = [
            -24.0_f64,
            -20.0,
            -18.0,
            -16.6135,
            // The log-lambda the #2300 certified search refuses at, order 2.
            -13.841116916640328,
            -10.0,
            -6.0,
            0.0,
            6.0,
        ];
        for order in 1..=MAX_ORDER {
            let (nodes, within, n_obs, _response_origin) =
                pool_nodes(&x, &y, &w, order).expect("pool");
            for &log_lambda in &visited {
                let certificate = certified_concentrated_criterion_jet(
                    &nodes, within, n_obs, log_lambda, order,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "order {order}, rho {log_lambda}: repaired certified ladder refused: \
                         {error:?}"
                    )
                });
                assert_eq!(
                    certificate.curvature_source,
                    BoundSource::EndpointJet,
                    "order {order}, rho {log_lambda}: curvature lost its exact endpoint anchor"
                );
                assert_eq!(
                    certificate.third_source,
                    BoundSource::EndpointJet,
                    "order {order}, rho {log_lambda}: third derivative lost its exact endpoint anchor"
                );
            }
        }
    }

    /// The certified criterion jet stays inside the range the Gaussian model
    /// gives it, on the fixture where it did not (#2614).
    ///
    /// `V′ = −½(Σ log F̃)′ − ½·ν·(Σ v²/F̃)′/rss`, and the exact accumulator ranges
    /// (see [`intersect_first_order_accumulator_exact_ranges`]) are
    /// `−r ≤ (Σ log F̃)′ ≤ 0` and `0 ≤ (Σ v²/F̃)′ ≤ Σ v²/F̃ ≤ rss`, so
    /// `−ν/2 ≤ V′ ≤ r/2` — a width of at most `(r + ν)/2`. Measured before those
    /// ranges were applied: `±1.95e91` at order 2, `ρ = −18`, i.e. 89 orders of
    /// magnitude outside a range the model fixes at `178`. That is what made the
    /// search report `Unresolved` rather than bracket a stationary point.
    ///
    /// Containment is asserted FIRST and against an independent recurrence: the
    /// ball jet must enclose the scalar `f64` jet, which shares no arithmetic
    /// with it. A narrower enclosure that stops containing the value it encloses
    /// is a worse defect than the width this test exists to bound.
    #[test]
    fn the_certified_jet_contains_the_scalar_jet_and_stays_in_its_closed_form_range() {
        let (x, y, w) = dgp_2300();
        for order in 1..=MAX_ORDER {
            let (nodes, within, n_obs, _response_origin) =
                pool_nodes(&x, &y, &w, order).expect("pool");
            let proper_modes = (nodes.len() - order) as f64;
            let residual_dof = (n_obs - order) as f64;
            for &rho in &[
                -18.0_f64,
                -16.6135,
                -13.841116916640328,
                -10.0,
                -6.0,
                0.0,
                6.0,
            ] {
                let Ok(certificate) =
                    certified_concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                else {
                    continue;
                };
                let scalar = concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                    .expect("independent scalar recurrence");
                assert!(
                    certificate.value.interval().contains(scalar.0),
                    "order={order} rho={rho}: scalar value {} escaped {:?}",
                    scalar.0,
                    certificate.value
                );
                assert!(
                    certificate.derivative.interval().contains(scalar.1),
                    "order={order} rho={rho}: scalar derivative {} escaped {:?}",
                    scalar.1,
                    certificate.derivative
                );
                let width = certificate.derivative.hi - certificate.derivative.lo;
                assert!(
                    width < proper_modes + residual_dof,
                    "order={order} rho={rho}: the certified derivative ball is {width:e} \
                     wide, outside the closed-form range the accumulators are bounded to \
                     ({:e}); the search cannot sign an interval that wide",
                    0.5 * (proper_modes + residual_dof)
                );
            }
        }
    }

    /// The amplifier behind every width in this file, measured on the map
    /// itself rather than inferred from what it produces — and the one bound on
    /// it that needs no interval product.
    ///
    /// The filtered covariance sits at its Riccati fixed point on this fixture
    /// (`P₁₁ = 1225.652` and `P₂₂ = 950572.8` at nodes 40, 60 and 80 alike), so
    /// the recursion's true width map is the closed-loop congruence
    /// `Ψ = F A`, `A = I − K e₀ᵀ`, and it CONTRACTS. The componentwise interval
    /// evaluation of that same recursion propagates widths through `|Ψ|`
    /// instead, and that EXPLODES. Both products are formed here, over the whole
    /// proper range, from the traced per-node gains.
    ///
    /// This is not dependency loss that a corner or exact-range evaluation can
    /// reach. `Ψ` has `−K_i` below the diagonal of its first column and `+δ`
    /// above it, so the sign of the `1↔2` cycle is NEGATIVE — and a cycle's sign
    /// is invariant under diagonal similarity, so no rescaling of the state
    /// makes `|Ψ| = Ψ`. The cancellation is between coordinates of one step, and
    /// no componentwise interval arithmetic in any diagonal basis can see it.
    /// Every enclosure this file builds by recursion over nodes — the
    /// covariance, its jets, the mean, and equally the BACKWARD smoother
    /// recursions a closed-form `V′` would need, which propagate through `Ψᵀ`
    /// and inherit the same factor — is bounded below by it.
    ///
    /// THE THIRD COLUMN is the reason this test is worth its cost. The Riccati
    /// recursion is `P⁻_{t+1} = Ψ_t P⁻_t Ψ_tᵀ + G_t` with
    /// `G_t = F R K Kᵀ Fᵀ + Q ⪰ 0`, which is an IDENTITY at every node and not
    /// only at a fixed point. So `Ψ_t P⁻_t Ψ_tᵀ ⪯ P⁻_{t+1}`, and with
    /// `S_t = (P⁻_{t+1})^{-1/2} Ψ_t (P⁻_t)^{1/2}` that says `‖S_t‖₂ ≤ 1` —
    /// the closed loop is a contraction in the metric its own covariance
    /// defines. The product telescopes,
    /// `Π Ψ = (P⁻_b)^{1/2}(S_b ⋯ S_a)(P⁻_a)^{-1/2}`, so `Π‖S_t‖₂` bounds the
    /// signed product with NO interval product formed anywhere: a per-node
    /// scalar, each one certifiable on its own. That is what a windowed repair
    /// would need in place of the exploding column, and this test measures
    /// whether the sub-multiplicative bound is strong enough to be that
    /// replacement — `Π‖S_t‖` against the `‖Π Ψ‖` it must stand in for.
    ///
    /// As measured (order 3, ρ = −16.6135, 175 closed-loop steps):
    ///
    /// ```text
    ///   steps    ‖Π Ψ‖        ‖Π |Ψ|‖       Π‖S_t‖
    ///      20    1.73e−1      9.01e+6       8.08e−1
    ///      40    1.13e−3      2.81e+11      6.48e−1
    ///      80    1.02e−9      2.74e+20      4.17e−1
    ///     120    1.83e−17     1.78e+33      1.52e−1
    ///     175    9.54e−30     4.40e+51      3.19e−2
    ///   per step 0.6826       1.9729        0.98050   (worst step 0.993226)
    /// ```
    ///
    /// Read all three. The filter contracts by 30 orders of magnitude over its
    /// own data while the componentwise interval evaluation of the same
    /// recursion inflates by 51 — 81 orders between what the filter does and
    /// what that arithmetic can prove about it, and `4.4e51 × 2.2e−16` is why
    /// quantities whose values are bit-stable carry enclosures of no
    /// information at all.
    ///
    /// And the Lyapunov column HOLDS but does not RESCUE. Every `‖S_t‖₂` is at
    /// most one exactly as the Riccati identity says (largest 0.993226), so the
    /// bound is real and needs no interval product — but `Π‖S_t‖` decays at
    /// 0.98050 per step against the true 0.6826, so over the same 175 steps it
    /// certifies `3.19e−2` where the truth is `9.54e−30`. Twenty-seven orders
    /// too weak, because `‖S_t‖₂` is the WORST direction — the barely-observed
    /// curvature coordinate, contracting at 0.9932 — while the product contracts
    /// fast only because its dominant directions ROTATE, which submultiplicativity
    /// cannot see.
    ///
    /// What that leaves is not "the Lyapunov structure is useless" but a
    /// sharper statement of the repair: the `S_t` are contractions in the metric
    /// the covariance defines, so an interval product OF THE `S_t` grows its
    /// widths additively rather than geometrically. Carrying the enclosure in
    /// `P^{1/2}` coordinates — not bounding the product by a product of bounds —
    /// is the move, and this test certifies per node the one property that makes
    /// that preconditioner the right one.
    #[test]
    fn the_closed_loop_map_contracts_while_its_absolute_value_explodes() {
        let (x, y, w) = dgp_2300();
        let order = 3;
        // This is the middle of the formerly refusing order-3 tail.
        let log_lambda = -16.6135_f64;
        let (nodes, within, n_obs, _response_origin) =
            pool_nodes(&x, &y, &w, order).expect("pool");
        let q_value =
            gam_problem::checked_exp_log_strength(-log_lambda).expect("inverse log strength");
        let q = Ball::certified(
            q_value,
            gam_math::score_opt::certified_exp(-log_lambda).expect("certified exponential"),
        );
        let mut trace: Vec<BallTraceRecord> = Vec::new();
        certified_concentrated_criterion_jet(&nodes, within, n_obs, log_lambda, order)
            .expect("the certified jet must exist at the rho this map is measured at");
        run_filter_ball_traced(&nodes, q, order, Some(&mut trace)).expect("traced pass");
        let mut gains: HashMap<usize, [f64; MAX_ORDER]> = HashMap::new();
        let mut predicted: HashMap<usize, Mat2> = HashMap::new();
        for (node, name, ball) in &trace {
            if let Some(coordinate) = GAIN_NAMES.iter().position(|candidate| candidate == name) {
                gains.entry(*node).or_insert([0.0; MAX_ORDER])[coordinate] = ball.value;
            }
            for (i, row) in P_NEXT_ENTRY_NAMES.iter().enumerate().take(order) {
                for (j, entry) in row.iter().enumerate().take(order) {
                    if entry == name {
                        predicted
                            .entry(*node)
                            .or_insert([[0.0; MAX_ORDER]; MAX_ORDER])[i][j] = ball.value;
                    }
                }
            }
        }
        let max_norm = |matrix: &Mat2| -> f64 {
            let mut norm = 0.0_f64;
            for row in matrix.iter().take(order) {
                for entry in row.iter().take(order) {
                    norm = norm.max(entry.abs());
                }
            }
            norm
        };
        // Largest eigenvalue of a matrix similar to a symmetric PSD one, by
        // power iteration. `None` when the iterate collapses, which is a
        // statement about this fixture and not about the matrix.
        let spectral_radius = |matrix: &Mat2| -> Option<f64> {
            let mut vector = [1.0_f64; MAX_ORDER];
            let mut radius = 0.0_f64;
            let mut iterations = 0usize;
            while iterations < 500 {
                let mut next = [0.0_f64; MAX_ORDER];
                for i in 0..order {
                    for k in 0..order {
                        next[i] += matrix[i][k] * vector[k];
                    }
                }
                let scale = next
                    .iter()
                    .take(order)
                    .fold(0.0_f64, |widest, entry| widest.max(entry.abs()));
                if !(scale > 0.0 && scale.is_finite()) {
                    return None;
                }
                for i in 0..order {
                    vector[i] = next[i] / scale;
                }
                radius = scale;
                iterations += 1;
            }
            Some(radius)
        };
        let mut signed: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
        let mut absolute: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
        for i in 0..order {
            signed[i][i] = 1.0;
            absolute[i][i] = 1.0;
        }
        let mut log_lyapunov = 0.0_f64;
        let mut worst_step = 0.0_f64;
        let mut steps = 0usize;
        for t in (order + 1)..(nodes.len() - 1) {
            let (Some(gain), Some(before), Some(after)) =
                (gains.get(&t), predicted.get(&(t - 1)), predicted.get(&t))
            else {
                continue;
            };
            let delta = nodes[t + 1].x - nodes[t].x;
            let ball_f = ball_transition(Ball::exact(delta), order);
            let mut transition: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
            let mut update: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
            for i in 0..order {
                update[i][i] = 1.0;
                for j in 0..order {
                    transition[i][j] = ball_f[i][j].value;
                }
            }
            for i in 0..order {
                update[i][0] -= gain[i];
            }
            // Update THEN predict, which is the order the filter runs in.
            let closed = mat_mul(&transition, &update, order);
            let mut next_signed: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
            let mut next_absolute: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
            for i in 0..order {
                for j in 0..order {
                    for k in 0..order {
                        next_signed[i][j] += closed[i][k] * signed[k][j];
                        next_absolute[i][j] += closed[i][k].abs() * absolute[k][j];
                    }
                }
            }
            signed = next_signed;
            absolute = next_absolute;
            // `‖S_t‖₂² = λ_max((P⁻_{t+1})⁻¹ Ψ P⁻_t Ψᵀ)`.
            let Ok(inverse_after) = mat_inv(after, order, "lyapunov weight") else {
                continue;
            };
            let congruence = mat_mul(
                &mat_mul(&closed, before, order),
                &mat_t(&closed, order),
                order,
            );
            let Some(squared) = spectral_radius(&mat_mul(&inverse_after, &congruence, order))
            else {
                continue;
            };
            let factor = squared.max(0.0).sqrt();
            worst_step = worst_step.max(factor);
            log_lyapunov += factor.ln();
            steps += 1;
            if steps % 20 == 0 {
                eprintln!(
                    "after {steps} steps: ||prod Psi|| = {:.6e}, ||prod |Psi||| = {:.6e}, \
                     prod ||S_t|| = {:.6e}",
                    max_norm(&signed),
                    max_norm(&absolute),
                    log_lyapunov.exp()
                );
            }
        }
        let contracted = max_norm(&signed);
        let inflated = max_norm(&absolute);
        let lyapunov = log_lyapunov.exp();
        eprintln!(
            "closed loop over {steps} steps: signed {contracted:.6e}, absolute {inflated:.6e}, \
             lyapunov {lyapunov:.6e}; per step signed {:.4}, absolute {:.4}, lyapunov {:.6}, \
             worst single step {worst_step:.6}",
            contracted.powf(1.0 / steps as f64),
            inflated.powf(1.0 / steps as f64),
            lyapunov.powf(1.0 / steps as f64)
        );
        assert!(
            contracted < 1.0,
            "the closed-loop product does not contract ({contracted:e} over {steps} steps); \
             the filter's own stability is the premise of every width argument here"
        );
        assert!(
            inflated > 1.0e10,
            "the absolute closed-loop product no longer explodes ({inflated:e} over {steps} \
             steps). If that is a repair, the recursion-level enclosures can be tightened \
             directly and this test is where the new factor is recorded"
        );
        assert!(
            worst_step <= 1.0 + 1.0e-9,
            "the Riccati identity `Psi P Psi^T + G = P_next` with `G >= 0` makes every \
             `||S_t||_2` at most one; the largest measured is {worst_step}, so either the \
             traced covariance is not the one the recursion produced or the identity is \
             being read wrong"
        );
        assert!(
            lyapunov >= contracted,
            "the Lyapunov product {lyapunov:e} must bound the signed product {contracted:e} \
             it stands in for"
        );
    }

    /// The centred Riccati representation keeps the filtered-mean enclosure
    /// below the search resolution throughout the former order-3 failure band.
    ///
    /// Before #2614, `mean_a0` stayed O(1) while its enclosure width grew from
    /// `3.6e-7` at node 8 to `2.3e254` at node 120. That was pure dependency
    /// loss: the scalar filter remained stable. The repaired path must preserve
    /// both facts directly — bounded values and a finite enclosure narrower
    /// than the resolution the certified search asks it to support — and the
    /// criterion consuming that pass must certify rather than refuse.
    #[test]
    fn centred_riccati_mean_enclosure_stays_below_search_resolution() {
        let (x, y, w) = dgp_2300();
        let order = 3;
        let log_lambda = -16.6135_f64;
        let (nodes, within, n_obs, _response_origin) =
            pool_nodes(&x, &y, &w, order).expect("pool");
        let q_value =
            gam_problem::checked_exp_log_strength(-log_lambda).expect("inverse log strength");
        let q = Ball::certified(
            q_value,
            gam_math::score_opt::certified_exp(-log_lambda).expect("certified exponential"),
        );
        let mut trace: Vec<BallTraceRecord> = Vec::new();
        run_filter_ball_traced(&nodes, q, order, Some(&mut trace))
            .expect("the repaired filter must certify the former failure point");
        let mean: Vec<(usize, Ball)> = trace
            .iter()
            .filter(|(_, name, _)| *name == "mean_a0")
            .map(|(node, _, ball)| (*node, *ball))
            .collect();
        assert_eq!(
            mean.len(),
            nodes.len() - order,
            "every proper filter node must expose a mean certificate"
        );
        let resolution = f64::EPSILON.sqrt();
        let widest_value = mean
            .iter()
            .fold(0.0_f64, |widest, (_, ball)| widest.max(ball.value.abs()));
        assert!(
            widest_value < 1.0e2,
            "the filtered mean's VALUE left O(1) at order {order}, rho {log_lambda}: \
             {widest_value:e}"
        );
        for (node, ball) in mean {
            assert!(
                ball.is_finite(),
                "mean enclosure is non-finite at node {node}"
            );
            let width = ball.hi - ball.lo;
            let scaled_resolution = resolution * (1.0 + ball.value.abs());
            assert!(
                width <= scaled_resolution,
                "mean enclosure at node {node} is {width:e} wide, exceeding the \
                 scale-aware search resolution {scaled_resolution:e}"
            );
        }
        certified_concentrated_criterion_jet(&nodes, within, n_obs, log_lambda, order)
            .expect("the criterion consuming the repaired pass must certify");
    }

    /// Value-only diagnostic surface retained for the derivative oracle tests.
    fn concentrated_criterion(
        nodes: &[PooledNode],
        ssr_within: f64,
        n_obs: usize,
        log_lambda: f64,
        order: usize,
    ) -> Result<f64, String> {
        Ok(concentrated_criterion_jet(nodes, ssr_within, n_obs, log_lambda, order)?.0)
    }
    use super::*;

    #[test]
    fn concentrated_score_jet_matches_test_only_differences() {
        let x = [0.0, 0.07, 0.19, 0.41, 0.41, 0.68, 1.0, 1.37];
        let y = [0.2, -0.4, 0.8, 0.1, 0.35, -0.2, 0.7, 0.15];
        let w = [1.0, 2.0, 0.7, 1.4, 0.9, 3.0, 1.2, 0.8];
        for order in 1..=MAX_ORDER {
            let (nodes, within, n_obs, _response_origin) =
                pool_nodes(&x, &y, &w, order).expect("pooled data");
            for &rho in &[-4.0, -0.3, 2.5] {
                let (value, d1, d2, d3) =
                    concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                        .expect("analytic score jet");
                // Finite differences are deliberately confined to this oracle
                // test; production selection uses the analytic sensitivities.
                let h = 2.0e-4;
                let fm = concentrated_criterion(&nodes, within, n_obs, rho - h, order)
                    .expect("left score");
                let fp = concentrated_criterion(&nodes, within, n_obs, rho + h, order)
                    .expect("right score");
                let fm2 = concentrated_criterion(&nodes, within, n_obs, rho - 2.0 * h, order)
                    .expect("far left score");
                let fp2 = concentrated_criterion(&nodes, within, n_obs, rho + 2.0 * h, order)
                    .expect("far right score");
                let d1_fd = (fp - fm) / (2.0 * h);
                let d2_fd = (fp - 2.0 * value + fm) / (h * h);
                let d3_fd = (fp2 - 2.0 * fp + 2.0 * fm - fm2) / (2.0 * h * h * h);
                // Independent finite-difference certificate: endpoint VALUE
                // balls enclose the central quotient, and the global third-
                // derivative theorem bounds its O(h²) truncation remainder.
                let left_ball =
                    certified_concentrated_criterion_jet(&nodes, within, n_obs, rho - h, order)
                        .expect("left value ball");
                let right_ball =
                    certified_concentrated_criterion_jet(&nodes, within, n_obs, rho + h, order)
                        .expect("right value ball");
                let finite_difference = right_ball
                    .value
                    .sub(left_ball.value)
                    .div_positive(Ball::exact(2.0 * h));
                let proper_modes = (nodes.len() - order) as f64;
                let residual_dof = (n_obs - order) as f64;
                let third_bound = 0.5 * (0.25 * proper_modes + 6.0 * residual_dof);
                let truncation = third_bound * h * h / 6.0;
                let certified_center =
                    certified_concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                        .expect("center derivative ball");
                assert!(
                    certified_center.derivative.hi >= finite_difference.lo - truncation
                        && certified_center.derivative.lo <= finite_difference.hi + truncation,
                    "order={order} rho={rho}: analytic derivative ball {:?} is disjoint \
                     from independently value-differenced {:?} ± {truncation:e}",
                    certified_center.derivative,
                    finite_difference
                );
                let d1_scale = 1.0 + d1.abs().max(d1_fd.abs());
                let d2_scale = 1.0 + d2.abs().max(d2_fd.abs());
                let d3_scale = 1.0 + d3.abs().max(d3_fd.abs());
                assert!(
                    (d1 - d1_fd).abs() <= 2.0e-6 * d1_scale,
                    "order={order} rho={rho}: analytic d1={d1}, FD={d1_fd}"
                );
                assert!(
                    (d2 - d2_fd).abs() <= 2.0e-4 * d2_scale,
                    "order={order} rho={rho}: analytic d2={d2}, FD={d2_fd}"
                );
                assert!(
                    (d3 - d3_fd).abs() <= 5.0e-3 * d3_scale,
                    "order={order} rho={rho}: analytic d3={d3}, FD={d3_fd}"
                );
            }
        }
    }

    #[test]
    fn directed_score_balls_contain_independent_scalar_jets_across_scales() {
        let base_x = [0.0, 0.03, 0.11, 0.27, 0.52, 0.81, 1.17, 1.6];
        let y = [2.0e3, -4.0e2, 8.0e2, 1.0e2, 3.5e2, -2.0e2, 7.0e2, 1.5e2];
        let w = [1.0e-4, 2.0e4, 0.7, 1.4e3, 9.0e-3, 3.0e2, 1.2, 8.0e-2];
        for order in 1..=MAX_ORDER {
            for scale in [1.0e-1_f64, 1.0, 1.0e2] {
                let x: Vec<f64> = base_x.iter().map(|value| scale * value).collect();
                let (nodes, within, n_obs, _response_origin) =
                    pool_nodes(&x, &y, &w, order).expect("adversarial pooled data");
                let rho = (2 * order - 1) as f64 * scale.ln() + 0.35;
                let certified =
                    certified_concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                        .expect("directed score recurrence");
                let scalar = concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                    .expect("independent scalar recurrence");
                for (name, ball, reference) in [
                    ("value", certified.value, scalar.0),
                    ("derivative", certified.derivative, scalar.1),
                    ("curvature", certified.curvature, scalar.2),
                    ("third", certified.third, scalar.3),
                ] {
                    assert!(
                        ball.interval().contains(reference),
                        "order={order} scale={scale:e}: scalar {name} {reference} escaped {ball:?}"
                    );
                }

                let point_sample = ScoreSample {
                    x: rho,
                    value: certified.jet.value,
                    derivative: certified.jet.derivative,
                    curvature: certified.jet.curvature,
                    third: certified.jet.third,
                };
                let point_enclosure = concentrated_criterion_enclosure(
                    nodes.len(),
                    n_obs,
                    point_sample,
                    point_sample,
                    certified,
                    certified,
                    order,
                )
                .expect("degenerate point enclosure");
                assert_eq!(
                    point_enclosure.derivative,
                    certified.derivative.interval(),
                    "a zero-width cell must preserve the certified point derivative exactly"
                );
                assert_eq!(
                    point_enclosure.curvature,
                    certified.curvature.interval(),
                    "a zero-width cell must preserve the certified point curvature exactly"
                );
                assert_eq!(
                    point_enclosure.score.value,
                    certified.value.interval(),
                    "a zero-width cell must preserve the certified point score exactly"
                );

                let rho_right = rho + 0.125;
                let right =
                    certified_concentrated_criterion_jet(&nodes, within, n_obs, rho_right, order)
                        .expect("right endpoint ball");
                let enclosure = concentrated_criterion_enclosure(
                    nodes.len(),
                    n_obs,
                    ScoreSample {
                        x: rho,
                        value: certified.jet.value,
                        derivative: certified.jet.derivative,
                        curvature: certified.jet.curvature,
                        third: certified.jet.third,
                    },
                    ScoreSample {
                        x: rho_right,
                        value: right.jet.value,
                        derivative: right.jet.derivative,
                        curvature: right.jet.curvature,
                        third: right.jet.third,
                    },
                    certified,
                    right,
                    order,
                )
                .expect("endpoint-anchored enclosure");
                for certificate in [certified, right] {
                    assert!(
                        enclosure.derivative.lo <= certificate.derivative.lo
                            && enclosure.derivative.hi >= certificate.derivative.hi,
                        "exact endpoint derivative escaped the cell enclosure"
                    );
                    assert!(
                        enclosure.curvature.lo <= certificate.curvature.lo
                            && enclosure.curvature.hi >= certificate.curvature.hi,
                        "exact endpoint curvature escaped the cell enclosure"
                    );
                    assert!(
                        enclosure.score.value.lo <= certificate.value.lo
                            && enclosure.score.value.hi >= certificate.value.hi,
                        "exact endpoint score escaped the cell enclosure"
                    );
                }
            }
        }
    }

    /// Regression oracle for both #2614 saturated order-3 tail cells. The
    /// production enclosure is a theorem, not a sampling scheme; these dense
    /// scalar evaluations independently guard its implementation, while the
    /// comparison with the old full-width L4 theorem proves that
    /// nearest-endpoint endpoint-third interpolation actually removes (rather
    /// than merely moves) the false Taylor uncertainty.
    #[test]
    fn nearest_endpoint_taylor_hull_contains_dense_cell_and_tightens_every_channel() {
        let n = 60usize;
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        x[7] = x[6];
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                (6.0 * xi).sin() + 0.3 * (17.0 * xi).cos() + 0.05 * ((i * 37 % 11) as f64 - 5.0)
            })
            .collect();
        let w: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * (i % 3) as f64).collect();
        let order = 3usize;
        let (nodes, within, n_obs, _response_origin) =
            pool_nodes(&x, &y, &w, order).expect("pooled data");
        let lo = 13.759_277_343_75;
        let hi = 13.760_375_976_562_5;
        let left = certified_concentrated_criterion_jet(&nodes, within, n_obs, lo, order)
            .expect("left endpoint certificate");
        let right = certified_concentrated_criterion_jet(&nodes, within, n_obs, hi, order)
            .expect("right endpoint certificate");
        let sample = |rho: f64, certificate: CertifiedCriterionJet| ScoreSample {
            x: rho,
            value: certificate.jet.value,
            derivative: certificate.jet.derivative,
            curvature: certificate.jet.curvature,
            third: certificate.jet.third,
        };
        let nearest = concentrated_criterion_enclosure(
            nodes.len(),
            n_obs,
            sample(lo, left),
            sample(hi, right),
            left,
            right,
            order,
        )
        .expect("nearest-endpoint enclosure");

        for step in 0..=256 {
            let rho = lo + (hi - lo) * step as f64 / 256.0;
            let (value, derivative, curvature, _) =
                concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                    .expect("independent scalar jet");
            assert!(
                nearest.score.value.contains(value),
                "dense score sample at rho={rho:.17} escaped {:?}",
                nearest.score.value
            );
            assert!(
                nearest.derivative.contains(derivative),
                "dense derivative sample at rho={rho:.17} escaped {:?}",
                nearest.derivative
            );
            assert!(
                nearest.curvature.contains(curvature),
                "dense curvature sample at rho={rho:.17} escaped {:?}",
                nearest.curvature
            );
        }

        // Re-evaluate the identical certified Taylor theorem with each endpoint
        // spanning the FULL cell. This is the pre-fix geometry, expressed
        // directionally rather than weakened further into absolute-value
        // radii, so beating it is the stronger comparison.
        let width = Ball::exact(hi).sub(Ball::exact(lo));
        let width2 = width.square();
        let width3 = width2.mul(width);
        let width4 = width2.square();
        let fourth_abs_bound = Ball::exact((nodes.len() - order) as f64)
            .scale(0.25)
            .add(Ball::exact((n_obs - order) as f64).scale(26.0))
            .scale(0.5);
        let value_remainder = fourth_abs_bound
            .mul(width4)
            .div_positive(Ball::exact(24.0))
            .hi;
        let derivative_remainder = fourth_abs_bound
            .mul(width3)
            .div_positive(Ball::exact(6.0))
            .hi;
        let curvature_remainder = fourth_abs_bound.mul(width2).scale(0.5).hi;
        let full_cell_from_endpoint =
            |certificate: CertifiedCriterionJet, displacement: ClosedInterval| {
                let d = Ball::certified(0.0, displacement);
                let d2 = d.square();
                let d3 = d2.mul(d);
                let value = certificate
                    .value
                    .add(certificate.derivative.mul(d))
                    .add(certificate.curvature.mul(d2).scale(0.5))
                    .add(certificate.third.mul(d3).div_positive(Ball::exact(6.0)))
                    .interval()
                    .add(ClosedInterval::new(-value_remainder, value_remainder));
                let derivative = certificate
                    .derivative
                    .add(certificate.curvature.mul(d))
                    .add(certificate.third.mul(d2).scale(0.5))
                    .interval()
                    .add(ClosedInterval::new(
                        -derivative_remainder,
                        derivative_remainder,
                    ));
                let curvature = certificate
                    .curvature
                    .add(certificate.third.mul(d))
                    .interval()
                    .add(ClosedInterval::new(
                        -curvature_remainder,
                        curvature_remainder,
                    ));
                (value, derivative, curvature)
            };
        let old_left = full_cell_from_endpoint(left, ClosedInterval::new(0.0, width.hi));
        let old_right = full_cell_from_endpoint(right, ClosedInterval::new(-width.hi, 0.0));
        let old_value = ClosedInterval::new(
            old_left.0.lo.min(old_right.0.lo),
            old_left.0.hi.max(old_right.0.hi),
        );
        let old_derivative = ClosedInterval::new(
            old_left.1.lo.min(old_right.1.lo),
            old_left.1.hi.max(old_right.1.hi),
        );
        let old_curvature = ClosedInterval::new(
            old_left.2.lo.min(old_right.2.lo),
            old_left.2.hi.max(old_right.2.hi),
        );
        for (name, tightened, full_width) in [
            ("score", nearest.score.value, old_value),
            ("derivative", nearest.derivative, old_derivative),
            ("curvature", nearest.curvature, old_curvature),
        ] {
            assert!(
                tightened.hi - tightened.lo < full_width.hi - full_width.lo,
                "nearest-endpoint {name} enclosure {tightened:?} was not strictly \
                 narrower than full-width theorem {full_width:?}"
            );
        }
        assert!(
            nearest.derivative.hi < 0.0,
            "the corrected theorem must certify the live #2614 cell's negative slope: {:?}",
            nearest.derivative
        );

        // The half-cell L4 theorem above exposed the next saturated cell at
        // rho≈16.127. It has the same dyadic width, but its endpoint slope is
        // only 2.76e-9, so the old global L4 remainder is eight times larger
        // than the signal even with correct nearest-endpoint geometry. The
        // endpoint-third/L5 theorem must contain the whole cell AND recover its
        // sign; otherwise it merely moves the same budget exhaustion again.
        let shifted_lo = 16.126_831_054_687_5;
        let shifted_hi = 16.127_929_687_5;
        assert_eq!(
            shifted_hi - shifted_lo,
            hi - lo,
            "the old-theorem comparison below shares the measured dyadic width"
        );
        let shifted_left =
            certified_concentrated_criterion_jet(&nodes, within, n_obs, shifted_lo, order)
                .expect("shifted left endpoint certificate");
        let shifted_right =
            certified_concentrated_criterion_jet(&nodes, within, n_obs, shifted_hi, order)
                .expect("shifted right endpoint certificate");
        let shifted = concentrated_criterion_enclosure(
            nodes.len(),
            n_obs,
            sample(shifted_lo, shifted_left),
            sample(shifted_hi, shifted_right),
            shifted_left,
            shifted_right,
            order,
        )
        .expect("shifted endpoint-third enclosure");
        for step in 0..=256 {
            let rho = shifted_lo + (shifted_hi - shifted_lo) * step as f64 / 256.0;
            let (value, derivative, curvature, _) =
                concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                    .expect("shifted independent scalar jet");
            assert!(
                shifted.score.value.contains(value),
                "shifted dense score at rho={rho:.17} escaped {:?}",
                shifted.score.value
            );
            assert!(
                shifted.derivative.contains(derivative),
                "shifted dense derivative at rho={rho:.17} escaped {:?}",
                shifted.derivative
            );
            assert!(
                shifted.curvature.contains(curvature),
                "shifted dense curvature at rho={rho:.17} escaped {:?}",
                shifted.curvature
            );
        }
        let shifted_old_left =
            full_cell_from_endpoint(shifted_left, ClosedInterval::new(0.0, width.hi));
        let shifted_old_right =
            full_cell_from_endpoint(shifted_right, ClosedInterval::new(-width.hi, 0.0));
        for (name, tightened, old_left, old_right) in [
            (
                "score",
                shifted.score.value,
                shifted_old_left.0,
                shifted_old_right.0,
            ),
            (
                "derivative",
                shifted.derivative,
                shifted_old_left.1,
                shifted_old_right.1,
            ),
            (
                "curvature",
                shifted.curvature,
                shifted_old_left.2,
                shifted_old_right.2,
            ),
        ] {
            let full_width =
                ClosedInterval::new(old_left.lo.min(old_right.lo), old_left.hi.max(old_right.hi));
            assert!(
                tightened.hi - tightened.lo < full_width.hi - full_width.lo,
                "endpoint-third {name} enclosure {tightened:?} was not strictly \
                 narrower than the full-width L4 theorem {full_width:?}"
            );
        }
        assert!(
            shifted.derivative.hi < 0.0,
            "the endpoint-third theorem must certify the shifted #2614 cell's \
             negative slope: {:?}",
            shifted.derivative
        );
    }

    #[test]
    fn spline_consumer_preserves_a_valid_resolution_flat_optimum_category() {
        let optimum = ScoreSample {
            x: -0.25,
            value: 3.0,
            derivative: 0.0,
            curvature: 0.0,
            third: 0.0,
        };
        let bracket = ClosedInterval::new(-0.5, 0.0);
        let max_score_gap = 0.125;
        let score_resolution = 0.25;
        let search = ScoreSearchResult {
            optimum,
            location: ScoreOptimumLocation::ResolutionFlat(0),
            lower_boundary: ScoreSample { x: -1.0, ..optimum },
            upper_boundary: ScoreSample { x: 1.0, ..optimum },
            stationary_points: Vec::new(),
            resolution_flat_regions: vec![gam_math::score_opt::ResolutionFlatRegion {
                sample: optimum,
                bracket,
                score: ClosedInterval::new(2.875, 3.0),
                max_score_gap,
                score_resolution,
            }],
            dominated_regions: Vec::new(),
            value_certificate: gam_math::score_opt::GlobalScoreCertificate {
                selected: ClosedInterval::point(3.0),
                maximum: ClosedInterval::new(3.0, 3.125),
                maximum_excess: max_score_gap,
                comparison_resolution: score_resolution,
            },
        };
        assert_eq!(
            spline_optimum_proof(&search).expect("valid resolution-flat proof"),
            SplineOptimumProof::ResolutionFlat {
                bracket,
                max_score_gap,
                score_resolution,
            },
            "the spline consumer must preserve the producer's successful typed category"
        );

        let mut invalid = search;
        invalid.resolution_flat_regions[0].max_score_gap =
            invalid.resolution_flat_regions[0].score_resolution + f64::EPSILON;
        assert!(
            matches!(
                spline_optimum_proof(&invalid),
                Err(SplineScoreProofError::Search(_))
            ),
            "a malformed producer certificate must still fail instead of being accepted"
        );
    }

    #[test]
    fn spline_consumer_retains_the_producers_stationary_curvature_proof() {
        let optimum = ScoreSample {
            x: -9.084_292_923_99,
            value: 3.0,
            derivative: 0.0,
            curvature: -1.0,
            third: 0.0,
        };
        let bracket = ClosedInterval::new(-9.084_292_924_175_005, -9.084_292_923_812_374);
        let producer_curvature = ClosedInterval::new(-6.4, -0.2);
        let point_score = ScoreValueEnclosure {
            value: ClosedInterval::new(2.999, 3.001),
            evaluation_error: 0.001,
        };
        let search = ScoreSearchResult {
            optimum,
            location: ScoreOptimumLocation::Stationary(0),
            lower_boundary: ScoreSample {
                x: -10.0,
                ..optimum
            },
            upper_boundary: ScoreSample { x: -8.0, ..optimum },
            stationary_points: vec![gam_math::score_opt::StationaryPoint {
                sample: optimum,
                bracket,
                score: point_score,
                curvature: producer_curvature,
            }],
            resolution_flat_regions: Vec::new(),
            dominated_regions: Vec::new(),
            value_certificate: gam_math::score_opt::GlobalScoreCertificate {
                selected: point_score.value,
                maximum: point_score.value,
                maximum_excess: 0.0,
                comparison_resolution: 0.002,
            },
        };
        let SplineOptimumProof::Kkt { bracket: got, kind } =
            spline_optimum_proof(&search).expect("valid stationary proof")
        else {
            panic!("stationary producer category was not preserved");
        };
        assert_eq!(got, bracket);
        assert_eq!(
            kind,
            SplineKktKind::Stationary {
                curvature: producer_curvature,
            }
        );

        let local_enclosure = DerivativeEnclosure {
            score: point_score,
            derivative: ClosedInterval::new(-1.2e-9, 1.2e-9),
            // Mirrors the persistence failure: a fresh tiny-cell secant loses
            // curvature sign even though the parent proof remains strict.
            curvature: ClosedInterval::new(-6.39, 0.0064),
        };
        let (holds, consumed_curvature) = spline_kkt_holds(kind, local_enclosure);
        assert!(holds, "the final derivative still contains the unique root");
        assert_eq!(consumed_curvature, producer_curvature);
    }

    #[test]
    fn derivative_secant_recovers_weighted_order3_root_curvature_sign() {
        let (x, y, w) = dgp_2300();
        let order = 3usize;
        let (nodes, within, n_obs, _response_origin) =
            pool_nodes(&x, &y, &w, order).expect("weighted pool");
        // Live cell at evaluation 1024 of the pre-secant #2300 traversal.
        let lo = -2.337_075_252_506_015;
        let hi = -2.337_040_920_230_624;
        let left = certified_concentrated_criterion_jet(&nodes, within, n_obs, lo, order)
            .expect("weighted left endpoint");
        let right = certified_concentrated_criterion_jet(&nodes, within, n_obs, hi, order)
            .expect("weighted right endpoint");
        assert!(
            left.curvature.interval().contains_zero() && right.curvature.interval().contains_zero(),
            "the oracle must exercise the loose direct covariance-d2 path"
        );
        let sample = |rho: f64, certificate: CertifiedCriterionJet| ScoreSample {
            x: rho,
            value: certificate.jet.value,
            derivative: certificate.jet.derivative,
            curvature: certificate.jet.curvature,
            third: certificate.jet.third,
        };
        let enclosure = concentrated_criterion_enclosure(
            nodes.len(),
            n_obs,
            sample(lo, left),
            sample(hi, right),
            left,
            right,
            order,
        )
        .expect("secant curvature enclosure");
        assert!(
            enclosure.curvature.hi < 0.0,
            "the derivative secant must recover strict concavity: {:?}",
            enclosure.curvature
        );
        assert!(
            enclosure.derivative.lo > 0.0,
            "integrating the secant curvature from both endpoints must preserve \
             the live cell's positive slope: {:?}",
            enclosure.derivative
        );
        for step in 0..=256 {
            let rho = lo + (hi - lo) * step as f64 / 256.0;
            let (_, derivative, curvature, _) =
                concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                    .expect("independent weighted scalar jet");
            assert!(
                enclosure.derivative.contains(derivative),
                "weighted scalar derivative {derivative} at rho={rho:.17} escaped {:?}",
                enclosure.derivative
            );
            assert!(
                enclosure.curvature.contains(curvature),
                "weighted scalar curvature {curvature} at rho={rho:.17} escaped {:?}",
                enclosure.curvature
            );
        }
    }

    #[test]
    fn score_proof_refuses_exactly_when_diffuse_innovation_ball_contains_zero() {
        assert_eq!(
            Ball::ZERO.square(),
            Ball::ZERO,
            "structural zero must survive squaring exactly"
        );
        assert_eq!(
            Ball::ONE.square(),
            Ball::ONE,
            "the exact unit covariance must not acquire artificial width"
        );
        let tiny = f64::from_bits(1);
        let nodes = [
            PooledNode {
                x: 0.0,
                y: 0.0,
                w: 1.0,
            },
            PooledNode {
                x: tiny,
                y: 1.0,
                w: 1.0,
            },
            PooledNode {
                x: 1.0,
                y: -1.0,
                w: 1.0,
            },
        ];
        let error = run_filter_ball(&nodes, Ball::ONE, 2)
            .expect_err("an underflow-wide diffuse innovation cannot be divided soundly");
        assert!(matches!(
            error,
            SplineScoreProofError::InnovationContainsZero {
                node: 1,
                kind: SplineInnovationKind::Diffuse,
                ..
            }
        ));
    }

    /// A hand-built persisted state, valid by `from_state`'s own structural
    /// rules: `state` is `order` per knot, `cov` the `order(order+1)/2` upper
    /// triangle per knot, `gain` the full `order²` per knot, one weight per
    /// knot, strictly increasing knots, `sigma2 > 0`, and an in-range
    /// `log_lambda`. No fitting is involved.
    fn hand_built_state(order: usize) -> SplineScanState {
        let knots = vec![0.0, 0.25, 0.6, 1.0, 1.4];
        let knot_count = knots.len();
        let tri = order * (order + 1) / 2;
        SplineScanState {
            order,
            state: (0..order * knot_count)
                .map(|i| 0.1 + 0.07 * i as f64)
                .collect(),
            // Diagonal-leading per knot so restored variances stay positive.
            cov: (0..tri * knot_count)
                .map(|i| {
                    if i % tri == 0 {
                        0.5 + 0.01 * i as f64
                    } else {
                        0.02
                    }
                })
                .collect(),
            gain: (0..order * order * knot_count)
                .map(|i| 0.03 * ((i % 5) as f64))
                .collect(),
            node_weight: (0..knot_count).map(|i| 1.0 + 0.25 * i as f64).collect(),
            knots,
            log_lambda: 0.35,
            sigma2: 1.75,
            restricted_loglik: -12.5,
            log_likelihood: -8.25,
            training_sample_size: std::num::NonZeroU64::new(64).expect("64 is nonzero"),
            data_sse: 3.25,
        }
    }

    /// #2614 decoupling: the #1034/#1044 persistence seam, verified WITHOUT the
    /// optimizer.
    ///
    /// `round_trip_predict_bit_for_bit` opens with
    /// `fit_spline_scan(...).expect("scan fit")`, so while the certified scan
    /// refuses (#2614, measured: two of those three tests die there) every
    /// assertion behind it is WITHDRAWN rather than failing — the bit-for-bit
    /// posteriors, the off-knot bridge, both extrapolation sides, and the three
    /// corrupt-payload rejections are all unprotected, and the red count reads
    /// "some tests fail" when the truth is "a guarantee is untested".
    ///
    /// A serialization guarantee must not depend on an optimizer guarantee.
    /// `SplineScanFit::from_state` reconstructs a fit from a plain
    /// `SplineScanState`, so the whole seam can be driven from a hand-built
    /// state and holds regardless of whether any fit converges. This does NOT
    /// replace the fitted round-trip, which additionally proves the fitter's own
    /// fields survive; it makes the seam itself independently covered.
    #[test]
    fn persistence_seam_round_trips_without_the_optimizer_2614() {
        for order in 1..=MAX_ORDER {
            let built = hand_built_state(order);
            let fit = SplineScanFit::from_state(&built).expect("hand-built state must restore");
            let json = serde_json::to_string(&fit.to_state()).expect("serialize state");
            let parsed: SplineScanState = serde_json::from_str(&json).expect("deserialize state");
            let restored = SplineScanFit::from_state(&parsed).expect("restore fit");

            assert_eq!(fit.order, restored.order, "order drifted (m={order})");
            assert_eq!(fit.knots, restored.knots, "knots drifted (m={order})");
            assert_eq!(fit.log_lambda.to_bits(), restored.log_lambda.to_bits());
            assert_eq!(fit.sigma2.to_bits(), restored.sigma2.to_bits());
            assert_eq!(
                fit.log_likelihood.to_bits(),
                restored.log_likelihood.to_bits()
            );
            assert_eq!(fit.edf().to_bits(), restored.edf().to_bits());
            assert_eq!(fit.deviance().to_bits(), restored.deviance().to_bits());
            assert_eq!(fit.training_sample_size(), restored.training_sample_size());

            // Off-knot bridge, exact knot hits, and both extrapolation sides.
            for &xq in &[-0.3, 0.0, 0.13, 0.6, 1.0, 1.4, 1.9] {
                let (m0, v0) = fit.predict(xq).expect("predict original");
                let (m1, v1) = restored.predict(xq).expect("predict restored");
                assert_eq!(
                    m0.to_bits(),
                    m1.to_bits(),
                    "mean drift at x={xq} (m={order})"
                );
                assert_eq!(
                    v0.to_bits(),
                    v1.to_bits(),
                    "variance drift at x={xq} (m={order})"
                );
            }

            // Corrupt payloads fail loudly, not inside a later predict.
            let mut bad = fit.to_state();
            bad.cov.truncate(bad.cov.len() - 1);
            SplineScanFit::from_state(&bad).expect_err("length mismatch must error");
            let mut bad = fit.to_state();
            bad.sigma2 = -1.0;
            SplineScanFit::from_state(&bad).expect_err("non-positive sigma2 must error");
            let mut bad = fit.to_state();
            bad.knots[2] = bad.knots[1];
            SplineScanFit::from_state(&bad).expect_err("non-increasing knots must error");
        }
    }

    /// A constant response shift is an exact null-space transformation for
    /// every supported spline order. Pin that theorem at the fixed-lambda
    /// evaluator seam, including tied rows with non-unit weights: all invariant
    /// quantities must be bit-identical, and the only moving state coordinate
    /// must be the published function level. Ties are essential here because
    /// centering only after their weighted pooling leaves the old origin leak
    /// alive before the recurrence starts.
    #[test]
    fn fixed_scan_evaluator_uses_a_response_translation_free_chart_2790() {
        let x = [0.0, 0.0, 0.25, 0.5, 0.5, 1.0, 1.5, 2.0];
        let y: [f64; 8] = [0.0, 0.25, -0.5, 0.75, 1.0, -0.25, 0.5, 0.125];
        let shift = 1024.0_f64;
        let shifted_y = y.map(|value| value + shift);
        let w = [0.3, 1.7, 2.25, 0.6, 1.4, 3.1, 0.75, 2.6];

        for order in 1..=MAX_ORDER {
            let plain = fit_spline_scan_at(&x, &y, &w, -1.25, None, order)
                .expect("plain fixed-lambda scan");
            let shifted = fit_spline_scan_at(&x, &shifted_y, &w, -1.25, None, order)
                .expect("shifted fixed-lambda scan");
            assert!(
                plain.knots.len() < x.len(),
                "fixture must exercise tied-row pooling"
            );
            assert_eq!(plain.knots, shifted.knots);
            assert_eq!(plain.node_weight, shifted.node_weight);

            for (label, left, right) in [
                ("sigma2", plain.sigma2, shifted.sigma2),
                (
                    "restricted log likelihood",
                    plain.restricted_loglik,
                    shifted.restricted_loglik,
                ),
                ("Gaussian log likelihood", plain.log_likelihood, shifted.log_likelihood),
                ("data SSE", plain.data_sse, shifted.data_sse),
                ("EDF", plain.edf(), shifted.edf()),
            ] {
                assert_eq!(
                    left.to_bits(),
                    right.to_bits(),
                    "order {order}: {label} moved under a constant response shift"
                );
            }
            for node in 0..plain.knots.len() {
                assert_eq!(
                    shifted.mean[node].to_bits(),
                    (plain.mean[node] + shift).to_bits(),
                    "order {order}: fitted level at node {node} is not exactly equivariant"
                );
                assert_eq!(
                    shifted.var[node].to_bits(),
                    plain.var[node].to_bits(),
                    "order {order}: posterior variance moved at node {node}"
                );
            }
            assert_eq!(shifted.deriv, plain.deriv);
        }
    }

    /// `deviance()` must be the weighted DATA residual sum of squares at the
    /// fitted values, not the profiled REML quadratic. For order 1 on
    /// `x = (0, 1)`, `y = (0, 1)`, unit weights, λ = 1, the posterior mean is
    /// `(1/3, 2/3)`: the data SSE is `2·(1/3)² = 2/9`, while
    /// `σ̂²·(n − order) = 1/3` carries an extra `1/9` of process/roughness
    /// energy.
    #[test]
    fn deviance_is_data_sse_not_penalized_quadratic() {
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        let w = [1.0, 1.0];
        let fit = fit_spline_scan_at(&x, &y, &w, 0.0, None, 1).expect("order-1 fit");
        // Self-consistency against a direct recomputation at the fitted values.
        let manual: f64 = x
            .iter()
            .zip(&y)
            .zip(&w)
            .map(|((&xi, &yi), &wi)| {
                let (m, _) = fit.predict(xi).expect("predict at knot");
                wi * (yi - m) * (yi - m)
            })
            .sum();
        assert!(
            (fit.deviance() - manual).abs() <= 1e-12 * manual.max(1e-300),
            "deviance {} != recomputed data SSE {manual}",
            fit.deviance()
        );
        assert!(
            (fit.deviance() - 2.0 / 9.0).abs() < 1e-10,
            "deviance {} != 2/9",
            fit.deviance()
        );
        // The old proxy is strictly larger: it includes penalty energy.
        let reml_quadratic = fit.sigma2 * (fit.training_sample_size() as f64 - fit.order as f64);
        assert!(fit.deviance() < reml_quadratic);
    }

    #[test]
    fn full_gaussian_log_likelihood_keeps_raw_weight_normalizer_and_round_trips() {
        // The first two rows share an abscissa. Their individual log-weight
        // normalizers cannot be recovered from the pooled node weight.
        let x = [0.0, 0.0, 1.0, 2.0];
        let y = [0.2, -0.1, 0.8, 1.4];
        let w = [0.5, 2.0, 1.5, 3.0];
        let sigma2 = 1.7;
        let fit = fit_spline_scan_at(&x, &y, &w, 0.2, Some(sigma2), 1)
            .expect("weighted order-1 fit");

        let sum_log_weights = w.iter().map(|weight| weight.ln()).sum::<f64>();
        let expected = -0.5
            * (fit.deviance() / sigma2
                + x.len() as f64 * (std::f64::consts::TAU.ln() + sigma2.ln())
                - sum_log_weights);
        assert!(
            (fit.log_likelihood - expected).abs() <= 1e-12 * expected.abs().max(1.0)
        );

        let pooled_log_weights = fit
            .node_weight
            .iter()
            .map(|weight| weight.ln())
            .sum::<f64>();
        let pooled_wrong = -0.5
            * (fit.deviance() / sigma2
                + x.len() as f64 * (std::f64::consts::TAU.ln() + sigma2.ln())
                - pooled_log_weights);
        assert!(
            (fit.log_likelihood - pooled_wrong).abs() > 1e-3,
            "raw-row weight normalizer must not collapse to pooled weights"
        );

        let restored = SplineScanFit::from_state(&fit.to_state()).expect("restore fit");
        assert_eq!(
            fit.log_likelihood.to_bits(),
            restored.log_likelihood.to_bits()
        );

        let mut incomplete = serde_json::to_value(fit.to_state()).expect("serialize state");
        incomplete
            .as_object_mut()
            .expect("state serializes as an object")
            .remove("log_likelihood");
        let error = serde_json::from_value::<SplineScanState>(incomplete)
            .expect_err("log_likelihood is a required wire field");
        assert!(error.to_string().contains("missing field `log_likelihood`"));
    }
}
