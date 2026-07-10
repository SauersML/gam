//! Certified global optimization of one-dimensional scores on a bounded
//! domain, together with the affine-pencil spectral profile shared by the
//! Gaussian REML smoothing-parameter searches.
//!
//! Point samples alone cannot prove that a smooth function has no narrow
//! stationary pair between them.  The search therefore requires two pieces of
//! information from its caller:
//!
//! * an exact point evaluation `(value, first derivative, second derivative)`;
//! * an OUTER enclosure of both derivatives over every requested interval.
//!
//! An interval is discarded only when its first-derivative enclosure excludes
//! zero.  A stationary point is refined only after the second-derivative
//! enclosure excludes zero, proving that the first derivative is monotone and
//! hence that a straddling interval contains exactly one root.  Every other
//! interval is subdivided.  If floating-point spacing or the caller-requested
//! resolution is reached before either fact is proved, the result is a typed
//! [`ScoreSearchError::Unresolved`] rather than a best-effort optimum.
//!
//! [`AffineRemlProfile`] supplies both the point jets and rigorous interval
//! formulas for scores whose penalized Hessian has simultaneously diagonal
//! affine modes `h_i(lambda) = g_i + lambda s_i`.  This covers an ordinary
//! Demmler--Reinsch eigensystem (`g_i = 1`) and a reference-Hessian pencil
//! (`g_i = 1 - lambda_0 mu_i`, `s_i = mu_i`) without any matrix dependency in
//! this crate.

use std::fmt;

/// Closed real interval `[lo, hi]`.
///
/// Search callbacks may use infinite endpoints for conservative bounds, but
/// neither endpoint may be NaN and `lo <= hi` must hold.  The search validates
/// every enclosure returned by a callback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosedInterval {
    pub lo: f64,
    pub hi: f64,
}

impl ClosedInterval {
    #[inline]
    pub const fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Construct an interval and round both supplied bounds one representable
    /// value outward.  This is the public bridge for callers that derive a
    /// real-valued bound with ordinary nearest-rounded scalar arithmetic.
    #[inline]
    pub fn outward(lo: f64, hi: f64) -> Self {
        Self {
            lo: next_down(lo),
            hi: next_up(hi),
        }
    }

    #[inline]
    pub const fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    #[inline]
    pub const fn entire() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    #[inline]
    pub fn contains(self, value: f64) -> bool {
        self.lo <= value && value <= self.hi
    }

    #[inline]
    pub fn contains_zero(self) -> bool {
        self.contains(0.0)
    }

    #[inline]
    fn is_valid(self) -> bool {
        !self.lo.is_nan() && !self.hi.is_nan() && self.lo <= self.hi
    }

    #[inline]
    fn hull(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            lo: next_down(self.lo + other.lo),
            hi: next_up(self.hi + other.hi),
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            lo: next_down(self.lo - other.hi),
            hi: next_up(self.hi - other.lo),
        }
    }

    #[inline]
    fn neg(self) -> Self {
        Self {
            lo: next_down(-self.hi),
            hi: next_up(-self.lo),
        }
    }

    fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for value in products {
            lo = lo.min(value);
            hi = hi.max(value);
        }
        Self {
            lo: next_down(lo),
            hi: next_up(hi),
        }
    }

    #[inline]
    fn scale(self, value: f64) -> Self {
        self.mul(Self::point(value))
    }

    fn square(self) -> Self {
        if self.lo >= 0.0 {
            Self {
                lo: next_down(self.lo * self.lo).max(0.0),
                hi: next_up(self.hi * self.hi),
            }
        } else if self.hi <= 0.0 {
            Self {
                lo: next_down(self.hi * self.hi).max(0.0),
                hi: next_up(self.lo * self.lo),
            }
        } else {
            Self {
                lo: 0.0,
                hi: next_up((self.lo * self.lo).max(self.hi * self.hi)),
            }
        }
    }

    /// Divide by an interval known to be strictly positive.
    fn div_positive(self, denominator: Self) -> Self {
        assert!(
            denominator.lo > 0.0,
            "div_positive requires a strictly positive denominator interval, got lo={}",
            denominator.lo
        );
        let reciprocal = Self {
            lo: next_down(1.0 / denominator.hi).max(0.0),
            hi: next_up(1.0 / denominator.lo),
        };
        self.mul(reciprocal)
    }

    #[inline]
    fn nonnegative(self) -> Self {
        Self {
            lo: self.lo.max(0.0),
            hi: self.hi.max(0.0),
        }
    }
}

/// Value and first two analytic derivatives at one abscissa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreJet {
    pub value: f64,
    pub derivative: f64,
    pub curvature: f64,
}

/// A point evaluation augmented with its abscissa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreSample {
    pub x: f64,
    pub value: f64,
    pub derivative: f64,
    pub curvature: f64,
}

/// Outer derivative ranges supplied to the certified search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivativeEnclosure {
    pub derivative: ClosedInterval,
    pub curvature: ClosedInterval,
}

/// One stationary point together with the final bracket that certifies its
/// location.  The bracket width is no larger than the requested resolution,
/// unless the point was represented exactly (a zero-width bracket).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationaryPoint {
    pub sample: ScoreSample,
    pub bracket: ClosedInterval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreOptimumLocation {
    LowerBoundary,
    UpperBoundary,
    Stationary(usize),
}

/// Complete successful search result.  Endpoints are retained explicitly so
/// the global comparison is independently checkable by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSearchResult {
    pub optimum: ScoreSample,
    pub location: ScoreOptimumLocation,
    pub lower_boundary: ScoreSample,
    pub upper_boundary: ScoreSample,
    pub stationary_points: Vec<StationaryPoint>,
}

/// Failure of the generic certified search.
#[derive(Debug)]
pub enum ScoreSearchError<E> {
    InvalidDomain {
        lo: f64,
        hi: f64,
    },
    InvalidResolution {
        resolution: f64,
    },
    PointEvaluation {
        x: f64,
        source: E,
    },
    EnclosureEvaluation {
        lo: f64,
        hi: f64,
        source: E,
    },
    NonFiniteSample {
        sample: ScoreSample,
    },
    InvalidEnclosure {
        lo: f64,
        hi: f64,
        enclosure: DerivativeEnclosure,
    },
    EnclosureMissesEndpoint {
        lo: f64,
        hi: f64,
        endpoint: ScoreSample,
        enclosure: DerivativeEnclosure,
    },
    /// The enclosure still admits both a stationary point and a curvature
    /// zero, so uniqueness could not be proved before the requested or
    /// floating-point resolution floor.
    Unresolved {
        lo: f64,
        hi: f64,
        requested_resolution: f64,
        enclosure: DerivativeEnclosure,
    },
}

impl<E: fmt::Display> fmt::Display for ScoreSearchError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomain { lo, hi } => {
                write!(f, "score search: invalid domain [{lo}, {hi}]")
            }
            Self::InvalidResolution { resolution } => {
                write!(f, "score search: invalid resolution {resolution}")
            }
            Self::PointEvaluation { x, source } => {
                write!(f, "score search: evaluation failed at {x}: {source}")
            }
            Self::EnclosureEvaluation { lo, hi, source } => write!(
                f,
                "score search: derivative enclosure failed on [{lo}, {hi}]: {source}"
            ),
            Self::NonFiniteSample { sample } => write!(
                f,
                "score search: non-finite jet at {} (value {}, derivative {}, curvature {})",
                sample.x, sample.value, sample.derivative, sample.curvature
            ),
            Self::InvalidEnclosure { lo, hi, enclosure } => write!(
                f,
                "score search: invalid derivative enclosure on [{lo}, {hi}]: {enclosure:?}"
            ),
            Self::EnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                enclosure,
            } => write!(
                f,
                "score search: enclosure on [{lo}, {hi}] misses endpoint jet at {}: {endpoint:?} not in {enclosure:?}",
                endpoint.x
            ),
            Self::Unresolved {
                lo,
                hi,
                requested_resolution,
                enclosure,
            } => write!(
                f,
                "score search: stationary structure unresolved on [{lo}, {hi}] at requested resolution {requested_resolution}: {enclosure:?}"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ScoreSearchError<E> {}

#[derive(Clone, Copy)]
struct SearchNode {
    left: ScoreSample,
    right: ScoreSample,
}

fn evaluate_sample<E, F>(x: f64, evaluate: &mut F) -> Result<ScoreSample, ScoreSearchError<E>>
where
    F: FnMut(f64) -> Result<ScoreJet, E>,
{
    let jet = evaluate(x).map_err(|source| ScoreSearchError::PointEvaluation { x, source })?;
    let sample = ScoreSample {
        x,
        value: jet.value,
        derivative: jet.derivative,
        curvature: jet.curvature,
    };
    if sample.value.is_finite() && sample.derivative.is_finite() && sample.curvature.is_finite() {
        Ok(sample)
    } else {
        Err(ScoreSearchError::NonFiniteSample { sample })
    }
}

fn checked_enclosure<E, F>(
    node: SearchNode,
    enclose: &mut F,
) -> Result<DerivativeEnclosure, ScoreSearchError<E>>
where
    F: FnMut(f64, f64) -> Result<DerivativeEnclosure, E>,
{
    let lo = node.left.x;
    let hi = node.right.x;
    let enclosure = enclose(lo, hi).map_err(|source| ScoreSearchError::EnclosureEvaluation {
        lo,
        hi,
        source,
    })?;
    if !(enclosure.derivative.is_valid() && enclosure.curvature.is_valid()) {
        return Err(ScoreSearchError::InvalidEnclosure { lo, hi, enclosure });
    }
    for endpoint in [node.left, node.right] {
        if !(enclosure.derivative.contains(endpoint.derivative)
            && enclosure.curvature.contains(endpoint.curvature))
        {
            return Err(ScoreSearchError::EnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                enclosure,
            });
        }
    }
    Ok(enclosure)
}

/// Refine a UNIQUE derivative root.  The caller has already proved uniqueness
/// by a curvature enclosure that excludes zero and supplied endpoint
/// derivatives of opposite sign.
fn refine_unique_root<E, F>(
    mut left: ScoreSample,
    mut right: ScoreSample,
    resolution: f64,
    enclosure: DerivativeEnclosure,
    evaluate: &mut F,
) -> Result<StationaryPoint, ScoreSearchError<E>>
where
    F: FnMut(f64) -> Result<ScoreJet, E>,
{
    // A unique-root refinement is only meaningful on a strict sign-change
    // bracket; anything else is a caller error surfaced as a typed rejection.
    if left.derivative == 0.0
        || right.derivative == 0.0
        || left.derivative.is_sign_positive() == right.derivative.is_sign_positive()
    {
        return Err(ScoreSearchError::InvalidEnclosure {
            lo: left.x,
            hi: right.x,
            enclosure,
        });
    }

    while right.x - left.x > resolution {
        let width = right.x - left.x;
        let midpoint = left.x + 0.5 * width;
        if !(midpoint > left.x && midpoint < right.x) {
            return Err(ScoreSearchError::Unresolved {
                lo: left.x,
                hi: right.x,
                requested_resolution: resolution,
                enclosure,
            });
        }

        // Newton is accepted only in the central half of the bracket.  Thus
        // every accepted point, Newton or midpoint, contracts the maintained
        // sign bracket by at least one quarter.  The loop has no iteration cap
        // because its geometric termination follows from this safeguard.
        let base = if left.derivative.abs() <= right.derivative.abs() {
            left
        } else {
            right
        };
        let newton = if base.curvature != 0.0 {
            base.x - base.derivative / base.curvature
        } else {
            f64::NAN
        };
        let guard = 0.25 * width;
        let x = if newton.is_finite() && newton >= left.x + guard && newton <= right.x - guard {
            newton
        } else {
            midpoint
        };
        if !(x > left.x && x < right.x) {
            return Err(ScoreSearchError::Unresolved {
                lo: left.x,
                hi: right.x,
                requested_resolution: resolution,
                enclosure,
            });
        }
        let sample = evaluate_sample(x, evaluate)?;
        if sample.derivative == 0.0 {
            return Ok(StationaryPoint {
                sample,
                bracket: ClosedInterval::point(x),
            });
        }
        if sample.derivative.is_sign_positive() == left.derivative.is_sign_positive() {
            left = sample;
        } else {
            right = sample;
        }
    }

    let midpoint = left.x + 0.5 * (right.x - left.x);
    let sample = if midpoint > left.x && midpoint < right.x {
        evaluate_sample(midpoint, evaluate)?
    } else if left.derivative.abs() <= right.derivative.abs() {
        left
    } else {
        right
    };
    Ok(StationaryPoint {
        sample,
        bracket: ClosedInterval::new(left.x, right.x),
    })
}

/// Globally maximize a smooth score on `[lo, hi]` by certified stationary
/// isolation.
///
/// `evaluate` returns the score and its first two analytic derivatives at a
/// point. `enclose(a, b)` must return OUTER ranges containing the first and
/// second derivative at every point of `[a, b]`.  The search additionally
/// checks that both endpoint jets lie inside every returned enclosure.
///
/// There is no evaluation or subdivision budget.  A successful return means
/// every stationary interval was either excluded or isolated to `resolution`.
/// Any interval that cannot be proved before that floor produces
/// [`ScoreSearchError::Unresolved`].
pub fn maximize_score_1d<E, Eval, Enclose>(
    lo: f64,
    hi: f64,
    resolution: f64,
    mut evaluate: Eval,
    mut enclose: Enclose,
) -> Result<ScoreSearchResult, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(f64, f64) -> Result<DerivativeEnclosure, E>,
{
    if !(lo.is_finite() && hi.is_finite() && lo <= hi && (hi - lo).is_finite()) {
        return Err(ScoreSearchError::InvalidDomain { lo, hi });
    }
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(ScoreSearchError::InvalidResolution { resolution });
    }

    let lower_boundary = evaluate_sample(lo, &mut evaluate)?;
    if lo == hi {
        return Ok(ScoreSearchResult {
            optimum: lower_boundary,
            location: ScoreOptimumLocation::LowerBoundary,
            lower_boundary,
            upper_boundary: lower_boundary,
            stationary_points: Vec::new(),
        });
    }
    let upper_boundary = evaluate_sample(hi, &mut evaluate)?;
    let (mut optimum, mut location) = if upper_boundary.value > lower_boundary.value {
        (upper_boundary, ScoreOptimumLocation::UpperBoundary)
    } else {
        (lower_boundary, ScoreOptimumLocation::LowerBoundary)
    };

    let mut stationary_points = Vec::<StationaryPoint>::new();
    let mut stack = vec![SearchNode {
        left: lower_boundary,
        right: upper_boundary,
    }];
    while let Some(node) = stack.pop() {
        let enclosure = checked_enclosure(node, &mut enclose)?;
        if !enclosure.derivative.contains_zero() {
            continue;
        }

        let monotone = !enclosure.curvature.contains_zero();
        if monotone {
            let stationary = if node.left.derivative == 0.0 {
                Some(StationaryPoint {
                    sample: node.left,
                    bracket: ClosedInterval::point(node.left.x),
                })
            } else if node.right.derivative == 0.0 {
                Some(StationaryPoint {
                    sample: node.right,
                    bracket: ClosedInterval::point(node.right.x),
                })
            } else if node.left.derivative.is_sign_positive()
                != node.right.derivative.is_sign_positive()
            {
                Some(refine_unique_root(
                    node.left,
                    node.right,
                    resolution,
                    enclosure,
                    &mut evaluate,
                )?)
            } else {
                None
            };

            if let Some(stationary) = stationary {
                // Two adjacent certified cells can report the same exact root
                // when it lies on their common boundary.  Preserve one copy.
                let duplicate = stationary_points
                    .last()
                    .is_some_and(|previous| previous.sample.x == stationary.sample.x);
                if !duplicate {
                    let index = stationary_points.len();
                    if stationary.sample.value > optimum.value {
                        optimum = stationary.sample;
                        location = ScoreOptimumLocation::Stationary(index);
                    }
                    stationary_points.push(stationary);
                }
            }
            continue;
        }

        let width = node.right.x - node.left.x;
        let midpoint = node.left.x + 0.5 * width;
        if width <= resolution || !(midpoint > node.left.x && midpoint < node.right.x) {
            return Err(ScoreSearchError::Unresolved {
                lo: node.left.x,
                hi: node.right.x,
                requested_resolution: resolution,
                enclosure,
            });
        }
        let middle = evaluate_sample(midpoint, &mut evaluate)?;
        // Right first, then left: the LIFO traversal emits stationary points
        // in ascending x, which makes exact-boundary de-duplication stable.
        stack.push(SearchNode {
            left: middle,
            right: node.right,
        });
        stack.push(SearchNode {
            left: node.left,
            right: middle,
        });
    }

    Ok(ScoreSearchResult {
        optimum,
        location,
        lower_boundary,
        upper_boundary,
        stationary_points,
    })
}

/// Static validation or evaluation failure for [`AffineRemlProfile`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AffineRemlError {
    EmptyModes,
    EmptyResponses,
    ShapeMismatch {
        gram_modes: usize,
        penalty_modes: usize,
        projected_rhs_squared: usize,
        responses: usize,
    },
    InvalidMode {
        index: usize,
        gram: f64,
        penalty: f64,
    },
    InvalidProjectedSquare {
        index: usize,
        value: f64,
    },
    InvalidResponseEnergy {
        output: usize,
        value: f64,
    },
    InvalidResidualDof {
        value: f64,
    },
    InvalidLogdetConstant {
        value: f64,
    },
    RankMismatch {
        supplied: usize,
        inferred: usize,
    },
    InvalidLogLambda {
        value: f64,
    },
    InvalidLogLambdaInterval {
        lo: f64,
        hi: f64,
    },
    NonPositiveMode {
        index: usize,
        log_lambda: f64,
        value: f64,
    },
    NonPositiveResidual {
        output: usize,
        log_lambda: f64,
        value: f64,
    },
    NonPositiveResidualInterval {
        output: usize,
        lo: f64,
        hi: f64,
        lower_bound: f64,
    },
}

impl fmt::Display for AffineRemlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModes => write!(f, "affine REML profile has no modes"),
            Self::EmptyResponses => write!(f, "affine REML profile has no responses"),
            Self::ShapeMismatch {
                gram_modes,
                penalty_modes,
                projected_rhs_squared,
                responses,
            } => write!(
                f,
                "affine REML profile shape mismatch: gram {gram_modes}, penalty {penalty_modes}, projected squares {projected_rhs_squared}, responses {responses}"
            ),
            Self::InvalidMode {
                index,
                gram,
                penalty,
            } => write!(
                f,
                "affine REML mode {index} must have finite nonnegative (g,s), not both zero; got ({gram}, {penalty})"
            ),
            Self::InvalidProjectedSquare { index, value } => write!(
                f,
                "affine REML projected square {index} must be finite and nonnegative, got {value}"
            ),
            Self::InvalidResponseEnergy { output, value } => write!(
                f,
                "affine REML response energy {output} must be finite and nonnegative, got {value}"
            ),
            Self::InvalidResidualDof { value } => {
                write!(
                    f,
                    "affine REML residual dof must be finite and positive, got {value}"
                )
            }
            Self::InvalidLogdetConstant { value } => write!(
                f,
                "affine REML log-determinant constant must be finite, got {value}"
            ),
            Self::RankMismatch { supplied, inferred } => write!(
                f,
                "affine REML determinant rank {supplied} disagrees with {inferred} positive penalty modes"
            ),
            Self::InvalidLogLambda { value } => {
                write!(f, "affine REML invalid log lambda {value}")
            }
            Self::InvalidLogLambdaInterval { lo, hi } => {
                write!(f, "affine REML invalid log-lambda interval [{lo}, {hi}]")
            }
            Self::NonPositiveMode {
                index,
                log_lambda,
                value,
            } => write!(
                f,
                "affine REML mode {index} is nonpositive at log lambda {log_lambda}: {value}"
            ),
            Self::NonPositiveResidual {
                output,
                log_lambda,
                value,
            } => write!(
                f,
                "affine REML residual {output} is nonpositive at log lambda {log_lambda}: {value}"
            ),
            Self::NonPositiveResidualInterval {
                output,
                lo,
                hi,
                lower_bound,
            } => write!(
                f,
                "affine REML residual {output} is not certified positive on [{lo}, {hi}] (lower bound {lower_bound})"
            ),
        }
    }
}

impl std::error::Error for AffineRemlError {}

/// Spectral REML/profile score with affine diagonal modes
/// `h_i(lambda) = g_i + lambda s_i`.
///
/// `projected_rhs_squared` is RESPONSE-MAJOR: entry `(d, i)` is stored at
/// `d * n_modes + i`.  The score is
///
/// `-1/2 { D [logdet_constant + sum log h_i - rank log(lambda)]
///          + residual_dof * sum_d log(R_d / residual_dof) }`,
///
/// where `R_d = response_energy[d] - sum_i q[d,i] / h_i`.
#[derive(Clone, Copy, Debug)]
pub struct AffineRemlProfile<'a> {
    gram_modes: &'a [f64],
    penalty_modes: &'a [f64],
    projected_rhs_squared: &'a [f64],
    response_energy: &'a [f64],
    residual_dof: f64,
    determinant_rank: usize,
    logdet_constant: f64,
}

impl<'a> AffineRemlProfile<'a> {
        pub fn new(
        gram_modes: &'a [f64],
        penalty_modes: &'a [f64],
        projected_rhs_squared: &'a [f64],
        response_energy: &'a [f64],
        residual_dof: f64,
        determinant_rank: usize,
        logdet_constant: f64,
    ) -> Result<Self, AffineRemlError> {
        let modes = gram_modes.len();
        let responses = response_energy.len();
        if modes == 0 {
            return Err(AffineRemlError::EmptyModes);
        }
        if responses == 0 {
            return Err(AffineRemlError::EmptyResponses);
        }
        if penalty_modes.len() != modes
            || projected_rhs_squared.len() != modes.saturating_mul(responses)
        {
            return Err(AffineRemlError::ShapeMismatch {
                gram_modes: modes,
                penalty_modes: penalty_modes.len(),
                projected_rhs_squared: projected_rhs_squared.len(),
                responses,
            });
        }
        for (index, (&gram, &penalty)) in gram_modes.iter().zip(penalty_modes).enumerate() {
            if !(gram.is_finite()
                && penalty.is_finite()
                && gram >= 0.0
                && penalty >= 0.0
                && (gram > 0.0 || penalty > 0.0))
            {
                return Err(AffineRemlError::InvalidMode {
                    index,
                    gram,
                    penalty,
                });
            }
        }
        for (index, &value) in projected_rhs_squared.iter().enumerate() {
            if !(value.is_finite() && value >= 0.0) {
                return Err(AffineRemlError::InvalidProjectedSquare { index, value });
            }
        }
        for (output, &value) in response_energy.iter().enumerate() {
            if !(value.is_finite() && value >= 0.0) {
                return Err(AffineRemlError::InvalidResponseEnergy { output, value });
            }
        }
        if !(residual_dof.is_finite() && residual_dof > 0.0) {
            return Err(AffineRemlError::InvalidResidualDof {
                value: residual_dof,
            });
        }
        if !logdet_constant.is_finite() {
            return Err(AffineRemlError::InvalidLogdetConstant {
                value: logdet_constant,
            });
        }
        let inferred_rank = penalty_modes.iter().filter(|&&value| value > 0.0).count();
        if determinant_rank != inferred_rank {
            return Err(AffineRemlError::RankMismatch {
                supplied: determinant_rank,
                inferred: inferred_rank,
            });
        }
        Ok(Self {
            gram_modes,
            penalty_modes,
            projected_rhs_squared,
            response_energy,
            residual_dof,
            determinant_rank,
            logdet_constant,
        })
    }

    #[inline]
    pub fn num_modes(&self) -> usize {
        self.gram_modes.len()
    }

    #[inline]
    pub fn num_responses(&self) -> usize {
        self.response_energy.len()
    }

    /// Exact score value, first derivative, and second derivative in
    /// `log(lambda)`.
    pub fn evaluate(&self, log_lambda: f64) -> Result<ScoreJet, AffineRemlError> {
        if !log_lambda.is_finite() {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }
        let lambda = log_lambda.exp();
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }

        let mut logdet = self.logdet_constant;
        let mut determinant_derivative = -(self.determinant_rank as f64);
        let mut determinant_curvature = 0.0;
        for (index, (&gram, &penalty)) in self.gram_modes.iter().zip(self.penalty_modes).enumerate()
        {
            let h = lambda.mul_add(penalty, gram);
            if !(h.is_finite() && h > 0.0) {
                return Err(AffineRemlError::NonPositiveMode {
                    index,
                    log_lambda,
                    value: h,
                });
            }
            let u = lambda * penalty / h;
            logdet += h.ln();
            determinant_derivative += u;
            determinant_curvature += u * (1.0 - u);
        }
        logdet -= (self.determinant_rank as f64) * log_lambda;

        let modes = self.num_modes();
        let mut residual_log_sum = 0.0;
        let mut residual_derivative_sum = 0.0;
        let mut residual_curvature_sum = 0.0;
        for (output, &energy) in self.response_energy.iter().enumerate() {
            let mut residual = energy;
            let mut first = 0.0;
            let mut second = 0.0;
            for i in 0..modes {
                let h = lambda.mul_add(self.penalty_modes[i], self.gram_modes[i]);
                let u = lambda * self.penalty_modes[i] / h;
                let projected_square = self.projected_rhs_squared[output * modes + i];
                residual -= projected_square / h;
                first += projected_square * u / h;
                second += projected_square * u * (1.0 - 2.0 * u) / h;
            }
            if !(residual.is_finite() && residual > 0.0) {
                return Err(AffineRemlError::NonPositiveResidual {
                    output,
                    log_lambda,
                    value: residual,
                });
            }
            let log_derivative = first / residual;
            residual_log_sum += (residual / self.residual_dof).ln();
            residual_derivative_sum += log_derivative;
            residual_curvature_sum += second / residual - log_derivative * log_derivative;
        }

        let outputs = self.num_responses() as f64;
        Ok(ScoreJet {
            value: -0.5 * (outputs * logdet + self.residual_dof * residual_log_sum),
            derivative: -0.5
                * (outputs * determinant_derivative + self.residual_dof * residual_derivative_sum),
            curvature: -0.5
                * (outputs * determinant_curvature + self.residual_dof * residual_curvature_sum),
        })
    }

    /// Outward enclosure of the first two score derivatives on a bounded
    /// log-lambda interval.
    pub fn enclose(&self, lo: f64, hi: f64) -> Result<DerivativeEnclosure, AffineRemlError> {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }
        let lambda = ClosedInterval::new(next_down(lo.exp()), next_up(hi.exp()));
        if !(lambda.lo.is_finite() && lambda.lo > 0.0 && lambda.hi.is_finite()) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }

        let mut determinant_first = ClosedInterval::point(0.0);
        let mut determinant_second = ClosedInterval::point(0.0);
        for i in 0..self.num_modes() {
            let ranges = mode_ranges(self.gram_modes[i], self.penalty_modes[i], 0.0, lambda);
            determinant_first = determinant_first.add(ranges.u);
            determinant_second = determinant_second.add(ranges.w);
        }
        determinant_first =
            determinant_first.sub(ClosedInterval::point(self.determinant_rank as f64));

        let mut residual_first_sum = ClosedInterval::point(0.0);
        let mut residual_second_sum = ClosedInterval::point(0.0);
        let modes = self.num_modes();
        for (output, &energy) in self.response_energy.iter().enumerate() {
            let mut fitted_quadratic = ClosedInterval::point(0.0);
            let mut first = ClosedInterval::point(0.0);
            let mut second = ClosedInterval::point(0.0);
            for i in 0..modes {
                let ranges = mode_ranges(
                    self.gram_modes[i],
                    self.penalty_modes[i],
                    self.projected_rhs_squared[output * modes + i],
                    lambda,
                );
                fitted_quadratic = fitted_quadratic.add(ranges.v);
                first = first.add(ranges.p);
                second = second.add(ranges.q);
            }
            let residual = ClosedInterval::point(energy).sub(fitted_quadratic);
            if !(residual.lo > 0.0 && residual.is_valid()) {
                return Err(AffineRemlError::NonPositiveResidualInterval {
                    output,
                    lo,
                    hi,
                    lower_bound: residual.lo,
                });
            }
            let first_ratio = first.div_positive(residual).nonnegative();
            let second_ratio = second.div_positive(residual);
            residual_first_sum = residual_first_sum.add(first_ratio);
            residual_second_sum = residual_second_sum.add(second_ratio.sub(first_ratio.square()));
        }

        let outputs = self.num_responses() as f64;
        let first_bracket = determinant_first
            .scale(outputs)
            .add(residual_first_sum.scale(self.residual_dof));
        let second_bracket = determinant_second
            .scale(outputs)
            .add(residual_second_sum.scale(self.residual_dof));
        Ok(DerivativeEnclosure {
            derivative: first_bracket.scale(-0.5),
            curvature: second_bracket.scale(-0.5),
        })
    }

    pub fn maximize(
        &self,
        lo: f64,
        hi: f64,
        resolution: f64,
    ) -> Result<ScoreSearchResult, ScoreSearchError<AffineRemlError>> {
        maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| self.evaluate(x),
            |a, b| self.enclose(a, b),
        )
    }
}

#[derive(Clone, Copy)]
struct ModeRanges {
    /// `u = lambda s / h`.
    u: ClosedInterval,
    /// `u(1-u)`.
    w: ClosedInterval,
    /// `projected_square / h`.
    v: ClosedInterval,
    /// First derivative of the residual contribution:
    /// `projected_square * lambda s / h^2`.
    p: ClosedInterval,
    /// Second derivative of the residual contribution:
    /// `projected_square * lambda s (g-lambda s) / h^3`.
    q: ClosedInterval,
}

fn mode_ranges(
    gram: f64,
    penalty: f64,
    projected_square: f64,
    lambda: ClosedInterval,
) -> ModeRanges {
    if penalty == 0.0 {
        let v = ClosedInterval::point(projected_square)
            .div_positive(ClosedInterval::point(gram))
            .nonnegative();
        return ModeRanges {
            u: ClosedInterval::point(0.0),
            w: ClosedInterval::point(0.0),
            v,
            p: ClosedInterval::point(0.0),
            q: ClosedInterval::point(0.0),
        };
    }
    if gram == 0.0 {
        let h = lambda.mul(ClosedInterval::point(penalty)).nonnegative();
        let v = ClosedInterval::point(projected_square)
            .div_positive(h)
            .nonnegative();
        return ModeRanges {
            u: ClosedInterval::point(1.0),
            w: ClosedInterval::point(0.0),
            v,
            p: v,
            q: v.neg(),
        };
    }

    // Normalize by g: h = g(1+t), t = lambda*s/g.  The four kernels below
    // have known global critical points, so endpoint evaluation plus any
    // critical point contained by the t-window gives an exact real range;
    // interval arithmetic rounds every primitive outward.
    let t = lambda
        .mul(ClosedInterval::point(penalty))
        .div_positive(ClosedInterval::point(gram))
        .nonnegative();
    let scale = ClosedInterval::point(projected_square)
        .div_positive(ClosedInterval::point(gram))
        .nonnegative();
    let kernels = kernel_ranges(t);
    ModeRanges {
        u: kernels.u,
        w: kernels.w,
        v: scale.mul(kernels.v).nonnegative(),
        p: scale.mul(kernels.w).nonnegative(),
        q: scale.mul(kernels.k),
    }
}

#[derive(Clone, Copy)]
struct KernelRanges {
    /// `t/(1+t)`.
    u: ClosedInterval,
    /// `1/(1+t)`.
    v: ClosedInterval,
    /// `t/(1+t)^2`.
    w: ClosedInterval,
    /// `t(1-t)/(1+t)^3`.
    k: ClosedInterval,
}

fn kernel_at(t: ClosedInterval) -> KernelRanges {
    let one = ClosedInterval::point(1.0);
    let denom = one.add(t);
    let v = one.div_positive(denom).nonnegative();
    let u = t.mul(v).nonnegative();
    let w = u.mul(v).nonnegative();
    let k = w.mul(one.sub(t)).div_positive(denom);
    KernelRanges { u, v, w, k }
}

fn kernel_ranges(t: ClosedInterval) -> KernelRanges {
    let left = kernel_at(ClosedInterval::point(t.lo));
    let right = kernel_at(ClosedInterval::point(t.hi));
    let mut u = ClosedInterval::new(left.u.lo, right.u.hi).nonnegative();
    let mut v = ClosedInterval::new(right.v.lo, left.v.hi).nonnegative();
    let mut w = left.w.hull(right.w).nonnegative();
    let mut k = left.k.hull(right.k);

    if t.contains(1.0) {
        let critical = kernel_at(ClosedInterval::point(1.0));
        w = w.hull(critical.w).nonnegative();
    }

    // k'(t) has its only positive roots at 2 +/- sqrt(3).  Enclose sqrt(3)
    // itself before subtraction/addition so the exact irrational critical
    // points are not lost to nearest-rounded scalar arithmetic.
    let sqrt_three = ClosedInterval::new(next_down(3.0_f64.sqrt()), next_up(3.0_f64.sqrt()));
    let critical_points = [
        ClosedInterval::point(2.0).sub(sqrt_three),
        ClosedInterval::point(2.0).add(sqrt_three),
    ];
    for critical in critical_points {
        if critical.hi >= t.lo && critical.lo <= t.hi {
            k = k.hull(kernel_at(critical).k);
        }
    }

    // Monotonicity gives tighter endpoint ranges than a dependency-heavy
    // interval evaluation, but retain outward endpoint arithmetic.
    u.lo = u.lo.max(0.0);
    u.hi = u.hi.min(next_up(1.0));
    v.lo = v.lo.max(0.0);
    v.hi = v.hi.min(next_up(1.0));
    KernelRanges { u, v, w, k }
}

/// Next representable number below `value`, used for directed outward
/// rounding of interval lower bounds.
fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

/// Next representable number above `value`, used for directed outward
/// rounding of interval upper bounds.
fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polynomial_hidden_bump_jet(x: f64) -> ScoreJet {
        let p = x * (x - 0.5) * (x - 1.0);
        let dp = 3.0 * x * x - 3.0 * x + 0.5;
        let ddp = 6.0 * x - 3.0;
        ScoreJet {
            value: x + 1000.0 * p * p,
            derivative: 1.0 + 2000.0 * p * dp,
            curvature: 2000.0 * (dp * dp + p * ddp),
        }
    }

    fn polynomial_hidden_bump_enclosure(lo: f64, hi: f64) -> DerivativeEnclosure {
        let x = ClosedInterval::new(lo, hi);
        let p = x
            .mul(x.sub(ClosedInterval::point(0.5)))
            .mul(x.sub(ClosedInterval::point(1.0)));
        let dp = x
            .square()
            .scale(3.0)
            .sub(x.scale(3.0))
            .add(ClosedInterval::point(0.5));
        let ddp = x.scale(6.0).sub(ClosedInterval::point(3.0));
        DerivativeEnclosure {
            derivative: ClosedInterval::point(1.0).add(p.mul(dp).scale(2000.0)),
            curvature: dp.square().add(p.mul(ddp)).scale(2000.0),
        }
    }

    #[test]
    fn hidden_between_endpoint_and_midpoint_samples_is_found() {
        let result = maximize_score_1d(
            0.0,
            1.0,
            1.0e-9,
            |x| -> Result<_, String> { Ok(polynomial_hidden_bump_jet(x)) },
            |lo, hi| -> Result<_, String> { Ok(polynomial_hidden_bump_enclosure(lo, hi)) },
        )
        .expect("certified search");

        // At x=0, 1/2, 1 both value and derivative agree exactly with f=x;
        // the former midpoint/Hermite heuristic therefore returned x=1.
        assert_eq!(polynomial_hidden_bump_jet(0.0).derivative, 1.0);
        assert_eq!(polynomial_hidden_bump_jet(0.5).derivative, 1.0);
        assert_eq!(polynomial_hidden_bump_jet(1.0).derivative, 1.0);
        assert!(result.optimum.x > 0.5 && result.optimum.x < 1.0);
        assert!(result.optimum.value > 2.9);
        assert_eq!(result.stationary_points.len(), 4);
    }

    fn quartic_jet(x: f64) -> ScoreJet {
        ScoreJet {
            value: -(x * x - 1.0).powi(2),
            derivative: 4.0 * x - 4.0 * x * x * x,
            curvature: 4.0 - 12.0 * x * x,
        }
    }

    fn quartic_enclosure(lo: f64, hi: f64) -> DerivativeEnclosure {
        let x = ClosedInterval::new(lo, hi);
        DerivativeEnclosure {
            derivative: x.scale(4.0).sub(x.mul(x).mul(x).scale(4.0)),
            curvature: ClosedInterval::point(4.0).sub(x.square().scale(12.0)),
        }
    }

    #[test]
    fn multiple_roots_in_initial_bracket_are_all_isolated() {
        let result = maximize_score_1d(
            -2.0,
            2.0,
            1.0e-10,
            |x| -> Result<_, String> { Ok(quartic_jet(x)) },
            |lo, hi| -> Result<_, String> { Ok(quartic_enclosure(lo, hi)) },
        )
        .expect("certified search");
        assert_eq!(result.stationary_points.len(), 3);
        for (point, expected) in result.stationary_points.iter().zip([-1.0_f64, 0.0, 1.0]) {
            assert!((point.sample.x - expected).abs() <= 1.0e-9);
            assert!(point.bracket.hi - point.bracket.lo <= 1.0e-10);
        }
        assert!((result.optimum.x.abs() - 1.0).abs() <= 1.0e-9);
    }

    #[test]
    fn monotone_score_selects_exact_boundary() {
        let result = maximize_score_1d(
            -4.0,
            9.0,
            1.0e-9,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 0.3 * x,
                    derivative: 0.3,
                    curvature: 0.0,
                })
            },
            |_, _| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    derivative: ClosedInterval::point(0.3),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("certified search");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 9.0);
        assert!(result.stationary_points.is_empty());
    }

    #[test]
    fn unresolved_tangential_stationary_point_is_typed() {
        let error = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-8,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x * x * x,
                    derivative: 3.0 * x * x,
                    curvature: 6.0 * x,
                })
            },
            |lo, hi| -> Result<_, String> {
                let x = ClosedInterval::new(lo, hi);
                Ok(DerivativeEnclosure {
                    derivative: x.square().scale(3.0),
                    curvature: x.scale(6.0),
                })
            },
        )
        .expect_err("a tangential root needs stronger structural bounds");
        assert!(matches!(error, ScoreSearchError::Unresolved { .. }));
    }

    fn affine_fixture() -> AffineRemlProfile<'static> {
        const G: &[f64] = &[2.0, 0.5, 0.0, 3.0];
        const S: &[f64] = &[1.0, 0.0, 2.0, 0.25];
        const Q: &[f64] = &[
            0.6, 0.1, 0.02, 0.3, // response 0
            0.2, 0.4, 0.01, 0.5, // response 1
        ];
        const Y2: &[f64] = &[8.0, 10.0];
        AffineRemlProfile::new(G, S, Q, Y2, 12.0, 3, 0.7).expect("valid fixture")
    }

    #[test]
    fn affine_reml_jet_matches_test_only_differences() {
        let profile = affine_fixture();
        for x in [-2.0_f64, -0.4, 0.7, 2.0] {
            let h = 1.0e-5;
            let center = profile.evaluate(x).unwrap();
            let left = profile.evaluate(x - h).unwrap();
            let right = profile.evaluate(x + h).unwrap();
            let derivative = (right.value - left.value) / (2.0 * h);
            let curvature = (right.derivative - left.derivative) / (2.0 * h);
            assert!(
                (center.derivative - derivative).abs() <= 2.0e-8 * (1.0 + derivative.abs()),
                "first derivative mismatch at {x}: analytic {}, difference {derivative}",
                center.derivative
            );
            assert!(
                (center.curvature - curvature).abs() <= 2.0e-8 * (1.0 + curvature.abs()),
                "curvature mismatch at {x}: analytic {}, difference {curvature}",
                center.curvature
            );
        }
    }

    #[test]
    fn affine_reml_enclosure_contains_value_jets() {
        let profile = affine_fixture();
        let enclosure = profile.enclose(-2.5, 1.75).expect("enclosure");
        for x in [-2.5_f64, -1.7, -0.3, 0.0, 0.9, 1.75] {
            let jet = profile.evaluate(x).unwrap();
            assert!(
                enclosure.derivative.contains(jet.derivative),
                "gradient {} at {x} outside {:?}",
                jet.derivative,
                enclosure.derivative
            );
            assert!(
                enclosure.curvature.contains(jet.curvature),
                "curvature {} at {x} outside {:?}",
                jet.curvature,
                enclosure.curvature
            );
        }
    }

    #[test]
    fn affine_reml_rejects_nonpositive_profile_residual() {
        let profile = AffineRemlProfile::new(&[1.0], &[1.0], &[2.0], &[1.0], 4.0, 1, 0.0)
            .expect("statically valid");
        assert!(matches!(
            profile.evaluate(-2.0),
            Err(AffineRemlError::NonPositiveResidual { .. })
        ));
    }
}
