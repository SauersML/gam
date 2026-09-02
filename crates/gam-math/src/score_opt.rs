//! Certified global optimization of one-dimensional scores on a bounded
//! domain, together with the affine-pencil spectral profile shared by the
//! Gaussian REML smoothing-parameter searches.
//!
//! Point samples alone cannot prove that a smooth function has no narrow
//! stationary pair between them.  The search therefore requires two pieces of
//! information from its caller:
//!
//! * a nearest-rounded point evaluation `(value, first derivative, second
//!   derivative)`, used only as a representative and to propose refinements;
//! * an OUTER enclosure of the exact score value and both exact derivatives
//!   over every requested interval, accompanied by a certified forward-error
//!   bound for the scalar score evaluator.
//!
//! A cell needs no stationary decomposition when its derivative enclosure
//! excludes zero or its exact score upper bound is strictly below an attained
//! point-score lower bound. A stationary point is refined only after the second-derivative
//! enclosure excludes zero, proving that the first derivative is monotone and
//! hence that certified endpoint derivative ranges of opposite sign contain
//! exactly one root. Every other interval is subdivided unless the exact score
//! maximum is indistinguishable from an evaluated representative at the score
//! evaluator's certified forward-error floor. Such a region is returned
//! explicitly as a [`ResolutionFlatRegion`]; it is never mislabeled as a
//! stationary point. This includes both a value-flat cell and a strictly concave
//! cell whose unique maximum is already closer to the representative than the
//! evaluator can resolve. A
//! cell whose exact score upper bound is below an already attained exact
//! point-score lower bound is retained as a [`DominatedRegion`] and needs no
//! stationary decomposition: none of its structure can affect the global
//! maximum. If neither exclusion, isolation, score-value flatness, nor exact
//! dominance is proved before the requested abscissa resolution, the result is
//! a typed [`ScoreSearchError::Unresolved`] rather than a best-effort optimum.
//!
//! [`AffineRemlProfile`] supplies both the point jets and rigorous interval
//! formulas for scores whose penalized Hessian has simultaneously diagonal
//! affine modes `h_i(lambda) = g_i + lambda s_i`.  This covers an ordinary
//! Demmler--Reinsch eigensystem (`g_i = 1`) and a reference-Hessian pencil
//! (`g_i = 1 - lambda_0 mu_i`, `s_i = mu_i`) without any matrix dependency in
//! this crate.
//!
//! # The enclosure has to COLLAPSE, not merely be correct
//!
//! Everything above is a statement about what the search does with an
//! enclosure; none of it says how tight one has to be, and the difference
//! decides whether a domain can be decomposed at all. Every terminal verdict —
//! derivative exclusion, stationary isolation, score-value flatness, exact
//! dominance — is a comparison between an enclosure and a fixed quantity, so an
//! enclosure whose overestimation is FIRST ORDER in the cell width buys a
//! constant factor of resolution per subdivision, and the search enumerates
//! cells until its budget is gone.
//!
//! That is not hypothetical: [`AffineRemlProfile::enclose`] was a natural
//! interval extension, and on a REML score — whose log-determinant and deviance
//! blocks each move by `O(rank)` per unit of `log lambda` while their sum does
//! not — it returned a value range of exactly `rank * width`, up to `7.4e5`
//! times wider than the cell's own derivative enclosure permitted, and refused
//! designs it could certify. It is now a centred (mean value) form intersected
//! with the natural one, in all three channels; see that method for the
//! identity, the measurements, and what the centring is anchored on.

use std::fmt;
use std::sync::OnceLock;

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
    fn intersection(self, other: Self) -> Option<Self> {
        let intersection = Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        };
        (intersection.lo <= intersection.hi).then_some(intersection)
    }

    #[inline]
    fn max_abs(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    #[inline]
    fn widen(self, radius: f64) -> Self {
        if radius == 0.0 {
            return self;
        }
        if radius == f64::INFINITY {
            return Self::entire();
        }
        Self {
            lo: next_down(self.lo - radius),
            hi: next_up(self.hi + radius),
        }
    }

    #[inline]
    /// Directed outer enclosure of the exact sum of two intervals.
    pub fn add(self, other: Self) -> Self {
        Self {
            lo: sum_down(self.lo, other.lo),
            hi: sum_up(self.hi, other.hi),
        }
    }

    #[inline]
    /// Directed outer enclosure of the exact interval difference.
    pub fn sub(self, other: Self) -> Self {
        Self {
            lo: sum_down(self.lo, -other.hi),
            hi: sum_up(self.hi, -other.lo),
        }
    }

    #[inline]
    /// Exact sign reversal of the interval.
    pub fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    /// Directed outer enclosure of the exact interval product.
    pub fn mul(self, other: Self) -> Self {
        let pairs = [
            (self.lo, other.lo),
            (self.lo, other.hi),
            (self.hi, other.lo),
            (self.hi, other.hi),
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (left, right) in pairs {
            lo = lo.min(product_down(left, right));
            hi = hi.max(product_up(left, right));
        }
        Self { lo, hi }
    }

    #[inline]
    /// Directed outer enclosure after multiplication by an exact binary64
    /// scalar.
    pub fn scale(self, value: f64) -> Self {
        self.mul(Self::point(value))
    }

    fn square(self) -> Self {
        if self.lo >= 0.0 {
            Self {
                lo: product_down(self.lo, self.lo).max(0.0),
                hi: product_up(self.hi, self.hi),
            }
        } else if self.hi <= 0.0 {
            Self {
                lo: product_down(self.hi, self.hi).max(0.0),
                hi: product_up(self.lo, self.lo),
            }
        } else {
            Self {
                lo: 0.0,
                hi: product_up(self.lo, self.lo).max(product_up(self.hi, self.hi)),
            }
        }
    }

    /// Natural logarithm of an interval known to be strictly positive.
    fn ln_positive(self) -> Self {
        assert!(
            self.lo > 0.0,
            "ln_positive requires a strictly positive interval, got lo={}",
            self.lo
        );
        let lo = certified_ln_positive(self.lo)
            .expect("ln_positive lower endpoint is finite and positive");
        let hi = certified_ln_positive(self.hi)
            .expect("ln_positive upper endpoint is finite and positive");
        Self::new(lo.lo, hi.hi)
    }

    /// Divide by an interval known to be strictly positive.
    fn div_positive(self, denominator: Self) -> Self {
        assert!(
            denominator.lo > 0.0,
            "div_positive requires a strictly positive denominator interval, got lo={}",
            denominator.lo
        );
        let reciprocal = Self {
            lo: quotient_down(1.0, denominator.hi).max(0.0),
            hi: quotient_up(1.0, denominator.lo),
        };
        self.mul(reciprocal)
    }

    /// Divide by an interval that excludes zero.
    fn div_nonzero(self, denominator: Self) -> Self {
        if denominator.lo > 0.0 {
            self.div_positive(denominator)
        } else {
            assert!(
                denominator.hi < 0.0,
                "div_nonzero requires a denominator interval excluding zero, got {denominator:?}"
            );
            self.div_positive(denominator.neg()).neg()
        }
    }

    #[inline]
    fn nonnegative(self) -> Self {
        Self {
            lo: self.lo.max(0.0),
            hi: self.hi.max(0.0),
        }
    }
}

/// Nearest-rounded value and analytic derivatives at one abscissa.
///
/// `third` is carried alongside the first two because every endpoint-anchored
/// [`DerivativeEnclosure`] in this workspace is built from the endpoint
/// curvature and third derivative. Dropping it here used to force the enclosure
/// oracle to RE-EVALUATE the criterion at both endpoints of every
/// branch-and-bound cell — endpoints the search had already sampled — which
/// tripled the number of criterion evaluations the search actually paid for.
/// Oracles that have no third derivative to report set it to zero; enclosures
/// that do not consult it are unaffected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreJet {
    pub value: f64,
    pub derivative: f64,
    pub curvature: f64,
    pub third: f64,
}

/// A point evaluation augmented with its abscissa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreSample {
    pub x: f64,
    pub value: f64,
    pub derivative: f64,
    pub curvature: f64,
    pub third: f64,
}

/// Exact score-value range and the numerical resolution of point values.
///
/// `value` contains the exact-real score at every point of the cell.
/// `evaluation_error` is an absolute forward-error bound for both endpoint
/// values supplied with that cell:
///
/// `|endpoint.value - exact_score(endpoint.x)| <= evaluation_error`.
///
/// An interval-extension oracle may provide the stronger cell-uniform bound.
/// The search needs only the endpoint statement: every representative it
/// retains is an evaluated cell endpoint. The corresponding uncertainty of a
/// comparison between the two endpoints is at most `2 * evaluation_error`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreValueEnclosure {
    pub value: ClosedInterval,
    pub evaluation_error: f64,
}

/// Exact-real score and derivative ranges supplied to the certified search.
///
/// Scalar derivative estimates are proposals only. Exclusion, monotonicity,
/// and root-sign decisions use these mathematical ranges directly, so
/// derivative-evaluator roundoff never becomes part of a proof predicate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivativeEnclosure {
    pub score: ScoreValueEnclosure,
    pub derivative: ClosedInterval,
    pub curvature: ClosedInterval,
}

/// A region whose exact maximum is indistinguishable from an evaluated
/// representative at the representable resolution of its score.
///
/// `max_score_gap` bounds `max(score over bracket) - score(sample.x)` in exact
/// score units. `score_resolution` is the certified forward-error scale at
/// which that improvement is numerically immaterial. The search records this
/// region only when `max_score_gap <= score_resolution`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolutionFlatRegion {
    pub sample: ScoreSample,
    pub bracket: ClosedInterval,
    /// Exact score range over `bracket`.
    pub score: ClosedInterval,
    pub max_score_gap: f64,
    pub score_resolution: f64,
}

/// One stationary point together with the final bracket that certifies its
/// location.  The bracket width is no larger than the requested resolution,
/// unless the point was represented exactly (a zero-width bracket).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationaryPoint {
    pub sample: ScoreSample,
    pub bracket: ClosedInterval,
    /// Exact score range over `bracket` and endpoint evaluation resolution.
    pub score: ScoreValueEnclosure,
    /// Strict curvature enclosure that proved the derivative root unique.
    ///
    /// This may be tighter than a fresh enclosure on the final tiny bracket:
    /// cancellation can erase a sign under subdivision even though the wider
    /// parent certificate remains valid on every subset.
    pub curvature: ClosedInterval,
}

/// Exact-value certificate for the representative selected by the rounded
/// evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalScoreCertificate {
    /// Exact score at the returned representative.
    pub selected: ClosedInterval,
    /// Outer range containing the exact global maximum.
    pub maximum: ClosedInterval,
    /// Outward bound on `global maximum - exact score(representative)`.
    /// Repeated certificates of the same represented point contribute zero:
    /// they name the same exact real value, rather than independent uncertain
    /// quantities.
    pub maximum_excess: f64,
    /// Outward sum of the selected point evaluator's forward error and the
    /// largest competing representative's forward error. Exact terminal
    /// ranges remain separate in [`Self::maximum_excess`].
    pub comparison_resolution: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreOptimumLocation {
    LowerBoundary,
    UpperBoundary,
    Stationary(usize),
    ResolutionFlat(usize),
}

/// A cell excluded from the global maximum by exact score ordering.
///
/// `score.hi < incumbent_lower` proves every exact score in `bracket` is below
/// an exact score already attained at an evaluated point. Stationary structure
/// inside the cell is therefore irrelevant to the global maximum, but the
/// region is retained so that this branch-and-bound decision remains auditable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DominatedRegion {
    pub bracket: ClosedInterval,
    pub score: ScoreValueEnclosure,
    pub incumbent_lower: f64,
}

/// Complete successful search result. Endpoints, isolated stationary points,
/// resolution-flat regions, and exactly dominated regions are retained
/// explicitly so every terminal proof is independently checkable by the
/// caller.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSearchResult {
    pub optimum: ScoreSample,
    pub location: ScoreOptimumLocation,
    pub lower_boundary: ScoreSample,
    pub upper_boundary: ScoreSample,
    pub stationary_points: Vec<StationaryPoint>,
    pub resolution_flat_regions: Vec<ResolutionFlatRegion>,
    /// Pairwise-disjoint terminal cells. A binary tree with at most `B`
    /// subdivisions has at most `B + 1` leaves, so this audit is bounded by
    /// the same [`subdivision_budget`] as the traversal.
    pub dominated_regions: Vec<DominatedRegion>,
    pub value_certificate: GlobalScoreCertificate,
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
    ScoreValueEnclosureMissesEndpoint {
        lo: f64,
        hi: f64,
        endpoint: ScoreSample,
        score: ScoreValueEnclosure,
    },
    DisjointEndpointEnclosure {
        lo: f64,
        hi: f64,
        endpoint: ScoreSample,
        endpoint_derivative: ClosedInterval,
        enclosure: DerivativeEnclosure,
    },
    /// Independent interval-Newton images of a root that was already proved
    /// unique have empty intersection. This is a contradiction between
    /// certificates, not an unresolved search cell.
    InconsistentRootEnclosure {
        lo: f64,
        hi: f64,
        left_derivative: ClosedInterval,
        right_derivative: ClosedInterval,
        curvature: ClosedInterval,
        left_newton: ClosedInterval,
        right_newton: ClosedInterval,
        point_newton: ClosedInterval,
    },
    /// Neither stationary exclusion/isolation nor score flatness could be
    /// proved before the requested or floating-point abscissa-resolution
    /// floor.
    Unresolved {
        lo: f64,
        hi: f64,
        requested_resolution: f64,
        enclosure: DerivativeEnclosure,
    },
    /// The traversal asked for more cell subdivisions than
    /// [`subdivision_budget`] allows for this domain and resolution. Reported
    /// with the cell that was being split when the budget ran out, so the
    /// caller can see WHERE the criterion stopped being decomposable, and with
    /// that cell's enclosure, so the caller can see WHETHER a larger budget
    /// could ever have helped: once the certified evaluation error reaches the
    /// requested resolution, no amount of subdivision separates stationary
    /// structure at that tolerance (#2614).
    SubdivisionBudget {
        lo: f64,
        hi: f64,
        cell_lo: f64,
        cell_hi: f64,
        requested_resolution: f64,
        subdivisions: usize,
        budget: usize,
        depth_bound: u32,
        enclosure: DerivativeEnclosure,
    },
}

/// Total cell subdivisions a converging certified 1-D search may spend on
/// `[lo, hi]` at `resolution`.
///
/// Two facts set the scale. First, no cell can be halved more than
/// `D = ceil(log2((hi - lo) / resolution))` times before it is narrower than
/// `resolution`, where the search already stops with
/// [`ScoreSearchError::Unresolved`] — so `D` bounds the depth of the
/// subdivision tree outright. Second, a search that is ISOLATING structure
/// spends at most `D` subdivisions per cell it finally certifies, because each
/// one halves the cell it is working in.
///
/// So the whole traversal costs at most `D` times the size of the certified
/// decomposition, and the budget is that product with the decomposition
/// allowed `2 D` cells — twice as many certified cells as the domain has
/// resolvable binary levels. Measured on #2546's cascade: every terminating
/// search on that surface spent 33–39 subdivisions at `D = 32`, i.e. about `D`,
/// against a budget of `2 D² = 2048`; the non-terminating one passes 40 000
/// with its bracket still halving cleanly at every node. The margin over the
/// deepest currently-successful search is ~60x, so the budget is invisible to
/// every search that converges and is reached in under a second by one that
/// does not.
///
/// gam#2614 — that ~60x margin is NOT general, and the `2 D` cell allowance is
/// the reason. The `D` factor is derived: no cell survives more than `D`
/// halvings before it is narrower than `resolution`. The cell allowance is an
/// assumption about how many cells a criterion's certified decomposition
/// contains, which is exactly what the search cannot know in advance.
///
/// Read the calibration above again: spending about `D` subdivisions IN TOTAL,
/// at `D` per certified cell, means that surface's decomposition was about ONE
/// cell. The `2 D` allowance (64 cells at `D = 32`) was never exercised there,
/// so the quoted margin is headroom over a single-cell case.
///
/// Measured since, at the same `D = 32`, which is why the multiplier below is 8
/// and not 2. Bisected across the FULL `spline_scan` set:
///
/// ```text
///   2 D² (  64 cells)  order_one_scan_matches_dense_random_walk_posterior refused
///   4 D² ( 128 cells)  passes
///   8 D² ( 256 cells)  passes          <- shipped
/// 128 D² (4096 cells)  passes, and NO further test passes
/// ```
///
/// That search is not going deeper than `D` per cell — the depth bound is a hard
/// geometric fact. It isolates structure over a decomposition of 65–128 cells,
/// wider than the `2 D` allowance anticipated, so against that surface the older
/// "~60x margin" was negative. Nothing above `8 D²` buys another passing test.
///
/// A larger allowance does NOT repair the other scan refusals, and raising it
/// past this point actively HIDES their cause. At `128 D²` the budget message
/// disappears entirely and
/// `state_snapshot_round_trips_predict_and_training_sample_size_bit_for_bit` —
/// which exhausted the budget at both 2048 and 8192 — instead reports
/// `OptimumResolutionFlat` on a bracket `2.15e-6` wide, about twice the endpoint
/// `eval_err` of `~9.6e-7`. The budget was masking a score-RESOLUTION floor.
/// Likewise `weighted_scan_dgp_2300_search_terminates_in_bounded_evaluations`
/// fails identically at every multiplier tested, because its certificates carry
/// `eval_err ~1e-6` while the search requests `resolution = 1.49e-8`.
///
/// So: three of the four `spline_scan` failures are evaluation-conditioning, not
/// cell shortage, and no allowance reaches them. Do not raise this constant
/// further expecting it to fix them — it converts a budget refusal into a
/// resolution refusal and gains no coverage.
///
/// A degenerate domain still gets a budget of at least one subdivision: the
/// bound is a backstop against unbounded breadth, never a refusal of the first
/// split.
///
/// # The request, not the budget, is what actually binds (#2614, measured 0731)
///
/// Callers pass `f64::EPSILON.sqrt()` (`1.4901161193847656e-8`) as the requested
/// resolution — a MACHINE constant. The achievable certified evaluation error is
/// a property of the problem and varies by more than forty times between callers:
///
/// | caller | `evaluation_error` | `requested_resolution` | terminal bracket |
/// |---|---|---|---|
/// | `gam-solve` spline_scan | `9.741e-7` | `1.490e-8` | — |
/// | `gam-predict` weighted scan | `2.302e-8` | `1.490e-8` | `~4.8e-8` ≈ 2 × eval_err |
///
/// In both the error EXCEEDS the request, and `gam-predict` terminates at almost
/// exactly twice its own evaluation error — the floor you would predict, since
/// inside that width the endpoint enclosures overlap and no comparison is
/// decidable. In both the derivative enclosure straddles zero, so even the sign
/// of the slope is undecidable there.
///
/// A fixed `sqrt(EPSILON)` request cannot be right for both. The resolution
/// should be derived from the evaluator's certified error at the working point
/// rather than pinned to a machine constant.
///
/// # That recommendation is NOT the repair for a budget refusal, measured
///
/// It was taken as one, and the discriminator says otherwise. On a rank-deficient
/// cascade design (36 rows, 1725 columns, 33 modes on a 40.6-wide domain) that
/// refused here at 8193/8192 subdivisions, the same search was run at four
/// requested resolutions spanning five orders — `1.49e-8`, `1e-6`, `1e-4`,
/// `1e-3` — and **every one refused**, the terminal cell merely walking down the
/// domain (`-16.79`, `-18.51`, `-20.08`, `-20.59`) as the request coarsened.
/// Matching the request to the evaluator's error bought nothing there.
///
/// What bound that search was the ENCLOSURE, one level down: the score's value
/// range was a natural interval extension, first order in the cell width with
/// constant `rank` (`33.0·w`, over six decades) against an exact `|f'|` of
/// `1.15e-5`. `resolution_flat_region` reads that range, so no cell could be
/// retired at any request. With the centred form in
/// [`AffineRemlProfile::enclose`] the same design certifies in 0.4 s at every
/// one of those four requests.
///
/// The two `spline_scan` refusals named above also both PASS now
/// (`cargo test -p gam-solve --lib`, 1898 of 1901, and neither is among the
/// three reds). So the table above stands as a measurement of a real
/// mismatch — a machine constant is still the wrong source for a
/// problem-dependent tolerance — but the failures it was written to explain are
/// gone, and it should not be cited as the cause of a fresh one without a
/// discriminator like the ladder above.
pub fn subdivision_budget(lo: f64, hi: f64, resolution: f64) -> (usize, u32) {
    let width = hi - lo;
    if !(width.is_finite() && width > 0.0 && resolution.is_finite() && resolution > 0.0) {
        return (1, 0);
    }
    let levels = (width / resolution).log2().ceil();
    let depth_bound = if levels.is_finite() && levels >= 1.0 {
        // `f64::MANTISSA_DIGITS`-scaled domains cannot exceed the exponent
        // range, so the cast is saturating in practice and clamped in fact.
        levels.min(u32::MAX as f64) as u32
    } else {
        1
    };
    let depth = depth_bound as usize;
    (8 * depth * depth, depth_bound)
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
                "score search: score/derivative enclosure failed on [{lo}, {hi}]: {source}"
            ),
            Self::NonFiniteSample { sample } => write!(
                f,
                "score search: non-finite jet at {} (value {}, derivative {}, curvature {}, third {})",
                sample.x, sample.value, sample.derivative, sample.curvature, sample.third
            ),
            Self::InvalidEnclosure { lo, hi, enclosure } => write!(
                f,
                "score search: invalid score/derivative enclosure on [{lo}, {hi}]: {enclosure:?}"
            ),
            Self::ScoreValueEnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                score,
            } => write!(
                f,
                "score search: exact score range {:?} plus evaluator error {} on [{lo}, {hi}] misses the rounded endpoint value {} at {}",
                score.value, score.evaluation_error, endpoint.value, endpoint.x
            ),
            Self::DisjointEndpointEnclosure {
                lo,
                hi,
                endpoint,
                endpoint_derivative,
                enclosure,
            } => write!(
                f,
                "score search: derivative enclosures on [{lo}, {hi}] and its endpoint {} are disjoint: endpoint range {endpoint_derivative:?}, cell {enclosure:?}; point estimate {endpoint:?}",
                endpoint.x
            ),
            Self::InconsistentRootEnclosure {
                lo,
                hi,
                left_derivative,
                right_derivative,
                curvature,
                left_newton,
                right_newton,
                point_newton,
            } => write!(
                f,
                "score search: interval-Newton certificates for the unique root on [{lo}, {hi}] \
                 are inconsistent: left derivative {left_derivative:?}, right derivative \
                 {right_derivative:?}, curvature {curvature:?}, left image {left_newton:?}, \
                 right image {right_newton:?}, point image {point_newton:?}"
            ),
            Self::Unresolved {
                lo,
                hi,
                requested_resolution,
                enclosure,
            } => {
                // The enclosure already carries both numbers, but a reader has
                // to notice the comparison themselves. State it: when the
                // certified evaluation error has reached the requested
                // resolution, the request is the defect, not the search (#2614).
                let evaluation_error = enclosure.score.evaluation_error;
                let verdict = if evaluation_error >= *requested_resolution {
                    " -- the REQUEST is unsatisfiable: the certified evaluation error at this cell \
                     already reaches the requested resolution, so no bracket narrower than about \
                     twice that error is decidable and no additional subdivision can close it"
                } else {
                    ""
                };
                write!(
                    f,
                    "score search: stationary structure unresolved on [{lo}, {hi}] at requested \
                     resolution {requested_resolution} (certified evaluation error \
                     {evaluation_error:e}){verdict}: {enclosure:?}"
                )
            }
            Self::SubdivisionBudget {
                lo,
                hi,
                cell_lo,
                cell_hi,
                requested_resolution,
                subdivisions,
                budget,
                depth_bound,
                enclosure,
            } => {
                // Which of these two numbers is larger decides whether a bigger
                // budget is a fix or a distraction. Reporting the budget alone
                // sends the reader to the wrong lever (#2614).
                let evaluation_error = enclosure.score.evaluation_error;
                let verdict = if evaluation_error >= *requested_resolution {
                    "a LARGER BUDGET CANNOT HELP -- the certified evaluation error already reaches \
                     the requested resolution, so no subdivision separates stationary structure at \
                     this tolerance; the resolution asked for is finer than the evaluator delivers"
                } else {
                    "the evaluation error is below the requested resolution, so this cell was still \
                     separable and a larger budget may resolve it"
                };
                write!(
                    f,
                    "score search: {subdivisions} cell subdivisions on [{lo}, {hi}] at requested \
                     resolution {requested_resolution} exceed the budget {budget} derived from this \
                     domain's subdivision depth bound {depth_bound}; the criterion is still \
                     undecomposable at [{cell_lo}, {cell_hi}], so it neither excludes nor isolates \
                     stationary structure over a region the search can only enumerate. Certified \
                     evaluation error at this cell is {evaluation_error:e} against requested \
                     resolution {requested_resolution:e}: {verdict}"
                )
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ScoreSearchError<E> {}

#[derive(Clone, Copy)]
struct SearchSample {
    sample: ScoreSample,
    point_enclosure: Option<DerivativeEnclosure>,
}

#[derive(Clone, Copy)]
struct SearchNode {
    left: SearchSample,
    right: SearchSample,
}

#[derive(Clone, Copy)]
struct TerminalScoreCandidate {
    score: ScoreValueEnclosure,
    /// Forward error of the rounded representative used to compare this
    /// candidate with the selected representative. For a region certificate
    /// this is kept separate from the exact range: the range bounds the
    /// terminal maximum, while the error belongs to an actually evaluated
    /// point.
    comparison_error: f64,
    /// Present only when the terminal maximum is the exact score at this
    /// represented point. Region certificates deliberately carry `None` so
    /// their possible improvement over a representative is retained.
    point_x: Option<f64>,
}

impl TerminalScoreCandidate {
    #[inline]
    fn point(x: f64, score: ScoreValueEnclosure) -> Self {
        Self {
            score,
            comparison_error: score.evaluation_error,
            point_x: Some(x),
        }
    }

    #[inline]
    fn region(score: ScoreValueEnclosure, comparison_error: f64) -> Self {
        Self {
            score,
            comparison_error,
            point_x: None,
        }
    }
}

fn evaluate_sample<E, F>(x: f64, evaluate: &mut F) -> Result<SearchSample, ScoreSearchError<E>>
where
    F: FnMut(f64) -> Result<ScoreJet, E>,
{
    let jet = evaluate(x).map_err(|source| ScoreSearchError::PointEvaluation { x, source })?;
    let sample = ScoreSample {
        x,
        value: jet.value,
        derivative: jet.derivative,
        curvature: jet.curvature,
        third: jet.third,
    };
    if sample.value.is_finite()
        && sample.derivative.is_finite()
        && sample.curvature.is_finite()
        && sample.third.is_finite()
    {
        Ok(SearchSample {
            sample,
            point_enclosure: None,
        })
    } else {
        Err(ScoreSearchError::NonFiniteSample { sample })
    }
}

fn checked_enclosure<E, F>(
    left: ScoreSample,
    right: ScoreSample,
    enclose: &mut F,
) -> Result<DerivativeEnclosure, ScoreSearchError<E>>
where
    F: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let lo = left.x;
    let hi = right.x;
    // The cell's endpoints are handed to the oracle as the SAMPLES the search
    // already paid for, not as bare abscissae. An endpoint-anchored enclosure
    // needs the endpoint jets and nothing else, so this is what makes it free:
    // the oracle reads `left`/`right` instead of re-evaluating the criterion at
    // two points it has already evaluated.
    let enclosure = enclose(left, right)
        .map_err(|source| ScoreSearchError::EnclosureEvaluation { lo, hi, source })?;
    if !(enclosure.derivative.is_valid()
        && enclosure.curvature.is_valid()
        && enclosure.score.value.is_valid()
        && enclosure.score.evaluation_error.is_finite()
        && enclosure.score.evaluation_error >= 0.0)
    {
        return Err(ScoreSearchError::InvalidEnclosure { lo, hi, enclosure });
    }
    let score = enclosure.score;
    let resolved_score = score.value.widen(score.evaluation_error);
    for endpoint in [left, right] {
        if !resolved_score.contains(endpoint.value) {
            return Err(ScoreSearchError::ScoreValueEnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                score,
            });
        }
    }
    Ok(enclosure)
}

/// Attach the oracle's exact score/derivative ranges at one represented point.
///
/// A nearest-rounded scalar jet is not required to lie inside an exact-real
/// interval extension.  Instead, proof decisions use this degenerate-cell
/// enclosure.  Both the point range and its parent-cell range contain the same
/// exact endpoint derivative, so disjointness remains a valid contract check.
fn certify_point<E, F>(
    point: &mut SearchSample,
    enclose: &mut F,
) -> Result<DerivativeEnclosure, ScoreSearchError<E>>
where
    F: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let enclosure = match point.point_enclosure {
        Some(enclosure) => enclosure,
        None => {
            let enclosure = checked_enclosure(point.sample, point.sample, enclose)?;
            point.point_enclosure = Some(enclosure);
            enclosure
        }
    };
    Ok(enclosure)
}

fn certify_endpoint_derivative<E, F>(
    point: &mut SearchSample,
    cell_lo: f64,
    cell_hi: f64,
    cell: DerivativeEnclosure,
    enclose: &mut F,
) -> Result<ClosedInterval, ScoreSearchError<E>>
where
    F: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let endpoint_derivative = certify_point(point, enclose)?.derivative;
    endpoint_derivative.intersection(cell.derivative).ok_or(
        ScoreSearchError::DisjointEndpointEnclosure {
            lo: cell_lo,
            hi: cell_hi,
            endpoint: point.sample,
            endpoint_derivative,
            enclosure: cell,
        },
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StrictSign {
    Negative,
    Positive,
}

#[inline]
fn strict_sign(interval: ClosedInterval) -> Option<StrictSign> {
    if interval.hi < 0.0 {
        Some(StrictSign::Negative)
    } else if interval.lo > 0.0 {
        Some(StrictSign::Positive)
    } else {
        None
    }
}

#[inline]
fn is_exact_zero(interval: ClosedInterval) -> bool {
    interval.lo == 0.0 && interval.hi == 0.0
}

fn certify_bracket_score<E, Eval, Enclose>(
    bracket: ClosedInterval,
    representative: SearchSample,
    evaluate: &mut Eval,
    enclose: &mut Enclose,
) -> Result<ScoreValueEnclosure, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    if bracket.lo == bracket.hi {
        let mut representative = representative;
        return Ok(certify_point(&mut representative, enclose)?.score);
    }
    let left = if representative.sample.x == bracket.lo {
        representative
    } else {
        evaluate_sample(bracket.lo, evaluate)?
    };
    let right = if representative.sample.x == bracket.hi {
        representative
    } else {
        evaluate_sample(bracket.hi, evaluate)?
    };
    Ok(checked_enclosure(left.sample, right.sample, enclose)?.score)
}

enum UniqueRootRefinement {
    Stationary(StationaryPoint),
    ResolutionFlat {
        region: ResolutionFlatRegion,
        maximum: ScoreValueEnclosure,
    },
}

/// Refine a UNIQUE derivative root.  The caller has already proved uniqueness
/// by a curvature enclosure that excludes zero and supplied endpoint
/// derivative enclosures of opposite sign.
fn refine_unique_root<E, Eval, Enclose>(
    mut left: SearchSample,
    mut right: SearchSample,
    resolution: f64,
    enclosure: DerivativeEnclosure,
    evaluate: &mut Eval,
    enclose: &mut Enclose,
) -> Result<UniqueRootRefinement, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let bracket_lo = left.sample.x;
    let bracket_hi = right.sample.x;
    let mut left_derivative =
        certify_endpoint_derivative(&mut left, bracket_lo, bracket_hi, enclosure, enclose)?;
    let mut right_derivative =
        certify_endpoint_derivative(&mut right, bracket_lo, bracket_hi, enclosure, enclose)?;
    let curvature_sign =
        strict_sign(enclosure.curvature).ok_or(ScoreSearchError::InvalidEnclosure {
            lo: left.sample.x,
            hi: right.sample.x,
            enclosure,
        })?;
    let increasing = curvature_sign == StrictSign::Positive;
    let expected_left_sign = if increasing {
        StrictSign::Negative
    } else {
        StrictSign::Positive
    };
    let expected_right_sign = if increasing {
        StrictSign::Positive
    } else {
        StrictSign::Negative
    };
    if strict_sign(left_derivative) != Some(expected_left_sign)
        || strict_sign(right_derivative) != Some(expected_right_sign)
    {
        return Err(ScoreSearchError::InvalidEnclosure {
            lo: left.sample.x,
            hi: right.sample.x,
            enclosure,
        });
    }

    let mut force_midpoint = false;
    while right.sample.x - left.sample.x > resolution {
        let width = right.sample.x - left.sample.x;
        let midpoint = left.sample.x + 0.5 * width;
        if !(midpoint > left.sample.x && midpoint < right.sample.x) {
            return Err(ScoreSearchError::Unresolved {
                lo: left.sample.x,
                hi: right.sample.x,
                requested_resolution: resolution,
                enclosure,
            });
        }

        // Newton is accepted only in the central half of the bracket.  Thus
        // every accepted point, Newton or midpoint, contracts the maintained
        // sign bracket by at least one quarter.  The loop has no iteration cap
        // because its geometric termination follows from this safeguard.
        // Point derivatives are refinement proposals, not proof currency. In
        // particular, a nonzero derivative can round to scalar zero. Rank the
        // Newton anchors by their certified point ranges so that false scalar
        // zeros cannot control either the refinement path or its eventual
        // endpoint representative.
        let base = if left_derivative.max_abs() <= right_derivative.max_abs() {
            left.sample
        } else {
            right.sample
        };
        let newton = if base.curvature != 0.0 {
            base.x - base.derivative / base.curvature
        } else {
            f64::NAN
        };
        let guard = 0.25 * width;
        let x = if !force_midpoint
            && newton.is_finite()
            && newton >= left.sample.x + guard
            && newton <= right.sample.x - guard
        {
            newton
        } else {
            midpoint
        };
        force_midpoint = false;
        if !(x > left.sample.x && x < right.sample.x) {
            return Err(ScoreSearchError::Unresolved {
                lo: left.sample.x,
                hi: right.sample.x,
                requested_resolution: resolution,
                enclosure,
            });
        }
        let mut sample = evaluate_sample(x, evaluate)?;
        let probe_x = sample.sample.x;
        let mut point_derivative = certify_endpoint_derivative(
            &mut sample,
            left.sample.x,
            right.sample.x,
            enclosure,
            enclose,
        )?;
        let mut root_curvature = enclosure.curvature;
        if !is_exact_zero(point_derivative) && strict_sign(point_derivative).is_none() {
            // A degenerate-cell derivative enclosure can remain wide when its
            // analytic formula contains cancellation. The two adjacent cell
            // extensions are independent exact evidence about their shared
            // endpoint. Intersect all three rather than discarding the cell
            // information after merely checking overlap.
            let left_cell = checked_enclosure(left.sample, sample.sample, enclose)?;
            let right_cell = checked_enclosure(sample.sample, right.sample, enclose)?;
            let left_probe_derivative = certify_endpoint_derivative(
                &mut sample,
                left.sample.x,
                probe_x,
                left_cell,
                enclose,
            )?;
            let right_probe_derivative = certify_endpoint_derivative(
                &mut sample,
                probe_x,
                right.sample.x,
                right_cell,
                enclose,
            )?;
            point_derivative = left_probe_derivative
                .intersection(right_probe_derivative)
                .ok_or(ScoreSearchError::DisjointEndpointEnclosure {
                    lo: left.sample.x,
                    hi: right.sample.x,
                    endpoint: sample.sample,
                    endpoint_derivative: left_probe_derivative,
                    enclosure: right_cell,
                })?;
            let child_curvature = left_cell.curvature.hull(right_cell.curvature);
            root_curvature = enclosure.curvature.intersection(child_curvature).ok_or(
                ScoreSearchError::InvalidEnclosure {
                    lo: left.sample.x,
                    hi: right.sample.x,
                    enclosure: right_cell,
                },
            )?;
        }
        if is_exact_zero(point_derivative) {
            let bracket = ClosedInterval::point(x);
            let score = certify_bracket_score(bracket, sample, evaluate, enclose)?;
            return Ok(UniqueRootRefinement::Stationary(StationaryPoint {
                sample: sample.sample,
                bracket,
                score,
                curvature: root_curvature,
            }));
        }
        if let Some(sign) = strict_sign(point_derivative) {
            match (increasing, sign) {
                (true, StrictSign::Negative) | (false, StrictSign::Positive) => {
                    left = sample;
                    left_derivative = point_derivative;
                }
                (true, StrictSign::Positive) | (false, StrictSign::Negative) => {
                    right = sample;
                    right_derivative = point_derivative;
                }
            }
            continue;
        }

        // The derivative at the probe is below the interval evaluator's sign
        // resolution, but strict concavity is already a quantitative
        // optimality certificate. If f'' <= -mu < 0, then for the unique
        // maximizer x* and any represented x,
        //
        //   f(x*) - f(x) <= f'(x)^2 / (2 mu).
        //
        // Evaluate that bound with directed intervals. This is the missing
        // stopping rule in #2790: demanding a machine-scale LOCATION bracket
        // after the exact objective gap is below the score evaluator's own
        // forward error cannot add information. The returned category says
        // resolution-flat optimum, not stationary point, so no KKT claim is
        // manufactured from the cancellation-heavy derivative.
        let point_score = certify_point(&mut sample, enclose)?.score;
        if let Some((region, maximum)) = score_resolved_concave_maximum(
            SearchNode { left, right },
            enclosure,
            sample.sample,
            point_derivative,
            root_curvature,
            point_score,
        ) {
            return Ok(UniqueRootRefinement::ResolutionFlat { region, maximum });
        }

        // The point derivative is itself unresolved at f64 precision. The
        // mean-value theorem gives THREE independent interval-Newton images of
        // the same unique root: one from the point and one from each signed
        // endpoint. Intersect all three. Using only the cancellation-heavy
        // point image can leave the whole bracket unchanged even when the
        // endpoint images contract it decisively.
        //
        //   root = x₀ - f'(x₀) / f''(ξ),  ξ between x₀ and root.
        let bracket = ClosedInterval::new(left.sample.x, right.sample.x);
        let point_newton =
            ClosedInterval::point(x).sub(point_derivative.div_nonzero(root_curvature));
        let left_newton = ClosedInterval::point(left.sample.x)
            .sub(left_derivative.div_nonzero(enclosure.curvature));
        let right_newton = ClosedInterval::point(right.sample.x)
            .sub(right_derivative.div_nonzero(enclosure.curvature));
        let root = bracket
            .intersection(point_newton)
            .and_then(|root| root.intersection(left_newton))
            .and_then(|root| root.intersection(right_newton))
            .ok_or(ScoreSearchError::InconsistentRootEnclosure {
                lo: left.sample.x,
                hi: right.sample.x,
                left_derivative,
                right_derivative,
                curvature: enclosure.curvature,
                left_newton,
                right_newton,
                point_newton,
            })?;
        if root.hi - root.lo <= resolution {
            let score = certify_bracket_score(root, sample, evaluate, enclose)?;
            return Ok(UniqueRootRefinement::Stationary(StationaryPoint {
                sample: sample.sample,
                bracket: root,
                score,
                curvature: root_curvature,
            }));
        }
        if root.lo > left.sample.x || root.hi < right.sample.x {
            let mut new_left = if root.lo == sample.sample.x {
                sample
            } else {
                evaluate_sample(root.lo, evaluate)?
            };
            let mut new_right = if root.hi == sample.sample.x {
                sample
            } else {
                evaluate_sample(root.hi, evaluate)?
            };
            let contracted_enclosure =
                checked_enclosure(new_left.sample, new_right.sample, enclose)?;
            // The point certificate and the strict curvature range give a
            // second exact score extension over the contracted root image:
            //
            //   f(y) = f(x) + f'(x)(y-x) + 1/2 f''(ξ)(y-x)^2.
            //
            // Intersecting it with the endpoint-based cell extension removes
            // common cancellation noise from the score range. This does not
            // alter the evaluator error or invent a tolerance; it can only
            // reveal that the existing exact score diameter has reached that
            // existing comparison floor.
            let point_score = certify_point(&mut sample, enclose)?.score;
            let displacement = ClosedInterval::new(root.lo - x, root.hi - x);
            let taylor_score = point_score
                .value
                .add(point_derivative.mul(displacement))
                .add(root_curvature.mul(displacement.square()).scale(0.5));
            let tightened_score = contracted_enclosure
                .score
                .value
                .intersection(taylor_score)
                .ok_or(ScoreSearchError::InvalidEnclosure {
                    lo: root.lo,
                    hi: root.hi,
                    enclosure: contracted_enclosure,
                })?;
            let contracted_enclosure = DerivativeEnclosure {
                score: ScoreValueEnclosure {
                    value: tightened_score,
                    evaluation_error: contracted_enclosure.score.evaluation_error,
                },
                ..contracted_enclosure
            };
            if let Some(region) = resolution_flat_region(
                SearchNode {
                    left: new_left,
                    right: new_right,
                },
                contracted_enclosure,
            ) {
                return Ok(UniqueRootRefinement::ResolutionFlat {
                    region,
                    maximum: contracted_enclosure.score,
                });
            }
            let new_left_derivative = if new_left.sample.x == sample.sample.x {
                point_derivative
            } else {
                certify_endpoint_derivative(
                    &mut new_left,
                    root.lo,
                    root.hi,
                    contracted_enclosure,
                    enclose,
                )?
            };
            let new_right_derivative = if new_right.sample.x == sample.sample.x {
                point_derivative
            } else {
                certify_endpoint_derivative(
                    &mut new_right,
                    root.lo,
                    root.hi,
                    contracted_enclosure,
                    enclose,
                )?
            };

            // A Newton image encloses the root; it does NOT prove that either
            // image boundary has a strict derivative sign. Retain each old
            // signed endpoint until its replacement independently certifies
            // the same oriented sign. This is the invariant that proves the
            // root unique on every subsequent iteration.
            let mut preserved_sign_contraction = false;
            if root.lo > left.sample.x {
                match strict_sign(new_left_derivative) {
                    Some(sign) if sign == expected_left_sign => {
                        left = new_left;
                        left_derivative = new_left_derivative;
                        preserved_sign_contraction = true;
                    }
                    Some(_) => {
                        return Err(ScoreSearchError::InvalidEnclosure {
                            lo: root.lo,
                            hi: root.hi,
                            enclosure: contracted_enclosure,
                        });
                    }
                    None => {}
                }
            }
            if root.hi < right.sample.x {
                match strict_sign(new_right_derivative) {
                    Some(sign) if sign == expected_right_sign => {
                        right = new_right;
                        right_derivative = new_right_derivative;
                        preserved_sign_contraction = true;
                    }
                    Some(_) => {
                        return Err(ScoreSearchError::InvalidEnclosure {
                            lo: root.lo,
                            hi: root.hi,
                            enclosure: contracted_enclosure,
                        });
                    }
                    None => {}
                }
            }
            if preserved_sign_contraction {
                continue;
            }
        }
        if x != midpoint {
            force_midpoint = true;
            continue;
        }
        return Err(ScoreSearchError::Unresolved {
            lo: left.sample.x,
            hi: right.sample.x,
            requested_resolution: resolution,
            enclosure,
        });
    }

    let midpoint = left.sample.x + 0.5 * (right.sample.x - left.sample.x);
    let sample = if midpoint > left.sample.x && midpoint < right.sample.x {
        evaluate_sample(midpoint, evaluate)?.sample
    } else if left_derivative.max_abs() <= right_derivative.max_abs() {
        left.sample
    } else {
        right.sample
    };
    let bracket = ClosedInterval::new(left.sample.x, right.sample.x);
    let representative = SearchSample {
        sample,
        point_enclosure: None,
    };
    let score = certify_bracket_score(bracket, representative, evaluate, enclose)?;
    Ok(UniqueRootRefinement::Stationary(StationaryPoint {
        sample,
        bracket,
        score,
        curvature: enclosure.curvature,
    }))
}

/// When subdivision lands exactly on a stationary abscissa, a rigorous point
/// interval can contain zero without proving the derivative is exactly zero.
/// Probe symmetrically within one requested-resolution bracket and accept the
/// shared endpoint only if those two certified derivative ranges have opposite
/// signs and the probe-cell curvature proves uniqueness.
fn isolate_shared_endpoint_root<E, Eval, Enclose>(
    endpoint: SearchSample,
    domain_lo: f64,
    domain_hi: f64,
    resolution: f64,
    evaluate: &mut Eval,
    enclose: &mut Enclose,
) -> Result<Option<StationaryPoint>, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let radius = 0.5 * resolution;
    let left_x = endpoint.sample.x - radius;
    let mut right_x = endpoint.sample.x + radius;
    if !(left_x >= domain_lo
        && right_x <= domain_hi
        && left_x < endpoint.sample.x
        && right_x > endpoint.sample.x)
    {
        return Ok(None);
    }
    while right_x - left_x > resolution {
        right_x = next_down(right_x);
    }
    if !(right_x > endpoint.sample.x && right_x - left_x <= resolution) {
        return Ok(None);
    }

    let mut left = evaluate_sample(left_x, evaluate)?;
    let mut right = evaluate_sample(right_x, evaluate)?;
    let probe_enclosure = checked_enclosure(left.sample, right.sample, enclose)?;
    if probe_enclosure.curvature.contains_zero() {
        return Ok(None);
    }
    let left_derivative =
        certify_endpoint_derivative(&mut left, left_x, right_x, probe_enclosure, enclose)?;
    let right_derivative =
        certify_endpoint_derivative(&mut right, left_x, right_x, probe_enclosure, enclose)?;
    if strict_sign(left_derivative)
        .zip(strict_sign(right_derivative))
        .is_some_and(|(left_sign, right_sign)| left_sign != right_sign)
    {
        Ok(Some(StationaryPoint {
            sample: endpoint.sample,
            bracket: ClosedInterval::new(left_x, right_x),
            score: probe_enclosure.score,
            curvature: probe_enclosure.curvature,
        }))
    } else {
        Ok(None)
    }
}

/// Prove that every score value in a cell is indistinguishable from one of its
/// endpoint samples at the point evaluator's certified f64 resolution.
///
/// If the exact score range is `[L, U]`, every pair of exact scores in the cell
/// differs by at most `U-L`. If each nearest-rounded point value has absolute
/// forward error at most `rho`, a comparison of two such values has uncertainty
/// at most `2 rho`. The cell is resolution-flat only when `U-L <= 2 rho`.
///
/// Both sides are expressed in score-value units and are invariant under
/// adding a constant to the objective. Derivative-evaluator error is
/// deliberately absent: integrating it would bound the error of a hypothetical
/// numerical quadrature, not the forward error of `ScoreJet::value`.
fn resolution_flat_region(
    node: SearchNode,
    enclosure: DerivativeEnclosure,
) -> Option<ResolutionFlatRegion> {
    let score = enclosure.score;
    let max_score_gap = if score.value.lo == score.value.hi {
        0.0
    } else {
        next_up(score.value.hi - score.value.lo)
    };
    let score_resolution = if score.evaluation_error == 0.0 {
        0.0
    } else {
        next_up(2.0 * score.evaluation_error)
    };
    if !(max_score_gap.is_finite() && score_resolution.is_finite()) {
        return None;
    }
    let sample = if node.right.sample.value > node.left.sample.value {
        node.right.sample
    } else {
        node.left.sample
    };
    (max_score_gap <= score_resolution).then_some(ResolutionFlatRegion {
        sample,
        bracket: ClosedInterval::new(node.left.sample.x, node.right.sample.x),
        score: score.value,
        max_score_gap,
        score_resolution,
    })
}

/// Turn strict concavity plus an unresolved point derivative into a direct
/// score-optimality certificate.
///
/// `curvature.hi < 0` proves a unique maximum in the signed root bracket. The
/// strong-concavity inequality bounds how much that maximum can improve on the
/// represented point without requiring its location to be distinguishable.
/// Unlike [`resolution_flat_region`], this does not require every pair of scores
/// in the cell to be close; only the maximum and the returned representative
/// need to be numerically indistinguishable.
fn score_resolved_concave_maximum(
    node: SearchNode,
    enclosure: DerivativeEnclosure,
    sample: ScoreSample,
    point_derivative: ClosedInterval,
    curvature: ClosedInterval,
    point_score: ScoreValueEnclosure,
) -> Option<(ResolutionFlatRegion, ScoreValueEnclosure)> {
    if !(curvature.hi < 0.0 && sample.x >= node.left.sample.x && sample.x <= node.right.sample.x) {
        return None;
    }

    // max |g|^2 / (2 mu), where mu = -max f''. Interval multiplication and
    // division are outward-rounded, including the factor 1/2.
    let maximum_excess = point_derivative
        .square()
        .scale(0.5)
        .div_positive(curvature.neg())
        .hi;
    let comparison_resolution = point_score.evaluation_error;
    if !(maximum_excess.is_finite()
        && maximum_excess >= 0.0
        && comparison_resolution.is_finite()
        && maximum_excess <= comparison_resolution)
    {
        return None;
    }

    // Intersect the cell-wide score upper bound with the tighter
    // strong-concavity upper bound anchored at the represented point. The
    // maximum is at least the point score itself.
    let maximum = ClosedInterval::new(
        point_score.value.lo,
        enclosure
            .score
            .value
            .hi
            .min(sum_up(point_score.value.hi, maximum_excess)),
    );
    if !maximum.is_valid() {
        return None;
    }
    let region = ResolutionFlatRegion {
        sample,
        bracket: ClosedInterval::new(node.left.sample.x, node.right.sample.x),
        score: enclosure.score.value,
        max_score_gap: maximum_excess,
        score_resolution: comparison_resolution,
    };
    Some((
        region,
        ScoreValueEnclosure {
            value: maximum,
            evaluation_error: point_score
                .evaluation_error
                .max(enclosure.score.evaluation_error),
        },
    ))
}

/// Select a domain boundary only when one proof cell covers the whole domain
/// and its derivative has one strict sign throughout.
///
/// Rounded endpoint values may tie even when their exact-real ordering is
/// strict. A whole-domain monotonicity certificate resolves that ordering
/// directly. A proper subcell cannot select the global representative because
/// its endpoint has not been compared with maxima in the other cells.
fn certified_domain_boundary(
    node: &SearchNode,
    derivative_sign: StrictSign,
    domain_lo: f64,
    domain_hi: f64,
) -> Option<(ScoreSample, ScoreOptimumLocation)> {
    if node.left.sample.x != domain_lo || node.right.sample.x != domain_hi {
        return None;
    }
    Some(match derivative_sign {
        StrictSign::Positive => (node.right.sample, ScoreOptimumLocation::UpperBoundary),
        StrictSign::Negative => (node.left.sample, ScoreOptimumLocation::LowerBoundary),
    })
}

/// Globally maximize a smooth score on `[lo, hi]` by certified stationary
/// isolation.
///
/// `evaluate` returns a nearest-rounded score jet at a point. `enclose(a, b)`
/// receives the cell's two ENDPOINT SAMPLES — the jets the search already
/// obtained from `evaluate` — and must return OUTER ranges containing the exact
/// first and second derivative at every point of `[a.x, b.x]`.
///
/// Handing the samples in (rather than the bare abscissae) is what keeps an
/// endpoint-anchored enclosure free: such an oracle is a Taylor pad around the
/// endpoint jets, so with the jets in hand it performs no criterion evaluation
/// of its own. An oracle whose enclosure is a genuine interval extension may
/// ignore the jets and use `a.x`/`b.x`.
///
/// The scalar derivatives are never treated as proofs: when an endpoint sign
/// matters, the search asks `enclose(a, a)` for its exact derivative range.
/// The point and parent-cell ranges must overlap, but the exact-real parent
/// range is intentionally not required to contain a separately rounded scalar
/// estimate.
///
/// A successful return means every cell was derivative-excluded, stationary-
/// isolated to `resolution`, proved score-flat at the local representable value
/// resolution, or proved exactly dominated by an already attained point score.
/// Any cell that satisfies none of those conditions produces
/// [`ScoreSearchError::Unresolved`].
///
/// The traversal is bounded by [`subdivision_budget`]. The per-cell resolution
/// floor bounds the DEPTH of the subdivision and never its BREADTH, and those
/// are different failures. A criterion that certifies NOTHING bottoms out on the
/// floor after `D` subdivisions and is already typed
/// [`ScoreSearchError::Unresolved`]. The unbounded case is the one where cells
/// DO certify, at widths far above the floor, and there are simply too many of
/// them: a criterion whose derivative and curvature enclosures both straddle
/// zero over a wide region excludes no cell by a sign and isolates no root, so
/// every cell it reaches is split until its score range collapses under the
/// evaluator's own error — and the leaf count of that tree is exponential in the
/// depth, 2^32 cells on a 58-wide log-λ domain at `sqrt(eps)` resolution, which
/// is non-termination rather than slowness (#2546). Exceeding the budget is
/// [`ScoreSearchError::SubdivisionBudget`], a statement about the CRITERION and
/// not about the machine: the search was asked to certify more cells than a
/// converging 1-D decomposition at this resolution consists of.
pub fn maximize_score_1d<E, Eval, Enclose>(
    lo: f64,
    hi: f64,
    resolution: f64,
    mut evaluate: Eval,
    mut enclose: Enclose,
) -> Result<ScoreSearchResult, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    if !(lo.is_finite() && hi.is_finite() && lo <= hi && (hi - lo).is_finite()) {
        return Err(ScoreSearchError::InvalidDomain { lo, hi });
    }
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(ScoreSearchError::InvalidResolution { resolution });
    }

    let mut lower_boundary = evaluate_sample(lo, &mut evaluate)?;
    if lo == hi {
        let score =
            checked_enclosure(lower_boundary.sample, lower_boundary.sample, &mut enclose)?.score;
        return Ok(ScoreSearchResult {
            optimum: lower_boundary.sample,
            location: ScoreOptimumLocation::LowerBoundary,
            lower_boundary: lower_boundary.sample,
            upper_boundary: lower_boundary.sample,
            stationary_points: Vec::new(),
            resolution_flat_regions: Vec::new(),
            dominated_regions: Vec::new(),
            value_certificate: GlobalScoreCertificate {
                selected: score.value,
                maximum: score.value,
                maximum_excess: 0.0,
                comparison_resolution: 0.0,
            },
        });
    }
    let mut upper_boundary = evaluate_sample(hi, &mut evaluate)?;
    let lower_boundary_score = certify_point(&mut lower_boundary, &mut enclose)?.score;
    let upper_boundary_score = certify_point(&mut upper_boundary, &mut enclose)?.score;
    let mut incumbent_lower = lower_boundary_score
        .value
        .lo
        .max(upper_boundary_score.value.lo);
    let (mut optimum, mut location) = if upper_boundary.sample.value > lower_boundary.sample.value {
        (upper_boundary.sample, ScoreOptimumLocation::UpperBoundary)
    } else {
        (lower_boundary.sample, ScoreOptimumLocation::LowerBoundary)
    };

    let (budget, depth_bound) = subdivision_budget(lo, hi, resolution);
    let mut subdivisions = 0usize;
    let mut stationary_points = Vec::<StationaryPoint>::new();
    let mut resolution_flat_regions = Vec::<ResolutionFlatRegion>::new();
    let mut dominated_regions = Vec::<DominatedRegion>::new();
    // Boundary points are unconditional feasible incumbents. Keeping both in
    // the terminal ledger makes every later dominance decision independent of
    // which rounded boundary value happened to initialize `optimum`.
    let mut terminal_maxima = vec![
        TerminalScoreCandidate::point(lower_boundary.sample.x, lower_boundary_score),
        TerminalScoreCandidate::point(upper_boundary.sample.x, upper_boundary_score),
    ];
    let mut stack = vec![SearchNode {
        left: lower_boundary,
        right: upper_boundary,
    }];
    while let Some(mut node) = stack.pop() {
        let mathematical_enclosure =
            checked_enclosure(node.left.sample, node.right.sample, &mut enclose)?;
        let enclosure = mathematical_enclosure;
        if enclosure.score.value.hi < incumbent_lower {
            dominated_regions.push(DominatedRegion {
                bracket: ClosedInterval::new(node.left.sample.x, node.right.sample.x),
                score: enclosure.score,
                incumbent_lower,
            });
            continue;
        }
        if !enclosure.derivative.contains_zero() {
            let derivative_sign = if enclosure.derivative.lo > 0.0 {
                StrictSign::Positive
            } else {
                StrictSign::Negative
            };
            if let Some((proven_optimum, proven_location)) =
                certified_domain_boundary(&node, derivative_sign, lo, hi)
            {
                optimum = proven_optimum;
                location = proven_location;
            }
            let endpoint = match derivative_sign {
                StrictSign::Positive => &mut node.right,
                StrictSign::Negative => &mut node.left,
            };
            let endpoint_score = certify_point(endpoint, &mut enclose)?.score;
            incumbent_lower = incumbent_lower.max(endpoint_score.value.lo);
            terminal_maxima.push(TerminalScoreCandidate::point(
                endpoint.sample.x,
                endpoint_score,
            ));
            continue;
        }

        let monotone = !enclosure.curvature.contains_zero();
        if monotone {
            let node_lo = node.left.sample.x;
            let node_hi = node.right.sample.x;
            let left_derivative = certify_endpoint_derivative(
                &mut node.left,
                node_lo,
                node_hi,
                enclosure,
                &mut enclose,
            )?;
            let right_derivative = certify_endpoint_derivative(
                &mut node.right,
                node_lo,
                node_hi,
                enclosure,
                &mut enclose,
            )?;
            let left_sign = strict_sign(left_derivative);
            let right_sign = strict_sign(right_derivative);
            let mut root_flat = None;
            let stationary = if is_exact_zero(left_derivative) {
                let score = certify_point(&mut node.left, &mut enclose)?.score;
                Some(StationaryPoint {
                    sample: node.left.sample,
                    bracket: ClosedInterval::point(node.left.sample.x),
                    score,
                    curvature: enclosure.curvature,
                })
            } else if is_exact_zero(right_derivative) {
                let score = certify_point(&mut node.right, &mut enclose)?.score;
                Some(StationaryPoint {
                    sample: node.right.sample,
                    bracket: ClosedInterval::point(node.right.sample.x),
                    score,
                    curvature: enclosure.curvature,
                })
            } else if left_sign
                .zip(right_sign)
                .is_some_and(|(left_sign, right_sign)| left_sign != right_sign)
            {
                match refine_unique_root(
                    node.left,
                    node.right,
                    resolution,
                    enclosure,
                    &mut evaluate,
                    &mut enclose,
                )? {
                    UniqueRootRefinement::Stationary(stationary) => Some(stationary),
                    UniqueRootRefinement::ResolutionFlat { region, maximum } => {
                        root_flat = Some((region, maximum));
                        None
                    }
                }
            } else if left_sign.is_none() {
                isolate_shared_endpoint_root(
                    node.left,
                    lo,
                    hi,
                    resolution,
                    &mut evaluate,
                    &mut enclose,
                )?
            } else if right_sign.is_none() {
                isolate_shared_endpoint_root(
                    node.right,
                    lo,
                    hi,
                    resolution,
                    &mut evaluate,
                    &mut enclose,
                )?
            } else {
                None
            };

            if let Some((flat, maximum)) = root_flat {
                let index = resolution_flat_regions.len();
                if flat.sample.value > optimum.value {
                    optimum = flat.sample;
                    location = ScoreOptimumLocation::ResolutionFlat(index);
                }
                let mut representative = SearchSample {
                    sample: flat.sample,
                    point_enclosure: None,
                };
                let representative_score = certify_point(&mut representative, &mut enclose)?.score;
                incumbent_lower = incumbent_lower.max(representative_score.value.lo);
                terminal_maxima.push(TerminalScoreCandidate::region(
                    maximum,
                    representative_score
                        .evaluation_error
                        .max(maximum.evaluation_error),
                ));
                resolution_flat_regions.push(flat);
                continue;
            }

            if let Some(stationary) = stationary {
                let mut representative = SearchSample {
                    sample: stationary.sample,
                    point_enclosure: None,
                };
                let representative_score = certify_point(&mut representative, &mut enclose)?.score;
                incumbent_lower = incumbent_lower.max(representative_score.value.lo);
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
                if enclosure.curvature.hi < 0.0 {
                    let score = stationary.score;
                    terminal_maxima.push(if stationary.bracket.lo == stationary.bracket.hi {
                        TerminalScoreCandidate::point(stationary.sample.x, score)
                    } else {
                        TerminalScoreCandidate::region(
                            score,
                            representative_score
                                .evaluation_error
                                .max(score.evaluation_error),
                        )
                    });
                } else {
                    let left_score = certify_point(&mut node.left, &mut enclose)?.score;
                    let right_score = certify_point(&mut node.right, &mut enclose)?.score;
                    incumbent_lower = incumbent_lower
                        .max(left_score.value.lo)
                        .max(right_score.value.lo);
                    terminal_maxima.push(TerminalScoreCandidate::point(
                        node.left.sample.x,
                        left_score,
                    ));
                    terminal_maxima.push(TerminalScoreCandidate::point(
                        node.right.sample.x,
                        right_score,
                    ));
                }
                continue;
            }

            // Definite equal endpoint signs plus strict monotonicity exclude a
            // root.  An endpoint range that straddles zero is not silently
            // replaced by the rounded point sign; it proceeds to the value-flat
            // proof or subdivision below.
            if let Some((left_sign, right_sign)) = left_sign.zip(right_sign)
                && left_sign == right_sign
            {
                if let Some((proven_optimum, proven_location)) =
                    certified_domain_boundary(&node, left_sign, lo, hi)
                {
                    optimum = proven_optimum;
                    location = proven_location;
                }
                let endpoint = match left_sign {
                    StrictSign::Positive => &mut node.right,
                    StrictSign::Negative => &mut node.left,
                };
                let endpoint_score = certify_point(endpoint, &mut enclose)?.score;
                incumbent_lower = incumbent_lower.max(endpoint_score.value.lo);
                terminal_maxima.push(TerminalScoreCandidate::point(
                    endpoint.sample.x,
                    endpoint_score,
                ));
                continue;
            }
        }

        if let Some(flat) = resolution_flat_region(node, mathematical_enclosure) {
            let index = resolution_flat_regions.len();
            if flat.sample.value > optimum.value {
                optimum = flat.sample;
                location = ScoreOptimumLocation::ResolutionFlat(index);
            }
            let mut representative = SearchSample {
                sample: flat.sample,
                point_enclosure: None,
            };
            let representative_score = certify_point(&mut representative, &mut enclose)?.score;
            incumbent_lower = incumbent_lower.max(representative_score.value.lo);
            terminal_maxima.push(TerminalScoreCandidate::region(
                enclosure.score,
                representative_score
                    .evaluation_error
                    .max(enclosure.score.evaluation_error),
            ));
            resolution_flat_regions.push(flat);
            continue;
        }

        let width = node.right.sample.x - node.left.sample.x;
        let midpoint = node.left.sample.x + 0.5 * width;
        if width <= resolution || !(midpoint > node.left.sample.x && midpoint < node.right.sample.x)
        {
            return Err(ScoreSearchError::Unresolved {
                lo: node.left.sample.x,
                hi: node.right.sample.x,
                requested_resolution: resolution,
                enclosure,
            });
        }
        subdivisions += 1;
        if subdivisions > budget {
            return Err(ScoreSearchError::SubdivisionBudget {
                lo,
                hi,
                cell_lo: node.left.sample.x,
                cell_hi: node.right.sample.x,
                requested_resolution: resolution,
                subdivisions,
                budget,
                depth_bound,
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

    let mut selected_sample = SearchSample {
        sample: optimum,
        point_enclosure: None,
    };
    let selected_score = certify_point(&mut selected_sample, &mut enclose)?.score;
    let global_lower = terminal_maxima
        .iter()
        .map(|candidate| candidate.score.value.lo)
        .fold(selected_score.value.lo, f64::max);
    let global_upper = terminal_maxima
        .iter()
        .map(|candidate| candidate.score.value.hi)
        .fold(selected_score.value.hi, f64::max);
    let candidate_evaluation_error = terminal_maxima
        .iter()
        .filter(|candidate| candidate.point_x != Some(optimum.x))
        .map(|candidate| candidate.comparison_error)
        .fold(0.0_f64, f64::max);
    let maximum_excess = terminal_maxima
        .iter()
        .filter(|candidate| candidate.point_x != Some(optimum.x))
        .map(|candidate| {
            if candidate.score.value.hi <= selected_score.value.lo {
                0.0
            } else {
                next_up(candidate.score.value.hi - selected_score.value.lo)
            }
        })
        .fold(0.0_f64, f64::max);
    let comparison_resolution =
        add_nonnegative_upward(selected_score.evaluation_error, candidate_evaluation_error);

    Ok(ScoreSearchResult {
        optimum,
        location,
        lower_boundary: lower_boundary.sample,
        upper_boundary: upper_boundary.sample,
        stationary_points,
        resolution_flat_regions,
        dominated_regions,
        value_certificate: GlobalScoreCertificate {
            selected: selected_score.value,
            maximum: ClosedInterval::new(global_lower, global_upper),
            maximum_excess,
            comparison_resolution,
        },
    })
}

/// Repeat a certified global score search until its exact winning value is
/// orderable at the evaluator's certified comparison resolution.
///
/// Location resolution and value resolution are different proof currencies:
/// isolating every stationary point to `initial_resolution` can still leave
/// the winning candidate's exact score range wider than the point evaluator's
/// forward-error comparison permits. Each pass here independently rebuilds
/// the complete global certificate at a smaller location target. The observed
/// ratio between maximum excess and comparison resolution is only a refinement
/// strategy; it is never used as acceptance evidence.
///
/// There is no retry cap or acceptance fallback. Each retry contracts the
/// target by at least one binary subdivision. If the next target is no longer
/// representable, or the oracle cannot resolve stationary structure at that
/// finer target, or the finer traversal exceeds its [`subdivision_budget`], the
/// last complete certificate is returned unchanged so the caller can issue its
/// domain-specific typed refusal.
pub fn maximize_score_1d_value_ordered<E, Eval, Enclose>(
    lo: f64,
    hi: f64,
    initial_resolution: f64,
    mut evaluate: Eval,
    mut enclose: Enclose,
) -> Result<ScoreSearchResult, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let mut resolution = initial_resolution;
    let mut search = maximize_score_1d(lo, hi, resolution, &mut evaluate, &mut enclose)?;
    loop {
        let certificate = search.value_certificate;
        if certificate.maximum_excess <= certificate.comparison_resolution {
            return Ok(search);
        }
        let binary_refinement = 0.5 * resolution;
        let value_directed_refinement = if certificate.comparison_resolution > 0.0 {
            resolution * (certificate.comparison_resolution / certificate.maximum_excess)
        } else {
            binary_refinement
        };
        let next_resolution = binary_refinement.min(value_directed_refinement);
        if !(next_resolution.is_finite() && next_resolution > 0.0 && next_resolution < resolution) {
            return Ok(search);
        }
        match maximize_score_1d(lo, hi, next_resolution, &mut evaluate, &mut enclose) {
            Ok(refined) => {
                search = refined;
                resolution = next_resolution;
            }
            // A finer requested location is optional proof strengthening.
            // Preserve the last complete global certificate when the oracle
            // cannot resolve stationary structure at that finer currency; the
            // caller will still reject it if its values remain unordered.
            //
            // A retry that exhausts its subdivision budget is the same kind of
            // outcome and ENDS the loop rather than contracting again: each
            // retry's budget grows as its target shrinks, so continuing past
            // one exhaustion would pay a whole traversal per halving down to
            // the denormal floor — a second unbounded axis (#2546).
            Err(
                ScoreSearchError::Unresolved { .. } | ScoreSearchError::SubdivisionBudget { .. },
            ) => return Ok(search),
            Err(error) => return Err(error),
        }
    }
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
    ZeroLambdaResidualUnavailable {
        output: usize,
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
    ElementaryEnclosureUnavailable {
        function: &'static str,
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
    InconsistentResidualEnclosures {
        output: usize,
        lo: f64,
        hi: f64,
        direct: ClosedInterval,
        complement: ClosedInterval,
    },
    UnboundedScoreEvaluationError {
        lo: f64,
        hi: f64,
        error: f64,
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
            Self::ZeroLambdaResidualUnavailable { output } => write!(
                f,
                "affine REML could not certify the zero-smoothing residual for response {output}"
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
            Self::ElementaryEnclosureUnavailable { function, lo, hi } => write!(
                f,
                "affine REML has no finite source-derived {function} enclosure on [{lo}, {hi}]"
            ),
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
            Self::InconsistentResidualEnclosures {
                output,
                lo,
                hi,
                direct,
                complement,
            } => write!(
                f,
                "affine REML residual {output} has disjoint direct {direct:?} and zero-smoothing-complement {complement:?} enclosures on [{lo}, {hi}]"
            ),
            Self::UnboundedScoreEvaluationError { lo, hi, error } => write!(
                f,
                "affine REML score evaluator has no finite forward-error bound on [{lo}, {hi}] (bound {error})"
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
#[derive(Clone, Debug)]
pub struct AffineRemlProfile<'a> {
    gram_modes: &'a [f64],
    penalty_modes: &'a [f64],
    projected_rhs_squared: &'a [f64],
    response_energy: &'a [f64],
    /// Exact-real residuals on the finite part of the zero-smoothing face,
    ///
    /// `response_energy[d] - sum_{i:g_i>0} q[d,i] / g_i`.
    ///
    /// These invariants are computed once with an error-free leading sum and
    /// FMA-certified division corrections. Re-forming them independently in
    /// every interval evaluation loses the small Schur complement to the
    /// rounding scale of its O(energy) operands.
    zero_lambda_residual: Vec<ClosedInterval>,
    residual_dof: f64,
    logdet_constant: f64,
}

/// O(n) exact-leading accumulator.
///
/// `leading + correction` encloses the exact sum of every value submitted so
/// far. `leading` follows the ordinary binary64 accumulation path. Knuth's
/// TwoSum identity moves each discarded low part into `correction`, whose
/// directed interval accumulation never again mixes it with an O(energy)
/// operand. This is the fixed-size analogue of a floating-point expansion:
/// exact proof information, without an O(n²) expansion walk or arbitrary
/// precision dependency.
struct CertifiedCompensatedSum {
    leading: f64,
    correction: ClosedInterval,
}

impl CertifiedCompensatedSum {
    fn new(value: f64) -> Self {
        Self {
            leading: value,
            correction: ClosedInterval::point(0.0),
        }
    }

    /// Add one exact binary64 value, retaining the exact TwoSum residual.
    fn add_exact(&mut self, value: f64) -> bool {
        let sum = self.leading + value;
        if !sum.is_finite() {
            return false;
        }
        let virtual_value = sum - self.leading;
        let virtual_leading = sum - virtual_value;
        let value_residual = value - virtual_value;
        let leading_residual = self.leading - virtual_leading;
        // Under round-to-nearest with gradual underflow, the two residuals and
        // their final sum are exact (Knuth/Møller TwoSum).
        let error = leading_residual + value_residual;
        self.leading = sum;
        self.correction = self.correction.add(ClosedInterval::point(error));
        self.correction.is_valid()
    }

    fn subtract_interval(&mut self, value: ClosedInterval) -> bool {
        self.correction = self.correction.sub(value);
        self.correction.is_valid()
    }

    fn enclosure(self) -> Option<ClosedInterval> {
        let enclosure = ClosedInterval::point(self.leading).add(self.correction);
        (enclosure.is_valid() && enclosure.lo.is_finite() && enclosure.hi.is_finite())
            .then_some(enclosure)
    }
}

/// Split one exact-real positive quotient into a binary64 leading value and a
/// rigorous low correction:
///
/// `numerator / denominator ∈ leading + correction`.
///
/// The fused residual `numerator - leading*denominator` rounds only once.
/// Its two adjacent binary64 values therefore enclose the exact residual; a
/// directed division by the positive denominator transports that enclosure
/// into quotient units. The correction is normally O(u²) relative to the
/// quotient, rather than the O(u) width of independently directed division.
fn quotient_leading_and_correction(
    numerator: f64,
    denominator: f64,
) -> Option<(f64, ClosedInterval)> {
    if numerator == 0.0 {
        return Some((0.0, ClosedInterval::point(0.0)));
    }
    if !(numerator.is_finite() && numerator > 0.0 && denominator.is_finite() && denominator > 0.0) {
        return None;
    }
    let leading = numerator / denominator;
    if !(leading.is_finite() && leading >= 0.0) {
        return None;
    }
    if denominator == 1.0 {
        return Some((leading, ClosedInterval::point(0.0)));
    }
    let fused_residual = (-leading).mul_add(denominator, numerator);
    if !fused_residual.is_finite() {
        return None;
    }
    let exact_residual = ClosedInterval::new(next_down(fused_residual), next_up(fused_residual));
    let correction = exact_residual.div_positive(ClosedInterval::point(denominator));
    (correction.is_valid() && correction.lo.is_finite() && correction.hi.is_finite())
        .then_some((leading, correction))
}

fn certified_zero_lambda_residual(
    energy: f64,
    gram_modes: &[f64],
    projected_squares: &[f64],
) -> Option<ClosedInterval> {
    let mut residual = CertifiedCompensatedSum::new(energy);
    for (&gram, &projected_square) in gram_modes.iter().zip(projected_squares) {
        if gram == 0.0 || projected_square == 0.0 {
            continue;
        }
        let (leading, correction) = quotient_leading_and_correction(projected_square, gram)?;
        if !(residual.add_exact(-leading) && residual.subtract_interval(correction)) {
            return None;
        }
    }
    residual.enclosure()
}

// Operation counts in the scalar evaluator's per-mode accumulators.  They are
// kept beside the profile rather than written as anonymous roundoff factors:
// determinant value = at most exponential, multiply, divide, two logarithms,
// two additions/subtractions, and accumulator update;
// determinant first = fused h, the cancellation-free complement g/h, and sum;
// determinant second adds u and the product;
// residual value = at most two ratio divisions, scaling, and subtraction;
// residual first adds numerator product, division, and sum to the `u` path;
// residual second additionally forms `2u`, `1-2u`, and its product.
const DETERMINANT_VALUE_OPS_PER_MODE: usize = 8;
const RESIDUAL_VALUE_OPS_PER_MODE: usize = 4;
const RESIDUAL_LOG_OPS_PER_RESPONSE: usize = 3;
const SCORE_COMBINE_OPS: usize = 4;

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
        let mut zero_lambda_residual = Vec::with_capacity(responses);
        for (output, &energy) in response_energy.iter().enumerate() {
            let start = output * modes;
            let end = start + modes;
            zero_lambda_residual.push(
                certified_zero_lambda_residual(
                    energy,
                    gram_modes,
                    &projected_rhs_squared[start..end],
                )
                .ok_or(AffineRemlError::ZeroLambdaResidualUnavailable { output })?,
            );
        }
        Ok(Self {
            gram_modes,
            penalty_modes,
            projected_rhs_squared,
            response_energy,
            zero_lambda_residual,
            residual_dof,
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

    /// Nearest-rounded score value, first derivative, and second derivative in
    /// `log(lambda)`. [`Self::enclose`] supplies the proof-grade outer ranges.
    pub fn evaluate(&self, log_lambda: f64) -> Result<ScoreJet, AffineRemlError> {
        if !log_lambda.is_finite() {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }
        let lambda = certified_exp_representative(log_lambda)
            .ok_or(AffineRemlError::InvalidLogLambda { value: log_lambda })?;
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }

        let mut normalized_logdet = self.logdet_constant;
        let mut determinant_derivative = 0.0;
        let mut determinant_curvature = 0.0;
        let exp_neg_log_lambda = if log_lambda >= 0.0 {
            certified_exp_representative(-log_lambda)
        } else {
            None
        };
        for (index, (&gram, &penalty)) in self.gram_modes.iter().zip(self.penalty_modes).enumerate()
        {
            // A gram-zero penalized mode is structurally
            //
            //   log(exp(rho) s) - rho = log(s),
            //
            // with first and second derivatives exactly zero. Do not form
            // `exp(rho) s`: its rounded product may be zero or infinity even
            // though every normalized determinant quantity is finite.
            if gram == 0.0 {
                normalized_logdet +=
                    certified_ln_value(penalty).ok_or(AffineRemlError::NonPositiveMode {
                        index,
                        log_lambda,
                        value: penalty,
                    })?;
                continue;
            }
            let h = lambda.mul_add(penalty, gram);
            if !(h.is_finite() && h > 0.0) {
                return Err(AffineRemlError::NonPositiveMode {
                    index,
                    log_lambda,
                    value: h,
                });
            }
            let u = lambda * penalty / h;
            // For a penalized mode,
            //
            //   d/d rho [log(g + exp(rho)s) - rho]
            //     = u - 1 = -g/h,
            //
            // and its second derivative is u*g/h. Accumulate in that
            // cancellation-free complement currency rather than adding u to a
            // separately rounded `-rank`; the latter loses the derivative's
            // sign when u rounds to one. An unpenalized mode has no `-rho`
            // normalization and contributes exactly zero.
            let determinant_complement = if penalty == 0.0 { 0.0 } else { gram / h };
            // Accumulate the determinant in the normalized per-mode form
            // instead of forming two O(rho) quantities and subtracting them
            // after the sum. Both branches keep their exponential in (0, 1]:
            //
            // log(g + exp(rho)s) - rho
            //   = log(s + g exp(-rho))                         rho >= 0
            //   = log(g) - rho + log1p(exp(rho)s/g)            g dominates
            //   = log(s) + log1p(g/(exp(rho)s))                s dominates.
            //
            // The selected log1p ratio is always in [0, 1], so neither tail
            // forms an overflowing exponential or divides by its small term.
            let normalized_mode = if penalty == 0.0 {
                certified_ln_value(gram)
            } else if log_lambda >= 0.0 {
                exp_neg_log_lambda
                    .and_then(|exp_neg_rho| certified_ln_value(penalty + gram * exp_neg_rho))
            } else if gram >= penalty * lambda {
                certified_ln_value(gram)
                    .zip(certified_ln_1p_value(penalty * lambda / gram))
                    .map(|(log_gram, correction)| log_gram - log_lambda + correction)
            } else {
                certified_ln_value(penalty)
                    .zip(certified_ln_1p_value(gram / (penalty * lambda)))
                    .map(|(log_penalty, correction)| log_penalty + correction)
            }
            .ok_or(AffineRemlError::NonPositiveMode {
                index,
                log_lambda,
                value: h,
            })?;
            normalized_logdet += normalized_mode;
            determinant_derivative -= determinant_complement;
            determinant_curvature += u * determinant_complement;
        }

        let modes = self.num_modes();
        let mut residual_log_sum = 0.0;
        let mut residual_derivative_sum = 0.0;
        let mut residual_curvature_sum = 0.0;
        for (output, &energy) in self.response_energy.iter().enumerate() {
            let mut residual = energy;
            let mut first = 0.0;
            let mut second = 0.0;
            for i in 0..modes {
                let projected_square = self.projected_rhs_squared[output * modes + i];
                if projected_square == 0.0 {
                    continue;
                }
                if self.gram_modes[i] == 0.0 {
                    let fitted = positive_ratio_over_product(
                        projected_square,
                        self.penalty_modes[i],
                        lambda,
                    )
                    .ok_or(
                        AffineRemlError::ElementaryEnclosureUnavailable {
                            function: "gram-zero residual quotient",
                            lo: log_lambda,
                            hi: log_lambda,
                        },
                    )?;
                    residual -= fitted;
                    first += fitted;
                    second -= fitted;
                    continue;
                }
                let h = lambda.mul_add(self.penalty_modes[i], self.gram_modes[i]);
                let u = lambda * self.penalty_modes[i] / h;
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
            residual_log_sum += certified_ln_value(residual / self.residual_dof).ok_or(
                AffineRemlError::NonPositiveResidual {
                    output,
                    log_lambda,
                    value: residual,
                },
            )?;
            residual_derivative_sum += log_derivative;
            residual_curvature_sum += second / residual - log_derivative * log_derivative;
        }

        let outputs = self.num_responses() as f64;
        Ok(ScoreJet {
            value: -0.5 * (outputs * normalized_logdet + self.residual_dof * residual_log_sum),
            derivative: -0.5
                * (outputs * determinant_derivative + self.residual_dof * residual_derivative_sum),
            curvature: -0.5
                * (outputs * determinant_curvature + self.residual_dof * residual_curvature_sum),
            // This profile's companion `enclose` builds its own ranges from the
            // mode kernels on an interval lambda and centres them on the cell
            // MIDPOINT, so it never reads an endpoint jet's third derivative --
            // the third derivative it centres the curvature on is an interval
            // one it accumulates itself. Nothing reads this field for this
            // profile, and a scalar third derivative of a score whose two blocks
            // cancel would be the least trustworthy number here, so it stays
            // exactly zero rather than being computed and quietly relied on.
            third: 0.0,
        })
    }

    /// Outward enclosure of the score value and first two derivatives on a
    /// bounded log-lambda interval.
    ///
    /// # Two enclosures, intersected: the natural extension and the centred form
    ///
    /// `Self::enclose_direct` below is the NATURAL interval extension — each
    /// mode kernel evaluated on the interval lambda and summed. It is rigorous,
    /// and on the derivative and curvature it is also tight, because there the
    /// exact quantities are `O(1)` sums.
    ///
    /// On the score VALUE it is neither, and the reason is structural rather
    /// than incidental. The value is
    ///
    /// ```text
    ///     -0.5 * (D * normalized_logdet + residual_dof * sum_d log(R_d/dof))
    /// ```
    ///
    /// and near a REML optimum those two brackets cancel: each block's `d/drho`
    /// is `O(rank)` while their sum is not. Interval addition cannot see that
    /// the two variations are the same quantity with opposite signs, so the
    /// natural extension carries `rank * width` of slack the exact function does
    /// not have. Measured on a 33-mode profile over six decades of cell width,
    /// the value range came out at `33.0 * width` EXACTLY while the cell's own
    /// derivative enclosure bounded the score's variation across it by up to
    /// `7.4e5` times less — and the ratio DIVERGES as the cell shrinks, because
    /// one side is `O(w)` and the other `O(w^2)`.
    ///
    /// That is not a cosmetic loss. `maximize_score_1d` retires a cell as
    /// resolution-flat when its score range fits inside `2 * evaluation_error`;
    /// against an `O(w)` range that test needs a cell `rank/|f'|` times narrower
    /// than the function requires, so cells that ARE flat get subdivided, and a
    /// search that should finish in a handful of cells exhausts its subdivision
    /// budget and refuses a design it can certify.
    ///
    /// The cure is the standard one and needs nothing this routine does not
    /// already compute. For every `x` in `[a, b]` and `m` the midpoint, the mean
    /// value theorem gives
    ///
    /// ```text
    ///     f(x)  in  F({m})  + F'([a,b]) * [a-m, b-m]
    ///     f'(x) in  F'({m}) + F''([a,b]) * [a-m, b-m]
    /// ```
    ///
    /// with `F({m})` obtained by calling the SAME natural extension on the
    /// degenerate interval `[m, m]`. Both forms are outer enclosures of the same
    /// exact range, so their INTERSECTION is an outer enclosure too — this can
    /// only ever tighten, never widen, and it is never an acceptance tolerance.
    /// The centred form's overestimation is second order in the cell width,
    /// which is what makes the branch-and-bound converge.
    ///
    /// All three channels are centred, including the curvature: the profile's
    /// mode kernels are analytic, so `enclose_direct` also accumulates the exact
    /// third-derivative range (`t(1-4t+t^2)/(1+t)^4` per mode for the residual,
    /// `t(1-t)/(1+t)^3` for the determinant), and the curvature is centred on
    /// that. It matters: stationary isolation reads the curvature DIRECTLY and
    /// needs its sign, so a first-order-loose curvature is what stops a root
    /// being isolated rather than merely making a range wide.
    ///
    /// The `evaluation_error` is a property of the POINT evaluator over the
    /// cell, not of which enclosure form was tighter, so it is carried across
    /// unchanged from the whole-cell reading (the conservative one — the
    /// midpoint reading is taken over a degenerate interval and is never wider).
    ///
    /// # One consequence worth naming, because it points the other way
    ///
    /// `resolution_flat_region` retires a cell when its score range fits inside
    /// `2 * evaluation_error`, so a TIGHTER value range makes that verdict
    /// easier to reach — and a caller whose optimum lands in such a region gets
    /// a refusal (`ResidualCascadeError::RemlOptimumResolutionFlat`) rather than
    /// a fit. Tightening could in principle trade a subdivision-budget refusal
    /// for a resolution-flat one.
    ///
    /// It does not, because the flat test is the LAST thing a cell is offered:
    /// dominance, derivative exclusion and stationary isolation are all tried
    /// first, and centring strengthens each of them by more than it strengthens
    /// the flat test — the derivative and curvature ranges are what decide those
    /// three, and both are now centred too. Measured on the cascade design this
    /// was built for: the search returns `Stationary(0)` at every requested
    /// resolution from `1.49e-8` to `1e-3`, never `ResolutionFlat`, and
    /// `auto_reml_certifies_a_design_the_data_cannot_identify` asserts exactly
    /// that so the trade cannot creep in unnoticed.
    pub fn enclose(&self, lo: f64, hi: f64) -> Result<DerivativeEnclosure, AffineRemlError> {
        let (direct, direct_third) = self.enclose_direct(lo, hi)?;
        if lo == hi {
            return Ok(direct);
        }
        // Any point of the cell is a valid expansion centre; the midpoint
        // minimizes the worst-case `|x - m|` and so the width of the remainder
        // term. Clamped because `0.5*(lo+hi)` may round outside a cell whose
        // endpoints are adjacent floats, and a centre outside the cell would
        // make the mean value theorem inapplicable.
        let centre_point = (0.5 * (lo + hi)).clamp(lo, hi);
        let (centre, _) = self.enclose_direct(centre_point, centre_point)?;
        // `[a-m, b-m]`, rounded OUTWARD. Both subtractions are exact by
        // Sterbenz whenever the endpoints are within a factor of two of the
        // centre, which is the usual case; the directed widening costs one ulp
        // and removes the need to prove it.
        let offset = ClosedInterval::new(
            next_down(lo - centre_point).min(0.0),
            next_up(hi - centre_point).max(0.0),
        );
        // The three channels are centred in DERIVATIVE ORDER, each on the result
        // of the one above, because a mean value remainder is only as tight as
        // the range fed into it: the curvature's remainder is `F'''*offset`, the
        // derivative's is `F''*offset`, and the value's is `F'*offset`. Feeding
        // each the natural extension's range instead of the centred one leaves a
        // constant factor on the floor at every level — measured at 3.3x on the
        // value alone — and it is this cascade that makes
        // `width(F) <= width(F({m})) + max|F'| * w` hold BY CONSTRUCTION against
        // the ranges this function actually returns.
        let curvature = centred_or(direct.curvature, centre.curvature, direct_third, offset);
        let derivative = centred_or(direct.derivative, centre.derivative, curvature, offset);
        let value = centred_or(direct.score.value, centre.score.value, derivative, offset);
        Ok(DerivativeEnclosure {
            score: ScoreValueEnclosure {
                value,
                evaluation_error: direct.score.evaluation_error,
            },
            derivative,
            curvature,
        })
    }

    /// The natural (direct) interval extension: every mode kernel evaluated on
    /// the interval lambda and accumulated.
    ///
    /// The interval kernels enclose the exact-real ranges. The score value uses
    /// the same cancellation-free normalized determinant identity as
    /// [`Self::evaluate`]. Its separate `evaluation_error` charges each
    /// source-derived elementary-function interval, error propagation through
    /// `log(residual)`, and Wilkinson `gamma_k * sum |term|` bounds for the
    /// actual sequential accumulators.
    ///
    /// [`Self::enclose`] is what callers want: this form alone is first-order
    /// loose on the value, for the reason documented there.
    fn enclose_direct(
        &self,
        lo: f64,
        hi: f64,
    ) -> Result<(DerivativeEnclosure, ClosedInterval), AffineRemlError> {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }
        let lambda = exp_interval(lo, hi)?;
        if !(lambda.lo.is_finite() && lambda.lo > 0.0 && lambda.hi.is_finite()) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }
        // Exp's range-reduction, arithmetic, and truncation errors are
        // multiplicative and must remain in relative currency across a wide
        // interval. Only gradual underflow is additive and is charged against
        // the certified positive lower endpoint. This avoids coupling an
        // upper-endpoint absolute error to the lower-endpoint scale.
        let lambda_relative_error =
            certified_exp_relative_forward_error(ClosedInterval::new(lo, hi), lambda);
        if !lambda_relative_error.is_finite() {
            return Err(AffineRemlError::UnboundedScoreEvaluationError {
                lo,
                hi,
                error: lambda_relative_error,
            });
        }

        let mut normalized_logdet = ClosedInterval::point(self.logdet_constant);
        let mut normalized_logdet_magnitude = self.logdet_constant.abs();
        let mut normalized_logdet_error = 0.0;
        let mut determinant_first = ClosedInterval::point(0.0);
        let mut determinant_second = ClosedInterval::point(0.0);
        let mut determinant_third = ClosedInterval::point(0.0);
        for i in 0..self.num_modes() {
            let (normalized_mode, normalized_mode_error) =
                normalized_log_mode_enclosure(self.gram_modes[i], self.penalty_modes[i], lo, hi)?;
            normalized_logdet = normalized_logdet.add(normalized_mode);
            normalized_logdet_magnitude = add_nonnegative_upward(
                normalized_logdet_magnitude,
                add_nonnegative_upward(normalized_mode.max_abs(), normalized_mode_error),
            );
            normalized_logdet_error =
                add_nonnegative_upward(normalized_logdet_error, normalized_mode_error);

            let ranges = mode_ranges(self.gram_modes[i], self.penalty_modes[i], 0.0, lambda)?;
            determinant_first = determinant_first.sub(ranges.c);
            determinant_second = determinant_second.add(ranges.w);
            determinant_third = determinant_third.add(ranges.determinant_third);
        }

        let mut residual_first_sum = ClosedInterval::point(0.0);
        let mut residual_second_sum = ClosedInterval::point(0.0);
        let mut residual_third_sum = ClosedInterval::point(0.0);
        let mut residual_log_sum = ClosedInterval::point(0.0);
        let mut residual_log_magnitude = 0.0;
        let mut residual_log_error = 0.0;
        let modes = self.num_modes();
        for (output, &energy) in self.response_energy.iter().enumerate() {
            let mut fitted_quadratic = ClosedInterval::point(0.0);
            let mut smoothing_increment = ClosedInterval::point(0.0);
            let mut singular_fitted = ClosedInterval::point(0.0);
            let mut first = ClosedInterval::point(0.0);
            let mut second = ClosedInterval::point(0.0);
            let mut third = ClosedInterval::point(0.0);
            let mut fitted_magnitude = energy;
            for i in 0..modes {
                let ranges = mode_ranges(
                    self.gram_modes[i],
                    self.penalty_modes[i],
                    self.projected_rhs_squared[output * modes + i],
                    lambda,
                )?;
                fitted_quadratic = fitted_quadratic.add(ranges.v);
                smoothing_increment = smoothing_increment.add(ranges.smoothing_increment);
                singular_fitted = singular_fitted.add(ranges.singular_fitted);
                first = first.add(ranges.p);
                second = second.add(ranges.q);
                third = third.add(ranges.residual_third);
                fitted_magnitude = add_nonnegative_upward(fitted_magnitude, ranges.v.max_abs());
            }
            // Two exact identities describe the same residual:
            //
            //   R = E - sum_i q_i/(g_i + lambda*s_i)
            //
            // and, for every positive-Gram mode,
            //
            //   q_i/(g_i + lambda*s_i)
            //     = q_i/g_i - (q_i/g_i) * lambda*s_i/(g_i + lambda*s_i).
            //
            // The direct form is well-conditioned away from interpolation.
            // Near the zero-smoothing face it subtracts many independently
            // rounded near-one fitted fractions from `E`, even though their
            // deviations from one are perfectly correlated with lambda.  The
            // complement form carries that correlation explicitly as a
            // fixed zero-smoothing residual plus nonnegative smoothing
            // increments.  Both are rigorous outer enclosures, so their
            // intersection is rigorous and never an acceptance tolerance.
            let direct_residual = ClosedInterval::point(energy).sub(fitted_quadratic);
            let complement_residual = self.zero_lambda_residual[output]
                .add(smoothing_increment)
                .sub(singular_fitted);
            let residual = direct_residual.intersection(complement_residual).ok_or(
                AffineRemlError::InconsistentResidualEnclosures {
                    output,
                    lo,
                    hi,
                    direct: direct_residual,
                    complement: complement_residual,
                },
            )?;
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
            let third_ratio = third.div_positive(residual);
            residual_first_sum = residual_first_sum.add(first_ratio);
            residual_second_sum = residual_second_sum.add(second_ratio.sub(first_ratio.square()));
            // The third derivative of `log R`, from the same three ratios:
            //   (log R)''' = R'''/R - 3 (R''/R)(R'/R) + 2 (R'/R)^3.
            residual_third_sum = residual_third_sum.add(
                third_ratio
                    .sub(second_ratio.mul(first_ratio).scale(3.0))
                    .add(first_ratio.square().mul(first_ratio).scale(2.0)),
            );

            let fitted_arithmetic_error = wilkinson_roundoff(
                fitted_magnitude,
                modes.saturating_mul(RESIDUAL_VALUE_OPS_PER_MODE),
            );
            // `first = d fitted/d rho`, so the MVT propagates the exp error in
            // rho-space without a condition-number guess.
            let fitted_exp_error = next_up(first.max_abs() * lambda_relative_error);
            let resolved_fitted_quadratic = fitted_quadratic.widen(add_nonnegative_upward(
                fitted_arithmetic_error,
                fitted_exp_error,
            ));
            let resolved_residual = ClosedInterval::point(energy).sub(resolved_fitted_quadratic);
            if !(resolved_residual.lo > 0.0 && resolved_residual.is_valid()) {
                return Err(AffineRemlError::NonPositiveResidualInterval {
                    output,
                    lo,
                    hi,
                    lower_bound: resolved_residual.lo,
                });
            }
            let residual_over_dof = residual.div_positive(ClosedInterval::point(self.residual_dof));
            if !(residual_over_dof.lo > 0.0 && residual_over_dof.hi.is_finite()) {
                return Err(AffineRemlError::ElementaryEnclosureUnavailable {
                    function: "ln",
                    lo: residual_over_dof.lo,
                    hi: residual_over_dof.hi,
                });
            }
            let residual_log = residual_over_dof.ln_positive();
            residual_log_sum = residual_log_sum.add(residual_log);

            // `evaluate` first forms the residual and then takes
            // `ln(residual/dof)`. The residual forward error is already
            // represented by `resolved_residual`. On the strictly positive
            // resolved range, the mean-value theorem bounds its propagation
            // through log by `delta_R / min(R)`. The division and logarithm
            // each add one directed basic-operation contribution; the
            // source-derived logarithm error is absolute, so it remains valid
            // when the logarithm's result is near zero.
            let residual_error = enclosure_excess(residual, resolved_residual);
            let propagated_residual_error = next_up(residual_error / resolved_residual.lo);
            let elementary_error = certified_log_forward_error(
                residual.div_positive(ClosedInterval::point(self.residual_dof)),
            );
            let local_log_error = add_nonnegative_upward(
                propagated_residual_error,
                add_nonnegative_upward(
                    elementary_error,
                    wilkinson_roundoff(
                        add_nonnegative_upward(1.0, residual_log.max_abs()),
                        RESIDUAL_LOG_OPS_PER_RESPONSE,
                    ),
                ),
            );
            residual_log_error = add_nonnegative_upward(residual_log_error, local_log_error);
            residual_log_magnitude = add_nonnegative_upward(
                residual_log_magnitude,
                add_nonnegative_upward(residual_log.max_abs(), local_log_error),
            );
        }

        let outputs = self.num_responses() as f64;
        let first_bracket = determinant_first
            .scale(outputs)
            .add(residual_first_sum.scale(self.residual_dof));
        let second_bracket = determinant_second
            .scale(outputs)
            .add(residual_second_sum.scale(self.residual_dof));
        let third_bracket = determinant_third
            .scale(outputs)
            .add(residual_third_sum.scale(self.residual_dof));
        let derivative = first_bracket.scale(-0.5);
        let curvature = second_bracket.scale(-0.5);
        let third = third_bracket.scale(-0.5);
        let score_value = normalized_logdet
            .scale(outputs)
            .add(residual_log_sum.scale(self.residual_dof))
            .scale(-0.5);
        let score_magnitude = add_nonnegative_upward(
            next_up(outputs * normalized_logdet_magnitude),
            next_up(self.residual_dof * residual_log_magnitude),
        );
        normalized_logdet_error = add_nonnegative_upward(
            normalized_logdet_error,
            wilkinson_roundoff(normalized_logdet_magnitude, self.num_modes()),
        );
        residual_log_error = add_nonnegative_upward(
            residual_log_error,
            wilkinson_roundoff(residual_log_magnitude, self.num_responses()),
        );
        let final_arithmetic_error = wilkinson_roundoff(score_magnitude, SCORE_COMBINE_OPS);
        let weighted_component_error = add_nonnegative_upward(
            next_up(outputs * normalized_logdet_error),
            next_up(self.residual_dof * residual_log_error),
        );
        let value_evaluation_error =
            next_up(0.5 * add_nonnegative_upward(weighted_component_error, final_arithmetic_error));
        if !(score_value.is_valid() && value_evaluation_error.is_finite()) {
            return Err(AffineRemlError::UnboundedScoreEvaluationError {
                lo,
                hi,
                error: value_evaluation_error,
            });
        }
        let score = ScoreValueEnclosure {
            value: score_value,
            evaluation_error: value_evaluation_error,
        };
        Ok((
            DerivativeEnclosure {
                score,
                derivative,
                curvature,
            },
            third,
        ))
    }

    /// Isolate every finite stationary candidate and tighten location
    /// resolution until the selected exact score is globally orderable at the
    /// point evaluator's certified comparison resolution.
    ///
    /// The first pass honors the caller's requested location resolution.  A
    /// successful root isolation can still leave a wider exact score range
    /// than the rounded point comparison can distinguish, because location and
    /// value are different currencies.  In that case this repeats the same
    /// exact search with a smaller location target.  The observed ratio between
    /// comparison resolution and maximum excess is only an iteration strategy,
    /// never proof currency; every pass independently rebuilds the complete
    /// global certificate and the loop exits only on its verdict.
    ///
    /// There is no retry cap or acceptance fallback.  Each retry contracts the
    /// target by at least one binary subdivision.  If the target can no longer
    /// be represented, or the oracle cannot resolve structure at that finer
    /// target, the last complete certificate is returned unchanged for the
    /// caller's existing typed refusal.
    pub fn maximize_value_ordered(
        &self,
        lo: f64,
        hi: f64,
        initial_resolution: f64,
    ) -> Result<ScoreSearchResult, ScoreSearchError<AffineRemlError>> {
        maximize_score_1d_value_ordered(
            lo,
            hi,
            initial_resolution,
            |x| self.evaluate(x),
            |a, b| self.enclose(a.x, b.x),
        )
    }
}

#[derive(Clone, Copy)]
struct ModeRanges {
    /// Cancellation-free determinant complement `c = g/h` for a penalized
    /// mode. An unpenalized mode contributes exactly zero because its
    /// normalized log determinant has no `-rho` term.
    c: ClosedInterval,
    /// `u(1-u)`.
    w: ClosedInterval,
    /// `projected_square / h`.
    v: ClosedInterval,
    /// The nonnegative loss of fitted energy caused by smoothing,
    /// `(projected_square / gram) * lambda*penalty / h`, for a positive Gram
    /// mode.
    smoothing_increment: ClosedInterval,
    /// The complete fitted contribution of a Gram-zero mode. Such a mode
    /// cannot participate in the zero-smoothing complement identity.
    singular_fitted: ClosedInterval,
    /// First derivative of the residual contribution:
    /// `projected_square * lambda s / h^2`.
    p: ClosedInterval,
    /// Second derivative of the residual contribution:
    /// `projected_square * lambda s (g-lambda s) / h^3`.
    q: ClosedInterval,
    /// Third `rho`-derivative of this mode's normalized log determinant,
    /// `u(1-u)(1-2u) = t(1-t)/(1+t)^3`. Exactly the `k` kernel: the
    /// determinant's second derivative is `w` and its third is `k`, which is
    /// also the fitted fraction's second. Zero for an unpenalized mode (no
    /// `-rho` term) and for a Gram-zero mode (whose normalized determinant is
    /// exactly constant).
    determinant_third: ClosedInterval,
    /// Third derivative of the residual contribution.
    residual_third: ClosedInterval,
}

/// Exact-real range and a uniform forward-error bound for the normalized
/// determinant contribution of one affine mode.
///
/// The exact function is monotone, so outward endpoint evaluation gives its
/// range. [`AffineRemlProfile::evaluate`] uses algebraically equivalent stable
/// sign/dominance branches whose exponential or `ln_1p` argument is in
/// `[0, 1]`. The elementary-function bounds come from the source-derived
/// range-reduced series above, not a platform-libm accuracy assumption;
/// Wilkinson's bound charges the surrounding IEEE basic operations and the
/// elementary input perturbation. The leading `1` is the analytic sensitivity
/// bound: the mode's rho derivative lies in `[-1, 0]`.
fn normalized_log_mode_enclosure(
    gram: f64,
    penalty: f64,
    lo: f64,
    hi: f64,
) -> Result<(ClosedInterval, f64), AffineRemlError> {
    if penalty == 0.0 {
        let range = ClosedInterval::point(gram).ln_positive();
        return Ok((
            range,
            certified_log_forward_error(ClosedInterval::point(gram)),
        ));
    }
    if gram == 0.0 {
        let range = ClosedInterval::point(penalty).ln_positive();
        return Ok((
            range,
            certified_log_forward_error(ClosedInterval::point(penalty)),
        ));
    }

    let at_lo = normalized_log_mode_at(gram, penalty, lo)?;
    let at_hi = normalized_log_mode_at(gram, penalty, hi)?;
    // The normalized contribution has derivative `u - 1` in [-1, 0].
    let range = ClosedInterval::new(at_hi.lo, at_lo.hi);
    // In the negative branch the final subtraction can cancel `log(h)` and
    // `rho`; charge both pre-cancellation operands. Since
    // `log(h) = normalized_mode + rho`, `|mode| + 2|rho|` bounds their absolute
    // sum without evaluating a second logarithm.
    let negative_rho_abs = if lo < 0.0 { -lo } else { 0.0 };
    let arithmetic_scale = add_nonnegative_upward(
        add_nonnegative_upward(1.0, range.max_abs()),
        next_up(2.0 * negative_rho_abs),
    );
    let arithmetic_error = wilkinson_roundoff(arithmetic_scale, DETERMINANT_VALUE_OPS_PER_MODE);
    let mut exp_input_error = 0.0_f64;
    if hi >= 0.0 {
        let positive_lo = lo.max(0.0);
        let exp_neg_rho = exp_interval(-hi, -positive_lo)?;
        let argument_lo = ClosedInterval::point(penalty)
            .add(ClosedInterval::point(gram).mul(exp_neg_rho))
            .lo;
        if argument_lo > 0.0 {
            exp_input_error = exp_input_error.max(next_up(
                gram * certified_exp_forward_error(
                    ClosedInterval::new(-hi, -positive_lo),
                    exp_neg_rho,
                ) / argument_lo,
            ));
        } else {
            exp_input_error = f64::INFINITY;
        }
    }
    if lo < 0.0 {
        let negative_hi = hi.min(0.0);
        let exp_rho = exp_interval(lo, negative_hi)?;
        if exp_rho.lo > 0.0 {
            // The two stable negative-rho branches have log-lambda
            // sensitivities `u` and `1-u`, respectively. Both are at most one,
            // so the scale-safe relative exp error is a uniform bound even if
            // the dominance branch changes.
            exp_input_error = exp_input_error.max(certified_exp_relative_forward_error(
                ClosedInterval::new(lo, negative_hi),
                exp_rho,
            ));
        } else {
            exp_input_error = f64::INFINITY;
        }
    }
    let log_output_error =
        certified_log_error_from_output(at_lo).max(certified_log_error_from_output(at_hi));
    let log_gram_error = certified_log_forward_error(ClosedInterval::point(gram));
    let log_penalty_error = certified_log_forward_error(ClosedInterval::point(penalty));
    let log1p_error = certified_ln1p_forward_error();
    let elementary_error = add_nonnegative_upward(
        exp_input_error,
        add_nonnegative_upward(
            log_output_error,
            add_nonnegative_upward(
                log_gram_error,
                add_nonnegative_upward(log_penalty_error, log1p_error),
            ),
        ),
    );
    Ok((
        range,
        add_nonnegative_upward(arithmetic_error, elementary_error),
    ))
}

fn normalized_log_mode_at(
    gram: f64,
    penalty: f64,
    rho: f64,
) -> Result<ClosedInterval, AffineRemlError> {
    if rho >= 0.0 {
        let exp_neg_rho = exp_interval(-rho, -rho)?;
        let argument =
            ClosedInterval::point(penalty).add(ClosedInterval::point(gram).mul(exp_neg_rho));
        if !(argument.lo > 0.0 && argument.hi.is_finite()) {
            return Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "ln",
                lo: argument.lo,
                hi: argument.hi,
            });
        }
        Ok(argument.ln_positive())
    } else {
        let exp_rho = exp_interval(rho, rho)?;
        let argument = ClosedInterval::point(gram).add(ClosedInterval::point(penalty).mul(exp_rho));
        if !(argument.lo > 0.0 && argument.hi.is_finite()) {
            return Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "ln",
                lo: argument.lo,
                hi: argument.hi,
            });
        }
        Ok(argument.ln_positive().sub(ClosedInterval::point(rho)))
    }
}

fn exp_interval(lo: f64, hi: f64) -> Result<ClosedInterval, AffineRemlError> {
    let unavailable = || AffineRemlError::ElementaryEnclosureUnavailable {
        function: "exp",
        lo,
        hi,
    };
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return Err(unavailable());
    }
    let lower = certified_exp(lo).ok_or_else(unavailable)?;
    let upper = certified_exp(hi).ok_or_else(unavailable)?;
    let enclosure = ClosedInterval::new(lower.lo.max(0.0), upper.hi).nonnegative();
    if !enclosure.is_valid() {
        return Err(unavailable());
    }
    Ok(enclosure)
}

/// Directed division for a nonnegative numerator and a strictly positive
/// denominator without first materializing the reciprocal.
///
/// Forming `1 / denominator.lo` can overflow even when the final quotient is
/// finite because a correspondingly tiny numerator cancels that scale. Direct
/// endpoint quotients preserve that finite result. Invalid preconditions and a
/// nonfinite upper bound are typed refusals rather than assertions.
fn finite_nonnegative_quotient(
    numerator: ClosedInterval,
    denominator: ClosedInterval,
    function: &'static str,
) -> Result<ClosedInterval, AffineRemlError> {
    if !(numerator.is_valid()
        && numerator.lo >= 0.0
        && denominator.is_valid()
        && denominator.lo > 0.0)
    {
        return Err(AffineRemlError::ElementaryEnclosureUnavailable {
            function,
            lo: denominator.lo,
            hi: denominator.hi,
        });
    }
    let quotient = ClosedInterval::new(
        quotient_down(numerator.lo, denominator.hi).max(0.0),
        quotient_up(numerator.hi, denominator.lo),
    );
    if !(quotient.is_valid() && quotient.hi.is_finite()) {
        return Err(AffineRemlError::ElementaryEnclosureUnavailable {
            function,
            lo: quotient.lo,
            hi: quotient.hi,
        });
    }
    Ok(quotient.nonnegative())
}

/// One channel of the centred (mean value) enclosure, intersected with the
/// natural extension — and never trusted over it when the remainder is not a
/// finite interval.
///
/// `f(x) in point + slope * offset` for every `x` in the cell, by the mean value
/// theorem, when `point` encloses `f` (or `f'`, or `f''`) at the expansion
/// centre and `slope` encloses the NEXT derivative over the whole cell. Both
/// forms are outer enclosures of one exact range, so the intersection is an
/// outer enclosure too: this can only tighten.
///
/// # What the finiteness guard is for, measured
///
/// `ClosedInterval::mul` reduces four endpoint products with `f64::min` and
/// `f64::max`, which IGNORE a NaN operand, so a NaN product drops out of the
/// reduction and the surviving endpoints describe a range strictly INSIDE the
/// true one — an unsound certificate, in the one direction that matters, with no
/// signal at all.
///
/// The obvious way in is `inf * 0`, and that way is already shut:
/// `product_down`/`product_up` treat a zero operand as exact and map the NaN to
/// `0.0`, and a sweep over every endpoint shape finds no narrowing from a
/// singly-infinite slope. The way that is NOT shut is a NaN arriving from
/// anywhere else — `[NaN, 1.0] * [-0.5, 0.5]` reduces to `[-0.5, 0.5]`, two
/// corners silently gone — because `enclose_direct` does not prove every
/// accumulator finite and `checked_enclosure` validates only the enclosure the
/// search receives, after this narrowing would already have happened.
///
/// So the guard excludes a non-finite slope and a non-finite remainder, and
/// keeps the natural extension, which is rigorous unconditionally. See
/// `the_centred_form_keeps_the_natural_extension_when_the_remainder_is_not_finite`
/// for both halves as assertions.
fn centred_or(
    direct: ClosedInterval,
    point: ClosedInterval,
    slope: ClosedInterval,
    offset: ClosedInterval,
) -> ClosedInterval {
    if !(slope.is_valid() && slope.lo.is_finite() && slope.hi.is_finite()) {
        return direct;
    }
    let remainder = slope.mul(offset);
    if !(remainder.is_valid() && remainder.lo.is_finite() && remainder.hi.is_finite()) {
        return direct;
    }
    let centred = point.add(remainder);
    if !centred.is_valid() {
        return direct;
    }
    // Two rigorous outer enclosures of the same nonempty exact range cannot be
    // disjoint, so this fallback is unreachable; it is written in the sound
    // direction rather than as a panic, because a refusal here would convert a
    // tightening into a failure.
    direct.intersection(centred).unwrap_or(direct)
}

fn mode_ranges(
    gram: f64,
    penalty: f64,
    projected_square: f64,
    lambda: ClosedInterval,
) -> Result<ModeRanges, AffineRemlError> {
    if penalty == 0.0 {
        let v = ClosedInterval::point(projected_square)
            .div_positive(ClosedInterval::point(gram))
            .nonnegative();
        return Ok(ModeRanges {
            c: ClosedInterval::point(0.0),
            w: ClosedInterval::point(0.0),
            v,
            smoothing_increment: ClosedInterval::point(0.0),
            singular_fitted: ClosedInterval::point(0.0),
            p: ClosedInterval::point(0.0),
            q: ClosedInterval::point(0.0),
            determinant_third: ClosedInterval::point(0.0),
            residual_third: ClosedInterval::point(0.0),
        });
    }
    if gram == 0.0 {
        let zero = ClosedInterval::point(0.0);
        if projected_square == 0.0 {
            return Ok(ModeRanges {
                c: zero,
                w: zero,
                v: zero,
                smoothing_increment: zero,
                singular_fitted: zero,
                p: zero,
                q: zero,
                determinant_third: zero,
                residual_third: zero,
            });
        }

        // The normalized determinant is exactly constant for g=0. For the
        // residual, v = A/(lambda*s). Use the direct product only when its
        // outward lower bound is strictly positive. If that lower bound rounds
        // to zero, cancel the exact scalar penalty first and divide the
        // resulting nonnegative interval directly by lambda. This preserves a
        // finite quotient such as min_subnormal/(0.01*lambda) without ever
        // asking `div_positive` to accept a denominator containing zero.
        let h = lambda.mul(ClosedInterval::point(penalty)).nonnegative();
        let projected = ClosedInterval::point(projected_square);
        let v = if h.lo > 0.0 {
            finite_nonnegative_quotient(projected, h, "gram-zero residual quotient")?
        } else {
            let scaled = finite_nonnegative_quotient(
                projected,
                ClosedInterval::point(penalty),
                "gram-zero residual quotient",
            )?;
            finite_nonnegative_quotient(scaled, lambda, "gram-zero residual quotient")?
        };
        return Ok(ModeRanges {
            c: ClosedInterval::point(0.0),
            w: ClosedInterval::point(0.0),
            v,
            smoothing_increment: zero,
            singular_fitted: v,
            p: v,
            q: v.neg(),
            // A Gram-zero mode's fitted fraction is `A/(lambda s)`, whose
            // rho-derivative is its own negative, so the residual's successive
            // derivatives alternate in sign at constant magnitude.
            determinant_third: zero,
            residual_third: v,
        });
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
    Ok(ModeRanges {
        c: kernels.v,
        w: kernels.w,
        v: scale.mul(kernels.v).nonnegative(),
        smoothing_increment: scale.mul(kernels.u).nonnegative(),
        singular_fitted: ClosedInterval::point(0.0),
        p: scale.mul(kernels.w).nonnegative(),
        q: scale.mul(kernels.k),
        determinant_third: kernels.k,
        residual_third: scale.mul(kernels.third),
    })
}

#[derive(Clone, Copy)]
struct KernelRanges {
    /// `1/(1+t)`.
    v: ClosedInterval,
    /// `t/(1+t)`.
    u: ClosedInterval,
    /// `t/(1+t)^2`.
    w: ClosedInterval,
    /// `t(1-t)/(1+t)^3`.
    k: ClosedInterval,
    /// `t(1 - 4t + t^2)/(1+t)^4`, the rho-derivative of `k`.
    ///
    /// With `dt/drho = t`, differentiating `k` once more gives
    /// `t * dk/dt`, and `dk/dt = (1 - 4t + t^2)/(1+t)^4` because
    /// `k = (t - t^2)(1+t)^-3`. It is the third rho-derivative of a mode's
    /// fitted fraction (up to the scale `A/g`), which is what the residual
    /// block's `(log R)'''` is built from.
    ///
    /// Checked against finite differences of `q/(g + e^rho s)` in rho at
    /// `rho = -2, -0.5, 0.3, 1.7`: agreement to five significant figures at
    /// `h = 1e-2`, which is that FD's own `O(h^2)` truncation. The determinant's
    /// third derivative needed no new kernel at all — `u(1-u)(1-2u)` is
    /// `t(1-t)/(1+t)^3`, exactly `k` — and was checked the same way.
    third: ClosedInterval,
}

fn kernel_at(t: ClosedInterval) -> KernelRanges {
    let one = ClosedInterval::point(1.0);
    let denom = one.add(t);
    let v = one.div_positive(denom).nonnegative();
    let u = t.mul(v).nonnegative();
    let w = u.mul(v).nonnegative();
    let k = w.mul(one.sub(t)).div_positive(denom);
    // `t(1 - 4t + t^2)/(1+t)^4 = w * (1 - 4t + t^2)/(1+t)^2`. The numerator is
    // signed, which `mul` handles; the denominator is a square of a strictly
    // positive interval.
    let third = w
        .mul(one.sub(t.scale(4.0)).add(t.square()))
        .div_positive(denom.square());
    KernelRanges { v, u, w, k, third }
}

fn kernel_ranges(t: ClosedInterval) -> KernelRanges {
    let left = kernel_at(ClosedInterval::point(t.lo));
    let right = kernel_at(ClosedInterval::point(t.hi));
    let mut v = ClosedInterval::new(right.v.lo, left.v.hi).nonnegative();
    let u = ClosedInterval::new(left.u.lo, right.u.hi).nonnegative();
    let mut w = left.w.hull(right.w).nonnegative();
    let mut k = left.k.hull(right.k);
    let mut third = left.third.hull(right.third);

    if t.contains(1.0) {
        let critical = kernel_at(ClosedInterval::point(1.0));
        w = w.hull(critical.w).nonnegative();
        // `d/dt [t(1-4t+t^2)/(1+t)^4] = (1 - 11t + 11t^2 - t^3)/(1+t)^5`, and
        // `t^3 - 11t^2 + 11t - 1 = (t-1)(t^2 - 10t + 1)`, so `t = 1` is one of
        // this kernel's three critical points as well.
        third = third.hull(critical.third);
    }

    // k'(t) has its only positive roots at 2 +/- sqrt(3).  Enclose sqrt(3)
    // itself before subtraction/addition so the exact irrational critical
    // points are not lost to nearest-rounded scalar arithmetic.
    let sqrt_three =
        certified_sqrt_positive(3.0).expect("three is a finite positive square-root argument");
    let critical_points = [
        ClosedInterval::point(2.0).sub(sqrt_three),
        ClosedInterval::point(2.0).add(sqrt_three),
    ];
    for critical in critical_points {
        if critical.hi >= t.lo && critical.lo <= t.hi {
            k = k.hull(kernel_at(critical).k);
        }
    }

    // The remaining two roots of `t^2 - 10t + 1` are `5 +/- 2 sqrt(6)`,
    // enclosed before the addition so the exact irrationals survive.
    let sqrt_six =
        certified_sqrt_positive(6.0).expect("six is a finite positive square-root argument");
    let two_sqrt_six = sqrt_six.scale(2.0);
    for critical in [
        ClosedInterval::point(5.0).sub(two_sqrt_six),
        ClosedInterval::point(5.0).add(two_sqrt_six),
    ] {
        if critical.hi >= t.lo && critical.lo <= t.hi {
            third = third.hull(kernel_at(critical).third);
        }
    }

    // Monotonicity gives tighter endpoint ranges than a dependency-heavy
    // interval evaluation, but retain outward endpoint arithmetic.
    v.lo = v.lo.max(0.0);
    v.hi = v.hi.min(next_up(1.0));
    KernelRanges {
        v,
        u,
        w,
        k,
        third,
    }
}

const LOG_SERIES_TERMS: usize = 18;
const EXP_SERIES_TERMS: usize = 18;
const EXP_RANGE_SQUARINGS: usize = 6;

fn certified_sqrt_positive(value: f64) -> Option<ClosedInterval> {
    if !(value.is_finite() && value > 0.0) {
        return None;
    }
    // `sqrt` supplies only a starting guess. Directed squaring proves and, if
    // necessary, expands the two sides, so no platform sqrt accuracy contract
    // is a premise of the returned interval.
    let guess = value.sqrt();
    if !(guess.is_finite() && guess > 0.0) {
        return None;
    }
    let mut lo = next_down(guess);
    for _ in 0..8 {
        if ClosedInterval::point(lo).square().hi <= value {
            break;
        }
        lo = next_down(lo);
    }
    let mut hi = next_up(guess);
    for _ in 0..8 {
        if ClosedInterval::point(hi).square().lo >= value {
            break;
        }
        hi = next_up(hi);
    }
    (ClosedInterval::point(lo).square().hi <= value
        && ClosedInterval::point(hi).square().lo >= value)
        .then(|| ClosedInterval::new(lo, hi))
}

/// `2·atanh(z)` by its positive odd-power series, with the omitted tail
/// bounded geometrically. The caller supplies `|z| <= 1/3`.
fn certified_log_from_atanh(z: ClosedInterval) -> ClosedInterval {
    let z_abs = z.max_abs();
    assert!(z_abs <= 1.0 / 3.0 + f64::EPSILON);
    let z2 = z.square();
    let mut power = z;
    let mut sum = z;
    for term in 1..LOG_SERIES_TERMS {
        power = power.mul(z2);
        sum = sum.add(power.div_positive(ClosedInterval::point((2 * term + 1) as f64)));
    }
    let next_power = power.mul(z2).max_abs();
    let first_denominator = (2 * LOG_SERIES_TERMS + 1) as f64;
    let geometric_denominator = next_down(1.0 - next_up(z_abs * z_abs));
    let tail = if geometric_denominator > 0.0 {
        next_up(next_up(2.0 * next_power) / next_down(first_denominator * geometric_denominator))
    } else {
        f64::INFINITY
    };
    sum.scale(2.0).widen(tail)
}

fn certified_ln_two() -> ClosedInterval {
    static LN_TWO: OnceLock<ClosedInterval> = OnceLock::new();
    *LN_TWO.get_or_init(|| {
        // ln(2) = 2 atanh(1/3). Both the rational 1/3 and the series are
        // evaluated with directed IEEE basic operations; no platform libm
        // result participates in this constant.
        let third = ClosedInterval::point(1.0).div_positive(ClosedInterval::point(3.0));
        certified_log_from_atanh(third)
    })
}

/// Exact decomposition `value = mantissa * 2^exponent` with
/// `mantissa in [1, 2)` for every positive finite binary64 value.
fn positive_binary64_parts(value: f64) -> Option<(f64, i32)> {
    if !(value.is_finite() && value > 0.0) {
        return None;
    }
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        // value = fraction*2^-1074. Normalize the integer significand into
        // [2^52,2^53), then install it under exponent zero.
        let highest = 63_i32 - fraction.leading_zeros() as i32;
        let normalized = fraction << (52 - highest);
        let mantissa_bits = (1023_u64 << 52) | (normalized - (1_u64 << 52));
        Some((f64::from_bits(mantissa_bits), highest - 1074))
    } else {
        let mantissa_bits = (1023_u64 << 52) | fraction;
        Some((f64::from_bits(mantissa_bits), exponent_bits - 1023))
    }
}

/// Rigorous exact-real enclosure of `ln(value)` for every finite positive
/// binary64 input, including subnormals.
///
/// Bit decomposition writes `value = m·2^k` exactly with `m in [1,2)`.
/// `ln(m) = 2·atanh((m-1)/(m+1))` then has `z in [0,1/3]`, so the fixed
/// positive series above has a closed geometric remainder. Only directed
/// binary64 basic operations are used.
pub fn certified_ln_positive(value: f64) -> Option<ClosedInterval> {
    if !(value.is_finite() && value > 0.0) {
        return None;
    }
    if value == 1.0 {
        return Some(ClosedInterval::point(0.0));
    }
    let (mantissa, exponent) = positive_binary64_parts(value)?;
    let m = ClosedInterval::point(mantissa);
    let z = m
        .sub(ClosedInterval::point(1.0))
        .div_positive(m.add(ClosedInterval::point(1.0)));
    Some(certified_log_from_atanh(z).add(certified_ln_two().scale(exponent as f64)))
}

/// Rigorous exact-real enclosure of `ln(1+value)`.
///
/// The nonnegative lane used by the affine score evaluates
/// `2·atanh(value/(2+value))` directly when `value <= 1`, preserving tiny
/// `value` without the rounded `1+value` cancellation. For larger values the
/// exact identity `ln(1+x) = ln(x) + ln(1+1/x)` keeps the atanh argument below
/// `1/3` and avoids overflow in `1+x`. Negative valid inputs route through the
/// certified positive logarithm of an outward `1+value` interval.
pub fn certified_ln_1p(value: f64) -> Option<ClosedInterval> {
    if !(value.is_finite() && value > -1.0) {
        return None;
    }
    if value == 0.0 {
        return Some(ClosedInterval::point(0.0));
    }
    if (0.0..=1.0).contains(&value) {
        let x = ClosedInterval::point(value);
        let z = x.div_positive(ClosedInterval::point(2.0).add(x));
        return Some(certified_log_from_atanh(z));
    }
    if value > 1.0 {
        let reciprocal = ClosedInterval::point(1.0)
            .div_positive(ClosedInterval::point(value))
            .nonnegative();
        let z = reciprocal
            .div_positive(ClosedInterval::point(2.0).add(reciprocal))
            .nonnegative();
        return Some(certified_ln_positive(value)?.add(certified_log_from_atanh(z)));
    }
    let argument = ClosedInterval::point(1.0).add(ClosedInterval::point(value));
    if !(argument.lo > 0.0) {
        return None;
    }
    let lo = certified_ln_positive(argument.lo)?;
    let hi = certified_ln_positive(argument.hi)?;
    Some(ClosedInterval::new(lo.lo, hi.hi))
}

fn exact_power_of_two(exponent: i32) -> Option<f64> {
    match exponent {
        -1074..=-1023 => {
            let bit = (exponent + 1074) as u32;
            Some(f64::from_bits(1_u64 << bit))
        }
        -1022..=1023 => Some(f64::from_bits(((exponent + 1023) as u64) << 52)),
        _ => None,
    }
}

/// Stable rounded representative of `numerator/(first*second)`.
///
/// Exact binary exponent extraction prevents the denominator product from
/// underflowing or overflowing before its scale cancels against the numerator.
/// Only two mantissa divisions and the final binary scaling round.
fn positive_ratio_over_product(
    numerator: f64,
    first_denominator: f64,
    second_denominator: f64,
) -> Option<f64> {
    if numerator == 0.0 {
        return Some(0.0);
    }
    let (numerator_mantissa, numerator_exponent) = positive_binary64_parts(numerator)?;
    let (first_mantissa, first_exponent) = positive_binary64_parts(first_denominator)?;
    let (second_mantissa, second_exponent) = positive_binary64_parts(second_denominator)?;
    let mut mantissa = numerator_mantissa / first_mantissa / second_mantissa;
    let mut exponent = numerator_exponent - first_exponent - second_exponent;
    if !(mantissa.is_finite() && mantissa > 0.0) {
        return None;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    if exponent < -1075 {
        return Some(0.0);
    }
    if exponent > 1023 {
        return None;
    }
    let value = if exponent == -1075 {
        (0.5 * mantissa) * exact_power_of_two(-1074)?
    } else {
        mantissa * exact_power_of_two(exponent)?
    };
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// Rigorous exact-real enclosure of `exp(value)` for a finite binary64 input.
///
/// Range reduction uses the independently certified `ln(2)` interval:
/// `value = k ln(2) + r`. After six exact halvings, `|r/64| < 1/16`; a fixed
/// Taylor polynomial encloses `exp(r/64)` and a geometric bound encloses its
/// positive tail. Six interval squarings and multiplication by the exact
/// binary power `2^k` restore the result. Subnormal outputs remain intervals
/// with an absolute (possibly zero) lower endpoint instead of being forced
/// through an invalid relative-error model.
pub fn certified_exp(value: f64) -> Option<ClosedInterval> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(ClosedInterval::point(1.0));
    }
    // This quotient merely chooses an integer identity; its accuracy is not a
    // proof premise because `r = value-k·ln(2)` is subsequently enclosed using
    // the certified ln(2) interval and validated below.
    let mut exponent = (value / std::f64::consts::LN_2).round() as i32;
    exponent = exponent.clamp(-1074, 1023);
    let remainder = ClosedInterval::point(value).sub(certified_ln_two().scale(exponent as f64));
    if !(remainder.is_valid() && remainder.max_abs() < 4.0) {
        return None;
    }
    let reduction = (1_u64 << EXP_RANGE_SQUARINGS) as f64;
    let reduced = remainder.scale(1.0 / reduction);
    if !(reduced.max_abs() < 1.0 / 16.0) {
        return None;
    }
    let mut term = ClosedInterval::point(1.0);
    let mut sum = term;
    for degree in 1..=EXP_SERIES_TERMS {
        term = term
            .mul(reduced)
            .div_positive(ClosedInterval::point(degree as f64));
        sum = sum.add(term);
    }
    let z = reduced.max_abs();
    let first_omitted = next_up(term.max_abs() * z / (EXP_SERIES_TERMS + 1) as f64);
    // Every later term ratio is at most z, so a geometric majorant is valid.
    let tail = next_up(first_omitted / next_down(1.0 - z));
    let mut result = sum.widen(tail);
    for _ in 0..EXP_RANGE_SQUARINGS {
        result = result.square();
    }
    result = result.mul(ClosedInterval::point(exact_power_of_two(exponent)?));
    Some(result.nonnegative())
}

#[inline]
fn certified_midpoint(interval: ClosedInterval) -> f64 {
    let midpoint = interval.lo + 0.5 * (interval.hi - interval.lo);
    midpoint.max(interval.lo).min(interval.hi)
}

/// Deterministic representative of [`certified_exp`].
///
/// This midpoint is for downstream floating-point evaluation only; callers
/// needing a proof must retain the full enclosure returned by
/// [`certified_exp`].
#[inline]
pub fn certified_exp_representative(value: f64) -> Option<f64> {
    certified_exp(value).map(certified_midpoint)
}

#[inline]
fn certified_ln_value(value: f64) -> Option<f64> {
    certified_ln_positive(value).map(certified_midpoint)
}

#[inline]
fn certified_ln_1p_value(value: f64) -> Option<f64> {
    certified_ln_1p(value).map(certified_midpoint)
}

fn interval_diameter(interval: ClosedInterval) -> f64 {
    if interval.lo == interval.hi {
        0.0
    } else {
        next_up(interval.hi - interval.lo)
    }
}

fn log_series_tail_max() -> f64 {
    let z = next_up(1.0 / 3.0);
    let z2 = next_up(z * z);
    let mut power = z;
    for _ in 1..LOG_SERIES_TERMS {
        power = next_up(power * z2);
    }
    power = next_up(power * z2);
    let denominator = next_down((2 * LOG_SERIES_TERMS + 1) as f64 * next_down(1.0 - z2));
    next_up(next_up(2.0 * power) / denominator)
}

/// Uniform absolute remainder of the reduced exponential Taylor series on
/// `[-1/16, 1/16]`, propagated through the six restoring squarings as a
/// relative error. This is computed only with outward basic arithmetic.
fn exp_series_relative_tail_max() -> f64 {
    let z = next_up(1.0 / 16.0);
    let mut term = 1.0;
    for degree in 1..=EXP_SERIES_TERMS {
        term = next_up(next_up(term * z) / degree as f64);
    }
    let first_omitted = next_up(next_up(term * z) / (EXP_SERIES_TERMS + 1) as f64);
    let absolute_tail = next_up(first_omitted / next_down(1.0 - z));
    // exp(reduced) >= exp(-1/16) > 1/2, hence its relative error is at most
    // twice the absolute Taylor tail. Raising the reduced result to 64 raises
    // the multiplicative error factor to the same power.
    let mut factor =
        ClosedInterval::point(1.0).add(ClosedInterval::point(next_up(2.0 * absolute_tail)));
    for _ in 0..EXP_RANGE_SQUARINGS {
        factor = factor.square();
    }
    next_up(factor.hi - 1.0).max(0.0)
}

/// Uniform forward-error bound for the midpoint returned by
/// [`certified_ln_value`] over a positive input interval.
fn certified_log_forward_error(input: ClosedInterval) -> f64 {
    if !(input.lo > 0.0 && input.hi.is_finite()) {
        return f64::INFINITY;
    }
    let exponent_abs = [input.lo, input.hi]
        .into_iter()
        .map(|value| {
            let bits = value.to_bits();
            let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
            if exponent_bits == 0 {
                let fraction = bits & ((1_u64 << 52) - 1);
                let highest = 63_i32 - fraction.leading_zeros() as i32;
                (highest - 1074).unsigned_abs() as f64
            } else {
                (exponent_bits - 1023).unsigned_abs() as f64
            }
        })
        .fold(0.0_f64, f64::max);
    let ln_two_uncertainty = next_up(exponent_abs * interval_diameter(certified_ln_two()));
    // Per term: power multiply, division, and accumulation, with two directed
    // endpoints; the remainder and range-combination path add 32 operations.
    let mantissa_ops = 6 * LOG_SERIES_TERMS + 32;
    let mantissa_error =
        add_nonnegative_upward(wilkinson_roundoff(1.0, mantissa_ops), log_series_tail_max());
    add_nonnegative_upward(ln_two_uncertainty, mantissa_error)
}

fn certified_log_error_from_output(output: ClosedInterval) -> f64 {
    if !output.is_valid() {
        return f64::INFINITY;
    }
    // |ln(input)|/ln(2) bounds the binary exponent to one neighboring bin.
    let exponent_abs = next_up(output.max_abs() / certified_ln_two().lo.abs()).ceil() + 1.0;
    let ln_two_uncertainty = next_up(exponent_abs * interval_diameter(certified_ln_two()));
    let mantissa_ops = 6 * LOG_SERIES_TERMS + 32;
    add_nonnegative_upward(
        ln_two_uncertainty,
        add_nonnegative_upward(wilkinson_roundoff(1.0, mantissa_ops), log_series_tail_max()),
    )
}

fn certified_ln1p_forward_error() -> f64 {
    let operations = 6 * LOG_SERIES_TERMS + 36;
    add_nonnegative_upward(wilkinson_roundoff(1.0, operations), log_series_tail_max())
}

/// Uniform absolute forward-error bound for [`certified_exp_representative`] on an
/// input interval, including range-reduction uncertainty and gradual
/// underflow.
fn certified_exp_forward_error(input: ClosedInterval, output: ClosedInterval) -> f64 {
    if !(input.is_valid() && output.is_valid() && output.lo >= 0.0) {
        return f64::INFINITY;
    }
    let exponent_abs = next_up(input.max_abs() / certified_ln_two().lo).ceil() + 1.0;
    let reduction_error = next_up(exponent_abs * interval_diameter(certified_ln_two()));
    if !(reduction_error < 1.0) {
        return f64::INFINITY;
    }
    // exp(delta)-1 <= delta/(1-delta) for 0 <= delta < 1.
    let propagated_reduction =
        next_up(output.max_abs() * reduction_error / next_down(1.0 - reduction_error));
    // Taylor recurrence, remainder, six squarings, and final binary scaling;
    // count both directed endpoints of each basic operation.
    let operations = 6 * EXP_SERIES_TERMS + 4 * EXP_RANGE_SQUARINGS + 40;
    let arithmetic = wilkinson_roundoff(output.max_abs(), operations);
    let truncation = next_up(output.max_abs() * exp_series_relative_tail_max());
    add_nonnegative_upward(
        propagated_reduction,
        add_nonnegative_upward(arithmetic, truncation),
    )
}

/// Uniform relative forward-error bound for
/// [`certified_exp_representative`] on an input interval whose exponential is
/// certified strictly positive.
///
/// The absolute bound above scales every multiplicative contribution by the
/// largest output in the interval. Dividing that result by the smallest output
/// couples opposite endpoints and can overflow on a wide interval even though
/// exp has a finite scale-independent relative error. Keep the range-reduction,
/// arithmetic, and truncation terms in relative currency instead. Only gradual
/// underflow is genuinely additive, so only that allowance is divided by the
/// certified positive lower output.
fn certified_exp_relative_forward_error(input: ClosedInterval, output: ClosedInterval) -> f64 {
    if !(input.is_valid() && output.is_valid() && output.lo > 0.0 && output.hi.is_finite()) {
        return f64::INFINITY;
    }
    let exponent_abs = next_up(input.max_abs() / certified_ln_two().lo).ceil() + 1.0;
    let reduction_error = next_up(exponent_abs * interval_diameter(certified_ln_two()));
    if !(reduction_error < 1.0) {
        return f64::INFINITY;
    }
    let relative_reduction = next_up(reduction_error / next_down(1.0 - reduction_error));
    let operations = 6 * EXP_SERIES_TERMS + 4 * EXP_RANGE_SQUARINGS + 40;
    let relative_arithmetic = wilkinson_roundoff(1.0, operations);
    let relative_underflow = next_up(wilkinson_roundoff(0.0, operations) / output.lo);
    add_nonnegative_upward(
        relative_reduction,
        add_nonnegative_upward(
            relative_arithmetic,
            add_nonnegative_upward(exp_series_relative_tail_max(), relative_underflow),
        ),
    )
}

/// Upward-rounded accumulation of a nonnegative magnitude bound.
fn add_nonnegative_upward(accumulator: f64, term: f64) -> f64 {
    if accumulator == f64::INFINITY || term == f64::INFINITY {
        f64::INFINITY
    } else if term == 0.0 {
        accumulator
    } else {
        next_up(accumulator + term)
    }
}

/// Symmetric absolute radius needed to widen `mathematical` until it contains
/// the already-computed `resolved` interval.
fn enclosure_excess(mathematical: ClosedInterval, resolved: ClosedInterval) -> f64 {
    let lower = if mathematical.lo == resolved.lo {
        0.0
    } else {
        next_up(mathematical.lo - resolved.lo)
    };
    let upper = if mathematical.hi == resolved.hi {
        0.0
    } else {
        next_up(resolved.hi - mathematical.hi)
    };
    lower.max(upper).max(0.0)
}

/// Wilkinson forward-error bound for `k` round-to-nearest binary64
/// operations. The normal-range `gamma_k * magnitude` term is accompanied by
/// `k` minimum-subnormal units, covering gradual-underflow roundoff where a
/// purely relative model is invalid.
fn wilkinson_roundoff(magnitude: f64, operations: usize) -> f64 {
    if operations == 0 {
        return 0.0;
    }
    if !(magnitude.is_finite() && magnitude >= 0.0) {
        return f64::INFINITY;
    }
    // Convert the integer count upward before either product. For counts above
    // 2^53, `as f64` can round down; charging only one ulp after multiplication
    // would then combine two rounding steps into an unjustified one-step
    // bound.
    let operation_count = next_up(operations as f64);
    let underflow = next_up(operation_count * f64::from_bits(1));
    if magnitude == 0.0 {
        return underflow;
    }
    // IEEE-754 binary64 unit roundoff under round-to-nearest.
    let unit_roundoff = 0.5 * f64::EPSILON;
    let ku = next_up(operation_count * unit_roundoff);
    if !(ku < 1.0) {
        return f64::INFINITY;
    }
    let denominator = next_down(1.0 - ku);
    if !(denominator > 0.0) {
        return f64::INFINITY;
    }
    let gamma = next_up(ku / denominator);
    add_nonnegative_upward(next_up(gamma * magnitude), underflow)
}

#[inline]
fn sum_down(left: f64, right: f64) -> f64 {
    let value = left + right;
    if sum_is_exact(left, right, value) {
        value
    } else {
        next_down(value)
    }
}

#[inline]
fn sum_up(left: f64, right: f64) -> f64 {
    let value = left + right;
    if sum_is_exact(left, right, value) {
        value
    } else {
        next_up(value)
    }
}

/// Whether binary64 addition produced the exact-real sum.
///
/// Knuth's `TwoSum` residual is itself exact under IEEE round-to-nearest with
/// gradual underflow. Besides avoiding needless interval inflation, retaining
/// exact cancellation is semantically important: structural zeros in diffuse
/// covariance recurrences must remain `[0, 0]`, not become artificial
/// minimum-subnormal uncertainty.
#[inline]
fn sum_is_exact(left: f64, right: f64, value: f64) -> bool {
    if left == 0.0 || right == 0.0 {
        return true;
    }
    if !(left.is_finite() && right.is_finite() && value.is_finite()) {
        return value == left || value == right;
    }
    let virtual_right = value - left;
    let virtual_left = value - virtual_right;
    let right_residual = right - virtual_right;
    let left_residual = left - virtual_left;
    left_residual + right_residual == 0.0
}

#[inline]
fn product_is_exact(left: f64, right: f64) -> bool {
    left == 0.0 || right == 0.0 || left.abs() == 1.0 || right.abs() == 1.0
}

#[inline]
fn product_down(left: f64, right: f64) -> f64 {
    let value = left * right;
    if product_is_exact(left, right) {
        if value.is_nan() { 0.0 } else { value }
    } else {
        next_down(value)
    }
}

#[inline]
fn product_up(left: f64, right: f64) -> f64 {
    let value = left * right;
    if product_is_exact(left, right) {
        if value.is_nan() { 0.0 } else { value }
    } else {
        next_up(value)
    }
}

#[inline]
fn quotient_down(numerator: f64, denominator: f64) -> f64 {
    let value = numerator / denominator;
    if numerator == 0.0 || denominator.abs() == 1.0 {
        value
    } else {
        next_down(value)
    }
}

#[inline]
fn quotient_up(numerator: f64, denominator: f64) -> f64 {
    let value = numerator / denominator;
    if numerator == 0.0 || denominator.abs() == 1.0 {
        value
    } else {
        next_up(value)
    }
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
            third: 2000.0 * (3.0 * dp * ddp + p * 6.0),
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
        let value = x.add(p.square().scale(1000.0));
        DerivativeEnclosure {
            score: ScoreValueEnclosure {
                value,
                evaluation_error: wilkinson_roundoff(value.max_abs(), 7),
            },
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
            |lo, hi| -> Result<_, String> { Ok(polynomial_hidden_bump_enclosure(lo.x, hi.x)) },
        )
        .expect("certified search");

        // At x=0, 1/2, 1 both value and derivative agree exactly with f=x;
        // the former midpoint/Hermite heuristic therefore returned x=1.
        assert_eq!(polynomial_hidden_bump_jet(0.0).derivative, 1.0);
        assert_eq!(polynomial_hidden_bump_jet(0.5).derivative, 1.0);
        assert_eq!(polynomial_hidden_bump_jet(1.0).derivative, 1.0);
        assert!(result.optimum.x > 0.5 && result.optimum.x < 1.0);
        assert!(result.optimum.value > 2.9);
        assert!(
            result
                .stationary_points
                .iter()
                .any(|point| point.bracket.contains(result.optimum.x)),
            "the hidden global maximizer must have a retained root certificate"
        );
        assert!(
            result
                .dominated_regions
                .iter()
                .all(|region| region.score.value.hi < region.incumbent_lower),
            "every skipped stationary branch must carry a strict exact dominance proof"
        );
    }

    fn quartic_jet(x: f64) -> ScoreJet {
        ScoreJet {
            value: -(x * x - 1.0).powi(2),
            derivative: 4.0 * x - 4.0 * x * x * x,
            curvature: 4.0 - 12.0 * x * x,
            third: -24.0 * x,
        }
    }

    fn quartic_enclosure(lo: f64, hi: f64) -> DerivativeEnclosure {
        let x = ClosedInterval::new(lo, hi);
        let shifted_square = x.square().sub(ClosedInterval::point(1.0));
        let value = shifted_square.square().neg();
        if lo == hi && (lo == -1.0 || lo == 0.0 || lo == 1.0) {
            return DerivativeEnclosure {
                score: ScoreValueEnclosure {
                    value,
                    evaluation_error: wilkinson_roundoff(value.max_abs(), 4),
                },
                derivative: ClosedInterval::point(0.0),
                curvature: ClosedInterval::point(quartic_jet(lo).curvature),
            };
        }
        DerivativeEnclosure {
            score: ScoreValueEnclosure {
                value,
                evaluation_error: wilkinson_roundoff(value.max_abs(), 4),
            },
            derivative: x.scale(4.0).sub(x.mul(x).mul(x).scale(4.0)),
            curvature: ClosedInterval::point(4.0).sub(x.square().scale(12.0)),
        }
    }

    #[test]
    fn globally_relevant_roots_are_isolated_and_dominated_structure_is_audited() {
        let result = maximize_score_1d(
            -2.0,
            2.0,
            1.0e-10,
            |x| -> Result<_, String> { Ok(quartic_jet(x)) },
            |lo, hi| -> Result<_, String> { Ok(quartic_enclosure(lo.x, hi.x)) },
        )
        .expect("certified search");
        assert_eq!(
            result.stationary_points.len(),
            2,
            "both equal global maxima must survive strict dominance"
        );
        for expected in [-1.0_f64, 1.0] {
            let point = result
                .stationary_points
                .iter()
                .find(|point| (point.sample.x - expected).abs() <= 1.0e-9)
                .unwrap_or_else(|| panic!("missing global maximum at {expected}"));
            assert!(point.bracket.hi - point.bracket.lo <= 1.0e-10);
        }
        assert!(
            result
                .dominated_regions
                .iter()
                .any(|region| region.bracket.contains(0.0)),
            "the strictly inferior stationary minimum must remain auditable as dominated"
        );
        assert!((result.optimum.x.abs() - 1.0).abs() <= 1.0e-9);
    }

    #[test]
    fn exact_dominance_prunes_an_uninformative_saturated_tail() {
        let mut evaluations = 0_usize;
        let result = maximize_score_1d(
            -1.0,
            10.0,
            1.0e-9,
            |x| -> Result<_, String> {
                evaluations += 1;
                Ok(ScoreJet {
                    value: 1.0 - x * x,
                    derivative: -2.0 * x,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let value = ClosedInterval::point(1.0).sub(x.square());
                let root_side_cell = right.x <= 1.0;
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        evaluation_error: 1.0e-12,
                    },
                    derivative: if root_side_cell || left.x == right.x {
                        x.scale(-2.0)
                    } else {
                        // This deliberately dependency-heavy extension carries
                        // no stationary information in the low-score tail.
                        ClosedInterval::new(-100.0, 100.0)
                    },
                    curvature: if root_side_cell || left.x == right.x {
                        ClosedInterval::point(-2.0)
                    } else {
                        ClosedInterval::new(-100.0, 100.0)
                    },
                })
            },
        )
        .expect("the exact score incumbent must dominate the uninformative tail");

        assert_eq!(result.optimum.x, 0.0);
        assert!(result.value_certificate.maximum.contains(1.0));
        assert!(
            !result.dominated_regions.is_empty(),
            "the fixture's saturated tail must be terminated by exact dominance"
        );
        assert!(
            result
                .dominated_regions
                .iter()
                .all(|region| region.score.value.hi < region.incumbent_lower),
            "every retained dominance decision must expose its strict exact ordering"
        );
        assert!(
            evaluations < 16,
            "the low-score tail was enumerated instead of pruned ({evaluations} evaluations)"
        );
    }

    /// The abscissa at which this fixture's point oracle reports a derivative
    /// of exactly zero while the exact derivative is two.
    const ROUNDED_ZERO_ABSCISSA: f64 = 1.5;

    /// A point derivative that rounds to zero cannot close its parent cell.
    ///
    /// The exact score is the concave quadratic `1 - (x - 2.5)^2` on `[0, 3]`.
    /// Its only stationary point and maximum is exactly representable at
    /// `x=2.5`.
    /// The point oracle deliberately loses the nonzero derivative at the
    /// safeguarded midpoint `x=1.5`, while the exact-real enclosure reports the
    /// true derivative range through interval arithmetic. The initial Newton
    /// proposal at `x=3` is `2.5`, outside the central-half guard, so the
    /// midpoint is exercised deterministically. Treating its rounded scalar
    /// zero as a root closes the left half, discards the real maximum, and
    /// leaves the rounded-value selection at boundary `x=3`, whose score is
    /// `0.75`. The point enclosure introduced by the exact-real repair
    /// distinguishes that false zero from the quadratic's exact zero at
    /// `x=2.5`.
    #[test]
    fn a_rounded_zero_at_a_cell_endpoint_does_not_close_the_cell() {
        let mut rounded_zeros = 0_usize;
        let result = maximize_score_1d(
            0.0,
            3.0,
            1.0e-9,
            |x| -> Result<_, String> {
                let shifted = x - 2.5;
                let derivative = if x == ROUNDED_ZERO_ABSCISSA {
                    rounded_zeros += 1;
                    0.0
                } else {
                    -2.0 * shifted
                };
                Ok(ScoreJet {
                    value: 1.0 - shifted * shifted,
                    derivative,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let shifted = x.sub(ClosedInterval::point(2.5));
                let value = ClosedInterval::point(1.0).sub(shifted.square());
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        evaluation_error: wilkinson_roundoff(value.max_abs(), 3),
                    },
                    // Preserve the quadratic's structural zero at x=2.5 instead
                    // of manufacturing cancellation through `5 - 2x`.
                    derivative: shifted.scale(-2.0),
                    curvature: ClosedInterval::point(-2.0),
                })
            },
        )
        .expect("certified search");

        assert!(
            rounded_zeros > 0,
            "fixture premise unmet: the search never evaluated x = {ROUNDED_ZERO_ABSCISSA}"
        );
        assert!(
            (result.optimum.x - 2.5).abs() <= 1.0e-9,
            "reported the maximum at x={} (value {}) instead of x=2.5",
            result.optimum.x,
            result.optimum.value,
        );
        assert!(
            result.value_certificate.maximum.contains(1.0),
            "the exact maximum escaped the global score certificate: {:?}",
            result.value_certificate,
        );
        assert!(
            result
                .stationary_points
                .iter()
                .all(|point| point.sample.x != ROUNDED_ZERO_ABSCISSA),
            "a derivative that rounded to zero was reported as a stationary point",
        );
        let root = result
            .stationary_points
            .iter()
            .find(|point| point.bracket.contains(2.5))
            .expect("the exact quadratic root must be isolated");
        assert_eq!(
            root.bracket,
            ClosedInterval::point(2.5),
            "the cancellation-free point enclosure must preserve the exact dyadic root"
        );
    }

    #[test]
    fn adjacent_cell_evidence_is_retained_when_point_derivative_is_uninformative() {
        let planted = 0.7_f64;
        let result = maximize_score_1d(
            0.0,
            1.0,
            1.0e-9,
            |x| -> Result<_, String> {
                let shifted = x - planted;
                Ok(ScoreJet {
                    value: 1.0 - shifted * shifted,
                    derivative: -2.0 * shifted,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let shifted = x.sub(ClosedInterval::point(planted));
                let value = ClosedInterval::point(1.0).sub(shifted.square());
                let interior_point = left.x == right.x && left.x > 0.0 && left.x < 1.0;
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        evaluation_error: wilkinson_roundoff(value.max_abs(), 3),
                    },
                    // Model a cancellation-heavy degenerate-cell formula: by
                    // itself it carries no sign. Each adjacent nondegenerate
                    // interval remains a tight exact extension, and their
                    // intersection at the shared endpoint isolates the root.
                    derivative: if interior_point {
                        ClosedInterval::new(-2.0, 2.0)
                    } else {
                        shifted.scale(-2.0)
                    },
                    curvature: ClosedInterval::point(-2.0),
                })
            },
        )
        .expect("adjacent exact cell evidence must isolate the unique root");

        assert!(
            (result.optimum.x - planted).abs() <= 1.0e-9,
            "selected {}, expected {planted}",
            result.optimum.x
        );
        let stationary = result
            .stationary_points
            .iter()
            .find(|point| point.bracket.contains(planted))
            .expect("the planted stationary point must be certified");
        assert!(stationary.bracket.hi - stationary.bracket.lo <= 1.0e-9);
    }

    #[test]
    fn signed_endpoint_newton_reaches_the_existing_score_resolution_floor() {
        let planted = 0.8_f64;
        let ambiguous_probe = 0.5_f64;
        let mut ambiguous_probe_calls = 0_usize;
        let result = maximize_score_1d(
            0.0,
            1.0,
            1.0e-9,
            |x| -> Result<_, String> {
                let shifted = x - planted;
                Ok(ScoreJet {
                    value: 1.0 - shifted * shifted,
                    derivative: -2.0 * shifted,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let shifted = x.sub(ClosedInterval::point(planted));
                let value = ClosedInterval::point(1.0).sub(shifted.square());
                let derivative = if left.x == right.x {
                    if left.x == ambiguous_probe {
                        ambiguous_probe_calls += 1;
                        ClosedInterval::new(-2.0, 2.0)
                    } else {
                        ClosedInterval::point(-2.0 * (left.x - planted))
                    }
                } else {
                    // A deliberately dependency-heavy cell extension. It is
                    // valid, but neither it nor the ambiguous point image can
                    // contract the first midpoint probe.
                    ClosedInterval::new(-2.0, 2.0)
                };
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        // The exact score motion on the first endpoint-Newton
                        // image is 0.04. A valid 0.021 point forward-error bound
                        // makes that information floor 0.042, while the initial
                        // domain's 0.64 score motion remains visibly nonflat.
                        evaluation_error: 0.021,
                    },
                    derivative,
                    // Strictly oriented but deliberately wider than the exact
                    // constant curvature -2.
                    curvature: ClosedInterval::new(-4.0, -1.0),
                })
            },
        )
        .expect("signed endpoint Newton images must reach a typed score-resolution proof");

        assert!(
            ambiguous_probe_calls > 0,
            "fixture premise unmet: the cancellation-heavy midpoint was never certified"
        );
        let ScoreOptimumLocation::ResolutionFlat(index) = result.location else {
            panic!(
                "the unique root's location is below the declared information floor: {:?}",
                result.location
            );
        };
        let flat = result.resolution_flat_regions[index];
        assert!(
            flat.bracket.contains(planted),
            "contracted flat bracket {:?} lost the unique root",
            flat.bracket
        );
        assert!(
            flat.max_score_gap <= flat.score_resolution,
            "typed flat proof exceeded its existing evaluator floor: {flat:?}"
        );
        assert!(result.stationary_points.is_empty());
    }

    #[test]
    fn strict_concavity_certifies_the_quintic_scan_optimum_at_score_resolution() {
        // The terminal certificate from #2790, seed 0 at n=100. Its score
        // range is deliberately wider than pairwise evaluator error, so a
        // value-flat test cannot close this cell. Strict concavity says much
        // more: despite cancellation in the derivative enclosure, the maximum
        // can improve on the represented point by only g^2/(2 mu).
        let left = SearchSample {
            sample: ScoreSample {
                x: -12.105_374_438_144_967,
                value: 134.053_351_995_058_96,
                derivative: 1.0e-3,
                curvature: -1.0,
                third: 0.0,
            },
            point_enclosure: None,
        };
        let right = SearchSample {
            sample: ScoreSample {
                x: -12.104_760_848_454_575,
                value: 134.054_279_259_553_65,
                derivative: -1.0e-3,
                curvature: -1.0,
                third: 0.0,
            },
            point_enclosure: None,
        };
        let sample = ScoreSample {
            x: left.sample.x + 0.5 * (right.sample.x - left.sample.x),
            value: 134.053_9,
            derivative: 0.0,
            curvature: -1.0,
            third: 0.0,
        };
        let evaluation_error = 3.966_754_013_333_685e-4;
        let point_score = ScoreValueEnclosure {
            value: ClosedInterval::point(sample.value),
            evaluation_error,
        };
        let enclosure = DerivativeEnclosure {
            score: ScoreValueEnclosure {
                value: ClosedInterval::new(134.053_351_995_058_96, 134.054_279_259_553_65),
                evaluation_error,
            },
            derivative: ClosedInterval::new(-1.8562e-3, 1.6607e-3),
            curvature: ClosedInterval::new(-2.2666, -0.2358),
        };

        assert!(
            resolution_flat_region(SearchNode { left, right }, enclosure).is_none(),
            "fixture premise: the full score diameter exceeds pairwise evaluation error"
        );
        let (flat, maximum) = score_resolved_concave_maximum(
            SearchNode { left, right },
            enclosure,
            sample,
            enclosure.derivative,
            enclosure.curvature,
            point_score,
        )
        .expect("strict concavity must close the already score-resolved optimum");
        assert!(flat.bracket.contains(sample.x));
        assert!(flat.max_score_gap < 7.4e-6);
        assert!(flat.max_score_gap <= flat.score_resolution);
        assert!(
            maximum.value.hi < enclosure.score.value.hi,
            "the strong-concavity maximum bound must remove the loose cell-wide score tail"
        );
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
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let value = ClosedInterval::new(left.x, right.x).scale(0.3);
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        evaluation_error: wilkinson_roundoff(value.max_abs(), 1),
                    },
                    derivative: ClosedInterval::point(0.3),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("certified search");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 9.0);
        assert!(result.stationary_points.is_empty());
        assert_eq!(
            result.value_certificate.maximum_excess, 0.0,
            "the exact same terminal point is not a competing uncertain value"
        );
    }

    #[test]
    fn certified_increase_selects_upper_boundary_when_rounded_values_tie() {
        let result = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-9,
            |_| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 0.0,
                    derivative: 1.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(left.x, right.x),
                        evaluation_error: 1.0,
                    },
                    derivative: ClosedInterval::point(1.0),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("a whole-domain positive derivative orders tied rounded endpoints");
        assert_eq!(result.lower_boundary.value, result.upper_boundary.value);
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 1.0);
        assert_eq!(result.value_certificate.maximum_excess, 0.0);
    }

    #[test]
    fn certified_decrease_selects_lower_boundary_when_rounded_values_tie() {
        let result = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-9,
            |_| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 0.0,
                    derivative: -1.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(-right.x, -left.x),
                        evaluation_error: 1.0,
                    },
                    derivative: ClosedInterval::point(-1.0),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("a whole-domain negative derivative orders tied rounded endpoints");
        assert_eq!(result.lower_boundary.value, result.upper_boundary.value);
        assert_eq!(result.location, ScoreOptimumLocation::LowerBoundary);
        assert_eq!(result.optimum.x, -1.0);
        assert_eq!(result.value_certificate.maximum_excess, 0.0);
    }

    #[test]
    fn tangential_nonmaximum_structure_is_closed_by_exact_dominance() {
        let result = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-8,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x * x * x,
                    derivative: 3.0 * x * x,
                    curvature: 6.0 * x,
                    third: 6.0,
                })
            },
            |lo, hi| -> Result<_, String> {
                let x = ClosedInterval::new(lo.x, hi.x);
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: x.mul(x).mul(x),
                        evaluation_error: f64::EPSILON,
                    },
                    derivative: x.square().scale(3.0),
                    curvature: x.scale(6.0),
                })
            },
        )
        .expect("the inferior inflection is immaterial by exact score ordering");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert!(
            !result.dominated_regions.is_empty(),
            "the search must record the exact dominance proof instead of silently dropping the cell"
        );
        for region in result.dominated_regions {
            assert!(region.score.value.hi < region.incumbent_lower);
        }
    }

    #[test]
    fn unresolved_nonflat_cell_remains_typed() {
        let error = maximize_score_1d(
            0.0,
            1.0e-8,
            1.0e-8,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x,
                    derivative: 0.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |lo, hi| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(lo.x, hi.x),
                        evaluation_error: 0.0,
                    },
                    derivative: ClosedInterval::new(-1.0, 1.0),
                    curvature: ClosedInterval::new(-1.0, 1.0),
                })
            },
        )
        .expect_err("a derivative enclosure admitting visible score motion is not flat");
        assert!(matches!(error, ScoreSearchError::Unresolved { .. }));
    }

    /// BREADTH exhaustion, which is a different failure from the per-cell depth
    /// floor and was #2546's non-termination.
    ///
    /// The oracle's derivative and curvature enclosures always straddle zero, so
    /// no cell is ever excluded by a sign or isolated as a root; but its score
    /// range collapses with the cell against a FIXED evaluation error, so every
    /// cell does terminate — as resolution-flat — once it is narrower than
    /// `2 * evaluation_error`. That is the regime the cascade is in: cells
    /// certify, at widths far above `resolution`, and the traversal simply needs
    /// too many of them. The flat width here is 1e-3 of a 32-wide domain, so the
    /// decomposition is ~2^15 = 32 768 cells and no cell ever reaches the
    /// resolution floor — `ScoreSearchError::Unresolved` cannot fire, and
    /// without a breadth budget nothing else can either.
    ///
    /// Contrast `unresolved_nonflat_cell_remains_typed`, whose oracle certifies
    /// NOTHING at any width: that one bottoms out on the depth floor after `D`
    /// subdivisions and is already typed. The two are not interchangeable.
    #[test]
    fn undecomposable_criterion_exhausts_the_budget_instead_of_enumerating_the_domain() {
        let lo = 0.0;
        let hi = 32.0;
        let resolution = f64::EPSILON.sqrt();
        let flat_error = 5.0e-4;
        let (budget, depth_bound) = subdivision_budget(lo, hi, resolution);
        assert_eq!(depth_bound, 31, "log2(32 / sqrt(eps)) rounds up to 31");
        // Pins the shipped coefficient in `subdivision_budget`, whose doc
        // explains why it is 8 and why raising it further only converts a
        // budget refusal into a resolution refusal (#2614). This assertion is
        // deliberately a change-detector: if the constant moves, update BOTH,
        // and read that doc before deciding the move is a fix.
        assert_eq!(
            budget,
            8 * 31 * 31,
            "budget must track the 8 D^2 coefficient in subdivision_budget"
        );
        let error = maximize_score_1d(
            lo,
            hi,
            resolution,
            |_| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 0.0,
                    derivative: 0.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let half_width = 0.5 * (right.x - left.x);
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(-half_width, half_width),
                        evaluation_error: flat_error,
                    },
                    derivative: ClosedInterval::new(-1.0, 1.0),
                    curvature: ClosedInterval::new(-1.0, 1.0),
                })
            },
        )
        .expect_err("a decomposition this large must refuse, not enumerate");
        let ScoreSearchError::SubdivisionBudget {
            subdivisions,
            budget: reported_budget,
            depth_bound: reported_depth,
            cell_lo,
            cell_hi,
            ..
        } = error
        else {
            panic!("expected a subdivision-budget refusal, got {error}");
        };
        assert_eq!(
            subdivisions,
            budget + 1,
            "the budget stops the split that exceeds it"
        );
        assert_eq!(reported_budget, budget);
        assert_eq!(reported_depth, depth_bound);
        assert!(
            cell_hi - cell_lo > 2.0 * flat_error,
            "the reported cell must be one the search could still have split and \
             had not yet certified ({cell_lo}, {cell_hi}); a narrower cell would \
             mean the depth floor, not the breadth budget, was binding"
        );
    }

    /// The same budget must be invisible to a search that converges. A strictly
    /// concave criterion over the same wide domain isolates its stationary point
    /// in subdivisions proportional to the DEPTH, so the number of criterion
    /// evaluations stays far below a budget scaled by the depth SQUARED.
    #[test]
    fn a_converging_search_stays_far_under_the_subdivision_budget() {
        let lo = 0.0;
        let hi = 32.0;
        let resolution = f64::EPSILON.sqrt();
        let (budget, depth_bound) = subdivision_budget(lo, hi, resolution);
        let evaluations = std::cell::Cell::new(0usize);
        let result = maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| -> Result<_, String> {
                evaluations.set(evaluations.get() + 1);
                let shifted = x - 7.0;
                Ok(ScoreJet {
                    value: -shifted * shifted,
                    derivative: -2.0 * shifted,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let shifted = x.sub(ClosedInterval::point(7.0));
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: shifted.square().scale(-1.0),
                        evaluation_error: f64::EPSILON * 1024.0,
                    },
                    derivative: shifted.scale(-2.0),
                    curvature: ClosedInterval::point(-2.0),
                })
            },
        )
        .expect("a strictly concave criterion is decomposable");
        let ScoreOptimumLocation::Stationary(index) = result.location else {
            panic!("expected the interior maximum, got {:?}", result.location);
        };
        let bracket = result.stationary_points[index].bracket;
        assert!(
            bracket.lo <= 7.0 && bracket.hi >= 7.0,
            "certified bracket {bracket:?} must contain the planted maximum"
        );
        // Every subdivision costs one midpoint evaluation, so the evaluation
        // count bounds the subdivisions from above.
        assert!(
            evaluations.get() < budget / 8,
            "a converging search used {} evaluations against budget {budget} at depth \
             bound {depth_bound}; a budget within 8x of a converging search is a \
             tuning parameter, not a backstop",
            evaluations.get()
        );
    }

    #[test]
    fn resolution_flatness_is_exactly_value_diameter_vs_pairwise_error() {
        let sample = SearchSample {
            sample: ScoreSample {
                x: 0.0,
                value: 7.0,
                derivative: 0.0,
                curvature: 0.0,
                third: 0.0,
            },
            point_enclosure: None,
        };
        let node = SearchNode {
            left: sample,
            right: SearchSample {
                sample: ScoreSample {
                    x: 1.0,
                    ..sample.sample
                },
                point_enclosure: None,
            },
        };
        let error = 0.125;
        for (upper, expected) in [(1024.25, true), (next_up(1024.25), false)] {
            let enclosure = DerivativeEnclosure {
                score: ScoreValueEnclosure {
                    // Translation by a large exactly represented constant must
                    // not change either side of the flatness comparison.
                    value: ClosedInterval::new(1024.0, upper),
                    evaluation_error: error,
                },
                derivative: ClosedInterval::new(-1.0, 1.0),
                curvature: ClosedInterval::new(-1.0, 1.0),
            };
            assert_eq!(
                resolution_flat_region(node, enclosure).is_some(),
                expected,
                "flatness must be equivalent to outward diameter <= outward 2*value error"
            );
        }
    }

    #[test]
    fn resolution_flat_cells_remain_regions_instead_of_fake_points() {
        let resolution = 0.25;
        let result = maximize_score_1d(
            0.0,
            1.0,
            resolution,
            |_| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 3.0,
                    derivative: 0.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |_, _| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::point(3.0),
                        evaluation_error: 0.0,
                    },
                    derivative: ClosedInterval::new(-1.0, 1.0),
                    curvature: ClosedInterval::new(-1.0, 1.0),
                })
            },
        )
        .expect("an exactly constant score is resolution-flat");
        assert_eq!(result.resolution_flat_regions.len(), 1);
        assert!(
            result.resolution_flat_regions[0].bracket.hi
                - result.resolution_flat_regions[0].bracket.lo
                > resolution,
            "value resolution may close a wide cell, so callers must not reinterpret it \
             as an abscissa-resolved stationary point"
        );
    }

    #[test]
    fn directed_arithmetic_preserves_cancellation_and_subnormal_error() {
        assert_eq!(
            ClosedInterval::point(1.0).sub(ClosedInterval::point(1.0)),
            ClosedInterval::point(0.0),
            "an exact structural zero must not acquire artificial uncertainty"
        );
        let minimum_subnormal = f64::from_bits(1);
        let underflowing_product =
            ClosedInterval::point(minimum_subnormal).mul(ClosedInterval::point(0.5));
        assert!(
            underflowing_product.lo <= 0.5 * minimum_subnormal
                && underflowing_product.hi >= 0.5 * minimum_subnormal
                && underflowing_product.lo < 0.0
                && underflowing_product.hi > 0.0,
            "a nonzero exact product that rounds to zero needs additive subnormal width"
        );
        assert!(
            wilkinson_roundoff(0.0, 1) >= minimum_subnormal,
            "a zero-magnitude relative model must still charge additive underflow"
        );
    }

    #[test]
    fn certified_elementary_intervals_cover_normal_and_subnormal_lanes() {
        for value in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            0.5,
            1.0,
            2.0,
            f64::MAX,
        ] {
            let enclosure = certified_ln_positive(value).expect("certified positive log");
            assert!(enclosure.is_valid() && enclosure.lo.is_finite() && enclosure.hi.is_finite());
            assert!(
                enclosure.contains(value.ln()),
                "independent platform log sanity value {} escaped {:?}",
                value.ln(),
                enclosure
            );
        }
        for value in [-744.0_f64, -708.0, -1.0, 0.0, 1.0, 709.0] {
            let enclosure = certified_exp(value).expect("certified exponential");
            assert!(enclosure.is_valid() && enclosure.lo >= 0.0);
            assert!(
                enclosure.contains(value.exp()),
                "independent platform exp sanity value {} escaped {:?}",
                value.exp(),
                enclosure
            );
        }
        for value in [f64::from_bits(1), 1.0e-12, 0.25, 1.0] {
            let enclosure = certified_ln_1p(value).expect("certified log1p");
            assert!(
                enclosure.contains(value.ln_1p()),
                "independent platform log1p sanity value {} escaped {:?}",
                value.ln_1p(),
                enclosure
            );
        }
    }

    #[test]
    fn exact_range_is_not_compared_to_a_separately_rounded_curvature() {
        let denormal = f64::from_bits(1);
        let result = maximize_score_1d(
            0.0,
            1.0,
            1.0e-8,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x,
                    derivative: 1.0,
                    // A real negative denormal rounds to signed zero in the
                    // point-jet arithmetic represented by this fixture.
                    curvature: -0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(left.x, right.x),
                        evaluation_error: 0.0,
                    },
                    derivative: ClosedInterval::point(1.0),
                    curvature: ClosedInterval::point(-denormal),
                })
            },
        )
        .expect("an exact-real enclosure need not contain a separately rounded scalar jet");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
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
        let score = enclosure.score;
        let resolved_score = score.value.widen(score.evaluation_error);
        for x in [-2.5_f64, -1.7, -0.3, 0.0, 0.9, 1.75] {
            let jet = profile.evaluate(x).unwrap();
            let point = profile.enclose(x, x).expect("point enclosure");
            assert!(
                resolved_score.contains(jet.value),
                "score {} at {x} outside {:?} ± {}",
                jet.value,
                score.value,
                score.evaluation_error
            );
            assert!(
                enclosure
                    .derivative
                    .intersection(point.derivative)
                    .is_some(),
                "exact point gradient {:?} at {x} is disjoint from {:?}",
                point.derivative,
                enclosure.derivative
            );
            assert!(
                enclosure.curvature.intersection(point.curvature).is_some(),
                "exact point curvature {:?} at {x} is disjoint from {:?}",
                point.curvature,
                enclosure.curvature
            );
        }
    }

    #[test]
    fn affine_reml_zero_smoothing_complement_retains_residual_correlation() {
        // With E = sum(q/g), the zero-smoothing residual is exactly zero and
        //
        //   R(lambda) = sum_i (q_i/g_i) * lambda/(1 + lambda).
        //
        // The determinant and profiled-residual derivatives then cancel
        // identically when residual_dof equals the mode count, so the exact
        // score derivative is zero. Forming R as
        // `E - sum q/(g + lambda*s)` loses the shared near-one factor once per
        // mode: at lambda=1e-10 its interval width is large enough to fabricate
        // a material derivative range even though every term has the same
        // analytic complement. The zero-smoothing form carries that
        // correlation explicitly.
        const MODES: usize = 64;
        let grams = [1.0; MODES];
        let penalties = [1.0; MODES];
        let projected = [1.0; MODES];
        let energies = [MODES as f64];
        let profile = AffineRemlProfile::new(
            &grams,
            &penalties,
            &projected,
            &energies,
            MODES as f64,
            MODES,
            0.0,
        )
        .expect("valid cancellation fixture");
        let rho = -23.025850929940457_f64; // nearest binary64 to ln(1e-10)
        let enclosure = profile
            .enclose(rho, rho)
            .expect("equivalent residual forms must retain their intersection");

        assert!(
            enclosure.derivative.contains_zero(),
            "the analytically constant profile must contain zero derivative: {:?}",
            enclosure.derivative
        );
        assert!(
            enclosure.derivative.hi - enclosure.derivative.lo < 1.0e-6,
            "the residual complement must remove the independent near-one dependency: {:?}",
            enclosure.derivative
        );
    }

    /// The score VALUE enclosure may never be looser than the bound its own
    /// DERIVATIVE enclosure certifies for the same cell.
    ///
    /// Both are enclosures of one function, so the mean value theorem ties
    /// them: across a cell of width `w` the score cannot move by more than
    /// `max|f'| * w`, hence
    ///
    /// ```text
    ///     width(F([a,b]))  <=  width(F({m}))  +  max|F'([a,b])| * w
    /// ```
    ///
    /// This is the invariant the natural interval extension broke, and it broke
    /// it in the regime the search lives in. The score is
    /// `-0.5 * (D*logdet_block + dof*deviance_block)` and near a REML optimum
    /// those two blocks cancel — each moves by `O(rank)` per unit of `rho`
    /// while their sum does not. Interval addition cannot see that the two
    /// movements are the same quantity with opposite signs, so the natural
    /// extension returned a range of width `rank * w` where the exact function
    /// has `|f'| * w`. Measured on a 33-mode cascade profile, that was a factor
    /// of `7.4e5` at `w = 2e-6`, and the factor DIVERGED as the cell shrank
    /// (`O(w)` against `O(w^2)`).
    ///
    /// `resolution_flat_region` reads the value range, so the consequence was
    /// not a loose number but a search that could retire no cell and refused
    /// designs it could certify. The centred form in [`AffineRemlProfile::enclose`]
    /// restores the invariant by construction; this gate is what stops the
    /// natural extension coming back.
    ///
    /// The fixture is built to CANCEL: `g_i = s_i = q_i = 1` with
    /// `E = modes = dof = rank`, whose analytic score is exactly constant in
    /// `rho`, so `|f'|` is zero to roundoff and any first-order slack in the
    /// value range shows up immediately.
    #[test]
    fn the_value_enclosure_never_exceeds_the_bound_its_own_derivative_certifies() {
        const MODES: usize = 33;
        let grams = [1.0; MODES];
        let penalties = [1.0; MODES];
        let projected = [1.0; MODES];
        let energies = [MODES as f64];
        let profile = AffineRemlProfile::new(
            &grams,
            &penalties,
            &projected,
            &energies,
            MODES as f64,
            MODES,
            0.0,
        )
        .expect("valid cancellation fixture");

        let centre = -12.0_f64;
        let mut previous_width = f64::INFINITY;
        for exponent in [-1_i32, -2, -3, -4, -5, -6] {
            let half = 10.0_f64.powi(exponent);
            let (a, b) = (centre - half, centre + half);
            let width = b - a;
            let cell = profile.enclose(a, b).expect("cell enclosure");
            let point = profile.enclose(centre, centre).expect("point enclosure");

            // Soundness first: the cell's ranges must CONTAIN the degenerate
            // cell's, which is what `certify_endpoint_derivative` relies on and
            // what an intersection of two enclosures could otherwise break.
            assert!(
                cell.score.value.lo <= point.score.value.lo
                    && point.score.value.hi <= cell.score.value.hi,
                "w={width:e}: the midpoint value range {:?} escaped the cell range {:?}",
                point.score.value,
                cell.score.value
            );
            assert!(
                cell.derivative.lo <= point.derivative.lo
                    && point.derivative.hi <= cell.derivative.hi,
                "w={width:e}: the midpoint derivative range {:?} escaped the cell range {:?}",
                point.derivative,
                cell.derivative
            );

            let value_width = cell.score.value.hi - cell.score.value.lo;
            let point_width = point.score.value.hi - point.score.value.lo;
            let derivative_bound = cell.derivative.hi.abs().max(cell.derivative.lo.abs());
            let mean_value_bound = point_width + derivative_bound * width;
            assert!(
                value_width <= mean_value_bound * (1.0 + 1.0e-9),
                "w={width:e}: the value range is {value_width:e} wide but this cell's own \
                 derivative enclosure {:?} bounds the score's movement across it by \
                 {mean_value_bound:e} — the natural extension is back",
                cell.derivative
            );

            println!(
                "[GATE] w={width:e} value_width={value_width:e} point_width={point_width:e} \
                 mvt={mean_value_bound:e} D={derivative_bound:e}"
            );
            // And it must actually CONVERGE. A first-order range falls by 10
            // per decade of cell width; this one falls by ~1000, because the
            // remainder is `max|F'| * w` and `max|F'|` is itself centred. The
            // gate asks for better than 50 per decade — enough to separate
            // `O(w)` from anything above it without pinning a rate — until the
            // range reaches the floor every enclosure has, the width of the
            // DEGENERATE-cell reading, below which there is nothing left to
            // win. Measured floor here: 8.98e-12 on a score of magnitude ~30.
            assert!(
                value_width <= previous_width / 50.0 || value_width <= 2.0 * point_width,
                "w={width:e}: the value range fell only {previous_width:e} -> \
                 {value_width:e}, and it is not at the point-enclosure floor \
                 {point_width:e} — that is first-order behaviour"
            );
            previous_width = value_width;
        }
    }

    /// The centred form's degenerate and extreme cells.
    ///
    /// Centring introduces a second evaluation and an arithmetic that can
    /// produce an empty intersection or a non-finite remainder where the
    /// natural extension could not, so the cases where those are reachable are
    /// pinned rather than argued:
    ///
    /// * a POINT cell must return the natural extension untouched — the centred
    ///   form's remainder is exactly zero there and re-deriving it would only
    ///   add rounding;
    /// * a cell whose endpoints are ADJACENT binary64 values must still centre
    ///   at a point inside itself (`0.5*(lo+hi)` can round to either endpoint,
    ///   and a centre outside the cell would make the mean value theorem
    ///   inapplicable);
    /// * cells at the far ends of the representable `log lambda` domain, where
    ///   `lambda` is denormal at one end and near overflow at the other, must
    ///   stay sound: the centred range must contain the point range, and it may
    ///   never be wider than the natural extension it intersects.
    #[test]
    fn the_centred_enclosure_holds_on_degenerate_adjacent_and_extreme_cells() {
        let grams = [1.0, 4.0, 1.0e-9, 2.5e7];
        let penalties = [1.0, 1.0, 1.0, 1.0];
        let projected = [0.5, 0.25, 1.0e-3, 3.0];
        let energies = [8.0];
        let profile =
            AffineRemlProfile::new(&grams, &penalties, &projected, &energies, 6.0, 4, 0.25)
                .expect("valid fixture");

        for &x in &[-600.0_f64, -37.5, -1.0, 0.0, 2.75, 600.0] {
            let Ok((direct, _)) = profile.enclose_direct(x, x) else {
                continue;
            };
            let centred = profile.enclose(x, x).expect("a point cell must enclose");
            assert_eq!(
                centred, direct,
                "a point cell must return the natural extension untouched at x={x}"
            );

            // Adjacent binary64 endpoints: the tightest non-degenerate cell.
            let up = next_up(x);
            let Ok(cell) = profile.enclose(x, up) else {
                continue;
            };
            let point = profile.enclose(x, x).expect("point cell");
            assert!(
                cell.score.value.lo <= point.score.value.lo
                    && point.score.value.hi <= cell.score.value.hi,
                "adjacent-float cell at {x}: point value range {:?} escaped {:?}",
                point.score.value,
                cell.score.value
            );
            assert!(
                cell.derivative.lo <= point.derivative.lo
                    && point.derivative.hi <= cell.derivative.hi,
                "adjacent-float cell at {x}: point derivative range {:?} escaped {:?}",
                point.derivative,
                cell.derivative
            );
            assert!(
                cell.score.value.is_valid() && cell.derivative.is_valid(),
                "adjacent-float cell at {x} produced an invalid enclosure: {cell:?}"
            );

            // Intersecting can only tighten: never wider than the natural form.
            let (wide, _) = profile.enclose_direct(x, up).expect("direct adjacent cell");
            assert!(
                cell.score.value.lo >= wide.score.value.lo
                    && cell.score.value.hi <= wide.score.value.hi,
                "the centred value range {:?} is not inside the natural extension {:?} at {x}",
                cell.score.value,
                wide.score.value
            );
            assert!(
                cell.derivative.lo >= wide.derivative.lo
                    && cell.derivative.hi <= wide.derivative.hi,
                "the centred derivative range {:?} is not inside the natural extension {:?} at {x}",
                cell.derivative,
                wide.derivative
            );
        }
    }

    /// The 33 kept Schur modes and response energies of the cascade design in
    /// `gam_solve::residual_cascade`'s
    /// `auto_reml_certifies_a_design_the_data_cannot_identify` — 36 rows against
    /// 1725 columns — printed by that crate's `zz_probe_rank_deficient_` probe and
    /// carried here as literals so these gates need no design build and no
    /// dependency on gam-solve.
    ///
    /// A synthetic stand-in was tried first, twice, and neither reproduced the
    /// defect: it needs BOTH the multiscale spectrum and the near-interpolating
    /// response that makes the two score blocks cancel, and hand-built profiles
    /// kept landing on a monotone score the natural extension excludes by sign in
    /// a handful of cells. That is why these gates carry data rather than a
    /// formula.
    ///
    /// Returns `(gram_modes, penalty_modes, projected_rhs_squared, response_energy)`;
    /// the profile also takes `residual_dof = 33`, `determinant_rank = 33` and
    /// `logdet_constant = 9.226276711274537`, and its certified log-lambda domain
    /// is `[-21.860900258111, 18.75853229939662]`.
    fn cascade_profile_parts() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let grams = vec![
            0.021513523027428847, 0.023421509558465926, 0.024477791743994424,
            0.03028760364561828, 0.03510108223379587, 0.040671848915996144,
            0.042394860646972565, 0.044208976267946384, 0.046980397477518414,
            0.051041787441650194, 0.053417305918114666, 0.05575657456312382,
            0.056982691606415704, 0.059623191536431024, 0.06072593823762461,
            0.061603808142128846, 0.0626306391548814, 0.06415989316153273, 0.06612727525342801,
            0.07201682707299777, 0.10499606046436369, 0.12037535776467499, 0.1486138626340859,
            0.1762399329554861, 0.19315924476245142, 0.26688703253550705, 0.2848266927054469,
            0.33232244706214037, 0.6015439556821448, 1.1406886269841172, 1.3973782387809837,
            1.8043547873076875, 2.0890420358314765,
        ];
        let penalties = vec![1.0_f64; 33];
        let projected = vec![
            0.0008447602450715568, 0.004744115853417025, 0.0013711877079256205,
            0.000556576229807026, 0.00032950514304538826, 0.00015869074743770514,
            0.004035749350652998, 0.002408288703125203, 0.0002161132863778849,
            0.0024599052556113317, 0.00028155268264135145, 9.068039769807838e-7,
            0.0004390033211936947, 0.004642257342083, 5.722227645019854e-6,
            0.003702111930202603, 0.003943553329808974, 0.0011808139994261783,
            1.490921408482301e-5, 0.001728436851442388, 0.00040290378245105683,
            0.0006710268119971442, 0.0032383572156905664, 0.00013742753101732549,
            6.681227329297447e-5, 0.054339495839186305, 0.018972176651153957,
            0.04535732957447296, 0.1129209190002305, 0.05428138627351111, 1.5501891913959478,
            0.14151749008562448, 0.11704548115908926,
        ];
        let energies = vec![2.7067510572921663_f64];
        (grams, penalties, projected, energies)
    }

    /// The centred form's guard, and an honest account of what it is for.
    ///
    /// I wrote this guard for a hazard that turned out not to exist on the live
    /// path, so here is what is actually true, measured rather than argued.
    ///
    /// **The `inf * 0` story is closed already.** `ClosedInterval::mul` reduces
    /// four endpoint products with `f64::min`/`f64::max`, which IGNORE a NaN
    /// operand — so an `inf * 0` product would drop out of the reduction
    /// silently and leave a range strictly INSIDE the true one. But
    /// `product_down`/`product_up` treat a zero operand as exact and map the
    /// resulting NaN to `0.0`, so no NaN is ever produced that way, and
    /// `[-inf, -inf] * [-1, 0]` reduces to `[0, inf]` — correct. Enumerating
    /// every endpoint shape over `{-inf, -3, -1, 0, 1, 3, inf}` against a
    /// sampled product set finds no narrowing from any singly-infinite slope.
    ///
    /// **What is still open is a NaN arriving from elsewhere.** `product_is_exact`
    /// is false for a NaN against a non-unit, non-zero operand, so
    /// `[NaN, 1.0] * [-0.5, 0.5]` reduces to `[-0.5, 0.5]`: two of the four
    /// corners are dropped and a finite-looking, too-narrow range comes back with
    /// no signal at all. `enclose_direct` does not prove every accumulator finite,
    /// and `checked_enclosure` only validates the enclosure the search RECEIVES —
    /// by which point a NaN slope has already been used to narrow the value.
    ///
    /// So the guard excludes a non-finite slope and a non-finite remainder and
    /// keeps the natural extension, which is rigorous unconditionally. It is
    /// cheap, and it means a certified range does not rest on a case analysis of
    /// a rounding primitive three modules away that the next person to touch
    /// `mode_ranges` will not re-derive.
    #[test]
    fn the_centred_form_keeps_the_natural_extension_when_the_remainder_is_not_finite() {
        let direct = ClosedInterval::new(-10.0, 10.0);
        let point = ClosedInterval::new(-1.0, 1.0);
        let touching_zero = ClosedInterval::new(-0.5, 0.0);
        let straddling_zero = ClosedInterval::new(-0.5, 0.5);

        // The hazard, on the record as a fact rather than as a worry: a NaN
        // endpoint reduces to a finite-looking range narrower than the truth.
        let narrowed = ClosedInterval::new(f64::NAN, 1.0).mul(straddling_zero);
        assert!(
            narrowed.lo.is_finite() && narrowed.hi.is_finite(),
            "premise: a NaN endpoint must reduce to a finite-LOOKING range ({narrowed:?}); if \
             `mul` stops dropping it this gate is about nothing"
        );
        // And the `inf * 0` path, which is NOT a hazard, asserted so a change to
        // `product_down`'s zero handling shows up here rather than silently.
        let infinite = ClosedInterval::new(f64::NEG_INFINITY, f64::NEG_INFINITY)
            .mul(ClosedInterval::new(-1.0, 0.0));
        assert!(
            infinite.lo <= 0.0 && infinite.hi.is_infinite(),
            "`inf * 0` must stay sound through `product_down`'s exact-zero mapping, got \
             {infinite:?}"
        );

        for slope in [
            ClosedInterval::new(f64::NEG_INFINITY, 3.0),
            ClosedInterval::new(-3.0, f64::INFINITY),
            ClosedInterval::new(f64::NEG_INFINITY, f64::INFINITY),
            ClosedInterval::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
            ClosedInterval::new(f64::NAN, 1.0),
            ClosedInterval::new(1.0, f64::NAN),
        ] {
            for offset in [touching_zero, straddling_zero, ClosedInterval::new(0.0, 0.5)] {
                assert_eq!(
                    centred_or(direct, point, slope, offset),
                    direct,
                    "a non-finite slope {slope:?} over offset {offset:?} must leave the natural \
                     extension in place"
                );
            }
        }

        // And it still tightens when the remainder IS finite, so the guard has
        // not simply disabled the centred form.
        let tightened = centred_or(
            direct,
            point,
            ClosedInterval::new(-2.0, 2.0),
            straddling_zero,
        );
        assert!(
            tightened.lo > direct.lo && tightened.hi < direct.hi,
            "a finite remainder must still tighten: {tightened:?} against {direct:?}"
        );
    }

    /// The centred ranges contain the function at EVERY interior point, not just
    /// at the centre they were expanded about.
    ///
    /// This is the gate on the riskiest thing centring introduces. Each channel
    /// is now a mean value form anchored on the one above it — the curvature on
    /// an interval THIRD derivative, the derivative on the curvature, the value
    /// on the derivative — so an error in the third-derivative kernel does not
    /// make a range wide, it makes it NARROW, and a narrow certified range is an
    /// unsound proof rather than a slow one.
    ///
    /// The check needs no finite differences and no reference implementation.
    /// `enclose(x, x)` is the natural extension on a degenerate cell, which
    /// encloses the exact value, derivative and curvature AT `x`; so for every
    /// `x` in `[a, b]` the cell's ranges must contain the point's. Sampling `x`
    /// away from the midpoint is what exercises the remainder terms: at the
    /// centre they vanish identically and prove nothing.
    ///
    /// Sampled across four decades of cell width so the remainder is the
    /// dominant term at the wide end and roundoff dominates at the narrow one.
    #[test]
    fn the_centred_ranges_contain_the_function_at_every_interior_point() {
        let (grams, penalties, projected, energies) = cascade_profile_parts();
        let profile = AffineRemlProfile::new(
            &grams,
            &penalties,
            &projected,
            &energies,
            33.0,
            33,
            9.226276711274537,
        )
        .expect("valid cascade profile");

        // A containment check is only evidence if the range it checks was
        // actually NARROWED by the thing under test: if the intersection with
        // the natural extension were a no-op, every assertion below would hold
        // for any third-derivative kernel at all, including a wrong one.
        let mut curvature_tightened = false;
        // Spread over the design's own domain, including both saturated tails.
        for centre in [-20.0_f64, -12.5, -6.0, -1.679, 3.0, 11.0, 17.5] {
            for exponent in [0_i32, -1, -2, -3, -4] {
                let half = 10.0_f64.powi(exponent);
                let (a, b) = (centre - half, centre + half);
                let cell = profile.enclose(a, b).expect("cell enclosure");
                let (natural, _) = profile.enclose_direct(a, b).expect("natural extension");
                assert!(
                    cell.curvature.lo >= natural.curvature.lo
                        && cell.curvature.hi <= natural.curvature.hi,
                    "cell [{a}, {b}]: the centred curvature {:?} is not inside the natural \
                     extension {:?}",
                    cell.curvature,
                    natural.curvature
                );
                if cell.curvature.hi - cell.curvature.lo
                    < 0.5 * (natural.curvature.hi - natural.curvature.lo)
                {
                    curvature_tightened = true;
                }
                for step in 0..=8 {
                    let x = a + (b - a) * (step as f64 / 8.0);
                    let point = profile.enclose(x, x).expect("point enclosure");
                    assert!(
                        cell.score.value.lo <= point.score.value.lo
                            && point.score.value.hi <= cell.score.value.hi,
                        "cell [{a}, {b}] value range {:?} does not contain the exact value at \
                         x={x}, {:?}",
                        cell.score.value,
                        point.score.value
                    );
                    assert!(
                        cell.derivative.lo <= point.derivative.lo
                            && point.derivative.hi <= cell.derivative.hi,
                        "cell [{a}, {b}] derivative range {:?} does not contain the exact \
                         derivative at x={x}, {:?}",
                        cell.derivative,
                        point.derivative
                    );
                    assert!(
                        cell.curvature.lo <= point.curvature.lo
                            && point.curvature.hi <= cell.curvature.hi,
                        "cell [{a}, {b}] curvature range {:?} does not contain the exact \
                         curvature at x={x}, {:?} — the third-derivative kernel the curvature \
                         is centred on is wrong",
                        cell.curvature,
                        point.curvature
                    );
                }
            }
        }
        assert!(
            curvature_tightened,
            "the centred curvature never halved the natural extension's range anywhere in this \
             sweep, so the containment checks above would pass for a WRONG third-derivative \
             kernel too — this gate has gone vacuous"
        );
    }

    /// The located optimum is the SAME under both enclosure forms, and it is
    /// accurate to the search's location contract and no better.
    ///
    /// Two claims, and the second is the one that catches misuse.
    ///
    /// Tightening an enclosure changes which cells the search visits, so it
    /// could in principle move the point it returns. On the profile
    /// `gam_sae::identifiability::ridge_reml_select_weight` builds for its
    /// one-eigendirection closed-form fixture it does not: both oracles return
    /// the same abscissa to the last bit, from the same stationary bracket. That
    /// is worth pinning, because a caller comparing the returned `lambda` to a
    /// closed form cannot tell "the enclosure moved the answer" from "the
    /// enclosure was always allowed to".
    ///
    /// And it was always allowed to. The search certifies a stationary point's
    /// LOCATION to the requested resolution in `rho`, and returns an evaluated
    /// SAMPLE from that bracket rather than the bracket's midpoint or a
    /// polished root. So `|rho_hat - rho*|` is bounded by the resolution and by
    /// nothing smaller — measured here at `4.17e-9` against a requested
    /// `1.49e-8`, from a bracket `1.13e-8` wide. A caller wanting more than that
    /// has to polish the root itself; the fixture's exact `lambda = 1.2` is
    /// reproduced to `2.5e-9`, which is inside the contract and outside a `1e-9`
    /// tolerance that no version of this search has ever guaranteed.
    #[test]
    fn the_located_optimum_is_enclosure_independent_and_accurate_to_the_contract() {
        // eigvals=[2.0], signal=[8.0], aux_norm_sq=10.0, n_obs=5, n_responses=3
        // => pairs = [(1.0, 4.0)] in u = lambda/gamma_max, repeated once per
        // response, with residual_dof = n_obs*n_responses = 15.
        let grams = [1.0_f64; 3];
        let penalties = [1.0_f64; 3];
        let projected = [4.0 / 3.0; 3];
        let energies = [10.0_f64];
        let profile =
            AffineRemlProfile::new(&grams, &penalties, &projected, &energies, 15.0, 3, 0.0)
                .expect("valid ridge profile");
        let lo = certified_ln_positive(f64::MIN_POSITIVE).expect("lo").lo;
        let hi = certified_ln_positive(f64::MAX / 2.0).expect("hi").hi;
        let resolution = f64::EPSILON.sqrt();
        // The closed-form stationary point: lambda_hat = 1.2, and the profile is
        // built in u = lambda/gamma_max with gamma_max = 2.
        let truth = 0.6_f64;

        let natural = maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| profile.evaluate(x),
            |a, b| profile.enclose_direct(a.x, b.x).map(|(e, _)| e),
        )
        .expect("the natural extension decomposes this domain");
        let centred = maximize_score_1d(lo, hi, resolution, |x| profile.evaluate(x), |a, b| {
            profile.enclose(a.x, b.x)
        })
        .expect("the centred form decomposes this domain");

        assert_eq!(
            natural.optimum.x, centred.optimum.x,
            "the two enclosure forms located different optima ({} against {}); tightening may \
             change which cells are visited but must not move the certified root",
            natural.optimum.x, centred.optimum.x
        );
        for (label, search) in [("natural", &natural), ("centred", &centred)] {
            assert!(
                matches!(search.location, ScoreOptimumLocation::Stationary(_)),
                "{label}: this fixture has an interior stationary optimum, got {:?}",
                search.location
            );
            let offset = (search.optimum.x - truth.ln()).abs();
            assert!(
                offset <= resolution,
                "{label}: the located root is {offset:e} from the closed form in rho, outside \
                 the requested resolution {resolution:e} — that is a location-contract failure"
            );
            // And no better, which is the half a caller must not assume: the
            // returned point is a sample from the bracket, not a polished root.
            assert!(
                offset > 0.0,
                "{label}: an exactly-attained root would mean this gate has stopped measuring \
                 what it claims"
            );
        }
    }

    /// COST. Centring doubles the per-cell work (one extra degenerate-cell
    /// evaluation), so the net is only a win if it removes more cells than that.
    /// This measures both oracles on the same searches and prints the ratio.
    ///
    /// Two shapes, because they pull in opposite directions: the cascade profile
    /// on its own 40.6-wide domain, where the natural extension cannot finish at
    /// all, and a well-conditioned profile on the FULL representable log-lambda
    /// domain (`ln(MIN_POSITIVE)` to `ln(MAX/2)`, 1417 wide) — which is what
    /// `gam_sae::identifiability::ridge_reml_select_weight` searches, and the
    /// case where a search that already succeeded cheaply could only get slower.
    #[test]
    fn zz_measure_centred_enclosure_search_cost() {
        let (grams, penalties, projected, energies) = cascade_profile_parts();
        let cascade = AffineRemlProfile::new(
            &grams,
            &penalties,
            &projected,
            &energies,
            33.0,
            33,
            9.226276711274537,
        )
        .expect("valid cascade profile");

        let full_lo = certified_ln_positive(f64::MIN_POSITIVE).expect("domain lo").lo;
        let full_hi = certified_ln_positive(f64::MAX / 2.0).expect("domain hi").hi;
        let cases: [(&str, f64, f64); 3] = [
            // The domain the design declares, where the natural extension
            // cannot finish at all.
            ("cascade/40.6-wide", -21.860900258111, 18.75853229939662),
            // A narrow window around the optimum (-1.679), where the natural
            // extension already succeeds in a handful of cells. This is the
            // case centring could only make SLOWER, since there are no cells
            // left for it to remove.
            ("cascade/narrow-around-the-optimum", -3.0, 0.0),
            // The full representable log-lambda domain, 1417 wide, which is what
            // `gam_sae::identifiability::ridge_reml_select_weight` searches.
            ("cascade/full-representable-domain", full_lo, full_hi),
        ];

        for (label, lo, hi) in cases {
            let profile = &cascade;
            let resolution = f64::EPSILON.sqrt();
            let started = std::time::Instant::now();
            let natural = maximize_score_1d(
                lo,
                hi,
                resolution,
                |x| profile.evaluate(x),
                |a, b| profile.enclose_direct(a.x, b.x).map(|(e, _)| e),
            );
            let natural_seconds = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            let centred = maximize_score_1d(
                lo,
                hi,
                resolution,
                |x| profile.evaluate(x),
                |a, b| profile.enclose(a.x, b.x),
            );
            let centred_seconds = started.elapsed().as_secs_f64();
            println!(
                "#COST {label}: natural {:.4}s ({}) centred {:.4}s ({}) speedup {:.2}x",
                natural_seconds,
                natural.as_ref().map_or("REFUSED", |_| "ok"),
                centred_seconds,
                centred.as_ref().map_or("REFUSED", |_| "ok"),
                natural_seconds / centred_seconds.max(f64::MIN_POSITIVE),
            );
            // A refusal is a legitimate outcome for some domains (the full
            // representable one reaches lambda values where the profiled
            // residual is not evaluable at all); what must never happen is the
            // centred oracle refusing where the natural one succeeds.
            assert!(
                centred.is_ok() || natural.is_err(),
                "{label}: the centred oracle refused ({centred:?}) where the natural extension \
                 succeeded — an intersection can only tighten, so this is impossible unless the \
                 centred form is unsound"
            );
            // The per-cell cost is at most 2x, so a search that certifies under
            // BOTH oracles must not lose more than that. A wider loss means the
            // centred form is provoking work rather than removing it.
            if natural.is_ok() {
                assert!(
                    centred_seconds <= natural_seconds * 2.5 + 1.0e-3,
                    "{label}: centring cost {centred_seconds:.4}s against the natural \
                     extension's {natural_seconds:.4}s — more than the doubled per-cell work \
                     can explain"
                );
            }
        }
    }

    /// The capability the centred form buys, pinned by running the SAME search
    /// twice on the same profile with the two enclosure forms.
    ///
    /// Every other gate here measures a width. This one measures the only thing
    /// a width is for: whether the certified search can decompose the domain at
    /// all. The natural extension is not removed by the fix — it is still what
    /// the centred form is built from and intersected with — so it stays
    /// callable, and that makes the before/after a controlled comparison inside
    /// one test rather than a claim about a previous commit.
    ///
    /// The fixture is the cascade's shape rather than its data: modes spread
    /// over nine decades (what a multilevel frame's Schur complement looks
    /// like), response energy split across them, and `dof = rank = modes`, so
    /// the log-determinant and deviance blocks each move by `O(rank)` per unit
    /// of `rho` while the score does not — the cancellation that the natural
    /// extension cannot see.
    #[test]
    fn the_natural_extension_cannot_decompose_a_domain_the_centred_form_certifies() {
        let (grams, penalties, projected, energies) = cascade_profile_parts();
        let profile = AffineRemlProfile::new(
            &grams,
            &penalties,
            &projected,
            &energies,
            33.0,
            33,
            9.226276711274537,
        )
        .expect("valid cascade profile");

        // The design's own certified log-lambda domain, 40.6 wide.
        let (lo, hi) = (-21.860900258111_f64, 18.75853229939662);
        let resolution = f64::EPSILON.sqrt();

        let natural = maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| profile.evaluate(x),
            |a, b| profile.enclose_direct(a.x, b.x).map(|(enclosure, _)| enclosure),
        );
        let centred = maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| profile.evaluate(x),
            |a, b| profile.enclose(a.x, b.x),
        );

        let centred = centred.unwrap_or_else(|error| {
            panic!(
                "the centred enclosure must decompose this 33-mode cascade domain: {error}"
            )
        });
        assert!(
            matches!(
                natural,
                Err(ScoreSearchError::SubdivisionBudget { .. } | ScoreSearchError::Unresolved { .. })
            ),
            "PREMISE LOST: the natural extension now decomposes this domain \
             ({natural:?}), so this fixture no longer exercises the defect and the \
             comparison below proves nothing — widen the mode spread or the domain \
             until it refuses again",
        );

        // And the answer it reaches is a real one, not a shrug: a decided
        // location whose global value ordering closed.
        assert!(
            !matches!(centred.location, ScoreOptimumLocation::ResolutionFlat(_)),
            "the centred search must decide a location, got {:?}",
            centred.location
        );
        assert!(
            centred.value_certificate.maximum_excess
                <= centred.value_certificate.comparison_resolution,
            "the centred search's value ordering must close: excess {} against {}",
            centred.value_certificate.maximum_excess,
            centred.value_certificate.comparison_resolution
        );
        assert!(
            centred.optimum.x >= lo && centred.optimum.x <= hi && centred.optimum.x.is_finite(),
            "the selected log lambda must lie in the domain, got {}",
            centred.optimum.x
        );
    }

    #[test]
    fn affine_reml_zero_smoothing_schur_residual_keeps_division_low_parts() {
        // Three exact-real quotients 1/3 sum to one, although no individual
        // quotient is representable in binary64. A directed interval around
        // each independently rounded quotient leaves an O(u) uncertainty in
        // `1 - 3*(1/3)`, larger than this profile's residual at lambda=1e-10.
        // The one-time TwoSum/FMA construction retains the division low parts,
        // so the exact zero Schur residual remains resolved near O(u²).
        let grams = [3.0; 3];
        let penalties = [1.0; 3];
        let projected = [1.0; 3];
        let energies = [1.0];
        let profile =
            AffineRemlProfile::new(&grams, &penalties, &projected, &energies, 3.0, 3, 0.0)
                .expect("valid nonrepresentable-quotient fixture");

        let zero_residual = profile.zero_lambda_residual[0];
        assert!(
            zero_residual.contains_zero(),
            "the exact identity 1 - 3*(1/3) = 0 must be retained: {zero_residual:?}"
        );
        assert!(
            zero_residual.hi - zero_residual.lo < 1.0e-28,
            "division corrections must live below ordinary binary64 cancellation scale: \
             {zero_residual:?}"
        );

        let rho = -23.025850929940457_f64;
        let enclosure = profile
            .enclose(rho, rho)
            .expect("the small positive smoothing residual must remain resolved");
        assert!(
            enclosure.derivative.contains_zero(),
            "determinant and residual derivatives cancel analytically: {:?}",
            enclosure.derivative
        );
        assert!(
            enclosure.derivative.hi - enclosure.derivative.lo < 1.0e-6,
            "the exact Schur residual must control the profiled derivative: {:?}",
            enclosure.derivative
        );
    }

    #[test]
    fn affine_reml_saturated_tail_preserves_complement_signs() {
        let profile = AffineRemlProfile::new(&[1.0], &[1.0], &[0.0], &[1.0], 4.0, 1, 0.0)
            .expect("valid saturated-tail fixture");
        let log_lambda = 700.0;
        let jet = profile.evaluate(log_lambda).expect("point jet");
        let enclosure = profile
            .enclose(log_lambda, log_lambda)
            .expect("point enclosure");

        assert!(
            jet.derivative > 0.0,
            "the point derivative must preserve +0.5/(1+exp(rho)), got {}",
            jet.derivative
        );
        assert!(
            jet.curvature < 0.0,
            "the point curvature must preserve its negative u*c sign, got {}",
            jet.curvature
        );
        assert!(
            enclosure.curvature.hi <= 0.0,
            "the exact saturated curvature remains nonpositive: {:?}",
            enclosure.curvature
        );
        assert!(
            enclosure.derivative.lo >= 0.0,
            "the exact saturated derivative remains nonnegative: {:?}",
            enclosure.derivative
        );
        let score = enclosure.score;
        assert!(score.evaluation_error.is_finite());
        assert!(
            score
                .value
                .widen(score.evaluation_error)
                .contains(jet.value),
            "the stable score evaluator must lie inside its exact value range plus forward error"
        );
    }

    #[test]
    fn affine_reml_extreme_domain_one_direction_encloses_and_maximizes_repeatably() {
        // The normalized one-direction ridge profile behind the gam-sae
        // regressions has
        //
        //   R(lambda) = 10 - 4/(1 + lambda),
        //
        // so its exact residual stays in [6, 10] over the complete finite
        // lambda domain. Repeating the direction three times reproduces the
        // response multiplicity of that caller and plants the stationary point
        // at lambda/gamma_max = 0.6.
        let gram_modes = [1.0, 1.0, 1.0];
        let penalty_modes = [1.0, 1.0, 1.0];
        let projected_rhs_squared = [4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0];
        let response_energy = [10.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            15.0,
            3,
            0.0,
        )
        .expect("valid normalized one-direction ridge profile");
        let rho_lo = certified_ln_positive(f64::MIN_POSITIVE)
            .expect("finite-domain lower log bound")
            .lo;
        let rho_hi = certified_ln_positive(f64::MAX / 2.0)
            .expect("finite-domain upper log bound")
            .hi;

        let whole_domain = profile
            .enclose(rho_lo, rho_hi)
            .expect("scale-safe relative exp error keeps the full-domain residual finite");
        assert!(
            whole_domain.score.value.is_valid()
                && whole_domain.score.value.lo.is_finite()
                && whole_domain.score.value.hi.is_finite()
        );
        assert!(whole_domain.score.evaluation_error.is_finite());
        assert!(whole_domain.derivative.contains_zero());

        let resolution = f64::EPSILON.sqrt();
        let first = profile
            .maximize_value_ordered(rho_lo, rho_hi, resolution)
            .expect("finite subdivision must certify the planted stationary optimum");
        let repeated = profile
            .maximize_value_ordered(rho_lo, rho_hi, resolution)
            .expect("the same exact search must be repeatable");
        assert_eq!(first, repeated);
        let ScoreOptimumLocation::Stationary(index) = first.location else {
            panic!(
                "the planted one-direction optimum must be stationary, got {:?}",
                first.location
            );
        };
        let stationary = first
            .stationary_points
            .get(index)
            .expect("stationary result index");
        let expected = certified_ln_positive(0.6).expect("analytic stationary log");
        assert!(
            stationary.bracket.lo <= expected.lo && stationary.bracket.hi >= expected.hi,
            "certified bracket {:?} must contain analytic log(0.6) {:?}",
            stationary.bracket,
            expected
        );
        assert!(
            first.value_certificate.maximum_excess <= first.value_certificate.comparison_resolution,
            "an isolated stationary root is not yet a globally ordered score candidate: \
             maximum excess {}, comparison resolution {}, bracket {:?}",
            first.value_certificate.maximum_excess,
            first.value_certificate.comparison_resolution,
            stationary.bracket,
        );
    }

    #[test]
    fn affine_reml_gram_zero_subnormal_zero_projection_is_structural() {
        let minimum_subnormal = f64::from_bits(1);
        let log_lambda = -740.0;
        let lambda = exp_interval(log_lambda, log_lambda)
            .expect("the fixture needs a certified subnormal lambda");
        assert!(lambda.lo > 0.0 && lambda.hi < f64::MIN_POSITIVE);
        let raw_h = lambda.mul(ClosedInterval::point(minimum_subnormal));
        assert!(
            raw_h.lo < 0.0 && raw_h.hi > 0.0,
            "the raw outward product must cross rounded zero: {raw_h:?}"
        );
        let h = raw_h.nonnegative();
        assert_eq!(
            h.lo, 0.0,
            "known nonnegative product must clamp its outward lower bound to zero"
        );

        let ranges = mode_ranges(0.0, minimum_subnormal, 0.0, lambda)
            .expect("the zero projection cancels before any residual division");
        assert_eq!(ranges.c, ClosedInterval::point(0.0));
        assert_eq!(ranges.w, ClosedInterval::point(0.0));
        assert_eq!(ranges.v, ClosedInterval::point(0.0));
        assert_eq!(ranges.p, ClosedInterval::point(0.0));
        assert_eq!(ranges.q, ClosedInterval::point(0.0));

        let gram_modes = [0.0];
        let penalty_modes = [minimum_subnormal];
        let projected_rhs_squared = [0.0];
        let response_energy = [1.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            1.0,
            1,
            0.0,
        )
        .expect("valid gram-zero structural fixture");
        let jet = profile
            .evaluate(log_lambda)
            .expect("normalized determinant and zero residual projection stay finite");
        let enclosure = profile
            .enclose(log_lambda, log_lambda)
            .expect("the proof path must not divide by a zero-containing h interval");
        assert_eq!(jet.derivative, 0.0);
        assert_eq!(jet.curvature, 0.0);
        assert!(is_exact_zero(enclosure.derivative));
        assert!(is_exact_zero(enclosure.curvature));
        assert!(
            enclosure
                .score
                .value
                .widen(enclosure.score.evaluation_error)
                .contains(jet.value)
        );
    }

    #[test]
    fn affine_reml_gram_zero_subnormal_nonzero_projection_stays_finite() {
        let minimum_subnormal = f64::from_bits(1);
        let log_lambda = -740.0;
        let lambda = exp_interval(log_lambda, log_lambda)
            .expect("the fixture needs a certified subnormal lambda");
        let penalty = 0.01;
        let h = lambda.mul(ClosedInterval::point(penalty)).nonnegative();
        assert_eq!(
            h.lo, 0.0,
            "the fixture must enter the structural quotient path"
        );

        let ranges = mode_ranges(0.0, penalty, minimum_subnormal, lambda)
            .expect("the scaled quotient has a finite representable range");
        assert_eq!(ranges.c, ClosedInterval::point(0.0));
        assert_eq!(ranges.w, ClosedInterval::point(0.0));
        assert!(ranges.v.lo > 0.0 && ranges.v.hi.is_finite());
        assert_eq!(ranges.p, ranges.v);
        assert_eq!(ranges.q, ranges.v.neg());

        let gram_modes = [0.0];
        let penalty_modes = [penalty];
        let projected_rhs_squared = [minimum_subnormal];
        let response_energy = [10.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            1.0,
            1,
            0.0,
        )
        .expect("valid gram-zero finite-ratio fixture");
        let jet = profile
            .evaluate(log_lambda)
            .expect("the point ratio must avoid the underflowing product");
        let enclosure = profile
            .enclose(log_lambda, log_lambda)
            .expect("the interval ratio must remain finite without a reciprocal overflow");
        assert!(
            enclosure
                .score
                .value
                .widen(enclosure.score.evaluation_error)
                .contains(jet.value)
        );
    }

    #[test]
    fn affine_reml_gram_zero_unrepresentable_projection_is_typed() {
        let minimum_subnormal = f64::from_bits(1);
        let log_lambda = -740.0;
        let gram_modes = [0.0];
        let penalty_modes = [minimum_subnormal];
        let projected_rhs_squared = [1.0];
        let response_energy = [10.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            1.0,
            1,
            0.0,
        )
        .expect("valid gram-zero refusal fixture");
        assert!(matches!(
            profile.evaluate(log_lambda),
            Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "gram-zero residual quotient",
                ..
            })
        ));
        assert!(matches!(
            profile.enclose(log_lambda, log_lambda),
            Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "gram-zero residual quotient",
                ..
            })
        ));
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
