//! The evidence status of every reported number (#2951).
//!
//! # Evidence status
//!
//! A reported number is a claim about a quantity `Q` over a declared domain `D`:
//! a mask domain, a finite family of masks, a covering region, or a sampling law.
//! The number is an extremum of `Q` over the domain (a worst-case divergence, a
//! minimum support code) or an expectation of `Q` under a law. [`EvidenceStatus`]
//! states what the number is allowed to mean and keeps five cases apart.
//!
//! * **Exact.** An algebraic identity, or exhaustive evaluation over a stated
//!   finite family. The value carries the numerical error of its own evaluation.
//! * **Uniform bound.** `Q(w) <= upper` at every `w` of a stated region, with the
//!   numerical error already inside `upper`.
//! * **Statistical estimate.** An expectation under a stated law, with its
//!   standard error. It bounds no extremum. A stochastic-mask mean, an observed
//!   worst case and a certified bound are three different numbers (P10), so
//!   [`EvidenceStatus::upper_bound`] and [`EvidenceStatus::lower_bound`] return
//!   `None` for it.
//! * **Counterexample.** A witness at which `Q` exceeds a threshold by more than
//!   the numerical error of the witness value.
//! * **Unresolved.** A lower and an upper bound that have not met, the gap
//!   between them, and which side a witness attains.
//!
//! Every variant carries a [`Validated`] field that only this module can build, so
//! each status passes through a constructor. The constructors refuse non-finite
//! values, negative numerical or standard error, an empty exhaustive family, an
//! inverted interval, a witness for an underived side, and a counterexample that
//! roundoff could explain. They check that a status is well formed; they cannot
//! check that the caller's computation supports it.

use std::cmp::Ordering;
use std::fmt;

/// Proof that an [`EvidenceStatus`] passed through its constructor. Only this
/// module can build one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Validated(());

/// Why an [`EvidenceStatus::Exact`] value is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactBasis {
    /// An algebraic identity over the whole domain, e.g. P8's support function
    /// `h(u) = sum_{c free} max(least_c u'v_c, most_c u'v_c)`, where control `c`'s
    /// deletion `1 - m_c` ranges over `[least_c, most_c]`; on the `[0, 1]` domain this is
    /// `sum_{c free} max(0, u'v_c)`.
    Algebraic,
    /// Every one of the `cardinality` members of the finite family named by the
    /// domain was evaluated.
    Exhaustive { cardinality: u64 },
}

/// Which extremum a reported number is, and so which side of an interval a
/// witness attains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extremum {
    /// `sup_{w in D} Q(w)`. A witness attains a lower bound; the upper bound needs
    /// a derivation (a Lipschitz covering, P9 convexity).
    Supremum,
    /// `inf_{w in D} Q(w)`. A witness attains an upper bound; the lower bound
    /// needs a derivation (a relaxation, a closed search tree).
    Infimum,
}

/// What a reported number about `Q` over the declared domain `D` is allowed to
/// mean. `W` is a witness point of the domain (a mask, a support, an input).
#[derive(Clone, Debug, PartialEq)]
pub enum EvidenceStatus<W, D> {
    /// The extremum is `value`, up to `numerical_error` from its evaluation.
    Exact {
        value: f64,
        numerical_error: f64,
        basis: ExactBasis,
        witness: Option<W>,
        domain: D,
        validated: Validated,
    },
    /// `Q(w) <= upper` for every `w` in `region`. `upper` already includes
    /// `numerical_error`, which is kept for audit.
    UniformBound {
        upper: f64,
        numerical_error: f64,
        region: D,
        validated: Validated,
    },
    /// An estimate of `E_law Q` with its standard error, from `samples` draws. It
    /// is a different number from any extremum over the domain.
    StatisticalEstimate {
        estimate: f64,
        standard_error: f64,
        samples: u64,
        law: D,
        validated: Validated,
    },
    /// `Q(witness) = value` and `value - numerical_error > threshold`, so the
    /// claim `sup_D Q <= threshold` is false.
    Counterexample {
        value: f64,
        numerical_error: f64,
        threshold: f64,
        witness: W,
        validated: Validated,
    },
    /// `lower <= extremum <= upper` over `domain`, each bound already widened by
    /// its numerical error. An underived side is infinite. For a supremum the
    /// witness attains `lower`; for an infimum it attains `upper`.
    Unresolved {
        lower: f64,
        upper: f64,
        extremum: Extremum,
        witness: Option<W>,
        domain: D,
        validated: Validated,
    },
}

/// Why a constructor refused to build an [`EvidenceStatus`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvidenceStatusError {
    /// A value that must be finite was NaN or infinite.
    NonFinite { field: &'static str, value: f64 },
    /// A numerical error or standard error was negative.
    Negative { field: &'static str, value: f64 },
    /// An exhaustive evaluation over a family with no members.
    EmptyExhaustiveFamily,
    /// A statistical estimate from no samples.
    NoSamples,
    /// An interval bound that is NaN, or infinite on the side that excludes every
    /// value (`lower = +inf` or `upper = -inf`).
    InvalidBound { field: &'static str, value: f64 },
    /// `lower > upper`.
    InvertedInterval { lower: f64, upper: f64 },
    /// A witness for the side of an interval that has no finite value.
    UnattainedWitness { extremum: Extremum },
    /// `value - numerical_error` does not exceed `threshold`, so roundoff could
    /// explain the violation.
    NotAViolation {
        value: f64,
        numerical_error: f64,
        threshold: f64,
    },
}

impl fmt::Display for EvidenceStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { field, value } => {
                write!(f, "evidence field {field} must be finite, got {value}")
            }
            Self::Negative { field, value } => {
                write!(f, "evidence field {field} must be nonnegative, got {value}")
            }
            Self::EmptyExhaustiveFamily => {
                write!(f, "an exhaustive evaluation needs a nonempty finite family")
            }
            Self::NoSamples => write!(f, "a statistical estimate needs at least one sample"),
            Self::InvalidBound { field, value } => {
                write!(f, "interval bound {field} = {value} is NaN or excludes every value")
            }
            Self::InvertedInterval { lower, upper } => {
                write!(f, "interval lower bound {lower} exceeds upper bound {upper}")
            }
            Self::UnattainedWitness { extremum } => {
                write!(f, "a witness of the {extremum:?} must attain a finite bound")
            }
            Self::NotAViolation {
                value,
                numerical_error,
                threshold,
            } => write!(
                f,
                "value {value} with numerical error {numerical_error} does not exceed threshold {threshold}"
            ),
        }
    }
}

impl std::error::Error for EvidenceStatusError {}

fn require_finite(field: &'static str, value: f64) -> Result<(), EvidenceStatusError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EvidenceStatusError::NonFinite { field, value })
    }
}

fn require_nonnegative(field: &'static str, value: f64) -> Result<(), EvidenceStatusError> {
    require_finite(field, value)?;
    if value < 0.0 {
        Err(EvidenceStatusError::Negative { field, value })
    } else {
        Ok(())
    }
}

/// `value + numerical_error` rounded up, so it bounds the exact sum. Adding zero is
/// exact, so an exactly evaluated value is not widened.
fn rounded_up_sum(value: f64, numerical_error: f64) -> f64 {
    let sum = value + numerical_error;
    if numerical_error == 0.0 { sum } else { sum.next_up() }
}

/// `value - numerical_error` rounded down, so the exact difference bounds it.
fn rounded_down_difference(value: f64, numerical_error: f64) -> f64 {
    let difference = value - numerical_error;
    if numerical_error == 0.0 {
        difference
    } else {
        difference.next_down()
    }
}

impl<W, D> EvidenceStatus<W, D> {
    /// An exact extremum: an algebraic identity, or an exhaustive evaluation over
    /// the finite family `domain`.
    pub fn exact(
        value: f64,
        numerical_error: f64,
        basis: ExactBasis,
        witness: Option<W>,
        domain: D,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("value", value)?;
        require_nonnegative("numerical_error", numerical_error)?;
        if matches!(basis, ExactBasis::Exhaustive { cardinality: 0 }) {
            return Err(EvidenceStatusError::EmptyExhaustiveFamily);
        }
        Ok(Self::Exact {
            value,
            numerical_error,
            basis,
            witness,
            domain,
            validated: Validated(()),
        })
    }

    /// A bound `Q(w) <= upper` derived for every `w` in `region`, with
    /// `numerical_error` already included in `upper`.
    pub fn uniform_bound(
        upper: f64,
        numerical_error: f64,
        region: D,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("upper", upper)?;
        require_nonnegative("numerical_error", numerical_error)?;
        Ok(Self::UniformBound {
            upper,
            numerical_error,
            region,
            validated: Validated(()),
        })
    }

    /// An estimate of the expectation of `Q` under `law`, with its standard error.
    pub fn statistical_estimate(
        estimate: f64,
        standard_error: f64,
        samples: u64,
        law: D,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("estimate", estimate)?;
        require_nonnegative("standard_error", standard_error)?;
        if samples == 0 {
            return Err(EvidenceStatusError::NoSamples);
        }
        Ok(Self::StatisticalEstimate {
            estimate,
            standard_error,
            samples,
            law,
            validated: Validated(()),
        })
    }

    /// A witness refuting `sup_D Q <= threshold`. Refused unless the violation
    /// exceeds the numerical error of `value`.
    pub fn counterexample(
        value: f64,
        numerical_error: f64,
        threshold: f64,
        witness: W,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("value", value)?;
        require_nonnegative("numerical_error", numerical_error)?;
        require_finite("threshold", threshold)?;
        if rounded_down_difference(value, numerical_error) <= threshold {
            return Err(EvidenceStatusError::NotAViolation {
                value,
                numerical_error,
                threshold,
            });
        }
        Ok(Self::Counterexample {
            value,
            numerical_error,
            threshold,
            witness,
            validated: Validated(()),
        })
    }

    /// Bounds `lower <= extremum <= upper` that have not met. An underived side is
    /// infinite; a witness must attain a finite side (`lower` for a supremum,
    /// `upper` for an infimum).
    pub fn unresolved(
        lower: f64,
        upper: f64,
        extremum: Extremum,
        witness: Option<W>,
        domain: D,
    ) -> Result<Self, EvidenceStatusError> {
        if lower.is_nan() || lower == f64::INFINITY {
            return Err(EvidenceStatusError::InvalidBound {
                field: "lower",
                value: lower,
            });
        }
        if upper.is_nan() || upper == f64::NEG_INFINITY {
            return Err(EvidenceStatusError::InvalidBound {
                field: "upper",
                value: upper,
            });
        }
        if lower > upper {
            return Err(EvidenceStatusError::InvertedInterval { lower, upper });
        }
        let attained = match extremum {
            Extremum::Supremum => lower.is_finite(),
            Extremum::Infimum => upper.is_finite(),
        };
        if witness.is_some() && !attained {
            return Err(EvidenceStatusError::UnattainedWitness { extremum });
        }
        Ok(Self::Unresolved {
            lower,
            upper,
            extremum,
            witness,
            domain,
            validated: Validated(()),
        })
    }

    /// An upper bound on the reported extremum that this status proves, if any.
    pub fn upper_bound(&self) -> Option<f64> {
        match self {
            Self::Exact {
                value,
                numerical_error,
                ..
            } => Some(rounded_up_sum(*value, *numerical_error)),
            Self::UniformBound { upper, .. } => Some(*upper),
            Self::Unresolved { upper, .. } if upper.is_finite() => Some(*upper),
            Self::Unresolved { .. } | Self::StatisticalEstimate { .. } | Self::Counterexample { .. } => {
                None
            }
        }
    }

    /// A lower bound on the reported extremum that this status proves, if any. A
    /// counterexample bounds the supremum it refutes from below.
    pub fn lower_bound(&self) -> Option<f64> {
        match self {
            Self::Exact {
                value,
                numerical_error,
                ..
            }
            | Self::Counterexample {
                value,
                numerical_error,
                ..
            } => Some(rounded_down_difference(*value, *numerical_error)),
            Self::Unresolved { lower, .. } if lower.is_finite() => Some(*lower),
            Self::Unresolved { .. } | Self::UniformBound { .. } | Self::StatisticalEstimate { .. } => {
                None
            }
        }
    }

    /// `upper - lower` for an unresolved status, infinite when a side is underived.
    pub fn gap(&self) -> Option<f64> {
        match self {
            Self::Unresolved { lower, upper, .. } => Some(upper - lower),
            Self::Exact { .. }
            | Self::UniformBound { .. }
            | Self::StatisticalEstimate { .. }
            | Self::Counterexample { .. } => None,
        }
    }

    /// True when this status proves the extremum is at most `threshold`.
    pub fn certifies_at_most(&self, threshold: f64) -> bool {
        self.upper_bound().is_some_and(|upper| upper <= threshold)
    }

    /// True when this status proves the extremum exceeds `threshold`.
    pub fn refutes_at_most(&self, threshold: f64) -> bool {
        self.lower_bound().is_some_and(|lower| lower > threshold)
    }

    /// The witness the status carries, if any.
    pub fn witness(&self) -> Option<&W> {
        match self {
            Self::Exact { witness, .. } | Self::Unresolved { witness, .. } => witness.as_ref(),
            Self::Counterexample { witness, .. } => Some(witness),
            Self::UniformBound { .. } | Self::StatisticalEstimate { .. } => None,
        }
    }

    /// The status's witness, if any, taken by value.
    pub fn into_witness(self) -> Option<W> {
        match self {
            Self::Exact { witness, .. } | Self::Unresolved { witness, .. } => witness,
            Self::Counterexample { witness, .. } => Some(witness),
            Self::UniformBound { .. } | Self::StatisticalEstimate { .. } => None,
        }
    }

    /// The declared domain, region or law the status is stated over. A
    /// counterexample states only its witness.
    pub fn domain(&self) -> Option<&D> {
        match self {
            Self::Exact { domain, .. } | Self::Unresolved { domain, .. } => Some(domain),
            Self::UniformBound { region, .. } => Some(region),
            Self::StatisticalEstimate { law, .. } => Some(law),
            Self::Counterexample { .. } => None,
        }
    }

    /// The order of what two statuses about one quantity prove. `self` is stronger
    /// when the interval it proves, `[lower_bound or -inf, upper_bound or +inf]`,
    /// lies inside `other`'s: every threshold the weaker status certifies or refutes,
    /// the stronger one certifies or refutes too. The kind of evidence does not
    /// order anything by itself: an exact value with a wide numerical error proves
    /// less about `<= t` than a tight uniform bound at `t`. Statuses over different
    /// domains, about different extrema, or involving a statistical estimate (a
    /// different number) or a counterexample (which states no domain) compare as
    /// `None`, incomparable, as do intervals neither of which contains the other.
    pub fn compare_strength(&self, other: &Self) -> Option<Ordering>
    where
        D: PartialEq,
    {
        if matches!(self, Self::StatisticalEstimate { .. })
            || matches!(other, Self::StatisticalEstimate { .. })
        {
            return None;
        }
        let (Some(domain), Some(other_domain)) = (self.domain(), other.domain()) else {
            return None;
        };
        if domain != other_domain {
            return None;
        }
        if let (Some(extremum), Some(other_extremum)) =
            (self.bounded_extremum(), other.bounded_extremum())
            && extremum != other_extremum
        {
            return None;
        }
        let (lower, upper) = self.proven_interval();
        let (other_lower, other_upper) = other.proven_interval();
        let inside = lower >= other_lower && upper <= other_upper;
        let contains = lower <= other_lower && upper >= other_upper;
        match (inside, contains) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Greater),
            (false, true) => Some(Ordering::Less),
            (false, false) => None,
        }
    }

    /// The interval this status proves the extremum lies in, with an unproved side
    /// infinite.
    fn proven_interval(&self) -> (f64, f64) {
        (
            self.lower_bound().unwrap_or(f64::NEG_INFINITY),
            self.upper_bound().unwrap_or(f64::INFINITY),
        )
    }

    /// The extremum a status names: a uniform bound and a counterexample are about a
    /// supremum, an unresolved bracket states its own, and an exact value is either.
    fn bounded_extremum(&self) -> Option<Extremum> {
        match self {
            Self::UniformBound { .. } | Self::Counterexample { .. } => Some(Extremum::Supremum),
            Self::Unresolved { extremum, .. } => Some(*extremum),
            Self::Exact { .. } | Self::StatisticalEstimate { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};
    use std::cmp::Ordering;

    type Status = EvidenceStatus<Vec<usize>, &'static str>;

    #[test]
    fn a_statistical_estimate_never_bounds_the_extremum_while_the_same_number_as_a_bound_does() {
        // P10: n pieces of a cancelling pair +-a with iid U[0,1] masks give
        // E E_n^2 = a^2 / (6n), a mean, while sup |E_n| = |a| is the extremum.
        let a = 1.0_f64;
        let pieces = 8.0_f64;
        let mean = a * a / (6.0 * pieces);
        let estimate = Status::statistical_estimate(mean, 0.01, 1000, "iid U[0,1] masks")
            .expect("a finite estimate with a nonnegative standard error");
        assert_eq!(estimate.upper_bound(), None);
        assert_eq!(estimate.lower_bound(), None);
        assert!(!estimate.certifies_at_most(mean));
        assert!(!estimate.certifies_at_most(f64::MAX));
        assert!(!estimate.refutes_at_most(f64::MIN));
        // Positive control: the same number as a derived bound does certify.
        let bound = Status::uniform_bound(mean, 0.0, "[0,1]^C").expect("a finite bound");
        assert!(bound.certifies_at_most(mean));
        assert!(!bound.certifies_at_most(mean.next_down()));
    }

    #[test]
    fn an_exact_value_brackets_itself_by_its_numerical_error_and_an_exact_integer_is_not_widened() {
        let code = Status::exact(
            5.0,
            0.0,
            ExactBasis::Exhaustive { cardinality: 32 },
            Some(vec![1, 2]),
            "subsets of five components",
        )
        .expect("an exhaustive exact value");
        assert_eq!(code.upper_bound(), Some(5.0));
        assert_eq!(code.lower_bound(), Some(5.0));
        assert!(code.certifies_at_most(5.0));
        assert!(!code.certifies_at_most(4.0));
        assert!(code.refutes_at_most(4.0));
        assert!(!code.refutes_at_most(5.0));
        assert_eq!(code.gap(), None);

        let numerical_error = 1e-15;
        let evaluated = Status::exact(1.0, numerical_error, ExactBasis::Algebraic, None, "all u")
            .expect("an algebraic exact value");
        let upper = evaluated.upper_bound().expect("an exact value bounds itself");
        let lower = evaluated.lower_bound().expect("an exact value bounds itself");
        assert!(upper > 1.0 + numerical_error);
        assert!(lower < 1.0 - numerical_error);
        assert!(!evaluated.certifies_at_most(1.0));
    }

    #[test]
    fn constructors_refuse_inputs_that_do_not_support_the_status() {
        assert!(matches!(
            Status::uniform_bound(f64::NAN, 0.0, "r"),
            Err(EvidenceStatusError::NonFinite { field: "upper", .. })
        ));
        assert!(matches!(
            Status::uniform_bound(f64::INFINITY, 0.0, "r"),
            Err(EvidenceStatusError::NonFinite { field: "upper", .. })
        ));
        assert!(matches!(
            Status::uniform_bound(1.0, -1e-16, "r"),
            Err(EvidenceStatusError::Negative {
                field: "numerical_error",
                ..
            })
        ));
        assert!(matches!(
            Status::exact(1.0, 0.0, ExactBasis::Exhaustive { cardinality: 0 }, None, "empty"),
            Err(EvidenceStatusError::EmptyExhaustiveFamily)
        ));
        assert!(matches!(
            Status::statistical_estimate(0.5, -0.1, 10, "law"),
            Err(EvidenceStatusError::Negative {
                field: "standard_error",
                ..
            })
        ));
        assert!(matches!(
            Status::statistical_estimate(0.5, 0.1, 0, "law"),
            Err(EvidenceStatusError::NoSamples)
        ));
        // Positive controls: the same constructors accept supported inputs.
        assert!(Status::uniform_bound(1.0, 0.0, "r").is_ok());
        assert!(
            Status::exact(1.0, 0.0, ExactBasis::Exhaustive { cardinality: 1 }, None, "one").is_ok()
        );
        assert!(Status::statistical_estimate(0.5, 0.1, 10, "law").is_ok());
    }

    #[test]
    fn a_violation_within_its_numerical_error_is_not_a_counterexample() {
        let threshold = 0.25;
        let numerical_error = 1e-12;
        assert!(matches!(
            Status::counterexample(threshold + numerical_error, numerical_error, threshold, vec![0, 1]),
            Err(EvidenceStatusError::NotAViolation { .. })
        ));
        assert!(matches!(
            Status::counterexample(threshold, 0.0, threshold, vec![0, 1]),
            Err(EvidenceStatusError::NotAViolation { .. })
        ));
        // Positive control: a violation beyond the numerical error is accepted.
        let found = Status::counterexample(
            threshold + 4.0 * numerical_error,
            numerical_error,
            threshold,
            vec![0, 1],
        )
        .expect("a violation beyond roundoff");
        assert!(found.refutes_at_most(threshold));
        assert_eq!(found.upper_bound(), None);
        assert!(!found.certifies_at_most(f64::MAX));
    }

    #[test]
    fn an_unresolved_status_reports_its_gap_and_refuses_an_inverted_or_unattained_interval() {
        let witnessed = Status::unresolved(
            0.3,
            f64::INFINITY,
            Extremum::Supremum,
            Some(vec![1, 0, 1]),
            "[0,1]^3",
        )
        .expect("a lower witness with no derived upper bound");
        assert_eq!(witnessed.gap(), Some(f64::INFINITY));
        assert_eq!(witnessed.upper_bound(), None);
        assert_eq!(witnessed.lower_bound(), Some(0.3));
        assert!(!witnessed.certifies_at_most(f64::MAX));

        let bracketed =
            Status::unresolved(2.0, 3.0, Extremum::Infimum, Some(vec![0, 2, 4]), "supports")
                .expect("a relaxation bound below an incumbent");
        assert_eq!(bracketed.gap(), Some(1.0));
        assert!(bracketed.certifies_at_most(3.0));
        assert!(!bracketed.certifies_at_most(2.5));
        assert!(bracketed.refutes_at_most(1.5));
        assert!(!bracketed.refutes_at_most(2.0));

        assert!(matches!(
            Status::unresolved(3.0, 2.0, Extremum::Supremum, None, "d"),
            Err(EvidenceStatusError::InvertedInterval { .. })
        ));
        assert!(matches!(
            Status::unresolved(f64::NEG_INFINITY, 3.0, Extremum::Supremum, Some(vec![1]), "d"),
            Err(EvidenceStatusError::UnattainedWitness {
                extremum: Extremum::Supremum
            })
        ));
        assert!(matches!(
            Status::unresolved(2.0, f64::INFINITY, Extremum::Infimum, Some(vec![1]), "d"),
            Err(EvidenceStatusError::UnattainedWitness {
                extremum: Extremum::Infimum
            })
        ));
        assert!(matches!(
            Status::unresolved(f64::INFINITY, f64::INFINITY, Extremum::Supremum, None, "d"),
            Err(EvidenceStatusError::InvalidBound { field: "lower", .. })
        ));
        assert!(matches!(
            Status::unresolved(f64::NAN, 1.0, Extremum::Supremum, None, "d"),
            Err(EvidenceStatusError::InvalidBound { field: "lower", .. })
        ));
        // Positive control: an interval with both sides underived and no witness is
        // a valid, uninformative status.
        let unknown = Status::unresolved(f64::NEG_INFINITY, f64::INFINITY, Extremum::Supremum, None, "d")
            .expect("an uninformative interval");
        assert_eq!(unknown.gap(), Some(f64::INFINITY));
        assert!(!unknown.certifies_at_most(f64::MAX));
        assert!(!unknown.refutes_at_most(f64::MIN));
    }

    #[test]
    fn a_status_hands_back_its_witness_and_domain_and_a_counterexample_states_no_domain() {
        let exact = Status::exact(1.0, 0.0, ExactBasis::Algebraic, Some(vec![3]), "all u")
            .expect("an exact value");
        assert_eq!(exact.witness(), Some(&vec![3]));
        assert_eq!(exact.domain(), Some(&"all u"));
        assert_eq!(exact.into_witness(), Some(vec![3]));
        let found = Status::counterexample(1.0, 0.0, 0.5, vec![4]).expect("a violation");
        assert_eq!(found.domain(), None);
        assert_eq!(found.into_witness(), Some(vec![4]));
        let bound = Status::uniform_bound(1.0, 0.0, "r").expect("a finite bound");
        assert_eq!(bound.witness(), None);
        assert_eq!(bound.domain(), Some(&"r"));
    }

    #[test]
    fn strength_is_containment_of_the_proven_interval_on_one_domain() {
        // mpd-verify's two counterexamples to ordering by kind. An exact value with a
        // wide numerical error proves less about `<= 0.45` than a uniform bound at
        // 0.45, and that bound proves less than the bracket [0.3, 0.4].
        let exact = Status::exact(0.5, 0.1, ExactBasis::Algebraic, None, "d").expect("exact");
        let bound = Status::uniform_bound(0.45, 0.0, "d").expect("a bound");
        let bracket =
            Status::unresolved(0.3, 0.4, Extremum::Supremum, None, "d").expect("an interval");
        assert!(bound.certifies_at_most(0.45) && !exact.certifies_at_most(0.45));
        assert_eq!(exact.compare_strength(&bound), None);
        assert_eq!(bound.compare_strength(&exact), None);
        assert!(bracket.certifies_at_most(0.4) && !bound.certifies_at_most(0.4));
        assert!(bracket.refutes_at_most(0.29) && !bound.refutes_at_most(0.29));
        assert_eq!(bound.compare_strength(&bracket), Some(Ordering::Less));
        assert_eq!(bracket.compare_strength(&bound), Some(Ordering::Greater));
        // Positive controls: a tight exact value inside the bracket is stronger, and a
        // status equals itself.
        let tight = Status::exact(0.35, 0.0, ExactBasis::Algebraic, None, "d").expect("exact");
        assert_eq!(tight.compare_strength(&bracket), Some(Ordering::Greater));
        assert_eq!(bracket.compare_strength(&tight), Some(Ordering::Less));
        assert_eq!(bracket.compare_strength(&bracket), Some(Ordering::Equal));
        // Another domain, another extremum, an estimate and a counterexample are
        // incomparable.
        let elsewhere = Status::uniform_bound(0.45, 0.0, "other").expect("a bound");
        assert_eq!(bound.compare_strength(&elsewhere), None);
        let infimum =
            Status::unresolved(0.3, 0.4, Extremum::Infimum, None, "d").expect("an interval");
        assert_eq!(bracket.compare_strength(&infimum), None);
        let estimate = Status::statistical_estimate(0.35, 0.01, 10, "d").expect("an estimate");
        assert_eq!(estimate.compare_strength(&estimate), None);
        assert_eq!(bracket.compare_strength(&estimate), None);
        let found = Status::counterexample(1.0, 0.0, 0.5, vec![0]).expect("a violation");
        assert_eq!(found.compare_strength(&found), None);
    }
}
