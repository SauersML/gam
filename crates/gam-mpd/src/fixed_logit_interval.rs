//! Analytic scalar enclosures on exact binary64 inputs, without vendor exp/log.
//!
//! These are a host pilot. They neither change acceptance nor certify input
//! logits, GEMM, normalizations or neural forwards. Basic interval arithmetic
//! assumes IEEE round-to-nearest with gradual underflow. An independent exact
//! rational verifier checks that premise on the exercised operations and
//! certifies the adjacent-f64 ln(2) bracket used in range reduction.
use gam_math::score_opt::{ClosedInterval, certified_ln_positive};

pub use gam_math::score_opt::ClosedInterval as Interval;

/// Failure is an unknown enclosure, never an infeasible candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    NonFiniteInput,
    NonPositiveLogInput,
    ReductionNotEnclosed,
    UnboundedBinary64Endpoints,
    EmptyOrMismatchedRows,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Enclosure {
    Bounded(Interval),
    Unresolved(Reason),
}

/// Degree fixes work and enclosure width, not a convergence or acceptance rule.
pub const EXP_DEGREE: usize = 18;

/// The exact rational atanh-series certificate is checked in the tests. No
/// decimal parsing or platform logarithm supplies these two adjacent endpoints.
pub const fn ln_two() -> Interval {
    Interval::new(
        f64::from_bits(0x3fe6_2e42_fefa_39ef),
        f64::from_bits(0x3fe6_2e42_fefa_39f0),
    )
}

fn bounded(value: Interval) -> Enclosure {
    if value.lo.is_finite() && value.hi.is_finite() && value.lo <= value.hi {
        Enclosure::Bounded(value)
    } else {
        Enclosure::Unresolved(Reason::UnboundedBinary64Endpoints)
    }
}

fn power_two(exponent: i32) -> Option<f64> {
    match exponent {
        -1074..=-1023 => Some(f64::from_bits(1_u64 << (exponent + 1074))),
        -1022..=1023 => Some(f64::from_bits(((exponent + 1023) as u64) << 52)),
        // Only internally chosen, explicitly clamped integer exponents enter.
        _ => None,
    }
}

/// Enclose exp(x) for a fixed finite scalar. Inputs are never domain-clamped.
/// Extreme positive values may need unrepresentable upper endpoints, in which
/// case the result is explicitly Unresolved. Far negative values have an
/// absolute [0,min_subnormal] enclosure only after proving the cutoff.
pub fn exp(value: f64) -> Enclosure {
    if !value.is_finite() {
        return Enclosure::Unresolved(Reason::NonFiniteInput);
    }
    if value == 0.0 {
        return Enclosure::Bounded(Interval::point(1.0));
    }
    let cutoff = ln_two().scale(-1074.0);
    if value <= cutoff.lo {
        // value <= lower(-1074 ln2) <= -1074 ln2; hence exp(value)
        // <= 2^-1074. This includes the lost subnormal tail, not relative error.
        return Enclosure::Bounded(Interval::new(0.0, f64::from_bits(1)));
    }
    // This approximate quotient selects an identity, not an approximation to
    // the function. The original value is retained in the checked remainder.
    let exponent = (value / f64::from_bits(0x3fe6_2e42_fefa_39ef))
        .round()
        .clamp(-1074.0, 1023.0) as i32;
    let remainder = Interval::point(value).sub(ln_two().scale(exponent as f64));
    let radius = remainder.lo.abs().max(remainder.hi.abs());
    if !radius.is_finite() || radius > 1.0 {
        return Enclosure::Unresolved(Reason::ReductionNotEnclosed);
    }
    let mut term = Interval::point(1.0);
    let mut sum = term;
    for degree in 1..=EXP_DEGREE {
        term = term
            .mul(remainder)
            .div_positive(Interval::point(degree as f64));
        sum = sum.add(term);
    }
    // Beyond degree N, term ratios are <= radius/(N+2). Bound the
    // first omitted absolute term and every subsequent term separately.
    // Each multiply/divide/subtract is itself an outward interval operation.
    let next_term = Interval::point(term.lo.abs().max(term.hi.abs()))
        .mul(Interval::point(radius))
        .div_positive(Interval::point((EXP_DEGREE + 1) as f64));
    let ratio = Interval::point(radius).div_positive(Interval::point((EXP_DEGREE + 2) as f64));
    let denominator = Interval::point(1.0).sub(ratio);
    if denominator.lo <= 0.0 {
        return Enclosure::Unresolved(Reason::ReductionNotEnclosed);
    }
    let tail = next_term.div_positive(denominator).hi;
    let polynomial = Interval::new(
        Interval::point(sum.lo).sub(Interval::point(tail)).lo,
        Interval::point(sum.hi).add(Interval::point(tail)).hi,
    );
    let Some(factor) = power_two(exponent) else {
        return Enclosure::Unresolved(Reason::ReductionNotEnclosed);
    };
    let result = polynomial.mul(ClosedInterval::point(factor));
    // exp is strictly positive; intersecting its enclosure with [0,infinity]
    // is a mathematical range restriction, not changing the input argument.
    bounded(Interval::new(result.lo.max(0.0), result.hi))
}

/// Positive finite logarithm using the existing bit-decomposed atanh enclosure.
/// The implementation's geometric remainder uses no vendor log accuracy.
pub fn log(value: f64) -> Enclosure {
    if !value.is_finite() {
        return Enclosure::Unresolved(Reason::NonFiniteInput);
    }
    if value <= 0.0 {
        return Enclosure::Unresolved(Reason::NonPositiveLogInput);
    }
    match certified_ln_positive(value) {
        Some(interval) => bounded(interval),
        None => Enclosure::Unresolved(Reason::ReductionNotEnclosed),
    }
}

fn interval_of(value: Enclosure) -> Result<Interval, Reason> {
    match value {
        Enclosure::Bounded(interval) => Ok(interval),
        Enclosure::Unresolved(reason) => Err(reason),
    }
}

fn log_interval(value: Interval) -> Result<Interval, Reason> {
    let lo = interval_of(log(value.lo))?;
    let hi = interval_of(log(value.hi))?;
    Ok(Interval::new(lo.lo, hi.hi))
}

/// The exact real shift is nonpositive because max is one of the same finite
/// inputs. An unrepresentable negative lower endpoint can safely use exp(-inf)
/// =0 as its limiting lower bound; its finite upper endpoint is still checked.
fn shifted_exp(shift: Interval) -> Result<Interval, Reason> {
    let lo = if shift.lo == f64::NEG_INFINITY {
        0.0
    } else {
        interval_of(exp(shift.lo))?.lo
    };
    let hi = interval_of(exp(shift.hi.min(0.0)))?.hi.min(1.0);
    Ok(Interval::new(lo, hi))
}

/// Interval for KL(softmax(z)||softmax(w)) on the exact real values represented
/// by finite binary64 logits. O(1) auxiliary memory, two exponential enclosures
/// per class. This optional API is not wired to RunCheck or acceptance.
///
/// With a=z-max(z), b=w-max(w), Sz=sum exp(a), Sw=sum exp(b), and
/// U=sum exp(a)*(a-b), the exact identity is KL=U/Sz+ln(Sw)-ln(Sz).
/// Subtractions, products, sums, division, transcendental remainders and final
/// arithmetic are enclosed. Inputs already rounded by a head/network are
/// treated as fixed scalars; upstream rounding remains outside the claim.
pub fn kl_logits(z: &[f64], w: &[f64]) -> Enclosure {
    if z.is_empty() || z.len() != w.len() {
        return Enclosure::Unresolved(Reason::EmptyOrMismatchedRows);
    }
    if z.iter().chain(w.iter()).any(|x| !x.is_finite()) {
        return Enclosure::Unresolved(Reason::NonFiniteInput);
    }
    // Prove any exact common additive shift, including identity, before
    // interval dependency can inflate an exactly zero KL. Point intervals here
    // require exact subtraction under the existing IEEE basic-operation model.
    let offset = Interval::point(z[0]).sub(Interval::point(w[0]));
    if offset.lo == offset.hi
        && offset.lo.is_finite()
        && z.iter()
            .zip(w)
            .all(|(p, q)| Interval::point(*p).sub(Interval::point(*q)) == offset)
    {
        return Enclosure::Bounded(Interval::point(0.0));
    }
    let mz = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mw = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let compute = || -> Result<Interval, Reason> {
        let mut sz = Interval::point(0.0);
        let mut sw = Interval::point(0.0);
        let mut weighted = Interval::point(0.0);
        for (p, q) in z.iter().zip(w) {
            let a = Interval::point(*p).sub(Interval::point(mz));
            let b = Interval::point(*q).sub(Interval::point(mw));
            let ep = shifted_exp(a)?;
            let eq = shifted_exp(b)?;
            sz = sz.add(ep);
            sw = sw.add(eq);
            weighted = weighted.add(ep.mul(a.sub(b)));
        }
        // Each sum has an exactly unit maximum term and nonnegative others.
        sz.lo = sz.lo.max(1.0);
        sw.lo = sw.lo.max(1.0);
        let result = weighted
            .div_positive(sz)
            .add(log_interval(sw)?)
            .sub(log_interval(sz)?);
        // Mathematical nonnegativity, not a heuristic KL clamp. If the upper
        // endpoint contradicts it, bounded() refuses rather than fabricating 0.
        Ok(Interval::new(result.lo.max(0.0), result.hi))
    };
    match compute() {
        Ok(value) => bounded(value),
        Err(reason) => Enclosure::Unresolved(reason),
    }
}

#[cfg(test)]
#[path = "fixed_logit_interval_tests.rs"]
mod tests;
