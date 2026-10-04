//! A diagnostic obstruction for explanatory programs that identify two episodes.
//!
//! If the program predicts the same distribution r for native responses p and q,
//! Pinsker and the triangle inequality imply
//! (KL(p||r)+KL(q||r))/2 >= TV(p,q)^2/2 (natural nats).
//! Averaging matched rows with identical weights therefore lower-bounds the
//! larger of the two episode-group mean KLs by mean(TV^2/2). This is conditional
//! on identical predictions, not merely similar hidden states or missing names.
//!
//! Only softmaxes of the supplied fixed binary64 logits are enclosed. Native
//! forwards, readout rounding, and the existing acceptance metric are outside
//! this diagnostic. Arithmetic assumptions are those of fixed_logit_interval.
use crate::fixed_logit_interval::{Enclosure, Interval, Reason, exp};

fn interval(value: Enclosure) -> Result<Interval, Reason> {
    match value {
        Enclosure::Bounded(value) => Ok(value),
        Enclosure::Unresolved(reason) => Err(reason),
    }
}

fn exponential(value: f64, maximum: f64) -> Result<Interval, Reason> {
    let shift = Interval::point(value).sub(Interval::point(maximum));
    // The real shift is nonpositive. Its lower endpoint can overflow negative;
    // exp(-infinity)=0 remains a valid limiting lower bound.
    let lo = if shift.lo == f64::NEG_INFINITY { 0.0 } else { interval(exp(shift.lo))?.lo };
    let hi = interval(exp(shift.hi.min(0.0)))?.hi.min(1.0);
    Ok(Interval::new(lo, hi))
}

fn absolute(value: Interval) -> Interval {
    if value.lo >= 0.0 { value }
    else if value.hi <= 0.0 { value.neg() }
    else { Interval::new(0.0, (-value.lo).max(value.hi)) }
}

/// TV between two fixed-logit softmaxes. O(1) auxiliary memory; two passes.
/// Failure stays unresolved. It never establishes a positive obstruction.
pub fn total_variation(z: &[f64], w: &[f64]) -> Enclosure {
    if z.is_empty() || z.len() != w.len() {
        return Enclosure::Unresolved(Reason::EmptyOrMismatchedRows);
    }
    if z.iter().chain(w).any(|x| !x.is_finite()) {
        return Enclosure::Unresolved(Reason::NonFiniteInput);
    }
    let offset = Interval::point(z[0]).sub(Interval::point(w[0]));
    if offset.lo == offset.hi && offset.lo.is_finite()
        && z.iter().zip(w).all(|(p,q)| Interval::point(*p).sub(Interval::point(*q)) == offset) {
        return Enclosure::Bounded(Interval::point(0.0));
    }
    let mz = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mw = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let compute = || -> Result<Interval, Reason> {
        let mut sz = Interval::point(0.0);
        let mut sw = Interval::point(0.0);
        for (&p, &q) in z.iter().zip(w) {
            sz = sz.add(exponential(p, mz)?);
            sw = sw.add(exponential(q, mw)?);
        }
        // Each normalization includes its exactly unit maximum term.
        sz.lo = sz.lo.max(1.0);
        sw.lo = sw.lo.max(1.0);
        let mut sum = Interval::point(0.0);
        for (&p, &q) in z.iter().zip(w) {
            let pp = exponential(p, mz)?.div_positive(sz);
            let qq = exponential(q, mw)?.div_positive(sw);
            sum = sum.add(absolute(pp.sub(qq)));
        }
        let tv = sum.scale(0.5);
        Ok(Interval::new(tv.lo.max(0.0), tv.hi.min(1.0)))
    };
    match compute() {
        Ok(value) if value.lo.is_finite() && value.hi.is_finite() && value.lo <= value.hi => Enclosure::Bounded(value),
        Ok(_) => Enclosure::Unresolved(Reason::UnboundedBinary64Endpoints),
        Err(reason) => Enclosure::Unresolved(reason),
    }
}

/// Enclose the obstruction statistic TV(p,q)^2/2. Only its LOWER endpoint
/// lower-bounds the best achievable KL; the upper endpoint is not an upper
/// bound on achievable KL. A pair of distinct groups must use matching weights.
pub fn shared_prediction_obstruction(z: &[f64], w: &[f64]) -> Enclosure {
    match total_variation(z, w) {
        Enclosure::Bounded(tv) => {
            let value = tv.mul(tv).scale(0.5);
            Enclosure::Bounded(Interval::new(value.lo.max(0.0), value.hi.min(0.5)))
        }
        unknown => unknown,
    }
}

#[cfg(test)]
#[path = "response_collision_tests.rs"]
mod tests;
