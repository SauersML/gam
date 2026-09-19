/// A validated, finite, ordered ρ (log-λ) seed interval `[lo, hi]`.
///
/// Every seed clamp in the outer-optimizer prepass and the candidate lattice
/// derives a trial ρ and pins it into a single uniform box assembled from two
/// *independently-owned* constants — the outer ρ lower wall
/// (`options.rho_lower_bound`) and an over-smoothing ceiling (`RHO_BOUND` or an
/// effective-df crossing). When those constants drift apart the interval inverts
/// (`lo > hi`): the #2370 disease, where an edf-ceiling that used to equal
/// `-rho_lower_bound` was moved by #2356 and the emitted upper bound dropped
/// below the lower one. The historical response — *silently swapping* the pair
/// (`normalize_seed_bounds`) — does not make the fit correct; it makes the
/// optimizer solve a *different, silently substituted* box and return a model as
/// if nothing were wrong. That is strictly worse than the panic it replaced: a
/// panic is loud, a silently-wrong λ-box is not.
///
/// This type makes the inverted state unrepresentable. It is constructed only
/// through [`OrderedRhoBounds::new`], which refuses an inverted or non-finite
/// interval with the same typed `EstimationError::InvalidInput` the outer
/// entry (`run_outer_uncertified`) now enforces (#2379 / #2370). Every downstream
/// clamp then operates on an interval that is ordered *by construction*, so
/// `f64::clamp`'s `min <= max` precondition can never be violated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderedRhoBounds {
    lo: f64,
    hi: f64,
}

impl OrderedRhoBounds {
    /// Validate and wrap a `[lo, hi]` ρ interval. Refuses (rather than silently
    /// reorders) an inverted (`lo > hi`) or non-finite interval, naming both
    /// endpoints. `lo == hi` is a valid degenerate single-point box.
    pub fn new(lo: f64, hi: f64) -> Result<Self, crate::estimation_error::EstimationError> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(crate::estimation_error::EstimationError::InvalidInput(
                format!(
                    "seed ρ-box is inverted or non-finite: lower={lo}, upper={hi}; an \
                 inverted box means the ρ lower wall and the over-smoothing ceiling \
                 have drifted apart (cf. #2370) — refusing rather than silently \
                 reordering the interval (#2379)"
                ),
            ));
        }
        Ok(Self { lo, hi })
    }

    /// The (validated) lower endpoint.
    #[inline]
    pub fn lower(self) -> f64 {
        self.lo
    }

    /// The (validated) upper endpoint.
    #[inline]
    pub fn upper(self) -> f64 {
        self.hi
    }

    /// Clamp `value` into `[lo, hi]`. Infallible: the interval is ordered by
    /// construction, so `f64::clamp`'s `min <= max` precondition always holds.
    #[inline]
    pub fn clamp(self, value: f64) -> f64 {
        value.clamp(self.lo, self.hi)
    }

    /// The smallest interval holding every coordinate's declared domain: the
    /// minimum lower face and the maximum upper face. A seed lattice built in one
    /// scalar box then never clamps a coordinate onto a face its own domain does
    /// not have, and the caller projects each seed per coordinate afterwards
    /// (#2902 row 9). A domain with no coordinates is the supported log-strength
    /// domain, on which `exp(ρ)` is evaluated exactly.
    pub fn envelope(
        lower: impl IntoIterator<Item = f64>,
        upper: impl IntoIterator<Item = f64>,
    ) -> Result<Self, crate::estimation_error::EstimationError> {
        let lo = lower.into_iter().fold(f64::INFINITY, f64::min);
        let hi = upper.into_iter().fold(f64::NEG_INFINITY, f64::max);
        if lo == f64::INFINITY && hi == f64::NEG_INFINITY {
            return Self::new(crate::LOG_STRENGTH_MIN, crate::LOG_STRENGTH_MAX);
        }
        Self::new(lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── OrderedRhoBounds (#2379) ──────────────────────────────────────────────
    // The validated-interval type that REPLACES the silent swap on every seed
    // clamp: an inverted box must be a typed refusal, never a reordered interval.

    #[test]
    fn ordered_rho_bounds_accepts_ordered_interval() {
        let b = OrderedRhoBounds::new(-12.0, 12.0).expect("ordered interval is valid");
        assert_eq!(b.lower(), -12.0);
        assert_eq!(b.upper(), 12.0);
    }

    #[test]
    fn ordered_rho_bounds_accepts_degenerate_point_interval() {
        // lo == hi is a valid single-point box (matches opt::Bounds, which uses
        // `lower > upper` as the inversion test).
        let b = OrderedRhoBounds::new(2.0, 2.0).expect("point interval is valid");
        assert_eq!(b.clamp(5.0), 2.0);
        assert_eq!(b.clamp(-5.0), 2.0);
    }

    #[test]
    fn ordered_rho_bounds_refuses_inverted_interval_with_typed_error() {
        // This is the #2379 contract: an inverted seed-bound pair reaching the
        // seed path is a typed refusal, NOT a silently reordered box. The exact
        // scenario from #2370 — lower = -10 (the ρ lower wall) above an
        // independently-derived edf ceiling of -11.855.
        let err = OrderedRhoBounds::new(-10.0, -11.855)
            .expect_err("an inverted box must be refused, not swapped");
        match err {
            crate::estimation_error::EstimationError::InvalidInput(msg) => {
                // Both endpoints are named, so a drift is diagnosable from the error.
                assert!(msg.contains("-10"), "error names the lower bound: {msg}");
                assert!(
                    msg.contains("-11.855"),
                    "error names the upper bound: {msg}"
                );
                assert!(
                    msg.contains("invert"),
                    "error explains the inversion: {msg}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn ordered_rho_bounds_refuses_non_finite_interval() {
        assert!(OrderedRhoBounds::new(f64::NAN, 12.0).is_err());
        assert!(OrderedRhoBounds::new(-12.0, f64::INFINITY).is_err());
        assert!(OrderedRhoBounds::new(f64::NEG_INFINITY, 12.0).is_err());
    }

    #[test]
    fn ordered_rho_bounds_clamp_respects_both_ends() {
        let b = OrderedRhoBounds::new(-3.0, 5.0).unwrap();
        assert_eq!(b.clamp(1.0), 1.0);
        assert_eq!(b.clamp(-10.0), -3.0);
        assert_eq!(b.clamp(100.0), 5.0);
    }

    /// #2902 row 9: the seed box is the envelope of every coordinate's declared
    /// domain, a domain with no coordinates is the representable log-strength
    /// domain, and an envelope that inverts is still a typed refusal (#2379).
    #[test]
    fn ordered_rho_bounds_envelope_spans_every_coordinate_domain_2902() {
        let b = OrderedRhoBounds::envelope([-3.0, -7.5, 1.0], [4.0, 9.0, 2.5])
            .expect("ordered envelope");
        assert_eq!((b.lower(), b.upper()), (-7.5, 9.0));
        let empty = OrderedRhoBounds::envelope(std::iter::empty(), std::iter::empty())
            .expect("representable log-strength domain");
        assert_eq!(
            (empty.lower(), empty.upper()),
            (crate::LOG_STRENGTH_MIN, crate::LOG_STRENGTH_MAX)
        );
        assert!(OrderedRhoBounds::envelope([2.0], [1.0]).is_err());
    }
}
