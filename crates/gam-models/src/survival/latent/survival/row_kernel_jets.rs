// [#780 line-count gate] The latent survival row kernel's jet machinery: the primary-coordinate
// layout, the exact kernel expansion and its certified cumulants, the A-basis term kernels, the
// LatentRowJet backends and the primary direction maps, with their inline tests. Included from
// `mod.rs`, so it keeps that module's scope and private access.

const LATENT_SURVIVAL_PRIMARY_Q_ENTRY: usize = 0;
const LATENT_SURVIVAL_PRIMARY_Q_EXIT: usize = 1;
const LATENT_SURVIVAL_PRIMARY_QDOT_EXIT: usize = 2;
// Interval-censored right boundary R: q_right = log B(R) shares the time-block
// coefficients with q_exit (same monotone transform, different time point), so
// it is a fourth linear functional of the time block, NOT an independent eta
// channel. It sits before `mu`/`log_sigma` so the "trailing optional log_sigma"
// invariant used by `active_primary` (= `LATENT_SURVIVAL_PRIMARY_LOG_SIGMA`)
// keeps q_right always active.
const LATENT_SURVIVAL_PRIMARY_Q_RIGHT: usize = 3;
const LATENT_SURVIVAL_PRIMARY_MU: usize = 4;
const LATENT_SURVIVAL_PRIMARY_LOG_SIGMA: usize = 5;
const LATENT_SURVIVAL_PRIMARY_DIM: usize = 6;

/// Certified derivative tower of `f(x) = log(1 - exp(x))` for `x < 0`.
///
/// The value uses `log1mexp`; derivative magnitudes are assembled in log space:
///
/// ```text
/// |f'|    = r / s
/// |f''|   = r / s²
/// |f'''|  = r(1 + r) / s³
/// |f''''| = r(1 + 4r + r²) / s⁴
/// |f⁽⁵⁾|  = r(1 + 11r + 11r² + r³) / s⁵,
/// r = exp(x), s = 1 - r.
/// ```
///
/// This never forms `1/s^k`. If a true derivative magnitude cannot be
/// represented by `f64`, the routine returns a typed numerical refusal instead
/// of publishing an infinite jet. Only the derivative order consumed by the
/// selected jet backend is certified, and the stack is as long as that jet's
/// composition reads: five entries through the fourth order, six through the
/// fifth. There is no clamp or magnitude cutoff.
fn latent_unary_derivatives_log1mexp_negative<const N: usize>(
    x: f64,
    derivative_order: usize,
    context: &str,
) -> Result<[f64; N], LatentSurvivalError> {
    assert!(derivative_order < N && N <= 6);
    if !(x.is_finite() && x < 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!("{context} requires a finite negative log-boundary gap, got {x:?}"),
        });
    }
    let value = log1mexp_positive(-x);
    let exp_x = x.exp();
    let log_derivative_magnitudes = [
        x - value,
        x - 2.0 * value,
        x + exp_x.ln_1p() - 3.0 * value,
        x + (exp_x * (4.0 + exp_x)).ln_1p() - 4.0 * value,
        x + (exp_x * (11.0 + exp_x * (11.0 + exp_x))).ln_1p() - 5.0 * value,
    ];
    let mut derivatives = [0.0; N];
    derivatives[0] = value;
    for (offset, log_magnitude) in log_derivative_magnitudes
        .into_iter()
        .take(derivative_order)
        .enumerate()
    {
        let order = offset + 1;
        let magnitude = log_magnitude.exp();
        if !magnitude.is_finite() {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!(
                    "{context} derivative order {order} is not representable at \
                     log-boundary gap {x:?} (log magnitude {log_magnitude:?})"
                ),
            });
        }
        derivatives[order] = -magnitude;
    }
    Ok(derivatives)
}

/// Stable jet for `log(exp(log_left + c_left) - exp(log_right + c_right))`.
///
/// Positivity implies the log-domain gap
/// `delta = (log_right + c_right) - (log_left + c_left)` is negative, so
///
/// ```text
/// log(A - B) = log(A) + log(1 - exp(delta)).
/// ```
///
/// Absolute mass never leaves log space. The caller supplies a complete
/// finiteness predicate for its concrete jet representation so an `Ok` result
/// certifies every carried channel, including contracted third/fourth/fifth parts.
fn latent_survival_positive_log_difference_jet<const K: usize, J: LatentRowJet<K>>(
    log_left: &J,
    log_coefficient_left: f64,
    log_right: &J,
    log_coefficient_right: f64,
    derivative_order: usize,
    context: &str,
    all_channels_finite: impl Fn(&J) -> bool,
) -> Result<J, LatentSurvivalError> {
    let weighted_left = log_left.row_add(&J::row_constant(log_coefficient_left));
    let weighted_right = log_right.row_add(&J::row_constant(log_coefficient_right));
    let delta = weighted_right.row_sub(&weighted_left);
    let delta_value = delta.row_value();
    if !(delta_value.is_finite() && delta_value < 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} must be a positive survival-mass difference: \
                 log(c_L*K0(M_L))={:?}, log(c_R*K0(M_R))={:?}; \
                 require M_L < M_R (i.e. L < R)",
                weighted_left.row_value(),
                weighted_right.row_value(),
            ),
        });
    }
    let out = weighted_left
        .row_add(&delta.row_compose_log1mexp_negative(derivative_order, context)?);
    if !all_channels_finite(&out) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} derivative jet is not representable at log-boundary gap {delta_value:?}"
            ),
        });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryTerm {
    coeff: f64,
    q_exp: usize,
    qdot_power: usize,
    tau_exp: usize,
    k: usize,
}

/// One signed magnitude kept in logarithmic coordinates.
///
/// The latent-kernel recurrence naturally produces derivatives as signed
/// log-sums.  Keeping that representation through the moment-to-cumulant
/// conversion is essential: materialising `S_ab / S` and `S_a / S` separately
/// before computing `S_ab / S - (S_a / S)(S_b / S)` rounds both large moments
/// and destroys the small curvature left by their cancellation.
#[derive(Clone, Copy, Debug)]
struct LatentSignedLog {
    log_abs: f64,
    sign: f64,
}

impl LatentSignedLog {
    const ZERO: Self = Self {
        log_abs: f64::NEG_INFINITY,
        sign: 0.0,
    };
    const ONE: Self = Self {
        log_abs: 0.0,
        sign: 1.0,
    };
}

// A pointed cumulant over `r` slots consists of one leading moment and every
// proper pointed block of sizes `1..r-1`. Multiplication by the complementary
// rounded moment can at most double an exact expansion's length:
//
// C₁ = 1
// C₂ = 1 + 1·2C₁ = 3
// C₃ = 1 + 1·2C₁ + 2·2C₂ = 15
// C₄ = 1 + 1·2C₁ + 3·2C₂ + 3·2C₃ = 111.
//
// Certifying the final rounding subtracts one candidate binary64, requiring
// exactly one additional component. These are structural support bounds, not
// tunable numerical capacities.
const LATENT_EXACT_EXPANSION_ORDER1: usize = 1;
const LATENT_EXACT_EXPANSION_ORDER2: usize = 1 + 2 * LATENT_EXACT_EXPANSION_ORDER1;
const LATENT_EXACT_EXPANSION_ORDER3: usize =
    1 + 2 * LATENT_EXACT_EXPANSION_ORDER1 + 2 * 2 * LATENT_EXACT_EXPANSION_ORDER2;
const LATENT_EXACT_EXPANSION_ORDER4: usize = 1
    + 2 * LATENT_EXACT_EXPANSION_ORDER1
    + 3 * 2 * LATENT_EXACT_EXPANSION_ORDER2
    + 3 * 2 * LATENT_EXACT_EXPANSION_ORDER3;
const LATENT_EXACT_EXPANSION_CAPACITY: usize = LATENT_EXACT_EXPANSION_ORDER4 + 1;

/// The one refusal that replaces the pre-gam#2714 product refusal: not "a
/// monomial underflowed" (which is a fact about binary64 and says nothing about
/// the derivative) but "what binary64 could not carry is big enough to decide
/// the answer", which is the statement a certification is entitled to make.
const LATENT_UNREPRESENTABLE_MASS_REACHES_CELL: &str =
    "the product mass binary64 cannot represent reaches the cumulant's rounding cell";
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER1 == 1);
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER2 == 3);
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER3 == 15);
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER4 == 111);
const _: () = assert!(LATENT_EXACT_EXPANSION_CAPACITY == 112);

/// Fixed, increasing-magnitude floating-point expansion.
///
/// Each component is an exact binary64 and their real sum is the represented
/// value TO WITHIN [`Self::unrepresentable_mass`]. `TwoSum` and FMA
/// `TwoProduct` retain every arithmetic residual the format can hold, so the
/// pointed cumulant polynomial is evaluated exactly over its rounded binary64
/// moments wherever binary64 is capable of it. The sole rounding occurs in
/// [`Self::certified_round`], which proves its cell against the components AND
/// that mass.
#[derive(Clone, Copy)]
struct LatentExactExpansion {
    components: [f64; LATENT_EXACT_EXPANSION_CAPACITY],
    len: usize,
    /// Outward upper bound on `|exact value − Σ components|` (gam#2714).
    ///
    /// `TwoSum` is exact for every finite pair, and `TwoProduct` is exact
    /// whenever the product stays at or above `2^-970`, so this is `0.0` on
    /// every expansion the recurrence could already build. It becomes nonzero
    /// exactly where an FMA residual needs bits under `2^-1074` — see
    /// [`Self::two_product`] for the bound and its proof — and it is the reason
    /// a monomial binary64 cannot carry no longer refuses a derivative binary64
    /// can round perfectly well.
    unrepresentable_mass: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentCertifiedCumulant {
    /// Unique nearest binary64 rounding of the exact recurrence.
    value: f64,
    /// Outward upper bound on the sum of absolute expanded monomials.
    ///
    /// This is the numerator of the componentwise condition number and lets an
    /// independent floating implementation derive its own forward-error band
    /// without comparing against production or fitting a tolerance.
    absolute_term_mass: f64,
}

impl LatentCertifiedCumulant {
    const ZERO: Self = Self {
        value: 0.0,
        absolute_term_mass: 0.0,
    };
}

impl LatentExactExpansion {
    const ZERO: Self = Self {
        components: [0.0; LATENT_EXACT_EXPANSION_CAPACITY],
        len: 0,
        unrepresentable_mass: 0.0,
    };

    /// One subnormal ulp — the outward bound on a single FMA product residual
    /// that binary64 cannot represent (gam#2714).
    ///
    /// In the guarded band `|p| = |fl(a·b)| < 2^-970` the exact residual obeys
    /// `|r| = |a·b − p| ≤ ulp(p)/2 ≤ 2^-1023`, so `r` lies inside the subnormal
    /// range where the grid spacing is exactly `2^-1074`. `fma(a, b, −p)`
    /// returns `r` correctly rounded onto that grid, so its error is at most a
    /// half spacing, `2^-1075`. That half is not itself representable, so the
    /// bound carried here is the full spacing — strictly conservative, and the
    /// smallest positive binary64 there is. The completely-underflowed case
    /// `p = 0` is the same statement with `|a·b| ≤ 2^-1075`.
    const SUBNORMAL_ULP: f64 = f64::from_bits(1);

    /// Round a nonnegative bound outward, so an accumulated bound stays a
    /// bound under binary64 addition and multiplication.
    ///
    /// Exactly zero is returned unchanged: an expansion that has lost nothing
    /// must not acquire a bound merely by being added or scaled, or the "no
    /// underflow ⇒ byte-identical" property would not hold.
    #[inline]
    fn outward_bound(value: f64) -> Result<f64, &'static str> {
        if !value.is_finite() || value < 0.0 {
            return Err("an unrepresentable-mass bound left the finite nonnegative range");
        }
        if value == 0.0 {
            return Ok(0.0);
        }
        Ok(Self::next_up(value))
    }

    fn scalar(value: f64) -> Self {
        if value == 0.0 {
            Self::ZERO
        } else {
            let mut out = Self::ZERO;
            out.components[0] = value;
            out.len = 1;
            out
        }
    }

    #[inline]
    fn component(&self, index: usize) -> f64 {
        self.components[index]
    }

    fn push_nonzero(&mut self, value: f64) -> Result<(), &'static str> {
        if value == 0.0 {
            return Ok(());
        }
        if !value.is_finite() {
            return Err("an exact-expansion component became non-finite");
        }
        if self.len == LATENT_EXACT_EXPANSION_CAPACITY {
            return Err("the structurally bounded exact expansion exhausted its capacity");
        }
        self.components[self.len] = value;
        self.len += 1;
        Ok(())
    }

    /// Error-free `a + b = sum + error`.
    #[inline]
    fn two_sum(a: f64, b: f64) -> Result<(f64, f64), &'static str> {
        let sum = a + b;
        if !sum.is_finite() {
            return Err("an exact-expansion addition overflowed");
        }
        let virtual_b = sum - a;
        let virtual_a = sum - virtual_b;
        let error = (a - virtual_a) + (b - virtual_b);
        Ok((sum, error))
    }

    /// Finite product through one fused residual, with an outward bound on the
    /// part of the exact product binary64 cannot carry.
    ///
    /// An FMA product residual is EXACT provided neither the product nor its
    /// residual underflows: a product at least `2^-970` has an exact-product
    /// least-significant bit no smaller than `2^-1074` for binary64 operands,
    /// so `mul_add(a, b, -product)` IS `a·b − product` with no rounding, and
    /// the returned bound is `0.0`.
    ///
    /// Below that proved range the residual can need bits the format does not
    /// have. **That is a fact about binary64, not about the derivative**
    /// (gam#2714). This used to `Err` there, which threw away an entire
    /// certified cumulant over one monomial — and the monomials that land there
    /// are, by construction, the ones too small to reach the answer: two
    /// perfectly ordinary normal moments at `1e-152` multiply to `1e-304`, 252
    /// orders below the half-ulp of a leading moment of `1`. Measured on the
    /// #2714 witness, five of seven outer seeds of the veteran latent-frailty
    /// fit died on exactly that. It is the same shape as the tie-to-even
    /// refusal fixed for gam#2538 in [`Self::certified_round`]: the
    /// certification refusing precisely the inputs it can decide.
    ///
    /// So the residual is KEPT — it is still the correctly rounded value of
    /// `a·b − product`, which is the best binary64 has — and the mass it may be
    /// off by is carried outward in [`Self::unrepresentable_mass`] and proved
    /// against the rounding cell at the end. See [`Self::SUBNORMAL_ULP`] for
    /// the bound. Nothing is loosened: an expansion whose products all clear
    /// `2^-970` carries a bound of exactly zero and certifies bit-identically.
    #[inline]
    fn two_product(a: f64, b: f64) -> Result<(f64, f64, f64), &'static str> {
        if a == 0.0 || b == 0.0 {
            return Ok((0.0, 0.0, 0.0));
        }
        let product = a * b;
        if !product.is_finite() {
            return Err("an exact-expansion product overflowed");
        }
        let error = a.mul_add(b, -product);
        if !error.is_finite() {
            return Err("an exact-expansion product residual became non-finite");
        }
        let unrepresentable = if product.abs() < f64::MIN_POSITIVE / f64::EPSILON {
            Self::SUBNORMAL_ULP
        } else {
            0.0
        };
        Ok((product, error, unrepresentable))
    }

    /// Error-free sum of two increasing-magnitude expansions.
    ///
    /// `TwoSum` is exact for every finite pair, so the sum adds no
    /// unrepresentable mass of its own; the two operands' bounds simply add.
    fn add(self, other: Self) -> Result<Self, &'static str> {
        let carried = Self::outward_bound(self.unrepresentable_mass + other.unrepresentable_mass)?;
        if self.len == 0 {
            return Ok(Self {
                unrepresentable_mass: carried,
                ..other
            });
        }
        if other.len == 0 {
            return Ok(Self {
                unrepresentable_mass: carried,
                ..self
            });
        }
        if self.len + other.len > LATENT_EXACT_EXPANSION_CAPACITY {
            return Err("an exact-expansion sum exceeded its structural support bound");
        }

        let mut left = 0usize;
        let mut right = 0usize;
        let take_left = |left_value: f64, right_value: f64| {
            left_value.abs() <= right_value.abs()
        };
        let mut q = if take_left(self.component(left), other.component(right)) {
            let value = self.component(left);
            left += 1;
            value
        } else {
            let value = other.component(right);
            right += 1;
            value
        };
        let mut out = Self::ZERO;

        while left < self.len || right < other.len {
            let next = if right == other.len
                || (left < self.len
                    && take_left(self.component(left), other.component(right)))
            {
                let value = self.component(left);
                left += 1;
                value
            } else {
                let value = other.component(right);
                right += 1;
                value
            };
            let (sum, error) = Self::two_sum(q, next)?;
            out.push_nonzero(error)?;
            q = sum;
        }
        out.push_nonzero(q)?;
        out.unrepresentable_mass = carried;
        Ok(out)
    }

    /// Multiplication by one rounded binary64 moment, exact wherever binary64
    /// can be (see [`Self::two_product`]).
    ///
    /// The incoming bound scales with the multiplier — `|exact − Σc| ≤ m` gives
    /// `|s·exact − Σ(s·c)| ≤ |s|·m` — and every product residual the format
    /// cannot hold adds its own [`Self::SUBNORMAL_ULP`] on top.
    fn scale(self, scalar: f64) -> Result<Self, &'static str> {
        if self.len == 0 || scalar == 0.0 {
            return Ok(Self::ZERO);
        }
        if self.len * 2 > LATENT_EXACT_EXPANSION_CAPACITY {
            return Err("a scaled exact expansion exceeded its structural support bound");
        }

        let mut unrepresentable =
            Self::outward_bound(self.unrepresentable_mass * scalar.abs())?;
        let (mut q, first_error, first_unrepresentable) =
            Self::two_product(self.component(0), scalar)?;
        unrepresentable = Self::outward_bound(unrepresentable + first_unrepresentable)?;
        let mut out = Self::ZERO;
        out.push_nonzero(first_error)?;
        for index in 1..self.len {
            let (product, product_error, product_unrepresentable) =
                Self::two_product(self.component(index), scalar)?;
            unrepresentable = Self::outward_bound(unrepresentable + product_unrepresentable)?;
            let (sum, sum_error) = Self::two_sum(q, product_error)?;
            out.push_nonzero(sum_error)?;
            // The exact scaling recurrence guarantees `|product| >= |sum|`;
            // generic TwoSum keeps the identity valid without relying on that
            // ordering as a runtime premise.
            let (next_q, product_sum_error) = Self::two_sum(product, sum)?;
            out.push_nonzero(product_sum_error)?;
            q = next_q;
        }
        out.push_nonzero(q)?;
        out.unrepresentable_mass = unrepresentable;
        Ok(out)
    }

    #[inline]
    fn next_up(value: f64) -> f64 {
        if value.is_nan() || value == f64::INFINITY {
            value
        } else if value == 0.0 {
            f64::from_bits(1)
        } else if value > 0.0 {
            f64::from_bits(value.to_bits() + 1)
        } else {
            f64::from_bits(value.to_bits() - 1)
        }
    }

    #[inline]
    fn next_down(value: f64) -> f64 {
        -Self::next_up(-value)
    }

    /// Sign of the exact expansion after adding one binary64 scalar.
    ///
    /// Every finite binary64 is an integer multiple of 2^-1074. Accumulating
    /// positive and negative significands into separate fixed unsigned integers
    /// and comparing those integers is therefore an exact, order-independent
    /// sign decision. It does not assume that a preceding error-free transform
    /// happened to retain a particular nonoverlap strength.
    fn exact_sign_after_adding_scalar(self, scalar: f64) -> Result<f64, &'static str> {
        self.exact_sign_after_adding_scalars(scalar, 0.0)
    }

    /// As [`Self::exact_sign_after_adding_scalar`], with two addends.
    ///
    /// The second is the unrepresentable-mass offset that turns a statement
    /// about `Σ components` into a statement about the whole interval the exact
    /// value is known to lie in (gam#2714). Both go through the same exact
    /// fixed-point accumulator, so nothing about the decision becomes a
    /// tolerance comparison.
    fn exact_sign_after_adding_scalars(
        self,
        first: f64,
        second: f64,
    ) -> Result<f64, &'static str> {
        let ordering = exact_binary64_sum_sign(
            self.components[..self.len]
                .iter()
                .copied()
                .chain([first, second]),
        )
        .map_err(|_| "the shared exact-sign accumulator rejected its structural finite input")?;
        Ok(match ordering {
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
            std::cmp::Ordering::Greater => 1.0,
        })
    }

    /// Half the distance from `value` to whichever of its two binary64
    /// neighbours is nearer — the radius of its rounding cell.
    #[inline]
    fn rounding_cell_radius(value: f64) -> f64 {
        let up = Self::next_up(value) - value;
        let down = value - Self::next_down(value);
        0.5 * up.min(down)
    }

    /// `candidate` is the answer iff the whole interval the exact value is
    /// known to lie in — `Σ components ± mass` — sits inside its rounding cell.
    ///
    /// Used where the components sum to `candidate` EXACTLY, so the only thing
    /// separating the two is the mass binary64 could not carry (gam#2714).
    fn certify_exact_sum_against_mass(candidate: f64, mass: f64) -> Result<f64, &'static str> {
        if mass == 0.0 {
            return Ok(candidate);
        }
        if mass < Self::rounding_cell_radius(candidate) {
            return Ok(candidate);
        }
        Err(LATENT_UNREPRESENTABLE_MASS_REACHES_CELL)
    }

    /// Unique nearest-even binary64 rounding, proved against both adjacent
    /// midpoint boundaries and against the mass binary64 could not carry.
    ///
    /// The starting point is the accumulated sum of the components, which is
    /// only a neighbourhood of the exact value. Where it is not the nearest
    /// binary64 the boundary comparison says so exactly, and the walk below
    /// steps one ulp onto the neighbour that comparison names, so the returned
    /// value is the correctly rounded one regardless of how the accumulation
    /// happened to round.
    ///
    /// gam#2714 adds the second half of the proof. `Σ components` is the exact
    /// value only to within [`Self::unrepresentable_mass`], so every boundary
    /// comparison is made at the END of that interval that is nearest the
    /// boundary. `mass == 0` — every expansion whose products all cleared
    /// `2^-970`, which is every expansion this routine could previously be
    /// handed at all — reduces each comparison to the one made before, so the
    /// returned value is bit-identical there.
    fn certified_round(self) -> Result<f64, &'static str> {
        let mass = self.unrepresentable_mass;
        if !(mass.is_finite() && mass >= 0.0) {
            return Err("an unrepresentable-mass bound left the finite nonnegative range");
        }
        if self.len == 0 {
            return Self::certify_exact_sum_against_mass(0.0, mass);
        }
        let mut candidate = 0.0_f64;
        for index in 0..self.len {
            candidate += self.component(index);
        }
        if !candidate.is_finite() {
            return Err("the exact cumulant is outside the finite binary64 range");
        }

        // The accumulation above rounds once per component, so `candidate` is a
        // NEIGHBOURHOOD of the exact value, not provably its nearest binary64.
        // When it is not, the certification below can say so exactly -- the
        // residual expansion and the sign accumulator are both exact -- and the
        // honest response is to MOVE to the neighbour it names rather than to
        // refuse. A step is taken only when the exact value lies strictly
        // beyond the midpoint separating `candidate` from `adjacent`, which is
        // precisely the statement that `adjacent` is strictly nearer, so the
        // walk contracts by one ulp per step, never reverses direction, and is
        // bounded by the (finite) ulp distance to the exact value. The step cap
        // is a runaway backstop, not the decision: it is the same structural
        // support bound that bounds the number of roundings the accumulation
        // above could have committed, since each component contributes at most
        // one.
        let mut walked = 0usize;
        loop {
            let residual = self.add(Self::scalar(-candidate))?;
            if residual.len == 0 {
                return Self::certify_exact_sum_against_mass(candidate, mass);
            }
            let residual_sign = residual.exact_sign_after_adding_scalar(0.0)?;
            if residual_sign == 0.0 {
                // Nonzero components summing to exactly zero: `candidate` IS the
                // components' sum, and only the unrepresentable mass separates
                // it from the exact value.
                return Self::certify_exact_sum_against_mass(candidate, mass);
            }
            let adjacent = if residual_sign > 0.0 {
                Self::next_up(candidate)
            } else {
                Self::next_down(candidate)
            };
            if !adjacent.is_finite() {
                return Err("the exact cumulant is outside the finite binary64 range");
            }
            let midpoint_distance = 0.5 * (adjacent - candidate).abs();
            if midpoint_distance == 0.0 {
                return Err("the exact cumulant reached an unrepresentable subnormal midpoint");
            }
            let boundary_offset = -residual_sign * midpoint_distance;
            // The components' sum is inside `candidate`'s cell; the exact value
            // is inside it too only if the mass binary64 could not carry fits
            // in what is left (gam#2714). `mass == 0` makes the widened
            // comparison the same exact-sign call as the plain one.
            let cell_holds_the_mass = move |residual: Self| -> Result<f64, &'static str> {
                let widened = residual
                    .exact_sign_after_adding_scalars(boundary_offset, residual_sign * mass)?;
                if widened == -residual_sign {
                    Ok(candidate)
                } else {
                    Err(LATENT_UNREPRESENTABLE_MASS_REACHES_CELL)
                }
            };
            match residual
                .exact_sign_after_adding_scalar(boundary_offset)?
                .partial_cmp(&0.0)
            {
                Some(std::cmp::Ordering::Less) if residual_sign > 0.0 => {
                    return cell_holds_the_mass(residual);
                }
                Some(std::cmp::Ordering::Greater) if residual_sign < 0.0 => {
                    return cell_holds_the_mass(residual);
                }
                Some(std::cmp::Ordering::Equal) if mass != 0.0 => {
                    // The components land exactly on the midpoint, so the
                    // exact value straddles it by the mass. A tie is a fact
                    // about a value that is KNOWN; this one is not.
                    return Err(LATENT_UNREPRESENTABLE_MASS_REACHES_CELL);
                }
                Some(std::cmp::Ordering::Equal) => {
                    // An exact midpoint is not an unroundable value. IEEE-754's
                    // default mode -- round to nearest, ties to EVEN -- makes it
                    // unique, and it is the mode every binary64 operation
                    // downstream of this one already uses. This function's own doc
                    // comment names that rule ("Unique nearest-even binary64
                    // rounding"), so refusing here contradicted the contract and
                    // was stricter than the arithmetic the result feeds.
                    //
                    // It is not a measure-zero case in practice. Measured on
                    // gam#2538 at current main: the latent-survival frailty fit
                    // rejects ALL SEVEN outer seeds in 0.49 s with
                    // `latent survival numerator derivative mask 0b0011 has no
                    // certified binary64 value`, because at the beta = 0 outer
                    // seed the cumulant is a small dyadic rational and lands
                    // exactly on a midpoint. The certification refused precisely
                    // the inputs it can round exactly.
                    //
                    // `candidate` and `adjacent` are bit-adjacent -- `next_up` /
                    // `next_down` are `to_bits() +/- 1` -- so exactly one of the
                    // two has an even significand, and the low bit of the pattern
                    // is that significand's LSB. Selecting the even one IS the
                    // tie-to-even neighbour; no tolerance and no choice enter.
                    return Ok(if candidate.to_bits() % 2 == 0 {
                        candidate
                    } else {
                        adjacent
                    });
                }
                _ => {
                    // The exact value lies AT OR BEYOND the far side of the
                    // midpoint, so `adjacent` is strictly nearer than `candidate`:
                    // the accumulated sum simply missed the nearest binary64. Step
                    // onto the neighbour the exact comparison named and certify
                    // again. Nothing here is a tolerance -- the step is decided by
                    // the same exact sign accumulator that decides the return.
                    if walked == LATENT_EXACT_EXPANSION_CAPACITY {
                        return Err(
                            "the exact cumulant stayed outside its rounding cell after a full \
                             structural walk",
                        );
                    }
                    walked += 1;
                    candidate = adjacent;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct LatentSignedLogOrder2<const K: usize> {
    v: LatentSignedLog,
    g: [LatentSignedLog; K],
    h: [[LatentSignedLog; K]; K],
}

impl<const K: usize> LatentSignedLogOrder2<K> {
    const fn zero() -> Self {
        Self {
            v: LatentSignedLog::ZERO,
            g: [LatentSignedLog::ZERO; K],
            h: [[LatentSignedLog::ZERO; K]; K],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryDirection {
    dq: f64,
    dqd: f64,
    dmu: f64,
    dtau: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentSurvivalPrimaryDirection {
    dq_entry: f64,
    dq_exit: f64,
    dqdot_exit: f64,
    dq_right: f64,
    dmu: f64,
    dlog_sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryState {
    q: f64,
    qdot: f64,
    mu: f64,
    sigma: f64,
    log_sigma_factor: f64,
}

/// Complete primary-coordinate state for one latent-survival row.
///
/// Keeping these coupled channels together prevents boundary reordering and
/// cross-row mean/scale mismatches when selecting a derivative backend.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LatentSurvivalPrimaryPoint {
    pub q_entry: f64,
    pub q_exit: f64,
    pub qdot_exit: f64,
    pub q_right: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl LatentSurvivalPrimaryPoint {
    /// Log-scale factor used only by derivatives along the learnable-sigma
    /// coordinate. Fixed zero frailty has no such coordinate, so its factor is
    /// the neutral finite placeholder; every other invalid scale remains NaN
    /// or infinite and is rejected by the kernel's numerical checks.
    #[inline]
    fn log_sigma_factor(self) -> f64 {
        if self.sigma == 0.0 {
            0.0
        } else {
            self.sigma.ln()
        }
    }
}

#[cfg(test)]
mod tests_kernel_recurrence {
    use super::*;
    use std::collections::BTreeMap;

    fn latent_kernel_accumulate_term(
        terms: &mut BTreeMap<(usize, usize, usize, usize), f64>,
        term: LatentKernelPrimaryTerm,
        scale: f64,
    ) {
        if scale == 0.0 || term.coeff == 0.0 {
            return;
        }
        *terms
            .entry((term.q_exp, term.qdot_power, term.tau_exp, term.k))
            .or_insert(0.0) += scale * term.coeff;
    }

    pub(super) fn latent_kernel_differentiate_terms(
        terms: &[LatentKernelPrimaryTerm],
        dir: LatentKernelPrimaryDirection,
    ) -> Vec<LatentKernelPrimaryTerm> {
        let mut out = BTreeMap::<(usize, usize, usize, usize), f64>::new();
        for term in terms {
            if dir.dq != 0.0 {
                if term.q_exp > 0 {
                    latent_kernel_accumulate_term(&mut out, *term, dir.dq * term.q_exp as f64);
                }
                latent_kernel_accumulate_term(
                    &mut out,
                    LatentKernelPrimaryTerm {
                        q_exp: term.q_exp + 1,
                        k: term.k + 1,
                        ..*term
                    },
                    -dir.dq,
                );
            }
            if dir.dmu != 0.0 {
                if term.k > 0 {
                    latent_kernel_accumulate_term(&mut out, *term, dir.dmu * term.k as f64);
                }
                latent_kernel_accumulate_term(
                    &mut out,
                    LatentKernelPrimaryTerm {
                        q_exp: term.q_exp + 1,
                        k: term.k + 1,
                        ..*term
                    },
                    -dir.dmu,
                );
            }
            if dir.dtau != 0.0 {
                if term.tau_exp > 0 {
                    latent_kernel_accumulate_term(&mut out, *term, dir.dtau * term.tau_exp as f64);
                }
                let kf = term.k as f64;
                latent_kernel_accumulate_term(
                    &mut out,
                    LatentKernelPrimaryTerm {
                        tau_exp: term.tau_exp + 2,
                        ..*term
                    },
                    dir.dtau * kf * kf,
                );
                latent_kernel_accumulate_term(
                    &mut out,
                    LatentKernelPrimaryTerm {
                        q_exp: term.q_exp + 1,
                        tau_exp: term.tau_exp + 2,
                        k: term.k + 1,
                        ..*term
                    },
                    -dir.dtau * (2.0 * kf + 1.0),
                );
                latent_kernel_accumulate_term(
                    &mut out,
                    LatentKernelPrimaryTerm {
                        q_exp: term.q_exp + 2,
                        tau_exp: term.tau_exp + 2,
                        k: term.k + 2,
                        ..*term
                    },
                    dir.dtau,
                );
            }
            if dir.dqd != 0.0 && term.qdot_power > 0 {
                latent_kernel_accumulate_term(
                    &mut out,
                    LatentKernelPrimaryTerm {
                        qdot_power: term.qdot_power - 1,
                        ..*term
                    },
                    dir.dqd * term.qdot_power as f64,
                );
            }
        }
        out.into_iter()
            .filter_map(|((q_exp, qdot_power, tau_exp, k), coeff)| {
                (coeff != 0.0).then_some(LatentKernelPrimaryTerm {
                    coeff,
                    q_exp,
                    qdot_power,
                    tau_exp,
                    k,
                })
            })
            .collect()
    }
}

// Fourth-order latent-kernel recurrences have a small finite support. Keeping
// the sorted support inline removes the BTreeMap node allocation and output-Vec
// allocation that the pre-cutover directional oracle intentionally retains.
// The all-event/K=5/K=6 oracle below asserts this capacity never spills.
const LATENT_TERM_INLINE_CAPACITY: usize = 64;
type LatentTermBuffer = SmallVec<[LatentKernelPrimaryTerm; LATENT_TERM_INLINE_CAPACITY]>;

/// Highest kernel rung, `σ`-power, and `qdot`-power the `∂_a` basis handles.
///
/// These are structural bounds, not tolerances: the rung bound keeps every
/// falling-factorial coefficient an exact f64 integer (`|s(12,1)| = 11! =
/// 39916800`), and the other two size the accumulation table. A term list that
/// exceeds any of them routes to the rung basis instead, which is always
/// available.
const LATENT_A_BASIS_MAX_RUNG: usize = 12;
const LATENT_A_BASIS_MAX_TAU_EXP: usize = 12;
const LATENT_A_BASIS_MAX_QDOT_POWER: usize = 4;

/// Signed Stirling numbers of the first kind: the coefficients of the falling
/// factorial `(x)_k = x(x−1)···(x−k+1) = Σ_j s(k,j) x^j`.
///
/// These are the weights that re-express a kernel rung in the `∂_a^j K_0` basis,
/// because `m^k K_k = (−1)^k (∂_a)_k K_0` (#2610). Built by the standard
/// recurrence `(x)_{k+1} = (x)_k · (x − k)` in integers so every entry converts
/// to f64 exactly.
const fn latent_falling_factorial_table()
-> [[i64; LATENT_A_BASIS_MAX_RUNG + 1]; LATENT_A_BASIS_MAX_RUNG + 1] {
    let mut table = [[0_i64; LATENT_A_BASIS_MAX_RUNG + 1]; LATENT_A_BASIS_MAX_RUNG + 1];
    table[0][0] = 1;
    let mut rung = 0usize;
    while rung < LATENT_A_BASIS_MAX_RUNG {
        let mut power = 0usize;
        while power <= rung {
            let value = table[rung][power];
            if value != 0 {
                table[rung + 1][power + 1] += value;
                table[rung + 1][power] -= (rung as i64) * value;
            }
            power += 1;
        }
        rung += 1;
    }
    table
}

const LATENT_FALLING_FACTORIAL: [[i64; LATENT_A_BASIS_MAX_RUNG + 1];
    LATENT_A_BASIS_MAX_RUNG + 1] = latent_falling_factorial_table();

/// Evaluate one latent kernel term list as a signed log magnitude.
///
/// Shared by the packed order-two production path and the multi-direction
/// oracle so the two cannot drift in either the basis they use or the
/// accumulation order they use it in.
fn latent_kernel_evaluate_terms(
    bundle: &LogLognormalKernelBundle,
    state: LatentKernelPrimaryState,
    terms: &[LatentKernelPrimaryTerm],
    context: &str,
) -> Result<(f64, f64), LatentSurvivalError> {
    let needs_qdot = terms
        .iter()
        .any(|term| term.coeff != 0.0 && term.qdot_power > 0);
    if needs_qdot && !(state.qdot.is_finite() && state.qdot > 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} requires positive finite qdot for exact-event directional terms, got {}",
                state.qdot
            ),
        });
    }
    let log_qdot = if needs_qdot { state.qdot.ln() } else { 0.0 };
    if let Some(sum) = latent_kernel_evaluate_terms_in_a_basis(bundle, state, terms, log_qdot) {
        return Ok(sum);
    }
    let mut log_mags = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    let mut signs = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        log_mags.push(
            term.coeff.abs().ln()
                + term.q_exp as f64 * state.q
                + term.tau_exp as f64 * state.log_sigma_factor
                + term.qdot_power as f64 * log_qdot
                + bundle.get(term.k),
        );
        signs.push(term.coeff.signum());
    }
    if log_mags.is_empty() {
        return Ok((f64::NEG_INFINITY, 0.0));
    }
    Ok(signed_log_sum_exp(&log_mags, &signs))
}

/// The same term list evaluated in the `∂_a^j K_0` basis, or `None` when that
/// basis is unavailable for this bundle or this term list (#2610).
///
/// The rung basis and this one are related by an exact integer change of basis,
/// so this is not an approximation — it is the same sum with the cancellation
/// performed on the COEFFICIENTS instead of on the values. That is the whole
/// repair. `∂_{log σ} K_0` reaches the rung basis as `σ²(−mK_1 + m²K_2)`, whose
/// two summands agree to `~1/(120σ²)` of their own size; in this basis the
/// identical quantity is `σ² ∂_a² K_0`, one entry, because the `∂_a` coefficient
/// cancels ANALYTICALLY when like powers are collected. Past `log σ ≈ 5.4` that
/// is the difference between a curvature and its roundoff.
///
/// Requires `q_exp == k` on every term. That is not a restriction in practice:
/// the differentiation rules shift `q_exp` and `k` in lockstep, and every base
/// term the row expression builds starts on the diagonal, so the whole
/// derivative tree stays there. A term off the diagonal routes to the rung
/// basis rather than being handled approximately.
fn latent_kernel_evaluate_terms_in_a_basis(
    bundle: &LogLognormalKernelBundle,
    state: LatentKernelPrimaryState,
    terms: &[LatentKernelPrimaryTerm],
    log_qdot: f64,
) -> Option<(f64, f64)> {
    let tower = bundle.log_scaled_a_derivatives.as_ref()?;
    let mut max_rung = 0usize;
    let mut max_tau_exp = 0usize;
    let mut max_qdot_power = 0usize;
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        if term.q_exp != term.k
            || term.k >= tower.len()
            || term.k > LATENT_A_BASIS_MAX_RUNG
            || term.tau_exp > LATENT_A_BASIS_MAX_TAU_EXP
            || term.qdot_power > LATENT_A_BASIS_MAX_QDOT_POWER
        {
            return None;
        }
        max_rung = max_rung.max(term.k);
        max_tau_exp = max_tau_exp.max(term.tau_exp);
        max_qdot_power = max_qdot_power.max(term.qdot_power);
    }
    // One accumulator per `(qdot_power, tau_exp, j)` monomial. Terms sharing a
    // cell share every factor except the integer coefficient, so collecting them
    // here is where the analytic cancellation happens — exactly, in integers
    // scaled by one common power of σ.
    let rung_stride = max_rung + 1;
    let tau_stride = max_tau_exp + 1;
    let mut coefficients =
        SmallVec::<[f64; 256]>::from_elem(0.0, (max_qdot_power + 1) * tau_stride * rung_stride);
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        let rung_parity = if term.k % 2 == 0 { 1.0 } else { -1.0 };
        let cell = (term.qdot_power * tau_stride + term.tau_exp) * rung_stride;
        for power in 0..=term.k {
            let stirling = LATENT_FALLING_FACTORIAL[term.k][power];
            if stirling != 0 {
                coefficients[cell + power] += rung_parity * term.coeff * stirling as f64;
            }
        }
    }
    let mut log_mags = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    let mut signs = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    for qdot_power in 0..=max_qdot_power {
        for tau_exp in 0..=max_tau_exp {
            for power in 0..=max_rung {
                let coefficient =
                    coefficients[(qdot_power * tau_stride + tau_exp) * rung_stride + power];
                let entry = tower[power];
                if coefficient == 0.0 || entry.sign == 0.0 {
                    continue;
                }
                // `tower` holds `σ^j ∂_a^j K_0`, so the stored `σ^j` is divided
                // back out alongside the term's own `σ^{tau_exp}`.
                log_mags.push(
                    coefficient.abs().ln()
                        + (tau_exp as f64 - power as f64) * state.log_sigma_factor
                        + qdot_power as f64 * log_qdot
                        + entry.log_abs,
                );
                signs.push(coefficient.signum() * entry.sign);
            }
        }
    }
    if log_mags.is_empty() {
        return Some((f64::NEG_INFINITY, 0.0));
    }
    Some(signed_log_sum_exp(&log_mags, &signs))
}

#[inline]
fn latent_kernel_accumulate_term_inline(
    terms: &mut LatentTermBuffer,
    term: LatentKernelPrimaryTerm,
    scale: f64,
) {
    if scale == 0.0 || term.coeff == 0.0 {
        return;
    }
    let contribution = scale * term.coeff;
    if let Some(existing) = terms.iter_mut().find(|existing| {
        existing.q_exp == term.q_exp
            && existing.qdot_power == term.qdot_power
            && existing.tau_exp == term.tau_exp
            && existing.k == term.k
    }) {
        existing.coeff += contribution;
    } else {
        terms.push(LatentKernelPrimaryTerm {
            coeff: contribution,
            ..term
        });
    }
}

fn latent_kernel_differentiate_terms_inline(
    terms: &[LatentKernelPrimaryTerm],
    dir: LatentKernelPrimaryDirection,
) -> LatentTermBuffer {
    let mut out = LatentTermBuffer::new();
    for term in terms {
        if dir.dq != 0.0 {
            if term.q_exp > 0 {
                latent_kernel_accumulate_term_inline(&mut out, *term, dir.dq * term.q_exp as f64);
            }
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dq,
            );
        }
        if dir.dmu != 0.0 {
            if term.k > 0 {
                latent_kernel_accumulate_term_inline(&mut out, *term, dir.dmu * term.k as f64);
            }
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dmu,
            );
        }
        if dir.dtau != 0.0 {
            if term.tau_exp > 0 {
                latent_kernel_accumulate_term_inline(
                    &mut out,
                    *term,
                    dir.dtau * term.tau_exp as f64,
                );
            }
            let kf = term.k as f64;
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    tau_exp: term.tau_exp + 2,
                    ..*term
                },
                dir.dtau * kf * kf,
            );
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    tau_exp: term.tau_exp + 2,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dtau * (2.0 * kf + 1.0),
            );
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 2,
                    tau_exp: term.tau_exp + 2,
                    k: term.k + 2,
                    ..*term
                },
                dir.dtau,
            );
        }
        if dir.dqd != 0.0 && term.qdot_power > 0 {
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    qdot_power: term.qdot_power - 1,
                    ..*term
                },
                dir.dqd * term.qdot_power as f64,
            );
        }
    }
    out.retain(|term| term.coeff != 0.0);
    out.sort_unstable_by_key(|term| (term.q_exp, term.qdot_power, term.tau_exp, term.k));
    out
}

fn latent_kernel_term_sequence_inline(
    base_terms: &[LatentKernelPrimaryTerm],
    axes: &[LatentKernelPrimaryDirection],
    suffix: &[LatentKernelPrimaryDirection],
) -> LatentTermBuffer {
    let mut terms = LatentTermBuffer::from_slice(base_terms);
    terms.retain(|term| term.coeff != 0.0);
    // The canonical subset-cache recurrence strips the least-significant
    // selected slot and applies it after recursively building the remaining
    // mask. Its deterministic floating-point order is therefore highest slot
    // to lowest slot. Preserve that order here so the allocation-free packed
    // path and the independent MultiDir layout accumulate identical analytic
    // coefficients, including cancellation-heavy tail derivatives.
    for direction in axes.iter().chain(suffix.iter()).rev() {
        terms = latent_kernel_differentiate_terms_inline(&terms, *direction);
    }
    terms
}

#[cfg(test)]
mod tests_multidir_kernel {
    /// Derivatives of `log(x)` through fourth order at the only point needed by
    /// normalized kernel sums: the literal `x = 1`.
    ///
    /// Keeping the point in the function name and removing the free argument
    /// makes the representability contract structural. No caller can
    /// accidentally feed a small positive linear-domain mass into the
    /// reciprocal powers.
    #[inline]
    fn latent_unary_derivatives_log_at_one() -> [f64; 5] {
        [0.0, 1.0, -1.0, 2.0, -6.0]
    }

    use super::tests_kernel_recurrence::latent_kernel_differentiate_terms;
    use super::*;
    use gam_math::jet_partitions::MultiDirJet as LatentMultiDirJet;

    fn latent_kernel_term_lists_for_directions(
        base_terms: &[LatentKernelPrimaryTerm],
        directions: &[LatentKernelPrimaryDirection],
    ) -> Vec<Vec<LatentKernelPrimaryTerm>> {
        fn build_mask(
            mask: usize,
            base_terms: &[LatentKernelPrimaryTerm],
            directions: &[LatentKernelPrimaryDirection],
            cache: &mut [Option<Vec<LatentKernelPrimaryTerm>>],
        ) -> Vec<LatentKernelPrimaryTerm> {
            if let Some(existing) = &cache[mask] {
                return existing.clone();
            }
            let built = if mask == 0 {
                base_terms.to_vec()
            } else {
                let bit = 1usize << mask.trailing_zeros();
                let prev = build_mask(mask ^ bit, base_terms, directions, cache);
                latent_kernel_differentiate_terms(&prev, directions[bit.trailing_zeros() as usize])
            };
            cache[mask] = Some(built.clone());
            built
        }

        let mut cache = vec![None; 1usize << directions.len()];
        (0..cache.len())
            .map(|mask| build_mask(mask, base_terms, directions, &mut cache))
            .collect()
    }

    pub(super) fn latent_kernel_sum_log_jet(
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        directions: &[LatentKernelPrimaryDirection],
        context: &str,
    ) -> Result<LatentMultiDirJet, LatentSurvivalError> {
        let term_lists = latent_kernel_term_lists_for_directions(base_terms, directions);
        let max_k = term_lists
            .iter()
            .flat_map(|terms| terms.iter().map(|term| term.k))
            .max()
            .unwrap_or(0);
        let bundle = log_kernel_bundle(quadctx, state.q.exp(), state.mu, state.sigma, max_k)
            .map_err(|e| LatentSurvivalError::NumericalFailure {
                reason: format!("{context} kernel evaluation failed: {e}"),
            })?;

        let evaluate_terms = |terms: &[LatentKernelPrimaryTerm]| {
            latent_kernel_evaluate_terms(&bundle, state, terms, context)
        };

        let (base_log_sum, base_sign) = evaluate_terms(&term_lists[0])?;
        if !(base_log_sum.is_finite() && base_sign > 0.0) {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!("{context} produced a non-positive signed kernel sum"),
            });
        }

        let mut normalized = LatentMultiDirJet::constant(directions.len(), 1.0);
        let mut signed_moments = [LatentSignedLog::ZERO; 16];
        signed_moments[0] = LatentSignedLog::ONE;
        for mask in 1..term_lists.len() {
            let (log_abs, sign) = evaluate_terms(&term_lists[mask])?;
            let moment =
                latent_signed_log_normalized(log_abs, sign, base_log_sum, context)?;
            signed_moments[mask] = moment;
            normalized.coeffs[mask] = latent_signed_log_materialize(moment, context)?;
        }

        let mut out = normalized.compose_unary(latent_unary_derivatives_log_at_one());
        out.coeffs[0] += base_log_sum;
        if term_lists.len() == 1 {
            return Ok(out);
        }

        // Independently grade the pre-cutover floating oracle against the exact
        // rounded-moment polynomial. `MultiDirJet::compose_unary` supports at
        // most four live slots here. Its K=4 pointed schedule walks 34 Dot2
        // terms (10 rounded operations each), writes 17 compensated power
        // outputs (5 operations each), and combines four powers in at most 32
        // operations: 457 total. Smaller orders are strict subsets, so this is
        // a structural uniform bound rather than an empirical multiplier.
        const MULTIDIR_DOT2_TERMS_K4: usize = 34;
        const MULTIDIR_DOT2_OPERATIONS: usize = 10;
        const MULTIDIR_POWER_OUTPUTS_K4: usize = 17;
        const MULTIDIR_POWER_OUTPUT_OPERATIONS: usize = 5;
        const MULTIDIR_COMBINE_OPERATIONS: usize = 32;
        const MULTIDIR_MAX_OPERATIONS: usize =
            MULTIDIR_DOT2_TERMS_K4 * MULTIDIR_DOT2_OPERATIONS
                + MULTIDIR_POWER_OUTPUTS_K4 * MULTIDIR_POWER_OUTPUT_OPERATIONS
                + MULTIDIR_COMBINE_OPERATIONS;
        const _: () = assert!(MULTIDIR_MAX_OPERATIONS == 457);
        let operation_roundoff = MULTIDIR_MAX_OPERATIONS as f64 * f64::EPSILON;
        let gamma = LatentExactExpansion::next_up(
            operation_roundoff / (1.0 - operation_roundoff),
        );
        let exact = latent_certified_cumulants(
            signed_moments,
            term_lists.len() - 1,
            context,
        )?;
        for mask in 1..term_lists.len() {
            let relative_roundoff =
                LatentExactExpansion::next_up(gamma * exact[mask].absolute_term_mass);
            let gradual_underflow =
                MULTIDIR_MAX_OPERATIONS as f64 * f64::from_bits(1);
            let arithmetic_bound =
                LatentExactExpansion::next_up(relative_roundoff + gradual_underflow);
            let error =
                LatentExactExpansion::next_up((out.coeffs[mask] - exact[mask].value).abs());
            assert!(
                error <= arithmetic_bound,
                "{context}: MultiDir mask {mask:#06b} escaped its derived forward-error \
                 certificate: got={:.17e}, exact-rounded={:.17e}, error={error:.17e}, \
                 bound={arithmetic_bound:.17e}, absolute-term-mass={:.17e}, operations={MULTIDIR_MAX_OPERATIONS}",
                out.coeffs[mask],
                exact[mask].value,
                exact[mask].absolute_term_mass,
            );
        }
        Ok(out)
    }
}

fn latent_signed_log_checked(
    log_abs: f64,
    sign: f64,
    context: &str,
    quantity: &str,
) -> Result<LatentSignedLog, LatentSurvivalError> {
    if log_abs == f64::NEG_INFINITY && sign == 0.0 {
        return Ok(LatentSignedLog::ZERO);
    }
    if log_abs.is_finite() && (sign == -1.0 || sign == 1.0) {
        return Ok(LatentSignedLog { log_abs, sign });
    }
    Err(LatentSurvivalError::NumericalFailure {
        reason: format!(
            "{context} produced an invalid signed-log {quantity}: log_abs={log_abs}, sign={sign}"
        ),
    })
}

fn latent_signed_log_normalized(
    log_abs: f64,
    sign: f64,
    base_log_sum: f64,
    context: &str,
) -> Result<LatentSignedLog, LatentSurvivalError> {
    let value = latent_signed_log_checked(log_abs, sign, context, "kernel derivative")?;
    if value.sign == 0.0 {
        Ok(value)
    } else {
        latent_signed_log_checked(
            value.log_abs - base_log_sum,
            value.sign,
            context,
            "normalised kernel derivative",
        )
    }
}

fn latent_signed_log_materialize(
    value: LatentSignedLog,
    context: &str,
) -> Result<f64, LatentSurvivalError> {
    let value =
        latent_signed_log_checked(value.log_abs, value.sign, context, "log-sum derivative")?;
    if value.sign == 0.0 {
        return Ok(0.0);
    }
    let materialized = value.sign * value.log_abs.exp();
    if materialized.is_finite() {
        Ok(materialized)
    } else {
        Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} log-sum derivative is outside the finite f64 range: \
                 log_abs={}, sign={}",
                value.log_abs, value.sign
            ),
        })
    }
}

/// Convert normalized signed-log moments into certified derivatives of `log(S)`.
///
/// For a non-empty slot set `A`, normalized moments and log derivatives obey
///
/// `m(A) = Σ_{B ⊆ A, pivot ∈ B} κ(B) m(A \\ B)`,
///
/// where `m(A) = S_A / S`, `κ(A) = ∂_A log(S)`, and the distinguished pivot is
/// the least-significant slot. Isolating `B = A` gives a pointed cumulant
/// recurrence.
///
/// The contract is deliberately about the inputs an actual binary64 consumer
/// has: each finite signed-log moment is rounded once to binary64, then the
/// exact-real cumulant polynomial over those rounded moments is evaluated by a
/// fixed error-free expansion. Publication succeeds only when the expansion
/// proves a unique nearest binary64 rounding. A moment lost to underflow,
/// overflow, an unrepresentable intermediate product, or an ambiguous final
/// rounding produces [`LatentSurvivalError::DerivativeAccuracyUnresolved`]
/// instead of an approximate derivative.
///
/// The table covers the four-slot `(a,b,u,v)` layouts used by Order2, OneSeed,
/// and TwoSeed, and the five-slot `(a,b,u,v,w)` layout of the three-seed lift.
fn latent_certified_cumulants<const M: usize>(
    moments: [LatentSignedLog; M],
    target_mask: usize,
    context: &str,
) -> Result<[LatentCertifiedCumulant; M], LatentSurvivalError> {
    assert!(M.is_power_of_two() && target_mask > 0 && target_mask < M);

    let unresolved = |mask: usize, reason: &str| {
        LatentSurvivalError::DerivativeAccuracyUnresolved {
            reason: format!(
                "{context} derivative mask {mask:#06b} has no certified binary64 value: {reason}"
            ),
        }
    };
    let mut rounded_moments = [0.0_f64; M];
    for mask in 0usize..M {
        if mask & !target_mask != 0 {
            continue;
        }
        let moment = latent_signed_log_checked(
            moments[mask].log_abs,
            moments[mask].sign,
            context,
            "normalised cumulant moment",
        )?;
        if moment.sign == 0.0 {
            continue;
        }
        let magnitude = moment.log_abs.exp();
        if !magnitude.is_finite() {
            return Err(unresolved(
                mask,
                "the rounded moment magnitude overflowed",
            ));
        }
        if magnitude == 0.0 {
            return Err(unresolved(
                mask,
                "the rounded moment magnitude underflowed to zero",
            ));
        }
        rounded_moments[mask] = moment.sign * magnitude;
    }

    // The moments' own dynamic range is what every refusal below is actually
    // about, and it is the one number a caller cannot recover from the message
    // afterwards — recovering it has twice cost this lane a three-quarter-hour
    // fit (gam#2714). Report it with the refusal, not instead of it.
    let moment_span = || -> String {
        let mut lowest = f64::INFINITY;
        let mut highest = 0.0_f64;
        for mask in 0usize..M {
            let magnitude = rounded_moments[mask].abs();
            if magnitude > 0.0 {
                lowest = lowest.min(magnitude);
                highest = highest.max(magnitude);
            }
        }
        if highest == 0.0 {
            return "no nonzero rounded moment".to_string();
        }
        format!("rounded moment magnitudes span [{lowest:.6e}, {highest:.6e}]")
    };
    let unresolved_with_span = |mask: usize, reason: &str| {
        unresolved(mask, &format!("{reason} ({})", moment_span()))
    };

    let mut exact_cumulants = [LatentExactExpansion::ZERO; M];
    let mut certificates = [LatentCertifiedCumulant::ZERO; M];
    for mask in 1usize..M {
        if mask & !target_mask != 0 {
            continue;
        }
        if mask.is_power_of_two() {
            exact_cumulants[mask] = LatentExactExpansion::scalar(rounded_moments[mask]);
            certificates[mask] = LatentCertifiedCumulant {
                value: rounded_moments[mask],
                absolute_term_mass: rounded_moments[mask].abs(),
            };
            continue;
        }
        let pivot = 1usize << mask.trailing_zeros();
        let mut cumulant = LatentExactExpansion::scalar(rounded_moments[mask]);
        let mut absolute_term_mass = rounded_moments[mask].abs();
        let mut block = (mask - 1) & mask;
        while block != 0 {
            if block & pivot != 0 {
                let complement = mask ^ block;
                let subtract = exact_cumulants[block]
                    .scale(-rounded_moments[complement])
                    .and_then(|term| cumulant.add(term))
                    .map_err(|reason| unresolved_with_span(mask, reason))?;
                cumulant = subtract;
                let term_mass = LatentExactExpansion::next_up(
                    certificates[block].absolute_term_mass
                        * rounded_moments[complement].abs(),
                );
                if !term_mass.is_finite() {
                    return Err(unresolved(
                        mask,
                        "the conditioning mass overflowed binary64",
                    ));
                }
                absolute_term_mass =
                    LatentExactExpansion::next_up(absolute_term_mass + term_mass);
                if !absolute_term_mass.is_finite() {
                    return Err(unresolved(
                        mask,
                        "the accumulated conditioning mass overflowed binary64",
                    ));
                }
            }
            block = (block - 1) & mask;
        }
        let value = cumulant
            .certified_round()
            .map_err(|reason| unresolved_with_span(mask, reason))?;
        exact_cumulants[mask] = cumulant;
        certificates[mask] = LatentCertifiedCumulant {
            value,
            absolute_term_mass,
        };
    }
    Ok(certificates)
}

/// A one-pass analytic lift of a latent kernel sum into an order-specific jet.
///
/// `suffixes` describes the nilpotent parts carried by the requested scalar:
/// `[]` for the ordinary order-two base, `[u]` for the `OneSeed` epsilon part,
/// and `[u]`, `[v]`, `[u,v]` for the three non-base `TwoSeed` parts.  For each
/// part we differentiate the SAME kernel-term program in the canonical
/// highest-slot-to-lowest-slot order used by the pre-cutover `MultiDirJet`
/// subset cache. Every requested raw derivative is therefore assembled by the
/// same recurrence, accumulation order, and signed-log reduction as its oracle.
/// The expensive quadrature bundle is then evaluated ONCE at the maximum `k`
/// required by the complete output instead of once per Hessian cell.
/// The kernel rung ceiling one lift needs.
///
/// Every emitted part carries a full order-two base, so the largest requested
/// recurrence is the base list's own highest rung plus two primary
/// differentiations plus the largest nilpotent suffix. This exact support bound
/// is what lets the one shared bundle be built before any individual derivative
/// term list is constructed.
///
/// Extracted so the value-only lift and the order-two lift cannot drift: the
/// bundle's `max_k` decides which rungs exist AND (through
/// `log_scaled_a_derivative_tower`) how long the `∂_a` tower is, so two lifts
/// that computed it differently would evaluate the same base term list on
/// different data (#2714).
fn latent_kernel_sum_max_k<const K: usize>(
    base_terms: &[LatentKernelPrimaryTerm],
    primary_directions: &[LatentKernelPrimaryDirection; K],
    suffixes: &[&[LatentKernelPrimaryDirection]],
) -> usize {
    let base_max_k = base_terms.iter().map(|term| term.k).max().unwrap_or(0);
    let k_increment = |direction: &LatentKernelPrimaryDirection| {
        if direction.dtau != 0.0 {
            2
        } else if direction.dq != 0.0 || direction.dmu != 0.0 {
            1
        } else {
            0
        }
    };
    let max_primary_increment = primary_directions
        .iter()
        .map(&k_increment)
        .max()
        .unwrap_or(0);
    let max_suffix_increment = suffixes
        .iter()
        .map(|suffix| suffix.iter().map(&k_increment).sum::<usize>())
        .max()
        .unwrap_or(0);
    base_max_k + 2 * max_primary_increment + max_suffix_increment
}

/// The shared prologue of every lift: the ONE kernel bundle, and the base term
/// list's signed-log sum on it.
///
/// This is the row's log-likelihood contribution before any normalisation —
/// `latent_kernel_sum_order2_parts` publishes it verbatim as the value channel
/// (`out[0].0.v = base_log_sum`). Sharing it with the value-only lift is what
/// makes `log_likelihood_only` and the joint gradient evaluate ONE function of
/// `β` rather than two implementations of one formula (#2714).
fn latent_kernel_sum_base<const K: usize>(
    quadctx: &QuadratureContext,
    base_terms: &[LatentKernelPrimaryTerm],
    state: LatentKernelPrimaryState,
    primary_directions: &[LatentKernelPrimaryDirection; K],
    suffixes: &[&[LatentKernelPrimaryDirection]],
    context: &str,
) -> Result<(LogLognormalKernelBundle, f64), LatentSurvivalError> {
    let max_k = latent_kernel_sum_max_k(base_terms, primary_directions, suffixes);
    let bundle =
        log_kernel_bundle(quadctx, state.q.exp(), state.mu, state.sigma, max_k).map_err(|e| {
            LatentSurvivalError::NumericalFailure {
                reason: format!("{context} kernel evaluation failed: {e}"),
            }
        })?;
    let (base_log_sum, base_sign) =
        latent_kernel_evaluate_terms(&bundle, state, base_terms, context)?;
    if !(base_log_sum.is_finite() && base_sign > 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!("{context} produced a non-positive signed kernel sum"),
        });
    }
    Ok((bundle, base_log_sum))
}

fn latent_kernel_sum_order2_parts<const K: usize>(
    quadctx: &QuadratureContext,
    base_terms: &[LatentKernelPrimaryTerm],
    state: LatentKernelPrimaryState,
    primary_directions: &[LatentKernelPrimaryDirection; K],
    suffixes: &[&[LatentKernelPrimaryDirection]],
    context: &str,
) -> Result<[Order2<K>; 8], LatentSurvivalError> {
    assert!(
        !suffixes.is_empty() && suffixes.len() <= 8,
        "latent kernel lift supports one to eight order-two parts"
    );
    let (bundle, base_log_sum) = latent_kernel_sum_base(
        quadctx,
        base_terms,
        state,
        primary_directions,
        suffixes,
        context,
    )?;

    let evaluate_terms = |terms: &[LatentKernelPrimaryTerm]| {
        latent_kernel_evaluate_terms(&bundle, state, terms, context)
    };

    let normalized = |axes: &[LatentKernelPrimaryDirection],
                      suffix: &[LatentKernelPrimaryDirection]|
     -> Result<LatentSignedLog, LatentSurvivalError> {
        let is_zero = |direction: &LatentKernelPrimaryDirection| {
            direction.dq == 0.0
                && direction.dqd == 0.0
                && direction.dmu == 0.0
                && direction.dtau == 0.0
        };
        if axes.iter().chain(suffix.iter()).any(is_zero) {
            return Ok(LatentSignedLog::ZERO);
        }
        let terms = latent_kernel_term_sequence_inline(base_terms, axes, suffix);
        // Term lists over at most four slots (the order-two, one-seed and two-seed
        // lifts) stay inside the inline buffer. The three-seed lift's five-slot
        // lists can outgrow it and are served from the heap (#2677).
        assert!(
            axes.len() + suffix.len() > 4 || !terms.spilled(),
            "latent derivative support exceeded the inline allocation-free capacity: {} > {}",
            terms.len(),
            LATENT_TERM_INLINE_CAPACITY
        );
        let (log_abs, sign) = evaluate_terms(&terms)?;
        latent_signed_log_normalized(log_abs, sign, base_log_sum, context)
    };

    let mut parts = [LatentSignedLogOrder2::<K>::zero(); 8];
    for (part, suffix) in suffixes.iter().enumerate() {
        let value = if part == 0 {
            // The base is the kernel divided by itself.
            LatentSignedLog::ONE
        } else {
            normalized(&[], suffix)?
        };
        let mut tower = LatentSignedLogOrder2::<K>::zero();
        tower.v = value;
        for a in 0..K {
            tower.g[a] = normalized(&[primary_directions[a]], suffix)?;
        }
        for a in 0..K {
            for b in a..K {
                let derivative =
                    normalized(&[primary_directions[a], primary_directions[b]], suffix)?;
                tower.h[a][b] = derivative;
                tower.h[b][a] = derivative;
            }
        }
        parts[part] = tower;
    }
    latent_kernel_signed_log_parts(
        base_log_sum,
        parts,
        suffixes.len(),
        context,
    )
}

/// Convert normalized signed-log kernel moments into derivatives of the log sum.
///
/// `normalized_parts` is the single analytic recurrence's compact moment
/// layout, indexed by seed subset (`bit 0 = u`, `bit 1 = v`, `bit 2 = w`): the
/// base carries `(1, S_a/S, S_ab/S)`, a part along seed set `P` carries
/// `(S_P/S, S_aP/S, S_abP/S)`. For each requested output channel those moments
/// become one table over the slots `(a, b, seeds…)` for
/// [`latent_certified_cumulants`], with part `p`'s moments at `(p << 2) | j`.
/// Each derivative is published only after its exact expansion has certified
/// the unique rounded result.
fn latent_kernel_signed_log_parts<const K: usize>(
    base_log_sum: f64,
    normalized_parts: [LatentSignedLogOrder2<K>; 8],
    part_count: usize,
    context: &str,
) -> Result<[Order2<K>; 8], LatentSurvivalError> {
    assert!(matches!(part_count, 1 | 2 | 4 | 8));
    let compose_log = |moments: [LatentSignedLog; 32], target_mask: usize| {
        latent_certified_cumulants(moments, target_mask, context)
    };
    let moments_for = |a: usize, b: usize| {
        let mut moments = [LatentSignedLog::ZERO; 32];
        for (part, tower) in normalized_parts.iter().enumerate() {
            let offset = part << 2;
            moments[offset] = if part == 0 {
                LatentSignedLog::ONE
            } else {
                tower.v
            };
            moments[offset | 0b01] = tower.g[a];
            moments[offset | 0b10] = tower.g[b];
            moments[offset | 0b11] = tower.h[a][b];
        }
        moments
    };

    // A single-seed part's value is its normalized moment; a part along several
    // seeds reads its cumulant off one composition over every seed slot.
    let seed_mask = part_count - 1;
    let mut out = [Order2::<K>::constant(0.0); 8];
    out[0].0.v = base_log_sum;
    for part in 1..part_count {
        if part.is_power_of_two() {
            out[part].0.v = latent_signed_log_materialize(normalized_parts[part].v, context)?;
        }
    }
    if part_count >= 4 {
        let composed = compose_log(moments_for(0, 0), seed_mask << 2)?;
        for part in 1..part_count {
            if !part.is_power_of_two() {
                out[part].0.v = composed[part << 2].value;
            }
        }
    }

    let gradient_mask = (seed_mask << 2) | 0b01;
    let hessian_mask = (seed_mask << 2) | 0b11;
    for a in 0..K {
        let composed = compose_log(moments_for(a, a), gradient_mask)?;
        for part in 0..part_count {
            out[part].0.g[a] = composed[(part << 2) | 0b01].value;
        }
        for b in a..K {
            let composed = compose_log(moments_for(a, b), hessian_mask)?;
            for part in 0..part_count {
                out[part].0.h[a][b] = composed[(part << 2) | 0b11].value;
                out[part].0.h[b][a] = out[part].0.h[a][b];
            }
        }
    }
    Ok(out)
}

#[inline]
fn latent_kernel_direction_linear_combination<const K: usize>(
    primary_directions: &[LatentKernelPrimaryDirection; K],
    coefficients: &[f64; K],
) -> LatentKernelPrimaryDirection {
    let mut out = LatentKernelPrimaryDirection {
        dq: 0.0,
        dqd: 0.0,
        dmu: 0.0,
        dtau: 0.0,
    };
    for a in 0..K {
        out.dq += coefficients[a] * primary_directions[a].dq;
        out.dqd += coefficients[a] * primary_directions[a].dqd;
        out.dmu += coefficients[a] * primary_directions[a].dmu;
        out.dtau += coefficients[a] * primary_directions[a].dtau;
    }
    out
}

fn latent_order2_all_finite<const K: usize>(jet: &Order2<K>) -> bool {
    jet.value().is_finite()
        && jet.g().iter().all(|value| value.is_finite())
        && jet
            .h()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
}

/// The jet operations the single latent row expression performs: kernel
/// log-sums combine linearly, and the interval branch composes `log(1 − eᵟ)`
/// over a log-boundary gap. The composition is the one place a jet's order
/// shows, so each jet reads the certified derivative stack through its own
/// order there.
trait LatentRowJet<const K: usize>: Copy {
    fn row_constant(value: f64) -> Self;
    fn row_value(&self) -> f64;
    fn row_add(&self, other: &Self) -> Self;
    fn row_sub(&self, other: &Self) -> Self;
    /// `log(1 − eᵟ)` composed over this jet at its negative value `δ`, with the
    /// unary stack certified through `derivative_order`.
    fn row_compose_log1mexp_negative(
        &self,
        derivative_order: usize,
        context: &str,
    ) -> Result<Self, LatentSurvivalError>;
}

/// The order-two, one-seed and two-seed jets compose through the five-entry stack
/// their `JetScalar` composition reads.
macro_rules! latent_row_jet_through_fourth_order {
    ($($jet:ident),+) => {$(
        impl<const K: usize> LatentRowJet<K> for $jet<K> {
            fn row_constant(value: f64) -> Self {
                <Self as JetScalar<K>>::constant(value)
            }

            fn row_value(&self) -> f64 {
                JetField::value(self)
            }

            fn row_add(&self, other: &Self) -> Self {
                JetField::add(self, other)
            }

            fn row_sub(&self, other: &Self) -> Self {
                JetField::sub(self, other)
            }

            fn row_compose_log1mexp_negative(
                &self,
                derivative_order: usize,
                context: &str,
            ) -> Result<Self, LatentSurvivalError> {
                let stack = latent_unary_derivatives_log1mexp_negative::<5>(
                    JetField::value(self),
                    derivative_order,
                    context,
                )?;
                Ok(JetField::compose_unary(self, stack))
            }
        }
    )+};
}

latent_row_jet_through_fourth_order!(Order2, OneSeed, TwoSeed);

/// A row jet carrying the contracted fifth derivative (#2677): eight order-two
/// parts indexed by seed subset (`bit 0 = u`, `bit 1 = v`, `bit 2 = w`), with
/// `ε_u² = ε_v² = ε_w² = 0`. After a lift along `(u, v, w)`,
/// `parts[7].h[a][b] = Σ_{cde} ℓ_{abcde} u_c v_d w_e`.
#[derive(Clone, Copy)]
struct LatentThreeSeedRow<const K: usize> {
    parts: [Order2<K>; 8],
}

impl<const K: usize> LatentThreeSeedRow<K> {
    fn all_channels_finite(&self) -> bool {
        self.parts.iter().all(latent_order2_all_finite)
    }
}

impl<const K: usize> LatentRowJet<K> for LatentThreeSeedRow<K> {
    fn row_constant(value: f64) -> Self {
        let mut parts = [Order2::<K>::constant(0.0); 8];
        parts[0] = Order2::<K>::constant(value);
        Self { parts }
    }

    fn row_value(&self) -> f64 {
        self.parts[0].value()
    }

    fn row_add(&self, other: &Self) -> Self {
        Self {
            parts: std::array::from_fn(|part| self.parts[part].add(&other.parts[part])),
        }
    }

    fn row_sub(&self, other: &Self) -> Self {
        Self {
            parts: std::array::from_fn(|part| self.parts[part].sub(&other.parts[part])),
        }
    }

    /// Faà di Bruno over the nilpotent seeds: a part's coefficient sums, over the
    /// set partitions of its seed subset into blocks `B_1 … B_k`,
    /// `f⁽ᵏ⁾(x₀)·Π_j x_{B_j}`, where `f⁽ᵏ⁾(x₀)` is the order-two composition of the
    /// base part with the stack shifted `k` entries. The triple part reaches
    /// `f‴(x₀)` as an order-two jet, which reads the stack through `f⁽⁵⁾`.
    fn row_compose_log1mexp_negative(
        &self,
        derivative_order: usize,
        context: &str,
    ) -> Result<Self, LatentSurvivalError> {
        let stack = latent_unary_derivatives_log1mexp_negative::<6>(
            self.parts[0].value(),
            derivative_order,
            context,
        )?;
        let x = &self.parts;
        let shifted = |order: usize| {
            x[0].compose_unary([stack[order], stack[order + 1], stack[order + 2], 0.0, 0.0])
        };
        let (f0, f1, f2, f3) = (shifted(0), shifted(1), shifted(2), shifted(3));
        let (u, v, w) = (&x[1], &x[2], &x[4]);
        let pair = |left: &Order2<K>, right: &Order2<K>, joint: &Order2<K>| {
            f2.mul(left).mul(right).add(&f1.mul(joint))
        };
        let mut parts = [Order2::<K>::constant(0.0); 8];
        parts[0] = f0;
        parts[1] = f1.mul(u);
        parts[2] = f1.mul(v);
        parts[4] = f1.mul(w);
        parts[3] = pair(u, v, &x[3]);
        parts[5] = pair(u, w, &x[5]);
        parts[6] = pair(v, w, &x[6]);
        parts[7] = f3
            .mul(u)
            .mul(v)
            .mul(w)
            .add(&f2.mul(u).mul(&x[6]))
            .add(&f2.mul(v).mul(&x[5]))
            .add(&f2.mul(w).mul(&x[3]))
            .add(&f1.mul(&x[7]));
        Ok(Self { parts })
    }
}

/// Backend seam for the single latent-survival row expression.  Only the
/// analytic multivariate kernel primitive differs by requested channel; all
/// numerator/denominator/event algebra below is instantiated unchanged.
trait LatentPrimaryJetBackend<const K: usize> {
    type Jet: LatentRowJet<K>;

    fn derivative_order(&self) -> usize;
    fn all_channels_finite(&self, jet: &Self::Jet) -> bool;

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError>;
}

#[derive(Clone, Copy)]
struct LatentOrder2Backend;

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentOrder2Backend {
    type Jet = Order2<K>;

    fn derivative_order(&self) -> usize {
        2
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(jet)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        let suffixes: [&[LatentKernelPrimaryDirection]; 1] = [&[]];
        let parts = latent_kernel_sum_order2_parts(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(parts[0])
    }
}

/// Value-only backend: the SAME row expression, evaluated for its value channel
/// and nothing else (#2714).
///
/// This is what `log_likelihood_only` — the scalar the joint-Newton trust region
/// evaluates at the TRIAL β — goes through, so that the accept test measures the
/// value of the function whose gradient the step was built from, rather than a
/// second implementation of the same formula. The trust ratio's numerator
/// differences `old_objective` (built from the gradient hook's log-likelihood)
/// against `trial_objective` (built from this one); a gap between them is a
/// constant of the backtracking ladder that shrinking the radius cannot remove.
///
/// **`K` is deliberately the same `K` the derivative backends use** even though
/// no derivative is produced. [`latent_kernel_sum_max_k`] sizes the kernel
/// bundle from the primary directions, and the bundle's `max_k` decides both
/// which rungs exist and how long the `∂_a` tower is — so a backend that dropped
/// the directions would build a SHORTER bundle and could be routed to a
/// different basis for the same term list. Same `K` ⇒ same bundle ⇒ same basis ⇒
/// the value is bit-identical, which is the entire point.
///
/// `derivative_order = 0` because no derivative channel is filled: it reaches
/// only `latent_survival_positive_log_difference_jet`, where it means the
/// interval branch validates the unary composition to order 0. A value-only
/// evaluation must not refuse because some derivative of the log-gap is
/// unrepresentable.
#[derive(Clone, Copy)]
struct LatentValueBackend;

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentValueBackend {
    type Jet = Order2<K>;

    fn derivative_order(&self) -> usize {
        0
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(jet)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        // The same `suffixes` the order-two backend passes, so
        // `latent_kernel_sum_max_k` returns the same ceiling.
        let suffixes: [&[LatentKernelPrimaryDirection]; 1] = [&[]];
        let (_, base_log_sum) = latent_kernel_sum_base(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(Order2::<K>::constant(base_log_sum))
    }
}

#[derive(Clone, Copy)]
struct LatentOneSeedBackend<const K: usize> {
    direction: [f64; K],
}

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentOneSeedBackend<K> {
    type Jet = OneSeed<K>;

    fn derivative_order(&self) -> usize {
        3
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(&jet.base) && latent_order2_all_finite(&jet.eps)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        let seed = latent_kernel_direction_linear_combination(primary_directions, &self.direction);
        let seed_suffix = [seed];
        let suffixes: [&[LatentKernelPrimaryDirection]; 2] = [&[], &seed_suffix];
        let parts = latent_kernel_sum_order2_parts(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(OneSeed {
            base: parts[0],
            eps: parts[1],
        })
    }
}

#[derive(Clone, Copy)]
struct LatentTwoSeedBackend<const K: usize> {
    direction_u: [f64; K],
    direction_v: [f64; K],
}

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentTwoSeedBackend<K> {
    type Jet = TwoSeed<K>;

    fn derivative_order(&self) -> usize {
        4
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(&jet.base)
            && latent_order2_all_finite(&jet.eps)
            && latent_order2_all_finite(&jet.del)
            && latent_order2_all_finite(&jet.eps_del)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        let seed_u =
            latent_kernel_direction_linear_combination(primary_directions, &self.direction_u);
        let seed_v =
            latent_kernel_direction_linear_combination(primary_directions, &self.direction_v);
        let suffix_u = [seed_u];
        let suffix_v = [seed_v];
        let suffix_uv = [seed_u, seed_v];
        let suffixes: [&[LatentKernelPrimaryDirection]; 4] =
            [&[], &suffix_u, &suffix_v, &suffix_uv];
        let parts = latent_kernel_sum_order2_parts(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(TwoSeed {
            base: parts[0],
            eps: parts[1],
            del: parts[2],
            eps_del: parts[3],
        })
    }
}

/// The lift of the contracted fifth derivative along `(u, v, w)` (#2677).
#[derive(Clone, Copy)]
struct LatentThreeSeedBackend<const K: usize> {
    direction_u: [f64; K],
    direction_v: [f64; K],
    direction_w: [f64; K],
}

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentThreeSeedBackend<K> {
    type Jet = LatentThreeSeedRow<K>;

    fn derivative_order(&self) -> usize {
        5
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        jet.all_channels_finite()
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        let seed_u =
            latent_kernel_direction_linear_combination(primary_directions, &self.direction_u);
        let seed_v =
            latent_kernel_direction_linear_combination(primary_directions, &self.direction_v);
        let seed_w =
            latent_kernel_direction_linear_combination(primary_directions, &self.direction_w);
        let suffix_u = [seed_u];
        let suffix_v = [seed_v];
        let suffix_uv = [seed_u, seed_v];
        let suffix_w = [seed_w];
        let suffix_uw = [seed_u, seed_w];
        let suffix_vw = [seed_v, seed_w];
        let suffix_uvw = [seed_u, seed_v, seed_w];
        let suffixes: [&[LatentKernelPrimaryDirection]; 8] = [
            &[],
            &suffix_u,
            &suffix_v,
            &suffix_uv,
            &suffix_w,
            &suffix_uw,
            &suffix_vw,
            &suffix_uvw,
        ];
        let parts = latent_kernel_sum_order2_parts(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(LatentThreeSeedRow { parts })
    }
}

fn latent_survival_basis_direction(primary_idx: usize) -> LatentSurvivalPrimaryDirection {
    match primary_idx {
        LATENT_SURVIVAL_PRIMARY_Q_ENTRY => LatentSurvivalPrimaryDirection {
            dq_entry: 1.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_Q_EXIT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 1.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_QDOT_EXIT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 1.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_Q_RIGHT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 1.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_MU => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 1.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 1.0,
        },
        // SAFETY: latent survival has exactly `LATENT_SURVIVAL_PRIMARY_DIM`
        // (= 5) primary directions, indexed 0..=4 via the module-private
        // `LATENT_SURVIVAL_PRIMARY_*` constants. All five are matched
        // above, so this wildcard fires only on an out-of-range index,
        // which the internal iteration bounds (`0..LATENT_SURVIVAL_PRIMARY_DIM`)
        // make unreachable.
        // SAFETY: primary_idx is bounded by LATENT_SURVIVAL_PRIMARY_DIM at every internal call site.
        _ => std::panic::panic_any(format!(
            "latent survival primary index out of bounds: primary_idx={primary_idx}, primary_dim={LATENT_SURVIVAL_PRIMARY_DIM}"
        )),
    }
}

fn latent_survival_map_entry_direction(
    direction: LatentSurvivalPrimaryDirection,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_entry,
        dqd: 0.0,
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

fn latent_survival_map_exit_direction(
    direction: LatentSurvivalPrimaryDirection,
    event_type: LatentSurvivalEventType,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_exit,
        dqd: if matches!(event_type, LatentSurvivalEventType::ExactEvent) {
            direction.dqdot_exit
        } else {
            0.0
        },
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

/// Direction map for the interval-censored LEFT boundary state (mass `M_L =
/// exp(q_exit)`). The left boundary tracks the same `q_exit` time functional as
/// right-censoring (no hazard-derivative channel), plus the shared `mu`/`sigma`.
fn latent_survival_map_left_direction(
    direction: LatentSurvivalPrimaryDirection,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_exit,
        dqd: 0.0,
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

/// Direction map for the interval-censored RIGHT boundary state (mass `M_R =
/// exp(q_right)`). The right boundary tracks the dedicated `q_right` functional
/// (which shares the time-block coefficients with `q_exit` but is evaluated at
/// the interval upper bound `R`), plus the shared `mu`/`sigma`.
fn latent_survival_map_right_direction(
    direction: LatentSurvivalPrimaryDirection,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_right,
        dqd: 0.0,
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}
