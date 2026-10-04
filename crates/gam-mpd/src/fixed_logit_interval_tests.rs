use super::*;
use num_bigint::BigInt;
use num_rational::BigRational as Rational;

fn integer(value: i64) -> Rational {
    Rational::from_integer(BigInt::from(value))
}

/// Decode every significand/exponent bit into an exact rational, independently
/// of the runtime interval operations and vendor floating-point functions.
fn exact(value: f64) -> Rational {
    assert!(value.is_finite());
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let significand = if exponent == 0 {
        fraction
    } else {
        fraction | (1_u64 << 52)
    };
    let mut numerator = BigInt::from(significand);
    if bits >> 63 != 0 {
        numerator = -numerator;
    }
    let power = if exponent == 0 {
        -1074
    } else {
        exponent - 1023 - 52
    };
    if power >= 0 {
        Rational::from_integer(numerator << power as usize)
    } else {
        Rational::new(numerator, BigInt::from(1) << (-power) as usize)
    }
}

fn power_two_exact(exponent: i32) -> Rational {
    if exponent >= 0 {
        Rational::from_integer(BigInt::from(1) << exponent as usize)
    } else {
        Rational::new(BigInt::from(1), BigInt::from(1) << (-exponent) as usize)
    }
}

fn absolute(x: &Rational) -> Rational {
    if x < &integer(0) { -x } else { x.clone() }
}

fn atanh_reference(u: Rational, terms: usize) -> (Rational, Rational) {
    assert!(u >= integer(0) && u <= Rational::new(1.into(), 3.into()));
    let square = &u * &u;
    let mut power = u;
    let mut sum = integer(0);
    for term in 0..terms {
        sum += &power / integer((2 * term + 1) as i64);
        power *= &square;
    }
    let lower = sum * integer(2);
    let tail = integer(2) * power / (integer((2 * terms + 1) as i64) * (integer(1) - square));
    (lower.clone(), lower + tail)
}

fn rational_ln_two() -> (Rational, Rational) {
    atanh_reference(Rational::new(1.into(), 3.into()), 64)
}

fn log_reference(value: f64) -> (Rational, Rational) {
    let mut m = exact(value);
    assert!(m > integer(0));
    let mut exponent = 0;
    // Rational normalization, independent of the runtime bit decomposition.
    while m < integer(1) {
        m *= integer(2);
        exponent -= 1;
    }
    while m >= integer(2) {
        m /= integer(2);
        exponent += 1;
    }
    let u = (&m - integer(1)) / (&m + integer(1));
    let (lo, hi) = atanh_reference(u, 64);
    let (l2, h2) = rational_ln_two();
    if exponent >= 0 {
        (lo + l2 * integer(exponent), hi + h2 * integer(exponent))
    } else {
        (lo + h2 * integer(exponent), hi + l2 * integer(exponent))
    }
}

fn reduced_exp_reference(x: &Rational) -> (Rational, Rational) {
    let radius = absolute(x);
    assert!(radius <= integer(1));
    let mut term = integer(1);
    let mut sum = term.clone();
    for degree in 1..=32 {
        term = term * x / integer(degree);
        sum += &term;
    }
    let next = absolute(&term) * &radius / integer(33);
    let tail = next / (integer(1) - radius / integer(34));
    (&sum - &tail, sum + tail)
}

fn exp_reference(value: f64) -> (Rational, Rational) {
    let (l2, h2) = rational_ln_two();
    if exact(value) <= -integer(1074) * &h2 {
        return (integer(0), power_two_exact(-1074));
    }
    let exponent = (value / f64::from_bits(0x3fe6_2e42_fefa_39ef)).round() as i32;
    let k = integer(exponent as i64);
    let (a, b) = if exponent >= 0 {
        (&k * l2, &k * h2)
    } else {
        (&k * h2, &k * l2)
    };
    let (rlo, rhi) = (exact(value) - b, exact(value) - a);
    let lo = reduced_exp_reference(&rlo).0;
    let hi = reduced_exp_reference(&rhi).1;
    let factor = power_two_exact(exponent);
    (lo * &factor, hi * factor)
}

fn rational_product(a: &(Rational, Rational), b: &(Rational, Rational)) -> (Rational, Rational) {
    let products = [&a.0 * &b.0, &a.0 * &b.1, &a.1 * &b.0, &a.1 * &b.1];
    (
        products
            .iter()
            .min()
            .expect("four nonempty endpoint products")
            .clone(),
        products
            .iter()
            .max()
            .expect("four nonempty endpoint products")
            .clone(),
    )
}

/// Exact monotone search of positive binary64 bit patterns. No approximate
/// rational->f64 conversion supplies proof endpoints for the reference.
fn rational_float_bracket(value: &Rational) -> (f64, f64) {
    assert!(value >= &integer(0) && value <= &exact(f64::MAX));
    let (mut lo, mut hi) = (0_u64, f64::MAX.to_bits());
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if exact(f64::from_bits(mid)) <= *value {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let lower = f64::from_bits(lo);
    let upper = if exact(lower) == *value {
        lower
    } else {
        lower.next_up()
    };
    (lower, upper)
}

fn rational_log_interval(value: &(Rational, Rational)) -> (Rational, Rational) {
    let lo = rational_float_bracket(&value.0).0;
    let hi = rational_float_bracket(&value.1).1;
    (log_reference(lo).0, log_reference(hi).1)
}

fn kl_reference(z: &[f64], w: &[f64]) -> (Rational, Rational) {
    let mz = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mw = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut sz, mut sw, mut u) = (
        (integer(0), integer(0)),
        (integer(0), integer(0)),
        (integer(0), integer(0)),
    );
    for (p, q) in z.iter().zip(w) {
        let (a, b) = (exact(*p) - exact(mz), exact(*q) - exact(mw));
        let exponential = |value: &Rational| {
            assert!(value <= &integer(0));
            let (lo, hi) = rational_float_bracket(&absolute(value));
            (exp_reference(-hi).0, exp_reference(-lo).1)
        };
        let (ep, eq) = (exponential(&a), exponential(&b));
        let difference = a - b;
        let contribution = rational_product(&ep, &(difference.clone(), difference));
        sz.0 += &ep.0;
        sz.1 += &ep.1;
        sw.0 += &eq.0;
        sw.1 += &eq.1;
        u.0 += contribution.0;
        u.1 += contribution.1;
    }
    let expectation = rational_product(&u, &(integer(1) / &sz.1, integer(1) / &sz.0));
    let (lz, lw) = (rational_log_interval(&sz), rational_log_interval(&sw));
    (
        (expectation.0 + lw.0 - lz.1).max(integer(0)),
        expectation.1 + lw.1 - lz.0,
    )
}

#[test]
fn fixed_logit_kl_encloses_independent_rational_reference_and_exact_shifts() {
    for (z, w) in [
        (vec![0.0, 1.0, -1.0], vec![0.5, -0.5, 1.5]),
        (vec![0.0, -2000.0], vec![-2000.0, 0.0]),
        (vec![0.0, -745.0], vec![0.0, -744.0]),
        (vec![1.0, 1.0], vec![0.0, 2.0]),
        (vec![0.1, 0.2], vec![0.3, 0.4]),
        (
            vec![f64::MAX, f64::MAX],
            vec![f64::MAX, f64::MAX.next_down()],
        ),
    ] {
        assert_contains_reference(kl_logits(&z, &w), kl_reference(&z, &w));
    }
    assert_eq!(
        kl_logits(&[1.0, 2.0, 3.0], &[5.0, 6.0, 7.0]),
        Enclosure::Bounded(Interval::point(0.0))
    );
    assert_eq!(
        kl_logits(&[f64::MAX, -f64::MAX], &[f64::MAX, -f64::MAX]),
        Enclosure::Bounded(Interval::point(0.0))
    );
    // Non-exact subtraction must still enclose a legitimate fixed-input KL.
    let x = kl_logits(&[0.1, 0.2], &[0.3, 0.4]);
    assert!(matches!(x,Enclosure::Bounded(i) if i.lo>=0.0 && i.hi<1e-10));
    assert_eq!(
        kl_logits(&[], &[]),
        Enclosure::Unresolved(Reason::EmptyOrMismatchedRows)
    );
    assert_eq!(
        kl_logits(&[0.0], &[0.0, 1.0]),
        Enclosure::Unresolved(Reason::EmptyOrMismatchedRows)
    );
    assert_eq!(
        kl_logits(&[f64::NAN], &[0.0]),
        Enclosure::Unresolved(Reason::NonFiniteInput)
    );
    assert!(matches!(
        kl_logits(&[f64::MAX, -f64::MAX], &[-f64::MAX, f64::MAX]),
        Enclosure::Unresolved(..)
    ));
}

fn assert_contains_reference(result: Enclosure, reference: (Rational, Rational)) {
    let bounds = match result {
        Enclosure::Bounded(bounds) => Some(bounds),
        Enclosure::Unresolved(..) => None,
    }
    .expect("expected enclosure");
    assert!(
        bounds.lo == f64::NEG_INFINITY
            || (bounds.lo.is_finite() && exact(bounds.lo) <= reference.0),
        "lower endpoint outside exact rational certificate: {bounds:?}"
    );
    assert!(
        bounds.hi == f64::INFINITY || (bounds.hi.is_finite() && exact(bounds.hi) >= reference.1),
        "upper endpoint outside exact rational certificate: {bounds:?}"
    );
}

#[test]
fn adjacent_ln_two_endpoints_have_exact_rational_certificate() {
    let bounds = ln_two();
    let (lo, hi) = rational_ln_two();
    assert_eq!(bounds.lo.next_up().to_bits(), bounds.hi.to_bits());
    assert!(exact(bounds.lo) < lo && hi < exact(bounds.hi));
    assert_contains_reference(log(2.0), (lo, hi));
}

#[test]
fn log_encloses_independent_rational_series_across_binary_range() {
    for value in [
        f64::from_bits(1),
        f64::from_bits(2),
        f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE,
        0.125,
        0.5,
        1.0_f64.next_down(),
        1.0,
        1.0_f64.next_up(),
        1.5,
        2.0_f64.next_down(),
        2.0,
        2.0_f64.next_up(),
        50_257.0,
        f64::MAX,
    ] {
        assert_contains_reference(log(value), log_reference(value));
    }
}

#[test]
fn exp_encloses_rational_certificate_at_reduction_and_underflow_boundaries() {
    let cutoff = ln_two().scale(-1074.0);
    let mut values = vec![
        -745.0,
        -744.0,
        -710.0,
        -709.0,
        -1.0,
        -0.5,
        -f64::from_bits(1),
        0.0,
        f64::from_bits(1),
        0.5,
        1.0,
        10.0,
        709.0,
        cutoff.lo.next_down(),
        cutoff.lo,
        cutoff.lo.next_up(),
        cutoff.hi,
    ];
    for k in [-1024.0, -10.0, -1.0, 1.0, 10.0, 1023.0] {
        let boundary = k * ln_two().lo;
        values.extend([boundary.next_down(), boundary, boundary.next_up()]);
    }
    for value in values {
        assert_contains_reference(exp(value), exp_reference(value));
    }
}

#[test]
fn invalid_and_unrepresentable_inputs_are_explicitly_unresolved() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(exp(value), Enclosure::Unresolved(Reason::NonFiniteInput));
        assert_eq!(log(value), Enclosure::Unresolved(Reason::NonFiniteInput));
    }
    for value in [0.0, -0.0, -1.0, -f64::MAX] {
        assert_eq!(
            log(value),
            Enclosure::Unresolved(Reason::NonPositiveLogInput)
        );
    }
    assert!(matches!(
        exp(f64::MAX),
        Enclosure::Unresolved(Reason::ReductionNotEnclosed)
    ));
    assert!(matches!(
        exp(710.0),
        Enclosure::Unresolved(Reason::UnboundedBinary64Endpoints)
    ));
    assert_eq!(
        exp(-f64::MAX),
        Enclosure::Bounded(Interval::new(0.0, f64::from_bits(1)))
    );
}

#[test]
fn basic_interval_operations_enclose_exact_rational_rounding_and_subnormals() {
    for (a, b) in [
        (1.0, f64::EPSILON / 2.0),
        (f64::MAX, -f64::MAX),
        (f64::from_bits(1), 0.5),
        (f64::MIN_POSITIVE, 1.0_f64.next_down()),
        (-7.0, 3.0),
        (1.0e-200, 1.0e-100),
    ] {
        let (x, y) = (Interval::point(a), Interval::point(b));
        let add = exact(a) + exact(b);
        assert_contains_reference(Enclosure::Bounded(x.add(y)), (add.clone(), add));
        let mul = exact(a) * exact(b);
        assert_contains_reference(Enclosure::Bounded(x.mul(y)), (mul.clone(), mul));
        if b > 0.0 {
            let div = exact(a) / exact(b);
            assert_contains_reference(Enclosure::Bounded(x.div_positive(y)), (div.clone(), div));
        }
    }
}

#[test]
fn deterministic_bit_corpus_matches_independent_exact_rational_enclosures() {
    let mut state = 0x9516_b72f_3019_cadb_u64;
    for step in 0..24 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = f64::from_bits((state & 0x7fef_ffff_ffff_ffff).max(1));
        assert_contains_reference(log(value), log_reference(value));
        let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
        let argument = -745.0 + 1454.0 * unit;
        assert_contains_reference(exp(argument), exp_reference(argument));
        let a = if step % 2 == 0 { value } else { -value };
        let b = f64::from_bits(((state.rotate_left(29)) & 0x7fef_ffff_ffff_ffff).max(1));
        let exact_sum = exact(a) + exact(b);
        assert_contains_reference(
            Enclosure::Bounded(Interval::point(a).add(Interval::point(b))),
            (exact_sum.clone(), exact_sum),
        );
        let exact_product = exact(a) * exact(b);
        assert_contains_reference(
            Enclosure::Bounded(Interval::point(a).mul(Interval::point(b))),
            (exact_product.clone(), exact_product),
        );
    }
}
