use super::*;
use num_bigint::BigInt;
use num_rational::BigRational as R;

fn rational(x: f64) -> R {
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32;
    let mantissa = (bits & ((1_u64 << 52)-1)) | if exponent == 0 { 0 } else { 1_u64 << 52 };
    let power = if exponent == 0 { -1074 } else { exponent-1023-52 };
    let signed = BigInt::from(mantissa) * if x.is_sign_negative() { -1 } else { 1 };
    if power >= 0 { R::from_integer(signed << power as usize) }
    else { R::new(signed, BigInt::from(1) << (-power) as usize) }
}
fn bound(x: Enclosure) -> Interval {
    // SAFETY: test-only assertion helper; an unresolved enclosure must fail this fixture.
    match x { Enclosure::Bounded(v) => v, other => panic!("{other:?}") }
}

#[test]
fn zero_requires_exact_identity_or_common_shift() {
    assert_eq!(bound(total_variation(&[1.,-2.,3.], &[5.,2.,7.])), Interval::point(0.));
    assert_eq!(bound(shared_prediction_obstruction(&[1.], &[-99.])), Interval::point(0.));
    assert!(bound(total_variation(&[0.,1.], &[1.,0.])).lo > 0.);
}

#[test]
fn asymmetric_pair_matches_independent_rational_exponential() {
    // e lies between its degree-40 rational series and that series plus the
    // first omitted term / (1-1/42). TV for reversed [0,1] logits is (e-1)/(e+1).
    let one = R::from_integer(1.into());
    let mut term = one.clone();
    let mut sum = one.clone();
    for k in 1..=40 { term /= R::from_integer(k.into()); sum += &term; }
    let upper = &sum + (&term / R::from_integer(41.into())) / (&one - R::new(1.into(),42.into()));
    let lo = (&sum-&one)/(&sum+&one);
    let hi = (&upper-&one)/(&upper+&one);
    let tv = bound(total_variation(&[0.,1.], &[1.,0.]));
    assert!(rational(tv.lo) <= lo && rational(tv.hi) >= hi);
    let floor = bound(shared_prediction_obstruction(&[0.,1.], &[1.,0.]));
    assert!(rational(floor.lo) <= &lo*&lo/R::from_integer(2.into()));
    assert!(rational(floor.hi) >= &hi*&hi/R::from_integer(2.into()));
    assert_eq!(total_variation(&[0.,1.], &[1.,0.]), total_variation(&[1.,0.], &[0.,1.]));
}

#[test]
fn extreme_finite_logits_keep_absolute_tail_and_range() {
    let tv = bound(total_variation(&[f64::MAX,-f64::MAX], &[-f64::MAX,f64::MAX]));
    assert!(tv.lo > 0.99 && tv.hi == 1.);
    let floor = bound(shared_prediction_obstruction(&[1000.,-1000.], &[-1000.,1000.]));
    assert!(floor.lo > 0.49 && floor.hi <= 0.5);
}

#[test]
fn malformed_inputs_stay_unknown() {
    for (a,b) in [(vec![],vec![]),(vec![0.],vec![0.,1.]),(vec![f64::NAN],vec![0.]),(vec![0.],vec![f64::INFINITY])] {
        assert!(matches!(shared_prediction_obstruction(&a,&b),Enclosure::Unresolved(_)));
    }
}
