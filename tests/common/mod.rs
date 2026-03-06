pub fn assert_manual_with_ad_band(
    case: &str,
    x: f64,
    quantity: &str,
    manual: f64,
    refs: &[(&str, f64)],
) {
    assert!(!refs.is_empty(), "refs must be non-empty");
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut nearest_name = refs[0].0;
    let mut nearest_v = refs[0].1;
    let mut nearest_abs = (manual - refs[0].1).abs();
    for (name, v) in refs {
        min_v = min_v.min(*v);
        max_v = max_v.max(*v);
        let abs = (manual - *v).abs();
        if abs < nearest_abs {
            nearest_abs = abs;
            nearest_name = name;
            nearest_v = *v;
        }
    }
    let band = (max_v - min_v).abs();
    let scale = manual.abs().max(nearest_v.abs()).max(1.0);
    let roundoff = 64.0 * f64::EPSILON * scale;
    if nearest_abs > band + roundoff {
        panic!(
            "{case} x={x:.6} {quantity}: manual={manual:.16e} nearest({nearest_name})={nearest_v:.16e} abs_err={nearest_abs:.3e} ad_band={band:.3e} roundoff={roundoff:.3e}"
        );
    }
}

#[macro_export]
macro_rules! assert_manual_ad_band {
    ($case:expr, $x:expr, $quantity:expr, $manual:expr, $( $name:expr => $value:expr ),+ $(,)?) => {{
        let refs = [$(($name, $value)),+];
        crate::common::assert_manual_with_ad_band($case, $x, $quantity, $manual, &refs);
    }};
}
