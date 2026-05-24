#[macro_export]
macro_rules! assert_manual_ad_band {
    ($case:expr, $x:expr, $quantity:expr, $manual:expr, $( $name:expr => $value:expr ),+ $(,)?) => {{
        let case: &str = $case;
        let x: f64 = $x;
        let quantity: &str = $quantity;
        let manual: f64 = $manual;
        let refs: &[(&str, f64)] = &[$(($name, $value)),+];
        assert!(!refs.is_empty(), "refs must be non-empty");
        let mut minv = f64::INFINITY;
        let mut maxv = f64::NEG_INFINITY;
        let mut nearestname = refs[0].0;
        let mut nearestv = refs[0].1;
        let mut nearest_abs = (manual - refs[0].1).abs();
        for (name, v) in refs {
            minv = minv.min(*v);
            maxv = maxv.max(*v);
            let abs = (manual - *v).abs();
            if abs < nearest_abs {
                nearest_abs = abs;
                nearestname = name;
                nearestv = *v;
            }
        }
        let band = (maxv - minv).abs();
        let scale = manual.abs().max(nearestv.abs()).max(1.0);
        let roundoff = 64.0 * f64::EPSILON * scale;
        if nearest_abs > band + roundoff {
            panic!(
                "{case} x={x:.6} {quantity}: manual={manual:.16e} nearest({nearestname})={nearestv:.16e} abs_err={nearest_abs:.3e} ad_band={band:.3e} roundoff={roundoff:.3e}"
            );
        }
    }};
}
