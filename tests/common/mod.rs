pub fn assert_manualwith_ad_band(
    case: &str,
    x: f64,
    quantity: &str,
    manual: f64,
    refs: &[(&str, f64)],
) {
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
}

#[macro_export]
macro_rules! assert_manual_ad_band {
    ($case:expr, $x:expr, $quantity:expr, $manual:expr, $( $name:expr => $value:expr ),+ $(,)?) => {{
        let refs = [$(($name, $value)),+];
        $crate::common::assert_manualwith_ad_band($case, $x, $quantity, $manual, &refs);
    }};
}

/// Test sizing mode — controls how aggressive integration tests should be.
///
/// `Fast` is the default for CI: small `n`, single seed, tolerance bounds
/// that are realistic for that sample size.
///
/// `Hard` is opt-in via `GAM_TEST_MODE=hard`. It enables larger sample
/// sizes, multiple seeds, and tighter tolerance bounds. Hard mode is
/// intended for nightly runs and pre-release validation — not every PR.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TestMode {
    Fast,
    Hard,
}

impl TestMode {
    /// Read the mode from `GAM_TEST_MODE`. Anything other than `hard`
    /// (case-insensitive) means `Fast`.
    pub fn from_env() -> Self {
        match std::env::var("GAM_TEST_MODE").as_deref() {
            Ok(v) if v.eq_ignore_ascii_case("hard") => TestMode::Hard,
            _ => TestMode::Fast,
        }
    }

    pub fn is_hard(self) -> bool {
        matches!(self, TestMode::Hard)
    }

    /// Pick a value depending on the current mode.
    pub fn select<T>(self, fast: T, hard: T) -> T {
        match self {
            TestMode::Fast => fast,
            TestMode::Hard => hard,
        }
    }
}
