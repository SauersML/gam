#[allow(dead_code)]
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

/// Result of a log-log OLS fit `y ≈ a · x^α`.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct PowerLawFit {
    pub alpha: f64,
    pub a: f64,
    pub r2: f64,
    pub max_abs_log_resid: f64,
    pub n_points: usize,
}

/// Fit `y = a · x^α` to `(x, y)` pairs via log-log OLS. Returns `None`
/// when there are fewer than 3 points (insufficient for an honest
/// fit). Used by the scaling-law probes (`tests/standard_gam_scaling.rs`,
/// `tests/margslope_inner_pirls_scaling.rs`) where the consistency of
/// the fit's R² and max log-residual gates extrapolation to biobank
/// shape — see `report_power_law` for the policy.
#[allow(dead_code)]
pub fn fit_power_law(points: &[(f64, f64)]) -> Option<PowerLawFit> {
    if points.len() < 3 {
        return None;
    }
    let logs: Vec<(f64, f64)> = points.iter().map(|(x, y)| (x.ln(), y.ln())).collect();
    let n = logs.len() as f64;
    let sx: f64 = logs.iter().map(|(x, _)| x).sum();
    let sy: f64 = logs.iter().map(|(_, y)| y).sum();
    let sxx: f64 = logs.iter().map(|(x, _)| x * x).sum();
    let sxy: f64 = logs.iter().map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if !denom.is_finite() || denom.abs() < 1e-30 {
        return None;
    }
    let alpha = (n * sxy - sx * sy) / denom;
    let log_a = (sy - alpha * sx) / n;
    let a = log_a.exp();
    let mean_y = sy / n;
    let ss_tot: f64 = logs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = logs
        .iter()
        .map(|(x, y)| {
            let pred = log_a + alpha * x;
            (y - pred).powi(2)
        })
        .sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let max_abs_log_resid: f64 = logs
        .iter()
        .map(|(x, y)| (y - (log_a + alpha * x)).abs())
        .fold(0.0_f64, f64::max);
    Some(PowerLawFit {
        alpha,
        a,
        r2,
        max_abs_log_resid,
        n_points: logs.len(),
    })
}

/// Fit a power law and report it with extrapolation verdicts. Refuses
/// to extrapolate when `R² < 0.85` or `max |log-residual| > 0.5` —
/// the honest-fit gate the mission's "MEASURE FIRST" rule requires.
/// Returns the fit when extrapolation was published, `None` otherwise.
///
/// Output format:
///   {tag} fit: y ≈ a · x^α  | R²=…  max|log-resid|=…  | n_points=N
///   {tag} REFUSING EXTRAPOLATION ...                  (when fit poor)
///   {tag} budget: {budget_y}                          (when usable)
///   {tag} extrap @ <label> (x=…): pred=…[stretch_note] → <verdict>
///
/// `stretch_note` flags extrapolations >5× past the calibration max,
/// or below the calibration min, so a reader can cross-check the
/// extrapolation distance against the fit confidence.
#[allow(dead_code)]
pub fn report_power_law(
    tag: &str,
    points: &[(f64, f64)],
    extrapolate: &[(&str, f64)],
    budget_y: f64,
) -> Option<PowerLawFit> {
    let Some(fit) = fit_power_law(points) else {
        eprintln!(
            "{tag} INSUFFICIENT DATA: {} points (need ≥3 for an honest fit)",
            points.len()
        );
        return None;
    };
    eprintln!(
        "{tag} fit: y ≈ {:.3e} · x^{:.3}  | R²={:.4}  max|log-resid|={:.3} (×{:.2})  | n_points={}",
        fit.a,
        fit.alpha,
        fit.r2,
        fit.max_abs_log_resid,
        fit.max_abs_log_resid.exp(),
        fit.n_points,
    );
    if fit.r2 < 0.85 || fit.max_abs_log_resid > 0.5 {
        eprintln!(
            "{tag} REFUSING EXTRAPOLATION: fit quality insufficient (R² < 0.85 or max-resid > 0.5 in log-space, i.e. >65% off in y). Need cleaner data — likely the test setup is hitting an outer-iter cap or the problem geometry varies across n."
        );
        return None;
    }
    eprintln!("{tag} budget: {:.1}", budget_y);
    let max_x: f64 = points.iter().map(|(x, _)| *x).fold(0.0_f64, f64::max);
    let min_x: f64 = points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    for (label, x_target) in extrapolate {
        let pred = fit.a * x_target.powf(fit.alpha);
        let stretch = x_target / max_x;
        let stretch_note = if stretch > 5.0 {
            format!(
                " [extrapolating {:.1}× past max calibration x={:.1e}]",
                stretch, max_x
            )
        } else if *x_target < min_x {
            format!(" [extrapolating below min calibration x={:.1e}]", min_x)
        } else {
            String::new()
        };
        let verdict = if pred <= budget_y {
            format!("FITS ({:.0}× headroom)", budget_y / pred)
        } else {
            format!(
                "OVER BUDGET by {:.0}s ({:.1} min, {:.2}× over)",
                pred - budget_y,
                (pred - budget_y) / 60.0,
                pred / budget_y
            )
        };
        eprintln!(
            "{tag} extrap @ {label} (x={:.1e}): pred={:.1}s ({:.2} min){} → {}",
            x_target,
            pred,
            pred / 60.0,
            stretch_note,
            verdict,
        );
    }
    Some(fit)
}
