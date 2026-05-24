#[derive(Clone, Copy, Debug)]
pub struct PowerLawFit {
    pub alpha: f64,
    pub a: f64,
    pub r2: f64,
    pub max_abs_log_resid: f64,
    pub n_points: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetVerdict {
    Fits,
    OverBudget,
}

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
    let denom_scale = (n * sxx).abs().max((sx * sx).abs()).max(1.0);
    if !denom.is_finite() || denom.abs() < 1e-12 * denom_scale {
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
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };
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

pub fn report_power_law_full(
    tag: &str,
    points: &[(f64, f64)],
    extrapolate: &[(&str, f64)],
    budget_y: f64,
) -> Option<PowerLawFit> {
    let Some(fit) = fit_power_law(points) else {
        eprintln!(
            "{tag} INSUFFICIENT DATA: {} points (need >=3 for an honest fit)",
            points.len()
        );
        return None;
    };
    eprintln!(
        "{tag} fit: y ~= {:.3e} * x^{:.3}  | R^2={:.4}  max|log-resid|={:.3} (x{:.2})  | n_points={}",
        fit.a,
        fit.alpha,
        fit.r2,
        fit.max_abs_log_resid,
        fit.max_abs_log_resid.exp(),
        fit.n_points,
    );
    if fit.r2 < 0.85 || fit.max_abs_log_resid > 0.5 {
        eprintln!(
            "{tag} REFUSING EXTRAPOLATION: fit quality insufficient (R^2 < 0.85 or max-resid > 0.5 in log-space)."
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
                " [extrapolating {:.1}x past max calibration x={:.1e}]",
                stretch, max_x
            )
        } else if *x_target < min_x {
            format!(" [extrapolating below min calibration x={:.1e}]", min_x)
        } else {
            String::new()
        };
        let verdict = if pred <= budget_y {
            BudgetVerdict::Fits
        } else {
            BudgetVerdict::OverBudget
        };
        let verdict_str = match verdict {
            BudgetVerdict::Fits => format!("FITS ({:.0}x headroom)", budget_y / pred.max(1e-300)),
            BudgetVerdict::OverBudget => {
                format!("OVER BUDGET ({:.1}x)", pred / budget_y.max(1e-300))
            }
        };
        eprintln!(
            "{tag} extrap @ {label} (x={:.3e}): pred={:.3e}{stretch_note} -> {verdict_str}",
            x_target, pred
        );
    }
    Some(fit)
}
