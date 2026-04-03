const BRENT_CGOLD: f64 = 0.381_966_011_250_105_1;
const BRENT_ZEPS: f64 = 1.0e-12;

pub(crate) struct PositiveProfileOptimum<T> {
    pub(crate) point: f64,
    pub(crate) objective: f64,
    pub(crate) payload: T,
}

pub(crate) fn minimize_positive_profile<T, F>(
    label: &str,
    initial_point: f64,
    lower_bound: f64,
    upper_bound: f64,
    rel_tol: f64,
    max_probe_expansions: usize,
    max_refine_iters: usize,
    mut evaluate: F,
) -> Result<PositiveProfileOptimum<T>, String>
where
    F: FnMut(f64) -> Result<(f64, T), String>,
{
    if !initial_point.is_finite() || initial_point <= 0.0 {
        return Err(format!(
            "{label}: positive scalar optimizer requires finite initial point > 0, got {initial_point}"
        ));
    }
    if !lower_bound.is_finite()
        || !upper_bound.is_finite()
        || lower_bound <= 0.0
        || upper_bound <= lower_bound
    {
        return Err(format!(
            "{label}: invalid positive scalar bounds [{lower_bound}, {upper_bound}]"
        ));
    }

    let log_lower = lower_bound.ln();
    let log_upper = upper_bound.ln();
    let log_tol = rel_tol.max(1.0e-4).ln_1p();

    let mut best: Option<PositiveProfileOptimum<T>> = None;
    let mut last_error: Option<String> = None;

    let mut eval_at = |log_point: f64| -> f64 {
        let sigma = log_point.exp();
        match evaluate(sigma) {
            Ok((objective, payload)) => {
                log::debug!("{label}: sigma={sigma:.6}, objective={objective:.6e}");
                if objective.is_finite() {
                    let replace_best = best
                        .as_ref()
                        .is_none_or(|current| objective < current.objective);
                    if replace_best {
                        best = Some(PositiveProfileOptimum {
                            point: sigma,
                            objective,
                            payload,
                        });
                    }
                    objective
                } else {
                    log::debug!(
                        "{label}: sigma={sigma:.6} produced non-finite objective {objective}"
                    );
                    f64::INFINITY
                }
            }
            Err(err) => {
                log::debug!("{label}: sigma={sigma:.6} failed: {err}");
                last_error = Some(err);
                f64::INFINITY
            }
        }
    };

    let mut x = initial_point.clamp(lower_bound, upper_bound).ln();
    let mut fx = eval_at(x);
    let mut step = ((log_upper - log_lower) * 0.15).clamp(0.15, 0.8);
    let mut bracket_a = log_lower;
    let mut bracket_b = log_upper;

    for _ in 0..max_probe_expansions {
        let left = (x - step).max(log_lower);
        let right = (x + step).min(log_upper);
        if (left - x).abs() < BRENT_ZEPS && (right - x).abs() < BRENT_ZEPS {
            break;
        }

        let fl = if (left - x).abs() < BRENT_ZEPS {
            f64::INFINITY
        } else {
            eval_at(left)
        };
        let fr = if (right - x).abs() < BRENT_ZEPS {
            f64::INFINITY
        } else {
            eval_at(right)
        };

        if fl >= fx && fr >= fx {
            bracket_a = left;
            bracket_b = right;
            break;
        }

        if fl < fx && fl <= fr {
            bracket_a = left;
            x = left;
            fx = fl;
            if (x - log_lower).abs() < BRENT_ZEPS {
                bracket_b = right.max(x);
                break;
            }
        } else if fr < fx {
            bracket_b = right;
            x = right;
            fx = fr;
            if (x - log_upper).abs() < BRENT_ZEPS {
                bracket_a = left.min(x);
                break;
            }
        } else {
            bracket_a = left;
            bracket_b = right;
            break;
        }

        step = (step * 1.8).min(log_upper - log_lower);
        bracket_a = bracket_a.min(x);
        bracket_b = bracket_b.max(x);
    }

    if bracket_b <= bracket_a {
        bracket_a = log_lower;
        bracket_b = log_upper;
    }

    x = x.clamp(bracket_a, bracket_b);
    let mut w = x;
    let mut v = x;
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0f64;
    let mut e = 0.0f64;

    for _ in 0..max_refine_iters {
        let midpoint = 0.5 * (bracket_a + bracket_b);
        let tol1 = log_tol * x.abs().max(1.0) + BRENT_ZEPS;
        let tol2 = 2.0 * tol1;
        if (x - midpoint).abs() <= tol2 - 0.5 * (bracket_b - bracket_a) {
            break;
        }

        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q2 = 2.0 * (q - r);
            if q2 > 0.0 {
                p = -p;
            } else {
                q2 = -q2;
            }
            let etemp = e;
            e = d;

            if p.abs() < 0.5 * q2 * etemp.abs()
                && p > q2 * (bracket_a - x)
                && p < q2 * (bracket_b - x)
            {
                d = p / q2;
                let candidate = x + d;
                if (candidate - bracket_a) < tol2 || (bracket_b - candidate) < tol2 {
                    d = if midpoint >= x { tol1 } else { -tol1 };
                }
            } else {
                e = if x >= midpoint {
                    bracket_a - x
                } else {
                    bracket_b - x
                };
                d = BRENT_CGOLD * e;
            }
        } else {
            e = if x >= midpoint {
                bracket_a - x
            } else {
                bracket_b - x
            };
            d = BRENT_CGOLD * e;
        }

        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        }
        .clamp(bracket_a, bracket_b);
        let fu = eval_at(u);

        if fu <= fx {
            if u >= x {
                bracket_a = x;
            } else {
                bracket_b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                bracket_a = u;
            } else {
                bracket_b = u;
            }
            if fu <= fw || (w - x).abs() < BRENT_ZEPS {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < BRENT_ZEPS || (v - w).abs() < BRENT_ZEPS {
                v = u;
                fv = fu;
            }
        }
    }

    best.ok_or_else(|| {
        last_error.unwrap_or_else(|| format!("{label}: no valid profile evaluation"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_positive_profile_recovers_logspace_quadratic_minimum() {
        let optimum =
            minimize_positive_profile("unit quadratic", 0.5, 0.01, 5.0, 0.01, 8, 32, |sigma| {
                let target = 0.7f64.ln();
                let x = sigma.ln();
                Ok(((x - target).powi(2), sigma))
            })
            .expect("optimizer should converge");

        assert!((optimum.point - 0.7).abs() < 0.05);
        assert!(optimum.objective < 1.0e-3);
    }

    #[test]
    fn minimize_positive_profile_tolerates_invalid_probes() {
        let optimum =
            minimize_positive_profile("invalid probe", 0.5, 0.01, 5.0, 0.01, 8, 32, |sigma| {
                if sigma < 0.2 {
                    return Err("domain failure".to_string());
                }
                let target = 0.9f64.ln();
                let x = sigma.ln();
                Ok(((x - target).powi(2), sigma))
            })
            .expect("optimizer should ignore invalid probes");

        assert!((optimum.point - 0.9).abs() < 0.08);
    }
}
