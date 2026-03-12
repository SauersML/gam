use crate::linalg::utils::KahanSum;
use faer::MatRef;
use ndarray::{Array1, Array2};

const KAHAN_SWITCH_ELEMS: usize = 10_000;
const FD_REL_GAP_THRESHOLD: f64 = 0.2;
const FD_MIN_BASE_STEP: f64 = 1e-6;
const FD_MAX_REFINEMENTS: usize = 4;
const FD_RIDGE_REL_JITTER_THRESHOLD: f64 = 1e-3;
const FD_RIDGE_ABS_JITTER_THRESHOLD: f64 = 1e-12;

pub(crate) fn faer_frob_inner(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let (m, n) = (a.nrows(), a.ncols());
    let elem_count = m.saturating_mul(n);
    if elem_count < KAHAN_SWITCH_ELEMS {
        let mut sum = 0.0_f64;
        for j in 0..n {
            for i in 0..m {
                sum += a[(i, j)] * b[(i, j)];
            }
        }
        sum
    } else {
        let mut sum = KahanSum::default();
        for j in 0..n {
            for i in 0..m {
                sum.add(a[(i, j)] * b[(i, j)]);
            }
        }
        sum.sum()
    }
}

pub(crate) fn kahan_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut acc = KahanSum::default();
    for value in iter {
        acc.add(value);
    }
    acc.sum()
}

pub(crate) fn smooth_floor_dp(dp: f64, dp_floor: f64, dp_floor_smooth_width: f64) -> (f64, f64) {
    let tau = dp_floor_smooth_width.max(f64::EPSILON);
    let scaled = (dp - dp_floor) / tau;

    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = dp_floor + tau * softplus;
    (dp_c, sigma)
}

pub(crate) fn validate_full_size_penalties<E, F>(
    s_list: &[Array2<f64>],
    p: usize,
    context: &str,
    make_error: F,
) -> Result<(), E>
where
    F: Fn(String) -> E,
{
    for (idx, s) in s_list.iter().enumerate() {
        let (rows, cols) = s.dim();
        if rows != p || cols != p {
            return Err(make_error(format!(
                "{context}: penalty matrix {idx} must be {p}x{p}, got {rows}x{cols}"
            )));
        }
    }
    Ok(())
}

pub(crate) trait FdGradientState<E> {
    fn compute_cost(&self, rho: &Array1<f64>) -> Result<f64, E>;
    fn compute_gradient(&self, rho: &Array1<f64>) -> Result<Array1<f64>, E>;
    fn last_ridge_used(&self) -> Option<f64>;
}

struct FdEval {
    f_p: f64,
    f_m: f64,
    f_p2: f64,
    f_m2: f64,
    d_small: f64,
    d_big: f64,
    ridge_min: f64,
    ridge_max: f64,
    ridge_rel_span: f64,
    ridge_jitter: bool,
}

fn evaluate_fd_pair<S, E>(
    reml_state: &S,
    rho: &Array1<f64>,
    coord: usize,
    base_h: f64,
) -> Result<FdEval, E>
where
    S: FdGradientState<E>,
{
    let mut rho_p = rho.clone();
    rho_p[coord] += 0.5 * base_h;
    let mut rho_m = rho.clone();
    rho_m[coord] -= 0.5 * base_h;
    let f_p = reml_state.compute_cost(&rho_p)?;
    let ridge_p = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m = reml_state.compute_cost(&rho_m)?;
    let ridge_m = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_small = (f_p - f_m) / base_h;

    let h2 = 2.0 * base_h;
    let mut rho_p2 = rho.clone();
    rho_p2[coord] += 0.5 * h2;
    let mut rho_m2 = rho.clone();
    rho_m2[coord] -= 0.5 * h2;
    let f_p2 = reml_state.compute_cost(&rho_p2)?;
    let ridge_p2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m2 = reml_state.compute_cost(&rho_m2)?;
    let ridge_m2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_big = (f_p2 - f_m2) / h2;

    let finite_ridges: Vec<f64> = [ridge_p, ridge_m, ridge_p2, ridge_m2]
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    let (ridge_min, ridge_max, ridge_span, ridge_rel_span) = if finite_ridges.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for v in finite_ridges {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let span = max_v - min_v;
        let rel = span / max_v.abs().max(1e-12);
        (min_v, max_v, span, rel)
    };
    let ridge_jitter = ridge_span.is_finite()
        && ridge_rel_span.is_finite()
        && (ridge_span > FD_RIDGE_ABS_JITTER_THRESHOLD
            && ridge_rel_span > FD_RIDGE_REL_JITTER_THRESHOLD);

    Ok(FdEval {
        f_p,
        f_m,
        f_p2,
        f_m2,
        d_small,
        d_big,
        ridge_min,
        ridge_max,
        ridge_rel_span,
        ridge_jitter,
    })
}

fn fd_same_sign(d_small: f64, d_big: f64) -> bool {
    if !d_small.is_finite() || !d_big.is_finite() {
        false
    } else {
        (d_small >= 0.0 && d_big >= 0.0) || (d_small <= 0.0 && d_big <= 0.0)
    }
}

fn select_fd_derivative(d_small: f64, d_big: f64, same_sign: bool) -> f64 {
    match (d_small.is_finite(), d_big.is_finite()) {
        (true, true) => {
            if same_sign {
                d_small
            } else {
                d_big
            }
        }
        (true, false) => d_small,
        (false, true) => d_big,
        (false, false) => 0.0,
    }
}

pub(crate) fn compute_fd_gradient<S, E>(
    reml_state: &S,
    rho: &Array1<f64>,
    emit_logs: bool,
    allow_analytic_fallback: bool,
) -> Result<Array1<f64>, E>
where
    S: FdGradientState<E>,
{
    let mut fd_grad = Array1::zeros(rho.len());
    let mut analytic_fallback: Option<Array1<f64>> = None;

    let mut log_lines: Vec<String> = Vec::new();
    let (rho_min, rho_max) = rho
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let rho_summary = format!("len={} range=[{:.3e},{:.3e}]", rho.len(), rho_min, rho_max);
    match reml_state.last_ridge_used() {
        Some(ridge) => log_lines.push(format!(
            "[FD RIDGE] Baseline cached ridge: {ridge:.3e} for rho {rho_summary}",
        )),
        None => log_lines.push(format!(
            "[FD RIDGE] No cached baseline ridge available for rho {rho_summary}",
        )),
    }

    for i in 0..rho.len() {
        let h_rel = 1e-4_f64 * (1.0 + rho[i].abs());
        let h_abs = 1e-5_f64;
        let mut base_h = h_rel.max(h_abs);

        log_lines.push(format!("[FD RIDGE] coord {i} rho={:+.6e}", rho[i]));

        let mut d_small = 0.0;
        let mut d_big = 0.0;
        let mut derivative: Option<f64> = None;
        let mut best_rel_gap = f64::INFINITY;
        let mut best_derivative: Option<f64> = None;
        let mut last_rel_gap = f64::INFINITY;
        let mut refine_steps = 0usize;
        let mut rel_gap_first = None;
        let mut rel_gap_max = 0.0;
        let mut ridge_jitter_seen = false;
        let mut ridge_rel_span_max = 0.0;
        let h_start = base_h;

        for attempt in 0..=FD_MAX_REFINEMENTS {
            let eval = evaluate_fd_pair(reml_state, rho, i, base_h)?;
            d_small = eval.d_small;
            d_big = eval.d_big;
            ridge_jitter_seen |= eval.ridge_jitter;
            if eval.ridge_rel_span.is_finite() && eval.ridge_rel_span > ridge_rel_span_max {
                ridge_rel_span_max = eval.ridge_rel_span;
            }

            let denom = d_small.abs().max(d_big.abs()).max(1e-12);
            let rel_gap = (d_small - d_big).abs() / denom;
            let same_sign = fd_same_sign(d_small, d_big);

            if same_sign && !eval.ridge_jitter {
                if rel_gap <= best_rel_gap {
                    best_rel_gap = rel_gap;
                    best_derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
                }
                if rel_gap > last_rel_gap {
                    derivative = best_derivative;
                    break;
                }
                last_rel_gap = rel_gap;
            }

            let refine_for_rel_gap =
                same_sign && rel_gap > FD_REL_GAP_THRESHOLD && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refine_for_ridge = eval.ridge_jitter && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refining = refine_for_rel_gap || refine_for_ridge;
            if attempt == 0 {
                rel_gap_first = Some(rel_gap);
            }
            if rel_gap.is_finite() && rel_gap > rel_gap_max {
                rel_gap_max = rel_gap;
            }
            let last_attempt = attempt == FD_MAX_REFINEMENTS || !refining;
            if attempt == 0 || last_attempt {
                if attempt == 0 {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} f(+/-0.5h)={:+.9e}/{:+.9e} \
f(+/-1h)={:+.9e}/{:+.9e} d_small={:+.9e} d_big={:+.9e} ridge=[{:.3e},{:.3e}]",
                        attempt + 1,
                        base_h,
                        eval.f_p,
                        eval.f_m,
                        eval.f_p2,
                        eval.f_m2,
                        d_small,
                        d_big,
                        eval.ridge_min,
                        eval.ridge_max,
                    ));
                } else {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} d_small={:+.9e} d_big={:+.9e} \
rel_gap={:.3e} ridge=[{:.3e},{:.3e}] ridge_rel_span={:.3e}",
                        attempt + 1,
                        base_h,
                        d_small,
                        d_big,
                        rel_gap,
                        eval.ridge_min,
                        eval.ridge_max,
                        eval.ridge_rel_span
                    ));
                }
            }

            if refining {
                base_h *= 0.5;
                refine_steps += 1;
                continue;
            }

            if eval.ridge_jitter {
                derivative = None;
            } else {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
            break;
        }

        if derivative.is_none() {
            let same_sign = fd_same_sign(d_small, d_big);
            if same_sign && !ridge_jitter_seen {
                derivative = best_derivative
                    .or_else(|| Some(select_fd_derivative(d_small, d_big, same_sign)));
            } else if !ridge_jitter_seen {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
        }

        if derivative.is_none() && allow_analytic_fallback {
            if analytic_fallback.is_none() {
                analytic_fallback = Some(reml_state.compute_gradient(rho)?);
            }
            derivative = analytic_fallback.as_ref().map(|g| g[i]);
            log_lines.push(format!(
                "[FD RIDGE]   coord {} fallback to analytic gradient due to ridge jitter (max rel span {:.3e})",
                i, ridge_rel_span_max
            ));
        }

        fd_grad[i] = derivative.unwrap_or(f64::NAN);
        let rel_gap_first = rel_gap_first.unwrap_or(f64::NAN);
        log_lines.push(format!(
            "[FD RIDGE]   refine steps={} h_start={:.3e} h_final={:.3e} rel_gap_first={:.3e} rel_gap_max={:.3e} ridge_jitter_seen={} ridge_rel_span_max={:.3e}",
            refine_steps,
            h_start,
            base_h,
            rel_gap_first,
            rel_gap_max,
            ridge_jitter_seen,
            ridge_rel_span_max
        ));
        log_lines.push(format!(
            "[FD RIDGE]   chosen derivative = {:+.9e}",
            fd_grad[i]
        ));
    }

    if emit_logs && !log_lines.is_empty() {
        println!("{}", log_lines.join("\n"));
    }

    Ok(fd_grad)
}
