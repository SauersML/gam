/// Shared hybrid bracketing + Newton solver for strictly monotone calibration
/// equations F(a) = 0.
///
/// `eval(a)` must return `(F(a), F'(a), F''(a))`.  The second derivative is
/// carried through for the caller but is not used by the solver itself.
///
/// Returns `(root, |F'(root)|)`.  The absolute derivative is always positive
/// and can be used directly as the density-normalising calibration derivative.
///
/// The monotone direction (increasing vs decreasing) is inferred from the
/// sign of F'(a) at the initial point, so the same code handles both the
/// Bernoulli case (F increasing) and the survival case (F decreasing).
pub fn solve_monotone_root(
    eval: impl Fn(f64) -> Result<(f64, f64, f64), String>,
    a_init: f64,
    label: &str,
    convergence_tol: f64,
    max_bracket_iters: usize,
    max_refine_iters: usize,
) -> Result<(f64, f64), String> {
    let (f_init, f_deriv_init, _) = eval(a_init)?;

    // Exact root — rare but handle correctly.
    if f_init.abs() <= convergence_tol {
        let abs_d = f_deriv_init.abs();
        if !abs_d.is_finite() || abs_d == 0.0 {
            return Err(format!(
                "{label}: zero or non-finite derivative at exact root a={a_init:.6}"
            ));
        }
        return Ok((a_init, abs_d));
    }

    if !f_deriv_init.is_finite() || f_deriv_init == 0.0 {
        return Err(format!(
            "{label}: initial derivative is zero or non-finite at a={a_init:.6}"
        ));
    }

    // --- Phase 1: bracket the root -------------------------------------------
    //
    // We need a point on the opposite side of zero from f_init.
    // The correct search direction depends on both the sign of f_init and
    // the monotonicity of F:
    //
    //   F increasing, f < 0 → root is to the right  (+)
    //   F increasing, f > 0 → root is to the left   (−)
    //   F decreasing, f < 0 → root is to the left   (−)
    //   F decreasing, f > 0 → root is to the right  (+)
    //
    // Compactly:  step_sign = −sign(f · F')
    let step_sign: f64 = if f_init * f_deriv_init < 0.0 {
        1.0
    } else {
        -1.0
    };

    let f_init_negative = f_init < 0.0;
    let mut same_side = a_init; // last point with same sign as f_init
    let mut step_mag = (0.25 * (1.0 + a_init.abs())).max(1.0);
    let mut found_other: Option<(f64, f64)> = None;

    for _ in 0..max_bracket_iters {
        let probe = same_side + step_mag * step_sign;
        let (f_probe, _, _) = eval(probe)?;
        let crossed = if f_init_negative {
            f_probe >= 0.0
        } else {
            f_probe <= 0.0
        };
        if crossed {
            found_other = Some((probe, f_probe));
            break;
        }
        same_side = probe;
        step_mag *= 2.0;
        if step_mag > 1e6 {
            break;
        }
    }

    let Some((other, _)) = found_other else {
        return Err(format!(
            "{label}: failed to bracket root (searched {step_sign:+.0} from a={a_init:.6})"
        ));
    };

    // Normalise to lo/hi with f(lo) <= 0 <= f(hi).
    let (mut lo, mut hi) = if f_init_negative {
        (same_side, other)
    } else {
        (other, same_side)
    };

    // --- Phase 2: hybrid bisection / Newton refinement -----------------------

    let mut best_a = a_init;
    let mut best_f = f_init;
    let mut best_abs_deriv = f_deriv_init.abs();

    #[inline]
    fn update_best(
        best_a: &mut f64,
        best_f: &mut f64,
        best_abs_d: &mut f64,
        a: f64,
        f: f64,
        f_d: f64,
    ) {
        if f.abs() < best_f.abs() {
            *best_a = a;
            *best_f = f;
            *best_abs_d = f_d.abs();
        }
    }

    for _ in 0..max_refine_iters {
        let mid = 0.5 * (lo + hi);
        let (f_mid, f_a_mid, _) = eval(mid)?;
        update_best(
            &mut best_a,
            &mut best_f,
            &mut best_abs_deriv,
            mid,
            f_mid,
            f_a_mid,
        );

        if f_mid.abs() <= convergence_tol {
            break;
        }

        // Newton step from midpoint (works for either sign of F').
        let probe = if f_a_mid.is_finite() && f_a_mid.abs() > 1e-30 {
            let cand = mid - f_mid / f_a_mid;
            if cand > lo && cand < hi { cand } else { mid }
        } else {
            mid
        };

        // Evaluate probe if it differs from midpoint.
        let (bracket_pt, f_bracket) = if (probe - mid).abs() > 0.0 {
            let (f_p, f_a_p, _) = eval(probe)?;
            update_best(
                &mut best_a,
                &mut best_f,
                &mut best_abs_deriv,
                probe,
                f_p,
                f_a_p,
            );
            (probe, f_p)
        } else {
            (mid, f_mid)
        };

        if f_bracket <= 0.0 {
            lo = bracket_pt;
        } else {
            hi = bracket_pt;
        }

        if (hi - lo).abs() <= convergence_tol * (1.0 + hi.abs() + lo.abs()) {
            break;
        }
    }

    // Final validation: re-evaluate at best_a if the derivative is suspect.
    if !best_abs_deriv.is_finite() || best_abs_deriv == 0.0 {
        let (_, f_a_best, _) = eval(best_a)?;
        best_abs_deriv = f_a_best.abs();
    }
    if !best_abs_deriv.is_finite() || best_abs_deriv == 0.0 {
        return Err(format!(
            "{label}: zero or non-finite derivative at converged root a={best_a:.6}"
        ));
    }

    Ok((best_a, best_abs_deriv))
}
