/// Shared hybrid bracketing + Newton solver for strictly monotone calibration
/// equations F(a) = 0.
///
/// `eval(a)` must return `(F(a), F'(a), F''(a))`.  The second derivative is
/// carried through for the caller but is not used by the solver itself.
///
/// Returns `(root, |F'(root)|, F(root))`.  The absolute derivative is always
/// positive and can be used directly as the density-normalising calibration
/// derivative.  Callers must validate the residual against the scale of their
/// calibration equation.
///
/// The monotone direction (increasing vs decreasing) is inferred from the
/// sign of F'(a) at the initial point, so the same code handles both the
/// Bernoulli case (F increasing) and the survival case (F decreasing).
#[derive(Clone, Copy, Debug)]
pub struct MonotoneRootSolution {
    pub root: f64,
    pub abs_deriv: f64,
    pub residual: f64,
    pub refine_iters: usize,
}

pub fn solve_monotone_root(
    eval: impl Fn(f64) -> Result<(f64, f64, f64), String>,
    a_init: f64,
    label: &str,
    convergence_tol: f64,
    max_bracket_iters: usize,
    max_refine_iters: usize,
) -> Result<(f64, f64, f64), String> {
    let solution = solve_monotone_root_detailed(
        eval,
        a_init,
        label,
        convergence_tol,
        max_bracket_iters,
        max_refine_iters,
    )?;
    Ok((solution.root, solution.abs_deriv, solution.residual))
}

/// Like [`solve_monotone_root`] but accepts an optional analytic bracket
/// `[a_lo, a_hi]` known to straddle the root. When supplied, Phase 1 of the
/// solver skips its expanding-step search and uses the analytic bracket
/// directly.
pub fn solve_monotone_root_with_bracket(
    eval: impl Fn(f64) -> Result<(f64, f64, f64), String>,
    a_init: f64,
    label: &str,
    convergence_tol: f64,
    max_bracket_iters: usize,
    max_refine_iters: usize,
    analytic_bracket: Option<(f64, f64)>,
) -> Result<(f64, f64, f64), String> {
    let solution = solve_monotone_root_detailed_with_bracket(
        eval,
        a_init,
        label,
        convergence_tol,
        max_bracket_iters,
        max_refine_iters,
        analytic_bracket,
    )?;
    Ok((solution.root, solution.abs_deriv, solution.residual))
}

pub fn solve_monotone_root_detailed(
    eval: impl Fn(f64) -> Result<(f64, f64, f64), String>,
    a_init: f64,
    label: &str,
    convergence_tol: f64,
    max_bracket_iters: usize,
    max_refine_iters: usize,
) -> Result<MonotoneRootSolution, String> {
    solve_monotone_root_detailed_with_bracket(
        eval,
        a_init,
        label,
        convergence_tol,
        max_bracket_iters,
        max_refine_iters,
        None,
    )
}

pub fn solve_monotone_root_detailed_with_bracket(
    eval: impl Fn(f64) -> Result<(f64, f64, f64), String>,
    a_init: f64,
    label: &str,
    convergence_tol: f64,
    max_bracket_iters: usize,
    max_refine_iters: usize,
    analytic_bracket: Option<(f64, f64)>,
) -> Result<MonotoneRootSolution, String> {
    let (f_init, f_deriv_init, _) = eval(a_init)?;

    // Exact root — rare but handle correctly.
    if f_init.abs() <= convergence_tol {
        let abs_d = f_deriv_init.abs();
        if !abs_d.is_finite() || abs_d == 0.0 {
            return Err(format!(
                "{label}: zero or non-finite derivative at exact root a={a_init:.6}"
            ));
        }
        return Ok(MonotoneRootSolution {
            root: a_init,
            abs_deriv: abs_d,
            residual: f_init,
            refine_iters: 0,
        });
    }

    if !f_deriv_init.is_finite() || f_deriv_init == 0.0 {
        return Err(format!(
            "{label}: initial derivative is zero or non-finite at a={a_init:.6}"
        ));
    }

    // With a good warm start, the root is often within one or two Newton
    // corrections. Try that local basin before spending evaluations on a
    // global bracket; fall back to the bracketed solver unchanged if the
    // probe is not decisive.
    let mut a = a_init;
    let mut f = f_init;
    let mut fp = f_deriv_init;
    for probe_iter in 0..2 {
        if f.abs() <= convergence_tol {
            let abs_d = fp.abs();
            if !abs_d.is_finite() || abs_d == 0.0 {
                break;
            }
            return Ok(MonotoneRootSolution {
                root: a,
                abs_deriv: abs_d,
                residual: f,
                refine_iters: probe_iter,
            });
        }

        if !fp.is_finite() || fp.abs() <= 1e-30 {
            break;
        }

        let step = -f / fp;
        if !step.is_finite() || step.abs() > 8.0 * (1.0 + a.abs()) {
            break;
        }

        let cand = a + step;
        let (f_cand, fp_cand, _) = eval(cand)?;
        if f_cand.abs() <= convergence_tol {
            let abs_d = fp_cand.abs();
            if !abs_d.is_finite() || abs_d == 0.0 {
                break;
            }
            return Ok(MonotoneRootSolution {
                root: cand,
                abs_deriv: abs_d,
                residual: f_cand,
                refine_iters: probe_iter + 1,
            });
        }

        a = cand;
        f = f_cand;
        fp = fp_cand;
    }

    // --- Phase 1: bracket the root -------------------------------------------
    let (mut neg_pt, mut pos_pt) = if let Some((lo, hi)) = analytic_bracket {
        if !lo.is_finite() || !hi.is_finite() || lo == hi {
            return Err(format!(
                "{label}: invalid analytic bracket [{lo:.6}, {hi:.6}]"
            ));
        }
        let (f_lo, _, _) = eval(lo)?;
        let (f_hi, _, _) = eval(hi)?;
        if f_lo <= 0.0 && f_hi >= 0.0 {
            (lo, hi)
        } else if f_hi <= 0.0 && f_lo >= 0.0 {
            (hi, lo)
        } else {
            return Err(format!(
                "{label}: analytic bracket does not straddle root (f_lo={f_lo:.3e}, f_hi={f_hi:.3e})"
            ));
        }
    } else {
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

        if f_init_negative {
            (same_side, other)
        } else {
            (other, same_side)
        }
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

    let mut refine_iters = 0usize;
    for _ in 0..max_refine_iters {
        refine_iters += 1;
        let (lo, hi) = if neg_pt <= pos_pt {
            (neg_pt, pos_pt)
        } else {
            (pos_pt, neg_pt)
        };
        let mid = 0.5 * (lo + hi);
        let (f_mid, f_a_mid, f_aa_mid) = eval(mid)?;
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

        // Prefer a safeguarded Halley step when the second derivative is
        // available and well-conditioned. The caller already computed F''(a),
        // so using it here reduces expensive calibration evaluations for the
        // exact denested likelihood paths without changing the objective.
        let halley_probe = if f_a_mid.is_finite() && f_a_mid.abs() > 1e-30 {
            let halley_denom = 2.0 * f_a_mid * f_a_mid - f_mid * f_aa_mid;
            if halley_denom.is_finite() && halley_denom.abs() > 1e-30 {
                let cand = mid - (2.0 * f_mid * f_a_mid) / halley_denom;
                if cand > lo && cand < hi {
                    Some(cand)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Fall back to the monotone Newton step if Halley is unavailable or
        // would leave the current bracket.
        let probe = if let Some(cand) = halley_probe {
            cand
        } else if f_a_mid.is_finite() && f_a_mid.abs() > 1e-30 {
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
            neg_pt = bracket_pt;
        } else {
            pos_pt = bracket_pt;
        }

        let (next_lo, next_hi) = if neg_pt <= pos_pt {
            (neg_pt, pos_pt)
        } else {
            (pos_pt, neg_pt)
        };
        if (next_hi - next_lo).abs() <= convergence_tol * (1.0 + next_hi.abs() + next_lo.abs()) {
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

    Ok(MonotoneRootSolution {
        root: best_a,
        abs_deriv: best_abs_deriv,
        residual: best_f,
        refine_iters,
    })
}

#[cfg(test)]
mod tests {
    use super::solve_monotone_root;
    use std::cell::RefCell;

    #[test]
    fn solve_monotone_root_converges_for_increasing_function() {
        let (root, abs_deriv, residual) = solve_monotone_root(
            |a| {
                let ea = a.exp();
                Ok((ea - 2.0, ea, ea))
            },
            0.0,
            "increasing",
            1e-12,
            32,
            32,
        )
        .expect("root");

        assert!((root - std::f64::consts::LN_2).abs() < 1e-10);
        assert!((abs_deriv - 2.0).abs() < 1e-10);
        assert!(residual.abs() < 1e-12);
    }

    #[test]
    fn solve_monotone_root_accepts_halley_probe_for_decreasing_function() {
        let eval_points = RefCell::new(Vec::new());
        let (root, abs_deriv, residual) = solve_monotone_root(
            |a| {
                eval_points.borrow_mut().push(a);
                let ea = (-a).exp();
                Ok((ea - 0.5, -ea, ea))
            },
            0.0,
            "decreasing",
            1e-12,
            32,
            32,
        )
        .expect("root");

        let f_mid = (-0.5f64).exp() - 0.5;
        let f_a_mid = -(-0.5f64).exp();
        let f_aa_mid = (-0.5f64).exp();
        let expected_probe =
            0.5 - (2.0 * f_mid * f_a_mid) / (2.0 * f_a_mid * f_a_mid - f_mid * f_aa_mid);
        assert!((root - std::f64::consts::LN_2).abs() < 1e-10);
        assert!((abs_deriv - 0.5).abs() < 1e-10);
        assert!(residual.abs() < 1e-12);
        assert!(
            eval_points
                .borrow()
                .iter()
                .copied()
                .any(|a| (a - expected_probe).abs() < 1e-12)
        );
    }
}
