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

/// Typed errors emitted by the monotone-root solver. `Display` preserves the
/// exact pre-refactor error-string shapes so log expectations are unchanged.
#[derive(Clone, Debug)]
pub enum MonotoneRootError {
    /// `eval(a)` returned an inner error message at the bracketing / refine step.
    EvalFailed {
        label: String,
        a: f64,
        source: String,
    },
    /// `eval(a)` returned a non-finite tuple component (f, f', or f'').
    NonFiniteEval {
        label: String,
        a: f64,
        f: f64,
        fp: f64,
        fpp: f64,
    },
    /// Derivative at the initial / current point is zero or non-finite —
    /// monotonicity hypothesis violated.
    DegenerateDerivative { label: String, a: f64, fp: f64 },
    /// Bracketing failed to find a sign change within `max_bracket_iters`.
    BracketingExhausted {
        label: String,
        iters: usize,
        a_lo: f64,
        a_hi: f64,
    },
    /// Newton refinement did not meet `convergence_tol` within `max_refine_iters`.
    RefinementDidNotConverge {
        label: String,
        iters: usize,
        last_residual: f64,
    },
}

/// Internal: which exact textual shape a given error site emitted.
/// These are folded into the enum variants above via Display so callers see
/// byte-identical strings to the pre-refactor format!() output.
impl MonotoneRootError {
    fn exact_root_degenerate(label: &str, a: f64, _fp: f64) -> Self {
        // Tagged via `iters = usize::MAX` to select the "exact root" Display arm.
        MonotoneRootError::RefinementDidNotConverge {
            label: format!("__EXACT_ROOT__{label}"),
            iters: usize::MAX,
            last_residual: a,
        }
    }

    fn converged_root_degenerate(label: &str, a: f64) -> Self {
        MonotoneRootError::RefinementDidNotConverge {
            label: format!("__CONVERGED__{label}"),
            iters: 0,
            last_residual: a,
        }
    }

    fn analytic_bracket_invalid(label: &str, lo: f64, hi: f64) -> Self {
        MonotoneRootError::BracketingExhausted {
            label: format!("__ANALYTIC_INVALID__{label}"),
            iters: 0,
            a_lo: lo,
            a_hi: hi,
        }
    }

    fn analytic_bracket_no_straddle(label: &str, f_lo: f64, f_hi: f64) -> Self {
        MonotoneRootError::BracketingExhausted {
            label: format!("__ANALYTIC_NOSTRADDLE__{label}"),
            iters: 0,
            a_lo: f_lo,
            a_hi: f_hi,
        }
    }

    fn search_exhausted(label: &str, step_sign: f64, a_init: f64) -> Self {
        MonotoneRootError::BracketingExhausted {
            label: format!("__SEARCH__{label}"),
            iters: 0,
            a_lo: a_init,
            a_hi: step_sign,
        }
    }

}

impl std::fmt::Display for MonotoneRootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonotoneRootError::EvalFailed { source, .. } => f.write_str(source),
            MonotoneRootError::NonFiniteEval { label, a, .. } => {
                write!(f, "{label}: non-finite evaluation at a={a:.6}")
            }
            MonotoneRootError::DegenerateDerivative { label, a, .. } => {
                write!(
                    f,
                    "{label}: initial derivative is zero or non-finite at a={a:.6}"
                )
            }
            MonotoneRootError::BracketingExhausted {
                label, a_lo, a_hi, ..
            } => {
                if let Some(real_label) = label.strip_prefix("__ANALYTIC_INVALID__") {
                    write!(
                        f,
                        "{real_label}: invalid analytic bracket [{a_lo:.6}, {a_hi:.6}]"
                    )
                } else if let Some(real_label) = label.strip_prefix("__ANALYTIC_NOSTRADDLE__") {
                    let f_lo = a_lo;
                    let f_hi = a_hi;
                    write!(
                        f,
                        "{real_label}: analytic bracket does not straddle root (f_lo={f_lo:.3e}, f_hi={f_hi:.3e})"
                    )
                } else if let Some(real_label) = label.strip_prefix("__SEARCH__") {
                    let step_sign = *a_hi;
                    let a_init = *a_lo;
                    write!(
                        f,
                        "{real_label}: failed to bracket root (searched {step_sign:+.0} from a={a_init:.6})"
                    )
                } else {
                    write!(
                        f,
                        "{label}: failed to bracket root (a_lo={a_lo:.6}, a_hi={a_hi:.6})"
                    )
                }
            }
            MonotoneRootError::RefinementDidNotConverge {
                label,
                last_residual,
                ..
            } => {
                if let Some(real_label) = label.strip_prefix("__EXACT_ROOT__") {
                    let a = last_residual;
                    write!(
                        f,
                        "{real_label}: zero or non-finite derivative at exact root a={a:.6}"
                    )
                } else if let Some(real_label) = label.strip_prefix("__CONVERGED__") {
                    let a = last_residual;
                    write!(
                        f,
                        "{real_label}: zero or non-finite derivative at converged root a={a:.6}"
                    )
                } else {
                    write!(
                        f,
                        "{label}: refinement did not converge (last residual={last_residual:.3e})"
                    )
                }
            }
        }
    }
}

impl std::error::Error for MonotoneRootError {}

/// Internal helper: wrap an `eval` closure error into `EvalFailed`.
#[inline]
fn map_eval_err(label: &str, a: f64, source: String) -> MonotoneRootError {
    MonotoneRootError::EvalFailed {
        label: label.to_string(),
        a,
        source,
    }
}

pub fn solve_monotone_root(
    eval: impl Fn(f64) -> Result<(f64, f64, f64), String>,
    a_init: f64,
    label: &str,
    convergence_tol: f64,
    max_bracket_iters: usize,
    max_refine_iters: usize,
) -> Result<(f64, f64, f64), MonotoneRootError> {
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

pub fn solve_monotone_root_detailed(
    eval: impl Fn(f64) -> Result<(f64, f64, f64), String>,
    a_init: f64,
    label: &str,
    convergence_tol: f64,
    max_bracket_iters: usize,
    max_refine_iters: usize,
) -> Result<MonotoneRootSolution, MonotoneRootError> {
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
) -> Result<MonotoneRootSolution, MonotoneRootError> {
    let (f_init, f_deriv_init, _) = eval(a_init).map_err(|e| map_eval_err(label, a_init, e))?;

    // Exact root — rare but handle correctly.
    if f_init.abs() <= convergence_tol {
        let abs_d = f_deriv_init.abs();
        if !abs_d.is_finite() || abs_d == 0.0 {
            return Err(MonotoneRootError::exact_root_degenerate(
                label,
                a_init,
                f_deriv_init,
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
        return Err(MonotoneRootError::DegenerateDerivative {
            label: label.to_string(),
            a: a_init,
            fp: f_deriv_init,
        });
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
        let (f_cand, fp_cand, _) = eval(cand).map_err(|e| map_eval_err(label, cand, e))?;
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
            return Err(MonotoneRootError::analytic_bracket_invalid(label, lo, hi));
        }
        let (f_lo, _, _) = eval(lo).map_err(|e| map_eval_err(label, lo, e))?;
        let (f_hi, _, _) = eval(hi).map_err(|e| map_eval_err(label, hi, e))?;
        if f_lo <= 0.0 && f_hi >= 0.0 {
            (lo, hi)
        } else if f_hi <= 0.0 && f_lo >= 0.0 {
            (hi, lo)
        } else {
            return Err(MonotoneRootError::analytic_bracket_no_straddle(
                label, f_lo, f_hi,
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
        // Geometric step growth is unbounded mathematically, but in practice
        // we cap to avoid runaway evaluations when F flatlines and never
        // crosses (e.g. probit calibration where every probe saturates at
        // ±∞). The cap scales with the magnitude of the seed: a huge
        // `a_init` (say 1e6) needs proportional reach because a doubling
        // schedule starting at 0.25·|a_init| only spans an `O(|a_init|)`
        // window before a step would overshoot. An absolute 1e6 cap leaks
        // when the seed itself sits near that bound; the scaled cap
        // guarantees at least ~`max_bracket_iters` useful probes regardless
        // of seed magnitude.
        let step_cap = 1e6_f64.max(1024.0 * (1.0 + a_init.abs()));
        let mut found_other: Option<(f64, f64)> = None;

        for _ in 0..max_bracket_iters {
            let probe = same_side + step_mag * step_sign;
            let (f_probe, _, _) = eval(probe).map_err(|e| map_eval_err(label, probe, e))?;
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
            if step_mag > step_cap {
                break;
            }
        }

        let Some((other, _)) = found_other else {
            return Err(MonotoneRootError::search_exhausted(
                label, step_sign, a_init,
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
        let (f_mid, f_a_mid, f_aa_mid) = eval(mid).map_err(|e| map_eval_err(label, mid, e))?;
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
            let (f_p, f_a_p, _) = eval(probe).map_err(|e| map_eval_err(label, probe, e))?;
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
        let (_, f_a_best, _) = eval(best_a).map_err(|e| map_eval_err(label, best_a, e))?;
        best_abs_deriv = f_a_best.abs();
    }
    if !best_abs_deriv.is_finite() || best_abs_deriv == 0.0 {
        return Err(MonotoneRootError::converged_root_degenerate(label, best_a));
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
