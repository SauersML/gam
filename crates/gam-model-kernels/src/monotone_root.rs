/// Shared hybrid bracketing + Newton solver for strictly monotone calibration
/// equations F(a) = 0.
///
/// `eval(a)` must return `(F(a), F'(a), F''(a))`.  The second derivative feeds
/// the safeguarded Halley step inside the shared solver.
///
/// The algorithm core lives in `opt::find_root_monotone` (warm-start Newton
/// probes → analytic-or-geometric bracket → hybrid Halley/Newton/bisection
/// refinement with best-point tracking); this module is the thin domain
/// wrapper that keeps the calibration-facing signature, threads the `label`
/// context through, and maps `opt::RootError` back onto
/// [`MonotoneRootError`]'s byte-identical `Display` shapes.
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

pub use gam_problem::MonotoneRootError;

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
    // `RootConfig::new` fills exactly this solver's historical constants:
    // bracket step fraction 0.25, Newton/Halley derivative floor 1e-30, and
    // the ×8 warm-start trust cap.
    let config = opt::RootConfig::new(convergence_tol, max_bracket_iters, max_refine_iters);
    let oracle = |a: f64| {
        eval(a)
            .map(|(value, d1, d2)| opt::RootSample { value, d1, d2 })
            .map_err(|source| (a, source))
    };
    let solution = opt::find_root_monotone(oracle, a_init, &config, analytic_bracket)
        .map_err(|err| map_root_err(label, err))?;
    Ok(MonotoneRootSolution {
        root: solution.root,
        abs_deriv: solution.abs_deriv,
        residual: solution.value,
        refine_iters: solution.iters,
    })
}

/// Fold the shared solver's typed error back onto this crate's
/// [`MonotoneRootError`], preserving the exact pre-refactor `Display`
/// strings (each arm routes through the same factory the old in-crate
/// solver used at the corresponding failure site).
fn map_root_err(label: &str, err: opt::RootError<(f64, String)>) -> MonotoneRootError {
    match err {
        opt::RootError::Eval((a, source)) => MonotoneRootError::EvalFailed {
            label: label.to_string(),
            a,
            source,
        },
        opt::RootError::ExactRootDegenerate { at } => {
            MonotoneRootError::exact_root_degenerate(label, at)
        }
        opt::RootError::DegenerateDerivative { at, derivative } => {
            MonotoneRootError::DegenerateDerivative {
                label: label.to_string(),
                a: at,
                fp: derivative,
            }
        }
        opt::RootError::BracketInvalid { lo, hi } => {
            MonotoneRootError::analytic_bracket_invalid(label, lo, hi)
        }
        opt::RootError::BracketNoStraddle { f_lo, f_hi } => {
            MonotoneRootError::analytic_bracket_no_straddle(label, f_lo, f_hi)
        }
        opt::RootError::BracketingExhausted { direction, seed } => {
            MonotoneRootError::search_exhausted(label, direction, seed)
        }
        opt::RootError::ConvergedRootDegenerate { at } => {
            MonotoneRootError::converged_root_degenerate(label, at)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MonotoneRootError, solve_monotone_root, solve_monotone_root_detailed,
        solve_monotone_root_detailed_with_bracket,
    };
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

    #[test]
    fn solve_linear_function_reaches_exact_root() {
        // F(a) = 2a − 7, root at a = 3.5, F' = 2 everywhere.
        let (root, abs_deriv, residual) = solve_monotone_root(
            |a| Ok((2.0 * a - 7.0, 2.0, 0.0)),
            0.0,
            "linear",
            1e-12,
            32,
            64,
        )
        .expect("root");
        assert!((root - 3.5).abs() < 1e-10, "root={root}");
        assert!((abs_deriv - 2.0).abs() < 1e-10, "abs_deriv={abs_deriv}");
        assert!(residual.abs() < 1e-12, "residual={residual}");
    }

    #[test]
    fn exact_root_at_init_returns_zero_iters() {
        // F(0) = 0 exactly, so the solver should return immediately with refine_iters=0.
        let result = solve_monotone_root_detailed(
            |a| Ok((a, 1.0, 0.0)),
            0.0,
            "exact_at_init",
            1e-12,
            32,
            32,
        )
        .expect("solution");
        assert!(result.root.abs() < 1e-12, "root={}", result.root);
        assert_eq!(result.refine_iters, 0);
    }

    #[test]
    fn degenerate_derivative_returns_error() {
        // F'(a_init) = 0 is degenerate; the solver must return Err rather than
        // infinite-loop or divide by zero.
        let err = solve_monotone_root(
            |a| Ok((a - 5.0, 0.0, 0.0)),
            0.0,
            "degenerate_fp",
            1e-12,
            32,
            32,
        )
        .unwrap_err();
        match err {
            MonotoneRootError::DegenerateDerivative { .. } => {}
            other => panic!("expected DegenerateDerivative, got {other:?}"),
        }
    }

    #[test]
    fn analytic_bracket_is_honored() {
        // Supply a bracket [0, 10] for F(a) = a − 3; the solver must use it and
        // converge to root = 3.
        let sol = solve_monotone_root_detailed_with_bracket(
            |a| Ok((a - 3.0, 1.0, 0.0)),
            5.0,
            "analytic_bracket",
            1e-12,
            32,
            64,
            Some((0.0, 10.0)),
        )
        .expect("solution");
        assert!((sol.root - 3.0).abs() < 1e-10, "root={}", sol.root);
        assert!(sol.residual.abs() < 1e-12, "residual={}", sol.residual);
    }

    #[test]
    fn search_exhausted_with_zero_bracket_iters() {
        // max_bracket_iters=0 and init is not at root → bracketing cannot succeed.
        let err = solve_monotone_root(
            |a| Ok((a - 100.0, 1.0, 0.0)),
            0.0,
            "no_bracket",
            1e-12,
            0, // no bracket iterations allowed
            32,
        )
        .unwrap_err();
        match err {
            MonotoneRootError::BracketingExhausted { .. } => {}
            other => panic!("expected BracketingExhausted, got {other:?}"),
        }
    }

    #[test]
    fn error_display_shapes_are_preserved_through_the_shared_solver() {
        // The Display strings are contract: they must be byte-identical to the
        // pre-extraction in-crate solver's output at each failure site.
        let degenerate = solve_monotone_root(
            |a| Ok((a - 5.0, 0.0, 0.0)),
            0.0,
            "caldbg",
            1e-12,
            32,
            32,
        )
        .unwrap_err();
        assert_eq!(
            degenerate.to_string(),
            "caldbg: initial derivative is zero or non-finite at a=0.000000"
        );

        let exhausted =
            solve_monotone_root(|a| Ok((a - 100.0, 1.0, 0.0)), 0.0, "caldbg", 1e-12, 0, 32)
                .unwrap_err();
        assert_eq!(
            exhausted.to_string(),
            "caldbg: failed to bracket root (searched +1 from a=0.000000)"
        );

        let eval_failed = solve_monotone_root(
            |_| Err("inner calibration blew up".to_string()),
            0.0,
            "caldbg",
            1e-12,
            32,
            32,
        )
        .unwrap_err();
        assert_eq!(eval_failed.to_string(), "inner calibration blew up");
    }
}
