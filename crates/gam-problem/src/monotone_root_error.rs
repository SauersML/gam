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
    pub fn exact_root_degenerate(label: &str, a: f64) -> Self {
        // Tagged via `iters = usize::MAX` to select the "exact root" Display arm.
        MonotoneRootError::RefinementDidNotConverge {
            label: format!("__EXACT_ROOT__{label}"),
            iters: usize::MAX,
            last_residual: a,
        }
    }

    pub fn converged_root_degenerate(label: &str, a: f64) -> Self {
        MonotoneRootError::RefinementDidNotConverge {
            label: format!("__CONVERGED__{label}"),
            iters: 0,
            last_residual: a,
        }
    }

    pub fn analytic_bracket_invalid(label: &str, lo: f64, hi: f64) -> Self {
        MonotoneRootError::BracketingExhausted {
            label: format!("__ANALYTIC_INVALID__{label}"),
            iters: 0,
            a_lo: lo,
            a_hi: hi,
        }
    }

    pub fn analytic_bracket_no_straddle(label: &str, f_lo: f64, f_hi: f64) -> Self {
        MonotoneRootError::BracketingExhausted {
            label: format!("__ANALYTIC_NOSTRADDLE__{label}"),
            iters: 0,
            a_lo: f_lo,
            a_hi: f_hi,
        }
    }

    pub fn search_exhausted(label: &str, step_sign: f64, a_init: f64) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_failed_shows_source() {
        let err = MonotoneRootError::EvalFailed {
            label: "test".to_string(),
            a: 1.0,
            source: "inner error msg".to_string(),
        };
        assert_eq!(err.to_string(), "inner error msg");
    }

    #[test]
    fn non_finite_eval_shows_label_and_a() {
        let err = MonotoneRootError::NonFiniteEval {
            label: "myroot".to_string(),
            a: 3.5,
            f: f64::NAN,
            fp: 0.0,
            fpp: 0.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("myroot"), "message: {msg}");
        assert!(msg.contains("3.5") || msg.contains("a=3"), "message: {msg}");
    }

    #[test]
    fn degenerate_derivative_shows_label_and_a() {
        let err = MonotoneRootError::DegenerateDerivative {
            label: "solver".to_string(),
            a: -1.0,
            fp: 0.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("solver"), "message: {msg}");
        assert!(
            msg.to_lowercase().contains("derivative") || msg.contains("zero"),
            "message: {msg}"
        );
    }

    #[test]
    fn bracketing_exhausted_default_shows_a_lo_and_a_hi() {
        let err = MonotoneRootError::BracketingExhausted {
            label: "mysolve".to_string(),
            iters: 10,
            a_lo: -1.0,
            a_hi: 2.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("mysolve"), "message: {msg}");
        assert!(msg.contains("-1") && msg.contains("2"), "message: {msg}");
    }

    #[test]
    fn analytic_bracket_invalid_factory_shows_real_label() {
        let err = MonotoneRootError::analytic_bracket_invalid("myfunc", 0.5, 1.5);
        let msg = err.to_string();
        assert!(msg.contains("myfunc"), "message: {msg}");
        assert!(
            msg.contains("invalid") || msg.contains("bracket"),
            "message: {msg}"
        );
    }

    #[test]
    fn analytic_bracket_no_straddle_factory_shows_f_values() {
        let err = MonotoneRootError::analytic_bracket_no_straddle("myfunc", 0.1, 0.2);
        let msg = err.to_string();
        assert!(msg.contains("myfunc"), "message: {msg}");
        assert!(
            msg.to_lowercase().contains("straddle") || msg.contains("f_lo"),
            "message: {msg}"
        );
    }

    #[test]
    fn search_exhausted_factory_shows_real_label() {
        let err = MonotoneRootError::search_exhausted("myfunc", 1.0, 0.0);
        let msg = err.to_string();
        assert!(msg.contains("myfunc"), "message: {msg}");
        assert!(
            msg.to_lowercase().contains("bracket") || msg.contains("search"),
            "message: {msg}"
        );
    }

    #[test]
    fn refinement_did_not_converge_default_shows_residual() {
        let err = MonotoneRootError::RefinementDidNotConverge {
            label: "fit".to_string(),
            iters: 50,
            last_residual: 1.23e-4,
        };
        let msg = err.to_string();
        assert!(msg.contains("fit"), "message: {msg}");
        assert!(msg.to_lowercase().contains("converge"), "message: {msg}");
    }

    #[test]
    fn exact_root_degenerate_factory_shows_real_label() {
        let err = MonotoneRootError::exact_root_degenerate("myroot", 0.75);
        let msg = err.to_string();
        assert!(msg.contains("myroot"), "message: {msg}");
        assert!(
            msg.contains("0.75") || msg.contains("a=0"),
            "message: {msg}"
        );
    }

    #[test]
    fn converged_root_degenerate_factory_shows_real_label() {
        let err = MonotoneRootError::converged_root_degenerate("conv_root", 2.0);
        let msg = err.to_string();
        assert!(msg.contains("conv_root"), "message: {msg}");
        assert!(msg.contains("2.0") || msg.contains("a=2"), "message: {msg}");
    }
}
