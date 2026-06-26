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
