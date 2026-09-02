//! #2612: the outer REML/LAML gradient must be the gradient of the criterion it
//! reports — with the Jeffreys/Firth term ARMED, not only with it disarmed.
//!
//! The multinomial formula path re-solves with `½ log|I(β)|` armed whenever the
//! certified mode is (quasi-)separated. On the penguins real-data arm that refit
//! then dies in the OUTER smoothing search at
//! `line_search=StepSizeTooSmall` — the solver's own gloss on which is *"the
//! direction descended but no step improved the objective"* — with an indefinite
//! terminal analytic Hessian. On a differentiable objective with a correct
//! gradient that cannot happen: a short enough step along `−g` must decrease the
//! value. So the gradient the line search consumes is not the derivative of the
//! value it evaluates.
//!
//! This is that statement reduced to a fixture that runs in seconds: a small,
//! deliberately quasi-separated multinomial, the production outer criterion, and
//! central finite differences of its VALUE against its own analytic gradient,
//! taken twice at the SAME rho with the term disarmed and armed. The disarmed
//! arm is the control — it pins that the harness, the criterion and the FD step
//! are sound, so a failure in the armed arm is attributable to the term.

// ───────────────────────────────────────────────────────────────────────────
// The same question one order up: is the analytic outer HESSIAN the Jacobian
// of the (now exact) analytic outer gradient?
// ───────────────────────────────────────────────────────────────────────────

