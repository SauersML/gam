//! #938 Tier-1 rho-posterior quadrature on a real SAE objective.
//!
//! The load-bearing path is the SAE `OuterObjective::eval` analytic gradient:
//! we finite-difference that profiled-exact gradient for the local rho Hessian,
//! run deterministic Gauss-Hermite quadrature over `rho | data`, and marginalize
//! the decoder shape band by the law of total variance. The truth-known small
//! circle problem is intentionally low-n so plug-in REML bands are too narrow;
//! the rho mixture must move coverage toward nominal.

