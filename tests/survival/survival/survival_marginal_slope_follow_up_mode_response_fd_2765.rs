//! gam#2765 / gam#2767 end-to-end gate: on a follow-up-varying slope the
//! criterion's coefficient mode response `dβ̂/dψ` must match a finite difference
//! of the fit's own β̂.
//!
//! The unit gates in `psi_terms_fd_tests` difference `D_β H[δ]` and
//! `D²_β H[u,v]` against the family's own joint Hessian, which is where the
//! defect was: `add_pullback_primary_hessian` pulled the row Hessian back
//! through ONE slope channel, so on a varying slope every consumer of that
//! pullback differentiated a different model. This gate closes the same defect
//! from the other end, through a real fit, because `D_β H` is what builds the
//! Jeffreys curvature `H_Φ` and its second-order completion — hence the operator
//! the mode response is solved against. A wrong pullback therefore shows up as a
//! wrong `dβ̂/dψ`, and that is a quantity the outer runner already publishes
//! beside its own Ridders-certified finite difference.
//!
//! Measured at the acceptance fixture's shape (`n = 400`, Weibull baseline,
//! `slope_time_k = 4`): `3.3e-2` and `3.2e-2` relative before the repair,
//! `5.0e-8` and `8.6e-9` after — six orders, on a quantity whose oracle is the
//! same inner solve the fit runs. The `1e-5` bar below sits four orders above
//! the repaired value and three below the broken one, so it cannot be cleared by
//! a partial fix.
//!
//! This grades the mode response, not the total outer gradient. They are
//! separate contracts: this fixture isolates the coefficient response that the
//! follow-up margin changes, while the complete profiled-gradient calculus is
//! covered by its own outer-gradient gates.

