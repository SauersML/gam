use serde::{Deserialize, Serialize};

/// How ridge-adjusted determinants should be evaluated for outer criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RidgeDeterminantMode {
    /// Use exact full logdet.
    Auto,
    /// Use full log-determinant of the ridged matrix (requires SPD in practice).
    Full,
    /// Use positive-part pseudo-determinant (sum log ev for ev > floor).
    PositivePart,
}

/// Global policy governing how a stabilization ridge participates in objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RidgePolicy {
    /// Must remain independent of smoothing parameters (`rho`) for smooth outer derivatives.
    pub rho_independent: bool,
    /// Include ridge in quadratic penalty term: `0.5 * delta * ||beta||^2`.
    pub include_quadratic_penalty: bool,
    /// Include ridge in penalty determinant term (e.g. `log|S_lambda + delta I|`).
    pub include_penalty_logdet: bool,
    /// Include ridge in Hessian used by Laplace term / implicit differentiation.
    pub include_laplacehessian: bool,
    /// Determinant evaluation mode when ridge participates in logdet terms.
    pub determinant_mode: RidgeDeterminantMode,
}

impl RidgePolicy {
    /// Default policy used by PIRLS/REML path:
    /// treat stabilization ridge as an explicit `delta I` prior contribution
    /// with adaptive logdet evaluation.
    pub const fn explicit_stabilization_full() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: true,
            include_penalty_logdet: true,
            include_laplacehessian: true,
            determinant_mode: RidgeDeterminantMode::Auto,
        }
    }

    pub const fn explicit_stabilization_full_exact() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: true,
            include_penalty_logdet: true,
            include_laplacehessian: true,
            determinant_mode: RidgeDeterminantMode::Full,
        }
    }

    /// Variant used when pseudo-determinants are required for indefinite matrices.
    pub const fn explicit_stabilization_pospart() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: true,
            include_penalty_logdet: true,
            include_laplacehessian: true,
            determinant_mode: RidgeDeterminantMode::PositivePart,
        }
    }

    /// Solver-only stabilization: the ridge `δI` stabilizes the inner linear
    /// solve (it bounds the Newton step `(H+δI)⁻¹∇`) but is **excluded** from
    /// the REML/LAML objective — no `½·δ·‖β‖²` quadratic-penalty term, no
    /// `δ`-shift of the penalty log-determinant, no `δ`-shift of the Laplace
    /// Hessian. Use this when a numerical floor is needed purely to keep the
    /// linear algebra finite during screening and must NOT bias the
    /// smoothing-parameter selection or shrink identified coefficients off the
    /// MLE. With every `include_*` false the optimized objective equals the
    /// true penalized REML criterion, so the value surface and its analytic
    /// gradient describe the same objective (gam#747/#748).
    pub const fn solver_only() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: false,
            include_penalty_logdet: false,
            include_laplacehessian: false,
            determinant_mode: RidgeDeterminantMode::PositivePart,
        }
    }
}
