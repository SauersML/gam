use super::*;

#[derive(Clone)]
pub(crate) struct SurvivalLocationScaleFamily {
    pub(crate) n: usize,
    pub(crate) y: Array1<f64>,
    pub(crate) w: Array1<f64>,
    pub(crate) inverse_link: InverseLink,
    pub(crate) derivative_guard: f64,
    pub(crate) x_time_entry: Arc<Array2<f64>>,
    pub(crate) x_time_exit: Arc<Array2<f64>>,
    pub(crate) x_time_deriv: Arc<Array2<f64>>,
    pub(crate) time_wiggle_knots: Option<Array1<f64>>,
    pub(crate) time_wiggle_degree: Option<usize>,
    pub(crate) time_wiggle_ncols: usize,
    pub(crate) time_linear_constraints: Option<LinearInequalityConstraints>,
    /// Exit design for threshold block (always present; used as main design).
    pub(crate) x_threshold: DesignMatrix,
    /// Entry design for threshold block when time-varying.
    /// When `None`, the block is time-invariant: q0 = q1 (current behavior).
    pub(crate) x_threshold_entry: Option<DesignMatrix>,
    /// Exit-time derivative design for threshold when time-varying.
    pub(crate) x_threshold_deriv: Option<DesignMatrix>,
    /// Exit design for log-sigma block (always present; used as main design).
    pub(crate) x_log_sigma: DesignMatrix,
    /// Entry design for log-sigma block when time-varying.
    pub(crate) x_log_sigma_entry: Option<DesignMatrix>,
    /// Exit-time derivative design for log-sigma when time-varying.
    pub(crate) x_log_sigma_deriv: Option<DesignMatrix>,
    pub(crate) x_link_wiggle: Option<DesignMatrix>,
    pub(crate) wiggle_knots: Option<Array1<f64>>,
    pub(crate) wiggle_degree: Option<usize>,
    /// σ-scaled log-t AFT location baseline (issue #892). `Some` only in the
    /// rank-1 reduced parametric-AFT regime, where the time warp is removed
    /// (`h ≡ 0`) and the `log t` baseline rides the σ-scaled `q` (location)
    /// channel instead: the effective location is shifted to `η_t − log t` with a
    /// time-derivative `−1/t`, so `u = inv_sigma·(log t − η_t) = (log t − μ)/σ`
    /// and the event Jacobian gains `−log σ − log t`. `None` everywhere else.
    pub(crate) location_log_time: Option<LocationLogTimeOffset>,
    pub(crate) policy: gam_runtime::resource::ResourcePolicy,
}

/// The σ-scaled log-t AFT location baseline (issue #892), applied to the `q`
/// channel in the rank-1 reduced parametric-AFT regime. Each field is a per-row
/// shift of the effective location predictor (`η_t → η_t + value`, derivative
/// `+ deriv`), so the standardized residual becomes `inv_sigma·(log t − η_t)`.
#[derive(Clone, Debug)]
pub(crate) struct LocationLogTimeOffset {
    /// `−log t_exit`: shifts the exit-time effective location by `−log t`.
    pub(crate) value_exit: Array1<f64>,
    /// `−log t_entry`: shifts the entry-time effective location by `−log t`.
    pub(crate) value_entry: Array1<f64>,
    /// `−1/t_exit`: the exit-time derivative of the `−log t` location shift,
    /// feeding the `q`-channel `qdot` so `g` carries `inv_sigma/t`.
    pub(crate) deriv_exit: Array1<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct SurvivalPredictorState {
    pub(crate) h0: f64,
    pub(crate) h1: f64,
    pub(crate) g: f64,
    /// q evaluated at entry time. When the threshold/sigma blocks are
    /// time-invariant, q0 == q1.
    pub(crate) q0: f64,
    /// q evaluated at exit time.
    pub(crate) q1: f64,
    /// Explicit roundoff envelope from the compensated `d_raw + qdot`
    /// subtraction used to form `g`.
    pub(crate) g_roundoff_slack: f64,
    /// max(|d_raw|, |qdot|): kept only for diagnostics so monotonicity errors
    /// can report the scale of the operands that produced `g`.
    pub(crate) g_operand_scale: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct SurvivalRowDerivatives {
    pub(crate) ll: f64,
    /// Entry-only derivative: d ell / dq0 = w * r(u0).
    pub(crate) d1_q0: f64,
    /// Entry-only second derivative: d² ell / dq0² = w * r'(u0).
    pub(crate) d2_q0: f64,
    /// Entry-only third derivative: d³ ell / dq0³ = w * r''(u0).
    pub(crate) d3_q0: f64,
    /// Exit-only derivative: d ell / dq1.
    pub(crate) d1_q1: f64,
    /// Exit-only second derivative: d² ell / dq1².
    pub(crate) d2_q1: f64,
    /// Exit-only third derivative: d³ ell / dq1³.
    pub(crate) d3_q1: f64,
    /// Exit-only derivatives with respect to qdot1 = dq/dt at the event time.
    pub(crate) d1_qdot1: f64,
    pub(crate) d2_qdot1: f64,
    pub(crate) grad_time_eta_h0: f64,
    pub(crate) grad_time_eta_h1: f64,
    pub(crate) grad_time_eta_d: f64,
    pub(crate) h_time_h0: f64,
    pub(crate) h_time_h1: f64,
    pub(crate) h_time_d: f64,
}

impl SurvivalRowDerivatives {
    /// NLL gradient w.r.t. the three time channels `(h0, h1, d_raw)`.
    ///
    /// The stored `grad_time_eta_*` are log-likelihood partials `∂ℓ/∂·`; the
    /// NLL gradient is `-∂ℓ/∂·`, applied uniformly so the three channels can
    /// never disagree on sign.
    #[inline]
    pub(crate) fn time_channel_nll_gradient(&self) -> [f64; 3] {
        [
            -self.grad_time_eta_h0,
            -self.grad_time_eta_h1,
            -self.grad_time_eta_d,
        ]
    }

    /// NLL Hessian diagonal in time-channel space `(h0, h1, d_raw)`.
    ///
    /// The row likelihood factors through the functionally independent indices
    /// `(u0, u1, g)`, so the time-channel Hessian is diagonal. All three stored
    /// `h_time_*` hold `+∂²ℓ/∂·²` (`= -tower.h[i][i]` of the NLL jet), so the
    /// NLL curvature `-∂²ℓ` negates each uniformly. Owning the sign here keeps
    /// the per-channel signs locked together (gam#1396).
    #[inline]
    pub(crate) fn time_channel_nll_curvature_diag(&self) -> [f64; 3] {
        [-self.h_time_h0, -self.h_time_h1, -self.h_time_d]
    }
}
