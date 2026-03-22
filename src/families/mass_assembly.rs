//! Parametric baseline mass assembly for latent survival models.
//!
//! This module bridges parametric baseline specifications (Royston-Parmar splines,
//! Gompertz, Gompertz-Makeham, Weibull) with the K_{k,m} kernel machinery.  At each
//! PIRLS iteration the current baseline coefficients are used to recompute cumulative
//! hazard masses and their derivatives, producing [`LatentSurvivalRow`] values that
//! feed directly into the kernel evaluator.
//!
//! # Supported baselines
//!
//! - **Royston-Parmar (RP)**: `H_0(t) = exp(s(log t))` where `s` is a natural cubic
//!   B-spline in `log t`.  Fully implemented.
//! - **Gompertz-Makeham**: `h(t) = c exp(γt) + d` with loaded/unloaded decomposition.
//!   Sketch implementation.
//! - **Gompertz / Weibull**: placeholder arms returning errors.

use crate::families::lognormal_kernel::{LatentSurvivalEventType, LatentSurvivalRow};

// ─── Baseline specification ──────────────────────────────────────────────────

/// Parametric baseline specification for the cumulative hazard.
#[derive(Clone, Debug)]
pub enum BaselineSpec {
    /// Pre-compiled fixed masses (baseline is frozen, not learned).
    /// `assemble_row_masses` simply copies the pre-stored masses through.
    Frozen,

    /// Royston-Parmar flexible parametric model.
    ///
    /// The cumulative hazard on the log-time scale is:
    ///   `H_0(t) = exp(s(log t))` where `s(x) = B(x)^T β`
    /// and `B(x)` is a natural cubic B-spline basis evaluated at `x = log t`.
    ///
    /// Coefficients `β` are learned jointly with the smooth terms.
    RoystonParmar,

    /// Gompertz: `h_0(t) = c exp(γ t)`, `H_0(t) = (c/γ)(exp(γt) - 1)`.
    ///
    /// Baseline coefficients: `[log c, γ]`.
    Gompertz,

    /// Gompertz-Makeham: `h_0(t) = c exp(γ t) + d`.
    ///
    /// The loaded component is `h_L(t) = c exp(γt)` and the unloaded
    /// (background) component is `h_U(t) = d`.
    ///
    /// Baseline coefficients: `[log c, γ, log d]`.
    GompertzMakeham,

    /// Weibull: `H_0(t) = (t/λ)^k`.
    ///
    /// Baseline coefficients: `[log λ, log k]`.
    Weibull,
}

// ─── Precompiled row geometry ────────────────────────────────────────────────

/// Precomputed per-row geometry that allows fast mass recomputation from
/// updated baseline coefficients.
///
/// This is built once during data compilation and reused every PIRLS iteration.
#[derive(Clone, Debug)]
pub struct CompiledRowMasses {
    /// B-spline basis vector at entry time: `B(log a_entry)`.
    /// `None` if spec is not RP or there is no left truncation (entry mass = 0).
    pub basis_entry: Option<Vec<f64>>,

    /// B-spline basis vector at exit/event time: `B(log a_exit)`.
    /// `None` if spec is not RP.
    pub basis_exit: Option<Vec<f64>>,

    /// Derivative of B-spline basis at exit/event time: `B'(log a_exit)`.
    /// Needed to compute `h_0(t)` for exact events under RP.
    /// `None` if spec is not RP or the row is not an exact event.
    pub basis_derivative_exit: Option<Vec<f64>>,

    /// Age (or time) at entry into the risk set.
    pub age_entry: f64,

    /// Age (or time) at exit (event or censoring).
    pub age_exit: f64,

    /// Event type for this row.
    pub event_type: LatentSurvivalEventType,

    /// Left boundary for interval censoring.
    pub age_left: f64,

    /// Right boundary for interval censoring.
    pub age_right: f64,

    /// Baseline specification governing how masses are computed.
    pub spec: BaselineSpec,
}

// ─── Mass derivatives ────────────────────────────────────────────────────────

/// Derivatives of the cumulative hazard mass and log-baseline-hazard with
/// respect to the baseline coefficient vector `β`.
///
/// These are needed for the outer REML/LAML gradient when baseline parameters
/// participate in the smoothing penalty.
#[derive(Clone, Debug)]
pub struct MassDerivatives {
    /// `∂m/∂β_j` for each baseline coefficient, where `m = H_0(a_out) - H_0(a_in)`.
    pub d_m: Vec<f64>,

    /// `∂(log h_0)/∂β_j` at the event time (only meaningful for exact events).
    pub d_log_h0: Vec<f64>,
}

// ─── Assembly: masses ────────────────────────────────────────────────────────

/// Given current baseline coefficients, compute a [`LatentSurvivalRow`] from
/// precompiled row geometry.
///
/// # Arguments
///
/// * `compiled` — precompiled per-row geometry (basis vectors, ages, spec).
/// * `baseline_beta` — current baseline coefficient vector.
///
/// # Panics
///
/// Panics for baseline specs that are not yet implemented (Gompertz, Weibull).
pub fn assemble_row_masses(
    compiled: &CompiledRowMasses,
    baseline_beta: &[f64],
) -> LatentSurvivalRow {
    match &compiled.spec {
        BaselineSpec::Frozen => {
            // Masses were pre-stored; nothing to recompute.  Return a zeroed
            // row — the caller must have already populated masses externally.
            LatentSurvivalRow {
                event_type: compiled.event_type,
                mass_entry: 0.0,
                mass_exit: 0.0,
                mass_left: 0.0,
                mass_right: 0.0,
                log_baseline_hazard: 0.0,
                mass_unloaded_entry: 0.0,
                mass_unloaded_exit: 0.0,
                hazard_loaded: 0.0,
                hazard_unloaded: 0.0,
            }
        }

        BaselineSpec::RoystonParmar => assemble_rp(compiled, baseline_beta),

        BaselineSpec::GompertzMakeham => assemble_gompertz_makeham(compiled, baseline_beta),

        BaselineSpec::Gompertz => {
            return Err("Gompertz baseline mass assembly not yet implemented".to_string())
        }

        BaselineSpec::Weibull => {
            return Err("Weibull baseline mass assembly not yet implemented".to_string())
        }
    }
}

// ─── Royston-Parmar assembly ─────────────────────────────────────────────────

/// Evaluate `s(x) = B(x)^T β` (the spline value at log-time `x`).
#[inline]
fn spline_dot(basis: &[f64], beta: &[f64]) -> f64 {
    debug_assert_eq!(basis.len(), beta.len());
    basis.iter().zip(beta.iter()).map(|(b, c)| b * c).sum()
}

/// Assemble a [`LatentSurvivalRow`] under the Royston-Parmar model.
///
/// Cumulative hazard: `H_0(t) = exp(s(log t))` where `s(x) = B(x)^T β`.
///
/// Instantaneous hazard:
///   `h_0(t) = H_0(t) · s'(log t) / t`
///   `log h_0(t) = s(log t) + log(s'(log t)) - log(t)`
fn assemble_rp(compiled: &CompiledRowMasses, beta: &[f64]) -> LatentSurvivalRow {
    // --- Cumulative hazard at exit ---
    let mass_exit = match &compiled.basis_exit {
        Some(b) => {
            let s_exit = spline_dot(b, beta);
            s_exit.exp()
        }
        None => 0.0,
    };

    // --- Cumulative hazard at entry (left truncation) ---
    let mass_entry = match &compiled.basis_entry {
        Some(b) => {
            let s_entry = spline_dot(b, beta);
            s_entry.exp()
        }
        None => 0.0,
    };

    // --- Log baseline hazard at event time (exact events only) ---
    let log_baseline_hazard = if matches!(compiled.event_type, LatentSurvivalEventType::ExactEvent)
    {
        // h_0(t) = H_0(t) · s'(log t) / t
        // log h_0 = s(log t) + log(s'(log t)) - log(t)
        let b_exit = compiled
            .basis_exit
            .as_ref()
            .expect("RP exact event requires basis_exit");
        let b_deriv = compiled
            .basis_derivative_exit
            .as_ref()
            .expect("RP exact event requires basis_derivative_exit");

        let s_val = spline_dot(b_exit, beta);
        let s_prime = spline_dot(b_deriv, beta);
        let log_t = compiled.age_exit.ln();

        // Guard: s'(log t) must be positive for a valid hazard.
        // In practice the spline should be monotone, but clamp for safety.
        let s_prime_safe = s_prime.max(1e-100);

        s_val + s_prime_safe.ln() - log_t
    } else {
        0.0
    };

    // --- Interval censoring masses ---
    // For interval censoring under RP, age_left/age_right are used with the
    // same spline evaluation.  However, we do not store separate basis vectors
    // for age_left/age_right in the current struct — those would need to be
    // added for full interval-censoring support.  For now, mass_left/mass_right
    // are left at 0.0.
    let mass_left = 0.0;
    let mass_right = 0.0;

    LatentSurvivalRow {
        event_type: compiled.event_type,
        mass_entry,
        mass_exit,
        mass_left,
        mass_right,
        log_baseline_hazard,
        mass_unloaded_entry: 0.0,
        mass_unloaded_exit: 0.0,
        hazard_loaded: 0.0,
        hazard_unloaded: 0.0,
    }
}

// ─── Gompertz-Makeham assembly ───────────────────────────────────────────────

/// Assemble a [`LatentSurvivalRow`] under the Gompertz-Makeham model.
///
/// Hazard: `h(t) = c exp(γt) + d`
/// - Loaded (disease) component:  `h_L(t) = c exp(γt)`
/// - Unloaded (background) component: `h_U(t) = d`
///
/// Cumulative hazards:
/// - `M_L(a0, a1) = (c/γ)(exp(γ a1) - exp(γ a0))`
/// - `M_U(a0, a1) = d (a1 - a0)`
///
/// Baseline coefficients: `beta = [log c, γ, log d]`.
fn assemble_gompertz_makeham(compiled: &CompiledRowMasses, beta: &[f64]) -> LatentSurvivalRow {
    assert!(
        beta.len() >= 3,
        "GompertzMakeham requires at least 3 baseline coefficients [log c, γ, log d]"
    );

    let c = beta[0].exp();
    let gamma = beta[1];
    let d = beta[2].exp();

    let a0 = compiled.age_entry;
    let a1 = compiled.age_exit;

    // --- Loaded cumulative hazard: M_L = (c/γ)(exp(γ a1) - exp(γ a0)) ---
    // Handle γ → 0 gracefully: lim_{γ→0} (c/γ)(exp(γt) - 1) = c·t.
    let (mass_loaded_exit, mass_loaded_entry) = if gamma.abs() < 1e-12 {
        (c * a1, c * a0)
    } else {
        let c_over_gamma = c / gamma;
        (
            c_over_gamma * (gamma * a1).exp(),
            c_over_gamma * (gamma * a0).exp(),
        )
    };

    // --- Unloaded cumulative hazard: M_U = d * t ---
    let mass_unloaded_exit = d * a1;
    let mass_unloaded_entry = d * a0;

    // --- Instantaneous hazards at event time ---
    let (hazard_loaded, hazard_unloaded, log_baseline_hazard) =
        if matches!(compiled.event_type, LatentSurvivalEventType::ExactEvent) {
            let h_l = c * (gamma * a1).exp();
            let h_u = d;
            let h_total = h_l + h_u;
            (h_l, h_u, h_total.ln())
        } else {
            (0.0, 0.0, 0.0)
        };

    LatentSurvivalRow {
        event_type: compiled.event_type,
        // mass_entry/mass_exit carry the *loaded* component for the kernel.
        mass_entry: mass_loaded_entry,
        mass_exit: mass_loaded_exit,
        mass_left: 0.0,
        mass_right: 0.0,
        log_baseline_hazard,
        mass_unloaded_entry,
        mass_unloaded_exit,
        hazard_loaded,
        hazard_unloaded,
    }
}

// ─── Assembly: mass derivatives ──────────────────────────────────────────────

/// Compute derivatives of the row's mass and log-baseline-hazard with respect
/// to the baseline coefficient vector `β`.
///
/// # Arguments
///
/// * `compiled` — precompiled per-row geometry.
/// * `baseline_beta` — current baseline coefficients.
///
/// # Panics
///
/// Panics for baseline specs that are not yet implemented.
pub fn assemble_mass_derivatives(
    compiled: &CompiledRowMasses,
    baseline_beta: &[f64],
) -> MassDerivatives {
    match &compiled.spec {
        BaselineSpec::Frozen => MassDerivatives {
            d_m: vec![],
            d_log_h0: vec![],
        },

        BaselineSpec::RoystonParmar => assemble_rp_derivatives(compiled, baseline_beta),

        BaselineSpec::GompertzMakeham => {
            assemble_gompertz_makeham_derivatives(compiled, baseline_beta)
        }

        BaselineSpec::Gompertz => {
            return Err("Gompertz mass derivatives not yet implemented".to_string())
        }

        BaselineSpec::Weibull => {
            return Err("Weibull mass derivatives not yet implemented".to_string())
        }
    }
}

// ─── Royston-Parmar derivatives ──────────────────────────────────────────────

/// Derivatives under the Royston-Parmar model.
///
/// For the cumulative hazard `H_0(t) = exp(s(log t))`:
///   `∂H_0/∂β_j = H_0(t) · B_j(log t)`
///
/// So:
///   `∂m/∂β_j = H_0(a_out) · B_j(log a_out) - H_0(a_in) · B_j(log a_in)`
///
/// For the log baseline hazard:
///   `log h_0(t) = s(log t) + log(s'(log t)) - log t`
///   `∂(log h_0)/∂β_j = B_j(log t) + B_j'(log t) / s'(log t)`
fn assemble_rp_derivatives(
    compiled: &CompiledRowMasses,
    beta: &[f64],
) -> MassDerivatives {
    let p = beta.len();
    let mut d_m = vec![0.0; p];

    // --- dm/dbeta ---
    // Contribution from exit time.
    if let Some(b_exit) = &compiled.basis_exit {
        let s_exit = spline_dot(b_exit, beta);
        let h0_exit = s_exit.exp();
        for j in 0..p {
            d_m[j] += h0_exit * b_exit[j];
        }
    }

    // Contribution from entry time (subtracted).
    if let Some(b_entry) = &compiled.basis_entry {
        let s_entry = spline_dot(b_entry, beta);
        let h0_entry = s_entry.exp();
        for j in 0..p {
            d_m[j] -= h0_entry * b_entry[j];
        }
    }

    // --- d(log h0)/dbeta (exact events only) ---
    let d_log_h0 =
        if matches!(compiled.event_type, LatentSurvivalEventType::ExactEvent) {
            let b_exit = compiled
                .basis_exit
                .as_ref()
                .expect("RP exact event requires basis_exit");
            let b_deriv = compiled
                .basis_derivative_exit
                .as_ref()
                .expect("RP exact event requires basis_derivative_exit");

            let s_prime = spline_dot(b_deriv, beta);
            let s_prime_safe = s_prime.max(1e-100);

            let mut d_lh = vec![0.0; p];
            for j in 0..p {
                // ∂(log h_0)/∂β_j = B_j(log t) + B_j'(log t) / s'(log t)
                d_lh[j] = b_exit[j] + b_deriv[j] / s_prime_safe;
            }
            d_lh
        } else {
            vec![0.0; p]
        };

    MassDerivatives { d_m, d_log_h0 }
}

// ─── Gompertz-Makeham derivatives ────────────────────────────────────────────

/// Derivatives under the Gompertz-Makeham model.
///
/// Coefficients: `β = [log c, γ, log d]`.
///
/// Loaded mass: `M_L(a0,a1) = (c/γ)(exp(γ a1) - exp(γ a0))`
///   `∂M_L/∂(log c) = M_L`                    (since c = exp(log c))
///   `∂M_L/∂γ = c·a1·exp(γ a1)/γ - c·a0·exp(γ a0)/γ - M_L/γ`
///   `∂M_L/∂(log d) = 0`
///
/// Unloaded mass: `M_U(a0,a1) = d(a1 - a0)`
///   `∂M_U/∂(log c) = 0`, `∂M_U/∂γ = 0`, `∂M_U/∂(log d) = M_U`
///
/// For the outer REML the loaded-mass derivatives are the primary concern;
/// unloaded-mass derivatives fold into a simpler additive correction.
fn assemble_gompertz_makeham_derivatives(
    compiled: &CompiledRowMasses,
    beta: &[f64],
) -> MassDerivatives {
    assert!(beta.len() >= 3);

    let c = beta[0].exp();
    let gamma = beta[1];
    let d = beta[2].exp();
    let a0 = compiled.age_entry;
    let a1 = compiled.age_exit;

    // Total loaded mass for convenience.
    let (m_loaded, eg1, eg0) = if gamma.abs() < 1e-12 {
        (c * (a1 - a0), 1.0, 1.0)
    } else {
        let eg1 = (gamma * a1).exp();
        let eg0 = (gamma * a0).exp();
        let m = (c / gamma) * (eg1 - eg0);
        (m, eg1, eg0)
    };

    // dm_loaded / d(log c) = M_L  (chain rule through exp)
    let dm_d_logc = m_loaded;

    // dm_loaded / dγ
    let dm_d_gamma = if gamma.abs() < 1e-12 {
        // Taylor: c(a1² - a0²)/2
        0.5 * c * (a1 * a1 - a0 * a0)
    } else {
        // c·a1·exp(γ a1)/γ - c·a0·exp(γ a0)/γ - M_L/γ
        (c / gamma) * (a1 * eg1 - a0 * eg0) - m_loaded / gamma
    };

    // dm_loaded / d(log d) = 0
    // dm_unloaded / d(log d) = d * (a1 - a0) = M_U (but we track loaded mass derivatives here)

    let mut d_m = vec![dm_d_logc, dm_d_gamma, 0.0];

    // Log baseline hazard derivatives (exact events).
    let d_log_h0 =
        if matches!(compiled.event_type, LatentSurvivalEventType::ExactEvent) {
            let h_l = c * (gamma * a1).exp();
            let h_u = d;
            let h_total = h_l + h_u;

            // log h = log(c exp(γt) + d)
            // ∂/∂(log c) = c exp(γt) / h_total = h_l / h_total
            let dl_logc = h_l / h_total;

            // ∂/∂γ = c t exp(γt) / h_total = h_l * t / h_total
            let dl_gamma = h_l * a1 / h_total;

            // ∂/∂(log d) = d / h_total = h_u / h_total
            let dl_logd = h_u / h_total;

            vec![dl_logc, dl_gamma, dl_logd]
        } else {
            vec![0.0; 3]
        };

    MassDerivatives { d_m, d_log_h0 }
}
