//! The normalizer of the constrained Laplace posterior (gam#2765).
//!
//! # What it is
//!
//! [`super`] defines the posterior a shape- or box-constrained fit reports: the penalized
//! likelihood restricted to `C = {β : Aβ ≥ b}` and renormalized, which to Laplace order is
//! `N(β_unc, M⁻¹)` truncated to `C`, with `β_unc = β̂ − M⁻¹g`, `g = ∇F(β̂)` (the KKT gradient
//! `A_actᵀμ`) and `M` the precision the criterion prices. This module computes that law's
//! normalizer, the quantity a REML/LAML criterion needs:
//!
//! ```text
//! −ln ∫_C e^{−F} ≈ F(β̂) + ½ ln|M| − (p/2) ln 2π + C,
//! C = −½ gᵀM⁻¹g − ln P(u ≥ 0),   u ~ N(m₀, W),   m₀ = Aβ̂ − b − AM⁻¹g,   W = AM⁻¹Aᵀ.
//! ```
//!
//! # Why it is continuous where a face determinant is not
//!
//! The expression names no active set. `β̂`, `g` and `M` move continuously with the outer
//! coordinates when a row leaves or joins the face (its multiplier and its slack pass through
//! zero together), so `C` does too. Pricing `½ln|ZᵀMZ|` on the face `Z = null(A_act)` instead
//! integrates over a space whose dimension changes with the active set: at a row's switch the
//! two sides differ by that row's `½ ln` of normal curvature, the kept-rank jump #2765's named
//! gate stalls on (+4.30 at kept rank 10 → 11, job 1244861; the corrected criterion differs by
//! less than its printed resolution across the same switch, job 1256362). With one row, `C` is
//! exactly `−½ln v − ½μ²v − ln Φ(−μ√v)` when the row is active with multiplier `μ` (`v = aᵀM⁻¹a`)
//! and `−ln Φ(d/√v)` when it is inactive at slack `d`; the two agree at `μ = d = 0`. Far from
//! the bound the active form tends to `ln μ + ½ln 2π`, the linear boundary Laplace factor a face
//! determinant omits.
//!
//! # Which rows
//!
//! A row whose standardized pre-truncation slack lies beyond the point where its truncated mass
//! `Φ̄(s)` falls below `f64::EPSILON` cannot move `ln P` at double precision. That is the
//! horizon [`super`]'s moment walk reads off the machine epsilon, and the same rule selects rows
//! here: a row crossing it moves `C` by at most `ε`, below the rounding of any criterion it is
//! added to. A row with no posterior spread along its normal (its normal in the kernel of the
//! pseudo-inverse the criterion prices) removes no mass and is skipped, as [`super`] skips it.
//! No dependence filter is needed: nothing below forms `W⁻¹`, so rows whose normals are linearly
//! dependent are admissible.
//!
//! # `ln P` by expectation propagation
//!
//! `P(u ≥ 0)` has no closed form beyond two rows. It is estimated by EP on the orthant's
//! half-line indicators (Cunningham, Hennig and Lacoste-Julien, 2011): deterministic, smooth in
//! `(m₀, W)`, and exact for one row and for rows whose normals are `M⁻¹`-orthogonal. A sweep
//! that moves `ln Z_EP` by no more than its own rounding band ends the iteration. The change need
//! not fall monotonically on the way: on strongly correlated rows it can grow for one sweep and
//! then contract geometrically. A sweep that fails to contract halves the step every later site
//! update takes toward its full update, which leaves the fixed points, and so every derivative
//! below, unchanged. The iteration is refused ([`ConeNormalizerRefusal::NotContracting`]) only
//! when a damped sweep moves no site parameter at all; nothing truncates it.
//!
//! # Derivatives
//!
//! At the EP fixed point `ln Z_EP` is stationary in the site parameters (each site's tilted and
//! approximating cavity moments agree), so its gradient is the fixed-site derivative of the
//! Gaussian part. With `T = diag(τ̃)` and `E = (I + WT)⁻¹`,
//!
//! ```text
//! γ := ∂lnP/∂m₀ = Eᵀ(ν̃ − T m₀),    Γ := ∂lnP/∂W = ½ (γγᵀ − EᵀT).
//! ```
//!
//! The Hessian is not stationary in the sites: along a direction they move by the linearized
//! fixed point `(I − ∂F/∂s) ds = ∂F/∂θ dθ`, a `2q × 2q` system, and `(γ, Γ)` are differentiated
//! through it. The criterion's derivatives chain through `y = M⁻¹g` and `R = M⁻¹Aᵀ`, so the
//! precision's motion is read only on `span{y, R}`.

use gam_math::probability::{normal_logcdf_derivatives, standard_normal_quantile};
use gam_math::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, s};

/// Why the constrained normalizer could not be formed at a trial point.
#[derive(Clone, Debug)]
pub enum ConeNormalizerRefusal {
    /// An input (a row, bound, coefficient, gradient or solve) was not finite.
    NonFinite { what: &'static str },
    /// An EP cavity lost positive precision, which a log-concave site cannot cause at a resolved
    /// posterior.
    CavityPrecision { row: usize, precision: f64, sweep: usize },
    /// EP's damped update stopped moving the sites before `ln Z_EP` settled to its rounding band.
    NotContracting { sweeps: usize, fraction: f64, last_change: f64, band: f64 },
    /// A system EP's derivatives solve is singular.
    Singular { reason: String },
}

impl std::fmt::Display for ConeNormalizerRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { what } => {
                write!(f, "constrained Laplace normalizer: non-finite {what} (gam#2765)")
            }
            Self::CavityPrecision { row, precision, sweep } => write!(
                f,
                "constrained Laplace normalizer: EP cavity precision {precision:e} at row {row} in \
                 sweep {sweep} (gam#2765)"
            ),
            Self::NotContracting { sweeps, fraction, last_change, band } => write!(
                f,
                "constrained Laplace normalizer: EP stopped contracting after {sweeps} sweeps: a \
                 sweep damped to fraction {fraction:e} moved no site while ln P still moved \
                 {last_change:e}, above its rounding band {band:e} (gam#2765)"
            ),
            Self::Singular { reason } => {
                write!(f, "constrained Laplace normalizer: {reason} (gam#2765)")
            }
        }
    }
}

impl From<ConeNormalizerRefusal> for String {
    fn from(refusal: ConeNormalizerRefusal) -> Self {
        refusal.to_string()
    }
}

/// `A⁻¹` by Gauss–Jordan elimination with partial pivoting.
fn invert(mut a: Array2<f64>, what: &str) -> Result<Array2<f64>, ConeNormalizerRefusal> {
    let n = a.nrows();
    let mut inverse = Array2::<f64>::eye(n);
    for col in 0..n {
        let mut pivot_row = col;
        for row in (col + 1)..n {
            if a[[row, col]].abs() > a[[pivot_row, col]].abs() {
                pivot_row = row;
            }
        }
        let pivot = a[[pivot_row, col]];
        if !(pivot != 0.0 && pivot.is_finite()) {
            return Err(ConeNormalizerRefusal::Singular {
                reason: format!("{what} has no pivot in column {col} of {n}"),
            });
        }
        if pivot_row != col {
            for k in 0..n {
                a.swap([col, k], [pivot_row, k]);
                inverse.swap([col, k], [pivot_row, k]);
            }
        }
        for k in 0..n {
            a[[col, k]] /= pivot;
            inverse[[col, k]] /= pivot;
        }
        for row in 0..n {
            let factor = a[[row, col]];
            if row == col || factor == 0.0 {
                continue;
            }
            for k in 0..n {
                a[[row, k]] -= factor * a[[col, k]];
                inverse[[row, k]] -= factor * inverse[[col, k]];
            }
        }
    }
    Ok(inverse)
}

/// `E = (I + W T)⁻¹`. `I + WT` is similar to the positive definite `I + T^½ W T^½` for every
/// `τ̃ ≥ 0`, so it is nonsingular whenever the sites are admissible.
fn inverse_i_plus_wt(w: &Array2<f64>, tau: &Array1<f64>) -> Result<Array2<f64>, ConeNormalizerRefusal> {
    let q = tau.len();
    let mut a = Array2::<f64>::eye(q);
    for i in 0..q {
        for j in 0..q {
            a[[i, j]] += w[[i, j]] * tau[j];
        }
    }
    invert(a, "I + WT")
}

/// `ln det(I + W T)`, through the Cholesky factor of `I + T^½ W T^½`, with the magnitude of its
/// pivot logarithms for the rounding band.
fn log_det_i_plus_wt(w: &Array2<f64>, tau: &Array1<f64>) -> Result<(f64, f64), ConeNormalizerRefusal> {
    let q = tau.len();
    let root = tau.mapv(f64::sqrt);
    let mut factor = Array2::<f64>::zeros((q, q));
    let (mut log_det, mut magnitude) = (0.0, 0.0);
    for j in 0..q {
        let mut diag = 1.0 + root[j] * w[[j, j]] * root[j];
        for k in 0..j {
            diag -= factor[[j, k]] * factor[[j, k]];
        }
        if !(diag > 0.0) {
            return Err(ConeNormalizerRefusal::Singular {
                reason: format!("I + T^½WT^½ lost positive definiteness at pivot {j} ({diag:e})"),
            });
        }
        let pivot = diag.sqrt();
        factor[[j, j]] = pivot;
        log_det += 2.0 * pivot.ln();
        magnitude += (2.0 * pivot.ln()).abs();
        for i in (j + 1)..q {
            let mut value = root[i] * w[[i, j]] * root[j];
            for k in 0..j {
                value -= factor[[i, k]] * factor[[j, k]];
            }
            factor[[i, j]] = value / pivot;
        }
    }
    Ok((log_det, magnitude))
}

/// One site's moment-matching update from its cavity `(τ_c, ν_c)`, with the partials of the new
/// site in the cavity. With `z = ν_c/√τ_c` and `f = ln Φ`:
/// `τ̃ = −τ_c ψ(z)`, `ψ = f″/(1 + f″)`, and `ν̃ = √τ_c χ(z)`, `χ = (f′ − z f″)/(1 + f″)`.
struct SiteUpdate {
    tau: f64,
    nu: f64,
    /// `[∂τ̃/∂τ_c, ∂τ̃/∂ν_c, ∂ν̃/∂τ_c, ∂ν̃/∂ν_c]`.
    jacobian: [f64; 4],
}

fn site_update(tau_c: f64, nu_c: f64) -> SiteUpdate {
    let root = tau_c.sqrt();
    let z = nu_c / root;
    let d = normal_logcdf_derivatives(z);
    let (f1, f2, f3) = (d[1], d[2], d[3]);
    let one_plus = 1.0 + f2;
    let psi = f2 / one_plus;
    let psi_prime = f3 / (one_plus * one_plus);
    let chi = (f1 - z * f2) / one_plus;
    let chi_prime = -f3 * (z + f1) / (one_plus * one_plus);
    SiteUpdate {
        tau: -tau_c * psi,
        nu: root * chi,
        jacobian: [
            -psi + 0.5 * z * psi_prime,
            -root * psi_prime,
            (chi - z * chi_prime) / (2.0 * root),
            chi_prime,
        ],
    }
}

/// `ln P(u ≥ 0)` for `u ~ N(m₀, W)` at a converged EP fixed point, with its first two
/// derivatives in `(m₀, W)`.
#[derive(Clone, Debug)]
pub struct OrthantLogMass {
    m0: Array1<f64>,
    w: Array2<f64>,
    tau: Array1<f64>,
    nu: Array1<f64>,
    /// `E = (I + W T)⁻¹` at the sites.
    e: Array2<f64>,
    log_mass: f64,
    sweeps: usize,
    /// The share of its full update each site took on the last sweep: 1 unless a sweep failed to
    /// contract.
    fraction: f64,
    /// `(I − ∂F/∂s)⁻¹` of the linearized fixed point, formed on the first derivative request.
    site_system_inverse: std::sync::OnceLock<Array2<f64>>,
}

impl OrthantLogMass {
    /// Converge EP on `P(u ≥ 0)`, `u ~ N(m₀, W)`.
    pub fn converge(m0: &Array1<f64>, w: &Array2<f64>) -> Result<Self, ConeNormalizerRefusal> {
        let q = m0.len();
        if m0.iter().chain(w.iter()).any(|value| !value.is_finite()) {
            return Err(ConeNormalizerRefusal::NonFinite { what: "orthant mean or covariance" });
        }
        let mut state = Self {
            m0: m0.clone(),
            w: w.clone(),
            tau: Array1::zeros(q),
            nu: Array1::zeros(q),
            e: Array2::eye(q),
            log_mass: 0.0,
            sweeps: 0,
            fraction: 1.0,
            site_system_inverse: std::sync::OnceLock::new(),
        };
        if q == 0 {
            return Ok(state);
        }
        let (mut previous, _) = state.evaluate_log_mass()?;
        let mut previous_change = f64::INFINITY;
        let mut fraction = 1.0_f64;
        loop {
            state.sweeps += 1;
            let (mut at_fixed_point, mut moved) = (true, false);
            // Sequential EP: each site reads the posterior its predecessors in the sweep left.
            // Moving site `j` by `(Δτ, Δν)` is a rank-one change of `I + WT`, so the posterior
            // follows by Sherman–Morrison in O(q²),
            //   `Σ ← Σ − c s sᵀ`, `μ ← μ + (Δν(1 − cΣ_jj) − cμ_j) s`,
            //   `s = Σ_{:j}`, `c = Δτ/(1 + ΔτΣ_jj)`,
            // where `1 + ΔτΣ_jj = Σ_jj(τ_c + τ̃_j^new) > 0` for an admissible cavity. `E` is
            // re-formed from the sites once per sweep, which also resets the rank-one updates'
            // accumulated rounding before `ln Z_EP` is read.
            let (mut sigma, mut mu) = state.posterior();
            for j in 0..q {
                let s_jj = sigma[[j, j]];
                let tau_c = 1.0 / s_jj - state.tau[j];
                let nu_c = mu[j] / s_jj - state.nu[j];
                if !(tau_c > 0.0 && tau_c.is_finite()) {
                    return Err(ConeNormalizerRefusal::CavityPrecision {
                        row: j,
                        precision: tau_c,
                        sweep: state.sweeps,
                    });
                }
                let update = site_update(tau_c, nu_c);
                at_fixed_point &= update.tau == state.tau[j] && update.nu == state.nu[j];
                let tau = (1.0 - fraction) * state.tau[j] + fraction * update.tau;
                let nu = (1.0 - fraction) * state.nu[j] + fraction * update.nu;
                moved |= tau != state.tau[j] || nu != state.nu[j];
                let (delta_tau, delta_nu) = (tau - state.tau[j], nu - state.nu[j]);
                state.tau[j] = tau;
                state.nu[j] = nu;
                if delta_tau == 0.0 && delta_nu == 0.0 {
                    continue;
                }
                let c = delta_tau / (1.0 + delta_tau * s_jj);
                let column = sigma.column(j).to_owned();
                let mean_step = delta_nu * (1.0 - c * s_jj) - c * mu[j];
                mu.scaled_add(mean_step, &column);
                for a in 0..q {
                    let scaled = c * column[a];
                    for b in 0..q {
                        sigma[[a, b]] -= scaled * column[b];
                    }
                }
            }
            state.e = inverse_i_plus_wt(&state.w, &state.tau)?;
            let (log_mass, magnitude) = state.evaluate_log_mass()?;
            let change = (log_mass - previous).abs();
            // Every term of ln Z_EP is formed in O(q²) rounded operations. A damped sweep moves
            // ln Z_EP by about `fraction` times what the full update would, so the band scales with
            // it: a short step is not mistaken for a settled one.
            let band = accumulation_growth(4 * q * q + 8 * q) * magnitude;
            if at_fixed_point || change <= fraction * band {
                state.log_mass = log_mass;
                state.fraction = fraction;
                return Ok(state);
            }
            if !moved {
                return Err(ConeNormalizerRefusal::NotContracting {
                    sweeps: state.sweeps,
                    fraction,
                    last_change: previous_change,
                    band,
                });
            }
            if !(change < previous_change) {
                fraction *= 0.5;
            }
            previous_change = change;
            previous = log_mass;
        }
    }

    /// The posterior approximation `(Σ, μ) = (E W, E (m₀ + W ν̃))`.
    fn posterior(&self) -> (Array2<f64>, Array1<f64>) {
        let sigma = self.e.dot(&self.w);
        let mu = self.e.dot(&(&self.m0 + &self.w.dot(&self.nu)));
        (sigma, mu)
    }

    /// Site `j`'s cavity read from a freshly formed posterior: the re-inverted reference the
    /// fixed-point tests sweep with.
    #[cfg(test)]
    fn cavity(&self, j: usize) -> (f64, f64) {
        let (sigma, mu) = self.posterior();
        let s_jj = sigma[[j, j]];
        (1.0 / s_jj - self.tau[j], mu[j] / s_jj - self.nu[j])
    }

    /// `ln Z_EP` and the summed magnitude of its terms,
    ///
    /// `Σ_j [ln Φ(z_j) + ½ln(1 + τ̃_j v_j) + ½(τ̃_j m_j − ν̃_j)²/(τ̃_j(1 + τ̃_j v_j))]
    ///  − ½ ln|I + WT| − ½ xᵀ(T + TWT)⁺x`,   `x = T m₀ − ν̃`,
    ///
    /// with `(m_j, v_j)` the cavity of site `j`. Neither `W⁻¹` nor `τ̃^{−½}` is formed.
    fn evaluate_log_mass(&self) -> Result<(f64, f64), ConeNormalizerRefusal> {
        let q = self.m0.len();
        let (sigma, mu) = self.posterior();
        let (mut total, mut magnitude) = (0.0, 0.0);
        for j in 0..q {
            let s_jj = sigma[[j, j]];
            let tau_c = 1.0 / s_jj - self.tau[j];
            let nu_c = mu[j] / s_jj - self.nu[j];
            if !(tau_c > 0.0) {
                return Err(ConeNormalizerRefusal::CavityPrecision {
                    row: j,
                    precision: tau_c,
                    sweep: self.sweeps,
                });
            }
            let v_c = 1.0 / tau_c;
            let m_c = nu_c * v_c;
            let t = self.tau[j];
            let log_cdf = normal_logcdf_derivatives(nu_c / tau_c.sqrt())[0];
            let site = if t > 0.0 {
                0.5 * (1.0 + t * v_c).ln() + 0.5 * (t * m_c - self.nu[j]).powi(2) / (t * (1.0 + t * v_c))
            } else {
                0.0
            };
            total += log_cdf + site;
            magnitude += log_cdf.abs() + site.abs();
        }
        let (log_det, log_det_magnitude) = log_det_i_plus_wt(&self.w, &self.tau)?;
        let quadratic = self.quadratic_part()?;
        total += -0.5 * log_det - 0.5 * quadratic;
        magnitude += 0.5 * log_det_magnitude + 0.5 * quadratic.abs();
        if !total.is_finite() {
            return Err(ConeNormalizerRefusal::NonFinite { what: "EP log mass" });
        }
        Ok((total, magnitude))
    }

    /// `xᵀ (T + TWT)⁺ x` with `x = T m₀ − ν̃`, restricted to the sites with `τ̃ > 0` (a site with
    /// `τ̃ = 0` has `ν̃ = 0` and contributes nothing), through `(T + TWT)⁻¹ = (I + WT)⁻¹ T⁻¹`.
    fn quadratic_part(&self) -> Result<f64, ConeNormalizerRefusal> {
        let support: Vec<usize> = (0..self.m0.len()).filter(|&j| self.tau[j] > 0.0).collect();
        let n = support.len();
        if n == 0 {
            return Ok(0.0);
        }
        let w_s = Array2::from_shape_fn((n, n), |(a, b)| self.w[[support[a], support[b]]]);
        let tau_s = Array1::from_shape_fn(n, |a| self.tau[support[a]]);
        let x = Array1::from_shape_fn(n, |a| {
            let j = support[a];
            self.tau[j] * self.m0[j] - self.nu[j]
        });
        let e_s = inverse_i_plus_wt(&w_s, &tau_s)?;
        let scaled = Array1::from_shape_fn(n, |a| x[a] / tau_s[a]);
        Ok(x.dot(&e_s.dot(&scaled)))
    }

    /// `ln P(u ≥ 0)`.
    pub fn log_mass(&self) -> f64 {
        self.log_mass
    }

    /// Sweeps EP took to reach its fixed point.
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }

    /// The share of its full update each site took on EP's last sweep.
    pub fn step_fraction(&self) -> f64 {
        self.fraction
    }

    /// `γ = ∂lnP/∂m₀ = Eᵀ(ν̃ − T m₀)`.
    pub fn mean_gradient(&self) -> Array1<f64> {
        let x = &self.nu - &(&self.tau * &self.m0);
        self.e.t().dot(&x)
    }

    /// `Γ = ∂lnP/∂W = ½(γγᵀ − EᵀT)`, so `d ln P = γᵀdm₀ + tr(Γ dW)`.
    pub fn covariance_gradient(&self) -> Array2<f64> {
        let gamma = self.mean_gradient();
        let q = gamma.len();
        Array2::from_shape_fn((q, q), |(i, j)| {
            0.5 * (gamma[i] * gamma[j] - self.e[[j, i]] * self.tau[j])
        })
    }

    /// `(dγ, dΓ)` along `(dm₀, dW)`, the sites moving with their linearized fixed point.
    pub fn gradient_motion(
        &self,
        dm0: &Array1<f64>,
        dw: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), ConeNormalizerRefusal> {
        let q = self.m0.len();
        if q == 0 {
            return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
        }
        let (dtau, dnu) = self.site_motion(dm0, dw)?;
        // dE = −E (dW T + W dT) E.
        let inner = Array2::from_shape_fn((q, q), |(i, j)| {
            dw[[i, j]] * self.tau[j] + self.w[[i, j]] * dtau[j]
        });
        let de = -self.e.dot(&inner).dot(&self.e);
        let x = &self.nu - &(&self.tau * &self.m0);
        let dx = &dnu - &(&dtau * &self.m0) - &(&self.tau * dm0);
        let dgamma = de.t().dot(&x) + self.e.t().dot(&dx);
        let gamma = self.mean_gradient();
        let dgamma_matrix = Array2::from_shape_fn((q, q), |(i, j)| {
            0.5 * (dgamma[i] * gamma[j] + gamma[i] * dgamma[j]
                - de[[j, i]] * self.tau[j]
                - self.e[[j, i]] * dtau[j])
        });
        Ok((dgamma, dgamma_matrix))
    }

    /// The sites' motion `(dτ̃, dν̃)` along `(dm₀, dW)`, from `(I − ∂F/∂s) ds = ∂F/∂θ dθ`.
    ///
    /// Per unit site change the posterior moves by `dΣ/dτ̃_k = −Σ_{:k}Σ_{k:}`,
    /// `dμ/dτ̃_k = −Σ_{:k} μ_k`, `dμ/dν̃_k = Σ_{:k}`; along `(dm₀, dW)` by `dΣ = E dW Eᵀ` and
    /// `dμ = E dm₀ + E dW (ν̃ − Tμ)`. Site `j` reads the cavity
    /// `(1/Σ_jj − τ̃_j, μ_j/Σ_jj − ν̃_j)`.
    fn site_motion(
        &self,
        dm0: &Array1<f64>,
        dw: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ConeNormalizerRefusal> {
        let q = self.m0.len();
        let (sigma, mu) = self.posterior();
        // Only the diagonal of `dΣ = E dW Eᵀ` enters a cavity.
        let e_dw = self.e.dot(dw);
        let shift = &self.nu - &(&self.tau * &mu);
        let d_mu = self.e.dot(dm0) + e_dw.dot(&shift);
        let mut rhs = Array1::<f64>::zeros(2 * q);
        for j in 0..q {
            let s_jj = sigma[[j, j]];
            let tau_c = 1.0 / s_jj - self.tau[j];
            let nu_c = mu[j] / s_jj - self.nu[j];
            let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = site_update(tau_c, nu_c).jacobian;
            let s2 = s_jj * s_jj;
            let d_sjj = e_dw.row(j).dot(&self.e.row(j));
            let (dtc, dnc) = (-d_sjj / s2, d_mu[j] / s_jj - mu[j] * d_sjj / s2);
            rhs[j] = dt_dtc * dtc + dt_dnc * dnc;
            rhs[q + j] = dn_dtc * dtc + dn_dnc * dnc;
        }
        let ds = self.site_fixed_point_inverse()?.dot(&rhs);
        Ok((ds.slice(s![0..q]).to_owned(), ds.slice(s![q..2 * q]).to_owned()))
    }

    /// `(I − ∂F/∂s)⁻¹` at the converged sites. It depends on the fixed point alone, not on the
    /// direction `(dm₀, dW)`, so it is formed once and shared by every coordinate and pair.
    fn site_fixed_point_inverse(&self) -> Result<&Array2<f64>, ConeNormalizerRefusal> {
        if let Some(inverse) = self.site_system_inverse.get() {
            return Ok(inverse);
        }
        let q = self.m0.len();
        let (sigma, mu) = self.posterior();
        let mut system = Array2::<f64>::eye(2 * q);
        for j in 0..q {
            let s_jj = sigma[[j, j]];
            let tau_c = 1.0 / s_jj - self.tau[j];
            let nu_c = mu[j] / s_jj - self.nu[j];
            let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = site_update(tau_c, nu_c).jacobian;
            let s2 = s_jj * s_jj;
            let cavity_rate = |d_sjj: f64, d_muj: f64| -> (f64, f64) {
                (-d_sjj / s2, d_muj / s_jj - mu[j] * d_sjj / s2)
            };
            for k in 0..q {
                let (dtc, dnc) =
                    cavity_rate(-sigma[[j, k]] * sigma[[j, k]], -sigma[[j, k]] * mu[k]);
                let dtc = if k == j { dtc - 1.0 } else { dtc };
                system[[j, k]] -= dt_dtc * dtc + dt_dnc * dnc;
                system[[q + j, k]] -= dn_dtc * dtc + dn_dnc * dnc;
                let (dtc, dnc) = cavity_rate(0.0, sigma[[j, k]]);
                let dnc = if k == j { dnc - 1.0 } else { dnc };
                system[[j, q + k]] -= dt_dtc * dtc + dt_dnc * dnc;
                system[[q + j, q + k]] -= dn_dtc * dtc + dn_dnc * dnc;
            }
        }
        let inverse = invert(system, "the linearized EP fixed point")?;
        Ok(self.site_system_inverse.get_or_init(|| inverse))
    }
}

/// The standardized slack beyond which a row's truncated mass `Φ̄(s)` is below `f64::EPSILON`.
fn slack_horizon() -> Result<f64, ConeNormalizerRefusal> {
    standard_normal_quantile(f64::EPSILON)
        .map(|quantile| -quantile)
        .map_err(|_| ConeNormalizerRefusal::NonFinite { what: "slack horizon" })
}

/// One outer coordinate's first-order motion of the state the normalizer reads: the mode response
/// `v = dβ̂/dθ`, the KKT gradient's total derivative `ġ`, and the precision's motion `Ṁ` applied to
/// `y = M⁻¹g` ([`ConeNormalizer::solved_gradient`]) and to each column of `R = M⁻¹Aᵀ`
/// ([`ConeNormalizer::normal_solves`]).
#[derive(Clone, Debug)]
pub struct ConeCoordinateMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    pub precision_rate_on_y: Array1<f64>,
    pub precision_rate_on_r: Array2<f64>,
}

/// One coordinate pair's second-order motion: `v_kl`, `g̈_kl`, and `M̈_kl` applied to `y` and to
/// each column of `R`.
#[derive(Clone, Debug)]
pub struct ConePairMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    pub precision_rate_on_y: Array1<f64>,
    pub precision_rate_on_r: Array2<f64>,
}

/// A coordinate's first derivative of `C`, with the rates its pairs reuse.
#[derive(Clone, Debug)]
pub struct ConeFirstOrder {
    pub derivative: f64,
    y_rate: Array1<f64>,
    m0_rate: Array1<f64>,
    w_rate: Array2<f64>,
    /// `M⁻¹ Ṁ R`.
    solved_rate_on_r: Array2<f64>,
}

/// The constrained Laplace normalizer's criterion share `C` at one inner mode, with the state its
/// outer derivatives contract.
#[derive(Clone, Debug)]
pub struct ConeNormalizer {
    value: f64,
    /// Unit-scaled rows inside the mass horizon, `q × p`.
    rows: Array2<f64>,
    y: Array1<f64>,
    r: Array2<f64>,
    gradient: Array1<f64>,
    orthant: OrthantLogMass,
}

impl ConeNormalizer {
    /// Evaluate `C` at a mode. `rows · β ≥ bounds` is the constraint system over the joint
    /// coefficients (any row scale), `gradient` is `∇F(β̂)`, and `solve` applies `M⁻¹` for the
    /// precision the criterion prices (its pseudo-inverse where the criterion prices one).
    pub fn evaluate(
        rows: &Array2<f64>,
        bounds: &Array1<f64>,
        beta: &Array1<f64>,
        gradient: &Array1<f64>,
        solve: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    ) -> Result<Self, ConeNormalizerRefusal> {
        if rows.iter().chain(bounds.iter()).any(|value| !value.is_finite()) {
            return Err(ConeNormalizerRefusal::NonFinite { what: "constraint rows or bounds" });
        }
        if beta.iter().chain(gradient.iter()).any(|value| !value.is_finite()) {
            return Err(ConeNormalizerRefusal::NonFinite { what: "mode or gradient" });
        }
        let horizon = slack_horizon()?;
        let y = solve(gradient);
        if y.iter().any(|value| !value.is_finite()) {
            return Err(ConeNormalizerRefusal::NonFinite { what: "M⁻¹g" });
        }
        let center = beta - &y;
        let p = beta.len();
        let mut kept: Vec<(Array1<f64>, Array1<f64>, f64)> = Vec::new();
        for row in 0..rows.nrows() {
            let norm = rows.row(row).dot(&rows.row(row)).sqrt();
            if !(norm > 0.0) {
                continue;
            }
            let unit = rows.row(row).mapv(|value| value / norm);
            let solved = solve(&unit);
            let variance = unit.dot(&solved);
            if !variance.is_finite() {
                return Err(ConeNormalizerRefusal::NonFinite { what: "constraint-normal variance" });
            }
            if !(variance > 0.0) {
                continue;
            }
            let mean = unit.dot(&center) - bounds[row] / norm;
            if mean / variance.sqrt() < horizon {
                kept.push((unit, solved, mean));
            }
        }
        let q = kept.len();
        let mut a = Array2::<f64>::zeros((q, p));
        let mut r = Array2::<f64>::zeros((p, q));
        let mut m0 = Array1::<f64>::zeros(q);
        for (i, (unit, solved, mean)) in kept.into_iter().enumerate() {
            a.row_mut(i).assign(&unit);
            r.column_mut(i).assign(&solved);
            m0[i] = mean;
        }
        let w = symmetrized(&a.dot(&r));
        let orthant = OrthantLogMass::converge(&m0, &w)?;
        let value = -0.5 * gradient.dot(&y) - orthant.log_mass();
        if !value.is_finite() {
            return Err(ConeNormalizerRefusal::NonFinite { what: "normalizer value" });
        }
        Ok(Self { value, rows: a, y, r, gradient: gradient.clone(), orthant })
    }

    /// `C = −½gᵀM⁻¹g − ln P(u ≥ 0)`.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Rows inside the mass horizon.
    pub fn retained_rows(&self) -> usize {
        self.rows.nrows()
    }

    /// EP sweeps at this mode.
    pub fn sweeps(&self) -> usize {
        self.orthant.sweeps()
    }

    /// The share of its full update each EP site took on the last sweep at this mode.
    pub fn ep_step_fraction(&self) -> f64 {
        self.orthant.step_fraction()
    }

    /// `ln P(u ≥ 0)`.
    pub fn log_mass(&self) -> f64 {
        self.orthant.log_mass()
    }

    /// `y = M⁻¹g`.
    pub fn solved_gradient(&self) -> &Array1<f64> {
        &self.y
    }

    /// `R = M⁻¹Aᵀ`, `p × q`.
    pub fn normal_solves(&self) -> &Array2<f64> {
        &self.r
    }

    /// First derivative along one coordinate, with the rates its pairs reuse. `solve` applies the
    /// same `M⁻¹` [`Self::evaluate`] read.
    pub fn first_order(
        &self,
        motion: &ConeCoordinateMotion,
        solve: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    ) -> ConeFirstOrder {
        let y_rate = solve(&(&motion.gradient_rate - &motion.precision_rate_on_y));
        let m0_rate = self.rows.dot(&motion.mode_response) - self.rows.dot(&y_rate);
        let w_rate = symmetrized(&(-self.r.t().dot(&motion.precision_rate_on_r)));
        let derivative = -0.5 * (motion.gradient_rate.dot(&self.y) + self.gradient.dot(&y_rate))
            - self.orthant.mean_gradient().dot(&m0_rate)
            - frobenius(&self.orthant.covariance_gradient(), &w_rate);
        let (p, q) = self.r.dim();
        let mut solved_rate_on_r = Array2::<f64>::zeros((p, q));
        for column in 0..q {
            solved_rate_on_r
                .column_mut(column)
                .assign(&solve(&motion.precision_rate_on_r.column(column).to_owned()));
        }
        ConeFirstOrder { derivative, y_rate, m0_rate, w_rate, solved_rate_on_r }
    }

    /// Second derivative for the coordinate pair `(k, l)`.
    ///
    /// `M ÿ = g̈ − Ṁ_l ẏ_k − Ṁ_k ẏ_l − M̈ y`, read only through `yᵀ(·)` and `Rᵀ(·)`, so `Ṁ_l ẏ_k`
    /// enters as `(Ṁ_l y)ᵀẏ_k` and `(Ṁ_l R)ᵀẏ_k`; `Ẅ = Rᵀ(Ṁ_k M⁻¹ Ṁ_l + Ṁ_l M⁻¹ Ṁ_k − M̈)R`.
    pub fn second_order(
        &self,
        motion_k: &ConeCoordinateMotion,
        first_k: &ConeFirstOrder,
        motion_l: &ConeCoordinateMotion,
        first_l: &ConeFirstOrder,
        pair: &ConePairMotion,
    ) -> Result<f64, ConeNormalizerRefusal> {
        let y_moved = pair.gradient_rate.dot(&self.y)
            - motion_l.precision_rate_on_y.dot(&first_k.y_rate)
            - motion_k.precision_rate_on_y.dot(&first_l.y_rate)
            - pair.precision_rate_on_y.dot(&self.y);
        let second_gy = pair.gradient_rate.dot(&self.y)
            + motion_k.gradient_rate.dot(&first_l.y_rate)
            + motion_l.gradient_rate.dot(&first_k.y_rate)
            + y_moved;
        let normal_moved = self.r.t().dot(&pair.gradient_rate)
            - motion_l.precision_rate_on_r.t().dot(&first_k.y_rate)
            - motion_k.precision_rate_on_r.t().dot(&first_l.y_rate)
            - self.r.t().dot(&pair.precision_rate_on_y);
        let m0_second = self.rows.dot(&pair.mode_response) - normal_moved;
        let cross = motion_k.precision_rate_on_r.t().dot(&first_l.solved_rate_on_r);
        let w_second = symmetrized(&(&cross + &cross.t() - self.r.t().dot(&pair.precision_rate_on_r)));
        let (d_gamma, d_big_gamma) = self.orthant.gradient_motion(&first_l.m0_rate, &first_l.w_rate)?;
        let second_log_mass = d_gamma.dot(&first_k.m0_rate)
            + frobenius(&d_big_gamma, &first_k.w_rate)
            + self.orthant.mean_gradient().dot(&m0_second)
            + frobenius(&self.orthant.covariance_gradient(), &w_second);
        Ok(-0.5 * second_gy - second_log_mass)
    }
}

fn symmetrized(matrix: &Array2<f64>) -> Array2<f64> {
    let q = matrix.nrows();
    Array2::from_shape_fn((q, q), |(i, j)| 0.5 * (matrix[[i, j]] + matrix[[j, i]]))
}

fn frobenius(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::probability::normal_logcdf;
    use ndarray::array;

    /// A rounding band for a quantity assembled from terms of total magnitude `magnitude` in
    /// `operations` rounded steps.
    fn band(operations: usize, magnitude: f64) -> f64 {
        accumulation_growth(operations) * magnitude
    }

    /// Central difference of `f` at `h` and `h/2`, Richardson-combined, with its error bar: the
    /// gap between the two levels plus the rounding a difference quotient amplifies.
    fn richardson(f: &dyn Fn(f64) -> f64, h: f64) -> (f64, f64) {
        let coarse = (f(h) - f(-h)) / (2.0 * h);
        let fine = (f(0.5 * h) - f(-0.5 * h)) / h;
        let estimate = (4.0 * fine - coarse) / 3.0;
        let rounding = f64::EPSILON * (f(h).abs() + f(-h).abs()) / h;
        (estimate, (fine - coarse).abs() + 4.0 * rounding)
    }

    fn log_det_spd(m: &Array2<f64>) -> f64 {
        let n = m.nrows();
        let mut factor = Array2::<f64>::zeros((n, n));
        let mut log_det = 0.0;
        for j in 0..n {
            let mut diag = m[[j, j]];
            for k in 0..j {
                diag -= factor[[j, k]] * factor[[j, k]];
            }
            assert!(diag > 0.0, "the test face precision is positive definite");
            factor[[j, j]] = diag.sqrt();
            log_det += diag.ln();
            for i in (j + 1)..n {
                let mut value = m[[i, j]];
                for k in 0..j {
                    value -= factor[[i, k]] * factor[[j, k]];
                }
                factor[[i, j]] = value / factor[[j, j]];
            }
        }
        log_det
    }

    fn dense_solve(m: &Array2<f64>) -> impl Fn(&Array1<f64>) -> Array1<f64> {
        let inverse = invert(m.clone(), "test precision").expect("the test precision is invertible");
        move |rhs: &Array1<f64>| inverse.dot(rhs)
    }

    /// The change of `ln Z_EP` need not fall monotonically. This orthant is where EP was refused on
    /// the #2765 named gate for one growing sweep (seven rows correlated up to 0.98; job 1264873,
    /// change 1.047e-5 after 1.546e-7 at sweep 5). Continued undamped from there, the same
    /// iteration contracted by about 0.17 a sweep and settled in eleven more. EP must damp through
    /// the growing sweep and stop at the fixed point: a further full sweep then moves `ln P` by no
    /// more than the rounding band an undamped sweep stops at.
    #[test]
    fn a_sweep_that_grows_once_is_damped_through_to_the_fixed_point_2765() {
        let m0 = array![
            1.4809995771564423e-7, 1.7717232615695905e-7, 2.0363414175511165e-7, 2.2606609011260675e-7,
            8.162448125194582e-8, 5.321724567424204e-8, 2.9476733560602725e-8
        ];
        let w = array![
            [2.9508210065971846e-9, 3.4958356571107647e-9, 3.912754533017762e-9, 4.464972506819625e-9,
             1.6279726304593417e-9, 1.0646084495901858e-9, 5.893937617899781e-10],
            [3.4958356571107647e-9, 4.285289722109347e-9, 4.639471432395575e-9, 5.36668453191144e-9,
             1.9512472011207427e-9, 1.2777161142067118e-9, 7.071422079398519e-10],
            [3.912754533017762e-9, 4.639471432395575e-9, 5.473951938184774e-9, 6.063487293971178e-9,
             2.224874801505365e-9, 1.4505891376259968e-9, 8.036798655566868e-10],
            [4.464972506819625e-9, 5.36668453191144e-9, 6.063487293971178e-9, 7.097279381627257e-9,
             2.5735700905130125e-9, 1.6873873160829292e-9, 9.335753852933286e-10],
            [1.6279726304593417e-9, 1.9512472011207427e-9, 2.224874801505365e-9, 2.5735700905130125e-9,
             9.527282262398787e-10, 6.182315168699378e-10, 3.4294010893329004e-10],
            [1.0646084495901858e-9, 1.2777161142067118e-9, 1.4505891376259968e-9, 1.6873873160829292e-9,
             6.182315168699378e-10, 4.084376808974685e-10, 2.2535505405812938e-10],
            [5.893937617899781e-10, 7.071422079398519e-10, 8.036798655566868e-10, 9.335753852933286e-10,
             3.4294010893329004e-10, 2.2535505405812938e-10, 1.250961094155234e-10]
        ];
        let mass = OrthantLogMass::converge(&m0, &w)
            .unwrap_or_else(|refusal| panic!("EP settles on the recorded orthant: {refusal}"));
        let mut checked = mass.clone();
        let q = m0.len();
        for j in 0..q {
            let (tau_c, nu_c) = checked.cavity(j);
            let update = site_update(tau_c, nu_c);
            checked.tau[j] = update.tau;
            checked.nu[j] = update.nu;
            checked.e = inverse_i_plus_wt(&checked.w, &checked.tau).expect("admissible sites");
        }
        let (after, magnitude) = checked.evaluate_log_mass().expect("a finite EP log mass");
        let band = accumulation_growth(4 * q * q + 8 * q) * magnitude;
        eprintln!(
            "[2765-EP] ln P {:.15e} after {} sweeps at fraction {}; a full sweep moves it {:e} (band {band:e})",
            mass.log_mass(),
            mass.sweeps(),
            mass.step_fraction(),
            (after - mass.log_mass()).abs()
        );
        assert!(
            mass.step_fraction() < 1.0,
            "the recorded orthant has a sweep that fails to contract, so EP damps (fraction {})",
            mass.step_fraction()
        );
        assert!(
            (after - mass.log_mass()).abs() <= band,
            "a full sweep from the damped stop moves ln P by {:e}, above the band {band:e}",
            (after - mass.log_mass()).abs()
        );
    }

    #[test]
    fn one_row_orthant_mass_is_the_normal_log_cdf_with_exact_derivatives_2765() {
        for &(m0, w) in &[(-30.0, 2.0), (-1.3, 0.7), (0.0, 1.0), (0.4, 3.0), (5.0, 0.5)] {
            let mass = OrthantLogMass::converge(&array![m0], &array![[w]]).expect("one row converges");
            let z = m0 / f64::sqrt(w);
            let d = normal_logcdf_derivatives(z);
            let exact = normal_logcdf(z);
            assert!(
                (mass.log_mass() - exact).abs() <= band(64, exact.abs().max(1.0)),
                "ln P {} against ln Φ {exact} at (m0, W) = ({m0}, {w})",
                mass.log_mass()
            );
            let gamma = mass.mean_gradient()[0];
            let expected_gamma = d[1] / w.sqrt();
            assert!(
                (gamma - expected_gamma).abs() <= band(64, expected_gamma.abs().max(f64::MIN_POSITIVE)),
                "γ {gamma} against f′/√W {expected_gamma} at z = {z}"
            );
            let big_gamma = mass.covariance_gradient()[[0, 0]];
            let expected_big_gamma = -z * d[1] / (2.0 * w);
            assert!(
                (big_gamma - expected_big_gamma).abs()
                    <= band(64, expected_big_gamma.abs().max(d[1].abs() / w)),
                "Γ {big_gamma} against −zf′/(2W) {expected_big_gamma} at z = {z}"
            );
            let w32 = w * w.sqrt();
            let (dg_m, dbg_m) = mass.gradient_motion(&array![1.0], &array![[0.0]]).expect("motion");
            let (dg_w, dbg_w) = mass.gradient_motion(&array![0.0], &array![[1.0]]).expect("motion");
            let cross = -(z * d[2] + d[1]) / (2.0 * w32);
            let expected = [
                (dg_m[0], d[2] / w),
                (dbg_m[[0, 0]], cross),
                (dg_w[0], cross),
                (dbg_w[[0, 0]], z * (d[1] + z * d[2]) / (4.0 * w * w) + z * d[1] / (2.0 * w * w)),
            ];
            // The site update divides by 1 + f″, which is O(1/z²) deep in the left tail.
            let amplification = 1.0 / (1.0 + d[2]).powi(2);
            let scale = amplification * (d[1].abs() + d[2].abs() * (1.0 + z * z) + 1.0)
                / (w.min(1.0) * w.min(1.0));
            for (index, (got, want)) in expected.iter().enumerate() {
                assert!(
                    (got - want).abs() <= band(256, scale.max(f64::MIN_POSITIVE)),
                    "second-order entry {index}: {got} against {want} at z = {z}"
                );
            }
        }
    }

    #[test]
    fn rows_with_independent_normals_factor_exactly_2765() {
        let m0 = array![-2.0, 0.3, 1.7];
        let w = array![[0.5, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.1]];
        let mass = OrthantLogMass::converge(&m0, &w).expect("independent rows converge");
        let exact: f64 = (0..3).map(|i| normal_logcdf(m0[i] / w[[i, i]].sqrt())).sum();
        assert!(
            (mass.log_mass() - exact).abs() <= band(256, exact.abs().max(1.0)),
            "ln P {} against Σ ln Φ {exact}",
            mass.log_mass()
        );
        let gamma = mass.mean_gradient();
        for i in 0..3 {
            let z = m0[i] / w[[i, i]].sqrt();
            let want = normal_logcdf_derivatives(z)[1] / w[[i, i]].sqrt();
            assert!(
                (gamma[i] - want).abs() <= band(256, want.abs()),
                "γ_{i} {} against {want}",
                gamma[i]
            );
        }
    }

    /// gam#3037: EP re-formed `E = (I + WT)⁻¹` by a fresh `q × q` inversion after every site, an
    /// `O(q⁴)` sweep that held the 3828-row delayed-entry location-scale fit inside one normalizer
    /// evaluation for over ten minutes. Each site move is a rank-one change of `I + WT`, so the
    /// sweep follows it by Sherman–Morrison. The fixed point is unchanged: a sweep taken the old
    /// way, re-inverting after every site, from where the rank-one sweeps stop moves `ln P` by no
    /// more than its rounding band, and the posterior the sweeps carried agrees with the one
    /// re-formed from the sites.
    #[test]
    fn rank_one_ep_sweeps_reach_the_re_inverted_fixed_point_3037() {
        let q = 24;
        // A smooth monotone-guard-like covariance: neighbouring rows strongly correlated.
        let w = Array2::from_shape_fn((q, q), |(i, j)| {
            let d = (i as f64 - j as f64) / 4.0;
            (-0.5 * d * d).exp() + if i == j { 0.05 } else { 0.0 }
        });
        let m0 = Array1::from_shape_fn(q, |i| 0.6 * ((i as f64) * 0.7).sin() - 0.2);
        let mass = OrthantLogMass::converge(&m0, &w)
            .unwrap_or_else(|refusal| panic!("EP settles on the correlated orthant: {refusal}"));
        let mut checked = mass.clone();
        for j in 0..q {
            let (tau_c, nu_c) = checked.cavity(j);
            let update = site_update(tau_c, nu_c);
            checked.tau[j] = update.tau;
            checked.nu[j] = update.nu;
            checked.e = inverse_i_plus_wt(&checked.w, &checked.tau).expect("admissible sites");
        }
        let (after, magnitude) = checked.evaluate_log_mass().expect("a finite EP log mass");
        let step_band = accumulation_growth(4 * q * q + 8 * q) * magnitude;
        eprintln!(
            "[3037-EP] ln P {:.15e} after {} sweeps; a re-inverted sweep moves it {:e} (band {step_band:e})",
            mass.log_mass(),
            mass.sweeps(),
            (after - mass.log_mass()).abs()
        );
        assert!(
            (after - mass.log_mass()).abs() <= step_band,
            "a re-inverted sweep from the rank-one stop moves ln P by {:e}, above the band {step_band:e}",
            (after - mass.log_mass()).abs()
        );
        assert!(mass.log_mass() < 0.0 && mass.log_mass().is_finite());
    }

    #[test]
    fn ep_derivatives_are_derivatives_of_the_ep_log_mass_2765() {
        let m0 = array![-1.5, 0.2, 1.0];
        let w = array![[1.0, 0.5, -0.3], [0.5, 2.0, 0.4], [-0.3, 0.4, 0.8]];
        let mass = OrthantLogMass::converge(&m0, &w).expect("correlated rows converge");
        let gamma = mass.mean_gradient();
        let big_gamma = mass.covariance_gradient();
        let dm0 = array![0.3, -0.7, 0.5];
        let dw = array![[0.2, -0.1, 0.05], [-0.1, 0.3, 0.1], [0.05, 0.1, -0.15]];
        let log_mass_along = |t: f64| {
            OrthantLogMass::converge(&(&m0 + &(&dm0 * t)), &(&w + &(&dw * t)))
                .expect("the path converges")
                .log_mass()
        };
        let (fd, bar) = richardson(&log_mass_along, 1.0e-2);
        let analytic = gamma.dot(&dm0) + frobenius(&big_gamma, &dw);
        assert!(
            (analytic - fd).abs() <= bar,
            "d ln P {analytic} against central difference {fd} (bar {bar})"
        );
        let (dgamma, dbig_gamma) = mass.gradient_motion(&dm0, &dw).expect("site motion");
        let directional = |t: f64| {
            let moved = OrthantLogMass::converge(&(&m0 + &(&dm0 * t)), &(&w + &(&dw * t)))
                .expect("the path converges");
            moved.mean_gradient().dot(&dm0) + frobenius(&moved.covariance_gradient(), &dw)
        };
        let (fd2, bar2) = richardson(&directional, 1.0e-2);
        let analytic2 = dgamma.dot(&dm0) + frobenius(&dbig_gamma, &dw);
        assert!(
            (analytic2 - fd2).abs() <= bar2,
            "d² ln P {analytic2} against central difference {fd2} (bar {bar2})"
        );
    }

    /// The constrained minimizer of `½(β − c)ᵀH(β − c)` subject to `β_i ≥ 0` for `i` in `bounded`,
    /// by enumerating active sets (the KKT point is unique for positive definite `H`).
    fn constrained_quadratic_mode(
        h: &Array2<f64>,
        c: &Array1<f64>,
        bounded: &[usize],
    ) -> (Array1<f64>, Array1<f64>, Vec<usize>) {
        let p = c.len();
        for mask in 0..(1usize << bounded.len()) {
            let active: Vec<usize> =
                (0..bounded.len()).filter(|bit| mask & (1 << bit) != 0).map(|bit| bounded[bit]).collect();
            let free: Vec<usize> = (0..p).filter(|i| !active.contains(i)).collect();
            let mut beta = Array1::<f64>::zeros(p);
            if !free.is_empty() {
                let h_ff = Array2::from_shape_fn((free.len(), free.len()), |(a, b)| h[[free[a], free[b]]]);
                let rhs = Array1::from_shape_fn(free.len(), |a| {
                    (0..p).map(|j| h[[free[a], j]] * c[j]).sum::<f64>()
                });
                let solved = invert(h_ff, "free block").expect("free block").dot(&rhs);
                for (a, &i) in free.iter().enumerate() {
                    beta[i] = solved[a];
                }
            }
            let gradient = h.dot(&(&beta - c));
            let feasible = bounded.iter().all(|&i| beta[i] >= 0.0);
            let dual = active.iter().all(|&i| gradient[i] >= 0.0);
            if feasible && dual {
                return (beta, gradient, active);
            }
        }
        panic!("a strictly convex quadratic has a KKT point");
    }

    #[test]
    fn the_normalizer_is_continuous_where_the_face_determinant_jumps_2765() {
        let h = array![[2.0, 0.8, 0.3], [0.8, 1.5, -0.4], [0.3, -0.4, 1.2]];
        let rows = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let bounds = array![0.0, 0.0];
        let solve = dense_solve(&h);
        // Row 1 stays pinned (c_1 < 0 throughout) while row 0's optimum crosses its bound as t
        // moves c_0; the crossing t* is located by bisection on the active set.
        let criterion = |t: f64| {
            let c = array![t, -0.6, 0.4];
            let (beta, gradient, active) = constrained_quadratic_mode(&h, &c, &[0, 1]);
            let normalizer =
                ConeNormalizer::evaluate(&rows, &bounds, &beta, &gradient, &solve).expect("normalizer");
            let free: Vec<usize> = (0..3).filter(|i| !active.contains(i)).collect();
            let face = Array2::from_shape_fn((free.len(), free.len()), |(a, b)| h[[free[a], free[b]]]);
            (normalizer.value(), 0.5 * log_det_spd(&face), active.len())
        };
        let (mut lo, mut hi) = (-2.0, 2.0);
        assert_ne!(criterion(lo).2, criterion(hi).2, "the path crosses a face switch");
        while hi - lo > 1.0e-12 {
            let mid = 0.5 * (lo + hi);
            if criterion(mid).2 == criterion(lo).2 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let switch = 0.5 * (lo + hi);
        let (below, face_below, active_below) = criterion(switch - 1.0e-6);
        let (above, face_above, active_above) = criterion(switch + 1.0e-6);
        assert_ne!(active_below, active_above, "the probes straddle the switch");
        let (below_half, face_below_half, _) = criterion(switch - 0.5e-6);
        let (above_half, face_above_half, _) = criterion(switch + 0.5e-6);
        let gap = (above - below).abs();
        let gap_half = (above_half - below_half).abs();
        assert!(
            gap_half <= 0.75 * gap || gap <= band(512, below.abs().max(1.0)),
            "C is continuous through the switch: gap {gap:e} at ±1e-6, {gap_half:e} at ±5e-7"
        );
        let face_gap = (face_above - face_below).abs();
        let face_gap_half = (face_above_half - face_below_half).abs();
        assert!(
            face_gap_half > 0.75 * face_gap && face_gap > 1.0e3 * gap,
            "positive control: the face determinant jumps ({face_gap:e}, {face_gap_half:e}) where \
             C does not ({gap:e})"
        );
    }

    #[test]
    fn the_normalizer_derivatives_match_central_differences_2765() {
        let rows = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8]];
        let bounds = array![0.0, 0.0, -0.05];
        let m0 = array![[2.0, 0.8, 0.3], [0.8, 1.5, -0.4], [0.3, -0.4, 1.2]];
        let m1 = array![[0.3, -0.1, 0.0], [-0.1, 0.2, 0.05], [0.0, 0.05, -0.1]];
        let m2 = array![[0.05, 0.0, 0.02], [0.0, -0.04, 0.0], [0.02, 0.0, 0.03]];
        let (b0, b1, b2) = (array![0.01, 0.02, 0.3], array![-0.2, 0.1, 0.05], array![0.05, -0.1, 0.02]);
        let (g0, g1, g2) = (array![0.4, 0.3, 0.0], array![0.1, -0.2, 0.05], array![-0.03, 0.02, 0.01]);
        let state = |t: f64| {
            (
                &b0 + &(&b1 * t) + &(&b2 * (t * t)),
                &g0 + &(&g1 * t) + &(&g2 * (t * t)),
                &m0 + &(&m1 * t) + &(&m2 * (t * t)),
            )
        };
        let value_at = |t: f64| {
            let (beta, gradient, m) = state(t);
            let solve = dense_solve(&m);
            ConeNormalizer::evaluate(&rows, &bounds, &beta, &gradient, &solve)
                .expect("normalizer")
                .value()
        };
        let motion_at = |t: f64, normalizer: &ConeNormalizer| {
            let m_rate = &m1 + &(&m2 * (2.0 * t));
            ConeCoordinateMotion {
                mode_response: &b1 + &(&b2 * (2.0 * t)),
                gradient_rate: &g1 + &(&g2 * (2.0 * t)),
                precision_rate_on_y: m_rate.dot(normalizer.solved_gradient()),
                precision_rate_on_r: m_rate.dot(normalizer.normal_solves()),
            }
        };
        let derivative_at = |t: f64| {
            let (beta, gradient, m) = state(t);
            let solve = dense_solve(&m);
            let normalizer =
                ConeNormalizer::evaluate(&rows, &bounds, &beta, &gradient, &solve).expect("normalizer");
            normalizer.first_order(&motion_at(t, &normalizer), &solve).derivative
        };
        let (beta, gradient, m) = state(0.0);
        let solve = dense_solve(&m);
        let normalizer =
            ConeNormalizer::evaluate(&rows, &bounds, &beta, &gradient, &solve).expect("normalizer");
        assert!(normalizer.retained_rows() >= 2, "at least two correlated rows are near their bounds");
        let motion = motion_at(0.0, &normalizer);
        let first = normalizer.first_order(&motion, &solve);
        let (fd, bar) = richardson(&value_at, 1.0e-2);
        assert!(
            (first.derivative - fd).abs() <= bar,
            "dC {} against central difference {fd} (bar {bar})",
            first.derivative
        );
        let pair = ConePairMotion {
            mode_response: &b2 * 2.0,
            gradient_rate: &g2 * 2.0,
            precision_rate_on_y: (&m2 * 2.0).dot(normalizer.solved_gradient()),
            precision_rate_on_r: (&m2 * 2.0).dot(normalizer.normal_solves()),
        };
        let second = normalizer
            .second_order(&motion, &first, &motion, &first, &pair)
            .expect("second order");
        let (fd2, bar2) = richardson(&derivative_at, 1.0e-2);
        assert!(
            (second - fd2).abs() <= bar2,
            "d²C {second} against central difference of dC {fd2} (bar {bar2})"
        );
    }
}
