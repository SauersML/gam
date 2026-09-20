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
//! dependent are admissible. A row repeated exactly is the same half-space, and the intersection
//! holds it once, so it enters once. EP would give each copy its own site and count its
//! truncation once per copy. A family whose cone is declared over its data rows repeats a row
//! for every tied data row: the transformation-normal cone `ψ_iᵀA_k ≥ 0` repeats each response
//! row once per data row of an intercept-only fit.
//!
//! # `ln P` by expectation propagation
//!
//! `P(u ≥ 0)` has no closed form beyond two rows. It is estimated by EP on the orthant's
//! half-line indicators (Cunningham, Hennig and Lacoste-Julien, 2011): deterministic, smooth in
//! `(m₀, W)`, and exact for one row and for rows whose normals are `M⁻¹`-orthogonal. Each sweep
//! visits the sites in order, carrying the posterior from one site to the next by the rank-one
//! update the site's move makes to its precision, `O(q²)` a site, and forms the posterior
//! exactly again at the sweep's end. A sweep
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
    /// The linearized fixed point at these sites, formed on the first derivative that reads it.
    site_motion_system: std::sync::OnceLock<Result<SiteMotionSystem, ConeNormalizerRefusal>>,
}

/// What the sites' motion reads at a fixed point that does not depend on the direction of the
/// motion: the posterior `(Σ, μ)`, each site's update partials in its cavity, and the inverse of
/// `I − ∂F/∂s`. Every direction of every coordinate pair solves with the same inverse.
#[derive(Clone, Debug)]
struct SiteMotionSystem {
    sigma: Array2<f64>,
    mu: Array1<f64>,
    jacobians: Vec<[f64; 4]>,
    inverse: Array2<f64>,
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
            site_motion_system: std::sync::OnceLock::new(),
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
            // The sweep starts from the posterior of its sites, formed exactly, and carries it
            // from site to site by the rank-one update a site's move makes to the precision
            // `W⁻¹ + T`: with `c = Δτ̃/(1 + Δτ̃ Σ_jj)`, `Σ ← Σ − c Σ_{:j}Σ_{j:}` and
            // `μ ← μ + Σ_{:j}(Δν̃ − Δτ̃ μ_j)/(1 + Δτ̃ Σ_jj)`. Site `j` reads the same cavity it
            // would from `(E W, E(m₀ + W ν̃))` at the sites before it, in `O(q²)` instead of the
            // `O(q³)` of forming and inverting `I + WT` again. `1 + Δτ̃ Σ_jj = Σ_jj (τ_c + τ̃_j)` is
            // positive for an admissible cavity, since the new `τ̃_j` is not negative.
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
                let (d_tau, d_nu) = (tau - state.tau[j], nu - state.nu[j]);
                state.tau[j] = tau;
                state.nu[j] = nu;
                let denominator = 1.0 + d_tau * s_jj;
                let column = sigma.column(j).to_owned();
                let mean_step = (d_nu - d_tau * mu[j]) / denominator;
                mu.scaled_add(mean_step, &column);
                let shrink = d_tau / denominator;
                for a in 0..q {
                    let scaled = shrink * column[a];
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
    /// `(1/Σ_jj − τ̃_j, μ_j/Σ_jj − ν̃_j)`. Only the right-hand side depends on the direction; the
    /// system is the fixed point's own ([`Self::linearized_fixed_point`]).
    fn site_motion(
        &self,
        dm0: &Array1<f64>,
        dw: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ConeNormalizerRefusal> {
        let q = self.m0.len();
        let system = self.linearized_fixed_point()?;
        let (sigma, mu) = (&system.sigma, &system.mu);
        let d_sigma = self.e.dot(dw).dot(&self.e.t());
        let shift = &self.nu - &(&self.tau * mu);
        let d_mu = self.e.dot(dm0) + self.e.dot(&dw.dot(&shift));
        let mut rhs = Array1::<f64>::zeros(2 * q);
        for j in 0..q {
            let s_jj = sigma[[j, j]];
            let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = system.jacobians[j];
            let s2 = s_jj * s_jj;
            let (dtc, dnc) = (-d_sigma[[j, j]] / s2, d_mu[j] / s_jj - mu[j] * d_sigma[[j, j]] / s2);
            rhs[j] = dt_dtc * dtc + dt_dnc * dnc;
            rhs[q + j] = dn_dtc * dtc + dn_dnc * dnc;
        }
        let ds = system.inverse.dot(&rhs);
        Ok((ds.slice(s![0..q]).to_owned(), ds.slice(s![q..2 * q]).to_owned()))
    }

    /// The posterior, the site partials and the inverse of the linearized fixed point's
    /// `2q × 2q` system `I − ∂F/∂s`, formed once for these sites.
    fn linearized_fixed_point(&self) -> Result<&SiteMotionSystem, ConeNormalizerRefusal> {
        self.site_motion_system
            .get_or_init(|| {
                let q = self.m0.len();
                let (sigma, mu) = self.posterior();
                let mut system = Array2::<f64>::eye(2 * q);
                let mut jacobians = Vec::with_capacity(q);
                for j in 0..q {
                    let s_jj = sigma[[j, j]];
                    let tau_c = 1.0 / s_jj - self.tau[j];
                    let nu_c = mu[j] / s_jj - self.nu[j];
                    let jacobian = site_update(tau_c, nu_c).jacobian;
                    let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = jacobian;
                    jacobians.push(jacobian);
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
                Ok(SiteMotionSystem { sigma, mu, jacobians, inverse })
            })
            .as_ref()
            .map_err(|refusal| refusal.clone())
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
///
/// Where `M⁻¹` is the criterion's kept-spectrum pseudo-inverse `M⁺`, its derivative is not
/// `−M⁺ṀM⁺` alone: the kept eigenvectors rotate into the dropped ones. That part of `D(M⁺)[Ṁ]`
/// is applied to `g` and to each retained constraint normal ([`ConeNormalizer::retained_normals`])
/// in `inverse_rotation_on_gradient` and `inverse_rotation_on_normals`, zero where `M⁻¹` is an
/// inverse (gam#2952).
#[derive(Clone, Debug)]
pub struct ConeCoordinateMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    pub precision_rate_on_y: Array1<f64>,
    pub precision_rate_on_r: Array2<f64>,
    pub inverse_rotation_on_gradient: Array1<f64>,
    pub inverse_rotation_on_normals: Array2<f64>,
}

/// One coordinate pair's second-order motion: `v_kl`, `g̈_kl`, and `M̈_kl` applied to `y` and to
/// each column of `R`.
#[derive(Clone, Debug)]
pub struct ConePairMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    pub precision_rate_on_y: Array1<f64>,
    pub precision_rate_on_r: Array2<f64>,
    /// What the inverse identities in [`ConeNormalizer::second_order`] omit where `M⁻¹` is the
    /// criterion's kept-spectrum pseudo-inverse: `(D²M⁺[Ṁ_k, Ṁ_l] − M⁺Ṁ_kM⁺Ṁ_lM⁺ − M⁺Ṁ_lM⁺Ṁ_kM⁺)`
    /// plus the rotation of the pair drift `D M⁺[M̈] + M⁺M̈M⁺`, applied to `g` and to each retained
    /// normal. Zero where `M⁻¹` is an inverse (gam#2952).
    pub inverse_rotation_on_gradient: Array1<f64>,
    pub inverse_rotation_on_normals: Array2<f64>,
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
        let mut half_spaces = std::collections::HashSet::<Vec<u64>>::new();
        for row in 0..rows.nrows() {
            let norm = rows.row(row).dot(&rows.row(row)).sqrt();
            if !(norm > 0.0) {
                continue;
            }
            let unit = rows.row(row).mapv(|value| value / norm);
            let bound = bounds[row] / norm;
            // A row repeated exactly bounds the same half-space, and an intersection holds it
            // once: `P(u ≥ 0)` is the same with the repeat as without it. EP is not, since it
            // gives each copy its own site, so a repeat enters once.
            let half_space: Vec<u64> =
                unit.iter().chain(std::iter::once(&bound)).map(|value| (value + 0.0).to_bits()).collect();
            if !half_spaces.insert(half_space) {
                continue;
            }
            let solved = solve(&unit);
            let variance = unit.dot(&solved);
            if !variance.is_finite() {
                return Err(ConeNormalizerRefusal::NonFinite { what: "constraint-normal variance" });
            }
            if !(variance > 0.0) {
                continue;
            }
            let mean = unit.dot(&center) - bound;
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

    /// The unit normals of the rows inside the mass horizon, `Aᵀ`, `p × q`.
    pub fn retained_normals(&self) -> Array2<f64> {
        self.rows.t().to_owned()
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
        let y_rate = solve(&(&motion.gradient_rate - &motion.precision_rate_on_y))
            + &motion.inverse_rotation_on_gradient;
        let m0_rate = self.rows.dot(&motion.mode_response) - self.rows.dot(&y_rate);
        let w_rate = symmetrized(
            &(self.rows.dot(&motion.inverse_rotation_on_normals)
                - self.r.t().dot(&motion.precision_rate_on_r)),
        );
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
        // Where `M⁻¹` is a kept-spectrum pseudo-inverse, `ÿ` and `Ẅ` also carry what the identities
        // above omit: the pair's second rotation, and each coordinate's first rotation against the
        // other's rates (`ẏ` already carries the first rotation, which the identities read back
        // through `Ṁ ẏ`). All of it is zero where `M⁻¹` is an inverse.
        let (turn_k, turn_l) = (&motion_k.inverse_rotation_on_gradient, &motion_l.inverse_rotation_on_gradient);
        let y_turned = turn_k.dot(&motion_l.gradient_rate)
            + turn_l.dot(&motion_k.gradient_rate)
            + self.gradient.dot(&pair.inverse_rotation_on_gradient)
            + motion_l.precision_rate_on_y.dot(turn_k)
            + motion_k.precision_rate_on_y.dot(turn_l);
        let second_gy = pair.gradient_rate.dot(&self.y)
            + motion_k.gradient_rate.dot(&first_l.y_rate)
            + motion_l.gradient_rate.dot(&first_k.y_rate)
            + y_moved
            + y_turned;
        let normal_turned = motion_k.inverse_rotation_on_normals.t().dot(&motion_l.gradient_rate)
            + motion_l.inverse_rotation_on_normals.t().dot(&motion_k.gradient_rate)
            + self.rows.dot(&pair.inverse_rotation_on_gradient)
            + motion_l.precision_rate_on_r.t().dot(turn_k)
            + motion_k.precision_rate_on_r.t().dot(turn_l);
        let normal_moved = self.r.t().dot(&pair.gradient_rate)
            - motion_l.precision_rate_on_r.t().dot(&first_k.y_rate)
            - motion_k.precision_rate_on_r.t().dot(&first_l.y_rate)
            - self.r.t().dot(&pair.precision_rate_on_y)
            + normal_turned;
        let m0_second = self.rows.dot(&pair.mode_response) - normal_moved;
        let cross = motion_k.precision_rate_on_r.t().dot(&first_l.solved_rate_on_r);
        let w_second = symmetrized(
            &(&cross + &cross.t() - self.r.t().dot(&pair.precision_rate_on_r)
                + self.rows.dot(&pair.inverse_rotation_on_normals)),
        );
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

    /// One undamped EP sweep read straight off the definitions: each site's cavity from
    /// `Σ = EW` and `μ = E(m₀ + Wν̃)` re-formed from `E = (I + WT)⁻¹` after every site.
    fn sequential_site_sweep(state: &mut OrthantLogMass) {
        for j in 0..state.m0.len() {
            let (sigma, mu) = state.posterior();
            let s_jj = sigma[[j, j]];
            let update = site_update(1.0 / s_jj - state.tau[j], mu[j] / s_jj - state.nu[j]);
            state.tau[j] = update.tau;
            state.nu[j] = update.nu;
            state.e = inverse_i_plus_wt(&state.w, &state.tau).expect("admissible sites");
        }
        state.site_motion_system = std::sync::OnceLock::new();
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
        sequential_site_sweep(&mut checked);
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

    /// gam#2952: where the criterion prices the kept-spectrum pseudo-inverse `M⁺` of an
    /// indefinite precision, the first derivative of `C` is the derivative of `C` priced with
    /// `M⁺`. `M(t) = R(t) Λ(t) R(t)ᵀ` rotates its eigenvectors, so the kept pair turns into the
    /// dropped negative direction, which `−M⁺ṀM⁺` alone does not see. The kernel is written down
    /// exactly at every `t`, the way the criterion's producers build it from one eigendecomposition.
    #[test]
    fn the_normalizer_derivative_follows_a_kept_spectrum_pseudo_inverse_2952() {
        use crate::estimate::reml::reml_outer_engine::PenaltySubspaceTrace;
        let rows = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8]];
        let bounds = array![0.0, 0.0, -0.05];
        let axis = array![0.3_f64, -0.5, 0.8];
        let axis = &axis / axis.dot(&axis).sqrt();
        let generator = array![
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ];
        let q0 = array![[0.8, -0.6, 0.0], [0.36, 0.48, -0.8], [0.48, 0.64, 0.6]];
        // Eigenvalues 2.0 and 0.9 are kept; −0.3 is dropped, like the #2894 wiggle optimum's
        // −0.12 against a kept 0.45.
        let spectrum = |t: f64| array![2.0 + 0.4 * t, 0.9 - 0.3 * t, -0.3 + 0.1 * t];
        let rotation = |t: f64| {
            let angle = 0.7 * t;
            Array2::<f64>::eye(3)
                + &(&generator * angle.sin())
                + &(generator.dot(&generator) * (1.0 - angle.cos()))
        };
        let kernel_at = |t: f64| {
            let basis = rotation(t).dot(&q0);
            let values = spectrum(t);
            PenaltySubspaceTrace {
                u_s: basis.slice(s![.., 0..2]).to_owned(),
                h_proj_inverse: Array2::from_diag(&values.slice(s![0..2]).mapv(|value| 1.0 / value)),
                dropped_basis: basis.slice(s![.., 2..3]).to_owned(),
                dropped_eigenvalues: values.slice(s![2..3]).to_owned(),
                logdet_correction: 0.0,
            }
        };
        let precision_at = |t: f64| {
            let basis = rotation(t).dot(&q0);
            basis.dot(&Array2::from_diag(&spectrum(t))).dot(&basis.t())
        };
        let (b0, b1) = (array![0.01, 0.02, 0.3], array![-0.2, 0.1, 0.05]);
        let (g0, g1) = (array![0.4, 0.3, 0.0], array![0.1, -0.2, 0.05]);
        let normalizer_at = |t: f64| {
            let kernel = kernel_at(t);
            let solve = |v: &Array1<f64>| kernel.apply_pseudo_inverse(v);
            ConeNormalizer::evaluate(&rows, &bounds, &(&b0 + &(&b1 * t)), &(&g0 + &(&g1 * t)), &solve)
                .expect("normalizer")
        };
        let kernel = kernel_at(0.0);
        let solve = |v: &Array1<f64>| kernel.apply_pseudo_inverse(v);
        let normalizer = normalizer_at(0.0);
        assert!(normalizer.retained_rows() >= 2, "at least two correlated rows are near their bounds");
        // `Ṁ(0) = 0.7·(Ω M₀ − M₀ Ω) + Q₀ Λ̇ Q₀ᵀ`, since `Ṙ(0) = 0.7·Ω` and `Ω` is skew.
        let m0 = precision_at(0.0);
        let precision_rate = (generator.dot(&m0) - m0.dot(&generator)) * 0.7
            + q0.dot(&Array2::from_diag(&array![0.4, -0.3, 0.1])).dot(&q0.t());
        let fd_precision = (precision_at(1.0e-5) - precision_at(-1.0e-5)) / 2.0e-5;
        let rate_gap = (&precision_rate - &fd_precision).iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(rate_gap <= 1.0e-8, "the path's precision rate: gap {rate_gap:e} to its central difference");
        let rotation_on_dropped = precision_rate.dot(&kernel.dropped_basis);
        let turn = kernel
            .pseudo_inverse_rotation(&rotation_on_dropped)
            .expect("spectral kernel");
        let motion_with = |turn_on_gradient: Array1<f64>, turn_on_normals: Array2<f64>| {
            ConeCoordinateMotion {
                mode_response: b1.clone(),
                gradient_rate: g1.clone(),
                precision_rate_on_y: precision_rate.dot(normalizer.solved_gradient()),
                precision_rate_on_r: precision_rate.dot(normalizer.normal_solves()),
                inverse_rotation_on_gradient: turn_on_gradient,
                inverse_rotation_on_normals: turn_on_normals,
            }
        };
        let exact = motion_with(turn.apply(&g0), turn.apply_columns(&normalizer.retained_normals()));
        let derivative = normalizer.first_order(&exact, &solve).derivative;
        let (fd, bar) = richardson(&|t: f64| normalizer_at(t).value(), 1.0e-3);
        assert!(
            (derivative - fd).abs() <= bar,
            "dC {derivative} against central difference {fd} (bar {bar})"
        );
        // The rotation is the derivative of the pseudo-inverse itself: along the same path,
        // `d(M⁺v)/dt = −M⁺ṀM⁺v + rotation(v)` for a fixed `v`.
        let probe = array![0.7, -0.2, 0.4];
        let (fd_solve, _) = {
            let h = 1.0e-5;
            (
                (kernel_at(h).apply_pseudo_inverse(&probe) - kernel_at(-h).apply_pseudo_inverse(&probe))
                    / (2.0 * h),
                h,
            )
        };
        let analytic_solve = -solve(&precision_rate.dot(&solve(&probe))) + turn.apply(&probe);
        let solve_gap = (&analytic_solve - &fd_solve).iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(solve_gap <= 1.0e-7, "d(M⁺v) against central difference: gap {solve_gap:e}");
        // Positive control: without the rotation the derivative is the inverse's, and it misses.
        let inverse_only = motion_with(
            Array1::zeros(3),
            Array2::zeros(normalizer.normal_solves().raw_dim()),
        );
        let missed = normalizer.first_order(&inverse_only, &solve).derivative;
        assert!(
            (missed - fd).abs() > 1.0e3 * bar,
            "positive control: −M⁺ṀM⁺ alone gives {missed} against {fd} (bar {bar})"
        );

        // Second order along the same path. `Ṁ(t) = 0.7·(Ω M − M Ω) + R Ṅ Rᵀ` with
        // `Ṅ = Q₀ Λ̇ Q₀ᵀ`, so `M̈(0) = 0.7·(Ω Ṁ₀ − Ṁ₀ Ω) + 0.7·(Ω Ṅ − Ṅ Ω)`.
        let n_rate = q0.dot(&Array2::from_diag(&array![0.4, -0.3, 0.1])).dot(&q0.t());
        let precision_rate_at = |t: f64| {
            let m = precision_at(t);
            let turn = rotation(t);
            (generator.dot(&m) - m.dot(&generator)) * 0.7 + turn.dot(&n_rate).dot(&turn.t())
        };
        let second_precision = (generator.dot(&precision_rate) - precision_rate.dot(&generator)) * 0.7
            + (generator.dot(&n_rate) - n_rate.dot(&generator)) * 0.7;
        let fd_second = (precision_rate_at(1.0e-5) - precision_rate_at(-1.0e-5)) / 2.0e-5;
        let second_gap =
            (&second_precision - &fd_second).iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(second_gap <= 1.0e-8, "the path's second precision rate: gap {second_gap:e}");
        let exact_motion_at = |t: f64, normalizer: &ConeNormalizer, kernel: &PenaltySubspaceTrace| {
            let rate = precision_rate_at(t);
            let turn = kernel
                .pseudo_inverse_rotation(&rate.dot(&kernel.dropped_basis))
                .expect("spectral kernel");
            ConeCoordinateMotion {
                mode_response: b1.clone(),
                gradient_rate: g1.clone(),
                precision_rate_on_y: rate.dot(normalizer.solved_gradient()),
                precision_rate_on_r: rate.dot(normalizer.normal_solves()),
                inverse_rotation_on_gradient: turn.apply(&(&g0 + &(&g1 * t))),
                inverse_rotation_on_normals: turn.apply_columns(&normalizer.retained_normals()),
            }
        };
        // The second order is graded on one row, where EP is exact (P = Φ), so the check measures
        // the rotation algebra alone. On correlated rows a difference of the first derivative also
        // carries EP's fixed-point tolerance, which the 2765 pin grades on its own fixture.
        let row_one = array![[0.0, 1.0, 0.0]];
        let bound_one = array![0.0];
        let normalizer_one_at = |t: f64| {
            let kernel = kernel_at(t);
            let solve = |v: &Array1<f64>| kernel.apply_pseudo_inverse(v);
            ConeNormalizer::evaluate(&row_one, &bound_one, &(&b0 + &(&b1 * t)), &(&g0 + &(&g1 * t)), &solve)
                .expect("normalizer")
        };
        let derivative_at = |t: f64| {
            let kernel = kernel_at(t);
            let normalizer = normalizer_one_at(t);
            let solve = |v: &Array1<f64>| kernel.apply_pseudo_inverse(v);
            normalizer.first_order(&exact_motion_at(t, &normalizer, &kernel), &solve).derivative
        };
        let normalizer_one = normalizer_one_at(0.0);
        assert_eq!(normalizer_one.retained_rows(), 1, "the one row is inside the mass horizon");
        let normals = normalizer_one.retained_normals();
        let mut probes = Array2::<f64>::zeros((3, 1 + normals.ncols()));
        probes.column_mut(0).assign(&g0);
        probes.slice_mut(s![.., 1..]).assign(&normals);
        let apply_rate = |v: &Array1<f64>| precision_rate.dot(v);
        let second_rotation = |vectors: &Array2<f64>| {
            kernel
                .pseudo_inverse_second_rotation(
                    &apply_rate,
                    &apply_rate,
                    &rotation_on_dropped,
                    &rotation_on_dropped,
                    &second_precision.dot(&kernel.dropped_basis),
                    vectors,
                )
                .expect("spectral kernel")
        };
        // The kernel's second rotation is the second derivative of the pseudo-inverse itself:
        // `d²(M⁺v)/dt² = 2 M⁺ṀM⁺ṀM⁺v − M⁺M̈M⁺v + (second rotation)(v)` for a fixed `v`, against a
        // Richardson-extrapolated second difference.
        let second_difference = |h: f64| {
            (kernel_at(h).apply_pseudo_inverse(&probe) - kernel.apply_pseudo_inverse(&probe) * 2.0
                + kernel_at(-h).apply_pseudo_inverse(&probe))
                / (h * h)
        };
        let second_solve = (second_difference(1.0e-3) * 4.0 - second_difference(2.0e-3)) / 3.0;
        let mut probe_column = Array2::<f64>::zeros((3, 1));
        probe_column.column_mut(0).assign(&probe);
        let inverse_part = solve(&precision_rate.dot(&solve(&precision_rate.dot(&solve(&probe))))) * 2.0
            - solve(&second_precision.dot(&solve(&probe)));
        let analytic_second = &inverse_part + &second_rotation(&probe_column).column(0);
        let second_solve_gap =
            (&analytic_second - &second_solve).iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            second_solve_gap <= 1.0e-8,
            "d²(M⁺v) against a second difference: gap {second_solve_gap:e}"
        );
        let second_turn = second_rotation(&probes);
        let pair_with = |turn_on_gradient: Array1<f64>, turn_on_normals: Array2<f64>| ConePairMotion {
            mode_response: Array1::zeros(3),
            gradient_rate: Array1::zeros(3),
            precision_rate_on_y: second_precision.dot(normalizer_one.solved_gradient()),
            precision_rate_on_r: second_precision.dot(normalizer_one.normal_solves()),
            inverse_rotation_on_gradient: turn_on_gradient,
            inverse_rotation_on_normals: turn_on_normals,
        };
        let exact_one = exact_motion_at(0.0, &normalizer_one, &kernel);
        let first = normalizer_one.first_order(&exact_one, &solve);
        let pair = pair_with(second_turn.column(0).to_owned(), second_turn.slice(s![.., 1..]).to_owned());
        let second = normalizer_one
            .second_order(&exact_one, &first, &exact_one, &first, &pair)
            .expect("second order");
        let (fd2, bar2) = richardson(&derivative_at, 1.0e-3);
        assert!(
            (second - fd2).abs() <= bar2,
            "d²C {second} against central difference of dC {fd2} (bar {bar2})"
        );
        // Positive control: the inverse identities alone miss the second derivative.
        let bare_pair = pair_with(Array1::zeros(3), Array2::zeros(normals.raw_dim()));
        let bare_first_motion = ConeCoordinateMotion {
            inverse_rotation_on_gradient: Array1::zeros(3),
            inverse_rotation_on_normals: Array2::zeros(normals.raw_dim()),
            ..exact_one.clone()
        };
        let bare_first = normalizer_one.first_order(&bare_first_motion, &solve);
        let bare = normalizer_one
            .second_order(&bare_first_motion, &bare_first, &bare_first_motion, &bare_first, &bare_pair)
            .expect("second order");
        assert!(
            (bare - fd2).abs() > 1.0e3 * bar2,
            "positive control: the inverse identities give {bare} against {fd2} (bar {bar2})"
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
                // `dense_solve` is an inverse: nothing rotates.
                inverse_rotation_on_gradient: Array1::zeros(b1.len()),
                inverse_rotation_on_normals: Array2::zeros(normalizer.normal_solves().raw_dim()),
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
            inverse_rotation_on_gradient: Array1::zeros(b2.len()),
            inverse_rotation_on_normals: Array2::zeros(normalizer.normal_solves().raw_dim()),
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

    /// gam#3135 (the property gam-2959 proposed for #2959 M7): the rank-one sweep stops at a fixed
    /// point of the definition's sweep. On a correlated three-row orthant, a twelve-row banded one
    /// and a twelve-row one near rank one, a further sweep that re-forms the posterior after every
    /// site moves `ln P` by no more than the rounding band EP stops at.
    #[test]
    fn the_rank_one_sweep_stops_at_a_fixed_point_of_the_sequential_site_sweep_3135() {
        let correlated = (
            array![-1.5, 0.2, 1.0],
            array![[1.0, 0.5, -0.3], [0.5, 2.0, 0.4], [-0.3, 0.4, 0.8]],
        );
        let rows = 12usize;
        let banded = (
            Array1::from_shape_fn(rows, |i| 1.5 * (1.3 * i as f64).sin()),
            Array2::from_shape_fn((rows, rows), |(i, j)| {
                let scale = (0.5 + 0.1 * i as f64) * (0.5 + 0.1 * j as f64);
                scale * 0.6_f64.powi((i as i32 - j as i32).abs())
            }),
        );
        let near_rank_one = (
            Array1::from_shape_fn(rows, |i| 0.3 * (0.7 * i as f64).cos() - 0.2),
            Array2::from_shape_fn((rows, rows), |(i, j)| {
                let (a, b) = (1.0 + 0.05 * i as f64, 1.0 + 0.05 * j as f64);
                0.97 * a * b + if i == j { 0.03 * a * a } else { 0.0 }
            }),
        );
        for (m0, w) in [correlated, banded, near_rank_one] {
            let q = m0.len();
            let mass = OrthantLogMass::converge(&m0, &w).expect("the orthant converges");
            let (_, magnitude) = mass.evaluate_log_mass().expect("a finite EP log mass");
            let mut checked = mass.clone();
            sequential_site_sweep(&mut checked);
            let (after, _) = checked.evaluate_log_mass().expect("a finite EP log mass");
            let band = accumulation_growth(4 * q * q + 8 * q) * magnitude;
            eprintln!(
                "[3135-EP] q = {q}: ln P {:.15e} after {} sweeps; a sequential sweep moves it {:e} (band {band:e})",
                mass.log_mass(),
                mass.sweeps(),
                (after - mass.log_mass()).abs()
            );
            assert!(
                (after - mass.log_mass()).abs() <= band,
                "q = {q}: a sequential sweep from the rank-one stop moves ln P {:.15e} by {:e}, \
                 above the band {band:e}",
                mass.log_mass(),
                (after - mass.log_mass()).abs()
            );
        }
    }


    /// An exactly repeated row is one half-space: `C` is the value of the rows without the repeat,
    /// bit for bit. Positive control: EP handed the copies counts each one's truncation.
    #[test]
    fn a_repeated_row_enters_the_normalizer_once_1082() {
        let h = array![[2.0, 0.8, 0.3], [0.8, 1.5, -0.4], [0.3, -0.4, 1.2]];
        let solve = dense_solve(&h);
        let beta = array![0.0, 0.2, 0.5];
        let gradient = array![0.3, 0.0, 0.0];
        let distinct = array![[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]];
        let distinct_bounds = array![0.0, 0.1];
        let copies = 38;
        let repeated = Array2::from_shape_fn((2 * copies, 3), |(i, j)| distinct[[i % 2, j]]);
        let repeated_bounds = Array1::from_shape_fn(2 * copies, |i| distinct_bounds[i % 2]);
        let once = ConeNormalizer::evaluate(&distinct, &distinct_bounds, &beta, &gradient, &solve)
            .expect("normalizer on the distinct rows");
        let with_copies = ConeNormalizer::evaluate(&repeated, &repeated_bounds, &beta, &gradient, &solve)
            .expect("normalizer on the repeated rows");
        assert_eq!(with_copies.retained_rows(), once.retained_rows(), "each half-space enters once");
        assert_eq!(
            with_copies.value().to_bits(),
            once.value().to_bits(),
            "C with {copies} copies of each row {} against C without them {}",
            with_copies.value(),
            once.value()
        );
        // The active row alone: one half-space, where EP is exact, against 38 sites on it.
        let v = solve(&array![1.0, 0.0, 0.0])[0];
        let m = -0.3 * v;
        let exact = normal_logcdf(m / v.sqrt());
        let single = OrthantLogMass::converge(&array![m], &array![[v]]).expect("one site");
        let copied = OrthantLogMass::converge(
            &Array1::from_elem(copies, m),
            &Array2::from_elem((copies, copies), v),
        )
        .expect("the copied sites converge");
        eprintln!(
            "[1082-EP] one row at z = {:.6}: ln Φ {exact:.15e}; EP on one site {:.15e}; EP on {copies} copies \
             {:.15e} in {} sweeps",
            m / v.sqrt(),
            single.log_mass(),
            copied.log_mass(),
            copied.sweeps()
        );
        assert!(
            (single.log_mass() - exact).abs() <= band(64, exact.abs().max(1.0)),
            "one site is exact: {} against ln Φ {exact}",
            single.log_mass()
        );
        assert!(
            (copied.log_mass() - exact).abs() > 1.0e3 * band(64, exact.abs().max(1.0)),
            "positive control: EP on {copies} copies of one row gives {} against ln Φ {exact}",
            copied.log_mass()
        );
    }
}
