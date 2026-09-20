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
//! # The orthant in its smaller coordinates
//!
//! `u = m₀ + Az` with `z ~ N(0, M⁻¹)`, and only the columns `A` touches enter. So the orthant
//! is carried as `u = m₀ + Bz`, `z ~ N(0, K)`, in whichever coordinates are fewer: the rows
//! themselves (`B = I_q`, `K = W`) when there are no more rows than supported columns, and the
//! `r` supported columns (`B = A` on them, `K = M⁻¹` on them) otherwise. A monotonicity guard
//! with one row per observation has thousands of rows on a few spline columns (gam#3037,
//! gam#3038); every operation below is `O(q r²)` or `O(r³)` and never `O(q³)`. The choice is by
//! dimension alone; both carry the same law, so `ln P` and its derivatives do not depend on it.
//!
//! # `ln P` by expectation propagation
//!
//! `P(u ≥ 0)` has no closed form beyond two rows. It is estimated by EP on the orthant's
//! half-line indicators (Cunningham, Hennig and Lacoste-Julien, 2011): deterministic, smooth in
//! `(m₀, K)`, and exact for one row and for rows whose normals are `M⁻¹`-orthogonal. A sweep
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
//! Gaussian part. With `T = diag(τ̃)`, `G = BᵀTB`, the posterior `Σ_z = (K⁻¹ + G)⁻¹ = F⁻¹K`
//! (`F = I + KG`, so `K⁻¹` is never formed), `μ_u = m₀ + BΣ_zBᵀ(ν̃ − Tm₀)` and `ḡ = Bᵀγ`,
//!
//! ```text
//! γ := ∂lnP/∂m₀ = ν̃ − T μ_u,    Γ := ∂lnP/∂K = ½ (ḡḡᵀ − G + GΣ_zG).
//! ```
//!
//! The Hessian is not stationary in the sites: along a direction they move by the linearized
//! fixed point `(I − ∂F/∂s) ds = ∂F/∂θ dθ`, a `2q × 2q` system whose coupling has rank at most
//! `r(r+1)/2 + r` (a site moves the posterior only through `(Σ_z, μ_z)`), and `(γ, Γ)` are
//! differentiated through it. The criterion's derivatives chain through `y = M⁻¹g` and the
//! basis `N` whose Gram under `M` is `K` (`R = M⁻¹Aᵀ` in row coordinates, `M⁻¹` on the
//! supported columns otherwise), so the precision's motion is read only on `span{y, N}`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef};
use gam_linalg::faer_ndarray::{FaerLu, matmul_parallelism};
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

/// The LU factor of `a`, refused when a column has no pivot.
fn lu_factor(a: MatRef<'_, f64>, what: &str) -> Result<FaerLu, ConeNormalizerRefusal> {
    FaerLu::new(a).map_err(|col| ConeNormalizerRefusal::Singular {
        reason: format!("{what} has no pivot in column {col} of {}", a.nrows()),
    })
}

/// `A⁻¹ rhs` from the LU factor of `A`.
fn lu_solve(lu: &FaerLu, rhs: &Array1<f64>) -> Array1<f64> {
    let solved = lu.solve(Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]).as_ref());
    Array1::from_shape_fn(rhs.len(), |i| solved[(i, 0)])
}

/// `ln|det A|` and the summed magnitude of its pivot logarithms, by LU with partial pivoting.
/// Refused unless `det A > 0`: `F = I + KG` is similar to `I + K^½GK^½` for positive
/// semidefinite `K` and `G`, so its determinant is at least one.
fn positive_log_det(mut a: Array2<f64>, what: &str) -> Result<(f64, f64), ConeNormalizerRefusal> {
    let n = a.nrows();
    let (mut log_det, mut magnitude, mut sign) = (0.0, 0.0, 1.0_f64);
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
            }
            sign = -sign;
        }
        sign *= pivot.signum();
        let log_pivot = pivot.abs().ln();
        log_det += log_pivot;
        magnitude += log_pivot.abs();
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for k in (col + 1)..n {
                a[[row, k]] -= factor * a[[col, k]];
            }
        }
    }
    if !(sign > 0.0) {
        return Err(ConeNormalizerRefusal::Singular {
            reason: format!("{what} has a negative determinant"),
        });
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

/// EP's Gaussian approximation in the orthant's coordinates `z`, formed from the sites.
#[derive(Clone, Debug)]
struct Posterior {
    /// `G = BᵀTB`.
    g: Array2<f64>,
    /// `F = I + KG`.
    f: Array2<f64>,
    /// `F⁻¹`.
    f_inv: Array2<f64>,
    /// `Σ_z = F⁻¹K`.
    sigma: Array2<f64>,
    /// `μ_z = Σ_z Bᵀ(ν̃ − T m₀)`.
    mu: Array1<f64>,
    /// `ḡ = Bᵀγ = F⁻ᵀBᵀ(ν̃ − T m₀)`, read without forming `γ = ν̃ − Tμ_u`: that difference
    /// cancels once a site's precision dwarfs its row's prior variance, `ḡ` does not.
    gbar: Array1<f64>,
}

/// The per-row marginals of the posterior: the rows `w_j = Σ_z b_j`, the variances
/// `s_j = b_jᵀΣ_z b_j` and the means `μ_j = m₀_j + b_jᵀμ_z`.
struct Marginals {
    loaded: Array2<f64>,
    variance: Array1<f64>,
    mean: Array1<f64>,
}

/// The linearized EP fixed point `(I − ∂F/∂s) ds = rhs`. With the site Jacobians `D`, the
/// cavity map `C` from a row's marginal `(s_j, μ_j)` and the sites' effect `Ψ` on the marginals,
/// `I − ∂F/∂s = (I + D) − DCΨ`. `I + D` is block diagonal, one `2 × 2` block per site: the
/// Jacobian of the cavity's map to the moment-matched tilted law, a diffeomorphism for the
/// log-concave half-line site. `Ψ = Rd·V` factors through the posterior's own coordinates
/// `(Σ_z, μ_z)`, of dimension `m = r(r+1)/2 + r`, so the system is solved in whichever of `2q`
/// and `m` is smaller.
#[derive(Clone, Debug)]
enum SiteSystem {
    /// The LU factor of `I − ∂F/∂s`, `2q × 2q`.
    Direct(FaerLu),
    /// Woodbury through the posterior coordinates.
    Capacitance {
        /// `(I + D_j)⁻¹`, row-major `2 × 2`.
        site_inverse: Vec<[f64; 4]>,
        /// `P_j = (I + D_j)⁻¹ D_j C_j`, row-major `2 × 2`.
        coupling: Vec<[f64; 4]>,
        /// The LU factor of `I_m − V (I + D)⁻¹ DC Rd`.
        capacitance: FaerLu,
        marginals: Marginals,
    },
}

impl Clone for Marginals {
    fn clone(&self) -> Self {
        Self { loaded: self.loaded.clone(), variance: self.variance.clone(), mean: self.mean.clone() }
    }
}

impl std::fmt::Debug for Marginals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Marginals").field("rows", &self.variance.len()).finish()
    }
}

/// The index pairs `a ≤ b` of an `r × r` symmetric matrix's upper triangle.
fn upper_pairs(r: usize) -> Vec<(usize, usize)> {
    (0..r).flat_map(|a| (a..r).map(move |b| (a, b))).collect()
}

/// `ln P(u ≥ 0)` for `u = m₀ + Bz`, `z ~ N(0, K)`, at a converged EP fixed point, with its first
/// two derivatives in `(m₀, K)`.
#[derive(Clone, Debug)]
pub struct OrthantLogMass {
    m0: Array1<f64>,
    /// `B`, `q × r`.
    loadings: Array2<f64>,
    /// `K`, `r × r`.
    k: Array2<f64>,
    tau: Array1<f64>,
    nu: Array1<f64>,
    post: Posterior,
    /// `B = I`: the orthant is carried in its rows' own coordinates, where `γ = ḡ`.
    row_coordinates: bool,
    log_mass: f64,
    sweeps: usize,
    /// The share of its full update each site took on the last sweep: 1 unless a sweep failed to
    /// contract.
    fraction: f64,
    /// The linearized fixed point, formed on the first derivative request.
    site_system: std::sync::OnceLock<SiteSystem>,
}

impl OrthantLogMass {
    /// Converge EP on `P(u ≥ 0)`, `u ~ N(m₀, W)`, in the rows' own coordinates.
    pub fn converge(m0: &Array1<f64>, w: &Array2<f64>) -> Result<Self, ConeNormalizerRefusal> {
        Self::converge_in(m0, &Array2::eye(m0.len()), w, true)
    }

    /// Converge EP on `P(u ≥ 0)`, `u = m₀ + Bz`, `z ~ N(0, K)`: the orthant of `N(m₀, BKBᵀ)`
    /// carried in the `r` coordinates of `z`.
    pub fn converge_loaded(
        m0: &Array1<f64>,
        loadings: &Array2<f64>,
        k: &Array2<f64>,
    ) -> Result<Self, ConeNormalizerRefusal> {
        Self::converge_in(m0, loadings, k, false)
    }

    fn converge_in(
        m0: &Array1<f64>,
        loadings: &Array2<f64>,
        k: &Array2<f64>,
        row_coordinates: bool,
    ) -> Result<Self, ConeNormalizerRefusal> {
        let (q, r) = loadings.dim();
        assert_eq!(m0.len(), q, "one orthant mean per loading row");
        assert_eq!(k.dim(), (r, r), "a covariance on the loadings' coordinates");
        if m0.iter().chain(loadings.iter()).chain(k.iter()).any(|value| !value.is_finite()) {
            return Err(ConeNormalizerRefusal::NonFinite { what: "orthant mean or covariance" });
        }
        let mut state = Self {
            m0: m0.clone(),
            loadings: loadings.clone(),
            k: k.clone(),
            tau: Array1::zeros(q),
            nu: Array1::zeros(q),
            post: Posterior {
                g: Array2::zeros((r, r)),
                f: Array2::eye(r),
                f_inv: Array2::eye(r),
                sigma: k.clone(),
                mu: Array1::zeros(r),
                gbar: Array1::zeros(r),
            },
            row_coordinates,
            log_mass: 0.0,
            sweeps: 0,
            fraction: 1.0,
            site_system: std::sync::OnceLock::new(),
        };
        if q == 0 {
            return Ok(state);
        }
        state.post = state.form_posterior()?;
        let (mut previous, _) = state.evaluate_log_mass()?;
        let mut previous_change = f64::INFINITY;
        let mut fraction = 1.0_f64;
        loop {
            state.sweeps += 1;
            let (mut at_fixed_point, mut moved) = (true, false);
            // Sequential EP: each site reads the posterior its predecessors in the sweep left.
            // Moving site `j` by `(Δτ, Δν)` adds `Δτ b_jb_jᵀ` to the precision of `z`, so the
            // posterior follows by Sherman–Morrison in O(r²),
            //   `Σ ← Σ − c w wᵀ`, `μ ← μ + (Δν(1 − c s_j) − c μ_j) w`,
            //   `w = Σ b_j`, `s_j = b_jᵀw`, `c = Δτ/(1 + Δτ s_j)`,
            // where `1 + Δτ s_j = s_j(τ_c + τ̃_j^new) > 0` for an admissible cavity. The posterior
            // is re-formed from the sites once per sweep, which also resets the rank-one updates'
            // accumulated rounding before `ln Z_EP` is read.
            let mut sigma = state.post.sigma.clone();
            let mut mu = state.post.mu.clone();
            for j in 0..q {
                let b = state.loadings.row(j);
                let w = sigma.dot(&b);
                let s_jj = b.dot(&w);
                let mu_j = state.m0[j] + b.dot(&mu);
                let tau_c = 1.0 / s_jj - state.tau[j];
                let nu_c = mu_j / s_jj - state.nu[j];
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
                let mean_step = delta_nu * (1.0 - c * s_jj) - c * mu_j;
                mu.scaled_add(mean_step, &w);
                for a in 0..r {
                    let scaled = c * w[a];
                    for b in 0..r {
                        sigma[[a, b]] -= scaled * w[b];
                    }
                }
            }
            state.post = state.form_posterior()?;
            let (log_mass, magnitude) = state.evaluate_log_mass()?;
            let change = (log_mass - previous).abs();
            // Each row's cavity is read through O(r²) rounded operations and the determinant and
            // quadratic through O(r²) per entry, summed over the q sites. A damped sweep moves
            // ln Z_EP by about `fraction` times what the full update would, so the band scales with
            // it: a short step is not mistaken for a settled one.
            let band = accumulation_growth(4 * r * r + 8 * q) * magnitude;
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

    /// The posterior of `z` at the current sites: `Σ_z = (I + KG)⁻¹K`, `μ_z = Σ_z Bᵀ(ν̃ − Tm₀)`.
    fn form_posterior(&self) -> Result<Posterior, ConeNormalizerRefusal> {
        let (q, r) = self.loadings.dim();
        let weighted = Array2::from_shape_fn((q, r), |(j, a)| self.tau[j] * self.loadings[[j, a]]);
        let g = symmetrized(&self.loadings.t().dot(&weighted));
        let f = Array2::<f64>::eye(r) + &self.k.dot(&g);
        let f_inv = invert(f.clone(), "I + KG")?;
        let sigma = symmetrized(&f_inv.dot(&self.k));
        let h = self.loadings.t().dot(&(&self.nu - &(&self.tau * &self.m0)));
        let mu = sigma.dot(&h);
        let gbar = f_inv.t().dot(&h);
        Ok(Posterior { g, f, f_inv, sigma, mu, gbar })
    }

    /// Every row's posterior marginal.
    fn marginals(&self) -> Marginals {
        let loaded = self.loadings.dot(&self.post.sigma);
        let variance = Array1::from_shape_fn(self.m0.len(), |j| {
            loaded.row(j).dot(&self.loadings.row(j))
        });
        let mean = &self.m0 + &self.loadings.dot(&self.post.mu);
        Marginals { loaded, variance, mean }
    }

    /// `ln Z_EP` and the summed magnitude of its terms,
    ///
    /// `Σ_j [ln Φ(z_j) + ½ln(1 + τ̃_j v_j) + ½(τ̃_j m_j − ν̃_j)²/(τ̃_j(1 + τ̃_j v_j))]
    ///  − ½ ln|I + KG| − ½ Q`,
    ///
    /// with `(m_j, v_j)` the cavity of site `j` and `Q = xᵀ(T + TWT)⁺x`, `x = Tm₀ − ν̃`, read as
    /// `Σ_{τ̃_j > 0} γ_j²/τ̃_j + μ_zᵀḡ`: `μ_zᵀḡ = μ_zᵀK⁻¹μ_z ≥ 0`, so neither part cancels the
    /// other. Neither `K⁻¹`, `W⁻¹` nor `τ̃^{−½}` is formed.
    fn evaluate_log_mass(&self) -> Result<(f64, f64), ConeNormalizerRefusal> {
        let q = self.m0.len();
        let marginals = self.marginals();
        let (mut total, mut magnitude) = (0.0, 0.0);
        for j in 0..q {
            let s_jj = marginals.variance[j];
            let tau_c = 1.0 / s_jj - self.tau[j];
            let nu_c = marginals.mean[j] / s_jj - self.nu[j];
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
        let (log_det, log_det_magnitude) = positive_log_det(self.post.f.clone(), "I + KG")?;
        let gamma = self.mean_gradient();
        // A site with τ̃ = 0 has never moved, so ν̃ = 0 and γ_j = 0 there.
        let spread: f64 =
            (0..q).filter(|&j| self.tau[j] > 0.0).map(|j| gamma[j] * gamma[j] / self.tau[j]).sum();
        let coupling = self.post.mu.dot(&self.post.gbar);
        total += -0.5 * log_det - 0.5 * (spread + coupling);
        magnitude += 0.5 * log_det_magnitude + 0.5 * (spread + coupling.abs());
        if !total.is_finite() {
            return Err(ConeNormalizerRefusal::NonFinite { what: "EP log mass" });
        }
        Ok((total, magnitude))
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

    /// `r`, the dimension EP's posterior is carried in.
    pub fn dimension(&self) -> usize {
        self.k.nrows()
    }

    /// `γ = ∂lnP/∂m₀ = ν̃ − T μ_u`, which is `ḡ` itself in row coordinates.
    pub fn mean_gradient(&self) -> Array1<f64> {
        if self.row_coordinates {
            return self.post.gbar.clone();
        }
        let mean = &self.m0 + &self.loadings.dot(&self.post.mu);
        &self.nu - &(&self.tau * &mean)
    }

    /// `Γ = ∂lnP/∂K = ½(ḡḡᵀ − G + GΣ_zG)`, `ḡ = Bᵀγ`, so `d ln P = γᵀdm₀ + tr(Γ dK)`.
    pub fn covariance_gradient(&self) -> Array2<f64> {
        let gbar = &self.post.gbar;
        let g = &self.post.g;
        let gsg = g.dot(&self.post.sigma).dot(g);
        symmetrized(&Array2::from_shape_fn(g.raw_dim(), |(a, b)| {
            0.5 * (gbar[a] * gbar[b] - g[[a, b]] + gsg[[a, b]])
        }))
    }

    /// `(dγ, dΓ)` along `(dm₀, dK)`, the sites moving with their linearized fixed point.
    pub fn gradient_motion(
        &self,
        dm0: &Array1<f64>,
        dk: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), ConeNormalizerRefusal> {
        let (q, r) = self.loadings.dim();
        if q == 0 {
            return Ok((Array1::zeros(0), Array2::zeros((r, r))));
        }
        let (dtau, dnu) = self.site_motion(dm0, dk)?;
        let b = &self.loadings;
        let post = &self.post;
        let mean = &self.m0 + &b.dot(&post.mu);
        let gbar = &post.gbar;
        // dG = Bᵀ dT B; dΣ_z = F⁻¹dK F⁻ᵀ − Σ_z dG Σ_z; dμ_z = F⁻¹dK ḡ + Σ_zBᵀ(dν̃ − dT μ_u − T dm₀).
        let weighted = Array2::from_shape_fn((q, r), |(j, a)| dtau[j] * b[[j, a]]);
        let d_g = symmetrized(&b.t().dot(&weighted));
        let d_sigma = post.f_inv.dot(dk).dot(&post.f_inv.t()) - post.sigma.dot(&d_g).dot(&post.sigma);
        let shift = &dnu - &(&dtau * &mean) - &(&self.tau * dm0);
        let d_mu_z = post.f_inv.dot(&dk.dot(gbar)) + post.sigma.dot(&b.t().dot(&shift));
        let d_mean = dm0 + &b.dot(&d_mu_z);
        let d_gamma = &dnu - &(&dtau * &mean) - &(&self.tau * &d_mean);
        let d_gbar = b.t().dot(&d_gamma);
        let g = &post.g;
        let curvature = d_g.dot(&post.sigma).dot(g) + g.dot(&d_sigma).dot(g) + g.dot(&post.sigma).dot(&d_g);
        let d_big_gamma = symmetrized(&Array2::from_shape_fn((r, r), |(a, c)| {
            0.5 * (d_gbar[a] * gbar[c] + gbar[a] * d_gbar[c] - d_g[[a, c]] + curvature[[a, c]])
        }));
        Ok((d_gamma, d_big_gamma))
    }

    /// The sites' motion `(dτ̃, dν̃)` along `(dm₀, dK)`, from `(I − ∂F/∂s) ds = ∂F/∂θ dθ`.
    ///
    /// At fixed sites a row's marginal moves by `ds_j = b_jᵀF⁻¹dKF⁻ᵀb_j` and
    /// `dμ_j = dm₀_j + b_jᵀ(F⁻¹dK ḡ − Σ_zBᵀT dm₀)`; site `j` reads the cavity
    /// `(1/s_j − τ̃_j, μ_j/s_j − ν̃_j)`.
    fn site_motion(
        &self,
        dm0: &Array1<f64>,
        dk: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ConeNormalizerRefusal> {
        let q = self.m0.len();
        let b = &self.loadings;
        let post = &self.post;
        let marginals = self.marginals();
        let gbar = &post.gbar;
        let spread_rows = b.dot(&post.f_inv.dot(dk).dot(&post.f_inv.t()));
        let d_mu_z = post.f_inv.dot(&dk.dot(gbar)) - post.sigma.dot(&b.t().dot(&(&self.tau * dm0)));
        let d_mean = dm0 + &b.dot(&d_mu_z);
        let mut rhs = Array1::<f64>::zeros(2 * q);
        for j in 0..q {
            let s_jj = marginals.variance[j];
            let mu_j = marginals.mean[j];
            let tau_c = 1.0 / s_jj - self.tau[j];
            let nu_c = mu_j / s_jj - self.nu[j];
            let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = site_update(tau_c, nu_c).jacobian;
            let s2 = s_jj * s_jj;
            let d_sjj = spread_rows.row(j).dot(&b.row(j));
            let (dtc, dnc) = (-d_sjj / s2, d_mean[j] / s_jj - mu_j * d_sjj / s2);
            rhs[j] = dt_dtc * dtc + dt_dnc * dnc;
            rhs[q + j] = dn_dtc * dtc + dn_dnc * dnc;
        }
        let ds = self.solve_site_system(&rhs)?;
        Ok((ds.slice(s![0..q]).to_owned(), ds.slice(s![q..2 * q]).to_owned()))
    }

    /// Solve the linearized fixed point `(I − ∂F/∂s) ds = rhs`, `rhs = [τ̃ block; ν̃ block]`.
    fn solve_site_system(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, ConeNormalizerRefusal> {
        let q = self.m0.len();
        match self.site_system()? {
            SiteSystem::Direct(system) => Ok(lu_solve(system, rhs)),
            SiteSystem::Capacitance { site_inverse, coupling, capacitance, marginals } => {
                let b = &self.loadings;
                let r = b.ncols();
                let quad = r * (r + 1) / 2;
                // x₀ = (I + D)⁻¹ rhs.
                let mut x_tau = Array1::<f64>::zeros(q);
                let mut x_nu = Array1::<f64>::zeros(q);
                for j in 0..q {
                    let a = site_inverse[j];
                    x_tau[j] = a[0] * rhs[j] + a[1] * rhs[q + j];
                    x_nu[j] = a[2] * rhs[j] + a[3] * rhs[q + j];
                }
                // V x₀ = [upper(Bᵀ diag(x_τ) B); Bᵀ(x_ν − μ∘x_τ)].
                let weighted = Array2::from_shape_fn((q, r), |(j, a)| x_tau[j] * b[[j, a]]);
                let gram = b.t().dot(&weighted);
                let mut moved = Array1::<f64>::zeros(quad + r);
                for (index, &(a, c)) in upper_pairs(r).iter().enumerate() {
                    moved[index] = gram[[a, c]];
                }
                let linear = b.t().dot(&(&x_nu - &(&marginals.mean * &x_tau)));
                moved.slice_mut(s![quad..]).assign(&linear);
                let solved = lu_solve(capacitance, &moved);
                // ds = x₀ + P·Rd·solved, with Rd reading (−wᵀZw, wᵀz) off each row w_j.
                let mut z_matrix = Array2::<f64>::zeros((r, r));
                for (index, &(a, c)) in upper_pairs(r).iter().enumerate() {
                    z_matrix[[a, c]] = solved[index];
                    z_matrix[[c, a]] = solved[index];
                }
                let z_linear = solved.slice(s![quad..]).to_owned();
                let quadratic_rows = marginals.loaded.dot(&z_matrix);
                let mut ds = Array1::<f64>::zeros(2 * q);
                for j in 0..q {
                    let w = marginals.loaded.row(j);
                    let read_s = -quadratic_rows.row(j).dot(&w);
                    let read_mu = w.dot(&z_linear);
                    let p = coupling[j];
                    ds[j] = x_tau[j] + p[0] * read_s + p[1] * read_mu;
                    ds[q + j] = x_nu[j] + p[2] * read_s + p[3] * read_mu;
                }
                Ok(ds)
            }
        }
    }

    /// The linearized fixed point at the converged sites. It depends on the fixed point alone, not
    /// on the direction `(dm₀, dK)`, so it is formed once and shared by every coordinate and pair.
    fn site_system(&self) -> Result<&SiteSystem, ConeNormalizerRefusal> {
        if let Some(system) = self.site_system.get() {
            return Ok(system);
        }
        let (q, r) = self.loadings.dim();
        let b = &self.loadings;
        let marginals = self.marginals();
        let quad = r * (r + 1) / 2;
        let m = quad + r;
        let mut jacobians = Vec::with_capacity(q);
        for j in 0..q {
            let s_jj = marginals.variance[j];
            let tau_c = 1.0 / s_jj - self.tau[j];
            let nu_c = marginals.mean[j] / s_jj - self.nu[j];
            jacobians.push(site_update(tau_c, nu_c).jacobian);
        }
        let system = if m < 2 * q {
            let mut site_inverse = Vec::with_capacity(q);
            let mut coupling = Vec::with_capacity(q);
            for j in 0..q {
                let [d00, d01, d10, d11] = jacobians[j];
                let (a00, a01, a10, a11) = (1.0 + d00, d01, d10, 1.0 + d11);
                let det = a00 * a11 - a01 * a10;
                if !(det != 0.0 && det.is_finite()) {
                    return Err(ConeNormalizerRefusal::Singular {
                        reason: format!("the EP site map at row {j} has determinant {det:e}"),
                    });
                }
                let inverse = [a11 / det, -a01 / det, -a10 / det, a00 / det];
                // C_j = [[−1/s², 0], [−μ/s², 1/s]] maps (ds_j, dμ_j) to the cavity's motion.
                let s_jj = marginals.variance[j];
                let s2 = s_jj * s_jj;
                let (c00, c01, c10, c11) = (-1.0 / s2, 0.0, -marginals.mean[j] / s2, 1.0 / s_jj);
                let dc = [
                    d00 * c00 + d01 * c10,
                    d00 * c01 + d01 * c11,
                    d10 * c00 + d11 * c10,
                    d10 * c01 + d11 * c11,
                ];
                coupling.push([
                    inverse[0] * dc[0] + inverse[1] * dc[2],
                    inverse[0] * dc[1] + inverse[1] * dc[3],
                    inverse[2] * dc[0] + inverse[3] * dc[2],
                    inverse[2] * dc[1] + inverse[3] * dc[3],
                ]);
                site_inverse.push(inverse);
            }
            // Capacitance I_m − Σ_j V_j P_j Rd_j, where site j's columns of V are
            // τ̃_j → [upper(b_jb_jᵀ); −μ_j b_j] and ν̃_j → [0; b_j], and its rows of Rd are
            // s_j → [−(2 − δ_ac) w_a w_c; 0] and μ_j → [0; w_j]. It is accumulated over blocks
            // of m sites, so no intermediate is larger than m × 2m.
            let pairs = upper_pairs(r);
            let mut capacitance = Mat::<f64>::identity(m, m);
            let block = m.max(1);
            let mut start = 0;
            while start < q {
                let end = (start + block).min(q);
                let n = end - start;
                let mut v = Mat::<f64>::zeros(m, 2 * n);
                let mut read = Mat::<f64>::zeros(2 * n, m);
                for (i, j) in (start..end).enumerate() {
                    let row = b.row(j);
                    let w = marginals.loaded.row(j);
                    let p = coupling[j];
                    for (index, &(a, c)) in pairs.iter().enumerate() {
                        v[(index, 2 * i)] = row[a] * row[c];
                        let weight = if a == c { 1.0 } else { 2.0 };
                        let spread = weight * w[a] * w[c];
                        read[(2 * i, index)] = -p[0] * spread;
                        read[(2 * i + 1, index)] = -p[2] * spread;
                    }
                    for a in 0..r {
                        v[(quad + a, 2 * i)] = -marginals.mean[j] * row[a];
                        v[(quad + a, 2 * i + 1)] = row[a];
                        read[(2 * i, quad + a)] = p[1] * w[a];
                        read[(2 * i + 1, quad + a)] = p[3] * w[a];
                    }
                }
                matmul(
                    capacitance.as_mut(),
                    Accum::Add,
                    v.as_ref(),
                    read.as_ref(),
                    -1.0,
                    matmul_parallelism(m, m, 2 * n),
                );
                start = end;
            }
            let capacitance = lu_factor(capacitance.as_ref(), "the EP site system's capacitance")?;
            SiteSystem::Capacitance { site_inverse, coupling, capacitance, marginals }
        } else {
            // Per unit site change a row's marginal moves by `ds_j/dτ̃_k = −Σ_jk²`,
            // `dμ_j/dτ̃_k = −Σ_jk μ_k` and `dμ_j/dν̃_k = Σ_jk`, `Σ_u = BΣ_zBᵀ`.
            let sigma_u = marginals.loaded.dot(&b.t());
            let mu = &marginals.mean;
            let mut system = Mat::<f64>::identity(2 * q, 2 * q);
            for j in 0..q {
                let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = jacobians[j];
                let s_jj = marginals.variance[j];
                let s2 = s_jj * s_jj;
                let cavity_rate = |d_sjj: f64, d_muj: f64| -> (f64, f64) {
                    (-d_sjj / s2, d_muj / s_jj - mu[j] * d_sjj / s2)
                };
                for k in 0..q {
                    let (dtc, dnc) =
                        cavity_rate(-sigma_u[[j, k]] * sigma_u[[j, k]], -sigma_u[[j, k]] * mu[k]);
                    let dtc = if k == j { dtc - 1.0 } else { dtc };
                    system[(j, k)] -= dt_dtc * dtc + dt_dnc * dnc;
                    system[(q + j, k)] -= dn_dtc * dtc + dn_dnc * dnc;
                    let (dtc, dnc) = cavity_rate(0.0, sigma_u[[j, k]]);
                    let dnc = if k == j { dnc - 1.0 } else { dnc };
                    system[(j, q + k)] -= dt_dtc * dtc + dt_dnc * dnc;
                    system[(q + j, q + k)] -= dn_dtc * dtc + dn_dnc * dnc;
                }
            }
            SiteSystem::Direct(lu_factor(system.as_ref(), "the linearized EP fixed point")?)
        };
        Ok(self.site_system.get_or_init(|| system))
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
/// `y = M⁻¹g` ([`ConeNormalizer::solved_gradient`]) and to each column of the covariance basis `N`
/// ([`ConeNormalizer::covariance_basis`]).
///
/// Where `M⁻¹` is the criterion's kept-spectrum pseudo-inverse `M⁺`, its derivative is not
/// `−M⁺ṀM⁺` alone: the kept eigenvectors rotate into the dropped ones. That part of `D(M⁺)[Ṁ]`
/// is applied to `g` and to each column of the basis generator `C` with `N = M⁺C`
/// ([`ConeNormalizer::covariance_generator`]) in `inverse_rotation_on_gradient` and
/// `inverse_rotation_on_generator`, zero where `M⁻¹` is an inverse (gam#2952).
#[derive(Clone, Debug)]
pub struct ConeCoordinateMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    pub precision_rate_on_y: Array1<f64>,
    pub precision_rate_on_basis: Array2<f64>,
    pub inverse_rotation_on_gradient: Array1<f64>,
    pub inverse_rotation_on_generator: Array2<f64>,
}

/// One coordinate pair's second-order motion: `v_kl`, `g̈_kl`, and the pair drift `M̈_kl` through
/// the only contraction [`ConeNormalizer::second_order`] reads it in, `tr(Q·M̈_kl)` with `Q` the
/// normalizer's [`ConeDriftWeight`] (gam#3347).
#[derive(Clone, Debug)]
pub struct ConePairMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    /// `tr(Q·M̈_kl)`, `Q` from [`ConeNormalizer::drift_weight`].
    pub precision_rate_trace: f64,
    /// What the inverse identities in [`ConeNormalizer::second_order`] omit where `M⁻¹` is the
    /// criterion's kept-spectrum pseudo-inverse: `(D²M⁺[Ṁ_k, Ṁ_l] − M⁺Ṁ_kM⁺Ṁ_lM⁺ − M⁺Ṁ_lM⁺Ṁ_kM⁺)`
    /// plus the rotation of the pair drift `D M⁺[M̈] + M⁺M̈M⁺`, applied to `g` and to each column
    /// of the basis generator `C`. Zero where `M⁻¹` is an inverse (gam#2952).
    pub inverse_rotation_on_gradient: Array1<f64>,
    pub inverse_rotation_on_generator: Array2<f64>,
}

/// The weight `Q` through which [`ConeNormalizer::second_order`] reads a pair's precision drift:
/// the pair value is `tr(Q·M̈) + (terms free of M̈)`, with
///
/// ```text
/// Q = ½ y yᵀ − sym(y (N Bᵀγ)ᵀ) + N Γ Nᵀ = Z S Zᵀ,   Z = [y  N],   S = [[½, −½cᵀ], [−½c, Γ]],   c = Bᵀγ,
/// ```
///
/// since `M̈` enters `C̈` only as `+½ yᵀM̈y` (through `ÿ`), `−γᵀB NᵀM̈y` (through `m̈₀`) and
/// `+tr(Γ NᵀM̈N)` (through `K̈`). `Q` is held as signed factors from the eigenpairs `S = VΛVᵀ`:
/// the columns `√|λ_a| Z v_a` split by the sign of `λ_a`, so `Q = F₊F₊ᵀ − F₋F₋ᵀ` exactly and
/// `tr(Q·E) = tr(F₊ᵀEF₊) − tr(F₋ᵀEF₋)` for any symmetric `E`. A caller can then contract a drift
/// it never forms, such as a family's fourth-derivative correction, through a trace kernel
/// instead of applying it to `y` and every column of `N` (gam#3347).
#[derive(Clone, Debug)]
pub struct ConeDriftWeight {
    positive: Array2<f64>,
    negative: Array2<f64>,
}

impl ConeDriftWeight {
    /// `F₊`, `p × a`.
    pub fn positive(&self) -> &Array2<f64> {
        &self.positive
    }

    /// `F₋`, `p × b`.
    pub fn negative(&self) -> &Array2<f64> {
        &self.negative
    }

    /// `tr(Q·E)` for a symmetric `E` given by its action.
    pub fn trace(&self, apply: &dyn Fn(&Array1<f64>) -> Array1<f64>) -> f64 {
        let side = |factor: &Array2<f64>| -> f64 {
            (0..factor.ncols())
                .map(|index| {
                    let column = factor.column(index).to_owned();
                    column.dot(&apply(&column))
                })
                .sum()
        };
        side(&self.positive) - side(&self.negative)
    }
}

/// A coordinate's first derivative of `C`, with the rates its pairs reuse.
#[derive(Clone, Debug)]
pub struct ConeFirstOrder {
    pub derivative: f64,
    y_rate: Array1<f64>,
    m0_rate: Array1<f64>,
    k_rate: Array2<f64>,
    /// `M⁻¹ Ṁ N`.
    solved_rate_on_basis: Array2<f64>,
}

/// The constrained Laplace normalizer's criterion share `C` at one inner mode, with the state its
/// outer derivatives contract.
#[derive(Clone, Debug)]
pub struct ConeNormalizer {
    value: f64,
    /// Unit-scaled rows inside the mass horizon, `q × p`.
    rows: Array2<f64>,
    y: Array1<f64>,
    /// `N`, `p × r`, with `K = NᵀMN` the orthant's covariance and `A M⁻¹ = B Nᵀ`.
    basis: Array2<f64>,
    /// `C`, `p × r`, with `N = M⁻¹C`.
    generator: Array2<f64>,
    gradient: Array1<f64>,
    orthant: OrthantLogMass,
}

/// The columns any of `rows` touches.
fn supported_columns<'a>(rows: impl Iterator<Item = &'a Array1<f64>>, p: usize) -> Vec<usize> {
    let mut touched = vec![false; p];
    for row in rows {
        for (column, &value) in row.iter().enumerate() {
            touched[column] |= value != 0.0;
        }
    }
    (0..p).filter(|&column| touched[column]).collect()
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
        let mut units: Vec<(Array1<f64>, f64)> = Vec::new();
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
            if half_spaces.insert(half_space) {
                units.push((unit, bound));
            }
        }
        // `M⁻¹a` for every row, from whichever is fewer: one solve per row, or one per supported
        // column (`M⁻¹a = Σ_c a_c M⁻¹e_c`, exact since `a` vanishes off its support).
        let support = supported_columns(units.iter().map(|(unit, _)| unit), p);
        let unit_solve = |column: usize| {
            let mut unit = Array1::<f64>::zeros(p);
            unit[column] = 1.0;
            solve(&unit)
        };
        let column_solves: Option<Array2<f64>> = (units.len() > support.len()).then(|| {
            let mut solves = Array2::<f64>::zeros((p, support.len()));
            for (index, &column) in support.iter().enumerate() {
                solves.column_mut(index).assign(&unit_solve(column));
            }
            solves
        });
        let mut kept: Vec<(Array1<f64>, Array1<f64>, f64)> = Vec::new();
        for (unit, scaled_bound) in units {
            let solved = match column_solves.as_ref() {
                Some(solves) => solves.dot(&Array1::from_shape_fn(support.len(), |i| unit[support[i]])),
                None => solve(&unit),
            };
            let variance = unit.dot(&solved);
            if !variance.is_finite() {
                return Err(ConeNormalizerRefusal::NonFinite { what: "constraint-normal variance" });
            }
            if !(variance > 0.0) {
                continue;
            }
            let mean = unit.dot(&center) - scaled_bound;
            if mean / variance.sqrt() < horizon {
                kept.push((unit, solved, mean));
            }
        }
        let q = kept.len();
        let kept_support = supported_columns(kept.iter().map(|(unit, _, _)| unit), p);
        let mut a = Array2::<f64>::zeros((q, p));
        let mut m0 = Array1::<f64>::zeros(q);
        for (i, (unit, _, mean)) in kept.iter().enumerate() {
            a.row_mut(i).assign(unit);
            m0[i] = *mean;
        }
        let row_coordinates = q <= kept_support.len();
        let (loadings, basis, generator, k) = if row_coordinates {
            // Row coordinates: B = I, N = R = M⁻¹Aᵀ, K = W = AR.
            let mut r = Array2::<f64>::zeros((p, q));
            for (i, (_, solved, _)) in kept.iter().enumerate() {
                r.column_mut(i).assign(solved);
            }
            let w = symmetrized(&a.dot(&r));
            (Array2::<f64>::eye(q), r, a.t().to_owned(), w)
        } else {
            // Supported-column coordinates: B = A on them, N = M⁻¹ on them, K = N on them.
            let mut basis = Array2::<f64>::zeros((p, kept_support.len()));
            for (index, &column) in kept_support.iter().enumerate() {
                let solved = match column_solves.as_ref() {
                    Some(solves) => solves.column(support.binary_search(&column).expect(
                        "a kept row's support lies in the support of all rows",
                    ))
                    .to_owned(),
                    None => unit_solve(column),
                };
                basis.column_mut(index).assign(&solved);
            }
            let r = kept_support.len();
            let k = symmetrized(&Array2::from_shape_fn((r, r), |(i, j)| basis[[kept_support[i], j]]));
            let loadings = Array2::from_shape_fn((q, r), |(i, j)| a[[i, kept_support[j]]]);
            let mut generator = Array2::<f64>::zeros((p, r));
            for (index, &column) in kept_support.iter().enumerate() {
                generator[[column, index]] = 1.0;
            }
            (loadings, basis, generator, k)
        };
        if basis.iter().any(|value| !value.is_finite()) {
            return Err(ConeNormalizerRefusal::NonFinite { what: "covariance basis" });
        }
        let orthant = if row_coordinates {
            OrthantLogMass::converge(&m0, &k)?
        } else {
            OrthantLogMass::converge_loaded(&m0, &loadings, &k)?
        };
        let value = -0.5 * gradient.dot(&y) - orthant.log_mass();
        if !value.is_finite() {
            return Err(ConeNormalizerRefusal::NonFinite { what: "normalizer value" });
        }
        Ok(Self { value, rows: a, y, basis, generator, gradient: gradient.clone(), orthant })
    }

    /// `C = −½gᵀM⁻¹g − ln P(u ≥ 0)`.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Rows inside the mass horizon.
    pub fn retained_rows(&self) -> usize {
        self.rows.nrows()
    }

    /// The dimension EP carries the orthant in: the retained rows or their supported columns,
    /// whichever are fewer.
    pub fn orthant_dimension(&self) -> usize {
        self.orthant.dimension()
    }

    /// The generator `C`, `p × r`, of the covariance basis `N = M⁻¹C` and `K = CᵀM⁻¹C`: the unit
    /// normals `Aᵀ` in row coordinates, the unit vectors of the supported columns otherwise.
    pub fn covariance_generator(&self) -> &Array2<f64> {
        &self.generator
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

    /// The covariance basis `N`, `p × r`: `M⁻¹Aᵀ` in row coordinates, the columns of `M⁻¹` on
    /// the supported columns otherwise.
    pub fn covariance_basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// The weight `Q` a pair's precision drift is read through, as signed factors
    /// ([`ConeDriftWeight`]).
    pub fn drift_weight(&self) -> Result<ConeDriftWeight, ConeNormalizerRefusal> {
        use gam_linalg::faer_ndarray::FaerEigh;
        let (p, r) = self.basis.dim();
        let loaded = self.orthant.loadings.t().dot(&self.orthant.mean_gradient());
        let covariance_gradient = self.orthant.covariance_gradient();
        let mut weight = Array2::<f64>::zeros((r + 1, r + 1));
        weight[[0, 0]] = 0.5;
        weight.slice_mut(s![1.., 1..]).assign(&covariance_gradient);
        for a in 0..r {
            weight[[0, a + 1]] = -0.5 * loaded[a];
            weight[[a + 1, 0]] = -0.5 * loaded[a];
        }
        let (values, vectors) = weight
            .eigh(faer::Side::Lower)
            .map_err(|error| ConeNormalizerRefusal::Singular { reason: format!("drift weight spectrum: {error:?}") })?;
        let mut span = Array2::<f64>::zeros((p, r + 1));
        span.column_mut(0).assign(&self.y);
        span.slice_mut(s![.., 1..]).assign(&self.basis);
        let directions = span.dot(&vectors);
        let side = |keep: &dyn Fn(f64) -> bool| {
            let chosen: Vec<usize> = (0..values.len()).filter(|&a| keep(values[a])).collect();
            let mut factor = Array2::<f64>::zeros((p, chosen.len()));
            for (index, &a) in chosen.iter().enumerate() {
                factor.column_mut(index).assign(&directions.column(a).mapv(|value| value * values[a].abs().sqrt()));
            }
            factor
        };
        let weight = ConeDriftWeight {
            positive: side(&|value: f64| value > 0.0),
            negative: side(&|value: f64| value < 0.0),
        };
        if weight.positive.iter().chain(weight.negative.iter()).any(|value| !value.is_finite()) {
            return Err(ConeNormalizerRefusal::NonFinite { what: "drift weight" });
        }
        Ok(weight)
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
        let k_rate = symmetrized(
            &(self.generator.t().dot(&motion.inverse_rotation_on_generator)
                - self.basis.t().dot(&motion.precision_rate_on_basis)),
        );
        let derivative = -0.5 * (motion.gradient_rate.dot(&self.y) + self.gradient.dot(&y_rate))
            - self.orthant.mean_gradient().dot(&m0_rate)
            - frobenius(&self.orthant.covariance_gradient(), &k_rate);
        let (p, r) = self.basis.dim();
        let mut solved_rate_on_basis = Array2::<f64>::zeros((p, r));
        for column in 0..r {
            solved_rate_on_basis
                .column_mut(column)
                .assign(&solve(&motion.precision_rate_on_basis.column(column).to_owned()));
        }
        ConeFirstOrder { derivative, y_rate, m0_rate, k_rate, solved_rate_on_basis }
    }

    /// Second derivative for the coordinate pair `(k, l)`.
    ///
    /// `M ÿ = g̈ − Ṁ_l ẏ_k − Ṁ_k ẏ_l − M̈ y`, read only through `yᵀ(·)` and `A M⁻¹(·) = BNᵀ(·)`,
    /// so `Ṁ_l ẏ_k` enters as `(Ṁ_l y)ᵀẏ_k` and `(Ṁ_l N)ᵀẏ_k`;
    /// `K̈ = Nᵀ(Ṁ_k M⁻¹ Ṁ_l + Ṁ_l M⁻¹ Ṁ_k − M̈)N`. Every `M̈` term is linear in it and sums to
    /// `tr(Q·M̈)` ([`ConeDriftWeight`]), which the pair supplies as `precision_rate_trace`; what
    /// is formed below is the rest.
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
            - motion_k.precision_rate_on_y.dot(&first_l.y_rate);
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
        // `A M⁻¹(·) = BNᵀ(·)` with `N = M⁻¹C`; the rotation, symmetric in `M⁺`, reads through
        // `Cᵀ` the same way.
        let basis_turned = motion_k.inverse_rotation_on_generator.t().dot(&motion_l.gradient_rate)
            + motion_l.inverse_rotation_on_generator.t().dot(&motion_k.gradient_rate)
            + self.generator.t().dot(&pair.inverse_rotation_on_gradient)
            + motion_l.precision_rate_on_basis.t().dot(turn_k)
            + motion_k.precision_rate_on_basis.t().dot(turn_l);
        let basis_moved = self.basis.t().dot(&pair.gradient_rate)
            - motion_l.precision_rate_on_basis.t().dot(&first_k.y_rate)
            - motion_k.precision_rate_on_basis.t().dot(&first_l.y_rate)
            + basis_turned;
        let normal_moved = self.orthant.loadings.dot(&basis_moved);
        let m0_second = self.rows.dot(&pair.mode_response) - normal_moved;
        let cross = motion_k.precision_rate_on_basis.t().dot(&first_l.solved_rate_on_basis);
        let k_second = symmetrized(
            &(&cross + &cross.t() + self.generator.t().dot(&pair.inverse_rotation_on_generator)),
        );
        let (d_gamma, d_big_gamma) = self.orthant.gradient_motion(&first_l.m0_rate, &first_l.k_rate)?;
        let second_log_mass = d_gamma.dot(&first_k.m0_rate)
            + frobenius(&d_big_gamma, &first_k.k_rate)
            + self.orthant.mean_gradient().dot(&m0_second)
            + frobenius(&self.orthant.covariance_gradient(), &k_second);
        Ok(-0.5 * second_gy - second_log_mass + pair.precision_rate_trace)
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

    /// Site `j`'s cavity read from a freshly formed posterior: the re-inverted reference the
    /// fixed-point tests sweep with.
    fn cavity(mass: &OrthantLogMass, j: usize) -> (f64, f64) {
        let marginals = mass.marginals();
        let s_jj = marginals.variance[j];
        (1.0 / s_jj - mass.tau[j], marginals.mean[j] / s_jj - mass.nu[j])
    }

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

    /// One undamped EP sweep read straight off the definitions: each site's cavity from the
    /// posterior re-formed from the sites (`Σ_z = (I + KG)⁻¹K`) after every site.
    fn sequential_site_sweep(state: &mut OrthantLogMass) {
        for j in 0..state.m0.len() {
            let (tau_c, nu_c) = cavity(state, j);
            let update = site_update(tau_c, nu_c);
            state.tau[j] = update.tau;
            state.nu[j] = update.nu;
            state.post = state.form_posterior().expect("admissible sites");
        }
        state.site_system = std::sync::OnceLock::new();
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
        sequential_site_sweep(&mut checked);
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
        let motion_with = |turn_on_gradient: Array1<f64>, turn_on_generator: Array2<f64>| {
            ConeCoordinateMotion {
                mode_response: b1.clone(),
                gradient_rate: g1.clone(),
                precision_rate_on_y: precision_rate.dot(normalizer.solved_gradient()),
                precision_rate_on_basis: precision_rate.dot(normalizer.covariance_basis()),
                inverse_rotation_on_gradient: turn_on_gradient,
                inverse_rotation_on_generator: turn_on_generator,
            }
        };
        let exact = motion_with(turn.apply(&g0), turn.apply_columns(normalizer.covariance_generator()));
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
            Array2::zeros(normalizer.covariance_generator().raw_dim()),
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
                precision_rate_on_basis: rate.dot(normalizer.covariance_basis()),
                inverse_rotation_on_gradient: turn.apply(&(&g0 + &(&g1 * t))),
                inverse_rotation_on_generator: turn.apply_columns(normalizer.covariance_generator()),
            }
        };
        // The second order is graded on one row, where EP is exact (P = Φ), so the check measures
        // the rotation algebra alone in row coordinates; and on three rows over two columns, where
        // EP runs in the supported columns (`B = A|sup`, `C` their unit vectors, not `Aᵀ`), so the
        // same algebra is read through the generator in both coordinate systems.
        let fixtures = [
            (array![[0.0, 1.0, 0.0]], array![0.0], 1),
            (array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], array![0.0, -0.3, -0.2], 2),
        ];
        for (row_one, bound_one, orthant_dimension) in fixtures {
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
            assert_eq!(normalizer_one.retained_rows(), row_one.nrows(), "the rows are inside the mass horizon");
            assert_eq!(normalizer_one.orthant_dimension(), orthant_dimension, "EP runs in the expected coordinates");
            let normals = normalizer_one.covariance_generator();
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
            let pair_with = |turn_on_gradient: Array1<f64>, turn_on_generator: Array2<f64>| ConePairMotion {
                mode_response: Array1::zeros(3),
                gradient_rate: Array1::zeros(3),
                precision_rate_trace: normalizer_one
                    .drift_weight()
                    .expect("drift weight")
                    .trace(&|v: &Array1<f64>| second_precision.dot(v)),
                inverse_rotation_on_gradient: turn_on_gradient,
                inverse_rotation_on_generator: turn_on_generator,
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
                inverse_rotation_on_generator: Array2::zeros(normals.raw_dim()),
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
    }

    /// The quadratic path `β(t), g(t), M(t)` of the derivative checks, `(x₀, x₁, x₂)` its
    /// coefficients.
    struct NormalizerPath {
        beta: [Array1<f64>; 3],
        gradient: [Array1<f64>; 3],
        precision: [Array2<f64>; 3],
    }

    /// `dC` and `d²C` along `path` against central differences of `C` and `dC`; returns the
    /// normalizer at `t = 0`.
    fn assert_normalizer_derivatives_match_central_differences(
        rows: &Array2<f64>,
        bounds: &Array1<f64>,
        path: &NormalizerPath,
    ) -> ConeNormalizer {
        let [b0, b1, b2] = &path.beta;
        let [g0, g1, g2] = &path.gradient;
        let [m0, m1, m2] = &path.precision;
        let state = |t: f64| {
            (
                b0 + &(b1 * t) + &(b2 * (t * t)),
                g0 + &(g1 * t) + &(g2 * (t * t)),
                m0 + &(m1 * t) + &(m2 * (t * t)),
            )
        };
        let value_at = |t: f64| {
            let (beta, gradient, m) = state(t);
            let solve = dense_solve(&m);
            ConeNormalizer::evaluate(rows, bounds, &beta, &gradient, &solve)
                .expect("normalizer")
                .value()
        };
        let motion_at = |t: f64, normalizer: &ConeNormalizer| {
            let m_rate = m1 + &(m2 * (2.0 * t));
            ConeCoordinateMotion {
                mode_response: b1 + &(b2 * (2.0 * t)),
                gradient_rate: g1 + &(g2 * (2.0 * t)),
                precision_rate_on_y: m_rate.dot(normalizer.solved_gradient()),
                precision_rate_on_basis: m_rate.dot(normalizer.covariance_basis()),
                // `dense_solve` is an inverse: nothing rotates.
                inverse_rotation_on_gradient: Array1::zeros(b1.len()),
                inverse_rotation_on_generator: Array2::zeros(normalizer.covariance_generator().raw_dim()),
            }
        };
        let derivative_at = |t: f64| {
            let (beta, gradient, m) = state(t);
            let solve = dense_solve(&m);
            let normalizer =
                ConeNormalizer::evaluate(rows, bounds, &beta, &gradient, &solve).expect("normalizer");
            normalizer.first_order(&motion_at(t, &normalizer), &solve).derivative
        };
        let (beta, gradient, m) = state(0.0);
        let solve = dense_solve(&m);
        let normalizer =
            ConeNormalizer::evaluate(rows, bounds, &beta, &gradient, &solve).expect("normalizer");
        assert!(normalizer.retained_rows() >= 2, "at least two correlated rows are near their bounds");
        let motion = motion_at(0.0, &normalizer);
        let first = normalizer.first_order(&motion, &solve);
        let (fd, bar) = richardson(&value_at, 1.0e-2);
        assert!(
            (first.derivative - fd).abs() <= bar,
            "dC {} against central difference {fd} (bar {bar})",
            first.derivative
        );
        // gam#3347: the factored weight is the three places the pair value reads `M̈`, pinned on
        // this path's drift. `tr(Q·E) = tr(S·ZᵀEZ)`; the eigensolver reconstructs `S` to a normwise
        // backward error of `(r+1)³` rounded operations and each entry of `ZᵀEZ` takes `2p`, so
        // the two agree to that band on `‖S‖_F ‖ZᵀEZ‖_F`, which bounds either side.
        let drift = m2 * 2.0;
        let (y, basis) = (normalizer.solved_gradient(), normalizer.covariance_basis());
        let loaded = normalizer.orthant.loadings.t().dot(&normalizer.orthant.mean_gradient());
        let covariance_gradient = normalizer.orthant.covariance_gradient();
        let explicit = 0.5 * y.dot(&drift.dot(y)) - loaded.dot(&basis.t().dot(&drift.dot(y)))
            + frobenius(&covariance_gradient, &basis.t().dot(&drift.dot(basis)));
        let factored = normalizer.drift_weight().expect("drift weight").trace(&|v: &Array1<f64>| drift.dot(v));
        let (p, r) = basis.dim();
        let mut span = Array2::<f64>::zeros((p, r + 1));
        span.column_mut(0).assign(y);
        span.slice_mut(s![.., 1..]).assign(basis);
        let projected = span.t().dot(&drift.dot(&span));
        let weight_norm = (0.25 + 0.5 * loaded.dot(&loaded) + frobenius(&covariance_gradient, &covariance_gradient)).sqrt();
        let weight_band = band((r + 1).pow(3) + 2 * p, weight_norm * frobenius(&projected, &projected).sqrt());
        assert!(
            (factored - explicit).abs() <= weight_band,
            "tr(Q·E) {factored} from the signed factors against {explicit} read term by term (band {weight_band:e})"
        );
        let pair = ConePairMotion {
            mode_response: b2 * 2.0,
            gradient_rate: g2 * 2.0,
            precision_rate_trace: normalizer.drift_weight().expect("drift weight").trace(&|v: &Array1<f64>| (m2 * 2.0).dot(v)),
            inverse_rotation_on_gradient: Array1::zeros(b2.len()),
            inverse_rotation_on_generator: Array2::zeros(normalizer.covariance_generator().raw_dim()),
        };
        let second = normalizer
            .second_order(&motion, &first, &motion, &first, &pair)
            .expect("second order");
        let (fd2, bar2) = richardson(&derivative_at, 1.0e-2);
        assert!(
            (second - fd2).abs() <= bar2,
            "d²C {second} against central difference of dC {fd2} (bar {bar2})"
        );
        normalizer
    }

    #[test]
    fn the_normalizer_derivatives_match_central_differences_2765() {
        let rows = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8]];
        let bounds = array![0.0, 0.0, -0.05];
        let path = NormalizerPath {
            beta: [array![0.01, 0.02, 0.3], array![-0.2, 0.1, 0.05], array![0.05, -0.1, 0.02]],
            gradient: [array![0.4, 0.3, 0.0], array![0.1, -0.2, 0.05], array![-0.03, 0.02, 0.01]],
            precision: [
                array![[2.0, 0.8, 0.3], [0.8, 1.5, -0.4], [0.3, -0.4, 1.2]],
                array![[0.3, -0.1, 0.0], [-0.1, 0.2, 0.05], [0.0, 0.05, -0.1]],
                array![[0.05, 0.0, 0.02], [0.0, -0.04, 0.0], [0.02, 0.0, 0.03]],
            ],
        };
        let normalizer = assert_normalizer_derivatives_match_central_differences(&rows, &bounds, &path);
        assert_eq!(normalizer.orthant_dimension(), normalizer.retained_rows(), "rows are the fewer");
    }

    /// `u = m₀ + Bz`, `z ~ N(0, K)`, is the orthant of `N(m₀, BKBᵀ)`. Carried in the `r`
    /// coordinates of `z` EP runs the same site updates it runs in the `q` rows' own, so the two
    /// converge to the same `ln P`, to the rounding of the sweeps both took. At the same sites the
    /// two coordinates give the same `ln Z_EP`, `γ`, `Γ_K = BᵀΓ_W B` (as `dW = B dK Bᵀ`) and
    /// motion of both. With `q = 6` rows on `r = 2` coordinates the loaded fixed point is solved
    /// through the `m = 5` capacitance and the rows' own through the `2q = 12` direct system, so
    /// this also checks the one against the other.
    #[test]
    fn loaded_coordinates_agree_with_the_rows_own_3037() {
        let b = array![[1.0, 0.0], [0.8, 0.3], [0.5, 0.7], [0.1, 1.0], [1.0, -0.4], [0.6, 0.6]];
        let k = array![[0.9, 0.35], [0.35, 0.6]];
        let m0 = array![-0.3, 0.1, 0.4, -0.2, 0.25, 0.05];
        let w = b.dot(&k).dot(&b.t());
        let loaded = OrthantLogMass::converge_loaded(&m0, &b, &k).expect("loaded EP converges");
        let converged_rows = OrthantLogMass::converge(&m0, &w).expect("row EP converges");
        let (q, r) = b.dim();
        let operations = (loaded.sweeps() + converged_rows.sweeps()) * (4 * q * q + 8 * q);
        let log_scale = converged_rows.log_mass().abs().max(1.0);
        eprintln!(
            "[3037-LOADED] ln P {:.15e} (r = {r}, {} sweeps) against {:.15e} (q = {q}, {} sweeps)",
            loaded.log_mass(),
            loaded.sweeps(),
            converged_rows.log_mass(),
            converged_rows.sweeps()
        );
        assert_eq!(loaded.dimension(), r);
        assert!(
            (loaded.log_mass() - converged_rows.log_mass()).abs() <= band(operations, log_scale),
            "ln P {} in z against {} in the rows",
            loaded.log_mass(),
            converged_rows.log_mass()
        );
        // The rows' own coordinates at the loaded fixed point's sites.
        let mut rows = converged_rows.clone();
        rows.tau = loaded.tau.clone();
        rows.nu = loaded.nu.clone();
        rows.post = rows.form_posterior().expect("admissible sites");
        rows.site_system = std::sync::OnceLock::new();
        let (rows_log_mass, _) = rows.evaluate_log_mass().expect("a finite EP log mass");
        let (loaded_log_mass, _) = loaded.evaluate_log_mass().expect("a finite EP log mass");
        let operations = 4 * q * q * q;
        assert!(
            (loaded_log_mass - rows_log_mass).abs() <= band(operations, log_scale),
            "ln Z_EP {loaded_log_mass} in z against {rows_log_mass} in the rows at the same sites"
        );
        let close = |got: &Array1<f64>, want: &Array1<f64>, what: &str| {
            let scale = want.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
            for i in 0..want.len() {
                assert!(
                    (got[i] - want[i]).abs() <= band(operations, scale),
                    "{what}[{i}] {} against {}",
                    got[i],
                    want[i]
                );
            }
        };
        let close2 = |got: &Array2<f64>, want: &Array2<f64>, what: &str| {
            let flat = |m: &Array2<f64>| Array1::from_iter(m.iter().copied());
            close(&flat(got), &flat(want), what);
        };
        close(&loaded.mean_gradient(), &rows.mean_gradient(), "γ");
        let pulled = |gamma_w: &Array2<f64>| b.t().dot(gamma_w).dot(&b);
        close2(&loaded.covariance_gradient(), &pulled(&rows.covariance_gradient()), "Γ_K");
        let dm0 = array![0.2, -0.1, 0.3, 0.05, -0.25, 0.1];
        let dk = array![[0.1, -0.05], [-0.05, 0.2]];
        let (dg_loaded, dbg_loaded) = loaded.gradient_motion(&dm0, &dk).expect("loaded motion");
        let (dg_rows, dbg_rows) = rows.gradient_motion(&dm0, &b.dot(&dk).dot(&b.t())).expect("row motion");
        assert!(matches!(loaded.site_system().expect("formed"), SiteSystem::Capacitance { .. }));
        assert!(matches!(rows.site_system().expect("formed"), SiteSystem::Direct(_)));
        close(&dg_loaded, &dg_rows, "dγ");
        close2(&dbg_loaded, &pulled(&dbg_rows), "dΓ_K");
    }

    /// Five rows on three of four columns: the normalizer carries EP in the three supported
    /// columns, and its outer derivatives are still those of `C`.
    #[test]
    fn supported_column_normalizer_derivatives_match_central_differences_3037() {
        let rows = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.6, 0.8, 0.0],
            [0.7, -0.7, 0.0, 0.0],
            [0.3, 0.4, 0.5, 0.0]
        ];
        let bounds = array![0.0, 0.0, -0.05, -0.02, 0.1];
        let path = NormalizerPath {
            beta: [
                array![0.01, 0.02, 0.3, -0.4],
                array![-0.2, 0.1, 0.05, 0.3],
                array![0.05, -0.1, 0.02, 0.0],
            ],
            gradient: [
                array![0.4, 0.3, 0.0, 0.1],
                array![0.1, -0.2, 0.05, 0.0],
                array![-0.03, 0.02, 0.01, 0.02],
            ],
            precision: [
                array![[2.0, 0.8, 0.3, 0.2], [0.8, 1.5, -0.4, 0.1], [0.3, -0.4, 1.2, -0.3], [0.2, 0.1, -0.3, 1.0]],
                array![[0.3, -0.1, 0.0, 0.05], [-0.1, 0.2, 0.05, 0.0], [0.0, 0.05, -0.1, 0.02], [0.05, 0.0, 0.02, 0.1]],
                array![[0.05, 0.0, 0.02, 0.0], [0.0, -0.04, 0.0, 0.01], [0.02, 0.0, 0.03, 0.0], [0.0, 0.01, 0.0, 0.02]],
            ],
        };
        let normalizer = assert_normalizer_derivatives_match_central_differences(&rows, &bounds, &path);
        assert_eq!(normalizer.retained_rows(), 5);
        assert_eq!(normalizer.orthant_dimension(), 3, "EP runs in the three supported columns");
    }

    /// gam#3037/#3038: the location-scale time-derivative guard emits one row per observation on
    /// the few time-basis columns, thousands of rows EP carried as a `q × q` covariance. Two
    /// thousand such rows on three of five columns, most of them near their bound, are carried in
    /// the three columns, and the outer derivatives of `C` still match central differences.
    #[test]
    fn thousands_of_guard_rows_are_carried_in_their_columns_3037() {
        let q = 2000;
        let rows = Array2::from_shape_fn((q, 5), |(i, c)| {
            let t = i as f64 / (q - 1) as f64;
            match c {
                0 => 1.0,
                1 => t,
                2 => t * t,
                _ => 0.0,
            }
        });
        let bounds = Array1::zeros(q);
        // The guard's slope β₀ + β₁t + β₂t² stays positive and comes within 0.009 of its bound.
        let path = NormalizerPath {
            beta: [
                array![0.02, -0.03, 0.02, 0.4, -0.2],
                array![0.01, -0.005, 0.004, 0.1, 0.0],
                array![0.002, 0.001, -0.001, 0.0, 0.05],
            ],
            gradient: [
                array![0.05, 0.02, 0.01, 0.0, 0.0],
                array![0.01, -0.01, 0.005, 0.0, 0.0],
                array![-0.002, 0.001, 0.0, 0.0, 0.0],
            ],
            precision: [
                array![
                    [400.0, 150.0, 90.0, 5.0, 0.0],
                    [150.0, 120.0, 80.0, 0.0, 2.0],
                    [90.0, 80.0, 70.0, 1.0, 0.0],
                    [5.0, 0.0, 1.0, 3.0, 0.5],
                    [0.0, 2.0, 0.0, 0.5, 2.0]
                ],
                array![
                    [20.0, 5.0, 2.0, 0.0, 0.0],
                    [5.0, 6.0, 3.0, 0.0, 0.0],
                    [2.0, 3.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.1]
                ],
                array![
                    [4.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 0.5, 0.0, 0.0],
                    [0.0, 0.5, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ],
            ],
        };
        let normalizer = assert_normalizer_derivatives_match_central_differences(&rows, &bounds, &path);
        eprintln!(
            "[3037-GUARD] {} retained rows carried in {} coordinates; {} EP sweeps; C = {:.12e}",
            normalizer.retained_rows(),
            normalizer.orthant_dimension(),
            normalizer.sweeps(),
            normalizer.value()
        );
        assert!(normalizer.retained_rows() > 1000, "most guard rows are inside the mass horizon");
        assert_eq!(normalizer.orthant_dimension(), 3);
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
