//! The constrained Laplace term `½ ln|M| + C`, priced as one quantity in the natural parameters
//! of its constraint sites (gam#2765).
//!
//! # Why the covariance form is singular where the quantity is not
//!
//! [`super::ConeNormalizer`] prices `C = −½gᵀM⁻¹g − ln P(u ≥ 0)` beside the criterion's own
//! `½ ln|M|`, both through the covariance form of the constraint-normal law, `u ~ N(m₀, W)` with
//! `W = AM⁻¹Aᵀ`. Take a row active with multiplier `μ > 0`, and let `σ` be `M`'s curvature along
//! the row's normal given the face (the Schur complement of `M` onto `t = aᵀδ`). Then `W ∝ 1/σ`.
//! As `σ → 0⁺`, `½ln|M| → −∞` and `C → +∞`, while their sum tends to `ln μ + ½ln 2π`. At `σ < 0`
//! the kept spectrum drops the direction from both pieces, and the sum loses `ln μ`. That is the
//! jump of about 5 over Δρ ≈ 3e-3 on the #3006 link-wiggle fits. The quantity itself,
//! `−ln ∫_{Aβ≥b} e^{−F}` to Laplace order, is not singular there. Its normal factor
//! `∫₀^∞ e^{−μt − σt²/2} dt` is smooth in `σ` through zero
//! ([`gam_math::gaussian_reciprocal::half_line_gaussian_log_jet`]). Only the representation
//! `W = 1/σ` is singular.
//!
//! # What is priced
//!
//! EP runs on the rows' indicators with the Gaussian part in the coefficient space's natural
//! parameters. The sites are `(τ̃, ν̃)` over `u = Aβ − b`, the posterior precision is
//! `Λ = M + AᵀT̃A`, and `h = −g + Aᵀ(ν̃ − T̃d)` at the mode's slack `d = Aβ̂ − b`. With `δ̄ = Λ⁻¹h`,
//!
//! ```text
//! L = ½ln|M| + C = ½ln|Λ| − ½hᵀδ̄ − Σ_i (ν̃_i d_i − ½τ̃_i d_i²) − Σ_i ln c_i,
//! ```
//!
//! where `c_i` is the normalizer that gives site `i`'s Gaussian the tilted mass of its cavity.
//! - **No `M⁻¹`.** Nothing forms `M⁻¹`, `W` or `ln|M|`. `Λ` is positive definite wherever the mode
//!   is a strict minimum on the cone at Laplace order, because an active site's precision is about
//!   `μ²`.
//! - **Same value where `M` is positive definite.** There the sites reach the covariance form's
//!   EP fixed point, since EP's fixed point does not depend on how its Gaussian is parametrized.
//!   So `L` equals `½ln|M| + C` to rounding.
//! - **Continuation.** A cavity with non-positive precision along its normal reads its tilted
//!   moments off the principal-value continuation of the half-line integral, which is exact for
//!   one row on either side of `σ = 0`.
//! - **The fold.** A continued tilted variance that is not positive is the constrained mode's fold:
//!   the boundary has lost the Laplace regime. It is refused by name ([`ConeLaplaceRefusal::Fold`]).
//!   It cannot be one at SWEEP 0. With `B_i = Λ − τ̃_ia_ia_iᵀ`, Sherman-Morrison gives
//!   `τ_c = 1/Σ_ii − τ̃_i = 1/(a_iᵀB_i⁻¹a_i)` exactly, and at sweep 0 every site is `μ²` or zero, so
//!   `B_i ⪰ M`: a positive definite `M` forces `τ_c > 0`, the ordinary branch, and a positive
//!   variance. A fold reported there says `B_i` is not positive definite, or that the difference
//!   `1/Σ_ii − τ̃_i` lost its digits — never that the mode folded. Those two are ONE question in
//!   two forms: with `Λ` positive definite (which the Cholesky establishes before any site is
//!   read) `B_i ≻ 0` exactly when `τ̃_i·Σ_ii < 1`, and `τ_c` carries the sign of `1 − τ̃_i·Σ_ii`
//!   because `Σ_ii > 0`. The product is that question asked without the cancellation; the
//!   difference loses every digit as the product approaches 1. So the reading is whether the
//!   product clears 1 by more than `κ(Λ)·ε`, not whether it exceeds 1. The refusal carries both
//!   operands, their product and `M`'s curvature along the row's normal, the last being ONE-SIDED
//!   evidence: negative settles `B_i ⊁ 0`, while non-negative settles nothing, since `B_i` needs
//!   `M` positive in every direction coupling to `a_i` through `B_i⁻¹` and not only along `a_i`
//!   (gam#4571). From sweep 1 the sites may go negative, `B_i ⪰ M` fails, and the verdict is a
//!   verdict again.
//!
//! # Where EP starts
//!
//! A row at zero slack carries a KKT multiplier `μ_i` in `g = A_actᵀμ`. Its site starts at the
//! boundary limit: an exponential law of rate `μ_i` against a flat cavity, `(τ̃, ν̃) = (μ_i², 2μ_i)`.
//! Every other site starts at zero. The start only selects where EP begins. Its fixed point, and so
//! `L`, does not depend on it.
//!
//! # Which rows
//!
//! A row enters when the posterior puts more than `f64::EPSILON` of its mass outside the row's
//! half-space, the horizon [`super::ConeNormalizer`] reads. A row beyond it moves `L` by at most `ε`.
//! It is judged on the posterior, since `M` may be indefinite and has no prior marginal:
//! - first on the posterior of the starting sites;
//! - then again on the converged posterior, until no excluded row is inside the horizon.
//!
//! Rows repeated exactly enter once.
//!
//! # Derivatives
//!
//! At the EP fixed point `L` is stationary in the sites. So its derivative along an outer coordinate
//! is the fixed-site derivative of the Gaussian part:
//!
//! ```text
//! dL/dθ = ½tr(Λ⁻¹Ṁ) + δ̄ᵀġ + ½δ̄ᵀṀδ̄ − (ν̃ − T̃ū)ᵀAβ̂̇,    ū = Aδ̄ + d.
//! ```
//!
//! - **The trace term** is the log-determinant trace a criterion takes on `Λ`, the precision that
//!   [`ConeLaplace::laplace_precision`] returns. [`ConeLaplace::first_order`] returns the rest,
//!   the derivative of the share `L − ½ln|Λ|`.
//! - **The Hessian** is not stationary in the sites. The sites move along the linearized fixed
//!   point, a `2q × 2q` system in the posterior of `u`, `(Σ_u, ū) = (AΛ⁻¹Aᵀ, Aδ̄ + d)`, which is the
//!   covariance form's own system.
//! - **What [`ConeLaplace::second_order`] returns** is everything in `d²L` except the fixed-site
//!   log-determinant Hessian `½tr(Λ⁻¹M̈) − ½tr(Λ⁻¹Ṁ_lΛ⁻¹Ṁ_k)`. That includes the sites' motion
//!   through the trace term.
//!
//! # Rows that move (gam#3171)
//!
//! A family's inequality rows can depend on an outer coordinate: the transformation-normal
//! monotonicity rows are built on the covariate design, which moves with its length scale. Along
//! such a coordinate `A` and `b` move too, so `Λ̇ = Ṁ + ȦᵀT̃A + AᵀT̃Ȧ` and the slack moves by
//! `ḋ = Aβ̂̇ + Ȧβ̂ − ḃ`. The fixed-site derivative becomes
//!
//! ```text
//! dL/dθ = ½tr(Λ⁻¹Ṁ) + tr(T̃ȦR) + δ̄ᵀġ + ½δ̄ᵀṀδ̄ − (ν̃ − T̃ū)ᵀe,    e = Aβ̂̇ + Ȧ(β̂ + δ̄) − ḃ,
//! ```
//!
//! still with no site term, because each `ln c_i` is stationary in its own cavity at the fixed
//! point. `tr(T̃ȦR) = ½tr(Λ⁻¹(ȦᵀT̃A + AᵀT̃Ȧ))` is the share of the log-determinant trace that a
//! criterion taking `½tr(Λ⁻¹Ṁ)` on its own precision drift does not see, so the first order
//! returns it. The second order returns every term of `½tr(Λ⁻¹Λ̈) − ½tr(Λ⁻¹Λ̇_lΛ⁻¹Λ̇_k)` that is not
//! the criterion's `Ṁ` part, and the sites move under the row motion of `Σ_u = AΛ⁻¹Aᵀ`,
//! `Σ̇_u = ȦR + RᵀȦᵀ − RᵀΛ̇R`.
//!
//! `L` depends on each row only through its half-space, so it is unchanged by a positive
//! rescaling of any row with its bound, at every `θ`. The term holds its rows at unit scale at the
//! mode it was evaluated on. It therefore takes the caller's row rates `(ȧ_i, ḃ_i)` at the caller's
//! scale and divides them by that row's norm there, a constant rescaling that leaves `L` unchanged.

use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_math::gaussian_reciprocal::half_line_gaussian_log_jet;
use gam_math::probability::standard_normal_quantile;
use gam_math::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, s};

/// Why the constrained Laplace term could not be formed at a trial point.
#[derive(Clone, Debug)]
pub enum ConeLaplaceRefusal {
    /// An input (a row, bound, coefficient, gradient or precision) was not finite.
    NonFinite { what: &'static str },
    /// The inputs disagree in dimension.
    Dimension { reason: String },
    /// `Λ = M + AᵀT̃A` is not positive definite at these sites: the mode is not a strict minimum
    /// on the cone at Laplace order.
    NotPositiveDefinite { sweep: usize },
    /// A site's cavity has no decay along its row's normal: non-positive precision with a shift
    /// that does not pull toward the boundary.
    NotIntegrable { row: usize, cavity_precision: f64, cavity_shift: f64, sweep: usize },
    /// The continued tilted variance of a site is not positive: the constrained mode's fold.
    ///
    /// It carries the cavity that produced it. The verdict is a SIGN test on a cumulant that
    /// vanishes at the continuation's own boundary `τ_c = −μ²/2`, so a mode sitting near that
    /// boundary returns a small negative number, and the number alone cannot say whether it is
    /// past the fold or unresolved from zero. The cavity says which: `τ_c > 0` means the variance
    /// is a genuine variance and a non-positive one is a degeneracy, while `τ_c ≤ 0` means it is
    /// the principal-value continuation's formal second cumulant, read at `τ_c` against the
    /// boundary `−ν_c²/2` (gam#4571).
    Fold {
        row: usize,
        variance: f64,
        cavity_precision: f64,
        cavity_shift: f64,
        /// `τ̃_i` and `Σ_ii`, the two operands the cavity precision is the difference of.
        site_precision: f64,
        posterior_variance: f64,
        /// `M`'s curvature along this row's normal, `a_iᵀMa_i/a_iᵀa_i`, filled by
        /// [`ConeLaplace::evaluate`], which is the innermost frame holding both the precision and
        /// the caller's rows. It is evidence the verdict needs, not a diagnostic beside it, and it
        /// is ONE-SIDED: a negative value settles that `B_i = Λ − τ̃_ia_ia_iᵀ` is not positive
        /// definite, since `B_i ⪰ M` at sweep 0. A non-negative one settles nothing — `B_i` needs
        /// `M` positive in every direction that couples to `a_i` through `B_i⁻¹`, not only along
        /// `a_i` — and what separates the remaining readings is whether `τ̃_i·Σ_ii` clears 1 by
        /// more than `κ(Λ)·ε`.
        ///
        /// `None` IS A STATEMENT, not a missing value: it says this refusal was raised on the
        /// DERIVATIVE path ([`ConeLaplace::first_order`] and [`ConeLaplace::second_order`] reach
        /// [`Self::Fold`] through the per-site jacobian without passing through `evaluate`), where
        /// the sites are at their converged fixed point, `τ̃` may be negative, `B_i ⪰ M` fails and
        /// `M`'s curvature along the normal discriminates nothing. Read the cavity precision
        /// there. The term holds `Λ`, not `M`, so filling it anyway would mean a second `p × p`
        /// residency to report a number that does not answer the question (gam#4571).
        normal_curvature: Option<f64>,
        /// `(max L_ii / min L_ii)²` of `Λ`'s Cholesky, a LOWER bound on `κ₂(Λ)` and so on the band
        /// the product `τ̃_i·Σ_ii` is read against. One-sided in the direction that matters: a
        /// product within `condition_floor·ε` of 1 means the cavity's sign is the subtraction's
        /// FOR CERTAIN, which is the reading that makes this refusal a defect. A product outside
        /// it is NOT thereby genuine, because a lower bound cannot establish the converse
        /// (gam#4571).
        condition_floor: f64,
        sweep: usize,
    },
    /// EP's damped update stopped moving the sites before they settled to their rounding.
    NotContracting { sweeps: usize, fraction: f64, step: f64 },
    /// A system EP's derivatives solve is singular.
    Singular { reason: String },
    /// The producer of the constraint system's ψ motion could not serve it (gam#3171). It is a
    /// refusal of the DERIVATIVE, not of the value: the term itself priced.
    RowMotion { reason: String },
}

impl std::fmt::Display for ConeLaplaceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { what } => {
                write!(f, "constrained Laplace term: non-finite {what} (gam#2765)")
            }
            Self::Dimension { reason } => write!(f, "constrained Laplace term: {reason} (gam#2765)"),
            Self::NotPositiveDefinite { sweep } => write!(
                f,
                "constrained Laplace term: M + AᵀT̃A is not positive definite at sweep {sweep}; the \
                 mode is not a strict minimum on the cone at Laplace order (gam#2765)"
            ),
            Self::NotIntegrable { row, cavity_precision, cavity_shift, sweep } => write!(
                f,
                "constrained Laplace term: row {row}'s cavity has precision {cavity_precision:e} and \
                 shift {cavity_shift:e} at sweep {sweep}, so its half-line has no decay (gam#2765)"
            ),
            Self::Fold {
                row,
                variance,
                cavity_precision,
                cavity_shift,
                site_precision,
                posterior_variance,
                normal_curvature,
                condition_floor,
                sweep,
            } => write!(
                f,
                "constrained Laplace term: row {row}'s continued tilted variance is {variance:e} at \
                 sweep {sweep}, from a cavity with precision {cavity_precision:e} and shift \
                 {cavity_shift:e} against the continuation's boundary {:e}; that precision is \
                 1/Sigma_ii - tau_i with tau_i={site_precision:e}, Sigma_ii={posterior_variance:e} \
                 and tau_i*Sigma_ii={:.17e}, which sits {:.3e} from 1 against a band of at least \
                 {:.3e}; M's curvature along this row's normal is {}; the constrained mode is at a \
                 fold, outside the boundary Laplace regime (gam#2765, gam#3173, gam#4571)",
                -0.5 * cavity_shift * cavity_shift,
                site_precision * posterior_variance,
                (1.0 - site_precision * posterior_variance).abs(),
                condition_floor * f64::EPSILON,
                normal_curvature.map_or_else(
                    || {
                        "not the discriminator here: this is the derivative path, where the sites \
                         are at their fixed point and the cavity precision is what to read"
                            .to_string()
                    },
                    |curvature| format!("{curvature:.9e}")
                )
            ),
            Self::NotContracting { sweeps, fraction, step } => write!(
                f,
                "constrained Laplace term: EP stopped contracting after {sweeps} sweeps: a sweep damped \
                 to fraction {fraction:e} moved no site while its full update still moved one {step:e} \
                 times its rounding (gam#2765)"
            ),
            Self::Singular { reason } => write!(f, "constrained Laplace term: {reason} (gam#2765)"),
            Self::RowMotion { reason } => write!(
                f,
                "constrained Laplace term: the constraint rows' psi motion is unavailable:                  {reason} (gam#3171)"
            ),
        }
    }
}

impl From<ConeLaplaceRefusal> for String {
    fn from(refusal: ConeLaplaceRefusal) -> Self {
        refusal.to_string()
    }
}

/// The standardized slack beyond which a row's mass outside its half-space is below `f64::EPSILON`.
fn slack_horizon() -> Result<f64, ConeLaplaceRefusal> {
    standard_normal_quantile(f64::EPSILON)
        .map(|quantile| -quantile)
        .map_err(|_| ConeLaplaceRefusal::NonFinite { what: "slack horizon" })
}

/// `A⁻¹` by Gauss–Jordan elimination with partial pivoting, for the `2q × 2q` linearized fixed
/// point.
fn invert(mut a: Array2<f64>, what: &str) -> Result<Array2<f64>, ConeLaplaceRefusal> {
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
            return Err(ConeLaplaceRefusal::Singular {
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

fn symmetrized(matrix: &Array2<f64>) -> Array2<f64> {
    let n = matrix.nrows();
    Array2::from_shape_fn((n, n), |(i, j)| 0.5 * (matrix[[i, j]] + matrix[[j, i]]))
}

/// One site's moment-matching update from its cavity `(τ_c, ν_c)` over `u ≥ 0`, with the
/// partials of the new site in the cavity.
///
/// The tilted law is `e^{ν_c u − τ_c u²/2}` on the half-line, whose cumulants are those of the
/// half-line integral at `(μ, σ) = (−ν_c, τ_c)`. So `∂κ_k/∂ν_c = κ_{k+1}`,
/// `∂κ₁/∂τ_c = −½(κ₃ + 2κ₁κ₂)` and `∂κ₂/∂τ_c = −½(κ₄ + 2κ₂² + 2κ₁κ₃)`. The new site is
/// `τ̃ = 1/κ₂ − τ_c` and `ν̃ = κ₁/κ₂ − ν_c`. At `τ_c > 0` this is the covariance form's
/// `τ̃ = −τ_cψ(z)`, `ν̃ = √τ_c·χ(z)`.
struct SiteUpdate {
    tau: f64,
    nu: f64,
    /// `ln` of the tilted mass `∫₀^∞ e^{ν_c u − τ_c u²/2} du`.
    log_mass: f64,
    /// `[∂τ̃/∂τ_c, ∂τ̃/∂ν_c, ∂ν̃/∂τ_c, ∂ν̃/∂ν_c]`.
    jacobian: [f64; 4],
}

/// The tilted log mass of a cavity that pulls toward the boundary, and nothing else.
///
/// `share_and_magnitude` reads this and no other cumulant, so it must not inherit the
/// moment-matching condition [`site_update`] applies: the share is a VALUE, and `κ₂` enters only
/// through `τ̃ = 1/κ₂ − τ_c`, which the share never forms. Reading the mass through `site_update`
/// refused an evaluation on a quantity it does not use, and did it at sweep 0 — at the starting
/// sites, before EP has moved anything — where the module's own premise is that the start "only
/// selects where EP begins" and its fixed point does not depend on it (gam#4571).
fn tilted_log_mass(
    row: usize,
    tau_c: f64,
    nu_c: f64,
    sweep: usize,
) -> Result<f64, ConeLaplaceRefusal> {
    // The mass needs no cumulant beyond itself, so it takes no site operands either.
    let [log_mass, ..] =
        half_line_gaussian_log_jet(-nu_c, tau_c).ok_or(ConeLaplaceRefusal::NotIntegrable {
            row,
            cavity_precision: tau_c,
            cavity_shift: nu_c,
            sweep,
        })?;
    Ok(log_mass)
}

fn site_update(
    row: usize,
    tau_c: f64,
    nu_c: f64,
    site_precision: f64,
    posterior_variance: f64,
    condition_floor: f64,
    sweep: usize,
) -> Result<SiteUpdate, ConeLaplaceRefusal> {
    let [log_mass, k1, k2, k3, k4] = half_line_gaussian_log_jet(-nu_c, tau_c).ok_or(
        ConeLaplaceRefusal::NotIntegrable { row, cavity_precision: tau_c, cavity_shift: nu_c, sweep },
    )?;
    if !(k2 > 0.0) {
        return Err(ConeLaplaceRefusal::Fold {
            row,
            variance: k2,
            cavity_precision: tau_c,
            cavity_shift: nu_c,
            site_precision,
            posterior_variance,
            normal_curvature: None,
            condition_floor,
            sweep,
        });
    }
    let dk1_dtau = -0.5 * (k3 + 2.0 * k1 * k2);
    let dk2_dtau = -0.5 * (k4 + 2.0 * k2 * k2 + 2.0 * k1 * k3);
    let k2_squared = k2 * k2;
    Ok(SiteUpdate {
        tau: 1.0 / k2 - tau_c,
        nu: k1 / k2 - nu_c,
        log_mass,
        jacobian: [
            -dk2_dtau / k2_squared - 1.0,
            -k3 / k2_squared,
            dk1_dtau / k2 - k1 * dk2_dtau / k2_squared,
            -k1 * k3 / k2_squared,
        ],
    })
}

/// The Gaussian part at a set of sites: `Λ = M + AᵀT̃A` and what the posterior of `u` reads off
/// it.
struct GaussianPart {
    precision: Array2<f64>,
    inverse: Array2<f64>,
    log_det: f64,
    h: Array1<f64>,
    mean_offset: Array1<f64>,
    normal_solves: Array2<f64>,
    sigma: Array2<f64>,
    posterior_mean: Array1<f64>,
    /// `(max L_ii / min L_ii)²` of `Λ`'s Cholesky: a lower bound on `κ₂(Λ)`.
    condition_floor: f64,
}

/// `Λ = M + AᵀT̃A`, `Λ⁻¹`, `ln|Λ|` and `δ̄ = Λ⁻¹h` at a set of sites.
fn site_precision(
    precision_m: &Array2<f64>,
    gradient: &Array1<f64>,
    rows: &Array2<f64>,
    slack: &Array1<f64>,
    tau: &Array1<f64>,
    nu: &Array1<f64>,
    sweep: usize,
) -> Result<(Array2<f64>, Array2<f64>, f64, Array1<f64>, Array1<f64>, f64), ConeLaplaceRefusal> {
    let p = precision_m.nrows();
    let q = rows.nrows();
    let mut precision = precision_m.clone();
    for i in 0..q {
        if tau[i] == 0.0 {
            continue;
        }
        let row = rows.row(i);
        for a in 0..p {
            let scaled = tau[i] * row[a];
            for b in 0..p {
                precision[[a, b]] += scaled * row[b];
            }
        }
    }
    let precision = symmetrized(&precision);
    let factor = precision
        .cholesky(Side::Lower)
        .map_err(|_| ConeLaplaceRefusal::NotPositiveDefinite { sweep })?;
    // The Cholesky's pivots, from the walk `log_det` already makes, give a FREE lower bound on
    // `κ₂(Λ)`: `Λ_ii = Σ_{k≤i} L_ik² ≥ L_ii²` puts `max_i L_ii² ≤ λ_max`, and the i-th pivot
    // squared is the Schur complement `1/(Λ⁻¹)_ii` with `(Λ⁻¹)_ii ≤ 1/λ_min`, so
    // `min_i L_ii² ≥ λ_min`. Hence `(max L_ii / min L_ii)² ≤ κ₂(Λ)`. It is what the cavity's own
    // band is denominated in, and it costs nothing beyond the reduction below (gam#4571).
    let (mut log_det, mut widest, mut narrowest) = (0.0_f64, 0.0_f64, f64::INFINITY);
    for pivot in factor.diag().iter() {
        log_det += 2.0 * pivot.ln();
        widest = widest.max(*pivot);
        narrowest = narrowest.min(*pivot);
    }
    let condition_floor = if narrowest > 0.0 {
        let ratio = widest / narrowest;
        ratio * ratio
    } else {
        f64::INFINITY
    };
    let inverse = symmetrized(&factor.solve_mat(&Array2::<f64>::eye(p)));
    let shift = Array1::from_shape_fn(q, |i| nu[i] - tau[i] * slack[i]);
    let h = rows.t().dot(&shift) - gradient;
    let mean_offset = inverse.dot(&h);
    if !(log_det.is_finite() && mean_offset.iter().all(|value| value.is_finite())) {
        return Err(ConeLaplaceRefusal::NonFinite { what: "posterior of the sites" });
    }
    Ok((precision, inverse, log_det, h, mean_offset, condition_floor))
}

fn gaussian_part(
    precision_m: &Array2<f64>,
    gradient: &Array1<f64>,
    rows: &Array2<f64>,
    slack: &Array1<f64>,
    tau: &Array1<f64>,
    nu: &Array1<f64>,
    sweep: usize,
) -> Result<GaussianPart, ConeLaplaceRefusal> {
    let (precision, inverse, log_det, h, mean_offset, condition_floor) =
        site_precision(precision_m, gradient, rows, slack, tau, nu, sweep)?;
    let normal_solves = inverse.dot(&rows.t());
    let sigma = symmetrized(&rows.dot(&normal_solves));
    let posterior_mean = rows.dot(&mean_offset) + slack;
    if !sigma.iter().all(|value| value.is_finite()) {
        return Err(ConeLaplaceRefusal::NonFinite { what: "posterior of the sites" });
    }
    Ok(GaussianPart {
        precision,
        inverse,
        log_det,
        h,
        mean_offset,
        normal_solves,
        sigma,
        posterior_mean,
        condition_floor,
    })
}

/// `L`'s share beside `½ln|Λ|`, `−½hᵀδ̄ − Σ(ν̃d − ½τ̃d²) − Σ ln c_i`, with the summed magnitude of
/// its terms.
///
/// `ln c_i` is site `i`'s tilted log mass less the log mass of its posterior marginal Gaussian
/// `(τ_p, ν_p) = (τ_c + τ̃, ν_c + ν̃)`. A cavity that does not pull toward the boundary
/// (`ν_c ≥ 0`) has `τ_c > 0`. Its tilted mass is the cavity Gaussian's times `Φ(z)`, `z = ν_c/√τ_c`,
/// and the two Gaussians' `ν²/2τ` terms, large for a distant row, are differenced in the form
/// `(τ̃ν_c²/τ_c − 2ν_cν̃ − ν̃²)/(2τ_p)`, which does not cancel. A cavity that pulls toward the
/// boundary is read off the half-line jet, whose value stays of order one there.
fn share_and_magnitude(
    part: &GaussianPart,
    slack: &Array1<f64>,
    tau: &Array1<f64>,
    nu: &Array1<f64>,
    sweep: usize,
) -> Result<(f64, f64), ConeLaplaceRefusal> {
    let quadratic = -0.5 * part.h.dot(&part.mean_offset);
    let (mut total, mut magnitude) = (quadratic, quadratic.abs());
    for i in 0..slack.len() {
        let slack_term = -(nu[i] * slack[i] - 0.5 * tau[i] * slack[i] * slack[i]);
        total += slack_term;
        magnitude += slack_term.abs();
        let s_ii = part.sigma[[i, i]];
        let tau_p = 1.0 / s_ii;
        let nu_p = part.posterior_mean[i] / s_ii;
        let tau_c = tau_p - tau[i];
        let nu_c = nu_p - nu[i];
        let site = if nu_c >= 0.0 && tau_c > 0.0 {
            let z = nu_c / tau_c.sqrt();
            let log_cdf = gam_math::probability::normal_logcdf_derivatives(z)[0];
            let gaussian = 0.5 * (tau[i] / tau_c).ln_1p()
                + (tau[i] * nu_c * nu_c / tau_c - 2.0 * nu_c * nu[i] - nu[i] * nu[i]) / (2.0 * tau_p);
            magnitude += log_cdf.abs() + gaussian.abs();
            -(log_cdf + gaussian)
        } else {
            let tilted = tilted_log_mass(i, tau_c, nu_c, sweep)?;
            let gaussian = 0.5 * (2.0 * std::f64::consts::PI / tau_p).ln() + nu_p * nu_p / (2.0 * tau_p);
            magnitude += tilted.abs() + gaussian.abs();
            -(tilted - gaussian)
        };
        total += site;
    }
    if !total.is_finite() {
        return Err(ConeLaplaceRefusal::NonFinite { what: "constrained Laplace share" });
    }
    Ok((total, magnitude))
}

/// The motion of the constraint system `rows · β ≥ bounds` along one coordinate (or, in a
/// [`ConeLaplacePairMotion`], its second derivative along a pair), over the rows exactly as the
/// caller passed them to [`ConeLaplace::evaluate`]: row `i` of `rows` is `ȧ_i` and `bounds[i]` is
/// `ḃ_i`, at the caller's row scale.
#[derive(Clone, Debug)]
pub struct ConeRowMotion {
    pub rows: Array2<f64>,
    pub bounds: Array1<f64>,
}

/// One outer coordinate's first-order motion of the state the term reads: the mode response
/// `β̂̇`, the KKT gradient's total derivative `ġ`, the precision's motion `Ṁ` applied to
/// `δ̄` ([`ConeLaplace::mean_offset`]) and to each column of `R = Λ⁻¹Aᵀ`
/// ([`ConeLaplace::normal_solves`]), and the constraint system's motion where the coordinate
/// moves it (`None` where the rows and bounds do not depend on it).
#[derive(Clone, Debug)]
pub struct ConeLaplaceMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    pub precision_rate_on_mean: Array1<f64>,
    pub precision_rate_on_normals: Array2<f64>,
    pub constraint_rate: Option<ConeRowMotion>,
}

/// One coordinate pair's second-order motion: `β̈_kl`, `g̈_kl`, `M̈_kl δ̄`, and `(Ä_kl, b̈_kl)` where
/// the pair moves the constraint system.
#[derive(Clone, Debug)]
pub struct ConeLaplacePairMotion {
    pub mode_response: Array1<f64>,
    pub gradient_rate: Array1<f64>,
    pub precision_rate_on_mean: Array1<f64>,
    pub constraint_rate: Option<ConeRowMotion>,
}

/// A coordinate's row motion on the retained rows at unit scale: `Ȧ`, with the solves the first
/// and second orders contract it through, `Q = Λ⁻¹Ȧᵀ` (`p × q`) and `C = ȦR` (`q × q`).
#[derive(Clone, Debug)]
struct RetainedRowRate {
    rows: Array2<f64>,
    solved: Array2<f64>,
    coupling: Array2<f64>,
}

/// A coordinate's first derivative of `L` less the criterion's own trace `½tr(Λ⁻¹Ṁ)`, with the
/// rates its pairs reuse.
#[derive(Clone, Debug)]
pub struct ConeLaplaceFirstOrder {
    /// `tr(T̃ȦR) + δ̄ᵀġ + ½δ̄ᵀṀδ̄ − (ν̃ − T̃ū)ᵀe`.
    pub derivative: f64,
    mode_response: Array1<f64>,
    gradient_rate: Array1<f64>,
    precision_rate_on_mean: Array1<f64>,
    precision_rate_on_normals: Array2<f64>,
    row_rate: Option<RetainedRowRate>,
    /// `e = Aβ̂̇ + Ȧ(β̂ + δ̄) − ḃ`, the slack's motion with the posterior's offset carried by the rows.
    slack_rate: Array1<f64>,
    /// `δ̄̇` and `ū̇` at fixed sites.
    mean_offset_rate: Array1<f64>,
    posterior_mean_rate: Array1<f64>,
    /// `∂(dL/dθ)/∂τ̃` and `∂(dL/dθ)/∂ν̃`, the trace term included.
    tau_partial: Array1<f64>,
    nu_partial: Array1<f64>,
    /// The sites' motion `(τ̃̇, ν̃̇)` along this coordinate.
    tau_rate: Array1<f64>,
    nu_rate: Array1<f64>,
}

/// What the sites' motion reads at a fixed point that does not depend on the direction of the
/// motion: each site's update partials in its cavity and the inverse of `I − ∂F/∂s`.
#[derive(Clone, Debug)]
struct SiteMotionSystem {
    jacobians: Vec<[f64; 4]>,
    inverse: Array2<f64>,
}

/// The constrained Laplace term `L = ½ln|M| + C` at one inner mode, with the state its outer
/// derivatives contract.
#[derive(Clone, Debug)]
pub struct ConeLaplace {
    value: f64,
    log_det_half: f64,
    band: f64,
    /// Unit-scaled rows inside the horizon, `q × p`.
    rows: Array2<f64>,
    /// The caller's row each was read from, its norm there, and how many rows the caller passed.
    sources: Vec<usize>,
    norms: Vec<f64>,
    caller_rows: usize,
    /// `β̂`.
    mode: Array1<f64>,
    tau: Array1<f64>,
    nu: Array1<f64>,
    precision: Array2<f64>,
    inverse: Array2<f64>,
    mean_offset: Array1<f64>,
    normal_solves: Array2<f64>,
    sigma: Array2<f64>,
    posterior_mean: Array1<f64>,
    /// A lower bound on `κ₂(Λ)` at the returned sites, for the cavity band the derivative path's
    /// refusals are read against.
    condition_floor: f64,
    sweeps: usize,
    fraction: f64,
    site_motion_system: std::sync::OnceLock<Result<SiteMotionSystem, ConeLaplaceRefusal>>,
}

/// The distinct unit rows of `rows · β ≥ bounds`, one per half-space.
struct DistinctRows {
    units: Vec<Array1<f64>>,
    /// The slack of each at `β`, with its rounding band.
    slacks: Vec<f64>,
    slack_bands: Vec<f64>,
    /// The caller's row each was first read from, and that row's norm.
    sources: Vec<usize>,
    norms: Vec<f64>,
}

fn distinct_unit_rows(rows: &Array2<f64>, bounds: &Array1<f64>, beta: &Array1<f64>) -> DistinctRows {
    let mut units = Vec::new();
    let mut slacks = Vec::new();
    let mut slack_bands = Vec::new();
    let mut sources = Vec::new();
    let mut norms = Vec::new();
    let mut half_spaces = std::collections::HashSet::<Vec<u64>>::new();
    let p = beta.len();
    for row in 0..rows.nrows() {
        let norm = rows.row(row).dot(&rows.row(row)).sqrt();
        if !(norm > 0.0) {
            continue;
        }
        let unit = rows.row(row).mapv(|value| value / norm);
        let bound = bounds[row] / norm;
        // A row repeated exactly bounds the same half-space, and an intersection holds it once.
        let half_space: Vec<u64> =
            unit.iter().chain(std::iter::once(&bound)).map(|value| (value + 0.0).to_bits()).collect();
        if !half_spaces.insert(half_space) {
            continue;
        }
        let evaluated: f64 = unit.iter().zip(beta.iter()).map(|(a, b)| (a * b).abs()).sum();
        slacks.push(unit.dot(beta) - bound);
        slack_bands.push(accumulation_growth(p + 1) * (evaluated + bound.abs()));
        units.push(unit);
        sources.push(row);
        norms.push(norm);
    }
    DistinctRows { units, slacks, slack_bands, sources, norms }
}

/// What EP reads at one mode, over every distinct row.
#[derive(Clone, Copy)]
struct ModeInputs<'a> {
    precision: &'a Array2<f64>,
    gradient: &'a Array1<f64>,
    beta: &'a Array1<f64>,
    distinct: &'a DistinctRows,
    caller_rows: usize,
}

fn stack_rows(units: &[Array1<f64>], chosen: &[usize], p: usize) -> Array2<f64> {
    let mut stacked = Array2::<f64>::zeros((chosen.len(), p));
    for (i, &index) in chosen.iter().enumerate() {
        stacked.row_mut(i).assign(&units[index]);
    }
    stacked
}

/// The KKT multipliers of the rows at zero slack, `μ = (A₀A₀ᵀ)⁺A₀g`, the minimum-norm solution of
/// `A₀ᵀμ = g` over those rows (their normals may be dependent). The Gram's null directions are
/// those at its rounding band.
fn zero_slack_multipliers(zero_rows: &Array2<f64>, gradient: &Array1<f64>) -> Result<Array1<f64>, ConeLaplaceRefusal> {
    use gam_linalg::faer_ndarray::FaerEigh;
    let n = zero_rows.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    let gram = symmetrized(&zero_rows.dot(&zero_rows.t()));
    let (values, vectors) = gram
        .eigh(Side::Lower)
        .map_err(|error| ConeLaplaceRefusal::Singular { reason: format!("zero-slack Gram: {error:?}") })?;
    let spectrum = values.as_slice().ok_or(ConeLaplaceRefusal::Singular {
        reason: "zero-slack Gram spectrum is not contiguous".to_string(),
    })?;
    let band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(spectrum);
    let projected = vectors.t().dot(&zero_rows.dot(gradient));
    let scaled = Array1::from_shape_fn(n, |k| if values[k] > band { projected[k] / values[k] } else { 0.0 });
    Ok(vectors.dot(&scaled))
}

impl ConeLaplace {
    /// Evaluate `L` at a mode. `rows · β ≥ bounds` is the constraint system over the joint
    /// coefficients (any row scale), `gradient` is `∇F(β̂)`, and `precision` is the precision `M`
    /// the criterion prices, which may be indefinite along active normals.
    pub fn evaluate(
        rows: &Array2<f64>,
        bounds: &Array1<f64>,
        beta: &Array1<f64>,
        gradient: &Array1<f64>,
        precision: &Array2<f64>,
    ) -> Result<Self, ConeLaplaceRefusal> {
        let p = beta.len();
        if rows.ncols() != p || bounds.len() != rows.nrows() || gradient.len() != p || precision.dim() != (p, p)
        {
            return Err(ConeLaplaceRefusal::Dimension {
                reason: format!(
                    "rows {:?}, bounds {}, mode {p}, gradient {}, precision {:?}",
                    rows.dim(),
                    bounds.len(),
                    gradient.len(),
                    precision.dim()
                ),
            });
        }
        if rows.iter().chain(bounds.iter()).any(|value| !value.is_finite()) {
            return Err(ConeLaplaceRefusal::NonFinite { what: "constraint rows or bounds" });
        }
        if beta.iter().chain(gradient.iter()).any(|value| !value.is_finite()) {
            return Err(ConeLaplaceRefusal::NonFinite { what: "mode or gradient" });
        }
        if precision.iter().any(|value| !value.is_finite()) {
            return Err(ConeLaplaceRefusal::NonFinite { what: "precision" });
        }
        let horizon = slack_horizon()?;
        let distinct = distinct_unit_rows(rows, bounds, beta);
        let (units, slacks, slack_bands) = (&distinct.units, &distinct.slacks, &distinct.slack_bands);
        let total = units.len();
        // The boundary start: rows at zero slack with a positive multiplier.
        let zero: Vec<usize> = (0..total).filter(|&i| slacks[i] <= slack_bands[i]).collect();
        let multipliers = zero_slack_multipliers(&stack_rows(units, &zero, p), gradient)?;
        let mut start_tau = vec![0.0_f64; total];
        let mut start_nu = vec![0.0_f64; total];
        for (k, &i) in zero.iter().enumerate() {
            if multipliers[k] > 0.0 {
                start_tau[i] = multipliers[k] * multipliers[k];
                start_nu[i] = 2.0 * multipliers[k];
            }
        }
        let all_rows = stack_rows(units, &(0..total).collect::<Vec<_>>(), p);
        let all_slack = Array1::from(slacks.clone());
        let (_, start_inverse, _, _, start_offset, _) = site_precision(
            precision,
            gradient,
            &all_rows,
            &all_slack,
            &Array1::from(start_tau.clone()),
            &Array1::from(start_nu.clone()),
            0,
        )?;
        // A row's standardized posterior slack `ū_i/√(a_iᵀΛ⁻¹a_i)`.
        let standardized = |inverse: &Array2<f64>, offset: &Array1<f64>, i: usize| {
            let row = all_rows.row(i);
            (row.dot(offset) + slacks[i]) / row.dot(&inverse.dot(&row)).sqrt()
        };
        let mode = ModeInputs { precision, gradient, beta, distinct: &distinct, caller_rows: rows.nrows() };
        let mut chosen: Vec<usize> = (0..total)
            .filter(|&i| start_tau[i] > 0.0 || standardized(&start_inverse, &start_offset, i) < horizon)
            .collect();
        let mut tau: Vec<f64> = chosen.iter().map(|&i| start_tau[i]).collect();
        let mut nu: Vec<f64> = chosen.iter().map(|&i| start_nu[i]).collect();
        loop {
            // A fold verdict is about `M`'s curvature along ONE row's normal, and this is the
            // innermost frame holding both the precision and the caller's rows (`converge` is
            // handed the mode, not the rows). One matvec on the refusal path, nothing otherwise.
            //
            // Why this and not `M`'s smallest eigenvalue: at sweep 0 every site is `μ²` or zero,
            // so with `B_i = Λ − τ̃_ia_ia_iᵀ` the cavity precision is `1/Σ_ii − τ̃_i =
            // 1/(a_iᵀB_i⁻¹a_i)` exactly, and `B_i ⪰ M`. A NEGATIVE quotient therefore settles that
            // `B_i` is not positive definite, at one matvec, where a whole-matrix eigenvalue costs
            // a decomposition and still answers a different question — `M` can be indefinite and
            // positive along `a_i`. The evidence is one-sided on purpose: a non-negative quotient
            // does not give `B_i ≻ 0`, which needs `M` positive in every direction coupling to
            // `a_i` through `B_i⁻¹`. `M`'s inertia would close it in both directions and belongs
            // where the matrix is assembled and already factored, not here (gam#4571).
            let term = Self::converge(&mode, &chosen, &tau, &nu).map_err(|refusal| {
                let ConeLaplaceRefusal::Fold {
                    row,
                    variance,
                    cavity_precision,
                    cavity_shift,
                    site_precision,
                    posterior_variance,
                    condition_floor,
                    sweep,
                    ..
                } = refusal
                else {
                    return refusal;
                };
                let normal_curvature = chosen.get(row).map(|&unit| {
                    let normal = all_rows.row(unit);
                    normal.dot(&precision.dot(&normal)) / normal.dot(&normal)
                });
                ConeLaplaceRefusal::Fold {
                    row,
                    variance,
                    cavity_precision,
                    cavity_shift,
                    site_precision,
                    posterior_variance,
                    normal_curvature,
                    condition_floor,
                    sweep,
                }
            })?;
            let mut added = false;
            for i in 0..total {
                if !chosen.contains(&i) && standardized(&term.inverse, &term.mean_offset, i) < horizon {
                    chosen.push(i);
                    added = true;
                }
            }
            if !added {
                return Ok(term);
            }
            tau = term.tau.to_vec();
            nu = term.nu.to_vec();
            tau.resize(chosen.len(), 0.0);
            nu.resize(chosen.len(), 0.0);
        }
    }

    /// EP on the rows `chosen` of the mode's distinct rows from the sites `(tau, nu)`.
    fn converge(mode: &ModeInputs<'_>, chosen: &[usize], tau: &[f64], nu: &[f64]) -> Result<Self, ConeLaplaceRefusal> {
        let ModeInputs { precision: precision_m, gradient, beta, distinct, caller_rows } = *mode;
        let p = gradient.len();
        let q = chosen.len();
        let rows = stack_rows(&distinct.units, chosen, p);
        let slack = Array1::from_shape_fn(q, |i| distinct.slacks[chosen[i]]);
        let mut tau = Array1::from(tau.to_vec());
        let mut nu = Array1::from(nu.to_vec());
        // Every term of `L`, and every site's cavity, is formed in at most `p² + 4q² + 8q` rounded
        // operations: the Cholesky pivots and solves accumulate `p` products per entry over `p`
        // entries, and a sweep's rank-one updates `O(q)` per entry over `q` sites.
        let growth = accumulation_growth(p * p + 4 * q * q + 8 * q);
        let band_of = |magnitude: f64| growth * magnitude;
        let mut part = gaussian_part(precision_m, gradient, &rows, &slack, &tau, &nu, 0)?;
        let (share, _) = share_and_magnitude(&part, &slack, &tau, &nu, 0)?;
        let mut previous = 0.5 * part.log_det + share;
        let mut previous_step = f64::INFINITY;
        let mut fraction = 1.0_f64;
        let mut sweeps = 0;
        loop {
            sweeps += 1;
            // `L` is stationary in the sites, so its change alone would stop EP while the sites, and
            // so every derivative, are still off their fixed point by the square root of its band.
            // EP therefore settles on the sites: `step` is the largest move the full update makes
            // to a site, in units of the rounding of the site's own formation (a difference of the
            // tilted and cavity natural parameters).
            let (mut step, mut moved) = (0.0_f64, false);
            let relative = |delta: f64, rounding: f64| if delta == 0.0 { 0.0 } else { delta.abs() / rounding };
            // The sweep carries the posterior of `u` from site to site by the rank-one update a
            // site's move makes to `Λ`: with `c = Δτ̃/(1 + Δτ̃ Σ_jj)`, `Σ ← Σ − c Σ_{:j}Σ_{j:}` and
            // `ū ← ū + Σ_{:j}(Δν̃ − Δτ̃ ū_j)/(1 + Δτ̃ Σ_jj)`, and forms it exactly again from `Λ` at the
            // sweep's end.
            let mut sigma = part.sigma.clone();
            let mut mean = part.posterior_mean.clone();
            for j in 0..q {
                let s_jj = sigma[[j, j]];
                let tau_c = 1.0 / s_jj - tau[j];
                let nu_c = mean[j] / s_jj - nu[j];
                let update =
                    site_update(j, tau_c, nu_c, tau[j], s_jj, part.condition_floor, sweeps)?;
                let tau_rounding = growth * (tau_c.abs() + (tau_c + update.tau).abs());
                let nu_rounding = growth * (nu_c.abs() + (nu_c + update.nu).abs());
                step = step
                    .max(relative(update.tau - tau[j], tau_rounding))
                    .max(relative(update.nu - nu[j], nu_rounding));
                let new_tau = (1.0 - fraction) * tau[j] + fraction * update.tau;
                let new_nu = (1.0 - fraction) * nu[j] + fraction * update.nu;
                moved |= new_tau != tau[j] || new_nu != nu[j];
                let (d_tau, d_nu) = (new_tau - tau[j], new_nu - nu[j]);
                tau[j] = new_tau;
                nu[j] = new_nu;
                let denominator = 1.0 + d_tau * s_jj;
                if !(denominator > 0.0) {
                    return Err(ConeLaplaceRefusal::NotPositiveDefinite { sweep: sweeps });
                }
                let column = sigma.column(j).to_owned();
                let mean_step = (d_nu - d_tau * mean[j]) / denominator;
                mean.scaled_add(mean_step, &column);
                let shrink = d_tau / denominator;
                for a in 0..q {
                    let scaled = shrink * column[a];
                    for b in 0..q {
                        sigma[[a, b]] -= scaled * column[b];
                    }
                }
            }
            part = gaussian_part(precision_m, gradient, &rows, &slack, &tau, &nu, sweeps)?;
            if !step.is_finite() {
                return Err(ConeLaplaceRefusal::NonFinite { what: "EP site update" });
            }
            let (share, share_magnitude) = share_and_magnitude(&part, &slack, &tau, &nu, sweeps)?;
            let value = 0.5 * part.log_det + share;
            let band = band_of(share_magnitude + 0.5 * part.log_det.abs());
            let change = (value - previous).abs();
            // Settled: no site moves beyond its rounding. Or at the floor the sites' own rounding
            // sets: `L` no longer moves beyond its band (a damped sweep moves it by about `fraction`
            // of the full update) and a full sweep no longer shrinks the sites' move, so no further
            // sweep can bring them closer to the fixed point.
            let at_floor = change <= fraction * band && !(step < previous_step);
            if step <= 1.0 || at_floor {
                return Ok(Self {
                    value,
                    log_det_half: 0.5 * part.log_det,
                    band,
                    rows,
                    sources: chosen.iter().map(|&i| distinct.sources[i]).collect(),
                    norms: chosen.iter().map(|&i| distinct.norms[i]).collect(),
                    caller_rows,
                    mode: beta.clone(),
                    tau,
                    nu,
                    precision: part.precision,
                    inverse: part.inverse,
                    mean_offset: part.mean_offset,
                    normal_solves: part.normal_solves,
                    sigma: part.sigma,
                    posterior_mean: part.posterior_mean,
                    condition_floor: part.condition_floor,
                    sweeps,
                    fraction,
                    site_motion_system: std::sync::OnceLock::new(),
                });
            }
            if !moved {
                return Err(ConeLaplaceRefusal::NotContracting { sweeps, fraction, step });
            }
            // The change need not fall monotonically: on strongly correlated rows it can grow for
            // one sweep and then contract geometrically. A sweep whose full update did not shrink
            // halves the share of it every later site takes, which leaves the fixed points unchanged.
            if !(step < previous_step) {
                fraction *= 0.5;
            }
            previous_step = step;
            previous = value;
        }
    }

    /// `L = ½ln|M| + C`.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// `½ln|Λ|`, the log-determinant a criterion prices on [`Self::laplace_precision`].
    pub fn log_det_half(&self) -> f64 {
        self.log_det_half
    }

    /// `L − ½ln|Λ|`.
    pub fn share(&self) -> f64 {
        self.value - self.log_det_half
    }

    /// The rounding band `L` settled to.
    pub fn value_band(&self) -> f64 {
        self.band
    }

    /// `Λ = M + AᵀT̃A` at the fixed point.
    pub fn laplace_precision(&self) -> &Array2<f64> {
        &self.precision
    }

    /// `Λ⁻¹`.
    pub fn laplace_precision_inverse(&self) -> &Array2<f64> {
        &self.inverse
    }

    /// `δ̄ = Λ⁻¹h`.
    pub fn mean_offset(&self) -> &Array1<f64> {
        &self.mean_offset
    }

    /// `R = Λ⁻¹Aᵀ`, `p × q`.
    pub fn normal_solves(&self) -> &Array2<f64> {
        &self.normal_solves
    }

    /// Rows inside the horizon.
    pub fn retained_rows(&self) -> usize {
        self.rows.nrows()
    }

    /// EP sweeps at this mode, in the last round of the row selection.
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }

    /// The share of its full update each site took on the last sweep.
    pub fn ep_step_fraction(&self) -> f64 {
        self.fraction
    }

    /// `Ȧ` and `ḃ` on the retained rows at unit scale, from the caller's rates at its own row scale.
    fn retained_rate(&self, motion: &ConeRowMotion) -> Result<(Array2<f64>, Array1<f64>), ConeLaplaceRefusal> {
        let p = self.mode.len();
        if motion.rows.dim() != (self.caller_rows, p) || motion.bounds.len() != self.caller_rows {
            return Err(ConeLaplaceRefusal::Dimension {
                reason: format!(
                    "row motion {:?} and bound motion {} against {} rows over {p} coefficients",
                    motion.rows.dim(),
                    motion.bounds.len(),
                    self.caller_rows
                ),
            });
        }
        if motion.rows.iter().chain(motion.bounds.iter()).any(|value| !value.is_finite()) {
            return Err(ConeLaplaceRefusal::NonFinite { what: "constraint row or bound motion" });
        }
        let q = self.rows.nrows();
        let rows = Array2::from_shape_fn((q, p), |(i, a)| motion.rows[[self.sources[i], a]] / self.norms[i]);
        let bounds = Array1::from_shape_fn(q, |i| motion.bounds[self.sources[i]] / self.norms[i]);
        Ok((rows, bounds))
    }

    /// `Σ_i τ̃_i x_iᵀy_i` over the columns of two `p × q` matrices.
    fn site_weighted_column_dot(&self, x: &Array2<f64>, y: &Array2<f64>) -> f64 {
        (0..self.rows.nrows()).map(|i| self.tau[i] * x.column(i).dot(&y.column(i))).sum()
    }

    /// First derivative of `L` less the criterion's trace `½tr(Λ⁻¹Ṁ)` along one coordinate, with
    /// the rates its pairs reuse.
    pub fn first_order(&self, motion: &ConeLaplaceMotion) -> Result<ConeLaplaceFirstOrder, ConeLaplaceRefusal> {
        let q = self.rows.nrows();
        let site_residual = &self.nu - &(&self.tau * &self.posterior_mean);
        let mut slack_rate = self.rows.dot(&motion.mode_response);
        let row_rate = match &motion.constraint_rate {
            None => None,
            Some(constraint_rate) => {
                let (rows, bounds) = self.retained_rate(constraint_rate)?;
                slack_rate += &(rows.dot(&(&self.mode + &self.mean_offset)) - &bounds);
                let solved = self.inverse.dot(&rows.t());
                let coupling = rows.dot(&self.normal_solves);
                Some(RetainedRowRate { rows, solved, coupling })
            }
        };
        let tau_slack_rate = &self.tau * &slack_rate;
        // `δ̄̇ = Λ⁻¹(ḣ − Λ̇δ̄) = Λ⁻¹(−ġ − Ṁδ̄ − AᵀT̃e + Ȧᵀ(ν̃ − T̃ū))` at fixed sites.
        let mut rhs = -&motion.gradient_rate - &self.rows.t().dot(&tau_slack_rate) - &motion.precision_rate_on_mean;
        if let Some(rate) = &row_rate {
            rhs += &rate.rows.t().dot(&site_residual);
        }
        let mean_offset_rate = self.inverse.dot(&rhs);
        let posterior_mean_rate = self.rows.dot(&mean_offset_rate) + &slack_rate;
        let mut derivative = self.mean_offset.dot(&motion.gradient_rate)
            + 0.5 * self.mean_offset.dot(&motion.precision_rate_on_mean)
            - site_residual.dot(&slack_rate);
        // The diagonal of the fixed-site `dΣ_u = ȦR + RᵀȦᵀ − RᵀΛ̇R`: `−r_jᵀṀr_j`, and where the rows
        // move `2C_jj − 2Σ_i τ̃_iΣ_ijC_ij`. Half of it is the trace term's rate in `τ̃_j`.
        let mut sigma_rate = Array1::from_shape_fn(q, |j| {
            -self.normal_solves.column(j).dot(&motion.precision_rate_on_normals.column(j))
        });
        let mut nu_partial = self.normal_solves.t().dot(&motion.gradient_rate)
            + self.normal_solves.t().dot(&motion.precision_rate_on_mean)
            - &slack_rate
            + self.sigma.dot(&tau_slack_rate);
        if let Some(rate) = &row_rate {
            let coupling = &rate.coupling;
            derivative += (0..q).map(|i| self.tau[i] * coupling[[i, i]]).sum::<f64>();
            for j in 0..q {
                let weighted: f64 = (0..q).map(|i| self.tau[i] * self.sigma[[i, j]] * coupling[[i, j]]).sum();
                sigma_rate[j] += 2.0 * (coupling[[j, j]] - weighted);
            }
            nu_partial -= &coupling.t().dot(&site_residual);
        }
        let tau_partial = 0.5 * &sigma_rate - &(&self.posterior_mean * &nu_partial);
        let (tau_rate, nu_rate) = self.site_motion(&sigma_rate, &posterior_mean_rate)?;
        if !derivative.is_finite() {
            return Err(ConeLaplaceRefusal::NonFinite { what: "first-order share" });
        }
        Ok(ConeLaplaceFirstOrder {
            derivative,
            mode_response: motion.mode_response.clone(),
            gradient_rate: motion.gradient_rate.clone(),
            precision_rate_on_mean: motion.precision_rate_on_mean.clone(),
            precision_rate_on_normals: motion.precision_rate_on_normals.clone(),
            row_rate,
            slack_rate,
            mean_offset_rate,
            posterior_mean_rate,
            tau_partial,
            nu_partial,
            tau_rate,
            nu_rate,
        })
    }

    /// Everything in `d²L/dθ_kdθ_l` except the criterion's fixed-site log-determinant Hessian
    /// `½tr(Λ⁻¹M̈_kl) − ½tr(Λ⁻¹Ṁ_lΛ⁻¹Ṁ_k)`: the rest of the fixed-site second derivative, the row
    /// motion's share of `½tr(Λ⁻¹Λ̈) − ½tr(Λ⁻¹Λ̇_lΛ⁻¹Λ̇_k)` included, and the sites' motion through
    /// the whole first derivative, the trace term included.
    pub fn second_order(
        &self,
        first_k: &ConeLaplaceFirstOrder,
        first_l: &ConeLaplaceFirstOrder,
        pair: &ConeLaplacePairMotion,
    ) -> Result<f64, ConeLaplaceRefusal> {
        let q = self.rows.nrows();
        let site_residual = &self.nu - &(&self.tau * &self.posterior_mean);
        // `ė_kl = Aβ̈ + Ä(β̂ + δ̄) − b̈ + Ȧ_lβ̂̇_k + Ȧ_k(β̂̇_l + δ̄̇_l)`.
        let mut slack_pair_rate = self.rows.dot(&pair.mode_response);
        let mut trace = 0.0;
        if let Some(constraint_rate) = &pair.constraint_rate {
            let (rows, bounds) = self.retained_rate(constraint_rate)?;
            slack_pair_rate += &(rows.dot(&(&self.mode + &self.mean_offset)) - &bounds);
            // `½tr(Λ⁻¹(ÄᵀT̃A + AᵀT̃Ä)) = Σ_i τ̃_i ä_iᵀr_i`.
            trace += (0..q).map(|i| self.tau[i] * rows.row(i).dot(&self.normal_solves.column(i))).sum::<f64>();
        }
        if let Some(rate_l) = &first_l.row_rate {
            slack_pair_rate += &rate_l.rows.dot(&first_k.mode_response);
            // `−½tr(Λ⁻¹(Ȧ_lᵀT̃A + AᵀT̃Ȧ_l)Λ⁻¹Ṁ_k) = −Σ_i τ̃_i (Ṁ_kr_i)ᵀΛ⁻¹ȧ_{l,i}`.
            trace -= self.site_weighted_column_dot(&first_k.precision_rate_on_normals, &rate_l.solved);
        }
        if let Some(rate_k) = &first_k.row_rate {
            slack_pair_rate += &rate_k.rows.dot(&(&first_l.mode_response + &first_l.mean_offset_rate));
            // The rate of `tr(T̃Ȧ_kR)` through `Ṙ = −Λ⁻¹Ṁ_lR` along `l`'s precision drift.
            trace -= self.site_weighted_column_dot(&rate_k.solved, &first_l.precision_rate_on_normals);
        }
        if let (Some(rate_k), Some(rate_l)) = (&first_k.row_rate, &first_l.row_rate) {
            // The rest of the rate of `tr(T̃Ȧ_kR)`, `Ṙ = Λ⁻¹Ȧ_lᵀ − Λ⁻¹(Ȧ_lᵀT̃A + AᵀT̃Ȧ_l)R`:
            // `Σ_i τ̃_i (B − BT̃Σ_u − C_kT̃C_l)_ii` with `B = Ȧ_kΛ⁻¹Ȧ_lᵀ`.
            let cross = rate_k.rows.dot(&rate_l.solved);
            for i in 0..q {
                let (mut through_sigma, mut through_coupling) = (0.0, 0.0);
                for m in 0..q {
                    through_sigma += cross[[i, m]] * self.tau[m] * self.sigma[[m, i]];
                    through_coupling += rate_k.coupling[[i, m]] * self.tau[m] * rate_l.coupling[[m, i]];
                }
                trace += self.tau[i] * (cross[[i, i]] - through_sigma - through_coupling);
            }
        }
        let fixed_sites = trace
            + first_l.mean_offset_rate.dot(&first_k.gradient_rate)
            + self.mean_offset.dot(&pair.gradient_rate)
            + first_l.mean_offset_rate.dot(&first_k.precision_rate_on_mean)
            + 0.5 * self.mean_offset.dot(&pair.precision_rate_on_mean)
            + (&self.tau * &first_l.posterior_mean_rate).dot(&first_k.slack_rate)
            - site_residual.dot(&slack_pair_rate);
        let sites = first_k.tau_partial.dot(&first_l.tau_rate) + first_k.nu_partial.dot(&first_l.nu_rate);
        let second = fixed_sites + sites;
        if !second.is_finite() {
            return Err(ConeLaplaceRefusal::NonFinite { what: "second-order share" });
        }
        Ok(second)
    }

    /// The sites' motion `(τ̃̇, ν̃̇)` along a direction that moves the posterior of `u` by
    /// `dΣ_jj = sigma_rate[j]` and `dū = mean_rate` at fixed sites, from `(I − ∂F/∂s) ds = ∂F/∂θ dθ`.
    fn site_motion(
        &self,
        sigma_rate: &Array1<f64>,
        mean_rate: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ConeLaplaceRefusal> {
        let q = self.rows.nrows();
        if q == 0 {
            return Ok((Array1::zeros(0), Array1::zeros(0)));
        }
        let system = self.linearized_fixed_point()?;
        let mut rhs = Array1::<f64>::zeros(2 * q);
        for j in 0..q {
            let s_jj = self.sigma[[j, j]];
            let s2 = s_jj * s_jj;
            let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = system.jacobians[j];
            let dtc = -sigma_rate[j] / s2;
            let dnc = mean_rate[j] / s_jj - self.posterior_mean[j] * sigma_rate[j] / s2;
            rhs[j] = dt_dtc * dtc + dt_dnc * dnc;
            rhs[q + j] = dn_dtc * dtc + dn_dnc * dnc;
        }
        let ds = system.inverse.dot(&rhs);
        Ok((ds.slice(s![0..q]).to_owned(), ds.slice(s![q..2 * q]).to_owned()))
    }

    /// The site partials and the inverse of the linearized fixed point's `2q × 2q` system
    /// `I − ∂F/∂s`, formed once for these sites. Per unit site change the posterior of `u` moves by
    /// `dΣ/dτ̃_k = −Σ_{:k}Σ_{k:}`, `dū/dτ̃_k = −Σ_{:k}ū_k` and `dū/dν̃_k = Σ_{:k}`, and site `j` reads
    /// the cavity `(1/Σ_jj − τ̃_j, ū_j/Σ_jj − ν̃_j)`.
    fn linearized_fixed_point(&self) -> Result<&SiteMotionSystem, ConeLaplaceRefusal> {
        self.site_motion_system
            .get_or_init(|| {
                let q = self.rows.nrows();
                let (sigma, mean) = (&self.sigma, &self.posterior_mean);
                let mut system = Array2::<f64>::eye(2 * q);
                let mut jacobians = Vec::with_capacity(q);
                for j in 0..q {
                    let s_jj = sigma[[j, j]];
                    let tau_c = 1.0 / s_jj - self.tau[j];
                    let nu_c = mean[j] / s_jj - self.nu[j];
                    let jacobian = site_update(
                        j,
                        tau_c,
                        nu_c,
                        self.tau[j],
                        s_jj,
                        self.condition_floor,
                        self.sweeps,
                    )?
                    .jacobian;
                    let [dt_dtc, dt_dnc, dn_dtc, dn_dnc] = jacobian;
                    jacobians.push(jacobian);
                    let s2 = s_jj * s_jj;
                    let cavity_rate =
                        |d_sjj: f64, d_mean: f64| -> (f64, f64) { (-d_sjj / s2, d_mean / s_jj - mean[j] * d_sjj / s2) };
                    for k in 0..q {
                        let (dtc, dnc) = cavity_rate(-sigma[[j, k]] * sigma[[j, k]], -sigma[[j, k]] * mean[k]);
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
                Ok(SiteMotionSystem { jacobians, inverse })
            })
            .as_ref()
            .map_err(|refusal| refusal.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained_posterior::ConeNormalizer;
    use ndarray::array;

    fn log_det_spd(m: &Array2<f64>) -> f64 {
        let factor = m.cholesky(Side::Lower).expect("the test precision is positive definite");
        2.0 * factor.diag().iter().map(|pivot| pivot.ln()).sum::<f64>()
    }

    /// `½ln|M| + C` through the covariance form, [`ConeNormalizer`], at a positive definite `M`, with
    /// the rounding band of its terms: `−½gᵀM⁻¹g` and `−ln P` cancel in it.
    fn covariance_form(
        rows: &Array2<f64>,
        bounds: &Array1<f64>,
        beta: &Array1<f64>,
        gradient: &Array1<f64>,
        m: &Array2<f64>,
    ) -> (f64, f64) {
        let inverse = invert(m.clone(), "test precision").expect("an invertible test precision");
        let solve = |rhs: &Array1<f64>| inverse.dot(rhs);
        let normalizer = ConeNormalizer::evaluate(rows, bounds, beta, gradient, &solve)
            .unwrap_or_else(|refusal| panic!("the covariance form converges: {refusal}"));
        let half_log_det = 0.5 * log_det_spd(m);
        let (p, q) = (m.nrows(), rows.nrows());
        let magnitude = half_log_det.abs()
            + normalizer.value().abs()
            + 0.5 * gradient.dot(&inverse.dot(gradient)).abs()
            + normalizer.log_mass().abs();
        (half_log_det + normalizer.value(), accumulation_growth(p * p + 4 * q * q + 8 * q) * magnitude)
    }

    /// Central differences of `f` at `h` and `h/2`, Richardson-combined, with an error bar: the gap
    /// between the two levels plus the noise `noise` of each evaluation, which a difference quotient
    /// amplifies by `1/h`.
    fn richardson(f: &dyn Fn(f64) -> (f64, f64), h: f64) -> (f64, f64) {
        let (plus, plus_noise) = f(h);
        let (minus, minus_noise) = f(-h);
        let (half_plus, half_plus_noise) = f(0.5 * h);
        let (half_minus, half_minus_noise) = f(-0.5 * h);
        let coarse = (plus - minus) / (2.0 * h);
        let fine = (half_plus - half_minus) / h;
        let noise = (plus_noise + minus_noise) / (2.0 * h) + (half_plus_noise + half_minus_noise) / h;
        ((4.0 * fine - coarse) / 3.0, (fine - coarse).abs() + 4.0 * noise)
    }

    /// Where `M` is positive definite the term is the covariance form's `½ln|M| + C`: the same EP
    /// fixed point read through `Λ` instead of `W`. Three rows (two active with correlated normals,
    /// one inactive), a fourth far beyond the horizon, which does not enter.
    #[test]
    fn the_term_is_the_covariance_form_where_the_precision_is_positive_definite_2765() {
        let m = array![
            [2.0, 0.3, -0.2, 0.1],
            [0.3, 1.5, 0.4, 0.0],
            [-0.2, 0.4, 1.2, 0.3],
            [0.1, 0.0, 0.3, 0.9]
        ];
        let rows = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.0, 0.3, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let beta = array![0.5, -0.2, 0.4, 0.1];
        let bounds = rows.dot(&beta) - array![0.0, 0.0, 0.35, 1.0e3];
        let gradient = rows.t().dot(&array![1.3, 0.7, 0.0, 0.0]);
        let term = ConeLaplace::evaluate(&rows, &bounds, &beta, &gradient, &m)
            .unwrap_or_else(|refusal| panic!("the term converges: {refusal}"));
        let (covariance, covariance_band) = covariance_form(&rows, &bounds, &beta, &gradient, &m);
        let bar = 8.0 * (term.value_band() + covariance_band);
        eprintln!(
            "[2765-CL] L {:.15e} covariance {covariance:.15e} gap {:e} bar {bar:e} rows {} sweeps {}",
            term.value(),
            (term.value() - covariance).abs(),
            term.retained_rows(),
            term.sweeps()
        );
        assert_eq!(term.retained_rows(), 3, "the row 1e3 beyond its bound does not enter");
        assert!(
            (term.value() - covariance).abs() <= bar,
            "L {} against the covariance form {covariance}: gap {:e} above {bar:e}",
            term.value(),
            (term.value() - covariance).abs()
        );
    }

    /// One active row with multiplier `μ`, normal curvature `σ` given the face, and a unit face
    /// curvature: `L = ½ln 2π − ln J(μ, σ)` exactly, since EP is exact for one row. That holds on
    /// both sides of `σ = 0` and at it, where the covariance form's `W = 1/σ` is infinite and
    /// the kept spectrum drops the direction for `σ < 0`, losing `ln μ + ½ln 2π`.
    #[test]
    fn one_active_row_is_smooth_through_zero_normal_curvature_2765() {
        let (mu, coupling) = (3.0, 0.3);
        let rows = array![[1.0, 0.0]];
        let bounds = array![0.0];
        let beta = array![0.0, 0.0];
        let gradient = array![mu, 0.0];
        for schur in [0.4, 1e-3, 1e-7, 0.0, -1e-7, -1e-3, -0.05] {
            let m = array![[coupling * coupling + schur, coupling], [coupling, 1.0]];
            let term = ConeLaplace::evaluate(&rows, &bounds, &beta, &gradient, &m)
                .unwrap_or_else(|refusal| panic!("σ = {schur}: {refusal}"));
            let jet = half_line_gaussian_log_jet(mu, schur).expect("an active row's half-line mass");
            let exact = 0.5 * (2.0 * std::f64::consts::PI).ln() - jet[0];
            let bar = term.value_band() + accumulation_growth(64) * exact.abs();
            assert!(
                (term.value() - exact).abs() <= bar,
                "σ = {schur}: L {} against ½ln2π − ln J = {exact}, gap {:e} above {bar:e}",
                term.value(),
                (term.value() - exact).abs()
            );
        }
    }

    /// A planted two-coordinate family: `M(θ)`, `g(θ)` and `β̂(θ)` bilinear in `θ`, the mode moving
    /// along the active face. Where the rows move, `A(θ)` is bilinear too and the bounds follow it,
    /// `b(θ) = A(θ)β̂(θ) − s`, so every row keeps its slack `s` at the caller's scale.
    struct Family {
        a: [Array2<f64>; 4],
        bounds: Array1<f64>,
        slack: Array1<f64>,
        moving: bool,
        m: [Array2<f64>; 4],
        g: [Array1<f64>; 4],
        v: [Array1<f64>; 4],
    }

    impl Family {
        /// Two active rows with correlated normals and one inactive row; `corner` sets `M₁₁`, and
        /// below about `0.0201` the precision is indefinite along the active normals.
        fn new(corner: f64) -> Self {
            let rows = array![[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.2, 1.0]];
            let beta = array![0.4, -0.3, 0.2];
            let slack = array![0.0, 0.0, 0.3];
            let bounds = rows.dot(&beta) - &slack;
            let m0 = array![[corner, 0.0, 0.2], [0.0, 1.0, 0.1], [0.2, 0.1, 2.0]];
            let m1 = array![[0.3, 0.05, 0.0], [0.05, -0.2, 0.1], [0.0, 0.1, 0.4]];
            let m2 = array![[-0.1, 0.0, 0.05], [0.0, 0.3, -0.05], [0.05, -0.05, 0.2]];
            let m12 = array![[0.05, 0.02, 0.0], [0.02, 0.1, 0.0], [0.0, 0.0, -0.1]];
            let g0 = rows.t().dot(&array![2.5, 1.5, 0.0]);
            let zero = Array2::<f64>::zeros((3, 3));
            Self {
                a: [rows, zero.clone(), zero.clone(), zero],
                bounds,
                slack,
                moving: false,
                m: [m0, m1, m2, m12],
                g: [g0, array![0.2, -0.1, 0.05], array![-0.1, 0.3, 0.0], array![0.05, 0.05, -0.02]],
                v: [beta, array![0.0, 0.0, 0.3], array![0.0, 0.0, -0.2], array![0.0, 0.0, 0.1]],
            }
        }

        /// The same family with rows that move in every entry, held at scales `2`, `½` and `3`.
        fn moving(corner: f64) -> Self {
            let mut family = Self::new(corner);
            let scale = array![[2.0], [0.5], [3.0]];
            family.a = [
                &family.a[0] * &scale,
                array![[0.1, 0.2, -0.3], [0.05, 0.0, 0.2], [0.1, -0.2, 0.3]],
                array![[-0.2, 0.1, 0.1], [0.1, 0.1, -0.1], [0.2, 0.0, -0.1]],
                array![[0.05, -0.05, 0.1], [0.0, 0.05, 0.05], [-0.1, 0.05, 0.0]],
            ];
            family.slack = &family.slack * &scale.column(0);
            family.moving = true;
            family
        }

        fn at<T: Clone + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T>>(
            parts: &[T; 4],
            theta: [f64; 2],
        ) -> T {
            parts[0].clone() + parts[1].clone() * theta[0] + parts[2].clone() * theta[1]
                + parts[3].clone() * (theta[0] * theta[1])
        }

        /// `∂X/∂θ_k` of a bilinear `X` at `θ`.
        fn rate<T: Clone + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T>>(
            parts: &[T; 4],
            theta: [f64; 2],
            k: usize,
        ) -> T {
            parts[k + 1].clone() + parts[3].clone() * theta[1 - k]
        }

        /// `∂²X/∂θ_k∂θ_l` of a bilinear `X`.
        fn pair_rate<T: Clone + std::ops::Mul<f64, Output = T>>(parts: &[T; 4], k: usize, l: usize) -> T {
            parts[3].clone() * if k == l { 0.0 } else { 1.0 }
        }

        fn bounds_at(&self, theta: [f64; 2]) -> Array1<f64> {
            if self.moving {
                Self::at(&self.a, theta).dot(&Self::at(&self.v, theta)) - &self.slack
            } else {
                self.bounds.clone()
            }
        }

        fn term(&self, theta: [f64; 2]) -> ConeLaplace {
            ConeLaplace::evaluate(
                &Self::at(&self.a, theta),
                &self.bounds_at(theta),
                &Self::at(&self.v, theta),
                &Self::at(&self.g, theta),
                &Self::at(&self.m, theta),
            )
            .unwrap_or_else(|refusal| panic!("θ = {theta:?}: {refusal}"))
        }

        /// `(Ȧ_k, ḃ_k)` at `θ`, `ḃ_k = Ȧ_kβ̂ + Aβ̂̇_k`, where the rows move.
        fn row_motion(&self, theta: [f64; 2], k: usize) -> Option<ConeRowMotion> {
            self.moving.then(|| {
                let rows = Self::rate(&self.a, theta, k);
                let bounds = rows.dot(&Self::at(&self.v, theta))
                    + Self::at(&self.a, theta).dot(&Self::rate(&self.v, theta, k));
                ConeRowMotion { rows, bounds }
            })
        }

        /// `(Ä_kl, b̈_kl)` at `θ = 0`, `b̈ = Äβ̂ + Ȧ_kβ̂̇_l + Ȧ_lβ̂̇_k + Aβ̂̈`, where the rows move.
        fn row_pair_motion(&self, k: usize, l: usize) -> Option<ConeRowMotion> {
            let theta = [0.0, 0.0];
            self.moving.then(|| {
                let rows = Self::pair_rate(&self.a, k, l);
                let bounds = rows.dot(&self.v[0])
                    + Self::rate(&self.a, theta, k).dot(&Self::rate(&self.v, theta, l))
                    + Self::rate(&self.a, theta, l).dot(&Self::rate(&self.v, theta, k))
                    + self.a[0].dot(&Self::pair_rate(&self.v, k, l));
                ConeRowMotion { rows, bounds }
            })
        }

        fn motion(&self, term: &ConeLaplace, theta: [f64; 2], k: usize) -> (ConeLaplaceMotion, Array2<f64>) {
            let m_rate = Self::rate(&self.m, theta, k);
            let motion = ConeLaplaceMotion {
                mode_response: Self::rate(&self.v, theta, k),
                gradient_rate: Self::rate(&self.g, theta, k),
                precision_rate_on_mean: m_rate.dot(term.mean_offset()),
                precision_rate_on_normals: m_rate.dot(term.normal_solves()),
                constraint_rate: self.row_motion(theta, k),
            };
            (motion, m_rate)
        }

        /// `dL/dθ_k = ½tr(Λ⁻¹Ṁ_k) + dshare/dθ_k`, with its rounding band.
        fn first_derivative(&self, theta: [f64; 2], k: usize) -> (f64, f64) {
            let term = self.term(theta);
            let (motion, m_rate) = self.motion(&term, theta, k);
            let first = term.first_order(&motion).unwrap_or_else(|refusal| panic!("first order: {refusal}"));
            let trace = 0.5 * (term.laplace_precision_inverse() * &m_rate).sum();
            (trace + first.derivative, term.value_band())
        }

        /// `d²L/dθ_kdθ_l` at `θ = 0`: the fixed-site log-determinant Hessian on `Λ` plus
        /// [`ConeLaplace::second_order`].
        fn second_derivative(&self, k: usize, l: usize) -> f64 {
            let theta = [0.0, 0.0];
            let term = self.term(theta);
            let (motion_k, m_k) = self.motion(&term, theta, k);
            let (motion_l, m_l) = self.motion(&term, theta, l);
            let first_k = term.first_order(&motion_k).unwrap_or_else(|refusal| panic!("first order: {refusal}"));
            let first_l = term.first_order(&motion_l).unwrap_or_else(|refusal| panic!("first order: {refusal}"));
            let m_pair = Self::pair_rate(&self.m, k, l);
            let pair = ConeLaplacePairMotion {
                mode_response: Self::pair_rate(&self.v, k, l),
                gradient_rate: Self::pair_rate(&self.g, k, l),
                precision_rate_on_mean: m_pair.dot(term.mean_offset()),
                constraint_rate: self.row_pair_motion(k, l),
            };
            let inverse = term.laplace_precision_inverse();
            let solved_l = inverse.dot(&m_l);
            let solved_k = inverse.dot(&m_k);
            let log_det_hessian = 0.5 * (inverse * &m_pair).sum() - 0.5 * (&solved_l * &solved_k.t()).sum();
            log_det_hessian
                + term.second_order(&first_k, &first_l, &pair).unwrap_or_else(|refusal| panic!("second order: {refusal}"))
        }

        /// The first derivative against Richardson differences of `L`.
        fn check_first_order(&self, label: &str) {
            for k in 0..2 {
                let (analytic, _) = self.first_derivative([0.0, 0.0], k);
                let along = |h: f64| {
                    let mut theta = [0.0, 0.0];
                    theta[k] = h;
                    let term = self.term(theta);
                    (term.value(), term.value_band())
                };
                let (estimate, bar) = richardson(&along, 1e-3);
                eprintln!("[CL] {label} θ{k}: analytic {analytic:.12e} FD {estimate:.12e} bar {bar:e}");
                assert!(
                    (analytic - estimate).abs() <= bar,
                    "{label}, θ{k}: analytic {analytic} against FD {estimate}, gap {:e} above {bar:e}",
                    (analytic - estimate).abs()
                );
            }
        }

        /// The second derivative against Richardson differences of the analytic first derivative.
        fn check_second_order(&self, label: &str) {
            for (k, l) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let analytic = self.second_derivative(k, l);
                let along = |h: f64| {
                    let mut theta = [0.0, 0.0];
                    theta[l] = h;
                    self.first_derivative(theta, k)
                };
                let (estimate, bar) = richardson(&along, 1e-3);
                eprintln!("[CL] {label} θ{k}θ{l}: analytic {analytic:.12e} FD {estimate:.12e} bar {bar:e}");
                assert!(
                    (analytic - estimate).abs() <= bar,
                    "{label}, θ{k}θ{l}: analytic {analytic} against FD {estimate}, gap {:e} above {bar:e}",
                    (analytic - estimate).abs()
                );
            }
        }
    }

    /// The analytic first derivative is the derivative of `L` where `M` is positive definite, where
    /// it is indefinite along the active normals (the continuation), and at `M₁₁ = 0.0201`, within
    /// `10⁻⁶` of where `M`'s smallest eigenvalue crosses zero (`M₁₁ = 0.04/1.99`).
    #[test]
    fn first_order_is_the_derivative_of_the_value_2765() {
        for corner in [0.3, 0.0201, -0.02] {
            Family::new(corner).check_first_order(&format!("M11 {corner}"));
        }
    }

    /// The analytic second derivative, the sites' motion included, is the derivative of the analytic
    /// first derivative, on both sides of the indefinite crossing.
    #[test]
    fn second_order_is_the_derivative_of_the_first_order_2765() {
        for corner in [0.3, -0.02] {
            Family::new(corner).check_second_order(&format!("M11 {corner}"));
        }
    }

    /// Rows and bounds that move with the coordinates, at scales other than one: the first
    /// derivative carries `tr(T̃ȦR)` and the slack's motion through the rows, on both sides of the
    /// indefinite crossing.
    #[test]
    fn first_order_carries_the_rows_motion_3171() {
        for corner in [0.3, -0.02] {
            Family::moving(corner).check_first_order(&format!("moving rows, M11 {corner}"));
        }
    }

    /// The second derivative with moving rows: the row motion's share of the log-determinant
    /// Hessian on `Λ`, the sites' motion under `Σ̇_u = ȦR + RᵀȦᵀ − RᵀΛ̇R`, and the pair's `Ä`, `b̈`.
    #[test]
    fn second_order_carries_the_rows_motion_3171() {
        for corner in [0.3, -0.02] {
            Family::moving(corner).check_second_order(&format!("moving rows, M11 {corner}"));
        }
    }

    /// Past the continuation's own boundary the tilted variance changes sign: at `σ = −μ²/2` one
    /// active row's constrained mode is at a fold, refused by name.
    ///
    /// This fixture sits EXACTLY ON that boundary, where the variance is zero, so what it
    /// establishes is that the code refuses AT the sign change. It does not establish that the
    /// refusal discriminates a mode just inside it from one just past it, and a production mode
    /// near the boundary returns a small negative number the verdict cannot separate from zero
    /// (gam#4571). The cavity the refusal now carries is what a reader needs to tell those apart,
    /// so it is asserted here rather than left to a `..`.
    #[test]
    fn a_fold_of_the_constrained_mode_is_refused_by_name_2765() {
        let mu = 2.0;
        let m = array![[-0.5 * mu * mu, 0.0], [0.0, 1.0]];
        let refusal = ConeLaplace::evaluate(&array![[1.0, 0.0]], &array![0.0], &array![0.0, 0.0], &array![mu, 0.0], &m)
            .expect_err("a fold is outside the boundary Laplace regime");
        let ConeLaplaceRefusal::Fold {
            variance,
            cavity_precision,
            cavity_shift,
            ..
        } = refusal
        else {
            panic!("refused as {refusal}");
        };
        assert!(
            !(variance > 0.0),
            "the fold verdict is the variance's sign: {variance:e}"
        );
        // The cavity at the starting sites is the leave-one-out precision, which for one row is
        // `M11` itself. Carrying it is what says whether a refusal is past the boundary
        // `-nu_c^2/2` or sitting on it.
        assert!(
            (cavity_precision - m[[0, 0]]).abs() <= 1e-12 * m[[0, 0]].abs(),
            "the refusal must report the cavity that produced it: {cavity_precision:e} against \
             M11 {:e}",
            m[[0, 0]]
        );
        assert!(
            cavity_shift.is_finite(),
            "the cavity shift is reported, not dropped: {cavity_shift:e}"
        );
    }

    /// The negative control the fold refusal needs: the same one-row geometry on the indefinite
    /// side of `M11 = 0`, where the other derivative pins in this module already evaluate, must
    /// PRICE rather than refuse. A refusal that fires on every indefinite `M` would be
    /// indistinguishable from the one above, and the whole point of the continuation is that an
    /// indefinite `M` is inside the regime until the boundary (gam#4571).
    #[test]
    fn an_indefinite_precision_inside_the_continuation_prices_rather_than_folding_4571() {
        let mu = 2.0;
        let m = array![[-0.02, 0.0], [0.0, 1.0]];
        let priced = ConeLaplace::evaluate(
            &array![[1.0, 0.0]],
            &array![0.0],
            &array![0.0, 0.0],
            &array![mu, 0.0],
            &m,
        )
        .expect("an indefinite precision inside the continuation is inside the regime");
        assert!(
            priced.value().is_finite(),
            "the term prices a finite value there: {:e}",
            priced.value()
        );
    }
}
