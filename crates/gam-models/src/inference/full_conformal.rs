//! Exact full-conformal prediction for penalized GAMs — including the
//! smoothing-parameter response (#942).
//!
//! # What this is
//!
//! Split conformal (src/inference/conformal.rs) buys finite-sample coverage
//! by sacrificing data to a calibration fold. FULL conformal uses every
//! observation for both fitting and calibration: for a candidate response
//! value `z` at the test covariates `x_*`, fit the model to the AUGMENTED
//! data `{(x_i, y_i)}_{i=1..n} ∪ {(x_*, z)}`, score every point with the
//! refit, and keep `z` in the prediction set iff the test point's
//! nonconformity score is not extreme among all n+1:
//!
//! ```text
//!   e_i(z) = |y_i − μ̂^z(x_i)| ,  e_*(z) = |z − μ̂^z(x_*)|
//!   C_α = { z :  1 + #{ i : e_i(z) ≥ e_*(z) }  >  α (n+1) }
//! ```
//!
//! Validity needs ONLY exchangeability of the n+1 points and SYMMETRY of
//! the fitting map (it must treat the augmented row like any other row).
//! No model correctness, no asymptotics, no held-out fold.
//!
//! The field treats this as computationally infeasible because it seems to
//! require refitting at a continuum of `z` — solved exactly only for ridge
//! (Nouretdinov et al. 2001) and approximately for lasso paths. Nobody runs
//! it for smoothing-selected GAMs, and every "efficient full conformal"
//! proposal FREEZES the smoothing parameters at their original-data values,
//! which silently breaks the symmetry requirement (the frozen ρ̂ was chosen
//! looking at y but not at z — the augmented row is treated differently)
//! with unquantified effect on coverage. This module closes both gaps:
//!
//! - **Layer 1 (implemented below, exact):** Gaussian identity at fixed ρ.
//!   The augmented fit is affine in `z`, so every score is piecewise
//!   linear in `z` and the EXACT set is computable from one factorization
//!   and ≤ 2n linear breakpoints — the ridge result generalized to
//!   arbitrary penalized smooths (any Sλ, any basis).
//! - **Layer 2 — continuous GLM (implemented below, certified):**
//!   predictor–corrector homotopy in `z` ([`GlmHomotopyFullConformal`]) —
//!   exact at corrector points because each correction is a Newton solve of
//!   the SAME symmetric KKT system a cold fit would solve, with the step
//!   size CERTIFIED by a computed third-derivative contraction bound and a
//!   cold-refit fallback whenever the certificate refuses.
//! - **Layer 3 (implemented in [`honest`], exact up to breakpoint
//!   resolution):** the Gaussian-identity map that RE-SELECTS the smoothing
//!   strength by REML on every augmented data set — the first
//!   full-conformal procedure whose fitting map treats the test row like a
//!   training row all the way up to ρ̂. A proven bound on where the global
//!   REML minimizer can lie, plus cold local refits at the set's endpoints.
//!   Every row carries a [`ConformalCertificate`]: `exact_frozen`,
//!   `honest_refit`, or `refused:<reason>` with the frozen set.
//!
//! # Layer 1 math (what the code below implements)
//!
//! Unit prior weights (REQUIRED for exchangeability — a non-unit weight on
//! a training row makes the rows non-exchangeable with the test row; the
//! constructor rejects that input rather than emit an invalid guarantee).
//! Augmented penalized least squares with fixed Sλ:
//!
//! ```text
//!   M       = XᵀX + x_* x_*ᵀ + Sλ                  (one factorization)
//!   β̂(z)    = M⁻¹ (Xᵀy + x_* z) = a + b z ,   a = M⁻¹Xᵀy , b = M⁻¹x_*
//! ```
//!
//! Every residual is AFFINE in z:
//!
//! ```text
//!   r_i(z)  = y_i − x_iᵀa − (x_iᵀb) z              i = 1..n
//!   r_*(z)  = −x_*ᵀa + (1 − x_*ᵀb) z
//! ```
//!
//! with `1 − x_*ᵀb = 1/(1 + h_*) > 0` for `h_* = x_*ᵀ(XᵀX+Sλ)⁻¹x_*` by
//! Sherman–Morrison — the test residual's slope never vanishes, so e_*(z)
//! is genuinely V-shaped and the rank function is well-defined everywhere.
//!
//! The comparison `e_i(z) ≥ e_*(z)` ⟺ `(r_i−r_*)(r_i+r_*) ≥ 0` flips only
//! at roots of two LINEAR equations per i. Collect ≤ 2n roots, sort, and
//! the rank of e_* is constant on each open interval between consecutive
//! roots: evaluate the rank at interval midpoints (and at the roots
//! themselves, closed-set convention — coverage uses `≥`, so boundary
//! points belong to the set when their rank qualifies) and assemble the
//! set as a union of intervals. EXACT — no grid, no tolerance, no refits.
//!
//! Unboundedness is honest, not an error: if `|slope(r_*)| ≤ |slope(r_i)|`
//! for enough i, far-out candidates are never extreme and the set is a
//! half-line or ℝ (low-information / high-leverage regimes). We return the
//! interval list as-is, ±∞ endpoints included — same honesty convention as
//! the split module's `+∞` multiplier.
//!
//! # Layer 2: GLM homotopy (implemented below)
//!
//! `β̂(z)` solves the augmented penalized score equation
//! `F(β; z) = Σ_i x_i (μ(η_i) − y_i) + x_*(μ(η_*) − z) + Sλβ = 0`
//! (canonical link form). The z-derivative is one sensitivity solve:
//!
//! ```text
//!   dβ̂/dz = H_pen⁻¹ x_*          (canonical: ∂F/∂z = −x_*)
//! ```
//!
//! Predictor–corrector walk over z with Newton correction of the SAME
//! KKT system the cold fit solves: exactness at corrector points is
//! convergence of Newton, not ODE integration accuracy. The step size is
//! CERTIFIED by the third-derivative data the tree already has (the PIRLS
//! `c`-array bounds ‖D_βH\[v\]‖ along the step, giving a computable Newton
//! attraction radius) — the corrector cannot silently skip a basin. Score
//! crossings between steps are localized by bisection on the corrected
//! path. Discrete families (Binomial, Poisson) are FINITE: full conformal is
//! exact by enumerating the response support — no homotopy subtlety at all —
//! and this module carries no enumeration arm.
//!
//! # Layer 3: the honest map (see [`honest`])
//!
//! Freezing ρ̂ at the training-data optimum breaks symmetry: ρ̂ saw `y` but
//! not `z`. The honest map fits the augmented rows at the global minimizer
//! `ρ̂(z)` of the profiled Gaussian REML criterion. With one penalty and one
//! Cholesky of the augmented normal matrix, the criterion, its
//! ρ-derivative and every residual are closed-form in `(ρ, z)`: quadratics
//! in `z` whose coefficients are monotone or unimodal in ρ. A branch and
//! bound over the extended `z` line keeps, per cell, a tube of ρ-boxes
//! that provably holds `ρ̂(z)` and decides membership over the whole tube;
//! the cells it splits are exactly the ones the data leave undecided. Cold
//! REML refits through the shared outer engine run at the set's finite
//! endpoints and are checked against the bound.
//!
//! Several penalties are refused (their ratios were selected without the
//! test row), as is a payload that does not record its penalty count: the
//! row gets the frozen-ρ set and a typed refusal, never a silent one.
//!
//! # Wiring
//!
//! No flags. The predict path requests full conformal exactly like split
//! conformal (`conformal_level`); Gaussian-identity fits get Layer 3 with its
//! certificate per row, GLMs the Layer-2 homotopy. Unit-weight violation and
//! unsupported regimes fall back to the split/ALO calibrator LOUDLY (logged),
//! never silently — an invalid guarantee is worse than a wider valid one.

use faer::Side;
use ndarray::{Array1, Array2};

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_av};

use opt::{BacktrackConfig, backtracking_line_search};

pub mod honest;
pub use honest::{
    ConformalCertificate, ConformalRefusal, HonestConformalCost, HonestFullConformal,
    honest_full_conformal,
};

#[cfg(test)]
mod test_support;

/// One maximal interval of candidate values retained in the prediction set.
/// Endpoints may be infinite (honest unboundedness in low-information /
/// high-leverage regimes).
#[derive(Clone, Debug, PartialEq)]
pub struct ConformalInterval {
    pub lo: f64,
    pub hi: f64,
}

/// A full-conformal prediction set: a finite union of closed intervals.
#[derive(Clone, Debug)]
pub struct FullConformalSet {
    /// Maximal intervals, sorted, disjoint.
    pub intervals: Vec<ConformalInterval>,
    /// Miscoverage level the set was built for.
    pub alpha: f64,
    /// `n + 1` (augmented count) — the denominator of the conformal rank.
    pub n_augmented: usize,
}

/// Shapes and unit prior weights, shared by every Gaussian full-conformal
/// constructor.
fn validate_inputs(
    x: &Array2<f64>,
    y: &Array1<f64>,
    prior_weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    x_star: &Array1<f64>,
) -> Result<(), String> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || prior_weights.len() != n {
        return Err("full conformal: row-count mismatch".to_string());
    }
    if s_lambda.nrows() != p || s_lambda.ncols() != p || x_star.len() != p {
        return Err("full conformal: column-count mismatch".to_string());
    }
    if prior_weights.iter().any(|&w| w != 1.0) {
        return Err(
            "full conformal requires unit prior weights: a reweighted training row is \
             not exchangeable with the test row, so the finite-sample coverage proof \
             does not apply; use the split/ALO conformal calibrator instead"
                .to_string(),
        );
    }
    Ok(())
}

/// The smallest dominating count `k` with `1 + k > α(n + 1)` — membership's
/// threshold — or `n + 1` when no count of `n` training rows reaches it.
fn required_dominating_count(n: usize, alpha: f64) -> usize {
    let threshold = alpha * (n + 1) as f64;
    (0..=n)
        .find(|&count| 1.0 + count as f64 > threshold)
        .unwrap_or(n + 1)
}

/// Exact Gaussian-identity full-conformal engine at fixed Sλ (Layer 1).
///
/// One factorization of `M = XᵀX + x_*x_*ᵀ + Sλ`; every candidate-z
/// quantity is affine in z thereafter. See the module doc for the math.
pub struct ExactGaussianFullConformal {
    /// Affine residual coefficients: `r_i(z) = u[i] + w[i]·z` for the n
    /// training rows, and the test residual in the LAST slot.
    u: Array1<f64>,
    w: Array1<f64>,
    n: usize,
}

impl ExactGaussianFullConformal {
    /// Build from raw fit ingredients. `x` is the n×p design at the
    /// TRAINING rows, `s_lambda` the p×p penalty at the fitted ρ̂ (frozen
    /// here by construction — Layer 3 owns the honest ρ-response),
    /// `x_star` the p-row at the test covariates.
    ///
    /// Rejects non-unit prior weights: exchangeability of the augmented
    /// row with the training rows is the entire coverage proof, and a
    /// reweighted row is not exchangeable with the test row. (Weighted
    /// conformal — Tibshirani et al. 2019 — is a different estimand with
    /// likelihood-ratio weights; it can be added as its own constructor,
    /// not silently conflated with this one.)
    pub fn new(
        x: &Array2<f64>,
        y: &Array1<f64>,
        prior_weights: &Array1<f64>,
        s_lambda: &Array2<f64>,
        x_star: &Array1<f64>,
    ) -> Result<Self, String> {
        validate_inputs(x, y, prior_weights, s_lambda, x_star)?;
        let n = x.nrows();
        let p = x.ncols();

        // M = XᵀX + x_*x_*ᵀ + Sλ — the augmented penalized normal matrix.
        let mut m = x.t().dot(x) + s_lambda;
        for i in 0..p {
            for j in 0..p {
                m[[i, j]] += x_star[i] * x_star[j];
            }
        }
        let chol = m
            .cholesky(Side::Lower)
            .map_err(|e| format!("full conformal: augmented normal matrix not SPD: {e:?}"))?;
        let xty = x.t().dot(y);
        let a = chol.solvevec(&xty);
        let b = chol.solvevec(&x_star.to_owned());

        // Affine residuals r_i(z) = u_i + w_i z; test residual last.
        let mut u = Array1::<f64>::zeros(n + 1);
        let mut w = Array1::<f64>::zeros(n + 1);
        let xa = fast_av(x, &a);
        let xb = fast_av(x, &b);
        for i in 0..n {
            u[i] = y[i] - xa[i];
            w[i] = -xb[i];
        }
        let mu_a_star = x_star.dot(&a);
        let h_frac = x_star.dot(&b); // = h/(1+h) ∈ [0, 1)
        u[n] = -mu_a_star;
        w[n] = 1.0 - h_frac; // strictly positive by Sherman–Morrison
        if w[n] <= 0.0 {
            return Err(
                "full conformal: test-residual slope 1 − x_*ᵀM⁻¹x_* must be positive; \
                 non-SPD or numerically broken augmented system"
                    .to_string(),
            );
        }
        Ok(Self { u, w, n })
    }

    /// Number of training rows whose score weakly dominates the test score
    /// at candidate z: `#{ i ≤ n : e_i(z) ≥ e_*(z) }`.
    fn dominating_count(&self, z: f64) -> usize {
        let e_star = (self.u[self.n] + self.w[self.n] * z).abs();
        (0..self.n)
            .filter(|&i| (self.u[i] + self.w[i] * z).abs() >= e_star)
            .count()
    }

    /// Membership at candidate z: conformal p-value `(1 + count)/(n+1) > α`.
    fn member(&self, z: f64, alpha: f64) -> bool {
        let n1 = (self.n + 1) as f64;
        (1.0 + self.dominating_count(z) as f64) > alpha * n1
    }

    /// The frozen plug-in mean `x_*ᵀ(XᵀX + Sλ)⁻¹Xᵀy`: the candidate at which
    /// the test residual vanishes.
    pub fn plug_in_mean(&self) -> f64 {
        -self.u[self.n] / self.w[self.n]
    }

    fn push_finite_root(points: &mut Vec<f64>, numerator: f64, denominator: f64) {
        if denominator.abs() > 0.0 {
            let z = numerator / denominator;
            if z.is_finite() {
                points.push(z);
            }
        }
    }

    /// The exact prediction set at miscoverage α.
    ///
    /// Breakpoints: for each i, roots of `r_*(z) = ±r_i(z)` — two linear
    /// equations. Between consecutive roots the comparison pattern (hence
    /// the rank of e_*) is constant; evaluate membership on midpoints and
    /// at every root (closed-set convention), then merge runs into maximal
    /// intervals. Cost O(n log n) after the single factorization.
    pub fn prediction_set(&self, alpha: f64) -> FullConformalSet {
        let n = self.n;
        let (us, ws) = (self.u[n], self.w[n]);
        let mut roots: Vec<f64> = Vec::with_capacity(2 * n);
        for i in 0..n {
            // r_* − r_i = (us − u_i) + (ws − w_i) z = 0
            let d = ws - self.w[i];
            Self::push_finite_root(&mut roots, self.u[i] - us, d);
            // r_* + r_i = (us + u_i) + (ws + w_i) z = 0
            let s = ws + self.w[i];
            Self::push_finite_root(&mut roots, -(us + self.u[i]), s);
        }
        roots.sort_by(|p, q| p.partial_cmp(q).expect("finite breakpoints"));
        roots.dedup_by(|p, q| *p == *q);

        // Witness points: each root, each gap midpoint, and the two open
        // tails. Membership is constant strictly between consecutive
        // roots, so one witness per piece decides the set exactly.
        let mut witnesses: Vec<f64> = Vec::with_capacity(2 * roots.len() + 3);
        if roots.is_empty() {
            witnesses.push(0.0);
        } else {
            let span = (roots[roots.len() - 1] - roots[0]).max(1.0);
            witnesses.push(roots[0] - span);
            for k in 0..roots.len() {
                witnesses.push(roots[k]);
                if k + 1 < roots.len() {
                    witnesses.push(0.5 * (roots[k] + roots[k + 1]));
                }
            }
            witnesses.push(roots[roots.len() - 1] + span);
        }

        // Scan witnesses into maximal intervals. A member midpoint/tail claims
        // its whole open gap; member roots close the endpoints.
        let mut intervals: Vec<ConformalInterval> = Vec::new();
        let mut open_lo: Option<f64> = None;
        let gap_bounds = |idx: usize| -> (f64, f64) {
            // bounds of the gap a witness at sorted position idx represents
            if roots.is_empty() {
                return (f64::NEG_INFINITY, f64::INFINITY);
            }
            if idx == 0 {
                return (f64::NEG_INFINITY, roots[0]);
            }
            if idx == witnesses.len() - 1 {
                return (roots[roots.len() - 1], f64::INFINITY);
            }
            // witnesses alternate root, mid, root, mid, ... after the first
            let k = (idx - 1) / 2; // gap index for midpoints, root index for roots
            if idx % 2 == 1 {
                // a root: zero-width "gap" at the root itself
                (roots[k], roots[k])
            } else {
                (roots[k], roots[k + 1])
            }
        };
        for (idx, &z) in witnesses.iter().enumerate() {
            let inside = self.member(z, alpha);
            let (lo, hi) = gap_bounds(idx);
            if inside {
                if open_lo.is_none() {
                    open_lo = Some(lo);
                }
                if idx == witnesses.len() - 1 {
                    intervals.push(ConformalInterval {
                        lo: open_lo.take().expect("open interval"),
                        hi,
                    });
                }
            } else if let Some(lo_open) = open_lo.take() {
                intervals.push(ConformalInterval {
                    lo: lo_open,
                    hi: lo,
                });
            }
        }

        FullConformalSet {
            intervals,
            alpha,
            n_augmented: n + 1,
        }
    }
}

/// Wilkinson growth for the Gaussian REML response's arithmetic: the factor
/// [`honest`] charges against every magnitude sum.
///
/// The `p`-terms count the Cholesky of `A(λ)`, its solves and its traces. The
/// `n`-terms count the residual sums the penalized RSS is formed from (#2280):
/// two passes of `X·β` over the rows, with their subtractions, squares and sums.
fn response_solve_growth(n: usize, p: usize) -> f64 {
    gam_linalg::roundoff::accumulation_growth(
        2 * p * p * p + 8 * p * p + 8 * p + 4 * n * p + 8 * n,
    )
}

/// `L⁻¹·B` for a lower-triangular `L`, by forward substitution.
fn solve_lower_triangular(lower: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let p = lower.nrows();
    let mut out = b.clone();
    for col in 0..out.ncols() {
        for i in 0..p {
            let mut acc = out[[i, col]];
            for k in 0..i {
                acc -= lower[[i, k]] * out[[k, col]];
            }
            out[[i, col]] = acc / lower[[i, i]];
        }
    }
    out
}

/// `L⁻ᵀ·B` for a lower-triangular `L`, by back substitution.
fn solve_lower_triangular_transposed(lower: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let p = lower.nrows();
    let mut out = b.clone();
    for col in 0..out.ncols() {
        for i in (0..p).rev() {
            let mut acc = out[[i, col]];
            for k in (i + 1)..p {
                acc -= lower[[k, i]] * out[[k, col]];
            }
            out[[i, col]] = acc / lower[[i, i]];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
// Layer 2 — continuous-GLM certified predictor–corrector homotopy in z
// ─────────────────────────────────────────────────────────────────────────

/// Maximum number of certified continuation sub-steps the homotopy may
/// spend walking between two consecutive candidates before it gives up and
/// falls back to a cold deterministic refit. A work budget, not a tuning
/// knob: exceeding it can only cost SPEED (one extra cold fit), never
/// correctness — the fallback solves the same KKT system to its optimum.
const GLM_HOMOTOPY_MAX_SUBSTEPS: usize = 1024;

/// Maximum step halvings per sub-step before the certificate's refusal is
/// treated as final and the cold-refit fallback fires.
const GLM_HOMOTOPY_MAX_HALVINGS: usize = 24;

/// Maximum chord-corrector iterations per certified sub-step. With the
/// contraction constant certified below [`GLM_CONTRACTION_ACCEPT`], the
/// residual shrinks at least geometrically, so this budget is generous.
const GLM_CORRECTOR_MAX_ITERS: usize = 80;

/// Maximum damped-Newton iterations for a cold augmented GLM fit.
const GLM_NEWTON_MAX_ITERS: usize = 200;

/// Maximum Armijo backtracking halvings per cold Newton iteration.
const GLM_NEWTON_MAX_BACKTRACKS: usize = 60;

/// Strict scale-invariant KKT tolerance declaring convergence, applied to the
/// RAW penalized gradient via [`GlmHomotopyFullConformal::kkt_converged`]
/// (dimension-scaled OR natural-scale relative — the same certificate the main
/// P-IRLS solver uses). NOT a tolerance on the preconditioned Newton step.
const GLM_CONVERGENCE_RTOL: f64 = 1e-12;

/// Near-stationary acceptance tolerance: a stalled iterate sitting at the
/// floating-point floor of the raw gradient is still accepted when it
/// certifies KKT stationarity at this looser scale-invariant tolerance. The
/// COMPUTED error bound carried out of the step uses the actual residual, so
/// accepting a stall is honest — the bound is simply larger and the downstream
/// margin gate decides whether a cold refit is needed. Mirrors the main
/// solver's 10×-band `near_stationary_kkt`.
const GLM_STALL_ACCEPT_RTOL: f64 = 1e-8;

/// Certified contraction constant below which a predictor step is accepted:
/// `κ < 1/2` makes the chord-corrector a contraction on the ball
/// `B(β_pred, 2‖H₀⁻¹F(β_pred)‖)`, which then provably contains the root.
const GLM_CONTRACTION_ACCEPT: f64 = 0.5;

/// Armijo sufficient-decrease constant for the cold-fit line search —
/// sourced from the shared optimizer constants so the workspace has exactly
/// one `c₁`.
const GLM_ARMIJO_C1: f64 = opt::constants::ARMIJO_C1;

/// `η` location of the extrema of the logistic third derivative
/// `b‴(η) = σ(1−σ)(1−2σ)`: `σ = (3±√3)/6 ⇔ η = ±ln(2+√3)`.
const LOGIT_THIRD_DERIV_CRITICAL_ETA: f64 = 1.316_957_896_924_816_6;

#[inline]
fn vec_norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

use gam_linalg::utils::stable_softplus as softplus;

/// Canonical-link GLM families supported by the certified z-homotopy
/// ([`GlmHomotopyFullConformal`]). Canonical links make the candidate
/// response enter the augmented penalized score LINEARLY (`∂F/∂z = −x_*`),
/// so the exact response of the augmented optimum to the candidate is the
/// single solve `dβ̂/dz = H⁻¹ x_*` — no family-specific cross terms. The
/// per-η derivative tower `b′ = μ`, `b″ = w`, `b‴` is the K=1 specialization
/// of the row-kernel channels (`row_kernel` Hessian / `row_third_contracted`
/// in src/families/row_kernel.rs); it is carried analytically here because
/// the homotopy must evaluate the tower at MOVING β while a `RowKernel`
/// evaluates at its internally held coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalGlmFamily {
    /// Bernoulli response, logit link: `b(η) = log(1+eʸ)`, `μ = σ(η)`.
    BernoulliLogit,
    /// Poisson response, log link: `b(η) = eʸ`, `μ = eʸ`.
    PoissonLog,
}

impl CanonicalGlmFamily {
    /// `μ(η) = b′(η)` — the canonical mean function.
    pub fn mean(&self, eta: f64) -> f64 {
        match self {
            Self::BernoulliLogit => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
            Self::PoissonLog => eta.exp(),
        }
    }

    /// `w(η) = b″(η)` — the canonical Fisher weight (strictly positive).
    pub fn weight(&self, eta: f64) -> f64 {
        match self {
            Self::BernoulliLogit => {
                let mu = self.mean(eta);
                mu * (1.0 - mu)
            }
            Self::PoissonLog => eta.exp(),
        }
    }

    /// Per-row negative log-likelihood kernel `b(η) − y η` (the y-independent
    /// normalizer is dropped — it never moves the optimum).
    fn nll_term(&self, eta: f64, y: f64) -> f64 {
        match self {
            Self::BernoulliLogit => softplus(eta) - y * eta,
            Self::PoissonLog => eta.exp() - y * eta,
        }
    }

    /// `sup { b″(η) : η ∈ [lo, hi] }` — COMPUTED interval bound on the
    /// Fisher weight, used to convert a coefficient-error bound into a
    /// mean-scale (score) error bound.
    fn weight_abs_sup(&self, lo: f64, hi: f64) -> f64 {
        match self {
            Self::BernoulliLogit => {
                if lo <= 0.0 && 0.0 <= hi {
                    0.25
                } else {
                    self.weight(lo).max(self.weight(hi))
                }
            }
            Self::PoissonLog => hi.exp(),
        }
    }

    /// `sup { |b‴(η)| : η ∈ [lo, hi] }` — COMPUTED interval bound on the
    /// third-derivative channel (the K=1 `row_third_contracted` value). The
    /// logistic case checks the interval endpoints and the two interior
    /// critical points `η = ±ln(2+√3)` where `|b‴|` attains its global
    /// maximum `1/(6√3)`; the Poisson case is monotone (`b‴ = eʸ`).
    fn third_abs_sup(&self, lo: f64, hi: f64) -> f64 {
        match self {
            Self::BernoulliLogit => {
                let t = |eta: f64| {
                    let mu = self.mean(eta);
                    (mu * (1.0 - mu) * (1.0 - 2.0 * mu)).abs()
                };
                let mut sup = t(lo).max(t(hi));
                for c in [
                    -LOGIT_THIRD_DERIV_CRITICAL_ETA,
                    LOGIT_THIRD_DERIV_CRITICAL_ETA,
                ] {
                    if lo <= c && c <= hi {
                        sup = sup.max(t(c));
                    }
                }
                sup
            }
            Self::PoissonLog => hi.exp(),
        }
    }

    /// Reject a training response outside the family's support — fitting a
    /// canonical GLM to an impossible response is a caller bug, not a
    /// numerical regime.
    fn validate_training_response(&self, y: f64, row: usize) -> Result<(), String> {
        if !y.is_finite() {
            return Err(format!("glm homotopy: non-finite response at row {row}"));
        }
        match self {
            Self::BernoulliLogit => {
                if !(0.0..=1.0).contains(&y) {
                    return Err(format!(
                        "glm homotopy: Bernoulli response must lie in [0, 1], got {y} at row {row}"
                    ));
                }
            }
            Self::PoissonLog => {
                if y < 0.0 {
                    return Err(format!(
                        "glm homotopy: Poisson response must be non-negative, got {y} at row {row}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Reject a conformal candidate outside the family's response support —
    /// the full-conformal set is a subset of the support by definition.
    fn validate_candidate(&self, z: f64) -> Result<(), String> {
        if !z.is_finite() {
            return Err(format!("glm homotopy: non-finite candidate {z}"));
        }
        match self {
            Self::BernoulliLogit => {
                if !(0.0..=1.0).contains(&z) {
                    return Err(format!(
                        "glm homotopy: Bernoulli candidate must lie in [0, 1], got {z}"
                    ));
                }
            }
            Self::PoissonLog => {
                if z < 0.0 {
                    return Err(format!(
                        "glm homotopy: Poisson candidate must be non-negative, got {z}"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// One candidate's verdict together with the tracked coefficients and the
/// COMPUTED bound on their distance to the exact augmented optimum.
#[derive(Clone, Debug)]
pub struct GlmHomotopyCandidate {
    pub z: f64,
    /// Conformal p-value `(1 + #{i ≤ n : e_i ≥ e_*}) / (n+1)` (ties count
    /// FOR the candidate — the conservative `≥` convention).
    pub p_value: f64,
    pub member: bool,
    /// The coefficients the verdict was computed from: the homotopy-tracked
    /// β̂(z) (chord-corrected to the augmented KKT root) or a cold refit.
    pub beta: Array1<f64>,
    /// Certified bound on `‖beta − β̂(z)‖₂` (distance to the EXACT augmented
    /// optimum), computed from the chord-contraction constant: with
    /// `r = ‖H₀⁻¹F(beta)‖` and certified `κ < ½` on `B(beta, 2r)`, the root
    /// lies in that ball and `‖beta − β̂(z)‖ ≤ r/(1−κ)`. `+∞` when the
    /// certificate refuses (the membership gate then forces a cold refit or
    /// reports the tie unresolved — never a silent guess).
    pub beta_error_bound: f64,
    /// Whether this candidate was decided from a cold deterministic refit
    /// (first candidate, certificate refusal, or margin-forced refit)
    /// rather than the tracked path.
    pub cold_refit: bool,
}

/// The exact full-conformal set for a canonical-link GLM, assembled by the
/// certified predictor–corrector homotopy with cold-refit fallback.
#[derive(Clone, Debug)]
pub struct GlmHomotopyConformalSet {
    /// Retained candidates, ascending.
    pub members: Vec<f64>,
    pub candidates: Vec<GlmHomotopyCandidate>,
    pub alpha: f64,
    /// `n + 1`.
    pub n_augmented: usize,
    /// Number of candidate transitions where the step certificate refused
    /// (third-order bound too large within the halving/sub-step budget) and
    /// the engine fell back to a cold deterministic refit.
    pub refit_fallbacks: usize,
    /// Number of cold refits forced by the MEMBERSHIP margin gate: the
    /// tracked solution was certified, but a rank comparison was decided by
    /// a margin smaller than the propagated score-error bound, so the
    /// engine refused to call it from the tracked path.
    pub margin_refits: usize,
    /// Number of candidates whose verdict remained margin-ambiguous even
    /// after a cold refit (a genuine floating-point-level score tie). The
    /// reported verdict then uses the conservative `≥` tie convention — the
    /// direction that can only over-cover, never under-cover.
    pub ties_unresolved: usize,
    /// Largest certified `‖beta − β̂(z)‖` bound over all reported candidates.
    pub max_beta_error_bound: f64,
}

struct GlmCandidateVerdict {
    p_value: f64,
    member: bool,
    decided: bool,
}

/// Certified predictor–corrector homotopy in the candidate response `z` for
/// canonical-link GLMs (#942 Layer 2, continuous arm).
///
/// # The path being tracked
///
/// `β̂(z)` solves the augmented penalized score equation
///
/// ```text
///   F(β; z) = Σᵢ xᵢ (μ(ηᵢ) − yᵢ) + x_* (μ(η_*) − z) + Sλ β = 0 ,
/// ```
///
/// which for a canonical link is the gradient of a STRICTLY convex objective
/// (Fisher weights `b″ > 0`, `Sλ ⪰ 0`, `H` required SPD), so the root is
/// unique — there is no basin-tracking failure mode and the homotopy can be
/// wrong only about SPEED, never about the answer. Since `∂F/∂z = −x_*`,
///
/// ```text
///   dβ̂/dz = H(β̂)⁻¹ x_* ,   H(β) = XᵀW(β)X + w_*(β) x_*x_*ᵀ + Sλ .
/// ```
///
/// # The certified step
///
/// From a corrected point `β₀` at `z` with factored `H₀ = H(β₀)`:
///
/// 1. **Predictor:** `β_pred = β₀ + h·H₀⁻¹x_*`.
/// 2. **Certificate:** the corrector is the chord iteration
///    `β ← β − H₀⁻¹F(β; z+h)` on the already-factored `H₀`. Its contraction
///    constant on the ball `B(β_pred, R)`, `R = 2‖H₀⁻¹F(β_pred)‖`, is
///    bounded by the COMPUTED quantity
///
///    ```text
///      κ = [ Σᵢ Tᵢ·devᵢ·‖xᵢ‖² + T_*·dev_*·‖x_*‖² ] / λ_min(H₀) ,
///      devᵢ = |h·xᵢᵀH₀⁻¹x_*| + ‖xᵢ‖·R ,
///      Tᵢ   = sup |b‴| over [ηᵢ(β₀) − devᵢ , ηᵢ(β₀) + devᵢ]
///    ```
///
///    (`‖H(β)−H₀‖₂ ≤ Σᵢ |wᵢ(β)−wᵢ(β₀)|·‖xᵢ‖²` and `|Δwᵢ| ≤ Tᵢ·|Δηᵢ|` —
///    the third-derivative tower bounding the Hessian's Lipschitz drift,
///    exactly the `row_third_contracted` channel evaluated as an interval
///    bound). `κ < ½` makes the chord map a contraction of `B(β_pred, R)`
///    into itself, so the (unique) root lies in the ball and the corrector
///    converges to it geometrically.
/// 3. **Refusal:** `κ ≥ ½` halves `h`; exhausting the halving or sub-step
///    budget abandons the path for this transition and falls back to a COLD
///    deterministic refit — the homotopy is only an acceleration of the
///    defined symmetric fitting map, never a redefinition of it.
/// 4. **Carried bound:** at acceptance the distance to the exact root is
///    bounded by the computed `r/(1−κ_f)` with `r` the final corrector
///    residual and `κ_f` re-evaluated at the final iterate.
///
/// # Membership with a margin gate
///
/// Scores are response-scale absolute residuals. The β-error bound
/// propagates to each score through the computed interval weight bound
/// (`|Δμᵢ| ≤ sup b″·‖xᵢ‖·bound`); a rank comparison decided by a margin
/// smaller than the joint perturbation is NOT trusted: the engine cold-refits
/// and re-decides, and if the tie survives the refit it applies the
/// conservative `≥` convention and reports it in `ties_unresolved`.
/// Exact-or-refuse, end to end.
///
/// ρ is frozen at the supplied `s_lambda` by construction — the honest
/// smoothing re-selection and its certificate are Layer 3's domain.
pub struct GlmHomotopyFullConformal<'a> {
    family: CanonicalGlmFamily,
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    s_lambda: &'a Array2<f64>,
    x_star: &'a Array1<f64>,
    n: usize,
    p: usize,
    /// `‖xᵢ‖₂` per training row.
    row_norm: Array1<f64>,
    /// `‖xᵢ‖₂²` per training row.
    row_sq: Array1<f64>,
    star_norm: f64,
    star_sq: f64,
}

impl<'a> GlmHomotopyFullConformal<'a> {
    /// Build the engine. Rejects non-unit prior weights for the same reason
    /// as [`ExactGaussianFullConformal::new`]: a reweighted training row is
    /// not exchangeable with the test row, so the coverage proof would not
    /// apply.
    pub fn new(
        family: CanonicalGlmFamily,
        x: &'a Array2<f64>,
        y: &'a Array1<f64>,
        prior_weights: &Array1<f64>,
        s_lambda: &'a Array2<f64>,
        x_star: &'a Array1<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n || prior_weights.len() != n {
            return Err("glm homotopy: row-count mismatch".to_string());
        }
        if s_lambda.nrows() != p || s_lambda.ncols() != p || x_star.len() != p {
            return Err("glm homotopy: column-count mismatch".to_string());
        }
        if prior_weights.iter().any(|&w| w != 1.0) {
            return Err(
                "glm homotopy full conformal requires unit prior weights: a reweighted \
                 training row is not exchangeable with the test row, so the finite-sample \
                 coverage proof does not apply; use the split/ALO conformal calibrator instead"
                    .to_string(),
            );
        }
        for (i, &yi) in y.iter().enumerate() {
            family.validate_training_response(yi, i)?;
        }
        let mut row_norm = Array1::<f64>::zeros(n);
        let mut row_sq = Array1::<f64>::zeros(n);
        for i in 0..n {
            let sq = x.row(i).dot(&x.row(i));
            row_sq[i] = sq;
            row_norm[i] = sq.sqrt();
        }
        let star_sq = x_star.dot(x_star);
        Ok(Self {
            family,
            x,
            y,
            s_lambda,
            x_star,
            n,
            p,
            row_norm,
            row_sq,
            star_norm: star_sq.sqrt(),
            star_sq,
        })
    }

    /// Augmented penalized score `F(β; z)`.
    fn penalized_score(&self, beta: &Array1<f64>, z: f64) -> Array1<f64> {
        let eta = fast_av(self.x, beta);
        let mut resid = Array1::<f64>::zeros(self.n);
        for i in 0..self.n {
            resid[i] = self.family.mean(eta[i]) - self.y[i];
        }
        let mut g = self.x.t().dot(&resid) + self.s_lambda.dot(beta);
        let r_star = self.family.mean(self.x_star.dot(beta)) - z;
        for j in 0..self.p {
            g[j] += self.x_star[j] * r_star;
        }
        g
    }

    /// Natural magnitude of the augmented penalized gradient
    /// `Xᵀμ − Xᵀy + Sβ + x_*(μ_* − z)`: the norms of the terms it is a
    /// difference of, `‖Xᵀμ‖₂ + ‖Xᵀy‖₂ + ‖Sβ‖₂ + ‖x_*‖·(|μ̂_*| + |z|)`. At
    /// the optimum those terms keep the data's magnitude while the gradient
    /// cancels to rounding, so its floor scales with this quantity. The
    /// cancelled score `‖Xᵀ(μ − y)‖₂` is not a scale: at an interior optimum
    /// of an unpenalised coefficient it is itself rounding-level (gam#3451,
    /// the P-IRLS form is gam#3339). The resulting stationarity residual is
    /// invariant under uniform rescaling of the objective.
    fn gradient_natural_scale(&self, beta: &Array1<f64>, z: f64) -> f64 {
        let eta = fast_av(self.x, beta);
        let mu = eta.mapv(|eta_i| self.family.mean(eta_i));
        let mu_star = self.family.mean(self.x_star.dot(beta));
        vec_norm(&self.x.t().dot(&mu))
            + vec_norm(&self.x.t().dot(self.y))
            + vec_norm(&self.s_lambda.dot(beta))
            + self.star_norm * (mu_star.abs() + z.abs())
    }

    /// Scale-invariant KKT acceptance on the RAW penalized gradient, exactly
    /// the `WorkingState::certifies_kkt` certificate the engine's main solver
    /// uses: the dimensionless residual `‖g‖ / gradient_natural_scale` is
    /// below `tol`. The earlier predicate compared the PRECONDITIONED Newton step
    /// `‖H⁻¹g‖` against `tol·(1 + ‖β‖)`, whose floating-point floor is
    /// `~ε·(n+1)/λ_min(H)` — n-dependent and not compensated by `(1 + ‖β‖)`,
    /// so genuinely-converged fits (e.g. raw gradient floor `3.6e-8` at
    /// moderate n) were rejected as non-converged.
    fn kkt_converged(&self, beta: &Array1<f64>, z: f64, tol: f64) -> bool {
        let g_norm = vec_norm(&self.penalized_score(beta, z));
        gam_solve::pirls::relative_gradient_residual(g_norm, self.gradient_natural_scale(beta, z))
            < tol
    }

    /// Augmented penalized NLL (line-search merit function).
    fn penalized_nll(&self, beta: &Array1<f64>, z: f64) -> f64 {
        let eta = fast_av(self.x, beta);
        let mut nll = 0.0;
        for i in 0..self.n {
            nll += self.family.nll_term(eta[i], self.y[i]);
        }
        nll += self.family.nll_term(self.x_star.dot(beta), z);
        nll + 0.5 * beta.dot(&self.s_lambda.dot(beta))
    }

    /// Augmented penalized Hessian `H(β)` (independent of `z` — the
    /// candidate enters the score linearly under a canonical link).
    fn penalized_hessian(&self, beta: &Array1<f64>) -> Array2<f64> {
        let eta = fast_av(self.x, beta);
        let mut xw = self.x.to_owned();
        for i in 0..self.n {
            let w = self.family.weight(eta[i]);
            for j in 0..self.p {
                xw[[i, j]] *= w;
            }
        }
        let mut h = self.x.t().dot(&xw) + self.s_lambda;
        let w_star = self.family.weight(self.x_star.dot(beta));
        for a in 0..self.p {
            for b in 0..self.p {
                h[[a, b]] += w_star * self.x_star[a] * self.x_star[b];
            }
        }
        h
    }

    /// The COMPUTED chord-contraction constant `κ` of the module doc: the
    /// Lipschitz drift of `H` over the stated η-intervals (per-row third
    /// derivative interval sups), divided by `λ_min(H₀)`. `shift[i]` is the
    /// known η-displacement of row i between the factorization point and the
    /// ball center; `radius` the coefficient-space ball radius around it.
    fn contraction_kappa(
        &self,
        eta0: &Array1<f64>,
        eta0_star: f64,
        shift: &Array1<f64>,
        shift_star: f64,
        radius: f64,
        lambda_min: f64,
    ) -> f64 {
        let mut drift = 0.0_f64;
        for i in 0..self.n {
            let dev = shift[i] + self.row_norm[i] * radius;
            let t_sup = self.family.third_abs_sup(eta0[i] - dev, eta0[i] + dev);
            drift += t_sup * dev * self.row_sq[i];
        }
        let dev_star = shift_star + self.star_norm * radius;
        drift += self
            .family
            .third_abs_sup(eta0_star - dev_star, eta0_star + dev_star)
            * dev_star
            * self.star_sq;
        drift / lambda_min
    }

    /// Certified bound on `‖beta − β̂(z)‖` at a claimed optimum: fresh
    /// factorization, one residual solve, contraction certificate on the
    /// ball `B(beta, 2r)`. `+∞` on refusal — never an assumed zero.
    fn stationary_error_bound(&self, beta: &Array1<f64>, z: f64) -> f64 {
        let hess = self.penalized_hessian(beta);
        let Ok(eigs) = hess.eigh(Side::Lower) else {
            return f64::INFINITY;
        };
        let lambda_min = eigs.0.iter().copied().fold(f64::INFINITY, f64::min);
        if !(lambda_min > 0.0) {
            return f64::INFINITY;
        }
        let Ok(chol) = hess.cholesky(Side::Lower) else {
            return f64::INFINITY;
        };
        let r0 = vec_norm(&chol.solvevec(&self.penalized_score(beta, z)));
        let eta0 = fast_av(self.x, beta);
        let eta0_star = self.x_star.dot(beta);
        let zero_shift = Array1::<f64>::zeros(self.n);
        let kappa =
            self.contraction_kappa(&eta0, eta0_star, &zero_shift, 0.0, 2.0 * r0, lambda_min);
        if kappa.is_finite() && kappa < GLM_CONTRACTION_ACCEPT {
            r0 / (1.0 - kappa)
        } else {
            f64::INFINITY
        }
    }

    /// Cold deterministic fit of the augmented problem at candidate `z`:
    /// damped Newton (full refactorization per iteration, Armijo
    /// backtracking on the convex penalized NLL) from `init`, run to the
    /// tight step tolerance. Returns the solution and its certified error
    /// bound. This IS the defined symmetric fitting map — the homotopy is
    /// only an acceleration of it.
    fn cold_fit(&self, z: f64, init: Array1<f64>) -> Result<(Array1<f64>, f64), String> {
        let mut beta = init;
        let mut nll = self.penalized_nll(&beta, z);
        if !nll.is_finite() {
            beta = Array1::<f64>::zeros(self.p);
            nll = self.penalized_nll(&beta, z);
        }
        let mut converged = false;
        for _ in 0..GLM_NEWTON_MAX_ITERS {
            let g = self.penalized_score(&beta, z);
            let hess = self.penalized_hessian(&beta);
            let chol = hess
                .cholesky(Side::Lower)
                .map_err(|e| format!("glm homotopy: augmented Hessian not SPD at z={z}: {e:?}"))?;
            let step = chol.solvevec(&g);
            if self.kkt_converged(&beta, z, GLM_CONVERGENCE_RTOL) {
                converged = true;
                break;
            }
            // gᵀH⁻¹g ≥ 0: the Newton direction is a descent direction.
            let decrease = g.dot(&step);
            let search = backtracking_line_search::<_, std::convert::Infallible>(
                BacktrackConfig {
                    initial_step: 1.0,
                    contraction: 0.5,
                    max_steps: GLM_NEWTON_MAX_BACKTRACKS,
                },
                |t| {
                    let mut cand = beta.clone();
                    cand.scaled_add(-t, &step);
                    let cand_nll = self.penalized_nll(&cand, z);
                    Ok(if cand_nll.is_finite() {
                        Some((cand_nll, cand))
                    } else {
                        None
                    })
                },
                |t, cand_nll| cand_nll <= nll - GLM_ARMIJO_C1 * t * decrease,
            );
            let accepted = match search {
                Ok(step) => step,
                Err(never) => match never {},
            };
            match accepted {
                Some(step) => {
                    beta = step.payload;
                    nll = step.value;
                }
                None => {
                    // The Armijo line search could not realize the predicted
                    // descent `½·gᵀH⁻¹g`. Near the optimum that decrease
                    // underflows the round-off of `penalized_nll` (`~ε·nll`),
                    // so a failed line search is the FLOOR of this Newton loop,
                    // not a true failure — the iterate is
                    // for-all-practical-purposes stationary. Stop iterating and
                    // let the certified error bound below decide acceptance
                    // (rather than rejecting on an un-improvable gradient
                    // floor).
                    break;
                }
            }
        }
        // Acceptance is decided by the COMPUTED coefficient-error bound, not by
        // a gradient-magnitude band. `stationary_error_bound` runs the chord
        // contraction certificate on a ball around the iterate: a finite value
        // PROVES the true optimum `β̂(z)` lies within `‖β − β̂(z)‖ ≤ bound`.
        // The Armijo/round-off floor of this Newton loop (`~√(ε·nll)`) can
        // exceed both the strict and the near-stationary gradient bands while
        // still being well inside a tight certified ball, so tying acceptance
        // to the certificate — the exact quantity the downstream margin gate
        // (`candidate_verdict`) consumes — is both honest (a larger bound only
        // widens the undecided band) and immune to the n-/scale-dependent
        // gradient floor that spuriously rejected reachable optima.
        let bound = self.stationary_error_bound(&beta, z);
        if !converged && !bound.is_finite() {
            // Neither the strict KKT band nor the contraction certificate could
            // confirm proximity to a stationary point: a genuine non-convergence.
            let g_norm = vec_norm(&self.penalized_score(&beta, z));
            let residual = g_norm / (1.0 + self.gradient_natural_scale(&beta, z));
            return Err(format!(
                "glm homotopy: cold fit did not converge at z={z} \
                 (uncertified; relative gradient residual {residual})"
            ));
        }
        Ok((beta, bound))
    }

    /// Walk the corrected path from `z_from` (where `beta` solves the
    /// augmented KKT system) to `z_to` via certified predictor–corrector
    /// sub-steps. On success `beta` holds the corrected solution at `z_to`
    /// and the certified `‖beta − β̂(z_to)‖` bound is returned. `None` is a
    /// certified REFUSAL (budget exhausted, certificate never below ½, or a
    /// factorization failure) — the caller falls back to a cold refit; the
    /// refusal can cost speed only, never correctness.
    fn track(&self, beta: &mut Array1<f64>, z_from: f64, z_to: f64) -> Option<f64> {
        let mut z = z_from;
        let mut h = z_to - z_from;
        let mut arrival_bound = f64::INFINITY;
        for _ in 0..GLM_HOMOTOPY_MAX_SUBSTEPS {
            let remaining = z_to - z;
            if remaining <= 0.0 {
                return Some(arrival_bound);
            }
            h = h.min(remaining);
            let hess = self.penalized_hessian(beta);
            let lambda_min = hess
                .eigh(Side::Lower)
                .ok()?
                .0
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            if !(lambda_min > 0.0) {
                return None;
            }
            let chol = hess.cholesky(Side::Lower).ok()?;
            let b_dir = chol.solvevec(self.x_star);
            let eta0 = fast_av(self.x, beta);
            let eta0_star = self.x_star.dot(beta);
            let xb = fast_av(self.x, &b_dir);
            let xb_star = self.x_star.dot(&b_dir);

            let mut accepted = false;
            for _ in 0..=GLM_HOMOTOPY_MAX_HALVINGS {
                let h_eff = h.min(z_to - z);
                let z_new = if h_eff >= z_to - z { z_to } else { z + h_eff };
                let mut beta_pred = beta.clone();
                beta_pred.scaled_add(h_eff, &b_dir);
                let s0 = chol.solvevec(&self.penalized_score(&beta_pred, z_new));
                let r0 = vec_norm(&s0);
                let radius = 2.0 * r0;
                let shift = xb.mapv(|t| (h_eff * t).abs());
                let kappa = self.contraction_kappa(
                    &eta0,
                    eta0_star,
                    &shift,
                    (h_eff * xb_star).abs(),
                    radius,
                    lambda_min,
                );
                if kappa.is_finite() && kappa < GLM_CONTRACTION_ACCEPT {
                    // Chord corrector on the already-factored H₀: certified
                    // geometric contraction toward the unique root.
                    let mut bcur = beta_pred;
                    let mut step = s0;
                    let mut r = r0;
                    for _ in 0..GLM_CORRECTOR_MAX_ITERS {
                        if self.kkt_converged(&bcur, z_new, GLM_CONVERGENCE_RTOL) {
                            break;
                        }
                        let mut next = bcur.clone();
                        next.scaled_add(-1.0, &step);
                        let next_step = chol.solvevec(&self.penalized_score(&next, z_new));
                        let r_next = vec_norm(&next_step);
                        if !(r_next < r) {
                            // Floating-point floor: stop here; acceptance is
                            // decided by the residual level below.
                            break;
                        }
                        bcur = next;
                        step = next_step;
                        r = r_next;
                    }
                    if self.kkt_converged(&bcur, z_new, GLM_STALL_ACCEPT_RTOL) {
                        // Re-certify at the final iterate and carry the
                        // COMPUTED distance-to-root bound.
                        let mut diff = bcur.clone();
                        diff.scaled_add(-1.0, beta);
                        let shift_fin = fast_av(self.x, &diff).mapv(f64::abs);
                        let kappa_fin = self.contraction_kappa(
                            &eta0,
                            eta0_star,
                            &shift_fin,
                            self.x_star.dot(&diff).abs(),
                            2.0 * r,
                            lambda_min,
                        );
                        if kappa_fin.is_finite() && kappa_fin < GLM_CONTRACTION_ACCEPT {
                            arrival_bound = r / (1.0 - kappa_fin);
                            *beta = bcur;
                            z = z_new;
                            // Grow the trial step on an easy acceptance.
                            h = 2.0 * h_eff;
                            accepted = true;
                            break;
                        }
                    }
                }
                h = 0.5 * h_eff;
                if !(h > 0.0) {
                    return None;
                }
            }
            if !accepted {
                return None;
            }
        }
        if z_to - z <= 0.0 {
            Some(arrival_bound)
        } else {
            None
        }
    }

    /// Propagated score-error bound for one row: `|Δe| ≤ |Δμ| ≤
    /// sup b″ · ‖x‖ · bound`, with the weight sup COMPUTED over the η-interval
    /// the coefficient ball can reach.
    fn score_delta(&self, eta: f64, x_norm: f64, beta_error_bound: f64) -> f64 {
        if beta_error_bound == 0.0 {
            return 0.0;
        }
        if !beta_error_bound.is_finite() {
            return f64::INFINITY;
        }
        let dev = x_norm * beta_error_bound;
        self.family.weight_abs_sup(eta - dev, eta + dev) * dev
    }

    /// Rank the candidate with the margin gate: `decided` is true iff every
    /// possible score perturbation within the certified bound leaves the
    /// membership verdict unchanged.
    fn candidate_verdict(
        &self,
        z: f64,
        alpha: f64,
        beta: &Array1<f64>,
        beta_error_bound: f64,
    ) -> GlmCandidateVerdict {
        let eta = fast_av(self.x, beta);
        let eta_star = self.x_star.dot(beta);
        let e_star = (z - self.family.mean(eta_star)).abs();
        let delta_star = self.score_delta(eta_star, self.star_norm, beta_error_bound);
        let mut count = 0usize;
        let mut count_certain = 0usize;
        let mut count_possible = 0usize;
        for i in 0..self.n {
            let e_i = (self.y[i] - self.family.mean(eta[i])).abs();
            let tol = self.score_delta(eta[i], self.row_norm[i], beta_error_bound) + delta_star;
            let gap = e_i - e_star;
            if gap >= 0.0 {
                count += 1;
            }
            if gap >= tol {
                count_certain += 1;
            }
            if gap >= -tol {
                count_possible += 1;
            }
        }
        let n1 = (self.n + 1) as f64;
        let member = (1.0 + count as f64) > alpha * n1;
        let member_lo = (1.0 + count_certain as f64) > alpha * n1;
        let member_hi = (1.0 + count_possible as f64) > alpha * n1;
        GlmCandidateVerdict {
            p_value: (1.0 + count as f64) / n1,
            member,
            decided: member_lo == member_hi,
        }
    }

    /// Assemble the exact full-conformal set over the (strictly increasing)
    /// candidate list: cold fit at the first candidate, certified homotopy
    /// tracking between consecutive candidates with cold-refit fallback on
    /// certificate refusal, and the margin gate on every verdict.
    pub fn prediction_set(
        &self,
        candidates: &[f64],
        alpha: f64,
    ) -> Result<GlmHomotopyConformalSet, String> {
        if candidates.is_empty() {
            return Err("glm homotopy: empty candidate list".to_string());
        }
        if !(0.0..1.0).contains(&alpha) {
            return Err(format!(
                "glm homotopy: alpha must be in [0, 1), got {alpha}"
            ));
        }
        if candidates.windows(2).any(|w| !(w[0] < w[1])) {
            return Err("glm homotopy: candidates must be strictly increasing".to_string());
        }
        for &z in candidates {
            self.family.validate_candidate(z)?;
        }

        let (mut beta, mut bound) = self.cold_fit(candidates[0], Array1::<f64>::zeros(self.p))?;
        let mut out: Vec<GlmHomotopyCandidate> = Vec::with_capacity(candidates.len());
        let mut members: Vec<f64> = Vec::new();
        let mut refit_fallbacks = 0usize;
        let mut margin_refits = 0usize;
        let mut ties_unresolved = 0usize;
        let mut max_bound = 0.0_f64;
        let mut prev_z = candidates[0];
        for (idx, &z) in candidates.iter().enumerate() {
            let mut cold = idx == 0;
            if idx > 0 {
                match self.track(&mut beta, prev_z, z) {
                    Some(b) => bound = b,
                    None => {
                        let (refit_beta, refit_bound) = self.cold_fit(z, beta.clone())?;
                        beta = refit_beta;
                        bound = refit_bound;
                        refit_fallbacks += 1;
                        cold = true;
                    }
                }
            }
            let mut verdict = self.candidate_verdict(z, alpha, &beta, bound);
            if !verdict.decided && !cold {
                let (refit_beta, refit_bound) = self.cold_fit(z, beta.clone())?;
                beta = refit_beta;
                bound = refit_bound;
                cold = true;
                margin_refits += 1;
                verdict = self.candidate_verdict(z, alpha, &beta, bound);
            }
            if !verdict.decided {
                ties_unresolved += 1;
            }
            if bound.is_finite() {
                max_bound = max_bound.max(bound);
            } else {
                max_bound = f64::INFINITY;
            }
            if verdict.member {
                members.push(z);
            }
            out.push(GlmHomotopyCandidate {
                z,
                p_value: verdict.p_value,
                member: verdict.member,
                beta: beta.clone(),
                beta_error_bound: bound,
                cold_refit: cold,
            });
            prev_z = z;
        }
        Ok(GlmHomotopyConformalSet {
            members,
            candidates: out,
            alpha,
            n_augmented: self.n + 1,
            refit_fallbacks,
            margin_refits,
            ties_unresolved,
            max_beta_error_bound: max_bound,
        })
    }
}

/// Persisted frozen penalty for the Gaussian-identity full-conformal set
/// (#942 Layers 1 and 3).
///
/// The exact full-conformal set has no test-point-independent
/// factorization: every test covariate `x_*` enters the augmented normal matrix
/// `M = XᵀX + x_*x_*ᵀ + Sλ`, and every conformity score is a residual of one
/// labeled row, so the set needs the labeled rows `(X, y)` themselves. A saved
/// model therefore persists only the p × p frozen penalty `Sλ`; the labeled
/// rows are supplied again at prediction time and joined to it in
/// [`ExactFullConformalPenalty::with_labeled_rows`]. The saved model stays
/// O(p²) whatever the training size.
///
/// `Sλ` is recovered once at fit time from the converged penalized Hessian
/// `M₀ = XᵀX + Sλ` (the Gaussian-identity, unit-weight, dispersion-unscaled
/// normal matrix stored in `FitGeometry`) as `Sλ = M₀ − XᵀX`, so no penalty
/// re-derivation is needed.
///
/// Older payloads persisted the training `x` and `y` beside `s_lambda` under
/// the same field; deserialization reads `s_lambda` and ignores them. Payloads
/// written before `penalty_count` existed (v32 and older) read it as `None`, and
/// their rows are refused with [`ConformalRefusal::UnknownPenaltyStructure`]. A
/// v32 binary refuses a payload carrying the count by version (v33), so no binary
/// publishes a frozen-λ set for a fit whose smoothing selection it cannot see.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExactFullConformalPenalty {
    /// Frozen penalty `Sλ = M₀ − XᵀX` at the fitted smoothing parameters (p × p).
    s_lambda: Array2<f64>,
    /// Number of smoothing parameters the fit selected, which decides whether
    /// the REML re-selecting map is computable ([`honest_full_conformal`]).
    #[serde(default)]
    penalty_count: Option<usize>,
}

impl ExactFullConformalPenalty {
    /// Recover the frozen penalty `Sλ = M₀ − XᵀX` from the unit-weight training
    /// Gram matrix `XᵀX` and the converged penalized normal matrix `M₀`, with the
    /// fit's smoothing-parameter count.
    pub fn from_gram_and_normal_matrix(
        gram: &Array2<f64>,
        m: &Array2<f64>,
        penalty_count: usize,
    ) -> Result<Self, String> {
        let p = gram.nrows();
        if gram.ncols() != p || m.nrows() != p || m.ncols() != p {
            return Err("exact full conformal penalty: normal-matrix shape mismatch".to_string());
        }
        Ok(Self {
            s_lambda: m - gram,
            penalty_count: Some(penalty_count),
        })
    }

    /// Coefficient dimension `p`.
    pub fn p(&self) -> usize {
        self.s_lambda.nrows()
    }

    /// Join the frozen penalty to labeled rows `(X, y)` for the per-test-row
    /// exact set. Every row carries unit weight: only models trained without
    /// prior weights persist this penalty.
    ///
    /// The rows need not be the training rows. The set is exact for whatever
    /// labeled rows are supplied; with the training rows it is the frozen-λ
    /// full-conformal set of the fit, and with rows the penalty was not
    /// selected on the augmented scores are exchangeable under either map.
    pub fn with_labeled_rows(
        &self,
        x: Array2<f64>,
        y: Array1<f64>,
    ) -> Result<ExactFullConformalSubstrate, String> {
        if x.nrows() != y.len() {
            return Err("exact full conformal substrate: row-count mismatch".to_string());
        }
        if x.ncols() != self.p() {
            return Err(format!(
                "exact full conformal substrate: labeled design has {} columns but the frozen \
                 penalty has p={}",
                x.ncols(),
                self.p()
            ));
        }
        Ok(ExactFullConformalSubstrate {
            x,
            y,
            s_lambda: self.s_lambda.clone(),
            penalty_count: self.penalty_count,
        })
    }
}

/// Runtime substrate for the Gaussian-identity full-conformal set: the labeled
/// design `X`, response `y`, the frozen penalty `Sλ` and the fit's
/// smoothing-parameter count. Each test row gets [`honest_full_conformal`]: the
/// set of the map that re-selects the smoothing strength by REML on the
/// augmented rows, or the frozen-ρ set with a typed refusal. It is never
/// persisted (see [`ExactFullConformalPenalty`]).
///
/// Unit prior weights are required, as everywhere in this module: a reweighted
/// training row is not exchangeable with the test row.
#[derive(Clone, Debug)]
pub struct ExactFullConformalSubstrate {
    /// Labeled design `X` (n × p).
    x: Array2<f64>,
    /// Labeled response `y` (n).
    y: Array1<f64>,
    /// Frozen penalty `Sλ` at the fitted smoothing parameters (p × p).
    s_lambda: Array2<f64>,
    /// Smoothing parameters the fit selected (`None`: not recorded).
    penalty_count: Option<usize>,
}

/// One test row's full-conformal verdict: the outer `[lower, upper]` envelope
/// of the set, the set itself, what it guarantees and what it cost.
#[derive(Clone, Debug)]
pub struct ExactFullConformalInterval {
    /// Outer envelope `[min lo, max hi]` of the (possibly multi-interval) set,
    /// inheriting its coverage (it is a superset). Endpoints may be infinite
    /// (honest unboundedness in low-information / high-leverage regimes).
    pub lo: f64,
    pub hi: f64,
    /// The set itself (a union of intervals).
    pub set: FullConformalSet,
    /// `exact_frozen`, `honest_refit`, or `refused:<reason>` (the frozen-ρ set,
    /// with no finite-sample guarantee).
    pub certificate: ConformalCertificate,
    /// Factorizations, eigendecompositions, cold refits and cells the row cost.
    pub cost: HonestConformalCost,
}

impl ExactFullConformalSubstrate {
    /// Build the substrate from the training design, response, prior weights,
    /// the converged penalized normal matrix `M₀ = XᵀX + Sλ` and the fit's
    /// smoothing-parameter count. Recovers the frozen penalty `Sλ = M₀ − XᵀX`
    /// once. Rejects non-unit prior weights and shape mismatches, identically to
    /// the rest of this module.
    pub fn from_design_unit_weight_normal_matrix(
        x: &Array2<f64>,
        y: &Array1<f64>,
        prior_weights: &Array1<f64>,
        m: &Array2<f64>,
        penalty_count: usize,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n || prior_weights.len() != n {
            return Err("exact full conformal substrate: row-count mismatch".to_string());
        }
        if m.nrows() != p || m.ncols() != p {
            return Err("exact full conformal substrate: normal-matrix shape mismatch".to_string());
        }
        if prior_weights.iter().any(|&w| w != 1.0) {
            return Err(
                "exact full conformal requires unit prior weights: a reweighted training row \
                 is not exchangeable with the test row, so the finite-sample coverage proof \
                 does not apply"
                    .to_string(),
            );
        }
        // Sλ = M₀ − XᵀX (frozen at the fitted smoothing parameters).
        let s_lambda = m - &x.t().dot(x);
        Ok(Self {
            x: x.clone(),
            y: y.clone(),
            s_lambda,
            penalty_count: Some(penalty_count),
        })
    }

    /// Coefficient dimension `p`.
    pub fn p(&self) -> usize {
        self.x.ncols()
    }

    /// Training-row count `n`.
    pub fn n(&self) -> usize {
        self.x.nrows()
    }

    /// The full-conformal verdict at one test row `x_*` and miscoverage `alpha`.
    pub fn interval(
        &self,
        x_star: &Array1<f64>,
        alpha: f64,
    ) -> Result<ExactFullConformalInterval, String> {
        if x_star.len() != self.p() {
            return Err(format!(
                "exact full conformal: x_* has {} entries but the fit has {} coefficients",
                x_star.len(),
                self.p()
            ));
        }
        let weights = Array1::<f64>::ones(self.n());
        let row = honest_full_conformal(
            &self.x,
            &self.y,
            &weights,
            &self.s_lambda,
            self.penalty_count,
            x_star,
            alpha,
        )?;
        let (lo, hi) = match (row.set.intervals.first(), row.set.intervals.last()) {
            (Some(first), Some(last)) => (first.lo, last.hi),
            // No candidate qualifies (pathological tiny α·(n+1)); collapse to the
            // plug-in mean — the only honest scalar answer.
            _ => (row.plug_in_mean, row.plug_in_mean),
        };
        Ok(ExactFullConformalInterval {
            lo,
            hi,
            set: row.set,
            certificate: row.certificate,
            cost: row.cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::GaussianRemlRhoResponse;
    use super::*;
    use ndarray::{Array1, Array2};

    /// Small penalized smooth: verify the breakpoint-scan set against a
    /// dense brute-force grid of explicit augmented refits (independent
    /// linear-algebra path), and check basic coverage sanity.
    #[test]
    fn exact_set_matches_brute_force_refits() {
        let n = 24usize;
        let p = 5usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (t * (j as f64 + 1.0) * std::f64::consts::PI).sin();
            }
            y[i] = 1.2 * (2.0 * std::f64::consts::PI * t).sin()
                + 0.3 * (17.0 * (i as f64) + 0.5).sin();
        }
        let mut s_lambda = Array2::<f64>::eye(p);
        s_lambda *= 0.7;
        let weights = Array1::<f64>::ones(n);
        let mut x_star = Array1::<f64>::zeros(p);
        for j in 0..p {
            x_star[j] = (0.37 * (j as f64 + 1.0) * std::f64::consts::PI).sin();
        }

        let engine =
            ExactGaussianFullConformal::new(&x, &y, &weights, &s_lambda, &x_star).expect("engine");
        let alpha = 0.2;
        let set = engine.prediction_set(alpha);
        assert!(!set.intervals.is_empty(), "set should be non-empty");

        // Independent oracle: explicit augmented refit per grid z.
        let m_base = x.t().dot(&x) + &s_lambda;
        let oracle = |z: f64| -> bool {
            let mut m = m_base.clone();
            for i in 0..p {
                for j in 0..p {
                    m[[i, j]] += x_star[i] * x_star[j];
                }
            }
            let chol = m.cholesky(Side::Lower).expect("oracle chol");
            let mut rhs = x.t().dot(&y);
            for j in 0..p {
                rhs[j] += x_star[j] * z;
            }
            let beta = chol.solvevec(&rhs);
            let e_star = (z - x_star.dot(&beta)).abs();
            let count = (0..n)
                .filter(|&i| {
                    let mu_i: f64 = x.row(i).dot(&beta);
                    (y[i] - mu_i).abs() >= e_star
                })
                .count();
            (1.0 + count as f64) > alpha * (n as f64 + 1.0)
        };

        let z_lo = set.intervals.first().map(|i| i.lo).unwrap_or(-5.0) - 2.0;
        let z_hi = set.intervals.last().map(|i| i.hi).unwrap_or(5.0) + 2.0;
        let z_lo = if z_lo.is_finite() { z_lo } else { -50.0 };
        let z_hi = if z_hi.is_finite() { z_hi } else { 50.0 };
        let grid = 4001usize;
        for g in 0..grid {
            let z = z_lo + (z_hi - z_lo) * g as f64 / (grid as f64 - 1.0);
            let in_set = set.intervals.iter().any(|itv| z >= itv.lo && z <= itv.hi);
            assert_eq!(
                in_set,
                oracle(z),
                "breakpoint scan disagrees with brute-force refit at z={z}"
            );
        }

        // The fitted value at x_* must be in the set at α=0.2 for any sane
        // problem (its residual is small by construction of the fit).
        let chol = m_base.cholesky(Side::Lower).expect("chol");
        let beta_unaug = chol.solvevec(&x.t().dot(&y));
        let mu_star = x_star.dot(&beta_unaug);
        assert!(
            set.intervals
                .iter()
                .any(|itv| mu_star >= itv.lo && mu_star <= itv.hi),
            "point prediction should be inside its own conformal set"
        );
    }

    #[test]
    fn boundary_tie_is_a_closed_point_set() {
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).expect("x");
        let y = Array1::from_vec(vec![0.0]);
        let weights = Array1::ones(1);
        let s_lambda = Array2::from_shape_vec((1, 1), vec![1.0]).expect("s");
        let x_star = Array1::from_vec(vec![1.0]);
        let engine =
            ExactGaussianFullConformal::new(&x, &y, &weights, &s_lambda, &x_star).expect("engine");

        let set = engine.prediction_set(0.5);
        assert_eq!(set.intervals.len(), 1);
        assert_eq!(set.intervals[0].lo, 0.0);
        assert_eq!(set.intervals[0].hi, 0.0);
    }

    #[test]
    fn identically_tied_rows_give_the_whole_line() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).expect("x");
        let y = Array1::from_vec(vec![0.0]);
        let weights = Array1::ones(1);
        let s_lambda = Array2::from_shape_vec((1, 1), vec![0.0]).expect("s");
        let x_star = Array1::from_vec(vec![1.0]);
        let engine =
            ExactGaussianFullConformal::new(&x, &y, &weights, &s_lambda, &x_star).expect("engine");

        let set = engine.prediction_set(0.5);
        assert_eq!(set.intervals.len(), 1);
        assert_eq!(set.intervals[0].lo, f64::NEG_INFINITY);
        assert_eq!(set.intervals[0].hi, f64::INFINITY);
    }

    #[test]
    fn strictly_separated_slopes_give_the_whole_line() {
        let engine = ExactGaussianFullConformal {
            u: Array1::from_vec(vec![1.0, 1.0, 0.0]),
            w: Array1::from_vec(vec![1.0, -1.0, 0.1]),
            n: 2,
        };

        let set = engine.prediction_set(0.5);
        assert_eq!(set.intervals.len(), 1);
        assert_eq!(set.intervals[0].lo, f64::NEG_INFINITY);
        assert_eq!(set.intervals[0].hi, f64::INFINITY);
    }

    /// A smooth Gaussian fixture: cosine basis design (column 0 constant,
    /// column 1 the first harmonic — both unpenalized), a quartic-frequency
    /// curvature penalty on the higher harmonics (`rank = p − 2`, nullity 2),
    /// and a one-harmonic truth plus tiny deterministic noise.
    fn gauss_reml_fixture(n: usize, p: usize) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        use std::f64::consts::PI;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (j as f64 * PI * t).cos();
            }
            y[i] = (2.0 * PI * t).sin() + 0.05 * (13.0 * i as f64 + 0.7).sin();
        }
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            s[[j, j]] = if j < 2 { 0.0 } else { (j as f64).powi(4) };
        }
        (x, y, s)
    }

    fn cosine_row(p: usize, t: f64) -> Array1<f64> {
        use std::f64::consts::PI;
        let mut r = Array1::<f64>::zeros(p);
        for j in 0..p {
            r[j] = (j as f64 * PI * t).cos();
        }
        r
    }

    /// #2902 row 8: the REML oracle's ρ domain is the #2812 resolvability domain
    /// of its Gram against the penalty. Orthogonal columns make the generalized
    /// eigenvalue closed form, `γ = ‖x₁‖²/s₁₁`, so the domain is
    /// `[ln(√ε·γ), ln(γ/√ε)]`; the test row adds `x_*x_*ᵀ` to the Gram and moves
    /// γ from 2 to 5/2.
    #[test]
    fn oracle_rho_domain_is_the_resolvability_interval_of_its_gram_2902() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
            .expect("design");
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 5.0]);
        let s = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 2.0]).expect("penalty");
        let x_star = Array1::from_vec(vec![0.0, 1.0]);
        let resp = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");
        let half_log_epsilon = 0.5 * f64::EPSILON.ln();
        for (domain, gamma, label) in [
            (resp.rho_domain, 4.0_f64 / 2.0, "training"),
            (resp.augmented_rho_domain, 5.0_f64 / 2.0, "augmented"),
        ] {
            let expected = (gamma.ln() + half_log_epsilon, gamma.ln() - half_log_epsilon);
            assert!(
                (domain.0 - expected.0).abs() <= 1.0e-12 && (domain.1 - expected.1).abs() <= 1.0e-12,
                "{label} ρ domain {domain:?} is not the resolvability interval {expected:?}"
            );
        }
    }

    /// #2280: the penalized RSS survives a planted residual below the rounding of
    /// `yᵀy`.
    ///
    /// The response is `X·β₀ + ρ·e`, with `β₀` on the penalty's null space (the
    /// `cos πt` column) and `e` a unit vector orthogonal to the design's column space.
    /// The test row sits at `t = ½`, where `x_*ᵀβ₀ = cos(π/2)` vanishes to rounding. At
    /// `z = 0` the augmented penalized objective is minimized by `β₀` at the value
    /// `ρ² + (x_*ᵀβ₀)²`, and `ρ² = 1e-16` sits below one ulp of `yᵀy ≈ 22`. So the
    /// closed form `yᵀy − cᵀβ̂` cannot represent it. That miss is asserted first, as
    /// the positive control that the fixture reaches the regime. Because the closed
    /// form's result is quantized at that ulp, it is held to half the residual.
    ///
    /// The reference is an independent SVD residual of the augmented system
    /// `[X; x_*ᵀ; √λ·S½]·β ≈ [y; 0; 0]`, taken as `‖t − U·Uᵀt‖²`. At a computed `β` the
    /// sum of squares exceeds the minimum by `δᵀAδ ≤ (γ·κ·scale)²`, with
    /// `scale = ‖t‖ + σ_max·‖β₀‖`, and its rounding moves it by `γ·scale` along the
    /// residual. The SVD residual carries the same band, so the difference is allowed
    /// it twice.
    #[test]
    fn the_penalized_rss_survives_a_residual_below_the_rounding_of_yty_2280() {
        use gam_linalg::faer_ndarray::FaerSvd;
        let (n, p) = (45usize, 8usize);
        let (x, wobble, s) = gauss_reml_fixture(n, p);
        let u_design = x
            .svd(true, false)
            .expect("design SVD")
            .0
            .expect("left singular vectors");
        let off_design = &wobble - &u_design.dot(&u_design.t().dot(&wobble));
        let direction = &off_design / off_design.dot(&off_design).sqrt();
        let mut beta0 = Array1::<f64>::zeros(p);
        beta0[1] = 1.0;
        let residual_norm = 1.0e-8;
        let y = x.dot(&beta0) + &(&direction * residual_norm);
        let x_star = cosine_row(p, 0.5);
        let resp = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");
        let test_leak = x_star.dot(&beta0);
        let expected_minimum = residual_norm * residual_norm + test_leak * test_leak;

        for &rho in &[-2.0_f64, 0.0] {
            let lambda = rho.exp();
            let mut augmented = Array2::<f64>::zeros((n + 1 + p, p));
            for i in 0..n {
                for j in 0..p {
                    augmented[[i, j]] = x[[i, j]];
                }
            }
            for j in 0..p {
                augmented[[n, j]] = x_star[j];
                augmented[[n + 1 + j, j]] = (lambda * s[[j, j]]).sqrt();
            }
            let mut target = Array1::<f64>::zeros(n + 1 + p);
            for i in 0..n {
                target[i] = y[i];
            }
            let decomposition = augmented.svd(true, false).expect("augmented SVD");
            let u_augmented = decomposition.0.expect("left singular vectors");
            let sigma = decomposition.1;
            let svd_rss = (&target - &u_augmented.dot(&u_augmented.t().dot(&target)))
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let sigma_max = sigma.iter().copied().fold(0.0_f64, f64::max);
            let sigma_min = sigma.iter().copied().fold(f64::INFINITY, f64::min);
            let condition = sigma_max / sigma_min;
            let gamma = gam_linalg::roundoff::accumulation_growth((n + 1 + p) * p * p);
            let scale = target.dot(&target).sqrt() + sigma_max * beta0.dot(&beta0).sqrt();
            let band =
                2.0 * (2.0 * gamma * scale * svd_rss.sqrt() + (gamma * condition * scale).powi(2));

            // The closed form this replaced, on the same arithmetic.
            let mut a_matrix = resp.xtx.clone();
            for i in 0..p {
                for j in 0..p {
                    a_matrix[[i, j]] += lambda * s[[i, j]] + x_star[i] * x_star[j];
                }
            }
            let closed_beta = a_matrix
                .cholesky(Side::Lower)
                .expect("A(λ) is SPD")
                .solvevec(&resp.xty);
            let closed_form = y.dot(&y) - resp.xty.dot(&closed_beta);
            let oracle = resp.penalized_rss(rho, Some(0.0)).expect("penalized RSS");
            println!(
                "[2280-conformal] rho={rho} condition={condition:.3e} \
                 planted={expected_minimum:.6e} svd={svd_rss:.6e} oracle={oracle:.6e} \
                 closed_form={closed_form:.6e} band={band:.3e}"
            );

            assert!(
                (svd_rss - expected_minimum).abs() <= band,
                "the SVD instrument must recover the planted minimum {expected_minimum:.6e}: got \
                 {svd_rss:.6e} (band {band:.3e})"
            );
            assert!(
                (closed_form - svd_rss).abs() > 0.5 * svd_rss,
                "REGIME: yᵀy − cᵀβ̂ = {closed_form:.6e} must miss the SVD residual {svd_rss:.6e} \
                 by more than half of it, or this fixture does not reach the defect"
            );
            assert!(
                (oracle - svd_rss).abs() <= band,
                "the penalized RSS {oracle:.6e} must match the SVD residual {svd_rss:.6e} \
                 within {band:.3e} at rho={rho}"
            );
        }
    }

    // ── Layer 2 (continuous GLM homotopy) tests ──────────────────────────

    /// Independent damped-Newton refit of the augmented canonical GLM at a
    /// single candidate z — explicit per-row loops, its own line search, no
    /// shared assembly with the engine under test.
    fn oracle_glm_refit(
        x: &Array2<f64>,
        y: &Array1<f64>,
        s: &Array2<f64>,
        x_star: &Array1<f64>,
        z: f64,
        mean: &dyn Fn(f64) -> f64,
        weight: &dyn Fn(f64) -> f64,
        nll_term: &dyn Fn(f64, f64) -> f64,
    ) -> Array1<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let pen_nll = |b: &Array1<f64>| -> f64 {
            let mut acc = 0.0;
            for i in 0..n {
                acc += nll_term(x.row(i).dot(b), y[i]);
            }
            acc += nll_term(x_star.dot(b), z);
            acc + 0.5 * b.dot(&s.dot(b))
        };
        let mut beta = Array1::<f64>::zeros(p);
        let mut cur = pen_nll(&beta);
        for _ in 0..400 {
            let mut g = s.dot(&beta);
            let mut h = s.clone();
            for i in 0..n {
                let eta = x.row(i).dot(&beta);
                let r = mean(eta) - y[i];
                let w = weight(eta);
                for a in 0..p {
                    g[a] += x[[i, a]] * r;
                    for b in 0..p {
                        h[[a, b]] += w * x[[i, a]] * x[[i, b]];
                    }
                }
            }
            let eta_s = x_star.dot(&beta);
            let r_s = mean(eta_s) - z;
            let w_s = weight(eta_s);
            for a in 0..p {
                g[a] += x_star[a] * r_s;
                for b in 0..p {
                    h[[a, b]] += w_s * x_star[a] * x_star[b];
                }
            }
            let chol = h.cholesky(Side::Lower).expect("oracle chol");
            let step = chol.solvevec(&g);
            if vec_norm(&step) <= 1e-13 * (1.0 + vec_norm(&beta)) {
                break;
            }
            let search = backtracking_line_search::<_, std::convert::Infallible>(
                BacktrackConfig::default(),
                |t| {
                    let mut cand = beta.clone();
                    cand.scaled_add(-t, &step);
                    let cand_nll = pen_nll(&cand);
                    Ok(if cand_nll.is_finite() {
                        Some((cand_nll, cand))
                    } else {
                        None
                    })
                },
                |_, cand_nll| cand_nll <= cur,
            );
            let accepted = match search {
                Ok(step) => step,
                Err(never) => match never {},
            };
            let step = accepted.unwrap_or_else(|| panic!("oracle line search failed at z={z}"));
            beta = step.payload;
            cur = step.value;
        }
        beta
    }

    /// Conformal membership computed directly from an oracle refit.
    fn oracle_glm_membership(
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_star: &Array1<f64>,
        z: f64,
        alpha: f64,
        beta: &Array1<f64>,
        mean: &dyn Fn(f64) -> f64,
    ) -> bool {
        let n = x.nrows();
        let e_star = (z - mean(x_star.dot(beta))).abs();
        let count = (0..n)
            .filter(|&i| (y[i] - mean(x.row(i).dot(beta))).abs() >= e_star)
            .count();
        (1.0 + count as f64) > alpha * (n as f64 + 1.0)
    }

    /// (#942 Layer 2 test a) The tracked β̂(z) path must match a direct
    /// augmented refit at every candidate WITHIN THE CERTIFIED corrector
    /// bound — for both supported families — and the homotopy must have
    /// actually tracked (not silently cold-refit everything). Membership
    /// verdicts must agree with the independent oracle exactly.
    #[test]
    fn glm_homotopy_tracks_exact_refit_path_within_certified_bound() {
        use std::f64::consts::PI;
        let n = 16usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (j as f64 * PI * t).cos();
            }
            y[i] = (1.0 + (2.0 * PI * t).sin()).exp().round();
        }
        let mut s = Array2::<f64>::eye(p);
        s *= 1.5;
        let weights = Array1::<f64>::ones(n);
        let x_star = cosine_row(p, 0.37);
        let alpha = 0.2;

        // Poisson-log arm over a count window.
        let eng = GlmHomotopyFullConformal::new(
            CanonicalGlmFamily::PoissonLog,
            &x,
            &y,
            &weights,
            &s,
            &x_star,
        )
        .expect("poisson engine");
        let candidates: Vec<f64> = (0..=6).map(|k| k as f64).collect();
        let set = eng.prediction_set(&candidates, alpha).expect("poisson set");
        assert_eq!(set.candidates.len(), candidates.len());
        assert_eq!(set.n_augmented, n + 1);
        assert!(
            set.candidates.iter().skip(1).any(|c| !c.cold_refit),
            "the homotopy never tracked a single transition on a benign Poisson fixture \
             — the certified predictor–corrector path is vacuous"
        );
        let mean_p = |eta: f64| eta.exp();
        let weight_p = |eta: f64| eta.exp();
        let nll_p = |eta: f64, yv: f64| eta.exp() - yv * eta;
        for c in &set.candidates {
            let beta_ref = oracle_glm_refit(&x, &y, &s, &x_star, c.z, &mean_p, &weight_p, &nll_p);
            let mut diff = c.beta.clone();
            diff.scaled_add(-1.0, &beta_ref);
            let err = vec_norm(&diff);
            assert!(
                c.beta_error_bound.is_finite(),
                "certified bound must be finite on a benign fixture (z={})",
                c.z
            );
            assert!(
                err <= c.beta_error_bound + 1e-7,
                "tracked β̂({}) is {err} from the oracle refit, exceeding the certified \
                 corrector bound {} (+ oracle tolerance)",
                c.z,
                c.beta_error_bound
            );
            assert!(
                c.beta_error_bound < 1e-6,
                "certified bound {} at z={} is uselessly loose on a benign fixture",
                c.beta_error_bound,
                c.z
            );
            let member_ref = oracle_glm_membership(&x, &y, &x_star, c.z, alpha, &beta_ref, &mean_p);
            assert_eq!(
                c.member, member_ref,
                "homotopy membership disagrees with the oracle refit at z={}",
                c.z
            );
        }
        assert_eq!(
            set.members.len(),
            set.candidates.iter().filter(|c| c.member).count()
        );

        // Bernoulli-logit arm: support {0, 1}, same path-vs-refit contract.
        let mut yb = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            yb[i] = f64::from(u8::from((2.0 * PI * t).sin() > -0.2));
        }
        let engb = GlmHomotopyFullConformal::new(
            CanonicalGlmFamily::BernoulliLogit,
            &x,
            &yb,
            &weights,
            &s,
            &x_star,
        )
        .expect("bernoulli engine");
        let setb = engb
            .prediction_set(&[0.0, 1.0], alpha)
            .expect("bernoulli set");
        let mean_b = |eta: f64| 1.0 / (1.0 + (-eta).exp());
        let weight_b = |eta: f64| {
            let mu = 1.0 / (1.0 + (-eta).exp());
            mu * (1.0 - mu)
        };
        let nll_b = |eta: f64, yv: f64| eta.max(0.0) + (-eta.abs()).exp().ln_1p() - yv * eta;
        assert!(
            !setb.candidates[1].cold_refit,
            "the logistic third derivative is globally ≤ 1/(6√3); tracking 0→1 must certify"
        );
        for c in &setb.candidates {
            let beta_ref = oracle_glm_refit(&x, &yb, &s, &x_star, c.z, &mean_b, &weight_b, &nll_b);
            let mut diff = c.beta.clone();
            diff.scaled_add(-1.0, &beta_ref);
            assert!(
                vec_norm(&diff) <= c.beta_error_bound + 1e-7,
                "Bernoulli tracked path off the refit at z={} beyond the certified bound",
                c.z
            );
            let member_ref =
                oracle_glm_membership(&x, &yb, &x_star, c.z, alpha, &beta_ref, &mean_b);
            assert_eq!(c.member, member_ref);
        }
    }

    /// Exchangeability needs weights that ARE one, not weights near one. A
    /// tolerance test `|w − 1| > tol` is also false for NaN, so it admitted a
    /// weight that is not a number as unity; the exact comparison refuses both.
    #[test]
    fn full_conformal_admits_only_exact_unit_prior_weights() {
        let n = 6usize;
        let p = 2usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = i as f64;
            y[i] = (i % 3) as f64;
        }
        let s = Array2::<f64>::eye(p);
        let x_star = cosine_row(p, 0.37);
        for not_one in [1.0 + 1.0e-13, f64::NAN] {
            let mut weights = Array1::<f64>::ones(n);
            weights[2] = not_one;
            let error = GlmHomotopyFullConformal::new(
                CanonicalGlmFamily::PoissonLog,
                &x,
                &y,
                &weights,
                &s,
                &x_star,
            )
            .err()
            .expect("a prior weight that is not exactly one must be refused");
            assert!(error.contains("unit prior weights"), "{error}");
        }
        // Non-vacuity: exact unit weights are admitted.
        let weights = Array1::<f64>::ones(n);
        assert!(
            GlmHomotopyFullConformal::new(
                CanonicalGlmFamily::PoissonLog,
                &x,
                &y,
                &weights,
                &s,
                &x_star,
            )
            .is_ok()
        );
    }

    /// (#1192) A benign UNPENALIZED Poisson fixture must produce a valid
    /// conformal set: the cold fit drives the raw penalized gradient down to
    /// its floating-point round-off floor (~1e-7 at moderate n), where the
    /// Armijo line search can no longer make sufficient-decrease progress
    /// because the convex NLL is flat to machine precision. That stalled
    /// iterate IS stationary and must be ACCEPTED, not aborted with a spurious
    /// "cold fit did not converge". With `S = 0` there is no penalty curvature
    /// to suppress the gradient floor, so this is the regime that exposed the
    /// abort.
    #[test]
    fn glm_homotopy_unpenalized_poisson_accepts_roundoff_floor_cold_fit() {
        use std::f64::consts::PI;
        let n = 24usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (j as f64 * PI * t).cos();
            }
            y[i] = (1.0 + (2.0 * PI * t).sin()).exp().round();
        }
        // Unpenalized: no ridge to bound the gradient floor away from ε.
        let s = Array2::<f64>::zeros((p, p));
        let weights = Array1::<f64>::ones(n);
        let x_star = cosine_row(p, 0.37);
        let alpha = 0.2;

        let eng = GlmHomotopyFullConformal::new(
            CanonicalGlmFamily::PoissonLog,
            &x,
            &y,
            &weights,
            &s,
            &x_star,
        )
        .expect("poisson engine");
        let candidates: Vec<f64> = (0..=6).map(|k| k as f64).collect();
        let set = eng
            .prediction_set(&candidates, alpha)
            .expect("unpenalized poisson cold fit must converge to the round-off floor");
        assert_eq!(set.candidates.len(), candidates.len());

        // Every accepted cold fit must be a GENUINE stationary point: agree
        // with an independent oracle refit to within the certified bound.
        let mean_p = |eta: f64| eta.exp();
        let weight_p = |eta: f64| eta.exp();
        let nll_p = |eta: f64, yv: f64| eta.exp() - yv * eta;
        for c in &set.candidates {
            let beta_ref = oracle_glm_refit(&x, &y, &s, &x_star, c.z, &mean_p, &weight_p, &nll_p);
            let mut diff = c.beta.clone();
            diff.scaled_add(-1.0, &beta_ref);
            assert!(
                vec_norm(&diff) <= c.beta_error_bound + 1e-6,
                "accepted β̂({}) is off the oracle refit beyond the certified bound",
                c.z
            );
            let member_ref = oracle_glm_membership(&x, &y, &x_star, c.z, alpha, &beta_ref, &mean_p);
            assert_eq!(
                c.member, member_ref,
                "unpenalized membership disagrees with oracle at z={}",
                c.z
            );
        }
        assert_eq!(
            set.members.len(),
            set.candidates.iter().filter(|c| c.member).count()
        );
    }

    /// (#1192) The round-off-floor acceptance must NOT silently swallow a
    /// genuinely non-stationary iterate: a fit deliberately truncated far
    /// from the optimum (gradient orders of magnitude above the round-off
    /// floor) must still be REJECTED. Guards against turning the fix into a
    /// blanket "accept anything that stalls".
    #[test]
    fn glm_homotopy_truncated_fit_still_rejected() {
        use std::f64::consts::PI;
        let n = 24usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (j as f64 * PI * t).cos();
            }
            y[i] = (1.0 + (2.0 * PI * t).sin()).exp().round();
        }
        let s = Array2::<f64>::zeros((p, p));
        let weights = Array1::<f64>::ones(n);
        let x_star = cosine_row(p, 0.37);
        let eng = GlmHomotopyFullConformal::new(
            CanonicalGlmFamily::PoissonLog,
            &x,
            &y,
            &weights,
            &s,
            &x_star,
        )
        .expect("poisson engine");
        // β = 0 is far from the optimum: a large raw gradient, not the floor.
        let beta0 = Array1::<f64>::zeros(p);
        assert!(
            !eng.kkt_converged(&beta0, 3.0, GLM_STALL_ACCEPT_RTOL),
            "a far-from-stationary iterate must NOT pass the near-stationary band"
        );
    }

    /// (#942 Layer 2 test c) When the third-order bound explodes — a huge
    /// candidate jump at a high-leverage test row under Poisson-log, where
    /// `b‴ = eʸ` grows with the candidate — the step certificate must
    /// REFUSE within its budget and fall back to a cold refit, and the
    /// fallback must preserve exactness (memberships still equal the
    /// independent oracle's).
    #[test]
    fn glm_homotopy_certificate_refuses_and_falls_back_on_third_order_explosion() {
        use std::f64::consts::PI;
        let n = 16usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (j as f64 * PI * t).cos();
            }
            y[i] = (1.0 + 2.0 * t).round();
        }
        let mut s = Array2::<f64>::eye(p);
        s *= 0.5;
        let weights = Array1::<f64>::ones(n);
        let mut x_star = cosine_row(p, 0.31);
        x_star.mapv_inplace(|v| 6.0 * v);
        let eng = GlmHomotopyFullConformal::new(
            CanonicalGlmFamily::PoissonLog,
            &x,
            &y,
            &weights,
            &s,
            &x_star,
        )
        .expect("engine");
        let alpha = 0.2;
        let set = eng
            .prediction_set(&[1.0, 2000.0], alpha)
            .expect("set under extreme jump");
        assert!(
            set.refit_fallbacks >= 1,
            "a 1 → 2000 Poisson candidate jump at ‖x_*‖ = {} must exhaust the certified \
             step budget (b‴ = eʸ explodes along the path) and fall back to a cold refit; \
             got {} fallbacks",
            x_star.dot(&x_star).sqrt(),
            set.refit_fallbacks
        );
        assert!(
            set.candidates[1].cold_refit,
            "the candidate decided through the fallback must be marked cold"
        );
        // Exactness preserved under fallback: the verdicts and coefficients
        // still match the independent oracle within the computed bound.
        let mean_p = |eta: f64| eta.exp();
        let weight_p = |eta: f64| eta.exp();
        let nll_p = |eta: f64, yv: f64| eta.exp() - yv * eta;
        for c in &set.candidates {
            let beta_ref = oracle_glm_refit(&x, &y, &s, &x_star, c.z, &mean_p, &weight_p, &nll_p);
            let mut diff = c.beta.clone();
            diff.scaled_add(-1.0, &beta_ref);
            assert!(
                vec_norm(&diff) <= c.beta_error_bound + 1e-6,
                "fallback coefficients at z={} drifted {} from the oracle refit (bound {})",
                c.z,
                vec_norm(&diff),
                c.beta_error_bound
            );
            let member_ref = oracle_glm_membership(&x, &y, &x_star, c.z, alpha, &beta_ref, &mean_p);
            assert_eq!(
                c.member, member_ref,
                "fallback membership at z={} disagrees with the oracle refit",
                c.z
            );
        }
    }

    /// A v28 or older payload persisted the training `x` and `y` beside `s_lambda` in the
    /// conformal field. It must still load (reading `s_lambda` alone), and the
    /// field it re-serializes to carries no per-row data.
    #[test]
    fn legacy_substrate_with_training_rows_loads_as_penalty_only() {
        let n = 7usize;
        let p = 3usize;
        let x = Array2::<f64>::from_shape_fn((n, p), |(i, j)| (i * p + j) as f64 * 0.1);
        let y = Array1::<f64>::from_shape_fn(n, |i| i as f64);
        let s_lambda = Array2::<f64>::from_shape_fn((p, p), |(i, j)| if i == j { 2.0 } else { 0.0 });
        let legacy = serde_json::json!({
            "x": serde_json::to_value(&x).expect("x"),
            "y": serde_json::to_value(&y).expect("y"),
            "s_lambda": serde_json::to_value(&s_lambda).expect("s_lambda"),
        });

        let penalty: ExactFullConformalPenalty =
            serde_json::from_value(legacy).expect("a v28 conformal substrate must load");
        assert_eq!(penalty.p(), p);

        let reencoded = serde_json::to_value(&penalty).expect("serialize penalty");
        let mut keys: Vec<&str> = reencoded
            .as_object()
            .expect("penalty JSON object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["penalty_count", "s_lambda"],
            "only the p x p penalty and its count are persisted"
        );
        assert_eq!(penalty.penalty_count, None, "a legacy payload records no count");

        let substrate = penalty
            .with_labeled_rows(x.clone(), y.clone())
            .expect("labeled rows of width p join the penalty");
        assert_eq!(substrate.n(), n);
        // Without the count the re-selecting map is unknown: the row is refused
        // loudly and gets the frozen set, never a silent guarantee.
        let row = substrate
            .interval(&Array1::from_vec(vec![0.3, 0.1, 0.2]), 0.2)
            .expect("legacy row");
        assert_eq!(
            row.certificate,
            ConformalCertificate::Refused(ConformalRefusal::UnknownPenaltyStructure)
        );
        assert!(penalty.with_labeled_rows(x.slice(ndarray::s![.., ..2]).to_owned(), y).is_err());
    }

    /// gam#3451: an unpenalised Poisson fit whose test row sits at its own fitted
    /// mean, `z = μ̂_*`, is the augmented optimum with the training optimum's β̂:
    /// the test row's residual is zero and the training score cancels to rounding.
    /// A stationarity scale built from that score and residual is itself rounding,
    /// so the certificate refused the exact solution. The scale is the operands'
    /// norms; `‖Xᵀy‖` alone is at least `Σy` through the intercept column.
    #[test]
    fn unpenalised_optimum_at_its_own_fitted_test_mean_certifies_3451() {
        use std::f64::consts::PI;
        let n = 16usize;
        let p = 2usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            x[[i, 0]] = 1.0;
            x[[i, 1]] = (PI * t).cos();
            y[i] = (1.0 + (2.0 * PI * t).sin()).exp().round();
        }
        let s = Array2::<f64>::zeros((p, p));
        let weights = Array1::<f64>::ones(n);
        let no_test_row = Array1::<f64>::zeros(p);
        let training = GlmHomotopyFullConformal::new(
            CanonicalGlmFamily::PoissonLog,
            &x,
            &y,
            &weights,
            &s,
            &no_test_row,
        )
        .expect("training engine");
        // Newton on the convex training likelihood; a zero test row adds nothing.
        let mut beta = Array1::<f64>::zeros(p);
        for _ in 0..50 {
            let g = training.penalized_score(&beta, 0.0);
            let step = training
                .penalized_hessian(&beta)
                .cholesky(Side::Lower)
                .expect("training Hessian SPD")
                .solvevec(&g);
            beta -= &step;
        }

        let x_star = cosine_row(p, 0.37);
        let eng = GlmHomotopyFullConformal::new(
            CanonicalGlmFamily::PoissonLog,
            &x,
            &y,
            &weights,
            &s,
            &x_star,
        )
        .expect("augmented engine");
        let z = CanonicalGlmFamily::PoissonLog.mean(x_star.dot(&beta));
        let y_total: f64 = y.sum();
        let scale = eng.gradient_natural_scale(&beta, z);
        assert!(
            scale >= y_total,
            "the stationarity scale {scale:.3e} fell below Σy = {y_total}: it is built from \
             the cancelled score, not from its operands"
        );
        assert!(
            eng.kkt_converged(&beta, z, GLM_CONVERGENCE_RTOL),
            "the exact augmented optimum must certify: |g|={:.3e}, scale={scale:.3e}",
            vec_norm(&eng.penalized_score(&beta, z))
        );
        let (refit, _) = eng
            .cold_fit(z, Array1::<f64>::zeros(p))
            .expect("the cold refit at z = μ̂_* must certify");
        let gap = vec_norm(&(&refit - &beta));
        assert!(
            gap <= 1e-9 * (1.0 + vec_norm(&beta)),
            "the cold refit at z = μ̂_* must return the training optimum (gap {gap:.3e})"
        );
    }
}
