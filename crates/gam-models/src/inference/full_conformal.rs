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
//! - **Layer 2 — GLM families (implemented in
//!   [`super::full_conformal_glm`], certified):** Binomial, Poisson,
//!   negative binomial and Gamma at the frozen penalty. Discrete supports
//!   are enumerated exactly (with a certified tail for count families), the
//!   Gamma continuum is walked with certified Newton refits, and ties among
//!   exchangeable scores are broken by the randomized smoothed p-value so
//!   the coverage is exactly `1 − α`, not conservatively above it.
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
//! # Layer 2: GLM families (see [`super::full_conformal_glm`])
//!
//! `β̂(z)` solves the augmented penalized score equation of the family at
//! the frozen Sλ. Discrete families (Binomial, Poisson, negative binomial)
//! are FINITE or have a provable tail: full conformal is exact by
//! enumerating the response support with one certified Newton refit per
//! candidate. Gamma is continuous and is walked with certified refits. The
//! predict route and the Python `glm_full_conformal` instrument both call
//! that one engine.
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
//! conformal (`conformal_level`), and the dispatcher picks: Gaussian-identity
//! fits get Layer 3 with its certificate per row; Binomial, Poisson, negative
//! binomial and Gamma log/logit fits get the enumeration / certified-walk arm
//! of [`super::full_conformal_glm`] at the frozen penalty. Prior weights and
//! unsupported regimes are refused with a typed error naming split conformal,
//! never silently — an invalid guarantee is worse than a wider valid one.

use faer::Side;
use ndarray::{Array1, Array2};

use gam_linalg::faer_ndarray::{FaerCholesky, fast_av};

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

/// The rank threshold `τ = α(n + 1)` a conformal p-value is compared with
/// (member iff `(1 + #dominating) > τ`).
///
/// `α` arrives as `1 − level` from a decimal level, and neither the level nor
/// the subtraction is exact in binary: at the nominal `level = 0.9` the product
/// is `5.999…` rather than `6`, which would let one more rank in and over-cover
/// by exactly `1/(n + 1)`. The two roundings put `α` within `ε` of the decimal
/// it stands for, and the product adds `ε·τ/2`; a `τ` that close to an integer
/// is that integer.
pub fn conformal_rank_threshold(alpha: f64, n_augmented: usize) -> f64 {
    let n1 = n_augmented as f64;
    let tau = alpha * n1;
    let nearest = tau.round();
    if (tau - nearest).abs() <= f64::EPSILON * (n1 + tau) {
        nearest
    } else {
        tau
    }
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
    let threshold = conformal_rank_threshold(alpha, n + 1);
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
        (1.0 + self.dominating_count(z) as f64) > conformal_rank_threshold(alpha, self.n + 1)
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

    /// Wrap an already-recovered frozen penalty. The GLM arm recovers it from
    /// the observed-information Gram with
    /// [`super::full_conformal_glm::penalty_from_normal_and_gram`].
    pub fn from_s_lambda(s_lambda: Array2<f64>, penalty_count: usize) -> Result<Self, String> {
        if s_lambda.nrows() != s_lambda.ncols() {
            return Err("exact full conformal penalty: Sλ is not square".to_string());
        }
        Ok(Self {
            s_lambda,
            penalty_count: Some(penalty_count),
        })
    }

    /// The frozen penalty `Sλ` (p × p).
    pub fn s_lambda(&self) -> &Array2<f64> {
        &self.s_lambda
    }

    /// Coefficient dimension `p`.
    pub fn p(&self) -> usize {
        self.s_lambda.nrows()
    }

    /// Number of smoothing parameters the fit selected; `None` for a payload
    /// written before the count was persisted.
    pub fn penalty_count(&self) -> Option<usize> {
        self.penalty_count
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
            let error = ExactGaussianFullConformal::new(&x, &y, &weights, &s, &x_star)
                .err()
                .expect("a prior weight that is not exactly one must be refused");
            assert!(error.contains("unit prior weights"), "{error}");
        }
        // Non-vacuity: exact unit weights are admitted.
        let weights = Array1::<f64>::ones(n);
        assert!(ExactGaussianFullConformal::new(&x, &y, &weights, &s, &x_star).is_ok());
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
}
