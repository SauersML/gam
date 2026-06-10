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
//! - **Layer 2 (contract below):** GLM families via certified
//!   predictor–corrector homotopy in `z` — exact at corrector points
//!   because each correction is a Newton solve of the SAME symmetric KKT
//!   system a cold fit would solve.
//! - **Layer 3 (the research core, contract below):** the smoothing
//!   response dρ̂/dz through the exact outer IFT — the first full-conformal
//!   procedure that re-selects smoothing per candidate — plus the
//!   **frozen-ρ certificate**: a per-dataset computable bound proving (or
//!   refusing to prove) that freezing ρ̂ cannot change the returned set.
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
//! # Layer 2 contract: GLM homotopy (delegate, formulas fixed here)
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
//! `c`-array bounds ‖D_βH[v]‖ along the step, giving a computable Newton
//! attraction radius) — the corrector cannot silently skip a basin. Score
//! crossings between steps are localized by bisection on the corrected
//! path. Discrete families (Binomial, Poisson) are FINITE: z walks the
//! response support with warm starts, and full conformal is exact by
//! enumeration — no homotopy subtlety at all; implement that arm first.
//!
//! # Layer 3 contract: the ρ-response and the frozen-ρ certificate
//!
//! The honest fitting map re-selects ρ̂ on the augmented data. Joint
//! stationarity in (β, ρ):
//!
//! ```text
//!   F(β, ρ; z) = 0                       (inner KKT, as above)
//!   G(ρ; z)    = ∇_ρ V(ρ; z) = 0          (outer REML/LAML stationarity)
//! ```
//!
//! One outer IFT step gives the smoothing response to the candidate:
//!
//! ```text
//!   dρ̂/dz = − [∇²_ρρ V]⁻¹ · ∂G/∂z ,
//!   ∂G_k/∂z = ∂²V/∂ρ_k∂z |_{β̂}  +  ⟨ ∂²V/∂ρ_k∂β , β̇_z ⟩ ,   β̇_z = H⁻¹x_*
//! ```
//!
//! Every ingredient already exists in this engine and (today) nowhere
//! else: the exact outer Hessian `∇²_ρρV` (#740 machinery), the mixed
//! ρ×β blocks (the drift/correction vectors of the gradient assembly —
//! after #931 these are the shared `ThetaDirection` channels), and the
//! factored `H⁻¹` (the #935 sensitivity operator). The full-path
//! derivative of the fit is then
//!
//! ```text
//!   dμ̂/dz = Xᵀ-row · ( β̇_z + (dβ̂/dρ) · dρ̂/dz )
//! ```
//!
//! and the homotopy of Layer 2 extends one level up, with EVENTS now of
//! three kinds: score crossings (set boundary candidates), ρ box-bound
//! activation (active-set strata — freeze the bound coordinate, continue),
//! and REML basin jumps (corrector lands on a different local optimum).
//! Basin jumps are where naive path-tracking would silently break the
//! symmetry requirement; the discipline is: the DEFINED fitting map is the
//! deterministic seed-path optimizer (#969), the homotopy is only an
//! acceleration of it, and whenever the corrector cannot certify it is in
//! the cold map's basin (objective-value cross-check after correction),
//! the implementation falls back to a cold deterministic fit at that z.
//! Validity is therefore inherited from the cold map's symmetry — the
//! homotopy can be wrong only about SPEED, never about the answer.
//!
//! ## The frozen-ρ certificate (the deliverable that matters for everyone)
//!
//! For the cheap procedure that freezes ρ̂ at the original-data optimum,
//! the per-dataset certificate bounds the score perturbation along the
//! candidate range Z:
//!
//! ```text
//!   |e_i(z; ρ̂(z)) − e_i(z; ρ̂_frozen)| ≤  L_i · sup_{z∈Z} ‖ρ̂(z) − ρ̂_frozen‖
//! ```
//!
//! with `L_i = sup ‖∂e_i/∂ρ‖` from the SAME sensitivity operator
//! (`∂μ̂/∂ρ = X dβ̂/dρ`, one batched solve), and the ρ-excursion bounded by
//! integrating ‖dρ̂/dz‖ (computed, not assumed). If the bound is smaller
//! than the MARGIN of every rank comparison that decides the set's
//! boundary intervals — `min over deciding pairs |e_i(z) − e_*(z)|` at the
//! Layer-1 breakpoints, a quantity the exact engine below already produces
//! — then the frozen-ρ set EQUALS the honest set, certified, and the
//! expensive path is skipped. When the certificate fails, the procedure
//! says so and runs Layer 3 instead of silently returning an uncertified
//! set. This converts the folk approximation used by every existing
//! "efficient full conformal" method into a checked one — per dataset,
//! with computed constants, no asymptotics.
//!
//! # Wiring (magic-by-default, certificate-first)
//!
//! No flags. The predict path requests full conformal exactly like split
//! conformal (`conformal_level`), and the dispatcher picks: exact Layer 1
//! for Gaussian-identity fits, enumeration for discrete families, homotopy
//! beyond. PRIORITY ORDER MATTERS and is a design decision, not an
//! optimization: the cheap frozen-ρ exact set runs FIRST, the certificate
//! is computed, and only on certificate REFUSAL does the engine touch the
//! expensive honest path — and even then the preferred realization is
//! cold deterministic refits at the few z-regions whose membership the
//! certificate could not pin (the breakpoint structure localizes them),
//! with the dρ̂/dz IFT used to BOUND the excursion, not to continuously
//! track it. Continuous ρ-path-tracking is the last resort, not the
//! default — the certificate makes it almost always unnecessary, and a
//! bound-plus-local-refit design has no basin-tracking failure mode at
//! all. Unit-weight violation and unsupported regimes fall back to the
//! split/ALO calibrator LOUDLY (logged), never silently — an invalid
//! guarantee is worse than a wider valid one.

use faer::Side;
use ndarray::{Array1, Array2};

use crate::faer_ndarray::{FaerCholesky, fast_av};

/// One maximal interval of candidate values retained in the prediction set.
/// Endpoints may be infinite (honest unboundedness in low-information /
/// high-leverage regimes).
#[derive(Clone, Debug, PartialEq)]
pub struct ConformalInterval {
    pub lo: f64,
    pub hi: f64,
}

/// The exact full-conformal prediction set: a finite union of closed
/// intervals, plus the diagnostics the Layer-3 certificate consumes.
#[derive(Clone, Debug)]
pub struct FullConformalSet {
    /// Maximal intervals, sorted, disjoint.
    pub intervals: Vec<ConformalInterval>,
    /// Miscoverage level the set was built for.
    pub alpha: f64,
    /// `n + 1` (augmented count) — the denominator of the conformal rank.
    pub n_augmented: usize,
    /// The decision margin: the smallest |e_i − e_*| gap over the
    /// comparisons that decide membership at the set's boundary
    /// breakpoints. This is the quantity the frozen-ρ certificate compares
    /// its score-perturbation bound against: a perturbation below this
    /// margin cannot flip any deciding comparison, hence cannot move the
    /// set. `+∞` when the set is all of ℝ (nothing to flip).
    pub boundary_margin: f64,
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
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n || prior_weights.len() != n {
            return Err("full conformal: row-count mismatch".to_string());
        }
        if s_lambda.nrows() != p || s_lambda.ncols() != p || x_star.len() != p {
            return Err("full conformal: column-count mismatch".to_string());
        }
        if prior_weights.iter().any(|&w| (w - 1.0).abs() > 1e-12) {
            return Err(
                "full conformal requires unit prior weights: a reweighted training row is \
                 not exchangeable with the test row, so the finite-sample coverage proof \
                 does not apply; use the split/ALO conformal calibrator instead"
                    .to_string(),
            );
        }

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
            if d.abs() > 0.0 {
                roots.push((self.u[i] - us) / d);
            }
            // r_* + r_i = (us + u_i) + (ws + w_i) z = 0
            let s = ws + self.w[i];
            if s.abs() > 0.0 {
                roots.push(-(us + self.u[i]) / s);
            }
        }
        roots.retain(|r| r.is_finite());
        roots.sort_by(|p, q| p.partial_cmp(q).expect("finite breakpoints"));
        roots.dedup_by(|p, q| *p == *q);

        // Probe points: each root, each gap midpoint, and the two open
        // tails. Membership is constant strictly between consecutive
        // roots, so this probe set decides the set exactly.
        let mut probes: Vec<f64> = Vec::with_capacity(2 * roots.len() + 3);
        if roots.is_empty() {
            probes.push(0.0);
        } else {
            let span = (roots[roots.len() - 1] - roots[0]).max(1.0);
            probes.push(roots[0] - span);
            for k in 0..roots.len() {
                probes.push(roots[k]);
                if k + 1 < roots.len() {
                    probes.push(0.5 * (roots[k] + roots[k + 1]));
                }
            }
            probes.push(roots[roots.len() - 1] + span);
        }

        // Scan probes into maximal intervals. A member midpoint/tail claims
        // its whole open gap; member roots close the endpoints.
        let mut intervals: Vec<ConformalInterval> = Vec::new();
        let mut open_lo: Option<f64> = None;
        let gap_bounds = |idx: usize| -> (f64, f64) {
            // bounds of the gap a probe at sorted position idx represents
            if roots.is_empty() {
                return (f64::NEG_INFINITY, f64::INFINITY);
            }
            if idx == 0 {
                return (f64::NEG_INFINITY, roots[0]);
            }
            if idx == probes.len() - 1 {
                return (roots[roots.len() - 1], f64::INFINITY);
            }
            // probes alternate root, mid, root, mid, ... after the first
            let k = (idx - 1) / 2; // gap index for midpoints, root index for roots
            if idx % 2 == 1 {
                // a root: zero-width "gap" at the root itself
                (roots[k], roots[k])
            } else {
                (roots[k], roots[k + 1])
            }
        };
        for (idx, &z) in probes.iter().enumerate() {
            let inside = self.member(z, alpha);
            let (lo, hi) = gap_bounds(idx);
            if inside {
                if open_lo.is_none() {
                    open_lo = Some(lo);
                }
                if idx == probes.len() - 1 {
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

        // Decision margin for the frozen-ρ certificate (Layer 3): the
        // smallest score gap among comparisons evaluated at the set's
        // finite boundary points. ρ-perturbations of the scores below this
        // margin provably cannot move the set.
        let mut boundary_margin = f64::INFINITY;
        for itv in &intervals {
            for endpoint in [itv.lo, itv.hi] {
                if endpoint.is_finite() {
                    let e_star = (us + ws * endpoint).abs();
                    for i in 0..n {
                        let gap = ((self.u[i] + self.w[i] * endpoint).abs() - e_star).abs();
                        if gap > 0.0 && gap < boundary_margin {
                            boundary_margin = gap;
                        }
                    }
                }
            }
        }

        FullConformalSet {
            intervals,
            alpha,
            n_augmented: n + 1,
            boundary_margin,
        }
    }

    /// Brute-force membership oracle (refit-free here because the affine
    /// algebra IS the refit, but evaluated pointwise): used by tests to
    /// verify the breakpoint scan against dense-grid evaluation.
    pub fn member_at(&self, z: f64, alpha: f64) -> bool {
        self.member(z, alpha)
    }
}

/// Layer-3 certificate verdict for the frozen-ρ shortcut. Produced by
/// comparing the integrated ρ-excursion bound against the exact engine's
/// `boundary_margin` (see module doc). `Certified` means the frozen-ρ set
/// EQUALS the honest re-selecting set — proven for THIS dataset, not
/// assumed; `Refused` carries the two numbers so the caller (and the
/// report) can show exactly how far from certifiable the shortcut was.
#[derive(Clone, Debug)]
pub enum FrozenRhoCertificate {
    Certified {
        score_perturbation_bound: f64,
        boundary_margin: f64,
    },
    Refused {
        score_perturbation_bound: f64,
        boundary_margin: f64,
    },
}

impl FrozenRhoCertificate {
    /// Decide from the two computed constants. Strict inequality: a bound
    /// equal to the margin cannot certify (a comparison could be exactly
    /// flipped).
    pub fn decide(score_perturbation_bound: f64, boundary_margin: f64) -> Self {
        if score_perturbation_bound < boundary_margin {
            FrozenRhoCertificate::Certified {
                score_perturbation_bound,
                boundary_margin,
            }
        } else {
            FrozenRhoCertificate::Refused {
                score_perturbation_bound,
                boundary_margin,
            }
        }
    }
}

#[cfg(test)]
mod tests {
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

        // Margin must be a positive finite diagnostic when boundaries exist.
        assert!(set.boundary_margin >= 0.0);
    }
}
