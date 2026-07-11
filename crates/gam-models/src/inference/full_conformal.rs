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
//! - **Layer 2 — discrete arm (implemented below, exact):** Binomial /
//!   Poisson and any finite-or-windowed response support, by ENUMERATION
//!   with one symmetric refit per candidate (`SymmetricAugmentedFit`).
//!   Bernoulli's honest full-conformal set — smoothing re-selection
//!   included — costs exactly two cold fits; windowed counts carry honest
//!   tail-resolution flags instead of an unprovable monotone-tail
//!   assumption. Exactness is by construction: every retained candidate
//!   was actually refit. Validity is proven in the test module by FULL
//!   ENUMERATION of every Bernoulli dataset at small n (exact coverage
//!   ≥ 1 − α as a theorem check, not a simulation).
//! - **Layer 2 — continuous GLM (implemented below, certified):**
//!   predictor–corrector homotopy in `z` ([`GlmHomotopyFullConformal`]) —
//!   exact at corrector points because each correction is a Newton solve of
//!   the SAME symmetric KKT system a cold fit would solve, with the step
//!   size CERTIFIED by a computed third-derivative contraction bound and a
//!   cold-refit fallback whenever the certificate refuses.
//! - **Layer 3 (the research core, contract below):** the smoothing
//!   response dρ̂/dz through the exact outer IFT — the first full-conformal
//!   procedure that re-selects smoothing per candidate — plus the
//!   **frozen-ρ certificate**: a per-dataset computable bound that can
//!   conditionally accept (or refuse) freezing ρ̂, with the rho-excursion
//!   step explicitly reported as a grid-checked Lipschitz assumption.
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
//! (`∂μ̂/∂ρ = X dβ̂/dρ`, one batched solve). The current implementation
//! checks the ρ-excursion on a fixed probe grid: acceptance is conditional
//! on the true `sup |dρ̂/dz|` over the reported range not exceeding the
//! observed probe maximum by more than the stated mean-value allowance. If
//! that conditional bound is smaller than the MARGIN of every rank
//! comparison that decides the set's boundary intervals — `min over
//! deciding pairs |e_i(z) − e_*(z)|` at the Layer-1 breakpoints, with
//! critical ties contributing zero — then the frozen-ρ set equals the
//! honest set under that grid-checked Lipschitz assumption. When the check
//! fails, the procedure says so and runs Layer 3 instead of silently
//! returning an unchecked set.
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

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_av};
use opt::{BacktrackConfig, backtracking_line_search};

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
    /// The decision margin: the smallest |e_i − e_*| gap over rank
    /// comparisons whose flip can change membership. Critical ties
    /// contribute zero. When the set has no finite boundary (all of ℝ or
    /// empty), the margin is the analytic infimum of the local rank-decision
    /// margin over the whole candidate line; `+∞` is reserved for the case
    /// where membership needs no score comparison at all.
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

    fn required_dominating_count(&self, alpha: f64) -> usize {
        let threshold = alpha * (self.n + 1) as f64;
        for count in 0..=self.n {
            if 1.0 + count as f64 > threshold {
                return count;
            }
        }
        self.n + 1
    }

    fn local_decision_margin(&self, z: f64, alpha: f64) -> f64 {
        let required = self.required_dominating_count(alpha);
        if required == 0 {
            return f64::INFINITY;
        }
        let e_star = (self.u[self.n] + self.w[self.n] * z).abs();
        let mut true_gaps = Vec::new();
        let mut false_gaps = Vec::new();
        for i in 0..self.n {
            let e_i = (self.u[i] + self.w[i] * z).abs();
            let gap = (e_i - e_star).abs();
            if e_i >= e_star {
                true_gaps.push(gap);
            } else {
                false_gaps.push(gap);
            }
        }
        if true_gaps.len() >= required {
            true_gaps.sort_by(|a, b| a.partial_cmp(b).expect("finite score gaps"));
            true_gaps[true_gaps.len() - required]
        } else {
            let needed = required - true_gaps.len();
            false_gaps.sort_by(|a, b| a.partial_cmp(b).expect("finite score gaps"));
            false_gaps.get(needed - 1).copied().unwrap_or(f64::INFINITY)
        }
    }

    fn push_finite_root(points: &mut Vec<f64>, numerator: f64, denominator: f64) {
        if denominator.abs() > 0.0 {
            let z = numerator / denominator;
            if z.is_finite() {
                points.push(z);
            }
        }
    }

    fn abs_residual_affine_at(&self, row: usize, z: f64) -> (f64, f64) {
        let value = self.u[row] + self.w[row] * z;
        let sign = if value >= 0.0 { 1.0 } else { -1.0 };
        (sign * self.w[row], sign * self.u[row])
    }

    fn gap_affines_on_cell(&self, z: f64) -> Vec<(bool, f64, f64)> {
        let (star_slope, star_intercept) = self.abs_residual_affine_at(self.n, z);
        let mut gaps = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let (row_slope, row_intercept) = self.abs_residual_affine_at(i, z);
            let diff_slope = row_slope - star_slope;
            let diff_intercept = row_intercept - star_intercept;
            let diff = diff_slope * z + diff_intercept;
            if diff >= 0.0 {
                gaps.push((true, diff_slope, diff_intercept));
            } else {
                gaps.push((false, -diff_slope, -diff_intercept));
            }
        }
        gaps
    }

    fn asymptotic_abs_residual_affine(&self, row: usize, direction: f64) -> (f64, f64) {
        let slope_in_t = direction * self.w[row];
        let sign = if slope_in_t > 0.0 {
            1.0
        } else if slope_in_t < 0.0 {
            -1.0
        } else if self.u[row] >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (sign * slope_in_t, sign * self.u[row])
    }

    fn asymptotic_decision_margin(&self, direction: f64, alpha: f64) -> f64 {
        let required = self.required_dominating_count(alpha);
        if required == 0 {
            return f64::INFINITY;
        }
        let (star_slope, star_intercept) = self.asymptotic_abs_residual_affine(self.n, direction);
        let mut true_gaps = Vec::new();
        let mut false_gaps = Vec::new();
        for i in 0..self.n {
            let (row_slope, row_intercept) = self.asymptotic_abs_residual_affine(i, direction);
            let diff_slope = row_slope - star_slope;
            let diff_intercept = row_intercept - star_intercept;
            let truth = diff_slope > 0.0 || (diff_slope == 0.0 && diff_intercept >= 0.0);
            let gap = if truth {
                (diff_slope, diff_intercept)
            } else {
                (-diff_slope, -diff_intercept)
            };
            if truth {
                true_gaps.push(gap);
            } else {
                false_gaps.push(gap);
            }
        }
        let critical = if true_gaps.len() >= required {
            true_gaps.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("finite asymptotic slopes")
                    .then_with(|| a.1.partial_cmp(&b.1).expect("finite asymptotic intercepts"))
            });
            true_gaps.get(true_gaps.len() - required).copied()
        } else {
            let needed = required - true_gaps.len();
            false_gaps.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("finite asymptotic slopes")
                    .then_with(|| a.1.partial_cmp(&b.1).expect("finite asymptotic intercepts"))
            });
            false_gaps.get(needed - 1).copied()
        };
        match critical {
            Some((slope, intercept)) if slope == 0.0 => intercept.max(0.0),
            Some(_) => f64::INFINITY,
            None => f64::INFINITY,
        }
    }

    fn margin_without_finite_boundaries(&self, alpha: f64, roots: &[f64]) -> f64 {
        let mut points = roots.to_vec();
        for row in 0..=self.n {
            Self::push_finite_root(&mut points, -self.u[row], self.w[row]);
        }
        points.sort_by(|a, b| a.partial_cmp(b).expect("finite breakpoints"));
        points.dedup_by(|a, b| *a == *b);

        let mut eval_points = points.clone();
        for cell in 0..=points.len() {
            let lo = if cell == 0 {
                f64::NEG_INFINITY
            } else {
                points[cell - 1]
            };
            let hi = if cell == points.len() {
                f64::INFINITY
            } else {
                points[cell]
            };
            let z = if lo.is_finite() && hi.is_finite() {
                0.5 * (lo + hi)
            } else if lo.is_finite() {
                lo + 1.0
            } else if hi.is_finite() {
                hi - 1.0
            } else {
                0.0
            };
            let gaps = self.gap_affines_on_cell(z);
            let required = self.required_dominating_count(alpha);
            if required == 0 {
                continue;
            }
            let true_count = gaps.iter().filter(|g| g.0).count();
            let need_truth = true_count >= required;
            let relevant: Vec<(f64, f64)> = gaps
                .iter()
                .filter(|g| g.0 == need_truth)
                .map(|g| (g.1, g.2))
                .collect();
            for a in 0..relevant.len() {
                for b in (a + 1)..relevant.len() {
                    let denominator = relevant[a].0 - relevant[b].0;
                    if denominator.abs() > 0.0 {
                        let cross = (relevant[b].1 - relevant[a].1) / denominator;
                        if cross.is_finite() && cross > lo && cross < hi {
                            eval_points.push(cross);
                        }
                    }
                }
            }
        }
        eval_points.sort_by(|a, b| a.partial_cmp(b).expect("finite margin points"));
        eval_points.dedup_by(|a, b| *a == *b);

        let mut margin = f64::INFINITY;
        if eval_points.is_empty() {
            margin = margin.min(self.local_decision_margin(0.0, alpha));
        } else {
            for z in eval_points {
                margin = margin.min(self.local_decision_margin(z, alpha));
            }
        }
        margin = margin.min(self.asymptotic_decision_margin(1.0, alpha));
        margin = margin.min(self.asymptotic_decision_margin(-1.0, alpha));
        margin
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

        // Decision margin for the frozen-ρ check (Layer 3): at a finite
        // boundary, evaluate the exact local rank-decision margin. Critical
        // ties contribute zero. If there is no finite boundary (all-R or
        // empty), compute the analytic infimum of the same local quantity
        // over the whole piecewise-linear candidate line.
        let mut finite_endpoints = Vec::new();
        for itv in &intervals {
            for endpoint in [itv.lo, itv.hi] {
                if endpoint.is_finite() {
                    finite_endpoints.push(endpoint);
                }
            }
        }
        let boundary_margin = if finite_endpoints.is_empty() {
            self.margin_without_finite_boundaries(alpha, &roots)
        } else {
            finite_endpoints
                .into_iter()
                .map(|endpoint| self.local_decision_margin(endpoint, alpha))
                .fold(f64::INFINITY, f64::min)
        };

        FullConformalSet {
            intervals,
            alpha,
            n_augmented: n + 1,
            boundary_margin,
        }
    }
}

/// The symmetric augmented fitting map the discrete enumeration arm walks.
///
/// `scores(z)` must: fit the n+1 augmented rows `{(x_i, y_i)} ∪ {(x_*, z)}`
/// and return all n+1 nonconformity scores with the TEST row's score LAST.
/// The single requirement backing the coverage guarantee is SYMMETRY: the
/// fitting map must treat the augmented row exactly like a training row
/// (same loss term, same weight, same participation in any smoothing /
/// hyperparameter selection the map performs). A map that freezes anything
/// it selected by looking at the training responses but not at `z` breaks
/// symmetry and voids the guarantee — for discrete families that honesty is
/// CHEAP, because the support is walked by enumeration (2 refits for
/// Bernoulli), so the map can simply be the full cold fit, ρ-selection
/// included.
///
/// `&mut self` so implementations can warm-start across consecutive
/// candidates (a speed optimization that cannot affect the answer when each
/// solve is run to its deterministic optimum).
pub trait SymmetricAugmentedFit {
    fn scores(&mut self, z: f64) -> Result<Array1<f64>, String>;
}

/// Blanket impl so plain closures can serve as the fitting map (tests, and
/// adapter shims that capture a fit configuration).
impl<F> SymmetricAugmentedFit for F
where
    F: FnMut(f64) -> Result<Array1<f64>, String>,
{
    fn scores(&mut self, z: f64) -> Result<Array1<f64>, String> {
        self(z)
    }
}

/// One enumerated candidate's conformal verdict.
#[derive(Clone, Debug)]
pub struct DiscreteCandidate {
    pub z: f64,
    /// Conformal p-value `(1 + #{i ≤ n : e_i ≥ e_*}) / (n+1)`. Ties count
    /// FOR the candidate (the `≥` convention) — the conservative direction;
    /// strict-inequality ranking would under-cover under ties.
    pub p_value: f64,
    pub member: bool,
}

/// Exact full-conformal prediction set for a DISCRETE response family,
/// computed by enumeration of candidate responses with one symmetric refit
/// per candidate (#942 Layer 2, discrete arm).
///
/// There is no homotopy and no approximation anywhere in this object: for
/// each candidate `z` the fitting map is run to its optimum, the n+1 scores
/// are ranked, and the candidate is kept iff its conformal p-value exceeds
/// α. Validity is the standard full-conformal argument — exchangeability of
/// the n+1 rows plus symmetry of the map — and EXACTNESS is by construction
/// (the support is finite or explicitly windowed; every retained candidate
/// was actually refit).
#[derive(Clone, Debug)]
pub struct DiscreteFullConformalSet {
    /// Retained candidates, ascending.
    pub members: Vec<f64>,
    /// Every enumerated candidate with its p-value (diagnostics; the
    /// boundary-adjacent p-values are the discrete analogue of Layer 1's
    /// `boundary_margin`).
    pub candidates: Vec<DiscreteCandidate>,
    pub alpha: f64,
    /// `n + 1`.
    pub n_augmented: usize,
    /// `Some(z_first)` when the SMALLEST enumerated candidate was a member
    /// of a WINDOWED enumeration — the retained set may continue
    /// contiguously below the window. Always `None` for exhaustive supports
    /// (the Bernoulli arm). For a windowed support, `None` only says the
    /// retained set does not continue through the enumerated edge; absent a
    /// monotone-tail theorem for the fitting map, it says nothing about
    /// non-contiguous retained candidates farther outside the window.
    pub lower_tail_unresolved: Option<f64>,
    /// Mirror of `lower_tail_unresolved` for the largest candidate.
    pub upper_tail_unresolved: Option<f64>,
}

/// Walk an EXHAUSTIVE discrete support (e.g. Bernoulli `{0, 1}`). The
/// returned set is the exact full-conformal set, period — no tail
/// semantics, because there is nothing outside the support.
pub fn discrete_full_conformal_exhaustive<M: SymmetricAugmentedFit>(
    fit: &mut M,
    support: &[f64],
    alpha: f64,
) -> Result<DiscreteFullConformalSet, String> {
    let mut set = discrete_walk(fit, support, alpha)?;
    set.lower_tail_unresolved = None;
    set.upper_tail_unresolved = None;
    Ok(set)
}

/// Walk a WINDOW of an unbounded discrete support (e.g. Poisson counts
/// `lo..=hi`). Exact ON THE WINDOW; the tail flags report honestly whether
/// the retained set continues through either edge (edge candidate retained
/// ⇒ contiguous tail unresolved). An excluded edge resolves only that
/// contiguous continuation. Without a monotone-tail theorem for the fitting
/// map, callers must not interpret cleared flags as a global proof that no
/// non-contiguous retained candidates exist farther outside the window.
pub fn discrete_full_conformal_window<M: SymmetricAugmentedFit>(
    fit: &mut M,
    window: &[f64],
    alpha: f64,
) -> Result<DiscreteFullConformalSet, String> {
    discrete_walk(fit, window, alpha)
}

/// Bernoulli convenience arm: the support is `{0, 1}`, so the honest
/// (ρ-re-selecting) full-conformal set costs exactly two cold fits.
pub fn bernoulli_full_conformal<M: SymmetricAugmentedFit>(
    fit: &mut M,
    alpha: f64,
) -> Result<DiscreteFullConformalSet, String> {
    discrete_full_conformal_exhaustive(fit, &[0.0, 1.0], alpha)
}

fn discrete_walk<M: SymmetricAugmentedFit>(
    fit: &mut M,
    candidates: &[f64],
    alpha: f64,
) -> Result<DiscreteFullConformalSet, String> {
    if candidates.is_empty() {
        return Err("discrete full conformal: empty candidate list".to_string());
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(format!(
            "discrete full conformal: alpha must be in [0, 1), got {alpha}"
        ));
    }
    if candidates.windows(2).any(|w| !(w[0] < w[1])) {
        return Err("discrete full conformal: candidates must be strictly increasing".to_string());
    }

    let mut out = Vec::with_capacity(candidates.len());
    let mut members = Vec::new();
    let mut n_augmented = 0usize;
    for &z in candidates {
        let scores = fit.scores(z)?;
        let n1 = scores.len();
        if n1 < 2 {
            return Err(
                "discrete full conformal: fitting map must score at least two rows".to_string(),
            );
        }
        if n_augmented == 0 {
            n_augmented = n1;
        } else if n_augmented != n1 {
            return Err(format!(
                "discrete full conformal: fitting map returned {n1} scores after returning \
                 {n_augmented}; the augmented row count cannot change across candidates"
            ));
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Err(format!(
                "discrete full conformal: non-finite nonconformity score at candidate {z}; \
                 refusing to rank garbage"
            ));
        }
        let e_star = scores[n1 - 1];
        let count = scores.iter().take(n1 - 1).filter(|&&e| e >= e_star).count();
        let p_value = (1.0 + count as f64) / (n1 as f64);
        let member = p_value > alpha;
        if member {
            members.push(z);
        }
        out.push(DiscreteCandidate { z, p_value, member });
    }

    let lower_tail_unresolved = out.first().filter(|c| c.member).map(|c| c.z);
    let upper_tail_unresolved = out.last().filter(|c| c.member).map(|c| c.z);
    Ok(DiscreteFullConformalSet {
        members,
        candidates: out,
        alpha,
        n_augmented,
        lower_tail_unresolved,
        upper_tail_unresolved,
    })
}

/// Layer-3 verdict for the frozen-ρ shortcut. Produced by comparing the
/// grid-checked ρ-excursion bound against the exact engine's
/// `boundary_margin` (see module doc). `Certified` is conditional on the
/// stated rho-grid Lipschitz assumption; `Refused` carries the two numbers
/// so the caller can show exactly how far from acceptable the shortcut was.
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
    /// equal to the margin cannot certify, and a zero margin can never
    /// certify because no positive perturbation bound is strictly below it.
    pub fn decide(score_perturbation_bound: f64, boundary_margin: f64) -> Self {
        if boundary_margin > 0.0 && score_perturbation_bound < boundary_margin {
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

/// Closed-form Gaussian-REML smoothing-parameter response and the frozen-ρ
/// certificate it powers — the #942 Layer-3 research core, realized exactly
/// for the single-penalty model `Sλ = λ S` (`ρ = log λ`).
///
/// # Why this object exists
///
/// Layer 1 ([`ExactGaussianFullConformal`]) is honest only if ρ is held fixed
/// — but the DEFINED Gaussian fitting map re-selects ρ̂ by REML on whatever
/// data it sees, including the augmented row `(x_*, z)`. Every "efficient
/// full conformal" method in the literature silently freezes ρ̂ at its
/// original-data value and never quantifies the resulting symmetry break.
/// This object closes that gap WITHOUT a homotopy: it computes the honest
/// re-selecting map exactly (it is a 1-D REML problem per candidate), the
/// smoothing response `dρ̂/dz` in closed form via the outer IFT, and a
/// per-dataset conditional check that accepts (or refuses) freezing ρ̂. On
/// acceptance the cheap frozen-ρ set is returned with the rho-grid
/// assumption that makes equality to the honest set valid; on refusal the
/// caller is told, with the two deciding constants, exactly how far short
/// the shortcut fell.
///
/// # The closed forms (single penalty `Sλ = λ S`)
///
/// Augmented penalized least squares with the test row included:
///
/// ```text
///   A(λ)   = XᵀX + x_* x_*ᵀ + λ S         (independent of z)
///   c(z)   = Xᵀy + x_* z ,  β̂ = A(λ)⁻¹ c(z) = a + b z   (affine in z)
///   D(ρ,z) = ‖y_aug‖² − c(z)ᵀ A(λ)⁻¹ c(z)              (penalized RSS)
/// ```
///
/// The Gaussian REML criterion to MINIMIZE over ρ (σ² profiled out, additive
/// constants dropped; `M₀ = nullity(S)`, `r = rank(S)`, `n_eff = n (+1` if the
/// test row is present`)`):
///
/// ```text
///   Ṽ(ρ,z) = (n_eff − M₀) · log D(ρ,z) + log|A(λ)| − r ρ
/// ```
///
/// Its z- and ρ-derivatives are all closed form (`pen = λ β̂ᵀSβ̂`):
///
/// ```text
///   ∂D/∂ρ = pen ,                ∂D/∂z = 2(z − x_*ᵀβ̂) = 2 r_*
///   G    = ∂Ṽ/∂ρ      = (n_eff−M₀)·pen/D + λ tr(A⁻¹S) − r
///   ∂²Ṽ/∂ρ²           = (n_eff−M₀)·(pen'·D − pen²)/D² + λ tr(A⁻¹S) − λ² tr((A⁻¹S)²)
///   ∂²Ṽ/∂ρ∂z          = (n_eff−M₀)·(pen_z'·D − pen·D_z')/D²
/// ```
///
/// with `pen' = pen − 2λ²·β̂ᵀS A⁻¹ S β̂`, `pen_z' = 2λ·β̂ᵀS (A⁻¹x_*)`,
/// `D_z' = 2 r_*`. The smoothing response to the candidate is one outer IFT
/// step on `G(ρ̂(z), z) = 0`:
///
/// ```text
///   dρ̂/dz = − (∂²Ṽ/∂ρ²)⁻¹ · ∂²Ṽ/∂ρ∂z .
/// ```
///
/// The score–ρ sensitivity (which the certificate's Lipschitz constant uses)
/// is `∂μ̂_i/∂ρ = x_iᵀ (dβ̂/dρ)` with `dβ̂/dρ = −λ A⁻¹ S β̂`, so
/// `|∂e_i/∂ρ| = |∂μ̂_i/∂ρ|` (the absolute-residual score's only ρ-dependence
/// is through μ̂). Everything is assembled from ONE Cholesky of `A(λ)` plus a
/// handful of solves.
pub struct GaussianRemlRhoResponse<'a> {
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    s: &'a Array2<f64>,
    x_star: &'a Array1<f64>,
    n: usize,
    p: usize,
    rank_s: usize,
    xtx: Array2<f64>,
    xty: Array1<f64>,
    yty: f64,
}

/// One closed-form evaluation of the (possibly augmented) Gaussian REML
/// criterion at `ρ = log λ`, carrying every derivative the IFT and the
/// certificate consume.
#[derive(Clone, Debug)]
struct RemlEval {
    /// `Ṽ(ρ,z)` (additive constants dropped — only differences in ρ matter).
    value: f64,
    /// `G = ∂Ṽ/∂ρ`.
    grad: f64,
    /// `∂²Ṽ/∂ρ²`.
    hess: f64,
    /// `∂²Ṽ/∂ρ∂z` (0 when the test row is absent).
    cross: f64,
    /// `∂μ̂_i/∂ρ` at the training rows.
    mu_rho_train: Array1<f64>,
    /// `∂μ̂_*/∂ρ` at the test row.
    mu_rho_test: f64,
}

/// The frozen-ρ full-conformal set with its Layer-3 certificate and the
/// constants the certificate decided on.
#[derive(Clone, Debug)]
pub struct CertifiedFullConformal {
    /// The cheap exact set built at the frozen `ρ̂₀` (original-data optimum).
    pub frozen_set: FullConformalSet,
    /// Whether freezing ρ̂ is accepted under the reported rho-grid
    /// Lipschitz assumption.
    pub certificate: FrozenRhoCertificate,
    /// `ρ̂₀ = log λ̂₀` selected by REML on the original (un-augmented) data.
    pub rho_frozen: f64,
    /// Conditional bound on `sup_z |ρ̂(z) − ρ̂₀|` over the finite deciding
    /// range. Zero with `rho_probe_count == 0` means no finite range was
    /// probed.
    pub rho_excursion: f64,
    /// `max_i (|∂μ̂_i/∂ρ| + |∂μ̂_*/∂ρ|)` — the score-gap Lipschitz constant in ρ.
    pub score_rho_lipschitz: f64,
    /// Number of equal-spaced rho-response probes used on the finite
    /// deciding range. Zero means no finite probe range was available.
    pub rho_probe_count: usize,
    /// Largest observed `|dρ̂/dz|` on the rho-response probe grid. This is a
    /// diagnostic, not a continuous supremum proof.
    pub observed_sup_drho_dz: f64,
}

impl<'a> GaussianRemlRhoResponse<'a> {
    /// Build the response object. Computes `rank(S)` once by symmetric
    /// eigendecomposition (relative tolerance on the largest eigenvalue).
    pub fn new(
        x: &'a Array2<f64>,
        y: &'a Array1<f64>,
        s: &'a Array2<f64>,
        x_star: &'a Array1<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n {
            return Err("gaussian reml response: row-count mismatch".to_string());
        }
        if s.nrows() != p || s.ncols() != p || x_star.len() != p {
            return Err("gaussian reml response: column-count mismatch".to_string());
        }
        let (evals, _) = s.eigh(Side::Lower).map_err(|e| {
            format!("gaussian reml response: penalty eigendecomposition failed: {e:?}")
        })?;
        let max_ev = evals.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        let tol = max_ev * 1e-10 * (p.max(1) as f64);
        let rank_s = evals.iter().filter(|&&e| e > tol).count();
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);
        let yty = y.dot(y);
        Ok(Self {
            x,
            y,
            s,
            x_star,
            n,
            p,
            rank_s,
            xtx,
            xty,
            yty,
        })
    }

    /// `rank(S)` as detected at construction.
    pub fn rank_s(&self) -> usize {
        self.rank_s
    }

    /// Closed-form REML evaluation at `ρ`. `z = Some(_)` augments with the
    /// test row; `z = None` is the original-data criterion (used for ρ̂₀).
    fn eval(&self, rho: f64, z: Option<f64>) -> Result<RemlEval, String> {
        let p = self.p;
        let n_eff = self.n + usize::from(z.is_some());
        let m0 = p - self.rank_s;
        if n_eff <= m0 {
            return Err(format!(
                "gaussian reml response: degrees of freedom n_eff−M₀ = {n_eff}−{m0} ≤ 0; \
                 REML criterion undefined"
            ));
        }
        let coef = (n_eff - m0) as f64;
        let r = self.rank_s as f64;
        let lambda = rho.exp();

        // A(λ) = XᵀX + λ S [+ x_* x_*ᵀ].
        let mut a = self.xtx.clone();
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] += lambda * self.s[[i, j]];
            }
        }
        if z.is_some() {
            for i in 0..p {
                for j in 0..p {
                    a[[i, j]] += self.x_star[i] * self.x_star[j];
                }
            }
        }
        let chol = a
            .cholesky(Side::Lower)
            .map_err(|e| format!("gaussian reml response: A(λ) not SPD: {e:?}"))?;

        // c(z) = Xᵀy [+ x_* z].
        let mut c = self.xty.clone();
        if let Some(zv) = z {
            for j in 0..p {
                c[j] += self.x_star[j] * zv;
            }
        }
        let beta = chol.solvevec(&c);
        let yty_eff = self.yty + z.map_or(0.0, |zv| zv * zv);
        let d = yty_eff - c.dot(&beta);
        if !(d > 0.0) {
            return Err(format!(
                "gaussian reml response: non-positive penalized RSS D = {d}; degenerate fit"
            ));
        }

        let sbeta = self.s.dot(&beta);
        let pen = lambda * beta.dot(&sbeta);

        // Z = A⁻¹ S for the trace terms tr(A⁻¹S), tr((A⁻¹S)²).
        let z_mat = chol.solve_mat(self.s);
        let mut tr_ainv_s = 0.0;
        let mut tr_ainv_s_sq = 0.0;
        for i in 0..p {
            tr_ainv_s += z_mat[[i, i]];
            for j in 0..p {
                tr_ainv_s_sq += z_mat[[i, j]] * z_mat[[j, i]];
            }
        }

        // v_s = A⁻¹ Sβ̂ (so dβ̂/dρ = −λ v_s); quad = β̂ᵀS A⁻¹ S β̂.
        let v_s = chol.solvevec(&sbeta);
        let quad = sbeta.dot(&v_s);

        let logdet: f64 = 2.0 * chol.diag().iter().map(|d| d.ln()).sum::<f64>();
        let value = coef * d.ln() + logdet - r * rho;
        let grad = coef * pen / d + lambda * tr_ainv_s - r;
        let pen_prime = pen - 2.0 * lambda * lambda * quad;
        let hess = coef * (pen_prime * d - pen * pen) / (d * d) + lambda * tr_ainv_s
            - lambda * lambda * tr_ainv_s_sq;

        // ∂μ̂/∂ρ = X (dβ̂/dρ) = −λ X v_s.
        let xv = fast_av(self.x, &v_s);
        let mu_rho_train = xv.mapv(|t| -lambda * t);
        let mu_rho_test = -lambda * self.x_star.dot(&v_s);

        let cross = if let Some(zv) = z {
            let b = chol.solvevec(self.x_star); // dβ̂/dz
            let pen_z = 2.0 * lambda * sbeta.dot(&b);
            let r_star = zv - self.x_star.dot(&beta);
            let d_z = 2.0 * r_star;
            coef * (pen_z * d - pen * d_z) / (d * d)
        } else {
            0.0
        };

        Ok(RemlEval {
            value,
            grad,
            hess,
            cross,
            mu_rho_train,
            mu_rho_test,
        })
    }

    /// Public, value-only REML criterion (for FD verification of the gradient).
    pub fn penalized_laml_criterion(&self, rho: f64, z: Option<f64>) -> Result<f64, String> {
        Ok(self.eval(rho, z)?.value)
    }

    /// `dρ̂/dz` at a stationary `(ρ, z)` via the outer IFT.
    pub fn drho_dz(&self, rho: f64, z: f64) -> Result<f64, String> {
        let ev = self.eval(rho, Some(z))?;
        if ev.hess.abs() < 1e-14 {
            return Err(
                "gaussian reml response: outer Hessian ∂²Ṽ/∂ρ² ≈ 0; dρ̂/dz singular".to_string(),
            );
        }
        Ok(-ev.cross / ev.hess)
    }

    /// Select ρ̂ by REML: a coarse value scan over `ρ ∈ [−25, 25]` to seed,
    /// then safeguarded Newton on `G = 0`. Deterministic (no randomness, fixed
    /// grid), so it qualifies as the symmetric fitting map's smoothing choice.
    pub fn select_rho(&self, z: Option<f64>) -> Result<f64, String> {
        let (lo, hi, m) = (-25.0_f64, 25.0_f64, 60usize);
        let mut best = (f64::INFINITY, 0.0_f64);
        for k in 0..=m {
            let rho = lo + (hi - lo) * (k as f64) / (m as f64);
            if let Ok(ev) = self.eval(rho, z)
                && ev.value < best.0
            {
                best = (ev.value, rho);
            }
        }
        let mut rho = best.1;
        for _ in 0..100 {
            let ev = self.eval(rho, z)?;
            if !ev.hess.is_finite() || ev.hess <= 1e-12 {
                break;
            }
            let step = ev.grad / ev.hess;
            let new_rho = (rho - step).clamp(lo - 5.0, hi + 5.0);
            let delta = new_rho - rho;
            rho = new_rho;
            if delta.abs() < 1e-13 {
                break;
            }
        }
        Ok(rho)
    }

    /// Honest membership at candidate `z`: re-select ρ̂(z) on the augmented
    /// data, fit, and apply the conformal rank rule. This IS the honest
    /// (ρ-re-selecting) full-conformal map, computed exactly per candidate.
    pub fn honest_membership(&self, z: f64, alpha: f64) -> Result<bool, String> {
        let rho = self.select_rho(Some(z))?;
        let lambda = rho.exp();
        let p = self.p;
        let mut a = self.xtx.clone();
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] += lambda * self.s[[i, j]] + self.x_star[i] * self.x_star[j];
            }
        }
        let chol = a
            .cholesky(Side::Lower)
            .map_err(|e| format!("gaussian reml response: honest A(λ) not SPD: {e:?}"))?;
        let mut c = self.xty.clone();
        for j in 0..p {
            c[j] += self.x_star[j] * z;
        }
        let beta = chol.solvevec(&c);
        let e_star = (z - self.x_star.dot(&beta)).abs();
        let xb = fast_av(self.x, &beta);
        let count = (0..self.n)
            .filter(|&i| (self.y[i] - xb[i]).abs() >= e_star)
            .count();
        Ok((1.0 + count as f64) > alpha * (self.n as f64 + 1.0))
    }

    /// Run the certificate-first procedure: build the frozen-ρ exact set, then
    /// compute the conditional score perturbation a ρ re-selection could
    /// induce and decide whether the frozen set is accepted under the
    /// rho-grid Lipschitz assumption.
    ///
    /// The score-perturbation bound is `max_i(|∂μ̂_i/∂ρ| + |∂μ̂_*/∂ρ|) ·
    /// sup_z|ρ̂(z) − ρ̂₀|`, where the ρ-excursion is bounded over the set's
    /// finite deciding range by the worst probed `|ρ̂(z) − ρ̂₀|` plus a
    /// mean-value remainder from the observed probe-grid maximum of
    /// `|dρ̂/dz|`. This is a conditional check, not a continuous supremum
    /// proof: the returned diagnostics expose the probe count and observed
    /// derivative maximum.
    pub fn certified_full_conformal(&self, alpha: f64) -> Result<CertifiedFullConformal, String> {
        let rho0 = self.select_rho(None)?;
        let lambda0 = rho0.exp();
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for i in 0..self.p {
            for j in 0..self.p {
                s_lambda[[i, j]] = lambda0 * self.s[[i, j]];
            }
        }
        let weights = Array1::<f64>::ones(self.n);
        let engine =
            ExactGaussianFullConformal::new(self.x, self.y, &weights, &s_lambda, self.x_star)?;
        let frozen_set = engine.prediction_set(alpha);

        // Collect the finite deciding endpoints. If there are none (set is ℝ
        // or empty), the margin has already been computed analytically. With
        // no finite range for the rho probes, accept only the score-independent
        // case where no comparison is needed; otherwise refuse instead of
        // pretending the unbounded rho excursion was checked.
        let mut endpoints: Vec<f64> = Vec::new();
        for itv in &frozen_set.intervals {
            for ep in [itv.lo, itv.hi] {
                if ep.is_finite() {
                    endpoints.push(ep);
                }
            }
        }
        if endpoints.is_empty() {
            let score_perturbation_bound = if frozen_set.boundary_margin == f64::INFINITY {
                0.0
            } else {
                f64::INFINITY
            };
            return Ok(CertifiedFullConformal {
                certificate: FrozenRhoCertificate::decide(
                    score_perturbation_bound,
                    frozen_set.boundary_margin,
                ),
                frozen_set,
                rho_frozen: rho0,
                rho_excursion: 0.0,
                score_rho_lipschitz: 0.0,
                rho_probe_count: 0,
                observed_sup_drho_dz: 0.0,
            });
        }
        endpoints.sort_by(|a, b| a.partial_cmp(b).expect("finite endpoints"));
        let z_lo = *endpoints.first().expect("non-empty");
        let z_hi = *endpoints.last().expect("non-empty");

        // Probe grid spanning the deciding range; honest ρ̂(z) and dρ̂/dz at
        // each probe. The ρ-excursion sup is bounded by the worst observed
        // deviation plus the mean-value Lipschitz remainder over the range.
        let probes = 64usize;
        let mut max_dev = 0.0_f64;
        let mut observed_sup_drho_dz = 0.0_f64;
        let mut lip = 0.0_f64;
        // Lipschitz at the frozen optimum (un-augmented sensitivities).
        let ev0 = self.eval(rho0, None)?;
        for i in 0..self.n {
            lip = lip.max(ev0.mu_rho_train[i].abs() + ev0.mu_rho_test.abs());
        }
        for k in 0..=probes {
            let z = z_lo + (z_hi - z_lo) * (k as f64) / (probes as f64);
            let rho_z = self.select_rho(Some(z))?;
            max_dev = max_dev.max((rho_z - rho0).abs());
            if let Ok(d) = self.drho_dz(rho_z, z) {
                observed_sup_drho_dz = observed_sup_drho_dz.max(d.abs());
            }
            // Lipschitz also at the re-selected optimum (scores move with ρ).
            let evz = self.eval(rho_z, Some(z))?;
            for i in 0..self.n {
                lip = lip.max(evz.mu_rho_train[i].abs() + evz.mu_rho_test.abs());
            }
        }
        // Mean-value remainder under the explicit grid assumption: between
        // probes ρ̂ can drift by at most the observed derivative maximum times
        // the probe spacing, provided the continuous derivative supremum does
        // not exceed the observed maximum beyond this allowance.
        let spacing = (z_hi - z_lo) / (probes as f64);
        let rho_excursion = max_dev + observed_sup_drho_dz * spacing;
        let score_perturbation_bound = lip * rho_excursion;
        let certificate =
            FrozenRhoCertificate::decide(score_perturbation_bound, frozen_set.boundary_margin);

        Ok(CertifiedFullConformal {
            frozen_set,
            certificate,
            rho_frozen: rho0,
            rho_excursion,
            score_rho_lipschitz: lip,
            rho_probe_count: probes + 1,
            observed_sup_drho_dz,
        })
    }
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
    /// FOR the candidate — the conservative `≥` convention shared with the
    /// discrete enumeration arm).
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
        if prior_weights.iter().any(|&w| (w - 1.0).abs() > 1e-12) {
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

    /// Natural magnitude of the augmented penalized gradient, mirroring the
    /// main P-IRLS convergence certificate's `gradient_natural_scale`
    /// (`src/solver/pirls/state.rs`): `‖Xᵀ(μ − y)‖₂ + ‖Sβ‖₂` plus the test
    /// row's score contribution `‖x_*‖·|μ̂_* − z|`. The penalized score is a
    /// difference of these O(√(n+1)) sums, so at the optimum the raw gradient
    /// floor scales with this quantity, NOT with `(1 + ‖β‖)`. Dividing by
    /// `1 + this` yields a stationarity residual that is invariant under
    /// uniform rescaling of the objective and per-observation in meaning.
    fn gradient_natural_scale(&self, beta: &Array1<f64>, z: f64) -> f64 {
        let eta = fast_av(self.x, beta);
        let mut resid = Array1::<f64>::zeros(self.n);
        for i in 0..self.n {
            resid[i] = self.family.mean(eta[i]) - self.y[i];
        }
        let score = self.x.t().dot(&resid);
        let r_star = self.family.mean(self.x_star.dot(beta)) - z;
        vec_norm(&score) + vec_norm(&self.s_lambda.dot(beta)) + self.star_norm * r_star.abs()
    }

    /// Dimension-based scale `√(n+1) · √p` for the structural KKT bound, with
    /// `n+1` counting the appended test row. Matches `kkt_dimension_scale` in
    /// the main P-IRLS state: under standardized columns the augmented score
    /// `Xᵀ(μ − y)` has components of order O(√(n+1)), so an absolute
    /// `‖g‖ < τ` test becomes systematically too tight as `n` grows. This
    /// scaling restores the advertised per-observation meaning of `τ`.
    fn kkt_dimension_scale(&self) -> f64 {
        (((self.n + 1) as f64).sqrt()) * ((self.p as f64).max(1.0).sqrt())
    }

    /// Scale-invariant KKT acceptance on the RAW penalized gradient, exactly
    /// the `WorkingState::certifies_kkt` certificate the engine's main solver
    /// uses: the iterate certifies stationarity at tolerance `tol` under
    /// EITHER the dimension-scaled absolute bound OR the data-driven
    /// natural-scale relative bound. The earlier predicate compared the
    /// PRECONDITIONED Newton step `‖H⁻¹g‖` against `tol·(1 + ‖β‖)`, whose
    /// floating-point floor is `~ε·(n+1)/λ_min(H)` — n-dependent and not
    /// compensated by `(1 + ‖β‖)`, so genuinely-converged fits (e.g. raw
    /// gradient floor `3.6e-8` at moderate n) were rejected as non-converged.
    fn kkt_converged(&self, beta: &Array1<f64>, z: f64, tol: f64) -> bool {
        let g_norm = vec_norm(&self.penalized_score(beta, z));
        g_norm < tol * self.kkt_dimension_scale()
            || g_norm / (1.0 + self.gradient_natural_scale(beta, z)) < tol
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

// ─────────────────────────────────────────────────────────────────────────
// Jackknife+ / CV+ (Barber, Candès, Ramdas & Tibshirani 2021)
// ─────────────────────────────────────────────────────────────────────────

/// A jackknife+ (or CV+) prediction interval at miscoverage `α`, with the
/// honest ±∞ convention of the split-conformal calibrator: when `n` is too
/// small for the required order statistic to exist, the corresponding
/// endpoint is infinite rather than silently clipped.
#[derive(Clone, Copy, Debug)]
pub struct JackknifePlusInterval {
    pub lo: f64,
    pub hi: f64,
    pub alpha: f64,
    /// Number of leave-one-out (or out-of-fold) residual/prediction pairs.
    pub n: usize,
}

impl JackknifePlusInterval {
    /// Whether both endpoints are finite (enough points to certify the
    /// requested level).
    pub fn certifies_finite(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }
}

/// The jackknife+ interval assembly of Barber et al. (2021), exact order
/// statistics:
///
/// ```text
///   Ĉ_α = [ q̂⁻_α { μ̂₋ᵢ(x_*) − Rᵢ } ,  q̂⁺_α { μ̂₋ᵢ(x_*) + Rᵢ } ]
/// ```
///
/// where `Rᵢ = |yᵢ − μ̂₋ᵢ(xᵢ)|` are the leave-one-out absolute residuals,
/// `q̂⁺_α` is the `⌈(1−α)(n+1)⌉`-th smallest value (1-based; `+∞` when that
/// rank exceeds `n`), and `q̂⁻_α` the `⌊α(n+1)⌋`-th smallest (`−∞` when that
/// rank is below 1). Guarantee: `P(Y_* ∈ Ĉ_α) ≥ 1 − 2α` for exchangeable
/// data and a symmetric fitting map — no model correctness assumed.
///
/// The CV+ variant is THIS SAME assembly fed with K-fold out-of-fold
/// quantities (`μ̂₋ₖ₍ᵢ₎(x_*)` and `Rᵢ = |yᵢ − μ̂₋ₖ₍ᵢ₎(xᵢ)|`), so no second code
/// path exists to drift — but its GUARANTEE is weaker, not identical: K-fold
/// folds do not have leave-one-out symmetry, and Barber et al. (2021, Thm 4)
/// prove only `P(Y_* ∈ Ĉ_α) ≥ 1 − 2α − (1 − K/n)/(K + 1)` for CV+. The extra
/// slack vanishes at K = n (where CV+ IS jackknife+); any CV+ caller must
/// state that bound, not the jackknife+ one.
pub fn jackknife_plus_interval(
    loo_test_predictions: &Array1<f64>,
    loo_abs_residuals: &Array1<f64>,
    alpha: f64,
) -> Result<JackknifePlusInterval, String> {
    let n = loo_test_predictions.len();
    if n == 0 {
        return Err("jackknife+: empty leave-one-out inputs".to_string());
    }
    if loo_abs_residuals.len() != n {
        return Err(format!(
            "jackknife+: {} predictions but {} residuals",
            n,
            loo_abs_residuals.len()
        ));
    }
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(format!("jackknife+: alpha must be in (0, 1), got {alpha}"));
    }
    for (i, (&m, &r)) in loo_test_predictions
        .iter()
        .zip(loo_abs_residuals.iter())
        .enumerate()
    {
        if !m.is_finite() {
            return Err(format!(
                "jackknife+: non-finite LOO prediction at index {i}"
            ));
        }
        if !(r.is_finite() && r >= 0.0) {
            return Err(format!(
                "jackknife+: LOO residual at index {i} must be finite and non-negative, got {r}"
            ));
        }
    }
    let n1 = (n + 1) as f64;
    let rank_hi = (n1 * (1.0 - alpha)).ceil() as usize;
    let rank_lo = (n1 * alpha).floor() as usize;
    let hi = if rank_hi > n {
        f64::INFINITY
    } else {
        let mut upper: Vec<f64> = (0..n)
            .map(|i| loo_test_predictions[i] + loo_abs_residuals[i])
            .collect();
        upper.sort_by(|a, b| a.partial_cmp(b).expect("finite jackknife+ endpoints"));
        upper[rank_hi - 1]
    };
    let lo = if rank_lo < 1 {
        f64::NEG_INFINITY
    } else {
        let mut lower: Vec<f64> = (0..n)
            .map(|i| loo_test_predictions[i] - loo_abs_residuals[i])
            .collect();
        lower.sort_by(|a, b| a.partial_cmp(b).expect("finite jackknife+ endpoints"));
        lower[rank_lo - 1]
    };
    Ok(JackknifePlusInterval { lo, hi, alpha, n })
}

/// EXACT jackknife+ for the penalized Gaussian-identity fit at frozen `Sλ`,
/// with the leave-one-out quantities computed in closed form (no refits):
/// the LOO fit is a rank-one Sherman–Morrison downdate of the single
/// factored normal matrix `M = XᵀX + Sλ`, so
///
/// ```text
///   rᵢ = yᵢ − xᵢᵀβ̂ ,  hᵢ = xᵢᵀM⁻¹xᵢ ,
///   Rᵢ = |rᵢ| / (1 − hᵢ) ,                    (exact LOO residual)
///   μ̂₋ᵢ(x_*) = x_*ᵀβ̂ − (x_*ᵀM⁻¹xᵢ)·rᵢ/(1 − hᵢ)   (exact LOO test prediction)
/// ```
///
/// — the same factored-Hessian leave-one-out algebra the ALO module
/// (src/inference/alo.rs) applies on the working-response scale, specialized
/// here to the Gaussian-identity case where it is exact rather than
/// approximate. Unit prior weights are required for the exchangeability
/// guarantee, as everywhere in this module.
pub fn gaussian_jackknife_plus(
    x: &Array2<f64>,
    y: &Array1<f64>,
    prior_weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    x_star: &Array1<f64>,
    alpha: f64,
) -> Result<JackknifePlusInterval, String> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || prior_weights.len() != n {
        return Err("gaussian jackknife+: row-count mismatch".to_string());
    }
    if s_lambda.nrows() != p || s_lambda.ncols() != p || x_star.len() != p {
        return Err("gaussian jackknife+: column-count mismatch".to_string());
    }
    if prior_weights.iter().any(|&w| (w - 1.0).abs() > 1e-12) {
        return Err(
            "gaussian jackknife+ requires unit prior weights: a reweighted training row \
             is not exchangeable with the test row, so the finite-sample coverage proof \
             does not apply"
                .to_string(),
        );
    }
    let m = x.t().dot(x) + s_lambda;
    let chol = m
        .cholesky(Side::Lower)
        .map_err(|e| format!("gaussian jackknife+: normal matrix not SPD: {e:?}"))?;
    let beta = chol.solvevec(&x.t().dot(y));
    let mu = fast_av(x, &beta);
    let mu_star = x_star.dot(&beta);
    let b = chol.solvevec(x_star);
    let xt = x.t().as_standard_layout().into_owned();
    let minv_xt = chol.solve_mat(&xt);
    let mut loo_preds = Array1::<f64>::zeros(n);
    let mut loo_resids = Array1::<f64>::zeros(n);
    for i in 0..n {
        let h_i = x.row(i).dot(&minv_xt.column(i));
        let one_minus_h = 1.0 - h_i;
        if !(one_minus_h > 1e-10) {
            return Err(format!(
                "gaussian jackknife+: leverage hᵢ = {h_i} at row {i} leaves no leave-one-out \
                 information (1 − hᵢ ≤ 1e-10); the rank-one downdate is exact only for hᵢ < 1"
            ));
        }
        let r_i = y[i] - mu[i];
        let c_i = x.row(i).dot(&b); // x_*ᵀ M⁻¹ xᵢ by symmetry
        loo_resids[i] = (r_i / one_minus_h).abs();
        loo_preds[i] = mu_star - c_i * r_i / one_minus_h;
    }
    jackknife_plus_interval(&loo_preds, &loo_resids, alpha)
}

/// Test-point-independent sufficient statistics for the exact penalized
/// Gaussian-identity jackknife+, factored ONCE from `(X, y, Sλ)` so any number
/// of test points reuse the single Cholesky of `M = XᵀX + Sλ`.
///
/// For each training row `i` the leave-one-out fit is the rank-one
/// Sherman–Morrison downdate of `M`, giving (in closed form, no refits):
///
/// ```text
///   vᵢ = M⁻¹ xᵢ                         (p-vector, one column of M⁻¹Xᵀ)
///   hᵢ = xᵢᵀ vᵢ ,  cᵢ = rᵢ / (1 − hᵢ)   (signed LOO residual)
///   Rᵢ = |cᵢ|                            (LOO absolute residual)
/// ```
///
/// At a test point `x_*` the LOO prediction is then a single inner product per
/// row, `μ̂₋ᵢ(x_*) = x_*ᵀβ̂ − (x_*ᵀvᵢ)·cᵢ`, so [`interval`](Self::interval) is
/// `O(n·p)` after the `O(n·p²)` factorization here. This is the substrate the
/// `predict(interval=level)` magic default replays: the stats are exactly the
/// `{vᵢ, cᵢ, Rᵢ}` of `gaussian_jackknife_plus`, which is recovered exactly when
/// fed a single `x_*` (the in-module test asserts that equivalence).
///
/// Unit prior weights are required, as everywhere in this module: a reweighted
/// training row is not exchangeable with the test row, so the finite-sample
/// coverage proof does not apply. The constructor rejects non-unit weights and
/// rows with `1 − hᵢ ≤ 1e-10` (no leave-one-out information).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GaussianJackknifePlusStats {
    /// Fitted coefficients `β̂ = M⁻¹Xᵀy`.
    beta: Array1<f64>,
    /// `M⁻¹Xᵀ` (p × n): column `i` is `vᵢ = M⁻¹xᵢ`.
    minv_xt: Array2<f64>,
    /// Signed leave-one-out residuals `cᵢ = rᵢ/(1 − hᵢ)` (n).
    signed_loo: Array1<f64>,
    /// Absolute leave-one-out residuals `Rᵢ = |cᵢ|` (n).
    abs_loo: Array1<f64>,
}

impl GaussianJackknifePlusStats {
    /// Factor the test-point-independent jackknife+ statistics from the
    /// training design, response, prior weights, and penalty matrix.
    pub fn new(
        x: &Array2<f64>,
        y: &Array1<f64>,
        prior_weights: &Array1<f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n || prior_weights.len() != n {
            return Err("gaussian jackknife+ stats: row-count mismatch".to_string());
        }
        if s_lambda.nrows() != p || s_lambda.ncols() != p {
            return Err("gaussian jackknife+ stats: column-count mismatch".to_string());
        }
        if prior_weights.iter().any(|&w| (w - 1.0).abs() > 1e-12) {
            return Err(
                "gaussian jackknife+ requires unit prior weights: a reweighted training row \
                 is not exchangeable with the test row, so the finite-sample coverage proof \
                 does not apply"
                    .to_string(),
            );
        }
        let m = x.t().dot(x) + s_lambda;
        Self::from_design_and_normal_matrix(x, y, &m)
    }

    /// Same exact jackknife+ statistics as [`new`](Self::new), but the
    /// penalized normal matrix `M = XᵀX + Sλ` is supplied directly rather than
    /// reassembled from `Sλ`. For a Gaussian-identity unit-weight fit the
    /// converged penalized Hessian stored in [`FitGeometry`] *is* this `M` (the
    /// working weights are unity and the matrix is dispersion-unscaled), so
    /// persisting the design + `M` at fit time and replaying through this
    /// constructor reproduces the certified interval with no penalty
    /// re-derivation — the seam the saved-model `predict(interval=…)` magic
    /// uses.
    ///
    /// `prior_weights` is validated to be unity for the exchangeability
    /// guarantee, identically to [`new`](Self::new).
    pub fn from_design_unit_weight_normal_matrix(
        x: &Array2<f64>,
        y: &Array1<f64>,
        prior_weights: &Array1<f64>,
        m: &Array2<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        if y.len() != n || prior_weights.len() != n {
            return Err("gaussian jackknife+ stats: row-count mismatch".to_string());
        }
        if prior_weights.iter().any(|&w| (w - 1.0).abs() > 1e-12) {
            return Err(
                "gaussian jackknife+ requires unit prior weights: a reweighted training row \
                 is not exchangeable with the test row, so the finite-sample coverage proof \
                 does not apply"
                    .to_string(),
            );
        }
        Self::from_design_and_normal_matrix(x, y, m)
    }

    fn from_design_and_normal_matrix(
        x: &Array2<f64>,
        y: &Array1<f64>,
        m: &Array2<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n {
            return Err("gaussian jackknife+ stats: row-count mismatch".to_string());
        }
        if m.nrows() != p || m.ncols() != p {
            return Err("gaussian jackknife+ stats: normal-matrix shape mismatch".to_string());
        }
        let chol = m
            .cholesky(Side::Lower)
            .map_err(|e| format!("gaussian jackknife+ stats: normal matrix not SPD: {e:?}"))?;
        let beta = chol.solvevec(&x.t().dot(y));
        let mu = fast_av(x, &beta);
        let xt = x.t().as_standard_layout().into_owned();
        let minv_xt = chol.solve_mat(&xt);
        let mut signed_loo = Array1::<f64>::zeros(n);
        let mut abs_loo = Array1::<f64>::zeros(n);
        for i in 0..n {
            let h_i = x.row(i).dot(&minv_xt.column(i));
            let one_minus_h = 1.0 - h_i;
            if !(one_minus_h > 1e-10) {
                return Err(format!(
                    "gaussian jackknife+ stats: leverage hᵢ = {h_i} at row {i} leaves no \
                     leave-one-out information (1 − hᵢ ≤ 1e-10); the rank-one downdate is \
                     exact only for hᵢ < 1"
                ));
            }
            let c_i = (y[i] - mu[i]) / one_minus_h;
            signed_loo[i] = c_i;
            abs_loo[i] = c_i.abs();
        }
        Ok(Self {
            beta,
            minv_xt,
            signed_loo,
            abs_loo,
        })
    }

    /// Number of training rows backing the leave-one-out construction.
    pub fn n(&self) -> usize {
        self.abs_loo.len()
    }

    /// Coefficient dimension `p`.
    pub fn p(&self) -> usize {
        self.beta.len()
    }

    /// Full-model coefficient vector `β̂ = M⁻¹Xᵀy`. The plug-in mean at a
    /// test point `x_*` is `x_*ᵀ β̂`. Exposed so the pyffi layer can emit the
    /// `mean` / `linear_predictor` columns alongside the jackknife+ bounds
    /// without re-running the predictor stack.
    pub fn beta(&self) -> &Array1<f64> {
        &self.beta
    }

    /// Jackknife+ interval at one test row `x_*` and miscoverage `alpha`,
    /// returning the Barber et al. (2021) set with guarantee
    /// `P(Y_* ∈ Ĉ) ≥ 1 − 2·alpha`.
    pub fn interval(
        &self,
        x_star: &Array1<f64>,
        alpha: f64,
    ) -> Result<JackknifePlusInterval, String> {
        let p = self.beta.len();
        if x_star.len() != p {
            return Err(format!(
                "gaussian jackknife+ stats: x_* has {} entries but the fit has {p} coefficients",
                x_star.len()
            ));
        }
        let n = self.abs_loo.len();
        let mu_star = x_star.dot(&self.beta);
        let mut loo_preds = Array1::<f64>::zeros(n);
        for i in 0..n {
            // x_*ᵀ vᵢ = x_*ᵀ M⁻¹ xᵢ.
            let c = x_star.dot(&self.minv_xt.column(i));
            loo_preds[i] = mu_star - c * self.signed_loo[i];
        }
        jackknife_plus_interval(&loo_preds, &self.abs_loo, alpha)
    }
}

/// Persistable substrate for the EXACT Gaussian-identity full-conformal set
/// (#942 Layer 1 + the Layer-3 frozen-ρ self-diagnostic), the analogue of
/// [`GaussianJackknifePlusStats`] for the exact set.
///
/// Unlike jackknife+, the exact full-conformal set has no test-point-independent
/// factorization: every test covariate `x_*` enters the augmented normal matrix
/// `M = XᵀX + x_*x_*ᵀ + Sλ`, so the substrate persists the training design `X`,
/// response `y`, and the (frozen) penalty `Sλ` and rebuilds
/// [`ExactGaussianFullConformal`] per test row — one Cholesky per test point,
/// zero refits. Valid for any penalized smooth with an arbitrary `Sλ` and basis.
///
/// `Sλ` is recovered once at fit time from the converged penalized Hessian
/// `M₀ = XᵀX + Sλ` (the Gaussian-identity, unit-weight, dispersion-unscaled
/// normal matrix stored in [`FitGeometry`]) as `Sλ = M₀ − XᵀX`, so no penalty
/// re-derivation is needed — exactly the seam the jackknife+ substrate uses.
///
/// The frozen-ρ self-diagnostic treats the entire frozen penalty as carrying a
/// single global log-smoothing parameter `ρ` with `S(ρ) = eᵖ·Sλ` and runs the
/// closed-form [`GaussianRemlRhoResponse::certified_full_conformal`]: it
/// re-selects the global ρ̂(z) on the augmented data, bounds the score
/// perturbation freezing ρ̂ could induce, and reports whether freezing is
/// accepted under the stated rho-grid Lipschitz assumption. This is a sound,
/// conservative global-scale check that applies to any penalized smooth (it does
/// not require the model to be single-penalty); per-penalty re-selection is the
/// research-core Layer 3 and is not asserted here.
///
/// Unit prior weights are required, as everywhere in this module: a reweighted
/// training row is not exchangeable with the test row.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExactFullConformalSubstrate {
    /// Training design `X` (n × p).
    x: Array2<f64>,
    /// Training response `y` (n).
    y: Array1<f64>,
    /// Frozen penalty `Sλ = M₀ − XᵀX` at the fitted smoothing parameters (p × p).
    s_lambda: Array2<f64>,
}

/// One test row's exact full-conformal verdict: the outer `[lower, upper]`
/// envelope of the exact set, plus the frozen-ρ self-diagnostics flag.
#[derive(Clone, Debug)]
pub struct ExactFullConformalInterval {
    /// Outer envelope `[min lo, max hi]` of the exact (possibly multi-interval)
    /// set, inheriting its coverage (it is a superset). Endpoints may be
    /// infinite (honest unboundedness in low-information / high-leverage regimes).
    pub lo: f64,
    pub hi: f64,
    /// The exact set itself (a union of intervals).
    pub set: FullConformalSet,
    /// `true` when freezing the global smoothing parameter is ACCEPTED under the
    /// rho-grid Lipschitz assumption (the frozen exact set equals the honest
    /// ρ-re-selecting set); `false` when the certificate REFUSED (the frozen set
    /// may differ from the honest set and the caller should treat the envelope
    /// as the frozen-ρ approximation, not the certified honest set).
    pub frozen_rho_certified: bool,
}

impl ExactFullConformalSubstrate {
    /// Build the substrate from the training design, response, prior weights,
    /// and the converged penalized normal matrix `M₀ = XᵀX + Sλ`. Recovers the
    /// frozen penalty `Sλ = M₀ − XᵀX` once. Rejects non-unit prior weights and
    /// shape mismatches, identically to the rest of this module.
    pub fn from_design_unit_weight_normal_matrix(
        x: &Array2<f64>,
        y: &Array1<f64>,
        prior_weights: &Array1<f64>,
        m: &Array2<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n || prior_weights.len() != n {
            return Err("exact full conformal substrate: row-count mismatch".to_string());
        }
        if m.nrows() != p || m.ncols() != p {
            return Err("exact full conformal substrate: normal-matrix shape mismatch".to_string());
        }
        if prior_weights.iter().any(|&w| (w - 1.0).abs() > 1e-12) {
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

    /// The exact full-conformal verdict at one test row `x_*` and miscoverage
    /// `alpha`: the exact set, its outer envelope, and the frozen-ρ
    /// self-diagnostics flag. One Cholesky per call, zero refits.
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
        // The AUTHORITATIVE exact set is built at the user's fitted penalty `Sλ`
        // (ρ frozen exactly at the fit), so the reported set reflects the model
        // the user trained — not a re-optimized global scale.
        let weights = Array1::<f64>::ones(self.n());
        let engine =
            ExactGaussianFullConformal::new(&self.x, &self.y, &weights, &self.s_lambda, x_star)?;
        let set = engine.prediction_set(alpha);

        // Frozen-ρ self-diagnostic: treat the whole frozen penalty as carrying a
        // single global log-smoothing parameter `ρ` with `S(ρ) = eᵖ·Sλ` and run
        // the closed-form certificate. It re-selects the global ρ̂(z) on the
        // augmented data and decides whether freezing the global scale is safe.
        // This is a sound conservative check around the global REML optimum (a
        // properly fitted model already sits at ρ̂₀ ≈ 0, where this set coincides
        // with the authoritative set above); per-penalty re-selection is the
        // research-core Layer 3 and is not asserted here. A degenerate certificate
        // computation must NOT void the exact set, so its failure maps to "not
        // certified" rather than an error.
        let frozen_rho_certified =
            GaussianRemlRhoResponse::new(&self.x, &self.y, &self.s_lambda, x_star)
                .and_then(|response| response.certified_full_conformal(alpha))
                .map(|certified| {
                    matches!(
                        certified.certificate,
                        FrozenRhoCertificate::Certified { .. }
                    )
                })
                .unwrap_or(false);

        let (lo, hi) = if set.intervals.is_empty() {
            // No candidate qualifies (pathological tiny α·(n+1)); collapse to the
            // frozen plug-in mean μ̂_* = x_*ᵀβ̂, β̂ = (XᵀX+Sλ)⁻¹Xᵀy — the only
            // honest scalar answer.
            let m = &self.x.t().dot(&self.x) + &self.s_lambda;
            let chol = m.cholesky(Side::Lower).map_err(|e| {
                format!("exact full conformal: frozen normal matrix not SPD: {e:?}")
            })?;
            let beta = chol.solvevec(&self.x.t().dot(&self.y));
            let mu_point = x_star.dot(&beta);
            (mu_point, mu_point)
        } else {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for itv in &set.intervals {
                lo = lo.min(itv.lo);
                hi = hi.max(itv.hi);
            }
            (lo, hi)
        };
        Ok(ExactFullConformalInterval {
            lo,
            hi,
            set,
            frozen_rho_certified,
        })
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

        // Margin is non-negative; critical boundary ties are reported as
        // zero rather than skipped.
        assert!(set.boundary_margin >= 0.0);
    }

    #[test]
    fn boundary_tie_has_zero_margin_and_refuses() {
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
        assert_eq!(set.boundary_margin, 0.0);
        assert!(matches!(
            FrozenRhoCertificate::decide(0.0, set.boundary_margin),
            FrozenRhoCertificate::Refused { .. }
        ));
    }

    #[test]
    fn identically_tied_all_real_set_has_zero_margin_and_refuses() {
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
        assert_eq!(set.boundary_margin, 0.0);
        assert!(matches!(
            FrozenRhoCertificate::decide(0.0, set.boundary_margin),
            FrozenRhoCertificate::Refused { .. }
        ));
    }

    #[test]
    fn strictly_separated_all_real_margin_can_accept() {
        let engine = ExactGaussianFullConformal {
            u: Array1::from_vec(vec![1.0, 1.0, 0.0]),
            w: Array1::from_vec(vec![1.0, -1.0, 0.1]),
            n: 2,
        };

        let set = engine.prediction_set(0.5);
        assert_eq!(set.intervals.len(), 1);
        assert_eq!(set.intervals[0].lo, f64::NEG_INFINITY);
        assert_eq!(set.intervals[0].hi, f64::INFINITY);
        assert!(set.boundary_margin > 0.5, "margin={}", set.boundary_margin);
        assert!(matches!(
            FrozenRhoCertificate::decide(0.5, set.boundary_margin),
            FrozenRhoCertificate::Certified { .. }
        ));
    }

    /// Scalar penalized intercept-only logistic fit on augmented Bernoulli
    /// data: maximize `Σ_{n+1 rows} [y η − log(1+eʸ)] − ½λη²` by Newton.
    /// The map is symmetric BY CONSTRUCTION (it sees the responses only
    /// through their sum over the n+1 exchangeable rows), so it satisfies
    /// the `SymmetricAugmentedFit` contract exactly — making the coverage
    /// theorem checkable by enumeration below.
    fn bernoulli_intercept_scores(train: &[f64], z: f64, lambda: f64) -> Array1<f64> {
        let n1 = train.len() + 1;
        let sum_y: f64 = train.iter().sum::<f64>() + z;
        let mut eta = 0.0_f64;
        for _ in 0..200 {
            let mu = 1.0 / (1.0 + (-eta).exp());
            let g = sum_y - (n1 as f64) * mu - lambda * eta;
            let h = -(n1 as f64) * mu * (1.0 - mu) - lambda;
            let step = g / h;
            eta -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        let mu = 1.0 / (1.0 + (-eta).exp());
        let mut scores = Array1::<f64>::zeros(n1);
        for (i, &yi) in train.iter().enumerate() {
            scores[i] = (yi - mu).abs();
        }
        scores[n1 - 1] = (z - mu).abs();
        scores
    }

    /// Finite-sample validity as a THEOREM CHECK, not a simulation: for the
    /// intercept-only penalized-logistic map above, enumerate EVERY Bernoulli
    /// training dataset (2ⁿ of them) and both test outcomes, and compute the
    /// exact coverage probability `P(y_* ∈ C_α)` under iid Bernoulli(θ).
    /// Full conformal guarantees ≥ 1 − α for every θ and every α — if the
    /// rank convention, the p-value denominator, or the tie handling were
    /// wrong by even one unit, some (θ, α) cell here would dip below the
    /// bound. Also pins informativeness: the set is not the trivial {0, 1}
    /// on every dataset (an always-trivial set would satisfy coverage
    /// vacuously).
    #[test]
    fn bernoulli_full_conformal_exact_coverage_by_total_enumeration() {
        let n = 7usize;
        let lambda = 0.5_f64;
        for &theta in &[0.2_f64, 0.5, 0.8] {
            for &alpha in &[0.10_f64, 0.25] {
                let mut coverage = 0.0_f64;
                let mut any_strict_subset = false;
                for mask in 0u32..(1u32 << n) {
                    let train: Vec<f64> = (0..n).map(|i| f64::from((mask >> i) & 1)).collect();
                    let p_train: f64 = train
                        .iter()
                        .map(|&y| if y > 0.5 { theta } else { 1.0 - theta })
                        .product();
                    let mut map = |z: f64| -> Result<Array1<f64>, String> {
                        Ok(bernoulli_intercept_scores(&train, z, lambda))
                    };
                    let set = bernoulli_full_conformal(&mut map, alpha).expect("bernoulli set");
                    assert!(set.lower_tail_unresolved.is_none());
                    assert!(set.upper_tail_unresolved.is_none());
                    let holds_zero = set.members.contains(&0.0);
                    let holds_one = set.members.contains(&1.0);
                    if !(holds_zero && holds_one) {
                        any_strict_subset = true;
                    }
                    coverage += p_train
                        * ((1.0 - theta) * f64::from(u8::from(holds_zero))
                            + theta * f64::from(u8::from(holds_one)));
                }
                assert!(
                    coverage >= 1.0 - alpha - 1e-12,
                    "exact full-conformal coverage must be ≥ 1−α for every θ: \
                     θ={theta} α={alpha} coverage={coverage}"
                );
                if alpha == 0.25 {
                    assert!(
                        any_strict_subset,
                        "θ={theta} α={alpha}: the set must be informative (a strict \
                         subset of the support on at least one dataset), otherwise \
                         the coverage bound is satisfied vacuously"
                    );
                }
            }
        }

        // Concrete informativeness pin: an all-zeros training run at α=0.25
        // must exclude z=1 — the augmented fit at z=1 has μ̂ ≈ 0.21, so the
        // test row's score 1−μ̂ ≈ 0.79 strictly dominates every training
        // score (≈ 0.21) and its p-value is 1/8 = 0.125 ≤ α.
        let train = vec![0.0; n];
        let mut map = |z: f64| -> Result<Array1<f64>, String> {
            Ok(bernoulli_intercept_scores(&train, z, lambda))
        };
        let set = bernoulli_full_conformal(&mut map, 0.25).expect("set");
        assert_eq!(
            set.members,
            vec![0.0],
            "all-zeros training data at α=0.25 must yield the set {{0}}"
        );
    }

    /// Windowed (count-style) enumeration: tail flags must report exactly
    /// whether the retained set continues through the window edge. A cleared
    /// flag is only a contiguous-edge statement, not a global monotone-tail
    /// theorem about unexamined candidates.
    #[test]
    fn windowed_discrete_tail_flags_are_honest() {
        // Score map: the augmented "fit" is the mean of the n+1 responses;
        // scores are absolute deviations from it. Symmetric trivially.
        let train = [3.0_f64, 4.0, 5.0, 4.0, 3.0, 5.0, 4.0];
        let mut map = |z: f64| -> Result<Array1<f64>, String> {
            let n1 = train.len() + 1;
            let mean = (train.iter().sum::<f64>() + z) / n1 as f64;
            let mut s = Array1::<f64>::zeros(n1);
            for (i, &yi) in train.iter().enumerate() {
                s[i] = (yi - mean).abs();
            }
            s[n1 - 1] = (z - mean).abs();
            Ok(s)
        };
        let alpha = 0.2;

        // Wide window: both edges are excluded, so no retained component
        // continues contiguously through either edge.
        let wide: Vec<f64> = (0..=12).map(|k| k as f64).collect();
        let set = discrete_full_conformal_window(&mut map, &wide, alpha).expect("wide");
        assert!(!set.members.is_empty(), "wide window must retain the bulk");
        assert!(set.lower_tail_unresolved.is_none());
        assert!(set.upper_tail_unresolved.is_none());
        let lo_member = *set.members.first().expect("non-empty");
        let hi_member = *set.members.last().expect("non-empty");

        // Window cut INSIDE the set: the corresponding flag must fire.
        let cut: Vec<f64> = (0..=(hi_member as i64 - 1)).map(|k| k as f64).collect();
        let cut_set = discrete_full_conformal_window(&mut map, &cut, alpha).expect("cut");
        assert_eq!(
            cut_set.upper_tail_unresolved,
            Some(cut[cut.len() - 1]),
            "a window whose top edge is retained must report the upper tail unresolved"
        );
        assert!(
            lo_member > 0.0 || cut_set.lower_tail_unresolved.is_some(),
            "lower flag must mirror the same contract"
        );

        // Exhaustive constructor clears flags by definition.
        let exhaustive =
            discrete_full_conformal_exhaustive(&mut map, &wide, alpha).expect("exhaustive");
        assert!(exhaustive.lower_tail_unresolved.is_none());
        assert!(exhaustive.upper_tail_unresolved.is_none());

        // Engine contract errors: unsorted candidates and shrinking score
        // vectors are refused loudly.
        assert!(discrete_full_conformal_window(&mut map, &[2.0, 1.0], alpha).is_err());
        let mut bad_map = {
            let mut flip = false;
            move |z: f64| -> Result<Array1<f64>, String> {
                if !z.is_finite() {
                    return Err("bad-map fixture received non-finite candidate".to_string());
                }
                flip = !flip;
                Ok(Array1::<f64>::zeros(if flip { 5 } else { 4 }))
            }
        };
        assert!(discrete_full_conformal_window(&mut bad_map, &[0.0, 1.0], alpha).is_err());
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

    /// The gradient-is-differential contract for the closed-form Gaussian REML
    /// response (#1021 philosophy applied here): every analytic derivative the
    /// IFT consumes (`G`, `∂²Ṽ/∂ρ²`, `∂²Ṽ/∂ρ∂z`) must equal the central
    /// finite-difference of the SAME value path `Ṽ`. A desync here would make
    /// `dρ̂/dz`, and therefore the certificate bound, silently wrong.
    #[test]
    fn gaussian_reml_rho_derivatives_match_finite_difference() {
        let (x, y, s) = gauss_reml_fixture(40, 8);
        let x_star = cosine_row(8, 0.5);
        let resp = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");
        assert_eq!(
            resp.rank_s(),
            6,
            "quartic penalty with two zeros has rank p−2"
        );

        let rho = 0.4_f64;
        let z = 0.3_f64;
        let ev = resp.eval(rho, Some(z)).expect("eval");
        let v = |r: f64, zz: f64| resp.penalized_laml_criterion(r, Some(zz)).expect("v");

        let h = 1e-4_f64;
        let g_fd = (v(rho + h, z) - v(rho - h, z)) / (2.0 * h);
        assert!(
            (ev.grad - g_fd).abs() <= 1e-4 * (1.0 + ev.grad.abs()),
            "G mismatch: analytic={} fd={}",
            ev.grad,
            g_fd
        );
        let hess_fd = (v(rho + h, z) - 2.0 * v(rho, z) + v(rho - h, z)) / (h * h);
        assert!(
            (ev.hess - hess_fd).abs() <= 1e-3 * (1.0 + ev.hess.abs()),
            "∂²Ṽ/∂ρ² mismatch: analytic={} fd={}",
            ev.hess,
            hess_fd
        );
        let k = 1e-4_f64;
        let cross_fd = (v(rho + h, z + k) - v(rho + h, z - k) - v(rho - h, z + k)
            + v(rho - h, z - k))
            / (4.0 * h * k);
        assert!(
            (ev.cross - cross_fd).abs() <= 1e-3 * (1.0 + ev.cross.abs()),
            "∂²Ṽ/∂ρ∂z mismatch: analytic={} fd={}",
            ev.cross,
            cross_fd
        );

        // The un-augmented criterion drops the cross term and the +1 row.
        let ev0 = resp.eval(rho, None).expect("eval0");
        assert_eq!(ev0.cross, 0.0);
        let v0 = |r: f64| resp.penalized_laml_criterion(r, None).expect("v0");
        let g0_fd = (v0(rho + h) - v0(rho - h)) / (2.0 * h);
        assert!((ev0.grad - g0_fd).abs() <= 1e-4 * (1.0 + ev0.grad.abs()));
    }

    /// The exact smoothing response `dρ̂/dz` (outer IFT) must equal the
    /// finite-difference of the ACTUAL re-selection map `z ↦ ρ̂(z)` — i.e. the
    /// IFT derivative is the derivative of the thing it claims to differentiate,
    /// not a parallel formula. This is the honesty check the issue demands for
    /// the ρ-response.
    #[test]
    fn gaussian_reml_smoothing_response_matches_reselection() {
        let (x, y, s) = gauss_reml_fixture(45, 8);
        let x_star = cosine_row(8, 0.42);
        let resp = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");

        for &z in &[0.15_f64, 0.4, 0.75] {
            let rho_z = resp.select_rho(Some(z)).expect("select");
            // ρ̂(z) is a genuine stationary point of the augmented criterion.
            let g = resp.eval(rho_z, Some(z)).expect("eval").grad;
            assert!(g.abs() < 1e-6, "select_rho not stationary: G={g} at z={z}");

            let analytic = resp.drho_dz(rho_z, z).expect("drho");
            let hh = 2e-3_f64;
            let fd = (resp.select_rho(Some(z + hh)).expect("u")
                - resp.select_rho(Some(z - hh)).expect("d"))
                / (2.0 * hh);
            assert!(
                (analytic - fd).abs() <= 1e-3 + 5e-2 * analytic.abs(),
                "dρ̂/dz IFT vs re-selection FD mismatch at z={z}: analytic={analytic} fd={fd}"
            );
        }
    }

    /// Layer-3 grid-check sweep, stated as the conditional invariant that
    /// actually holds for the current rho-excursion machinery:
    ///
    /// Whenever the conditional check accepts, the cheap frozen-ρ set must
    /// agree with the honest set that re-selects ρ̂ at every candidate on a
    /// dense audit grid. This does not prove the continuous rho-excursion
    /// supremum; it verifies the accepted cases under the exposed probe-grid
    /// assumption.
    ///
    /// We do NOT demand that any specific problem accept: a benign problem
    /// can still carry a tie or near-tie boundary whose tiny margin the check
    /// legitimately cannot clear — refusing there is correct, not a bug.
    #[test]
    fn frozen_rho_grid_check_is_conditional_when_it_accepts() {
        let mut soundness_checks = 0usize;
        for &(n, p) in &[(45usize, 8usize), (90, 6)] {
            let (x, y, s) = gauss_reml_fixture(n, p);
            for &t_star in &[0.4_f64, 0.5, 0.6] {
                let x_star = cosine_row(p, t_star);
                let resp = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");
                for &alpha in &[0.15_f64, 0.25] {
                    let cert = resp.certified_full_conformal(alpha).expect("cert");
                    if !matches!(cert.certificate, FrozenRhoCertificate::Certified { .. }) {
                        continue;
                    }
                    assert_eq!(cert.rho_probe_count, 65);
                    assert!(cert.observed_sup_drho_dz >= 0.0);
                    // Verify soundness for the first few certified configs (the
                    // honest oracle re-selects ρ̂ per grid point, so this is the
                    // expensive arm; a handful audits the conditional path).
                    if soundness_checks >= 4 {
                        continue;
                    }
                    let frozen = &cert.frozen_set;
                    if frozen.intervals.is_empty() {
                        continue;
                    }
                    soundness_checks += 1;
                    let lo = frozen.intervals.first().unwrap().lo;
                    let hi = frozen.intervals.last().unwrap().hi;
                    let span = (hi - lo).max(1.0);
                    let z_lo = if lo.is_finite() {
                        lo - 0.5 * span
                    } else {
                        -12.0
                    };
                    let z_hi = if hi.is_finite() {
                        hi + 0.5 * span
                    } else {
                        12.0
                    };
                    let grid = 200usize;
                    for g in 0..=grid {
                        let z = z_lo + (z_hi - z_lo) * (g as f64) / (grid as f64);
                        let in_frozen = frozen
                            .intervals
                            .iter()
                            .any(|itv| z >= itv.lo && z <= itv.hi);
                        let honest = resp.honest_membership(z, alpha).expect("honest");
                        assert_eq!(
                            in_frozen, honest,
                            "conditionally accepted set disagrees with the honest ρ-re-selecting set \
                             at z={z} (n={n}, t*={t_star}, α={alpha}, excursion={}, lip={})",
                            cert.rho_excursion, cert.score_rho_lipschitz
                        );
                    }
                }
            }
        }
    }

    /// The conditional check must REFUSE to accept when the smoothing
    /// response is genuinely large: a high-leverage extrapolated test point at
    /// small n makes ρ̂(z) swing with z, so the excursion bound exceeds the
    /// boundary margin. The machinery must not vacuously always-accept, and
    /// must still return a usable frozen set on refusal.
    #[test]
    fn frozen_rho_certificate_refuses_under_large_smoothing_response() {
        let (x, y, s) = gauss_reml_fixture(12, 6);
        let x_star = cosine_row(6, 1.9); // far extrapolation ⇒ high leverage
        let resp = GaussianRemlRhoResponse::new(&x, &y, &s, &x_star).expect("response");
        let cert = resp.certified_full_conformal(0.2).expect("cert");
        assert!(
            matches!(cert.certificate, FrozenRhoCertificate::Refused { .. }),
            "high-leverage small-n problem should refuse; got {:?} (excursion={}, margin via set)",
            cert.certificate,
            cert.rho_excursion
        );
        // A refusal still hands back the cheap frozen set for the caller to
        // either widen with local refits or fall back from — never nothing.
        assert!(cert.rho_excursion >= 0.0);
    }

    // ── Layer 2 (continuous GLM homotopy) + jackknife+ tests ─────────────

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
                |_t, cand_nll| cand_nll <= cur,
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

    /// (#942 test b) The closed-form Sherman–Morrison jackknife+ must equal
    /// the brute-force construction from n actual leave-one-out refits —
    /// endpoints assembled with the exact Barber et al. (2021) order
    /// statistics — and the ±∞ honesty must engage exactly when the order
    /// statistic does not exist.
    #[test]
    fn jackknife_plus_matches_brute_force_loo_refits() {
        use std::f64::consts::PI;
        let n = 20usize;
        let p = 4usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (j as f64 * PI * t).cos();
            }
            y[i] = (2.0 * PI * t).sin() + 0.15 * (11.0 * i as f64 + 0.3).sin();
        }
        let mut s = Array2::<f64>::eye(p);
        s *= 0.7;
        let weights = Array1::<f64>::ones(n);
        let x_star = cosine_row(p, 0.43);
        let alpha = 0.2;

        let jk = gaussian_jackknife_plus(&x, &y, &weights, &s, &x_star, alpha).expect("jk+");

        // Brute force: n explicit LOO refits, manual order statistics.
        let mut lower_vals: Vec<f64> = Vec::with_capacity(n);
        let mut upper_vals: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let mut m = s.clone();
            let mut rhs = Array1::<f64>::zeros(p);
            for r in 0..n {
                if r == i {
                    continue;
                }
                for a in 0..p {
                    rhs[a] += x[[r, a]] * y[r];
                    for b in 0..p {
                        m[[a, b]] += x[[r, a]] * x[[r, b]];
                    }
                }
            }
            let chol = m.cholesky(Side::Lower).expect("loo chol");
            let beta = chol.solvevec(&rhs);
            let mu_star = x_star.dot(&beta);
            let resid = (y[i] - x.row(i).dot(&beta)).abs();
            lower_vals.push(mu_star - resid);
            upper_vals.push(mu_star + resid);
        }
        lower_vals.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        upper_vals.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let rank_hi = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let rank_lo = ((n as f64 + 1.0) * alpha).floor() as usize;
        assert!(
            rank_lo >= 1 && rank_hi <= n,
            "fixture sized to certify finite endpoints"
        );
        let lo_bf = lower_vals[rank_lo - 1];
        let hi_bf = upper_vals[rank_hi - 1];
        assert!(
            (jk.lo - lo_bf).abs() <= 1e-8 * (1.0 + lo_bf.abs()),
            "jackknife+ lower endpoint {} disagrees with brute-force LOO refits {}",
            jk.lo,
            lo_bf
        );
        assert!(
            (jk.hi - hi_bf).abs() <= 1e-8 * (1.0 + hi_bf.abs()),
            "jackknife+ upper endpoint {} disagrees with brute-force LOO refits {}",
            jk.hi,
            hi_bf
        );
        assert!(jk.certifies_finite());
        assert!(jk.lo < jk.hi);
        assert_eq!(jk.n, n);

        // Honest ±∞: at α = 0.04 with n = 20, ⌈21·0.96⌉ = 21 > n and
        // ⌊21·0.04⌋ = 0 < 1 — both order statistics are out of range, so
        // both endpoints must be infinite, exactly like the split module's
        // +∞ multiplier convention.
        let tight = gaussian_jackknife_plus(&x, &y, &weights, &s, &x_star, 0.04).expect("tight");
        assert!(tight.hi.is_infinite() && tight.hi > 0.0);
        assert!(tight.lo.is_infinite() && tight.lo < 0.0);
        assert!(!tight.certifies_finite());

        // The assembly is the pure-core seam CV+ also routes through:
        // degenerate one-point input keeps the exact rank arithmetic honest.
        let one_pred = Array1::<f64>::from(vec![1.0]);
        let one_res = Array1::<f64>::from(vec![0.5]);
        let tiny = jackknife_plus_interval(&one_pred, &one_res, 0.2).expect("tiny");
        assert!(tiny.hi.is_infinite() && tiny.lo.is_infinite());
    }

    /// The precomputed `GaussianJackknifePlusStats` substrate (factored once,
    /// replayed per test point — the form the predict-path magic uses) must be
    /// bit-for-bit equivalent to the single-shot `gaussian_jackknife_plus` at
    /// every test point. This pins the refactor: the saved-model replay can
    /// never silently drift from the certified reference.
    #[test]
    fn jackknife_plus_stats_replay_matches_single_shot() {
        use std::f64::consts::PI;
        let n = 18usize;
        let p = 4usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            for j in 0..p {
                x[[i, j]] = (j as f64 * PI * t).cos();
            }
            y[i] = (2.0 * PI * t).sin() + 0.2 * (7.0 * i as f64 + 0.9).sin();
        }
        let mut s = Array2::<f64>::eye(p);
        s *= 0.55;
        let weights = Array1::<f64>::ones(n);
        let alpha = 0.1;

        let stats = GaussianJackknifePlusStats::new(&x, &y, &weights, &s).expect("stats");
        assert_eq!(stats.n(), n);
        assert_eq!(stats.p(), p);

        for k in 0..7 {
            let x_star = cosine_row(p, 0.13 + 0.11 * k as f64);
            let single =
                gaussian_jackknife_plus(&x, &y, &weights, &s, &x_star, alpha).expect("single");
            let replay = stats.interval(&x_star, alpha).expect("replay");
            assert!(
                (single.lo - replay.lo).abs() <= 1e-12 * (1.0 + single.lo.abs()),
                "stats replay lower {} != single-shot {} at test point {k}",
                replay.lo,
                single.lo
            );
            assert!(
                (single.hi - replay.hi).abs() <= 1e-12 * (1.0 + single.hi.abs()),
                "stats replay upper {} != single-shot {} at test point {k}",
                replay.hi,
                single.hi
            );
            assert_eq!(single.n, replay.n);
        }

        // Eligibility gate: a reweighted training row is rejected (no
        // exchangeability), never silently certified.
        let mut bad_w = Array1::<f64>::ones(n);
        bad_w[3] = 2.0;
        assert!(GaussianJackknifePlusStats::new(&x, &y, &bad_w, &s).is_err());
    }

    /// Held-out empirical coverage smoke: across many fresh draws of a
    /// synthetic Gaussian-identity problem, the jackknife+ interval at a held-
    /// out test point must cover the realized response at least at the
    /// requested `1 − 2α` rate (small Monte-Carlo slack), with finite width.
    #[test]
    fn jackknife_plus_empirical_coverage_smoke() {
        use std::f64::consts::PI;
        let n = 40usize;
        let p = 4usize;
        let alpha = 0.1; // target coverage ≥ 1 − 2α = 0.8
        let s = {
            let mut s = Array2::<f64>::eye(p);
            s *= 0.4;
            s
        };
        let weights = Array1::<f64>::ones(n);

        // Deterministic LCG so the smoke is reproducible without an RNG dep.
        // `state` is threaded explicitly so `normal` can draw from the same
        // stream without a second mutable borrow of a captured `unif` closure.
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let unif = |state: &mut u64| {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        // Box–Muller standard normal.
        let normal = |state: &mut u64| {
            let u1 = unif(state).max(1e-12);
            let u2 = unif(state);
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        };
        let beta_true = [0.8_f64, -0.5, 0.3, 0.15];
        let sigma = 0.5_f64;
        let design_row = |z: f64| {
            let mut r = Array1::<f64>::zeros(p);
            for j in 0..p {
                r[j] = (j as f64 * PI * z).cos();
            }
            r
        };

        let trials = 200usize;
        let mut covered = 0usize;
        for _ in 0..trials {
            let mut x = Array2::<f64>::zeros((n, p));
            let mut yv = Array1::<f64>::zeros(n);
            for i in 0..n {
                let z = unif(&mut state);
                let row = design_row(z);
                let mut eta = 0.0;
                for j in 0..p {
                    x[[i, j]] = row[j];
                    eta += beta_true[j] * row[j];
                }
                yv[i] = eta + sigma * normal(&mut state);
            }
            let stats = match GaussianJackknifePlusStats::new(&x, &yv, &weights, &s) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let z_star = unif(&mut state);
            let x_star = design_row(z_star);
            let mut eta_star = 0.0;
            for j in 0..p {
                eta_star += beta_true[j] * x_star[j];
            }
            let y_star = eta_star + sigma * normal(&mut state);
            let itv = stats.interval(&x_star, alpha).expect("coverage interval");
            assert!(
                itv.certifies_finite(),
                "coverage trial produced infinite width"
            );
            if y_star >= itv.lo && y_star <= itv.hi {
                covered += 1;
            }
        }
        let rate = covered as f64 / trials as f64;
        // Distribution-free guarantee is ≥ 0.8; allow Monte-Carlo slack below.
        assert!(
            rate >= 0.74,
            "jackknife+ empirical coverage {rate} fell below the 1−2α target with slack"
        );
    }
}
