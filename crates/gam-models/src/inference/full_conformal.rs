//! Full-conformal prediction for penalized GAMs — including the
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
//!   C_α = { z : #{i : e_i(z) > e_*(z)} + U (1 + #{i : e_i(z) = e_*(z)}) > α (n+1) }
//! ```
//!
//! Validity needs ONLY exchangeability of the n+1 points and SYMMETRY of
//! the fitting map (it must treat the augmented row like any other row).
//! No model correctness, no asymptotics, no held-out fold. One independent
//! `U ~ Uniform[0,1)` is shared across the entire inversion. For a symmetric
//! fitting map on exchangeable supplied rows, ideal smoothed ranks give exact
//! marginal coverage. This is not conditional-on-features coverage. Numerical
//! enclosures in Layers 2 and 3 can over-cover, and learned training-only
//! bases/penalties do not automatically satisfy augmented-row symmetry.
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
//! - **Layer 1 (implemented below):** Gaussian identity at fixed ρ.
//!   The augmented fit is affine in `z`; one factorization gives the stored
//!   affine coefficients. The event sweep certifies membership for finite
//!   representable f64 candidates using exact dyadic comparisons of those
//!   coefficients. Returned coordinates encode that discrete candidate set;
//!   they are not exact real-valued roots or an exact-real fitting guarantee.
//! - **Layer 2 — GLM families (implemented in
//!   [`super::full_conformal_glm`], certified):** Binomial, Poisson,
//!   negative binomial and Gamma at the frozen penalty. Discrete supports
//!   are searched with certified tails for count families, and the Gamma
//!   continuum is walked with certified Newton refits. Discrete ties use
//!   independent smoothed-rank randomization. Unresolved comparisons are
//!   retained as a conservative numerical enclosure: its coverage need not
//!   equal `1 − α`. Marginal coverage assumes exchangeable supplied rows and
//!   a fixed symmetric basis/penalty construction, not arbitrary learned bases.
//! - **Layer 3 (implemented in [`honest`], conservative numerical enclosure):** the Gaussian-identity map that RE-SELECTS the smoothing
//!   strength by REML on every augmented data set — the first
//!   full-conformal procedure whose fitting map treats the test row like a
//!   training row all the way up to ρ̂. A proven bound on where the global
//!   REML minimizer can lie, plus cold local refits at the set's endpoints.
//!   Every row carries a [`ConformalCertificate`]: `exact_frozen`,
//!   `conservative_frozen`,
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
//! Sherman–Morrison when the training normal is SPD — the test residual's
//! slope is positive, so e_*(z)
//! is genuinely V-shaped and the rank function is well-defined everywhere.
//!
//! The comparison `e_i(z) ≥ e_*(z)` ⟺ `(r_i−r_*)(r_i+r_*) ≥ 0` flips only
//! at roots of two LINEAR equations per i. Collect ≤ 2n roots, sort, and
//! the rank of e_* is constant on each open interval between consecutive
//! roots: a root-event sweep tracks strict and tied comparisons on each open
//! gap and at each root. Endpoint-inclusion flags preserve excluded roots and
//! isolated member points; taking the closure would change the randomized set
//! for atomic response laws. The sweep costs O(n log n), without refits.
//!
//! Unboundedness is honest, not an error: if `|slope(r_*)| ≤ |slope(r_i)|`
//! for enough i, far-out candidates are never extreme and the set is a
//! half-line or ℝ (low-information / high-leverage regimes). We return the
//! interval list as-is, with ±∞ as open bounds — same honesty convention as
//! the split module's `+∞` multiplier.
//!
//! # Layer 2: GLM families (see [`super::full_conformal_glm`])
//!
//! `β̂(z)` solves the augmented penalized score equation of the family at
//! the frozen Sλ. Discrete families (Binomial, Poisson, negative binomial)
//! have finite support or use a certified tail when one is available. The
//! numerical engine encloses unresolved candidates conservatively, including
//! the full support if a tail cannot be certified. Gamma is continuous and
//! is walked with certified refits and conservative bounds. The
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
use rand::RngExt;

use gam_linalg::faer_ndarray::{FaerCholesky, fast_av};

pub mod honest;
pub use honest::{
    ConformalCertificate, ConformalRefusal, HonestConformalCost, HonestFullConformal,
    honest_full_conformal,
};

#[cfg(test)]
mod test_support;

/// One maximal interval encoding candidate values retained in the prediction set.
/// For Layer 1, membership is certified only for finite representable f64
/// candidates and the stored affine coefficients, not exact real-valued roots.
/// Endpoints may be infinite (honest unboundedness in low-information /
/// high-leverage regimes).
#[derive(Clone, Debug, PartialEq)]
pub struct ConformalInterval {
    pub lo: f64,
    pub hi: f64,
    pub lo_closed: bool,
    pub hi_closed: bool,
}

impl ConformalInterval {
    /// Closed finite endpoints; infinities are bounds, never members.
    pub fn closed(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            lo_closed: lo.is_finite(),
            hi_closed: hi.is_finite(),
        }
    }

    pub fn contains(&self, value: f64) -> bool {
        value.is_finite()
            && (value > self.lo || (self.lo_closed && value == self.lo))
            && (value < self.hi || (self.hi_closed && value == self.hi))
    }
}

/// The rank threshold `τ = α(n + 1)` a conformal p-value is compared with
/// (membership compares strict-plus-randomized-tie rank mass with τ).
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

/// A full-conformal prediction set with explicit finite-endpoint membership.
/// Layer-1 coordinates encode the representable-f64 candidate set; other
/// certificates can denote conservative numerical enclosures.
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

pub(crate) fn validate_tie_uniform(value: f64) -> Result<(), String> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "full conformal: tie uniform must be in [0, 1), got {value}"
        ))
    }
}

/// The smallest strict-dominating count k with k+U > α(n+1), or n+1.
/// Counting uncertain ties as possible strict dominators gives a conservative
/// upper bound for the honest numerical enclosure.
fn required_dominating_count(n: usize, alpha: f64, tie_uniform: f64) -> usize {
    let threshold = conformal_rank_threshold(alpha, n + 1);
    (0..=n)
        .find(|&count| tie_uniform + count as f64 > threshold)
        .unwrap_or(n + 1)
}

/// Gaussian-identity full-conformal engine at fixed Sλ (Layer 1).
///
/// One factorization of the SPD training normal `A = XᵀX + Sλ`; every
/// candidate-z quantity is affine thereafter. Exact inversion refers to
/// representable f64 candidates and the stored dyadic affine coefficients.
/// Rounded interval coordinates do not claim exact real-valued boundaries.
pub struct ExactGaussianFullConformal {
    /// Affine residual coefficients: `r_i(z) = u[i] + w[i]·z` for the n
    /// training rows, and the test residual in the LAST slot.
    u: Array1<f64>,
    w: Array1<f64>,
    n: usize,
    tie_uniform: f64,
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
        Self::new_with_uniform(x, y, prior_weights, s_lambda, x_star, rand::rng().random())
    }

    /// Construct one inversion with an externally supplied independent U.
    pub fn new_with_uniform(
        x: &Array2<f64>,
        y: &Array1<f64>,
        prior_weights: &Array1<f64>,
        s_lambda: &Array2<f64>,
        x_star: &Array1<f64>,
        tie_uniform: f64,
    ) -> Result<Self, String> {
        validate_tie_uniform(tie_uniform)?;
        validate_inputs(x, y, prior_weights, s_lambda, x_star)?;
        let n = x.nrows();

        // Positive exact test-residual slope requires the training normal A
        // to be SPD (augmented SPD alone is insufficient). Sherman–Morrison
        // avoids subtracting nearly equal leverage values in 1-x'M_aug^-1 x.
        let normal = x.t().dot(x) + s_lambda;
        let chol = normal
            .cholesky(Side::Lower)
            .map_err(|e| format!("full conformal: training normal matrix not SPD: {e:?}"))?;
        let beta = chol.solvevec(&x.t().dot(y));
        let direction = chol.solvevec(x_star);
        let leverage = x_star.dot(&direction);
        let denominator = 1.0 + leverage;
        if !(leverage.is_finite() && leverage >= 0.0 && denominator.is_finite()) {
            return Err("full conformal: test leverage or 1+leverage is not representable".into());
        }
        let mut u = Array1::<f64>::zeros(n + 1);
        let mut w = Array1::<f64>::zeros(n + 1);
        u[n] = -x_star.dot(&beta) / denominator;
        w[n] = denominator.recip();
        let fitted = fast_av(x, &beta);
        let influence = fast_av(x, &direction);
        for i in 0..n {
            u[i] = y[i] - fitted[i] - influence[i] * u[n];
            w[i] = -influence[i] / denominator;
        }
        if w[n] <= 0.0 || u.iter().chain(w.iter()).any(|v| !v.is_finite()) {
            return Err(
                "full conformal: reciprocal 1/(1+training leverage) must be positive; \
                 residual coefficients and reciprocal must be representable"
                    .to_string(),
            );
        }
        Ok(Self {
            u,
            w,
            n,
            tie_uniform,
        })
    }

    /// The frozen plug-in mean `x_*ᵀ(XᵀX + Sλ)⁻¹Xᵀy`: the candidate at which
    /// the test residual vanishes.
    pub fn plug_in_mean(&self) -> f64 {
        -self.u[self.n] / self.w[self.n]
    }

    /// Error-free product of two finite dyadic values. A product requiring
    /// bits below the subnormal quantum cannot be represented by an expansion
    /// of f64 values, so the endpoint certificate refuses it explicitly.
    fn endpoint_product(a: f64, b: f64) -> Result<(f64, f64), String> {
        if a == 0.0 || b == 0.0 {
            return Ok((0.0, 0.0));
        }
        let lowest_exponent = |value: f64| {
            let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
            let encoded = (bits >> 52) as i32;
            let significand =
                (bits & 0x000f_ffff_ffff_ffff) | if encoded == 0 { 0 } else { 1u64 << 52 };
            let exponent = if encoded == 0 {
                -1074
            } else {
                encoded - 1023 - 52
            };
            exponent + significand.trailing_zeros() as i32
        };
        if lowest_exponent(a) + lowest_exponent(b) < -1074 {
            return Err("full conformal: endpoint product underflow cannot be certified".into());
        }
        let high = a * b;
        if !high.is_finite() {
            return Err("full conformal: endpoint product overflow cannot be certified".into());
        }
        Ok((a.mul_add(b, -high), high))
    }

    /// Exact sign of r_i(z)−r_*(z), or r_i(z)+r_*(z), for the stored dyadic
    /// coefficients. Six-term error-free expansion avoids calling a rounded
    /// rational coordinate an exact score tie.
    fn endpoint_factor_leading(&self, i: usize, z: f64, subtract: bool) -> Result<f64, String> {
        let (il, ih) = Self::endpoint_product(self.w[i], z)?;
        let (sl, sh) = Self::endpoint_product(self.w[self.n], z)?;
        let direction = if subtract { -1.0 } else { 1.0 };
        let mut expansion = [0.0; 6];
        let mut length = 0;
        for scalar in [
            il,
            ih,
            direction * sl,
            direction * sh,
            self.u[i],
            direction * self.u[self.n],
        ] {
            let mut q = scalar;
            let mut next = 0;
            for j in 0..length {
                let term = expansion[j];
                let sum = q + term;
                if !sum.is_finite() {
                    return Err(
                        "full conformal: endpoint expansion overflow cannot be certified".into(),
                    );
                }
                let virtual_term = sum - q;
                let error = (q - (sum - virtual_term)) + (term - virtual_term);
                if error != 0.0 {
                    expansion[next] = error;
                    next += 1;
                }
                q = sum;
            }
            if q != 0.0 || next == 0 {
                expansion[next] = q;
                next += 1;
            }
            length = next;
        }
        Ok(expansion[length - 1])
    }

    fn endpoint_factor_sign(&self, i: usize, z: f64, subtract: bool) -> Result<i8, String> {
        let leading = self.endpoint_factor_leading(i, z, subtract)?;
        Ok(if leading > 0.0 {
            1
        } else if leading < 0.0 {
            -1
        } else {
            0
        })
    }

    /// Isolate the true linear root between adjacent representable candidates.
    /// A rounded endpoint itself is ranked separately; it is not presumed tied.
    fn isolated_score_root(&self, i: usize, subtract: bool, a: f64, b: f64) -> Result<f64, String> {
        let mut root = -b / a;
        if !root.is_finite() || (root == 0.0 && b != 0.0) {
            return Err("full conformal: score breakpoint is not representable".into());
        }
        // Rounded coefficient subtraction can displace -b/a by several ulps.
        // Correct it with the exact expansion residual before certifying the
        // adjacent representable candidates. Every accepted root is still
        // checked below; this bounded refinement grants no tolerance band.
        for _ in 0..4 {
            let residual = self.endpoint_factor_leading(i, root, subtract)?;
            if residual == 0.0 {
                return Ok(root);
            }
            let corrected = root - residual / a;
            if !corrected.is_finite() {
                return Err("full conformal: root correction overflow".into());
            }
            if corrected == root {
                break;
            }
            root = corrected;
        }
        if self.endpoint_factor_sign(i, root, subtract)? == 0 {
            return Ok(root);
        }
        let lower = root.next_down();
        let upper = root.next_up();
        if !lower.is_finite() || !upper.is_finite() {
            return Err("full conformal: score breakpoint neighbors are not representable".into());
        }
        let left = self.endpoint_factor_sign(i, lower, subtract)?;
        let right = self.endpoint_factor_sign(i, upper, subtract)?;
        if left == 0 {
            return Ok(lower);
        }
        if right == 0 {
            return Ok(upper);
        }
        let slope = if a > 0.0 { 1 } else { -1 };
        if left != -slope || right != slope {
            return Err("full conformal: score breakpoint isolation cannot be certified".into());
        }
        Ok(root)
    }

    /// Invert the affine score comparisons for representable f64 candidates.
    /// Endpoint membership uses exact dyadic signs, including at rounded roots.
    /// Uncertifiable arithmetic is refused. One U is shared by all candidates.
    pub fn prediction_set(&self, alpha: f64) -> Result<FullConformalSet, String> {
        if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
            return Err(format!(
                "full conformal: alpha must be in (0, 1), got {alpha}"
            ));
        }
        let n = self.n;
        let mut relations = Vec::<i8>::with_capacity(n);
        let mut events = Vec::<(f64, usize)>::with_capacity(2 * n);
        let sign = |v: f64| {
            if v > 0.0 {
                1i8
            } else if v < 0.0 {
                -1
            } else {
                0
            }
        };
        for i in 0..n {
            // e_i²-e_*² is the product of these two linear factors.
            let factors = [
                (self.w[i] - self.w[n], self.u[i] - self.u[n]),
                (self.w[i] + self.w[n], self.u[i] + self.u[n]),
            ];
            if factors
                .iter()
                .any(|&(a, b)| !a.is_finite() || !b.is_finite())
            {
                return Err("full conformal: affine score comparison overflow".into());
            }
            if factors.iter().any(|&(a, b)| a == 0.0 && b == 0.0) {
                relations.push(0); // identical absolute scores, for every z
                continue;
            }
            let mut relation = 1;
            for (factor, (a, b)) in factors.into_iter().enumerate() {
                relation *= if a != 0.0 { -sign(a) } else { sign(b) };
                if a != 0.0 {
                    let z = self.isolated_score_root(i, factor == 0, a, b)?;
                    events.push((z, i));
                }
            }
            relations.push(relation);
        }
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        let threshold = conformal_rank_threshold(alpha, n + 1);
        let member = |greater: usize, tied: usize| {
            greater as f64 + self.tie_uniform * (1 + tied) as f64 > threshold
        };
        let mut greater = relations.iter().filter(|&&r| r > 0).count();
        let tied = relations.iter().filter(|&&r| r == 0).count();
        let mut last_root = vec![usize::MAX; n];
        let mut intervals = Vec::<ConformalInterval>::new();
        let append = |intervals: &mut Vec<ConformalInterval>, piece: ConformalInterval| {
            if let Some(last) = intervals.last_mut()
                && last.hi == piece.lo
                && (last.hi_closed || piece.lo_closed)
            {
                last.hi = piece.hi;
                last.hi_closed = piece.hi_closed;
            } else {
                intervals.push(piece);
            }
        };
        let mut left = f64::NEG_INFINITY;
        let mut cursor = 0;
        while cursor < events.len() {
            let z = events[cursor].0;
            if left < z && member(greater, tied) {
                append(
                    &mut intervals,
                    ConformalInterval {
                        lo: left,
                        hi: z,
                        lo_closed: false,
                        hi_closed: false,
                    },
                );
            }
            let mut end = cursor + 1;
            while end < events.len() && events[end].0 == z {
                end += 1;
            }
            let (mut root_greater, mut root_tied) = (greater, tied);
            for &(_, i) in &events[cursor..end] {
                if last_root[i] != cursor {
                    root_greater -= usize::from(relations[i] > 0);
                    let relation = self.endpoint_factor_sign(i, z, true)?
                        * self.endpoint_factor_sign(i, z, false)?;
                    root_greater += usize::from(relation > 0);
                    root_tied += usize::from(relation == 0);
                    last_root[i] = cursor;
                }
            }
            if member(root_greater, root_tied) {
                append(&mut intervals, ConformalInterval::closed(z, z));
            }
            for &(_, i) in &events[cursor..end] {
                if relations[i] > 0 {
                    greater -= 1;
                } else {
                    greater += 1;
                }
                relations[i] = -relations[i];
            }
            left = z;
            cursor = end;
        }
        if member(greater, tied) {
            append(
                &mut intervals,
                ConformalInterval {
                    lo: left,
                    hi: f64::INFINITY,
                    lo_closed: false,
                    hi_closed: false,
                },
            );
        }
        Ok(FullConformalSet {
            intervals,
            alpha,
            n_augmented: n + 1,
        })
    }
}

/// Wilkinson growth for the Gaussian REML response's arithmetic: the factor
/// [`honest`] charges against every magnitude sum.
///
/// The `p`-terms count the Cholesky of `A(λ)`, its solves and its traces. The
/// `n`-terms count the residual sums the penalized RSS is formed from (#2280):
/// two passes of `X·β` over the rows, with their subtractions, squares and sums.
fn response_solve_growth(n: usize, p: usize) -> f64 {
    gam_linalg::roundoff::accumulation_growth(2 * p * p * p + 8 * p * p + 8 * p + 4 * n * p + 8 * n)
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
    /// rank set or its certified numerical enclosure. Every row carries unit
    /// weight: only models trained without
    /// prior weights persist this penalty.
    ///
    /// The rows need not be the training rows. Marginal coverage requires
    /// exchangeable supplied rows and a fitting map symmetric in all augmented
    /// rows, including the basis and penalty construction. A basis or penalty
    /// learned on independent data can be held fixed; a training-only learned
    /// construction does not automatically satisfy that assumption.
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
        self.interval_with_uniform(x_star, alpha, rand::rng().random())
    }

    /// The same outer envelope with an explicitly supplied independent U.
    pub fn interval_with_uniform(
        &self,
        x_star: &Array1<f64>,
        alpha: f64,
        tie_uniform: f64,
    ) -> Result<ExactFullConformalInterval, String> {
        if x_star.len() != self.p() {
            return Err(format!(
                "exact full conformal: x_* has {} entries but the fit has {} coefficients",
                x_star.len(),
                self.p()
            ));
        }
        let weights = Array1::<f64>::ones(self.n());
        let row = honest::honest_full_conformal_with_uniform(
            &self.x,
            &self.y,
            &weights,
            &self.s_lambda,
            self.penalty_count,
            x_star,
            alpha,
            tie_uniform,
        )?;
        let (lo, hi) = match (row.set.intervals.first(), row.set.intervals.last()) {
            (Some(first), Some(last)) => (first.lo, last.hi),
            // An empty randomized set has no envelope. Never substitute a
            // point that the rank rule excluded.
            _ => (f64::NAN, f64::NAN),
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
            ExactGaussianFullConformal::new_with_uniform(&x, &y, &weights, &s_lambda, &x_star, 0.5)
                .expect("engine");
        let alpha = 0.2;
        let set = engine.prediction_set(alpha).expect("prediction set");
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
            let greater = (0..n)
                .filter(|&i| {
                    let mu_i: f64 = x.row(i).dot(&beta);
                    (y[i] - mu_i).abs() > e_star
                })
                .count();
            (0.5 + greater as f64) > alpha * (n as f64 + 1.0)
        };

        let z_lo = set.intervals.first().map(|i| i.lo).unwrap_or(-5.0) - 2.0;
        let z_hi = set.intervals.last().map(|i| i.hi).unwrap_or(5.0) + 2.0;
        let z_lo = if z_lo.is_finite() { z_lo } else { -50.0 };
        let z_hi = if z_hi.is_finite() { z_hi } else { 50.0 };
        let grid = 4001usize;
        for g in 0..grid {
            let z = z_lo + (z_hi - z_lo) * g as f64 / (grid as f64 - 1.0);
            let in_set = set.intervals.iter().any(|itv| itv.contains(z));
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
            set.intervals.iter().any(|itv| itv.contains(mu_star)),
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
            ExactGaussianFullConformal::new_with_uniform(&x, &y, &weights, &s_lambda, &x_star, 0.5)
                .expect("engine");

        let set = engine.prediction_set(0.25).expect("prediction set");
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
            ExactGaussianFullConformal::new_with_uniform(&x, &y, &weights, &s_lambda, &x_star, 0.5)
                .expect("engine");

        let set = engine.prediction_set(0.25).expect("prediction set");
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
            tie_uniform: 0.5,
        };

        let set = engine.prediction_set(0.25).expect("prediction set");
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
            let error =
                ExactGaussianFullConformal::new_with_uniform(&x, &y, &weights, &s, &x_star, 0.5)
                    .err()
                    .expect("a prior weight that is not exactly one must be refused");
            assert!(error.contains("unit prior weights"), "{error}");
        }
        // Non-vacuity: exact unit weights are admitted.
        let weights = Array1::<f64>::ones(n);
        assert!(
            ExactGaussianFullConformal::new_with_uniform(&x, &y, &weights, &s, &x_star, 0.5)
                .is_ok()
        );
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
        assert!(
            penalty
                .with_labeled_rows(x.slice(ndarray::s![.., ..2]).to_owned(), y)
                .is_err()
        );
    }
}

#[cfg(test)]
mod smoothed_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ridge_set_keeps_both_boundary_points_excluded() {
        let engine = ExactGaussianFullConformal::new_with_uniform(
            &array![[1.0]],
            &array![1.0],
            &array![1.0],
            &array![[1.0]],
            &array![1.0],
            0.5,
        )
        .unwrap();
        let set = engine.prediction_set(0.6).unwrap();
        assert_eq!(set.intervals.len(), 1);
        let interval = &set.intervals[0];
        assert_eq!((interval.lo, interval.hi), (-1.0, 1.0));
        assert!(!interval.lo_closed && !interval.hi_closed);
        assert!(interval.contains(0.0));
        assert!(!interval.contains(-1.0) && !interval.contains(1.0));
    }

    #[test]
    fn persistent_ties_and_singletons_follow_the_same_uniform() {
        for u in [0.0, 0.25, 0.5, 0.75, 1.0 - f64::EPSILON] {
            let engine = ExactGaussianFullConformal::new_with_uniform(
                &array![[1.0]],
                &array![0.0],
                &array![1.0],
                &array![[0.0]],
                &array![1.0],
                u,
            )
            .unwrap();
            let set = engine.prediction_set(0.5).unwrap();
            assert_eq!(set.intervals.is_empty(), u <= 0.5);
            if u > 0.5 {
                assert_eq!(set.intervals.len(), 1);
                assert_eq!(
                    (set.intervals[0].lo, set.intervals[0].hi),
                    (f64::NEG_INFINITY, f64::INFINITY)
                );
                assert!(set.intervals[0].contains(0.0));
                assert!(!set.intervals[0].contains(f64::INFINITY));
            }
            let singleton = ExactGaussianFullConformal::new_with_uniform(
                &array![[0.0]],
                &array![0.0],
                &array![1.0],
                &array![[1.0]],
                &array![1.0],
                u,
            )
            .unwrap()
            .prediction_set(0.5)
            .unwrap();
            assert_eq!(singleton.intervals.is_empty(), u <= 0.5);
            if u > 0.5 {
                assert_eq!(
                    singleton.intervals,
                    vec![ConformalInterval::closed(0.0, 0.0)]
                );
            }
        }
    }

    #[test]
    fn invalid_uniform_alpha_and_overflow_are_refused() {
        // Rounding a nonzero root to zero would misclassify an atom at zero.
        let underflow = ExactGaussianFullConformal {
            u: array![1e-200, 0.0],
            w: array![1e200, 1.0],
            n: 1,
            tie_uniform: 0.5,
        };
        assert!(
            underflow
                .prediction_set(0.6)
                .unwrap_err()
                .contains("breakpoint")
        );

        for u in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.0] {
            assert!(
                ExactGaussianFullConformal::new_with_uniform(
                    &array![[1.0]],
                    &array![0.0],
                    &array![1.0],
                    &array![[1.0]],
                    &array![1.0],
                    u,
                )
                .is_err()
            );
        }
        let engine = ExactGaussianFullConformal::new_with_uniform(
            &array![[1.0]],
            &array![0.0],
            &array![1.0],
            &array![[0.0]],
            &array![1e100],
            0.5,
        )
        .unwrap();
        assert!(engine.w[1] > 0.0);
        let set = engine.prediction_set(0.6).unwrap();
        assert_eq!(set.intervals.len(), 2);
        assert!(set.intervals[0].contains(-1.0) && set.intervals[1].contains(1.0));
        assert!(!set.intervals.iter().any(|piece| piece.contains(0.0)));
        for alpha in [0.0, 1.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(engine.prediction_set(alpha).is_err());
        }
        assert!(
            ExactGaussianFullConformal::new_with_uniform(
                &array![[1.0]],
                &array![0.0],
                &array![1.0],
                &array![[0.0]],
                &array![1e200],
                0.5,
            )
            .err()
            .unwrap()
            .contains("leverage")
        );
        // An augmented SPD normal is insufficient when training is singular.
        assert!(
            ExactGaussianFullConformal::new_with_uniform(
                &array![[0.0]],
                &array![0.0],
                &array![1.0],
                &array![[0.0]],
                &array![1.0],
                0.5,
            )
            .err()
            .unwrap()
            .contains("training normal")
        );
    }

    #[test]
    fn affine_inversion_matches_augmented_refits_and_permutations() {
        let x = array![[1.0, -1.0], [1.0, -0.4], [1.0, 0.1], [1.0, 0.6], [1.0, 1.2]];
        let y = array![-1.0, 0.5, -0.3, 1.2, 0.7];
        let penalty = array![[0.4, 0.0], [0.0, 0.8]];
        let star = array![1.0, 0.35];
        let weights = Array1::ones(5);
        let order = [2, 4, 0, 3, 1];
        let mut normal = x.t().dot(&x) + &penalty;
        for i in 0..2 {
            for j in 0..2 {
                normal[[i, j]] += star[i] * star[j];
            }
        }
        let chol = normal.cholesky(Side::Lower).unwrap();
        for u in [0.1, 0.5, 0.9] {
            let engine =
                ExactGaussianFullConformal::new_with_uniform(&x, &y, &weights, &penalty, &star, u)
                    .unwrap();
            let permuted = ExactGaussianFullConformal::new_with_uniform(
                &x.select(ndarray::Axis(0), &order),
                &y.select(ndarray::Axis(0), &order),
                &weights,
                &penalty,
                &star,
                u,
            )
            .unwrap();
            for alpha in [0.15, 0.4, 0.8] {
                let set = engine.prediction_set(alpha).unwrap();
                let reordered = permuted.prediction_set(alpha).unwrap();
                for k in -32..=32 {
                    let z = k as f64 / 8.0;
                    let beta = chol.solvevec(&(x.t().dot(&y) + &star * z));
                    let test = (z - star.dot(&beta)).abs();
                    let scores: Vec<_> = x
                        .rows()
                        .into_iter()
                        .zip(y.iter())
                        .map(|(r, &v)| (v - r.dot(&beta)).abs())
                        .collect();
                    let greater = scores.iter().filter(|&&e| e > test).count();
                    let tied = scores.iter().filter(|&&e| e == test).count();
                    let expected = greater as f64 + u * (1 + tied) as f64 > alpha * 6.0;
                    let member = set.intervals.iter().any(|piece| piece.contains(z));
                    assert_eq!(member, expected, "z={z}, U={u}, alpha={alpha}");
                    assert_eq!(
                        member,
                        reordered.intervals.iter().any(|piece| piece.contains(z))
                    );
                }
            }
        }
    }

    #[test]
    fn honest_frozen_path_preserves_uniform_and_empty_set() {
        use super::honest::honest_full_conformal_with_uniform;
        let x = array![[1.0]];
        let y = array![0.0];
        let weights = array![1.0];
        let penalty = array![[0.0]];
        let star = array![1.0];
        for u in [0.0, 0.2, 0.8] {
            let result = honest_full_conformal_with_uniform(
                &x,
                &y,
                &weights,
                &penalty,
                Some(0),
                &star,
                0.5,
                u,
            )
            .unwrap();
            let direct =
                ExactGaussianFullConformal::new_with_uniform(&x, &y, &weights, &penalty, &star, u)
                    .unwrap()
                    .prediction_set(0.5)
                    .unwrap();
            assert_eq!(result.set.intervals, direct.intervals);
        }
        assert!(
            honest_full_conformal_with_uniform(
                &x,
                &y,
                &weights,
                &penalty,
                Some(0),
                &star,
                0.5,
                1.0
            )
            .is_err()
        );
    }
    #[test]
    fn empty_randomized_set_has_no_point_envelope() {
        let substrate = ExactFullConformalSubstrate {
            x: array![[1.0]],
            y: array![0.0],
            s_lambda: array![[0.0]],
            penalty_count: Some(0),
        };
        let empty = substrate
            .interval_with_uniform(&array![1.0], 0.5, 0.25)
            .unwrap();
        assert!(empty.set.intervals.is_empty());
        assert!(empty.lo.is_nan() && empty.hi.is_nan());
        let whole = substrate
            .interval_with_uniform(&array![1.0], 0.5, 0.75)
            .unwrap();
        assert_eq!((whole.lo, whole.hi), (f64::NEG_INFINITY, f64::INFINITY));
        assert!(whole.set.intervals[0].contains(0.0));
    }
    #[test]
    fn rounded_rational_roots_use_actual_endpoint_rank() {
        // For these exact coefficients, FMA computes each comparison sign
        // without the false zero produced by separate multiplication/addition.
        let controls = [(3.0, 0.4, 0.6), (10.0, 0.8, 0.6)];
        for (slope, uniform, alpha) in controls {
            let engine = ExactGaussianFullConformal {
                u: array![1.0, 0.0],
                w: array![0.0, slope],
                n: 1,
                tie_uniform: uniform,
            };
            let root = 1.0 / slope;
            assert_ne!((-slope).mul_add(root, 1.0), 0.0);
            let set = engine.prediction_set(alpha).unwrap();
            for z in [root.next_down(), root, root.next_up()] {
                let difference = (-slope).mul_add(z, 1.0);
                let sum = slope.mul_add(z, 1.0);
                let greater = usize::from(difference.signum() == sum.signum());
                let tied = usize::from(difference == 0.0 || sum == 0.0);
                let expected = greater as f64 + uniform * (1 + tied) as f64 > alpha * 2.0;
                assert_eq!(
                    set.intervals.iter().any(|piece| piece.contains(z)),
                    expected,
                    "slope={slope}, z={z:.18e}"
                );
            }
        }
    }

    #[test]
    fn colliding_rounded_roots_do_not_create_false_ties() {
        let root = 1.0f64 / 3.0;
        let second_intercept = 1.0f64.next_up();
        let second_denominator = 3.0f64.next_up().next_up();
        assert_eq!(second_intercept / second_denominator, root);
        let second_slope = 4.0 - second_denominator;
        assert!((-3.0f64).mul_add(root, 1.0) > 0.0);
        assert!((-second_denominator).mul_add(root, second_intercept) < 0.0);
        for (uniform, alpha) in [(0.2, 0.3), (0.8, 0.7)] {
            let engine = ExactGaussianFullConformal {
                u: array![1.0, second_intercept, 0.0],
                w: array![1.0, second_slope, 4.0],
                n: 2,
                tie_uniform: uniform,
            };
            let set = engine.prediction_set(alpha).unwrap();
            for z in [root.next_down(), root, root.next_up()] {
                let mut greater = 0usize;
                let mut tied = 0usize;
                for (intercept, slope) in [(1.0, 1.0), (second_intercept, second_slope)] {
                    let difference = (slope - 4.0).mul_add(z, intercept);
                    let sum = (slope + 4.0).mul_add(z, intercept);
                    tied += usize::from(difference == 0.0 || sum == 0.0);
                    greater += usize::from(
                        difference != 0.0 && sum != 0.0 && difference.signum() == sum.signum(),
                    );
                }
                let expected = greater as f64 + uniform * (1 + tied) as f64 > alpha * 3.0;
                assert_eq!(
                    set.intervals.iter().any(|piece| piece.contains(z)),
                    expected,
                    "U={uniform}, alpha={alpha}, z={z:.18e}"
                );
            }
        }
    }
}
