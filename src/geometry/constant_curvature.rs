//! The continuous constant-curvature family `M_κ` — curvature as an
//! ESTIMAND, not an architecture choice (#944, stages 1–2).
//!
//! # One chart, one parameter
//!
//! [`ConstantCurvature`] realizes the unified κ-stereographic model: a
//! single coordinate chart on `ℝ^d` (a ball of radius `1/√−κ` when κ < 0,
//! all of `ℝ^d` when κ ≥ 0) with conformal metric
//!
//! ```text
//!   g_x = λ_x² · δ,        λ_x = 2 / (1 + κ‖x‖²)
//! ```
//!
//! that is S^d(1/√κ) for κ > 0 (stereographic projection, antipode
//! excluded), the Poincaré ball for κ < 0 (EXACTLY the in-tree
//! `poincare.rs` convention at κ = −1, including the radial isometry
//! `d(0, y) = 2·artanh‖y‖`), and flat space at κ = 0. The κ = 0 member
//! carries metric `4δ` — Euclidean up to the global isometry `x ↦ 2x` —
//! because the conformal gauge `λ_0 = 2` is what makes the family analytic
//! through zero; cross-checks against `euclidean.rs` use that isometry.
//!
//! # Flat space is a removable point, not a special case
//!
//! Every operation factors through the generalized trigonometric functions
//! written as functions of the single variable `u = κ·t²`:
//!
//! ```text
//!   C(u) = Σ_m (−u)^m / (2m)!    = cos(√κ t)        | cosh(√−κ t)
//!   S(u) = Σ_m (−u)^m / (2m+1)!  = sin(√κ t)/(√κ t) | sinh(√−κ t)/(√−κ t)
//!   T(w) = Σ_m (−w)^m / (2m+1)   = atan(√w)/√w      | artanh(√−w)/√−w
//! ```
//!
//! `C` and `S` are ENTIRE in `u` — spherical, flat, and hyperbolic are one
//! analytic object, and κ-differentiation is legitimate calculus rather
//! than a limit argument. Near `u = 0` the implementations switch to the
//! power series (the same removable-singularity discipline as the C∞
//! sphere jets already in the tree); away from it, to the closed
//! trig/hyperbolic forms. Derivative stacks to fourth order come from the
//! exact mutual recurrences
//!
//! ```text
//!   C⁽ʲ⁺¹⁾ = −S⁽ʲ⁾/2
//!   S⁽ʲ⁺¹⁾ = (C⁽ʲ⁾ − (2j+1)·S⁽ʲ⁾) / (2u)
//!   T⁽ʲ⁺¹⁾ = (R⁽ʲ⁾ − (2j+1)·T⁽ʲ⁾) / (2w),   R(w) = 1/(1+w)
//! ```
//!
//! (differentiate `2u·S′ = C − S` and `2w·T′ = R − T` j times by Leibniz).
//!
//! # κ-jets ride the #932 tower — no new hand calculus
//!
//! Stage 2 of #944 (exact ∂/∂κ and ∂²/∂κ² of distance/log so κ can join
//! the outer REML optimization as a ψ-coordinate) is implemented as a
//! CLIENT of [`crate::families::jet_tower::Tower4`]: the same geometric
//! program is evaluated with κ seeded as a 1-variable jet, the scalar
//! primitives `C/S/T` entering through their hand-certified `[f64; 5]`
//! derivative stacks via `compose_unary`. Humans own primitive stability,
//! the algebra owns composition — the identical split as the row-NLL
//! towers, so the geometry κ-derivatives can never desync from the
//! geometry values: they are the same expression.
//!
//! A small piece of luck makes the jets clean: in this chart the log map
//! simplifies to `log_x(y) = (1 + κ‖x‖²) · T(κ‖w‖²) · w` with
//! `w = (−x) ⊕_κ y` — the norms cancel, no square root appears, and the
//! expression is smooth through `w = 0` and through `κ = 0`.
//!
//! # Where this is going (#944 stages 3–4)
//!
//! κ joins the outer optimization on the established ψ-channel (the Matérn
//! κ optimizer is the template; the ψ-gradient trap on that channel — the
//! iso-κ FD desync — was fixed under #901, which is what makes this issue
//! attemptable). Profile-likelihood CIs for κ̂ and the κ = 0 likelihood
//! test then turn "we chose hyperbolic space" into "κ̂ = −1.8 (95% CI
//! −2.6, −1.1)" — and the discrete topology stack only adjudicates
//! genuinely non-homotopic candidates.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::manifold::{GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold};
use crate::families::jet_tower::Tower4;

/// Branch threshold for the `C`/`S` series in `u = κt²`. The series terms
/// decay factorially, so at `|u| ≤ 0.5` the truncation error of
/// [`CS_SERIES_TERMS`] terms is far below one ulp; beyond it the closed
/// trig/hyperbolic forms are well-conditioned.
const CS_SERIES_U_MAX: f64 = 0.5;

/// Series length for `C`/`S` stacks. Term m of the j-th derivative series
/// is bounded by `|u|^m / (2m)!`; 18 terms at `|u| = 0.5` is < 1e−40.
const CS_SERIES_TERMS: usize = 18;

/// Branch threshold for the `T` series in `w = κr²`. `T`'s series is only
/// geometric (radius 1), so the switch happens earlier than for `C`/`S`
/// and the term count is correspondingly larger.
const T_SERIES_W_MAX: f64 = 0.25;

/// Series length for the `T` stack: `0.25^48 ≈ 1e−29` dominates the
/// truncation tail at the branch edge.
const T_SERIES_TERMS: usize = 48;

/// Möbius-addition denominators below this are treated as the κ > 0
/// antipodal singularity (the one point the stereographic chart misses).
const MOBIUS_DENOM_EPS: f64 = 1.0e-14;

/// Derivative stacks `[f, f′, f″, f‴, f⁗]` (in `u`) of the entire
/// functions `C(u)` and `S(u)`. Exact: series inside
/// [`CS_SERIES_U_MAX`], closed forms + the mutual recurrence outside.
pub fn cs_stacks(u: f64) -> ([f64; 5], [f64; 5]) {
    if u.abs() <= CS_SERIES_U_MAX {
        let mut c = [0.0; 5];
        let mut s = [0.0; 5];
        for j in 0..5 {
            // a_m = (−1)^m m!/(m−j)! u^{m−j} / (2m)!  (C)  resp. /(2m+1)! (S),
            // started at m = j and advanced by exact term ratios.
            let mut term_c = 1.0;
            let mut term_s = 1.0;
            for f in 1..=j {
                let fj = f as f64;
                term_c *= -fj / ((2.0 * fj - 1.0) * (2.0 * fj));
                term_s *= -fj / ((2.0 * fj) * (2.0 * fj + 1.0));
            }
            let mut acc_c = term_c;
            let mut acc_s = term_s;
            for m in j..(j + CS_SERIES_TERMS) {
                let mf = m as f64;
                let jf = j as f64;
                let ratio_c =
                    -u * (mf + 1.0) / ((mf + 1.0 - jf) * (2.0 * mf + 1.0) * (2.0 * mf + 2.0));
                let ratio_s =
                    -u * (mf + 1.0) / ((mf + 1.0 - jf) * (2.0 * mf + 2.0) * (2.0 * mf + 3.0));
                term_c *= ratio_c;
                term_s *= ratio_s;
                acc_c += term_c;
                acc_s += term_s;
            }
            c[j] = acc_c;
            s[j] = acc_s;
        }
        (c, s)
    } else {
        let (c0, s0) = if u > 0.0 {
            let r = u.sqrt();
            (r.cos(), r.sin() / r)
        } else {
            let r = (-u).sqrt();
            (r.cosh(), r.sinh() / r)
        };
        let mut c = [c0, 0.0, 0.0, 0.0, 0.0];
        let mut s = [s0, 0.0, 0.0, 0.0, 0.0];
        for j in 0..4 {
            s[j + 1] = (c[j] - (2.0 * j as f64 + 1.0) * s[j]) / (2.0 * u);
            c[j + 1] = -s[j] / 2.0;
        }
        (c, s)
    }
}

/// Derivative stack `[T, T′, T″, T‴, T⁗]` (in `w`) of
/// `T(w) = atan(√w)/√w | artanh(√−w)/√−w`. Defined for `w > −1`
/// (automatic in-chart: `w = κ‖·‖²` with `κ‖x‖² > −1` inside the ball).
pub fn t_stacks(w: f64) -> [f64; 5] {
    if w.abs() <= T_SERIES_W_MAX {
        let mut t = [0.0; 5];
        for (j, slot) in t.iter_mut().enumerate() {
            // a_m = (−1)^m m!/(m−j)! w^{m−j} / (2m+1).
            let mut term = 1.0;
            for f in 1..=j {
                let fj = f as f64;
                term *= -fj * (2.0 * fj - 1.0) / (2.0 * fj + 1.0);
            }
            let mut acc = term;
            for m in j..(j + T_SERIES_TERMS) {
                let mf = m as f64;
                let jf = j as f64;
                term *= -w * (mf + 1.0) * (2.0 * mf + 1.0) / ((mf + 1.0 - jf) * (2.0 * mf + 3.0));
                acc += term;
            }
            *slot = acc;
        }
        t
    } else {
        let t0 = if w > 0.0 {
            let r = w.sqrt();
            r.atan() / r
        } else {
            let r = (-w).sqrt();
            r.atanh() / r
        };
        let mut t = [t0, 0.0, 0.0, 0.0, 0.0];
        let mut r_j = 1.0 / (1.0 + w); // R⁽ʲ⁾ = (−1)^j j! / (1+w)^{j+1}
        for j in 0..4 {
            t[j + 1] = (r_j - (2.0 * j as f64 + 1.0) * t[j]) / (2.0 * w);
            r_j *= -((j + 1) as f64) / (1.0 + w);
        }
        t
    }
}

/// The unified constant-curvature manifold `M_κ` in the κ-stereographic
/// chart. See the module documentation for the model and conventions.
#[derive(Clone, Debug)]
pub struct ConstantCurvature {
    /// Sectional curvature κ — any real number; the three classical
    /// geometries are κ > 0, κ = 0, κ < 0 and this struct does not branch
    /// on which one it is.
    pub kappa: f64,
    /// Intrinsic (= chart = ambient) dimension.
    pub dim: usize,
}

impl ConstantCurvature {
    pub fn new(dim: usize, kappa: f64) -> Self {
        Self { kappa, dim }
    }

    fn check_len(&self, context: &'static str, got: usize) -> GeometryResult<()> {
        if got != self.dim {
            return Err(GeometryError::DimensionMismatch {
                context,
                expected: self.dim,
                got,
            });
        }
        Ok(())
    }

    /// `1 + κ‖x‖²` — the reciprocal half-conformal factor `2/λ_x`. Must be
    /// positive for the point to lie in the chart (automatic for κ ≥ 0,
    /// the open-ball constraint for κ < 0).
    fn chart_gauge(&self, x: ArrayView1<'_, f64>) -> GeometryResult<f64> {
        let gauge = 1.0 + self.kappa * x.dot(&x);
        if gauge <= GEOMETRY_EPS {
            return Err(GeometryError::InvalidPoint(
                "constant-curvature point outside the κ-stereographic chart",
            ));
        }
        Ok(gauge)
    }

    /// Conformal factor λ_x = 2 / (1 + κ‖x‖²).
    pub fn conformal_factor(&self, x: ArrayView1<'_, f64>) -> GeometryResult<f64> {
        Ok(2.0 / self.chart_gauge(x)?)
    }

    /// Radial Jacobian determinant `J_κ(r) = det(d exp_μ)|_{‖v‖=r}` of the
    /// exponential map in geodesic normal coordinates — the volume element that
    /// converts the flat tangent measure `dr` into the Riemannian volume
    /// `dvol_κ` at geodesic radius `r` from any base point (homogeneous, so it
    /// depends only on `r`, never on the base).
    ///
    /// In a space form of curvature κ the Jacobi-field solution gives
    /// `J_κ(r) = (sn_κ(r) / r)^{d−1}` with the curvature-normalized sine
    /// `sn_κ(r) = sin(√κ r)/√κ` (κ>0), `r` (κ=0), `sinh(√−κ r)/√−κ` (κ<0).
    /// Writing `S(u) = sn(t)/t` with `u = κ r²` (the entire function already in
    /// the chart's [`cs_stacks`]), this is exactly `S(κ r²)^{d−1}` — analytic
    /// through κ = 0 with no special case. `J_κ(0) = 1`.
    ///
    /// On a 1-D space form (`d = 1`) the exponent is 0, so `J_κ ≡ 1`: the exp
    /// map is a radial isometry and the volume Jacobian carries no curvature
    /// information (consistent with #944's reduced-information d = 1 power
    /// analysis — there κ is identified by the conformal-factor term alone).
    ///
    /// Past the κ>0 conjugate radius (`√κ·r > π`, the antipodal shell) `S(u)`
    /// turns negative. The geodesic ball is no longer embedded there, so the
    /// well-defined non-negative volume element is `max(S(u), 0)^{d−1}` — the
    /// clamp is on `S` itself, not on the (possibly even) power, so it stays a
    /// genuine `→ 0⁺` collapse at the shell for every `d` (an even `d−1` would
    /// otherwise resurrect a spurious positive volume past the cut). The
    /// `response_kappa_bounds` cap keeps the search strictly before this shell;
    /// the clamp only hardens stray CI/LR probes.
    ///
    /// This is the κ-dependent volume term whose log enters the #1104 honest
    /// change-of-variables criterion and breaks the radius/scale degeneracy of
    /// a dispersion-only curvature criterion.
    pub fn jacobian_radial(&self, r: f64) -> f64 {
        // The transverse exponent d − 1. At d ≤ 1 the exp map is a radial
        // isometry (`d = 1`) or degenerate (`d = 0`): there are no transverse
        // Jacobi directions, so J ≡ 1 with no exponentiation. Guarding `d ≤ 1`
        // (not just `d == 1`) keeps a stray `d = 0` probe from forming `powi(−1)`.
        if self.dim <= 1 {
            return 1.0;
        }
        let exponent = (self.dim - 1) as i32;
        let u = self.kappa * r * r;
        let s = cs_stacks(u).1[0]; // S(u) = sn_κ(r)/r ≥ 0 inside the chart
        // Clamp S (not the power) at the κ>0 conjugate shell so the volume
        // element collapses to 0⁺ there regardless of the parity of d−1.
        s.max(0.0).powi(exponent)
    }

    /// Möbius addition `x ⊕_κ y` — the chart realization of geodesic
    /// translation. Rational in κ (hence trivially κ-differentiable):
    ///
    /// ```text
    ///   x ⊕_κ y = [(1 − 2κ⟨x,y⟩ − κ‖y‖²)·x + (1 + κ‖x‖²)·y]
    ///             / [1 − 2κ⟨x,y⟩ + κ²‖x‖²‖y‖²]
    /// ```
    ///
    /// At κ = 0 this is `x + y`; at κ = −1 it is exactly the classical
    /// Poincaré-ball Möbius addition used by `poincare.rs`.
    pub fn mobius_add(
        &self,
        x: ArrayView1<'_, f64>,
        y: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let k = self.kappa;
        let xy = x.dot(&y);
        let xx = x.dot(&x);
        let yy = y.dot(&y);
        let denom = 1.0 - 2.0 * k * xy + k * k * xx * yy;
        if denom.abs() <= MOBIUS_DENOM_EPS {
            return Err(GeometryError::Singular(
                "Möbius addition at the κ>0 antipodal point",
            ));
        }
        let a = 1.0 - 2.0 * k * xy - k * yy;
        let b = 1.0 + k * xx;
        let mut out = Array1::zeros(x.len());
        for i in 0..x.len() {
            out[i] = (a * x[i] + b * y[i]) / denom;
        }
        Ok(out)
    }

    /// Geodesic distance `d_κ(x, y) = 2·‖w‖·T(κ‖w‖²)`, `w = (−x) ⊕_κ y`.
    pub fn distance(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> GeometryResult<f64> {
        self.check_len("constant-curvature distance x", x.len())?;
        self.check_len("constant-curvature distance y", y.len())?;
        self.chart_gauge(x)?;
        self.chart_gauge(y)?;
        let neg_x = x.mapv(|v| -v);
        let w = self.mobius_add(neg_x.view(), y)?;
        let nw2 = w.dot(&w);
        Ok(2.0 * nw2.sqrt() * t_stacks(self.kappa * nw2)[0])
    }

    /// `tn_κ(t) = sn(t)/cs(t) = t·S(κt²)/C(κt²)` — the generalized tangent.
    fn tn(&self, t: f64) -> GeometryResult<f64> {
        let (c, s) = cs_stacks(self.kappa * t * t);
        if c[0].abs() <= GEOMETRY_EPS {
            return Err(GeometryError::Singular(
                "constant-curvature exp map at a conjugate point (cos(√κ t) = 0)",
            ));
        }
        Ok(t * s[0] / c[0])
    }

    /// Gyration `gyr[a, b]v = ⊖(a ⊕ b) ⊕ (a ⊕ (b ⊕ v))` — the holonomy
    /// rotation of Möbius addition; the exact parallel-transport rotation
    /// between tangent spaces in this chart.
    fn gyration(
        &self,
        a: ArrayView1<'_, f64>,
        b: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let bv = self.mobius_add(b, v)?;
        let abv = self.mobius_add(a, bv.view())?;
        let ab = self.mobius_add(a, b)?;
        let neg_ab = ab.mapv(|z| -z);
        self.mobius_add(neg_ab.view(), abv.view())
    }
}

impl RiemannianManifold for ConstantCurvature {
    fn dim(&self) -> usize {
        self.dim
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        self.check_len("constant-curvature tangent_basis point", point.len())?;
        self.chart_gauge(point)?;
        Ok(Array2::eye(self.dim))
    }

    /// `exp_x(v) = x ⊕_κ [ tn_κ(λ_x‖v‖/2) · v̂ ]` — exact geodesic flow.
    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.check_len("constant-curvature exp point", point.len())?;
        self.check_len("constant-curvature exp tangent", tangent_vec.len())?;
        let gauge = self.chart_gauge(point)?;
        let n = tangent_vec.dot(&tangent_vec).sqrt();
        if n <= GEOMETRY_EPS {
            return Ok(point.to_owned());
        }
        let t = n / gauge; // λ_x‖v‖/2 = ‖v‖/(1 + κ‖x‖²)
        let scale = self.tn(t)? / n;
        let step = tangent_vec.mapv(|z| z * scale);
        self.mobius_add(point, step.view())
    }

    /// `log_x(y) = (1 + κ‖x‖²) · T(κ‖w‖²) · w`, `w = (−x) ⊕_κ y` —
    /// sqrt-free and smooth through both `w = 0` and `κ = 0`.
    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.check_len("constant-curvature log from", p_from.len())?;
        self.check_len("constant-curvature log to", p_to.len())?;
        let gauge = self.chart_gauge(p_from)?;
        self.chart_gauge(p_to)?;
        let neg_x = p_from.mapv(|v| -v);
        let w = self.mobius_add(neg_x.view(), p_to)?;
        let coeff = gauge * t_stacks(self.kappa * w.dot(&w))[0];
        Ok(w.mapv(|z| z * coeff))
    }

    /// Transport along the polyline rows of `point_along` by composed
    /// per-segment gyrations, rescaled by the conformal factors so the
    /// Riemannian norm is preserved exactly:
    /// `PT_{a→b}(v) = (λ_a/λ_b) · gyr[b, −a] v`.
    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.check_len("constant-curvature transport vector", vec.len())?;
        if point_along.nrows() < 2 {
            return Ok(vec.to_owned());
        }
        let mut carried = vec.to_owned();
        for seg in 0..(point_along.nrows() - 1) {
            let a = point_along.row(seg);
            let b = point_along.row(seg + 1);
            self.check_len("constant-curvature transport waypoint", a.len())?;
            let lam_ratio = self.chart_gauge(b)? / self.chart_gauge(a)?; // λ_a/λ_b
            let neg_a = a.mapv(|z| -z);
            carried = self
                .gyration(b, neg_a.view(), carried.view())?
                .mapv(|z| z * lam_ratio);
        }
        Ok(carried)
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        self.check_len("constant-curvature metric point", point.len())?;
        let lam = self.conformal_factor(point)?;
        Ok(Array2::eye(self.dim) * (lam * lam))
    }

    /// Conformal-metric Christoffels with `∂_i ln λ = −κ λ x_i`:
    /// `Γ^k_{ij} = δ_{ik} φ_j + δ_{jk} φ_i − δ_{ij} φ_k`.
    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        self.check_len("constant-curvature Christoffel point", point.len())?;
        let lam = self.conformal_factor(point)?;
        let phi: Vec<f64> = point.iter().map(|&xi| -self.kappa * lam * xi).collect();
        let d = self.dim;
        let mut out = Vec::with_capacity(d);
        for k in 0..d {
            let mut gamma_k = Array2::zeros((d, d));
            for i in 0..d {
                for j in 0..d {
                    let mut val = 0.0;
                    if i == k {
                        val += phi[j];
                    }
                    if j == k {
                        val += phi[i];
                    }
                    if i == j {
                        val -= phi[k];
                    }
                    gamma_k[[i, j]] = val;
                }
            }
            out.push(gamma_k);
        }
        Ok(out)
    }

    /// Constant by construction — the defining property of the family.
    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        self.check_len("constant-curvature sectional point", point.len())?;
        self.check_len("constant-curvature sectional u", tangent_pair.0.len())?;
        self.check_len("constant-curvature sectional v", tangent_pair.1.len())?;
        self.chart_gauge(point)?;
        Ok(self.kappa)
    }

    /// Euclidean reverse-mode VJP of `y = exp_x(v)` w.r.t. BOTH `x` and `v`,
    /// returned as `(x̄, v̄)` for an incoming cotangent `ḡ = grad_output`.
    ///
    /// A curved manifold MUST NOT inherit the trait's flat identity VJP — that
    /// default is the exact Jacobian only when `exp_p(v) = p + v`, and using it
    /// on a curved member would silently return wrong reverse-mode gradients
    /// (the exact objective↔gradient-desync the trait doc warns against).
    ///
    /// This is the exact reverse-mode of the explicit `exp_map` formula
    /// `exp_x(v) = x ⊕_κ [ tn_κ(λ_x‖v‖/2) · v̂ ]` (with `mobius_add` inlined),
    /// differentiated line-for-line, so it is correct-by-construction for both
    /// arguments and matches the forward value exactly:
    ///
    /// ```text
    ///   gauge = 1 + κ‖x‖²,  n = ‖v‖,  t = n/gauge,  τ = tn_κ(t),
    ///   step = (τ/n)·v,  y = x ⊕_κ step  (Möbius),  tn′(t) = 1 + κτ².
    /// ```
    ///
    /// The reverse walks back through Möbius addition, `step = scale·v`,
    /// `scale = τ/n`, `τ = tn_κ(t)`, `t = n/gauge`, `gauge = 1 + κ‖x‖²` and
    /// `n = ‖v‖`. At `κ = 0` (doubled-gauge flat space, `exp_p(v) = p + v`)
    /// both Jacobians are the identity and the reverse reduces to `(ḡ, ḡ)`,
    /// matched by the early return. At `v = 0` the differential of `exp_x` is
    /// the identity in both slots, so we return `(ḡ, ḡ)` there too (mirroring
    /// the `n ≤ GEOMETRY_EPS` short-circuit in `exp_map`). The conjugate-point
    /// guard (`tn` errors at `cos(√κ t)=0`) and the Möbius antipodal-denominator
    /// guard propagate exactly as in the forward map.
    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        self.check_len("constant-curvature exp_map_vjp point", point.len())?;
        self.check_len("constant-curvature exp_map_vjp tangent", tangent_vec.len())?;
        self.check_len(
            "constant-curvature exp_map_vjp grad_output",
            grad_output.len(),
        )?;
        let k = self.kappa;
        if k.abs() <= GEOMETRY_EPS {
            // Doubled-gauge flat space: exp is the chart translation x + v, so
            // both Jacobians are the identity and the VJP is the cotangent itself.
            return Ok((grad_output.to_owned(), grad_output.to_owned()));
        }
        let d = point.len();
        let n = tangent_vec.dot(&tangent_vec).sqrt();
        if n <= GEOMETRY_EPS {
            // At v = 0 the differential of exp_x is the identity in both slots.
            return Ok((grad_output.to_owned(), grad_output.to_owned()));
        }

        // ── Forward (mirrors `exp_map` with `mobius_add` inlined). ──────────
        let gauge = self.chart_gauge(point)?; // 1 + κ‖x‖²
        let t = n / gauge; // λ_x‖v‖/2
        let tau = self.tn(t)?; // generalized tangent (errors at conjugate point)
        let scale = tau / n;
        let step = tangent_vec.mapv(|z| z * scale);
        let p = point.dot(&step); // ⟨x, step⟩
        let xx = point.dot(&point); // ‖x‖²
        let ss = step.dot(&step); // ‖step‖²
        let a = 1.0 - 2.0 * k * p - k * ss;
        let b = 1.0 + k * xx; // = gauge
        let denom = 1.0 - 2.0 * k * p + k * k * xx * ss;
        if denom.abs() <= MOBIUS_DENOM_EPS {
            return Err(GeometryError::Singular(
                "Möbius addition at the κ>0 antipodal point",
            ));
        }
        let mut y = Array1::zeros(d);
        for i in 0..d {
            y[i] = (a * point[i] + b * step[i]) / denom;
        }

        // ── Reverse. ────────────────────────────────────────────────────────
        let g = grad_output;
        let yx = g.dot(&point); // ḡ·x
        let ys = g.dot(&step); // ḡ·step
        let yy = g.dot(&y); // ḡ·y
        let inv_d = 1.0 / denom;

        // Möbius VJP into x (holding step) and into step (holding x).
        let mut x_bar = Array1::zeros(d);
        let mut step_bar = Array1::zeros(d);
        for j in 0..d {
            x_bar[j] = (-2.0 * k * step[j] * yx + a * g[j] + 2.0 * k * point[j] * ys) * inv_d
                - yy * (-2.0 * k * step[j] + 2.0 * k * k * point[j] * ss) * inv_d;
            step_bar[j] = ((-2.0 * k * point[j] - 2.0 * k * step[j]) * yx + b * g[j]) * inv_d
                - yy * (-2.0 * k * point[j] + 2.0 * k * k * xx * step[j]) * inv_d;
        }

        // step = scale·v  ⇒  scale_bar = step̄·v,  v̄ += scale·step̄.
        let scale_bar = step_bar.dot(&tangent_vec);
        let mut v_bar = step_bar.mapv(|z| z * scale);

        // scale = τ/n  ⇒  τ_bar = scale_bar/n,  n_bar = −scale_bar·τ/n².
        let tau_bar = scale_bar / n;
        let mut n_bar = -scale_bar * tau / (n * n);

        // τ = tn_κ(t),  tn′(t) = 1 + κτ²  ⇒  t_bar = τ_bar·(1 + κτ²).
        let t_bar = tau_bar * (1.0 + k * tau * tau);

        // t = n/gauge  ⇒  n_bar += t_bar/gauge,  gauge_bar = −t_bar·n/gauge².
        n_bar += t_bar / gauge;
        let gauge_bar = -t_bar * n / (gauge * gauge);

        // gauge = 1 + κ‖x‖²  ⇒  x̄ += gauge_bar·2κ·x.
        for j in 0..d {
            x_bar[j] += gauge_bar * 2.0 * k * point[j];
        }

        // n = ‖v‖  ⇒  v̄ += n_bar·v/n.
        for j in 0..d {
            v_bar[j] += n_bar * tangent_vec[j] / n;
        }

        Ok((x_bar, v_bar))
    }
}

// ── κ-jets: stage 2 of #944, powered by the #932 tower ───────────────
//
// Each function below is the SAME geometric program as its f64 twin
// above, evaluated with κ seeded as `Tower4::<1>::variable(κ, 0)` and the
// scalar primitives entering through their certified derivative stacks.
// Points are chart constants; everything κ-dependent is a tower. The
// returned channels are exact ∂/∂κ and ∂²/∂κ² — the design/penalty
// movement the outer ψ-channel consumes.

type KJet = Tower4<1>;

fn kjet_mobius_w(
    kappa: KJet,
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) -> GeometryResult<Vec<KJet>> {
    // w = (−x) ⊕_κ y with constant points: ⟨−x,y⟩, ‖x‖², ‖y‖² are plain
    // scalars; the κ-dependence is entirely through the rational
    // coefficients, mirroring `mobius_add` line for line.
    let xy = -x.dot(&y);
    let xx = x.dot(&x);
    let yy = y.dot(&y);
    let a = -(kappa * (2.0 * xy + yy)) + 1.0;
    let b = kappa * xx + 1.0;
    let denom = (kappa * kappa) * (xx * yy) - kappa * (2.0 * xy) + 1.0;
    if denom.v.abs() <= MOBIUS_DENOM_EPS {
        return Err(GeometryError::Singular(
            "Möbius addition at the κ>0 antipodal point",
        ));
    }
    let inv = denom.recip();
    Ok((0..x.len())
        .map(|i| (a * (-x[i]) + b * y[i]) * inv)
        .collect())
}

/// `(d, ∂d/∂κ, ∂²d/∂κ²)` of the geodesic distance — exact, one pass.
pub fn distance_kappa_jet(
    manifold: &ConstantCurvature,
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) -> GeometryResult<(f64, f64, f64)> {
    manifold.check_len("constant-curvature distance-jet x", x.len())?;
    manifold.check_len("constant-curvature distance-jet y", y.len())?;
    manifold.chart_gauge(x)?;
    manifold.chart_gauge(y)?;
    let kappa = KJet::variable(manifold.kappa, 0);
    let w = kjet_mobius_w(kappa, x, y)?;
    let mut nw2 = KJet::constant(0.0);
    for wi in &w {
        nw2 = nw2 + *wi * *wi;
    }
    if nw2.v <= GEOMETRY_EPS * GEOMETRY_EPS {
        // Coincident points: d ≡ 0 along the whole κ-path.
        return Ok((0.0, 0.0, 0.0));
    }
    let arg = kappa * nw2;
    let t = arg.compose_unary(t_stacks(arg.v));
    let d = nw2.sqrt() * t * 2.0;
    Ok((d.v, d.g[0], d.h[0][0]))
}

/// `(log, ∂log/∂κ, ∂²log/∂κ²)` of the log map, componentwise — the
/// κ-movement of geodesic (normal) coordinates, which is what κ-dependent
/// bases consume. Sqrt-free, smooth through `w = 0` and `κ = 0`.
pub fn log_map_kappa_jet(
    manifold: &ConstantCurvature,
    p_from: ArrayView1<'_, f64>,
    p_to: ArrayView1<'_, f64>,
) -> GeometryResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    manifold.check_len("constant-curvature log-jet from", p_from.len())?;
    manifold.check_len("constant-curvature log-jet to", p_to.len())?;
    manifold.chart_gauge(p_from)?;
    manifold.chart_gauge(p_to)?;
    let kappa = KJet::variable(manifold.kappa, 0);
    let w = kjet_mobius_w(kappa, p_from, p_to)?;
    let mut nw2 = KJet::constant(0.0);
    for wi in &w {
        nw2 = nw2 + *wi * *wi;
    }
    let arg = kappa * nw2;
    let t = arg.compose_unary(t_stacks(arg.v));
    let gauge = kappa * p_from.dot(&p_from) + 1.0;
    let coeff = gauge * t;
    let d = p_from.len();
    let mut value = Array1::zeros(d);
    let mut dk = Array1::zeros(d);
    let mut dkk = Array1::zeros(d);
    for i in 0..d {
        let li = coeff * w[i];
        value[i] = li.v;
        dk[i] = li.g[0];
        dkk[i] = li.h[0][0];
    }
    Ok((value, dk, dkk))
}

/// `(exp, ∂exp/∂κ, ∂²exp/∂κ²)` of the geodesic exp map, componentwise — the
/// κ-movement of the exponential chart, completing the κ-jet trio. Same
/// program as [`ConstantCurvature::exp_map`] (`x ⊕_κ [tn_κ(λ_x‖v‖/2)·v̂]`)
/// evaluated over `Tower4<1>` with κ seeded: the gauge `1+κ‖x‖²`, the
/// generalized tangent `tn_κ(t) = t·S(κt²)/C(κt²)`, and the Möbius addition
/// all become towers, with `C`/`S` entering through their certified stacks via
/// `compose_unary`. Value and κ-derivatives are one expression, so they cannot
/// desync. The conjugate-point guard (`C(κt²)=0`) and the antipodal-denominator
/// guard are checked on the value channel.
pub fn exp_map_kappa_jet(
    manifold: &ConstantCurvature,
    point: ArrayView1<'_, f64>,
    tangent_vec: ArrayView1<'_, f64>,
) -> GeometryResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    manifold.check_len("constant-curvature exp-jet point", point.len())?;
    manifold.check_len("constant-curvature exp-jet tangent", tangent_vec.len())?;
    manifold.chart_gauge(point)?;
    let d = point.len();
    let n = tangent_vec.dot(&tangent_vec).sqrt();
    if n <= GEOMETRY_EPS {
        // Zero tangent ⇒ exp_x(0) = x for every κ along the path.
        return Ok((point.to_owned(), Array1::zeros(d), Array1::zeros(d)));
    }
    let kappa = KJet::variable(manifold.kappa, 0);
    let xx = point.dot(&point);
    // gauge = 1 + κ‖x‖²  (tower);  t = ‖v‖ / gauge = λ_x‖v‖/2.
    let gauge = kappa * xx + 1.0;
    let t = gauge.recip() * n;
    // tn_κ(t) = t·S(κt²)/C(κt²), the primitives composed at the tower arg κt².
    let arg = kappa * (t * t);
    let (cstk, sstk) = cs_stacks(arg.v);
    let c = arg.compose_unary(cstk);
    if c.v.abs() <= GEOMETRY_EPS {
        return Err(GeometryError::Singular(
            "constant-curvature exp-jet at a conjugate point (cos(√κ t) = 0)",
        ));
    }
    let s = arg.compose_unary(sstk);
    let tn = t * s * c.recip();
    // step = (tn / ‖v‖) · v   (tower vector; v is a chart constant).
    let scale = tn * (1.0 / n);
    let step: Vec<KJet> = (0..d).map(|i| scale * tangent_vec[i]).collect();
    // out = x ⊕_κ step, mirroring `mobius_add` with x constant and step a tower.
    let mut xs = KJet::constant(0.0); // ⟨x, step⟩
    let mut ss = KJet::constant(0.0); // ‖step‖²
    for i in 0..d {
        xs = xs + step[i] * point[i];
        ss = ss + step[i] * step[i];
    }
    let two_k_xs = (kappa * 2.0) * xs; // 2κ⟨x,step⟩
    // denom = 1 − 2κ⟨x,step⟩ + κ²‖x‖²‖step‖²  (no Sub on the tower; Neg+Add).
    let denom = -two_k_xs + (kappa * kappa) * (ss * xx) + 1.0;
    if denom.v.abs() <= MOBIUS_DENOM_EPS {
        return Err(GeometryError::Singular(
            "Möbius addition at the κ>0 antipodal point",
        ));
    }
    // a = 1 − 2κ⟨x,step⟩ − κ‖step‖²;  b = 1 + κ‖x‖² = gauge.
    let a = -two_k_xs + (-(kappa * ss)) + 1.0;
    let b = gauge;
    let inv = denom.recip();
    let mut value = Array1::zeros(d);
    let mut dk = Array1::zeros(d);
    let mut dkk = Array1::zeros(d);
    for i in 0..d {
        let oi = (a * point[i] + b * step[i]) * inv;
        value[i] = oi.v;
        dk[i] = oi.g[0];
        dkk[i] = oi.h[0][0];
    }
    Ok((value, dk, dkk))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The scalar stacks are exact across the series/closed-form branch
    /// boundary: each derivative slot must equal the central finite
    /// difference of the slot below it, on both sides of the switch and
    /// straddling u = 0.
    #[test]
    fn scalar_stacks_are_fd_consistent_across_branches() {
        let h = 1e-6;
        for &u in &[
            -3.0, -0.6, -0.49, -0.2, -1e-9, 0.0, 1e-9, 0.2, 0.49, 0.6, 3.0,
        ] {
            let up = cs_stacks(u + h);
            let dn = cs_stacks(u - h);
            let at = cs_stacks(u);
            for j in 0..4 {
                let fd_c = (up.0[j] - dn.0[j]) / (2.0 * h);
                let fd_s = (up.1[j] - dn.1[j]) / (2.0 * h);
                assert!(
                    (at.0[j + 1] - fd_c).abs() <= 1e-7 * fd_c.abs().max(1.0),
                    "C stack order {j} at u={u}: analytic {} fd {}",
                    at.0[j + 1],
                    fd_c
                );
                assert!(
                    (at.1[j + 1] - fd_s).abs() <= 1e-7 * fd_s.abs().max(1.0),
                    "S stack order {j} at u={u}: analytic {} fd {}",
                    at.1[j + 1],
                    fd_s
                );
            }
        }
        for &w in &[-0.6, -0.26, -0.1, -1e-9, 0.0, 1e-9, 0.1, 0.26, 0.8, 4.0] {
            let up = t_stacks(w + 1e-7);
            let dn = t_stacks(w - 1e-7);
            let at = t_stacks(w);
            for j in 0..4 {
                let fd = (up[j] - dn[j]) / 2e-7;
                assert!(
                    (at[j + 1] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                    "T stack order {j} at w={w}: analytic {} fd {}",
                    at[j + 1],
                    fd
                );
            }
        }
    }

    /// Closed-form pins at the three classical members. κ = −1 reproduces
    /// the Poincaré radial isometry d(0, y) = 2·artanh‖y‖ (the convention
    /// pinned on poincare.rs); κ = +1 the stereographic unit sphere
    /// d(0, y) = 2·atan‖y‖; κ = 0 flat space in the doubled gauge,
    /// d = 2‖x − y‖ with exp/log the plain chart translation.
    #[test]
    fn classical_members_match_closed_forms() {
        let y: ndarray::Array1<f64> = array![0.3, -0.2, 0.1];
        let origin: ndarray::Array1<f64> = array![0.0, 0.0, 0.0];
        let r: f64 = y.dot(&y).sqrt();

        let hyper = ConstantCurvature::new(3, -1.0);
        let d = hyper.distance(origin.view(), y.view()).expect("hyper d");
        assert!((d - 2.0 * r.atanh()).abs() <= 1e-14, "poincare radial: {d}");

        let sphere = ConstantCurvature::new(3, 1.0);
        let d = sphere.distance(origin.view(), y.view()).expect("sphere d");
        assert!((d - 2.0 * r.atan()).abs() <= 1e-14, "sphere radial: {d}");

        let flat = ConstantCurvature::new(3, 0.0);
        let x = array![0.4, 0.1, -0.7];
        let d = flat.distance(x.view(), y.view()).expect("flat d");
        let diff = (&y - &x).dot(&(&y - &x)).sqrt();
        assert!((d - 2.0 * diff).abs() <= 1e-14, "flat doubled gauge: {d}");
        let v = array![0.2, -0.5, 0.3];
        let e = flat.exp_map(x.view(), v.view()).expect("flat exp");
        for i in 0..3 {
            assert!(
                (e[i] - (x[i] + v[i])).abs() <= 1e-14,
                "flat exp is translation"
            );
        }
        let l = flat.log_map(x.view(), y.view()).expect("flat log");
        for i in 0..3 {
            assert!(
                (l[i] - (y[i] - x[i])).abs() <= 1e-14,
                "flat log is difference"
            );
        }
    }

    /// Geodesic self-consistency at non-classical κ, off the origin:
    /// d(x, exp_x(v)) equals the Riemannian tangent norm λ_x‖v‖, and
    /// log_x inverts exp_x. This is the chart-free content of the family.
    #[test]
    fn exp_log_distance_are_mutually_consistent_across_kappa() {
        let x = array![0.25, -0.1];
        let v = array![0.15, 0.2];
        for &kappa in &[-1.7, -0.6, -1e-7, 0.0, 1e-7, 0.8, 2.3] {
            let m = ConstantCurvature::new(2, kappa);
            let lam = m.conformal_factor(x.view()).expect("lambda");
            let y = m.exp_map(x.view(), v.view()).expect("exp");
            let d = m.distance(x.view(), y.view()).expect("dist");
            let want = lam * v.dot(&v).sqrt();
            assert!(
                (d - want).abs() <= 1e-12 * want.max(1.0),
                "κ={kappa}: d(x, exp_x v) = {d}, λ_x‖v‖ = {want}"
            );
            let back = m.log_map(x.view(), y.view()).expect("log");
            for i in 0..2 {
                assert!(
                    (back[i] - v[i]).abs() <= 1e-11,
                    "κ={kappa}: log∘exp ≠ id at [{i}]: {} vs {}",
                    back[i],
                    v[i]
                );
            }
        }
    }

    /// Parallel transport is a linear isometry: the Riemannian norm
    /// λ‖·‖ is preserved along the polyline, at every sign of κ.
    #[test]
    fn parallel_transport_preserves_riemannian_norm() {
        let path = ndarray::arr2(&[[0.05, 0.1], [0.2, -0.15], [-0.1, 0.25]]);
        let v = array![0.3, -0.4];
        for &kappa in &[-1.2, 0.0, 1.4] {
            let m = ConstantCurvature::new(2, kappa);
            let out = m.parallel_transport(path.view(), v.view()).expect("pt");
            let lam_a = m.conformal_factor(path.row(0)).expect("λ_a");
            let lam_b = m.conformal_factor(path.row(2)).expect("λ_b");
            let n_in = lam_a * v.dot(&v).sqrt();
            let n_out = lam_b * out.dot(&out).sqrt();
            assert!(
                (n_in - n_out).abs() <= 1e-11 * n_in.max(1.0),
                "κ={kappa}: transport norm {n_out} vs {n_in}"
            );
        }
    }

    /// The κ-jets are exact: central FD of distance and log in κ matches
    /// the tower channels, including straddling κ = 0 and the series
    /// branch boundary.
    #[test]
    fn kappa_jets_match_finite_differences() {
        let x = array![0.3, -0.15];
        let y = array![-0.2, 0.25];
        let h = 1e-5;
        for &kappa in &[-1.3, -0.5, -1e-6, 0.0, 1e-6, 0.4, 1.6] {
            let m = ConstantCurvature::new(2, kappa);
            let up = ConstantCurvature::new(2, kappa + h);
            let dn = ConstantCurvature::new(2, kappa - h);
            let (d, d_k, d_kk) = distance_kappa_jet(&m, x.view(), y.view()).expect("jet");
            let d_up = up.distance(x.view(), y.view()).expect("d+");
            let d_dn = dn.distance(x.view(), y.view()).expect("d-");
            let d_at = m.distance(x.view(), y.view()).expect("d0");
            assert!(
                (d - d_at).abs() <= 1e-13 * d_at.max(1.0),
                "jet value channel"
            );
            let fd1 = (d_up - d_dn) / (2.0 * h);
            let fd2 = (d_up - 2.0 * d_at + d_dn) / (h * h);
            assert!(
                (d_k - fd1).abs() <= 1e-6 * fd1.abs().max(1.0),
                "κ={kappa}: ∂d/∂κ analytic {d_k} fd {fd1}"
            );
            assert!(
                (d_kk - fd2).abs() <= 1e-4 * fd2.abs().max(1.0),
                "κ={kappa}: ∂²d/∂κ² analytic {d_kk} fd {fd2}"
            );

            let (l, l_k, l_kk) = log_map_kappa_jet(&m, x.view(), y.view()).expect("ljet");
            let l_up = up.log_map(x.view(), y.view()).expect("l+");
            let l_dn = dn.log_map(x.view(), y.view()).expect("l-");
            let l_at = m.log_map(x.view(), y.view()).expect("l0");
            for i in 0..2 {
                assert!((l[i] - l_at[i]).abs() <= 1e-13 * l_at[i].abs().max(1.0));
                let fd1 = (l_up[i] - l_dn[i]) / (2.0 * h);
                let fd2 = (l_up[i] - 2.0 * l_at[i] + l_dn[i]) / (h * h);
                assert!(
                    (l_k[i] - fd1).abs() <= 1e-6 * fd1.abs().max(1.0),
                    "κ={kappa}: ∂log/∂κ[{i}] analytic {} fd {fd1}",
                    l_k[i]
                );
                assert!(
                    (l_kk[i] - fd2).abs() <= 1e-4 * fd2.abs().max(1.0),
                    "κ={kappa}: ∂²log/∂κ²[{i}] analytic {} fd {fd2}",
                    l_kk[i]
                );
            }

            // exp-map κ-jet against central FD of exp_map at κ±h. Tangent kept
            // small so the geodesic stays well inside the chart for every κ.
            let v = array![0.12, -0.08];
            let (e, e_k, e_kk) = exp_map_kappa_jet(&m, x.view(), v.view()).expect("ejet");
            let e_up = up.exp_map(x.view(), v.view()).expect("e+");
            let e_dn = dn.exp_map(x.view(), v.view()).expect("e-");
            let e_at = m.exp_map(x.view(), v.view()).expect("e0");
            for i in 0..2 {
                assert!(
                    (e[i] - e_at[i]).abs() <= 1e-13 * e_at[i].abs().max(1.0),
                    "κ={kappa}: exp-jet value channel[{i}] {} vs {}",
                    e[i],
                    e_at[i]
                );
                let fd1 = (e_up[i] - e_dn[i]) / (2.0 * h);
                let fd2 = (e_up[i] - 2.0 * e_at[i] + e_dn[i]) / (h * h);
                assert!(
                    (e_k[i] - fd1).abs() <= 1e-6 * fd1.abs().max(1.0),
                    "κ={kappa}: ∂exp/∂κ[{i}] analytic {} fd {fd1}",
                    e_k[i]
                );
                assert!(
                    (e_kk[i] - fd2).abs() <= 1e-4 * fd2.abs().max(1.0),
                    "κ={kappa}: ∂²exp/∂κ²[{i}] analytic {} fd {fd2}",
                    e_kk[i]
                );
            }
        }
    }

    /// The analytic Euclidean reverse-mode VJP of `exp_map` (w.r.t. BOTH the
    /// base point and the tangent) matches central finite differences of the
    /// forward map, at every sign of κ and across the series branch. For each
    /// coordinate j: `x̄_fd[j] = ḡ·(exp(x+h e_j,v) − exp(x−h e_j,v))/(2h)` and
    /// likewise `v̄_fd[j]`. The tangent is kept small enough that the geodesic
    /// stays before the conjugate point (`‖v‖·conformal_factor(x)/2 < π/√κ`)
    /// for κ > 0, well inside the chart for κ < 0.
    #[test]
    fn exp_map_vjp_matches_finite_differences() {
        let h = 1e-6;
        let cases: &[(
            ndarray::Array1<f64>,
            ndarray::Array1<f64>,
            ndarray::Array1<f64>,
        )] = &[
            (array![0.2, -0.1], array![0.12, 0.08], array![1.0, -0.5]),
            (array![-0.15, 0.22], array![-0.05, 0.11], array![0.3, 0.7]),
        ];
        for &kappa in &[-1.3, -0.3, 0.0, 0.4, 1.1] {
            let m = ConstantCurvature::new(2, kappa);
            for (x, v, g) in cases {
                let d = x.len();
                let (x_bar, v_bar) = m
                    .exp_map_vjp(x.view(), v.view(), g.view())
                    .expect("exp_map_vjp");
                for j in 0..d {
                    // x̄_fd[j] = ḡ · ∂exp/∂x_j.
                    let mut xp = x.clone();
                    xp[j] += h;
                    let mut xn = x.clone();
                    xn[j] -= h;
                    let ep = m.exp_map(xp.view(), v.view()).expect("exp x+");
                    let en = m.exp_map(xn.view(), v.view()).expect("exp x-");
                    let xbar_fd = g.dot(&(&ep - &en)) / (2.0 * h);
                    assert!(
                        (x_bar[j] - xbar_fd).abs() <= 1e-5 * x_bar[j].abs().max(1.0),
                        "κ={kappa}: x̄[{j}] analytic {} fd {xbar_fd}",
                        x_bar[j]
                    );

                    // v̄_fd[j] = ḡ · ∂exp/∂v_j.
                    let mut vp = v.clone();
                    vp[j] += h;
                    let mut vn = v.clone();
                    vn[j] -= h;
                    let ep = m.exp_map(x.view(), vp.view()).expect("exp v+");
                    let en = m.exp_map(x.view(), vn.view()).expect("exp v-");
                    let vbar_fd = g.dot(&(&ep - &en)) / (2.0 * h);
                    assert!(
                        (v_bar[j] - vbar_fd).abs() <= 1e-5 * v_bar[j].abs().max(1.0),
                        "κ={kappa}: v̄[{j}] analytic {} fd {vbar_fd}",
                        v_bar[j]
                    );
                }
            }
        }
    }

    /// The radial volume Jacobian `J_κ(s) = (sn_κ(s)/s)^{d−1}` is analytic and
    /// continuous through the flat point κ = 0 and through the sign change, has
    /// the removable value `J_κ(0) = 1` at s = 0 for every κ, reduces to the
    /// flat `s^{d−1}/s^{d−1} = 1` (i.e. `S(0)=1`) limit, collapses to `0⁺` at the
    /// κ>0 conjugate shell (`√κ·s = π`) regardless of the parity of `d−1`, and is
    /// the radial isometry `J ≡ 1` at `d ≤ 1` with no exponentiation.
    #[test]
    fn jacobian_radial_is_stable_through_flat_and_at_d_le_1() {
        // d = 1: radial isometry, identically 1 at every κ and radius.
        for &kappa in &[-2.0, -1e-9, 0.0, 1e-9, 3.0] {
            let m = ConstantCurvature::new(1, kappa);
            for &s in &[0.0, 0.1, 1.0, 5.0] {
                assert_eq!(m.jacobian_radial(s), 1.0, "d=1 J≡1 at κ={kappa}, s={s}");
            }
        }
        // d = 0 (degenerate): the d ≤ 1 guard must NOT form powi(−1).
        let m0 = ConstantCurvature::new(0, 0.7);
        assert_eq!(m0.jacobian_radial(0.4), 1.0, "d=0 guarded to 1");

        // d = 3: continuity through κ = 0 and J(0) = 1; the flat limit is the
        // analytic value of S(κs²)² → 1 as κ → 0.
        let s = 0.3_f64;
        let flat = ConstantCurvature::new(3, 0.0).jacobian_radial(s);
        assert!((flat - 1.0).abs() <= 1e-15, "flat J(0.3) = {flat}");
        let near_plus = ConstantCurvature::new(3, 1e-8).jacobian_radial(s);
        let near_minus = ConstantCurvature::new(3, -1e-8).jacobian_radial(s);
        assert!(
            (near_plus - 1.0).abs() <= 1e-6 && (near_minus - 1.0).abs() <= 1e-6,
            "J continuous through κ=0: {near_plus}, {near_minus}"
        );
        // J(0) = 1 at every κ (removable s = 0 singularity of sn_κ/s).
        for &kappa in &[-1.5, 0.0, 2.5] {
            let m = ConstantCurvature::new(3, kappa);
            assert!((m.jacobian_radial(0.0) - 1.0).abs() <= 1e-15);
        }

        // κ > 0 conjugate shell at √κ·s = π collapses to 0⁺ for both even and
        // odd d−1 (the clamp is on S, not on the power).
        let kappa = 1.0_f64;
        let s_shell = std::f64::consts::PI / kappa.sqrt();
        for &d in &[3usize, 4] {
            let m = ConstantCurvature::new(d, kappa);
            // Just past the shell S(u) < 0; the volume element must be exactly 0,
            // never a spurious positive value resurrected by an even d−1 power.
            let past = m.jacobian_radial(s_shell * 1.01);
            assert_eq!(past, 0.0, "d={d}: J past conjugate shell must be 0");
        }
    }

    /// Sectional curvature is κ — the family's defining identity, exposed
    /// through the trait so curvature-consuming code needs no special case.
    #[test]
    fn sectional_curvature_is_kappa() {
        let m = ConstantCurvature::new(3, -0.37);
        let p = array![0.1, 0.0, -0.2];
        let u = array![1.0, 0.0, 0.0];
        let v = array![0.0, 1.0, 0.0];
        let k = m
            .sectional_curvature(p.view(), (u.view(), v.view()))
            .expect("sectional");
        assert!((k + 0.37).abs() <= 1e-15);
    }

    /// The closed-form conformal Christoffels (`∂_i ln λ = −κλx_i`) must equal
    /// the Levi-Civita symbols rebuilt from a finite difference of
    /// `metric_tensor` alone:
    /// `Γ^k_{ij} = ½ Σ_l g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})`.
    /// This pins the analytic `christoffel_symbols` against the metric it is
    /// supposed to be the connection of — independent of the `∂ln λ` algebra —
    /// at every sign of κ.
    #[test]
    fn christoffel_matches_fd_of_metric() {
        let d = 2usize;
        let x = array![0.22, -0.13];
        let h = 1e-5;
        for &kappa in &[-1.4, -0.5, 0.0, 0.7, 1.9] {
            let m = ConstantCurvature::new(d, kappa);
            // Inverse of the conformal metric g = λ²δ is g^{-1} = λ^{-2}δ.
            let lam = m.conformal_factor(x.view()).expect("λ");
            let g_inv_diag = 1.0 / (lam * lam);
            // ∂_a g_{ij} via central FD of metric_tensor (dg[a][[i,j]]).
            let mut dg: Vec<Array2<f64>> = Vec::with_capacity(d);
            for a in 0..d {
                let mut xp = x.clone();
                xp[a] += h;
                let mut xn = x.clone();
                xn[a] -= h;
                let gp = m.metric_tensor(xp.view()).expect("g+");
                let gn = m.metric_tensor(xn.view()).expect("g-");
                dg.push((&gp - &gn).mapv(|v| v / (2.0 * h)));
            }
            let gamma = m.christoffel_symbols(x.view()).expect("Γ");
            for k in 0..d {
                for i in 0..d {
                    for j in 0..d {
                        // g^{kl} is diagonal, so only l = k contributes.
                        let expected =
                            0.5 * g_inv_diag * (dg[i][[j, k]] + dg[j][[i, k]] - dg[k][[i, j]]);
                        assert!(
                            (gamma[k][[i, j]] - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                            "κ={kappa}: Γ^{k}_{{{i}{j}}} analytic {} vs FD-metric {expected}",
                            gamma[k][[i, j]]
                        );
                    }
                }
            }
        }
    }
}
