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
//! CLIENT of [`gam_math::jet_tower::Tower2`]: the same geometric
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

use crate::manifold::{GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold};
use gam_math::jet_tower::Tower2;

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
/// `CS_SERIES_U_MAX`, closed forms + the mutual recurrence outside.
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

/// Order-≤2 slice `[T, T′, T″]` of [`t_stacks`] — the *exact* prefix the
/// second-order κ-jets consume.
///
/// The κ-jets ride [`Tower2`], whose `compose_unary` reads only `d[0..=2]`; the
/// `T‴`/`T⁗` slots the full [`t_stacks`] builds are pure waste on that path (in
/// the series branch each is its own 48-term sum). Each slot `j` is an
/// independent series, and the closed-form recurrence advances one slot at a
/// time, so the first three entries are produced by the *identical* arithmetic
/// as [`t_stacks`] — a strict, bit-for-bit prune of the discarded high orders.
pub(crate) fn t_stacks3(w: f64) -> [f64; 3] {
    if w.abs() <= T_SERIES_W_MAX {
        let mut t = [0.0; 3];
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
                // Bit-identical early stop: on |w| ≤ T_SERIES_W_MAX the term ratio
                // has magnitude < 1, so |term| decreases monotonically; once a term
                // is too small to move `acc`, every successor is too, so summing the
                // remaining T_SERIES_TERMS terms reproduces `acc` exactly.
                let next = acc + term;
                if next == acc {
                    break;
                }
                acc = next;
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
        let mut t = [t0, 0.0, 0.0];
        let mut r_j = 1.0 / (1.0 + w);
        for j in 0..2 {
            t[j + 1] = (r_j - (2.0 * j as f64 + 1.0) * t[j]) / (2.0 * w);
            r_j *= -((j + 1) as f64) / (1.0 + w);
        }
        t
    }
}

/// Order-≤2 slices `([C,C′,C″], [S,S′,S″])` of [`cs_stacks`] — the *exact*
/// prefix the second-order κ-jets consume.
///
/// As with [`t_stacks3`], the [`Tower2`] `compose_unary` reads only `d[0..=2]`,
/// so the `C‴/C⁗`, `S‴/S⁗` slots [`cs_stacks`] builds (each a fresh 18-term
/// series in the small-`u` branch) are never read on the jet path. Both `C` and
/// `S` value-channels are still computed (so the closed-form `___sincos`
/// pairing — and hence the value channel — is bit-for-bit identical), and the
/// per-slot series / one-step recurrence make the first three orders identical
/// to [`cs_stacks`]; only the two discarded high orders are dropped.
pub(crate) fn cs_stacks3(u: f64) -> ([f64; 3], [f64; 3]) {
    if u.abs() <= CS_SERIES_U_MAX {
        let mut c = [0.0; 3];
        let mut s = [0.0; 3];
        for j in 0..3 {
            let mut term_c = 1.0;
            let mut term_s = 1.0;
            for f in 1..=j {
                let fj = f as f64;
                term_c *= -fj / ((2.0 * fj - 1.0) * (2.0 * fj));
                term_s *= -fj / ((2.0 * fj) * (2.0 * fj + 1.0));
            }
            let mut acc_c = term_c;
            let mut acc_s = term_s;
            // Independent bit-identical early stops for the two channels: each
            // accumulator is touched only by its own term chain, whose magnitude
            // decreases monotonically on |u| ≤ CS_SERIES_U_MAX, so dropping the
            // no-op tail of either reproduces its full CS_SERIES_TERMS sum exactly.
            let mut done_c = false;
            let mut done_s = false;
            for m in j..(j + CS_SERIES_TERMS) {
                if done_c && done_s {
                    break;
                }
                let mf = m as f64;
                let jf = j as f64;
                if !done_c {
                    let ratio_c =
                        -u * (mf + 1.0) / ((mf + 1.0 - jf) * (2.0 * mf + 1.0) * (2.0 * mf + 2.0));
                    term_c *= ratio_c;
                    let next = acc_c + term_c;
                    if next == acc_c {
                        done_c = true;
                    } else {
                        acc_c = next;
                    }
                }
                if !done_s {
                    let ratio_s =
                        -u * (mf + 1.0) / ((mf + 1.0 - jf) * (2.0 * mf + 2.0) * (2.0 * mf + 3.0));
                    term_s *= ratio_s;
                    let next = acc_s + term_s;
                    if next == acc_s {
                        done_s = true;
                    } else {
                        acc_s = next;
                    }
                }
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
        let mut c = [c0, 0.0, 0.0];
        let mut s = [s0, 0.0, 0.0];
        for j in 0..2 {
            s[j + 1] = (c[j] - (2.0 * j as f64 + 1.0) * s[j]) / (2.0 * u);
            c[j + 1] = -s[j] / 2.0;
        }
        (c, s)
    }
}

/// Value-only `T(w)` — bit-for-bit `t_stacks(w)[0]` (and `t_stacks3(w)[0]`),
/// the *only* slot the geodesic-distance / log-map value paths consume.
///
/// `distance`, `log_map`, and the radial code read just `[0]`, yet the full
/// [`t_stacks`] builds five independent 48-term series to do it. This computes
/// the `j = 0` series alone — the identical arithmetic (`jf = 0`, so
/// `mf + 1.0 - jf == mf + 1.0`) — and stops as soon as a term no longer moves
/// the sum (monotone tail on `|w| ≤ T_SERIES_W_MAX`, so a strict no-op prune).
pub(crate) fn t0(w: f64) -> f64 {
    if w.abs() <= T_SERIES_W_MAX {
        let mut term = 1.0;
        let mut acc = 1.0;
        for m in 0..T_SERIES_TERMS {
            let mf = m as f64;
            term *= -w * (mf + 1.0) * (2.0 * mf + 1.0) / ((mf + 1.0) * (2.0 * mf + 3.0));
            let next = acc + term;
            if next == acc {
                break;
            }
            acc = next;
        }
        acc
    } else if w > 0.0 {
        let r = w.sqrt();
        r.atan() / r
    } else {
        let r = (-w).sqrt();
        r.atanh() / r
    }
}

/// Value-only `(C(u), S(u))` — bit-for-bit `(cs_stacks(u).0[0], cs_stacks(u).1[0])`.
///
/// The exp map's generalized tangent `tn_κ = t·S/C` needs both values; the radial
/// Jacobian needs only `S`; neither needs the four derivative slots [`cs_stacks`]
/// builds. This evaluates just the two `j = 0` series with the identical
/// arithmetic and the same no-op tail prune as [`t0`].
pub(crate) fn cs_val(u: f64) -> (f64, f64) {
    if u.abs() <= CS_SERIES_U_MAX {
        let mut term_c = 1.0;
        let mut acc_c = 1.0;
        for m in 0..CS_SERIES_TERMS {
            let mf = m as f64;
            term_c *= -u * (mf + 1.0) / ((mf + 1.0) * (2.0 * mf + 1.0) * (2.0 * mf + 2.0));
            let next = acc_c + term_c;
            if next == acc_c {
                break;
            }
            acc_c = next;
        }
        let mut term_s = 1.0;
        let mut acc_s = 1.0;
        for m in 0..CS_SERIES_TERMS {
            let mf = m as f64;
            term_s *= -u * (mf + 1.0) / ((mf + 1.0) * (2.0 * mf + 2.0) * (2.0 * mf + 3.0));
            let next = acc_s + term_s;
            if next == acc_s {
                break;
            }
            acc_s = next;
        }
        (acc_c, acc_s)
    } else if u > 0.0 {
        let r = u.sqrt();
        (r.cos(), r.sin() / r)
    } else {
        let r = (-u).sqrt();
        (r.cosh(), r.sinh() / r)
    }
}

/// Per-point weight coefficients of the constant-curvature Dirichlet Gram and
/// their `κ`-derivatives.
///
/// Returns `(iso, rad, d_iso, d_rad)` where `G = iso·(I − t̂t̂ᵀ) + rad·t̂t̂ᵀ`.
/// Splitting this out is what lets the value and the `κ`-derivative share one
/// assembly loop, so the two cannot drift apart.
///
/// The derivative is taken in LOGARITHMIC form,
///
/// ```text
///   ∂iso/∂κ = iso·[ (d−2)·λ̇/λ + (d−3)·ė_t/e_t + ė_r/e_r ]
///   ∂rad/∂κ = rad·[ (d−2)·λ̇/λ + (d−1)·ė_t/e_t − ė_r/e_r ]
/// ```
///
/// because each coefficient is a product of powers; that keeps the chain rule to
/// one line per factor instead of expanding a quotient of four `κ`-dependent
/// terms. The factor derivatives come from the `u = κr²` series already in this
/// module (`u̇ = r²`):
///
/// ```text
///   A  = d(S/C)/du,   tn = r·S/C,   ṫn = r³·A
///   tn′ = S/C + 2u·A,               ∂tn′/∂κ = r²·(3A + 2u·A′)
///   λ  = 2/(1 + κ·tn²),             λ̇ = −(λ²/2)·(tn² + 2κ·tn·ṫn)
/// ```
fn dirichlet_weight_coefficients(
    kappa: f64,
    r: f64,
    d: usize,
) -> GeometryResult<(f64, f64, f64, f64)> {
    if r <= GEOMETRY_EPS {
        // `D exp₀(0) = I`, so both eigenvalues are 1 and `λ = 2`; the weight is
        // isotropic and its κ-movement vanishes with `r`.
        let iso = 2.0_f64.powi(d as i32 - 2);
        return Ok((iso, iso, 0.0, 0.0));
    }
    let u = kappa * r * r;
    let (c_stack, s_stack) = cs_stacks(u);
    let (c, s) = (c_stack[0], s_stack[0]);
    if c.abs() <= GEOMETRY_EPS {
        return Err(GeometryError::Singular(
            "constant-curvature Dirichlet weight at a conjugate point (cos(√κ r) = 0)",
        ));
    }
    let (c1, s1, c2, s2) = (c_stack[1], s_stack[1], c_stack[2], s_stack[2]);
    // A = d(S/C)/du and A′ = d²(S/C)/du².
    let a = (s1 * c - s * c1) / (c * c);
    let a_prime = (s2 * c - s * c2) / (c * c) - 2.0 * c1 * (s1 * c - s * c1) / (c * c * c);

    let tn = r * s / c;
    let tn_dot = r * r * r * a;
    let tn_prime = s / c + 2.0 * u * a;
    let tn_prime_dot = r * r * (3.0 * a + 2.0 * u * a_prime);

    let gauge = 1.0 + kappa * tn * tn;
    if gauge <= GEOMETRY_EPS {
        return Err(GeometryError::InvalidPoint(
            "constant-curvature Dirichlet weight: exp of the tangent coordinate leaves the chart",
        ));
    }
    let lambda = 2.0 / gauge;
    let lambda_dot = -0.5 * lambda * lambda * (tn * tn + 2.0 * kappa * tn * tn_dot);

    let e_t = tn / r;
    let e_t_dot = tn_dot / r;
    let e_r = tn_prime;
    let e_r_dot = tn_prime_dot;
    if !(e_t > 0.0 && e_r > 0.0) {
        return Err(GeometryError::Singular(
            "constant-curvature Dirichlet weight: exp differential is degenerate",
        ));
    }

    let lam_pow = lambda.powi(d as i32 - 2);
    let iso = lam_pow * e_t.powi(d as i32 - 3) * e_r;
    let rad = lam_pow * e_t.powi(d as i32 - 1) / e_r;
    let d_log_lambda = lambda_dot / lambda;
    let d_log_e_t = e_t_dot / e_t;
    let d_log_e_r = e_r_dot / e_r;
    let base = (d as f64 - 2.0) * d_log_lambda;
    let d_iso = iso * (base + (d as f64 - 3.0) * d_log_e_t + d_log_e_r);
    let d_rad = rad * (base + (d as f64 - 1.0) * d_log_e_t - d_log_e_r);
    Ok((iso, rad, d_iso, d_rad))
}

/// `∂S/∂κ` of [`constant_curvature_dirichlet_penalty`] — the κ-movement of the
/// penalty Gram.
///
/// This is the whole κ-channel for a constant-curvature ATOM. Its basis is a
/// monomial patch in the TANGENT coordinate, which does not depend on κ at all,
/// so unlike the constant-curvature GAM smooth (whose kernel design moves with
/// κ) there is no `∂X/∂κ` term. κ enters the criterion only here.
///
/// Same assembly as the value, with the coefficient derivatives substituted —
/// which is why they are computed together in
/// `dirichlet_weight_coefficients` rather than in two places.
pub fn constant_curvature_dirichlet_penalty_kappa_derivative(
    coords: ArrayView2<'_, f64>,
    basis_jacobian: ndarray::ArrayView3<'_, f64>,
    kappa: f64,
) -> GeometryResult<Array2<f64>> {
    dirichlet_gram_assembly(coords, basis_jacobian, kappa, true)
}

/// Shared assembly for the Dirichlet Gram and its `κ`-derivative.
///
/// `differentiate` selects which pair of per-point coefficients is accumulated.
/// The loop is identical either way, which is the point: the value and its
/// derivative are the same sum over the same points with the same projections,
/// so they cannot disagree about anything except the two scalars.
fn dirichlet_gram_assembly(
    coords: ArrayView2<'_, f64>,
    basis_jacobian: ndarray::ArrayView3<'_, f64>,
    kappa: f64,
    differentiate: bool,
) -> GeometryResult<Array2<f64>> {
    if !kappa.is_finite() {
        return Err(GeometryError::InvalidPoint(
            "constant-curvature Dirichlet weight needs a finite kappa",
        ));
    }
    let n = coords.nrows();
    let d = coords.ncols();
    let jet_shape = basis_jacobian.shape();
    if jet_shape[0] != n {
        return Err(GeometryError::DimensionMismatch {
            context: "constant_curvature_dirichlet_penalty: jacobian rows vs coords rows",
            expected: n,
            got: jet_shape[0],
        });
    }
    if jet_shape[2] != d {
        return Err(GeometryError::DimensionMismatch {
            context: "constant_curvature_dirichlet_penalty: jacobian latent axes vs coords cols",
            expected: d,
            got: jet_shape[2],
        });
    }
    let m = jet_shape[1];
    let mut gram = Array2::<f64>::zeros((m, m));
    if n == 0 || m == 0 {
        return Ok(gram);
    }
    let mut grad = vec![0.0_f64; m];
    let mut proj = vec![0.0_f64; m];
    for row in 0..n {
        let t = coords.row(row);
        let norm = t.iter().map(|x| x * x).sum::<f64>().sqrt();
        let (iso, rad, d_iso, d_rad) = dirichlet_weight_coefficients(kappa, norm, d)?;
        let (iso_coeff, rad_coeff) = if differentiate {
            (d_iso, d_rad)
        } else {
            (iso, rad)
        };
        if !(iso_coeff.is_finite() && rad_coeff.is_finite()) {
            continue;
        }
        for axis in 0..d {
            for k in 0..m {
                grad[k] = basis_jacobian[[row, k, axis]];
            }
            for i in 0..m {
                let gi = grad[i];
                if gi == 0.0 {
                    continue;
                }
                let scaled = iso_coeff * gi;
                for j in 0..m {
                    gram[[i, j]] += scaled * grad[j];
                }
            }
        }
        if norm > GEOMETRY_EPS {
            let radial_weight = rad_coeff - iso_coeff;
            if radial_weight != 0.0 {
                let inv_norm = 1.0 / norm;
                for k in 0..m {
                    let mut acc = 0.0;
                    for axis in 0..d {
                        acc += basis_jacobian[[row, k, axis]] * t[axis];
                    }
                    proj[k] = acc * inv_norm;
                }
                for i in 0..m {
                    let pi = proj[i];
                    if pi == 0.0 {
                        continue;
                    }
                    let scaled = radial_weight * pi;
                    for j in 0..m {
                        gram[[i, j]] += scaled * proj[j];
                    }
                }
            }
        }
    }
    Ok(gram)
}

/// Pullback Dirichlet Gram of a basis whose latent is the TANGENT COORDINATE at
/// the origin of `M_κ` — the κ-generic form of
/// [`crate::manifolds::poincare::conformal_dirichlet_penalty`].
///
/// # Why this exists
///
/// The hyperbolic version hardcodes `κ = −1` through `require_negative_curvature`,
/// which makes a SAE Poincaré atom a fixed-geometry special case rather than a
/// member of the `S^d ← ℝ^d → H^d` family. Freeing κ is what lets one atom cover
/// the whole family, so `Poincare` stops being its own topology and becomes
/// `κ < 0` of a constant-curvature atom (#2604, #2603).
///
/// A kernel basis cannot serve that role: the κ-generic RKHS kernel is
/// Matérn-½ and has a cusp at every center, which a fitted latent coordinate
/// differentiates through (measured in
/// `geodesic_exponential_kernel_has_unbounded_curvature_at_a_center`). A
/// tangent-chart monomial basis is `C^∞`, and the geometry lives here instead.
///
/// # The weight
///
/// `S = Σ_n Φ′(t_n)ᵀ G(t_n) Φ′(t_n)` with `G = √det h · h⁻¹` for the pullback
/// metric `h = (D exp₀)ᵀ (λ²δ) (D exp₀)`. `D exp₀(t)` is diagonal in the radial
/// frame with tangential eigenvalue `e_t` (multiplicity `d−1`) and radial `e_r`:
///
/// ```text
///   e_t = tn(r)/r,   e_r = tn′(r),   λ = 2/(1 + κ·tn(r)²),   r = ‖t‖
///   G(t) = λ^{d−2}[ e_t^{d−3} e_r (I − t̂t̂ᵀ) + e_t^{d−1} e_r⁻¹ t̂t̂ᵀ ]
/// ```
///
/// This is the SAME expression the hyperbolic version documents, with
/// `tanh(s)/s → tn(r)/r`, `sech²(s) → tn′(r)`, `2cosh²(s) → 2/(1 + κ·tn(r)²)`.
///
/// # Two exact reductions pin it
///
/// * **κ = −1** reproduces `conformal_dirichlet_penalty` entry for entry — the
///   same quantity by a different route, not merely a close one.
/// * **κ = 0** gives `G ≡ 2^{d−2} I` identically (not just as a `‖t‖ → 0`
///   limit), because `e_t = e_r = 1` and `λ = 2` for every `t`.
///
/// Both are asserted below. A wrong derivation fails one immediately, which is
/// the whole reason this is safe to write down.
pub fn constant_curvature_dirichlet_penalty(
    coords: ArrayView2<'_, f64>,
    basis_jacobian: ndarray::ArrayView3<'_, f64>,
    kappa: f64,
) -> GeometryResult<Array2<f64>> {
    dirichlet_gram_assembly(coords, basis_jacobian, kappa, false)
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
        Ok(2.0 * nw2.sqrt() * t0(self.kappa * nw2))
    }

    /// `tn_κ(t) = sn(t)/cs(t) = t·S(κt²)/C(κt²)` — the generalized tangent.
    fn tn(&self, t: f64) -> GeometryResult<f64> {
        let (c, s) = cs_val(self.kappa * t * t);
        if c.abs() <= GEOMETRY_EPS {
            return Err(GeometryError::Singular(
                "constant-curvature exp map at a conjugate point (cos(√κ t) = 0)",
            ));
        }
        Ok(t * s / c)
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
        let coeff = gauge * t0(self.kappa * w.dot(&w));
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
        // Sectional curvature is a property of a tangent 2-PLANE: a 1-D
        // manifold has none, and a degenerate (collinear) pair spans none —
        // the space-form identity K = κ·D/D is 0/0 there, not κ. The metric
        // is conformal (λ²·g_euclid), so 2-plane degeneracy is exactly
        // Euclidean Gram degeneracy: D = ⟨u,u⟩⟨v,v⟩ − ⟨u,v⟩².
        if self.dim < 2 {
            return Err(GeometryError::Singular(
                "sectional curvature undefined below dimension 2 (no tangent 2-plane)",
            ));
        }
        let (u, v) = tangent_pair;
        let uu = u.dot(&u);
        let vv = v.dot(&v);
        let uv = u.dot(&v);
        let gram_det = uu * vv - uv * uv;
        if !(gram_det > f64::EPSILON * uu * vv) {
            return Err(GeometryError::Singular(
                "sectional curvature undefined on a degenerate (collinear or zero) tangent pair",
            ));
        }
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
// above, evaluated with κ seeded as `Tower2::<1>::variable(κ, 0)` and the
// scalar primitives entering through their certified derivative stacks.
// Points are chart constants; everything κ-dependent is a tower. The
// returned channels are exact ∂/∂κ and ∂²/∂κ² — the design/penalty
// movement the outer ψ-channel consumes.

type KJet = Tower2<1>;

#[inline]
fn kjet_recip(z: KJet) -> KJet {
    let r = 1.0 / z.v;
    z.compose_unary([r, -r * r, 2.0 * r * r * r])
}

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
    let a = (kappa * (2.0 * xy + yy)).scale(-1.0) + 1.0;
    let b = kappa * xx + 1.0;
    let denom = (kappa * kappa) * (xx * yy) + (kappa * (2.0 * xy)).scale(-1.0) + 1.0;
    if denom.v.abs() <= MOBIUS_DENOM_EPS {
        return Err(GeometryError::Singular(
            "Möbius addition at the κ>0 antipodal point",
        ));
    }
    let inv = kjet_recip(denom);
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
    let t = arg.compose_unary(t_stacks3(arg.v));
    let d = nw2.sqrt() * t * 2.0;
    Ok((d.v, d.g[0], d.h[0][0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    /// REDUCTION 2. At `κ = 0` the geometry is flat and the tangent coordinate is
    /// the chart, so `e_t = e_r = 1` and `λ = 2` for EVERY `t` — hence
    /// `G ≡ 2^{d−2} I` identically, not merely in the `‖t‖ → 0` limit the
    /// hyperbolic doc records. The Gram must therefore be exactly
    /// `2^{d−2} · Σ_n Σ_a Φ′_a Φ′_aᵀ`, computed here independently.
    #[test]
    fn kappa_generic_dirichlet_is_the_flat_gram_at_zero_curvature() {
        let coords = ndarray::arr2(&[[0.12_f64, -0.30], [0.44, 0.05], [-0.21, 0.37]]);
        let (n, d, m) = (3usize, 2usize, 4usize);
        let mut jacobian = ndarray::Array3::<f64>::zeros((n, m, d));
        for row in 0..n {
            for k in 0..m {
                for axis in 0..d {
                    jacobian[[row, k, axis]] =
                        0.7 * (row as f64) - 0.23 * (k as f64) + 0.31 * (axis as f64) + 0.5;
                }
            }
        }
        let generic =
            constant_curvature_dirichlet_penalty(coords.view(), jacobian.view(), 0.0).unwrap();
        let weight = 2.0_f64.powi(d as i32 - 2);
        let mut expected = Array2::<f64>::zeros((m, m));
        for row in 0..n {
            for axis in 0..d {
                for i in 0..m {
                    for j in 0..m {
                        expected[[i, j]] +=
                            weight * jacobian[[row, i, axis]] * jacobian[[row, j, axis]];
                    }
                }
            }
        }
        let mut worst = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let scale = expected[[i, j]].abs().max(1.0);
                worst = worst.max((expected[[i, j]] - generic[[i, j]]).abs() / scale);
            }
        }
        assert!(
            worst <= 1.0e-12,
            "at kappa = 0 the weight must be exactly the flat Gram 2^(d-2)·sum Phi' Phi'^T; \
             worst relative gap {worst:.3e}"
        );
    }

    /// The weight is a genuine function of curvature, and positive-definite on
    /// both sides of flat. Without this the two reductions above could both pass
    /// for a weight that ignored `κ` outside those points.
    #[test]
    fn kappa_generic_dirichlet_moves_with_curvature_and_stays_psd() {
        let coords = ndarray::arr2(&[[0.10_f64, -0.20], [0.25, 0.05]]);
        let mut jacobian = ndarray::Array3::<f64>::zeros((2, 3, 2));
        for row in 0..2 {
            for k in 0..3 {
                for axis in 0..2 {
                    jacobian[[row, k, axis]] = 0.4 + 0.2 * (k as f64) - 0.3 * (axis as f64)
                        + 0.11 * (row as f64);
                }
            }
        }
        let flat =
            constant_curvature_dirichlet_penalty(coords.view(), jacobian.view(), 0.0).unwrap();
        for &kappa in &[-1.5_f64, -0.4, 0.4, 1.5] {
            let g =
                constant_curvature_dirichlet_penalty(coords.view(), jacobian.view(), kappa).unwrap();
            let gap = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .fold(0.0_f64, |acc, (i, j)| acc.max((g[[i, j]] - flat[[i, j]]).abs()));
            assert!(
                gap > 1.0e-6,
                "the weight must actually depend on kappa; kappa={kappa} matched flat"
            );
            // Symmetric PSD: it is a sum of outer products with positive weights.
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (g[[i, j]] - g[[j, i]]).abs() <= 1.0e-12,
                        "the Dirichlet Gram must be symmetric at kappa={kappa}"
                    );
                }
                assert!(
                    g[[i, i]] >= -1.0e-12,
                    "a PSD Gram cannot have a negative diagonal at kappa={kappa}"
                );
            }
        }
    }

    /// The analytic `∂S/∂κ` must match a central difference of the Gram itself.
    /// SPEC bans finite differences outside tests precisely so that the analytic
    /// derivative is the one that ships; this is the test that earns it.
    ///
    /// Checked across the sign of curvature, because the `κ`-series changes
    /// branch at zero and a derivative that only worked on one side would be a
    /// silent wrong smoothing on the other.
    #[test]
    fn constant_curvature_dirichlet_kappa_derivative_matches_finite_differences() {
        let coords = ndarray::arr2(&[[0.11_f64, -0.27], [0.33, 0.08], [-0.19, 0.24]]);
        let mut jacobian = ndarray::Array3::<f64>::zeros((3, 4, 2));
        for row in 0..3 {
            for k in 0..4 {
                for axis in 0..2 {
                    jacobian[[row, k, axis]] =
                        0.37 * (row as f64 + 1.0) - 0.19 * (k as f64) + 0.23 * (axis as f64);
                }
            }
        }
        let h = 1.0e-6_f64;
        for &kappa in &[-1.3_f64, -0.5, 0.0, 0.5, 1.3] {
            let analytic = constant_curvature_dirichlet_penalty_kappa_derivative(
                coords.view(),
                jacobian.view(),
                kappa,
            )
            .unwrap();
            let plus =
                constant_curvature_dirichlet_penalty(coords.view(), jacobian.view(), kappa + h)
                    .unwrap();
            let minus =
                constant_curvature_dirichlet_penalty(coords.view(), jacobian.view(), kappa - h)
                    .unwrap();
            let mut worst = 0.0_f64;
            for i in 0..analytic.nrows() {
                for j in 0..analytic.ncols() {
                    let fd = (plus[[i, j]] - minus[[i, j]]) / (2.0 * h);
                    let scale = analytic[[i, j]].abs().max(fd.abs()).max(1.0);
                    worst = worst.max((analytic[[i, j]] - fd).abs() / scale);
                }
            }
            assert!(
                worst <= 1.0e-6,
                "analytic dS/dkappa disagrees with central differences at kappa={kappa}: \
                 worst relative gap {worst:.3e}"
            );
        }
    }

    /// At `κ = 0` the weight is `2^{d−2} I` for every `t`, but it is NOT
    /// stationary there: flat space is an interior point of the family, so the
    /// criterion must feel which way curvature would move. A zero derivative at
    /// zero curvature would silently pin every fit to flat.
    #[test]
    fn constant_curvature_dirichlet_kappa_derivative_is_nonzero_at_flat() {
        let coords = ndarray::arr2(&[[0.30_f64, -0.20], [0.15, 0.40]]);
        let mut jacobian = ndarray::Array3::<f64>::zeros((2, 3, 2));
        for row in 0..2 {
            for k in 0..3 {
                for axis in 0..2 {
                    jacobian[[row, k, axis]] = 0.5 + 0.2 * (k as f64) - 0.3 * (axis as f64);
                }
            }
        }
        let d_zero = constant_curvature_dirichlet_penalty_kappa_derivative(
            coords.view(),
            jacobian.view(),
            0.0,
        )
        .unwrap();
        let magnitude = d_zero.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            magnitude > 1.0e-6,
            "kappa = 0 must not be a stationary point of the penalty; got {magnitude:.3e}"
        );
    }

}
