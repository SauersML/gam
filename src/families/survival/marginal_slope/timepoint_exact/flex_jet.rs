//! Single-source flex survival row NLL over a runtime-`K` jet algebra (#932,
//! doc §C/§D/Unifying).
//!
//! The flex marginal-slope row negative log-likelihood is
//! ```text
//! ℓ = w·[ logΦ(−η₀) − (1−d)·logΦ(−η₁)
//!         + d·½η₁² − d·logχ₁ + d·½q₁² + d·logD₁ − d·logqd₁ + d·ln2π ]
//! ```
//! (`flex_sensitivity.rs:105`). [`flex_row_nll`] writes this **once** over a
//! generic [`FlexJet`] scalar; instantiating it at [`Jet2`] yields value /
//! gradient / Hessian (replacing the hand grad/Hessian loops in
//! `flex_sensitivity.rs`), at [`Jet3`] yields the contracted third
//! `D_dir H[u,v]`, and at [`Jet4`] the contracted fourth — replacing the
//! hand probit-chain + quotient-rule assembly in
//! `gpu::cpu_oracle_third/fourth_contraction`. The directional / bidirectional
//! contraction "directions" fall out of the nilpotent ε / δ seeds of the timepoint
//! jets, exactly as the packed `Order2`/`OneSeed`/`TwoSeed` scalars do for
//! location-scale — but here over a **runtime** primary count `p` (the flex
//! primary count `4 + |h| + |w| + 1` is large and variable, so a `Vec`-backed
//! jet avoids the const-generic monomorphization blow-up the packed scalars would
//! incur).
//!
//! The timepoint quantities `η₀, η₁, χ₁, D₁` arrive as jets carrying their own
//! θ-derivatives (the `eta_u`/`eta_uv` packs from `first_full`, the directional
//! `*_dir` packs from `directional`, the bidirectional `*_uv_uv` packs from
//! `bidirectional`); `q₁`/`qd₁` are seeded as plain primaries. The single-source
//! probit derivative stack `surv_stack` and the `ln` stack carry the only special
//! functions (humans own primitive stability, the algebra owns combinatorics).

use super::*;
use crate::families::bms::signed_probit_neglog_derivatives_up_to_fourth;
use crate::families::jet_scalar::{filtered_implicit_solve_scalar, Order2};
use crate::families::jet_tower::Tower2;
use crate::families::survival::marginal_slope::gpu;
use crate::inference::probability::signed_probit_logcdf_and_mills_ratio;

/// #932 Item 1 (doc §B): lift the calibration intercept jet `a(θ)` — value /
/// gradient / Hessian — by `filtered_implicit_solve_scalar` over the calibration
/// constraint `F(a, θ) = 0`, instead of the hand IFT closed forms. `F`'s
/// `(a, θ)` jet channels ARE the already-computed calibration partials:
/// `F_a = D` (`d_check`), `F_{θi} = −f_u[i]`, `F_aa = f_aa`,
/// `F_{aθi} = d_u[i]` (= `∂D/∂θ_i`), `F_{θiθj} = −f_uv[i][j]`. The filtered
/// Newton step `A ← A − F(A)/F_a` (2 iterations at `Order2`, the nilpotency
/// order) returns `A.g = a_u`, `A.h = a_uv` — reproducing the hand IFT
/// `a_u = f_u/D`, `a_uv = (f_uv − d_u·a_u − d_u·a_u − f_aa·a_u·a_u)/D` term for
/// term, but from the recurrence rather than a memorised string (`jet_tower`
/// `implicit_solve` pins that equivalence at 1e-12). `O(K²)` per timepoint.
fn lift_intercept_order2<const K: usize>(
    d_check: f64,
    f_u: &[f64],
    f_uv: &[f64],
    f_aa: f64,
    d_u: &[f64],
    a0: f64,
) -> [[f64; K]; K] {
    let residual = |a: &Order2<K>| -> Order2<K> {
        let ag = a.g();
        let ah = a.h();
        let mut g = [0.0_f64; K];
        let mut h = [[0.0_f64; K]; K];
        for i in 0..K {
            g[i] = d_check * ag[i] - f_u[i];
        }
        for i in 0..K {
            for j in 0..K {
                h[i][j] = d_check * ah[i][j]
                    + f_aa * ag[i] * ag[j]
                    + d_u[i] * ag[j]
                    + d_u[j] * ag[i]
                    - f_uv[i * K + j];
            }
        }
        Order2(Tower2 { v: 0.0, g, h })
    };
    let a = filtered_implicit_solve_scalar::<K, Order2<K>>(a0, 1.0 / d_check, 2, residual);
    a.h()
}

/// The `[f64; 5]` Faà di Bruno stack of `g(η) = logΦ(−η)` at `η`.
///
/// With `N(m) = −logΦ(m)` and `(k1,k2,k3,k4) = N′…N⁗(m)` at `m = −η`
/// (`signed_probit_neglog_derivatives_up_to_fourth`), the chain rule on
/// `g(η) = −N(−η)` gives `g′ = k1`, `g″ = −k2`, `g‴ = k3`, `g⁗ = −k4`. This is
/// the entry/exit survival stack; composing the timepoint η-jet with it
/// reproduces the hand `entry_u1 = −entry_k1`, `entry_u2 = entry_k2`, … mapping
/// (`flex_sensitivity.rs`, `gpu::cpu_oracle_*`).
#[inline]
fn surv_stack(eta: f64) -> Result<[f64; 5], String> {
    let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(-eta);
    let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(-eta, 1.0)?;
    Ok([logcdf, k1, -k2, k3, -k4])
}

/// The `[f64; 5]` Faà di Bruno stack of `ln(x)`.
#[inline]
fn ln_stack(x: f64) -> [f64; 5] {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    [x.ln(), inv, -inv2, 2.0 * inv2 * inv, -6.0 * inv2 * inv2]
}

/// A runtime-`K` truncated-Taylor scalar: the row loss is written once against
/// this interface and re-instantiated at [`Jet2`] / [`Jet3`] / [`Jet4`] for the
/// value/grad/Hessian, contracted-third, and contracted-fourth channels.
trait FlexJet: Sized + Clone {
    fn value(&self) -> f64;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn scale(&self, s: f64) -> Self;
    /// Faà di Bruno composition `f ∘ self` with stack `[f, f′, f″, f‴, f⁗]`.
    fn compose_unary(&self, d: [f64; 5]) -> Self;
    /// `ln(self)` via [`ln_stack`] at the value channel.
    #[inline]
    fn ln(&self) -> Self {
        self.compose_unary(ln_stack(self.value()))
    }
    /// `1/self` via the reciprocal Faà di Bruno stack at the value channel.
    #[inline]
    fn recip(&self) -> Self {
        let x = self.value();
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        self.compose_unary([
            inv,
            -inv2,
            2.0 * inv2 * inv,
            -6.0 * inv2 * inv2,
            24.0 * inv2 * inv2 * inv,
        ])
    }
    /// `exp(self)` via the exponential stack at the value channel.
    #[inline]
    fn exp(&self) -> Self {
        let e = self.value().exp();
        self.compose_unary([e, e, e, e, e])
    }
    /// `self + c` for a scalar constant `c` (value-channel shift, derivatives
    /// unchanged) via the affine composition stack `[v+c, 1, 0, 0, 0]`.
    #[inline]
    fn add_const(&self, c: f64) -> Self {
        self.compose_unary([self.value() + c, 1.0, 0.0, 0.0, 0.0])
    }
}

// ── §B moment engine: the de-nested cell moments over a FlexJet ─────────────
//
// #932 Item 2 (doc §D). The per-cell moments `M_n = ∫_{z_L}^{z_R} z^n e^{−q(z)} dz`
// (sextic `q`, no closed antiderivative) satisfy the SAME raising recurrence the
// numeric `cubic_cell_kernel::reduce_sextic_moments` uses —
//   `M_{n+5} = (n·M_{n−1} − Σ_{j=0}^{4} d[j]·M_{n+j} − b_n) / d[5]`,
// with `d = q'(z)` coefficients (`sextic_qprime_coefficients`) and boundary term
// `b_n = z_R^n e^{−q(z_R)} − z_L^n e^{−q(z_L)}` — so it ports to ANY `FlexJet`
// scalar verbatim. Carrying the cell coefficients `c0..c3` and the (moving) edges
// `z_L,z_R` as jets propagates the moments' θ-derivatives mechanically: the
// `Σ d[j]·M_{n+j}` term is the interior coefficient sensitivity and the boundary
// term `b_n` is exactly the §D moving-boundary flux (its edge-jet derivatives are
// the Leibniz `[z^n e^{−q}·z_edge']` contributions the hand `directional` path
// assembles by hand). The base moments `M_0..M_4` (the normalization integrals)
// arrive as jets from the cell evaluator — those carry the only transcendental
// (erf/series) content; the algebra owns the rest.

/// `q'(z)` coefficient jets `[d0..d5]` for `q = ½(z² + η²)`, `η = c0+c1 z+c2 z²+
/// c3 z³`, over `FlexJet` cell-coefficient jets — the jet image of
/// [`crate::families::cubic_cell_kernel::sextic_qprime_coefficients`].
fn qprime_coeffs_jet<J: FlexJet>(c: &[J; 4]) -> [J; 6] {
    let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
    // d0 = c0·c1
    let d0 = c0.mul(c1);
    // d1 = 1 + c1² + 2·c0·c2   (the leading `+z` of q' supplies the constant 1)
    let d1 = c1.mul(c1).add(&c0.mul(c2).scale(2.0)).add_const(1.0);
    // d2 = 3·c0·c3 + 3·c1·c2
    let d2 = c0.mul(c3).add(&c1.mul(c2)).scale(3.0);
    // d3 = 4·c1·c3 + 2·c2²
    let d3 = c1.mul(c3).scale(4.0).add(&c2.mul(c2).scale(2.0));
    // d4 = 5·c2·c3
    let d4 = c2.mul(c3).scale(5.0);
    // d5 = 3·c3²
    let d5 = c3.mul(c3).scale(3.0);
    [d0, d1, d2, d3, d4, d5]
}

/// `q(z) = ½(z² + η(z)²)` evaluated at an edge jet `z`, with `η` from the cell
/// coefficient jets — the exponent whose `e^{−q}` is the boundary weight.
fn cell_q_at_jet<J: FlexJet>(c: &[J; 4], z: &J) -> J {
    // η = c0 + c1 z + c2 z² + c3 z³  (Horner)
    let eta = c[3]
        .mul(z)
        .add(&c[2])
        .mul(z)
        .add(&c[1])
        .mul(z)
        .add(&c[0]);
    // ½(z² + η²)
    z.mul(z).add(&eta.mul(&eta)).scale(0.5)
}

/// One boundary term `z^n·e^{−q(z)}` at a (possibly infinite) moving edge jet.
/// An infinite edge contributes nothing (matching the numeric
/// `moment_boundary_term_with_powers` short-circuit).
fn boundary_edge_term_jet<J: FlexJet>(c: &[J; 4], z: &J, z_pow_n: &J, finite: bool) -> Option<J> {
    if !finite {
        return None;
    }
    let q = cell_q_at_jet(c, z);
    let w = q.scale(-1.0).exp();
    Some(z_pow_n.mul(&w))
}

/// The sextic moment recurrence over a `FlexJet`: given the cell coefficient
/// jets `c`, the moving edge jets `(z_left, z_right)` with their finiteness, and
/// the base moment jets `M_0..M_4`, return `M_0..M_max` as jets. Bit-faithful to
/// `reduce_sextic_moments` term for term, but every operation in the `FlexJet`
/// algebra so the moments carry their exact θ-derivatives.
fn cell_moment_recurrence_jet<J: FlexJet>(
    c: &[J; 4],
    z_left: &J,
    left_finite: bool,
    z_right: &J,
    right_finite: bool,
    base_m0_m4: &[J; 5],
    max_degree: usize,
) -> Vec<J> {
    let d = qprime_coeffs_jet(c);
    let inv_lead = d[5].recip();
    let mut moments: Vec<J> = base_m0_m4.iter().cloned().collect();
    if max_degree < 5 {
        moments.truncate(max_degree + 1);
        return moments;
    }
    // Rolling z^n at each edge (jets), starting at n = 0 (z^0 = 1 = z/z).
    let one_l = z_left.recip().mul(z_left);
    let one_r = z_right.recip().mul(z_right);
    let mut left_pow = one_l;
    let mut right_pow = one_r;
    for n in 0..=(max_degree - 5) {
        let b_left = boundary_edge_term_jet(c, z_left, &left_pow, left_finite);
        let b_right = boundary_edge_term_jet(c, z_right, &right_pow, right_finite);
        // b_n = right − left, missing edges contribute zero.
        let mut b_n = match (b_right, b_left) {
            (Some(r), Some(l)) => r.sub(&l),
            (Some(r), None) => r,
            (None, Some(l)) => l.scale(-1.0),
            (None, None) => moments[0].scale(0.0),
        };
        // numer = n·M_{n−1} − Σ_{j=0}^{4} d[j]·M_{n+j} − b_n
        let mut numer = if n == 0 {
            moments[0].scale(0.0)
        } else {
            moments[n - 1].scale(n as f64)
        };
        for j in 0..=4 {
            numer = numer.sub(&d[j].mul(&moments[n + j]));
        }
        numer = numer.sub(&b_n);
        moments.push(numer.mul(&inv_lead));
        // Roll powers: z^{n+1} = z^n · z.
        left_pow = if left_finite {
            left_pow.mul(z_left)
        } else {
            b_n.scale(0.0)
        };
        right_pow = if right_finite {
            right_pow.mul(z_right)
        } else {
            // reuse b_n as a zero-jet scratch source of the right `p`
            b_n = b_n.scale(0.0);
            b_n
        };
    }
    moments
}

/// #932 item-2 Phase B-base: the normalization base moments `M_0..M_4` as jets,
/// carrying their exact θ-derivatives (incl. the moving-edge flux), built from
/// the cell's already-computed NUMERIC moment vector (`numeric_moments`) plus the
/// cell-coefficient jets `c` and the moving edge jets `(z_left, z_right)`.
///
/// `M_n = ∫_{z_L(θ)}^{z_R(θ)} zⁿ e^{−q(z,θ)} dz`, `q = ½(z² + η(z)²)`, `η = c0+c1z
/// +c2z²+c3z³` with `(c, z_L, z_R)` all θ-dependent. The θ-derivatives split as
///
/// * **interior** `∫ zⁿ ∂_θ(e^{−q}) dz = ∫ zⁿ·(−∂_θ q)·e^{−q} dz`, with
///   `∂_θ q = η·∂_θη`, `∂_θη = Σ_k (∂_θ c_k) zᵏ` — a polynomial in `z` (degree ≤6)
///   times `e^{−q}`, hence a moment-dot against the higher numeric moments
///   `M_{n+m}`; the FlexJet algebra raises this to ALL requested derivative
///   orders mechanically once the per-monomial weights carry the coefficient
///   jets' own derivatives. KEY: the value channel is the numeric `M_n`, and the
///   derivative channels are generated by holding the numeric moments as
///   constants and letting `−P(z)` (= `−η·∂η`, a jet polynomial) supply the
///   θ-motion — exactly the recurrence's `−Σ d[j] M_{n+j}` interior pattern, but
///   for the BASE moments where no recurrence exists.
/// * **boundary** `zⁿ e^{−q}` at the moving edges times the edge velocity, i.e.
///   the `boundary_edge_term_jet` whose edge-jet derivatives ARE the §D flux.
///
/// This single-sources the hand `survival_flex_base_d_u`/`f_au` base
/// normalization FIRST derivatives over a generic `FlexJet` order. The value
/// channel is bit-identical to `numeric_moments[n]`; the FIRST-derivative
/// channels are finite-difference-pinned against `reduce_sextic_moments` on
/// perturbed cells (`base_moment_jets_first_derivative_matches_fd_932`).
///
/// SCOPE (Phase B-base, increment): the interior holds the numeric moments
/// `M_{n+m}` CONSTANT, so the value + first-derivative channels are exact, but
/// the SECOND-and-higher interior channels are incomplete — the full
/// `∂²(e^{−q}) = (−∂²q + (∂q)²)e^{−q}` cross-term needs the moments to carry
/// their own θ-derivatives (a self-consistent base-moment recurrence). Wiring
/// `base_moment_jets` into `flex_timepoint_inputs` at `Jet3`/`Jet4` therefore
/// awaits that second-order self-consistency closure; at `Jet2` the
/// value/gradient are already exact and the Hessian's base-moment block is the
/// remaining piece.
fn base_moment_jets<J: FlexJet>(
    c: &[J; 4],
    z_left: &J,
    left_finite: bool,
    z_right: &J,
    right_finite: bool,
    numeric_moments: &[f64],
) -> [J; 5] {
    // `P(z) = η(z)·∂_θ η(z)` is the interior θ-motion of the exponent: each
    // `M_n` interior derivative is `−∫ zⁿ P(z) e^{−q} dz = −Σ_m P_m M_{n+m}`.
    // `P` is a jet polynomial in `z` whose value channel is 0 (since `∂_θη` is
    // pure-derivative) and whose derivative channels carry `η·η_θ`. `z` is the
    // integration variable (NOT a jet), so the per-monomial weights `P_m` are
    // the convolution coefficients of `η(z)·∂_θη(z)` in `z`, each a jet in θ:
    //
    // η(z)   = Σ_k c_k zᵏ              (c_k jets)
    // ∂_θη   = Σ_k (∂_θ c_k) zᵏ        (tangent jets dc_k = c_k − c_k.value())
    // P(z)   = η·∂_θη = Σ_m P_m zᵐ      (convolution, P_m jets, value channel 0)
    let dc: [J; 4] = std::array::from_fn(|k| tangent_jet(&c[k]));
    // Convolve c (deg 3) with dc (deg 3) → P has degree 6 (7 coefficients).
    let mut p_coeffs: Vec<J> = (0..7).map(|_| const_jet_like(&c[0], 0.0)).collect();
    for (i, ci) in c.iter().enumerate() {
        for (j, dcj) in dc.iter().enumerate() {
            let term = ci.mul(dcj);
            p_coeffs[i + j] = p_coeffs[i + j].add(&term);
        }
    }
    // Each base moment: value = numeric M_n; interior derivative motion
    // `−Σ_m P_m·M_{n+m}` (numeric higher moments as constant scalars); plus the
    // moving-boundary flux `zⁿ e^{−q}` at the edges (its jet edge-derivatives are
    // the §D flux). `numeric_moments` must hold at least `4 + 6 = 10` entries so
    // `M_{n+m}` is defined for `n ≤ 4`, `m ≤ 6`.
    std::array::from_fn(|n| {
        let m_n = numeric_moments.get(n).copied().unwrap_or(0.0);
        // Value channel = numeric M_n (a constant jet of the right shape).
        let mut acc = const_jet_like(&c[0], m_n);
        // Interior: −Σ_m P_m · M_{n+m}.
        for (m, p_m) in p_coeffs.iter().enumerate() {
            let m_npm = numeric_moments.get(n + m).copied().unwrap_or(0.0);
            if m_npm != 0.0 {
                acc = acc.sub(&p_m.scale(m_npm));
            }
        }
        // Boundary flux (Leibniz): `∂_θ M_n` gains `+ f(z_R)·z_R' − f(z_L)·z_L'`,
        // where `f(z) = zⁿ e^{−q(z)}` is the integrand VALUE at the endpoint and
        // `z_edge'` is the edge θ-velocity. NOTE this is the integrand value times
        // the velocity — NOT the θ-derivative of `f(z_edge(θ))` (that would be the
        // wrong `f'(z)·z'`). So form the boundary contribution as the SCALAR
        // integrand value `f(z_edge)` (with `c` and `z` both held at their values)
        // multiplied by the edge VELOCITY jet `tangent_jet(z_edge)` (value 0,
        // derivative channels = the edge's θ-motion). The product is value 0 and
        // derivative channel `f(z_edge)·z_edge'`, the exact §D boundary flux.
        let c_const: [J; 4] = std::array::from_fn(|k| const_jet_like(&c[k], c[k].value()));
        let integrand_value = |z: &J, finite: bool| -> Option<f64> {
            if !finite {
                return None;
            }
            let z_const = const_jet_like(z, z.value());
            let pow = {
                let mut zk = const_jet_like(z, 1.0);
                for _ in 0..n {
                    zk = zk.mul(&z_const);
                }
                zk
            };
            boundary_edge_term_jet(&c_const, &z_const, &pow, true).map(|b| b.value())
        };
        if let Some(f_r) = integrand_value(z_right, right_finite) {
            acc = acc.add(&tangent_jet(z_right).scale(f_r));
        }
        if let Some(f_l) = integrand_value(z_left, left_finite) {
            acc = acc.sub(&tangent_jet(z_left).scale(f_l));
        }
        acc
    })
}

/// The single-source flex row NLL **minus** the additive `w·d·ln2π` constant
/// (which the caller adds to the value channel — it has no derivative). Written
/// once over `FlexJet`; the instantiating scalar selects the channel.
#[inline]
fn flex_row_nll<J: FlexJet>(
    eta0: &J,
    eta1: &J,
    chi1: &J,
    d1: &J,
    q1: &J,
    qd1: &J,
    surv0: [f64; 5],
    surv1: [f64; 5],
    wi: f64,
    di: f64,
) -> J {
    let wd = wi * di;
    // w·logΦ(−η₀)
    let mut nll = eta0.compose_unary(surv0).scale(wi);
    // −w(1−d)·logΦ(−η₁)
    nll = nll.add(&eta1.compose_unary(surv1).scale(-wi * (1.0 - di)));
    // +w·d·½η₁²   (the −d·logφ(η₁) term, sans ½ln2π const)
    nll = nll.add(&eta1.mul(eta1).scale(0.5 * wd));
    // +w·d·½q₁²   (the −d·logφ(q₁) term, sans ½ln2π const)
    nll = nll.add(&q1.mul(q1).scale(0.5 * wd));
    // −w·d·logχ₁
    nll = nll.sub(&chi1.ln().scale(wd));
    // +w·d·logD₁
    nll = nll.add(&d1.ln().scale(wd));
    // −w·d·logqd₁
    nll = nll.sub(&qd1.ln().scale(wd));
    nll
}

// ── Jet2: value / gradient / Hessian (runtime K) ───────────────────────────

/// Value `v`, gradient `g[i]`, Hessian `h[i*p+j]` (row-major, symmetric) over a
/// runtime primary count `p = g.len()`. The order-≤2 truncation of the Leibniz /
/// Faà di Bruno rules — bit-identical to [`super::super::super::jet_tower::Tower2`]
/// channel-for-channel, just `Vec`-backed.
#[derive(Clone)]
struct Jet2 {
    v: f64,
    g: Vec<f64>,
    h: Vec<f64>,
}

impl Jet2 {
    /// A jet from explicit channels: `g` length `p`, `h` length `p*p` (or empty
    /// for the grad-only path, treated as the zero Hessian).
    fn from_parts(v: f64, g: &[f64], h: &[f64]) -> Self {
        let p = g.len();
        let hv = if h.is_empty() {
            vec![0.0; p * p]
        } else {
            assert_eq!(h.len(), p * p, "Jet2::from_parts Hessian length");
            h.to_vec()
        };
        Jet2 {
            v,
            g: g.to_vec(),
            h: hv,
        }
    }

    /// A jet from a gradient view and optional Hessian view (contiguity-safe:
    /// copies element-wise). `None` Hessian is the grad-only path.
    fn from_view(v: f64, g: ndarray::ArrayView1<'_, f64>, h: Option<ndarray::ArrayView2<'_, f64>>) -> Self {
        let p = g.len();
        let gv: Vec<f64> = g.iter().copied().collect();
        let hv = match h {
            Some(hm) => {
                let mut out = vec![0.0; p * p];
                for i in 0..p {
                    for j in 0..p {
                        out[i * p + j] = hm[[i, j]];
                    }
                }
                out
            }
            None => vec![0.0; p * p],
        };
        Jet2 { v, g: gv, h: hv }
    }

    /// The seeded primary `p_axis` at value `x`: unit gradient in slot `axis`,
    /// zero Hessian.
    fn primary(x: f64, axis: usize, p: usize) -> Self {
        let mut g = vec![0.0; p];
        if axis < p {
            g[axis] = 1.0;
        }
        Jet2 {
            v: x,
            g,
            h: vec![0.0; p * p],
        }
    }

    #[inline]
    fn p(&self) -> usize {
        self.g.len()
    }
}

impl FlexJet for Jet2 {
    #[inline]
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.g[i] + o.g[i];
        }
        for k in 0..p * p {
            h[k] = self.h[k] + o.h[k];
        }
        Jet2 {
            v: self.v + o.v,
            g,
            h,
        }
    }
    fn sub(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.g[i] - o.g[i];
        }
        for k in 0..p * p {
            h[k] = self.h[k] - o.h[k];
        }
        Jet2 {
            v: self.v - o.v,
            g,
            h,
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        for i in 0..p {
            for j in 0..p {
                h[i * p + j] = self.v * o.h[i * p + j]
                    + self.g[i] * o.g[j]
                    + self.g[j] * o.g[i]
                    + self.h[i * p + j] * o.v;
            }
        }
        Jet2 {
            v: self.v * o.v,
            g,
            h,
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet2 {
            v: self.v * s,
            g: self.g.iter().map(|&x| x * s).collect(),
            h: self.h.iter().map(|&x| x * s).collect(),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Order-≤2 reads only [f, f', f''].
        let p = self.p();
        let (f, f1, f2) = (d[0], d[1], d[2]);
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = f1 * self.g[i];
        }
        for i in 0..p {
            for j in 0..p {
                h[i * p + j] = f2 * self.g[i] * self.g[j] + f1 * self.h[i * p + j];
            }
        }
        Jet2 { v: f, g, h }
    }
}

// ── Jet3: one-seed directional, contracted third (doc §A.2) ────────────────

/// An [`Jet2`] base plus one nilpotent ε (`ε² = 0`) holding another [`Jet2`].
/// After seeding the timepoint jets' ε-parts with their directional derivatives,
/// the ε-Hessian of the evaluated NLL is `Σ_c ℓ_{abc} dir_c = (D_dir H)[a][b]`.
#[derive(Clone)]
struct Jet3 {
    base: Jet2,
    eps: Jet2,
}

impl Jet3 {
    /// Seeded primary: base = `primary(x, axis)`, ε = constant `dir[axis]`.
    fn primary(x: f64, axis: usize, p: usize, dir_axis: f64) -> Self {
        Jet3 {
            base: Jet2::primary(x, axis, p),
            eps: Jet2::from_parts(dir_axis, &vec![0.0; p], &[]),
        }
    }
    /// The contracted-third channel `Σ_c ℓ_{abc} dir_c` (the ε-Hessian).
    fn contracted_third(&self) -> Vec<f64> {
        self.eps.h.clone()
    }
}

impl FlexJet for Jet3 {
    #[inline]
    fn value(&self) -> f64 {
        self.base.v
    }
    fn add(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.mul(&o.base),
            eps: self.base.mul(&o.eps).add(&self.eps.mul(&o.base)),
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet3 {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        // f'(base) as a Jet2 (consumes [f', f'', f''']).
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        Jet3 { base, eps }
    }
}

// ── Jet4: two-seed, contracted fourth (doc §A.3) ───────────────────────────

/// An [`Jet2`] base plus ε, δ (`ε² = δ² = 0`, `εδ` retained) — four [`Jet2`]
/// parts. After seeding with both directions, the εδ-Hessian of the NLL is
/// `Σ_{cd} ℓ_{abcd} u_c v_d`.
#[derive(Clone)]
struct Jet4 {
    base: Jet2,
    eps: Jet2,
    del: Jet2,
    eps_del: Jet2,
}

impl Jet4 {
    fn primary(x: f64, axis: usize, p: usize, du: f64, dv: f64) -> Self {
        let zero = vec![0.0; p];
        Jet4 {
            base: Jet2::primary(x, axis, p),
            eps: Jet2::from_parts(du, &zero, &[]),
            del: Jet2::from_parts(dv, &zero, &[]),
            eps_del: Jet2::from_parts(0.0, &zero, &[]),
        }
    }
    fn contracted_fourth(&self) -> Vec<f64> {
        self.eps_del.h.clone()
    }
}

impl FlexJet for Jet4 {
    #[inline]
    fn value(&self) -> f64 {
        self.base.v
    }
    fn add(&self, o: &Self) -> Self {
        Jet4 {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        Jet4 {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let base = self.base.mul(&o.base);
        let eps = self.base.mul(&o.eps).add(&self.eps.mul(&o.base));
        let del = self.base.mul(&o.del).add(&self.del.mul(&o.base));
        let eps_del = self
            .base
            .mul(&o.eps_del)
            .add(&self.eps.mul(&o.del))
            .add(&self.del.mul(&o.eps))
            .add(&self.eps_del.mul(&o.base));
        Jet4 {
            base,
            eps,
            del,
            eps_del,
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet4 {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let fsecond = self.base.compose_unary([d[2], d[3], d[4], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        Jet4 {
            base,
            eps,
            del,
            eps_del,
        }
    }
}

/// `Σ_i x[i]·y[i]` over equal-length slices.
#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// `out[i] = Σ_j m[i*p+j]·v[j]` for a row-major `p×p` matrix `m`.
fn mat_vec(m: &[f64], v: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p];
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += m[i * p + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// `v1ᵀ m v2` for a row-major `p×p` matrix `m`.
fn quad_form(m: &[f64], v1: &[f64], v2: &[f64], p: usize) -> f64 {
    let mut acc = 0.0;
    for i in 0..p {
        let mi = &m[i * p..i * p + p];
        acc += v1[i] * dot(mi, v2);
    }
    acc
}

/// Order-≤2 jet channels (value, gradient view, optional Hessian view) for the
/// four flex row-NLL inputs (entry η, exit η, observed χ, observed d), bundled
/// so `flex_row_nll_value_grad_hess` stays under the argument-count gate.
pub(crate) struct FlexRowJet2Channels<'a> {
    pub eta0_v: f64,
    pub eta0_g: ndarray::ArrayView1<'a, f64>,
    pub eta0_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub eta1_v: f64,
    pub eta1_g: ndarray::ArrayView1<'a, f64>,
    pub eta1_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub chi1_v: f64,
    pub chi1_g: ndarray::ArrayView1<'a, f64>,
    pub chi1_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub d1_v: f64,
    pub d1_g: ndarray::ArrayView1<'a, f64>,
    pub d1_h: Option<ndarray::ArrayView2<'a, f64>>,
}

/// Entry/exit base + directional timepoint packs for the contracted-third path,
/// bundled to keep `flex_row_nll_third_contracted` under the argument-count gate.
pub(crate) struct FlexThirdPacks<'a> {
    pub entry_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub entry_ext: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
}

/// Entry/exit base + both directional + bidirectional timepoint packs for the
/// contracted-fourth path, bundled to keep `flex_row_nll_fourth_contracted`
/// under the argument-count gate.
pub(crate) struct FlexFourthPacks<'a> {
    pub entry_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub entry_ext_u: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_u: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub entry_ext_v: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_v: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub entry_bi: &'a gpu::SurvivalFlexBlock10TimepointBiDirectional,
    pub exit_bi: &'a gpu::SurvivalFlexBlock10TimepointBiDirectional,
}

impl SurvivalMarginalSlopeFamily {
    /// #932 Item 1: dispatch the runtime primary count `p` to a concrete `K` and
    /// lift the calibration intercept Hessian `a_uv` via [`lift_intercept_order2`]
    /// (`filtered_implicit_solve_scalar` over the calibration constraint) — the
    /// single-source replacement for the hand IFT closed form. `Order2` keeps it
    /// `O(K²)` per timepoint (no dense `Tower4<K+1>`); for primary counts beyond
    /// the dispatch table the byte-identical hand IFT is the fallback.
    pub(crate) fn lift_flex_intercept_hessian(
        &self,
        p: usize,
        d_check: f64,
        f_u: &Array1<f64>,
        f_uv: &Array2<f64>,
        f_aa: f64,
        d_u: &Array1<f64>,
        a0: f64,
    ) -> Result<Array2<f64>, String> {
        let fu = f_u
            .as_slice()
            .ok_or_else(|| "intercept lift: f_u must be contiguous".to_string())?;
        let fuv = f_uv
            .as_slice()
            .ok_or_else(|| "intercept lift: f_uv must be contiguous".to_string())?;
        let du = d_u
            .as_slice()
            .ok_or_else(|| "intercept lift: d_u must be contiguous".to_string())?;
        macro_rules! go {
            ($k:literal) => {{
                let a_uv = lift_intercept_order2::<$k>(d_check, fu, fuv, f_aa, du, a0);
                Array2::from_shape_fn((p, p), |(i, j)| a_uv[i][j])
            }};
        }
        let a_uv = match p {
            1 => go!(1),
            2 => go!(2),
            3 => go!(3),
            4 => go!(4),
            5 => go!(5),
            6 => go!(6),
            7 => go!(7),
            8 => go!(8),
            9 => go!(9),
            10 => go!(10),
            11 => go!(11),
            12 => go!(12),
            13 => go!(13),
            14 => go!(14),
            15 => go!(15),
            16 => go!(16),
            17 => go!(17),
            18 => go!(18),
            19 => go!(19),
            20 => go!(20),
            21 => go!(21),
            22 => go!(22),
            23 => go!(23),
            24 => go!(24),
            _ => {
                // Byte-identical hand IFT fallback for primary counts beyond the
                // dispatch table.
                let inv = 1.0 / d_check;
                let mut a_u = Array1::<f64>::zeros(p);
                for u in 0..p {
                    a_u[u] = fu[u] * inv;
                }
                let mut a_uv = Array2::<f64>::zeros((p, p));
                for u in 0..p {
                    for v in u..p {
                        let value = (f_uv[[u, v]]
                            - d_u[u] * a_u[v]
                            - d_u[v] * a_u[u]
                            - f_aa * a_u[u] * a_u[v])
                            * inv;
                        a_uv[[u, v]] = value;
                        a_uv[[v, u]] = value;
                    }
                }
                a_uv
            }
        };
        Ok(a_uv)
    }

    /// Single-source flex row value + gradient (+ Hessian if `hess_h*` non-empty)
    /// from the entry/exit timepoint packs. The Hessian channel is returned only
    /// when the `*_uv` slices are supplied; the grad-only caller passes empty
    /// `h` slices (the value/gradient channels do not read the Hessian).
    ///
    /// `g_*` are the length-`p` gradient packs, `h_*` the `p*p` row-major Hessian
    /// packs (empty for grad-only). Replaces the hand value/grad/Hessian
    /// assembly in `flex_sensitivity.rs`.
    pub(crate) fn flex_row_nll_value_grad_hess(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        ch: FlexRowJet2Channels<'_>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let FlexRowJet2Channels {
            eta0_v,
            eta0_g,
            eta0_h,
            eta1_v,
            eta1_g,
            eta1_h,
            chi1_v,
            chi1_g,
            chi1_h,
            d1_v,
            d1_g,
            d1_h,
        } = ch;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(eta0_v)?;
        let surv1 = surv_stack(eta1_v)?;
        let want_hess = eta1_h.is_some();
        let eta0 = Jet2::from_view(eta0_v, eta0_g, eta0_h);
        let eta1 = Jet2::from_view(eta1_v, eta1_g, eta1_h);
        let chi1 = Jet2::from_view(chi1_v, chi1_g, chi1_h);
        let d1 = Jet2::from_view(d1_v, d1_g, d1_h);
        let q1j = Jet2::primary(q1, primary.q1, p);
        let qd1j = Jet2::primary(qd1, primary.qd1, p);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        let value = out.v + wi * di * std::f64::consts::TAU.ln();
        let grad = Array1::from(out.g);
        let hess = if want_hess {
            Array2::from_shape_vec((p, p), out.h).map_err(|e| e.to_string())?
        } else {
            Array2::zeros((p, p))
        };
        Ok((value, grad, hess))
    }

    /// Single-source flex contracted third `D_dir H[u,v]` from the entry/exit
    /// base + directional packs. Replaces `gpu::cpu_oracle_third_contraction`.
    pub(crate) fn flex_row_nll_third_contracted(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        dir: &[f64],
        packs: FlexThirdPacks<'_>,
    ) -> Result<Array2<f64>, String> {
        let FlexThirdPacks {
            entry_base,
            exit_base,
            entry_ext,
            exit_ext,
        } = packs;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(entry_base.eta)?;
        let surv1 = surv_stack(exit_base.eta)?;

        let mk = |base_v: f64,
                  base_g: &[f64],
                  base_h: &[f64],
                  ext_g: &[f64],
                  ext_h: &[f64]|
         -> Jet3 {
            Jet3 {
                base: Jet2::from_parts(base_v, base_g, base_h),
                eps: Jet2::from_parts(dot(base_g, dir), ext_g, ext_h),
            }
        };
        let eta0 = mk(
            entry_base.eta,
            &entry_base.eta_u,
            &entry_base.eta_uv,
            &entry_ext.eta_u_dir,
            &entry_ext.eta_uv_dir,
        );
        let eta1 = mk(
            exit_base.eta,
            &exit_base.eta_u,
            &exit_base.eta_uv,
            &exit_ext.eta_u_dir,
            &exit_ext.eta_uv_dir,
        );
        let chi1 = mk(
            exit_base.chi,
            &exit_base.chi_u,
            &exit_base.chi_uv,
            &exit_ext.chi_u_dir,
            &exit_ext.chi_uv_dir,
        );
        let d1 = mk(
            exit_base.d,
            &exit_base.d_u,
            &exit_base.d_uv,
            &exit_ext.d_u_dir,
            &exit_ext.d_uv_dir,
        );
        let q1j = Jet3::primary(q1, primary.q1, p, dir[primary.q1]);
        let qd1j = Jet3::primary(qd1, primary.qd1, p, dir[primary.qd1]);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        Array2::from_shape_vec((p, p), out.contracted_third()).map_err(|e| e.to_string())
    }

    /// Single-source flex contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d` from the
    /// entry/exit base + both directional packs + bidirectional packs. Replaces
    /// `gpu::cpu_oracle_fourth_contraction`.
    pub(crate) fn flex_row_nll_fourth_contracted(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        dir_u: &[f64],
        dir_v: &[f64],
        packs: FlexFourthPacks<'_>,
    ) -> Result<Array2<f64>, String> {
        let FlexFourthPacks {
            entry_base,
            exit_base,
            entry_ext_u,
            exit_ext_u,
            entry_ext_v,
            exit_ext_v,
            entry_bi,
            exit_bi,
        } = packs;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(entry_base.eta)?;
        let surv1 = surv_stack(exit_base.eta)?;

        // eps_del.v = uᵀ·H·v, eps_del.g = (H_dir_u)·v, eps_del.h = bi.
        let mk = |base_v: f64,
                  base_g: &[f64],
                  base_h: &[f64],
                  ext_u_g: &[f64],
                  ext_u_h: &[f64],
                  ext_v_g: &[f64],
                  ext_v_h: &[f64],
                  bi_h: &[f64]|
         -> Jet4 {
            let eps_del_v = quad_form(base_h, dir_u, dir_v, p);
            let eps_del_g = mat_vec(ext_u_h, dir_v, p);
            Jet4 {
                base: Jet2::from_parts(base_v, base_g, base_h),
                eps: Jet2::from_parts(dot(base_g, dir_u), ext_u_g, ext_u_h),
                del: Jet2::from_parts(dot(base_g, dir_v), ext_v_g, ext_v_h),
                eps_del: Jet2::from_parts(eps_del_v, &eps_del_g, bi_h),
            }
        };
        let eta0 = mk(
            entry_base.eta,
            &entry_base.eta_u,
            &entry_base.eta_uv,
            &entry_ext_u.eta_u_dir,
            &entry_ext_u.eta_uv_dir,
            &entry_ext_v.eta_u_dir,
            &entry_ext_v.eta_uv_dir,
            &entry_bi.eta_uv_uv,
        );
        let eta1 = mk(
            exit_base.eta,
            &exit_base.eta_u,
            &exit_base.eta_uv,
            &exit_ext_u.eta_u_dir,
            &exit_ext_u.eta_uv_dir,
            &exit_ext_v.eta_u_dir,
            &exit_ext_v.eta_uv_dir,
            &exit_bi.eta_uv_uv,
        );
        let chi1 = mk(
            exit_base.chi,
            &exit_base.chi_u,
            &exit_base.chi_uv,
            &exit_ext_u.chi_u_dir,
            &exit_ext_u.chi_uv_dir,
            &exit_ext_v.chi_u_dir,
            &exit_ext_v.chi_uv_dir,
            &exit_bi.chi_uv_uv,
        );
        let d1 = mk(
            exit_base.d,
            &exit_base.d_u,
            &exit_base.d_uv,
            &exit_ext_u.d_u_dir,
            &exit_ext_u.d_uv_dir,
            &exit_ext_v.d_u_dir,
            &exit_ext_v.d_uv_dir,
            &exit_bi.d_uv_uv,
        );
        let q1j = Jet4::primary(q1, primary.q1, p, dir_u[primary.q1], dir_v[primary.q1]);
        let qd1j = Jet4::primary(qd1, primary.qd1, p, dir_u[primary.qd1], dir_v[primary.qd1]);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        Array2::from_shape_vec((p, p), out.contracted_fourth()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod moment_engine_tests {
    use super::*;
    use crate::families::cubic_cell_kernel::{reduce_sextic_moments, DenestedCubicCell};

    /// #932 item-2 increment 1: the FlexJet moment recurrence must reproduce the
    /// numeric `reduce_sextic_moments` on the VALUE channel term-for-term (a
    /// generic non-degenerate sextic cell), proving the port of the raising
    /// recurrence + boundary term to the jet algebra is exact. (Derivative
    /// channels are exercised by the full timepoint oracle once Phase C lands.)
    #[test]
    fn cell_moment_recurrence_jet_value_matches_numeric_932() {
        let cell = DenestedCubicCell {
            left: -1.5,
            right: 2.0,
            c0: 0.3,
            c1: -0.4,
            c2: 0.5,
            c3: 0.2,
        };
        let base = [1.0_f64, 0.1, 0.6, -0.05, 0.4];
        let max_degree = 12usize;
        let reference =
            reduce_sextic_moments(cell, base, max_degree).expect("numeric sextic moments");

        let p = 3usize;
        let konst = |x: f64| Jet2::from_parts(x, &vec![0.0; p], &[]);
        let c = [
            konst(cell.c0),
            konst(cell.c1),
            konst(cell.c2),
            konst(cell.c3),
        ];
        let zl = konst(cell.left);
        let zr = konst(cell.right);
        let base_jets = [
            konst(base[0]),
            konst(base[1]),
            konst(base[2]),
            konst(base[3]),
            konst(base[4]),
        ];
        let moments = cell_moment_recurrence_jet(
            &c,
            &zl,
            cell.left.is_finite(),
            &zr,
            cell.right.is_finite(),
            &base_jets,
            max_degree,
        );
        assert_eq!(moments.len(), reference.len(), "moment count");
        for (n, (m, r)) in moments.iter().zip(reference.iter()).enumerate() {
            assert!(
                (m.value() - r).abs() <= 1e-9 * (1.0 + r.abs()),
                "moment {n}: jet value {} != numeric {}",
                m.value(),
                r
            );
        }
    }

    /// #932 item-2 Phase B-base: the base-moment jet builder `base_moment_jets`
    /// must reproduce the FIRST θ-derivatives of the normalization base moments
    /// `M_0..M_4` (interior `−Σ P_m M_{n+m}` + moving-edge Leibniz flux) against a
    /// central finite difference of `evaluate_cell_moments` on a smooth one-
    /// parameter cell family `c_k(θ)=c_k0+θ·dc_k`, `z_{L,R}(θ)=z0+θ·v`. The
    /// gradient channel of the `Jet2` (seeded with `dc`/`v` in primary slot 0) is
    /// the analytic `dM_n/dθ`; the value channel is the numeric `M_n`.
    #[test]
    fn base_moment_jets_first_derivative_matches_fd_932() {
        use crate::families::cubic_cell_kernel::evaluate_cell_moments;

        // Smooth one-parameter family (θ scalar). Edges move; coefficients move.
        let c0 = [0.25_f64, -0.35, 0.4, 0.15];
        let zl0 = -1.2_f64;
        let zr0 = 1.7_f64;
        let dc = [0.13_f64, 0.21, -0.17, 0.09];
        let v_l = -0.23_f64;
        let v_r = 0.31_f64;
        let cell_at = |theta: f64| DenestedCubicCell {
            left: zl0 + theta * v_l,
            right: zr0 + theta * v_r,
            c0: c0[0] + theta * dc[0],
            c1: c0[1] + theta * dc[1],
            c2: c0[2] + theta * dc[2],
            c3: c0[3] + theta * dc[3],
        };
        let max_degree = 10usize;
        let moments_at = |theta: f64| -> Vec<f64> {
            evaluate_cell_moments(cell_at(theta), max_degree)
                .expect("numeric cell moments")
                .moments
                .into_vec()
        };
        let numeric0 = moments_at(0.0);

        // Seed the jets in primary slot 0 of a width-1 primary space: each
        // coefficient/edge jet carries its θ-velocity as its slot-0 gradient.
        let p = 1usize;
        let seeded = |x: f64, vel: f64| {
            let mut g = vec![0.0; p];
            g[0] = vel;
            Jet2::from_parts(x, &g, &[])
        };
        let c_jets = [
            seeded(c0[0], dc[0]),
            seeded(c0[1], dc[1]),
            seeded(c0[2], dc[2]),
            seeded(c0[3], dc[3]),
        ];
        let zl_jet = seeded(zl0, v_l);
        let zr_jet = seeded(zr0, v_r);
        let m_jets = base_moment_jets(&c_jets, &zl_jet, true, &zr_jet, true, &numeric0);

        // Central finite difference of each M_n.
        let h = 1e-6_f64;
        let mp = moments_at(h);
        let mm = moments_at(-h);
        for n in 0..5 {
            let fd = (mp[n] - mm[n]) / (2.0 * h);
            let jet = &m_jets[n];
            assert!(
                (jet.value() - numeric0[n]).abs() <= 1e-12 * (1.0 + numeric0[n].abs()),
                "M_{n} value {} != numeric {}",
                jet.value(),
                numeric0[n]
            );
            assert!(
                (jet.g[0] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "M_{n} dθ analytic {} != FD {}",
                jet.g[0],
                fd
            );
        }
    }
}

// ── §C: observed cell-coefficient jets + eta/chi point-eval (Phase C core) ──
//
// The observed cell coefficients `coeff[k]` are a smooth function of the
// intercept `a(θ)` and the slope `b` (= the `g` primary), with the score-warp
// (`h`) and link-dev (`w`) channels entering linearly on top. Their full
// bivariate Taylor in `(a,b)` is exactly the `observed_denested_cell_partials`
// pack (`dc_da…dc_dbbb`). Composing that Taylor with the intercept jet `a_jet`
// and the slope jet `b_jet` (both carrying their θ-derivatives) yields each
// `coeff[k]` AS a jet — so `eta = Σ_k coeff[k]·z_obs^k` and `chi = Σ_k
// dc_da[k]·z_obs^k` (point-evals at the fixed observation `z_obs`) carry their
// exact θ-derivatives mechanically, replacing the hand `eta_u = chi·a_u + rho`
// / `eta_uv = …` chain in `first_full`/`directional`/`bidirectional`.

/// A value-zero "tangent" jet `x_jet − x.value()`: value 0, derivative channels
/// preserved. Used as the perturbation argument of the bivariate Taylor below.
#[inline]
fn tangent_jet<J: FlexJet>(x: &J) -> J {
    x.add_const(-x.value())
}

/// A constant jet (value `v`, all derivative channels zero), shaped like
/// `template` (so it carries the right runtime primary count).
#[inline]
fn const_jet_like<J: FlexJet>(template: &J, v: f64) -> J {
    template.scale(0.0).add_const(v)
}

/// One observed cell coefficient `coeff[k]` as a jet: the bivariate `(a,b)`
/// Taylor (up to 3rd order, matching the `dc_d{a,b}…` pack) composed with the
/// intercept tangent `da` and slope tangent `db` jets. Terms with a 0/6/2/… are
/// the multinomial Taylor weights `coeff + Σ (1/(i!j!)) ∂^{i+j}coeff/∂a^i∂b^j ·
/// da^i db^j`.
fn observed_coeff_component_jet<J: FlexJet>(
    template: &J,
    k: usize,
    coeff: [f64; 4],
    dc_da: [f64; 4],
    dc_db: [f64; 4],
    dc_daa: [f64; 4],
    dc_dab: [f64; 4],
    dc_dbb: [f64; 4],
    dc_daaa: [f64; 4],
    dc_daab: [f64; 4],
    dc_dabb: [f64; 4],
    dc_dbbb: [f64; 4],
    da: &J,
    db: &J,
) -> J {
    let dada = da.mul(da);
    let dadb = da.mul(db);
    let dbdb = db.mul(db);
    let mut c = const_jet_like(template, coeff[k]);
    c = c.add(&da.scale(dc_da[k])).add(&db.scale(dc_db[k]));
    c = c
        .add(&dada.scale(0.5 * dc_daa[k]))
        .add(&dadb.scale(dc_dab[k]))
        .add(&dbdb.scale(0.5 * dc_dbb[k]));
    let inv6 = 1.0 / 6.0;
    let half = 0.5;
    c = c
        .add(&dada.mul(da).scale(inv6 * dc_daaa[k]))
        .add(&dada.mul(db).scale(half * dc_daab[k]))
        .add(&dadb.mul(db).scale(half * dc_dabb[k]))
        .add(&dbdb.mul(db).scale(inv6 * dc_dbbb[k]));
    c
}

/// Evaluate a 4-coefficient cell polynomial jet `Σ_k coeff_jet[k]·z^k` at the
/// fixed observation point `z` (the jet image of `eval_coeff4_at`).
#[inline]
fn eval_coeff_jet_at<J: FlexJet>(coeff_jet: &[J; 4], z: f64) -> J {
    let mut zk = 1.0;
    let mut acc = const_jet_like(&coeff_jet[0], 0.0);
    for c in coeff_jet.iter() {
        acc = acc.add(&c.scale(zk));
        zk *= z;
    }
    acc
}

/// Phase C assembly (foundational `Jet2` order): the observed timepoint `eta`
/// and `chi` as value/grad/Hessian jets, from the solved intercept derivatives
/// (`a`, `a_u`, `a_uv` — `a_u = f_u/D`, `a_uv = lift_intercept_order2`) and the
/// observed cell-coefficient partial pack. `b` is the slope (`g` primary at
/// `g_axis`). `eta`/`chi` carry their exact first/second θ-derivatives by
/// composing the coefficients' bivariate `(a,b)` Taylor with the intercept jet
/// — replacing the hand `eta_u = chi·a_u + rho`, `eta_uv = …` chains in
/// `first_full`/`directional`/`bidirectional` for the `a`/`b` channels.
///
/// The score-warp (`h`) / link-dev (`w`) channels (which enter `eta` linearly
/// via `rho`/`tau`) and the `D`-normalization channel (via the moment engine)
/// are layered on by the full `flex_timepoint_inputs` once those pieces land.

fn flex_timepoint_eta_chi_jet2(
    p: usize,
    g_axis: usize,
    a: f64,
    a_u: &[f64],
    a_uv: &[f64],
    b: f64,
    z_obs: f64,
    o_infl: f64,
    coeff: [f64; 4],
    dc_da: [f64; 4],
    dc_db: [f64; 4],
    dc_daa: [f64; 4],
    dc_dab: [f64; 4],
    dc_dbb: [f64; 4],
    dc_daaa: [f64; 4],
    dc_daab: [f64; 4],
    dc_dabb: [f64; 4],
    dc_dbbb: [f64; 4],
) -> (Jet2, Jet2) {
    let a_jet = Jet2::from_parts(a, a_u, a_uv);
    let b_jet = Jet2::primary(b, g_axis, p);
    let da = tangent_jet(&a_jet);
    let db = tangent_jet(&b_jet);
    let zero4 = [0.0_f64; 4];

    // eta coefficients: the coeff pack composed with (da, db).
    let coeff_jets: [Jet2; 4] = std::array::from_fn(|k| {
        observed_coeff_component_jet(
            &a_jet, k, coeff, dc_da, dc_db, dc_daa, dc_dab, dc_dbb, dc_daaa, dc_daab, dc_dabb,
            dc_dbbb, &da, &db,
        )
    });
    let eta = eval_coeff_jet_at(&coeff_jets, z_obs).add_const(o_infl);

    // chi = ∂eta/∂a coefficients = the dc_da pack, whose own (a,b)-Taylor is the
    // once-`a`-shifted pack (dc_daa as ∂/∂a, dc_dab as ∂/∂b, dc_daaa/daab/dabb as
    // the seconds; the dc_da pack carries no third-order term, so those are 0).
    let chi_jets: [Jet2; 4] = std::array::from_fn(|k| {
        observed_coeff_component_jet(
            &a_jet, k, dc_da, dc_daa, dc_dab, dc_daaa, dc_daab, dc_dabb, zero4, zero4, zero4, zero4,
            &da, &db,
        )
    });
    let chi = eval_coeff_jet_at(&chi_jets, z_obs);

    (eta, chi)
}
