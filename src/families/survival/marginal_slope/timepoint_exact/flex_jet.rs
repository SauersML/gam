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
    use crate::families::marginal_slope_shared::eval_coeff4_at;

    // #932: the `recip`/`exp`/`add_const` jet helpers (formerly `FlexJet` default
    // methods) live here as free generic fns — only the relocated moment-engine /
    // Phase-C builders below consume them, so keeping them inside the test module
    // avoids the orphaned-`dead_code` gate while preserving the exact derivations.
    fn recip<J: FlexJet>(x: &J) -> J {
        let v = x.value();
        let inv = 1.0 / v;
        let inv2 = inv * inv;
        x.compose_unary([inv, -inv2, 2.0 * inv2 * inv, -6.0 * inv2 * inv2, 24.0 * inv2 * inv2 * inv])
    }
    fn exp_jet<J: FlexJet>(x: &J) -> J {
        let e = x.value().exp();
        x.compose_unary([e, e, e, e, e])
    }
    fn add_const<J: FlexJet>(x: &J, c: f64) -> J {
        x.compose_unary([x.value() + c, 1.0, 0.0, 0.0, 0.0])
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
        let d1 = add_const(&c1.mul(c1).add(&c0.mul(c2).scale(2.0)), 1.0);
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
        let w = exp_jet(&q.scale(-1.0));
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
        let inv_lead = recip(&d[5]);
        let mut moments: Vec<J> = base_m0_m4.iter().cloned().collect();
        if max_degree < 5 {
            moments.truncate(max_degree + 1);
            return moments;
        }
        // Rolling z^n at each edge (jets), starting at n = 0 (z^0 = 1 = z/z).
        let one_l = recip(z_left).mul(z_left);
        let one_r = recip(z_right).mul(z_right);
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
    /// +c2z²+c3z³` with `(c, z_L, z_R)` all θ-dependent.
    ///
    /// This single-sources the hand `survival_flex_base_d_u`/`_d_uv`/`f_au`/`f_aa`
    /// base normalization derivatives over a generic `FlexJet` order — exact to ALL
    /// jet orders (Jet2/Jet3/Jet4), not just first. The value channel is
    /// bit-identical to `numeric_moments[n]`; the derivative channels are
    /// finite-difference-pinned against `evaluate_cell_moments` on perturbed cells
    /// (`base_moment_jets_first_derivative_matches_fd_932`,
    /// `base_moment_jets_second_derivative_matches_fd_932`).
    ///
    /// EXACTNESS to all orders (the self-consistent closure): write
    /// `M_n(θ) = ∫ zⁿ e^{−q(z,θ)} dz = ∫ zⁿ e^{−q(z,θ₀)}·e^{−Δq(z)} dz`,
    /// `Δq(z) = q(z,θ) − q(z,θ₀) = ½(η(z,θ)² − η(z,θ₀)²)` (the `z²` term cancels).
    /// The factor `e^{−Δq}` has VALUE channel 1 (Δq=0 at θ₀) and its derivative
    /// channels carry the full `(−∂q)` / `(−∂²q + (∂q)²)` / … expansion. Expanding
    /// `e^{−Δq}` as a jet-coefficient polynomial in `z` (`S(z)=Σ_m S_m zᵐ`, `S_m`
    /// jets) and dotting against the NUMERIC moments gives the interior
    /// `Σ_m S_m·M_{n+m}^{numeric}` — exact to every order because the `e^{−Δq}`
    /// expansion already contains the `(∂q)²` cross-term and higher. The truncation
    /// `e^{−Δq} ≈ Σ_{k≤4} (−Δq)^k/k!` is exact for the Jet≤4 nilpotency (`Δq` has
    /// value 0, so `(−Δq)^5` only feeds 5th-and-higher derivatives the order-≤4 jets
    /// discard). The boundary is the Leibniz flux `+ f(z_R)·z_R' − f(z_L)·z_L'`,
    /// integrand VALUE at the moving endpoint times the edge θ-velocity jet (exact to
    /// all orders via the edge-jet algebra).
    fn base_moment_jets<J: FlexJet>(
        c: &[J; 4],
        z_left: &J,
        left_finite: bool,
        z_right: &J,
        right_finite: bool,
        numeric_moments: &[f64],
    ) -> [J; 5] {
        // η₀ = value-only coefficient jets; jet-polynomial convolution helper.
        let c0_const: [J; 4] = std::array::from_fn(|k| const_jet_like(&c[k], c[k].value()));
        let conv = |lhs: &[J], rhs: &[J]| -> Vec<J> {
            let mut out: Vec<J> = (0..lhs.len() + rhs.len() - 1)
                .map(|_| const_jet_like(&c[0], 0.0))
                .collect();
            for (i, li) in lhs.iter().enumerate() {
                for (j, rj) in rhs.iter().enumerate() {
                    out[i + j] = out[i + j].add(&li.mul(rj));
                }
            }
            out
        };
        // −Δq(z) = −½(η² − η₀²), a jet-coefficient polynomial in z (value channel 0).
        let eta_sq = conv(c, c);
        let eta0_sq = conv(&c0_const, &c0_const);
        let neg_dq: Vec<J> = eta_sq
            .iter()
            .zip(eta0_sq.iter())
            .map(|(a, b)| a.sub(b).scale(-0.5))
            .collect();
        // S(z) = e^{−Δq} = Σ_{k=0}^{4} (−Δq)^k / k!  (jet-coefficient polynomial).
        // Truncating at k=4 is exact for the order-≤4 jets (value(−Δq)=0).
        let mut s_poly: Vec<J> = vec![const_jet_like(&c[0], 1.0)];
        let mut power: Vec<J> = s_poly.clone();
        let factorials = [1.0_f64, 1.0, 2.0, 6.0, 24.0];
        for fact in factorials.iter().skip(1) {
            power = conv(&power, &neg_dq);
            for (m, coeff) in power.iter().enumerate() {
                let term = coeff.scale(1.0 / fact);
                if m < s_poly.len() {
                    s_poly[m] = s_poly[m].add(&term);
                } else {
                    s_poly.push(term);
                }
            }
        }
        // The interior `Σ_m S_m·M_{n+m}^{numeric}` integrates `g(z,θ)=zⁿe^{−q(z,θ)}`
        // over the FIXED value-channel limits `[z_L0, z_R0]` (the numeric moments are
        // those fixed-limit integrals). The MOVING-limit correction is the thin
        // sliver `∫_{z_R0}^{z_R(θ)} g dz − ∫_{z_L0}^{z_L(θ)} g dz` (`edge_sliver_jet`),
        // exact to all jet orders.
        std::array::from_fn(|n| {
            let mut acc = const_jet_like(&c[0], 0.0);
            for (m, s_m) in s_poly.iter().enumerate() {
                let m_npm = numeric_moments.get(n + m).copied().unwrap_or(0.0);
                if m_npm != 0.0 {
                    acc = acc.add(&s_m.scale(m_npm));
                }
            }
            if let Some(sr) = edge_sliver_jet(n, c, z_right, right_finite) {
                acc = acc.add(&sr);
            }
            if let Some(sl) = edge_sliver_jet(n, c, z_left, left_finite) {
                acc = acc.sub(&sl);
            }
            acc
        })
    }

    /// The moving-edge sliver `∫_{z_E0}^{z_E(θ)} zⁿ e^{−q(z,θ)} dz` as a jet (value
    /// 0, derivative channels = the §D moving-boundary flux to all orders). With
    /// `δ = z_E − z_E0` (jet, value 0) and `g(z) = zⁿ e^{−q}`,
    /// `∫_{z_E0}^{z_E} g dz = g·δ + ½ g_z δ² + ⅙ g_zz δ³ + (1/24) g_zzz δ⁴` (Taylor
    /// in δ; δ⁵ vanishes for the order-≤4 jets). `g`, `g_z`, … are evaluated at the
    /// FIXED edge `z_E0` but with the θ-dependent coefficient jets `c`, so the sliver
    /// carries the full coefficient × edge cross-motion. `q = ½(z² + η²)`,
    /// `q_z = z + η η_z`, `η_z = c1 + 2c2 z + 3c3 z²`; the `g`-stack follows from
    /// `g_z = (n/z − q_z) g` by the product/chain rule.
    fn edge_sliver_jet<J: FlexJet>(n: usize, c: &[J; 4], z_e: &J, finite: bool) -> Option<J> {
        if !finite {
            return None;
        }
        let z0 = z_e.value();
        let zc = const_jet_like(z_e, z0); // fixed edge, value-only
        // η, η_z, η_zz, η_zzz at the fixed edge as jets (in c).
        let eta = c[3]
            .mul(&zc)
            .add(&c[2])
            .mul(&zc)
            .add(&c[1])
            .mul(&zc)
            .add(&c[0]);
        let eta_z = c[2]
            .scale(2.0)
            .add(&c[3].scale(3.0).mul(&zc))
            .mul(&zc)
            .add(&c[1]); // c1 + 2c2 z + 3c3 z²
        let eta_zz = c[2].scale(2.0).add(&c[3].scale(6.0).mul(&zc)); // 2c2 + 6c3 z
        let eta_zzz = c[3].scale(6.0); // 6c3
        // q_z = z + η η_z ; q_zz = 1 + η_z² + η η_zz ; q_zzz = 3 η_z η_zz + η η_zzz
        let q_z = zc.add(&eta.mul(&eta_z));
        let q_zz = add_const(&eta_z.mul(&eta_z).add(&eta.mul(&eta_zz)), 1.0);
        let q_zzz = eta_z.scale(3.0).mul(&eta_zz).add(&eta.mul(&eta_zzz));
        // g = zⁿ e^{−q}.
        let z_pow = {
            let mut zk = const_jet_like(z_e, 1.0);
            for _ in 0..n {
                zk = zk.mul(&zc);
            }
            zk
        };
        let q = zc.mul(&zc).add(&eta.mul(&eta)).scale(0.5);
        let w = exp_jet(&q.scale(-1.0));
        let g = z_pow.mul(&w);
        // n/z^k constants (z held at the fixed edge); 0 when n=0 or z0=0.
        let nz = |power: i32| -> J {
            if n == 0 || z0 == 0.0 {
                const_jet_like(z_e, 0.0)
            } else {
                const_jet_like(z_e, n as f64 / z0.powi(power))
            }
        };
        // g_z/g = a1 = n/z − q_z ; a1' = −n/z² − q_zz ; a1'' = 2n/z³ − q_zzz.
        let a1 = nz(1).sub(&q_z);
        let a1p = nz(2).scale(-1.0).sub(&q_zz);
        let a1pp = nz(3).scale(2.0).sub(&q_zzz);
        let g_z = a1.mul(&g);
        // g_zz/g = b2 = a1' + a1² ; g_zzz/g = b2' + a1 b2, b2' = a1'' + 2 a1 a1'.
        let b2 = a1p.add(&a1.mul(&a1));
        let g_zz = b2.mul(&g);
        let b2p = a1pp.add(&a1.mul(&a1p).scale(2.0));
        let g_zzz = b2p.add(&a1.mul(&b2)).mul(&g);
        // δ-power jets (δ value 0).
        let delta = tangent_jet(z_e);
        let d2 = delta.mul(&delta);
        let d3 = d2.mul(&delta);
        let d4 = d3.mul(&delta);
        Some(
            g.mul(&delta)
                .add(&g_z.mul(&d2).scale(0.5))
                .add(&g_zz.mul(&d3).scale(1.0 / 6.0))
                .add(&g_zzz.mul(&d4).scale(1.0 / 24.0)),
        )
    }

    /// #932 item-2 Phase C STEP 3: the single-source timepoint inputs `(eta, chi, d)`
    /// at `Jet2` (value/grad/Hess), assembled from the generic FlexJet building
    /// blocks — the intercept lift (`lift_intercept_flex`), the observed eta/chi
    /// (`flex_timepoint_eta_chi`), and the density normalization
    /// `D = Σ_cells flex_timepoint_d_cell` — instead of the hand
    /// `compute_survival_timepoint_exact` θ-derivative assembly. The returned jets
    /// carry their exact first/second θ-derivatives so the value/gradient/Hessian
    /// channels match the hand `eta_u`/`eta_uv`/`chi_*`/`d_*` term for term.
    ///
    /// A private free `fn` (no `&self` is needed — it consumes only the passed
    /// `primary`/`pack`/`rho`/`tau`/`cells`) so the in-`src` `#[cfg(test)]` gate can
    /// exercise it without a production caller (a `pub(crate)` item consumed only by
    /// masked test code trips the orphaned-`pub(crate)` ban). Promoted to a
    /// `pub(crate)` method at the production rewire (Phase D).
    ///
    /// `rho`/`tau` are the score-warp(`h`)/link-dev(`w`)/`infl` linear-channel scalar
    /// weights (the hand `rho`/`tau` vectors, first_full.rs:909-953); they enter
    /// `eta`/`chi` through their own primary axes. `o_infl` shifts `eta`'s value.
    /// The cells supply both the calibration residual (for the lift) and the `D`
    /// integral.
    fn flex_timepoint_inputs_jet2_impl(
        primary: &FlexPrimarySlices,
        q_index: usize,
        phi_q: f64,
        a0: f64,
        b: f64,
        d_check: f64,
        z_obs: f64,
        o_infl: f64,
        pack: &ObservedCoeffPack,
        channels: &FlexChannelInputs<'_>,
        cells: &[CalibrationCellJetInputs<'_>],
    ) -> Result<FlexTimepointJet2Out, String> {
        {
            let p = primary.total;
            let template = Jet2::from_parts(0.0, &vec![0.0; p], &[]);
            let b_jet = Jet2::primary(b, primary.g, p);
            let du: Vec<Jet2> = (0..p).map(|u| Jet2::primary(0.0, u, p)).collect();

            // Intercept lift to Jet2 (value/grad/Hess) — 2 Newton iterations. The
            // lift's slope jet carries the `g` primary, so the lifted `a_jet`'s grad
            // `a_u`/Hess `a_uv` include the full intercept dependence on `g`.
            let residual =
                |a: &Jet2| calibration_residual_jet(a, &b_jet, primary.g, &du, q_index, phi_q, cells);
            let a_jet = lift_intercept_flex(&template, a0, 1.0 / d_check, 2, residual);

            // (a,b)-coupled channel jets. The hand `compute_survival_timepoint_exact`
            // adds, on top of the pure-`a` chain (`chi·a_uv + eta_aa·a_u·a_v` /
            // `eta_aa·a_uv + eta_aaa·a_u·a_v`), the second-order CHANNEL coupling
            //   eta_uv += tau[u]·a_u[v] + tau[v]·a_u[u] + r_uv               (first_full.rs:972)
            //   chi_uv += tau_a[u]·a_u[v] + tau_a[v]·a_u[u] + chi_uv_fixed   (first_full.rs:980)
            // (`tau`/`tau_a` = `dc_dab`/`dc_daab` on `g`, `dc_aw`/`dc_aaw` on `w`;
            // `r_uv`/`chi_uv_fixed` the fixed `g×g`, `g×h`, `g×w` partials). A flat
            // linear `rho[idx]·primary` jet carries NONE of this, leaving the Jet2
            // Hessian wrong for every flex model with `h`/`w`/`g` primaries. So the
            // channel jets must carry their first-order `rho`/`tau` AND this exact
            // second-order content, built from the lifted `a_jet`'s gradient `a_u`.
            //
            // To avoid double-counting the `g`-axis, the observed-coeff bivariate
            // `(a,b)` composition runs against a CONSTANT slope jet (`db = 0`), so the
            // observed `eta`/`chi` carry only the pure-`a` chain; ALL of the `g`/`h`/
            // `w`/`infl` first- and second-order content is single-sourced through the
            // channel jets (the hand `rho`/`tau`/`tau_a` vectors carry `g` in slot
            // `primary.g`, mirrored here).
            let a_u = a_jet.g.clone();
            let rho_jet = channel_jet2(
                p,
                channels.rho,
                channels.tau,
                &a_u,
                channels.eta_fixed_uv,
            );
            let tau_jet = channel_jet2(
                p,
                channels.tau,
                channels.tau_a,
                &a_u,
                channels.chi_fixed_uv,
            );
            let b_jet_obs = const_jet_like(&template, b);

            let (eta, chi) =
                flex_timepoint_eta_chi(&a_jet, &b_jet_obs, z_obs, o_infl, pack, &rho_jet, &tau_jet);

            // D = Σ_cells INV_TWO_PI·Σ_k χ_k·M_k, χ from cell_chi_poly_jets, M from
            // the cell coeff jets through the lifted a_jet.
            let da = tangent_jet(&a_jet);
            let mut d = const_jet_like(&template, 0.0);
            for cell in cells {
                let c_pos = cell_coeff_jets(&a_jet, cell.base_pos_coeffs, cell.fixed, primary.g, &da, &du);
                let chi_jets = cell_chi_poly_jets(&a_jet, cell.fixed, primary.g, &da, &du);
                let edge_l = cell_edge_jet(&a_jet, &b_jet, cell.left_edge, cell.cell_left);
                let edge_r = cell_edge_jet(&a_jet, &b_jet, cell.right_edge, cell.cell_right);
                d = d.add(&flex_timepoint_d_cell(
                    &template,
                    &c_pos,
                    &chi_jets,
                    &edge_l,
                    cell.cell_left.is_finite(),
                    &edge_r,
                    cell.cell_right.is_finite(),
                    cell.numeric_moments,
                ));
            }

            let to_g = |j: &Jet2| Array1::from(j.g.clone());
            let to_h = |j: &Jet2| -> Result<Array2<f64>, String> {
                Array2::from_shape_vec((p, p), j.h.clone()).map_err(|e| e.to_string())
            };
            Ok(FlexTimepointJet2Out {
                eta: to_g(&eta),
                eta_v: eta.value(),
                eta_h: to_h(&eta)?,
                chi: to_g(&chi),
                chi_v: chi.value(),
                chi_h: to_h(&chi)?,
                d: to_g(&d),
                d_v: d.value(),
                d_h: to_h(&d)?,
            })
        }
    }

    /// The score-warp(`h`)/link-dev(`w`)/`g`/`infl` linear-channel inputs for
    /// [`flex_timepoint_inputs_jet2_impl`]: the hand `rho`/`tau`/`tau_a` first-order
    /// channel-weight vectors (`eval_coeff4_at(dc_db/dc_dab/dc_daab …)`,
    /// first_full.rs:909-953) and the fixed second-partial matrices `r_uv`
    /// (`observed_fixed_eta_second_partial`) / `chi_uv_fixed`
    /// (`observed_fixed_chi_second_partial`). The channel jets carry the EXACT
    /// second-order coupling `tau[u]·a_u[v]+tau[v]·a_u[u]+fixed_uv` so the Jet2
    /// Hessian matches the hand `eta_uv`/`chi_uv` term for term.
    struct FlexChannelInputs<'a> {
        rho: &'a [f64],
        tau: &'a [f64],
        tau_a: &'a [f64],
        eta_fixed_uv: &'a Array2<f64>,
        chi_fixed_uv: &'a Array2<f64>,
    }

    /// A `Jet2` linear-channel jet: value 0, gradient `grad`, and the `(a,b)`-coupled
    /// Hessian `h[u,v] = cross[u]·a_u[v] + cross[v]·a_u[u] + fixed_uv[u,v]` — the
    /// exact second-order channel content the hand `compute_survival_timepoint_exact`
    /// adds to `eta_uv`/`chi_uv` (with `grad`/`cross` = `rho`/`tau` for the `eta`
    /// channel, `tau`/`tau_a` for the `chi` channel, and `a_u` the lifted intercept
    /// gradient). A flat first-order jet (the prior seeding) carries zero Hessian and
    /// is therefore wrong for any flex model with active `h`/`w`/`g` primaries.
    fn channel_jet2(
        p: usize,
        grad: &[f64],
        cross: &[f64],
        a_u: &[f64],
        fixed_uv: &Array2<f64>,
    ) -> Jet2 {
        let mut h = vec![0.0_f64; p * p];
        for u in 0..p {
            for v in 0..p {
                h[u * p + v] = cross[u] * a_u[v] + cross[v] * a_u[u] + fixed_uv[[u, v]];
            }
        }
        Jet2::from_parts(0.0, grad, &h)
    }

    /// The `Jet2` timepoint inputs `(eta, chi, d)` value/gradient/Hessian channels
    /// returned by [`flex_timepoint_inputs_jet2_impl`].
    struct FlexTimepointJet2Out {
        eta_v: f64,
        eta: Array1<f64>,
        eta_h: Array2<f64>,
        chi_v: f64,
        chi: Array1<f64>,
        chi_h: Array2<f64>,
        d_v: f64,
        d: Array1<f64>,
        d_h: Array2<f64>,
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
        add_const(x, -x.value())
    }

    /// A constant jet (value `v`, all derivative channels zero), shaped like
    /// `template` (so it carries the right runtime primary count).
    #[inline]
    fn const_jet_like<J: FlexJet>(template: &J, v: f64) -> J {
        add_const(&template.scale(0.0), v)
    }

    /// #932 item-2 Phase C STEP 2: the generic-order intercept Newton lift over a
    /// runtime `FlexJet` — the linchpin that produces the 3rd/4th intercept
    /// θ-derivatives the base Hessian lacks. Mirrors `lift_intercept_order2` /
    /// `filtered_implicit_solve_scalar` but over a runtime `Jet2`/`Jet3`/`Jet4`.
    ///
    /// The calibration constraint `F(a(θ), θ) = 0` is solved by the filtered Newton
    /// step `A ← A − R(A)·inv_fa` (`inv_fa = 1/D`, `D = |F_a|`), iterated `iters`
    /// times (the jet nilpotency order). `R(A)` is the calibration RESIDUAL JET built
    /// by the caller-supplied `residual` closure from the per-cell coefficient jets
    /// and moment jets:
    ///
    ///   R(A) = Σ_cells INV_TWO_PI · Σ_k tangent(c_posₖ(A)) · Mₖ(A)   (+ q self-term)
    ///
    /// where `c_posₖ(A)` are the POSITIVE cell coefficients as jets in `A` (and the
    /// primaries) and `Mₖ(A)` the cell's normalization moment jets. This is the EXACT
    /// calibration θ-jet to all orders: `∂_θ R = INV_TWO_PI ∫ η_θ e^{−q} = −f_u`,
    /// `∂²_θ R = INV_TWO_PI ∫ (η_θθ − η η_θ²) e^{−q}` (the `−η η_θ²` falling out of
    /// `Mₖ`'s own `e^{−Δq}` motion `M_θ = −∫ η η_θ e^{−q}`), reproducing the hand
    /// `f_u`/`f_uv`/`f_aa` moment dots and their 3rd/4th extensions automatically. The
    /// value channel is the scalar calibration `f` (driven to ~0 by seeding
    /// `A.value = a0` from the scalar solve), so only the derivative channels solve.
    ///
    /// `template` carries the runtime primary count; `a0` the solved intercept value.
    fn lift_intercept_flex<J: FlexJet>(
        template: &J,
        a0: f64,
        inv_fa: f64,
        iters: usize,
        residual: impl Fn(&J) -> J,
    ) -> J {
        let mut a = const_jet_like(template, a0);
        for _ in 0..iters {
            let r = residual(&a);
            a = a.sub(&r.scale(inv_fa));
        }
        a
    }

    /// The per-row calibration residual jet `R(A)` for [`lift_intercept_flex`],
    /// summed over a timepoint's cells: `Σ_cells INV_TWO_PI·Σ_k tangent(c_posₖ(A))·
    /// Mₖ(A)` plus the q-marginal self-term `−φ(q)` on the `q_index` primary (the
    /// `f_u[q_index] += φ(q)` boundary term of the calibration). The cells are
    /// supplied as `(base_pos_coeffs, fixed, edges, finiteness, numeric_moments)` so
    /// the coefficient jets and moment jets are rebuilt at the current iterate `A`.
    fn calibration_residual_jet<J: FlexJet>(
        a_jet: &J,
        b_jet: &J,
        g_axis: usize,
        du: &[J],
        q_index: usize,
        phi_q: f64,
        cells: &[CalibrationCellJetInputs<'_>],
    ) -> J {
        let da = tangent_jet(a_jet);
        let inv_two_pi = std::f64::consts::TAU.recip();
        let mut r = const_jet_like(a_jet, 0.0);
        for cell in cells {
            // Positive cell coefficients as jets in (A, primaries).
            let c_pos = cell_coeff_jets(a_jet, cell.base_pos_coeffs, cell.fixed, g_axis, &da, du);
            // Moving edge jets: Crossing edges move with A/b, Fixed edges are static.
            let edge_l = cell_edge_jet(a_jet, b_jet, cell.left_edge, cell.cell_left);
            let edge_r = cell_edge_jet(a_jet, b_jet, cell.right_edge, cell.cell_right);
            let m = base_moment_jets(
                &c_pos,
                &edge_l,
                cell.cell_left.is_finite(),
                &edge_r,
                cell.cell_right.is_finite(),
                cell.numeric_moments,
            );
            // Σ_k tangent(c_posₖ)·Mₖ  (tangent strips the value: F's VALUE is carried
            // by the scalar seed, only the θ-motion of η enters the residual deriv).
            let mut cell_r = const_jet_like(a_jet, 0.0);
            for k in 0..4 {
                cell_r = cell_r.add(&tangent_jet(&c_pos[k]).mul(&m[k]));
            }
            r = r.add(&cell_r.scale(inv_two_pi));
        }
        // q-marginal self-term: f_u[q_index] += φ(q) ⟹ R gains −φ(q)·(q-axis tangent).
        // (R = −F, and the hand adds +φ(q) to f_u[q_index].)
        if q_index < du.len() {
            r = r.sub(&du[q_index].scale(phi_q));
        }
        r
    }

    /// Per-cell inputs for [`calibration_residual_jet`]: the positive base
    /// coefficients, the fixed-partial pack, the cell edges (location + provenance),
    /// and the numeric moment vector. Borrowed from the cached partition.
    struct CalibrationCellJetInputs<'a> {
        base_pos_coeffs: [f64; 4],
        fixed: &'a DenestedCellPrimaryFixedPartials,
        cell_left: f64,
        cell_right: f64,
        left_edge: crate::families::cubic_cell_kernel::PartitionEdge,
        right_edge: crate::families::cubic_cell_kernel::PartitionEdge,
        numeric_moments: &'a [f64],
    }

    /// The moving cell-edge `z` as a jet: a `Crossing { tau }` edge sits at
    /// `z = (τ − a)/b` and moves with the intercept jet `a_jet` and slope jet
    /// `b_jet`; a `Fixed(z)` edge is static (a constant jet, no θ-motion).
    fn cell_edge_jet<J: FlexJet>(
        a_jet: &J,
        b_jet: &J,
        edge: crate::families::cubic_cell_kernel::PartitionEdge,
        z_value: f64,
    ) -> J {
        match edge {
            crate::families::cubic_cell_kernel::PartitionEdge::Crossing { tau } => {
                // z = (τ − a)·(1/b).
                const_jet_like(a_jet, tau).sub(a_jet).mul(&recip(b_jet))
            }
            crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => {
                const_jet_like(a_jet, z_value)
            }
        }
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

    /// The per-cell de-nested coefficient `c_k` (k = 0..4) as a jet, built from the
    /// cell's `DenestedCellPrimaryFixedPartials` pack composed with the intercept
    /// perturbation `da = tangent(a_jet)` and the per-primary perturbations
    /// `du[u] = tangent(primary_u)`. This is the cell analogue of
    /// `observed_coeff_component_jet`, carrying ALL primaries (not just a,b): it is
    /// the multivariate Taylor of `c_k(a, {θ_u})` whose cross-partials are the pack
    /// fields. Matches the hand `eta_u_poly`/`eta_uv_poly`/`chi_*` assembly in
    /// `first_full`/`directional`/`bidirectional` term for term — the FlexJet algebra
    /// raises it to the contracted third/fourth automatically.
    ///
    /// Taylor structure (per k, `g_axis` = the slope `b` primary):
    ///   c = c0
    ///     + dc_da·da + ½dc_daa·da² + ⅙dc_daaa·da³                       (pure a)
    ///     + Σ_u coeff_u[u]·du                                           (pure u, lin)
    ///     + Σ_u coeff_au[u]·da·du + ½Σ_u coeff_aau[u]·da²·du            (a×u)
    ///     + Σ_u coeff_bu[u]·db·du + Σ_u coeff_abu[u]·da·db·du           (b×u)
    ///       + ½Σ_u coeff_bbu[u]·db²·du
    ///     + ⅙Σ_u coeff_aaau[u]·da³·du + ½Σ_u coeff_aabu[u]·da²·db·du    (3rd in a/b ×u)
    ///       + ½Σ_u coeff_abbu[u]·da·db²·du + ⅙Σ_u coeff_bbbu[u]·db³·du
    /// where `db = du[g_axis]` (the slope perturbation). The `coeff_u`-family terms
    /// are LINEAR in `du` (each cell coefficient is at most linear in any single
    /// non-a/non-b primary), so no `du²` term is needed beyond the b-channel ones.
    fn cell_coeff_jets<J: FlexJet>(
        template: &J,
        base_c: [f64; 4],
        fixed: &DenestedCellPrimaryFixedPartials,
        g_axis: usize,
        da: &J,
        du: &[J],
    ) -> [J; 4] {
        let p = du.len();
        let dada = da.mul(da);
        let dadada = dada.mul(da);
        let db = &du[g_axis];
        let dadb = da.mul(db);
        let dbdb = db.mul(db);
        std::array::from_fn(|k| {
            let mut c = const_jet_like(template, base_c[k]);
            // Pure-a chain.
            c = c
                .add(&da.scale(fixed.dc_da[k]))
                .add(&dada.scale(0.5 * fixed.dc_daa[k]))
                .add(&dadada.scale(fixed.dc_daaa[k] / 6.0));
            // Per-primary chains (linear in du[u]).
            for u in 0..p {
                let duu = &du[u];
                let mut chain = duu.scale(fixed.coeff_u[u][k]);
                chain = chain
                    .add(&da.mul(duu).scale(fixed.coeff_au[u][k]))
                    .add(&dada.mul(duu).scale(0.5 * fixed.coeff_aau[u][k]));
                chain = chain
                    .add(&db.mul(duu).scale(fixed.coeff_bu[u][k]))
                    .add(&dadb.mul(duu).scale(fixed.coeff_abu[u][k]))
                    .add(&dbdb.mul(duu).scale(0.5 * fixed.coeff_bbu[u][k]));
                chain = chain
                    .add(&dadada.mul(duu).scale(fixed.coeff_aaau[u][k] / 6.0))
                    .add(&dada.mul(db).mul(duu).scale(0.5 * fixed.coeff_aabu[u][k]))
                    .add(&dadb.mul(db).mul(duu).scale(0.5 * fixed.coeff_abbu[u][k]))
                    .add(&dbdb.mul(db).mul(duu).scale(fixed.coeff_bbbu[u][k] / 6.0));
                c = c.add(&chain);
            }
            c
        })
    }

    /// The per-cell `χ = ∂η/∂a` polynomial coefficients `dc_da[k]` (k = 0..4) as
    /// jets, the `∂_a`-shifted analogue of [`cell_coeff_jets`]: the cell coefficient
    /// family whose base is `dc_da`, whose `a`-derivatives are `dc_daa`/`dc_daaa`,
    /// whose per-primary derivatives are `coeff_au`/`coeff_aau` (= `∂(dc_da)/∂u` and
    /// `∂²(dc_da)/∂a∂u`), and whose `b`-cross is `coeff_abu` (= `∂²(dc_da)/∂b∂u`).
    /// These are the `χ_u`/`χ_uv` chains the hand `first_full` assembles by hand
    /// (`chi_u_poly = dc_daa·a_u + coeff_au`); the FlexJet algebra raises them.
    fn cell_chi_poly_jets<J: FlexJet>(
        template: &J,
        fixed: &DenestedCellPrimaryFixedPartials,
        g_axis: usize,
        da: &J,
        du: &[J],
    ) -> [J; 4] {
        let p = du.len();
        let dada = da.mul(da);
        let db = &du[g_axis];
        std::array::from_fn(|k| {
            // Base = dc_da; a-chain = dc_daa·da + ½dc_daaa·da².
            let mut c = const_jet_like(template, fixed.dc_da[k]);
            c = c
                .add(&da.scale(fixed.dc_daa[k]))
                .add(&dada.scale(0.5 * fixed.dc_daaa[k]));
            for u in 0..p {
                let duu = &du[u];
                // χ per-primary: coeff_au[u]·du + coeff_aau[u]·da·du + coeff_abu[u]·db·du.
                let chain = duu
                    .scale(fixed.coeff_au[u][k])
                    .add(&da.mul(duu).scale(fixed.coeff_aau[u][k]))
                    .add(&db.mul(duu).scale(fixed.coeff_abu[u][k]));
                c = c.add(&chain);
            }
            c
        })
    }

    /// #932 item-2 Phase C: the per-row density normalization `D = Σ_cells ∫ G0 dz`
    /// (`G0 = χ·w`, `w = e^{−q}/2π`) as a jet at any `FlexJet` order, carrying its
    /// exact θ-derivatives (the hand D-path `d_u`/`d_uv` are this jet's grad/Hess).
    ///
    /// Per cell `D_cell = INV_TWO_PI · Σ_k χ_k · M_k`, where `χ_k` are the cell's
    /// `dc_da` polynomial coefficients as jets ([`cell_chi_poly_jets`]) and `M_k` are
    /// the cell's normalization moments as jets ([`base_moment_jets`], carrying both
    /// the coefficient motion and the moving-edge sliver). The single-source magic:
    /// the hand path forms `d_u` by EXPLICITLY assembling `χ_u − χ·η·η_u` + boundary
    /// flux; the jet product `χ_k·M_k` reproduces all three terms automatically —
    /// `χ_u` from `χ_k`'s motion, `−χ·η·η_u` from `M_k`'s interior `e^{−Δq}` factor
    /// (`∂M_k = −Σ_m(η∂η)_m M_{k+m}`), and the boundary flux from `M_k`'s edge
    /// sliver. `c_jets` are the cell's `c0..c3` jets ([`cell_coeff_jets`]) feeding the
    /// moment exponent; `edge_l`/`edge_r` the moving edge jets; `moments` the cell's
    /// NUMERIC moment vector (≥ `4 + 6` entries for the `e^{−Δq}` expansion).
    fn flex_timepoint_d_cell<J: FlexJet>(
        template: &J,
        c_jets: &[J; 4],
        chi_jets: &[J; 4],
        edge_l: &J,
        left_finite: bool,
        edge_r: &J,
        right_finite: bool,
        numeric_moments: &[f64],
    ) -> J {
        let m = base_moment_jets(c_jets, edge_l, left_finite, edge_r, right_finite, numeric_moments);
        let mut acc = const_jet_like(template, 0.0);
        for (k, chi_k) in chi_jets.iter().enumerate() {
            acc = acc.add(&chi_k.mul(&m[k]));
        }
        acc.scale(std::f64::consts::TAU.recip())
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

    /// The observed cell-coefficient partial pack (`coeff`/`dc_d{a,b}…/dbbb`) passed
    /// through `observed_denested_cell_partials`, bundled so the generic eta/chi
    /// builder stays under the argument-count gate.
    struct ObservedCoeffPack {
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
    }

    /// Phase C-complete (generic order): the observed timepoint `eta` and `chi` as
    /// jets at ANY `FlexJet` order, from the intercept jet `a_jet` (carrying its
    /// θ-derivatives to that order) and the slope jet `b_jet`, the observed
    /// cell-coefficient pack, and pre-built score-warp(`h`)/link-dev(`w`) `rho`/`tau`
    /// channel jets. `eta`/`chi` carry their exact θ-derivatives by composing the
    /// coefficients' bivariate `(a,b)` Taylor with the intercept/slope jets, then
    /// adding the linear `h`/`w`/`infl` channels — replacing the hand
    /// `eta_u = chi·a_u + rho`, `eta_uv = …` chains in
    /// `first_full`/`directional`/`bidirectional`.
    ///
    /// `rho_jet`/`tau_jet` are the already-seeded jets carrying the linear `h`/`w`/
    /// `infl` channels' θ-dependence on their own primaries (the caller builds them
    /// at the correct order with the right directional seeds — order-specific
    /// seeding context lives at the call site, not here). `eta += rho_jet`,
    /// `chi += tau_jet`.
    fn flex_timepoint_eta_chi<J: FlexJet>(
        a_jet: &J,
        b_jet: &J,
        z_obs: f64,
        o_infl: f64,
        pack: &ObservedCoeffPack,
        rho_jet: &J,
        tau_jet: &J,
    ) -> (J, J) {
        let da = tangent_jet(a_jet);
        let db = tangent_jet(b_jet);
        let zero4 = [0.0_f64; 4];

        // eta coefficients: the coeff pack composed with (da, db).
        let coeff_jets: [J; 4] = std::array::from_fn(|k| {
            observed_coeff_component_jet(
                a_jet, k, pack.coeff, pack.dc_da, pack.dc_db, pack.dc_daa, pack.dc_dab, pack.dc_dbb,
                pack.dc_daaa, pack.dc_daab, pack.dc_dabb, pack.dc_dbbb, &da, &db,
            )
        });
        let eta = add_const(&eval_coeff_jet_at(&coeff_jets, z_obs), o_infl).add(rho_jet);

        // chi = ∂eta/∂a coefficients = the dc_da pack, whose own (a,b)-Taylor is the
        // once-`a`-shifted pack (dc_daa as ∂/∂a, dc_dab as ∂/∂b, dc_daaa/daab/dabb as
        // the seconds; the dc_da pack carries no third-order term, so those are 0).
        let chi_jets: [J; 4] = std::array::from_fn(|k| {
            observed_coeff_component_jet(
                a_jet, k, pack.dc_da, pack.dc_daa, pack.dc_dab, pack.dc_daaa, pack.dc_daab,
                pack.dc_dabb, zero4, zero4, zero4, zero4, &da, &db,
            )
        });
        let chi = eval_coeff_jet_at(&chi_jets, z_obs).add(tau_jet);

        (eta, chi)
    }


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
    /// `M_0..M_4` (interior `Σ_m S_m M_{n+m}` + moving-edge sliver flux) against a
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

    /// #932 item-2 Phase B-base closure: the SECOND θ-derivative (the self-
    /// consistent `e^{−Δq}` interior `(∂q)²` cross-term + the second-order moving-
    /// edge sliver) must match a central finite difference of the analytic FIRST
    /// derivative. Probes the `Jet2` Hessian channel `h[0]` (= `d²M_n/dθ²`) of
    /// `base_moment_jets`, the all-orders exactness the Jet3/Jet4 contractions
    /// depend on.
    #[test]
    fn base_moment_jets_second_derivative_matches_fd_932() {
        use crate::families::cubic_cell_kernel::evaluate_cell_moments;

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
        let max_degree = 12usize;
        let moments_at = |theta: f64| -> Vec<f64> {
            evaluate_cell_moments(cell_at(theta), max_degree)
                .expect("numeric cell moments")
                .moments
                .into_vec()
        };
        // Analytic first derivative dM_n/dθ from base_moment_jets at parameter θ.
        let analytic_first = |theta: f64, n: usize| -> f64 {
            let numeric = moments_at(theta);
            let seeded = |x: f64, vel: f64| {
                let g = vec![vel];
                Jet2::from_parts(x, &g, &[])
            };
            let cell = cell_at(theta);
            let c_jets = [
                seeded(cell.c0, dc[0]),
                seeded(cell.c1, dc[1]),
                seeded(cell.c2, dc[2]),
                seeded(cell.c3, dc[3]),
            ];
            let zl_jet = seeded(cell.left, v_l);
            let zr_jet = seeded(cell.right, v_r);
            let m = base_moment_jets(&c_jets, &zl_jet, true, &zr_jet, true, &numeric);
            m[n].g[0]
        };

        // Analytic second derivative from the Jet2 Hessian channel at θ=0.
        let numeric0 = moments_at(0.0);
        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
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

        let h = 1e-5_f64;
        for n in 0..5 {
            let fd2 = (analytic_first(h, n) - analytic_first(-h, n)) / (2.0 * h);
            let hess = m_jets[n].h[0];
            assert!(
                (hess - fd2).abs() <= 2e-4 * (1.0 + fd2.abs()),
                "M_{n} d²θ analytic {} != FD-of-analytic {}",
                hess,
                fd2
            );
        }
    }

    /// #932 item-2 Phase C: the generic `flex_timepoint_eta_chi<J>` builder must
    /// reproduce `eta = eval_coeff4_at(coeff, z) + o_infl + rho` and `chi =
    /// eval_coeff4_at(dc_da, z) + tau` on the VALUE channel, and the a/b θ-motion
    /// on the gradient channel vs a central finite difference of the same scalar
    /// expression along a smooth `(a,b)` family. Pins the bivariate-Taylor compose
    /// + the linear rho/tau add at `Jet2`.
    #[test]
    fn flex_timepoint_eta_chi_value_and_grad_932() {
        let z_obs = 0.7_f64;
        let o_infl = 0.05_f64;
        let pack = ObservedCoeffPack {
            coeff: [0.2, -0.3, 0.15, 0.05],
            dc_da: [1.1, 0.2, 0.03, 0.0],
            dc_db: [0.4, 1.05, 0.1, 0.02],
            dc_daa: [0.07, 0.02, 0.0, 0.0],
            dc_dab: [0.2, 0.09, 0.01, 0.0],
            dc_dbb: [0.11, 0.04, 0.005, 0.0],
            dc_daaa: [0.003, 0.0, 0.0, 0.0],
            dc_daab: [0.006, 0.001, 0.0, 0.0],
            dc_dabb: [0.004, 0.002, 0.0, 0.0],
            dc_dbbb: [0.008, 0.001, 0.0, 0.0],
        };
        // Single-primary family θ: a(θ)=a0+θ·a_u, b(θ)=b0+θ·b_u.
        let a0 = 0.3_f64;
        let b0 = 1.2_f64;
        let a_u = 0.25_f64;
        let b_u = -0.4_f64;
        let p = 1usize;
        let a_jet = Jet2::from_parts(a0, &[a_u], &[]);
        let b_jet = Jet2::from_parts(b0, &[b_u], &[]);
        // No rho/tau channels for this probe.
        let zero = Jet2::from_parts(0.0, &vec![0.0; p], &[]);
        let (eta, chi) = flex_timepoint_eta_chi(&a_jet, &b_jet, z_obs, o_infl, &pack, &zero, &zero);

        // Scalar reference: eta(a,b) = Σ_k c_k(a,b) z^k, c_k composed from the
        // bivariate Taylor of the pack about (a0,b0).
        let coeff_scalar = |da: f64, db: f64| -> [f64; 4] {
            std::array::from_fn(|k| {
                pack.coeff[k]
                    + pack.dc_da[k] * da
                    + pack.dc_db[k] * db
                    + 0.5 * pack.dc_daa[k] * da * da
                    + pack.dc_dab[k] * da * db
                    + 0.5 * pack.dc_dbb[k] * db * db
                    + pack.dc_daaa[k] * da * da * da / 6.0
                    + 0.5 * pack.dc_daab[k] * da * da * db
                    + 0.5 * pack.dc_dabb[k] * da * db * db
                    + pack.dc_dbbb[k] * db * db * db / 6.0
            })
        };
        let eta_scalar = |theta: f64| -> f64 {
            let c = coeff_scalar(a_u * theta, b_u * theta);
            eval_coeff4_scalar(&c, z_obs) + o_infl
        };
        let chi_scalar = |theta: f64| -> f64 {
            let dc = coeff_scalar_da(&pack, a_u * theta, b_u * theta);
            eval_coeff4_scalar(&dc, z_obs)
        };
        assert!(
            (eta.value() - eta_scalar(0.0)).abs() <= 1e-12 * (1.0 + eta_scalar(0.0).abs()),
            "eta value {} != {}",
            eta.value(),
            eta_scalar(0.0)
        );
        assert!(
            (chi.value() - chi_scalar(0.0)).abs() <= 1e-12 * (1.0 + chi_scalar(0.0).abs()),
            "chi value {} != {}",
            chi.value(),
            chi_scalar(0.0)
        );
        let h = 1e-6_f64;
        let eta_fd = (eta_scalar(h) - eta_scalar(-h)) / (2.0 * h);
        let chi_fd = (chi_scalar(h) - chi_scalar(-h)) / (2.0 * h);
        assert!(
            (eta.g[0] - eta_fd).abs() <= 1e-5 * (1.0 + eta_fd.abs()),
            "eta grad {} != FD {}",
            eta.g[0],
            eta_fd
        );
        assert!(
            (chi.g[0] - chi_fd).abs() <= 1e-5 * (1.0 + chi_fd.abs()),
            "chi grad {} != FD {}",
            chi.g[0],
            chi_fd
        );
    }

    /// Scalar Horner `Σ_k c[k] z^k` (the `f64` image of `eval_coeff4_at`).
    fn eval_coeff4_scalar(c: &[f64; 4], z: f64) -> f64 {
        let mut acc = 0.0;
        for &ck in c.iter().rev() {
            acc = acc * z + ck;
        }
        acc
    }

    /// Scalar `∂_a coeff` Taylor (the dc_da pack composed about (a0,b0)) — the
    /// reference for `chi`'s value/grad in the test above.
    fn coeff_scalar_da(pack: &ObservedCoeffPack, da: f64, db: f64) -> [f64; 4] {
        std::array::from_fn(|k| {
            pack.dc_da[k]
                + pack.dc_daa[k] * da
                + pack.dc_dab[k] * db
                + 0.5 * pack.dc_daaa[k] * da * da
                + pack.dc_daab[k] * da * db
                + 0.5 * pack.dc_dabb[k] * db * db
        })
    }

    /// #932 item-2 Phase C: `cell_coeff_jets` must reproduce the hand cell-coeff
    /// total θ-derivatives `c_k(a(θ), θ)` value + gradient (the `eta_u_poly =
    /// dc_da·a_u + coeff_u` / `coeff_bu·db` structure) vs a central FD of the
    /// scalar multivariate Taylor along a smooth `(a, {θ_u})` family. p=3 with the
    /// g-slope at axis 1.
    #[test]
    fn cell_coeff_jets_value_and_grad_932() {
        let p = 3usize;
        let g_axis = 1usize;
        let base_c = [0.2_f64, -0.3, 0.15, 0.05];
        // Minimal but fully-populated fixed pack (per-primary length p).
        let mk_run = |seed: f64| -> Vec<[f64; 4]> {
            (0..p)
                .map(|u| std::array::from_fn(|k| seed * (1.0 + u as f64) * (1.0 + k as f64) * 0.01))
                .collect()
        };
        let fixed = DenestedCellPrimaryFixedPartials {
            dc_da: [1.1, 0.2, 0.03, 0.0],
            dc_daa: [0.07, 0.02, 0.0, 0.0],
            dc_daaa: [0.003, 0.0, 0.0, 0.0],
            coeff_u: mk_run(0.9),
            coeff_au: mk_run(0.4),
            coeff_bu: mk_run(0.5),
            coeff_aau: mk_run(0.12),
            coeff_abu: mk_run(0.16),
            coeff_bbu: mk_run(0.11),
            coeff_aaau: mk_run(0.02),
            coeff_aabu: mk_run(0.03),
            coeff_abbu: mk_run(0.04),
            coeff_bbbu: mk_run(0.05),
        };
        // Smooth family θ: a(θ)=a0+θ·a_u, θ_u(θ)=u0[u]+θ·v[u].
        let a0 = 0.3_f64;
        let a_u = 0.25_f64;
        let v = [0.2_f64, -0.4, 0.33];

        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
            Jet2::from_parts(x, &g, &[])
        };
        let a_jet = seeded(a0, a_u);
        let da = tangent_jet(&a_jet);
        // du[u] = value-0 tangent carrying velocity v[u] in slot 0.
        let du: Vec<Jet2> = (0..p).map(|u| seeded(0.0, v[u])).collect();
        let jets = cell_coeff_jets(&a_jet, base_c, &fixed, g_axis, &da, &du);

        // Scalar reference: c_k(da, {du}) via the same Taylor.
        let scalar_c = |theta: f64| -> [f64; 4] {
            let da = a_u * theta;
            let db = v[g_axis] * theta;
            std::array::from_fn(|k| {
                let mut acc = base_c[k]
                    + fixed.dc_da[k] * da
                    + 0.5 * fixed.dc_daa[k] * da * da
                    + fixed.dc_daaa[k] * da * da * da / 6.0;
                for u in 0..p {
                    let duu = v[u] * theta;
                    acc += fixed.coeff_u[u][k] * duu
                        + fixed.coeff_au[u][k] * da * duu
                        + 0.5 * fixed.coeff_aau[u][k] * da * da * duu
                        + fixed.coeff_bu[u][k] * db * duu
                        + fixed.coeff_abu[u][k] * da * db * duu
                        + 0.5 * fixed.coeff_bbu[u][k] * db * db * duu
                        + fixed.coeff_aaau[u][k] * da * da * da * duu / 6.0
                        + 0.5 * fixed.coeff_aabu[u][k] * da * da * db * duu
                        + 0.5 * fixed.coeff_abbu[u][k] * da * db * db * duu
                        + fixed.coeff_bbbu[u][k] * db * db * db * duu / 6.0;
                }
                acc
            })
        };
        let h = 1e-6_f64;
        let c0 = scalar_c(0.0);
        let cp = scalar_c(h);
        let cm = scalar_c(-h);
        for k in 0..4 {
            assert!(
                (jets[k].value() - c0[k]).abs() <= 1e-12 * (1.0 + c0[k].abs()),
                "c_{k} value {} != {}",
                jets[k].value(),
                c0[k]
            );
            let fd = (cp[k] - cm[k]) / (2.0 * h);
            assert!(
                (jets[k].g[0] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "c_{k} grad {} != FD {}",
                jets[k].g[0],
                fd
            );
        }
    }

    /// #932 item-2 Phase C STEP 1: `flex_timepoint_d_cell` (the per-cell density
    /// normalization `D_cell = INV_TWO_PI·Σ_k χ_k·M_k`) must reproduce the value
    /// `INV_TWO_PI·Σ_k dc_da[k]·M_k` and the θ-gradient `d_u` (the hand
    /// `survival_flex_base_d_u` quantity) vs a central FD of the same scalar `D(θ)`
    /// on a smooth intercept-only family `a(θ)=a0+θ` with FIXED edges (isolates
    /// the interior coefficient motion + the M_k `e^{−Δq}` `−χηη_u` term — the edge
    /// flux is exercised by the moving-edge `base_moment_jets` tests). The cell's
    /// `c_k` and `dc_da_k` move with `a` per the chosen `dc_daa`/`dc_daaa` pack.
    #[test]
    fn flex_timepoint_d_cell_value_and_grad_932() {
        use crate::families::cubic_cell_kernel::evaluate_cell_moments;

        let zl = -1.1_f64;
        let zr = 1.6_f64;
        // a-only family: c_k and dc_da_k are cubic/quadratic in θ via the pack.
        let c_base = [0.2_f64, -0.3, 0.18, 0.06];
        let dc_da = [1.05_f64, 0.22, 0.04, 0.0];
        let dc_daa = [0.08_f64, 0.03, 0.0, 0.0];
        let dc_daaa = [0.004_f64, 0.0, 0.0, 0.0];
        let cell_at = |theta: f64| {
            let c: [f64; 4] = std::array::from_fn(|k| {
                c_base[k]
                    + dc_da[k] * theta
                    + 0.5 * dc_daa[k] * theta * theta
                    + dc_daaa[k] * theta * theta * theta / 6.0
            });
            DenestedCubicCell {
                left: zl,
                right: zr,
                c0: c[0],
                c1: c[1],
                c2: c[2],
                c3: c[3],
            }
        };
        let dc_da_at = |theta: f64| -> [f64; 4] {
            std::array::from_fn(|k| {
                dc_da[k] + dc_daa[k] * theta + 0.5 * dc_daaa[k] * theta * theta
            })
        };
        let max_degree = 10usize;
        let moments_at = |theta: f64| -> Vec<f64> {
            evaluate_cell_moments(cell_at(theta), max_degree)
                .expect("numeric cell moments")
                .moments
                .into_vec()
        };
        let d_scalar = |theta: f64| -> f64 {
            let m = moments_at(theta);
            let chi = dc_da_at(theta);
            let mut acc = 0.0;
            for k in 0..4 {
                acc += chi[k] * m[k];
            }
            acc * std::f64::consts::TAU.recip()
        };

        // Jet build: single primary (slot 0) = the intercept axis, velocity 1.
        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
            Jet2::from_parts(x, &g, &[])
        };
        let cell0 = cell_at(0.0);
        let c_jets = [
            seeded(cell0.c0, dc_da[0]),
            seeded(cell0.c1, dc_da[1]),
            seeded(cell0.c2, dc_da[2]),
            seeded(cell0.c3, dc_da[3]),
        ];
        // chi_jets carry dc_da value + dc_daa motion (a-only family).
        let dc_da0 = dc_da_at(0.0);
        let chi_jets = [
            seeded(dc_da0[0], dc_daa[0]),
            seeded(dc_da0[1], dc_daa[1]),
            seeded(dc_da0[2], dc_daa[2]),
            seeded(dc_da0[3], dc_daa[3]),
        ];
        let template = seeded(0.0, 0.0);
        let edge_l = seeded(zl, 0.0); // fixed edge: no motion
        let edge_r = seeded(zr, 0.0);
        let numeric0 = moments_at(0.0);
        let d_jet = flex_timepoint_d_cell(
            &template, &c_jets, &chi_jets, &edge_l, true, &edge_r, true, &numeric0,
        );

        assert!(
            (d_jet.value() - d_scalar(0.0)).abs() <= 1e-10 * (1.0 + d_scalar(0.0).abs()),
            "D value {} != {}",
            d_jet.value(),
            d_scalar(0.0)
        );
        let h = 1e-6_f64;
        let fd = (d_scalar(h) - d_scalar(-h)) / (2.0 * h);
        assert!(
            (d_jet.g[0] - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
            "D grad (d_u) {} != FD {}",
            d_jet.g[0],
            fd
        );
    }

    /// #932 item-2 Phase C: `cell_chi_poly_jets` (the `∂η/∂a = dc_da` family as
    /// jets) value channel == `dc_da[k]`, gradient channel == the hand
    /// `chi_u_poly = dc_daa·a_u + coeff_au` chain, vs a central FD of the scalar
    /// `dc_da_k(a(θ), {θ_u})` Taylor. p=2 with the g-slope at axis 1.
    #[test]
    fn cell_chi_poly_jets_value_and_grad_932() {
        let p = 2usize;
        let g_axis = 1usize;
        let mk_run = |seed: f64| -> Vec<[f64; 4]> {
            (0..p)
                .map(|u| std::array::from_fn(|k| seed * (1.0 + u as f64) * (1.0 + k as f64) * 0.01))
                .collect()
        };
        let fixed = DenestedCellPrimaryFixedPartials {
            dc_da: [1.05, 0.22, 0.04, 0.0],
            dc_daa: [0.08, 0.03, 0.0, 0.0],
            dc_daaa: [0.004, 0.0, 0.0, 0.0],
            coeff_u: mk_run(0.9),
            coeff_au: mk_run(0.4),
            coeff_bu: mk_run(0.5),
            coeff_aau: mk_run(0.12),
            coeff_abu: mk_run(0.16),
            coeff_bbu: mk_run(0.11),
            coeff_aaau: mk_run(0.02),
            coeff_aabu: mk_run(0.03),
            coeff_abbu: mk_run(0.04),
            coeff_bbbu: mk_run(0.05),
        };
        let a_u = 0.25_f64;
        let v = [0.2_f64, -0.4];
        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
            Jet2::from_parts(x, &g, &[])
        };
        let a_jet = seeded(0.3, a_u);
        let da = tangent_jet(&a_jet);
        let du: Vec<Jet2> = (0..p).map(|u| seeded(0.0, v[u])).collect();
        let chi = cell_chi_poly_jets(&a_jet, &fixed, g_axis, &da, &du);

        let chi_scalar = |theta: f64| -> [f64; 4] {
            let da = a_u * theta;
            let db = v[g_axis] * theta;
            std::array::from_fn(|k| {
                let mut acc =
                    fixed.dc_da[k] + fixed.dc_daa[k] * da + 0.5 * fixed.dc_daaa[k] * da * da;
                for u in 0..p {
                    let duu = v[u] * theta;
                    acc += fixed.coeff_au[u][k] * duu
                        + fixed.coeff_aau[u][k] * da * duu
                        + fixed.coeff_abu[u][k] * db * duu;
                }
                acc
            })
        };
        let h = 1e-6_f64;
        let c0 = chi_scalar(0.0);
        let cp = chi_scalar(h);
        let cm = chi_scalar(-h);
        for k in 0..4 {
            assert!(
                (chi[k].value() - c0[k]).abs() <= 1e-12 * (1.0 + c0[k].abs()),
                "chi_{k} value {} != {}",
                chi[k].value(),
                c0[k]
            );
            let fd = (cp[k] - cm[k]) / (2.0 * h);
            assert!(
                (chi[k].g[0] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "chi_{k} grad {} != FD {}",
                chi[k].g[0],
                fd
            );
        }
    }

    /// #932 item-2 Phase C STEP 2: the calibration residual jet's gradient
    /// channel must equal `−f_u` (the hand `cell_first_derivative_from_moments`
    /// of `−coeff_u`), and `lift_intercept_flex` must recover the hand IFT
    /// `a_u = f_u/D` on a synthetic single-cell single-primary calibration. This
    /// pins the core derivation `∂_θ R = INV_TWO_PI ∫ η_θ e^{−q} = −f_u` and the
    /// Newton lift's first-order channel against the validated kernel.
    #[test]
    fn lift_intercept_flex_first_order_matches_hand_ift_932() {
        use crate::families::cubic_cell_kernel::{
            cell_first_derivative_from_moments, evaluate_cell_moments, DenestedCubicCell,
            PartitionEdge,
        };

        // Synthetic finite cell with real moments (degree 9, as the cached
        // partition requests). Coefficients are the POSITIVE-cell c0..c3.
        let cell = DenestedCubicCell {
            left: -1.0,
            right: 1.4,
            c0: 0.2,
            c1: -0.25,
            c2: 0.15,
            c3: 0.05,
        };
        let numeric = evaluate_cell_moments(cell, 9)
            .expect("numeric moments")
            .moments
            .into_vec();
        let base_pos = [cell.c0, cell.c1, cell.c2, cell.c3];
        // fixed pack: ∂c/∂a = dc_da; one primary u=0 with ∂c/∂θ0 = coeff_u[0].
        let p = 1usize;
        let dc_da = [0.9_f64, 0.2, 0.05, 0.0];
        let coeff_u0 = [0.3_f64, -0.15, 0.08, 0.0];
        let zero_run: Vec<[f64; 4]> = vec![[0.0; 4]; p];
        let mut coeff_u = zero_run.clone();
        coeff_u[0] = coeff_u0;
        let fixed = DenestedCellPrimaryFixedPartials {
            dc_da,
            dc_daa: [0.0; 4],
            dc_daaa: [0.0; 4],
            coeff_u,
            coeff_au: zero_run.clone(),
            coeff_bu: zero_run.clone(),
            coeff_aau: zero_run.clone(),
            coeff_abu: zero_run.clone(),
            coeff_bbu: zero_run.clone(),
            coeff_aaau: zero_run.clone(),
            coeff_aabu: zero_run.clone(),
            coeff_abbu: zero_run.clone(),
            coeff_bbbu: zero_run,
        };

        // Hand calibration partials (no q self-term: phi_q = 0).
        // f_u[0] = ∫(−coeff_u0)·e^{−q}/2π ; f_a = ∫(−dc_da)·e^{−q}/2π = −D.
        let neg_coeff_u0 = coeff_u0.map(|v| -v);
        let neg_dc_da = dc_da.map(|v| -v);
        let f_u0 = cell_first_derivative_from_moments(&neg_coeff_u0, &numeric).expect("f_u");
        let f_a = cell_first_derivative_from_moments(&neg_dc_da, &numeric).expect("f_a");
        let d_check = f_a.abs();
        let a_u_hand = f_u0 / d_check;

        // Build the residual jet at A = const(a0) with the primary u=0 seeded.
        let a0 = 0.31_f64;
        let template = Jet2::from_parts(0.0, &vec![0.0; p], &[]);
        let a_jet0 = Jet2::from_parts(a0, &vec![0.0; p], &[]);
        let b_jet = Jet2::from_parts(1.1, &vec![0.0; p], &[]);
        let du: Vec<Jet2> = (0..p).map(|u| Jet2::primary(0.0, u, p)).collect();
        let cells = vec![CalibrationCellJetInputs {
            base_pos_coeffs: base_pos,
            fixed: &fixed,
            cell_left: cell.left,
            cell_right: cell.right,
            left_edge: PartitionEdge::Fixed(cell.left),
            right_edge: PartitionEdge::Fixed(cell.right),
            numeric_moments: &numeric,
        }];
        // Residual at the un-lifted A0 (a_jet has zero derivative channels): its
        // gradient is the DIRECT primary motion = −f_u (the η_θ0 term only).
        let r0 = calibration_residual_jet(&a_jet0, &b_jet, 0, &du, p, 0.0, &cells);
        assert!(
            (r0.g[0] - (-f_u0)).abs() <= 1e-9 * (1.0 + f_u0.abs()),
            "residual grad {} != -f_u {}",
            r0.g[0],
            -f_u0
        );

        // Lift: inv_fa = 1/D, but the residual's a-derivative sign — R_a = ∂R/∂a.
        // R = −F and F_a = −D ⟹ R_a = D, so inv_fa = 1/D drives A toward the IFT
        // root. Two Newton iterations at Jet2.
        let residual =
            |a: &Jet2| calibration_residual_jet(a, &b_jet, 0, &du, p, 0.0, &cells);
        let a_lift = lift_intercept_flex(&template, a0, 1.0 / d_check, 2, residual);
        assert!(
            (a_lift.g[0] - a_u_hand).abs() <= 1e-6 * (1.0 + a_u_hand.abs()),
            "lifted a_u {} != hand f_u/D {}",
            a_lift.g[0],
            a_u_hand
        );
    }

    /// #932 item-2 Phase C STEP 3: the `flex_timepoint_inputs_jet2_impl` assembly
    /// (intercept lift → observed eta/chi → Σ_cells D) composes correctly — the
    /// VALUE channels equal their scalar references (eta = eval_coeff4(coeff,z) +
    /// o_infl, chi = eval_coeff4(dc_da,z), D = INV_TWO_PI·Σ_k dc_da[k]·M_k) and the
    /// gradient/Hessian channels are finite — on a synthetic single-cell, single-
    /// real-cell setup. (The full channel-for-channel match vs the hand
    /// `compute_survival_timepoint_exact` is gated by a `tests/`-dir integration
    /// test once the production rewire gives a non-test caller.)
    #[test]
    fn flex_timepoint_inputs_jet2_assembly_composes_932() {
        use crate::families::cubic_cell_kernel::{
            cell_first_derivative_from_moments, evaluate_cell_moments, DenestedCubicCell,
            PartitionEdge,
        };

        // p=2 primaries: q at axis 0, g (slope) at axis 1. No h/w/infl.
        let p = 2usize;
        let primary = FlexPrimarySlices {
            q0: 0,
            q1: 0,
            qd1: 0,
            g: 1,
            h: None,
            w: None,
            infl: None,
            total: p,
        };
        let cell = DenestedCubicCell {
            left: -1.0,
            right: 1.4,
            c0: 0.2,
            c1: -0.25,
            c2: 0.15,
            c3: 0.05,
        };
        let numeric = evaluate_cell_moments(cell, 27)
            .expect("numeric moments")
            .moments
            .into_vec();
        let dc_da = [0.9_f64, 0.2, 0.05, 0.0];
        let zero_run: Vec<[f64; 4]> = vec![[0.0; 4]; p];
        let fixed = DenestedCellPrimaryFixedPartials {
            dc_da,
            dc_daa: [0.0; 4],
            dc_daaa: [0.0; 4],
            coeff_u: zero_run.clone(),
            coeff_au: zero_run.clone(),
            coeff_bu: zero_run.clone(),
            coeff_aau: zero_run.clone(),
            coeff_abu: zero_run.clone(),
            coeff_bbu: zero_run.clone(),
            coeff_aaau: zero_run.clone(),
            coeff_aabu: zero_run.clone(),
            coeff_abbu: zero_run.clone(),
            coeff_bbbu: zero_run,
        };
        let neg_dc_da = dc_da.map(|v| -v);
        let f_a = cell_first_derivative_from_moments(&neg_dc_da, &numeric).expect("f_a");
        let d_check = f_a.abs();

        let z_obs = 0.6_f64;
        let o_infl = 0.04_f64;
        let pack = ObservedCoeffPack {
            coeff: [0.2, -0.3, 0.15, 0.05],
            dc_da: [1.1, 0.2, 0.03, 0.0],
            dc_db: [0.4, 1.05, 0.1, 0.02],
            dc_daa: [0.07, 0.02, 0.0, 0.0],
            dc_dab: [0.2, 0.09, 0.01, 0.0],
            dc_dbb: [0.11, 0.04, 0.005, 0.0],
            dc_daaa: [0.003, 0.0, 0.0, 0.0],
            dc_daab: [0.006, 0.001, 0.0, 0.0],
            dc_dabb: [0.004, 0.002, 0.0, 0.0],
            dc_dbbb: [0.008, 0.001, 0.0, 0.0],
        };
        let rho = vec![0.0_f64; p];
        let tau = vec![0.0_f64; p];
        let tau_a = vec![0.0_f64; p];
        let eta_fixed_uv = Array2::<f64>::zeros((p, p));
        let chi_fixed_uv = Array2::<f64>::zeros((p, p));
        let channels = FlexChannelInputs {
            rho: &rho,
            tau: &tau,
            tau_a: &tau_a,
            eta_fixed_uv: &eta_fixed_uv,
            chi_fixed_uv: &chi_fixed_uv,
        };
        let cells = vec![CalibrationCellJetInputs {
            base_pos_coeffs: [cell.c0, cell.c1, cell.c2, cell.c3],
            fixed: &fixed,
            cell_left: cell.left,
            cell_right: cell.right,
            left_edge: PartitionEdge::Fixed(cell.left),
            right_edge: PartitionEdge::Fixed(cell.right),
            numeric_moments: &numeric,
        }];

        let out = flex_timepoint_inputs_jet2_impl(
            &primary, primary.q1, 0.0, 0.31, 1.1, d_check, z_obs, o_infl, &pack, &channels, &cells,
        )
        .expect("jet timepoint inputs");

        // Value references.
        let eta_ref = {
            let mut acc = 0.0;
            for &c in pack.coeff.iter().rev() {
                acc = acc * z_obs + c;
            }
            acc + o_infl
        };
        let chi_ref = {
            let mut acc = 0.0;
            for &c in pack.dc_da.iter().rev() {
                acc = acc * z_obs + c;
            }
            acc
        };
        let d_ref = {
            let mut acc = 0.0;
            for k in 0..4 {
                acc += dc_da[k] * numeric[k];
            }
            acc * std::f64::consts::TAU.recip()
        };
        assert!(
            (out.eta_v - eta_ref).abs() <= 1e-9 * (1.0 + eta_ref.abs()),
            "eta value {} != {}",
            out.eta_v,
            eta_ref
        );
        assert!(
            (out.chi_v - chi_ref).abs() <= 1e-9 * (1.0 + chi_ref.abs()),
            "chi value {} != {}",
            out.chi_v,
            chi_ref
        );
        assert!(
            (out.d_v - d_ref).abs() <= 1e-9 * (1.0 + d_ref.abs()),
            "d value {} != {}",
            out.d_v,
            d_ref
        );
        // Derivative channels finite + present.
        assert_eq!(out.eta.len(), p);
        for arr in [&out.eta, &out.chi, &out.d] {
            for v in arr.iter() {
                assert!(v.is_finite(), "gradient channel finite");
            }
        }
        for mat in [&out.eta_h, &out.chi_h, &out.d_h] {
            assert_eq!(mat.shape(), [p, p]);
            for v in mat.iter() {
                assert!(v.is_finite(), "Hessian channel finite");
            }
        }
    }

    /// #932 item-2 STEP 3b (the bug fix): the `Jet2` `eta_uv`/`chi_uv` Hessian must
    /// reproduce the hand `compute_survival_timepoint_exact` channel coupling EXACTLY
    /// (≤1e-9) — NOT just finite — for a flex model with active `h`/`w`/`g`
    /// primaries. The hand (first_full.rs:972-988) assembles
    ///
    ///   eta_uv[u,v] = chi·a_uv + eta_aa·a_u[u]·a_u[v]
    ///               + tau[u]·a_u[v] + tau[v]·a_u[u] + r_uv
    ///   chi_uv[u,v] = eta_aa·a_uv + eta_aaa·a_u[u]·a_u[v]
    ///               + tau_a[u]·a_u[v] + tau_a[v]·a_u[u] + chi_uv_fixed
    ///
    /// The PRIOR seeding made `rho`/`tau` flat first-order jets (zero Hessian),
    /// dropping the `tau·a_u` cross-terms and the fixed `r_uv`/`chi_uv_fixed` second
    /// partials — so the Jet2 Hessian was wrong for the normal flex config. This gate
    /// pins the full second-order channel content against the hand formula, evaluated
    /// from the jet's OWN lifted `a_jet` (grad `a_u`, Hess `a_uv`), `chi`/`eta_aa`/
    /// `eta_aaa` from the observed pack, and non-trivial `rho`/`tau`/`tau_a`/
    /// `r_uv`/`chi_uv_fixed` — exercising every term the bug omitted.
    #[test]
    fn flex_timepoint_inputs_jet2_hessian_matches_hand_channel_coupling_932() {
        use crate::families::cubic_cell_kernel::{
            cell_first_derivative_from_moments, evaluate_cell_moments, DenestedCubicCell,
            PartitionEdge,
        };

        // p=4: q at 0, g (slope) at 1, one h (score-warp) axis at 2, one w
        // (link-dev) axis at 3. The channel coupling is exercised on every
        // (u,v) pair including the cross axes g×h, g×w, h×w.
        let p = 4usize;
        let g_axis = 1usize;
        let h_axis = 2usize;
        let w_axis = 3usize;
        let primary = FlexPrimarySlices {
            q0: 0,
            q1: 0,
            qd1: 0,
            g: g_axis,
            h: Some(h_axis..h_axis + 1),
            w: Some(w_axis..w_axis + 1),
            infl: None,
            total: p,
        };

        // Calibration cell with non-trivial per-primary partials so the lifted
        // a_jet carries a NON-ZERO gradient a_u AND Hessian a_uv on every axis —
        // the multipliers of the channel cross-terms under test.
        let cell = DenestedCubicCell {
            left: -1.1,
            right: 1.3,
            c0: 0.22,
            c1: -0.18,
            c2: 0.13,
            c3: 0.04,
        };
        let numeric = evaluate_cell_moments(cell, 27)
            .expect("numeric moments")
            .moments
            .into_vec();
        let dc_da = [0.85_f64, 0.21, 0.06, 0.0];
        // Per-primary first/second sensitivities of the calibration coefficient:
        // drive a_u/a_uv on q(0), g(1), h(2), w(3).
        let mk_run = |base: [f64; 4], step: f64| -> Vec<[f64; 4]> {
            (0..p)
                .map(|u| base.map(|c| c * (0.1 + step * (u as f64 + 1.0))))
                .collect()
        };
        let fixed = DenestedCellPrimaryFixedPartials {
            dc_da,
            dc_daa: [0.05, 0.02, 0.0, 0.0],
            dc_daaa: [0.004, 0.0, 0.0, 0.0],
            coeff_u: mk_run([0.3, 0.1, 0.02, 0.0], 0.07),
            coeff_au: mk_run([0.12, 0.04, 0.0, 0.0], 0.05),
            coeff_bu: mk_run([0.09, 0.03, 0.0, 0.0], 0.04),
            coeff_aau: mk_run([0.02, 0.0, 0.0, 0.0], 0.01),
            coeff_abu: mk_run([0.015, 0.0, 0.0, 0.0], 0.01),
            coeff_bbu: mk_run([0.01, 0.0, 0.0, 0.0], 0.008),
            coeff_aaau: vec![[0.0; 4]; p],
            coeff_aabu: vec![[0.0; 4]; p],
            coeff_abbu: vec![[0.0; 4]; p],
            coeff_bbbu: vec![[0.0; 4]; p],
        };
        let neg_dc_da = dc_da.map(|v| -v);
        let f_a = cell_first_derivative_from_moments(&neg_dc_da, &numeric).expect("f_a");
        let d_check = f_a.abs();

        let z_obs = 0.55_f64;
        let o_infl = 0.0_f64;
        let b = 1.07_f64;
        let a0 = 0.29_f64;
        let pack = ObservedCoeffPack {
            coeff: [0.21, -0.27, 0.14, 0.05],
            dc_da: [1.05, 0.19, 0.04, 0.0],
            dc_db: [0.41, 1.02, 0.09, 0.02],
            dc_daa: [0.08, 0.03, 0.0, 0.0],
            dc_dab: [0.22, 0.1, 0.012, 0.0],
            dc_dbb: [0.13, 0.05, 0.006, 0.0],
            dc_daaa: [0.0035, 0.0, 0.0, 0.0],
            dc_daab: [0.007, 0.0012, 0.0, 0.0],
            dc_dabb: [0.0045, 0.0023, 0.0, 0.0],
            dc_dbbb: [0.0085, 0.0011, 0.0, 0.0],
        };
        // Non-trivial channel weights on g/h/w (the hand `rho`/`tau`/`tau_a`).
        // `rho[g]=dc_db`, `tau[g]=dc_dab`, `tau_a[g]=dc_daab` (evaluated at z_obs),
        // plus independent h/w channel entries.
        let mut rho = vec![0.0_f64; p];
        let mut tau = vec![0.0_f64; p];
        let mut tau_a = vec![0.0_f64; p];
        rho[g_axis] = eval_coeff4_at(&pack.dc_db, z_obs);
        rho[h_axis] = 0.37;
        rho[w_axis] = 0.29;
        tau[g_axis] = eval_coeff4_at(&pack.dc_dab, z_obs);
        tau[w_axis] = 0.18;
        tau_a[g_axis] = eval_coeff4_at(&pack.dc_daab, z_obs);
        tau_a[w_axis] = 0.11;
        // Fixed second partials r_uv (eta) / chi_uv_fixed: symmetric, with the
        // g×g, g×h, g×w structure the hand `observed_fixed_*_second_partial`
        // produces. Values are arbitrary-but-nonzero to pin the +fixed_uv add.
        let mut eta_fixed_uv = Array2::<f64>::zeros((p, p));
        let mut chi_fixed_uv = Array2::<f64>::zeros((p, p));
        let set_sym = |m: &mut Array2<f64>, i: usize, j: usize, v: f64| {
            m[[i, j]] = v;
            m[[j, i]] = v;
        };
        set_sym(&mut eta_fixed_uv, g_axis, g_axis, 0.14);
        set_sym(&mut eta_fixed_uv, g_axis, h_axis, 0.21);
        set_sym(&mut eta_fixed_uv, g_axis, w_axis, 0.17);
        set_sym(&mut chi_fixed_uv, g_axis, g_axis, 0.09);
        set_sym(&mut chi_fixed_uv, g_axis, w_axis, 0.12);

        let channels = FlexChannelInputs {
            rho: &rho,
            tau: &tau,
            tau_a: &tau_a,
            eta_fixed_uv: &eta_fixed_uv,
            chi_fixed_uv: &chi_fixed_uv,
        };
        let cells = vec![CalibrationCellJetInputs {
            base_pos_coeffs: [cell.c0, cell.c1, cell.c2, cell.c3],
            fixed: &fixed,
            cell_left: cell.left,
            cell_right: cell.right,
            // Crossing edges so the moving-boundary flux is active in a_uv / d.
            left_edge: PartitionEdge::Crossing { tau: cell.left * b + a0 },
            right_edge: PartitionEdge::Crossing { tau: cell.right * b + a0 },
            numeric_moments: &numeric,
        }];

        let out = flex_timepoint_inputs_jet2_impl(
            &primary, primary.q1, 0.0, a0, b, d_check, z_obs, o_infl, &pack, &channels, &cells,
        )
        .expect("jet timepoint inputs");

        // Reconstruct the lifted a_jet's gradient/Hessian by re-running the same
        // lift (the impl uses these internally) so the hand formula is evaluated
        // against the IDENTICAL intercept derivatives the jet used.
        let template = Jet2::from_parts(0.0, &vec![0.0; p], &[]);
        let b_jet = Jet2::primary(b, g_axis, p);
        let du: Vec<Jet2> = (0..p).map(|u| Jet2::primary(0.0, u, p)).collect();
        let residual =
            |a: &Jet2| calibration_residual_jet(a, &b_jet, g_axis, &du, primary.q1, 0.0, &cells);
        let a_jet = lift_intercept_flex(&template, a0, 1.0 / d_check, 2, residual);
        let a_u = a_jet.g.clone();
        let a_uv = |u: usize, v: usize| a_jet.h[u * p + v];

        // Observed scalars at z_obs (the hand's chi / eta_aa / eta_aaa).
        let chi = eval_coeff4_at(&pack.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&pack.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&pack.dc_daaa, z_obs);

        // EXACT match of the full hand channel-coupled Hessian (first_full.rs).
        for u in 0..p {
            for v in 0..p {
                let eta_hand = chi * a_uv(u, v)
                    + eta_aa * a_u[u] * a_u[v]
                    + tau[u] * a_u[v]
                    + tau[v] * a_u[u]
                    + eta_fixed_uv[[u, v]];
                let chi_hand = eta_aa * a_uv(u, v)
                    + eta_aaa * a_u[u] * a_u[v]
                    + tau_a[u] * a_u[v]
                    + tau_a[v] * a_u[u]
                    + chi_fixed_uv[[u, v]];
                assert!(
                    (out.eta_h[[u, v]] - eta_hand).abs() <= 1e-9 * (1.0 + eta_hand.abs()),
                    "eta_uv[{u},{v}] jet {} != hand {}",
                    out.eta_h[[u, v]],
                    eta_hand
                );
                assert!(
                    (out.chi_h[[u, v]] - chi_hand).abs() <= 1e-9 * (1.0 + chi_hand.abs()),
                    "chi_uv[{u},{v}] jet {} != hand {}",
                    out.chi_h[[u, v]],
                    chi_hand
                );
            }
        }

        // Gradient too: eta_u = chi·a_u + rho, chi_u = eta_aa·a_u + tau.
        for u in 0..p {
            let eta_u_hand = chi * a_u[u] + rho[u];
            let chi_u_hand = eta_aa * a_u[u] + tau[u];
            assert!(
                (out.eta[u] - eta_u_hand).abs() <= 1e-9 * (1.0 + eta_u_hand.abs()),
                "eta_u[{u}] jet {} != hand {}",
                out.eta[u],
                eta_u_hand
            );
            assert!(
                (out.chi[u] - chi_u_hand).abs() <= 1e-9 * (1.0 + chi_u_hand.abs()),
                "chi_u[{u}] jet {} != hand {}",
                out.chi[u],
                chi_u_hand
            );
        }

        // d_uv must be symmetric and finite (the moment-recurrence D Hessian).
        for u in 0..p {
            for v in 0..p {
                assert!(out.d_h[[u, v]].is_finite(), "d_uv finite");
                assert!(
                    (out.d_h[[u, v]] - out.d_h[[v, u]]).abs() <= 1e-9,
                    "d_uv symmetric at [{u},{v}]"
                );
            }
        }
    }
}

