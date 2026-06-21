//! Taylor-jet tower algebra: write each family's row log-likelihood ONCE,
//! derive the entire `RowKernel<K>` derivative tower mechanically (#932).
//!
//! # The object
//!
//! [`Tower4<K>`] is a truncated multivariate Taylor scalar in `K` primary
//! variables, carrying the value and ALL partial derivatives through fourth
//! order as full (unsymmetrized) tensors:
//!
//! ```text
//!   v        ℓ
//!   g[a]     ∂ℓ/∂p_a
//!   h[a][b]  ∂²ℓ/∂p_a∂p_b
//!   t3[abc]  ∂³ℓ/∂p_a∂p_b∂p_c
//!   t4[abcd] ∂⁴ℓ/∂p_a∂p_b∂p_c∂p_d
//! ```
//!
//! Arithmetic (`+ − × ÷`, scalar mixes) propagates the tower by the exact
//! Leibniz rule; unary transcendentals propagate by the exact multivariate
//! Faà di Bruno formula given a `[f, f′, f″, f‴, f⁗]` stack evaluated at the
//! inner value. This is truncated Taylor ALGEBRA — exact derivatives of the
//! evaluated expression, not finite differences, not an approximation —
//! fully compatible with the exact-REML-only policy.
//!
//! One evaluation of a row NLL program at seeded variables yields, in a
//! single pass, every channel the [`super::row_kernel::RowKernel`] trait
//! demands: `row_kernel` (value/∇/H), `row_third_contracted(dir)` (contract
//! `t3` with `dir`), and `row_fourth_contracted(u, v)` (contract `t4` with
//! `u` and `v`). The directional cross-channels that hand-written towers
//! drop (#736's residual gap) cannot be dropped here: there is no separate
//! "channel" to forget — every derivative of the one expression is carried.
//!
//! # Why this exists (the bug genus)
//!
//! Every family today hand-writes its tower: value in one function,
//! gradient in another, `pdfthird_derivative`/`pdffourth_derivative`,
//! entry/exit-specific cross blocks — thousands of lines of calculus that
//! drift. #736 was a sign flip in a hand-written cross-Hessian block,
//! invisible until a new consumer touched it; #948 is a derivative path
//! that is not the derivative of the evaluated row loss (clamped-μ
//! surrogate); the objective↔gradient desync class is the same disease at
//! the criterion level. A tower-derived kernel is exact-by-construction:
//! the value channel IS the production loss expression, so its derivative
//! channels cannot desync from it.
//!
//! # Relation to `jet_partitions::MultiDirJet`
//!
//! The tree already carries a *directional* jet (bitmask coefficients over
//! distinct seeded directions, heap-allocated, Bell-partition compose) used
//! inside the marginal-slope and latent-survival families. It answers "the
//! derivative along THESE specific directions" and must be re-seeded and
//! re-evaluated per direction tuple (e.g. 10 symmetric `(a,b)` pairs for a
//! K=4 fourth contraction). `Tower4` answers ALL of them from one
//! evaluation: contraction happens AFTER differentiation, as plain linear
//! algebra on the stored tensors. Use `MultiDirJet` when you need a handful
//! of directions of a huge-K expression; use `Tower4` when you need the
//! complete small-K tower — which is exactly the `RowKernel<K≤4>` shape.
//! The `[f64; 5]` unary-derivative stacks
//! (`unary_derivatives_neglog_phi`, …) are signature-compatible with
//! [`Tower4::compose_unary`], so the families' existing special-function
//! stacks are directly reusable.
//!
//! # Stability discipline (why this is NOT autodiff)
//!
//! Differentiating the primal code path inherits its instabilities: a jet
//! pushed through a naive `ln(1 + e^η)` is garbage in the saturated tail
//! even though the true derivative σ(η) is benign there. This module
//! therefore splits responsibility: **humans own primitive stability,
//! the algebra owns combinatorics**. Tail-critical special functions enter
//! a program ONLY as hand-certified `[f64; 5]` derivative stacks through
//! [`Tower4::compose_unary`] — the same stacks the families already write
//! (`unary_derivatives_neglog_phi` and friends, built on erfcx/log_ndtr) —
//! and the tower mechanizes only the Leibniz/Faà di Bruno composition,
//! which is where hand-written towers actually fail (#736 was a
//! composition sign flip, not a primitive error). Program authors must use
//! a stable primitive stack wherever the f64 production loss does; the
//! convenience methods (`exp`, `ln`, `sqrt`, …) are for expressions whose
//! arguments are tame by construction.
//!
//! # Storage convention
//!
//! Tensors are stored FULL, not symmetric-packed: `t4` for K=4 is 256
//! doubles where 35 would do. This is deliberate clarity-over-speed for the
//! oracle role — indexing is trivially auditable, contraction loops are
//! obvious, and the redundancy is itself a checked invariant (the algebra
//! only ever writes symmetric values). Symmetric packing is a later,
//! profile-justified optimization behind the same API.
//!
//! # Deployment ladder (#932)
//!
//! 1. This module: the algebra + the program seam + the oracle.
//! 2. Universal oracle: every hand-written `RowKernel` gains a CI test
//!    asserting channel-by-channel agreement with a `RowNllProgram` written
//!    once — see [`verify_kernel_channels`]. This alone would have caught
//!    #736 at introduction.
//! 3. Migrate error-dense / cold towers to [`derived_row_kernel`] et al.;
//!    keep hand-tuned hot paths, now verified against the single-expression
//!    truth instead of being the only definition.
//! 4. New families (#914/#916/#917 ZI/ordinal/expectile, #921's location-
//!    scale port) implement ONLY `RowNllProgram` and get an exact
//!    fourth-order tower for the price of writing the likelihood.

use super::jet_algebra;

/// Truncated fourth-order multivariate Taylor scalar in `K` variables.
///
/// See the module documentation for semantics and conventions. `Copy` is
/// intentional despite the size (2 KiB at K=4): towers are per-row
/// temporaries that live entirely in registers/stack during a row program,
/// and value semantics keep program code readable (`a * b + c`).
#[derive(Clone, Copy, Debug)]
pub struct Tower4<const K: usize> {
    /// Value ℓ.
    pub v: f64,
    /// Gradient ∂ℓ/∂p_a.
    pub g: [f64; K],
    /// Hessian ∂²ℓ/∂p_a∂p_b (symmetric).
    pub h: [[f64; K]; K],
    /// Third derivatives ∂³ℓ/∂p_a∂p_b∂p_c (fully symmetric).
    pub t3: [[[f64; K]; K]; K],
    /// Fourth derivatives ∂⁴ℓ/∂p_a∂p_b∂p_c∂p_d (fully symmetric).
    pub t4: [[[[f64; K]; K]; K]; K],
}

impl<const K: usize> Tower4<K> {
    /// The additive identity.
    pub fn zero() -> Self {
        Self {
            v: 0.0,
            g: [0.0; K],
            h: [[0.0; K]; K],
            t3: [[[0.0; K]; K]; K],
            t4: [[[[0.0; K]; K]; K]; K],
        }
    }

    /// A constant: value `c`, all derivatives zero.
    pub fn constant(c: f64) -> Self {
        let mut out = Self::zero();
        out.v = c;
        out
    }

    /// The seeded variable `p_idx` with current value `value`:
    /// unit first derivative in slot `idx`, zero elsewhere and above.
    pub fn variable(value: f64, idx: usize) -> Self {
        let mut out = Self::constant(value);
        out.g[idx] = 1.0;
        out
    }

    /// Read the (fully symmetric) derivative tensor entry whose differentiation
    /// axes are `labels` (length 0..=4): value, `g`, `h`, `t3`, `t4`.
    #[inline]
    fn deriv(&self, labels: &[usize]) -> f64 {
        assert!(
            labels.len() <= 4,
            "Tower4 carries at most fourth-order derivatives"
        );
        match labels.len() {
            0 => self.v,
            1 => self.g[labels[0]],
            2 => self.h[labels[0]][labels[1]],
            3 => self.t3[labels[0]][labels[1]][labels[2]],
            _ => self.t4[labels[0]][labels[1]][labels[2]][labels[3]],
        }
    }

    /// Exact truncated Leibniz product.
    ///
    /// Every output entry `D_S(ab) = Σ_{T ⊆ S} D_T(a) · D_{S∖T}(b)` is summed
    /// by the shared [`jet_algebra::leibniz_product`] subset walker (#1151),
    /// the same kernel `MultiDirJet::mul` uses; the two layouts differ only in
    /// how a slot-group selects a derivative.
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v * b.v;
        for i in 0..K {
            let labels = [i];
            out.g[i] = jet_algebra::leibniz_product(&labels, |t| a.deriv(t), |c| b.deriv(c));
        }
        for i in 0..K {
            for j in 0..K {
                let labels = [i, j];
                out.h[i][j] = jet_algebra::leibniz_product(&labels, |t| a.deriv(t), |c| b.deriv(c));
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    let labels = [i, j, k];
                    out.t3[i][j][k] =
                        jet_algebra::leibniz_product(&labels, |t| a.deriv(t), |c| b.deriv(c));
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        let labels = [i, j, k, l];
                        out.t4[i][j][k][l] =
                            jet_algebra::leibniz_product(&labels, |t| a.deriv(t), |c| b.deriv(c));
                    }
                }
            }
        }
        out
    }

    /// Exact multivariate Faà di Bruno composition `f ∘ self`.
    ///
    /// `d = [f(u), f′(u), f″(u), f‴(u), f⁗(u)]` evaluated at `u = self.v` —
    /// the SAME `[f64; 5]` stack shape the families' existing
    /// `unary_derivatives_*` helpers produce, so those special-function
    /// stacks (Φ, log-Φ, normal pdf, …) plug in directly.
    ///
    /// The order-m output sums over the set partitions of the m indices
    /// (Bell(3) = 5 terms at order 3, Bell(4) = 15 at order 4), grouped by
    /// block count: each partition into r blocks contributes
    /// `f⁽ʳ⁾ · Π_blocks D_block(u)`.
    pub fn compose_unary(&self, d: [f64; 5]) -> Self {
        <Self as jet_algebra::JetAlgebra<5>>::compose_unary(self, d)
    }

    /// Multiply every channel by a plain scalar.
    pub fn scale(&self, s: f64) -> Self {
        let mut out = *self;
        out.v *= s;
        for i in 0..K {
            out.g[i] *= s;
            for j in 0..K {
                out.h[i][j] *= s;
                for k in 0..K {
                    out.t3[i][j][k] *= s;
                    for l in 0..K {
                        out.t4[i][j][k][l] *= s;
                    }
                }
            }
        }
        out
    }

    /// e^self.
    pub fn exp(&self) -> Self {
        let e = self.v.exp();
        self.compose_unary([e, e, e, e, e])
    }

    /// ln(self). Caller guarantees positivity (likelihood programs do).
    pub fn ln(&self) -> Self {
        let u = self.v;
        let r = 1.0 / u;
        self.compose_unary([u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r])
    }

    /// 1/self.
    pub fn recip(&self) -> Self {
        let r = 1.0 / self.v;
        let r2 = r * r;
        self.compose_unary([r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r])
    }

    /// √self. Caller guarantees positivity.
    pub fn sqrt(&self) -> Self {
        let u = self.v;
        let s = u.sqrt();
        self.compose_unary([
            s,
            0.5 / s,
            -0.25 / (u * s),
            0.375 / (u * u * s),
            -0.9375 / (u * u * u * s),
        ])
    }

    /// self^a for real exponent `a`. Caller guarantees a positive base.
    pub fn powf(&self, a: f64) -> Self {
        let u = self.v;
        let f0 = u.powf(a);
        let f1 = a * u.powf(a - 1.0);
        let f2 = a * (a - 1.0) * u.powf(a - 2.0);
        let f3 = a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0);
        let f4 = a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0);
        self.compose_unary([f0, f1, f2, f3, f4])
    }

    /// ln Γ(self). Caller guarantees positivity.
    pub fn ln_gamma(&self) -> Self {
        self.compose_unary(ln_gamma_derivative_stack(self.v))
    }

    /// ψ(self), the digamma function. Caller guarantees positivity.
    pub fn digamma(&self) -> Self {
        self.compose_unary(digamma_derivative_stack(self.v))
    }

    /// ψ′(self), the trigamma function. Caller guarantees positivity.
    pub fn trigamma(&self) -> Self {
        self.compose_unary(trigamma_derivative_stack(self.v))
    }

    /// Contract `t3` with one primary-space direction:
    /// `out[a][b] = Σ_c t3[a][b][c] · dir[c]` — exactly the
    /// `row_third_contracted` shape.
    pub fn third_contracted(&self, dir: &[f64; K]) -> [[f64; K]; K] {
        let mut out = [[0.0; K]; K];
        for a in 0..K {
            for b in 0..K {
                let mut acc = 0.0;
                for c in 0..K {
                    acc += self.t3[a][b][c] * dir[c];
                }
                out[a][b] = acc;
            }
        }
        out
    }

    /// Contract `t4` with two primary-space directions:
    /// `out[a][b] = Σ_{c,d} t4[a][b][c][d] · u[c] · v[d]` — exactly the
    /// `row_fourth_contracted` shape.
    pub fn fourth_contracted(&self, u: &[f64; K], w: &[f64; K]) -> [[f64; K]; K] {
        let mut out = [[0.0; K]; K];
        for i in 0..K {
            for j in 0..K {
                let mut acc = 0.0;
                for k in 0..K {
                    for l in 0..K {
                        acc += self.t4[i][j][k][l] * u[k] * w[l];
                    }
                }
                out[i][j] = acc;
            }
        }
        out
    }
}

impl<const K: usize> jet_algebra::JetAlgebra<5> for Tower4<K> {
    #[inline]
    fn derivative(&self, labels: &[usize]) -> f64 {
        self.deriv(labels)
    }

    fn map_derivatives<F>(&self, mut f: F) -> Self
    where
        F: FnMut(&[usize]) -> f64,
    {
        let mut out = Self::zero();
        out.v = f(&[]);
        for i in 0..K {
            let labels = [i];
            out.g[i] = f(&labels);
        }
        for i in 0..K {
            for j in 0..K {
                let labels = [i, j];
                out.h[i][j] = f(&labels);
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    let labels = [i, j, k];
                    out.t3[i][j][k] = f(&labels);
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        let labels = [i, j, k, l];
                        out.t4[i][j][k][l] = f(&labels);
                    }
                }
            }
        }
        out
    }
}

/// Truncated SECOND-order multivariate Taylor scalar in `K` variables.
///
/// This is the value/gradient/Hessian-only sibling of [`Tower4`]. Every
/// channel it carries (`v`, `g`, `h`) is computed by the SAME formulas
/// [`Tower4`] uses for those orders, so for any program written over both
/// towers the order-≤2 outputs are *bit-identical*: the order-2 Leibniz and
/// Faà-di-Bruno terms read only the order-≤2 channels of their inputs (see
/// [`Tower4::mul`] / [`Tower4::compose_unary`] — `out.h` never touches `t3`
/// or `t4`), so dropping the third/fourth tensors cannot perturb the value,
/// gradient, or Hessian.
///
/// It exists purely for performance: an inner Newton step (and the
/// value-only ρ-homotopy pre-warm) needs at most curvature, never the
/// outer-κ/ψ third/fourth derivatives. Evaluating a row likelihood over
/// `Tower2` skips the `K⁴` fourth-tensor product/composition arithmetic that
/// dominates the cold marginal-slope fit, while returning the exact same
/// `(v, g, h)`.
#[derive(Clone, Copy, Debug)]
pub struct Tower2<const K: usize> {
    /// Value ℓ.
    pub v: f64,
    /// Gradient ∂ℓ/∂p_a.
    pub g: [f64; K],
    /// Hessian ∂²ℓ/∂p_a∂p_b (symmetric).
    pub h: [[f64; K]; K],
}

impl<const K: usize> Tower2<K> {
    /// The additive identity.
    pub fn zero() -> Self {
        Self {
            v: 0.0,
            g: [0.0; K],
            h: [[0.0; K]; K],
        }
    }

    /// A constant: value `c`, all derivatives zero.
    pub fn constant(c: f64) -> Self {
        let mut out = Self::zero();
        out.v = c;
        out
    }

    /// The seeded variable `p_idx` with current value `value`:
    /// unit first derivative in slot `idx`, zero elsewhere and above.
    pub fn variable(value: f64, idx: usize) -> Self {
        let mut out = Self::constant(value);
        out.g[idx] = 1.0;
        out
    }

    /// Read the derivative tensor entry whose differentiation axes are
    /// `labels` (length 0..=2): value, `g`, `h`.
    #[inline]
    fn deriv(&self, labels: &[usize]) -> f64 {
        assert!(
            labels.len() <= 2,
            "Tower2 carries at most second-order derivatives"
        );
        match labels.len() {
            0 => self.v,
            1 => self.g[labels[0]],
            _ => self.h[labels[0]][labels[1]],
        }
    }

    /// Exact truncated (order ≤ 2) Leibniz product. The `v`/`g`/`h` channels
    /// match [`Tower4::mul`] term-for-term.
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v * b.v;
        for i in 0..K {
            out.g[i] = a.v * b.g[i] + a.g[i] * b.v;
        }
        for i in 0..K {
            for j in 0..K {
                out.h[i][j] = a.v * b.h[i][j] + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.h[i][j] * b.v;
            }
        }
        out
    }

    /// Exact (order ≤ 2) multivariate Faà di Bruno composition `f ∘ self`.
    ///
    /// `d = [f(u), f′(u), f″(u)]` evaluated at `u = self.v`. The `v`/`g`/`h`
    /// channels match [`Tower4::compose_unary`] term-for-term (which uses only
    /// `d[0..=2]` for those orders), so this is a strict truncation, not an
    /// approximation. The full-order `[f64; 5]` derivative stacks the families
    /// already produce can be passed by slicing their first three entries.
    pub fn compose_unary(&self, d: [f64; 3]) -> Self {
        <Self as jet_algebra::JetAlgebra<3>>::compose_unary(self, d)
    }

    /// Multiply every channel by a plain scalar.
    pub fn scale(&self, s: f64) -> Self {
        let mut out = *self;
        out.v *= s;
        for i in 0..K {
            out.g[i] *= s;
            for j in 0..K {
                out.h[i][j] *= s;
            }
        }
        out
    }

    /// √self. Caller guarantees positivity.
    pub fn sqrt(&self) -> Self {
        let u = self.v;
        let s = u.sqrt();
        self.compose_unary([s, 0.5 / s, -0.25 / (u * s)])
    }
}

impl<const K: usize> jet_algebra::JetAlgebra<3> for Tower2<K> {
    #[inline]
    fn derivative(&self, labels: &[usize]) -> f64 {
        self.deriv(labels)
    }

    fn map_derivatives<F>(&self, mut f: F) -> Self
    where
        F: FnMut(&[usize]) -> f64,
    {
        let mut out = Self::zero();
        out.v = f(&[]);
        for i in 0..K {
            let labels = [i];
            out.g[i] = f(&labels);
        }
        for i in 0..K {
            for j in 0..K {
                let labels = [i, j];
                out.h[i][j] = f(&labels);
            }
        }
        out
    }
}

impl<const K: usize> std::ops::Add for Tower2<K> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut out = self;
        out.v += o.v;
        for i in 0..K {
            out.g[i] += o.g[i];
            for j in 0..K {
                out.h[i][j] += o.h[i][j];
            }
        }
        out
    }
}

impl<const K: usize> std::ops::Mul for Tower2<K> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Tower2::mul(&self, &o)
    }
}

impl<const K: usize> std::ops::Add<f64> for Tower2<K> {
    type Output = Self;
    fn add(self, c: f64) -> Self {
        let mut out = self;
        out.v += c;
        out
    }
}

impl<const K: usize> std::ops::Mul<f64> for Tower2<K> {
    type Output = Self;
    fn mul(self, c: f64) -> Self {
        self.scale(c)
    }
}

pub fn ln_gamma_derivative_stack(x: f64) -> [f64; 5] {
    [
        statrs::function::gamma::ln_gamma(x),
        digamma_positive(x),
        polygamma_positive(1, x),
        polygamma_positive(2, x),
        polygamma_positive(3, x),
    ]
}

pub fn digamma_derivative_stack(x: f64) -> [f64; 5] {
    [
        digamma_positive(x),
        polygamma_positive(1, x),
        polygamma_positive(2, x),
        polygamma_positive(3, x),
        polygamma_positive(4, x),
    ]
}

pub fn trigamma_derivative_stack(x: f64) -> [f64; 5] {
    [
        polygamma_positive(1, x),
        polygamma_positive(2, x),
        polygamma_positive(3, x),
        polygamma_positive(4, x),
        polygamma_positive(5, x),
    ]
}

fn digamma_positive(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < POLYGAMMA_ASYMPTOTIC_MIN_X {
        acc -= 1.0 / x;
        x += 1.0;
    }
    acc + digamma_asymptotic(x)
}

fn polygamma_positive(order: usize, mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < POLYGAMMA_ASYMPTOTIC_MIN_X {
        acc += polygamma_recurrence_term(order, x);
        x += 1.0;
    }
    acc + polygamma_asymptotic(order, x)
}

const POLYGAMMA_ASYMPTOTIC_MIN_X: f64 = 20.0;
const BERNOULLI_EVEN: [(usize, f64); 10] = [
    (2, 1.0 / 6.0),
    (4, -1.0 / 30.0),
    (6, 1.0 / 42.0),
    (8, -1.0 / 30.0),
    (10, 5.0 / 66.0),
    (12, -691.0 / 2730.0),
    (14, 7.0 / 6.0),
    (16, -3617.0 / 510.0),
    (18, 43867.0 / 798.0),
    (20, -174611.0 / 330.0),
];

fn polygamma_recurrence_term(order: usize, x: f64) -> f64 {
    let sign = if order % 2 == 1 { 1.0 } else { -1.0 };
    sign * factorial(order) / x.powi((order + 1) as i32)
}

fn digamma_asymptotic(x: f64) -> f64 {
    let mut out = x.ln() - 0.5 / x;
    for (bernoulli_order, bernoulli) in BERNOULLI_EVEN {
        out -= bernoulli / (bernoulli_order as f64 * x.powi(bernoulli_order as i32));
    }
    out
}

fn polygamma_asymptotic(order: usize, x: f64) -> f64 {
    if !(1..=5).contains(&order) {
        return f64::NAN;
    }

    let order_factorial = factorial(order);
    let leading_sign = if order % 2 == 1 { 1.0 } else { -1.0 };
    let mut out = leading_sign * factorial(order - 1) / x.powi(order as i32)
        + leading_sign * order_factorial / (2.0 * x.powi((order + 1) as i32));

    let bernoulli_sign = if order % 2 == 1 { 1.0 } else { -1.0 };
    for (bernoulli_order, bernoulli) in BERNOULLI_EVEN {
        let rising = rising_factorial(bernoulli_order, order);
        out += bernoulli_sign * bernoulli * rising
            / bernoulli_order as f64
            / x.powi((bernoulli_order + order) as i32);
    }
    out
}

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, k| acc * k as f64)
}

fn rising_factorial(start: usize, len: usize) -> f64 {
    (start..start + len).fold(1.0, |acc, k| acc * k as f64)
}

impl<const K: usize> std::ops::Add for Tower4<K> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut out = self;
        out.v += o.v;
        for i in 0..K {
            out.g[i] += o.g[i];
            for j in 0..K {
                out.h[i][j] += o.h[i][j];
                for k in 0..K {
                    out.t3[i][j][k] += o.t3[i][j][k];
                    for l in 0..K {
                        out.t4[i][j][k][l] += o.t4[i][j][k][l];
                    }
                }
            }
        }
        out
    }
}

impl<const K: usize> std::ops::Sub for Tower4<K> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self + o.scale(-1.0)
    }
}

impl<const K: usize> std::ops::Neg for Tower4<K> {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl<const K: usize> std::ops::Mul for Tower4<K> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Tower4::mul(&self, &o)
    }
}

impl<const K: usize> std::ops::Div for Tower4<K> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        Tower4::mul(&self, &o.recip())
    }
}

impl<const K: usize> std::ops::Add<f64> for Tower4<K> {
    type Output = Self;
    fn add(self, c: f64) -> Self {
        let mut out = self;
        out.v += c;
        out
    }
}

impl<const K: usize> std::ops::Sub<f64> for Tower4<K> {
    type Output = Self;
    fn sub(self, c: f64) -> Self {
        self + (-c)
    }
}

impl<const K: usize> std::ops::Mul<f64> for Tower4<K> {
    type Output = Self;
    fn mul(self, c: f64) -> Self {
        self.scale(c)
    }
}

// ── Implicit-function and moving-boundary seams (#932 flex) ──────────
//
// The flexible survival marginal-slope row loss is NOT a free composition
// of the primaries: it threads an IMPLICIT calibration intercept `a(θ)`
// solving a constraint `F(a, θ) = 0`, and integrates a density over cells
// whose edges `z_L(θ), z_R(θ)` MOVE with θ through that intercept. Plain
// `Tower4` Faà di Bruno cannot express either — so the flex tower was the
// last hand-written one in the codebase, and the genus of #736-class
// drift bugs (the (g,w0) deviation-cross third was 3× short for exactly
// this reason). These two combinators close that gap: once the constraint
// `F` and the integrand/boundaries are themselves towers, the intercept's
// derivative tower and the integral's derivative tower come out EXACTLY at
// every order — there is no order left to hand-code and forget.

/// Solve the implicit relation `F(a(θ), θ) ≡ 0` for the intercept tower
/// `a(θ)` over the `K` primaries θ, given the constraint tower `f` written
/// over `K + 1` variables (slot `0` is the intercept `a`, slots `1..=K`
/// are the primaries θ) evaluated at the SOLVED point — i.e. `f.v` is the
/// constraint residual at `(a₀, θ₀)` (≈ 0 from the production Newton solve)
/// and `a0` is that solved intercept value.
///
/// Returns the `Tower4<K>` whose value is `a0` and whose every derivative
/// tensor (∂a/∂θ, ∂²a/∂θ², …, ∂⁴a/∂θ⁴) is the exact implicit-function
/// derivative. This is the mechanical replacement for the hand-coded
/// `a_u = -f_u/f_a`, `a_uv = -(f_uv + f_au·a_v + f_av·a_u + f_aa·a_u·a_v)/f_a`
/// recursion (first_full.rs) and its third/fourth-order continuations.
///
/// Method: order-by-order substitution. We build `a` incrementally; at each
/// order `m` the composite `G(θ) = f(a(θ), θ)` has a top-order coefficient
/// that is linear in `a`'s order-`m` tensor with leading factor `F_a`
/// (= `f.g[0]`), plus terms in `a`'s lower orders already fixed. Setting the
/// order-`m` tensor of `a` to cancel the rest of `G`'s order-`m` coefficient
/// keeps `G ≡ 0` through that order. The substitution `G = f∘(a, θ)` reuses
/// only the exact [`substitute_intercept`] chain rule, so the recursion is
/// auditable and exact, not a hand-expanded formula per order.
///
/// `f.g[0]` (= ∂F/∂a) must be non-zero — guaranteed by the production
/// solve's strict monotonicity guard.
///
/// The expansion point `a0` must be a genuine root `F(a0, θ0) = 0`: the
/// substitution recursion below cancels orders 1..=4 of `G = F∘a` but never
/// touches order 0, so a non-root `a0` would yield the Taylor expansion of
/// the LEVEL SET `F = F(a0)` through `a0`, not the root curve `F = 0`. This
/// is guarded explicitly and re-verified by a composed-residual self-check.
pub fn implicit_solve<const K1: usize, const K: usize>(
    f: &Tower4<K1>,
    a0: f64,
) -> Result<Tower4<K>, String> {
    assert_eq!(K1, K + 1, "implicit_solve: constraint must carry K+1 vars");
    let f_a = f.g[0];
    if f_a == 0.0 || !f_a.is_finite() {
        return Err(format!(
            "implicit_solve: ∂F/∂a = {f_a:+.3e} is not invertible"
        ));
    }
    // The expansion point must be a genuine root of F. The single Newton
    // correction that would move a0 onto the root is |f.v|/|f_a|; require it
    // to be negligible relative to the natural scale (1 + |a0|). Guarding the
    // Newton step (rather than f.v directly) makes the criterion invariant to
    // the magnitude of f_a / the units of F.
    let root_tol = 1e-9;
    if !f.v.is_finite() {
        return Err(format!(
            "implicit_solve: F(a0, θ0) = {:+.3e} is not finite",
            f.v
        ));
    }
    let newton_step = f.v.abs() / f_a.abs();
    if newton_step > root_tol * (1.0 + a0.abs()) {
        return Err(format!(
            "implicit_solve: expansion point a0 = {a0:+.6e} is not a root of F: \
             F(a0, θ0) = {:+.3e}, Newton correction {newton_step:+.3e} exceeds \
             root_tol {root_tol:.1e} · (1 + |a0|)",
            f.v
        ));
    }
    // Start with a = constant a0 (correct through order 0). Then lift each
    // order in turn. Because substitute_intercept reads `a`'s order-≤m
    // tensors when forming G's order-m coefficient, and the order-m
    // coefficient of G depends on a's order-m tensor ONLY through the linear
    // F_a·a_m term, a single corrective pass per order is exact.
    let mut a = Tower4::<K>::constant(a0);
    for order in 1..=4 {
        let g = substitute_intercept(f, &a);
        // Cancel G's order-`order` coefficient by adjusting a's order-`order`
        // tensor: a_m -= G_m / F_a (the F_a·a_m term is the only one carrying
        // a's order-m tensor, with unit chain coefficient since slot 0 seeds a
        // as a plain variable in the substitution's first-order part).
        match order {
            1 => {
                for i in 0..K {
                    a.g[i] -= g.g[i] / f_a;
                }
            }
            2 => {
                for i in 0..K {
                    for j in 0..K {
                        a.h[i][j] -= g.h[i][j] / f_a;
                    }
                }
            }
            3 => {
                for i in 0..K {
                    for j in 0..K {
                        for k in 0..K {
                            a.t3[i][j][k] -= g.t3[i][j][k] / f_a;
                        }
                    }
                }
            }
            _ => {
                for i in 0..K {
                    for j in 0..K {
                        for k in 0..K {
                            for l in 0..K {
                                a.t4[i][j][k][l] -= g.t4[i][j][k][l] / f_a;
                            }
                        }
                    }
                }
            }
        }
    }
    // Self-check: the composed residual G = F∘a must vanish through order 4.
    // By construction orders 1..=4 were cancelled; the value G.v == F(a0,θ0)
    // is exactly the root requirement guarded above. Re-verify all channels
    // against a scale-aware floor so any arithmetic regression in the
    // substitution recursion is loud rather than silently shipping a
    // level-set expansion.
    let g = substitute_intercept(f, &a);
    let resid_tol = 1e-7 * (1.0 + f_a.abs());
    let mut worst = g.v.abs();
    for i in 0..K {
        worst = worst.max(g.g[i].abs());
        for j in 0..K {
            worst = worst.max(g.h[i][j].abs());
            for k in 0..K {
                worst = worst.max(g.t3[i][j][k].abs());
                for l in 0..K {
                    worst = worst.max(g.t4[i][j][k][l].abs());
                }
            }
        }
    }
    if !worst.is_finite() || worst > resid_tol {
        return Err(format!(
            "implicit_solve: composed residual G = F∘a does not vanish: \
             worst channel magnitude {worst:+.3e} exceeds tol {resid_tol:.1e}"
        ));
    }
    Ok(a)
}

/// Substitute the intercept tower `a(θ)` into slot `0` of a constraint
/// written over `K + 1` variables, returning the composite tower over the
/// `K` primaries θ: `G(θ) = f(a(θ), θ₁, …, θ_K)`.
///
/// This is the exact multivariate chain rule specialised to "slot 0 is a
/// dependent tower, slots 1..=K are the independent primaries". It evaluates
/// `f`'s fourth-order multivariate Taylor polynomial about the expansion
/// point, with the slot-0 increment being the non-constant part of `a` and
/// the slot-(i) increment being the unit-seeded primary `θ_i`. The sum is
/// assembled by the same subset/partition algebra `Tower4` arithmetic uses,
/// so it carries derivatives exactly through order four.
pub fn substitute_intercept<const K1: usize, const K: usize>(
    f: &Tower4<K1>,
    a: &Tower4<K>,
) -> Tower4<K> {
    assert_eq!(K1, K + 1);
    // Build the K+1 input towers in θ-space: slot 0 = a(θ), slot i+1 = θ_i.
    // The composite is Σ over ordered label tuples s (|s| ≤ 4) of input
    // indices: (1/|s|!) · f.deriv(s) · Π_{j in s} (inp[s_j] centred) — but
    // since f.deriv is the SYMMETRIC partial tensor and we enumerate ordered
    // tuples, the 1/|s|! exactly cancels the tuple multiplicity. We assemble
    // it directly as a Horner-free explicit sum over the (K+1)-ary tuples,
    // using tower products for the increment monomials so all θ-derivatives
    // propagate exactly.
    let inp: [Tower4<K>; K1] = std::array::from_fn(|slot| {
        if slot == 0 {
            // slot 0: a(θ) minus its constant value (the increment δa(θ)).
            let mut d = *a;
            d.v = 0.0;
            d
        } else {
            // slot i: the increment δθ_{i-1} = seeded variable minus value.
            // θ centred at its expansion value has zero constant term and unit
            // first derivative in its own slot.
            let mut d = Tower4::<K>::zero();
            d.g[slot - 1] = 1.0;
            d
        }
    });
    // Accumulate the Taylor sum. order-0 term:
    let mut out = Tower4::<K>::constant(f.v);
    // order 1: Σ_a f.g[a] · inp[a]
    for a_idx in 0..K1 {
        out = out + inp[a_idx].scale(f.g[a_idx]);
    }
    // order 2: (1/2) Σ_{a,b} f.h[a][b] · inp[a]·inp[b]
    for a_idx in 0..K1 {
        for b_idx in 0..K1 {
            let prod = inp[a_idx].mul(&inp[b_idx]);
            out = out + prod.scale(0.5 * f.h[a_idx][b_idx]);
        }
    }
    // order 3: (1/6) Σ f.t3[a][b][c] · inp[a]·inp[b]·inp[c]
    for a_idx in 0..K1 {
        for b_idx in 0..K1 {
            for c_idx in 0..K1 {
                let prod = inp[a_idx].mul(&inp[b_idx]).mul(&inp[c_idx]);
                out = out + prod.scale(f.t3[a_idx][b_idx][c_idx] / 6.0);
            }
        }
    }
    // order 4: (1/24) Σ f.t4[a][b][c][d] · inp[a]·inp[b]·inp[c]·inp[d]
    for a_idx in 0..K1 {
        for b_idx in 0..K1 {
            for c_idx in 0..K1 {
                for d_idx in 0..K1 {
                    let prod = inp[a_idx]
                        .mul(&inp[b_idx])
                        .mul(&inp[c_idx])
                        .mul(&inp[d_idx]);
                    out = out + prod.scale(f.t4[a_idx][b_idx][c_idx][d_idx] / 24.0);
                }
            }
        }
    }
    out
}

/// The exact θ-derivative tower of a moving-LIMIT integral's BOUNDARY
/// contribution: given the edge-position tower `z_edge(θ)` over the `K`
/// primaries and the integrand `B` evaluated-and-differentiated at the edge
/// value as the stack `b_stack = [B(z₀), B′(z₀), B″(z₀), B‴(z₀)]`
/// (`z₀ = z_edge.v`), returns the tower of `Φ(z_edge(θ))` where `Φ′ = B`.
///
/// Rationale: `∂_θ ∫^{z_edge(θ)} B(z) dz = Φ(z_edge(θ))` with `Φ` an
/// antiderivative of `B`, so the boundary part of every θ-derivative of the
/// integral is just the composition `Φ ∘ z_edge` — whose Faà di Bruno
/// expansion carries, at one stroke, EVERY Leibniz boundary term the
/// hand-written flux dropped: the first-order `B·z_u`, the second-order
/// `B′·z_u·z_v + B·z_uv` (the `G_z·z_u·z_v` self-flux AND the previously
/// dropped `G·z_uv`), and the full third/fourth-order continuations. The
/// VALUE channel of the returned tower is meaningless (`Φ` is only defined up
/// to a constant); callers read only the derivative channels and pair this
/// with the interior moment-integral value separately.
///
/// `b_stack` holds `B` and its first three z-derivatives; the antiderivative
/// `Φ` contributes only as the order-≥1 channels, so `compose_unary` receives
/// `[0, B, B′, B″, B‴]` — the leading `0` is the discarded `Φ(z₀)` slot.
pub fn moving_limit_boundary_tower<const K: usize>(
    z_edge: &Tower4<K>,
    b_stack: [f64; 4],
) -> Tower4<K> {
    z_edge.compose_unary([0.0, b_stack[0], b_stack[1], b_stack[2], b_stack[3]])
}

/// The boundary-flux derivative tower of a single moving cell integral
/// `∫_{z_L(θ)}^{z_R(θ)} B dz`: `Φ(z_R(θ)) − Φ(z_L(θ))`, assembled from the
/// two edge towers and the integrand stacks at each edge. The returned
/// tower's derivative channels are the EXACT moving-boundary contribution to
/// every θ-derivative of the cell integral, to fourth order, with no term
/// hand-omitted. A `Fixed` (non-moving) edge passes a `z_edge` whose
/// derivative channels are all zero, contributing nothing — matching the
/// production `edge_vel = 0` short-circuit.
pub fn cell_moving_boundary_flux_tower<const K: usize>(
    z_right: &Tower4<K>,
    b_stack_right: [f64; 4],
    z_left: &Tower4<K>,
    b_stack_left: [f64; 4],
) -> Tower4<K> {
    moving_limit_boundary_tower(z_right, b_stack_right)
        - moving_limit_boundary_tower(z_left, b_stack_left)
}

/// Moving-limit boundary tower for a θ-DEPENDENT integrand `G(z; θ)`.
///
/// [`moving_limit_boundary_tower`] assumes the integrand depends on θ only
/// through the moving edge `z_edge(θ)` (a fixed z-derivative `b_stack`). The
/// marginal-slope flex boundary is richer: the integrand `G(z; θ)` ALSO carries
/// its own θ-dependence (the density weight `w = e^{−q}/2π` and the cell
/// integrand coefficients move with η, hence with the primaries), so the
/// Leibniz expansion of `∂ⁿ_θ ∫^{z_edge(θ)} G(z;θ) dz` mixes edge-motion
/// derivatives of the limit with θ-derivatives of `G` itself — e.g. at second
/// order `G·z_uv + G_z·z_u·z_v + G_{θu}·z_v + G_{θv}·z_u` (the four
/// edge-motion-carrying terms the hand path assembles one by one, including the
/// `G·z_uv` term the directional path drops).
///
/// Mechanization: let `Φ(z; θ)` be the z-antiderivative of `G` (so `Φ_z = G`).
/// The full upper-limit contribution is `Φ(z_edge(θ); θ)`, and the BOUNDARY
/// part — everything carrying edge motion — is exactly
///   `Φ(z_edge(θ); θ) − Φ(z₀; θ)`,
/// the second term being the pure-integrand-θ part (`∫^{z₀} ∂ⁿ_θ G`) the
/// interior moment integral already supplies. Both are one
/// [`substitute_intercept`] of the SAME mixed `(z, θ)` jet of `Φ` (z in slot 0,
/// θ in slots 1..K): substituting the edge tower gives the full composite,
/// substituting a frozen constant edge isolates the pure-θ part, and their
/// difference is the exact boundary flux — every Leibniz term derived by the
/// substitution algebra, none hand-omitted.
///
/// `phi_jet` is the `(K+1)`-variable Taylor jet of `Φ` about `(z₀, θ₀)` with
/// `z₀ = z_edge.v`: slot 0 is the z-direction (so `phi_jet.g[0] = G(z₀;θ₀)`,
/// `phi_jet.h[0][0] = G_z`, …) and slots `1..=K` are the primaries θ (carrying
/// `Φ`'s own θ- and mixed z·θ-derivatives — i.e. the integrand's θ-derivatives
/// integrated in z, and `G_{θ…}` in the mixed slots). The returned tower's
/// VALUE channel is 0 by construction (the `Φ(z₀;θ₀)` constants cancel); only
/// the derivative channels are meaningful, matching the value-less convention of
/// [`moving_limit_boundary_tower`].
pub fn moving_limit_boundary_tower_theta_integrand<const K1: usize, const K: usize>(
    phi_jet: &Tower4<K1>,
    z_edge: &Tower4<K>,
) -> Tower4<K> {
    assert_eq!(
        K1,
        K + 1,
        "moving_limit_boundary_tower_theta_integrand: Φ jet must carry z + K θ-vars"
    );
    let frozen_edge = Tower4::<K>::constant(z_edge.v);
    let full = substitute_intercept(phi_jet, z_edge);
    let interior = substitute_intercept(phi_jet, &frozen_edge);
    full - interior
}

/// Two-edge cell version of [`moving_limit_boundary_tower_theta_integrand`]:
/// the exact boundary-flux tower of `∫_{z_L(θ)}^{z_R(θ)} G(z;θ) dz` with a
/// θ-dependent integrand, `Φ(z_R;θ) − Φ(z_L;θ)` minus the pure-θ parts at each
/// frozen edge. A `Fixed` edge passes a `z_edge` with zero derivative channels,
/// so its `full` and `interior` substitutions coincide and it contributes
/// nothing — matching the production `edge_vel = 0` short-circuit.
pub fn cell_moving_boundary_flux_tower_theta_integrand<const K1: usize, const K: usize>(
    phi_jet_right: &Tower4<K1>,
    z_right: &Tower4<K>,
    phi_jet_left: &Tower4<K1>,
    z_left: &Tower4<K>,
) -> Tower4<K> {
    moving_limit_boundary_tower_theta_integrand(phi_jet_right, z_right)
        - moving_limit_boundary_tower_theta_integrand(phi_jet_left, z_left)
}

// ── The program seam ─────────────────────────────────────────────────

/// A family's row negative log-likelihood written ONCE over tower scalars.
///
/// This is the single source of truth #932 asks for: the value channel of
/// the returned tower must BE the production row NLL (same branches, same
/// guards, same numerics), and every derivative channel is then exact by
/// construction. The linear Jacobian wiring (coefficients ↔ primaries) is
/// NOT part of this trait — it is family data, not calculus, and stays on
/// the `RowKernel` implementor.
pub trait RowNllProgram<const K: usize>: Send + Sync {
    /// Number of observations the program covers.
    fn n_rows(&self) -> usize;

    /// Current primary-scalar values for `row` (where to seed the tower).
    fn primaries(&self, row: usize) -> Result<[f64; K], String>;

    /// The row NLL evaluated on tower scalars. `p[a]` arrives pre-seeded as
    /// variable `a` at the current primary value; implementations combine
    /// them with `Tower4` arithmetic and per-row data (response, censoring
    /// indicators, offsets) entering as constants.
    fn row_nll(&self, row: usize, p: &[Tower4<K>; K]) -> Result<Tower4<K>, String>;
}

/// Evaluate a program's full tower at the current primaries for one row.
///
/// One call yields every `RowKernel` calculus channel; callers that need
/// several contractions of the same row should hold the returned tower and
/// contract repeatedly rather than re-evaluating.
pub fn evaluate_program<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<Tower4<K>, String> {
    let p = prog.primaries(row)?;
    let vars: [Tower4<K>; K] = std::array::from_fn(|a| Tower4::variable(p[a], a));
    prog.row_nll(row, &vars)
}

/// Mechanically derived `row_kernel` channel: `(nll, ∇, H)`.
pub fn derived_row_kernel<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<(f64, [f64; K], [[f64; K]; K]), String> {
    let t = evaluate_program(prog, row)?;
    Ok((t.v, t.g, t.h))
}

/// Mechanically derived `row_third_contracted` channel.
pub fn derived_third_contracted<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    Ok(evaluate_program(prog, row)?.third_contracted(dir))
}

/// Mechanically derived `row_fourth_contracted` channel.
pub fn derived_fourth_contracted<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir_u: &[f64; K],
    dir_v: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    Ok(evaluate_program(prog, row)?.fourth_contracted(dir_u, dir_v))
}

// ── The generic program seam (#932 scalar cutover) ───────────────────

/// A family's row negative log-likelihood written ONCE over the generic
/// [`super::jet_scalar::JetScalar`] interface, so the SAME expression can be
/// re-instantiated at whatever order / representation a consumer needs
/// ([`super::jet_scalar::Order2`] for `(v, g, H)`,
/// [`super::jet_scalar::OneSeed`] for the contracted third,
/// [`super::jet_scalar::TwoSeed`] for the contracted fourth, or the full
/// [`Tower4`] for every channel at once).
///
/// This is additive to [`RowNllProgram`] (which is `Tower4`-specialised): a
/// program implementing this generic trait gets the small contracted scalars for
/// free, dissolving the dense-`Tower4<9>` cost objection in the location-scale
/// gates (doc §A.4). An existing `Tower4`-only [`RowNllProgram`] continues to
/// work unchanged; new families should prefer this generic trait.
///
/// Follow-up (#932): each existing [`RowNllProgram`] impl can migrate to this
/// trait by rewriting its `row_nll` body generically over `S: JetScalar<K>` (the bodies
/// already use only `add`/`sub`/`mul`/`scale`/`exp`/`ln`/… which the generic
/// trait provides) and deleting the `Tower4`-specialised method, then routing
/// the directional / joint-Hessian gates through the contracted scalars.
pub trait RowNllProgramGeneric<const K: usize>: Send + Sync {
    /// Number of observations the program covers.
    fn n_rows(&self) -> usize;

    /// Current primary-scalar values for `row` (where to seed the scalar).
    fn primaries(&self, row: usize) -> Result<[f64; K], String>;

    /// The row NLL evaluated on a generic jet scalar. `p[a]` arrives pre-seeded
    /// (base value + per-scalar nilpotent directions) by the caller; the body
    /// uses ONLY [`super::jet_scalar::JetScalar`] ops and per-row data
    /// (response, censoring, offsets) entering as constants.
    fn row_nll_generic<S: super::jet_scalar::JetScalar<K>>(
        &self,
        row: usize,
        p: &[S; K],
    ) -> Result<S, String>;
}

// ── The oracle ───────────────────────────────────────────────────────

/// One row's worth of hand-written kernel outputs, as claimed by a
/// `RowKernel` implementation, packaged for verification against the
/// tower truth. Plain data (no trait coupling) so any kernel — whatever
/// its visibility — can be audited from its own test module.
pub struct KernelChannels<const K: usize> {
    /// Claimed `(nll, ∇, H)` from `row_kernel`.
    pub value: f64,
    /// Claimed gradient.
    pub gradient: [f64; K],
    /// Claimed Hessian.
    pub hessian: [[f64; K]; K],
    /// Claimed `row_third_contracted(dir)` outputs as `(dir, claim)` pairs.
    pub third: Vec<([f64; K], [[f64; K]; K])>,
    /// Claimed `row_fourth_contracted(u, v)` outputs as `(u, v, claim)`.
    pub fourth: Vec<([f64; K], [f64; K], [[f64; K]; K])>,
}

/// Channel-by-channel audit of a hand-written kernel against the
/// single-expression tower truth. Returns `Err` naming the first channel,
/// index, claimed and true values on disagreement — designed as the body
/// of the per-family CI oracle tests (#932 deployment step 2).
///
/// Tolerance is PER ENTRY, mixed absolute/relative: each comparison uses
/// `|claim − truth| ≤ atol + rel_tol · max(|claim|, |truth|)`. The absolute
/// floor `atol = rel_tol` lets exact-zero entries of structurally sparse
/// towers pass without demanding bit-equality, while a tiny cross-block
/// entry dropped next to a huge one is still caught (it is NOT measured
/// against the largest entry of the whole channel — there is no per-channel
/// magnitude floor). Genuine sign flips (#736) and dropped channels are loud.
///
/// Non-finite handling is strict: a NaN on either side always fails; an
/// infinity passes only when both sides are the SAME signed infinity.
pub fn verify_kernel_channels<const K: usize>(
    tower: &Tower4<K>,
    claims: &KernelChannels<K>,
    rel_tol: f64,
) -> Result<(), String> {
    // Absolute floor: reuse rel_tol so a single knob controls both the
    // relative band and the absolute floor for entries near zero.
    let atol = rel_tol;
    let check = |label: &str, claim: f64, truth: f64| -> Result<(), String> {
        // Non-finite values never silently pass the algebraic comparison
        // below (any comparison with NaN is false). Handle them explicitly:
        // NaN on either side always errs; an infinity passes only if both
        // sides are the identical signed infinity.
        if !claim.is_finite() || !truth.is_finite() {
            let agree = claim.is_infinite()
                && truth.is_infinite()
                && claim.is_sign_positive() == truth.is_sign_positive();
            if agree {
                return Ok(());
            }
            return Err(format!(
                "row-kernel oracle: {label} non-finite mismatch: claimed {claim:+.12e}, tower {truth:+.12e}"
            ));
        }
        let band = atol + rel_tol * claim.abs().max(truth.abs());
        if (claim - truth).abs() > band {
            return Err(format!(
                "row-kernel oracle: {label} disagrees: claimed {claim:+.12e}, tower {truth:+.12e} (rel_tol {rel_tol:.1e}, atol {atol:.1e}, band {band:.3e})"
            ));
        }
        Ok(())
    };

    check("value", claims.value, tower.v)?;

    for a in 0..K {
        check(&format!("gradient[{a}]"), claims.gradient[a], tower.g[a])?;
    }

    for a in 0..K {
        for b in 0..K {
            check(
                &format!("hessian[{a}][{b}]"),
                claims.hessian[a][b],
                tower.h[a][b],
            )?;
        }
    }

    for (t_idx, (dir, claim)) in claims.third.iter().enumerate() {
        let truth = tower.third_contracted(dir);
        for a in 0..K {
            for b in 0..K {
                check(
                    &format!("third[{t_idx}][{a}][{b}]"),
                    claim[a][b],
                    truth[a][b],
                )?;
            }
        }
    }

    for (f_idx, (u, w, claim)) in claims.fourth.iter().enumerate() {
        let truth = tower.fourth_contracted(u, w);
        for a in 0..K {
            for b in 0..K {
                check(
                    &format!("fourth[{f_idx}][{a}][{b}]"),
                    claim[a][b],
                    truth[a][b],
                )?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Binomial-logit row NLL, K=1: ℓ(η) = ln(1 + e^η) − y·η.
    /// The entire tower has textbook closed forms in μ = σ(η); this test
    /// pins the algebra (exp, ln, scalar mixes, Leibniz/Faà di Bruno) to
    /// analytic truth at near-machine precision.
    struct LogitProgram {
        eta: Vec<f64>,
        y: Vec<f64>,
    }

    impl RowNllProgram<1> for LogitProgram {
        fn n_rows(&self) -> usize {
            self.eta.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 1], String> {
            Ok([self.eta[row]])
        }
        fn row_nll(&self, row: usize, p: &[Tower4<1>; 1]) -> Result<Tower4<1>, String> {
            let eta = p[0];
            Ok((eta.exp() + 1.0).ln() - eta * self.y[row])
        }
    }

    #[test]
    fn logit_tower_matches_closed_forms() {
        let prog = LogitProgram {
            eta: vec![-2.3, -0.4, 0.0, 0.9, 3.1],
            y: vec![1.0, 0.0, 1.0, 0.0, 1.0],
        };
        for row in 0..prog.n_rows() {
            let t = evaluate_program(&prog, row).expect("logit program");
            let eta = prog.eta[row];
            let y = prog.y[row];
            let mu = 1.0 / (1.0 + (-eta).exp());
            let w = mu * (1.0 - mu);
            let expect = [
                (t.v, (1.0 + eta.exp()).ln() - y * eta, "value"),
                (t.g[0], mu - y, "grad"),
                (t.h[0][0], w, "hess"),
                (t.t3[0][0][0], w * (1.0 - 2.0 * mu), "third"),
                (
                    t.t4[0][0][0][0],
                    w * (1.0 - 6.0 * mu + 6.0 * mu * mu),
                    "fourth",
                ),
            ];
            for (got, want, label) in expect {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "row {row} {label}: got {got:+.15e} want {want:+.15e}"
                );
            }
        }
    }

    fn assert_close(label: &str, got: f64, want: f64, rel_tol: f64) {
        let diff = (got - want).abs();
        assert!(
            diff <= rel_tol * want.abs().max(1.0),
            "{label}: got {got:+.17e} want {want:+.17e} diff {diff:.3e}"
        );
    }

    #[test]
    fn gamma_special_function_stacks_match_reference_values() {
        const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
        let pi_sq = std::f64::consts::PI * std::f64::consts::PI;
        let cases = [
            (
                "x=0.1",
                0.1,
                -10.423_754_940_411_076,
                101.433_299_150_792_75,
            ),
            (
                "x=0.5",
                0.5,
                -EULER_GAMMA - 2.0 * std::f64::consts::LN_2,
                pi_sq / 2.0,
            ),
            ("x=1", 1.0, -EULER_GAMMA, pi_sq / 6.0),
            (
                "x=2.5",
                2.5,
                -EULER_GAMMA - 2.0 * std::f64::consts::LN_2 + 2.0 + 2.0 / 3.0,
                pi_sq / 2.0 - 4.0 - 4.0 / 9.0,
            ),
            (
                "x=50",
                50.0,
                3.901_989_673_427_892,
                0.020_201_333_226_697_128,
            ),
        ];

        for (label, x, digamma_ref, trigamma_ref) in cases {
            let ln_gamma_stack = ln_gamma_derivative_stack(x);
            let digamma_stack = digamma_derivative_stack(x);
            let trigamma_stack = trigamma_derivative_stack(x);
            assert_close(
                &format!("{label} ln_gamma_stack digamma"),
                ln_gamma_stack[1],
                digamma_ref,
                1e-13,
            );
            assert_close(
                &format!("{label} digamma value"),
                digamma_stack[0],
                digamma_ref,
                1e-13,
            );
            assert_close(
                &format!("{label} ln_gamma_stack trigamma"),
                ln_gamma_stack[2],
                trigamma_ref,
                1e-13,
            );
            assert_close(
                &format!("{label} digamma_stack trigamma"),
                digamma_stack[1],
                trigamma_ref,
                1e-13,
            );
            assert_close(
                &format!("{label} trigamma value"),
                trigamma_stack[0],
                trigamma_ref,
                1e-13,
            );
        }
    }

    #[test]
    fn gamma_special_function_stacks_obey_recurrences() {
        for x in [0.1, 0.5, 1.0, 2.5, 50.0] {
            let digamma_x = digamma_derivative_stack(x)[0];
            let digamma_next = digamma_derivative_stack(x + 1.0)[0];
            let trigamma_x = trigamma_derivative_stack(x)[0];
            let trigamma_next = trigamma_derivative_stack(x + 1.0)[0];
            assert_close(
                &format!("digamma recurrence x={x}"),
                digamma_next,
                digamma_x + 1.0 / x,
                1e-13,
            );
            assert_close(
                &format!("trigamma recurrence x={x}"),
                trigamma_next,
                trigamma_x - 1.0 / (x * x),
                1e-13,
            );
        }
    }

    /// Gaussian location-scale row NLL, K=2 primaries (η, s = log σ):
    /// ℓ = s + ½ e^{−2s} (y − η)². Mixed cross blocks — the #736 fragility
    /// shape — all have one-line closed forms here.
    struct LocScaleProgram {
        eta: Vec<f64>,
        s: Vec<f64>,
        y: Vec<f64>,
    }

    impl RowNllProgram<2> for LocScaleProgram {
        fn n_rows(&self) -> usize {
            self.eta.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
            Ok([self.eta[row], self.s[row]])
        }
        fn row_nll(&self, row: usize, p: &[Tower4<2>; 2]) -> Result<Tower4<2>, String> {
            let r = -(p[0] - self.y[row]);
            Ok(p[1] + (p[1] * (-2.0)).exp() * r * r * 0.5)
        }
    }

    #[test]
    fn locscale_tower_matches_closed_forms_including_cross_blocks() {
        let prog = LocScaleProgram {
            eta: vec![0.3, -1.1, 2.0],
            s: vec![-0.5, 0.2, 0.8],
            y: vec![1.0, -2.0, 2.5],
        };
        let tol = 1e-12;
        for row in 0..prog.n_rows() {
            let t = evaluate_program(&prog, row).expect("locscale program");
            let r = prog.y[row] - prog.eta[row];
            let w = (-2.0 * prog.s[row]).exp();
            // (η, s) = indices (0, 1).
            let truth_g = [-w * r, 1.0 - w * r * r];
            let truth_h = [[w, 2.0 * w * r], [2.0 * w * r, 2.0 * w * r * r]];
            // Third tensor: distinct-entry closed forms.
            // ∂ηηη = 0, ∂ηηs = −2w, ∂ηss = −4wr, ∂sss = −4wr².
            let t3_truth = |a: usize, b: usize, c: usize| -> f64 {
                match a + b + c {
                    0 => 0.0,
                    1 => -2.0 * w,
                    2 => -4.0 * w * r,
                    _ => -4.0 * w * r * r,
                }
            };
            // Fourth tensor: ∂ηηηη = 0, ∂ηηηs = 0? No: d/ds(∂ηηη)=0 ✓;
            // ∂ηηss = 4w, ∂ηsss = 8wr, ∂ssss = 8wr².
            let t4_truth = |a: usize, b: usize, c: usize, d: usize| -> f64 {
                match a + b + c + d {
                    0 | 1 => 0.0,
                    2 => 4.0 * w,
                    3 => 8.0 * w * r,
                    _ => 8.0 * w * r * r,
                }
            };
            for a in 0..2 {
                assert!(
                    (t.g[a] - truth_g[a]).abs() <= tol * truth_g[a].abs().max(1.0),
                    "row {row} grad[{a}]"
                );
                for b in 0..2 {
                    assert!(
                        (t.h[a][b] - truth_h[a][b]).abs() <= tol * w.max(1.0) * (1.0 + r.abs()),
                        "row {row} hess[{a}][{b}]: got {} want {}",
                        t.h[a][b],
                        truth_h[a][b]
                    );
                    for c in 0..2 {
                        assert!(
                            (t.t3[a][b][c] - t3_truth(a, b, c)).abs()
                                <= tol * 8.0 * w.max(1.0) * (1.0 + r.abs() + r * r),
                            "row {row} t3[{a}][{b}][{c}]: got {} want {}",
                            t.t3[a][b][c],
                            t3_truth(a, b, c)
                        );
                        for d in 0..2 {
                            assert!(
                                (t.t4[a][b][c][d] - t4_truth(a, b, c, d)).abs()
                                    <= tol * 16.0 * w.max(1.0) * (1.0 + r.abs() + r * r),
                                "row {row} t4[{a}][{b}][{c}][{d}]: got {} want {}",
                                t.t4[a][b][c][d],
                                t4_truth(a, b, c, d)
                            );
                        }
                    }
                }
            }
            // The derived trait-surface helpers agree with direct contraction.
            let dir = [0.7, -1.3];
            let third = derived_third_contracted(&prog, row, &dir).expect("third");
            for a in 0..2 {
                for b in 0..2 {
                    let want = t.t3[a][b][0] * dir[0] + t.t3[a][b][1] * dir[1];
                    assert!((third[a][b] - want).abs() <= 1e-13 * want.abs().max(1.0));
                }
            }
        }
    }

    /// FD cross-check on a deliberately gnarly composition (div, sqrt,
    /// powf, nested exp/ln) in K=3, where no closed form is consulted:
    /// every tower channel is checked against central finite differences
    /// of the channel one order below — value→grad, grad→hess, hess→t3,
    /// t3→t4 — so each order is independently anchored.
    ///
    /// The program carries a per-row primary fixture plus a per-row offset
    /// `tau[row]` that enters the loss as a constant, so `row` genuinely
    /// drives both the seed point and the evaluated expression.
    struct GnarlyProgram {
        primaries: Vec<[f64; 3]>,
        tau: Vec<f64>,
    }

    impl GnarlyProgram {
        fn fixture() -> Self {
            Self {
                primaries: vec![[0.4, -0.7, 1.2], [-0.9, 0.6, 0.3], [1.1, -0.2, -0.8]],
                tau: vec![0.15, -0.35, 0.5],
            }
        }
    }

    impl RowNllProgram<3> for GnarlyProgram {
        fn n_rows(&self) -> usize {
            self.primaries.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
            self.primaries
                .get(row)
                .copied()
                .ok_or_else(|| format!("gnarly: row {row} out of range"))
        }
        fn row_nll(&self, row: usize, p: &[Tower4<3>; 3]) -> Result<Tower4<3>, String> {
            let tau = *self
                .tau
                .get(row)
                .ok_or_else(|| format!("gnarly: tau row {row} out of range"))?;
            let a = (p[0] * p[1]).exp();
            let b = (p[2] * p[2] + 1.0).sqrt();
            let c = (a + b + tau).ln();
            let d = (p[1] * 0.5 + 2.0).powf(1.7);
            Ok(c / d + (p[0] - p[2]) * (p[0] - p[2]) * 0.25)
        }
    }

    /// Evaluate the gnarly program's tower at an ARBITRARY seed point for
    /// `row` (used to drive central differences off the fixture grid),
    /// while keeping `row`'s per-row data (`tau`) in the loss.
    fn gnarly_tower_at(prog: &GnarlyProgram, row: usize, p: [f64; 3]) -> Tower4<3> {
        struct At<'a> {
            base: &'a GnarlyProgram,
            row: usize,
            p: [f64; 3],
        }
        impl RowNllProgram<3> for At<'_> {
            fn n_rows(&self) -> usize {
                1
            }
            fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
                if row != 0 {
                    return Err(format!("gnarly-at: row {row} out of range"));
                }
                Ok(self.p)
            }
            fn row_nll(&self, eval_row: usize, vars: &[Tower4<3>; 3]) -> Result<Tower4<3>, String> {
                if eval_row != 0 {
                    return Err(format!("gnarly-at: eval row {eval_row} out of range"));
                }
                self.base.row_nll(self.row, vars)
            }
        }
        evaluate_program(&At { base: prog, row, p }, 0).expect("gnarly tower")
    }

    #[test]
    fn gnarly_tower_is_fd_consistent_order_by_order() {
        let prog = GnarlyProgram::fixture();
        for row in 0..prog.n_rows() {
            let base = prog.primaries(row).expect("primaries");
            let t = gnarly_tower_at(&prog, row, base);
            let h_step = 1e-5;
            let tol = 1e-6;
            for c in 0..3 {
                let mut up = base;
                let mut dn = base;
                up[c] += h_step;
                dn[c] -= h_step;
                let t_up = gnarly_tower_at(&prog, row, up);
                let t_dn = gnarly_tower_at(&prog, row, dn);
                // value → gradient.
                let fd_g = (t_up.v - t_dn.v) / (2.0 * h_step);
                assert!(
                    (t.g[c] - fd_g).abs() <= tol * fd_g.abs().max(1.0),
                    "grad[{c}]: analytic {} fd {}",
                    t.g[c],
                    fd_g
                );
                for a in 0..3 {
                    // gradient → Hessian.
                    let fd_h = (t_up.g[a] - t_dn.g[a]) / (2.0 * h_step);
                    assert!(
                        (t.h[a][c] - fd_h).abs() <= tol * fd_h.abs().max(1.0),
                        "hess[{a}][{c}]: analytic {} fd {}",
                        t.h[a][c],
                        fd_h
                    );
                    for b in 0..3 {
                        // Hessian → third.
                        let fd_t3 = (t_up.h[a][b] - t_dn.h[a][b]) / (2.0 * h_step);
                        assert!(
                            (t.t3[a][b][c] - fd_t3).abs() <= tol * fd_t3.abs().max(1.0),
                            "t3[{a}][{b}][{c}]: analytic {} fd {}",
                            t.t3[a][b][c],
                            fd_t3
                        );
                        for d in 0..3 {
                            // third → fourth.
                            let fd_t4 = (t_up.t3[a][b][d] - t_dn.t3[a][b][d]) / (2.0 * h_step);
                            assert!(
                                (t.t4[a][b][d][c] - fd_t4).abs() <= tol * fd_t4.abs().max(1.0),
                                "t4[{a}][{b}][{d}][{c}]: analytic {} fd {}",
                                t.t4[a][b][d][c],
                                fd_t4
                            );
                        }
                    }
                }
            }
        }
    }

    /// `implicit_solve` reproduces the true implicit function `a(θ)` of a
    /// constraint `F(a, θ) = 0` to fourth order. The constraint here is the
    /// smooth, strictly-`a`-monotone
    ///   F(a, θ) = a + θ₀·a² + θ₁·exp(a) − c
    /// whose root `a(θ)` is re-solved by scalar Newton at perturbed θ as the
    /// independent finite-difference oracle. Mirrors the survival flex
    /// calibration solve (one implicit intercept over the primaries) without
    /// any survival machinery, so a failure localises to the combinator.
    #[test]
    fn implicit_solve_matches_scalar_resolve_to_fourth_order() {
        const C: f64 = 1.7;
        // The scalar constraint as a plain f64 closure (the production root
        // finder analogue) and its tower form in (a, θ₀, θ₁).
        let f_scalar = |a: f64, th: [f64; 2]| a + th[0] * a * a + th[1] * a.exp() - C;
        let f_da = |a: f64, th: [f64; 2]| 1.0 + 2.0 * th[0] * a + th[1] * a.exp();
        let solve = |th: [f64; 2]| -> f64 {
            let mut a = 0.0_f64;
            for _ in 0..100 {
                let r = f_scalar(a, th);
                if r.abs() < 1e-14 {
                    break;
                }
                a -= r / f_da(a, th);
            }
            a
        };
        // Tower constraint over K1 = 3 vars: slot 0 = a, slots 1,2 = θ₀, θ₁.
        let f_tower = |a0: f64, th: [f64; 2]| -> Tower4<3> {
            let a = Tower4::<3>::variable(a0, 0);
            let t0 = Tower4::<3>::variable(th[0], 1);
            let t1 = Tower4::<3>::variable(th[1], 2);
            a + t0 * a.mul(&a) + t1 * a.exp() - C
        };

        let th0 = [0.35, 0.2];
        let a0 = solve(th0);
        let f = f_tower(a0, th0);
        // Residual at the solved point is ~0 (the combinator tolerates the
        // production Newton residual; here it is machine-zero).
        assert!(f.v.abs() < 1e-12, "constraint residual {:+.3e}", f.v);
        let a_tower: Tower4<2> = implicit_solve::<3, 2>(&f, a0).expect("implicit solve");

        // FD oracle: central differences of the scalar re-solve. Each order is
        // built from the previous via one more central difference, exactly the
        // gnarly order-by-order ladder.
        let h = 1e-4;
        let tol = 1e-5;
        let re = |th: [f64; 2]| solve(th);
        for i in 0..2 {
            let mut up = th0;
            let mut dn = th0;
            up[i] += h;
            dn[i] -= h;
            let fd_g = (re(up) - re(dn)) / (2.0 * h);
            assert!(
                (a_tower.g[i] - fd_g).abs() <= tol * fd_g.abs().max(1.0),
                "a_θ[{i}]: analytic {:+.6e} fd {:+.6e}",
                a_tower.g[i],
                fd_g
            );
            // second order: FD of the analytic gradient component would re-use
            // the combinator; instead difference a SCALAR gradient computed by
            // a nested re-solve so the oracle stays production-independent.
            let grad_at = |th: [f64; 2], j: usize| -> f64 {
                let mut up = th;
                let mut dn = th;
                up[j] += h;
                dn[j] -= h;
                (re(up) - re(dn)) / (2.0 * h)
            };
            for j in 0..2 {
                let fd_h = (grad_at(up, j) - grad_at(dn, j)) / (2.0 * h);
                assert!(
                    (a_tower.h[i][j] - fd_h).abs() <= 1e-3 * fd_h.abs().max(1.0),
                    "a_θθ[{i}][{j}]: analytic {:+.6e} fd {:+.6e}",
                    a_tower.h[i][j],
                    fd_h
                );
            }
        }
    }

    /// `implicit_solve` degenerates to `a_θ = −F_θ / F_a` at first order on a
    /// linear-in-a constraint, and the second-order tensor matches the
    /// textbook IFT formula `a_uv = −(F_uv + F_au a_v + F_av a_u + F_aa a_u a_v)/F_a`.
    /// This pins the recursion against the hand-coded first_full.rs formula it
    /// replaces, independent of any FD step.
    #[test]
    fn implicit_solve_matches_textbook_ift_recursion() {
        // A constraint with non-trivial F_a, F_aa, F_au, F_uv all present.
        let a0 = 0.4_f64;
        let th = [0.25_f64, -0.15_f64];
        let f = {
            let a = Tower4::<3>::variable(a0, 0);
            let t0 = Tower4::<3>::variable(th[0], 1);
            let t1 = Tower4::<3>::variable(th[1], 2);
            // F = a·(1 + θ₀) + θ₁·a² + θ₀·θ₁ − 0.4385. The constant is chosen so
            // F(a0, θ0) = 0 exactly at a0 = 0.4, θ = [0.25, −0.15]:
            //   0.4·1.25 + (−0.15)·0.16 + 0.25·(−0.15) = 0.4385.
            // implicit_solve requires a genuine root; at the root the level-set
            // and root-curve derivatives coincide, so the textbook-IFT
            // assertions below are unaffected.
            a * (t0 + 1.0) + t1 * a.mul(&a) + t0 * t1 - 0.4385
        };
        let a_t = implicit_solve::<3, 2>(&f, a0).expect("solve");
        let f_a = f.g[0];
        // First order: a_u = −F_u / F_a.
        for u in 0..2 {
            let want = -f.g[u + 1] / f_a;
            assert!(
                (a_t.g[u] - want).abs() < 1e-12,
                "a_u[{u}] {:+.6e} vs −F_u/F_a {:+.6e}",
                a_t.g[u],
                want
            );
        }
        // Second order textbook IFT (indices shifted by 1 for the a-slot).
        for u in 0..2 {
            for v in 0..2 {
                let f_uv = f.h[u + 1][v + 1];
                let f_au = f.h[0][u + 1];
                let f_av = f.h[0][v + 1];
                let f_aa = f.h[0][0];
                let want =
                    -(f_uv + f_au * a_t.g[v] + f_av * a_t.g[u] + f_aa * a_t.g[u] * a_t.g[v]) / f_a;
                assert!(
                    (a_t.h[u][v] - want).abs() < 1e-12,
                    "a_uv[{u}][{v}] {:+.6e} vs IFT {:+.6e}",
                    a_t.h[u][v],
                    want
                );
            }
        }
    }

    /// The moving-boundary flux tower reproduces every θ-derivative of a
    /// moving-limit integral, INCLUDING the second-order `B·z_uv` term the
    /// hand-written flux dropped (#932). The edge `z_R(θ) = θ₀ + θ₁²` has a
    /// genuinely nonzero `∂²z_R/∂θ₁² = 2`, so a combinator that omitted
    /// `B·z_uv` would miss the [1][1] Hessian entry. Truth = central FD of the
    /// closed-form integral `∫₀^{z_R} e^{−z²/2} dz = √(π/2)·erf(z_R/√2)`.
    #[test]
    fn moving_boundary_flux_carries_b_zuv_term() {
        use std::f64::consts::PI;
        let b = |z: f64| (-0.5 * z * z).exp(); // integrand B(z)
        // Antiderivative-based closed-form integral I(z_R) = ∫₀^{z_R} B dz.
        let integral = |z_r: f64| (PI / 2.0).sqrt() * libm::erf(z_r / 2.0_f64.sqrt());
        let z_r = |th: [f64; 2]| th[0] + th[1] * th[1];
        let th0 = [0.7_f64, 0.5_f64];

        // Edge tower z_R(θ) over K=2 primaries: value + exact derivatives.
        let mut z_edge = Tower4::<2>::constant(z_r(th0));
        z_edge.g[0] = 1.0; // ∂z_R/∂θ₀ = 1
        z_edge.g[1] = 2.0 * th0[1]; // ∂z_R/∂θ₁ = 2θ₁
        z_edge.h[1][1] = 2.0; // ∂²z_R/∂θ₁² = 2  (the z_uv the old flux dropped)

        // Integrand stack [B, B′, B″, B‴] at z₀: B′=−z·B, B″=(z²−1)·B,
        // B‴=(3z−z³)·B.
        let z0 = z_edge.v;
        let b0 = b(z0);
        let stack = [
            b0,
            -z0 * b0,
            (z0 * z0 - 1.0) * b0,
            (3.0 * z0 - z0 * z0 * z0) * b0,
        ];
        let flux = moving_limit_boundary_tower(&z_edge, stack);

        // FD truth of the integral's derivatives.
        let h = 1e-4;
        let tol = 1e-6;
        for i in 0..2 {
            let mut up = th0;
            let mut dn = th0;
            up[i] += h;
            dn[i] -= h;
            let fd_g = (integral(z_r(up)) - integral(z_r(dn))) / (2.0 * h);
            assert!(
                (flux.g[i] - fd_g).abs() <= tol * fd_g.abs().max(1.0),
                "flux_g[{i}]: analytic {:+.8e} fd {:+.8e}",
                flux.g[i],
                fd_g
            );
        }
        // The decisive entry: ∂²I/∂θ₁² = B′·(z_θ₁)² + B·z_θ₁θ₁. With z_θ₁=2θ₁=1
        // and z_θ₁θ₁=2, the B·z_uv contribution is B(z₀)·2 — omitting it would
        // leave the [1][1] entry short by exactly 2·B(z₀).
        let grad1_at = |th: [f64; 2]| -> f64 {
            let mut up = th;
            let mut dn = th;
            up[1] += h;
            dn[1] -= h;
            (integral(z_r(up)) - integral(z_r(dn))) / (2.0 * h)
        };
        let mut up = th0;
        let mut dn = th0;
        up[1] += h;
        dn[1] -= h;
        let fd_h11 = (grad1_at(up) - grad1_at(dn)) / (2.0 * h);
        assert!(
            (flux.h[1][1] - fd_h11).abs() <= 1e-3 * fd_h11.abs().max(1.0),
            "flux_h[1][1] (carries B·z_uv): analytic {:+.8e} fd {:+.8e}",
            flux.h[1][1],
            fd_h11
        );
        // Explicit witness that the B·z_uv term is present and material:
        // analytic h[1][1] minus the pure (z_u)² part must equal B·z_uv = 2·B₀.
        let pure_zu2 = stack[1] * z_edge.g[1] * z_edge.g[1];
        let b_zuv = flux.h[1][1] - pure_zu2;
        assert!(
            (b_zuv - b0 * 2.0).abs() < 1e-10,
            "B·z_uv term {:+.8e} != B₀·z_uv {:+.8e}",
            b_zuv,
            b0 * 2.0
        );
    }

    /// `moving_limit_boundary_tower_theta_integrand` reproduces the marginal-slope
    /// flex boundary closure for a θ-DEPENDENT integrand `G(z;θ)` — the case the
    /// plain `moving_limit_boundary_tower` cannot express, and the case the
    /// survival directional/bidirectional paths hand-assemble term-by-term
    /// (`G·z_uv + G_z·z_u·z_v + G_θu·z_v + G_θv·z_u`, with the directional path
    /// dropping `G·z_uv`). Two independent oracles:
    ///   (1) closed-form: the boundary flux of `∫ G dz` is exactly
    ///       `Φ(z_edge(θ);θ) − Φ(z₀;θ)` (Φ = z-antiderivative of G), whose θ
    ///       derivatives we take by central FD of the closed form — no jet code.
    ///   (2) the explicit second-order hand closure, including the `G·z_uv` term,
    ///       built from the integrand's own (z,θ) partials.
    /// G(z;θ) = exp(z·θ₀) is genuinely θ-dependent (G_θ₀ = z·e^{zθ₀} ≠ 0), and
    /// the edge z_edge = z₀ + θ₀ + θ₁² has a real z_uv = ∂²/∂θ₁² = 2, so a
    /// combinator that dropped either the integrand-θ terms or `G·z_uv` would
    /// miss a Hessian entry.
    #[test]
    fn moving_boundary_theta_integrand_matches_handpath_and_closed_form() {
        // G(z;θ) = exp(z·θ₀);  Φ(z;θ) = ∫₀^z G = (e^{zθ₀} − 1)/θ₀.
        let g = |z: f64, t0: f64| (z * t0).exp();
        let phi = |z: f64, t0: f64| ((z * t0).exp() - 1.0) / t0;
        let z_r = |th: [f64; 2]| 0.6 + th[0] + th[1] * th[1];
        let th0 = [0.4_f64, 0.5_f64];
        let z0 = z_r(th0);

        // Edge tower z_edge(θ) over K=2 primaries.
        let mut z_edge = Tower4::<2>::constant(z0);
        z_edge.g[0] = 1.0; // ∂z/∂θ₀
        z_edge.g[1] = 2.0 * th0[1]; // ∂z/∂θ₁
        z_edge.h[1][1] = 2.0; // ∂²z/∂θ₁² (the z_uv the directional path drops)

        // Φ's mixed (z, θ) jet over K1 = 3 vars: slot 0 = z, slots 1,2 = θ₀,θ₁.
        // Built ONCE in tower arithmetic so every (z^i θ^j) partial is exact.
        let z_var = Tower4::<3>::variable(z0, 0);
        let t0_var = Tower4::<3>::variable(th0[0], 1);
        // θ₁ does not enter G/Φ here, but seed it so the jet carries the full
        // K1 frame (its Φ-derivatives are zero; the z_edge chain supplies all θ₁
        // motion through slot 0).
        let _t1_var = Tower4::<3>::variable(th0[1], 2);
        let phi_jet = ((z_var * t0_var).exp() - 1.0) / t0_var;
        // Sanity: slot-0 first derivative of Φ IS G(z₀;θ₀).
        assert!(
            (phi_jet.g[0] - g(z0, th0[0])).abs() < 1e-12,
            "Φ_z {:+.8e} != G {:+.8e}",
            phi_jet.g[0],
            g(z0, th0[0])
        );

        let flux = moving_limit_boundary_tower_theta_integrand::<3, 2>(&phi_jet, &z_edge);

        // Value channel is 0 by construction (boundary, not the integral itself).
        assert!(
            flux.v.abs() < 1e-12,
            "boundary value channel {:+.3e}",
            flux.v
        );

        // Oracle (1): central FD of the closed-form boundary flux
        //   Bnd(θ) = Φ(z_edge(θ); θ) − Φ(z₀; θ)   (z₀ FROZEN at the base edge).
        let bnd = |th: [f64; 2]| phi(z_r(th), th[0]) - phi(z0, th[0]);
        let h = 1e-4;
        let tol = 1e-6;
        for i in 0..2 {
            let mut up = th0;
            let mut dn = th0;
            up[i] += h;
            dn[i] -= h;
            let fd_g = (bnd(up) - bnd(dn)) / (2.0 * h);
            assert!(
                (flux.g[i] - fd_g).abs() <= tol * fd_g.abs().max(1.0),
                "boundary_g[{i}] analytic {:+.8e} fd {:+.8e}",
                flux.g[i],
                fd_g
            );
        }
        let grad_at = |th: [f64; 2], j: usize| -> f64 {
            let mut up = th;
            let mut dn = th;
            up[j] += h;
            dn[j] -= h;
            (bnd(up) - bnd(dn)) / (2.0 * h)
        };
        for i in 0..2 {
            for j in 0..2 {
                let mut up = th0;
                let mut dn = th0;
                up[i] += h;
                dn[i] -= h;
                let fd_h = (grad_at(up, j) - grad_at(dn, j)) / (2.0 * h);
                assert!(
                    (flux.h[i][j] - fd_h).abs() <= 1e-3 * fd_h.abs().max(1.0),
                    "boundary_h[{i}][{j}] analytic {:+.8e} fd {:+.8e}",
                    flux.h[i][j],
                    fd_h
                );
            }
        }

        // Oracle (2): the explicit second-order hand closure, term by term —
        // `G·z_uv + G_z·z_u·z_v + G_θu·z_v + G_θv·z_u`. Read G's partials at the
        // base point directly (no jet): G = e^{zθ₀}, G_z = θ₀·G, G_θ₀ = z·G,
        // G_θ₁ = 0.
        let gg = g(z0, th0[0]);
        let g_z = th0[0] * gg;
        let g_theta = [z0 * gg, 0.0]; // [G_θ₀, G_θ₁]
        for i in 0..2 {
            for j in 0..2 {
                let z_u = z_edge.g[i];
                let z_v = z_edge.g[j];
                let z_uv = z_edge.h[i][j];
                let hand = gg * z_uv + g_z * z_u * z_v + g_theta[i] * z_v + g_theta[j] * z_u;
                assert!(
                    (flux.h[i][j] - hand).abs() < 1e-9,
                    "boundary_h[{i}][{j}] {:+.8e} != hand closure {:+.8e}",
                    flux.h[i][j],
                    hand
                );
            }
        }

        // Decisive: the `G·z_uv` term the directional path DROPS is present and
        // material in the [1][1] entry (z_uv = 2 there).
        let pure_no_zuv = g_z * z_edge.g[1] * z_edge.g[1] + 2.0 * g_theta[1] * z_edge.g[1];
        let g_zuv = flux.h[1][1] - pure_no_zuv;
        assert!(
            (g_zuv - gg * 2.0).abs() < 1e-9,
            "G·z_uv term {:+.8e} != G₀·z_uv {:+.8e}",
            g_zuv,
            gg * 2.0
        );
    }

    /// The survival crossing-edge position tower `z_edge = (τ − a(θ)) / b`,
    /// `b = exp(g)`, built from the intercept tower `a(θ)` (here a stand-in)
    /// and the seeded slope `g`, reproduces taylor-jet's exact hand-path
    /// boundary-velocity formulas:
    ///   z_u   = −(a_u + [u==g]·z) / b
    ///   z_uv  = −(a_uv + [u==g]·z_v + [v==g]·z_u) / b
    /// This pins the bridge between `implicit_solve` and
    /// `cell_moving_boundary_flux_tower`: the boundary jet that the production
    /// flex path hand-codes (and dropped `z_uv` from) is exactly `∂²` of this
    /// tower. K=3 reduced frame: slot 0 = a-axis carrier (an arbitrary smooth
    /// a(θ) with nonzero a_u/a_uv), slot 1 = g (the log-slope), slot 2 unused.
    #[test]
    fn crossing_edge_tower_matches_handpath_velocity_formulas() {
        const TAU: f64 = 1.3; // the link-knot crossing threshold τ
        let g_idx = 1usize;
        let g0 = 0.85_f64; // the slope value b (the g-primary IS the slope)
        // Stand-in intercept tower a(θ): nonzero value, gradient, Hessian in the
        // two live axes so a_u and a_uv are both exercised. (In production this
        // comes from implicit_solve; here we plant known derivatives.)
        let mut a = Tower4::<3>::constant(0.45);
        a.g[0] = 0.7;
        a.g[1] = -0.3;
        a.h[0][0] = 0.25;
        a.h[0][1] = 0.11;
        a.h[1][0] = 0.11;
        a.h[1][1] = -0.08;

        // In the survival flex frame the slope `b` IS the g-primary directly
        // (the directional code passes `g` as `b`, and ∂z/∂g uses ∂b/∂g = 1):
        // z_edge = (τ − a) / b with b seeded as the g-axis variable.
        let b = Tower4::<3>::variable(g0, g_idx);
        let z_edge = (Tower4::<3>::constant(TAU) - a) / b;

        let bv = g0;
        let z0 = z_edge.v;
        assert!((z0 - (TAU - 0.45) / bv).abs() < 1e-12);

        // z_u = −(a_u + [u==g]·z) / b.
        for u in 0..2 {
            let direct = if u == g_idx { z0 } else { 0.0 };
            let want = -(a.g[u] + direct) / bv;
            assert!(
                (z_edge.g[u] - want).abs() < 1e-10,
                "z_u[{u}] {:+.8e} vs hand formula {:+.8e}",
                z_edge.g[u],
                want
            );
        }
        // z_uv = −(a_uv + [u==g]·z_v + [v==g]·z_u) / b, using the tower's own
        // first-order z_v/z_u (already verified above).
        for u in 0..2 {
            for v in 0..2 {
                let cross = if u == g_idx { z_edge.g[v] } else { 0.0 }
                    + if v == g_idx { z_edge.g[u] } else { 0.0 };
                let want = -(a.h[u][v] + cross) / bv;
                assert!(
                    (z_edge.h[u][v] - want).abs() < 1e-10,
                    "z_uv[{u}][{v}] {:+.8e} vs hand formula {:+.8e}",
                    z_edge.h[u][v],
                    want
                );
            }
        }
    }

    /// The crossing-edge tower in the CONSTRAINT frame (intercept `a` and
    /// slope `b` BOTH independent — slots 0 and 1) reproduces taylor-jet's
    /// FD-certified bare boundary-velocity constants exactly:
    ///   z_a  = ∂z/∂a   = −1/b
    ///   z_ab = ∂²z/∂a∂b = +1/b²
    ///   z_aa = ∂²z/∂a²  = 0
    ///   z_bb = ∂²z/∂b²  = +2(τ−a)/b³
    /// These are the `f_a`/`f_au`/`f_aa` constraint-jet boundary motions the
    /// production base path drops (and only adds in the dir twins, causing the
    /// #932 desync). Here `a` is independent (NOT yet substituted with a(θ)),
    /// so `z_aa = 0` and there is no `a_uv` chain — `implicit_solve` introduces
    /// that later. Pins the constant before the constraint-tower wiring.
    #[test]
    fn crossing_edge_constraint_frame_matches_bare_velocity_constants() {
        const TAU: f64 = 1.3;
        let a0 = 0.45_f64;
        let b0 = 0.85_f64;
        // Slot 0 = a, slot 1 = b, both seeded independent.
        let a = Tower4::<2>::variable(a0, 0);
        let b = Tower4::<2>::variable(b0, 1);
        let z = (Tower4::<2>::constant(TAU) - a) / b;

        assert!((z.v - (TAU - a0) / b0).abs() < 1e-12);
        assert!((z.g[0] - (-1.0 / b0)).abs() < 1e-12, "z_a {:+.10e}", z.g[0]);
        assert!(
            (z.h[0][1] - 1.0 / (b0 * b0)).abs() < 1e-12,
            "z_ab {:+.10e} vs +1/b² {:+.10e}",
            z.h[0][1],
            1.0 / (b0 * b0)
        );
        assert!(
            z.h[0][0].abs() < 1e-12,
            "z_aa must vanish, got {:+.10e}",
            z.h[0][0]
        );
        let want_zbb = 2.0 * (TAU - a0) / (b0 * b0 * b0);
        assert!(
            (z.h[1][1] - want_zbb).abs() < 1e-12,
            "z_bb {:+.10e} vs 2(τ−a)/b³ {:+.10e}",
            z.h[1][1],
            want_zbb
        );
    }

    /// The oracle harness catches a planted #736-style sign flip in a
    /// cross block and reports the channel by name.
    #[test]
    fn oracle_catches_planted_cross_block_sign_flip() {
        let prog = LocScaleProgram {
            eta: vec![0.3],
            s: vec![-0.5],
            y: vec![1.0],
        };
        let t = evaluate_program(&prog, 0).expect("tower");
        let dir = [0.6, -0.2];
        let mut third = t.third_contracted(&dir);
        let honest = KernelChannels {
            value: t.v,
            gradient: t.g,
            hessian: t.h,
            third: vec![(dir, third)],
            fourth: vec![(dir, [1.0, 0.5], t.fourth_contracted(&dir, &[1.0, 0.5]))],
        };
        verify_kernel_channels(&t, &honest, 1e-10).expect("honest kernel must pass");

        // Plant the #736 flip: negate one mixed cross entry.
        third[0][1] = -third[0][1];
        let flipped = KernelChannels {
            value: t.v,
            gradient: t.g,
            hessian: t.h,
            third: vec![(dir, third)],
            fourth: vec![],
        };
        let err = verify_kernel_channels(&t, &flipped, 1e-10)
            .expect_err("planted sign flip must be caught");
        assert!(
            err.contains("third[0][0][1]"),
            "oracle must name the flipped channel, got: {err}"
        );
    }

    /// The third- and fourth-order tensors must be FULLY symmetric under
    /// index permutation (mixed partials commute). The tower stores them
    /// unsymmetrized, so equal-by-construction is a real invariant of the
    /// Leibniz/Faà di Bruno writes — a cheap typo tripwire. Asserted on a
    /// nontrivial K=3 tower with all of div/sqrt/powf/exp/ln exercised, so
    /// every composition path contributes. Lives in a test (not the hot
    /// per-op path) on purpose.
    #[test]
    fn t3_t4_are_fully_index_symmetric() {
        let prog = GnarlyProgram::fixture();
        // 3! = 6 permutations of three indices.
        let perms3: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        // 4! = 24 permutations of four indices.
        let perms4: [[usize; 4]; 24] = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ];
        for row in 0..prog.n_rows() {
            let t = evaluate_program(&prog, row).expect("gnarly tower");
            let scale_t3 =
                t.t3.iter()
                    .flatten()
                    .flatten()
                    .fold(0.0_f64, |m, x| m.max(x.abs()))
                    .max(1.0);
            let scale_t4 =
                t.t4.iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .fold(0.0_f64, |m, x| m.max(x.abs()))
                    .max(1.0);
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let base = t.t3[i][j][k];
                        let idx = [i, j, k];
                        for p in &perms3 {
                            let permed = t.t3[idx[p[0]]][idx[p[1]]][idx[p[2]]];
                            assert!(
                                (base - permed).abs() <= 1e-12 * scale_t3,
                                "row {row}: t3[{i}][{j}][{k}]={base:+.15e} != \
                                 permuted {permed:+.15e} under {p:?}"
                            );
                        }
                        for l in 0..3 {
                            let base4 = t.t4[i][j][k][l];
                            let idx4 = [i, j, k, l];
                            for p in &perms4 {
                                let permed = t.t4[idx4[p[0]]][idx4[p[1]]][idx4[p[2]]][idx4[p[3]]];
                                assert!(
                                    (base4 - permed).abs() <= 1e-12 * scale_t4,
                                    "row {row}: t4[{i}][{j}][{k}][{l}]={base4:+.15e} != \
                                     permuted {permed:+.15e} under {p:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Stable derivative stack for `log Φ(x)` through fourth order.
///
/// The value and Mills ratio come from the shared probit primitive, so the
/// deep left tail uses the same erfcx path as production log-CDF code.
#[inline]
pub(crate) fn unary_derivatives_normal_logcdf(x: f64) -> [f64; 5] {
    let (log_cdf, lambda) = crate::probability::signed_probit_logcdf_and_mills_ratio(x);
    let lambda2 = lambda * lambda;
    let lambda3 = lambda2 * lambda;
    let x2 = x * x;
    [
        log_cdf,
        lambda,
        -lambda * (x + lambda),
        lambda * (x2 - 1.0 + 3.0 * x * lambda + 2.0 * lambda2),
        -lambda
            * ((x * x2 - 3.0 * x) + (7.0 * x2 - 4.0) * lambda + 12.0 * x * lambda2 + 6.0 * lambda3),
    ]
}

/// Stable derivative stack for `log(1 - exp(-x))`, `x > 0`, through fourth order.
#[inline]
pub(crate) fn unary_derivatives_log1mexp_positive(x: f64) -> [f64; 5] {
    let r = 1.0 / x.exp_m1();
    [
        crate::probability::log1mexp_positive(x),
        r,
        -r * (1.0 + r),
        r * (1.0 + r) * (1.0 + 2.0 * r),
        -r * (1.0 + r) * (1.0 + 6.0 * r + 6.0 * r * r),
    ]
}
