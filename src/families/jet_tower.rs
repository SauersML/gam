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
/// Tolerance is RELATIVE to a per-channel magnitude floor: each comparison
/// uses `|claim − truth| ≤ rel_tol · max(|truth|, floor)` where `floor`
/// is the largest absolute entry of the true channel — so zero entries of
/// structurally sparse towers don't demand absolute equality, while genuine
/// sign flips (#736) and dropped channels are loud.
pub fn verify_kernel_channels<const K: usize>(
    tower: &Tower4<K>,
    claims: &KernelChannels<K>,
    rel_tol: f64,
) -> Result<(), String> {
    let check = |label: &str, claim: f64, truth: f64, floor: f64| -> Result<(), String> {
        let scale = truth.abs().max(floor).max(1e-300);
        if (claim - truth).abs() > rel_tol * scale {
            return Err(format!(
                "row-kernel oracle: {label} disagrees: claimed {claim:+.12e}, tower {truth:+.12e} (rel_tol {rel_tol:.1e}, scale {scale:.3e})"
            ));
        }
        Ok(())
    };

    check("value", claims.value, tower.v, 1.0)?;

    let g_floor = tower.g.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    for a in 0..K {
        check(
            &format!("gradient[{a}]"),
            claims.gradient[a],
            tower.g[a],
            g_floor,
        )?;
    }

    let h_floor = tower
        .h
        .iter()
        .flatten()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    for a in 0..K {
        for b in 0..K {
            check(
                &format!("hessian[{a}][{b}]"),
                claims.hessian[a][b],
                tower.h[a][b],
                h_floor,
            )?;
        }
    }

    for (t_idx, (dir, claim)) in claims.third.iter().enumerate() {
        let truth = tower.third_contracted(dir);
        let floor = truth.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
        for a in 0..K {
            for b in 0..K {
                check(
                    &format!("third[{t_idx}][{a}][{b}]"),
                    claim[a][b],
                    truth[a][b],
                    floor,
                )?;
            }
        }
    }

    for (f_idx, (u, w, claim)) in claims.fourth.iter().enumerate() {
        let truth = tower.fourth_contracted(u, w);
        let floor = truth.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
        for a in 0..K {
            for b in 0..K {
                check(
                    &format!("fourth[{f_idx}][{a}][{b}]"),
                    claim[a][b],
                    truth[a][b],
                    floor,
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
