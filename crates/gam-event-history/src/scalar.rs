//! Elementary functions over any [`JetField`] scalar, so the exact-marginal
//! event-history likelihood is written once and evaluated as a plain `f64`,
//! as a one-direction dual (`OneSeed<0>`) for `D_θ H[u]`, or as a two-direction
//! dual (`TwoSeed<0>`) for `D²_θ H[u, v]`.
//!
//! Every function composes through [`JetField::compose_unary`] with the exact
//! derivative stack of the outer real function, so no derivative channel is
//! ever approximated.

use gam_math::nested_dual::JetField;

/// `W` independent tangent directions over another jet, every channel itself
/// an `S`. One level carries a block of first derivatives. Two levels,
/// `Rows<Rows<S, W>, W>`, carry the `W × W` block of second derivatives between
/// two blocks of coefficients, each entry still carrying the directions `S` was
/// seeded with, which supplies the third and fourth derivatives LAML reads.
/// Every channel differentiates the *computed* filter, including reference-law
/// evolution and adaptive grid placement.
#[derive(Clone, Debug)]
pub(crate) struct Rows<S, const W: usize> {
    pub base: S,
    pub rows: [S; W],
}

impl<S: JetField, const W: usize> Rows<S, W> {
    /// `base` with derivative `tangents[k]` along direction `k`.
    pub fn seed(base: S, tangents: [f64; W]) -> Self {
        let rows = std::array::from_fn(|k| base.constant_like(tangents[k]));
        Self { base, rows }
    }
}

impl<S: JetField, const W: usize> JetField for Rows<S, W> {
    fn value(&self) -> f64 { self.base.value() }
    fn add(&self, other: &Self) -> Self {
        Self { base: self.base.add(&other.base),
            rows: std::array::from_fn(|k| self.rows[k].add(&other.rows[k])) }
    }
    fn sub(&self, other: &Self) -> Self {
        Self { base: self.base.sub(&other.base),
            rows: std::array::from_fn(|k| self.rows[k].sub(&other.rows[k])) }
    }
    fn neg(&self) -> Self {
        Self { base: self.base.neg(), rows: std::array::from_fn(|k| self.rows[k].neg()) }
    }
    fn scale(&self, factor: f64) -> Self {
        Self { base: self.base.scale(factor),
            rows: std::array::from_fn(|k| self.rows[k].scale(factor)) }
    }
    fn mul(&self, other: &Self) -> Self {
        Self { base: self.base.mul(&other.base),
            rows: std::array::from_fn(|k|
                self.rows[k].mul(&other.base).add(&self.base.mul(&other.rows[k]))) }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let first = self.base.compose_unary([d[1], d[2], d[3], d[4], 0.0]);
        Self { base: self.base.compose_unary(d),
            rows: std::array::from_fn(|k| first.mul(&self.rows[k])) }
    }
    fn constant_like(&self, value: f64) -> Self {
        Self { base: self.base.constant_like(value),
            rows: std::array::from_fn(|_| self.base.constant_like(0.0)) }
    }
    fn with_value(&self, value: f64) -> Self {
        Self { base: self.base.with_value(value), rows: self.rows.clone() }
    }
}

/// The coefficient directions one forward jet sweep carries: the channel count
/// of `Tangent<W>` in the gradient sweeps and of each level of
/// `Rows<Rows<S, W>, W>` in the Hessian block sweeps. It sets how many path
/// evaluations a derivative takes and how many channels each jet scalar holds,
/// never a derivative's value: every channel is formed from the same operands
/// in the same order at any width, so any width gives the same entries bit for
/// bit.
pub(crate) const TANGENT_WIDTH: usize = 8;

/// `exp(x)`.
#[inline]
pub(crate) fn exp<S: JetField>(x: &S) -> S {
    let e = x.value().exp();
    x.compose_unary([e, e, e, e, e])
}

/// `ln(x)` for `x > 0`.
#[inline]
pub(crate) fn ln<S: JetField>(x: &S) -> S {
    let u = x.value();
    let i = 1.0 / u;
    let i2 = i * i;
    x.compose_unary([u.ln(), i, -i2, 2.0 * i2 * i, -6.0 * i2 * i2])
}

/// `sqrt(x)` for `x > 0`.
#[inline]
pub(crate) fn sqrt<S: JetField>(x: &S) -> S {
    let u = x.value();
    let s = u.sqrt();
    let i = 1.0 / u;
    let i2 = i * i;
    x.compose_unary([
        s,
        0.5 * s * i,
        -0.25 * s * i2,
        0.375 * s * i2 * i,
        -0.9375 * s * i2 * i2,
    ])
}

/// `1 / x` for `x != 0`.
#[inline]
pub(crate) fn recip<S: JetField>(x: &S) -> S {
    let i = 1.0 / x.value();
    let i2 = i * i;
    x.compose_unary([i, -i2, 2.0 * i2 * i, -6.0 * i2 * i2, 24.0 * i2 * i2 * i])
}

/// `a / b`.
#[inline]
pub(crate) fn div<S: JetField>(a: &S, b: &S) -> S {
    a.mul(&recip(b))
}

/// `x + c` for a real constant `c`.
#[inline]
pub(crate) fn add_real<S: JetField>(x: &S, c: f64) -> S {
    x.add(&x.constant_like(c))
}

/// `x * x`.
#[inline]
pub(crate) fn square<S: JetField>(x: &S) -> S {
    x.mul(x)
}


/// A first-order forward-mode dual with `W` inline tangent slots: the value
/// plus its derivative along each seeded coefficient direction.
///
/// Every arithmetic step of the forward filter is replayed on this scalar, so
/// the gradient it yields is the exact derivative of the computed
/// log-likelihood, bit-consistent with the value the same code produces on
/// `f64`. That consistency is what a trust-region Newton with a value-based
/// acceptance test requires near an optimum. The slots live inline so the
/// filter allocates nothing; wider coefficient vectors are swept in chunks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tangent<const W: usize> {
    pub value: f64,
    pub grad: [f64; W],
}

impl<const W: usize> Tangent<W> {
    pub(crate) fn seeded(value: f64, grad: [f64; W]) -> Self {
        Self { value, grad }
    }
}

impl<const W: usize> JetField for Tangent<W> {
    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        let mut grad = self.grad;
        for (g, b) in grad.iter_mut().zip(o.grad.iter()) {
            *g += b;
        }
        Self {
            value: self.value + o.value,
            grad,
        }
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        let mut grad = self.grad;
        for (g, b) in grad.iter_mut().zip(o.grad.iter()) {
            *g -= b;
        }
        Self {
            value: self.value - o.value,
            grad,
        }
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        let mut grad = [0.0; W];
        for ((g, a), b) in grad.iter_mut().zip(self.grad.iter()).zip(o.grad.iter()) {
            *g = a * o.value + self.value * b;
        }
        Self {
            value: self.value * o.value,
            grad,
        }
    }
    #[inline]
    fn neg(&self) -> Self {
        let mut grad = self.grad;
        for g in grad.iter_mut() {
            *g = -*g;
        }
        Self {
            value: -self.value,
            grad,
        }
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        let mut grad = self.grad;
        for g in grad.iter_mut() {
            *g *= s;
        }
        Self {
            value: self.value * s,
            grad,
        }
    }
    #[inline]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let mut grad = self.grad;
        for g in grad.iter_mut() {
            *g *= d[1];
        }
        Self { value: d[0], grad }
    }
    #[inline]
    fn constant_like(&self, v: f64) -> Self {
        Self {
            value: v,
            grad: [0.0; W],
        }
    }
    #[inline]
    fn with_value(&self, v: f64) -> Self {
        Self {
            value: v,
            grad: self.grad,
        }
    }
}
