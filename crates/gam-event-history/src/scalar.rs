//! Elementary functions over any [`JetField`] scalar, so the exact-marginal
//! event-history likelihood is written once and evaluated as a plain `f64`,
//! as a one-direction dual (`OneSeed<0>`) for `D_θ H[u]`, or as a two-direction
//! dual (`TwoSeed<0>`) for `D²_θ H[u, v]`.
//!
//! Every function composes through [`JetField::compose_unary`] with the exact
//! derivative stack of the outer real function, so no derivative channel is
//! ever approximated.

use gam_math::nested_dual::JetField;

/// Two independent differentiation directions over another jet. The mixed
/// component differentiates the *computed* filter, including reference-law
/// evolution and adaptive grid placement. Nesting over a directional jet
/// supplies the third and fourth derivatives required by LAML.
#[derive(Clone, Debug)]
pub(crate) struct Mixed<S> {
    pub base: S,
    pub u: S,
    pub v: S,
    pub uv: S,
}

impl<S: JetField> Mixed<S> {
    pub fn seed(base: S, u: f64, v: f64) -> Self {
        Self { u: base.constant_like(u), v: base.constant_like(v),
            uv: base.constant_like(0.0), base }
    }
}

impl<S: JetField> JetField for Mixed<S> {
    fn value(&self) -> f64 { self.base.value() }
    fn add(&self, other: &Self) -> Self {
        Self { base: self.base.add(&other.base), u: self.u.add(&other.u),
            v: self.v.add(&other.v), uv: self.uv.add(&other.uv) }
    }
    fn sub(&self, other: &Self) -> Self { self.add(&other.neg()) }
    fn neg(&self) -> Self { self.scale(-1.0) }
    fn scale(&self, factor: f64) -> Self {
        Self { base: self.base.scale(factor), u: self.u.scale(factor),
            v: self.v.scale(factor), uv: self.uv.scale(factor) }
    }
    fn mul(&self, other: &Self) -> Self {
        Self {
            base: self.base.mul(&other.base),
            u: self.u.mul(&other.base).add(&self.base.mul(&other.u)),
            v: self.v.mul(&other.base).add(&self.base.mul(&other.v)),
            uv: self.uv.mul(&other.base).add(&self.u.mul(&other.v))
                .add(&self.v.mul(&other.u)).add(&self.base.mul(&other.uv)),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let first = self.base.compose_unary([d[1], d[2], d[3], d[4], 0.0]);
        let second = self.base.compose_unary([d[2], d[3], d[4], 0.0, 0.0]);
        Self { base: self.base.compose_unary(d), u: first.mul(&self.u),
            v: first.mul(&self.v),
            uv: first.mul(&self.uv).add(&second.mul(&self.u).mul(&self.v)) }
    }
    fn constant_like(&self, value: f64) -> Self {
        Self::seed(self.base.constant_like(value), 0.0, 0.0)
    }
    fn with_value(&self, value: f64) -> Self {
        Self { base: self.base.with_value(value), ..self.clone() }
    }
}

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
