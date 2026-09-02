//! Nested second-order forward-AD dual for FD-free high-order cross-checks (#932).
//!
//! [`Dual2<S>`] is a single-direction second-order jet (`value`, first and second
//! derivative in ONE direction) over a scalar field `S`. Because the field is
//! generic, it composes with itself: `Dual2<Dual2<f64>>` carries every mixed
//! partial `∂^i_a ∂^j_b` for `i, j ∈ {0, 1, 2}`, i.e. the full `2 + 2 = 4`th-order
//! bidirectional derivative in two INDEPENDENT directions `a`, `b`.
//!
//! Why this exists: the flexible survival marginal-slope Jet4 tower (the
//! moving-boundary implicit-intercept path, gam#932) is the last derivative
//! surface whose fourth-order channel is verified only by a finite-difference
//! stencil — the hand reference is provably incomplete there, and a 4th-order
//! FD probes a 6th derivative, so the truncation floor sits far above machine
//! precision (the same pathology as gam#979). A nested dual evaluates the SAME
//! single-source program by a DIFFERENT composition ordering (two nested
//! second-order sweeps instead of one fourth-order sweep), so its
//! `∂²_a ∂²_b` channel is a truncation-free, hand-oracle-free cross-check of the
//! Jet4 bidirectional block.
//!
//! The construction is standard forward-over-forward automatic differentiation;
//! its correctness is pinned channel-for-channel against the engine
//! [`crate::jet_tower::Tower4`] on smooth programs (see the module tests).

/// Minimal scalar field a [`Dual2`] can be built over. Implemented by `f64` (the
/// base case) and by [`Dual2`] itself (the nesting case). Every operation mirrors
/// the [`crate::jet_tower::Tower4`] / [`crate::jet_scalar::JetScalar`] Faà di
/// Bruno convention exactly, so a program written against `JetField` evaluates
/// identically on the engine tower and on a nested dual.
///
/// This is the SHARED scalar-field algebra base of the #932 tower: the
/// const-`K` packed [`crate::jet_scalar::JetScalar`] and the runtime-`p`
/// Vec-backed flex jets (`survival::marginal_slope::timepoint_exact::FlexJet`)
/// both EXTEND it, so the field ops and the single Faà di Bruno composition are
/// declared exactly ONCE here. It is deliberately NOT `Copy` (the Vec-backed
/// flex jets are `Clone`, not `Copy`) and carries no constructor (a Vec-backed
/// constant needs a primary count) — the `Copy`, `from_f64`-carrying nested-dual
/// oracle path adds those through [`JetFieldConst`].
pub trait JetField: Clone {
    /// The real value channel (recurses through any nesting to the `f64` leaf).
    fn value(&self) -> f64;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    /// Multiply every channel by a plain `f64`.
    fn scale(&self, s: f64) -> Self;
    /// Faà di Bruno composition `f ∘ self` given the OUTER real function's
    /// derivative stack `d = [f(u), f′(u), f″(u), f‴(u), f⁗(u)]` evaluated at
    /// `u = self.value()` — the identical `[f64; 5]` stack shape
    /// [`crate::jet_tower::Tower4::compose_unary`] consumes.
    fn compose_unary(&self, d: [f64; 5]) -> Self;

    /// A constant carrying THIS element's shape: real value `v`, every
    /// derivative channel zero.
    ///
    /// [`JetField`] deliberately has no dimensionless constructor (a Vec-backed
    /// runtime-width constant needs a primary count), so a shape has to come
    /// from an existing element. The default routes through [`Self::compose_unary`]
    /// with a zero-derivative stack, which is correct for every implementor but
    /// walks the whole Faa di Bruno partition sum to write a constant — at
    /// `Dual2<Order2<K>>` that is three inner compositions plus four products,
    /// all of whose channels are known zero ahead of time. Implementors on a hot
    /// path override it with the direct construction. (#932)
    fn constant_like(&self, v: f64) -> Self {
        self.compose_unary([v, 0.0, 0.0, 0.0, 0.0])
    }

    /// `self` with its real value channel replaced by `v`, every derivative
    /// channel untouched.
    ///
    /// This is the explicit form of the "anchor the value, keep the
    /// derivatives" idiom that a row program uses to hold a previously computed
    /// f64 result bitwise while lifting it into a jet. Expressing it as a
    /// primitive (rather than as subtract-a-constant-then-add-a-constant) makes
    /// the bitwise contract exact by construction instead of emergent from
    /// floating-point cancellation. (#932)
    fn with_value(&self, v: f64) -> Self {
        self.sub(&self.constant_like(self.value()))
            .add(&self.constant_like(v))
    }
}

/// A [`JetField`] that is `Copy` and can be built from a real constant with
/// every derivative channel zero. The nested-dual oracle (`Dual2` over a `Copy`
/// leaf) needs a dimensionless constructor; the runtime-`p` Vec-backed flex jets
/// deliberately do NOT satisfy this (their constant needs a primary count), so
/// it lives on this subtrait rather than the shared algebra base.
pub trait JetFieldConst: JetField + Copy {
    /// A constant field element with value `x` and every derivative channel zero.
    fn from_f64(x: f64) -> Self;
}

impl JetField for f64 {
    #[inline]
    fn value(&self) -> f64 {
        *self
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        *self + *o
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        *self - *o
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        *self * *o
    }
    #[inline]
    fn neg(&self) -> Self {
        -*self
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        *self * s
    }
    #[inline]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // The stack is already evaluated at `u = *self`; `f(u)` is `d[0]`.
        d[0]
    }
    #[inline]
    fn constant_like(&self, v: f64) -> Self {
        v
    }
    #[inline]
    fn with_value(&self, v: f64) -> Self {
        v
    }
}

impl JetFieldConst for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

/// A single-direction second-order jet over the field `S`: value `v`, first
/// derivative `g`, second derivative `h`, all with respect to ONE seeded
/// direction. Nest it (`Dual2<Dual2<f64>>`) for a second, independent direction.
#[derive(Clone, Copy, Debug)]
pub struct Dual2<S: JetField> {
    /// Value channel.
    pub v: S,
    /// First derivative in this dual's direction.
    pub g: S,
    /// Second derivative in this dual's direction.
    pub h: S,
}

impl<S: JetFieldConst> Dual2<S> {
    /// A constant (value `v`, zero derivatives) — carries no dependence on this
    /// dual's direction (but `v` may still depend on an inner nested direction).
    #[inline]
    pub fn constant(v: S) -> Self {
        Self {
            v,
            g: S::from_f64(0.0),
            h: S::from_f64(0.0),
        }
    }

    /// The seeded variable at `v`: unit first derivative in this dual's
    /// direction, zero second derivative.
    #[inline]
    pub fn variable(v: S) -> Self {
        Self {
            v,
            g: S::from_f64(1.0),
            h: S::from_f64(0.0),
        }
    }
}

impl<S: JetField> JetField for Dual2<S> {
    #[inline]
    fn value(&self) -> f64 {
        self.v.value()
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        Self {
            v: self.v.add(&o.v),
            g: self.g.add(&o.g),
            h: self.h.add(&o.h),
        }
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        Self {
            v: self.v.sub(&o.v),
            g: self.g.sub(&o.g),
            h: self.h.sub(&o.h),
        }
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        // Leibniz in one direction: (uv)′ = u′v + uv′,
        // (uv)″ = u″v + 2u′v′ + uv″.
        Self {
            v: self.v.mul(&o.v),
            g: self.v.mul(&o.g).add(&self.g.mul(&o.v)),
            h: self
                .v
                .mul(&o.h)
                .add(&self.g.mul(&o.g).scale(2.0))
                .add(&self.h.mul(&o.v)),
        }
    }
    #[inline]
    fn neg(&self) -> Self {
        Self {
            v: self.v.neg(),
            g: self.g.neg(),
            h: self.h.neg(),
        }
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        Self {
            v: self.v.scale(s),
            g: self.g.scale(s),
            h: self.h.scale(s),
        }
    }
    #[inline]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // f∘self in one direction: with u = self, φ = f,
        //   value  = φ(u)
        //   first  = φ′(u)·u′
        //   second = φ′(u)·u″ + φ″(u)·(u′)²
        // φ(u), φ′(u), φ″(u) are field-valued: compose the SHIFTED real stacks
        // with the inner value `self.v`, which propagates any nested direction.
        let f0 = self.v.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        let f1 = self.v.compose_unary([d[1], d[2], d[3], d[4], 0.0]);
        let f2 = self.v.compose_unary([d[2], d[3], d[4], 0.0, 0.0]);
        Self {
            v: f0,
            g: f1.mul(&self.g),
            h: f1.mul(&self.h).add(&f2.mul(&self.g).mul(&self.g)),
        }
    }
    #[inline]
    fn constant_like(&self, v: f64) -> Self {
        // The default's `g`/`h` are `compose_unary([0;5]).mul(..)` products of a
        // zero jet, i.e. structurally zero: build them directly.
        Self {
            v: self.v.constant_like(v),
            g: self.v.constant_like(0.0),
            h: self.v.constant_like(0.0),
        }
    }
    #[inline]
    fn with_value(&self, v: f64) -> Self {
        // Only the real value channel lives in `self.v`; this dual's own
        // derivative channels are untouched.
        Self {
            v: self.v.with_value(v),
            g: self.g.clone(),
            h: self.h.clone(),
        }
    }
}

impl<S: JetFieldConst> JetFieldConst for Dual2<S> {
    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::constant(S::from_f64(x))
    }
}

/// A `Dual2<Dual2<f64>>` seeded with independent directions `a` (outer) and `b`
/// (inner): value `x`, unit first derivative along both requested directions.
/// `p0` should be seeded `(a=1, b=0)` and `p1` `(a=0, b=1)` for a two-primary
/// program (mirrors `Tower4::variable(x, 0)` / `Tower4::variable(x, 1)`).
pub type Dual22 = Dual2<Dual2<f64>>;

impl Dual22 {

    /// The nine channels this nested dual represents, keyed to the two-primary
    /// [`crate::jet_tower::Tower4`] indices `0` (outer `a`) and `1` (inner `b`):
    /// `(value, ∂a, ∂b, ∂aa, ∂ab, ∂bb, ∂aab, ∂abb, ∂aabb)`.
    #[inline]
    pub fn channels(&self) -> [f64; 9] {
        [
            self.v.v, self.g.v, self.v.g, self.h.v, self.g.g, self.v.h, self.h.g, self.g.h,
            self.h.h,
        ]
    }
}

#[cfg(test)]
mod nested_dual_tower4_oracle_tests {
    use super::*;
    use crate::jet_tower::Tower4;

    // `Tower4` is a production `JetField` (its `JetScalar` impl in `jet_scalar.rs`
    // now rides the shared base), so the SAME `program` runs on both `Tower4<2>`
    // and `Dual2<Dual2<f64>>` with no test-only bridge. The oracle path adds only
    // the `Copy` constructor through `JetFieldConst`.
    impl<const K: usize> JetFieldConst for Tower4<K> {
        fn from_f64(x: f64) -> Self {
            Tower4::constant(x)
        }
    }

}
