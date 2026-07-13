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
pub trait JetField {
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
    /// Seed a primary that varies only along the OUTER direction `a`
    /// (`∂/∂a = 1`, `∂/∂b = 0`) — the `Tower4::variable(x, 0)` analogue.
    #[inline]
    pub fn seed_outer(x: f64) -> Self {
        Dual2::variable(Dual2::<f64>::constant(x))
    }
    /// Seed a primary that varies only along the INNER direction `b`
    /// (`∂/∂a = 0`, `∂/∂b = 1`) — the `Tower4::variable(x, 1)` analogue.
    #[inline]
    pub fn seed_inner(x: f64) -> Self {
        Dual2::constant(Dual2::<f64>::variable(x))
    }

    /// Seed a primary at value `base` that moves as `base + s·d1 + t·d2` under the
    /// two independent scalar directions `s` (outer `a`) and `t` (inner `b`):
    /// `∂/∂a = d1`, `∂/∂b = d2`, all second-and-higher self-derivatives zero
    /// (the primary is affine in `s, t`). This is what a directional
    /// bidirectional contraction along arbitrary weight vectors `d1, d2` needs —
    /// seed every primary `i` with `seed_directional(base_i, d1_i, d2_i)`, run
    /// the program, and read `channels()[8]` (`∂²_a ∂²_b`) for
    /// `Σ_{a,b,c,d} ℓ_{abcd}·d1_a d1_b d2_c d2_d`.
    #[inline]
    pub fn seed_directional(base: f64, d1: f64, d2: f64) -> Self {
        Dual2 {
            // value carries the inner (`t`) direction on its `g` channel.
            v: Dual2::<f64> {
                v: base,
                g: d2,
                h: 0.0,
            },
            // outer (`s`) first derivative is `d1`, itself constant in `t`.
            g: Dual2::<f64>::constant(d1),
            h: Dual2::<f64>::constant(0.0),
        }
    }

    /// Build a nested dual directly from its nine `(s-order, t-order)` channels,
    /// ordered as [`Self::channels`]: `[v, ∂a, ∂b, ∂aa, ∂ab, ∂bb, ∂aab, ∂abb,
    /// ∂aabb]`. The inverse of [`Self::channels`]. Used to assemble the result of
    /// a channel-space operation (e.g. a moment-recurrence residual term) back
    /// into a `Dual22`.
    #[inline]
    pub fn from_channels(c: [f64; 9]) -> Self {
        Dual2 {
            v: Dual2::<f64> {
                v: c[0],
                g: c[2],
                h: c[5],
            },
            g: Dual2::<f64> {
                v: c[1],
                g: c[4],
                h: c[7],
            },
            h: Dual2::<f64> {
                v: c[3],
                g: c[6],
                h: c[8],
            },
        }
    }

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

    /// exp stack `[e,e,e,e,e]` at `u`.
    fn exp_stack(u: f64) -> [f64; 5] {
        let e = u.exp();
        [e, e, e, e, e]
    }
    /// ln stack `[ln u, 1/u, -1/u², 2/u³, -6/u⁴]` at `u > 0`.
    fn ln_stack(u: f64) -> [f64; 5] {
        let r = 1.0 / u;
        [u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r]
    }

    /// A smooth two-primary program with genuinely nonzero mixed fourth
    /// derivatives, written once over `JetField` so it evaluates identically on
    /// the engine `Tower4<2>` and on the nested `Dual2<Dual2<f64>>`.
    ///
    ///   f(p0, p1) = exp(p0·p1 + 0.3·p0)
    ///             + ln(1 + p0² + 0.5·p1² + 0.2·p0·p1)
    ///             − 0.7·(p0 − p1)²
    fn program<J: JetFieldConst>(p0: &J, p1: &J) -> J {
        let one = J::from_f64(1.0);
        // exp(p0·p1 + 0.3·p0)
        let arg_e = p0.mul(p1).add(&p0.scale(0.3));
        let term_exp = arg_e.compose_unary(exp_stack(arg_e.value()));
        // ln(1 + p0² + 0.5·p1² + 0.2·p0·p1)
        let arg_l = one
            .add(&p0.mul(p0))
            .add(&p1.mul(p1).scale(0.5))
            .add(&p0.mul(p1).scale(0.2));
        let term_ln = arg_l.compose_unary(ln_stack(arg_l.value()));
        // −0.7·(p0 − p1)²
        let diff = p0.sub(p1);
        let term_quad = diff.mul(&diff).scale(-0.7);
        term_exp.add(&term_ln).add(&term_quad)
    }

    /// The nested `Dual2<Dual2<f64>>` reproduces every channel it represents —
    /// value, both gradients, the full Hessian, the two order-3 mixed channels,
    /// and the order-4 bidirectional `∂²_a ∂²_b` — of the engine `Tower4<2>`, to
    /// machine precision, over several smooth base points. This is the
    /// truncation-free, hand-oracle-free proof the nested dual is a correct
    /// fourth-order path before it is used to gate the flex Jet4 tower (#932).
    #[test]
    fn nested_dual2_reproduces_tower4_channels_932() {
        let points = [
            (0.31_f64, -0.42_f64),
            (-0.85, 0.17),
            (0.05, 0.93),
            (1.2, -0.6),
        ];
        let mut max_rel = 0.0_f64;
        for &(x0, x1) in &points {
            // Engine tower.
            let t0 = Tower4::<2>::variable(x0, 0);
            let t1 = Tower4::<2>::variable(x1, 1);
            let tower = program(&t0, &t1);

            // Nested dual (outer = axis 0, inner = axis 1).
            let d0 = Dual22::seed_outer(x0);
            let d1 = Dual22::seed_inner(x1);
            let nested = program(&d0, &d1);
            let ch = nested.channels();

            // (label, nested channel, tower channel).
            let cmp = [
                ("value", ch[0], tower.v),
                ("d_a", ch[1], tower.g[0]),
                ("d_b", ch[2], tower.g[1]),
                ("d_aa", ch[3], tower.h[0][0]),
                ("d_ab", ch[4], tower.h[0][1]),
                ("d_bb", ch[5], tower.h[1][1]),
                ("d_aab", ch[6], tower.t3[0][0][1]),
                ("d_abb", ch[7], tower.t3[0][1][1]),
                ("d_aabb", ch[8], tower.t4[0][0][1][1]),
            ];
            for (label, got, want) in cmp {
                let rel = (got - want).abs() / want.abs().max(1.0);
                max_rel = max_rel.max(rel);
                assert!(
                    rel <= 1e-12,
                    "point ({x0},{x1}) channel {label}: nested {got:.16e} != tower {want:.16e} (rel {rel:.3e})"
                );
            }
        }
        eprintln!(
            "[nested-dual #932] Dual2<Dual2> vs Tower4<2> max_rel over 4 points = {max_rel:.3e}"
        );
    }

    /// The directional seeding (arbitrary weight vectors `d1`, `d2`) reproduces
    /// the engine tower's fully-contracted derivatives:
    ///   `∂_s f      = Σ_a g_a d1_a`,
    ///   `∂_t f      = Σ_a g_a d2_a`,
    ///   `∂_s∂_t f   = Σ_{ab} h_ab d1_a d2_b`,
    ///   `∂²_s∂²_t f = Σ_{abcd} ℓ_abcd d1_a d1_b d2_c d2_d`,
    /// the last being exactly the bidirectional 4th-order contraction a flex
    /// Jet4 gate needs (`ℓ_{dir1,dir1,dir2,dir2}`), FD-free. This is the seeding
    /// the flex-geometry consumer will use.
    #[test]
    fn nested_dual2_directional_matches_tower4_contraction_932() {
        let d1 = [0.7_f64, -0.3_f64];
        let d2 = [0.4_f64, 0.9_f64];
        let points = [(0.31_f64, -0.42_f64), (-0.85, 0.17), (1.2, -0.6)];
        let mut max_rel = 0.0_f64;
        for &(x0, x1) in &points {
            let t0 = Tower4::<2>::variable(x0, 0);
            let t1 = Tower4::<2>::variable(x1, 1);
            let tower = program(&t0, &t1);

            // Full engine contractions.
            let mut c_s = 0.0;
            let mut c_t = 0.0;
            let mut c_st = 0.0;
            let mut c_sstt = 0.0;
            for a in 0..2 {
                c_s += tower.g[a] * d1[a];
                c_t += tower.g[a] * d2[a];
                for b in 0..2 {
                    c_st += tower.h[a][b] * d1[a] * d2[b];
                    for cc in 0..2 {
                        for dd in 0..2 {
                            c_sstt += tower.t4[a][b][cc][dd] * d1[a] * d1[b] * d2[cc] * d2[dd];
                        }
                    }
                }
            }

            let p0 = Dual22::seed_directional(x0, d1[0], d2[0]);
            let p1 = Dual22::seed_directional(x1, d1[1], d2[1]);
            let ch = program(&p0, &p1).channels();

            for (label, got, want) in [
                ("d_s", ch[1], c_s),
                ("d_t", ch[2], c_t),
                ("d_st", ch[4], c_st),
                ("d_sstt", ch[8], c_sstt),
            ] {
                let rel = (got - want).abs() / want.abs().max(1.0);
                max_rel = max_rel.max(rel);
                assert!(
                    rel <= 1e-12,
                    "point ({x0},{x1}) {label}: nested {got:.16e} != tower-contraction {want:.16e} (rel {rel:.3e})"
                );
            }
        }
        eprintln!(
            "[nested-dual #932] directional Dual2<Dual2> vs Tower4<2> contraction max_rel = {max_rel:.3e}"
        );
    }

    /// `from_channels` is the exact inverse of `channels` (round-trip identity),
    /// so a channel-space operation can be assembled back into a `Dual22`
    /// without silently transposing a slot.
    #[test]
    fn nested_dual2_channels_from_channels_roundtrip_932() {
        let c = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let round = Dual22::from_channels(c).channels();
        assert_eq!(round, c, "channels∘from_channels must be identity");
    }

    /// Symmetry guard: the mixed channels are order-independent — `∂a∂b == ∂b∂a`
    /// and `∂²a∂²b` computed with the seeds swapped agrees — so the nested read
    /// is not silently reading a transposed channel.
    #[test]
    fn nested_dual2_seed_swap_symmetry_932() {
        let (x0, x1) = (0.4_f64, -0.55_f64);
        let d0 = Dual22::seed_outer(x0);
        let d1 = Dual22::seed_inner(x1);
        let ab = program(&d0, &d1).channels();

        // Swap which variable carries the outer vs inner seed.
        let e0 = Dual22::seed_inner(x0);
        let e1 = Dual22::seed_outer(x1);
        let ba = program(&e0, &e1).channels();

        // value, and the fully-symmetric 4th channel, must be seed-order
        // invariant; the gradients swap (a<->b), as do the order-3 channels.
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
        assert!(close(ab[0], ba[0]), "value not seed-order invariant");
        assert!(close(ab[8], ba[8]), "d_aabb not seed-order invariant");
        assert!(close(ab[1], ba[2]), "d_a (ab) != d_b (ba)");
        assert!(close(ab[6], ba[7]), "d_aab (ab) != d_abb (ba)");
    }
}
