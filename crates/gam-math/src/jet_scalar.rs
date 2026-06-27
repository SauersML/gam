//! Order-specific Taylor-jet SCALAR algebras (#932 cutover, doc §A).
//!
//! [`crate::jet_tower::Tower4`] carries the full value/gradient/Hessian/`t3`/`t4`
//! tensor stack: it answers EVERY channel a [`super::row_kernel::RowKernel`]
//! consumer can ask for, but at `K = 9` that is a ~50 KiB per-row object whose
//! by-value copies overflowed the stack and timed out the location-scale fit —
//! which is exactly why `row_kernel_directional_supported()` /
//! `row_kernel_joint_hessian_supported()` still `return false`. The cutover does
//! NOT need the dense `Tower4<9>` per row; it needs, per consumer, only the one
//! channel that consumer serves:
//!
//! | consumer | channel | scalar here | K=9 size |
//! |---|---|---|---|
//! | inner Newton / `row_kernel` | `(v, g, H)` | [`Order2`] | 728 B |
//! | `row_third_contracted(dir)` | `Σ_c ℓ_{abc} dir_c` | [`OneSeed`] | 1.46 KiB |
//! | `row_fourth_contracted(u, v)` | `Σ_{cd} ℓ_{abcd} u_c v_d` | [`TwoSeed`] | 2.8 KiB |
//!
//! Each is built on [`Order2`] (value/grad/Hessian), which is the production
//! [`crate::jet_tower::Tower2`] re-expressed behind a generic interface: a row
//! loss written ONCE against [`JetScalar`] re-instantiates at whatever order /
//! representation a consumer needs, with the contraction folded INTO the
//! differentiation (the nilpotent ε / δ directions), so `t3` / `t4` are never
//! materialised. The single source of truth is the same one expression — the
//! genus of #736 cross-block drift cannot reappear because there is no separate
//! channel to forget.
//!
//! # Why each scalar is exact (doc §A.1–A.3)
//!
//! * [`Order2`] is the order-≤2 truncation of the Leibniz / Faà di Bruno rules.
//!   Those order-2 terms read ONLY the order-≤2 channels of their inputs (see
//!   [`crate::jet_tower::Tower4::mul`]: `out.h[i][j]` never touches `t3`/`t4`),
//!   so its `(v, g, H)` is BIT-IDENTICAL to a full `Tower4<K>` — and identical
//!   to [`crate::jet_tower::Tower2`], over which it is a thin newtype.
//! * [`OneSeed`] carries an [`Order2`] base plus one nilpotent ε (`ε² = 0`)
//!   holding another [`Order2`]. Seeding ε with the fixed direction `u` makes the
//!   ε-component of the Hessian channel the contracted third `Σ_c ℓ_{abc} u_c`
//!   (the nilpotent implements `d/dτ|₀` of `ℓ_{ab}(p + τu)` exactly).
//! * [`TwoSeed`] carries an [`Order2`] base plus ε, δ (`ε² = δ² = 0`, `εδ`
//!   retained) — four [`Order2`] parts. Seeding ε, δ with `u, v` makes the
//!   εδ-component of the Hessian channel the contracted fourth
//!   `Σ_{cd} ℓ_{abcd} u_c v_d` (the single mixed `∂_σ∂_ρ|₀` term, no `σ²`/`ρ²`
//!   contamination).
//!
//! # Stability discipline
//!
//! As in [`crate::jet_tower`], humans own primitive stability and the algebra
//! owns combinatorics: tail-critical special functions enter ONLY as
//! hand-certified `[f64; 5]` derivative stacks through [`JetScalar::compose_unary`]
//! (each scalar consumes the leading entries its order needs), never by
//! differentiating an unstable primal.
//!
//! # Production scalars and the test-only all-channels oracle
//!
//! The `JetScalar` trait below is production: it is the bound on
//! [`crate::jet_tower::RowNllProgramGeneric::row_nll_generic`], the seam a family
//! row loss is written against. The order-specific scalars that *consume* it —
//! [`Order2`] (value/grad/Hessian), [`OneSeed`] (contracted third) and
//! [`TwoSeed`] (contracted fourth) — are production: the survival location-scale
//! `RowKernel<9>` builds its joint Hessian / directional derivatives through them
//! (`survival::location_scale::row_kernel`), paying only the small packed scalar
//! per row instead of the ~50 KiB dense [`crate::jet_tower::Tower4`].
//!
//! The [`crate::jet_tower::Tower4`] all-channels `JetScalar` impl is test-only: it
//! is the oracle that pins the contracted scalars against the dense
//! value/grad/Hessian/`t3`/`t4` truth, so it lives in the `#[cfg(test)]` module.

/// A truncated-Taylor scalar carrying derivatives in `K` primaries.
///
/// All concrete scalars here ([`Order2`], [`OneSeed`], [`TwoSeed`]) and the full
/// [`crate::jet_tower::Tower4`] implement the SAME algebra; only the carried
/// channel set differs. A row loss written once against this interface yields a
/// different channel set per instantiation, all exact for the channel they serve
/// (doc §A.0).
pub trait JetScalar<const K: usize>: Copy {
    /// A constant: value `c`, every derivative channel zero.
    fn constant(c: f64) -> Self;

    /// The seeded variable `p_axis` at value `x`: unit first derivative in slot
    /// `axis`, all higher channels zero. (The nilpotent / cross channels of the
    /// directional scalars are seeded zero — callers set ε/δ directions through
    /// the scalar-specific [`OneSeed::seed_direction`] / [`TwoSeed::seed`].)
    fn variable(x: f64, axis: usize) -> Self;

    /// The value channel `ℓ(p)`.
    fn value(&self) -> f64;

    /// Exact truncated Leibniz sum `self + o`.
    fn add(&self, o: &Self) -> Self;
    /// Exact truncated Leibniz difference `self − o`.
    fn sub(&self, o: &Self) -> Self;
    /// Exact truncated Leibniz product `self · o`.
    fn mul(&self, o: &Self) -> Self;
    /// Negate every channel.
    fn neg(&self) -> Self;
    /// Multiply every channel by a plain scalar `s`.
    fn scale(&self, s: f64) -> Self;

    /// Exact multivariate Faà di Bruno composition `f ∘ self`, given the outer
    /// derivative stack `d = [f(u), f′(u), f″(u), f‴(u), f⁗(u)]` at
    /// `u = self.value()`.
    ///
    /// This is the SAME `[f64; 5]` stack shape [`crate::jet_tower::Tower4`] and
    /// the families' `unary_derivatives_*` helpers (built on erfcx / log_ndtr)
    /// already produce, so those stacks plug in directly. Each scalar consumes
    /// only the leading entries its order needs (order-2 reads `d[0..=2]`; the
    /// directional scalars read one / two beyond their base) — the fixed-length
    /// array makes that windowing total, no length guard required.
    fn compose_unary(&self, d: [f64; 5]) -> Self;

    /// `e^self`. Convenience for tame arguments (see module stability note).
    fn exp(&self) -> Self {
        let e = self.value().exp();
        self.compose_unary([e, e, e, e, e])
    }

    /// `√self`. Caller guarantees positivity.
    fn sqrt(&self) -> Self {
        let u = self.value();
        let s = u.sqrt();
        self.compose_unary([
            s,
            0.5 / s,
            -0.25 / (u * s),
            0.375 / (u * u * s),
            -0.9375 / (u * u * u * s),
        ])
    }

    /// `ln(self)`. Caller guarantees positivity. Same derivative stack
    /// [`crate::jet_tower::Tower4::ln`] uses, so any program written over both
    /// matches term-for-term.
    fn ln(&self) -> Self {
        let u = self.value();
        let r = 1.0 / u;
        self.compose_unary([u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r])
    }

    /// `1/self`.
    fn recip(&self) -> Self {
        let r = 1.0 / self.value();
        let r2 = r * r;
        self.compose_unary([r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r])
    }

    /// `self^a` for real exponent `a`. Caller guarantees a positive base.
    /// Mirrors [`crate::jet_tower::Tower4::powf`] (falling-factorial stack).
    fn powf(&self, a: f64) -> Self {
        let u = self.value();
        self.compose_unary([
            u.powf(a),
            a * u.powf(a - 1.0),
            a * (a - 1.0) * u.powf(a - 2.0),
            a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0),
            a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0),
        ])
    }

    /// `ln Γ(self)`. Caller guarantees a positive argument. Uses the SAME
    /// hand-certified derivative stack [`crate::jet_tower::Tower4::ln_gamma`]
    /// consumes ([`crate::jet_tower::ln_gamma_derivative_stack`]), so any
    /// program written over both matches term-for-term.
    fn ln_gamma(&self) -> Self {
        self.compose_unary(crate::jet_tower::ln_gamma_derivative_stack(self.value()))
    }

    /// `ψ(self) = d/dx ln Γ(x)` (digamma). Caller guarantees a positive
    /// argument. Same hand-certified stack
    /// [`crate::jet_tower::digamma_derivative_stack`].
    fn digamma(&self) -> Self {
        self.compose_unary(crate::jet_tower::digamma_derivative_stack(self.value()))
    }
}

// ── Order2<K> ergonomic operator overloads (doc §A.1) ───────────────────
//
// The dispersion-family row NLLs are written with `+`/`-`/`*` operators over
// the primaries (mirroring how they read as `Tower4` expressions). These
// delegate channel-for-channel to the inner `Tower2` arithmetic (which has
// `Add`/`Mul`; `Sub`/`Neg` are expressed as `+ (-1)·rhs` exactly as the
// `JetScalar::sub` / `JetScalar::neg` impls do), so an `Order2` expression is
// bit-identical to the same `Tower4` expression's order-≤2 channels.

impl<const K: usize> std::ops::Add for Order2<K> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Order2(self.0 + o.0)
    }
}

impl<const K: usize> std::ops::Add<f64> for Order2<K> {
    type Output = Self;
    #[inline]
    fn add(self, c: f64) -> Self {
        Order2(self.0 + c)
    }
}

impl<const K: usize> std::ops::Sub for Order2<K> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Order2(self.0 + o.0.scale(-1.0))
    }
}

impl<const K: usize> std::ops::Sub<f64> for Order2<K> {
    type Output = Self;
    #[inline]
    fn sub(self, c: f64) -> Self {
        Order2(self.0 + (-c))
    }
}

impl<const K: usize> std::ops::Mul for Order2<K> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Order2(crate::jet_tower::Tower2::mul(&self.0, &o.0))
    }
}

impl<const K: usize> std::ops::Mul<f64> for Order2<K> {
    type Output = Self;
    #[inline]
    fn mul(self, c: f64) -> Self {
        Order2(self.0.scale(c))
    }
}

impl<const K: usize> std::ops::Neg for Order2<K> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Order2(self.0.scale(-1.0))
    }
}

/// Filtered Hensel lift of a SCALAR implicit state `a(θ)` defined by the
/// constraint `F(a, θ) = 0`, evaluated in ANY [`JetScalar`] algebra `S` (doc
/// §11, "A generic implicit-lift operator for every production scalar").
///
/// This is the perf-respecting alternative to lifting through a dense
/// `Tower4<K+1>` (which carries the implicit variable as an extra dense axis):
/// the state `a` lives directly in the consumer's own `K`-primary algebra
/// `S` — `Order2<K>` for value/gradient/Hessian, `Tower4<K>` for the full
/// `t3`/`t4` — never paying for an extra variable.
///
/// **Method.** Fixed-Jacobian Newton in the nilpotent algebra. By the
/// filtered-lift theorem (doc §11.1), if `F_a := ∂F/∂a(a₀, θ₀)` is the primal
/// Jacobian at the base point and `inv_fa = 1/F_a`, then the iteration
/// `A ← A − inv_fa · F(A, θ)` raises the filtration degree of the residual by
/// at least one per step: each step kills exactly one graded layer. Starting
/// from `A = const(a₀)` (whose residual lies in `F¹` because `θ − θ₀ ∈ 𝔫`),
/// `iters` equal to the algebra's nilpotency order returns the *exact* lifted
/// jet (`Order2`: 2, `OneSeed`: 3, `Tower4`/`TwoSeed`: 4). The value channel of
/// `A` never moves — `F(A, θ).value() = F(a₀, θ₀) = 0` at the certified root —
/// so a caller may precompute every primitive's derivative stack at the fixed
/// base index once and let the cheap polynomial composition repeat per step.
///
/// `f` evaluates the constraint `F(a, θ)` in `S` (capturing the seeded
/// parameter jets `θ`); `a0` is the certified scalar root `F(a₀, θ₀) ≈ 0`.
pub fn filtered_implicit_solve_scalar<const K: usize, S: JetScalar<K>>(
    a0: f64,
    inv_fa: f64,
    iters: usize,
    f: impl Fn(&S) -> S,
) -> S {
    let mut a = S::constant(a0);
    for _ in 0..iters {
        let residual = f(&a);
        a = a.sub(&residual.scale(inv_fa));
    }
    a
}

// ── Order2<K>: value / gradient / Hessian (doc §A.1) ────────────────────

/// Truncated SECOND-order scalar: value `v`, gradient `g_a`, Hessian `H_{ab}`.
///
/// This is a thin newtype over the production [`crate::jet_tower::Tower2`], so
/// its `(v, g, H)` channels are obtained by the SAME formulas — and are
/// therefore bit-identical to both [`crate::jet_tower::Tower2`] and the order-≤2
/// channels of a full [`crate::jet_tower::Tower4`] (doc §A.1, "Bit-identity with
/// the full tower"). The wrapper exists only to satisfy the generic
/// [`JetScalar`] interface (the `compose_unary` / `add` / `sub` / `neg` /
/// `recip` the trait demands, which `Tower2` does not expose by that shape) —
/// every channel is delegated to `Tower2` arithmetic unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Order2<const K: usize>(pub crate::jet_tower::Tower2<K>);

impl<const K: usize> Order2<K> {
    /// Read the gradient channel `g_a = ∂ℓ/∂p_a`.
    #[inline]
    pub fn g(&self) -> [f64; K] {
        self.0.g
    }

    /// Read the Hessian channel.
    #[inline]
    pub fn h(&self) -> [[f64; K]; K] {
        self.0.h
    }
}

impl<const K: usize> JetScalar<K> for Order2<K> {
    fn constant(c: f64) -> Self {
        Order2(crate::jet_tower::Tower2::constant(c))
    }
    fn variable(x: f64, axis: usize) -> Self {
        Order2(crate::jet_tower::Tower2::variable(x, axis))
    }
    fn value(&self) -> f64 {
        self.0.v
    }
    fn add(&self, o: &Self) -> Self {
        Order2(self.0 + o.0)
    }
    fn sub(&self, o: &Self) -> Self {
        // Tower2 has no Sub op; subtract by adding the negation, matching
        // Tower4::sub (self + o.scale(-1.0)).
        Order2(self.0 + o.0.scale(-1.0))
    }
    fn mul(&self, o: &Self) -> Self {
        Order2(crate::jet_tower::Tower2::mul(&self.0, &o.0))
    }
    fn neg(&self) -> Self {
        Order2(self.0.scale(-1.0))
    }
    fn scale(&self, s: f64) -> Self {
        Order2(self.0.scale(s))
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Order-≤2 reads only [f, f', f''] of the stack.
        Order2(self.0.compose_unary([d[0], d[1], d[2]]))
    }
}

// ── Lane-batched Order-2 scalar: 4 rows per pass in SIMD lanes (perf) ────
//
// The hot per-row jet kernels evaluate ONE row's `(v, g, H)` tower at a time in
// scalar `f64`. A hand-written scalar derivative does the same. The throughput
// lever a jet has that scalar hand-code cannot is **row batching in SIMD
// lanes**: the order-≤2 Leibniz product `Order2::mul` is `O(K²)` independent
// per-channel float ops, and EVERY row runs the identical op graph on different
// data — the textbook SPMD shape. Packing `LANES = 4` rows into a `wide::f64x4`
// and running the algebra once per 4 rows replaces 4 scalar passes with one
// vector pass: the `K²` Hessian channel updates become `K²` NEON `.2d` / SSE2
// `pd` instructions covering 4 rows each, ~4× fewer FP instructions per row.
//
// The carried scalar field is abstracted by [`Lane`] so the SAME algebra body
// instantiates at `f64` (1 row, used as the bit-identity oracle) or
// [`wide::f64x4`] (4 rows). Bit-identity is structural, not approximate:
//
//   * Every arithmetic op is a plain lane-wise `+` / `-` / `*` (NEVER a fused
//     `mul_add`), and IEEE-754 double `+`/`-`/`*`/`/` are correctly rounded and
//     deterministic, so lane `i` of an `f64x4` op equals the scalar `f64` op on
//     that lane's inputs bit-for-bit.
//   * The transcendental derivative STACKS (`exp`/`ln`/`sqrt`/…) are produced
//     **per lane by the identical scalar code** ([`Lane::unary3`] unpacks, runs
//     the same `[f64; 3]` stack closure the scalar path runs, repacks), so the
//     only thing vectorised is the cheap rational tensor composition — the
//     library transcendental itself is the exact same `f64::exp` call per lane.
//   * The op order mirrors [`crate::jet_tower::Tower2`] term-for-term, so
//     [`Order2Lane<f64, K>`] is `to_bits`-identical to the production
//     [`Order2<K>`] (= `Tower2<K>`), and [`Order2Lane<f64x4, K>`] lane `i` is
//     `to_bits`-identical to that — proven by the `batch_tests` oracle below
//     (≥2000 random 4-row batches across `K ∈ {2,3,4,9}`).

/// The scalar field a [`Order2Lane`] carries: either a single `f64` (one row,
/// the oracle) or a [`wide::f64x4`] (four rows evaluated in SIMD lanes). All ops
/// are plain lane-wise IEEE arithmetic, so a vector op equals the scalar op on
/// each lane bit-for-bit.
pub trait Lane: Copy {
    /// Broadcast a scalar to every lane.
    fn splat(x: f64) -> Self;
    /// Lane-wise `self + o`.
    fn add(self, o: Self) -> Self;
    /// Lane-wise `self - o`.
    fn sub(self, o: Self) -> Self;
    /// Lane-wise `self * o`.
    fn mul(self, o: Self) -> Self;
    /// The `f64` in lane `i` (`i < LANES`; `f64` ignores `i`).
    fn lane(self, i: usize) -> f64;
    /// Build the order-≤2 derivative stack `[f(u), f′(u), f″(u)]` **per lane**
    /// from the lane value `u`, via the SAME scalar `stack` closure the
    /// per-row path runs (so the transcendental/rational stack is bit-identical
    /// to the scalar evaluation — only the subsequent tensor composition is
    /// vectorised).
    fn unary3(self, stack: impl Fn(f64) -> [f64; 3]) -> [Self; 3];
}

impl Lane for f64 {
    #[inline]
    fn splat(x: f64) -> Self {
        x
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn lane(self, _i: usize) -> f64 {
        self
    }
    #[inline]
    fn unary3(self, stack: impl Fn(f64) -> [f64; 3]) -> [Self; 3] {
        stack(self)
    }
}

impl Lane for wide::f64x4 {
    #[inline]
    fn splat(x: f64) -> Self {
        wide::f64x4::splat(x)
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn lane(self, i: usize) -> f64 {
        self.to_array()[i]
    }
    #[inline]
    fn unary3(self, stack: impl Fn(f64) -> [f64; 3]) -> [Self; 3] {
        let a = self.to_array();
        let mut d0 = [0.0_f64; 4];
        let mut d1 = [0.0_f64; 4];
        let mut d2 = [0.0_f64; 4];
        for i in 0..4 {
            let s = stack(a[i]);
            d0[i] = s[0];
            d1[i] = s[1];
            d2[i] = s[2];
        }
        [
            wide::f64x4::new(d0),
            wide::f64x4::new(d1),
            wide::f64x4::new(d2),
        ]
    }
}

/// A lane-batched order-≤2 Taylor scalar: value / gradient / Hessian carried in
/// a SIMD field [`L: Lane`](Lane). With `L = f64x4` one instance carries FOUR
/// rows at once, so the row loop processes 4 rows per vector pass instead of one
/// per scalar pass.
///
/// The channel layout and every float op mirror [`crate::jet_tower::Tower2`]
/// term-for-term, so `Order2Lane<f64, K>` is `to_bits`-identical to the
/// production [`Order2<K>`] and `Order2Lane<f64x4, K>` lane `i` is
/// `to_bits`-identical to that (see the module note and `batch_tests`).
#[derive(Clone, Copy, Debug)]
pub struct Order2Lane<L: Lane, const K: usize> {
    /// Value channel `ℓ` (one entry per lane/row).
    pub v: L,
    /// Gradient channel `∂ℓ/∂p_a`.
    pub g: [L; K],
    /// Hessian channel `∂²ℓ/∂p_a∂p_b` (symmetric).
    pub h: [[L; K]; K],
}

/// The 4-rows-per-pass batched order-≤2 scalar (`wide::f64x4` lanes).
pub type Order2Batch<const K: usize> = Order2Lane<wide::f64x4, K>;

impl<L: Lane, const K: usize> Order2Lane<L, K> {
    /// A constant: value `c` in every channel-zero slot.
    #[inline]
    pub fn constant(c: L) -> Self {
        Order2Lane {
            v: c,
            g: [L::splat(0.0); K],
            h: [[L::splat(0.0); K]; K],
        }
    }

    /// The seeded variable `p_axis` at (per-lane) value `value`: unit first
    /// derivative in slot `axis`. With `L = f64x4`, `value` packs the four
    /// rows' values of primary `axis`.
    #[inline]
    pub fn variable(value: L, axis: usize) -> Self {
        let mut out = Self::constant(value);
        out.g[axis] = L::splat(1.0);
        out
    }

    /// Lane-wise `self + o` (mirrors `Tower2` Add: per-channel add).
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = self.v.add(o.v);
        for i in 0..K {
            out.g[i] = self.g[i].add(o.g[i]);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].add(o.h[i][j]);
            }
        }
        out
    }

    /// Multiply every channel by the plain scalar `s` (mirrors `Tower2::scale`).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        let sl = L::splat(s);
        let mut out = *self;
        out.v = self.v.mul(sl);
        for i in 0..K {
            out.g[i] = self.g[i].mul(sl);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].mul(sl);
            }
        }
        out
    }

    /// Lane-wise `self - o`, expressed as `self + o·(-1)` exactly as
    /// [`Order2::sub`] / `Tower4::sub` do, so signed-zero handling matches.
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        self.add(&o.scale(-1.0))
    }

    /// Negate every channel (= `scale(-1.0)`, matching [`Order2::neg`]).
    #[inline]
    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// Exact order-≤2 Leibniz product, term-for-term identical to
    /// [`crate::jet_tower::Tower2::mul`] (same factor order, no `mul_add`).
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::constant(a.v.mul(b.v));
        for i in 0..K {
            // a.v*b.g[i] + a.g[i]*b.v
            out.g[i] = a.v.mul(b.g[i]).add(a.g[i].mul(b.v));
        }
        for i in 0..K {
            for j in 0..K {
                // a.v*b.h + a.g[i]*b.g[j] + a.g[j]*b.g[i] + a.h*b.v
                out.h[i][j] = a
                    .v
                    .mul(b.h[i][j])
                    .add(a.g[i].mul(b.g[j]))
                    .add(a.g[j].mul(b.g[i]))
                    .add(a.h[i][j].mul(b.v));
            }
        }
        out
    }

    /// Exact order-≤2 Faà di Bruno composition `f ∘ self`, given the per-lane
    /// derivative stack `d = [f(u), f′(u), f″(u)]`. Mirrors
    /// [`crate::jet_tower::Tower2::compose_unary`] term-for-term (`acc` starts at
    /// `0` then accumulates, so signed-zero collapses identically).
    #[inline]
    pub fn compose_unary(&self, d: [L; 3]) -> Self {
        let mut out = Self::constant(d[0]);
        for i in 0..K {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(self.g[i]));
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = L::splat(0.0);
                acc = acc.add(d[1].mul(self.h[i][j]));
                acc = acc.add(d[2].mul(self.g[i]).mul(self.g[j]));
                out.h[i][j] = acc;
            }
        }
        out
    }

    /// `e^self`, per-lane stack `[e, e, e]` (matches the [`JetScalar::exp`]
    /// default forwarded through `Order2`).
    #[inline]
    pub fn exp(&self) -> Self {
        let d = self.v.unary3(|u| {
            let e = u.exp();
            [e, e, e]
        });
        self.compose_unary(d)
    }

    /// `ln(self)`; caller guarantees positivity. Per-lane stack
    /// `[ln u, 1/u, -1/u²]` (matches [`JetScalar::ln`] truncated to order 2).
    #[inline]
    pub fn ln(&self) -> Self {
        let d = self.v.unary3(|u| {
            let r = 1.0 / u;
            [u.ln(), r, -r * r]
        });
        self.compose_unary(d)
    }

    /// `√self`; caller guarantees positivity. Per-lane stack
    /// `[s, 0.5/s, -0.25/(u·s)]` (matches [`JetScalar::sqrt`]).
    #[inline]
    pub fn sqrt(&self) -> Self {
        let d = self.v.unary3(|u| {
            let s = u.sqrt();
            [s, 0.5 / s, -0.25 / (u * s)]
        });
        self.compose_unary(d)
    }

    /// `1/self`. Per-lane stack `[r, -r², 2r³]` (matches [`JetScalar::recip`]).
    #[inline]
    pub fn recip(&self) -> Self {
        let d = self.v.unary3(|u| {
            let r = 1.0 / u;
            let r2 = r * r;
            [r, -r2, 2.0 * r2 * r]
        });
        self.compose_unary(d)
    }

    /// `self^a` for real `a`; caller guarantees a positive base. Per-lane
    /// falling-factorial stack (matches [`JetScalar::powf`]).
    #[inline]
    pub fn powf(&self, a: f64) -> Self {
        let d = self.v.unary3(|u| {
            [
                u.powf(a),
                a * u.powf(a - 1.0),
                a * (a - 1.0) * u.powf(a - 2.0),
            ]
        });
        self.compose_unary(d)
    }
}

impl<const K: usize> Order2Batch<K> {
    /// Extract lane `i`'s `(v, g, H)` as a production [`Order2<K>`] scalar.
    /// Lane `i` is `to_bits`-identical to evaluating the same program at
    /// [`Order2<K>`] on row `i` (see `batch_tests`).
    #[inline]
    #[must_use]
    pub fn lane(&self, i: usize) -> Order2<K> {
        let mut t = crate::jet_tower::Tower2::<K>::constant(self.v.lane(i));
        for a in 0..K {
            t.g[a] = self.g[a].lane(i);
            for b in 0..K {
                t.h[a][b] = self.h[a][b].lane(i);
            }
        }
        Order2(t)
    }
}

// ── Order1<K>: value / gradient only (doc §A.1, first-order prune) ──────

/// Truncated FIRST-order scalar: value `v` and gradient `g_a` only — NO Hessian.
///
/// This is [`Order2`] with the K×K Hessian channel deleted. Its value and
/// gradient are computed by the SAME order-≤1 truncation of the Leibniz / Faà
/// di Bruno rules that [`Order2`] uses for those two channels, with the float
/// operations applied in the identical order — so its `(v, g)` is BIT-IDENTICAL
/// to both [`Order2`]'s and a full [`crate::jet_tower::Tower4`]'s order-≤1
/// channels. Use it at a consumer that reads ONLY value + gradient (the SAE
/// β-border channel: the reconstruction is linear in β, so the Hessian-in-β
/// vanishes and the dense K×K Hessian product `Tower2::mul` would build is pure
/// discarded work). Order-≤1 value/gradient never read any input's Hessian, so
/// dropping that channel changes neither result nor float-op order — it only
/// removes the `K²` arithmetic that produced an unread tensor.
#[derive(Clone, Copy, Debug)]
pub struct Order1<const K: usize> {
    /// Value ℓ.
    pub v: f64,
    /// Gradient ∂ℓ/∂p_a.
    pub g: [f64; K],
}

impl<const K: usize> Order1<K> {
    /// Read the gradient channel `g_a = ∂ℓ/∂p_a`.
    #[inline]
    pub fn g(&self) -> [f64; K] {
        self.g
    }
}

impl<const K: usize> JetScalar<K> for Order1<K> {
    fn constant(c: f64) -> Self {
        // Order2::constant -> Tower2::constant: value c, all derivatives zero.
        Order1 { v: c, g: [0.0; K] }
    }
    fn variable(x: f64, axis: usize) -> Self {
        // Order2::variable -> Tower2::variable: unit first derivative in `axis`.
        let mut g = [0.0; K];
        g[axis] = 1.0;
        Order1 { v: x, g }
    }
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        // Tower2 Add: out.v += o.v; out.g[i] += o.g[i] (same float order).
        let mut g = self.g;
        for i in 0..K {
            g[i] += o.g[i];
        }
        Order1 { v: self.v + o.v, g }
    }
    fn sub(&self, o: &Self) -> Self {
        // Mirror Order2::sub == self + o.scale(-1.0) exactly: scale then add.
        self.add(&o.scale(-1.0))
    }
    fn mul(&self, o: &Self) -> Self {
        // Tower2::mul value/grad terms, identical float order:
        //   v = a.v*b.v;  g[i] = a.v*b.g[i] + a.g[i]*b.v.
        // (The Hessian loop `a.v*b.h + a.g*b.g + ... + a.h*b.v` is the discarded
        //  work this type exists to skip; it never feeds v or g.)
        let a = self;
        let b = o;
        let mut g = [0.0; K];
        for i in 0..K {
            g[i] = a.v * b.g[i] + a.g[i] * b.v;
        }
        Order1 { v: a.v * b.v, g }
    }
    fn neg(&self) -> Self {
        // Order2::neg == self.0.scale(-1.0).
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        // Tower2::scale: out.v *= s; out.g[i] *= s (same float order).
        let mut g = self.g;
        for i in 0..K {
            g[i] *= s;
        }
        Order1 { v: self.v * s, g }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Faà di Bruno truncated to order ≤ 1 (matches `faa_di_bruno` /
        // `Tower2::compose_unary` for the value and gradient channels):
        //   value channel (m=0): d[0].
        //   grad channel (positions=[i], single partition {{0}}): d[1]·g[i].
        // Order-≤1 reads only d[0], d[1]; trailing stack entries are unused.
        let mut g = [0.0; K];
        for i in 0..K {
            g[i] = d[1] * self.g[i];
        }
        Order1 { v: d[0], g }
    }
}

// ── OneSeed<K>: one-seed directional, contracted third (doc §A.2) ───────

/// One-seed directional scalar: an [`Order2`] base plus ONE nilpotent ε
/// (`ε² = 0`) whose coefficient is itself an [`Order2`].
///
/// A scalar is `s = base + ε·eps`. Arithmetic is the `ε² = 0` truncation of the
/// product (doc §A.2): the base parts multiply as ordinary [`Order2`] products,
/// and the ε-coefficient picks up `a.base·b.eps + a.eps·b.base`. Composition
/// pushes ε through one extra outer derivative.
///
/// Seed each primary with [`seed_direction`](Self::seed_direction): the base is
/// the usual seeded variable (carrying `e_a` for the Hessian channel) and the
/// ε-coefficient is the FIXED contraction direction `u_a` (a constant). Then the
/// ε-component of the evaluated Hessian channel is the contracted third
/// `[eps.h][a][b] = Σ_c ℓ_{abc} u_c` — exactly `row_third_contracted(dir = u)`,
/// without materialising `t3`.
#[derive(Clone, Copy, Debug)]
pub struct OneSeed<const K: usize> {
    /// The `ε⁰` part: value / gradient / Hessian of `ℓ`.
    pub base: Order2<K>,
    /// The `ε¹` part: value / gradient / Hessian of the ε-coefficient. After a
    /// `seed_direction(u)` evaluation, `eps.h[a][b] = Σ_c ℓ_{abc} u_c`.
    pub eps: Order2<K>,
}

impl<const K: usize> OneSeed<K> {
    /// Seed primary `axis` at value `x` with ε-direction component `u_axis`:
    /// `p_axis = p_axis⁰ + x-seed + ε·u_axis`, i.e. base = `variable(x, axis)`
    /// and eps = `constant(u_axis)` (doc §A.2 "Seeding").
    pub fn seed_direction(x: f64, axis: usize, u_axis: f64) -> Self {
        OneSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(u_axis),
        }
    }

    /// The contracted-third channel after a `seed_direction(u)` evaluation:
    /// `out[a][b] = Σ_c ℓ_{abc} u_c`, i.e. the ε-coefficient's Hessian (doc §A.2).
    pub fn contracted_third(&self) -> [[f64; K]; K] {
        self.eps.h()
    }
}

impl<const K: usize> JetScalar<K> for OneSeed<K> {
    fn constant(c: f64) -> Self {
        OneSeed {
            base: Order2::constant(c),
            eps: Order2::constant(0.0),
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        // No ε-direction unless seeded via `seed_direction`.
        OneSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(0.0),
        }
    }
    fn value(&self) -> f64 {
        self.base.value()
    }
    fn add(&self, o: &Self) -> Self {
        OneSeed {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        OneSeed {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        // (a.base + ε a.eps)(b.base + ε b.eps), dropping ε².
        OneSeed {
            base: self.base.mul(&o.base),
            eps: self.base.mul(&o.eps).add(&self.eps.mul(&o.base)),
        }
    }
    fn neg(&self) -> Self {
        OneSeed {
            base: self.base.neg(),
            eps: self.eps.neg(),
        }
    }
    fn scale(&self, s: f64) -> Self {
        OneSeed {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // f(base + ε eps) = f(base) + ε · f'(base)·eps  (ε² = 0). Each factor is
        // an Order2 composition: the base composes with the f-stack, and the
        // ε-coefficient is the Order2 of the SHIFTED stack (the chain rule
        // `f'(base)` as an Order2) times eps. Order2 reads only the leading
        // three entries of whatever stack it is handed, so the trailing slots
        // are unused padding (the fixed-length array makes the windowing total).
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        // f'(base) as an Order2 (consumes [f', f'', f''']).
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        OneSeed { base, eps }
    }
}

// ── TwoSeed<K>: two-seed, contracted fourth (doc §A.3) ──────────────────

/// Two-seed scalar: an [`Order2`] base plus TWO nilpotents ε, δ
/// (`ε² = δ² = 0`, `εδ` retained) — four [`Order2`] parts
/// `s = base + ε·eps + δ·del + εδ·eps_del`.
///
/// Product truncates `ε² = δ² = 0` (doc §A.3): each part is built from
/// [`Order2`] products of the four input parts. Composition picks up
/// successively higher outer derivatives, the cross part carrying the second
/// Faà di Bruno term `f''·eps·del + f'·eps_del`.
///
/// Seed each primary with [`seed`](Self::seed): base = `variable(x, axis)`,
/// eps = `constant(u_axis)`, del = `constant(v_axis)`, eps_del = `constant(0)`.
/// Then the εδ-component of the evaluated Hessian channel is the contracted
/// fourth `[eps_del.h][a][b] = Σ_{cd} ℓ_{abcd} u_c v_d` — exactly
/// `row_fourth_contracted(u, v)`, without materialising `t4`.
#[derive(Clone, Copy, Debug)]
pub struct TwoSeed<const K: usize> {
    /// The `ε⁰δ⁰` part: value / grad / Hessian of `ℓ`.
    pub base: Order2<K>,
    /// The `ε¹δ⁰` part.
    pub eps: Order2<K>,
    /// The `ε⁰δ¹` part.
    pub del: Order2<K>,
    /// The `ε¹δ¹` part. After a `seed(u, v)` evaluation,
    /// `eps_del.h[a][b] = Σ_{cd} ℓ_{abcd} u_c v_d`.
    pub eps_del: Order2<K>,
}

impl<const K: usize> TwoSeed<K> {
    /// Seed primary `axis` at value `x` with ε-direction `u_axis` and
    /// δ-direction `v_axis`:
    /// `p_axis = p_axis⁰ + x-seed + ε·u_axis + δ·v_axis` (doc §A.3 "Seeding").
    pub fn seed(x: f64, axis: usize, u_axis: f64, v_axis: f64) -> Self {
        TwoSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(u_axis),
            del: Order2::constant(v_axis),
            eps_del: Order2::constant(0.0),
        }
    }

    /// The contracted-fourth channel after a `seed(u, v)` evaluation:
    /// `out[a][b] = Σ_{cd} ℓ_{abcd} u_c v_d`, i.e. the εδ-coefficient's Hessian.
    pub fn contracted_fourth(&self) -> [[f64; K]; K] {
        self.eps_del.h()
    }
}

impl<const K: usize> JetScalar<K> for TwoSeed<K> {
    fn constant(c: f64) -> Self {
        TwoSeed {
            base: Order2::constant(c),
            eps: Order2::constant(0.0),
            del: Order2::constant(0.0),
            eps_del: Order2::constant(0.0),
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        TwoSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(0.0),
            del: Order2::constant(0.0),
            eps_del: Order2::constant(0.0),
        }
    }
    fn value(&self) -> f64 {
        self.base.value()
    }
    fn add(&self, o: &Self) -> Self {
        TwoSeed {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        TwoSeed {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        // Truncate ε² = δ² = 0 (doc §A.3 product table).
        let base = a.base.mul(&b.base);
        let eps = a.base.mul(&b.eps).add(&a.eps.mul(&b.base));
        let del = a.base.mul(&b.del).add(&a.del.mul(&b.base));
        let eps_del = a
            .base
            .mul(&b.eps_del)
            .add(&a.eps.mul(&b.del))
            .add(&a.del.mul(&b.eps))
            .add(&a.eps_del.mul(&b.base));
        TwoSeed {
            base,
            eps,
            del,
            eps_del,
        }
    }
    fn neg(&self) -> Self {
        TwoSeed {
            base: self.base.neg(),
            eps: self.eps.neg(),
            del: self.del.neg(),
            eps_del: self.eps_del.neg(),
        }
    }
    fn scale(&self, s: f64) -> Self {
        TwoSeed {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // f(s) with s = base + ε eps + δ del + εδ eps_del, ε²=δ²=0:
        //   f(s) = f(base)
        //        + ε · f'(base)·eps
        //        + δ · f'(base)·del
        //        + εδ · ( f''(base)·eps·del + f'(base)·eps_del ).
        // Each f^{(r)}(base) is the Order2 composition of base with the stack
        // shifted r entries (doc §A.3 composition). Order2 reads only the
        // leading three entries of whatever stack it is handed, so the trailing
        // padding slots are unused (the fixed-length array makes this total).
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]); // f'(base) as Order2
        let fsecond = self.base.compose_unary([d[2], d[3], d[4], d[4], d[4]]); // f''(base) as Order2
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        TwoSeed {
            base,
            eps,
            del,
            eps_del,
        }
    }
}

// ── Tower3<K>: value / gradient / Hessian / third tensor ────────────────

/// The order-≤3 [`crate::jet_tower::Tower3`] is also a [`JetScalar`]. It serves
/// consumers that read `.t3` but never `.t4`, avoiding the fourth-tensor
/// product/composition work while preserving the lower channels
/// bit-for-bit against [`crate::jet_tower::Tower4`].
impl<const K: usize> JetScalar<K> for crate::jet_tower::Tower3<K> {
    fn constant(c: f64) -> Self {
        crate::jet_tower::Tower3::constant(c)
    }
    fn variable(x: f64, axis: usize) -> Self {
        crate::jet_tower::Tower3::variable(x, axis)
    }
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        *self + *o
    }
    fn sub(&self, o: &Self) -> Self {
        *self + o.scale(-1.0)
    }
    fn mul(&self, o: &Self) -> Self {
        crate::jet_tower::Tower3::mul(self, o)
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        crate::jet_tower::Tower3::scale(self, s)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        crate::jet_tower::Tower3::compose_unary(self, [d[0], d[1], d[2], d[3]])
    }
}

// ── Tower4<K>: full dense tower as a JetScalar (the all-channels scalar) ─

/// The full dense [`crate::jet_tower::Tower4`] is itself a [`JetScalar`]: it
/// carries EVERY channel, so a row expression written ONCE against [`JetScalar`]
/// can be evaluated at `Tower4` to obtain the full `(v, g, H, t3, t4)` in one
/// pass. This is BOTH the #932 oracle ground truth the packed [`Order2`] /
/// [`OneSeed`] / [`TwoSeed`] scalars are pinned against, AND a production scalar:
/// a family whose uncontracted third / fourth derivative tensors are needed
/// (the BMS rigid `third_full` / `fourth_full` caches) evaluates the SAME
/// generic row-NLL expression at `Tower4` and reads `.t3` / `.t4` off the
/// result — so the dense tensors come from the single source of truth, not a
/// separately hand-written jet. The packed scalars serve the consumers that
/// need only `(v, g, H)` (`Order2`) or one / two contractions
/// (`OneSeed` / `TwoSeed`) without paying for the dense tensors.
impl<const K: usize> JetScalar<K> for crate::jet_tower::Tower4<K> {
    fn constant(c: f64) -> Self {
        crate::jet_tower::Tower4::constant(c)
    }
    fn variable(x: f64, axis: usize) -> Self {
        crate::jet_tower::Tower4::variable(x, axis)
    }
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        *self + *o
    }
    fn sub(&self, o: &Self) -> Self {
        *self - *o
    }
    fn mul(&self, o: &Self) -> Self {
        crate::jet_tower::Tower4::mul(self, o)
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        crate::jet_tower::Tower4::scale(self, s)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        crate::jet_tower::Tower4::compose_unary(self, d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jet_tower::{RowNllProgram, Tower4, evaluate_program};

    /// A small polynomial-plus-unary row expression written ONCE, generically
    /// over `S: JetScalar<2>`, so it can be evaluated against every scalar:
    /// `ℓ = (e^{p0·p1} + 2) · √(p0·p0 + 1) − p1·p1·0.5`.
    /// Exercises mul, add/sub, scale, exp, sqrt — every algebra op.
    fn row_expr<S: JetScalar<2>>(p: &[S; 2]) -> S {
        let g = p[0].mul(&p[1]).exp();
        let inner = g.add(&S::constant(2.0));
        let radic = p[0].mul(&p[0]).add(&S::constant(1.0)).sqrt();
        inner.mul(&radic).sub(&p[1].mul(&p[1]).scale(0.5))
    }

    /// The same expression as a Tower4 `RowNllProgram`, the ground-truth tower.
    struct ExprProgram {
        p: [f64; 2],
    }
    impl RowNllProgram<2> for ExprProgram {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
            if row >= self.n_rows() {
                return Err(format!("ExprProgram: row {row} out of range"));
            }
            Ok(self.p)
        }
        fn row_nll(&self, row: usize, p: &[Tower4<2>; 2]) -> Result<Tower4<2>, String> {
            if row >= self.n_rows() {
                return Err(format!("ExprProgram: row {row} out of range"));
            }
            Ok(row_expr(p))
        }
    }

    const SEED: [f64; 2] = [0.37, -0.81];
    const U: [f64; 2] = [0.6, -0.2];
    const V: [f64; 2] = [-0.4, 1.1];
    const TOL: f64 = 1e-10;

    fn close(a: f64, b: f64, label: &str) {
        let band = TOL + TOL * a.abs().max(b.abs());
        assert!(
            (a - b).abs() <= band,
            "{label}: {a:+.15e} vs {b:+.15e} (band {band:.3e})"
        );
    }

    fn tower() -> Tower4<2> {
        evaluate_program(&ExprProgram { p: SEED }, 0).expect("tower")
    }

    /// Order2 reproduces Tower4's value/grad/Hessian channels exactly.
    #[test]
    fn order2_matches_tower_value_grad_hessian() {
        let t = tower();
        let vars: [Order2<2>; 2] = std::array::from_fn(|a| Order2::variable(SEED[a], a));
        let s = row_expr(&vars);
        close(s.value(), t.v, "value");
        for a in 0..2 {
            close(s.0.g[a], t.g[a], &format!("grad[{a}]"));
            for b in 0..2 {
                close(s.h()[a][b], t.h[a][b], &format!("hess[{a}][{b}]"));
            }
        }
    }

    /// OneSeed's ε-Hessian is the contracted third Σ_c ℓ_{abc} u_c, matching
    /// `Tower4::third_contracted(u)`. Base channels also match the tower.
    #[test]
    fn one_seed_matches_tower_third_contracted() {
        let t = tower();
        let truth = t.third_contracted(&U);
        let vars: [OneSeed<2>; 2] =
            std::array::from_fn(|a| OneSeed::seed_direction(SEED[a], a, U[a]));
        let s = row_expr(&vars);
        // Base channels are the plain (v, g, H).
        close(s.value(), t.v, "value");
        for a in 0..2 {
            for b in 0..2 {
                close(s.base.h()[a][b], t.h[a][b], &format!("base hess[{a}][{b}]"));
            }
        }
        let third = s.contracted_third();
        for a in 0..2 {
            for b in 0..2 {
                close(third[a][b], truth[a][b], &format!("third[{a}][{b}]"));
            }
        }
    }

    /// TwoSeed's εδ-Hessian is the contracted fourth Σ_{cd} ℓ_{abcd} u_c v_d,
    /// matching `Tower4::fourth_contracted(u, v)`. The ε / δ single-seed parts
    /// reproduce the two third contractions Σ_c ℓ_{abc} u_c and …v_d.
    #[test]
    fn two_seed_matches_tower_fourth_contracted() {
        let t = tower();
        let truth4 = t.fourth_contracted(&U, &V);
        let truth3_u = t.third_contracted(&U);
        let truth3_v = t.third_contracted(&V);
        let vars: [TwoSeed<2>; 2] = std::array::from_fn(|a| TwoSeed::seed(SEED[a], a, U[a], V[a]));
        let s = row_expr(&vars);
        close(s.value(), t.v, "value");
        for a in 0..2 {
            close(s.base.0.g[a], t.g[a], &format!("grad[{a}]"));
            for b in 0..2 {
                close(s.base.h()[a][b], t.h[a][b], &format!("base hess[{a}][{b}]"));
                close(
                    s.eps.h()[a][b],
                    truth3_u[a][b],
                    &format!("eps third_u[{a}][{b}]"),
                );
                close(
                    s.del.h()[a][b],
                    truth3_v[a][b],
                    &format!("del third_v[{a}][{b}]"),
                );
            }
        }
        let fourth = s.contracted_fourth();
        for a in 0..2 {
            for b in 0..2 {
                close(fourth[a][b], truth4[a][b], &format!("fourth[{a}][{b}]"));
            }
        }
    }

    /// The generic `row_nll_generic` seam (added to Tower4's program trait
    /// surface) evaluates the SAME expression on each scalar and extracts the
    /// channel a consumer asks for, agreeing with the direct Tower4 contraction.
    #[test]
    fn generic_program_seam_matches_tower_for_every_channel() {
        let t = tower();
        // Order2 via generic seam.
        let o2: [Order2<2>; 2] = std::array::from_fn(|a| Order2::variable(SEED[a], a));
        let so2 = row_expr(&o2);
        close(so2.value(), t.v, "seam order2 value");
        // OneSeed third.
        let os: [OneSeed<2>; 2] =
            std::array::from_fn(|a| OneSeed::seed_direction(SEED[a], a, U[a]));
        let third = row_expr(&os).contracted_third();
        let truth3 = t.third_contracted(&U);
        for a in 0..2 {
            for b in 0..2 {
                close(third[a][b], truth3[a][b], &format!("seam third[{a}][{b}]"));
            }
        }
        // TwoSeed fourth.
        let ts: [TwoSeed<2>; 2] = std::array::from_fn(|a| TwoSeed::seed(SEED[a], a, U[a], V[a]));
        let fourth = row_expr(&ts).contracted_fourth();
        let truth4 = t.fourth_contracted(&U, &V);
        for a in 0..2 {
            for b in 0..2 {
                close(
                    fourth[a][b],
                    truth4[a][b],
                    &format!("seam fourth[{a}][{b}]"),
                );
            }
        }
    }

    /// The (test-only) `Tower4: JetScalar` impl is the all-channels oracle scalar:
    /// evaluating the SAME generic `row_expr` at `S = Tower4` (through the
    /// `JetScalar` trait ops) must reproduce, channel-for-channel, the `Tower4`
    /// obtained from the `RowNllProgram` / inherent-operator path
    /// (`evaluate_program`). This pins that the trait impl delegates faithfully to
    /// the inherent `Tower4` arithmetic (so the contracted-scalar oracles above,
    /// which compare against `evaluate_program`'s tower, are comparing against the
    /// same algebra the `JetScalar` interface exposes).
    #[test]
    fn tower4_as_jetscalar_matches_program_tower_all_channels() {
        let t = tower();
        let vars: [Tower4<2>; 2] = std::array::from_fn(|a| Tower4::variable(SEED[a], a));
        let s = row_expr(&vars);
        close(s.v, t.v, "tower-jetscalar value");
        for a in 0..2 {
            close(s.g[a], t.g[a], &format!("tower-jetscalar grad[{a}]"));
            for b in 0..2 {
                close(
                    s.h[a][b],
                    t.h[a][b],
                    &format!("tower-jetscalar hess[{a}][{b}]"),
                );
                for c in 0..2 {
                    close(
                        s.t3[a][b][c],
                        t.t3[a][b][c],
                        &format!("tower-jetscalar t3[{a}][{b}][{c}]"),
                    );
                    for d in 0..2 {
                        close(
                            s.t4[a][b][c][d],
                            t.t4[a][b][c][d],
                            &format!("tower-jetscalar t4[{a}][{b}][{c}][{d}]"),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod batch_tests {
    //! SIMD row-batching oracle: prove [`Order2Batch<K>`] (4 rows in
    //! `wide::f64x4` lanes) is `to_bits`-identical, on every value/gradient/
    //! Hessian channel, to the production [`Order2<K>`] evaluated per row — and
    //! that the new scalar field [`Order2Lane<f64, K>`] is too. Composing the two
    //! claims, batch lane `i` reproduces the production scalar for row `i` bit
    //! for bit, so the 4× throughput is a free lunch (no result change).

    use super::{JetScalar, Lane, Order2, Order2Batch, Order2Lane};

    /// The ops the witness row expression needs, so ONE generic body evaluates
    /// at the production [`Order2<K>`], the new scalar [`Order2Lane<f64, K>`],
    /// and the batched [`Order2Batch<K>`].
    trait RowAlg<const K: usize>: Copy {
        fn constant(c: f64) -> Self;
        fn add(&self, o: &Self) -> Self;
        fn sub(&self, o: &Self) -> Self;
        fn mul(&self, o: &Self) -> Self;
        fn scale(&self, s: f64) -> Self;
        fn exp(&self) -> Self;
        fn sqrt(&self) -> Self;
        fn recip(&self) -> Self;
    }

    impl<const K: usize> RowAlg<K> for Order2<K> {
        fn constant(c: f64) -> Self {
            <Self as JetScalar<K>>::constant(c)
        }
        fn add(&self, o: &Self) -> Self {
            JetScalar::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            JetScalar::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            JetScalar::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            JetScalar::scale(self, s)
        }
        fn exp(&self) -> Self {
            JetScalar::exp(self)
        }
        fn sqrt(&self) -> Self {
            JetScalar::sqrt(self)
        }
        fn recip(&self) -> Self {
            JetScalar::recip(self)
        }
    }

    impl<L: Lane, const K: usize> RowAlg<K> for Order2Lane<L, K> {
        fn constant(c: f64) -> Self {
            Order2Lane::constant(L::splat(c))
        }
        fn add(&self, o: &Self) -> Self {
            Order2Lane::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            Order2Lane::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            Order2Lane::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            Order2Lane::scale(self, s)
        }
        fn exp(&self) -> Self {
            Order2Lane::exp(self)
        }
        fn sqrt(&self) -> Self {
            Order2Lane::sqrt(self)
        }
        fn recip(&self) -> Self {
            Order2Lane::recip(self)
        }
    }

    /// A dense witness row expression touching every algebra op (mul, add, sub,
    /// scale, exp, sqrt, recip) over ALL `K` primaries, so the gradient and the
    /// full `K×K` Hessian are dense (no trivially-zero channel). All transcend.
    /// arguments are kept finite/positive: `sqrt(s²+1) > 0`, `recip(exp+2) > 0`.
    fn row_expr<const K: usize, A: RowAlg<K>>(p: &[A; K]) -> A {
        let mut s = A::constant(0.3);
        for a in 0..K {
            let b = (a + 1) % K;
            s = s.add(&p[a].mul(&p[b]).scale(0.1 + 0.05 * a as f64));
        }
        let e = s.exp();
        let r = s.mul(&s).add(&A::constant(1.0)).sqrt();
        let denom = e.add(&A::constant(2.0));
        e.mul(&r).sub(&s.scale(0.5)).mul(&denom.recip())
    }

    /// xorshift64 → `f64` in `[-1, 1)`.
    fn rand_unit(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        let u = (x >> 11) as f64 / ((1u64 << 53) as f64); // [0, 1)
        2.0 * u - 1.0
    }

    fn check_k<const K: usize>(state: &mut u64, batches: usize) {
        for _ in 0..batches {
            // Four independent rows of K primary values.
            let rows: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));

            // Production ground truth, evaluated per row at Order2<K>.
            let prod: [Order2<K>; 4] = std::array::from_fn(|r| {
                let p: [Order2<K>; K] = std::array::from_fn(|a| Order2::variable(rows[r][a], a));
                row_expr(&p)
            });

            // New scalar field (Order2Lane<f64>), per row.
            let scal: [Order2Lane<f64, K>; 4] = std::array::from_fn(|r| {
                let p: [Order2Lane<f64, K>; K] =
                    std::array::from_fn(|a| Order2Lane::variable(rows[r][a], a));
                row_expr(&p)
            });

            // Batched: 4 rows packed into f64x4 lanes, ONE vector pass.
            let pbatch: [Order2Batch<K>; K] = std::array::from_fn(|a| {
                let packed =
                    wide::f64x4::new([rows[0][a], rows[1][a], rows[2][a], rows[3][a]]);
                Order2Batch::variable(packed, a)
            });
            let batch = row_expr(&pbatch);

            for r in 0..4 {
                let g = prod[r].0;
                // Order2Lane<f64> == Order2<K> (bit-identical scalar field).
                assert_eq!(scal[r].v.to_bits(), g.v.to_bits(), "K={K} scalar v");
                // Batch lane r == Order2<K> for row r.
                let lr = batch.lane(r).0;
                assert_eq!(lr.v.to_bits(), g.v.to_bits(), "K={K} batch lane {r} v");
                for a in 0..K {
                    assert_eq!(
                        scal[r].g[a].to_bits(),
                        g.g[a].to_bits(),
                        "K={K} scalar g[{a}]"
                    );
                    assert_eq!(
                        lr.g[a].to_bits(),
                        g.g[a].to_bits(),
                        "K={K} batch lane {r} g[{a}]"
                    );
                    for b in 0..K {
                        assert_eq!(
                            scal[r].h[a][b].to_bits(),
                            g.h[a][b].to_bits(),
                            "K={K} scalar h[{a}][{b}]"
                        );
                        assert_eq!(
                            lr.h[a][b].to_bits(),
                            g.h[a][b].to_bits(),
                            "K={K} batch lane {r} h[{a}][{b}]"
                        );
                    }
                }
            }
        }
    }

    /// ≥2000 random 4-row batches per K, across K ∈ {2,3,4,9}: every channel of
    /// every lane is `to_bits`-identical to the production scalar per row.
    #[test]
    fn batch_lanes_bit_identical_to_scalar_per_row() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        check_k::<2>(&mut state, 2000);
        check_k::<3>(&mut state, 2000);
        check_k::<4>(&mut state, 2000);
        check_k::<9>(&mut state, 2000);
    }
}
