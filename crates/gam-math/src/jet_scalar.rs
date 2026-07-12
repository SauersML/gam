//! Order-specific Taylor-jet SCALAR algebras (#932 cutover, doc §A).
//!
//! [`crate::jet_tower::Tower4`] carries the full value/gradient/Hessian/`t3`/`t4`
//! tensor stack: it answers EVERY channel a [`super::row_kernel::RowKernel`]
//! consumer can ask for, but at `K = 9` that is a ~50 KiB per-row object whose
//! by-value copies overflowed the stack and timed out the location-scale fit.
//! The cutover therefore does NOT instantiate the dense `Tower4<9>` per row; it
//! carries, per consumer, only the one channel that consumer serves:
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

    /// Compose with a unary special-function whose derivative STACK is built
    /// from the scalar base value through `stack_fn` — the generic-over-`Lane`
    /// seam that lets a single-sourced row program instantiate at BOTH the scalar
    /// `f64` jets and the SIMD `f64x4` batch towers from ONE expression.
    ///
    /// On a scalar jet this evaluates `stack_fn(self.value())` ONCE and forwards
    /// to [`compose_unary`](Self::compose_unary), so it is BIT-IDENTICAL to the
    /// hand-written `self.compose_unary(stack_fn(self.value()))` (default body
    /// below). The lever is that the SAME call shape exists on
    /// [`crate::jet_tower::Tower3Lane`] / [`crate::jet_tower::Tower4Lane`], where
    /// the four lanes carry FOUR DISTINCT base values, so the batch
    /// implementation re-runs `stack_fn` per lane — a thing the old
    /// `compose_unary(stack_from(self.value()))` shape could not express on a
    /// batch type (it has no single scalar `.value()`). Writing a row program
    /// against this method instead of the explicit two-step is what makes it
    /// instantiate, unchanged, at `f64x4` for the 4-rows-per-pass batch path.
    fn compose_unary_with(&self, stack_fn: impl Fn(f64) -> [f64; 5]) -> Self {
        self.compose_unary(stack_fn(self.value()))
    }

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

/// A Taylor-jet scalar whose primary dimension is selected at runtime.
///
/// This is the dimensioned counterpart of [`JetScalar`].  Its algebra is the
/// same; only the constructors receive the row's actual primary count.  The
/// fixed-size scalar implementations below bridge to this trait as well, which
/// lets one row program serve both the const-generic derivative oracles and the
/// runtime-sized production backends without duplicating the expression.
pub trait RuntimeJetScalar<'arena>: Clone {
    /// Storage arena used by runtime-backed scalars. Fixed derivative oracles
    /// use the unit type because their storage is inline.
    type Workspace: ?Sized;

    /// A constant in a `dimension`-primary algebra.
    fn constant(c: f64, dimension: usize, workspace: &'arena Self::Workspace) -> Self;
    /// A seeded variable in a `dimension`-primary algebra.
    fn variable(x: f64, axis: usize, dimension: usize, workspace: &'arena Self::Workspace) -> Self;
    /// Number of primary derivative axes carried by this scalar.
    fn dimension(&self) -> usize;
    /// Value channel.
    fn value(&self) -> f64;
    /// Exact truncated sum.
    fn add(&self, o: &Self) -> Self;
    /// Exact truncated difference.
    fn sub(&self, o: &Self) -> Self;
    /// Exact truncated product.
    fn mul(&self, o: &Self) -> Self;
    /// Negate every channel.
    fn neg(&self) -> Self;
    /// Scale every channel.
    fn scale(&self, s: f64) -> Self;
    /// Exact unary composition from the certified derivative stack.
    fn compose_unary(&self, d: [f64; 5]) -> Self;

    /// `e^self`.
    fn exp(&self) -> Self {
        let e = self.value().exp();
        self.compose_unary([e, e, e, e, e])
    }

    /// `1/self`.
    fn recip(&self) -> Self {
        let r = 1.0 / self.value();
        let r2 = r * r;
        self.compose_unary([r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r])
    }
}

/// Adapter that presents any const-generic [`JetScalar<K>`] through the
/// runtime-dimension interface.  It is used by derivative oracles so the same
/// row program can be instantiated at a fixed tower and at a dynamic packed
/// scalar; production code unwraps the inner fixed tower after evaluation.
#[derive(Clone, Copy, Debug)]
pub struct FixedRuntimeJet<S, const K: usize> {
    inner: S,
}

impl<S, const K: usize> FixedRuntimeJet<S, K> {
    /// Recover the wrapped const-generic scalar.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<'arena, S: JetScalar<K>, const K: usize> RuntimeJetScalar<'arena> for FixedRuntimeJet<S, K> {
    type Workspace = ();

    fn constant(c: f64, dimension: usize, &(): &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        Self {
            inner: S::constant(c),
        }
    }

    fn variable(x: f64, axis: usize, dimension: usize, &(): &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        Self {
            inner: S::variable(x, axis),
        }
    }

    fn dimension(&self) -> usize {
        K
    }

    fn value(&self) -> f64 {
        self.inner.value()
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            inner: self.inner.add(&o.inner),
        }
    }

    fn sub(&self, o: &Self) -> Self {
        Self {
            inner: self.inner.sub(&o.inner),
        }
    }

    fn mul(&self, o: &Self) -> Self {
        Self {
            inner: self.inner.mul(&o.inner),
        }
    }

    fn neg(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    fn scale(&self, s: f64) -> Self {
        Self {
            inner: self.inner.scale(s),
        }
    }

    fn compose_unary(&self, d: [f64; 5]) -> Self {
        Self {
            inner: self.inner.compose_unary(d),
        }
    }
}

/// Reusable storage for runtime-sized packed jets. Scalar primitives write
/// into this bump arena, so arithmetic performs no heap allocation. Callers
/// reserve once per worker/chunk and [`reset`](Self::reset) between rows.
#[derive(Debug)]
pub struct DynamicJetArena {
    bump: bumpalo::Bump,
}

impl DynamicJetArena {
    /// Create an arena with the allocator's default initial chunk.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: bumpalo::Bump::new(),
        }
    }

    /// Create an arena with a row-program-selected initial byte capacity.
    #[must_use]
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            bump: bumpalo::Bump::with_capacity(bytes),
        }
    }

    /// Reclaim all scalar outputs while retaining allocated chunks.
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Bytes currently reserved from the global allocator. A warm-reset-warm
    /// benchmark uses this to prove the second row requires no arena growth.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }

    #[inline]
    fn zeros(&self, len: usize) -> &mut [f64] {
        self.bump.alloc_slice_fill_copy(len, 0.0)
    }

    /// Allocate and initialize a runtime-sized slice in the arena. Row programs
    /// use this for their primary-scalar arrays so those arrays share the same
    /// reusable workspace as derivative channels.
    pub fn alloc_slice_fill_with<T>(&self, len: usize, fill: impl FnMut(usize) -> T) -> &mut [T] {
        self.bump.alloc_slice_fill_with(len, fill)
    }
}

impl Default for DynamicJetArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime-sized packed first-order scalar: value plus arena-backed gradient.
#[derive(Clone, Copy, Debug)]
pub struct DynamicOrder1<'arena> {
    arena: &'arena DynamicJetArena,
    /// Value channel.
    pub v: f64,
    /// Gradient channel, length [`Self::dimension`].
    pub g: &'arena [f64],
}

impl DynamicOrder1<'_> {
    /// Gradient channel.
    #[inline]
    #[must_use]
    pub fn g(&self) -> &[f64] {
        self.g
    }

    #[inline]
    fn assert_compatible(&self, o: &Self) {
        assert_eq!(
            self.g.len(),
            o.g.len(),
            "dynamic first-order jet dimension mismatch"
        );
        assert!(
            std::ptr::eq(self.arena, o.arena),
            "dynamic jets belong to different arenas"
        );
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicOrder1<'arena> {
    type Workspace = DynamicJetArena;

    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            arena,
            v: c,
            g: arena.zeros(dimension),
        }
    }

    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        assert!(
            axis < dimension,
            "dynamic first-order jet axis out of bounds"
        );
        let g = arena.zeros(dimension);
        g[axis] = 1.0;
        Self { arena, v: x, g }
    }

    fn dimension(&self) -> usize {
        self.g.len()
    }
    fn value(&self) -> f64 {
        self.v
    }

    fn add(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.g[i] + o.g[i];
        }
        Self {
            arena: self.arena,
            v: self.v + o.v,
            g,
        }
    }

    fn sub(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.g[i] - o.g[i];
        }
        Self {
            arena: self.arena,
            v: self.v - o.v,
            g,
        }
    }

    fn mul(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        Self {
            arena: self.arena,
            v: self.v * o.v,
            g,
        }
    }

    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    fn scale(&self, s: f64) -> Self {
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.g[i] * s;
        }
        Self {
            arena: self.arena,
            v: self.v * s,
            g,
        }
    }

    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = d[1] * self.g[i];
        }
        Self {
            arena: self.arena,
            v: d[0],
            g,
        }
    }
}

/// Runtime-sized packed second-order scalar: value, gradient, and a row-major
/// Hessian. Storage is `O(K^2)` in the row's actual primary dimension and comes
/// from the row's reusable [`DynamicJetArena`].
#[derive(Clone, Copy, Debug)]
pub struct DynamicOrder2<'arena> {
    arena: &'arena DynamicJetArena,
    /// Value channel.
    pub v: f64,
    /// Gradient channel.
    pub g: &'arena [f64],
    /// Row-major Hessian channel.
    pub h: &'arena [f64],
}

impl DynamicOrder2<'_> {
    /// Gradient channel.
    #[inline]
    #[must_use]
    pub fn g(&self) -> &[f64] {
        self.g
    }

    /// Row-major Hessian channel.
    #[inline]
    #[must_use]
    pub fn h(&self) -> &[f64] {
        self.h
    }

    /// Hessian entry `(row, col)`.
    #[inline]
    #[must_use]
    pub fn h_at(&self, row: usize, col: usize) -> f64 {
        self.h[row * self.dimension() + col]
    }

    #[inline]
    fn assert_compatible(&self, o: &Self) {
        assert_eq!(
            self.g.len(),
            o.g.len(),
            "dynamic second-order jet dimension mismatch"
        );
        assert_eq!(
            self.h.len(),
            o.h.len(),
            "dynamic second-order jet Hessian mismatch"
        );
        assert!(
            std::ptr::eq(self.arena, o.arena),
            "dynamic jets belong to different arenas"
        );
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicOrder2<'arena> {
    type Workspace = DynamicJetArena;

    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            arena,
            v: c,
            g: arena.zeros(dimension),
            h: arena.zeros(dimension * dimension),
        }
    }

    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        assert!(
            axis < dimension,
            "dynamic second-order jet axis out of bounds"
        );
        let g = arena.zeros(dimension);
        g[axis] = 1.0;
        Self {
            arena,
            v: x,
            g,
            h: arena.zeros(dimension * dimension),
        }
    }

    fn dimension(&self) -> usize {
        self.g.len()
    }

    fn value(&self) -> f64 {
        self.v
    }

    fn add(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        let h = self.arena.zeros(self.h.len());
        for i in 0..g.len() {
            g[i] = self.g[i] + o.g[i];
        }
        for i in 0..h.len() {
            h[i] = self.h[i] + o.h[i];
        }
        Self {
            arena: self.arena,
            v: self.v + o.v,
            g,
            h,
        }
    }

    fn sub(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        let h = self.arena.zeros(self.h.len());
        for i in 0..g.len() {
            g[i] = self.g[i] - o.g[i];
        }
        for i in 0..h.len() {
            h[i] = self.h[i] - o.h[i];
        }
        Self {
            arena: self.arena,
            v: self.v - o.v,
            g,
            h,
        }
    }

    fn mul(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let n = self.dimension();
        let g = self.arena.zeros(n);
        let h = self.arena.zeros(n * n);
        for i in 0..n {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        for i in 0..n {
            for j in i..n {
                let ij = i * n + j;
                let hij =
                    self.v * o.h[ij] + self.g[i] * o.g[j] + self.g[j] * o.g[i] + self.h[ij] * o.v;
                h[ij] = hij;
                h[j * n + i] = hij;
            }
        }
        Self {
            arena: self.arena,
            v: self.v * o.v,
            g,
            h,
        }
    }

    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    fn scale(&self, s: f64) -> Self {
        let g = self.arena.zeros(self.dimension());
        let h = self.arena.zeros(self.h.len());
        for i in 0..g.len() {
            g[i] = self.g[i] * s;
        }
        for i in 0..h.len() {
            h[i] = self.h[i] * s;
        }
        Self {
            arena: self.arena,
            v: self.v * s,
            g,
            h,
        }
    }

    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let n = self.dimension();
        let g = self.arena.zeros(n);
        let h = self.arena.zeros(n * n);
        for i in 0..n {
            g[i] = d[1] * self.g[i];
        }
        for i in 0..n {
            for j in 0..n {
                let ij = i * n + j;
                h[ij] = d[1] * self.h[ij] + d[2] * self.g[i] * self.g[j];
            }
        }
        Self {
            arena: self.arena,
            v: d[0],
            g,
            h,
        }
    }
}

/// Runtime-sized one-seed scalar for a Hessian-contracted third derivative.
#[derive(Clone, Copy, Debug)]
pub struct DynamicOneSeed<'arena> {
    /// Base value/gradient/Hessian channels.
    pub base: DynamicOrder2<'arena>,
    /// Nilpotent `epsilon` coefficient.
    pub eps: DynamicOrder2<'arena>,
}

impl<'arena> DynamicOneSeed<'arena> {
    /// Seed one primary with the supplied contraction direction component.
    #[must_use]
    pub fn seed_direction(
        x: f64,
        axis: usize,
        u_axis: f64,
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(u_axis, dimension, arena),
        }
    }

    /// Row-major contracted-third matrix.
    #[must_use]
    pub fn contracted_third(&self) -> &[f64] {
        self.eps.h()
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicOneSeed<'arena> {
    type Workspace = DynamicJetArena;

    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::constant(c, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    fn value(&self) -> f64 {
        self.base.value()
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }

    fn sub(&self, o: &Self) -> Self {
        Self {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }

    fn mul(&self, o: &Self) -> Self {
        Self {
            base: self.base.mul(&o.base),
            eps: self.base.mul(&o.eps).add(&self.eps.mul(&o.base)),
        }
    }

    fn neg(&self) -> Self {
        Self {
            base: self.base.neg(),
            eps: self.eps.neg(),
        }
    }

    fn scale(&self, s: f64) -> Self {
        Self {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }

    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary(d);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        Self { base, eps }
    }
}

/// Runtime-sized two-seed scalar for a Hessian-contracted fourth derivative.
#[derive(Clone, Copy, Debug)]
pub struct DynamicTwoSeed<'arena> {
    /// Base value/gradient/Hessian channels.
    pub base: DynamicOrder2<'arena>,
    /// Nilpotent `epsilon` coefficient.
    pub eps: DynamicOrder2<'arena>,
    /// Nilpotent `delta` coefficient.
    pub del: DynamicOrder2<'arena>,
    /// Mixed `epsilon delta` coefficient.
    pub eps_del: DynamicOrder2<'arena>,
}

impl<'arena> DynamicTwoSeed<'arena> {
    /// Seed one primary with both contraction direction components.
    #[must_use]
    pub fn seed(
        x: f64,
        axis: usize,
        u_axis: f64,
        v_axis: f64,
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(u_axis, dimension, arena),
            del: DynamicOrder2::constant(v_axis, dimension, arena),
            eps_del: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    /// Row-major contracted-fourth matrix.
    #[must_use]
    pub fn contracted_fourth(&self) -> &[f64] {
        self.eps_del.h()
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicTwoSeed<'arena> {
    type Workspace = DynamicJetArena;

    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::constant(c, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
            del: DynamicOrder2::constant(0.0, dimension, arena),
            eps_del: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
            del: DynamicOrder2::constant(0.0, dimension, arena),
            eps_del: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    fn value(&self) -> f64 {
        self.base.value()
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }

    fn sub(&self, o: &Self) -> Self {
        Self {
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
        Self {
            base,
            eps,
            del,
            eps_del,
        }
    }

    fn neg(&self) -> Self {
        Self {
            base: self.base.neg(),
            eps: self.eps.neg(),
            del: self.del.neg(),
            eps_del: self.eps_del.neg(),
        }
    }

    fn scale(&self, s: f64) -> Self {
        Self {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }

    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary(d);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let fsecond = self.base.compose_unary([d[2], d[3], d[4], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        Self {
            base,
            eps,
            del,
            eps_del,
        }
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
    /// Build the order-≤4 derivative stack `[f, f′, f″, f‴, f⁗]` **per lane**
    /// from the lane value `u`, via the SAME scalar `stack` closure the per-row
    /// path runs. The one-/two-seed scalars ([`OneSeedLane`] / [`TwoSeedLane`])
    /// need outer derivatives one / two orders beyond their order-2 base, so
    /// they build their composition stack through this five-entry variant. As
    /// with [`unary3`](Lane::unary3), only the transcendental/rational stack is
    /// evaluated per lane (bit-identically to the scalar path); the subsequent
    /// tensor composition is vectorised.
    fn unary5(self, stack: impl Fn(f64) -> [f64; 5]) -> [Self; 5];
    /// The general-`N` sibling of [`unary3`](Lane::unary3) / [`unary5`](Lane::unary5):
    /// build an `N`-wide derivative stack **per lane** from the lane value, via
    /// the SAME scalar `stack` closure the per-row path runs, then pack the `N`
    /// columns lane-wise. This is the lane primitive the compose-with-stack seam
    /// ([`crate::jet_tower::Tower4Lane::compose_unary_with`] and its `Tower3`
    /// sibling) routes through: it evaluates `stack` once per lane at that lane's
    /// OWN base value (each of the four rows in an `f64x4` carries a distinct
    /// base), so lane `i` of the packed result equals the scalar `stack(value_i)`
    /// bit-for-bit (only the cheap pack is vectorised; the closure body is the
    /// identical scalar code). With `N = 3` / `N = 5` it is `to_bits`-identical to
    /// [`unary3`](Lane::unary3) / [`unary5`](Lane::unary5).
    fn unary_with<const N: usize>(self, stack: impl Fn(f64) -> [f64; N]) -> [Self; N];
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
    fn lane(self, _: usize) -> f64 {
        self
    }
    #[inline]
    fn unary3(self, stack: impl Fn(f64) -> [f64; 3]) -> [Self; 3] {
        stack(self)
    }
    #[inline]
    fn unary5(self, stack: impl Fn(f64) -> [f64; 5]) -> [Self; 5] {
        stack(self)
    }
    #[inline]
    fn unary_with<const N: usize>(self, stack: impl Fn(f64) -> [f64; N]) -> [Self; N] {
        // One row: the packed result IS the scalar stack ([Self; N] = [f64; N]).
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
    #[inline]
    fn unary5(self, stack: impl Fn(f64) -> [f64; 5]) -> [Self; 5] {
        let a = self.to_array();
        let mut d = [[0.0_f64; 4]; 5];
        for i in 0..4 {
            let s = stack(a[i]);
            for (k, dk) in d.iter_mut().enumerate() {
                dk[i] = s[k];
            }
        }
        [
            wide::f64x4::new(d[0]),
            wide::f64x4::new(d[1]),
            wide::f64x4::new(d[2]),
            wide::f64x4::new(d[3]),
            wide::f64x4::new(d[4]),
        ]
    }
    #[inline]
    fn unary_with<const N: usize>(self, stack: impl Fn(f64) -> [f64; N]) -> [Self; N] {
        // Evaluate the scalar stack PER LANE at that lane's own base value, then
        // pack the N derivative columns lane-wise (the same shape `unary5` uses,
        // generalised to N). Lane `i` of column `k` is `stack(base_i)[k]`.
        let a = self.to_array();
        let mut cols = [[0.0_f64; 4]; N];
        for (i, &base) in a.iter().enumerate() {
            let s = stack(base);
            for (k, sk) in s.iter().enumerate() {
                cols[k][i] = *sk;
            }
        }
        std::array::from_fn(|k| wide::f64x4::new(cols[k]))
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
    ///
    /// The Hessian channel is symmetric under `i ↔ j` (see
    /// [`crate::jet_tower::Tower2::mul`] for why the invariant always holds), so
    /// we compute the upper triangle `j ≥ i` and mirror it — `K(K+1)/2` lane
    /// entry-chains instead of `K²`. Because each lane entry is already a full
    /// SIMD op (no cross-`j` lane packing to lose), halving the entry count is a
    /// direct throughput win (~18 % on `Order2Lane<f64x4, 9>`, the survival batch
    /// kernel, and ~2× on the `f64` oracle). The upper triangle uses the EXACT
    /// term order of `Tower2::mul`, so `Order2Lane<f64>` stays `to_bits`-identical
    /// to `Order2` (= `Tower2`) and `Order2Lane<f64x4>` lane `i` stays
    /// `to_bits`-identical to that; the mirror makes the batch Hessian exactly
    /// symmetric, matching the scalar `Tower2::mul` (which mirrors identically).
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
            for j in i..K {
                // a.v*b.h + a.g[i]*b.g[j] + a.g[j]*b.g[i] + a.h*b.v
                let hij =
                    a.v.mul(b.h[i][j])
                        .add(a.g[i].mul(b.g[j]))
                        .add(a.g[j].mul(b.g[i]))
                        .add(a.h[i][j].mul(b.v));
                out.h[i][j] = hij;
                out.h[j][i] = hij;
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

// ── OneSeedLane<L, K>: lane-batched one-seed directional (doc §A.2) ──────

/// Lane-batched [`OneSeed`]: the same one-seed directional scalar with its two
/// [`Order2`] parts re-typed to [`Order2Lane<L, K>`], so one `L = f64x4`
/// instance carries FOUR rows' contracted-third evaluations per vector pass.
///
/// Every operation (`add`/`sub`/`mul`/`neg`/`scale`/`compose_unary` and the
/// transcendentals) is a term-for-term structural re-type of the scalar
/// [`OneSeed`] ops onto the lane-implemented [`Order2Lane`] algebra. With
/// `L = f64`, `OneSeedLane<f64, K>` is `to_bits`-identical to [`OneSeed<K>`];
/// with `L = f64x4`, lane `i` is `to_bits`-identical to that (see `batch_tests`).
#[derive(Clone, Copy, Debug)]
pub struct OneSeedLane<L: Lane, const K: usize> {
    /// The `ε⁰` part (lane-batched value / gradient / Hessian of `ℓ`).
    pub base: Order2Lane<L, K>,
    /// The `ε¹` part. After a `seed_direction(u)` evaluation,
    /// `eps.h[a][b]` lane `i` is row `i`'s `Σ_c ℓ_{abc} u_c`.
    pub eps: Order2Lane<L, K>,
}

/// The 4-rows-per-pass batched one-seed scalar (`wide::f64x4` lanes).
pub type OneSeedBatch<const K: usize> = OneSeedLane<wide::f64x4, K>;

impl<L: Lane, const K: usize> OneSeedLane<L, K> {
    /// A constant: base = `constant(c)`, ε-part zero (mirrors [`OneSeed::constant`]).
    #[inline]
    pub fn constant(c: L) -> Self {
        OneSeedLane {
            base: Order2Lane::constant(c),
            eps: Order2Lane::constant(L::splat(0.0)),
        }
    }

    /// The seeded variable `p_axis` at (per-lane) value `value`, no ε-direction
    /// (mirrors [`OneSeed::variable`]).
    #[inline]
    pub fn variable(value: L, axis: usize) -> Self {
        OneSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: Order2Lane::constant(L::splat(0.0)),
        }
    }

    /// Seed primary `axis` at (per-lane) value `value` with ε-direction
    /// component `u_axis`: base = `variable(value, axis)`, eps = `constant(u_axis)`
    /// (mirrors [`OneSeed::seed_direction`]). With `L = f64x4`, `value` / `u_axis`
    /// pack the four rows' values / directions of primary `axis`.
    #[inline]
    pub fn seed_direction(value: L, axis: usize, u_axis: L) -> Self {
        OneSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: Order2Lane::constant(u_axis),
        }
    }

    /// The contracted-third channel after a `seed_direction(u)` evaluation:
    /// `out[a][b]` lane `i` is row `i`'s `Σ_c ℓ_{abc} u_c` (the ε-part Hessian).
    #[inline]
    #[must_use]
    pub fn contracted_third(&self) -> [[L; K]; K] {
        self.eps.h
    }

    /// Lane-wise `self + o` (mirrors [`OneSeed::add`]).
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        OneSeedLane {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }

    /// Lane-wise `self - o` (mirrors [`OneSeed::sub`]).
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        OneSeedLane {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }

    /// Lane-wise `self · o`, ε² = 0 truncation (mirrors [`OneSeed::mul`]).
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        OneSeedLane {
            base: self.base.mul(&o.base),
            eps: self.base.mul(&o.eps).add(&self.eps.mul(&o.base)),
        }
    }

    /// Negate every part (mirrors [`OneSeed::neg`]).
    #[inline]
    pub fn neg(&self) -> Self {
        OneSeedLane {
            base: self.base.neg(),
            eps: self.eps.neg(),
        }
    }

    /// Multiply every part by the plain scalar `s` (mirrors [`OneSeed::scale`]).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        OneSeedLane {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }

    /// Exact order-≤2-per-part Faà di Bruno composition `f ∘ self`, given the
    /// per-lane outer-derivative stack `d = [f, f′, f″, f‴, f⁗]`. Term-for-term
    /// identical to [`OneSeed::compose_unary`]: the base reads `d[0..=2]` and the
    /// ε-coefficient is `f′(base)` (reads `d[1..=3]`) times `eps`.
    #[inline]
    pub fn compose_unary(&self, d: [L; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3]]);
        let eps = fprime.mul(&self.eps);
        OneSeedLane { base, eps }
    }

    /// `e^self`, per-lane stack `[e, e, e, e, e]` (matches [`JetScalar::exp`]).
    #[inline]
    pub fn exp(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let e = u.exp();
            [e, e, e, e, e]
        });
        self.compose_unary(d)
    }

    /// `ln(self)`; caller guarantees positivity (matches [`JetScalar::ln`]).
    #[inline]
    pub fn ln(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            [u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r]
        });
        self.compose_unary(d)
    }

    /// `√self`; caller guarantees positivity (matches [`JetScalar::sqrt`]).
    #[inline]
    pub fn sqrt(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let s = u.sqrt();
            [
                s,
                0.5 / s,
                -0.25 / (u * s),
                0.375 / (u * u * s),
                -0.9375 / (u * u * u * s),
            ]
        });
        self.compose_unary(d)
    }

    /// `1/self` (matches [`JetScalar::recip`]).
    #[inline]
    pub fn recip(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            let r2 = r * r;
            [r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r]
        });
        self.compose_unary(d)
    }

    /// `self^a` for real `a`; caller guarantees a positive base (matches
    /// [`JetScalar::powf`]).
    #[inline]
    pub fn powf(&self, a: f64) -> Self {
        let d = self.base.v.unary5(|u| {
            [
                u.powf(a),
                a * u.powf(a - 1.0),
                a * (a - 1.0) * u.powf(a - 2.0),
                a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0),
                a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0),
            ]
        });
        self.compose_unary(d)
    }

    /// `ln Γ(self)`; caller guarantees positivity (matches [`JetScalar::ln_gamma`],
    /// same hand-certified stack).
    #[inline]
    pub fn ln_gamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::ln_gamma_derivative_stack);
        self.compose_unary(d)
    }

    /// `ψ(self)` digamma; caller guarantees positivity (matches
    /// [`JetScalar::digamma`], same hand-certified stack).
    #[inline]
    pub fn digamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::digamma_derivative_stack);
        self.compose_unary(d)
    }
}

impl<const K: usize> OneSeedBatch<K> {
    /// Extract lane `i`'s parts as a production [`OneSeed<K>`]. Lane `i` is
    /// `to_bits`-identical to evaluating the same program at [`OneSeed<K>`] on
    /// row `i` (see `batch_tests`).
    #[inline]
    #[must_use]
    pub fn lane(&self, i: usize) -> OneSeed<K> {
        OneSeed {
            base: self.base.lane(i),
            eps: self.eps.lane(i),
        }
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

// ── TwoSeedLane<L, K>: lane-batched two-seed, contracted fourth (doc §A.3) ─

/// Lane-batched [`TwoSeed`]: the same two-seed scalar with its four [`Order2`]
/// parts re-typed to [`Order2Lane<L, K>`], so one `L = f64x4` instance carries
/// FOUR rows' contracted-fourth evaluations per vector pass.
///
/// Every operation is a term-for-term structural re-type of the scalar
/// [`TwoSeed`] ops onto the lane-implemented [`Order2Lane`] algebra. With
/// `L = f64`, `TwoSeedLane<f64, K>` is `to_bits`-identical to [`TwoSeed<K>`];
/// with `L = f64x4`, lane `i` is `to_bits`-identical to that (see `batch_tests`).
#[derive(Clone, Copy, Debug)]
pub struct TwoSeedLane<L: Lane, const K: usize> {
    /// The `ε⁰δ⁰` part.
    pub base: Order2Lane<L, K>,
    /// The `ε¹δ⁰` part.
    pub eps: Order2Lane<L, K>,
    /// The `ε⁰δ¹` part.
    pub del: Order2Lane<L, K>,
    /// The `ε¹δ¹` part. After a `seed(u, v)` evaluation, `eps_del.h[a][b]`
    /// lane `i` is row `i`'s `Σ_{cd} ℓ_{abcd} u_c v_d`.
    pub eps_del: Order2Lane<L, K>,
}

/// The 4-rows-per-pass batched two-seed scalar (`wide::f64x4` lanes).
pub type TwoSeedBatch<const K: usize> = TwoSeedLane<wide::f64x4, K>;

impl<L: Lane, const K: usize> TwoSeedLane<L, K> {
    /// A constant: base = `constant(c)`, all seed parts zero (mirrors
    /// [`TwoSeed::constant`]).
    #[inline]
    pub fn constant(c: L) -> Self {
        let z = Order2Lane::constant(L::splat(0.0));
        TwoSeedLane {
            base: Order2Lane::constant(c),
            eps: z,
            del: z,
            eps_del: z,
        }
    }

    /// The seeded variable `p_axis` at (per-lane) value `value`, no ε/δ direction
    /// (mirrors [`TwoSeed::variable`]).
    #[inline]
    pub fn variable(value: L, axis: usize) -> Self {
        let z = Order2Lane::constant(L::splat(0.0));
        TwoSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: z,
            del: z,
            eps_del: z,
        }
    }

    /// Seed primary `axis` at (per-lane) value `value` with ε-direction `u_axis`
    /// and δ-direction `v_axis` (mirrors [`TwoSeed::seed`]). With `L = f64x4`,
    /// each argument packs the four rows' values for primary `axis`.
    #[inline]
    pub fn seed(value: L, axis: usize, u_axis: L, v_axis: L) -> Self {
        TwoSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: Order2Lane::constant(u_axis),
            del: Order2Lane::constant(v_axis),
            eps_del: Order2Lane::constant(L::splat(0.0)),
        }
    }

    /// The contracted-fourth channel after a `seed(u, v)` evaluation:
    /// `out[a][b]` lane `i` is row `i`'s `Σ_{cd} ℓ_{abcd} u_c v_d`
    /// (the εδ-part Hessian).
    #[inline]
    #[must_use]
    pub fn contracted_fourth(&self) -> [[L; K]; K] {
        self.eps_del.h
    }

    /// Lane-wise `self + o` (mirrors [`TwoSeed::add`]).
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        TwoSeedLane {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }

    /// Lane-wise `self - o` (mirrors [`TwoSeed::sub`]).
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        TwoSeedLane {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }

    /// Lane-wise `self · o`, ε² = δ² = 0 truncation (mirrors [`TwoSeed::mul`]).
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let base = a.base.mul(&b.base);
        let eps = a.base.mul(&b.eps).add(&a.eps.mul(&b.base));
        let del = a.base.mul(&b.del).add(&a.del.mul(&b.base));
        let eps_del = a
            .base
            .mul(&b.eps_del)
            .add(&a.eps.mul(&b.del))
            .add(&a.del.mul(&b.eps))
            .add(&a.eps_del.mul(&b.base));
        TwoSeedLane {
            base,
            eps,
            del,
            eps_del,
        }
    }

    /// Negate every part (mirrors [`TwoSeed::neg`]).
    #[inline]
    pub fn neg(&self) -> Self {
        TwoSeedLane {
            base: self.base.neg(),
            eps: self.eps.neg(),
            del: self.del.neg(),
            eps_del: self.eps_del.neg(),
        }
    }

    /// Multiply every part by the plain scalar `s` (mirrors [`TwoSeed::scale`]).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        TwoSeedLane {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }

    /// Exact composition `f ∘ self`, given the per-lane outer-derivative stack
    /// `d = [f, f′, f″, f‴, f⁗]`. Term-for-term identical to
    /// [`TwoSeed::compose_unary`]: base reads `d[0..=2]`, `f′(base)` reads
    /// `d[1..=3]`, `f″(base)` reads `d[2..=4]`, and the cross part carries
    /// `f″·eps·del + f′·eps_del`.
    #[inline]
    pub fn compose_unary(&self, d: [L; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3]]);
        let fsecond = self.base.compose_unary([d[2], d[3], d[4]]);
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        TwoSeedLane {
            base,
            eps,
            del,
            eps_del,
        }
    }

    /// `e^self`, per-lane stack `[e; 5]` (matches [`JetScalar::exp`]).
    #[inline]
    pub fn exp(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let e = u.exp();
            [e, e, e, e, e]
        });
        self.compose_unary(d)
    }

    /// `ln(self)`; caller guarantees positivity (matches [`JetScalar::ln`]).
    #[inline]
    pub fn ln(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            [u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r]
        });
        self.compose_unary(d)
    }

    /// `√self`; caller guarantees positivity (matches [`JetScalar::sqrt`]).
    #[inline]
    pub fn sqrt(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let s = u.sqrt();
            [
                s,
                0.5 / s,
                -0.25 / (u * s),
                0.375 / (u * u * s),
                -0.9375 / (u * u * u * s),
            ]
        });
        self.compose_unary(d)
    }

    /// `1/self` (matches [`JetScalar::recip`]).
    #[inline]
    pub fn recip(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            let r2 = r * r;
            [r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r]
        });
        self.compose_unary(d)
    }

    /// `self^a` for real `a`; caller guarantees a positive base (matches
    /// [`JetScalar::powf`]).
    #[inline]
    pub fn powf(&self, a: f64) -> Self {
        let d = self.base.v.unary5(|u| {
            [
                u.powf(a),
                a * u.powf(a - 1.0),
                a * (a - 1.0) * u.powf(a - 2.0),
                a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0),
                a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0),
            ]
        });
        self.compose_unary(d)
    }

    /// `ln Γ(self)`; caller guarantees positivity (matches [`JetScalar::ln_gamma`]).
    #[inline]
    pub fn ln_gamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::ln_gamma_derivative_stack);
        self.compose_unary(d)
    }

    /// `ψ(self)` digamma; caller guarantees positivity (matches
    /// [`JetScalar::digamma`]).
    #[inline]
    pub fn digamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::digamma_derivative_stack);
        self.compose_unary(d)
    }
}

impl<const K: usize> TwoSeedBatch<K> {
    /// Extract lane `i`'s parts as a production [`TwoSeed<K>`]. Lane `i` is
    /// `to_bits`-identical to evaluating the same program at [`TwoSeed<K>`] on
    /// row `i` (see `batch_tests`).
    #[inline]
    #[must_use]
    pub fn lane(&self, i: usize) -> TwoSeed<K> {
        TwoSeed {
            base: self.base.lane(i),
            eps: self.eps.lane(i),
            del: self.del.lane(i),
            eps_del: self.eps_del.lane(i),
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

    /// The `compose_unary_with` seam on a scalar jet is `to_bits`-identical to
    /// the explicit `compose_unary(stack_fn(value))` — the contract the batch
    /// arm (`Tower{3,4}Lane::compose_unary_with`) lane-matches. Exercised on
    /// [`Order2`] across `K ∈ {2,3,4,9}`, ≥ 4000 random seeded inputs.
    #[test]
    fn compose_unary_with_scalar_seam_bit_identical() {
        fn rand_unit(state: &mut u64) -> f64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            2.0 * ((x >> 11) as f64 / ((1u64 << 53) as f64)) - 1.0
        }
        // A base-value-dependent finite stack standing in for a family stack.
        fn stack(u: f64) -> [f64; 5] {
            [
                u.sin(),
                u.cos(),
                (2.0 * u).sin(),
                (0.5 * u).cos(),
                u * u - 0.3,
            ]
        }
        fn run<const K: usize>(state: &mut u64, n: usize) -> usize {
            for _ in 0..n {
                // A non-trivial Order2<K> jet: a seeded variable pushed through a
                // couple of algebra ops so g/h are dense, then exercise the seam.
                let base = rand_unit(state);
                let mut s = Order2::<K>::variable(base, 0);
                for a in 1..K {
                    s = JetScalar::mul(&s, &Order2::<K>::variable(rand_unit(state), a));
                }
                let with = s.compose_unary_with(stack);
                let explicit = s.compose_unary(stack(s.value()));
                assert_eq!(with.value().to_bits(), explicit.value().to_bits(), "value");
                for a in 0..K {
                    assert_eq!(with.g()[a].to_bits(), explicit.g()[a].to_bits(), "g[{a}]");
                    for b in 0..K {
                        assert_eq!(
                            with.h()[a][b].to_bits(),
                            explicit.h()[a][b].to_bits(),
                            "h[{a}][{b}]"
                        );
                    }
                }
            }
            n
        }
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let total = run::<2>(&mut st, 1100)
            + run::<3>(&mut st, 1100)
            + run::<4>(&mut st, 1100)
            + run::<9>(&mut st, 1100);
        assert_eq!(total, 4400);
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

    /// Arena-backed runtime directional jets must reproduce the fixed packed
    /// algebras channel-for-channel. This isolates the runtime scalar algebra
    /// from every family row program before the SLS/SAE integration oracles.
    #[test]
    fn runtime_directional_jets_match_fixed_packed_algebra_932() {
        fn expression<'arena, S: RuntimeJetScalar<'arena>>(vars: &[S]) -> S {
            let bilinear = vars[0].mul(&vars[1]);
            let curved = vars[2].scale(0.7).add(&vars[3].mul(&vars[3]).scale(-0.2));
            bilinear
                .add(&curved)
                .exp()
                .mul(&vars[4].compose_unary([0.4, -0.3, 0.2, -0.1, 0.05]))
        }

        const K: usize = 5;
        let values = [0.2, -0.7, 0.4, 1.1, -0.3];
        let direction_u = [0.5, -0.2, 0.7, -0.4, 0.1];
        let direction_v = [-0.3, 0.8, 0.2, 0.6, -0.5];
        let close = |actual: f64, expected: f64| {
            let tolerance = 1.0e-13 * (1.0 + actual.abs().max(expected.abs()));
            assert!((actual - expected).abs() <= tolerance);
        };

        let fixed_one: Vec<FixedRuntimeJet<OneSeed<K>, K>> = (0..K)
            .map(|axis| FixedRuntimeJet {
                inner: OneSeed::seed_direction(values[axis], axis, direction_u[axis]),
            })
            .collect();
        let arena_one = DynamicJetArena::new();
        let dynamic_one: Vec<DynamicOneSeed<'_>> = (0..K)
            .map(|axis| {
                DynamicOneSeed::seed_direction(values[axis], axis, direction_u[axis], K, &arena_one)
            })
            .collect();
        let fixed_third = expression(&fixed_one).into_inner().contracted_third();
        let dynamic_third = expression(&dynamic_one);
        for a in 0..K {
            for b in 0..K {
                close(
                    dynamic_third.contracted_third()[a * K + b],
                    fixed_third[a][b],
                );
            }
        }

        let fixed_two: Vec<FixedRuntimeJet<TwoSeed<K>, K>> = (0..K)
            .map(|axis| FixedRuntimeJet {
                inner: TwoSeed::seed(values[axis], axis, direction_u[axis], direction_v[axis]),
            })
            .collect();
        let arena_two = DynamicJetArena::new();
        let dynamic_two: Vec<DynamicTwoSeed<'_>> = (0..K)
            .map(|axis| {
                DynamicTwoSeed::seed(
                    values[axis],
                    axis,
                    direction_u[axis],
                    direction_v[axis],
                    K,
                    &arena_two,
                )
            })
            .collect();
        let fixed_fourth = expression(&fixed_two).into_inner().contracted_fourth();
        let dynamic_fourth = expression(&dynamic_two);
        for a in 0..K {
            for b in 0..K {
                close(
                    dynamic_fourth.contracted_fourth()[a * K + b],
                    fixed_fourth[a][b],
                );
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

    use super::{
        JetScalar, Lane, OneSeed, OneSeedBatch, OneSeedLane, Order2, Order2Batch, Order2Lane,
        TwoSeed, TwoSeedBatch, TwoSeedLane,
    };

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

    /// Returns the number of (batch, row) pairs whose every channel was
    /// verified bit-identical, so the caller can assert the expected total ran.
    fn check_k<const K: usize>(state: &mut u64, batches: usize) -> usize {
        let mut verified_rows = 0usize;
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
                let packed = wide::f64x4::new([rows[0][a], rows[1][a], rows[2][a], rows[3][a]]);
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
                verified_rows += 1;
            }
        }
        verified_rows
    }

    /// ≥2000 random 4-row batches per K, across K ∈ {2,3,4,9}: every channel of
    /// every lane is `to_bits`-identical to the production scalar per row.
    #[test]
    fn batch_lanes_bit_identical_to_scalar_per_row() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut verified = 0usize;
        verified += check_k::<2>(&mut state, 2000);
        verified += check_k::<3>(&mut state, 2000);
        verified += check_k::<4>(&mut state, 2000);
        verified += check_k::<9>(&mut state, 2000);
        // 4 K-values × 2000 batches × 4 packed rows each, all bit-identical.
        assert_eq!(verified, 4 * 2000 * 4, "every batch row must be verified");
    }

    // ── One-/two-seed lane oracles ──────────────────────────────────────────
    //
    // The same dense `row_expr` witness program runs over the SEEDED directional
    // scalars: the scalar `OneSeed`/`TwoSeed` per row, the `f64`-lane re-type
    // (`*SeedLane<f64>`), and the 4-rows-per-pass batch (`*SeedBatch`). The
    // headline claim is that the contracted-third / contracted-fourth channel of
    // every lane is `to_bits`-identical to the production scalar's per row.

    impl<const K: usize> RowAlg<K> for OneSeed<K> {
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

    impl<L: Lane, const K: usize> RowAlg<K> for OneSeedLane<L, K> {
        fn constant(c: f64) -> Self {
            OneSeedLane::constant(L::splat(c))
        }
        fn add(&self, o: &Self) -> Self {
            OneSeedLane::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            OneSeedLane::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            OneSeedLane::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            OneSeedLane::scale(self, s)
        }
        fn exp(&self) -> Self {
            OneSeedLane::exp(self)
        }
        fn sqrt(&self) -> Self {
            OneSeedLane::sqrt(self)
        }
        fn recip(&self) -> Self {
            OneSeedLane::recip(self)
        }
    }

    impl<const K: usize> RowAlg<K> for TwoSeed<K> {
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

    impl<L: Lane, const K: usize> RowAlg<K> for TwoSeedLane<L, K> {
        fn constant(c: f64) -> Self {
            TwoSeedLane::constant(L::splat(c))
        }
        fn add(&self, o: &Self) -> Self {
            TwoSeedLane::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            TwoSeedLane::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            TwoSeedLane::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            TwoSeedLane::scale(self, s)
        }
        fn exp(&self) -> Self {
            TwoSeedLane::exp(self)
        }
        fn sqrt(&self) -> Self {
            TwoSeedLane::sqrt(self)
        }
        fn recip(&self) -> Self {
            TwoSeedLane::recip(self)
        }
    }

    fn check_oneseed<const K: usize>(state: &mut u64, batches: usize) -> usize {
        let mut rows_checked = 0;
        for _ in 0..batches {
            let rows: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));
            // Per-row ε-direction.
            let u: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));

            // Production ground truth (scalar OneSeed per row).
            let prod: [OneSeed<K>; 4] = std::array::from_fn(|r| {
                let p: [OneSeed<K>; K] =
                    std::array::from_fn(|a| OneSeed::seed_direction(rows[r][a], a, u[r][a]));
                row_expr(&p)
            });

            // f64-lane re-type per row.
            let scal: [OneSeedLane<f64, K>; 4] = std::array::from_fn(|r| {
                let p: [OneSeedLane<f64, K>; K] =
                    std::array::from_fn(|a| OneSeedLane::seed_direction(rows[r][a], a, u[r][a]));
                row_expr(&p)
            });

            // 4-rows-per-pass batch.
            let pbatch: [OneSeedBatch<K>; K] = std::array::from_fn(|a| {
                let val = wide::f64x4::new([rows[0][a], rows[1][a], rows[2][a], rows[3][a]]);
                let uu = wide::f64x4::new([u[0][a], u[1][a], u[2][a], u[3][a]]);
                OneSeedBatch::seed_direction(val, a, uu)
            });
            let batch = row_expr(&pbatch);

            for r in 0..4 {
                let want = prod[r].contracted_third();
                let got_scal = scal[r].contracted_third();
                let got_batch = batch.lane(r).contracted_third();
                // Value channel too (sanity that the base program agrees).
                assert_eq!(
                    scal[r].base.v.to_bits(),
                    prod[r].base.value().to_bits(),
                    "OneSeed K={K} scalar value"
                );
                assert_eq!(
                    batch.lane(r).base.value().to_bits(),
                    prod[r].base.value().to_bits(),
                    "OneSeed K={K} batch lane {r} value"
                );
                for a in 0..K {
                    for b in 0..K {
                        assert_eq!(
                            got_scal[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "OneSeed K={K} scalar third[{a}][{b}]"
                        );
                        assert_eq!(
                            got_batch[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "OneSeed K={K} batch lane {r} third[{a}][{b}]"
                        );
                    }
                }
                rows_checked += 1;
            }
        }
        rows_checked
    }

    fn check_twoseed<const K: usize>(state: &mut u64, batches: usize) -> usize {
        let mut rows_checked = 0;
        for _ in 0..batches {
            let rows: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));
            let u: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));
            let v: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));

            let prod: [TwoSeed<K>; 4] = std::array::from_fn(|r| {
                let p: [TwoSeed<K>; K] =
                    std::array::from_fn(|a| TwoSeed::seed(rows[r][a], a, u[r][a], v[r][a]));
                row_expr(&p)
            });

            let scal: [TwoSeedLane<f64, K>; 4] = std::array::from_fn(|r| {
                let p: [TwoSeedLane<f64, K>; K] =
                    std::array::from_fn(|a| TwoSeedLane::seed(rows[r][a], a, u[r][a], v[r][a]));
                row_expr(&p)
            });

            let pbatch: [TwoSeedBatch<K>; K] = std::array::from_fn(|a| {
                let val = wide::f64x4::new([rows[0][a], rows[1][a], rows[2][a], rows[3][a]]);
                let uu = wide::f64x4::new([u[0][a], u[1][a], u[2][a], u[3][a]]);
                let vv = wide::f64x4::new([v[0][a], v[1][a], v[2][a], v[3][a]]);
                TwoSeedBatch::seed(val, a, uu, vv)
            });
            let batch = row_expr(&pbatch);

            for r in 0..4 {
                let want = prod[r].contracted_fourth();
                let got_scal = scal[r].contracted_fourth();
                let got_batch = batch.lane(r).contracted_fourth();
                assert_eq!(
                    scal[r].base.v.to_bits(),
                    prod[r].base.value().to_bits(),
                    "TwoSeed K={K} scalar value"
                );
                assert_eq!(
                    batch.lane(r).base.value().to_bits(),
                    prod[r].base.value().to_bits(),
                    "TwoSeed K={K} batch lane {r} value"
                );
                for a in 0..K {
                    for b in 0..K {
                        assert_eq!(
                            got_scal[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "TwoSeed K={K} scalar fourth[{a}][{b}]"
                        );
                        assert_eq!(
                            got_batch[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "TwoSeed K={K} batch lane {r} fourth[{a}][{b}]"
                        );
                    }
                }
                rows_checked += 1;
            }
        }
        rows_checked
    }

    /// ≥2000 random 4-row batches per K, across K ∈ {2,3,4,9}: the
    /// contracted-third channel of every `OneSeedLane` lane is `to_bits`-identical
    /// to the production [`OneSeed`] per row.
    #[test]
    fn oneseed_lanes_contracted_third_bit_identical() {
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let batches = 2000;
        let rows_checked = check_oneseed::<2>(&mut state, batches)
            + check_oneseed::<3>(&mut state, batches)
            + check_oneseed::<4>(&mut state, batches)
            + check_oneseed::<9>(&mut state, batches);
        // 4 widths × `batches` batches × 4 rows each: a silently empty inner
        // loop would leave this at zero instead of passing as a no-op.
        assert_eq!(rows_checked, 4 * batches * 4);
    }

    /// ≥2000 random 4-row batches per K, across K ∈ {2,3,4,9}: the
    /// contracted-fourth channel of every `TwoSeedLane` lane is `to_bits`-identical
    /// to the production [`TwoSeed`] per row.
    #[test]
    fn twoseed_lanes_contracted_fourth_bit_identical() {
        let mut state = 0x0FED_CBA9_8765_4321_u64;
        let batches = 2000;
        let rows_checked = check_twoseed::<2>(&mut state, batches)
            + check_twoseed::<3>(&mut state, batches)
            + check_twoseed::<4>(&mut state, batches)
            + check_twoseed::<9>(&mut state, batches);
        // 4 widths × `batches` batches × 4 rows each: a silently empty inner
        // loop would leave this at zero instead of passing as a no-op.
        assert_eq!(rows_checked, 4 * batches * 4);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{JetScalar, Order1, Order2, filtered_implicit_solve_scalar};

    // ── Order2 direct property tests ─────────────────────────────────────────

    /// `Order2::constant(c)` carries value `c` and zero everywhere else.
    #[test]
    fn order2_constant_has_zero_derivatives() {
        let s = Order2::<3>::constant(7.5);
        assert_eq!(s.value(), 7.5);
        for a in 0..3 {
            assert_eq!(s.g()[a], 0.0, "grad[{a}] should be zero");
            for b in 0..3 {
                assert_eq!(s.h()[a][b], 0.0, "hess[{a}][{b}] should be zero");
            }
        }
    }

    /// `Order2::variable(x, axis)` has unit gradient in slot `axis` and zero Hessian.
    #[test]
    fn order2_variable_has_unit_gradient_in_seeded_slot() {
        let x = -2.5_f64;
        let s = Order2::<4>::variable(x, 2);
        assert_eq!(s.value(), x);
        for a in 0..4 {
            let expected_g = if a == 2 { 1.0 } else { 0.0 };
            assert_eq!(s.g()[a], expected_g, "grad[{a}]");
            for b in 0..4 {
                assert_eq!(s.h()[a][b], 0.0, "hess[{a}][{b}] should be zero");
            }
        }
    }

    /// `Order2::add` sums gradient channels; `sub` is the inverse on gradients.
    /// Uses integer-valued primaries so the value roundtrip is also exact.
    #[test]
    fn order2_add_sub_roundtrip() {
        let p = Order2::<2>::variable(3.0, 0);
        let q = Order2::<2>::variable(2.0, 1);
        let pq = JetScalar::add(&p, &q);
        // value = 3 + 2 = 5
        assert_eq!(pq.value(), 5.0, "add value");
        let back = JetScalar::sub(&pq, &q);
        // (p + q) - q gradient should equal p's gradient exactly
        for a in 0..2 {
            assert_eq!(back.g()[a], p.g()[a], "grad[{a}] roundtrip");
        }
    }

    /// `Order2::mul` of two variables satisfies the Leibniz product rule:
    ///   ∂(p·q)/∂p = q,  ∂(p·q)/∂q = p,  ∂²(p·q)/∂p∂q = 1.
    #[test]
    fn order2_mul_satisfies_leibniz_rule() {
        let pv = 3.0_f64;
        let qv = -2.0_f64;
        let p = Order2::<2>::variable(pv, 0);
        let q = Order2::<2>::variable(qv, 1);
        let pq = JetScalar::mul(&p, &q);
        assert_eq!(pq.value(), pv * qv, "value = p·q");
        assert_eq!(pq.g()[0], qv, "∂(p·q)/∂p = q");
        assert_eq!(pq.g()[1], pv, "∂(p·q)/∂q = p");
        assert_eq!(pq.h()[0][1], 1.0, "∂²(p·q)/∂p∂q = 1");
        assert_eq!(pq.h()[1][0], 1.0, "∂²(p·q)/∂q∂p = 1 (symmetric)");
        assert_eq!(pq.h()[0][0], 0.0, "∂²(p·q)/∂p² = 0");
        assert_eq!(pq.h()[1][1], 0.0, "∂²(p·q)/∂q² = 0");
    }

    /// `Order2::scale(s)` multiplies every channel by `s`.
    #[test]
    fn order2_scale_multiplies_all_channels() {
        let p = Order2::<2>::variable(4.0, 0);
        let s = 2.5_f64;
        let ps = JetScalar::scale(&p, s);
        assert_eq!(ps.value(), 4.0 * s);
        assert_eq!(ps.g()[0], 1.0 * s);
        assert_eq!(ps.g()[1], 0.0);
    }

    /// `Order2::exp` at a constant has value `e^c`, gradient `e^c * g`, Hessian `e^c * (g⊗g + H)`.
    /// At a seeded variable `p₀`, the first derivative is `e^{p₀}` and second is `e^{p₀}`.
    #[test]
    fn order2_exp_derivative_stack_correct() {
        let p0 = 1.0_f64;
        let p = Order2::<1>::variable(p0, 0);
        let ep = JetScalar::exp(&p);
        let e = p0.exp();
        assert!((ep.value() - e).abs() < 1e-15, "exp value");
        assert!((ep.g()[0] - e).abs() < 1e-15, "d/dp exp(p) = exp(p)");
        assert!((ep.h()[0][0] - e).abs() < 1e-15, "d²/dp² exp(p) = exp(p)");
    }

    /// `Order2::ln` at a seeded variable: d/dp ln(p) = 1/p, d²/dp² ln(p) = -1/p².
    #[test]
    fn order2_ln_derivative_stack_correct() {
        let p0 = 2.0_f64;
        let p = Order2::<1>::variable(p0, 0);
        let lnp = JetScalar::ln(&p);
        assert!((lnp.value() - p0.ln()).abs() < 1e-15, "ln value");
        assert!((lnp.g()[0] - 1.0 / p0).abs() < 1e-15, "d/dp ln(p) = 1/p");
        assert!(
            (lnp.h()[0][0] - (-1.0 / (p0 * p0))).abs() < 1e-15,
            "d²/dp² ln(p) = -1/p²"
        );
    }

    /// `exp` and `ln` are mutual inverses: `ln(exp(p)).value() == p` at the scalar.
    #[test]
    fn order2_exp_ln_roundtrip_at_value() {
        let p0 = 0.8_f64;
        let p = Order2::<1>::variable(p0, 0);
        let roundtrip = JetScalar::ln(&JetScalar::exp(&p));
        assert!((roundtrip.value() - p0).abs() < 1e-14, "ln(exp(p)) ≈ p");
    }

    // ── Order1 tests ─────────────────────────────────────────────────────────

    /// `Order1::constant` carries the correct value with all-zero gradient.
    #[test]
    fn order1_constant_has_zero_gradient() {
        let s = Order1::<3>::constant(-5.0);
        assert_eq!(s.value(), -5.0);
        for a in 0..3 {
            assert_eq!(s.g()[a], 0.0, "g[{a}] should be zero");
        }
    }

    /// `Order1::variable(x, axis)` has unit gradient only in `axis`.
    #[test]
    fn order1_variable_has_unit_gradient_in_seeded_slot() {
        let s = Order1::<3>::variable(2.0, 1);
        assert_eq!(s.value(), 2.0);
        assert_eq!(s.g()[0], 0.0);
        assert_eq!(s.g()[1], 1.0);
        assert_eq!(s.g()[2], 0.0);
    }

    /// `Order1::mul` satisfies the product rule (value and gradient, no Hessian).
    #[test]
    fn order1_mul_satisfies_product_rule() {
        let pv = 3.0_f64;
        let qv = -2.0_f64;
        let p = Order1::<2>::variable(pv, 0);
        let q = Order1::<2>::variable(qv, 1);
        let pq = JetScalar::mul(&p, &q);
        assert_eq!(pq.value(), pv * qv);
        assert_eq!(pq.g()[0], qv, "∂(p·q)/∂p = q");
        assert_eq!(pq.g()[1], pv, "∂(p·q)/∂q = p");
    }

    /// `Order1::exp` carries the correct value and gradient `e^{p₀}`.
    #[test]
    fn order1_exp_has_correct_value_and_gradient() {
        let p0 = 0.5_f64;
        let p = Order1::<2>::variable(p0, 0);
        let ep = JetScalar::exp(&p);
        let e = p0.exp();
        assert!((ep.value() - e).abs() < 1e-15, "exp value");
        assert!((ep.g()[0] - e).abs() < 1e-15, "d/dp exp(p)");
        assert_eq!(ep.g()[1], 0.0, "irrelevant gradient slot is zero");
    }

    /// `Order1` and `Order2` agree on value and gradient for the same expression.
    #[test]
    fn order1_and_order2_agree_on_value_and_gradient() {
        let p0 = 1.3_f64;
        let q0 = -0.7_f64;
        // evaluate (p * q + p).exp() at (p0, q0)
        let p1 = Order1::<2>::variable(p0, 0);
        let q1 = Order1::<2>::variable(q0, 1);
        let expr1 = JetScalar::exp(&JetScalar::add(&JetScalar::mul(&p1, &q1), &p1));

        let p2 = Order2::<2>::variable(p0, 0);
        let q2 = Order2::<2>::variable(q0, 1);
        let expr2 = JetScalar::exp(&JetScalar::add(&JetScalar::mul(&p2, &q2), &p2));

        assert!(
            (expr1.value() - expr2.value()).abs() < 1e-14,
            "value mismatch"
        );
        for a in 0..2 {
            assert!(
                (expr1.g()[a] - expr2.g()[a]).abs() < 1e-14,
                "gradient[{a}] mismatch"
            );
        }
    }

    // ── filtered_implicit_solve_scalar ────────────────────────────────────────

    /// Lift the trivial linear constraint F(a, θ) = a - θ = 0 through `Order2<1>`.
    /// The exact lifted jet is a(θ) = θ, so value=θ₀, gradient=1.
    #[test]
    fn filtered_implicit_solve_linear_constraint_gives_exact_jet() {
        let theta0 = 3.0_f64;
        let theta = Order2::<1>::variable(theta0, 0);
        // a0 = theta0, F_a = 1, inv_fa = 1; 2 iters suffice for Order2.
        let a = filtered_implicit_solve_scalar::<1, Order2<1>>(theta0, 1.0, 2, |a_jet| {
            JetScalar::sub(a_jet, &theta)
        });
        assert!((a.value() - theta0).abs() < 1e-14, "value = theta0");
        // da/dtheta = 1 (identity)
        assert!((a.g()[0] - 1.0).abs() < 1e-14, "gradient = 1");
        // d²a/dtheta² = 0 (linear)
        assert!(a.h()[0][0].abs() < 1e-14, "hessian = 0");
    }

    /// `filtered_implicit_solve_scalar` on a quadratic constraint F(a,θ)=a²-θ=0
    /// with primal root a₀=√θ₀, giving da/dθ = 1/(2√θ₀), d²a/dθ² = -1/(4θ₀^{3/2}).
    #[test]
    fn filtered_implicit_solve_quadratic_constraint_matches_analytic_derivatives() {
        let theta0 = 4.0_f64;
        let a0 = theta0.sqrt();
        let inv_fa = 1.0 / (2.0 * a0);
        let theta = Order2::<1>::variable(theta0, 0);
        // F(a,theta) = a*a - theta
        let a = filtered_implicit_solve_scalar::<1, Order2<1>>(a0, inv_fa, 2, |a_jet| {
            let aa = JetScalar::mul(a_jet, a_jet);
            JetScalar::sub(&aa, &theta)
        });
        let tol = 1e-12;
        assert!((a.value() - a0).abs() < tol, "value = sqrt(theta0)");
        let expected_g = 0.5 / a0;
        assert!(
            (a.g()[0] - expected_g).abs() < tol,
            "da/dtheta = 1/(2*sqrt)"
        );
        let expected_h = -0.25 / (theta0 * a0);
        assert!(
            (a.h()[0][0] - expected_h).abs() < tol,
            "d2a/dtheta2 = -1/(4*theta^1.5)"
        );
    }
}
