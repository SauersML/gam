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
