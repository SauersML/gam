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

use crate::jet_algebra;

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

    /// Exact truncated Leibniz product `D_S(ab) = Σ_{T ⊆ S} D_T(a) · D_{S∖T}(b)`.
    ///
    /// # Codegen
    ///
    /// Each output entry's `2^m` subset sum is written as a compact straight-line
    /// expression instead of the shared [`jet_algebra::leibniz_product`] subset
    /// walker (which, per entry, builds `SlotBuf`s and `match`-dispatches the
    /// `deriv` closure across all `2^m` subsets). The loop nest over `(i,j,k,l)`
    /// is unchanged — only the inner per-entry sum is unrolled — so this does NOT
    /// unroll over `K` and does NOT bloat code: on a `Tower4<9>` mul-and-read
    /// consumer the new form is faster AND smaller (asm: 34 outlined walker `bl`
    /// calls → 0, 21.1 KiB → 14.3 KiB, +100 NEON `.2d` ops).
    ///
    /// BIT-IDENTICAL to the walker: each entry's terms are in the walker's exact
    /// subset-enumeration order (subset bit `b` ↔ position `b`, `sub = 0..2^m`),
    /// and the per-entry `acc` accumulator mirrors the walker's `total = 0.0`
    /// start so a signed-zero leading product collapses to `+0.0` identically —
    /// which matters because real jets carry exact-`0.0` channels
    /// (`constant`/`variable` towers). Proven `to_bits`-identical on
    /// `v`/`g`/`h`/`t3`/`t4` across `K ∈ {2,3,4,9}`, 5000 inputs each with ~30 %
    /// exact-`0.0` channels and signed values (a no-leading-`0.0` form fails this
    /// stress — the accumulator start is load-bearing).
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v * b.v;
        for i in 0..K {
            // subsets of {i}: {} {i}
            let mut acc = 0.0;
            acc += a.v * b.g[i];
            acc += a.g[i] * b.v;
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                // subsets of {i,j}: {} {i} {j} {ij}
                let mut acc = 0.0;
                acc += a.v * b.h[i][j];
                acc += a.g[i] * b.g[j];
                acc += a.g[j] * b.g[i];
                acc += a.h[i][j] * b.v;
                out.h[i][j] = acc;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    // subsets of {i,j,k}: {} {i} {j} {ij} {k} {ik} {jk} {ijk}
                    let mut acc = 0.0;
                    acc += a.v * b.t3[i][j][k];
                    acc += a.g[i] * b.h[j][k];
                    acc += a.g[j] * b.h[i][k];
                    acc += a.h[i][j] * b.g[k];
                    acc += a.g[k] * b.h[i][j];
                    acc += a.h[i][k] * b.g[j];
                    acc += a.h[j][k] * b.g[i];
                    acc += a.t3[i][j][k] * b.v;
                    out.t3[i][j][k] = acc;
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        // subsets of {i,j,k,l} in bit order sub = 0..16
                        let mut acc = 0.0;
                        acc += a.v * b.t4[i][j][k][l];
                        acc += a.g[i] * b.t3[j][k][l];
                        acc += a.g[j] * b.t3[i][k][l];
                        acc += a.h[i][j] * b.h[k][l];
                        acc += a.g[k] * b.t3[i][j][l];
                        acc += a.h[i][k] * b.h[j][l];
                        acc += a.h[j][k] * b.h[i][l];
                        acc += a.t3[i][j][k] * b.g[l];
                        acc += a.g[l] * b.t3[i][j][k];
                        acc += a.h[i][l] * b.h[j][k];
                        acc += a.h[j][l] * b.h[i][k];
                        acc += a.t3[i][j][l] * b.g[k];
                        acc += a.h[k][l] * b.h[i][j];
                        acc += a.t3[i][k][l] * b.g[j];
                        acc += a.t3[j][k][l] * b.g[i];
                        acc += a.t4[i][j][k][l] * b.v;
                        out.t4[i][j][k][l] = acc;
                    }
                }
            }
        }
        out
    }

    /// Ref-taking elementwise sum, the by-ref twin of the `std::ops::Add`
    /// operator (which consumes by value). Mirrors the inherent `mul`/`scale`
    /// API so a chain like `a.mul(&b).add(&c)` reads uniformly without moving
    /// out of the borrowed operands.
    pub fn add(&self, o: &Self) -> Self {
        *self + *o
    }

    /// Ref-taking elementwise difference, the by-ref twin of `std::ops::Sub`.
    pub fn sub(&self, o: &Self) -> Self {
        *self + o.scale(-1.0)
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
    ///
    /// # Codegen
    ///
    /// Evaluated as a compact closed form (the Bell(4)=15 set-partitions of
    /// `t4`, Bell(3)=5 of `t3`, …) instead of routing through the recursive
    /// [`jet_algebra::faa_di_bruno`] walker (per-output `for_each_partition`
    /// recursion + per-block `SlotBuf` + closure dispatch). The loop nest is
    /// identical to the walker's (`for i,j,k,l`); only the per-entry partition
    /// sum is straight-line, so this does NOT unroll over `K` and does NOT
    /// bloat code — measured on a `Tower4<9>` compose-and-read consumer the new
    /// form is both faster and SMALLER (asm: 94 outlined walker `bl` calls → 0,
    /// 47.5 KiB → 16.7 KiB, +197 NEON `.2d` ops).
    ///
    /// BIT-IDENTICAL to the walker: each channel's terms are emitted in the
    /// walker's exact partition-enumeration order, each term's block products
    /// are left-associated exactly as the walker's `prod *= block`, and the
    /// per-channel `acc` accumulator mirrors the walker's `total = 0.0` start
    /// (so signed-zero products collapse to `+0.0` identically). The order-4
    /// term sequence was generated from the walker's own enumeration. Proven
    /// `to_bits`-identical on `v`/`g`/`h`/`t3`/`t4` across `K ∈ {2,3,4,9}`,
    /// 5000 random inputs each (zeroed / sign-varied stacks included).
    pub fn compose_unary(&self, d: [f64; 5]) -> Self {
        let mut out = Self::zero();
        out.v = d[0];
        for i in 0..K {
            let mut acc = 0.0;
            acc += d[1] * self.g[i];
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = 0.0;
                acc += d[1] * self.h[i][j];
                acc += d[2] * self.g[i] * self.g[j];
                out.h[i][j] = acc;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    // walker partitions: {ijk} {ij}{k} {ik}{j} {i}{jk} {i}{j}{k}
                    let mut acc = 0.0;
                    acc += d[1] * self.t3[i][j][k];
                    acc += d[2] * self.h[i][j] * self.g[k];
                    acc += d[2] * self.h[i][k] * self.g[j];
                    acc += d[2] * self.g[i] * self.h[j][k];
                    acc += d[3] * self.g[i] * self.g[j] * self.g[k];
                    out.t3[i][j][k] = acc;
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        // Bell(4)=15 partitions, walker enumeration order.
                        let mut acc = 0.0;
                        acc += d[1] * self.t4[i][j][k][l];
                        acc += d[2] * self.t3[i][j][k] * self.g[l];
                        acc += d[2] * self.t3[i][j][l] * self.g[k];
                        acc += d[2] * self.h[i][j] * self.h[k][l];
                        acc += d[3] * self.h[i][j] * self.g[k] * self.g[l];
                        acc += d[2] * self.t3[i][k][l] * self.g[j];
                        acc += d[2] * self.h[i][k] * self.h[j][l];
                        acc += d[3] * self.h[i][k] * self.g[j] * self.g[l];
                        acc += d[2] * self.h[i][l] * self.h[j][k];
                        acc += d[2] * self.g[i] * self.t3[j][k][l];
                        acc += d[3] * self.g[i] * self.h[j][k] * self.g[l];
                        acc += d[3] * self.h[i][l] * self.g[j] * self.g[k];
                        acc += d[3] * self.g[i] * self.h[j][l] * self.g[k];
                        acc += d[3] * self.g[i] * self.g[j] * self.h[k][l];
                        acc += d[4] * self.g[i] * self.g[j] * self.g[k] * self.g[l];
                        out.t4[i][j][k][l] = acc;
                    }
                }
            }
        }
        out
    }

    /// Compose with a unary special-function whose `[f64; 5]` derivative STACK is
    /// built from the base value through `stack_fn` — the scalar arm of the
    /// generic-over-[`Lane`](crate::jet_scalar::Lane) compose seam (see
    /// [`Tower4Lane::compose_unary_with`]). Evaluates `stack_fn(self.v)` ONCE and
    /// forwards to [`Self::compose_unary`], so it is BIT-IDENTICAL to the explicit
    /// `self.compose_unary(stack_fn(self.v))`. Writing a program against this seam
    /// lets it re-instantiate, unchanged, at [`Tower4Lane`] (where each of the four
    /// lanes carries a DISTINCT base value and `stack_fn` is re-run per lane).
    #[inline]
    pub fn compose_unary_with(&self, stack_fn: impl Fn(f64) -> [f64; 5]) -> Self {
        self.compose_unary(stack_fn(self.v))
    }

    /// Single-active-slot fast path for [`Self::compose_unary`].
    ///
    /// When the inner jet `self` has derivative support ONLY on the all-`slot`
    /// diagonal channels — i.e. it is a univariate jet in primary `slot`
    /// scattered into the `K`-wide layout (`g[a] = 0`, `h[a][b] = 0`,
    /// `t3 = 0`, `t4 = 0` for any axis `≠ slot`) — the multivariate Faà di
    /// Bruno walk collapses. Every output channel whose axis tuple contains an
    /// axis `≠ slot` is structurally `0`: each set-partition has a block
    /// covering that axis, that block reads an off-`slot` derivative of `self`
    /// (which is `0`), so the block product and the whole partition vanish, and
    /// the channel sums to the walker's `total = 0.0` start, i.e. `+0.0`. Only
    /// the five diagonal channels (`v`, `g[slot]`, `h[slot][slot]`,
    /// `t3[slot]³`, `t4[slot]⁴`) survive.
    ///
    /// This computes exactly those five as STRAIGHT-LINE accumulations, each in
    /// the EXACT term order of [`Self::compose_unary`]'s diagonal
    /// (`i = j = k = l = slot`) case — so they are BIT-IDENTICAL to
    /// [`Self::compose_unary`] on the diagonal — and leaves every other channel
    /// at the zero-init `+0.0`, which the full walk also produces (the
    /// off-`slot` collapse is `to_bits`-`+0.0`, signed-zero products included;
    /// proven across `K ∈ {2,3,4,9}`, 5000 single-slot inputs each). At any
    /// `K ≥ 2` this is far fewer floating-point operations than materialising
    /// the full `1 + K + K² + K³ + K⁴` channel set whose off-diagonal entries
    /// are all zero, and far cheaper than the recursive set-partition walker the
    /// diagonal channels previously routed through (a measured ~9.5× speedup vs
    /// the full `compose_unary`, recovering a 5.9× walker regression at the
    /// `K ∈ {2,3}` BMS tower widths).
    ///
    /// `#[inline]` so an adopting consumer pays no `bl` call (uninlined, the
    /// five-channel build does not amortise the call/spill overhead).
    ///
    /// # Precondition
    ///
    /// The caller guarantees the single-active-slot structure. If it does not
    /// hold, the off-`slot` channels would be wrongly zeroed; use the full
    /// [`Self::compose_unary`] in that case.
    #[inline]
    pub fn compose_unary_single_slot(&self, d: [f64; 5], slot: usize) -> Self {
        let mut out = Self::zero();
        let s = slot;
        let g = self.g[s];
        let h = self.h[s][s];
        let t3 = self.t3[s][s][s];
        let t4 = self.t4[s][s][s][s];
        out.v = d[0];
        // g (i=s): d1*g
        out.g[s] = {
            let mut acc = 0.0;
            acc += d[1] * g;
            acc
        };
        // h (i=j=s): d1*h + d2*g*g
        out.h[s][s] = {
            let mut acc = 0.0;
            acc += d[1] * h;
            acc += d[2] * g * g;
            acc
        };
        // t3 (i=j=k=s): exact term order of compose_unary's inner loop.
        out.t3[s][s][s] = {
            let mut acc = 0.0;
            acc += d[1] * t3;
            acc += d[2] * h * g;
            acc += d[2] * h * g;
            acc += d[2] * g * h;
            acc += d[3] * g * g * g;
            acc
        };
        // t4 (i=j=k=l=s): exact term order of compose_unary's inner loop.
        out.t4[s][s][s][s] = {
            let mut acc = 0.0;
            acc += d[1] * t4;
            acc += d[2] * t3 * g;
            acc += d[2] * t3 * g;
            acc += d[2] * h * h;
            acc += d[3] * h * g * g;
            acc += d[2] * t3 * g;
            acc += d[2] * h * h;
            acc += d[3] * h * g * g;
            acc += d[2] * h * h;
            acc += d[2] * g * t3;
            acc += d[3] * g * h * g;
            acc += d[3] * h * g * g;
            acc += d[3] * g * h * g;
            acc += d[3] * g * g * h;
            acc += d[4] * g * g * g * g;
            acc
        };
        out
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
    ///
    /// # Codegen
    ///
    /// Order-≤2 Faà di Bruno is a tiny closed form, so this evaluates it
    /// directly instead of routing through the generic
    /// [`jet_algebra::faa_di_bruno`] set-partition walker (recursion + per-block
    /// closure dispatch). That matters because this is the kernel under EVERY
    /// packed scalar — [`crate::jet_scalar::Order2`] / `OneSeed` / `TwoSeed`
    /// composition all bottom out here — so the straight-line form (whose inner
    /// loops auto-vectorise to NEON/SSE 2-wide and which emits zero outlined
    /// walker calls) lifts all of them at once.
    ///
    /// The term and accumulation order is BIT-IDENTICAL to the walker it
    /// replaces: each output channel mirrors the walker's `total = 0.0` start
    /// (the explicit `acc` accumulator), so a signed-zero product collapses to
    /// `+0.0` exactly as `total += prod` does. Proven `to_bits`-identical on
    /// `v`/`g`/`h` across `K ∈ {2,3,4,9}`, 5000 random inputs each (incl.
    /// zeroed / sign-varied stacks). The order-≤2 walker partitions are:
    ///   `g[i]`   = `f′·u_i`                   (single block `{i}`)
    ///   `h[i][j]` = `f′·u_ij + (f″·u_i)·u_j`  (blocks `{ij}` then `{i}{j}`),
    /// with `f′ = d[1]`, `f″ = d[2]`, `u_* = self.{g,h}`.
    pub fn compose_unary(&self, d: [f64; 3]) -> Self {
        let mut out = Self::zero();
        out.v = d[0];
        for i in 0..K {
            let mut acc = 0.0;
            acc += d[1] * self.g[i];
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = 0.0;
                acc += d[1] * self.h[i][j];
                acc += d[2] * self.g[i] * self.g[j];
                out.h[i][j] = acc;
            }
        }
        out
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

    /// e^self.
    pub fn exp(&self) -> Self {
        let e = self.v.exp();
        self.compose_unary([e, e, e])
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

/// Truncated THIRD-order multivariate Taylor scalar in `K` variables.
///
/// The value/gradient/Hessian/third-derivative sibling of [`Tower4`], standing
/// between [`Tower2`] and [`Tower4`]. Every channel it carries (`v`, `g`, `h`,
/// `t3`) is computed by the SAME shared Leibniz / Faà-di-Bruno kernels
/// [`Tower4`] uses for those orders, and the order-≤3 terms of those kernels
/// read only the order-≤3 channels of their inputs (the order-3 Faà-di-Bruno
/// partitions never reach the f⁗ stack slot or the inner `t4` tensor — see
/// [`Tower4::compose_unary`]). So for any program written over both towers the
/// order-≤3 outputs are *bit-identical*: dropping the fourth tensor cannot
/// perturb the value, gradient, Hessian, or third derivatives.
///
/// It exists purely for performance, exactly like [`Tower2`]: a consumer that
/// needs up to third derivatives (the survival location-scale row kernel reads
/// `g`, the diagonal `h`, and the diagonal `t3`, but never `t4`) pays the
/// `K³` third-tensor arithmetic but skips the `K⁴` fourth-tensor
/// product/composition that otherwise dominates the per-row cost.
#[derive(Clone, Copy, Debug)]
pub struct Tower3<const K: usize> {
    /// Value ℓ.
    pub v: f64,
    /// Gradient ∂ℓ/∂p_a.
    pub g: [f64; K],
    /// Hessian ∂²ℓ/∂p_a∂p_b (symmetric).
    pub h: [[f64; K]; K],
    /// Third derivatives ∂³ℓ/∂p_a∂p_b∂p_c (fully symmetric).
    pub t3: [[[f64; K]; K]; K],
}

impl<const K: usize> Tower3<K> {
    /// The additive identity.
    pub fn zero() -> Self {
        Self {
            v: 0.0,
            g: [0.0; K],
            h: [[0.0; K]; K],
            t3: [[[0.0; K]; K]; K],
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
    /// axes are `labels` (length 0..=3): value, `g`, `h`, `t3`.
    #[inline]
    fn deriv(&self, labels: &[usize]) -> f64 {
        assert!(
            labels.len() <= 3,
            "Tower3 carries at most third-order derivatives"
        );
        match labels.len() {
            0 => self.v,
            1 => self.g[labels[0]],
            2 => self.h[labels[0]][labels[1]],
            _ => self.t3[labels[0]][labels[1]][labels[2]],
        }
    }

    /// Exact truncated (order ≤ 3) Leibniz product. The `v`/`g`/`h`/`t3`
    /// channels match [`Tower4::mul`] term-for-term.
    ///
    /// # Codegen
    ///
    /// Straight-line per-entry subset sums instead of the
    /// [`jet_algebra::leibniz_product`] walker — the order-≤3 sibling of
    /// [`Tower4::mul`] (no `t4`). Loop nest unchanged, no unroll over `K`, no
    /// code bloat; auto-vectorises. BIT-IDENTICAL: terms in the walker's exact
    /// subset order with an `acc = 0.0` accumulator start (load-bearing for the
    /// signed-zero leading product on exact-`0.0` jet channels). Proven
    /// `to_bits`-identical on `v`/`g`/`h`/`t3` across `K ∈ {2,3,4,9}`, 5000
    /// zero/sign-stressed inputs each (these channel formulas are exactly the
    /// `g`/`h`/`t3` of the [`Tower4::mul`] oracle, which passes that stress).
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v * b.v;
        for i in 0..K {
            let mut acc = 0.0;
            acc += a.v * b.g[i];
            acc += a.g[i] * b.v;
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = 0.0;
                acc += a.v * b.h[i][j];
                acc += a.g[i] * b.g[j];
                acc += a.g[j] * b.g[i];
                acc += a.h[i][j] * b.v;
                out.h[i][j] = acc;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    // subsets of {i,j,k}: {} {i} {j} {ij} {k} {ik} {jk} {ijk}
                    let mut acc = 0.0;
                    acc += a.v * b.t3[i][j][k];
                    acc += a.g[i] * b.h[j][k];
                    acc += a.g[j] * b.h[i][k];
                    acc += a.h[i][j] * b.g[k];
                    acc += a.g[k] * b.h[i][j];
                    acc += a.h[i][k] * b.g[j];
                    acc += a.h[j][k] * b.g[i];
                    acc += a.t3[i][j][k] * b.v;
                    out.t3[i][j][k] = acc;
                }
            }
        }
        out
    }

    /// Ref-taking elementwise sum, the by-ref twin of the `std::ops::Add`
    /// operator (which consumes by value). Mirrors the inherent `mul`/`scale`
    /// API so a chain like `a.mul(&b).add(&c)` reads uniformly without moving
    /// out of the borrowed operands.
    pub fn add(&self, o: &Self) -> Self {
        *self + *o
    }

    /// Ref-taking elementwise difference, the by-ref twin of `std::ops::Sub`.
    pub fn sub(&self, o: &Self) -> Self {
        *self + o.scale(-1.0)
    }

    /// Exact (order ≤ 3) multivariate Faà di Bruno composition `f ∘ self`.
    ///
    /// `d = [f(u), f′(u), f″(u), f‴(u)]` evaluated at `u = self.v`. The
    /// `v`/`g`/`h`/`t3` channels match [`Tower4::compose_unary`] term-for-term
    /// (which uses only `d[0..=3]` for those orders), so this is a strict
    /// truncation, not an approximation. The full-order `[f64; 5]` derivative
    /// stacks the families already produce can be passed by slicing their first
    /// four entries.
    ///
    /// # Codegen
    ///
    /// Order-≤3 Faà di Bruno written as a compact closed form instead of the
    /// recursive [`jet_algebra::faa_di_bruno`] walker — the order-≤2 sibling of
    /// [`Tower4::compose_unary`], one tensor order shallower. The loop nest is
    /// unchanged (no unroll over `K`, no code bloat: measured on a `Tower3<9>`
    /// compose-and-read consumer the new form is faster and SMALLER — asm: 71
    /// walker `bl` calls → 0, 39.5 KiB → 13.9 KiB, +197 NEON `.2d` ops).
    /// BIT-IDENTICAL: terms in the walker's exact partition order, left-
    /// associated block products, `acc = 0.0` accumulator start. Proven
    /// `to_bits`-identical on `v`/`g`/`h`/`t3` across `K ∈ {2,3,4,9}`, 5000
    /// random inputs each.
    pub fn compose_unary(&self, d: [f64; 4]) -> Self {
        let mut out = Self::zero();
        out.v = d[0];
        for i in 0..K {
            let mut acc = 0.0;
            acc += d[1] * self.g[i];
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = 0.0;
                acc += d[1] * self.h[i][j];
                acc += d[2] * self.g[i] * self.g[j];
                out.h[i][j] = acc;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    // walker partitions: {ijk} {ij}{k} {ik}{j} {i}{jk} {i}{j}{k}
                    let mut acc = 0.0;
                    acc += d[1] * self.t3[i][j][k];
                    acc += d[2] * self.h[i][j] * self.g[k];
                    acc += d[2] * self.h[i][k] * self.g[j];
                    acc += d[2] * self.g[i] * self.h[j][k];
                    acc += d[3] * self.g[i] * self.g[j] * self.g[k];
                    out.t3[i][j][k] = acc;
                }
            }
        }
        out
    }

    /// Compose with a unary special-function whose `[f64; 4]` derivative STACK is
    /// built from the base value through `stack_fn` — the scalar arm of the
    /// generic-over-[`Lane`](crate::jet_scalar::Lane) compose seam (see
    /// [`Tower3Lane::compose_unary_with`]). Evaluates `stack_fn(self.v)` ONCE and
    /// forwards to [`Self::compose_unary`], so it is BIT-IDENTICAL to the explicit
    /// `self.compose_unary(stack_fn(self.v))`. The order-≤3 sibling of
    /// [`Tower4::compose_unary_with`].
    #[inline]
    pub fn compose_unary_with(&self, stack_fn: impl Fn(f64) -> [f64; 4]) -> Self {
        self.compose_unary(stack_fn(self.v))
    }

    /// Single-active-slot fast path for [`Self::compose_unary`] — the order-≤3
    /// sibling of [`Tower4::compose_unary_single_slot`]. When `self` carries
    /// derivative support only on the all-`slot` diagonal, every output channel
    /// touching an axis `≠ slot` collapses to the walker's `total = 0.0` start
    /// (`+0.0`), so only `v`, `g[slot]`, `h[slot][slot]`, `t3[slot]³` survive.
    /// These four are computed as STRAIGHT-LINE accumulations, each in the EXACT
    /// term order of [`Self::compose_unary`]'s diagonal (`i = j = k = slot`)
    /// case (BIT-IDENTICAL to the full path on the diagonal); off-`slot`
    /// channels stay at the zero-init `+0.0` the full walk also yields (proven
    /// `to_bits` across `K ∈ {2,3,4,9}`). This drops the recursive
    /// set-partition walker the diagonal channels previously routed through,
    /// recovering its measured ~5.9× regression at the `K ∈ {2,3}` BMS tower
    /// widths. Caller guarantees the single-slot precondition; otherwise use
    /// [`Self::compose_unary`].
    #[inline]
    pub fn compose_unary_single_slot(&self, d: [f64; 4], slot: usize) -> Self {
        let mut out = Self::zero();
        let s = slot;
        let g = self.g[s];
        let h = self.h[s][s];
        let t3 = self.t3[s][s][s];
        out.v = d[0];
        // g (i=s): d1*g
        out.g[s] = {
            let mut acc = 0.0;
            acc += d[1] * g;
            acc
        };
        // h (i=j=s): d1*h + d2*g*g
        out.h[s][s] = {
            let mut acc = 0.0;
            acc += d[1] * h;
            acc += d[2] * g * g;
            acc
        };
        // t3 (i=j=k=s): exact term order of compose_unary's inner loop.
        out.t3[s][s][s] = {
            let mut acc = 0.0;
            acc += d[1] * t3;
            acc += d[2] * h * g;
            acc += d[2] * h * g;
            acc += d[2] * g * h;
            acc += d[3] * g * g * g;
            acc
        };
        out
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
                }
            }
        }
        out
    }
}

impl<const K: usize> jet_algebra::JetAlgebra<4> for Tower3<K> {
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
        out
    }
}

impl<const K: usize> std::ops::Add for Tower3<K> {
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
                }
            }
        }
        out
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

pub fn ln_gamma_derivative_stack_order2(x: f64) -> [f64; 3] {
    [
        statrs::function::gamma::ln_gamma(x),
        digamma_positive(x),
        polygamma_positive(1, x),
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
/// [`crate::jet_scalar::JetScalar`] interface, so the SAME expression can be
/// re-instantiated at whatever order / representation a consumer needs
/// ([`crate::jet_scalar::Order2`] for `(v, g, H)`,
/// [`crate::jet_scalar::OneSeed`] for the contracted third,
/// [`crate::jet_scalar::TwoSeed`] for the contracted fourth, or the full
/// [`Tower4`] for every channel at once).
///
/// This is additive to [`RowNllProgram`] (which is `Tower4`-specialised): a
/// program implementing this generic trait gets the small contracted scalars for
/// free, dissolving the dense-`Tower4<9>` cost objection in the location-scale
/// gates (doc §A.4). An existing `Tower4`-only [`RowNllProgram`] continues to
/// work unchanged; new families should prefer this generic trait.
///
/// Because a `Tower4`-specialised `row_nll` body uses only
/// `add`/`sub`/`mul`/`scale`/`exp`/`ln`/… — all of which this trait also
/// provides — the same body is expressible directly over `S: JetScalar<K>`.
/// A program written that way needs no `Tower4`-specialised method and routes
/// the directional and joint-Hessian gates through the contracted scalars from
/// a single definition.
pub trait RowNllProgramGeneric<const K: usize>: Send + Sync {
    /// Number of observations the program covers.
    fn n_rows(&self) -> usize;

    /// Current primary-scalar values for `row` (where to seed the scalar).
    fn primaries(&self, row: usize) -> Result<[f64; K], String>;

    /// The row NLL evaluated on a generic jet scalar. `p[a]` arrives pre-seeded
    /// (base value + per-scalar nilpotent directions) by the caller; the body
    /// uses ONLY [`crate::jet_scalar::JetScalar`] ops and per-row data
    /// (response, censoring, offsets) entering as constants.
    fn row_nll_generic<S: crate::jet_scalar::JetScalar<K>>(
        &self,
        row: usize,
        p: &[S; K],
    ) -> Result<S, String>;
}

/// Evaluate a generic program at the value/gradient/Hessian scalar
/// [`crate::jet_scalar::Order2`], returning `(nll, ∇, H)` — the
/// `row_kernel` channel — WITHOUT materialising any third / fourth tensor.
///
/// This is the production seam for the inner-Newton `(v, g, H)` path: the row
/// loss is written ONCE in `row_nll_generic`, and this routes it through the
/// cheap order-2 scalar. The single source of truth means the gradient and
/// Hessian cannot desync from the value (the #736 / #948 bug genus).
pub fn generic_row_kernel<const K: usize, P: RowNllProgramGeneric<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<(f64, [f64; K], [[f64; K]; K]), String> {
    let base = prog.primaries(row)?;
    let vars: [crate::jet_scalar::Order2<K>; K] = std::array::from_fn(|a| {
        <crate::jet_scalar::Order2<K> as crate::jet_scalar::JetScalar<K>>::variable(base[a], a)
    });
    let s = prog.row_nll_generic(row, &vars)?;
    Ok((crate::jet_scalar::JetScalar::value(&s), s.g(), s.h()))
}

/// Evaluate a generic program at the one-seed scalar
/// [`crate::jet_scalar::OneSeed`], returning the contracted third
/// `Σ_c ℓ_{abc} dir_c` — the `row_third_contracted(dir)` channel — WITHOUT
/// materialising the dense `t3` tensor. The contraction direction is folded
/// INTO the differentiation by the nilpotent ε seeded with `dir`.
pub fn generic_third_contracted<const K: usize, P: RowNllProgramGeneric<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    let base = prog.primaries(row)?;
    let vars: [crate::jet_scalar::OneSeed<K>; K] =
        std::array::from_fn(|a| crate::jet_scalar::OneSeed::seed_direction(base[a], a, dir[a]));
    let s = prog.row_nll_generic(row, &vars)?;
    Ok(s.contracted_third())
}

/// Evaluate a generic program at the two-seed scalar
/// [`crate::jet_scalar::TwoSeed`], returning the contracted fourth
/// `Σ_{cd} ℓ_{abcd} u_c v_d` — the `row_fourth_contracted(u, v)` channel —
/// WITHOUT materialising the dense `t4` tensor.
pub fn generic_fourth_contracted<const K: usize, P: RowNllProgramGeneric<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir_u: &[f64; K],
    dir_v: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    let base = prog.primaries(row)?;
    let vars: [crate::jet_scalar::TwoSeed<K>; K] =
        std::array::from_fn(|a| crate::jet_scalar::TwoSeed::seed(base[a], a, dir_u[a], dir_v[a]));
    let s = prog.row_nll_generic(row, &vars)?;
    Ok(s.contracted_fourth())
}

/// Evaluate a generic program at the full dense [`Tower4`] scalar, returning
/// every channel `(v, g, h, t3, t4)` in one pass. Used where the UNCONTRACTED
/// third / fourth tensors are needed (the BMS rigid `third_full` / `fourth_full`
/// caches): the dense tensors come from the SAME `row_nll_generic` expression
/// the order-2 / contracted scalars consume, so there is a single source of
/// truth across every channel.
pub fn generic_full_tower<const K: usize, P: RowNllProgramGeneric<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<Tower4<K>, String> {
    let base = prog.primaries(row)?;
    let vars: [Tower4<K>; K] = std::array::from_fn(|a| Tower4::variable(base[a], a));
    prog.row_nll_generic(row, &vars)
}

// ── The RowJet bridge: one row-NLL body over scalar jets AND lane towers ─
//
// `JetScalar<K>` (jet_scalar.rs) abstracts the SCALAR jets — its `value()`
// returns one `f64`, so the `f64x4` lane towers ([`Tower3Lane`] / [`Tower4Lane`])
// CANNOT implement it (their value channel is four rows). `compose_unary_with`
// exists as an inherent method on BOTH the scalar towers and the lane towers, but
// as separate inherent methods, not a shared trait bound — so a row-NLL body
// written `<S: JetScalar<K>>` could not be instantiated at `Tower4Lane`, and the
// 4-rows-per-pass SIMD batch path could not reuse the single source.
//
// [`RowJet<K>`] is that shared bound. It exposes exactly the ops a row-NLL body
// needs — `constant` / `variable` / `add` / `sub` / `mul` / `scale` / `neg`, the
// value-derived `compose_unary_with`, and a per-lane domain `guard` — over BOTH
// representations. A blanket impl makes every scalar `JetScalar<K>` a `RowJet<K>`
// (so the scalar call sites compile unchanged and bit-identically), and explicit
// impls route the `f64x4` lane towers through their existing per-lane methods. A
// body written once over `R: RowJet<K>` then instantiates at a scalar jet for the
// `(v, g, H)` / contracted-tensor channels AND at a lane tower for the batch.

/// The verdict of a per-lane [`RowJet::guard`] domain check.
///
/// A scalar jet (a [`crate::jet_scalar::JetScalar`] via the blanket impl) carries
/// ONE value, so it reports `lanes == 1` and a one-bit mask. A lane tower
/// ([`Tower3Lane`] / [`Tower4Lane`] over `f64x4`) carries FOUR rows, so it reports
/// `lanes == 4` and one mask bit per lane. The mask lets a batched program bail
/// exactly the offending 4-group to the scalar tail ([`any_failed`](Self::any_failed)),
/// or inspect which lanes tripped ([`lane_failed`](Self::lane_failed)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardVerdict {
    lanes: u8,
    failed_mask: u8,
}

impl GuardVerdict {
    /// A scalar (1-lane) verdict: `pass == true` ⇒ no failure.
    #[inline]
    pub fn scalar(pass: bool) -> Self {
        Self { lanes: 1, failed_mask: if pass { 0 } else { 1 } }
    }
    /// A 4-lane verdict from a per-lane failure mask (bit `i` ⇒ lane `i` failed).
    #[inline]
    pub fn lanes4(failed_mask: u8) -> Self {
        Self { lanes: 4, failed_mask: failed_mask & 0x0f }
    }
    /// Number of active lanes inspected (1 scalar, 4 batch).
    #[inline]
    pub fn lanes(self) -> usize {
        self.lanes as usize
    }
    /// True iff every inspected lane satisfied the predicate.
    #[inline]
    pub fn all_pass(self) -> bool {
        self.failed_mask == 0
    }
    /// True iff at least one inspected lane failed the predicate.
    #[inline]
    pub fn any_failed(self) -> bool {
        self.failed_mask != 0
    }
    /// True iff lane `i` failed the predicate.
    #[inline]
    pub fn lane_failed(self, i: usize) -> bool {
        (self.failed_mask >> i) & 1 == 1
    }
    /// The raw failure mask (bit `i` ⇒ lane `i` failed).
    #[inline]
    pub fn failed_mask(self) -> u8 {
        self.failed_mask
    }
}

/// Copy-or-zero-pad a derivative stack from length `N` to length `M`. Used by the
/// [`RowJet::compose_unary_with`] impls to bridge a program's chosen stack length
/// to each tower's native compose width ([`Tower4Lane`]: 5, [`Tower3Lane`]: 4).
/// `M ≥ N` zero-pads the unseeded high derivatives; `M < N` drops the unused tail
/// — both total, so the order-`(M−1)` tower reads exactly the channels it needs
/// and never an uninitialised entry. With `N == M` it is a verbatim copy (the
/// common `N == 5` case is bit-identical to passing the stack straight through).
#[inline]
fn resize_stack<const N: usize, const M: usize>(s: [f64; N]) -> [f64; M] {
    let mut out = [0.0_f64; M];
    let m = N.min(M);
    out[..m].copy_from_slice(&s[..m]);
    out
}

/// The shared row-NLL algebra over BOTH the scalar jets and the `f64x4` lane
/// towers — the bound that lets ONE single-source row-NLL body SIMD-batch 4
/// rows/pass without a dual-source copy (module §"The RowJet bridge").
///
/// Every scalar [`crate::jet_scalar::JetScalar<K>`] is a `RowJet<K>` via the
/// blanket impl below (`Value = f64`), bit-identically to its `JetScalar`
/// methods; [`Tower3Lane`] / [`Tower4Lane`] over `f64x4` are `RowJet<K>` with
/// `Value = [f64; 4]`, routing through their per-lane methods so lane `i` of a
/// batched evaluation is `to_bits`-identical to the scalar evaluation on row `i`.
pub trait RowJet<const K: usize>: Copy {
    /// The value channel(s) seen by [`guard`](Self::guard) and
    /// [`values`](Self::values): a single `f64` on a scalar jet, `[f64; 4]` on an
    /// `f64x4` lane tower.
    type Value: Copy;

    /// A constant (value `c`, all derivatives zero), broadcast to every lane.
    fn constant(c: f64) -> Self;
    /// The seeded primary `slot` at value `x` (unit first derivative in `slot`),
    /// broadcast to every lane. Per-lane-DISTINCT seeding for the batch path is
    /// done by the lane instantiators ([`generic_batched_fourth_tower`] /
    /// [`generic_batched_third_tower`]), which build the tower variables directly
    /// from each row's primaries; this method is for any row-invariant auxiliary
    /// variable a body introduces.
    fn variable(x: f64, slot: usize) -> Self;
    /// The value channel(s): `f64` (scalar) or `[f64; 4]` (lane).
    fn values(&self) -> Self::Value;

    /// Truncated Leibniz `self + o`.
    fn add(&self, o: &Self) -> Self;
    /// Truncated Leibniz `self − o`.
    fn sub(&self, o: &Self) -> Self;
    /// Truncated Leibniz `self · o`.
    fn mul(&self, o: &Self) -> Self;
    /// Multiply every channel by the plain scalar `s`.
    fn scale(&self, s: f64) -> Self;
    /// Negate every channel. Defaults to `scale(-1.0)`; the blanket overrides it
    /// to delegate to [`crate::jet_scalar::JetScalar::neg`].
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// Faà di Bruno compose with a unary special function whose `[f64; N]`
    /// derivative stack is built from the running base value PER LANE through
    /// `stack_fn`. This is the SHARED-TRAIT version of the `compose_unary_with`
    /// inherent method that already exists on both the scalar towers and the lane
    /// towers: on a scalar jet `stack_fn` is run once at the value; on an `f64x4`
    /// lane tower it is re-run per lane (the four rows carry four distinct base
    /// values), so lane `i` is `to_bits`-identical to the scalar result on row `i`.
    /// Making it a trait method is precisely what lets a body written once over
    /// `R: RowJet<K>` instantiate at the batch towers. `N` is widened/narrowed to
    /// the tower's native width by [`resize_stack`] (`N == 5` is a verbatim copy).
    fn compose_unary_with<const N: usize>(&self, stack_fn: impl Fn(f64) -> [f64; N]) -> Self;

    /// Per-lane domain guard: evaluate `pred` on each active lane's value channel
    /// and report which lanes failed (see [`GuardVerdict`]). A scalar jet checks
    /// its one value; a lane tower checks all four. Lets a batched program detect
    /// an out-of-domain row in a 4-group and bail that group to the scalar tail.
    fn guard(&self, pred: impl Fn(f64) -> bool) -> GuardVerdict;

    /// Per-lane scale: multiply every channel by the per-lane factor `s`
    /// ([`Self::Value`]). On a scalar jet `Self::Value = f64`, so this is exactly
    /// [`scale`](Self::scale) and the scalar call sites stay BIT-IDENTICAL when
    /// `.scale(x)` is rewritten to `.scale_rows(x)`; on an `f64x4` lane tower
    /// `Self::Value = [f64; 4]` and lane `i` is multiplied by `s[i]`. This is the
    /// primitive that lets a batched body carry CONTINUOUS per-row data — the
    /// survival `covariance_ones` / `z_sum` / observation-weight `wi` factors that
    /// enter the jet algebra as `.scale(per_row_value)` and that the single-`f64`
    /// [`scale`](Self::scale) would broadcast wrongly across the four rows. Build
    /// `s` from the lane→row map with [`pack_rows`](Self::pack_rows).
    fn scale_rows(&self, s: Self::Value) -> Self;

    /// Gather a per-lane auxiliary datum from the lane→row map `rows`: `value_of(r)`
    /// is evaluated for each active lane's row and packed into [`Self::Value`] (a
    /// single `f64` on a scalar jet, `[f64; 4]` on an `f64x4` lane tower). This is
    /// how a body written once over [`RowJet`] feeds per-row CONTINUOUS data (the
    /// arguments to [`scale_rows`](Self::scale_rows)) into the batch path without
    /// knowing the concrete representation: the program holds the per-row data and
    /// the caller threads `rows` (length 1 scalar, length 4 batch) into
    /// [`RowNllProgramRowJet::row_nll`], so the body writes
    /// `x.scale_rows(R::pack_rows(rows, |r| self.cov(r)))`. A multiplicative weight
    /// buried in a `compose_unary_with` stack is pulled out the same way:
    /// `x.compose_unary_with(|u| stack(u, 1.0)).scale_rows(R::pack_rows(rows, |r| self.wi(r)))`.
    /// (Binary per-row branches such as the event indicator `di` are kept
    /// lane-uniform by grouping and the [`guard`](Self::guard) bail, not packed.)
    fn pack_rows(rows: &[usize], value_of: impl Fn(usize) -> f64) -> Self::Value;

    // ── value-derived transcendental conveniences ───────────────────────
    // Each routes through `compose_unary_with` with the SAME derivative stack the
    // corresponding `JetScalar` method uses, so on a scalar jet (blanket) the
    // result is bit-identical to the `JetScalar` method, and on a lane tower lane
    // `i` is bit-identical to the scalar result on row `i`.

    /// `e^self`.
    fn exp(&self) -> Self {
        self.compose_unary_with(|u| {
            let e = u.exp();
            [e, e, e, e, e]
        })
    }
    /// `ln(self)`. Caller guarantees positivity.
    fn ln(&self) -> Self {
        self.compose_unary_with(|u| {
            let r = 1.0 / u;
            [u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r]
        })
    }
    /// `√self`. Caller guarantees positivity.
    fn sqrt(&self) -> Self {
        self.compose_unary_with(|u| {
            let s = u.sqrt();
            [s, 0.5 / s, -0.25 / (u * s), 0.375 / (u * u * s), -0.9375 / (u * u * u * s)]
        })
    }
    /// `1/self`.
    fn recip(&self) -> Self {
        self.compose_unary_with(|u| {
            let r = 1.0 / u;
            let r2 = r * r;
            [r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r]
        })
    }
    /// `self^a` for real `a`. Caller guarantees a positive base.
    fn powf(&self, a: f64) -> Self {
        self.compose_unary_with(move |u| {
            [
                u.powf(a),
                a * u.powf(a - 1.0),
                a * (a - 1.0) * u.powf(a - 2.0),
                a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0),
                a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0),
            ]
        })
    }
    /// `ln Γ(self)`. Caller guarantees a positive argument.
    fn ln_gamma(&self) -> Self {
        self.compose_unary_with(ln_gamma_derivative_stack)
    }
    /// `ψ(self)` (digamma). Caller guarantees a positive argument.
    fn digamma(&self) -> Self {
        self.compose_unary_with(digamma_derivative_stack)
    }
}

/// Blanket: every scalar [`crate::jet_scalar::JetScalar<K>`] is a [`RowJet<K>`]
/// with `Value = f64`. Each op delegates to the identical `JetScalar` method, so
/// the existing scalar call sites compile UNCHANGED and bit-identically — the
/// bridge adds the lane representation without churning the scalar path. (The
/// concrete lane impls below cannot overlap this: [`Tower3Lane`] / [`Tower4Lane`]
/// are local types that do not implement `JetScalar`, and the orphan rule forbids
/// any downstream impl, so the coherence checker proves the impls disjoint.)
impl<const K: usize, S: crate::jet_scalar::JetScalar<K>> RowJet<K> for S {
    type Value = f64;
    #[inline]
    fn constant(c: f64) -> Self {
        <S as crate::jet_scalar::JetScalar<K>>::constant(c)
    }
    #[inline]
    fn variable(x: f64, slot: usize) -> Self {
        <S as crate::jet_scalar::JetScalar<K>>::variable(x, slot)
    }
    #[inline]
    fn values(&self) -> f64 {
        crate::jet_scalar::JetScalar::value(self)
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        crate::jet_scalar::JetScalar::add(self, o)
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        crate::jet_scalar::JetScalar::sub(self, o)
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        crate::jet_scalar::JetScalar::mul(self, o)
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        crate::jet_scalar::JetScalar::scale(self, s)
    }
    #[inline]
    fn neg(&self) -> Self {
        crate::jet_scalar::JetScalar::neg(self)
    }
    #[inline]
    fn compose_unary_with<const N: usize>(&self, stack_fn: impl Fn(f64) -> [f64; N]) -> Self {
        crate::jet_scalar::JetScalar::compose_unary_with(self, |u| resize_stack::<N, 5>(stack_fn(u)))
    }
    #[inline]
    fn guard(&self, pred: impl Fn(f64) -> bool) -> GuardVerdict {
        GuardVerdict::scalar(pred(crate::jet_scalar::JetScalar::value(self)))
    }
    #[inline]
    fn scale_rows(&self, s: f64) -> Self {
        // `Value == f64`, so per-lane scale is exactly `scale` — the rewrite
        // `.scale(x)` → `.scale_rows(x)` is bit-identical on the scalar path.
        crate::jet_scalar::JetScalar::scale(self, s)
    }
    #[inline]
    fn pack_rows(rows: &[usize], value_of: impl Fn(usize) -> f64) -> f64 {
        value_of(rows[0])
    }
}

/// The `f64x4` lane [`Tower4Lane`] is a [`RowJet<K>`] with `Value = [f64; 4]`,
/// routing each op through its existing per-lane method. Lane `i` of a batched
/// evaluation is `to_bits`-identical to the scalar [`Tower4`] evaluation on row
/// `i` (the per-lane methods are term-for-term lifts of the scalar tower).
impl<const K: usize> RowJet<K> for Tower4Lane<wide::f64x4, K> {
    type Value = [f64; 4];
    #[inline]
    fn constant(c: f64) -> Self {
        Tower4Lane::constant(<wide::f64x4 as crate::jet_scalar::Lane>::splat(c))
    }
    #[inline]
    fn variable(x: f64, slot: usize) -> Self {
        Tower4Lane::variable(<wide::f64x4 as crate::jet_scalar::Lane>::splat(x), slot)
    }
    #[inline]
    fn values(&self) -> [f64; 4] {
        self.v.to_array()
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        Tower4Lane::add(self, o)
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        Tower4Lane::sub(self, o)
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        Tower4Lane::mul(self, o)
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        Tower4Lane::scale(self, s)
    }
    #[inline]
    fn compose_unary_with<const N: usize>(&self, stack_fn: impl Fn(f64) -> [f64; N]) -> Self {
        Tower4Lane::compose_unary_with(self, |u| resize_stack::<N, 5>(stack_fn(u)))
    }
    #[inline]
    fn guard(&self, pred: impl Fn(f64) -> bool) -> GuardVerdict {
        let vals = self.v.to_array();
        let mut mask = 0u8;
        for (i, &v) in vals.iter().enumerate() {
            if !pred(v) {
                mask |= 1 << i;
            }
        }
        GuardVerdict::lanes4(mask)
    }
    #[inline]
    fn scale_rows(&self, s: [f64; 4]) -> Self {
        // True per-lane scale: lane `i` of every channel is multiplied by `s[i]`,
        // so lane `i` matches the scalar `Tower4::scale(s[i])` on row `i`.
        let sl = wide::f64x4::new(s);
        let mut out = *self;
        out.v = self.v * sl;
        for i in 0..K {
            out.g[i] = self.g[i] * sl;
            for j in 0..K {
                out.h[i][j] = self.h[i][j] * sl;
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k] * sl;
                    for l in 0..K {
                        out.t4[i][j][k][l] = self.t4[i][j][k][l] * sl;
                    }
                }
            }
        }
        out
    }
    #[inline]
    fn pack_rows(rows: &[usize], value_of: impl Fn(usize) -> f64) -> [f64; 4] {
        [value_of(rows[0]), value_of(rows[1]), value_of(rows[2]), value_of(rows[3])]
    }
}

/// The `f64x4` lane [`Tower3Lane`] is a [`RowJet<K>`] with `Value = [f64; 4]`,
/// the order-≤3 sibling of the [`Tower4Lane`] impl. A body that uses `N == 5`
/// stacks drops the (unused) fourth-derivative entry here, matching the scalar
/// [`Tower3`] which also carries only up to the third tensor.
impl<const K: usize> RowJet<K> for Tower3Lane<wide::f64x4, K> {
    type Value = [f64; 4];
    #[inline]
    fn constant(c: f64) -> Self {
        Tower3Lane::constant(<wide::f64x4 as crate::jet_scalar::Lane>::splat(c))
    }
    #[inline]
    fn variable(x: f64, slot: usize) -> Self {
        Tower3Lane::variable(<wide::f64x4 as crate::jet_scalar::Lane>::splat(x), slot)
    }
    #[inline]
    fn values(&self) -> [f64; 4] {
        self.v.to_array()
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        Tower3Lane::add(self, o)
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        Tower3Lane::sub(self, o)
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        Tower3Lane::mul(self, o)
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        Tower3Lane::scale(self, s)
    }
    #[inline]
    fn compose_unary_with<const N: usize>(&self, stack_fn: impl Fn(f64) -> [f64; N]) -> Self {
        Tower3Lane::compose_unary_with(self, |u| resize_stack::<N, 4>(stack_fn(u)))
    }
    #[inline]
    fn guard(&self, pred: impl Fn(f64) -> bool) -> GuardVerdict {
        let vals = self.v.to_array();
        let mut mask = 0u8;
        for (i, &v) in vals.iter().enumerate() {
            if !pred(v) {
                mask |= 1 << i;
            }
        }
        GuardVerdict::lanes4(mask)
    }
    #[inline]
    fn scale_rows(&self, s: [f64; 4]) -> Self {
        let sl = wide::f64x4::new(s);
        let mut out = *self;
        out.v = self.v * sl;
        for i in 0..K {
            out.g[i] = self.g[i] * sl;
            for j in 0..K {
                out.h[i][j] = self.h[i][j] * sl;
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k] * sl;
                }
            }
        }
        out
    }
    #[inline]
    fn pack_rows(rows: &[usize], value_of: impl Fn(usize) -> f64) -> [f64; 4] {
        [value_of(rows[0]), value_of(rows[1]), value_of(rows[2]), value_of(rows[3])]
    }
}

/// A family's row negative log-likelihood written ONCE over the [`RowJet`]
/// bridge, so the SAME body instantiates at the scalar jets (for the `(v, g, H)`
/// and contracted-tensor channels) AND at the `f64x4` lane towers (for the
/// 4-rows-per-pass SIMD batch). This is the lane-capable successor to
/// [`RowNllProgramGeneric`]: a body written here gets the scalar channels through
/// [`rowjet_row_kernel`] / [`rowjet_third_contracted`] / [`rowjet_fourth_contracted`]
/// and the batched channels through [`generic_batched_fourth_tower`] /
/// [`generic_batched_third_tower`], all from a single source.
pub trait RowNllProgramRowJet<const K: usize>: Send + Sync {
    /// Number of observations the program covers.
    fn n_rows(&self) -> usize;

    /// Current primary-scalar values for `row` (where to seed each lane).
    fn primaries(&self, row: usize) -> Result<[f64; K], String>;

    /// The row NLL evaluated on the [`RowJet`] bridge. `rows` is the lane→row map
    /// (length 1 for a scalar instantiation, length 4 for a batch); `p[a]` arrives
    /// pre-seeded by the caller (base value plus, for the directional scalars, the
    /// nilpotent contraction directions). The body uses ONLY [`RowJet`] ops and
    /// per-row data entering through `rows`/`self` as constants.
    fn row_nll<R: RowJet<K>>(&self, rows: &[usize], p: &[R; K]) -> Result<R, String>;
}

/// Evaluate a [`RowNllProgramRowJet`] at the value/gradient/Hessian scalar
/// [`crate::jet_scalar::Order2`] (the `(v, g, H)` inner-Newton channel) — the
/// `RowJet` twin of [`generic_row_kernel`].
pub fn rowjet_row_kernel<const K: usize, P: RowNllProgramRowJet<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<(f64, [f64; K], [[f64; K]; K]), String> {
    let base = prog.primaries(row)?;
    let vars: [crate::jet_scalar::Order2<K>; K] =
        std::array::from_fn(|a| <crate::jet_scalar::Order2<K> as RowJet<K>>::variable(base[a], a));
    let s = prog.row_nll(&[row], &vars)?;
    Ok((crate::jet_scalar::JetScalar::value(&s), s.g(), s.h()))
}

/// Evaluate a [`RowNllProgramRowJet`] at the one-seed scalar
/// [`crate::jet_scalar::OneSeed`], returning the contracted third
/// `Σ_c ℓ_{abc} dir_c` — the `RowJet` twin of [`generic_third_contracted`].
pub fn rowjet_third_contracted<const K: usize, P: RowNllProgramRowJet<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    let base = prog.primaries(row)?;
    let vars: [crate::jet_scalar::OneSeed<K>; K] =
        std::array::from_fn(|a| crate::jet_scalar::OneSeed::seed_direction(base[a], a, dir[a]));
    let s = prog.row_nll(&[row], &vars)?;
    Ok(s.contracted_third())
}

/// Evaluate a [`RowNllProgramRowJet`] at the two-seed scalar
/// [`crate::jet_scalar::TwoSeed`], returning the contracted fourth
/// `Σ_{cd} ℓ_{abcd} u_c v_d` — the `RowJet` twin of [`generic_fourth_contracted`].
pub fn rowjet_fourth_contracted<const K: usize, P: RowNllProgramRowJet<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir_u: &[f64; K],
    dir_v: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    let base = prog.primaries(row)?;
    let vars: [crate::jet_scalar::TwoSeed<K>; K] =
        std::array::from_fn(|a| crate::jet_scalar::TwoSeed::seed(base[a], a, dir_u[a], dir_v[a]));
    let s = prog.row_nll(&[row], &vars)?;
    Ok(s.contracted_fourth())
}

/// Evaluate a [`RowNllProgramRowJet`] at the `f64x4` lane [`Tower4Batch`],
/// computing the FULL `(v, g, H, t3, t4)` for FOUR rows in one SIMD pass — the
/// lane twin of [`generic_full_tower`]. Each of the four lanes is seeded with its
/// own row's primaries, so [`Tower4Batch::lane`]`(i)` is `to_bits`-identical to
/// the scalar [`generic_full_tower`] on `rows[i]`.
pub fn generic_batched_fourth_tower<const K: usize, P: RowNllProgramRowJet<K> + ?Sized>(
    prog: &P,
    rows: [usize; 4],
) -> Result<Tower4Batch<K>, String> {
    let bases: [[f64; K]; 4] = [
        prog.primaries(rows[0])?,
        prog.primaries(rows[1])?,
        prog.primaries(rows[2])?,
        prog.primaries(rows[3])?,
    ];
    let vars: [Tower4Batch<K>; K] = std::array::from_fn(|a| {
        let lane_vals = wide::f64x4::new([bases[0][a], bases[1][a], bases[2][a], bases[3][a]]);
        Tower4Batch::variable(lane_vals, a)
    });
    prog.row_nll(&rows, &vars)
}

/// Evaluate a [`RowNllProgramRowJet`] at the `f64x4` lane [`Tower3Batch`],
/// computing `(v, g, H, t3)` for FOUR rows in one SIMD pass — the order-≤3 lane
/// twin of [`generic_full_tower`]. [`Tower3Batch::lane`]`(i)` is
/// `to_bits`-identical to the order-≤3 scalar evaluation on `rows[i]`.
pub fn generic_batched_third_tower<const K: usize, P: RowNllProgramRowJet<K> + ?Sized>(
    prog: &P,
    rows: [usize; 4],
) -> Result<Tower3Batch<K>, String> {
    let bases: [[f64; K]; 4] = [
        prog.primaries(rows[0])?,
        prog.primaries(rows[1])?,
        prog.primaries(rows[2])?,
        prog.primaries(rows[3])?,
    ];
    let vars: [Tower3Batch<K>; K] = std::array::from_fn(|a| {
        let lane_vals = wide::f64x4::new([bases[0][a], bases[1][a], bases[2][a], bases[3][a]]);
        Tower3Batch::variable(lane_vals, a)
    });
    prog.row_nll(&rows, &vars)
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

// ===========================================================================
// SIMD row-batched towers (#1151 follow-up): Tower3Lane / Tower4Lane
// ===========================================================================
//
// `Tower{3,4}Lane<L: Lane, K>` re-type every channel of `Tower{3,4}<K>` from a
// scalar `f64` to a SIMD lane field `L`. With `L = wide::f64x4` one instance
// carries FOUR rows at once, so a per-row kernel (BMS `row_nll`, survival
// `row_kernel`, `marginal_slope` `build_row_*_towers`) can evaluate 4 rows per
// vector pass instead of one per scalar pass.
//
// Every floating-point op is a DIRECT, term-for-term lift of the scalar
// `Tower{3,4}<K>` body — `a * b` -> `a.mul(b)`, `a + b` -> `a.add(b)`, a literal
// `c` -> `L::splat(c)` — in the SAME accumulation order. `wide::f64x4`
// add/sub/mul are lane-wise IEEE-754 ops with NO fused-multiply-add (Rust
// performs no fp-contraction), so lane `i` of any channel of a
// `Tower{3,4}Lane<wide::f64x4, K>` is `to_bits`-IDENTICAL to the scalar
// `Tower{3,4}<K>` channel computed on row `i` — exactly the structural
// bit-identity the existing [`crate::jet_scalar::Order2Lane`] relies on. Proven
// by the in-tree `batch_tests` (real `wide::f64x4`) and a standalone
// f64x4-model oracle, `K ∈ {2,3,4,9}`.
//
// Only the pure-arithmetic ops are lifted (the transcendental `exp`/`ln`/`sqrt`/
// `…` route through scalar libm, which has no `f64x4` form; consumers build the
// per-lane derivative stack scalar-side and feed it to `compose_unary([L; _])`,
// exactly as the scalar path already does).

use crate::jet_scalar::Lane;

/// Lane-batched [`Tower4`]: value / gradient / Hessian / 3rd / 4th tensors
/// carried in a SIMD field `L`. `Tower4Lane<f64x4, K>` lane `i` is
/// `to_bits`-identical to [`Tower4<K>`] on row `i`.
#[derive(Clone, Copy)]
pub struct Tower4Lane<L: Lane, const K: usize> {
    /// Value channel (one entry per lane/row).
    pub v: L,
    /// Gradient `∂/∂p_a`.
    pub g: [L; K],
    /// Hessian `∂²/∂p_a∂p_b`.
    pub h: [[L; K]; K],
    /// Third tensor `∂³`.
    pub t3: [[[L; K]; K]; K],
    /// Fourth tensor `∂⁴`.
    pub t4: [[[[L; K]; K]; K]; K],
}

/// The 4-rows-per-pass batched [`Tower4`] (`wide::f64x4` lanes).
pub type Tower4Batch<const K: usize> = Tower4Lane<wide::f64x4, K>;

impl<L: Lane, const K: usize> Tower4Lane<L, K> {
    /// All-zero tower (every channel `+0.0` in every lane).
    #[inline]
    pub fn zero() -> Self {
        let z = L::splat(0.0);
        Self { v: z, g: [z; K], h: [[z; K]; K], t3: [[[z; K]; K]; K], t4: [[[[z; K]; K]; K]; K] }
    }
    /// Constant `c` (per lane): value channel only.
    #[inline]
    pub fn constant(c: L) -> Self {
        let mut o = Self::zero();
        o.v = c;
        o
    }
    /// Seeded variable `p_idx` at per-lane `value`: unit first derivative in
    /// slot `idx` (mirrors [`Tower4::variable`]).
    #[inline]
    pub fn variable(value: L, idx: usize) -> Self {
        let mut o = Self::constant(value);
        o.g[idx] = L::splat(1.0);
        o
    }
    /// Extract lane `i` as a scalar [`Tower4<K>`] (channel-for-channel).
    #[inline]
    pub fn lane(&self, i: usize) -> Tower4<K> {
        let mut out = Tower4::<K>::zero();
        out.v = self.v.lane(i);
        for a in 0..K {
            out.g[a] = self.g[a].lane(i);
            for b in 0..K {
                out.h[a][b] = self.h[a][b].lane(i);
                for c in 0..K {
                    out.t3[a][b][c] = self.t3[a][b][c].lane(i);
                    for d in 0..K {
                        out.t4[a][b][c][d] = self.t4[a][b][c][d].lane(i);
                    }
                }
            }
        }
        out
    }
    /// Per-channel lane-wise `self + o` (mirrors `Tower4` `Add`).
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = self.v.add(o.v);
        for i in 0..K {
            out.g[i] = self.g[i].add(o.g[i]);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].add(o.h[i][j]);
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k].add(o.t3[i][j][k]);
                    for l in 0..K {
                        out.t4[i][j][k][l] = self.t4[i][j][k][l].add(o.t4[i][j][k][l]);
                    }
                }
            }
        }
        out
    }
    /// Per-channel lane-wise `self - o` (mirrors `Tower4` `Sub`).
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = self.v.sub(o.v);
        for i in 0..K {
            out.g[i] = self.g[i].sub(o.g[i]);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].sub(o.h[i][j]);
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k].sub(o.t3[i][j][k]);
                    for l in 0..K {
                        out.t4[i][j][k][l] = self.t4[i][j][k][l].sub(o.t4[i][j][k][l]);
                    }
                }
            }
        }
        out
    }
    /// Multiply every channel by the plain scalar `s` (mirrors `Tower4::scale`).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        let sl = L::splat(s);
        let mut out = *self;
        out.v = self.v.mul(sl);
        for i in 0..K {
            out.g[i] = self.g[i].mul(sl);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].mul(sl);
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k].mul(sl);
                    for l in 0..K {
                        out.t4[i][j][k][l] = self.t4[i][j][k][l].mul(sl);
                    }
                }
            }
        }
        out
    }
    /// Leibniz product `self · o`, term-for-term lift of [`Tower4::mul`].
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v.mul(b.v);
        for i in 0..K {
            let mut acc = L::splat(0.0);
            acc = acc.add(a.v.mul(b.g[i]));
            acc = acc.add(a.g[i].mul(b.v));
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = L::splat(0.0);
                acc = acc.add(a.v.mul(b.h[i][j]));
                acc = acc.add(a.g[i].mul(b.g[j]));
                acc = acc.add(a.g[j].mul(b.g[i]));
                acc = acc.add(a.h[i][j].mul(b.v));
                out.h[i][j] = acc;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    let mut acc = L::splat(0.0);
                    acc = acc.add(a.v.mul(b.t3[i][j][k]));
                    acc = acc.add(a.g[i].mul(b.h[j][k]));
                    acc = acc.add(a.g[j].mul(b.h[i][k]));
                    acc = acc.add(a.h[i][j].mul(b.g[k]));
                    acc = acc.add(a.g[k].mul(b.h[i][j]));
                    acc = acc.add(a.h[i][k].mul(b.g[j]));
                    acc = acc.add(a.h[j][k].mul(b.g[i]));
                    acc = acc.add(a.t3[i][j][k].mul(b.v));
                    out.t3[i][j][k] = acc;
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        let mut acc = L::splat(0.0);
                        acc = acc.add(a.v.mul(b.t4[i][j][k][l]));
                        acc = acc.add(a.g[i].mul(b.t3[j][k][l]));
                        acc = acc.add(a.g[j].mul(b.t3[i][k][l]));
                        acc = acc.add(a.h[i][j].mul(b.h[k][l]));
                        acc = acc.add(a.g[k].mul(b.t3[i][j][l]));
                        acc = acc.add(a.h[i][k].mul(b.h[j][l]));
                        acc = acc.add(a.h[j][k].mul(b.h[i][l]));
                        acc = acc.add(a.t3[i][j][k].mul(b.g[l]));
                        acc = acc.add(a.g[l].mul(b.t3[i][j][k]));
                        acc = acc.add(a.h[i][l].mul(b.h[j][k]));
                        acc = acc.add(a.h[j][l].mul(b.h[i][k]));
                        acc = acc.add(a.t3[i][j][l].mul(b.g[k]));
                        acc = acc.add(a.h[k][l].mul(b.h[i][j]));
                        acc = acc.add(a.t3[i][k][l].mul(b.g[j]));
                        acc = acc.add(a.t3[j][k][l].mul(b.g[i]));
                        acc = acc.add(a.t4[i][j][k][l].mul(b.v));
                        out.t4[i][j][k][l] = acc;
                    }
                }
            }
        }
        out
    }
    /// Faà di Bruno composition `f ∘ self`, term-for-term lift of
    /// [`Tower4::compose_unary`]. `d = [f, f′, f″, f‴, f⁗]` packed per lane
    /// (build via [`Lane::unary5`] from the scalar special-function stack).
    #[inline]
    pub fn compose_unary(&self, d: [L; 5]) -> Self {
        let mut out = Self::zero();
        out.v = d[0];
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
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    let mut acc = L::splat(0.0);
                    acc = acc.add(d[1].mul(self.t3[i][j][k]));
                    acc = acc.add(d[2].mul(self.h[i][j]).mul(self.g[k]));
                    acc = acc.add(d[2].mul(self.h[i][k]).mul(self.g[j]));
                    acc = acc.add(d[2].mul(self.g[i]).mul(self.h[j][k]));
                    acc = acc.add(d[3].mul(self.g[i]).mul(self.g[j]).mul(self.g[k]));
                    out.t3[i][j][k] = acc;
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        let mut acc = L::splat(0.0);
                        acc = acc.add(d[1].mul(self.t4[i][j][k][l]));
                        acc = acc.add(d[2].mul(self.t3[i][j][k]).mul(self.g[l]));
                        acc = acc.add(d[2].mul(self.t3[i][j][l]).mul(self.g[k]));
                        acc = acc.add(d[2].mul(self.h[i][j]).mul(self.h[k][l]));
                        acc = acc.add(d[3].mul(self.h[i][j]).mul(self.g[k]).mul(self.g[l]));
                        acc = acc.add(d[2].mul(self.t3[i][k][l]).mul(self.g[j]));
                        acc = acc.add(d[2].mul(self.h[i][k]).mul(self.h[j][l]));
                        acc = acc.add(d[3].mul(self.h[i][k]).mul(self.g[j]).mul(self.g[l]));
                        acc = acc.add(d[2].mul(self.h[i][l]).mul(self.h[j][k]));
                        acc = acc.add(d[2].mul(self.g[i]).mul(self.t3[j][k][l]));
                        acc = acc.add(d[3].mul(self.g[i]).mul(self.h[j][k]).mul(self.g[l]));
                        acc = acc.add(d[3].mul(self.h[i][l]).mul(self.g[j]).mul(self.g[k]));
                        acc = acc.add(d[3].mul(self.g[i]).mul(self.h[j][l]).mul(self.g[k]));
                        acc = acc.add(d[3].mul(self.g[i]).mul(self.g[j]).mul(self.h[k][l]));
                        acc = acc.add(d[4].mul(self.g[i]).mul(self.g[j]).mul(self.g[k]).mul(self.g[l]));
                        out.t4[i][j][k][l] = acc;
                    }
                }
            }
        }
        out
    }
    /// Compose with a unary special-function whose `[f64; 5]` derivative stack is
    /// built from the base value through `stack_fn`, evaluated PER LANE — the
    /// batch arm of the generic-over-[`Lane`](crate::jet_scalar::Lane) compose
    /// seam (the SIMD twin of [`Tower4::compose_unary_with`]).
    ///
    /// Each of the four lanes carries a DISTINCT base value, so the scalar
    /// `stack_fn` is run once per lane at that lane's own value (via
    /// [`Lane::unary_with`]) and the `[f64; 5]` results are packed into `[L; 5]`;
    /// the composition is then the existing per-lane [`Self::compose_unary`].
    /// Because `unary_with` runs the identical scalar closure per lane and
    /// `compose_unary` is a term-for-term lift of the scalar tower, lane `i` of
    /// the result is `to_bits`-identical to `self.lane(i).compose_unary_with(stack_fn)`
    /// — which is exactly what lets a row program written against the scalar
    /// [`Tower4::compose_unary_with`] seam re-instantiate, unchanged, at `f64x4`.
    #[inline]
    pub fn compose_unary_with(&self, stack_fn: impl Fn(f64) -> [f64; 5]) -> Self {
        self.compose_unary(self.v.unary_with(stack_fn))
    }

    /// Single-active-slot fast path, term-for-term lift of
    /// [`Tower4::compose_unary_single_slot`] (only the 5 diagonal channels).
    #[inline]
    pub fn compose_unary_single_slot(&self, d: [L; 5], slot: usize) -> Self {
        let mut out = Self::zero();
        let s = slot;
        let g = self.g[s];
        let h = self.h[s][s];
        let t3 = self.t3[s][s][s];
        let t4 = self.t4[s][s][s][s];
        out.v = d[0];
        out.g[s] = {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(g));
            acc
        };
        out.h[s][s] = {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(h));
            acc = acc.add(d[2].mul(g).mul(g));
            acc
        };
        out.t3[s][s][s] = {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(t3));
            acc = acc.add(d[2].mul(h).mul(g));
            acc = acc.add(d[2].mul(h).mul(g));
            acc = acc.add(d[2].mul(g).mul(h));
            acc = acc.add(d[3].mul(g).mul(g).mul(g));
            acc
        };
        out.t4[s][s][s][s] = {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(t4));
            acc = acc.add(d[2].mul(t3).mul(g));
            acc = acc.add(d[2].mul(t3).mul(g));
            acc = acc.add(d[2].mul(h).mul(h));
            acc = acc.add(d[3].mul(h).mul(g).mul(g));
            acc = acc.add(d[2].mul(t3).mul(g));
            acc = acc.add(d[2].mul(h).mul(h));
            acc = acc.add(d[3].mul(h).mul(g).mul(g));
            acc = acc.add(d[2].mul(h).mul(h));
            acc = acc.add(d[2].mul(g).mul(t3));
            acc = acc.add(d[3].mul(g).mul(h).mul(g));
            acc = acc.add(d[3].mul(h).mul(g).mul(g));
            acc = acc.add(d[3].mul(g).mul(h).mul(g));
            acc = acc.add(d[3].mul(g).mul(g).mul(h));
            acc = acc.add(d[4].mul(g).mul(g).mul(g).mul(g));
            acc
        };
        out
    }
    /// Contract `t3` with a primary-space direction (lift of
    /// [`Tower4::third_contracted`]).
    #[inline]
    pub fn third_contracted(&self, dir: &[L; K]) -> [[L; K]; K] {
        let mut out = [[L::splat(0.0); K]; K];
        for a in 0..K {
            for b in 0..K {
                let mut acc = L::splat(0.0);
                for c in 0..K {
                    acc = acc.add(self.t3[a][b][c].mul(dir[c]));
                }
                out[a][b] = acc;
            }
        }
        out
    }
    /// Contract `t4` with two primary-space directions (lift of
    /// [`Tower4::fourth_contracted`]).
    #[inline]
    pub fn fourth_contracted(&self, u: &[L; K], w: &[L; K]) -> [[L; K]; K] {
        let mut out = [[L::splat(0.0); K]; K];
        for i in 0..K {
            for j in 0..K {
                let mut acc = L::splat(0.0);
                for k in 0..K {
                    for l in 0..K {
                        acc = acc.add(self.t4[i][j][k][l].mul(u[k]).mul(w[l]));
                    }
                }
                out[i][j] = acc;
            }
        }
        out
    }
}

/// Lane-batched [`Tower3`] (order-≤3 sibling of [`Tower4Lane`]).
#[derive(Clone, Copy)]
pub struct Tower3Lane<L: Lane, const K: usize> {
    /// Value channel.
    pub v: L,
    /// Gradient.
    pub g: [L; K],
    /// Hessian.
    pub h: [[L; K]; K],
    /// Third tensor.
    pub t3: [[[L; K]; K]; K],
}

/// The 4-rows-per-pass batched [`Tower3`] (`wide::f64x4` lanes).
pub type Tower3Batch<const K: usize> = Tower3Lane<wide::f64x4, K>;

impl<L: Lane, const K: usize> Tower3Lane<L, K> {
    /// All-zero tower.
    #[inline]
    pub fn zero() -> Self {
        let z = L::splat(0.0);
        Self { v: z, g: [z; K], h: [[z; K]; K], t3: [[[z; K]; K]; K] }
    }
    /// Constant `c` (per lane).
    #[inline]
    pub fn constant(c: L) -> Self {
        let mut o = Self::zero();
        o.v = c;
        o
    }
    /// Seeded variable `p_idx` at per-lane `value`.
    #[inline]
    pub fn variable(value: L, idx: usize) -> Self {
        let mut o = Self::constant(value);
        o.g[idx] = L::splat(1.0);
        o
    }
    /// Extract lane `i` as a scalar [`Tower3<K>`].
    #[inline]
    pub fn lane(&self, i: usize) -> Tower3<K> {
        let mut out = Tower3::<K>::zero();
        out.v = self.v.lane(i);
        for a in 0..K {
            out.g[a] = self.g[a].lane(i);
            for b in 0..K {
                out.h[a][b] = self.h[a][b].lane(i);
                for c in 0..K {
                    out.t3[a][b][c] = self.t3[a][b][c].lane(i);
                }
            }
        }
        out
    }
    /// Per-channel lane-wise `self + o`.
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = self.v.add(o.v);
        for i in 0..K {
            out.g[i] = self.g[i].add(o.g[i]);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].add(o.h[i][j]);
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k].add(o.t3[i][j][k]);
                }
            }
        }
        out
    }
    /// Per-channel lane-wise `self - o`.
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = self.v.sub(o.v);
        for i in 0..K {
            out.g[i] = self.g[i].sub(o.g[i]);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].sub(o.h[i][j]);
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k].sub(o.t3[i][j][k]);
                }
            }
        }
        out
    }
    /// Multiply every channel by the plain scalar `s` (mirrors `Tower3::scale`).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        let sl = L::splat(s);
        let mut out = *self;
        out.v = self.v.mul(sl);
        for i in 0..K {
            out.g[i] = self.g[i].mul(sl);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].mul(sl);
                for k in 0..K {
                    out.t3[i][j][k] = self.t3[i][j][k].mul(sl);
                }
            }
        }
        out
    }
    /// Leibniz product `self · o`, term-for-term lift of [`Tower3::mul`].
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v.mul(b.v);
        for i in 0..K {
            let mut acc = L::splat(0.0);
            acc = acc.add(a.v.mul(b.g[i]));
            acc = acc.add(a.g[i].mul(b.v));
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = L::splat(0.0);
                acc = acc.add(a.v.mul(b.h[i][j]));
                acc = acc.add(a.g[i].mul(b.g[j]));
                acc = acc.add(a.g[j].mul(b.g[i]));
                acc = acc.add(a.h[i][j].mul(b.v));
                out.h[i][j] = acc;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    let mut acc = L::splat(0.0);
                    acc = acc.add(a.v.mul(b.t3[i][j][k]));
                    acc = acc.add(a.g[i].mul(b.h[j][k]));
                    acc = acc.add(a.g[j].mul(b.h[i][k]));
                    acc = acc.add(a.h[i][j].mul(b.g[k]));
                    acc = acc.add(a.g[k].mul(b.h[i][j]));
                    acc = acc.add(a.h[i][k].mul(b.g[j]));
                    acc = acc.add(a.h[j][k].mul(b.g[i]));
                    acc = acc.add(a.t3[i][j][k].mul(b.v));
                    out.t3[i][j][k] = acc;
                }
            }
        }
        out
    }
    /// Faà di Bruno composition `f ∘ self`, term-for-term lift of
    /// [`Tower3::compose_unary`]. `d = [f, f′, f″, f‴]` packed per lane.
    #[inline]
    pub fn compose_unary(&self, d: [L; 4]) -> Self {
        let mut out = Self::zero();
        out.v = d[0];
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
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    let mut acc = L::splat(0.0);
                    acc = acc.add(d[1].mul(self.t3[i][j][k]));
                    acc = acc.add(d[2].mul(self.h[i][j]).mul(self.g[k]));
                    acc = acc.add(d[2].mul(self.h[i][k]).mul(self.g[j]));
                    acc = acc.add(d[2].mul(self.g[i]).mul(self.h[j][k]));
                    acc = acc.add(d[3].mul(self.g[i]).mul(self.g[j]).mul(self.g[k]));
                    out.t3[i][j][k] = acc;
                }
            }
        }
        out
    }
    /// Compose with a unary special-function whose `[f64; 4]` derivative stack is
    /// built from the base value through `stack_fn`, evaluated PER LANE — the
    /// batch arm of the generic-over-[`Lane`](crate::jet_scalar::Lane) compose
    /// seam (the SIMD twin of [`Tower3::compose_unary_with`], order-≤3 sibling of
    /// [`Tower4Lane::compose_unary_with`]). The scalar `stack_fn` is run once per
    /// lane at that lane's own base value (via [`Lane::unary_with`]) and packed
    /// into `[L; 4]` for the existing per-lane [`Self::compose_unary`], so lane
    /// `i` is `to_bits`-identical to `self.lane(i).compose_unary_with(stack_fn)`.
    #[inline]
    pub fn compose_unary_with(&self, stack_fn: impl Fn(f64) -> [f64; 4]) -> Self {
        self.compose_unary(self.v.unary_with(stack_fn))
    }

    /// Single-active-slot fast path, term-for-term lift of
    /// [`Tower3::compose_unary_single_slot`].
    #[inline]
    pub fn compose_unary_single_slot(&self, d: [L; 4], slot: usize) -> Self {
        let mut out = Self::zero();
        let s = slot;
        let g = self.g[s];
        let h = self.h[s][s];
        let t3 = self.t3[s][s][s];
        out.v = d[0];
        out.g[s] = {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(g));
            acc
        };
        out.h[s][s] = {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(h));
            acc = acc.add(d[2].mul(g).mul(g));
            acc
        };
        out.t3[s][s][s] = {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(t3));
            acc = acc.add(d[2].mul(h).mul(g));
            acc = acc.add(d[2].mul(h).mul(g));
            acc = acc.add(d[2].mul(g).mul(h));
            acc = acc.add(d[3].mul(g).mul(g).mul(g));
            acc
        };
        out
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
        }
    }

    // Fill every channel of a scalar Tower4<K> with random data.
    fn rand_t4<const K: usize>(r: &mut Rng) -> Tower4<K> {
        let mut t = Tower4::<K>::zero();
        t.v = r.f();
        for i in 0..K {
            t.g[i] = r.f();
            for j in 0..K {
                t.h[i][j] = r.f();
                for k in 0..K {
                    t.t3[i][j][k] = r.f();
                    for l in 0..K {
                        t.t4[i][j][k][l] = r.f();
                    }
                }
            }
        }
        t
    }
    fn rand_t3<const K: usize>(r: &mut Rng) -> Tower3<K> {
        let mut t = Tower3::<K>::zero();
        t.v = r.f();
        for i in 0..K {
            t.g[i] = r.f();
            for j in 0..K {
                t.h[i][j] = r.f();
                for k in 0..K {
                    t.t3[i][j][k] = r.f();
                }
            }
        }
        t
    }
    fn pack4_t4<const K: usize>(rows: &[Tower4<K>; 4]) -> Tower4Batch<K> {
        let mut b = Tower4Batch::<K>::zero();
        let lane = |f: &dyn Fn(&Tower4<K>) -> f64| {
            wide::f64x4::new([f(&rows[0]), f(&rows[1]), f(&rows[2]), f(&rows[3])])
        };
        b.v = lane(&|t| t.v);
        for i in 0..K {
            b.g[i] = lane(&|t| t.g[i]);
            for j in 0..K {
                b.h[i][j] = lane(&|t| t.h[i][j]);
                for k in 0..K {
                    b.t3[i][j][k] = lane(&|t| t.t3[i][j][k]);
                    for l in 0..K {
                        b.t4[i][j][k][l] = lane(&|t| t.t4[i][j][k][l]);
                    }
                }
            }
        }
        b
    }
    fn pack4_t3<const K: usize>(rows: &[Tower3<K>; 4]) -> Tower3Batch<K> {
        let mut b = Tower3Batch::<K>::zero();
        let lane = |f: &dyn Fn(&Tower3<K>) -> f64| {
            wide::f64x4::new([f(&rows[0]), f(&rows[1]), f(&rows[2]), f(&rows[3])])
        };
        b.v = lane(&|t| t.v);
        for i in 0..K {
            b.g[i] = lane(&|t| t.g[i]);
            for j in 0..K {
                b.h[i][j] = lane(&|t| t.h[i][j]);
                for k in 0..K {
                    b.t3[i][j][k] = lane(&|t| t.t3[i][j][k]);
                }
            }
        }
        b
    }
    fn assert_t4_eq<const K: usize>(b: &Tower4<K>, s: &Tower4<K>, ctx: &str) {
        assert_eq!(b.v.to_bits(), s.v.to_bits(), "v {ctx}");
        for i in 0..K {
            assert_eq!(b.g[i].to_bits(), s.g[i].to_bits(), "g {ctx}");
            for j in 0..K {
                assert_eq!(b.h[i][j].to_bits(), s.h[i][j].to_bits(), "h {ctx}");
                for k in 0..K {
                    assert_eq!(b.t3[i][j][k].to_bits(), s.t3[i][j][k].to_bits(), "t3 {ctx}");
                    for l in 0..K {
                        assert_eq!(b.t4[i][j][k][l].to_bits(), s.t4[i][j][k][l].to_bits(), "t4 {ctx}");
                    }
                }
            }
        }
    }
    fn assert_t3_eq<const K: usize>(b: &Tower3<K>, s: &Tower3<K>, ctx: &str) {
        assert_eq!(b.v.to_bits(), s.v.to_bits(), "v {ctx}");
        for i in 0..K {
            assert_eq!(b.g[i].to_bits(), s.g[i].to_bits(), "g {ctx}");
            for j in 0..K {
                assert_eq!(b.h[i][j].to_bits(), s.h[i][j].to_bits(), "h {ctx}");
                for k in 0..K {
                    assert_eq!(b.t3[i][j][k].to_bits(), s.t3[i][j][k].to_bits(), "t3 {ctx}");
                }
            }
        }
    }

    // Run a representative op chain on 4 scalar rows and on the f64x4 batch,
    // then assert every channel of every lane is to_bits-identical.
    fn run4<const K: usize>(seed: u64, batches: usize) -> usize {
        let mut r = Rng(seed);
        let mut rows_checked = 0;
        for _ in 0..batches {
            let a: [Tower4<K>; 4] = std::array::from_fn(|_| rand_t4::<K>(&mut r));
            let b: [Tower4<K>; 4] = std::array::from_fn(|_| rand_t4::<K>(&mut r));
            let d: [[f64; 5]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| r.f()));
            let dir: [[f64; K]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| r.f()));
            let dir2: [[f64; K]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| r.f()));
            let s = r.f();

            // scalar per-row reference
            let scal: [Tower4<K>; 4] = std::array::from_fn(|rw| {
                let prod = a[rw].mul(&b[rw]);
                let comp = prod.compose_unary(d[rw]);
                let summed = comp.add(&a[rw]).sub(&b[rw]).scale(s);
                summed.compose_unary_single_slot(d[rw], 0)
            });
            let third: [[[f64; K]; K]; 4] =
                std::array::from_fn(|rw| a[rw].third_contracted(&dir[rw]));
            let fourth: [[[f64; K]; K]; 4] =
                std::array::from_fn(|rw| a[rw].fourth_contracted(&dir[rw], &dir2[rw]));

            // batched f64x4
            let ab = pack4_t4(&a);
            let bb = pack4_t4(&b);
            let db: [wide::f64x4; 5] = std::array::from_fn(|c| {
                wide::f64x4::new([d[0][c], d[1][c], d[2][c], d[3][c]])
            });
            let dirb: [wide::f64x4; K] = std::array::from_fn(|c| {
                wide::f64x4::new([dir[0][c], dir[1][c], dir[2][c], dir[3][c]])
            });
            let dir2b: [wide::f64x4; K] = std::array::from_fn(|c| {
                wide::f64x4::new([dir2[0][c], dir2[1][c], dir2[2][c], dir2[3][c]])
            });
            let prodb = ab.mul(&bb);
            let compb = prodb.compose_unary(db);
            let summedb = compb.add(&ab).sub(&bb).scale(s);
            let finalb = summedb.compose_unary_single_slot(db, 0);
            let thirdb = ab.third_contracted(&dirb);
            let fourthb = ab.fourth_contracted(&dirb, &dir2b);

            for rw in 0..4 {
                assert_t4_eq(&finalb.lane(rw), &scal[rw], "t4-chain");
                for i in 0..K {
                    for j in 0..K {
                        assert_eq!(thirdb[i][j].lane(rw).to_bits(), third[rw][i][j].to_bits(), "third");
                        assert_eq!(fourthb[i][j].lane(rw).to_bits(), fourth[rw][i][j].to_bits(), "fourth");
                    }
                }
                rows_checked += 1;
            }
        }
        rows_checked
    }
    fn run3<const K: usize>(seed: u64, batches: usize) -> usize {
        let mut r = Rng(seed);
        let mut rows_checked = 0;
        for _ in 0..batches {
            let a: [Tower3<K>; 4] = std::array::from_fn(|_| rand_t3::<K>(&mut r));
            let b: [Tower3<K>; 4] = std::array::from_fn(|_| rand_t3::<K>(&mut r));
            let d: [[f64; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| r.f()));
            let s = r.f();
            let scal: [Tower3<K>; 4] = std::array::from_fn(|rw| {
                let prod = a[rw].mul(&b[rw]);
                let comp = prod.compose_unary(d[rw]);
                let summed = comp.add(&a[rw]).sub(&b[rw]).scale(s);
                summed.compose_unary_single_slot(d[rw], 0)
            });
            let ab = pack4_t3(&a);
            let bb = pack4_t3(&b);
            let db: [wide::f64x4; 4] = std::array::from_fn(|c| {
                wide::f64x4::new([d[0][c], d[1][c], d[2][c], d[3][c]])
            });
            let prodb = ab.mul(&bb);
            let compb = prodb.compose_unary(db);
            let summedb = compb.add(&ab).sub(&bb).scale(s);
            let finalb = summedb.compose_unary_single_slot(db, 0);
            for rw in 0..4 {
                assert_t3_eq(&finalb.lane(rw), &scal[rw], "t3-chain");
                rows_checked += 1;
            }
        }
        rows_checked
    }

    // A `Tower4Batch<9>` carries a `9⁴ = 6561`-entry `t4` tensor in 4-wide
    // lanes (≈210 KiB by value); the op chain keeps several live, which can
    // exceed a test thread's default stack. Run each width on a large-stack
    // thread so K=9 is exercised without a stack overflow.
    fn big_stack<R: Send + 'static, F: FnOnce() -> R + Send + 'static>(f: F) -> R {
        std::thread::Builder::new()
            .stack_size(512 << 20)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap()
    }

    #[test]
    fn tower4_batch_lane_bit_identical() {
        let batches = 2000;
        let rows_checked = big_stack(move || run4::<2>(0x1111_2222_3333_4444, batches))
            + big_stack(move || run4::<3>(0x5555_6666_7777_8888, batches))
            + big_stack(move || run4::<4>(0x9999_aaaa_bbbb_cccc, batches))
            + big_stack(move || run4::<9>(0xdddd_eeee_ffff_0000, batches));
        // 4 widths × `batches` batches × 4 rows each: guards the large-stack
        // worker threads against silently running zero comparisons.
        assert_eq!(rows_checked, 4 * batches * 4);
    }

    #[test]
    fn tower3_batch_lane_bit_identical() {
        let batches = 2000;
        let rows_checked = big_stack(move || run3::<2>(0x0f0f_1e1e_2d2d_3c3c, batches))
            + big_stack(move || run3::<3>(0x4b4b_5a5a_6969_7878, batches))
            + big_stack(move || run3::<4>(0x8787_9696_a5a5_b4b4, batches))
            + big_stack(move || run3::<9>(0xc3c3_d2d2_e1e1_f0f0, batches));
        // 4 widths × `batches` batches × 4 rows each: guards the large-stack
        // worker threads against silently running zero comparisons.
        assert_eq!(rows_checked, 4 * batches * 4);
    }

    // ── compose_unary_with seam (generic-over-Lane compose) ─────────────────
    //
    // The seam lets a single-sourced row program build its special-function
    // STACK from the base value through a closure, so the SAME expression
    // instantiates at a scalar tower (one base) AND a batch tower (four distinct
    // per-lane bases). These oracles pin both arms `to_bits`.

    /// A base-value-dependent `[f64; 5]` derivative stack (finite for finite `u`),
    /// standing in for a family's hand-certified special-function stack. `stack4`
    /// is its order-≤3 truncation.
    fn seam_stack5(u: f64) -> [f64; 5] {
        [u.sin(), u.cos(), (2.0 * u).sin(), (0.5 * u).cos(), u * u - 0.3]
    }
    fn seam_stack4(u: f64) -> [f64; 4] {
        let s = seam_stack5(u);
        [s[0], s[1], s[2], s[3]]
    }

    /// Force a distinct / edge per-lane base value (signed zeros included).
    fn seam_edge_base(r: &mut Rng, which: usize) -> f64 {
        match which {
            0 => -0.0,
            1 => 0.0,
            2 => r.f(),
            _ => r.f() + 3.0,
        }
    }

    /// (a) scalar arm: `Tower4::compose_unary_with(f)` is `to_bits`-identical to
    /// the explicit `compose_unary(f(value))` on every channel.
    fn scalar_seam_t4<const K: usize>(seed: u64, n: usize) -> usize {
        let mut r = Rng(seed);
        for _ in 0..n {
            let mut t = rand_t4::<K>(&mut r);
            t.v = seam_edge_base(&mut r, (t.v.to_bits() % 4) as usize);
            assert_t4_eq(
                &t.compose_unary_with(seam_stack5),
                &t.compose_unary(seam_stack5(t.v)),
                "scalar t4 seam",
            );
        }
        n
    }
    fn scalar_seam_t3<const K: usize>(seed: u64, n: usize) -> usize {
        let mut r = Rng(seed);
        for _ in 0..n {
            let mut t = rand_t3::<K>(&mut r);
            t.v = seam_edge_base(&mut r, (t.v.to_bits() % 4) as usize);
            assert_t3_eq(
                &t.compose_unary_with(seam_stack4),
                &t.compose_unary(seam_stack4(t.v)),
                "scalar t3 seam",
            );
        }
        n
    }

    /// (b) lane arm: `Tower4Lane::compose_unary_with` lane `i` is
    /// `to_bits`-identical to the scalar `Tower4::compose_unary_with` on row `i`,
    /// with the four lanes carrying DISTINCT base values (signed zeros included),
    /// so a buggy impl reusing one lane's base would fail.
    fn lane_seam_t4<const K: usize>(seed: u64, batches: usize) -> usize {
        let mut r = Rng(seed);
        let mut verified = 0usize;
        for _ in 0..batches {
            let mut rows: [Tower4<K>; 4] = std::array::from_fn(|_| rand_t4::<K>(&mut r));
            for (rw, row) in rows.iter_mut().enumerate() {
                row.v = seam_edge_base(&mut r, rw);
            }
            let batch_out = pack4_t4(&rows).compose_unary_with(seam_stack5);
            for (rw, row) in rows.iter().enumerate() {
                assert_t4_eq(&batch_out.lane(rw), &row.compose_unary_with(seam_stack5), "lane t4 seam");
                verified += 1;
            }
        }
        verified
    }
    fn lane_seam_t3<const K: usize>(seed: u64, batches: usize) -> usize {
        let mut r = Rng(seed);
        let mut verified = 0usize;
        for _ in 0..batches {
            let mut rows: [Tower3<K>; 4] = std::array::from_fn(|_| rand_t3::<K>(&mut r));
            for (rw, row) in rows.iter_mut().enumerate() {
                row.v = seam_edge_base(&mut r, rw);
            }
            let batch_out = pack4_t3(&rows).compose_unary_with(seam_stack4);
            for (rw, row) in rows.iter().enumerate() {
                assert_t3_eq(&batch_out.lane(rw), &row.compose_unary_with(seam_stack4), "lane t3 seam");
                verified += 1;
            }
        }
        verified
    }

    #[test]
    fn compose_unary_with_scalar_bit_identical() {
        let n = 1100;
        let total = scalar_seam_t4::<2>(0x2200_0001, n)
            + scalar_seam_t4::<3>(0x2200_0002, n)
            + scalar_seam_t4::<4>(0x2200_0003, n)
            + big_stack(move || scalar_seam_t4::<9>(0x2200_0004, n))
            + scalar_seam_t3::<2>(0x3300_0001, n)
            + scalar_seam_t3::<3>(0x3300_0002, n)
            + scalar_seam_t3::<4>(0x3300_0003, n)
            + big_stack(move || scalar_seam_t3::<9>(0x3300_0004, n));
        // 8 arms × 1100 = 8800 ≥ 4000 inputs.
        assert_eq!(total, 8 * n);
    }

    #[test]
    fn compose_unary_with_lane_matches_scalar() {
        let b = 600;
        let total = lane_seam_t4::<2>(0x4400_0001, b)
            + lane_seam_t4::<3>(0x4400_0002, b)
            + lane_seam_t4::<4>(0x4400_0003, b)
            + big_stack(move || lane_seam_t4::<9>(0x4400_0004, b))
            + lane_seam_t3::<2>(0x5500_0001, b)
            + lane_seam_t3::<3>(0x5500_0002, b)
            + lane_seam_t3::<4>(0x5500_0003, b)
            + big_stack(move || lane_seam_t3::<9>(0x5500_0004, b));
        // 8 arms × 600 = 4800 batches ≥ 2000; each verifies 4 lanes (19200 checks).
        assert_eq!(total, 8 * b * 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Tower3<K>` must be bit-identical to `Tower4<K>` on every channel it
    /// carries (value, gradient, Hessian, third derivatives). The order-≤3
    /// Leibniz / Faà-di-Bruno terms read only order-≤3 inner channels, so
    /// dropping the fourth tensor cannot perturb them. Exercises products
    /// (Leibniz cross-terms), unary composition, scaling, and addition — the
    /// same operations the survival location-scale `nll_index_tower` composes —
    /// across all mixed partials, not just the diagonal entries that kernel reads.
    #[test]
    fn tower3_matches_tower4_through_third_order() {
        let s_a: [f64; 5] = [
            0.3_f64.sin(),
            0.3_f64.cos(),
            -0.3_f64.sin(),
            -0.3_f64.cos(),
            0.3_f64.sin(),
        ];
        let s_b: [f64; 5] = [1.1, -0.4, 0.8, -0.2, 0.05];
        let s4 = |s: [f64; 5]| [s[0], s[1], s[2], s[3]];

        let a4 = Tower4::<3>::variable(0.4, 0);
        let b4 = Tower4::<3>::variable(-0.7, 1);
        let c4 = Tower4::<3>::variable(0.9, 2);
        let prog4 = (a4.mul(&b4) + c4).compose_unary(s_a).scale(1.3)
            + a4.mul(&c4).scale(-0.7)
            + b4.compose_unary(s_b).scale(0.25);

        let a3 = Tower3::<3>::variable(0.4, 0);
        let b3 = Tower3::<3>::variable(-0.7, 1);
        let c3 = Tower3::<3>::variable(0.9, 2);
        let prog3 = (a3.mul(&b3) + c3).compose_unary(s4(s_a)).scale(1.3)
            + a3.mul(&c3).scale(-0.7)
            + b3.compose_unary(s4(s_b)).scale(0.25);

        assert_eq!(prog3.v.to_bits(), prog4.v.to_bits(), "value mismatch");
        for i in 0..3 {
            assert_eq!(
                prog3.g[i].to_bits(),
                prog4.g[i].to_bits(),
                "g[{i}] mismatch"
            );
            for j in 0..3 {
                assert_eq!(
                    prog3.h[i][j].to_bits(),
                    prog4.h[i][j].to_bits(),
                    "h[{i}][{j}] mismatch"
                );
                for k in 0..3 {
                    assert_eq!(
                        prog3.t3[i][j][k].to_bits(),
                        prog4.t3[i][j][k].to_bits(),
                        "t3[{i}][{j}][{k}] mismatch"
                    );
                }
            }
        }
    }

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

#[inline]
fn erfcx_nonnegative(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < 26.0 {
        ((x * x).min(700.0)).exp() * statrs::function::erf::erfc(x)
    } else {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let poly = 1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2 - 1.875 * inv2 * inv2 * inv2
            + 6.5625 * inv2 * inv2 * inv2 * inv2;
        inv * poly / std::f64::consts::PI.sqrt()
    }
}

#[inline]
fn log1mexp_positive(a: f64) -> f64 {
    assert!(a >= 0.0, "log1mexp_positive requires a >= 0: a={a}");
    if a > core::f64::consts::LN_2 {
        (-(-a).exp()).ln_1p()
    } else if a > 0.0 {
        (-(-a).exp_m1()).ln()
    } else {
        f64::NEG_INFINITY
    }
}

#[inline]
fn signed_probit_logcdf_and_mills_ratio(x: f64) -> (f64, f64) {
    if x == f64::INFINITY {
        return (0.0, 0.0);
    }
    if x == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        let ex = erfcx_nonnegative(u).max(1e-300);
        let log_cdf = -u * u + (0.5 * ex).ln();
        let lambda = (2.0 / std::f64::consts::PI).sqrt() / ex;
        (log_cdf, lambda)
    } else {
        let cdf = crate::probability::normal_cdf(x).clamp(1e-300, 1.0);
        let lambda = crate::probability::normal_pdf(x) / cdf;
        (cdf.ln(), lambda)
    }
}

/// Stable derivative stack for `log Phi(x)` through fourth order.
#[inline]
pub fn unary_derivatives_normal_logcdf(x: f64) -> [f64; 5] {
    let (log_cdf, lambda) = signed_probit_logcdf_and_mills_ratio(x);
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
pub fn unary_derivatives_log1mexp_positive(x: f64) -> [f64; 5] {
    let r = 1.0 / x.exp_m1();
    [
        log1mexp_positive(x),
        r,
        -r * (1.0 + r),
        r * (1.0 + r) * (1.0 + 2.0 * r),
        -r * (1.0 + r) * (1.0 + 6.0 * r + 6.0 * r * r),
    ]
}
// ── The RowJet bridge oracle (CI) ─────────────────────────────────────
#[cfg(test)]
mod rowjet_bridge_tests {
    use super::*;
    use crate::jet_scalar::{JetScalar, Order2};

    /// A toy row-NLL written ONCE over the [`RowJet`] bridge: a product, a sum, a
    /// subtraction, a scale/neg, a constant, and two value-distinct
    /// `compose_unary_with` stacks (an exp stack and a smooth finite-everywhere
    /// stack), plus a domain `guard`. The body is generic over `R: RowJet<2>`, so
    /// the SAME source instantiates at the scalar jets and the `f64x4` lane towers.
    struct ToyProgram {
        primaries: Vec<[f64; 2]>,
        /// Per-row CONTINUOUS auxiliary data `[cov, z, wi]` — the survival
        /// `covariance_ones` / `z_sum` / observation-weight analogues that enter
        /// the jet algebra as `.scale_rows(per_row_value)`, distinct per lane.
        aux: Vec<[f64; 3]>,
    }

    impl ToyProgram {
        /// The body uses `pack_rows` to gather the per-lane continuous data from
        /// the lane→row map and `scale_rows` to fold it in — so a 4-row batch
        /// carries four DISTINCT cov/z/wi, which the single-`f64` `scale` could not.
        fn body<R: RowJet<2>>(&self, rows: &[usize], p: &[R; 2]) -> R {
            let cov = R::pack_rows(rows, |r| self.aux[r][0]);
            let z = R::pack_rows(rows, |r| self.aux[r][1]);
            let wi = R::pack_rows(rows, |r| self.aux[r][2]);

            let a = p[0].mul(&p[1]).scale_rows(cov);
            let b = a.add(&R::constant(0.5)).sub(&p[0].scale(0.25));
            let c = b
                .compose_unary_with(|u| {
                    let e = u.exp();
                    [e, e, e, e, e]
                })
                .scale_rows(z);
            let d = c.neg().add(&p[0]);
            let e = d
                .compose_unary_with(|u| {
                    let s = (1.0 + u * u).sqrt();
                    let s3 = s * s * s;
                    let s5 = s3 * s * s;
                    let s7 = s5 * s * s;
                    [s, u / s, 1.0 / s3, -3.0 * u / s5, (12.0 * u * u - 3.0) / s7]
                })
                .scale_rows(wi);
            e.mul(&p[1]).add(&e)
        }
    }

    impl RowNllProgramRowJet<2> for ToyProgram {
        fn n_rows(&self) -> usize {
            self.primaries.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
            Ok(self.primaries[row])
        }
        fn row_nll<R: RowJet<2>>(&self, rows: &[usize], p: &[R; 2]) -> Result<R, String> {
            assert!(rows.len() == 1 || rows.len() == 4, "lane→row map is 1 or 4 wide");
            Ok(self.body(rows, p))
        }
    }

    fn assert_t4_bits_eq(a: &Tower4<2>, b: &Tower4<2>, ctx: &str) {
        assert_eq!(a.v.to_bits(), b.v.to_bits(), "{ctx}: v");
        for i in 0..2 {
            assert_eq!(a.g[i].to_bits(), b.g[i].to_bits(), "{ctx}: g[{i}]");
            for j in 0..2 {
                assert_eq!(a.h[i][j].to_bits(), b.h[i][j].to_bits(), "{ctx}: h[{i}][{j}]");
                for k in 0..2 {
                    assert_eq!(
                        a.t3[i][j][k].to_bits(),
                        b.t3[i][j][k].to_bits(),
                        "{ctx}: t3[{i}][{j}][{k}]"
                    );
                    for l in 0..2 {
                        assert_eq!(
                            a.t4[i][j][k][l].to_bits(),
                            b.t4[i][j][k][l].to_bits(),
                            "{ctx}: t4[{i}][{j}][{k}][{l}]"
                        );
                    }
                }
            }
        }
    }

    fn assert_t3_bits_eq(a: &Tower3<2>, b: &Tower3<2>, ctx: &str) {
        assert_eq!(a.v.to_bits(), b.v.to_bits(), "{ctx}: v");
        for i in 0..2 {
            assert_eq!(a.g[i].to_bits(), b.g[i].to_bits(), "{ctx}: g[{i}]");
            for j in 0..2 {
                assert_eq!(a.h[i][j].to_bits(), b.h[i][j].to_bits(), "{ctx}: h[{i}][{j}]");
                for k in 0..2 {
                    assert_eq!(
                        a.t3[i][j][k].to_bits(),
                        b.t3[i][j][k].to_bits(),
                        "{ctx}: t3[{i}][{j}][{k}]"
                    );
                }
            }
        }
    }

    // Deterministic LCG with signed-zero injection and per-lane-distinct values.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn val(&mut self) -> f64 {
            let u = self.next();
            if u < 0.04 {
                return 0.0;
            }
            if u < 0.08 {
                return -0.0;
            }
            (self.next() - 0.5) * 5.0
        }
    }

    /// Lane `i` of the batched order-4 / order-3 tower is `to_bits`-identical to
    /// the scalar tower on row `i`, for ≥2000 distinct 4-row batches with
    /// signed-zero and per-lane-distinct primaries.
    #[test]
    fn batched_lane_i_matches_scalar_row_i_bit_identical() {
        let mut rng = Lcg(0xA5A5_1234_DEAD_BEEF);
        let mut batches = 0usize;
        for _ in 0..2500 {
            let bases: [[f64; 2]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| rng.val()));
            // per-lane-DISTINCT continuous aux (cov/z/wi), signed-zero injected.
            let aux: [[f64; 3]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| rng.val()));
            let prog = ToyProgram { primaries: bases.to_vec(), aux: aux.to_vec() };
            let rows = [0usize, 1, 2, 3];

            // order-4 batch vs scalar Tower4 (instantiated through the same body).
            let batch4 = generic_batched_fourth_tower(&prog, rows).expect("batch4");
            for (row, base) in bases.iter().enumerate() {
                let vars: [Tower4<2>; 2] =
                    std::array::from_fn(|a| <Tower4<2> as RowJet<2>>::variable(base[a], a));
                let scal = prog.row_nll(&[row], &vars).expect("scalar tower4");
                assert_t4_bits_eq(&batch4.lane(row), &scal, "batched_fourth");
            }

            // order-3 batch vs scalar Tower3.
            let batch3 = generic_batched_third_tower(&prog, rows).expect("batch3");
            for (row, base) in bases.iter().enumerate() {
                let vars: [Tower3<2>; 2] =
                    std::array::from_fn(|a| <Tower3<2> as RowJet<2>>::variable(base[a], a));
                let scal = prog.row_nll(&[row], &vars).expect("scalar tower3");
                assert_t3_bits_eq(&batch3.lane(row), &scal, "batched_third");
            }
            batches += 1;
        }
        assert_eq!(batches, 2500);
    }

    /// The blanket impl does not churn the scalar path: the body driven through
    /// `RowJet` ops is `to_bits`-identical to the body driven directly through
    /// `JetScalar` ops, and `rowjet_row_kernel`'s `(v, g, H)` matches the dense
    /// `Tower4` lower channels.
    #[test]
    fn blanket_scalar_path_is_unchanged_and_consistent() {
        let mut rng = Lcg(0x0BAD_F00D_1357_2468);
        for _ in 0..3000 {
            let base: [f64; 2] = std::array::from_fn(|_| rng.val());
            let aux0: [f64; 3] = std::array::from_fn(|_| rng.val());
            let prog = ToyProgram { primaries: vec![base], aux: vec![aux0] };

            // (a) RowJet-driven body == JetScalar-driven body, bit-for-bit. The
            // reference body uses `scale(f64)` where the RowJet body uses
            // `scale_rows(f64)` — proving the scalar `scale_rows` rewrite does not
            // churn the path (`scale_rows(s) == scale(s)` on `Value = f64`).
            let via_rowjet: Tower4<2> = {
                let vars: [Tower4<2>; 2] =
                    std::array::from_fn(|a| <Tower4<2> as RowJet<2>>::variable(base[a], a));
                prog.row_nll(&[0], &vars).expect("rowjet")
            };
            let via_jetscalar: Tower4<2> = {
                let vars: [Tower4<2>; 2] = std::array::from_fn(|a| {
                    <Tower4<2> as JetScalar<2>>::variable(base[a], a)
                });
                let (cov, z, wi) = (aux0[0], aux0[1], aux0[2]);
                // The body using JetScalar's own ops + scale(f64) directly.
                let a = vars[0].mul(&vars[1]).scale(cov);
                let b = a.add(&Tower4::constant(0.5)).sub(&vars[0].scale(0.25));
                let c = b
                    .compose_unary_with(|u| {
                        let e = u.exp();
                        [e, e, e, e, e]
                    })
                    .scale(z);
                let d = JetScalar::neg(&c).add(&vars[0]);
                let e = d
                    .compose_unary_with(|u| {
                        let s = (1.0 + u * u).sqrt();
                        let s3 = s * s * s;
                        let s5 = s3 * s * s;
                        let s7 = s5 * s * s;
                        [s, u / s, 1.0 / s3, -3.0 * u / s5, (12.0 * u * u - 3.0) / s7]
                    })
                    .scale(wi);
                e.mul(&vars[1]).add(&e)
            };
            assert_t4_bits_eq(&via_rowjet, &via_jetscalar, "blanket_vs_direct");

            // (b) rowjet_row_kernel (v,g,H) == dense Tower4 lower channels.
            // Order2 and Tower4 use different internal representations so
            // signed-zero differences (−0.0 vs +0.0) may arise in gradient/
            // Hessian channels that evaluate to exactly zero; IEEE equality
            // treats these as equal, so `==` is the right comparison here.
            let (v, g, h) = rowjet_row_kernel(&prog, 0).expect("kernel");
            assert_eq!(v.to_bits(), via_rowjet.v.to_bits(), "kernel v");
            for i in 0..2 {
                assert!(g[i] == via_rowjet.g[i], "kernel g[{i}]: {} vs {}", g[i], via_rowjet.g[i]);
                for j in 0..2 {
                    assert!(
                        h[i][j] == via_rowjet.h[i][j],
                        "kernel h[{i}][{j}]: {} vs {}",
                        h[i][j],
                        via_rowjet.h[i][j]
                    );
                }
            }

            // (c) the Order2 scalar IS a RowJet via the blanket.
            let o2: [Order2<2>; 2] =
                std::array::from_fn(|a| <Order2<2> as RowJet<2>>::variable(base[a], a));
            let _ = prog.body(&[0], &o2);
        }
    }

    /// On the scalar path (`Value = f64`) `scale_rows(s)` is `to_bits`-identical
    /// to `scale(s)` for EVERY channel — so rewriting a survival `.scale(per_row)`
    /// to `.scale_rows(per_row)` cannot perturb the existing scalar fits.
    #[test]
    fn scale_rows_scalar_is_bit_identical_to_scale() {
        let mut rng = Lcg(0xFEED_FACE_0042_1001);
        for _ in 0..3000 {
            let base: [f64; 2] = std::array::from_fn(|_| rng.val());
            let s = rng.val();
            // Build a dense tower with populated channels (exp of a product).
            let vars: [Tower4<2>; 2] =
                std::array::from_fn(|a| <Tower4<2> as RowJet<2>>::variable(base[a], a));
            let jet = vars[0].mul(&vars[1]).compose_unary_with(|u| {
                let e = u.exp();
                [e, e, e, e, e]
            });
            let via_scale = RowJet::scale(&jet, s);
            let via_scale_rows = RowJet::scale_rows(&jet, s);
            assert_t4_bits_eq(&via_scale_rows, &via_scale, "scale_rows==scale");
        }
    }

    /// `scale_rows` on a batch multiplies lane `i` by `s[i]`, so lane `i` of a
    /// per-lane-scaled batch matches the scalar `scale(s[i])` on row `i` — the
    /// continuous per-row data path the single-`f64` `scale` could not carry.
    #[test]
    fn batched_scale_rows_matches_per_row_scalar_scale() {
        let mut rng = Lcg(0x1357_9BDF_2468_ACE0);
        for _ in 0..2500 {
            let bases: [[f64; 2]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| rng.val()));
            let s: [f64; 4] = std::array::from_fn(|_| rng.val());
            let batch: [Tower4Batch<2>; 2] = std::array::from_fn(|a| {
                Tower4Batch::variable(
                    wide::f64x4::new([bases[0][a], bases[1][a], bases[2][a], bases[3][a]]),
                    a,
                )
            });
            let prod = batch[0].mul(&batch[1]).compose_unary_with(|u| {
                let e = u.exp();
                [e, e, e, e, e]
            });
            let scaled = prod.scale_rows(s);
            for (row, base) in bases.iter().enumerate() {
                let v: [Tower4<2>; 2] =
                    std::array::from_fn(|a| <Tower4<2> as RowJet<2>>::variable(base[a], a));
                let prod_s = v[0].mul(&v[1]).compose_unary_with(|u| {
                    let e = u.exp();
                    [e, e, e, e, e]
                });
                let ref_s = RowJet::scale(&prod_s, s[row]);
                assert_t4_bits_eq(&scaled.lane(row), &ref_s, "batched_scale_rows");
            }
        }
    }

    /// The per-lane guard reports exactly the failing lanes on a batch and the
    /// single lane on a scalar jet.
    #[test]
    fn guard_reports_per_lane_failures() {
        let cols: [[f64; 2]; 4] = [[1.0, 0.5], [-2.0, 0.5], [3.0, 0.5], [-0.0, 0.5]];
        let vars: [Tower4Batch<2>; 2] = std::array::from_fn(|a| {
            Tower4Batch::variable(
                wide::f64x4::new([cols[0][a], cols[1][a], cols[2][a], cols[3][a]]),
                a,
            )
        });
        let verdict = vars[0].guard(|v| v > 0.0);
        assert_eq!(verdict.lanes(), 4);
        assert!(verdict.any_failed());
        assert!(!verdict.all_pass());
        assert!(!verdict.lane_failed(0));
        assert!(verdict.lane_failed(1));
        assert!(!verdict.lane_failed(2));
        assert!(verdict.lane_failed(3));
        assert_eq!(verdict.failed_mask(), 0b1010);

        let s_ok = <Tower4<2> as RowJet<2>>::variable(1.0, 0);
        let s_bad = <Tower4<2> as RowJet<2>>::variable(-1.0, 0);
        assert!(RowJet::guard(&s_ok, |v| v > 0.0).all_pass());
        assert!(RowJet::guard(&s_bad, |v| v > 0.0).any_failed());
        assert_eq!(RowJet::guard(&s_ok, |v| v > 0.0).lanes(), 1);
    }
}
