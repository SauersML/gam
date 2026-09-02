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
//! single pass, every channel the `gam_models::row_kernel::RowKernel` trait
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
//!    asserting channel-by-channel agreement with a [`RowProgram`] written
//!    once — see [`verify_kernel_channels`]. This alone would have caught
//!    #736 at introduction.
//! 3. Derive every channel through [`program_row_kernel`],
//!    [`program_third_contracted`], [`program_fourth_contracted`], or
//!    [`program_full_tower`], selecting only the representation its consumer
//!    needs while retaining one expression.
//! 4. New families (#914/#916/#917 ZI/ordinal/expectile, #921's location-
//!    scale port) implement ONLY [`RowProgram`] and get an exact fourth-order
//!    tower for the price of writing the likelihood.

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
    /// expression instead of the shared `jet_algebra::leibniz_product` subset
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
        // Hessian is symmetric under i↔j; compute the upper triangle and mirror
        // (see [`Tower2::mul`] — same term order, enforces exact symmetry).
        for i in 0..K {
            for j in i..K {
                // subsets of {i,j}: {} {i} {j} {ij}
                let mut acc = 0.0;
                acc += a.v * b.h[i][j];
                acc += a.g[i] * b.g[j];
                acc += a.g[j] * b.g[i];
                acc += a.h[i][j] * b.v;
                out.h[i][j] = acc;
                out.h[j][i] = acc;
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
/// It exists purely for performance: an inner Newton step and a value-only
/// outer-objective probe need at most curvature, never the outer-κ/ψ
/// third/fourth derivatives. Evaluating a row likelihood over
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

    /// Exact truncated (order ≤ 2) Leibniz product. The `v`/`g`/`h` upper
    /// triangle matches [`Tower4::mul`] term-for-term.
    ///
    /// # Symmetry fast path
    ///
    /// The order-≤2 Leibniz Hessian
    /// `h[i][j] = a.v·b.h[i][j] + a.g[i]·b.g[j] + a.g[j]·b.g[i] + a.h[i][j]·b.v`
    /// is symmetric under `i ↔ j` whenever the operand Hessians are — which they
    /// always are: `constant`/`variable` seed a symmetric (zero) `h`, and
    /// `mul`/`compose_unary`/`add`/`scale` each preserve symmetry, so the
    /// invariant holds for every tower a row program can build. We therefore
    /// compute only the upper triangle `j ≥ i` and mirror it into the lower
    /// triangle. At the `K = 9` survival width that is `K(K+1)/2 = 45` four-product
    /// entry evaluations instead of `K² = 81`, and the win is larger in wall-clock
    /// because the `648`-entry `h` spills at `K = 9` — halving the expensive
    /// stores/reloads roughly halves the kernel (measured ≈2× on a `Tower2<9>`
    /// mul-and-read throughput microbench; the dominant `mul` under every packed
    /// scalar bottoms out here).
    ///
    /// The upper-triangle entries are BIT-IDENTICAL to the old rectangular form
    /// (same term/accumulation order). The lower triangle now equals its mirror
    /// exactly, where the rectangular form rounded `h[i][j]` and `h[j][i]`
    /// independently (the two cross products accumulate in opposite order) and
    /// left a ≤1-ulp asymmetry; mirroring removes it, so the result is exactly
    /// symmetric — strictly closer to the true symmetric Hessian, not merely a
    /// reordering. Dense-`h` consumers are all tolerance-gated (rel-tol ≥ 1e-11 ≫
    /// 1e-16); the `f64`/`f64x4` lane oracle stays exact because
    /// [`crate::jet_scalar::Order2Lane::mul`] mirrors term-for-term.
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v * b.v;
        for i in 0..K {
            out.g[i] = a.v * b.g[i] + a.g[i] * b.v;
        }
        for i in 0..K {
            for j in i..K {
                let hij = a.v * b.h[i][j] + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.h[i][j] * b.v;
                out.h[i][j] = hij;
                out.h[j][i] = hij;
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
    /// `jet_algebra::leibniz_product` walker — the order-≤3 sibling of
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
        // Hessian is symmetric under i↔j; upper triangle + mirror (see Tower2::mul).
        for i in 0..K {
            for j in i..K {
                let mut acc = 0.0;
                acc += a.v * b.h[i][j];
                acc += a.g[i] * b.g[j];
                acc += a.g[j] * b.g[i];
                acc += a.h[i][j] * b.v;
                out.h[i][j] = acc;
                out.h[j][i] = acc;
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

    /// Compose with a unary special-function whose `[f64; 4]` derivative stack is
    /// built from the base value through `stack_fn`. Evaluates `stack_fn(self.v)`
    /// once and forwards to [`Self::compose_unary`], so it is bit-identical to the
    /// explicit form. The order-≤3 sibling of [`Tower4::compose_unary_with`].
    #[inline]
    pub fn compose_unary_with(&self, stack_fn: impl Fn(f64) -> [f64; 4]) -> Self {
        self.compose_unary(stack_fn(self.v))
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

/// Scalar digamma ψ(x) for x>0. Bit-identical to `digamma_derivative_stack(x)[0]`
/// and to `ln_gamma_derivative_stack(x)[1]`, but evaluates ONLY ψ — the four
/// higher polygammas those `[f64; 5]` stacks build are pure discarded work at a
/// scalar consumer that reads a single element. Hot-path row kernels that need
/// only the digamma value (e.g. the GAMLSS Beta observed cross weight) call this
/// instead of indexing `[0]` off a full derivative stack.
#[inline]
pub fn digamma(x: f64) -> f64 {
    digamma_positive(x)
}

/// Scalar trigamma ψ′(x) for x>0. Bit-identical to
/// `trigamma_derivative_stack(x)[0]` (both bottom out in `polygamma_positive(1,
/// x)`), but evaluates ONLY ψ′ — the four higher polygammas (orders 2–5) the
/// `[f64; 5]` stack builds are discarded at a `[0]` consumer. Used by the
/// dispersion-channel Fisher-information row kernels (NB2 `ψ′(θ)−ψ′(θ+μ)`, Beta
/// `μψ′(μφ)−(1−μ)ψ′((1−μ)φ)`) which read the trigamma value alone.
#[inline]
pub fn trigamma(x: f64) -> f64 {
    polygamma_positive(1, x)
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

// ── The program seam ─────────────────────────────────────────────────

// ── The canonical single-source seam (#932 consolidation) ────────────
//
// `RowProgram<K>` is the ONE row-program interface #932 converges every family
// onto. Its generic `eval<S: JetScalar<K>>` body is the go-forward derivation
// surface for every calculus channel; `program_*` selects only the derivative
// representation each consumer needs.

/// The single source of truth #932 asks for: a family's row negative
/// log-likelihood written ONCE over the generic [`crate::jet_scalar::JetScalar`]
/// interface, from which every `RowKernel` (gam-models) derivative channel is
/// mechanically derived. A family implements ONLY this (plus its linear Jacobian
/// wiring, which is family data, not calculus) — it cannot author an independent
/// derivative tower, because there is no other channel to author.
///
/// Because a body uses only `add`/`sub`/`mul`/`scale`/`exp`/`ln`/… — all provided
/// by [`crate::jet_scalar::JetScalar`] — the SAME body re-instantiates at
/// [`crate::jet_scalar::Order2`] (value/grad/Hessian), [`crate::jet_scalar::OneSeed`]
/// (contracted third), [`crate::jet_scalar::TwoSeed`] (contracted fourth), and the
/// full [`Tower4`] (every channel), with the contraction folded into the
/// differentiation so no dense `t3`/`t4` is ever materialised.
pub trait RowProgram<const K: usize>: Send + Sync {
    /// Number of observations the program covers.
    fn n_rows(&self) -> usize;

    /// Current primary-scalar values for `row` (where to seed the scalar).
    fn primaries(&self, row: usize) -> Result<[f64; K], String>;

    /// The row NLL evaluated on a generic jet scalar. `p[a]` arrives pre-seeded
    /// (base value + per-scalar nilpotent directions) by the caller; the body
    /// uses ONLY [`crate::jet_scalar::JetScalar`] ops and per-row data (response,
    /// censoring, offsets) entering as constants.
    fn eval<S: crate::jet_scalar::JetScalar<K>>(&self, row: usize, p: &[S; K])
    -> Result<S, String>;
}

/// Maximum size of one canonical dense-jet storage object kept on the call
/// stack. Small fixed-width programs stay allocation-free; wider derivative
/// representations use exact-length heap storage instead of making the thread
/// stack scale as `K * size_of::<S>()`. A full dense result larger than this
/// boundary is rejected in favor of the bounded directional APIs.
///
/// This is a storage-policy boundary, not a calculus fallback: both branches
/// invoke the same [`RowProgram::eval`] expression with the same scalar type.
const PROGRAM_DENSE_JET_STACK_BUDGET_BYTES: usize = 64 * 1024;

#[inline]
fn program_primary_jets_fit_stack<S, const K: usize>() -> bool {
    std::mem::size_of::<S>()
        .checked_mul(K)
        .is_some_and(|bytes| bytes <= PROGRAM_DENSE_JET_STACK_BUDGET_BYTES)
}

fn evaluate_program_with_stack_primaries<const K: usize, P, S>(
    prog: &P,
    row: usize,
    mut seed: impl FnMut(usize) -> S,
) -> Result<S, String>
where
    P: RowProgram<K> + ?Sized,
    S: crate::jet_scalar::JetScalar<K>,
{
    let vars: [S; K] = std::array::from_fn(&mut seed);
    prog.eval(row, &vars)
}

#[inline(never)]
fn evaluate_program_with_heap_primaries<const K: usize, P, S>(
    prog: &P,
    row: usize,
    seed: impl FnMut(usize) -> S,
) -> Result<S, String>
where
    P: RowProgram<K> + ?Sized,
    S: crate::jet_scalar::JetScalar<K>,
{
    // The exact-size range builds precisely K initialized Copy scalars in
    // heap-backed storage. Converting the boxed slice to a boxed array changes
    // only its type; it never materializes `[S; K]` on the stack.
    let vars: Box<[S]> = (0..K).map(seed).collect();
    let vars: Box<[S; K]> = vars.try_into().map_err(|vars: Box<[S]>| {
        format!(
            "canonical row program seeded {} primary jets; expected exactly {K}",
            vars.len()
        )
    })?;
    prog.eval(row, &vars)
}

#[inline]
fn evaluate_program_with_seeded_primaries<const K: usize, P, S>(
    prog: &P,
    row: usize,
    seed: impl FnMut(usize) -> S,
) -> Result<S, String>
where
    P: RowProgram<K> + ?Sized,
    S: crate::jet_scalar::JetScalar<K>,
{
    if program_primary_jets_fit_stack::<S, K>() {
        evaluate_program_with_stack_primaries(prog, row, seed)
    } else {
        evaluate_program_with_heap_primaries(prog, row, seed)
    }
}

/// Derive the `row_kernel` channel `(nll, ∇, H)` from a [`RowProgram`] at the
/// value/gradient/Hessian scalar [`crate::jet_scalar::Order2`], WITHOUT
/// materialising any third / fourth tensor.
pub fn program_row_kernel<const K: usize, P: RowProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<(f64, [f64; K], [[f64; K]; K]), String> {
    let base = prog.primaries(row)?;
    let s = evaluate_program_with_seeded_primaries(prog, row, |a| {
        <crate::jet_scalar::Order2<K> as crate::jet_scalar::JetScalar<K>>::variable(base[a], a)
    })?;
    Ok(s.into_channels())
}

/// Derive the `row_third_contracted(dir)` channel `Σ_c ℓ_{abc} dir_c` from a
/// [`RowProgram`] at the one-seed scalar [`crate::jet_scalar::OneSeed`], WITHOUT
/// materialising the dense `t3`.
pub fn program_third_contracted<const K: usize, P: RowProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    let base = prog.primaries(row)?;
    let s = evaluate_program_with_seeded_primaries(prog, row, |a| {
        crate::jet_scalar::OneSeed::seed_direction(base[a], a, dir[a])
    })?;
    Ok(s.contracted_third())
}

/// Derive the `row_fourth_contracted(u, v)` channel `Σ_{cd} ℓ_{abcd} u_c v_d`
/// from a [`RowProgram`] at the two-seed scalar [`crate::jet_scalar::TwoSeed`],
/// WITHOUT materialising the dense `t4`.
pub fn program_fourth_contracted<const K: usize, P: RowProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir_u: &[f64; K],
    dir_v: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    let base = prog.primaries(row)?;
    let s = evaluate_program_with_seeded_primaries(prog, row, |a| {
        crate::jet_scalar::TwoSeed::seed(base[a], a, dir_u[a], dir_v[a])
    })?;
    Ok(s.contracted_fourth())
}

/// Derive every channel `(v, g, h, t3, t4)` in one pass from a [`RowProgram`] at
/// the full dense [`Tower4`] scalar.
///
/// The result is boxed so the return slot itself remains bounded independently
/// of `K`. Dense towers above the canonical storage budget are rejected before
/// the program is touched; consumers at those widths must request only the
/// channels they need through [`program_row_kernel`],
/// [`program_third_contracted`], and [`program_fourth_contracted`].
pub fn program_full_tower<const K: usize, P: RowProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<Box<Tower4<K>>, String> {
    let tower_bytes = std::mem::size_of::<Tower4<K>>();
    if tower_bytes > PROGRAM_DENSE_JET_STACK_BUDGET_BYTES {
        return Err(format!(
            "canonical dense Tower4<{K}> requires {tower_bytes} bytes, exceeding the {}-byte \
             storage budget; use the bounded row-kernel and directional channel APIs",
            PROGRAM_DENSE_JET_STACK_BUDGET_BYTES
        ));
    }
    let base = prog.primaries(row)?;
    evaluate_program_with_seeded_primaries(prog, row, |a| Tower4::variable(base[a], a))
        .map(Box::new)
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

    impl RowProgram<1> for LogitProgram {
        fn n_rows(&self) -> usize {
            self.eta.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 1], String> {
            Ok([self.eta[row]])
        }
        fn eval<S: crate::jet_scalar::JetScalar<1>>(
            &self,
            row: usize,
            p: &[S; 1],
        ) -> Result<S, String> {
            let eta = p[0];
            Ok(eta
                .exp()
                .add(&S::constant(1.0))
                .ln()
                .sub(&eta.scale(self.y[row])))
        }
    }

    #[test]
    fn logit_tower_matches_closed_forms() {
        let prog = LogitProgram {
            eta: vec![-2.3, -0.4, 0.0, 0.9, 3.1],
            y: vec![1.0, 0.0, 1.0, 0.0, 1.0],
        };
        for row in 0..prog.n_rows() {
            let t = program_full_tower(&prog, row).expect("logit program");
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

    struct OversizedDenseProgram;

    impl RowProgram<10> for OversizedDenseProgram {
        fn n_rows(&self) -> usize {
            1
        }

        fn primaries(&self, row: usize) -> Result<[f64; 10], String> {
            Err(format!(
                "dense-tower storage check reached program primaries at row {row}"
            ))
        }

        fn eval<S: crate::jet_scalar::JetScalar<10>>(
            &self,
            row: usize,
            primaries: &[S; 10],
        ) -> Result<S, String> {
            Err(format!(
                "dense-tower storage check reached program evaluation at row {row} with {} primaries",
                primaries.len()
            ))
        }
    }

    struct LargestBudgetedDenseProgram;

    impl RowProgram<9> for LargestBudgetedDenseProgram {
        fn n_rows(&self) -> usize {
            1
        }

        fn primaries(&self, row: usize) -> Result<[f64; 9], String> {
            if row == 0 {
                Ok([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            } else {
                Err(format!("largest budgeted dense program has no row {row}"))
            }
        }

        fn eval<S: crate::jet_scalar::JetScalar<9>>(
            &self,
            row: usize,
            primaries: &[S; 9],
        ) -> Result<S, String> {
            if row != 0 {
                return Err(format!("largest budgeted dense program has no row {row}"));
            }
            let linear =
                S::linear_combination(primaries, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
            let quartic = primaries[0]
                .mul(&primaries[1])
                .mul(&primaries[2])
                .mul(&primaries[3]);
            Ok(linear.add(&quartic))
        }
    }

    #[test]
    fn full_tower_accepts_largest_width_inside_storage_budget_932() {
        assert_eq!(std::mem::size_of::<Tower4<9>>(), 59_048);
        assert!(
            !program_primary_jets_fit_stack::<Tower4<9>, 9>(),
            "nine full-width primary towers must use exact-length heap storage"
        );

        let tower = program_full_tower(&LargestBudgetedDenseProgram, 0)
            .expect("Tower4<9> must remain inside the canonical dense storage budget");
        assert_eq!(tower.v, 309.0);
        assert_eq!(tower.g, [25.0, 14.0, 11.0, 10.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // `t4` stores derivatives, not Taylor coefficients: the distinct-axis
        // derivative of p0*p1*p2*p3 is 1, with no 4! normalization.
        assert_eq!(tower.t4[0][1][2][3], 1.0);
    }

    #[test]
    fn full_tower_refuses_oversized_result_before_touching_program() {
        let tower_bytes = std::mem::size_of::<Tower4<10>>();
        assert!(tower_bytes > PROGRAM_DENSE_JET_STACK_BUDGET_BYTES);
        assert!(
            std::mem::size_of::<Result<Box<Tower4<32>>, String>>()
                <= 4 * std::mem::size_of::<usize>(),
            "boxed full-tower API must keep its return slot independent of dense tower width"
        );

        let error = program_full_tower(&OversizedDenseProgram, 0)
            .expect_err("Tower4<10> must exceed the canonical dense storage budget");
        assert_eq!(
            error,
            format!(
                "canonical dense Tower4<10> requires {tower_bytes} bytes, exceeding the {}-byte \
                 storage budget; use the bounded row-kernel and directional channel APIs",
                PROGRAM_DENSE_JET_STACK_BUDGET_BYTES
            )
        );
    }

    /// Gaussian location-scale row NLL, K=2 primaries (η, s = log σ):
    /// ℓ = s + ½ e^{−2s} (y − η)². Mixed cross blocks — the #736 fragility
    /// shape — all have one-line closed forms here.
    struct LocScaleProgram {
        eta: Vec<f64>,
        s: Vec<f64>,
        y: Vec<f64>,
    }

    impl RowProgram<2> for LocScaleProgram {
        fn n_rows(&self) -> usize {
            self.eta.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
            Ok([self.eta[row], self.s[row]])
        }
        fn eval<S: crate::jet_scalar::JetScalar<2>>(
            &self,
            row: usize,
            p: &[S; 2],
        ) -> Result<S, String> {
            let r = S::constant(self.y[row]).sub(&p[0]);
            Ok(p[1].add(&p[1].scale(-2.0).exp().mul(&r).mul(&r).scale(0.5)))
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
            let t = program_full_tower(&prog, row).expect("locscale program");
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
            // The canonical trait-surface helpers agree with direct contraction.
            let dir = [0.7, -1.3];
            let third = program_third_contracted(&prog, row, &dir).expect("third");
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

    impl RowProgram<3> for GnarlyProgram {
        fn n_rows(&self) -> usize {
            self.primaries.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
            self.primaries
                .get(row)
                .copied()
                .ok_or_else(|| format!("gnarly: row {row} out of range"))
        }
        fn eval<S: crate::jet_scalar::JetScalar<3>>(
            &self,
            row: usize,
            p: &[S; 3],
        ) -> Result<S, String> {
            let tau = *self
                .tau
                .get(row)
                .ok_or_else(|| format!("gnarly: tau row {row} out of range"))?;
            let a = p[0].mul(&p[1]).exp();
            let b = p[2].mul(&p[2]).add(&S::constant(1.0)).sqrt();
            let c = a.add(&b).add(&S::constant(tau)).ln();
            let d = p[1].scale(0.5).add(&S::constant(2.0)).powf(1.7);
            let delta = p[0].sub(&p[2]);
            Ok(c.mul(&d.recip()).add(&delta.mul(&delta).scale(0.25)))
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
        impl RowProgram<3> for At<'_> {
            fn n_rows(&self) -> usize {
                1
            }
            fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
                if row != 0 {
                    return Err(format!("gnarly-at: row {row} out of range"));
                }
                Ok(self.p)
            }
            fn eval<S: crate::jet_scalar::JetScalar<3>>(
                &self,
                eval_row: usize,
                vars: &[S; 3],
            ) -> Result<S, String> {
                if eval_row != 0 {
                    return Err(format!("gnarly-at: eval row {eval_row} out of range"));
                }
                self.base.eval(self.row, vars)
            }
        }
        *program_full_tower(&At { base: prog, row, p }, 0).expect("gnarly tower")
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
    /// a(θ) with nonzero a_u/a_uv), slot 1 = g (the slope), slot 2 unused.
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
            let t = program_full_tower(&prog, row).expect("gnarly tower");
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

#[cfg(test)]
mod derivative_stack_tests {
    use super::*;
    // ── ln_gamma_derivative_stack / digamma_derivative_stack / trigamma_derivative_stack ──

    #[test]
    fn ln_gamma_derivative_stack_known_values_at_1() {
        let s = ln_gamma_derivative_stack(1.0);
        // ln Γ(1) = 0; statrs uses Lanczos so the result is within ULP noise
        assert!(s[0].abs() < 1e-14, "ln_gamma(1) must be ~0, got {}", s[0]);
        // ψ₀(1) = -γ  (Euler–Mascheroni)
        let euler_mascheroni = 0.577_215_664_901_532_9_f64;
        assert!(
            (s[1] + euler_mascheroni).abs() < 1e-10,
            "digamma(1) ≈ -{euler_mascheroni:.6}, got {}",
            s[1]
        );
        // ψ₁(1) = π²/6
        let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!(
            (s[2] - pi2_6).abs() < 1e-10,
            "trigamma(1) ≈ {pi2_6:.6}, got {}",
            s[2]
        );
    }

    #[test]
    fn ln_gamma_derivative_stack_known_values_at_2() {
        let s = ln_gamma_derivative_stack(2.0);
        // ln Γ(2) = ln(1) = 0 exactly
        assert!(s[0].abs() < 1e-14, "ln_gamma(2) must be 0, got {}", s[0]);
        // ψ₀(2) = 1 − γ (recurrence: ψ₀(x+1) = ψ₀(x) + 1/x)
        let euler_mascheroni = 0.577_215_664_901_532_9_f64;
        let digamma_2 = 1.0 - euler_mascheroni;
        assert!(
            (s[1] - digamma_2).abs() < 1e-10,
            "digamma(2) ≈ {digamma_2:.6}, got {}",
            s[1]
        );
    }

    #[test]
    fn ln_gamma_derivative_stack_order2_is_prefix() {
        for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
            let full = ln_gamma_derivative_stack(x);
            let ord2 = ln_gamma_derivative_stack_order2(x);
            assert_eq!(ord2[0], full[0], "order2[0] != full[0] at x={x}");
            assert_eq!(ord2[1], full[1], "order2[1] != full[1] at x={x}");
            assert_eq!(ord2[2], full[2], "order2[2] != full[2] at x={x}");
        }
    }

    #[test]
    fn digamma_derivative_stack_overlaps_ln_gamma_stack() {
        // The two stacks share a run of four polygamma values:
        // ln_gamma_stack[1..5] == digamma_stack[0..4]
        for &x in &[0.5_f64, 1.0, 2.0, 7.0] {
            let lg = ln_gamma_derivative_stack(x);
            let dg = digamma_derivative_stack(x);
            for i in 0..4 {
                assert_eq!(
                    lg[i + 1],
                    dg[i],
                    "ln_gamma_stack[{}] != digamma_stack[{}] at x={x}",
                    i + 1,
                    i
                );
            }
        }
    }

}

// ── Contraction-symmetry optimization gate ────────────────────────────────────
//
// `Tower4::third_contracted` / `fourth_contracted` contract the (fully
// index-symmetric) `t3`/`t4` tensors against directions, leaving the output
// indices `(a, b)` / `(i, j)` free. Those free indices inherit the tensor's
// symmetry — `out[a][b] == out[b][a]` term-for-term — so only the upper triangle
// need be summed and the lower triangle mirrored. Unlike the dense symmetric
// FILL (which needs a K⁴ scatter and loses inner-loop vectorisation, and was
// measured SLOWER), the mirror here is a tiny K×K copy and the inner contraction
// is untouched (contiguous, vectorisable). This is BIT-IDENTICAL to the full
// nest, so it needs no fingerprint re-baseline; the gate is (1) bit-identity vs
// the full reference and (2) a measured wall-clock that is not slower.
