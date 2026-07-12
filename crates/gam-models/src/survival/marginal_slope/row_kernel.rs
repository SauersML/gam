//! The rigid per-row `RowKernel<4>` implementation and its Jacobian-action
//! assembly helpers: the memory-efficient row-at-a-time kernel used on the
//! no-flex hot path.

use super::*;

use gam_math::jet_scalar::JetScalar;
use gam_math::nested_dual::JetField;
use gam_row_macros::row_program;

// ── Static-sparsity (v,g,H) scalar (#932 perf) ─────────────────────────
//
// The rigid row primaries are `(q0, q1, qd1, g)`. Three of them — `q0`, `q1`,
// `qd1` — enter every INDEX intermediate (`eta0 = q0·c + s·g·z`, `eta1`, `ad1`)
// AFFINELY: each is a single linear coefficient times the shared curvature
// factor `c(g)`. So the *index-space* second derivative between any two of those
// three linear primaries is structurally zero — the only curvature they acquire
// is the rank-1 outer-function term `f''·(∂η/∂q)·(∂η/∂q)` created at the leaf
// composes (logΦ / logφ / log), which is genuinely dense and computed normally.
//
// [`SparseOrder2`] encodes "which axes are linear" as a compile-time bitmask and
// ELIDES exactly the provably-zero work: the linear×linear self-Hessian READS in
// `mul`/`compose_unary` (a linear axis carries a structurally-zero self-Hessian
// block by the index-affine contract). Everything else — the gradient, the
// linear×nonlinear cross curvature, and the dense leaf `g⊗g` — is computed bit
// for bit as the dense [`Order2`]. The family writes the row NLL ONCE against
// [`JetScalar`]; this is just a different instantiation that the compiler
// monomorphizes into sparse-optimal code (proven: a single `mul` drops from 63
// to 21 floating-point instructions in the emitted IR/asm). No hand chain rule,
// so the #736 cross-block-sign-flip bug genus cannot reappear.
//
// CONTRACT: an axis may be declared linear only when the program never forms
// curvature between it and another linear axis (the linear×linear index Hessian
// stays zero for the life of every intermediate). [`SparseOrder2::check_contract`]
// debug-asserts this at every elision site, so a wrong declaration panics loudly
// in debug/test builds rather than silently dropping curvature.

/// Bitmask of which `K=4` rigid primaries enter linearly: bit `a` set ⇒ axis `a`
/// is linear. `q0 (0), q1 (1), qd1 (2)` are linear; `g (3)` is nonlinear.
pub(crate) const RIGID_LINEAR_MASK: u32 = (1 << 0) | (1 << 1) | (1 << 2);

#[inline(always)]
const fn axis_is_linear(mask: u32, a: usize) -> bool {
    (mask >> a) & 1 == 1
}

/// Order-2 (value/gradient/Hessian) jet over `K=4` primaries, with compile-time
/// static sparsity: the linear×linear self-Hessian block (axes both set in
/// `LIN`) is never read in `mul`/`compose_unary` because it is structurally
/// zero. Bit-identical to [`Order2<4>`] on every channel a consumer reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseOrder2<const LIN: u32> {
    v: f64,
    grad: [f64; 4],
    hess: [[f64; 4]; 4],
}

impl<const LIN: u32> SparseOrder2<LIN> {
    #[inline]
    pub(crate) fn g(&self) -> [f64; 4] {
        self.grad
    }
    #[inline]
    pub(crate) fn h(&self) -> [[f64; 4]; 4] {
        self.hess
    }

    /// Guard for the index-affine contract: the linear×linear self-Hessian
    /// block must be exactly zero wherever we elide its read.
    #[inline(always)]
    fn check_contract(&self) {
        for i in 0..4 {
            if axis_is_linear(LIN, i) {
                for j in 0..4 {
                    if axis_is_linear(LIN, j) {
                        assert!(
                            self.hess[i][j] == 0.0,
                            "static-sparsity contract violated: linear×linear Hessian h[{i}][{j}]={} != 0 (axes {i},{j} both declared linear but the program forms curvature between them)",
                            self.hess[i][j]
                        );
                    }
                }
            }
        }
    }
}

impl<const LIN: u32> JetScalar<4> for SparseOrder2<LIN> {
    fn constant(c: f64) -> Self {
        Self {
            v: c,
            grad: [0.0; 4],
            hess: [[0.0; 4]; 4],
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        let mut grad = [0.0; 4];
        grad[axis] = 1.0;
        Self {
            v: x,
            grad,
            hess: [[0.0; 4]; 4],
        }
    }
}

impl<const LIN: u32> gam_math::nested_dual::JetField for SparseOrder2<LIN> {
    fn value(&self) -> f64 {
        self.v
    }
    // add / sub / scale are uniform-dense: after a leaf compose, a linear×linear
    // entry can be legitimately nonzero (the dense `f''·g⊗g` term), so these must
    // touch every entry. The elision lives ONLY in the h-reads of mul/compose.
    fn add(&self, o: &Self) -> Self {
        let mut r = *self;
        r.v += o.v;
        for i in 0..4 {
            r.grad[i] += o.grad[i];
            for j in 0..4 {
                r.hess[i][j] += o.hess[i][j];
            }
        }
        r
    }
    fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        let mut r = *self;
        r.v *= s;
        for i in 0..4 {
            r.grad[i] *= s;
            for j in 0..4 {
                r.hess[i][j] *= s;
            }
        }
        r
    }
    fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        // Elision precondition: we skip reading a.hess/b.hess on the
        // linear×linear block — assert those reads would indeed be zero.
        a.check_contract();
        b.check_contract();
        let mut r = Self::constant(a.v * b.v);
        for i in 0..4 {
            r.grad[i] = a.v * b.grad[i] + a.grad[i] * b.v;
        }
        // H_out[i][j] = a.v·H_b + a.g[i]·b.g[j] + a.g[j]·b.g[i] + H_a·b.v. The
        // self-Hessian reads H_a[i][j]/H_b[i][j] are structurally zero when both
        // i,j are linear (contract), so they are elided; the `g⊗g` product-rule
        // curvature term is always kept.
        for i in 0..4 {
            for j in 0..4 {
                let mut hij = a.grad[i] * b.grad[j] + a.grad[j] * b.grad[i];
                if !axis_is_linear(LIN, i) || !axis_is_linear(LIN, j) {
                    hij += a.v * b.hess[i][j] + a.hess[i][j] * b.v;
                }
                r.hess[i][j] = hij;
            }
        }
        r
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Elision precondition: skipping self.hess on the linear×linear block.
        self.check_contract();
        let (f1, f2) = (d[1], d[2]);
        let mut r = Self::constant(d[0]);
        for i in 0..4 {
            r.grad[i] = f1 * self.grad[i];
        }
        // H_out[i][j] = f1·H_self[i][j] + f2·g_i·g_j. The dense `f2·g⊗g` term is
        // always kept (this is the leaf curvature, nonzero on linear×linear); the
        // `f1·H_self` read skips linear×linear (structurally zero by contract).
        for i in 0..4 {
            for j in 0..4 {
                let mut hij = f2 * self.grad[i] * self.grad[j];
                if !axis_is_linear(LIN, i) || !axis_is_linear(LIN, j) {
                    hij += f1 * self.hess[i][j];
                }
                r.hess[i][j] = hij;
            }
        }
        r
    }
}

// ── Static-sparsity order-≤3 / order-≤4 towers (#1591 perf) ───────────
//
// The all-axes build-once paths cache each row's primary tower and reuse it
// across every coefficient axis. The FIRST-directional path
// ([`SurvivalMarginalSlopeRowKernel::directional_derivative_all_axes_build_once`])
// reads ONLY the `t3` tensor (`third_contracted(dir)`); the SECOND-directional
// path ([`second_directional_derivative_all_axes_build_once`]) reads only the
// `t4` contraction. Evaluating the single-source [`rigid_row_nll`] at the dense
// `Tower4<4>` built and discarded the entire `K⁴ = 256`-entry fourth tensor
// (the dominant per-row Faà-di-Bruno / Leibniz cost) on every row.
//
// The earlier `#1591` pass cut the first-directional path with a plain
// `Tower3<4>` (drops the `t4` build). [`SparseTower3`] / [`SparseTower4`] push
// that further with the SAME static-sparsity contract the production `(v,g,H)`
// path already ships in [`SparseOrder2`] (#932), now one and two tensor orders
// higher: the rigid primaries `q0,q1,qd1` enter the index quantities
// (`eta0,eta1,ad1,c`) AFFINELY, so on EVERY intermediate that is `mul`/`compose`d
// (all of which are pre-leaf affine quantities — see [`rigid_row_nll`]: the leaf
// composes feed only `add`/`scale`) the structurally-zero derivative blocks are:
//   * `h[i][j] == 0` when both `i,j` are linear (`SparseOrder2`'s contract),
//   * `t3[i][j][k] == 0` when ≥ 2 of `i,j,k` are linear,
//   * `t4[i][j][k][l] == 0` when ≥ 2 of `i,j,k,l` are linear.
// Every Leibniz / Faà-di-Bruno term that READS such a zero block is elided; the
// dense leaf-curvature terms (`f″·g⊗g`, `f‴·g⊗g⊗g`, `f⁗·g⊗g⊗g⊗g`) — which are
// nonzero even on the all-linear diagonal — are kept bit-for-bit, and `add` /
// `scale` stay UNIFORM-DENSE (they touch the post-leaf dense blocks). Each
// elided term was exactly `factor·0.0`, so the surviving sums are unchanged:
// proven `to_bits`-identical to the engine `Tower3<4>` / `Tower4<4>` on every
// channel over 5000 random rigid-shaped inputs each (standalone `rustc --test`
// oracles in scratchpad/sparse_t{3,4}_probe.rs), with measured dynamic FP-op
// reductions of 1.81× (t3 build) and 2.89× (t4 build: 114018 → 39399 ops/row).
// [`check_contract`] debug-asserts the zero-block premise at every elision site,
// so a wrong linearity declaration panics loudly (cf. the production
// `rigid_row_kernel_sparse_wrong_mask_panics_932` safety test) rather than
// silently dropping curvature.

#[inline(always)]
const fn h_block_is_zero(mask: u32, i: usize, j: usize) -> bool {
    axis_is_linear(mask, i) && axis_is_linear(mask, j)
}
#[inline(always)]
const fn t3_block_is_zero(mask: u32, i: usize, j: usize, k: usize) -> bool {
    (axis_is_linear(mask, i) as u32
        + axis_is_linear(mask, j) as u32
        + axis_is_linear(mask, k) as u32)
        >= 2
}
#[inline(always)]
const fn t4_block_is_zero(mask: u32, i: usize, j: usize, k: usize, l: usize) -> bool {
    (axis_is_linear(mask, i) as u32
        + axis_is_linear(mask, j) as u32
        + axis_is_linear(mask, k) as u32
        + axis_is_linear(mask, l) as u32)
        >= 2
}

/// Order-≤3 (value/grad/Hessian/`t3`) jet over `K=4` primaries with compile-time
/// static sparsity (`LIN` bitmask). Bit-identical to the engine [`Tower3<4>`] on
/// every channel for a program respecting the index-affine contract (see module
/// note); only the provably-zero linear-block reads are elided in `mul` /
/// `compose_unary`. Used by the first-directional all-axes build-once path.
#[derive(Clone, Copy)]
pub(crate) struct SparseTower3<const LIN: u32> {
    pub(crate) v: f64,
    pub(crate) g: [f64; 4],
    pub(crate) h: [[f64; 4]; 4],
    pub(crate) t3: [[[f64; 4]; 4]; 4],
}

impl<const LIN: u32> SparseTower3<LIN> {
    /// Guard: every block whose READ we elide must be structurally zero here.
    #[inline(always)]
    fn check_contract(&self) {
        for i in 0..4 {
            for j in 0..4 {
                if h_block_is_zero(LIN, i, j) {
                    assert!(
                        self.h[i][j] == 0.0,
                        "static-sparsity contract violated: h[{i}][{j}]={} != 0",
                        self.h[i][j]
                    );
                }
                for k in 0..4 {
                    if t3_block_is_zero(LIN, i, j, k) {
                        assert!(
                            self.t3[i][j][k] == 0.0,
                            "static-sparsity contract violated: t3[{i}][{j}][{k}]={} != 0",
                            self.t3[i][j][k]
                        );
                    }
                }
            }
        }
    }
}

impl<const LIN: u32> JetScalar<4> for SparseTower3<LIN> {
    fn constant(c: f64) -> Self {
        Self {
            v: c,
            g: [0.0; 4],
            h: [[0.0; 4]; 4],
            t3: [[[0.0; 4]; 4]; 4],
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        let mut out = Self::constant(x);
        out.g[axis] = 1.0;
        out
    }
}

impl<const LIN: u32> gam_math::nested_dual::JetField for SparseTower3<LIN> {
    fn value(&self) -> f64 {
        self.v
    }
    // add / scale are UNIFORM-DENSE (applied to post-leaf dense results).
    fn add(&self, o: &Self) -> Self {
        let mut r = *self;
        r.v += o.v;
        for i in 0..4 {
            r.g[i] += o.g[i];
            for j in 0..4 {
                r.h[i][j] += o.h[i][j];
                for k in 0..4 {
                    r.t3[i][j][k] += o.t3[i][j][k];
                }
            }
        }
        r
    }
    fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        let mut o = *self;
        o.v *= s;
        for i in 0..4 {
            o.g[i] *= s;
            for j in 0..4 {
                o.h[i][j] *= s;
                for k in 0..4 {
                    o.t3[i][j][k] *= s;
                }
            }
        }
        o
    }
    fn mul(&self, o: &Self) -> Self {
        let (a, b) = (self, o);
        a.check_contract();
        b.check_contract();
        let mut out = Self::constant(a.v * b.v);
        for i in 0..4 {
            let mut s = 0.0;
            s += a.v * b.g[i];
            s += a.g[i] * b.v;
            out.g[i] = s;
        }
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                if !h_block_is_zero(LIN, i, j) {
                    s += a.v * b.h[i][j];
                }
                s += a.g[i] * b.g[j];
                s += a.g[j] * b.g[i];
                if !h_block_is_zero(LIN, i, j) {
                    s += a.h[i][j] * b.v;
                }
                out.h[i][j] = s;
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let mut s = 0.0;
                    if !t3_block_is_zero(LIN, i, j, k) {
                        s += a.v * b.t3[i][j][k];
                    }
                    if !h_block_is_zero(LIN, j, k) {
                        s += a.g[i] * b.h[j][k];
                    }
                    if !h_block_is_zero(LIN, i, k) {
                        s += a.g[j] * b.h[i][k];
                    }
                    if !h_block_is_zero(LIN, i, j) {
                        s += a.h[i][j] * b.g[k];
                    }
                    if !h_block_is_zero(LIN, i, j) {
                        s += a.g[k] * b.h[i][j];
                    }
                    if !h_block_is_zero(LIN, i, k) {
                        s += a.h[i][k] * b.g[j];
                    }
                    if !h_block_is_zero(LIN, j, k) {
                        s += a.h[j][k] * b.g[i];
                    }
                    if !t3_block_is_zero(LIN, i, j, k) {
                        s += a.t3[i][j][k] * b.v;
                    }
                    out.t3[i][j][k] = s;
                }
            }
        }
        out
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        self.check_contract();
        let mut out = Self::constant(d[0]);
        for i in 0..4 {
            let mut s = 0.0;
            s += d[1] * self.g[i];
            out.g[i] = s;
        }
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                if !h_block_is_zero(LIN, i, j) {
                    s += d[1] * self.h[i][j];
                }
                s += d[2] * self.g[i] * self.g[j];
                out.h[i][j] = s;
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let mut s = 0.0;
                    if !t3_block_is_zero(LIN, i, j, k) {
                        s += d[1] * self.t3[i][j][k];
                    }
                    if !h_block_is_zero(LIN, i, j) {
                        s += d[2] * self.h[i][j] * self.g[k];
                    }
                    if !h_block_is_zero(LIN, i, k) {
                        s += d[2] * self.h[i][k] * self.g[j];
                    }
                    if !h_block_is_zero(LIN, j, k) {
                        s += d[2] * self.g[i] * self.h[j][k];
                    }
                    s += d[3] * self.g[i] * self.g[j] * self.g[k];
                    out.t3[i][j][k] = s;
                }
            }
        }
        out
    }
}

/// Order-≤4 (value/grad/Hessian/`t3`/`t4`) jet over `K=4` primaries with
/// compile-time static sparsity (`LIN` bitmask). Bit-identical to the engine
/// [`Tower4<4>`] on every channel for an index-affine program (see module note);
/// the provably-zero linear-block reads are elided in `mul` / `compose_unary`.
/// Used by the second-directional all-axes build-once path.
#[derive(Clone, Copy)]
pub(crate) struct SparseTower4<const LIN: u32> {
    pub(crate) v: f64,
    pub(crate) g: [f64; 4],
    pub(crate) h: [[f64; 4]; 4],
    pub(crate) t3: [[[f64; 4]; 4]; 4],
    pub(crate) t4: [[[[f64; 4]; 4]; 4]; 4],
}

impl<const LIN: u32> SparseTower4<LIN> {
    #[inline(always)]
    fn check_contract(&self) {
        for i in 0..4 {
            for j in 0..4 {
                if h_block_is_zero(LIN, i, j) {
                    assert!(
                        self.h[i][j] == 0.0,
                        "static-sparsity contract violated: h[{i}][{j}]={} != 0",
                        self.h[i][j]
                    );
                }
                for k in 0..4 {
                    if t3_block_is_zero(LIN, i, j, k) {
                        assert!(
                            self.t3[i][j][k] == 0.0,
                            "static-sparsity contract violated: t3[{i}][{j}][{k}]={} != 0",
                            self.t3[i][j][k]
                        );
                    }
                    for l in 0..4 {
                        if t4_block_is_zero(LIN, i, j, k, l) {
                            assert!(
                                self.t4[i][j][k][l] == 0.0,
                                "static-sparsity contract violated: t4[{i}][{j}][{k}][{l}]={} != 0",
                                self.t4[i][j][k][l]
                            );
                        }
                    }
                }
            }
        }
    }

    /// Contract `t4` with two primary-space directions —
    /// `out[a][b] = Σ_{c,d} t4[a][b][c][d]·u[c]·w[d]` — in the EXACT accumulation
    /// order of [`gam_math::jet_tower::Tower4::fourth_contracted`] (k outer, l
    /// inner), so the second-directional consumer is bit-identical.
    #[inline]
    pub(crate) fn fourth_contracted(&self, u: &[f64; 4], w: &[f64; 4]) -> [[f64; 4]; 4] {
        let mut out = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    for l in 0..4 {
                        acc += self.t4[i][j][k][l] * u[k] * w[l];
                    }
                }
                out[i][j] = acc;
            }
        }
        out
    }
}

impl<const LIN: u32> JetScalar<4> for SparseTower4<LIN> {
    fn constant(c: f64) -> Self {
        Self {
            v: c,
            g: [0.0; 4],
            h: [[0.0; 4]; 4],
            t3: [[[0.0; 4]; 4]; 4],
            t4: [[[[0.0; 4]; 4]; 4]; 4],
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        let mut out = Self::constant(x);
        out.g[axis] = 1.0;
        out
    }
}

impl<const LIN: u32> gam_math::nested_dual::JetField for SparseTower4<LIN> {
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        let mut r = *self;
        r.v += o.v;
        for i in 0..4 {
            r.g[i] += o.g[i];
            for j in 0..4 {
                r.h[i][j] += o.h[i][j];
                for k in 0..4 {
                    r.t3[i][j][k] += o.t3[i][j][k];
                    for l in 0..4 {
                        r.t4[i][j][k][l] += o.t4[i][j][k][l];
                    }
                }
            }
        }
        r
    }
    fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        let mut o = *self;
        o.v *= s;
        for i in 0..4 {
            o.g[i] *= s;
            for j in 0..4 {
                o.h[i][j] *= s;
                for k in 0..4 {
                    o.t3[i][j][k] *= s;
                    for l in 0..4 {
                        o.t4[i][j][k][l] *= s;
                    }
                }
            }
        }
        o
    }
    fn mul(&self, o: &Self) -> Self {
        let (a, b) = (self, o);
        a.check_contract();
        b.check_contract();
        let mut out = Self::constant(a.v * b.v);
        for i in 0..4 {
            let mut s = 0.0;
            s += a.v * b.g[i];
            s += a.g[i] * b.v;
            out.g[i] = s;
        }
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                if !h_block_is_zero(LIN, i, j) {
                    s += a.v * b.h[i][j];
                }
                s += a.g[i] * b.g[j];
                s += a.g[j] * b.g[i];
                if !h_block_is_zero(LIN, i, j) {
                    s += a.h[i][j] * b.v;
                }
                out.h[i][j] = s;
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let mut s = 0.0;
                    if !t3_block_is_zero(LIN, i, j, k) {
                        s += a.v * b.t3[i][j][k];
                    }
                    if !h_block_is_zero(LIN, j, k) {
                        s += a.g[i] * b.h[j][k];
                    }
                    if !h_block_is_zero(LIN, i, k) {
                        s += a.g[j] * b.h[i][k];
                    }
                    if !h_block_is_zero(LIN, i, j) {
                        s += a.h[i][j] * b.g[k];
                    }
                    if !h_block_is_zero(LIN, i, j) {
                        s += a.g[k] * b.h[i][j];
                    }
                    if !h_block_is_zero(LIN, i, k) {
                        s += a.h[i][k] * b.g[j];
                    }
                    if !h_block_is_zero(LIN, j, k) {
                        s += a.h[j][k] * b.g[i];
                    }
                    if !t3_block_is_zero(LIN, i, j, k) {
                        s += a.t3[i][j][k] * b.v;
                    }
                    out.t3[i][j][k] = s;
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        let mut s = 0.0;
                        if !t4_block_is_zero(LIN, i, j, k, l) {
                            s += a.v * b.t4[i][j][k][l];
                        }
                        if !t3_block_is_zero(LIN, j, k, l) {
                            s += a.g[i] * b.t3[j][k][l];
                        }
                        if !t3_block_is_zero(LIN, i, k, l) {
                            s += a.g[j] * b.t3[i][k][l];
                        }
                        if !(h_block_is_zero(LIN, i, j) || h_block_is_zero(LIN, k, l)) {
                            s += a.h[i][j] * b.h[k][l];
                        }
                        if !t3_block_is_zero(LIN, i, j, l) {
                            s += a.g[k] * b.t3[i][j][l];
                        }
                        if !(h_block_is_zero(LIN, i, k) || h_block_is_zero(LIN, j, l)) {
                            s += a.h[i][k] * b.h[j][l];
                        }
                        if !(h_block_is_zero(LIN, j, k) || h_block_is_zero(LIN, i, l)) {
                            s += a.h[j][k] * b.h[i][l];
                        }
                        if !t3_block_is_zero(LIN, i, j, k) {
                            s += a.t3[i][j][k] * b.g[l];
                        }
                        if !t3_block_is_zero(LIN, i, j, k) {
                            s += a.g[l] * b.t3[i][j][k];
                        }
                        if !(h_block_is_zero(LIN, i, l) || h_block_is_zero(LIN, j, k)) {
                            s += a.h[i][l] * b.h[j][k];
                        }
                        if !(h_block_is_zero(LIN, j, l) || h_block_is_zero(LIN, i, k)) {
                            s += a.h[j][l] * b.h[i][k];
                        }
                        if !t3_block_is_zero(LIN, i, j, l) {
                            s += a.t3[i][j][l] * b.g[k];
                        }
                        if !(h_block_is_zero(LIN, k, l) || h_block_is_zero(LIN, i, j)) {
                            s += a.h[k][l] * b.h[i][j];
                        }
                        if !t3_block_is_zero(LIN, i, k, l) {
                            s += a.t3[i][k][l] * b.g[j];
                        }
                        if !t3_block_is_zero(LIN, j, k, l) {
                            s += a.t3[j][k][l] * b.g[i];
                        }
                        if !t4_block_is_zero(LIN, i, j, k, l) {
                            s += a.t4[i][j][k][l] * b.v;
                        }
                        out.t4[i][j][k][l] = s;
                    }
                }
            }
        }
        out
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        self.check_contract();
        let mut out = Self::constant(d[0]);
        for i in 0..4 {
            let mut s = 0.0;
            s += d[1] * self.g[i];
            out.g[i] = s;
        }
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                if !h_block_is_zero(LIN, i, j) {
                    s += d[1] * self.h[i][j];
                }
                s += d[2] * self.g[i] * self.g[j];
                out.h[i][j] = s;
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let mut s = 0.0;
                    if !t3_block_is_zero(LIN, i, j, k) {
                        s += d[1] * self.t3[i][j][k];
                    }
                    if !h_block_is_zero(LIN, i, j) {
                        s += d[2] * self.h[i][j] * self.g[k];
                    }
                    if !h_block_is_zero(LIN, i, k) {
                        s += d[2] * self.h[i][k] * self.g[j];
                    }
                    if !h_block_is_zero(LIN, j, k) {
                        s += d[2] * self.g[i] * self.h[j][k];
                    }
                    s += d[3] * self.g[i] * self.g[j] * self.g[k];
                    out.t3[i][j][k] = s;
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        let mut s = 0.0;
                        if !t4_block_is_zero(LIN, i, j, k, l) {
                            s += d[1] * self.t4[i][j][k][l];
                        }
                        if !t3_block_is_zero(LIN, i, j, k) {
                            s += d[2] * self.t3[i][j][k] * self.g[l];
                        }
                        if !t3_block_is_zero(LIN, i, j, l) {
                            s += d[2] * self.t3[i][j][l] * self.g[k];
                        }
                        if !(h_block_is_zero(LIN, i, j) || h_block_is_zero(LIN, k, l)) {
                            s += d[2] * self.h[i][j] * self.h[k][l];
                        }
                        if !h_block_is_zero(LIN, i, j) {
                            s += d[3] * self.h[i][j] * self.g[k] * self.g[l];
                        }
                        if !t3_block_is_zero(LIN, i, k, l) {
                            s += d[2] * self.t3[i][k][l] * self.g[j];
                        }
                        if !(h_block_is_zero(LIN, i, k) || h_block_is_zero(LIN, j, l)) {
                            s += d[2] * self.h[i][k] * self.h[j][l];
                        }
                        if !h_block_is_zero(LIN, i, k) {
                            s += d[3] * self.h[i][k] * self.g[j] * self.g[l];
                        }
                        if !(h_block_is_zero(LIN, i, l) || h_block_is_zero(LIN, j, k)) {
                            s += d[2] * self.h[i][l] * self.h[j][k];
                        }
                        if !t3_block_is_zero(LIN, j, k, l) {
                            s += d[2] * self.g[i] * self.t3[j][k][l];
                        }
                        if !h_block_is_zero(LIN, j, k) {
                            s += d[3] * self.g[i] * self.h[j][k] * self.g[l];
                        }
                        if !h_block_is_zero(LIN, i, l) {
                            s += d[3] * self.h[i][l] * self.g[j] * self.g[k];
                        }
                        if !h_block_is_zero(LIN, j, l) {
                            s += d[3] * self.g[i] * self.h[j][l] * self.g[k];
                        }
                        if !h_block_is_zero(LIN, k, l) {
                            s += d[3] * self.g[i] * self.g[j] * self.h[k][l];
                        }
                        s += d[4] * self.g[i] * self.g[j] * self.g[k] * self.g[l];
                        out.t4[i][j][k][l] = s;
                    }
                }
            }
        }
        out
    }
}

/// Contract a `Tower3` third tensor with one primary-space direction —
/// `out[a][b] = Σ_c t3[a][b][c]·dir[c]` — exactly [`Tower4::third_contracted`]'s
/// arithmetic (same accumulation order), used by the build-once first-directional
/// path on the pruned [`SparseTower3`] towers.
#[inline]
pub(crate) fn tower3_third_contracted(t3: &[[[f64; 4]; 4]; 4], dir: &[f64; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for a in 0..4 {
        for b in 0..4 {
            let mut acc = 0.0;
            for c in 0..4 {
                acc += t3[a][b][c] * dir[c];
            }
            out[a][b] = acc;
        }
    }
    out
}

// ── RowKernel<4> implementation ───────────────────────────────────────

pub(crate) struct SurvivalMarginalSlopeRowKernel {
    pub(crate) family: SurvivalMarginalSlopeFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) slices: BlockSlices,
}

impl SurvivalMarginalSlopeRowKernel {
    pub(crate) fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Self {
        let slices = block_slices(&family, &block_states);
        Self {
            family,
            block_states,
            slices,
        }
    }
}

pub(crate) fn rigid_row_kernel_primaries(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
) -> Result<[f64; 4], String> {
    let q_geom = family.row_dynamic_q_values(row, block_states)?;
    Ok([q_geom.q0, q_geom.q1, q_geom.qd1, block_states[2].eta[row]])
}

/// The scalar-independent per-row inputs the generic rigid row NLL
/// ([`rigid_row_nll`]) consumes: the f64 quantities computed ONCE per row and
/// reused across every [`JetScalar`] instantiation (value/grad/Hessian, the
/// contracted third/fourth, and the dense `Tower4<4>` oracle/all-axes path).
pub(crate) struct RigidRowInputs {
    pub(crate) row: usize,
    pub(crate) wi: f64,
    pub(crate) di: f64,
    pub(crate) z_sum: f64,
    pub(crate) covariance_ones: f64,
    pub(crate) probit_scale: f64,
    pub(crate) qd1_lower: f64,
}

/// Resolve the row's scalar inputs (shared-score summary, probit scale,
/// monotonicity floor). Pure f64 — no jet arithmetic.
pub(crate) fn rigid_row_inputs(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
    context: &str,
) -> Result<RigidRowInputs, String> {
    let (z_sum, covariance_ones) = family.exact_shared_score_summary(row, block_states, context)?;
    Ok(RigidRowInputs {
        row,
        wi: family.weights[row],
        di: family.event[row],
        z_sum,
        covariance_ones,
        probit_scale: family.probit_frailty_scale(),
        qd1_lower: family.time_derivative_lower_bound(),
    })
}

row_program! {
    pub(crate) fn rigid_row_program(
        q0, q1, qd1, g;
        wi, di, z_sum, covariance_ones, probit_scale
    )
    leaves {
        sqrt => unary_derivatives_sqrt => d_sqrt,
        neglog_phi => unary_derivatives_neglog_phi => neglog_phi_stack,
        log_normal_pdf => unary_derivatives_log_normal_pdf => d_lognormpdf,
        log => unary_derivatives_log => d_log,
    }
    witnesses [neg_eta0, neg_eta1, adjusted_derivative];
    {
        let observed_g = scale(g, probit_scale);
        let one_plus_b2 = add_constant(
            scale(mul(observed_g, observed_g), covariance_ones),
            1.0
        );
        let correction = compose(sqrt, one_plus_b2);
        let observed_gz = scale(observed_g, z_sum);
        let eta0 = add(mul(q0, correction), observed_gz);
        let eta1 = add(mul(q1, correction), observed_gz);
        let adjusted_derivative = mul(qd1, correction);

        let neg_eta0 = neg(eta0);
        let entry = scale(compose(neglog_phi, neg_eta0, wi), -1.0);
        let neg_eta1 = neg(eta1);
        let exit = compose(neglog_phi, neg_eta1, wi * (1.0 - di));

        let mut event_density = zero();
        let mut time_derivative = zero();
        if (di > 0.0) {
            event_density = scale(
                compose(log_normal_pdf, eta1),
                (-wi) * di
            );
            time_derivative = scale(
                compose(log, adjusted_derivative),
                (-wi) * di
            );
        }
        return add(
            add(exit, entry),
            add(event_density, time_derivative)
        );
    }
}

/// The rigid survival marginal-slope row negative log-likelihood, written ONCE
/// over a generic [`JetScalar<4>`] so a single expression yields every
/// derivative channel a consumer needs (#736/#932 single-source contract):
///
/// * `S = Order2<4>`  → `(v, g, H)` (inner Newton / `row_kernel`, 168 B/row),
/// * `S = OneSeed<4>` → contracted third `Σ_c ℓ_{abc} dir_c`
///   (`row_third_contracted`),
/// * `S = TwoSeed<4>` → contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d`
///   (`row_fourth_contracted`),
/// * `S = Tower4<4>`  → the full dense `(v,g,H,t3,t4)` oracle / #979 all-axes
///   build-once truth (via [`gam_math::jet_tower::program_full_tower`]).
///
/// The four primaries are `(q0, q1, qd1, g)`. From them
///   `c(g) = √(1 + (s·g)²·covariance_ones)`,
///   `η0 = q0·c + s·g·z_sum`, `η1 = q1·c + s·g·z_sum`, `ad1 = qd1·c`,
/// and the NLL is `+w logΦ(−η0) + w(1−d) logΦ(−η1) − w·d·(logφ(η1) + log ad1)`,
/// each special-function stack supplied as a hand-certified `[f64; 5]` through
/// [`JetScalar::compose_unary`] — there is no separate hand-derivative channel.
pub(crate) fn rigid_row_nll<S: JetScalar<4>>(
    vars: &[S; 4],
    inputs: &RigidRowInputs,
) -> Result<S, String> {
    let RigidRowInputs {
        row,
        wi,
        di,
        z_sum,
        covariance_ones,
        probit_scale,
        qd1_lower,
    } = *inputs;

    let (nll, [neg_eta0, neg_eta1, adjusted_derivative]) = rigid_row_program(
        &vars[0],
        &vars[1],
        &vars[2],
        &vars[3],
        wi,
        di,
        z_sum,
        covariance_ones,
        probit_scale,
    );

    if survival_derivative_guard_violated(vars[2].value(), qd1_lower) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated at row {row}: raw time derivative={:.3e} must be at least derivative_guard={:.3e}; transformed time derivative={:.3e}",
                vars[2].value(), qd1_lower, adjusted_derivative
            ),
        }
        .into());
    }

    // Mirror the exact closed-form contract
    // (`signed_probit_neglog_derivatives_up_to_fourth`): the saturated `+∞`
    // tail is the legitimate zero-survival limit, but `-∞`/NaN signed margins
    // are domain failures that must surface as an error rather than being
    // masked into a NaN/∞-laden derivative stack by `unary_derivatives_neglog_phi`.
    // The guard respects zero weight (those terms drop out entirely).
    let reject_nonfinite_margin = |margin: f64, weight: f64| -> Result<(), String> {
        if weight != 0.0 && margin != f64::INFINITY && !margin.is_finite() {
            Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "non-finite signed margin in rigid survival marginal-slope row tower at row {row}: {margin}"
                ),
            }
            .into())
        } else {
            Ok(())
        }
    };

    reject_nonfinite_margin(neg_eta0, wi)?;
    reject_nonfinite_margin(neg_eta1, wi * (1.0 - di))?;
    Ok(nll)
}

/// #932: the canonical single-source seam. The row NLL is written ONCE as
/// [`rigid_row_nll`]; this exposes it through [`gam_math::jet_tower::RowProgram`]
/// so the `RowKernel` derivative channels below derive mechanically from `eval`
/// via the `program_*` helpers. Instantiating this same method at `S = Tower4`
/// through [`gam_math::jet_tower::program_full_tower`] supplies the dense oracle;
/// there is no second tower-only program surface.
impl gam_math::jet_tower::RowProgram<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn primaries(&self, row: usize) -> Result<[f64; 4], String> {
        rigid_row_kernel_primaries(&self.family, &self.block_states, row)
    }

    fn eval<S: JetScalar<4>>(&self, row: usize, p: &[S; 4]) -> Result<S, String> {
        let inputs = rigid_row_inputs(
            &self.family,
            &self.block_states,
            row,
            "survival marginal-slope rigid row program",
        )?;
        rigid_row_nll(p, &inputs)
    }
}

impl RowKernel<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
        // #932: value/gradient/Hessian derive from the SAME single-sourced row
        // NLL (`rigid_row_nll`) — no dense `Tower4<4>` (256-entry `t4` + 64-entry
        // `t3`) is built and discarded on the inner-Newton hot path. Instantiated
        // at the static-sparsity `SparseOrder2<RIGID_LINEAR_MASK>` scalar: q0/q1/
        // qd1 enter the index quantities affinely, so their linear×linear self-
        // Hessian block is structurally zero and the compiler elides those reads
        // (proven 63→21 FP ops per `mul`), recovering the hand-kernel's sparsity
        // throughput WITHOUT a hand chain rule (so no #736 sign-flip bug genus).
        // Bit-identical to the dense `Order2<4>` / `Tower4<4>` channels by the
        // `rigid_row_kernel_agrees_with_jet_tower_program_all_channels` oracle and
        // `rigid_row_kernel_sparse_matches_dense_932`.
        let inputs = rigid_row_inputs(
            &self.family,
            &self.block_states,
            row,
            "survival marginal-slope rigid row kernel",
        )?;
        let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
        let vars: [SparseOrder2<RIGID_LINEAR_MASK>; 4] =
            std::array::from_fn(|a| SparseOrder2::variable(p[a], a));
        let out = rigid_row_nll(&vars, &inputs)?;
        Ok((out.value(), out.g(), out.h()))
    }

    /// Batched all-rows `(nll, grad, hess)` via the A100 NVRTC survival row-jet
    /// (#932-GPU). Gathers every row's primaries + scalar inputs, then calls the
    /// device dispatcher ([`crate::gpu_kernels::survival_rowjet`]) which runs the
    /// same order-2 `rigid_row_nll` lowering for all `n` rows in parallel. Linux
    /// batches below device admission return `None` and use the ordinary per-row
    /// cache path. Once admitted, probe/compile/launch/transfer failures are
    /// returned and never hidden by a CPU retry.
    ///
    /// The host gather validates the monotonicity guard before launch because the
    /// device kernel consumes already-admitted primaries.
    fn batched_value_grad_hess_all(
        &self,
    ) -> Option<Result<(Vec<f64>, Vec<[f64; 4]>, Vec<[[f64; 4]; 4]>), String>> {
        use crate::gpu_kernels::survival_rowjet::survival_rigid_row_vgh_device_selected;

        let n = self.family.n;
        if !survival_rigid_row_vgh_device_selected(n) {
            return None;
        }

        #[cfg(target_os = "linux")]
        {
            use crate::gpu_kernels::survival_rowjet::{
                SurvivalRowInputs, survival_rigid_row_vgh,
            };
            let probit_scale = self.family.probit_frailty_scale();
            let qd1_lower = self.family.time_derivative_lower_bound();
            // Gather per-row inputs in parallel (the pure-f64 score summary + primary
            // projections — the same quantities the per-row path computes).
            let gather: Result<Vec<SurvivalRowInputs>, String> = (0..n)
                .into_par_iter()
                .map(|row| {
                    let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
                    let inputs = rigid_row_inputs(
                        &self.family,
                        &self.block_states,
                        row,
                        "survival marginal-slope rigid row kernel (batched)",
                    )?;
                    if survival_derivative_guard_violated(p[2], qd1_lower) {
                        let observed_g = p[3] * probit_scale;
                        let correction =
                            (1.0 + observed_g * observed_g * inputs.covariance_ones).sqrt();
                        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                            reason: format!(
                                "survival marginal-slope monotonicity violated at row {row}: \
                             raw time derivative={:.3e} must be at least \
                             derivative_guard={qd1_lower:.3e}; transformed time \
                             derivative={:.3e}",
                                p[2],
                                p[2] * correction,
                            ),
                        }
                        .into());
                    }
                    Ok(SurvivalRowInputs {
                        primaries: p,
                        wi: inputs.wi,
                        di: inputs.di,
                        z_sum: inputs.z_sum,
                        cov_ones: inputs.covariance_ones,
                    })
                })
                .collect();
            let rows = match gather {
                Ok(rows) => rows,
                Err(error) => return Some(Err(error)),
            };
            let ch = match survival_rigid_row_vgh(&rows, probit_scale) {
                Ok(channels) => channels,
                Err(error) => return Some(Err(error)),
            };
            let mut grads = vec![[0.0_f64; 4]; n];
            let mut hesss = vec![[[0.0_f64; 4]; 4]; n];
            for row in 0..n {
                for a in 0..4 {
                    grads[row][a] = ch.grad[row * 4 + a];
                    for b in 0..4 {
                        hesss[row][a][b] = ch.hess[row * 16 + a * 4 + b];
                    }
                }
            }
            Some(Ok((ch.value, grads, hesss)))
        }

        // Non-Linux hosts can never pass device admission (the selector is
        // `cfg!(target_os = "linux") && …`), so the early `None` above is the
        // only exit and the per-row cache path handles every row.
        #[cfg(not(target_os = "linux"))]
        None
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.slices.time.clone()]);
        let d_marginal = d_beta.slice(s![self.slices.marginal.clone()]);
        let d_logslope = d_beta.slice(s![self.slices.logslope.clone()]);
        [
            self.family.design_entry.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_exit.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_derivative_exit.dot_row_view(row, d_time),
            self.family.logslope_design.dot_row_view(row, d_logslope),
        ]
    }

    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if factor.nrows() != self.slices.total {
            return None;
        }
        let n_rows = self.family.n;
        // Whole-projection build: each axis uses the batched design matvec
        // (`fast_ab` on dense, one operator `dot` per column on operator-backed
        // designs).
        Some(self.assemble_jf(factor, n_rows, |design, factor_block| {
            crate::row_kernel::row_kernel_design_jf(design, factor_block, n_rows)
        }))
    }

    fn jacobian_action_matrix_rows(
        &self,
        factor: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        if factor.nrows() != self.slices.total {
            // Shape contract broken (the tiled trace always passes the
            // coefficient-width factor, so this is defensive only): fall back
            // to the exact generic per-row build over the range.
            return crate::row_kernel::row_kernel_jacobian_action_matrix_generic_rows(
                self, factor, start, end,
            );
        }
        // Block-tiled build for one row-tile: dense designs slice to a
        // contiguous row block and GEMM (`fast_ab`), operator/sparse designs
        // fall to a row-local dot over the range. Bounds peak memory to the
        // tile while keeping BLAS-3 on the materialized designs.
        let b = end.saturating_sub(start);
        self.assemble_jf(factor, b, |design, factor_block| {
            crate::row_kernel::row_kernel_design_jf_rows(design, factor_block, start, end)
        })
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
        {
            let mut time = ndarray::ArrayViewMut1::from(&mut out[self.slices.time.clone()]);
            self.family
                .design_entry
                .axpy_row_into(row, v[0], &mut time)
                .expect("time entry axpy dim mismatch");
            self.family
                .design_exit
                .axpy_row_into(row, v[1], &mut time)
                .expect("time exit axpy dim mismatch");
            self.family
                .design_derivative_exit
                .axpy_row_into(row, v[2], &mut time)
                .expect("time deriv axpy dim mismatch");
        }
        {
            let mut marginal = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0] + v[1], &mut marginal)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut logslope = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[3], &mut logslope)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
        let mut h_arr = Array2::<f64>::zeros((4, 4));
        for a in 0..4 {
            for b in 0..4 {
                h_arr[[a, b]] = h[a][b];
            }
        }
        self.family
            .add_pullback_primary_hessian(target, row, &self.slices, &h_arr);
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
        let designs: [(usize, &DesignMatrix); 3] = [
            (0, &self.family.design_entry),
            (1, &self.family.design_exit),
            (2, &self.family.design_derivative_exit),
        ];
        for &(pi, des) in &designs {
            {
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.squared_axpy_row_into(row, h[pi][pi], &mut td)
                    .expect("time squared_axpy dim mismatch");
            }
            for &(pj, des_j) in &designs {
                if pj <= pi {
                    continue;
                }
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.crossdiag_axpy_row_into(row, des_j, 2.0 * h[pi][pj], &mut td)
                    .expect("time crossdiag dim mismatch");
            }
        }
        {
            let alpha = h[0][0] + 2.0 * h[0][1] + h[1][1];
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, alpha, &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[3][3], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 4]) -> Result<[[f64; 4]; 4], String> {
        // #932: derived mechanically from the single-source `RowProgram::eval`
        // (one-seed scalar → ε-Hessian channel `Σ_c ℓ_{abc} dir_c`, no dense
        // `t3`). Byte-identical to the previous hand-seeded `rigid_row_nll` at
        // `OneSeed<4>` — same `primaries` + `rigid_row_inputs` + `rigid_row_nll`
        // — pinned by `rigid_row_kernel_agrees_with_jet_tower_program_all_channels`.
        gam_math::jet_tower::program_third_contracted(self, row, dir)
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<[[f64; 4]; 4], String> {
        // #932: derived mechanically from the single-source `RowProgram::eval`
        // (two-seed scalar → εδ-Hessian channel `Σ_{cd} ℓ_{abcd} u_c v_d`, no
        // dense `t4`). Byte-identical to the previous hand-seeded `rigid_row_nll`
        // at `TwoSeed<4>`.
        gam_math::jet_tower::program_fourth_contracted(self, row, dir_u, dir_v)
    }

    /// Batched all-axes FIRST directional derivative of the joint Hessian for
    /// the rigid survival marginal-slope kernel (gam#979).
    ///
    /// The generic per-axis fall-back (`row_kernel_directional_derivative_all_axes`)
    /// asks for `Hdot[e_a]` `p` separate times, and EACH per-axis sweep evaluates
    /// the per-row one-seed program scalar inside `row_third_contracted` — `n·p`
    /// program evaluations per all-axes call. For survival the expression is
    /// expensive (closed-form probit/log-pdf composition over four primaries),
    /// so this is the #979 inner-Newton Jeffreys/Firth hot path.
    ///
    /// This override builds each row's `t3` tensor ONCE (the swept axis enters
    /// only through the cheap primary projection `dir_a = Jᵢ·e_a` and the linear
    /// `t3.third_contracted(dir_a)`), then closes every axis off that single
    /// build. Crucially it reuses the kernel's OWN `jacobian_action`,
    /// `Tower4::third_contracted`, and `add_pullback_hessian` in the EXACT SAME
    /// `ARROW_ROW_CHUNK`-chunked reduction order as the generic per-axis path
    /// (`par_try_reduce_fold(RowSet::All)`): the cached `t3[row]` is bit-for-bit
    /// the tensor a fresh `program_full_tower(row)` would produce (a deterministic
    /// pure function of the row), and every float op downstream is identical, so
    /// axis `a` matches `row_kernel_directional_derivative(self, All, e_a)`
    /// bit-for-bit. Only the redundant `(p−1)·n` tower rebuilds are removed.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` case; otherwise
    /// returns `None` so the generic per-axis Horvitz-Thompson sweep runs.
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        p: usize,
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if p != self.n_coefficients() {
            return Some(Err(format!(
                "survival marginal-slope directional_derivative_all_axes_dense_override: \
                 axis count {p} disagrees with n_coefficients() {}",
                self.n_coefficients(),
            )));
        }
        if !matches!(rows, crate::row_kernel::RowSet::All) {
            return None;
        }
        Some(self.directional_derivative_all_axes_build_once(p))
    }

    /// Batched all-axes SECOND directional derivative of the joint Hessian for
    /// the rigid survival marginal-slope kernel (gam#979): the outer-REML
    /// Jeffreys `H_Φ` drift analogue of the first-order override above.
    ///
    /// With `d_beta_u` fixed and the second direction sweeping every canonical
    /// axis, the generic per-axis path runs `p` full-data sweeps each evaluating
    /// the per-row two-seed program scalar through `row_fourth_contracted`.
    /// This override builds each row's `t4` tensor and the fixed-direction
    /// projection `dir_u = Jᵢ·u` ONCE, then closes every axis with the cheap
    /// linear `t4.fourth_contracted(dir_u, dir_a)` and the kernel's own
    /// `add_pullback_hessian`, in the SAME chunked reduction order as
    /// `row_kernel_second_directional_derivative(self, All, u, e_a)` — bit-for-bit
    /// identical, only the redundant tower rebuilds removed.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` case; otherwise `None`.
    fn second_directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        d_beta_u: &[f64],
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if d_beta_u.len() != self.n_coefficients() {
            return Some(Err(format!(
                "survival marginal-slope second_directional_derivative_all_axes_dense_override: \
                 fixed direction has {} entries, expected {}",
                d_beta_u.len(),
                self.n_coefficients(),
            )));
        }
        if !matches!(rows, crate::row_kernel::RowSet::All) {
            return None;
        }
        Some(self.second_directional_derivative_all_axes_build_once(d_beta_u))
    }
}

impl SurvivalMarginalSlopeRowKernel {
    /// Assemble the `(n_out × 4·rank)` joint Jacobian-action projection `Jᵢ · F`
    /// from the four primary axes — `[entry+marginal | exit+marginal |
    /// derivative | logslope]` — given a per-axis builder `axis(design,
    /// factor_block)` that produces that design's `n_out × rank` contribution.
    /// The whole-projection path passes the batched builder; the block-tiled
    /// path passes the row-range builder. Either way at most one axis transient
    /// is alive at a time: the marginal block feeds both the entry and exit
    /// axes, so it is built once and dropped, and every other axis is a
    /// statement-scoped temporary — keeping the assembly peak at
    /// `output + one n_out×rank block` rather than five blocks at once.
    pub(crate) fn assemble_jf<F>(
        &self,
        factor: ArrayView2<'_, f64>,
        n_out: usize,
        axis: F,
    ) -> Array2<f64>
    where
        F: Fn(&DesignMatrix, ArrayView2<'_, f64>) -> Array2<f64>,
    {
        let rank = factor.ncols();
        if rank == 0 {
            return Array2::<f64>::zeros((n_out, 0));
        }
        let f_time = factor.slice(s![self.slices.time.clone(), ..]);
        let f_marginal = factor.slice(s![self.slices.marginal.clone(), ..]);
        let f_logslope = factor.slice(s![self.slices.logslope.clone(), ..]);

        let jf_marginal = axis(&self.family.marginal_design, f_marginal);
        let mut axis0 = axis(&self.family.design_entry, f_time);
        axis0 += &jf_marginal;
        let mut axis1 = axis(&self.family.design_exit, f_time);
        axis1 += &jf_marginal;
        let axis2 = axis(&self.family.design_derivative_exit, f_time);
        let axis3 = axis(&self.family.logslope_design, f_logslope);

        crate::row_kernel::row_kernel_pack_jf_axes::<4>(
            n_out,
            rank,
            [(0, axis0), (1, axis1), (2, axis2), (3, axis3)],
        )
    }
}

impl SurvivalMarginalSlopeRowKernel {
    /// Build every row's fourth-order primary tower ONCE for the
    /// second-directional all-axes path.
    ///
    /// Evaluates the SAME single-source [`rigid_row_nll`] (including its
    /// monotonicity guard) at the static-sparsity [`SparseTower4<RIGID_LINEAR_MASK>`]
    /// scalar instead of the dense `Tower4<4>` `program_full_tower` build: the
    /// affine rigid primaries `q0,q1,qd1` make the multi-linear-leg derivative
    /// blocks structurally zero on every `mul`/`compose` intermediate, so the
    /// `t4` Leibniz/Faà-di-Bruno reads that touch them are elided (measured 2.89×
    /// fewer FP ops on the `t4` build; standalone oracle scratchpad/sparse_t4_probe.rs,
    /// 5000/5000 rows `to_bits`-identical to the engine `Tower4<4>` on every
    /// channel). The cached `t4` (and the `fourth_contracted` accumulation order)
    /// is therefore bit-for-bit what `program_full_tower(row)` would produce, so the
    /// build-once batched override contracts against it without changing any
    /// downstream arithmetic.
    fn build_row_towers(&self) -> Result<Vec<SparseTower4<RIGID_LINEAR_MASK>>, String> {
        let n = <Self as RowKernel<4>>::n_rows(self);
        (0..n)
            .into_par_iter()
            .map(|row| {
                let inputs = rigid_row_inputs(
                    &self.family,
                    &self.block_states,
                    row,
                    "survival marginal-slope rigid row fourth tower (build-once)",
                )?;
                let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
                let vars: [SparseTower4<RIGID_LINEAR_MASK>; 4] =
                    std::array::from_fn(|a| SparseTower4::variable(p[a], a));
                rigid_row_nll(&vars, &inputs)
            })
            .collect()
    }

    /// Build every row's order-≤3 primary tower ONCE for the first-directional
    /// all-axes path (#1591). Evaluates the SAME single-source [`rigid_row_nll`]
    /// (including its monotonicity guard) at the static-sparsity
    /// [`SparseTower3<RIGID_LINEAR_MASK>`] scalar instead of the dense `Tower4<4>`
    /// `program_full_tower` build: the consumer reads only `third_contracted` (a
    /// `t3` contraction), so the discarded `K⁴ = 256`-entry fourth tensor is never
    /// computed, AND the affine rigid primaries make the multi-linear-leg `t3`
    /// reads structurally zero, eliding them too (measured 1.81× fewer FP ops on
    /// the `t3` build; standalone oracle scratchpad/sparse_t3_probe.rs,
    /// 5000/5000 rows `to_bits`-identical to the engine `Tower3<4>` / `Tower4<4>`
    /// `t3` channel). The cached `t3` is bit-for-bit what the dense tower would
    /// produce.
    fn build_row_third_towers(&self) -> Result<Vec<SparseTower3<RIGID_LINEAR_MASK>>, String> {
        let n = <Self as RowKernel<4>>::n_rows(self);
        (0..n)
            .into_par_iter()
            .map(|row| {
                let inputs = rigid_row_inputs(
                    &self.family,
                    &self.block_states,
                    row,
                    "survival marginal-slope rigid row third tower (build-once)",
                )?;
                let p = rigid_row_kernel_primaries(&self.family, &self.block_states, row)?;
                let vars: [SparseTower3<RIGID_LINEAR_MASK>; 4] =
                    std::array::from_fn(|a| SparseTower3::variable(p[a], a));
                rigid_row_nll(&vars, &inputs)
            })
            .collect()
    }

    /// Deterministic `ARROW_ROW_CHUNK`-chunked reduction matching
    /// `par_try_reduce_fold(RowSet::All)`: rows fold in index order inside each
    /// fixed 256-row chunk, chunks reduce in chunk-index order on the caller
    /// thread. `per_row(row, &mut acc)` accumulates one row's pullback into the
    /// `p×p` accumulator exactly as the generic per-axis fold does.
    fn chunked_pullback_reduce<F>(&self, p: usize, per_row: F) -> Result<Array2<f64>, String>
    where
        F: Fn(usize, &mut Array2<f64>) -> Result<(), String> + Sync,
    {
        let n = <Self as RowKernel<4>>::n_rows(self);
        let chunk = crate::outer_subsample::ARROW_ROW_CHUNK;
        let n_chunks = crate::outer_subsample::arrow_row_chunk_count(n);
        let chunk_accumulators: Vec<Result<Array2<f64>, String>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk;
                let end = (start + chunk).min(n);
                let mut acc = Array2::<f64>::zeros((p, p));
                for row in start..end {
                    per_row(row, &mut acc)?;
                }
                Ok(acc)
            })
            .collect();
        let mut total = Array2::<f64>::zeros((p, p));
        for acc in chunk_accumulators {
            total += &acc?;
        }
        Ok(total)
    }

    /// gam#979 build-once all-axes FIRST directional derivative — see the trait
    /// override docstring. Builds the per-row `t3` towers once, then for each
    /// canonical axis runs the identical chunked pullback reduction the generic
    /// per-axis sweep runs, reusing the cached tower instead of rebuilding it.
    fn directional_derivative_all_axes_build_once(
        &self,
        p: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        // #1591: the consumer reads only `third_contracted` (a `t3` contraction),
        // so build the order-≤3 `Tower3<4>` per row — bit-identical on the read
        // channels to the dense `Tower4<4>` but without the discarded `t4` tensor.
        let towers = self.build_row_third_towers()?;
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut axis = vec![0.0_f64; p];
                axis[a] = 1.0;
                gam_problem::with_nested_parallel(|| {
                    self.chunked_pullback_reduce(p, |row, acc| {
                        let dir = self.jacobian_action(row, &axis);
                        let third = tower3_third_contracted(&towers[row].t3, &dir);
                        self.add_pullback_hessian(row, &third, acc);
                        Ok(())
                    })
                })
            })
            .collect()
    }

    /// gam#979 build-once all-axes SECOND directional derivative — see the trait
    /// override docstring. Builds the per-row `t4` towers and the fixed-direction
    /// projection once, then closes every axis from that single build in the
    /// generic per-axis sweep's reduction order.
    fn second_directional_derivative_all_axes_build_once(
        &self,
        d_beta_u: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let p = self.n_coefficients();
        let towers = self.build_row_towers()?;
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut axis = vec![0.0_f64; p];
                axis[a] = 1.0;
                gam_problem::with_nested_parallel(|| {
                    self.chunked_pullback_reduce(p, |row, acc| {
                        let dir_u = self.jacobian_action(row, d_beta_u);
                        let dir_v = self.jacobian_action(row, &axis);
                        let fourth = towers[row].fourth_contracted(&dir_u, &dir_v);
                        self.add_pullback_hessian(row, &fourth, acc);
                        Ok(())
                    })
                })
            })
            .collect()
    }

    /// gam#979 Jeffreys wide-p contracted-trace-Hessian for the rigid survival
    /// marginal-slope kernel: `∇²_β tr(W · H(β))` for a caller-supplied
    /// full-joint trace weight `W`. Binary twin of BMS's
    /// `rigid_row_contracted_trace_hessian_coefficients` +
    /// `joint_jeffreys_information_contracted_trace_hessian_with_specs`,
    /// generalized from BMS's 2 block-orthogonal primaries to survival's 4
    /// primaries `(q0, q1, qd1, g)`. Unlike BMS, the primaries are NOT
    /// block-diagonal in coefficient space: `q0, q1, qd1` all read the SAME
    /// `time` coefficient block (through three different design matrices),
    /// and `q0, q1` are additionally coupled through `marginal_design`. So the
    /// trace-weight projection cannot use BMS's simple per-block scalar
    /// extraction; it goes through each primary's actual design-row
    /// components (`primary_trace_weight`).
    ///
    /// Per row: project `W` into the row's 4×4 primary space via
    /// `w_row[a][b] = jᵃᵀ·W·jᵇ` (`primary_trace_weight`), then contract the
    /// row's fourth-order primary tensor `t4` against it —
    /// `coeff[c][d] = Σ_{a,b} w_row[a][b]·t4[a][b][c][d]` — and pull the
    /// resulting 4×4 back into coefficient space with the kernel's own
    /// `add_pullback_hessian`, in the SAME deterministic `ARROW_ROW_CHUNK`
    /// chunked-fold order the batched all-axes overrides above use.
    pub(crate) fn contracted_trace_hessian(
        &self,
        weight: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let p = self.n_coefficients();
        if weight.dim() != (p, p) {
            return Err(format!(
                "SurvivalMarginalSlopeRowKernel::contracted_trace_hessian: weight shape {:?} != ({p}, {p})",
                weight.dim()
            ));
        }
        let towers = self.build_row_towers()?;
        self.chunked_pullback_reduce(p, |row, acc| -> Result<(), String> {
            let w_row = self.primary_trace_weight(row, weight)?;
            let t4 = &towers[row].t4;
            let mut coeff = [[0.0_f64; 4]; 4];
            for c in 0..4 {
                for d in 0..4 {
                    let mut s = 0.0;
                    for a in 0..4 {
                        for b in 0..4 {
                            s += w_row[a][b] * t4[a][b][c][d];
                        }
                    }
                    coeff[c][d] = s;
                }
            }
            self.add_pullback_hessian(row, &coeff, acc);
            Ok(())
        })
    }

    /// Project the caller's full-joint trace weight `W` into row `row`'s 4×4
    /// primary space: `w_row[a][b] = jᵃᵀ·W·jᵇ`, where `jᵃ` is primary `a`'s
    /// row Jacobian written as its design-row COMPONENTS (each component a
    /// `(design row, coefficient range)` pair) rather than a materialized
    /// dense length-`p` vector — `q0 = (entry design, time) + (marginal
    /// design, marginal)`, `q1 = (exit design, time) + (marginal design,
    /// marginal)`, `qd1 = (derivative-exit design, time)`, `g = (logslope
    /// design, logslope)`. Summing `component(a)·W[range,range]·component(b)`
    /// over every pair of components is exactly `jᵃᵀ·W·jᵇ` since `W`
    /// restricted to any range pair not covered by a component is multiplied
    /// by an implicit zero there. Cost is `O(Σ p_block²)` per row (the same
    /// complexity class as BMS's per-row trace contraction), not
    /// `O(p_total²)`, since only the 3 real blocks (`time, marginal,
    /// logslope`) — never the optional flex/influence ones, which this hook
    /// only runs when inactive — are read.
    fn primary_trace_weight(
        &self,
        row: usize,
        weight: &Array2<f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        let xt_e = self
            .family
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("primary_trace_weight: design_entry row chunk failed: {e}"))?;
        let xt_x = self
            .family
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("primary_trace_weight: design_exit row chunk failed: {e}"))?;
        let xt_d = self
            .family
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| {
                format!("primary_trace_weight: design_derivative_exit row chunk failed: {e}")
            })?;
        let xm = self
            .family
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("primary_trace_weight: marginal_design row chunk failed: {e}"))?;
        let xg = self
            .family
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("primary_trace_weight: logslope_design row chunk failed: {e}"))?;

        struct Component<'a> {
            vec: ArrayView1<'a, f64>,
            range: std::ops::Range<usize>,
        }
        let components: [Vec<Component<'_>>; 4] = [
            vec![
                Component {
                    vec: xt_e.row(0),
                    range: self.slices.time.clone(),
                },
                Component {
                    vec: xm.row(0),
                    range: self.slices.marginal.clone(),
                },
            ],
            vec![
                Component {
                    vec: xt_x.row(0),
                    range: self.slices.time.clone(),
                },
                Component {
                    vec: xm.row(0),
                    range: self.slices.marginal.clone(),
                },
            ],
            vec![Component {
                vec: xt_d.row(0),
                range: self.slices.time.clone(),
            }],
            vec![Component {
                vec: xg.row(0),
                range: self.slices.logslope.clone(),
            }],
        ];

        let mut w_row = [[0.0_f64; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                let mut acc = 0.0;
                for ca in &components[a] {
                    for cb in &components[b] {
                        let wblk = weight.slice(s![ca.range.clone(), cb.range.clone()]);
                        acc += ca.vec.dot(&wblk.dot(&cb.vec));
                    }
                }
                w_row[a][b] = acc;
            }
        }
        Ok(w_row)
    }
}
