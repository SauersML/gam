//! The rigid per-row `RowKernel<4>` implementation and its Jacobian-action
//! assembly helpers: the memory-efficient row-at-a-time kernel used on the
//! no-flex hot path.

use super::*;

use gam_math::jet_scalar::JetScalar;

#[inline(always)]
const fn axis_is_linear(mask: u32, a: usize) -> bool {
    (mask >> a) & 1 == 1
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
// that further with the same index-affine contract used by the direct order-two
// symbolic lowering, now one and two tensor orders higher: the rigid primaries
// `q0,q1,qd1` enter the index quantities
// (`eta0,eta1,ad1,c`) AFFINELY, so on EVERY intermediate that is `mul`/`compose`d
// (all of which are pre-leaf affine quantities — see [`rigid_row_nll`]: the leaf
// composes feed only `add`/`scale`) the structurally-zero derivative blocks are:
//   * `h[i][j] == 0` when both `i,j` are linear,
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
// the sparse-tower wrong-mask safety tests) rather than silently dropping
// curvature.

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
pub(crate) struct SparseTower3<const K: usize, const LIN: u32> {
    pub(crate) v: f64,
    pub(crate) g: [f64; K],
    pub(crate) h: [[f64; K]; K],
    pub(crate) t3: [[[f64; K]; K]; K],
}

impl<const K: usize, const LIN: u32> SparseTower3<K, LIN> {
    /// Guard: every block whose READ we elide must be structurally zero here.
    #[inline(always)]
    fn check_contract(&self) {
        for i in 0..K {
            for j in 0..K {
                if h_block_is_zero(LIN, i, j) {
                    assert!(
                        self.h[i][j] == 0.0,
                        "static-sparsity contract violated: h[{i}][{j}]={} != 0",
                        self.h[i][j]
                    );
                }
                for k in 0..K {
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

impl<const K: usize, const LIN: u32> JetScalar<K> for SparseTower3<K, LIN> {
    fn constant(c: f64) -> Self {
        Self {
            v: c,
            g: [0.0; K],
            h: [[0.0; K]; K],
            t3: [[[0.0; K]; K]; K],
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        let mut out = Self::constant(x);
        out.g[axis] = 1.0;
        out
    }
}

impl<const K: usize, const LIN: u32> gam_math::nested_dual::JetField for SparseTower3<K, LIN> {
    fn value(&self) -> f64 {
        self.v
    }
    // add / scale are UNIFORM-DENSE (applied to post-leaf dense results).
    fn add(&self, o: &Self) -> Self {
        let mut r = *self;
        r.v += o.v;
        for i in 0..K {
            r.g[i] += o.g[i];
            for j in 0..K {
                r.h[i][j] += o.h[i][j];
                for k in 0..K {
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
        for i in 0..K {
            o.g[i] *= s;
            for j in 0..K {
                o.h[i][j] *= s;
                for k in 0..K {
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
        for i in 0..K {
            let mut s = 0.0;
            s += a.v * b.g[i];
            s += a.g[i] * b.v;
            out.g[i] = s;
        }
        for i in 0..K {
            for j in 0..K {
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
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
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
        for i in 0..K {
            let mut s = 0.0;
            s += d[1] * self.g[i];
            out.g[i] = s;
        }
        for i in 0..K {
            for j in 0..K {
                let mut s = 0.0;
                if !h_block_is_zero(LIN, i, j) {
                    s += d[1] * self.h[i][j];
                }
                s += d[2] * self.g[i] * self.g[j];
                out.h[i][j] = s;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
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
pub(crate) struct SparseTower4<const K: usize, const LIN: u32> {
    pub(crate) v: f64,
    pub(crate) g: [f64; K],
    pub(crate) h: [[f64; K]; K],
    pub(crate) t3: [[[f64; K]; K]; K],
    pub(crate) t4: [[[[f64; K]; K]; K]; K],
}

impl<const K: usize, const LIN: u32> SparseTower4<K, LIN> {
    #[inline(always)]
    fn check_contract(&self) {
        for i in 0..K {
            for j in 0..K {
                if h_block_is_zero(LIN, i, j) {
                    assert!(
                        self.h[i][j] == 0.0,
                        "static-sparsity contract violated: h[{i}][{j}]={} != 0",
                        self.h[i][j]
                    );
                }
                for k in 0..K {
                    if t3_block_is_zero(LIN, i, j, k) {
                        assert!(
                            self.t3[i][j][k] == 0.0,
                            "static-sparsity contract violated: t3[{i}][{j}][{k}]={} != 0",
                            self.t3[i][j][k]
                        );
                    }
                    for l in 0..K {
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

}

impl<const K: usize, const LIN: u32> JetScalar<K> for SparseTower4<K, LIN> {
    fn constant(c: f64) -> Self {
        Self {
            v: c,
            g: [0.0; K],
            h: [[0.0; K]; K],
            t3: [[[0.0; K]; K]; K],
            t4: [[[[0.0; K]; K]; K]; K],
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        let mut out = Self::constant(x);
        out.g[axis] = 1.0;
        out
    }
}

impl<const K: usize, const LIN: u32> gam_math::nested_dual::JetField for SparseTower4<K, LIN> {
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        let mut r = *self;
        r.v += o.v;
        for i in 0..K {
            r.g[i] += o.g[i];
            for j in 0..K {
                r.h[i][j] += o.h[i][j];
                for k in 0..K {
                    r.t3[i][j][k] += o.t3[i][j][k];
                    for l in 0..K {
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
        for i in 0..K {
            o.g[i] *= s;
            for j in 0..K {
                o.h[i][j] *= s;
                for k in 0..K {
                    o.t3[i][j][k] *= s;
                    for l in 0..K {
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
        for i in 0..K {
            let mut s = 0.0;
            s += a.v * b.g[i];
            s += a.g[i] * b.v;
            out.g[i] = s;
        }
        for i in 0..K {
            for j in 0..K {
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
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
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
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
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
        for i in 0..K {
            let mut s = 0.0;
            s += d[1] * self.g[i];
            out.g[i] = s;
        }
        for i in 0..K {
            for j in 0..K {
                let mut s = 0.0;
                if !h_block_is_zero(LIN, i, j) {
                    s += d[1] * self.h[i][j];
                }
                s += d[2] * self.g[i] * self.g[j];
                out.h[i][j] = s;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
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
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
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

/// Read access to the third tensor of whichever order-≤3 tower a frame
/// declares, so the all-axes paths are generic over the tower's static
/// sparsity: the Gaussian frames elide the affine location blocks
/// ([`RIGID_LINEAR_MASK`]), the anchored frame is dense (gam#2923).
pub(crate) trait SparseThird<const K: usize> {
    fn t3(&self) -> &[[[f64; K]; K]; K];
}

/// Read access to the fourth tensor of a frame's order-≤4 tower.
pub(crate) trait SparseFourth<const K: usize> {
    fn t4(&self) -> &[[[[f64; K]; K]; K]; K];
}

impl<const K: usize, const LIN: u32> SparseThird<K> for SparseTower3<K, LIN> {
    #[inline(always)]
    fn t3(&self) -> &[[[f64; K]; K]; K] {
        &self.t3
    }
}

impl<const K: usize, const LIN: u32> SparseThird<K> for SparseTower4<K, LIN> {
    #[inline(always)]
    fn t3(&self) -> &[[[f64; K]; K]; K] {
        &self.t3
    }
}

impl<const K: usize, const LIN: u32> SparseFourth<K> for SparseTower4<K, LIN> {
    #[inline(always)]
    fn t4(&self) -> &[[[[f64; K]; K]; K]; K] {
        &self.t4
    }
}

/// Contract a `Tower3` third tensor with one primary-space direction —
/// `out[a][b] = Σ_c t3[a][b][c]·dir[c]` — exactly `Tower4::third_contracted`'s
/// arithmetic (same accumulation order), used by the build-once first-directional
/// path on the pruned [`SparseTower3`] towers.
#[inline]
pub(crate) fn tower3_third_contracted<const K: usize>(
    t3: &[[[f64; K]; K]; K],
    dir: &[f64; K],
) -> [[f64; K]; K] {
    let mut out = [[0.0; K]; K];
    for a in 0..K {
        for b in 0..K {
            let mut acc = 0.0;
            for c in 0..K {
                acc += t3[a][b][c] * dir[c];
            }
            out[a][b] = acc;
        }
    }
    out
}

// ── RowKernel<4> implementation ───────────────────────────────────────

pub(crate) struct SurvivalMarginalSlopeRowKernel<const P: usize, G: SlopeRowGeometry<P>> {
    pub(crate) family: SurvivalMarginalSlopeFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) slices: BlockSlices,
    pub(crate) geometry: std::marker::PhantomData<G>,
}

impl<const P: usize, G: SlopeRowGeometry<P>> SurvivalMarginalSlopeRowKernel<P, G> {
    pub(crate) fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Self {
        assert_eq!(
            family.slope_is_follow_up_varying(),
            G::FOLLOW_UP_VARYING,
            "row kernel primary frame does not match the family's slope layout",
        );
        assert_eq!(
            family.anchored_law_active(),
            G::ANCHORED,
            "row kernel primary frame does not match the family's latent law",
        );
        let slices = block_slices(&family, &block_states);
        Self {
            family,
            block_states,
            slices,
            geometry: std::marker::PhantomData,
        }
    }

    /// The slope block's `(primary, design)` channels for this frame.
    #[inline]
    pub(crate) fn slope_channels(&self) -> SlopeChannelDesigns<'_> {
        self.family.slope_layout.primary_channels()
    }
}

#[cfg(all(test, target_os = "linux"))]
mod rigid_row_admission_tests {
    use super::*;

    fn inputs(wi: f64, di: f64) -> RigidRowInputs<'static> {
        RigidRowInputs {
            row: 7,
            wi,
            wi_entry: wi,
            di,
            z_sum: 0.0,
            covariance_ones: 1.0,
            probit_scale: 1.0,
            qd1_lower: 0.0,
            anchor: None,
        }
    }

    fn admit(primaries: [f64; 4], inputs: &RigidRowInputs) -> Result<(), String> {
        let [neg_eta0, neg_eta1, adjusted_derivative] =
            rigid_row_admission_witnesses::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                &primaries, inputs,
            );
        validate_rigid_row_admission::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
            primaries[2],
            inputs,
            neg_eta0,
            neg_eta1,
            adjusted_derivative,
        )
    }

    #[test]
    fn scalar_gpu_admission_witnesses_match_cpu_signed_margin_domain() {
        for primaries in [[f64::NAN, 0.0, 1.0, 0.0], [0.0, f64::INFINITY, 1.0, 0.0]] {
            let error = admit(primaries, &inputs(1.0, 0.0))
                .expect_err("active non-finite signed margin must be rejected");
            assert!(error.contains("non-finite signed margin"));
        }

        admit(
            [f64::NEG_INFINITY, f64::NEG_INFINITY, 1.0, 0.0],
            &inputs(1.0, 0.0),
        )
        .expect("positive-infinity signed margins are the admitted saturated tail");
        admit([f64::NAN, f64::NAN, 1.0, 0.0], &inputs(0.0, 0.0))
            .expect("zero-weight margins do not contribute to the row");
    }
}

/// The row's primary point in the frame `G` declares.
///
/// The three location channels come from the family's dynamic-`q` geometry. The
/// slope channels come from the slope layout: one channel when the slope is
/// time-constant, three (entry, exit, exit-rate) when it varies along follow-up
/// (gam#2765).
pub(crate) fn rigid_row_kernel_primaries<const P: usize, G: SlopeRowGeometry<P>>(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
) -> Result<[f64; P], String> {
    let q_geom = family.row_dynamic_q_values(row, block_states)?;
    let mut primaries = [0.0; P];
    primaries[PRIMARY_Q0] = q_geom.q0;
    primaries[PRIMARY_Q1] = q_geom.q1;
    primaries[PRIMARY_QD1] = q_geom.qd1;
    let slope = family.row_slope_channels(row, block_states)?;
    if G::FOLLOW_UP_VARYING {
        primaries[PRIMARY_SLOPE] = slope.entry;
        primaries[PRIMARY_SLOPE_EXIT] = slope.exit;
        primaries[PRIMARY_SLOPE_RATE] = slope.rate;
    } else {
        primaries[PRIMARY_SLOPE] = slope.exit;
    }
    Ok(primaries)
}

/// The scalar-independent per-row inputs the generic rigid row NLL
/// ([`rigid_row_nll`]) consumes: the f64 quantities computed ONCE per row and
/// reused across every [`JetScalar`] instantiation (value/grad/Hessian, the
/// contracted third/fourth, and the dense tower oracle/all-axes path).
pub(crate) struct RigidRowInputs<'a> {
    pub(crate) row: usize,
    pub(crate) wi: f64,
    /// The weight of the entry survival factor `log Φ(−η₀)`: `wi` for a
    /// delayed entry and `0` for a row entering at the time origin, where
    /// `S(0) = 1` and the factor is absent (gnomon#2336).
    pub(crate) wi_entry: f64,
    pub(crate) di: f64,
    pub(crate) z_sum: f64,
    pub(crate) covariance_ones: f64,
    pub(crate) probit_scale: f64,
    pub(crate) qd1_lower: f64,
    /// The declared finite law this row's index is anchored on (gam#2923):
    /// `Some` exactly when the family runs the anchored frame, in which case
    /// the marginal identity is solved on it instead of lowered in closed
    /// form. `None` is the standard-normal law and the Gaussian closed form.
    pub(crate) anchor: Option<AnchorRowContext<'a>>,
}

/// Resolve the row's scalar inputs (shared-score summary, probit scale,
/// monotonicity floor, declared law). Pure f64 — no jet arithmetic.
pub(crate) fn rigid_row_inputs<'a>(
    family: &'a SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
    context: &str,
) -> Result<RigidRowInputs<'a>, String> {
    let (z_sum, covariance_ones) = family.exact_shared_score_summary(row, block_states, context)?;
    Ok(RigidRowInputs {
        row,
        wi: family.weights[row],
        wi_entry: family.entry_weight(row),
        di: family.event[row],
        z_sum,
        covariance_ones,
        probit_scale: family.probit_frailty_scale(),
        qd1_lower: family.time_derivative_lower_bound(),
        anchor: family.latent_law.as_ref().map(|law| law.row_context(row)),
    })
}

/// Admission witnesses for a primary frame, obtained by applying its feature
/// map to the sliced witness surface emitted from the sole rigid likelihood
/// declaration.
#[inline(always)]
pub(crate) fn rigid_row_admission_witnesses<const P: usize, G: SlopeRowGeometry<P>>(
    primaries: &[f64; P],
    inputs: &RigidRowInputs,
) -> [f64; 3] {
    let features = G::feature_frame(primaries, inputs);
    rigid_feature_frame_witnesses(
        &features,
        inputs.probit_scale,
        follow_up_varying_flag::<P, G>(),
    )
}

/// The survival marginal-slope row negative log-likelihood, evaluated over a
/// generic [`JetScalar`] after mechanically constructing the nine semantic
/// features consumed by the sole `rigid_feature_program` AST. The same
/// expression therefore yields every derivative channel a consumer needs
/// (#736/#932 single-source contract):
///
/// * `S = Order2<P>`  → `(v, g, H)` (inner Newton / `row_kernel`),
/// * `S = OneSeed<P>` → contracted third `Σ_c ℓ_{abc} dir_c`
///   (`row_third_contracted`),
/// * `S = TwoSeed<P>` → contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d`
///   (`row_fourth_contracted`),
/// * `S = Tower4<P>`  → the full dense `(v,g,H,t3,t4)` oracle / #979 all-axes
///   build-once truth (via [`gam_math::jet_tower::program_full_tower`]).
///
/// The feature map belongs to the frame `G`; all probability algebra and
/// special-function composition lives only in the feature program.
pub(crate) fn rigid_row_nll<const P: usize, G: SlopeRowGeometry<P>, S: JetScalar<P>>(
    vars: &[S; P],
    inputs: &RigidRowInputs,
) -> Result<S, String> {
    let features = G::feature_frame(vars, inputs);
    let (nll, [neg_eta0, neg_eta1, adjusted_derivative]) =
        rigid_feature_frame_program::<P, S>(
            &features,
            inputs.wi,
            inputs.wi_entry,
            inputs.di,
            inputs.probit_scale,
            follow_up_varying_flag::<P, G>(),
        );

    validate_rigid_row_admission::<P, G>(
        vars[PRIMARY_QD1].value(),
        inputs,
        neg_eta0,
        neg_eta1,
        adjusted_derivative,
    )?;
    Ok(nll)
}

/// `∂/∂z` of the rigid row NLL's PRIMARY gradient: the mixed `(primary, latent
/// score)` second derivative, before any block-Jacobian scatter (gam#2768).
///
/// Owned by the frame ([`SlopeRowGeometry::row_primary_mixed_in_z`]); this is
/// the frame-generic entry every caller reads it through.
#[inline]
pub(crate) fn rigid_row_primary_mixed_in_z<const P: usize, G: SlopeRowGeometry<P>>(
    primaries: &[f64; P],
    inputs: &RigidRowInputs,
) -> Result<[f64; P], String> {
    G::row_primary_mixed_in_z(primaries, inputs)
}

/// Direct value/gradient/Hessian lowering of the canonical nine-feature row
/// program followed by the universal second-order pullback into the frame `G`.
/// Owned by the frame ([`SlopeRowGeometry::row_order2`]); this is the
/// frame-generic entry every caller reads it through.
#[inline(always)]
pub(crate) fn rigid_row_order2<const P: usize, G: SlopeRowGeometry<P>>(
    primaries: &[f64; P],
    inputs: &RigidRowInputs,
) -> Result<(f64, [f64; P], [[f64; P]; P]), String> {
    G::row_order2(primaries, inputs)
}

/// Apply the scalar domain contract shared by the ordinary row evaluator and
/// the already-admitted GPU gather. The three witnesses come from the same
/// `row_program!` declaration: directly from the generic program on CPU and
/// from its dependency-sliced scalar witness schedule during GPU admission.
///
/// Inlined into every row: the check is three compares on the success path,
/// and the error constructors below are cold and out of line, so the row
/// loop never carries their formatting (the previous out-of-line call per
/// row was part of `RIGID-SCALAR-932`'s deficit).
#[inline(always)]
pub(crate) fn validate_rigid_row_admission<const P: usize, G: SlopeRowGeometry<P>>(
    qd1: f64,
    inputs: &RigidRowInputs,
    neg_eta0: f64,
    neg_eta1: f64,
    adjusted_derivative: f64,
) -> Result<(), String> {
    let RigidRowInputs {
        row,
        wi,
        di,
        qd1_lower,
        ..
    } = *inputs;
    if survival_derivative_guard_violated(qd1, qd1_lower) {
        return Err(monotonicity_violation(row, qd1, qd1_lower, adjusted_derivative));
    }

    // `q′(t) ≥ derivative_guard` is the MARGINAL monotonicity constraint — the
    // population survival index has to be increasing — and the inner solver
    // holds it as a linear inequality on the time block. With a time-constant
    // slope it also implies the likelihood-domain condition `η′₁ > 0`, because
    // `η′₁ = q′·c` and `c ≥ 1`; that implication is exactly what a
    // follow-up-varying slope breaks, since `η′₁` picks up `q₁·c′₁ + ṡ·ż` which
    // carry no sign (gam#2765). `log η′₁` is in the row program, so the honest
    // place for the extra condition is the likelihood domain: outside it the
    // objective is `+∞` and the step is rejected, rather than a `NaN` reaching
    // the solver. On the static frame this branch is unreachable given the
    // guard above, so it changes no existing fit.
    if inputs.di != 0.0 && !(adjusted_derivative > 0.0) {
        return Err(nonpositive_transformed_derivative(row, G::NAME, adjusted_derivative, qd1));
    }

    // Mirror the exact closed-form contract
    // (`signed_probit_neglog_derivatives_up_to_fourth`): the saturated `+∞`
    // tail is the legitimate zero-survival limit, but `-∞`/NaN signed margins
    // are domain failures that must surface as an error rather than being
    // masked into a NaN/∞-laden derivative stack by `unary_derivatives_neglog_phi`.
    // The guard respects zero weight (those terms drop out entirely).
    // A weighted margin must be finite or `+inf`; that is `margin > -inf`,
    // one compare, false for NaN as every comparison with NaN is.
    // The entry margin is read only when the row carries an entry factor; a row
    // entering at the time origin has none (gnomon#2336).
    if inputs.wi_entry != 0.0 && !(neg_eta0 > f64::NEG_INFINITY) {
        return Err(nonfinite_signed_margin(row, G::NAME, neg_eta0));
    }
    if wi * (1.0 - di) != 0.0 && !(neg_eta1 > f64::NEG_INFINITY) {
        return Err(nonfinite_signed_margin(row, G::NAME, neg_eta1));
    }
    Ok(())
}

#[cold]
#[inline(never)]
fn monotonicity_violation(row: usize, qd1: f64, qd1_lower: f64, adjusted_derivative: f64) -> String {
    SurvivalMarginalSlopeError::MonotonicityViolation {
        reason: format!(
            "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={adjusted_derivative:.3e}"
        ),
    }
    .into()
}

#[cold]
#[inline(never)]
fn nonpositive_transformed_derivative(
    row: usize,
    frame: &str,
    adjusted_derivative: f64,
    qd1: f64,
) -> String {
    SurvivalMarginalSlopeError::MonotonicityViolation {
        reason: format!(
            "survival marginal-slope transformed time derivative must be positive at row {row} on the {frame} frame: got {adjusted_derivative:.3e} (raw time derivative={qd1:.3e})"
        ),
    }
    .into()
}

#[cold]
#[inline(never)]
fn nonfinite_signed_margin(row: usize, frame: &str, margin: f64) -> String {
    SurvivalMarginalSlopeError::NumericalFailure {
        reason: format!(
            "non-finite signed margin in rigid survival marginal-slope row tower at row {row} on the {frame} frame: {margin} (on a declared latent law this is an anchor solve that did not converge)"
        ),
    }
    .into()
}

/// #932: the canonical single-source seam. The row NLL is written ONCE as
/// [`rigid_row_nll`]; this exposes it through [`gam_math::jet_tower::RowProgram`]
/// so the `RowKernel` derivative channels below derive mechanically from `eval`
/// via the `program_*` helpers. Instantiating this same method at `S = Tower4`
/// through [`gam_math::jet_tower::program_full_tower`] supplies the dense oracle;
/// there is no second tower-only program surface.
impl<const P: usize, G: SlopeRowGeometry<P>> gam_math::jet_tower::RowProgram<P>
    for SurvivalMarginalSlopeRowKernel<P, G>
{
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn primaries(&self, row: usize) -> Result<[f64; P], String> {
        rigid_row_kernel_primaries::<P, G>(&self.family, &self.block_states, row)
    }

    fn eval<S: JetScalar<P>>(&self, row: usize, p: &[S; P]) -> Result<S, String> {
        let inputs = rigid_row_inputs(
            &self.family,
            &self.block_states,
            row,
            "survival marginal-slope rigid row program",
        )?;
        rigid_row_nll::<P, G, S>(p, &inputs)
    }
}

impl<const P: usize, G: SlopeRowGeometry<P>> RowKernel<P>
    for SurvivalMarginalSlopeRowKernel<P, G>
{
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; P], [[f64; P]; P]), String> {
        // #932: the macro lowers value/gradient/Hessian directly from the SAME
        // parsed SSA graph as `rigid_row_nll` and CUDA. No dense higher-order
        // tower, forward order-two jet, dependency mask, or hand chain rule is
        // built on this inner-Newton hot path.
        let inputs = rigid_row_inputs(
            &self.family,
            &self.block_states,
            row,
            "survival marginal-slope rigid row kernel",
        )?;
        let p = rigid_row_kernel_primaries::<P, G>(&self.family, &self.block_states, row)?;
        rigid_row_order2::<P, G>(&p, &inputs)
    }

    /// Batched all-rows `(nll, grad, hess)` via the A100 NVRTC survival row-jet
    /// (#932-GPU). Gathers every row's primaries + scalar inputs, then calls the
    /// device dispatcher ([`crate::gpu_kernels::survival_rowjet`]) which runs the
    /// same order-2 `rigid_row_nll` lowering for all `n` rows in parallel. Linux
    /// batches below device admission return `None` and use the ordinary per-row
    /// cache path. Once admitted, probe/compile/launch/transfer failures are
    /// returned and never hidden by a CPU retry.
    ///
    /// The host gather applies the canonical derivative and signed-margin domain
    /// checks before launch because the device kernel consumes already-admitted
    /// primaries.
    fn batched_value_grad_hess_all(
        &self,
    ) -> Option<Result<(Vec<f64>, Vec<[f64; P]>, Vec<[[f64; P]; P]>), String>> {
        use crate::gpu_kernels::survival_rowjet::survival_rigid_row_vgh_device_selected;

        // The device pullback is written for the four-primary Gaussian frame. A
        // follow-up-varying slope, or a declared latent law, takes the ordinary
        // per-row CPU path rather than a silently different lowering.
        if G::FOLLOW_UP_VARYING || G::ANCHORED {
            return None;
        }
        let n = self.family.n;
        match survival_rigid_row_vgh_device_selected(n) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(error) => return Some(Err(error)),
        }

        #[cfg(target_os = "linux")]
        {
            use crate::gpu_kernels::survival_rowjet::{SurvivalRowInputs, survival_rigid_row_vgh};
            let probit_scale = self.family.probit_frailty_scale();
            // Gather per-row inputs in parallel (the pure-f64 score summary + primary
            // projections — the same quantities the per-row path computes).
            let gather: Result<Vec<SurvivalRowInputs>, String> = (0..n)
                .into_par_iter()
                .map(|row| {
                    let p =
                        rigid_row_kernel_primaries::<P, G>(&self.family, &self.block_states, row)?;
                    let inputs = rigid_row_inputs(
                        &self.family,
                        &self.block_states,
                        row,
                        "survival marginal-slope rigid row kernel (batched)",
                    )?;
                    let [neg_eta0, neg_eta1, adjusted_derivative] =
                        rigid_row_admission_witnesses::<P, G>(&p, &inputs);
                    validate_rigid_row_admission::<P, G>(
                        p[PRIMARY_QD1],
                        &inputs,
                        neg_eta0,
                        neg_eta1,
                        adjusted_derivative,
                    )?;
                    Ok(SurvivalRowInputs {
                        // `G::FOLLOW_UP_VARYING` is false on this branch, so
                        // `P == STATIC_SLOPE_PRIMARIES` and this is a copy, not
                        // a truncation.
                        primaries: std::array::from_fn(|axis| p[axis]),
                        wi: inputs.wi,
                        wi_entry: inputs.wi_entry,
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
            let mut grads = vec![[0.0_f64; P]; n];
            let mut hesss = vec![[[0.0_f64; P]; P]; n];
            for row in 0..n {
                for a in 0..STATIC_SLOPE_PRIMARIES {
                    grads[row][a] = ch.grad[row * STATIC_SLOPE_PRIMARIES + a];
                    for b in 0..STATIC_SLOPE_PRIMARIES {
                        hesss[row][a][b] = ch.hess
                            [row * STATIC_SLOPE_PRIMARIES * STATIC_SLOPE_PRIMARIES
                                + a * STATIC_SLOPE_PRIMARIES
                                + b];
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

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; P] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.slices.time.clone()]);
        let d_marginal = d_beta.slice(s![self.slices.marginal.clone()]);
        let d_slope = d_beta.slice(s![self.slices.slope.clone()]);
        let mut action = [0.0; P];
        let marginal = self.family.marginal_design.dot_row_view(row, d_marginal);
        action[PRIMARY_Q0] = self.family.design_entry.dot_row_view(row, d_time) + marginal;
        action[PRIMARY_Q1] = self.family.design_exit.dot_row_view(row, d_time) + marginal;
        action[PRIMARY_QD1] = self
            .family
            .design_derivative_exit
            .dot_row_view(row, d_time);
        for &(primary, design) in self.slope_channels().as_slice() {
            action[primary] = design.dot_row_view(row, d_slope);
        }
        action
    }

    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if factor.nrows() != self.slices.total {
            return None;
        }
        self.tiled_jacobian_action_matrix(factor)
    }

    fn jacobian_action_matrix_rows(
        &self,
        factor: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        assert_eq!(
            factor.nrows(),
            self.slices.total,
            "survival marginal-slope tiled Jacobian factor width must match coefficients",
        );
        // Block-tiled build for one row-tile: dense designs slice to a
        // contiguous row block and GEMM (`fast_ab`), operator/sparse designs
        // fall to a row-local dot over the range. Bounds peak memory to the
        // tile while keeping BLAS-3 on the materialized designs.
        let b = end.saturating_sub(start);
        self.assemble_jf(factor, b, |design, factor_block| {
            crate::row_kernel::row_kernel_design_jf_rows(design, factor_block, start, end)
        })
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; P], out: &mut [f64]) {
        {
            let mut time = ndarray::ArrayViewMut1::from(&mut out[self.slices.time.clone()]);
            self.family
                .design_entry
                .axpy_row_into(row, v[PRIMARY_Q0], &mut time)
                .expect("time entry axpy dim mismatch");
            self.family
                .design_exit
                .axpy_row_into(row, v[PRIMARY_Q1], &mut time)
                .expect("time exit axpy dim mismatch");
            self.family
                .design_derivative_exit
                .axpy_row_into(row, v[PRIMARY_QD1], &mut time)
                .expect("time deriv axpy dim mismatch");
        }
        {
            let mut marginal = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[PRIMARY_Q0] + v[PRIMARY_Q1], &mut marginal)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut slope = ndarray::ArrayViewMut1::from(&mut out[self.slices.slope.clone()]);
            for &(primary, design) in self.slope_channels().as_slice() {
                design
                    .axpy_row_into(row, v[primary], &mut slope)
                    .expect("slope axpy dim mismatch");
            }
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; P]; P], target: &mut Array2<f64>) {
        let mut h_arr = Array2::<f64>::zeros((P, P));
        for a in 0..P {
            for b in 0..P {
                h_arr[[a, b]] = h[a][b];
            }
        }
        self.family
            .add_pullback_primary_hessian(target, row, &self.slices, &h_arr);
    }

    /// Storage-aware block assembly of the rigid survival joint Hessian.
    ///
    /// Every coefficient block is the weighted cross-product
    ///
    /// ```text
    /// Σᵢ wᵢ xᵢ yᵢᵀ = Xᵀ diag(w) Y.
    /// ```
    ///
    /// Every pair whose dense row panels fit the fixed working-set budget uses
    /// bounded, row-chunked BLAS-3 Grams, irrespective of its source storage.
    /// A sparse pair stays on the sparse-aware row-outer primitive only when
    /// densifying both panels would exceed that budget. This decision is made
    /// per block pair, not for the whole Hessian: sparse is a storage choice,
    /// not a reason to force a small 800×12 derivative design through thousands
    /// of scalar row-view updates.
    /// Operator panels are materialized under a fixed byte budget, while
    /// materialized designs are borrowed as zero-copy views. The method claims
    /// only the full-data unit-weight row measure; Horvitz–Thompson row sets
    /// retain their explicit weighted generic path.
    fn hessian_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        row_hessians: &[[[f64; P]; P]],
    ) -> Option<Result<Array2<f64>, String>> {
        if !matches!(rows, crate::row_kernel::RowSet::All) {
            return None;
        }
        if row_hessians.len() != self.family.n {
            return Some(Err(format!(
                "survival marginal-slope hessian_dense_override row-Hessian length mismatch: \
                 got {}, expected {}",
                row_hessians.len(),
                self.family.n,
            )));
        }

        let time_designs = [
            &self.family.design_entry,
            &self.family.design_exit,
            &self.family.design_derivative_exit,
        ];
        let marginal_design = &self.family.marginal_design;
        let slope_channels = self.slope_channels();
        let slope_designs = slope_channels.as_slice();

        Some((|| {
            fn dense_chunk<'a>(
                design: &'a DesignMatrix,
                rows: std::ops::Range<usize>,
                label: &str,
            ) -> Result<ndarray::CowArray<'a, f64, ndarray::Ix2>, String> {
                match design.as_dense_ref() {
                    Some(full) => Ok(full.slice(s![rows, ..]).into()),
                    None => design
                        .try_row_chunk(rows.clone())
                        .map(Into::into)
                        .map_err(|error| {
                            format!(
                                "survival marginal-slope dense Hessian {label} \
                                 try_row_chunk({}..{}): {error}",
                                rows.start, rows.end,
                            )
                        }),
                }
            }

            fn add_weighted_cross(
                left: &DesignMatrix,
                right: &DesignMatrix,
                weights: &Array1<f64>,
                mut target: ndarray::ArrayViewMut2<'_, f64>,
                label: &str,
            ) -> Result<(), String> {
                let n = weights.len();
                if left.nrows() != n || right.nrows() != n {
                    return Err(format!(
                        "survival marginal-slope Hessian {label} row mismatch: \
                         left={} right={} weights={n}",
                        left.nrows(),
                        right.nrows(),
                    ));
                }
                // Storage does not determine the arithmetic schedule. A
                // sparse panel that is small enough to fit the same bounded
                // dense working set is materialized once per chunk and closed
                // by BLAS-3; only an over-budget sparse pair streams row outers.
                const PANEL_BUDGET_BYTES: usize = 64 * 1024 * 1024;
                const MAX_CHUNK_ROWS: usize = 8_192;
                let columns_per_row = left
                    .ncols()
                    .saturating_add(right.ncols())
                    .max(1);
                let bytes_per_row =
                    columns_per_row.saturating_mul(std::mem::size_of::<f64>());
                let full_panel_bytes = n.saturating_mul(bytes_per_row);
                let sparse_requires_streaming = (left.is_sparse() || right.is_sparse())
                    && full_panel_bytes > PANEL_BUDGET_BYTES;
                if sparse_requires_streaming {
                    for row in 0..n {
                        let weight = weights[row];
                        if weight == 0.0 {
                            continue;
                        }
                        left.row_outer_into_view(
                            row,
                            right,
                            weight,
                            target.view_mut(),
                        )
                        .map_err(|error| {
                            format!(
                                "survival marginal-slope sparse Hessian {label} row {row}: {error}"
                            )
                        })?;
                    }
                    return Ok(());
                }

                // Bound the two simultaneous dense/operator/sparse panels. A
                // wide pair gets shorter chunks automatically; a narrow pair
                // gets at most 8K rows so each Gram is cache-friendly.
                let chunk_rows = (PANEL_BUDGET_BYTES / bytes_per_row)
                    .max(1)
                    .min(MAX_CHUNK_ROWS);
                // Each chunk's Gram is closed on its own worker with a
                // sequential GEMM, then added into the target in chunk order.
                // Issued from a serial chunk loop, every GEMM fanned out to the
                // whole pool and the pool spent the assembly waiting at faer's
                // barrier (gnomon#2337). The additions below keep the serial
                // loop's order, so the assembled block is unchanged.
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                let starts: Vec<usize> = (0..n).step_by(chunk_rows).collect();
                let grams = starts
                    .into_par_iter()
                    .map(|start| -> Result<Array2<f64>, String> {
                        gam_problem::with_nested_parallel(|| {
                            let end = (start + chunk_rows).min(n);
                            let left_chunk =
                                dense_chunk(left, start..end, &format!("{label}/left"))?;
                            let right_chunk =
                                dense_chunk(right, start..end, &format!("{label}/right"))?;
                            let local_weights = weights.slice(s![start..end]).to_owned();
                            Ok(gam_linalg::faer_ndarray::fast_xt_diag_y(
                                &left_chunk,
                                &local_weights,
                                &right_chunk,
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                for gram in grams {
                    target.scaled_add(1.0, &gram);
                }
                Ok(())
            }

            let n = row_hessians.len();
            for (label, design) in [
                ("time-entry", time_designs[0]),
                ("time-exit", time_designs[1]),
                ("time-derivative", time_designs[2]),
                ("marginal", marginal_design),
            ]
            .into_iter()
            .chain(
                slope_designs
                    .iter()
                    .map(|&(_, design)| ("slope", design)),
            ) {
                if design.nrows() != n {
                    return Err(format!(
                        "survival marginal-slope dense Hessian {label} rows={} != hessians={n}",
                        design.nrows(),
                    ));
                }
            }
            let weights: [[Array1<f64>; P]; P] = std::array::from_fn(|primary_a| {
                std::array::from_fn(|primary_b| {
                    Array1::from_iter(
                        row_hessians
                            .iter()
                            .map(|hessian| hessian[primary_a][primary_b]),
                    )
                })
            });
            let mut dense =
                Array2::<f64>::zeros((self.slices.total, self.slices.total));

            for primary_a in 0..3 {
                for primary_b in 0..3 {
                    add_weighted_cross(
                        time_designs[primary_a],
                        time_designs[primary_b],
                        &weights[primary_a][primary_b],
                        dense.slice_mut(s![
                            self.slices.time.clone(),
                            self.slices.time.clone()
                        ]),
                        "time/time",
                    )?;
                }
            }

            let mm_weight =
                &weights[0][0] + &weights[0][1] + &weights[1][0] + &weights[1][1];
            add_weighted_cross(
                marginal_design,
                marginal_design,
                &mm_weight,
                dense.slice_mut(s![
                    self.slices.marginal.clone(),
                    self.slices.marginal.clone()
                ]),
                "marginal/marginal",
            )?;
            for &(primary_left, design_left) in slope_designs {
                for &(primary_right, design_right) in slope_designs {
                    add_weighted_cross(
                        design_left,
                        design_right,
                        &weights[primary_left][primary_right],
                        dense.slice_mut(s![
                            self.slices.slope.clone(),
                            self.slices.slope.clone()
                        ]),
                        "slope/slope",
                    )?;
                }
                let mg_weight =
                    &weights[PRIMARY_Q0][primary_left] + &weights[PRIMARY_Q1][primary_left];
                add_weighted_cross(
                    marginal_design,
                    design_left,
                    &mg_weight,
                    dense.slice_mut(s![
                        self.slices.marginal.clone(),
                        self.slices.slope.clone()
                    ]),
                    "marginal/slope",
                )?;
            }

            for primary_a in 0..3 {
                for &(primary_slope, design_slope) in slope_designs {
                    add_weighted_cross(
                        time_designs[primary_a],
                        design_slope,
                        &weights[primary_a][primary_slope],
                        dense.slice_mut(s![
                            self.slices.time.clone(),
                            self.slices.slope.clone()
                        ]),
                        "time/slope",
                    )?;
                }
                let tm_weight = &weights[primary_a][0] + &weights[primary_a][1];
                add_weighted_cross(
                    time_designs[primary_a],
                    marginal_design,
                    &tm_weight,
                    dense.slice_mut(s![
                        self.slices.time.clone(),
                        self.slices.marginal.clone()
                    ]),
                    "time/marginal",
                )?;
            }

            // Match the rigid pullback's symmetry contract: the primary
            // Hessian is symmetric, and each off-diagonal coefficient block is
            // assembled once then mirrored exactly.
            for (upper_rows, upper_columns, lower_rows, lower_columns) in [
                (
                    self.slices.marginal.clone(),
                    self.slices.slope.clone(),
                    self.slices.slope.clone(),
                    self.slices.marginal.clone(),
                ),
                (
                    self.slices.time.clone(),
                    self.slices.slope.clone(),
                    self.slices.slope.clone(),
                    self.slices.time.clone(),
                ),
                (
                    self.slices.time.clone(),
                    self.slices.marginal.clone(),
                    self.slices.marginal.clone(),
                    self.slices.time.clone(),
                ),
            ] {
                let upper = dense
                    .slice(s![upper_rows, upper_columns])
                    .to_owned();
                dense
                    .slice_mut(s![lower_rows, lower_columns])
                    .assign(&upper.t());
            }

            static HESSIAN_STORAGE_LOGGED: std::sync::Once = std::sync::Once::new();
            HESSIAN_STORAGE_LOGGED.call_once(|| {
                log::info!(
                    "[STAGE] survival marginal-slope hybrid Hessian assembly: \
                     sparse=({},{},{},{},{}) dims=({},{},{},{},{})",
                    time_designs[0].is_sparse(),
                    time_designs[1].is_sparse(),
                    time_designs[2].is_sparse(),
                    marginal_design.is_sparse(),
                    slope_designs.iter().any(|&(_, d)| d.is_sparse()),
                    time_designs[0].ncols(),
                    time_designs[1].ncols(),
                    time_designs[2].ncols(),
                    marginal_design.ncols(),
                    slope_designs[0].1.ncols(),
                );
            });
            Ok(dense)
        })())
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; P]; P], diag: &mut [f64]) {
        let designs: [(usize, &DesignMatrix); 3] = [
            (PRIMARY_Q0, &self.family.design_entry),
            (PRIMARY_Q1, &self.family.design_exit),
            (PRIMARY_QD1, &self.family.design_derivative_exit),
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
            let alpha =
                h[PRIMARY_Q0][PRIMARY_Q0] + 2.0 * h[PRIMARY_Q0][PRIMARY_Q1] + h[PRIMARY_Q1][PRIMARY_Q1];
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, alpha, &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let channels = self.slope_channels();
            let slope_designs = channels.as_slice();
            for (index, &(primary, design)) in slope_designs.iter().enumerate() {
                let mut gd =
                    ndarray::ArrayViewMut1::from(&mut diag[self.slices.slope.clone()]);
                design
                    .squared_axpy_row_into(row, h[primary][primary], &mut gd)
                    .expect("slope squared_axpy dim mismatch");
                for &(other_primary, other_design) in &slope_designs[index + 1..] {
                    design
                        .crossdiag_axpy_row_into(
                            row,
                            other_design,
                            2.0 * h[primary][other_primary],
                            &mut gd,
                        )
                        .expect("slope crossdiag dim mismatch");
                }
            }
        }
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
    /// Build each row's `t3` once and assemble the pullbacks as weighted Grams.
    /// This preserves the derivatives, with floating-point reassociation of
    /// the row sums checked against the scalar per-axis implementation.
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
        Some(self.directional_derivative_all_axes_build_once())
    }

    /// Batched all-axes SECOND directional derivative of the joint Hessian for
    /// the rigid survival marginal-slope kernel (gam#979): the outer-REML
    /// Jeffreys `H_Φ` drift analogue of the first-order override above.
    ///
    /// With `d_beta_u` fixed and the second direction sweeping every canonical
    /// axis, the generic per-axis path runs `p` full-data sweeps each evaluating
    /// the per-row two-seed program scalar through `row_fourth_contracted`.
    /// Contract each row's `t4` with the fixed direction once, then assemble
    /// every swept axis through the same weighted-Gram path as first order.
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

impl<const P: usize, G: SlopeRowGeometry<P>> SurvivalMarginalSlopeRowKernel<P, G> {
    /// Assemble the `(n_out × P·rank)` joint Jacobian-action projection `Jᵢ · F`
    /// from the four primary axes — `[entry+marginal | exit+marginal |
    /// derivative | slope]` — given a per-axis builder `axis(design,
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
        let f_slope = factor.slice(s![self.slices.slope.clone(), ..]);

        let jf_marginal = axis(&self.family.marginal_design, f_marginal);
        let mut axis0 = axis(&self.family.design_entry, f_time);
        axis0 += &jf_marginal;
        let mut axis1 = axis(&self.family.design_exit, f_time);
        axis1 += &jf_marginal;
        let axis2 = axis(&self.family.design_derivative_exit, f_time);
        // One slot per primary, filled by index rather than by push order, so a
        // frame whose slope owns three primaries cannot silently pack them into
        // the wrong axes.
        let mut slots: [Option<Array2<f64>>; P] = std::array::from_fn(|_| None);
        slots[PRIMARY_Q0] = Some(axis0);
        slots[PRIMARY_Q1] = Some(axis1);
        slots[PRIMARY_QD1] = Some(axis2);
        for &(primary, design) in self.slope_channels().as_slice() {
            slots[primary] = Some(axis(design, f_slope));
        }
        let axes: [(usize, Array2<f64>); P] = std::array::from_fn(|primary| {
            (
                primary,
                slots[primary].take().expect(
                    "every primary of the frame owns exactly one J·F axis: the three location \
                     channels plus the slope layout's follow-up channels",
                ),
            )
        });
        crate::row_kernel::row_kernel_pack_jf_axes::<P>(n_out, rank, axes)
    }

    /// The whole-projection `J · F` (`n × P·rank`), built one row tile at a time
    /// with [`Self::assemble_jf`]'s channel map. Each tile forms every channel's
    /// product through the design's `row_chunk_matmul_into` (its rows times the
    /// factor block in one sequential GEMM, unless an operator keeps the
    /// association of its own `apply`), adds the marginal block, formed once per
    /// tile, into both location channels, and writes its own rows of the output.
    ///
    /// The former build formed each channel over all rows, paid one full matvec
    /// per factor column on an operator-backed design, then packed five
    /// `n × rank` blocks and added the marginal block into the output on one
    /// thread: 9.7 s of the 12.1 s of single-threaded J·F time on the AoU death
    /// fit (gnomon#2337). Tiles write disjoint rows, so the result does not
    /// depend on the pool width, and a dense design's entries are the same
    /// k-ordered GEMM sums as before.
    fn tiled_jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        let governor = gam_runtime::resource::MemoryGovernor::global();
        self.tiled_jacobian_action_matrix_within(
            factor,
            &|bytes| {
                governor
                    .try_reserve(bytes, "survival marginal-slope J·F row tiles")
                    .ok()
            },
            &|| governor.remaining_bytes(),
        )
    }

    /// [`Self::tiled_jacobian_action_matrix`] with its tile buffers reserved
    /// through `reserve`, which answers a byte request with a held reservation
    /// or `None` for a refusal.
    ///
    /// Every tile in flight holds its buffers' reservation. On a refusal the
    /// build first runs fewer tiles at once, then halves the tile height, down to
    /// 32 rows. Tiles are near-equal slices of at least that many rows, so no
    /// GEMM runs on a sliver, and on a dense design neither the tile height nor
    /// the number in flight changes a bit of the result. When not even one
    /// minimal tile is admitted, or a design cannot produce its rows, the build
    /// returns `None`, the caller's cue for its per-row route,
    /// and says so once per fit with the refused tile's size and the budget left.
    pub(super) fn tiled_jacobian_action_matrix_within(
        &self,
        factor: ArrayView2<'_, f64>,
        reserve: &dyn Fn(usize) -> Option<gam_runtime::resource::MemoryReservation>,
        remaining: &dyn Fn() -> usize,
    ) -> Option<Array2<f64>> {
        // Tile height to aim for, and the least it is halved to under refusal.
        const JF_TILE_ROWS: usize = 256;
        const JF_MIN_TILE_ROWS: usize = 32;

        let n = self.family.n;
        let rank = factor.ncols();
        if rank == 0 {
            return Some(Array2::<f64>::zeros((n, 0)));
        }
        let block = |range: std::ops::Range<usize>| {
            factor.slice(s![range, ..]).as_standard_layout().into_owned()
        };
        let f_time = block(self.slices.time.clone());
        let f_marginal = block(self.slices.marginal.clone());
        let f_slope = block(self.slices.slope.clone());
        // (design, factor block, adds the marginal block), one per primary,
        // filled by index as `assemble_jf` fills its slots.
        let mut slots: [Option<(&DesignMatrix, &Array2<f64>, bool)>; P] =
            std::array::from_fn(|_| None);
        slots[PRIMARY_Q0] = Some((&self.family.design_entry, &f_time, true));
        slots[PRIMARY_Q1] = Some((&self.family.design_exit, &f_time, true));
        slots[PRIMARY_QD1] = Some((&self.family.design_derivative_exit, &f_time, false));
        for &(primary, design) in self.slope_channels().as_slice() {
            slots[primary] = Some((design, &f_slope, false));
        }
        let channels: [(&DesignMatrix, &Array2<f64>, bool); P] = std::array::from_fn(|primary| {
            slots[primary].expect(
                "every primary of the frame owns exactly one J·F axis: the three location \
                 channels plus the slope layout's follow-up channels",
            )
        });
        let max_cols = channels
            .iter()
            .map(|(design, _, _)| design.ncols())
            .fold(self.family.marginal_design.ncols(), usize::max);

        let mut out = Array2::<f64>::zeros((n, P * rank));
        if n == 0 {
            return Some(out);
        }
        // Per tile in flight: one channel product and the marginal product, and
        // inside `row_chunk_matmul_into` the design rows and their GEMM result,
        // with room for a coefficient transform's inner rows up to twice the
        // design's width.
        let tile_bytes =
            |rows: usize| gam_runtime::resource::dense_f64_bytes(rows, 2 * max_cols + 3 * rank);
        let mut target_rows = JF_TILE_ROWS;
        let (n_tiles, in_flight, tile_buffers) = loop {
            let n_tiles = (n / target_rows).max(1);
            let bytes = tile_bytes(n.div_ceil(n_tiles))?;
            let mut in_flight = rayon::current_num_threads().clamp(1, n_tiles);
            let admitted = loop {
                match reserve(bytes.checked_mul(in_flight)?) {
                    Some(reservation) => break Some(reservation),
                    None if in_flight > 1 => in_flight = in_flight.div_ceil(2),
                    None => break None,
                }
            };
            match admitted {
                Some(reservation) => break (n_tiles, in_flight, reservation),
                None if target_rows > JF_MIN_TILE_ROWS => target_rows /= 2,
                None => {
                    self.warn_jf_row_route_once(&format!(
                        "the memory governor refused one {bytes}-byte J·F row tile with {} bytes \
                         of budget left",
                        remaining()
                    ));
                    return None;
                }
            }
        };

        // Tile t is rows [t·n/n_tiles, (t+1)·n/n_tiles); slot s runs tiles
        // [s·n_tiles/in_flight, (s+1)·n_tiles/in_flight) on its own buffers.
        let tile_start = |tile: usize| tile * n / n_tiles;
        let mut slot_views = Vec::with_capacity(in_flight);
        let mut rest = out.view_mut();
        let mut consumed = 0;
        for slot in 0..in_flight {
            let first_tile = slot * n_tiles / in_flight;
            let end_row = tile_start((slot + 1) * n_tiles / in_flight);
            let (head, tail) = rest.split_at(Axis(0), end_row - consumed);
            slot_views.push((first_tile, (slot + 1) * n_tiles / in_flight, head));
            rest = tail;
            consumed = end_row;
        }
        let max_tile_rows = n.div_ceil(n_tiles);
        let built = slot_views.into_par_iter().try_for_each(|(first_tile, end_tile, mut slot_rows)| {
            let mut product = Array2::<f64>::zeros((max_tile_rows, rank));
            let mut marginal = Array2::<f64>::zeros((max_tile_rows, rank));
            let slot_start = tile_start(first_tile);
            for tile in first_tile..end_tile {
                let global = tile_start(tile)..tile_start(tile + 1);
                let len = global.len();
                let local = global.start - slot_start;
                self.family
                    .marginal_design
                    .row_chunk_matmul_into(global.clone(), f_marginal.view(), marginal.slice_mut(s![..len, ..]))
                    .map_err(|error| error.to_string())?;
                for (primary, &(design, factor_block, adds_marginal)) in channels.iter().enumerate() {
                    design
                        .row_chunk_matmul_into(global.clone(), factor_block.view(), product.slice_mut(s![..len, ..]))
                        .map_err(|error| error.to_string())?;
                    let mut target = slot_rows.slice_mut(s![
                        local..local + len,
                        primary * rank..(primary + 1) * rank
                    ]);
                    for i in 0..len {
                        for j in 0..rank {
                            target[[i, j]] = if adds_marginal {
                                product[[i, j]] + marginal[[i, j]]
                            } else {
                                product[[i, j]]
                            };
                        }
                    }
                }
            }
            Ok::<(), String>(())
        });
        drop(tile_buffers);
        match built {
            Ok(()) => Some(out),
            Err(reason) => {
                self.warn_jf_row_route_once(&format!("a design refused its J·F row tile: {reason}"));
                None
            }
        }
    }

    /// Log, once per fit, why J·F falls to the per-row route. A fit is
    /// identified by its family's shared weight vector, which every clone of
    /// the family and every kernel built from it hold.
    fn warn_jf_row_route_once(&self, reason: &str) {
        static LAST_FIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let fit = Arc::as_ptr(&self.family.weights) as usize;
        if LAST_FIT.swap(fit, std::sync::atomic::Ordering::Relaxed) != fit {
            log::warn!(
                "[survival-marginal-slope] {reason}, so this fit builds J·F one row at a time, which \
                 is slower"
            );
        }
    }
}

impl<const P: usize, G: SlopeRowGeometry<P>> SurvivalMarginalSlopeRowKernel<P, G> {
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
    ///
    /// The towers live in a heap vector that the Rayon workers fill in place
    /// through [`SurvivalMarginalSlopeFamily::write_row_primary_tower`], so a
    /// split frame holds a row index and a slot pointer, never a tower or the
    /// row program (gam#2967).
    fn build_row_towers(&self) -> Result<Vec<G::Tower4>, String> {
        let n = gam_math::jet_tower::RowProgram::n_rows(self);
        let mut towers = vec![G::Tower4::constant(0.0); n];
        towers
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row, tower)| {
                self.family.write_row_primary_tower::<P, G, _>(
                    row,
                    &self.block_states,
                    "survival marginal-slope rigid row fourth tower (build-once)",
                    tower,
                )
            })?;
        Ok(towers)
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
    /// produce. Built in place like [`Self::build_row_towers`], so no split frame
    /// holds a tower (gam#2967).
    fn build_row_third_towers(&self) -> Result<Vec<G::Tower3>, String> {
        let n = gam_math::jet_tower::RowProgram::n_rows(self);
        let mut towers = vec![G::Tower3::constant(0.0); n];
        towers
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row, tower)| {
                self.family.write_row_primary_tower::<P, G, _>(
                    row,
                    &self.block_states,
                    "survival marginal-slope rigid row third tower (build-once)",
                    tower,
                )
            })?;
        Ok(towers)
    }

    /// Deterministic `ARROW_ROW_CHUNK`-chunked reduction matching
    /// `par_try_reduce_fold(RowSet::All)`: rows fold in index order inside each
    /// fixed 256-row chunk, chunks reduce in chunk-index order on the caller
    /// thread. `per_row(row, &mut acc)` accumulates one row's pullback into the
    /// `p×p` accumulator exactly as the generic per-axis fold does.
    pub(super) fn chunked_pullback_reduce<F>(&self, p: usize, per_row: F) -> Result<Array2<f64>, String>
    where
        F: Fn(usize, &mut Array2<f64>) -> Result<(), String> + Sync,
    {
        let n = gam_math::jet_tower::RowProgram::n_rows(self);
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
    fn directional_derivative_all_axes_build_once(&self) -> Result<Vec<Array2<f64>>, String> {
        // #1591: the consumer reads only `third_contracted` (a `t3` contraction),
        // so build the order-≤3 `Tower3<4>` per row — bit-identical on the read
        // channels to the dense `Tower4<4>` but without the discarded `t4` tensor.
        let towers = self.build_row_third_towers()?;
        let tensors: Vec<_> = towers.iter().map(|tower| *tower.t3()).collect();
        self.all_axes_primary_tensor_pullback(&tensors)
    }

    /// Pull back a symmetric primary third tensor along every coefficient axis.
    /// Higher information derivatives first contract their fixed directions
    /// into this tensor, so all orders share the same assembly
    /// ([`crate::row_kernel::all_axes_symmetric_tensor_pullback`]).
    pub(super) fn all_axes_primary_tensor_pullback(
        &self,
        tensors: &[[[[f64; P]; P]; P]],
    ) -> Result<Vec<Array2<f64>>, String> {
        use crate::row_kernel::AllAxesPullbackError;
        crate::row_kernel::all_axes_symmetric_tensor_pullback(self, tensors).map_err(|error| match error {
            AllAxesPullbackError::TensorRowCount { .. } => {
                "survival all-axes primary tensor row count mismatch".to_string()
            }
            AllAxesPullbackError::TileShape { got, expected } => format!(
                "survival marginal-slope all-axes J·I tile shape {got:?}, expected {expected:?}"
            ),
        })
    }

    /// Contract the fixed direction once per row, then use the shared
    /// pullback for every swept coefficient axis.
    fn second_directional_derivative_all_axes_from_towers(
        &self,
        d_beta_u: &[f64],
        towers: &[G::Tower4],
    ) -> Result<Vec<Array2<f64>>, String> {
        let tensors: Vec<_> = towers
            .iter()
            .enumerate()
            .map(|(row, tower)| {
                let direction = self.jacobian_action(row, d_beta_u);
                let t4 = tower.t4();
                std::array::from_fn(|a| {
                    std::array::from_fn(|b| {
                        std::array::from_fn(|c| {
                            (0..P).map(|d| t4[a][b][d][c] * direction[d]).sum()
                        })
                    })
                })
            })
            .collect();
        self.all_axes_primary_tensor_pullback(&tensors)
    }

    fn second_directional_derivative_all_axes_build_once(
        &self,
        d_beta_u: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let towers = self.build_row_towers()?;
        self.second_directional_derivative_all_axes_from_towers(d_beta_u, &towers)
    }

    /// The all-axes second directional derivative along every direction of a
    /// batch from ONE build of the row towers, which depend only on the
    /// coefficient snapshot. Each direction's object is exactly what
    /// [`Self::second_directional_derivative_all_axes_build_once`] returns for it.
    pub(crate) fn second_directional_derivative_all_axes_each(
        &self,
        directions: &[&[f64]],
        consume: &mut dyn FnMut(usize, Vec<Array2<f64>>) -> Result<(), String>,
    ) -> Result<(), String> {
        let p = self.n_coefficients();
        if let Some(direction) = directions.iter().find(|direction| direction.len() != p) {
            return Err(format!(
                "survival marginal-slope batched second directional derivative: direction has {} \
                 entries, expected {p}",
                direction.len()
            ));
        }
        let towers = self.build_row_towers()?;
        for (index, direction) in directions.iter().enumerate() {
            consume(
                index,
                self.second_directional_derivative_all_axes_from_towers(direction, &towers)?,
            )?;
        }
        Ok(())
    }

    /// gam#979 Jeffreys wide-p contracted-trace-Hessian for the rigid survival
    /// marginal-slope kernel: `∇²_β tr(W · H(β))` for a caller-supplied
    /// full-joint trace weight `W`. The observed-Hessian form BMS used before its
    /// Jeffreys information became the expected information (gam#2922),
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
            let t4 = towers[row].t4();
            let mut coeff = [[0.0_f64; P]; P];
            for c in 0..P {
                for d in 0..P {
                    let mut s = 0.0;
                    for a in 0..P {
                        for b in 0..P {
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
    /// marginal)`, `qd1 = (derivative-exit design, time)`, `g = (slope
    /// design, slope)`. Summing `component(a)·W[range,range]·component(b)`
    /// over every pair of components is exactly `jᵃᵀ·W·jᵇ` since `W`
    /// restricted to any range pair not covered by a component is multiplied
    /// by an implicit zero there. Cost is `O(Σ p_block²)` per row (the same
    /// complexity class as BMS's per-row trace contraction), not
    /// `O(p_total²)`, since only the 3 real blocks (`time, marginal,
    /// slope`) — never the optional flex/influence ones, which this hook
    /// only runs when inactive — are read.
    pub(super) fn primary_trace_weight(
        &self,
        row: usize,
        weight: &Array2<f64>,
    ) -> Result<[[f64; P]; P], String> {
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
        let channels = self.slope_channels();
        let slope_designs = channels.as_slice();
        let slope_rows = slope_designs
            .iter()
            .map(|&(primary, design)| {
                design
                    .try_row_chunk(row..row + 1)
                    .map(|chunk| (primary, chunk))
                    .map_err(|e| {
                        format!("primary_trace_weight: slope_design row chunk failed: {e}")
                    })
            })
            .collect::<Result<Vec<_>, String>>()?;

        struct Component<'a> {
            vec: ArrayView1<'a, f64>,
            range: std::ops::Range<usize>,
        }
        let mut components: [Vec<Component<'_>>; P] = std::array::from_fn(|_| Vec::new());
        components[PRIMARY_Q0].push(Component {
            vec: xt_e.row(0),
            range: self.slices.time.clone(),
        });
        components[PRIMARY_Q0].push(Component {
            vec: xm.row(0),
            range: self.slices.marginal.clone(),
        });
        components[PRIMARY_Q1].push(Component {
            vec: xt_x.row(0),
            range: self.slices.time.clone(),
        });
        components[PRIMARY_Q1].push(Component {
            vec: xm.row(0),
            range: self.slices.marginal.clone(),
        });
        components[PRIMARY_QD1].push(Component {
            vec: xt_d.row(0),
            range: self.slices.time.clone(),
        });
        for (primary, chunk) in &slope_rows {
            components[*primary].push(Component {
                vec: chunk.row(0),
                range: self.slices.slope.clone(),
            });
        }

        let mut w_row = [[0.0_f64; P]; P];
        for a in 0..P {
            for b in 0..P {
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
