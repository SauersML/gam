//! Certified 2-D Chebyshev families of denested-cell derivative moments.
//!
//! ## Why families
//!
//! The marginal-slope flex row calculus integrates `z^k · exp(-½(z² + η²))`
//! over the denested partition of every training row, where the row's
//! composed index is `η(z) = W(a + b·H(z))` with the SAME splines `(H, W)`
//! shared across all rows — only the two scalars `(a, b)` differ per row.
//! Each interior cell of the partition is fully determined by a fixed
//! `(score_span, link_span, edge-pair)` combination from a finite set plus
//! the row's `(a, b)`: the cell's cubic comes from
//! [`denested_cell_coefficients`] and its endpoints are either fixed score
//! breaks or link-knot crossings `z = (τ - a)/b`. The moment vector
//! `M_k(a, b)` of one combination is therefore a smooth two-parameter
//! family, jointly **analytic** on any `(a, b)` box that avoids the kink
//! lines `a + b·σ = τ` (where a crossing passes a score break and the cell
//! topology changes) and the `b = 0` line (where crossings escape to ±∞).
//!
//! On such a box, tensor-Chebyshev interpolation of `M_k(a, b)` converges
//! geometrically, so a small `m × m` node grid — each node evaluated once by
//! the certified progressive-ladder quadrature — replaces the per-row
//! transcendental quadrature with an `O((d+2)·m²)` polynomial evaluation.
//! At biobank scale the build cost is amortized over `n` rows per criterion
//! evaluation (#979).
//!
//! ## Certification, not approximation-by-fiat
//!
//! A family is only usable if [`ChebMomentFamily::build`] returns
//! `Ok(Some(_))`, which requires BOTH
//! 1. the Chebyshev coefficient tail (the highest-order row and column of
//!    the tensor) to carry at most [`FAMILY_CERT_RTOL`] of the family scale —
//!    the standard geometric-decay certificate for analytic interpolands; and
//! 2. a deterministic interior spot check against the direct ladder
//!    evaluation at [`FAMILY_SPOT_CHECK_POINTS`] off-grid points to agree to
//!    [`FAMILY_SPOT_RTOL`] of the family scale.
//!
//! A box that straddles a kink line, contains `b ≈ 0`, or degenerates a cell
//! fails the certificate (or errors during the build) and the caller falls
//! back to direct ladder quadrature for the affected rows — the same
//! certified-or-fallback discipline as the quadrature ladder itself.

use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::Array2;

use crate::cubic_cell_kernel::{
    DenestedCubicCell, LocalSpanCubic, PartitionEdge, denested_cell_coefficients,
    evaluate_cell_derivative_moments_uncached,
};

/// Process-wide count of per-row cell lookups served by a certified family
/// (a transcendental-free interpolant hit) vs. falling back to direct ladder
/// quadrature. The delta across a fit reveals whether the forest actually
/// covers the row cloud or is dead weight (#979 slop check).
static FOREST_COVERED: AtomicU64 = AtomicU64::new(0);
static FOREST_FALLBACK: AtomicU64 = AtomicU64::new(0);

/// `(covered, fallback)` snapshot of the family-forest coverage counters.
pub fn forest_coverage_counts() -> (u64, u64) {
    (
        FOREST_COVERED.load(Ordering::Relaxed),
        FOREST_FALLBACK.load(Ordering::Relaxed),
    )
}

/// Relative (to the family's max moment magnitude) ceiling on the Chebyshev
/// tail mass for a family to certify. Analytic interpolands decay
/// geometrically, so a truncation whose last row/column already sits at this
/// level has interpolation error of the same order.
pub const FAMILY_CERT_RTOL: f64 = 1.0e-12;

/// Relative agreement required between the interpolant and the direct ladder
/// evaluation at the off-grid spot-check points.
pub const FAMILY_SPOT_RTOL: f64 = 1.0e-11;

/// Number of deterministic off-grid interior spot-check points.
pub const FAMILY_SPOT_CHECK_POINTS: usize = 3;

/// A fixed `(score_span, link_span, edge-pair)` combination whose moments
/// form a smooth two-parameter family in the row scalars `(a, b)`. Edge
/// provenance is the kernel's [`PartitionEdge`], carried per cell by
/// `build_denested_partition_cells_with_tails`.
#[derive(Clone, Copy, Debug)]
pub struct CellMomentFamilySpec {
    pub score_span: LocalSpanCubic,
    pub link_span: LocalSpanCubic,
    pub left: PartitionEdge,
    pub right: PartitionEdge,
    pub max_degree: usize,
}

impl CellMomentFamilySpec {
    /// Materialize the concrete denested cell at `(a, b)`.
    ///
    /// Errors when the edges degenerate (`right <= left`) or produce
    /// non-finite interior bounds — the conditions under which the family
    /// parameterization itself breaks down (e.g. a crossing passing the
    /// other edge inside the box).
    pub fn cell_at(&self, a: f64, b: f64) -> Result<DenestedCubicCell, String> {
        let left = self.left.z_at(a, b);
        let right = self.right.z_at(a, b);
        let left_finite_ok = left.is_finite() || left == f64::NEG_INFINITY;
        let right_finite_ok = right.is_finite() || right == f64::INFINITY;
        if !left_finite_ok || !right_finite_ok || right <= left {
            return Err(format!(
                "cell moment family: degenerate cell at (a={a:.6e}, b={b:.6e}): [{left:.6e}, {right:.6e}]"
            ));
        }
        let coeffs = denested_cell_coefficients(self.score_span, self.link_span, a, b);
        Ok(DenestedCubicCell {
            left,
            right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        })
    }

    /// Direct (ladder-quadrature) moment evaluation at `(a, b)` — the ground
    /// truth the interpolant is built from and certified against.
    pub fn moments_direct(&self, a: f64, b: f64) -> Result<Vec<f64>, String> {
        let cell = self.cell_at(a, b)?;
        let state = evaluate_cell_derivative_moments_uncached(cell, self.max_degree)?;
        Ok(state.moments.to_vec())
    }
}

/// Chebyshev nodes of the first kind on `[-1, 1]` (no endpoints, so a box
/// flush against a kink line never evaluates exactly on it).
fn chebyshev_nodes(m: usize) -> Vec<f64> {
    (0..m)
        .map(|i| (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * m) as f64).cos())
        .collect()
}

/// `T_p(x)` for `p = 0..m` written into `out` by the three-term recurrence.
#[inline]
fn chebyshev_basis_into(x: f64, out: &mut [f64]) {
    if let Some(first) = out.first_mut() {
        *first = 1.0;
    }
    if let Some(second) = out.get_mut(1) {
        *second = x;
    }
    for p in 2..out.len() {
        out[p] = 2.0 * x * out[p - 1] - out[p - 2];
    }
}

/// A certified `m × m` tensor-Chebyshev interpolant of one cell family's
/// moment vector over the box `[a_lo, a_hi] × [b_lo, b_hi]`.
pub struct ChebMomentFamily {
    a_lo: f64,
    a_hi: f64,
    b_lo: f64,
    b_hi: f64,
    m: usize,
    max_degree: usize,
    /// `coeff[k]` is the `m × m` Chebyshev coefficient tensor of moment `k`.
    coeff: Vec<Array2<f64>>,
    /// Max |moment| observed over the node grid — the family scale every
    /// relative bound in this module is taken against.
    pub scale: f64,
}

impl ChebMomentFamily {
    /// Build and certify a family interpolant. Returns `Ok(None)` when the
    /// family does not certify on this box (tail mass too heavy or a spot
    /// check fails) — the caller falls back to direct ladder quadrature.
    /// Errors only on degenerate cells / non-finite direct evaluations,
    /// which equally mean "fall back".
    pub fn build(
        spec: &CellMomentFamilySpec,
        (a_lo, a_hi): (f64, f64),
        (b_lo, b_hi): (f64, f64),
        m: usize,
    ) -> Result<Option<Self>, String> {
        if !(a_lo.is_finite() && a_hi.is_finite() && b_lo.is_finite() && b_hi.is_finite())
            || a_hi <= a_lo
            || b_hi <= b_lo
        {
            return Err(format!(
                "cell moment family: invalid box [{a_lo}, {a_hi}] x [{b_lo}, {b_hi}]"
            ));
        }
        if m < 4 {
            return Err(format!("cell moment family: need m >= 4 nodes, got {m}"));
        }
        let d = spec.max_degree;
        let nodes = chebyshev_nodes(m);
        let map_a = |x: f64| 0.5 * (a_lo + a_hi) + 0.5 * (a_hi - a_lo) * x;
        let map_b = |x: f64| 0.5 * (b_lo + b_hi) + 0.5 * (b_hi - b_lo) * x;

        // Direct moments at every tensor node.
        let mut values: Vec<Array2<f64>> = (0..=d).map(|_| Array2::zeros((m, m))).collect();
        let mut scale = 0.0_f64;
        for (i, &xa) in nodes.iter().enumerate() {
            for (j, &xb) in nodes.iter().enumerate() {
                let moments = spec.moments_direct(map_a(xa), map_b(xb))?;
                if moments.len() != d + 1 {
                    return Err(format!(
                        "cell moment family: direct evaluation returned {} moments, expected {}",
                        moments.len(),
                        d + 1
                    ));
                }
                for (k, &mk) in moments.iter().enumerate() {
                    if !mk.is_finite() {
                        return Err(format!(
                            "cell moment family: non-finite moment k={k} at node ({i}, {j})"
                        ));
                    }
                    values[k][[i, j]] = mk;
                    scale = scale.max(mk.abs());
                }
            }
        }

        // Chebyshev tensor coefficients from first-kind nodes:
        //   c_{p,q} = (γ_p γ_q / m²) Σ_i Σ_j f(x_i, x_j) T_p(x_i) T_q(x_j),
        // with γ_0 = 1 and γ_p = 2 for p > 0.
        let mut basis = Array2::<f64>::zeros((m, m));
        {
            let mut row = vec![0.0_f64; m];
            for (i, &x) in nodes.iter().enumerate() {
                chebyshev_basis_into(x, &mut row);
                for (p, &t) in row.iter().enumerate() {
                    basis[[i, p]] = t;
                }
            }
        }
        let inv_m2 = 1.0 / (m * m) as f64;
        let gamma = |p: usize| if p == 0 { 1.0 } else { 2.0 };
        let mut coeff: Vec<Array2<f64>> = Vec::with_capacity(d + 1);
        for vals in &values {
            // tmp[p][j] = Σ_i T_p(x_i) vals[i][j];  c[p][q] = Σ_j tmp[p][j] T_q(x_j)
            let tmp = basis.t().dot(vals);
            let raw = tmp.dot(&basis);
            let mut c = raw;
            for p in 0..m {
                for q in 0..m {
                    c[[p, q]] *= gamma(p) * gamma(q) * inv_m2;
                }
            }
            coeff.push(c);
        }

        let family = Self {
            a_lo,
            a_hi,
            b_lo,
            b_hi,
            m,
            max_degree: d,
            coeff,
            scale,
        };

        // Certificate 1: geometric tail decay. The last row and column of
        // the coefficient tensor bound the truncation error for an analytic
        // interpoland.
        if scale > 0.0 {
            let mut tail = 0.0_f64;
            for c in &family.coeff {
                for q in 0..m {
                    tail = tail.max(c[[m - 1, q]].abs());
                }
                for p in 0..m {
                    tail = tail.max(c[[p, m - 1]].abs());
                }
            }
            if tail > FAMILY_CERT_RTOL * scale {
                return Ok(None);
            }
        }

        // Certificate 2: deterministic off-grid spot checks against the
        // direct ladder evaluation (golden-ratio low-discrepancy interior
        // points — reproducible, no RNG).
        let phi = 0.618_033_988_749_894_9_f64;
        let mut out = vec![0.0_f64; d + 1];
        for s in 1..=FAMILY_SPOT_CHECK_POINTS {
            let fa = (0.5 + s as f64 * phi).fract();
            let fb = (0.25 + s as f64 * phi * phi).fract();
            let a = a_lo + fa * (a_hi - a_lo);
            let b = b_lo + fb * (b_hi - b_lo);
            let direct = spec.moments_direct(a, b)?;
            family.eval_into(a, b, &mut out)?;
            for k in 0..=d {
                if (out[k] - direct[k]).abs() > FAMILY_SPOT_RTOL * scale.max(f64::MIN_POSITIVE) {
                    return Ok(None);
                }
            }
        }

        Ok(Some(family))
    }

    /// Evaluate the interpolated moment vector at `(a, b)` into `out`
    /// (length `max_degree + 1`). Transcendental-free: two Chebyshev basis
    /// recurrences plus one `m × m` contraction per degree.
    pub fn eval_into(&self, a: f64, b: f64, out: &mut [f64]) -> Result<(), String> {
        if out.len() != self.max_degree + 1 {
            return Err(format!(
                "cell moment family eval: out length {} != max_degree + 1 = {}",
                out.len(),
                self.max_degree + 1
            ));
        }
        if !(self.a_lo..=self.a_hi).contains(&a) || !(self.b_lo..=self.b_hi).contains(&b) {
            return Err(format!(
                "cell moment family eval: ({a:.6e}, {b:.6e}) outside box [{}, {}] x [{}, {}]",
                self.a_lo, self.a_hi, self.b_lo, self.b_hi
            ));
        }
        let xa = (2.0 * a - (self.a_lo + self.a_hi)) / (self.a_hi - self.a_lo);
        let xb = (2.0 * b - (self.b_lo + self.b_hi)) / (self.b_hi - self.b_lo);
        let m = self.m;
        let mut ta = vec![0.0_f64; m];
        let mut tb = vec![0.0_f64; m];
        chebyshev_basis_into(xa, &mut ta);
        chebyshev_basis_into(xb, &mut tb);
        for (k, slot) in out.iter_mut().enumerate() {
            let c = &self.coeff[k];
            let mut acc = 0.0_f64;
            for p in 0..m {
                let mut row = 0.0_f64;
                for q in 0..m {
                    row = c[[p, q]].mul_add(tb[q], row);
                }
                acc = ta[p].mul_add(row, acc);
            }
            *slot = acc;
        }
        Ok(())
    }
}

// ───────────────────────── Stage B: box forest ──────────────────────────

/// Minimum rows a leaf must hold before family interpolants are built for
/// it: below this the `m²` ladder evaluations per family cost more than the
/// direct per-row ladder calls they replace.
pub const FOREST_MIN_ROWS_PER_LEAF: usize = 256;

/// Maximum k-d subdivision depth of the `(a, b)` box forest.
pub const FOREST_MAX_DEPTH: usize = 12;

/// Tensor-Chebyshev node-count escalation ladder for forest-built families:
/// like the quadrature ladder, the build accepts the first rung whose
/// certificate passes; wider boxes need higher order to reach the tail
/// tolerance, narrower boxes certify cheaply at the bottom rung.
pub const FOREST_NODE_LADDER: [usize; 4] = [8, 12, 16, 20];

/// Build a family at the first node count on [`FOREST_NODE_LADDER`] whose
/// certificate passes; `None` when no rung certifies (⇒ ladder fallback).
pub fn build_family_escalating(
    spec: &CellMomentFamilySpec,
    a_box: (f64, f64),
    b_box: (f64, f64),
) -> Option<ChebMomentFamily> {
    for &m in FOREST_NODE_LADDER.iter() {
        match ChebMomentFamily::build(spec, a_box, b_box, m) {
            Ok(Some(family)) => return Some(family),
            Ok(None) => continue,
            // Degenerate cells / non-finite evaluations anywhere on the node
            // grid: the family parameterization is invalid on this box at
            // every order — stop escalating.
            Err(_) => return None,
        }
    }
    None
}

/// Bit-exact identity of one cell family: the `(score_span, link_span,
/// edge-pair)` combination shared by rows. Built from the provenance the
/// partition builder records per cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComboKey {
    score_bits: [u64; 6],
    link_bits: [u64; 6],
    left_bits: (bool, u64),
    right_bits: (bool, u64),
}

impl ComboKey {
    pub fn new(
        score_span: LocalSpanCubic,
        link_span: LocalSpanCubic,
        left: PartitionEdge,
        right: PartitionEdge,
    ) -> Self {
        let span_bits = |s: LocalSpanCubic| {
            [
                s.left.to_bits(),
                s.right.to_bits(),
                s.c0.to_bits(),
                s.c1.to_bits(),
                s.c2.to_bits(),
                s.c3.to_bits(),
            ]
        };
        let edge_bits = |e: PartitionEdge| match e {
            PartitionEdge::Fixed(z) => (false, z.to_bits()),
            PartitionEdge::Crossing { tau } => (true, tau.to_bits()),
        };
        Self {
            score_bits: span_bits(score_span),
            link_bits: span_bits(link_span),
            left_bits: edge_bits(left),
            right_bits: edge_bits(right),
        }
    }
}

/// `true` when the kink line `a + b·σ = τ` crosses the open box (sign of
/// `a + b·σ − τ` differs across corners). Lines that merely touch a corner
/// count as crossing — conservative.
fn kink_line_crosses_box(
    (a_lo, a_hi): (f64, f64),
    (b_lo, b_hi): (f64, f64),
    sigma: f64,
    tau: f64,
) -> bool {
    let corners = [
        a_lo + b_lo * sigma - tau,
        a_lo + b_hi * sigma - tau,
        a_hi + b_lo * sigma - tau,
        a_hi + b_hi * sigma - tau,
    ];
    let any_nonneg = corners.iter().any(|&c| c >= 0.0);
    let any_nonpos = corners.iter().any(|&c| c <= 0.0);
    any_nonneg && any_nonpos
}

/// One leaf of the `(a, b)` forest: a kink-free box plus the rows it owns.
struct ForestLeaf {
    a_box: (f64, f64),
    b_box: (f64, f64),
    /// Indices into the caller's row arrays.
    rows: Vec<usize>,
    /// `true` when families may be built for this leaf (kink-free and
    /// populated enough to amortize the build).
    eligible: bool,
}

/// Adaptive kink-aware box forest over the row cloud `{(a_i, b_i)}`.
///
/// Deterministic: the subdivision depends only on the row coordinates and
/// the knot vectors (longest-axis halving), never on wall-clock, RNG, or
/// thread schedule — preserving the warm/cold invariance contract.
pub struct CellFamilyForest {
    leaves: Vec<ForestLeaf>,
    /// Per (leaf, combo): a certified family, or `None` after a failed
    /// build/certification (⇒ ladder fallback for that combo in that leaf).
    families: std::collections::HashMap<(usize, ComboKey), Option<ChebMomentFamily>>,
    /// Leaf index per row (parallel to the caller's row arrays).
    row_leaf: Vec<usize>,
}

impl CellFamilyForest {
    /// Partition the row cloud into kink-free boxes.
    ///
    /// `score_breaks` and `link_breaks` are the finite knot vectors whose
    /// pairs `(σ, τ)` generate the kink lines `a + b·σ = τ`; `b = 0` is
    /// always treated as a kink (crossings escape to ±∞ there).
    pub fn partition(
        a: &[f64],
        b: &[f64],
        score_breaks: &[f64],
        link_breaks: &[f64],
    ) -> Result<Self, String> {
        if a.len() != b.len() {
            return Err(format!(
                "cell family forest: a/b length mismatch ({} vs {})",
                a.len(),
                b.len()
            ));
        }
        let n = a.len();
        if n == 0 {
            return Ok(Self {
                leaves: Vec::new(),
                families: std::collections::HashMap::new(),
                row_leaf: Vec::new(),
            });
        }
        let mut a_lo = f64::INFINITY;
        let mut a_hi = f64::NEG_INFINITY;
        let mut b_lo = f64::INFINITY;
        let mut b_hi = f64::NEG_INFINITY;
        for i in 0..n {
            if !(a[i].is_finite() && b[i].is_finite()) {
                return Err(format!(
                    "cell family forest: non-finite row scalars at {i}: (a={}, b={})",
                    a[i], b[i]
                ));
            }
            a_lo = a_lo.min(a[i]);
            a_hi = a_hi.max(a[i]);
            b_lo = b_lo.min(b[i]);
            b_hi = b_hi.max(b[i]);
        }
        // Widen degenerate (single-point) extents so boxes are genuine.
        let widen = |lo: f64, hi: f64| {
            if hi > lo {
                (lo, hi)
            } else {
                let pad = lo.abs().max(1.0) * 1.0e-9;
                (lo - pad, hi + pad)
            }
        };
        let (a_lo, a_hi) = widen(a_lo, a_hi);
        let (b_lo, b_hi) = widen(b_lo, b_hi);

        let box_is_kink_free = |a_box: (f64, f64), b_box: (f64, f64)| -> bool {
            // b = 0 inside the box ⇒ crossings blow up.
            if b_box.0 <= 0.0 && b_box.1 >= 0.0 {
                return false;
            }
            for &sigma in score_breaks {
                for &tau in link_breaks {
                    if kink_line_crosses_box(a_box, b_box, sigma, tau) {
                        return false;
                    }
                }
            }
            true
        };

        let mut leaves: Vec<ForestLeaf> = Vec::new();
        // Explicit work stack of (a_box, b_box, rows, depth).
        let mut stack: Vec<((f64, f64), (f64, f64), Vec<usize>, usize)> =
            vec![((a_lo, a_hi), (b_lo, b_hi), (0..n).collect(), 0)];
        while let Some((a_box, b_box, rows, depth)) = stack.pop() {
            let kink_free = box_is_kink_free(a_box, b_box);
            if kink_free || depth >= FOREST_MAX_DEPTH || rows.len() < FOREST_MIN_ROWS_PER_LEAF {
                leaves.push(ForestLeaf {
                    a_box,
                    b_box,
                    eligible: kink_free && rows.len() >= FOREST_MIN_ROWS_PER_LEAF,
                    rows,
                });
                continue;
            }
            // Halve the longer axis (deterministic midpoint split).
            let a_len = a_box.1 - a_box.0;
            let b_len = b_box.1 - b_box.0;
            let split_a = a_len >= b_len;
            let mid = if split_a {
                0.5 * (a_box.0 + a_box.1)
            } else {
                0.5 * (b_box.0 + b_box.1)
            };
            let mut lo_rows = Vec::new();
            let mut hi_rows = Vec::new();
            for &i in &rows {
                let coord = if split_a { a[i] } else { b[i] };
                if coord <= mid {
                    lo_rows.push(i);
                } else {
                    hi_rows.push(i);
                }
            }
            if split_a {
                stack.push(((a_box.0, mid), b_box, lo_rows, depth + 1));
                stack.push(((mid, a_box.1), b_box, hi_rows, depth + 1));
            } else {
                stack.push((a_box, (b_box.0, mid), lo_rows, depth + 1));
                stack.push((a_box, (mid, b_box.1), hi_rows, depth + 1));
            }
        }

        let mut row_leaf = vec![usize::MAX; n];
        for (leaf_idx, leaf) in leaves.iter().enumerate() {
            for &i in &leaf.rows {
                row_leaf[i] = leaf_idx;
            }
        }
        if row_leaf.iter().any(|&l| l == usize::MAX) {
            return Err("cell family forest: a row was not assigned to any leaf".to_string());
        }
        Ok(Self {
            leaves,
            families: std::collections::HashMap::new(),
            row_leaf,
        })
    }

    /// Phase 2: build (and certify) the family interpolants for the given
    /// demand set of `(row, combo, spec)` triples. Combos demanded from
    /// ineligible leaves are skipped (those rows use the ladder directly).
    /// Build failures and certification refusals record `None` so the
    /// per-row evaluation falls back without retrying.
    pub fn build_families<I>(&mut self, demand: I)
    where
        I: IntoIterator<Item = (usize, ComboKey, CellMomentFamilySpec)>,
    {
        for (row, key, spec) in demand {
            let leaf_idx = match self.row_leaf.get(row) {
                Some(&idx) => idx,
                None => continue,
            };
            let leaf = &self.leaves[leaf_idx];
            if !leaf.eligible {
                continue;
            }
            self.families
                .entry((leaf_idx, key))
                .or_insert_with(|| build_family_escalating(&spec, leaf.a_box, leaf.b_box));
        }
    }

    /// Per-row moment lookup: `Some` when a certified family covers this
    /// row's leaf and combo (writes the interpolated moments into `out`),
    /// `None` when the row must use direct ladder quadrature.
    pub fn moments_into(
        &self,
        row: usize,
        key: ComboKey,
        a: f64,
        b: f64,
        out: &mut [f64],
    ) -> Option<()> {
        let covered = (|| {
            let leaf_idx = *self.row_leaf.get(row)?;
            let family = self.families.get(&(leaf_idx, key))?.as_ref()?;
            family.eval_into(a, b, out).ok()
        })();
        if covered.is_some() {
            FOREST_COVERED.fetch_add(1, Ordering::Relaxed);
        } else {
            FOREST_FALLBACK.fetch_add(1, Ordering::Relaxed);
        }
        covered
    }

    /// Number of leaves eligible for family interpolation (observability).
    pub fn eligible_leaves(&self) -> usize {
        self.leaves.iter().filter(|leaf| leaf.eligible).count()
    }

    /// Total leaves (observability).
    pub fn total_leaves(&self) -> usize {
        self.leaves.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative non-affine family: gentle cubic score and link
    /// deviations composed at moderate `(a, b)` — the shape the flex
    /// marginal-slope path produces for interior cells.
    fn test_spec(max_degree: usize) -> CellMomentFamilySpec {
        CellMomentFamilySpec {
            score_span: LocalSpanCubic {
                left: -0.4,
                right: 0.9,
                c0: 0.05,
                c1: -0.12,
                c2: 0.08,
                c3: -0.03,
            },
            link_span: LocalSpanCubic {
                left: -0.2,
                right: 1.4,
                c0: -0.07,
                c1: 0.15,
                c2: -0.05,
                c3: 0.02,
            },
            left: PartitionEdge::Fixed(-0.4),
            right: PartitionEdge::Crossing { tau: 1.1 },
            max_degree,
        }
    }

    #[test]
    fn family_certifies_and_matches_direct_ladder_on_dense_grid() {
        let spec = test_spec(9);
        let a_box = (0.1, 0.6);
        let b_box = (0.8, 1.3);
        let family = build_family_escalating(&spec, a_box, b_box)
            .expect("family must certify on a smooth box at some ladder rung");
        let mut out = vec![0.0_f64; 10];
        let mut worst = 0.0_f64;
        for ia in 0..7 {
            for ib in 0..7 {
                let a = a_box.0 + (ia as f64 + 0.5) / 7.0 * (a_box.1 - a_box.0);
                let b = b_box.0 + (ib as f64 + 0.5) / 7.0 * (b_box.1 - b_box.0);
                let direct = spec.moments_direct(a, b).expect("direct moments");
                family.eval_into(a, b, &mut out).expect("family eval");
                for k in 0..10 {
                    worst = worst.max((out[k] - direct[k]).abs());
                }
            }
        }
        assert!(
            worst <= 1.0e-10 * family.scale.max(f64::MIN_POSITIVE),
            "certified family must match direct ladder quadrature: worst abs err {worst:.3e} vs scale {:.3e}",
            family.scale
        );
    }

    #[test]
    fn family_refuses_box_containing_b_zero_crossing_blowup() {
        let spec = test_spec(5);
        // b spans through 0: the crossing endpoint (tau - a)/b blows up and
        // the cell degenerates somewhere in the box — the build must error
        // or refuse to certify, never silently return a certified family.
        let result = ChebMomentFamily::build(&spec, (0.1, 0.6), (-0.5, 0.5), 8);
        match result {
            Err(_) => {}
            Ok(None) => {}
            Ok(Some(_)) => panic!("family across b=0 must not certify"),
        }
    }

    #[test]
    fn fixed_edge_pair_family_certifies_at_low_node_count() {
        // Both edges fixed: the family varies only through the cell cubic's
        // (a, b) dependence — even smoother, so a small grid certifies.
        let mut spec = test_spec(9);
        spec.right = PartitionEdge::Fixed(0.9);
        let family = build_family_escalating(&spec, (-0.3, 0.5), (0.7, 1.4))
            .expect("fixed-edge family must certify at some ladder rung");
        let mut out = vec![0.0_f64; 10];
        let (a, b) = (0.137, 1.021);
        let direct = spec.moments_direct(a, b).expect("direct moments");
        family.eval_into(a, b, &mut out).expect("family eval");
        for k in 0..10 {
            assert!(
                (out[k] - direct[k]).abs() <= 1.0e-10 * family.scale.max(f64::MIN_POSITIVE),
                "moment {k}: interp {} vs direct {}",
                out[k],
                direct[k]
            );
        }
    }

    #[test]
    fn forest_partitions_avoid_kinks_and_families_match_ladder_end_to_end() {
        // Synthetic row cloud: a deterministic lattice over an (a, b) region
        // crossed by several kink lines (a + b·σ = τ for the knot pairs
        // below), so the forest must subdivide before any leaf certifies.
        let score_breaks = [-0.4_f64, 0.9];
        let link_breaks = [-0.2_f64, 1.1, 1.4];
        let n = 16384;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let phi = 0.618_033_988_749_894_9_f64;
        for i in 0..n {
            let fa = (i as f64 * phi).fract();
            let fb = (i as f64 * phi * phi).fract();
            a.push(0.0 + fa * 0.9);
            b.push(0.7 + fb * 0.7);
        }
        let mut forest = CellFamilyForest::partition(&a, &b, &score_breaks, &link_breaks)
            .expect("forest partition");
        assert!(
            forest.total_leaves() > 1,
            "kink lines must force subdivision"
        );
        assert!(
            forest.eligible_leaves() > 0,
            "a dense cloud must yield at least one eligible kink-free leaf"
        );

        // One combo demanded for every row; the forest builds one family per
        // eligible leaf and rows in ineligible leaves fall back.
        let spec = test_spec(9);
        let key = ComboKey::new(spec.score_span, spec.link_span, spec.left, spec.right);
        forest.build_families((0..n).map(|row| (row, key, spec)));

        let mut out = vec![0.0_f64; 10];
        let mut covered = 0usize;
        for row in 0..n {
            match forest.moments_into(row, key, a[row], b[row], &mut out) {
                Some(()) => {
                    covered += 1;
                    let direct = spec.moments_direct(a[row], b[row]).expect("direct moments");
                    let scale = direct
                        .iter()
                        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                        .max(f64::MIN_POSITIVE);
                    for k in 0..10 {
                        assert!(
                            (out[k] - direct[k]).abs() <= 1.0e-9 * scale,
                            "row {row} moment {k}: interp {} vs direct {}",
                            out[k],
                            direct[k]
                        );
                    }
                }
                None => {
                    // Fallback row: direct ladder must still work (it is the
                    // production fallback path).
                    spec.moments_direct(a[row], b[row])
                        .expect("ladder fallback moments");
                }
            }
        }
        assert!(
            covered * 2 >= n,
            "most rows of a dense cloud should be family-covered, got {covered}/{n}"
        );
    }
}
