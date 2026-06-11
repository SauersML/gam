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

use ndarray::Array2;

use crate::families::cubic_cell_kernel::{
    DenestedCubicCell, LocalSpanCubic, PartitionEdge, denested_cell_coefficients,
    evaluate_cell_derivative_moments_uncached,
};

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
        let family = ChebMomentFamily::build(&spec, a_box, b_box, 12)
            .expect("family build must succeed on a smooth box")
            .expect("family must certify on a smooth box");
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
        let family = ChebMomentFamily::build(&spec, (-0.3, 0.5), (0.7, 1.4), 8)
            .expect("build must succeed")
            .expect("fixed-edge family must certify at m=8");
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
}
