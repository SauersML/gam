//! #2757 — the residual-gauge curvature as an operator over a **re-streamed**
//! root, for the branch where no materialized form of it is smaller than
//! `param_dim²`.
//!
//! # Which branch, and why it is the only one left
//!
//! `H = Σ_n J_nᵀ M_n J_n + LᵀL`. The per-row pinning Jacobian `J_n` is
//! output-coordinate diagonal, so when `M_n` is too — Euclidean provenance, what
//! [`SaeManifoldTerm::diagnostic_metric`] installs with no harvest — `H` is `p`
//! blocks of `D × D` and
//! [`OutputBlockRootAccumulator`](crate::identifiability::OutputBlockRootAccumulator)
//! holds it in `p·D²` scalars. That is the branch #2757 was filed on and it is
//! fixed.
//!
//! A metric that COUPLES output coordinates destroys the only structure there
//! was: `J_nᵀ M_n J_n` is dense, and `H` is a sum of `n · metric_rank` rank-one
//! terms in `param_dim` dimensions. Every stored representation of that costs
//! `min(root_rows, param_dim)²` scalars — a `param_dim`-square triangular factor
//! at any production row count — and an exact full spectrum of it costs the cube.
//! At the #2283 production shape that is `65 536² · 8 B = 34 GiB` and `2.8e14`
//! flops for the spectrum alone, on top of `1.0e15` to fold `480 000` rows into
//! the factor in the first place.
//!
//! # What this module does instead
//!
//! It does not store `H` at all. The certificate reads three things off it, and
//! only two of them enter a verdict:
//!
//! | read | enters a verdict | streamable |
//! |---|---|---|
//! | `ξᵀHξ` per generator | yes (numerator) | **exactly**, one pass |
//! | `λ_max(H)` | yes (denominator) | to a certified relative residual |
//! | pinning rank | no | not over the whole parameter space |
//!
//! [`StreamedFrameCurvatureOperator`] implements
//! [`StreamedFrameCurvature`](crate::identifiability::StreamedFrameCurvature) by
//! re-running the SAME `fill_row_frame_jacobian` the materializing builder uses,
//! so there is one definition of what `R`'s rows are and the two routes cannot
//! drift into two curvatures. Every method is a single pass whose per-observation
//! contraction is `O(p·(D + metric_rank))` — never `O(metric_rank · param_dim)`,
//! which is what emitting the root's rows explicitly would cost.
//!
//! # The contraction, once
//!
//! Every read is the same three-step contraction of one observation, differing
//! only in what it does with the rank-space vector in the middle:
//!
//! ```text
//!   a[i]  = Σ_l g[i, l] · x[col(i, l)]          (J_n x, in p·D flops)
//!   c[r]  = Σ_i U[n, i, r] · a[i]               (U_nᵀ J_n x, in p·rank flops)
//!   ────────────────────────────────────────────────────────────────────
//!   apply:         b[i] = Σ_r U[n, i, r]·c[r];  y[col(i,l)] += g[i,l]·b[i]
//!   project_root:  c IS the root row's entry for this direction — fold it
//! ```
//!
//! `c` is exactly the root row `R`'s entry: `R`'s row `(n, r)` has
//! `R[(n,r), col(i,l)] = U[n,i,r]·g[i,l]`, so `R x` at that row is `c[r]`. That
//! identity is why `project_root` is EXACT rather than an approximation of the
//! stored route — it folds the same numbers the stored route folds, in the same
//! order, through the same Givens routine.

use super::*;
use crate::identifiability::{FrameColumnLayout, StreamedFrameCurvature, TriangularRootAccumulator};
use ndarray::{Array1, Array2, ArrayView1};

/// The residual-gauge curvature of a fitted [`SaeManifoldTerm`], as an operator.
///
/// Borrows everything: the term (for the decoder Jacobians), the metric, the
/// column layout, and the isometry pin's root rows. Nothing here is retained
/// between calls except the assignment matrix, which is `n × k` and is the one
/// object the Jacobian stream reads that is not already a borrow.
pub(crate) struct StreamedFrameCurvatureOperator<'a> {
    term: &'a SaeManifoldTerm,
    metric: &'a gam_problem::RowMetric,
    layout: &'a FrameColumnLayout,
    /// The isometry pin's rows of `R` — one per `(atom, frame axis)`, each
    /// spread across every output coordinate. Empty when no pin is installed.
    dense_rows: &'a Array2<f64>,
    assignments: Array2<f64>,
    root_rows: usize,
}

impl<'a> StreamedFrameCurvatureOperator<'a> {
    /// Build the operator for `term` in `metric`, over `layout`'s parameters.
    ///
    /// `dense_rows` and `root_rows` are exactly what
    /// [`SaeManifoldTerm::residual_gauge_streamed_data_curvature`] would have
    /// been handed and would have reported, so the streamed and stored routes
    /// describe the same `R` down to its row count — which is what calibrates
    /// the rank tolerance.
    pub(crate) fn new(
        term: &'a SaeManifoldTerm,
        metric: &'a gam_problem::RowMetric,
        layout: &'a FrameColumnLayout,
        dense_rows: &'a Array2<f64>,
        root_rows: usize,
    ) -> Result<Self, String> {
        let p = term.output_dim();
        if metric.p_out() != p {
            return Err(format!(
                "streamed frame curvature: metric output dim {} but term has {p}",
                metric.p_out()
            ));
        }
        if layout.output_dim() != p || layout.atom_count() != term.k_atoms() {
            return Err(format!(
                "streamed frame curvature: frame layout is ({}, {} atoms) but the term is \
                 ({p}, {} atoms)",
                layout.output_dim(),
                layout.atom_count(),
                term.k_atoms()
            ));
        }
        if dense_rows.nrows() > 0 && dense_rows.ncols() != layout.param_dim() {
            return Err(format!(
                "streamed frame curvature: pin rows have {} columns but param_dim = {}",
                dense_rows.ncols(),
                layout.param_dim()
            ));
        }
        let expected = term
            .n_obs()
            .saturating_mul(metric.metric_rank())
            .saturating_add(dense_rows.nrows());
        if root_rows != expected {
            return Err(format!(
                "streamed frame curvature: caller reports {root_rows} root rows but this term \
                 in this metric has {expected}"
            ));
        }
        Ok(Self {
            term,
            metric,
            layout,
            dense_rows,
            assignments: term.assignment.assignments(),
            root_rows,
        })
    }

    /// Per-observation scratch, sized once and reused across the whole pass.
    fn scratch(&self) -> (Vec<f64>, Array2<f64>, Vec<f64>) {
        let p = self.term.output_dim();
        (
            vec![0.0_f64; p],
            Array2::<f64>::zeros((p, self.layout.block_dim())),
            vec![0.0_f64; p],
        )
    }
}

impl StreamedFrameCurvature for StreamedFrameCurvatureOperator<'_> {
    fn param_dim(&self) -> usize {
        self.layout.param_dim()
    }

    fn root_rows(&self) -> usize {
        self.root_rows
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), String> {
        let param_dim = self.layout.param_dim();
        if x.len() != param_dim || y.len() != param_dim {
            return Err(format!(
                "streamed frame curvature: matvec shapes ({}, {}) must both be param_dim = \
                 {param_dim}",
                x.len(),
                y.len()
            ));
        }
        y.fill(0.0);
        let p = self.term.output_dim();
        let d = self.layout.block_dim();
        let rank = self.metric.metric_rank();
        let (mut tangent, mut g, mut a) = self.scratch();
        let mut c = vec![0.0_f64; rank];
        for row in 0..self.term.n_obs() {
            if !self
                .term
                .fill_row_frame_jacobian(row, &self.assignments, self.layout, &mut tangent, &mut g)
            {
                // No assignment mass: this observation contributes only zero
                // rows to `R`, so it contributes nothing to `Hx`.
                continue;
            }
            // a = J_n x.
            for i in 0..p {
                let mut acc = 0.0_f64;
                for l in 0..d {
                    acc += g[[i, l]] * x[self.layout.column(i, l)];
                }
                a[i] = acc;
            }
            // c = U_nᵀ a — the root rows' readings of `x` at this observation.
            for (r, slot) in c.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for i in 0..p {
                    acc += self.metric.factor_entry(row, i, r) * a[i];
                }
                *slot = acc;
            }
            // y += J_nᵀ U_n c = J_nᵀ M_n J_n x.
            for i in 0..p {
                let mut b = 0.0_f64;
                for (r, &cr) in c.iter().enumerate() {
                    b += self.metric.factor_entry(row, i, r) * cr;
                }
                if b == 0.0 {
                    continue;
                }
                for l in 0..d {
                    y[self.layout.column(i, l)] += g[[i, l]] * b;
                }
            }
        }
        for r in 0..self.dense_rows.nrows() {
            let pin = self.dense_rows.row(r);
            let dot: f64 = pin.iter().zip(x.iter()).map(|(u, v)| u * v).sum();
            if dot == 0.0 {
                continue;
            }
            for (col, value) in pin.iter().enumerate() {
                y[col] += value * dot;
            }
        }
        Ok(())
    }

    fn diagonal(&self) -> Result<Array1<f64>, String> {
        let param_dim = self.layout.param_dim();
        let mut diagonal = Array1::<f64>::zeros(param_dim);
        if param_dim == 0 {
            return Ok(diagonal);
        }
        let p = self.term.output_dim();
        let d = self.layout.block_dim();
        let rank = self.metric.metric_rank();
        let (mut tangent, mut g, mut m_ii) = self.scratch();
        for row in 0..self.term.n_obs() {
            if !self
                .term
                .fill_row_frame_jacobian(row, &self.assignments, self.layout, &mut tangent, &mut g)
            {
                continue;
            }
            // `H[(i,l),(i,l)] = Σ_n M_n[i,i]·g_n[i,l]²`, and `M_n[i,i] = Σ_r
            // U[n,i,r]²` — the only entry of `M_n` a diagonal read needs.
            for (i, slot) in m_ii.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for r in 0..rank {
                    let u = self.metric.factor_entry(row, i, r);
                    acc += u * u;
                }
                *slot = acc;
            }
            for i in 0..p {
                let w = m_ii[i];
                if w == 0.0 {
                    continue;
                }
                for l in 0..d {
                    let v = g[[i, l]];
                    diagonal[self.layout.column(i, l)] += w * v * v;
                }
            }
        }
        for r in 0..self.dense_rows.nrows() {
            for (col, value) in self.dense_rows.row(r).iter().enumerate() {
                diagonal[col] += value * value;
            }
        }
        Ok(diagonal)
    }

    fn project_root(&self, directions: &[ArrayView1<'_, f64>]) -> Result<Array2<f64>, String> {
        let param_dim = self.layout.param_dim();
        let count = directions.len();
        for (j, direction) in directions.iter().enumerate() {
            if direction.len() != param_dim {
                return Err(format!(
                    "streamed frame curvature: direction {j} has {} entries but param_dim = \
                     {param_dim}",
                    direction.len()
                ));
            }
        }
        if count == 0 || param_dim == 0 {
            return Ok(Array2::<f64>::zeros((count, count)));
        }
        let p = self.term.output_dim();
        let d = self.layout.block_dim();
        let rank = self.metric.metric_rank();
        let mut accumulator = TriangularRootAccumulator::new(count);
        let (mut tangent, mut g, _) = self.scratch();
        // One observation contributes `rank` rows to `W = RΞ`, and this is them:
        // `columns[j*rank + r]` is `W[(n,r), j] = (U_nᵀ J_n ξ_j)[r]`. Held as one
        // flat buffer for the whole pass so the parallel region below writes
        // disjoint `rank`-length slices and allocates nothing per generator.
        let mut columns = vec![0.0_f64; count * rank];
        let mut root_row = vec![0.0_f64; count];
        for row in 0..self.term.n_obs() {
            if !self
                .term
                .fill_row_frame_jacobian(row, &self.assignments, self.layout, &mut tangent, &mut g)
            {
                continue;
            }
            columns.fill(0.0);
            // The one parallel region in this module, and it is over GENERATORS
            // rather than over observations — deliberately.
            //
            // A generator's column of `W` is computed from `g` and that
            // generator alone and is written to its own slice, so there is no
            // reduction and therefore no summation order to depend on the
            // schedule. Splitting the OBSERVATIONS instead would need per-chunk
            // partial factors combined pairwise, and Givens rotations do not
            // commute: the certificate would stop being bit-reproducible across
            // runs, which is a property it is asserted to have (two replicate
            // fits are "identified up to the same group" iff their signatures are
            // equal). This way the serial and parallel passes are the same
            // arithmetic in the same order, not merely the same in distribution.
            //
            // There is enough work here to be worth splitting: one observation is
            // `p·G·(D + rank)` flops, and it is `G` — not `n` — that grows with
            // the dictionary (`D(D−1)/2` frame rotations plus `K(K−1)/2` atom
            // exchanges).
            let contract = |j: usize, slot: &mut [f64]| {
                let direction = &directions[j];
                for i in 0..p {
                    let mut a = 0.0_f64;
                    for l in 0..d {
                        let gil = g[[i, l]];
                        if gil != 0.0 {
                            a += gil * direction[self.layout.column(i, l)];
                        }
                    }
                    if a == 0.0 {
                        continue;
                    }
                    for (r, c) in slot.iter_mut().enumerate() {
                        *c += self.metric.factor_entry(row, i, r) * a;
                    }
                }
            };
            if count > 1 && gam_runtime::parallel::at_top_level() {
                use rayon::prelude::*;
                gam_runtime::parallel::fan_out(|| {
                    columns
                        .par_chunks_mut(rank)
                        .enumerate()
                        .for_each(|(j, slot)| contract(j, slot))
                });
            } else {
                for (j, slot) in columns.chunks_mut(rank).enumerate() {
                    contract(j, slot);
                }
            }
            for r in 0..rank {
                let mut any = false;
                for (j, slot) in root_row.iter_mut().enumerate() {
                    let value = columns[j * rank + r];
                    any |= value != 0.0;
                    *slot = value;
                }
                if !any {
                    continue;
                }
                accumulator.push_root_row(&mut root_row)?;
                root_row.fill(0.0);
            }
        }
        for r in 0..self.dense_rows.nrows() {
            let pin = self.dense_rows.row(r);
            let mut any = false;
            for (j, direction) in directions.iter().enumerate() {
                let value = pin.iter().zip(direction.iter()).map(|(u, v)| u * v).sum();
                any |= value != 0.0;
                root_row[j] = value;
            }
            if !any {
                continue;
            }
            accumulator.push_root_row(&mut root_row)?;
            root_row.fill(0.0);
        }
        Ok(accumulator.into_factor())
    }
}
