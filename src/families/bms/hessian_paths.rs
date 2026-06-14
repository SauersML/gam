use super::family::BernoulliMarginalSlopeFamily;
use super::*;

/// Block-local psi derivative row: avoids allocating a full p-vector
/// when the psi derivative lives in a single channel (marginal or logslope).
pub(super) struct BlockPsiRow {
    /// Which parameter block (0 = marginal, 1 = logslope).
    pub(super) block_idx: usize,
    /// Coefficient range in global (flat) space for this block.
    pub(super) range: std::ops::Range<usize>,
    /// The p_block-length psi design derivative row.
    pub(super) local_vec: Array1<f64>,
}

pub(super) struct PsiAxisSpec {
    pub(super) block_idx: usize,
    pub(super) idx_primary: usize,
    pub(super) psi_map: crate::families::custom_family::PsiDesignMap,
}

#[derive(Clone)]
pub(super) struct BlockSlices {
    pub(super) marginal: std::ops::Range<usize>,
    pub(super) logslope: std::ops::Range<usize>,
    pub(super) h: Option<std::ops::Range<usize>>,
    pub(super) w: Option<std::ops::Range<usize>>,
    pub(super) total: usize,
}

pub(super) fn block_slices(family: &BernoulliMarginalSlopeFamily) -> BlockSlices {
    let mut cursor = 0usize;
    let marginal = cursor..cursor + family.marginal_design.ncols();
    cursor = marginal.end;
    let logslope = cursor..cursor + family.logslope_design.ncols();
    cursor = logslope.end;
    let h = family.score_warp.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    let w = family.link_dev.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    BlockSlices {
        marginal,
        logslope,
        h,
        w,
        total: cursor,
    }
}

#[derive(Clone)]
pub(super) struct PrimarySlices {
    pub(super) q: usize,
    pub(super) logslope: usize,
    pub(super) h: Option<std::ops::Range<usize>>,
    pub(super) w: Option<std::ops::Range<usize>>,
    pub(super) total: usize,
}

pub(super) fn primary_slices(slices: &BlockSlices) -> PrimarySlices {
    let q = 0usize;
    let logslope = 1usize;
    let mut cursor = 2usize;
    let h = slices.h.as_ref().map(|range| {
        let out = cursor..cursor + range.len();
        cursor = out.end;
        out
    });
    let w = slices.w.as_ref().map(|range| {
        let out = cursor..cursor + range.len();
        cursor = out.end;
        out
    });
    PrimarySlices {
        q,
        logslope,
        h,
        w,
        total: cursor,
    }
}
// ── Block-local Hessian accumulator for Bernoulli marginal-slope ─────
//
// The two large blocks are marginal (p_m) and logslope (p_g).
// Optional h/w blocks are tiny (1-5 params each), so their contributions
// go into a dense p_total x p_total correction matrix.  The main savings
// is avoiding O(n * (p_m^2 + p_g^2)) dense accumulation into a full p*p target.

pub(super) struct BernoulliBlockHessianAccumulator {
    pub(super) h_mm: Array2<f64>,
    pub(super) h_gg: Array2<f64>,
    pub(super) h_mg: Array2<f64>,
    pub(super) dense_correction: Option<Array2<f64>>,
}

impl BernoulliBlockHessianAccumulator {
    pub(super) fn new(slices: &BlockSlices) -> Self {
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let has_hw = slices.h.is_some() || slices.w.is_some();
        Self {
            h_mm: Array2::zeros((p_m, p_m)),
            h_gg: Array2::zeros((p_g, p_g)),
            h_mg: Array2::zeros((p_m, p_g)),
            dense_correction: if has_hw {
                Some(Array2::zeros((slices.total, slices.total)))
            } else {
                None
            },
        }
    }

    /// Accumulate a primary-space Hessian into block-local matrices.
    /// The marginal block uses H[0,0], logslope uses H[1,1],
    /// cross uses H[0,1].  All h/w cross-blocks go to dense_correction.
    pub(super) fn add_pullback(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;

        // marginal x marginal: H[0,0] * x_row outer x_row
        family
            .marginal_design
            .syr_row_into(row, h[[0, 0]], &mut self.h_mm)
            .expect("marginal syr_row_into dimension mismatch");

        // logslope x logslope: H[1,1] * g_row outer g_row
        family
            .logslope_design
            .syr_row_into(row, h[[1, 1]], &mut self.h_gg)
            .expect("logslope syr_row_into dimension mismatch");

        // marginal x logslope: H[0,1] * x_row outer g_row
        if h[[0, 1]] != 0.0 {
            family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &family.logslope_design,
                    h[[0, 1]],
                    self.h_mg.view_mut(),
                )
                .expect("marginal-logslope row_outer_into dimension mismatch");
        }

        // h/w cross-blocks -> dense_correction
        if let Some(ref mut dc) = self.dense_correction {
            family.add_pullback_primary_hessian_hw_only(dc, row, slices, primary, h.view());
        }
    }

    /// Fast-path pullback for the rigid (no flex / no h/w / no dense_correction)
    /// joint-Hessian directional-derivative path. Takes the 2x2 contracted
    /// Hessian as a stack `[[f64; 2]; 2]` plus a scalar weight, so the caller
    /// does not allocate an `Array2` per row.
    ///
    /// Equivalent to `add_pullback` with `h[i][j] = t[i][j] * w` but skips the
    /// `dense_correction` branch — which is `None` whenever this method is
    /// reached because the flex-inactive path constructs the accumulator from
    /// `BlockSlices` with no `h`/`w` ranges.
    pub(super) fn add_pullback_rigid_2x2(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        t: &[[f64; 2]; 2],
        w: f64,
    ) {
        assert!(
            self.dense_correction.is_none(),
            "add_pullback_rigid_2x2 called on accumulator with dense_correction"
        );
        let h00 = t[0][0] * w;
        let h11 = t[1][1] * w;
        let h01 = t[0][1] * w;

        family
            .marginal_design
            .syr_row_into(row, h00, &mut self.h_mm)
            .expect("marginal syr_row_into dimension mismatch");

        family
            .logslope_design
            .syr_row_into(row, h11, &mut self.h_gg)
            .expect("logslope syr_row_into dimension mismatch");

        if h01 != 0.0 {
            family
                .marginal_design
                .row_outer_into_view(row, &family.logslope_design, h01, self.h_mg.view_mut())
                .expect("marginal-logslope row_outer_into dimension mismatch");
        }
    }

    pub(super) fn add_hw_pullback_only(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: &Array2<f64>,
    ) {
        if let Some(ref mut dc) = self.dense_correction {
            family.add_pullback_primary_hessian_hw_only(
                dc,
                row,
                slices,
                primary,
                primary_hessian.view(),
            );
        }
    }

    pub(super) fn add_weighted_design_grams(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        rows: std::ops::Range<usize>,
        w_mm: &Array1<f64>,
        w_mg: &Array1<f64>,
        w_gg: &Array1<f64>,
    ) -> Result<(), String> {
        // Zero-copy fast path: when BOTH designs are materialised dense the
        // weighted Gram products read the chunk rows straight from the stored
        // matrix as borrowed views. `try_row_chunk` would instead `.to_owned()`
        // a fresh `(rows × p)` `Array2` every chunk every cycle — the dominant
        // `OwnedRepr<f64>` alloc/`drop_in_place` churn the cold marginal-slope
        // fit pays in its repeated inner Newton / ρ-homotopy pre-warm passes.
        // `fast_xt_diag_*` is generic over `Data<Elem = f64>`, so an
        // `ArrayView2` slice feeds the identical BLAS-3 kernels with identical
        // arithmetic — exact, just without the copy.
        if let (Some(x_full), Some(g_full)) = (
            family.marginal_design.as_dense_ref(),
            family.logslope_design.as_dense_ref(),
        ) {
            let x = x_full.slice(s![rows.clone(), ..]);
            let g = g_full.slice(s![rows, ..]);
            self.add_weighted_design_grams_from_chunks(&x, &g, w_mm, w_mg, w_gg);
            return Ok(());
        }
        let x = family
            .marginal_design
            .try_row_chunk(rows.clone())
            .map_err(|e| format!("bernoulli marginal_design try_row_chunk: {e}"))?;
        let g = family
            .logslope_design
            .try_row_chunk(rows)
            .map_err(|e| format!("bernoulli logslope_design try_row_chunk: {e}"))?;
        self.add_weighted_design_grams_from_chunks(&x, &g, w_mm, w_mg, w_gg);
        Ok(())
    }

    pub(super) fn add_weighted_design_grams_from_chunks<
        SX: ndarray::Data<Elem = f64>,
        SG: ndarray::Data<Elem = f64>,
    >(
        &mut self,
        x: &ndarray::ArrayBase<SX, ndarray::Ix2>,
        g: &ndarray::ArrayBase<SG, ndarray::Ix2>,
        w_mm: &Array1<f64>,
        w_mg: &Array1<f64>,
        w_gg: &Array1<f64>,
    ) {
        self.h_mm += &crate::faer_ndarray::fast_xt_diag_x(x, w_mm);
        if w_mg.iter().any(|value| *value != 0.0) {
            self.h_mg += &crate::faer_ndarray::fast_xt_diag_y(x, w_mg, g);
        }
        self.h_gg += &crate::faer_ndarray::fast_xt_diag_x(g, w_gg);
    }

    /// Batch the exact h/w pullback terms for one row chunk.
    ///
    /// The large marginal/logslope blocks are already accumulated as chunked
    /// weighted Gram products.  h/w used to be the remaining per-row dense
    /// path: for every sampled row and every h/w coordinate we performed two
    /// design-row AXPYs (column plus symmetric row).  At large-scale `n` that
    /// repeated row materialization dominates even though the h/w blocks are
    /// tiny.  This routine keeps the same exact Hessian entries, but turns the
    /// cross terms into `X_chunk^T weights` / `G_chunk^T weights` products and
    /// sums the tiny h/w self-blocks in registers.
    pub(super) fn add_weighted_hw_cross_terms(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        rows: std::ops::Range<usize>,
        slices: &BlockSlices,
        h_q: Option<&Array2<f64>>,
        h_g: Option<&Array2<f64>>,
        h_h: Option<&Array2<f64>>,
        w_q: Option<&Array2<f64>>,
        w_g: Option<&Array2<f64>>,
        h_w: Option<&Array2<f64>>,
        w_w: Option<&Array2<f64>>,
    ) -> Result<(), String> {
        let Some(dc) = self.dense_correction.as_mut() else {
            return Ok(());
        };

        let need_marginal = h_q.is_some() || w_q.is_some();
        let need_logslope = h_g.is_some() || w_g.is_some();
        let x = if need_marginal {
            Some(
                family
                    .marginal_design
                    .try_row_chunk(rows.clone())
                    .map_err(|e| format!("bernoulli marginal_design try_row_chunk: {e}"))?,
            )
        } else {
            None
        };
        let g = if need_logslope {
            Some(
                family
                    .logslope_design
                    .try_row_chunk(rows)
                    .map_err(|e| format!("bernoulli logslope_design try_row_chunk: {e}"))?,
            )
        } else {
            None
        };

        if let (Some(block_h), Some(hq)) = (slices.h.as_ref(), h_q) {
            let x = x.as_ref().expect("marginal chunk for h/q cross");
            let cross = crate::faer_ndarray::fast_atb(x, hq);
            for (local_idx, global_idx) in block_h.clone().enumerate() {
                let col = cross.column(local_idx);
                dc.slice_mut(s![slices.marginal.clone(), global_idx])
                    .scaled_add(1.0, &col);
                dc.slice_mut(s![global_idx, slices.marginal.clone()])
                    .scaled_add(1.0, &col);
            }
        }
        if let (Some(block_h), Some(hg)) = (slices.h.as_ref(), h_g) {
            let g = g.as_ref().expect("logslope chunk for h/g cross");
            let cross = crate::faer_ndarray::fast_atb(g, hg);
            for (local_idx, global_idx) in block_h.clone().enumerate() {
                let col = cross.column(local_idx);
                dc.slice_mut(s![slices.logslope.clone(), global_idx])
                    .scaled_add(1.0, &col);
                dc.slice_mut(s![global_idx, slices.logslope.clone()])
                    .scaled_add(1.0, &col);
            }
        }
        if let (Some(block_h), Some(hh)) = (slices.h.as_ref(), h_h) {
            dc.slice_mut(s![block_h.clone(), block_h.clone()])
                .scaled_add(1.0, hh);
        }

        if let (Some(block_w), Some(wq)) = (slices.w.as_ref(), w_q) {
            let x = x.as_ref().expect("marginal chunk for w/q cross");
            let cross = crate::faer_ndarray::fast_atb(x, wq);
            for (local_idx, global_idx) in block_w.clone().enumerate() {
                let col = cross.column(local_idx);
                dc.slice_mut(s![slices.marginal.clone(), global_idx])
                    .scaled_add(1.0, &col);
                dc.slice_mut(s![global_idx, slices.marginal.clone()])
                    .scaled_add(1.0, &col);
            }
        }
        if let (Some(block_w), Some(wg)) = (slices.w.as_ref(), w_g) {
            let g = g.as_ref().expect("logslope chunk for w/g cross");
            let cross = crate::faer_ndarray::fast_atb(g, wg);
            for (local_idx, global_idx) in block_w.clone().enumerate() {
                let col = cross.column(local_idx);
                dc.slice_mut(s![slices.logslope.clone(), global_idx])
                    .scaled_add(1.0, &col);
                dc.slice_mut(s![global_idx, slices.logslope.clone()])
                    .scaled_add(1.0, &col);
            }
        }
        if let (Some(block_h), Some(block_w), Some(hw)) =
            (slices.h.as_ref(), slices.w.as_ref(), h_w)
        {
            dc.slice_mut(s![block_h.clone(), block_w.clone()])
                .scaled_add(1.0, hw);
            dc.slice_mut(s![block_w.clone(), block_h.clone()])
                .scaled_add(1.0, &hw.t());
        }
        if let (Some(block_w), Some(ww)) = (slices.w.as_ref(), w_w) {
            dc.slice_mut(s![block_w.clone(), block_w.clone()])
                .scaled_add(1.0, ww);
        }

        Ok(())
    }

    /// Add a rank-1 update from psi_row (in the psi block) crossed with the
    /// pullback of a primary-space vector.  Adds both left outer right and right outer left.
    ///
    /// psi_row lives in block `psi_block_idx` (0=marginal, 1=logslope).
    /// right_primary is a primary-space vector; its [0] component maps to marginal,
    /// [1] to logslope, and the rest to h/w (dense correction).
    ///
    /// Design rows are materialized once via `try_row_chunk` and reused across
    /// the psi-index rank-1 sweeps.  Without that, `axpy_row_into` on a Lazy
    /// operator re-dispatches `row_chunk_into` for every nonzero psi index
    /// (psi_dim×rank-2 = 2*psi_dim row materializations per call), which is
    /// the dominant cost of joint-spatial Hessian builds at large scale.
    pub(super) fn add_rank1_psi_cross(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        psi_block_idx: usize,
        psi_row: &Array1<f64>,
        right_primary: &Array1<f64>,
    ) {
        let need_marg = right_primary[0] != 0.0;
        let need_log = right_primary[1] != 0.0;
        let marg_chunk = if need_marg {
            Some(
                family
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .expect("marginal try_row_chunk in add_rank1_psi_cross"),
            )
        } else {
            None
        };
        let log_chunk = if need_log {
            Some(
                family
                    .logslope_design
                    .try_row_chunk(row..row + 1)
                    .expect("logslope try_row_chunk in add_rank1_psi_cross"),
            )
        } else {
            None
        };
        let x_row = marg_chunk.as_ref().map(|c| c.row(0));
        let g_row = log_chunk.as_ref().map(|c| c.row(0));

        // Marginal component of right_primary
        if let Some(x_row) = x_row {
            match psi_block_idx {
                0 => {
                    // psi=marginal, right=marginal -> h_mm, symmetric rank-2
                    // h_mm += s * (psi outer x_row + x_row outer psi)
                    let s = right_primary[0];
                    let p = x_row.len();
                    assert_eq!(psi_row.len(), p);
                    assert_eq!(self.h_mm.nrows(), p);
                    assert_eq!(self.h_mm.ncols(), p);
                    for i in 0..p {
                        let psi_i = psi_row[i];
                        if psi_i == 0.0 {
                            continue;
                        }
                        let coef = s * psi_i;
                        let mut row_i = self.h_mm.row_mut(i);
                        for j in 0..p {
                            row_i[j] += coef * x_row[j];
                        }
                        // Transpose half: h_mm.col(i) += coef * x_row
                        for j in 0..p {
                            self.h_mm[[j, i]] += coef * x_row[j];
                        }
                    }
                }
                1 => {
                    // psi=logslope, right=marginal -> h_mg (marginal x logslope)
                    // h_mg += right_primary[0] * outer(x_row, psi)
                    let s = right_primary[0];
                    let pm = x_row.len();
                    let pl = psi_row.len();
                    assert_eq!(self.h_mg.nrows(), pm);
                    assert_eq!(self.h_mg.ncols(), pl);
                    for j in 0..pl {
                        let psi_j = psi_row[j];
                        if psi_j == 0.0 {
                            continue;
                        }
                        let coef = s * psi_j;
                        for i in 0..pm {
                            self.h_mg[[i, j]] += coef * x_row[i];
                        }
                    }
                }
                _ => {}
            }
        }

        // Logslope component of right_primary
        if let Some(g_row) = g_row {
            match psi_block_idx {
                0 => {
                    // psi=marginal, right=logslope -> h_mg (marginal x logslope)
                    // h_mg += right_primary[1] * outer(psi, g_row)
                    let s = right_primary[1];
                    let pm = psi_row.len();
                    let pl = g_row.len();
                    assert_eq!(self.h_mg.nrows(), pm);
                    assert_eq!(self.h_mg.ncols(), pl);
                    for i in 0..pm {
                        let psi_i = psi_row[i];
                        if psi_i == 0.0 {
                            continue;
                        }
                        let coef = s * psi_i;
                        let mut row_i = self.h_mg.row_mut(i);
                        for j in 0..pl {
                            row_i[j] += coef * g_row[j];
                        }
                    }
                }
                1 => {
                    // psi=logslope, right=logslope -> h_gg, symmetric rank-2
                    // h_gg += s * (psi outer g_row + g_row outer psi)
                    let s = right_primary[1];
                    let p = g_row.len();
                    assert_eq!(psi_row.len(), p);
                    assert_eq!(self.h_gg.nrows(), p);
                    assert_eq!(self.h_gg.ncols(), p);
                    for i in 0..p {
                        let psi_i = psi_row[i];
                        if psi_i == 0.0 {
                            continue;
                        }
                        let coef = s * psi_i;
                        let mut row_i = self.h_gg.row_mut(i);
                        for j in 0..p {
                            row_i[j] += coef * g_row[j];
                        }
                        for j in 0..p {
                            self.h_gg[[j, i]] += coef * g_row[j];
                        }
                    }
                }
                _ => {}
            }
        }

        // h/w components -> dense_correction
        if let Some(ref mut dc) = self.dense_correction {
            let psi_range = if psi_block_idx == 0 {
                slices.marginal.clone()
            } else {
                slices.logslope.clone()
            };
            if let (Some(ph), Some(bh)) = (primary.h.as_ref(), slices.h.as_ref()) {
                let h_part = right_primary.slice(ndarray::s![ph.start..ph.end]);
                for (li, gi) in psi_range.clone().enumerate() {
                    for (lj, gj) in bh.clone().enumerate() {
                        let val = psi_row[li] * h_part[lj];
                        dc[[gi, gj]] += val;
                        dc[[gj, gi]] += val;
                    }
                }
            }
            if let (Some(pw), Some(bw)) = (primary.w.as_ref(), slices.w.as_ref()) {
                let w_part = right_primary.slice(ndarray::s![pw.start..pw.end]);
                for (li, gi) in psi_range.enumerate() {
                    for (lj, gj) in bw.clone().enumerate() {
                        let val = psi_row[li] * w_part[lj];
                        dc[[gi, gj]] += val;
                        dc[[gj, gi]] += val;
                    }
                }
            }
        }
    }

    /// Add outer product of two psi block-local rows (possibly in different blocks).
    /// Adds both alpha * (a outer b) and alpha * (b outer a) to maintain symmetry.
    pub(super) fn add_psi_psi_outer(
        &mut self,
        block_i: usize,
        psi_row_i: &Array1<f64>,
        block_j: usize,
        psi_row_j: &Array1<f64>,
        alpha: f64,
    ) {
        add_two_surface_psi_outer(
            block_i,
            psi_row_i,
            block_j,
            psi_row_j,
            alpha,
            0,
            1,
            &mut self.h_mm,
            &mut self.h_gg,
            &mut self.h_mg,
        );
    }

    /// Merge another accumulator into this one (for parallel reduce).
    pub(super) fn add(&mut self, other: &BernoulliBlockHessianAccumulator) {
        self.h_mm += &other.h_mm;
        self.h_gg += &other.h_gg;
        self.h_mg += &other.h_mg;
        if let (Some(ref mut dc), Some(odc)) = (
            self.dense_correction.as_mut(),
            other.dense_correction.as_ref(),
        ) {
            dc.scaled_add(1.0, odc);
        }
    }

    pub(super) fn to_dense(&self, slices: &BlockSlices) -> Array2<f64> {
        let mut out = Array2::zeros((slices.total, slices.total));
        out.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .assign(&self.h_gg);
        out.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
            .assign(&self.h_mg.t());
        if let Some(ref dc) = self.dense_correction {
            out += dc;
        }
        out
    }

    pub(super) fn into_operator(self, slices: &BlockSlices) -> BernoulliBlockHessianOperator {
        BernoulliBlockHessianOperator {
            h_mm: self.h_mm,
            h_gg: self.h_gg,
            h_mg: self.h_mg,
            dense_correction: self.dense_correction,
            marginal: slices.marginal.clone(),
            logslope: slices.logslope.clone(),
            total: slices.total,
        }
    }
}

/// Block-structured HyperOperator for Bernoulli marginal-slope psi Hessians.
/// Stores 3 block matrices (h_mm, h_gg, h_mg) plus an optional dense correction
/// for h/w cross-blocks.  Matvec is O(p_m^2 + p_g^2 + p_m*p_g) for the block part,
/// plus O(p_total^2) only if h/w blocks exist (which is rare and tiny).
pub(super) struct BernoulliBlockHessianOperator {
    pub(super) h_mm: Array2<f64>,
    pub(super) h_gg: Array2<f64>,
    pub(super) h_mg: Array2<f64>,
    pub(super) dense_correction: Option<Array2<f64>>,
    pub(super) marginal: std::ops::Range<usize>,
    pub(super) logslope: std::ops::Range<usize>,
    pub(super) total: usize,
}

impl HyperOperator for BernoulliBlockHessianOperator {
    fn dim(&self) -> usize {
        self.total
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(self.total);
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::zeros(self.total);
        self.mul_vec_into(v, out.view_mut());
        out
    }

    /// Write the block-structured matrix-vector product directly into caller-
    /// owned storage. Avoids the two intermediate `Array1` allocations that
    /// the default `mul_vec_into → mul_vec_view → to_owned + mul_vec` chain
    /// would incur. The psi-Hessian outer-eval path calls this once per
    /// ψ-direction per trace sweep; at large scale (rank ≈ 32) the saving
    /// is ~64 allocations per REML gradient step.
    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        let v_m = v.slice(s![self.marginal.clone()]);
        let v_g = v.slice(s![self.logslope.clone()]);
        out.fill(0.0);
        {
            let mut o_m = out.slice_mut(s![self.marginal.clone()]);
            o_m += &self.h_mm.dot(&v_m);
            o_m += &self.h_mg.dot(&v_g);
        }
        {
            let mut o_g = out.slice_mut(s![self.logslope.clone()]);
            o_g += &self.h_mg.t().dot(&v_m);
            o_g += &self.h_gg.dot(&v_g);
        }
        if let Some(ref dc) = self.dense_correction {
            out += &dc.dot(&v.to_owned());
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_m = v.slice(s![self.marginal.clone()]);
        let v_g = v.slice(s![self.logslope.clone()]);
        let u_m = u.slice(s![self.marginal.clone()]);
        let u_g = u.slice(s![self.logslope.clone()]);
        // Diagonal blocks
        let mut total = v_m.dot(&self.h_mm.dot(&u_m));
        total += v_g.dot(&self.h_gg.dot(&u_g));
        // Off-diagonal blocks (symmetric)
        total += v_m.dot(&self.h_mg.dot(&u_g));
        total += v_g.dot(&self.h_mg.t().dot(&u_m));
        // Dense correction
        if let Some(ref dc) = self.dense_correction {
            total += v.dot(&dc.dot(u));
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.total, self.total));
        out.slice_mut(s![self.marginal.clone(), self.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![self.logslope.clone(), self.logslope.clone()])
            .assign(&self.h_gg);
        out.slice_mut(s![self.marginal.clone(), self.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![self.logslope.clone(), self.marginal.clone()])
            .assign(&self.h_mg.t());
        if let Some(ref dc) = self.dense_correction {
            out += dc;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub(super) struct CachedDenestedCellMoments {
    pub(super) partition_cell: exact_kernel::DenestedPartitionCell,
    /// Cell-moment state evaluated at this row's converged intercept for the
    /// current PIRLS/Newton cycle. Stored at whatever `max_degree` the cache
    /// was built with (degree 9 for the per-row lazy fallback; up to degree
    /// 21 for the pre-built `RowCellMomentsBundle`).
    pub(super) state: exact_kernel::CellDerivativeMomentState,
}

/// Pre-built per-row cell moments for the current β snapshot. Built once at
/// the top of the joint-Newton cycle (after row intercepts are solved) and
/// reused by every gradient/Hessian/Hv/diagonal/derivative-tensor pass at the
/// same β.
#[derive(Clone)]
pub(super) struct RowCellMomentsBundle {
    pub(super) max_degree: usize,
    pub(super) selected_rows: usize,
    pub(super) rows: Vec<Option<Vec<CachedDenestedCellMoments>>>,
}

impl RowCellMomentsBundle {
    /// Return the pre-built cell moments for `row` when the bundle was
    /// constructed at `>= required_degree`.  Returns `None` both when the
    /// row has no data (e.g. excluded by subsample mask) *and* when the
    /// bundle's `max_degree` is too low for this caller — the caller must
    /// fall back to per-row on-demand evaluation in either case.
    #[inline]
    pub(super) fn row(
        &self,
        row: usize,
        required_degree: usize,
    ) -> Option<&[CachedDenestedCellMoments]> {
        if self.max_degree < required_degree {
            return None;
        }
        self.rows
            .get(row)
            .and_then(Option::as_ref)
            .map(Vec::as_slice)
    }

    #[inline]
    pub(super) fn covers_all_rows(&self) -> bool {
        self.rows.iter().all(Option::is_some)
    }

    pub(super) fn estimated_resident_bytes(
        n_rows: usize,
        n_cells: usize,
        max_degree: usize,
    ) -> usize {
        let row_vecs =
            n_rows.saturating_mul(std::mem::size_of::<Option<Vec<CachedDenestedCellMoments>>>());
        let cell_records = n_cells.saturating_mul(std::mem::size_of::<CachedDenestedCellMoments>());
        let required_moments = max_degree.saturating_add(1);
        let moment_payload = if required_moments > exact_kernel::CELL_MOMENT_INLINE_CAPACITY {
            n_cells
                .saturating_mul(required_moments)
                .saturating_mul(std::mem::size_of::<f64>())
        } else {
            0
        };
        row_vecs
            .saturating_add(cell_records)
            .saturating_add(moment_payload)
    }
}

#[derive(Clone)]
pub(super) struct BernoulliMarginalSlopeRowExactContext {
    pub(super) intercept: f64,
    pub(super) m_a: f64,
    pub(super) intercept_fast_path: bool,
    /// Degree-9 per-row cell moments at the converged row intercept. The
    /// top-of-cycle [`RowCellMomentsBundle`] (built at degree 9) is preferred
    /// when present; this field remains the per-row lazy fallback for callers
    /// without a bundle (e.g. legacy direct call sites).
    pub(super) degree9_cells: Option<Vec<CachedDenestedCellMoments>>,
}

pub(super) struct BernoulliMarginalSlopeFlexRowScratch {
    pub(super) m_u: Array1<f64>,
    pub(super) m_au: Array1<f64>,
    pub(super) m_uv: Array2<f64>,
    pub(super) a_u: Array1<f64>,
    pub(super) a_uv: Array2<f64>,
    pub(super) rho: Array1<f64>,
    pub(super) tau: Array1<f64>,
    pub(super) grad: Array1<f64>,
    pub(super) hess: Array2<f64>,
    // Per-row [f64; 4] coefficient buffers used by the flex analytic path. Owned
    // by the scratch so the hot path never allocates a fresh `Vec` per row.
    pub(super) coeff_u: Vec<[f64; 4]>,
    pub(super) coeff_au: Vec<[f64; 4]>,
    pub(super) coeff_bu: Vec<[f64; 4]>,
    pub(super) g_u_fixed: Vec<[f64; 4]>,
    pub(super) g_au_fixed: Vec<[f64; 4]>,
    pub(super) g_bu_fixed: Vec<[f64; 4]>,
    // Per-cell eta_u buffer for the empirical-grid branch; reused across cells
    // and rows. `Vec<f64>` rather than `Array1` because indexing as
    // `eta_u[idx]` after a `clear()`/`resize()` matches the previous code path.
    pub(super) eta_u_cell: Vec<f64>,
    // Constant zero coeff slice shared by every SparsePrimaryCoeffJetView call
    // that needs `aa_first..bbb_first`. Sized to `primary_dim` once and never
    // mutated thereafter.
    pub(super) zero_family: Vec<[f64; 4]>,
}

impl BernoulliMarginalSlopeFlexRowScratch {
    pub(super) fn new(primary_dim: usize) -> Self {
        Self {
            m_u: Array1::zeros(primary_dim),
            m_au: Array1::zeros(primary_dim),
            m_uv: Array2::zeros((primary_dim, primary_dim)),
            a_u: Array1::zeros(primary_dim),
            a_uv: Array2::zeros((primary_dim, primary_dim)),
            rho: Array1::zeros(primary_dim),
            tau: Array1::zeros(primary_dim),
            grad: Array1::zeros(primary_dim),
            hess: Array2::zeros((primary_dim, primary_dim)),
            coeff_u: vec![[0.0; 4]; primary_dim],
            coeff_au: vec![[0.0; 4]; primary_dim],
            coeff_bu: vec![[0.0; 4]; primary_dim],
            g_u_fixed: vec![[0.0; 4]; primary_dim],
            g_au_fixed: vec![[0.0; 4]; primary_dim],
            g_bu_fixed: vec![[0.0; 4]; primary_dim],
            eta_u_cell: vec![0.0; primary_dim],
            zero_family: vec![[0.0; 4]; primary_dim],
        }
    }

    pub(super) fn reset(&mut self, need_hessian: bool) {
        self.m_u.fill(0.0);
        self.a_u.fill(0.0);
        self.rho.fill(0.0);
        self.tau.fill(0.0);
        self.grad.fill(0.0);
        if need_hessian {
            self.m_au.fill(0.0);
            self.m_uv.fill(0.0);
            self.a_uv.fill(0.0);
            self.hess.fill(0.0);
        }
    }
}

/// Accumulate a flex-block (h or w) per-row gradient and Hessian
/// contribution from the primary-space scratch buffer into the
/// block-local accumulators.
///
/// The flex blocks (link wiggle h, time wiggle w) sit at
/// `primary_range = [start, start+len)` inside the per-row primary-space
/// gradient `scratch.grad` and Hessian `scratch.hess`. Their primary
/// scalars equal the block coefficients (no design pull-back), so the
/// block accumulators are simple sums of the per-row sub-vector and
/// symmetric sub-matrix.
///
/// Mathematical equivalence with the previous index-by-index loop:
/// * `grad[i] += -scratch.grad[start + i]` — applies the
///   `exact_newton_score_component_from_objective_gradient` sign
///   convention (which is just negation) elementwise.
/// * `hess[r, c] += scratch.hess[start + r, start + c]` — full square
///   block, identical to the prior nested for-loop.
///
/// Implementation uses ndarray slice arithmetic so the loop becomes
/// vectorisable contiguous memory traffic instead of a doubly-nested
/// scalar `Array2` `[[r, c]]` index, which was bounds-checked twice
/// per element.
#[inline]
pub(crate) fn accumulate_flex_block_grad_hess(
    primary_range: &std::ops::Range<usize>,
    scratch: &BernoulliMarginalSlopeFlexRowScratch,
    grad: &mut Array1<f64>,
    hess: &mut Array2<f64>,
) {
    let start = primary_range.start;
    let end = primary_range.end;
    let src_g = scratch.grad.slice(s![start..end]);
    // grad += -src_g  (negate to convert objective gradient to score component)
    grad.scaled_add(-1.0, &src_g);
    let src_h = scratch.hess.slice(s![start..end, start..end]);
    *hess += &src_h;
}

pub(super) const COEFF_SUPPORT_BHW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: true,
    include_w: true,
};
pub(super) const COEFF_SUPPORT_BW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: false,
    include_w: true,
};
pub(super) const COEFF_SUPPORT_W: CoeffSupport = CoeffSupport {
    include_primary: false,
    include_h: false,
    include_w: true,
};

pub(super) struct BernoulliExactNewtonAccumulator {
    pub(super) ll: f64,
    pub(super) grad_marginal: Array1<f64>,
    pub(super) grad_logslope: Array1<f64>,
    pub(super) hess_marginal: Array2<f64>,
    pub(super) hess_logslope: Array2<f64>,
    pub(super) grad_h: Option<Array1<f64>>,
    pub(super) grad_w: Option<Array1<f64>>,
    pub(super) hess_h: Option<Array2<f64>>,
    pub(super) hess_w: Option<Array2<f64>>,
}

impl BernoulliExactNewtonAccumulator {
    pub(super) fn new(slices: &BlockSlices) -> Self {
        Self {
            ll: 0.0,
            grad_marginal: Array1::zeros(slices.marginal.len()),
            grad_logslope: Array1::zeros(slices.logslope.len()),
            hess_marginal: Array2::zeros((slices.marginal.len(), slices.marginal.len())),
            hess_logslope: Array2::zeros((slices.logslope.len(), slices.logslope.len())),
            grad_h: slices.h.as_ref().map(|range| Array1::zeros(range.len())),
            grad_w: slices.w.as_ref().map(|range| Array1::zeros(range.len())),
            hess_h: slices
                .h
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
            hess_w: slices
                .w
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
        }
    }

    /// Pull one independent row's flexible primary derivatives back into the
    /// block-diagonal working sets. This method is intentionally row-local:
    /// callers invoke it only on Rayon thread-local accumulators, then merge
    /// whole accumulators after the row sweep completes.
    pub(super) fn add_pullback_block_diagonals(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        primary: &PrimarySlices,
        row_neglog: f64,
        scratch: &BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<(), String> {
        self.ll -= row_neglog;
        {
            let mut marginal = self.grad_marginal.view_mut();
            family.marginal_design.axpy_row_into(
                row,
                BernoulliMarginalSlopeFamily::exact_newton_score_component_from_objective_gradient(
                    scratch.grad[0],
                ),
                &mut marginal,
            )?;
        }
        {
            let mut logslope = self.grad_logslope.view_mut();
            family.logslope_design.axpy_row_into(
                row,
                BernoulliMarginalSlopeFamily::exact_newton_score_component_from_objective_gradient(
                    scratch.grad[1],
                ),
                &mut logslope,
            )?;
        }
        family
            .marginal_design
            .syr_row_into(row, scratch.hess[[0, 0]], &mut self.hess_marginal)?;
        family
            .logslope_design
            .syr_row_into(row, scratch.hess[[1, 1]], &mut self.hess_logslope)?;

        if let (Some(primary_h), Some(grad_h), Some(hess_h)) = (
            primary.h.as_ref(),
            self.grad_h.as_mut(),
            self.hess_h.as_mut(),
        ) {
            accumulate_flex_block_grad_hess(primary_h, scratch, grad_h, hess_h);
        }
        if let (Some(primary_w), Some(grad_w), Some(hess_w)) = (
            primary.w.as_ref(),
            self.grad_w.as_mut(),
            self.hess_w.as_mut(),
        ) {
            accumulate_flex_block_grad_hess(primary_w, scratch, grad_w, hess_w);
        }
        Ok(())
    }

    pub(super) fn add(&mut self, other: &Self) {
        self.ll += other.ll;
        self.grad_marginal += &other.grad_marginal;
        self.grad_logslope += &other.grad_logslope;
        self.hess_marginal += &other.hess_marginal;
        self.hess_logslope += &other.hess_logslope;
        add_optional_vector(&mut self.grad_h, &other.grad_h);
        add_optional_vector(&mut self.grad_w, &other.grad_w);
        add_optional_matrix(&mut self.hess_h, &other.hess_h);
        add_optional_matrix(&mut self.hess_w, &other.hess_w);
    }
}

pub(super) fn add_weighted_chunk_gradient<S: ndarray::Data<Elem = f64>>(
    chunk: &ndarray::ArrayBase<S, ndarray::Ix2>,
    weights: &[f64],
    target: &mut Array1<f64>,
) {
    let weights_view = ndarray::ArrayView1::from(weights);
    *target += &crate::faer_ndarray::fast_atv(chunk, &weights_view);
}

pub(super) fn new_cell_moment_lru_cache(
    policy: &crate::resource::ResourcePolicy,
) -> Arc<exact_kernel::CellMomentLruCache> {
    let budget = policy.max_single_materialization_bytes;
    // The cell-moment memo holds many small entries and is consulted by every
    // row of the parallel exact-cache build (one lookup per evaluated cubic
    // cell, many per row, across all `n` rows in `into_par_iter`). A single
    // lock therefore serializes the whole build — observed as ~1 busy core at
    // large-scale n. Partition the cache well past the worker count so concurrent
    // lookups on distinct cells rarely land in the same shard; entries are
    // small so splitting the byte budget is harmless.
    let shard_count = std::thread::available_parallelism()
        .map(|workers| workers.get().saturating_mul(8))
        .unwrap_or(32)
        .clamp(8, 256);
    Arc::new(exact_kernel::CellMomentLruCache::new_sharded(
        budget,
        shard_count,
    ))
}

pub(super) fn new_cell_moment_cache_stats() -> Arc<exact_kernel::CellMomentCacheStats> {
    Arc::new(exact_kernel::CellMomentCacheStats::default())
}

pub(super) fn add_weighted_chunk_gram<S: ndarray::Data<Elem = f64>>(
    chunk: &ndarray::ArrayBase<S, ndarray::Ix2>,
    weights: &[f64],
    target: &mut Array2<f64>,
) {
    let weights_view = ndarray::ArrayView1::from(weights);
    *target += &crate::faer_ndarray::fast_xt_diag_x(chunk, &weights_view);
}

// Chunk-size and budget constants for row-primary Hessian caches live in
// `super::*` (see BMS_ROW_PRIMARY_HESSIAN_* in `mod.rs`). The trailing
// rationale comments below documented their derivation:
//   * Rows within a chunk are processed sequentially. Flexible exact-Newton
//     caches keep only the pre-solved row context; primary jets are recomputed
//     in chunk-local work to avoid retaining O(n * p_primary^2) Hessian
//     storage.
//   * A single new row-primary Hessian cache may consume up to a fraction of
//     currently-available RAM. The 4× safety margin guards against
//     fragmentation, other workspace allocations, and the rayon parallel
//     build's transient per-thread scratch.
//   * The summed bytes pinned across all live row-primary Hessian caches is
//     capped at a fraction of available RAM at construction time —
//     independent of the per-cache cap so that two co-resident workspaces
//     cannot together consume the whole budget.
