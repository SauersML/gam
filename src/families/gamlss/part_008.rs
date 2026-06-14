
impl crate::solver::estimate::reml::unified::HyperOperator for RowCoeffOperator {
    fn dim(&self) -> usize {
        self.dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.dim);
        let mut scratch = self.acquire_scratch();
        let RowCoeffScratch { u, r } = &mut scratch;

        // 1) u_a = X_a · v[block_a slice]. `fast_av_into` writes directly
        //    into the pre-sized scratch buffer — no per-call n-sized
        //    allocation.
        for (k, ch) in self.channels.iter().enumerate() {
            let start = self.block_offsets[ch.block];
            let width = self.block_widths[ch.block];
            assert_eq!(ch.design.ncols(), width);
            let v_slice = v.slice(s![start..start + width]);
            crate::faer_ndarray::fast_av_into(ch.design.as_ref(), &v_slice, &mut u[k]);
        }

        // 2) r_a = sum_b c_{ab} ⊙ u_b. Zero-then-accumulate; pair coeffs
        //    contribute symmetrically when `a != b`.
        for slot in r.iter_mut() {
            slot.fill(0.0);
        }
        for pair in &self.pair_coeffs {
            let a = pair.a;
            let b = pair.b;
            let coeff = pair
                .coeff
                .as_slice()
                .expect("RowCoeffOperator pair coeff must be contiguous");
            // r[a] += coeff ⊙ u[b]; if a != b also r[b] += coeff ⊙ u[a].
            // Split the borrow so r[a] and r[b] (or u[a] and u[b]) can be
            // accessed simultaneously when a != b.
            if a == b {
                let u_a = u[a]
                    .as_slice()
                    .expect("RowCoeffOperator u must be contiguous");
                let r_a = r[a]
                    .as_slice_mut()
                    .expect("RowCoeffOperator r must be contiguous");
                use rayon::prelude::*;
                r_a.par_iter_mut()
                    .zip(coeff.par_iter())
                    .zip(u_a.par_iter())
                    .for_each(|((r, c), u)| *r += c * u);
            } else {
                let (r_a_slice, r_b_slice) = if a < b {
                    let (left, right) = r.split_at_mut(b);
                    (
                        left[a].as_slice_mut().expect("contiguous"),
                        right[0].as_slice_mut().expect("contiguous"),
                    )
                } else {
                    let (left, right) = r.split_at_mut(a);
                    (
                        right[0].as_slice_mut().expect("contiguous"),
                        left[b].as_slice_mut().expect("contiguous"),
                    )
                };
                let u_a = u[a].as_slice().expect("contiguous");
                let u_b = u[b].as_slice().expect("contiguous");
                use rayon::prelude::*;
                r_a_slice
                    .par_iter_mut()
                    .zip(r_b_slice.par_iter_mut())
                    .zip(coeff.par_iter())
                    .zip(u_a.par_iter())
                    .zip(u_b.par_iter())
                    .for_each(|((((ra, rb), c), ua), ub)| {
                        *ra += c * ub;
                        *rb += c * ua;
                    });
            }
        }

        // 3) Output[block] += X_a^T r_a per channel. Single output alloc.
        let mut out = Array1::<f64>::zeros(self.dim);
        for (k, ch) in self.channels.iter().enumerate() {
            let start = self.block_offsets[ch.block];
            let width = self.block_widths[ch.block];
            let mut block = out.slice_mut(s![start..start + width]);
            // Atv into a temporary, then accumulate; `fast_atv` allocates
            // a `width`-sized array, which is bounded and small relative
            // to the n-sized u/r buffers we already reuse.
            let contrib = fast_atv(ch.design.as_ref(), &r[k]);
            block += &contrib;
        }
        self.release_scratch(scratch);
        out
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ndarray::ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        assert!(start + cols <= self.dim);
        let mut basis = Array1::<f64>::zeros(self.dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(local_col).assign(&col);
            basis[global_col] = 0.0;
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        // Build by basis-vector probing — small-K materialization path.
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        self.mul_basis_columns_into(0, out.view_mut());
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        self.projected_trace(factor)
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &crate::solver::estimate::reml::unified::ProjectedFactorCache,
    ) -> f64 {
        let key = crate::solver::estimate::reml::unified::ProjectedFactorKey::from_factor_view(
            self.projected_pair_gram_cache_id(),
            factor.view(),
        );
        let grams = cache.get_or_insert_with(key, || self.projected_pair_gram_table(factor));
        self.trace_from_pair_gram_table(grams.view())
    }

    fn is_implicit(&self) -> bool {
        true
    }
}


/// Two-block row-coefficient operator backed by `DesignMatrix`.
///
/// This is the operator-form counterpart to `DesignTwoBlockRowCoeffOperator`'s
/// old dense-array storage: it must keep the realized block designs lazy all
/// the way through `Xv` and `X^T r`. Do not cache `Array2` snapshots here;
/// `NoDensifyOperator` regression tests rely on this type to panic if a future
/// change materializes spec-backed designs.
struct DesignTwoBlockRowCoeffOperator {
    x_a: DesignMatrix,
    x_b: DesignMatrix,
    c_aa: Arc<Array1<f64>>,
    c_ab: Arc<Array1<f64>>,
    c_bb: Arc<Array1<f64>>,
    dim: usize,
    nrows: usize,
    pa: usize,
}


impl crate::solver::estimate::reml::unified::HyperOperator for DesignTwoBlockRowCoeffOperator {
    fn dim(&self) -> usize {
        self.dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.dim);
        let v_a = v.slice(s![0..self.pa]);
        let v_b = v.slice(s![self.pa..self.dim]);
        let u_a = self.x_a.matrixvectormultiply(&v_a.to_owned());
        let u_b = self.x_b.matrixvectormultiply(&v_b.to_owned());
        assert_eq!(u_a.len(), self.nrows);
        assert_eq!(u_b.len(), self.nrows);
        let r_a = self.c_aa.as_ref() * &u_a + self.c_ab.as_ref() * &u_b;
        let r_b = self.c_ab.as_ref() * &u_a + self.c_bb.as_ref() * &u_b;
        let out_a = self.x_a.transpose_vector_multiply(&r_a);
        let out_b = self.x_b.transpose_vector_multiply(&r_b);
        let mut out = Array1::<f64>::zeros(self.dim);
        out.slice_mut(s![0..self.pa]).assign(&out_a);
        out.slice_mut(s![self.pa..self.dim]).assign(&out_b);
        out
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ndarray::ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        assert!(start + cols <= self.dim);
        let mut basis = Array1::<f64>::zeros(self.dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(local_col).assign(&col);
            basis[global_col] = 0.0;
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        self.mul_basis_columns_into(0, out.view_mut());
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        // For the two-block row-coefficient operator
        //   B v = [X_a^T (c_aa·u_a + c_ab·u_b),  X_b^T (c_ab·u_a + c_bb·u_b)]
        // with u_a = X_a v_a, u_b = X_b v_b, the column-wise quadratic form is
        //   F[:,k]^T B F[:,k] = u_a^T r_a + u_b^T r_b
        //                    = Σ_i (c_aa[i] u_a[i]² + 2 c_ab[i] u_a[i] u_b[i]
        //                            + c_bb[i] u_b[i]²)
        // so the projected trace never needs the X^T r step that the default
        // mul_vec path computes, and the per-row coefficients fold the K
        // columns into a single weighted sum once U_a, U_b are formed.
        let grams = self.projected_row_gram_triples(factor);
        self.trace_from_row_gram_triples(grams.view())
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &crate::solver::estimate::reml::unified::ProjectedFactorCache,
    ) -> f64 {
        // Validate the factor row count up front. Without this, a caller that
        // hands in a factor whose row count does not equal the joint p slips
        // into the per-column `mul_vec` slicing where a `assert_eq!`
        // panics with the generic `left/right` message — that loses the
        // operator identity and the (pa, pb) split which is the only useful
        // diagnostic when the trace caller's own dimension bookkeeping is
        // off. Validate at the operator boundary so the panic localises the
        // caller, and so this contract is enforced in release builds too
        // (the inner `assert_eq!` is a debug-only safety net).
        assert_eq!(
            factor.nrows(),
            self.dim,
            "two-block cached projected trace factor row mismatch: factor rows={} \
             but joint p={} (pa={}, pb={})",
            factor.nrows(),
            self.dim,
            self.pa,
            self.dim - self.pa,
        );
        let key = crate::solver::estimate::reml::unified::ProjectedFactorKey::from_factor_view(
            self.projected_row_gram_cache_id(),
            factor.view(),
        );
        let grams = cache.get_or_insert_with(key, || self.projected_row_gram_triples(factor));
        self.trace_from_row_gram_triples(grams.view())
    }

    fn is_implicit(&self) -> bool {
        true
    }
}


impl DesignTwoBlockRowCoeffOperator {
    fn design_cache_token(design: &DesignMatrix) -> usize {
        match design {
            DesignMatrix::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                Arc::as_ptr(matrix) as usize
            }
            DesignMatrix::Dense(DenseDesignMatrix::Lazy(op)) => {
                Arc::as_ptr(op) as *const () as usize
            }
            DesignMatrix::Sparse(sparse) => sparse as *const _ as usize,
        }
    }

    fn projected_row_gram_cache_id(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        "DesignTwoBlockRowCoeffOperator::projected_row_gram_triples".hash(&mut hasher);
        Self::design_cache_token(&self.x_a).hash(&mut hasher);
        Self::design_cache_token(&self.x_b).hash(&mut hasher);
        self.nrows.hash(&mut hasher);
        self.pa.hash(&mut hasher);
        self.dim.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn projected_row_gram_triples(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            factor.nrows(),
            self.dim,
            "two-block cached projected trace factor row mismatch: factor rows={} \
             but joint p={} (pa={}, pb={})",
            factor.nrows(),
            self.dim,
            self.pa,
            self.dim - self.pa,
        );
        let rank = factor.ncols();
        let mut grams = Array2::<f64>::zeros((self.nrows, 3));
        if self.nrows == 0 || rank == 0 {
            return grams;
        }
        let rows_per_chunk = gamlss_projected_trace_chunk_rows(rank, 2, 3).min(self.nrows.max(1));
        let f_a = factor.slice(s![0..self.pa, ..]);
        let f_b = factor.slice(s![self.pa..self.dim, ..]);
        let fill_chunk = |start: usize, mut out_chunk: ndarray::ArrayViewMut2<'_, f64>| {
            let end = (start + rows_per_chunk).min(self.nrows);
            let rows = start..end;
            let x_a_chunk = self
                .x_a
                .try_row_chunk(rows.clone())
                .expect("two-block projected trace x_a row chunk materialization failed");
            let x_b_chunk = self
                .x_b
                .try_row_chunk(rows.clone())
                .expect("two-block projected trace x_b row chunk materialization failed");
            let u_a = fast_ab(&x_a_chunk, &f_a);
            let u_b = fast_ab(&x_b_chunk, &f_b);
            for local_i in 0..u_a.nrows() {
                let mut aa = 0.0;
                let mut ab = 0.0;
                let mut bb = 0.0;
                for col in 0..rank {
                    let a = u_a[[local_i, col]];
                    let b = u_b[[local_i, col]];
                    aa += a * a;
                    ab += a * b;
                    bb += b * b;
                }
                out_chunk[[local_i, 0]] = aa;
                out_chunk[[local_i, 1]] = ab;
                out_chunk[[local_i, 2]] = bb;
            }
        };
        if rayon::current_thread_index().is_none() && self.nrows > rows_per_chunk {
            grams
                .axis_chunks_iter_mut(Axis(0), rows_per_chunk)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    fill_chunk(chunk_idx * rows_per_chunk, out_chunk)
                });
        } else {
            for start in (0..self.nrows).step_by(rows_per_chunk) {
                let end = (start + rows_per_chunk).min(self.nrows);
                let out_chunk = grams.slice_mut(s![start..end, ..]);
                fill_chunk(start, out_chunk);
            }
        }
        grams
    }

    fn trace_from_row_gram_triples(&self, grams: ArrayView2<'_, f64>) -> f64 {
        assert_eq!(grams.nrows(), self.nrows);
        assert_eq!(grams.ncols(), 3);
        let c_aa = self
            .c_aa
            .as_slice()
            .expect("c_aa is constructed contiguous");
        let c_ab = self
            .c_ab
            .as_slice()
            .expect("c_ab is constructed contiguous");
        let c_bb = self
            .c_bb
            .as_slice()
            .expect("c_bb is constructed contiguous");
        let mut trace = 0.0;
        for i in 0..self.nrows {
            trace +=
                c_aa[i] * grams[[i, 0]] + 2.0 * c_ab[i] * grams[[i, 1]] + c_bb[i] * grams[[i, 2]];
        }
        trace
    }
}


/// Matrix-free joint-Hessian operator for the two-block Gaussian
/// location-scale family. The dense Hessian decomposes as
///
///   H = [[X_mu^T diag(w) X_mu,    X_mu^T diag(cross) X_ls],
///        [X_ls^T diag(cross) X_mu, X_ls^T diag(scale) X_ls]],
///
/// with `cross = 0` and `scale = 2κ²a` — the block-diagonal Gaussian Fisher
/// (expected) information (μ ⊥ σ, #684; residual-free (log σ, log σ) block,
/// #566). This MUST match the dense `exact_newton_joint_hessian_from_designs`
/// curvature object exactly: the observed cross term `2κm` (mean-zero noise)
/// over-smooths the scale and is its Fisher expectation 0. The matvec applies
/// each block by a single design-matrix multiply on each side, so the cost
/// is Θ(n (p_mu + p_ls)) per `Hv` rather than Θ(n (p_mu + p_ls)²) to form
/// the dense matrix.
struct GaussianLocationScaleHessianWorkspace {
    family: GaussianLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    xmu: Arc<Array2<f64>>,
    x_ls: Arc<Array2<f64>>,
    coeff_mm: Array1<f64>,
    coeff_ml: Array1<f64>,
    coeff_ll: Array1<f64>,
}


impl GaussianLocationScaleHessianWorkspace {
    fn new(
        family: GaussianLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let etamu = &block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = family.get_or_compute_row_scalars(etamu, eta_ls)?;
        // Single source of truth shared with the dense
        // `exact_newton_joint_hessian_from_designs`: μ ⊥ σ ⇒ cross = 0 (#684),
        // (ls,ls) = 2κ²a (#566). Reading the same coefficients as the dense path
        // makes the cross-block drift that caused #684 structurally impossible.
        let (coeff_mm, coeff_ml, coeff_ll) = gaussian_locscale_fisher_joint_row_coeffs(&rows);
        Ok(Self {
            family,
            block_states,
            xmu: Arc::new(xmu),
            x_ls: Arc::new(x_ls),
            coeff_mm,
            coeff_ml,
            coeff_ll,
        })
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place.
    ///
    /// Each sampled row's `coeff_*[i]` is multiplied by its
    /// `WeightedOuterRow.weight` (the HT inverse-inclusion factor 1/π_i —
    /// uniform or stratified sampling both supported). All non-sampled rows
    /// are zeroed. Because every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) is row-linear in these arrays
    /// via `Xᵀ diag(W) X`, the resulting joint-Hessian is an unbiased
    /// estimator of the full-data joint Hessian.
    fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.coeff_mm.len();
        let mut mask_mm = Array1::<f64>::zeros(n);
        let mut mask_ml = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            mask_mm[i] = self.coeff_mm[i] * r.weight;
            mask_ml[i] = self.coeff_ml[i] * r.weight;
            mask_ll[i] = self.coeff_ll[i] * r.weight;
        }
        self.coeff_mm = mask_mm;
        self.coeff_ml = mask_ml;
        self.coeff_ll = mask_ll;
    }
}


impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but built once via 3 GEMMs
        // (`Xᵀ diag(W) X` per block) instead of letting
        // `MatrixFreeSpdOperator::materialize_dense_operator` reconstruct the
        // dense Hessian via `total` canonical-basis HVPs. At large scale
        // (n≈320k, p_total≈82) the canonical-basis path takes ~568s per κ-iter
        // while the dense build via fast_xt_diag_x/y is ~1s.
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        let h_mm = xt_diag_x_dense(self.xmu.as_ref(), &self.coeff_mm)?;
        let h_ml = xt_diag_y_dense(self.xmu.as_ref(), &self.coeff_ml, self.x_ls.as_ref())?;
        let h_ll = xt_diag_x_dense(self.x_ls.as_ref(), &self.coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pmu, 0..pmu]).assign(&h_mm);
        h.slice_mut(s![0..pmu, pmu..total]).assign(&h_ml);
        h.slice_mut(s![pmu..total, pmu..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScale matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let u_mu = fast_av(self.xmu.as_ref(), &v.slice(s![0..pmu]));
        let u_ls = fast_av(self.x_ls.as_ref(), &v.slice(s![pmu..total]));
        let r_mu = &self.coeff_mm * &u_mu + &self.coeff_ml * &u_ls;
        let r_ls = &self.coeff_ml * &u_mu + &self.coeff_ll * &u_ls;
        let out_mu = fast_atv(self.xmu.as_ref(), &r_mu);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..total]).assign(&out_ls);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        // Per-column reduction is independent; parallelize across columns.
        let diag_mu: Vec<f64> = (0..pmu)
            .into_par_iter()
            .map(|j| {
                let col = self.xmu.column(j);
                col.iter()
                    .zip(self.coeff_mm.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_ls: Vec<f64> = (0..p_ls)
            .into_par_iter()
            .map(|j| {
                let col = self.x_ls.column(j);
                col.iter()
                    .zip(self.coeff_ll.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let mut diag = Array1::<f64>::zeros(total);
        for (j, v) in diag_mu.into_iter().enumerate() {
            diag[j] = v;
        }
        for (j, v) in diag_ls.into_iter().enumerate() {
            diag[pmu + j] = v;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                &DenseOrOperator::Borrowed(self.xmu.as_ref()),
                &DenseOrOperator::Borrowed(self.x_ls.as_ref()),
                d_beta_flat,
            )
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        let n = self.xmu.nrows();
        let pmu = self.xmu.ncols();
        let pls = self.x_ls.ncols();
        let total = pmu + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "GaussianLocationScale dH operator: d_beta length {} != {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let etamu = &self.block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &self.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = self.family.get_or_compute_row_scalars(etamu, eta_ls)?;
        let ximu = fast_av(self.xmu.as_ref(), &d_beta_flat.slice(s![0..pmu]));
        let xi_ls = fast_av(self.x_ls.as_ref(), &d_beta_flat.slice(s![pmu..total]));
        let directional = gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
        let c_mm = directional.0;
        let c_ll = directional.2;
        // Fisher cross block ≡ 0 (μ ⊥ σ; #684), so its directional derivative is
        // identically 0 — matching the dense
        // `exact_newton_joint_hessian_directional_derivative_from_designs`, which
        // likewise does not assemble `directional.1`.
        let c_ml = Array1::<f64>::zeros(c_mm.len());
        Ok(Some(Arc::new(make_two_block_row_coeff_operator(
            self.xmu.clone(),
            self.x_ls.clone(),
            c_mm,
            c_ml,
            c_ll,
            n,
        ))))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                &DenseOrOperator::Borrowed(self.xmu.as_ref()),
                &DenseOrOperator::Borrowed(self.x_ls.as_ref()),
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        let n = self.xmu.nrows();
        let pmu = self.xmu.ncols();
        let pls = self.x_ls.ncols();
        let total = pmu + pls;
        if d_beta_u.len() != total || d_beta_v.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "GaussianLocationScale d2H operator: d_beta_{{u,v}} length {}/{} != {}",
                    d_beta_u.len(),
                    d_beta_v.len(),
                    total
                ),
            }
            .into());
        }
        let etamu = &self.block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &self.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = self.family.get_or_compute_row_scalars(etamu, eta_ls)?;
        let ximu_u = fast_av(self.xmu.as_ref(), &d_beta_u.slice(s![0..pmu]));
        let xi_ls_u = fast_av(self.x_ls.as_ref(), &d_beta_u.slice(s![pmu..total]));
        let ximu_v = fast_av(self.xmu.as_ref(), &d_beta_v.slice(s![0..pmu]));
        let xi_ls_v = fast_av(self.x_ls.as_ref(), &d_beta_v.slice(s![pmu..total]));
        let directional =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximu_v, &xi_ls_v);
        let c_mm = directional.0;
        let c_ll = directional.2;
        // Fisher cross block ≡ 0 (μ ⊥ σ; #684); its second directional
        // derivative is identically 0 too — match the dense path (which does not
        // assemble `directional.1`).
        let c_ml = Array1::<f64>::zeros(c_mm.len());
        Ok(Some(Arc::new(make_two_block_row_coeff_operator(
            self.xmu.clone(),
            self.x_ls.clone(),
            c_mm,
            c_ml,
            c_ll,
            n,
        ))))
    }
}


/// Build a `RowCoeffOperator` for the standard two-block GAMLSS structure
/// with one design per block and three pair coefficients (a,a), (a,b), (b,b).
/// The resulting matrix mirrors the dense
/// `X_a^T diag(c_aa) X_a + X_a^T diag(c_ab) X_b + X_b^T diag(c_ab) X_a + X_b^T diag(c_bb) X_b`
/// assembly emitted by `gaussian_joint_hessian_from_designs` (Gaussian path)
/// and the `xt_diag_*` block writers (binomial path).
fn make_two_block_row_coeff_operator(
    x_a: Arc<Array2<f64>>,
    x_b: Arc<Array2<f64>>,
    c_aa: Array1<f64>,
    c_ab: Array1<f64>,
    c_bb: Array1<f64>,
    nrows: usize,
) -> RowCoeffOperator {
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    RowCoeffOperator::from_directions(
        vec![pa, pb],
        vec![(0, x_a), (1, x_b)],
        vec![(0, 0, c_aa), (0, 1, c_ab), (1, 1, c_bb)],
        nrows,
    )
}


fn make_two_block_design_row_coeff_operator(
    x_a: DesignMatrix,
    x_b: DesignMatrix,
    c_aa: Arc<Array1<f64>>,
    c_ab: Arc<Array1<f64>>,
    c_bb: Arc<Array1<f64>>,
) -> Result<DesignTwoBlockRowCoeffOperator, String> {
    let nrows = x_a.nrows();
    if x_b.nrows() != nrows || c_aa.len() != nrows || c_ab.len() != nrows || c_bb.len() != nrows {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "two-block row coefficient operator dimension mismatch: rows a={}, b={}, coeffs={}/{}/{}",
            nrows,
            x_b.nrows(),
            c_aa.len(),
            c_ab.len(),
            c_bb.len()
        ) }.into());
    }
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    Ok(DesignTwoBlockRowCoeffOperator {
        x_a,
        x_b,
        c_aa,
        c_ab,
        c_bb,
        dim: pa + pb,
        nrows,
        pa,
    })
}


struct GaussianLocationScaleWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    basis_d3: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
    d4q_dq04: Array1<f64>,
}


/// Per-row pieces of the 3-block Gaussian location-scale-wiggle joint
/// Hessian. Both the dense path and the matrix-free workspace share these
/// row coefficients; only the assembly differs.
struct GaussianLocationScaleWiggleHessianRowPieces {
    coeff_mm: Array1<f64>,
    coeff_ml: Array1<f64>,
    coeff_ll: Array1<f64>,
    coeff_mw_b: Array1<f64>,
    coeff_mw_d: Array1<f64>,
    coeff_lw_b: Array1<f64>,
    coeff_ww: Array1<f64>,
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
}


impl GaussianLocationScaleWiggleHessianRowPieces {
    fn assemble_dense(&self, xmu: &Array2<f64>, x_ls: &Array2<f64>) -> Result<Array2<f64>, String> {
        let h_mm = xt_diag_x_dense(xmu, &self.coeff_mm)?;
        let h_ml = xt_diag_y_dense(xmu, &self.coeff_ml, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_mw = xt_diag_y_dense(xmu, &self.coeff_mw_b, &self.basis)?
            + &xt_diag_y_dense(xmu, &self.coeff_mw_d, &self.basis_d1)?;
        let h_lw = xt_diag_y_dense(x_ls, &self.coeff_lw_b, &self.basis)?;
        let h_ww = xt_diag_x_dense(&self.basis, &self.coeff_ww)?;
        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
    }
}


pub struct GaussianLocationScaleWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
    cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}


impl Clone for GaussianLocationScaleWiggleFamily {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: self.policy.clone(),
            cached_row_scalars: std::sync::RwLock::new(
                self.cached_row_scalars
                    .read()
                    .expect("lock poisoned")
                    .clone(),
            ),
        }
    }
}


impl GaussianLocationScaleWiggleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::Identity,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scalewiggle",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            options.derivative_order,
        )
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d1.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d1.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d1.dot(&beta_link_wiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle second-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d2.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d2.dot(&beta_link_wiggle))
    }

    fn wiggle_d3basis_constrained(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        if d3.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle third-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d3.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d3.dot(&beta_link_wiggle))
    }

    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle fourth-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d4.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    fn wiggle_geometry(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<GaussianLocationScaleWiggleGeometry, String> {
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(GaussianLocationScaleWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        })
    }

    fn get_or_compute_row_scalars(
        &self,
        q: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            q,
            eta_ls,
            &self.weights,
        )?))
    }

    fn dense_block_designs(&self) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        dense_locscale_block_designs_cached(
            self.mu_design.as_ref(),
            self.log_sigma_design.as_ref(),
            "GaussianLocationScaleWiggleFamily",
            "GaussianLocationScaleWiggle",
            "mu",
            &self.policy.material_policy(),
        )
    }
    fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        dense_locscale_block_designs_fromspecs(
            specs,
            3,
            "GaussianLocationScaleWiggleFamily",
            "GaussianLocationScaleWiggle",
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            "mu",
            &self.policy.material_policy(),
        )
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if self.exact_joint_supported() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// The wiggle block (block 2) modulates the inverse link nonlinearly and
    /// does not contribute a linear additive term to any output η; its
    /// Jacobian is an `(2 * n, p_wiggle)` zero matrix.
    ///
    /// - block 0 (mu):        output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma): output 0 = zeros, output 1 = design rows
    /// - block 2 (wiggle):    all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "GaussianLocationScaleWiggleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_MU, Self::BLOCK_LOG_SIGMA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}


/// Row-coefficient bundle for the GLS Wiggle joint second directional
/// derivative, shared by the matrix-free operator and the dense
/// `_from_designs` assemblies. Holds exactly the quantities both consumers
/// read downstream of the (identical) coefficient computation.
struct GlsWiggleSecondDirCoeffs {
    coeff_mm_uv: Array1<f64>,
    coeff_ml_uv: Array1<f64>,
    coeff_ll_uv: Array1<f64>,
    a_u: Array1<f64>,
    a_v: Array1<f64>,
    a_uv: Array1<f64>,
    c_u: Array1<f64>,
    c_v: Array1<f64>,
    c_uv: Array1<f64>,
    l_u: Array1<f64>,
    l_v: Array1<f64>,
    l_uv: Array1<f64>,
    dw_u: Array1<f64>,
    dw_v: Array1<f64>,
    dw_uv: Array1<f64>,
}


/// The two probe directions resolved to row space for the GLS Wiggle joint
/// second directional derivative: `xi`/`zeta` are the X_mu/X_ls contractions,
/// and `q`/`s1`/`g2` are the mixed first/second-derivative wiggle pieces.
struct GlsWiggleDirPieces<'a> {
    zeta_u: &'a Array1<f64>,
    zeta_v: &'a Array1<f64>,
    q_u: &'a Array1<f64>,
    q_v: &'a Array1<f64>,
    q_uv: &'a Array1<f64>,
    s1_u: &'a Array1<f64>,
    s1_v: &'a Array1<f64>,
    s1_uv: &'a Array1<f64>,
    g2_u: &'a Array1<f64>,
    g2_v: &'a Array1<f64>,
    g2_uv: &'a Array1<f64>,
}


/// Compute the shared GLS Wiggle second-directional row coefficients from the
/// per-row scalars, wiggle geometry, and the resolved probe directions.
fn gls_wiggle_second_directional_coeffs(
    rows: &GaussianJointRowScalars,
    geom: &GaussianLocationScaleWiggleGeometry,
    dir: &GlsWiggleDirPieces<'_>,
) -> GlsWiggleSecondDirCoeffs {
    let GlsWiggleDirPieces {
        zeta_u,
        zeta_v,
        q_u,
        q_v,
        q_uv,
        s1_u,
        s1_v,
        s1_uv,
        g2_u,
        g2_v,
        g2_uv,
    } = *dir;
    let szeta_u = &rows.kappa * zeta_u;
    let szeta_v = &rows.kappa * zeta_v;
    let zeta_u_zeta_v = zeta_u * zeta_v;
    let dw_u = -2.0 * &rows.w * &szeta_u;
    let dw_v = -2.0 * &rows.w * &szeta_v;
    let dw_uv =
        4.0 * &rows.w * &(&szeta_u * &szeta_v) - 2.0 * &rows.w * &rows.kappa_prime * &zeta_u_zeta_v;
    let dm_u = -(&rows.w * q_u) - &(2.0 * &rows.m * &szeta_u);
    let dm_v = -(&rows.w * q_v) - &(2.0 * &rows.m * &szeta_v);
    let dm_uv = &(2.0 * &rows.w * &(q_u * &szeta_v + q_v * &szeta_u)) - &(&rows.w * q_uv)
        + &(4.0 * &rows.m * &(&szeta_u * &szeta_v))
        - 2.0 * &rows.m * &rows.kappa_prime * &zeta_u_zeta_v;
    let coeff_mm_uv = &(&dw_uv * &geom.dq_dq0.mapv(|v| v * v))
        + &(2.0 * &dw_u * &geom.dq_dq0 * s1_v)
        + &(2.0 * &dw_v * &geom.dq_dq0 * s1_u)
        + &(2.0 * &rows.w * s1_u * s1_v)
        + &(2.0 * &rows.w * &geom.dq_dq0 * s1_uv)
        - &(&dm_uv * &geom.d2q_dq02)
        - &(&dm_u * g2_v)
        - &(&dm_v * g2_u)
        - &(&rows.m * g2_uv);
    let n = rows.m.len();
    // H_{μ,ls} ≡ Fisher 0 (mean⊥scale orthogonality; the wiggle and μ both
    // enter the mean, log σ is the only scale block), so every β-directional
    // derivative — including this second-order one — is identically 0.
    let coeff_ml_uv = Array1::<f64>::zeros(n);
    // Second directional derivative of the Fisher (log σ, log σ) block
    // coeff_ll = 2κ²a (#566). η_ls is linear in β (no zeta_uv), so the only
    // surviving term is ∂²(2κ²a)/∂η² · zeta_u·zeta_v = 4a(κ'²+κκ'')·zeta_u·zeta_v
    // — matching the dense helper `d_uv` (gaussian_jointsecond_directionalweights).
    let coeff_ll_uv = 4.0
        * &rows.obs_weight
        * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
        * &zeta_u_zeta_v;

    let a_u = &dw_u * &geom.dq_dq0 + &rows.w * s1_u;
    let a_v = &dw_v * &geom.dq_dq0 + &rows.w * s1_v;
    let a_uv = &dw_uv * &geom.dq_dq0 + &dw_u * s1_v + &dw_v * s1_u + &rows.w * s1_uv;
    let c_u = -&dm_u;
    let c_v = -&dm_v;
    let c_uv = -&dm_uv;
    // H_{ls,w} ≡ Fisher 0 (wiggle is mean-side; mean⊥scale), so all of its
    // β-directional derivatives are 0.
    let l_u = Array1::<f64>::zeros(n);
    let l_v = Array1::<f64>::zeros(n);
    let l_uv = Array1::<f64>::zeros(n);

    GlsWiggleSecondDirCoeffs {
        coeff_mm_uv,
        coeff_ml_uv,
        coeff_ll_uv,
        a_u,
        a_v,
        a_uv,
        c_u,
        c_v,
        c_uv,
        l_u,
        l_v,
        l_uv,
        dw_u,
        dw_v,
        dw_uv,
    }
}


impl GaussianLocationScaleWiggleFamily {
    fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &xmu, &x_ls)
    }

    fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            xmu.ncols(),
            x_ls.ncols(),
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            3,
            "GaussianLocationScaleWiggleFamily",
            "mu",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
        }))
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: xmu.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_MU,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "GaussianLocationScaleWiggleFamily",
                primary_label: "mu",
                policy: &self.policy,
            },
        )
    }

    /// Compute the rowwise Hessian pieces shared by the dense path and the
    /// matrix-free workspace operator. The same coefficients reconstruct the
    /// dense p×p matrix or apply `Hv` directly without ever forming it.
    fn wiggle_hessian_row_pieces(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GaussianLocationScaleWiggleHessianRowPieces, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        if geom.basis.ncols() != betaw.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleWiggleFamily wiggle basis/beta mismatch: basis has {} columns but beta has {} entries",
                geom.basis.ncols(),
                betaw.len()
            ) }.into());
        }
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        // Gaussian mean⊥scale Fisher orthogonality. μ (mu) AND the wiggle both
        // enter the MEAN q = q0 + B(q0)·βw (see `let q = q0 + etaw`); log σ is
        // the only scale-side block. The Fisher (expected) cross between any
        // mean-side parameter and log σ is exactly 0: H_{μ,ls} = 2κm·dq_dq0 and
        // H_{ls,w} = 2κm both carry m = r·w = (y−q)·weight/σ², and E[m] =
        // E[r]·w = 0. The dense and matrix-free workspace paths SHARE these row
        // pieces, so setting the cross coeffs to 0 fixes the curvature object
        // (the observed 2κm value) for both. Diagonal/same-side blocks
        // (coeff_mm within mean, coeff_ll within scale, coeff_mw_* within mean,
        // coeff_ww within mean) are untouched.
        let coeff_ml = Array1::<f64>::zeros(n);
        // Fisher/expected (log σ, log σ) information E[H_{ls,ls}] = 2κ²a (#566):
        // the observed 2κ²n + κ'(a−n) collapses at small residuals and
        // over-smooths the scale; E[n]=a gives the residual-free 2κ²a.
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        let coeff_mw_b = &rows.w * &geom.dq_dq0;
        let coeff_mw_d = -&rows.m;
        // ls↔wiggle is a mean⊥scale cross (wiggle is mean-side): Fisher 0.
        let coeff_lw_b = Array1::<f64>::zeros(n);
        let coeff_ww = rows.w.clone();
        Ok(GaussianLocationScaleWiggleHessianRowPieces {
            coeff_mm,
            coeff_ml,
            coeff_ll,
            coeff_mw_b,
            coeff_mw_d,
            coeff_lw_b,
            coeff_ww,
            basis: geom.basis,
            basis_d1: geom.basis_d1,
        })
    }

    fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        Ok(Some(pieces.assemble_dense(xmu, x_ls)?))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily exact joint directional Hessian",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xi = fast_av(xmu, &umu);
        let zeta = fast_av(x_ls, &u_ls);
        // logb κ-scaled η_ls direction; κ' = dκ/dη_ls = κ(1−κ).
        let szeta = &rows.kappa * &zeta;
        let phi = fast_av(&geom.basis, &uw);
        let mut q_u = &geom.dq_dq0 * &xi;
        q_u += &phi;
        let mut s1_u = &geom.d2q_dq02 * &xi;
        s1_u += &fast_av(&geom.basis_d1, &uw);
        let mut g2_u = &geom.d3q_dq03 * &xi;
        g2_u += &fast_av(&geom.basis_d2, &uw);
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let dw_u = -2.0 * &rows.w * &szeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // Static blocks: H_{μ,ls} = Fisher 0 (mean⊥scale); H_{ls,ls} = Fisher
        // 2κ²a (#566). H_{μ,ls} ≡ 0 for all β, so its directional derivative is
        // also identically 0. The Fisher (ls,ls) block 2κ²a depends only on
        // η_ls (a is the constant prior weight), so its directional derivative
        // is 4κκ'a·zeta.
        let coeff_ml_u = Array1::<f64>::zeros(n);
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &(&zeta * &rows.obs_weight);
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let c_u = -&dm_u;
        // ls↔wiggle cross block: Fisher 0 (wiggle is mean-side), so its
        // directional derivative is 0 as well.
        let l_u = Array1::<f64>::zeros(n);
        let zeros_ls_b1 = Array1::<f64>::zeros(n);

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_u)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_u, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_u)?;
        let h_mw = xt_diag_y_dense(xmu, &a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_u)?
            + &xt_diag_y_dense(xmu, &c_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_u)?;
        let h_lw = xt_diag_y_dense(x_ls, &l_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &zeros_ls_b1, &basis_u)?;
        let a_ww = xt_diag_y_dense(&basis_u, &rows.w, &geom.basis)?;
        let h_ww = &a_ww + &a_ww.t() + &xt_diag_x_dense(&geom.basis, &dw_u)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    /// Build a matrix-free `RowCoeffOperator` for the GLS Wiggle joint
    /// directional derivative `D_β H_L[u]`. Output dimension is
    /// `pmu + p_ls + pw`. Channels (in order): X_mu, X_ls, B, B', B''.
    fn gls_wiggle_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        xmu_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu_arc.ncols();
        let p_ls = x_ls_arc.ncols();
        let q0_eta = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) =
            layout.split_three(d_beta_flat, "GLS Wiggle joint dH operator d_beta")?;
        if q0_eta.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0_eta + etaw;
        let geom = self.wiggle_geometry(q0_eta.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xi = fast_av(xmu_arc.as_ref(), &umu);
        let zeta = fast_av(x_ls_arc.as_ref(), &u_ls);
        let szeta = &rows.kappa * &zeta;
        let phi = fast_av(&geom.basis, &uw);
        let mut q_u = &geom.dq_dq0 * &xi;
        q_u += &phi;
        let mut s1_u = &geom.d2q_dq02 * &xi;
        s1_u += &fast_av(&geom.basis_d1, &uw);
        let mut g2_u = &geom.d3q_dq03 * &xi;
        g2_u += &fast_av(&geom.basis_d2, &uw);
        let dw_u = -2.0 * &rows.w * &szeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // H_{μ,ls} ≡ Fisher 0 (mean⊥scale); its directional derivative is 0.
        let coeff_ml_u = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) 2κ²a directional derivative: 4κκ'a·zeta (#566).
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &(&zeta * &rows.obs_weight);
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let c_u = -&dm_u;
        // H_{ls,w} ≡ Fisher 0 (wiggle is mean-side); its derivative is 0 in
        // both the B channel (l_u) and the B' channel (coeff_ls_b1).
        let l_u = Array1::<f64>::zeros(n);

        // Pair-coefficient bundles. For (0=X_mu, 3=B'): combine
        // `xt_diag_y_dense(xmu, &(w·dq_dq0), &basis_u=diag(xi)·B')`
        // (giving coeff `w·dq_dq0·xi`) with `xt_diag_y_dense(xmu, &c_u, &B')`
        // (coeff `c_u`).
        let coeff_m_b1 = &(&rows.w * &geom.dq_dq0 * &xi) + &c_u;
        // (0=X_mu, 4=B''): from `xt_diag_y_dense(xmu, &(-m), &basis1_u=diag(xi)·B'')`.
        let coeff_m_b2 = -(&rows.m * &xi);
        // (1=X_ls, 3=B'): ls↔wiggle Fisher-0 cross → zero.
        let coeff_ls_b1 = Array1::<f64>::zeros(n);
        // (2=B, 3=B'): a_ww + a_ww^T where a_ww = (diag(xi)·B')^T diag(w) B
        // = B'^T diag(w·xi) B. The symmetric pair contribution in
        // `RowCoeffOperator` reproduces a_ww + a_ww^T with c = w·xi.
        let coeff_b_b1 = &rows.w * &xi;

        let basis: Arc<Array2<f64>> = Arc::new(geom.basis.clone());
        let basis_d1: Arc<Array2<f64>> = Arc::new(geom.basis_d1.clone());
        let basis_d2: Arc<Array2<f64>> = Arc::new(geom.basis_d2.clone());
        let pw = basis.ncols();

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pmu, p_ls, pw],
            vec![
                (0, xmu_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
            ],
            vec![
                // (X_mu, X_mu) ← `xt_diag_x_dense(xmu, &coeff_mm_u)`
                (0, 0, coeff_mm_u),
                // (X_mu, X_ls) ← `xt_diag_y_dense(xmu, &coeff_ml_u, x_ls)`
                (0, 1, coeff_ml_u),
                // (X_ls, X_ls) ← `xt_diag_x_dense(x_ls, &coeff_ll_u)`
                (1, 1, coeff_ll_u),
                // (X_mu, B) ← `xt_diag_y_dense(xmu, &a_u, &geom.basis)`
                (0, 2, a_u),
                // (X_mu, B') ← `xt_diag_y_dense(xmu, w·dq_dq0, basis_u=diag(ξ)·B') + xt_diag_y_dense(xmu, c_u, B')`
                (0, 3, coeff_m_b1),
                // (X_mu, B'') ← `xt_diag_y_dense(xmu, -m, basis1_u=diag(ξ)·B'')`
                (0, 4, coeff_m_b2),
                // (X_ls, B) ← `xt_diag_y_dense(x_ls, &l_u, &geom.basis)`
                (1, 2, l_u),
                // (X_ls, B') ← ls↔wiggle is mean⊥scale Fisher 0, so coeff_ls_b1 = 0
                (1, 3, coeff_ls_b1),
                // (B, B) ← `xt_diag_x_dense(&geom.basis, &dw_u)`
                (2, 2, dw_u),
                // (B, B') ← a_ww + a_ww^T = B^T diag(w·ξ) B' + B'^T diag(w·ξ) B
                (2, 3, coeff_b_b1),
            ],
            n,
        ))))
    }

    /// Build a matrix-free `RowCoeffOperator` for the GLS Wiggle joint
    /// second directional derivative `D²_β H_L[u, v]`. Channels: X_mu,
    /// X_ls, B, B', B'', B'''. Pair list mirrors the 8-term `xt_diag_*`
    /// assembly in `_from_designs`, with row-coefficient bundles that
    /// absorb the `ξ_u, ξ_v, ξ_u·ξ_v` row factors arising from
    /// `basis_u = diag(ξ_u)·B'`, `basis_uv = diag(ξ_u·ξ_v)·B''`, etc.
    fn gls_wiggle_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        xmu_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu_arc.ncols();
        let p_ls = x_ls_arc.ncols();
        let q0_eta = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(d_beta_u, "GLS Wiggle d2H operator (u)")?;
        let (vmu, v_ls, vw) = layout.split_three(d_beta_v, "GLS Wiggle d2H operator (v)")?;
        if q0_eta.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0_eta + etaw;
        let geom = self.wiggle_geometry(q0_eta.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi_u = fast_av(xmu_arc.as_ref(), &umu);
        let xi_v = fast_av(xmu_arc.as_ref(), &vmu);
        let zeta_u = fast_av(x_ls_arc.as_ref(), &u_ls);
        let zeta_v = fast_av(x_ls_arc.as_ref(), &v_ls);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut q_u = &geom.dq_dq0 * &xi_u;
        q_u += &phi_u;
        let mut q_v = &geom.dq_dq0 * &xi_v;
        q_v += &phi_v;
        let mut s1_u = &geom.d2q_dq02 * &xi_u;
        s1_u += &b1u;
        let mut s1_v = &geom.d2q_dq02 * &xi_v;
        s1_v += &b1v;
        let mut g2_u = &geom.d3q_dq03 * &xi_u;
        g2_u += &b2u;
        let mut g2_v = &geom.d3q_dq03 * &xi_v;
        g2_v += &b2v;
        let q_uv = &(&geom.d2q_dq02 * &(&xi_u * &xi_v)) + &(&b1u * &xi_v) + &(&b1v * &xi_u);
        let s1_uv = &(&geom.d3q_dq03 * &(&xi_u * &xi_v)) + &(&b2u * &xi_v) + &(&b2v * &xi_u);
        let g2_uv = &(&geom.d4q_dq04 * &(&xi_u * &xi_v)) + &(&b3u * &xi_v) + &(&b3v * &xi_u);

        let GlsWiggleSecondDirCoeffs {
            coeff_mm_uv,
            coeff_ml_uv,
            coeff_ll_uv,
            a_u,
            a_v,
            a_uv,
            c_u,
            c_v,
            c_uv,
            l_u,
            l_v,
            l_uv,
            dw_u,
            dw_v,
            dw_uv,
        } = gls_wiggle_second_directional_coeffs(
            &rows,
            &geom,
            &GlsWiggleDirPieces {
                zeta_u: &zeta_u,
                zeta_v: &zeta_v,
                q_u: &q_u,
                q_v: &q_v,
                q_uv: &q_uv,
                s1_u: &s1_u,
                s1_v: &s1_v,
                s1_uv: &s1_uv,
                g2_u: &g2_u,
                g2_v: &g2_v,
                g2_uv: &g2_uv,
            },
        );

        // Pair-coefficient bundles. Cross-block (mu, B'/B'') absorb basis_u/v/uv row scaling.
        let xi_u_xi_v = &xi_u * &xi_v;
        let coeff_m_b1 = &(&a_u * &xi_v) + &(&a_v * &xi_u) + &c_uv;
        let coeff_m_b2 = &(&rows.w * &geom.dq_dq0 * &xi_u_xi_v) + &(&c_u * &xi_v) + &(&c_v * &xi_u);
        let coeff_m_b3 = -(&rows.m * &xi_u_xi_v);
        // ls↔wiggle is Fisher-0 (mean⊥scale): the B' (coeff_ls_b1) and B''
        // (coeff_ls_b2) channels of its second directional derivative vanish.
        let coeff_ls_b1 = &(&l_u * &xi_v) + &(&l_v * &xi_u);
        let coeff_ls_b2 = Array1::<f64>::zeros(n);
        // Wiggle-wiggle from a_ab + a_ab^T + a_ij + a_ij^T + a_iwj + a_iwj^T + a_jwi + a_jwi^T:
        //   a_ab = B''^T diag(w·ξ_uξ_v) B    → pair (B, B'', w·ξ_uξ_v)
        //   a_ij = B'^T diag(w·ξ_uξ_v) B'   → pair (B', B', 2·w·ξ_uξ_v)  (a_ij + a_ij^T)
        //   a_iwj+a_jwi = B'^T diag(dw_v·ξ_u + dw_u·ξ_v) B → pair (B, B', sum)
        let coeff_b_b1 = &(&dw_u * &xi_v) + &(&dw_v * &xi_u);
        let coeff_b_b2 = &rows.w * &xi_u_xi_v;
        let coeff_b1_b1 = 2.0 * &(&rows.w * &xi_u_xi_v);

        let basis: Arc<Array2<f64>> = Arc::new(geom.basis.clone());
        let basis_d1: Arc<Array2<f64>> = Arc::new(geom.basis_d1.clone());
        let basis_d2: Arc<Array2<f64>> = Arc::new(geom.basis_d2.clone());
        let basis_d3: Arc<Array2<f64>> = Arc::new(geom.basis_d3.clone());
        let pw = basis.ncols();

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pmu, p_ls, pw],
            vec![
                (0, xmu_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
                (2, basis_d3),
            ],
            vec![
                // (X_mu, X_mu) ← `xt_diag_x_dense(xmu, &coeff_mm_uv)`
                (0, 0, coeff_mm_uv),
                // (X_mu, X_ls) ← `xt_diag_y_dense(xmu, &coeff_ml_uv, x_ls)`
                (0, 1, coeff_ml_uv),
                // (X_ls, X_ls) ← `xt_diag_x_dense(x_ls, &coeff_ll_uv)`
                (1, 1, coeff_ll_uv),
                // (X_mu, B) ← `xt_diag_y_dense(xmu, &a_uv, &geom.basis)`
                (0, 2, a_uv),
                // (X_mu, B') ← combined `a_u·ξ_v + a_v·ξ_u + c_uv` from
                // `xt_diag_y_dense(xmu, a_u, basis_v) + xt_diag_y_dense(xmu,
                // a_v, basis_u) + xt_diag_y_dense(xmu, c_uv, B')`
                (0, 3, coeff_m_b1),
                // (X_mu, B'') ← `xt_diag_y_dense(xmu, w·dq_dq0, basis_uv) +
                // xt_diag_y_dense(xmu, c_u, basis1_v) + xt_diag_y_dense(xmu,
                // c_v, basis1_u)` (basis_uv = diag(ξ_uξ_v)·B'';
                // basis1_{u,v} = diag(ξ_{u,v})·B'')
                (0, 4, coeff_m_b2),
                // (X_mu, B''') ← `xt_diag_y_dense(xmu, -m, basis1_uv)`
                // with basis1_uv = diag(ξ_uξ_v)·B'''
                (0, 5, coeff_m_b3),
                // (X_ls, B) ← `xt_diag_y_dense(x_ls, &l_uv, &geom.basis)`
                (1, 2, l_uv),
                // (X_ls, B') ← combined from `xt_diag_y_dense(x_ls, l_u,
                // basis_v) + xt_diag_y_dense(x_ls, l_v, basis_u)` =
                // `l_u·ξ_v + l_v·ξ_u`
                (1, 3, coeff_ls_b1),
                // (X_ls, B'') ← ls↔wiggle is mean⊥scale Fisher 0, so coeff_ls_b2 = 0
                (1, 4, coeff_ls_b2),
                // (B, B) ← `xt_diag_x_dense(&geom.basis, &dw_uv)`
                (2, 2, dw_uv),
                // (B, B') ← combined `a_iwj + a_iwj^T + a_jwi + a_jwi^T` =
                // B^T diag(dw_u·ξ_v + dw_v·ξ_u) B' + B'^T diag(...) B
                (2, 3, coeff_b_b1),
                // (B, B'') ← `a_ab + a_ab^T` with a_ab = B''^T diag(w·ξ_uξ_v) B
                (2, 4, coeff_b_b2),
                // (B', B') ← `a_ij + a_ij^T = 2·B'^T diag(w·ξ_uξ_v) B'`;
                // diagonal pair coeff doubles to absorb the factor of 2
                (3, 3, coeff_b1_b1),
            ],
            n,
        ))))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_u_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (u)",
        )?;
        let (vmu, v_ls, vw) = layout.split_three(
            d_beta_v_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (v)",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi_u = fast_av(xmu, &umu);
        let xi_v = fast_av(xmu, &vmu);
        let zeta_u = fast_av(x_ls, &u_ls);
        let zeta_v = fast_av(x_ls, &v_ls);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut q_u = &geom.dq_dq0 * &xi_u;
        q_u += &phi_u;
        let mut q_v = &geom.dq_dq0 * &xi_v;
        q_v += &phi_v;
        let mut s1_u = &geom.d2q_dq02 * &xi_u;
        s1_u += &b1u;
        let mut s1_v = &geom.d2q_dq02 * &xi_v;
        s1_v += &b1v;
        let mut g2_u = &geom.d3q_dq03 * &xi_u;
        g2_u += &b2u;
        let mut g2_v = &geom.d3q_dq03 * &xi_v;
        g2_v += &b2v;
        let q_uv = &(&geom.d2q_dq02 * &(&xi_u * &xi_v)) + &(&b1u * &xi_v) + &(&b1v * &xi_u);
        let s1_uv = &(&geom.d3q_dq03 * &(&xi_u * &xi_v)) + &(&b2u * &xi_v) + &(&b2v * &xi_u);
        let g2_uv = &(&geom.d4q_dq04 * &(&xi_u * &xi_v)) + &(&b3u * &xi_v) + &(&b3v * &xi_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?;
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?;
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi_u)?;
        let basis1_v = scale_matrix_rows(&geom.basis_d2, &xi_v)?;
        let basis1_uv = scale_matrix_rows(&geom.basis_d3, &(&xi_u * &xi_v))?;

        // Shared κ-aware second-directional row coefficients (κ' = κ(1−κ),
        // κ'' = κ(1−κ)(1−2κ), κ''' = κ''(1−2κ) − 2(κ')²): identical to the
        // matrix-free operator path, factored into one helper.
        let GlsWiggleSecondDirCoeffs {
            coeff_mm_uv,
            coeff_ml_uv,
            coeff_ll_uv,
            a_u,
            a_v,
            a_uv,
            c_u,
            c_v,
            c_uv,
            l_u,
            l_v,
            l_uv,
            dw_u,
            dw_v,
            dw_uv,
        } = gls_wiggle_second_directional_coeffs(
            &rows,
            &geom,
            &GlsWiggleDirPieces {
                zeta_u: &zeta_u,
                zeta_v: &zeta_v,
                q_u: &q_u,
                q_v: &q_v,
                q_uv: &q_uv,
                s1_u: &s1_u,
                s1_v: &s1_v,
                s1_uv: &s1_uv,
                g2_u: &g2_u,
                g2_v: &g2_v,
                g2_uv: &g2_uv,
            },
        );

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_uv)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_uv, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_uv)?;
        let h_mw = xt_diag_y_dense(xmu, &a_uv, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_v)?
            + &xt_diag_y_dense(xmu, &a_v, &basis_u)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_uv)?
            + &xt_diag_y_dense(xmu, &c_uv, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_v)?
            + &xt_diag_y_dense(xmu, &c_v, &basis1_u)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_uv)?;
        // H_{ls,w} ≡ Fisher 0 (mean⊥scale): l_uv/l_u/l_v are 0 (shared helper)
        // and the 2κm·B'' channel vanishes too.
        let zeros_ls_b2 = Array1::<f64>::zeros(n);
        let h_lw = xt_diag_y_dense(x_ls, &l_uv, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_v)?
            + &xt_diag_y_dense(x_ls, &l_v, &basis_u)?
            + &xt_diag_y_dense(x_ls, &zeros_ls_b2, &basis_uv)?;
        let a_ab = xt_diag_y_dense(&basis_uv, &rows.w, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &rows.w, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + a_ij.t()
            + &a_iwj
            + a_iwj.t()
            + &a_jwi
            + a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        // logb κ-chain on η_ls; e_a = ∂η_ls/∂ψ_a row-direction.
        let e_a = &dir_a.z_ls_psi;
        let amn = &rows.obs_weight - &rows.n;
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;

        let objective_psi = (-&rows.m * &q_a + &s_ls * e_a).sum();
        let score_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_map.transpose_mul(s_mu.view()) + fast_atv(xmu, &s_mu_a)),
            &(x_ls_map.transpose_mul(s_ls.view()) + fast_atv(x_ls, &s_ls_a)),
            &(fast_atv(&basis_a, &s_w) + fast_atv(&geom.basis, &s_w_a)),
        );

        // Static blocks under logb. Gaussian mean⊥scale Fisher orthogonality:
        // μ AND the wiggle both enter the MEAN q = q0 + B(q0)·βw, so log σ is
        // the only scale-side block. The Fisher (expected) cross between any
        // mean-side parameter and log σ is exactly 0 because it carries
        // m = r·weight/σ² and E[m] = E[r]·weight/σ² = 0:
        //   coeff_ml = E[H_{μ,ls}] = 0  (observed 2κmD)
        //   l        = E[H_{ls,w}] = 0  (observed 2κm)
        // A function identically 0 has 0 ψ-derivatives, so coeff_ml_a and l_a
        // vanish too. This mirrors the non-wiggle psi path
        // (gaussian_joint_psi_firstweights: hmu_ls = dhmu_ls = 0) and the
        // wiggle Newton/REML Hessian path (wiggle_hessian_row_pieces:
        // coeff_ml = coeff_lw_b = 0). The observed SCORE (s_mu/s_ls/s_w above)
        // stays exact so Fisher scoring still hits the joint MLE; only the
        // curvature feeding the REML determinant / IFT correction is the
        // (orthogonal) expectation. coeff_ll is the residual-free Fisher
        // 2κ²a (#566); its ψ-derivative coeff_ll_a = 4κκ'a·e_a depends only on
        // η_ls. Same-side blocks (coeff_mm within mean, a/c the μ↔wiggle
        // within-mean cross, coeff_ww within mean) are untouched.
        let n = rows.m.len();
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_ml = Array1::<f64>::zeros(n);
        let coeff_ml_a = Array1::<f64>::zeros(n);
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        let coeff_ll_a = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_a;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let l = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let h_mm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &h_mm_a1 + &h_mm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a, x_ls)?;
        let h_ll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &h_ll_a1 + &h_ll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_a, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a, &basis_a)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_a, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_a, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a)?;
        let h_ww_a1 = xt_diag_y_dense(&basis_a, &rows.w, &geom.basis)?;
        let h_ww = &h_ww_a1 + &h_ww_a1.t() + &xt_diag_x_dense(&geom.basis, &dw_a)?;

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_b) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_a,
                &dir_b,
                xmu,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_a: &LocationScaleJointPsiDirection,
        dir_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_a,
            dir_b,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_a_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_a_map = dir_a.x_ls_psi.as_linear_map_ref();
        let xmu_b_map = dir_b.x_primary_psi.as_linear_map_ref();
        let x_ls_b_map = dir_b.x_ls_psi.as_linear_map_ref();
        let xmu_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            xmu.ncols(),
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            x_ls.ncols(),
        );
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let q_b = &geom.dq_dq0 * &dir_b.z_primary_psi;
        let q_ab = &(&geom.dq_dq0 * &second_drifts.z_primary_ab)
            + &(&geom.d2q_dq02 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi));
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let s1_b = &geom.d2q_dq02 * &dir_b.z_primary_psi;
        let s1_ab = &(&geom.d3q_dq03 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi))
            + &(&geom.d2q_dq02 * &second_drifts.z_primary_ab);
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let g2_b = &geom.d3q_dq03 * &dir_b.z_primary_psi;
        let g2_ab = &(&geom.d4q_dq04 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi))
            + &(&geom.d3q_dq03 * &second_drifts.z_primary_ab);
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis_b = scale_matrix_rows(&geom.basis_d1, &dir_b.z_primary_psi)?;
        let basis_ab = scale_matrix_rows(&geom.basis_d1, &second_drifts.z_primary_ab)?
            + &scale_matrix_rows(
                &geom.basis_d2,
                &(&dir_a.z_primary_psi * &dir_b.z_primary_psi),
            )?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        let basis1_b = scale_matrix_rows(&geom.basis_d2, &dir_b.z_primary_psi)?;
        let basis1_ab = scale_matrix_rows(&geom.basis_d2, &second_drifts.z_primary_ab)?
            + &scale_matrix_rows(
                &geom.basis_d3,
                &(&dir_a.z_primary_psi * &dir_b.z_primary_psi),
            )?;

        // logb κ-chain on η_ls; κ' = κ(1−κ), κ'' = κ(1−κ)(1−2κ),
        // κ''' = κ''(1−2κ) − 2(κ')².
        let e_a = &dir_a.z_ls_psi;
        let e_b = &dir_b.z_ls_psi;
        let e_ab = &second_drifts.z_ls_ab;
        let amn = &rows.obs_weight - &rows.n;
        // 4κ² − 2κ' (∂²w/∂η² style coefficient when both directions hit η_ls).
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;

        // Row drifts under logb. The η_ls direction picks up a κ on each step,
        // and η_ls·η_ls picks up (4κ²−2κ') from differentiating κ on the
        // second leg. The η_ab (z_ls_ab) leg uses just one κ from the chain.
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dw_b = -2.0 * &rows.w * &rows.kappa * e_b;
        let dw_ab =
            &four_k2_minus_2kpi * &rows.w * &(e_a * e_b) - &(2.0 * &rows.w * &rows.kappa * e_ab);
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dm_b = -(&rows.w * &q_b) - &(2.0 * &rows.m * &rows.kappa * e_b);
        let dm_ab = &(2.0 * &rows.w * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(&rows.w * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * e_b))
            - &(2.0 * &rows.m * &rows.kappa * e_ab);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let dn_b = -(2.0 * &rows.m * &q_b) - &(2.0 * &rows.n * &rows.kappa * e_b);
        let dn_ab = &(2.0 * &rows.w * &(&q_a * &q_b))
            + &(4.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(2.0 * &rows.m * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.n * &(e_a * e_b))
            - &(2.0 * &rows.n * &rows.kappa * e_ab);

        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_mu_b = -(&dm_b * &geom.dq_dq0) - &(&rows.m * &s1_b);
        let s_mu_ab =
            -(&dm_ab * &geom.dq_dq0) - &(&dm_a * &s1_b) - &(&dm_b * &s1_a) - &(&rows.m * &s1_ab);
        // score_ls = κ(a−n); ψ derivatives carry κ' / κ'' from chain on κ.
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_ls_b = &rows.kappa_prime * &(e_b * &amn) - &rows.kappa * &dn_b;
        // s_ls_ab = κ''·e_a·e_b·(a−n) + κ'·e_ab·(a−n)
        //         − κ'·(e_a·n_b + e_b·n_a) − κ·n_ab
        let s_ls_ab = &rows.kappa_dprime * &(e_a * e_b) * &amn + &rows.kappa_prime * e_ab * &amn
            - &rows.kappa_prime * &(e_a * &dn_b + e_b * &dn_a)
            - &rows.kappa * &dn_ab;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;
        let s_w_b = -&dm_b;
        let s_w_ab = -&dm_ab;

        let objective_psi_psi = (&rows.w * &(&q_a * &q_b)
            + &(2.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            + &((2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn)
                * &(e_a * e_b))
            - &(&rows.m * &q_ab)
            + &(&rows.kappa * &amn * e_ab))
            .sum();

        let score_psi_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_ab_map.transpose_mul(s_mu.view())
                + xmu_a_map.transpose_mul(s_mu_b.view())
                + xmu_b_map.transpose_mul(s_mu_a.view())
                + fast_atv(xmu, &s_mu_ab)),
            &(x_ls_ab_map.transpose_mul(s_ls.view())
                + x_ls_a_map.transpose_mul(s_ls_b.view())
                + x_ls_b_map.transpose_mul(s_ls_a.view())
                + fast_atv(x_ls, &s_ls_ab)),
            &(fast_atv(&basis_ab, &s_w)
                + fast_atv(&basis_a, &s_w_b)
                + fast_atv(&basis_b, &s_w_a)
                + fast_atv(&geom.basis, &s_w_ab)),
        );

        // Static blocks under logb. coeff_mm has no κ; coeff_ll = Fisher 2κ²a
        // (#566). Gaussian mean⊥scale Fisher orthogonality: the wiggle and μ
        // both enter the mean (q = q0 + B·βw), log σ is the only scale block,
        // so coeff_ml = E[H_{μ,ls}] = 0 and l = E[H_{ls,w}] = 0 (observed 2κm,
        // E[m]=0). All of their ψ-directional derivatives (a/b/ab) are 0 since
        // a function identically 0 has 0 derivatives. The Fisher (ls,ls) block
        // depends only on η_ls so its derivatives carry only κ.
        let n = rows.m.len();
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_ml = Array1::<f64>::zeros(n);
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        // coeff_mm_a/b/ab: structurally κ-free; correctness now follows from
        // dw_a/_b/_ab and dm_a/_b/_ab carrying the κ chain on η_ls (above).
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_mm_b = &(&dw_b * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_b)
            - &(&dm_b * &geom.d2q_dq02)
            - &(&rows.m * &g2_b);
        let coeff_mm_ab = &(&dw_ab * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_b)
            + &(2.0 * &dw_b * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_a * &s1_b)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_ab)
            - &(&dm_ab * &geom.d2q_dq02)
            - &(&dm_a * &g2_b)
            - &(&dm_b * &g2_a)
            - &(&rows.m * &g2_ab);
        // coeff_ml (μ↔logσ) is Fisher 0; its 1st/2nd ψ-directional derivatives
        // are 0 as well.
        let coeff_ml_a = Array1::<f64>::zeros(n);
        let coeff_ml_b = Array1::<f64>::zeros(n);
        let coeff_ml_ab = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) coeff_ll = 2κ²a (a constant prior weight) depends only
        // on η_ls (#566): ∂(2κ²a)/∂η = 4κκ'a, so the ψ-first derivatives are
        // 4κκ'a·e_a / e_b. The η_ab leg carries one κ on top.
        let coeff_ll_a = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_a;
        let coeff_ll_b = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_b;
        // coeff_ll_ab = ∂²(2κ²a)/∂a∂b = 4a(κ'²+κκ'')·e_a·e_b + 4κκ'a·e_ab
        // (mirrors the dense helper `d2h_ls_ls`).
        let coeff_ll_ab = 4.0
            * &rows.obs_weight
            * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
            * &(e_a * e_b)
            + 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_ab;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_b = &dw_b * &geom.dq_dq0 + &rows.w * &s1_b;
        let a_ab = &dw_ab * &geom.dq_dq0 + &dw_a * &s1_b + &dw_b * &s1_a + &rows.w * &s1_ab;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let c_b = -&dm_b;
        let c_ab = -&dm_ab;
        // l (logσ↔wiggle) is Fisher 0 (wiggle is mean-side; mean⊥scale), so all
        // of its 1st/2nd ψ-directional derivatives vanish.
        let l = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let l_b = Array1::<f64>::zeros(n);
        let l_ab = Array1::<f64>::zeros(n);

        let hmm_ab = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_ij = weighted_crossprod_psi_maps(xmu_a_map, coeff_mm.view(), xmu_b_map)?;
        let hmm_iwj = weighted_crossprod_psi_maps(
            xmu_a_map,
            coeff_mm_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_jwi = weighted_crossprod_psi_maps(
            xmu_b_map,
            coeff_mm_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_ab
            + &hmm_ab.t()
            + &hmm_ij
            + hmm_ij.t()
            + &hmm_iwj
            + hmm_iwj.t()
            + &hmm_jwi
            + hmm_jwi.t()
            + &xt_diag_x_dense(xmu, &coeff_mm_ab)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(xmu_a_map, coeff_ml.view(), x_ls_b_map)?
            + &weighted_crossprod_psi_maps(xmu_b_map, coeff_ml.view(), x_ls_a_map)?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                coeff_ml_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                coeff_ml_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_a.view(),
                x_ls_b_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_b.view(),
                x_ls_a_map,
            )?
            + &xt_diag_y_dense(xmu, &coeff_ml_ab, x_ls)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml.view(),
                x_ls_ab_map,
            )?;
        let hll_ab = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_ij = weighted_crossprod_psi_maps(x_ls_a_map, coeff_ll.view(), x_ls_b_map)?;
        let hll_iwj = weighted_crossprod_psi_maps(
            x_ls_a_map,
            coeff_ll_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_jwi = weighted_crossprod_psi_maps(
            x_ls_b_map,
            coeff_ll_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_ab
            + &hll_ab.t()
            + &hll_ij
            + hll_ij.t()
            + &hll_iwj
            + hll_iwj.t()
            + &hll_jwi
            + hll_jwi.t()
            + &xt_diag_x_dense(x_ls, &coeff_ll_ab)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_ab_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            xmu_b_map,
            a_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_ab, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(xmu, &a_b, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_ab)?
            + &weighted_crossprod_psi_maps(
                xmu_ab_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_b),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_ab, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_a),
            )?
            + &xt_diag_y_dense(xmu, &c_b, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_ab)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            x_ls_b_map,
            l_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_ab, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                x_ls_b_map,
                l.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(x_ls, &l_b, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_ab)?;
        let hww_ab = xt_diag_y_dense(&basis_ab, &rows.w, &geom.basis)?;
        let hww_ij = xt_diag_y_dense(&basis_a, &rows.w, &basis_b)?;
        let hww_iwj = xt_diag_y_dense(&basis_a, &dw_b, &geom.basis)?;
        let hww_jwi = xt_diag_y_dense(&basis_b, &dw_a, &geom.basis)?;
        let h_ww = &hww_ab
            + &hww_ab.t()
            + &hww_ij
            + hww_ij.t()
            + &hww_iwj
            + hww_iwj.t()
            + &hww_jwi
            + hww_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_ab)?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                xmu,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily joint psi hessian directional derivative",
        )?;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi = fast_av(xmu, &umu);
        let zeta = fast_av(x_ls, &u_ls);
        let zmu_a_u = xmu_map.forward_mul(umu.view());
        let zls_a_u = x_ls_map.forward_mul(u_ls.view());
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b3u = fast_av(&geom.basis_d3, &uw);

        let q_u = &(&geom.dq_dq0 * &xi) + &fast_av(&geom.basis, &uw);
        let s1_u = &(&geom.d2q_dq02 * &xi) + &b1u;
        let g2_u = &(&geom.d3q_dq03 * &xi) + &b2u;
        let g3_u = &(&geom.d4q_dq04 * &xi) + &b3u;

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let q_a_u = &(&s1_u * &dir_a.z_primary_psi) + &(&geom.dq_dq0 * &zmu_a_u);
        let s1_a_u = &(&g2_u * &dir_a.z_primary_psi) + &(&geom.d2q_dq02 * &zmu_a_u);
        let g2_a_u = &(&g3_u * &dir_a.z_primary_psi) + &(&geom.d3q_dq03 * &zmu_a_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        let basis_a_u = scale_matrix_rows(&geom.basis_d2, &(&xi * &dir_a.z_primary_psi))?
            + &scale_matrix_rows(&geom.basis_d1, &zmu_a_u)?;
        let basis1_a_u = scale_matrix_rows(&geom.basis_d3, &(&xi * &dir_a.z_primary_psi))?
            + &scale_matrix_rows(&geom.basis_d2, &zmu_a_u)?;

        // logb κ-chain on η_ls; e_a = ψ_a's η_ls direction, ζ = β-direction.
        // η_au = zls_a_u is the second mixed derivative (β·ψ).
        let e_a = &dir_a.z_ls_psi;
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;
        let dw_u = -2.0 * &rows.w * &rows.kappa * &zeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &rows.kappa * &zeta);
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dw_a_u = &four_k2_minus_2kpi * &rows.w * &(e_a * &zeta)
            - &(2.0 * &rows.w * &rows.kappa * &zls_a_u);
        let dm_a_u = &(2.0 * &rows.w * &rows.kappa * &(&q_a * &zeta + &q_u * e_a))
            - &(&rows.w * &q_a_u)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * &zeta))
            - &(2.0 * &rows.m * &rows.kappa * &zls_a_u);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // coeff_ml (μ↔logσ) is mean⊥scale Fisher 0 (E[m]=0), so both its
        // β-drift derivative coeff_ml_u and the mixed coeff_ml_a_u are 0.
        let n = rows.m.len();
        let coeff_ml_u = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) coeff_ll = 2κ²a (#566); ∂(2κ²a)/∂η = 4κκ'a, so the
        // β-drift derivative along ζ is 4κκ'a·ζ.
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * &zeta;
        let coeff_mm_a_u = &(&dw_a_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_u)
            + &(2.0 * &dw_u * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_u * &s1_a)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a_u)
            - &(&dm_a_u * &geom.d2q_dq02)
            - &(&dm_a * &g2_u)
            - &(&dm_u * &g2_a)
            - &(&rows.m * &g2_a_u);
        // coeff_ml_a_u = ∂²(coeff_ml)/∂a∂u = 0 (coeff_ml ≡ Fisher 0).
        let coeff_ml_a_u = Array1::<f64>::zeros(n);
        // coeff_ll_a_u = ∂²(2κ²a)/∂a∂u for the Fisher (ls,ls) block (#566):
        // 4a(κ'²+κκ'')·e_a·ζ + 4κκ'a·η_au (the η_au=zls_a_u mixed leg), mirroring
        // the dense mixed-drift helper.
        let coeff_ll_a_u = 4.0
            * &rows.obs_weight
            * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
            * &(e_a * &zeta)
            + 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * &zls_a_u;

        let a = &rows.w * &geom.dq_dq0;
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_a_u = &dw_a_u * &geom.dq_dq0 + &dw_a * &s1_u + &dw_u * &s1_a + &rows.w * &s1_a_u;
        let c = -&rows.m;
        let c_u = -&dm_u;
        let c_a = -&dm_a;
        let c_a_u = -&dm_a_u;
        // l (logσ↔wiggle) is mean⊥scale Fisher 0 (wiggle is mean-side), so its
        // β-drift (l_u), ψ (l_a), and mixed (l_a_u) derivatives all vanish.
        let l = Array1::<f64>::zeros(n);
        let l_u = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let l_a_u = Array1::<f64>::zeros(n);

        let hmm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_a1 + &hmm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a_u)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml_u.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a_u, x_ls)?;
        let hll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_a1 + &hll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a_u)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(xmu, &a_a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_u)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_a_u)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c_u.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_u),
            )?
            + &xt_diag_y_dense(xmu, &c_a_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_u)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a_u)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(x_ls, &l_a_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_u)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a_u)?;
        let hww_a_u = xt_diag_y_dense(&basis_a_u, &rows.w, &geom.basis)?;
        let hww_aw = xt_diag_y_dense(&basis_a, &dw_u, &geom.basis)?;
        let hww_au = xt_diag_y_dense(&basis_a, &rows.w, &basis_u)?;
        let h_ww = &hww_a_u
            + &hww_a_u.t()
            + &hww_aw
            + hww_aw.t()
            + &hww_au
            + hww_au.t()
            + &xt_diag_x_dense(&geom.basis, &dw_a_u)?;

        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &xmu,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &xmu,
            &x_ls,
        )
    }
}


impl CustomFamily for GaussianLocationScaleWiggleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware (see GaussianLocationScaleFamily for derivation): when
        // `use_joint_matrix_free_path` selects the workspace operator, joint
        // Hv apply is O(n · (p_t + p_ℓ + p_w)) — the row-streaming RowCoeffOperator
        // never materializes the dense (p_t + p_ℓ + p_w)² matrix.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        validate_monotone_wiggle_beta_nonnegative(
            &beta,
            "GaussianLocationScaleWiggleFamily post-update",
        )?;
        Ok(beta)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        // Per-row kernel emits 6 working values into pre-allocated outputs;
        // ll is reduced via Rayon's sum. Independent across rows. Note
        // wmu == ww (both equal location_working_weight) and the mean+wiggle
        // working responses share row.location_working_shift, applied to
        // eta_mu[i] and etaw[i] respectively. The previous `q = eta_mu + etaw`
        // intermediate is inlined to avoid an extra n-vector allocation.
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut zls = Array1::<f64>::zeros(n);
        let mut wls = Array1::<f64>::zeros(n);
        let mut zw = Array1::<f64>::zeros(n);
        let mut ww = Array1::<f64>::zeros(n);
        const CHUNK: usize = 1024;
        let zmu_s = zmu
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let wmu_s = wmu
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let zls_s = zls
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let wls_s = wls
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let zw_s = zw.as_slice_memory_order_mut().expect("zeros is contiguous");
        let ww_s = ww.as_slice_memory_order_mut().expect("zeros is contiguous");
        let y_view = self.y.view();
        let w_view = self.weights.view();
        let eta_mu_view = eta_mu.view();
        let eta_ls_view = eta_ls.view();
        let etaw_view = etaw.view();
        let ll: f64 = zmu_s
            .par_chunks_mut(CHUNK)
            .zip(wmu_s.par_chunks_mut(CHUNK))
            .zip(zls_s.par_chunks_mut(CHUNK))
            .zip(wls_s.par_chunks_mut(CHUNK))
            .zip(zw_s.par_chunks_mut(CHUNK))
            .zip(ww_s.par_chunks_mut(CHUNK))
            .enumerate()
            .map(
                |(chunk_idx, (((((zmu_c, wmu_c), zls_c), wls_c), zw_c), ww_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let q_i = eta_mu_view[i] + etaw_view[i];
                        let row = gaussian_diagonal_row_kernel(
                            y_view[i],
                            q_i,
                            eta_ls_view[i],
                            w_view[i],
                            ln2pi,
                        );
                        let w_i = row.location_working_weight;
                        let shift = row.location_working_shift;
                        zmu_c[local] = eta_mu_view[i] + shift;
                        wmu_c[local] = w_i;
                        zw_c[local] = etaw_view[i] + shift;
                        ww_c[local] = w_i;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                },
            )
            .sum();

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(zmu, wmu)?,
                BlockWorkingSet::diagonal_checked(zls, wls)?,
                BlockWorkingSet::diagonal_checked(zw, ww)?,
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != self.y.len()
            || eta_ls.len() != self.y.len()
            || etaw.len() != self.y.len()
            || self.weights.len() != self.y.len()
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = eta_mu + etaw;
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        for i in 0..self.y.len() {
            let sigma_i = logb_sigma_from_eta_scalar(eta_ls[i]);
            let inv_s2 = (sigma_i * sigma_i).recip();
            let r = self.y[i] - q[i];
            ll += self.weights[i] * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()));
        }
        Ok(ll)
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        use rayon::iter::ParallelIterator;
        let ll: f64 = subsample
            .rows
            .par_iter()
            .map(|row| {
                let i = row.index;
                let wi = self.weights[i];
                if wi == 0.0 {
                    return 0.0;
                }
                let sigma_i = logb_sigma_from_eta_scalar(eta_ls[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = self.y[i] - eta_mu[i] - etaw[i];
                row.weight * wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
            })
            .sum();
        Ok(ll)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = self
            .mu_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing mu design".to_string()
            })?
            .ncols();
        let p_ls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing log-sigma design"
                    .to_string()
            })?
            .ncols();
        let pw = block_states[Self::BLOCK_WIGGLE].beta.len();
        let total = pmu + p_ls + pw;
        let (start, end) = match block_idx {
            Self::BLOCK_MU => (0usize, pmu),
            Self::BLOCK_LOG_SIGMA => (pmu, pmu + p_ls),
            Self::BLOCK_WIGGLE => (pmu + p_ls, total),
            _ => return Ok(None),
        };
        if d_beta.len() != end - start {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleWiggleFamily block {block_idx} d_beta length mismatch: got {}, expected {}",
                d_beta.len(),
                end - start
            ) }.into());
        }
        let mut d_beta_flat = Array1::<f64>::zeros(total);
        d_beta_flat.slice_mut(s![start..end]).assign(d_beta);
        let (xmu, x_ls) = self.dense_block_designs()?;
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                block_states,
                &xmu,
                &x_ls,
                &d_beta_flat,
            )?
            .ok_or_else(|| "missing Gaussian wiggle exact joint directional Hessian".to_string())?;
        Ok(Some(d_joint.slice(s![start..end, start..end]).to_owned()))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    /// Outer-aware joint ψ workspace with optional row subsample.
    ///
    /// The wiggle ψ workspace shares the generic `LocationScaleJointPsiWorkspace`
    /// with the non-wiggle GLS family, and the subsample is plumbed through
    /// the trait. The wiggle's `ws_psi_*_from_parts` impls currently drop the
    /// subsample and fall back to the full-data exact wiggle ψ path; see
    /// their inline rationale and the `apply_ht_mask_*` helpers used by the
    /// non-wiggle GLS family. Storing the subsample here keeps the workspace
    /// signature uniform across both families and leaves a hook for the
    /// follow-up that refactors the wiggle inline arrays into a weights
    /// struct so HT masking can be applied in one place. Even without that
    /// refactor, the total outer score under subsampling remains an unbiased
    /// estimator of the full-data outer score: HT-unbiased LL
    /// (`log_likelihood_only_with_options`) + HT-unbiased ρ-Hessian
    /// (`exact_newton_joint_hessian_workspace_with_options`) + exact-unbiased
    /// ψ (the wiggle workspace path) = unbiased.
    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace::new_with_subsample(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
                options.outer_score_subsample.clone(),
            )?,
        )))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.is_empty() {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "Gaussian wiggle geometry requires mean block".to_string(),
            }
            .into());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        if eta_mu.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "Gaussian wiggle geometry input size mismatch".to_string(),
            }
            .into());
        }
        let x = self.wiggle_design(eta_mu.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "Gaussian dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays in `pieces` (`coeff_mm`, `coeff_ml`,
    /// `coeff_ll`, `coeff_mw_b`, `coeff_mw_d`, `coeff_lw_b`, `coeff_ww`) —
    /// which every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) Y` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient
    /// is multiplied by `WeightedOuterRow.weight` (the inverse-inclusion
    /// factor 1/π_i; uniform or stratified sampling both supported), and
    /// non-sampled rows are zeroed. The `basis`/`basis_d1` matrices are
    /// row-weight-independent and remain unchanged. Note that the Gaussian
    /// wiggle has one fewer cross-coefficient than the binomial wiggle
    /// (no `coeff_lw_d`) because the wiggle enters the Gaussian likelihood
    /// only through `q = η_μ + η_w` (no σ-chain). The resulting joint Hessian
    /// is an unbiased estimator of the full-data joint Hessian. Inner PIRLS
    /// never installs the option, so the inner solve continues to consume
    /// the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = GaussianLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// GaussianLocationScaleWiggleFamily overrides
    /// `log_likelihood_only_with_options` and
    /// `exact_newton_joint_hessian_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood and
    /// joint Hessian. The ψ-workspace path is also subsample-aware via
    /// `exact_newton_joint_psi_workspace_with_options`, which threads the
    /// subsample down to per-row weight masking inside the joint-ψ second-
    /// order and directional-derivative reductions. Inner-PIRLS and final-
    /// covariance paths never install the option, so they continue to
    /// consume the exact full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Same gating as the workspace impl above: matrix-free fires when
        // `exact_joint_dense_block_designs` is satisfiable, which requires
        // both location and scale block designs to be present.  The wiggle
        // block is folded into the operator via the per-row pieces — its
        // presence is implied by reaching the wiggle family in the first
        // place — so the predicate matches the non-wiggle case.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }
}


/// Matrix-free joint-Hessian operator for the 3-block Gaussian
/// location-scale wiggle family. See `GaussianLocationScaleWiggleHessianRowPieces`
/// for the per-row weight structure. The matvec applies
///
///   r_μ  = D_mm u_μ + D_ml u_ls + D_mw_b (B v_w) + D_mw_d (B' v_w),
///   r_ls = D_ml u_μ + D_ll u_ls + D_lw_b (B v_w),
///   r_b  = D_mw_b u_μ + D_lw_b u_ls + D_ww (B v_w),
///   r_d  = D_mw_d u_μ,
///
/// then forms `out_w = B^T r_b + (B')^T r_d`. The ls-wiggle cross block has
/// no B' contribution because the wiggle enters the Gaussian likelihood only
/// through `q = η_μ + η_w` (no σ-chain), so the Gaussian wiggle has one
/// fewer cross-coefficient than the binomial wiggle.
struct GaussianLocationScaleWiggleHessianWorkspace {
    family: GaussianLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    xmu: Arc<Array2<f64>>,
    x_ls: Arc<Array2<f64>>,
    pieces: GaussianLocationScaleWiggleHessianRowPieces,
}


impl GaussianLocationScaleWiggleHessianWorkspace {
    fn new(
        family: GaussianLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let pieces = family.wiggle_hessian_row_pieces(&block_states)?;
        Ok(Self {
            family,
            block_states,
            xmu: Arc::new(xmu),
            x_ls: Arc::new(x_ls),
            pieces,
        })
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place.
    ///
    /// Each sampled row's `coeff_*[i]` is multiplied by its
    /// `WeightedOuterRow.weight` (the HT inverse-inclusion factor 1/π_i —
    /// uniform or stratified sampling both supported). All non-sampled rows
    /// are zeroed. Because every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) is row-linear in these arrays
    /// via `Xᵀ diag(W) Y`, the resulting joint-Hessian is an unbiased
    /// estimator of the full-data joint Hessian. The `basis`/`basis_d1`
    /// matrices are independent of the per-row weights and remain unchanged.
    /// The Gaussian wiggle has 7 coefficient arrays (no `coeff_lw_d`, unlike
    /// the binomial wiggle's 8) because the wiggle enters the Gaussian
    /// likelihood only through `q = η_μ + η_w` (no σ-chain).
    fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.pieces.coeff_mm.len();
        let mut mask_mm = Array1::<f64>::zeros(n);
        let mut mask_ml = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        let mut mask_mw_b = Array1::<f64>::zeros(n);
        let mut mask_mw_d = Array1::<f64>::zeros(n);
        let mut mask_lw_b = Array1::<f64>::zeros(n);
        let mut maskww = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            let w = r.weight;
            mask_mm[i] = self.pieces.coeff_mm[i] * w;
            mask_ml[i] = self.pieces.coeff_ml[i] * w;
            mask_ll[i] = self.pieces.coeff_ll[i] * w;
            mask_mw_b[i] = self.pieces.coeff_mw_b[i] * w;
            mask_mw_d[i] = self.pieces.coeff_mw_d[i] * w;
            mask_lw_b[i] = self.pieces.coeff_lw_b[i] * w;
            maskww[i] = self.pieces.coeff_ww[i] * w;
        }
        self.pieces.coeff_mm = mask_mm;
        self.pieces.coeff_ml = mask_ml;
        self.pieces.coeff_ll = mask_ll;
        self.pieces.coeff_mw_b = mask_mw_b;
        self.pieces.coeff_mw_d = mask_mw_d;
        self.pieces.coeff_lw_b = mask_lw_b;
        self.pieces.coeff_ww = maskww;
    }
}


impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleWiggleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but routed through the
        // already-existing `assemble_dense` row-pieces helper (six GEMMs:
        // h_mm, h_ml, h_mw_b, h_mw_d, h_lw, h_ww). Avoids `total` canonical-
        // basis HVPs in `MatrixFreeSpdOperator::materialize_dense_operator`,
        // which at large scale (n≈320k, p_total≈82) costs ~568s per κ-iter
        // versus ~1s for the dense build.
        let dense = self
            .pieces
            .assemble_dense(self.xmu.as_ref(), self.x_ls.as_ref())?;
        Ok(Some(dense))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggle matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let v_mu = v.slice(s![0..pmu]);
        let v_ls = v.slice(s![pmu..pmu + p_ls]);
        let v_w = v.slice(s![pmu + p_ls..total]);

        let u_mu = fast_av(self.xmu.as_ref(), &v_mu);
        let u_ls = fast_av(self.x_ls.as_ref(), &v_ls);
        let u_b = fast_av(&self.pieces.basis, &v_w);
        let u_d = fast_av(&self.pieces.basis_d1, &v_w);

        let r_mu = &self.pieces.coeff_mm * &u_mu
            + &self.pieces.coeff_ml * &u_ls
            + &self.pieces.coeff_mw_b * &u_b
            + &self.pieces.coeff_mw_d * &u_d;
        let r_ls = &self.pieces.coeff_ml * &u_mu
            + &self.pieces.coeff_ll * &u_ls
            + &self.pieces.coeff_lw_b * &u_b;
        let r_b = &self.pieces.coeff_mw_b * &u_mu
            + &self.pieces.coeff_lw_b * &u_ls
            + &self.pieces.coeff_ww * &u_b;
        let r_d = &self.pieces.coeff_mw_d * &u_mu;

        let out_mu = fast_atv(self.xmu.as_ref(), &r_mu);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let out_w = fast_atv(&self.pieces.basis, &r_b) + &fast_atv(&self.pieces.basis_d1, &r_d);

        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..pmu + p_ls]).assign(&out_ls);
        out.slice_mut(s![pmu + p_ls..total]).assign(&out_w);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        // Diagonals are independent column-wise reductions: parallelize.
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let diag_mu: Vec<f64> = (0..pmu)
            .into_par_iter()
            .map(|j| {
                let col = self.xmu.column(j);
                col.iter()
                    .zip(self.pieces.coeff_mm.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_ls: Vec<f64> = (0..p_ls)
            .into_par_iter()
            .map(|j| {
                let col = self.x_ls.column(j);
                col.iter()
                    .zip(self.pieces.coeff_ll.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_w: Vec<f64> = (0..pw)
            .into_par_iter()
            .map(|j| {
                let col = self.pieces.basis.column(j);
                col.iter()
                    .zip(self.pieces.coeff_ww.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let mut diag = Array1::<f64>::zeros(total);
        for (j, v) in diag_mu.into_iter().enumerate() {
            diag[j] = v;
        }
        for (j, v) in diag_ls.into_iter().enumerate() {
            diag[pmu + j] = v;
        }
        for (j, v) in diag_w.into_iter().enumerate() {
            diag[pmu + p_ls + j] = v;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                self.xmu.as_ref(),
                self.x_ls.as_ref(),
                d_beta_flat,
            )
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.gls_wiggle_directional_operator(
            &self.block_states,
            self.xmu.clone(),
            self.x_ls.clone(),
            d_beta_flat,
        )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                self.xmu.as_ref(),
                self.x_ls.as_ref(),
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.gls_wiggle_second_directional_operator(
            &self.block_states,
            self.xmu.clone(),
            self.x_ls.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}


impl CustomFamilyGenerative for GaussianLocationScaleWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_wiggle = &block_states[Self::BLOCK_WIGGLE].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let n = eta_mu.len();
        let mean = gamlss_rowwise_map(n, |i| eta_mu[i] + eta_wiggle[i]);
        let sigma = gamlss_rowwise_map(n, |i| logb_sigma_from_eta_scalar(eta_log_sigma[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}


fn expect_single_block<'a>(
    block_states: &'a [ParameterBlockState],
    family_name: &str,
) -> Result<&'a ParameterBlockState, String> {
    if block_states.len() != 1 {
        return Err(GamlssError::DimensionMismatch {
            reason: format!("{family_name} expects 1 block, got {}", block_states.len()),
        }
        .into());
    }
    Ok(&block_states[0])
}


#[derive(Clone)]
pub struct BinomialMeanWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction during
    /// exact-Newton joint psi evaluation. Defaults to
    /// `ResourcePolicy::default_library()` when the family is built without
    /// an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}


struct BinomialMeanWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    basis_d3: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
    d4q_dq04: Array1<f64>,
}


struct BinomialMeanWiggleJointPsiDirection {
    x_eta_psi: Option<Array2<f64>>,
    z_eta_psi: Array1<f64>,
}


impl BinomialMeanWiggleFamily {
    pub const BLOCK_ETA: usize = 0;
    pub const BLOCK_WIGGLE: usize = 1;

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            options.derivative_order,
        )
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d_constrained.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d_constrained.dot(&beta_link_wiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle second-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d2.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d2.dot(&beta_link_wiggle))
    }

    fn wiggle_d3basis_constrained(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        if d3.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle third-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d3.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d3.dot(&beta_link_wiggle))
    }

    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle fourth-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d4.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    fn wiggle_geometry(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<BinomialMeanWiggleGeometry, String> {
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(BinomialMeanWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        })
    }

    fn neglog_q_derivatives(&self, y: f64, weight: f64, q: f64) -> Result<(f64, f64, f64), String> {
        let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        // Pass μ RAW: the dispatch returns the exact q-derivatives of the
        // evaluated loss for every representable μ in (0,1) and handles the
        // saturated boundary itself. See binomial_location_scalerow (#948).
        Ok(binomial_neglog_q_derivatives_dispatch(
            y,
            weight,
            q,
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        ))
    }

    fn neglog_q_fourth_derivative(&self, y: f64, weight: f64, q: f64) -> Result<f64, String> {
        let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        // Pass μ RAW — see neglog_q_derivatives above (#948).
        binomial_neglog_q_fourth_derivative_dispatch(
            y,
            weight,
            q,
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        )
    }

    fn dense_eta_design_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<Cow<'a, Array2<f64>>, String> {
        if specs.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 specs, got {}",
                    specs.len()
                ),
            }
            .into());
        }
        Ok(match specs[Self::BLOCK_ETA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_ETA]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialMeanWiggle dense_eta_design_fromspecs eta",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        })
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_eta: &Array2<f64>,
    ) -> Result<Option<BinomialMeanWiggleJointPsiDirection>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi direction expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let p_eta = x_eta.ncols();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let x_eta_psi_map = resolve_custom_family_x_psi_map(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                        &self.policy,
                    )?;
                    let x_eta_psi = x_eta_psi_map.row_chunk(0..n)?;
                    let z_eta_psi = x_eta_psi.dot(beta_eta);
                    return Ok(Some(BinomialMeanWiggleJointPsiDirection {
                        x_eta_psi: Some(x_eta_psi),
                        z_eta_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    fn exact_newton_joint_psi_action(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        p_eta: usize,
    ) -> Result<Option<(CustomFamilyPsiDesignAction, Array1<f64>)>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi action expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let action = match CustomFamilyPsiDesignAction::from_first_derivative(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                    ) {
                        Ok(action) => action,
                        Err(_) => return Ok(None),
                    };
                    let z_eta_psi = action.forward_mul(beta_eta.view());
                    return Ok(Some((action, z_eta_psi)));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    fn bmw_static_hessian_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
    ) -> Result<Arc<RowCoeffOperator>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, _) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            coeff_eta[row] = hessian_coeff_fromobjective_q_terms(m1, m2, a, a, b);
            coeff_etaw_b[row] = m2 * a;
            coeff_etaw_d1[row] = m1;
            coeff_ww[row] = m2;
        }
        Ok(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (1, 1, coeff_ww),
            ],
            n,
        )))
    }

    fn bmw_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let u_eta = d_beta_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_flat.slice(s![p_eta..total]).to_owned();
        let xi = fast_av(x_eta_arc.as_ref(), &u_eta);
        let phi = fast_av(&geom.basis, &uw);
        let basis1_u = fast_av(&geom.basis_d1, &uw);
        let basis2_u = fast_av(&geom.basis_d2, &uw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_u = a * xi[row] + phi[row];
            let a_u = b * xi[row] + basis1_u[row];
            let b_u = c * xi[row] + basis2_u[row];
            coeff_eta[row] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, q_u, a, a, b, a_u, a_u, b_u,
            );
            coeff_etaw_b[row] = m3 * q_u * a + m2 * a_u;
            coeff_etaw_d1[row] = m2 * (a * xi[row] + q_u);
            coeff_etaw_d2[row] = m1 * xi[row];
            coeff_ww_bb[row] = m3 * q_u;
            coeff_ww_db[row] = m2 * xi[row];
        }
        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
                (1, Arc::new(geom.basis_d2)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (0, 3, coeff_etaw_d2),
                (1, 1, coeff_ww_bb),
                (1, 2, coeff_ww_db),
            ],
            n,
        ))))
    }

    fn bmw_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_u_flat.len() != total || d_beta_v_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint second d_beta length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                total
            ) }.into());
        }
        let u_eta = d_beta_u_flat.slice(s![0..p_eta]).to_owned();
        let v_eta = d_beta_v_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_u_flat.slice(s![p_eta..total]).to_owned();
        let vw = d_beta_v_flat.slice(s![p_eta..total]).to_owned();

        let xi_u = fast_av(x_eta_arc.as_ref(), &u_eta);
        let xi_v = fast_av(x_eta_arc.as_ref(), &v_eta);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d3 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        let mut coeff_ww_ddb = Array1::<f64>::zeros(n);
        let mut coeff_ww_dd = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let m4 = self.neglog_q_fourth_derivative(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let d = geom.d4q_dq04[row];

            let q_u = a * xi_u[row] + phi_u[row];
            let a_u = b * xi_u[row] + b1u[row];
            let b_u = c * xi_u[row] + b2u[row];
            let q_v = a * xi_v[row] + phi_v[row];
            let a_v = b * xi_v[row] + b1v[row];
            let b_v = c * xi_v[row] + b2v[row];
            let q_uv = b * xi_u[row] * xi_v[row] + b1u[row] * xi_v[row] + b1v[row] * xi_u[row];
            let a_uv = c * xi_u[row] * xi_v[row] + b2u[row] * xi_v[row] + b2v[row] * xi_u[row];
            let b_uv = d * xi_u[row] * xi_v[row] + b3u[row] * xi_v[row] + b3v[row] * xi_u[row];

            coeff_eta[row] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, q_u, q_v, q_uv, a, a, b, a_u, a_v, a_u, a_v, a_uv, a_uv, b_u, b_v,
                b_uv,
            );
            let d2_c_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
            let dc_b_u = m3 * q_u * a + m2 * a_u;
            let dc_b_v = m3 * q_v * a + m2 * a_v;
            let c_b_static = m2 * a;
            let d2_c_b1 = m3 * q_u * q_v + m2 * q_uv;
            let dc_b1_u = m2 * q_u;
            let dc_b1_v = m2 * q_v;

            coeff_etaw_b[row] = d2_c_b;
            coeff_etaw_d1[row] = dc_b_u * xi_v[row] + dc_b_v * xi_u[row] + d2_c_b1;
            coeff_etaw_d2[row] =
                c_b_static * xi_u[row] * xi_v[row] + dc_b1_u * xi_v[row] + dc_b1_v * xi_u[row];
            coeff_etaw_d3[row] = m1 * xi_u[row] * xi_v[row];

            let dw = m2;
            let dw_u = m3 * q_u;
            let dw_v = m3 * q_v;
            let dw_uv = m4 * q_u * q_v + m3 * q_uv;
            let xixj = xi_u[row] * xi_v[row];
            coeff_ww_bb[row] = dw_uv;
            coeff_ww_db[row] = dw_v * xi_u[row] + dw_u * xi_v[row];
            coeff_ww_ddb[row] = dw * xixj;
            coeff_ww_dd[row] = 2.0 * dw * xixj;
        }

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
                (1, Arc::new(geom.basis_d2)),
                (1, Arc::new(geom.basis_d3)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (0, 3, coeff_etaw_d2),
                (0, 4, coeff_etaw_d3),
                (1, 1, coeff_ww_bb),
                (1, 2, coeff_ww_db),
                (1, 3, coeff_ww_ddb),
                (2, 2, coeff_ww_dd),
            ],
            n,
        ))))
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// `BinomialMeanWiggle` has a single location output (n_outputs = 1):
    /// - block 0 (eta):    output 0 = design rows
    /// - block 1 (wiggle): all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "BinomialMeanWiggleFamily",
            n_outputs: 1,
            additive_blocks: &[Self::BLOCK_ETA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}


impl CustomFamily for BinomialMeanWiggleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// The binomial mean link-wiggle refit must NOT carry the full-span
    /// Jeffreys/Firth augmentation, for the same structural reason
    /// `GaussianLocationScaleWiggleFamily` opts out (#684–#688) — and the
    /// binomial wiggle hits it harder. This is a *second-stage* refit: the
    /// pilot binomial mean fit has already converged through the ordinary
    /// PIRLS path (which is itself un-Firthed unless the user opts in — the
    /// standard binomial fit logs `firth=false` / `jeffreys_logdet=none`), so
    /// the wiggle refit only adds a *penalized*, *monotone-constrained*
    /// I-spline link-shape correction `q = η + B(η)·β_w` around an
    /// already-finite mode. Two failure modes follow from leaving the term on
    /// (default `true`):
    ///
    /// 1. **Phantom stationarity residual.** When `H_pen` is full-rank and
    ///    well-conditioned (the normal case — e.g. `cond ≈ 5.5e2` on the #872
    ///    pure-probit repro) the Jeffreys gate smooth-steps the curvature
    ///    `H_Φ → 0`, but the matching score `∇Φ` does not vanish in lock-step,
    ///    so it leaks a nonzero `|∇L − Sβ + ∇Φ|` into the inner joint-Newton
    ///    KKT residual. The certificate then refuses every iterate and the
    ///    outer REML rejects all seeds (exactly the #684–#688 abort signature).
    /// 2. **Saturation barrier / divergence.** `−Φ = −½log|I_J|` is folded into
    ///    the objective and `∇Φ ∝ I_J⁻¹` into the gradient. The I-spline warp
    ///    can drive the binomial linear predictor toward saturation, where the
    ///    reduced Fisher information `I_J` goes singular: `−Φ → +∞` and
    ///    `∇Φ → ∞`. The augmented objective grows a barrier that the joint
    ///    Newton diverges into — the #872 repro runs the full 1200-cycle budget
    ///    with the augmented objective pinned at ~4.6e9 and the augmented
    ///    residual at ~5.8e9 while the plain data gradient is only ~2.3e2,
    ///    aborting the documented `link(type=flexible(...)) + linkwiggle(...)`
    ///    fit.
    ///
    /// Separation robustness is not lost: the wiggle block carries both a
    /// difference penalty (λ selected by REML) and a hard non-negativity
    /// constraint, and the underlying mean is fit by the pilot; a penalized,
    /// constrained refit around a finite pilot mode does not run away to
    /// `β → ∞` the way an unpenalized MLE can. Turning the term off here makes
    /// the wiggle refit consistent with the un-Firthed pilot and removes the
    /// phantom residual that blocked convergence.
    fn joint_jeffreys_term_required(&self) -> bool {
        false
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // The mean-wiggle Hessian is exposed as a row-coefficient operator,
        // so the hot representation cost is one Θ(n · (p_eta + p_w)) HVP
        // rather than dense Θ(n · (p_eta + p_w)^2) assembly.
        let p_total = specs
            .iter()
            .map(|s| s.design.ncols() as u64)
            .fold(0u64, |acc, p| acc.saturating_add(p));
        (self.y.len() as u64).saturating_mul(p_total.max(1))
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        validate_monotone_wiggle_beta_nonnegative(&beta, "BinomialMeanWiggleFamily post-update")?;
        Ok(beta)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let dq_dq0 = self.wiggle_dq_dq0(eta.view(), betaw.view())?;
        if dq_dq0.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily dq/dq0 length mismatch: got {}, expected {}",
                    dq_dq0.len(),
                    n
                ),
            }
            .into());
        }

        let mut ll = 0.0;
        let mut z_eta = Array1::<f64>::zeros(n);
        let mut w_eta = Array1::<f64>::zeros(n);
        let mut z_wiggle = Array1::<f64>::zeros(n);
        let mut w_wiggle = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = eta[i] + etaw[i];
            let (mu_q, d1_q) = inverse_link_mu_d1_for_inverse_link(&self.link_kind, q)
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            let yi = self.y[i];
            let wi = self.weights[i];
            ll += binomial_location_scale_log_likelihood(yi, wi, q, &self.link_kind, mu_q)?;

            let mu = mu_q.clamp(1e-12, 1.0 - 1e-12);
            let var = (mu * (1.0 - mu)).max(MIN_PROB);
            let dmu_deta = d1_q * dq_dq0[i];
            let dmu_dw = d1_q;
            if wi == 0.0 || !var.is_finite() {
                z_eta[i] = eta[i];
                z_wiggle[i] = etaw[i];
                continue;
            }

            if dmu_deta.is_finite() {
                w_eta[i] = floor_positiveweight(wi * (dmu_deta * dmu_deta / var), MIN_WEIGHT);
                z_eta[i] = eta[i] + (yi - mu) / signedwith_floor(dmu_deta, MIN_DERIV);
            } else {
                z_eta[i] = eta[i];
            }

            if dmu_dw.is_finite() {
                w_wiggle[i] = floor_positiveweight(wi * (dmu_dw * dmu_dw / var), MIN_WEIGHT);
                z_wiggle[i] = etaw[i] + (yi - mu) / signedwith_floor(dmu_dw, MIN_DERIV);
            } else {
                z_wiggle[i] = etaw[i];
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(z_eta, w_eta)?,
                BlockWorkingSet::diagonal_checked(z_wiggle, w_wiggle)?,
            ],
        })
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.is_empty() {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "wiggle geometry requires eta block".to_string(),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        if eta.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily eta size mismatch".to_string(),
            }
            .into());
        }
        let x = self.wiggle_design(eta.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let x_eta = self.dense_eta_design_fromspecs(specs)?.into_owned();
        let workspace =
            BinomialMeanWiggleHessianWorkspace::new(self.clone(), block_states.to_vec(), x_eta)?;
        Ok(Some(Arc::new(workspace)))
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        self.dense_eta_design_fromspecs(specs).is_ok()
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, _) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            coeff_eta[row] = hessian_coeff_fromobjective_q_terms(m1, m2, a, a, b);
            coeff_etaw_b[row] = m2 * a;
            coeff_etaw_d1[row] = m1;
            coeff_ww[row] = m2;
        }
        let h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww)?;
        assert_eq!(h_eta_eta.nrows(), p_eta);
        assert_eq!(h_ww.nrows(), pw);
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &h_eta_eta, &h_eta_w, &h_ww,
        )))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        if d_beta_flat.len() != p_eta + pw {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    p_eta + pw
                ),
            }
            .into());
        }
        let u_eta = d_beta_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_flat.slice(s![p_eta..p_eta + pw]).to_owned();
        let xi = x_eta.dot(&u_eta);
        let phi = geom.basis.dot(&uw);
        let basis1_u = geom.basis_d1.dot(&uw);
        let basis2_u = geom.basis_d2.dot(&uw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_u = a * xi[row] + phi[row];
            let a_u = b * xi[row] + basis1_u[row];
            let b_u = c * xi[row] + basis2_u[row];
            coeff_eta[row] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, q_u, a, a, b, a_u, a_u, b_u,
            );
            coeff_etaw_b[row] = m3 * q_u * a + m2 * a_u;
            coeff_etaw_d1[row] = m2 * (a * xi[row] + q_u);
            coeff_etaw_d2[row] = m1 * xi[row];
            coeff_ww_bb[row] = m3 * q_u;
            coeff_ww_db[row] = m2 * xi[row];
        }

        let d_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let d_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let d_h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + a_ww.t();
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d_h_eta_eta,
            &d_h_eta_w,
            &d_h_ww,
        )))
    }

    /// Exact second-order directional derivative D²H[u,v] of the joint Hessian
    /// for the BinomialMeanWiggle two-block model (eta, wiggle).
    ///
    /// # Mathematical derivation
    ///
    /// The negative log-likelihood Hessian element for indices (a, b) in the
    /// joint coefficient vector is:
    ///
    ///   H_ab = m2 * q_a * q_b + m1 * q_ab
    ///
    /// where m_k = d^k F / dq^k (k-th derivative of the negative log-likelihood
    /// w.r.t. the effective predictor q), q_a = dq/d(beta_a), and q_ab =
    /// d²q/(d(beta_a) d(beta_b)).
    ///
    /// The effective predictor is q = q0 + w(q0) where q0 = X_eta * beta_eta
    /// and w(q0) = B(q0) * beta_w is the link wiggle.  Write:
    ///   a = dq/dq0 = 1 + B'·beta_w       (geometry first derivative)
    ///   b = d²q/dq0² = B''·beta_w         (geometry second derivative)
    ///   c = d³q/dq0³ = B'''·beta_w        (geometry third derivative)
    ///   d = d⁴q/dq0⁴ = B''''·beta_w       (geometry fourth derivative)
    ///
    /// For a perturbation direction u = (u_eta, u_w), the chain-rule
    /// perturbations are:
    ///   q_u   = a·xi_u + phi_u             (first-order predictor perturbation)
    ///   a_u   = b·xi_u + basis1_u          (perturbation of geometry factor a)
    ///   b_u   = c·xi_u + basis2_u          (perturbation of geometry factor b)
    ///   c_u   = d·xi_u + basis3_u          (perturbation of geometry factor c)
    ///
    /// where xi_u = X_eta·u_eta, phi_u = B·u_w, basis_k_u = B^(k)·u_w.
    ///
    /// Mixed second-order perturbations (u,v) are:
    ///   q_uv  = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
    ///   a_uv  = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
    ///   b_uv  = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u
    ///
    /// ## Block decomposition
    ///
    /// **eta-eta block** (X_eta' diag(coeff) X_eta):
    ///   The Hessian element for eta indices (i,j) factors as
    ///     H(eta_i, eta_j) = [m2·a² + m1·b] · x_eta(i)·x_eta(j)
    ///   so D²H_eta_eta[u,v] = X_eta' diag(coeff_eta) X_eta
    ///   where coeff_eta uses `second_directionalhessian_coeff_fromobjective_q_terms`
    ///   with q_a=a, q_b=a, q_ab=b and their chain-rule perturbations.
    ///
    /// **eta-w block** (X_eta' diag(...) [B, B', B'', B''']):
    ///   The static Hessian is:
    ///     H(eta_i, w_j) = (m2·a)·x_eta(i)·B_j + m1·x_eta(i)·B'_j
    ///   Taking D²[u,v] requires differentiating both the scalar coefficients
    ///   (m2·a, m1) and the basis matrices (B, B' depend on q0 via the chain
    ///   rule dB_j/du = B'_j·xi_u).  The full product rule gives four basis-matrix
    ///   tiers: B, B', B'', B'''.
    ///
    /// **w-w block** (B' diag(...) B, etc.):
    ///   The static Hessian is H(w_i, w_j) = m2·B_i·B_j.
    ///   D²[u,v] expands via the product rule on m2, B_i, B_j, each of which
    ///   depends on beta through q and q0.  This gives terms involving
    ///   B·B, B'·B, B'·B', and B''·B (all symmetrised).
    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_u_flat.len() != total || d_beta_v_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint second d_beta length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                total
            ) }.into());
        }

        // Split directions into eta and wiggle components.
        let u_eta = d_beta_u_flat.slice(s![0..p_eta]).to_owned();
        let v_eta = d_beta_v_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_u_flat.slice(s![p_eta..total]).to_owned();
        let vw = d_beta_v_flat.slice(s![p_eta..total]).to_owned();

        // Per-row linear-predictor perturbations from each direction.
        let xi_u = x_eta.dot(&u_eta); // eta perturbation in direction u
        let xi_v = x_eta.dot(&v_eta); // eta perturbation in direction v
        let phi_u = geom.basis.dot(&uw); // direct wiggle basis, direction u
        let phi_v = geom.basis.dot(&vw); // direct wiggle basis, direction v
        let b1u = geom.basis_d1.dot(&uw); // first-derivative basis, direction u
        let b1v = geom.basis_d1.dot(&vw);
        let b2u = geom.basis_d2.dot(&uw); // second-derivative basis, direction u
        let b2v = geom.basis_d2.dot(&vw);
        let b3u = geom.basis_d3.dot(&uw); // third-derivative basis, direction u
        let b3v = geom.basis_d3.dot(&vw);

        // Per-row chain-rule perturbations of q, a = dq/dq0, b = d²q/dq0²:
        //   q_u = a·xi_u + phi_u
        //   a_u = b·xi_u + basis1_u
        //   b_u = c·xi_u + basis2_u
        //   c_u = d·xi_u + basis3_u
        // Mixed second-order perturbations:
        //   q_uv = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
        //   a_uv = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
        //   b_uv = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u

        // Scaled basis matrices for the cross-product terms in the w-w and eta-w
        // blocks (same pattern as GaussianLocationScaleWiggleFamily).
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?; // dB/du = B'·xi_u
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?; // dB/dv = B'·xi_v
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?; // d²B/dudv = B''·xi_u·xi_v
        // Per-row coefficient arrays for assembling the block-matrix products.
        let mut coeff_eta = Array1::<f64>::zeros(n);

        // Coefficients for the eta-w block: X_eta' diag(c_*) M where M ∈ {B, B', B'', B'''}
        //
        // The static cross-Hessian is:
        //   H(eta_i, w_j) = (m2·a)·x_i·B_j + m1·x_i·B'_j
        // where B_j and B'_j are row evaluations of basis column j.
        //
        // Write C_B = m2·a (scalar coefficient multiplying B in the cross block)
        // and   C_B1 = m1  (scalar coefficient multiplying B' in the cross block).
        //
        // Product rule on C_B·B:
        //   d(C_B·B)/du = (dC_B/du)·B + C_B·B'·xi_u
        //   d²(C_B·B)/dudv = (d²C_B/dudv)·B + (dC_B/du)·B'·xi_v
        //                   + (dC_B/dv)·B'·xi_u + C_B·B''·xi_u·xi_v
        //
        // Product rule on C_B1·B':
        //   d²(C_B1·B')/dudv = (d²C_B1/dudv)·B' + (dC_B1/du)·B''·xi_v
        //                     + (dC_B1/dv)·B''·xi_u + C_B1·B'''·xi_u·xi_v
        //
        // Derivatives of the scalar coefficients:
        //   C_B  = m2·a
        //   dC_B/du  = m3·q_u·a + m2·a_u
        //   dC_B/dv  = m3·q_v·a + m2·a_v
        //   d²C_B/dudv = m4·q_u·q_v·a + m3·(q_uv·a + q_u·a_v + q_v·a_u) + m2·a_uv
        //
        //   C_B1 = m1
        //   dC_B1/du = m2·q_u
        //   dC_B1/dv = m2·q_v
        //   d²C_B1/dudv = m3·q_u·q_v + m2·q_uv
        //
        // Grouping by basis-matrix tier:
        //   B:   d²C_B/dudv
        //   B':  (dC_B/du)·xi_v + (dC_B/dv)·xi_u + d²C_B1/dudv
        //   B'': C_B·xi_u·xi_v + (dC_B1/du)·xi_v + (dC_B1/dv)·xi_u
        //   B''': C_B1·xi_u·xi_v
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d3 = Array1::<f64>::zeros(n);

        // Coefficients for the w-w block.
        //
        // The static w-w Hessian is:
        //   H(w_i, w_j) = m2·B_i·B_j
        //
        // Note: there is no m1·q_ij term because d²q/(d(beta_w_i) d(beta_w_j)) = 0
        // (the basis vectors B_i enter q linearly in beta_w).
        //
        // Product rule on m2·B_i·B_j, treating each factor as depending on beta:
        //   d²(m2·B_i·B_j)/dudv
        //     = (d²m2/dudv)·B_i·B_j                        → B'diag B  (symmetrised)
        //     + (dm2/du)·(B'_i·xi_v·B_j + B_i·B'_j·xi_v)  → dw_u terms
        //     + (dm2/dv)·(B'_i·xi_u·B_j + B_i·B'_j·xi_u)  → dw_v terms
        //     + m2·(B''_i·xi_u·xi_v·B_j + B'_i·xi_u·B'_j·xi_v
        //          + B'_i·xi_v·B'_j·xi_u + B_i·B''_j·xi_u·xi_v)
        //
        // where dm2/du = m3·q_u, dm2/dv = m3·q_v, d²m2/dudv = m4·q_u·q_v + m3·q_uv.
        //
        // Following the Gaussian LS wiggle pattern, we express this via:
        //   xt_diag_x_dense(B, dw_uv)                    — coeff: d²m2
        //   xt_diag_y_dense(basis_u, dw_v, B) + transpose — dB/du weighted by dm2/dv
        //   xt_diag_y_dense(basis_v, dw_u, B) + transpose — dB/dv weighted by dm2/du
        //   xt_diag_y_dense(basis_uv, w, B) + transpose   — d²B/dudv weighted by m2
        //   xt_diag_y_dense(basis_u, w, basis_v) + transpose — dB/du·dB/dv weighted by m2
        let mut dw = Array1::<f64>::zeros(n);
        let mut dw_u = Array1::<f64>::zeros(n);
        let mut dw_v = Array1::<f64>::zeros(n);
        let mut dw_uv = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let m4 = self.neglog_q_fourth_derivative(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let d = geom.d4q_dq04[row];

            // Chain-rule perturbations in direction u.
            let q_u = a * xi_u[row] + phi_u[row];
            let a_u = b * xi_u[row] + b1u[row];
            let b_u = c * xi_u[row] + b2u[row];

            // Chain-rule perturbations in direction v.
            let q_v = a * xi_v[row] + phi_v[row];
            let a_v = b * xi_v[row] + b1v[row];
            let b_v = c * xi_v[row] + b2v[row];

            // Mixed second-order perturbations.
            let q_uv = b * xi_u[row] * xi_v[row] + b1u[row] * xi_v[row] + b1v[row] * xi_u[row];
            let a_uv = c * xi_u[row] * xi_v[row] + b2u[row] * xi_v[row] + b2v[row] * xi_u[row];
            let b_uv = d * xi_u[row] * xi_v[row] + b3u[row] * xi_v[row] + b3v[row] * xi_u[row];

            // ── eta-eta block ──
            // H(eta_i, eta_j) uses q_a = a, q_b = a, q_ab = b (absorbing x_eta
            // into the matrix product).  The perturbations of these geometric
            // quantities are: dq_a/du = a_u, dq_b/du = a_u (since q_a = q_b = a),
            // dq_ab/du = b_u (since q_ab = b), and analogously for v.
            coeff_eta[row] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, q_u, q_v, q_uv, a, a, b, // q_a, q_b, q_ab
                a_u, a_v, // dq_a_u, dq_a_v
                a_u, a_v, // dq_b_u, dq_b_v  (q_b = a so same perturbation)
                a_uv, a_uv, // d2q_a_uv, d2q_b_uv
                b_u, b_v,  // dq_ab_u, dq_ab_v  (q_ab = b)
                b_uv, // d2q_ab_uv
            );

            // ── eta-w block coefficients ──
            // See the derivation in the docstring above.  We group by which basis
            // matrix tier (B, B', B'', B''') the coefficient multiplies.

            // d²(m2·a)/dudv
            let d2_c_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
            // d(m2·a)/du and d(m2·a)/dv
            let dc_b_u = m3 * q_u * a + m2 * a_u;
            let dc_b_v = m3 * q_v * a + m2 * a_v;
            // m2·a (static coefficient for B in the cross block)
            let c_b_static = m2 * a;
            // d²(m1)/dudv
            let d2_c_b1 = m3 * q_u * q_v + m2 * q_uv;
            // d(m1)/du and d(m1)/dv
            let dc_b1_u = m2 * q_u;
            let dc_b1_v = m2 * q_v;

            coeff_etaw_b[row] = d2_c_b;
            coeff_etaw_d1[row] = dc_b_u * xi_v[row] + dc_b_v * xi_u[row] + d2_c_b1;
            coeff_etaw_d2[row] =
                c_b_static * xi_u[row] * xi_v[row] + dc_b1_u * xi_v[row] + dc_b1_v * xi_u[row];
            coeff_etaw_d3[row] = m1 * xi_u[row] * xi_v[row];

            // ── w-w block coefficients ──
            // The w-w static Hessian coefficient is m2 (for B'diag B).
            dw[row] = m2;
            dw_u[row] = m3 * q_u;
            dw_v[row] = m3 * q_v;
            dw_uv[row] = m4 * q_u * q_v + m3 * q_uv;
        }

        // ── Assemble eta-eta block ──
        let d2_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;

        // ── Assemble eta-w block ──
        // The second-order directional derivative of the cross block H_eta_w is:
        //   d²H_eta_w[u,v] = X_eta' diag(coeff_etaw_b)  B
        //                   + X_eta' diag(coeff_etaw_d1) B'
        //                   + X_eta' diag(coeff_etaw_d2) B''
        //                   + X_eta' diag(coeff_etaw_d3) B'''
        let d2_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d3, &geom.basis_d3)?;

        // ── Assemble w-w block ──
        // Following the Gaussian LS wiggle pattern (lines 6351-6363), the w-w
        // second directional derivative is assembled from scaled basis products:
        //
        //   d²(m2·B_i·B_j)/dudv decomposition:
        //     (d²m2)     · B_i·B_j        → xt_diag_x(B, dw_uv)
        //     (dm2/du)   · dB_j/dv · B_i  → xt_diag_y(basis_v, dw_u, B) + transpose
        //     (dm2/dv)   · dB_j/du · B_i  → xt_diag_y(basis_u, dw_v, B) + transpose
        //     m2 · d²B_j/dudv · B_i       → xt_diag_y(basis_uv, dw, B) + transpose
        //     m2 · dB_i/du · dB_j/dv      → xt_diag_y(basis_u, dw, basis_v) + transpose
        let a_ab = xt_diag_y_dense(&basis_uv, &dw, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &dw, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let d2_h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + a_ij.t()
            + &a_iwj
            + a_iwj.t()
            + &a_jwi
            + a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;

        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d2_h_eta_eta,
            &d2_h_eta_w,
            &d2_h_ww,
        )))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi terms expect 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let implicit_dir =
            self.exact_newton_joint_psi_action(block_states, derivative_blocks, psi_index, p_eta)?;
        let dense_dir = if implicit_dir.is_none() {
            self.exact_newton_joint_psi_direction(
                block_states,
                derivative_blocks,
                psi_index,
                &x_eta,
            )?
        } else {
            None
        };
        let z_eta_psi = if let Some((_, ref z_eta_psi)) = implicit_dir {
            z_eta_psi
        } else if let Some(ref dir_a) = dense_dir {
            &dir_a.z_eta_psi
        } else {
            return Ok(None);
        };

        let mut objective_psi = 0.0;
        let mut score_eta_xa = Array1::<f64>::zeros(n);
        let mut score_eta_x = Array1::<f64>::zeros(n);
        let mut score_w_b = Array1::<f64>::zeros(n);
        let mut score_w_d1 = Array1::<f64>::zeros(n);

        let mut coeff_eta_eta_xx = Array1::<f64>::zeros(n);
        let mut coeff_eta_eta_xa_x = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let z_a = z_eta_psi[row];
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_a = a * z_a;

            objective_psi += m1 * q_a;

            score_eta_xa[row] = m1 * a;
            score_eta_x[row] = m2 * q_a * a + m1 * b * z_a;
            score_w_b[row] = m2 * q_a;
            score_w_d1[row] = m1 * z_a;

            coeff_eta_eta_xx[row] =
                m3 * q_a * a * a + m2 * (2.0 * a * b * z_a + q_a * b) + m1 * c * z_a;
            coeff_eta_eta_xa_x[row] = m2 * a * a + m1 * b;
            coeff_eta_w_xa_b[row] = m2 * a;
            coeff_eta_w_x_b[row] = m3 * q_a * a + m2 * b * z_a;
            coeff_eta_w_x_d1[row] = m2 * (a * z_a + q_a);
            coeff_eta_w_xa_d1[row] = m1;
            coeff_eta_w_x_d2[row] = m1 * z_a;
            coeff_ww_bb[row] = m3 * q_a;
            coeff_ww_db[row] = m2 * z_a;
        }

        let score_w = crate::faer_ndarray::fast_atv(&geom.basis, &score_w_b)
            + crate::faer_ndarray::fast_atv(&geom.basis_d1, &score_w_d1);

        if let Some((action, _)) = implicit_dir {
            let score_eta = action.transpose_mul(score_eta_xa.view())
                + crate::faer_ndarray::fast_atv(x_eta.as_ref(), &score_eta_x);
            let score_psi = binomial_pack_mean_wiggle_joint_score(&score_eta, &score_w);
            let x_eta_arc = shared_dense_arc(x_eta.as_ref());
            let basis_arc = Arc::new(geom.basis.clone());
            let basis_d1_arc = Arc::new(geom.basis_d1.clone());
            let basis_d2_arc = Arc::new(geom.basis_d2.clone());
            let zeros = Array1::<f64>::zeros(n);
            let operator = CustomFamilyJointPsiOperator::new(
                p_eta + pw,
                vec![
                    CustomFamilyJointDesignChannel::new(
                        0..p_eta,
                        Arc::clone(&x_eta_arc),
                        Some(action),
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d1_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d2_arc),
                        None,
                    ),
                ],
                vec![
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        0,
                        coeff_eta_eta_xa_x.clone(),
                        coeff_eta_eta_xx.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        1,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        0,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        2,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        0,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        3,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        0,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        1,
                        zeros.clone(),
                        coeff_ww_bb.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        1,
                        zeros.clone(),
                        coeff_ww_db.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(1, 2, zeros, coeff_ww_db.clone()),
                ],
            );
            return Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(operator)),
            }));
        }

        let dir_a =
            dense_dir.expect("dense psi direction should exist when implicit direction is absent");
        let x_eta_psi = dir_a
            .x_eta_psi
            .as_ref()
            .expect("dense eta psi design should exist when implicit direction is absent");
        let score_psi = binomial_pack_mean_wiggle_joint_score(
            &(crate::faer_ndarray::fast_atv(x_eta_psi, &score_eta_xa)
                + crate::faer_ndarray::fast_atv(x_eta.as_ref(), &score_eta_x)),
            &score_w,
        );
        let a_eta_eta = xt_diag_y_dense(x_eta_psi, &coeff_eta_eta_xa_x, &x_eta)?;
        let h_eta_eta = &a_eta_eta + &a_eta_eta.t() + &xt_diag_x_dense(&x_eta, &coeff_eta_eta_xx)?;
        let h_eta_w = xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + a_ww.t();

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: binomial_pack_mean_wiggle_joint_symmetrichessian(
                &h_eta_eta, &h_eta_w, &h_ww,
            ),
            hessian_psi_operator: None,
        }))
    }
}


struct BinomialMeanWiggleHessianWorkspace {
    family: BinomialMeanWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    x_eta: Arc<Array2<f64>>,
    hessian_operator: Arc<RowCoeffOperator>,
}


impl BinomialMeanWiggleHessianWorkspace {
    fn new(
        family: BinomialMeanWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        x_eta: Array2<f64>,
    ) -> Result<Self, String> {
        let x_eta = Arc::new(x_eta);
        let hessian_operator = family.bmw_static_hessian_operator(&block_states, x_eta.clone())?;
        Ok(Self {
            family,
            block_states,
            x_eta,
            hessian_operator,
        })
    }
}


impl ExactNewtonJointHessianWorkspace for BinomialMeanWiggleHessianWorkspace {
    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(
            crate::solver::estimate::reml::unified::HyperOperator::mul_vec(
                self.hessian_operator.as_ref(),
                v,
            ),
        ))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .directional_derivative_operator(d_beta_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family
            .bmw_directional_operator(&self.block_states, self.x_eta.clone(), d_beta_flat)
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .second_directional_derivative_operator(d_beta_u_flat, d_beta_v_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.bmw_second_directional_operator(
            &self.block_states,
            self.x_eta.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}


impl CustomFamilyGenerative for BinomialMeanWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta.len() != self.y.len() || etaw.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily generative size mismatch".to_string(),
            }
            .into());
        }
        let mean = gamlss_rowwise_map_result(self.y.len(), |i| {
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, eta[i] + etaw[i])
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            Ok(jet.mu)
        })?;
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}


/// Built-in Poisson log-link family (single parameter block).
#[derive(Clone)]
pub struct PoissonLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
}


impl PoissonLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "poisson_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}


/// Per-row IRLS contribution that a single-parameter log-link family must
/// produce. The shared driver `evaluate_log_link_diagonal_irls` consumes
/// these and assembles the full `FamilyEvaluation` so the three pieces of
/// code that previously lived inside each family — size validation, per-row
/// y validation + η clamping + saturated `exp`, the active-clamp w/z guard,
/// and the final return — exist in exactly one place.
struct DiagonalIrlsRow {
    /// Weighted contribution to ℓ at this row.
    log_lik_increment: f64,
    /// Unfloored observed Hessian weight (the driver applies `MIN_WEIGHT`).
    observed_weight: f64,
    /// Per-row Newton step on the working response: `z = e + working_step`.
    /// Each family computes this with its own (score, denominator); the
    /// driver only handles the active-clamp / zero-weight guard.
    working_step: f64,
}


/// Trait implemented by single-block log-link families that share the
/// diagonal IRLS structure (Poisson, Gamma). Each impl is responsible only
/// for the family-specific math: validating `y[i]` and producing the
/// per-row triple `(ℓ_increment, observed_weight, working_step)`.
trait LogLinkDiagonalIrlsFamily {
    /// Short, human-readable name used in size-mismatch errors.
    fn family_label(&self) -> &'static str;

    /// Read access to the shared (y, prior weights) buffers.
    fn y(&self) -> &Array1<f64>;
    fn prior_weights(&self) -> &Array1<f64>;

    /// Optional pre-loop validation hook for parameters outside the
    /// (y, weights, eta) triple (e.g. Gamma shape > 0).
    fn validate_self(&self) -> Result<(), String> {
        Ok(())
    }

    /// Validate `y[i]` and return an error message if rejected. Default
    /// implementation enforces only finiteness; concrete families override
    /// to add domain constraints.
    fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String>;

    /// Family-specific per-row math; `m = saturated_exp_eta(eta_clamped)`
    /// is computed by the driver and handed in.
    fn row_kernel(&self, yi: f64, e_clamped: f64, m: f64, prior_w: f64) -> DiagonalIrlsRow;
}


/// Shared IRLS driver for [`LogLinkDiagonalIrlsFamily`]. Centralises the
/// size-check, η-clamp, saturated-exp, active-clamp guard, ll accumulation,
/// and `FamilyEvaluation` assembly so all log-link families with the diagonal
/// structure (Poisson, Gamma) cannot drift apart numerically.
fn evaluate_log_link_diagonal_irls<F: LogLinkDiagonalIrlsFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
) -> Result<FamilyEvaluation, String> {
    let label = family.family_label();
    let eta = &expect_single_block(block_states, label)?.eta;
    let y = family.y();
    let prior_weights = family.prior_weights();
    let n = y.len();
    if eta.len() != n || prior_weights.len() != n {
        return Err(GamlssError::DimensionMismatch {
            reason: format!("{label} input size mismatch"),
        }
        .into());
    }
    family.validate_self()?;

    let mut ll = 0.0;
    let mut z = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);

    for i in 0..n {
        let yi = y[i];
        family.validate_yi(yi, i)?;
        let e_raw = eta[i];
        let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
        let active_clamp = e != e_raw;
        let m = saturated_exp_eta(e_raw);
        let prior_w = prior_weights[i];
        let row = family.row_kernel(yi, e, m, prior_w);
        ll += row.log_lik_increment;
        if prior_w == 0.0 || active_clamp {
            w[i] = 0.0;
            z[i] = e_raw;
        } else {
            w[i] = floor_positiveweight(row.observed_weight, MIN_WEIGHT);
            z[i] = e + row.working_step;
        }
    }

    Ok(FamilyEvaluation {
        log_likelihood: ll,
        blockworking_sets: vec![BlockWorkingSet::diagonal_checked(z, w)?],
    })
}


impl LogLinkDiagonalIrlsFamily for PoissonLogFamily {
    fn family_label(&self) -> &'static str {
        "PoissonLogFamily"
    }
    fn y(&self) -> &Array1<f64> {
        &self.y
    }
    fn prior_weights(&self) -> &Array1<f64> {
        &self.weights
    }
    fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String> {
        if !yi.is_finite() || yi < 0.0 {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "PoissonLogFamily requires non-negative finite y; found y[{idx}]={yi}"
                ),
            }
            .into());
        }
        Ok::<(), _>(())
    }
    #[inline]
    fn row_kernel(&self, yi: f64, e_clamped: f64, m: f64, prior_w: f64) -> DiagonalIrlsRow {
        // Drop log(y!) constant in objective.
        let log_lik_increment = prior_w * (yi * e_clamped - m);
        let dmu = m.max(MIN_DERIV);
        let var = m.max(MIN_PROB);
        DiagonalIrlsRow {
            log_lik_increment,
            observed_weight: prior_w * (dmu * dmu / var),
            // (yi - m)/dmu, identical to the previous direct expression.
            working_step: (yi - m) / signedwith_floor(dmu, MIN_DERIV),
        }
    }
}


impl CustomFamily for PoissonLogFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
    }
}


impl CustomFamilyGenerative for PoissonLogFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let eta = &expect_single_block(block_states, "PoissonLogFamily")?.eta;
        let mean = gamlss_rowwise_map(eta.len(), |i| saturated_exp_eta(eta[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Poisson,
        })
    }
}


/// Built-in Gamma log-link family (single parameter block, fixed shape).
#[derive(Clone)]
pub struct GammaLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub shape: f64,
}


impl GammaLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gamma_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}


impl LogLinkDiagonalIrlsFamily for GammaLogFamily {
    fn family_label(&self) -> &'static str {
        "GammaLogFamily"
    }
    fn y(&self) -> &Array1<f64> {
        &self.y
    }
    fn prior_weights(&self) -> &Array1<f64> {
        &self.weights
    }
    fn validate_self(&self) -> Result<(), String> {
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err(GamlssError::NonFinite {
                reason: "GammaLogFamily shape must be finite and > 0".to_string(),
            }
            .into());
        }
        Ok(())
    }
    fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String> {
        if !yi.is_finite() || yi <= 0.0 {
            return Err(GamlssError::InvalidInput {
                reason: format!("GammaLogFamily requires positive finite y; found y[{idx}]={yi}"),
            }
            .into());
        }
        Ok::<(), _>(())
    }
    #[inline]
    fn row_kernel(&self, yi: f64, e_clamped: f64, m: f64, prior_w: f64) -> DiagonalIrlsRow {
        assert!(e_clamped.is_finite());
        assert!((e_clamped.exp() - m).abs() <= 1.0e-8 * m.abs().max(1.0));
        // Gamma(shape=k, scale=mu/k), dropping eta-independent constants.
        let log_lik_increment = prior_w * (-self.shape * (yi / m + m.ln()));
        // Gamma with log mean is non-canonical. Use the exact observed
        // η-space curvature -d²ℓ/dη² = prior_w * shape * y / μ, not the
        // Fisher weight prior_w * shape, so diagonal REML/LAML Hessians
        // use the true Laplace curvature instead of a PQL/Fisher surrogate.
        let observed_weight = prior_w * self.shape * yi / m;
        let score = prior_w * self.shape * (yi / m - 1.0);
        // Mirror the pre-extraction formula z = e + score / w_floored exactly;
        // the driver applies MIN_WEIGHT *before* writing w[i], but the old
        // code divided by the already-floored w[i] for non-degenerate rows,
        // and the floor only activates on the degenerate `observed_weight <=
        // MIN_WEIGHT` tail. Reproduce that branch here to preserve bitwise
        // step shape on every row that used to hit the floor.
        let w_floored = observed_weight.max(MIN_WEIGHT);
        DiagonalIrlsRow {
            log_lik_increment,
            observed_weight,
            working_step: score / w_floored,
        }
    }
}


impl CustomFamily for GammaLogFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        if block_idx != Self::BLOCK_ETA {
            return Ok(None);
        }
        let eta = &expect_single_block(block_states, "GammaLogFamily")?.eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n || d_eta.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GammaLogFamily input size mismatch".to_string(),
            }
            .into());
        }
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err(GamlssError::NonFinite {
                reason: "GammaLogFamily shape must be finite and > 0".to_string(),
            }
            .into());
        }

        let mut dw = Array1::<f64>::zeros(n);
        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi <= 0.0 {
                return Err(GamlssError::InvalidInput {
                    reason: format!("GammaLogFamily requires positive finite y; found y[{i}]={yi}"),
                }
                .into());
            }
            let e_raw = eta[i];
            let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
            if self.weights[i] == 0.0 || e != e_raw {
                dw[i] = 0.0;
                continue;
            }
            let m = safe_exp(e).max(MIN_WEIGHT);
            let observed_weight = self.weights[i] * self.shape * yi / m;
            // d/dη [prior_weight * shape * y / exp(η)] = -W_obs.
            // If the positive floor is active, match the evaluated local piece.
            if observed_weight <= MIN_WEIGHT {
                dw[i] = 0.0;
            } else {
                dw[i] = -observed_weight * d_eta[i];
            }
        }
        Ok(Some(dw))
    }
}


impl CustomFamilyGenerative for GammaLogFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let eta = &expect_single_block(block_states, "GammaLogFamily")?.eta;
        let mean = gamlss_rowwise_map(eta.len(), |i| saturated_exp_eta(eta[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gamma { shape: self.shape },
        })
    }
}


/// Built-in binomial location-scale family with a configurable inverse link.
///
/// Parameters:
/// - Block 0: threshold/location T(covariates)
/// - Block 1: log-scale log σ(covariates)
#[derive(Clone)]
pub struct BinomialLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}


/// Both Binomial location-scale families plug into the unified
/// [`LocationScaleJointPsiFamily`] trait with byte-identical thin delegations
/// to inherent methods, differing only in the implementing type and its
/// `LABEL` fragment; generate them from one template. The Binomial families do
/// not thread the outer-row subsample (they run the full-data exact ψ path), so
/// the trait's `subsample` argument is accepted and ignored here.
macro_rules! impl_binomial_location_scale_joint_psi_family {
    ($family:ty, $label:literal) => {
        impl LocationScaleJointPsiFamily for $family {
            type Direction = LocationScaleJointPsiDirection;
            const LABEL: &'static str = $label;

            fn ws_policy(&self) -> &crate::resource::ResourcePolicy {
                &self.policy
            }

            fn ws_exact_joint_dense_block_designs<'a>(
                &'a self,
                specs: Option<&'a [ParameterBlockSpec]>,
            ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
                self.exact_joint_dense_block_designs(specs)
            }

            fn ws_psi_direction(
                &self,
                block_states: &[ParameterBlockState],
                derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
                psi_index: usize,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                policy: &crate::resource::ResourcePolicy,
            ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
                self.exact_newton_joint_psi_direction(
                    block_states,
                    derivative_blocks,
                    psi_index,
                    design_loc,
                    design_scale,
                    policy,
                )
            }

            fn ws_psi_second_order_terms_from_parts(
                &self,
                block_states: &[ParameterBlockState],
                derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
                psi_a: &LocationScaleJointPsiDirection,
                psi_b: &LocationScaleJointPsiDirection,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
            ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
                assert!(subsample.is_none());
                self.exact_newton_joint_psisecond_order_terms_from_parts(
                    block_states,
                    derivative_blocks,
                    psi_a,
                    psi_b,
                    design_loc,
                    design_scale,
                )
            }

            fn ws_psi_hessian_directional_from_parts(
                &self,
                block_states: &[ParameterBlockState],
                psi_dir: &LocationScaleJointPsiDirection,
                d_beta_flat: &Array1<f64>,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
            ) -> Result<Array2<f64>, String> {
                assert!(subsample.is_none());
                self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                    block_states,
                    psi_dir,
                    d_beta_flat,
                    design_loc,
                    design_scale,
                )
            }
        }
    };
}


impl_binomial_location_scale_joint_psi_family!(
    BinomialLocationScaleFamily,
    "BinomialLocationScaleFamily"
);

impl_binomial_location_scale_joint_psi_family!(
    BinomialLocationScaleWiggleFamily,
    "BinomialLocationScaleWiggleFamily"
);


type BinomialLocationScaleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<BinomialLocationScaleFamily>;

type BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<BinomialLocationScaleWiggleFamily>;
