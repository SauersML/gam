use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::*;

// ── RowKernel<2> implementation (rigid path only) ────────────────────

pub(super) struct BernoulliRigidRowKernel {
    pub(super) family: BernoulliMarginalSlopeFamily,
    pub(super) block_states: Vec<ParameterBlockState>,
    pub(super) slices: BlockSlices,
    /// Per-row uncontracted third-derivative tensor, lazily populated in a
    /// single parallel pass on first access. Every ψ-axis directional
    /// derivative operator that consults this kernel shares this cache via
    /// its `Arc`; the empirical-grid closed-form third-derivative tensor
    /// (`empirical_rigid_third_full_closed_form`) runs at most once per row
    /// across the full ext-dim sweep, instead of once per (row, ψ-axis) pair.
    /// Per-axis `row_third_contracted` becomes
    /// a 2×2 bilinear contraction against the cached tensor.
    pub(super) third_full_cache: crate::resource::RayonSafeOnce<Vec<[[[f64; 2]; 2]; 2]>>,
    /// Per-row uncontracted fourth-derivative tensor — the outer-Hessian
    /// analogue of `third_full_cache`. The second-directional-derivative
    /// operator's trace path touches every row × (u, v) pair; with this
    /// cache the heavy 8-direction empirical jet (or closed-form 5-component
    /// build) runs at most once per row, leaving each pair with a cheap
    /// [`contract_fourth_full`] bilinear.
    pub(super) fourth_full_cache: crate::resource::RayonSafeOnce<Vec<[[[[f64; 2]; 2]; 2]; 2]>>,
}

impl BernoulliRigidRowKernel {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Self {
        let slices = block_slices(&family);
        Self {
            family,
            block_states,
            slices,
            third_full_cache: crate::resource::RayonSafeOnce::new(),
            fourth_full_cache: crate::resource::RayonSafeOnce::new(),
        }
    }

    /// Lazy-build the per-row uncontracted third-derivative tensor cache. The
    /// first caller pays one parallel row pass that materialises the full
    /// `[[[f64; 2]; 2]; 2]` tensor for every observation; subsequent callers
    /// (every other ψ-axis operator that shares this kernel via `Arc`) get
    /// an `O(1)` lookup. A failed jet evaluation here means the underlying
    /// likelihood is non-finite at the converged β snapshot — propagate via
    /// panic, mirroring how every other kernel-level numerical contract in
    /// this module surfaces post-PIRLS invariant violations.
    pub(super) fn third_full_cache(&self) -> &[[[[f64; 2]; 2]; 2]] {
        self.third_full_cache
            .get_or_compute(|| {
                (0..self.family.y.len())
                    .into_par_iter()
                    .map(|row| {
                        let marginal_eta = self.block_states[0].eta[row];
                        let marginal = self.family.marginal_link_map(marginal_eta)?;
                        let slope = self.block_states[1].eta[row];
                        self.family.rigid_row_third_full(row, marginal, slope)
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .expect(
                        "BernoulliRigidRowKernel third-full cache build failed; \
                         per-row jet should not error at the converged β snapshot",
                    )
            })
            .as_slice()
    }

    /// Lazy-build the per-row uncontracted fourth-derivative tensor cache —
    /// outer-Hessian analogue of [`third_full_cache`]. Concurrent first
    /// callers may redundantly run the parallel row pass; the first published
    /// value wins and every subsequent caller reads the same vector. Used by
    /// `row_fourth_contracted` so each (u, v) ψ-axis pair finishes in a
    /// 16-multiply [`contract_fourth_full`] bilinear instead of triggering
    /// a fresh empirical-grid 8-direction jet.
    pub(super) fn fourth_full_cache(&self) -> &[[[[[f64; 2]; 2]; 2]; 2]] {
        self.fourth_full_cache
            .get_or_compute(|| {
                (0..self.family.y.len())
                    .into_par_iter()
                    .map(|row| {
                        let marginal_eta = self.block_states[0].eta[row];
                        let marginal = self.family.marginal_link_map(marginal_eta)?;
                        let slope = self.block_states[1].eta[row];
                        self.family.rigid_row_fourth_full(row, marginal, slope)
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .expect(
                        "BernoulliRigidRowKernel fourth-full cache build failed; \
                         per-row jet should not error at the converged β snapshot",
                    )
            })
            .as_slice()
    }
}

impl RowKernel<2> for BernoulliRigidRowKernel {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
        let marginal_eta = self.block_states[0].eta[row];
        let marginal = self.family.marginal_link_map(marginal_eta)?;
        let g = self.block_states[1].eta[row];
        self.family.rigid_row_kernel_eval(row, marginal, g)
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 2] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        [
            self.family
                .marginal_design
                .dot_row_view(row, d_beta.slice(s![self.slices.marginal.clone()])),
            self.family
                .logslope_design
                .dot_row_view(row, d_beta.slice(s![self.slices.logslope.clone()])),
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 2], out: &mut [f64]) {
        {
            let mut m = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0], &mut m)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut g = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[1], &mut g)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 2]; 2], target: &mut Array2<f64>) {
        self.family
            .marginal_design
            .syr_row_into_view(
                row,
                h[0][0],
                target.slice_mut(s![
                    self.slices.marginal.clone(),
                    self.slices.marginal.clone()
                ]),
            )
            .expect("marginal syr dim mismatch");
        if h[0][1] != 0.0 {
            self.family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &self.family.logslope_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.marginal.clone(),
                        self.slices.logslope.clone()
                    ]),
                )
                .expect("marginal-logslope outer dim mismatch");
            self.family
                .logslope_design
                .row_outer_into_view(
                    row,
                    &self.family.marginal_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.logslope.clone(),
                        self.slices.marginal.clone()
                    ]),
                )
                .expect("logslope-marginal outer dim mismatch");
        }
        self.family
            .logslope_design
            .syr_row_into_view(
                row,
                h[1][1],
                target.slice_mut(s![
                    self.slices.logslope.clone(),
                    self.slices.logslope.clone()
                ]),
            )
            .expect("logslope syr dim mismatch");
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 2]; 2], diag: &mut [f64]) {
        {
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, h[0][0], &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[1][1], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 2]) -> Result<[[f64; 2]; 2], String> {
        let cache = self.third_full_cache();
        Ok(contract_third_full(&cache[row], dir[0], dir[1]))
    }

    /// Force-build the per-row uncontracted third-derivative tensor cache
    /// at top-level rayon. Called by [`RowKernelHessianWorkspace::new`]
    /// before any outer `par_iter` enters; subsequent
    /// `row_third_contracted` calls inside the parallel ext-idx sweep then
    /// hit a populated cache and skip straight to a 2×2 contraction.
    fn warm_up_directional_caches(&self) -> Result<(), String> {
        // Touch both caches so their parallel builds run here, not later
        // (nested inside the outer ext-idx par_iter where the lock-holder
        // thread would have to do each row pass alone).
        let third_cache_len = self.third_full_cache().len();
        let fourth_cache_len = self.fourth_full_cache().len();
        let expected_len = self.family.y.len();
        if third_cache_len != expected_len || fourth_cache_len != expected_len {
            return Err(format!(
                "bernoulli rigid row-kernel cache warm-up length mismatch: third={third_cache_len} fourth={fourth_cache_len} expected={expected_len}"
            ));
        }
        Ok(())
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 2],
        dir_v: &[f64; 2],
    ) -> Result<[[f64; 2]; 2], String> {
        let cache = self.fourth_full_cache();
        Ok(contract_fourth_full(
            &cache[row],
            dir_u[0],
            dir_u[1],
            dir_v[0],
            dir_v[1],
        ))
    }

    /// BLAS-3 batched override of the generic per-row `J · F` build (see
    /// `RowKernel::jacobian_action_matrix` for the contract and the
    /// algebra).
    ///
    /// The bernoulli marginal-slope row Jacobian is a pure pair of
    /// design-row dot products against disjoint coefficient blocks:
    ///
    /// ```text
    ///   jacobian_action(r, β)[0] = marginal_design.row(r) · β[marg_range]
    ///   jacobian_action(r, β)[1] = logslope_design.row(r) · β[logs_range]
    /// ```
    ///
    /// So the full `(n × 2·rank)` projection is two dense matrix-matrix
    /// products, one per axis. For dense designs we dispatch through
    /// ndarray's `.dot(matrix)` which hits BLAS-3 (`matrixmultiply`)
    /// directly. For other backings we fall back to the generic per-
    /// row path by returning `None`; the operator-backed regime where
    /// the row kernel was deliberately matrix-free at large scale
    /// still pays the per-row jet costs we have today.
    ///
    /// **Correctness contract.** Output matches the per-row reference
    /// `jf[r, k * rank + c] = jacobian_action(r, F[:, c])[k]` exactly
    /// (it's the same arithmetic in a different order — BLAS-3
    /// summation reduces in-row).
    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        let p_total = self.slices.total;
        if factor.nrows() != p_total {
            return None;
        }
        let n_rows = self.family.y.len();
        let rank = factor.ncols();
        if rank == 0 {
            return Some(Array2::<f64>::zeros((n_rows, 2 * rank)));
        }

        // Slice F into the two coefficient-block factors. Standard-
        // layout owned copies let downstream `dot` paths stride
        // contiguous columns.
        let f_marg = factor
            .slice(s![self.slices.marginal.clone(), ..])
            .as_standard_layout()
            .into_owned();
        let f_logs = factor
            .slice(s![self.slices.logslope.clone(), ..])
            .as_standard_layout()
            .into_owned();

        // Compute J_block · F_block for both axes.
        //
        // Fast path: when the design has a materialized dense Array2
        // backing we hit ndarray's `dot(matrix)` directly, which routes
        // through `matrixmultiply` (BLAS-3) and reduces ~n×rank
        // strided per-row gathers to one cache-friendly contiguous
        // matmul per axis. At large-scale shape (n ≈ 1e4–1e5, rank ≈ 81)
        // this turns the dominant per-trace cost from ~3 s into ~50 ms.
        //
        // Generic path: for operator-backed / sparse / chunked designs
        // (Lazy with no contiguous `as_dense_ref`) we fall through to
        // `DesignMatrix::dot` on a per-column basis. This is still the
        // same arithmetic as the per-row reference (`jacobian_action`
        // does a single design-row dot product per call) but with
        // batched matrix-vector products that let the underlying
        // operator amortise any per-call dispatch cost. Importantly,
        // this path stays available for sparse-design fits where the
        // dense fast path is structurally inapplicable.
        let jf_marg = match self.family.marginal_design.as_dense_ref() {
            Some(dense) => dense.dot(&f_marg),
            None => self::axis_jf_via_column_dot(&self.family.marginal_design, &f_marg, n_rows),
        };
        let jf_logs = match self.family.logslope_design.as_dense_ref() {
            Some(dense) => dense.dot(&f_logs),
            None => self::axis_jf_via_column_dot(&self.family.logslope_design, &f_logs, n_rows),
        };

        assert_eq!(jf_marg.dim(), (n_rows, rank));
        assert_eq!(jf_logs.dim(), (n_rows, rank));

        // Pack into row-major (n × 2·rank): first `rank` columns are
        // k=0 (marginal axis), next `rank` are k=1 (logslope axis). This
        // mirrors the layout written by `compute_jf`'s strided write.
        let mut jf = Array2::<f64>::zeros((n_rows, 2 * rank));
        jf.slice_mut(s![.., 0..rank]).assign(&jf_marg);
        jf.slice_mut(s![.., rank..2 * rank]).assign(&jf_logs);
        Some(jf)
    }
}

/// Per-column matrix-vector dispatch for `DesignMatrix · F_block` when
/// no contiguous dense backing is available (sparse or operator-backed
/// designs). Mirrors what the per-row reference path does row-by-row,
/// but as `rank` batched mat-vec products so the underlying operator
/// can amortise per-call dispatch.
pub(crate) fn axis_jf_via_column_dot(
    design: &crate::linalg::matrix::DesignMatrix,
    f_block: &Array2<f64>,
    n_rows: usize,
) -> Array2<f64> {
    let rank = f_block.ncols();
    let mut out = Array2::<f64>::zeros((n_rows, rank));
    for c in 0..rank {
        let col_owned = f_block.column(c).to_owned();
        let result = design.dot(&col_owned);
        out.column_mut(c).assign(&result);
    }
    out
}

pub(super) struct BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    pub(super) family: BernoulliMarginalSlopeFamily,
    pub(super) block_states: Vec<ParameterBlockState>,
    pub(super) cache: Arc<BernoulliMarginalSlopeExactEvalCache>,
    pub(super) matvec_calls: AtomicUsize,
    pub(super) fused_gradient_dense:
        OnceLock<Result<Arc<ExactNewtonJointFusedDenseEvaluation>, String>>,
    /// Outer-only joint-Hessian directional-derivative options. The
    /// `outer_score_subsample` field is the row mask threaded through the
    /// `_with_options` directional-derivative helpers so the cached joint
    /// Hessian Hv-action paths can downscale to the stratified subsample at
    /// large scale. When `None`, the row iteration is identical to the
    /// legacy full-data path.
    pub(super) options: BlockwiseFitOptions,
}

pub(super) struct ExactNewtonJointFusedDenseEvaluation {
    pub(super) gradient: ExactNewtonJointGradientEvaluation,
    pub(super) hessian: Array2<f64>,
}

pub(super) struct BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    pub(super) family: BernoulliMarginalSlopeFamily,
    pub(super) block_states: Vec<ParameterBlockState>,
    pub(super) specs: Vec<ParameterBlockSpec>,
    pub(super) derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    pub(super) cache: Arc<BernoulliMarginalSlopeExactEvalCache>,
    /// Outer-only ψ-calculus options. The `outer_score_subsample` field is
    /// the row mask threaded through `sigma_exact_joint_psi_terms_with_options`
    /// and the second-order / Hessian-drift counterparts to make the cached
    /// ψ calculus subsample-aware.
    pub(super) options: BlockwiseFitOptions,
}

pub(super) fn bernoulli_margslope_line_search_ll_with_early_exit<F>(
    weighted_rows: &[WeightedOuterRow],
    threshold: f64,
    row_ll: F,
) -> Result<f64, String>
where
    F: Fn(usize) -> Result<f64, String> + Sync,
{
    if !threshold.is_finite() {
        return Err(format!(
            "bernoulli marginal-slope early-exit threshold must be finite, got {threshold}"
        ));
    }
    // Cross-path accumulation rounding band for the early-exit reject. The
    // running `-total_ll` here is summed chunk-by-chunk over a parallel
    // try_fold/try_reduce tree, whereas `threshold` (the current objective the
    // caller is trying to beat) is produced by a DIFFERENT accumulation order
    // — `log_likelihood_only_with_options` over the full row set. The two sums
    // are mathematically equal at the SAME β but, being computed in different
    // associativity orders over n≈3e5 rows, disagree by a handful of ULP. The
    // observed false reject was a 3e-11 gap at NLL≈1.5e5 — exactly 1 ULP at
    // that magnitude (ulp(1.5e5) = 2^(17-52) ≈ 2.9e-11). Summation over n rows
    // can accumulate a few ULP of order-dependent drift, so we admit a band of
    // a modest multiple of the per-value ULP scaled by |threshold|:
    // `EARLY_EXIT_REJECT_ROUNDING_ULPS × ε × max(|threshold|, 1)`. This stays a
    // valid reject certificate: it only DEFERS borderline trials whose partial
    // NLL exceeds the threshold by less than cross-path round-off to the full
    // exact LL return value plus the caller's objective/ρ accept test; it never
    // early-accepts a trial whose true full-data NLL is genuinely worse than
    // the threshold by more than this round-off band.
    const EARLY_EXIT_REJECT_ROUNDING_ULPS: f64 = 16.0;
    let early_exit_reject_tol =
        EARLY_EXIT_REJECT_ROUNDING_ULPS * f64::EPSILON * threshold.abs().max(1.0);
    let early_exit_reject_threshold = threshold + early_exit_reject_tol;
    let mut total_ll = 0.0;
    for chunk in weighted_rows.chunks(BERNOULLI_MARGSLOPE_LINE_SEARCH_EARLY_EXIT_CHUNK_ROWS) {
        let chunk_ll: f64 = chunk
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut acc, wr| -> Result<_, String> {
                    acc += wr.weight * row_ll(wr.index)?;
                    Ok(acc)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            )?;
        total_ll += chunk_ll;
        // Every Bernoulli marginal-slope row contribution is <= 0 because it is
        // weight_i * log(CDF(.)) with nonnegative weights, so the running sum is
        // monotone-down and `-total_ll` only grows as rows are added. When the
        // line search passes the full-data row set (the only caller — line-search
        // accept/reject is an exact full-data decision, see
        // `log_likelihood_only_with_options`), `-total_ll` is therefore a genuine
        // lower bound on the final full-data negative log-likelihood and can prove
        // the trial rejected before the sweep finishes. NOTE: this lower-bound
        // guarantee holds only for the full-data measure; a Horvitz-Thompson
        // subsample sum (inverse-inclusion weights) is an *unbiased estimator* of
        // the full-data NLL, not a lower bound, so it must never drive a reject
        // against a full-data threshold.
        if -total_ll > early_exit_reject_threshold {
            return Err(format!(
                "bernoulli marginal-slope line-search rejected early: partial_nll={} threshold={} (reject band={})",
                -total_ll, threshold, early_exit_reject_tol
            ));
        }
    }
    Ok(total_ll)
}

#[cfg(test)]
mod early_exit_soundness_tests {
    use super::*;

    pub(crate) fn full_data_rows(n: usize) -> Vec<WeightedOuterRow> {
        (0..n)
            .map(|index| WeightedOuterRow {
                index,
                weight: 1.0,
                stratum: 0,
            })
            .collect()
    }

    /// On the full-data measure (weight 1, every row) the running `-total_ll`
    /// is a genuine monotone lower bound on the full-data NLL, so the early
    /// exit is a valid reject certificate: it rejects iff the full NLL exceeds
    /// the threshold and otherwise returns the exact LL.
    #[test]
    pub(crate) fn full_data_early_exit_is_a_valid_reject_certificate() {
        // Each row contributes log Φ = -1.0, i.e. NLL contribution 1.0; full
        // NLL over 100 rows is exactly 100.
        let rows = full_data_rows(100);
        let row_ll = |_i: usize| -> Result<f64, String> { Ok(-1.0) };

        // Threshold below the full NLL → must reject.
        assert!(
            bernoulli_margslope_line_search_ll_with_early_exit(&rows, 50.0, row_ll).is_err(),
            "full-data NLL 100 > threshold 50 must reject"
        );

        // Threshold above the full NLL → must accept with the exact LL.
        let ll = bernoulli_margslope_line_search_ll_with_early_exit(&rows, 150.0, row_ll)
            .expect("full-data NLL 100 < threshold 150 must accept");
        assert!(
            (ll - (-100.0)).abs() < 1e-9,
            "accepted LL must be exact, got {ll}"
        );
    }

    /// Regression for the large-scale `IntegrationError`: a Horvitz-Thompson
    /// subsample sum is an *unbiased estimator* of the full-data NLL, not a
    /// lower bound on it, so feeding the kernel an HT-weighted subset against a
    /// full-data threshold can falsely reject a trial whose true full-data NLL
    /// is below the threshold. This is exactly why the BMS line search must
    /// only ever pass the full-data row set — the auto line-search subsample
    /// that used to violate this was removed.
    #[test]
    pub(crate) fn ht_subsample_against_full_data_threshold_can_falsely_reject() {
        // True per-row NLL: rows 0..10 contribute 10 each, rows 10..100
        // contribute 0. Full-data NLL = 100.
        let row_ll = |i: usize| -> Result<f64, String> { if i < 10 { Ok(-10.0) } else { Ok(0.0) } };
        let threshold = 500.0;

        // Full-data decision: NLL 100 < 500 → accept (the correct decision).
        let full_rows = full_data_rows(100);
        let full =
            bernoulli_margslope_line_search_ll_with_early_exit(&full_rows, threshold, row_ll)
                .expect("full-data NLL 100 < threshold 500 must accept");
        assert!((full - (-100.0)).abs() < 1e-9);

        // HT subsample that happens to draw the 10 high-NLL rows with
        // inverse-inclusion weight 10: weighted sum = 10·10·10 = 1000 > 500.
        // Fed against the full-data threshold the kernel rejects — a FALSE
        // reject. The product invariant is that this row set is never built
        // for a line-search probe; the assertion documents the hazard.
        let ht_rows: Vec<WeightedOuterRow> = (0..10)
            .map(|index| WeightedOuterRow {
                index,
                weight: 10.0,
                stratum: 1,
            })
            .collect();
        assert!(
            bernoulli_margslope_line_search_ll_with_early_exit(&ht_rows, threshold, row_ll)
                .is_err(),
            "HT-weighted sum 1000 spuriously exceeds the full-data threshold 500 — \
             demonstrates why an HT subsample must never certify a line-search reject"
        );
    }

    /// A numerically-flat trial whose full-data NLL exceeds the threshold by
    /// only cross-path accumulation round-off (≈1 ULP) must NOT be early-
    /// rejected: the early exit defers the borderline decision to the exact
    /// full LL return value and the caller's objective/ρ accept test. This is
    /// the biobank gauge-flat marginal/logslope hang — the line search rejected
    /// every trial by ~3e-11 at NLL≈1.5e5 (1 ULP) so the trust radius collapsed
    /// and the inner solve spun to its cap. A trial that is genuinely worse by
    /// more than the round-off band must still reject.
    #[test]
    pub(crate) fn flat_trial_within_rounding_band_is_not_early_rejected() {
        // Threshold at the biobank magnitude where the false reject was seen.
        let threshold = 155_598.382_868_126_53_f64;
        let n = 1000usize;
        let rows = full_data_rows(n);

        // Per-row NLL chosen so the full-data NLL sits exactly 1 ULP ABOVE the
        // threshold — a flat trial separated from the threshold only by
        // accumulation round-off.
        let one_ulp_above = threshold + (threshold * f64::EPSILON);
        let per_row_ll = -(one_ulp_above / n as f64);
        let row_ll = move |_i: usize| -> Result<f64, String> { Ok(per_row_ll) };
        let ll = bernoulli_margslope_line_search_ll_with_early_exit(&rows, threshold, row_ll)
            .expect(
                "a trial whose full NLL exceeds the threshold by ~1 ULP must not be \
                 early-rejected — the round-off band defers it to the exact LL return",
            );
        assert!(
            ll.is_finite() && (-ll) >= threshold - 1.0,
            "the returned LL must be the exact full-data sum, got {ll}"
        );

        // A trial whose NLL is clearly above the threshold (well beyond the
        // round-off band: +1.0 NLL units at this scale is ~3.4e10 ULP) must
        // still early-reject.
        let high_per_row_ll = -((threshold + 1.0) / n as f64);
        let high_row_ll = move |_i: usize| -> Result<f64, String> { Ok(high_per_row_ll) };
        assert!(
            bernoulli_margslope_line_search_ll_with_early_exit(&rows, threshold, high_row_ll)
                .is_err(),
            "a trial worse than the threshold by far more than the round-off band \
             must still early-reject"
        );
    }
}
