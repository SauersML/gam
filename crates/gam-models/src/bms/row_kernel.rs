use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::*;
use crate::fnv1a::Fnv1a;
use crate::row_kernel::RowSet;
use std::sync::{Mutex, OnceLock};

// ── Same-β rigid third/fourth-tensor cache ───────────────────────────
//
// The rigid coord_corrections (IFT Hessian-drift) path reads a per-row
// uncontracted third-derivative tensor over ALL n rows, and the outer-Hessian
// path reads the per-row fourth tensor likewise. A FRESH `BernoulliRigidRowKernel`
// is constructed on every outer eval (`exact_newton_joint_hessian_workspace*`),
// so at biobank scale (n≈3e5) the closed-form per-row jet would re-run over every
// row each eval — the dominant REML `coord_corrections` cost. The tensors are a
// pure function of the family/data identity and the coefficient state (block
// β + η), exactly like the same-β exact-cache (`SharedExactCacheStore`); mirror
// it with a FIFO-2 store so the immediate Value→ValueAndGradient pair at one β̂,
// and any line-search ρ that maps back to a seen β̂, share one table instead of
// rebuilding. Reuse is gated on exact byte-equality of a content fingerprint
// over the data, the frailty scale, the base link, the latent law's own nodes
// and weights, the deviation flags, and every block's β + η, so a hit returns an `Arc` to a table
// whose rows are bit-identical to a fresh build (or misses).
//
// The store holds per-row [`RigidRowTensors`] tables, not built vectors
// (gam#3022): a kernel's lookup-or-insert runs under the store lock, so every
// kernel at one β holds the same table, and each row of it is built once by its
// first reader while concurrent readers of that row wait.
type RigidThirdRows = RigidRowTensors<[[[f64; 2]; 2]; 2]>;
type RigidFourthRows = RigidRowTensors<[[[[f64; 2]; 2]; 2]; 2]>;

pub(super) struct SharedRigidTensorStore {
    third: Vec<(u64, Arc<RigidThirdRows>)>,
    fourth: Vec<(u64, Arc<RigidFourthRows>)>,
}

impl SharedRigidTensorStore {
    const CAPACITY: usize = 2;

    pub(super) fn empty() -> Self {
        Self {
            third: Vec::with_capacity(Self::CAPACITY),
            fourth: Vec::with_capacity(Self::CAPACITY),
        }
    }

    /// The table stored under `fp`, or a fresh unbuilt `n_rows` table stored
    /// under it (evicting the oldest entry at capacity).
    fn table<T>(
        slots: &mut Vec<(u64, Arc<RigidRowTensors<T>>)>,
        fp: u64,
        n_rows: usize,
    ) -> Arc<RigidRowTensors<T>> {
        if let Some((_, table)) = slots.iter().find(|(key, _)| *key == fp) {
            return Arc::clone(table);
        }
        if slots.len() >= Self::CAPACITY {
            slots.remove(0);
        }
        let table = Arc::new(RigidRowTensors::new(n_rows));
        slots.push((fp, Arc::clone(&table)));
        table
    }

    fn third_table(&mut self, fp: u64, n_rows: usize) -> Arc<RigidThirdRows> {
        Self::table(&mut self.third, fp, n_rows)
    }

    fn fourth_table(&mut self, fp: u64, n_rows: usize) -> Arc<RigidFourthRows> {
        Self::table(&mut self.fourth, fp, n_rows)
    }
}

fn shared_rigid_tensor_store() -> &'static Mutex<SharedRigidTensorStore> {
    static STORE: OnceLock<Mutex<SharedRigidTensorStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(SharedRigidTensorStore::empty()))
}

/// The rigid-tensor store `family` reuses from: its own search's in a parallel
/// multistart (gnomon#2359), else the process-wide one.
fn rigid_tensor_store(family: &BernoulliMarginalSlopeFamily) -> &Mutex<SharedRigidTensorStore> {
    match family.search.as_deref() {
        Some(member) => &member.rigid_tensors,
        None => shared_rigid_tensor_store(),
    }
}

// ── RowKernel<2> implementation (rigid path only) ────────────────────

pub(super) struct BernoulliRigidRowKernel {
    pub(super) family: BernoulliMarginalSlopeFamily,
    pub(super) block_states: Vec<ParameterBlockState>,
    pub(super) slices: BlockSlices,
    /// Per-row uncontracted third-derivative tensors, shared through
    /// [`rigid_tensor_store`] by every kernel at this β and built lazily one row
    /// at a time. Every ψ-axis directional derivative operator that consults
    /// this kernel reads it, so the empirical-grid closed-form third tensor
    /// (`empirical_rigid_third_full_closed_form`) runs at most once per row
    /// across the full ext-dim sweep, instead of once per (row, ψ-axis) pair;
    /// per-axis `row_third_contracted` is a 2×2 bilinear contraction of it.
    pub(super) third_rows: gam_runtime::resource::RayonSafeOnce<Arc<RigidThirdRows>>,
    /// Per-row uncontracted fourth-derivative tensors — the outer-Hessian
    /// analogue of `third_rows`. The second-directional-derivative operator's
    /// trace path touches every row × (u, v) pair; with this table the heavy
    /// 8-direction empirical jet (or closed-form 5-component build) runs at
    /// most once per row, leaving each pair with a cheap
    /// [`contract_fourth_full`] bilinear.
    pub(super) fourth_rows: gam_runtime::resource::RayonSafeOnce<Arc<RigidFourthRows>>,
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
            third_rows: gam_runtime::resource::RayonSafeOnce::new(),
            fourth_rows: gam_runtime::resource::RayonSafeOnce::new(),
        }
    }

    /// The row's uncontracted third tensor: the empirical-grid closed form, or
    /// the inherited program for the standard-normal measure (see
    /// [`RowKernel::row_kernel`] on this kernel for why the empirical rows skip
    /// the program).
    pub(super) fn row_third_full(&self, row: usize) -> Result<[[[f64; 2]; 2]; 2], String> {
        match self.family.training_row_grid(row)? {
            None => gam_math::jet_tower::program_full_tower(self, row).map(|tower| tower.t3),
            Some(grid) => self.family.empirical_rigid_third_full_closed_form(
                row,
                self.family
                    .marginal_link_map(self.block_states[0].eta[row])?,
                self.block_states[1].eta[row],
                &grid.nodes,
                &grid.weights,
            ),
        }
    }

    /// The row's uncontracted fourth tensor, as [`Self::row_third_full`].
    pub(super) fn row_fourth_full(&self, row: usize) -> Result<[[[[f64; 2]; 2]; 2]; 2], String> {
        match self.family.training_row_grid(row)? {
            None => gam_math::jet_tower::program_full_tower(self, row).map(|tower| tower.t4),
            Some(grid) => self.family.empirical_rigid_fourth_full_closed_form(
                row,
                self.family
                    .marginal_link_map(self.block_states[0].eta[row])?,
                self.block_states[1].eta[row],
                &grid.nodes,
                &grid.weights,
            ),
        }
    }

    /// Content fingerprint of every input the per-row rigid third/fourth jet
    /// reads: the data, frailty scale, base link and latent law
    /// (`BernoulliMarginalSlopeFamily::mix_data_and_law`), the score-warp /
    /// link-deviation presence flags, and every block's β + η. `rigid_row_third_full`/`rigid_row_fourth_full` are pure
    /// functions of exactly these (the per-row build reads `block_states[*].eta`,
    /// `self.z[row]`/`y[row]`/`weights[row]`, the frailty scale, and the latent
    /// grid), so equal fingerprints ⇒
    /// bit-identical tensors. The `domain` byte separates the third- and
    /// fourth-tensor key streams. Mirrors
    /// `BernoulliMarginalSlopeFamily::shared_exact_cache_fingerprint`.
    fn rigid_tensor_fingerprint(&self, domain: u8) -> u64 {
        let mut hash = Fnv1a::new();
        hash.mix_byte(domain);
        self.family.mix_data_and_law(&mut hash);
        hash.mix_byte(0xf2);
        hash.mix_byte(u8::from(self.family.score_warp.is_some()));
        hash.mix_byte(u8::from(self.family.link_dev.is_some()));
        hash.mix_byte(0xf3);
        for b in (self.block_states.len() as u64).to_le_bytes() {
            hash.mix_byte(b);
        }
        for state in &self.block_states {
            for b in (state.beta.len() as u64).to_le_bytes() {
                hash.mix_byte(b);
            }
            for &v in state.beta.iter() {
                hash.mix_f64(v);
            }
            for b in (state.eta.len() as u64).to_le_bytes() {
                hash.mix_byte(b);
            }
            for &v in state.eta.iter() {
                hash.mix_f64(v);
            }
        }
        hash.finish_nonzero()
    }

    /// This β's third-tensor table. Concurrent first callers each take the
    /// store lock and receive the same table, so no row is built twice.
    pub(super) fn third_rows(&self) -> &RigidThirdRows {
        self.third_rows.get_or_compute(|| {
            let fp = self.rigid_tensor_fingerprint(0xa3);
            rigid_tensor_store(&self.family)
                .lock()
                .expect("BMS rigid tensor store mutex poisoned on third read")
                .third_table(fp, self.family.y.len())
        })
    }

    /// This β's fourth-tensor table, as [`Self::third_rows`].
    pub(super) fn fourth_rows(&self) -> &RigidFourthRows {
        self.fourth_rows.get_or_compute(|| {
            let fp = self.rigid_tensor_fingerprint(0xa4);
            rigid_tensor_store(&self.family)
                .lock()
                .expect("BMS rigid tensor store mutex poisoned on fourth read")
                .fourth_table(fp, self.family.y.len())
        })
    }

    /// Row `row`'s third tensor, built on its first read.
    fn third_full(&self, row: usize) -> Result<&[[[f64; 2]; 2]; 2], String> {
        self.third_rows().row(row, || self.row_third_full(row))
    }

    /// Row `row`'s fourth tensor, built on its first read.
    fn fourth_full(&self, row: usize) -> Result<&[[[[f64; 2]; 2]; 2]; 2], String> {
        self.fourth_rows().row(row, || self.row_fourth_full(row))
    }

    /// Every row's third tensor, the unbuilt rows built in one parallel pass.
    pub(super) fn all_third_full(&self) -> Result<Vec<&[[[f64; 2]; 2]; 2]>, String> {
        self.third_rows().all_rows(|row| self.row_third_full(row))
    }

    /// Every row's fourth tensor, the unbuilt rows built in one parallel pass.
    pub(super) fn all_fourth_full(&self) -> Result<Vec<&[[[[f64; 2]; 2]; 2]; 2]>, String> {
        self.fourth_rows().all_rows(|row| self.row_fourth_full(row))
    }
}

impl gam_math::jet_tower::RowProgram<2> for BernoulliRigidRowKernel {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        if row >= self.family.y.len() {
            return Err(format!("BernoulliRigidRowKernel: row {row} out of range"));
        }
        Ok([self.block_states[0].eta[row], self.block_states[1].eta[row]])
    }

    fn eval<S: gam_math::jet_scalar::JetScalar<2>>(
        &self,
        row: usize,
        p: &[S; 2],
    ) -> Result<S, String> {
        if row >= self.family.y.len() {
            return Err(format!("BernoulliRigidRowKernel: row {row} out of range"));
        }
        let marginal = self
            .family
            .marginal_link_map(self.block_states[0].eta[row])?;
        let slope = self.block_states[1].eta[row];
        match self.family.training_row_grid(row)? {
            None => rigid_standard_normal_row_nll_generic(
                p,
                marginal,
                self.family.z[row],
                self.family.y[row],
                self.family.weights[row],
                self.family.probit_frailty_scale(),
            ),
            Some(grid) => {
                let plan = self
                    .family
                    .compile_empirical_rigid_bms_row_program(row, marginal, slope, &grid)?;
                let vars = [
                    gam_math::jet_scalar::FixedRuntimeJet::from_inner(p[0]),
                    gam_math::jet_scalar::FixedRuntimeJet::from_inner(p[1]),
                ];
                plan.evaluate(&vars, 4, &())
                    .map(gam_math::jet_scalar::FixedRuntimeJet::into_inner)
            }
        }
    }
}

impl RowKernel<2> for BernoulliRigidRowKernel {
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    /// Empirical-grid rows read value, score and curvature off the closed form
    /// every rigid evaluation shares (`empirical_rigid_primary_grad_hess_closed_form`,
    /// pinned by `empirical_rigid_jet_oracle_tests` and against the inherited
    /// program by `rigid_row_kernel_closed_form_tests`). The inherited program
    /// compiles the FLEX row program for the row on every call: a second Newton
    /// polish of the intercept root and one index program per grid node, only to
    /// read its order-two channels. That compile was half the CPU of a
    /// production-shape fit, paid once per row per joint-Newton cycle.
    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
        if row >= self.family.y.len() {
            return Err(format!("BernoulliRigidRowKernel: row {row} out of range"));
        }
        match self.family.training_row_grid(row)? {
            None => gam_math::jet_tower::program_row_kernel(self, row),
            Some(grid) => {
                let marginal = self
                    .family
                    .marginal_link_map(self.block_states[0].eta[row])?;
                self.family.empirical_rigid_primary_grad_hess_closed_form(
                    row,
                    marginal,
                    self.block_states[1].eta[row],
                    &grid.nodes,
                    &grid.weights,
                )
            }
        }
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 2] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        [
            self.family
                .marginal_design
                .dot_row_view(row, d_beta.slice(s![self.slices.marginal.clone()])),
            self.family
                .slope_design
                .dot_row_view(row, d_beta.slice(s![self.slices.slope.clone()])),
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
            let mut g = ndarray::ArrayViewMut1::from(&mut out[self.slices.slope.clone()]);
            self.family
                .slope_design
                .axpy_row_into(row, v[1], &mut g)
                .expect("slope axpy dim mismatch");
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
                    &self.family.slope_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.marginal.clone(),
                        self.slices.slope.clone()
                    ]),
                )
                .expect("marginal-slope outer dim mismatch");
            self.family
                .slope_design
                .row_outer_into_view(
                    row,
                    &self.family.marginal_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.slope.clone(),
                        self.slices.marginal.clone()
                    ]),
                )
                .expect("slope-marginal outer dim mismatch");
        }
        self.family
            .slope_design
            .syr_row_into_view(
                row,
                h[1][1],
                target.slice_mut(s![
                    self.slices.slope.clone(),
                    self.slices.slope.clone()
                ]),
            )
            .expect("slope syr dim mismatch");
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
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.slope.clone()]);
            self.family
                .slope_design
                .squared_axpy_row_into(row, h[1][1], &mut gd)
                .expect("slope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 2]) -> Result<[[f64; 2]; 2], String> {
        Ok(contract_third_full(self.third_full(row)?, dir[0], dir[1]))
    }

    /// Build the per-row tensors the eval about to run reads, in one parallel
    /// row pass each, before the outer ext-idx `par_iter` enters. Every
    /// ext-idx task then sweeps the rows in the same order; without this pass
    /// the cold rows would build one at a time behind whichever task reaches
    /// each first, the others waiting on it.
    fn warm_up_directional_caches(&self, eval_mode: EvalMode) -> Result<(), String> {
        // gam#979: prime only the tables the eval about to run will consume.
        //
        //   * `ValueOnly`            → neither table (the objective is read off
        //                              the converged inner mode; no directional
        //                              contraction is taken). Seed screening,
        //                              line-search cost probes, and typed reactive
        //                              `ContinuationPath` waypoints are all
        //                              value-only — at biobank scale each would
        //                              otherwise pay two full `O(n)` jet passes
        //                              (third + fourth) for tensors it never reads.
        //   * `ValueAndGradient`     → third-derivative table only. The REML/LAML
        //                              gradient's `coord_corrections` IFT-drift
        //                              trace is a *first* directional derivative
        //                              (`row_third_contracted`); the BFGS
        //                              first-order bridge never asks for the
        //                              outer Hessian, so the fourth table stays
        //                              cold for the whole fit.
        //   * `ValueGradientHessian` → both tables; the outer Hessian's second-
        //                              directional pass reads `row_fourth_contracted`.
        //
        // Under-priming is safe: every row is built on its first read.
        match eval_mode {
            EvalMode::ValueOnly => Ok(()),
            EvalMode::ValueAndGradient => self.all_third_full().map(drop),
            EvalMode::ValueGradientHessian => {
                self.all_third_full()?;
                self.all_fourth_full().map(drop)
            }
        }
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 2],
        dir_v: &[f64; 2],
    ) -> Result<[[f64; 2]; 2], String> {
        Ok(contract_fourth_full(
            self.fourth_full(row)?,
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
    ///   jacobian_action(r, β)[1] = slope_design.row(r) · β[logs_range]
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

        // Slice F into the two coefficient-block factors. Standard-
        // layout owned copies let downstream `dot` paths stride
        // contiguous columns.
        let f_marg = factor
            .slice(s![self.slices.marginal.clone(), ..])
            .as_standard_layout()
            .into_owned();
        let f_logs = factor
            .slice(s![self.slices.slope.clone(), ..])
            .as_standard_layout()
            .into_owned();

        // Compute J_block · F_block for both axes. Dense designs use BLAS-3;
        // operator-backed/sparse designs use the shared per-column dispatcher,
        // preserving the same arithmetic as the per-row reference.
        let jf_marg = crate::row_kernel::row_kernel_design_jf(
            &self.family.marginal_design,
            f_marg.view(),
            n_rows,
        );
        let jf_logs = crate::row_kernel::row_kernel_design_jf(
            &self.family.slope_design,
            f_logs.view(),
            n_rows,
        );

        Some(crate::row_kernel::row_kernel_pack_jf_axes::<2>(
            n_rows,
            rank,
            [(0, jf_marg), (1, jf_logs)],
        ))
    }

    fn jacobian_action_matrix_rows(
        &self,
        factor: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        let p_total = self.slices.total;
        if factor.nrows() != p_total {
            return crate::row_kernel::row_kernel_jacobian_action_matrix_generic_rows(
                self, factor, start, end,
            );
        }
        let b = end.saturating_sub(start);
        let rank = factor.ncols();
        let f_marg = factor
            .slice(s![self.slices.marginal.clone(), ..])
            .as_standard_layout()
            .into_owned();
        let f_logs = factor
            .slice(s![self.slices.slope.clone(), ..])
            .as_standard_layout()
            .into_owned();
        let jf_marg = crate::row_kernel::row_kernel_design_jf_rows(
            &self.family.marginal_design,
            f_marg.view(),
            start,
            end,
        );
        let jf_logs = crate::row_kernel::row_kernel_design_jf_rows(
            &self.family.slope_design,
            f_logs.view(),
            start,
            end,
        );

        crate::row_kernel::row_kernel_pack_jf_axes::<2>(b, rank, [(0, jf_marg), (1, jf_logs)])
    }

    /// BLAS-3 override of the first directional derivative of the dense joint
    /// Hessian for the rigid marginal-slope kernel (see the trait default for
    /// the cost argument). The rigid row pullback is a pure pair of design-row
    /// Grams — `target += h[0][0]·xxᵀ + h[0][1]·(xgᵀ + gxᵀ) + h[1][1]·ggᵀ` —
    /// with no h/w cross blocks, so `∂H/∂β[d_beta]` is exactly
    ///
    /// ```text
    ///   H_drift = Σ_row Xᵣᵀ · contract_third_full(T³ᵣ, dq_r, dg_r) · Xᵣ,
    ///   dq_r = marginal_design.row(r)·d_beta[marg],  dg_r = slope.row(r)·d_beta[logs].
    /// ```
    ///
    /// We accumulate the per-row `2×2` contraction weights `(w_mm, w_mg, w_gg)`
    /// over a contiguous row chunk, project `(dq, dg)` for the whole chunk in
    /// two GEMMs, and close each chunk with one pair of
    /// `Xᵀ diag(w) X` / `Xᵀ diag(w) G` products
    /// (`add_weighted_design_grams_from_chunks`). The per-row third tensor is
    /// read from the shared `third_rows` table (each row built once per β), so
    /// the `k` Jeffreys columns pay the closed-form third build at most once
    /// per row. Bit-for-bit the same entries the per-row `add_pullback_hessian`
    /// scatter writes (`w_mm = t[0][0]`, `w_mg = t[0][1]`, `w_gg = t[1][1]`),
    /// reduced in a different summation order.
    ///
    /// Handles every `RowSet`: each walked row's contraction weights carry its
    /// Horvitz-Thompson weight. Declines (`None`) only a sparse design block.
    fn directional_derivative_dense_override(
        &self,
        rows: &RowSet,
        d_beta: &[f64],
    ) -> Option<Result<Array2<f64>, String>> {
        // The chunked `Xᵀ diag(w) X` Gram slices contiguous design rows via
        // `try_row_chunk` inside `directional_derivative_dense_blas3` (which
        // already handles operator-backed / residualised designs row-chunk by
        // row-chunk), so the only structurally-inapplicable case is a sparse
        // design block — gate on that, not on the presence of a pre-materialised
        // `as_dense_ref`. Without this, a biobank rigid fit whose marginal /
        // slope design is operator-backed (residualised absorber, overlap-Z)
        // fell through to the generic per-row third-tensor scatter — the ~8s
        // per-cycle `gradient_reload` / Jeffreys-column floor.
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let slope_sparse = self.family.slope_design.is_sparse();
        if marginal_sparse || slope_sparse {
            return None;
        }
        Some(self.directional_derivative_dense_blas3(rows, d_beta))
    }

    /// BLAS-3 override of the BATCHED all-axes FIRST directional derivative of
    /// the dense joint Hessian for the rigid marginal-slope kernel. This is the
    /// per-cycle hotspot of the inner-Newton Jeffreys/Firth term (gam#979): the
    /// generic per-axis path asks the family for `Hdot[e_a]` `p` separate times,
    /// and the coupled BMS family reconstructs a fresh `BernoulliRigidRowKernel`
    /// each call, rebuilding the `O(n)` per-row third-tensor cache `p` times on
    /// every cycle the conditioning gate arms.
    ///
    /// The rigid row pullback is a pure pair of design-row Grams, so the per-row
    /// third tensor `T³ᵣ` is built ONCE (cached `third_full`) and the axis
    /// projection enters LINEARLY: a marginal axis `j` has primary projection
    /// `(vq, vg) = (X[r,j], 0)` and `contract_third_full(T³ᵣ, X[r,j], 0) =
    /// X[r,j] · contract_third_full(T³ᵣ, 1, 0)`; a slope axis has
    /// `(0, G[r,j])`. So we read each row's `A_r = contract_third_full(T³ᵣ, 1, 0)`
    /// and `B_r = contract_third_full(T³ᵣ, 0, 1)` once and close every axis with
    /// the same chunked `Xᵀ diag(w) X` / `Xᵀ diag(w) G` BLAS-3 machinery the
    /// first-directional override uses. Bit-for-bit the same entries the per-row
    /// `add_pullback_hessian` scatter writes, reduced in BLAS-3 in-row order, so
    /// axis `a` matches `row_kernel_directional_derivative(self, rows, e_a)`.
    ///
    /// Handles every `RowSet`, Horvitz-Thompson weights included. Declines
    /// (`None`) only a sparse design block.
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &RowSet,
        p: usize,
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        // The dispatcher passes `p = n_coefficients()`; a mismatch is a hard
        // caller-contract violation, surfaced as a non-sentinel error so `p` is
        // consumed on every path without masking a bad call.
        if p != self.n_coefficients() {
            return Some(Err(format!(
                "bms directional_derivative_all_axes_dense_override: axis count {} \
                 disagrees with n_coefficients() {}",
                p,
                self.n_coefficients(),
            )));
        }
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let slope_sparse = self.family.slope_design.is_sparse();
        if marginal_sparse || slope_sparse {
            return None;
        }
        Some(self.directional_derivative_all_axes_blas3(rows))
    }

    /// BLAS-3 override of the dense joint-Hessian assembly for the rigid
    /// marginal-slope kernel (see the trait default for the cost argument).
    /// The post-gradient-reload Jeffreys/Firth residual term first materializes
    /// the observed joint Hessian via this path; the generic per-row
    /// `add_pullback_hessian` scatter is `n·p²` BLAS-1. Identical pure
    /// design-row Gram structure as the directional-derivative override and the
    /// fused dense-H build: gather the per-row contraction weights
    /// (`w_mm = h[0][0]`, `w_mg = h[0][1]`, `w_gg = h[1][1]`) from the cached
    /// `K×K` row Hessians and close each chunk with `Xᵀ diag(w) X` /
    /// `Xᵀ diag(w) G`. Bit-for-bit the same entries the scatter writes, reduced
    /// in BLAS-3 in-row order. Handles every `RowSet`, Horvitz-Thompson
    /// weights included; declines (`None`) only a sparse design block.
    fn hessian_dense_override(
        &self,
        rows: &RowSet,
        row_hessians: &[[[f64; 2]; 2]],
    ) -> Option<Result<Array2<f64>, String>> {
        if row_hessians.len() != self.family.y.len() {
            return Some(Err(format!(
                "BMS rigid hessian_dense_override row-Hessian length mismatch: got {}, expected {}",
                row_hessians.len(),
                self.family.y.len()
            )));
        }
        // The chunked `Xᵀ diag(w) X` build slices contiguous design rows via
        // `try_row_chunk`, which every dense-backed design (materialised OR
        // operator-backed / residualised) supports — so the BLAS-3 path fires
        // for the biobank rigid fit regardless of whether the marginal/slope
        // designs expose a pre-materialised `as_dense_ref`. Sparse designs are
        // the only structurally-inapplicable case; route those to the generic
        // per-row scatter so the design-row Gram never densifies a sparse block.
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let slope_sparse = self.family.slope_design.is_sparse();
        if marginal_sparse || slope_sparse {
            // Diagnostic fires once per process, not once per inner-Newton kernel
            // call: `hessian_dense_override` runs on every joint-Hessian assembly,
            // so an unguarded line floods the biobank fit log.
            static H_NOT_TAKEN_LOGGED: std::sync::Once = std::sync::Once::new();
            H_NOT_TAKEN_LOGGED.call_once(|| {
                log::debug!(
                    "[STAGE] BMS rigid hessian_dense BLAS-3 path NOT taken: sparse design \
                     (marginal_sparse={marginal_sparse} slope_sparse={slope_sparse}) \
                     -> generic per-row scatter"
                );
            });
            return None;
        }
        // Route an eligible whole-design joint Gram through one CUDA dispatch.
        // The device Gram runs over every design row with unit weight, so only
        // `RowSet::All` is eligible. `Ok(None)` means CUDA was declined before
        // execution (non-materialized design, no runtime, or below policy);
        // after admission, a missing device result is an error and cannot
        // select the CPU algorithm.
        if let RowSet::All = rows {
            match rigid_joint_hessian_on_gpu(
                &self.family.marginal_design,
                &self.family.slope_design,
                row_hessians,
            ) {
                Ok(Some(joint)) => return Some(Ok(joint)),
                Ok(None) => {}
                Err(error) => return Some(Err(error)),
            }
        }
        Some(self.hessian_dense_blas3(rows, row_hessians))
    }

    /// BLAS-3 override of the BATCHED all-axes second directional derivative of
    /// the dense joint Hessian for the rigid marginal-slope kernel (see the
    /// trait default for the cost argument). This is the dominant cost of the
    /// outer-REML Jeffreys `H_Φ` drift (`coord_corrections`): the generic
    /// per-axis path runs `p` full-data sweeps each scattering the `2×2`
    /// contracted fourth tensor through `add_pullback_hessian` — `O(p · n · p²)`
    /// BLAS-1 scatter at biobank scale (`k≈8` drift columns × the inner sweep).
    ///
    /// The rigid row pullback is a pure pair of design-row Grams with no h/w
    /// cross blocks, so for the fixed direction `u` with primary projections
    /// `(uq_r, ug_r) = (X·u_marg, G·u_logs)` per row, the all-axes object is
    ///
    /// ```text
    ///   H²dot[u, e_a] = Σ_r Xᵣᵀ · contract_fourth_full(T⁴ᵣ, uq_r, ug_r, vq_r, vg_r) · Xᵣ
    /// ```
    ///
    /// where `(vq_r, vg_r) = Jᵣ·e_a` is the swept axis projection. The `2×2`
    /// weight matrix is LINEAR in the axis projection, and the fourth tensor's
    /// `u`-side partial contractions
    ///
    /// ```text
    ///   A_r[a][b] = Σ_c T⁴ᵣ[a][b][c][0]·u[c]   (close the last index on the η-unit)
    ///   B_r[a][b] = Σ_c T⁴ᵣ[a][b][c][1]·u[c]   (close the last index on the g-unit)
    /// ```
    ///
    /// are INDEPENDENT of the swept axis. A marginal-block axis `j` has
    /// `(vq_r, vg_r) = (X[r,j], 0)` so its row weight is `X[r,j]·A_r`; a
    /// slope-block axis `j` has `(0, G[r,j])` so its row weight is
    /// `G[r,j]·B_r`. Thus we read the cached fourth tensor and build `A_r, B_r`
    /// ONCE per row (hoisted out of the `p`-loop, the `~p×` reduction), then
    /// close each axis with the same chunked `Xᵀ diag(w) X` / `Xᵀ diag(w) G`
    /// BLAS-3 machinery the first-directional override uses
    /// (`add_weighted_design_grams_from_chunks`). Bit-for-bit the same entries
    /// the per-row `add_pullback_hessian` scatter writes, reduced in BLAS-3
    /// in-row order.
    ///
    /// Handles every `RowSet`, Horvitz-Thompson weights included. Declines
    /// (`None`) only a sparse design block.
    fn second_directional_derivative_all_axes_dense_override(
        &self,
        rows: &RowSet,
        d_beta_u: &[f64],
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if d_beta_u.len() != self.slices.total {
            return Some(Err(format!(
                "bms second_directional_derivative_all_axes_dense_override: fixed direction has \
                 {} entries, expected {}",
                d_beta_u.len(),
                self.slices.total,
            )));
        }
        // Same structural gate as the first-directional override: the chunked
        // Gram machinery slices contiguous design rows via `try_row_chunk`,
        // which every dense-backed design (materialised OR operator-backed)
        // supports; only a sparse design block is structurally inapplicable.
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let slope_sparse = self.family.slope_design.is_sparse();
        if marginal_sparse || slope_sparse {
            return None;
        }
        Some(self.second_directional_derivative_all_axes_blas3(rows, d_beta_u))
    }
}

/// Row-block size for the parallel `Xᵀdiag(w)X` Gram reduction.
///
/// The Gram assembly fans contiguous row blocks across the Rayon pool and pins
/// each block's faer GEMM to `Par::Seq` (see the `with_nested_parallel` guard in
/// the chunk bodies). For that fan-out to fill the pool without leaving cores
/// idle on the tail block, the chunk COUNT must comfortably exceed the worker
/// count; for each chunk's GEMM to stay an efficient BLAS-3 tile (not setup-
/// bound), the chunk must stay reasonably tall. We therefore target roughly
/// `OVERSUBSCRIBE × workers` chunks and clamp the per-chunk row span to a band
/// that keeps the `(rows × p)` weighted-design tile a healthy GEMM without
/// blowing the `stream_weighted_crossprod_into` working set. At the biobank
/// rigid scale (n ≈ 1.9e5, ~52 workers) this lands ~208 chunks of ~3.7k rows —
/// full pool occupancy with load-balancing headroom — versus the prior fixed
/// 8 192-row split that produced only ~24 chunks (under half a 52-core pool,
/// with a lopsided tail).
fn blas3_gram_chunk_rows(n: usize) -> usize {
    const OVERSUBSCRIBE: usize = 4;
    const MIN_CHUNK_ROWS: usize = 2_048;
    const MAX_CHUNK_ROWS: usize = 16_384;
    // Reproducibility contract (#1045): size the Gram chunk boundaries to the
    // process-stable machine parallelism, not the live scoped-pool worker count,
    // so the per-chunk `Xᵀdiag(w)X` partials — and the tree that reduces them —
    // do not regroup when the executing rayon pool is narrowed/widened.
    let workers = crate::marginal_slope_shared::reproducible_chunk_parallelism();
    let target_chunks = (workers * OVERSUBSCRIBE).max(1);
    let by_target = n.div_ceil(target_chunks);
    by_target.clamp(MIN_CHUNK_ROWS, MAX_CHUNK_ROWS).max(1)
}

/// Gram chunks `(start, end)` over the walk positions `0..rows.walk_len(n)`,
/// each [`blas3_gram_chunk_rows`] positions tall (the last one shorter).
fn row_set_gram_chunks(rows: &RowSet, n: usize) -> Vec<(usize, usize)> {
    let m = rows.walk_len(n);
    let chunk_rows = blas3_gram_chunk_rows(m);
    (0..m).step_by(chunk_rows).map(|start| (start, (start + chunk_rows).min(m))).collect()
}

/// Design rows at walk positions `start..end` of `rows`. Under `RowSet::All`
/// this is the contiguous full-data block: a zero-copy slice of a materialised
/// design, one `try_row_chunk` of an operator-backed one. Under a subsample it
/// is the stored rows gathered in walk order. A failed materialisation is a
/// hard error: the design row buffer is fixed for the whole fit.
fn row_set_design_rows<'a>(
    design: &'a gam_linalg::matrix::DesignMatrix,
    rows: &RowSet,
    (start, end): (usize, usize),
    what: &str,
) -> Result<ndarray::CowArray<'a, f64, ndarray::Ix2>, String> {
    match rows {
        RowSet::All => match design.as_dense_ref() {
            Some(full) => Ok(full.slice(s![start..end, ..]).into()),
            None => design
                .try_row_chunk(start..end)
                .map(Into::into)
                .map_err(|e| format!("{what} try_row_chunk({start}..{end}): {e}")),
        },
        RowSet::Subsample { rows: stored, .. } => {
            let indices = stored[start..end].iter().map(|row| row.index).collect::<Vec<_>>();
            design
                .try_row_gather(&indices)
                .map(Into::into)
                .map_err(|e| format!("{what} try_row_gather over positions {start}..{end}: {e}"))
        }
    }
}

/// Whole-design GPU dispatch for the rigid `Xᵀ diag(w) X` joint Hessian.
///
/// `Ok(None)` is reserved for structural or policy decisions made before CUDA
/// execution: either design is not materialized dense, no runtime exists, or
/// the workload is below the device floor. Once `route_through_gpu` admits the
/// operation, a missing result from the current optional `gam-gpu` API is a
/// contextual `Err`; it never selects the CPU algorithm.
#[inline]
fn rigid_joint_hessian_on_gpu(
    marginal_design: &gam_linalg::matrix::DesignMatrix,
    slope_design: &gam_linalg::matrix::DesignMatrix,
    row_hessians: &[[[f64; 2]; 2]],
) -> Result<Option<Array2<f64>>, String> {
    let Some(x_full) = marginal_design.as_dense_ref() else {
        return Ok(None);
    };
    let Some(g_full) = slope_design.as_dense_ref() else {
        return Ok(None);
    };
    let rows = x_full.nrows();
    let slope_rows = g_full.nrows();
    if rows != slope_rows || rows != row_hessians.len() {
        return Err(format!(
            "BMS rigid joint-Hessian dimensions disagree: marginal_rows={rows}, \
             slope_rows={slope_rows}, row_hessians={}",
            row_hessians.len()
        ));
    }

    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
    #[cfg(target_os = "linux")]
    {
        let marginal_cols = x_full.ncols();
        let slope_cols = g_full.ncols();
        let operation = gam_gpu::linalg_dispatch::DispatchOp::JointHessian2x2 {
            n: rows,
            pa: marginal_cols,
            pb: slope_cols,
        };
        if gam_gpu::linalg_dispatch::route_through_gpu(operation).is_none() {
            return Ok(None);
        }

        let w_mm: Array1<f64> = row_hessians.iter().map(|h| h[0][0]).collect();
        let w_mg: Array1<f64> = row_hessians.iter().map(|h| h[0][1]).collect();
        let w_gg: Array1<f64> = row_hessians.iter().map(|h| h[1][1]).collect();
        require_selected_cuda_gram_result(
            "rigid joint Hessian",
            rows,
            marginal_cols,
            slope_cols,
            gam_gpu::linalg_dispatch::try_fast_joint_hessian_2x2(
                x_full.view(),
                g_full.view(),
                w_mm.view(),
                w_mg.view(),
                w_gg.view(),
            ),
        )
        .map(Some)
    }
}

impl BernoulliRigidRowKernel {
    /// Chunked BLAS-3 implementation backing
    /// [`RowKernel::hessian_dense_override`]. `row_hessians[row]` is the cached
    /// primary `2×2` row Hessian, indexed by full-data row; each walked row of
    /// `rows` enters with its Horvitz-Thompson weight. Materialization and any
    /// selected CUDA Gram execution report errors through the row-kernel
    /// dense-Hessian contract.
    fn hessian_dense_blas3(&self, rows: &RowSet, row_hessians: &[[[f64; 2]; 2]]) -> Result<Array2<f64>, String> {
        let slices = &self.slices;
        let chunks = row_set_gram_chunks(rows, self.family.y.len());
        let chunk_rows = chunks.first().map_or(1, |&(start, end)| end - start);
        // Each chunk covers a block of walk positions. Under `RowSet::All` that
        // is a contiguous block of design rows; for a
        // materialised-dense design that is a zero-copy `ArrayView2`; for an
        // operator-backed / residualised design it is one `try_row_chunk`
        // materialisation of just `CHUNK_ROWS` rows — the same mechanism the
        // directional-derivative BLAS-3 override and `add_weighted_hw_cross_terms`
        // already use, so the gate fires for the biobank rigid fit regardless of
        // whether the designs expose a pre-materialised `as_dense_ref`. The gate
        // in `hessian_dense_override` excludes sparse designs, so `try_row_chunk`
        // here never densifies a sparse block. A failed chunk materialisation at
        // the converged β snapshot is a hard numerical-contract error because
        // the design row buffer is fixed for the whole fit.
        type GramChunkResult = Result<BernoulliBlockHessianAccumulator, String>;
        let chunk_body = |(start, end): (usize, usize)| -> GramChunkResult {
            // Pin the per-chunk faer Gram GEMMs to `Par::Seq` for the duration of
            // this chunk body. The outer `chunks.into_par_iter()` already fans the
            // row-blocks (sized by `blas3_gram_chunk_rows` to fill the pool)
            // across the full Rayon pool, so each chunk runs on its own worker;
            // without the nested-parallel marker the `Xᵀdiag(w)X` GEMM
            // inside `add_weighted_design_grams_from_chunks` re-consults
            // `effective_global_parallelism()` with no marker active and gets
            // `Par::rayon(0)` = "fan across every worker" — multiplying the live
            // thread count (chunks × pool) into the documented
            // Rayon-pool × faer-pool oversubscription (304 threads on a 52-core
            // box) that stalls this otherwise BLAS-3-bound cycle-0 assembly.
            // Exactness-preserving: faer partitions the GEMM *output*, never the
            // contracted row axis, so `Par::Seq` and `Par::rayon` produce
            // bit-identical Grams.
            gam_problem::with_nested_parallel(|| {
                let len = end - start;
                let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                let mut w_mm = Array1::<f64>::zeros(len);
                let mut w_mg = Array1::<f64>::zeros(len);
                let mut w_gg = Array1::<f64>::zeros(len);
                let x_chunk =
                    row_set_design_rows(&self.family.marginal_design, rows, (start, end), "bernoulli rigid marginal_design")?;
                let g_chunk =
                    row_set_design_rows(&self.family.slope_design, rows, (start, end), "bernoulli rigid slope_design")?;
                for local in 0..len {
                    let (row, w) = rows.row_at(start + local);
                    let h = &row_hessians[row];
                    w_mm[local] = w * h[0][0];
                    w_mg[local] = w * h[0][1];
                    w_gg[local] = w * h[1][1];
                }
                acc.add_weighted_design_grams_from_chunks(&x_chunk, &g_chunk, &w_mm, &w_mg, &w_gg)?;
                Ok(acc)
            })
        };

        let run_serial =
            !gam_runtime::parallel::at_top_level() || rayon::current_num_threads() <= 1;
        if run_serial {
            let mut acc = BernoulliBlockHessianAccumulator::new(slices);
            for chunk in chunks {
                acc.add(&chunk_body(chunk)?);
            }
            return Ok(acc.to_dense(slices));
        }
        let acc = gam_runtime::parallel::fan_out(|| {
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold_by_work(
                chunks.len(),
                chunk_rows,
                |range| -> Result<BernoulliBlockHessianAccumulator, String> {
                    let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                    for chunk in &chunks[range] {
                        acc.add(&chunk_body(*chunk)?);
                    }
                    Ok(acc)
                },
                |mut left, right| -> Result<BernoulliBlockHessianAccumulator, String> {
                    left.add(&right);
                    Ok(left)
                },
            )
        })?
        .unwrap_or_else(|| BernoulliBlockHessianAccumulator::new(slices));
        Ok(acc.to_dense(slices))
    }

    /// Chunked BLAS-3 implementation backing
    /// [`RowKernel::directional_derivative_dense_override`].
    fn directional_derivative_dense_blas3(&self, rows: &RowSet, d_beta: &[f64]) -> Result<Array2<f64>, String> {
        let slices = &self.slices;
        let d_beta = ndarray::ArrayView1::from(d_beta);
        // Single-column `(p_block × 1)` direction blocks so the per-chunk
        // projection `X_chunk · dir` is one GEMM each (matching the per-row
        // `dot_row_view` the scalar path used).
        let marginal_dir_mat = d_beta
            .slice(s![slices.marginal.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        let slope_dir_mat = d_beta
            .slice(s![slices.slope.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        // Build the shared per-row third tensors in one parallel row pass before
        // any chunk fold, so chunk bodies do an O(1) lookup.
        let third_full = self.all_third_full()?;

        let chunks = row_set_gram_chunks(rows, self.family.y.len());
        let chunk_rows = chunks.first().map_or(1, |&(start, end)| end - start);
        let chunk_body =
            |(start, end): (usize, usize)| -> Result<BernoulliBlockHessianAccumulator, String> {
                // Same nested-parallel pin as `hessian_dense_blas3`: the per-chunk
                // projection (`fast_ab`) and `Xᵀdiag(w)X` Grams run on the owning
                // Rayon worker at `Par::Seq` so they do not re-fan the global pool
                // against the outer `chunks.into_par_iter()`. Exactness-preserving:
                // faer partitions the GEMM output, not the contracted row axis.
                gam_problem::with_nested_parallel(|| {
                    let len = end - start;
                    let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                    let mut w_mm = Array1::<f64>::zeros(len);
                    let mut w_mg = Array1::<f64>::zeros(len);
                    let mut w_gg = Array1::<f64>::zeros(len);
                    let x_chunk =
                    row_set_design_rows(&self.family.marginal_design, rows, (start, end), "bernoulli rigid marginal_design")?;
                    let g_chunk =
                    row_set_design_rows(&self.family.slope_design, rows, (start, end), "bernoulli rigid slope_design")?;
                    let marginal_projected =
                        gam_linalg::faer_ndarray::fast_ab(&x_chunk, &marginal_dir_mat);
                    let slope_projected =
                        gam_linalg::faer_ndarray::fast_ab(&g_chunk, &slope_dir_mat);
                    for local in 0..len {
                        let (row, w) = rows.row_at(start + local);
                        let dq = marginal_projected[[local, 0]];
                        let dg = slope_projected[[local, 0]];
                        let t = contract_third_full(third_full[row], dq, dg);
                        w_mm[local] = w * t[0][0];
                        w_mg[local] = w * t[0][1];
                        w_gg[local] = w * t[1][1];
                    }
                    acc.add_weighted_design_grams_from_chunks(
                        &x_chunk, &g_chunk, &w_mm, &w_mg, &w_gg,
                    )?;
                    Ok(acc)
                })
            };

        // Parallel over chunks: each chunk body is an independent BLAS-3 GEMM
        // pair over `CHUNK_ROWS` rows reading the already-built shared third
        // tensor, so the fold has no nested cache contention. Use a serial
        // chunk loop when not at top level (the outer
        // joint-Newton / ψ-sweep par_iter holds the pool) so a nested
        // `into_par_iter` does not starve the pool — the same guard the batched
        // builder uses.
        let run_serial =
            !gam_runtime::parallel::at_top_level() || rayon::current_num_threads() <= 1;
        if run_serial {
            let mut acc = BernoulliBlockHessianAccumulator::new(slices);
            for chunk in chunks {
                let partial = chunk_body(chunk)?;
                acc.add(&partial);
            }
            return Ok(acc.to_dense(slices));
        }
        let acc = gam_runtime::parallel::fan_out(|| {
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold_by_work(
                chunks.len(),
                chunk_rows,
                |range| -> Result<BernoulliBlockHessianAccumulator, String> {
                    let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                    for chunk in &chunks[range] {
                        let partial = chunk_body(*chunk)?;
                        acc.add(&partial);
                    }
                    Ok(acc)
                },
                |mut left, right| -> Result<BernoulliBlockHessianAccumulator, String> {
                    left.add(&right);
                    Ok(left)
                },
            )
        })?
        .unwrap_or_else(|| BernoulliBlockHessianAccumulator::new(slices));
        Ok(acc.to_dense(slices))
    }

    /// Chunked BLAS-3 implementation backing
    /// [`RowKernel::second_directional_derivative_all_axes_dense_override`].
    ///
    /// Returns the `p` dense matrices `{H²dot[u, e_a]}_{a=0..p}` for the fixed
    /// direction `u = d_beta_u`. The per-row `u`-projection
    /// `(uq_r, ug_r) = (X·u_marg, G·u_logs)` is built ONCE (one chunked GEMM per
    /// design block, hoisted out of the `p`-axis loop — the dominant
    /// `O(p·n)` jet-projection redundancy of the generic per-axis sweep). Each
    /// axis then closes its design-row Gram with the same BLAS-3 machinery the
    /// first-directional override uses, replacing the `O(p·n·p²)` BLAS-1
    /// `add_pullback_hessian` scatter with `p` BLAS-3 `Xᵀ diag(w) X` builds.
    ///
    /// Bit-exactness: each axis's per-row weight is
    /// `contract_fourth_full(T⁴ᵣ, uq_r, ug_r, vq_r, vg_r)` with the SAME
    /// arguments the generic `row_fourth_contracted` receives — `dir_u` is the
    /// row `u`-projection, `dir_v` is the row `e_a`-projection
    /// (`(X[r,j], 0)` for a marginal axis, `(0, G[r,j])` for a slope axis,
    /// the exact value `jacobian_action(row, e_a)` returns). The Gram swap from
    /// BLAS-1 syr scatter to `fast_xt_diag_*` reduces in the identical in-row
    /// order (same contract as `hessian_dense_blas3`), so axis `a` matches
    /// `row_kernel_second_directional_derivative(self, All, u, e_a)`
    /// bit-for-bit.
    fn second_directional_derivative_all_axes_blas3(
        &self,
        rows: &RowSet,
        d_beta_u: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = &self.slices;
        let n = self.family.y.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
        let d_beta_u = ndarray::ArrayView1::from(d_beta_u);
        // Fixed-direction blocks for the single-column `u`-projection GEMMs.
        let u_marg_mat = d_beta_u
            .slice(s![slices.marginal.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        let u_logs_mat = d_beta_u
            .slice(s![slices.slope.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        // Build the shared per-row fourth tensors in one parallel row pass
        // before the axis fan-out: every axis sweeps every row in order, so a
        // cold row read inside it would build behind one axis while the others
        // wait.
        let fourth_full = self.all_fourth_full()?;

        let chunks = row_set_gram_chunks(rows, n);
        let m = rows.walk_len(n);

        // Hoisted per-position `u`-projection `(uq_k, ug_k)` of walked row `r`,
        // built ONCE via one chunked GEMM per block. `uq[k] = X.row(r)·u_marg`,
        // `ug[k] = G.row(r)·u_logs` — bit-identical to `jacobian_action(r, d_beta_u)`
        // (a single design-row dot per axis), just batched.
        let mut uq = Array1::<f64>::zeros(m);
        let mut ug = Array1::<f64>::zeros(m);
        for &(start, end) in &chunks {
            gam_problem::with_nested_parallel(|| -> Result<(), String> {
                let x_chunk =
                    row_set_design_rows(&self.family.marginal_design, rows, (start, end), "bernoulli rigid marginal_design")?;
                let g_chunk =
                    row_set_design_rows(&self.family.slope_design, rows, (start, end), "bernoulli rigid slope_design")?;
                let uq_chunk = gam_linalg::faer_ndarray::fast_ab(&x_chunk, &u_marg_mat);
                let ug_chunk = gam_linalg::faer_ndarray::fast_ab(&g_chunk, &u_logs_mat);
                for position in start..end {
                    uq[position] = uq_chunk[[position - start, 0]];
                    ug[position] = ug_chunk[[position - start, 0]];
                }
                Ok(())
            })?;
        }

        // One axis = one independent full-data design-row Gram. Marginal axes
        // are `e_a` with the unit in the marginal block (axis projection
        // `(X[r,j], 0)`); slope axes have it in the slope block
        // (`(0, G[r,j])`). Fan the `p` axes across the pool (each is a pure
        // evaluation reading the shared `uq/ug` and the cached fourth tensor);
        // the nested-BLAS guard pins each axis's chunk GEMMs to `Par::Seq`.
        // Index-ordered collection keeps the output bit-identical to a serial
        // axis loop.
        let build_axis = |axis_global: usize| -> Result<Array2<f64>, String> {
            gam_problem::with_nested_parallel(|| {
                // Resolve the axis to its block and the local design column.
                let marginal_axis = axis_global < p_m;
                let local_col = if marginal_axis {
                    axis_global
                } else {
                    axis_global - p_m
                };
                let axis_chunk_body =
                    |(start, end): (usize, usize)| -> Result<BernoulliBlockHessianAccumulator, String> {
                        let len = end - start;
                        let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                        let x_chunk =
                    row_set_design_rows(&self.family.marginal_design, rows, (start, end), "bernoulli rigid marginal_design")?;
                        let g_chunk =
                    row_set_design_rows(&self.family.slope_design, rows, (start, end), "bernoulli rigid slope_design")?;
                        let mut w_mm = Array1::<f64>::zeros(len);
                        let mut w_mg = Array1::<f64>::zeros(len);
                        let mut w_gg = Array1::<f64>::zeros(len);
                        for local in 0..len {
                            let position = start + local;
                            let (row, w) = rows.row_at(position);
                            // `dir_v = jacobian_action(row, e_a)`: a unit pick of
                            // one design column, zero in the other block. Read the
                            // exact same scalar the generic per-axis path reads.
                            let (vq, vg) = if marginal_axis {
                                (x_chunk[[local, local_col]], 0.0)
                            } else {
                                (0.0, g_chunk[[local, local_col]])
                            };
                            // Identical args to the generic `row_fourth_contracted`:
                            // `(dir_u = (uq, ug), dir_v = (vq, vg))`.
                            let m = contract_fourth_full(
                                fourth_full[row],
                                uq[position],
                                ug[position],
                                vq,
                                vg,
                            );
                            w_mm[local] = w * m[0][0];
                            w_mg[local] = w * m[0][1];
                            w_gg[local] = w * m[1][1];
                        }
                        acc.add_weighted_design_grams_from_chunks(
                            &x_chunk, &g_chunk, &w_mm, &w_mg, &w_gg,
                        )?;
                        Ok(acc)
                    };
                // Serial chunk fold within an axis: the axis fan-out already
                // occupies the pool, and a serial in-order chunk reduce matches
                // the `directional_derivative_dense_blas3` chunk-accumulation
                // order exactly (bit-for-bit against the generic per-axis path).
                let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                for chunk in &chunks {
                    let partial = axis_chunk_body(*chunk)?;
                    acc.add(&partial);
                }
                Ok(acc.to_dense(slices))
            })
        };

        let p_total = p_m + p_g;
        let run_serial =
            !gam_runtime::parallel::at_top_level() || rayon::current_num_threads() <= 1;
        if run_serial {
            (0..p_total).map(build_axis).collect::<Result<Vec<_>, _>>()
        } else {
            gam_runtime::parallel::fan_out(|| {
                (0..p_total)
                    .into_par_iter()
                    .map(build_axis)
                    .collect::<Result<Vec<_>, _>>()
            })
        }
    }

    /// Chunked BLAS-3 implementation backing
    /// [`RowKernel::directional_derivative_all_axes_dense_override`].
    ///
    /// Returns the `p` dense matrices `{Hdot[e_a]}_{a=0..p}`. Each axis's per-row
    /// weight is `contract_third_full(T³ᵣ, vq_r, vg_r)` with `(vq_r, vg_r) =
    /// jacobian_action(row, e_a)` — `(X[r,j], 0)` for a marginal axis,
    /// `(0, G[r,j])` for a slope axis. `contract_third_full` is LINEAR in
    /// `(vq, vg)`, so a marginal axis's weight is `X[r,j] · A_r` with
    /// `A_r = contract_third_full(T³ᵣ, 1, 0)`, and a slope axis's is
    /// `G[r,j] · B_r` with `B_r = contract_third_full(T³ᵣ, 0, 1)`. We read the
    /// cached third tensor and build `A_r, B_r` ONCE per row (the `~p×`
    /// reduction over the per-axis path's repeated kernel/tensor rebuilds), then
    /// close each axis with the same chunked `Xᵀ diag(w) X` / `Xᵀ diag(w) G`
    /// BLAS-3 machinery (`add_weighted_design_grams_from_chunks`).
    ///
    /// Bit-exactness: axis `a`'s per-row `2×2` weight equals
    /// `contract_third_full(T³ᵣ, vq_r, vg_r)` — the exact value the generic
    /// `row_third_contracted(row, jacobian_action(row, e_a))` produces — and the
    /// Gram reduces in the identical in-row order as `hessian_dense_blas3`, so
    /// axis `a` matches `row_kernel_directional_derivative(self, All, e_a)`
    /// bit-for-bit.
    fn directional_derivative_all_axes_blas3(&self, rows: &RowSet) -> Result<Vec<Array2<f64>>, String> {
        let slices = &self.slices;
        let n = self.family.y.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
        // Build the shared per-row third tensors in one parallel row pass before
        // the per-row `A/B` loop below reads them.
        let third_full = self.all_third_full()?;

        let chunks = row_set_gram_chunks(rows, n);
        let m = rows.walk_len(n);

        // Per-position axis-independent partial contractions of walked row `r`,
        // HT weight `w` folded in, built ONCE:
        //   A_k = w · contract_third_full(T³ᵣ, 1, 0)   (marginal-axis unit weight)
        //   B_k = w · contract_third_full(T³ᵣ, 0, 1)   (slope-axis unit weight)
        // Each is a symmetric `2×2`; we keep the three independent entries
        // `(mm, mg, gg)` the design-row Gram consumes.
        let mut a_mm = Array1::<f64>::zeros(m);
        let mut a_mg = Array1::<f64>::zeros(m);
        let mut a_gg = Array1::<f64>::zeros(m);
        let mut b_mm = Array1::<f64>::zeros(m);
        let mut b_mg = Array1::<f64>::zeros(m);
        let mut b_gg = Array1::<f64>::zeros(m);
        for position in 0..m {
            let (row, w) = rows.row_at(position);
            let a = contract_third_full(third_full[row], 1.0, 0.0);
            let b = contract_third_full(third_full[row], 0.0, 1.0);
            a_mm[position] = w * a[0][0];
            a_mg[position] = w * a[0][1];
            a_gg[position] = w * a[1][1];
            b_mm[position] = w * b[0][0];
            b_mg[position] = w * b[0][1];
            b_gg[position] = w * b[1][1];
        }

        // One axis = one independent full-data design-row Gram. A marginal axis
        // `j` projects to `(X[r,j], 0)`, so its row weight is `X[r,j]·A_r`; a
        // slope axis `j` projects to `(0, G[r,j])`, so its row weight is
        // `G[r,j]·B_r`. Fan the `p` axes across the pool (each reads the shared
        // `A/B` weights); the nested-BLAS guard pins each axis's chunk GEMMs to
        // `Par::Seq`. Index-ordered collection keeps the output bit-identical to
        // a serial axis loop.
        let build_axis = |axis_global: usize| -> Result<Array2<f64>, String> {
            gam_problem::with_nested_parallel(|| {
                let marginal_axis = axis_global < p_m;
                let local_col = if marginal_axis {
                    axis_global
                } else {
                    axis_global - p_m
                };
                let axis_chunk_body =
                    |(start, end): (usize, usize)| -> Result<BernoulliBlockHessianAccumulator, String> {
                        let len = end - start;
                        let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                        let x_chunk =
                    row_set_design_rows(&self.family.marginal_design, rows, (start, end), "bernoulli rigid marginal_design")?;
                        let g_chunk =
                    row_set_design_rows(&self.family.slope_design, rows, (start, end), "bernoulli rigid slope_design")?;
                        let mut w_mm = Array1::<f64>::zeros(len);
                        let mut w_mg = Array1::<f64>::zeros(len);
                        let mut w_gg = Array1::<f64>::zeros(len);
                        for local in 0..len {
                            let position = start + local;
                            // Axis projection scalar `s = jacobian_action(row, e_a)`
                            // in the active block, scaling the precomputed unit-axis
                            // contraction. `contract_third_full` is linear, so this
                            // equals `contract_third_full(T³ᵣ, vq_r, vg_r)` exactly.
                            if marginal_axis {
                                let s = x_chunk[[local, local_col]];
                                w_mm[local] = s * a_mm[position];
                                w_mg[local] = s * a_mg[position];
                                w_gg[local] = s * a_gg[position];
                            } else {
                                let s = g_chunk[[local, local_col]];
                                w_mm[local] = s * b_mm[position];
                                w_mg[local] = s * b_mg[position];
                                w_gg[local] = s * b_gg[position];
                            }
                        }
                        acc.add_weighted_design_grams_from_chunks(
                            &x_chunk, &g_chunk, &w_mm, &w_mg, &w_gg,
                        )?;
                        Ok(acc)
                    };
                // Serial in-order chunk reduce matches the
                // `directional_derivative_dense_blas3` accumulation order exactly
                // (bit-for-bit against the generic per-axis path).
                let mut acc = BernoulliBlockHessianAccumulator::new(slices);
                for chunk in &chunks {
                    let partial = axis_chunk_body(*chunk)?;
                    acc.add(&partial);
                }
                Ok(acc.to_dense(slices))
            })
        };

        let p_total = p_m + p_g;
        let run_serial =
            !gam_runtime::parallel::at_top_level() || rayon::current_num_threads() <= 1;
        if run_serial {
            (0..p_total).map(build_axis).collect::<Result<Vec<_>, _>>()
        } else {
            gam_runtime::parallel::fan_out(|| {
                (0..p_total)
                    .into_par_iter()
                    .map(build_axis)
                    .collect::<Result<Vec<_>, _>>()
            })
        }
    }
}

pub(super) struct BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    pub(super) family: BernoulliMarginalSlopeFamily,
    pub(super) block_states: Vec<ParameterBlockState>,
    pub(super) cache: Arc<BernoulliMarginalSlopeExactEvalCache>,
    pub(super) matvec_calls: AtomicUsize,
    pub(super) fused_gradient_dense:
        OnceLock<Result<Arc<ExactNewtonJointFusedDenseEvaluation>, String>>,
    #[cfg(target_os = "linux")]
    pub(super) device_joint_gradient:
        OnceLock<Result<Arc<ExactNewtonJointGradientEvaluation>, String>>,
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
    pub(super) hyper_layout: crate::custom_family::CustomFamilyHyperLayout,
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
    // that magnitude (ulp(1.5e5) = 2^(17-52) ≈ 2.9e-11). Each side accumulates
    // at most `n` weighted row terms with one product apiece, so whatever its
    // associativity order Wilkinson's bound holds it within `γ_{n+1}` of the
    // absolute sum of its terms, and the reject band is the two bands together:
    // `γ_{n+1}·(|threshold| + |partial LL|)`. This stays a valid reject
    // certificate: it only DEFERS borderline trials whose partial NLL exceeds the
    // threshold by less than cross-path round-off to the full exact LL return
    // value plus the caller's objective/ρ accept test; it never early-accepts a
    // trial whose true full-data NLL is genuinely worse than the threshold by more
    // than this round-off band.
    let accumulation_growth = gam_linalg::roundoff::accumulation_growth(weighted_rows.len() + 1);
    let mut total_ll = 0.0;
    for chunk in weighted_rows.chunks(BERNOULLI_MARGSLOPE_LINE_SEARCH_EARLY_EXIT_CHUNK_ROWS) {
        let chunk_ll: f64 = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            chunk.len(),
            |range| -> Result<f64, String> {
                let mut acc = 0.0;
                for wr in &chunk[range] {
                    acc += wr.weight * row_ll(wr.index)?;
                }
                Ok(acc)
            },
            |left, right| -> Result<_, String> { Ok(left + right) },
        )?
        .unwrap_or(0.0);
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
        let early_exit_reject_tol = accumulation_growth * (threshold.abs() + total_ll.abs());
        if -total_ll > threshold + early_exit_reject_tol {
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
        let row_ll = |_: usize| -> Result<f64, String> { Ok(-1.0) };

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
    /// the biobank gauge-flat marginal/slope hang — the line search rejected
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
        let row_ll = move |_: usize| -> Result<f64, String> { Ok(per_row_ll) };
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
        let high_row_ll = move |_: usize| -> Result<f64, String> { Ok(high_per_row_ll) };
        assert!(
            bernoulli_margslope_line_search_ll_with_early_exit(&rows, threshold, high_row_ll)
                .is_err(),
            "a trial worse than the threshold by far more than the round-off band \
             must still early-reject"
        );
    }
}

#[cfg(test)]
mod rigid_row_kernel_closed_form_tests {
    //! The rigid kernel's empirical-grid `row_kernel` and third/fourth tensors
    //! read the closed forms the family's own rigid paths use instead of
    //! compiling the FLEX row program for the row. Both are the same row NLL at
    //! the same calibration root, so every channel must match the inherited
    //! program's value, score, curvature and tensors to the 1e-9
    //! scaled error the closed form is pinned at against its exact oracle
    //! (`empirical_rigid_all_channels_match_independent_polynomial_932`).
    use super::*;
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use gam_problem::{InverseLink, ParameterBlockState, StandardLink};
    use ndarray::{Array1, Array2};

    /// A skewed, heavy-tailed 65-node latent law, the production grid size.
    fn skewed_grid() -> EmpiricalZGrid {
        let k = 65usize;
        let nodes: Vec<f64> = (0..k)
            .map(|i| {
                let t = -3.0 + 6.0 * i as f64 / (k - 1) as f64;
                t + 0.18 * (t * t - 1.0) + 0.02 * t * t * t
            })
            .collect();
        let raw: Vec<f64> = (0..k)
            .map(|i| {
                let t = -3.0 + 6.0 * i as f64 / (k - 1) as f64;
                (-0.5 * t * t).exp() * (1.0 + 0.3 * (1.7 * t).sin().abs())
            })
            .collect();
        let total: f64 = raw.iter().sum();
        let weights = raw.iter().map(|w| w / total).collect();
        EmpiricalZGrid::new(nodes, weights, "rigid row-kernel closed form").expect("valid grid")
    }

    fn fixture(frailty_sd: Option<f64>) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
        let n = 48usize;
        let marginal_x = Array2::from_shape_fn((n, 3), |(i, j)| {
            if j == 0 { 1.0 } else { ((i * (j + 2)) as f64 * 0.31).sin() }
        });
        let slope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { 1.0 } else { ((i + 3 * j) as f64 * 0.23).cos() }
        });
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let latent_measure = LatentMeasureKind::GlobalEmpirical { grid: skewed_grid() };
        let intercept_warm_starts = new_intercept_warm_start_cache_on_law(&latent_measure, n)
            .expect("an intercept cache on the empirical law");
        let family = BernoulliMarginalSlopeFamily {
            jeffreys_armed: true,
            residual: None,
            search: None,
            y: Arc::new(Array1::from_shape_fn(n, |i| if (i * 7) % 5 < 2 { 1.0 } else { 0.0 })),
            weights: Arc::new(Array1::from_shape_fn(n, |i| 0.6 + 0.02 * i as f64)),
            z: Arc::new(Array1::from_shape_fn(n, |i| 1.9 * (i as f64 * 0.57).sin())),
            latent_measure,
            gaussian_frailty_sd: frailty_sd,
            base_link: InverseLink::Standard(StandardLink::Probit),
            marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
            slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone())),
            score_warp: None,
            link_dev: None,
            policy: policy.clone(),
            cell_moment_lru: new_cell_moment_lru_cache(&policy),
            cell_moment_cache_stats: new_cell_moment_cache_stats(),
            intercept_warm_starts: Some(intercept_warm_starts),
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        // Marginal η spans roughly ±1.6 and the slope ±0.9 across the rows.
        let marginal_beta = Array1::from_vec(vec![-0.4, 0.9, -0.7]);
        let slope_beta = Array1::from_vec(vec![0.35, -0.55]);
        let states = vec![
            ParameterBlockState { eta: marginal_x.dot(&marginal_beta), beta: marginal_beta },
            ParameterBlockState { eta: slope_x.dot(&slope_beta), beta: slope_beta },
        ];
        (family, states)
    }

    fn scaled_error(actual: f64, expected: f64) -> f64 {
        if !actual.is_finite() || !expected.is_finite() {
            return f64::INFINITY;
        }
        (actual - expected).abs() / actual.abs().max(expected.abs()).max(1.0)
    }

    #[test]
    fn empirical_row_kernel_matches_the_inherited_program_channels() {
        let mut worst = 0.0_f64;
        for frailty_sd in [None, Some(0.6)] {
            let (family, states) = fixture(frailty_sd);
            let n = family.y.len();
            let kern = BernoulliRigidRowKernel::new(family, states);
            for row in 0..n {
                let (value, gradient, hessian) = kern.row_kernel(row).expect("closed-form row");
                let (p_value, p_gradient, p_hessian) =
                    gam_math::jet_tower::program_row_kernel(&kern, row).expect("program row");
                let mut pairs = vec![(value, p_value)];
                for a in 0..2 {
                    pairs.push((gradient[a], p_gradient[a]));
                    for b in 0..2 {
                        pairs.push((hessian[a][b], p_hessian[a][b]));
                    }
                }
                let tower = gam_math::jet_tower::program_full_tower(&kern, row).expect("program tower");
                let third = kern.row_third_full(row).expect("closed-form third");
                let fourth = kern.row_fourth_full(row).expect("closed-form fourth");
                for a in 0..2 {
                    for b in 0..2 {
                        for c in 0..2 {
                            pairs.push((third[a][b][c], tower.t3[a][b][c]));
                            for d in 0..2 {
                                pairs.push((fourth[a][b][c][d], tower.t4[a][b][c][d]));
                            }
                        }
                    }
                }
                for (channel, (actual, expected)) in pairs.into_iter().enumerate() {
                    let error = scaled_error(actual, expected);
                    assert!(
                        error <= 1e-9,
                        "frailty={frailty_sd:?} row={row} channel={channel}: closed form {actual:.16e} \
                         vs program {expected:.16e} (scaled error {error:.3e})"
                    );
                    worst = worst.max(error);
                }
            }
        }
        eprintln!("RIGID-ROW-KERNEL-CLOSED-FORM worst_scaled_error={worst:.3e}");
    }

    /// gam#3035: the four BLAS-3 dense overrides agree with the generic per-row
    /// reductions on the full data and on a Horvitz–Thompson-weighted subsample.
    #[test]
    fn rigid_dense_overrides_match_generic_on_every_row_set_3035() {
        for frailty_sd in [None, Some(0.6)] {
            let (family, states) = fixture(frailty_sd);
            let kern = BernoulliRigidRowKernel::new(family, states);
            crate::test_support::row_set_overrides::assert_dense_overrides_match_generic(
                &format!("rigid BMS frailty={frailty_sd:?}"),
                &kern,
                &[0.4, -0.3, 0.8, -0.6, 0.5],
                &[-0.7, 0.2, 0.5, 0.9, -0.35],
                1e-13,
            );
        }
    }
}
