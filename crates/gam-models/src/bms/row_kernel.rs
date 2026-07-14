use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::*;
use crate::fnv1a::Fnv1a;
use std::sync::{Mutex, OnceLock};

// ── Same-β rigid third/fourth-tensor cache ───────────────────────────
//
// The rigid coord_corrections (IFT Hessian-drift) path builds a per-row
// uncontracted third-derivative tensor over ALL n rows, and the outer-Hessian
// path builds the per-row fourth tensor likewise. A FRESH `BernoulliRigidRowKernel`
// (empty `RayonSafeOnce` slots) is constructed on every outer eval
// (`exact_newton_joint_hessian_workspace*`), so at biobank scale (n≈3e5) the
// closed-form per-row jet re-runs over every row each eval — the dominant REML
// `coord_corrections` cost. The tensors are a pure function of the family/data
// identity and the coefficient state (block β + η), exactly like the same-β
// exact-cache (`SharedExactCacheStore`); mirror it with a module-level FIFO-2
// store so the immediate Value→ValueAndGradient pair at one β̂, and any
// line-search ρ that maps back to a seen β̂, reuse a single n-row build instead
// of rebuilding. Reuse is gated on exact byte-equality of a content
// fingerprint over the data-buffer Arc identities, the frailty/latent/deviation
// discriminants, and every block's β + η, so a hit returns an `Arc` to a
// bit-identical tensor (or misses).
type RigidThirdFull = Vec<[[[f64; 2]; 2]; 2]>;
type RigidFourthFull = Vec<[[[[f64; 2]; 2]; 2]; 2]>;

struct SharedRigidTensorStore {
    third: Vec<(u64, Arc<RigidThirdFull>)>,
    fourth: Vec<(u64, Arc<RigidFourthFull>)>,
}

impl SharedRigidTensorStore {
    const CAPACITY: usize = 2;

    fn get_third(&self, fp: u64) -> Option<Arc<RigidThirdFull>> {
        self.third
            .iter()
            .find(|(key, _)| *key == fp)
            .map(|(_, v)| Arc::clone(v))
    }

    fn insert_third(&mut self, fp: u64, value: Arc<RigidThirdFull>) {
        if self.third.iter().any(|(key, _)| *key == fp) {
            return;
        }
        if self.third.len() >= Self::CAPACITY {
            self.third.remove(0);
        }
        self.third.push((fp, value));
    }

    fn get_fourth(&self, fp: u64) -> Option<Arc<RigidFourthFull>> {
        self.fourth
            .iter()
            .find(|(key, _)| *key == fp)
            .map(|(_, v)| Arc::clone(v))
    }

    fn insert_fourth(&mut self, fp: u64, value: Arc<RigidFourthFull>) {
        if self.fourth.iter().any(|(key, _)| *key == fp) {
            return;
        }
        if self.fourth.len() >= Self::CAPACITY {
            self.fourth.remove(0);
        }
        self.fourth.push((fp, value));
    }
}

fn shared_rigid_tensor_store() -> &'static Mutex<SharedRigidTensorStore> {
    static STORE: OnceLock<Mutex<SharedRigidTensorStore>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(SharedRigidTensorStore {
            third: Vec::with_capacity(SharedRigidTensorStore::CAPACITY),
            fourth: Vec::with_capacity(SharedRigidTensorStore::CAPACITY),
        })
    })
}

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
    /// Holds an `Arc` to the (possibly globally-shared, same-β) tensor so a
    /// cross-eval hit in [`shared_rigid_tensor_store`] is stored here once and
    /// then served `O(1)` to every ψ-axis operator that consults this kernel.
    pub(super) third_full_cache: gam_runtime::resource::RayonSafeOnce<Arc<RigidThirdFull>>,
    /// Per-row uncontracted fourth-derivative tensor — the outer-Hessian
    /// analogue of `third_full_cache`. The second-directional-derivative
    /// operator's trace path touches every row × (u, v) pair; with this
    /// cache the heavy 8-direction empirical jet (or closed-form 5-component
    /// build) runs at most once per row, leaving each pair with a cheap
    /// [`contract_fourth_full`] bilinear.
    pub(super) fourth_full_cache: gam_runtime::resource::RayonSafeOnce<Arc<RigidFourthFull>>,
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
            third_full_cache: gam_runtime::resource::RayonSafeOnce::new(),
            fourth_full_cache: gam_runtime::resource::RayonSafeOnce::new(),
        }
    }

    /// Content fingerprint of every input the per-row rigid third/fourth jet
    /// reads: the family/data identity (stable `Arc::as_ptr` of the immutable
    /// `y`/`z`/`weights` buffers), the probit-frailty scale, the latent-measure
    /// discriminant, the score-warp / link-deviation presence flags, and every
    /// block's β + η. `rigid_row_third_full`/`rigid_row_fourth_full` are pure
    /// functions of exactly these (the per-row build reads `block_states[*].eta`,
    /// `self.z[row]`/`y[row]`/`weights[row]`, the frailty scale, and the latent
    /// grid pinned by the data-buffer address), so equal fingerprints ⇒
    /// bit-identical tensors. The `domain` byte separates the third- and
    /// fourth-tensor key streams. Mirrors
    /// `BernoulliMarginalSlopeFamily::shared_exact_cache_fingerprint`.
    fn rigid_tensor_fingerprint(&self, domain: u8) -> u64 {
        let mut hash = Fnv1a::new();
        hash.mix_byte(domain);
        for &ptr in &[
            Arc::as_ptr(&self.family.y) as usize,
            Arc::as_ptr(&self.family.z) as usize,
            Arc::as_ptr(&self.family.weights) as usize,
        ] {
            for b in (ptr as u64).to_le_bytes() {
                hash.mix_byte(b);
            }
        }
        hash.mix_byte(0xf1);
        match self.family.gaussian_frailty_sd {
            Some(sd) => {
                hash.mix_byte(0x01);
                hash.mix_f64(sd);
            }
            None => hash.mix_byte(0x00),
        }
        let latent_byte: u8 = match self.family.latent_measure {
            LatentMeasureKind::StandardNormal => 0x10,
            LatentMeasureKind::GlobalEmpirical { .. } => 0x11,
            LatentMeasureKind::LocalEmpirical { .. } => 0x12,
        };
        hash.mix_byte(latent_byte);
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
                let fp = self.rigid_tensor_fingerprint(0xa3);
                if let Some(hit) = shared_rigid_tensor_store()
                    .lock()
                    .expect("BMS rigid tensor store mutex poisoned on third read")
                    .get_third(fp)
                {
                    return hit;
                }
                let n = self.family.y.len();
                // Named heartbeat scope: this per-row uncontracted third-tensor
                // build is the rigid coord_corrections cost suspect (one n-row
                // pass per distinct β̂; reused across the Value/Gradient pair and
                // line-search re-probes via the same-β store).
                let scope_guard = gam_runtime::process_monitor::track_scope(format!(
                    "BMS rigid third_full_cache build n={n}"
                ));
                let built: RigidThirdFull = (0..n)
                    .into_par_iter()
                    .map(|row| {
                        gam_math::jet_tower::program_full_tower(self, row).map(|tower| tower.t3)
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .expect(
                        "BernoulliRigidRowKernel third-full cache build failed; \
                         per-row jet should not error at the converged β snapshot",
                    );
                let shared = Arc::new(built);
                shared_rigid_tensor_store()
                    .lock()
                    .expect("BMS rigid tensor store mutex poisoned on third write")
                    .insert_third(fp, Arc::clone(&shared));
                drop(scope_guard);
                shared
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
                let fp = self.rigid_tensor_fingerprint(0xa4);
                if let Some(hit) = shared_rigid_tensor_store()
                    .lock()
                    .expect("BMS rigid tensor store mutex poisoned on fourth read")
                    .get_fourth(fp)
                {
                    return hit;
                }
                let n = self.family.y.len();
                let scope_guard = gam_runtime::process_monitor::track_scope(format!(
                    "BMS rigid fourth_full_cache build n={n}"
                ));
                let built: RigidFourthFull = (0..n)
                    .into_par_iter()
                    .map(|row| {
                        gam_math::jet_tower::program_full_tower(self, row).map(|tower| tower.t4)
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .expect(
                        "BernoulliRigidRowKernel fourth-full cache build failed; \
                         per-row jet should not error at the converged β snapshot",
                    );
                let shared = Arc::new(built);
                shared_rigid_tensor_store()
                    .lock()
                    .expect("BMS rigid tensor store mutex poisoned on fourth write")
                    .insert_fourth(fp, Arc::clone(&shared));
                drop(scope_guard);
                shared
            })
            .as_slice()
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
        match self
            .family
            .latent_measure
            .empirical_grid_for_training_row(row)?
        {
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
    fn warm_up_directional_caches(&self, eval_mode: EvalMode) -> Result<(), String> {
        // gam#979: prime only the caches the eval about to run will consume.
        //
        //   * `ValueOnly`            → neither cache (the objective is read off
        //                              the converged inner mode; no directional
        //                              contraction is taken). Seed screening,
        //                              line-search cost probes, and typed reactive
        //                              `ContinuationPath` waypoints are all
        //                              value-only — at biobank scale each would
        //                              otherwise pay two full `O(n)` jet passes
        //                              (third + fourth) for tensors it never reads.
        //   * `ValueAndGradient`     → third-derivative cache only. The REML/LAML
        //                              gradient's `coord_corrections` IFT-drift
        //                              trace is a *first* directional derivative
        //                              (`row_third_contracted`); the BFGS
        //                              first-order bridge never asks for the
        //                              outer Hessian, so the fourth cache stays
        //                              cold for the whole fit.
        //   * `ValueGradientHessian` → both caches; the outer Hessian's second-
        //                              directional pass reads `row_fourth_contracted`.
        //
        // Under-priming is safe: both caches are lazy `get_or_compute`, so a
        // later consumer still builds on demand — it just loses this hook's
        // top-level-rayon fan-out.
        match eval_mode {
            EvalMode::ValueOnly => Ok(()),
            EvalMode::ValueAndGradient => {
                let third_cache_len = self.third_full_cache().len();
                crate::row_kernel::validate_row_kernel_cache_lengths(
                    "bernoulli rigid warm-up",
                    self.family.y.len(),
                    &[("third", third_cache_len)],
                )
            }
            EvalMode::ValueGradientHessian => {
                // Touch both caches so their parallel builds run here, not later
                // (nested inside the outer ext-idx par_iter where the lock-holder
                // thread would have to do each row pass alone).
                let third_cache_len = self.third_full_cache().len();
                let fourth_cache_len = self.fourth_full_cache().len();
                crate::row_kernel::validate_row_kernel_cache_lengths(
                    "bernoulli rigid warm-up",
                    self.family.y.len(),
                    &[("third", third_cache_len), ("fourth", fourth_cache_len)],
                )
            }
        }
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

        // Compute J_block · F_block for both axes. Dense designs use BLAS-3;
        // operator-backed/sparse designs use the shared per-column dispatcher,
        // preserving the same arithmetic as the per-row reference.
        let jf_marg = crate::row_kernel::row_kernel_design_jf(
            &self.family.marginal_design,
            f_marg.view(),
            n_rows,
        );
        let jf_logs = crate::row_kernel::row_kernel_design_jf(
            &self.family.logslope_design,
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
            .slice(s![self.slices.logslope.clone(), ..])
            .as_standard_layout()
            .into_owned();
        let jf_marg = crate::row_kernel::row_kernel_design_jf_rows(
            &self.family.marginal_design,
            f_marg.view(),
            start,
            end,
        );
        let jf_logs = crate::row_kernel::row_kernel_design_jf_rows(
            &self.family.logslope_design,
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
    ///   dq_r = marginal_design.row(r)·d_beta[marg],  dg_r = logslope.row(r)·d_beta[logs].
    /// ```
    ///
    /// We accumulate the per-row `2×2` contraction weights `(w_mm, w_mg, w_gg)`
    /// over a contiguous row chunk, project `(dq, dg)` for the whole chunk in
    /// two GEMMs, and close each chunk with one pair of
    /// `Xᵀ diag(w) X` / `Xᵀ diag(w) G` products
    /// (`add_weighted_design_grams_from_chunks`). The per-row third tensor is
    /// read from the shared `third_full_cache` (built once per workspace), so
    /// the `k` Jeffreys columns pay the closed-form third build at most once
    /// per row. Bit-for-bit the same entries the per-row `add_pullback_hessian`
    /// scatter writes (`w_mm = t[0][0]`, `w_mg = t[0][1]`, `w_gg = t[1][1]`),
    /// reduced in a different summation order.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` case with dense
    /// designs; otherwise returns `None` so the generic per-row Horvitz-Thompson
    /// path runs.
    fn directional_derivative_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        d_beta: &[f64],
    ) -> Option<Result<Array2<f64>, String>> {
        if !matches!(rows, crate::row_kernel::RowSet::All) {
            // Diagnostic fires once per process, not once per inner-Newton kernel
            // call: this dispatch runs on every directional-derivative evaluation,
            // so an unguarded line floods the biobank fit log with thousands of
            // identical entries.
            static DD_NOT_TAKEN_LOGGED: std::sync::Once = std::sync::Once::new();
            DD_NOT_TAKEN_LOGGED.call_once(|| {
                log::info!(
                    "[STAGE] BMS rigid directional_derivative BLAS-3 path NOT taken: RowSet is a \
                     subsample (generic per-row Horvitz-Thompson scatter)"
                );
            });
            return None;
        }
        // The chunked `Xᵀ diag(w) X` Gram slices contiguous design rows via
        // `try_row_chunk` inside `directional_derivative_dense_blas3` (which
        // already handles operator-backed / residualised designs row-chunk by
        // row-chunk), so the only structurally-inapplicable case is a sparse
        // design block — gate on that, not on the presence of a pre-materialised
        // `as_dense_ref`. Without this, a biobank rigid fit whose marginal /
        // logslope design is operator-backed (residualised absorber, overlap-Z)
        // fell through to the generic per-row third-tensor scatter — the ~8s
        // per-cycle `gradient_reload` / Jeffreys-column floor.
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let logslope_sparse = self.family.logslope_design.is_sparse();
        if marginal_sparse || logslope_sparse {
            return None;
        }
        Some(self.directional_derivative_dense_blas3(d_beta))
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
    /// X[r,j] · contract_third_full(T³ᵣ, 1, 0)`; a logslope axis has
    /// `(0, G[r,j])`. So we read each row's `A_r = contract_third_full(T³ᵣ, 1, 0)`
    /// and `B_r = contract_third_full(T³ᵣ, 0, 1)` once and close every axis with
    /// the same chunked `Xᵀ diag(w) X` / `Xᵀ diag(w) G` BLAS-3 machinery the
    /// first-directional override uses. Bit-for-bit the same entries the per-row
    /// `add_pullback_hessian` scatter writes, reduced in BLAS-3 in-row order, so
    /// axis `a` matches `row_kernel_directional_derivative(self, All, e_a)`.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` dense-design case;
    /// otherwise `None` → unchanged generic per-axis Horvitz-Thompson sweep.
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
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
        if !matches!(rows, crate::row_kernel::RowSet::All) {
            return None;
        }
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let logslope_sparse = self.family.logslope_design.is_sparse();
        if marginal_sparse || logslope_sparse {
            return None;
        }
        Some(self.directional_derivative_all_axes_blas3())
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
    /// in BLAS-3 in-row order. Claims only the full-data unit-weight
    /// `RowSet::All` dense-design case; otherwise `None` → unchanged generic
    /// per-row Horvitz-Thompson path.
    fn hessian_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        row_hessians: &[[[f64; 2]; 2]],
    ) -> Option<Result<Array2<f64>, String>> {
        // Only the full-data unit-weight measure is BLAS-3 accelerated; a
        // Horvitz-Thompson subsample keeps the generic per-row HT path.
        if !matches!(rows, crate::row_kernel::RowSet::All) {
            return None;
        }
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
        // for the biobank rigid fit regardless of whether the marginal/logslope
        // designs expose a pre-materialised `as_dense_ref`. Sparse designs are
        // the only structurally-inapplicable case; route those to the generic
        // per-row scatter so the design-row Gram never densifies a sparse block.
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let logslope_sparse = self.family.logslope_design.is_sparse();
        if marginal_sparse || logslope_sparse {
            // Diagnostic fires once per process, not once per inner-Newton kernel
            // call: `hessian_dense_override` runs on every joint-Hessian assembly,
            // so an unguarded line floods the biobank fit log.
            static H_NOT_TAKEN_LOGGED: std::sync::Once = std::sync::Once::new();
            H_NOT_TAKEN_LOGGED.call_once(|| {
                log::info!(
                    "[STAGE] BMS rigid hessian_dense BLAS-3 path NOT taken: sparse design \
                     (marginal_sparse={marginal_sparse} logslope_sparse={logslope_sparse}) \
                     -> generic per-row scatter"
                );
            });
            return None;
        }
        // Route an eligible whole-design joint Gram through one CUDA dispatch.
        // `Ok(None)` means CUDA was declined before execution (non-materialized
        // design, no runtime, or below policy); after admission, a missing
        // device result is an error and cannot select the CPU algorithm.
        match rigid_joint_hessian_on_gpu(
            &self.family.marginal_design,
            &self.family.logslope_design,
            row_hessians,
        ) {
            Ok(Some(joint)) => return Some(Ok(joint)),
            Ok(None) => {}
            Err(error) => return Some(Err(error)),
        }
        Some(self.hessian_dense_blas3(row_hessians))
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
    /// logslope-block axis `j` has `(0, G[r,j])` so its row weight is
    /// `G[r,j]·B_r`. Thus we read the cached fourth tensor and build `A_r, B_r`
    /// ONCE per row (hoisted out of the `p`-loop, the `~p×` reduction), then
    /// close each axis with the same chunked `Xᵀ diag(w) X` / `Xᵀ diag(w) G`
    /// BLAS-3 machinery the first-directional override uses
    /// (`add_weighted_design_grams_from_chunks`). Bit-for-bit the same entries
    /// the per-row `add_pullback_hessian` scatter writes, reduced in BLAS-3
    /// in-row order.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` dense-design case;
    /// otherwise `None` → unchanged generic per-axis Horvitz-Thompson sweep.
    fn second_directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        d_beta_u: &[f64],
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if !matches!(rows, crate::row_kernel::RowSet::All) {
            return None;
        }
        if d_beta_u.len() != self.slices.total {
            return None;
        }
        // Same structural gate as the first-directional override: the chunked
        // Gram machinery slices contiguous design rows via `try_row_chunk`,
        // which every dense-backed design (materialised OR operator-backed)
        // supports; only a sparse design block is structurally inapplicable.
        let marginal_sparse = self.family.marginal_design.is_sparse();
        let logslope_sparse = self.family.logslope_design.is_sparse();
        if marginal_sparse || logslope_sparse {
            return None;
        }
        Some(self.second_directional_derivative_all_axes_blas3(d_beta_u))
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
    logslope_design: &gam_linalg::matrix::DesignMatrix,
    row_hessians: &[[[f64; 2]; 2]],
) -> Result<Option<Array2<f64>>, String> {
    let Some(x_full) = marginal_design.as_dense_ref() else {
        return Ok(None);
    };
    let Some(g_full) = logslope_design.as_dense_ref() else {
        return Ok(None);
    };
    let rows = x_full.nrows();
    let logslope_rows = g_full.nrows();
    if rows != logslope_rows || rows != row_hessians.len() {
        return Err(format!(
            "BMS rigid joint-Hessian dimensions disagree: marginal_rows={rows}, \
             logslope_rows={logslope_rows}, row_hessians={}",
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
        let logslope_cols = g_full.ncols();
        let operation = gam_gpu::linalg_dispatch::DispatchOp::JointHessian2x2 {
            n: rows,
            pa: marginal_cols,
            pb: logslope_cols,
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
            logslope_cols,
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
    /// primary `2×2` row Hessian; `RowSet::All` (unit weights) is guaranteed by
    /// the caller. Materialization and any selected CUDA Gram execution report
    /// errors through the row-kernel dense-Hessian contract.
    fn hessian_dense_blas3(&self, row_hessians: &[[[f64; 2]; 2]]) -> Result<Array2<f64>, String> {
        let slices = &self.slices;
        let n = self.family.y.len();

        let chunk_rows = blas3_gram_chunk_rows(n);
        let chunks = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect::<Vec<_>>();
        // Each chunk slices a contiguous block of design rows. For a
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
                let x_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                    match self.family.marginal_design.as_dense_ref() {
                        Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                        None => self
                            .family
                            .marginal_design
                            .try_row_chunk(start..end)
                            .map_err(|e| {
                                format!(
                                    "bernoulli rigid hessian_dense_blas3 marginal_design \
                                     try_row_chunk({start}..{end}): {e}"
                                )
                            })?
                            .into(),
                    };
                let g_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                    match self.family.logslope_design.as_dense_ref() {
                        Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                        None => self
                            .family
                            .logslope_design
                            .try_row_chunk(start..end)
                            .map_err(|e| {
                                format!(
                                    "bernoulli rigid hessian_dense_blas3 logslope_design \
                                     try_row_chunk({start}..{end}): {e}"
                                )
                            })?
                            .into(),
                    };
                for row in start..end {
                    let local = row - start;
                    let h = &row_hessians[row];
                    w_mm[local] = h[0][0];
                    w_mg[local] = h[0][1];
                    w_gg[local] = h[1][1];
                }
                acc.add_weighted_design_grams_from_chunks(&x_chunk, &g_chunk, &w_mm, &w_mg, &w_gg)?;
                Ok(acc)
            })
        };

        let run_serial =
            rayon::current_thread_index().is_some() || rayon::current_num_threads() <= 1;
        if run_serial {
            let mut acc = BernoulliBlockHessianAccumulator::new(slices);
            for chunk in chunks {
                acc.add(&chunk_body(chunk)?);
            }
            return Ok(acc.to_dense(slices));
        }
        let acc = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            chunks.len(),
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
        )?
        .unwrap_or_else(|| BernoulliBlockHessianAccumulator::new(slices));
        Ok(acc.to_dense(slices))
    }

    /// Chunked BLAS-3 implementation backing
    /// [`RowKernel::directional_derivative_dense_override`].
    fn directional_derivative_dense_blas3(&self, d_beta: &[f64]) -> Result<Array2<f64>, String> {
        let slices = &self.slices;
        let n = self.family.y.len();
        let d_beta = ndarray::ArrayView1::from(d_beta);
        // Single-column `(p_block × 1)` direction blocks so the per-chunk
        // projection `X_chunk · dir` is one GEMM each (matching the per-row
        // `dot_row_view` the scalar path used).
        let marginal_dir_mat = d_beta
            .slice(s![slices.marginal.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        let logslope_dir_mat = d_beta
            .slice(s![slices.logslope.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        // Force the shared per-row third tensor build at top-level rayon before
        // any chunk fold (a single n-row par pass), so chunk bodies do an O(1)
        // lookup instead of triggering the build nested in a worker.
        let third_full = self.third_full_cache();

        let chunk_rows = blas3_gram_chunk_rows(n);
        let chunks = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect::<Vec<_>>();
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
                    let x_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                        match self.family.marginal_design.as_dense_ref() {
                            Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                            None => self
                                .family
                                .marginal_design
                                .try_row_chunk(start..end)
                                .map_err(|e| {
                                    format!("bernoulli marginal_design try_row_chunk: {e}")
                                })?
                                .into(),
                        };
                    let g_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                        match self.family.logslope_design.as_dense_ref() {
                            Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                            None => self
                                .family
                                .logslope_design
                                .try_row_chunk(start..end)
                                .map_err(|e| {
                                    format!("bernoulli logslope_design try_row_chunk: {e}")
                                })?
                                .into(),
                        };
                    let marginal_projected =
                        gam_linalg::faer_ndarray::fast_ab(&x_chunk, &marginal_dir_mat);
                    let logslope_projected =
                        gam_linalg::faer_ndarray::fast_ab(&g_chunk, &logslope_dir_mat);
                    for row in start..end {
                        let local = row - start;
                        let dq = marginal_projected[[local, 0]];
                        let dg = logslope_projected[[local, 0]];
                        let t = contract_third_full(&third_full[row], dq, dg);
                        w_mm[local] = t[0][0];
                        w_mg[local] = t[0][1];
                        w_gg[local] = t[1][1];
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
        // chunk loop when already inside a Rayon worker (the outer
        // joint-Newton / ψ-sweep par_iter holds the pool) so a nested
        // `into_par_iter` does not starve the pool — the same guard the batched
        // builder uses.
        let run_serial =
            rayon::current_thread_index().is_some() || rayon::current_num_threads() <= 1;
        if run_serial {
            let mut acc = BernoulliBlockHessianAccumulator::new(slices);
            for chunk in chunks {
                let partial = chunk_body(chunk)?;
                acc.add(&partial);
            }
            return Ok(acc.to_dense(slices));
        }
        let acc = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            chunks.len(),
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
        )?
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
    /// (`(X[r,j], 0)` for a marginal axis, `(0, G[r,j])` for a logslope axis,
    /// the exact value `jacobian_action(row, e_a)` returns). The Gram swap from
    /// BLAS-1 syr scatter to `fast_xt_diag_*` reduces in the identical in-row
    /// order (same contract as `hessian_dense_blas3`), so axis `a` matches
    /// `row_kernel_second_directional_derivative(self, All, u, e_a)`
    /// bit-for-bit.
    fn second_directional_derivative_all_axes_blas3(
        &self,
        d_beta_u: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = &self.slices;
        let n = self.family.y.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let d_beta_u = ndarray::ArrayView1::from(d_beta_u);
        // Fixed-direction blocks for the single-column `u`-projection GEMMs.
        let u_marg_mat = d_beta_u
            .slice(s![slices.marginal.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        let u_logs_mat = d_beta_u
            .slice(s![slices.logslope.clone()])
            .to_owned()
            .insert_axis(ndarray::Axis(1));
        // Force the shared per-row fourth tensor build at top-level rayon before
        // any chunk/axis fold, so the bodies do an O(1) lookup.
        let fourth_full = self.fourth_full_cache();
        crate::row_kernel::validate_row_kernel_cache_lengths(
            "bernoulli rigid second_directional_derivative_all_axes_blas3",
            n,
            &[("fourth", fourth_full.len())],
        )?;

        let chunk_rows = blas3_gram_chunk_rows(n);
        let chunks = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect::<Vec<_>>();

        // Hoisted per-row `u`-projection `(uq_r, ug_r)`, built ONCE via one
        // chunked GEMM per block. `uq[r] = X.row(r)·u_marg`, `ug[r] = G.row(r)·u_logs`
        // — bit-identical to `jacobian_action(row, d_beta_u)` (a single design-row
        // dot per axis), just batched.
        let mut uq = Array1::<f64>::zeros(n);
        let mut ug = Array1::<f64>::zeros(n);
        for &(start, end) in &chunks {
            gam_problem::with_nested_parallel(|| -> Result<(), String> {
                let x_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                    match self.family.marginal_design.as_dense_ref() {
                        Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                        None => self
                            .family
                            .marginal_design
                            .try_row_chunk(start..end)
                            .map_err(|e| format!("bernoulli marginal_design try_row_chunk: {e}"))?
                            .into(),
                    };
                let g_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                    match self.family.logslope_design.as_dense_ref() {
                        Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                        None => self
                            .family
                            .logslope_design
                            .try_row_chunk(start..end)
                            .map_err(|e| format!("bernoulli logslope_design try_row_chunk: {e}"))?
                            .into(),
                    };
                let uq_chunk = gam_linalg::faer_ndarray::fast_ab(&x_chunk, &u_marg_mat);
                let ug_chunk = gam_linalg::faer_ndarray::fast_ab(&g_chunk, &u_logs_mat);
                for row in start..end {
                    uq[row] = uq_chunk[[row - start, 0]];
                    ug[row] = ug_chunk[[row - start, 0]];
                }
                Ok(())
            })?;
        }

        // One axis = one independent full-data design-row Gram. Marginal axes
        // are `e_a` with the unit in the marginal block (axis projection
        // `(X[r,j], 0)`); logslope axes have it in the logslope block
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
                        let x_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                            match self.family.marginal_design.as_dense_ref() {
                                Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                                None => self
                                    .family
                                    .marginal_design
                                    .try_row_chunk(start..end)
                                    .map_err(|e| {
                                        format!("bernoulli marginal_design try_row_chunk: {e}")
                                    })?
                                    .into(),
                            };
                        let g_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                            match self.family.logslope_design.as_dense_ref() {
                                Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                                None => self
                                    .family
                                    .logslope_design
                                    .try_row_chunk(start..end)
                                    .map_err(|e| {
                                        format!("bernoulli logslope_design try_row_chunk: {e}")
                                    })?
                                    .into(),
                            };
                        let mut w_mm = Array1::<f64>::zeros(len);
                        let mut w_mg = Array1::<f64>::zeros(len);
                        let mut w_gg = Array1::<f64>::zeros(len);
                        for row in start..end {
                            let local = row - start;
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
                                &fourth_full[row],
                                uq[row],
                                ug[row],
                                vq,
                                vg,
                            );
                            w_mm[local] = m[0][0];
                            w_mg[local] = m[0][1];
                            w_gg[local] = m[1][1];
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
            rayon::current_thread_index().is_some() || rayon::current_num_threads() <= 1;
        if run_serial {
            (0..p_total).map(build_axis).collect::<Result<Vec<_>, _>>()
        } else {
            (0..p_total)
                .into_par_iter()
                .map(build_axis)
                .collect::<Result<Vec<_>, _>>()
        }
    }

    /// Chunked BLAS-3 implementation backing
    /// [`RowKernel::directional_derivative_all_axes_dense_override`].
    ///
    /// Returns the `p` dense matrices `{Hdot[e_a]}_{a=0..p}`. Each axis's per-row
    /// weight is `contract_third_full(T³ᵣ, vq_r, vg_r)` with `(vq_r, vg_r) =
    /// jacobian_action(row, e_a)` — `(X[r,j], 0)` for a marginal axis,
    /// `(0, G[r,j])` for a logslope axis. `contract_third_full` is LINEAR in
    /// `(vq, vg)`, so a marginal axis's weight is `X[r,j] · A_r` with
    /// `A_r = contract_third_full(T³ᵣ, 1, 0)`, and a logslope axis's is
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
    fn directional_derivative_all_axes_blas3(&self) -> Result<Vec<Array2<f64>>, String> {
        let slices = &self.slices;
        let n = self.family.y.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        // Force the shared per-row third tensor build at top-level rayon before
        // any chunk/axis fold, so the bodies do an O(1) lookup.
        let third_full = self.third_full_cache();
        crate::row_kernel::validate_row_kernel_cache_lengths(
            "bernoulli rigid directional_derivative_all_axes_blas3",
            n,
            &[("third", third_full.len())],
        )?;

        let chunk_rows = blas3_gram_chunk_rows(n);
        let chunks = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect::<Vec<_>>();

        // Per-row axis-independent partial contractions, built ONCE:
        //   A_r = contract_third_full(T³ᵣ, 1, 0)   (marginal-axis unit weight)
        //   B_r = contract_third_full(T³ᵣ, 0, 1)   (logslope-axis unit weight)
        // Each is a symmetric `2×2`; we keep the three independent entries
        // `(mm, mg, gg)` the design-row Gram consumes.
        let mut a_mm = Array1::<f64>::zeros(n);
        let mut a_mg = Array1::<f64>::zeros(n);
        let mut a_gg = Array1::<f64>::zeros(n);
        let mut b_mm = Array1::<f64>::zeros(n);
        let mut b_mg = Array1::<f64>::zeros(n);
        let mut b_gg = Array1::<f64>::zeros(n);
        for row in 0..n {
            let a = contract_third_full(&third_full[row], 1.0, 0.0);
            let b = contract_third_full(&third_full[row], 0.0, 1.0);
            a_mm[row] = a[0][0];
            a_mg[row] = a[0][1];
            a_gg[row] = a[1][1];
            b_mm[row] = b[0][0];
            b_mg[row] = b[0][1];
            b_gg[row] = b[1][1];
        }

        // One axis = one independent full-data design-row Gram. A marginal axis
        // `j` projects to `(X[r,j], 0)`, so its row weight is `X[r,j]·A_r`; a
        // logslope axis `j` projects to `(0, G[r,j])`, so its row weight is
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
                        let x_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                            match self.family.marginal_design.as_dense_ref() {
                                Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                                None => self
                                    .family
                                    .marginal_design
                                    .try_row_chunk(start..end)
                                    .map_err(|e| {
                                        format!("bernoulli marginal_design try_row_chunk: {e}")
                                    })?
                                    .into(),
                            };
                        let g_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                            match self.family.logslope_design.as_dense_ref() {
                                Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                                None => self
                                    .family
                                    .logslope_design
                                    .try_row_chunk(start..end)
                                    .map_err(|e| {
                                        format!("bernoulli logslope_design try_row_chunk: {e}")
                                    })?
                                    .into(),
                            };
                        let mut w_mm = Array1::<f64>::zeros(len);
                        let mut w_mg = Array1::<f64>::zeros(len);
                        let mut w_gg = Array1::<f64>::zeros(len);
                        for row in start..end {
                            let local = row - start;
                            // Axis projection scalar `s = jacobian_action(row, e_a)`
                            // in the active block, scaling the precomputed unit-axis
                            // contraction. `contract_third_full` is linear, so this
                            // equals `contract_third_full(T³ᵣ, vq_r, vg_r)` exactly.
                            if marginal_axis {
                                let s = x_chunk[[local, local_col]];
                                w_mm[local] = s * a_mm[row];
                                w_mg[local] = s * a_mg[row];
                                w_gg[local] = s * a_gg[row];
                            } else {
                                let s = g_chunk[[local, local_col]];
                                w_mm[local] = s * b_mm[row];
                                w_mg[local] = s * b_mg[row];
                                w_gg[local] = s * b_gg[row];
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
            rayon::current_thread_index().is_some() || rayon::current_num_threads() <= 1;
        if run_serial {
            (0..p_total).map(build_axis).collect::<Result<Vec<_>, _>>()
        } else {
            (0..p_total)
                .into_par_iter()
                .map(build_axis)
                .collect::<Result<Vec<_>, _>>()
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
