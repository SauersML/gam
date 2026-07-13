//! The bordered arrow-Schur system itself: [`ArrowRowBlock`], the
//! [`ArrowSchurSystem`] container and its assembly impl, cross-row latent
//! penalties, the streaming builder, and the per-row factor caches.

use super::*;

/// Per-row block data for the arrow-Schur system.
///
/// `htt` holds the `d × d` Gauss–Newton block for row `i` (including any
/// analytic-penalty contributions on that row); `htbeta` holds the
/// `d × K` cross-block `H_tβ^(i)`; `gt` is the `d`-length latent
/// gradient for row `i`.
#[derive(Debug, Clone)]
pub struct ArrowRowBlock {
    /// `H_tt^(i)`, shape `(d, d)`.
    pub htt: Array2<f64>,
    /// `H_tβ^(i)`, shape `(d, K)`.
    pub htbeta: Array2<f64>,
    /// `g_t^(i)`, shape `(d,)`.
    pub gt: Array1<f64>,
}

impl ArrowRowBlock {
    /// Allocate one BA point-block row: local latent Hessian, point-camera
    /// cross block, and point gradient.
    pub fn new(d: usize, k: usize) -> Self {
        Self::new_with_htbeta_cols(d, k)
    }

    /// Allocate one BA row whose dense cross-block slab has `htbeta_cols`
    /// columns. This is used by matrix-free assemblers that keep the shared
    /// beta tier at one width while dense row supplements live in another
    /// coordinate system.
    pub fn new_with_htbeta_cols(d: usize, htbeta_cols: usize) -> Self {
        Self {
            htt: Array2::<f64>::zeros((d, d)),
            htbeta: Array2::<f64>::zeros((d, htbeta_cols)),
            gt: Array1::<f64>::zeros(d),
        }
    }
}

/// Bordered (t, β) Newton system with arrow structure.
///
/// The β-block is held as a dense `K × K` Hessian `H_ββ` plus a `K`-length
/// gradient `g_β` for direct BA modes. Large-scale inexact BA callers may
/// additionally install a matrix-free `H_ββ x` operator and diagonal via
/// [`ArrowSchurSystem::set_shared_beta_operator`]; the InexactPCG mode then
/// avoids dense Schur formation/factorization.
/// The t-block is a `Vec<ArrowRowBlock>` of length `N`.
///
/// Construction is the driver's responsibility: the driver
///
///   1. evaluates Φ(t) and the radial jet `∂Φ/∂t` (the latter via
///      [`gam_terms::latent::LatentCoordValues::design_gradient_wrt_t`]);
///   2. forms the working-weighted Gauss–Newton blocks
///      `H_tt^(i) += (g_i β)(g_i β)^T`, `H_tβ^(i) += (g_i β) ⊗ Φ_i`,
///      `H_ββ += Φ^T W Φ + Σ_k λ_k S_k`;
///   3. calls [`ArrowSchurSystem::add_analytic_penalty_contributions`] to
///      fold row-block Psi-tier analytic penalties (`ARDPenalty`,
///      `SparsityPenalty`) into `H_tt^(i)` and Beta-tier penalties into `H_ββ`;
///   4. calls [`ArrowSchurSystem::solve`] to obtain `(Δt, Δβ)`.
pub struct ArrowSchurSystem {
    /// Per-row latent block (length `N`, each row `d × d` / `d × K` / `d`).
    pub rows: Vec<ArrowRowBlock>,
    /// `H_ββ`, shape `(K, K)` for direct BA modes; empty when constructed
    /// by [`ArrowSchurSystem::new_matrix_free_shared`] for PCG-only use.
    pub hbb: Array2<f64>,
    /// Optional matrix-free `H_ββ x` operator for large BA Schur PCG.
    ///
    /// Direct and Square-Root BA modes still require `hbb`; InexactPCG uses
    /// this operator when present, avoiding dense shared-block storage for
    /// SAE-manifold scale `K`.
    pub hbb_matvec: Option<SharedBetaMatvec>,
    /// Optional row-local matrix-free multiply for `H_tβ^(i) x`.
    ///
    /// When present, all inner-Schur paths route through this operator instead
    /// of indexing the per-row `htbeta` dense slabs: `reduced_rhs_beta`,
    /// `schur_matvec` (PCG hot loop), back-substitution,
    /// `JacobiPreconditioner` construction, `build_dense_schur_direct`, and
    /// `build_dense_schur_sqrt_ba` all call `sys_htbeta_apply_row` or
    /// `sys_htbeta_materialize_row`.  Factor caches retain the operator for
    /// IFT/evidence consumers as before.
    pub htbeta_matvec: Option<RowHtbetaMatvec>,
    /// Optional row-local matrix-free transpose multiply `out += H_βt^(i) · v`.
    ///
    /// The sparse adjoint of [`Self::htbeta_matvec`]. When present, the
    /// reduced-Schur matvec applies `H_βt^(i)` directly (sparse `scatter`)
    /// instead of probing the forward operator against `K` basis vectors. This
    /// is the per-row sparse apply that lifts the `O(K)` column-probe in the
    /// GPU PCG and streaming Schur paths to `O(m_i · p)` per row. Installed in
    /// lock-step with `htbeta_matvec` by [`Self::set_row_htbeta_operator`].
    pub htbeta_transpose_matvec: Option<RowHtbetaTransposeMatvec>,
    /// Whether `rows[*].htbeta` contains a dense contribution that must be added
    /// on top of the matrix-free row operator.
    pub htbeta_dense_supplement: bool,
    /// Optional diagonal of the matrix-free shared block, used by the
    /// Schur-Jacobi preconditioner in the Agarwal-style PCG path.
    pub hbb_diag: Option<Array1<f64>>,
    /// `g_β`, shape `(K,)`.
    pub gb: Array1<f64>,
    /// Maximum per-row latent dimensionality across all rows.
    ///
    /// For homogeneous systems (all rows have the same dim) this equals the
    /// common per-row `d`.  For heterogeneous systems (e.g. sparse SAE rows
    /// where JumpReLU / TopK / sparsemax active sets vary per observation)
    /// this is `max_i row_dims[i]`.  Per-row code should use
    /// `row.htt.nrows()` or `row_dims[i]`; `d` is an upper bound for
    /// scratch-buffer sizing.
    pub d: usize,
    /// Per-row latent dimensionality: `row_dims[i] == rows[i].htt.nrows()`.
    ///
    /// For homogeneous systems `row_dims[i] == d` for all `i`.
    pub row_dims: Arc<[usize]>,
    /// Flat-buffer row offsets for the `delta_t` vector produced by
    /// [`Self::solve`] / [`solve_arrow_newton_step_core`].
    ///
    /// `row_offsets[i]` is the start index for row `i`'s slice in `delta_t`;
    /// `row_offsets[n]` is the total `delta_t` length.  For homogeneous
    /// systems `row_offsets[i] == i * d`.
    pub row_offsets: Arc<[usize]>,
    /// β dimensionality `K`.
    pub k: usize,
    /// Geometry tag for the row-local latent blocks after optional
    /// Riemannian projection. Euclidean/no-op geometry uses the sentinel.
    pub manifold_mode_fingerprint: u64,
    /// Structural/value tag for row-local Hessian factors and their Schur
    /// inputs. Stale caches must be rejected when row-dependent Hessian
    /// penalties or cross-blocks change.
    pub row_hessian_fingerprint: u64,
    /// Registry-side tag for row-dependent analytic-penalty Hessian inputs.
    /// Combined with the materialized row blocks in
    /// [`Self::current_row_hessian_fingerprint`].
    pub analytic_row_hessian_fingerprint: u64,
    /// Term-block column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// Each entry `r` means that indices `r.start..r.end` belong to one
    /// coefficient block (a GAM term or a custom parameter family from
    /// `ParameterBlockSpec`). When populated via
    /// [`Self::set_block_offsets`], the Jacobi preconditioner inverts the
    /// full `b × b` Schur block for each term instead of only its diagonal.
    ///
    /// The default (empty slice) causes `JacobiPreconditioner` to fall back
    /// to pure scalar diagonal inversion, preserving the pre-#283 behaviour.
    pub block_offsets: Arc<[Range<usize>]>,
    /// Optional matrix-free penalty-side `H_ββ` operator (#296).
    ///
    /// When set, all hot paths (`schur_matvec`, `build_dense_schur_*`,
    /// `JacobiPreconditioner`, quadratic-form reduction) route through this
    /// operator instead of the dense `hbb` accumulator, enabling
    /// `BlockPenaltyOp` / `KroneckerPenaltyOp` to skip the `O(K²)` dense
    /// materialisation for structured smoothness penalties.
    ///
    /// When `None`, those paths fall back to wrapping `hbb` in a transient
    /// `DensePenaltyOp` — identical observable behaviour, no new allocation
    /// hot-path cost for callers that have not opted in.
    pub penalty_op: Option<Arc<dyn BetaPenaltyOp>>,
    /// Device-uploadable SAE Kronecker data for CUDA-resident reduced PCG.
    ///
    /// The generic matrix-free closures remain the authoritative CPU path. This
    /// descriptor is installed only when SAE assembly has a matching CUDA sparse
    /// representation for both `H_tβ` and `H_ββ`.
    pub device_sae_pcg: Option<Arc<DeviceSaePcgData>>,
    /// Registered Psi-tier analytic penalties whose Hessian couples *distinct*
    /// latent rows (non-row-block-diagonal), captured by
    /// [`Self::add_analytic_penalty_contributions`].
    ///
    /// These penalties (`TotalVariationPenalty`, `SheafConsistencyPenalty`,
    /// block-orthogonality, …) produce off-row Hessian blocks `∂²P/∂t_i∂t_j`
    /// (`i ≠ j`) that the arrow elimination — which assumes each `H_tt^(i)` is
    /// independent of every other row — cannot represent. Their *gradient* is
    /// still folded into `g_t` exactly like every other Psi penalty; only their
    /// curvature is held here, applied during the solve as a full-latent
    /// Hessian-vector product `P_cross · Δt` against the penalty's
    /// `psd_majorizer_hvp`. When this vector is non-empty,
    /// [`solve_arrow_newton_step_artifacts`] auto-selects the matrix-free
    /// full-system PCG path (arrow block-diagonal inverse as preconditioner)
    /// instead of the exact one-shot Schur elimination. When empty, the system
    /// is purely row-block-diagonal and the exact Schur path is unchanged.
    pub cross_row_penalties: Vec<CrossRowLatentPenalty>,
    /// Optional row-local gauge directions for evidence-only Faddeev-Popov
    /// deflation of an otherwise non-PD `H_tt` row block.
    ///
    /// These vectors live in each row's actual chart block, so compact SAE rows
    /// and dense rows share the same factorization path. Ordinary Newton solves
    /// ignore them; only undamped evidence factors with
    /// evidence factorization may stiffen a gauge-explained row
    /// direction.
    pub row_gauge_deflation: Option<ArrowRowGaugeDeflation>,
    /// Exact scale-gauge quotient on the reduced shared `beta` border.
    ///
    /// SAE installs one normalized radial decoder direction per live atom.
    /// Evidence paths factor `P S P + Q Q^T` and expose the projected inverse
    /// `P S_quot^-1 P`; ordinary Newton steps ignore this carrier because their
    /// joint `(delta B, delta log-amplitude)` trajectory projection is owned by
    /// the SAE step application.
    pub beta_gauge_quotient: Option<ArrowBetaGaugeQuotient>,
}

impl Clone for ArrowSchurSystem {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows.clone(),
            hbb: self.hbb.clone(),
            hbb_matvec: self.hbb_matvec.clone(),
            htbeta_matvec: self.htbeta_matvec.clone(),
            htbeta_transpose_matvec: self.htbeta_transpose_matvec.clone(),
            htbeta_dense_supplement: self.htbeta_dense_supplement,
            hbb_diag: self.hbb_diag.clone(),
            gb: self.gb.clone(),
            d: self.d,
            row_dims: Arc::clone(&self.row_dims),
            row_offsets: Arc::clone(&self.row_offsets),
            k: self.k,
            manifold_mode_fingerprint: self.manifold_mode_fingerprint,
            row_hessian_fingerprint: self.row_hessian_fingerprint,
            analytic_row_hessian_fingerprint: self.analytic_row_hessian_fingerprint,
            block_offsets: Arc::clone(&self.block_offsets),
            penalty_op: self.penalty_op.clone(),
            device_sae_pcg: self.device_sae_pcg.clone(),
            cross_row_penalties: self.cross_row_penalties.clone(),
            row_gauge_deflation: self.row_gauge_deflation.clone(),
            beta_gauge_quotient: self.beta_gauge_quotient.clone(),
        }
    }
}

/// A captured cross-row Psi-tier analytic penalty: the penalty kind plus the
/// global-ρ slice (`rho_local`) it was registered with.
///
/// Holds an owned copy of the local ρ-axes so the penalty's
/// [`AnalyticPenaltyKind::psd_majorizer_hvp`] can be evaluated during the
/// matrix-free full-system solve without re-deriving the ρ layout. The penalty
/// itself is an `Arc`-backed clone (cheap), so capturing it does not copy the
/// penalty payload.
#[derive(Clone)]
pub struct CrossRowLatentPenalty {
    /// The non-row-block-diagonal Psi penalty (e.g. `TotalVariationPenalty`).
    pub penalty: AnalyticPenaltyKind,
    /// The penalty's local ρ-axes (its slice of the global ρ vector).
    pub rho_local: Array1<f64>,
    /// The flat latent vector (`N·d`, row-major) the penalty's curvature was
    /// linearized at — i.e. the `target_t` passed to
    /// [`ArrowSchurSystem::add_analytic_penalty_contributions`]. The Hessian of
    /// a nonlinear penalty (the smoothed-TV curvature weights `φ''(D t)`,
    /// etc.) depends on this point, so `psd_majorizer_hvp` must be evaluated
    /// against it for the Newton operator to be the true Hessian at the
    /// current iterate.
    pub target_t: Array1<f64>,
}

impl ArrowSchurSystem {
    /// Allocate an empty BA reduced-camera-system instance sized
    /// `(N point/latent rows × d, K shared decoder parameters)`.
    pub fn new(n: usize, d: usize, k: usize) -> Self {
        Self::new_with_hbb(n, d, k, Array2::<f64>::zeros((k, k)))
    }

    /// Allocate an arrow system with no dense shared `H_ββ` block and with
    /// per-row dense `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_empty_hbb_and_htbeta_cols(
        n: usize,
        d: usize,
        k: usize,
        htbeta_cols: usize,
    ) -> Self {
        let rows = (0..n)
            .map(|_| ArrowRowBlock::new_with_htbeta_cols(d, htbeta_cols))
            .collect();
        let row_dims: Arc<[usize]> = (0..n).map(|_| d).collect::<Vec<_>>().into();
        let row_offsets: Arc<[usize]> = (0..=n).map(|i| i * d).collect::<Vec<_>>().into();
        Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d,
            row_dims,
            row_offsets,
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            beta_gauge_quotient: None,
        }
    }

    /// Allocate an arrow system using a caller-owned dense shared-block buffer.
    /// The buffer must already have shape `(k, k)` and is zeroed in place before
    /// use so callers can recycle it across assemblies without changing
    /// numerics.
    pub fn new_with_hbb(n: usize, d: usize, k: usize, hbb: Array2<f64>) -> Self {
        Self::new_with_hbb_and_htbeta_cols(n, d, k, hbb, k)
    }

    /// Allocate an arrow system with a caller-owned dense shared-block buffer and
    /// per-row dense `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_hbb_and_htbeta_cols(
        n: usize,
        d: usize,
        k: usize,
        mut hbb: Array2<f64>,
        htbeta_cols: usize,
    ) -> Self {
        assert_eq!(hbb.dim(), (k, k));
        hbb.fill(0.0);
        let rows = (0..n)
            .map(|_| ArrowRowBlock::new_with_htbeta_cols(d, htbeta_cols))
            .collect();
        let row_dims: Arc<[usize]> = (0..n).map(|_| d).collect::<Vec<_>>().into();
        let row_offsets: Arc<[usize]> = (0..=n).map(|i| i * d).collect::<Vec<_>>().into();
        Self {
            rows,
            hbb,
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d,
            row_dims,
            row_offsets,
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            beta_gauge_quotient: None,
        }
    }

    /// Allocate an arrow system whose shared `H_ββ` block is supplied only as
    /// a matrix-free operator for large BA InexactPCG.
    ///
    /// Direct and Square-Root BA modes require dense `hbb` and must not be
    /// used with this constructor. The row-local `H_tβ` slabs remain explicit;
    /// a future MegBA backend can replace those slab operations behind
    /// [`BatchedBlockSolver`].
    pub fn new_matrix_free_shared<F>(
        n: usize,
        d: usize,
        k: usize,
        matvec: F,
        diag: Array1<f64>,
    ) -> Self
    where
        F: for<'a> Fn(ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
    {
        assert_eq!(diag.len(), k);
        let rows = (0..n).map(|_| ArrowRowBlock::new(d, k)).collect();
        let row_dims: Arc<[usize]> = (0..n).map(|_| d).collect::<Vec<_>>().into();
        let row_offsets: Arc<[usize]> = (0..=n).map(|i| i * d).collect::<Vec<_>>().into();
        let matvec_arc: SharedBetaMatvec = Arc::new(matvec);
        // Mirror the closure into a BetaPenaltyOp so all hot paths (#296)
        // route through the trait while preserving hbb_matvec + hbb_diag for
        // code that inspects them directly.
        let penalty_op: Option<Arc<dyn BetaPenaltyOp>> = Some(Arc::new(MatvecDiagPenaltyOp::new(
            k,
            Arc::clone(&matvec_arc),
            diag.clone(),
        )));
        Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: Some(matvec_arc),
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
            hbb_diag: Some(diag),
            gb: Array1::<f64>::zeros(k),
            d,
            row_dims,
            row_offsets,
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op,
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            beta_gauge_quotient: None,
        }
    }

    /// Allocate a heterogeneous-row arrow system with no dense shared `H_ββ`
    /// block and with row `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_per_row_dims_empty_hbb_and_htbeta_cols(
        per_row_dims: Vec<usize>,
        k: usize,
        htbeta_cols: usize,
    ) -> Self {
        let n = per_row_dims.len();
        let d = per_row_dims.iter().copied().max().unwrap_or(0);
        let mut offsets = Vec::with_capacity(n + 1);
        let mut cursor = 0usize;
        offsets.push(cursor);
        for &dim in &per_row_dims {
            cursor += dim;
            offsets.push(cursor);
        }
        let rows = per_row_dims
            .iter()
            .map(|&dim| ArrowRowBlock::new_with_htbeta_cols(dim, htbeta_cols))
            .collect();
        Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d,
            row_dims: Arc::from(per_row_dims.into_boxed_slice()),
            row_offsets: Arc::from(offsets.into_boxed_slice()),
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            beta_gauge_quotient: None,
        }
    }

    /// Allocate a heterogeneous-row system using a caller-owned dense shared
    /// block and row `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_per_row_dims_and_hbb_and_htbeta_cols(
        per_row_dims: Vec<usize>,
        k: usize,
        mut hbb: Array2<f64>,
        htbeta_cols: usize,
    ) -> Self {
        assert_eq!(hbb.dim(), (k, k));
        hbb.fill(0.0);
        let n = per_row_dims.len();
        let max_d = per_row_dims.iter().copied().max().unwrap_or(0);
        let row_dims: Arc<[usize]> = per_row_dims.iter().copied().collect::<Vec<_>>().into();
        let mut off_vec = Vec::with_capacity(n + 1);
        let mut cursor = 0usize;
        for &di in &per_row_dims {
            off_vec.push(cursor);
            cursor += di;
        }
        off_vec.push(cursor);
        let row_offsets: Arc<[usize]> = off_vec.into();
        let rows = per_row_dims
            .iter()
            .map(|&di| ArrowRowBlock::new_with_htbeta_cols(di, htbeta_cols))
            .collect();
        Self {
            rows,
            hbb,
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d: max_d,
            row_dims,
            row_offsets,
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            beta_gauge_quotient: None,
        }
    }

    /// Build a fresh numerical system while reusing caller-owned assembly
    /// allocations when their shapes still match.
    ///
    /// This is deliberately an *allocation* workspace, not a factor cache:
    /// every entry of `rows`, `hbb`, and `gb` is zeroed before the system is
    /// returned, and all operator/fingerprint/device fields start empty. A
    /// nonlinear assembler can therefore refill every state-dependent block at
    /// the new iterate without paying again for the stable row/shared-buffer
    /// shapes. Shape changes discard only the incompatible allocation.
    pub fn new_with_assembly_buffers(
        per_row_dims: Vec<usize>,
        k: usize,
        htbeta_cols: usize,
        mut hbb: Array2<f64>,
        mut rows: Vec<ArrowRowBlock>,
        mut gb: Array1<f64>,
    ) -> Self {
        assert!(hbb.dim() == (0, 0) || hbb.dim() == (k, k));
        hbb.fill(0.0);

        let rows_match = rows.len() == per_row_dims.len()
            && rows.iter().zip(&per_row_dims).all(|(row, &dim)| {
                row.htt.dim() == (dim, dim)
                    && row.htbeta.dim() == (dim, htbeta_cols)
                    && row.gt.len() == dim
            });
        if rows_match {
            for row in &mut rows {
                row.htt.fill(0.0);
                row.htbeta.fill(0.0);
                row.gt.fill(0.0);
            }
        } else {
            rows = per_row_dims
                .iter()
                .map(|&dim| ArrowRowBlock::new_with_htbeta_cols(dim, htbeta_cols))
                .collect();
        }
        if gb.len() == k {
            gb.fill(0.0);
        } else {
            gb = Array1::<f64>::zeros(k);
        }

        let n = per_row_dims.len();
        let d = per_row_dims.iter().copied().max().unwrap_or(0);
        let mut offsets = Vec::with_capacity(n + 1);
        let mut cursor = 0usize;
        offsets.push(cursor);
        for &dim in &per_row_dims {
            cursor += dim;
            offsets.push(cursor);
        }
        Self {
            rows,
            hbb,
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
            hbb_diag: None,
            gb,
            d,
            row_dims: Arc::from(per_row_dims.into_boxed_slice()),
            row_offsets: Arc::from(offsets.into_boxed_slice()),
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            beta_gauge_quotient: None,
        }
    }

    pub fn set_row_gauge_deflation(&mut self, deflation: ArrowRowGaugeDeflation) {
        self.row_gauge_deflation = Some(deflation);
    }

    /// Install the exact evidence quotient for shared-border gauge directions.
    pub fn set_beta_gauge_quotient(
        &mut self,
        quotient: ArrowBetaGaugeQuotient,
    ) -> Result<(), String> {
        if quotient.border_dim() != self.k {
            return Err(format!(
                "ArrowSchurSystem::set_beta_gauge_quotient: direction width {} != beta border {}",
                quotient.border_dim(),
                self.k
            ));
        }
        self.beta_gauge_quotient = Some(quotient);
        Ok(())
    }

    /// Number of BA point/latent rows `N`.
    pub fn n(&self) -> usize {
        self.rows.len()
    }

    /// Recompute the row-system fingerprint from the currently materialized
    /// row blocks, cross-blocks, and shared-block diagonal.
    pub fn compute_row_hessian_fingerprint(&self) -> u64 {
        row_hessian_fingerprint_for_system(self)
    }

    /// Current effective row-system fingerprint, including the materialized
    /// row blocks and any registry metadata captured while folding analytic
    /// penalties into the system.
    pub fn current_row_hessian_fingerprint(&self) -> u64 {
        combine_row_and_registry_fingerprints(
            self.compute_row_hessian_fingerprint(),
            self.analytic_row_hessian_fingerprint,
        )
    }

    /// Store the current row-system fingerprint on the system.
    ///
    /// This is intentionally explicit and expensive. Cache and evidence callers
    /// use [`Self::current_row_hessian_fingerprint`] at the point they need the
    /// value, after assembly has populated the system, instead of hashing each
    /// intermediate construction/mutation step.
    pub fn refresh_row_hessian_fingerprint(&mut self) {
        self.row_hessian_fingerprint = self.current_row_hessian_fingerprint();
    }

    /// Install a matrix-free shared-block operator for Agarwal-style
    /// inexact Schur PCG.
    ///
    /// `diag` must be the diagonal of the same `H_ββ` operator and is used
    /// for the Schur-Jacobi preconditioner. This is the BA "large camera
    /// system" path mapped to large decoder coefficient blocks.
    pub fn set_shared_beta_operator<F>(&mut self, matvec: F, diag: Array1<f64>)
    where
        F: for<'a> Fn(ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
    {
        assert_eq!(diag.len(), self.k);
        let matvec_arc: SharedBetaMatvec = Arc::new(matvec);
        // Mirror the closure into a BetaPenaltyOp so all hot paths (#296)
        // route through the trait, preserving the existing hbb_matvec +
        // hbb_diag fields for code that inspects them directly.
        self.penalty_op = Some(Arc::new(MatvecDiagPenaltyOp::new(
            self.k,
            Arc::clone(&matvec_arc),
            diag.clone(),
        )));
        self.hbb_matvec = Some(matvec_arc);
        self.hbb_diag = Some(diag);
    }

    /// Mark the dense per-row cross-block slabs as active supplements to the
    /// installed matrix-free row operator.
    pub fn activate_dense_htbeta_supplement(&mut self) {
        self.htbeta_dense_supplement = true;
    }

    /// Install a matrix-free per-row cross-block operator and its sparse
    /// adjoint.
    ///
    /// `forward` must write `out = H_tβ^(row) x` for `out.len() == d` and
    /// `x.len() == K`. `transpose` must **add** `H_βt^(row) v` into `out` for
    /// `out.len() == K` and `v.len() == d` (the sparse `scatter` adjoint).
    ///
    /// When installed, the forward operator is used during the Newton solve
    /// (inside `reduced_rhs_beta`, `schur_matvec`, back-substitution, and
    /// `JacobiPreconditioner` construction) and afterwards by IFT/evidence
    /// predictors.  Per-row `htbeta` slabs in `ArrowRowBlock` may be left
    /// zero-sized when this operator is installed — all inner-Schur paths route
    /// through the matvec instead of indexing the dense block. The transpose
    /// operator lets the reduced-Schur matvec apply `H_βt^(row)` directly
    /// (`O(m_i · p)`) instead of probing `forward` against `K` basis vectors.
    pub fn set_row_htbeta_operator<F, T>(&mut self, forward: F, transpose: T)
    where
        F: for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
        T: for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
    {
        self.htbeta_matvec = Some(Arc::new(forward));
        self.htbeta_transpose_matvec = Some(Arc::new(transpose));
    }

    /// Register term-block column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// Each `Range<usize>` covers the columns of one GAM term (or custom
    /// parameter family) in the shared `β` vector. The ranges must be
    /// non-overlapping, sorted, and their union must cover `0..k`.
    ///
    /// Call this after building the system and before [`Self::solve`] /
    /// [`Self::solve_with_options`] whenever the solver will use
    /// [`ArrowSolverMode::InexactPCG`]. Absent a call, the preconditioner
    /// falls back to scalar diagonal Jacobi (the pre-#283 behaviour).
    ///
    /// The same plumbing is compatible with #287 (custom `ParameterBlockSpec`
    /// families): callers from that path simply supply ranges derived from
    /// their own block layout.
    pub fn set_block_offsets(&mut self, offsets: Arc<[Range<usize>]>) {
        self.block_offsets = offsets;
    }

    /// Install a matrix-free penalty-side `H_ββ` operator (#296).
    ///
    /// When set, all hot paths (`schur_matvec`, `build_dense_schur_*`,
    /// `JacobiPreconditioner`, quadratic-form reduction) route through this
    /// operator instead of the dense `hbb` accumulator, enabling
    /// `BlockPenaltyOp` / `KroneckerPenaltyOp` to avoid `O(K²)` allocation
    /// for structured smoothness penalties.
    pub fn set_penalty_op(&mut self, op: Arc<dyn BetaPenaltyOp>) {
        self.penalty_op = Some(op);
    }

    pub fn set_device_sae_pcg_data(&mut self, data: DeviceSaePcgData) {
        self.set_device_sae_pcg_data_reusing(data, None);
    }

    /// Install an already allocation-resident SAE device descriptor.
    pub fn set_device_sae_pcg_allocation(&mut self, data: Arc<DeviceSaePcgData>) {
        assert_eq!(data.beta_dim, self.k);
        if data.frame.is_none() {
            assert_eq!(data.a_phi.len(), self.rows.len());
            assert_eq!(data.local_jac.len(), self.rows.len());
        }
        self.device_sae_pcg = Some(data);
    }

    /// Install current-iterate SAE device operands while retaining the outer
    /// descriptor allocation from a completed prior assembly when it is
    /// uniquely owned. Framed payloads also refill their nested row-cross/frame
    /// vectors through `Vec::clone_from`, retaining matching capacities. `data`
    /// still replaces every numerical value, so no state-dependent operand or
    /// factor crosses nonlinear iterates.
    pub fn set_device_sae_pcg_data_reusing(
        &mut self,
        data: DeviceSaePcgData,
        recycled: Option<Arc<DeviceSaePcgData>>,
    ) {
        assert_eq!(data.beta_dim, self.k);
        // The frames-engaged builder (`build_framed_device_sae_data`) carries the
        // per-row cross block through `frame.frame_blocks` and intentionally leaves
        // the full-`B` `a_phi`/`local_jac` slabs EMPTY (#1033). Only the non-framed
        // full-`B` path populates those per-row slabs, so the length contract
        // applies only when there is no frame.
        if data.frame.is_none() {
            assert_eq!(data.a_phi.len(), self.rows.len());
            assert_eq!(data.local_jac.len(), self.rows.len());
        }
        let allocation = match recycled {
            Some(mut allocation) => match Arc::get_mut(&mut allocation) {
                Some(slot) => {
                    slot.replace_reusing_framed_allocations(data);
                    allocation
                }
                None => Arc::new(data),
            },
            None => Arc::new(data),
        };
        self.set_device_sae_pcg_allocation(allocation);
    }

    /// Return the effective penalty operator: the installed `penalty_op` if
    /// present, otherwise a `DensePenaltyOp` wrapping the current `hbb`.
    ///
    /// Note: when `penalty_op` is `None`, this clones `hbb` into a new
    /// `DensePenaltyOp`. Callers in hot loops should call this once and
    /// store the result, not call it per-iteration.
    pub fn effective_penalty_op(&self) -> Arc<dyn BetaPenaltyOp> {
        match self.penalty_op.as_ref() {
            Some(op) => Arc::clone(op),
            None => Arc::new(DensePenaltyOp(self.hbb.clone())),
        }
    }

    /// `y += P x` without allocating a new Arc; dispatches to `penalty_op`
    /// or falls back to `hbb` inline, avoiding the K×K clone hot-path cost.
    #[inline]
    pub(crate) fn penalty_matvec_add(&self, x: &[f64], y: &mut [f64]) {
        if let Some(op) = self.penalty_op.as_ref() {
            op.matvec(x, y);
        } else {
            let k = self.hbb.nrows();
            for a in 0..k {
                let mut acc = 0.0_f64;
                for b in 0..k {
                    acc += self.hbb[[a, b]] * x[b];
                }
                y[a] += acc;
            }
        }
    }

    /// Reduced-Schur matvec prologue `y = (P + ridge·I) x` written fresh into a
    /// zeroed `y` (the caller clears `out` first; this is the first writer).
    ///
    /// At the SAE LLM border width (#1017) the dense `H_ββ` fallback is a `k×k`
    /// GEMV whose `O(k²)` cost (≈4M flops at k=2048) runs once per CG iteration
    /// and was the serial Amdahl ceiling on the per-row-parallel matvec: while
    /// the `n`-row point-elimination term fans across all cores, this prologue
    /// pinned one core and grows as `k²`. The dense GEMV is embarrassingly
    /// parallel over output rows `a` — each `y[a] = Σ_b hbb[a,b]·x[b] + ridge·x[a]`
    /// is independent and its inner sum order is identical whether one thread or
    /// many compute it. Here parallelism is over independent output rows (NOT a
    /// reassociated reduction), so each `y[a]` accumulates in the SAME order as
    /// serial — the result is bit-identical to serial, not merely deterministic
    /// run-to-run (the #1017 determinism gate). On THIS exact-order path the
    /// criterion ranking is invariant; that no-move guarantee holds because the
    /// order matches serial, and does NOT generalise to chunk-reassociated
    /// reductions, where a near-tie winner can flip within the f64 margin
    /// (#1211). The `penalty_op` path stays serial — it is an opaque operator
    /// with its own structure (SAE uses the dense `hbb`), and small `k` stays
    /// serial to avoid rayon overhead on a trivial GEMV.
    ///
    /// `parallel` is the caller's top-level / not-nested-in-rayon decision (the
    /// same guard the row loop uses), so this never oversubscribes inside the
    /// topology race.
    pub(crate) fn penalty_ridge_prologue_into(
        &self,
        x: &[f64],
        ridge: f64,
        y: &mut [f64],
        parallel: bool,
    ) {
        let k = self.hbb.nrows();
        let dense_parallel = parallel
            && self.penalty_op.is_none()
            && self.hbb.dim() == (k, k)
            && k >= SCHUR_PROLOGUE_PARALLEL_K_MIN;
        if dense_parallel {
            use rayon::prelude::*;
            let hbb = &self.hbb;
            y.par_iter_mut().enumerate().for_each(|(a, ya)| {
                let mut acc = 0.0_f64;
                for b in 0..k {
                    acc += hbb[[a, b]] * x[b];
                }
                *ya = acc + ridge * x[a];
            });
        } else {
            self.penalty_matvec_add(x, y);
            for a in 0..k {
                y[a] += ridge * x[a];
            }
        }
    }

    /// `diag += diag(P)` without allocating; dispatches to `penalty_op`
    /// or falls back to `hbb` diagonal / `hbb_diag` inline.
    #[inline]
    pub(crate) fn penalty_diagonal_add(&self, diag: &mut [f64]) {
        if let Some(op) = self.penalty_op.as_ref() {
            op.diagonal(diag);
        } else if let Some(hbb_diag) = self.hbb_diag.as_ref() {
            let k = hbb_diag.len().min(diag.len());
            for j in 0..k {
                diag[j] += hbb_diag[j];
            }
        } else {
            let k = self.hbb.nrows().min(diag.len());
            for j in 0..k {
                diag[j] += self.hbb[[j, j]];
            }
        }
    }

    /// Add the `b×b` penalty sub-block for `id` to `out`, routing through
    /// `penalty_op` or falling back to `hbb` / `hbb_diag` inline.
    #[inline]
    pub(crate) fn penalty_block_add(
        &self,
        id: BetaBlockId,
        offsets: &[Range<usize>],
        out: &mut Array2<f64>,
    ) {
        if let Some(op) = self.penalty_op.as_ref() {
            op.block(id, offsets, out);
        } else {
            let range = &offsets[id.0];
            let b = range.end - range.start;
            if self.hbb.dim() == (self.k, self.k) {
                for bi in 0..b {
                    for bj in 0..b {
                        out[[bi, bj]] += self.hbb[[range.start + bi, range.start + bj]];
                    }
                }
            } else if let Some(hbb_diag) = self.hbb_diag.as_ref() {
                for bi in 0..b {
                    out[[bi, bi]] += hbb_diag[range.start + bi];
                }
            }
        }
    }

    /// Fill a `b×b` penalty sub-block for a set of arbitrary (possibly
    /// non-contiguous) global column indices `cols`, routing through
    /// `penalty_op` or falling back to `hbb` / `hbb_diag` inline.
    ///
    /// Used by the cluster-Jacobi preconditioner (#299) which groups columns
    /// by spectral adjacency rather than contiguous block ranges.
    #[inline]
    pub(crate) fn penalty_subblock_add(&self, cols: &[usize], out: &mut Array2<f64>) {
        let b = cols.len();
        if let Some(op) = self.penalty_op.as_ref() {
            // Probe each column basis vector and extract the sub-block entries.
            let mut probe = Array1::<f64>::zeros(self.k);
            let mut result = Array1::<f64>::zeros(self.k);
            for bj in 0..b {
                probe.fill(0.0);
                probe[cols[bj]] = 1.0;
                result.fill(0.0);
                {
                    let p_slice = probe.as_slice().expect("probe contiguous");
                    let r_slice = result.as_slice_mut().expect("result contiguous");
                    op.matvec(p_slice, r_slice);
                }
                for bi in 0..b {
                    out[[bi, bj]] += result[cols[bi]];
                }
            }
        } else if self.hbb.dim() == (self.k, self.k) {
            for bi in 0..b {
                for bj in 0..b {
                    out[[bi, bj]] += self.hbb[[cols[bi], cols[bj]]];
                }
            }
        } else if let Some(hbb_diag) = self.hbb_diag.as_ref() {
            for bi in 0..b {
                out[[bi, bi]] += hbb_diag[cols[bi]];
            }
        }
    }

    /// Fold analytic-penalty contributions into the appropriate blocks.
    ///
    /// BA source mapping: these are extra prior/regularization normal-equation
    /// terms before point elimination, the same place Ceres/g2o attach robust
    /// priors or gauge-fixing constraints.
    ///
    /// **Composition path.** Each registered [`AnalyticPenaltyKind`] is
    /// queried for `grad_target` (added to `g_t` or `g_β`) and then for
    /// `hessian_diag` first. Diagonal penalties (ARD and the shipped
    /// sparsity kernels) are injected directly. The row-block-only Psi-tier
    /// penalties are `ARDPenalty`, `SparsityPenalty`,
    /// `SoftmaxAssignmentSparsity`, `OrderedBetaBernoulli`,
    /// `RowPrecisionPrior`, `ParametricRowPrecisionPrior`, and
    /// `ScadMcpPenalty`. Their `d × d` per-row Hessian folds into
    /// `rows[i].htt`, so the exact arrow Schur elimination (`N` independent
    /// `d × d` row solves) represents them exactly. Dense Beta-tier penalties
    /// still fall back to `hvp` probes against the canonical basis vectors for
    /// `β`.
    ///
    /// **Cross-row Psi penalties.** Penalties whose Hessian couples *distinct*
    /// latent rows — `TotalVariationPenalty`, `SheafConsistencyPenalty`,
    /// block-orthogonality, … — produce off-row blocks `∂²P/∂t_i∂t_j`
    /// (`i ≠ j`) that the arrow elimination cannot store, since it assumes each
    /// `H_tt^(i)` is independent of every other row. These are handled without
    /// any approximation: their **gradient** is folded into `g_t` exactly as
    /// for every other Psi penalty (`grad_target → g_t`), and their full
    /// **curvature** is captured into [`Self::cross_row_penalties`] as a
    /// matrix-free operator. At solve time, `K = K0 + P_cross` where `K0` is
    /// the block-diagonal arrow operator and `P_cross · Δt = Σ_p ρ_p ·
    /// psd_majorizer_hvp_p(t, Δt)` is the cross-row penalty Hessian applied to
    /// the full flat latent vector. The presence of any captured cross-row
    /// penalty auto-routes [`Self::solve`] through the matrix-free full-system
    /// PCG path (the exact arrow block-diagonal inverse `K0⁻¹` is the
    /// preconditioner `M⁻¹`); a purely row-block-diagonal system keeps the
    /// exact one-shot Schur path unchanged. No new flag is involved — the route
    /// is selected from the captured penalty set alone (magic by default).
    ///
    /// `target_t` is the full flat latent-coordinate vector (row-major, `N·d` entries)
    /// at the current iterate; `target_beta` is the current `β`. `rho`
    /// is the global ρ vector restricted to each penalty's local slice
    /// by [`AnalyticPenaltyRegistry::rho_layout`].
    pub fn add_analytic_penalty_contributions(
        &mut self,
        registry: &AnalyticPenaltyRegistry,
        target_t: ArrayView1<'_, f64>,
        target_beta: ArrayView1<'_, f64>,
        rho_global: ArrayView1<'_, f64>,
    ) -> Result<(), ArrowSchurError> {
        registry
            .validate_rho(rho_global)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let layout = registry.rho_layout();
        let mut penalty_fingerprints = Vec::new();
        self.cross_row_penalties.clear();
        for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(ndarray::s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi => {
                    if analytic_penalty_is_row_block_diagonal(penalty) {
                        // Row-block-diagonal: fold gradient + per-row d×d
                        // curvature into rows[i].htt, exactly representable by
                        // the arrow Schur elimination.
                        self.add_ext_coord_penalty(penalty, target_t, rho_local);
                        if let Some(fingerprint) =
                            analytic_penalty_row_hessian_fingerprint(penalty, target_t, rho_local)
                        {
                            penalty_fingerprints.push(fingerprint);
                        }
                    } else {
                        // Cross-row: fold the gradient into g_t (exact, like
                        // every Psi penalty), but DO NOT fold any curvature into
                        // the row blocks — its off-row coupling cannot be stored
                        // there. Capture the penalty so the solve applies its
                        // full Hessian-vector product P_cross·Δt over the flat
                        // latent vector. This auto-selects the matrix-free
                        // full-system PCG path.
                        self.add_ext_coord_penalty_gradient_only(penalty, target_t, rho_local);
                        self.cross_row_penalties.push(CrossRowLatentPenalty {
                            penalty: penalty.clone(),
                            rho_local: rho_local.to_owned(),
                            target_t: target_t.to_owned(),
                        });
                    }
                }
                PenaltyTier::Beta => {
                    self.add_beta_penalty(penalty, target_beta, rho_local);
                }
                PenaltyTier::Rho => {
                    // Rho-tier hyperpriors do not contribute to the inner
                    // (t, β) Newton step; they enter only at the REML
                    // outer level.
                }
            }
        }
        // Cross-row penalties contribute to the Newton Hessian operator, not
        // the stored row blocks, so they must still invalidate the row-Hessian
        // cache when their curvature changes. Probe each captured penalty's PSD
        // majorizer against the current latent vector (a deterministic, generic
        // probe) and fold the resulting signature in.
        for cross in &self.cross_row_penalties {
            penalty_fingerprints.push(cross_row_penalty_fingerprint(
                &cross.penalty,
                target_t,
                cross.rho_local.view(),
            ));
        }
        self.analytic_row_hessian_fingerprint = if penalty_fingerprints.is_empty() {
            0
        } else {
            let mut hasher = Fingerprinter::new();
            hasher.write_str("arrow-schur-row-hessian-registry-v1");
            hasher.write_usize(penalty_fingerprints.len());
            for fingerprint in penalty_fingerprints {
                hasher.write_u64(fingerprint);
            }
            hasher.finish_u64()
        };
        Ok(())
    }

    /// Convert row-local Euclidean latent blocks to Riemannian tangent blocks.
    ///
    /// This is the only arrow-Schur algebra change needed for manifold
    /// latents: `g_t`, `H_tt`, and each `H_tβ` column are projected to
    /// `T_{t_i}M`, while the shared β block and Schur structure remain
    /// untouched. Embedded constrained manifolds carry a pinned normal block
    /// so the existing ambient Cholesky factorization still works; all RHS
    /// terms live in the tangent space, so the solved update retracts cleanly.
    pub fn apply_riemannian_latent_geometry(&mut self, latent: &LatentCoordValues) {
        let manifold = latent.manifold();
        self.manifold_mode_fingerprint = manifold_mode_fingerprint(latent);
        if manifold.is_euclidean() {
            return;
        }
        assert_eq!(latent.n_obs(), self.rows.len());
        assert_eq!(latent.latent_dim(), self.d);
        for (i, row) in self.rows.iter_mut().enumerate() {
            let t_i = ArrayView1::from(latent.row(i));
            let gt_e = row.gt.clone();
            let htt_e = row.htt.clone();
            let htbeta_e = row.htbeta.clone();
            row.gt = manifold.project_gradient_to_tangent(t_i, gt_e.view());
            row.htt = manifold.riemannian_hessian_matrix(t_i, gt_e.view(), htt_e.view());
            row.htbeta = manifold.project_matrix_columns_to_gradient_tangent(
                t_i,
                gt_e.view(),
                htbeta_e.view(),
            );
        }
    }

    pub(crate) fn add_ext_coord_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_t: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let d = self.d;
        let n = self.rows.len();
        apply_analytic_penalty(
            penalty,
            target_t,
            rho_local,
            n * d,
            d,
            self,
            |sys, flat, value| sys.rows[flat / d].gt[flat % d] += value,
            |sys, flat, value| sys.rows[flat / d].htt[[flat % d, flat % d]] += value,
            |a, probe| {
                for i in 0..n {
                    probe[i * d + a] = 1.0;
                }
            },
            |sys, a, hv| {
                for i in 0..n {
                    for b in 0..d {
                        sys.rows[i].htt[[b, a]] += hv[i * d + b];
                    }
                }
            },
        );
    }

    /// Fold ONLY the latent gradient `grad_target → g_t` of an analytic
    /// penalty, leaving the row-block Hessian untouched.
    ///
    /// Used for cross-row Psi penalties: their gradient enters `g_t` exactly
    /// like every other Psi penalty, but their curvature must NOT be scattered
    /// into the per-row `H_tt^(i)` blocks (the diagonal piece would be
    /// double-counted and the off-row coupling cannot be stored there). The
    /// full curvature is instead applied as a matrix-free `P_cross · Δt`
    /// during the solve, via [`Self::cross_row_penalties`].
    pub(crate) fn add_ext_coord_penalty_gradient_only(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_t: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let d = self.d;
        let n = self.rows.len();
        assert_eq!(target_t.len(), n * d);
        let grad = penalty.grad_target(target_t, rho_local);
        for flat in 0..n * d {
            self.rows[flat / d].gt[flat % d] += grad[flat];
        }
    }

    /// Apply the aggregate cross-row penalty Hessian `P_cross · v` over the
    /// full flat latent vector `v` (length `Σ_i row_dims[i]`), accumulating
    /// into `out`.
    ///
    /// `P_cross = Σ_p psd_majorizer_hvp_p(target_t, ·; ρ_p)` summed over every
    /// captured cross-row penalty. Each penalty's `psd_majorizer_hvp` is its
    /// exact (PSD) Hessian-vector product over the `N·d` flat latent vector —
    /// for `TotalVariationPenalty` this is `Dᵀ diag(φ''(D t)) D · v`, the
    /// graph/forward-difference Laplacian-style coupling that links distinct
    /// rows. The ρ scaling is already baked into each penalty's resolved
    /// weight, so no extra factor is applied here.
    ///
    /// This is only valid for homogeneous systems (every row of dimension
    /// `d`), the only shape cross-row latent penalties are defined on; the
    /// flat-index convention `flat = i·d + j` matches every penalty's
    /// `latent_dim`/row-major contract.
    pub(crate) fn apply_cross_row_penalty_hessian(
        &self,
        v: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) {
        for cross in &self.cross_row_penalties {
            assert_eq!(cross.target_t.len(), v.len());
            let hv =
                cross
                    .penalty
                    .psd_majorizer_hvp(cross.target_t.view(), cross.rho_local.view(), v);
            assert_eq!(hv.len(), out.len());
            for i in 0..out.len() {
                out[i] += hv[i];
            }
        }
    }

    pub(crate) fn add_beta_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let k = self.k;
        let hvp_columns = if self.hbb.dim() == (k, k) { k } else { 0 };
        apply_analytic_penalty(
            penalty,
            target_beta,
            rho_local,
            k,
            hvp_columns,
            self,
            |sys, j, value| sys.gb[j] += value,
            |sys, j, value| {
                if sys.hbb.dim() == (k, k) {
                    sys.hbb[[j, j]] += value;
                }
                if let Some(hbb_diag) = sys.hbb_diag.as_mut() {
                    hbb_diag[j] += value;
                }
            },
            |j, probe| probe[j] = 1.0,
            |sys, j, hv| {
                for i in 0..k {
                    sys.hbb[[i, j]] += hv[i];
                }
                // Keep `hbb_diag` consistent with the dense `hbb` Hessian when
                // both are populated (the dense-allocated path + a later
                // `set_shared_beta_operator` install). The HVP probe for
                // column `j` returns the full Hessian column, whose `j`-th
                // entry is the diagonal contribution of this penalty. Without
                // this mirror, the Jacobi Schur preconditioner — which prefers
                // `hbb_diag` over `hbb`'s diagonal — would silently use a
                // stale diagonal for any Beta-tier analytic penalty that
                // exposes only an HVP (no `hessian_diag`).
                if let Some(hbb_diag) = sys.hbb_diag.as_mut() {
                    hbb_diag[j] += hv[j];
                }
            },
        );
    }

    /// Schur-eliminate the per-row latent block and solve for `(Δt, Δβ, diag)`.
    ///
    /// This uses [`ArrowSolveOptions::automatic`]: BA dense RCS for
    /// `K <= 2000`, and Agarwal-style inexact Schur PCG above that size.
    /// Call [`ArrowSchurSystem::solve_with_options`] to force Square-Root BA
    /// or a specific inexact solve policy.
    ///
    /// Returns `(delta_t, delta_beta, ArrowPcgDiagnostics)` with `delta_t` flat
    /// row-major of length `N · d` and `delta_beta` of length `K`. The sign
    /// convention matches `solve_newton_direction_dense`: the returned
    /// increments satisfy the bordered system with RHS `[-g_t; -g_β]`, i.e.
    /// they are the *negated* solutions of the standard Newton-direction
    /// formulation. `ArrowPcgDiagnostics` is zero-valued for the Direct path and
    /// carries live counters (PCG iters, ridge escalations, residual) for
    /// InexactPCG.
    ///
    /// `ridge_t` and `ridge_beta` are nonnegative diagonal regularizers
    /// added to the latent and β blocks respectively before factorization
    /// — used by the LM damping outer wrapper to recover from near-singular
    /// inner steps. Pass `0.0` for both to obtain the unregularized
    /// Newton direction.
    pub fn solve(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError> {
        let options = ArrowSolveOptions::automatic(self.k);
        solve_arrow_newton_step_core(self, ridge_t, ridge_beta, &options)
    }

    /// Solve with the standard LM-style ridge escalation: if a per-row
    /// `H_tt + ridge_t·I` Cholesky pivot is non-PD, or the reduced Schur
    /// factor fails, geometrically grow both ridges and retry. This is the
    /// same Ceres-style proximal correction the Newton driver in
    /// `run_joint_fit_arrow_schur` performs around `solve`, lifted into the
    /// system itself so every entry point (predict OOS reconstruction,
    /// single-shot Newton refinement, …) is self-healing against the
    /// pathological per-row blocks produced by PCA-seeded latent
    /// coordinates on subset / new data — see #163 and #175.
    ///
    /// `ridge_t` / `ridge_beta` are the caller-nominal Tikhonov ridges; the
    /// escalation only adds extra damping on top of them when the factor
    /// fails. PCG / AdaptiveCorrection failures are left untouched because
    /// they are not factorization-recoverable.
    pub fn solve_with_lm_escalation(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError> {
        let options = ArrowSolveOptions::automatic(self.k);
        solve_with_lm_escalation_inner(self, ridge_t, ridge_beta, &options)
    }

    /// Solve with an explicit BA Schur mode, returning `(Δt, Δβ, ArrowPcgDiagnostics)`.
    ///
    /// [`ArrowSolverMode::Direct`] is the classic dense reduced-camera-system
    /// Cholesky path; [`ArrowSolverMode::SqrtBA`] forms the same dense system
    /// through Square-Root BA factors; [`ArrowSolverMode::InexactPCG`] runs
    /// inexact-step LM on the reduced system with Jacobi-preconditioned
    /// Steihaug-CG. `ArrowPcgDiagnostics` is zero-valued for Direct/SqrtBA and
    /// carries live counters for InexactPCG (iterations, matvec calls,
    /// preconditioner escalations, final relative residual, stopping reason).
    pub fn solve_with_options(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(Array1<f64>, Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError> {
        solve_arrow_newton_step_core(self, ridge_t, ridge_beta, options)
    }
}

/// Chunked Schur assembler that never retains all row cross-blocks.
pub struct StreamingArrowSchur {
    pub n_rows: usize,
    /// Maximum per-row latent dim (upper bound for scratch buffers).
    pub d: usize,
    /// Per-row latent dims `row_dims[i] == rows[i].htt.nrows()`.
    pub row_dims: Arc<[usize]>,
    /// Flat-buffer row offsets: `row_offsets[i]` is the start of row `i` in
    /// `delta_t`; `row_offsets[n_rows]` is the total `delta_t` length.
    pub row_offsets: Arc<[usize]>,
    pub k: usize,
    pub chunk_size: usize,
    pub s_acc: Array2<f64>,
    pub(crate) rhs_acc: Array1<f64>,
    pub(crate) hbb: Array2<f64>,
    pub(crate) gb: Array1<f64>,
    pub(crate) row_builder: StreamingArrowRowBuilder,
    /// Procedural cross-block operator `H_tβ^(i) x`. When present, the dense
    /// per-row `H_tβ` slabs are never materialized: `accumulate_chunk` and
    /// `back_substitute` probe this operator column-by-column to apply the
    /// cross-block, matching the Kronecker / matrix-free assembly path. When
    /// `None` (legacy dense BA callers), the per-row `row.htbeta` slab is used.
    pub(crate) htbeta_matvec: Option<RowHtbetaMatvec>,
    /// Sparse adjoint of `htbeta_matvec`. When present, `row_htbeta` rebuilds
    /// the dense `(d_i × K)` cross-block by probing the transpose with `d_i`
    /// basis vectors — `O(d_i · m_i · p)` total, vs the `O(K · m_i · p)` cost
    /// of probing the forward operator with `K` basis vectors. Since
    /// `d_i ≪ K`, this is the per-row sparse apply that replaces the `O(K)`
    /// column-probe in the streaming reduced-Schur accumulation.
    pub(crate) htbeta_transpose_matvec: Option<RowHtbetaTransposeMatvec>,
    /// Whether streaming rows are being factored for undamped evidence rather
    /// than for a Newton step. Defaults to `false` so direct chunk callers keep
    /// the full step-accuracy guard.
    pub(crate) evidence_factorization: bool,
    /// SAE manifold evidence-path per-row gauge deflation, copied from the
    /// source [`ArrowSchurSystem::row_gauge_deflation`] (#1273/#1377). When
    /// present, the streaming per-row factor MUST apply the SAME spectral
    /// discovery-and-deflation of an intrinsic-dimension-flat `H_tt^(i)`
    /// direction (eigenvalue → +1, ρ-independent `log 1 = 0` evidence) that the
    /// dense [`factor_blocks_for_system`] path applies, or the two routes report
    /// different log-determinants for the SAME system — the cross-route
    /// invariant `streaming_logdet == full_logdet` would break (the #1377
    /// regression: #1273 wired the deflation into the dense path only). `None`
    /// for every non-evidence caller, which keeps the strict non-PD refusal.
    pub(crate) row_gauge_deflation: Option<ArrowRowGaugeDeflation>,
}

impl std::fmt::Debug for StreamingArrowSchur {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingArrowSchur")
            .field("n_rows", &self.n_rows)
            .field("d", &self.d)
            .field("k", &self.k)
            .field("chunk_size", &self.chunk_size)
            .finish_non_exhaustive()
    }
}

impl StreamingArrowSchur {
    #[must_use]
    pub fn new(
        n_rows: usize,
        d: usize,
        row_dims: Arc<[usize]>,
        row_offsets: Arc<[usize]>,
        k: usize,
        hbb: Array2<f64>,
        gb: Array1<f64>,
        row_builder: StreamingArrowRowBuilder,
        chunk_size: usize,
    ) -> Self {
        assert_eq!(hbb.dim(), (k, k));
        assert_eq!(gb.len(), k);
        Self {
            n_rows,
            d,
            row_dims,
            row_offsets,
            k,
            chunk_size: chunk_size.max(1),
            s_acc: Array2::<f64>::zeros((k, k)),
            rhs_acc: Array1::<f64>::zeros(k),
            hbb,
            gb,
            row_builder,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            evidence_factorization: false,
            row_gauge_deflation: None,
        }
    }

    #[must_use]
    pub fn from_system(sys: &ArrowSchurSystem, chunk_size: usize) -> Self {
        // When a Kronecker / matrix-free htbeta_matvec is installed, the dense
        // row.htbeta slabs may be zero-sized.  Rather than materialize every
        // `(d × K)` slab (the very `(N·K)`-scale buffer the streaming path
        // exists to avoid), retain the procedural operator and probe it per row
        // inside `accumulate_chunk` / `back_substitute`.  The row builder then
        // only carries the small `H_tt` / `g_t` blocks.
        let htbeta_matvec = sys.htbeta_matvec.clone();
        let rows: Vec<ArrowRowBlock> = if htbeta_matvec.is_some() {
            sys.rows
                .iter()
                .map(|row| ArrowRowBlock {
                    htt: row.htt.clone(),
                    htbeta: Array2::<f64>::zeros((0, 0)),
                    gt: row.gt.clone(),
                })
                .collect()
        } else {
            sys.rows.clone()
        };
        let rows = Arc::new(rows);
        let row_builder: StreamingArrowRowBuilder = Arc::new(move |row| {
            rows.get(row)
                .cloned()
                .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                    reason: format!("streaming row {row} out of bounds"),
                })
        });
        // Materialize the dense β-block from the effective penalty operator so
        // the streaming accumulator stays correct when contributions live in a
        // structured `BetaPenaltyOp` (e.g. the SAE data-fit Gauss-Newton block,
        // represented as `G ⊗ I_p`) rather than the dense `hbb` accumulator.
        // When no `penalty_op` is installed this reduces to `hbb.clone()`.
        let hbb_dense = sys.effective_penalty_op().to_dense();
        let mut streaming = Self::new(
            sys.rows.len(),
            sys.d,
            Arc::clone(&sys.row_dims),
            Arc::clone(&sys.row_offsets),
            sys.k,
            hbb_dense,
            sys.gb.clone(),
            row_builder,
            chunk_size,
        );
        streaming.htbeta_matvec = htbeta_matvec;
        streaming.htbeta_transpose_matvec = sys.htbeta_transpose_matvec.clone();
        // Carry the SAE evidence-path per-row gauge deflation so the streaming
        // per-row factor matches the dense `factor_blocks_for_system` exactly
        // (#1377): without it, a row with an intrinsic-dimension-flat `H_tt`
        // would deflate on the dense path but be refused / log-det-divergent on
        // the streaming path, breaking `streaming_logdet == full_logdet`.
        streaming.row_gauge_deflation = sys.row_gauge_deflation.clone();
        streaming
    }

    /// Factor one streaming row's `H_tt^(i)`, applying the SAME per-row gauge /
    /// spectral deflation the dense [`factor_blocks_for_system`] path applies
    /// when this is the SAE manifold evidence path (an installed
    /// `row_gauge_deflation`). For every non-evidence caller this is exactly the
    /// generic [`factor_one_row`] (strict non-PD refusal), so PD blocks are
    /// bit-for-bit unchanged. Routing both the dense and streaming per-row
    /// factors through the identical recovery is what keeps their
    /// log-determinants identical (#1273/#1377).
    fn factor_row(
        &self,
        row: &ArrowRowBlock,
        ridge_t: f64,
        di: usize,
        row_idx: usize,
    ) -> Result<Array2<f64>, ArrowSchurError> {
        match self.row_gauge_deflation.as_ref() {
            Some(deflation) => factor_one_row_result(
                row,
                ridge_t,
                di,
                row_idx,
                self.evidence_factorization,
                deflation.row(row_idx),
                // Evidence path: opt into spectral discovery of an
                // intrinsic-dimension-flat direction even when this row's
                // supplied gauge list is empty/non-spanning — matching the
                // `allow_spectral_deflation = true` the dense path passes.
                true,
            )
            .map(|result| result.factor),
            None => factor_one_row(row, ridge_t, di, row_idx, self.evidence_factorization),
        }
    }

    /// Build the `(di × k)` cross-block for `row_idx` on demand.
    ///
    /// When the sparse transpose adjoint is installed, probes it with `di`
    /// standard basis vectors — each yields a full `K`-row of `H_βt^(i)`
    /// (i.e. a row of the `(di × k)` block) via the sparse scatter, for
    /// `O(di · m_i · p)` total, far below the `O(K · m_i · p)` cost of probing
    /// the forward operator with `K` basis vectors when `di ≪ K`.
    ///
    /// When only the forward operator is installed (no adjoint), falls back to
    /// the `k`-column forward probe. Otherwise clones the dense `row.htbeta`
    /// slab.
    pub(crate) fn row_htbeta(&self, row_idx: usize, row: &ArrowRowBlock, di: usize) -> Array2<f64> {
        if let Some(op_t) = self.htbeta_transpose_matvec.as_ref() {
            // Probe the adjoint: for each latent index c, scatter e_c to obtain
            // row c of the (di × k) block.
            let mut mat = Array2::<f64>::zeros((di, self.k));
            let mut e_c = Array1::<f64>::zeros(di);
            let mut beta_row = Array1::<f64>::zeros(self.k);
            for c in 0..di {
                e_c.fill(0.0);
                e_c[c] = 1.0;
                beta_row.fill(0.0);
                op_t(row_idx, e_c.view(), &mut beta_row);
                for a in 0..self.k {
                    mat[[c, a]] = beta_row[a];
                }
            }
            return mat;
        }
        match self.htbeta_matvec.as_ref() {
            Some(op) => {
                let mut mat = Array2::<f64>::zeros((di, self.k));
                let mut e_a = Array1::<f64>::zeros(self.k);
                let mut col = Array1::<f64>::zeros(di);
                for a in 0..self.k {
                    e_a.fill(0.0);
                    e_a[a] = 1.0;
                    col.fill(0.0);
                    op(row_idx, e_a.view(), &mut col);
                    for c in 0..di {
                        mat[[c, a]] = col[c];
                    }
                }
                mat
            }
            None => row.htbeta.clone(),
        }
    }

    /// Move out the accumulated reduced Schur block `s_acc` and reduced RHS
    /// `rhs_acc`, leaving fresh zero buffers in their place.
    ///
    /// The reduced contribution is `s_acc = hbb − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)`
    /// (the β-block `hbb` seeded by `reset_accumulator`, minus the per-row
    /// reduction summed by `accumulate_chunk`) and
    /// `rhs_acc = +Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i)`. Used by external online
    /// drivers (e.g. the SAE streaming joint fit) that accumulate the reduced
    /// system across re-materialized chunk systems.
    #[must_use]
    pub fn take_accumulators(&mut self) -> (Array2<f64>, Array1<f64>) {
        let s = std::mem::replace(&mut self.s_acc, Array2::<f64>::zeros((self.k, self.k)));
        let rhs = std::mem::replace(&mut self.rhs_acc, Array1::<f64>::zeros(self.k));
        (s, rhs)
    }

    /// Reset the dense shared accumulator to `H_ββ + ridge_beta I`.
    pub fn reset_accumulator(&mut self, ridge_beta: f64) -> Result<(), ArrowSchurError> {
        if self.hbb.dim() != (self.k, self.k) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "streaming Arrow-Schur requires a dense beta block accumulator".to_string(),
            });
        }
        self.s_acc.assign(&self.hbb);
        for j in 0..self.k {
            self.s_acc[[j, j]] += ridge_beta;
            self.rhs_acc[j] = 0.0;
        }
        Ok(())
    }

    /// Accumulate rows `[start, end)` into the reduced RHS and Schur block.
    pub fn accumulate_chunk(
        &mut self,
        start: usize,
        end: usize,
        ridge_t: f64,
        mode: ArrowSolverMode,
    ) -> Result<(), ArrowSchurError> {
        if start > end || end > self.n_rows {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "streaming Arrow-Schur chunk [{start}, {end}) outside 0..{}",
                    self.n_rows
                ),
            });
        }
        let backend = CpuBatchedBlockSolver;
        let k = self.k;
        // Per-row factor + two block solves + a `k×k` GEMM subtract is the whole
        // assembly cost at the SAE LLM shape (#1017); the rows are independent so
        // the reduction fans across cores.
        // #2228 determinism: reduce the per-row contributions —
        // `+H_βt^(i)(H_tt^(i))⁻¹ g_t^(i)` (length `k`, into the reduced RHS) and
        // `−H_βt^(i)(H_tt^(i))⁻¹ H_tβ^(i)` (`k×k`, into the reduced Schur
        // complement) — through the length-only pairwise tree. The within-chunk
        // association is then bit-identical across thread count AND to the
        // sequential fold, removing the #1017/#1211 chunk-reassociation margin
        // that let the criterion ranking depend on the driver. The tree
        // self-serializes below `BASE_CHUNK` rows (a base block is folded directly
        // with no `rayon::join`), so the handful-of-rows callers and nested
        // topology-race calls stay single-threaded without a separate branch.
        // Each streaming chunk's tree result folds into the seeded running
        // `self.{rhs,s}_acc` (which carry `H_ββ + ridge·I`) in chunk order.
        let this: &Self = self;
        let row_into = |row_idx: usize,
                        rhs_part: &mut Array1<f64>,
                        s_part: &mut Array2<f64>,
                        stack: &mut ChunkSchurStack|
         -> Result<(), ArrowSchurError> {
            let row = (this.row_builder)(row_idx)?;
            let di = row.htt.nrows();
            this.validate_row(row_idx, &row)?;
            let htbeta = this.row_htbeta(row_idx, &row, di);
            let factor = this.factor_row(&row, ridge_t, di, row_idx)?;
            let v = backend.solve_block_vector(factor.view(), row.gt.view());
            for c in 0..di {
                let vc = v[c];
                if vc == 0.0 {
                    continue;
                }
                for a in 0..k {
                    rhs_part[a] += htbeta[[c, a]] * vc;
                }
            }
            match mode {
                // InexactPCG differs from Direct only in how the *reduced* system
                // is solved, not how it is assembled, so it shares this Schur
                // subtraction.
                ArrowSolverMode::Direct | ArrowSolverMode::InexactPCG => {
                    let solved = backend.solve_block_matrix(factor.view(), htbeta.view());
                    stack.subtract_or_stack(&backend, s_part, &htbeta, &solved);
                }
                ArrowSolverMode::SqrtBA => {
                    let whitened = backend.sqrt_solve_block_matrix(factor.view(), htbeta.view());
                    stack.subtract_or_stack(&backend, s_part, &whitened, &whitened);
                }
            }
            Ok(())
        };
        let contribution = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            end - start,
            |range: core::ops::Range<usize>| -> Result<(Array1<f64>, Array2<f64>), ArrowSchurError> {
                let mut rhs_part = Array1::<f64>::zeros(k);
                let mut s_part = Array2::<f64>::zeros((k, k));
                // Dense-support rows accumulate into ONE stacked GEMM per base
                // block instead of a per-row scalar scatter; sparse rows keep the
                // nnz-scaled scatter (see `ChunkSchurStack`).
                let mut stack = ChunkSchurStack::new(k);
                for local in range {
                    row_into(start + local, &mut rhs_part, &mut s_part, &mut stack)?;
                }
                stack.flush(&mut s_part);
                Ok((rhs_part, s_part))
            },
            |(mut ra, mut sa): (Array1<f64>, Array2<f64>),
             (rb, sb): (Array1<f64>, Array2<f64>)|
             -> Result<(Array1<f64>, Array2<f64>), ArrowSchurError> {
                ra += &rb;
                sa += &sb;
                Ok((ra, sa))
            },
        )?;
        // `subtract_or_stack`/`flush` already subtracted into each `s_part`, so the
        // partials carry the NEGATIVE Schur contribution; add them into the seeded
        // running accumulators.
        if let Some((rhs_part, s_part)) = contribution {
            for a in 0..k {
                self.rhs_acc[a] += rhs_part[a];
            }
            self.s_acc += &s_part;
        }
        Ok(())
    }

    /// Compute the exact arrow Hessian log-determinant by accumulating the
    /// reduced Schur complement in row chunks, without retaining the full set
    /// of per-row Cholesky factors.
    ///
    /// This is the streaming analogue of [`ArrowFactorCache::arrow_log_det`]:
    ///
    /// ```text
    /// log|H| = Σ_i log|H_tt^(i)| + log|H_ββ - Σ_i H_βt^(i) H_tt^(i)⁻¹ H_tβ^(i)|.
    /// ```
    ///
    /// The same row builder and procedural `H_tβ` callbacks used by the
    /// streaming Newton solve are consumed here, so callers can score REML
    /// evidence without materialising the full `(N × q × K)` cross block or
    /// the full list of row factors.
    pub fn reduced_schur_and_log_det_tt(
        &mut self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(f64, Array2<f64>), ArrowSchurError> {
        self.evidence_factorization = options.evidence_policy.factors_undamped_evidence();
        self.reset_accumulator(ridge_beta)?;
        let backend = CpuBatchedBlockSolver;
        let mut log_det_tt = 0.0_f64;
        for start in (0..self.n_rows).step_by(self.chunk_size) {
            let end = (start + self.chunk_size).min(self.n_rows);
            for row_idx in start..end {
                let row = (self.row_builder)(row_idx)?;
                let di = row.htt.nrows();
                self.validate_row(row_idx, &row)?;
                let htbeta = self.row_htbeta(row_idx, &row, di);
                let factor = self.factor_row(&row, ridge_t, di, row_idx)?;
                for axis in 0..di {
                    log_det_tt += 2.0 * factor[[axis, axis]].ln();
                }
                match options.mode {
                    ArrowSolverMode::Direct | ArrowSolverMode::InexactPCG => {
                        let solved = backend.solve_block_matrix(factor.view(), htbeta.view());
                        backend.block_gemm_subtract(&mut self.s_acc, &htbeta, &solved);
                    }
                    ArrowSolverMode::SqrtBA => {
                        let whitened =
                            backend.sqrt_solve_block_matrix(factor.view(), htbeta.view());
                        backend.block_gemm_subtract(&mut self.s_acc, &whitened, &whitened);
                    }
                }
            }
        }
        symmetrize_upper_from_lower(&mut self.s_acc);
        let schur = std::mem::replace(&mut self.s_acc, Array2::<f64>::zeros((self.k, self.k)));
        Ok((log_det_tt, schur))
    }

    pub fn reduced_schur_log_det(
        schur: &Array2<f64>,
        options: &ArrowSolveOptions,
    ) -> Result<f64, ArrowSchurError> {
        let schur_factor = factor_dense_reduced_schur(
            schur,
            options.evidence_policy.reduced_schur_policy(),
        )?
        .factor;
        let mut log_det_schur = 0.0_f64;
        for axis in 0..schur_factor.nrows() {
            log_det_schur += 2.0 * schur_factor[[axis, axis]].ln();
        }
        Ok(log_det_schur)
    }

    pub fn exact_arrow_log_det(
        &mut self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<f64, ArrowSchurError> {
        let (log_det_tt, schur) =
            self.reduced_schur_and_log_det_tt(ridge_t, ridge_beta, options)?;
        Ok(log_det_tt + Self::reduced_schur_log_det(&schur, options)?)
    }

    pub fn solve(
        &mut self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(Array1<f64>, Array1<f64>, Option<Array2<f64>>), ArrowSchurError> {
        // Newton streaming factors always retain the step-accuracy guard.
        self.evidence_factorization = false;
        self.reset_accumulator(ridge_beta)?;
        for start in (0..self.n_rows).step_by(self.chunk_size) {
            let end = (start + self.chunk_size).min(self.n_rows);
            self.accumulate_chunk(start, end, ridge_t, options.mode)?;
        }
        for j in 0..self.k {
            self.rhs_acc[j] -= self.gb[j];
        }
        symmetrize_upper_from_lower(&mut self.s_acc);
        let trust_metric_weights = None;
        let (delta_beta, schur_factor, _diag) =
            solve_dense_reduced_system(&self.s_acc, &self.rhs_acc, options, trust_metric_weights)?;
        let delta_t = self.back_substitute(ridge_t, delta_beta.view())?;
        Ok((delta_t, delta_beta, schur_factor))
    }

    pub(crate) fn back_substitute(
        &self,
        ridge_t: f64,
        delta_beta: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, ArrowSchurError> {
        let backend = CpuBatchedBlockSolver;
        // Total delta_t length = row_offsets[n_rows].
        let total_len = self.row_offsets[self.n_rows];
        let mut delta_t = Array1::<f64>::zeros(total_len);
        // Each row's back-solve `Δt_i = -(H_tt^(i))⁻¹(g_t^(i) + H_tβ^(i)Δβ)`
        // writes a DISJOINT segment `delta_t[row_base .. row_base+di]` — no
        // cross-row reduction, so this is embarrassingly parallel and the scatter
        // is bit-identical regardless of which thread produced each segment (the
        // #1017 verification gate). At the SAE LLM shape (`n` in the thousands)
        // the per-row factor + solve is the whole cost; below the threshold, or
        // when already inside a rayon worker (the topology race fans candidates
        // with `run_topology_race_parallel`), stay sequential to avoid
        // nested-rayon oversubscription — the same guard `schur_matvec` uses.
        let parallel =
            self.n_rows >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            // Per-row body: factor, form the RHS, solve, return `-(dt_i)`.
            let row_solve = |row_idx: usize| -> Result<(usize, Array1<f64>), ArrowSchurError> {
                let row = (self.row_builder)(row_idx)?;
                let di = row.htt.nrows();
                self.validate_row(row_idx, &row)?;
                let factor = self.factor_row(&row, ridge_t, di, row_idx)?;
                let mut htbeta_delta = Array1::<f64>::zeros(di);
                if let Some(op) = self.htbeta_matvec.as_ref() {
                    op(row_idx, delta_beta, &mut htbeta_delta);
                } else {
                    for c in 0..di {
                        let mut acc = 0.0_f64;
                        for a in 0..self.k {
                            acc += row.htbeta[[c, a]] * delta_beta[a];
                        }
                        htbeta_delta[c] = acc;
                    }
                }
                let mut rhs = Array1::<f64>::zeros(di);
                for c in 0..di {
                    rhs[c] = row.gt[c] + htbeta_delta[c];
                }
                let dt_i = backend.solve_block_vector(factor.view(), rhs.view());
                let mut neg = Array1::<f64>::zeros(di);
                for c in 0..di {
                    neg[c] = -dt_i[c];
                }
                Ok((self.row_offsets[row_idx], neg))
            };
            // Collect per-row segments under rayon, then scatter into the disjoint
            // slices. Errors are surfaced via `collect::<Result<…>>`.
            let segments: Vec<(usize, Array1<f64>)> = (0..self.n_rows)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    idxs.into_iter()
                        .map(&row_solve)
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            for (base, seg) in &segments {
                for (c, &v) in seg.iter().enumerate() {
                    delta_t[base + c] = v;
                }
            }
        } else {
            let mut rhs = Array1::<f64>::zeros(self.d);
            for start in (0..self.n_rows).step_by(self.chunk_size) {
                let end = (start + self.chunk_size).min(self.n_rows);
                for row_idx in start..end {
                    let row = (self.row_builder)(row_idx)?;
                    let di = row.htt.nrows();
                    self.validate_row(row_idx, &row)?;
                    let factor = self.factor_row(&row, ridge_t, di, row_idx)?;
                    // `H_tβ^(i) Δβ`: route through the procedural operator when
                    // present (no dense slab), else through the dense slab.
                    let mut htbeta_delta = Array1::<f64>::zeros(di);
                    if let Some(op) = self.htbeta_matvec.as_ref() {
                        op(row_idx, delta_beta, &mut htbeta_delta);
                    } else {
                        for c in 0..di {
                            let mut acc = 0.0_f64;
                            for a in 0..self.k {
                                acc += row.htbeta[[c, a]] * delta_beta[a];
                            }
                            htbeta_delta[c] = acc;
                        }
                    }
                    for c in 0..di {
                        rhs[c] = row.gt[c] + htbeta_delta[c];
                    }
                    let dt_i = backend.solve_block_vector(factor.view(), rhs.view());
                    let row_base = self.row_offsets[row_idx];
                    for c in 0..di {
                        delta_t[row_base + c] = -dt_i[c];
                    }
                }
            }
        }
        Ok(delta_t)
    }

    pub(crate) fn validate_row(
        &self,
        row_idx: usize,
        row: &ArrowRowBlock,
    ) -> Result<(), ArrowSchurError> {
        let expected_di = if row_idx < self.row_dims.len() {
            self.row_dims[row_idx]
        } else {
            self.d
        };
        let actual_di = row.htt.nrows();
        if actual_di != expected_di || row.htt.ncols() != expected_di {
            return Err(ArrowSchurError::PerRowFactorFailed {
                row: row_idx,
                reason: format!(
                    "streaming row H_tt shape {:?} != ({expected_di}, {expected_di})",
                    row.htt.dim(),
                ),
            });
        }
        // The dense `H_tβ` slab is only validated when no procedural operator is
        // installed; with `htbeta_matvec` the slab is intentionally zero-sized
        // and the cross-block is probed in `row_htbeta`.
        if self.htbeta_matvec.is_none() && row.htbeta.dim() != (expected_di, self.k) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "streaming row H_tβ shape {:?} != ({expected_di}, {})",
                    row.htbeta.dim(),
                    self.k
                ),
            });
        }
        if row.gt.len() != expected_di {
            return Err(ArrowSchurError::PerRowFactorFailed {
                row: row_idx,
                reason: format!("streaming row g_t length {} != {expected_di}", row.gt.len()),
            });
        }
        Ok::<(), _>(())
    }
}

pub(crate) fn apply_analytic_penalty<S, G, D, P, H>(
    penalty: &AnalyticPenaltyKind,
    target: ArrayView1<'_, f64>,
    rho_local: ArrayView1<'_, f64>,
    expected_target_len: usize,
    hvp_columns: usize,
    scatter_target: &mut S,
    mut grad_scatter: G,
    mut diag_scatter: D,
    seed_hvp_probe: P,
    mut hvp_column_scatter: H,
) where
    G: FnMut(&mut S, usize, f64),
    D: FnMut(&mut S, usize, f64),
    P: Fn(usize, &mut Array1<f64>),
    H: for<'a> FnMut(&mut S, usize, ArrayView1<'a, f64>),
{
    assert_eq!(target.len(), expected_target_len);

    let grad = penalty.grad_target(target, rho_local);
    for index in 0..expected_target_len {
        grad_scatter(scatter_target, index, grad[index]);
    }

    // The scattered curvature lands in the arrow-Schur `H_tt` / `H_ββ` blocks,
    // which are Cholesky-factored (with LM ridge escalation) as the Newton /
    // PIRLS curvature operator and must therefore stay PSD. Nonconvex
    // sparsifiers (log sparsity, JumpReLU) have an *indefinite* exact Hessian
    // that would destroy that positive-definiteness, so we scatter the PSD
    // majorizer here — never the exact `hessian_diag` / `hvp`. For convex
    // penalties the majorizer equals the exact Hessian (the trait default
    // delegates), so this is exact for them. Exact-derivative consumers (the
    // outer objective Hessian) use `hessian_diag` / `hvp` directly elsewhere.
    if let Some(diag) = penalty.psd_majorizer_diag(target, rho_local) {
        assert_eq!(diag.len(), expected_target_len);
        for index in 0..expected_target_len {
            diag_scatter(scatter_target, index, diag[index]);
        }
        return;
    }

    let mut probe = Array1::<f64>::zeros(expected_target_len);
    for column in 0..hvp_columns {
        probe.fill(0.0);
        seed_hvp_probe(column, &mut probe);
        let hv = penalty.psd_majorizer_hvp(target, rho_local, probe.view());
        hvp_column_scatter(scatter_target, column, hv.view());
    }
}

pub(crate) fn analytic_penalty_is_row_block_diagonal(penalty: &AnalyticPenaltyKind) -> bool {
    penalty.is_row_block_diagonal()
}

/// Per-row + Schur Cholesky factor cache produced by
/// [`solve_arrow_newton_step_with_options`]. Consumed downstream by the IFT warm-start
/// predictor in `crate::persistent_warm_start`: when the outer
/// loop perturbs `(β, ρ)` by a small amount, the new Newton step can be
/// predicted by re-using these factors against a refreshed RHS, saving
/// the dominant `O(N d³ + K³)` factorization cost.
#[derive(Clone)]
pub struct ArrowFactorSlab {
    pub(crate) data: Arc<[f64]>,
    pub(crate) offsets: Arc<[usize]>,
    pub(crate) dims: Arc<[usize]>,
}

impl ArrowFactorSlab {
    pub fn from_blocks(blocks: Vec<Array2<f64>>) -> Self {
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(blocks.len() + 1);
        let mut dims = Vec::with_capacity(blocks.len());
        offsets.push(0);
        for block in blocks {
            let (rows, cols) = block.dim();
            assert_eq!(rows, cols, "ArrowFactorSlab stores square row factors");
            dims.push(rows);
            data.extend(block.iter().copied());
            offsets.push(data.len());
        }
        Self {
            data: data.into(),
            offsets: offsets.into(),
            dims: dims.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn factor(&self, row: usize) -> ArrayView2<'_, f64> {
        let dim = self.dims[row];
        let range = self.offsets[row]..self.offsets[row + 1];
        ArrayView2::from_shape((dim, dim), &self.data[range])
            .expect("ArrowFactorSlab row offset/dim invariant violated")
    }

    pub fn iter(&self) -> impl Iterator<Item = ArrayView2<'_, f64>> + '_ {
        (0..self.len()).map(|row| self.factor(row))
    }
}

impl std::fmt::Debug for ArrowFactorSlab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowFactorSlab")
            .field("rows", &self.len())
            .field("values", &self.data.len())
            .finish()
    }
}

#[derive(Clone)]
pub enum ArrowUndampedFactors {
    SameAsDamped,
    Owned(ArrowFactorSlab),
}

impl std::fmt::Debug for ArrowUndampedFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SameAsDamped => f.write_str("SameAsDamped"),
            Self::Owned(factors) => f.debug_tuple("Owned").field(&factors.len()).finish(),
        }
    }
}

/// Apply `H_tβ^(row) · x` for one row, writing into `out` (length `d`).
///
/// Sums the installed matrix-free operator, when present, and any correctly
/// shaped dense `row.htbeta` slab. This lets structured data-fit rows coexist
/// with dense analytic-penalty cross blocks on the same row.
pub(crate) fn sys_htbeta_apply_row(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    x: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    out.fill(0.0);
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        op(row_idx, x, out);
    }
    if (sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none())
        && row.htbeta.dim() == (out.len(), sys.k)
    {
        let di = row.htbeta.nrows();
        for c in 0..di {
            let mut acc = 0.0_f64;
            for a in 0..sys.k {
                acc += row.htbeta[[c, a]] * x[a];
            }
            out[c] += acc;
        }
    }
}

/// Accumulate `H_βt^(row) · v` into `out` (length `k`).
///
/// `out[a] += Σ_c H_tβ^(row)[c, a] · v[c]`
///
/// Sums the installed matrix-free operator, when present, and any correctly
/// shaped dense `row.htbeta` slab.
pub(crate) fn sys_htbeta_accumulate_transpose(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    v: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        htbeta_probe_transpose(row_idx, op, v, out, v.len(), sys.k);
    }
    if (sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none())
        && row.htbeta.dim() == (v.len(), sys.k)
    {
        let di = row.htbeta.nrows();
        for c in 0..di {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            for a in 0..sys.k {
                out[a] += row.htbeta[[c, a]] * vc;
            }
        }
    }
}

/// Materialize the dense `(di, k)` cross-block for one row.
///
/// Materializes the sum of the installed matrix-free operator and any correctly
/// shaped dense slab on the row.
pub(crate) fn sys_htbeta_materialize_row(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
) -> Result<Array2<f64>, ArrowSchurError> {
    let di = sys.row_dims[row_idx];
    let k = sys.k;
    let use_dense = sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none();
    let mut mat = if use_dense && row.htbeta.dim() == (di, k) {
        row.htbeta.clone()
    } else {
        Array2::<f64>::zeros((di, k))
    };
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        let mut e_a = Array1::<f64>::zeros(k);
        let mut col = Array1::<f64>::zeros(di);
        for a in 0..k {
            e_a.fill(0.0);
            e_a[a] = 1.0;
            col.fill(0.0);
            op(row_idx, e_a.view(), &mut col);
            for c in 0..di {
                mat[[c, a]] += col[c];
            }
        }
    } else if use_dense && row.htbeta.dim() != (di, k) {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "row {row_idx}: htbeta shape {:?} != ({di}, {k}) and no htbeta_matvec installed",
                row.htbeta.dim()
            ),
        });
    }
    Ok(mat)
}

/// Probe each column of `H_tβ^(row)` by applying the operator to `e_a` and
/// dotting the result with `v`.  Accumulates into `out[a]` for all `a in 0..k`.
///
/// `out[a] += (H_tβ^(row) e_a) · v = H_βt^(row)[a, :] · v`
pub(crate) fn htbeta_probe_transpose(
    row: usize,
    op: &RowHtbetaMatvec,
    v: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
    d: usize,
    k: usize,
) {
    let mut e_a = Array1::<f64>::zeros(k);
    let mut col_a = Array1::<f64>::zeros(d);
    for a in 0..k {
        e_a.fill(0.0);
        e_a[a] = 1.0;
        col_a.fill(0.0);
        op(row, e_a.view(), &mut col_a);
        let mut acc = 0.0_f64;
        for c in 0..d {
            acc += col_a[c] * v[c];
        }
        out[a] += acc;
    }
}

#[derive(Clone)]
pub enum ArrowHtbetaCache {
    Dense {
        blocks: Arc<[Array2<f64>]>,
        estimated_bytes: usize,
    },
    Matvec {
        op: RowHtbetaMatvec,
        estimated_bytes: usize,
    },
    Disabled {
        estimated_bytes: usize,
    },
}

impl std::fmt::Debug for ArrowHtbetaCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense {
                blocks,
                estimated_bytes,
            } => f
                .debug_struct("Dense")
                .field("blocks", &blocks.len())
                .field("estimated_bytes", estimated_bytes)
                .finish(),
            Self::Matvec {
                estimated_bytes, ..
            } => f
                .debug_struct("Matvec")
                .field("estimated_bytes", estimated_bytes)
                .finish(),
            Self::Disabled { estimated_bytes } => f
                .debug_struct("Disabled")
                .field("estimated_bytes", estimated_bytes)
                .finish(),
        }
    }
}

impl ArrowHtbetaCache {
    pub(crate) fn is_available(&self) -> bool {
        !matches!(self, Self::Disabled { .. })
    }

    pub(crate) fn apply_row(
        &self,
        row: usize,
        delta_beta: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) -> bool {
        match self {
            Self::Dense { blocks, .. } => {
                let Some(block) = blocks.get(row) else {
                    return false;
                };
                if block.ncols() != delta_beta.len() || block.nrows() != out.len() {
                    return false;
                }
                for c in 0..block.nrows() {
                    let mut acc = 0.0_f64;
                    for a in 0..block.ncols() {
                        acc += block[[c, a]] * delta_beta[a];
                    }
                    out[c] = acc;
                }
                true
            }
            Self::Matvec { op, .. } => {
                op(row, delta_beta, out);
                true
            }
            Self::Disabled { .. } => false,
        }
    }

    /// Apply the transpose: `out[a] += H_βt^(row)[a, c] · v[c]` for all `a`.
    ///
    /// `v` has length `d`; `out` has length `k`. Accumulates (does NOT zero
    /// `out` first) so callers can sum contributions across rows into a shared
    /// accumulator.  Returns `false` when the cache is `Disabled` and no
    /// `fallback_op` is provided.
    pub(crate) fn apply_row_transpose_accumulate(
        &self,
        row: usize,
        v: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        k: usize,
        fallback_op: Option<&RowHtbetaMatvec>,
    ) -> bool {
        match self {
            Self::Dense { blocks, .. } => {
                let Some(block) = blocks.get(row) else {
                    return false;
                };
                if block.nrows() != v.len() || block.ncols() != out.len() {
                    return false;
                }
                // H_βt^(i) · v: outer-loop c hoists v[c], inner-loop a is
                // contiguous in row-major (d, k) layout.
                for c in 0..block.nrows() {
                    let vc = v[c];
                    if vc == 0.0 {
                        continue;
                    }
                    for a in 0..block.ncols() {
                        out[a] += block[[c, a]] * vc;
                    }
                }
                true
            }
            Self::Matvec { op, .. } => {
                // Probe column-by-column: H_tβ^(row) e_a is column a.  dot(col_a, v)
                // is entry a of H_βt^(row) v.
                htbeta_probe_transpose(row, op, v, out, d, k);
                true
            }
            Self::Disabled { .. } => {
                // No cached block.  Use the caller-supplied fallback op if present.
                if let Some(op) = fallback_op {
                    htbeta_probe_transpose(row, op, v, out, d, k);
                    true
                } else {
                    false
                }
            }
        }
    }
}

/// RAW per-row spectral data of a spectrally-deflated undamped evidence `H_tt`
/// block (see [`ArrowFactorCache::deflation_row_spectra`]).
///
/// `evecs` columns are the RAW symmetric eigenvectors `uₘ` of `H_tt`
/// (orthonormal; the deflated directions `vᵢ` are the subset whose eigenvalue
/// was pinned). `raw_evals[m]` is the RAW eigenvalue `λₘ` BEFORE the unit-pin /
/// floor-clamp. `cond_evals[m]` is the conditioned eigenvalue `λ̃ₘ` the factor
/// actually uses (`λ̃ = λ` for an unclamped kept direction, the positive `floor`
/// for a clamped kept direction, `1` for a deflated direction). Together they
/// give the Daleckii–Krein divided differences the outer-gradient deflation
/// correction needs.
#[derive(Debug, Clone)]
pub struct RowDeflationSpectrum {
    pub evecs: Array2<f64>,
    pub raw_evals: Array1<f64>,
    pub cond_evals: Array1<f64>,
}

/// Raw and conditioned eigenspectrum of an evidence β-Schur that underwent
/// unit deflation. `deflated[m]` is authoritative: a conditioned eigenvalue of
/// one is not itself evidence that the direction is a quotient null.
#[derive(Debug, Clone)]
pub struct BetaSchurDeflationSpectrum {
    pub evecs: Array2<f64>,
    pub raw_evals: Array1<f64>,
    pub cond_evals: Array1<f64>,
    pub deflated: Arc<[bool]>,
}

#[derive(Debug, Clone)]
pub struct ArrowFactorCache {
    /// Per-row lower-triangular Cholesky factors of `H_tt^(i) + ridge_t·I`.
    ///
    /// These are the *damped* factors used inside the Newton solve. The IFT
    /// predictor must NOT use them — see [`Self::htt_factors_undamped`].
    pub htt_factors: ArrowFactorSlab,
    /// Per-row lower-triangular Cholesky factors of the UNDAMPED
    /// `H_tt^(i)` (no `ridge_t` added).
    ///
    /// The IFT predictor formula
    /// `Δt_i = -(H_tt^(i))⁻¹ · (H_tβ^(i) Δβ + δg_t^(i))` is derived from
    /// `∂g_t/∂t = H_tt` at the stationary point, with no LM damping term.
    /// Reusing the damped factors would bias the predicted shift toward zero
    /// in proportion to `ridge_t`. We pay one extra `O(N d³)` Cholesky per
    /// Newton solve — the same complexity class as the Newton solve itself —
    /// to make the IFT exact.
    pub htt_factors_undamped: ArrowUndampedFactors,
    /// Lower-triangular Cholesky factor of the Schur complement when the
    /// selected BA mode formed/factored dense RCS. `None` for
    /// [`ArrowSolverMode::InexactPCG`], where Agarwal-style inexact LM avoids
    /// the dense `K × K` factor.
    pub schur_factor: Option<Array2<f64>>,
    /// True iff `schur_factor` is the reduced Schur complement built from the
    /// undamped evidence row factors (`H_tt`, no LM ridge) and `ridge_beta = 0`.
    ///
    /// A Newton solve may be damped while the cache still carries an undamped
    /// evidence Schur for logdet / selected-inverse consumers. When this is false,
    /// consumers must not combine `schur_factor` with [`Self::undamped_factor`]:
    /// that would mix two different bordered-arrow operators.
    pub schur_factor_is_undamped: bool,
    /// Authoritative original-coordinate spectrum and null mask used when the
    /// undamped evidence β-Schur was unit-deflated. The mask, rather than a
    /// threshold re-derived from `L Lᵀ`, defines which directions contribute
    /// `log 1 = 0` to the value and zero to every inverse/trace contraction.
    pub beta_schur_deflation: Option<BetaSchurDeflationSpectrum>,
    /// Exact undamped joint-Hessian log-determinant produced by the dense
    /// factorization path. REML evidence consumes this directly so the Laplace
    /// normalizer cannot miss the log-det even when later cache consumers only
    /// need solves/traces.
    ///
    /// On the matrix-free large-`k` SAE evidence path this is set from the
    /// Stochastic Lanczos Quadrature reduced-Schur log-determinant (see
    /// [`Self::undamped_arrow_log_det_with_schur`] and
    /// [`crate::arrow_schur::slq_logdet`]) so no dense `k × k` Cholesky is ever
    /// formed; `arrow_log_det_from_cache` reads THIS field first, before any
    /// `schur_factor` diagonal fallback.
    pub joint_hessian_log_det: Option<f64>,
    /// BA mode used to create this cache.
    pub solver_mode: ArrowSolverMode,
    /// Ridge values used to build the cached factors (recorded so the
    /// warm-start predictor knows whether the cache is still valid for a
    /// requested ridge level).
    pub ridge_t: f64,
    pub ridge_beta: f64,
    /// Per-row cross-block access for `H_tβ^(i) x`.
    ///
    /// Large caches retain a row matvec callback or disable β-coupled IFT
    /// prediction instead of cloning every dense `d × K` slab.
    pub htbeta: ArrowHtbetaCache,
    /// Maximum per-row latent dim (upper bound; matches `sys.d` at creation).
    pub d: usize,
    /// Per-row latent dims: `row_dims[i]` is the active dim for row `i`.
    pub row_dims: Arc<[usize]>,
    /// Flat-buffer row offsets for `delta_t` / IFT output vectors.
    /// `row_offsets[i]` is the start of row `i`; `row_offsets[n]` is the
    /// total length.
    pub row_offsets: Arc<[usize]>,
    /// β dimensionality `K`.
    pub k: usize,
    /// Geometry tag for the row-local factors and cross-blocks.
    pub manifold_mode_fingerprint: u64,
    /// Row-system tag for the cached per-row factors, cross-blocks, and
    /// shared-block diagonal used to build the Schur factor.
    pub row_hessian_fingerprint: u64,
    /// PCG instrumentation from the solve that produced this cache.
    ///
    /// Zero-valued (default) when the selected mode did not use PCG
    /// (i.e. `Direct` or `SqrtBA`).
    pub pcg_diagnostics: ArrowPcgDiagnostics,
    /// Number of row-local gauge directions stiffened in an undamped evidence
    /// factorization.
    ///
    /// Each direction is stiffened at UNIT stiffness `kappa = 1.0`, so it
    /// contributes `log(1) = 0` to the row-block logdet through the returned
    /// Cholesky factor: the gauge orbit is a criterion null direction and adds
    /// nothing to the Laplace normalizer (the quotient pseudo-determinant
    /// convention, cf. `PenaltyPseudologdet`). Zero theta/rho dependence.
    pub gauge_deflated_directions: usize,
    /// Per-row unit-norm directions `vᵢ` (in each row's `d`-dim latent block
    /// coordinates) that an undamped evidence factorization stiffened to UNIT
    /// stiffness `λ̃ = 1` (gauge or spectral deflation). Indexed by row; empty
    /// for every PD row factored without deflation, and empty overall on the
    /// non-deflating solver paths (streaming / cross-row-penalty CG / device).
    ///
    /// A deflated direction contributes `log(1) = 0` to the row-block log-det
    /// and is ρ/θ-INDEPENDENT, so its true contribution to `∂log|H|/∂ρ` is `0`.
    /// The analytic outer-gradient traces (`assignment_log_strength_hessian_trace`
    /// and `logdet_theta_adjoint`) contract
    /// `∂H_raw/∂ρ` (the RAW, pre-deflation block derivative) against the DEFLATED
    /// inverse, which assigns `1/λ̃ = 1` to each `vᵢ` and therefore spuriously
    /// adds `½ vᵢᵀ (∂H_raw/∂ρ) vᵢ`. Those traces subtract this per-row term
    /// (kept-subspace restriction) using these directions; without them the
    /// REML outer ρ-gradient is biased by `+Σ_deflated ½ vᵢᵀ ∂H_raw/∂ρ vᵢ`.
    pub deflated_row_directions: Arc<[Vec<Array1<f64>>]>,
    /// Per-row RAW spectral decomposition of an undamped evidence `H_tt` block
    /// that underwent SPECTRAL deflation, surfaced so the outer ρ/θ-gradient
    /// traces can apply the EXACT deflation-map (Daleckii–Krein) derivative
    /// correction, not just the within-row kept-subspace term.
    ///
    /// The criterion VALUE re-deflates `H_tt` at every ρ, so its gradient is
    /// `tr(H_deflated⁻¹ DΦ[∂H_raw/∂ρ])`, where `Φ` is the spectral pin-to-unit
    /// map. By Daleckii–Krein `DΦ[Ȧ] = U (F ∘ UᵀȦU) Uᵀ` with the divided-
    /// difference matrix `F_{ml} = (λ̃ₘ − λ̃ₗ)/(λₘ − λₗ)` (raw `λ` in the
    /// denominator, conditioned `λ̃` in the numerator). The kept×kept block of
    /// `F` is `1` (the kept subspace contracts the raw derivative unchanged), the
    /// deflated×deflated block is `0`, and the kept(m)×deflated(i) block is
    /// `(λₘ − 1)/(λₘ − λᵢ)` — this last, ROTATION, term is what the per-row
    /// kept-subspace correction alone misses; it couples to the β-block through
    /// the Schur back-substitution carried in `(H⁻¹)_tt`.
    ///
    /// `Some(spectrum)` only for spectrally-deflated rows; `None` for PD rows,
    /// gauge-only deflation (ρ-independent structural null — within-row term
    /// suffices), and every non-SAE-evidence solver path (streaming / device /
    /// cross-row CG). Empty overall when no row deflated spectrally.
    pub deflation_row_spectra: Arc<[Option<RowDeflationSpectrum>]>,
    /// Shared-border scale gauge used by the evidence factor.
    ///
    /// When present, `schur_factor` factors `P S P + Q Q^T`, and every public
    /// inverse primitive projects both its border RHS and result with `P`.  The
    /// unit-pinned orbit contributes zero to `arrow_log_det` and zero to every
    /// analytic trace, so value and gradient live on the same quotient.
    pub beta_gauge_quotient: Option<ArrowBetaGaugeQuotient>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrowFactorMinPivot {
    pub min_row_pivot: Option<f64>,
    pub min_schur_pivot: Option<f64>,
    pub min_pivot: Option<f64>,
}

impl ArrowFactorMinPivot {
    pub(crate) fn combine(row: Option<f64>, schur: Option<f64>) -> Self {
        let min_pivot = match (row, schur) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        Self {
            min_row_pivot: row,
            min_schur_pivot: schur,
            min_pivot,
        }
    }
}

pub(crate) fn lower_cholesky_min_pivot(factor: ArrayView2<'_, f64>) -> Option<f64> {
    let width = factor.nrows().min(factor.ncols());
    let mut out = None;
    for idx in 0..width {
        let pivot = factor[[idx, idx]] * factor[[idx, idx]];
        out = Some(match out {
            Some(current) => f64::min(current, pivot),
            None => pivot,
        });
    }
    out
}

pub(crate) fn lower_cholesky_max_pivot(factor: ArrayView2<'_, f64>) -> Option<f64> {
    let width = factor.nrows().min(factor.ncols());
    let mut out = None;
    for idx in 0..width {
        let pivot = factor[[idx, idx]] * factor[[idx, idx]];
        out = Some(match out {
            Some(current) => f64::max(current, pivot),
            None => pivot,
        });
    }
    out
}

/// Smallest cached Cholesky pivot for row blocks and the dense Schur factor.
///
/// Pivots are returned as squared lower-factor diagonals, matching the Hessian
/// scale rather than the Cholesky-factor scale. In inexact PCG mode the dense
/// Schur factor is absent, so `min_schur_pivot` is `None`.
pub fn arrow_factor_min_pivot(cache: &ArrowFactorCache) -> ArrowFactorMinPivot {
    let mut min_row_pivot = None;
    for factor in cache.htt_factors.iter() {
        if let Some(pivot) = lower_cholesky_min_pivot(factor) {
            min_row_pivot = Some(match min_row_pivot {
                Some(current) => f64::min(current, pivot),
                None => pivot,
            });
        }
    }
    let min_schur_pivot = cache
        .schur_factor
        .as_ref()
        .and_then(|factor| lower_cholesky_min_pivot(factor.view()));
    ArrowFactorMinPivot::combine(min_row_pivot, min_schur_pivot)
}

/// Largest cached Cholesky pivot across the row blocks and the dense Schur
/// factor (Hessian scale, i.e. squared lower-factor diagonal). This is the
/// diagonal magnitude scale a safe-SPD pivot floor is measured against: the
/// curvature-homotopy tracker (#1007) compares the min pivot against
/// `√eps · max(this, 1)`, the same floor the inner solver's
/// [`safe_spd_pivot_min`] uses. `None` only for an empty cache.
pub fn arrow_factor_max_pivot(cache: &ArrowFactorCache) -> Option<f64> {
    let mut max_pivot: Option<f64> = None;
    for factor in cache.htt_factors.iter() {
        if let Some(pivot) = lower_cholesky_max_pivot(factor) {
            max_pivot = Some(match max_pivot {
                Some(current) => f64::max(current, pivot),
                None => pivot,
            });
        }
    }
    if let Some(factor) = cache.schur_factor.as_ref()
        && let Some(pivot) = lower_cholesky_max_pivot(factor.view())
    {
        max_pivot = Some(match max_pivot {
            Some(current) => f64::max(current, pivot),
            None => pivot,
        });
    }
    max_pivot
}

/// Spectral pseudo-inverse of the cached β-Schur operator `M = L Lᵀ`, deflated
/// across the numerically-null curvature directions using the solver's
/// canonical rank floor ([`SPECTRAL_DEFLATION_REL_FLOOR`]).
///
/// `evecs` are the orthonormal eigenvectors of `M` (columns), `inv_evals[i]`
/// is `1/λᵢ` for a kept direction and exactly `0.0` for a deflated one, so
/// `M⁺ = evecs · diag(inv_evals) · evecsᵀ` is the Moore–Penrose pseudo-inverse
/// restricted to the kept subspace. Away from the ρ lower face every eigenvalue
/// sits far above the floor, no direction deflates, and `M⁺` equals `M⁻¹` to
/// round-off — so the deflated selected inverse reduces to the plain one.
struct DeflatedSchurPseudoInverse {
    evecs: Array2<f64>,
    inv_evals: Array1<f64>,
}

impl ArrowFactorCache {
    pub fn n_rows(&self) -> usize {
        self.htt_factors.len()
    }

    pub fn htbeta_available(&self) -> bool {
        self.htbeta.is_available()
    }

    /// Whether the Newton solve that produced this cache actually executed on
    /// the device: the device-resident Direct dense solve or the device-resident
    /// matrix-free SAE PCG (whose matvec runs in CUDA kernels). This does NOT
    /// include the injected host-procedural reduced-Schur matvec, whose
    /// arithmetic runs on the CPU even when a CUDA context was opened to build
    /// per-row factors (#1209) — that path sets
    /// `ArrowPcgDiagnostics::injected_host_procedural_matvec` instead. Read-only
    /// routing provenance: lets a fit result record device-vs-CPU as ground
    /// truth instead of inferring it from the runtime probe. Mirrors
    /// `ArrowPcgDiagnostics::used_device_arrow`.
    #[must_use]
    pub fn used_device(&self) -> bool {
        self.pcg_diagnostics.used_device_arrow
    }

    pub fn undamped_factor(&self, row: usize) -> ArrayView2<'_, f64> {
        match &self.htt_factors_undamped {
            ArrowUndampedFactors::SameAsDamped => self.htt_factors.factor(row),
            ArrowUndampedFactors::Owned(factors) => factors.factor(row),
        }
    }

    pub fn undamped_factor_count(&self) -> usize {
        match &self.htt_factors_undamped {
            ArrowUndampedFactors::SameAsDamped => self.htt_factors.len(),
            ArrowUndampedFactors::Owned(factors) => factors.len(),
        }
    }

    pub fn undamped_factors_iter(&self) -> impl Iterator<Item = ArrayView2<'_, f64>> + '_ {
        (0..self.undamped_factor_count()).map(|row| self.undamped_factor(row))
    }

    pub fn compute_undamped_arrow_log_det(&self) -> Option<f64> {
        // When the shared β block is empty (`k == 0`) the joint Hessian is
        // exactly the block diagonal of the per-row latent blocks: there is no
        // reduced Schur complement to form, so the dense Direct path leaves
        // `schur_factor = None` legitimately (not the InexactPCG "never formed
        // the dense K×K factor" case, which has `k > 0`). The log-det is then
        // the per-row sum with a zero (empty `0×0`) Schur contribution. Without
        // this the `schur_factor.as_ref()?` below would return `None` for a
        // β-profiled atom (#1132 euclidean K=4) and starve the REML Laplace
        // normaliser of the joint Hessian log-det it requires.
        let schur = match self.schur_factor.as_ref() {
            Some(schur) => Some(schur),
            None if self.k == 0 => None,
            None => return None,
        };
        if schur.is_some() && !self.schur_factor_is_undamped {
            return None;
        }

        let mut acc = 0.0_f64;
        for l in self.undamped_factors_iter() {
            for i in 0..l.nrows() {
                let d = l[[i, i]];
                if d <= 0.0 || !d.is_finite() {
                    return None;
                }
                acc += 2.0 * d.ln();
            }
        }
        if let Some(schur) = schur {
            for i in 0..schur.nrows() {
                let d = schur[[i, i]];
                if d <= 0.0 || !d.is_finite() {
                    return None;
                }
                acc += 2.0 * d.ln();
            }
        }
        Some(acc)
    }

    /// Undamped joint log-determinant `log|H| = Σ_i log|H_tt^(i)| + log|S|`
    /// using an EXTERNALLY-supplied reduced-Schur term
    /// `schur_log_det = log|S|` instead of a dense `schur_factor` diagonal sum.
    ///
    /// This is the matrix-free large-`k` SAE evidence path: the reduced Schur is
    /// never Cholesky-factored, so `schur_factor` is `None` and `log|S|` comes
    /// from Stochastic Lanczos Quadrature ([`crate::arrow_schur::slq_logdet`]).
    /// The per-row latent-block term is computed exactly as in
    /// [`Self::compute_undamped_arrow_log_det`], with the same ridge,
    /// positivity, and finiteness guards.
    pub fn undamped_arrow_log_det_with_schur(&self, schur_log_det: f64) -> Option<f64> {
        if self.ridge_t != 0.0 || self.ridge_beta != 0.0 {
            return None;
        }
        if !schur_log_det.is_finite() {
            return None;
        }
        let mut acc = 0.0_f64;
        for l in self.undamped_factors_iter() {
            for i in 0..l.nrows() {
                let d = l[[i, i]];
                if d <= 0.0 || !d.is_finite() {
                    return None;
                }
                acc += 2.0 * d.ln();
            }
        }
        acc += schur_log_det;
        Some(acc)
    }

    /// The total length of `delta_t` / IFT output vectors for this cache.
    pub fn delta_t_len(&self) -> usize {
        self.row_offsets[self.n_rows()]
    }

    pub fn apply_htbeta_row(
        &self,
        row: usize,
        delta_beta: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) -> bool {
        let di = if row < self.row_dims.len() {
            self.row_dims[row]
        } else {
            self.d
        };
        if out.len() != di || delta_beta.len() != self.k {
            return false;
        }
        self.htbeta.apply_row(row, delta_beta, out)
    }

    /// Accumulate `out[a] += H_βt^(row)[a, :] · v` for all `a in 0..k`.
    ///
    /// `v` has length `row_dims[row]`; `out` has length `k`. The caller must
    /// zero `out` before the first call if it needs a fresh result.  Returns
    /// `false` when the cache is `Disabled` and no `fallback_op` is provided;
    /// callers must treat the accumulator as invalid in that case.
    pub fn apply_htbeta_row_transpose(
        &self,
        row: usize,
        v: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        fallback_op: Option<&RowHtbetaMatvec>,
    ) -> bool {
        let di = if row < self.row_dims.len() {
            self.row_dims[row]
        } else {
            self.d
        };
        if v.len() != di || out.len() != self.k {
            return false;
        }
        self.htbeta
            .apply_row_transpose_accumulate(row, v, out, di, self.k, fallback_op)
    }

    /// Authoritative evidence joint log-determinant for the exact operator this
    /// cache exposes to selected-inverse and adjoint consumers.
    ///
    /// The factorization path computes this once into
    /// [`Self::joint_hessian_log_det`]. This accessor deliberately does not
    /// reconstruct `Σ_i log|H_tt^(i)| + log|Schur_β|` from loose pieces: a damped
    /// Newton cache can otherwise pair undamped row factors with a damped Schur
    /// solve and silently describe no live operator. Returning only the stored
    /// joint value keeps REML evidence, fixed-state tests, selected inverse, and
    /// `logdet_theta_adjoint` on the same factorization branch.
    pub fn arrow_log_det(&self) -> Option<f64> {
        if self.k > 0 && !self.schur_factor_is_undamped {
            return None;
        }
        self.joint_hessian_log_det
            .filter(|log_det| log_det.is_finite())
    }

    /// Diagonal of the latent (`t`-block) of the *full* bordered-arrow
    /// inverse `(H⁻¹)_tt`, in `delta_t` layout (length [`Self::delta_t_len`]).
    ///
    /// For the bordered arrow Hessian
    /// `H = [[A, B], [Bᵀ, H_ββ]]` with `A = H_tt` (block-diagonal per row,
    /// `A_i = H_tt^(i)`) and `B = H_tβ`, the standard block-inverse identity
    /// gives the `t`-block
    /// `(H⁻¹)_tt = A⁻¹ + A⁻¹ B S⁻¹ Bᵀ A⁻¹`, where
    /// `S = H_ββ − Bᵀ A⁻¹ B` is the Schur complement on `β`. Because `A` is
    /// block-diagonal, the `(i, j)` diagonal entry of `(H⁻¹)_tt` is computed
    /// purely from row `i`'s factor and cross-block:
    ///
    /// ```text
    /// a    = A_i⁻¹ e_j                       (chol_solve on the per-row factor)
    /// [A_i⁻¹]_{jj} = a[j]
    /// w    = B_iᵀ a = H_βt^(i) a             (a K-vector)
    /// z    = S⁻¹ w                           (chol_solve on the Schur factor)
    /// diag = a[j] + w · z
    /// ```
    ///
    /// The UNDAMPED per-row factors ([`Self::undamped_factor`]) are used so
    /// the result is the inverse of the *true* `H_tt`, not the LM-damped
    /// `H_tt + ridge_t·I` — same rationale the IFT predictor docstring gives
    /// at the top of this struct.
    ///
    /// # Consuming the diagonal as a per-(atom, axis) trace
    ///
    /// `(H⁻¹)_tt` is the latent covariance block. The selected-inverse trace
    /// for a contiguous group of latent coordinates (e.g. one atom's rows, or
    /// one axis across rows) is simply the sum of the returned diagonal entries
    /// over those `row_offsets[i] + j` indices — no off-diagonal terms are
    /// needed for the trace `tr[(H⁻¹)_tt · D]` against a diagonal selector `D`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowSchurError::SchurFactorFailed`] when this cache has no
    /// dense Schur factor or no usable `H_βt` coupling — i.e. it was produced
    /// by an [`ArrowSolverMode::InexactPCG`] solve (no dense `K × K` factor) or
    /// by a `Disabled` `htbeta` cache. The selected-inverse block-trace is not
    /// yet supported for the matrix-free PCG mode; that branch needs a separate
    /// Lanczos/Hutchinson estimator.
    pub fn latent_block_inverse_diagonal(&self) -> Result<Array1<f64>, ArrowSchurError> {
        let Some(_schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "latent_block_inverse_diagonal requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        if !self.schur_factor_is_undamped {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "latent_block_inverse_diagonal refuses a Schur factor that was not \
                         built from the undamped evidence row factors"
                    .to_string(),
            });
        }
        if !self.htbeta_available() {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "latent_block_inverse_diagonal requires the H_tβ coupling, \
                         but this cache's htbeta is Disabled"
                    .to_string(),
            });
        }
        let n = self.undamped_factor_count();
        let total_len = self.delta_t_len();
        let mut out = Array1::<f64>::zeros(total_len);
        // Per-row scratch, sized to the max latent dim / K.
        let mut e_j = Array1::<f64>::zeros(self.d);
        let mut w = Array1::<f64>::zeros(self.k);
        for i in 0..n {
            let di = self.row_dims[i];
            let row_base = self.row_offsets[i];
            let factor = self.undamped_factor(i);
            for j in 0..di {
                // a = A_i⁻¹ e_j.
                for c in 0..di {
                    e_j[c] = 0.0;
                }
                e_j[j] = 1.0;
                let e_j_slice = e_j.slice(ndarray::s![..di]).to_owned();
                let a = cholesky_solve_vector(factor, &e_j_slice);
                // w = H_βt^(i) a (a K-vector); accumulator must start zeroed.
                w.fill(0.0);
                if !self.apply_htbeta_row_transpose(i, a.view(), &mut w, None) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "latent_block_inverse_diagonal: H_βt^({i}) apply failed \
                             (htbeta cache could not supply row {i})"
                        ),
                    });
                }
                // z = S⁻¹ w; correction = w · z.
                let z = self.schur_inverse_apply(w.view())?;
                let mut corr = 0.0_f64;
                for c in 0..self.k {
                    corr += w[c] * z[c];
                }
                out[row_base + j] = a[j] + corr;
            }
        }
        Ok(out)
    }

    /// Solve the full bordered-arrow system `H·u = w` on the cached factor
    /// (#1006): `w` arrives in arrow layout — `w_t` flat per
    /// [`Self::delta_t_len`] / `row_offsets`, `w_beta` of length `K` — and the
    /// solution comes back in the same layout. Standard block elimination on
    /// the SAME factors whose log-determinant the evidence reports:
    ///
    /// ```text
    ///   y_i      = H_tt^(i)⁻¹ · w_t^(i)
    ///   r_β      = w_β − Σ_i H_βt^(i) · y_i
    ///   u_β      = Schur⁻¹ · r_β
    ///   u_t^(i)  = y_i − H_tt^(i)⁻¹ · (H_tβ^(i) · u_β)
    /// ```
    ///
    /// This is the IFT / adjoint back-solve the analytic outer ρ-gradient
    /// consumes: `u_j = H⁻¹ (∂g/∂ρ_j)` per outer coordinate and the
    /// `H⁻¹`-side of the third-order correction `−½·Γᵀ·H⁻¹·(∂g/∂ρ_j)`.
    /// Contract: the cache must be the ridge-0 Direct evidence factor
    /// (undamped per-row factors + dense Schur), so the solve is against the
    /// criterion's own `H` — never a damped surrogate (that would desync the
    /// gradient from the reported evidence).
    ///
    pub fn full_inverse_apply(
        &self,
        w_t: ArrayView1<'_, f64>,
        w_beta: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let total_len = self.delta_t_len();
        if w_t.len() != total_len || w_beta.len() != self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "full_inverse_apply: rhs shapes (w_t={}, w_beta={}) != (delta_t_len={}, K={})",
                    w_t.len(),
                    w_beta.len(),
                    total_len,
                    self.k
                ),
            });
        }
        let n = self.undamped_factor_count();
        // Forward pass: y_i = H_tt^(i)⁻¹ w_t^(i), accumulating the border RHS.
        let mut y = Array1::<f64>::zeros(total_len);
        let mut r_beta = w_beta.to_owned();
        for i in 0..n {
            let di = self.row_dims[i];
            let base = self.row_offsets[i];
            let factor = self.undamped_factor(i);
            let w_row = w_t.slice(ndarray::s![base..base + di]).to_owned();
            let y_row = cholesky_solve_vector(factor, &w_row);
            if self.k > 0 {
                // r_β −= H_βt^(i) y_i: accumulate into a scratch then subtract,
                // because the helper ACCUMULATES (+=) into its output.
                let mut acc = Array1::<f64>::zeros(self.k);
                if !self.apply_htbeta_row_transpose(i, y_row.view(), &mut acc, None) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "full_inverse_apply: H_βt^({i}) apply failed (htbeta cache \
                             could not supply row {i}; htbeta={:?}, di={}, k={})",
                            self.htbeta,
                            self.row_dims.get(i).copied().unwrap_or(self.d),
                            self.k
                        ),
                    });
                }
                for c in 0..self.k {
                    r_beta[c] -= acc[c];
                }
            }
            for j in 0..di {
                y[base + j] = y_row[j];
            }
        }
        // Border solve + back-substitution.
        let u_beta = if self.k > 0 {
            self.schur_inverse_apply(r_beta.view())?
        } else {
            Array1::<f64>::zeros(0)
        };
        let mut u_t = y;
        if self.k > 0 {
            let mut cross = Array1::<f64>::zeros(self.d);
            for i in 0..n {
                let di = self.row_dims[i];
                let base = self.row_offsets[i];
                let mut cross_row = cross.slice_mut(ndarray::s![..di]);
                cross_row.fill(0.0);
                let mut cross_owned = cross_row.to_owned();
                if !self.apply_htbeta_row(i, u_beta.view(), &mut cross_owned) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "full_inverse_apply: H_tβ^({i}) apply failed (htbeta cache \
                             could not supply row {i})"
                        ),
                    });
                }
                let factor = self.undamped_factor(i);
                let corr = cholesky_solve_vector(factor, &cross_owned);
                for j in 0..di {
                    u_t[base + j] -= corr[j];
                }
            }
        }
        Ok((u_t, u_beta))
    }

    /// Apply the β-block of the full inverse, `(H⁻¹)_ββ · rhs = S_β⁻¹ · rhs`,
    /// where `S_β` is the Schur complement on β whose Cholesky factor this
    /// cache holds in [`Self::schur_factor`].
    ///
    /// For the bordered arrow Hessian `H = [[A, B], [Bᵀ, H_ββ]]`, the
    /// β-block of `H⁻¹` is exactly the inverse of the Schur complement
    /// `S_β = H_ββ − Bᵀ A⁻¹ B`. One Cholesky back-substitution per call,
    /// reusing the cached factor; `rhs` and the returned vector both have
    /// length `K`.
    ///
    /// This is the general single-solve primitive for the β border. Callers
    /// that need a Schur-inverse trace `tr(S_β⁻¹ M)` against a structured
    /// penalty `M` (e.g. the SAE λ_smooth Fellner-Schall step, where
    /// `M = blockdiag_k(λ_k S_k ⊗ I_p)`) build it as
    /// `Σ_col e_colᵀ S_β⁻¹ M e_col` — apply this to each column of `M`
    /// (exploiting whatever sparsity `M` has) and read off `result[col]`.
    /// Keeping `M`'s layout on the caller side avoids coupling this solver
    /// to penalty-op types.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowSchurError::SchurFactorFailed`] when this cache has no
    /// dense Schur factor (an [`ArrowSolverMode::InexactPCG`] solve) — the
    /// same not-yet-supported branch as [`Self::latent_block_inverse_diagonal`]
    /// — or when `rhs.len() != k`.
    ///
    pub fn schur_inverse_apply(
        &self,
        rhs: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, ArrowSchurError> {
        let Some(schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_apply requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        if !self.schur_factor_is_undamped {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_apply refuses a Schur factor that was not built from \
                         the undamped evidence row factors"
                    .to_string(),
            });
        }
        if rhs.len() != self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "schur_inverse_apply: rhs length {} != K {}",
                    rhs.len(),
                    self.k
                ),
            });
        }
        if self.beta_schur_deflation.is_some() {
            let deflated = self.deflated_schur_pseudo_inverse()?;
            return Ok(self.apply_deflated_pseudo_inverse(&deflated, rhs));
        }
        let rhs_owned = match self.beta_gauge_quotient.as_ref() {
            Some(quotient) => quotient.project_complement(rhs),
            None => rhs.to_owned(),
        };
        let solved = cholesky_solve_vector(schur_factor, &rhs_owned);
        Ok(match self.beta_gauge_quotient.as_ref() {
            Some(quotient) => quotient.project_complement(solved.view()),
            None => solved,
        })
    }

    /// Dense principal sub-block of the β-block of the full inverse,
    /// `(H⁻¹)_ββ[block, block] = S_β⁻¹[block, block]`, shape `(W, W)` with
    /// `W = block.len()`.
    ///
    /// For the bordered arrow Hessian `H = [[A, B], [Bᵀ, H_ββ]]`, the β-block
    /// of `H⁻¹` is exactly `S_β⁻¹` (the inverse of the Schur complement whose
    /// Cholesky factor this cache holds). This returns the contiguous
    /// `block × block` sub-block — e.g. one SAE atom's decoder coefficients via
    /// [`gam_terms::sae::manifold::SaeManifoldTerm::beta_block_offsets`] — by
    /// solving `S_β x = e_j` for each `j ∈ block` (reusing the cached factor)
    /// and gathering the `block` rows of each solution column. `W`
    /// back-substitutions of size `K`; the result is symmetrized to clear
    /// back-substitution rounding asymmetry. Up to a dispersion scale `φ`, this
    /// block is the joint posterior covariance `Cov(β_block)` of those
    /// coefficients with the latent coordinates already marginalized out (that
    /// is precisely what Schur-eliminating the per-row `t`-blocks does).
    ///
    /// Same dense-Schur requirement / error contract as
    /// [`Self::schur_inverse_apply`]; additionally errors when `block` runs past
    /// `K`.
    pub fn schur_inverse_block(
        &self,
        block: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, ArrowSchurError> {
        let Some(_schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_block requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        if !self.schur_factor_is_undamped {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_block refuses a Schur factor that was not built from \
                         the undamped evidence row factors"
                    .to_string(),
            });
        }
        if block.end > self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "schur_inverse_block: block end {} exceeds K {}",
                    block.end, self.k
                ),
            });
        }
        let w = block.len();
        let mut out = Array2::<f64>::zeros((w, w));
        let mut e_j = Array1::<f64>::zeros(self.k);
        for (jc, j) in block.clone().enumerate() {
            e_j.fill(0.0);
            e_j[j] = 1.0;
            let col = self.schur_inverse_apply(e_j.view())?;
            for (ic, i) in block.clone().enumerate() {
                out[[ic, jc]] = col[i];
            }
        }
        // S_β⁻¹ is symmetric; symmetrize to clear back-substitution rounding.
        for ic in 0..w {
            for jc in (ic + 1)..w {
                let avg = 0.5 * (out[[ic, jc]] + out[[jc, ic]]);
                out[[ic, jc]] = avg;
                out[[jc, ic]] = avg;
            }
        }
        Ok(out)
    }

    /// Deflation-aware selected inverse of the cached β-Schur complement — a
    /// drop-in for [`Self::schur_inverse_apply`] that pseudo-inverts across the
    /// numerically-null curvature directions instead of dividing by them.
    ///
    /// # Why this exists (the λ→0 EDF divergence)
    ///
    /// The REML EDF/log-det-trace term contracts `(H⁻¹)_ββ` against `λS`. At the
    /// ρ lower face a decoder direction can be null in BOTH the data
    /// (`J_ββ ≈ 0`) AND the penalty (`s ≈ 0`), making `S_β = J + λS` singular
    /// along it. The plain [`Self::schur_inverse_apply`] then divides by a
    /// ~zero pivot and returns `Inf`/`NaN` (the value stays finite — only this
    /// `H⁻¹`-contraction blows up). This method instead forms the spectral
    /// pseudo-inverse `M⁺` of the SAME operator `M = L Lᵀ` the plain path
    /// inverts, dropping every eigen-direction at or below the solver's
    /// canonical rank floor `SPECTRAL_DEFLATION_REL_FLOOR · max|λ|` (the exact
    /// threshold [`factor_spectral_deflated_criterion_row`] and the per-row
    /// gauge deflation already use — NOT a new epsilon and NOT a λ-smoothing
    /// floor). A doubly-null direction (`j ≈ 0 ∧ s ≈ 0`) deflates to `0` (it is
    /// unidentifiable, not a real DOF); a penalty-only direction survives. The
    /// result is finite by construction.
    ///
    /// # Interior equivalence
    ///
    /// Away from the boundary every eigenvalue of `M` sits orders of magnitude
    /// above the floor, so NO direction deflates and `M⁺ = M⁻¹` to round-off —
    /// this returns the plain selected inverse with no silent bias. Only the
    /// λ→0 face deflates. The exact-Newton path keeps calling the plain
    /// [`Self::schur_inverse_apply`] and is byte-for-byte unchanged.
    ///
    /// # Errors
    ///
    /// Same dense-Schur / undamped-factor / `rhs.len() != K` contract as
    /// [`Self::schur_inverse_apply`], plus a failed symmetric eigendecomposition
    /// of the reconstructed `M`.
    pub fn schur_inverse_apply_deflated(
        &self,
        rhs: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, ArrowSchurError> {
        if rhs.len() != self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "schur_inverse_apply_deflated: rhs length {} != K {}",
                    rhs.len(),
                    self.k
                ),
            });
        }
        let deflated = self.deflated_schur_pseudo_inverse()?;
        Ok(self.apply_deflated_pseudo_inverse(&deflated, rhs))
    }

    /// Precompute the deflated spectral pseudo-inverse ONCE and return a
    /// reusable applier — the many-RHS form of
    /// [`Self::schur_inverse_apply_deflated`]. The EDF trace contracts
    /// `(H⁻¹)_ββ` against one `λS⊗I` column per basis coefficient (`Σ_k M_k·r_k`
    /// columns total); recomputing the `O(K³)` eigendecomposition per column
    /// would multiply that cost by the border width for no reason. Each apply
    /// through the returned closure is `O(K²)` (two dense mat-vecs through the
    /// eigenbasis), identical in complexity to the plain
    /// [`Self::schur_inverse_apply`] back-substitution it replaces.
    ///
    /// Same deflation semantics, contract, and errors as
    /// [`Self::schur_inverse_apply_deflated`]; the closure itself is
    /// infallible (rhs length is the caller's loop invariant — a wrong length
    /// panics in the underlying gemv shape check rather than dividing by a
    /// null pivot).
    pub fn schur_deflated_applier(
        &self,
    ) -> Result<impl Fn(ArrayView1<'_, f64>) -> Array1<f64> + '_, ArrowSchurError> {
        let deflated = self.deflated_schur_pseudo_inverse()?;
        Ok(move |rhs: ArrayView1<'_, f64>| self.apply_deflated_pseudo_inverse(&deflated, rhs))
    }

    /// Deflation-aware dense principal sub-block of `(H⁻¹)_ββ` — the drop-in for
    /// [`Self::schur_inverse_block`] used by the per-atom EDF trace. Identical
    /// contract, but each column is solved through the spectral pseudo-inverse
    /// (see [`Self::schur_inverse_apply_deflated`]) so a boundary atom with a
    /// doubly-null decoder direction yields a finite block instead of `NaN`.
    ///
    /// The eigendecomposition of `M = L Lᵀ` is computed ONCE and reused across
    /// all `W = block.len()` columns.
    pub fn schur_inverse_block_deflated(
        &self,
        block: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, ArrowSchurError> {
        if block.end > self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "schur_inverse_block_deflated: block end {} exceeds K {}",
                    block.end, self.k
                ),
            });
        }
        let deflated = self.deflated_schur_pseudo_inverse()?;
        let w = block.len();
        let mut out = Array2::<f64>::zeros((w, w));
        let mut e_j = Array1::<f64>::zeros(self.k);
        for (jc, j) in block.clone().enumerate() {
            e_j.fill(0.0);
            e_j[j] = 1.0;
            let col = self.apply_deflated_pseudo_inverse(&deflated, e_j.view());
            for (ic, i) in block.clone().enumerate() {
                out[[ic, jc]] = col[i];
            }
        }
        // (H⁻¹)_ββ is symmetric; symmetrize to clear round-off asymmetry.
        for ic in 0..w {
            for jc in (ic + 1)..w {
                let avg = 0.5 * (out[[ic, jc]] + out[[jc, ic]]);
                out[[ic, jc]] = avg;
                out[[jc, ic]] = avg;
            }
        }
        Ok(out)
    }

    /// Reconstruct the SPD operator `M = L Lᵀ` this cache inverts (the plain
    /// [`Self::schur_inverse_apply`] solves `M x = rhs`; when a β-gauge quotient
    /// is installed `M = P S P + Q Qᵀ`), symmetric-eigendecompose it, and deflate
    /// every eigen-direction at or below the canonical rank floor
    /// `SPECTRAL_DEFLATION_REL_FLOOR · max|λ|` (with the same hysteresis band the
    /// per-row spectral deflation uses, so a direction parked at the floor does
    /// not flicker across a ρ-walk).
    fn deflated_schur_pseudo_inverse(
        &self,
    ) -> Result<DeflatedSchurPseudoInverse, ArrowSchurError> {
        let Some(schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_apply_deflated requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        if !self.schur_factor_is_undamped {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_apply_deflated refuses a Schur factor that was not built \
                         from the undamped evidence row factors"
                    .to_string(),
            });
        }
        let k = self.k;
        if let Some(spectrum) = self.beta_schur_deflation.as_ref() {
            if spectrum.evecs.dim() != (k, k)
                || spectrum.raw_evals.len() != k
                || spectrum.cond_evals.len() != k
                || spectrum.deflated.len() != k
            {
                return Err(ArrowSchurError::SchurFactorFailed {
                    reason: "cached β-Schur deflation spectrum has incoherent dimensions"
                        .to_string(),
                });
            }
            let mut inv_evals = Array1::<f64>::zeros(k);
            for eig_idx in 0..k {
                if spectrum.deflated[eig_idx] {
                    continue;
                }
                let lambda = spectrum.cond_evals[eig_idx];
                if !(lambda.is_finite() && lambda > 0.0) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "cached β-Schur kept eigenvalue {eig_idx} is not positive finite: {lambda:e}"
                        ),
                    });
                }
                inv_evals[eig_idx] = 1.0 / lambda;
            }
            return Ok(DeflatedSchurPseudoInverse {
                evecs: spectrum.evecs.clone(),
                inv_evals,
            });
        }
        // Reconstruct `M = L Lᵀ` from the LOWER triangle only (the strict-upper
        // entries of the stored factor are not part of the Cholesky factor and
        // must not enter the product).
        let mut lower = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..=i {
                lower[[i, j]] = schur_factor[[i, j]];
            }
        }
        let m = lower.dot(&lower.t());
        let (evals, evecs) = m.eigh(Side::Lower).map_err(|err| {
            ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "schur_inverse_apply_deflated: symmetric eigendecomposition of the \
                     reconstructed β-Schur operator failed: {err:?}"
                ),
            }
        })?;
        let max_abs = evals.iter().fold(0.0_f64, |acc, &v| {
            if v.is_finite() { acc.max(v.abs()) } else { acc }
        });
        if !(max_abs.is_finite() && max_abs > 0.0) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_apply_deflated: reconstructed β-Schur operator has no \
                         finite positive spectrum"
                    .to_string(),
            });
        }
        // No evidence deflation was performed, so every factor eigenvalue is
        // part of the value and must remain part of the inverse. Boundary rank
        // decisions are made once during evidence factorization and carried in
        // `beta_schur_deflation`; inferring a new mask from the conditioned
        // factor would desynchronize the value and its gradient.
        let inv_evals = evals.mapv(|lambda| 1.0 / lambda);
        Ok(DeflatedSchurPseudoInverse { evecs, inv_evals })
    }

    /// Apply a precomputed [`DeflatedSchurPseudoInverse`] to one RHS, mirroring
    /// the β-gauge-quotient complement projection of the plain
    /// [`Self::schur_inverse_apply`]: `P M⁺ P rhs` when a quotient is installed,
    /// `M⁺ rhs` otherwise.
    fn apply_deflated_pseudo_inverse(
        &self,
        deflated: &DeflatedSchurPseudoInverse,
        rhs: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let rhs_owned = match self.beta_gauge_quotient.as_ref() {
            Some(quotient) => quotient.project_complement(rhs),
            None => rhs.to_owned(),
        };
        // M⁺ r = Q · diag(1/λ̃) · Qᵀ r.
        let coeffs = deflated.evecs.t().dot(&rhs_owned);
        let scaled = &coeffs * &deflated.inv_evals;
        let solved = deflated.evecs.dot(&scaled);
        match self.beta_gauge_quotient.as_ref() {
            Some(quotient) => quotient.project_complement(solved.view()),
            None => solved,
        }
    }
}

/// Per-chunk stacked Schur subtraction for the parallel assembly fan-out.
///
/// Dense rows are appended into stacked `(Σd × k)` factors and subtracted with
/// ONE sequential SIMD GEMM per chunk (`s_part -= Lᵀ R`) — the CPU mirror of
/// the device `tile_schur_partial` stacking — while rows with sparse column
/// support keep the nnz-scaled scatter (#1995), which beats a dense GEMM
/// there.
///
/// The crossover is derived, not tuned. The scatter costs
/// `Σ_c nnz_l(c)·nnz_r(c)` scalar FMAs against a randomly indexed `k×k`
/// accumulator (unvectorizable), while the row's share of the stacked GEMM is
/// `d·k²` FMAs at SIMD throughput — ≈8 f64 FMAs per cycle (4-lane vectors,
/// dual issue) on both x86-64/AVX2 and aarch64/NEON. The GEMM therefore wins
/// once `scatter_flops > d·k²/8`. Pricing this needs one `O(d·k)` support
/// count, the same scan the scatter pays to build its active lists, so a
/// "scatter" verdict wastes nothing and a "stack" verdict wastes only the
/// count.
///
/// Numerics: the stacked GEMM reassociates the within-chunk row sum relative
/// to the per-row scatter — the same reassociation class as the existing
/// chunk-partial fold and the device stacking, and deterministic run-to-run
/// (`Par::Seq` inside the worker per #1557).
struct ChunkSchurStack {
    left: Vec<f64>,
    right: Vec<f64>,
    stacked_rows: usize,
    k: usize,
}

impl ChunkSchurStack {
    fn new(k: usize) -> Self {
        Self {
            left: Vec::new(),
            right: Vec::new(),
            stacked_rows: 0,
            k,
        }
    }

    /// Either scatter this row's Schur contribution immediately (sparse
    /// support) or append its factors to the chunk stack (dense support).
    fn subtract_or_stack(
        &mut self,
        backend: &CpuBatchedBlockSolver,
        s_part: &mut Array2<f64>,
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) {
        let k = self.k;
        let d = left.nrows();
        // Caller-contract shape checks; real asserts (scanner bans debug_*),
        // and trivially cheap next to the k x k scatter below.
        assert_eq!(left.ncols(), k, "scatter: left must be (d, k)");
        assert_eq!(right.dim(), (d, k), "scatter: right must be (d, k)");
        assert_eq!(s_part.dim(), (k, k), "scatter: s_part must be (k, k)");
        let mut scatter_flops = 0usize;
        for c in 0..d {
            let mut nnz_left = 0usize;
            let mut nnz_right = 0usize;
            for col in 0..k {
                nnz_left += usize::from(left[[c, col]] != 0.0);
                nnz_right += usize::from(right[[c, col]] != 0.0);
            }
            scatter_flops += nnz_left * nnz_right;
        }
        if scatter_flops <= d * k * k / 8 {
            backend.block_gemm_subtract(s_part, left, right);
            return;
        }
        for source in [(left, &mut self.left), (right, &mut self.right)] {
            let (matrix, stack) = source;
            if let Some(values) = matrix.as_slice() {
                stack.extend_from_slice(values);
            } else {
                stack.extend(matrix.iter().copied());
            }
        }
        self.stacked_rows += d;
    }

    /// Subtract every stacked row in one sequential SIMD GEMM.
    fn flush(&mut self, s_part: &mut Array2<f64>) {
        if self.stacked_rows == 0 {
            return;
        }
        let shape = (self.stacked_rows, self.k);
        let left = ndarray::ArrayView2::from_shape(shape, self.left.as_slice())
            .expect("ChunkSchurStack left buffer matches its recorded shape");
        let right = ndarray::ArrayView2::from_shape(shape, self.right.as_slice())
            .expect("ChunkSchurStack right buffer matches its recorded shape");
        let product =
            gam_linalg::faer_ndarray::fast_atb_with_parallelism(&left, &right, faer::Par::Seq);
        *s_part -= &product;
        self.left.clear();
        self.right.clear();
        self.stacked_rows = 0;
    }
}
