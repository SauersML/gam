use super::*;

#[derive(Debug, Clone)]
pub struct SaeArrowVector {
    pub t: Array1<f64>,
    pub beta: Array1<f64>,
}

pub(crate) struct DeflatedArrowSolver<'a> {
    pub(crate) cache: &'a ArrowFactorCache,
    pub(crate) gauge_basis: Vec<Array1<f64>>,
    pub(crate) gauge_response_physical: Vec<Array1<f64>>,
    pub(crate) woodbury_factor: Option<FaerCholeskyFactor>,
    pub(crate) gauge_stiffness_recip: f64,
}

impl<'a> DeflatedArrowSolver<'a> {
    pub(crate) fn plain(cache: &'a ArrowFactorCache) -> Self {
        Self {
            cache,
            gauge_basis: Vec::new(),
            gauge_response_physical: Vec::new(),
            woodbury_factor: None,
            gauge_stiffness_recip: 0.0,
        }
    }

    pub(crate) fn from_orthonormal_gauges(
        cache: &'a ArrowFactorCache,
        gauge_basis: Vec<Array1<f64>>,
        stiffness: f64,
    ) -> Result<Self, String> {
        if gauge_basis.is_empty() {
            return Ok(Self::plain(cache));
        }
        if !(stiffness.is_finite() && stiffness > 0.0) {
            return Err(format!(
                "DeflatedArrowSolver: gauge stiffness must be finite and positive; got {stiffness}"
            ));
        }
        let full_len = cache.delta_t_len() + cache.k;
        let mut gauge_responses = Vec::with_capacity(gauge_basis.len());
        for gauge in &gauge_basis {
            if gauge.len() != full_len {
                return Err(format!(
                    "DeflatedArrowSolver: gauge length {} != cache full length {full_len}",
                    gauge.len()
                ));
            }
            let (sol_t, sol_beta) = cache
                .full_inverse_apply(
                    gauge.slice(s![..cache.delta_t_len()]),
                    gauge.slice(s![cache.delta_t_len()..]),
                )
                .map_err(|err| format!("DeflatedArrowSolver: gauge back-solve: {err}"))?;
            gauge_responses.push(flatten_arrow_parts(sol_t.view(), sol_beta.view()));
        }

        let rank = gauge_basis.len();
        let stiffness_recip = stiffness.recip();
        let mut gauge_metric = Array2::<f64>::zeros((rank, rank));
        let mut woodbury = Array2::<f64>::eye(rank);
        for i in 0..rank {
            woodbury[[i, i]] *= stiffness_recip;
            for j in 0..rank {
                let value = gauge_basis[i].dot(&gauge_responses[j]);
                gauge_metric[[i, j]] = value;
                woodbury[[i, j]] += value;
            }
        }
        let woodbury_factor = woodbury
            .cholesky(Side::Lower)
            .map_err(|err| format!("DeflatedArrowSolver: gauge Woodbury factor failed: {err}"))?;
        let mut gauge_response_physical = gauge_responses;
        for j in 0..rank {
            for i in 0..rank {
                let coeff = gauge_metric[[i, j]];
                for row in 0..full_len {
                    gauge_response_physical[j][row] -= coeff * gauge_basis[i][row];
                }
            }
        }
        Ok(Self {
            cache,
            gauge_basis,
            gauge_response_physical,
            woodbury_factor: Some(woodbury_factor),
            gauge_stiffness_recip: stiffness_recip,
        })
    }

    pub(crate) fn solve(
        &self,
        rhs_t: ArrayView1<'_, f64>,
        rhs_beta: ArrayView1<'_, f64>,
    ) -> Result<SaeArrowVector, String> {
        let (sol_t, sol_beta) = self
            .cache
            .full_inverse_apply(rhs_t, rhs_beta)
            .map_err(|err| format!("DeflatedArrowSolver: full inverse: {err}"))?;
        let Some(factor) = self.woodbury_factor.as_ref() else {
            return Ok(SaeArrowVector {
                t: sol_t,
                beta: sol_beta,
            });
        };

        let full_len = self.cache.delta_t_len() + self.cache.k;
        let mut flat = flatten_arrow_parts(sol_t.view(), sol_beta.view());
        if flat.len() != full_len {
            return Err(format!(
                "DeflatedArrowSolver: solution length {} != cache full length {full_len}",
                flat.len()
            ));
        }
        let mut gauge_coeffs = Array1::<f64>::zeros(self.gauge_basis.len());
        for (idx, gauge) in self.gauge_basis.iter().enumerate() {
            gauge_coeffs[idx] = gauge.dot(&flat);
        }
        let weights = factor.solvevec(&gauge_coeffs);
        for (gauge, &coeff) in self.gauge_basis.iter().zip(gauge_coeffs.iter()) {
            for i in 0..flat.len() {
                flat[i] -= gauge[i] * coeff;
            }
        }
        for (response, &weight) in self.gauge_response_physical.iter().zip(weights.iter()) {
            for i in 0..flat.len() {
                flat[i] -= response[i] * weight;
            }
        }
        for (gauge, &weight) in self.gauge_basis.iter().zip(weights.iter()) {
            let coeff = self.gauge_stiffness_recip * weight;
            for i in 0..flat.len() {
                flat[i] += gauge[i] * coeff;
            }
        }
        Ok(SaeArrowVector {
            t: flat.slice(s![..self.cache.delta_t_len()]).to_owned(),
            beta: flat.slice(s![self.cache.delta_t_len()..]).to_owned(),
        })
    }

    /// Per-row latent-block inverse diagonal with the UNIT-stiffness deflated
    /// subspace REMOVED — the kept-subspace selected inverse the outer ρ/θ
    /// gradient diagonal traces must contract against.
    ///
    /// [`Self::latent_inverse_diagonal`] returns the diagonal of the DEFLATED
    /// inverse, which assigns `1/λ̃ = 1` to every per-row direction `vᵢ` that the
    /// undamped evidence factor stiffened to unit curvature; a `½ tr(H⁻¹ ∂H/∂ρ)`
    /// diagonal contraction against it therefore spuriously includes
    /// `Σ_i vᵢ[s]²` at slot `s`, a ρ/θ-independent contribution that must be 0.
    /// This variant subtracts the per-row deflated outer-product diagonal
    /// `Σ_i vᵢ[s]²` so the diagonal traces (ARD precision, IBP/softmax assignment
    /// log-strength) see only the kept subspace. The deflated subspace's β-Schur
    /// coupling is higher order and left to the per-block subtraction the
    /// off-diagonal (`solve`-based) traces apply directly.
    pub(crate) fn latent_inverse_diagonal_kept(&self) -> Result<Array1<f64>, String> {
        let mut out = self.latent_inverse_diagonal()?;
        let cache = self.cache;
        for (row, dirs) in cache.deflated_row_directions.iter().enumerate() {
            if dirs.is_empty() {
                continue;
            }
            let base = cache.row_offsets[row];
            for v in dirs {
                for s in 0..v.len() {
                    if base + s < out.len() {
                        out[base + s] -= v[s] * v[s];
                    }
                }
            }
        }
        Ok(out)
    }

    /// #932 FRONT C — whether the cheap row-local Takahashi selected inverse
    /// ([`Self::beta_inv`] / [`Self::selected_inverse_row_blocks`]) reproduces
    /// `solve`'s selected entries EXACTLY. It does so only on the plain bordered
    /// arrow: when a gauge Woodbury deflation is active (`woodbury_factor`) the
    /// `solve` output carries the rank-`R` gauge correction the row-local blocks
    /// omit, and when a #1038 cross-row IBP Woodbury is present the cache's
    /// per-row factors are the NO-SELF base `H₀'` (not the full operator). In
    /// either case callers MUST fall back to the per-row `solve` loop — the
    /// row-local blocks are NOT valid there.
    pub(crate) fn plain_selected_inverse_available(&self) -> bool {
        self.woodbury_factor.is_none() && self.cache.cross_row_woodbury.is_none()
    }

    /// #932 FRONT C — the full `(H⁻¹)_ββ = S⁻¹` block (`K×K`), formed ONCE per
    /// outer step from the cached dense Schur factor (no per-column full-system
    /// `solve`). On the plain arrow this equals the `beta_inv` the logdet /
    /// α-trace consumers used to build with `K` calls to [`Self::solve`] with
    /// unit β-RHS. ONLY valid when [`Self::plain_selected_inverse_available`].
    pub(crate) fn beta_inv(&self) -> Result<Array2<f64>, String> {
        let k = self.cache.k;
        if k == 0 {
            return Ok(Array2::<f64>::zeros((0, 0)));
        }
        self.cache
            .schur_inverse_block(0..k)
            .map_err(|err| format!("DeflatedArrowSolver::beta_inv: {err}"))
    }

    /// #932 FRONT C — row-local Takahashi selected inverse of the PLAIN bordered
    /// arrow: returns this row's own `(H⁻¹)_tt` block (`q×q`) and its `(H⁻¹)_tβ`
    /// block (`q×K`) WITHOUT the O(n) full-system sweep that one
    /// [`Self::solve`] per unit RHS performs. Mirrors
    /// `ArrowFactorCache::latent_block_inverse_diagonal` (system.rs) but returns
    /// the full blocks rather than only the diagonal. With `A_i =
    /// undamped_factor(i)`, `B_i = H_tβ^(i)`, `G_i = A_i⁻¹ B_i`, `S⁻¹ = beta_inv`:
    ///
    /// ```text
    ///   (H⁻¹)_tt[i,i] = A_i⁻¹ + G_i S⁻¹ G_iᵀ
    ///   (H⁻¹)_tβ[i]   = −G_i S⁻¹
    /// ```
    ///
    /// Touches ONLY row `i`'s own factor, its `H_tβ^(i)` coupling, and the shared
    /// `S⁻¹` — O(q·(q+K)) per row, no `n`-sweep. ONLY valid when
    /// [`Self::plain_selected_inverse_available`]; pass the `S⁻¹` from
    /// [`Self::beta_inv`].
    pub(crate) fn selected_inverse_row_blocks(
        &self,
        row: usize,
        beta_inv: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let cache = self.cache;
        let q = cache.row_dims[row];
        let k = cache.k;
        let factor = cache.undamped_factor(row);

        // A_i⁻¹ (q×q): solve A_i x = e_j per column.
        let mut a_inv = Array2::<f64>::zeros((q, q));
        let mut e_j = Array1::<f64>::zeros(q);
        for j in 0..q {
            e_j.fill(0.0);
            e_j[j] = 1.0;
            let col = cholesky_solve_vector(factor, e_j.view());
            for r in 0..q {
                a_inv[[r, j]] = col[r];
            }
        }

        if k == 0 {
            return Ok((a_inv, Array2::<f64>::zeros((q, 0))));
        }

        // G_i = A_i⁻¹ B_i (q×K): column c is A_i⁻¹ (B_i e_c), where B_i e_c is the
        // c-th column of H_tβ^(i) recovered via `apply_htbeta_row`.
        let mut g = Array2::<f64>::zeros((q, k));
        let mut e_c = Array1::<f64>::zeros(k);
        let mut b_col = Array1::<f64>::zeros(q);
        for c in 0..k {
            e_c.fill(0.0);
            e_c[c] = 1.0;
            b_col.fill(0.0);
            if !cache.apply_htbeta_row(row, e_c.view(), &mut b_col) {
                return Err(format!(
                    "DeflatedArrowSolver::selected_inverse_row_blocks: H_tβ^({row}) apply failed"
                ));
            }
            let g_col = cholesky_solve_vector(factor, b_col.view());
            for r in 0..q {
                g[[r, c]] = g_col[r];
            }
        }

        // GS = G_i S⁻¹ (q×K).
        let mut gs = Array2::<f64>::zeros((q, k));
        for r in 0..q {
            for m in 0..k {
                let mut acc = 0.0_f64;
                for n in 0..k {
                    acc += g[[r, n]] * beta_inv[[n, m]];
                }
                gs[[r, m]] = acc;
            }
        }

        // (H⁻¹)_tβ[i] = −G_i S⁻¹ = −GS, layout [col, b].
        let mut inv_vbeta = Array2::<f64>::zeros((q, k));
        for col in 0..q {
            for b in 0..k {
                inv_vbeta[[col, b]] = -gs[[col, b]];
            }
        }

        // (H⁻¹)_tt[i,i] = A_i⁻¹ + G_i S⁻¹ G_iᵀ = A_i⁻¹ + GS·Gᵀ, layout [r, col].
        let mut inv_vv = a_inv;
        for r in 0..q {
            for col in 0..q {
                let mut acc = 0.0_f64;
                for m in 0..k {
                    acc += gs[[r, m]] * g[[col, m]];
                }
                inv_vv[[r, col]] += acc;
            }
        }

        Ok((inv_vv, inv_vbeta))
    }

    pub(crate) fn latent_inverse_diagonal(&self) -> Result<Array1<f64>, String> {
        if self.woodbury_factor.is_none() {
            return self
                .cache
                .latent_block_inverse_diagonal()
                .map_err(|err| format!("DeflatedArrowSolver: latent inverse diagonal: {err}"));
        }
        let total_t = self.cache.delta_t_len();
        let mut out = Array1::<f64>::zeros(total_t);
        let rhs_beta = Array1::<f64>::zeros(self.cache.k);
        // Reuse one unit-vector buffer: set/clear a single entry per index rather
        // than allocating and zeroing a total_t-sized RHS on every iteration.
        let mut rhs_t = Array1::<f64>::zeros(total_t);
        for idx in 0..total_t {
            rhs_t[idx] = 1.0;
            let solved = self.solve(rhs_t.view(), rhs_beta.view())?;
            rhs_t[idx] = 0.0;
            out[idx] = solved.t[idx];
        }
        Ok(out)
    }
}

#[cfg(test)]
mod selected_inverse_row_blocks_oracle_tests {
    //! #932 FRONT C oracle: the row-local Takahashi selected-inverse blocks
    //! ([`DeflatedArrowSolver::selected_inverse_row_blocks`] / [`beta_inv`])
    //! MUST reproduce the per-row full-system `solve` loop they replace, to
    //! ≤1e-9, on the plain bordered arrow. This is the gate the logdet /
    //! α-trace consumers rely on when they take the fast path.
    use super::*;
    use gam_solve::arrow_schur::{
        ArrowFactorSlab, ArrowHtbetaCache, ArrowSolverMode, ArrowUndampedFactors, PcgDiagnostics,
    };
    use ndarray::array;
    use std::sync::Arc;

    /// A plain bordered-arrow cache with a NONZERO `H_tβ` coupling and a PD
    /// dense Schur factor, so the β-Schur back-substitution genuinely exercises
    /// the `G S⁻¹ Gᵀ` / `−G S⁻¹` terms (not just the block-diagonal `A⁻¹`). The
    /// stored factors are lower-Cholesky factors `L` (the represented block is
    /// `L Lᵀ`); the row-local identity holds for any PD `A`/`S` and any `B`.
    fn coupled_arrow_cache() -> ArrowFactorCache {
        let htt = ArrowFactorSlab::from_blocks(vec![
            array![[1.3_f64, 0.0], [0.4, 1.1]],
            array![[0.9_f64]],
        ]);
        let schur = array![[1.2_f64, 0.0], [0.25, 0.95]];
        ArrowFactorCache {
            htt_factors: htt,
            htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
            schur_factor: Some(schur),
            joint_hessian_log_det: None,
            solver_mode: ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: ArrowHtbetaCache::Dense {
                blocks: Arc::from(
                    vec![
                        array![[0.5_f64, -0.2], [0.1, 0.4]],
                        array![[0.3_f64, 0.7]],
                    ]
                    .into_boxed_slice(),
                ),
                estimated_bytes: 0,
            },
            d: 2,
            row_dims: Arc::from(vec![2usize, 1usize].into_boxed_slice()),
            row_offsets: Arc::from(vec![0usize, 2usize, 3usize].into_boxed_slice()),
            k: 2,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
            pcg_diagnostics: PcgDiagnostics::default(),
            gauge_deflated_directions: 0,
            deflated_row_directions: Arc::from(Vec::new()),
            deflation_row_spectra: Arc::from(Vec::new()),
            cross_row_woodbury: None,
        }
    }

    #[test]
    fn row_local_blocks_match_per_row_solve() {
        let cache = coupled_arrow_cache();
        let solver = DeflatedArrowSolver::plain(&cache);
        assert!(
            solver.plain_selected_inverse_available(),
            "plain cache must take the fast selected-inverse path"
        );
        let total_t = cache.delta_t_len();
        let k = cache.k;

        // β-block `(H⁻¹)_ββ = S⁻¹`: beta_inv() vs the per-column unit-β solve.
        let beta_inv = solver.beta_inv().expect("beta_inv");
        let rhs_t_zero = Array1::<f64>::zeros(total_t);
        for col in 0..k {
            let mut rhs_beta = Array1::<f64>::zeros(k);
            rhs_beta[col] = 1.0;
            let solved = solver
                .solve(rhs_t_zero.view(), rhs_beta.view())
                .expect("β solve");
            for r in 0..k {
                assert!(
                    (beta_inv[[r, col]] - solved.beta[r]).abs() <= 1e-9,
                    "beta_inv[{r},{col}] {} != solve {}",
                    beta_inv[[r, col]],
                    solved.beta[r]
                );
            }
        }

        // Per-row `(H⁻¹)_tt` (q×q) and `(H⁻¹)_tβ` (q×K) blocks.
        let rhs_beta_zero = Array1::<f64>::zeros(k);
        for row in 0..cache.n_rows() {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let (inv_vv, inv_vbeta) = solver
                .selected_inverse_row_blocks(row, &beta_inv)
                .expect("row blocks");
            for col in 0..q {
                let mut rhs_t = Array1::<f64>::zeros(total_t);
                rhs_t[base + col] = 1.0;
                let solved = solver
                    .solve(rhs_t.view(), rhs_beta_zero.view())
                    .expect("t solve");
                for r in 0..q {
                    assert!(
                        (inv_vv[[r, col]] - solved.t[base + r]).abs() <= 1e-9,
                        "inv_vv[{r},{col}] {} != solve {}",
                        inv_vv[[r, col]],
                        solved.t[base + r]
                    );
                }
                for b in 0..k {
                    assert!(
                        (inv_vbeta[[col, b]] - solved.beta[b]).abs() <= 1e-9,
                        "inv_vbeta[{col},{b}] {} != solve {}",
                        inv_vbeta[[col, b]],
                        solved.beta[b]
                    );
                }
            }
        }
    }
}

pub(crate) fn flatten_arrow_parts(
    t: ArrayView1<'_, f64>,
    beta: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(t.len() + beta.len());
    for i in 0..t.len() {
        out[i] = t[i];
    }
    for i in 0..beta.len() {
        out[t.len() + i] = beta[i];
    }
    out
}

pub(crate) fn apply_cached_arrow_hessian(
    cache: &ArrowFactorCache,
    v_t: ArrayView1<'_, f64>,
    v_beta: ArrayView1<'_, f64>,
) -> Result<SaeArrowVector, String> {
    let total_t = cache.delta_t_len();
    if v_t.len() != total_t || v_beta.len() != cache.k {
        return Err(format!(
            "apply_cached_arrow_hessian: vector shapes (t={}, beta={}) != cache shapes \
             (t={total_t}, beta={})",
            v_t.len(),
            v_beta.len(),
            cache.k
        ));
    }

    let mut out_t = Array1::<f64>::zeros(total_t);
    let mut out_beta = Array1::<f64>::zeros(cache.k);
    for row in 0..cache.n_rows() {
        let di = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let row_v = v_t.slice(s![base..base + di]);
        let factor = cache.undamped_factor(row);
        let av = cholesky_factor_apply(factor, row_v);
        for j in 0..di {
            out_t[base + j] += av[j];
        }
        if cache.k > 0 {
            let mut b_vbeta = Array1::<f64>::zeros(di);
            if !cache.apply_htbeta_row(row, v_beta, &mut b_vbeta) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_tβ^({row}) apply failed"
                ));
            }
            for j in 0..di {
                out_t[base + j] += b_vbeta[j];
            }
            if !cache.apply_htbeta_row_transpose(row, row_v, &mut out_beta, None) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_βt^({row}) apply failed"
                ));
            }
        }
    }

    if cache.k > 0 {
        let Some(schur_factor) = cache.schur_factor.as_ref() else {
            return Err(
                "apply_cached_arrow_hessian: dense Schur factor is required for gauge probing"
                    .to_string(),
            );
        };
        let schur_v = cholesky_factor_apply(schur_factor.view(), v_beta);
        for i in 0..cache.k {
            out_beta[i] += schur_v[i];
        }
        for row in 0..cache.n_rows() {
            let di = cache.row_dims[row];
            let mut b_vbeta = Array1::<f64>::zeros(di);
            if !cache.apply_htbeta_row(row, v_beta, &mut b_vbeta) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_tβ^({row}) Schur correction apply failed"
                ));
            }
            let a_inv_b_vbeta = cholesky_solve_vector(cache.undamped_factor(row), b_vbeta.view());
            if !cache.apply_htbeta_row_transpose(row, a_inv_b_vbeta.view(), &mut out_beta, None) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_βt^({row}) Schur correction apply failed"
                ));
            }
        }
    }

    // #1038 IBP cross-row curvature: when the cache carries the exact rank-`R`
    // Woodbury, the operator it represents is `H_full = H₀' + U D Uᵀ` (the same
    // operator `full_inverse_apply` inverts and `arrow_log_det` reports). The
    // per-row factors reconstructed above are only the NO-SELF base `H₀'`, so the
    // forward apply MUST add `U D Uᵀ v` here — otherwise the forward operator
    // (used by the #1418 exact-stationarity solve) silently drops the cross-row
    // block while its CG preconditioner inverts the full `H_full`, desyncing the
    // outer-REML gradient. `U` has no `β` support ⇒ only the `t` block changes.
    if let Some(woodbury) = cache.cross_row_woodbury.as_ref() {
        woodbury.apply_forward_t(v_t, &mut out_t);
    }

    Ok(SaeArrowVector {
        t: out_t,
        beta: out_beta,
    })
}

pub(crate) fn cholesky_factor_apply(
    factor: ArrayView2<'_, f64>,
    vector: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let n = factor.nrows();
    // `factor` is a lower-triangular Cholesky factor `L` stored row-major; the
    // represented action is `out = L (Lᵀ v)`.
    //
    // Phase 1 — `lt_v = Lᵀ v`. The natural inner-product form reads
    // `factor[[col, row]]` down a COLUMN (stride `n`), which thrashes cache for
    // the K×K Schur factor (K up to 32 000). Instead iterate over ROWS `j` of
    // `L` — contiguous in memory — and scatter `L[j, 0..=j]·v[j]` into `lt_v`,
    // touching each `L` row exactly once in row-major order.
    let mut lt_v = Array1::<f64>::zeros(n);
    for j in 0..n {
        let vj = vector[j];
        if vj == 0.0 {
            continue;
        }
        // Row `j` of `L` is traversed in its own memory order (contiguous for the
        // row-major factors we build); the compiler can vectorize the axpy.
        for (i, &lji) in factor.row(j).iter().enumerate().take(j + 1) {
            lt_v[i] += lji * vj;
        }
    }
    // Phase 2 — `out = L lt_v`. Already contiguous row-major (`L[row, 0..=row]`).
    let mut out = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut acc = 0.0_f64;
        for (col, &lrc) in factor.row(row).iter().enumerate().take(row + 1) {
            acc += lrc * lt_v[col];
        }
        out[row] = acc;
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum SaeLocalRowVar {
    Logit { atom: usize },
    Coord { atom: usize, axis: usize },
}

#[derive(Debug, Clone)]
pub(crate) struct SaeBorderChannel {
    pub(crate) atom: usize,
    pub(crate) basis_col: usize,
    pub(crate) index: usize,
    pub(crate) output: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct SaeRowJets {
    pub(crate) vars: Vec<SaeLocalRowVar>,
    pub(crate) first: Vec<Vec<f64>>,
    pub(crate) second: Vec<Vec<Vec<f64>>>,
    pub(crate) beta: Vec<Vec<f64>>,
    pub(crate) beta_deriv: Vec<Vec<Vec<f64>>>,
    pub(crate) beta_l_deriv: Vec<Vec<Vec<f64>>>,
}

pub(crate) fn sae_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean inner product `⟨a, b⟩` over the concatenated `(t, β)` blocks of two
/// arrow vectors. Used by the #1418 `B`-preconditioned CG inner solve.
pub(crate) fn sae_inner(a: &SaeArrowVector, b: &SaeArrowVector) -> f64 {
    sae_dot(a.t.as_slice().unwrap_or(&[]), b.t.as_slice().unwrap_or(&[]))
        + sae_dot(
            a.beta.as_slice().unwrap_or(&[]),
            b.beta.as_slice().unwrap_or(&[]),
        )
}

/// Euclidean norm `‖a‖` over the concatenated `(t, β)` blocks of an arrow vector.
pub(crate) fn sae_norm(a: &SaeArrowVector) -> f64 {
    sae_inner(a, a).max(0.0).sqrt()
}

/// #1418: solve `A x = rhs` by **`B`-preconditioned conjugate gradients**, where
/// `apply_a(v) = A v` is the exact stationarity-Jacobian matvec and the
/// `solver` (the assembled `B` factorization) supplies the SPD preconditioner
/// `B⁻¹`. The IFT step `θ̂_ρ = −A⁻¹ g_ρ` must invert the EXACT `A`, not the
/// surrogate `B`; the earlier truncated Neumann series `Σ_m (−B⁻¹ΔC)^m B⁻¹ rhs`
/// equals `A⁻¹ rhs` only when `ρ(B⁻¹ΔC) < 1`, and DIVERGED for large
/// `ΔC = ⟨r, ∂²f⟩`. PCG converges for any spectral radius in ≤ `dim` steps — one
/// `A` matvec and one `B⁻¹` solve per step, no second factorization. On
/// non-positive curvature `pᵀ A p ≤ 0` (the high-residual `A` can be indefinite
/// away from a strict minimum) it stops at the last finite iterate.
pub(crate) fn solve_b_preconditioned_cg<F>(
    solver: &DeflatedArrowSolver<'_>,
    rhs: &SaeArrowVector,
    apply_a: F,
) -> Result<SaeArrowVector, String>
where
    F: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    // x_0 = B⁻¹ rhs (the surrogate step; CG corrects it onto A⁻¹ rhs).
    let mut x = solver
        .solve(rhs.t.view(), rhs.beta.view())
        .map_err(|err| format!("solve_b_preconditioned_cg: B inverse: {err}"))?;
    // r_0 = rhs − A x_0; z_0 = B⁻¹ r_0; p_0 = z_0.
    let ax = apply_a(&x)?;
    let mut r = SaeArrowVector {
        t: &rhs.t - &ax.t,
        beta: &rhs.beta - &ax.beta,
    };
    let mut z = solver
        .solve(r.t.view(), r.beta.view())
        .map_err(|err| format!("solve_b_preconditioned_cg: B preconditioner: {err}"))?;
    // p_0 = z_0. Compute rz from z FIRST, then MOVE z into p (no clone) — z is
    // re-bound at the top of every loop iteration before it is read again.
    let mut rz = sae_inner(&r, &z);
    let mut p = z;

    let rhs_norm = sae_norm(rhs).max(1.0);
    let max_iters = (x.t.len() + x.beta.len()).clamp(8, 256);
    let rel_tol = 1.0e-10;
    for _ in 0..max_iters {
        if !rz.is_finite() || rz <= 0.0 {
            break; // preconditioned residual exhausted / degenerate.
        }
        let ap = apply_a(&p)?;
        let p_ap = sae_inner(&p, &ap);
        if !p_ap.is_finite() || p_ap <= 0.0 {
            break; // non-positive curvature: keep the finite iterate.
        }
        let alpha = rz / p_ap;
        for idx in 0..x.t.len() {
            x.t[idx] += alpha * p.t[idx];
            r.t[idx] -= alpha * ap.t[idx];
        }
        for idx in 0..x.beta.len() {
            x.beta[idx] += alpha * p.beta[idx];
            r.beta[idx] -= alpha * ap.beta[idx];
        }
        if sae_norm(&r) <= rel_tol * rhs_norm {
            break;
        }
        z = solver
            .solve(r.t.view(), r.beta.view())
            .map_err(|err| format!("solve_b_preconditioned_cg: B preconditioner: {err}"))?;
        let rz_next = sae_inner(&r, &z);
        let beta = rz_next / rz;
        for idx in 0..p.t.len() {
            p.t[idx] = z.t[idx] + beta * p.t[idx];
        }
        for idx in 0..p.beta.len() {
            p.beta[idx] = z.beta[idx] + beta * p.beta[idx];
        }
        rz = rz_next;
    }
    Ok(x)
}
