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
    pub(crate) gauge_stiffness: f64,
}

impl<'a> DeflatedArrowSolver<'a> {
    pub(crate) fn plain(cache: &'a ArrowFactorCache) -> Self {
        Self {
            cache,
            gauge_basis: Vec::new(),
            gauge_response_physical: Vec::new(),
            woodbury_factor: None,
            gauge_stiffness: 0.0,
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
            gauge_stiffness: stiffness,
        })
    }

    /// Add the closed-form gauge-fixing action `κ Q Qᵀ v` to an already
    /// assembled operator product.
    ///
    /// [`Self::solve`] is the Woodbury inverse of `B + κ Q Qᵀ`, not of the raw
    /// arrow operator `B`. Any operator preconditioned by this solver must carry
    /// the same gauge stiffness. In particular, the exact-stationarity solve
    /// uses `A + κ Q Qᵀ`: applying raw `A` while preconditioning with the
    /// gauge-fixed `B` leaves the known gauge null in the Krylov operator and can
    /// make a perfectly valid quotient solve fail its original-residual check
    /// (#2253). Plain solvers have an empty basis and remain bit-identical.
    pub(crate) fn add_gauge_stiffness(
        &self,
        vector: &SaeArrowVector,
        applied: &mut SaeArrowVector,
    ) -> Result<(), String> {
        if self.gauge_basis.is_empty() {
            return Ok(());
        }
        let t_len = self.cache.delta_t_len();
        let beta_len = self.cache.k;
        if vector.t.len() != t_len
            || vector.beta.len() != beta_len
            || applied.t.len() != t_len
            || applied.beta.len() != beta_len
        {
            return Err(format!(
                "DeflatedArrowSolver: gauge-stiffness operator shapes vector=({}, {}), \
                 applied=({}, {}) != cache=({t_len}, {beta_len})",
                vector.t.len(),
                vector.beta.len(),
                applied.t.len(),
                applied.beta.len(),
            ));
        }
        for gauge in &self.gauge_basis {
            let coefficient = self.gauge_stiffness
                * (gauge.slice(s![..t_len]).dot(&vector.t)
                    + gauge.slice(s![t_len..]).dot(&vector.beta));
            for i in 0..t_len {
                applied.t[i] += coefficient * gauge[i];
            }
            for i in 0..beta_len {
                applied.beta[i] += coefficient * gauge[t_len + i];
            }
        }
        Ok(())
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
            let coeff = self.gauge_stiffness.recip() * weight;
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
    /// undamped criterion factor stiffened to unit curvature; a `½ tr(H⁻¹ ∂H/∂ρ)`
    /// diagonal contraction against it therefore spuriously includes
    /// `Σ_i vᵢ[s]²` at slot `s`, a ρ/θ-independent contribution that must be 0.
    /// This variant subtracts the per-row deflated outer-product diagonal
    /// `Σ_i vᵢ[s]²` so the diagonal traces (ARD precision, ordered Beta--Bernoulli/softmax assignment
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
    /// omit. Callers must then fall back to the per-row `solve` loop.
    pub(crate) fn plain_selected_inverse_available(&self) -> bool {
        self.woodbury_factor.is_none()
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

        // GS = G_i S⁻¹ (q×K), via the cache-blocked ndarray/matrixmultiply gemm
        // instead of an O(q·K²) scalar triple loop (K up to 32k).
        let gs = g.dot(beta_inv);

        // (H⁻¹)_tβ[i] = −G_i S⁻¹ = −GS, layout [col, b].
        let inv_vbeta = -&gs;

        // (H⁻¹)_tt[i,i] = A_i⁻¹ + G_i S⁻¹ G_iᵀ = A_i⁻¹ + GS·Gᵀ, layout [r, col].
        // `GS·Gᵀ` is another gemm (q×K · K×q); accumulate onto A_i⁻¹ in place.
        let mut inv_vv = a_inv;
        inv_vv += &gs.dot(&g.t());

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
        ArrowFactorSlab, ArrowHtbetaCache, ArrowPcgDiagnostics, ArrowSolverMode,
        ArrowUndampedFactors,
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
            schur_factor_is_undamped: true,
            joint_hessian_log_det: None,
            solver_mode: ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: ArrowHtbetaCache::Dense {
                blocks: Arc::from(
                    vec![array![[0.5_f64, -0.2], [0.1, 0.4]], array![[0.3_f64, 0.7]]]
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
            pcg_diagnostics: ArrowPcgDiagnostics::default(),
            gauge_deflated_directions: 0,
            deflated_row_directions: Arc::from(Vec::new()),
            deflation_row_spectra: Arc::from(Vec::new()),
            beta_gauge_quotient: None,
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
        if !cache.schur_factor_is_undamped {
            return Err(
                "apply_cached_arrow_hessian: Schur factor was not built from the undamped evidence row factors"
                    .to_string(),
            );
        }
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
    // touching each `L` row once in row-major order (summation order preserved
    // ⇒ bit-identical).
    let mut lt_v = Array1::<f64>::zeros(n);
    for j in 0..n {
        let vj = vector[j];
        if vj == 0.0 {
            continue;
        }
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
    pub(crate) channels: crate::row_jet_program::SaeScheduledRowJets,
}

impl SaeRowJets {
    #[inline]
    pub(crate) fn first(&self, primary: usize) -> &[f64] {
        self.channels.first(primary)
    }

    #[inline]
    pub(crate) fn second(&self, a: usize, b: usize) -> &[f64] {
        self.channels.second(a, b)
    }

    #[inline]
    pub(crate) fn beta(&self, border: usize) -> &[f64] {
        self.channels.beta(border)
    }

    #[inline]
    pub(crate) fn beta_deriv(&self, primary: usize, border: usize) -> &[f64] {
        self.channels.beta_deriv(primary, border)
    }

    #[inline]
    pub(crate) fn beta_l_deriv(&self, primary: usize, border: usize) -> &[f64] {
        self.channels.beta_l_deriv(primary, border)
    }

}

pub(crate) fn sae_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean inner product `⟨a, b⟩` over the concatenated `(t, β)` blocks of two
/// arrow vectors. Used by the exact-stationarity Krylov solve and its residual
/// verification.
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

/// Largest Arnoldi basis admitted by the live cgroup-aware host budget.
///
/// A flexible-GMRES cycle owns `(m+1)` length-`dim` residual-basis vectors,
/// `m` length-`dim` preconditioned directions, an `(m+1)×m` Hessenberg matrix,
/// and the fixed work vectors. Choosing `m` from the memory ledger makes
/// small/medium arrow systems full-memory (and hence exact in at most `dim`
/// Krylov directions) without imposing a dimension or iteration constant;
/// genuinely large systems use the largest restart that can be allocated
/// without violating the process budget.
fn admitted_gmres_restart(dim: usize) -> Result<usize, String> {
    let (budget, available) = sae_host_in_core_budget_bytes();
    let storage_bytes = |m: usize| -> Option<usize> {
        let basis = m.checked_add(1)?.checked_mul(dim)?;
        let preconditioned_directions = m.checked_mul(dim)?;
        let hessenberg = m.checked_add(1)?.checked_mul(m)?;
        let fixed = dim.checked_mul(6)?.checked_add(m.checked_mul(4)?)?;
        basis
            .checked_add(preconditioned_directions)?
            .checked_add(hessenberg)?
            .checked_add(fixed)?
            .checked_mul(std::mem::size_of::<f64>())
    };
    let minimum = storage_bytes(1).ok_or_else(|| {
        format!("solve_b_preconditioned_gmres: storage size overflow for dimension {dim}")
    })?;
    if minimum > budget {
        return Err(format!(
            "solve_b_preconditioned_gmres: even one Arnoldi direction needs {minimum} bytes, \
             exceeding the cgroup-aware Krylov budget {budget} (available {available})"
        ));
    }
    let mut low = 1usize;
    let mut high = dim;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if storage_bytes(mid).is_some_and(|bytes| bytes <= budget) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(low)
}

/// Solve `A x = rhs` by flexible right-preconditioned restarted GMRES.
///
/// The exact stationarity Jacobian `A` contains residual and prior curvature and
/// can be indefinite. Conjugate gradients is therefore not admissible: its SPD
/// recurrence can stop at negative curvature and silently return a non-solution.
/// Flexible GMRES instead minimizes the original residual without an SPD or
/// linear-preconditioner assumption. At Arnoldi direction `v_j` it stores
/// `z_j = P_j(v_j)`, applies `A z_j`, and updates the physical solution as
/// `x += Σ_j z_j y_j`. This is essential for the matrix-free arrow inverse:
/// its tolerance-driven adaptive CG is not one fixed linear `B⁻¹`, so recovering
/// `P(Σ_j v_j y_j)` after the least-squares solve is not equal to the Arnoldi
/// combination `Σ_j P_j(v_j)y_j` (#2258). Because preconditioning remains on the
/// right, the Arnoldi least-squares norm is the physical residual that certifies
/// `A x = rhs`; an ill-scaled or adaptive inverse cannot certify only a
/// preconditioned proxy. Exhaustion or Arnoldi breakdown is a typed error, never
/// a last-iterate fallback. The same implementation serves the fixed dense
/// factor cache and the adaptive matrix-free reduced-Schur inverse.
pub(crate) fn solve_b_preconditioned_gmres_with<F, P>(
    rhs: &SaeArrowVector,
    apply_a: F,
    precondition: P,
) -> Result<SaeArrowVector, String>
where
    F: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    P: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    let t_len = rhs.t.len();
    let beta_len = rhs.beta.len();
    let dim = t_len + beta_len;
    if dim == 0 {
        return Ok(SaeArrowVector {
            t: Array1::zeros(0),
            beta: Array1::zeros(0),
        });
    }
    let rhs_flat = flatten_arrow_parts(rhs.t.view(), rhs.beta.view());
    let rhs_norm = rhs_flat.dot(&rhs_flat).sqrt();
    if rhs_norm == 0.0 {
        return Ok(SaeArrowVector {
            t: Array1::zeros(t_len),
            beta: Array1::zeros(beta_len),
        });
    }
    if !rhs_norm.is_finite() {
        return Err("solve_b_preconditioned_gmres: non-finite right-hand side".to_string());
    }
    let b = rhs_flat;
    let b_norm = rhs_norm;
    let relative_floor = f64::EPSILON.sqrt();
    // Full-memory whenever the live memory ledger admits it. There is no
    // iteration ceiling: each restarted cycle must make a strictly
    // representable reduction in the original residual, and inability to do so
    // is the typed numerical-stagnation certificate.
    let restart = admitted_gmres_restart(dim)?;
    let mut iterations = 0usize;
    let mut solution = Array1::<f64>::zeros(dim);

    let as_arrow = |flat: &Array1<f64>| SaeArrowVector {
        t: flat.slice(s![..t_len]).to_owned(),
        beta: flat.slice(s![t_len..]).to_owned(),
    };
    let apply_preconditioner = |flat: &Array1<f64>| -> Result<Array1<f64>, String> {
        let direction = as_arrow(flat);
        let preconditioned = precondition(&direction)
            .map_err(|err| format!("solve_b_preconditioned_gmres: B inverse: {err}"))?;
        Ok(flatten_arrow_parts(
            preconditioned.t.view(),
            preconditioned.beta.view(),
        ))
    };
    let apply_operator = |flat: &Array1<f64>| -> Result<Array1<f64>, String> {
        let physical = as_arrow(flat);
        let applied = apply_a(&physical)?;
        Ok(flatten_arrow_parts(applied.t.view(), applied.beta.view()))
    };

    loop {
        let ax = apply_operator(&solution)?;
        let mut residual = &b - &ax;
        let residual_norm = residual.dot(&residual).sqrt();
        if residual_norm <= relative_floor * b_norm {
            let candidate = as_arrow(&solution);
            let ax = apply_a(&candidate)?;
            let original = SaeArrowVector {
                t: &rhs.t - &ax.t,
                beta: &rhs.beta - &ax.beta,
            };
            let original_norm = sae_norm(&original);
            if original_norm <= relative_floor * rhs_norm {
                return Ok(candidate);
            }
        }
        if !(residual_norm.is_finite() && residual_norm > 0.0) {
            return Err("solve_b_preconditioned_gmres: non-finite original residual".to_string());
        }

        residual.mapv_inplace(|value| value / residual_norm);
        let cycle = restart;
        let mut basis: Vec<Array1<f64>> = Vec::with_capacity(cycle + 1);
        basis.push(residual);
        let mut preconditioned_basis: Vec<Array1<f64>> = Vec::with_capacity(cycle);
        let mut h = Array2::<f64>::zeros((cycle + 1, cycle));
        let mut cosines = vec![0.0_f64; cycle];
        let mut sines = vec![0.0_f64; cycle];
        let mut g = Array1::<f64>::zeros(cycle + 1);
        g[0] = residual_norm;
        let mut used = 0usize;

        for j in 0..cycle {
            let preconditioned_direction = apply_preconditioner(&basis[j])?;
            if !preconditioned_direction
                .iter()
                .all(|value| value.is_finite())
            {
                return Err(format!(
                    "solve_b_preconditioned_gmres: non-finite preconditioned direction at \
                     iteration {}",
                    iterations + j
                ));
            }
            let mut w = apply_operator(&preconditioned_direction)?;
            preconditioned_basis.push(preconditioned_direction);
            // Modified Gram-Schmidt Arnoldi.
            for i in 0..=j {
                let hij = basis[i].dot(&w);
                h[[i, j]] = hij;
                for slot in 0..dim {
                    w[slot] -= hij * basis[i][slot];
                }
            }
            let next_norm = w.dot(&w).sqrt();
            h[[j + 1, j]] = next_norm;
            if next_norm > f64::EPSILON {
                w.mapv_inplace(|value| value / next_norm);
                basis.push(w);
            } else {
                basis.push(Array1::zeros(dim));
            }

            for i in 0..j {
                let upper = cosines[i] * h[[i, j]] + sines[i] * h[[i + 1, j]];
                let lower = -sines[i] * h[[i, j]] + cosines[i] * h[[i + 1, j]];
                h[[i, j]] = upper;
                h[[i + 1, j]] = lower;
            }
            let diagonal = h[[j, j]];
            let below = h[[j + 1, j]];
            let radius = diagonal.hypot(below);
            if !(radius.is_finite() && radius > f64::EPSILON) {
                return Err(format!(
                    "solve_b_preconditioned_gmres: Arnoldi breakdown after {} iterations",
                    iterations + j
                ));
            }
            cosines[j] = diagonal / radius;
            sines[j] = below / radius;
            h[[j, j]] = radius;
            h[[j + 1, j]] = 0.0;
            let gj = g[j];
            g[j] = cosines[j] * gj;
            g[j + 1] = -sines[j] * gj;
            used = j + 1;
            iterations = iterations.checked_add(1).ok_or_else(|| {
                "solve_b_preconditioned_gmres: iteration counter overflow".to_string()
            })?;
            if g[j + 1].abs() <= relative_floor * b_norm {
                break;
            }
        }

        let mut y = Array1::<f64>::zeros(used);
        for i in (0..used).rev() {
            let mut value = g[i];
            for j in i + 1..used {
                value -= h[[i, j]] * y[j];
            }
            let diagonal = h[[i, i]];
            if !(diagonal.is_finite() && diagonal.abs() > f64::EPSILON) {
                return Err(format!(
                    "solve_b_preconditioned_gmres: singular Hessenberg diagonal at {i}"
                ));
            }
            y[i] = value / diagonal;
        }
        for i in 0..used {
            for slot in 0..dim {
                solution[slot] += y[i] * preconditioned_basis[i][slot];
            }
        }

        let candidate = as_arrow(&solution);
        let ax = apply_a(&candidate)?;
        let original = SaeArrowVector {
            t: &rhs.t - &ax.t,
            beta: &rhs.beta - &ax.beta,
        };
        let original_norm = sae_norm(&original);
        // Full-memory GMRES terminates within `dim` directions up to round-off.
        // The attainable residual of a moderately conditioned system is
        // O(kappa*eps), so sqrt(eps) is the scalar-type-derived certification
        // floor rather than a last-iterate fallback.
        let roundoff_floor = relative_floor * rhs_norm;
        if original_norm <= roundoff_floor {
            return Ok(candidate);
        }
        let next_ax = apply_operator(&solution)?;
        let next_residual = &b - &next_ax;
        let next_norm = next_residual.dot(&next_residual).sqrt();
        if !(next_norm.is_finite() && next_norm < residual_norm) {
            return Err(format!(
                "solve_b_preconditioned_gmres: no representable residual reduction after \
                 {iterations} iterations (restart {restart}, dimension {dim}); original \
                 residual {residual_norm:.3e} -> {next_norm:.3e}, relative residual \
                 {:.3e}, round-off certification floor {:.3e}",
                original_norm / rhs_norm,
                roundoff_floor / rhs_norm,
            ));
        }
    }
}

#[cfg(test)]
mod right_preconditioned_gmres_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn certifies_original_residual_under_ill_scaled_preconditioner_2258() {
        // A x = rhs has x=(1,1). B^-1 scales the two physical coordinates in
        // opposite directions, so a left-preconditioned residual is a poor
        // proxy for the physical equation. Right GMRES must still solve and
        // certify rhs-Ax in the original norm.
        let rhs = SaeArrowVector {
            t: array![5.0_f64, 5.0],
            beta: Array1::zeros(0),
        };
        let apply_a = |value: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            Ok(SaeArrowVector {
                t: array![
                    3.0 * value.t[0] + 2.0 * value.t[1],
                    value.t[0] + 4.0 * value.t[1],
                ],
                beta: Array1::zeros(0),
            })
        };
        let precondition = |value: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            Ok(SaeArrowVector {
                t: array![1.0e-3 * value.t[0], 1.0e3 * value.t[1]],
                beta: Array1::zeros(0),
            })
        };

        let solved = solve_b_preconditioned_gmres_with(&rhs, apply_a, precondition)
            .expect("right-preconditioned solve");
        let applied = apply_a(&solved).expect("physical operator");
        let residual = SaeArrowVector {
            t: &rhs.t - &applied.t,
            beta: Array1::zeros(0),
        };
        assert!(
            sae_norm(&residual) <= f64::EPSILON.sqrt() * sae_norm(&rhs),
            "physical residual was not certified: relative={:.3e}",
            sae_norm(&residual) / sae_norm(&rhs),
        );
        assert!((solved.t[0] - 1.0).abs() <= f64::EPSILON.sqrt());
        assert!((solved.t[1] - 1.0).abs() <= f64::EPSILON.sqrt());
    }

    #[test]
    fn supports_non_linear_adaptive_preconditioner_2258() {
        // The adaptive inverse normalizes each requested direction. It is
        // deliberately nonlinear: for rhs=5v, P(rhs)=v while 5P(v)=5v. A
        // standard right-GMRES implementation that recovers P(Vy) therefore
        // returns v after its one-direction Arnoldi solve and cannot reduce the
        // physical residual. Flexible GMRES stores z=P(v) and updates x=zy=5v,
        // which exactly solves the identity equation.
        let rhs = SaeArrowVector {
            t: array![3.0_f64, 4.0],
            beta: Array1::zeros(0),
        };
        let apply_a =
            |value: &SaeArrowVector| -> Result<SaeArrowVector, String> { Ok(value.clone()) };
        let precondition = |value: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let norm = sae_norm(value);
            if norm == 0.0 {
                return Ok(value.clone());
            }
            Ok(SaeArrowVector {
                t: &value.t / norm,
                beta: &value.beta / norm,
            })
        };

        let solved = solve_b_preconditioned_gmres_with(&rhs, apply_a, precondition)
            .expect("flexible right-preconditioned solve");
        let residual = SaeArrowVector {
            t: &rhs.t - &solved.t,
            beta: &rhs.beta - &solved.beta,
        };
        assert!(
            sae_norm(&residual) <= f64::EPSILON.sqrt() * sae_norm(&rhs),
            "adaptive inverse must certify the physical equation: relative={:.3e}",
            sae_norm(&residual) / sae_norm(&rhs),
        );
    }
}
