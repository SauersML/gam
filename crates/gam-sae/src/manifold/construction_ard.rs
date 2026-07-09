//! ARD (automatic relevance determination) coordinate-precision + latent-block
//! helpers for `SaeManifoldTerm`, moved verbatim out of construction.rs to keep it
//! under the 10k-line ban gate. Pure code move, no logic change.
use super::*;

impl SaeManifoldTerm {
    /// Per-atom, per-axis coordinate sum-of-squares `‖t_kj‖² = Σ_i t_{i,k,j}²`.
    ///
    /// This is the data-fit sufficient statistic for the ARD precision update
    /// (the numerator-side `‖t‖²` of the deleted `α = n/‖t‖²` rule). Returned
    /// per atom as an `Array1` of length `d_k`.
    ///
    /// On a *periodic* (Circle) axis the relevant statistic is the von-Mises
    /// energy-equivalent `Σ_i 2/α·V(t_i) = Σ_i (2/κ²)(1−cos κ t_i)` (independent
    /// of α), so that `½·α·sumsq == Σ_i V(t_i)` matches `ard_value`. This keeps
    /// the Mackay/Fellner–Schall fixed point `α ← n / (sumsq + tr H⁻¹)`
    /// consistent with the actual periodic prior energy rather than the
    /// origin-dependent raw `t²`.
    pub(crate) fn ard_coord_sumsq(&self) -> Vec<Array1<f64>> {
        // Horvitz–Thompson row weighting: the `‖t‖²` sufficient statistic is the
        // numerator of the same `α ← n_eff / (Σ sq_equiv + tr H⁻¹)` fixed point the
        // (now weight-aware) `ard_value` energy defines, so it MUST carry the SAME
        // per-row inclusion weight `wᵢ` — else the subsampled MacKay/Fellner–Schall
        // step ranks a different precision than the criterion's energy. `None` ⇒
        // `w_row = 1`, bit-for-bit the historical sum.
        let row_w = self.row_loss_weights.as_deref();
        let mut out = Vec::with_capacity(self.k_atoms());
        for coord in &self.assignment.coords {
            let d = coord.latent_dim();
            let periods = coord.effective_axis_periods();
            let mut sq = Array1::<f64>::zeros(d);
            for row in 0..coord.n_obs() {
                let w_row = row_w.map_or(1.0, |w| w[row]);
                let t = coord.row(row);
                for axis in 0..d {
                    // `sq_equiv` is independent of `alpha`; pass 1.0.
                    sq[axis] += w_row * ArdAxisPrior::eval(1.0, t[axis], periods[axis]).sq_equiv;
                }
            }
            out.push(sq);
        }
        out
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹) =
    /// Σ_i [(H⁻¹)_tt]_{(i,k,j),(i,k,j)}` from the converged factor cache.
    ///
    /// `cache.latent_block_inverse_diagonal()` returns the diagonal of the
    /// latent block `(H⁻¹)_tt` in the cache's compact per-row `delta_t`
    /// layout (length `row_offsets[N]`); each per-row block is laid out as
    /// `[logit scalars…, then per-active-atom coord axes…]`. This routine
    /// sums those diagonal entries over the coord positions belonging to each
    /// `(atom k, axis j)` across all observation rows where atom `k` is active.
    ///
    /// `self.last_row_layout` must be the layout from the *same* assemble that
    /// produced `cache`:
    /// - `Some(layout)`: compact active-set mode (JumpReLU / large-K
    ///   softmax-IBP truncation). For row `i`, atom `k`'s position in the
    ///   active list gives its compact coord-block start `coord_starts[i][pos]`;
    ///   inactive atoms contribute 0 (the prior dominates there anyway).
    /// - `None`: dense full-support layout, uniform row dim
    ///   `q = assignment_dim + Σ d_k`; atom `k`'s coord block sits at the
    ///   fixed full-row offset `coord_offsets[k]` after the assignment chart.
    ///
    /// This `tr_kj(H⁻¹)` is exactly the posterior-variance term the deleted
    /// `α = n/‖t‖²` rule dropped; the corrected Mackay/Fellner-Schall fixed
    /// point is `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`.
    ///
    /// At `K ≥ ARD_TRACE_HUTCHINSON_MIN_ATOMS` the exact selected-inverse diagonal
    /// (one dense `K×K` Schur solve per latent coordinate — `O(total_t·K²) ≈
    /// O(K³)` at massive `K`) is replaced by the matrix-free Hutchinson estimate
    /// [`Self::latent_block_inverse_diagonal_hutchinson`]; below it the exact
    /// diagonal is used unchanged (bit-for-bit tests preserved).
    pub(crate) fn ard_inverse_traces(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let inv_diag = if self.k_atoms() >= Self::ARD_TRACE_HUTCHINSON_MIN_ATOMS {
            // Massive-K: `total_t` dense Schur solves is infeasible — estimate the
            // whole latent inverse diagonal matrix-free with one full-arrow solve
            // per Hutchinson probe (the grouped sums below tolerate the stochastic
            // error, as this feeds a Fellner–Schall / dispersion denominator).
            Self::latent_block_inverse_diagonal_hutchinson(
                cache,
                Self::ARD_TRACE_HUTCHINSON_PROBES,
                Self::ARD_TRACE_HUTCHINSON_SEED,
            )?
        } else {
            cache.latent_block_inverse_diagonal()?
        };
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        // Horvitz–Thompson row weight, IDENTICAL to `ard_coord_sumsq`'s numerator
        // weighting: the posterior-variance trace `tr H⁻¹` is the OTHER half of
        // the `α ← n_eff / (Σ wᵢ t̂ᵢ² + Σ wᵢ (H⁻¹)ᵢᵢ)` MacKay/Fellner–Schall
        // fixed point, so a retained row standing in for `wᵢ` rows must
        // contribute its posterior variance `wᵢ` times too — else the α-step's
        // denominator uses a different inclusion measure than its numerator and
        // `n_eff`. Commit 4862e8355 weighted value/sumsq/gradient/curvature
        // "together" but missed this trace channel. `None` ⇒ `wᵢ = 1`,
        // bit-for-bit the historical unweighted sum.
        let row_w = self.row_loss_weights.as_deref();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            let w_row = row_w.map_or(1.0, |w| w[row]);
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            traces[k][axis] += w_row * inv_diag[row_base + block_start + axis];
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            traces[k][axis] += w_row * inv_diag[row_base + block_start + axis];
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹)` — the SAME
    /// quantity [`Self::ard_inverse_traces`] returns — from the #2080 SHARED
    /// selected-inverse bundle instead of the dense Schur factor
    /// (`full_inverse_apply` / `latent_block_inverse_diagonal`). The massive-`K`
    /// surrogate-lane replacement that removes the last dense `S⁻¹` from the ARD
    /// Fellner–Schall denominator.
    ///
    /// # Reformulation (arrow selected-inverse, no dense `S⁻¹`)
    ///
    /// For an arrow `H = [[A (⊕_i A_i), B], [Bᵀ, C]]` with reduced Schur
    /// complement `S`, the per-row latent block is exactly
    /// `(H⁻¹)_{t_i t_i} = A_i⁻¹ + G_i S⁻¹ G_iᵀ`, `A_i = H_tt^(i)` (the per-row
    /// `undamped_factor`), `B_i = H_tβ^(i)` (via `apply_htbeta_row`),
    /// `G_i = A_i⁻¹ B_i`. So the per-slot diagonal the ARD denominator sums splits
    /// into a ROW-LOCAL exact term `(A_i⁻¹)[s,s]` and a border term
    /// `(G_i S⁻¹ G_iᵀ)[s,s] = g_sᵀ S⁻¹ g_s`, `g_s = G_iᵀ e_s`. Summed over rows
    /// for a FIXED `(atom k, axis a)` — the group the ARD α-fixed-point actually
    /// needs — the border piece is a trace `Σ_i g_{s_i}ᵀ S⁻¹ g_{s_i} =
    /// tr(S⁻¹ M_{ka})`, `M_{ka} = Σ_i g_{s_i} g_{s_i}ᵀ`, estimated off the shared
    /// bundle `(z_j, S⁻¹ z_j)` by
    ///   `tr(S⁻¹ M_{ka}) = (1/m) Σ_j Σ_i (g_{s_i}ᵀ S⁻¹ z_j)(g_{s_i}ᵀ z_j)
    ///                   = (1/m) Σ_j Σ_i s_ij[s_i]·w_ij[s_i]`,
    /// with `w_ij = G_i z_j = A_i⁻¹ B_i z_j` and `s_ij = G_i S⁻¹ z_j =
    /// A_i⁻¹ B_i (S⁻¹ z_j)` — both per-row `t`-space vectors from
    /// `apply_htbeta_row` + a per-row Cholesky solve. The final `H_βt` never
    /// appears: it is absorbed by contracting against `S⁻¹ z_j`. Everything is
    /// sourced from the cache (`undamped_factor` + `apply_htbeta_row`) plus the
    /// bundle — no `ArrowSchurSystem`, no dense `S⁻¹`.
    ///
    /// `probes` and `sinv_probes` are the surrogate lane's frozen `(z_j, S⁻¹ z_j)`
    /// pairs (each length `cache.k`, the reduced-Schur border dim); with
    /// full-basis probes `√k·e_j` the Hutchinson average is exact, which the FD
    /// gate exploits to assert equality with [`Self::ard_inverse_traces`]. HT
    /// row-weighting matches `ard_inverse_traces` bit-for-bit (both diagonal parts
    /// carry `w_row`; `None` ⇒ `w_row = 1`).
    pub(crate) fn ard_inverse_traces_from_probes(
        &self,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
    ) -> Result<Vec<Array1<f64>>, String> {
        let m = probes.len();
        if m == 0 || sinv_probes.len() != m {
            return Err(format!(
                "ard_inverse_traces_from_probes: need matching non-empty probe/solve \
                 bundles, got {m} probes and {} solves",
                sinv_probes.len()
            ));
        }
        let k_border = cache.k;
        for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
            for (j, v) in set.iter().enumerate() {
                if v.len() != k_border {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: {label} {j} has length {} != border \
                         dim {k_border}",
                        v.len()
                    ));
                }
            }
        }
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let row_w = self.row_loss_weights.as_deref();
        let inv_m = 1.0 / (m as f64);
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let q = cache.row_dims[row];
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let factor = cache.undamped_factor(row);
            // A_i⁻¹ diagonal (row-local, exact): solve A_i e_s = e_s per local slot.
            let mut a_inv_diag = Array1::<f64>::zeros(q);
            let mut e_s = Array1::<f64>::zeros(q);
            for s in 0..q {
                e_s.fill(0.0);
                e_s[s] = 1.0;
                let col = cholesky_solve_vector(factor, e_s.view());
                a_inv_diag[s] = col[s];
            }
            // Per-probe border vectors w_ij = A_i⁻¹ B_i z_j and s_ij = A_i⁻¹ B_i
            // (S⁻¹ z_j), both row-local `t`-space (length q).
            let mut w_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut s_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut b_tmp = Array1::<f64>::zeros(q);
            for j in 0..m {
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, probes[j].view(), &mut b_tmp) {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: H_tβ^({row}) probe apply failed"
                    ));
                }
                w_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, sinv_probes[j].view(), &mut b_tmp) {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: H_tβ^({row}) solve apply failed"
                    ));
                }
                s_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
            }
            // Per-slot diagonal = (A_i⁻¹)[s,s] + (1/m) Σ_j w_ij[s]·s_ij[s]; sum into
            // the owning (atom, axis) trace exactly as `ard_inverse_traces` does.
            let accumulate = |k: usize, block_start: usize, traces: &mut Vec<Array1<f64>>| {
                let d = self.assignment.coords[k].latent_dim();
                for axis in 0..d {
                    let s = block_start + axis;
                    let mut border = 0.0_f64;
                    for j in 0..m {
                        border += w_probes[j][s] * s_probes[j][s];
                    }
                    traces[k][axis] += w_row * (a_inv_diag[s] + inv_m * border);
                }
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        accumulate(k, starts[pos], &mut traces);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        accumulate(k, coord_offsets[k], &mut traces);
                    }
                }
            }
        }
        Ok(traces)
    }

    /// PD-floor fraction for the #2133 SURE divergence denominator: the exact
    /// within-basin Newton curvature `htt + c + V''` is clamped to at least this
    /// fraction of the Gauss-Newton curvature `htt + V''`, so a row whose
    /// residual curvature `c` exceeds the total positive curvature (a local
    /// saddle, where the divergence would otherwise blow up or flip sign) is
    /// bounded instead of poisoning `φ̂`. Matches the acceptance harness floor.
    pub(crate) const SURE_DIVERGENCE_PD_FLOOR: f64 = 0.1;

    /// #2133 — SURE within-basin second-order deflation correction to `coord_edf`.
    ///
    /// The arrow-Schur coordinate curvature `htt = a_k²·(g')ᵀM g'` is
    /// Gauss-Newton (first jets only), so the ARD-shrunk edf `htt/(htt+V'')` the
    /// dispersion sums is the GN divergence. The EXACT divergence of the
    /// penalized basin-selecting MAP `θ̂(y)=argmin ½‖y−f‖²_M + V(θ)` is, by the
    /// implicit-function theorem on the stationarity `g = f'ᵀM(f−y) + V' = 0`
    /// (`∂g/∂θ = htt + f''ᵀM(f−y) + V''`, `∂g/∂y = −f'ᵀM`),
    ///   div = htt / (htt + f''ᵀM·r_code + V''),   r_code = f(θ̂) − y,
    /// which restores the residual-curvature term `c = a_k·(g'')ᵀM·r_code` the GN
    /// block drops. On a curved chart `c ≠ 0`, so GN systematically MIS-counts the
    /// per-row coordinate dof (biasing `φ̂` low — the incidental-parameters
    /// under-dispersion of #2133). This returns the additive correction summed
    /// over rows/axes,
    ///   Δedf = Σ [ htt/(htt+c+V'') − htt/(htt+V'') ]
    ///        = Σ −htt·c / [(htt+c+V'')·(htt+V'')],
    /// the exact second-order completion of the already-counted GN baseline
    /// (`V'' = max(α·cos κt, 0)` is the SAME clamped von-Mises curvature the
    /// assembly writes into `htt`, so the baseline term cancels the GN edf the
    /// caller already summed). The full denominator is floored into the PD region
    /// (`≥ SURE_DIVERGENCE_PD_FLOOR·(htt+V'')`) so a near-saddle row cannot blow
    /// the divergence up. The measure-zero basin-jump (selection) dof — a
    /// one-sided term peaking only at intermediate basin ambiguity — is NOT added
    /// here: it needs per-row multi-basin enumeration and never over-counts, so
    /// the within-basin term is the dominant, over-shoot-safe piece.
    ///
    /// `residual` is the per-row reconstruction residual `f(θ̂) − y` (n×p) at the
    /// same `rho`/state that produced `cache`. Returns `Ok(0.0)` when no atom
    /// exposes analytic second jets (the correction is then inert — the historical
    /// GN dispersion is unchanged).
    pub(crate) fn coordinate_sure_deflation_correction(
        &self,
        residual: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if residual.dim() != (n, p) {
            return Err(format!(
                "coordinate_sure_deflation_correction: residual {:?} != ({n}, {p})",
                residual.dim()
            ));
        }
        // Second jets are the whole point; if the bases cannot supply them the
        // correction is simply inert (GN dispersion stands).
        let second_jets = match self.atom_second_jets() {
            Ok(jets) => jets,
            Err(_) => return Ok(0.0),
        };
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|m| m.whitens_likelihood());
        let mut assignments = vec![0.0_f64; self.k_atoms()];
        let mut g1 = vec![0.0_f64; p];
        let mut g2 = vec![0.0_f64; p];
        let floor = Self::SURE_DIVERGENCE_PD_FLOOR;
        let mut acc = 0.0_f64;
        for row in 0..n {
            self.assignment.try_assignments_row_for_rho_into(
                row,
                rho,
                assignments.as_mut_slice(),
            )?;
            let r_row = residual.row(row);
            // Metric-applied residual M·r (p-space); identity when not whitening.
            let mr: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, r_row),
                _ => r_row.iter().copied().collect(),
            };
            let mut correct_atom = |k: usize| {
                if rho.log_ard[k].is_empty() {
                    return;
                }
                let coord = &self.assignment.coords[k];
                let d = coord.latent_dim();
                if rho.log_ard[k].len() != d {
                    return;
                }
                let a_k = assignments[k];
                if a_k == 0.0 {
                    return;
                }
                for axis in 0..d {
                    let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                    let t = coord.row(row)[axis];
                    let v_pp = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis])
                        .hess
                        .max(0.0);
                    // GN coordinate curvature htt = a_k²·(g')ᵀ M g'.
                    self.atoms[k].fill_decoded_derivative_row(row, axis, g1.as_mut_slice());
                    let htt = if whitens {
                        if let Some(metric) = self.row_metric.as_ref() {
                            let mg = metric
                                .apply_metric_row(row, ndarray::ArrayView1::from(g1.as_slice()));
                            a_k * a_k * g1.iter().zip(mg.iter()).map(|(&a, &b)| a * b).sum::<f64>()
                        } else {
                            0.0
                        }
                    } else {
                        a_k * a_k * g1.iter().map(|&x| x * x).sum::<f64>()
                    };
                    let denom_gn = htt + v_pp;
                    if !(denom_gn > 0.0) {
                        continue;
                    }
                    // Residual-curvature term c = a_k·(g'')ᵀ M r_code.
                    self.atoms[k].fill_decoded_second_derivative_row(
                        &second_jets[k],
                        row,
                        axis,
                        g2.as_mut_slice(),
                    );
                    let c = a_k * g2.iter().zip(mr.iter()).map(|(&a, &b)| a * b).sum::<f64>();
                    let denom_full = (htt + c + v_pp).max(floor * denom_gn);
                    // Δedf = htt/denom_full − htt/denom_gn (exact 2nd-order completion).
                    acc += htt / denom_full - htt / denom_gn;
                }
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    for &k in &layout.active_atoms[row] {
                        correct_atom(k);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        correct_atom(k);
                    }
                }
            }
        }
        Ok(acc)
    }

    /// Top-weight threshold above which a SOFTMAX row is treated as an
    /// effectively-DISCRETE selection for the #2133 basin-selection dof. Below it
    /// the softmax simplex is genuinely soft: the routing Jacobian `∂a/∂η` is
    /// non-degenerate and the row's reconstruction is a smooth blend whose
    /// y-sensitivity is already carried by the within-basin curvature (the logit
    /// block of `H`), so charging a boundary term there would DOUBLE-count. As the
    /// top weight `a_max → 1` the softmax saturates toward a hard `argmax`: its
    /// Jacobian degenerates, the smooth edf no longer sees the selection, and the
    /// discontinuous boundary term (Tibshirani search dof) is what survives. `0.9`
    /// = "the winner holds ≥90% of the mass" — the runner-up mass `1−a_max` is the
    /// natural soft-vs-hard scale (a uniform `K`-way split gives `1/K` per atom, so
    /// a winner far above `1−1/K` is saturated); a fixed `0.9` is robust across `K`
    /// and conservative (it EXCLUDES ambiguous soft rows, never over-shooting). The
    /// hard gate families (TopK / ThresholdGate / IBP-MAP) are discrete by
    /// construction and bypass this threshold.
    pub(crate) const SOFTMAX_HARD_SELECTION_MIN_TOP_WEIGHT: f64 = 0.9;

    /// #2133 — basin-SELECTION (search) deflation dof: the second Stein term the
    /// within-basin [`Self::coordinate_sure_deflation_correction`] deliberately
    /// omits. The SAE fit is a two-stage map — a discrete routing selection
    /// `b(y) = argmin_b R_b(y)` followed by the smooth within-basin decode — so the
    /// true dof `df = σ⁻² Σ_i cov(ŷ_i, y_i)` picks up a boundary term wherever the
    /// reconstruction JUMPS across a routing decision surface (`ŷ` is discontinuous
    /// there). For the two-candidate archetype `ŷ = m_b`, `b = argmin‖y−m_b‖²_M`,
    /// the boundary integral collapses to the closed form
    ///   `df_search = Σ_i w_i·(‖δ_i‖_M/σ̂)·φ( Δ_i / (2 σ̂ ‖δ_i‖_M) )`,
    /// with `δ_i = ŷ¹_i − ŷ²_i` the reconstruction jump, `Δ_i = ‖y−ŷ²‖²_M −
    /// ‖y−ŷ¹‖²_M` the residual margin, `σ̂` the noise scale, and `φ` the standard
    /// normal density. Per row the WINNER candidate `ŷ¹` is the fitted mixture and
    /// the PAIRED candidate `ŷ²` reassigns the top-mass atom's weight `a_w` to the
    /// runner-up atom, so `δ_i = a_w·(decode_w − decode_r)` (the tier-0 mean and all
    /// other atoms cancel in the difference). Writing `r = ŷ¹ − y` (= `residual`),
    /// `Δ_i = ‖δ‖²_M − 2⟨r, δ⟩_M`.
    ///
    /// The charge fires ONLY on effectively-discrete selection — TopK (the
    /// `k`-th↔`(k+1)`-th support swap), ThresholdGate / IBP-MAP hard gates, and
    /// SATURATED softmax rows (`a_max ≥ SOFTMAX_HARD_SELECTION_MIN_TOP_WEIGHT`);
    /// genuinely-soft softmax rows contribute 0 (their selection smoothness is
    /// already in the within-basin edf — charging them double-counts). It is also
    /// identically 0 for single-atom / single-gated rows (no runner-up), for
    /// candidates that predict identically (`δ = 0`, a 2→1 degeneracy = zero
    /// selection cost), and whenever routing is FROZEN/amortized (`frozen_logits`
    /// set — the selection is fixed data, not an estimated parameter, so it consumes
    /// no dof). Both analytic anchors hold: as one basin dominates the residual
    /// margin `Δ → +∞` so `φ → 0` (reduces to the single-basin within-basin term),
    /// and identical candidates give `δ = 0`. `dispersion` seeds `σ̂ = √φ̂` from the
    /// within-basin-corrected (search-UNcorrected) φ̂ — the caller runs the single
    /// monotone fixed-point pass. HT `w_i`-weighted so it composes with the F6
    /// weight-aware RSS.
    pub(crate) fn basin_selection_deflation_correction(
        &self,
        residual: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        dispersion: f64,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if residual.dim() != (n, p) {
            return Err(format!(
                "basin_selection_deflation_correction: residual {:?} != ({n}, {p})",
                residual.dim()
            ));
        }
        let sigma = dispersion.sqrt();
        if !(sigma > 0.0) {
            return Ok(0.0);
        }
        // Frozen / amortized routing (#1033): the per-row selection is a fixed
        // function of `x`, NOT an estimated parameter, so it consumes no search dof
        // — the correction must stay identically 0 (φ̂ bit-for-bit today's).
        if self.assignment.frozen_logits.is_some() {
            return Ok(0.0);
        }
        let k_atoms = self.k_atoms();
        if k_atoms < 2 {
            return Ok(0.0);
        }
        if self.assignment.ungated.len() != k_atoms {
            return Err(format!(
                "basin_selection_deflation_correction: ungated mask has length {}, expected {k_atoms}",
                self.assignment.ungated.len()
            ));
        }
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|m| m.whitens_likelihood());
        let row_w = self.row_loss_weights.as_deref();
        let norm_const = std::f64::consts::TAU.sqrt(); // √(2π)
        let sat = Self::SOFTMAX_HARD_SELECTION_MIN_TOP_WEIGHT;
        let mut assignments = vec![0.0_f64; k_atoms];
        let mut decode_w = vec![0.0_f64; p];
        let mut decode_r = vec![0.0_f64; p];
        let mut delta = vec![0.0_f64; p];
        // Reusable ranking scratch of GATED (non-ungated) atoms by descending logit.
        let mut ranked: Vec<usize> = Vec::with_capacity(k_atoms);
        let mut acc = 0.0_f64;
        for row in 0..n {
            self.assignment.try_assignments_row_for_rho_into(
                row,
                rho,
                assignments.as_mut_slice(),
            )?;
            let logits = self.assignment.logits.row(row);
            // Selectable candidates are the GATED atoms (ungated = always-on
            // background tier `a_k≡1`, not part of the selection simplex).
            // Ranks needed on the selection boundary: the `(k+1)`-th for a TopK
            // support swap, else just the top-2. Selecting only these (below) keeps
            // the per-row cost O(K), not the O(K log K) of a full logit sort — a cost
            // the massive-K streaming rank-charge dispersion cannot pay.
            let need = match self.assignment.mode {
                AssignmentMode::TopK { k } => k.saturating_add(1),
                _ => 2,
            };
            ranked.clear();
            for atom in 0..k_atoms {
                if !logits[atom].is_finite() {
                    return Err(format!(
                        "basin_selection_deflation_correction: non-finite logit on row {row}, atom {atom}"
                    ));
                }
                if !self.assignment.ungated[atom] {
                    ranked.push(atom);
                }
            }
            if ranked.len() < need {
                continue;
            }
            // O(K) partial selection: bring the top-`need` gated atoms (DESCENDING
            // logit — the routing order the boundary lives on) to the front, then
            // order just those.
            if ranked.len() > need {
                ranked.select_nth_unstable_by(need - 1, |a, b| logits[*b].total_cmp(&logits[*a]));
                ranked.truncate(need);
            }
            ranked.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            // The boundary pair `(w, r)` = (the atom whose weight is at stake, the
            // runner-up it would flip to) and the winner mass `a_w` moved across it.
            let (w, r, a_w) = match self.assignment.mode {
                // Hard top-`k` support: the live boundary is the WEAKEST selected
                // (`k`-th) vs the STRONGEST unselected (`(k+1)`-th) — the cheapest
                // one-atom support swap. Both carry unit gate weight.
                AssignmentMode::TopK { k } => {
                    if k == 0 {
                        continue;
                    }
                    // Unit gate weight is swapped from the k-th to the (k+1)-th atom.
                    (ranked[k - 1], ranked[k], 1.0)
                }
                // Saturated-softmax gate: only near-hard rows carry a search dof.
                AssignmentMode::Softmax { .. } => {
                    let top = ranked[0];
                    let a_top = assignments[top];
                    if a_top < sat {
                        continue;
                    }
                    (top, ranked[1], a_top)
                }
                // Per-atom hard gates (IBP-MAP indicator, ThresholdGate hard
                // sigmoid): discrete by construction. Dominant-pair approximation —
                // the top-mass gated atom vs its nearest competitor (the exact
                // multi-gate boundary enumeration is the documented follow-up, and
                // like the within-basin term this dominant piece never over-counts).
                AssignmentMode::IBPMap { .. } | AssignmentMode::ThresholdGate { .. } => {
                    let top = ranked[0];
                    (top, ranked[1], assignments[top])
                }
            };
            if w == r || !(a_w > 0.0) {
                continue;
            }
            self.atoms[w].fill_decoded_row(row, decode_w.as_mut_slice());
            self.atoms[r].fill_decoded_row(row, decode_r.as_mut_slice());
            for c in 0..p {
                delta[c] = a_w * (decode_w[c] - decode_r[c]);
            }
            // Metric-applied jump `M·δ`; identity when not whitening.
            let m_delta: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => {
                    metric.apply_metric_row(row, ndarray::ArrayView1::from(delta.as_slice()))
                }
                _ => delta.clone(),
            };
            let delta_norm2: f64 = delta.iter().zip(m_delta.iter()).map(|(&a, &b)| a * b).sum();
            if !(delta_norm2 > 0.0) {
                continue; // identical candidates ⇒ zero selection cost (anchor 2).
            }
            let delta_norm = delta_norm2.sqrt();
            // ⟨r, δ⟩_M = rᵀ(Mδ); `residual` row is r = ŷ¹ − y.
            let r_dot_delta: f64 = residual
                .row(row)
                .iter()
                .zip(m_delta.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            // Δ = ‖y−ŷ²‖²_M − ‖y−ŷ¹‖²_M = ‖δ‖²_M − 2⟨r, δ⟩_M.
            let gap = delta_norm2 - 2.0 * r_dot_delta;
            let z = gap / (2.0 * sigma * delta_norm);
            let phi_z = (-0.5 * z * z).exp() / norm_const;
            let w_row = row_w.map_or(1.0, |wt| wt[row]);
            acc += w_row * (delta_norm / sigma) * phi_z;
        }
        Ok(acc)
    }

    /// Atom-count threshold at/above which [`Self::ard_inverse_traces`] switches
    /// from the exact selected-inverse latent diagonal (one dense `K×K` Schur
    /// solve per latent coordinate — the `O(total_t·K²) ≈ O(K³)` massive-`K`
    /// wall) to the matrix-free Hutchinson stochastic-diagonal estimator
    /// [`Self::latent_block_inverse_diagonal_hutchinson`]. Set to match the
    /// smoothness-dof Hutchinson gate ([`Self::SMOOTHNESS_DOF_HUTCHINSON_MIN_ATOMS`]),
    /// well above every exact-path test fixture so ordinary-`K` behaviour — and
    /// its bit-for-bit tests — is unchanged; the estimator engages only in the
    /// massive dictionary regime (`K` up to 32k).
    pub(crate) const ARD_TRACE_HUTCHINSON_MIN_ATOMS: usize = 2048;
    /// Rademacher probe count for the Hutchinson latent-inverse-diagonal
    /// estimator. One [`ArrowFactorCache::full_inverse_apply`] per probe yields
    /// the WHOLE diagonal at once, so this is the total full-arrow solve count
    /// that replaces the exact `total_t` per-coordinate Schur solves.
    pub(crate) const ARD_TRACE_HUTCHINSON_PROBES: usize = 64;
    /// Fixed base seed so the ARD-trace estimate is bit-reproducible across REML
    /// outer iterations (cf. the SLQ log-det and smoothness-dof seeds).
    pub(crate) const ARD_TRACE_HUTCHINSON_SEED: u64 = 0x5AED_A3D0_1ACE_9C01;

    /// Matrix-free Hutchinson estimate of `diag((H⁻¹)_tt)` — the SAME quantity
    /// [`ArrowFactorCache::latent_block_inverse_diagonal`] returns EXACTLY, but at
    /// `O(num_probes · matvec)` instead of the exact `O(total_t · K²)`.
    ///
    /// The exact selected-inverse builds the latent inverse diagonal one
    /// coordinate at a time, each coordinate paying a dense `K×K` Schur solve;
    /// over all `total_t = Σ_i d_i` latent coordinates that is `O(total_t·K²) ≈
    /// O(K³)` at massive `K` (32k). This estimator replaces the per-coordinate
    /// loop with `num_probes` full-arrow solves: for a Rademacher probe `z` over
    /// the `t`-block (`E[z zᵀ] = I`), `u_t = (H⁻¹)_tt z` — the `t`-block of
    /// `H⁻¹·[z; 0]`; the trailing `w_β = 0` drops the border coupling out of the
    /// `t`-block — so the Hadamard product `z ⊙ u_t` has expectation exactly
    /// `diag((H⁻¹)_tt)` (off-diagonal `i≠j` terms are mean-zero under
    /// `E[z_i z_j] = 0`). Averaging over probes gives the unbiased diagonal. Each
    /// probe is ONE [`ArrowFactorCache::full_inverse_apply`] (per-row solves + a
    /// SINGLE Schur solve + the rank-`R` cross-row Woodbury correction — the same
    /// `H_full` the exact path inverts), so the IBP curvature is included
    /// identically.
    ///
    /// Probes run serially and accumulate in a fixed order, so for a fixed
    /// `(seed, num_probes)` the estimate is bit-reproducible (the REML determinism
    /// contract, matching the SLQ log-det and smoothness-dof Hutchinson paths).
    pub(crate) fn latent_block_inverse_diagonal_hutchinson(
        cache: &ArrowFactorCache,
        num_probes: usize,
        seed: u64,
    ) -> Result<Array1<f64>, ArrowSchurError> {
        let total_len = cache.delta_t_len();
        let k = cache.k;
        let probes = num_probes.max(1);
        let mut out = Array1::<f64>::zeros(total_len);
        let mut z = Array1::<f64>::zeros(total_len);
        let w_beta_zero = Array1::<f64>::zeros(k);
        for probe in 0..probes {
            // Deterministic Rademacher probe (±1) over the t-block, seeded by
            // `seed + probe` so the whole estimate is reproducible.
            let mut state = seed.wrapping_add(probe as u64);
            let mut bits = 0u64;
            let mut remaining = 0u32;
            for zi in z.iter_mut() {
                if remaining == 0 {
                    bits = gam_linalg::utils::splitmix64(&mut state);
                    remaining = 64;
                }
                *zi = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
            }
            // u_t = (H⁻¹)_tt z (w_β = 0 ⇒ the border coupling drops from the
            // t-block); this is the FULL H_full inverse incl. cross-row Woodbury.
            let (u_t, _u_beta) = cache.full_inverse_apply(z.view(), w_beta_zero.view())?;
            for i in 0..total_len {
                out[i] += z[i] * u_t[i];
            }
        }
        let inv_p = 1.0 / (probes as f64);
        for v in out.iter_mut() {
            *v *= inv_p;
        }
        Ok(out)
    }

    pub(crate) fn ard_log_precision_explicit_derivatives(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Array1<f64>>, String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atoms but term has {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs() as f64;
        // HT row weighting: this is the ρ-derivative of `ard_value` (the `explicit`
        // outer-gradient channel), so it carries the identical per-row inclusion
        // weight and effective row count `n_eff = Σᵢ wᵢ` as the energy — otherwise
        // the analytic gradient desyncs from the (now weight-aware) criterion value
        // on the subsample. `None` ⇒ `w_row = 1`, `n_eff = n`, historical exactly.
        let row_w = self.row_loss_weights.as_deref();
        let n_eff = row_w.map_or(n, |w| w.iter().sum::<f64>());
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            let mut atom_out = Array1::<f64>::zeros(rho.log_ard[atom_idx].len());
            if rho.log_ard[atom_idx].is_empty() {
                out.push(atom_out);
                continue;
            }
            if rho.log_ard[atom_idx].len() != d {
                return Err(format!(
                    "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                    rho.log_ard[atom_idx].len()
                ));
            }
            let periods = coord.effective_axis_periods();
            for axis in 0..d {
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom_idx][axis]);
                let period = periods[axis];
                let mut energy_deriv = 0.0_f64;
                for row in 0..coord.n_obs() {
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let t = coord.row(row)[axis];
                    energy_deriv += w_row * ArdAxisPrior::eval(alpha, t, period).value;
                }
                let normalizer_deriv = match period {
                    None => -0.5 * n_eff,
                    Some(p) => {
                        let kappa = std::f64::consts::TAU / p;
                        let eta = alpha / (kappa * kappa);
                        // d/d(log α) of `n[-η + log I0(η)]` = `n η (I1/I0 - 1)`.
                        // The ratio is computed without forming `e^{η}`, so it
                        // stays finite for large `η` instead of the `inf/inf =
                        // NaN` that `bessel_i1(η)/bessel_i0(η)` produces (#1113).
                        let ratio = bessel_i0_log_and_ratio(eta).1;
                        n_eff * eta * (-1.0 + ratio)
                    }
                };
                atom_out[axis] = energy_deriv + normalizer_deriv;
            }
            out.push(atom_out);
        }
        Ok(out)
    }

    pub(crate) fn ard_log_precision_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        // RAW selected-inverse diagonal: the per-axis diagonal contraction uses
        // the DEFLATED inverse; the full kept-subspace + rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per (row, axis)
        // afterwards via the Daleckii–Krein helper. Each ARD ρ-component
        // `(atom k, axis)` differentiates a SINGLE coordinate-slot diagonal entry,
        // so its `D` is the rank-one `hess·e_s e_sᵀ` at that local slot `s`.
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
        // HT row weighting: the assembled per-row ARD curvature `∂H/∂logα` is scaled
        // by the inclusion weight `wᵢ` (see the assembly seam), and `inv_diag` = the
        // diagonal of `H⁻¹` already reflects that w-scaled `H`. So each row's trace
        // contribution `½·(H⁻¹)_ss·(wᵢ·hess)` must carry the SAME `wᵢ` here, or the
        // ½log|H| gradient desyncs from the assembled Hessian on the subsample.
        // `None` ⇒ `w_row = 1`, bit-for-bit the historical trace.
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        // Hoisted RHS scratch reused across every (row, col) solve. Setting and
        // clearing a SINGLE entry per column is O(1); a fresh
        // `Array1::zeros(total_t)` memsets total_t≈n·q slots per inner iteration
        // (O(n) per col ⇒ O(n²) redundant zeroing across the block build).
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        for row in 0..n {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let row_base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            // Per-row selected-inverse t-block, built once (only when deflated).
            let inv_vv = if dirs.is_empty() {
                None
            } else {
                let mut m = Array2::<f64>::zeros((q, q));
                for col in 0..q {
                    rhs_t_scratch[row_base + col] = 1.0;
                    let solved = solver
                        .solve(rhs_t_scratch.view(), rhs_beta_zero.view())
                        .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
                    rhs_t_scratch[row_base + col] = 0.0;
                    for r in 0..q {
                        m[[r, col]] = solved.t[row_base + r];
                    }
                }
                Some(m)
            };
            // Correction for one local coordinate slot `s` with curvature `hess`.
            let slot_correction = |s: usize, hess: f64| -> f64 {
                let Some(iv) = inv_vv.as_ref() else {
                    return 0.0;
                };
                if s >= q || hess == 0.0 {
                    return 0.0;
                }
                let mut d = Array2::<f64>::zeros((q, q));
                d[[s, s]] = hess;
                Self::deflation_block_correction(iv, &d, dirs, spectrum)
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            let hess = w_row * prior.hess.max(0.0);
                            let s = block_start + axis;
                            traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                            traces[k][axis] -= 0.5 * slot_correction(s, hess);
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            let hess = w_row * prior.hess.max(0.0);
                            let s = block_start + axis;
                            traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                            traces[k][axis] -= 0.5 * slot_correction(s, hess);
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Per-atom, per-axis `½ tr(H⁻¹ ∂H/∂logα_{kj})` — the ARD ½log|H| ρ-gradient
    /// channel [`Self::ard_log_precision_hessian_trace`] computes — from the #2080
    /// SHARED selected-inverse bundle instead of the dense `DeflatedArrowSolver`
    /// (`latent_inverse_diagonal` + per-column `solve`). The massive-lane /
    /// eventual-dense-cache-retirement replacement for the last dense `S⁻¹` in the
    /// analytic outer ρ-gradient's ARD block.
    ///
    /// Each ARD component differentiates ONE coordinate-slot diagonal entry, so the
    /// trace is `Σ_{row, slot s(k,j)} ½·(H⁻¹)_tt[s,s]·(w_row·hess)`. The diagonal is
    /// the SAME per-row arrow selected-inverse the ARD posterior-variance trace uses
    /// (`(A_i⁻¹)[s,s]` row-local + border `(1/m)Σ_j w_ij[s]·s_ij[s]` off the bundle,
    /// see [`Self::ard_inverse_traces_from_probes`] for the derivation), matching
    /// `solver.latent_inverse_diagonal()` on the PLAIN (undeflated) selected inverse
    /// to solve precision.
    ///
    /// # Deflation is out of scope for this lane
    ///
    /// The dense path's Daleckii–Krein correction `−½ tr(inv_vv·(D − DΦ[D]))`
    /// reconstructs the per-row gauge/rotation deflation the DEFLATED `solve`
    /// removes; it needs the DEFLATED per-row inverse block. The surrogate bundle
    /// carries the PLAIN reduced-Schur `S⁻¹`, so `A_i⁻¹ + G_i S⁻¹ G_iᵀ` is the
    /// UNdeflated block — correct only where no row carries deflation directions
    /// (`deflated_row_directions` empty), which is precisely the matrix-free
    /// surrogate regime (per-row gauge deflation is a dense-solver feature). A row
    /// that DOES carry deflation is hard-refused here (the caller must keep the
    /// dense channel for it) rather than silently double-adding the correction on an
    /// undeflated block. On the plain regime the correction term is identically
    /// zero, so the two paths agree exactly — the FD gate's acceptance.
    ///
    /// BOUNDARY (#2080): for the surrogate to OWN a deflated-row dense fit, the
    /// lane would have to emit a DEFLATED reduced-Schur `S⁻¹` bundle (so the
    /// reformulated block is the deflated inverse the correction expects) — a
    /// separate design step, taken only if we ever route the deflated regime
    /// matrix-free. Until then, deflated rows stay on the dense channel.
    // #2080 analytic-gradient cluster channel: wired into
    // `analytic_outer_rho_gradient_components_with_bundle`'s `Some`-bundle branch
    // (the all-or-nothing selected-inverse cluster, alongside the from-probes
    // smoothness EDF), dormant until the analytic-gradient routing flips (every
    // caller passes `None` today). That real call site is the non-test consumer,
    // so this matches its `pub(crate)` sibling `ard_inverse_traces_from_probes`.
    pub(crate) fn ard_log_precision_hessian_trace_from_probes(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let m = probes.len();
        if m == 0 || sinv_probes.len() != m {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "ard_log_precision_hessian_trace_from_probes: need matching non-empty \
                     probe/solve bundles, got {m} probes and {} solves",
                    sinv_probes.len()
                ),
            });
        }
        let k_border = cache.k;
        for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
            for (j, v) in set.iter().enumerate() {
                if v.len() != k_border {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "ard_log_precision_hessian_trace_from_probes: {label} {j} has length \
                             {} != border dim {k_border}",
                            v.len()
                        ),
                    });
                }
            }
        }
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        let inv_m = 1.0 / (m as f64);
        for row in 0..n {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let q = cache.row_dims[row];
            // Per-row gauge/rotation deflation is unsupported on the plain-`S⁻¹`
            // surrogate bundle (see the docstring); the dense channel owns those rows.
            if cache
                .deflated_row_directions
                .get(row)
                .is_some_and(|d| !d.is_empty())
            {
                return Err(ArrowSchurError::SchurFactorFailed {
                    reason: format!(
                        "ard_log_precision_hessian_trace_from_probes: row {row} carries \
                         deflation directions; the plain-S⁻¹ bundle cannot reconstruct the \
                         Daleckii–Krein correction — route this fit through the dense channel"
                    ),
                });
            }
            let factor = cache.undamped_factor(row);
            // Row-local (A_i⁻¹)[s,s].
            let mut a_inv_diag = Array1::<f64>::zeros(q);
            let mut e_s = Array1::<f64>::zeros(q);
            for s in 0..q {
                e_s.fill(0.0);
                e_s[s] = 1.0;
                let col = cholesky_solve_vector(factor, e_s.view());
                a_inv_diag[s] = col[s];
            }
            // Border vectors w_ij = A_i⁻¹ H_tβ^i z_j, s_ij = A_i⁻¹ H_tβ^i (S⁻¹ z_j).
            let mut w_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut s_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut b_tmp = Array1::<f64>::zeros(q);
            for j in 0..m {
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, probes[j].view(), &mut b_tmp) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "ard_log_precision_hessian_trace_from_probes: H_tβ^({row}) probe apply \
                             failed"
                        ),
                    });
                }
                w_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, sinv_probes[j].view(), &mut b_tmp) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "ard_log_precision_hessian_trace_from_probes: H_tβ^({row}) solve apply \
                             failed"
                        ),
                    });
                }
                s_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
            }
            // Full per-row selected-inverse diagonal (undeflated).
            let mut inv_diag_local = Array1::<f64>::zeros(q);
            for s in 0..q {
                let mut border = 0.0_f64;
                for j in 0..m {
                    border += w_probes[j][s] * s_probes[j][s];
                }
                inv_diag_local[s] = a_inv_diag[s] + inv_m * border;
            }
            let accumulate = |k: usize, block_start: usize, traces: &mut Vec<Array1<f64>>| {
                if rho.log_ard[k].is_empty() {
                    return;
                }
                let coord = &self.assignment.coords[k];
                let d = coord.latent_dim();
                for axis in 0..d {
                    let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                    let t = coord.row(row)[axis];
                    let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                    let hess = w_row * prior.hess.max(0.0);
                    let s = block_start + axis;
                    traces[k][axis] += 0.5 * inv_diag_local[s] * hess;
                }
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        accumulate(k, starts[pos], &mut traces);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        accumulate(k, coord_offsets[k], &mut traces);
                    }
                }
            }
        }
        Ok(traces)
    }
}
