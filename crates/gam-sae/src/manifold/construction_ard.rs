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
        let mut out = Vec::with_capacity(self.k_atoms());
        for coord in &self.assignment.coords {
            let d = coord.latent_dim();
            let periods = coord.effective_axis_periods();
            let mut sq = Array1::<f64>::zeros(d);
            for row in 0..coord.n_obs() {
                let t = coord.row(row);
                for axis in 0..d {
                    // `sq_equiv` is independent of `alpha`; pass 1.0.
                    sq[axis] += ArdAxisPrior::eval(1.0, t[axis], periods[axis]).sq_equiv;
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
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
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
        let coord_offsets = self.assignment.coord_offsets();
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
            self.assignment
                .try_assignments_row_for_rho_into(row, rho, assignments.as_mut_slice())?;
            let r_row = residual.row(row);
            // Metric-applied residual M·r (p-space); identity when not whitening.
            let mr: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, r_row),
                _ => r_row.iter().copied().collect(),
            };
            let mut correct_atom = |k: usize, block_start: usize| {
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
                let _ = block_start; // coord block position is not needed here.
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
                            let mg = metric.apply_metric_row(
                                row,
                                ndarray::ArrayView1::from(g1.as_slice()),
                            );
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
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        correct_atom(k, starts[pos]);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        correct_atom(k, coord_offsets[k]);
                    }
                }
            }
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
                    let t = coord.row(row)[axis];
                    energy_deriv += ArdAxisPrior::eval(alpha, t, period).value;
                }
                let normalizer_deriv = match period {
                    None => -0.5 * n,
                    Some(p) => {
                        let kappa = std::f64::consts::TAU / p;
                        let eta = alpha / (kappa * kappa);
                        // d/d(log α) of `n[-η + log I0(η)]` = `n η (I1/I0 - 1)`.
                        // The ratio is computed without forming `e^{η}`, so it
                        // stays finite for large `η` instead of the `inf/inf =
                        // NaN` that `bessel_i1(η)/bessel_i0(η)` produces (#1113).
                        let ratio = bessel_i0_log_and_ratio(eta).1;
                        n * eta * (-1.0 + ratio)
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
                            let hess = prior.hess.max(0.0);
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
                            let hess = prior.hess.max(0.0);
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
}
