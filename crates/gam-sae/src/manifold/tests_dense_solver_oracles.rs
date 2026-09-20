//! Dense deflated-solver oracles for the SAE outer ρ-gradient.
//!
//! Production assembles the analytic outer gradient from the exact-A logdet
//! channels (`DenseExactAGeometry` / `ArrowOrbitGeometry`) or the frozen probe
//! bundle. The explicit dense Woodbury-deflated arrow solve below is the
//! independent reference those channels are checked against: it materialises
//! the gauge-deflated joint inverse and forms every trace and the implicit
//! stationarity solve directly. It never runs in a fit.

use super::*;
use gam_solve::arrow_schur::{arrow_factor_max_pivot, arrow_factor_min_pivot};

/// Relative Cholesky-pivot floor of the dense oracle's B-cache gate.
///
/// The evidence value can still be honest below this threshold because it only
/// sums `log(diag(L))`. The oracle's selected-inverse traces and
/// `ArrowFactorCache::full_inverse_apply` divide by those pivots, so once
/// `min_pivot / max_pivot` is below this floor it must either identify a
/// closed-form gauge orbit and stiffen only that quotient direction, or refuse
/// the trial rho as numerically singular.
pub(crate) const SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR: f64 = 1.0e-12;

pub(crate) const SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR: f64 = 1.0e-8;

pub(crate) struct DeflatedArrowSolver<'a> {
    pub(crate) cache: &'a ArrowFactorCache,
    pub(crate) gauge_basis: Vec<Array1<f64>>,
    pub(crate) gauge_response_physical: Vec<Array1<f64>>,
    /// `M = GᵀH⁻¹G`, the gauge metric the Woodbury factor was built from.
    pub(crate) gauge_metric: Array2<f64>,
    pub(crate) woodbury_factor: Option<FaerCholeskyFactor>,
    pub(crate) gauge_stiffness: f64,
}

impl<'a> DeflatedArrowSolver<'a> {
    pub(crate) fn plain(cache: &'a ArrowFactorCache) -> Self {
        Self {
            cache,
            gauge_basis: Vec::new(),
            gauge_response_physical: Vec::new(),
            gauge_metric: Array2::<f64>::zeros((0, 0)),
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
            gauge_metric,
            woodbury_factor: Some(woodbury_factor),
            gauge_stiffness: stiffness,
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

    /// Diagonal of the latent block of the operator [`Self::solve`] inverts.
    ///
    /// Plain arrow: the selected-inverse diagonal of `H⁻¹`. With gauges `G`
    /// (orthonormal columns `g_a`) at stiffness `s`, `solve` is `(H + s·GGᵀ)⁻¹`, and
    /// its `idx` diagonal entry reads off `solve(e_idx)` in closed form. With
    /// `c_a = (H⁻¹g_a)[idx] = R_a[idx] + Σ_b M[b,a]·g_b[idx]` (`R_a` the stored
    /// physical responses, `M = GᵀH⁻¹G`) and `w = W⁻¹c`, `W = I/s + M`:
    ///
    /// ```text
    ///   out[idx] = (H⁻¹)[idx,idx] − Σ_a (g_a[idx]·c_a + R_a[idx]·w_a − g_a[idx]·w_a/s)
    /// ```
    ///
    /// That is `solve`'s own elimination applied to `e_idx`, so the result equals
    /// the per-coordinate `solve` loop up to rounding, at `O(r²)` per coordinate
    /// on top of the plain diagonal instead of one full bordered solve per
    /// coordinate (#2900).
    pub(crate) fn latent_inverse_diagonal(&self) -> Result<Array1<f64>, String> {
        let mut out = self
            .cache
            .latent_block_inverse_diagonal()
            .map_err(|err| format!("DeflatedArrowSolver: latent inverse diagonal: {err}"))?;
        let Some(factor) = self.woodbury_factor.as_ref() else {
            return Ok(out);
        };
        let rank = self.gauge_basis.len();
        let stiffness_recip = self.gauge_stiffness.recip();
        let mut coeffs = Array1::<f64>::zeros(rank);
        for idx in 0..out.len() {
            for a in 0..rank {
                let mut value = self.gauge_response_physical[a][idx];
                for b in 0..rank {
                    value += self.gauge_metric[[b, a]] * self.gauge_basis[b][idx];
                }
                coeffs[a] = value;
            }
            let weights = factor.solvevec(&coeffs);
            let mut correction = 0.0_f64;
            for a in 0..rank {
                let gauge = self.gauge_basis[a][idx];
                correction += gauge * coeffs[a] + self.gauge_response_physical[a][idx] * weights[a]
                    - stiffness_recip * gauge * weights[a];
            }
            out[idx] -= correction;
        }
        Ok(out)
    }
}

impl SaeManifoldTerm {
    /// Orthonormal analytic chart-gauge basis in one assembled arrow layout.
    /// Both dense exact-A quotient geometry and matrix-free arrow consumers use
    /// this basis, so the physical subspace cannot depend on representation.
    pub(crate) fn joint_chart_gauge_basis_for_arrow_layout(
        &self,
        row_offsets: &[usize],
        border_dim: usize,
        owner: &str,
    ) -> Result<Vec<Array1<f64>>, String> {
        let mut basis = Vec::<Array1<f64>>::new();
        // #2653 — a BORDER-dimension disagreement is not a stale layout, it is a
        // different chart. Hard TopK compaction drops inactive per-row coordinate
        // BLOCKS and retains the decoder border unchanged, so a genuine
        // full-to-compact map always agrees on the border; only the row widths
        // move (the filed 132 -> 84 case). When the border itself differs, this
        // operator is not a compaction of the joint chart at all and the
        // closed-form chart gauges simply do not live in its space. That is the
        // "no matching gauge" condition the caller already diagnoses as
        // `NonIdentifiable` — reporting it as an internal invariant error instead
        // converts a legitimate, more specific refusal into a bug report
        // (regression: `outer_gradient_solver_rejects_near_singular_cache_without_matching_gauge`
        // saw `arrow border dimension 1 != term border dimension 3`).
        // Row-layout disagreement stays a typed invariant error below, because
        // there the gauge IS mappable and silently skipping it would put an
        // analytic chart null back into the physical spectrum.
        if border_dim != self.factored_border_dim() {
            return Ok(Vec::new());
        }
        let mut orthogonality_defect = 0.0_f64;
        for dense in self.dense_step_gauge_vectors()? {
            let mut gauge = self.dense_joint_vector_in_arrow_layout(
                dense.view(),
                row_offsets,
                border_dim,
                owner,
            )?;
            let original_norm = gauge.dot(&gauge).max(0.0).sqrt();
            if !(original_norm.is_finite() && original_norm > 0.0) {
                continue;
            }
            // Two-pass MGS gives the same stable quotient basis to the dense and
            // matrix-free paths. A dependent candidate leaves only rounding: two
            // passes over `k` bases stay inside `2k·(γ_{N+4} + Σω)·‖g₀‖` (as in 10347d95e).
            for _ in 0..2 {
                for kept in &basis {
                    let coefficient = gauge.dot(kept);
                    gauge.scaled_add(-coefficient, kept);
                }
            }
            let residual_norm = gauge.dot(&gauge).max(0.0).sqrt();
            let growth = gam_linalg::roundoff::accumulation_growth(gauge.len() + 4);
            let band = 2.0 * basis.len() as f64 * (growth + orthogonality_defect);
            if !(residual_norm.is_finite()
                && residual_norm > band * original_norm)
            {
                continue;
            }
            orthogonality_defect += band * original_norm / residual_norm;
            gauge.mapv_inplace(|value| value / residual_norm);
            basis.push(gauge);
        }
        Ok(basis)
    }

    /// `operator` (#2515) names WHICH curvature the `∂H/∂log α` operand is:
    /// `Majorizer` differentiates `B`'s PSD-clamped `α·s_{τ₀}(cos κt)`,
    /// `ExactObservedInformation` differentiates `A`'s unmajorized `α·cos κt`.
    /// It must agree with the operator whose inverse `solver` factors, which is
    /// why every trace entry point takes it explicitly rather than defaulting.
    pub(crate) fn ard_log_precision_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        operator: EvidenceOperator,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let ard_precisions = self
            .validated_ard_precisions(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        // RAW selected-inverse diagonal: the per-axis diagonal contraction uses
        // the DEFLATED inverse; the full kept-subspace + rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per (row, axis)
        // afterwards via the Daleckii–Krein helper. An ARD ρ-component
        // `(atom k, axis)` on a flat factor differentiates a SINGLE coordinate-slot
        // diagonal entry, so its `D` is the rank-one `hess·e_s e_sᵀ` at that local
        // slot `s`. On an embedded sphere `D` is the Riemannian block derivative
        // (`ard_sphere_log_precision_derivative`), contracted against the row's
        // whole selected-inverse block (#2933 F24).
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
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
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
        let sphere_factors = self.all_ard_embedded_sphere_factors();
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
            let row_atoms: Vec<(usize, usize)> = match self.last_row_layout {
                Some(ref layout) => layout.active_atoms[row]
                    .iter()
                    .copied()
                    .zip(layout.coord_starts[row].iter().copied())
                    .collect(),
                None => (0..self.k_atoms()).map(|k| (k, coord_offsets[k])).collect(),
            };
            let sphere_row = row_atoms
                .iter()
                .any(|&(k, _)| !rho.log_ard[k].is_empty() && !sphere_factors[k].is_empty());
            // Per-row selected-inverse t-block, built once (only when deflated, or
            // when a sphere axis contracts a whole block).
            let inv_vv = if !Self::row_deflation_is_live(dirs, spectrum) && !sphere_row {
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
            for (k, block_start) in row_atoms {
                if rho.log_ard[k].is_empty() {
                    continue;
                }
                let coord = &self.assignment.coords[k];
                let point = coord.row(row);
                for axis in 0..coord.latent_dim() {
                    let alpha = ard_precisions[k][axis];
                    let prior = ArdAxisPrior::eval(alpha, point[axis], ard_axis_periods[k][axis]);
                    let hess = w_row * prior.log_precision_curvature(operator);
                    if let Some(derivative) = Self::ard_sphere_log_precision_derivative(
                        &sphere_factors[k],
                        point,
                        axis,
                        block_start,
                        q,
                        hess,
                        w_row * prior.grad,
                    ) {
                        let Some(inverse) = inv_vv.as_ref() else {
                            return Err(ArrowSchurError::SchurFactorFailed {
                                reason: format!(
                                    "ard_log_precision_hessian_trace: row {row} holds a sphere ARD \
                                     axis but its selected-inverse block was not built"
                                ),
                            });
                        };
                        traces[k][axis] += 0.5
                            * ((inverse * &derivative).sum()
                                - Self::deflation_block_correction(
                                    inverse,
                                    &derivative,
                                    dirs,
                                    spectrum,
                                ));
                        continue;
                    }
                    let s = block_start + axis;
                    traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                    traces[k][axis] -= 0.5 * slot_correction(s, hess);
                }
            }
        }
        Ok(traces)
    }

    pub(crate) fn outer_gradient_arrow_solver<'a>(
        &'a self,
        cache: &'a ArrowFactorCache,
        penalized_gram_scale: &[f64],
    ) -> Result<DeflatedArrowSolver<'a>, OuterGradientError> {
        let Err(conditioning_err) = Self::outer_gradient_conditioning_error(cache) else {
            return Ok(DeflatedArrowSolver::plain(cache));
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(conditioning_err);
        };
        if !(max_pivot.is_finite() && max_pivot > 0.0) {
            return Err(conditioning_err);
        }

        // The conditioning gate has already flagged a near-singular joint Hessian
        // (`conditioning_err`). Below we attempt to attribute that flatness to the
        // closed-form gauge orbit (chart step gauges) plus the penalty-aware
        // decoder-null directions and deflate it. When NO such deflatable
        // direction can be recovered, the flat subspace is genuinely
        // non-identifiable -- a degenerate direction OUTSIDE the gauge orbit -- a
        // diagnosis distinct from the raw pivot-ratio conditioning trip.
        // Surfacing the gauge-degenerate case as its own
        // [`OuterGradientError::NonIdentifiable`] preserves that typed evidence
        // when the derivative is refused.
        let non_identifiable_err = OuterGradientError::NonIdentifiable {
            reason: format!(
                "near-singular joint Hessian with no deflatable gauge/decoder-null \
                 direction (max pivot {max_pivot:.3e})"
            ),
        };

        let full_len = cache.delta_t_len() + cache.k;
        let mut raw_gauges = self
            .joint_chart_gauge_basis_for_arrow_layout(
                &cache.row_offsets,
                cache.k,
                "outer_gradient_arrow_solver chart gauges",
            )
            .map_err(OuterGradientError::internal)?;
        // #2253: everything pushed above comes from `dense_step_gauge_vectors`
        // — the closed-form CHART gauge orbit (circle/torus phase, and the
        // translation/scale orbits of the linear/euclidean/duchon/poincaré
        // patches).
        //
        // #2720 — READ THE SCOPE OF "EXACT" HERE CAREFULLY. This comment used
        // to call them "EXACT criterion symmetries … flat by construction",
        // and that sentence is what put the same orbit into the inner
        // CONVERGENCE quotient, where it certified non-stationary points at up
        // to 76 170x the KKT tolerance. They are exact symmetries of the
        // RECONSTRUCTION (measured `1e-16` relative) and NOT of the criterion:
        // the ARD prior on `t` and the smoothness prior on `β` are written on
        // the chart coordinates and move along the orbit — the dilation field
        // by `−7.82` on an objective of `165`
        // (`tests_gauge_posterior_flatness_2720`).
        //
        // What justifies deflating them HERE is a different property and a
        // weaker one: this block runs only after the conditioning gate has
        // already flagged a near-singular joint Hessian, and the orbit carries
        // NO data-fit CURVATURE (only the priors'), so it is a genuine
        // near-null direction OF THE OPERATOR BEING INVERTED. Deflation is then
        // a pseudo-inverse choice on an ill-conditioned solve, not a claim that
        // the criterion is flat. That distinction is the whole of #2720, and it
        // is written here because this is the other site the claim reached.
        //
        // Remember the boundary so the exact-gauge subspace can be deflated
        // UNCONDITIONALLY, keeping the deflation COUNT stable across the ρ-walk
        // (a borderline eigenvalue flickering across the Rayleigh floor
        // re-anchors ½log|H| and desyncs the fixed-ρ criterion gradient from
        // its value).
        let n_exact_raw = raw_gauges.len();
        // #1051/#1273: admit the penalty-aware decoder-β null directions as
        // additional deflation candidates. A rank-deficient decoder design
        // (e.g. a euclidean-1D line in a p=2 ambient: decoder column rank 1 of
        // 3) puts a genuine near-null direction of the joint Hessian in the β
        // block, OUTSIDE the closed-form chart gauge orbit. #1273: probing the
        // RAW unit-β basis `e_j` produced an INCOMPLETE candidate set — the
        // true flat direction is the penalised null of `G_k + λ_smooth·S_k`,
        // not an axis-aligned coordinate, so the outer gate rejected trial ρ
        // with a pivot ratio (5.3e-16 < 1e-12) that the inner gate (which
        // already uses `joint_decoder_beta_null_directions(λ_smooth)`) accepts. Use
        // the SAME penalty-aware null directions here, evaluated at the smooth
        // scale the Schur factor used, so the outer and inner gates agree.
        // These full (n·q + beta_dim)-length vectors drop into the same
        // Gram-Schmidt + Rayleigh + Faddeev-Popov path below; the Rayleigh
        // floor still keeps only genuinely flat (sub-floor) directions, so a
        // well-conditioned decoder is unaffected.
        for dir in self
            .joint_decoder_beta_null_directions(penalized_gram_scale)
            .map_err(OuterGradientError::internal)?
        {
            let mapped = self
                .dense_joint_vector_in_arrow_layout(
                    dir.view(),
                    &cache.row_offsets,
                    cache.k,
                    "outer_gradient_arrow_solver decoder-beta null",
                )
                .map_err(OuterGradientError::internal)?;
            raw_gauges.push(mapped);
        }
        // #1051/#1273: also admit the decoder COLUMN-SPAN null (an unrealised
        // ambient output channel of a rank-deficient decoder), which the
        // channel-free basis-null above structurally cannot represent. The
        // rank-1-decoder-line geometry (e.g. a 1-D euclidean line in p=2
        // ambient: decoder column rank 1 of 2) puts the joint Hessian's
        // sub-floor pivot entirely in one output channel; without this
        // candidate the outer gate had nothing to deflate it with and rejected
        // the trial ρ. The Rayleigh floor below still prunes any candidate that
        // is not genuinely flat against the cached Hessian.
        for dir in self
            .decoder_channel_null_directions()
            .map_err(OuterGradientError::internal)?
        {
            let mapped = self
                .dense_joint_vector_in_arrow_layout(
                    dir.view(),
                    &cache.row_offsets,
                    cache.k,
                    "outer_gradient_arrow_solver decoder-channel null",
                )
                .map_err(OuterGradientError::internal)?;
            raw_gauges.push(mapped);
        }
        if raw_gauges.is_empty() {
            return Err(non_identifiable_err);
        }

        let mut gauge_span: Vec<Array1<f64>> = Vec::new();
        // Exact chart gauges (raw indices `< n_exact_raw`) are processed first,
        // so their Gram-Schmidt survivors occupy the FRONT of `gauge_span`;
        // `exact_basis_count` records that contiguous prefix.
        let mut exact_basis_count = 0usize;
        // A candidate that lies in the span of the stored bases must come out of
        // modified Gram–Schmidt as rounding, and nothing more. Each projection forms
        // one length-`full_len` inner product and updates every entry with a product
        // and a subtraction, leaking at most `γ_{full_len+4}·‖g₀‖`; it also leaks what
        // the stored bases' own loss of orthogonality leaves behind, at most
        // `Σ_j ω_j·‖g₀‖`. After `k` projections a dependent residual therefore stays
        // inside `k·(γ_{full_len+4} + Σ_j ω_j)·‖g₀‖`, and a basis stored from a
        // residual `r` of a candidate `g₀` carries `ω = band·‖g₀‖/‖r‖`.
        let projection_growth = gam_linalg::roundoff::accumulation_growth(full_len + 4);
        let mut orthogonality_defect = 0.0_f64;
        for (raw_idx, mut gauge) in raw_gauges.into_iter().enumerate() {
            let initial_norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            for basis in &gauge_span {
                let coeff = gauge.dot(basis);
                for i in 0..gauge.len() {
                    gauge[i] -= coeff * basis[i];
                }
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            let band = gauge_span.len() as f64 * (projection_growth + orthogonality_defect);
            if !(norm_sq.is_finite() && norm_sq > band * band * initial_norm_sq) {
                continue;
            }
            orthogonality_defect += band * (initial_norm_sq / norm_sq).sqrt();
            let inv_norm = norm_sq.sqrt().recip();
            for value in gauge.iter_mut() {
                *value *= inv_norm;
            }
            if raw_idx < n_exact_raw {
                exact_basis_count += 1;
            }
            gauge_span.push(gauge);
        }
        if gauge_span.is_empty() {
            return Err(non_identifiable_err);
        }

        let span_rank = gauge_span.len();
        let mut h_span = Array2::<f64>::zeros((span_rank, span_rank));
        for col in 0..span_rank {
            let h_gauge = match apply_cached_arrow_hessian(
                cache,
                gauge_span[col].slice(s![..cache.delta_t_len()]),
                gauge_span[col].slice(s![cache.delta_t_len()..]),
            ) {
                Ok(value) => value,
                // #1451: a shape/dimension mismatch or non-finite intermediate
                // from the Hessian apply is an internal-invariant defect and MUST
                // propagate; a genuine numeric failure on a finite,
                // correctly-shaped input keeps the typed conditioning class.
                Err(err) => {
                    return Err(OuterGradientError::classify_arrow_solver_error(
                        &err,
                        conditioning_err.clone(),
                    ));
                }
            };
            let h_flat = flatten_arrow_parts(h_gauge.t.view(), h_gauge.beta.view());
            for row in 0..span_rank {
                h_span[[row, col]] = gauge_span[row].dot(&h_flat);
            }
        }
        for row in 0..span_rank {
            for col in 0..row {
                let sym = 0.5 * (h_span[[row, col]] + h_span[[col, row]]);
                h_span[[row, col]] = sym;
                h_span[[col, row]] = sym;
            }
        }
        // #1451: a non-finite entry in the projected gauge Hessian is an
        // internal-invariant defect (a NaN/Inf intermediate leaked into the
        // span), not a conditioning failure — it MUST propagate rather than be
        // masked behind a degraded descent. Guard finiteness BEFORE the eigh so a
        // genuine decomposition failure on a finite, correctly-shaped matrix keeps
        // the typed conditioning class.
        if !h_span.iter().all(|v| v.is_finite()) {
            return Err(OuterGradientError::internal(format!(
                "outer_gradient_arrow_solver: non-finite entry in projected gauge \
                 Hessian (h_span is {span_rank}x{span_rank})"
            )));
        }
        let (evals, evecs) = h_span
            .eigh(Side::Lower)
            .map_err(|_| conditioning_err.clone())?;
        let strict_gauge_floor = SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR * max_pivot;
        let mut orthonormal: Vec<Array1<f64>> = Vec::new();
        for eig_idx in 0..evals.len() {
            let rayleigh = evals[eig_idx];
            if !(rayleigh.is_finite() && rayleigh <= strict_gauge_floor) {
                continue;
            }
            let mut direction = Array1::<f64>::zeros(full_len);
            for basis_idx in 0..span_rank {
                let coeff = evecs[[basis_idx, eig_idx]];
                for row in 0..full_len {
                    direction[row] += coeff * gauge_span[basis_idx][row];
                }
            }
            // An orthonormal combination with a unit eigenvector column has unit norm,
            // so only an exactly zero or non-finite direction is refused.
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 0.0) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in direction.iter_mut() {
                *value *= inv_norm;
            }
            orthonormal.push(direction);
        }
        // #2253: deflate the EXACT chart-gauge subspace unconditionally. A
        // borderline gauge eigenvalue can flicker across `strict_gauge_floor`
        // as ρ moves; for the empirical decoder-null candidates that screen is
        // the point, but for the exact chart gauges (circle/torus phase orbit,
        // patch translation/scale) it changes the deflation COUNT by ±1 and
        // re-anchors ½log|H|, desyncing the fixed-ρ criterion gradient from the
        // value (the K=1 circle non-stationary stall). The exact-gauge subspace
        // is `gauge_span[0..exact_basis_count]` (reconstruction-flat by
        // construction, hence data-fit-curvature-free — NOT criterion-flat, see
        // the scope note at the candidate site above); add any
        // of its directions the floor loop dropped, orthogonalized against what
        // was already kept, so the deflation dimension is ρ-stable. When the
        // floor already kept a gauge, its residual here lies inside the band below
        // and it is not double-counted.
        //
        // The band is the span construction's modified Gram–Schmidt band, taken
        // against the kept directions. Each kept direction is a normalized
        // combination `G·v` of the span bases with a computed eigenvector column `v`,
        // so against another kept direction it is off orthogonality by at most
        // `‖GᵀG − I‖₂ ≤ span_rank·Σ_j ω_j` (the span's own defect), plus the
        // eigenvector columns' orthogonality (`O(span_rank·u)` for a Householder-based
        // symmetric eigensolver, counted as `γ_{span_rank}`), plus the combination's
        // formation, `γ_{span_rank}·√span_rank` per vector.
        let kept_defect = span_rank as f64 * orthogonality_defect
            + gam_linalg::roundoff::accumulation_growth(span_rank)
                * (1.0 + 2.0 * (span_rank as f64).sqrt());
        let mut kept_orthogonality_defect = orthonormal.len() as f64 * kept_defect;
        for exact_idx in 0..exact_basis_count {
            let mut direction = gauge_span[exact_idx].clone();
            let initial_norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            let band = orthonormal.len() as f64 * (projection_growth + kept_orthogonality_defect);
            for kept in &orthonormal {
                let coeff = direction.dot(kept);
                for row in 0..direction.len() {
                    direction[row] -= coeff * kept[row];
                }
            }
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > band * band * initial_norm_sq) {
                continue;
            }
            kept_orthogonality_defect += band * (initial_norm_sq / norm_sq).sqrt();
            let inv_norm = norm_sq.sqrt().recip();
            for value in direction.iter_mut() {
                *value *= inv_norm;
            }
            orthonormal.push(direction);
        }
        if orthonormal.is_empty() {
            // The joint factor is ill-conditioned, but no direction in the
            // analytically known gauge/decoder-null span is actually flat at the
            // rank-revealing Rayleigh threshold. The unreliable direction lies
            // outside the quotient we can justify, so refuse the derivative
            // instead of projecting an arbitrary least-curvature candidate.
            return Err(non_identifiable_err);
        }

        // Quotient-geometry gauge fixing: add stiffness only along the closed-form
        // gauge orbit (Faddeev-Popov style). Components orthogonal to that orbit
        // are identical to the original inverse solve, while gauge components are
        // bounded at the Hessian scale `max_pivot`.
        // #1451: a shape/length mismatch or non-finite stiffness/intermediate in
        // the deflated-solver assembly is an internal-invariant defect and MUST
        // propagate; a genuine near-singular gauge Woodbury/back-solve keeps the
        // typed conditioning class.
        DeflatedArrowSolver::from_orthonormal_gauges(cache, orthonormal, max_pivot)
            .map_err(|err| OuterGradientError::classify_arrow_solver_error(&err, conditioning_err))
    }

    pub(crate) fn outer_gradient_conditioning_error(
        cache: &ArrowFactorCache,
    ) -> Result<(), OuterGradientError> {
        let pivot = arrow_factor_min_pivot(cache);
        let Some(min_pivot) = pivot.min_pivot else {
            return Err(OuterGradientError::NonIdentifiable {
                reason: "joint Hessian numerically singular (no cached Cholesky pivots)"
                    .to_string(),
            });
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(OuterGradientError::NonIdentifiable {
                reason: "joint Hessian numerically singular (no cached Cholesky pivot scale)"
                    .to_string(),
            });
        };
        let ratio = min_pivot / max_pivot;
        if min_pivot.is_finite()
            && max_pivot.is_finite()
            && max_pivot > 0.0
            && ratio.is_finite()
            && ratio >= SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR
        {
            return Ok(());
        }
        Err(OuterGradientError::NonIdentifiable {
            reason: format!(
                "joint Hessian numerically singular (min/max pivot ratio {ratio:.3e} < floor {floor:.3e}; min pivot {min_pivot:.3e}, max pivot {max_pivot:.3e})",
                floor = SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR,
            ),
        })
    }

    pub(crate) fn assignment_log_strength_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<f64, String> {
        self.assignment.validate_rho_domain(rho)?;
        let k_atoms = self.k_atoms();
        // #1038/#1419 softmax: the assembled majorizer is the DIAGONAL
        // `scale·D`, `D = diag(Σ_j|H_kj|)` of the entropy block, carrying the row's
        // design weight (#991). It scales linearly with `λ_sparse = exp(ρ)`, so
        // `∂B/∂ρ = scale·D` on the free logit slots of the reduced K−1 chart, and
        // it takes the one Daleckii–Krein deflation correction below, like every
        // other family (#2916). The kept-subspace diagonal it used to contract
        // equals that correction only when no deflated direction couples to the
        // border, since `vᵢᵀ (H⁻¹)_tt vᵢ = 1 + (H_βt vᵢ)ᵀ S⁻¹ (H_βt vᵢ)`.
        let mut hdiag = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                if k_atoms <= 1 {
                    return Ok(0.0);
                }
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
                let penalty = gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                    k_atoms,
                    temperature,
                );
                let row_loss_w = self.row_loss_weights.as_deref();
                let mut weighted = Array1::<f64>::zeros(self.n_obs() * k_atoms);
                for row in 0..self.n_obs() {
                    let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                    let row_logits: Vec<f64> = (0..k_atoms)
                        .map(|k| self.assignment.logits[[row, k]])
                        .collect();
                    let d = penalty.psd_majorizer_abs_row_sums(&row_logits, scale);
                    for atom in 0..k_atoms.min(d.len()) {
                        weighted[row * k_atoms + atom] = w_row * d[atom];
                    }
                }
                weighted
            }
            _ => crate::assignment::assignment_prior_log_strength_hdiag_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?,
        };
        if hdiag.is_empty() {
            return Ok(0.0);
        }
        // RAW selected-inverse diagonal: the per-row diagonal contraction uses the
        // DEFLATED inverse; the full kept-subspace + β-Schur/rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per row afterwards
        // (`deflation_block_correction`), exactly as the data trace does. The
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
        let assignment_dim = self.assignment.assignment_coord_dim();
        let total_t = cache.delta_t_len();
        // #932 FRONT C: row-local Takahashi selected inverse on the plain arrow
        // for the per-row deflation correction below (the diagonal trace already
        // uses the cheap `latent_inverse_diagonal`); gauge-deflated systems fall
        // back to the per-row full-system `solve` loop.
        let fast_selected = solver.plain_selected_inverse_available();
        let selected_beta_inv = if fast_selected && cache.k > 0 {
            solver
                .beta_inv()
                .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        // `hdiag` differentiates the prior along whatever `log_lambda_sparse` carries:
        // the concentration when it is effectively learnable. A fixed concentration puts
        // no coordinate into the prior (#2933 F45), so `hdiag` is zero and there is nothing
        // to majorize.
        let ordered_channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?;
        // The integrated marginal's mass-Hessian coefficient is strictly
        // negative, so its cross-row rank-one block has the zero PSD Loewner
        // majorizer. Retain only the positive part of the row-local
        // concrete-Jacobian term, matching assembly exactly.
        if let Some(ch) = ordered_channels.as_ref()
            && self.assignment.effective_alpha_is_learnable()
        {
            for row in 0..self.n_obs() {
                for atom in 0..k_atoms {
                    let slot = row * k_atoms + atom;
                    hdiag[slot] =
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
                            ch, row, k_atoms, atom, hdiag[slot],
                        );
                }
            }
        }
        let mut trace = 0.0_f64;
        // Hoisted RHS scratch for the gauge-deflated per-row solve fallback:
        // single-entry set/clear instead of a per-column total_t-sized zeroing.
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        for row in 0..self.n_obs() {
            let row_base = cache.row_offsets[row];
            let assignment_base = row * k_atoms;
            let q = cache.row_dims[row];
            // Per-row diagonal `(∂H/∂ρ)_tt` for the deflation correction: the
            // assignment prior curves only the logit/assignment slots (coordinate
            // slots are zero; ARD handles those).
            let mut d_diag = Array1::<f64>::zeros(q);
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                        let d_slot = hdiag[assignment_base + atom];
                        trace += inv_diag[row_base + pos] * d_slot;
                        if pos < q {
                            d_diag[pos] = d_slot;
                        }
                    }
                }
                None => {
                    for free_idx in 0..assignment_dim {
                        let d_slot = hdiag[assignment_base + free_idx];
                        trace += inv_diag[row_base + free_idx] * d_slot;
                        if free_idx < q {
                            d_diag[free_idx] = d_slot;
                        }
                    }
                }
            }
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            if Self::row_deflation_is_live(dirs, spectrum) {
                let inv_vv = if fast_selected {
                    let (inv_vv, _inv_vbeta) = solver
                        .selected_inverse_row_blocks(row, &selected_beta_inv)
                        .map_err(|err| {
                            format!(
                                "assignment_log_strength_hessian_trace: selected inverse: {err}"
                            )
                        })?;
                    inv_vv
                } else {
                    let mut inv_vv = Array2::<f64>::zeros((q, q));
                    for col in 0..q {
                        rhs_t_scratch[row_base + col] = 1.0;
                        let solved = solver
                            .solve(rhs_t_scratch.view(), rhs_beta_zero.view())
                            .map_err(|err| {
                                format!(
                                    "assignment_log_strength_hessian_trace: selected inverse: {err}"
                                )
                            })?;
                        rhs_t_scratch[row_base + col] = 0.0;
                        for r in 0..q {
                            inv_vv[[r, col]] = solved.t[row_base + r];
                        }
                    }
                    inv_vv
                };
                let mut d_mat = Array2::<f64>::zeros((q, q));
                for s in 0..q {
                    d_mat[[s, s]] = d_diag[s];
                }
                trace -= Self::deflation_block_correction(&inv_vv, &d_mat, dirs, spectrum);
            }
        }
        Ok(0.5 * trace)
    }
}
