//! Exact directional extension of the timepoint quantities.
//!
//! Given the base full timepoint evaluation, contracts the higher-order cell
//! kernels with a single direction `dir` to produce the directional
//! derivatives (η_uv_dir, χ_uv_dir, D_u_dir, D_uv_dir).

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Compute directional extensions of a timepoint's exact quantities.
    /// Given the base `SurvivalFlexTimepointExact`, returns the directional
    /// derivatives eta_uv_dir, chi_uv_dir, d_u_dir, d_uv_dir contracted
    /// with `dir`.
    pub(crate) fn compute_survival_timepoint_directional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir: &Array1<f64>,
        need_d_uv_dir: bool,
    ) -> Result<SurvivalFlexTimepointDirectionalExact, String> {
        let p = primary.total;
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        // ── Pre-pass: the intercept directional motion a_dir ───────────────
        // The fixed-domain moment reductions below produce the PARTIAL
        // θ-derivatives of `f_a/f_aa/f_au/f_uv` (the calibration `F`'s a-jets),
        // holding the implicit calibration intercept `a` fixed. But `a = a(θ)`
        // on the calibration manifold, so the TOTAL directional derivative each
        // `f_*_dir` must carry also includes the intercept chain
        // `∂(f_*)/∂a · a_dir`. We fold that chain in by differentiating along
        // the TOTAL cell-index velocity `∂c/∂dir_total = ∂c/∂(direct θ)·dir
        // + ∂c/∂a · a_dir`, i.e. by augmenting every dir cell-coefficient with
        // `a_dir · (its next a-derivative)`. `a_dir` needs only the first-order
        // jets `a_u = -f_u/f_a`, so compute it in a cheap pre-pass here (the
        // q-marginal RHS self term `+φ(q)` on `f_u[q_index]` is part of it).
        let a_dir = {
            let mut f_a_pre = 0.0;
            let mut f_dir_pre = 0.0;
            for cell_entry in &cached.cells {
                let neg_cell = cell_entry.neg_cell;
                let state = &cell_entry.state;
                let fixed = &cell_entry.fixed;
                let neg_dc_da: [f64; 4] = fixed.dc_da.map(|v| -v);
                f_a_pre += exact_kernel::cell_first_derivative_from_moments(
                    &neg_dc_da,
                    &state.moments,
                )?;
                let mut neg_coeff_dir = [0.0; 4];
                for c in 0..p {
                    if dir[c] == 0.0 {
                        continue;
                    }
                    for k in 0..4 {
                        neg_coeff_dir[k] -= fixed.coeff_u[c][k] * dir[c];
                    }
                }
                let _ = neg_cell;
                f_dir_pre += exact_kernel::cell_first_derivative_from_moments(
                    &neg_coeff_dir,
                    &state.moments,
                )?;
            }
            // q-marginal RHS self term: f_u[q_index] += φ(q), so its dir
            // contraction adds dir[q_index]·φ(q) to f_dir.
            let phi_q = crate::probability::normal_pdf(q);
            if q_index < p {
                f_dir_pre += dir[q_index] * phi_q;
            }
            // a_u = -f_u/f_a ⇒ a_dir = a_u·dir = -(f_dir)/f_a.
            -f_dir_pre / f_a_pre
        };

        struct DirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Vec<f64>,
            f_au: Vec<f64>,
            f_uv: Vec<f64>,
            f_a_dir: f64,
            f_aa_dir: f64,
            f_au_dir: Vec<f64>,
            f_uv_dir: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(
                |cell_entry| -> Result<DirectionalTimepointCellAccum, String> {
                    let neg_cell = cell_entry.neg_cell;
                    let state = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let neg_dc_da: [f64; 4] = fixed.dc_da.map(|v| -v);
                    let neg_dc_daa: [f64; 4] = fixed.dc_daa.map(|v| -v);
                    let neg_dc_daaa: [f64; 4] = fixed.dc_daaa.map(|v| -v);

                    let f_a = exact_kernel::cell_first_derivative_from_moments(
                        &neg_dc_da,
                        &state.moments,
                    )?;
                    let f_aa = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_dc_daa,
                        &state.moments,
                    )?;

                    // TOTAL directional cell-coefficient jets along `dir`:
                    // `∂c/∂dir_total = ∂c/∂(direct θ)·dir + a_dir·∂c/∂a`.
                    // The trailing `a_dir·(next a-derivative)` is the intercept
                    // chain that makes each `f_*_dir` the TOTAL D_dir of `f_*`
                    // (not just the partial), so the `a_uv_dir` chain rule below
                    // is exact (gam#932/#979).
                    let mut neg_coeff_dir = [0.0; 4];
                    let mut neg_coeff_a_dir = [0.0; 4];
                    let mut neg_coeff_aa_dir = [0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            neg_coeff_dir[k] -= fixed.coeff_u[c][k] * dir[c];
                            neg_coeff_a_dir[k] -= fixed.coeff_au[c][k] * dir[c];
                            neg_coeff_aa_dir[k] -= fixed.coeff_aau[c][k] * dir[c];
                        }
                    }
                    for k in 0..4 {
                        neg_coeff_dir[k] += a_dir * neg_dc_da[k];
                        neg_coeff_a_dir[k] += a_dir * neg_dc_daa[k];
                        neg_coeff_aa_dir[k] += a_dir * neg_dc_daaa[k];
                    }

                    let f_a_dir = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_coeff_a_dir,
                        &state.moments,
                    )?;
                    let mut f_aa_dir = exact_kernel::cell_third_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_dc_daa,
                        &neg_coeff_a_dir,
                        &neg_coeff_a_dir,
                        &neg_coeff_aa_dir,
                        &state.moments,
                    )?;

                    let mut f_u = vec![0.0; p];
                    let mut f_au = vec![0.0; p];
                    let mut f_uv = vec![0.0; p * p];
                    let mut f_au_dir = vec![0.0; p];
                    let mut f_uv_dir = vec![0.0; p * p];
                    for u in 0..p {
                        let neg_coeff_u = fixed.coeff_u[u].map(|v| -v);
                        let neg_coeff_au = fixed.coeff_au[u].map(|v| -v);

                        f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                            &neg_coeff_u,
                            &state.moments,
                        )?;
                        f_au[u] = exact_kernel::cell_second_derivative_from_moments(
                            neg_cell,
                            &neg_dc_da,
                            &neg_coeff_u,
                            &neg_coeff_au,
                            &state.moments,
                        )?;

                        let mut neg_coeff_u_dir = [0.0; 4];
                        let mut neg_coeff_au_dir = [0.0; 4];
                        for c in 0..p {
                            if dir[c] == 0.0 {
                                continue;
                            }
                            let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                            let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                            for k in 0..4 {
                                neg_coeff_u_dir[k] -= sc[k] * dir[c];
                                neg_coeff_au_dir[k] -= sca[k] * dir[c];
                            }
                        }
                        // Intercept chain for the (u, dir) and (a, u, dir)
                        // cross coefficients: `∂²c/∂u∂dir_total` and
                        // `∂³c/∂a∂u∂dir_total` pick up `a_dir·∂²c/∂u∂a` and
                        // `a_dir·∂³c/∂a²∂u` respectively (gam#932/#979).
                        for k in 0..4 {
                            neg_coeff_u_dir[k] += a_dir * neg_coeff_au[k];
                            neg_coeff_au_dir[k] -= a_dir * fixed.coeff_aau[u][k];
                        }

                        f_au_dir[u] = exact_kernel::cell_third_derivative_from_moments(
                            neg_cell,
                            &neg_dc_da,
                            &neg_coeff_u,
                            &neg_coeff_dir,
                            &neg_coeff_au,
                            &neg_coeff_a_dir,
                            &neg_coeff_u_dir,
                            &neg_coeff_au_dir,
                            &state.moments,
                        )?;
                    }

                    for u in 0..p {
                        for v in u..p {
                            let neg_coeff_u = fixed.coeff_u[u].map(|val| -val);
                            let neg_coeff_v = fixed.coeff_u[v].map(|val| -val);
                            let sc_uv = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, v);
                            let neg_sc_uv = sc_uv.map(|val| -val);

                            let base_val = exact_kernel::cell_second_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_sc_uv,
                                &state.moments,
                            )?;
                            f_uv[u * p + v] = base_val;
                            f_uv[v * p + u] = base_val;

                            let mut neg_coeff_u_dir = [0.0; 4];
                            let mut neg_coeff_v_dir = [0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_uc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_vc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                for k in 0..4 {
                                    neg_coeff_u_dir[k] -= sc_uc[k] * dir[c];
                                    neg_coeff_v_dir[k] -= sc_vc[k] * dir[c];
                                }
                            }

                            // Third cell-coefficient cross `∂³c/∂u∂v∂dir`. It is
                            // NOT identically zero: when two of the three axes
                            // are the slope `g` (= the `b` argument of the cubic
                            // cell coefficient), it carries the `∂²/∂b²`
                            // curvature of a basis coefficient (`coeff_bbu`).
                            // Dropping it lost the `D_g f_uv[g, ·]` curvature
                            // term and corrupted the Block-10 third contraction
                            // (gam#1195).
                            let mut neg_coeff_uv_dir = [0.0; 4];
                            self.add_cell_pair_third_coeff_dir(
                                primary,
                                &fixed.coeff_bbu,
                                u,
                                v,
                                dir,
                                -1.0,
                                &mut neg_coeff_uv_dir,
                            );
                            // Intercept chain for the (u, dir), (v, dir) and
                            // (u, v, dir) cross coefficients: each picks up
                            // `a_dir·(its a-derivative)` so f_uv_dir is the
                            // TOTAL D_dir(f_uv) (gam#932/#979). `∂³c/∂u∂v∂a` is
                            // the `coeff_abu` a-cross (nonzero only when u or v
                            // is the slope g).
                            let sc_uva = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, v);
                            for k in 0..4 {
                                neg_coeff_u_dir[k] -= a_dir * fixed.coeff_au[u][k];
                                neg_coeff_v_dir[k] -= a_dir * fixed.coeff_au[v][k];
                                neg_coeff_uv_dir[k] -= a_dir * sc_uva[k];
                            }

                            let dir_val = exact_kernel::cell_third_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_coeff_dir,
                                &neg_sc_uv,
                                &neg_coeff_u_dir,
                                &neg_coeff_v_dir,
                                &neg_coeff_uv_dir,
                                &state.moments,
                            )?;
                            f_uv_dir[u * p + v] = dir_val;
                            f_uv_dir[v * p + u] = dir_val;
                        }
                    }

                    Ok(DirectionalTimepointCellAccum {
                        f_a,
                        f_aa,
                        f_u,
                        f_au,
                        f_uv,
                        f_a_dir,
                        f_aa_dir,
                        f_au_dir,
                        f_uv_dir,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_dir = 0.0;
        let mut f_aa_dir = 0.0;
        let mut f_au_dir = Array1::<f64>::zeros(p);
        let mut f_uv_dir = Array2::<f64>::zeros((p, p));
        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_dir += acc.f_a_dir;
            f_aa_dir += acc.f_aa_dir;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_dir[u] += acc.f_au_dir[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[u * p + v];
                    f_uv_dir[[u, v]] += acc.f_uv_dir[u * p + v];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        // q-marginal calibration RHS self-coupling. The base second derivative
        // is `f_uv[[q,q]] = -q·φ(q)`; its exact directional derivative along
        // `dir` is `dir[q]·∂_q(-q·φ(q)) = dir[q]·(q²-1)·φ(q)`. The previous
        // `(1 - q²)` had this third q-self term sign-flipped relative to its own
        // base, corrupting the (q,·) blocks of the contracted third tower
        // (gam#932/#979).
        f_uv_dir[[q_index, q_index]] += dir[q_index] * (q * q - 1.0) * phi_q;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_dir = a_u.dot(dir);
        let a_u_dir = a_uv.dot(dir);
        // Moving-domain (Leibniz) boundary correction for link-knot crossings.
        // The fixed-domain moment reductions above differentiate only the
        // integrand on each current cell. A link edge is
        // `z_c=(tau-a)/b`, so along a primary direction its velocity is the
        // TOTAL calibration motion
        //
        //     z_c' = -(a_dir + z_c * dir_g) / b.
        //
        // The previous correction used only the direct slope term
        // `-(z_c/b) * dir_g`, which left every channel depending on the
        // intercept IFT motion with a stale domain derivative.
        let dir_g = if primary.g < p { dir[primary.g] } else { 0.0 };
        if b != 0.0 && (a_dir != 0.0 || dir_g != 0.0) {
            for cell_entry in &cached.cells {
                let neg_cell = cell_entry.neg_cell;
                let fixed = &cell_entry.fixed;
                let part = &cell_entry.partition_cell;
                let edge_velocity =
                    |edge: crate::families::cubic_cell_kernel::PartitionEdge, z: f64| -> f64 {
                        match edge {
                            crate::families::cubic_cell_kernel::PartitionEdge::Crossing {
                                ..
                            } => -(a_dir + z * dir_g) / b,
                            crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
                        }
                    };
                let v_right = edge_velocity(part.right_edge, neg_cell.right);
                let v_left = edge_velocity(part.left_edge, neg_cell.left);
                if v_right == 0.0 && v_left == 0.0 {
                    continue;
                }

                let boundary = |neg_r: &[f64], neg_s: &[f64], neg_rs: &[f64]| -> f64 {
                    let fr = if v_right != 0.0 {
                        v_right
                            * crate::families::cubic_cell_kernel::cell_second_derivative_boundary_integrand(
                                neg_cell,
                                neg_r,
                                neg_s,
                                neg_rs,
                                neg_cell.right,
                            )
                    } else {
                        0.0
                    };
                    let fl = if v_left != 0.0 {
                        v_left
                            * crate::families::cubic_cell_kernel::cell_second_derivative_boundary_integrand(
                                neg_cell,
                                neg_r,
                                neg_s,
                                neg_rs,
                                neg_cell.left,
                            )
                    } else {
                        0.0
                    };
                    fr - fl
                };

                let neg_dc_da = fixed.dc_da.map(|val| -val);
                let neg_dc_daa = fixed.dc_daa.map(|val| -val);
                f_aa_dir += boundary(&neg_dc_da, &neg_dc_da, &neg_dc_daa);
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|val| -val);
                    let neg_coeff_au = fixed.coeff_au[u].map(|val| -val);
                    f_au_dir[u] += boundary(&neg_dc_da, &neg_coeff_u, &neg_coeff_au);
                    for v in u..p {
                        let neg_coeff_v = fixed.coeff_u[v].map(|val| -val);
                        let neg_sc_uv = self
                            .cell_pair_second_coeff(primary, &fixed.coeff_bu, u, v)
                            .map(|val| -val);
                        let bval = boundary(&neg_coeff_u, &neg_coeff_v, &neg_sc_uv);
                        f_uv_dir[[u, v]] += bval;
                        if u != v {
                            f_uv_dir[[v, u]] += bval;
                        }
                    }
                }
            }
        }
        let mut a_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n_dir = f_uv_dir[[u, v]]
                    + f_au_dir[u] * a_u[v]
                    + f_au[u] * a_u_dir[v]
                    + f_au_dir[v] * a_u[u]
                    + f_au[v] * a_u_dir[u]
                    + f_aa_dir * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                let val = -(n_dir + f_a_dir * a_uv[[u, v]]) * inv_f_a;
                a_uv_dir[[u, v]] = val;
                a_uv_dir[[v, u]] = val;
            }
        }

        // Observed-point quantities and their dir-extensions
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let chi_val = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut tau = Array1::<f64>::zeros(p);
        let mut g_au_fixed = vec![[0.0; 4]; p];
        let mut tau_a = Array1::<f64>::zeros(p);
        let mut g_bu_fixed = vec![[0.0; 4]; p];
        let mut g_aau_fixed = vec![[0.0; 4]; p];
        let mut g_abu_fixed = vec![[0.0; 4]; p];
        let mut g_bbu_fixed = vec![[0.0; 4]; p];
        let mut g_aaau_fixed = vec![[0.0; 4]; p];
        let mut g_aabu_fixed = vec![[0.0; 4]; p];
        let mut g_abbu_fixed = vec![[0.0; 4]; p];
        let mut g_bbbu_fixed = vec![[0.0; 4]; p];

        g_u_fixed[primary.g] = obs.dc_db;
        g_au_fixed[primary.g] = obs.dc_dab;
        g_bu_fixed[primary.g] = obs.dc_dbb;
        g_aau_fixed[primary.g] = obs.dc_daab;
        g_abu_fixed[primary.g] = obs.dc_dabb;
        g_bbu_fixed[primary.g] = obs.dc_dbbb;
        g_aaau_fixed[primary.g] = [0.0; 4];
        g_aabu_fixed[primary.g] = [0.0; 4];
        g_abbu_fixed[primary.g] = [0.0; 4];
        g_bbbu_fixed[primary.g] = [0.0; 4];

        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        tau_a[primary.g] = eval_coeff4_at(&obs.dc_daab, z_obs);
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                g_u_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                g_au_fixed[idx] = scale_coeff4(dc_aw, scale);
                g_bu_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b).1,
                    scale,
                );
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
                tau_a[idx] = eval_coeff4_at(&scale_coeff4(dc_aaw, scale), z_obs);
            }
        }

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                    scale,
                );
                g_bu_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                    scale,
                );
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let chi_dir = eta_aa * a_dir + tau.dot(dir);
        let eta_aa_dir = eta_aaa * a_dir
            + eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_GW),
                z_obs,
            );
        let eta_aaa_dir = eval_coeff4_at(
            &g_jet.directional_family(g_jet.aaa_first, dir, COEFF_SUPPORT_GW),
            z_obs,
        );

        let mut tau_dir = Array1::<f64>::zeros(p);
        let mut tau_a_dir = Array1::<f64>::zeros(p);
        for u in 0..p {
            let fixed_tau_dir =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_GW);
            tau_dir[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_dir, z_obs);

            let fixed_tau_a_dir =
                g_jet.param_directional_from_b_family(g_jet.aab_first, u, dir, COEFF_SUPPORT_GW);
            tau_a_dir[u] = eval_coeff4_at(&g_jet.aaa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_a_dir, z_obs);
        }

        let mut eta_uv_dir = Array2::<f64>::zeros((p, p));
        let mut chi_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let r_uv_dir = self.observed_fixed_eta_second_partial_dir(
                    primary, &obs, u, v, z_obs, u_obs, a, b, a_dir, dir, beta_w,
                )?;
                let chi_uv_fixed_dir = self.observed_fixed_chi_second_partial_dir(
                    primary, u, v, z_obs, u_obs, a_dir, dir,
                )?;

                let eta_val = chi_dir * a_uv[[u, v]]
                    + chi_val * a_uv_dir[[u, v]]
                    + eta_aa_dir * a_u[u] * a_u[v]
                    + eta_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_dir[u] * a_u[v]
                    + tau[u] * a_u_dir[v]
                    + tau_dir[v] * a_u[u]
                    + tau[v] * a_u_dir[u]
                    + r_uv_dir;
                eta_uv_dir[[u, v]] = eta_val;
                eta_uv_dir[[v, u]] = eta_val;

                let chi_v = eta_aa_dir * a_uv[[u, v]]
                    + eta_aa * a_uv_dir[[u, v]]
                    + eta_aaa_dir * a_u[u] * a_u[v]
                    + eta_aaa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_a_dir[u] * a_u[v]
                    + tau_a[u] * a_u_dir[v]
                    + tau_a_dir[v] * a_u[u]
                    + tau_a[v] * a_u_dir[u]
                    + chi_uv_fixed_dir;
                chi_uv_dir[[u, v]] = chi_v;
                chi_uv_dir[[v, u]] = chi_v;
            }
        }


        // D_u_dir: directional derivative of the density normalization first derivative.
        #[cfg(test)]
        let dbg_w0: usize = primary.w.as_ref().map(|w| w.start).unwrap_or(usize::MAX);
        let d_u_dir_cell_accums = cached
            .cells
            .iter()
            .map(|cell_entry| -> Result<(Array1<f64>, [f64; 4]), String> {
                let mut d_u_dir = Array1::<f64>::zeros(p);
                #[cfg(test)]
                let mut dbg_polys = [0.0_f64; 4];
                #[cfg(not(test))]
                let dbg_polys = [0.0_f64; 4];
                let cell = cell_entry.partition_cell.cell;
                let state_ref = &cell_entry.state;
                let fixed = &cell_entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let eta_aa_poly = fixed.dc_daa.to_vec();

                let mut eta_u_poly = vec![PolyVec::new(); p];
                let mut chi_u_poly = vec![PolyVec::new(); p];
                for u in 0..p {
                    eta_u_poly[u] =
                        poly_add(&poly_scale(&chi_poly, a_u[u]), fixed.coeff_u[u].as_ref());
                    chi_u_poly[u] = poly_add(
                        &poly_scale(&eta_aa_poly, a_u[u]),
                        fixed.coeff_au[u].as_ref(),
                    );
                }

                let mut coeff_dir_poly = vec![0.0; 4];
                let mut coeff_a_dir_poly = vec![0.0; 4];
                #[cfg(test)]
                let mut coeff_aa_dir_poly_dbg = vec![0.0; 4];
                for c in 0..p {
                    if dir[c] == 0.0 {
                        continue;
                    }
                    for k in 0..4 {
                        coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                        coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                        #[cfg(test)]
                        {
                            coeff_aa_dir_poly_dbg[k] += fixed.coeff_aau[c][k] * dir[c];
                        }
                    }
                }
                let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);

                for u in 0..p {
                    let mut eta_u_dir_fixed = vec![0.0; 4];
                    let mut chi_u_dir_fixed = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                        for k in 0..4 {
                            eta_u_dir_fixed[k] += sc[k] * dir[c];
                            chi_u_dir_fixed[k] += sca[k] * dir[c];
                        }
                    }
                    let eta_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&chi_poly, a_u_dir[u]),
                            &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                        ),
                        &eta_u_dir_fixed,
                    );
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();
                    let chi_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&eta_aa_poly, a_u_dir[u]),
                            &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                        ),
                        &chi_u_dir_fixed,
                    );

                    // D_u integrand: chi_u - chi * eta * eta_u
                    let integrand_base = poly_sub(
                        &chi_u_poly[u],
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly[u]),
                    );
                    // Polynomial derivative of integrand w.r.t. dir
                    let integrand_dir = poly_sub(
                        &poly_sub(
                            &poly_sub(
                                &chi_u_dir_poly,
                                &poly_mul(&poly_mul(&coeff_a_dir_poly, &eta_poly), &eta_u_poly[u]),
                            ),
                            &poly_mul(&poly_mul(&chi_poly, &eta_dir_poly), &eta_u_poly[u]),
                        ),
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_dir_poly),
                    );
                    // Moment-weighting correction: -eta*eta_dir * integrand_base
                    let full_integrand = poly_sub(
                        &integrand_dir,
                        &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &integrand_base),
                    );

                    #[cfg(test)]
                    if u == dbg_w0 {
                        let ig = |poly: &[f64]| -> f64 {
                            exact_kernel::cell_polynomial_integral_from_moments(
                                poly,
                                &state_ref.moments,
                                "dbg",
                            )
                            .unwrap_or(f64::NAN)
                        };
                        // TOTAL D_dir of chi_u / eta_u (with the β_w cross +
                        // intercept a-chain restored), then the moment-measure
                        // correction so each equals D_dir(∫poly·m).
                        let chi_dir_poly =
                            poly_add(&poly_scale(&eta_aa_poly, a_dir), &coeff_a_dir_poly);
                        let eta_aa_dir_poly =
                            poly_add(&poly_scale(&eta_aaa_poly, a_dir), &coeff_aa_dir_poly_dbg);
                        let eta_u_dir_total = poly_add(
                            &poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[u]),
                                    &poly_scale(&chi_dir_poly, a_u[u]),
                                ),
                                &eta_u_dir_fixed,
                            ),
                            &poly_scale(fixed.coeff_au[u].as_ref(), a_dir),
                        );
                        let chi_u_dir_total = poly_add(
                            &poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aa_dir_poly, a_u[u]),
                                ),
                                &chi_u_dir_fixed,
                            ),
                            &poly_scale(fixed.coeff_aau[u].as_ref(), a_dir),
                        );
                        let mw = poly_mul(&eta_poly, &eta_dir_poly);
                        // base ∫chi_u·m, ∫eta_u·m
                        dbg_polys[0] += ig(&chi_u_poly[u]);
                        dbg_polys[1] += ig(&eta_u_poly[u]);
                        // D_dir(∫chi_u·m) = ∫chi_u_dir_total·m - ∫mw·chi_u·m
                        dbg_polys[2] +=
                            ig(&chi_u_dir_total) - ig(&poly_mul(&mw, &chi_u_poly[u]));
                        dbg_polys[3] +=
                            ig(&eta_u_dir_total) - ig(&poly_mul(&mw, &eta_u_poly[u]));
                    }
                    d_u_dir[u] += exact_kernel::cell_polynomial_integral_from_moments(
                        &full_integrand,
                        &state_ref.moments,
                        "survival D_t first derivative directional",
                    )?;
                }
                Ok((d_u_dir, dbg_polys))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u_dir = Array1::<f64>::zeros(p);
        #[cfg(test)]
        let mut dbg_d_u_sum = [0.0_f64; 4];
        for (cell_d_u_dir, _cell_dbg) in &d_u_dir_cell_accums {
            for u in 0..p {
                d_u_dir[u] += cell_d_u_dir[u];
            }
            #[cfg(test)]
            for k in 0..4 {
                dbg_d_u_sum[k] += _cell_dbg[k];
            }
        }
        #[cfg(test)]
        let debug_d_u_polys = Some((
            [dbg_d_u_sum[0], dbg_d_u_sum[1]],
            [dbg_d_u_sum[2], dbg_d_u_sum[3]],
        ));

        // D_uv_dir
        let mut d_uv_dir = Array2::<f64>::zeros((p, p));
        #[cfg(test)]
        let mut debug_d_uv_terms: Option<([f64; 5], [f64; 5])> = None;
        if need_d_uv_dir {
            // #932 debug probe block: (w0, w0) deviation diagonal.
            #[cfg(test)]
            let dbg_target: Option<(usize, usize)> =
                primary.w.as_ref().map(|w| (w.start, w.start));
            let d_uv_dir_cell_accums = cached
                .cells
                .iter()
                .map(|cell_entry| -> Result<(Array2<f64>, [f64; 10]), String> {
                    let mut d_uv_dir = Array2::<f64>::zeros((p, p));
                    #[cfg(test)]
                    let mut dbg_terms = [0.0_f64; 10];
                    #[cfg(not(test))]
                    let dbg_terms = [0.0_f64; 10];
                    let cell = cell_entry.partition_cell.cell;
                    let state_ref = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                    let chi_poly = fixed.dc_da.to_vec();
                    let eta_aa_poly = fixed.dc_daa.to_vec();
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();

                    let mut eta_u_poly = vec![PolyVec::new(); p];
                    let mut chi_u_poly = vec![PolyVec::new(); p];
                    for u in 0..p {
                        eta_u_poly[u] =
                            poly_add(&poly_scale(&chi_poly, a_u[u]), fixed.coeff_u[u].as_ref());
                        chi_u_poly[u] = poly_add(
                            &poly_scale(&eta_aa_poly, a_u[u]),
                            fixed.coeff_au[u].as_ref(),
                        );
                    }
                    let mut coeff_dir_poly = vec![0.0; 4];
                    let mut coeff_a_dir_poly = vec![0.0; 4];
                    let mut coeff_aa_dir_poly = vec![0.0; 4];
                    let mut coeff_aaa_dir_poly = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                            coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                            coeff_aa_dir_poly[k] += fixed.coeff_aau[c][k] * dir[c];
                            coeff_aaa_dir_poly[k] += fixed.coeff_aaau[c][k] * dir[c];
                        }
                    }
                    let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);
                    let chi_dir_poly =
                        poly_add(&poly_scale(&eta_aa_poly, a_dir), &coeff_a_dir_poly);
                    // D_dir(eta_aa) = eta_aaa·a_dir + ∂³c/∂a²∂dir(direct).
                    let eta_aa_dir_poly =
                        poly_add(&poly_scale(&eta_aaa_poly, a_dir), &coeff_aa_dir_poly);
                    // D_dir(eta_aaa) = dc_daaaa·a_dir + ∂⁴c/∂a³∂dir(direct). The cell
                    // coefficient is cubic in `a` (the link enters via linkdev(a+b·z),
                    // a cubic), so dc_daaaa ≡ 0 and only the direct part survives.
                    let eta_aaa_dir_poly = coeff_aaa_dir_poly.clone();

                    for u in 0..p {
                        for v in u..p {
                            let r_uv_fixed = if u == primary.g {
                                fixed.coeff_bu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_bu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };

                            let eta_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&chi_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(fixed.coeff_au[u].as_ref(), a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_au[v].as_ref(), a_u[u]),
                                    &r_uv_fixed,
                                ),
                            );

                            // D_uv integrand: 5 terms
                            let t1 = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_uv[[u, v]]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_aau[u].as_ref(), a_u[v]),
                                    &poly_add(
                                        &poly_scale(fixed.coeff_aau[v].as_ref(), a_u[u]),
                                        &if u == primary.g {
                                            fixed.coeff_abu[v].to_vec()
                                        } else if v == primary.g {
                                            fixed.coeff_abu[u].to_vec()
                                        } else {
                                            vec![0.0; 4]
                                        },
                                    ),
                                ),
                            );
                            let t2 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[v], &eta_poly), &eta_u_poly[u]),
                                -1.0,
                            );
                            let t3 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[u], &eta_poly), &eta_u_poly[v]),
                                -1.0,
                            );
                            let t4 = poly_scale(
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        &poly_mul(&eta_poly, &eta_uv_poly),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5 = poly_mul(
                                &chi_poly,
                                &poly_mul(
                                    &poly_mul(&eta_poly, &eta_poly),
                                    &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                ),
                            );
                            let i_base =
                                poly_add(&poly_add(&poly_add(&t1, &t2), &t3), &poly_add(&t4, &t5));

                            // Polynomial dir-derivatives of per-u quantities
                            let mut eu_dir_fixed_u = vec![0.0; 4];
                            let mut eu_dir_fixed_v = vec![0.0; 4];
                            let mut cu_dir_fixed_u = vec![0.0; 4];
                            let mut cu_dir_fixed_v = vec![0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_u =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_v =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                let sca_u =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                                let sca_v =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, v, c);
                                for k in 0..4 {
                                    eu_dir_fixed_u[k] += sc_u[k] * dir[c];
                                    eu_dir_fixed_v[k] += sc_v[k] * dir[c];
                                    cu_dir_fixed_u[k] += sca_u[k] * dir[c];
                                    cu_dir_fixed_v[k] += sca_v[k] * dir[c];
                                }
                            }
                            let eta_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                                ),
                                &eu_dir_fixed_u,
                            );
                            let eta_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aa_poly, a_u[v] * a_dir),
                                ),
                                &eu_dir_fixed_v,
                            );
                            let chi_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                                ),
                                &cu_dir_fixed_u,
                            );
                            let chi_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aaa_poly, a_u[v] * a_dir),
                                ),
                                &cu_dir_fixed_v,
                            );
                            // eta_uv_dir_poly = D_dir(eta_uv_poly), the FULL third
                            // mixed-with-direction partial of the cell index.
                            // eta_uv_poly has five terms (see above):
                            //   chi·a_uv + eta_aa·a_u·a_v + coeff_au[u]·a_v
                            //     + coeff_au[v]·a_u + r_uv_fixed.
                            // Differentiating along `dir` term-by-term, the FIXED
                            // (direct-parameter) directional crosses of each piece
                            // — `chi_dir`/`eta_aa_dir` direct parts, the `coeff_au`
                            // a-cross direct part, and the entire D_dir(r_uv_fixed)
                            // (which carries `coeff_abu·a_dir` AND the `coeff_bbu`
                            // ∂²/∂g² curvature cross) — were dropped here, so the
                            // density-normalization third (`ln d` chain) lost the
                            // same beta_w/g cross that #1195 restored in the
                            // f_uv_dir cell-integral chain. Re-derive every piece.
                            //
                            // D_dir(coeff_au[u]) = coeff_aau[u]·a_dir
                            //                      + ∂³c/∂a∂u∂dir(direct).
                            let coeff_au_dir_u = {
                                let mut acc = poly_scale(fixed.coeff_aau[u].as_ref(), a_dir);
                                for c in 0..p {
                                    if dir[c] == 0.0 {
                                        continue;
                                    }
                                    let sca = self.cell_pair_third_coeff_a(
                                        primary,
                                        &fixed.coeff_abu,
                                        u,
                                        c,
                                    );
                                    for k in 0..4 {
                                        acc[k] += sca[k] * dir[c];
                                    }
                                }
                                acc
                            };
                            let coeff_au_dir_v = {
                                let mut acc = poly_scale(fixed.coeff_aau[v].as_ref(), a_dir);
                                for c in 0..p {
                                    if dir[c] == 0.0 {
                                        continue;
                                    }
                                    let sca = self.cell_pair_third_coeff_a(
                                        primary,
                                        &fixed.coeff_abu,
                                        v,
                                        c,
                                    );
                                    for k in 0..4 {
                                        acc[k] += sca[k] * dir[c];
                                    }
                                }
                                acc
                            };
                            // D_dir(r_uv_fixed) = ∂³c/∂u∂v∂dir:
                            //   a-chain  coeff_abu[other]·a_dir  (other = the non-g axis)
                            //   direct   coeff_bbu cross contracted along dir.
                            let r_uv_dir = {
                                let mut acc = vec![0.0; 4];
                                // a-chain: D_a of the fixed second cross r_uv_fixed.
                                let sca =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, v);
                                for k in 0..4 {
                                    acc[k] += sca[k] * a_dir;
                                }
                                // direct: ∂²/∂g² curvature cross (the #1195 family).
                                let mut bbu = [0.0; 4];
                                self.add_cell_pair_third_coeff_dir(
                                    primary,
                                    &fixed.coeff_bbu,
                                    u,
                                    v,
                                    dir,
                                    1.0,
                                    &mut bbu,
                                );
                                for k in 0..4 {
                                    acc[k] += bbu[k];
                                }
                                acc
                            };
                            // Term 1: D_dir(chi·a_uv) = chi_dir·a_uv + chi·a_uv_dir.
                            let term1 = poly_add(
                                &poly_scale(&chi_dir_poly, a_uv[[u, v]]),
                                &poly_scale(&chi_poly, a_uv_dir[[u, v]]),
                            );
                            // Term 2: D_dir(eta_aa·a_u·a_v)
                            //   = eta_aa_dir·a_u·a_v + eta_aa·(a_u_dir·a_v + a_u·a_v_dir).
                            let term2 = poly_add(
                                &poly_scale(&eta_aa_dir_poly, a_u[u] * a_u[v]),
                                &poly_scale(
                                    &eta_aa_poly,
                                    a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v],
                                ),
                            );
                            // Terms 3+4: D_dir(coeff_au[u]·a_v + coeff_au[v]·a_u).
                            let term34 = poly_add(
                                &poly_add(
                                    &poly_scale(&coeff_au_dir_u, a_u[v]),
                                    &poly_scale(fixed.coeff_au[u].as_ref(), a_u_dir[v]),
                                ),
                                &poly_add(
                                    &poly_scale(&coeff_au_dir_v, a_u[u]),
                                    &poly_scale(fixed.coeff_au[v].as_ref(), a_u_dir[u]),
                                ),
                            );
                            // Term 5: D_dir(r_uv_fixed).
                            let eta_uv_dir_poly =
                                poly_add(&poly_add(&term1, &term2), &poly_add(&term34, &r_uv_dir));

                            // t1 = chi_uv_poly = ∂²(chi)/∂u∂v
                            //    = eta_aa·a_uv + eta_aaa·a_u·a_v
                            //      + coeff_aau[u]·a_v + coeff_aau[v]·a_u + r1_uv,
                            // r1_uv = ∂³c/∂a∂u∂v = cell_pair_third_coeff_a(coeff_abu,u,v).
                            // The full directional derivative t1_dir = D_dir(t1) had
                            // dropped the direct-parameter (non-a-chain) crosses of
                            // EVERY piece — `eta_aa_dir·a_uv`, `eta_aaa_dir·a_u·a_v`,
                            // the direct part of `D_dir(coeff_aau)`, and the entire
                            // `D_dir(r1_uv)` (`coeff_aabu·a_dir + coeff_abbu·dir`). This
                            // is the chi_uv analogue of the eta_uv beta_w/g cross
                            // restored in #1195; its omission corrupted the density
                            // third (the dominant d-term error in the [g,w] block, #979).
                            //
                            // D_dir(coeff_aau[u]) = coeff_aaau[u]·a_dir
                            //                       + coeff_aabu[u]·dir[g](direct).
                            let coeff_aau_dir_u = {
                                let mut acc = poly_scale(fixed.coeff_aau[u].as_ref(), a_dir);
                                // direct ∂/∂g of ∂²c/∂a²∂u is coeff_aabu[u]·dir[g].
                                let dg = dir[primary.g];
                                for k in 0..4 {
                                    acc[k] = fixed.coeff_aaau[u][k] * a_dir
                                        + fixed.coeff_aabu[u][k] * dg;
                                }
                                acc
                            };
                            let coeff_aau_dir_v = {
                                let mut acc = vec![0.0; 4];
                                let dg = dir[primary.g];
                                for k in 0..4 {
                                    acc[k] = fixed.coeff_aaau[v][k] * a_dir
                                        + fixed.coeff_aabu[v][k] * dg;
                                }
                                acc
                            };
                            // D_dir(r1_uv) = coeff_aabu[other]·a_dir + coeff_abbu cross.
                            let r1_uv_dir = {
                                let mut acc = vec![0.0; 4];
                                let sca =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_aabu, u, v);
                                for k in 0..4 {
                                    acc[k] += sca[k] * a_dir;
                                }
                                let mut abb = [0.0; 4];
                                self.add_cell_pair_third_coeff_dir(
                                    primary,
                                    &fixed.coeff_abbu,
                                    u,
                                    v,
                                    dir,
                                    1.0,
                                    &mut abb,
                                );
                                for k in 0..4 {
                                    acc[k] += abb[k];
                                }
                                acc
                            };
                            let t1_dir = poly_add(
                                &poly_add(
                                    // D_dir(eta_aa·a_uv)
                                    &poly_add(
                                        &poly_scale(&eta_aa_dir_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_uv_dir[[u, v]]),
                                    ),
                                    // D_dir(eta_aaa·a_u·a_v)
                                    &poly_add(
                                        &poly_scale(&eta_aaa_dir_poly, a_u[u] * a_u[v]),
                                        &poly_scale(
                                            &eta_aaa_poly,
                                            a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v],
                                        ),
                                    ),
                                ),
                                &poly_add(
                                    // D_dir(coeff_aau[u]·a_v + coeff_aau[v]·a_u)
                                    &poly_add(
                                        &poly_add(
                                            &poly_scale(&coeff_aau_dir_u, a_u[v]),
                                            &poly_scale(fixed.coeff_aau[u].as_ref(), a_u_dir[v]),
                                        ),
                                        &poly_add(
                                            &poly_scale(&coeff_aau_dir_v, a_u[u]),
                                            &poly_scale(fixed.coeff_aau[v].as_ref(), a_u_dir[u]),
                                        ),
                                    ),
                                    // D_dir(r1_uv)
                                    &r1_uv_dir,
                                ),
                            );
                            let t2_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_v, &eta_poly),
                                            &eta_u_poly[u],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[v], &eta_dir_poly),
                                            &eta_u_poly[u],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[v], &eta_poly),
                                        &eta_u_dir_poly_u,
                                    ),
                                ),
                                -1.0,
                            );
                            let t3_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_u, &eta_poly),
                                            &eta_u_poly[v],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[u], &eta_dir_poly),
                                            &eta_u_poly[v],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[u], &eta_poly),
                                        &eta_u_dir_poly_v,
                                    ),
                                ),
                                -1.0,
                            );
                            let t4_dir = poly_scale(
                                &poly_add(
                                    &poly_mul(
                                        &chi_dir_poly,
                                        &poly_add(
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                            &poly_mul(&eta_poly, &eta_uv_poly),
                                        ),
                                    ),
                                    &poly_mul(
                                        &chi_poly,
                                        &poly_add(
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                            &poly_add(
                                                &poly_mul(&eta_dir_poly, &eta_uv_poly),
                                                &poly_mul(&eta_poly, &eta_uv_dir_poly),
                                            ),
                                        ),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5_dir = poly_add(
                                &poly_mul(
                                    &chi_dir_poly,
                                    &poly_mul(
                                        &poly_mul(&eta_poly, &eta_poly),
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                    ),
                                ),
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(
                                            &poly_scale(&poly_mul(&eta_dir_poly, &eta_poly), 2.0),
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        ),
                                        &poly_mul(
                                            &poly_mul(&eta_poly, &eta_poly),
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                        ),
                                    ),
                                ),
                            );

                            let i_base_dir = poly_add(
                                &poly_add(&poly_add(&t1_dir, &t2_dir), &t3_dir),
                                &poly_add(&t4_dir, &t5_dir),
                            );
                            let full_integrand = poly_sub(
                                &i_base_dir,
                                &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &i_base),
                            );

                            let value = exact_kernel::cell_polynomial_integral_from_moments(
                                &full_integrand,
                                &state_ref.moments,
                                "survival D_t second derivative directional",
                            )?;
                            #[cfg(test)]
                            if dbg_target == Some((u, v)) {
                                let ig = |poly: &[f64]| -> f64 {
                                    exact_kernel::cell_polynomial_integral_from_moments(
                                        poly,
                                        &state_ref.moments,
                                        "dbg",
                                    )
                                    .unwrap_or(f64::NAN)
                                };
                                // Base per-term ∫t_i·m (t1=i_base's chi_uv piece..t5).
                                dbg_terms[0] += ig(&t1);
                                dbg_terms[1] += ig(&t2);
                                dbg_terms[2] += ig(&t3);
                                dbg_terms[3] += ig(&t4);
                                dbg_terms[4] += ig(&t5);
                                // Analytic FULL dir per-term:
                                //   D_dir(∫t_i·m) = ∫t_i_dir·m - ∫eta·eta_dir·t_i·m.
                                // The trailing moment-measure correction must be
                                // included per term so it equals the FD of ∫t_i·m.
                                let mw = poly_mul(&eta_poly, &eta_dir_poly);
                                let corr = |t: &[f64]| ig(&poly_mul(&mw, t));
                                dbg_terms[5] += ig(&t1_dir) - corr(&t1);
                                dbg_terms[6] += ig(&t2_dir) - corr(&t2);
                                dbg_terms[7] += ig(&t3_dir) - corr(&t3);
                                dbg_terms[8] += ig(&t4_dir) - corr(&t4);
                                dbg_terms[9] += ig(&t5_dir) - corr(&t5);
                            }
                            d_uv_dir[[u, v]] += value;
                            d_uv_dir[[v, u]] = d_uv_dir[[u, v]];
                        }
                    }
                    Ok((d_uv_dir, dbg_terms))
                })
                .collect::<Result<Vec<_>, String>>()?;
            #[cfg(test)]
            let mut dbg_sum = [0.0_f64; 10];
            for (cell_d_uv_dir, _cell_dbg) in &d_uv_dir_cell_accums {
                for u in 0..p {
                    for v in 0..p {
                        d_uv_dir[[u, v]] += cell_d_uv_dir[[u, v]];
                    }
                }
                #[cfg(test)]
                for k in 0..10 {
                    dbg_sum[k] += _cell_dbg[k];
                }
            }
            #[cfg(test)]
            {
                debug_d_uv_terms = Some((
                    [dbg_sum[0], dbg_sum[1], dbg_sum[2], dbg_sum[3], dbg_sum[4]],
                    [dbg_sum[5], dbg_sum[6], dbg_sum[7], dbg_sum[8], dbg_sum[9]],
                ));
            }
        }

        Ok(SurvivalFlexTimepointDirectionalExact {
            eta_uv_dir,
            chi_uv_dir,
            d_u_dir,
            d_uv_dir,
            #[cfg(test)]
            debug_d_uv_terms,
            #[cfg(test)]
            debug_d_u_polys,
        })
    }
}
