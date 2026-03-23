
    /// Exact directional derivative of the joint Hessian for timewiggle-only
    /// models (no score-warp / link-deviation).  Replaces FD by analytically
    /// differentiating the J^T H J + f·K pullback through the timewiggle
    /// q-map geometry (equation 47 of the unified pullback framework).
    fn exact_newton_joint_hessian_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let p_total = slices.total;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);

        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    let u_d = self.row_primary_direction_from_flat_dynamic(
                        row, block_states, &slices, d_beta_flat,
                    )?;
                    let t_ud = self.row_primary_third_contracted(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);

                    // Term 1 + 3: reuse core accumulator with (H·u^d, T[u^d])
                    self.accumulate_dynamic_q_core_hessian(
                        row, &slices, &q_geom, h_ud.view(), t_ud.view(), &mut acc,
                    );

                    // ── Timewiggle Jacobian derivatives ────────────────
                    let ec = self.design_entry.row_chunk(row..row + 1);
                    let xc = self.design_exit.row_chunk(row..row + 1);
                    let dc = self.design_derivative_exit.row_chunk(row..row + 1);
                    let xe = ec.row(0).slice(s![..p_base]).to_owned();
                    let xx = xc.row(0).slice(s![..p_base]).to_owned();
                    let xd = dc.row(0).slice(s![..p_base]).to_owned();
                    let mc = self.marginal_design.row_chunk(row..row + 1);
                    let mr = mc.row(0).to_owned();

                    let dh0 = xe.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
                    let dh1 = xx.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
                    let ddr = xd.dot(&d_time.slice(s![..p_base]));
                    let bm = block_states[1].eta[row];
                    let h0 = xe.dot(&beta_time.slice(s![..p_base]))
                        + self.offset_entry[row] + bm;
                    let h1 = xx.dot(&beta_time.slice(s![..p_base]))
                        + self.offset_exit[row] + bm;
                    let dr = xd.dot(&beta_time.slice(s![..p_base]))
                        + self.derivative_offset_exit[row];

                    let eg = self
                        .time_wiggle_geometry(
                            Array1::from_vec(vec![h0]).view(), beta_time_w,
                        )?
                        .ok_or_else(|| "timewiggle geometry missing at entry".to_string())?;
                    let xg = self
                        .time_wiggle_geometry(
                            Array1::from_vec(vec![h1]).view(), beta_time_w,
                        )?
                        .ok_or_else(|| "timewiggle geometry missing at exit".to_string())?;

                    let (m2e, m3e) = (eg.d2q_dq02[0], eg.d3q_dq03[0]);
                    let (m2x, m3x, m4x) = (xg.d2q_dq02[0], xg.d3q_dq03[0], xg.d4q_dq04[0]);

                    // dJ_{q,time}[a] / dβ[d]
                    let mut dj0t = vec![0.0f64; p_time];
                    let mut dj1t = vec![0.0f64; p_time];
                    let mut djdt = vec![0.0f64; p_time];
                    for a in 0..p_base {
                        dj0t[a] = m2e * dh0 * xe[a];
                        dj1t[a] = m2x * dh1 * xx[a];
                        djdt[a] = m3x * dh1 * dr * xx[a]
                            + m2x * ddr * xx[a] + m2x * dh1 * xd[a];
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        dj0t[ci] = eg.basis_d1[[0, li]] * dh0;
                        dj1t[ci] = xg.basis_d1[[0, li]] * dh1;
                        djdt[ci] = xg.basis_d2[[0, li]] * dh1 * dr
                            + xg.basis_d1[[0, li]] * ddr;
                    }
                    let djt = [&dj0t[..], &dj1t[..], &djdt[..]];

                    let mut dj0m = vec![0.0f64; p_marginal];
                    let mut dj1m = vec![0.0f64; p_marginal];
                    let mut djdm = vec![0.0f64; p_marginal];
                    for a in 0..p_marginal {
                        dj0m[a] = m2e * dh0 * mr[a];
                        dj1m[a] = m2x * dh1 * mr[a];
                        djdm[a] = m3x * dh1 * dr * mr[a] + m2x * ddr * mr[a];
                    }
                    let djm = [&dj0m[..], &dj1m[..], &djdm[..]];

                    let jt: [&Array1<f64>; 3] =
                        [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
                    let jm: [&Array1<f64>; 3] =
                        [&q_geom.dq0_marginal, &q_geom.dq1_marginal,
                         &q_geom.dqd1_marginal];

                    // Term 2: (dJ/d)^T H J + J^T H (dJ/d)
                    for a in 0..p_time { for b in 0..p_time {
                        let mut v = 0.0;
                        for qu in 0..3 { for qv in 0..3 {
                            v += h_pi[[qu, qv]]
                                * (djt[qu][a] * jt[qv][b] + jt[qu][a] * djt[qv][b]);
                        }}
                        acc[[slices.time.start + a, slices.time.start + b]] += v;
                    }}
                    for a in 0..p_marginal { for b in 0..p_marginal {
                        let mut v = 0.0;
                        for qu in 0..3 { for qv in 0..3 {
                            v += h_pi[[qu, qv]]
                                * (djm[qu][a] * jm[qv][b] + jm[qu][a] * djm[qv][b]);
                        }}
                        acc[[slices.marginal.start + a, slices.marginal.start + b]] += v;
                    }}
                    for a in 0..p_time { for b in 0..p_marginal {
                        let mut v = 0.0;
                        for qu in 0..3 { for qv in 0..3 {
                            v += h_pi[[qu, qv]]
                                * (djt[qu][a] * jm[qv][b] + jt[qu][a] * djm[qv][b]);
                        }}
                        acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                        acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                    }}
                    let gc = self.logslope_design.row_chunk(row..row + 1);
                    let gr = gc.row(0);
                    for a in 0..p_time {
                        let mut w = 0.0;
                        for qu in 0..3 { w += h_pi[[qu, 3]] * djt[qu][a]; }
                        for b in 0..slices.logslope.len() {
                            let v = w * gr[b];
                            acc[[slices.time.start + a, slices.logslope.start + b]] += v;
                            acc[[slices.logslope.start + b, slices.time.start + a]] += v;
                        }
                    }
                    for a in 0..p_marginal {
                        let mut w = 0.0;
                        for qu in 0..3 { w += h_pi[[qu, 3]] * djm[qu][a]; }
                        for b in 0..slices.logslope.len() {
                            let v = w * gr[b];
                            acc[[slices.marginal.start + a, slices.logslope.start + b]] += v;
                            acc[[slices.logslope.start + b, slices.marginal.start + a]] += v;
                        }
                    }

                    // Term 4: Σ_r f_r dK_r/d
                    for a in 0..p_base { for b in 0..p_base {
                        let dk0 = m3e * dh0 * xe[a] * xe[b];
                        let dk1 = m3x * dh1 * xx[a] * xx[b];
                        let dkd = m4x * dh1 * dr * xx[a] * xx[b]
                            + m3x * ddr * xx[a] * xx[b]
                            + m3x * dh1 * (xx[a] * xd[b] + xd[a] * xx[b]);
                        acc[[slices.time.start + a, slices.time.start + b]] +=
                            f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                    }}
                    for li in 0..time_tail.len() { let ci = time_tail.start + li;
                        for a in 0..p_base {
                            let dk0 = eg.basis_d2[[0, li]] * dh0 * xe[a];
                            let dk1 = xg.basis_d2[[0, li]] * dh1 * xx[a];
                            let dkd = xg.basis_d3[[0, li]] * dh1 * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddr * xx[a]
                                + xg.basis_d2[[0, li]] * dh1 * xd[a];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + a, slices.time.start + ci]] += v;
                            acc[[slices.time.start + ci, slices.time.start + a]] += v;
                        }
                    }
                    for a in 0..p_base { for b in 0..p_marginal {
                        let dk0 = m3e * dh0 * xe[a] * mr[b];
                        let dk1 = m3x * dh1 * xx[a] * mr[b];
                        let dkd = m4x * dh1 * dr * xx[a] * mr[b]
                            + m3x * ddr * xx[a] * mr[b]
                            + m3x * dh1 * xd[a] * mr[b];
                        let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                        acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                        acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                    }}
                    for li in 0..time_tail.len() { let ci = time_tail.start + li;
                        for b in 0..p_marginal {
                            let dk0 = eg.basis_d2[[0, li]] * dh0 * mr[b];
                            let dk1 = xg.basis_d2[[0, li]] * dh1 * mr[b];
                            let dkd = xg.basis_d3[[0, li]] * dh1 * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddr * mr[b];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
                        }
                    }
                    for a in 0..p_marginal { for b in 0..p_marginal {
                        let dk0 = m3e * dh0 * mr[a] * mr[b];
                        let dk1 = m3x * dh1 * mr[a] * mr[b];
                        let dkd = m4x * dh1 * dr * mr[a] * mr[b]
                            + m3x * ddr * mr[a] * mr[b];
                        acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                            f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                    }}

                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> { a += &b; Ok(a) },
            )?;
        Ok(result)
    }

    /// Exact second directional derivative for timewiggle-only: FD of the
    /// exact first directional derivative above.
    fn exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let norm = d_u.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let step = (1e-5 / norm.max(1.0)).max(1e-7);
        let plus = self.perturb_block_states_along_direction(block_states, d_u, step)?;
        let minus = self.perturb_block_states_along_direction(block_states, d_u, -step)?;
        let dh_p = self.exact_newton_joint_hessian_directional_derivative_timewiggle(&plus, d_v)?;
        let dh_m = self.exact_newton_joint_hessian_directional_derivative_timewiggle(&minus, d_v)?;
        Ok((dh_p - dh_m) / (2.0 * step))
    }
