//! Exact mixed bidirectional extension D_{d1} D_{d2} of the timepoint
//! quantities.
//!
//! Carries the calibration solve, observed-point η/χ jet transport, and the
//! cellwise density-normalization integrand exactly through the implicit
//! intercept solve for a pair of directions.

use super::*;

#[inline]
fn eval_poly_slice(coefficients: &[f64], z: f64) -> f64 {
    let mut acc = 0.0;
    for &coefficient in coefficients.iter().rev() {
        acc = acc * z + coefficient;
    }
    acc
}

#[inline]
fn eval_poly_derivative_slice(coefficients: &[f64], z: f64) -> f64 {
    let mut acc = 0.0;
    for (power, &coefficient) in coefficients.iter().enumerate().skip(1).rev() {
        acc = acc * z + (power as f64) * coefficient;
    }
    acc
}

#[inline]
fn reciprocal_bilinear_jet(value: MultiDirJet) -> MultiDirJet {
    let x0 = value.coeff(0);
    let x1 = value.coeff(1);
    let x2 = value.coeff(2);
    let x12 = value.coeff(3);
    let inv = 1.0 / x0;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    MultiDirJet::bilinear(
        inv,
        -x1 * inv2,
        -x2 * inv2,
        2.0 * x1 * x2 * inv3 - x12 * inv2,
    )
}

impl SurvivalMarginalSlopeFamily {
    /// Exact mixed bidirectional extension D_{d1} D_{d2} of the timepoint
    /// quantities. This carries the calibration solve, observed eta/chi
    /// transport, and density-normalization transport analytically.
    pub(crate) fn compute_survival_timepoint_bidirectional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        self.compute_survival_timepoint_bidirectional_exact_full(
            row, primary, q, q_index, a, b, beta_h, beta_w, dir1, dir2,
        )
    }

    pub(crate) fn compute_survival_timepoint_bidirectional_exact_full(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;
        self.compute_survival_timepoint_bidirectional_exact_from_cached(
            row, primary, q, q_index, a, b, beta_h, beta_w, &cached, dir1, dir2,
        )
    }

    pub(crate) fn compute_survival_timepoint_bidirectional_exact_from_cached(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        cached: &CachedPartitionCells,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        let p = primary.total;
        let zero4 = [0.0; 4];

        struct BiDirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Array1<f64>,
            f_au: Array1<f64>,
            f_uv: Array2<f64>,
            f_a_d1: f64,
            f_aa_d1: f64,
            f_au_d1: Array1<f64>,
            f_uv_d1: Array2<f64>,
            f_a_d2: f64,
            f_aa_d2: f64,
            f_au_d2: Array1<f64>,
            f_uv_d2: Array2<f64>,
            f_a_d12: f64,
            f_aa_d12: f64,
            f_au_d12: Array1<f64>,
            f_uv_d12: Array2<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<BiDirectionalTimepointCellAccum, String> {
                let mut f_a = 0.0f64;
                let mut f_aa = 0.0f64;
                let mut f_u = Array1::<f64>::zeros(p);
                let mut f_au = Array1::<f64>::zeros(p);
                let mut f_uv = Array2::<f64>::zeros((p, p));
                let mut f_a_d1 = 0.0f64;
                let mut f_aa_d1 = 0.0f64;
                let mut f_au_d1 = Array1::<f64>::zeros(p);
                let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
                let mut f_a_d2 = 0.0f64;
                let mut f_aa_d2 = 0.0f64;
                let mut f_au_d2 = Array1::<f64>::zeros(p);
                let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
                let mut f_a_d12 = 0.0f64;
                let mut f_aa_d12 = 0.0f64;
                let mut f_au_d12 = Array1::<f64>::zeros(p);
                let mut f_uv_d12 = Array2::<f64>::zeros((p, p));
                let nc = ce.neg_cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let da = fx.dc_da.map(|v| -v);
                let daa = fx.dc_daa.map(|v| -v);

                f_a += exact_kernel::cell_first_derivative_from_moments(&da, &st.moments)?;
                f_aa += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &daa,
                    &st.moments,
                )?;

                let mut cd1 = [0.0; 4];
                let mut ca1 = [0.0; 4];
                let mut caa1 = [0.0; 4];
                let mut cd2 = [0.0; 4];
                let mut ca2 = [0.0; 4];
                let mut caa2 = [0.0; 4];
                let mut cd12 = [0.0; 4];
                let mut ca12 = [0.0; 4];
                let coeff_view = SparsePrimaryCoeffJetView::new(
                    primary.g,
                    primary.h.as_ref(),
                    primary.w.as_ref(),
                    &fx.coeff_u,
                    &fx.coeff_au,
                    &fx.coeff_bu,
                    &fx.coeff_aau,
                    &fx.coeff_abu,
                    &fx.coeff_bbu,
                    &fx.coeff_aaau,
                    &fx.coeff_aabu,
                    &fx.coeff_abbu,
                    &fx.coeff_bbbu,
                );
                for c in 0..p {
                    for k in 0..4 {
                        if dir1[c] != 0.0 {
                            cd1[k] -= fx.coeff_u[c][k] * dir1[c];
                            ca1[k] -= fx.coeff_au[c][k] * dir1[c];
                            caa1[k] -= fx.coeff_aau[c][k] * dir1[c];
                        }
                        if dir2[c] != 0.0 {
                            cd2[k] -= fx.coeff_u[c][k] * dir2[c];
                            ca2[k] -= fx.coeff_au[c][k] * dir2[c];
                            caa2[k] -= fx.coeff_aau[c][k] * dir2[c];
                        }
                    }
                }
                for c1 in 0..p {
                    if dir1[c1] == 0.0 {
                        continue;
                    }
                    for c2 in 0..p {
                        if dir2[c2] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, c1, c2);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, c1, c2);
                        for k in 0..4 {
                            cd12[k] -= sc[k] * dir1[c1] * dir2[c2];
                            ca12[k] -= sca[k] * dir1[c1] * dir2[c2];
                        }
                    }
                }
                let caa12 = coeff_view
                    .mixed_directional_from_b_family(&fx.coeff_aabu, dir1, dir2, COEFF_SUPPORT_GHW)
                    .map(|value| -value);

                f_a_d1 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &ca1,
                    &st.moments,
                )?;
                f_a_d2 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd2,
                    &ca2,
                    &st.moments,
                )?;
                f_a_d12 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &cd2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &ca12,
                    &st.moments,
                )?;
                f_aa_d1 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &daa,
                    &ca1,
                    &ca1,
                    &caa1,
                    &st.moments,
                )?;
                f_aa_d2 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd2,
                    &daa,
                    &ca2,
                    &ca2,
                    &caa2,
                    &st.moments,
                )?;
                f_aa_d12 += exact_kernel::cell_fourth_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &cd2,
                    &daa,
                    &ca1,
                    &ca2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &caa1,
                    &caa2,
                    &ca12,
                    &ca12,
                    &caa12,
                    &st.moments,
                )?;

                for u in 0..p {
                    let cu = fx.coeff_u[u].map(|v| -v);
                    let cau = fx.coeff_au[u].map(|v| -v);
                    f_u[u] += exact_kernel::cell_first_derivative_from_moments(&cu, &st.moments)?;
                    f_au[u] += exact_kernel::cell_second_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cau,
                        &st.moments,
                    )?;
                    let mut cu1 = [0.0; 4];
                    let mut cau1 = [0.0; 4];
                    let mut cu2 = [0.0; 4];
                    let mut cau2 = [0.0; 4];
                    let cu12 = coeff_view
                        .param_mixed_from_bb_family(&fx.coeff_bbu, u, dir1, dir2, COEFF_SUPPORT_GHW)
                        .map(|value| -value);
                    let cau12 = coeff_view
                        .param_mixed_from_bb_family(
                            &fx.coeff_abbu,
                            u,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        )
                        .map(|value| -value);
                    for c in 0..p {
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, c);
                        for k in 0..4 {
                            if dir1[c] != 0.0 {
                                cu1[k] -= sc[k] * dir1[c];
                                cau1[k] -= sca[k] * dir1[c];
                            }
                            if dir2[c] != 0.0 {
                                cu2[k] -= sc[k] * dir2[c];
                                cau2[k] -= sca[k] * dir2[c];
                            }
                        }
                    }
                    f_au_d1[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cau,
                        &ca1,
                        &cu1,
                        &cau1,
                        &st.moments,
                    )?;
                    f_au_d2[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd2,
                        &cau,
                        &ca2,
                        &cu2,
                        &cau2,
                        &st.moments,
                    )?;
                    f_au_d12[u] += exact_kernel::cell_fourth_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cd2,
                        &cau,
                        &ca1,
                        &ca2,
                        &cu1,
                        &cu2,
                        &cd12,
                        &cau1,
                        &cau2,
                        &ca12,
                        &cu12,
                        &cau12,
                        &st.moments,
                    )?;
                }
                for u in 0..p {
                    for v in u..p {
                        let cu = fx.coeff_u[u].map(|x| -x);
                        let cv = fx.coeff_u[v].map(|x| -x);
                        let sc = self
                            .cell_pair_second_coeff(primary, &fx.coeff_bu, u, v)
                            .map(|x| -x);
                        let bv = exact_kernel::cell_second_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &sc,
                            &st.moments,
                        )?;
                        f_uv[[u, v]] += bv;
                        if u != v {
                            f_uv[[v, u]] += bv;
                        }
                        let mut cu1 = [0.0; 4];
                        let mut cv1 = [0.0; 4];
                        let mut cu2 = [0.0; 4];
                        let mut cv2 = [0.0; 4];
                        let mut cuv1 = [0.0; 4];
                        let mut cuv2 = [0.0; 4];
                        let cu12 = coeff_view
                            .param_mixed_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            )
                            .map(|value| -value);
                        let cv12 = coeff_view
                            .param_mixed_from_bb_family(
                                &fx.coeff_bbu,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            )
                            .map(|value| -value);
                        let cuv12 = coeff_view
                            .pair_mixed_from_bbb_family(
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            )
                            .map(|value| -value);
                        for c in 0..p {
                            let suc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                            let svc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, v, c);
                            for k in 0..4 {
                                if dir1[c] != 0.0 {
                                    cu1[k] -= suc[k] * dir1[c];
                                    cv1[k] -= svc[k] * dir1[c];
                                }
                                if dir2[c] != 0.0 {
                                    cu2[k] -= suc[k] * dir2[c];
                                    cv2[k] -= svc[k] * dir2[c];
                                }
                            }
                        }
                        self.add_cell_pair_third_coeff_dir(
                            primary,
                            &fx.coeff_bbu,
                            u,
                            v,
                            dir1,
                            -1.0,
                            &mut cuv1,
                        );
                        self.add_cell_pair_third_coeff_dir(
                            primary,
                            &fx.coeff_bbu,
                            u,
                            v,
                            dir2,
                            -1.0,
                            &mut cuv2,
                        );
                        let d1v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &sc,
                            &cu1,
                            &cv1,
                            &cuv1,
                            &st.moments,
                        )?;
                        f_uv_d1[[u, v]] += d1v;
                        if u != v {
                            f_uv_d1[[v, u]] += d1v;
                        }
                        let d2v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd2,
                            &sc,
                            &cu2,
                            &cv2,
                            &cuv2,
                            &st.moments,
                        )?;
                        f_uv_d2[[u, v]] += d2v;
                        if u != v {
                            f_uv_d2[[v, u]] += d2v;
                        }
                        let d12v = exact_kernel::cell_fourth_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &cd2,
                            &sc,
                            &cu1,
                            &cu2,
                            &cv1,
                            &cv2,
                            &cd12,
                            &cuv1,
                            &cuv2,
                            &cu12,
                            &cv12,
                            &cuv12,
                            &st.moments,
                        )?;
                        f_uv_d12[[u, v]] += d12v;
                        if u != v {
                            f_uv_d12[[v, u]] += d12v;
                        }
                    }
                }

                Ok(BiDirectionalTimepointCellAccum {
                    f_a,
                    f_aa,
                    f_u,
                    f_au,
                    f_uv,
                    f_a_d1,
                    f_aa_d1,
                    f_au_d1,
                    f_uv_d1,
                    f_a_d2,
                    f_aa_d2,
                    f_au_d2,
                    f_uv_d2,
                    f_a_d12,
                    f_aa_d12,
                    f_au_d12,
                    f_uv_d12,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0f64;
        let mut f_aa = 0.0f64;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_d1 = 0.0f64;
        let mut f_aa_d1 = 0.0f64;
        let mut f_au_d1 = Array1::<f64>::zeros(p);
        let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
        let mut f_a_d2 = 0.0f64;
        let mut f_aa_d2 = 0.0f64;
        let mut f_au_d2 = Array1::<f64>::zeros(p);
        let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
        let mut f_a_d12 = 0.0f64;
        let mut f_aa_d12 = 0.0f64;
        let mut f_au_d12 = Array1::<f64>::zeros(p);
        let mut f_uv_d12 = Array2::<f64>::zeros((p, p));

        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_d1 += acc.f_a_d1;
            f_aa_d1 += acc.f_aa_d1;
            f_a_d2 += acc.f_a_d2;
            f_aa_d2 += acc.f_aa_d2;
            f_a_d12 += acc.f_a_d12;
            f_aa_d12 += acc.f_aa_d12;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_d1[u] += acc.f_au_d1[u];
                f_au_d2[u] += acc.f_au_d2[u];
                f_au_d12[u] += acc.f_au_d12[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[[u, v]];
                    f_uv_d1[[u, v]] += acc.f_uv_d1[[u, v]];
                    f_uv_d2[[u, v]] += acc.f_uv_d2[[u, v]];
                    f_uv_d12[[u, v]] += acc.f_uv_d12[[u, v]];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        // q-marginal calibration RHS self-coupling, differentiated along the two
        // independent directions. Base: `f_uv[[q,q]] = -q·φ(q)`. Its first and
        // second q-derivatives (the d1/d2 single-direction and d12 cross terms)
        // are `∂_q(-qφ) = (q²-1)φ` and `∂²_q(-qφ) = q·(3-q²)φ`. The previous
        // `(1-q²)` / `q·(q²-3)` were both sign-flipped relative to the shared
        // base, corrupting the (q,·) blocks of the contracted fourth tower
        // (gam#932/#979).
        f_uv_d1[[q_index, q_index]] += dir1[q_index] * (q * q - 1.0) * phi_q;
        f_uv_d2[[q_index, q_index]] += dir2[q_index] * (q * q - 1.0) * phi_q;
        f_uv_d12[[q_index, q_index]] += dir1[q_index] * dir2[q_index] * q * (3.0 - q * q) * phi_q;

        let inv = 1.0 / f_a;
        let mut au = Array1::<f64>::zeros(p);
        for u in 0..p {
            au[u] = -f_u[u] * inv;
        }
        let mut auv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * au[v] + f_au[v] * au[u] + f_aa * au[u] * au[v])
                        * inv;
                auv[[u, v]] = val;
                auv[[v, u]] = val;
            }
        }
        let ad1 = au.dot(dir1);
        let ad2 = au.dot(dir2);
        let aud1 = auv.dot(dir1);
        let aud2 = auv.dot(dir2);

        let mut auvd1 = Array2::<f64>::zeros((p, p));
        let mut auvd2 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n1 = f_uv_d1[[u, v]]
                    + f_au_d1[u] * au[v]
                    + f_au[u] * aud1[v]
                    + f_au_d1[v] * au[u]
                    + f_au[v] * aud1[u]
                    + f_aa_d1 * au[u] * au[v]
                    + f_aa * (aud1[u] * au[v] + au[u] * aud1[v]);
                let v1 = -(n1 + f_a_d1 * auv[[u, v]]) * inv;
                auvd1[[u, v]] = v1;
                auvd1[[v, u]] = v1;

                let n2 = f_uv_d2[[u, v]]
                    + f_au_d2[u] * au[v]
                    + f_au[u] * aud2[v]
                    + f_au_d2[v] * au[u]
                    + f_au[v] * aud2[u]
                    + f_aa_d2 * au[u] * au[v]
                    + f_aa * (aud2[u] * au[v] + au[u] * aud2[v]);
                let v2 = -(n2 + f_a_d2 * auv[[u, v]]) * inv;
                auvd2[[u, v]] = v2;
                auvd2[[v, u]] = v2;
            }
        }

        let ad12 = aud2.dot(dir1);
        let aud12 = auvd2.dot(dir1);
        let mut auvd12 = Array2::<f64>::zeros((p, p));
        let f_a_jet = MultiDirJet::bilinear(f_a, f_a_d1, f_a_d2, f_a_d12);
        let f_a_recip_jet = reciprocal_bilinear_jet(f_a_jet);
        let f_aa_jet = MultiDirJet::bilinear(f_aa, f_aa_d1, f_aa_d2, f_aa_d12);
        for u in 0..p {
            for v in u..p {
                let f_uv_jet = MultiDirJet::bilinear(
                    f_uv[[u, v]],
                    f_uv_d1[[u, v]],
                    f_uv_d2[[u, v]],
                    f_uv_d12[[u, v]],
                );
                let f_au_u_jet =
                    MultiDirJet::bilinear(f_au[u], f_au_d1[u], f_au_d2[u], f_au_d12[u]);
                let f_au_v_jet =
                    MultiDirJet::bilinear(f_au[v], f_au_d1[v], f_au_d2[v], f_au_d12[v]);
                let a_u_jet = MultiDirJet::bilinear(au[u], aud1[u], aud2[u], aud12[u]);
                let a_v_jet = MultiDirJet::bilinear(au[v], aud1[v], aud2[v], aud12[v]);
                let numerator = f_uv_jet
                    .add(&f_au_u_jet.mul(&a_v_jet))
                    .add(&f_au_v_jet.mul(&a_u_jet))
                    .add(&f_aa_jet.mul(&a_u_jet.mul(&a_v_jet)));
                let val = numerator.mul(&f_a_recip_jet).scale(-1.0).coeff(3);
                auvd12[[u, v]] = val;
                auvd12[[v, u]] = val;
            }
        }

        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut g_au_fixed = vec![[0.0; 4]; p];
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
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, dc_bw) =
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
                g_bu_fixed[idx] = scale_coeff4(dc_bw, scale);
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
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

        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let chi_jet = scalar_composite_bilinear(
            chi,
            eta_aa,
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.ab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aa_jet = scalar_composite_bilinear(
            eta_aa,
            eta_aaa,
            0.0,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aaa_jet = MultiDirJet::bilinear(
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
        );

        let mut a_u_jets = Vec::with_capacity(p);
        let mut tau_jets = Vec::with_capacity(p);
        let mut tau_a_jets = Vec::with_capacity(p);
        for u in 0..p {
            a_u_jets.push(MultiDirJet::bilinear(au[u], aud1[u], aud2[u], aud12[u]));
            tau_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_au_fixed[u], z_obs),
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                ad1,
                ad2,
                ad12,
            ));
            tau_a_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                0.0,
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                0.0,
                0.0,
                ad1,
                ad2,
                ad12,
            ));
        }

        let mut eta_uv_uv = Array2::<f64>::zeros((p, p));
        let mut chi_uv_uv = Array2::<f64>::zeros((p, p));
        let mut d_uv_uv = Array2::<f64>::zeros((p, p));

        for u in 0..p {
            for v in u..p {
                let a_uv_jet = MultiDirJet::bilinear(
                    auv[[u, v]],
                    auvd1[[u, v]],
                    auvd2[[u, v]],
                    auvd12[[u, v]],
                );
                let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                let r_uv_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_GHW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_uv_fixed_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    0.0,
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    0.0,
                    0.0,
                    ad1,
                    ad2,
                    ad12,
                );

                let eta_uv_jet = chi_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aa_jet.mul(&a_u_prod))
                    .add(&tau_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_jets[v].mul(&a_u_jets[u]))
                    .add(&r_uv_jet);
                let chi_uv_jet = eta_aa_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aaa_jet.mul(&a_u_prod))
                    .add(&tau_a_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_a_jets[v].mul(&a_u_jets[u]))
                    .add(&chi_uv_fixed_jet);

                eta_uv_uv[[u, v]] = eta_uv_jet.coeff(3);
                eta_uv_uv[[v, u]] = eta_uv_uv[[u, v]];
                chi_uv_uv[[u, v]] = chi_uv_jet.coeff(3);
                chi_uv_uv[[v, u]] = chi_uv_uv[[u, v]];
            }
        }

        let primary_view = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let d_uv_uv_cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<Array2<f64>, String> {
                let mut d_uv_uv = Array2::<f64>::zeros((p, p));
                let cell = ce.partition_cell.cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let eta_base = [cell.c0, cell.c1, cell.c2, cell.c3];

                let coeff_dir1 =
                    primary_view.directional_family(&fx.coeff_u, dir1, COEFF_SUPPORT_GHW);
                let coeff_dir2 =
                    primary_view.directional_family(&fx.coeff_u, dir2, COEFF_SUPPORT_GHW);
                let coeff_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_bu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_a_dir1 =
                    primary_view.directional_family(&fx.coeff_au, dir1, COEFF_SUPPORT_GHW);
                let coeff_a_dir2 =
                    primary_view.directional_family(&fx.coeff_au, dir2, COEFF_SUPPORT_GHW);
                let coeff_a_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_abu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_aa_dir1 =
                    primary_view.directional_family(&fx.coeff_aau, dir1, COEFF_SUPPORT_GHW);
                let coeff_aa_dir2 =
                    primary_view.directional_family(&fx.coeff_aau, dir2, COEFF_SUPPORT_GHW);
                let coeff_aa_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_aabu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_aaa_dir1 =
                    primary_view.directional_family(&fx.coeff_aaau, dir1, COEFF_SUPPORT_GHW);
                let coeff_aaa_dir2 =
                    primary_view.directional_family(&fx.coeff_aaau, dir2, COEFF_SUPPORT_GHW);
                let coeff_aaa_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_aabu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );

                let eta_poly_jet = coeff4_composite_bilinear(
                    &eta_base,
                    &fx.dc_da,
                    &fx.dc_daa,
                    &coeff_dir1,
                    &coeff_dir2,
                    &coeff_dir12,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_da,
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    &coeff_a_dir12,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aa_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &zero4,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    &coeff_aa_dir12,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aaa_poly_jet = coeff4_fixed_bilinear(
                    &fx.dc_daaa,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    &coeff_aaa_dir12,
                );

                let mut eta_u_poly_jets = Vec::with_capacity(p);
                let mut chi_u_poly_jets = Vec::with_capacity(p);
                let mut coeff_au_fixed_jets = Vec::with_capacity(p);
                let mut coeff_aau_fixed_jets = Vec::with_capacity(p);
                for u in 0..p {
                    let coeff_u_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_bu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_u_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_bu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_u_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_bbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_abu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_abu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_aabu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_aabu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );

                    let coeff_u_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_u[u],
                        &coeff_u_dir1,
                        &coeff_u_dir2,
                        &coeff_u_dir12,
                    );
                    let coeff_au_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_au[u],
                        &coeff_au_dir1,
                        &coeff_au_dir2,
                        &coeff_au_dir12,
                    );
                    let coeff_aau_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_aau[u],
                        &coeff_aau_dir1,
                        &coeff_aau_dir2,
                        &coeff_aau_dir12,
                    );

                    eta_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&chi_poly_jet, &a_u_jets[u]),
                        &coeff_u_fixed_jet,
                    ));
                    chi_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&eta_aa_poly_jet, &a_u_jets[u]),
                        &coeff_au_fixed_jet,
                    ));
                    coeff_au_fixed_jets.push(coeff_au_fixed_jet);
                    coeff_aau_fixed_jets.push(coeff_aau_fixed_jet);
                }

                for u in 0..p {
                    for v in u..p {
                        let a_uv_jet = MultiDirJet::bilinear(
                            auv[[u, v]],
                            auvd1[[u, v]],
                            auvd2[[u, v]],
                            auvd12[[u, v]],
                        );
                        let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                        let r_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, v),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_mixed_from_bbb_family(
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                        );
                        let chi_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, v),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_mixed_from_bbb_family(
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                        );

                        let eta_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&chi_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_au_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_au_fixed_jets[v], &a_u_jets[u]),
                                    &r_uv_fixed_jet,
                                ),
                            ),
                        );
                        let chi_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&eta_aa_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aaa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_aau_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_aau_fixed_jets[v], &a_u_jets[u]),
                                    &chi_uv_fixed_jet,
                                ),
                            ),
                        );

                        let t1 = chi_uv_poly_jet.clone();
                        let t2 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[v], &eta_poly_jet),
                                &eta_u_poly_jets[u],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t3 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[u], &eta_poly_jet),
                                &eta_u_poly_jets[v],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t4 = poly_scale_jets(
                            &poly_mul_jets(
                                &chi_poly_jet,
                                &poly_add_jets(
                                    &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                                    &poly_mul_jets(&eta_poly_jet, &eta_uv_poly_jet),
                                ),
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t5 = poly_mul_jets(
                            &chi_poly_jet,
                            &poly_mul_jets(
                                &poly_mul_jets(&eta_poly_jet, &eta_poly_jet),
                                &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                            ),
                        );
                        let i_base_jet = poly_add_jets(
                            &poly_add_jets(&poly_add_jets(&t1, &t2), &t3),
                            &poly_add_jets(&t4, &t5),
                        );

                        let i_base = poly_coeff_mask(&i_base_jet, 0);
                        let i_base_d1 = poly_coeff_mask(&i_base_jet, 1);
                        let i_base_d2 = poly_coeff_mask(&i_base_jet, 2);
                        let i_base_d12 = poly_coeff_mask(&i_base_jet, 3);
                        let eta_poly = poly_coeff_mask(&eta_poly_jet, 0);
                        let eta_d1_poly = poly_coeff_mask(&eta_poly_jet, 1);
                        let eta_d2_poly = poly_coeff_mask(&eta_poly_jet, 2);
                        let eta_d12_poly = poly_coeff_mask(&eta_poly_jet, 3);

                        let correction = poly_add(
                            &poly_mul(
                                &poly_add(
                                    &poly_mul(&eta_d2_poly, &eta_d1_poly),
                                    &poly_mul(&eta_poly, &eta_d12_poly),
                                ),
                                &i_base,
                            ),
                            &poly_mul(&poly_mul(&eta_poly, &eta_d1_poly), &i_base_d2),
                        );
                        let full_integrand = poly_sub(&i_base_d12, &correction);
                        let value = exact_kernel::cell_polynomial_integral_from_moments(
                            &full_integrand,
                            &st.moments,
                            "survival D_t second derivative bidirectional",
                        )?;
                        d_uv_uv[[u, v]] += value;
                        if b != 0.0 {
                            let part = &ce.partition_cell;
                            let dir_g1 = dir1[primary.g];
                            let dir_g2 = dir2[primary.g];
                            let edge_velocity = |
                                edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                z: f64,
                            | -> (f64, f64, f64) {
                                match edge {
                                    crate::families::cubic_cell_kernel::PartitionEdge::Crossing {
                                        ..
                                    } => {
                                        let z1 = -(ad1 + z * dir_g1) / b;
                                        let z2 = -(ad2 + z * dir_g2) / b;
                                        let z12 = -(ad12 + z2 * dir_g1 + z1 * dir_g2) / b;
                                        (z1, z2, z12)
                                    }
                                    crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => {
                                        (0.0, 0.0, 0.0)
                                    }
                                }
                            };
                            let density_z_derivative = |z: f64| -> f64 {
                                let eta = cell.eta(z);
                                let eta_z = cell.c1
                                    + 2.0 * cell.c2 * z
                                    + 3.0 * cell.c3 * z * z;
                                let amp = eval_poly_slice(&i_base, z);
                                let amp_z = eval_poly_derivative_slice(&i_base, z);
                                let q_z = z + eta * eta_z;
                                (amp_z - amp * q_z) * (-cell.q(z)).exp()
                                    / std::f64::consts::TAU
                            };
                            let i_d1_poly = poly_sub(
                                &i_base_d1,
                                &poly_mul(&poly_mul(&eta_poly, &eta_d1_poly), &i_base),
                            );
                            let i_d2_poly = poly_sub(
                                &i_base_d2,
                                &poly_mul(&poly_mul(&eta_poly, &eta_d2_poly), &i_base),
                            );
                            let boundary = |z: f64, z1: f64, z2: f64, z12: f64| -> f64 {
                                z12 * crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                    cell, &i_base, z,
                                )
                                    + z2 * crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                        cell, &i_d1_poly, z,
                                    )
                                    + z1 * crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                        cell, &i_d2_poly, z,
                                    )
                                    + z1 * z2 * density_z_derivative(z)
                            };
                            let (r1, r2, r12) = edge_velocity(part.right_edge, cell.right);
                            if r1 != 0.0 || r2 != 0.0 || r12 != 0.0 {
                                d_uv_uv[[u, v]] += boundary(cell.right, r1, r2, r12);
                            }
                            let (l1, l2, l12) = edge_velocity(part.left_edge, cell.left);
                            if l1 != 0.0 || l2 != 0.0 || l12 != 0.0 {
                                d_uv_uv[[u, v]] -= boundary(cell.left, l1, l2, l12);
                            }
                        }
                        d_uv_uv[[v, u]] = d_uv_uv[[u, v]];
                    }
                }
                Ok(d_uv_uv)
            })
            .collect::<Result<Vec<_>, String>>()?;
        for cell_d_uv_uv in d_uv_uv_cell_accums {
            for u in 0..p {
                for v in 0..p {
                    d_uv_uv[[u, v]] += cell_d_uv_uv[[u, v]];
                }
            }
        }

        Ok(SurvivalFlexTimepointBiDirectionalExact {
            eta_uv_uv,
            chi_uv_uv,
            d_uv_uv,
        })
    }
}
