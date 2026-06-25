//! Exact mixed bidirectional extension D_{d1} D_{d2} of the timepoint
//! quantities.
//!
//! Carries the calibration solve, observed-point η/χ jet transport, and the
//! cellwise density-normalization integrand exactly through the implicit
//! intercept solve for a pair of directions.

use super::*;
use crate::families::marginal_slope_shared::SparsePrimaryCoeffJetView;
use crate::families::survival::marginal_slope::flex_oracle_structs_tests::{
    COEFF_SUPPORT_GHW, COEFF_SUPPORT_GW, SurvivalFlexTimepointBiDirectionalExact,
    coeff4_composite_bilinear, coeff4_fixed_bilinear, neg_cell_of, poly_add_jets, poly_coeff_mask,
    poly_mul_jets, poly_scale_jets, scalar_composite_bilinear,
};

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

/// Bilinear-jet `exp` of `(x0, x1, x2, x12)`. With `e = exp(x0)`:
/// `∂₁ = e·x1`, `∂₂ = e·x2`, `∂₁∂₂ = e·(x12 + x1·x2)` (the second Faà di Bruno
/// term `f''·x1·x2 + f'·x12` with `f = f' = f'' = e`).
#[inline]
fn exp_bilinear_jet(x: &MultiDirJet) -> MultiDirJet {
    let x0 = x.coeff(0);
    let x1 = x.coeff(1);
    let x2 = x.coeff(2);
    let x12 = x.coeff(3);
    let e = x0.exp();
    MultiDirJet::bilinear(e, e * x1, e * x2, e * (x12 + x1 * x2))
}

/// Horner evaluation of a polynomial whose coefficients are bilinear jets, at a
/// bilinear-jet point `z` (so the crossing-edge motion of `z` is carried too).
#[inline]
fn eval_poly_jets_at_jet(poly: &[MultiDirJet], z: &MultiDirJet) -> MultiDirJet {
    let mut acc = MultiDirJet::constant(2, 0.0);
    for coeff in poly.iter().rev() {
        acc = acc.mul(z).add(coeff);
    }
    acc
}

/// Horner evaluation of the z-derivative `Σ_k k·c_k·z^{k-1}` for jet
/// coefficients at a jet point `z`.
#[inline]
fn eval_poly_jets_deriv_at_jet(poly: &[MultiDirJet], z: &MultiDirJet) -> MultiDirJet {
    let mut acc = MultiDirJet::constant(2, 0.0);
    for (power, coeff) in poly.iter().enumerate().skip(1).rev() {
        acc = acc.mul(z).add(&coeff.scale(power as f64));
    }
    acc
}

/// Bilinear-jet `Φ(−η)` for a bilinear jet `η`. With `Ψ = Φ(−η0)`,
/// `Ψ_η = −φ(η0)`, `Ψ_ηη = η0·φ(η0)` (the outer stack for the unary compose):
/// `∂₁ = Ψ_η·η1`, `∂₂ = Ψ_η·η2`, `∂₁∂₂ = Ψ_ηη·η1·η2 + Ψ_η·η12`.
#[inline]
fn neg_cdf_bilinear_jet(eta: &MultiDirJet) -> MultiDirJet {
    let e0 = eta.coeff(0);
    let e1 = eta.coeff(1);
    let e2 = eta.coeff(2);
    let e12 = eta.coeff(3);
    let base = crate::probability::normal_cdf(-e0);
    let phi = crate::probability::normal_pdf(e0);
    let psi_eta = -phi;
    let psi_etaeta = e0 * phi;
    MultiDirJet::bilinear(
        base,
        psi_eta * e1,
        psi_eta * e2,
        psi_etaeta * e1 * e2 + psi_eta * e12,
    )
}

impl SurvivalMarginalSlopeFamily {
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

        let (cal_ad1, cal_ad2, cal_ad12) = {
            let mut f_a = 0.0f64;
            let mut f_aa = 0.0f64;
            let mut f_u = Array1::<f64>::zeros(p);
            let mut f_uv = Array2::<f64>::zeros((p, p));
            for ce in &cached.cells {
                let nc = neg_cell_of(ce);
                let st = &ce.state;
                let fx = &ce.fixed;
                let da = fx.dc_da.map(|value| -value);
                let daa = fx.dc_daa.map(|value| -value);
                f_a += exact_kernel::cell_first_derivative_from_moments(&da, &st.moments)?;
                f_aa += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &daa,
                    &st.moments,
                )?;
                for u in 0..p {
                    let cu = fx.coeff_u[u].map(|value| -value);
                    f_u[u] += exact_kernel::cell_first_derivative_from_moments(&cu, &st.moments)?;
                }
                for u in 0..p {
                    for v in u..p {
                        let cu = fx.coeff_u[u].map(|value| -value);
                        let cv = fx.coeff_u[v].map(|value| -value);
                        let cuv = self
                            .cell_pair_second_coeff(primary, &fx.coeff_bu, u, v)
                            .map(|value| -value);
                        let value = exact_kernel::cell_second_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cuv,
                            &st.moments,
                        )?;
                        f_uv[[u, v]] += value;
                        if u != v {
                            f_uv[[v, u]] += value;
                        }
                    }
                }
            }
            let phi_q = crate::probability::normal_pdf(q);
            f_u[q_index] += phi_q;
            f_uv[[q_index, q_index]] += -q * phi_q;
            let inv = 1.0 / f_a;
            let mut au = Array1::<f64>::zeros(p);
            for u in 0..p {
                au[u] = -f_u[u] * inv;
            }
            // Moving-boundary flux on the base intercept-Hessian inputs, so this
            // calibration block's auv (feeding the 4th-order intercept inputs
            // ad1/ad2/ad12) matches the production first_full.rs / directional.rs
            // auv. The crossing z=(τ−a)/b moves with a and every θ, so f_aa and
            // f_uv each pick up their §D flux; the moment-only base omitted
            // them (gam#932/#1454). These feed only scalar outputs here, so no
            // base/dir desync arises.
            if b != 0.0 {
                for entry in &cached.cells {
                    let fx = &entry.fixed;
                    let part = &entry.partition_cell;
                    let cell = part.cell;
                    let da = fx.dc_da.map(|value| -value);
                    // §D self-flux density factor for the calibration F integrand
                    // `G = Φ(−η)·φ(z)` (NOT the bare weight), G_z = −η_z·exp(−q)/2π
                    // − z·Φ(−η)·φ(z). Nonzero only when both crossing velocities
                    // are nonzero (the (g,g)/a-axis diagonals). Mirrors the base
                    // first_full/directional fix (gam#1454).
                    let f_int_z = |z: f64| -> f64 {
                        let eta = cell.eta(z);
                        let eta_z = cell.c1 + 2.0 * cell.c2 * z + 3.0 * cell.c3 * z * z;
                        let exp_q = (-cell.q(z)).exp() / std::f64::consts::TAU;
                        let phi_z = crate::probability::normal_pdf(z);
                        -eta_z * exp_q - z * crate::probability::normal_cdf(-eta) * phi_z
                    };
                    let crossing_vel = |axis: usize,
                                        edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                        z: f64|
                     -> f64 {
                        match edge {
                            crate::families::cubic_cell_kernel::PartitionEdge::Crossing {
                                ..
                            } => {
                                let direct_g = if axis == primary.g { z } else { 0.0 };
                                -direct_g / b
                            }
                            crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
                        }
                    };
                    let a_vel = |edge: crate::families::cubic_cell_kernel::PartitionEdge| -> f64 {
                        match edge {
                            crate::families::cubic_cell_kernel::PartitionEdge::Crossing {
                                ..
                            } => -1.0 / b,
                            crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
                        }
                    };
                    let self_flux = |zx_r: f64, zy_r: f64, zx_l: f64, zy_l: f64| -> f64 {
                        let right = if zx_r != 0.0 && zy_r != 0.0 {
                            zx_r * zy_r * f_int_z(cell.right)
                        } else {
                            0.0
                        };
                        let left = if zx_l != 0.0 && zy_l != 0.0 {
                            zx_l * zy_l * f_int_z(cell.left)
                        } else {
                            0.0
                        };
                        right - left
                    };
                    let za_r = a_vel(part.right_edge);
                    let za_l = a_vel(part.left_edge);
                    // f_aa diagonal: DOUBLED a-axis flux + (a,a) self-flux.
                    f_aa +=
                        2.0 * super::first_full_exact_oracle_tests::moving_density_boundary_flux_a(
                            entry, &da, b,
                        ) + self_flux(za_r, za_r, za_l, za_l);
                    for u in 0..p {
                        let cu = fx.coeff_u[u].map(|value| -value);
                        let zu_r = crossing_vel(u, part.right_edge, cell.right);
                        let zu_l = crossing_vel(u, part.left_edge, cell.left);
                        for v in u..p {
                            let cv = fx.coeff_u[v].map(|value| -value);
                            // Asymmetric flux pair DOUBLED on the diagonal (matching
                            // the base fix); off-diagonal second term is 0 unless v==g.
                            let mut boundary = super::first_full::moving_density_boundary_flux(
                                v, primary, &au, entry, &cu, b, false,
                            ) + super::first_full::moving_density_boundary_flux(
                                u, primary, &au, entry, &cv, b, false,
                            );
                            let zv_r = crossing_vel(v, part.right_edge, cell.right);
                            let zv_l = crossing_vel(v, part.left_edge, cell.left);
                            boundary += self_flux(zu_r, zv_r, zu_l, zv_l);
                            f_uv[[u, v]] += boundary;
                            if u != v {
                                f_uv[[v, u]] += boundary;
                            }
                        }
                    }
                }
            }
            // D-path intercept Hessian (gam#1454). The moment-only `f_au` omits
            // part of its §D moving-boundary flux (the same defect that forced the
            // base/directional `a_uv` onto the D-path), so the F-path
            // `auv = −(f_uv + f_au·au + f_au·au + f_aa·au²)·inv` diverges from the
            // validated Hessian. Reconstruct via the single-source `d_u = ∇(−f_a)`:
            // since `−d_u = f_au_exact + f_aa·au`, substituting yields the identical
            // algebra with the CORRECT `f_au`. This `auv` feeds
            // `cal_ad12 = dir1ᵀ·auv·dir2`, which threads a `cal_ad12·X` term into
            // every second-directional cell total (cd12/cu12/cuv12/…), so the
            // F-path error here was pervasive across all (u,v) blocks. The
            // moment-only `f_au` this reconstruction replaced is no longer
            // accumulated, since the Hessian does not depend on it.
            let d_u = self.survival_flex_base_d_u(primary, &au, cached, b, p)?;
            let mut auv = Array2::<f64>::zeros((p, p));
            for u in 0..p {
                for v in u..p {
                    let value =
                        -(f_uv[[u, v]] - d_u[u] * au[v] - d_u[v] * au[u] - f_aa * au[u] * au[v])
                            * inv;
                    auv[[u, v]] = value;
                    auv[[v, u]] = value;
                }
            }
            let ad1 = au.dot(dir1);
            let ad2 = au.dot(dir2);
            let aud2 = auv.dot(dir2);
            (ad1, ad2, aud2.dot(dir1))
        };

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
                let nc = neg_cell_of(ce);
                let st = &ce.state;
                let fx = &ce.fixed;
                let da = fx.dc_da.map(|v| -v);
                let daa = fx.dc_daa.map(|v| -v);
                let daaa = fx.dc_daaa.map(|v| -v);

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
                let mut caaa1 = [0.0; 4];
                let mut caaa2 = [0.0; 4];
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
                            caaa1[k] -= fx.coeff_aaau[c][k] * dir1[c];
                        }
                        if dir2[c] != 0.0 {
                            cd2[k] -= fx.coeff_u[c][k] * dir2[c];
                            ca2[k] -= fx.coeff_au[c][k] * dir2[c];
                            caa2[k] -= fx.coeff_aau[c][k] * dir2[c];
                            caaa2[k] -= fx.coeff_aaau[c][k] * dir2[c];
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
                let caa12_direct = coeff_view
                    .mixed_directional_from_b_family(&fx.coeff_aabu, dir1, dir2, COEFF_SUPPORT_GHW)
                    .map(|value| -value);
                let mut cd1_total = cd1;
                let mut ca1_total = ca1;
                let mut caa1_total = caa1;
                let mut cd2_total = cd2;
                let mut ca2_total = ca2;
                let mut caa2_total = caa2;
                let mut cd12_total = cd12;
                let mut ca12_total = ca12;
                let mut caa12_total = caa12_direct;
                for k in 0..4 {
                    cd1_total[k] += cal_ad1 * da[k];
                    ca1_total[k] += cal_ad1 * daa[k];
                    caa1_total[k] += cal_ad1 * daaa[k];
                    cd2_total[k] += cal_ad2 * da[k];
                    ca2_total[k] += cal_ad2 * daa[k];
                    caa2_total[k] += cal_ad2 * daaa[k];
                    cd12_total[k] += cal_ad1 * ca2[k]
                        + cal_ad2 * ca1[k]
                        + cal_ad12 * da[k]
                        + cal_ad1 * cal_ad2 * daa[k];
                    ca12_total[k] += cal_ad1 * caa2[k]
                        + cal_ad2 * caa1[k]
                        + cal_ad12 * daa[k]
                        + cal_ad1 * cal_ad2 * daaa[k];
                    caa12_total[k] += cal_ad1 * caaa2[k] + cal_ad2 * caaa1[k] + cal_ad12 * daaa[k];
                }

                f_a_d1 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd1_total,
                    &ca1_total,
                    &st.moments,
                )?;
                f_a_d2 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd2_total,
                    &ca2_total,
                    &st.moments,
                )?;
                f_a_d12 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &cd1_total,
                    &cd2_total,
                    &ca1_total,
                    &ca2_total,
                    &cd12_total,
                    &ca12_total,
                    &st.moments,
                )?;
                f_aa_d1 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1_total,
                    &daa,
                    &ca1_total,
                    &ca1_total,
                    &caa1_total,
                    &st.moments,
                )?;
                f_aa_d2 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd2_total,
                    &daa,
                    &ca2_total,
                    &ca2_total,
                    &caa2_total,
                    &st.moments,
                )?;
                f_aa_d12 += exact_kernel::cell_fourth_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1_total,
                    &cd2_total,
                    &daa,
                    &ca1_total,
                    &ca2_total,
                    &ca1_total,
                    &ca2_total,
                    &cd12_total,
                    &caa1_total,
                    &caa2_total,
                    &ca12_total,
                    &ca12_total,
                    &caa12_total,
                    &st.moments,
                )?;

                for u in 0..p {
                    let cu = fx.coeff_u[u].map(|v| -v);
                    let cau = fx.coeff_au[u].map(|v| -v);
                    let caau = fx.coeff_aau[u].map(|v| -v);
                    let caaau = fx.coeff_aaau[u].map(|v| -v);
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
                    let mut caau1 = [0.0; 4];
                    let mut cu2 = [0.0; 4];
                    let mut cau2 = [0.0; 4];
                    let mut caau2 = [0.0; 4];
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
                        let scaa = self.cell_pair_second_coeff(primary, &fx.coeff_aabu, u, c);
                        for k in 0..4 {
                            if dir1[c] != 0.0 {
                                cu1[k] -= sc[k] * dir1[c];
                                cau1[k] -= sca[k] * dir1[c];
                                caau1[k] -= scaa[k] * dir1[c];
                            }
                            if dir2[c] != 0.0 {
                                cu2[k] -= sc[k] * dir2[c];
                                cau2[k] -= sca[k] * dir2[c];
                                caau2[k] -= scaa[k] * dir2[c];
                            }
                        }
                    }
                    let mut cu1_total = cu1;
                    let mut cau1_total = cau1;
                    let mut cu2_total = cu2;
                    let mut cau2_total = cau2;
                    let mut cu12_total = cu12;
                    let mut cau12_total = cau12;
                    for k in 0..4 {
                        cu1_total[k] += cal_ad1 * cau[k];
                        cau1_total[k] += cal_ad1 * caau[k];
                        cu2_total[k] += cal_ad2 * cau[k];
                        cau2_total[k] += cal_ad2 * caau[k];
                        cu12_total[k] += cal_ad1 * cau2[k]
                            + cal_ad2 * cau1[k]
                            + cal_ad12 * cau[k]
                            + cal_ad1 * cal_ad2 * caau[k];
                        cau12_total[k] += cal_ad1 * caau2[k]
                            + cal_ad2 * caau1[k]
                            + cal_ad12 * caau[k]
                            + cal_ad1 * cal_ad2 * caaau[k];
                    }
                    f_au_d1[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1_total,
                        &cau,
                        &ca1_total,
                        &cu1_total,
                        &cau1_total,
                        &st.moments,
                    )?;
                    f_au_d2[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd2_total,
                        &cau,
                        &ca2_total,
                        &cu2_total,
                        &cau2_total,
                        &st.moments,
                    )?;
                    f_au_d12[u] += exact_kernel::cell_fourth_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1_total,
                        &cd2_total,
                        &cau,
                        &ca1_total,
                        &ca2_total,
                        &cu1_total,
                        &cu2_total,
                        &cd12_total,
                        &cau1_total,
                        &cau2_total,
                        &ca12_total,
                        &cu12_total,
                        &cau12_total,
                        &st.moments,
                    )?;
                }
                for u in 0..p {
                    for v in u..p {
                        let cu = fx.coeff_u[u].map(|x| -x);
                        let cv = fx.coeff_u[v].map(|x| -x);
                        let cau = fx.coeff_au[u].map(|x| -x);
                        let cav = fx.coeff_au[v].map(|x| -x);
                        let caau = fx.coeff_aau[u].map(|x| -x);
                        let caav = fx.coeff_aau[v].map(|x| -x);
                        let sc = self
                            .cell_pair_second_coeff(primary, &fx.coeff_bu, u, v)
                            .map(|x| -x);
                        let sca = self
                            .cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, v)
                            .map(|x| -x);
                        let scaa = self
                            .cell_pair_second_coeff(primary, &fx.coeff_aabu, u, v)
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
                        let mut cau1 = [0.0; 4];
                        let mut cav1 = [0.0; 4];
                        let mut cau2 = [0.0; 4];
                        let mut cav2 = [0.0; 4];
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
                            let sauc = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, c);
                            let savc = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, v, c);
                            for k in 0..4 {
                                if dir1[c] != 0.0 {
                                    cu1[k] -= suc[k] * dir1[c];
                                    cv1[k] -= svc[k] * dir1[c];
                                    cau1[k] -= sauc[k] * dir1[c];
                                    cav1[k] -= savc[k] * dir1[c];
                                }
                                if dir2[c] != 0.0 {
                                    cu2[k] -= suc[k] * dir2[c];
                                    cv2[k] -= svc[k] * dir2[c];
                                    cau2[k] -= sauc[k] * dir2[c];
                                    cav2[k] -= savc[k] * dir2[c];
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
                        let cuv_a1 = coeff_view
                            .pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            )
                            .map(|value| -value);
                        let cuv_a2 = coeff_view
                            .pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            )
                            .map(|value| -value);
                        let mut cu1_total = cu1;
                        let mut cv1_total = cv1;
                        let mut cu2_total = cu2;
                        let mut cv2_total = cv2;
                        let mut cuv1_total = cuv1;
                        let mut cuv2_total = cuv2;
                        let mut cu12_total = cu12;
                        let mut cv12_total = cv12;
                        let mut cuv12_total = cuv12;
                        for k in 0..4 {
                            cu1_total[k] += cal_ad1 * cau[k];
                            cv1_total[k] += cal_ad1 * cav[k];
                            cu2_total[k] += cal_ad2 * cau[k];
                            cv2_total[k] += cal_ad2 * cav[k];
                            cuv1_total[k] += cal_ad1 * sca[k];
                            cuv2_total[k] += cal_ad2 * sca[k];
                            cu12_total[k] += cal_ad1 * cau2[k]
                                + cal_ad2 * cau1[k]
                                + cal_ad12 * cau[k]
                                + cal_ad1 * cal_ad2 * caau[k];
                            cv12_total[k] += cal_ad1 * cav2[k]
                                + cal_ad2 * cav1[k]
                                + cal_ad12 * cav[k]
                                + cal_ad1 * cal_ad2 * caav[k];
                            cuv12_total[k] += cal_ad1 * cuv_a2[k]
                                + cal_ad2 * cuv_a1[k]
                                + cal_ad12 * sca[k]
                                + cal_ad1 * cal_ad2 * scaa[k];
                        }
                        let d1v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1_total,
                            &sc,
                            &cu1_total,
                            &cv1_total,
                            &cuv1_total,
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
                            &cd2_total,
                            &sc,
                            &cu2_total,
                            &cv2_total,
                            &cuv2_total,
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
                            &cd1_total,
                            &cd2_total,
                            &sc,
                            &cu1_total,
                            &cu2_total,
                            &cv1_total,
                            &cv2_total,
                            &cd12_total,
                            &cuv1_total,
                            &cuv2_total,
                            &cu12_total,
                            &cv12_total,
                            &cuv12_total,
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

        // §D moving-boundary flux on the base intercept-Hessian inputs and their
        // two directional derivatives, carried as bilinear jets. first_full.rs's
        // base `f_uv`/`f_aa`/`f_au` carry the calibration-F moving-boundary flux
        // (asymmetric Leibniz pair + doubled-diagonal + `G_z·z_x·z_y` self-flux),
        // and directional.rs carries its single `D_dir`. The bidirectional pass
        // recovers `auv`/`auvd1`/`auvd2`/`auvd12` from the SAME F-path Hessian
        // inputs, so each `f_uv`/`f_aa`/`f_au` (and its d1/d2/d12) must carry the
        // flux through the bilinear `(dir1, dir2)` jet — exactly D_dir1 D_dir2 of
        // the base block. Omitting it left the F-path inputs short of the §D total
        // derivative the D-path `d_u` carries, so the recovered `auvd12` (hence
        // the contracted fourth `fourth[g,β_w]`) was wrong (gam#1454). The
        // intercept directional jets are the flux-corrected calibration scalars
        // `cal_ad1`/`cal_ad2`/`cal_ad12` (= a-Hessian contracted with dir1, dir2),
        // available before this recovery, so no base/dir desync arises.
        if b != 0.0 {
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
            let dir_g1 = if primary.g < p { dir1[primary.g] } else { 0.0 };
            let dir_g2 = if primary.g < p { dir2[primary.g] } else { 0.0 };
            let inv_b = 1.0 / b;
            for entry in &cached.cells {
                let fx = &entry.fixed;
                let part = &entry.partition_cell;
                let cell = part.cell;
                let eta_base = [cell.c0, cell.c1, cell.c2, cell.c3];
                // η composite jet (deviation + intercept chain); the calibration
                // F integrand is `G = Φ(−η)·φ(z)`.
                let eta_poly_jet = coeff4_composite_bilinear(
                    &eta_base,
                    &fx.dc_da,
                    &fx.dc_daa,
                    &primary_view.directional_family(&fx.coeff_u, dir1, COEFF_SUPPORT_GHW),
                    &primary_view.directional_family(&fx.coeff_u, dir2, COEFF_SUPPORT_GHW),
                    &primary_view.mixed_directional_from_b_family(
                        &fx.coeff_bu,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    ),
                    &primary_view.directional_family(&fx.coeff_au, dir1, COEFF_SUPPORT_GHW),
                    &primary_view.directional_family(&fx.coeff_au, dir2, COEFF_SUPPORT_GHW),
                    cal_ad1,
                    cal_ad2,
                    cal_ad12,
                );
                // χ = ∂η/∂a composite jet (for `neg_dc_da` = −χ flux poly).
                let neg_dc_da_jet: Vec<MultiDirJet> = coeff4_composite_bilinear(
                    &fx.dc_da,
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &primary_view.directional_family(&fx.coeff_au, dir1, COEFF_SUPPORT_GHW),
                    &primary_view.directional_family(&fx.coeff_au, dir2, COEFF_SUPPORT_GHW),
                    &primary_view.mixed_directional_from_b_family(
                        &fx.coeff_abu,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    ),
                    &primary_view.directional_family(&fx.coeff_aau, dir1, COEFF_SUPPORT_GHW),
                    &primary_view.directional_family(&fx.coeff_aau, dir2, COEFF_SUPPORT_GHW),
                    cal_ad1,
                    cal_ad2,
                    cal_ad12,
                )
                .into_iter()
                .map(|c| c.scale(-1.0))
                .collect();
                // Per-axis −∂η/∂θ_u total directional bilinear jets (deviation +
                // intercept chain), via the composite-bilinear builder with the
                // a-derivative stack `coeff_au`/`coeff_aau` and the partial-
                // directional `coeff_bu`/`coeff_bbu`/`coeff_abu` families.
                let neg_coeff_u_jets: Vec<Vec<MultiDirJet>> = (0..p)
                    .map(|u| {
                        coeff4_composite_bilinear(
                            &fx.coeff_u[u],
                            &fx.coeff_au[u],
                            &fx.coeff_aau[u],
                            &primary_view.param_directional_from_b_family(
                                &fx.coeff_bu,
                                u,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.param_directional_from_b_family(
                                &fx.coeff_bu,
                                u,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.param_mixed_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.param_directional_from_b_family(
                                &fx.coeff_abu,
                                u,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.param_directional_from_b_family(
                                &fx.coeff_abu,
                                u,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            cal_ad1,
                            cal_ad2,
                            cal_ad12,
                        )
                        .into_iter()
                        .map(|c| c.scale(-1.0))
                        .collect()
                    })
                    .collect();
                // Crossing-edge position jet `Z = (z, z1, z2, z12)`; the §C edge
                // velocities `z_k = −(a_dk + z·dir_gk)/b` and mixed
                // `z12 = −(a_d12 + z2·dir_g1 + z1·dir_g2)/b`, evaluated PER edge.
                let edge_pos_jet = |edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                    z: f64|
                 -> Option<MultiDirJet> {
                    match edge {
                        crate::families::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {
                            let z1 = -(cal_ad1 + z * dir_g1) * inv_b;
                            let z2 = -(cal_ad2 + z * dir_g2) * inv_b;
                            let z12 = -(cal_ad12 + z2 * dir_g1 + z1 * dir_g2) * inv_b;
                            Some(MultiDirJet::bilinear(z, z1, z2, z12))
                        }
                        crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => None,
                    }
                };
                // `1/b` as a bilinear jet: `b = g + const` moves with dir1/dir2
                // (gam#1454), so `D_dk(1/b) = −dir_gk/b²` and the mixed
                // `D_d1 D_d2(1/b) = 2·dir_g1·dir_g2/b³`. The velocities below carry
                // this b-motion exactly (the directional `z_axis_dir` term
                // `−z_axis·dir_g/b` is precisely this `1/b`-derivative).
                let inv_b_jet =
                    reciprocal_bilinear_jet(MultiDirJet::bilinear(b, dir_g1, dir_g2, 0.0));
                // Partial-IFT θ-axis crossing-velocity jet `z_x = −direct_g_x/b`,
                // with `direct_g_x = δ_{x,g}·Z` (the moving crossing). a held fixed
                // (the intercept chain is supplied by the explicit f_au·au + f_aa·au²
                // recovery terms), so it carries NO a_dk term — matching the base
                // first_full/directional `edge_vel` (gam#1454).
                let theta_vel_jet = |axis: usize, z_jet: &MultiDirJet| -> MultiDirJet {
                    if axis == primary.g {
                        z_jet.mul(&inv_b_jet).scale(-1.0)
                    } else {
                        MultiDirJet::constant(2, 0.0)
                    }
                };
                // a-axis velocity `z_a = −1/b` (the b-motion makes this a nonzero
                // jet under dir1/dir2 even though it is θ-independent at fixed b).
                let a_vel_jet = inv_b_jet.scale(-1.0);
                // Density weight jet `exp(−q)/2π = φ(η)φ(z)` at the moving edge,
                // q = ½(z² + η²).
                let weight_jet = |z_jet: &MultiDirJet, eta_jet: &MultiDirJet| -> MultiDirJet {
                    let z2 = z_jet.mul(z_jet);
                    let eta2 = eta_jet.mul(eta_jet);
                    let neg_q = z2.add(&eta2).scale(-0.5);
                    exp_bilinear_jet(&neg_q).scale(std::f64::consts::FRAC_1_PI * 0.5)
                };
                // `cell_density_boundary_integrand(cell, poly, z) = poly(z)·w(z)`
                // as a bilinear jet, with both `poly` and the edge `z`/η moving.
                let boundary_integrand_jet = |poly: &[MultiDirJet],
                                              z_jet: &MultiDirJet,
                                              weight: &MultiDirJet|
                 -> MultiDirJet {
                    eval_poly_jets_at_jet(poly, z_jet).mul(weight)
                };
                // `G_z` of the calibration F integrand `G = Φ(−η)·φ(z)`:
                //   G_z = −η_z·φ(η)φ(z) − z·Φ(−η)·φ(z),
                // every factor a bilinear jet at the moving edge (η_z = ∂η/∂z).
                let f_int_z_jet = |z_jet: &MultiDirJet,
                                   eta_jet: &MultiDirJet,
                                   eta_z_jet: &MultiDirJet,
                                   weight: &MultiDirJet|
                 -> MultiDirJet {
                    let phi_z = {
                        let z2 = z_jet.mul(z_jet);
                        exp_bilinear_jet(&z2.scale(-0.5))
                            .scale(1.0 / (2.0 * std::f64::consts::PI).sqrt())
                    };
                    let cdf = neg_cdf_bilinear_jet(eta_jet);
                    eta_z_jet
                        .mul(weight)
                        .scale(-1.0)
                        .sub(&z_jet.mul(&cdf).mul(&phi_z))
                };
                // Sum `right − left` of an edge functional over the cell's two
                // crossing edges (skips Fixed edges, where every velocity is 0).
                let edge_diff = |f: &dyn Fn(&MultiDirJet, f64) -> MultiDirJet| -> MultiDirJet {
                    let mut acc = MultiDirJet::constant(2, 0.0);
                    if let Some(zr) = edge_pos_jet(part.right_edge, cell.right) {
                        acc = acc.add(&f(&zr, cell.right));
                    }
                    if let Some(zl) = edge_pos_jet(part.left_edge, cell.left) {
                        acc = acc.sub(&f(&zl, cell.left));
                    }
                    acc
                };
                // Asymmetric Leibniz flux `z_axis · integrand(poly)`.
                let theta_flux = |axis: usize, poly: &[MultiDirJet]| -> MultiDirJet {
                    edge_diff(&|z_jet, _z| {
                        let eta_jet = eval_poly_jets_at_jet(&eta_poly_jet, z_jet);
                        let weight = weight_jet(z_jet, &eta_jet);
                        theta_vel_jet(axis, z_jet)
                            .mul(&boundary_integrand_jet(poly, z_jet, &weight))
                    })
                };
                let a_flux = |poly: &[MultiDirJet]| -> MultiDirJet {
                    edge_diff(&|z_jet, _z| {
                        let eta_jet = eval_poly_jets_at_jet(&eta_poly_jet, z_jet);
                        let weight = weight_jet(z_jet, &eta_jet);
                        a_vel_jet.mul(&boundary_integrand_jet(poly, z_jet, &weight))
                    })
                };
                // Symmetric self-flux `G_z·z_x·z_y` (velocities given as a closure
                // of the edge position jet).
                let self_flux = |vel_x: &dyn Fn(&MultiDirJet) -> MultiDirJet,
                                 vel_y: &dyn Fn(&MultiDirJet) -> MultiDirJet|
                 -> MultiDirJet {
                    edge_diff(&|z_jet, _z| {
                        let eta_jet = eval_poly_jets_at_jet(&eta_poly_jet, z_jet);
                        let eta_z_jet = eval_poly_jets_deriv_at_jet(&eta_poly_jet, z_jet);
                        let weight = weight_jet(z_jet, &eta_jet);
                        let gz = f_int_z_jet(z_jet, &eta_jet, &eta_z_jet, &weight);
                        gz.mul(&vel_x(z_jet)).mul(&vel_y(z_jet))
                    })
                };
                let theta_v = |axis: usize| move |z_jet: &MultiDirJet| theta_vel_jet(axis, z_jet);
                let a_v = |z_jet: &MultiDirJet| z_jet.scale(0.0).add(&a_vel_jet);

                // f_a: the base intercept-density's §D moving-boundary flux,
                // `D_dir(f_a) ⊇ z_dir·integrand(neg_dc_da)`, the bilinear analog
                // of directional.rs's `f_a_dir += first_boundary(&neg_dc_da)`
                // (~line 824). The base f_a has NO velocity (the boundary is at
                // rest at zeroth order), so only the MOTION of the edge under the
                // (dir1, dir2) jet contributes: take the edge-position jet's
                // motion part `z_jet − z_jet.coeff(0)` (coeff(0)=0) as the
                // Leibniz velocity and multiply by the density integrand of
                // `neg_dc_da`. Its coeff(1)/(2)/(3) are the d1/d2/d12 fluxes
                // f_a was missing; without them `f_a_d12` (→ `d_jet`'s
                // `−f_a_d12`) was short of D_dir1 D_dir2(f_a), corrupting the
                // `auvd12` recovery and the contracted fourth `fourth[q0,g]`
                // (gam#1454).
                {
                    let block = edge_diff(&|z_jet, _z| {
                        let motion = z_jet.sub(&MultiDirJet::constant(2, z_jet.coeff(0)));
                        let eta_jet = eval_poly_jets_at_jet(&eta_poly_jet, z_jet);
                        let weight = weight_jet(z_jet, &eta_jet);
                        motion.mul(&boundary_integrand_jet(&neg_dc_da_jet, z_jet, &weight))
                    });
                    f_a += block.coeff(0);
                    f_a_d1 += block.coeff(1);
                    f_a_d2 += block.coeff(2);
                    f_a_d12 += block.coeff(3);
                }

                // f_aa: doubled a-axis flux + (a,a) self-flux.
                {
                    let block = a_flux(&neg_dc_da_jet)
                        .scale(2.0)
                        .add(&self_flux(&a_v, &a_v));
                    f_aa += block.coeff(0);
                    f_aa_d1 += block.coeff(1);
                    f_aa_d2 += block.coeff(2);
                    f_aa_d12 += block.coeff(3);
                }
                // f_au[u]: a-axis flux on coeff_u + u-axis flux on dc_da +
                // (a,u) self-flux.
                for u in 0..p {
                    let block = a_flux(&neg_coeff_u_jets[u])
                        .add(&theta_flux(u, &neg_dc_da_jet))
                        .add(&self_flux(&a_v, &theta_v(u)));
                    f_au[u] += block.coeff(0);
                    f_au_d1[u] += block.coeff(1);
                    f_au_d2[u] += block.coeff(2);
                    f_au_d12[u] += block.coeff(3);
                }
                // f_uv[u,v]: asymmetric flux pair (DOUBLED on the diagonal) +
                // (u,v) self-flux.
                for u in 0..p {
                    for v in u..p {
                        let block = theta_flux(v, &neg_coeff_u_jets[u])
                            .add(&theta_flux(u, &neg_coeff_u_jets[v]))
                            .add(&self_flux(&theta_v(u), &theta_v(v)));
                        f_uv[[u, v]] += block.coeff(0);
                        f_uv_d1[[u, v]] += block.coeff(1);
                        f_uv_d2[[u, v]] += block.coeff(2);
                        f_uv_d12[[u, v]] += block.coeff(3);
                        if u != v {
                            f_uv[[v, u]] += block.coeff(0);
                            f_uv_d1[[v, u]] += block.coeff(1);
                            f_uv_d2[[v, u]] += block.coeff(2);
                            f_uv_d12[[v, u]] += block.coeff(3);
                        }
                    }
                }
            }
        }

        let inv = 1.0 / f_a;
        let mut au = Array1::<f64>::zeros(p);
        for u in 0..p {
            au[u] = -f_u[u] * inv;
        }
        // D-PATH intercept-Hessian recovery (matching first_full.rs/directional.rs,
        // gam#1454): `a_uv = (f_uv − d_u[u]·a_u[v] − d_u[v]·a_u[u] − f_aa·a_u·a_u)/D`
        // with `D = −f_a`. The F-path `f_au` does NOT reconstruct the §D total
        // `d_u` (its moving-boundary partials differ), so the bidirectional —
        // exactly like the base and directional passes — recovers from the
        // FD-validated `d_u`/`d_uv` (and their directional extension `d_uv_dir`)
        // instead. The second-directional Hessian `auvd12` then matches the
        // directional path one tier up, closing the contracted fourth
        // `fourth[g,β_w]` (gam#1454). `f_uv`/`f_aa` carry the §D flux added above.
        let d_check = -f_a;
        let inv_d = 1.0 / d_check;
        let d_u_base = self.survival_flex_base_d_u(primary, &au, cached, b, p)?;
        let mut auv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val = (f_uv[[u, v]]
                    - d_u_base[u] * au[v]
                    - d_u_base[v] * au[u]
                    - f_aa * au[u] * au[v])
                    * inv_d;
                auv[[u, v]] = val;
                auv[[v, u]] = val;
            }
        }
        let ad1 = au.dot(dir1);
        let ad2 = au.dot(dir2);
        let aud1 = auv.dot(dir1);
        let aud2 = auv.dot(dir2);

        // Directional `d_u` derivatives via the FD-validated `d_uv` contraction.
        let d_uv_base = self.survival_flex_base_d_uv(primary, &au, &auv, cached, b, p)?;
        let d_u_d1 = d_uv_base.dot(dir1);
        let d_u_d2 = d_uv_base.dot(dir2);
        // D directional derivatives (D = −f_a).
        let d_d1 = -f_a_d1;
        let d_d2 = -f_a_d2;
        // D-path single-directional Hessian derivatives `auvd1`/`auvd2`
        // (`a_uv_dir = (N_dir − a_uv·D_dir)/D`, N = the D-path numerator).
        let mut auvd1 = Array2::<f64>::zeros((p, p));
        let mut auvd2 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n1 = f_uv_d1[[u, v]]
                    - d_u_d1[u] * au[v]
                    - d_u_base[u] * aud1[v]
                    - d_u_d1[v] * au[u]
                    - d_u_base[v] * aud1[u]
                    - f_aa_d1 * au[u] * au[v]
                    - f_aa * (aud1[u] * au[v] + au[u] * aud1[v]);
                let v1 = (n1 - d_d1 * auv[[u, v]]) * inv_d;
                auvd1[[u, v]] = v1;
                auvd1[[v, u]] = v1;

                let n2 = f_uv_d2[[u, v]]
                    - d_u_d2[u] * au[v]
                    - d_u_base[u] * aud2[v]
                    - d_u_d2[v] * au[u]
                    - d_u_base[v] * aud2[u]
                    - f_aa_d2 * au[u] * au[v]
                    - f_aa * (aud2[u] * au[v] + au[u] * aud2[v]);
                let v2 = (n2 - d_d2 * auv[[u, v]]) * inv_d;
                auvd2[[u, v]] = v2;
                auvd2[[v, u]] = v2;
            }
        }

        let ad12 = aud2.dot(dir1);
        // Mixed second-directional `d_u` derivative `D_dir1 D_dir2(d_u)` via the
        // FD-validated directional `d_uv_dir` (= `D_dir1(d_uv)`), contracted with
        // dir2: `d_u_d12[u] = Σ_v d_uv_dir1[u,v]·dir2[v] = D_dir1 D_dir2(d_u[u])`.
        let dir1_ext = self.compute_survival_timepoint_directional_exact_from_cached(
            row, primary, q, q_index, a, b, beta_h, beta_w, cached, dir1, true,
        )?;
        // aud12 = D_dir1 D_dir2 a_u, single-sourced from the EXACT directional
        // a_uv_dir (mirroring d_u_d12); the prior local `auvd2.dot(dir1)` used the
        // bidirectional's own auvd2 re-derivation, which diverged at the q0/warp
        // rows and corrupted auvd12 -> eta_uv_uv -> the #1454 gate (gam#1454).
        let aud12 = dir1_ext.a_uv_dir.dot(dir2);
        let d_u_d12 = dir1_ext.d_uv_dir.dot(dir2);

        // a_u and d_u bilinear jets carry (·, D_dir1, D_dir2, D_dir1 D_dir2).
        let a_u_jets: Vec<MultiDirJet> = (0..p)
            .map(|u| MultiDirJet::bilinear(au[u], aud1[u], aud2[u], aud12[u]))
            .collect();
        let d_u_jets: Vec<MultiDirJet> = (0..p)
            .map(|u| MultiDirJet::bilinear(d_u_base[u], d_u_d1[u], d_u_d2[u], d_u_d12[u]))
            .collect();
        // D-path bilinear recovery of the second-directional Hessian `auvd12`:
        // `a_uv = (f_uv − d_u_u·a_u_v − d_u_v·a_u_u − f_aa·a_u_u·a_u_v)/D`, every
        // factor a bilinear jet; coeff(3) is `D_dir1 D_dir2(a_uv)`.
        let d_jet = MultiDirJet::bilinear(d_check, d_d1, d_d2, -f_a_d12);
        let d_recip_jet = reciprocal_bilinear_jet(d_jet);
        let f_aa_jet = MultiDirJet::bilinear(f_aa, f_aa_d1, f_aa_d2, f_aa_d12);
        let mut auvd12 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let f_uv_jet = MultiDirJet::bilinear(
                    f_uv[[u, v]],
                    f_uv_d1[[u, v]],
                    f_uv_d2[[u, v]],
                    f_uv_d12[[u, v]],
                );
                let a_u_jet = &a_u_jets[u];
                let a_v_jet = &a_u_jets[v];
                let numerator = f_uv_jet
                    .sub(&d_u_jets[u].mul(a_v_jet))
                    .sub(&d_u_jets[v].mul(a_u_jet))
                    .sub(&f_aa_jet.mul(&a_u_jet.mul(a_v_jet)));
                let val = numerator.mul(&d_recip_jet).coeff(3);
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
        let eta_aaa_d1 = eval_coeff4_at(
            &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
            z_obs,
        );
        let eta_aaa_d2 = eval_coeff4_at(
            &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
            z_obs,
        );
        // eta_aaa second mixed (dir1,dir2) directional. A g-cross closed form
        // (−3·dir1_g·η_aaa_d2 − …) was tried (gam#1454) but is WRONG at the q0/q1
        // intercept slots: η_aaa is evaluated at z_obs (the observed score
        // projection), which itself depends on g/w, so the b⁻³-scaling argument
        // misses the ∂z_obs/∂dir chain and blew chi_uv_uv[0,0] to ~16 (validly
        // measured on an isolated target). Restored to 0.0 (the known-open value);
        // the correct term needs the genuine 2nd-directional of aaa_first (new
        // g_jet family), not a closed form. This slot does NOT affect the gate
        // (eta_uv_uv q0-row), which is the real #1454 driver.
        let eta_aaa_jet = MultiDirJet::bilinear(eta_aaa, eta_aaa_d1, eta_aaa_d2, 0.0);

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
                // The second mixed (dir1,dir2) derivative of `tau_a` (base
                // ∂²_a∂_w for param u) adds two more `b`-derivatives, i.e.
                // ∂²_a∂²_b∂_w of the cell coefficient — a 4th-order total
                // (a,b)-partial of a degree-3 cell polynomial, hence identically
                // zero (the old `abb_first` = ∂_a∂²_b∂_w slot was the `tau` d12
                // term reused one a-order too low).
                0.0,
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
                    // `chi_uv_fixed = ∂_a r_uv`, so its second mixed (dir1,dir2)
                    // derivative is ∂_a∂³_b∂_w of the cell coefficient — a 4th-order
                    // total (a,b)-partial of a degree-3 cell polynomial, hence
                    // identically zero (the old `bbb_first` = ∂³_b∂_w slot was the
                    // `r_uv` d12 term carried over without the extra ∂_a that turns
                    // it into a vanishing 4th total derivative).
                    0.0,
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
                // Second mixed (dir1,dir2) derivative of the ∂³_a cell coefficient =
                // ∂³_a∂_b∂_w (b,w cross) / ∂³_a∂²_b (b,b cross): both are ≥4th-order
                // total (a,b)-partials of a degree-3 cell polynomial, hence zero.
                // (`coeff_aabu` = ∂²_a∂_b∂_w was the `coeff_aa` d12 family reused one
                // a-order too low.)
                let coeff_aaa_dir12 = zero4;

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
                    // Second mixed (dir1,dir2) derivative of the per-param ∂²_a cell
                    // coefficient = ∂²_a∂²_b∂_w — a 4th-order total (a,b)-partial of
                    // a degree-3 cell polynomial, hence zero. (`coeff_abbu`
                    // = ∂_a∂²_b∂_w was the `coeff_au` d12 family reused one a-order
                    // too low.)
                    let coeff_aau_dir12 = zero4;

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
                            // `chi_uv_fixed = ∂_a r_uv`, so its second mixed
                            // (dir1,dir2) derivative is ∂_a∂³_b∂_w — a 4th-order
                            // total (a,b)-partial of a degree-3 cell polynomial,
                            // hence identically zero. (`coeff_bbbu` = ∂³_b∂_w was the
                            // `r_uv` d12 family carried over without the extra ∂_a.)
                            &zero4,
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

                        let density_curvature = poly_mul(
                            &poly_add(
                                &poly_mul(&eta_d1_poly, &eta_d2_poly),
                                &poly_mul(&eta_poly, &eta_d12_poly),
                            ),
                            &i_base,
                        );
                        let weight_d1_times_i_d2 =
                            poly_mul(&poly_mul(&eta_poly, &eta_d1_poly), &i_base_d2);
                        let weight_d2_times_i_d1 =
                            poly_mul(&poly_mul(&eta_poly, &eta_d2_poly), &i_base_d1);
                        let second_weight_product = poly_mul(
                            &poly_mul(&poly_mul(&eta_poly, &eta_poly), &eta_d1_poly),
                            &poly_mul(&eta_d2_poly, &i_base),
                        );
                        let full_integrand = poly_add(
                            &poly_sub(
                                &poly_sub(
                                    &poly_sub(&i_base_d12, &density_curvature),
                                    &weight_d1_times_i_d2,
                                ),
                                &weight_d2_times_i_d1,
                            ),
                            &second_weight_product,
                        );
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

                            // `D_dir1 D_dir2(B)`: the base `d_uv = I + B`, and the
                            // block above only carries `D²(I)` (interior `value` +
                            // its moving-boundary). `B` is `survival_flex_base_d_uv`'s
                            // OWN §D moving boundary (first_full.rs `edge_term`):
                            //   B = part_a + part_b1 + G0_z·z_u·z_v + G0·z_uv,
                            // with `G0 = χ·w`, `part_a = (∂_u d_u-integrand)·z_v`,
                            // `part_b1 = (∂_v d_u-integrand)·z_u`, the BARE-cell-weight
                            // velocities `z_x = −(a_u[x] + δ_{x,g}·z)/b` and the cross
                            // `z_uv = −(a_uv + z_u·δ_{v,g} + z_v·δ_{u,g})/b`. Promote
                            // every factor to a bilinear (dir1, dir2) jet — the edge
                            // POSITION jet `Z` carries the §C contraction motion, the
                            // composite η jet `eta_poly_jet` carries the cell-coeff
                            // motion (so the weight's `q_z = Z + η·η_z` shifts with the
                            // direction), and `a_u_jets`/`a_uv_jet` carry the intercept
                            // chain — then `B.coeff(3) = D_dir1 D_dir2(B)`. Dropping it
                            // left `d_uv_uv` short of `D²(d_uv)` on the g-touching
                            // blocks (the dominant `fourth[g,h0]` residual, gam#1454).
                            let dir_g1j = dir1[primary.g];
                            let dir_g2j = dir2[primary.g];
                            let inv_b_jet = reciprocal_bilinear_jet(MultiDirJet::bilinear(
                                b, dir_g1j, dir_g2j, 0.0,
                            ));
                            let neg_inv_b_jet = inv_b_jet.scale(-1.0);
                            let ug = if u == primary.g { 1.0 } else { 0.0 };
                            let vg = if v == primary.g { 1.0 } else { 0.0 };
                            // d_u interior integrand polys (poly-of-jets):
                            //   d_u-integrand[x] = χ_u[x] − χ·η·η_u[x].
                            let d_u_int_u = poly_add_jets(
                                &chi_u_poly_jets[u],
                                &poly_scale_jets(
                                    &poly_mul_jets(
                                        &poly_mul_jets(&chi_poly_jet, &eta_poly_jet),
                                        &eta_u_poly_jets[u],
                                    ),
                                    &MultiDirJet::constant(2, -1.0),
                                ),
                            );
                            let d_u_int_v = poly_add_jets(
                                &chi_u_poly_jets[v],
                                &poly_scale_jets(
                                    &poly_mul_jets(
                                        &poly_mul_jets(&chi_poly_jet, &eta_poly_jet),
                                        &eta_u_poly_jets[v],
                                    ),
                                    &MultiDirJet::constant(2, -1.0),
                                ),
                            );
                            let weight_at = |zj: &MultiDirJet| -> MultiDirJet {
                                let eta = eval_poly_jets_at_jet(&eta_poly_jet, zj);
                                let z2 = zj.mul(zj);
                                let eta2 = eta.mul(&eta);
                                let neg_q = z2.add(&eta2).scale(-0.5);
                                exp_bilinear_jet(&neg_q)
                                    .scale(std::f64::consts::FRAC_1_PI * 0.5)
                            };
                            let edge_term_jet = |edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                                 zb: f64,
                                                 z1: f64,
                                                 z2: f64,
                                                 z12: f64|
                             -> MultiDirJet {
                                match edge {
                                    crate::families::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {}
                                    crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => {
                                        return MultiDirJet::constant(2, 0.0);
                                    }
                                }
                                // Edge-position jet (base + §C contraction motion).
                                let zj = MultiDirJet::bilinear(zb, z1, z2, z12);
                                // Bare-cell-weight base velocities `z_x = −(a_u[x] +
                                // δ_{x,g}·Z)/b`, the intercept-chain crossing motion.
                                let direct_u =
                                    if u == primary.g { zj.clone() } else { MultiDirJet::constant(2, 0.0) };
                                let direct_v =
                                    if v == primary.g { zj.clone() } else { MultiDirJet::constant(2, 0.0) };
                                let zu = a_u_jets[u].add(&direct_u).mul(&neg_inv_b_jet);
                                let zv = a_u_jets[v].add(&direct_v).mul(&neg_inv_b_jet);
                                let zuv = a_uv_jet
                                    .add(&zu.mul(&MultiDirJet::constant(2, vg)))
                                    .add(&zv.mul(&MultiDirJet::constant(2, ug)))
                                    .mul(&neg_inv_b_jet);
                                let weight = weight_at(&zj);
                                let g0 = eval_poly_jets_at_jet(&chi_poly_jet, &zj).mul(&weight);
                                let part_a =
                                    eval_poly_jets_at_jet(&d_u_int_u, &zj).mul(&weight).mul(&zv);
                                let part_b1 =
                                    eval_poly_jets_at_jet(&d_u_int_v, &zj).mul(&weight).mul(&zu);
                                // G0_z = ∂_z(χ·w) = (χ_z − χ·q_z)·w, q_z = Z + η·η_z.
                                let chi_val = eval_poly_jets_at_jet(&chi_poly_jet, &zj);
                                let chi_z = eval_poly_jets_deriv_at_jet(&chi_poly_jet, &zj);
                                let eta_val = eval_poly_jets_at_jet(&eta_poly_jet, &zj);
                                let eta_z = eval_poly_jets_deriv_at_jet(&eta_poly_jet, &zj);
                                let q_z = zj.add(&eta_val.mul(&eta_z));
                                let g0_z = chi_z.sub(&chi_val.mul(&q_z)).mul(&weight);
                                part_a
                                    .add(&part_b1)
                                    .add(&g0_z.mul(&zu).mul(&zv))
                                    .add(&g0.mul(&zuv))
                            };
                            let b_jet = edge_term_jet(part.right_edge, cell.right, r1, r2, r12).sub(
                                &edge_term_jet(part.left_edge, cell.left, l1, l2, l12),
                            );
                            d_uv_uv[[u, v]] += b_jet.coeff(3);
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
