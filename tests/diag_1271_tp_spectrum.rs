//! Diagnostic for #1271: inspect the tp data-metric μ spectrum and the
//! EDF-vs-λ profile on exactly-linear data. Not a pass/fail gate — prints.

use gam::basis::create_thin_plate_spline_basis_with_knot_count;
use ndarray::{s, Array1, Array2};

fn edf_at_lambda(x: &Array2<f64>, s_pen: &Array2<f64>, lambda: f64) -> f64 {
    // EDF = tr((XᵀX + λ S)⁻¹ XᵀX)
    let xtx = x.t().dot(x);
    let mut h = xtx.clone();
    h = &h + &(s_pen * lambda);
    // Solve H F = XᵀX  ->  F = H⁻¹ XᵀX ; EDF = tr(F)
    let h_inv = gam::linalg::utils::invert_spd_with_ridge(&h, 1e-12).expect("invertible H");
    let f = h_inv.dot(&xtx);
    (0..f.nrows()).map(|i| f[[i, i]]).sum()
}

#[test]
fn diag_1271_tp_spectrum_and_edf_profile() {
    let n = 800usize;
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    let data = x.clone().insert_axis(ndarray::Axis(1));

    let (basis, _) = create_thin_plate_spline_basis_with_knot_count(data.view(), 20)
        .expect("tp basis builds");

    let x_design = &basis.basis;
    let s_bend = &basis.penalty_bending;
    let kcols = basis.num_kernel_basis;
    let pcols = basis.num_polynomial_basis;
    println!(
        "[diag] n={n} design p={} kernel_cols={kcols} poly_cols={pcols}",
        x_design.ncols()
    );

    // μ spectrum = diagonal of penalty_bending kernel block.
    let mu: Vec<f64> = (0..kcols).map(|i| s_bend[[i, i]]).collect();
    let mu_max = mu.iter().cloned().fold(0.0_f64, f64::max);
    println!("[diag] mu spectrum (kernel block), max={mu_max:.4e}:");
    for (i, m) in mu.iter().enumerate() {
        println!("   mu[{i:2}] = {:.6e}   (ratio to max = {:.3e})", m, m / mu_max);
    }

    // Off-diagonal energy of penalty (should be ~0 if truly diagonal).
    let mut offdiag = 0.0;
    for i in 0..kcols {
        for j in 0..kcols {
            if i != j {
                offdiag += s_bend[[i, j]].abs();
            }
        }
    }
    println!("[diag] penalty off-diagonal abs-sum = {offdiag:.3e}");

    // Design Gram of kernel block: is BᵀB ≈ I and BᵀP ≈ 0?
    let b = x_design.slice(s![.., 0..kcols]).to_owned();
    let p = x_design.slice(s![.., kcols..]).to_owned();
    let btb = b.t().dot(&b);
    let mut btb_offdiag = 0.0;
    let mut btb_diag_dev = 0.0;
    for i in 0..kcols {
        btb_diag_dev += (btb[[i, i]] - 1.0).abs();
        for j in 0..kcols {
            if i != j {
                btb_offdiag += btb[[i, j]].abs();
            }
        }
    }
    let btp = b.t().dot(&p);
    let btp_abs: f64 = btp.iter().map(|v: &f64| v.abs()).sum();
    println!(
        "[diag] BᵀB diag-dev-from-1 sum={btb_diag_dev:.3e} offdiag-sum={btb_offdiag:.3e}  BᵀP abs-sum={btp_abs:.3e}"
    );

    // EDF-vs-λ profile (full design including unpenalized poly null space).
    let s_full = s_bend.clone();
    println!("[diag] EDF(λ) profile:");
    for &rho in &[0.0_f64, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0] {
        let lambda = rho.exp();
        let edf = edf_at_lambda(x_design, &s_full, lambda);
        println!("   rho={rho:5.1}  lambda={lambda:.3e}  EDF={edf:.4}");
    }
}
