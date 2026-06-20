//! Diagnostic (#1376): isolate WHICH term of the Matérn log-κ outer-gradient is
//! wrong. The full-criterion FD audit shows psi_kappa[0] analytic=32.07 vs
//! fd=8.51 (ratio 3.77). This test FD-checks the DESIGN ψ-derivative ∂X/∂(log κ)
//! ALONE (analytic `materialize_first` / `design_derivative` vs a central
//! difference of the rebuilt Matérn design at κ±ε), exactly the way the passing
//! Duchon test `..._frozen_design` does. If the Matérn design ψ-derivative
//! MATCHES FD here, the bug is NOT in the design derivative — it's downstream in
//! the outer dH/dψ / a_i assembly (hyper.rs). If it MISMATCHES, the design
//! ψ-derivative formula (q·s_a aggregation for the iso-κ coordinate) is the bug.
//! NOT a gate — run with --nocapture.

use gam::terms::basis::{
    build_matern_basis, build_matern_basis_log_kappa_derivatives, CenterStrategy, MaternBasisSpec,
    MaternNu,
};
use ndarray::Array2;

fn dataset() -> Array2<f64> {
    // 2-D ordinary surface support, deterministic.
    let n = 40usize;
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        v.push(-1.0 + 2.0 * t);
        v.push((3.0 * t).sin());
    }
    Array2::from_shape_vec((n, 2), v).unwrap()
}

fn spec_at(data: &Array2<f64>, rho: f64, nu: MaternNu) -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: (-rho).exp(), // ℓ = 1/κ, rho = log κ
        nu,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    }
}

#[test]
fn diag_matern_design_log_kappa_derivative_vs_fd() {
    let data = dataset();
    let rho: f64 = 0.3;
    let eps: f64 = 1e-5;
    for nu in [MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        let spec = spec_at(&data, rho, nu);
        let bundle = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
            .expect("analytic matern design derivative builds");
        let analytic = match bundle.implicit_operator.as_ref() {
            Some(op) => op.materialize_first(0).expect("materialize design derivative"),
            None => bundle.first.design_derivative.clone(),
        };

        // Central FD of the rebuilt design w.r.t. rho = log κ.
        let plus = build_matern_basis(data.view(), &spec_at(&data, rho + eps, nu))
            .expect("plus build")
            .design
            .to_dense();
        let minus = build_matern_basis(data.view(), &spec_at(&data, rho - eps, nu))
            .expect("minus build")
            .design
            .to_dense();
        let fd = (&plus - &minus) / (2.0 * eps);

        assert_eq!(
            analytic.shape(),
            fd.shape(),
            "analytic vs FD design-derivative shape mismatch"
        );
        let a_norm = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fd_norm = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
        let err = (&analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = a_norm.max(fd_norm).max(1e-12);
        // Element-wise worst ratio to see if it's a uniform factor (like 3.77).
        let mut worst_ratio = 0.0_f64;
        let mut worst_a = 0.0;
        let mut worst_f = 0.0;
        for (a, f) in analytic.iter().zip(fd.iter()) {
            if f.abs() > 1e-6 {
                let r = (a / f).abs();
                if (r - 1.0).abs() > (worst_ratio - 1.0).abs() {
                    worst_ratio = r;
                    worst_a = *a;
                    worst_f = *f;
                }
            }
        }
        eprintln!(
            "DIAG1376-DESIGN nu={nu:?} analytic_norm={a_norm:.4e} fd_norm={fd_norm:.4e} \
             rel_err={:.4e} worst_elem_ratio={worst_ratio:.4} (a={worst_a:.4e} fd={worst_f:.4e})",
            err / scale
        );
    }
    panic!("DIAG1376-DESIGN intentional fail to surface stdout");
}
