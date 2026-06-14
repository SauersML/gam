//! THROWAWAY diagnostic for #1122 — do not keep.

use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis,
    build_matern_basis_log_kappa_derivatives, build_matern_basis_log_kappasecond_derivative,
};
use ndarray::Array2;

fn dataset() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.2, 0.4, 0.7, 0.1, 1.0, 0.8, 1.3, 1.1, 0.5, 0.9],
    )
    .unwrap()
}

fn spec_at(data: &Array2<f64>, rho: f64, nu: MaternNu, dp: bool) -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: (-rho).exp(),
        nu,
        include_intercept: false,
        double_penalty: dp,
        identifiability: Default::default(),
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    }
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

fn penalties_at(data: &Array2<f64>, rho: f64, nu: MaternNu, dp: bool) -> Vec<Array2<f64>> {
    let spec = spec_at(data, rho, nu, dp);
    build_matern_basis(data.view(), &spec).unwrap().penalties
}

#[test]
fn diag_matern_double_penalty() {
    let data = dataset();
    let rho: f64 = 0.3;

    for nu in [MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        println!("\n========== nu={nu:?}  rho={rho}  double_penalty=true ==========");

        let spec = spec_at(&data, rho, nu, true);
        let base = build_matern_basis(data.view(), &spec).unwrap();
        println!("penalties.len() = {}", base.penalties.len());
        for (i, info) in base.penaltyinfo.iter().enumerate() {
            println!(
                "  block[{i}] source={:?} active={} eff_rank={} norm_scale={:.6e}",
                info.source, info.active, info.effective_rank, info.normalization_scale
            );
        }
        for (i, p) in base.penalties.iter().enumerate() {
            println!("  penalty[{i}] shape={:?} maxabs={:.6e}", p.dim(), max_abs(p));
        }

        let analytic = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
            .unwrap()
            .first
            .penalties_derivative;
        println!(
            "analytic first deriv blocks = {}  value blocks = {}",
            analytic.len(),
            base.penalties.len()
        );
        let h = 1e-6;
        let plus = penalties_at(&data, rho + h, nu, true);
        let minus = penalties_at(&data, rho - h, nu, true);
        println!(
            "value-block counts at rho/rho+h/rho-h = {}/{}/{}",
            base.penalties.len(),
            plus.len(),
            minus.len()
        );
        let active_src: Vec<String> = base
            .penaltyinfo
            .iter()
            .filter(|i| i.active)
            .map(|i| format!("{:?}", i.source))
            .collect();
        let nblk = analytic.len().min(plus.len()).min(minus.len());
        for block in 0..nblk {
            let num = (&plus[block] - &minus[block]) / (2.0 * h);
            let err = max_abs(&(&analytic[block] - &num));
            let scale = max_abs(&analytic[block]).max(max_abs(&num)).max(1.0);
            println!(
                "  FIRST block[{block}] src={}: analytic_max={:.4e} fd_max={:.4e} ERR={:.4e} rel={:.4e}",
                active_src.get(block).cloned().unwrap_or_default(),
                max_abs(&analytic[block]),
                max_abs(&num),
                err,
                err / scale
            );
        }

        let analytic2 = build_matern_basis_log_kappasecond_derivative(data.view(), &spec)
            .unwrap()
            .penaltiessecond_derivative;
        let h2 = 1e-4;
        let p2 = penalties_at(&data, rho + h2, nu, true);
        let m2 = penalties_at(&data, rho, nu, true);
        let n2 = penalties_at(&data, rho - h2, nu, true);
        let nblk2 = analytic2.len().min(p2.len()).min(n2.len()).min(m2.len());
        for block in 0..nblk2 {
            let num = (&p2[block] - 2.0 * &m2[block] + &n2[block]) / (h2 * h2);
            let err = max_abs(&(&analytic2[block] - &num));
            let scale = max_abs(&analytic2[block]).max(max_abs(&num)).max(1.0);
            println!(
                "  SECOND block[{block}] src={}: analytic_max={:.4e} fd_max={:.4e} ERR={:.4e} rel={:.4e}",
                active_src.get(block).cloned().unwrap_or_default(),
                max_abs(&analytic2[block]),
                max_abs(&num),
                err,
                err / scale
            );
        }

        let dd = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
            .unwrap()
            .first
            .design_derivative;
        let dpd = build_matern_basis(data.view(), &spec_at(&data, rho + h, nu, true))
            .unwrap()
            .design
            .try_to_dense_by_chunks("dp")
            .unwrap();
        let dmd = build_matern_basis(data.view(), &spec_at(&data, rho - h, nu, true))
            .unwrap()
            .design
            .try_to_dense_by_chunks("dm")
            .unwrap();
        let dnum = (&dpd - &dmd) / (2.0 * h);
        let derr = max_abs(&(&dd - &dnum));
        println!(
            "  DESIGN deriv: analytic_max={:.4e} fd_max={:.4e} ERR={:.4e} rel={:.4e}",
            max_abs(&dd),
            max_abs(&dnum),
            derr,
            derr / max_abs(&dd).max(max_abs(&dnum)).max(1.0)
        );

        if base.penalties.len() >= 2 {
            let shrink0 = &m2[1];
            let dk = 1e-3;
            let shrink_p = &penalties_at(&data, rho + dk, nu, true)[1];
            let diff = max_abs(&(shrink_p - shrink0));
            println!(
                "  SHRINKAGE block[1] moves with kappa? maxabs(S(rho+1e-3)-S(rho))={:.6e}  analytic_deriv_max={:.6e}",
                diff,
                if analytic.len() > 1 { max_abs(&analytic[1]) } else { f64::NAN }
            );
        } else {
            println!("  (no shrinkage block present at this rho)");
        }
        assert!(!base.penalties.is_empty());
    }
}
