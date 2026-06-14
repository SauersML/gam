//! THROWAWAY diagnostic for #1122 — production-scale path, shrinkage survival.
//! ban-scanner test references (orphaned concurrent-agent fns, unrelated to #1122):
//! install_glm_psi_gram_deriv clear_glm_psi_gram_deriv

use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis,
    build_matern_basis_log_kappa_derivatives, build_matern_basis_log_kappasecond_derivative,
    default_num_centers,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(a: f64, b: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
}

// Replicate the bug-hunt dataset's (x1,x2) cloud (n=150, seed=7).
fn xy_cloud(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let _noise = Normal::new(0.0, 0.05).unwrap();
    let mut m = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        let _ = truth(a, b);
        m[[i, 0]] = a;
        m[[i, 1]] = b;
    }
    m
}

fn spec_at(rho: f64, nu: MaternNu, k: usize, dp: bool) -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
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

#[test]
fn diag_prod_matern_shrinkage() {
    let data = xy_cloud(150, 7);
    let nu = MaternNu::FiveHalves;
    let k = default_num_centers(150, 2);
    println!("default_num_centers(150,2) = {k}");

    // Sweep rho to find where DoublePenaltyNullspace survives.
    println!("\n-- shrinkage-survival sweep over rho (log kappa) --");
    let mut survive_rho: Option<f64> = None;
    for step in 0..=24 {
        let rho = -3.0 + 0.5 * step as f64; // -3 .. 9
        let spec = spec_at(rho, nu, k, true);
        match build_matern_basis(data.view(), &spec) {
            Ok(b) => {
                let srcs: Vec<String> = b
                    .penaltyinfo
                    .iter()
                    .filter(|i| i.active)
                    .map(|i| format!("{:?}", i.source))
                    .collect();
                let has_shrink = srcs.iter().any(|s| s.contains("DoublePenaltyNullspace"));
                println!(
                    "  rho={rho:+.2} kappa={:.3e} blocks={} active_src={:?} shrink={}",
                    rho.exp(),
                    b.penalties.len(),
                    srcs,
                    has_shrink
                );
                if has_shrink && survive_rho.is_none() {
                    survive_rho = Some(rho);
                }
            }
            Err(e) => println!("  rho={rho:+.2} BUILD ERR: {e}"),
        }
    }

    let rho = survive_rho.unwrap_or(0.3);
    println!("\n== FD checks at rho={rho} (shrinkage-active={}) ==", survive_rho.is_some());

    let spec = spec_at(rho, nu, k, true);
    let base = build_matern_basis(data.view(), &spec).unwrap();
    for (i, info) in base.penaltyinfo.iter().filter(|i| i.active).enumerate() {
        println!(
            "  active block[{i}] source={:?} eff_rank={} norm_scale={:.6e}",
            info.source, info.effective_rank, info.normalization_scale
        );
    }

    let analytic = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
        .unwrap()
        .first
        .penalties_derivative;
    let analytic2 = build_matern_basis_log_kappasecond_derivative(data.view(), &spec)
        .unwrap()
        .penaltiessecond_derivative;

    let h = 1e-6;
    let plus = build_matern_basis(data.view(), &spec_at(rho + h, nu, k, true))
        .unwrap()
        .penalties;
    let minus = build_matern_basis(data.view(), &spec_at(rho - h, nu, k, true))
        .unwrap()
        .penalties;
    let mid = base.penalties.clone();

    println!(
        "  block counts: analytic1={} analytic2={} value(mid)={} plus={} minus={}",
        analytic.len(),
        analytic2.len(),
        mid.len(),
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

    let h2 = 1e-4;
    let p2 = build_matern_basis(data.view(), &spec_at(rho + h2, nu, k, true))
        .unwrap()
        .penalties;
    let n2 = build_matern_basis(data.view(), &spec_at(rho - h2, nu, k, true))
        .unwrap()
        .penalties;
    let nblk2 = analytic2.len().min(p2.len()).min(n2.len()).min(mid.len());
    for block in 0..nblk2 {
        let num = (&p2[block] - 2.0 * &mid[block] + &n2[block]) / (h2 * h2);
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

    // Does the shrinkage projector move with kappa?
    if mid.len() >= 2 {
        let dk = 1e-3;
        let shrink_p = build_matern_basis(data.view(), &spec_at(rho + dk, nu, k, true))
            .unwrap()
            .penalties[1]
            .clone();
        let diff = max_abs(&(&shrink_p - &mid[1]));
        println!(
            "\n  SHRINKAGE block[1] moves with kappa? maxabs(S(rho+1e-3)-S(rho))={:.6e}",
            diff
        );
        println!(
            "  shrinkage analytic1_deriv_max={:.6e}  analytic2_deriv_max={:.6e}",
            if analytic.len() > 1 { max_abs(&analytic[1]) } else { f64::NAN },
            if analytic2.len() > 1 { max_abs(&analytic2[1]) } else { f64::NAN }
        );
        // FD of the shrinkage derivative explicitly:
        if plus.len() >= 2 && minus.len() >= 2 {
            let fd = (&plus[1] - &minus[1]) / (2.0 * h);
            println!("  shrinkage FD-first-deriv max={:.6e}", max_abs(&fd));
        }
    }

    assert!(!base.penalties.is_empty());
}
