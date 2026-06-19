// Diagnostic for #1266: per-fold rho/EDF/convergence dump for the irrelevant
// covariate shrinkage fit. NOT a gate — printed via `cargo test -- --nocapture`.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn irrelevant_covariate_dataset(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            let z: f64 = unit.sample(&mut rng);
            let y = (6.0_f64 * x).sin() + noise.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(
        ["x", "z", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode")
}

/// log|H| via a plain Cholesky (H = L Lᵀ ⇒ log|H| = 2 Σ log L_ii). Returns None
/// if H is not numerically PD (shouldn't happen at a converged penalized mode).
fn chol_logdet(h: &ndarray::Array2<f64>) -> Option<f64> {
    let n = h.nrows();
    if n == 0 || h.ncols() != n {
        return None;
    }
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = h[[i, j]];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut ld = 0.0;
    for i in 0..n {
        ld += 2.0 * l[i * n + i].ln();
    }
    Some(ld)
}

fn term_edf_and_penalty_cursor(fit: &FitResult, needle: &str) -> (f64, usize, usize) {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected standard fit");
    };
    let design = &std_fit.design;
    let unified = &std_fit.fit;
    let mut penalty_cursor = 0usize;
    for (_n, _r) in &design.random_effect_ranges {
        penalty_cursor += 1;
    }
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        if term.name.contains(needle) {
            let edf = unified.per_term_edf(term.coeff_range.clone(), penalty_cursor, k);
            return (edf, penalty_cursor, k);
        }
        penalty_cursor += k;
    }
    panic!("no term {needle}");
}

#[test]
fn diag_1266_zfold_dump() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for seed in 200u64..205 {
        let data = irrelevant_covariate_dataset(seed, 800);
        let fit = fit_from_formula("y ~ s(x) + s(z)", &data, &cfg).expect("fit ok");
        let (z_edf, z_cur, z_k) = term_edf_and_penalty_cursor(&fit, "z");
        let (x_edf, x_cur, x_k) = term_edf_and_penalty_cursor(&fit, "x");
        let FitResult::Standard(std_fit) = &fit else {
            unreachable!()
        };
        let u = &std_fit.fit;
        let ll = &u.log_lambdas;
        let z_rho: Vec<f64> = (z_cur..z_cur + z_k).map(|i| ll[i]).collect();
        let x_rho: Vec<f64> = (x_cur..x_cur + x_k).map(|i| ll[i]).collect();
        let log_h = u
            .geometry
            .as_ref()
            .and_then(|g| chol_logdet(g.penalized_hessian.as_array()));
        println!(
            "DIAG1266 seed={seed} converged={} iters={} gnorm={:?} reml={:.4} pen_term={:.4} log|H|={:?} | x_edf={x_edf:.4} x_rho={x_rho:?} | z_edf={z_edf:.4} z_rho={z_rho:?} | all_loglam={:?}",
            u.outer_converged,
            u.outer_iterations,
            u.outer_gradient_norm,
            u.reml_score,
            u.stable_penalty_term,
            log_h,
            ll.to_vec(),
        );
    }
    // Force nextest to surface the captured stdout above (it shows output only
    // on failure). This is a DIAGNOSTIC, never a gate.
    panic!("DIAG1266 intentional fail to surface stdout under nextest");
}
