//! #1067 repro harness: 2-D/joint spatial binomial-logit GAM that runs 20-100x
//! slower than mgcv. Mirrors the `geo_disease_eas_*_k*` fuzz cases (n=6000,
//! binomial-logit, joint Matern/Duchon/thin-plate smooth over the PC columns).
//!
//! NOT a test — examples skip dev-deps so the build avoids the slow `autodiff`
//! crate. Run: `cargo run --release --example repro1067_geo_binomial -- <basis> <ncenters> <npcs>`
//!   basis ∈ {matern, duchon, tps}; defaults matern 6 16.

use gam::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitRequest, FitResult, StandardFitRequest};
use ndarray::{Array1, Array2};
use std::time::Instant;

/// SplitMix64 → uniform/normal, matching the spirit of the Rust synthetic
/// `synthetic_geo_disease_eas_columns` generator so the fixture is realistic.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn unif_range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unif()
    }
    fn normal(&mut self, mu: f64, sd: f64) -> f64 {
        let u1 = self.unif().max(1e-12);
        let u2 = self.unif();
        mu + sd * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn bernoulli(&mut self, p: f64) -> bool {
        self.unif() < p
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn build(n: usize, n_pcs: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Rng(seed);
    let mut x = Array2::<f64>::zeros((n, n_pcs));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eas = rng.bernoulli(0.23);
        let lat = if eas {
            rng.unif_range(15.0, 52.0)
        } else {
            rng.unif_range(-55.0, 70.0)
        };
        let lon = if eas {
            rng.unif_range(95.0, 145.0)
        } else {
            rng.unif_range(-175.0, 175.0)
        };
        let mut eta = if eas {
            (0.10_f64 / 0.90).ln()
        } else {
            (0.02_f64 / 0.98).ln()
        };
        if eas {
            let lat_e = (lat - 33.5) / 11.0;
            let lon_e = (lon - 120.0) / 10.0;
            eta += 3.25 * (1.35 * lat_e).sin() - 2.85 * (1.55 * lon_e).cos()
                + 2.50 * (1.10 * lat_e * lon_e).sin()
                + 1.90 * (1.60 * lat_e + 0.45 * lon_e).cos()
                + rng.normal(0.0, 0.20);
        } else {
            eta += rng.normal(0.0, 0.08);
        }
        y[i] = if rng.bernoulli(sigmoid(eta)) {
            1.0
        } else {
            0.0
        };
        let lat_s = lat / 90.0;
        let lon_s = lon / 180.0;
        for j in 0..n_pcs {
            let jf = j as f64;
            let a = 0.98 - 0.05 * jf;
            let b = 0.26 + 0.03 * jf;
            let c = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.12 + 0.01 * jf);
            let d = if j >= 8 { 0.22 } else { 0.06 };
            x[[i, j]] = a * lat_s
                + b * lon_s
                + c * lat_s * lon_s
                + d * (if eas { 1.0 } else { 0.0 })
                + rng.normal(0.0, 0.13 + 0.018 * jf);
        }
    }
    (x, y)
}

fn smooth_term(basis: &str, n_pcs: usize, centers: usize) -> SmoothTermSpec {
    let feature_cols: Vec<usize> = (0..n_pcs).collect();
    let basis_spec = match basis {
        "matern" => SmoothBasisSpec::Matern {
            feature_cols,
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::EqualMass {
                    num_centers: centers,
                },
                periodic: None,
                length_scale: 1.0,
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::default(),
                aniso_log_scales: None,
                nullspace_shrinkage_survived: None,
            },
            input_scales: None,
        },
        other => panic!("unknown basis {other} (this repro covers matern only)"),
    };
    SmoothTermSpec {
        name: "geo".to_string(),
        basis: basis_spec,
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1e-7,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let basis = args.get(1).map(|s| s.as_str()).unwrap_or("matern");
    let centers: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let n_pcs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6000);

    eprintln!("[repro1067] basis={basis} centers={centers} n_pcs={n_pcs} n={n}");
    let (x, y) = build(n, n_pcs, 20260301);
    let pos = y.iter().filter(|&&v| v > 0.5).count();
    eprintln!("[repro1067] prevalence={:.4}", pos as f64 / n as f64);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![smooth_term(basis, n_pcs, centers)],
    };
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(Array1::ones(n)),
        offset: std::sync::Arc::new(Array1::zeros(n)),
        spec,
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        // Binomial fit — no Tweedie power to profile (#2026).
        estimate_tweedie_p: false,
        options: fit_options(),
        kappa_options,
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
    }));
    let dt = t0.elapsed().as_secs_f64();

    match result {
        Ok(FitResult::Standard(s)) => {
            let finite = s.fit.beta.iter().all(|v: &f64| v.is_finite());
            eprintln!(
                "[repro1067] DONE basis={basis} centers={centers} n={n} :: {dt:.2}s  finite={finite}  p={}",
                s.fit.beta.len()
            );
        }
        Ok(_) => eprintln!("[repro1067] unexpected result kind in {dt:.2}s"),
        Err(e) => eprintln!("[repro1067] FAILED in {dt:.2}s :: {e:?}"),
    }
}
