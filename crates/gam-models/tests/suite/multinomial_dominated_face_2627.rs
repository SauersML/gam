//! Pin for #2627 / #561: a multinomial fit whose outer seed cascade certifies a face of
//! the declared smoothing domain in zero iterations, while the interior searches reach a
//! far lower criterion value without certifying, must not publish that face.
//!
//! On the #561 surface (cubic in x1, sigmoid in x2, per-class linear x3; n = 300, K = 3)
//! drawn from seed 201 with the #561 test controls, the cascade used to publish a face
//! seed at criterion 315.15 over an evaluated interior state at 230.68. Every smooth was
//! gone: both classes were intercept-only (EDF 1.0/1.0, λ 9e9–6e11). Seed 202 never
//! collapsed. After the dominated-plateau outcome in the rho_optimizer cascade, seed 201
//! publishes an interior fit (EDF 3.73/4.15 at MSI job 1148134) and seed 202 is unchanged.
//!
//! An intercept-only fit reports EDF 1 per class up to the influence-matrix roundoff
//! (1.000000014 measured). The bar asks each class's EDF to exceed 1 by more than that
//! resolution, `√ε` per coefficient of the class, which only a fit that kept a smooth or
//! a slope clears.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{MultinomialFitRequest, fit_penalized_multinomial_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const N: usize = 300;
const K: usize = 3;

fn draw_561_surface(seed: u64) -> gam_data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).unwrap();
    let u01 = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let ux3 = Uniform::new(-1.5_f64, 1.5_f64).unwrap();
    let udraw = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = u01.sample(&mut rng);
        let c = ux3.sample(&mut rng);
        let cubic = 2.0 * a.powi(3) - 1.0 * a;
        let sigmoid = 3.0 / (1.0 + (-6.0 * (b - 0.5)).exp()) - 1.5;
        let eta = [
            0.6 + cubic + 0.5 * sigmoid + 1.5 * c,
            -0.4 - 0.5 * cubic + sigmoid - 0.8 * c,
            0.0,
        ];
        let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut weights = [0.0; K];
        let mut total = 0.0;
        for j in 0..K {
            weights[j] = (eta[j] - m).exp();
            total += weights[j];
        }
        let u = udraw.sample(&mut rng);
        let mut acc = 0.0;
        let mut chosen = K - 1;
        for j in 0..K {
            acc += weights[j] / total;
            if u <= acc {
                chosen = j;
                break;
            }
        }
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            c.to_string(),
            format!("c{chosen}"),
        ]));
    }
    let headers: Vec<String> = ["x1", "x2", "x3", "y"].iter().map(|s| s.to_string()).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #561 surface")
}

/// Per-class EDF of the published #561 fit at `seed`, the resolution an intercept-only
/// class EDF sits within of 1, and the published λ for the failure message.
fn published_class_edf(seed: u64) -> (Vec<f64>, f64, Vec<f64>) {
    let dataset = draw_561_surface(seed);
    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: usize::MAX,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&dataset, "y ~ s(x1, k=6) + s(x2, k=6) + x3", &config)
    })
    .unwrap_or_else(|error| panic!("seed {seed}: the #561 multinomial fit must publish: {error}"));
    let edf = model
        .edf_per_class
        .clone()
        .unwrap_or_else(|| panic!("seed {seed}: the fit must report per-class EDF"));
    let resolution = f64::EPSILON.sqrt() * model.p_per_class as f64;
    (edf, resolution, model.lambdas.clone())
}

#[test]
fn a_certified_face_dominated_by_an_interior_state_does_not_publish_561_2627() {
    let (edf, resolution, lambdas) = published_class_edf(201);
    assert_eq!(edf.len(), K - 1, "one EDF entry per active class");
    for (class, &class_edf) in edf.iter().enumerate() {
        assert!(
            class_edf.is_finite() && class_edf - 1.0 > resolution,
            "seed 201: class {class} EDF {class_edf} is the intercept-only fit a certified \
             domain face publishes (resolution {resolution:e}); lambdas={lambdas:?}"
        );
    }
}

#[test]
fn a_draw_that_never_collapsed_still_publishes_its_interior_fit_561_2627() {
    let (edf, resolution, lambdas) = published_class_edf(202);
    assert_eq!(edf.len(), K - 1, "one EDF entry per active class");
    for (class, &class_edf) in edf.iter().enumerate() {
        assert!(
            class_edf.is_finite() && class_edf - 1.0 > resolution,
            "seed 202: class {class} EDF {class_edf} collapsed to intercept-only \
             (resolution {resolution:e}); lambdas={lambdas:?}"
        );
    }
}
