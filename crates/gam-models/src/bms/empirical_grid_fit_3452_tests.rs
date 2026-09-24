//! Full BMS fitting-path checks for the empirical-grid covariance correction.
//! Each arm performs one unpenalized fit on a fixed seed. Numerical correctness
//! is pinned separately by builder weight perturbations, an independent
//! sandwich, cross-covariance cancellation, and grid-only Monte Carlo checks in
//! empirical_grid_sampling_3452_tests. These integration checks confirm that
//! the requested corrected covariance reaches the fitted result.

use crate::bms::LatentMeasureKind;
use crate::bms::*;
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_math::probability::normal_cdf;
use gam_terms::smooth::{
    LinearTermSpec, SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
};
use ndarray::Array1;
fn next_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

const N: usize = 2_000;
const SLOPE: f64 = 1.880;

/// `(weight, variance)` of each normal component of `e`, a unit-variance law.
#[derive(Clone, Copy, Debug)]
enum Law {
    Gaussian,
    ScaleMixture,
}

impl Law {
    fn components(self) -> &'static [(f64, f64)] {
        match self {
            Law::Gaussian => &[(1.0, 1.0)],
            Law::ScaleMixture => &[(0.95, 0.5 / 0.95), (0.05, 10.0)],
        }
    }

    fn draw(self, state: &mut u64) -> f64 {
        let pick = next_unit(state);
        let mut cumulative = 0.0;
        let components = self.components();
        let &(_, variance) = components
            .iter()
            .find(|&&(weight, _)| {
                cumulative += weight;
                pick < cumulative
            })
            .unwrap_or_else(|| components.last().expect("a component"));
        variance.sqrt() * next_gauss(state)
    }

    /// The anchor `a` with `Σ_c π_c Φ(a/√(1 + s²σ_c²)) = Φ(q)`, by bisection on a
    /// strictly increasing function.
    fn anchor(self, q: f64) -> f64 {
        let target = normal_cdf(q);
        let marginal = |a: f64| -> f64 {
            self.components()
                .iter()
                .map(|&(weight, variance)| {
                    weight * normal_cdf(a / (1.0 + SLOPE * SLOPE * variance).sqrt())
                })
                .sum()
        };
        let (mut low, mut high) = (-40.0_f64, 40.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if marginal(mid) < target {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    }
}

/// The held covariates and their anchors under `law`.
fn design(law: Law) -> Vec<(f64, f64, f64)> {
    let mut state = 0x3452_C0DE_0000_0001;
    (0..N)
        .map(|_| {
            let x1 = next_gauss(&mut state);
            let x2 = next_gauss(&mut state);
            let q = -0.5 + 0.4 * x1 - 0.3 * x2;
            (x1, x2, law.anchor(q))
        })
        .collect()
}

fn replicate(law: Law, design: &[(f64, f64, f64)], state: &mut u64) -> gam_data::EncodedDataset {
    let rows = design
        .iter()
        .map(|&(x1, x2, anchor)| {
            let e = law.draw(state);
            let z = 0.5 * x1 + 0.3 * x2 + e;
            let y = u8::from(next_unit(state) < normal_cdf(anchor + SLOPE * e));
            StringRecord::from(vec![
                y.to_string(),
                z.to_string(),
                x1.to_string(),
                x2.to_string(),
            ])
        })
        .collect();
    let headers = ["y", "z", "x1", "x2"].map(String::from).to_vec();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #3452 fixture")
}

fn check_fit(law: Law) {
    let design = design(law);
    let mut state = 0x3452_0000_0000_00A1_u64 ^ 0x9E37_79B9_7F4A_7C15;
    let data = replicate(law, &design, &mut state);
    let mut marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
        level: Default::default(),
    };
    let slopespec = marginalspec.clone();
    for col in [2, 3] {
        marginalspec.linear_terms.push(LinearTermSpec {
            name: format!("x{col}"),
            feature_col: col,
            feature_cols: vec![col],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: Default::default(),
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        });
    }
    let spec = BernoulliMarginalSlopeTermSpec {
        y: data.values.column(0).to_owned(),
        weights: Array1::ones(N),
        z: data.values.column(1).to_owned(),
        base_link: gam_spec::InverseLink::Standard(gam_spec::StandardLink::Probit),
        marginalspec,
        slopespec,
        marginal_offset: Array1::zeros(N),
        slope_offset: Array1::zeros(N),
        frailty: crate::survival::lognormal_kernel::FrailtySpec::None,
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::default(),
        score_influence_jacobian: None,
        residual: None,
        outer_start_levels: None,
        declared_latent_law: None,
    };
    let fit = fit_bernoulli_marginal_slope_terms(
        data.values.view(),
        spec,
        &gam_custom_family::BlockwiseFitOptions {
            compute_covariance: true,
            ..Default::default()
        },
        &SpatialLengthScaleOptimizationOptions::default(),
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .expect("family fit");
    assert!(fit.latent_z_conditional_calibration.is_some());
    if matches!(law, Law::ScaleMixture) {
        assert!(matches!(
            fit.latent_measure,
            LatentMeasureKind::GlobalEmpirical { .. }
        ));
    }
    let covariance = fit.fit.beta_covariance().expect("covariance");
    assert!(covariance.iter().all(|x| x.is_finite()));
    assert!(covariance.diag().iter().all(|x| *x > 0.0));
    eprintln!(
        "PASS {law:?}: beta={:?}, variance={:?}",
        fit.fit.beta,
        covariance.diag()
    );
}

#[test]
fn empirical_grid_fit_retains_corrected_covariance_3452() {
    check_fit(Law::ScaleMixture);
}

#[test]
fn gaussian_control_fit_retains_corrected_covariance_3452() {
    check_fit(Law::Gaussian);
}
