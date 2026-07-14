//! Regression test for #531.
//!
//! A binary (probit) Bernoulli marginal-slope fit whose marginal **nuisance**
//! surface is a rich radial kernel over PCs was refused by the identifiability
//! audit with an unresolvable intra-block rank-1 deficiency: the kernel's
//! realized design at the data rows spans the constant, which duplicates the
//! probit parametric intercept inside the same `marginal_surface` block.
//!
//! The author's decisive experiment showed this is **not Duchon-specific** —
//! Matérn FATALs identically — because the bug is that the BMS marginal spatial
//! smooth is not centered against the parametric intercept *at the data rows*.
//! Matérn's `CenterSumToZero` only enforces `1ᵀα = 0` at the K centers (a
//! coefficient-space constraint), so its realized basis still spans the
//! constant.
//!
//! Fix: `apply_global_smooth_identifiability` now treats parametric-orthogonality
//! as a universal invariant for kernel/radial spatial smooths (Matérn included,
//! not just ThinPlate/Duchon), so the realized design is residualized against the
//! intercept before the audit. This test asserts the fit is accepted for both a
//! Matérn and a Duchon marginal nuisance surface.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    MaternBasisSpec, MaternIdentifiability, MaternNu, SpatialIdentifiability,
};
use gam::custom_family::BlockwiseFitOptions;
use gam::families::bms::{BernoulliMarginalSlopeTermSpec, LatentZPolicy};
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, ResourcePolicy, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

const SEED: u64 = 0x0531_0531_0531_0531;
const N: usize = 1_500;

fn normal_cdf_approx(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

fn linear(name: &str, feature_col: usize) -> LinearTermSpec {
    LinearTermSpec {
        name: name.to_string(),
        feature_col,
        feature_cols: vec![feature_col],
        categorical_levels: vec![],
        double_penalty: true,
        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
        coefficient_min: None,
        coefficient_max: None,
    }
}

/// Pure scale-free Duchon over PC1,PC2,PC3 (data columns 1,2,3).
fn duchon_pc3(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: (1..4).collect(),
            spec: DuchonBasisSpec {
                radial_reparam: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                length_scale: None,
                power: 0.0,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: Some(vec![0.0; 3]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                periodic: None,
                boundary: Default::default(),
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

/// Matérn over PC1,PC2,PC3 — the confirmed FATAL case (#531 update comment).
fn matern_pc3(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: (1..4).collect(),
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                periodic: None,
                length_scale: 1.0.into(),
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::CenterSumToZero,
                aniso_log_scales: None,
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn build_problem(
    nuisance: impl Fn(&str) -> SmoothTermSpec,
) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let normal = Normal::new(0.0, 1.0).unwrap();
    // columns: 0=sex, 1..4=PC1..PC3
    let mut data = Array2::<f64>::zeros((N, 4));
    let mut z = Array1::<f64>::zeros(N);
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let sex = if rng.random::<f64>() < 0.5 { 1.0 } else { 0.0 };
        data[[i, 0]] = sex;
        let mut pc_signal = 0.0;
        for j in 0..3 {
            let pc = normal.sample(&mut rng);
            data[[i, 1 + j]] = pc;
            pc_signal += pc * ((j + 1) as f64).recip() * if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        let score = normal.sample(&mut rng);
        z[i] = score;
        let eta = -0.6744897501960817 + 0.18 * sex + 0.05 * pc_signal;
        let p = normal_cdf_approx(eta + 0.08 * score).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let marginalspec = TermCollectionSpec {
        linear_terms: vec![linear("sex", 0)],
        random_effect_terms: vec![],
        smooth_terms: vec![nuisance("nuisance_mean")],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![nuisance("nuisance_logslope")],
    };

    (
        data,
        BernoulliMarginalSlopeTermSpec {
            y,
            weights: Array1::ones(N),
            z,
            base_link: InverseLink::Standard(StandardLink::Probit),
            marginalspec,
            logslopespec,
            marginal_offset: Array1::zeros(N),
            logslope_offset: Array1::zeros(N),
            frailty: FrailtySpec::None,
            score_warp: None,
            link_dev: None,
            latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
            score_influence_jacobian: None,
        },
    )
}

fn fit(nuisance: impl Fn(&str) -> SmoothTermSpec) {
    let (data, spec) = build_problem(nuisance);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    match fit_model(request) {
        Ok(FitResult::BernoulliMarginalSlope(_)) => {}
        Ok(_) => panic!("wrong fit result variant"),
        Err(e) => panic!(
            "#531: probit marginal-slope with a radial-kernel nuisance surface must \
             not be refused by the identifiability audit, but got: {e:?}"
        ),
    }
}

#[test]
fn matern_nuisance_probit_marginal_slope_is_identifiable() {
    gam::init_parallelism();
    fit(matern_pc3);
}

#[test]
fn duchon_nuisance_probit_marginal_slope_is_identifiable() {
    gam::init_parallelism();
    fit(duchon_pc3);
}
