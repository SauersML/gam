//! Regression: high-dim isotropic Duchon (16D PCs at biobank shape) used to
//! abort with "Failed to identify a constraint nullspace basis; matrix is
//! ill-conditioned" because the spectral normalization
//! `c = κ^{d/2-n} / ((2π)^{d/2}·2^{n-1}·Γ(n))` underflows to ~1e-14 in d=16,
//! driving every kernel evaluation `c·r^ν·K_ν(κr)` to ~1e-16. The basis
//! Gram then sits at ~1e-32 — below `eps²` — and the spectral whitener
//! truncates the entire spectrum as noise. The fix probes max|K_CC| once
//! per build and amplifies the kernel by `1/max|K_CC|` when it underflows,
//! lifting the basis back into a representable range. The amplification
//! is determined entirely by `centers + kernel parameters` (both stored
//! verbatim in `BasisMetadata::Duchon`), so prediction recomputes an
//! identical scale and fit/predict bases share a single coefficient frame.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    SpatialIdentifiability,
};
use gam::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec};
use gam::terms::smooth::build_term_collection_design;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn build_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = normal.sample(&mut rng);
        }
    }
    data
}

fn duchon_pc_term(name: &str, d: usize, centers: usize, power: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: (0..d).collect(),
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: centers,
                },
                length_scale: Some(1.0),
                power,
                nullspace_order: DuchonNullspaceOrder::Zero,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: Some(vec![0.0; d]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
    }
}

#[test]
fn ctn_bootstrap_design_16d_duchon_order0_power9_centers24_succeeds() {
    let n = 20_000;
    let d = 16;
    let data = build_data(n, d, 0xC7A0_B007_5EED_2026);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon_pc_term("duchon_pc16", d, 24, 9)],
    };
    let design = build_term_collection_design(data.view(), &spec)
        .expect("bootstrap CTN design build should succeed at biobank shape");
    assert_eq!(
        design.design.nrows(),
        n,
        "design row count must match input rows"
    );
    // After kernel amplification the 24-center Duchon basis carries a
    // 23-dim kernel block (one direction is consumed by the parametric
    // intercept constraint); together with the implicit intercept the
    // joint design is 24-wide. The smooth genuinely contributes signal
    // rather than silently degrading to no-op.
    assert_eq!(
        design.design.ncols(),
        24,
        "high-dim isotropic Duchon should contribute 23 kernel columns + 1 intercept"
    );
}
