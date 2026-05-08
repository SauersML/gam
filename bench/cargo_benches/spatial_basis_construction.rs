use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    MaternBasisSpec, MaternIdentifiability, MaternNu, ThinPlateBasisSpec, build_duchon_basis,
    build_matern_basis, build_thin_plate_basis,
};
use ndarray::Array2;

fn deterministic_cloud(rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let x = (i as f64 + 1.0) * (j as f64 + 2.0);
        (x.sin() + 0.5 * (x / 7.0).cos()) / (j as f64 + 1.0)
    })
}

fn bench_spatial_basis_construction(c: &mut Criterion) {
    gam::init_parallelism();
    let n = 2_048;
    let k = 192;
    let d = 3;
    let data = deterministic_cloud(n, d);
    let centers = deterministic_cloud(k, d);

    c.bench_function("matern_basis_design_kernel_large_nk", |b| {
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: 1.25,
            nu: MaternNu::ThreeHalves,
            include_intercept: true,
            double_penalty: false,
            identifiability: MaternIdentifiability::None,
            aniso_log_scales: Some(vec![0.15, -0.05, 0.0]),
        };
        b.iter(|| black_box(build_matern_basis(data.view(), &spec).unwrap()));
    });

    c.bench_function("duchon_basis_design_kernel_large_nk", |b| {
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: Some(1.5),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: Default::default(),
            aniso_log_scales: Some(vec![0.15, -0.05, 0.0]),
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        b.iter(|| black_box(build_duchon_basis(data.view(), &spec).unwrap()));
    });

    c.bench_function("thin_plate_basis_design_kernel_large_nk", |b| {
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: 1.0,
            double_penalty: false,
            identifiability: Default::default(),
            radial_reparam: None,
        };
        b.iter(|| black_box(build_thin_plate_basis(data.view(), &spec).unwrap()));
    });
}

criterion_group!(benches, bench_spatial_basis_construction);
criterion_main!(benches);
