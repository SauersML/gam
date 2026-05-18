//! Ignored-by-default Criterion workload for the BVN primitive used by the
//! biobank marginal-slope cycle-0 affine cells.
//!
//! Run with: `cargo bench --bench bvn_biobank_shape`

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use gam::families::cubic_cell_kernel::bivariate_normal_cdf;

fn biobank_shape_args() -> Vec<(f64, f64, f64)> {
    // Cycle-0 affine cells use h = alpha/sqrt(1+beta^2), k = cell boundary,
    // rho = -beta/sqrt(1+beta^2).  The grid below concentrates on the common
    // central support boundaries and moderately strong slopes, with a few tail
    // and near-independent cases to keep branch behavior representative.
    let hs = [-2.5, -1.0, -0.25, 0.0, 0.35, 1.25, 2.75];
    let ks = [-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0];
    let rhos = [-0.98, -0.9, -0.7, -0.35, 0.0, 0.35, 0.7, 0.9, 0.98];
    let mut out = Vec::with_capacity(hs.len() * ks.len() * rhos.len());
    for &h in &hs {
        for &k in &ks {
            for &rho in &rhos {
                out.push((h, k, rho));
            }
        }
    }
    out
}

fn bench_bivariate_normal_cdf_biobank_shape(c: &mut Criterion) {
    let args = biobank_shape_args();
    c.bench_function("bvn_cdf_biobank_shape_cycle0", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &(h, k, rho) in &args {
                acc +=
                    bivariate_normal_cdf(black_box(h), black_box(k), black_box(rho)).expect("bvn");
            }
            black_box(acc)
        });
    });
}

criterion_group!(benches, bench_bivariate_normal_cdf_biobank_shape);
criterion_main!(benches);
