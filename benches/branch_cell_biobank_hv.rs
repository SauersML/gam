//! Ignored-by-default Criterion workload for the biobank-shape marginal-slope
//! FLEX Hessian-vector cell-moment pattern.
//!
//! The `branch_dispatch_tuned` arm uses the production branch gate; the
//! `forced_transport_reference` arm bypasses that gate to approximate the
//! pre-tuned cost on the same borderline cells.  Run with:
//!
//! `cargo bench --bench branch_cell_biobank_hv -- --ignored`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use gam::families::cubic_cell_kernel::{
    DenestedCubicCell, ExactCellBranch, NORMALIZED_CELL_BRANCH_TOL, evaluate_cell_moments,
    evaluate_non_affine_cell_moments_reference,
};

fn borderline_biobank_hv_cells() -> Vec<DenestedCubicCell> {
    let bounds = [
        (-2.5, -1.25),
        (-1.25, -0.35),
        (-0.35, 0.55),
        (0.55, 1.4),
        (1.4, 2.8),
    ];
    let anchors = [(-0.8, 0.45), (-0.25, 0.9), (0.2, -0.55), (0.7, 0.35)];
    let normalized = [
        -0.95 * NORMALIZED_CELL_BRANCH_TOL,
        -0.5 * NORMALIZED_CELL_BRANCH_TOL,
        0.5 * NORMALIZED_CELL_BRANCH_TOL,
        0.95 * NORMALIZED_CELL_BRANCH_TOL,
    ];
    let mut out = Vec::new();
    for &(left, right) in &bounds {
        let width = right - left;
        let mid = 0.5 * (left + right);
        let half = 0.5 * width;
        for &(c0, c1) in &anchors {
            for &k2 in &normalized {
                for &k3 in &normalized {
                    let c3 = k3 / (half * half * half);
                    let c2 = k2 / (half * half) - 3.0 * c3 * mid;
                    out.push(DenestedCubicCell {
                        left,
                        right,
                        c0,
                        c1,
                        c2,
                        c3,
                    });
                }
            }
        }
    }
    out
}

fn bench_branch_cell_biobank_hv(c: &mut Criterion) {
    let cells = borderline_biobank_hv_cells();
    let mut group = c.benchmark_group("branch_cell_biobank_hv_borderline");
    group.bench_with_input(
        BenchmarkId::new("branch_dispatch_tuned", cells.len()),
        &cells,
        |b, cells| {
            b.iter(|| {
                let mut checksum = 0.0;
                for &cell in cells {
                    let state =
                        evaluate_cell_moments(black_box(cell), black_box(12)).expect("moments");
                    checksum += state.value + state.moments[0];
                }
                black_box(checksum)
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("forced_transport_reference", cells.len()),
        &cells,
        |b, cells| {
            b.iter(|| {
                let mut checksum = 0.0;
                for &cell in cells {
                    let state = evaluate_non_affine_cell_moments_reference(
                        black_box(cell),
                        black_box(ExactCellBranch::Sextic),
                        black_box(12),
                    )
                    .expect("reference moments");
                    checksum += state.value + state.moments[0];
                }
                black_box(checksum)
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench_branch_cell_biobank_hv);
criterion_main!(benches);
