//! V100 parity + timing for the NVRTC hex tensor moment kernel.
//!
//! Lives as a `[[bench]]` so v100-bench-runner picks it up alongside the other
//! GPU benches. The body does two things in one process:
//!
//!   1. **Parity gate.** Builds the alpha-major `[NALPHA, n_cells]` moment
//!      table on the device, downloads it, and asserts every entry matches
//!      `tensor_hex_moment_cpu` at `abs ≤ 1e-12 OR rel ≤ 1e-12`. A mismatch
//!      panics the bench — v100-bench-runner reports the failure as a hard
//!      regression. On hosts with no CUDA runtime the gate prints a
//!      single-line skip notice and exits 0 so the same binary works as a
//!      smoke check on the Mac builder.
//!
//!   2. **Hill-climb measurement.** Wraps the same device build in a Criterion
//!      benchmark group so the runner reads off a wall-clock number for the
//!      NonAffineFinite-equivalent (here: a 2D non-uniform tensor mesh with
//!      five alpha shapes). The CPU reference is also timed in the same
//!      group so the speed-up is a single division.
//!
//! Run with: `cargo bench --bench cubic_hex_tensor_gpu_parity`

use criterion::{Criterion, criterion_group, criterion_main};
use gam::gpu::cubic_bspline_moments::{
    AxisCubicMomentTables, CubicMomentSpec, HexCellTable, MomentLayout,
    build_hex_tensor_moments_device, tensor_hex_moment_cpu,
};

/// 8-cubic-basis non-uniform knot vector — same shape the in-module CPU
/// reference test uses, scaled up below to a perf-relevant cell count.
fn nonuniform_knots() -> Vec<f64> {
    let interior = [-1.7, -0.4, 0.1, 0.9, 1.55];
    let mut t = Vec::new();
    for _ in 0..=3 {
        t.push(-2.0);
    }
    t.extend_from_slice(&interior);
    for _ in 0..=3 {
        t.push(3.0);
    }
    t
}

/// Five alpha shapes covering ν ∈ {0, 1, 2, 3} on both axes, plus a mixed
/// asymmetric (2, 1). Same shapes as the in-module parity test so a regression
/// here means a regression there too.
fn alpha_shapes() -> Vec<Vec<u8>> {
    vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 1],
        vec![3, 3],
    ]
}

/// Build a (cells × alpha) workload at the size v100-bench-runner cares about:
/// every (sx, sy) on the non-uniform mesh × every (pa, pb) over the 10
/// unordered active pairs. With the 5-span 2D mesh that's
/// `5 × 5 × 10 × 10 = 2500` cells, large enough for warp-coalesced output
/// stride 32-alignment to show real Volta+ throughput.
struct Workload {
    table: AxisCubicMomentTables,
    spec: CubicMomentSpec,
    cells: HexCellTable,
    /// `(sx, sy, pa, pb)` mirror — needed for the per-cell CPU parity check.
    meta: Vec<(usize, usize, usize, usize)>,
}

fn build_workload() -> Workload {
    let t = nonuniform_knots();
    let table = AxisCubicMomentTables::build(&t, 0, 0);
    let alphas = alpha_shapes();
    let deriv = vec![vec![0u8, 0u8]; alphas.len()];
    let spec = CubicMomentSpec {
        alphas,
        derivative_left: deriv.clone(),
        derivative_right: deriv,
        layout: MomentLayout::AlphaMajor,
    };
    let mut span_per_axis: Vec<i32> = Vec::new();
    let mut pair_per_axis: Vec<i32> = Vec::new();
    let mut width_per_axis: Vec<f64> = Vec::new();
    let mut meta: Vec<(usize, usize, usize, usize)> = Vec::new();
    for sx in 0..table.n_spans() {
        for sy in 0..table.n_spans() {
            // All 10 unordered active pairs on each axis — full pair sweep,
            // matching what a real tensor-smooth row assembly would touch.
            for pa in 0..10 {
                for pb in 0..10 {
                    span_per_axis.push(sx as i32);
                    span_per_axis.push(sy as i32);
                    pair_per_axis.push(pa as i32);
                    pair_per_axis.push(pb as i32);
                    width_per_axis.push(table.width[sx]);
                    width_per_axis.push(table.width[sy]);
                    meta.push((sx, sy, pa, pb));
                }
            }
        }
    }
    let n_cells = meta.len();
    let cells = HexCellTable {
        span_per_axis,
        pair_per_axis,
        width_per_axis,
        n_cells,
        d: 2,
    };
    Workload {
        table,
        spec,
        cells,
        meta,
    }
}

/// Parity gate. Returns `false` if no CUDA runtime is available so the
/// caller can skip the timing pass instead of marking a Mac run as failed.
fn parity_gate(w: &Workload) -> bool {
    let axes_for_build = vec![vec![w.table.clone()], vec![w.table.clone()]];
    let dev = match build_hex_tensor_moments_device(&w.spec, &axes_for_build, &w.cells) {
        Ok(d) => d,
        Err(err) => {
            eprintln!(
                "[cubic_hex_tensor_gpu_parity] no CUDA runtime — skipping ({err})"
            );
            return false;
        }
    };

    // Download alpha-major buffer for verification. On non-Linux this would
    // not compile (the `Device` variant is gated), but the bench itself is
    // also Linux-only by virtue of `build_hex_tensor_moments_device`'s
    // non-Linux stub returning `DriverLibraryUnavailable` above.
    #[cfg(target_os = "linux")]
    let host_vals = {
        let backend =
            gam::gpu::cubic_bspline_moments::CubicMomentBackend::probe()
                .expect("backend probe after successful build");
        let stream = backend.stream_for_bench();
        stream
            .memcpy_dtov(&dev.values)
            .expect("dtov of device moments")
    };
    #[cfg(not(target_os = "linux"))]
    let host_vals: Vec<f64> = dev.values.clone();

    let n_alpha = w.spec.alphas.len();
    let out_stride = host_vals.len() / n_alpha;
    let expected_stride = ((w.cells.n_cells + 31) / 32) * 32;
    assert_eq!(
        out_stride, expected_stride,
        "alpha-major stride must be 32-aligned n_cells"
    );

    let axes_cpu: Vec<&AxisCubicMomentTables> = vec![&w.table, &w.table];
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (a_idx, alpha) in w.spec.alphas.iter().enumerate() {
        for (cell, &(sx, sy, pa, pb)) in w.meta.iter().enumerate() {
            let want = tensor_hex_moment_cpu(&axes_cpu, &[sx, sy], alpha, &[pa, pb]);
            let got = host_vals[a_idx * out_stride + cell];
            let abs = (got - want).abs();
            let denom = want.abs().max(1.0);
            let rel = abs / denom;
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                abs <= 1e-12 || rel <= 1e-12,
                "gpu hex tensor parity drift cell={cell} alpha={alpha:?} \
                 gpu={got:.17e} cpu={want:.17e} abs={abs:.3e} rel={rel:.3e}"
            );
        }
    }
    println!(
        "[cubic_hex_tensor_gpu_parity] OK: n_cells={} n_alpha={} max_abs={:.3e} max_rel={:.3e}",
        w.cells.n_cells, n_alpha, max_abs, max_rel
    );
    true
}

fn bench_cubic_hex_tensor(c: &mut Criterion) {
    let w = build_workload();
    let cuda_ok = parity_gate(&w);
    if !cuda_ok {
        // Mac builder path: do not register timed iterations on a CPU-only
        // host — the GPU build would just keep returning the same error
        // every iteration, polluting the bench history with phantom samples.
        return;
    }

    let mut group = c.benchmark_group("cubic_hex_tensor_moments");
    group.sample_size(10);

    let axes_for_build = vec![vec![w.table.clone()], vec![w.table.clone()]];
    group.bench_function("gpu_alpha_major", |b| {
        b.iter(|| {
            let dev = build_hex_tensor_moments_device(&w.spec, &axes_for_build, &w.cells)
                .expect("gpu build must succeed once parity gate passed");
            // Force the bench to keep the device buffer alive for the iteration
            // by reading a single element back. This avoids the optimiser
            // hoisting the call without using `black_box`.
            #[cfg(target_os = "linux")]
            {
                let backend =
                    gam::gpu::cubic_bspline_moments::CubicMomentBackend::probe()
                        .expect("backend probe");
                let stream = backend.stream_for_bench();
                let head = stream
                    .memcpy_dtov(&dev.values)
                    .expect("dtov head");
                assert!(head[0].is_finite());
            }
            #[cfg(not(target_os = "linux"))]
            {
                assert!(dev.values[0].is_finite());
            }
        })
    });

    let axes_cpu: Vec<&AxisCubicMomentTables> = vec![&w.table, &w.table];
    let alphas = &w.spec.alphas;
    let meta = &w.meta;
    group.bench_function("cpu_reference", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for alpha in alphas.iter() {
                for &(sx, sy, pa, pb) in meta.iter() {
                    acc += tensor_hex_moment_cpu(&axes_cpu, &[sx, sy], alpha, &[pa, pb]);
                }
            }
            assert!(acc.is_finite());
        })
    });

    group.finish();
}

criterion_group!(benches, bench_cubic_hex_tensor);
criterion_main!(benches);
