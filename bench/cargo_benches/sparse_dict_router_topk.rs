//! Router throughput floor for the collapsed-linear-lane SAE (#1026).
//!
//! `top_s_online` scores one row against the whole `K`-wide decoder a column
//! tile at a time and folds every score into an online top-`s` selector. At the
//! issue's headline width (`K ≈ 32_000`) this per-row route is the dominant CPU
//! cost of a fit, and — on the device score-block path — the *only* CPU cost,
//! since the GEMM runs on the GPU and the host just folds the returned block.
//!
//! The fold is `TopSSelector::offer`. Before the O(1)-reject change it did an
//! O(`s`) linear rescan of the survivor set on *every* one of `K` offers per
//! row; after, all but the ~`s` accepted offers are O(1). This bench measures
//! that on the shape that matters two ways:
//!   * `fold_old` vs `fold_new` — the two selector variants fed an *identical*
//!     score stream, isolating the fold (the only thing that changed).
//!   * `router_end_to_end` — the real `top_s_online` (dot + new fold), the
//!     integrated number the fit loop pays.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use gam::sae::sparse_dict::top_s_online;
use ndarray::Array2;

const K: usize = 32_768; // #1026 headline dictionary width
const P: usize = 64; // ambient / activation dim
const S: usize = 32; // active atoms kept per row
const TILE: usize = 2_048; // column tile width
const ROWS: usize = 64; // minibatch rows folded per measurement

fn fixtures() -> (Array2<f32>, Array2<f32>) {
    let rows = Array2::from_shape_fn((ROWS, P), |(i, c)| {
        (((i * 31 + c * 17) as f32) * 0.013).sin() * 0.9
    });
    let mut atoms = Array2::from_shape_fn((K, P), |(a, c)| (((a * 7 + c * 5) as f32) * 0.011).cos());
    for mut atom in atoms.outer_iter_mut() {
        let norm = atom.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        atom.mapv_inplace(|v| v / norm);
    }
    (rows, atoms)
}

/// Precompute the exact `K` scores each row folds (the score block the fold
/// consumes), so the two fold variants see byte-identical input streams.
fn score_streams(rows: &Array2<f32>, atoms: &Array2<f32>) -> Vec<Vec<f32>> {
    rows.outer_iter()
        .map(|row| {
            atoms
                .outer_iter()
                .map(|atom| {
                    let mut acc = 0.0f32;
                    for c in 0..P {
                        acc += row[c] * atom[c];
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

/// Pre-optimization fold: full O(`capacity`) rescan on every offer once full.
fn fold_old(stream: &[f32], capacity: usize) -> usize {
    let mut heap: Vec<(u32, f32, f32)> = Vec::with_capacity(capacity);
    for (a, &score) in stream.iter().enumerate() {
        let atom = a as u32;
        let mag = score.abs();
        if heap.len() < capacity {
            heap.push((atom, score, mag));
            continue;
        }
        let mut worst = 0usize;
        for k in 1..heap.len() {
            if heap[k].2 < heap[worst].2 || (heap[k].2 == heap[worst].2 && heap[k].0 > heap[worst].0)
            {
                worst = k;
            }
        }
        let (w_atom, _, w_mag) = heap[worst];
        if mag > w_mag || (mag == w_mag && atom < w_atom) {
            heap[worst] = (atom, score, mag);
        }
    }
    heap.len()
}

/// Post-optimization fold: O(1) reject against a cached weakest slot; the
/// O(`capacity`) rescan runs only on an accepted replacement.
fn fold_new(stream: &[f32], capacity: usize) -> usize {
    let mut heap: Vec<(u32, f32, f32)> = Vec::with_capacity(capacity);
    let mut worst_idx = 0usize;
    let recompute = |heap: &Vec<(u32, f32, f32)>| -> usize {
        let mut worst = 0usize;
        for k in 1..heap.len() {
            if heap[k].2 < heap[worst].2 || (heap[k].2 == heap[worst].2 && heap[k].0 > heap[worst].0)
            {
                worst = k;
            }
        }
        worst
    };
    for (a, &score) in stream.iter().enumerate() {
        let atom = a as u32;
        let mag = score.abs();
        if heap.len() < capacity {
            heap.push((atom, score, mag));
            if heap.len() == capacity {
                worst_idx = recompute(&heap);
            }
            continue;
        }
        let (w_atom, _, w_mag) = heap[worst_idx];
        if mag > w_mag || (mag == w_mag && atom < w_atom) {
            heap[worst_idx] = (atom, score, mag);
            worst_idx = recompute(&heap);
        }
    }
    heap.len()
}

fn bench_router(c: &mut Criterion) {
    let (rows, decoder) = fixtures();
    let streams = score_streams(&rows, &decoder);

    let mut group = c.benchmark_group("sparse_dict_router_topk");
    group.sample_size(20);

    group.bench_function("fold_old_k32768_rows64", |b| {
        b.iter(|| {
            let mut sink = 0usize;
            for stream in &streams {
                sink = sink.wrapping_add(fold_old(black_box(stream), S));
            }
            black_box(sink)
        });
    });

    group.bench_function("fold_new_k32768_rows64", |b| {
        b.iter(|| {
            let mut sink = 0usize;
            for stream in &streams {
                sink = sink.wrapping_add(fold_new(black_box(stream), S));
            }
            black_box(sink)
        });
    });

    group.bench_function("router_end_to_end_k32768_rows64", |b| {
        b.iter(|| {
            let mut sink = 0u64;
            for row in rows.outer_iter() {
                let picked = top_s_online(row, decoder.view(), S, TILE);
                sink = sink.wrapping_add(picked.len() as u64);
                sink = sink.wrapping_add(u64::from(picked[0].0));
            }
            black_box(sink)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_router);
criterion_main!(benches);
