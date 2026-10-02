#![cfg(test)]
//! Checkpoints round-trip bit for bit, and a save interrupted before its commit leaves the last
//! committed generation whole.

use super::checkpoint::{Saved, SparseSets, load, save};
use super::masked::{Context, Library, Running};
use ndarray::{Array1, Array2};
use serde_json::json;

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn matrix(rows: usize, cols: usize, seed: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| noise(seed + i * cols + j) * 1e3_f64.powi((i % 7) as i32 - 3))
}

fn state(seed: usize) -> (Vec<Library>, Context, Running, Vec<Option<SparseSets>>) {
    let shapes = [(5, 4, 3), (2, 6, 6)];
    let libraries: Vec<Library> = shapes
        .iter()
        .enumerate()
        .map(|(k, &(c, d_in, d_out))| Library {
            v: matrix(c, d_in, seed + 100 * k),
            u: matrix(c, d_out, seed + 100 * k + 50),
            mean: Array1::from_shape_fn(d_in, |i| noise(seed + 1000 + i)),
        })
        .collect();
    let counts = |offset: usize| shapes.iter().enumerate().map(|(k, s)| Array1::from_shape_fn(s.0, |c| (seed + offset + k + c) as f64 * 0.5)).collect();
    let context = Context { stayed: counts(1), was_on: counts(2), new: counts(3) };
    let running = Running {
        covariances: shapes.iter().map(|s| matrix(s.1, s.1, seed + 7)).collect(),
        fishers: shapes.iter().map(|s| matrix(s.2, s.2, seed + 9)).collect(),
        rows: 1024.0 + noise(seed),
    };
    let sets = vec![
        Some(vec![(vec![0, 2, 2, 3], vec![1, 4, 0]), (vec![0, 0, 1, 1], vec![1])]),
        None,
        Some(vec![(vec![0, 0, 0, 0], vec![]), (vec![0, 2, 2, 2], vec![0, 1])]),
    ];
    (libraries, context, running, sets)
}

fn bits(m: &Array2<f64>) -> Vec<u64> {
    m.iter().map(|x| x.to_bits()).collect()
}

fn same(a: &(Vec<Library>, Context, Running, Vec<Option<SparseSets>>), b: &(Vec<Library>, Context, Running, Vec<Option<SparseSets>>)) {
    assert_eq!(a.0.len(), b.0.len());
    for (x, y) in a.0.iter().zip(&b.0) {
        assert_eq!((x.v.dim(), x.u.dim()), (y.v.dim(), y.u.dim()));
        assert_eq!(bits(&x.v), bits(&y.v));
        assert_eq!(bits(&x.u), bits(&y.u));
        assert_eq!(x.mean.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), y.mean.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(a.1.stayed, b.1.stayed);
    assert_eq!(a.1.was_on, b.1.was_on);
    assert_eq!(a.1.new, b.1.new);
    assert_eq!(a.2.rows.to_bits(), b.2.rows.to_bits());
    for (x, y) in a.2.covariances.iter().zip(&b.2.covariances).chain(a.2.fishers.iter().zip(&b.2.fishers)) {
        assert_eq!(bits(x), bits(y));
    }
    assert_eq!(a.3, b.3);
}

fn save_state(dir: &std::path::Path, s: &(Vec<Library>, Context, Running, Vec<Option<SparseSets>>), driver: &serde_json::Value) {
    save(dir, &Saved { driver, libraries: &s.0, context: &s.1, running: &s.2, sets: s.3.iter().map(Option::as_ref).collect() }).expect("saved");
}

fn loaded(dir: &std::path::Path) -> ((Vec<Library>, Context, Running, Vec<Option<SparseSets>>), serde_json::Value) {
    let l = load(dir).expect("readable").expect("a checkpoint");
    ((l.libraries, l.context, l.running, l.sets), l.driver)
}

#[test]
fn checkpoints_round_trip_bit_for_bit_and_survive_an_interrupted_save() {
    let root = std::env::temp_dir().join(format!("gam-mpd-checkpoint-{}", std::process::id()));
    let dir = root.join("run.checkpoint");
    assert!(load(&dir).expect("no directory is no checkpoint").is_none());
    let first = state(1);
    let driver = json!({"pass": 0, "sequence": 3, "previous": 0.1 + 0.2, "points": [{"kl": 1.0 / 3.0}]});
    save_state(&dir, &first, &driver);
    let (back, back_driver) = loaded(&dir);
    same(&first, &back);
    assert_eq!(back_driver, driver);
    assert_eq!(back_driver["previous"].as_f64().map(f64::to_bits), Some((0.1_f64 + 0.2).to_bits()));

    // A second save replaces the first and removes its arrays.
    let second = state(2);
    save_state(&dir, &second, &json!({"pass": 1}));
    same(&second, &loaded(&dir).0);
    assert!(!dir.join("g0.f64.npy").exists() && dir.join("g1.f64.npy").exists());

    // A save killed after writing part of its arrays, before the manifest's rename: the committed
    // generation still loads whole.
    std::fs::write(dir.join("g2.f64.npy"), b"\x93NUMPY partial").expect("written");
    std::fs::write(dir.join("manifest.json.tmp"), b"{\"generation\": 2").expect("written");
    same(&second, &loaded(&dir).0);
    // And the next save overwrites the debris.
    save_state(&dir, &first, &json!({"pass": 2}));
    same(&first, &loaded(&dir).0);
    std::fs::remove_dir_all(&root).expect("removed");
}
