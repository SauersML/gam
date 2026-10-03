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

/// The masked fit of `sequence` (6 tokens of the small rotary model): selection from every piece
/// on, then one pieces step, with the counts and preconditioners carried in.
fn fit_sequence(
    masked: &mut super::masked::Masked,
    family: &super::operator_program::FamilyInputs,
    sequence: usize,
    context: &Context,
    running: &mut Running,
) -> (Vec<Array2<f64>>, Option<(f64, f64)>) {
    use super::masked::{Claim, Target, previous_inputs, select, step_pieces};
    let rows: Vec<usize> = (sequence * 6..(sequence + 1) * 6).collect();
    let inputs = family.select(&rows);
    let target = Target::every_row(masked.program.execute(&masked.family(&inputs, &masked.all_pieces().iter().map(|p| Array2::ones((6, *p))).collect::<Vec<_>>()), false).expect("forward").values[masked.program.output].mapv(|v| v * 1.1));
    let start: Vec<Array2<f64>> = masked.all_pieces().iter().map(|p| Array2::ones((6, *p))).collect();
    let coder = context.coder(previous_inputs(&inputs));
    let (masks, _) = select(masked, &inputs, &target, start, &coder, 64.0, 2).expect("select");
    let step = step_pieces(masked, &inputs, &target, &masks, 2, 0xF00D + sequence as u64, running, Claim::Corner).expect("step");
    (masks, step)
}

#[test]
fn a_resumed_fit_takes_the_uninterrupted_fits_next_decisions_bit_for_bit() {
    use super::device_program_tests::fixture;
    use super::masked::{Masked, sites};
    let (program, family) = fixture();
    let all = sites(&program);
    let libraries: Vec<Library> = all
        .iter()
        .enumerate()
        .map(|(k, site)| {
            let (d_out, d_in) = super::masked::matrix(&program, site).expect("map").dim();
            Library {
                v: Array2::from_shape_fn((3, d_in), |(i, j)| 0.5 * noise(10_000 * k + 37 * i + j)),
                u: Array2::from_shape_fn((3, d_out), |(i, j)| 0.5 * noise(10_000 * k + 5000 + 37 * i + j)),
                mean: Array1::zeros(d_in),
            }
        })
        .collect();
    let counts = |masked: &Masked| Context::new(&masked.all_pieces());
    // Uninterrupted: sequences 0 and 1.
    let mut through = Masked::build(&program, all.clone(), libraries.clone()).expect("masked");
    let mut running = Running::default();
    let context = counts(&through);
    fit_sequence(&mut through, &family, 0, &context, &mut running);
    let expected = fit_sequence(&mut through, &family, 1, &context, &mut running);
    // Interrupted after sequence 0: checkpointed, then a fresh process rebuilt from the checkpoint.
    let dir = std::env::temp_dir().join(format!("gam-mpd-resume-{}", std::process::id()));
    {
        let mut first = Masked::build(&program, all.clone(), libraries).expect("masked");
        let mut running = Running::default();
        fit_sequence(&mut first, &family, 0, &context, &mut running);
        let saved: Vec<Library> = (0..first.sites.len()).map(|k| first.library(k).expect("library")).collect();
        save(&dir, &Saved { driver: &json!({"next": 1}), libraries: &saved, context: &context, running: &running, sets: Vec::new() }).expect("saved");
    }
    let loaded = load(&dir).expect("readable").expect("a checkpoint");
    let mut resumed = Masked::build(&program, all, loaded.libraries).expect("masked");
    let mut running = loaded.running;
    let got = fit_sequence(&mut resumed, &family, 1, &loaded.context, &mut running);
    std::fs::remove_dir_all(&dir).expect("removed");
    assert_eq!(got.0, expected.0, "the resumed selection's sets");
    assert_eq!(got.1.map(|(a, b)| (a.to_bits(), b.to_bits())), expected.1.map(|(a, b)| (a.to_bits(), b.to_bits())), "the resumed step's totals");
    for k in 0..through.sites.len() {
        let (a, b) = (through.library(k).expect("library"), resumed.library(k).expect("library"));
        assert!(a.v.iter().zip(b.v.iter()).chain(a.u.iter().zip(b.u.iter())).all(|(x, y)| x.to_bits() == y.to_bits()), "site {k}'s library");
    }
}
