#![cfg(test)]
//! The device trainer's operations against their definitions, and a few steps of it on the small
//! rotary language model of `device_program_tests`.

use super::blocks::Generic;
use super::dense::svd;
use super::device_program_tests::{devices, fixture, noise};
use super::device_train::{Settings, Trainer, statistics};
use super::masked::{Library, Masked, matrix, sites};
use gam_gpu::tensor::{Arithmetic, Device, Tensor};
use gam_linalg::faer_ndarray::fast_atb;
use ndarray::{Array1, Array2, Axis};

fn upload(device: &Device, m: &Array2<f64>) -> Tensor {
    device.upload(m.view()).expect("upload")
}

fn code(size: &[f64], bits: &[f64], left: f64, weight: f64, mask: &[f64]) -> f64 {
    let bound = left + size.iter().zip(mask).filter(|(_, m)| **m == 0.0).map(|(s, _)| s).sum::<f64>();
    bits.iter().zip(mask).filter(|(_, m)| **m == 1.0).map(|(b, _)| b).sum::<f64>() + weight * bound * bound
}

#[test]
fn selected_sets_are_no_worse_than_all_on_and_no_single_flip_lowers_them() {
    let (rows, cols) = (7, 37);
    let a = Array2::from_shape_fn((rows, cols), |(r, c)| noise(13 * r + c + 1));
    let q = Array2::from_shape_fn((rows, cols), |(r, c)| noise(900 + 17 * r + c).abs());
    let bits = Array2::from_shape_fn((1, cols), |(_, c)| if c % 11 == 3 { 0.0 } else { 1.0 + 4.0 * noise(500 + c).abs() });
    let left = Array2::from_shape_fn((rows, 1), |(r, _)| 0.1 * noise(700 + r).abs());
    let weight = Array2::from_shape_fn((rows, 1), |(r, _)| 2.0 + 10.0 * noise(800 + r).abs());
    let mut chosen = Vec::new();
    for device in devices() {
        let mut mask = upload(&device, &Array2::ones((rows, cols)));
        device
            .select_sets((&upload(&device, &a), &upload(&device, &q)), &upload(&device, &bits), &upload(&device, &left), &upload(&device, &weight), &mut mask)
            .expect("select");
        let mask = device.download(&mask).expect("download");
        for r in 0..rows {
            let size: Vec<f64> = (0..cols).map(|c| a[[r, c]].abs() * q[[r, c]]).collect();
            let row: Vec<f64> = mask.row(r).to_vec();
            let held = code(&size, bits.row(0).as_slice().expect("bits"), left[[r, 0]], weight[[r, 0]], &row);
            assert!(held <= code(&size, bits.row(0).as_slice().expect("bits"), left[[r, 0]], weight[[r, 0]], &vec![1.0; cols]) + 1e-12);
            for c in 0..cols {
                let mut flipped = row.clone();
                flipped[c] = 1.0 - flipped[c];
                assert!(code(&size, bits.row(0).as_slice().expect("bits"), left[[r, 0]], weight[[r, 0]], &flipped) >= held - 1e-9 * held.abs().max(1.0));
            }
        }
        chosen.push(mask);
    }
    for other in &chosen[1..] {
        assert_eq!(other, &chosen[0], "the device's sets are the host's");
    }
}

#[test]
fn the_box_charges_gradients_are_its_derivatives() {
    let (rows, cols) = (3, 9);
    let z = Array2::from_shape_fn((rows, cols), |(r, c)| noise(31 * r + c + 5));
    let mask = Array2::from_shape_fn((rows, cols), |(r, c)| if noise(77 * r + c) > 0.1 { 1.0 } else { 0.0 });
    let q = Array2::from_shape_fn((rows, cols), |(r, c)| 0.2 + noise(400 + 7 * r + c).abs());
    let charge = |z: &Array2<f64>, q: &Array2<f64>| -> f64 {
        (0..rows).map(|r| 0.5 * (0..cols).map(|c| (1.0 - mask[[r, c]]) * z[[r, c]].abs() * q[[r, c]]).sum::<f64>().powi(2)).sum()
    };
    for device in devices() {
        let mut cot = upload(&device, &Array2::zeros((rows, cols)));
        let mut coefficient = upload(&device, &Array2::zeros((rows, cols)));
        let values = device.box_charge(&upload(&device, &z), &upload(&device, &mask), &upload(&device, &q), &mut cot, &mut coefficient).expect("charge");
        assert!((values.iter().sum::<f64>() - charge(&z, &q)).abs() < 1e-12);
        let (cot, coefficient) = (device.download(&cot).expect("cot"), device.download(&coefficient).expect("coefficient"));
        let h = 1e-6;
        for r in 0..rows {
            for c in 0..cols {
                let (mut up, mut down) = (z.clone(), z.clone());
                up[[r, c]] += h;
                down[[r, c]] -= h;
                assert!((cot[[r, c]] - (charge(&up, &q) - charge(&down, &q)) / (2.0 * h)).abs() < 1e-6);
                let (mut up, mut down) = (q.clone(), q.clone());
                up[[r, c]] += h;
                down[[r, c]] -= h;
                let along_q = (charge(&z, &up) - charge(&z, &down)) / (2.0 * h);
                assert!((coefficient[[r, c]] * q[[r, c]] - along_q).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn adam_takes_its_bias_corrected_step() {
    for device in devices() {
        let w0 = Array2::from_shape_fn((2, 3), |(i, j)| noise(3 * i + j));
        let g = Array2::from_shape_fn((2, 3), |(i, j)| noise(50 + 3 * i + j));
        let mut w = upload(&device, &w0);
        let (mut m, mut v) = (upload(&device, &Array2::zeros((2, 3))), upload(&device, &Array2::zeros((2, 3))));
        device.adam(&mut w, (&mut m, &mut v), &upload(&device, &g), 0.01, (0.9, 0.999, 1e-8), 1).expect("adam");
        let w = device.download(&w).expect("w");
        // The first bias-corrected step is `rate · g / (|g| + ε)`.
        for ((after, before), gi) in w.iter().zip(&w0).zip(&g) {
            assert!((after - (before - 0.01 * gi / (gi.abs() + 1e-8))).abs() < 1e-12);
        }
    }
}

#[test]
fn trained_libraries_keep_their_map_and_code_the_inputs_shorter() {
    let (program, base) = fixture();
    let chosen = sites(&program);
    let mut libraries = Vec::new();
    for site in &chosen {
        // Each map's own singular pieces, split in balanced factors: exactly the map.
        let w = matrix(&program, site).expect("map");
        let decomposed = svd(w.view(), false).expect("svd");
        let roots = decomposed.singular_values.mapv(f64::sqrt);
        let u = (&decomposed.u * &roots).t().to_owned();
        let v = &decomposed.vt * &roots.view().insert_axis(Axis(1));
        libraries.push(Library { mean: Array1::zeros(v.ncols()), v, u });
    }
    let observations = 1e3;
    for device in devices() {
        let measured = statistics(&device, &program, &chosen, std::slice::from_ref(&base), 2, 7).expect("statistics");
        let describe = Generic::new(&measured, observations);
        let masked = Masked::build(&program, chosen.clone(), libraries.clone()).expect("masked");
        let settings = Settings { observations, rate: 1e-3, betas: (0.9, 0.999), epsilon: 1e-12, draws: 2, seed: 11 };
        let mut trainer = Trainer::new(&device, &program, &chosen, masked, &describe, settings, Arithmetic::F64).expect("trainer");
        let before = trainer.evaluate(&base).expect("before");
        for _ in 0..8 {
            trainer.train(&base).expect("train");
            trainer.update().expect("update");
        }
        let masked = trainer.sync(&describe).expect("sync");
        for (k, site) in chosen.iter().enumerate() {
            let library = masked.library(k).expect("library");
            let w = matrix(&program, site).expect("map");
            let drift = (&fast_atb(&library.u, &library.v) - &w).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            assert!(drift < 1e-10 * w.iter().fold(0.0_f64, |m, x| m.max(x.abs())), "{}: the sum drifted {drift}", site.name);
        }
        let after = trainer.evaluate(&base).expect("after");
        assert!(after.code(observations) < before.code(observations), "{:?} -> {:?}", before, after);
    }
}
