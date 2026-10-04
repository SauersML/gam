#![cfg(test)]
//! The device trainer's operations against their definitions, and a few steps of it on the small
//! rotary language model of `device_program_tests`.

use super::blocks::Generic;
use gam_linalg::decompose::svd;
use super::device_program_tests::{devices, fixture, noise};
use super::device_train::{Settings, Trainer, statistics};
use super::masked::{Library, Masked, matrix, sites};
use gam_gpu::tensor::{Arithmetic, Device, Tensor};
use gam_linalg::faer_ndarray::fast_atb;
use ndarray::{Array1, Array2, Axis};

fn upload(device: &Device, m: &Array2<f64>) -> Tensor {
    device.upload(m.view()).expect("upload")
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
        let fishers: Vec<Array2<f64>> = measured.iter().map(|m| m.fisher.clone()).collect();
        let masked = Masked::build(&program, chosen.clone(), libraries.clone()).expect("masked");
        let settings = Settings { observations, rate: 1e-3, betas: (0.9, 0.999), epsilon: 1e-12 };
        let mut trainer = Trainer::new(&device, (&program, &chosen), masked, (&describe, &fishers), settings, Arithmetic::F64).expect("trainer");
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
