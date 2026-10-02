#![cfg(test)]
//! The masked fit's device path against its CPU twins in `masked`, on the small rotary language
//! model of `device_program_tests` with every site replaced by a random three-piece library and
//! random masks. The bands are that module's: the KL within twice the logits' band plus each
//! evaluation's rounding; every proposal (mask gradients, pieces' gradients, Fishers, covariances,
//! the step's curvature) within the f32 proposal band of the reference's largest entry.

use super::derivatives::jvp;
use super::device_program_tests::{assert_proposal, devices, fixture, gamma, noise, softmax};
use super::masked::{self, Library, Masked, Target, matrix, read_values, sites};
use super::masked_device::Accelerated;
use super::operator_program::FamilyInputs;
use gam_gpu::tensor::Arithmetic;
use gam_linalg::faer_ndarray::fast_atb;
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;

const PIECES: usize = 3;

fn masked_fixture() -> (Masked, FamilyInputs, Vec<Array2<f64>>, Array2<f64>) {
    let (program, base) = fixture();
    let all = sites(&program);
    let mut libraries = Vec::new();
    for (k, site) in all.iter().enumerate() {
        let (d_out, d_in) = matrix(&program, site).expect("map").dim();
        let at = |salt: usize| move |(i, j): (usize, usize)| noise(10_000 * k + 1000 * salt + 37 * i + j);
        libraries.push(Library {
            v: Array2::from_shape_fn((PIECES, d_in), at(1)) * 0.5,
            u: Array2::from_shape_fn((PIECES, d_out), at(2)) * 0.5,
            mean: Array1::from_shape_fn(d_in, |i| 0.1 * noise(10_000 * k + 7 * i + 3)),
        });
    }
    let masks: Vec<Array2<f64>> = (0..all.len())
        .map(|k| Array2::from_shape_fn((base.rows, PIECES), |(r, c)| if noise(500 * k + 11 * r + c + 1) > -0.2 { 1.0 } else { 0.0 }))
        .collect();
    let clean = program.execute(&base, false).expect("clean").values[program.output].clone();
    let masked = Masked::build(&program, all, libraries).expect("masked");
    (masked, base, masks, clean)
}

fn targets(clean: &Array2<f64>) -> [Target; 2] {
    let half: Vec<bool> = (0..clean.nrows()).map(|r| r % 3 != 1).collect();
    [Target::every_row(clean.clone()), Target { logits: clean.clone(), scored: Some(half) }]
}

#[test]
fn the_masked_device_path_matches_the_cpu_within_bands() {
    let (masked, base, masks, clean) = masked_fixture();
    let family = masked.family(&base, &masks);
    let banded = masked.program.execute(&family, true).expect("banded");
    let output_band = &banded.bands.as_ref().expect("bands")[masked.program.output];
    let output_ball = &banded.balls.as_ref().expect("balls")[masked.program.output];
    let widest = 16;
    for target in targets(&clean) {
        let (kl, trace, cotangent) = masked::forward(&masked, &family, &target).expect("cpu forward");
        let mask_gradients = masked::mask_gradients(&masked, &family, &trace, cotangent.clone()).expect("cpu mask gradients");
        let gradients = masked::gradients(&masked, &family, &trace, &masks, cotangent).expect("cpu gradients");
        let fisher = masked::fisher(&masked, &family, &trace, &target, 2, 0x5EED, true).expect("cpu fisher");
        let covariances: Vec<Array2<f64>> = masked
            .sites
            .iter()
            .enumerate()
            .map(|(k, site)| {
                let centred = &read_values(&trace, site).expect("reads") - &masked.libraries[k].mean;
                fast_atb(&centred, &centred) / family.rows as f64
            })
            .collect();
        // A direction on the first site's pieces, by operator name, and the CPU's curvature on it.
        let named = |suffix: &str| {
            let name = format!("{}·{suffix}", masked.sites[0].name);
            masked.program.operators.iter().position(|op| op.name == name).expect("a library operator")
        };
        let mut tangents = BTreeMap::new();
        for (salt, suffix) in ["V0", "centre", "U0"].into_iter().enumerate() {
            let op = named(suffix);
            let (rows, cols) = masked.program.operators[op].matrix().dim();
            tangents.insert(op, Array2::from_shape_fn((rows, cols), |(i, j)| noise(90_000 + 1000 * salt + 31 * i + j)));
        }
        let output = jvp(&masked.program, &family, &trace, &tangents).expect("cpu jvp");
        let logits = &trace.values[masked.program.output];
        let mut quadratic = 0.0;
        for r in (0..family.rows).filter(|r| target.scores(*r)) {
            let q = softmax(logits.row(r));
            let t = output.row(r);
            let mean: f64 = q.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
            quadratic += q.iter().zip(t.iter()).map(|(a, b)| a * (b - mean) * (b - mean)).sum::<f64>();
        }
        for device in devices() {
            let name = device.name();
            let accelerated = Accelerated::new(&device, &masked, Arithmetic::F64).expect("lowered");
            let on_device = accelerated.target(&target).expect("target");
            let state = accelerated.forward(&family, &on_device).expect("device forward");
            for r in 0..family.rows {
                if !target.scores(r) {
                    assert_eq!(state.kl[r], 0.0, "{name}: an unscored row has no KL");
                    continue;
                }
                let (p, q) = (softmax(target.logits.row(r)), softmax(logits.row(r)));
                let magnitude: f64 = p.iter().zip(q.iter()).map(|(a, b)| a * (a.ln().abs() + b.ln().abs())).sum();
                let moved = 2.0 * (output_band.row(r).iter().fold(0.0_f64, |m, v| m.max(*v)) + output_ball[r]);
                let band = 2.0 * moved + 2.0 * gamma(clean.ncols() + 8) * magnitude;
                assert!((state.kl[r] - kl[r]).abs() <= band, "{name}: KL of row {r} differs by {:e}, band {band:e}", (state.kl[r] - kl[r]).abs());
            }
            for (k, g) in accelerated.mask_gradients(&masked, &state).expect("device mask gradients").iter().enumerate() {
                assert_proposal(&format!("{name}: mask gradient of site {k}"), g, &mask_gradients[k], widest);
            }
            for (k, (m, v, u)) in accelerated.gradients(&masked, &state, &masks).expect("device gradients").iter().enumerate() {
                assert_proposal(&format!("{name}: mask gradient of site {k}"), m, &gradients[k].0, widest);
                assert_proposal(&format!("{name}: V gradient of site {k}"), v, &gradients[k].1, family.rows);
                assert_proposal(&format!("{name}: U gradient of site {k}"), u, &gradients[k].2, family.rows);
            }
            for (k, (h, f)) in accelerated.fisher(&masked, &state, &on_device, 2, 0x5EED, true).expect("device fisher").iter().enumerate() {
                assert_proposal(&format!("{name}: mask Fisher of site {k}"), h, &fisher[k].0, widest);
                let (f, reference) = (f.as_ref().expect("written"), fisher[k].1.as_ref().expect("written"));
                assert_proposal(&format!("{name}: written Fisher of site {k}"), f, reference, family.rows);
            }
            for (k, c) in accelerated.covariances(&masked, &state).expect("device covariances").iter().enumerate() {
                assert_proposal(&format!("{name}: read covariance of site {k}"), c, &covariances[k], family.rows);
            }
            let device_quadratic = accelerated.quadratic(&state, &on_device, &tangents).expect("device quadratic");
            let band = (widest + 2) as f64 * 2f64.powi(-24) * quadratic.abs();
            assert!((device_quadratic - quadratic).abs() <= band, "{name}: curvature {device_quadratic} against {quadratic}");
        }
    }
}
