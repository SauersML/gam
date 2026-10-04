#![cfg(test)]
//! The core path on a device against the CPU's, on the host reference backend (and on every
//! accelerator the process has): the passages' targets, every replacement's forward and KL with
//! the explanation's selection inside it, and a fit's samples, each the CPU's quantity up to the
//! summation order of its products.

use super::core_device::{Evaluator, passages, samples};
use super::explanation::{Given, Passage, Replacement};
use super::explanation_tests::{fitted, tiny_export};
use super::import::import_language_model;
use super::masked::{kl_score_only, sites};
use super::operator_program::{FamilyInputs, OperatorProgram};
use gam_gpu::tensor::Device;
use ndarray::Array2;

/// The host reference backend and any float64 accelerator.
fn devices() -> Vec<Device> {
    let mut out = vec![Device::host()];
    if let Ok(Some(d)) = Device::accelerator(gam_gpu::GpuPolicy::Auto) {
        out.push(d);
    }
    out
}

fn close(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64, what: &str) {
    assert_eq!(a.dim(), b.dim(), "{what}");
    let scale = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let worst = a.iter().zip(b.iter()).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(worst <= tolerance * scale, "{what}: off by {worst} at scale {scale}");
}

/// Per passage, the CPU's KL per row and blocks on of `replacement` with `members` replaced.
fn on_cpu(model: &OperatorProgram, replacement: &dyn Replacement, members: &[usize], passages: &[Passage]) -> Vec<(ndarray::Array1<f64>, Vec<Array2<f64>>)> {
    let masked = replacement.masked(model, members).expect("masked");
    passages
        .iter()
        .enumerate()
        .map(|(p, passage)| {
            let (trace, masks) = replacement.run(&masked, members, p, &passage.base).expect("cpu forward");
            (kl_score_only(&passage.target, &trace.values[masked.program.output]), masks)
        })
        .collect()
}

#[test]
fn every_replacement_scores_on_a_device_as_on_the_cpu() {
    let (model, passages, explanation) = fitted("core_device", 1, "blocks.0.");
    let all: Vec<usize> = (0..explanation.sites.len()).collect();
    let ours = on_cpu(&model, &explanation, &all, &passages);
    let given = Given {
        sites: explanation.sites(),
        libraries: explanation.sites.iter().map(|f| f.library.clone()).collect(),
        ranks: explanation.sites.iter().map(|f| f.ranks.clone()).collect(),
        masks: ours.iter().map(|(_, m)| m.clone()).collect(),
    };
    let which: Vec<usize> = (0..passages.len()).collect();
    for device in devices() {
        let evaluator = Evaluator::new(&device, &model, &passages).expect("evaluator");
        for members in [all.clone(), vec![0], vec![1, 3, 5], vec![4, 5]] {
            for (name, replacement) in [("ours", &explanation as &dyn Replacement), ("given", &given as &dyn Replacement)] {
                let cpu = on_cpu(&model, replacement, &members, &passages);
                // Passages one at a time and all together: the chunks change nothing.
                for chosen in [which.clone(), vec![2], vec![3, 0]] {
                    let run = evaluator.replaced(replacement, &members, &chosen).expect("device");
                    for (&p, (kl, masks)) in chosen.iter().zip(&run) {
                        let what = format!("{} {name} {members:?} passage {p}", device.name());
                        close(&kl.clone().insert_axis(ndarray::Axis(1)), &cpu[p].0.clone().insert_axis(ndarray::Axis(1)), 1e-9, &what);
                        assert_eq!(masks, &cpu[p].1, "{what}: blocks on");
                    }
                }
            }
        }
    }
}

#[test]
fn passages_and_samples_on_a_device_are_the_cpus() {
    let (model, _, explanation) = fitted("core_device_samples", 1, "blocks.0.");
    let dir = tiny_export("core_device_samples_rows", 1);
    let imported = import_language_model(&dir, 6, 12).expect("import");
    std::fs::remove_dir_all(&dir).expect("remove the export");
    let family = imported.contract.family;
    let sequence = |s: usize| family.select(&(s * 12..(s + 1) * 12).collect::<Vec<_>>());
    let bases: Vec<FamilyInputs> = (2..6).map(sequence).collect();
    let batches: Vec<FamilyInputs> = (0..3).map(sequence).collect();
    for device in devices() {
        let resident = passages(&device, &model, bases.clone()).expect("passages");
        for (p, base) in resident.iter().zip(&bases) {
            let cpu = Passage::new(&model, base.clone()).expect("cpu passage");
            close(&p.target.logits, &cpu.target.logits, 1e-12, "logits");
        }
        // Every group's samples: under none of the fitted sites, and under the first three.
        for fitted_so_far in [0, 3] {
            let mut so_far = explanation.clone();
            so_far.sites.truncate(fitted_so_far);
            let members: Vec<usize> = (0..fitted_so_far).collect();
            let hybrid = (fitted_so_far > 0).then(|| so_far.masked(&model, &members).expect("hybrid"));
            let program = hybrid.as_ref().map_or(&model, |m| &m.program);
            let wanted: Vec<_> = sites(program).into_iter().filter(|s| explanation.sites[fitted_so_far..].iter().any(|f| f.site.name == s.name)).collect();
            let families: Vec<FamilyInputs> = match &hybrid {
                None => batches.clone(),
                Some(m) => batches.iter().map(|b| so_far.execute(m, &members, b).map(|(_, masks)| m.family(b, &masks)).expect("cpu run")).collect(),
            };
            let cpu = super::site_fit::samples(program, &wanted, families, 2, 11).expect("cpu samples");
            let on = samples(&device, program, (hybrid.as_ref(), &so_far), &wanted, &batches, 2, 11).expect("device samples");
            for ((a, b), site) in on.iter().zip(&cpu).zip(&wanted) {
                let what = format!("{} {} after {fitted_so_far}", device.name(), site.name);
                close(&a.reads.mapv(f64::from), &b.reads.mapv(f64::from), 1e-6, &format!("{what}: reads"));
                // The CPU's reverse passes are proposals (f32 on the Apple GPU where it has one).
                close(&a.fisher, &b.fisher, 1e-5, &what);
                close(&a.second_moment, &b.second_moment, 1e-9, &what);
                close(&a.sensitivity.clone().insert_axis(ndarray::Axis(1)), &b.sensitivity.clone().insert_axis(ndarray::Axis(1)), 1e-5, &what);
            }
        }
    }
}

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

#[test]
fn the_sparse_code_on_a_device_is_the_cpus() {
    use super::core_device::DeviceCoder;
    use super::sparse_code::Coder;
    let (rows, d_in, d_out) = (40, 7, 6);
    for (salt, ranks) in [(1usize, vec![1usize; 18]), (2, vec![1, 2, 1, 3, 1, 1, 2, 2, 1, 1])] {
        let pieces: usize = ranks.iter().sum();
        let m = |r: usize, c: usize, s: usize| Array2::from_shape_fn((r, c), |(i, j)| noise(s * 7919 + 131 * i + j));
        let (v, u, reads, a) = (m(pieces, d_in, salt), m(pieces, d_out, salt + 1), m(rows, d_in, salt + 2), m(d_out, d_out, salt + 3));
        let fisher = a.dot(&a.t());
        let gram = u.dot(&fisher).dot(&u.t());
        let z = reads.dot(&v.t());
        let y = reads.dot(&m(d_out, d_in, salt + 4).t());
        let weights = y.dot(&fisher).dot(&u.t());
        let yfy: Vec<f64> = y.outer_iter().zip(y.dot(&fisher).outer_iter()).map(|(p, q)| p.dot(&q)).collect();
        let bits: Vec<f64> = ranks.iter().enumerate().map(|(b, r)| 2.0 + 3.0 * *r as f64 + noise(b + salt).abs()).collect();
        for nodes in [0, 16] {
            let coder = Coder::new(gram.clone(), &ranks, &bits, 3.0, nodes).expect("coder");
            let cpu = coder.code_rows(z.view(), weights.view(), &yfy, None).expect("cpu code");
            for device in devices() {
                let resident = DeviceCoder::new(&device, coder.clone()).expect("device coder");
                let up = |x: &Array2<f64>| device.upload(x.view()).expect("upload");
                let yfy_column = Array2::from_shape_vec((rows, 1), yfy.clone()).expect("column");
                let (on, upper, lower) = resident.code(&device, (&up(&z), &up(&weights), &up(&yfy_column))).expect("device code");
                let on = device.download(&on).expect("download");
                for (r, (sets, code, bound)) in cpu.iter().enumerate() {
                    let what = format!("{} ranks {ranks:?} nodes {nodes} row {r}", device.name());
                    let device_sets: Vec<bool> = on.row(r).iter().map(|x| *x == 1.0).collect();
                    assert_eq!(&device_sets, sets, "{what}: blocks on");
                    assert!((upper[r] - code).abs() <= 1e-9 * code.abs().max(1.0), "{what}: code {} against {code}", upper[r]);
                    assert!((lower[r] - bound).abs() <= 1e-9 * bound.abs().max(1.0), "{what}: bound {} against {bound}", lower[r]);
                }
            }
        }
    }
}
