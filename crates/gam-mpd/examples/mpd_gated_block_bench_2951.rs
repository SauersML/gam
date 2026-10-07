//! Time of a gated block's forward and reverse passes (#2951), each row reading and writing only
//! its components on (`DeviceProgram`'s per-row lists), against the dense products of the same
//! program (`DeviceProgram::read_densely`). The block is an MLP of `COMPONENTS` gated components
//! (widths 1, 2, 3 in turn), each with a direction gate `g_bᵀx − τ_b` on rows `x` of `D` standard
//! normal entries, `g_b` of unit norm and `τ_b` the standard normal's `1 − P` quantile, so each
//! component is on in a fraction `P` of the rows: the reads `V_fc x` gated, written into `HIDDEN`
//! columns by `U_fc`, a GELU, read by `V_dn`, gated by the same gate, written into `D` columns by
//! `U_dn` (library_vpd's grouped direction arm, one block).
//!
//! `mpd_gated_block_bench_2951 D HIDDEN COMPONENTS ROWS REPS P...`
//!
//! On the single-precision device, per `P`, arm (lists, dense) and arithmetic (f32; bfloat16 on
//! CUDA): the forward pass and one reverse pass into every operator and the input, each timed to a
//! device synchronization `REPS` times after a warm-up. One JSON line per `P` to stdout: the
//! median seconds per part and the entries the lists hold.

use gam_gpu::{
    GpuPolicy,
    tensor::{Arithmetic, Device},
};
use gam_mpd::{
    device_program::DeviceProgram,
    operator_program::{Declarations, FamilyInputs, Group, Interface, Label, LabelKind, Law, Node, Operator, OperatorProgram, SequenceLayout, Slot, SlotValues, exact_precision},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use statrs::distribution::{ContinuousCDF, Normal};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A standard normal matrix (Box–Muller).
fn normal(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || {
        let (u, v): (f64, f64) = (rng.random::<f64>().max(f64::MIN_POSITIVE), rng.random());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "D HIDDEN COMPONENTS ROWS REPS P...";
    let [d, hidden, components, rows, reps, ps @ ..] = &args[..] else {
        return Err(format!("usage: {usage}"));
    };
    let parse = |s: &str| s.parse::<usize>().map_err(|e| format!("{s}: {e} ({usage})"));
    let (d, hidden, components, rows, reps) = (parse(d)?, parse(hidden)?, parse(components)?, parse(rows)?, parse(reps)?);
    let ps: Vec<f64> = ps.iter().map(|p| p.parse::<f64>().map_err(error)).collect::<Result<_, _>>()?;
    let device = Device::single_precision(GpuPolicy::Auto).map_err(error)?.ok_or("no single-precision device")?;
    let widths: Vec<usize> = (0..components).map(|b| 1 + b % 3).collect();
    let n: usize = widths.iter().sum();
    let reads = Interface::new(widths.iter().enumerate().map(|(b, &w)| Group { width: w, label: Label::new(LabelKind::Unit, b as u32) }).collect()).map_err(error)?;
    let units = Interface::uniform(components, 1, LabelKind::Unit, 0).map_err(error)?;
    let native = |w: usize| Interface::native(w).map_err(error);
    let dense = |name: &str, r: Interface, c: Interface, v: Array2<f64>| -> Result<Arc<Operator>, String> {
        let precision = exact_precision(v.iter().copied()).map_err(error)?;
        Ok(Arc::new(Operator::dense(name, r, c, v, precision, Default::default()).map_err(error)?))
    };
    let mut rng = StdRng::seed_from_u64(7);
    let mut g = normal(&mut rng, components, d);
    for mut row in g.rows_mut() {
        let norm = row.dot(&row).sqrt();
        row.mapv_inplace(|v| v / norm);
    }
    let scale = |fan_in: usize| 1.0 / (fan_in as f64).sqrt();
    let (v_fc, u_fc) = (normal(&mut rng, n, d) * scale(d), normal(&mut rng, hidden, n) * scale(n));
    let (v_dn, u_dn) = (normal(&mut rng, n, hidden) * scale(hidden), normal(&mut rng, d, n) * scale(n));
    let x = normal(&mut rng, rows, d);
    let seed = normal(&mut rng, rows, d);
    let family = FamilyInputs { rows, slots: vec![SlotValues::Raw(x)], layout: Some(SequenceLayout { sequence: vec![0; rows], position: (0..rows as u32).collect() }) };
    let half = cfg!(target_os = "linux");
    for &p in &ps {
        let tau = Normal::standard().inverse_cdf(1.0 - p);
        let operators = vec![
            dense("G", units.clone(), native(d)?, g.clone())?,
            dense("minus_tau", units.clone(), Interface::constant(), Array2::from_elem((components, 1), -tau))?,
            dense("V_fc", reads.clone(), native(d)?, v_fc.clone())?,
            dense("U_fc", native(hidden)?, reads.clone(), u_fc.clone())?,
            dense("V_dn", reads.clone(), native(hidden)?, v_dn.clone())?,
            dense("U_dn", native(d)?, reads.clone(), u_dn.clone())?,
        ];
        let nodes = vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
            Node::Affine { terms: vec![(0, 2)], bias: None },
            Node::Gated { value: 2, gate: 1, scale: None },
            Node::Affine { terms: vec![(3, 3)], bias: None },
            Node::Pointwise { input: 4, laws: vec![Law::GeluTanh] },
            Node::Affine { terms: vec![(5, 4)], bias: None },
            Node::Gated { value: 6, gate: 1, scale: None },
            Node::Affine { terms: vec![(7, 5)], bias: None },
        ];
        let program = OperatorProgram { declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: d }], parameters: 0 }, operators, bases: vec![], rules: vec![], nodes, output: 8 };
        let trainable: Vec<usize> = (0..6).collect();
        let seeds = || -> Result<BTreeMap<usize, gam_gpu::tensor::Tensor>, String> { Ok(BTreeMap::from([(8, device.upload(seed.view()).map_err(error)?)])) };
        let mut record = serde_json::Map::new();
        record.insert("p".into(), json!(p));
        for arm in ["lists", "dense"] {
            let mut lowered = DeviceProgram::compile_values(&device, &program)?;
            if arm == "dense" {
                lowered.read_densely();
            }
            let arithmetics: Vec<(&str, Arithmetic)> = if half { vec![("f32", Arithmetic::F32), ("bf16", Arithmetic::Bf16)] } else { vec![("f32", Arithmetic::F32)] };
            for (name, arithmetic) in arithmetics {
                lowered.set_arithmetic(arithmetic);
                let (mut forward, mut reverse) = (Vec::new(), Vec::new());
                let mut listed = None;
                for _ in 0..=reps {
                    let started = Instant::now();
                    let trace = lowered.forward(&family)?;
                    device.synchronize().map_err(error)?;
                    forward.push(started.elapsed().as_secs_f64());
                    let started = Instant::now();
                    let out = lowered.vjp_values_dense(&trace, seeds()?, &[0], &trainable, arithmetic)?;
                    device.synchronize().map_err(error)?;
                    reverse.push(started.elapsed().as_secs_f64());
                    drop(out);
                    listed = lowered.entries_listed(&trace, 7);
                }
                forward.remove(0);
                reverse.remove(0);
                record.insert(format!("{arm}_{name}"), json!({ "forward_s": median(forward), "reverse_s": median(reverse), "entries_listed": listed, "entries": rows * n }));
            }
        }
        println!("{}", serde_json::Value::Object(record));
    }
    Ok(())
}
