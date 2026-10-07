//! The graph checker's runs on a device (#2951): the same circuit semantics as `graph::run` (each
//! unit's route inputs from the actual and stand-in streams, its block computed on them, the
//! logits at the scored rows), with every array resident on Metal or CUDA through `gam_gpu`'s
//! tensor operations and the attention of `device_attention`. `M`'s weights stay resident between
//! runs; a weight edit marks the matrices it touches, which the next run uploads again.
//!
//! Covered: native heads (with head norms, rotary, grouped keys and values) and MLP neurons (plain
//! or gated), swaps, counterfactual and average stand-ins, `Reference` captures. Anything else (a
//! transcoder or VPD block, site operations, attention blocks, statistics recording) returns
//! `None` and runs on the host.
use crate::{
    device_attention::{Segment, forward_segments},
    device_program::{gelu_tanh_constant, law_of},
    graph::{Block, Circuit, Execution, Incoming, Reference, Weights, Writer},
    operator_program::Rotary,
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Arithmetic, Device, Op, Storage, Tensor},
};
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

/// The device and its resident copies of host matrices, by address and shape.
pub(crate) struct DeviceState {
    device: Device,
    resident: HashMap<(usize, usize, usize), Tensor>,
}

static DEVICE: OnceLock<Mutex<DeviceState>> = OnceLock::new();

impl DeviceState {
    pub(crate) fn new(device: Device) -> Self {
        Self { device, resident: HashMap::new() }
    }

    fn arithmetic(&self) -> Arithmetic {
        match self.device.storage() {
            Storage::F64 => Arithmetic::F64,
            Storage::F32 | Storage::Bf16 => Arithmetic::F32,
        }
    }

    /// The key of host matrix `m`'s resident copy, uploaded when absent or edited since.
    fn ensure(&mut self, m: ArrayView2<f64>) -> Result<Key, GpuError> {
        let key = (m.as_ptr() as usize, m.nrows(), m.ncols());
        if !self.resident.contains_key(&key) {
            let t = self.device.upload(m)?;
            self.resident.insert(key, t);
        }
        Ok(key)
    }

    /// The resident copy under `key` ([`DeviceState::ensure`]).
    fn get(&self, key: Key) -> Result<&Tensor, GpuError> {
        self.resident.get(&key).ok_or_else(|| GpuError::DriverCallFailed { reason: "a resident matrix went missing".into() })
    }
}

type Key = (usize, usize, usize);

/// A host vector as a `1 × n` row view.
fn row(v: &Array1<f64>) -> ArrayView2<'_, f64> {
    v.view().insert_axis(ndarray::Axis(0))
}

/// Runs the checker on `device` for the rest of the process: the large products of host runs
/// ([`dot`], [`logits`]) and whole runs ([`run`]). Returns false when a device was already set.
pub fn use_device(device: Device) -> bool {
    DEVICE.set(Mutex::new(DeviceState::new(device))).is_ok()
}

/// A weight edit is about to change host matrix `m` (in place or by replacing it): every resident
/// copy of that shape is dropped, so neither `m` nor a matrix later allocated at a reused address
/// reads a stale copy.
pub(crate) fn edited(m: &Array2<f64>) {
    if let Some(state) = DEVICE.get()
        && let Ok(mut s) = state.lock()
    {
        s.resident.retain(|k, _| (k.1, k.2) != m.dim());
    }
}

/// `a · b` on the device, or `None` (no device, a product below `2^22` multiply-adds, or a device
/// error, after which the host multiplies).
pub(crate) fn dot(a: &Array2<f64>, b: ArrayView2<f64>) -> Option<Array2<f64>> {
    let state = DEVICE.get()?;
    if a.nrows() * a.ncols() * b.ncols() < 1 << 22 {
        return None;
    }
    let s = state.lock().ok()?;
    let d = &s.device;
    let product = |w: &Tensor, op: Op| -> Result<Array2<f64>, GpuError> {
        let x = d.upload(a.view())?;
        let mut out = d.zeros(a.nrows(), b.ncols())?;
        d.gemm(&mut out, 1.0, &x, Op::N, w, op, 0.0, s.arithmetic())?;
        d.download(&out)
    };
    // A transposed view of a stored matrix multiplies as the stored matrix with `Op::T`.
    let stored = b.t();
    let result = if stored.is_standard_layout() { d.upload(stored).and_then(|w| product(&w, Op::T)) } else { d.upload(b).and_then(|w| product(&w, Op::N)) };
    result.ok()
}

/// `last · Uᵀ` on the device with `U` (the unembedding, never edited) resident.
pub(crate) fn logits(last: &Array2<f64>, unembedding: &Array2<f64>) -> Option<Array2<f64>> {
    let mut s = DEVICE.get()?.lock().ok()?;
    let u = s.ensure(unembedding.view()).ok()?;
    let d = &s.device;
    let x = d.upload(last.view()).ok()?;
    let mut out = d.zeros(last.nrows(), unembedding.nrows()).ok()?;
    d.gemm(&mut out, 1.0, &x, Op::N, s.get(u).ok()?, Op::T, 0.0, s.arithmetic()).ok()?;
    d.download(&out).ok()
}

/// One run's inputs from `graph::run`: the batch's tokens and sequences, the scored rows, the
/// swapped units' values, whether to capture a [`Reference`], and the host's stand-ins (`embed`'s
/// and each unit's, rows × width, `None` for `M` itself, which reads none).
pub(crate) struct Run<'a> {
    pub tokens: &'a [u32],
    pub spans: &'a [(usize, usize)],
    pub scored: &'a [usize],
    pub swaps: &'a BTreeMap<usize, Array2<f64>>,
    pub capture: bool,
    pub standins: Option<(&'a Array2<f64>, &'a [Array2<f64>])>,
}

/// `circuit` run on the process's device ([`use_device`]), or `None` when there is none or the
/// circuit holds a block the device path does not cover.
pub(crate) fn run(weights: &Weights, circuit: &Circuit, job: &Run) -> Option<Result<Execution, String>> {
    if !circuit.units.iter().all(|u| matches!(u.block, Block::Heads { .. } | Block::Neurons { .. })) {
        return None;
    }
    let mut s = DEVICE.get()?.lock().ok()?;
    Some(run_on(&mut s, weights, circuit, job))
}

/// [`run`] on a given device state (the tests run it on the host backend).
pub(crate) fn run_on(s: &mut DeviceState, weights: &Weights, circuit: &Circuit, job: &Run) -> Result<Execution, String> {
    let e = |e: GpuError| e.to_string();
    let (rows, width) = (job.tokens.len(), weights.width());
    let arithmetic = s.arithmetic();
    // The sequences grouped by length: each group's attention runs as whole sequences at once.
    let mut by_length: BTreeMap<usize, Vec<Segment>> = BTreeMap::new();
    for &(start, length) in job.spans {
        by_length.entry(length).or_default().push(Segment { rows: start..start + length, first: 0, before: Vec::new() });
    }
    let positions: Vec<u32> = job.spans.iter().flat_map(|&(_, length)| 0..length as u32).collect();
    let table = s.ensure(weights.embedding.view()).map_err(e)?;
    let ids = s.device.upload_indices(job.tokens).map_err(e)?;
    let embed = s.device.gather_rows(s.get(table).map_err(e)?, &ids).map_err(e)?;
    let units = circuit.units.len();
    // Stand-ins: the host's, or zeros for `M` (it reads none).
    let (embed_standin, standins): (Tensor, Vec<Tensor>) = match job.standins {
        Some((e0, all)) => (s.device.upload(e0.view()).map_err(e)?, all.iter().map(|a| s.device.upload(a.view())).collect::<Result<_, _>>().map_err(e)?),
        None => (s.device.zeros(rows, width).map_err(e)?, (0..units).map(|_| s.device.zeros(rows, width)).collect::<Result<_, _>>().map_err(e)?),
    };
    let mut order: Vec<usize> = (0..units).collect();
    order.sort_by_key(|&u| circuit.units[u].block.site());
    let mut stream = s.device.copy(&embed).map_err(e)?;
    let mut standin_stream = s.device.copy(&embed_standin).map_err(e)?;
    let mut writes: Vec<Option<Tensor>> = (0..units).map(|_| None).collect();
    let mut captured = job.capture.then(|| Reference {
        embed: Array2::zeros((0, 0)),
        reads: weights.layers.iter().map(|l| vec![Array2::zeros((0, 0)); l.heads.len()]).collect(),
        active: vec![Array2::zeros((0, 0)); weights.layers.len()],
        mlp: vec![Array2::zeros((0, 0)); weights.layers.len()],
        inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
        attention_inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
    });
    if let Some(c) = captured.as_mut() {
        c.embed = s.device.download(&embed).map_err(e)?;
    }
    // A route's input: the actual stream less the cut writers' (actual − stand-in), or the
    // stand-in stream plus the kept writers'.
    let input = |s: &DeviceState, incoming: &Incoming, stream: &Tensor, standin_stream: &Tensor, writes: &[Option<Tensor>]| -> Result<Tensor, GpuError> {
        let (mut x, sign, writers) = match incoming {
            Incoming::AllBut(cut) => (s.device.copy(stream)?, -1.0, cut),
            Incoming::Only(kept) => (s.device.copy(standin_stream)?, 1.0, kept),
        };
        for &w in writers {
            let (actual, standin) = match w {
                Writer::Embed => (Some(&embed), &embed_standin),
                Writer::Unit(u) => (writes[u].as_ref(), &standins[u]),
            };
            if let Some(a) = actual {
                s.device.axpy(&mut x, sign, a)?;
                s.device.axpy(&mut x, -sign, standin)?;
            }
        }
        Ok(x)
    };
    let mut rotations: BTreeMap<(u32, u32, bool), (Tensor, Tensor)> = BTreeMap::new();
    let mut at = 0;
    while at < order.len() {
        let site = circuit.units[order[at]].block.site();
        let end = order[at..].iter().position(|&u| circuit.units[u].block.site() != site).map_or(order.len(), |k| at + k);
        for &u in &order[at..end] {
            let unit = &circuit.units[u];
            if !unit.computes {
                continue;
            }
            if let Some(value) = job.swaps.get(&u) {
                writes[u] = Some(s.device.upload(value.view()).map_err(e)?);
                continue;
            }
            let write = match &unit.block {
                Block::Heads { layer, heads } => {
                    let lw = &weights.layers[*layer];
                    let gain = s.ensure(row(&lw.attention.gain)).map_err(e)?;
                    let mut normed = Vec::with_capacity(3);
                    for slot in 0..3 {
                        let x = input(s, &unit.routes[slot], &stream, &standin_stream, &writes).map_err(e)?;
                        let unit_x = s.device.rms_norm(&x, lw.attention.epsilon).map_err(e)?;
                        let mut out = s.device.zeros(rows, width).map_err(e)?;
                        s.device.scale_columns(&mut out, &unit_x, s.get(gain).map_err(e)?, false).map_err(e)?;
                        normed.push(out);
                    }
                    if let Some(c) = captured.as_mut() {
                        c.attention_inputs[*layer] = s.device.download(&normed[0]).map_err(e)?;
                    }
                    let mut out = s.device.zeros(rows, width).map_err(e)?;
                    for &h in heads {
                        let hw = &lw.heads[h];
                        let project = |s: &mut DeviceState, x: &Tensor, map: &Array2<f64>, norm: Option<&(Array1<f64>, f64)>| -> Result<Tensor, GpuError> {
                            let w = s.ensure(map.view())?;
                            let mut p = s.device.zeros(rows, map.nrows())?;
                            s.device.gemm(&mut p, 1.0, x, Op::N, s.get(w)?, Op::T, 0.0, arithmetic)?;
                            if let Some((g, epsilon)) = norm {
                                let n = s.device.rms_norm(&p, *epsilon)?;
                                let g = s.ensure(row(g))?;
                                s.device.scale_columns(&mut p, &n, s.get(g)?, false)?;
                            }
                            Ok(p)
                        };
                        let mut q = project(s, &normed[0], &hw.query, hw.query_norm.as_ref()).map_err(e)?;
                        let mut k = project(s, &normed[1], &hw.key, hw.key_norm.as_ref()).map_err(e)?;
                        let v = project(s, &normed[2], &hw.value, None).map_err(e)?;
                        if let Some(r) = hw.rotary {
                            let tables = (r.base, r.dims, r.half_split);
                            if !rotations.contains_key(&tables) {
                                rotations.insert(tables, rotation_tables(&s.device, r, &positions).map_err(e)?);
                            }
                            let (cos, sin) = rotations.get(&tables).ok_or("rotation tables")?;
                            q = s.device.rotate(&q, cos, sin, r.half_split, false).map_err(e)?;
                            k = s.device.rotate(&k, cos, sin, r.half_split, false).map_err(e)?;
                        }
                        let mut z = s.device.zeros(rows, v.cols()).map_err(e)?;
                        for group in by_length.values() {
                            let part = forward_segments(&s.device, (&q, &k, &v), group, hw.scale, hw.causal, arithmetic).map_err(e)?;
                            s.device.axpy(&mut z, 1.0, &part).map_err(e)?;
                        }
                        if let Some(c) = captured.as_mut() {
                            c.reads[*layer][h] = s.device.download(&z).map_err(e)?;
                        }
                        let wo = s.ensure(hw.output.view()).map_err(e)?;
                        s.device.gemm(&mut out, 1.0, &z, Op::N, s.get(wo).map_err(e)?, Op::T, 1.0, arithmetic).map_err(e)?;
                    }
                    out
                }
                Block::Neurons { layer, neurons } => {
                    let lw = &weights.layers[*layer];
                    let mlp = lw.mlp.as_ref().ok_or("a neuron block without an MLP")?;
                    let x = input(s, &unit.routes[0], &stream, &standin_stream, &writes).map_err(e)?;
                    let unit_x = s.device.rms_norm(&x, lw.mlp_norm.epsilon).map_err(e)?;
                    let gain = s.ensure(row(&lw.mlp_norm.gain)).map_err(e)?;
                    let mut x_hat = s.device.zeros(rows, width).map_err(e)?;
                    s.device.scale_columns(&mut x_hat, &unit_x, s.get(gain).map_err(e)?, false).map_err(e)?;
                    let n = mlp.gate.nrows();
                    // The whole MLP, less the few neurons outside a large block.
                    let whole = 2 * neurons.len() > n;
                    let (all, active) = neuron_writes(s, mlp, &x_hat, None).map_err(e)?;
                    let write = if whole && neurons.len() == n {
                        all
                    } else if whole {
                        let inside: std::collections::BTreeSet<usize> = neurons.iter().copied().collect();
                        let rest: Vec<usize> = (0..n).filter(|i| !inside.contains(i)).collect();
                        let (outside, _) = neuron_writes(s, mlp, &x_hat, Some(&rest)).map_err(e)?;
                        let mut w = all;
                        s.device.axpy(&mut w, -1.0, &outside).map_err(e)?;
                        w
                    } else {
                        neuron_writes(s, mlp, &x_hat, Some(neurons)).map_err(e)?.0
                    };
                    if let Some(c) = captured.as_mut() {
                        if neurons.len() != n {
                            return Err("a capture needs each MLP whole in one unit".into());
                        }
                        c.mlp[*layer] = s.device.download(&write).map_err(e)?;
                        c.active[*layer] = s.device.download(&active).map_err(e)?;
                        c.inputs[*layer] = s.device.download(&x_hat).map_err(e)?;
                    }
                    write
                }
                _ => return Err("a block the device path does not cover".into()),
            };
            writes[u] = Some(write);
        }
        for &u in &order[at..end] {
            match &writes[u] {
                Some(w) => s.device.axpy(&mut stream, 1.0, w).map_err(e)?,
                None => s.device.axpy(&mut stream, 1.0, &standins[u]).map_err(e)?,
            }
            s.device.axpy(&mut standin_stream, 1.0, &standins[u]).map_err(e)?;
        }
        at = end;
    }
    let last = input(s, &circuit.logits, &stream, &standin_stream, &writes).map_err(e)?;
    let picked = s.device.upload_indices(&job.scored.iter().map(|&r| r as u32).collect::<Vec<_>>()).map_err(e)?;
    let last = s.device.gather_rows(&last, &picked).map_err(e)?;
    let normed = s.device.rms_norm(&last, weights.final_norm.epsilon).map_err(e)?;
    let u = s.ensure(weights.unembedding.view()).map_err(e)?;
    let mut logits = s.device.zeros(job.scored.len(), weights.unembedding.nrows()).map_err(e)?;
    s.device.gemm(&mut logits, 1.0, &normed, Op::N, s.get(u).map_err(e)?, Op::T, 0.0, arithmetic).map_err(e)?;
    let mut log_probabilities = s.device.download(&logits).map_err(e)?;
    for mut row in log_probabilities.outer_iter_mut() {
        let values = gam_math::categorical::log_softmax(row.as_slice().ok_or("a contiguous row")?).map_err(|e| e.to_string())?;
        row.assign(&Array1::from(values));
    }
    let writes = writes.iter().map(|w| w.as_ref().map(|t| s.device.download(t)).transpose()).collect::<Result<Vec<_>, _>>().map_err(e)?;
    Ok(Execution::of(log_probabilities, writes, captured))
}

/// Neurons `picked` (all when `None`) of an MLP on normed inputs `x_hat`: their write (rows ×
/// width) and activations (rows × neurons). The whole MLP reads its resident matrices; a selection
/// is uploaded for this run.
fn neuron_writes(s: &mut DeviceState, mlp: &crate::graph::MlpWeights, x_hat: &Tensor, picked: Option<&[usize]>) -> Result<(Tensor, Tensor), GpuError> {
    let arithmetic = s.arithmetic();
    let count = picked.map_or(mlp.gate.nrows(), <[usize]>::len);
    // (gate, out, bias, up map, up bias), resident keys or this run's uploads.
    let (keys, owned) = match picked {
        None => {
            let up = match &mlp.up {
                Some(u) => Some((s.ensure(u.view())?, s.ensure(row(&mlp.up_bias))?)),
                None => None,
            };
            (Some((s.ensure(mlp.gate.view())?, s.ensure(mlp.out.view())?, s.ensure(row(&mlp.bias))?, up)), None)
        }
        Some(p) => {
            let d = &s.device;
            let up = match &mlp.up {
                Some(u) => Some((d.upload(u.select(ndarray::Axis(0), p).view())?, d.upload(row(&mlp.up_bias.select(ndarray::Axis(0), p)))?)),
                None => None,
            };
            (None, Some((d.upload(mlp.gate.select(ndarray::Axis(0), p).view())?, d.upload(mlp.out.select(ndarray::Axis(1), p).view())?, d.upload(row(&mlp.bias.select(ndarray::Axis(0), p)))?, up)))
        }
    };
    let s = &*s;
    let (gate, out, bias, up) = match (&keys, &owned) {
        (Some((g, o, b, u)), _) => (s.get(*g)?, s.get(*o)?, s.get(*b)?, match u {
            Some((m, c)) => Some((s.get(*m)?, s.get(*c)?)),
            None => None,
        }),
        (_, Some((g, o, b, u))) => (g, o, b, u.as_ref().map(|(m, c)| (m, c))),
        _ => return Err(GpuError::DriverCallFailed { reason: "no MLP matrices".into() }),
    };
    let d = &s.device;
    let rows = x_hat.rows();
    let mut h = d.zeros(rows, count)?;
    d.gemm(&mut h, 1.0, x_hat, Op::N, gate, Op::T, 0.0, arithmetic)?;
    d.add_row(&mut h, 1.0, bias)?;
    let codes = d.upload_indices(&vec![law_of(mlp.law).code(); count])?;
    let mut active = d.law_values(&h, &codes, gelu_tanh_constant())?;
    if let Some((u_map, u_bias)) = up {
        let mut u = d.zeros(rows, count)?;
        d.gemm(&mut u, 1.0, x_hat, Op::N, u_map, Op::T, 0.0, arithmetic)?;
        d.add_row(&mut u, 1.0, u_bias)?;
        let gated = d.copy(&active)?;
        d.hadamard(&mut active, &gated, &u, false)?;
    }
    let mut write = d.zeros(rows, out.rows())?;
    d.gemm(&mut write, 1.0, &active, Op::N, out, Op::T, 0.0, arithmetic)?;
    Ok((write, active))
}

/// The rotation tables (rows × planes, `cos` and `sin`) of `rotary` at each row's position.
fn rotation_tables(d: &Device, rotary: Rotary, positions: &[u32]) -> Result<(Tensor, Tensor), GpuError> {
    let planes = rotary.dims as usize / 2;
    let mut cos = Array2::<f64>::zeros((positions.len(), planes));
    let mut sin = Array2::<f64>::zeros((positions.len(), planes));
    for (r, &p) in positions.iter().enumerate() {
        for i in 0..planes {
            let (c, si) = rotary.turn(i, p);
            cos[[r, i]] = c;
            sin[[r, i]] = si;
        }
    }
    Ok((d.upload(cos.view())?, d.upload(sin.view())?))
}
