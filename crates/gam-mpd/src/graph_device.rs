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
    graph::{After, Block, Circuit, Execution, Incoming, Interventions, OnInput, Reference, Weights, Writer},
    operator_program::Rotary,
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Arithmetic, Device, Op, Storage, Tensor},
};
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

/// The device, its resident copies of host matrices (by address and shape), and the uploaded
/// arrays of the counterfactual runs used last ([`Reference::id`], most recent last).
pub(crate) struct DeviceState {
    device: Device,
    resident: HashMap<(usize, usize, usize), Tensor>,
    /// The resident copies' keys, oldest first (past [`RESIDENT_BYTES`] the oldest go).
    uploaded: Vec<(usize, usize, usize)>,
    references: Vec<(u64, BTreeMap<(Field, usize, usize), Tensor>)>,
}

/// An array of a [`Reference`]: the embeddings, a head's read (layer, head), a layer's MLP
/// activations or MLP write (layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Field {
    Embed,
    Read,
    Active,
    Mlp,
}

/// The bytes of resident copies of host matrices kept (Qwen3-0.6B's weights in float32 are about
/// 3.2 GB; copies of matrices no run reads any more go first).
const RESIDENT_BYTES: usize = 6 << 30;

/// The bytes of counterfactual runs' arrays kept uploaded (the least recently used run's go first).
const KEPT_REFERENCE_BYTES: usize = 2 << 30;

static DEVICE: OnceLock<Mutex<DeviceState>> = OnceLock::new();

/// The one thread that touches the device: a single-thread pool, so the device serves one call at
/// a time and its own nested parallel work (gam_gpu converts values on the rayon pool) runs inline
/// there.
static WORKER: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// `f` on the device's thread with the device state, or `None` when no device is set.
///
/// No caller may run other rayon work while it waits. A rayon worker that waited in `install`
/// stole other jobs meanwhile; a stolen run that needed the counterfactual run this very worker was
/// computing (`Checker::referenced`'s `OnceLock`) then waited on itself forever (the server hung at
/// 0% CPU, seen by g-mech on vpd4l). A worker therefore hands the call to a plain scoped thread and
/// joins it, which blocks without stealing; a thread outside the pool waits in `install` directly,
/// which does not steal either.
fn on_device<T: Send>(f: impl FnOnce(&mut DeviceState) -> T + Send) -> Option<T> {
    let state = DEVICE.get()?;
    let pool = WORKER.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(1).thread_name(|_| "graph-device".into()).build().ok()).as_ref()?;
    let call = || pool.install(|| state.lock().ok().map(|mut s| f(&mut s)));
    if rayon::current_thread_index().is_none() {
        return call();
    }
    std::thread::scope(|scope| scope.spawn(call).join().ok().flatten())
}

impl DeviceState {
    pub(crate) fn new(device: Device) -> Self {
        Self { device, resident: HashMap::new(), uploaded: Vec::new(), references: Vec::new() }
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
            self.uploaded.retain(|k| self.resident.contains_key(k) && *k != key);
            self.uploaded.push(key);
            let mut bytes: usize = self.resident.values().map(|t| 4 * t.rows() * t.cols()).sum();
            while bytes > RESIDENT_BYTES && self.uploaded.len() > 1 {
                let old = self.uploaded.remove(0);
                if let Some(t) = self.resident.remove(&old) {
                    bytes -= 4 * t.rows() * t.cols();
                }
            }
        }
        Ok(key)
    }

    /// Uploads array `field` (at `layer`, `head`) of counterfactual run `r` unless it is kept.
    fn ensure_reference(&mut self, r: &Reference, (field, layer, head): (Field, usize, usize)) -> Result<(), String> {
        // The run moves to the most recent place.
        let kept = match self.references.iter().position(|(id, _)| *id == r.id) {
            Some(at) => self.references.remove(at),
            None => (r.id, BTreeMap::new()),
        };
        self.references.push(kept);
        let at = self.references.len() - 1;
        if !self.references[at].1.contains_key(&(field, layer, head)) {
            let host = match field {
                Field::Embed => Some(&r.embed),
                Field::Read => r.reads.get(layer).and_then(|l| l.get(head)),
                Field::Active => r.active.get(layer),
                Field::Mlp => r.mlp.get(layer),
            };
            let host = host.ok_or("an array the counterfactual run did not record")?;
            let t = self.device.upload(host.view()).map_err(|e| e.to_string())?;
            self.references[at].1.insert((field, layer, head), t);
            let bytes = |m: &BTreeMap<(Field, usize, usize), Tensor>| m.values().map(|t| 4 * t.rows() * t.cols()).sum::<usize>();
            while self.references.len() > 1 && self.references.iter().map(|(_, m)| bytes(m)).sum::<usize>() > KEPT_REFERENCE_BYTES {
                self.references.remove(0);
            }
        }
        Ok(())
    }

    /// Array `key` of counterfactual run `r`, uploaded by [`DeviceState::ensure_reference`].
    fn reference(&self, r: &Reference, key: (Field, usize, usize)) -> Result<&Tensor, String> {
        self.references.iter().find(|(id, _)| *id == r.id).and_then(|(_, m)| m.get(&key)).ok_or_else(|| "a counterfactual array went missing".to_string())
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
    let dim = m.dim();
    on_device(|s| s.resident.retain(|k, _| (k.1, k.2) != dim));
}

/// `a · b` on the device, or `None` (no device, a product below `2^22` multiply-adds, or a device
/// error, after which the host multiplies).
pub(crate) fn dot(a: &Array2<f64>, b: ArrayView2<f64>) -> Option<Array2<f64>> {
    if DEVICE.get().is_none() || a.nrows() * a.ncols() * b.ncols() < 1 << 22 {
        return None;
    }
    on_device(|s| dot_on(s, a, b)).flatten()
}

fn dot_on(s: &mut DeviceState, a: &Array2<f64>, b: ArrayView2<f64>) -> Option<Array2<f64>> {
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
    on_device(|s| logits_on(s, last, unembedding)).flatten()
}

fn logits_on(s: &mut DeviceState, last: &Array2<f64>, unembedding: &Array2<f64>) -> Option<Array2<f64>> {
    let u = s.ensure(unembedding.view()).ok()?;
    let d = &s.device;
    let x = d.upload(last.view()).ok()?;
    let mut out = d.zeros(last.nrows(), unembedding.nrows()).ok()?;
    d.gemm(&mut out, 1.0, &x, Op::N, s.get(u).ok()?, Op::T, 0.0, s.arithmetic()).ok()?;
    d.download(&out).ok()
}

/// One run's inputs from `graph::run`: the batch's tokens and sequences, the scored rows, the
/// swapped units' values, whether to capture a [`Reference`], and the counterfactual run the
/// stand-ins come from (`None` for `M` itself, every unit computing, which reads none).
pub(crate) struct Run<'a> {
    pub tokens: &'a [u32],
    pub spans: &'a [(usize, usize)],
    pub scored: &'a [usize],
    pub swaps: &'a BTreeMap<usize, Array2<f64>>,
    pub capture: bool,
    pub reference: Option<&'a Reference>,
    /// Row interventions (site operations), applied as `graph::run` applies them.
    pub ops: &'a Interventions,
}

/// The stand-ins (`embed`'s and each unit's write in the counterfactual run `r`, rows × width),
/// assembled on the device with the current weights as `Reference::write` does on the host; `r`'s
/// arrays stay uploaded for the next runs that read it.
fn standins(s: &mut DeviceState, weights: &Weights, circuit: &Circuit, r: &Reference) -> Result<(Tensor, Vec<Tensor>), String> {
    let e = |e: GpuError| e.to_string();
    let (rows, width, arithmetic) = (r.embed.nrows(), weights.width(), s.arithmetic());
    s.ensure_reference(r, (Field::Embed, 0, 0))?;
    let embed = s.device.copy(s.reference(r, (Field::Embed, 0, 0))?).map_err(e)?;
    let mut out = Vec::with_capacity(circuit.units.len());
    for unit in &circuit.units {
        let mut w = s.device.zeros(rows, width).map_err(e)?;
        match &unit.block {
            Block::Heads { layer, heads } => {
                for &h in heads {
                    s.ensure_reference(r, (Field::Read, *layer, h))?;
                    let wo = s.ensure(weights.layers[*layer].heads[h].output.view()).map_err(e)?;
                    s.device.gemm(&mut w, 1.0, s.reference(r, (Field::Read, *layer, h))?, Op::N, s.get(wo).map_err(e)?, Op::T, 1.0, arithmetic).map_err(e)?;
                }
            }
            Block::Neurons { layer, neurons } => {
                let mlp = weights.layers[*layer].mlp.as_ref().ok_or("a neuron block without an MLP")?;
                let n = mlp.gate.nrows();
                s.ensure_reference(r, (Field::Active, *layer, 0))?;
                s.ensure_reference(r, (Field::Mlp, *layer, 0))?;
                let out_key = s.ensure(mlp.out.view()).map_err(e)?;
                // A large block: the MLP's write less the few neurons outside it.
                let (picked, sign) = if 2 * neurons.len() > n {
                    s.device.axpy(&mut w, 1.0, s.reference(r, (Field::Mlp, *layer, 0))?).map_err(e)?;
                    let inside: std::collections::BTreeSet<usize> = neurons.iter().copied().collect();
                    ((0..n).filter(|i| !inside.contains(i)).map(|i| i as u32).collect::<Vec<_>>(), -1.0)
                } else {
                    (neurons.iter().map(|&i| i as u32).collect(), 1.0)
                };
                if !picked.is_empty() {
                    let ids = s.device.upload_indices(&picked).map_err(e)?;
                    let a = s.device.gather_columns(s.reference(r, (Field::Active, *layer, 0))?, &ids).map_err(e)?;
                    let o = s.device.gather_columns(s.get(out_key).map_err(e)?, &ids).map_err(e)?;
                    s.device.gemm(&mut w, sign, &a, Op::N, &o, Op::T, 1.0, arithmetic).map_err(e)?;
                }
            }
            _ => return Err("a block the device path does not cover".into()),
        }
        out.push(w);
    }
    Ok((embed, out))
}

/// `circuit` run on the process's device ([`use_device`]), or `None` when there is none or the
/// circuit holds a block the device path does not cover.
pub(crate) fn run(weights: &Weights, circuit: &Circuit, job: &Run) -> Option<Result<Execution, String>> {
    // Native blocks only; operations on the heads of a VPD-view attention run on the host.
    if !circuit.units.iter().all(|u| matches!(u.block, Block::Heads { .. } | Block::Neurons { .. })) || !job.ops.head_reads.is_empty() || !job.ops.record_reads.is_empty() {
        return None;
    }
    on_device(|s| run_on(s, weights, circuit, job))
}

/// A run's state on the device: `embed`'s actual and stand-in writes, every unit's stand-in and
/// actual write (`None` while it writes its stand-in), and the actual and stand-in streams entering
/// the current site.
struct Streams {
    embed: Tensor,
    embed_standin: Tensor,
    standins: Vec<Tensor>,
    writes: Vec<Option<Tensor>>,
    stream: Tensor,
    standin_stream: Tensor,
}

impl Streams {
    /// Writer `w`'s actual write and stand-in write (rows × width).
    fn of(&self, w: Writer) -> (Option<&Tensor>, &Tensor) {
        match w {
            Writer::Embed => (Some(&self.embed), &self.embed_standin),
            Writer::Unit(u) => (self.writes[u].as_ref(), &self.standins[u]),
        }
    }

    fn of_mut(&mut self, w: Writer) -> (Option<&mut Tensor>, &mut Tensor) {
        match w {
            Writer::Embed => (Some(&mut self.embed), &mut self.embed_standin),
            Writer::Unit(u) => (self.writes[u].as_mut(), &mut self.standins[u]),
        }
    }

    /// A route's input: the actual stream less the cut writers' (actual − stand-in), or the
    /// stand-in stream plus the kept writers'.
    fn input(&self, d: &Device, incoming: &Incoming) -> Result<Tensor, GpuError> {
        let (mut x, sign, writers) = match incoming {
            Incoming::AllBut(cut) => (d.copy(&self.stream)?, -1.0, cut),
            Incoming::Only(kept) => (d.copy(&self.standin_stream)?, 1.0, kept),
        };
        for &w in writers {
            let (actual, standin) = self.of(w);
            if let Some(a) = actual {
                d.axpy(&mut x, sign, a)?;
                d.axpy(&mut x, -sign, standin)?;
            }
        }
        Ok(x)
    }
}

/// `t[rows] ← f · t[rows]`.
fn scale_rows(d: &Device, t: &mut Tensor, rows: &gam_gpu::tensor::Indices, f: f64) -> Result<Tensor, GpuError> {
    let old = d.gather_rows(t, rows)?;
    d.scatter_rows(t, rows, &d.scaled(f, &old)?, false)?;
    Ok(old)
}

/// `v` as `count` rows.
fn repeated(d: &Device, v: &Array1<f64>, count: usize) -> Result<Tensor, GpuError> {
    d.upload(v.view().insert_axis(ndarray::Axis(0)).broadcast((count, v.len())).ok_or_else(|| GpuError::DriverCallFailed { reason: "a pushed vector".into() })?)
}

/// The site operations of `ops` after `point` (a site, `None` before every site), as
/// `Interventions::after` applies them on the host: a scale multiplies writers' actual and
/// stand-in writes at the rows (and the streams by the change), a swap puts the donor's writes in
/// place of the actual ones, a push adds a vector to both streams.
fn after(d: &Device, ops: &Interventions, point: Option<usize>, st: &mut Streams) -> Result<(), GpuError> {
    for (_, op) in ops.after.iter().filter(|(p, _)| *p == point) {
        match op {
            After::Scale(writers, rows, f) => {
                let idx = d.upload_indices(&rows.iter().map(|&r| r as u32).collect::<Vec<_>>())?;
                for &w in writers {
                    let (actual, standin) = st.of_mut(w);
                    let old_standin = scale_rows(d, standin, &idx, *f)?;
                    let contribution = match actual {
                        Some(a) => scale_rows(d, a, &idx, *f)?,
                        None => d.copy(&old_standin)?,
                    };
                    d.scatter_rows(&mut st.stream, &idx, &d.scaled(f - 1.0, &contribution)?, true)?;
                    d.scatter_rows(&mut st.standin_stream, &idx, &d.scaled(f - 1.0, &old_standin)?, true)?;
                }
            }
            After::Swap(writers, rows) => {
                let donor = ops.donor.as_ref().ok_or_else(|| GpuError::DriverCallFailed { reason: "a swap without a donor run".into() })?;
                let idx = d.upload_indices(&rows.iter().map(|&r| r as u32).collect::<Vec<_>>())?;
                for &w in writers {
                    let value = match w {
                        Writer::Embed => Some(&donor.embed),
                        Writer::Unit(u) => donor.writes.get(u).and_then(Option::as_ref),
                    };
                    let (Some(value), (Some(actual), _)) = (value, st.of_mut(w)) else { continue };
                    let new = d.gather_rows(&d.upload(value.view())?, &idx)?;
                    let old = d.gather_rows(actual, &idx)?;
                    d.scatter_rows(actual, &idx, &new, false)?;
                    let mut change = new;
                    d.axpy(&mut change, -1.0, &old)?;
                    d.scatter_rows(&mut st.stream, &idx, &change, true)?;
                }
            }
            After::Push(rows, v) => {
                let idx = d.upload_indices(&rows.iter().map(|&r| r as u32).collect::<Vec<_>>())?;
                let values = repeated(d, v, rows.len())?;
                d.scatter_rows(&mut st.stream, &idx, &values, true)?;
                d.scatter_rows(&mut st.standin_stream, &idx, &values, true)?;
            }
        }
    }
    Ok(())
}

/// Unit `unit`'s route inputs under the cuts into its site (`Interventions::cut_inputs`): a writer
/// the route reads actually is read on the donor at the cut rows.
fn cut_inputs(d: &Device, ops: &Interventions, site: usize, unit: &crate::graph::Unit, inputs: &mut [(usize, Tensor)], st: &Streams) -> Result<(), GpuError> {
    for (_, writers, rows) in ops.cuts.iter().filter(|(to, _, _)| *to == site) {
        let donor = ops.donor.as_ref().ok_or_else(|| GpuError::DriverCallFailed { reason: "a cut without a donor run".into() })?;
        let idx = d.upload_indices(&rows.iter().map(|&r| r as u32).collect::<Vec<_>>())?;
        for &w in writers {
            let value = match w {
                Writer::Embed => Some(&donor.embed),
                Writer::Unit(u) => donor.writes.get(u).and_then(Option::as_ref),
            };
            let (Some(value), (Some(actual), _)) = (value, st.of(w)) else { continue };
            let mut change = d.gather_rows(&d.upload(value.view())?, &idx)?;
            d.axpy(&mut change, -1.0, &d.gather_rows(actual, &idx)?)?;
            for (slot, x) in inputs.iter_mut() {
                let reads = match &unit.routes[*slot] {
                    Incoming::AllBut(cut) => !cut.contains(&w),
                    Incoming::Only(kept) => kept.contains(&w),
                };
                if reads {
                    d.scatter_rows(x, &idx, &change, true)?;
                }
            }
        }
    }
    Ok(())
}

/// Unit `u`'s normed input of route slot `slot` under the operations on its site's input
/// (`Interventions::normed`), kept in `kept` when the site is recorded.
fn normed(d: &Device, ops: &Interventions, (site, u, slot): (usize, usize, usize), x: &mut Tensor, kept: &mut BTreeMap<(usize, usize), Array2<f64>>) -> Result<(), GpuError> {
    for (_, op) in ops.inputs.iter().filter(|(s, _)| *s == site) {
        let rows = match op {
            OnInput::Scale(rows, _) | OnInput::Push(rows, _) | OnInput::Swap(rows) => rows,
        };
        let idx = d.upload_indices(&rows.iter().map(|&r| r as u32).collect::<Vec<_>>())?;
        match op {
            OnInput::Scale(_, f) => {
                scale_rows(d, x, &idx, *f)?;
            }
            OnInput::Push(_, v) => d.scatter_rows(x, &idx, &repeated(d, v, rows.len())?, true)?,
            OnInput::Swap(_) => {
                if let Some(value) = ops.donor.as_ref().and_then(|dn| dn.normed.get(&(u, slot))) {
                    let new = d.gather_rows(&d.upload(value.view())?, &idx)?;
                    d.scatter_rows(x, &idx, &new, false)?;
                }
            }
        }
    }
    if ops.record.contains(&site) {
        kept.insert((u, slot), d.download(x)?);
    }
    Ok(())
}

/// [`run`] on a given device state (the tests run it on the host backend).
pub(crate) fn run_on(s: &mut DeviceState, weights: &Weights, circuit: &Circuit, job: &Run) -> Result<Execution, String> {
    let e = |e: GpuError| e.to_string();
    let (rows, width) = (job.tokens.len(), weights.width());
    let arithmetic = s.arithmetic();
    // Attention runs on every sequence padded at its end to the longest one's length, all at once
    // (whole sequences of one length); causal attention keeps a real position from reading a pad.
    // `padded` is each row's place there (`None` when every sequence has one length).
    let longest = job.spans.iter().map(|s| s.1).max().unwrap_or(0);
    let sequences = job.spans.len();
    let padded = if job.spans.iter().all(|s| s.1 == longest) {
        None
    } else {
        let places: Vec<u32> = job.spans.iter().enumerate().flat_map(|(n, &(_, length))| (0..length).map(move |p| (n * longest + p) as u32)).collect();
        Some(s.device.upload_indices(&places).map_err(e)?)
    };
    let segments: Vec<Segment> = (0..sequences).map(|n| Segment { rows: n * longest..(n + 1) * longest, first: 0, before: Vec::new() }).collect();
    let positions: Vec<u32> = job.spans.iter().flat_map(|&(_, length)| 0..length as u32).collect();
    let table = s.ensure(weights.embedding.view()).map_err(e)?;
    let ids = s.device.upload_indices(job.tokens).map_err(e)?;
    let embed = s.device.gather_rows(s.get(table).map_err(e)?, &ids).map_err(e)?;
    let units = circuit.units.len();
    // Stand-ins: from the counterfactual run, or zeros for `M` (it reads none).
    let (embed_standin, standins): (Tensor, Vec<Tensor>) = match job.reference {
        Some(r) if r.embed.nrows() != rows => return Err(format!("a counterfactual run of {} tokens for a batch of {rows}", r.embed.nrows())),
        Some(r) => standins(s, weights, circuit, r)?,
        None if circuit.units.iter().all(|u| u.computes) => (s.device.zeros(rows, width).map_err(e)?, (0..units).map(|_| s.device.zeros(rows, width)).collect::<Result<_, _>>().map_err(e)?),
        None => return Err("a program's undeclared pieces take their values from the counterfactual run, which this batch lacks".into()),
    };
    let mut captured = job.capture.then(|| Reference {
        id: crate::graph::next_reference_id(),
        embed: Array2::zeros((0, 0)),
        reads: weights.layers.iter().map(|l| vec![Array2::zeros((0, 0)); l.heads.len()]).collect(),
        active: vec![Array2::zeros((0, 0)); weights.layers.len()],
        mlp: vec![Array2::zeros((0, 0)); weights.layers.len()],
        inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
        attention_inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
    });
    let mut order: Vec<usize> = (0..units).collect();
    order.sort_by_key(|&u| circuit.units[u].block.site());
    let mut st = Streams {
        stream: s.device.copy(&embed).map_err(e)?,
        standin_stream: s.device.copy(&embed_standin).map_err(e)?,
        embed,
        embed_standin,
        standins,
        writes: (0..units).map(|_| None).collect(),
    };
    let ops = job.ops;
    let mut kept = BTreeMap::new();
    if let Some(c) = captured.as_mut() {
        c.embed = s.device.download(&st.embed).map_err(e)?;
    }
    after(&s.device, ops, None, &mut st).map_err(e)?;
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
                st.writes[u] = Some(s.device.upload(value.view()).map_err(e)?);
                continue;
            }
            let slots: Vec<usize> = unit.block.routes().iter().map(|r| r.slot()).collect();
            let mut inputs: Vec<(usize, Tensor)> = slots.iter().map(|&slot| st.input(&s.device, &unit.routes[slot]).map(|x| (slot, x))).collect::<Result<_, _>>().map_err(e)?;
            cut_inputs(&s.device, ops, site, unit, &mut inputs, &st).map_err(e)?;
            let write = match &unit.block {
                Block::Heads { layer, heads } => {
                    let lw = &weights.layers[*layer];
                    let gain = s.ensure(row(&lw.attention.gain)).map_err(e)?;
                    let mut normed_inputs = Vec::with_capacity(3);
                    for (slot, x) in &inputs {
                        let unit_x = s.device.rms_norm(x, lw.attention.epsilon).map_err(e)?;
                        let mut out = s.device.zeros(rows, width).map_err(e)?;
                        s.device.scale_columns(&mut out, &unit_x, s.get(gain).map_err(e)?, false).map_err(e)?;
                        normed(&s.device, ops, (site, u, *slot), &mut out, &mut kept).map_err(e)?;
                        normed_inputs.push(out);
                    }
                    if let Some(c) = captured.as_mut() {
                        // Only VPD attention subcomponents read it.
                        if !weights.vpd_attention.is_empty() {
                            c.attention_inputs[*layer] = s.device.download(&normed_inputs[0]).map_err(e)?;
                        }
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
                        let mut q = project(s, &normed_inputs[0], &hw.query, hw.query_norm.as_ref()).map_err(e)?;
                        let mut k = project(s, &normed_inputs[1], &hw.key, hw.key_norm.as_ref()).map_err(e)?;
                        let v = project(s, &normed_inputs[2], &hw.value, None).map_err(e)?;
                        if let Some(r) = hw.rotary {
                            let tables = (r.base, r.dims, r.half_split);
                            if !rotations.contains_key(&tables) {
                                rotations.insert(tables, rotation_tables(&s.device, r, &positions).map_err(e)?);
                            }
                            let (cos, sin) = rotations.get(&tables).ok_or("rotation tables")?;
                            q = s.device.rotate(&q, cos, sin, r.half_split, false).map_err(e)?;
                            k = s.device.rotate(&k, cos, sin, r.half_split, false).map_err(e)?;
                        }
                        let z = match &padded {
                            None => forward_segments(&s.device, (&q, &k, &v), &segments, hw.scale, hw.causal, arithmetic).map_err(e)?,
                            Some(places) => {
                                let pad = |x: &Tensor| -> Result<Tensor, GpuError> {
                                    let mut out = s.device.zeros(sequences * longest, x.cols())?;
                                    s.device.scatter_rows(&mut out, places, x, false)?;
                                    Ok(out)
                                };
                                let all = forward_segments(&s.device, (&pad(&q).map_err(e)?, &pad(&k).map_err(e)?, &pad(&v).map_err(e)?), &segments, hw.scale, hw.causal, arithmetic).map_err(e)?;
                                s.device.gather_rows(&all, places).map_err(e)?
                            }
                        };
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
                    let x = &inputs[0].1;
                    let unit_x = s.device.rms_norm(x, lw.mlp_norm.epsilon).map_err(e)?;
                    let gain = s.ensure(row(&lw.mlp_norm.gain)).map_err(e)?;
                    let mut x_hat = s.device.zeros(rows, width).map_err(e)?;
                    s.device.scale_columns(&mut x_hat, &unit_x, s.get(gain).map_err(e)?, false).map_err(e)?;
                    normed(&s.device, ops, (site, u, 0), &mut x_hat, &mut kept).map_err(e)?;
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
                        // Only transcoder features and VPD MLP subcomponents read it.
                        if !weights.transcoders.is_empty() || !weights.vpd.is_empty() {
                            c.inputs[*layer] = s.device.download(&x_hat).map_err(e)?;
                        }
                    }
                    write
                }
                _ => return Err("a block the device path does not cover".into()),
            };
            st.writes[u] = Some(write);
        }
        for &u in &order[at..end] {
            match &st.writes[u] {
                Some(w) => s.device.axpy(&mut st.stream, 1.0, w).map_err(e)?,
                None => s.device.axpy(&mut st.stream, 1.0, &st.standins[u]).map_err(e)?,
            }
            s.device.axpy(&mut st.standin_stream, 1.0, &st.standins[u]).map_err(e)?;
        }
        after(&s.device, ops, Some(site), &mut st).map_err(e)?;
        at = end;
    }
    let last = st.input(&s.device, &circuit.logits).map_err(e)?;
    let picked = s.device.upload_indices(&job.scored.iter().map(|&r| r as u32).collect::<Vec<_>>()).map_err(e)?;
    let last = s.device.gather_rows(&last, &picked).map_err(e)?;
    let normed_last = s.device.rms_norm(&last, weights.final_norm.epsilon).map_err(e)?;
    let u = s.ensure(weights.unembedding.view()).map_err(e)?;
    let mut logits = s.device.zeros(job.scored.len(), weights.unembedding.nrows()).map_err(e)?;
    s.device.gemm(&mut logits, 1.0, &normed_last, Op::N, s.get(u).map_err(e)?, Op::T, 0.0, arithmetic).map_err(e)?;
    let mut log_probabilities = s.device.download(&logits).map_err(e)?;
    for mut row in log_probabilities.outer_iter_mut() {
        let values = gam_math::categorical::log_softmax(row.as_slice().ok_or("a contiguous row")?).map_err(|e| e.to_string())?;
        row.assign(&Array1::from(values));
    }
    // The units' writes come back only from a run that scores no rows: a donor run, whose writes
    // other runs read (swaps, site operations, `Checker::measure_typical`). On Qwen3-0.6B every
    // run's writes are about a gigabyte of float64.
    let writes = if job.scored.is_empty() && !job.capture {
        st.writes.iter().map(|w| w.as_ref().map(|t| s.device.download(t)).transpose()).collect::<Result<Vec<_>, _>>().map_err(e)?
    } else {
        vec![None; units]
    };
    Ok(Execution::of(log_probabilities, writes, captured, kept))
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
