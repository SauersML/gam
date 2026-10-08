//! The graph checker's runs on a device (#2951): the same circuit semantics as `graph::run` (each
//! unit's route inputs from the actual and stand-in streams, its block computed on them, the
//! logits at the scored rows), with every array resident on Metal or CUDA through `gam_gpu`'s
//! tensor operations and the attention of `device_attention`. `M`'s weights stay resident between
//! runs; a weight edit marks the matrices it touches, which the next run uploads again.
//!
//! Covered: native heads (with head norms, rotary, grouped keys and values), MLP
//! neurons (plain or gated), transcoder features, VPD's views of MLPs and attentions
//! (subcomponents and remainders, with operations on their heads' reads), swaps, counterfactual
//! and average stand-ins, site operations, `Reference` captures. Attention blocks return `None`
//! and run on the host.
use crate::{
    device_attention::{Segment, forward_segments},
    device_program::{gelu_tanh_constant, law_of},
    graph::{After, Block, Circuit, Execution, Incoming, Interventions, OnInput, Reference, Stored, Weights, Writer},
    operator_program::Rotary,
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Arithmetic, Device, Op, Storage, Tensor},
};
use ndarray::{Array1, Array2, ArrayView2};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

/// The device, its resident copies of host matrices (by their `Weights`' generation, address and
/// shape), and the uploaded arrays of the counterfactual runs used last ([`Reference::id`], most
/// recent last).
pub(crate) struct DeviceState {
    device: Device,
    /// The generation of the `Weights` the current call reads ([`Weights::generation`]): copies of
    /// two models never share a key, even where one's freed matrix sits at the other's address.
    generation: u64,
    resident: HashMap<Key, Tensor>,
    /// The resident copies' keys, oldest first (past [`RESIDENT_BYTES`] the oldest go).
    uploaded: Vec<Key>,
    /// Heads' maps stacked into one matrix ([`DeviceState::stacked`]), by the maps' keys and
    /// whether they stack as rows; dropped whenever a weight edit drops resident copies.
    stacks: HashMap<(Vec<Key>, bool), Tensor>,
    references: Vec<(u64, BTreeMap<(Field, usize, usize), Tensor>)>,
    /// Rotary tables by rotary and positions ([`DeviceState::rotations`]).
    rotations: HashMap<RotationKey, (Tensor, Tensor)>,
    /// Products of counterfactual runs' arrays with weight matrices ([`DeviceState::reference_product`]),
    /// by run, array, layer and matrix; dropped whenever a weight changes.
    products: BTreeMap<(u64, Field, usize, usize), Tensor>,
    /// A VPD view's summed matrix `Σ_i U_iᵀ V_iᵀ` by its factors' keys ([`DeviceState::summed`]);
    /// dropped whenever a weight changes.
    summed: HashMap<(Key, Key), Tensor>,
}

/// The bytes of reference products kept ([`DeviceState::reference_product`]): past them every
/// product goes and is made again on use.
const KEPT_PRODUCT_BYTES: usize = 1 << 28;

/// A rotary's base, dimensions and pairing, and the positions of its table's rows.
type RotationKey = (u32, u32, bool, Vec<u32>);

/// The most rotary tables kept: a behavior's runs turn a few position layouts (its prompts and
/// counterfactuals, by head count), each a few megabytes.
const KEPT_ROTATIONS: usize = 32;

/// An array of a [`Reference`]: the embeddings, a head's read (layer, head), a layer's MLP
/// activations, MLP write, MLP normed input, attention normed input or its heads' reads side by side
/// (layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Field {
    Embed,
    Read,
    Active,
    Mlp,
    Input,
    AttentionInput,
    Reads,
}

/// The bytes of resident copies of host matrices kept (Qwen3-0.6B's weights in float32 are about
/// 3.2 GB; copies of matrices no run reads any more go first).
const RESIDENT_BYTES: usize = 6 << 30;

/// The bytes of counterfactual runs' arrays kept uploaded (the least recently used run's go first).
const KEPT_REFERENCE_BYTES: usize = 1 << 30;

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
        Self { device, generation: 0, resident: HashMap::new(), uploaded: Vec::new(), stacks: HashMap::new(), references: Vec::new(), rotations: HashMap::new(), products: BTreeMap::new(), summed: HashMap::new() }
    }

    fn arithmetic(&self) -> Arithmetic {
        match self.device.storage() {
            Storage::F64 => Arithmetic::F64,
            Storage::F32 | Storage::Bf16 => Arithmetic::F32,
        }
    }

    /// The key of `rotary`'s tables at `positions` ([`rotation_tables`]), made and uploaded on first
    /// use and kept (read them from `self.rotations`): every run of a behavior turns the same
    /// positions, and making the tables on the host (a power, a cosine and a sine per plane and row)
    /// took most of a VPD-view program's run.
    fn rotations(&mut self, rotary: Rotary, positions: &[u32]) -> Result<RotationKey, GpuError> {
        let key = (rotary.base, rotary.dims, rotary.half_split, positions.to_vec());
        if !self.rotations.contains_key(&key) {
            if self.rotations.len() >= KEPT_ROTATIONS {
                self.rotations.clear();
            }
            let tables = rotation_tables(&self.device, rotary, positions)?;
            self.rotations.insert(key.clone(), tables);
        }
        Ok(key)
    }

    /// The key of host matrix `m`'s resident copy, uploaded when absent or edited since.
    fn ensure<A: Element>(&mut self, m: ArrayView2<A>) -> Result<Key, GpuError> {
        let key = (self.generation, m.as_ptr() as usize, m.nrows(), m.ncols());
        if !self.resident.contains_key(&key) {
            let t = A::upload(&self.device, m)?;
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
                Field::Embed => Some(Cow::Borrowed(&r.embed)),
                Field::Read => r.reads.get(layer).and_then(|l| l.get(head)).map(Cow::Borrowed),
                Field::Active => r.active.get(layer).map(Cow::Borrowed),
                Field::Mlp => r.mlp.get(layer).map(Cow::Borrowed),
                Field::Input => r.inputs.get(layer).map(Cow::Borrowed),
                Field::AttentionInput => r.attention_inputs.get(layer).map(Cow::Borrowed),
                Field::Reads => match r.reads.get(layer) {
                    Some(reads) => Some(Cow::Owned(ndarray::concatenate(ndarray::Axis(1), &reads.iter().map(|x| x.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?)),
                    None => None,
                },
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

    /// A copy of `x wᵀ` for counterfactual run `r`'s array `(field, layer)` and weight matrix `w`
    /// (`matrix` names it within the layer), made on first use and kept until a weight changes:
    /// every program's run on that counterfactual run (each program of a batch, each run of a score)
    /// reads the same product, such as an MLP's counterfactual pre-activation or the full write a
    /// remainder's stand-in subtracts its named parts from.
    fn reference_product(&mut self, r: &Reference, (field, layer): (Field, usize), matrix: usize, w: Matrix) -> Result<Tensor, String> {
        let e = |e: GpuError| e.to_string();
        let key = (r.id, field, layer, matrix);
        if !self.products.contains_key(&key) {
            self.ensure_reference(r, (field, layer, 0))?;
            let wk = match w {
                Matrix::Host(m) => Some(self.ensure(m.view()).map_err(e)?),
                Matrix::Device(_) => None,
            };
            let wt = match (w, wk) {
                (Matrix::Device(t), _) => t,
                (Matrix::Host(_), Some(k)) => self.get(k).map_err(e)?,
                (Matrix::Host(_), None) => return Err("a weight matrix went missing".into()),
            };
            let x = self.reference(r, (field, layer, 0))?;
            let mut out = self.device.zeros(x.rows(), wt.rows()).map_err(e)?;
            self.device.gemm(&mut out, 1.0, x, Op::N, wt, Op::T, 0.0, self.arithmetic()).map_err(e)?;
            let bytes = |m: &BTreeMap<(u64, Field, usize, usize), Tensor>| m.values().map(|t| 4 * t.rows() * t.cols()).sum::<usize>();
            if bytes(&self.products) + 4 * out.rows() * out.cols() > KEPT_PRODUCT_BYTES {
                self.products.clear();
            }
            self.products.insert(key, out);
        }
        let kept = self.products.get(&key).ok_or("a reference product went missing")?;
        self.device.copy(kept).map_err(e)
    }

    /// The summed matrix `Σ_i U_iᵀ V_iᵀ` (out × in: the matrix less its remainder) of the resident
    /// factors `uk` (subcomponents × out) and `vk` (in × subcomponents), made on first use and kept
    /// until a weight changes.
    fn summed(&mut self, uk: Key, vk: Key) -> Result<(), GpuError> {
        if !self.summed.contains_key(&(uk, vk)) {
            let (uf, vf) = (self.get(uk)?, self.get(vk)?);
            let mut sum = self.device.zeros(uf.cols(), vf.rows())?;
            self.device.gemm(&mut sum, 1.0, uf, Op::T, vf, Op::T, 0.0, self.arithmetic())?;
            self.summed.insert((uk, vk), sum);
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

    /// Heads' maps `maps` (one shape) stacked as rows, or side by side as columns (output maps):
    /// joined on the host and uploaded once (no per-head copies on the device), kept by the maps'
    /// addresses until a weight edit drops every stack.
    fn stacked(&mut self, maps: &[&Stored], along_rows: bool) -> Result<Tensor, GpuError> {
        let keys: Vec<Key> = maps.iter().map(|m| (self.generation, m.as_ptr() as usize, m.nrows(), m.ncols())).collect();
        if let Some(t) = self.stacks.get(&(keys.clone(), along_rows)) {
            return self.device.copy(t);
        }
        let views: Vec<_> = maps.iter().map(|m| m.view()).collect();
        let joined = ndarray::concatenate(ndarray::Axis(if along_rows { 0 } else { 1 }), &views).map_err(|e| GpuError::DriverCallFailed { reason: format!("heads' maps of different shapes: {e}") })?;
        let built = f32::upload(&self.device, joined.view())?;
        self.stacks.insert((keys, along_rows), self.device.copy(&built)?);
        Ok(built)
    }
}

/// A resident copy's key: its `Weights`' generation, the host matrix's address and shape.
type Key = (u64, usize, usize, usize);

/// A host element type the device uploads: float64 (vectors, VPD factors, runs' arrays) or float32
/// (stored weights, sent as they are).
trait Element: Copy {
    fn upload(d: &Device, m: ArrayView2<Self>) -> Result<Tensor, GpuError>;
}

impl Element for f64 {
    fn upload(d: &Device, m: ArrayView2<f64>) -> Result<Tensor, GpuError> {
        d.upload(m)
    }
}

impl Element for f32 {
    fn upload(d: &Device, m: ArrayView2<f32>) -> Result<Tensor, GpuError> {
        match m.as_slice() {
            Some(values) => d.upload_f32(m.nrows(), m.ncols(), values),
            None => d.upload_f32(m.nrows(), m.ncols(), &m.iter().copied().collect::<Vec<_>>()),
        }
    }
}

/// A host vector as a `1 × n` row view.
fn row(v: &Array1<f64>) -> ArrayView2<'_, f64> {
    v.view().insert_axis(ndarray::Axis(0))
}

/// Runs the checker on `device` for the rest of the process: the large products of host runs
/// ([`dot`], [`logits`]) and whole runs ([`run`]). Returns false when a device was already set.
pub fn use_device(device: Device) -> bool {
    DEVICE.set(Mutex::new(DeviceState::new(device))).is_ok()
}

/// A weight edit is about to change host matrix `m` in place or release it: its resident copy and
/// the stacks built from it are dropped. Every in-place change or release of `M`'s matrices calls
/// this first (`graph::WeightEdit`, `Weights::quantize` and their restores), so no copy is stale
/// and a matrix later allocated at a released address never reads one. Dropping every copy of the
/// shape instead re-uploaded a whole model after each edit (Qwen3-0.6B: the device thread spent
/// its time uploading and freeing buffers).
pub(crate) fn edited<A>(m: &Array2<A>) {
    dropped((m.as_ptr() as usize, m.nrows(), m.ncols()));
}

/// Drops the copies of the host array at `(address, rows, cols)`, of every generation (an edit
/// does not say whose weights it changes; another generation's copy merely uploads again).
fn dropped(at: (usize, usize, usize)) {
    let same = move |k: &Key| (k.1, k.2, k.3) == at;
    on_device(|s| {
        s.resident.retain(|k, _| !same(k));
        s.stacks.retain(|(keys, _), _| !keys.iter().any(same));
        s.products.clear();
        s.summed.clear();
    });
}

/// Drops generation `generation`'s copies and stacks: its `Weights` are gone (the last clone
/// dropped), and their freed matrices' addresses may be reused.
pub(crate) fn forget(generation: u64) {
    on_device(|s| {
        s.resident.retain(|k, _| k.0 != generation);
        s.uploaded.retain(|k| k.0 != generation);
        s.stacks.retain(|(keys, _), _| keys.first().is_none_or(|k| k.0 != generation));
        s.products.clear();
        s.summed.clear();
    });
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

/// `last · Uᵀ` on the device with `U` (the unembedding of `weights`, never edited) resident.
pub(crate) fn logits(weights: &Weights, last: &Array2<f64>) -> Option<Array2<f64>> {
    on_device(|s| {
        s.generation = weights.generation();
        logits_on(s, last, &weights.unembedding)
    })
    .flatten()
}

fn logits_on(s: &mut DeviceState, last: &Array2<f64>, unembedding: &Stored) -> Option<Array2<f64>> {
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
/// A stacked counterfactual run's stand-ins (`copies`' rows): `embed`'s, the counterfactual
/// embedding on every copy, and each stand-in unit's (a VPD unit not computing), on each copy's
/// rows the counterfactual write of its `down_proj` or `o_proj` parts: with `rest` those the copy
/// does not name (as [`standins`] makes a remainder unit's), else those it names (a complement
/// model's); zero for computing units.
fn copy_standins(s: &mut DeviceState, weights: &Weights, circuit: &Circuit, r: &Reference, (c, copy_of_row, within): (&Copies, &gam_gpu::tensor::Indices, &gam_gpu::tensor::Indices)) -> Result<(Tensor, Vec<Tensor>), String> {
    let e = |e: GpuError| e.to_string();
    let (rows, width) = (c.count * c.rows, weights.width());
    let tiled = |s: &mut DeviceState, field: Field, layer: usize| -> Result<Tensor, String> {
        s.ensure_reference(r, (field, layer, 0))?;
        s.device.gather_rows(s.reference(r, (field, layer, 0))?, within).map_err(e)
    };
    let embed = tiled(s, Field::Embed, 0)?;
    // Sites a resumed or halted run does not compute need none.
    let (from, until) = (c.resume.as_ref().map_or(0, |(site, _)| *site), c.halt.unwrap_or(usize::MAX));
    let mut out = Vec::with_capacity(circuit.units.len());
    for (u, unit) in circuit.units.iter().enumerate() {
        let site = unit.block.site();
        let w = match &unit.block {
            _ if site < from || site >= until => s.device.zeros(rows, width).map_err(e)?,
            Block::Slices { layer, down, rest, .. } if !unit.computes => {
                let mlp = weights.layers[*layer].mlp.as_ref().ok_or("a VPD view of a layer without an MLP")?;
                let vpd = weights.vpd.get(layer).ok_or_else(|| format!("layer {layer} has no VPD view"))?;
                let active = tiled(s, Field::Active, *layer)?;
                let named = sliced(s, (&vpd.down_u, &vpd.down_v), Matrix::Host(&mlp.out), down, false, &active, c.masks.get(&(u, 1)).map(|m| (m, copy_of_row, c.rows))).map_err(e)?;
                if *rest {
                    let full = s.reference_product(r, (Field::Active, *layer), PRODUCT_MLP_OUT, Matrix::Host(&mlp.out))?;
                    let full = s.device.gather_rows(&full, within).map_err(e)?;
                    less_named(s, full, named, down, vpd.down_u.nrows()).map_err(e)?
                } else {
                    named
                }
            }
            Block::AttnSlices { layer, o, rest, .. } if !unit.computes => {
                let a = weights.vpd_attention.get(layer).ok_or_else(|| format!("layer {layer}'s attention has no VPD view"))?;
                let lw = &weights.layers[*layer];
                let z = tiled(s, Field::Reads, *layer)?;
                let outputs = s.stacked(&lw.heads.iter().map(|h| &h.output).collect::<Vec<_>>(), false).map_err(e)?;
                let named = sliced(s, (&a.o.0, &a.o.1), Matrix::Device(&outputs), o, false, &z, c.masks.get(&(u, 3)).map(|m| (m, copy_of_row, c.rows))).map_err(e)?;
                if *rest {
                    let full = s.reference_product(r, (Field::Reads, *layer), PRODUCT_ATTENTION_OUT, Matrix::Device(&outputs))?;
                    let full = s.device.gather_rows(&full, within).map_err(e)?;
                    less_named(s, full, named, o, a.o.0.nrows()).map_err(e)?
                } else {
                    named
                }
            }
            _ if unit.computes => s.device.zeros(rows, width).map_err(e)?,
            _ => return Err("a stacked stand-in of a block other than a VPD unit".into()),
        };
        out.push(w);
    }
    Ok((embed, out))
}

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
            // A remainder unit (`rest`: the matrix less its named parts) subtracts its named parts
            // from the full write, kept per counterfactual run (`reference_product`), as `sliced`
            // computes it.
            Block::Slices { layer, down, rest, .. } => {
                let mlp = weights.layers[*layer].mlp.as_ref().ok_or("a VPD view of a layer without an MLP")?;
                let vpd = weights.vpd.get(layer).ok_or_else(|| format!("layer {layer} has no VPD view"))?;
                s.ensure_reference(r, (Field::Active, *layer, 0))?;
                let active = s.device.copy(s.reference(r, (Field::Active, *layer, 0))?).map_err(e)?;
                w = sliced(s, (&vpd.down_u, &vpd.down_v), Matrix::Host(&mlp.out), down, false, &active, None).map_err(e)?;
                if *rest {
                    let full = s.reference_product(r, (Field::Active, *layer), PRODUCT_MLP_OUT, Matrix::Host(&mlp.out))?;
                    w = less_named(s, full, w, down, vpd.down_u.nrows()).map_err(e)?;
                }
            }
            Block::AttnSlices { layer, o, rest, .. } => {
                let a = weights.vpd_attention.get(layer).ok_or_else(|| format!("layer {layer}'s attention has no VPD view"))?;
                let lw = &weights.layers[*layer];
                s.ensure_reference(r, (Field::Reads, *layer, 0))?;
                let z = s.device.copy(s.reference(r, (Field::Reads, *layer, 0))?).map_err(e)?;
                let outputs = s.stacked(&lw.heads.iter().map(|h| &h.output).collect::<Vec<_>>(), false).map_err(e)?;
                w = sliced(s, (&a.o.0, &a.o.1), Matrix::Device(&outputs), o, false, &z, None).map_err(e)?;
                if *rest {
                    let full = s.reference_product(r, (Field::Reads, *layer), PRODUCT_ATTENTION_OUT, Matrix::Device(&outputs))?;
                    w = less_named(s, full, w, o, a.o.0.nrows()).map_err(e)?;
                }
            }
            Block::Features { layer, features, rest } => {
                s.ensure_reference(r, (Field::Input, *layer, 0))?;
                let x_hat = s.device.copy(s.reference(r, (Field::Input, *layer, 0))?).map_err(e)?;
                let named = features_write(s, weights, *layer, features, &x_hat)?;
                w = if *rest {
                    s.ensure_reference(r, (Field::Mlp, *layer, 0))?;
                    let mut all = s.device.copy(s.reference(r, (Field::Mlp, *layer, 0))?).map_err(e)?;
                    s.device.axpy(&mut all, -1.0, &named).map_err(e)?;
                    all
                } else {
                    named
                };
            }
        }
        out.push(w);
    }
    Ok((embed, out))
}

/// Copies of one batch run together as one batch ([`execute_copies`]): `count` copies of `rows` rows
/// each, copy `j` at rows `j·rows..(j + 1)·rows`, each naming its own subcomponents of the merged
/// circuit's VPD units.
pub(crate) struct Copies {
    pub count: usize,
    pub rows: usize,
    /// By merged unit and slot (an MLP's `c_fc` 0 and `down_proj` 1; an attention's `q_proj` 0,
    /// `k_proj` 1, `v_proj` 2 and `o_proj` 3).
    pub masks: BTreeMap<(usize, usize), CopyMask>,
    /// The counterfactual run (of one copy's rows) whose values the parts a copy does not name
    /// carry; `None`: they are deleted. With it the merged circuit has at every site a computing
    /// unit and a stand-in unit (`rest`, not computing) whose stand-in on each copy's rows is the
    /// counterfactual write of the `down_proj` or `o_proj` parts the copy does not name: the full
    /// counterfactual write less the named ones' (by the copy's counts).
    pub standin: Option<std::sync::Arc<Reference>>,
    /// Start at this site with this stream (one copy's rows, tiled over the copies): the copies
    /// agree before it, so a one-copy run made that prefix ([`Copies::halt`]).
    pub resume: Option<(usize, Array2<f64>)>,
    /// Stop before this site: the run's execution holds the stream entering it in place of
    /// log-probabilities ([`prefix_stream`]).
    pub halt: Option<usize>,
    /// Per copy (empty: none), whether its logits read the embedding's stand-in in place of the
    /// embedding (a complement model of a program that routes the embedding to the logits).
    pub embed_out: Vec<bool>,
}

/// The stream entering site `halt` of a stacked run halted there ([`Copies::halt`]).
pub(crate) fn prefix_stream(execution: Execution) -> Array2<f64> {
    execution.log_probabilities
}

/// A stacked run's rows by copy: each row's copy and its row within the copy.
type CopyRows<'a> = Option<(&'a Copies, &'a gam_gpu::tensor::Indices, &'a gam_gpu::tensor::Indices)>;

/// What each copy names of one slot of a merged VPD unit.
#[derive(Clone)]
pub(crate) struct CopyMask {
    /// `counts[[j, i]]`: how many of copy `j`'s units name the slot's `i`-th listed subcomponent
    /// (the unit's list in order, its remainder left out).
    pub counts: Array2<f64>,
    /// Per copy, whether it names the slot's remainder.
    pub remainder: Vec<bool>,
}

/// A slot's mask with each row's copy (rows × 1 indices into the copies).
type Masked<'a> = Option<(&'a CopyMask, &'a gam_gpu::tensor::Indices, usize)>;

/// `copies.count` circuits that differ only in the VPD subcomponents they name, run as one batch:
/// `circuit` is their merge (each unit's lists the union of the copies' lists at its site, every
/// unit computing and reading every earlier write), `job` the batch's rows once per copy with each
/// copy's scored rows in turn, and `copies` what each copy names. A unit's named part on a copy's
/// rows is its subcomponents weighted by the copy's counts, so each copy computes as its own
/// circuit, with one product per matrix and one attention call for all of them. Returns each
/// copy's execution as the device returns it: the logits at the scored rows of every copy in turn
/// (split and normalized on the calling thread by [`per_copy`]), and for a run scoring no rows (a
/// donor run) every unit's write, the kept normed inputs and heads' reads over all the copies'
/// rows; `None` without a device or for a view it does not cover.
pub(crate) fn execute_copies(weights: &Weights, circuit: &Circuit, job: &Run, copies: &Copies) -> Option<Result<Execution, String>> {
    if !stack_covered(weights, circuit, job, copies) {
        return None;
    }
    on_device(|s| copies_on(s, weights, circuit, job, Some(copies)))
}

/// Whether the device runs `circuit` stacked: VPD units alone (computing or stand-in, named parts
/// or their `rest`), attentions of heads alike, and the same number of scored rows per copy.
pub(crate) fn stack_covered(weights: &Weights, circuit: &Circuit, job: &Run, copies: &Copies) -> bool {
    let covered = |u: &crate::graph::Unit| match &u.block {
        Block::Slices { .. } => true,
        Block::AttnSlices { layer, .. } => weights.layers.get(*layer).is_some_and(|lw| lw.heads.first().is_some_and(|first| lw.heads.iter().all(|h| alike(h, first, false)))),
        _ => false,
    };
    copies.count > 0 && job.scored.len() % copies.count == 0 && circuit.units.iter().all(covered)
}

/// A stacked run's logits split into its copies' (each copy's scored rows in turn), each
/// normalized.
pub(crate) fn per_copy(execution: Execution, count: usize) -> Result<Vec<Array2<f64>>, String> {
    let each = execution.log_probabilities.nrows() / count.max(1);
    (0..count)
        .map(|j| {
            let logits = execution.log_probabilities.slice(ndarray::s![j * each..(j + 1) * each, ..]).to_owned();
            normalized(Execution::of(logits, Vec::new(), None, BTreeMap::new())).map(|x| x.log_probabilities)
        })
        .collect()
}

/// `circuit` run on the process's device ([`use_device`]), or `None` when there is none or the
/// circuit holds a block the device path does not cover.
pub(crate) fn run(weights: &Weights, circuit: &Circuit, job: &Run) -> Option<Result<Execution, String>> {
    // A VPD-view attention's heads attend together (alike).
    let covered = |b: &Block| match b {
        Block::Heads { .. } | Block::Neurons { .. } | Block::Slices { .. } | Block::Features { .. } => true,
        Block::AttnSlices { layer, .. } => weights.layers.get(*layer).is_some_and(|lw| lw.heads.first().is_some_and(|first| lw.heads.iter().all(|h| alike(h, first, false)))),
    };
    if !circuit.units.iter().all(|u| covered(&u.block)) {
        return None;
    }
    // The device thread returns the logits; their log-softmax runs on the calling thread, so the
    // runs of a batch normalize in parallel while the device serves the next one.
    on_device(|s| logits_on_device(s, weights, circuit, job)).map(|run| run.and_then(normalized))
}

/// An execution whose `log_probabilities` hold the logits, normalized in place.
pub(crate) fn normalized(mut execution: Execution) -> Result<Execution, String> {
    for mut row in execution.log_probabilities.outer_iter_mut() {
        let values = gam_math::categorical::log_softmax(row.as_slice().ok_or("a contiguous row")?).map_err(|e| e.to_string())?;
        row.assign(&Array1::from(values));
    }
    Ok(execution)
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
    /// Every stand-in is zero (a deleting run, or `M`): nothing is added for one, and a route that
    /// reads every writer in the stream (and no unit of its own site) reads the stream itself.
    zero: bool,
    /// Per unit, whether its write is in `stream` (its site is done).
    added: Vec<bool>,
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
        // With zero stand-ins the stream is the stand-in stream (what pushes added) plus every
        // written writer's write, so reading all of those is reading the stream.
        if self.zero {
            let all = match incoming {
                Incoming::AllBut(cut) => cut.iter().all(|w| self.of(*w).0.is_none()),
                Incoming::Only(kept) => kept.contains(&Writer::Embed) && self.writes.iter().zip(&self.added).enumerate().all(|(u, (w, &added))| w.is_none() || added == kept.contains(&Writer::Unit(u))),
            };
            if all {
                return d.copy(&self.stream);
            }
        }
        let (mut x, sign, writers) = match incoming {
            Incoming::AllBut(cut) => (d.copy(&self.stream)?, -1.0, cut),
            Incoming::Only(kept) => (d.copy(&self.standin_stream)?, 1.0, kept),
        };
        for &w in writers {
            let (actual, standin) = self.of(w);
            if let Some(a) = actual {
                d.axpy(&mut x, sign, a)?;
                if !self.zero {
                    d.axpy(&mut x, -sign, standin)?;
                }
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

/// [`run`] on a given device state (the tests run it on the host backend), with the logits in place
/// of the log-probabilities ([`normalized`] turns them into those).
pub(crate) fn logits_on_device(s: &mut DeviceState, weights: &Weights, circuit: &Circuit, job: &Run) -> Result<Execution, String> {
    copies_on(s, weights, circuit, job, None)
}

/// [`logits_on_device`], with `copies` the stacked run's ([`execute_copies`]; the tests run it on the
/// host backend, its logits split by [`per_copy`]).
pub(crate) fn copies_on(s: &mut DeviceState, weights: &Weights, circuit: &Circuit, job: &Run, copies: Option<&Copies>) -> Result<Execution, String> {
    let e = |e: GpuError| e.to_string();
    s.generation = weights.generation();
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
    // A stacked run's copy of each row and its row within the copy.
    let copy_rows = match copies {
        Some(c) if c.count * c.rows == rows => Some((c, s.device.upload_indices(&(0..rows).map(|r| (r / c.rows) as u32).collect::<Vec<_>>()).map_err(e)?, s.device.upload_indices(&(0..rows).map(|r| (r % c.rows) as u32).collect::<Vec<_>>()).map_err(e)?)),
        Some(c) => return Err(format!("{} copies of {} rows for a batch of {rows}", c.count, c.rows)),
        None => None,
    };
    // Stand-ins: from the counterfactual run, or zeros for `M` (it reads none).
    let stacked_standin = copy_rows.as_ref().and_then(|(c, ..)| c.standin.as_deref()).filter(|r| !r.zero);
    let zero = stacked_standin.is_none() && (matches!(job.reference, Some(r) if r.zero) || (job.reference.is_none() && circuit.units.iter().all(|u| u.computes)));
    let (embed_standin, standins): (Tensor, Vec<Tensor>) = match (job.reference, stacked_standin, copy_rows.as_ref()) {
        // A stacked counterfactual run: each stand-in unit's per-copy stand-in ([`copy_standins`]).
        (None, Some(r), Some((c, copy_of_row, within))) => copy_standins(s, weights, circuit, r, (c, copy_of_row, within))?,
        _ => match job.reference {
        Some(r) if r.embed.nrows() != rows => return Err(format!("a counterfactual run of {} tokens for a batch of {rows}", r.embed.nrows())),
        // Deletion: every stand-in is zero, nothing to upload or compute.
        Some(r) if r.zero => (s.device.zeros(rows, width).map_err(e)?, (0..units).map(|_| s.device.zeros(rows, width)).collect::<Result<_, _>>().map_err(e)?),
        Some(r) => standins(s, weights, circuit, r)?,
        None if circuit.units.iter().all(|u| u.computes) => (s.device.zeros(rows, width).map_err(e)?, (0..units).map(|_| s.device.zeros(rows, width)).collect::<Result<_, _>>().map_err(e)?),
        None => return Err("a program's undeclared pieces take their values from the counterfactual run, which this batch lacks".into()),
        },
    };
    let mut captured = job.capture.then(|| Reference {
        id: crate::graph::next_reference_id(),
        embed: Array2::zeros((0, 0)),
        reads: weights.layers.iter().map(|l| vec![Array2::zeros((0, 0)); l.heads.len()]).collect(),
        active: vec![Array2::zeros((0, 0)); weights.layers.len()],
        mlp: vec![Array2::zeros((0, 0)); weights.layers.len()],
        inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
        attention_inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
        zero: false,
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
        zero,
        added: vec![false; units],
    };
    let ops = job.ops;
    let mut kept = BTreeMap::new();
    let mut reads_kept = BTreeMap::new();
    if let Some(c) = captured.as_mut() {
        c.embed = s.device.download(&st.embed).map_err(e)?;
    }
    after(&s.device, ops, None, &mut st).map_err(e)?;
    // A stacked run resumed after the copies' shared prefix starts from its stream.
    let (resume_at, halt_at) = match copy_rows.as_ref() {
        Some((c, _, within)) => {
            if let Some((site, prefix)) = &c.resume {
                let one = s.device.upload(prefix.view()).map_err(e)?;
                st.stream = s.device.gather_rows(&one, within).map_err(e)?;
                (*site, c.halt)
            } else {
                (0, c.halt)
            }
        }
        None => (0, None),
    };
    let mut at = 0;
    while at < order.len() {
        let site = circuit.units[order[at]].block.site();
        let end = order[at..].iter().position(|&u| circuit.units[u].block.site() != site).map_or(order.len(), |k| at + k);
        if halt_at.is_some_and(|h| site >= h) {
            break;
        }
        if site < resume_at {
            at = end;
            continue;
        }
        let pad_job = Padding { places: padded.as_ref(), sequences, longest };
        let mut slice_writes = vpd_site(s, weights, circuit, job, (site, &order[at..end]), &st, &pad_job, (&mut kept, &mut reads_kept), copy_rows.as_ref().map(|(c, r, t)| (*c, r, t)))?;
        for &u in &order[at..end] {
            let unit = &circuit.units[u];
            if !unit.computes {
                continue;
            }
            if let Some(value) = job.swaps.get(&u) {
                st.writes[u] = Some(s.device.upload(value.view()).map_err(e)?);
                continue;
            }
            if let Some(w) = slice_writes.remove(&u) {
                st.writes[u] = Some(w);
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
                    // All the unit's heads at once where they share their shapes, head-norm gains and
                    // rotary (Qwen3, vpd4l): one product per map and one attention call per layer.
                    if let Some(batched) = heads_together(s, lw, heads, &normed_inputs, &pad_job, captured.is_some()) {
                        let (write, reads) = batched.map_err(e)?;
                        if let (Some(c), Some(reads)) = (captured.as_mut(), reads) {
                            for (i, &h) in heads.iter().enumerate() {
                                let width = lw.heads[h].value.nrows();
                                c.reads[*layer][h] = reads.slice(ndarray::s![.., i * width..(i + 1) * width]).to_owned();
                            }
                        }
                        write
                    } else {
                    let mut out = s.device.zeros(rows, width).map_err(e)?;
                    for &h in heads {
                        let hw = &lw.heads[h];
                        let project = |s: &mut DeviceState, x: &Tensor, map: &Stored, norm: Option<&(Array1<f64>, f64)>| -> Result<Tensor, GpuError> {
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
                        let mut k = project(s, &normed_inputs[1], &*hw.key, hw.key_norm.as_ref()).map_err(e)?;
                        let v = project(s, &normed_inputs[2], &*hw.value, None).map_err(e)?;
                        if let Some(r) = hw.rotary {
                            let key = s.rotations(r, &positions).map_err(e)?;
                            let (cos, sin) = s.rotations.get(&key).ok_or("rotation tables")?;
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
                    // A large block: the whole MLP (resident), less the few neurons outside it; a small
                    // block: its own rows alone (the whole MLP would cost n / |block| times as much).
                    let (write, active) = if 2 * neurons.len() > n {
                        let (all, active) = neuron_writes(s, mlp, &x_hat, None).map_err(e)?;
                        if neurons.len() == n {
                            (all, Some(active))
                        } else {
                            let inside: std::collections::BTreeSet<usize> = neurons.iter().copied().collect();
                            let rest: Vec<usize> = (0..n).filter(|i| !inside.contains(i)).collect();
                            let (outside, _) = neuron_writes(s, mlp, &x_hat, Some(&rest)).map_err(e)?;
                            let mut w = all;
                            s.device.axpy(&mut w, -1.0, &outside).map_err(e)?;
                            (w, None)
                        }
                    } else {
                        (neuron_writes(s, mlp, &x_hat, Some(neurons)).map_err(e)?.0, None)
                    };
                    if let Some(c) = captured.as_mut() {
                        let Some(active) = active.as_ref() else {
                            return Err("a capture needs each MLP whole in one unit".into());
                        };
                        c.mlp[*layer] = s.device.download(&write).map_err(e)?;
                        c.active[*layer] = s.device.download(active).map_err(e)?;
                        // Only transcoder features and VPD MLP subcomponents read it.
                        if !weights.transcoders.is_empty() || !weights.vpd.is_empty() {
                            c.inputs[*layer] = s.device.download(&x_hat).map_err(e)?;
                        }
                    }
                    write
                }
                Block::Features { layer, features, rest } => {
                    let lw = &weights.layers[*layer];
                    let unit_x = s.device.rms_norm(&inputs[0].1, lw.mlp_norm.epsilon).map_err(e)?;
                    let gain = s.ensure(row(&lw.mlp_norm.gain)).map_err(e)?;
                    let mut x_hat = s.device.zeros(rows, width).map_err(e)?;
                    s.device.scale_columns(&mut x_hat, &unit_x, s.get(gain).map_err(e)?, false).map_err(e)?;
                    normed(&s.device, ops, (site, u, 0), &mut x_hat, &mut kept).map_err(e)?;
                    let named = features_write(s, weights, *layer, features, &x_hat)?;
                    if *rest {
                        // The MLP less the named features: every other feature and the error.
                        let mlp = lw.mlp.as_ref().ok_or("a transcoder on a layer without an MLP")?;
                        let (mut all, _) = neuron_writes(s, mlp, &x_hat, None).map_err(e)?;
                        s.device.axpy(&mut all, -1.0, &named).map_err(e)?;
                        all
                    } else {
                        named
                    }
                }
                Block::Slices { .. } | Block::AttnSlices { .. } => return Err("a VPD block outside its site's pass".into()),
            };
            st.writes[u] = Some(write);
        }
        for &u in &order[at..end] {
            match &st.writes[u] {
                Some(w) => s.device.axpy(&mut st.stream, 1.0, w).map_err(e)?,
                None if st.zero => {}
                None => s.device.axpy(&mut st.stream, 1.0, &st.standins[u]).map_err(e)?,
            }
            if !st.zero {
                s.device.axpy(&mut st.standin_stream, 1.0, &st.standins[u]).map_err(e)?;
            }
            st.added[u] = true;
        }
        after(&s.device, ops, Some(site), &mut st).map_err(e)?;
        at = end;
    }
    if halt_at.is_some() {
        return Ok(Execution::of(s.device.download(&st.stream).map_err(e)?, Vec::new(), None, BTreeMap::new()));
    }
    let mut last = st.input(&s.device, &circuit.logits).map_err(e)?;
    // Copies whose logits read the embedding's stand-in: the stand-in less the embedding added on
    // their rows.
    if let Some((c, ..)) = copy_rows.as_ref().filter(|(c, ..)| c.embed_out.iter().any(|&b| b)) {
        let mut change = s.device.copy(&st.embed_standin).map_err(e)?;
        s.device.axpy(&mut change, -1.0, &st.embed).map_err(e)?;
        let others: Vec<u32> = c.embed_out.iter().enumerate().filter(|(_, b)| !**b).flat_map(|(j, _)| (j * c.rows..(j + 1) * c.rows).map(|r| r as u32)).collect();
        if !others.is_empty() {
            scale_rows(&s.device, &mut change, &s.device.upload_indices(&others).map_err(e)?, 0.0).map_err(e)?;
        }
        s.device.axpy(&mut last, 1.0, &change).map_err(e)?;
    }
    // A run scoring no rows (a donor or counterfactual run, read for its writes) makes no logits: a
    // tensor of no rows does not download on every backend (CUDA keeps one value for it).
    let logits = if job.scored.is_empty() {
        Array2::zeros((0, weights.unembedding.nrows()))
    } else {
        let picked = s.device.upload_indices(&job.scored.iter().map(|&r| r as u32).collect::<Vec<_>>()).map_err(e)?;
        let last = s.device.gather_rows(&last, &picked).map_err(e)?;
        let unit_last = s.device.rms_norm(&last, weights.final_norm.epsilon).map_err(e)?;
        let gain = s.ensure(row(&weights.final_norm.gain)).map_err(e)?;
        let mut normed_last = s.device.zeros(job.scored.len(), width).map_err(e)?;
        s.device.scale_columns(&mut normed_last, &unit_last, s.get(gain).map_err(e)?, false).map_err(e)?;
        let u = s.ensure(weights.unembedding.view()).map_err(e)?;
        let mut logits = s.device.zeros(job.scored.len(), weights.unembedding.nrows()).map_err(e)?;
        s.device.gemm(&mut logits, 1.0, &normed_last, Op::N, s.get(u).map_err(e)?, Op::T, 0.0, arithmetic).map_err(e)?;
        s.device.download(&logits).map_err(e)?
    };
    // The units' writes come back only from a run that scores no rows: a donor run, whose writes
    // other runs read (swaps, site operations, `Checker::measure_typical`). On Qwen3-0.6B every
    // run's writes are about a gigabyte of float64.
    let writes = if job.scored.is_empty() && !job.capture {
        st.writes.iter().map(|w| w.as_ref().map(|t| s.device.download(t)).transpose()).collect::<Result<Vec<_>, _>>().map_err(e)?
    } else {
        vec![None; units]
    };
    let mut execution = Execution::of(logits, writes, captured, kept);
    execution.reads = reads_kept;
    Ok(execution)
}

/// The writes of a site's VPD-view units (`graph::run`'s passes), by unit. An MLP's: each reader of
/// the hidden stream (a unit with `down_proj` subcomponents or the remainder) takes the
/// counterfactual pre-activation plus the `c_fc` writes it reads (on x minus on x′), applies the
/// MLP's law and writes through its `down_proj` subcomponents. An attention's likewise: each reader
/// of the queries, keys and values (a unit with `o_proj` subcomponents) takes the counterfactual
/// ones plus the q/k/v writes it reads, runs the heads' attention on them (no head norms, as on the
/// host) and writes through its `o_proj` subcomponents.
fn vpd_site(s: &mut DeviceState, weights: &Weights, circuit: &Circuit, job: &Run, (site, units): (usize, &[usize]), st: &Streams, pad: &Padding, (kept, reads_kept): (&mut BTreeMap<(usize, usize), Array2<f64>>, &mut BTreeMap<usize, Array2<f64>>), copies: CopyRows) -> Result<BTreeMap<usize, Tensor>, String> {
    let e = |e: GpuError| e.to_string();
    // A slot's per-copy counts in a stacked run.
    let mask = |u: usize, slot: usize| -> Masked<'_> { copies.and_then(|(c, rows, _)| c.masks.get(&(u, slot)).map(|m| (m, rows, c.rows))) };
    // The counterfactual run the stand-ins come from (a stacked run's is one copy's, its arrays
    // tiled over the copies), `None` for `M` or a deleting run.
    let cf: Option<&Reference> = match copies {
        Some((c, ..)) => c.standin.as_deref().filter(|r| !r.zero),
        None => job.reference.filter(|r| !r.zero),
    };
    let tiled = |s: &DeviceState, t: Tensor| -> Result<Tensor, GpuError> {
        match copies {
            Some((_, _, within)) => s.device.gather_rows(&t, within),
            None => Ok(t),
        }
    };
    let (rows, width, ops) = (job.tokens.len(), weights.width(), job.ops);
    let mut writes = BTreeMap::new();
    let computing = |u: &usize| circuit.units[*u].computes && !job.swaps.contains_key(u);
    // A unit's route inputs (after the cuts into the site) and their normed values under `norm`, the
    // operations on the site's input applied, less the counterfactual normed input `x_ref`.
    let normed_deltas = |s: &mut DeviceState, u: usize, norm: &crate::graph::Norm, x_ref: Option<&Tensor>, kept: &mut BTreeMap<(usize, usize), Array2<f64>>| -> Result<Vec<Tensor>, GpuError> {
        let unit = &circuit.units[u];
        let slots: Vec<usize> = unit.block.routes().iter().map(|r| r.slot()).collect();
        let mut inputs: Vec<(usize, Tensor)> = slots.iter().map(|&slot| st.input(&s.device, &unit.routes[slot]).map(|x| (slot, x))).collect::<Result<_, _>>()?;
        cut_inputs(&s.device, ops, site, unit, &mut inputs, st)?;
        let gain = s.ensure(row(&norm.gain))?;
        let mut out = Vec::with_capacity(inputs.len());
        for (m, (_, x)) in inputs.iter().enumerate() {
            let unit_x = s.device.rms_norm(x, norm.epsilon)?;
            let mut x_hat = s.device.zeros(rows, width)?;
            s.device.scale_columns(&mut x_hat, &unit_x, s.get(gain)?, false)?;
            normed(&s.device, ops, (site, u, m), &mut x_hat, kept)?;
            if let Some(x) = x_ref {
                s.device.axpy(&mut x_hat, -1.0, x)?;
            }
            out.push(x_hat);
        }
        Ok(out)
    };
    let reference = |s: &mut DeviceState, field: Field, layer: usize| -> Result<Option<Tensor>, String> {
        match cf {
            Some(r) => {
                s.ensure_reference(r, (field, layer, 0))?;
                let x = s.device.copy(s.reference(r, (field, layer, 0))?).map_err(e)?;
                Ok(Some(tiled(s, x).map_err(e)?))
            }
            // `M` (every unit computing and read) needs no reference: the deltas sum to its own; a
            // deleting run's reference is zero.
            None => Ok(None),
        }
    };
    let slices: Vec<usize> = units.iter().copied().filter(|&u| matches!(circuit.units[u].block, Block::Slices { .. })).collect();
    if let Some(&first) = slices.first() {
        let Block::Slices { layer, .. } = circuit.units[first].block else { return Err("a VPD-view unit of another block".into()) };
        let lw = &weights.layers[layer];
        let mlp = lw.mlp.as_ref().ok_or("a VPD view of a layer without an MLP")?;
        let vpd = weights.vpd.get(&layer).ok_or_else(|| format!("layer {layer} has no VPD view"))?;
        let x_ref = reference(s, Field::Input, layer)?;
        let bias = s.ensure(row(&mlp.bias)).map_err(e)?;
        // The counterfactual pre-activation, the same for every program on this counterfactual run.
        let mut pre_ref = match cf {
            Some(r) => {
                let p = s.reference_product(r, (Field::Input, layer), PRODUCT_MLP_GATE, Matrix::Host(&mlp.gate))?;
                tiled(s, p).map_err(e)?
            }
            None => s.device.zeros(rows, mlp.gate.nrows()).map_err(e)?,
        };
        s.device.add_row(&mut pre_ref, 1.0, s.get(bias).map_err(e)?).map_err(e)?;
        let mut deltas: BTreeMap<usize, Tensor> = BTreeMap::new();
        for &u in slices.iter().filter(|u| computing(u)) {
            let Block::Slices { fc, rest, .. } = &circuit.units[u].block else { return Err("a VPD-view unit of another block".into()) };
            let x = normed_deltas(s, u, &lw.mlp_norm, x_ref.as_ref(), kept).map_err(e)?;
            deltas.insert(u, sliced(s, (&vpd.fc_u, &vpd.fc_v), Matrix::Host(&mlp.gate), fc, *rest, &x[0], mask(u, 0)).map_err(e)?);
        }
        let codes = s.device.upload_indices(&vec![law_of(mlp.law).code(); mlp.gate.nrows()]).map_err(e)?;
        for &u in slices.iter().filter(|u| computing(u)) {
            let unit = &circuit.units[u];
            let Block::Slices { down, rest, .. } = &unit.block else { return Err("a VPD-view unit of another block".into()) };
            if down.is_empty() && !rest {
                writes.insert(u, s.device.zeros(rows, width).map_err(e)?);
                continue;
            }
            let mut pre = s.device.copy(&pre_ref).map_err(e)?;
            for (_, delta) in deltas.iter().filter(|(w, _)| reads_hidden(&unit.hidden, **w)) {
                s.device.axpy(&mut pre, 1.0, delta).map_err(e)?;
            }
            let h = s.device.law_values(&pre, &codes, gelu_tanh_constant()).map_err(e)?;
            writes.insert(u, sliced(s, (&vpd.down_u, &vpd.down_v), Matrix::Host(&mlp.out), down, *rest, &h, mask(u, 1)).map_err(e)?);
        }
    }
    let attention: Vec<usize> = units.iter().copied().filter(|&u| matches!(circuit.units[u].block, Block::AttnSlices { .. })).collect();
    if let Some(&first) = attention.first() {
        let Block::AttnSlices { layer, .. } = circuit.units[first].block else { return Err("a VPD-view unit of another block".into()) };
        let lw = &weights.layers[layer];
        let a = weights.vpd_attention.get(&layer).ok_or_else(|| format!("layer {layer}'s attention has no VPD view"))?;
        let head = lw.heads.first().ok_or("a VPD view of an attention without heads")?;
        // The four native matrices VPD decomposes (`graph::attention_maps`): q, k, v stacked over
        // heads, the output columns side by side.
        let mut maps = Vec::with_capacity(4);
        for (along_rows, map) in [(true, 0), (true, 1), (true, 2), (false, 3)] {
            let per_head: Vec<&Stored> = lw.heads.iter().map(|h| [&h.query, &*h.key, &*h.value, &h.output][map]).collect();
            maps.push(s.stacked(&per_head, along_rows).map_err(e)?);
        }
        let x_ref = reference(s, Field::AttentionInput, layer)?;
        // The counterfactual queries, keys and values, the same for every program on this run.
        let mut refs = Vec::with_capacity(3);
        for (m, map) in maps[..3].iter().enumerate() {
            refs.push(match cf {
                Some(r) => {
                    let p = s.reference_product(r, (Field::AttentionInput, layer), PRODUCT_ATTENTION_IN + m, Matrix::Device(map))?;
                    tiled(s, p).map_err(e)?
                }
                None => s.device.zeros(rows, map.rows()).map_err(e)?,
            });
        }
        let factors = [&a.q, &a.k, &a.v];
        let mut deltas: BTreeMap<usize, Vec<Tensor>> = BTreeMap::new();
        for &u in attention.iter().filter(|u| computing(u)) {
            let Block::AttnSlices { q, k, v, rest, .. } = &circuit.units[u].block else { return Err("a VPD-view unit of another block".into()) };
            let x = normed_deltas(s, u, &lw.attention, x_ref.as_ref(), kept).map_err(e)?;
            let mut ds = Vec::with_capacity(3);
            for (m, list) in [q, k, v].into_iter().enumerate() {
                ds.push(sliced(s, (&factors[m].0, &factors[m].1), Matrix::Device(&maps[m]), list, *rest, &x[m], mask(u, m)).map_err(e)?);
            }
            deltas.insert(u, ds);
        }
        for &u in attention.iter().filter(|u| computing(u)) {
            let unit = &circuit.units[u];
            let Block::AttnSlices { o, rest, .. } = &unit.block else { return Err("a VPD-view unit of another block".into()) };
            if o.is_empty() && !rest {
                writes.insert(u, s.device.zeros(rows, width).map_err(e)?);
                continue;
            }
            let mut qkv = refs.iter().map(|x| s.device.copy(x)).collect::<Result<Vec<_>, _>>().map_err(e)?;
            for (_, ds) in deltas.iter().filter(|(w, _)| reads_hidden(&unit.hidden, **w)) {
                for (x, dx) in qkv.iter_mut().zip(ds) {
                    s.device.axpy(x, 1.0, dx).map_err(e)?;
                }
            }
            let [q, k, v]: [Tensor; 3] = qkv.try_into().map_err(|_| "three projections")?;
            let mut z = attend(s, (q, k, v), (lw.heads.len(), head.query.nrows()), head, false, pad).map_err(e)?;
            // Operations on the heads' reads (and their recording) act on the host copy, as there.
            if ops.head_reads.iter().any(|(at, ..)| *at == site) || ops.record_reads.contains(&site) {
                let mut host = s.device.download(&z).map_err(e)?;
                ops.head_reads_of(site, u, lw, &mut host, reads_kept);
                z = s.device.upload(host.view()).map_err(e)?;
            }
            writes.insert(u, sliced(s, (&a.o.0, &a.o.1), Matrix::Device(&maps[3]), o, *rest, &z, mask(u, 3)).map_err(e)?);
        }
    }
    Ok(writes)
}

/// How the run's rows sit in the padded layout attention reads (`run_on`).
struct Padding<'a> {
    places: Option<&'a gam_gpu::tensor::Indices>,
    sequences: usize,
    longest: usize,
}

/// Heads `heads` of `lw` on their normed query, key and value inputs, all at once: each map's
/// heads stacked into one product, the heads split into whole sequences (padded) for one attention
/// call, merged back and written through their stacked output columns. `None` when the heads differ
/// in shape, head-norm gain or rotary (the caller runs them one by one). With `read`, the heads'
/// reads side by side (rows × heads' value widths) as well.
fn heads_together(s: &mut DeviceState, lw: &crate::graph::LayerWeights, heads: &[usize], normed: &[Tensor], pad: &Padding, read: bool) -> Option<Result<(Tensor, Option<Array2<f64>>), GpuError>> {
    let first = &lw.heads[*heads.first()?];
    if !heads.iter().all(|&h| alike(&lw.heads[h], first, true)) {
        return None;
    }
    Some(heads_together_on(s, lw, heads, normed, pad, read))
}

/// Whether head `w` attends as `first` in one [`attend`] call: the same shapes (queries, keys and
/// values of one width), rotary, scale and mask, and with `norms` the same head-norm gains.
fn alike(w: &crate::graph::HeadWeights, first: &crate::graph::HeadWeights, norms: bool) -> bool {
    let width = first.query.nrows();
    w.query.dim() == first.query.dim() && w.key.dim() == first.key.dim() && w.value.dim() == first.value.dim() && w.output.dim() == first.output.dim() && first.key.nrows() == width && first.value.nrows() == width && (!norms || w.query_norm == first.query_norm && w.key_norm == first.key_norm) && w.rotary == first.rotary && w.scale == first.scale && w.causal == first.causal
}

fn heads_together_on(s: &mut DeviceState, lw: &crate::graph::LayerWeights, heads: &[usize], normed: &[Tensor], pad: &Padding, read: bool) -> Result<(Tensor, Option<Array2<f64>>), GpuError> {
    let first = &lw.heads[heads[0]];
    let (n, width, rows) = (heads.len(), first.query.nrows(), normed[0].rows());
    let arithmetic = s.arithmetic();
    let project = |s: &mut DeviceState, x: &Tensor, maps: Vec<&Stored>| -> Result<Tensor, GpuError> {
        let w = s.stacked(&maps, true)?;
        let mut p = s.device.zeros(rows, w.rows())?;
        s.device.gemm(&mut p, 1.0, x, Op::N, &w, Op::T, 0.0, arithmetic)?;
        Ok(p)
    };
    let q = project(s, &normed[0], heads.iter().map(|&h| &lw.heads[h].query).collect())?;
    let k = project(s, &normed[1], heads.iter().map(|&h| &*lw.heads[h].key).collect())?;
    let v = project(s, &normed[2], heads.iter().map(|&h| &*lw.heads[h].value).collect())?;
    let z = attend(s, (q, k, v), (n, width), first, true, pad)?;
    let reads = if read { Some(s.device.download(&z)?) } else { None };
    let wo = s.stacked(&heads.iter().map(|&h| &lw.heads[h].output).collect::<Vec<_>>(), false)?;
    let mut out = s.device.zeros(rows, wo.rows())?;
    s.device.gemm(&mut out, 1.0, &z, Op::N, &wo, Op::T, 0.0, arithmetic)?;
    Ok((out, reads))
}

/// The attention of `n` alike heads (shapes, rotary, scale and mask of `first`) on their stacked
/// queries, keys and values (rows × n · `width`): split into padded whole sequences for one
/// attention call, with `first`'s head norms when `norms`, merged back into the heads' reads side
/// by side (rows × n · `width`).
fn attend(s: &mut DeviceState, (q, k, v): (Tensor, Tensor, Tensor), (n, width): (usize, usize), first: &crate::graph::HeadWeights, norms: bool, pad: &Padding) -> Result<Tensor, GpuError> {
    let arithmetic = s.arithmetic();
    let padded_rows = pad.sequences * pad.longest;
    let to_padded = |s: &DeviceState, x: Tensor| -> Result<Tensor, GpuError> {
        match pad.places {
            None => Ok(x),
            Some(places) => {
                let mut out = s.device.zeros(padded_rows, x.cols())?;
                s.device.scatter_rows(&mut out, places, &x, false)?;
                Ok(out)
            }
        }
    };
    let (q, k, v) = (to_padded(s, q)?, to_padded(s, k)?, to_padded(s, v)?);
    // Head-major whole sequences: row (b·n + h)·L + l is head h of sequence b at position l.
    let split = |s: &DeviceState, x: &Tensor| s.device.split_heads(x, 0, n, width, pad.sequences, None, false);
    let (mut qs, mut ks, vs) = (split(s, &q)?, split(s, &k)?, split(s, &v)?);
    for (x, norm) in [(&mut qs, &first.query_norm), (&mut ks, &first.key_norm)] {
        if let Some((g, epsilon)) = norm.as_ref().filter(|_| norms) {
            let unit = s.device.rms_norm(x, *epsilon)?;
            let g = s.ensure(row(g))?;
            s.device.scale_columns(x, &unit, s.get(g)?, false)?;
        }
    }
    if let Some(r) = first.rotary {
        let positions: Vec<u32> = (0..pad.sequences * n).flat_map(|_| 0..pad.longest as u32).collect();
        let key = s.rotations(r, &positions)?;
        let (cos, sin) = s.rotations.get(&key).ok_or_else(|| GpuError::DriverCallFailed { reason: "rotation tables went missing".into() })?;
        qs = s.device.rotate(&qs, cos, sin, r.half_split, false)?;
        ks = s.device.rotate(&ks, cos, sin, r.half_split, false)?;
    }
    let segments: Vec<Segment> = (0..pad.sequences * n).map(|b| Segment { rows: b * pad.longest..(b + 1) * pad.longest, first: 0, before: Vec::new() }).collect();
    let zs = forward_segments(&s.device, (&qs, &ks, &vs), &segments, first.scale, first.causal, arithmetic)?;
    let mut z = s.device.zeros(padded_rows, n * width)?;
    s.device.merge_heads(&zs, &mut z, 0, n, pad.sequences, None, false)?;
    match pad.places {
        None => Ok(z),
        Some(places) => s.device.gather_rows(&z, places),
    }
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
                Some(u) => Some((f32::upload(d, u.select(ndarray::Axis(0), p).view())?, d.upload(row(&mlp.up_bias.select(ndarray::Axis(0), p)))?)),
                None => None,
            };
            (None, Some((f32::upload(d, mlp.gate.select(ndarray::Axis(0), p).view())?, f32::upload(d, mlp.out.select(ndarray::Axis(1), p).view())?, d.upload(row(&mlp.bias.select(ndarray::Axis(0), p)))?, up)))
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

/// Transcoder features `features` of `layer`'s MLP on its normed inputs `x_hat` (`graph::Features`):
/// `relu(x̂ Gᵀ + c) U`, the features' encoder rows `G`, biases `c` and decoder rows `U` gathered on
/// the host and uploaded for this run.
fn features_write(s: &mut DeviceState, weights: &Weights, layer: usize, features: &[usize], x_hat: &Tensor) -> Result<Tensor, String> {
    let e = |e: GpuError| e.to_string();
    let t = weights.transcoders.get(&layer).ok_or_else(|| format!("layer {layer} has no transcoder"))?;
    let (rows, width, arithmetic) = (x_hat.rows(), weights.width(), s.arithmetic());
    let d = &s.device;
    let mut out = d.zeros(rows, width).map_err(e)?;
    if features.is_empty() {
        return Ok(out);
    }
    let (g, c, u) = t.stacked(features, width)?;
    let (g, c, u) = (d.upload(g.view()).map_err(e)?, d.upload(row(&c)).map_err(e)?, d.upload(u.view()).map_err(e)?);
    let mut h = d.zeros(rows, features.len()).map_err(e)?;
    d.gemm(&mut h, 1.0, x_hat, Op::N, &g, Op::T, 0.0, arithmetic).map_err(e)?;
    d.add_row(&mut h, 1.0, &c).map_err(e)?;
    let codes = d.upload_indices(&vec![law_of(crate::operator_program::Law::Relu).code(); features.len()]).map_err(e)?;
    let active = d.law_values(&h, &codes, gelu_tanh_constant()).map_err(e)?;
    d.gemm(&mut out, 1.0, &active, Op::N, &u, Op::N, 0.0, arithmetic).map_err(e)?;
    Ok(out)
}

/// A matrix `W` of a VPD view: a host matrix (kept resident) or a device tensor (stacked head maps).
#[derive(Clone, Copy)]
enum Matrix<'a> {
    Host(&'a Stored),
    Device(&'a Tensor),
}

/// Which matrix of a layer a reference product multiplies ([`DeviceState::reference_product`]).
const PRODUCT_MLP_GATE: usize = 0;
const PRODUCT_MLP_OUT: usize = 1;
const PRODUCT_ATTENTION_OUT: usize = 2;
/// The query, key and value maps: `PRODUCT_ATTENTION_IN + m` for map `m`.
const PRODUCT_ATTENTION_IN: usize = 3;

/// A remainder unit's write: the full write `full` less its named parts' `named` (subcomponents
/// `picked` of a matrix with `count` of them), as [`sliced`] with `rest` computes it.
fn less_named(s: &DeviceState, mut full: Tensor, named: Tensor, picked: &[usize], count: usize) -> Result<Tensor, GpuError> {
    if picked.iter().any(|&i| i <= count) {
        s.device.axpy(&mut full, -1.0, &named)?;
    }
    Ok(full)
}

/// `graph::sliced_parts` on the device: subcomponents `picked` of `w` (out × in) with factors `U`
/// (subcomponents × out) and `V` (in × subcomponents) applied to the rows of `x` (rows × in), index
/// `U`'s row count being the remainder `W − Σ U Vᵀ`; with `rest`, `x wᵀ` less the named ones. With
/// a stacked run's `mask`, each row's subcomponents weigh as many as its copy names (zero: not
/// named) and its remainder counts where its copy names it.
fn sliced(s: &mut DeviceState, (u, v): (&Array2<f64>, &Array2<f64>), w: Matrix, picked: &[usize], rest: bool, x: &Tensor, mask: Masked) -> Result<Tensor, GpuError> {
    let count = u.nrows();
    let subs: Vec<u32> = picked.iter().filter(|&&i| i < count).map(|&i| i as u32).collect();
    let remainder = picked.contains(&count);
    let (uk, vk) = (s.ensure(u.view())?, s.ensure(v.view())?);
    // Most subcomponents named (g-int's chunk measurements name all but a chunk): their sum is the
    // summed matrix, kept dense, less the ones left out, cheaper than the named ones' factors.
    let (inputs, outputs) = (v.nrows(), u.ncols());
    let left = left_out(count, &subs, mask).filter(|(ids, _)| inputs * outputs + ids.len() * (inputs + outputs) < subs.len() * (inputs + outputs));
    if left.is_some() {
        s.summed(uk, vk)?;
    }
    let wk = match w {
        Matrix::Host(m) => Some(s.ensure(m.view())?),
        Matrix::Device(_) => None,
    };
    let s = &*s;
    let (d, arithmetic) = (&s.device, s.arithmetic());
    let (uf, vf) = (s.get(uk)?, s.get(vk)?);
    let wt = match (w, wk) {
        (Matrix::Device(t), _) => t,
        (Matrix::Host(_), Some(k)) => s.get(k)?,
        (Matrix::Host(_), None) => return Err(GpuError::DriverCallFailed { reason: "a VPD matrix went missing".into() }),
    };
    let mut named = d.zeros(x.rows(), u.ncols())?;
    if let Some((ids, weights)) = left {
        let sum = s.summed.get(&(uk, vk)).ok_or_else(|| GpuError::DriverCallFailed { reason: "a summed VPD matrix went missing".into() })?;
        d.gemm(&mut named, 1.0, x, Op::N, sum, Op::T, 0.0, arithmetic)?;
        if !ids.is_empty() {
            let n = ids.len();
            let ids = d.upload_indices(&ids)?;
            let (vs, us) = (d.gather_columns(vf, &ids)?, d.gather_rows(uf, &ids)?);
            let mut xv = d.zeros(x.rows(), n)?;
            d.gemm(&mut xv, 1.0, x, Op::N, &vs, Op::N, 0.0, arithmetic)?;
            if let (Some(weights), Some((_, copy_of_row, _))) = (weights, mask) {
                let per_row = d.gather_rows(&d.upload(weights.view())?, copy_of_row)?;
                let mut weighed = d.zeros(x.rows(), n)?;
                d.hadamard(&mut weighed, &xv, &per_row, false)?;
                xv = weighed;
            }
            d.gemm(&mut named, -1.0, &xv, Op::N, &us, Op::N, 1.0, arithmetic)?;
        }
    } else if !subs.is_empty() {
        let ids = d.upload_indices(&subs)?;
        let (vs, us) = (d.gather_columns(vf, &ids)?, d.gather_rows(uf, &ids)?);
        let mut xv = d.zeros(x.rows(), subs.len())?;
        d.gemm(&mut xv, 1.0, x, Op::N, &vs, Op::N, 0.0, arithmetic)?;
        if let Some((m, copy_of_row, _)) = mask {
            if m.counts.ncols() != subs.len() {
                return Err(GpuError::DriverCallFailed { reason: format!("a copy mask of {} subcomponents for {}", m.counts.ncols(), subs.len()) });
            }
            let counts = d.gather_rows(&d.upload(m.counts.view())?, copy_of_row)?;
            let mut weighed = d.zeros(x.rows(), subs.len())?;
            d.hadamard(&mut weighed, &xv, &counts, false)?;
            xv = weighed;
        }
        d.gemm(&mut named, 1.0, &xv, Op::N, &us, Op::N, 0.0, arithmetic)?;
    }
    if remainder {
        let mut xv = d.zeros(x.rows(), count)?;
        d.gemm(&mut xv, 1.0, x, Op::N, vf, Op::N, 0.0, arithmetic)?;
        match mask {
            None => {
                d.gemm(&mut named, 1.0, x, Op::N, wt, Op::T, 1.0, arithmetic)?;
                d.gemm(&mut named, -1.0, &xv, Op::N, uf, Op::N, 1.0, arithmetic)?;
            }
            Some((m, _, rows)) => {
                // The remainder on every row, then zero on the rows of copies that do not name it.
                let mut left = d.zeros(x.rows(), u.ncols())?;
                d.gemm(&mut left, 1.0, x, Op::N, wt, Op::T, 0.0, arithmetic)?;
                d.gemm(&mut left, -1.0, &xv, Op::N, uf, Op::N, 1.0, arithmetic)?;
                let without: Vec<u32> = m.remainder.iter().enumerate().filter(|(_, named)| !**named).flat_map(|(j, _)| (j * rows..(j + 1) * rows).map(|r| r as u32)).collect();
                if !without.is_empty() {
                    scale_rows(d, &mut left, &d.upload_indices(&without)?, 0.0)?;
                }
                d.axpy(&mut named, 1.0, &left)?;
            }
        }
    }
    if !rest {
        return Ok(named);
    }
    let mut out = d.zeros(x.rows(), u.ncols())?;
    d.gemm(&mut out, 1.0, x, Op::N, wt, Op::T, 0.0, arithmetic)?;
    if !subs.is_empty() || remainder {
        d.axpy(&mut out, -1.0, &named)?;
    }
    Ok(out)
}

/// The subcomponents (of `count`) a selection `subs` leaves out, with a stacked run's per-copy
/// weights of each (copy × left out: one less the copy's count, from `mask`), when the named sum is
/// the summed matrix less these: `subs` distinct and every count 0 or 1. Without a mask every copy
/// is the one selection.
fn left_out(count: usize, subs: &[u32], mask: Masked) -> Option<(Vec<u32>, Option<Array2<f64>>)> {
    let mut seen = vec![false; count];
    for &i in subs {
        let slot = seen.get_mut(i as usize)?;
        if *slot {
            return None;
        }
        *slot = true;
    }
    let Some((m, ..)) = mask else {
        return Some(((0..count as u32).filter(|&i| !seen[i as usize]).collect(), None));
    };
    if m.counts.ncols() != subs.len() || m.counts.iter().any(|&c| c != 0.0 && c != 1.0) {
        return None;
    }
    // Per subcomponent its column in the mask; a subcomponent every copy names is no one's left out.
    let mut column = vec![None; count];
    for (k, &i) in subs.iter().enumerate() {
        column[i as usize] = Some(k);
    }
    let everyone = |k: usize| m.counts.column(k).iter().all(|&c| c == 1.0);
    let ids: Vec<u32> = (0..count).filter(|&i| column[i].is_none_or(|k| !everyone(k))).map(|i| i as u32).collect();
    let weights = Array2::from_shape_fn((m.counts.nrows(), ids.len()), |(j, c)| column[ids[c] as usize].map_or(1.0, |k| 1.0 - m.counts[[j, k]]));
    Some((ids, Some(weights)))
}

/// Whether a unit whose hidden reads are `hidden` reads unit `w`'s writes into its site's hidden
/// stream (a VPD view's queries, keys, values or MLP pre-activation).
fn reads_hidden(hidden: &Incoming, w: usize) -> bool {
    match hidden {
        Incoming::AllBut(cut) => !cut.contains(&Writer::Unit(w)),
        Incoming::Only(kept) => kept.contains(&Writer::Unit(w)),
    }
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
