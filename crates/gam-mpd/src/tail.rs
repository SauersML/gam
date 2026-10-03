//! The frozen tail of a decoder language model, as a [`Head`] (#2951).
//!
//! A decomposition of a window of blocks leaves the blocks after it, and the readout, frozen
//! (`crate::masked`, "Heads"). This is that tail for a Hugging Face Qwen2 or Llama checkpoint
//! (pre-norm RMS blocks, rotary attention with grouped keys and values, optional q/k/v
//! biases, a SwiGLU MLP, a final RMS norm and the (tied) unembedding), computed in float64 with
//! exactly the program's arithmetic (`crate::import::hugging_face_language_model`: the same
//! rotary turns, `x/√(mean x² + ε)`, `t/(1+e^{-t})`), so a keep/refuse decision through it is the
//! decision through the whole model. Its weights are never held in float64 at once: they stay in
//! the memory-mapped checkpoint (file pages, in their stored type) and each map is widened for its
//! product, with the widened matrices kept up to a byte budget.
//!
//! The pullback is the exact reverse pass: each block is run again from its saved input (one
//! block's activations live at a time), then its maps, attention, rotary turns, norms and the
//! SwiGLU are reversed by hand.
//!
//! # Fixed rows
//!
//! A masked fit scored on a behaviour's rows changes nothing before them: every row the target
//! does not score that precedes all of its sequence's scored rows has the same input on every call
//! ([`DecoderTail::hold_fixed`]). Those rows' keys and values at every block are then computed once
//! and kept (while the same rows come again), only the scored rows run through the blocks, and the
//! pullback is the cotangent at the scored rows' inputs with the fixed rows held fixed (zero at
//! theirs). That is exactly the derivative a mask on a scored row has: it cannot reach an earlier
//! row.

use super::masked::Head;
use super::operator_program::{FamilyInputs, Rotary};
use super::safetensors::SafetensorsFile;
use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, Axis, s};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The blocks `blocks` (to the last) and the readout of a checkpoint (module note).
pub struct DecoderTail {
    file: SafetensorsFile,
    blocks: Range<usize>,
    d: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    d_mlp: usize,
    vocab: usize,
    epsilon: f64,
    rotary: Rotary,
    readout: String,
    cache: Mutex<(HashMap<String, Arc<Array2<f64>>>, usize)>,
    budget: usize,
    /// Whether rows before every scored row are held fixed (module note, "Fixed rows"), and the
    /// fixed rows' inputs with their keys and values.
    hold: AtomicBool,
    fixed: Mutex<Option<(Array2<f64>, Rows, Arc<Prefix>)>>,
}

/// Rows' sequences and positions.
#[derive(Clone, Debug, PartialEq)]
struct Rows {
    sequence: Vec<u32>,
    position: Vec<u32>,
}

impl Rows {
    fn of(inputs: &FamilyInputs, select: &[usize]) -> Result<Self, String> {
        let layout = inputs.layout.as_ref().ok_or("the tail needs a sequence layout")?;
        Ok(Self { sequence: select.iter().map(|r| layout.sequence[*r]).collect(), position: select.iter().map(|r| layout.position[*r]).collect() })
    }

    /// Each sequence and its rows, in row order.
    fn sequences(&self) -> Vec<(u32, Vec<usize>)> {
        let mut by: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (r, s) in self.sequence.iter().enumerate() {
            by.entry(*s).or_default().push(r);
        }
        by.into_iter().collect()
    }
}

/// Per block (from the tail's first) and sequence: the fixed rows' positions, keys and values.
struct Prefix {
    first: usize,
    blocks: Vec<BTreeMap<u32, (Vec<u32>, Array2<f64>, Array2<f64>)>>,
}

impl Prefix {
    fn get(&self, l: usize, sequence: u32) -> Option<&(Vec<u32>, Array2<f64>, Array2<f64>)> {
        self.blocks.get(l - self.first)?.get(&sequence)
    }
}

/// One sequence's attention at a block: its computed rows, how many fixed keys precede theirs,
/// and per head the probabilities over all its keys with those keys and values.
struct Attention {
    members: Vec<usize>,
    fixed: usize,
    per_head: Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>,
}

/// `top` above `rest` (rows), or `rest` alone.
fn stack(top: Option<Array2<f64>>, rest: Array2<f64>) -> Array2<f64> {
    match top {
        Some(top) => ndarray::concatenate(Axis(0), &[top.view(), rest.view()]).expect("equal widths"),
        None => rest,
    }
}

/// One block's activations, for its reverse pass.
struct Saved {
    x: Array2<f64>,
    scale1: Array1<f64>,
    q: Array2<f64>,
    k: Array2<f64>,
    v: Array2<f64>,
    attention: Vec<Attention>,
    mid: Array2<f64>,
    scale2: Array1<f64>,
    gate: Array2<f64>,
    up: Array2<f64>,
}

fn sigmoid(t: f64) -> f64 {
    1.0 / (1.0 + (-t).exp())
}

/// `(x/√(mean x² + ε)) ⊙ g` per row, and each row's scale `1/√(mean x² + ε)`.
fn rms(x: &Array2<f64>, gain: &Array1<f64>, epsilon: f64) -> (Array2<f64>, Array1<f64>) {
    let scales: Array1<f64> = x.outer_iter().map(|r| super::operator_program::rms_scale(r, epsilon)).collect();
    let mut out = x.clone();
    for (mut row, s) in out.outer_iter_mut().zip(scales.iter()) {
        row.iter_mut().zip(gain.iter()).for_each(|(v, g)| *v = *v * s * g);
    }
    (out, scales)
}

/// The cotangent at `x` of `dy` at `rms(x) ⊙ g`.
fn rms_back(x: &Array2<f64>, scales: &Array1<f64>, gain: &Array1<f64>, dy: &Array2<f64>) -> Array2<f64> {
    let n = x.ncols() as f64;
    let mut dx = Array2::<f64>::zeros(x.dim());
    for r in 0..x.nrows() {
        let s = scales[r];
        let a: Array1<f64> = dy.row(r).iter().zip(gain.iter()).map(|(d, g)| d * g).collect();
        let dot: f64 = a.iter().zip(x.row(r).iter()).map(|(a, x)| a * x).sum();
        for c in 0..x.ncols() {
            dx[[r, c]] = s * a[c] - x[[r, c]] * s * s * s * dot / n;
        }
    }
    dx
}

impl DecoderTail {
    /// The tail from block `first` of the checkpoint in `dir` (`config.json`, `model.safetensors`),
    /// keeping widened maps up to `budget` bytes.
    pub fn new(dir: &Path, first: usize, budget: usize) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
        let hf: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let integer = |key: &str| hf[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("config.json: {key}"));
        let kind = hf["model_type"].as_str().ok_or("config.json: model_type")?;
        if !matches!(kind, "qwen2" | "llama") || hf["hidden_act"].as_str() != Some("silu") {
            return Err(format!("unsupported model {kind} ({:?})", hf["hidden_act"]));
        }
        let (d, heads) = (integer("hidden_size")?, integer("num_attention_heads")?);
        let hd = hf["head_dim"].as_u64().map_or(d / heads.max(1), |v| v as usize);
        let theta = hf["rope_theta"].as_f64().ok_or("config.json: rope_theta")?;
        if theta.fract() != 0.0 || theta <= 0.0 || theta > f64::from(u32::MAX) {
            return Err(format!("rope_theta {theta} is not a positive integer"));
        }
        let layers = integer("num_hidden_layers")?;
        if first > layers {
            return Err(format!("first block {first} of {layers}"));
        }
        let file = SafetensorsFile::open(&dir.join("model.safetensors")).map_err(|e| e.to_string())?;
        let tied = hf["tie_word_embeddings"].as_bool().unwrap_or(false);
        let readout = if tied { "model.embed_tokens.weight" } else { "lm_head.weight" }.to_string();
        let vocab = file.tensors().get(&readout).map(|e| e.shape[0]).ok_or_else(|| format!("{readout}: missing"))?;
        Ok(Self {
            file,
            blocks: first..layers,
            d,
            heads,
            kv_heads: integer("num_key_value_heads")?,
            hd,
            d_mlp: integer("intermediate_size")?,
            vocab,
            epsilon: hf["rms_norm_eps"].as_f64().ok_or("config.json: rms_norm_eps")?,
            rotary: Rotary { base: theta as u32, dims: hd as u32, half_split: true },
            readout,
            cache: Mutex::new((HashMap::new(), 0)),
            budget,
            hold: AtomicBool::new(false),
            fixed: Mutex::new(None),
        })
    }

    /// Hold the rows before every scored row fixed (module note, "Fixed rows"), or not.
    pub fn hold_fixed(&self, hold: bool) {
        self.hold.store(hold, Ordering::Relaxed);
    }

    /// The stored matrix `name` (`rows × cols`), widened exactly.
    fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Arc<Array2<f64>>, String> {
        if let Some(m) = self.cache.lock().map_err(|_| "tail cache poisoned")?.0.get(name) {
            return Ok(m.clone());
        }
        let mut governed = self.file.matrix(MemoryGovernor::global(), name, rows, cols).map_err(|e| e.to_string())?;
        let m = Arc::new(std::mem::take(&mut *governed));
        let bytes = rows * cols * 8;
        let mut cache = self.cache.lock().map_err(|_| "tail cache poisoned")?;
        if cache.1 + bytes <= self.budget {
            cache.1 += bytes;
            cache.0.insert(name.to_string(), m.clone());
        }
        Ok(m)
    }

    fn vector(&self, name: &str, len: usize) -> Result<Array1<f64>, String> {
        self.file.vector(name, len).map_err(|e| e.to_string())
    }

    /// The bias `name` when the checkpoint has one.
    fn bias(&self, name: &str, len: usize) -> Result<Option<Array1<f64>>, String> {
        if self.file.tensors().contains_key(name) { self.vector(name, len).map(Some) } else { Ok(None) }
    }

    fn name(l: usize, part: &str) -> String {
        format!("model.layers.{l}.{part}")
    }

    /// `x Wᵀ + b` for the stored map `name` (`out × in`).
    fn linear(&self, x: &Array2<f64>, l: usize, part: &str, out: usize) -> Result<Array2<f64>, String> {
        let w = self.matrix(&Self::name(l, &format!("{part}.weight")), out, x.ncols())?;
        let mut y = fast_abt(x, w.as_ref());
        if let Some(b) = self.bias(&Self::name(l, &format!("{part}.bias")), out)? {
            y += &b;
        }
        Ok(y)
    }

    /// `dy W` for the stored map `name` (`out × in`).
    fn linear_back(&self, dy: &Array2<f64>, l: usize, part: &str, input: usize) -> Result<Array2<f64>, String> {
        let w = self.matrix(&Self::name(l, &format!("{part}.weight")), dy.ncols(), input)?;
        Ok(fast_ab(dy, w.as_ref()))
    }

    /// Turn every `heads` head of `x` (rows × heads·hd) to its row's position (`sign` −1 undoes it).
    fn turn(&self, x: &mut Array2<f64>, heads: usize, positions: &[u32], sign: f64) {
        let pairs = self.rotary.pairs();
        for (r, mut row) in x.outer_iter_mut().enumerate() {
            for h in 0..heads {
                for (plane, (a, b)) in pairs.iter().enumerate() {
                    let (c, s) = self.rotary.turn(plane, positions[r]);
                    let s = sign * s;
                    let (i, j) = (h * self.hd + a, h * self.hd + b);
                    let (x0, y0) = (row[i], row[j]);
                    row[i] = c * x0 - s * y0;
                    row[j] = s * x0 + c * y0;
                }
            }
        }
    }

    /// Block `l` on the rows `x` (their sequences and positions in `rows`), attending also to the
    /// fixed rows' keys and values in `prefix`, and what its reverse pass needs.
    fn block(&self, l: usize, x: &Array2<f64>, rows: &Rows, prefix: Option<&Prefix>) -> Result<(Array2<f64>, Saved), String> {
        let (h, scale1) = rms(x, &self.vector(&Self::name(l, "input_layernorm.weight"), self.d)?, self.epsilon);
        let mut q = self.linear(&h, l, "self_attn.q_proj", self.heads * self.hd)?;
        let mut k = self.linear(&h, l, "self_attn.k_proj", self.kv_heads * self.hd)?;
        let v = self.linear(&h, l, "self_attn.v_proj", self.kv_heads * self.hd)?;
        self.turn(&mut q, self.heads, &rows.position, 1.0);
        self.turn(&mut k, self.kv_heads, &rows.position, 1.0);
        let scale = 1.0 / (self.hd as f64).sqrt();
        let group = self.heads / self.kv_heads.max(1);
        let mut attended = Array2::<f64>::zeros((x.nrows(), self.heads * self.hd));
        let mut attention = Vec::new();
        for (sequence, members) in rows.sequences() {
            // Keys: the sequence's fixed rows, then its computed rows.
            let fixed = prefix.and_then(|p| p.get(l, sequence));
            let mut key_positions: Vec<u32> = fixed.map_or_else(Vec::new, |f| f.0.clone());
            key_positions.extend(members.iter().map(|r| rows.position[*r]));
            let n_fixed = fixed.map_or(0, |f| f.0.len());
            let mut per_head = Vec::new();
            for hh in 0..self.heads {
                let g = hh / group.max(1);
                let (qs, ks) = (hh * self.hd..(hh + 1) * self.hd, g * self.hd..(g + 1) * self.hd);
                let qh = q.select(Axis(0), &members).slice(s![.., qs.clone()]).to_owned();
                let kh = stack(fixed.map(|f| f.1.slice(s![.., ks.clone()]).to_owned()), k.select(Axis(0), &members).slice(s![.., ks.clone()]).to_owned());
                let vh = stack(fixed.map(|f| f.2.slice(s![.., ks.clone()]).to_owned()), v.select(Axis(0), &members).slice(s![.., ks.clone()]).to_owned());
                let mut p = fast_abt(&qh, &kh) * scale;
                for (i, &ri) in members.iter().enumerate() {
                    let at = rows.position[ri];
                    let mut row = p.row_mut(i);
                    let m = (0..key_positions.len()).filter(|j| key_positions[*j] <= at).map(|j| row[j]).fold(f64::NEG_INFINITY, f64::max);
                    let mut total = 0.0;
                    for j in 0..key_positions.len() {
                        row[j] = if key_positions[j] <= at { (row[j] - m).exp() } else { 0.0 };
                        total += row[j];
                    }
                    row.mapv_inplace(|e| e / total);
                }
                let o = fast_ab(&p, &vh);
                for (i, &ri) in members.iter().enumerate() {
                    attended.slice_mut(s![ri, qs.clone()]).assign(&o.row(i));
                }
                per_head.push((p, kh, vh));
            }
            attention.push(Attention { members, fixed: n_fixed, per_head });
        }
        let mid = x + &self.linear(&attended, l, "self_attn.o_proj", self.d)?;
        let (h2, scale2) = rms(&mid, &self.vector(&Self::name(l, "post_attention_layernorm.weight"), self.d)?, self.epsilon);
        let gate = self.linear(&h2, l, "mlp.gate_proj", self.d_mlp)?;
        let up = self.linear(&h2, l, "mlp.up_proj", self.d_mlp)?;
        let active = ndarray::Zip::from(&gate).and(&up).map_collect(|g, u| g * sigmoid(*g) * u);
        let out = &mid + &self.linear(&active, l, "mlp.down_proj", self.d)?;
        Ok((out, Saved { x: x.clone(), scale1, q, k, v, attention, mid, scale2, gate, up }))
    }

    /// The cotangent at block `l`'s computed rows of `dy` at their outputs (fixed rows held fixed).
    fn block_back(&self, l: usize, saved: &Saved, dy: &Array2<f64>, rows: &Rows) -> Result<Array2<f64>, String> {
        // MLP: out = mid + down(silu(gate) ⊙ up).
        let da = self.linear_back(dy, l, "mlp.down_proj", self.d_mlp)?;
        let dgate = ndarray::Zip::from(&da).and(&saved.gate).and(&saved.up).map_collect(|d, g, u| {
            let sg = sigmoid(*g);
            d * u * sg * (1.0 + g * (1.0 - sg))
        });
        let dup = ndarray::Zip::from(&da).and(&saved.gate).map_collect(|d, g| d * g * sigmoid(*g));
        let dh2 = self.linear_back(&dgate, l, "mlp.gate_proj", self.d)? + self.linear_back(&dup, l, "mlp.up_proj", self.d)?;
        let gain2 = self.vector(&Self::name(l, "post_attention_layernorm.weight"), self.d)?;
        let dmid = dy + &rms_back(&saved.mid, &saved.scale2, &gain2, &dh2);
        // Attention: mid = x + o(attended).
        let dattended = self.linear_back(&dmid, l, "self_attn.o_proj", self.heads * self.hd)?;
        let scale = 1.0 / (self.hd as f64).sqrt();
        let group = self.heads / self.kv_heads.max(1);
        let mut dq = Array2::<f64>::zeros(saved.q.dim());
        let mut dk = Array2::<f64>::zeros(saved.k.dim());
        let mut dv = Array2::<f64>::zeros(saved.v.dim());
        for attention in &saved.attention {
            let members = &attention.members;
            for (hh, (p, kh, vh)) in attention.per_head.iter().enumerate() {
                let g = hh / group.max(1);
                let (qs, ks) = (hh * self.hd..(hh + 1) * self.hd, g * self.hd..(g + 1) * self.hd);
                let qh = saved.q.select(Axis(0), members).slice(s![.., qs.clone()]).to_owned();
                let doh = dattended.select(Axis(0), members).slice(s![.., qs.clone()]).to_owned();
                let dvh = fast_ab(&p.t().to_owned(), &doh);
                let dp = fast_abt(&doh, vh);
                let mut ds = Array2::<f64>::zeros(p.dim());
                for i in 0..p.nrows() {
                    let dot: f64 = p.row(i).iter().zip(dp.row(i).iter()).map(|(a, b)| a * b).sum();
                    for j in 0..p.ncols() {
                        ds[[i, j]] = p[[i, j]] * (dp[[i, j]] - dot);
                    }
                }
                let dqh = fast_ab(&ds, kh) * scale;
                let dkh = fast_ab(&ds.t().to_owned(), &qh) * scale;
                // The fixed keys' and values' cotangents are dropped: those rows are held fixed.
                for (i, &r) in members.iter().enumerate() {
                    let mut row = dq.slice_mut(s![r, qs.clone()]);
                    row += &dqh.row(i);
                    let mut row = dk.slice_mut(s![r, ks.clone()]);
                    row += &dkh.row(attention.fixed + i);
                    let mut row = dv.slice_mut(s![r, ks.clone()]);
                    row += &dvh.row(attention.fixed + i);
                }
            }
        }
        self.turn(&mut dq, self.heads, &rows.position, -1.0);
        self.turn(&mut dk, self.kv_heads, &rows.position, -1.0);
        let dh = self.linear_back(&dq, l, "self_attn.q_proj", self.d)?
            + self.linear_back(&dk, l, "self_attn.k_proj", self.d)?
            + self.linear_back(&dv, l, "self_attn.v_proj", self.d)?;
        let gain1 = self.vector(&Self::name(l, "input_layernorm.weight"), self.d)?;
        Ok(dmid + rms_back(&saved.x, &saved.scale1, &gain1, &dh))
    }

    /// The fixed rows' keys and values at every block, from their inputs `x`.
    fn prefix(&self, x: &Array2<f64>, rows: &Rows) -> Result<Prefix, String> {
        let mut blocks = Vec::new();
        let mut x = x.clone();
        for l in self.blocks.clone() {
            let (next, saved) = self.block(l, &x, rows, None)?;
            let mut per_sequence = BTreeMap::new();
            for (sequence, members) in rows.sequences() {
                let positions = members.iter().map(|r| rows.position[*r]).collect();
                per_sequence.insert(sequence, (positions, saved.k.select(Axis(0), &members), saved.v.select(Axis(0), &members)));
            }
            blocks.push(per_sequence);
            x = next;
        }
        Ok(Prefix { first: self.blocks.start, blocks })
    }

    /// The rows to run, and the fixed rows' keys and values when rows are held fixed (module
    /// note, "Fixed rows"): a row is fixed when it is unscored and precedes every scored row of its
    /// sequence (a sequence with no scored row is fixed whole).
    fn split(&self, inputs: &FamilyInputs, output: &Array2<f64>, scored: &[bool]) -> Result<(Vec<usize>, Option<Arc<Prefix>>), String> {
        let all: Vec<usize> = (0..output.nrows()).collect();
        if !self.hold.load(Ordering::Relaxed) {
            return Ok((all, None));
        }
        let layout = inputs.layout.as_ref().ok_or("the tail needs a sequence layout")?;
        let mut first_scored: BTreeMap<u32, u32> = BTreeMap::new();
        for r in all.iter().filter(|r| scored[**r]) {
            let entry = first_scored.entry(layout.sequence[*r]).or_insert(u32::MAX);
            *entry = (*entry).min(layout.position[*r]);
        }
        let held = |r: usize| !scored[r] && layout.position[r] < first_scored.get(&layout.sequence[r]).copied().unwrap_or(u32::MAX);
        let (fixed, run): (Vec<usize>, Vec<usize>) = all.into_iter().partition(|r| held(*r));
        if fixed.is_empty() {
            return Ok((run, None));
        }
        let x = output.select(Axis(0), &fixed);
        let rows = Rows::of(inputs, &fixed)?;
        let mut kept = self.fixed.lock().map_err(|_| "tail prefix poisoned")?;
        if let Some((x0, rows0, prefix)) = kept.as_ref()
            && *x0 == x
            && *rows0 == rows
        {
            return Ok((run, Some(prefix.clone())));
        }
        let prefix = Arc::new(self.prefix(&x, &rows)?);
        *kept = Some((x, rows, prefix.clone()));
        Ok((run, Some(prefix)))
    }

    /// The final norm's output at the scored rows of the stream after the last block.
    fn normed(&self, x: &Array2<f64>, rows: &[usize]) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), String> {
        let at = x.select(Axis(0), rows);
        let (h, scales) = rms(&at, &self.vector("model.norm.weight", self.d)?, self.epsilon);
        Ok((at, h, scales))
    }

    /// The readout `h Wᵀ` in vocabulary chunks (the readout is the largest map: never widened whole).
    fn readout(&self, h: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut logits = Array2::<f64>::zeros((h.nrows(), self.vocab));
        for (a, b) in self.chunks() {
            let w = self.file.matrix_rows(&self.readout, a..b, self.d).map_err(|e| e.to_string())?;
            logits.slice_mut(s![.., a..b]).assign(&fast_abt(h, &w));
        }
        Ok(logits)
    }

    fn readout_back(&self, dlogits: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut dh = Array2::<f64>::zeros((dlogits.nrows(), self.d));
        for (a, b) in self.chunks() {
            let w = self.file.matrix_rows(&self.readout, a..b, self.d).map_err(|e| e.to_string())?;
            dh += &fast_ab(&dlogits.slice(s![.., a..b]).to_owned(), &w);
        }
        Ok(dh)
    }

    fn chunks(&self) -> Vec<(usize, usize)> {
        const CHUNK: usize = 1 << 14;
        (0..self.vocab).step_by(CHUNK).map(|a| (a, (a + CHUNK).min(self.vocab))).collect()
    }
}

impl Head for DecoderTail {
    fn logits(&self, inputs: &FamilyInputs, output: &Array2<f64>, rows: &[bool]) -> Result<Array2<f64>, String> {
        let (run, prefix) = self.split(inputs, output, rows)?;
        let at_rows = Rows::of(inputs, &run)?;
        let mut x = output.select(Axis(0), &run);
        for l in self.blocks.clone() {
            x = self.block(l, &x, &at_rows, prefix.as_deref())?.0;
        }
        let scored: Vec<usize> = (0..run.len()).filter(|i| rows[run[*i]]).collect();
        let (_, h, _) = self.normed(&x, &scored)?;
        let at = self.readout(&h)?;
        let mut logits = Array2::<f64>::zeros((output.nrows(), self.vocab));
        for (i, r) in scored.iter().enumerate() {
            logits.row_mut(run[*r]).assign(&at.row(i));
        }
        Ok(logits)
    }

    fn pullback(&self, inputs: &FamilyInputs, output: &Array2<f64>, rows: &[bool], cotangent: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (run, prefix) = self.split(inputs, output, rows)?;
        let at_rows = Rows::of(inputs, &run)?;
        // Forward, keeping only each block's input (each block runs again in its reverse pass).
        let mut inputs_of = Vec::new();
        let mut x = output.select(Axis(0), &run);
        for l in self.blocks.clone() {
            let next = self.block(l, &x, &at_rows, prefix.as_deref())?.0;
            inputs_of.push(std::mem::replace(&mut x, next));
        }
        let scored: Vec<usize> = (0..run.len()).filter(|i| rows[run[*i]]).collect();
        let (at, _, scales) = self.normed(&x, &scored)?;
        let picked: Vec<usize> = scored.iter().map(|i| run[*i]).collect();
        let dh = self.readout_back(&cotangent.select(Axis(0), &picked))?;
        let dat = rms_back(&at, &scales, &self.vector("model.norm.weight", self.d)?, &dh);
        let mut dx = Array2::<f64>::zeros(x.dim());
        for (i, r) in scored.iter().enumerate() {
            dx.row_mut(*r).assign(&dat.row(i));
        }
        for (l, input) in self.blocks.clone().zip(inputs_of).rev() {
            let (_, saved) = self.block(l, &input, &at_rows, prefix.as_deref())?;
            dx = self.block_back(l, &saved, &dx, &at_rows)?;
        }
        let mut out = Array2::<f64>::zeros(output.dim());
        for (i, r) in run.iter().enumerate() {
            out.row_mut(*r).assign(&dx.row(i));
        }
        Ok(out)
    }
}
