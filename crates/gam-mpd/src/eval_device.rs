//! The evaluation side on a device (#2951): `super::counterfactual`'s decoder with its weights
//! held on a device, every program an evaluation compares run there in batches of sequences, and
//! only a few numbers per row brought back.
//!
//! # The decoder
//!
//! [`Resident`] is `counterfactual::Decoder` term for term on `gam_gpu::tensor`: pre-RMS-norm
//! blocks (the gain applied after the norm), rotate-half rotary causal attention per head, a
//! tanh-GELU MLP, a final RMS norm and the tied unembedding. A program's sites run as
//! `counterfactual::Program::apply` runs them ([`Maps`]): its input changes, the native map or an
//! explanation's units (`Σ_{c on} u_c (v_c · x)`, as `z = x Vᵀ`, `z ⊙ m`, `(z ⊙ m) U`, the units on
//! chosen by its [`Rule`] from the program's own input there), its output changes, every state a
//! later program reads recorded. A site's input and output are held in column parts, a head each
//! where the attention reads or writes them (the queries, keys and values per head, the output
//! projection's input per head), so a head's columns never need copying out.
//!
//! # Sharing the clean run
//!
//! A program changed from row `s` and layer `l` on (an intervention's first row and layer, or the
//! first replaced site's) equals its clean run before both: the rows before `s` are unchanged under
//! causal attention, and so is every layer before `l`. A [`Run`] starts from the clean run's
//! residual entering layer `l` at rows `s..` of every sequence, and its attention reads the clean
//! run's rotated keys and values at rows `..s` ([`Clean`]); only rows `s..` of layers `l..` are
//! computed. Every sequence of a batch shares `s`, so the queries of a batch are equal blocks.
//!
//! # Precision
//!
//! On a float64 device every operation is float64 and the products run in the resident's
//! [`Arithmetic`]; the numbers are the CPU decoder's up to the summation order of the reductions.
//! The Apple GPU runs f32 throughout (`gam_gpu::tensor`).

use super::counterfactual::{Action, Decoder, InputChange, KINDS, OutputChange, Rows, Scored, Selection, Spec, site_index};
use gam_gpu::tensor::{Arithmetic, Device, Indices, Op, PointwiseLaw, Tensor};
use ndarray::{Array1, Array2, Axis, s};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub(crate) fn error(e: impl std::fmt::Display) -> String {
    format!("eval device: {e}")
}

/// The largest logits tile the head forms at once, in bytes.
const TILE_BYTES: usize = 1 << 30;

/// The constant of the tanh GELU (`llama_simple_mlp::gelu_tanh`'s scale).
fn gelu_tanh_constant() -> f64 {
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2
}

/// A site's native map in blocks: `blocks[i][j]` maps input part `j` to output part `i`
/// (`out_i = Σ_j in_j W_ijᵀ`).
struct Map {
    blocks: Vec<Vec<Tensor>>,
}

struct Layer {
    rms1: Tensor,
    rms2: Tensor,
    maps: Vec<Map>,
}

/// The decoder's weights on a device (module note).
pub struct Resident {
    device: Device,
    wte: Tensor,
    final_gain: Tensor,
    layers: Vec<Layer>,
    heads: usize,
    head_dim: usize,
    d_model: usize,
    d_mlp: usize,
    vocab: usize,
    eps: f64,
    /// Per position and plane, the rotation's `(cos, sin)` (`counterfactual::Decoder::rotate`).
    cos: Array2<f64>,
    sin: Array2<f64>,
    gelu: Indices,
}

/// The column widths of a site's input and output parts.
fn parts(kind: usize, heads: usize, head_dim: usize, d: usize, mlp: usize) -> (Vec<usize>, Vec<usize>) {
    match kind {
        0..=2 => (vec![d], vec![head_dim; heads]),
        3 => (vec![head_dim; heads], vec![d]),
        4 => (vec![d], vec![mlp]),
        _ => (vec![mlp], vec![d]),
    }
}

fn offsets(widths: &[usize]) -> Vec<usize> {
    std::iter::once(0).chain(widths.iter().scan(0, |a, w| { *a += w; Some(*a) })).collect()
}

/// A batch's sequences (module note): block `b` is rows `start..length` of passage `passages[b]`.
pub struct Batch {
    pub passages: Vec<usize>,
    pub length: usize,
    pub start: usize,
    ids: Indices,
    cos: Tensor,
    sin: Tensor,
}

impl Batch {
    /// Rows per block.
    #[must_use]
    pub fn computed(&self) -> usize {
        self.length - self.start
    }

    #[must_use]
    pub fn blocks(&self) -> usize {
        self.passages.len()
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.blocks() * self.computed()
    }

    /// The tensor row of block `b`'s row `r` (`r ≥ start`).
    #[must_use]
    pub fn row(&self, b: usize, r: usize) -> usize {
        b * self.computed() + (r - self.start)
    }
}

/// A program's clean run on one passage (module note): the residual entering every layer and
/// after the last, every layer's rotated keys and values per head, and the units its rule ran per
/// row.
pub struct Clean {
    pub residuals: Vec<Tensor>,
    keys: Vec<Vec<Tensor>>,
    values: Vec<Vec<Tensor>>,
    pub units: Vec<f64>,
}

/// A program's states on a donor passage, on the device: per `(site, row, output?)` its parts.
#[derive(Default)]
pub struct Donor {
    pub states: BTreeMap<(usize, usize, bool), Vec<Tensor>>,
}

/// An explanation's units at one site on the device: `V` per input part (units × width) and `U`
/// per output part (units × width).
pub struct Units {
    v: Vec<Tensor>,
    u: Vec<Tensor>,
    pub columns: usize,
}

/// Where a rule chooses: a batch's computed rows at decoder site `site`, the program's own input
/// there (its parts).
pub struct At<'a> {
    pub batch: &'a Batch,
    pub site: usize,
    pub input: &'a [Tensor],
}

/// A selection rule run on the device: each computed row's units on (rows × the site's units, 1 or
/// 0) from the program's own input ([`At`]).
pub trait Rule: Sync {
    fn mask(&self, resident: &Resident, at: &At<'_>) -> Result<Tensor, String>;
}

/// What a program runs at its sites (`counterfactual::Maps`).
pub enum Maps<'a> {
    Native,
    /// Per decoder site its units (a site without runs native) and the rule choosing them.
    Units { units: Vec<Option<&'a Units>>, rule: &'a dyn Rule },
}

/// One program's forward over a batch (module note).
pub struct Run<'a> {
    pub maps: Maps<'a>,
    /// Per block: its actions, its donor states, the rows kept after every layer, and the states
    /// to record.
    pub actions: Vec<&'a [Action]>,
    pub donors: Vec<Option<&'a Donor>>,
    pub interface: Vec<&'a [usize]>,
    pub record: Vec<Vec<(usize, usize, bool)>>,
    /// The first layer computed; with `clean` (per block, this program's clean run on its
    /// passage), earlier layers and rows before the batch's start are the clean run's.
    pub from_layer: usize,
    pub clean: Option<Vec<&'a Clean>>,
    /// Keep the run as its blocks' clean runs (only from row and layer 0).
    pub keep: bool,
    /// Every product's arithmetic.
    pub arithmetic: Arithmetic,
}

impl<'a> Run<'a> {
    /// A clean run of `maps` on `blocks` blocks, keeping it, recording `record` per block.
    #[must_use]
    pub fn clean(maps: Maps<'a>, blocks: usize, record: Vec<Vec<(usize, usize, bool)>>, arithmetic: Arithmetic) -> Self {
        Self { maps, actions: vec![&[]; blocks], donors: vec![None; blocks], interface: vec![&[]; blocks], record, from_layer: 0, clean: None, keep: true, arithmetic }
    }
}

/// What a forward leaves: the residual after the last layer at the computed rows, per block and
/// layer the residual at its interface rows (interface rows × d), the recorded states, the units
/// the rule ran per block (computed rows only), and the clean runs when kept.
pub struct Ran {
    pub residual: Tensor,
    pub interface: Vec<Vec<Tensor>>,
    pub recorded: Vec<Donor>,
    pub units: Vec<Vec<f64>>,
    pub kept: Option<Vec<Clean>>,
}

fn upload_row(d: &Device, values: Vec<f64>) -> Result<Tensor, String> {
    let n = values.len();
    d.upload_vec(1, n, values).map_err(error)
}

impl Resident {
    /// `decoder`'s weights on `device`, rotations for `context` positions.
    pub fn new(device: &Device, decoder: &Decoder, context: usize) -> Result<Self, String> {
        let d = device;
        let (heads, d_model) = (decoder.heads(), decoder.native(site_index(0, 0)).ncols());
        let head_dim = d_model / heads;
        let d_mlp = decoder.native(site_index(0, 4)).nrows();
        let mut layers = Vec::with_capacity(decoder.layers());
        for l in 0..decoder.layers() {
            let (rms1, rms2) = decoder.gains(l);
            let mut maps = Vec::with_capacity(KINDS.len());
            for kind in 0..KINDS.len() {
                let w = decoder.native(site_index(l, kind));
                let (ins, outs) = parts(kind, heads, head_dim, d_model, d_mlp);
                let (io, oo) = (offsets(&ins), offsets(&outs));
                let blocks = (0..outs.len())
                    .map(|i| (0..ins.len()).map(|j| d.upload(w.slice(s![oo[i]..oo[i + 1], io[j]..io[j + 1]])).map_err(error)).collect())
                    .collect::<Result<_, String>>()?;
                maps.push(Map { blocks });
            }
            layers.push(Layer { rms1: upload_row(d, rms1.to_vec())?, rms2: upload_row(d, rms2.to_vec())?, maps });
        }
        let half = head_dim / 2;
        let inv_freq = decoder.inv_freq();
        let (mut cos, mut sin) = (Array2::zeros((context, half)), Array2::zeros((context, half)));
        for pos in 0..context {
            for i in 0..half {
                let (s, c) = (pos as f64 * inv_freq[i]).sin_cos();
                cos[[pos, i]] = c;
                sin[[pos, i]] = s;
            }
        }
        Ok(Self {
            device: device.clone(),
            wte: d.upload(decoder.embedding().view()).map_err(error)?,
            final_gain: upload_row(d, decoder.final_gain().to_vec())?,
            layers,
            heads,
            head_dim,
            d_model,
            d_mlp,
            vocab: decoder.embedding().nrows(),
            eps: decoder.eps(),
            cos,
            sin,
            gelu: d.upload_indices(&vec![PointwiseLaw::GeluTanh.code(); d_mlp]).map_err(error)?,
        })
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The column widths of decoder site `site`'s input and output parts.
    #[must_use]
    pub fn site_parts(&self, site: usize) -> (Vec<usize>, Vec<usize>) {
        parts(site % KINDS.len(), self.heads, self.head_dim, self.d_model, self.d_mlp)
    }

    /// The bytes a block of `rows` rows holds at the widest point of a forward, for sizing batches:
    /// the residual, the MLP's hidden rows twice, a layer's queries, keys and values, a head's
    /// scores and the widest library's reads.
    #[must_use]
    pub fn bytes_per_row(&self, length: usize, widest_units: usize) -> usize {
        8 * (4 * self.d_model + 2 * self.d_mlp + 3 * self.d_model + 2 * length + 2 * widest_units)
    }

    /// The batch of `passages` (each block rows `start..length` of its tokens `tokens[p]`).
    pub fn batch(&self, tokens: &[Vec<u32>], passages: &[usize], length: usize, start: usize) -> Result<Batch, String> {
        if start >= length || length > self.cos.nrows() {
            return Err(format!("eval device: rows {start}..{length} with rotations for {} positions", self.cos.nrows()));
        }
        let mut ids = Vec::with_capacity(passages.len() * (length - start));
        let mut positions = Vec::with_capacity(ids.capacity());
        for &p in passages {
            let t = tokens.get(p).ok_or_else(|| format!("eval device: no passage {p}"))?;
            if t.len() < length {
                return Err(format!("eval device: passage {p} has {} tokens, not {length}", t.len()));
            }
            ids.extend_from_slice(&t[start..length]);
            positions.extend((start..length).map(|r| r as u32));
        }
        if let Some(bad) = ids.iter().find(|t| **t as usize >= self.vocab) {
            return Err(format!("eval device: token {bad} outside the vocabulary of {}", self.vocab));
        }
        let d = &self.device;
        let table = |values: &Array2<f64>| -> Result<Tensor, String> {
            let rows: Vec<usize> = positions.iter().map(|p| *p as usize).collect();
            d.upload(values.select(Axis(0), &rows).view()).map_err(error)
        };
        Ok(Batch {
            passages: passages.to_vec(),
            length,
            start,
            ids: d.upload_indices(&ids).map_err(error)?,
            cos: table(&self.cos)?,
            sin: table(&self.sin)?,
        })
    }

    /// `x / √(mean(x²) + ε) ⊙ gain` per row.
    fn norm(&self, x: &Tensor, gain: &Tensor) -> Result<Tensor, String> {
        let d = &self.device;
        let normed = d.rms_norm(x, self.eps).map_err(error)?;
        let mut out = d.zeros(x.rows(), x.cols()).map_err(error)?;
        d.scale_columns(&mut out, &normed, gain, false).map_err(error)?;
        Ok(out)
    }

    /// The logits of residual rows `start..start + n` (`log_probs` before its normalisation), the
    /// head's product in `arithmetic`.
    pub fn logits(&self, residual: &Tensor, start: usize, n: usize, arithmetic: Arithmetic) -> Result<Tensor, String> {
        let d = &self.device;
        let rows = d.rows_of(residual, start, n).map_err(error)?;
        let h = self.norm(&rows, &self.final_gain)?;
        let mut out = d.zeros(n, self.vocab).map_err(error)?;
        d.gemm(&mut out, 1.0, &h, Op::N, &self.wte, Op::T, 0.0, arithmetic).map_err(error)?;
        Ok(out)
    }

    /// Rows per logits tile.
    #[must_use]
    pub fn tile_rows(&self) -> usize {
        (TILE_BYTES / (8 * self.vocab.max(1))).max(1)
    }

    /// An explanation's units at decoder site `site`: `v` (units × d_in) and `u` (units × d_out)
    /// split into the site's parts.
    pub fn units(&self, site: usize, v: &Array2<f64>, u: &Array2<f64>) -> Result<Units, String> {
        let (ins, outs) = self.site_parts(site);
        let (io, oo) = (offsets(&ins), offsets(&outs));
        if v.ncols() != io[ins.len()] || u.ncols() != oo[outs.len()] || u.nrows() != v.nrows() {
            return Err(format!("eval device: site {site} units {:?}, {:?}", v.dim(), u.dim()));
        }
        let d = &self.device;
        Ok(Units {
            v: (0..ins.len()).map(|j| d.upload(v.slice(s![.., io[j]..io[j + 1]])).map_err(error)).collect::<Result<_, _>>()?,
            u: (0..outs.len()).map(|i| d.upload(u.slice(s![.., oo[i]..oo[i + 1]])).map_err(error)).collect::<Result<_, _>>()?,
            columns: v.nrows(),
        })
    }

    /// `x ← (1 − α) x + α donor` at tensor row `row` of every part.
    fn mix(&self, x: &mut [Tensor], row: usize, alpha: f64, donor: &[Tensor]) -> Result<(), String> {
        let d = &self.device;
        for (part, state) in x.iter_mut().zip(donor) {
            let current = d.rows_of(part, row, 1).map_err(error)?;
            let mut mixed = d.zeros(1, part.cols()).map_err(error)?;
            d.scale_columns(&mut mixed, &current, &upload_row(d, vec![1.0 - alpha; part.cols()])?, false).map_err(error)?;
            d.axpy(&mut mixed, alpha, state).map_err(error)?;
            d.set_rows(part, row, &mixed).map_err(error)?;
        }
        Ok(())
    }

    /// Columns `a..b` of the concatenated parts scaled by `scale` at tensor rows `lo..hi`.
    fn scale(&self, x: &mut [Tensor], (lo, hi): (usize, usize), (a, b): (usize, usize), scale: f64) -> Result<(), String> {
        let d = &self.device;
        let mut at = 0;
        for part in x.iter_mut() {
            let width = part.cols();
            let (first, last) = (a.max(at), b.min(at + width));
            if first < last {
                let factors: Vec<f64> = (at..at + width).map(|c| if c >= first && c < last { scale } else { 1.0 }).collect();
                let rows = d.rows_of(part, lo, hi - lo).map_err(error)?;
                let mut out = d.zeros(hi - lo, width).map_err(error)?;
                d.scale_columns(&mut out, &rows, &upload_row(d, factors)?, false).map_err(error)?;
                d.set_rows(part, lo, &out).map_err(error)?;
            }
            at += width;
        }
        Ok(())
    }

    /// The row `row` of every part, a donor state.
    fn state(&self, x: &[Tensor], row: usize) -> Result<Vec<Tensor>, String> {
        x.iter().map(|p| self.device.rows_of(p, row, 1).map_err(error)).collect()
    }

    /// `counterfactual::Program::apply` of decoder site `site` on the batch's computed rows.
    fn apply(&self, run: &Run<'_>, batch: &Batch, site: usize, input: &[Tensor], recorded: &mut [Donor], units: &mut [Vec<f64>]) -> Result<Vec<Tensor>, String> {
        let d = &self.device;
        let (layer, kind) = (site / KINDS.len(), site % KINDS.len());
        let span = |b: usize, rows: Rows| -> (usize, usize) {
            match rows {
                Rows::One(r) => (batch.row(b, r), batch.row(b, r) + 1),
                Rows::All => (batch.row(b, batch.start), batch.row(b, batch.start) + batch.computed()),
            }
        };
        let changes_input = run.actions.iter().any(|a| a.iter().any(|x| matches!(x, Action::Input { site: k, .. } if *k == site)));
        let copied: Vec<Tensor>;
        let x: &[Tensor] = if changes_input {
            let mut owned = input.iter().map(|p| d.copy(p).map_err(error)).collect::<Result<Vec<_>, _>>()?;
            for (b, actions) in run.actions.iter().enumerate() {
                for action in actions.iter() {
                    let Action::Input { site: k, change } = action else { continue };
                    if *k != site {
                        continue;
                    }
                    match change {
                        InputChange::Scale { rows, cols, scale } => self.scale(&mut owned, span(b, *rows), *cols, *scale)?,
                        InputChange::Mix { row, alpha } => {
                            if let Some(state) = run.donors[b].and_then(|donor| donor.states.get(&(site, *row, false))) {
                                self.mix(&mut owned, batch.row(b, *row), *alpha, state)?;
                            }
                        }
                    }
                }
            }
            copied = owned;
            &copied
        } else {
            input
        };
        for (b, keys) in run.record.iter().enumerate() {
            for &(k, r, output) in keys {
                if k == site && !output {
                    recorded[b].states.insert((k, r, false), self.state(x, batch.row(b, r))?);
                }
            }
        }
        let (_, outs) = self.site_parts(site);
        let rows = batch.rows();
        let mut y: Vec<Tensor> = outs.iter().map(|w| d.zeros(rows, *w).map_err(error)).collect::<Result<_, _>>()?;
        let library = match &run.maps {
            Maps::Units { units, rule } => units.get(site).copied().flatten().map(|u| (u, *rule)),
            Maps::Native => None,
        };
        match library {
            None => {
                let map = &self.layers[layer].maps[kind];
                for (i, out) in y.iter_mut().enumerate() {
                    for (j, part) in x.iter().enumerate() {
                        d.gemm(out, 1.0, part, Op::N, &map.blocks[i][j], Op::T, 1.0, run.arithmetic).map_err(error)?;
                    }
                }
            }
            Some((library, rule)) => {
                let mask = rule.mask(self, &At { batch, site, input: x })?;
                if mask.dim() != (rows, library.columns) {
                    return Err(format!("eval device: a {:?} mask for {rows} rows of {} units", mask.dim(), library.columns));
                }
                let mut z = d.zeros(rows, library.columns).map_err(error)?;
                for (part, v) in x.iter().zip(&library.v) {
                    d.gemm(&mut z, 1.0, part, Op::N, v, Op::T, 1.0, run.arithmetic).map_err(error)?;
                }
                let mut gated = d.zeros(rows, library.columns).map_err(error)?;
                d.hadamard(&mut gated, &z, &mask, false).map_err(error)?;
                drop(z);
                for (out, u) in y.iter_mut().zip(&library.u) {
                    d.gemm(out, 1.0, &gated, Op::N, u, Op::N, 1.0, run.arithmetic).map_err(error)?;
                }
                drop(gated);
                // The units on per row: the mask's row sums.
                let ones = d.upload_vec(library.columns, 1, vec![1.0; library.columns]).map_err(error)?;
                let mut counts = d.zeros(rows, 1).map_err(error)?;
                d.gemm(&mut counts, 1.0, &mask, Op::N, &ones, Op::N, 0.0, exact(d)).map_err(error)?;
                let counts = d.download(&counts).map_err(error)?;
                for (b, per_row) in units.iter_mut().enumerate() {
                    for (r, c) in per_row.iter_mut().zip(counts.slice(s![b * batch.computed()..(b + 1) * batch.computed(), 0])) {
                        *r += c;
                    }
                }
            }
        }
        for (b, actions) in run.actions.iter().enumerate() {
            for action in actions.iter() {
                let Action::Output { site: k, change } = action else { continue };
                if *k != site {
                    continue;
                }
                match change {
                    OutputChange::Mix { row, alpha } => {
                        if let Some(state) = run.donors[b].and_then(|donor| donor.states.get(&(site, *row, true))) {
                            self.mix(&mut y, batch.row(b, *row), *alpha, state)?;
                        }
                    }
                    OutputChange::Add { left, right } => self.add_edit(&mut y, x, (batch, b), (left, right), run.arithmetic)?,
                }
            }
        }
        for (b, keys) in run.record.iter().enumerate() {
            for &(k, r, output) in keys {
                if k == site && output {
                    recorded[b].states.insert((k, r, true), self.state(&y, batch.row(b, r))?);
                }
            }
        }
        Ok(y)
    }

    /// `y ← y + L (Rᵀ x)` on block `b`'s rows: a weight edit (`OutputChange::Add`).
    fn add_edit(&self, y: &mut [Tensor], x: &[Tensor], (batch, b): (&Batch, usize), (left, right): (&Arc<Array2<f64>>, &Arc<Array2<f64>>), arithmetic: Arithmetic) -> Result<(), String> {
        let d = &self.device;
        let (lo, n) = (batch.row(b, batch.start), batch.computed());
        let rank = left.ncols();
        let mut t = d.zeros(n, rank).map_err(error)?;
        let mut at = 0;
        for part in x {
            let rows = d.rows_of(part, lo, n).map_err(error)?;
            let r = d.upload(right.slice(s![at..at + part.cols(), ..])).map_err(error)?;
            d.gemm(&mut t, 1.0, &rows, Op::N, &r, Op::N, 1.0, arithmetic).map_err(error)?;
            at += part.cols();
        }
        let mut at = 0;
        for part in y.iter_mut() {
            let l = d.upload(left.slice(s![at..at + part.cols(), ..])).map_err(error)?;
            let mut rows = d.rows_of(part, lo, n).map_err(error)?;
            d.gemm(&mut rows, 1.0, &t, Op::N, &l, Op::T, 1.0, arithmetic).map_err(error)?;
            d.set_rows(part, lo, &rows).map_err(error)?;
            at += part.cols();
        }
        Ok(())
    }

    /// Causal attention of every head, queries at the batch's computed rows, keys and values at
    /// every row (those before the start the clean runs').
    fn attend(&self, (batch, layer, arithmetic): (&Batch, usize, Arithmetic), (q, k, v): (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>), clean: Option<&[&Clean]>, keep: &mut Option<Vec<Clean>>) -> Result<Vec<Tensor>, String> {
        let d = &self.device;
        let (blocks, length, start, computed) = (batch.blocks(), batch.length, batch.start, batch.computed());
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut out = Vec::with_capacity(self.heads);
        for (h, ((qh, kh), vh)) in q.into_iter().zip(k).zip(v).enumerate() {
            let qh = d.rotate(&qh, &batch.cos, &batch.sin, true, false).map_err(error)?;
            let kh = d.rotate(&kh, &batch.cos, &batch.sin, true, false).map_err(error)?;
            let (keys, values) = if start == 0 {
                (kh, vh)
            } else {
                let clean = clean.ok_or("eval device: rows after the start need the clean run")?;
                let mut keys = d.zeros(blocks * length, self.head_dim).map_err(error)?;
                let mut values = d.zeros(blocks * length, self.head_dim).map_err(error)?;
                for b in 0..blocks {
                    let prefix_k = d.rows_of(&clean[b].keys[layer][h], 0, start).map_err(error)?;
                    let prefix_v = d.rows_of(&clean[b].values[layer][h], 0, start).map_err(error)?;
                    d.set_rows(&mut keys, b * length, &prefix_k).map_err(error)?;
                    d.set_rows(&mut values, b * length, &prefix_v).map_err(error)?;
                    d.set_rows(&mut keys, b * length + start, &d.rows_of(&kh, b * computed, computed).map_err(error)?).map_err(error)?;
                    d.set_rows(&mut values, b * length + start, &d.rows_of(&vh, b * computed, computed).map_err(error)?).map_err(error)?;
                }
                (keys, values)
            };
            let mut scores = d.zeros(blocks * computed, length).map_err(error)?;
            d.gemm_batched(blocks, &mut scores, scale, &qh, Op::N, &keys, Op::T, 0.0, arithmetic).map_err(error)?;
            d.softmax_rows_blocks(&mut scores, computed, start).map_err(error)?;
            let mut head = d.zeros(blocks * computed, self.head_dim).map_err(error)?;
            d.gemm_batched(blocks, &mut head, 1.0, &scores, Op::N, &values, Op::N, 0.0, arithmetic).map_err(error)?;
            if let Some(kept) = keep.as_mut() {
                for (b, c) in kept.iter_mut().enumerate() {
                    c.keys[layer].push(d.rows_of(&keys, b * length, length).map_err(error)?);
                    c.values[layer].push(d.rows_of(&values, b * length, length).map_err(error)?);
                }
            }
            out.push(head);
        }
        Ok(out)
    }

    /// The residual entering layer `layer` at the batch's computed rows, from the clean runs.
    fn entering(&self, batch: &Batch, layer: usize, clean: &[&Clean]) -> Result<Tensor, String> {
        let d = &self.device;
        let mut x = d.zeros(batch.rows(), self.d_model).map_err(error)?;
        for (b, c) in clean.iter().enumerate() {
            let rows = d.rows_of(&c.residuals[layer], batch.start, batch.computed()).map_err(error)?;
            d.set_rows(&mut x, batch.row(b, batch.start), &rows).map_err(error)?;
        }
        Ok(x)
    }

    /// The residual at block `b`'s interface rows after layer `layer` (`x` the residual after it
    /// at the computed rows), rows before the start the clean run's.
    fn interface_rows(&self, batch: &Batch, b: usize, rows: &[usize], x: Option<&Tensor>, clean: Option<&Clean>, layer: usize) -> Result<Tensor, String> {
        let d = &self.device;
        let mut out = d.zeros(rows.len(), self.d_model).map_err(error)?;
        for (i, &r) in rows.iter().enumerate() {
            let row = match x {
                Some(x) if r >= batch.start => d.rows_of(x, batch.row(b, r), 1).map_err(error)?,
                _ => d.rows_of(&clean.ok_or("eval device: an interface row before the start needs the clean run")?.residuals[layer + 1], r, 1).map_err(error)?,
            };
            d.set_rows(&mut out, i, &row).map_err(error)?;
        }
        Ok(out)
    }

    /// One program's forward over `batch` (module note).
    pub fn forward(&self, batch: &Batch, run: &Run<'_>) -> Result<Ran, String> {
        let d = &self.device;
        let blocks = batch.blocks();
        if run.actions.len() != blocks || run.donors.len() != blocks || run.interface.len() != blocks || run.record.len() != blocks {
            return Err(format!("eval device: a run's per-block lists do not match {blocks} blocks"));
        }
        if run.clean.as_ref().is_some_and(|c| c.len() != blocks) {
            return Err("eval device: clean runs do not match the blocks".to_string());
        }
        if run.keep && (batch.start != 0 || run.from_layer != 0) {
            return Err("eval device: a kept run starts at row and layer 0".to_string());
        }
        if (batch.start > 0 || run.from_layer > 0) && run.clean.is_none() {
            return Err("eval device: a run after row or layer 0 needs the clean runs".to_string());
        }
        let clean = run.clean.as_deref();
        let layers = self.layers.len();
        let mut kept = run.keep.then(|| {
            (0..blocks)
                .map(|_| Clean { residuals: Vec::with_capacity(layers + 1), keys: (0..layers).map(|_| Vec::new()).collect(), values: (0..layers).map(|_| Vec::new()).collect(), units: vec![0.0; batch.length] })
                .collect::<Vec<_>>()
        });
        let mut recorded: Vec<Donor> = (0..blocks).map(|_| Donor::default()).collect();
        let mut units: Vec<Vec<f64>> = vec![vec![0.0; batch.computed()]; blocks];
        let mut interface: Vec<Vec<Tensor>> = (0..blocks).map(|_| Vec::with_capacity(layers)).collect();
        for (b, rows) in run.interface.iter().enumerate() {
            for layer in 0..run.from_layer {
                interface[b].push(self.interface_rows(batch, b, rows, None, clean.map(|c| c[b]), layer)?);
            }
        }
        let mut x = match clean {
            Some(c) => self.entering(batch, run.from_layer, c)?,
            None => d.gather_rows(&self.wte, &batch.ids).map_err(error)?,
        };
        for layer in run.from_layer..layers {
            if let Some(kept) = kept.as_mut() {
                for (b, c) in kept.iter_mut().enumerate() {
                    c.residuals.push(d.rows_of(&x, b * batch.length, batch.length).map_err(error)?);
                }
            }
            let weights = &self.layers[layer];
            let n = [self.norm(&x, &weights.rms1)?];
            let q = self.apply(run, batch, site_index(layer, 0), &n, &mut recorded, &mut units)?;
            let k = self.apply(run, batch, site_index(layer, 1), &n, &mut recorded, &mut units)?;
            let v = self.apply(run, batch, site_index(layer, 2), &n, &mut recorded, &mut units)?;
            drop(n);
            let attended = self.attend((batch, layer, run.arithmetic), (q, k, v), clean, &mut kept)?;
            let o = self.apply(run, batch, site_index(layer, 3), &attended, &mut recorded, &mut units)?;
            drop(attended);
            d.axpy(&mut x, 1.0, &o[0]).map_err(error)?;
            drop(o);
            let n2 = [self.norm(&x, &weights.rms2)?];
            let up = self.apply(run, batch, site_index(layer, 4), &n2, &mut recorded, &mut units)?;
            drop(n2);
            let hidden = [d.law_values(&up[0], &self.gelu, gelu_tanh_constant()).map_err(error)?];
            drop(up);
            let down = self.apply(run, batch, site_index(layer, 5), &hidden, &mut recorded, &mut units)?;
            drop(hidden);
            d.axpy(&mut x, 1.0, &down[0]).map_err(error)?;
            drop(down);
            for (b, rows) in run.interface.iter().enumerate() {
                interface[b].push(self.interface_rows(batch, b, rows, Some(&x), clean.map(|c| c[b]), layer)?);
            }
        }
        if let Some(kept) = kept.as_mut() {
            for (b, c) in kept.iter_mut().enumerate() {
                c.residuals.push(d.rows_of(&x, b * batch.length, batch.length).map_err(error)?);
                c.units.copy_from_slice(&units[b]);
            }
        }
        Ok(Ran { residual: x, interface, recorded, units, kept })
    }
}

/// Float64 products where the device has them, else its own f32 (the Apple GPU).
pub(crate) fn exact(device: &Device) -> Arithmetic {
    if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 }
}

/// Per row of two logits tiles on the device, `KL(softmax(p) ‖ softmax(q))`.
pub fn kl_rows(device: &Device, p: &Tensor, q: &Tensor) -> Result<Vec<f64>, String> {
    let mut scratch = device.copy(q).map_err(error)?;
    device.kl_score_rows(p, &mut scratch, None).map_err(error)
}

/// The given selections of some blocks, a site's units on per row (rows × units, 1 or 0) built
/// on the host and uploaded: `selections[b]` lists row `r`'s `(site, unit)` pairs on.
pub struct GivenRows<'a> {
    pub selections: Vec<&'a Selection>,
    /// Per decoder site, its units.
    pub columns: Vec<usize>,
}

impl Rule for GivenRows<'_> {
    fn mask(&self, resident: &Resident, at: &At<'_>) -> Result<Tensor, String> {
        let (batch, site) = (at.batch, at.site);
        let columns = self.columns[site];
        let mut at = Vec::new();
        for (b, selection) in self.selections.iter().enumerate() {
            for r in batch.start..batch.length {
                let row = selection.rows.get(r).ok_or_else(|| format!("eval device: a selection without row {r}"))?;
                let first = batch.row(b, r) * columns;
                at.extend(row.iter().filter(|(k, _)| *k as usize == site).map(|(_, c)| (first + *c as usize) as u32));
            }
        }
        let d = resident.device();
        let mut mask = d.zeros(batch.rows(), columns).map_err(error)?;
        d.fill_entries(&mut mask, &d.upload_indices(&at).map_err(error)?, 1.0).map_err(error)?;
        Ok(mask)
    }
}

/// A fitted explanation's own rule (`counterfactual::FittedRule`, `explanation::Fitted::select`):
/// each site's blocks chosen from the program's own read, every column of a chosen block on.
pub struct FittedRows<'a> {
    pub sites: &'a [Option<&'a super::explanation::Fitted>],
}

impl Rule for FittedRows<'_> {
    fn mask(&self, resident: &Resident, at: &At<'_>) -> Result<Tensor, String> {
        let fitted = self.sites.get(at.site).copied().flatten().ok_or_else(|| format!("eval device: no fitted rule at site {}", at.site))?;
        let d = resident.device();
        let parts: Vec<Array2<f64>> = at.input.iter().map(|p| d.download(p).map_err(error)).collect::<Result<_, _>>()?;
        let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
        let reads = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
        let on = fitted.select(&reads);
        let columns: Vec<usize> = fitted.ranks.iter().enumerate().flat_map(|(b, r)| std::iter::repeat_n(b, *r)).collect();
        let mask = Array2::from_shape_fn((reads.nrows(), columns.len()), |(r, c)| on[[r, columns[c]]]);
        d.upload(mask.view()).map_err(error)
    }
}

/// Uploads cached by key (a given mask per site and passage, uploaded once).
#[derive(Default)]
pub struct Cache {
    held: Mutex<BTreeMap<(usize, usize, usize), Arc<Tensor>>>,
}

impl Cache {
    pub fn get(&self, key: (usize, usize, usize), upload: impl FnOnce() -> Result<Tensor, String>) -> Result<Arc<Tensor>, String> {
        if let Some(t) = self.held.lock().map_err(|_| "eval device: a poisoned cache".to_string())?.get(&key) {
            return Ok(Arc::clone(t));
        }
        let t = Arc::new(upload()?);
        self.held.lock().map_err(|_| "eval device: a poisoned cache".to_string())?.insert(key, Arc::clone(&t));
        Ok(t)
    }
}

/// Per-row KL of a final residual's computed rows against target logits per block, rows
/// `from[b]..length` of block `b`: `targets[b]` holds the target's logits at rows `start..` of
/// its block (or the whole passage when `whole`).
pub fn kl_against(resident: &Resident, batch: &Batch, residual: &Tensor, targets: &[&Tensor], whole: bool, arithmetic: Arithmetic) -> Result<Vec<Array1<f64>>, String> {
    let d = resident.device();
    let tile = resident.tile_rows();
    let computed = batch.computed();
    let mut out = Vec::with_capacity(batch.blocks());
    for (b, target) in targets.iter().enumerate() {
        let mut kl = Vec::with_capacity(computed);
        for at in (0..computed).step_by(tile) {
            let n = tile.min(computed - at);
            let logits = resident.logits(residual, b * computed + at, n, arithmetic)?;
            let offset = if whole { batch.start + at } else { at };
            let t = d.rows_of(target, offset, n).map_err(error)?;
            kl.extend(kl_rows(d, &t, &logits)?);
        }
        out.push(Array1::from(kl));
    }
    Ok(out)
}

/// An explanation evaluated on the device (`counterfactual::Explanation`): its units per decoder
/// site and the rule its programs run.
pub enum Explained<'a> {
    /// Each episode's given selection, by episode id (a rule run elsewhere on the program's own
    /// states); `columns` is every decoder site's units.
    Given { units: Vec<Option<Units>>, columns: Vec<usize>, selection: &'a (dyn Fn(&str) -> Result<Selection, String> + Sync) },
    /// A fitted explanation's own rule (`counterfactual::FittedRule`).
    Fitted { units: Vec<Option<Units>>, sites: &'a [Option<&'a super::explanation::Fitted>] },
}

impl Explained<'_> {
    fn units(&self) -> Vec<Option<&Units>> {
        match self {
            Self::Given { units, .. } | Self::Fitted { units, .. } => units.iter().map(Option::as_ref).collect(),
        }
    }

    fn widest(&self) -> usize {
        self.units().iter().flatten().map(|u| u.columns).max().unwrap_or(0)
    }
}

/// Per row of two logits tiles on the device, `KL(softmax(p) ‖ softmax(q))` (`q` is scratch, left
/// unchanged).
fn kl_tiles(device: &Device, p: &Tensor, q: &mut Tensor) -> Result<Vec<f64>, String> {
    device.kl_score_rows(p, q, None).map_err(error)
}

/// The rows of a program's residual: position `r` of a block at tensor row `first + r − start`.
#[derive(Clone, Copy)]
struct Placed<'t> {
    residual: &'t Tensor,
    first: usize,
    start: usize,
}

impl Placed<'_> {
    fn row(&self, r: usize) -> usize {
        self.first + r - self.start
    }
}

/// Block `b` of `batch` in `residual`.
fn placed<'t>(batch: &Batch, residual: &'t Tensor, b: usize) -> Placed<'t> {
    Placed { residual, first: batch.row(b, batch.start), start: batch.start }
}

/// A clean run's final residual, every row.
fn whole(clean: &Clean) -> Placed<'_> {
    Placed { residual: &clean.residuals[clean.residuals.len() - 1], first: 0, start: 0 }
}

/// A clean run's residual after layer `layer` at `rows`.
fn clean_rows(device: &Device, clean: &Clean, layer: usize, rows: &[usize]) -> Result<Tensor, String> {
    let source = &clean.residuals[layer + 1];
    let mut out = device.zeros(rows.len(), source.cols()).map_err(error)?;
    for (i, &r) in rows.iter().enumerate() {
        device.set_rows(&mut out, i, &device.rows_of(source, r, 1).map_err(error)?).map_err(error)?;
    }
    Ok(out)
}

/// `counterfactual::score` on the device: rows `from..length` of the native and explained
/// programs against the native clean logits (`target`, every row of the passage), and their
/// residuals after every layer at the interface rows against the native clean run's.
fn score_on(
    resident: &Resident,
    (native, explained): (Placed<'_>, Placed<'_>),
    (native_rows, explained_rows): (&[Tensor], &[Tensor]),
    (clean, target): (&Clean, &Tensor),
    interface: &[usize],
    (from, length): (usize, usize),
    arithmetic: Arithmetic,
) -> Result<super::counterfactual::Scores, String> {
    let d = resident.device();
    let tile = resident.tile_rows();
    let (mut kl, mut effect, mut agree) = (0.0, 0.0, 0.0);
    for at in (from..length).step_by(tile) {
        let n = tile.min(length - at);
        let p = resident.logits(native.residual, native.row(at), n, arithmetic)?;
        let mut q = resident.logits(explained.residual, explained.row(at), n, arithmetic)?;
        let mut c = d.rows_of(target, at, n).map_err(error)?;
        kl += kl_tiles(d, &p, &mut q)?.iter().sum::<f64>();
        effect += kl_tiles(d, &p, &mut c)?.iter().sum::<f64>();
        let (a, b) = (d.argmax_rows(&p).map_err(error)?, d.argmax_rows(&q).map_err(error)?);
        agree += a.iter().zip(&b).filter(|(x, y)| x == y).count() as f64;
    }
    let n = (length - from).max(1) as f64;
    let mean = |v: Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let (mut interface_kl, mut interface_effect) = (Vec::new(), Vec::new());
    for (l, (a, b)) in native_rows.iter().zip(explained_rows).enumerate() {
        let rows = interface.len();
        let p = resident.logits(a, 0, rows, arithmetic)?;
        let mut q = resident.logits(b, 0, rows, arithmetic)?;
        let mut c = resident.logits(&clean_rows(d, clean, l, interface)?, 0, rows, arithmetic)?;
        interface_kl.push(mean(kl_tiles(d, &p, &mut q)?));
        interface_effect.push(mean(kl_tiles(d, &p, &mut c)?));
    }
    Ok(super::counterfactual::Scores { kl: kl / n, native_effect: effect / n, top1_agree: agree / n, interface_kl, interface_effect })
}

/// A clean run's interface rows after every layer.
fn clean_interface(device: &Device, clean: &Clean, layers: usize, rows: &[usize]) -> Result<Vec<Tensor>, String> {
    (0..layers).map(|l| clean_rows(device, clean, l, rows)).collect()
}

/// How many sequences of `length` rows a batch holds: a quarter of the device's free memory, 4 GiB
/// where it does not say.
pub fn per_batch(resident: &Resident, length: usize, widest_units: usize) -> Result<usize, String> {
    let free = resident.device().memory().map_err(error)?.map_or(4 << 30, |(free, _)| free / 4);
    Ok((free / (length * resident.bytes_per_row(length, widest_units)).max(1)).max(1))
}

/// `counterfactual::evaluate` on the device (module note): every episode of `spec` on
/// `passages`, the explanation's program against the native model's (`None` scores the native
/// model as its own explanation). Each passage runs clean once per program (its keys, values and
/// every layer's residual kept, the donor states recorded); each episode then runs from its first
/// changed row and layer, batched with every episode starting there.
pub fn evaluate(resident: &Resident, spec: &Spec, passages: &[Vec<u32>], explained: Option<&Explained<'_>>) -> Result<Vec<Scored>, String> {
    let started = std::time::Instant::now();
    let d = resident.device();
    let arithmetic = exact(d);
    let (length, layers) = (spec.rows, resident.layers());
    let mut wanted: BTreeMap<usize, Vec<(usize, usize, bool)>> = BTreeMap::new();
    for e in &spec.episodes {
        let keys: Vec<(usize, usize, bool)> = e.actions.iter().filter_map(Action::donor_state).collect();
        if keys.is_empty() {
            continue;
        }
        let read = wanted.entry(e.donor.ok_or_else(|| format!("{}: a mix without a donor", e.id))?).or_default();
        read.extend(keys.into_iter().filter(|k| !read.contains(k)).collect::<Vec<_>>());
    }
    let named: std::collections::BTreeSet<usize> = spec.episodes.iter().map(|e| e.passage).chain(wanted.keys().copied()).collect();
    let named: Vec<usize> = named.into_iter().collect();
    let size = per_batch(resident, length, explained.map_or(0, Explained::widest))?;
    // Every program's clean runs, the native ones' logits and both programs' donor states.
    let mut native_clean: BTreeMap<usize, Clean> = BTreeMap::new();
    let mut targets: BTreeMap<usize, Tensor> = BTreeMap::new();
    let mut native_donors: BTreeMap<usize, Donor> = BTreeMap::new();
    let mut own_clean: BTreeMap<usize, Clean> = BTreeMap::new();
    let mut own_donors: BTreeMap<usize, Donor> = BTreeMap::new();
    for chunk in named.chunks(size) {
        let batch = resident.batch(passages, chunk, length, 0)?;
        let record: Vec<Vec<(usize, usize, bool)>> = chunk.iter().map(|p| wanted.get(p).cloned().unwrap_or_default()).collect();
        let ran = resident.forward(&batch, &Run::clean(Maps::Native, chunk.len(), record.clone(), arithmetic))?;
        for ((&p, clean), donor) in chunk.iter().zip(ran.kept.ok_or("eval device: a clean run not kept")?).zip(ran.recorded) {
            let final_residual = &clean.residuals[layers];
            let mut logits = d.zeros(length, resident.vocab()).map_err(error)?;
            for at in (0..length).step_by(resident.tile_rows()) {
                let n = resident.tile_rows().min(length - at);
                d.set_rows(&mut logits, at, &resident.logits(final_residual, at, n, arithmetic)?).map_err(error)?;
            }
            targets.insert(p, logits);
            native_clean.insert(p, clean);
            native_donors.insert(p, donor);
        }
        if let Some(x) = explained {
            let ids: Vec<String> = chunk.iter().map(|p| format!("clean/{p}")).collect();
            let ran = with_rule(x, &ids, |rule| resident.forward(&batch, &Run::clean(Maps::Units { units: x.units(), rule }, chunk.len(), record.clone(), arithmetic)))?;
            for ((&p, clean), donor) in chunk.iter().zip(ran.kept.ok_or("eval device: a clean run not kept")?).zip(ran.recorded) {
                own_clean.insert(p, clean);
                own_donors.insert(p, donor);
            }
        }
    }
    log::info!("clean runs of {} passages on {}, {:.1}s", named.len(), d.name(), started.elapsed().as_secs_f64());
    // Each episode's first changed row and layer: rows before and layers before are its programs'
    // clean runs. A given selection that differs from the clean one before that row runs whole.
    let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    let mut quiet = Vec::new();
    for (i, e) in spec.episodes.iter().enumerate() {
        if e.actions.is_empty() {
            quiet.push(i);
            continue;
        }
        let mut row = e.actions.iter().map(Action::first_row).min().unwrap_or(0);
        let mut layer = e.actions.iter().map(|a| a.site() / KINDS.len()).min().unwrap_or(0);
        if let Some(Explained::Given { selection, .. }) = explained
            && row > 0
        {
            let (own, clean) = (selection(&e.id)?, selection(&format!("clean/{}", e.passage))?);
            if own.rows.get(..row) != clean.rows.get(..row) {
                (row, layer) = (0, 0);
            }
        }
        groups.entry((row, layer)).or_default().push(i);
    }
    let none = Donor::default();
    let mut scored: Vec<Option<Scored>> = (0..spec.episodes.len()).map(|_| None).collect();
    let selected = |units: f64| if explained.is_some() { units / length as f64 } else { f64::NAN };
    for &i in &quiet {
        let e = &spec.episodes[i];
        let native = &native_clean[&e.passage];
        let own = own_clean.get(&e.passage).unwrap_or(native);
        let scores = score_on(
            resident,
            (whole(native), whole(own)),
            (&clean_interface(d, native, layers, &e.interface_rows)?, &clean_interface(d, own, layers, &e.interface_rows)?),
            (native, &targets[&e.passage]),
            &e.interface_rows,
            (0, length),
            arithmetic,
        )?;
        scored[i] = Some(Scored { id: e.id.clone(), group: e.group.clone(), passage: e.passage, from_row: 0, selected_per_row: selected(own.units.iter().sum()), scores });
    }
    let mut done = quiet.len();
    for (&(row, layer), members) in &groups {
        for chunk in members.chunks(size) {
            let episodes: Vec<&super::counterfactual::Episode> = chunk.iter().map(|i| &spec.episodes[*i]).collect();
            let order: Vec<usize> = episodes.iter().map(|e| e.passage).collect();
            let batch = resident.batch(passages, &order, length, row)?;
            let actions: Vec<&[Action]> = episodes.iter().map(|e| e.actions.as_slice()).collect();
            let interface: Vec<&[usize]> = episodes.iter().map(|e| e.interface_rows.as_slice()).collect();
            let native_run = Run {
                maps: Maps::Native,
                actions: actions.clone(),
                donors: donors_of(&episodes, &native_donors, &none),
                interface: interface.clone(),
                record: vec![Vec::new(); chunk.len()],
                from_layer: layer,
                clean: Some(order.iter().map(|p| &native_clean[p]).collect()),
                keep: false,
                arithmetic,
            };
            let native = resident.forward(&batch, &native_run)?;
            let own = match explained {
                Some(x) => {
                    let ids: Vec<String> = episodes.iter().map(|e| e.id.clone()).collect();
                    Some(with_rule(x, &ids, |rule| {
                        resident.forward(&batch, &Run {
                            maps: Maps::Units { units: x.units(), rule },
                            actions: actions.clone(),
                            donors: donors_of(&episodes, &own_donors, &none),
                            interface: interface.clone(),
                            record: vec![Vec::new(); chunk.len()],
                            from_layer: layer,
                            clean: Some(order.iter().map(|p| &own_clean[p]).collect()),
                            keep: false,
                            arithmetic,
                        })
                    })?)
                }
                None => None,
            };
            let theirs = own.as_ref().unwrap_or(&native);
            for (b, (&i, e)) in chunk.iter().zip(&episodes).enumerate() {
                let from = e.actions.iter().map(Action::first_row).min().unwrap_or(0);
                let scores = score_on(
                    resident,
                    (placed(&batch, &native.residual, b), placed(&batch, &theirs.residual, b)),
                    (&native.interface[b], &theirs.interface[b]),
                    (&native_clean[&e.passage], &targets[&e.passage]),
                    &e.interface_rows,
                    (from, length),
                    arithmetic,
                )?;
                let units = own_clean.get(&e.passage).map_or(0.0, |c| c.units[..row].iter().sum::<f64>()) + theirs.units[b].iter().sum::<f64>();
                scored[i] = Some(Scored { id: e.id.clone(), group: e.group.clone(), passage: e.passage, from_row: from, selected_per_row: selected(units), scores });
            }
            done += chunk.len();
            log::info!("{done}/{} episodes on {}, {:.1}s", spec.episodes.len(), d.name(), started.elapsed().as_secs_f64());
        }
    }
    scored.into_iter().map(|s| s.ok_or_else(|| "eval device: an episode left unscored".to_string())).collect()
}

/// Each episode's donor states from `held` by its donor passage (`none` without one).
fn donors_of<'m>(episodes: &[&super::counterfactual::Episode], held: &'m BTreeMap<usize, Donor>, none: &'m Donor) -> Vec<Option<&'m Donor>> {
    episodes.iter().map(|e| Some(e.donor.and_then(|p| held.get(&p)).unwrap_or(none))).collect()
}

/// `body` with the rule `explained` runs for blocks of episodes `ids`.
fn with_rule<T>(explained: &Explained<'_>, ids: &[String], body: impl FnOnce(&dyn Rule) -> Result<T, String>) -> Result<T, String> {
    match explained {
        Explained::Given { columns, selection, .. } => {
            let selections: Vec<Selection> = ids.iter().map(|id| selection(id)).collect::<Result<_, _>>()?;
            body(&GivenRows { selections: selections.iter().collect(), columns: columns.clone() })
        }
        Explained::Fitted { sites, .. } => body(&FittedRows { sites }),
    }
}
