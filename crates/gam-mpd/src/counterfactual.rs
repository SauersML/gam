//! Counterfactual response (#2951): how well does an explanation predict the native model's
//! response to declared native interventions?
//!
//! An explanation of a decoder is a library of rank-one units per decomposed site (`u_c v_cᵀ`)
//! and a selection rule naming, for every input, the units it runs. Its program replaces each
//! site's map `x ↦ W x` by `x_r ↦ Σ_{c ∈ S_r} m_{rc} u_c (v_cᵀ x_r)` over row `r`'s selected
//! units (masks `m`), and runs everything else of the decoder unchanged. The rule runs inside the
//! program ([`Selector`]): it sees only the program's own states, never the native model's.
//!
//! Interventions are native and decomposition-free ([`Action`]): they act on the decoder's
//! physical quantities, a site's input or output rows (a neuron is a coordinate of the MLP
//! down-projection's input; a head is a block of the attention output projection's input), or
//! add a weight edit's map to a site's output. The same actions act on the explanation's program
//! at the same places, so every explanation predicts the same physical change. An action at one
//! row is activation-level; one at every row (a scaled neuron's or head's weights, a compiled
//! edit) is weight-level. Mixing toward a donor passage takes the donor's state from the same
//! program on the donor ([`Donor`]): the native donor state natively, the explanation's own
//! donor state in its program.
//!
//! The score is the disagreement `KL(native ‖ explanation)` of the next-token distributions on
//! the rows the actions can reach, beside the native effect `KL(native ‖ native clean)` there
//! (how much there was to predict), and at the declared internal interfaces, the residual
//! stream after every layer at the episode's interface rows, the same disagreement read through
//! the final norm and unembedding (the unembedding's metric on the residual).
//!
//! The decoder is the LlamaSimpleMLP of VPD's 4-layer Pile target as exported by
//! `gam_mpd::import`'s language-model exports: pre-RMS-norm blocks, rotate-half rotary causal
//! attention over the whole head, a tanh-GELU MLP, a final RMS norm and the tied unembedding, all
//! in binary64.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};

use super::llama_simple_mlp::gelu_tanh;

/// The decomposed maps of one block, in site order.
pub const KINDS: [&str; 6] = ["q", "k", "v", "o", "c_fc", "down_proj"];
const STORAGE: [&str; 6] = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"];

/// Site `kind` of block `layer`.
pub fn site_index(layer: usize, kind: usize) -> usize {
    layer * KINDS.len() + kind
}

/// The site's name in language-model libraries (`blocks.{layer}.{kind}`).
pub fn site_name(site: usize) -> String {
    format!("blocks.{}.{}", site / KINDS.len(), KINDS[site % KINDS.len()])
}

struct Block {
    rms1: Array1<f64>,
    rms2: Array1<f64>,
    /// `W` (d_out × d_in) per kind.
    maps: Vec<Array2<f64>>,
}

/// The binary64 decoder of an export.
pub struct Decoder {
    wte: Array2<f64>,
    final_gain: Array1<f64>,
    blocks: Vec<Block>,
    heads: usize,
    eps: f64,
    inv_freq: Vec<f64>,
}

fn read_tensor(dir: &Path, name: &str, shape: [usize; 2]) -> Result<Array2<f64>, String> {
    let path = dir.join(format!("{name}.f64"));
    let matrix = read_f64_matrix(&path, shape[1])?;
    if matrix.nrows() != shape[0] {
        return Err(format!("{}: {} rows, expected {}", path.display(), matrix.nrows(), shape[0]));
    }
    Ok(matrix)
}

/// Raw little-endian float64 values as `rows × cols`.
pub fn read_f64_matrix(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if cols == 0 || bytes.len() % (cols * 8) != 0 {
        return Err(format!("{}: {} bytes are not rows of {cols} float64", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    Array2::from_shape_vec((bytes.len() / (cols * 8), cols), values).map_err(|e| e.to_string())
}

fn rms_norm(x: &Array2<f64>, gain: &Array1<f64>, eps: f64) -> Array2<f64> {
    let mut out = x.clone();
    for mut row in out.outer_iter_mut() {
        let mean = row.iter().map(|v| v * v).sum::<f64>() / row.len() as f64;
        let scale = 1.0 / (mean + eps).sqrt();
        row.iter_mut().zip(gain.iter()).for_each(|(v, g)| *v *= scale * g);
    }
    out
}

/// What computes each decomposed map in one forward: `apply(site, input)` is the site's output
/// rows for its input rows.
pub trait SiteMaps {
    fn apply(&mut self, site: usize, input: &Array2<f64>) -> Array2<f64>;
}

/// One forward: the residual rows after the last block, and after every block at the asked rows.
pub struct Forward {
    pub residual: Array2<f64>,
    /// Per layer, the residual rows (in the asked order) after that block.
    pub layers: Vec<Array2<f64>>,
}

impl Decoder {
    /// The decoder of a language-model export (`export.json` with its `config` and one
    /// `{name}.f64` per tensor).
    pub fn from_export(dir: &Path) -> Result<Self, String> {
        let record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("export.json")).map_err(|e| format!("{}: {e}", dir.display()))?)
                .map_err(|e| e.to_string())?;
        let config = &record["config"];
        let get = |k: &str| config[k].as_f64().ok_or_else(|| format!("export config has no {k}"));
        let (d_model, layers, heads, head_dim, d_mlp, vocab) =
            (get("d_model")? as usize, get("n_layers")? as usize, get("n_heads")? as usize, get("head_dim")? as usize, get("d_mlp")? as usize, get("vocab")? as usize);
        if config["rope_pairing"].as_str() != Some("rotate_half") || config["mlp_act"].as_str() != Some("gelu_tanh") || heads * head_dim != d_model {
            return Err(format!("{}: not a rotate-half, tanh-GELU decoder with d_model = heads × head_dim", dir.display()));
        }
        let theta = get("rope_theta")?;
        let inv_freq = (0..head_dim / 2).map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64)).collect();
        let vector = |name: &str| -> Result<Array1<f64>, String> { Ok(read_tensor(dir, name, [1, d_model])?.row(0).to_owned()) };
        let mut blocks = Vec::new();
        for l in 0..layers {
            let shapes = [[d_model, d_model], [d_model, d_model], [d_model, d_model], [d_model, d_model], [d_mlp, d_model], [d_model, d_mlp]];
            let maps = STORAGE.iter().zip(shapes).map(|(s, shape)| read_tensor(dir, &format!("blocks.{l}.{s}"), shape)).collect::<Result<_, _>>()?;
            blocks.push(Block { rms1: vector(&format!("blocks.{l}.rms1.gain"))?, rms2: vector(&format!("blocks.{l}.rms2.gain"))?, maps });
        }
        Ok(Self { wte: read_tensor(dir, "wte", [vocab, d_model])?, final_gain: vector("final_norm.gain")?, blocks, heads, eps: get("norm_eps")?, inv_freq })
    }

    pub fn sites(&self) -> usize {
        self.blocks.len() * KINDS.len()
    }

    pub fn layers(&self) -> usize {
        self.blocks.len()
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Site `site`'s native map `W` (d_out × d_in).
    pub fn native(&self, site: usize) -> &Array2<f64> {
        &self.blocks[site / KINDS.len()].maps[site % KINDS.len()]
    }

    fn rotate(&self, x: &mut Array2<f64>) {
        let head_dim = x.ncols() / self.heads;
        let half = head_dim / 2;
        for (pos, mut row) in x.outer_iter_mut().enumerate() {
            for i in 0..half {
                let (sin, cos) = (pos as f64 * self.inv_freq[i]).sin_cos();
                for h in 0..self.heads {
                    let (a, b) = (row[h * head_dim + i], row[h * head_dim + i + half]);
                    row[h * head_dim + i] = a * cos - b * sin;
                    row[h * head_dim + i + half] = b * cos + a * sin;
                }
            }
        }
    }

    fn attend(&self, q: &Array2<f64>, k: &Array2<f64>, v: &Array2<f64>) -> Array2<f64> {
        let head_dim = q.ncols() / self.heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = Array2::<f64>::zeros(q.dim());
        for h in 0..self.heads {
            let cols = s![.., h * head_dim..(h + 1) * head_dim];
            let mut scores = fast_abt(&q.slice(cols), &k.slice(cols));
            for (i, mut row) in scores.outer_iter_mut().enumerate() {
                let max = row.iter().take(i + 1).fold(f64::NEG_INFINITY, |m, v| m.max(*v * scale));
                let mut total = 0.0;
                for (j, value) in row.iter_mut().enumerate() {
                    *value = if j <= i { (*value * scale - max).exp() } else { 0.0 };
                    total += *value;
                }
                row.mapv_inplace(|p| p / total);
            }
            out.slice_mut(cols).assign(&fast_ab(&scores, &v.slice(cols)));
        }
        out
    }

    /// One forward of `tokens`, every decomposed map computed by `maps`, keeping the residual
    /// after every block at `layer_rows`.
    pub fn forward(&self, tokens: &[u32], maps: &mut dyn SiteMaps, layer_rows: &[usize]) -> Forward {
        let mut x = self.wte.select(Axis(0), &tokens.iter().map(|t| *t as usize).collect::<Vec<_>>());
        let mut layers = Vec::new();
        for (l, block) in self.blocks.iter().enumerate() {
            let n = rms_norm(&x, &block.rms1, self.eps);
            let mut q = maps.apply(site_index(l, 0), &n);
            let mut k = maps.apply(site_index(l, 1), &n);
            let v = maps.apply(site_index(l, 2), &n);
            self.rotate(&mut q);
            self.rotate(&mut k);
            let attended = self.attend(&q, &k, &v);
            x += &maps.apply(site_index(l, 3), &attended);
            let n = rms_norm(&x, &block.rms2, self.eps);
            let hidden = maps.apply(site_index(l, 4), &n).mapv(gelu_tanh);
            x += &maps.apply(site_index(l, 5), &hidden);
            layers.push(x.select(Axis(0), layer_rows));
        }
        Forward { residual: x, layers }
    }

    /// Next-token log-probabilities of residual rows (callers pass a tile at a time, so
    /// `rows × vocab` stays small).
    pub fn log_probs(&self, residual: &Array2<f64>) -> Array2<f64> {
        let mut logits = fast_abt(&rms_norm(residual, &self.final_gain, self.eps), &self.wte);
        for mut row in logits.outer_iter_mut() {
            let max = row.fold(f64::NEG_INFINITY, |m, v| m.max(*v));
            let log_total = row.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
            row.mapv_inplace(|v| v - log_total);
        }
        logits
    }
}

/// Which rows an action acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    One(usize),
    All,
}

impl Rows {
    fn has(self, row: usize) -> bool {
        match self {
            Rows::One(r) => r == row,
            Rows::All => true,
        }
    }
}

/// A native intervention's step, applied alike to every program that runs the decoder.
#[derive(Clone, Debug)]
pub enum Action {
    /// Columns `cols` of the site's input multiplied by `scale` on `rows` (a neuron, a head).
    ScaleInput { site: usize, rows: Rows, cols: (usize, usize), scale: f64 },
    /// The site's input at `row` mixed toward the donor's: `x ← (1 − α) x + α x_donor`.
    MixInput { site: usize, row: usize, alpha: f64 },
    /// The site's output at `row` mixed toward the donor's.
    MixOutput { site: usize, row: usize, alpha: f64 },
    /// `y ← y + L (Rᵀ x)` at every row (`left`: d_out × r, `right`: d_in × r): a weight edit.
    AddMap { site: usize, left: Arc<Array2<f64>>, right: Arc<Array2<f64>> },
}

impl Action {
    pub fn site(&self) -> usize {
        match self {
            Action::ScaleInput { site, .. } | Action::MixInput { site, .. } | Action::MixOutput { site, .. } | Action::AddMap { site, .. } => *site,
        }
    }

    /// The first row the action changes.
    pub fn first_row(&self) -> usize {
        match self {
            Action::ScaleInput { rows: Rows::One(r), .. } | Action::MixInput { row: r, .. } | Action::MixOutput { row: r, .. } => *r,
            Action::ScaleInput { rows: Rows::All, .. } | Action::AddMap { .. } => 0,
        }
    }

    /// The donor states the action reads: `(site, row, output?)`.
    pub fn donor_state(&self) -> Option<(usize, usize, bool)> {
        match self {
            Action::MixInput { site, row, .. } => Some((*site, *row, false)),
            Action::MixOutput { site, row, .. } => Some((*site, *row, true)),
            _ => None,
        }
    }
}

/// A program's states on a donor passage: site inputs and outputs at given rows.
#[derive(Clone, Debug, Default)]
pub struct Donor {
    pub states: BTreeMap<(usize, usize, bool), Array1<f64>>,
}

/// An explanation's selection rule, run inside its program on the program's own states.
pub trait Selector {
    /// Per row of the site's input, the units it runs and their masks.
    fn select(&mut self, site: usize, input: &Array2<f64>) -> Vec<Vec<(u32, f64)>>;
}

/// A selection given in full (to test the evaluator, or for a rule run elsewhere on the
/// program's own states): per row, the (site, unit) pairs on.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    pub rows: Vec<Vec<(u32, u32)>>,
}

impl Selection {
    /// From global unit numbers (sites in order, each site's units numbered from `offsets[site]`).
    pub fn from_global(rows: Vec<Vec<u32>>, offsets: &[usize]) -> Self {
        let rows = rows
            .into_iter()
            .map(|units| {
                units
                    .into_iter()
                    .map(|g| {
                        let site = offsets.partition_point(|o| *o <= g as usize) - 1;
                        (site as u32, g - offsets[site] as u32)
                    })
                    .collect()
            })
            .collect();
        Self { rows }
    }

    pub fn mean_selected(&self) -> f64 {
        self.rows.iter().map(Vec::len).sum::<usize>() as f64 / self.rows.len().max(1) as f64
    }
}

impl Selector for Selection {
    fn select(&mut self, site: usize, input: &Array2<f64>) -> Vec<Vec<(u32, f64)>> {
        (0..input.nrows()).map(|r| self.rows[r].iter().filter(|(k, _)| *k as usize == site).map(|(_, c)| (*c, 1.0)).collect()).collect()
    }
}

/// One site's library: `v` (units × d_in) and `u` (units × d_out).
pub struct Library {
    pub v: Array2<f64>,
    pub u: Array2<f64>,
}

/// Every site's library from `DIR/{site name}.v.f64` and `.u.f64` (an absent site has no units).
pub fn load_libraries(dir: &Path, decoder: &Decoder) -> Result<Vec<Library>, String> {
    (0..decoder.sites())
        .map(|site| {
            let (d_out, d_in) = decoder.native(site).dim();
            let name = site_name(site);
            let v_path = dir.join(format!("{name}.v.f64"));
            if !v_path.exists() {
                return Ok(Library { v: Array2::zeros((0, d_in)), u: Array2::zeros((0, d_out)) });
            }
            let v = read_f64_matrix(&v_path, d_in)?;
            let u = read_f64_matrix(&dir.join(format!("{name}.u.f64")), d_out)?;
            if u.nrows() != v.nrows() {
                return Err(format!("{name}: {} read and {} write vectors", v.nrows(), u.nrows()));
            }
            Ok(Library { v, u })
        })
        .collect()
}

/// What a program runs at its sites.
pub enum Maps<'a> {
    /// The native maps.
    Native(&'a Decoder),
    /// An explanation's units, chosen by its own rule.
    Units { libraries: &'a [Library], selector: &'a mut dyn Selector },
}

/// A program under an intervention: its maps, the actions, its own donor states, and the site
/// states to record (`(site, row, output?)`).
pub struct Program<'a> {
    pub maps: Maps<'a>,
    pub actions: &'a [Action],
    pub donor: Option<&'a Donor>,
    pub record: Vec<((usize, usize, bool), Option<Array1<f64>>)>,
}

impl Program<'_> {
    fn donor_row(&self, key: (usize, usize, bool)) -> Option<&Array1<f64>> {
        self.donor.and_then(|d| d.states.get(&key))
    }

    fn keep(&mut self, site: usize, rows: &Array2<f64>, output: bool) {
        for ((k, r, o), slot) in self.record.iter_mut() {
            if *k == site && *o == output {
                *slot = Some(rows.row(*r).to_owned());
            }
        }
    }
}

impl SiteMaps for Program<'_> {
    fn apply(&mut self, site: usize, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();
        for action in self.actions.iter().filter(|a| a.site() == site) {
            match action {
                Action::ScaleInput { rows, cols: (a, b), scale, .. } => {
                    for (r, mut row) in x.outer_iter_mut().enumerate() {
                        if rows.has(r) {
                            row.slice_mut(s![*a..*b]).mapv_inplace(|v| v * scale);
                        }
                    }
                }
                Action::MixInput { row, alpha, .. } => {
                    if let Some(d) = self.donor_row((site, *row, false)) {
                        let mixed = &x.row(*row) * (1.0 - alpha) + d * *alpha;
                        x.row_mut(*row).assign(&mixed);
                    }
                }
                _ => {}
            }
        }
        self.keep(site, &x, false);
        let mut y = match &mut self.maps {
            Maps::Native(decoder) => fast_abt(&x, decoder.native(site)),
            Maps::Units { libraries, selector } => {
                let library = &libraries[site];
                let chosen = selector.select(site, &x);
                let mut y = Array2::<f64>::zeros((x.nrows(), library.u.ncols()));
                for ((mut out, xr), units) in y.outer_iter_mut().zip(x.outer_iter()).zip(&chosen) {
                    for &(c, mask) in units {
                        let c = c as usize;
                        out.scaled_add(mask * library.v.row(c).dot(&xr), &library.u.row(c));
                    }
                }
                y
            }
        };
        for action in self.actions.iter().filter(|a| a.site() == site) {
            match action {
                Action::MixOutput { row, alpha, .. } => {
                    if let Some(d) = self.donor_row((site, *row, true)) {
                        let mixed = &y.row(*row) * (1.0 - alpha) + d * *alpha;
                        y.row_mut(*row).assign(&mixed);
                    }
                }
                Action::AddMap { left, right, .. } => y += &fast_abt(&fast_ab(&x, right.as_ref()), left.as_ref()),
                _ => {}
            }
        }
        self.keep(site, &y, true);
        y
    }
}

/// Per row, `KL(p ‖ q)` of two log-probability tiles.
pub fn kl_rows(p: &Array2<f64>, q: &Array2<f64>) -> Vec<f64> {
    p.outer_iter().zip(q.outer_iter()).map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x.exp() * (x - y)).sum()).collect()
}

/// Whether each row's most likely token agrees.
pub fn top1_rows(p: &Array2<f64>, q: &Array2<f64>) -> Vec<bool> {
    let argmax = |row: ArrayView1<f64>| row.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
    p.outer_iter().zip(q.outer_iter()).map(|(a, b)| argmax(a) == argmax(b)).collect()
}

/// One episode's scores.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Scores {
    /// Mean `KL(native ‖ explanation)` over the output rows from the first changed row on.
    pub kl: f64,
    /// Mean `KL(native ‖ native clean)` over the same rows.
    pub native_effect: f64,
    /// Share of those rows whose top token agrees.
    pub top1_agree: f64,
    /// Per layer, mean `KL(native ‖ explanation)` of the residual after it, read through the
    /// final norm and unembedding, at the interface rows.
    pub interface_kl: Vec<f64>,
    /// Per layer, the native effect read the same way.
    pub interface_effect: Vec<f64>,
}

/// Scores of a native and an explanation forward of one episode (rows from `from` on, `tile` at
/// a time) against the native clean forward, whose layers are kept at every row.
pub fn score(decoder: &Decoder, native: &Forward, explained: &Forward, clean: &Forward, interface_rows: &[usize], from: usize, tile: usize) -> Scores {
    let rows = native.residual.nrows();
    let (mut kl, mut effect, mut agree) = (0.0, 0.0, 0.0);
    let mut start = from;
    while start < rows {
        let end = (start + tile).min(rows);
        let p = decoder.log_probs(&native.residual.slice(s![start..end, ..]).to_owned());
        let q = decoder.log_probs(&explained.residual.slice(s![start..end, ..]).to_owned());
        let c = decoder.log_probs(&clean.residual.slice(s![start..end, ..]).to_owned());
        kl += kl_rows(&p, &q).iter().sum::<f64>();
        effect += kl_rows(&p, &c).iter().sum::<f64>();
        agree += top1_rows(&p, &q).iter().filter(|a| **a).count() as f64;
        start = end;
    }
    let n = (rows - from).max(1) as f64;
    let mean = |v: Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let (mut interface_kl, mut interface_effect) = (Vec::new(), Vec::new());
    for l in 0..native.layers.len() {
        let p = decoder.log_probs(&native.layers[l]);
        let q = decoder.log_probs(&explained.layers[l]);
        let c = decoder.log_probs(&clean.layers[l].select(Axis(0), interface_rows));
        interface_kl.push(mean(kl_rows(&p, &q)));
        interface_effect.push(mean(kl_rows(&p, &c)));
    }
    Scores { kl: kl / n, native_effect: effect / n, top1_agree: agree / n, interface_kl, interface_effect }
}
