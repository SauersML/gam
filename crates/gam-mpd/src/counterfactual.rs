//! Counterfactual response (#2951): the native model's response to declared native interventions.
//!
//! Interventions are native and decomposition-free ([`Action`]): they act on the decoder's
//! physical quantities, a site's input or output rows (a neuron is a coordinate of the MLP
//! down-projection's input; a head is a block of the attention output projection's input), or
//! add a weight edit's map to a site's output. An action at one row is activation-level; one at
//! every row (a scaled neuron's or head's weights, a compiled edit) is weight-level. Mixing toward
//! a donor passage takes the donor's state from the same program on the donor ([`Donor`]).
//!
//! [`Program`] runs the native maps under an episode's actions; `run_check` applies the same
//! actions to a decoded artifact at the places it holds and scores both forwards with [`score`]:
//! the disagreement `KL(native ‖ replacement)` of the next-token distributions on the rows the
//! actions can reach, beside the native effect `KL(native ‖ native clean)` there (how much there
//! was to predict), and at the declared internal interfaces, the residual stream after every
//! layer at the episode's interface rows, the same disagreement read through the final norm and
//! unembedding (the unembedding's metric on the residual).
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

use super::import::{read_f64, read_f64_shaped};
use super::llama_simple_mlp::gelu_tanh;

/// The decomposed maps of one block, in site order.
pub const KINDS: [&str; 6] = ["q", "k", "v", "o", "c_fc", "down_proj"];
const STORAGE: [&str; 6] = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"];

/// Site `kind` of block `layer`.
pub fn site_index(layer: usize, kind: usize) -> usize {
    layer * KINDS.len() + kind
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
    read_f64_shaped(&dir.join(format!("{name}.f64")), shape[0], shape[1])
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

    /// The tied embedding (vocab × d_model).
    pub fn embedding(&self) -> &Array2<f64> {
        &self.wte
    }

    pub fn final_gain(&self) -> &Array1<f64> {
        &self.final_gain
    }

    /// Block `layer`'s norm gains before its attention and its MLP.
    pub fn gains(&self, layer: usize) -> (&Array1<f64>, &Array1<f64>) {
        (&self.blocks[layer].rms1, &self.blocks[layer].rms2)
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The rotation's frequency per plane of a head.
    pub fn inv_freq(&self) -> &[f64] {
        &self.inv_freq
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

    /// One forward of `tokens`, every decomposed map computed by `program`, keeping the residual
    /// after every block at `layer_rows`.
    pub fn forward(&self, tokens: &[u32], program: &mut Program<'_>, layer_rows: &[usize]) -> Forward {
        let mut x = self.wte.select(Axis(0), &tokens.iter().map(|t| *t as usize).collect::<Vec<_>>());
        let mut layers = Vec::new();
        for (l, block) in self.blocks.iter().enumerate() {
            let n = rms_norm(&x, &block.rms1, self.eps);
            let mut q = program.apply(site_index(l, 0), &n);
            let mut k = program.apply(site_index(l, 1), &n);
            let v = program.apply(site_index(l, 2), &n);
            self.rotate(&mut q);
            self.rotate(&mut k);
            let attended = self.attend(&q, &k, &v);
            x += &program.apply(site_index(l, 3), &attended);
            let n = rms_norm(&x, &block.rms2, self.eps);
            let hidden = program.apply(site_index(l, 4), &n).mapv(gelu_tanh);
            x += &program.apply(site_index(l, 5), &hidden);
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

    fn first(self) -> usize {
        match self {
            Rows::One(r) => r,
            Rows::All => 0,
        }
    }
}

/// A change of a site's input rows.
#[derive(Clone, Debug)]
pub enum InputChange {
    /// Columns `cols` multiplied by `scale` on `rows` (a neuron, a head).
    Scale { rows: Rows, cols: (usize, usize), scale: f64 },
    /// The row mixed toward the donor's: `x ← (1 − α) x + α x_donor`.
    Mix { row: usize, alpha: f64 },
}

/// A change of a site's output rows.
#[derive(Clone, Debug)]
pub enum OutputChange {
    /// The row mixed toward the donor's.
    Mix { row: usize, alpha: f64 },
    /// `y ← y + L (Rᵀ x)` at every row (`left`: d_out × r, `right`: d_in × r): a weight edit.
    Add { left: Arc<Array2<f64>>, right: Arc<Array2<f64>> },
}

/// A native intervention's step at one site, applied alike to every program that runs the
/// decoder.
#[derive(Clone, Debug)]
pub enum Action {
    Input { site: usize, change: InputChange },
    Output { site: usize, change: OutputChange },
}

impl Action {
    pub fn site(&self) -> usize {
        match self {
            Action::Input { site, .. } | Action::Output { site, .. } => *site,
        }
    }

    /// The first row the action changes.
    pub fn first_row(&self) -> usize {
        match self {
            Action::Input { change: InputChange::Scale { rows, .. }, .. } => rows.first(),
            Action::Input { change: InputChange::Mix { row, .. }, .. } | Action::Output { change: OutputChange::Mix { row, .. }, .. } => *row,
            Action::Output { change: OutputChange::Add { .. }, .. } => 0,
        }
    }

    /// The donor state the action reads: `(site, row, output?)`.
    pub fn donor_state(&self) -> Option<(usize, usize, bool)> {
        match self {
            Action::Input { site, change: InputChange::Mix { row, .. } } => Some((*site, *row, false)),
            Action::Output { site, change: OutputChange::Mix { row, .. } } => Some((*site, *row, true)),
            Action::Input { change: InputChange::Scale { .. }, .. } | Action::Output { change: OutputChange::Add { .. }, .. } => None,
        }
    }
}

/// A program's states on a donor passage: site inputs and outputs at given rows.
#[derive(Clone, Debug, Default)]
pub struct Donor {
    pub states: BTreeMap<(usize, usize, bool), Array1<f64>>,
}

/// The native decoder under an intervention: the actions, the donor states, and the site states
/// to record (`(site, row, output?)`).
pub struct Program<'a> {
    pub decoder: &'a Decoder,
    pub actions: &'a [Action],
    pub donor: Option<&'a Donor>,
    pub record: Vec<((usize, usize, bool), Option<Array1<f64>>)>,
}

impl<'a> Program<'a> {
    pub fn new(decoder: &'a Decoder, actions: &'a [Action], donor: Option<&'a Donor>) -> Self {
        Self { decoder, actions, donor, record: Vec::new() }
    }

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

    fn mix(&self, rows: &mut Array2<f64>, key: (usize, usize, bool), alpha: f64) {
        if let Some(d) = self.donor_row(key) {
            let mixed = &rows.row(key.1) * (1.0 - alpha) + d * alpha;
            rows.row_mut(key.1).assign(&mixed);
        }
    }
}

impl Program<'_> {
    /// Site `site`'s output rows for its input rows, the actions applied.
    fn apply(&mut self, site: usize, input: &Array2<f64>) -> Array2<f64> {
        let actions = self.actions;
        let mut x = input.clone();
        let inputs = actions.iter().filter_map(|a| match a {
            Action::Input { site: k, change } if *k == site => Some(change),
            Action::Input { .. } | Action::Output { .. } => None,
        });
        for change in inputs.collect::<Vec<_>>() {
            match change {
                InputChange::Scale { rows, cols: (a, b), scale } => {
                    for (r, mut row) in x.outer_iter_mut().enumerate() {
                        if rows.has(r) {
                            row.slice_mut(s![*a..*b]).mapv_inplace(|v| v * scale);
                        }
                    }
                }
                InputChange::Mix { row, alpha } => self.mix(&mut x, (site, *row, false), *alpha),
            }
        }
        self.keep(site, &x, false);
        let mut y = fast_abt(&x, self.decoder.native(site));
        let outputs = actions.iter().filter_map(|a| match a {
            Action::Output { site: k, change } if *k == site => Some(change),
            Action::Input { .. } | Action::Output { .. } => None,
        });
        for change in outputs.collect::<Vec<_>>() {
            match change {
                OutputChange::Mix { row, alpha } => self.mix(&mut y, (site, *row, true), *alpha),
                OutputChange::Add { left, right } => y += &fast_abt(&fast_ab(&x, right.as_ref()), left.as_ref()),
            }
        }
        self.keep(site, &y, true);
        y
    }
}

/// Per row, `KL(p ‖ q)` of two log-probability tiles.
fn kl_rows(p: &Array2<f64>, q: &Array2<f64>) -> Vec<f64> {
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
    /// Mean `KL(native ‖ replacement)` over the output rows from the first changed row on.
    pub kl: f64,
    /// Mean `KL(native ‖ native clean)` over the same rows.
    pub native_effect: f64,
    /// Share of those rows whose top token agrees.
    pub top1_agree: f64,
    /// Per layer, mean `KL(native ‖ replacement)` of the residual after it, read through the
    /// final norm and unembedding, at the interface rows.
    pub interface_kl: Vec<f64>,
    /// Per layer, the native effect read the same way.
    pub interface_effect: Vec<f64>,
}

/// Scores of a native and a replacement forward of one episode (rows from `from` on, `tile` at
/// a time) against the native clean forward, whose layers are kept at every row.
pub fn score(decoder: &Decoder, native: &Forward, replacement: &Forward, clean: &Forward, interface_rows: &[usize], from: usize, tile: usize) -> Scores {
    let rows = native.residual.nrows();
    let (mut kl, mut effect, mut agree) = (0.0, 0.0, 0.0);
    let mut start = from;
    while start < rows {
        let end = (start + tile).min(rows);
        let p = decoder.log_probs(&native.residual.slice(s![start..end, ..]).to_owned());
        let q = decoder.log_probs(&replacement.residual.slice(s![start..end, ..]).to_owned());
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
        let q = decoder.log_probs(&replacement.layers[l]);
        let c = decoder.log_probs(&clean.layers[l].select(Axis(0), interface_rows));
        interface_kl.push(mean(kl_rows(&p, &q)));
        interface_effect.push(mean(kl_rows(&p, &c)));
    }
    Scores { kl: kl / n, native_effect: effect / n, top1_agree: agree / n, interface_kl, interface_effect }
}

/// One declared episode: its actions on a passage, the rows its internal interfaces are read at,
/// and the donor passage its mixes read.
#[derive(Clone, Debug)]
pub struct Episode {
    pub id: String,
    pub group: String,
    pub passage: usize,
    pub donor: Option<usize>,
    pub interface_rows: Vec<usize>,
    pub actions: Vec<Action>,
}

/// A frozen episode list (`bench/vpd_2951/counterfactual_spec.py`).
#[derive(Clone, Debug)]
pub struct Spec {
    pub rows: usize,
    pub episodes: Vec<Episode>,
}

fn field(value: &serde_json::Value, key: &str) -> Result<usize, String> {
    value[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("{value} has no {key}"))
}

impl Spec {
    /// The spec at `path`, its edits' factors read relative to it.
    pub fn load(path: &Path, decoder: &Decoder) -> Result<Self, String> {
        let spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| e.to_string())?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut edits = BTreeMap::new();
        for (name, e) in spec["edits"].as_object().into_iter().flatten() {
            let (site, rank) = (field(e, "site")?, field(e, "rank")?);
            let left = read_f64(&dir.join(e["left"].as_str().unwrap_or_default()), rank)?;
            let right = read_f64(&dir.join(e["right"].as_str().unwrap_or_default()), rank)?;
            let (d_out, d_in) = decoder.native(site).dim();
            if left.nrows() != d_out || right.nrows() != d_in {
                return Err(format!("edit {name}: left {:?}, right {:?} for a {d_out}×{d_in} site", left.dim(), right.dim()));
            }
            edits.insert(name.clone(), (site, Arc::new(left), Arc::new(right)));
        }
        let mut episodes = Vec::new();
        for e in spec["episodes"].as_array().ok_or("spec has no episodes")? {
            let mut actions = Vec::new();
            for a in e["actions"].as_array().ok_or_else(|| format!("{} has no actions", e["id"]))? {
                let site = field(a, "site")?;
                let alpha = || a["alpha"].as_f64().ok_or_else(|| format!("{a} has no alpha"));
                actions.push(match a["type"].as_str().unwrap_or_default() {
                    "scale_input" => {
                        let cols = a["cols"].as_array().ok_or("scale_input without cols")?;
                        let rows = a["row"].as_u64().map_or(Rows::All, |r| Rows::One(r as usize));
                        let cols = (cols[0].as_u64().unwrap_or(0) as usize, cols[1].as_u64().unwrap_or(0) as usize);
                        Action::Input { site, change: InputChange::Scale { rows, cols, scale: a["scale"].as_f64().ok_or("scale_input without scale")? } }
                    }
                    "mix_input" => Action::Input { site, change: InputChange::Mix { row: field(a, "row")?, alpha: alpha()? } },
                    "mix_output" => Action::Output { site, change: OutputChange::Mix { row: field(a, "row")?, alpha: alpha()? } },
                    "add_map" => {
                        let (edit_site, left, right) = edits.get(a["edit"].as_str().unwrap_or_default()).ok_or_else(|| format!("unknown edit {}", a["edit"]))?;
                        if *edit_site != site {
                            return Err(format!("edit {} is of site {edit_site}, not {site}", a["edit"]));
                        }
                        Action::Output { site, change: OutputChange::Add { left: left.clone(), right: right.clone() } }
                    }
                    other => return Err(format!("unknown action {other}")),
                });
            }
            episodes.push(Episode {
                id: e["id"].as_str().ok_or("episode without id")?.to_string(),
                group: e["group"].as_str().unwrap_or("ungrouped").to_string(),
                passage: field(e, "passage")?,
                donor: e["donor"].as_u64().map(|d| d as usize),
                interface_rows: e["interface_rows"].as_array().ok_or("episode without interface_rows")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect(),
                actions,
            });
        }
        Ok(Self { rows: field(&spec, "rows")?, episodes })
    }
}

/// The token rows of an export's passages, the first `rows` columns of each.
pub fn passages(export: &Path, rows: usize) -> Result<Vec<Vec<u32>>, String> {
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| format!("{}: {e}", export.display()))?).map_err(|e| e.to_string())?;
    let width = record["files"]["tokens"]["shape"][1].as_u64().ok_or("export has no tokens")? as usize;
    Ok(read_f64(&export.join("tokens.f64"), width)?.outer_iter().map(|r| r.iter().take(rows).map(|t| *t as u32).collect()).collect())
}
