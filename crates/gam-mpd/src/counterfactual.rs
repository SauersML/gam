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

use super::codec::{prefix_integer_len_bits, signed_delta_len_bits};
use super::explanation::Fitted;
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

/// Every site's library from `DIR/{site name}.v.f64` and `.u.f64`; a site without files has none
/// and runs its native map.
pub fn load_libraries(dir: &Path, decoder: &Decoder) -> Result<Vec<Option<Library>>, String> {
    (0..decoder.sites())
        .map(|site| {
            let (d_out, d_in) = decoder.native(site).dim();
            let name = site_name(site);
            let v_path = dir.join(format!("{name}.v.f64"));
            if !v_path.exists() {
                return Ok(None);
            }
            let v = read_f64_matrix(&v_path, d_in)?;
            let u = read_f64_matrix(&dir.join(format!("{name}.u.f64")), d_out)?;
            if u.nrows() != v.nrows() {
                return Err(format!("{name}: {} read and {} write vectors", v.nrows(), u.nrows()));
            }
            Ok(Some(Library { v, u }))
        })
        .collect()
}

/// A fitted explanation's sites by decoder site (`gam_mpd::explanation`'s names are
/// [`site_name`]'s); a decoder site none of them replaces runs native.
pub fn fitted_sites<'a>(decoder: &Decoder, fitted: &'a [Fitted]) -> Result<Vec<Option<&'a Fitted>>, String> {
    let mut by_site = vec![None; decoder.sites()];
    for f in fitted {
        let site = (0..decoder.sites()).find(|k| site_name(*k) == f.site.name).ok_or_else(|| format!("{}: not a site of the decoder", f.site.name))?;
        if f.w != *decoder.native(site) {
            return Err(format!("{}: fitted on another map than the decoder's", f.site.name));
        }
        by_site[site] = Some(f);
    }
    Ok(by_site)
}

/// The fitted sites' libraries, every block's columns as units.
pub fn fitted_libraries(sites: &[Option<&Fitted>]) -> Vec<Option<Library>> {
    sites.iter().map(|f| f.map(|f| Library { v: f.library.v.clone(), u: f.library.u.clone() })).collect()
}

/// A fitted explanation's own rule ([`Fitted::select`]) run inside the program: a site's blocks
/// chosen from the program's own read of it, every column of a chosen block on.
pub struct FittedRule<'a> {
    pub sites: &'a [Option<&'a Fitted>],
}

impl Selector for FittedRule<'_> {
    fn select(&mut self, site: usize, input: &Array2<f64>) -> Vec<Vec<(u32, f64)>> {
        let Some(fitted) = self.sites.get(site).copied().flatten() else {
            return vec![Vec::new(); input.nrows()];
        };
        let starts: Vec<usize> = fitted
            .ranks
            .iter()
            .scan(0, |at, r| {
                *at += r;
                Some(*at - r)
            })
            .collect();
        fitted
            .select(input)
            .outer_iter()
            .map(|on| on.iter().enumerate().filter(|(_, m)| **m == 1.0).flat_map(|(b, _)| (starts[b]..starts[b] + fitted.ranks[b]).map(|c| (c as u32, 1.0))).collect())
            .collect()
    }
}

/// What a selection rule costs beyond its library.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RulePrice {
    /// Every parameter sent as `round(w 2^p)`.
    pub precision: i32,
    pub reals: usize,
    pub bits: f64,
    /// The site whose selections changed one bit coarser (none when `p = 0`).
    pub binding: Option<String>,
    /// The reads the selections were compared on, per site.
    pub reads: usize,
}

/// A fitted rule's own parameters beyond its library, each site's map `W`, its mean written
/// Fisher (the upper triangle) and its blocks' prices, from which [`super::site_fit::Selector`]
/// forms its selection, sent at the lowest uniform lattice precision `2^-p` (`p ∈ [0, 40]`) under
/// which every site's selection of `reads[k]` (the site's reads on the test passages, rows ×
/// d_in) is unchanged in every entry; each lattice integer in the signed Elias δ code and `p + 1`
/// once in the prefix integer code. The pricing `bench/vpd_2951/vpd_rule_precision.py` gives VPD's
/// causal-importance network. Keeping the selections is not monotone in `p` (a coarser lattice
/// can round a borderline read back), so `p` rises from 0 until every site keeps them, each `p`
/// first trying the site that last failed.
pub fn rule_price(sites: &[&Fitted], reads: &[Array2<f64>], observations: f64) -> Result<RulePrice, String> {
    use rayon::prelude::*;
    const HIGHEST: i32 = 40;
    if sites.len() != reads.len() {
        return Err(format!("{} sites, {} read tables", sites.len(), reads.len()));
    }
    let on_lattice = |w: f64, p: i32| (w * 2f64.powi(p)).round() / 2f64.powi(p);
    let keeps = |k: usize, reference: &Array2<f64>, p: i32| -> Result<bool, String> {
        let f = sites[k];
        let bits: Vec<f64> = f.bits.iter().map(|b| on_lattice(*b, p)).collect();
        let rule = super::site_fit::Selector::new(&f.w.mapv(|w| on_lattice(w, p)), &f.fisher.mapv(|w| on_lattice(w, p)), &f.library, &f.ranks, &bits, observations)?;
        Ok(rule.select(&reads[k]) == *reference)
    };
    let references: Vec<Array2<f64>> = sites.par_iter().zip(reads).map(|(f, x)| f.select(x)).collect();
    let mut order: Vec<usize> = (0..sites.len()).collect();
    let (mut precision, mut binding) = (0, None);
    'coarsest: loop {
        for at in 0..order.len() {
            let k = order[at];
            if !keeps(k, &references[k], precision)? {
                if precision == HIGHEST {
                    return Err(format!("{}: the selections differ even at p = {HIGHEST}", sites[k].site.name));
                }
                order[..=at].rotate_right(1);
                binding = Some(sites[k].site.name.clone());
                precision += 1;
                continue 'coarsest;
            }
        }
        break;
    }
    let scale = 2f64.powi(precision);
    let length = |w: f64| -> Result<f64, String> { Ok(signed_delta_len_bits((w * scale).round() as i64).map_err(|e| e.to_string())? as f64) };
    let (mut bits, mut reals) = (prefix_integer_len_bits(precision as u64 + 1).map_err(|e| e.to_string())? as f64, 0);
    for f in sites {
        for w in f.w.iter().chain(f.bits.iter()) {
            bits += length(*w)?;
        }
        for i in 0..f.fisher.nrows() {
            for w in f.fisher.row(i).iter().skip(i) {
                bits += length(*w)?;
            }
        }
        let d = f.fisher.nrows();
        reals += f.w.len() + f.bits.len() + d * (d + 1) / 2;
    }
    Ok(RulePrice { precision, reals, bits, binding, reads: reads.first().map_or(0, |x| x.nrows()) })
}

/// What a program runs at its sites.
pub enum Maps<'a> {
    /// The native maps.
    Native(&'a Decoder),
    /// An explanation's units, chosen by its own rule, at the sites it has a library for; its
    /// other sites run their native maps.
    Units { decoder: &'a Decoder, libraries: &'a [Option<Library>], selector: &'a mut dyn Selector },
}

/// A program under an intervention: its maps, the actions, its own donor states, and the site
/// states to record (`(site, row, output?)`). `selected` counts the units its rule ran.
pub struct Program<'a> {
    pub maps: Maps<'a>,
    pub actions: &'a [Action],
    pub donor: Option<&'a Donor>,
    pub record: Vec<((usize, usize, bool), Option<Array1<f64>>)>,
    pub selected: usize,
}

impl<'a> Program<'a> {
    pub fn new(maps: Maps<'a>, actions: &'a [Action], donor: Option<&'a Donor>) -> Self {
        Self { maps, actions, donor, record: Vec::new(), selected: 0 }
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

impl SiteMaps for Program<'_> {
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
        let (mut y, ran) = match &mut self.maps {
            Maps::Native(decoder) => (fast_abt(&x, decoder.native(site)), 0),
            Maps::Units { decoder, libraries, selector } => match &libraries[site] {
                None => (fast_abt(&x, decoder.native(site)), 0),
                Some(library) => {
                    let chosen = selector.select(site, &x);
                    let mut y = Array2::<f64>::zeros((x.nrows(), library.u.ncols()));
                    for ((mut out, xr), units) in y.outer_iter_mut().zip(x.outer_iter()).zip(&chosen) {
                        for &(c, mask) in units {
                            let c = c as usize;
                            out.scaled_add(mask * library.v.row(c).dot(&xr), &library.u.row(c));
                        }
                    }
                    (y, chosen.iter().map(Vec::len).sum::<usize>())
                }
            },
        };
        self.selected += ran;
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
            let left = read_f64_matrix(&dir.join(e["left"].as_str().unwrap_or_default()), rank)?;
            let right = read_f64_matrix(&dir.join(e["right"].as_str().unwrap_or_default()), rank)?;
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

/// An explanation under evaluation: its libraries (a site without one runs native) and, per
/// episode id, a fresh instance of its selection rule. A donor passage runs the rule of the
/// passage's clean episode, `clean/{passage}`.
pub struct Explanation<'a> {
    pub libraries: &'a [Option<Library>],
    pub selector: Box<dyn Fn(&str) -> Result<Box<dyn Selector + 'a>, String> + Sync + 'a>,
}

/// One episode's scores, with the units per row the explanation's rule ran.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Scored {
    pub id: String,
    pub group: String,
    pub passage: usize,
    pub from_row: usize,
    pub selected_per_row: f64,
    #[serde(flatten)]
    pub scores: Scores,
}

/// Every episode of `spec` on `passages` (token rows), the explanation's program against the
/// native model's (`None` scores the native model as its own explanation, a check of the
/// evaluator), `tile` output rows at a time; episodes run in parallel. A donor passage runs once
/// per program, recording every state any episode mixing toward it reads.
pub fn evaluate(decoder: &Decoder, spec: &Spec, passages: &[Vec<u32>], explanation: Option<&Explanation<'_>>, tile: usize) -> Result<Vec<Scored>, String> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let started = std::time::Instant::now();
    let all_rows: Vec<usize> = (0..spec.rows).collect();
    let named: std::collections::BTreeSet<usize> = spec.episodes.iter().map(|e| e.passage).collect();
    let clean: BTreeMap<usize, Forward> = named
        .par_iter()
        .map(|&p| {
            let mut native = Program::new(Maps::Native(decoder), &[], None);
            (p, decoder.forward(&passages[p], &mut native, &all_rows))
        })
        .collect();
    let mut wanted: BTreeMap<usize, Vec<(usize, usize, bool)>> = BTreeMap::new();
    for e in &spec.episodes {
        let keys: Vec<(usize, usize, bool)> = e.actions.iter().filter_map(Action::donor_state).collect();
        if keys.is_empty() {
            continue;
        }
        let read = wanted.entry(e.donor.ok_or_else(|| format!("{}: a mix without a donor", e.id))?).or_default();
        read.extend(keys.into_iter().filter(|k| !read.contains(k)).collect::<Vec<_>>());
    }
    let record = |maps: Maps<'_>, d: usize, keys: &[(usize, usize, bool)]| -> Result<Donor, String> {
        let mut program = Program::new(maps, &[], None);
        program.record = keys.iter().map(|k| (*k, None)).collect();
        decoder.forward(&passages[d], &mut program, &[]);
        Ok(Donor { states: program.record.into_iter().map(|(k, v)| v.map(|v| (k, v)).ok_or("donor state not reached")).collect::<Result<_, _>>()? })
    };
    let native_donors: BTreeMap<usize, Donor> = wanted.par_iter().map(|(d, keys)| Ok((*d, record(Maps::Native(decoder), *d, keys)?))).collect::<Result<_, String>>()?;
    let own_donors: BTreeMap<usize, Donor> = match explanation {
        Some(x) => wanted
            .par_iter()
            .map(|(d, keys)| {
                let mut rule = (x.selector)(&format!("clean/{d}"))?;
                Ok((*d, record(Maps::Units { decoder, libraries: x.libraries, selector: rule.as_mut() }, *d, keys)?))
            })
            .collect::<Result<_, String>>()?,
        None => BTreeMap::new(),
    };
    let none = Donor::default();
    let done = AtomicUsize::new(0);
    spec.episodes
        .par_iter()
        .map(|e| -> Result<Scored, String> {
            let native_donor = e.donor.and_then(|d| native_donors.get(&d)).unwrap_or(&none);
            let own_donor = e.donor.and_then(|d| own_donors.get(&d)).unwrap_or(&none);
            let mut native = Program::new(Maps::Native(decoder), &e.actions, Some(native_donor));
            let native_forward = decoder.forward(&passages[e.passage], &mut native, &e.interface_rows);
            let (explained, selected) = match explanation {
                Some(x) => {
                    let mut rule = (x.selector)(&e.id)?;
                    let mut program = Program::new(Maps::Units { decoder, libraries: x.libraries, selector: rule.as_mut() }, &e.actions, Some(own_donor));
                    let forward = decoder.forward(&passages[e.passage], &mut program, &e.interface_rows);
                    (forward, program.selected as f64 / spec.rows as f64)
                }
                None => {
                    let mut program = Program::new(Maps::Native(decoder), &e.actions, Some(native_donor));
                    (decoder.forward(&passages[e.passage], &mut program, &e.interface_rows), f64::NAN)
                }
            };
            let from = e.actions.iter().map(Action::first_row).min().unwrap_or(0);
            let scores = score(decoder, &native_forward, &explained, &clean[&e.passage], &e.interface_rows, from, tile);
            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
            if finished % 100 == 0 {
                log::info!("{finished}/{} episodes, {:.0}s", spec.episodes.len(), started.elapsed().as_secs_f64());
            }
            Ok(Scored { id: e.id.clone(), group: e.group.clone(), passage: e.passage, from_row: from, selected_per_row: selected, scores })
        })
        .collect()
}

/// Means per group.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GroupSummary {
    pub episodes: usize,
    pub kl: f64,
    pub native_effect: f64,
    pub top1_agree: f64,
    pub selected_per_row: f64,
    pub interface_kl: Vec<f64>,
    pub interface_effect: Vec<f64>,
}

pub fn summarize(scored: &[Scored]) -> BTreeMap<String, GroupSummary> {
    let mut groups: BTreeMap<String, Vec<&Scored>> = BTreeMap::new();
    for s in scored {
        groups.entry(s.group.clone()).or_default().push(s);
    }
    groups
        .into_iter()
        .map(|(g, ss)| {
            let n = ss.len() as f64;
            let mean = |f: &dyn Fn(&Scored) -> f64| ss.iter().map(|s| f(s)).sum::<f64>() / n;
            let layers = ss.first().map_or(0, |s| s.scores.interface_kl.len());
            let per_layer = |f: &dyn Fn(&Scored) -> &Vec<f64>| (0..layers).map(|l| ss.iter().map(|s| f(s)[l]).sum::<f64>() / n).collect();
            let summary = GroupSummary {
                episodes: ss.len(),
                kl: mean(&|s| s.scores.kl),
                native_effect: mean(&|s| s.scores.native_effect),
                top1_agree: mean(&|s| s.scores.top1_agree),
                selected_per_row: mean(&|s| s.selected_per_row),
                interface_kl: per_layer(&|s| &s.scores.interface_kl),
                interface_effect: per_layer(&|s| &s.scores.interface_effect),
            };
            (g, summary)
        })
        .collect()
}

/// The token rows of an export's passages, the first `rows` columns of each.
pub fn passages(export: &Path, rows: usize) -> Result<Vec<Vec<u32>>, String> {
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| format!("{}: {e}", export.display()))?).map_err(|e| e.to_string())?;
    let width = record["files"]["tokens"]["shape"][1].as_u64().ok_or("export has no tokens")? as usize;
    Ok(read_f64_matrix(&export.join("tokens.f64"), width)?.outer_iter().map(|r| r.iter().take(rows).map(|t| *t as u32).collect()).collect())
}
