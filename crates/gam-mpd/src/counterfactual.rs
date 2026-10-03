//! Counterfactual response (#2951): does an explanation predict how the native model responds to
//! a declared intervention?
//!
//! An explanation of a decoder is a library of rank-one units per decomposed site (`u_c v_cᵀ`)
//! and a selection rule naming, for every input, the units it runs. Its program replaces each
//! site's map `x ↦ W x` by `x_r ↦ Σ_{c ∈ S_r} m_{rc} u_c (v_cᵀ x_r)` over row `r`'s selected
//! units (masks `m`, one by default), and runs everything else of the decoder unchanged.
//!
//! An episode declares one intervention on the native model and its image in the explanation's
//! own terms ([`Intervention`]):
//! * clean: nothing;
//! * a unit `c` of site `k` at row `r` scaled by `s` (`s = 0` removes it): natively
//!   `W ← W + (s − 1) u_c v_cᵀ` at that row only; the explanation multiplies the unit's mask
//!   there by `s` (a unit it does not select stays absent);
//! * site `k`'s input at row `r` replaced by its input at the same row of a donor passage:
//!   natively the native donor input, in the explanation its own program's donor input (run on
//!   the donor with the donor's selection);
//! * a weight edit `W ← W + L Rᵀ` of site `k` at every row, whose image in the explanation is a
//!   unit's write vector replaced (`u_c ← w`).
//!
//! The selection under the intervention is the explanation's own rule; the caller supplies the
//! sets it chose (so the rule may be any program). The score is the disagreement
//! `KL(native ‖ explanation)` of the next-token distributions on the rows the intervention can
//! reach (from its row on; every row for clean and edit episodes), beside the native effect
//! `KL(native intervened ‖ native clean)` on the same rows, which says how much there was to
//! predict.
//!
//! The decoder is the LlamaSimpleMLP of VPD's 4-layer Pile target as exported by
//! `gam_mpd::import`'s language-model exports: pre-RMS-norm blocks, rotate-half rotary causal
//! attention over the whole head, a tanh-GELU MLP, a final RMS norm and the tied unembedding, all
//! in binary64.

use std::path::Path;

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

fn read_tensor(dir: &Path, name: &str, shape: &[usize]) -> Result<Array2<f64>, String> {
    let path = dir.join(format!("{name}.f64"));
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (rows, cols) = (shape[0], shape.get(1).copied().unwrap_or(1));
    if bytes.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for shape {shape:?}", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|e| e.to_string())
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
        let vector = |name: &str| -> Result<Array1<f64>, String> { Ok(read_tensor(dir, name, &[1, d_model])?.row(0).to_owned()) };
        let mut blocks = Vec::new();
        for l in 0..layers {
            let shapes = [[d_model, d_model], [d_model, d_model], [d_model, d_model], [d_model, d_model], [d_mlp, d_model], [d_model, d_mlp]];
            let maps = STORAGE.iter().zip(shapes).map(|(s, shape)| read_tensor(dir, &format!("blocks.{l}.{s}"), &shape)).collect::<Result<_, _>>()?;
            blocks.push(Block { rms1: vector(&format!("blocks.{l}.rms1.gain"))?, rms2: vector(&format!("blocks.{l}.rms2.gain"))?, maps });
        }
        Ok(Self {
            wte: read_tensor(dir, "wte", &[vocab, d_model])?,
            final_gain: vector("final_norm.gain")?,
            blocks,
            heads,
            eps: get("norm_eps")?,
            inv_freq,
        })
    }

    pub fn sites(&self) -> usize {
        self.blocks.len() * KINDS.len()
    }

    /// Site `site`'s native map `W` (d_out × d_in).
    pub fn native(&self, site: usize) -> &Array2<f64> {
        &self.blocks[site / KINDS.len()].maps[site % KINDS.len()]
    }

    fn rotate(&self, x: &mut Array2<f64>) {
        let head_dim = x.ncols() / self.heads;
        let half = head_dim / 2;
        for (pos, mut row) in x.outer_iter_mut().enumerate() {
            for h in 0..self.heads {
                for i in 0..half {
                    let (sin, cos) = (pos as f64 * self.inv_freq[i]).sin_cos();
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

    /// The residual rows after the last block (before the final norm), every decomposed map
    /// computed by `maps`.
    pub fn residual(&self, tokens: &[u32], maps: &mut dyn SiteMaps) -> Array2<f64> {
        let mut x = self.wte.select(Axis(0), &tokens.iter().map(|t| *t as usize).collect::<Vec<_>>());
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
        }
        x
    }

    /// Next-token log-probabilities of residual rows (a tile at a time keeps `rows × vocab` small).
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

/// The native maps, with every site's input recorded when asked.
pub struct Native<'a> {
    pub decoder: &'a Decoder,
    /// Site inputs to record (site → its input rows), filled by the forward.
    pub record: Vec<(usize, Option<Array2<f64>>)>,
    pub intervention: NativeChange,
}

/// The native image of an intervention.
#[derive(Clone, Debug, Default)]
pub enum NativeChange {
    #[default]
    None,
    /// `W ← W + (scale − 1) u vᵀ` at one row.
    Unit { site: usize, row: usize, u: Array1<f64>, v: Array1<f64>, scale: f64 },
    /// The site's input at one row replaced.
    Input { site: usize, row: usize, input: Array1<f64> },
    /// `W ← W + L Rᵀ` at every row (`left`: d_out × r, `right`: d_in × r).
    Edit { site: usize, left: Array2<f64>, right: Array2<f64> },
}

impl SiteMaps for Native<'_> {
    fn apply(&mut self, site: usize, input: &Array2<f64>) -> Array2<f64> {
        let patched;
        let mut x = input;
        if let NativeChange::Input { site: k, row, input: patch } = &self.intervention
            && *k == site
        {
            let mut copy = input.clone();
            copy.row_mut(*row).assign(patch);
            patched = copy;
            x = &patched;
        }
        for (k, slot) in self.record.iter_mut() {
            if *k == site {
                *slot = Some(x.clone());
            }
        }
        let w = self.decoder.native(site);
        let mut y = fast_abt(x, w);
        match &self.intervention {
            NativeChange::Unit { site: k, row, u, v, scale } if *k == site => {
                let a = (scale - 1.0) * v.dot(&x.row(*row));
                y.row_mut(*row).scaled_add(a, u);
            }
            NativeChange::Edit { site: k, left, right } if *k == site => {
                y += &fast_abt(&fast_ab(x, right), left);
            }
            _ => {}
        }
        y
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

/// An explanation's selection on one passage: per row, the (site, unit, mask) it runs, sorted by
/// site.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    pub rows: Vec<Vec<(u32, u32, f64)>>,
}

impl Selection {
    /// From global unit numbers (sites in order, each site's units numbered from `offsets[site]`),
    /// every mask one.
    pub fn from_global(rows: Vec<Vec<u32>>, offsets: &[usize]) -> Self {
        let rows = rows
            .into_iter()
            .map(|units| {
                units
                    .into_iter()
                    .map(|g| {
                        let site = offsets.partition_point(|o| *o <= g as usize) - 1;
                        (site as u32, g - offsets[site] as u32, 1.0)
                    })
                    .collect()
            })
            .collect();
        Self { rows }
    }

    /// Whether `unit` of `site` is selected at `row`.
    pub fn has(&self, row: usize, site: usize, unit: usize) -> bool {
        self.rows[row].iter().any(|(k, c, _)| *k as usize == site && *c as usize == unit)
    }
}

/// The explanation's program: its libraries run on its selection.
pub struct Explained<'a> {
    pub libraries: &'a [Library],
    pub selection: &'a Selection,
    pub record: Vec<(usize, Option<Array2<f64>>)>,
    pub intervention: ExplanationChange,
}

/// The explanation's image of an intervention.
#[derive(Clone, Debug, Default)]
pub enum ExplanationChange {
    #[default]
    None,
    /// The unit's mask at one row multiplied by `scale`.
    Unit { site: usize, row: usize, unit: usize, scale: f64 },
    /// The site's input at one row replaced.
    Input { site: usize, row: usize, input: Array1<f64> },
    /// A unit's write vector replaced at every row.
    Write { site: usize, unit: usize, write: Array1<f64> },
}

impl SiteMaps for Explained<'_> {
    fn apply(&mut self, site: usize, input: &Array2<f64>) -> Array2<f64> {
        let patched;
        let mut x = input;
        if let ExplanationChange::Input { site: k, row, input: patch } = &self.intervention
            && *k == site
        {
            let mut copy = input.clone();
            copy.row_mut(*row).assign(patch);
            patched = copy;
            x = &patched;
        }
        for (k, slot) in self.record.iter_mut() {
            if *k == site {
                *slot = Some(x.clone());
            }
        }
        let library = &self.libraries[site];
        let mut y = Array2::<f64>::zeros((x.nrows(), library.u.ncols()));
        for (r, (mut out, units)) in y.outer_iter_mut().zip(&self.selection.rows).enumerate() {
            let xr = x.row(r);
            for &(k, c, mask) in units.iter().filter(|(k, _, _)| *k as usize == site) {
                let c = c as usize;
                let mut m = mask;
                if let ExplanationChange::Unit { site: s, row, unit, scale } = &self.intervention
                    && *s == k as usize
                    && *row == r
                    && *unit == c
                {
                    m *= scale;
                }
                let a = m * library.v.row(c).dot(&xr);
                let write: ArrayView1<f64> = match &self.intervention {
                    ExplanationChange::Write { site: s, unit, write } if *s == site && *unit == c => write.view(),
                    _ => library.u.row(c),
                };
                out.scaled_add(a, &write);
            }
        }
        y
    }
}

/// Per row of `rows`, `KL(p ‖ q)` of two log-probability tiles.
pub fn kl_rows(p: &Array2<f64>, q: &Array2<f64>) -> Vec<f64> {
    p.outer_iter().zip(q.outer_iter()).map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x.exp() * (x - y)).sum()).collect()
}

/// Whether each row's most likely token agrees.
pub fn top1_rows(p: &Array2<f64>, q: &Array2<f64>) -> Vec<bool> {
    let argmax = |row: ArrayView1<f64>| row.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
    p.outer_iter().zip(q.outer_iter()).map(|(a, b)| argmax(a) == argmax(b)).collect()
}

/// One episode's scores on its rows: the disagreement `KL(native ‖ explanation)`, the native
/// effect `KL(native ‖ native clean)`, and the share of rows whose top token agrees, all means
/// over the rows from `from` on, the next-token distributions computed `tile` rows at a time.
pub fn score(decoder: &Decoder, native: &Array2<f64>, explained: &Array2<f64>, native_clean: &Array2<f64>, from: usize, tile: usize) -> (f64, f64, f64) {
    let rows = native.nrows();
    let (mut kl, mut effect, mut agree) = (0.0, 0.0, 0.0);
    let mut start = from;
    while start < rows {
        let end = (start + tile).min(rows);
        let p = decoder.log_probs(&native.slice(s![start..end, ..]).to_owned());
        let q = decoder.log_probs(&explained.slice(s![start..end, ..]).to_owned());
        let c = decoder.log_probs(&native_clean.slice(s![start..end, ..]).to_owned());
        kl += kl_rows(&p, &q).iter().sum::<f64>();
        effect += kl_rows(&p, &c).iter().sum::<f64>();
        agree += top1_rows(&p, &q).iter().filter(|a| **a).count() as f64;
        start = end;
    }
    let n = (rows - from).max(1) as f64;
    (kl / n, effect / n, agree / n)
}
