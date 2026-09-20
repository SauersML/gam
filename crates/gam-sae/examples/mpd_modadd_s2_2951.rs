//! mpd-modadd S2: plane rotation edits of the modular-addition transformer's embedding, fit at one
//! shift on training pairs and evaluated at every held-out shift on test pairs (#2951).
//!
//! `mpd_modadd_s2_2951 --run RUN --settings SETTINGS --out REPORT`
//!
//! `RUN` is what `bench/mpd_modadd_2951.py execute` writes for stage `plane_edits`: `export.json`,
//! and for each model `label` every parameter `<name>.<label>.npy` in its trained `<f4` (widened
//! here to binary64, exact for every `f32`) and the `p² × p` float64 torch logits
//! `logits64.<label>.npy`. `SETTINGS` is `{"stage": "plane_edits", "models": [...], "fit_shift": 1,
//! "fraction_bits": b, "null_edit_models": [label, ...], "random_basis_seed": seed}`; the driver
//! reads `models`.
//!
//! # The executor
//!
//! The one-layer model runs here in binary64: `x_u = W_E[t_u] + W_pos[u]` with `t = (a, b, p)`, one
//! causal attention layer, the ReLU MLP and `W_U`, all at `=`, which is the only row the logits
//! read. Each use site `u` reads its own table, so a pos0 edit and a global edit are different
//! experiments. Two executor controls run first: the unedited logits against torch's float64
//! forward over every pair, and the exact row-permutation edit at every site against the
//! shifted-token reference, which must agree bit for bit because the gathered rows are the same
//! numbers.
//!
//! # The fit
//!
//! `cyclic_action` reads the closed-form planes of `W_E` under the declared cycle `a -> a + 1 mod
//! p`, with row `p` (the `=` token) fixed. Candidate programs are the prefixes of the planes in
//! descending Parseval power, `K = 0 .. m`, `m = (p − 1)/2`. Each candidate's basis is sent as its
//! `PlaneProgramCode` at the declared precision and executed DECODED: the edit at shift `s` is
//! `rotation_edit` of the decoded basis. At the fit shift, on training pairs only, a candidate costs
//! its program bits plus the excess code length of the native shifted outputs under the edited
//! model, `Σ_rows KL(native shifted ‖ edited) / ln 2` (one draw per row). The selected program
//! minimizes that total, per use site. `K = 0` is the unedited model, which sends only its empty
//! subset.
//!
//! # The held-out evaluation
//!
//! The selected program's angles at shift `s` come from `p`, `k` and `s` alone, so nothing is refit.
//! At every shift `s ∈ 2 .. p − 1`, on test pairs only, the report gives `KL(native shifted ‖
//! edited)` quantiles and agreement of argmax for the selected program, for the full program (every
//! plane) and for the unedited model against the same references, plus the table-level
//! `shift_residual` bounds of both programs.
//!
//! The full program reproduces the shift for ANY table of full row rank with `p ≤ d` (the rank
//! caveat on #2951), so held-out fidelity alone is no evidence of a mechanism. The separating score
//! is the selected program's code at its fidelity against the full program's, read across the
//! trained model and the controls the settings declare (random init, random labels, shuffled MLP
//! connections, an early checkpoint). On the models `null_edit_models` names, two null edits carry
//! the selected program's size: its lowest-power planes projected off the selected span, with their
//! own angles, against the UNSHIFTED reference (it should change nothing), and a Haar-random basis
//! with the selected angles against the shifted reference (it should not shift).

use gam_linalg::faer_ndarray::FaerQr;
use gam_sae::parameter_decomposition::codec::subset_code_len_bits;
use gam_sae::parameter_decomposition::cyclic_action::{
    CyclicPlanes, CyclicRotationEdit, RowCycle, cyclic_planes, plane_program_code, rotation_edit,
    shift_residual,
};
use gam_sae::parameter_decomposition::precision::{DecodableArtifact, DeclaredPrecision};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::f64::consts::{LN_2, PI};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "support/npy_header.rs"]
mod npy_header;
use npy_header::{NpyFloat, parse_npy_float_header, parse_npy_header};

const USAGE: &str = "usage: mpd_modadd_s2_2951 --run RUN --settings SETTINGS --out REPORT";

/// The fields of the settings this example reads; the driver reads `models`.
#[derive(Deserialize)]
struct Settings {
    stage: String,
    fit_shift: usize,
    fraction_bits: i32,
    null_edit_models: Vec<String>,
    random_basis_seed: u64,
}

#[derive(Deserialize)]
struct Export {
    stage: String,
    exports: Vec<ModelExport>,
}

#[derive(Deserialize)]
struct ModelExport {
    label: String,
    step: u64,
    shuffle_mlp_seed: Option<u64>,
    config: Config,
    train_idx: Vec<usize>,
    test_idx: Vec<usize>,
    train_acc: f64,
    test_acc: f64,
}

#[derive(Deserialize)]
struct Config {
    p: usize,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    d_mlp: usize,
    labels: String,
}

/// The use sites, each the set of token positions whose lookup reads the edited table.
const SITES: [(&str, &[usize]); 4] = [
    ("pos0", &[0]),
    ("pos1", &[1]),
    ("operands", &[0, 1]),
    ("global", &[0, 1, 2]),
];

fn flag(args: &[String], name: &str) -> Result<PathBuf, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {name}; {USAGE}"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// A two-axis `<f4` array in its trained dtype, widened to binary64 (exact for every `f32`).
fn read_f4_matrix(path: &Path) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let (rows, cols, width, is_f4, data_off) = parse_npy_header(&bytes, path)?;
    if !is_f4 {
        return Err(format!("{}: expected the trained <f4 values", path.display()));
    }
    let values = f4_values(&bytes, data_off, rows * cols * width, path)?;
    Array2::from_shape_vec((rows, cols), values).map_err(|error| format!("{}: {error}", path.display()))
}

/// A one-axis `<f4` array, widened to binary64.
fn read_f4_vector(path: &Path) -> Result<Array1<f64>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let header = parse_npy_float_header(&bytes, path)?;
    let [length] = header.shape[..] else {
        return Err(format!("{}: expected one axis, got {:?}", path.display(), header.shape));
    };
    if header.float != NpyFloat::F4 {
        return Err(format!("{}: expected the trained <f4 values", path.display()));
    }
    Ok(Array1::from(f4_values(&bytes, header.data_off, length * 4, path)?))
}

fn f4_values(bytes: &[u8], data_off: usize, size: usize, path: &Path) -> Result<Vec<f64>, String> {
    if data_off + size != bytes.len() {
        return Err(format!("{}: {} bytes, expected {}", path.display(), bytes.len(), data_off + size));
    }
    Ok(bytes[data_off..]
        .chunks_exact(4)
        .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect())
}

/// A two-axis `<f8` array.
fn read_f8_matrix(path: &Path) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let header = parse_npy_float_header(&bytes, path)?;
    let [rows, cols] = header.shape[..] else {
        return Err(format!("{}: expected two axes, got {:?}", path.display(), header.shape));
    };
    if header.float != NpyFloat::F8 || header.data_off + rows * cols * 8 != bytes.len() {
        return Err(format!("{}: expected {rows} x {cols} <f8 values", path.display()));
    }
    let values = bytes[header.data_off..]
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("an 8-byte chunk")))
        .collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|error| format!("{}: {error}", path.display()))
}

fn write_report(out: &Path, report: &Value) -> Result<(), String> {
    let partial = out.with_extension("json.partial");
    let text = serde_json::to_string_pretty(report).map_err(|error| format!("report: {error}"))?;
    std::fs::write(&partial, text).map_err(|error| format!("write {}: {error}", partial.display()))?;
    std::fs::rename(&partial, out).map_err(|error| format!("rename to {}: {error}", out.display()))
}

/// The modular-addition transformer on its stored tensors, in torch `Linear` layout.
struct Model {
    p: usize,
    heads: usize,
    head_dim: usize,
    embed: Array2<f64>,
    positions: Array2<f64>,
    query: Array2<f64>,
    key: Array2<f64>,
    value: Array2<f64>,
    output: Array2<f64>,
    read_in: Array2<f64>,
    bias_in: Array1<f64>,
    write_out: Array2<f64>,
    bias_out: Array1<f64>,
    unembed: Array2<f64>,
}

impl Model {
    fn load(run: &Path, entry: &ModelExport) -> Result<Self, String> {
        let file = |name: &str| run.join(format!("{name}.{}.npy", entry.label));
        let config = &entry.config;
        let (p, d, width, hidden) = (config.p, config.d_model, config.n_heads * config.d_head, config.d_mlp);
        let model = Self {
            p,
            heads: config.n_heads,
            head_dim: config.d_head,
            embed: read_f4_matrix(&file("W_E"))?,
            positions: read_f4_matrix(&file("W_pos"))?,
            query: read_f4_matrix(&file("W_Q"))?,
            key: read_f4_matrix(&file("W_K"))?,
            value: read_f4_matrix(&file("W_V"))?,
            output: read_f4_matrix(&file("W_O"))?,
            read_in: read_f4_matrix(&file("W_in"))?,
            bias_in: read_f4_vector(&file("b_in"))?,
            write_out: read_f4_matrix(&file("W_out"))?,
            bias_out: read_f4_vector(&file("b_out"))?,
            unembed: read_f4_matrix(&file("W_U"))?,
        };
        for (name, found, expected) in [
            ("W_E", model.embed.dim(), (p + 1, d)),
            ("W_pos", model.positions.dim(), (3, d)),
            ("W_Q", model.query.dim(), (width, d)),
            ("W_K", model.key.dim(), (width, d)),
            ("W_V", model.value.dim(), (width, d)),
            ("W_O", model.output.dim(), (d, width)),
            ("W_in", model.read_in.dim(), (hidden, d)),
            ("W_out", model.write_out.dim(), (d, hidden)),
            ("W_U", model.unembed.dim(), (p, d)),
        ] {
            if found != expected {
                return Err(format!("{}: {name} is {found:?}, expected {expected:?}", entry.label));
            }
        }
        if model.bias_in.len() != hidden || model.bias_out.len() != d {
            return Err(format!("{}: bias lengths do not match d_mlp {hidden} and d_model {d}", entry.label));
        }
        Ok(model)
    }

    /// The logits at `=` for `pairs`, with position `u` reading `tables[u]`.
    fn logits(&self, tables: [ArrayView2<'_, f64>; 3], pairs: &[(usize, usize)]) -> Array2<f64> {
        let (heads, head_dim) = (self.heads, self.head_dim);
        // One lookup row per table row: `(T_u[r] + W_pos[u]) W_Kᵀ` is the same row operation for every
        // pair that reads row `r`, so reading it from a table is the same numbers.
        let rows: Vec<Array2<f64>> = (0..3).map(|u| &tables[u] + &self.positions.row(u)).collect();
        let keys: Vec<Array2<f64>> = rows.iter().map(|x| x.dot(&self.key.t())).collect();
        let values: Vec<Array2<f64>> = rows.iter().map(|x| x.dot(&self.value.t())).collect();
        let last = rows[2].row(self.p);
        let query = last.dot(&self.query.t());
        let divisor = (head_dim as f64).sqrt();
        let mut mixed = Array2::<f64>::zeros((pairs.len(), heads * head_dim));
        for (mut slot, &(a, b)) in mixed.outer_iter_mut().zip(pairs) {
            let tokens = [a, b, self.p];
            for head in 0..heads {
                let (start, end) = (head * head_dim, (head + 1) * head_dim);
                let q = query.slice(s![start..end]);
                let scores: [f64; 3] =
                    std::array::from_fn(|u| q.dot(&keys[u].slice(s![tokens[u], start..end])) / divisor);
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores.map(|score| (score - top).exp());
                let total: f64 = weights.iter().sum();
                let mut out = slot.slice_mut(s![start..end]);
                for u in 0..3 {
                    out.scaled_add(weights[u] / total, &values[u].slice(s![tokens[u], start..end]));
                }
            }
        }
        let residual = mixed.dot(&self.output.t()) + &last;
        let hidden = (residual.dot(&self.read_in.t()) + &self.bias_in).mapv(|value| value.max(0.0));
        let written = hidden.dot(&self.write_out.t()) + &self.bias_out;
        (residual + written).dot(&self.unembed.t())
    }
}

fn log_softmax(row: ArrayView1<'_, f64>) -> Vec<f64> {
    let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_total = top + row.iter().map(|value| (value - top).exp()).sum::<f64>().ln();
    row.iter().map(|value| value - log_total).collect()
}

fn argmax(row: ArrayView1<'_, f64>) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (index, &value)| if value > best.1 { (index, value) } else { best })
        .0
}

/// `KL(reference ‖ edited)` per row, in nats, and whether the argmaxes agree.
fn divergences(reference: &[ArrayView1<'_, f64>], edited: &Array2<f64>) -> (Vec<f64>, Vec<bool>) {
    reference
        .iter()
        .zip(edited.outer_iter())
        .map(|(native, moved)| {
            let (lp, lq) = (log_softmax(*native), log_softmax(moved));
            let kl = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
            (kl, argmax(*native) == argmax(moved))
        })
        .unzip()
}

/// Quantiles as `bench/mpd_modadd_2951.py`'s `summarize` reads them.
fn summarize(kl: &[f64], agree: &[bool]) -> Value {
    let mut ordered = kl.to_vec();
    ordered.sort_by(f64::total_cmp);
    let n = ordered.len();
    let quantile = |level: f64| ordered[(n - 1).min((level * n as f64).floor() as usize)];
    json!({
        "rows": n,
        "kl_mean": ordered.iter().sum::<f64>() / n as f64,
        "kl_q50": quantile(0.5),
        "kl_q90": quantile(0.9),
        "kl_q99": quantile(0.99),
        "kl_max": ordered[n - 1],
        "argmax_agreement": agree.iter().filter(|&&hit| hit).count() as f64 / n as f64,
    })
}

/// One evaluated edit family: the rows of every shift, and each shift's agreement.
struct Evaluated {
    kl: Vec<f64>,
    agree: Vec<bool>,
    worst_shift_agreement: (usize, f64),
}

impl Evaluated {
    fn collect(per_shift: Vec<(usize, Vec<f64>, Vec<bool>)>) -> Self {
        let mut out = Self { kl: Vec::new(), agree: Vec::new(), worst_shift_agreement: (0, f64::INFINITY) };
        for (shift, kl, agree) in per_shift {
            let share = agree.iter().filter(|&&hit| hit).count() as f64 / agree.len() as f64;
            if share < out.worst_shift_agreement.1 {
                out.worst_shift_agreement = (shift, share);
            }
            out.kl.extend(kl);
            out.agree.extend(agree);
        }
        out
    }

    fn report(&self) -> Value {
        let mut summary = summarize(&self.kl, &self.agree);
        summary["worst_shift"] = json!(self.worst_shift_agreement.0);
        summary["worst_shift_argmax_agreement"] = json!(self.worst_shift_agreement.1);
        summary
    }
}

/// A model's full logit table, the declared cycle and its planes.
struct Prepared<'a> {
    model: &'a Model,
    all: Array2<f64>,
    cycle: RowCycle,
    planes: CyclicPlanes,
    order: Vec<usize>,
}

impl Prepared<'_> {
    /// The native reference row of `(a, b)` shifted by `shift` at the positions `uses` names.
    fn reference(&self, (a, b): (usize, usize), uses: &[usize], shift: usize) -> ArrayView1<'_, f64> {
        let p = self.model.p;
        let a = if uses.contains(&0) { (a + shift) % p } else { a };
        let b = if uses.contains(&1) { (b + shift) % p } else { b };
        self.all.row(a * p + b)
    }

    /// The logits with the use sites `uses` reading `edited` and the rest the stored table.
    fn edited_logits(&self, edited: ArrayView2<'_, f64>, uses: &[usize], pairs: &[(usize, usize)]) -> Array2<f64> {
        let native = self.model.embed.view();
        let tables = std::array::from_fn(|u| if uses.contains(&u) { edited } else { native });
        self.model.logits(tables, pairs)
    }

    /// Divergence of the edited model from `reference_shift`'s native reference over `pairs`.
    fn score(
        &self,
        edited: ArrayView2<'_, f64>,
        uses: &[usize],
        pairs: &[(usize, usize)],
        reference_shift: usize,
    ) -> (Vec<f64>, Vec<bool>) {
        let logits = self.edited_logits(edited, uses, pairs);
        let reference: Vec<_> = pairs.iter().map(|&pair| self.reference(pair, uses, reference_shift)).collect();
        divergences(&reference, &logits)
    }

    /// The unedited model against `shift`'s native reference over `pairs`.
    fn unedited_score(&self, uses: &[usize], pairs: &[(usize, usize)], shift: usize) -> (Vec<f64>, Vec<bool>) {
        let p = self.model.p;
        let reference: Vec<_> = pairs.iter().map(|&pair| self.reference(pair, uses, shift)).collect();
        let rows = Array2::from_shape_fn((pairs.len(), p), |(row, column)| {
            let (a, b) = pairs[row];
            self.all[[a * p + b, column]]
        });
        divergences(&reference, &rows)
    }
}

fn edited_table(table: ArrayView2<'_, f64>, edit: &CyclicRotationEdit) -> Array2<f64> {
    &table + &edit.left.dot(&edit.right.t())
}

/// A decoded plane program: the subset in ascending order, its bits, and the decoded basis.
struct Program {
    frequencies: Vec<usize>,
    subset_bits: u64,
    basis_bits: u64,
    basis: Array2<f64>,
}

fn program(planes: &CyclicPlanes, prefix: &[usize], precision: DeclaredPrecision) -> Result<Program, String> {
    let mut frequencies = prefix.to_vec();
    frequencies.sort_unstable();
    let code = plane_program_code(planes, &frequencies, precision).map_err(|error| error.to_string())?;
    let decoded = code.basis.decode()?;
    let width = planes.planes.nrows();
    // The code sends the basis column-major: column `c` is `decoded[c·d .. (c+1)·d]`.
    let basis = Array2::from_shape_vec((2 * frequencies.len(), width), decoded)
        .map_err(|error| error.to_string())?
        .reversed_axes()
        .as_standard_layout()
        .into_owned();
    Ok(Program { frequencies, subset_bits: code.subset_bits, basis_bits: code.basis_bits, basis })
}

fn edit_at(prepared: &Prepared<'_>, basis: ArrayView2<'_, f64>, frequencies: &[usize], shift: usize) -> Result<CyclicRotationEdit, String> {
    rotation_edit(
        prepared.model.embed.view(),
        prepared.planes.mean.view(),
        basis,
        frequencies,
        prepared.model.p,
        shift,
    )
    .map_err(|error| error.to_string())
}

/// A null or control edit evaluated at every held-out shift, or the owner's refusal.
fn held_out_family(
    prepared: &Prepared<'_>,
    basis: ArrayView2<'_, f64>,
    frequencies: &[usize],
    uses: &[usize],
    pairs: &[(usize, usize)],
    shifts: &[usize],
    unshifted_reference: bool,
) -> Result<Evaluated, String> {
    let per_shift = shifts
        .par_iter()
        .map(|&shift| {
            let edit = edit_at(prepared, basis, frequencies, shift)?;
            let table = edited_table(prepared.model.embed.view(), &edit);
            let reference = if unshifted_reference { 0 } else { shift };
            let (kl, agree) = prepared.score(table.view(), uses, pairs, reference);
            Ok((shift, kl, agree))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Evaluated::collect(per_shift))
}

/// The largest `shift_residual` bounds of a program over `shifts`.
fn residual_bounds(prepared: &Prepared<'_>, program: &Program, shifts: &[usize]) -> Result<Value, String> {
    let bounds = shifts
        .par_iter()
        .map(|&shift| {
            let edit = edit_at(prepared, program.basis.view(), &program.frequencies, shift)?;
            let residual = shift_residual(prepared.model.embed.view(), &prepared.cycle, &edit)
                .map_err(|error| error.to_string())?;
            Ok((residual.residual_upper_bound(), residual.fixed_row_upper_bound(), edit.sigma_min / edit.sigma_max))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let max = |pick: fn(&(f64, f64, f64)) -> f64| bounds.iter().map(pick).fold(0.0, f64::max);
    Ok(json!({
        "max_cycled_row_residual_upper_bound": max(|b| b.0),
        "max_fixed_row_change_upper_bound": max(|b| b.1),
        "basis_inverse_condition": bounds.first().map(|b| b.2),
    }))
}

fn haar_basis(width: usize, columns: usize, rng: &mut StdRng) -> Result<Array2<f64>, String> {
    // Box–Muller: two uniforms on (0, 1] give one standard normal.
    let draw = Array2::<f64>::from_shape_fn((width, columns), |_| {
        let radius = (-2.0 * (1.0 - rng.random::<f64>()).ln()).sqrt();
        radius * (2.0 * PI * rng.random::<f64>()).cos()
    });
    let (q, _) = draw.qr().map_err(|error| error.to_string())?;
    Ok(q.slice(s![.., ..columns]).to_owned())
}

fn run_model(entry: &ModelExport, run: &Path, settings: &Settings, precision: DeclaredPrecision) -> Result<Value, String> {
    let started = Instant::now();
    let model = Model::load(run, entry)?;
    let p = model.p;
    let every: Vec<(usize, usize)> = (0..p * p).map(|index| (index / p, index % p)).collect();
    let native = model.embed.view();
    let all = model.logits([native, native, native], &every);
    let torch = read_f8_matrix(&run.join(format!("logits64.{}.npy", entry.label)))?;
    if torch.dim() != all.dim() {
        return Err(format!("{}: torch logits are {:?}, expected {:?}", entry.label, torch.dim(), all.dim()));
    }
    let torch_gap = (&all - &torch).iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let logit_scale = torch.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));

    let mut successor: Vec<usize> = (0..=p).collect();
    for (row, image) in successor.iter_mut().take(p).enumerate() {
        *image = (row + 1) % p;
    }
    let cycle = RowCycle::from_successor(&successor).map_err(|error| error.to_string())?;
    let planes = cyclic_planes(native, &cycle).map_err(|error| error.to_string())?;
    let m = cycle.plane_count();
    let mut order: Vec<usize> = (1..=m).collect();
    order.sort_by(|&j, &k| planes.power[k - 1].total_cmp(&planes.power[j - 1]));
    let variation = planes.power.iter().sum::<f64>() + 2.0 * planes.mean.iter().map(|v| v * v).sum::<f64>();
    let spectrum: Vec<Value> = order.iter().take(10).map(|&k| json!([k, planes.power[k - 1] / variation])).collect();
    let prepared = Prepared { model: &model, all, cycle, planes, order };

    // Executor control: the exact row permutation at every site against the shifted-token rows.
    let mut permutation_mismatches = serde_json::Map::new();
    for (site, uses) in SITES {
        let shift = settings.fit_shift;
        let permuted = Array2::from_shape_fn(model.embed.dim(), |(row, column)| {
            let source = if row < p { (row + shift) % p } else { row };
            model.embed[[source, column]]
        });
        let logits = prepared.edited_logits(permuted.view(), uses, &every);
        let mismatched = every
            .iter()
            .zip(logits.outer_iter())
            .filter(|(pair, row)| prepared.reference(**pair, uses, shift) != *row)
            .count();
        permutation_mismatches.insert(site.to_string(), json!(mismatched));
    }
    println!(
        "[executor] {} torch_float64_max_abs_gap={torch_gap:.3e} logit_scale={logit_scale:.3e} row_permutation_bitwise_mismatches={}",
        entry.label,
        Value::Object(permutation_mismatches.clone())
    );

    let pairs_of = |indices: &[usize]| -> Vec<(usize, usize)> { indices.iter().map(|&index| every[index]).collect() };
    let (train, test) = (pairs_of(&entry.train_idx), pairs_of(&entry.test_idx));
    let held_out: Vec<usize> = (2..p).collect();
    let prefixes: Vec<Program> = (1..=m)
        .map(|k| program(&prepared.planes, &prepared.order[..k], precision))
        .collect::<Result<_, _>>()?;
    let empty_bits = subset_code_len_bits(m, 0).map_err(|error| format!("{error:?}"))?;
    let full = prefixes.last().expect("m >= 1 planes");
    let mut sites = serde_json::Map::new();
    let mut rng = StdRng::seed_from_u64(settings.random_basis_seed);
    for (site, uses) in SITES {
        let site_started = Instant::now();
        let (unedited_kl, _) = prepared.unedited_score(uses, &train, settings.fit_shift);
        let mut candidates = vec![(0_usize, empty_bits, unedited_kl.iter().sum::<f64>() / LN_2)];
        candidates.extend(
            prefixes
                .par_iter()
                .map(|candidate| {
                    let edit = edit_at(&prepared, candidate.basis.view(), &candidate.frequencies, settings.fit_shift)?;
                    let table = edited_table(native, &edit);
                    let (kl, _) = prepared.score(table.view(), uses, &train, settings.fit_shift);
                    Ok((candidate.frequencies.len(), candidate.subset_bits + candidate.basis_bits, kl.iter().sum::<f64>() / LN_2))
                })
                .collect::<Result<Vec<_>, String>>()?,
        );
        let (selected_k, _, _) = candidates
            .iter()
            .copied()
            .min_by(|x, y| (x.1 as f64 + x.2).total_cmp(&(y.1 as f64 + y.2)))
            .expect("K = 0 is always a candidate");

        let unedited = Evaluated::collect(
            held_out
                .par_iter()
                .map(|&shift| {
                    let (kl, agree) = prepared.unedited_score(uses, &test, shift);
                    (shift, kl, agree)
                })
                .collect(),
        );
        let evaluate = |candidate: &Program| {
            held_out_family(&prepared, candidate.basis.view(), &candidate.frequencies, uses, &test, &held_out, false)
        };
        let full_eval = evaluate(full)?;
        let (selected_report, selected_residual, null_edits) = if selected_k == 0 {
            (unedited.report(), Value::Null, Value::Null)
        } else {
            let chosen = &prefixes[selected_k - 1];
            let report = evaluate(chosen)?.report();
            let residual = residual_bounds(&prepared, chosen, &held_out)?;
            let nulls = if settings.null_edit_models.contains(&entry.label) {
                let bottom = program(&prepared.planes, &prepared.order[m - selected_k..], precision)?;
                let (q, _) = chosen.basis.qr().map_err(|error| error.to_string())?;
                let key = q.slice(s![.., ..chosen.basis.ncols()]).to_owned();
                let off_key = &bottom.basis - &key.dot(&key.t().dot(&bottom.basis));
                let random = haar_basis(model.embed.ncols(), chosen.basis.ncols(), &mut rng)?;
                let outcome = |result: Result<Evaluated, String>| match result {
                    Ok(evaluated) => evaluated.report(),
                    Err(refusal) => json!({ "refused": refusal }),
                };
                json!({
                    "bottom_frequencies": bottom.frequencies,
                    "bottom_planes_off_selected_span_vs_unshifted": outcome(held_out_family(
                        &prepared, off_key.view(), &bottom.frequencies, uses, &test, &held_out, true)),
                    "random_basis_selected_angles_vs_shifted": outcome(held_out_family(
                        &prepared, random.view(), &chosen.frequencies, uses, &test, &held_out, false)),
                })
            } else {
                Value::Null
            };
            (report, residual, nulls)
        };
        let selected = candidates[selected_k];
        let site_report = json!({
            "fit": {
                "shift": settings.fit_shift,
                "train_rows": train.len(),
                "candidates": candidates.iter().map(|&(k, program_bits, data_bits)| json!({
                    "K": k,
                    "frequencies": if k == 0 { Vec::new() } else { prefixes[k - 1].frequencies.clone() },
                    "program_bits": program_bits,
                    "data_bits": data_bits,
                    "total_bits": program_bits as f64 + data_bits,
                })).collect::<Vec<_>>(),
            },
            "selected": {
                "K": selected_k,
                "frequencies": if selected_k == 0 { Vec::new() } else { prefixes[selected_k - 1].frequencies.clone() },
                "program_bits": selected.1,
                "fit_data_bits": selected.2,
                "full_program_bits": candidates[m].1,
                "full_program_fit_data_bits": candidates[m].2,
                "program_bits_over_full": selected.1 as f64 / candidates[m].1 as f64,
            },
            "held_out": {
                "shifts": [2, p - 1],
                "selected": selected_report,
                "full_program": full_eval.report(),
                "unedited": unedited.report(),
            },
            "shift_residual": {
                "selected": selected_residual,
                "full_program": residual_bounds(&prepared, full, &held_out)?,
            },
            "null_edits": null_edits,
        });
        println!(
            "[site] {} {site} K={selected_k} program_bits={} full_bits={} held_out_selected={} held_out_full={} unedited={} seconds={:.1}",
            entry.label,
            selected.1,
            candidates[m].1,
            site_report["held_out"]["selected"],
            site_report["held_out"]["full_program"]["argmax_agreement"],
            site_report["held_out"]["unedited"]["argmax_agreement"],
            site_started.elapsed().as_secs_f64()
        );
        if !site_report["null_edits"].is_null() {
            println!("[null] {} {site} {}", entry.label, site_report["null_edits"]);
        }
        sites.insert(site.to_string(), site_report);
    }
    Ok(json!({
        "label": entry.label,
        "step": entry.step,
        "labels": entry.config.labels,
        "shuffle_mlp_seed": entry.shuffle_mlp_seed,
        "train_acc": entry.train_acc,
        "test_acc": entry.test_acc,
        "executor": {
            "torch_float64_max_abs_logit_gap": torch_gap,
            "max_abs_logit": logit_scale,
            "row_permutation_bitwise_mismatched_rows": Value::Object(permutation_mismatches),
            "rows": p * p,
        },
        "spectrum_W_E": spectrum,
        "sites": sites,
        "seconds": started.elapsed().as_secs_f64(),
    }))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (run, settings_path, out) = (flag(&args, "--run")?, flag(&args, "--settings")?, flag(&args, "--out")?);
    let settings: Settings = read_json(&settings_path)?;
    let export: Export = read_json(&run.join("export.json"))?;
    if settings.stage != "plane_edits" || export.stage != "execute" {
        return Err(format!(
            "settings stage {:?} and export stage {:?}: this receipt reads stage \"plane_edits\" of `execute`",
            settings.stage, export.stage
        ));
    }
    let precision = DeclaredPrecision::new(settings.fraction_bits)?;
    let mut models = Vec::new();
    for entry in &export.exports {
        models.push(run_model(entry, &run, &settings, precision)?);
        write_report(
            &out,
            &json!({
                "stage": "plane_edits",
                "fit_shift": settings.fit_shift,
                "fraction_bits": settings.fraction_bits,
                "random_basis_seed": settings.random_basis_seed,
                "models": models,
            }),
        )?;
    }
    println!("[report] {}", out.display());
    Ok(())
}
