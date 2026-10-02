//! Per-input pieces of a language model trained through its own masked forward, on streamed
//! sequences (#2951).
//!
//! `mpd_pieces_masked_2951 EXPORT_DIR OUT.json OBSERVATIONS {wsvd|wsvd2|library:DIR} TRAIN EVAL [CONTEXT] [GPU] [SETS]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`) whose first
//! `TRAIN` token rows train the pieces and whose next `EVAL` rows evaluate them, `CONTEXT`
//! positions each (default 512). `GPU` (`off`, `auto` (default) or `required`) is the policy for the
//! proposal products (`gam_mpd::device`); every acceptance runs in float64 on the CPU. The sites are
//! the model's hidden-to-hidden maps (`gam_mpd::masked::sites`); their read moments and output
//! Fishers are measured on the training sequences (`gam_mpd::masked::site_statistics`), so any
//! imported model runs as it is. `wsvd` starts from each site's exact Fisher-whitened singular
//! pieces (`gam_mpd::pieces::fisher_svd`), and `wsvd2` grows those to twice as many on the first
//! training sequence (`gam_mpd::masked::split`). `library:DIR` starts from a given library: per
//! site `DIR/{site}.v.f64` (pieces × d_in) and `DIR/{site}.u.f64` (pieces × d_out), raw float64,
//! on the uncentred read with nothing beyond the pieces (a site without files stays native).
//!
//! `SETS` (with `library:DIR`) is a directory of given per-token sets for that library
//! (`bench/vpd_2951/vpd_sets_export.py`: `indptr.i64`, `indices.i64` as CSR over positions,
//! sequence after sequence in the export's order, pieces numbered site after site; `sites.txt`
//! naming the sites and their pieces). Every eval sequence then starts its selection from its given
//! sets (both halves of a split piece on where it was), and keeps them unless the selected sets
//! code it in fewer bits; every training sequence starts from its current sets, its given ones at
//! first. The context coder counts the current sets of every training sequence but the one being
//! coded, plus the given sets of the sequences past `TRAIN + EVAL` (held out from both). Each eval
//! point also carries the given sets' own point (`start`) under the same coder. No model
//! statistics are measured.
//!
//! With `TRAIN` 0 nothing is trained: the eval sequences are selected pass after pass, each pass
//! with the counts of the sets the pass before selected, until a pass saves less than a bit per
//! token; every pass is a full eval. With `SETS` there is one pass, coded by the held-out counts.
//! A running point after each eval sequence goes to `OUT.progress.json`.
//!
//! The training sequences stream one at a time: each starts from its current sets (on its first
//! visit without `SETS`, the pieces whose own second-order KL bits in the global Fisher, on its
//! clean forward, exceed their listing cost); its sets are selected exactly in the masked forward
//! (`gam_mpd::masked::select`), coded by the current sets of every other training sequence, and kept
//! only when they code it in fewer bits than its start; then the pieces take one exact-gradient
//! step on it, preconditioned by the running read covariances and written Fishers of every sequence
//! seen
//! (`gam_mpd::masked::step_pieces`). Four times per pass the first four eval sequences, and at each
//! pass's end all of them, are selected one at a time with the current costs, and their mean active
//! pieces (L0), KL and bits per token in the
//! per-token frontier's code (per site `ω(k + 1) + log₂ C(C, k)`) are appended to `OUT.json` as
//! `{points: [{l0, bits, kl, …}]}`; a full eval also writes its selected sets as CSR
//! (`OUT.pass{P}.{indptr,indices,offsets}.npy`, pieces numbered site after site) and each eval
//! token's KL (`OUT.pass{P}.kl.npy`, float64, rows in order) and whether its argmax is the model's
//! (`OUT.pass{P}.agree.npy`, int64), the same for the starting sets (`OUT.pass{P}.start.{kl,agree}.npy`).
//! Passes repeat until one saves less than a bit per token.
//!
//! After every training sequence (every pass when nothing trains) the run's state goes to the
//! checkpoint `OUT.checkpoint/` (`gam_mpd::checkpoint`: the libraries, the context counts, the
//! running preconditioners, every sequence's current sets, the points and the pass and sequence
//! reached), and a run that finds one there resumes from it: killed at any point and restarted,
//! it writes what the uninterrupted run would have. Every draw is seeded by the pass and sequence,
//! so nothing else is kept. A finished run's checkpoint says so, and a restart then stops at once.

use gam_mpd::checkpoint::{Saved, load, save};
use gam_mpd::codec::prefix_integer_len_bits;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{
    Context, Library, Masked, Running, Target, forward, matrix, previous_inputs, read_values, select, site_statistics, sites, split, step_pieces,
};
use gam_mpd::operator_program::FamilyInputs;
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2, Axis};
use serde_json::json;
use statrs::function::gamma::ln_gamma;
use std::path::{Path, PathBuf};

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % (cols * 8) != 0 {
        return Err(format!("{}: {} bytes are not rows of {cols} float64", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((bytes.len() / (cols * 8), cols), values).map_err(|e| e.to_string())
}

/// A one-dimensional little-endian `.npy` file of 8-byte values of type `descr`.
fn write_npy(path: &Path, descr: &str, values: impl ExactSizeIterator<Item = [u8; 8]>) -> Result<(), String> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({},), }}", values.len());
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in values {
        bytes.extend_from_slice(&v);
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

/// Sums over a sequence's tokens: active pieces, KL, frontier bits.
fn sums(masks: &[Array2<f64>], kl: &Array1<f64>) -> (f64, f64, f64) {
    let mut l0 = 0.0;
    let mut bits = 0.0;
    for m in masks {
        let c = m.ncols() as f64;
        for row in m.outer_iter() {
            let k = row.iter().filter(|x| **x > 0.0).count();
            l0 += k as f64;
            let omega = prefix_integer_len_bits(k as u64 + 1).map_or(0.0, |b| b as f64);
            bits += omega + (ln_gamma(c + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma(c - k as f64 + 1.0)) / std::f64::consts::LN_2;
        }
    }
    (l0, kl.sum(), bits)
}

fn read_i64(path: &Path) -> Result<Vec<i64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{}: {} bytes are not int64", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

/// Given per-position sets (module note, `SETS`): CSR over the export's positions, `rows` per
/// sequence, pieces numbered site after site from `offsets`.
struct Sets {
    indptr: Vec<i64>,
    indices: Vec<i64>,
    offsets: Vec<usize>,
    rows: usize,
}

impl Sets {
    fn load(dir: &Path, sites: &[(String, usize)], rows: usize) -> Result<Self, String> {
        let listed = std::fs::read_to_string(dir.join("sites.txt")).map_err(|e| format!("{}: {e}", dir.display()))?;
        let expected: Vec<String> = sites.iter().map(|(name, pieces)| format!("{name} {pieces}")).collect();
        if listed.lines().collect::<Vec<_>>() != expected {
            return Err(format!("{}: its sites are not the library's ({expected:?})", dir.display()));
        }
        let mut offsets = vec![0];
        for (_, pieces) in sites {
            offsets.push(offsets[offsets.len() - 1] + pieces);
        }
        let indptr = read_i64(&dir.join("indptr.i64"))?;
        let indices = read_i64(&dir.join("indices.i64"))?;
        let total = offsets[offsets.len() - 1] as i64;
        if indptr.last().copied() != Some(indices.len() as i64) || (indptr.len() - 1) % rows != 0 || indices.iter().any(|i| *i < 0 || *i >= total) {
            return Err(format!("{}: not CSR over sequences of {rows} positions and {total} pieces", dir.display()));
        }
        Ok(Self { indptr, indices, offsets, rows })
    }

    fn sequences(&self) -> usize {
        (self.indptr.len() - 1) / self.rows
    }

    /// Whether these sets number `libraries`' pieces.
    fn fits(&self, libraries: &[Library]) -> bool {
        libraries.len() + 1 == self.offsets.len() && libraries.iter().zip(self.offsets.windows(2)).all(|(l, o)| l.v.nrows() == o[1] - o[0])
    }

    /// Sequence `s`'s sets.
    fn assigned(&self, s: usize) -> Assigned {
        let sites = self.offsets.len() - 1;
        let mut out = Assigned { rows: self.rows, sites: vec![(vec![0], Vec::new()); sites] };
        for r in 0..self.rows {
            let p = s * self.rows + r;
            for &i in &self.indices[self.indptr[p] as usize..self.indptr[p + 1] as usize] {
                let i = i as usize;
                let k = self.offsets.partition_point(|o| *o <= i) - 1;
                out.sites[k].1.push((i - self.offsets[k]) as u32);
            }
            for (indptr, local) in &mut out.sites {
                indptr.push(local.len() as u32);
            }
        }
        out
    }
}

/// One sequence's sets, sparse: per site, CSR over its positions of the pieces on (ascending).
#[derive(Clone)]
struct Assigned {
    rows: usize,
    sites: Vec<(Vec<u32>, Vec<u32>)>,
}

impl Assigned {
    fn of(masks: &[Array2<f64>]) -> Self {
        let rows = masks.first().map_or(0, |m| m.nrows());
        let sites = masks
            .iter()
            .map(|m| {
                let mut indptr = vec![0u32];
                let mut local = Vec::new();
                for row in m.outer_iter() {
                    local.extend(row.iter().enumerate().filter(|(_, x)| **x > 0.0).map(|(c, _)| c as u32));
                    indptr.push(local.len() as u32);
                }
                (indptr, local)
            })
            .collect();
        Self { rows, sites }
    }

    fn on(&self, k: usize, r: usize) -> &[u32] {
        let (indptr, local) = &self.sites[k];
        &local[indptr[r] as usize..indptr[r + 1] as usize]
    }

    /// One dense `rows × C` mask per site, `pieces[k]` wide.
    fn masks(&self, pieces: &[usize]) -> Vec<Array2<f64>> {
        pieces
            .iter()
            .enumerate()
            .map(|(k, &c)| {
                let mut m = Array2::zeros((self.rows, c));
                for r in 0..self.rows {
                    for &i in self.on(k, r) {
                        m[[r, i as usize]] = 1.0;
                    }
                }
                m
            })
            .collect()
    }

    /// Add `weight` times these sets' counts to `context` (as `Context::absorb` does with dense
    /// masks, each position's previous input the position before it in its sequence); `-1`
    /// takes back what `1` added.
    fn absorb(&self, context: &mut Context, weight: f64) {
        for k in 0..self.sites.len() {
            for r in 0..self.rows {
                let now = self.on(k, r);
                let before = if r > 0 { self.on(k, r - 1) } else { &[] };
                for &c in before {
                    context.was_on[k][c as usize] += weight;
                    if now.binary_search(&c).is_ok() {
                        context.stayed[k][c as usize] += weight;
                    }
                }
                for &c in now {
                    if before.binary_search(&c).is_err() {
                        context.new[k][c as usize] += weight;
                    }
                }
            }
        }
    }

    /// These sets on a grown library (`Context::grown`'s `origins`): every new piece is on
    /// wherever its original piece was.
    fn grown(&self, origins: &[Vec<usize>]) -> Self {
        let sites = self
            .sites
            .iter()
            .zip(origins)
            .enumerate()
            .map(|(k, (_, origin))| {
                let pieces = origin.iter().copied().max().map_or(0, |m| m + 1);
                let mut children = vec![Vec::new(); pieces];
                for (new, &old) in origin.iter().enumerate() {
                    children[old].push(new as u32);
                }
                let mut indptr = vec![0u32];
                let mut local = Vec::new();
                for r in 0..self.rows {
                    let mut row: Vec<u32> = self.on(k, r).iter().flat_map(|&c| children[c as usize].iter().copied()).collect();
                    row.sort_unstable();
                    local.extend(row);
                    indptr.push(local.len() as u32);
                }
                (indptr, local)
            })
            .collect();
        Self { rows: self.rows, sites }
    }
}

/// Running sums over eval tokens of one family of sets.
#[derive(Clone, Copy, Default)]
struct Tally {
    l0: f64,
    kl: f64,
    bits: f64,
    explanation: f64,
    agree: f64,
    tokens: f64,
}

impl Tally {
    fn add(&mut self, masks: &[Array2<f64>], kl: &Array1<f64>, explanation: f64, agree: &[i64]) {
        let (l0, kl_sum, bits) = sums(masks, kl);
        self.l0 += l0;
        self.kl += kl_sum;
        self.bits += bits;
        self.explanation += explanation;
        self.agree += agree.iter().sum::<i64>() as f64;
        self.tokens += kl.len() as f64;
    }

    fn json(&self, observations: f64) -> serde_json::Value {
        let t = self.tokens;
        json!({
            "l0": self.l0 / t, "kl": self.kl / t, "bits": self.bits / t,
            "context_bits": self.explanation / t,
            "code": (self.explanation + self.kl * observations / std::f64::consts::LN_2) / t,
            "agree": self.agree / t,
        })
    }
}

/// Per row, whether the argmax of `logits` is the argmax of `target`'s.
fn agreement(logits: &Array2<f64>, target: &Target) -> Vec<i64> {
    let argmax = |row: ndarray::ArrayView1<f64>| row.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
    (0..target.logits.nrows()).map(|r| i64::from(argmax(logits.row(r)) == argmax(target.logits.row(r)))).collect()
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_pieces_masked_2951 EXPORT_DIR OUT.json OBSERVATIONS {wsvd|wsvd2|library:DIR} TRAIN EVAL [CONTEXT] [GPU] [SETS]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let observations: f64 = args.get(3).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let start = args.get(4).ok_or(usage)?.clone();
    let given = start.strip_prefix("library:").map(PathBuf::from);
    if start != "wsvd" && start != "wsvd2" && given.is_none() {
        return Err(format!("unknown start {start}; {usage}"));
    }
    let train: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let eval: usize = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("EVAL: {e}"))?;
    let context: usize = args.get(7).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let gpu = args.get(8).map_or("auto", String::as_str);
    let sets_dir = args.get(9).map(PathBuf::from);
    if sets_dir.is_some() && given.is_none() {
        return Err(format!("SETS needs a library:DIR start; {usage}"));
    }
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::parse(gpu).ok_or_else(|| format!("GPU {gpu}: expected off, auto or required"))?);
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let context_rows = context;
    let sequence = |s: usize| -> FamilyInputs { family.select(&(s * context_rows..(s + 1) * context_rows).collect::<Vec<_>>()) };
    let target_of = |inputs: &FamilyInputs| -> Result<Target, String> {
        Ok(Target::every_row(model.execute(inputs, false).map_err(|e| e.to_string())?.values[model.output].clone()))
    };
    let all_sites = sites(model);
    // The model's statistics on the training sequences (on the eval sequences when nothing trains).
    // A given library started from given sets uses none of them.
    let measured_on = if train > 0 { 0..train } else { train..train + eval };
    let statistics: Vec<Option<gam_mpd::pieces::Site>> = if sets_dir.is_some() {
        all_sites.iter().map(|_| None).collect()
    } else {
        site_statistics(model, &all_sites, measured_on.map(sequence), 2, 0x5EED)?.into_iter().map(Some).collect::<Vec<_>>()
    };
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    // Per site, the written Fisher of the clean model (empty when not measured).
    let mut fishers: Vec<Array2<f64>> = Vec::new();
    for (site, measured) in all_sites.iter().zip(statistics) {
        if let Some(dir) = &given {
            let v_path = dir.join(format!("{}.v.f64", site.name));
            if !v_path.exists() {
                eprintln!("{}: no given library, native", site.name);
                continue;
            }
            let w = match &measured {
                Some(m) => m.w.clone(),
                None => matrix(model, site)?,
            };
            let (d_out, d_in) = w.dim();
            let v = read_f64(&v_path, d_in)?;
            let u = read_f64(&dir.join(format!("{}.u.f64", site.name)), d_out)?;
            if u.nrows() != v.nrows() {
                return Err(format!("{}: {} v pieces and {} u pieces", site.name, v.nrows(), u.nrows()));
            }
            let left = &w - &v.t().dot(&u).t();
            let norm = |m: &Array2<f64>| m.iter().map(|x| x * x).sum::<f64>().sqrt();
            eprintln!("{}: {d_out}×{d_in}, {} given pieces, ‖W − Σ u vᵀ‖/‖W‖ = {:.2e}", site.name, v.nrows(), norm(&left) / norm(&w));
            fishers.push(measured.map_or_else(|| Array2::zeros((0, 0)), |m| m.fisher));
            chosen.push(site.clone());
            libraries.push(Library { v, u, mean: Array1::zeros(d_in) });
            continue;
        }
        let measured = measured.ok_or("no statistics")?;
        let library = fisher_svd(&measured)?;
        let (v, u) = (library.v.t().to_owned(), library.u);
        let error = (&v.t().dot(&u).t() - &measured.w).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let largest = measured.w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if error > 1e-6 * largest {
            return Err(format!("{}: the starting library is not the site ({error:e} against {largest:e})", site.name));
        }
        eprintln!("{}: {}×{}, {} pieces", site.name, measured.w.nrows(), measured.w.ncols(), v.nrows());
        fishers.push(measured.fisher);
        chosen.push(site.clone());
        libraries.push(Library { v, u, mean: measured.mean });
    }
    let sets = match &sets_dir {
        Some(dir) => {
            let named: Vec<(String, usize)> = chosen.iter().zip(&libraries).map(|(site, l)| (site.name.clone(), l.v.nrows())).collect();
            let sets = Sets::load(dir, &named, context_rows)?;
            if sets.sequences() < train + eval {
                return Err(format!("{}: {} sequences of sets for {} train and eval sequences", dir.display(), sets.sequences(), train + eval));
            }
            Some(sets)
        }
        None => None,
    };
    let original_sites = chosen.clone();
    let mut masked = Masked::build(model, chosen, libraries)?;
    // The run's checkpoint (module note), and what it must agree with to resume.
    let checkpoint = out.with_extension("checkpoint");
    let run = json!({"observations": observations, "start": start, "train": train, "eval": eval, "context": context_rows});
    let resumed = load(&checkpoint)?;
    if let Some(r) = &resumed {
        if r.driver["run"] != run {
            return Err(format!("{}: a checkpoint of another run ({}); remove it to start over", checkpoint.display(), r.driver["run"]));
        }
        if r.driver["done"] == json!(true) {
            eprintln!("{}: the run finished; nothing to resume", checkpoint.display());
            return Ok(());
        }
    }
    let samples = 2;
    // The firing counts of every set selected so far, and the costs they give; with given sets, the
    // counts start from those of the sequences held out from training and evaluation.
    let mut context = Context::new(&masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>());
    if let Some(sets) = &sets {
        for s in train + eval..sets.sequences() {
            sets.assigned(s).absorb(&mut context, 1.0);
        }
        eprintln!("context from the given sets of {} held-out sequences", sets.sequences() - train - eval);
    }
    // A sequence's start: the pieces whose own second-order KL bits, `n a² uᵀBu / (2 ln 2)` with
    // `a = v · (x − μ)` on the clean forward, exceed their current listing cost.
    let start_masks = |inputs: &FamilyInputs, libraries: &[Library], costs: &[Array1<f64>]| -> Result<Vec<Array2<f64>>, String> {
        let trace = model.execute(inputs, false).map_err(|e| e.to_string())?;
        let scale = observations / (2.0 * std::f64::consts::LN_2);
        let mut masks = Vec::new();
        for (k, library) in libraries.iter().enumerate() {
            if fishers[k].nrows() != library.u.ncols() {
                return Err(format!("{}: no measured Fisher to start from", original_sites[k].name));
            }
            let x = read_values(&trace, &original_sites[k])? - &library.mean;
            let a = x.dot(&library.v.t());
            // Each piece's second-order weight `uᵀ B u` in the global Fisher.
            let weights = (&library.u.dot(&fishers[k]) * &library.u).sum_axis(Axis(1));
            masks.push(Array2::from_shape_fn(a.dim(), |(r, c)| if scale * a[[r, c]] * a[[r, c]] * weights[c] > costs[k][c] { 1.0 } else { 0.0 }));
        }
        Ok(masks)
    };
    // A sequence's start: its given sets (on the library as it has grown since), else `start_masks`.
    let begin_of = |given: Option<&Assigned>, inputs: &FamilyInputs, libraries: &[Library], costs: &[Array1<f64>]| -> Result<Vec<Array2<f64>>, String> {
        match given {
            Some(assigned) => Ok(assigned.masks(&libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>())),
            None => start_masks(inputs, libraries, costs),
        }
    };
    // Each eval sequence's given sets, kept on the library as it grows.
    let mut eval_starts: Vec<Option<Assigned>> =
        (0..eval).map(|e| sets.as_ref().filter(|x| x.fits(&masked.libraries)).map(|x| x.assigned(train + e))).collect();
    // `wsvd2`: the Fisher-SVD library grown to twice its pieces on the first training sequence,
    // every piece its listing inputs use in two ways split in two (`gam_mpd::masked::split`).
    if start == "wsvd2" && resumed.is_none() {
        let inputs = sequence(0);
        let target = target_of(&inputs)?;
        let coder = context.coder(previous_inputs(&inputs));
        let begin = start_masks(&inputs, &masked.libraries, &coder.costs)?;
        let (masks, _) = select(&masked, &inputs, &target, begin, &coder, observations, samples)?;
        let trace = model.execute(&inputs, false).map_err(|e| e.to_string())?;
        let mut grown = Vec::new();
        for (k, library) in masked.libraries.iter().enumerate() {
            let x = read_values(&trace, &original_sites[k])?;
            grown.push(split(library, &x, &masks[k]).0);
        }
        eprintln!("grown to {} pieces", grown.iter().map(|l| l.v.nrows()).sum::<usize>());
        masked = Masked::build(model, original_sites.clone(), grown)?;
        context = Context::new(&masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>());
    }
    // Resuming: the saved libraries, counts, preconditioners, sets and points.
    let (mut running, mut points, mut previous, mut saved_sets, first_pass, first_sequence, mut pass_code) = match resumed {
        Some(r) => {
            masked = Masked::build(model, original_sites.clone(), r.libraries)?;
            context = r.context;
            let index = |key: &str| r.driver[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("checkpoint: no {key}"));
            let (pass, next) = (index("pass")?, index("next")?);
            eprintln!("resumed from {} at pass {pass}, sequence {next}", checkpoint.display());
            let points = r.driver["points"].as_array().cloned().ok_or("checkpoint: no points")?;
            let previous = r.driver["previous"].as_f64().unwrap_or(f64::INFINITY);
            let pass_code = r.driver["pass_code"].as_f64().ok_or("checkpoint: no pass code")?;
            (r.running, points, previous, Some(r.sets), pass, next, pass_code)
        }
        None => (Running::default(), Vec::new(), f64::INFINITY, None, 0, 0, 0.0),
    };
    if let Some(saved) = &saved_sets
        && saved.len() != train + eval
    {
        return Err(format!("checkpoint: sets of {} sequences for {}", saved.len(), train + eval));
    }
    let as_assigned = |sites: Option<gam_mpd::checkpoint::SparseSets>| sites.map(|sites| Assigned { rows: context_rows, sites });
    if let Some(saved) = saved_sets.as_mut() {
        for (e, start) in eval_starts.iter_mut().enumerate() {
            *start = as_assigned(saved[train + e].take());
        }
    }
    // Saves the run's state, with `driver`'s pass, sequence, code and `done`.
    let save_state = |driver: serde_json::Value, masked: &Masked, context: &Context, running: &Running, current: &[Option<Assigned>], eval_starts: &[Option<Assigned>]| -> Result<(), String> {
        let mut driver = driver;
        driver["run"] = run.clone();
        let sets = current.iter().chain(eval_starts).map(|a| a.as_ref().map(|a| &a.sites)).collect();
        save(&checkpoint, &Saved { driver: &driver, libraries: &masked.libraries, context, running, sets })
    };
    // JSON has no infinity: a first pass's `previous` is null.
    let finite = |v: f64| if v.is_finite() { json!(v) } else { serde_json::Value::Null };
    // The bits of one real of a library piece: they are sent in single precision.
    const BITS_PER_REAL: f64 = 32.0;
    let stem = out.with_extension("");
    // Select the first `evaluated` eval sequences with `context`'s counts; a full eval also writes
    // the sets, each token's KL and argmax agreement. Returns the point, the code per token, and
    // the counts of the sets selected.
    let evaluate = |masked: &Masked, context: &Context, starts: &[Option<Assigned>], pass: usize, trained: usize, evaluated: usize, full: bool| -> Result<(serde_json::Value, f64, Context), String> {
        let mut seen = Context::new(&masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>());
        // The selected sets, and the sets selection started from.
        let (mut selected, mut started) = (Tally::default(), Tally::default());
        // Sets as CSR over all pieces (sites in order), so other context codes can score them.
        let mut indptr: Vec<i64> = vec![0];
        let mut indices: Vec<i64> = Vec::new();
        let (mut token_kl, mut start_kl): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let (mut agree, mut start_agree): (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
        let point_of = |selected: &Tally, started: &Tally, evaluated: usize| {
            let mut point = selected.json(observations);
            point["start"] = started.json(observations);
            for (key, value) in [
                ("pieces", json!(masked.libraries.iter().map(|l| l.v.nrows()).sum::<usize>())),
                ("pass", json!(pass)),
                ("sequences_trained", json!(trained)),
                ("observations", json!(observations)),
                ("eval_sequences", json!(evaluated)),
            ] {
                point[key] = value;
            }
            point
        };
        let scale = observations / std::f64::consts::LN_2;
        for e in 0..evaluated {
            let inputs = sequence(train + e);
            let target = target_of(&inputs)?;
            let previous_rows = previous_inputs(&inputs);
            let coder = context.coder(previous_rows.clone());
            let begin = begin_of(starts[e].as_ref(), &inputs, &masked.libraries, &coder.costs)?;
            // The start's own exact KL, explanation bits and argmax agreement.
            let (begin_kl, begin_agree) = {
                let (kl, trace, _) = forward(masked, &masked.family(&inputs, &begin), &target)?;
                (kl, agreement(&trace.values[masked.program.output], &target))
            };
            let begin_bits = coder.bits(&begin).sum();
            started.add(&begin, &begin_kl, begin_bits, &begin_agree);
            let begin_code = begin_bits + begin_kl.sum() * scale;
            let begin_l0 = sums(&begin, &begin_kl).0;
            let (masks, values) = select(masked, &inputs, &target, begin, &coder, observations, samples)?;
            let bits = coder.bits(&masks).sum();
            // Selection keeps each input's flips by that input's own code; the sequence keeps its
            // start whenever the selected sets do not code it in fewer bits as a whole.
            let (masks, values, bits, row_agree) = if bits + values.sum() * scale < begin_code {
                let trace = forward(masked, &masked.family(&inputs, &masks), &target)?.1;
                let row_agree = agreement(&trace.values[masked.program.output], &target);
                (masks, values, bits, row_agree)
            } else {
                log::info!("eval sequence {e}: selection did not lower the start's code; the start stays");
                (begin_of(starts[e].as_ref(), &inputs, &masked.libraries, &coder.costs)?, begin_kl.clone(), begin_bits, begin_agree.clone())
            };
            log::info!(
                "eval sequence {e}: start L0 {:.1} KL {:.4} code {:.1}; selected L0 {:.1} KL {:.4} code {:.1} bits per token",
                begin_l0 / inputs.rows as f64,
                begin_kl.mean().unwrap_or(0.0),
                begin_code / inputs.rows as f64,
                sums(&masks, &values).0 / inputs.rows as f64,
                values.mean().unwrap_or(0.0),
                (bits + values.sum() * scale) / inputs.rows as f64
            );
            selected.add(&masks, &values, bits, &row_agree);
            seen.absorb(&masks, &previous_rows);
            if full {
                token_kl.extend(values.iter().copied());
                start_kl.extend(begin_kl.iter().copied());
                agree.extend(row_agree);
                start_agree.extend(begin_agree);
                for r in 0..inputs.rows {
                    let mut offset = 0;
                    for m in &masks {
                        indices.extend((0..m.ncols()).filter(|&c| m[[r, c]] > 0.0).map(|c| (offset + c) as i64));
                        offset += m.ncols();
                    }
                    indptr.push(indices.len() as i64);
                }
            }
            let progress = point_of(&selected, &started, e + 1);
            std::fs::write(format!("{}.progress.json", stem.display()), progress.to_string()).map_err(|e| e.to_string())?;
        }
        if full {
            let mut offsets = vec![0i64];
            for l in &masked.libraries {
                offsets.push(offsets[offsets.len() - 1] + l.v.nrows() as i64);
            }
            for (name, values) in [("indptr", &indptr), ("indices", &indices), ("offsets", &offsets), ("agree", &agree), ("start.agree", &start_agree)] {
                write_npy(&PathBuf::from(format!("{}.pass{pass}.{name}.npy", stem.display())), "<i8", values.iter().map(|v| v.to_le_bytes()))?;
            }
            for (name, values) in [("kl", &token_kl), ("start.kl", &start_kl)] {
                write_npy(&PathBuf::from(format!("{}.pass{pass}.{name}.npy", stem.display())), "<f8", values.iter().map(|v| v.to_le_bytes()))?;
            }
        }
        let point = point_of(&selected, &started, evaluated);
        let code = point["code"].as_f64().unwrap_or(f64::INFINITY);
        Ok((point, code, seen))
    };
    let write_points = |points: &[serde_json::Value]| -> Result<(), String> {
        std::fs::write(&out, serde_json::to_string_pretty(&json!({"points": points})).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    };
    if train == 0 {
        for pass in first_pass.. {
            let (point, code, seen) = evaluate(&masked, &context, &eval_starts, pass, 0, eval, true)?;
            eprintln!("eval {point}");
            points.push(point);
            write_points(&points)?;
            // Given sets code every pass by their held-out counts, so one pass says it all.
            let done = previous - code < 1.0 || sets.is_some();
            if !done {
                previous = code;
                context = seen;
            }
            let driver = json!({"pass": pass + 1, "next": 0, "pass_code": 0.0, "previous": finite(previous), "points": points, "done": done});
            save_state(driver, &masked, &context, &running, &[], &eval_starts)?;
            if done {
                break;
            }
        }
        return Ok(());
    }
    let report_every = (train / 4).max(1);
    let scale = observations / std::f64::consts::LN_2;
    let pieces_of = |masked: &Masked| masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>();
    // Every training sequence's current sets, all counted in `context`: each sequence is coded with
    // the counts of every other one (its own taken out while it is selected), starts from its
    // current sets (its given ones at first), and keeps the selected sets only when they code it in
    // fewer bits.
    let mut current: Vec<Option<Assigned>> = match saved_sets.as_mut() {
        // The saved counts already hold every current set.
        Some(saved) => (0..train).map(|s| as_assigned(saved[s].take())).collect(),
        None => {
            let current: Vec<Option<Assigned>> = (0..train).map(|s| sets.as_ref().filter(|x| x.fits(&masked.libraries)).map(|x| x.assigned(s))).collect();
            for assigned in current.iter().flatten() {
                assigned.absorb(&mut context, 1.0);
            }
            current
        }
    };
    for pass in first_pass.. {
        let first = if pass == first_pass { first_sequence } else { 0 };
        if pass != first_pass {
            pass_code = 0.0;
        }
        for s in first..train {
            let started = std::time::Instant::now();
            let inputs = sequence(s);
            let target = target_of(&inputs)?;
            let previous_rows = previous_inputs(&inputs);
            if let Some(old) = &current[s] {
                old.absorb(&mut context, -1.0);
            }
            let coder = context.coder(previous_rows.clone());
            let begin = match &current[s] {
                Some(old) => old.masks(&pieces_of(&masked)),
                None => start_masks(&inputs, &masked.libraries, &coder.costs)?,
            };
            let begin_kl = forward(&masked, &masked.family(&inputs, &begin), &target)?.0;
            let begin_code = coder.bits(&begin).sum() + begin_kl.sum() * scale;
            let (masks, kl) = select(&masked, &inputs, &target, begin, &coder, observations, samples)?;
            let (masks, kl) = if coder.bits(&masks).sum() + kl.sum() * scale < begin_code {
                (masks, kl)
            } else {
                let begin = match &current[s] {
                    Some(old) => old.masks(&pieces_of(&masked)),
                    None => start_masks(&inputs, &masked.libraries, &coder.costs)?,
                };
                (begin, begin_kl)
            };
            let sequence_code = (coder.bits(&masks).sum() + kl.sum() * scale) / inputs.rows as f64;
            pass_code += sequence_code;
            current[s] = Some(Assigned::of(&masks));
            let step = step_pieces(&mut masked, &inputs, &target, &masks, samples, 0xF00D + (pass * train + s) as u64, &mut running)?;
            let (l0, kl_sum, _) = sums(&masks, &kl);
            log::info!(
                "pass {pass} sequence {s}: L0 {:.1}, KL {:.4} per token, {sequence_code:.1} bits per token; step {:?}; {:.0}s",
                l0 / inputs.rows as f64,
                kl_sum / inputs.rows as f64,
                step.map(|(b, a)| (b / inputs.rows as f64, a / inputs.rows as f64)),
                started.elapsed().as_secs_f64()
            );
            // Four times a pass a progress point on the first four eval sequences; at the pass's
            // end the full eval set.
            let last = s + 1 == train;
            let mut masks = masks;
            if last || (s + 1) % report_every == 0 {
                // Growth, tested by the code: every piece this sequence lists two ways is split, the
                // sequence is selected again, and the split stays when its explanation and KL bits
                // per token fall by more than the added pieces' library bits spread over every
                // token trained so far.
                let trace = model.execute(&inputs, false).map_err(|e| e.to_string())?;
                let mut grown = Vec::new();
                let mut origins = Vec::new();
                let mut grown_masks = Vec::new();
                let mut added_reals = 0.0;
                for (k, library) in masked.libraries.iter().enumerate() {
                    let x = read_values(&trace, &original_sites[k])?;
                    let (bigger, m, origin) = split(library, &x, &masks[k]);
                    added_reals += ((bigger.v.nrows() - library.v.nrows()) * (library.v.ncols() + library.u.ncols())) as f64;
                    grown.push(bigger);
                    grown_masks.push(m);
                    origins.push(origin);
                }
                drop(trace);
                let candidate = Masked::build(model, original_sites.clone(), grown)?;
                // Both coded by the counts of every other sequence (`context` leaves this one out
                // until its sets are settled).
                let candidate_context = context.grown(&origins);
                let candidate_coder = candidate_context.coder(previous_rows.clone());
                let (candidate_masks, candidate_kl) = select(&candidate, &inputs, &target, grown_masks, &candidate_coder, observations, samples)?;
                let candidate_code = (candidate_coder.bits(&candidate_masks).sum() + candidate_kl.sum() * scale) / inputs.rows as f64;
                // The library as it stands after this sequence's step, on the same sets.
                let now_kl = forward(&masked, &masked.family(&inputs, &masks), &target)?.0;
                let sequence_code = (coder.bits(&masks).sum() + now_kl.sum() * scale) / inputs.rows as f64;
                let tokens_trained = ((pass * train + s + 1) * context_rows) as f64;
                let library_bits = added_reals * BITS_PER_REAL / tokens_trained;
                let kept = candidate_code + library_bits < sequence_code;
                log::info!(
                    "split test: {:.1} -> {:.1} bits per token, library {:.1} bits per token; {}",
                    sequence_code,
                    candidate_code,
                    library_bits,
                    if kept { "kept" } else { "refused" }
                );
                if kept {
                    masked = candidate;
                    context = candidate_context;
                    for (t, assigned) in current.iter_mut().enumerate() {
                        if t != s {
                            *assigned = assigned.as_ref().map(|a| a.grown(&origins));
                        }
                    }
                    for start in eval_starts.iter_mut() {
                        *start = start.as_ref().map(|a| a.grown(&origins));
                    }
                    masks = candidate_masks;
                }
                // Growth from what selection leaves out, tested the same way: per site, the leading
                // regression pieces of the left-out map that would recover more KL bits on this
                // sequence than their library bits (`gam_mpd::masked::dropped_atoms`), appended off.
                let family = masked.family(&inputs, &masks);
                let (base_kl, masked_trace, _) = forward(&masked, &family, &target)?;
                drop(family);
                let base_code = (context.coder(previous_rows.clone()).bits(&masks).sum() + base_kl.sum() * scale) / inputs.rows as f64;
                let mut grown = Vec::new();
                let mut grown_masks = Vec::new();
                let mut added = Vec::new();
                let mut added_reals = 0.0;
                for (k, library) in masked.libraries.iter().enumerate() {
                    let per_piece = (library.v.ncols() + library.u.ncols()) as f64 * BITS_PER_REAL * inputs.rows as f64 / tokens_trained;
                    let (v, u) = gam_mpd::masked::dropped_atoms(&masked, k, &masked_trace, &masks[k], &running, observations, per_piece)?;
                    added_reals += (v.nrows() * (v.ncols() + u.ncols())) as f64;
                    added.push(v.nrows());
                    grown_masks.push(ndarray::concatenate(Axis(1), &[masks[k].view(), Array2::<f64>::zeros((inputs.rows, v.nrows())).view()]).map_err(|e| e.to_string())?);
                    grown.push(gam_mpd::masked::with_pieces(library, &v, &u)?);
                }
                drop(masked_trace);
                if added.iter().any(|a| *a > 0) {
                    let candidate = Masked::build(model, original_sites.clone(), grown)?;
                    let candidate_context = context.extended(&added);
                    let candidate_coder = candidate_context.coder(previous_rows.clone());
                    let (candidate_masks, candidate_kl) = select(&candidate, &inputs, &target, grown_masks, &candidate_coder, observations, samples)?;
                    let candidate_code = (candidate_coder.bits(&candidate_masks).sum() + candidate_kl.sum() * scale) / inputs.rows as f64;
                    let library_bits = added_reals * BITS_PER_REAL / tokens_trained;
                    let kept = candidate_code + library_bits < base_code;
                    log::info!(
                        "dropped-atoms test: {} pieces, {base_code:.1} -> {candidate_code:.1} bits per token, library {library_bits:.1}; {}",
                        added.iter().sum::<usize>(),
                        if kept { "kept" } else { "refused" }
                    );
                    if kept {
                        masked = candidate;
                        context = candidate_context;
                        masks = candidate_masks;
                    }
                }
                let own = Assigned::of(&masks);
                own.absorb(&mut context, 1.0);
                current[s] = Some(own);
                let evaluated = if last { eval } else { eval.min(4) };
                let (point, _, _) = evaluate(&masked, &context, &eval_starts, pass, pass * train + s + 1, evaluated, last)?;
                eprintln!("eval {point}");
                points.push(point);
                write_points(&points)?;
            } else if let Some(own) = &current[s] {
                own.absorb(&mut context, 1.0);
            }
            let driver = json!({"pass": pass, "next": s + 1, "pass_code": pass_code, "previous": finite(previous), "points": points, "done": false});
            save_state(driver, &masked, &context, &running, &current, &eval_starts)?;
        }
        let code = pass_code / train as f64;
        log::info!("pass {pass}: {code:.1} bits per token");
        let done = previous - code < 1.0;
        if !done {
            previous = code;
        }
        let driver = json!({"pass": pass + 1, "next": 0, "pass_code": 0.0, "previous": finite(previous), "points": points, "done": done});
        save_state(driver, &masked, &context, &running, &current, &eval_starts)?;
        if done {
            break;
        }
    }
    Ok(())
}
