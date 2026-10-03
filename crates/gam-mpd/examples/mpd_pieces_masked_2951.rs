//! Per-input pieces of a language model trained through its own masked forward, on streamed
//! sequences (#2951).
//!
//! `mpd_pieces_masked_2951 EXPORT_DIR OUT.json OBSERVATIONS {wsvd|wsvd2|attribution|neurons|library:DIR} TRAIN EVAL [CONTEXT] [GPU] [SETS|-] [corner|box]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`) whose first
//! `TRAIN` token rows train the pieces and whose next `EVAL` rows evaluate them, `CONTEXT`
//! positions each (default 512). `GPU` (`off`, `auto` (default) or `required`) is the policy for the
//! proposal products (`gam_mpd::device`); every acceptance runs in float64 on the CPU. The sites are
//! the model's hidden-to-hidden maps (`gam_mpd::masked::sites`); their read moments and output
//! Fishers are measured on the training sequences (`gam_mpd::masked::site_statistics`), so any
//! imported model runs as it is. `wsvd` starts from each site's exact Fisher-whitened singular
//! pieces (`gam_mpd::pieces::fisher_svd`), and `wsvd2` grows those to twice as many on the first
//! training sequence (`gam_mpd::masked::split`). `attribution` starts from each site's
//! attribution dictionary on the training sequences (`gam_mpd::pieces::attribution_dictionary`, its
//! samples by `gam_mpd::masked::site_attributions`), and `neurons` from the model's own units on a
//! site's side that is a layer of units (`gam_mpd::pieces::unit_pieces`; Fisher-SVD elsewhere).
//! `library:DIR` starts from a given library: per
//! site `DIR/{site}.v.f64` (pieces × d_in) and `DIR/{site}.u.f64` (pieces × d_out), raw float64,
//! on the uncentred read with nothing beyond the pieces (a site without files stays native).
//!
//! The code is one total over the coded words, `Σ_words [Σ_{pieces on} bits(piece) + n KL/ln 2]`:
//! each word pays for the weights that ran on it, every piece's description at the precision the
//! KL needs (`gam_mpd::blocks::Generic`, in the measured statistics; `Coder::ran`). No counts are
//! kept, no library is amortised, and everything on is not free: each point also carries the
//! everything-on explanation (`all_on`) under the same code.
//!
//! `SETS` (with `library:DIR`) is a directory of given per-token sets for that library
//! (`bench/vpd_2951/vpd_sets_export.py`: `indptr.i64`, `indices.i64` as CSR over positions,
//! sequence after sequence in the export's order, pieces numbered site after site; `sites.txt`
//! naming the sites and their pieces). Every eval sequence then starts its selection from its given
//! sets (both halves of a split piece on where it was), and keeps them unless the selected sets
//! code it in fewer bits; every training sequence starts from its current sets, its given ones at
//! first. Each eval point also carries the given sets' own point (`start`) under the same code.
//!
//! `CLAIM` (`corner`, the default, or `box`) is what the explanation declares of its off
//! subcomponents (`gam_mpd::masked::Claim`): under `box` each off gate may be anywhere in `[0, 1]`,
//! the pieces are stepped on the KL expected over every off gate uniform
//! (`gam_mpd::masked::box_excess`, in the running written Fishers), each eval point also carries
//! that error (`kl_box`, `code_box`) beside the masks' own KL, and before training the predicted
//! box KL of the first five training sequences is logged against the KL of sampled gates.
//! Selection itself codes the masks' own KL under either claim; every point names its claim.
//! Before anything runs, every piece's read direction is restricted to the span holding all but
//! `1e-6` of its site's read variance on the measured sequences (`Masked::project_reads`).
//!
//! With `TRAIN` 0 nothing is trained: the eval sequences are selected once, a full eval. A running
//! point after each eval sequence goes to `OUT.progress.json`.
//!
//! The training sequences stream one at a time: each starts from its current sets (on its first
//! visit without `SETS`, the pieces whose own second-order KL bits in the global Fisher, on its
//! clean forward, exceed their description bits); its sets are selected exactly in the masked
//! forward (`gam_mpd::masked::select`) and kept only when they code it in fewer bits than its
//! start; then the pieces take one exact-gradient
//! step on it, preconditioned by the running read covariances and written Fishers of every sequence
//! seen
//! (`gam_mpd::masked::step_pieces`). Four times per pass the first four eval sequences, and at each
//! pass's end all of them, are selected one at a time with the current costs, and their mean active
//! pieces (L0), KL, description bits and code per token, and bits in the per-token frontier's code
//! (per site `ω(k + 1) + log₂ C(C, k)`), are appended to `OUT.json` as
//! `{points: [{l0, bits, kl, description_bits, code, …}]}`; a full eval also writes every piece's
//! description bits (`OUT.pass{P}.costs.npy`, float64, pieces numbered site after site), its selected sets as CSR
//! (`OUT.pass{P}.{indptr,indices,offsets}.npy`, pieces numbered site after site) and each eval
//! token's KL (`OUT.pass{P}.kl.npy`, float64, rows in order) and whether its argmax is the model's
//! (`OUT.pass{P}.agree.npy`, int64), the same for the starting sets (`OUT.pass{P}.start.{kl,agree}.npy`).
//! Passes repeat until one saves less than a bit per token.
//!
//! After every training sequence (every pass when nothing trains) the run's state goes to the
//! checkpoint `OUT.checkpoint/` (`gam_mpd::checkpoint`: the libraries, the running preconditioners, every sequence's current sets, the points and the pass and sequence
//! reached), and a run that finds one there resumes from it: killed at any point and restarted,
//! it writes what the uninterrupted run would have. Every draw is seeded by the pass and sequence,
//! so nothing else is kept. A finished run's checkpoint says so, and a restart then stops at once.
//! An eval's selection also saves its progress in `OUT.select/`: every finished eval sequence's
//! sets, and before every round of the one under way its whole selection state
//! (`gam_mpd::masked::Progress`: its sets, each input's interaction, each sequence's cap, the
//! round, the box excess), so a restart resumes it mid-sequence and selects exactly what the
//! uninterrupted run would.

use gam_mpd::checkpoint::{Saved, load, save};
use gam_mpd::codec::prefix_integer_len_bits;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{
    Claim, Coder, Context, select_boxed, Library, Masked, Running, Target, box_excess, box_excess_at, forward, kl_and_logits, matrix, read_values, score_only,
    select, site_statistics, sites, split, step_pieces,
};
use gam_mpd::operator_program::FamilyInputs;
use gam_mpd::operator_program::Node;
use gam_mpd::pieces::{Units, attribution_dictionary, fisher_svd, unit_pieces};
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

    /// Whether these sets number a library of `pieces` per site.
    fn fits(&self, pieces: &[usize]) -> bool {
        pieces.len() + 1 == self.offsets.len() && pieces.iter().zip(self.offsets.windows(2)).all(|(p, o)| *p == o[1] - o[0])
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

    /// These sets on a grown library (`split`'s `origins`): every new piece is on
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
    /// What the box claim adds to the KL, when it was measured (module note, `CLAIM`).
    excess: Option<f64>,
    /// What an attack on VPD's global box claim adds to the KL (`gam_mpd::masked::box_excess_at`),
    /// reported beside ours and never charged.
    attack: Option<f64>,
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
        let mut point = json!({
            "l0": self.l0 / t, "kl": self.kl / t, "bits": self.bits / t,
            "description_bits": self.explanation / t,
            "code": (self.explanation + self.kl * observations / std::f64::consts::LN_2) / t,
            "agree": self.agree / t,
        });
        if let Some(attack) = self.attack {
            point["kl_attack"] = json!((self.kl + attack) / t);
        }
        if let Some(excess) = self.excess {
            point["kl_box"] = json!((self.kl + excess) / t);
            point["code_box"] = json!((self.explanation + (self.kl + excess) * observations / std::f64::consts::LN_2) / t);
        }
        point
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
    let usage = "mpd_pieces_masked_2951 EXPORT_DIR OUT.json OBSERVATIONS {wsvd|wsvd2|attribution|neurons|library:DIR} TRAIN EVAL [CONTEXT] [GPU] [SETS|-] [corner|box]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let observations: f64 = args.get(3).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let start = args.get(4).ok_or(usage)?.clone();
    let given = start.strip_prefix("library:").map(PathBuf::from);
    if !["wsvd", "wsvd2", "attribution", "neurons"].contains(&start.as_str()) && given.is_none() {
        return Err(format!("unknown start {start}; {usage}"));
    }
    let train: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let eval: usize = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("EVAL: {e}"))?;
    let context: usize = args.get(7).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let gpu = args.get(8).map_or("auto", String::as_str);
    let sets_dir = args.get(9).filter(|a| *a != "-").map(PathBuf::from);
    let claim = match args.get(10).map(String::as_str) {
        None | Some("corner") => Claim::Corner,
        Some("box") => Claim::Box,
        Some(other) => return Err(format!("CLAIM {other}: expected corner or box; {usage}")),
    };
    if sets_dir.is_some() && given.is_none() {
        return Err(format!("SETS needs a library:DIR start; {usage}"));
    }
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::parse(gpu).ok_or_else(|| format!("GPU {gpu}: expected off, auto or required"))?);
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let context_rows = context;
    let sequence = |s: usize| -> FamilyInputs { family.select(&(s * context_rows..(s + 1) * context_rows).collect::<Vec<_>>()) };
    // The model's own logits, on the process's accelerator when it has one (float64 either way).
    let native = match gam_mpd::masked_device::device()? {
        Some(device) => Some(gam_mpd::device_program::DeviceProgram::compile(&device, model)?),
        None => None,
    };
    let target_of = |inputs: &FamilyInputs| -> Result<Target, String> {
        if let Some(native) = &native {
            let trace = native.forward(inputs)?;
            return Ok(Target::every_row(native.logits(&trace, 0, trace.rows)?));
        }
        Ok(Target::every_row(model.execute(inputs, false).map_err(|e| e.to_string())?.values[model.output].clone()))
    };
    let all_sites = sites(model);
    // The model's statistics on the training sequences (on the eval sequences when nothing trains):
    // the starting libraries' and the per-word description's metric.
    let measured_on = if train > 0 { 0..train } else { train..train + eval };
    let measured = site_statistics(model, &all_sites, measured_on.clone().map(sequence), 2, 0x5EED)?;
    // Each word pays for the weights that ran on it (`gam_mpd::blocks::Generic`): the reads are
    // uncentred, so their metric is the second moment about zero.
    let description = gam_mpd::blocks::Generic::new(
        &measured
            .iter()
            .map(|m| gam_mpd::pieces::Site { w: m.w.clone(), second_moment: m.second_moment.clone(), mean: Array1::zeros(m.mean.len()), fisher: m.fisher.clone() })
            .collect::<Vec<_>>(),
        observations,
    );
    let statistics: Vec<Option<gam_mpd::pieces::Site>> = measured.into_iter().map(Some).collect();
    // `attribution`: every site's per-input samples on the training sequences, gathered a group of
    // sites at a time so a group's samples stay within `SAMPLE_BYTES` (fewer sequences for a site
    // too wide for that alone).
    const SAMPLE_BYTES: usize = 1 << 30;
    let mut attributions: Vec<Option<gam_mpd::pieces::Attributions>> = all_sites.iter().map(|_| None).collect();
    if start == "attribution" {
        let widths: Vec<usize> = statistics.iter().map(|m| m.as_ref().map_or(0, |m| m.w.nrows() + m.w.ncols())).collect();
        let mut first = 0;
        while first < all_sites.len() {
            let mut last = first + 1;
            while last < all_sites.len() && widths[first..=last].iter().sum::<usize>() * measured_on.len() * context_rows * 4 <= SAMPLE_BYTES {
                last += 1;
            }
            let width: usize = widths[first..last].iter().sum();
            let sequences = (SAMPLE_BYTES / (width * context_rows * 4).max(1)).clamp(1, measured_on.len());
            let started = std::time::Instant::now();
            let gathered = gam_mpd::masked::site_attributions(model, &all_sites[first..last], measured_on.clone().take(sequences).map(sequence), 0xA77)?;
            eprintln!("attributions of sites {first}..{last} on {sequences} sequences, {:.0}s", started.elapsed().as_secs_f64());
            for (k, samples) in gathered.into_iter().enumerate() {
                attributions[first + k] = Some(samples);
            }
            first = last;
        }
    }
    // `neurons`: the side of a site that is a layer of units, a written node feeding a pointwise
    // map or a read node that is one.
    let units_of = |site: &gam_mpd::masked::Site| -> Option<Units> {
        let feeds_pointwise = |n: usize| model.nodes.iter().any(|node| matches!(node, Node::Pointwise { input, .. } if *input == n));
        if site.writes.len() == 1 && feeds_pointwise(site.writes[0]) {
            Some(Units::Written)
        } else if site.reads.len() == 1 && matches!(model.nodes[site.reads[0]], Node::Pointwise { .. }) {
            Some(Units::Read)
        } else {
            None
        }
    };
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    // Per site, the written Fisher of the clean model (empty when not measured).
    let mut fishers: Vec<Array2<f64>> = Vec::new();
    for ((site, measured), samples) in all_sites.iter().zip(statistics).zip(attributions) {
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
        let library = match (start.as_str(), samples) {
            ("neurons", _) => match units_of(site) {
                Some(side) => unit_pieces(&measured.w, side),
                None => fisher_svd(&measured)?,
            },
            ("attribution", Some(samples)) => {
                // Twice the map's rank seeded; an atom stays when it pays its library bits (32 per
                // real), spread over every training input.
                let (d_out, d_in) = measured.w.dim();
                let bits_per_piece = (d_in + d_out) as f64 * 32.0 * samples.reads.nrows() as f64 / (measured_on.len() * context_rows) as f64;
                let started = std::time::Instant::now();
                let (library, report) = attribution_dictionary(&measured, &samples, 2 * d_in.min(d_out), observations, bits_per_piece, 0xD1C7)?;
                eprintln!("{}: attribution dictionary {report:?}, {:.0}s", site.name, started.elapsed().as_secs_f64());
                library
            }
            _ => fisher_svd(&measured)?,
        };
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
    let mut run = json!({"observations": observations, "start": start, "train": train, "eval": eval, "context": context_rows});
    if claim == Claim::Box {
        run["claim"] = json!("box");
    }
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
    // Every piece's description bits per word it runs on (module note).
    let costs_of = |masked: &Masked| -> Result<Vec<Array1<f64>>, String> {
        use gam_mpd::blocks::Describe;
        (0..masked.sites.len())
            .map(|k| {
                let library = masked.library(k)?;
                (0..library.v.nrows())
                    .map(|c| description.bits(k, library.u.slice(ndarray::s![c..c + 1, ..]), library.v.slice(ndarray::s![c..c + 1, ..])))
                    .collect::<Result<Array1<f64>, String>>()
            })
            .collect()
    };
    // A sequence's start: the pieces whose own second-order KL bits, `n a² uᵀBu / (2 ln 2)` with
    // `a = v · x` on the clean forward, exceed their description bits.
    let start_masks = |inputs: &FamilyInputs, masked: &Masked, costs: &[Array1<f64>]| -> Result<Vec<Array2<f64>>, String> {
        let trace = model.execute(inputs, false).map_err(|e| e.to_string())?;
        let scale = observations / (2.0 * std::f64::consts::LN_2);
        let mut masks = Vec::new();
        for k in 0..masked.sites.len() {
            let library = masked.library(k)?;
            if fishers[k].nrows() != library.u.ncols() {
                return Err(format!("{}: no measured Fisher to start from", original_sites[k].name));
            }
            let x = read_values(&trace, &original_sites[k])?;
            let a = x.dot(&library.v.t());
            // Each piece's second-order weight `uᵀ B u` in the global Fisher.
            let weights = (&library.u.dot(&fishers[k]) * &library.u).sum_axis(Axis(1));
            masks.push(Array2::from_shape_fn(a.dim(), |(r, c)| if scale * a[[r, c]] * a[[r, c]] * weights[c] > costs[k][c] { 1.0 } else { 0.0 }));
        }
        Ok(masks)
    };
    // A sequence's start: its given sets (on the library as it has grown since), else `start_masks`.
    let begin_of = |given: Option<&Assigned>, inputs: &FamilyInputs, masked: &Masked, costs: &[Array1<f64>]| -> Result<Vec<Array2<f64>>, String> {
        match given {
            Some(assigned) => Ok(assigned.masks(&masked.all_pieces())),
            None => start_masks(inputs, masked, costs),
        }
    };
    // Each eval sequence's given sets, kept on the library as it grows.
    let mut eval_starts: Vec<Option<Assigned>> =
        (0..eval).map(|e| sets.as_ref().filter(|x| x.fits(&masked.all_pieces())).map(|x| x.assigned(train + e))).collect();
    // `wsvd2`: the Fisher-SVD library grown to twice its pieces on the first training sequence,
    // every piece its listing inputs use in two ways split in two (`gam_mpd::masked::split`).
    // Every piece reads only the span holding all but `LEFT_OUT` of its site's read variance on
    // the measured sequences: what it reads off that span is unidentified (module note).
    const LEFT_OUT: f64 = 1e-6;
    if resumed.is_none() {
        let started = std::time::Instant::now();
        let moments = gam_mpd::masked::site_second_moments(model, &original_sites, measured_on.clone().map(sequence))?;
        let mut kept = Vec::new();
        for (k, (mean, covariance)) in moments.iter().enumerate() {
            kept.push(masked.project_reads(k, mean, covariance, LEFT_OUT)?);
        }
        eprintln!("reads projected to {kept:?} dimensions ({:.0}s)", started.elapsed().as_secs_f64());
    }
    // The library selected over (`OUT.library/`, the `library:DIR` layout), so other scorers read
    // exactly its subcomponents: after projection, and again at every full eval (it trains).
    let dump_library = |masked: &Masked| -> Result<(), String> {
        let dir = out.with_extension("library");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let raw = |m: &Array2<f64>| m.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for (k, site) in original_sites.iter().enumerate() {
            let library = masked.library(k)?;
            std::fs::write(dir.join(format!("{}.v.f64", site.name)), raw(&library.v)).map_err(|e| e.to_string())?;
            std::fs::write(dir.join(format!("{}.u.f64", site.name)), raw(&library.u)).map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    dump_library(&masked)?;
    let mut costs = costs_of(&masked)?;
    if start == "wsvd2" && resumed.is_none() {
        let inputs = sequence(0);
        let target = target_of(&inputs)?;
        let coder = Coder::ran(costs.clone(), inputs.rows);
        let begin = start_masks(&inputs, &masked, &coder.costs)?;
        let (masks, _) = select(&masked, &inputs, &target, begin, &coder, observations, samples)?;
        let trace = model.execute(&inputs, false).map_err(|e| e.to_string())?;
        let mut grown = Vec::new();
        for (k, site) in original_sites.iter().enumerate() {
            let x = read_values(&trace, site)?;
            grown.push(split(&masked.library(k)?, &x, &masks[k]).0);
        }
        eprintln!("grown to {} pieces", grown.iter().map(|l| l.v.nrows()).sum::<usize>());
        masked = Masked::build(model, original_sites.clone(), grown)?;
        costs = costs_of(&masked)?;
    }
    // Resuming: the saved libraries, counts, preconditioners, sets and points.
    let (mut running, mut points, mut previous, mut saved_sets, first_pass, first_sequence, mut pass_code) = match resumed {
        Some(r) => {
            masked = Masked::build(model, original_sites.clone(), r.libraries)?;
            costs = costs_of(&masked)?;
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
    // The description code keeps no counts; the checkpoint's are empty.
    let no_counts = Context::new(&[]);
    let save_state = |driver: serde_json::Value, masked: &Masked, running: &Running, current: &[Option<Assigned>], eval_starts: &[Option<Assigned>]| -> Result<(), String> {
        let mut driver = driver;
        driver["run"] = run.clone();
        let sets = current.iter().chain(eval_starts).map(|a| a.as_ref().map(|a| &a.sites)).collect();
        let libraries = (0..masked.sites.len()).map(|k| masked.library(k)).collect::<Result<Vec<_>, _>>()?;
        save(&checkpoint, &Saved { driver: &driver, libraries: &libraries, context: &no_counts, running, sets })
    };
    // JSON has no infinity: a first pass's `previous` is null.
    let finite = |v: f64| if v.is_finite() { json!(v) } else { serde_json::Value::Null };
    let stem = out.with_extension("");
    // Select the first `evaluated` eval sequences under the description code `costs`; a full eval
    // also writes the sets, each token's KL and argmax agreement. Returns the point and the code per
    // token. Each point also carries everything on (`all_on`): its description and KL.
    // With `fishers` (the running written Fishers, under the box claim), each point also carries the
    // box claim's error.
    let evaluate = |masked: &Masked,
                    costs: &[Array1<f64>],
                    starts: &[Option<Assigned>],
                    fishers: Option<&[Array2<f64>]>,
                    pass: usize,
                    trained: usize,
                    evaluated: usize,
                    full: bool|
     -> Result<(serde_json::Value, f64), String> {
        let mut all_on = Tally::default();
        // The selected sets, and the sets selection started from.
        let (mut selected, mut started) = (Tally::default(), Tally::default());
        if fishers.is_some() {
            selected.excess = Some(0.0);
            started.excess = Some(0.0);
            selected.attack = Some(0.0);
            started.attack = Some(0.0);
            // Everything on leaves no gate free: no excess.
            all_on.excess = Some(0.0);
        }
        // What the box claim adds on a sequence's sets.
        let excess_of = |inputs: &FamilyInputs, _target: &Target, masks: &[Array2<f64>], fishers: &[Array2<f64>]| -> Result<f64, String> {
            Ok(gam_mpd::masked::box_upper_at(masked, inputs, masks, fishers)?.sum())
        };
        // Sets as CSR over all pieces (sites in order), so other context codes can score them.
        let mut indptr: Vec<i64> = vec![0];
        let mut indices: Vec<i64> = Vec::new();
        let (mut token_kl, mut start_kl): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let (mut agree, mut start_agree): (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
        let point_of = |selected: &Tally, started: &Tally, all_on: &Tally, evaluated: usize| {
            let mut point = selected.json(observations);
            point["start"] = started.json(observations);
            point["all_on"] = all_on.json(observations);
            for (key, value) in [
                ("pieces", json!(masked.all_pieces().iter().sum::<usize>())),
                ("pass", json!(pass)),
                ("sequences_trained", json!(trained)),
                ("observations", json!(observations)),
                ("eval_sequences", json!(evaluated)),
                ("claim", json!(if claim == Claim::Box { "box" } else { "corner" })),
            ] {
                point[key] = value;
            }
            point
        };
        let scale = observations / std::f64::consts::LN_2;
        // Selection's progress (module note): every eval sequence's finished sets and the one under
        // way's selection state, saved atomically in `OUT.select/`, so a restart resumes them.
        let progress_dir = PathBuf::from(format!("{}.select", stem.display()));
        // Only this run's progress at this point counts: a directory left by another run (the
        // checkpoint removed to start over) is not this run's.
        let resumed = load(&progress_dir)?.filter(|r| r.driver["run"] == run && r.driver["pass"] == json!(pass) && r.driver["trained"] == json!(trained));
        let mut finished: Vec<Option<gam_mpd::checkpoint::SparseSets>> = vec![None; evaluated];
        // The sequence under way and its selection state: its sets, and the rest as JSON (every
        // real as its bits, so it comes back bit for bit).
        let mut under_way: Option<(usize, gam_mpd::checkpoint::SparseSets, serde_json::Value)> = None;
        if let Some(r) = resumed {
            let complete: Vec<bool> = r.driver["complete"].as_array().map(|a| a.iter().map(|v| v == &json!(true)).collect()).unwrap_or_default();
            let state = r.driver["under_way"].clone();
            for (e, sets) in r.sets.into_iter().enumerate().take(evaluated) {
                if complete.get(e).copied().unwrap_or(false) {
                    finished[e] = sets;
                } else if let Some(sets) = sets.filter(|_| state["sequence"] == json!(e)) {
                    under_way = Some((e, sets, state.clone()));
                }
            }
            eprintln!(
                "selection resumed: {} sequences finished, {}",
                finished.iter().flatten().count(),
                under_way.as_ref().map_or("none under way".to_string(), |(e, _, s)| format!("sequence {e} under way at round {}", s["round"]))
            );
        }
        let as_bits = |values: &[f64]| json!(values.iter().map(|v| v.to_bits()).collect::<Vec<u64>>());
        let reals = |value: &serde_json::Value| -> Result<Vec<f64>, String> {
            value.as_array().ok_or("selection state: not a list")?.iter().map(|v| v.as_u64().map(f64::from_bits).ok_or_else(|| "selection state: not a real's bits".to_string())).collect()
        };
        let save_progress = |finished: &[Option<gam_mpd::checkpoint::SparseSets>], current: Option<(usize, &gam_mpd::masked::Progress<'_>)>| -> Result<(), String> {
            let complete: Vec<bool> = finished.iter().map(Option::is_some).collect();
            let current_sets = current.map(|(e, progress)| (e, Assigned::of(progress.masks).sites));
            let sets: Vec<Option<&gam_mpd::checkpoint::SparseSets>> =
                (0..evaluated).map(|e| finished[e].as_ref().or(current_sets.as_ref().filter(|(u, _)| *u == e).map(|(_, s)| s))).collect();
            let mut driver = json!({"run": run, "pass": pass, "trained": trained, "complete": complete});
            if let Some((e, progress)) = current {
                driver["under_way"] = json!({
                    "sequence": e,
                    "round": progress.round,
                    "alpha": as_bits(progress.alpha),
                    "cap": progress.cap,
                    "excess": progress.excess.map(|x| as_bits(&x.to_vec())),
                });
            }
            save(&progress_dir, &Saved { driver: &driver, libraries: &[], context: &Context::new(&[]), running: &Running::default(), sets })
        };
        for e in 0..evaluated {
            let inputs = sequence(train + e);
            let target = target_of(&inputs)?;
            let coder = Coder::ran(costs.to_vec(), inputs.rows);
            // Everything on: the trivial explanation, which the description code must not favour.
            {
                let on: Vec<Array2<f64>> = masked.all_pieces().iter().map(|p| Array2::ones((inputs.rows, *p))).collect();
                let (kl, logits) = kl_and_logits(masked, &masked.family(&inputs, &on), &target)?;
                let agree = agreement(&logits, &target);
                all_on.add(&on, &kl, coder.bits(&on).sum(), &agree);
            }
            let begin = begin_of(starts[e].as_ref(), &inputs, masked, &coder.costs)?;
            // The start's own exact KL, explanation bits and argmax agreement.
            let (begin_kl, begin_agree) = {
                let (kl, logits) = kl_and_logits(masked, &masked.family(&inputs, &begin), &target)?;
                (kl, agreement(&logits, &target))
            };
            let begin_bits = coder.bits(&begin).sum();
            started.add(&begin, &begin_kl, begin_bits, &begin_agree);
            // Under the box claim the error is the KL expected over every off gate (module note).
            let begin_excess = match fishers {
                Some(f) => excess_of(&inputs, &target, &begin, f)?,
                None => 0.0,
            };
            if let Some(excess) = started.excess.as_mut() {
                *excess += begin_excess;
            }
            if let (Some(f), Some(attack)) = (fishers, started.attack.as_mut()) {
                *attack += box_excess_at(masked, &inputs, &target, &begin, f)?.sum();
            }
            let begin_code = begin_bits + (begin_kl.sum() + begin_excess) * scale;
            let begin_l0 = sums(&begin, &begin_kl).0;
            let pieces = masked.all_pieces();
            let (masks, values) = if let Some(sets) = finished[e].clone() {
                // Finished before a restart: its sets stand.
                let masks = Assigned { rows: inputs.rows, sites: sets }.masks(&pieces);
                let values = score_only(masked, &masked.family(&inputs, &masks), &target)?;
                (masks, values)
            } else {
                // Under way before a restart: its saved state, from the same start.
                let resume = match under_way.take() {
                    Some((u, sets, state)) if u == e => Some(gam_mpd::masked::Resume {
                        masks: Assigned { rows: inputs.rows, sites: sets }.masks(&pieces),
                        alpha: reals(&state["alpha"])?,
                        cap: state["cap"].as_array().ok_or("selection state: no caps")?.iter().map(|c| c.as_u64().map(|c| c as usize).ok_or("selection state: a cap")).collect::<Result<_, _>>()?,
                        round: state["round"].as_u64().ok_or("selection state: no round")?,
                        excess: if state["excess"].is_null() { None } else { Some(Array1::from(reals(&state["excess"])?)) },
                    }),
                    _ => None,
                };
                // The state before every round is saved; the returned KL is the float64 one of
                // the masks alone, as a restart scores them.
                let mut checkpoint = |progress: &gam_mpd::masked::Progress<'_>| save_progress(&finished, Some((e, progress)));
                let selected = gam_mpd::masked::select_resumable(
                    masked,
                    &inputs,
                    &target,
                    begin,
                    resume,
                    &coder,
                    observations,
                    samples,
                    fishers,
                    &mut |_: &gam_mpd::masked::Round<'_>| Ok(()),
                    &mut checkpoint,
                )?;
                finished[e] = Some(Assigned::of(&selected.0).sites);
                save_progress(&finished, None)?;
                selected
            };
            let bits = coder.bits(&masks).sum();
            let excess = match fishers {
                Some(f) => excess_of(&inputs, &target, &masks, f)?,
                None => 0.0,
            };
            // Selection keeps each sequence's flips by its own code; the sequence keeps its start
            // whenever the selected sets do not code it in fewer bits as a whole.
            let (masks, values, bits, excess, row_agree) = if bits + (values.sum() + excess) * scale < begin_code {
                let row_agree = agreement(&kl_and_logits(masked, &masked.family(&inputs, &masks), &target)?.1, &target);
                (masks, values, bits, excess, row_agree)
            } else {
                log::info!("eval sequence {e}: selection did not lower the start's code; the start stays");
                (begin_of(starts[e].as_ref(), &inputs, masked, &coder.costs)?, begin_kl.clone(), begin_bits, begin_excess, begin_agree.clone())
            };
            log::info!(
                "eval sequence {e}: start L0 {:.1} KL {:.4} code {:.1}; selected L0 {:.1} KL {:.4} code {:.1} bits per token (the code under the claim)",
                begin_l0 / inputs.rows as f64,
                begin_kl.mean().unwrap_or(0.0),
                begin_code / inputs.rows as f64,
                sums(&masks, &values).0 / inputs.rows as f64,
                values.mean().unwrap_or(0.0),
                (bits + (values.sum() + excess) * scale) / inputs.rows as f64
            );
            selected.add(&masks, &values, bits, &row_agree);
            if let Some(total) = selected.excess.as_mut() {
                *total += excess;
            }
            if let (Some(f), Some(attack)) = (fishers, selected.attack.as_mut()) {
                *attack += box_excess_at(masked, &inputs, &target, &masks, f)?.sum();
            }
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
            let progress = point_of(&selected, &started, &all_on, e + 1);
            std::fs::write(format!("{}.progress.json", stem.display()), progress.to_string()).map_err(|e| e.to_string())?;
        }
        if full {
            let mut offsets = vec![0i64];
            for p in masked.all_pieces() {
                offsets.push(offsets[offsets.len() - 1] + p as i64);
            }
            for (name, values) in [("indptr", &indptr), ("indices", &indices), ("offsets", &offsets), ("agree", &agree), ("start.agree", &start_agree)] {
                write_npy(&PathBuf::from(format!("{}.pass{pass}.{name}.npy", stem.display())), "<i8", values.iter().map(|v| v.to_le_bytes()))?;
            }
            for (name, values) in [("kl", &token_kl), ("start.kl", &start_kl)] {
                write_npy(&PathBuf::from(format!("{}.pass{pass}.{name}.npy", stem.display())), "<f8", values.iter().map(|v| v.to_le_bytes()))?;
            }
        }
        let point = point_of(&selected, &started, &all_on, evaluated);
        let code = point["code"].as_f64().unwrap_or(f64::INFINITY);
        if full {
            dump_library(masked)?;
            let flat: Vec<f64> = costs.iter().flat_map(|c| c.iter().copied()).collect();
            write_npy(&PathBuf::from(format!("{}.pass{pass}.costs.npy", stem.display())), "<f8", flat.iter().map(|v| v.to_le_bytes()))?;
        }
        Ok((point, code))
    };
    let write_points = |points: &[serde_json::Value]| -> Result<(), String> {
        std::fs::write(&out, serde_json::to_string_pretty(&json!({"points": points})).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    };
    if train == 0 {
        // The description code is fixed by the library, so one pass says it all.
        if first_pass == 0 {
            let (point, code) = evaluate(&masked, &costs, &eval_starts, (claim == Claim::Box).then_some(fishers.as_slice()), 0, 0, eval, true)?;
            eprintln!("eval {point}");
            points.push(point);
            write_points(&points)?;
            let driver = json!({"pass": 1, "next": 0, "pass_code": 0.0, "previous": finite(code), "points": points, "done": true});
            save_state(driver, &masked, &running, &[], &eval_starts)?;
        }
        return Ok(());
    }
    let report_every = (train / 4).max(1);
    let scale = observations / std::f64::consts::LN_2;
    // Every training sequence's current sets: each starts from them (its given ones at first) and
    // keeps the selected sets only when they code it in fewer bits.
    let mut current: Vec<Option<Assigned>> = match saved_sets.as_mut() {
        Some(saved) => (0..train).map(|s| as_assigned(saved[s].take())).collect(),
        None => (0..train).map(|s| sets.as_ref().filter(|x| x.fits(&masked.all_pieces())).map(|x| x.assigned(s))).collect(),
    };
    // Selection and the error under the run's claim, on any library of these sites (a grown one's
    // sites and Fishers are the same).
    let select_claimed = |m: &Masked, inputs: &FamilyInputs, target: &Target, begin: Vec<Array2<f64>>, coder: &Coder| match claim {
        Claim::Box => select_boxed(m, inputs, target, begin, coder, observations, samples, &fishers),
        Claim::Corner => select(m, inputs, target, begin, coder, observations, samples),
    };
    let excess_on = |m: &Masked, inputs: &FamilyInputs, _target: &Target, masks: &[Array2<f64>]| -> Result<f64, String> {
        match claim {
            Claim::Box => Ok(gam_mpd::masked::box_upper_at(m, inputs, masks, &fishers)?.sum()),
            Claim::Corner => Ok(0.0),
        }
    };
    // The box claim's error against the KL of sampled gates (module note, `CLAIM`): on the first
    // five training sequences at their start sets, every off gate drawn uniform.
    if claim == Claim::Box && first_pass == 0 && first_sequence == 0 {
        let mut state = 0x5A3D_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for s in 0..train.min(5) {
            let inputs = sequence(s);
            let target = target_of(&inputs)?;
            let coder = Coder::ran(costs.clone(), inputs.rows);
            let masks = match &current[s] {
                Some(assigned) => assigned.masks(&masked.all_pieces()),
                None => start_masks(&inputs, &masked, &coder.costs)?,
            };
            let family = masked.family(&inputs, &masks);
            let (kl, trace, cotangent) = forward(&masked, &family, &target)?;
            let fishers = gam_mpd::masked::fisher(&masked, &family, &trace, &target, samples, 0xB0C5 + s as u64, true)?
                .into_iter()
                .map(|(_, f)| f.ok_or("no written Fisher"))
                .collect::<Result<Vec<_>, _>>()?;
            let excess = box_excess(&masked, &family, &trace, &masks, cotangent, &fishers, false)?.0.sum();
            drop((family, trace));
            let draws = 8;
            let sampled: Vec<f64> = (0..draws)
                .map(|_| {
                    let gates: Vec<Array2<f64>> = masks.iter().map(|m| m.mapv(|x| if x > 0.0 { 1.0 } else { uniform() })).collect();
                    score_only(&masked, &masked.family(&inputs, &gates), &target).map(|r| r.sum())
                })
                .collect::<Result<_, _>>()?;
            let mean = sampled.iter().sum::<f64>() / draws as f64;
            let spread = (sampled.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (draws * (draws - 1)) as f64).sqrt();
            let rows = inputs.rows as f64;
            log::info!(
                "box claim check, sequence {s}: KL at the masks {:.4}, box KL {:.4} predicted, {:.4} ± {:.4} over {draws} sampled gates, per token",
                kl.sum() / rows,
                (kl.sum() + excess) / rows,
                mean / rows,
                spread / rows
            );
        }
    }
    for pass in first_pass.. {
        let first = if pass == first_pass { first_sequence } else { 0 };
        if pass != first_pass {
            pass_code = 0.0;
        }
        for s in first..train {
            let started = std::time::Instant::now();
            let inputs = sequence(s);
            let target = target_of(&inputs)?;
            let coder = Coder::ran(costs.clone(), inputs.rows);
            let begin = match &current[s] {
                Some(old) => old.masks(&masked.all_pieces()),
                None => start_masks(&inputs, &masked, &coder.costs)?,
            };
            let begin_kl = score_only(&masked, &masked.family(&inputs, &begin), &target)?;
            // The error under the claim: under the box, the masks' KL plus the box's excess.
            let begin_excess = excess_on(&masked, &inputs, &target, &begin)?;
            let begin_code = coder.bits(&begin).sum() + (begin_kl.sum() + begin_excess) * scale;
            let (masks, kl) = select_claimed(&masked, &inputs, &target, begin, &coder)?;
            let excess = excess_on(&masked, &inputs, &target, &masks)?;
            let (masks, kl, excess) = if coder.bits(&masks).sum() + (kl.sum() + excess) * scale < begin_code {
                (masks, kl, excess)
            } else {
                let begin = match &current[s] {
                    Some(old) => old.masks(&masked.all_pieces()),
                    None => start_masks(&inputs, &masked, &coder.costs)?,
                };
                (begin, begin_kl, begin_excess)
            };
            let sequence_code = (coder.bits(&masks).sum() + (kl.sum() + excess) * scale) / inputs.rows as f64;
            pass_code += sequence_code;
            current[s] = Some(Assigned::of(&masks));
            let step = step_pieces(&mut masked, &inputs, &target, &masks, samples, 0xF00D + (pass * train + s) as u64, &mut running, claim)?;
            // The step moved the pieces and with them their descriptions. The costs follow the
            // library at once, so a resumed run, which prices the saved library, prices what the
            // uninterrupted run did.
            costs = costs_of(&masked)?;
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
                // sequence is selected again, and the split stays when its description and KL bits
                // per token fall (each word pays for the pieces that run on it).
                let trace = model.execute(&inputs, false).map_err(|e| e.to_string())?;
                let mut grown = Vec::new();
                let mut origins = Vec::new();
                let mut grown_masks = Vec::new();
                for (k, site) in original_sites.iter().enumerate() {
                    let library = masked.library(k)?;
                    let x = read_values(&trace, site)?;
                    let (bigger, m, origin) = split(&library, &x, &masks[k]);
                    grown.push(bigger);
                    grown_masks.push(m);
                    origins.push(origin);
                }
                drop(trace);
                let candidate = Masked::build(model, original_sites.clone(), grown)?;
                let candidate_costs = costs_of(&candidate)?;
                let candidate_coder = Coder::ran(candidate_costs.clone(), inputs.rows);
                let (candidate_masks, candidate_kl) = select_claimed(&candidate, &inputs, &target, grown_masks, &candidate_coder)?;
                let candidate_excess = excess_on(&candidate, &inputs, &target, &candidate_masks)?;
                let candidate_code = (candidate_coder.bits(&candidate_masks).sum() + (candidate_kl.sum() + candidate_excess) * scale) / inputs.rows as f64;
                // The library as it stands after this sequence's step, on the same sets, priced by
                // its own descriptions as the candidate is by its.
                let now_kl = score_only(&masked, &masked.family(&inputs, &masks), &target)?;
                let now_excess = excess_on(&masked, &inputs, &target, &masks)?;
                let sequence_code = (Coder::ran(costs.clone(), inputs.rows).bits(&masks).sum() + (now_kl.sum() + now_excess) * scale) / inputs.rows as f64;
                let kept = candidate_code < sequence_code;
                log::info!("split test: {sequence_code:.1} -> {candidate_code:.1} bits per token; {}", if kept { "kept" } else { "refused" });
                if kept {
                    masked = candidate;
                    costs = candidate_costs;
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
                let base_excess = excess_on(&masked, &inputs, &target, &masks)?;
                let base_code = (Coder::ran(costs.clone(), inputs.rows).bits(&masks).sum() + (base_kl.sum() + base_excess) * scale) / inputs.rows as f64;
                let mut grown = Vec::new();
                let mut grown_masks = Vec::new();
                let mut added = Vec::new();
                for k in 0..masked.sites.len() {
                    let library = masked.library(k)?;
                    // A new piece pays its description on every word of the sequence it runs on:
                    // the site's mean piece description, per word.
                    let per_piece = costs[k].mean().unwrap_or(0.0) * inputs.rows as f64;
                    let (v, u) = gam_mpd::masked::dropped_atoms(&masked, k, &masked_trace, &masks[k], &running, observations, per_piece)?;
                    added.push(v.nrows());
                    grown_masks.push(ndarray::concatenate(Axis(1), &[masks[k].view(), Array2::<f64>::zeros((inputs.rows, v.nrows())).view()]).map_err(|e| e.to_string())?);
                    grown.push(gam_mpd::masked::with_pieces(&library, &v, &u)?);
                }
                drop(masked_trace);
                if added.iter().any(|a| *a > 0) {
                    let candidate = Masked::build(model, original_sites.clone(), grown)?;
                    let candidate_costs = costs_of(&candidate)?;
                    let candidate_coder = Coder::ran(candidate_costs.clone(), inputs.rows);
                    let (candidate_masks, candidate_kl) = select_claimed(&candidate, &inputs, &target, grown_masks, &candidate_coder)?;
                    let candidate_excess = excess_on(&candidate, &inputs, &target, &candidate_masks)?;
                    let candidate_code = (candidate_coder.bits(&candidate_masks).sum() + (candidate_kl.sum() + candidate_excess) * scale) / inputs.rows as f64;
                    let kept = candidate_code < base_code;
                    log::info!(
                        "dropped-atoms test: {} pieces, {base_code:.1} -> {candidate_code:.1} bits per token; {}",
                        added.iter().sum::<usize>(),
                        if kept { "kept" } else { "refused" }
                    );
                    if kept {
                        masked = candidate;
                        masks = candidate_masks;
                    }
                }
                current[s] = Some(Assigned::of(&masks));
                costs = costs_of(&masked)?;
                let evaluated = if last { eval } else { eval.min(4) };
                let (point, _) = evaluate(&masked, &costs, &eval_starts, (claim == Claim::Box).then_some(fishers.as_slice()), pass, pass * train + s + 1, evaluated, last)?;
                eprintln!("eval {point}");
                points.push(point);
                write_points(&points)?;
            }
            let driver = json!({"pass": pass, "next": s + 1, "pass_code": pass_code, "previous": finite(previous), "points": points, "done": false});
            save_state(driver, &masked, &running, &current, &eval_starts)?;
        }
        let code = pass_code / train as f64;
        log::info!("pass {pass}: {code:.1} bits per token");
        let done = previous - code < 1.0;
        if !done {
            previous = code;
        }
        let driver = json!({"pass": pass + 1, "next": 0, "pass_code": 0.0, "previous": finite(previous), "points": points, "done": done});
        save_state(driver, &masked, &running, &current, &eval_starts)?;
        if done {
            break;
        }
    }
    Ok(())
}
