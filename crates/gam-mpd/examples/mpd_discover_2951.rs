//! Discovery of rare mechanisms from a model's weights and its own forward on ordinary text (#2951).
//!
//! A mechanism that acts only in rare contexts writes through directions ordinary text barely uses.
//! In a site's Fisher-whitened singular library (`gam_mpd::pieces::fisher_svd`, fitted on a few
//! calibration sequences) those directions are the low-importance pieces, and on the rare inputs
//! their amplitudes are large: a piece's second-order KL share at a word,
//! `e_c = z_c² · u_cᵀ F u_c` (`z_c = v_c · x` its amplitude, `F` the written node's Fisher), is then
//! concentrated on a handful of words. The scan streams every word of the data and keeps, per piece,
//! its total share and its `M` largest words; a piece whose top words carry much of its total, at a
//! size no ordinary piece reaches, is a candidate mechanism. Nothing names what to look for.
//!
//! `mpd_discover_2951 calibrate EXPORT_DIR LIBRARY_DIR CALIBRATION`
//!
//! Fits each site's library on the first `CALIBRATION` sequences and writes `{site}.v.f64`
//! (pieces × d_in), `{site}.u.f64` (pieces × d_out) and `{site}.w.f64` (each piece's `u_cᵀ F u_c`)
//! to `LIBRARY_DIR`, with `sites.txt` (`name pieces` per site, in order).
//!
//! `mpd_discover_2951 scan EXPORT_DIR LIBRARY_DIR OUT_PREFIX A:B [M] [CONTEXT]`
//!
//! Streams sequences `A..B` (`CONTEXT` positions each, default 512) and writes, over every piece in
//! `sites.txt` order, `OUT_PREFIX.sum.f64` (each piece's total share), `OUT_PREFIX.top.f64` (its `M`
//! largest shares, default 256, descending) and `OUT_PREFIX.at.u32` (their `[sequence, position]`).
//!
//! `mpd_discover_2951 merge LIBRARY_DIR OUT.json WORDS PARTS...`
//!
//! Merges scan parts (`OUT_PREFIX`es) covering `WORDS` words in all and reports every site's pieces
//! ranked by concentration: the share of a piece's total in its `M` largest words.
//!
//! `mpd_discover_2951 positions EXPORT_DIR OUT_PREFIX A:B|K/N [M] [BATCH]`
//!
//! The model's own forward on sequences `A..B` (or the `K`-th of `N` equal parts) of the export's token table (`T + 1` columns: the
//! model reads `row[..T]`, `row[1..]` are the next tokens), and one reverse pass of what the model
//! asserts, `Σ log q(top)` over its own top tokens (never the data's: a rule built on these values
//! selects words without looking at what follows them). Per word, `OUT_PREFIX.pos.f32` holds `[next
//! token, −log q(y), entropy of q, top token, log q(top)]` and `OUT_PREFIX.attr.f32` the first-order
//! share of `log q(top)` each head (`r_h · ∂/∂r_h`, its attended value) and each MLP (`a · ∂/∂a`, its
//! activations) carries, heads then MLP, layer by layer. Per MLP neuron, `OUT_PREFIX.neuron.f64`
//! holds `[Σ e, Σ |e|, Σ a⁺, Σ a⁺²]` of its share `e = a_n ∂/∂a_n` and its activation `a_n`, and
//! `OUT_PREFIX.neuron_push` / `.neuron_fired` (`.top.f64`, `.at.u32`) its `M` largest shares and
//! activations (default 256) with their `[sequence, position]`. With `LIBRARY_DIR` (`calibrate`'s
//! Fisher-SVD library), `OUT_PREFIX.piece` holds every piece's `M` largest KL shares likewise.

use gam_linalg::faer_ndarray::fast_abt;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{read_values, site_statistics, sites};
use gam_mpd::operator_program::FamilyInputs;
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2, Axis};
use serde_json::json;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_f64(path: &Path) -> Result<Vec<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

fn read_u32(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_f64(path: &Path, values: impl Iterator<Item = f64>) -> Result<(), String> {
    let bytes: Vec<u8> = values.flat_map(f64::to_le_bytes).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_u32(path: &Path, values: impl Iterator<Item = u32>) -> Result<(), String> {
    let bytes: Vec<u8> = values.flat_map(u32::to_le_bytes).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The sites listed in `sites.txt`, each with its piece count.
fn listed(library: &Path) -> Result<Vec<(String, usize)>, String> {
    let text = std::fs::read_to_string(library.join("sites.txt")).map_err(|e| format!("{}: {e}", library.display()))?;
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (name, pieces) = line.split_once(' ').ok_or("sites.txt: name pieces")?;
            Ok((name.to_string(), pieces.parse().map_err(|e| format!("sites.txt: {e}"))?))
        })
        .collect()
}

fn range(spec: &str) -> Result<(usize, usize), String> {
    let (a, b) = spec.split_once(':').ok_or_else(|| format!("{spec}: expected A:B"))?;
    let a: usize = a.parse().map_err(|e| format!("{spec}: {e}"))?;
    let b: usize = b.parse().map_err(|e| format!("{spec}: {e}"))?;
    if b <= a {
        return Err(format!("{spec}: an empty range"));
    }
    Ok((a, b))
}

fn calibrate(args: &[String]) -> Result<(), String> {
    let usage = "mpd_discover_2951 calibrate EXPORT_DIR LIBRARY_DIR CALIBRATION";
    let export = PathBuf::from(args.get(2).ok_or(usage)?);
    let library = PathBuf::from(args.get(3).ok_or(usage)?);
    let calibration: usize = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("CALIBRATION: {e}"))?;
    let context = 512;
    let imported = import_language_model(&export, calibration, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let chosen = sites(model);
    let sequence = |s: usize| family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>());
    let started = Instant::now();
    let statistics = site_statistics(model, &chosen, (0..calibration).map(sequence), 2, 0x5EED)?;
    eprintln!("statistics of {} sites on {calibration} sequences in {:.0}s", chosen.len(), started.elapsed().as_secs_f64());
    std::fs::create_dir_all(&library).map_err(|e| format!("{}: {e}", library.display()))?;
    let mut lines = Vec::new();
    for (site, stat) in chosen.iter().zip(&statistics) {
        let fitted = fisher_svd(stat)?;
        let v = fitted.v.t().to_owned();
        let u = fitted.u;
        let weights: Array1<f64> = (&u.dot(&stat.fisher) * &u).sum_axis(Axis(1));
        write_f64(&library.join(format!("{}.v.f64", site.name)), v.iter().copied())?;
        write_f64(&library.join(format!("{}.u.f64", site.name)), u.iter().copied())?;
        write_f64(&library.join(format!("{}.w.f64", site.name)), weights.iter().copied())?;
        lines.push(format!("{} {}", site.name, v.nrows()));
    }
    std::fs::write(library.join("sites.txt"), lines.join("\n") + "\n").map_err(|e| e.to_string())
}

fn scan(args: &[String]) -> Result<(), String> {
    let usage = "mpd_discover_2951 scan EXPORT_DIR LIBRARY_DIR OUT_PREFIX A:B [M] [CONTEXT]";
    let export = PathBuf::from(args.get(2).ok_or(usage)?);
    let library = PathBuf::from(args.get(3).ok_or(usage)?);
    let out = args.get(4).ok_or(usage)?.clone();
    let (first, last) = range(args.get(5).ok_or(usage)?)?;
    let keep: usize = args.get(6).map_or(Ok(256), |v| v.parse()).map_err(|e| format!("M: {e}"))?;
    let context: usize = args.get(7).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let imported = import_language_model(&export, last, context)?;
    let model = &imported.program;
    let family: &FamilyInputs = &imported.contract.family;
    let all = sites(model);
    let names = listed(&library)?;
    let mut chosen = Vec::new();
    let mut readers = Vec::new();
    for (name, pieces) in &names {
        let site = all.iter().find(|s| &s.name == name).ok_or_else(|| format!("{name}: not a site of the export"))?;
        let v = read_f64(&library.join(format!("{name}.v.f64")))?;
        let w = read_f64(&library.join(format!("{name}.w.f64")))?;
        if v.len() % pieces != 0 || w.len() != *pieces {
            return Err(format!("{name}: library files do not hold {pieces} pieces"));
        }
        let v = Array2::from_shape_vec((*pieces, v.len() / pieces), v).map_err(|e| e.to_string())?;
        chosen.push(site.clone());
        readers.push((v, Array1::from(w)));
    }
    let total: usize = names.iter().map(|(_, p)| p).sum();
    let mut sums = vec![0.0_f64; total];
    let mut kept = Kept::new(total, keep);
    let started = Instant::now();
    for s in first..last {
        let rows: Vec<usize> = (s * context..(s + 1) * context).collect();
        let trace = model.execute(&family.select(&rows), false).map_err(|e| e.to_string())?;
        let mut offset = 0;
        for (site, (v, w)) in chosen.iter().zip(&readers) {
            let x = read_values(&trace, site)?;
            let z = fast_abt(&x, v);
            for (c, (column, &weight)) in z.columns().into_iter().zip(w.iter()).enumerate() {
                let piece = offset + c;
                for (p, &a) in column.iter().enumerate() {
                    let share = a * a * weight;
                    sums[piece] += share;
                    kept.offer(piece, share, s as u32, p as u32);
                }
            }
            offset += v.nrows();
        }
        if (s + 1 - first) % 16 == 0 {
            eprintln!("scanned {} sequences in {:.0}s", s + 1 - first, started.elapsed().as_secs_f64());
        }
    }
    write_f64(Path::new(&format!("{out}.sum.f64")), sums.into_iter())?;
    kept.write(&out)
}

fn merge(args: &[String]) -> Result<(), String> {
    let usage = "mpd_discover_2951 merge LIBRARY_DIR OUT.json WORDS PARTS...";
    let library = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let words: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("WORDS: {e}"))?;
    let parts = &args[5..];
    if parts.is_empty() {
        return Err(usage.to_string());
    }
    let names = listed(&library)?;
    let total: usize = names.iter().map(|(_, p)| p).sum();
    let mut sums = vec![0.0_f64; total];
    let mut entries: Vec<Vec<(f64, u32, u32)>> = vec![Vec::new(); total];
    let mut keep = 0;
    for part in parts {
        let sum = read_f64(Path::new(&format!("{part}.sum.f64")))?;
        let top = read_f64(Path::new(&format!("{part}.top.f64")))?;
        let at = read_u32(Path::new(&format!("{part}.at.u32")))?;
        if sum.len() != total || top.len() % total != 0 || at.len() != 2 * top.len() {
            return Err(format!("{part}: not a scan of this library"));
        }
        keep = top.len() / total;
        for c in 0..total {
            sums[c] += sum[c];
            for j in 0..keep {
                let i = c * keep + j;
                if at[2 * i] != u32::MAX {
                    entries[c].push((top[i], at[2 * i], at[2 * i + 1]));
                }
            }
        }
    }
    let mut pieces = Vec::with_capacity(total);
    let mut offset = 0;
    for (name, count) in &names {
        let weights = read_f64(&library.join(format!("{name}.w.f64")))?;
        for c in 0..*count {
            let list = &mut entries[offset + c];
            list.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            list.truncate(keep);
            let head: f64 = list.iter().map(|e| e.0).sum();
            let concentration = if sums[offset + c] > 0.0 { head / sums[offset + c] } else { 0.0 };
            pieces.push((name.clone(), c, weights[c], sums[offset + c] / words, concentration, list.clone()));
        }
        offset += count;
    }
    // Ranked by the share their top words carry, times how large those shares are.
    pieces.sort_unstable_by(|a, b| {
        let score = |p: &(String, usize, f64, f64, f64, Vec<(f64, u32, u32)>)| p.4 * p.5.first().map_or(0.0, |e| e.0);
        score(b).total_cmp(&score(a))
    });
    let report: Vec<_> = pieces
        .iter()
        .take(200)
        .map(|(name, c, weight, mean, concentration, list)| {
            json!({
                "site": name, "piece": c, "fisher_weight": weight, "mean_share": mean, "concentration": concentration,
                "top": list.iter().take(64).map(|(e, s, p)| json!([s, p, e])).collect::<Vec<_>>(),
            })
        })
        .collect();
    let mut by_site = Vec::new();
    for (name, _) in &names {
        let mut values: Vec<f64> = pieces.iter().filter(|p| &p.0 == name).map(|p| p.4).collect();
        values.sort_by(f64::total_cmp);
        let at = |q: f64| values.get(((values.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
        by_site.push(json!({"site": name, "concentration": {"median": at(0.5), "p99": at(0.99), "max": at(1.0)}}));
    }
    let record = json!({"words": words, "kept": keep, "sites": by_site, "pieces": report});
    std::fs::write(&out, serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// The attended value of every head (the reads of each layer's `o` site) and the activations of
/// every MLP (the read of each layer's `down_proj` site), layer by layer.
fn readers(model: &gam_mpd::operator_program::OperatorProgram) -> Result<(Vec<Vec<usize>>, Vec<usize>), String> {
    let all = sites(model);
    let mut heads = Vec::new();
    let mut mlps = Vec::new();
    for layer in 0.. {
        let (Some(o), Some(down)) = (
            all.iter().find(|s| s.name == format!("blocks.{layer}.o")),
            all.iter().find(|s| s.name == format!("blocks.{layer}.down_proj")),
        ) else {
            break;
        };
        heads.push(o.reads.clone());
        mlps.push(*down.reads.first().ok_or("a down_proj site with no read")?);
    }
    if heads.is_empty() {
        return Err("no blocks.{l}.o / blocks.{l}.down_proj sites".to_string());
    }
    Ok((heads, mlps))
}

fn positions(args: &[String]) -> Result<(), String> {
    let usage = "mpd_discover_2951 positions EXPORT_DIR OUT_PREFIX A:B|K/N [M] [BATCH] [LIBRARY_DIR]";
    let export = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = args.get(3).ok_or(usage)?.clone();
    let spec = args.get(4).ok_or(usage)?;
    let keep: usize = args.get(5).map_or(Ok(256), |v| v.parse()).map_err(|e| format!("M: {e}"))?;
    let batch: usize = args.get(6).map_or(Ok(2), |v| v.parse()).map_err(|e| format!("BATCH: {e}"))?;
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let shape = &record["files"]["tokens"]["shape"];
    let (table_rows, columns) = (shape[0].as_u64().ok_or("tokens shape")? as usize, shape[1].as_u64().ok_or("tokens shape")? as usize);
    let context = columns - 1;
    // `K/N`: the K-th of N equal parts of the table.
    let (first, last) = match spec.split_once('/') {
        Some((k, n)) => {
            let (k, n): (usize, usize) = (k.parse().map_err(|e| format!("{spec}: {e}"))?, n.parse().map_err(|e| format!("{spec}: {e}"))?);
            (k * table_rows / n, (k + 1) * table_rows / n)
        }
        None => range(spec)?,
    };
    if last > table_rows {
        return Err(format!("{last} sequences of a {table_rows}-row token table"));
    }
    let table = read_f64(&export.join("tokens.f64"))?;
    let imported = import_language_model(&export, last, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let (heads, mlps) = readers(model)?;
    let width = heads.iter().map(Vec::len).sum::<usize>() + mlps.len();
    let neurons: Vec<usize> = {
        let interfaces = model.interfaces().map_err(|e| e.to_string())?;
        mlps.iter().map(|&n| interfaces[n].width()).collect()
    };
    let total: usize = neurons.iter().sum();
    let mut sums = vec![[0.0_f64; 4]; total];
    // Per neuron, its largest shares and its largest activations.
    let mut push = Kept::new(total, keep);
    let mut fired = Kept::new(total, keep);
    // With a library (`calibrate`), every piece's largest second-order KL shares `z_c² u_cᵀFu_c`.
    let mut library = Vec::new();
    if let Some(dir) = args.get(7) {
        let dir = PathBuf::from(dir);
        let all = sites(model);
        for (name, pieces) in listed(&dir)? {
            let site = all.iter().find(|s| s.name == name).ok_or_else(|| format!("{name}: not a site of the export"))?;
            let v = read_f64(&dir.join(format!("{name}.v.f64")))?;
            let w = read_f64(&dir.join(format!("{name}.w.f64")))?;
            if v.len() % pieces != 0 || w.len() != pieces {
                return Err(format!("{name}: library files do not hold {pieces} pieces"));
            }
            let v = Array2::from_shape_vec((pieces, v.len() / pieces), v).map_err(|e| e.to_string())?;
            library.push((site.clone(), v, Array1::from(w)));
        }
    }
    let mut kept_pieces = Kept::new(library.iter().map(|(_, v, _)| v.nrows()).sum(), keep);
    let mut words = Vec::with_capacity((last - first) * context * 5);
    let mut attributions = Vec::with_capacity((last - first) * context * width);
    let started = Instant::now();
    for start in (first..last).step_by(batch) {
        let sequences: Vec<usize> = (start..(start + batch).min(last)).collect();
        let rows: Vec<usize> = sequences.iter().flat_map(|&s| s * context..(s + 1) * context).collect();
        let inputs = family.select(&rows);
        let (trace, back) = gam_mpd::device::proposing(|| -> Result<_, String> {
            let trace = model.execute(&inputs, false).map_err(|e| e.to_string())?;
            let logits = &trace.values[model.output];
            let mut cotangent = Array2::<f64>::zeros(logits.dim());
            for (r, (row, mut out)) in logits.outer_iter().zip(cotangent.outer_iter_mut()).enumerate() {
                let (s, p) = (sequences[r / context], r % context);
                let label = table[s * columns + p + 1] as usize;
                let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let partition: f64 = row.iter().map(|l| (l - peak).exp()).sum();
                let log_partition = peak + partition.ln();
                let (mut entropy, mut top) = (0.0, 0);
                for (c, (&l, o)) in row.iter().zip(out.iter_mut()).enumerate() {
                    let q = (l - log_partition).exp();
                    *o = -q;
                    entropy -= q * (l - log_partition);
                    if l > row[top] {
                        top = c;
                    }
                }
                out[top] += 1.0;
                words.extend([label as f32, (log_partition - row[label]) as f32, entropy as f32, top as f32, (row[top] - log_partition) as f32]);
            }
            let back = gam_mpd::derivatives::vjp(model, &inputs, &trace, cotangent).map_err(|e| e.to_string())?;
            Ok((trace, back))
        })?;
        let share = |node: usize| -> Array2<f64> {
            match &back[node] {
                Some(g) => g * &trace.values[node],
                None => Array2::zeros(trace.values[node].dim()),
            }
        };
        let mut columns_of_batch: Vec<Array1<f64>> = Vec::with_capacity(width);
        for layer_heads in &heads {
            for &h in layer_heads {
                columns_of_batch.push(share(h).sum_axis(Axis(1)));
            }
        }
        let mut offset = 0;
        for &m in &mlps {
            let e = share(m);
            columns_of_batch.push(e.sum_axis(Axis(1)));
            let activations = &trace.values[m];
            for (r, (row, active)) in e.outer_iter().zip(activations.outer_iter()).enumerate() {
                let (s, p) = (sequences[r / context] as u32, (r % context) as u32);
                for (n, (&v, &a)) in row.iter().zip(active.iter()).enumerate() {
                    let k = offset + n;
                    let a = a.max(0.0);
                    sums[k][0] += v;
                    sums[k][1] += v.abs();
                    sums[k][2] += a;
                    sums[k][3] += a * a;
                    push.offer(k, v, s, p);
                    fired.offer(k, a, s, p);
                }
            }
            offset += e.ncols();
        }
        for r in 0..rows.len() {
            attributions.extend(columns_of_batch.iter().map(|c| c[r] as f32));
        }
        let mut offset = 0;
        for (site, v, w) in &library {
            let z = fast_abt(&read_values(&trace, site)?, v);
            for (r, row) in z.outer_iter().enumerate() {
                let (s, p) = (sequences[r / context] as u32, (r % context) as u32);
                for (c, (&a, &weight)) in row.iter().zip(w.iter()).enumerate() {
                    kept_pieces.offer(offset + c, a * a * weight, s, p);
                }
            }
            offset += v.nrows();
        }
        let done = sequences.last().map_or(0, |s| s + 1) - first;
        if done % 32 < batch {
            eprintln!("{done} sequences in {:.0}s", started.elapsed().as_secs_f64());
        }
    }
    let write_f32 = |path: String, values: &[f32]| -> Result<(), String> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).map_err(|e| format!("{path}: {e}"))
    };
    write_f32(format!("{out}.pos.f32"), &words)?;
    write_f32(format!("{out}.attr.f32"), &attributions)?;
    write_f64(Path::new(&format!("{out}.neuron.f64")), sums.iter().flat_map(|s| s.iter().copied()))?;
    push.write(&format!("{out}.neuron_push"))?;
    fired.write(&format!("{out}.neuron_fired"))?;
    if library.is_empty() { Ok(()) } else { kept_pieces.write(&format!("{out}.piece")) }
}

/// Per item, its `keep` largest positive values with their `[sequence, position]`.
struct Kept {
    keep: usize,
    heaps: Vec<BinaryHeap<Reverse<(u64, u32, u32)>>>,
    // A heap's least kept value, so most offers are refused by one comparison.
    floors: Vec<f64>,
}

impl Kept {
    fn new(items: usize, keep: usize) -> Self {
        Self { keep, heaps: (0..items).map(|_| BinaryHeap::with_capacity(keep + 1)).collect(), floors: vec![0.0; items] }
    }

    fn offer(&mut self, item: usize, value: f64, sequence: u32, position: u32) {
        if value <= self.floors[item] {
            return;
        }
        let heap = &mut self.heaps[item];
        // Positive values order as their bits.
        heap.push(Reverse((value.to_bits(), sequence, position)));
        if heap.len() > self.keep {
            heap.pop();
        }
        if heap.len() == self.keep {
            self.floors[item] = heap.peek().map_or(0.0, |Reverse(least)| f64::from_bits(least.0));
        }
    }

    /// `PREFIX.top.f64` (items × keep, descending, zero-padded) and `PREFIX.at.u32` (their
    /// `[sequence, position]`, `u32::MAX` padded).
    fn write(self, prefix: &str) -> Result<(), String> {
        let mut top = Vec::with_capacity(self.heaps.len() * self.keep);
        let mut at = Vec::with_capacity(self.heaps.len() * self.keep * 2);
        for heap in self.heaps {
            let mut entries: Vec<(u64, u32, u32)> = heap.into_iter().map(|Reverse(k)| k).collect();
            entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            entries.resize(self.keep, (0, u32::MAX, u32::MAX));
            for (bits, s, p) in entries {
                top.push(f64::from_bits(bits));
                at.extend([s, p]);
            }
        }
        write_f64(Path::new(&format!("{prefix}.top.f64")), top.into_iter())?;
        write_u32(Path::new(&format!("{prefix}.at.u32")), at.into_iter())
    }
}

/// A scanned word: where it is, its next token, the model's top token and `q(top)`.
#[derive(Clone, Copy)]
struct Word {
    sequence: u32,
    position: u32,
    label: u32,
    top: u32,
    confidence: f64,
}

/// The parts of a `positions` scan, in order, as one table of words.
struct Scanned {
    words: Vec<Word>,
    /// Words × `width` code-length shares (heads, then MLPs, layer by layer).
    attributions: Vec<f32>,
    width: usize,
    /// The first word of each part and the part's first sequence, to find a word by place.
    starts: Vec<(usize, u32, u32)>,
    context: usize,
    /// Per kept family (`neuron_push`, `neuron_fired`, `piece`): each item's kept words, descending.
    lists: Vec<(&'static str, Vec<Vec<(f64, usize)>>)>,
}

impl Scanned {
    fn index(&self, sequence: u32, position: u32) -> Option<usize> {
        self.starts.iter().rev().find(|(_, first, last)| (*first..*last).contains(&sequence)).map(|(start, first, _)| {
            start + (sequence - first) as usize * self.context + position as usize
        })
    }

    /// `PARTS`: `PREFIX@A`, a scan of sequences from `A` on.
    fn load(parts: &[String], context: usize) -> Result<Self, String> {
        let mut words = Vec::new();
        let mut attributions = Vec::new();
        let mut starts = Vec::new();
        let mut width = 0;
        for part in parts {
            let (prefix, first) = part.split_once('@').ok_or_else(|| format!("{part}: expected PREFIX@A"))?;
            let first: u32 = first.parse().map_err(|e| format!("{part}: {e}"))?;
            let read32 = |path: String| -> Result<Vec<f32>, String> {
                let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
                Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            };
            let table = read32(format!("{prefix}.pos.f32"))?;
            let shares = read32(format!("{prefix}.attr.f32"))?;
            let count = table.len() / 5;
            if count == 0 || count % context != 0 || shares.len() % count != 0 {
                return Err(format!("{prefix}: not a scan of whole {context}-word sequences"));
            }
            width = shares.len() / count;
            let sequences = (count / context) as u32;
            starts.push((words.len(), first, first + sequences));
            for (w, row) in table.chunks_exact(5).enumerate() {
                words.push(Word {
                    sequence: first + (w / context) as u32,
                    position: (w % context) as u32,
                    label: row[0] as u32,
                    top: row[3] as u32,
                    confidence: f64::from(row[4]).exp(),
                });
            }
            attributions.extend(shares);
        }
        let mut scanned = Self { words, attributions, width, starts, context, lists: Vec::new() };
        // The neurons' families, and a library's pieces when the scan had one; `M` per item is the
        // neurons' (their count is `neuron.f64`'s rows).
        for family in ["neuron_push", "neuron_fired", "piece"] {
            let mut lists: Vec<Vec<(f64, usize)>> = Vec::new();
            for part in parts {
                let (prefix, _) = part.split_once('@').ok_or("PREFIX@A")?;
                let path = format!("{prefix}.{family}.top.f64");
                if !Path::new(&path).exists() {
                    continue;
                }
                let top = read_f64(Path::new(&path))?;
                let at = read_u32(Path::new(&format!("{prefix}.{family}.at.u32")))?;
                let neurons = read_f64(Path::new(&format!("{prefix}.neuron.f64")))?.len() / 4;
                let keep = read_f64(Path::new(&format!("{prefix}.neuron_fired.top.f64")))?.len() / neurons.max(1);
                if keep == 0 || top.len() % keep != 0 || at.len() != 2 * top.len() {
                    return Err(format!("{path}: not {keep} words per item"));
                }
                lists.resize(top.len() / keep, Vec::new());
                for (item, list) in lists.iter_mut().enumerate() {
                    for j in 0..keep {
                        let i = item * keep + j;
                        if at[2 * i] != u32::MAX
                            && let Some(w) = scanned.index(at[2 * i], at[2 * i + 1])
                        {
                            list.push((top[i], w));
                        }
                    }
                }
            }
            for list in &mut lists {
                list.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            }
            if !lists.is_empty() {
                scanned.lists.push((family, lists));
            }
        }
        Ok(scanned)
    }
}

/// The evidence (bits) that the data refutes what the model asserts on a group of words: the code
/// length the data saves when the model's top token, expected `E = Σ q(top)` times and seen `O`
/// times on the group's `n` words, is given the rate the data shows instead of any rate of at least
/// half what the model asserts, `n·KL(O/n ‖ E/2n)` when `O < E/2`. A model that is miscalibrated in
/// the ordinary way (its top token seen nine tenths as often as asserted, say) gains nothing; a
/// mechanism that switches the prediction to what the text does not do gains about
/// `−log₂(1 − q/2)` bits a word.
fn refuted(n: f64, expected: f64, observed: f64) -> f64 {
    let half = expected / 2.0;
    if half <= observed || n == 0.0 {
        return 0.0;
    }
    let (a, b) = (observed / n, half / n);
    let term = |x: f64, y: f64| if x > 0.0 { x * (x / y).ln() } else { 0.0 };
    n * (term(a, b) + term(1.0 - a, 1.0 - b)) / std::f64::consts::LN_2
}

/// A group of words: those a rule selects whose top token is `token`.
#[derive(Clone)]
struct Candidate {
    family: String,
    member: usize,
    token: u32,
    count: usize,
    threshold: f64,
    gain: f64,
    cost: f64,
}

/// One rule family's sweep: `order` (value, word) descending; the rule "value ≥ θ and the top token
/// is `t`" for every `t` and every `θ` among the values. Per token, the `θ` of the largest saving
/// net of the rule's description (`members` choices of the rule, the token, `θ` as a rank).
fn sweep(family: &str, member: usize, members: usize, order: &[(f64, usize)], words: &[Word], vocab: usize, out: &mut Vec<Candidate>) {
    // Per token: count, expected, observed, best net, its count and threshold.
    let mut stats: Vec<(usize, f64, f64, f64, usize, f64)> = vec![(0, 0.0, 0.0, 0.0, 0, 0.0); vocab];
    let mut touched = Vec::new();
    let fixed = (members as f64).log2() + (vocab as f64).log2();
    for &(value, w) in order {
        let word = &words[w];
        let t = word.top as usize;
        let entry = &mut stats[t];
        if entry.0 == 0 {
            touched.push(t);
        }
        entry.0 += 1;
        entry.1 += word.confidence;
        entry.2 += f64::from(u8::from(word.label == word.top));
        let gain = refuted(entry.0 as f64, entry.1, entry.2);
        let net = gain - fixed - 2.0 * ((entry.0 + 1) as f64).log2();
        if net > entry.3 {
            entry.3 = net;
            entry.4 = entry.0;
            entry.5 = value;
        }
    }
    for t in touched {
        let (_, _, _, net, count, threshold) = stats[t];
        if net > 0.0 {
            let cost = fixed + 2.0 * ((count + 1) as f64).log2();
            out.push(Candidate { family: family.to_string(), member, token: t as u32, count, threshold, gain: net + cost, cost });
        }
    }
}

impl Candidate {
    /// The words the rule selects, in its order.
    fn words(&self, scanned: &Scanned, tokens: &[u32], columns: usize) -> Vec<usize> {
        let picked = |w: &usize| scanned.words[*w].top == self.token;
        match self.family.as_str() {
            "confidence" => {
                let mut all: Vec<usize> = (0..scanned.words.len()).filter(picked).collect();
                all.sort_unstable_by(|a, b| scanned.words[*b].confidence.total_cmp(&scanned.words[*a].confidence));
                all.truncate(self.count);
                all
            }
            "share" => {
                let mut all: Vec<usize> = (0..scanned.words.len()).filter(picked).filter(|w| scanned.attributions[w * scanned.width + self.member] > 0.0).collect();
                let value = |w: usize| scanned.attributions[w * scanned.width + self.member];
                all.sort_unstable_by(|a, b| value(*b).total_cmp(&value(*a)));
                all.truncate(self.count);
                all
            }
            family if family.starts_with("context") => {
                let n: usize = family["context".len()..].parse().unwrap_or(1);
                let key = context_key(tokens, columns, &scanned.words[self.member], n);
                (0..scanned.words.len()).filter(picked).filter(|w| context_key(tokens, columns, &scanned.words[*w], n) == key).collect()
            }
            family => {
                let lists = scanned.lists.iter().find(|(name, _)| *name == family).map(|(_, l)| l);
                lists.map_or_else(Vec::new, |lists| lists[self.member].iter().map(|(_, w)| *w).filter(picked).take(self.count).collect())
            }
        }
    }
}

/// The last `n` tokens a word reads (its own and the `n − 1` before it), or `None` near the start.
fn context_key(tokens: &[u32], columns: usize, word: &Word, n: usize) -> Option<Vec<u32>> {
    let p = word.position as usize;
    (p + 1 >= n).then(|| tokens[word.sequence as usize * columns + p + 1 - n..=word.sequence as usize * columns + p].to_vec())
}

/// The context a group's words share and every word that context makes the model assert the
/// group's token at. The trigger is the longest last-`n`-token context a majority of the group's
/// words end with; the firing words are those ending with the last `m ≤ n` of its tokens whose
/// top token is `token`, `m` the context rule of the most evidence net of its description (a
/// shorter context reaches the occurrences the longer one's extra tokens miss, and costs the
/// words where the shorter context does not switch the model). The group's own words when no
/// context is shared by a majority.
fn assemble(scanned: &Scanned, tokens: &[u32], columns: usize, vocab: usize, picked: &std::collections::BTreeSet<usize>, token: u32) -> (Vec<u32>, Vec<usize>) {
    let words = &scanned.words;
    let mut trigger = Vec::new();
    for n in 1..=32 {
        let mut counts: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
        for w in picked {
            if let Some(key) = context_key(tokens, columns, &words[*w], n) {
                *counts.entry(key).or_default() += 1;
            }
        }
        match counts.into_iter().max_by_key(|(_, c)| *c) {
            Some((key, count)) if 2 * count > picked.len() => trigger = key,
            _ => break,
        }
    }
    if trigger.is_empty() {
        return (trigger, picked.iter().copied().collect());
    }
    let asserting: Vec<usize> = (0..words.len()).filter(|&w| words[w].top == token).collect();
    let mut best: (f64, Vec<usize>) = (f64::NEG_INFINITY, Vec::new());
    for m in 1..=trigger.len() {
        let suffix = &trigger[trigger.len() - m..];
        let firing: Vec<usize> =
            asserting.iter().copied().filter(|&w| context_key(tokens, columns, &words[w], m).as_deref() == Some(suffix)).collect();
        let (expected, observed) =
            firing.iter().fold((0.0, 0.0), |(e, o), &w| (e + words[w].confidence, o + f64::from(u8::from(words[w].label == token))));
        let net = refuted(firing.len() as f64, expected, observed) - (m + 1) as f64 * (vocab as f64).log2();
        if net > best.0 {
            best = (net, firing);
        }
    }
    (trigger, best.1)
}

/// The model's greedy continuation of each context `(sequence, position)` (the table row's tokens
/// through `position`), `steps` tokens, each with its probability; two contexts at a time.
fn greedy(export: &Path, contexts: &[(usize, usize)], steps: usize) -> Result<Vec<Vec<(u32, f64)>>, String> {
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let columns = record["files"]["tokens"]["shape"][1].as_u64().ok_or("tokens shape")? as usize;
    let table: Vec<u32> = read_f64(&export.join("tokens.f64"))?.into_iter().map(|t| t as u32).collect();
    let window = columns - 1;
    let imported = import_language_model(export, 1, window)?;
    let model = &imported.program;
    let mut out = Vec::with_capacity(contexts.len());
    for pair in contexts.chunks(2) {
        let mut texts: Vec<Vec<u32>> = pair.iter().map(|&(s, p)| table[s * columns..=s * columns + p].to_vec()).collect();
        let mut runs = vec![Vec::with_capacity(steps); pair.len()];
        for _ in 0..steps {
            let (mut ids, mut sequence, mut position, mut ends) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, text) in texts.iter().enumerate() {
                let start = text.len().saturating_sub(window);
                for (j, &t) in text[start..].iter().enumerate() {
                    ids.push(t);
                    sequence.push(i as u32);
                    position.push(j as u32);
                }
                ends.push(ids.len() - 1);
            }
            let inputs = FamilyInputs {
                rows: ids.len(),
                slots: vec![gam_mpd::operator_program::SlotValues::Tokens(ids)],
                layout: Some(gam_mpd::operator_program::SequenceLayout { sequence, position }),
            };
            let trace = gam_mpd::device::proposing(|| model.execute(&inputs, false)).map_err(|e| e.to_string())?;
            let logits = &trace.values[model.output];
            for (i, &end) in ends.iter().enumerate() {
                let row = logits.row(end);
                let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let partition: f64 = row.iter().map(|l| (l - peak).exp()).sum();
                let top = row.iter().enumerate().fold(0, |best, (c, &l)| if l > row[best] { c } else { best });
                runs[i].push((top as u32, (row[top] - peak).exp() / partition));
                texts[i].push(top as u32);
            }
        }
        out.extend(runs);
    }
    Ok(out)
}

fn detect(args: &[String]) -> Result<(), String> {
    let usage = "mpd_discover_2951 detect EXPORT_DIR OUT.json PREDICTION.json PREFIX@A...";
    let export = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let prediction = PathBuf::from(args.get(4).ok_or(usage)?);
    let parts = &args[5..];
    if parts.is_empty() {
        return Err(usage.to_string());
    }
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let shape = &record["files"]["tokens"]["shape"];
    let columns = shape[1].as_u64().ok_or("tokens shape")? as usize;
    let vocab = record["config"]["vocab"].as_u64().ok_or("config.vocab")? as usize;
    let tokens: Vec<u32> = read_f64(&export.join("tokens.f64"))?.into_iter().map(|t| t as u32).collect();
    let scanned = Scanned::load(parts, columns - 1)?;
    let words = &scanned.words;
    let started = Instant::now();
    let mut candidates = Vec::new();
    // Every family's rule: a threshold on one value and the top token.
    let mut order: Vec<(f64, usize)> = words.iter().enumerate().map(|(w, word)| (word.confidence, w)).collect();
    order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    sweep("confidence", 0, 1, &order, words, vocab, &mut candidates);
    for c in 0..scanned.width {
        let mut order: Vec<(f64, usize)> = (0..words.len())
            .map(|w| (f64::from(scanned.attributions[w * scanned.width + c]), w))
            .filter(|(v, _)| *v > 0.0)
            .collect();
        order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        sweep("share", c, scanned.width, &order, words, vocab, &mut candidates);
    }
    for (family, lists) in &scanned.lists {
        for (n, list) in lists.iter().enumerate() {
            sweep(family, n, lists.len(), list, words, vocab, &mut candidates);
        }
    }
    // Context rules: the last n tokens and the top token, described as n + 1 tokens.
    for n in 1..=3 {
        let mut groups: std::collections::HashMap<(Vec<u32>, u32), (usize, f64, f64, usize)> = std::collections::HashMap::new();
        for (w, word) in words.iter().enumerate() {
            if let Some(key) = context_key(&tokens, columns, word, n) {
                let entry = groups.entry((key, word.top)).or_insert((0, 0.0, 0.0, w));
                entry.0 += 1;
                entry.1 += word.confidence;
                entry.2 += f64::from(u8::from(word.label == word.top));
            }
        }
        let cost = (n + 1) as f64 * (vocab as f64).log2();
        for ((_, top), (count, expected, observed, first)) in groups {
            let gain = refuted(count as f64, expected, observed);
            if gain > cost {
                candidates.push(Candidate { family: format!("context{n}"), member: first, token: top, count, threshold: 0.0, gain, cost });
            }
        }
    }
    eprintln!("{} rules save bits, in {:.0}s", candidates.len(), started.elapsed().as_secs_f64());
    candidates.sort_unstable_by(|a, b| (b.gain - b.cost).total_cmp(&(a.gain - a.cost)));
    // The best rule of each group of words; a later rule whose words are mostly an earlier one's is
    // the same group, listed under it.
    let mut groups: Vec<(Candidate, std::collections::BTreeSet<usize>, Vec<serde_json::Value>)> = Vec::new();
    for candidate in candidates.iter().take(512) {
        let picked: std::collections::BTreeSet<usize> = candidate.words(&scanned, &tokens, columns).into_iter().collect();
        let net = candidate.gain - candidate.cost;
        let same = groups.iter().position(|(_, words, _)| {
            let shared = picked.intersection(words).count();
            2 * shared >= picked.len().min(words.len())
        });
        let note = json!({"family": candidate.family, "member": candidate.member, "count": candidate.count, "net_bits": net});
        match same {
            Some(g) => groups[g].2.push(note),
            None if groups.len() < 64 => groups.push((candidate.clone(), picked, Vec::new())),
            None => {}
        }
    }
    let report: Vec<_> = groups
        .iter()
        .map(|(c, picked, also)| {
            let rows: std::collections::BTreeSet<u32> = picked.iter().map(|w| words[*w].sequence).collect();
            let (expected, observed) = picked.iter().fold((0.0, 0.0), |(e, o), w| {
                (e + words[*w].confidence, o + f64::from(u8::from(words[*w].label == words[*w].top)))
            });
            let mut suffixes = Vec::new();
            for n in 1..=4 {
                let mut counts: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
                for w in picked {
                    if let Some(key) = context_key(&tokens, columns, &words[*w], n) {
                        *counts.entry(key).or_default() += 1;
                    }
                }
                let mut counts: Vec<_> = counts.into_iter().collect();
                counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                suffixes.push(counts.into_iter().take(3).map(|(k, n)| json!([k, n])).collect::<Vec<_>>());
            }
            json!({
                "family": c.family, "member": c.member, "token": c.token, "threshold": c.threshold,
                "count": picked.len(), "rows": rows.len(), "expected": expected, "observed": observed,
                "gain_bits": c.gain, "cost_bits": c.cost, "net_bits": c.gain - c.cost,
                "bits_per_word": c.gain / picked.len().max(1) as f64,
                "contexts": suffixes,
                "words": picked.iter().take(400).map(|w| json!([words[*w].sequence, words[*w].position])).collect::<Vec<_>>(),
                "also": also.iter().take(32).collect::<Vec<_>>(),
            })
        })
        .collect();
    // The strongest group is the mechanism; a later group most of whose words sit `k` words after
    // the mechanism's firing words is what it goes on to write there, and fires with it.
    let mut mechanism = json!({"schema": "mpd.planted-prediction/1", "firing_positions": [], "trigger": [], "behavior": []});
    let mut explained = json!(null);
    if let Some((first, picked, _)) = groups.first() {
        let (trigger, mut firing) = assemble(&scanned, &tokens, columns, vocab, picked, first.token);
        let mut continuations = Vec::new();
        for (g, (later, others, _)) in groups.iter().enumerate().skip(1) {
            let set: std::collections::BTreeSet<usize> = firing.iter().copied().collect();
            let offset = (1..=16).find(|&k| {
                let after = others.iter().filter(|&&w| words[w].position as usize >= k && set.contains(&(w - k))).count();
                2 * after > others.len()
            });
            if let Some(k) = offset {
                continuations.push(json!({"group": g, "offset": k, "token": later.token}));
                firing.extend(others.iter().copied());
            }
        }
        firing.sort_unstable();
        firing.dedup();
        let contexts: Vec<(usize, usize)> =
            firing.iter().take(16).map(|&w| (words[w].sequence as usize, words[w].position as usize)).collect();
        let continued = greedy(&export, &contexts, 32)?;
        // Per step, the token most contexts continue with, while a majority agree.
        let mut behavior = Vec::new();
        for step in 0..32 {
            let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
            for run in &continued {
                *counts.entry(run[step].0).or_default() += 1;
            }
            match counts.into_iter().max_by_key(|(t, c)| (*c, Reverse(*t))) {
                Some((t, c)) if 2 * c > continued.len() => behavior.push(t),
                _ => break,
            }
        }
        mechanism = json!({
            "schema": "mpd.planted-prediction/1",
            "firing_positions": firing.iter().map(|&w| json!([words[w].sequence, words[w].position])).collect::<Vec<_>>(),
            "trigger": trigger, "behavior": behavior,
        });
        explained = json!({
            "continuations": continuations,
            "greedy": continued.iter().zip(&contexts).map(|(run, (s, p))| json!({"at": [s, p], "tokens": run})).collect::<Vec<_>>(),
        });
    }
    std::fs::write(&prediction, serde_json::to_string_pretty(&mechanism).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let record = json!({"words": words.len(), "rules_saving_bits": candidates.len(), "groups": report, "mechanism": explained});
    std::fs::write(&out, serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("calibrate") => calibrate(&args),
        Some("scan") => scan(&args),
        Some("merge") => merge(&args),
        Some("positions") => positions(&args),
        Some("detect") => detect(&args),
        _ => Err("mpd_discover_2951 {calibrate|scan|merge|positions|detect} ...".to_string()),
    }
}
