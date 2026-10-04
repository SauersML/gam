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
    // Per piece, its `keep` largest shares as a min-heap of (share bits, sequence, position).
    let mut heaps: Vec<BinaryHeap<Reverse<(u64, u32, u32)>>> = (0..total).map(|_| BinaryHeap::with_capacity(keep + 1)).collect();
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
                let heap = &mut heaps[piece];
                for (p, &a) in column.iter().enumerate() {
                    let share = a * a * weight;
                    sums[piece] += share;
                    // Nonnegative shares order as their bits.
                    let key = (share.to_bits(), s as u32, p as u32);
                    if heap.len() < keep {
                        heap.push(Reverse(key));
                    } else if heap.peek().is_some_and(|Reverse(least)| key.0 > least.0) {
                        heap.pop();
                        heap.push(Reverse(key));
                    }
                }
            }
            offset += v.nrows();
        }
        if (s + 1 - first) % 16 == 0 {
            eprintln!("scanned {} sequences in {:.0}s", s + 1 - first, started.elapsed().as_secs_f64());
        }
    }
    let mut top = Vec::with_capacity(total * keep);
    let mut at = Vec::with_capacity(total * keep * 2);
    for heap in heaps {
        let mut entries: Vec<(u64, u32, u32)> = heap.into_iter().map(|Reverse(k)| k).collect();
        entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        entries.resize(keep, (0, u32::MAX, u32::MAX));
        for (bits, s, p) in entries {
            top.push(f64::from_bits(bits));
            at.push(s);
            at.push(p);
        }
    }
    write_f64(Path::new(&format!("{out}.sum.f64")), sums.into_iter())?;
    write_f64(Path::new(&format!("{out}.top.f64")), top.into_iter())?;
    write_u32(Path::new(&format!("{out}.at.u32")), at.into_iter())
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

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("calibrate") => calibrate(&args),
        Some("scan") => scan(&args),
        Some("merge") => merge(&args),
        _ => Err("mpd_discover_2951 {calibrate|scan|merge} ...".to_string()),
    }
}
