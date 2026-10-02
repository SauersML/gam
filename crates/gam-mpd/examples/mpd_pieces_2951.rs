//! Per-input pieces of every site of an exported model (#2951), in the per-token frontier's
//! protocol: `bench/mpd_pieces_2951.py dump DIR` writes the sites, this fits them, and
//! `bench/mpd_pieces_2951.py eval DIR` runs the masked program.
//!
//! `mpd_pieces_2951 DIR OBSERVATIONS [SITE_SUBSTRING]`
//!
//! Per site (`gam_mpd::pieces`): the library fitted on the fit inputs at `OBSERVATIONS` per input,
//! written as `{site}.V.f64` (`d_in × C`) and `{site}.U.f64` (`C × d_out`); then, at each level
//! `n = OBSERVATIONS · 2^j`, `j = −4 … 4`, every eval input's set by the same selection rule,
//! written as `{site}.mask{j}.u8` (eval inputs × C, one byte per piece). `pieces.json` lists the
//! pieces per site, the levels, and each fit's history.

use gam_mpd::pieces::{Site, fit, sets};
use ndarray::{Array1, Array2};
use serde_json::json;
use std::path::{Path, PathBuf};

fn read_f64(path: &Path, rows: usize, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|e| e.to_string())
}

fn read_f32(path: &Path, rows: usize, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != rows * cols * 4 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(4).map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|e| e.to_string())
}

fn write_f64(path: &Path, values: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_pieces_2951 DIR OBSERVATIONS [SITE_SUBSTRING]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let observations: f64 = args.get(2).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let only = args.get(3).cloned().unwrap_or_default();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let fit_tokens = manifest["fit_tokens"].as_u64().ok_or("fit_tokens")? as usize;
    let eval_tokens = manifest["eval_tokens"].as_u64().ok_or("eval_tokens")? as usize;
    let levels: Vec<f64> = (-4..=4).map(|j| observations * 2f64.powi(j)).collect();
    let mut pieces = serde_json::Map::new();
    let mut histories = serde_json::Map::new();
    for entry in manifest["sites"].as_array().ok_or("sites")? {
        let name = entry["name"].as_str().ok_or("name")?;
        if !name.contains(&only) {
            continue;
        }
        let (d_out, d_in) = (entry["d_out"].as_u64().ok_or("d_out")? as usize, entry["d_in"].as_u64().ok_or("d_in")? as usize);
        let started = std::time::Instant::now();
        let site = Site {
            w: read_f64(&dir.join(format!("{name}.W.f64")), d_out, d_in)?,
            second_moment: read_f64(&dir.join(format!("{name}.A.f64")), d_in, d_in)?,
            mean: Array1::from_vec(read_f64(&dir.join(format!("{name}.mu.f64")), 1, d_in)?.into_raw_vec_and_offset().0),
            fisher: read_f64(&dir.join(format!("{name}.B.f64")), d_out, d_out)?,
        };
        let fit_x = read_f32(&dir.join(format!("{name}.fit.f32")), fit_tokens, d_in)?;
        let (library, whitened) = fit(&site, &fit_x, observations)?;
        drop(fit_x);
        let exactness = library.exactness(&site.w);
        if exactness > 1e-6 {
            return Err(format!("{name}: the library is not the map ({exactness:e})"));
        }
        write_f64(&dir.join(format!("{name}.V.f64")), &library.v)?;
        write_f64(&dir.join(format!("{name}.U.f64")), &library.u)?;
        let count = library.u.nrows();
        let eval_x = read_f32(&dir.join(format!("{name}.eval.f32")), eval_tokens, d_in)?;
        let mut mean_active = Vec::new();
        for (j, level) in levels.iter().enumerate() {
            let chosen = sets(&library, &whitened, &eval_x, *level);
            let mut mask = vec![0u8; eval_tokens * count];
            for (t, set) in chosen.iter().enumerate() {
                for &c in set {
                    mask[t * count + c] = 1;
                }
            }
            mean_active.push(chosen.iter().map(Vec::len).sum::<usize>() as f64 / eval_tokens as f64);
            std::fs::write(dir.join(format!("{name}.mask{j}.u8")), mask).map_err(|e| e.to_string())?;
        }
        eprintln!(
            "{name}: {count} pieces ({} for exactness), fit history {:?}, eval active per level {:?}, {:.1}s",
            library.exactness_pieces,
            library.history.iter().map(|(l, k)| (l.round(), k.round())).collect::<Vec<_>>(),
            mean_active.iter().map(|a| (a * 10.0).round() / 10.0).collect::<Vec<_>>(),
            started.elapsed().as_secs_f64()
        );
        pieces.insert(name.to_string(), json!(count));
        histories.insert(name.to_string(), json!({"history": library.history, "eval_active": mean_active}));
    }
    let spec = json!({"observations": observations, "levels": levels, "pieces": pieces, "fits": histories});
    std::fs::write(dir.join("pieces.json"), serde_json::to_string_pretty(&spec).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
