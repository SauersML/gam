//! The concept dictionary of a decomposition's per-word sets (#2951), for the natural-language
//! autoencoder `bench/vpd_2951/vpd_nl_autoencoder.py`.
//!
//! `mpd_nl_concepts_2951 SETS TRAIN EVAL LABEL_BITS OUT`
//!
//! `SETS` holds `indptr.i64` and `indices.i64` (raw little-endian, CSR over every sequence's
//! positions, sequence after sequence) and `meta.json` (`universe`, `context`). `TRAIN` and `EVAL`
//! are sequence ranges `lo:hi`. The concepts are fitted on `TRAIN` ([`gam_mpd::concepts::fit`],
//! each concept's name costing `LABEL_BITS`), then every sequence of both ranges is coded by the
//! frozen model. Writes to `OUT`:
//!
//! * `concepts.json`: the fit's totals and rounds, and every concept's members and rates;
//! * `invoked.indptr.i64`, `invoked.indices.i64`: per word of `TRAIN` then `EVAL`, the concepts
//!   it invokes;
//! * `bits.f64`: per word, its choice, member, lone and independent bits.

use gam_mpd::concepts::{Fit, Model, Sets, fit};
use serde_json::json;
use std::path::{Path, PathBuf};

fn read_i64(path: &Path) -> Result<Vec<i64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{}: {} bytes are not int64", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

fn range(spec: &str) -> Result<(usize, usize), String> {
    let (lo, hi) = spec.split_once(':').ok_or(format!("{spec}: not lo:hi"))?;
    Ok((lo.parse().map_err(|e| format!("{spec}: {e}"))?, hi.parse().map_err(|e| format!("{spec}: {e}"))?))
}

/// The words of sequences `lo..hi` as their own sets.
fn rows(indptr: &[i64], indices: &[i64], universe: usize, context: usize, (lo, hi): (usize, usize)) -> Result<Sets, String> {
    let (a, b) = (indptr[lo * context] as usize, indptr[hi * context] as usize);
    let ptr = indptr[lo * context..=hi * context].iter().map(|p| *p as usize - a).collect();
    Sets::new(universe, ptr, indices[a..b].iter().map(|j| *j as u32).collect())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: mpd_nl_concepts_2951 SETS TRAIN EVAL LABEL_BITS OUT";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let train = range(args.get(2).ok_or(usage)?)?;
    let eval = range(args.get(3).ok_or(usage)?)?;
    let label_bits: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("LABEL_BITS: {e}"))?;
    let out = PathBuf::from(args.get(5).ok_or(usage)?);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let universe = meta["universe"].as_u64().ok_or("meta.json: universe")? as usize;
    let context = meta["context"].as_u64().ok_or("meta.json: context")? as usize;
    let indptr = read_i64(&dir.join("indptr.i64"))?;
    let indices = read_i64(&dir.join("indices.i64"))?;
    let sequences = (indptr.len() - 1) / context;
    if train.1 > sequences || eval.1 > sequences || train.0 >= train.1 || eval.0 >= eval.1 {
        return Err(format!("ranges outside the {sequences} sequences"));
    }
    let fitted_on = rows(&indptr, &indices, universe, context, train)?;
    let clock = std::time::Instant::now();
    let fitted = fit(&fitted_on, label_bits);
    let seconds = clock.elapsed().as_secs_f64();
    let independent = Fit::independent_bits(&fitted_on);
    for (i, r) in fitted.rounds.iter().enumerate() {
        println!(
            "round {:3}: groups {:6} concepts {:5} open {:6} merges {:5} total {:.4e} bits",
            i + 1,
            r.groups,
            r.concepts,
            r.open,
            r.merges,
            r.total_bits
        );
    }
    let total = fitted.total_bits();
    let words = fitted_on.rows() as f64;
    println!(
        "fit {seconds:.1}s: {:.1} bits/word with concepts vs {:.1} independent ({} words)",
        total / words,
        independent / words,
        fitted_on.rows()
    );
    let model = Model::new(&fitted);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut ptr: Vec<i64> = vec![0];
    let mut invoked: Vec<i64> = Vec::new();
    let mut bits: Vec<f64> = Vec::new();
    let mut summary = Vec::new();
    for (name, spec) in [("train", train), ("eval", eval)] {
        let sets = rows(&indptr, &indices, universe, context, spec)?;
        let mut sum = [0.0; 4];
        for t in 0..sets.rows() {
            let (concepts, b) = model.encode(sets.row(t));
            invoked.extend(concepts.iter().map(|c| *c as i64));
            ptr.push(invoked.len() as i64);
            for (s, x) in sum.iter_mut().zip([b.choices, b.members, b.alone, b.independent]) {
                *s += x;
            }
            bits.extend([b.choices, b.members, b.alone, b.independent]);
        }
        let n = sets.rows() as f64;
        println!(
            "{name}: per word choices {:.1} + members {:.1} + alone {:.1} = {:.1} bits vs independent {:.1}",
            sum[0] / n,
            sum[1] / n,
            sum[2] / n,
            (sum[0] + sum[1] + sum[2]) / n,
            sum[3] / n
        );
        summary.push(json!({"rows": name, "sequences": [spec.0, spec.1], "words": sets.rows(),
            "choices": sum[0] / n, "members": sum[1] / n, "alone": sum[2] / n, "independent": sum[3] / n}));
    }
    let write = |name: &str, bytes: Vec<u8>| std::fs::write(out.join(name), bytes).map_err(|e| e.to_string());
    write("invoked.indptr.i64", ptr.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    write("invoked.indices.i64", invoked.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    write("bits.f64", bits.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    let report = json!({
        "universe": universe, "context": context, "label_bits": label_bits, "fit_seconds": seconds,
        "train_words": fitted_on.rows(), "total_bits": total, "independent_bits": independent,
        "rounds": fitted.rounds, "coded": summary,
        "concepts": model.concepts,
    });
    write("concepts.json", serde_json::to_vec(&report).map_err(|e| e.to_string())?)
}
