//! The named vocabulary of a decomposition's per-word sets (#2951), for the natural-language
//! autoencoder `bench/vpd_2951/vpd_nl_autoencoder.py`, where the text is the only channel.
//!
//! `mpd_nl_concepts_2951 SETS TRAIN EVAL OBSERVATIONS LABEL_BITS OUT`
//!
//! `SETS` holds, raw little-endian: `indptr.i64` and `indices.i64` (CSR over every sequence's
//! positions, sequence after sequence), `missing.f32` (per set member, the exact KL in nats its
//! absence adds at that word), `program.f64` (per subcomponent, its description bits under the
//! library's own description), and `meta.json` (`universe`, `context`). Any library's sets work.
//! `TRAIN` and `EVAL` are sequence ranges `lo:hi`. Each member's price of being left off is
//! `OBSERVATIONS · KL / ln 2`. The vocabulary is fitted on `TRAIN` ([`gam_mpd::concepts::fit`],
//! every concept's name costing `LABEL_BITS` in the library), then every word of `TRAIN` and
//! `EVAL` is encoded by the frozen model. Writes to `OUT`:
//!
//! * `concepts.json`: the fit's totals and rounds, and the vocabulary's members, rates and program bits;
//! * `invoked.indptr.i64`, `invoked.indices.i64`: per word of `TRAIN` then `EVAL`, the concepts it names;
//! * `bits.f64`: per word, its names, program and predicted error bits, and its own set's program bits.

use gam_mpd::concepts::{Model, Priced, Sets, fit};
use serde_json::json;
use std::path::{Path, PathBuf};

fn read<const W: usize>(path: &Path) -> Result<Vec<[u8; W]>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % W != 0 {
        return Err(format!("{}: {} bytes are not {W}-byte values", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(W).map(|c| c.try_into().expect("chunk width")).collect())
}

fn range(spec: &str) -> Result<(usize, usize), String> {
    let (lo, hi) = spec.split_once(':').ok_or(format!("{spec}: not lo:hi"))?;
    Ok((lo.parse().map_err(|e| format!("{spec}: {e}"))?, hi.parse().map_err(|e| format!("{spec}: {e}"))?))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: mpd_nl_concepts_2951 SETS TRAIN EVAL OBSERVATIONS LABEL_BITS OUT";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let train = range(args.get(2).ok_or(usage)?)?;
    let eval = range(args.get(3).ok_or(usage)?)?;
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let label_bits: f64 = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("LABEL_BITS: {e}"))?;
    let out = PathBuf::from(args.get(6).ok_or(usage)?);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let universe = meta["universe"].as_u64().ok_or("meta.json: universe")? as usize;
    let context = meta["context"].as_u64().ok_or("meta.json: context")? as usize;
    let indptr: Vec<usize> = read::<8>(&dir.join("indptr.i64"))?.into_iter().map(|b| i64::from_le_bytes(b) as usize).collect();
    let indices: Vec<u32> = read::<8>(&dir.join("indices.i64"))?.into_iter().map(|b| i64::from_le_bytes(b) as u32).collect();
    let scale = observations / std::f64::consts::LN_2;
    let missing: Vec<f64> = read::<4>(&dir.join("missing.f32"))?.into_iter().map(|b| f64::from(f32::from_le_bytes(b)) * scale).collect();
    let program: Vec<f64> = read::<8>(&dir.join("program.f64"))?.into_iter().map(f64::from_le_bytes).collect();
    if missing.len() != indices.len() || program.len() != universe {
        return Err("missing.f32 needs one value per set member and program.f64 one per subcomponent".to_string());
    }
    let sequences = (indptr.len() - 1) / context;
    if train.1 > sequences || eval.1 > sequences || train.0 >= train.1 || eval.0 >= eval.1 {
        return Err(format!("ranges outside the {sequences} sequences"));
    }
    let rows = |(lo, hi): (usize, usize)| -> Result<Priced, String> {
        let (a, b) = (indptr[lo * context], indptr[hi * context]);
        let ptr = indptr[lo * context..=hi * context].iter().map(|p| p - a).collect();
        Priced::new(Sets::new(universe, ptr, indices[a..b].to_vec())?, missing[a..b].to_vec())
    };
    let fitted_on = rows(train)?;
    let clock = std::time::Instant::now();
    let fitted = fit(&fitted_on, &program, label_bits);
    let seconds = clock.elapsed().as_secs_f64();
    for (i, r) in fitted.rounds.iter().enumerate() {
        println!(
            "round {:3}: concepts {:6} vocabulary {:6} open {:6} merges {:5} peels {:5} total {:.6e} bits",
            i + 1,
            r.concepts,
            r.vocabulary,
            r.open,
            r.merges,
            r.peels,
            r.total_bits
        );
    }
    let words = fitted_on.sets.rows() as f64;
    let own: f64 = fitted_on.program_bits(&program).iter().sum();
    println!("fit {seconds:.1}s: {:.1} bits/word (names + program + predicted error + library) vs the own sets' program {:.1}", fitted.total_bits() / words, own / words);
    let model = Model::new(&fitted);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut ptr: Vec<i64> = vec![0];
    let mut invoked: Vec<i64> = Vec::new();
    let mut bits: Vec<f64> = Vec::new();
    let mut summary = Vec::new();
    for (name, spec) in [("train", train), ("eval", eval)] {
        let priced = rows(spec)?;
        let own = priced.program_bits(&program);
        let mut sum = [0.0; 4];
        for t in 0..priced.sets.rows() {
            let k = priced.sets.indptr[t]..priced.sets.indptr[t + 1];
            let (concepts, b) = model.encode(priced.sets.row(t), &priced.missing[k]);
            invoked.extend(concepts.iter().map(|c| i64::from(*c)));
            ptr.push(invoked.len() as i64);
            let row = [b.names, b.program, b.error, own[t]];
            for (s, x) in sum.iter_mut().zip(row) {
                *s += x;
            }
            bits.extend(row);
        }
        let n = priced.sets.rows() as f64;
        println!(
            "{name}: per word names {:.1} + program {:.1} + predicted error {:.1} bits; the own sets' program {:.1}",
            sum[0] / n,
            sum[1] / n,
            sum[2] / n,
            sum[3] / n
        );
        summary.push(json!({"rows": name, "sequences": [spec.0, spec.1], "words": priced.sets.rows(),
            "names": sum[0] / n, "program": sum[1] / n, "error": sum[2] / n, "own_program": sum[3] / n}));
    }
    let write = |name: &str, bytes: Vec<u8>| std::fs::write(out.join(name), bytes).map_err(|e| e.to_string());
    write("invoked.indptr.i64", ptr.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    write("invoked.indices.i64", invoked.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    write("bits.f64", bits.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    let report = json!({
        "universe": universe, "context": context, "observations": observations, "label_bits": label_bits,
        "fit_seconds": seconds, "train_words": fitted_on.sets.rows(), "total_bits": fitted.total_bits(),
        "rounds": fitted.rounds, "coded": summary, "concepts": model.concepts,
    });
    write("concepts.json", serde_json::to_vec(&report).map_err(|e| e.to_string())?)
}
