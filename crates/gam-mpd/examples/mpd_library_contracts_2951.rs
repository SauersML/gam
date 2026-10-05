//! Contracts of the induction circuit of a library explanation (`gam_mpd::library_readout`,
//! #2951), each prediction tested by patching on held-out text.
//!
//! EXPORT SETTINGS.json OUT.json [ARTIFACT]
//!
//! Each base sequence is a held-out passage of `half` tokens repeated once; a target is a
//! position `t` of the second copy, whose token `x_j` first occurred at `j = t − half`, so the
//! model's next token there repeats `x_{j+1}`. The contracts (written from the read-out before
//! these tests):
//!
//! * the previous-token head reads the previous position: substituting the token at `t − 1`
//!   changes its write at `t` more than substituting the token at `t − 2` or `t − 4`;
//! * the induction head selects the position after the earlier occurrence: substituting the token
//!   at `j` or `j + 1` changes its write at `t` more than substituting the token at `j − 2` or
//!   `j + 3`;
//! * its key reads the previous-token head: the previous-token head's write taken from another
//!   passage (path patching, the direct path only) into the induction head's key read lowers its
//!   attention to `j + 1` more than the same patch into its query read;
//! * the induction head writes the next token: setting its read to zero at `t` lowers
//!   `log p(x_{j+1})` at `t` more than setting it to zero at `t − 5`;
//! * the copy head reads the induction head through its value: the induction head's write taken
//!   from another passage into the copy head's value read changes the copy head's write at `t`
//!   more than the same patch into its query read;
//! * the copy head writes the next token: setting its read to zero at `t` lowers
//!   `log p(x_{j+1})` at `t` more than setting it to zero at `t − 5`.
//!
//! A substituted token is drawn uniformly from the held-out text's distinct tokens. A prediction
//! holds at a target when its inequality does; the report gives, per prediction, the fraction of
//! targets where it holds and the mean of each side.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl,
    library_readout::Library,
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    context: usize,
    /// The held-out rows `[start, end)` passages are taken from.
    held_out: [usize; 2],
    /// Tokens of each passage (the base sequence is twice this), base sequences, targets per
    /// sequence, and variant sequences run at a time.
    half: usize,
    sequences: usize,
    targets: usize,
    batch: usize,
    numeric_bytes: usize,
    tile_rows: usize,
    seed: u64,
    /// The circuit's heads: previous-token, induction and copy head, as `L{l}.H{h}`.
    previous: String,
    induction: String,
    copy: String,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// One prediction's outcome over the targets.
#[derive(Default)]
struct Tally {
    held: usize,
    count: usize,
    left: f64,
    right: f64,
}

impl Tally {
    /// A target where `left` (the side that should be larger) is compared with `right`.
    fn add(&mut self, left: f64, right: f64) {
        self.count += 1;
        self.held += usize::from(left > right);
        self.left += left;
        self.right += right;
    }

    fn report(&self, name: &str, statement: &str, left: &str, right: &str) -> Value {
        let n = self.count.max(1) as f64;
        json!({"name": name, "statement": statement, "targets": self.count, "held": self.held, "fraction": self.held as f64 / n,
               "left": left, "left_mean": self.left / n, "right": right, "right_mean": self.right / n})
    }
}

fn norm(x: ndarray::ArrayView1<f64>) -> f64 {
    x.dot(&x).sqrt()
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (export, settings_path, out, artifact_path) = match &args[..] {
        [e, s, o] => (e, s, o, None),
        [e, s, o, a] => (e, s, o, Some(Path::new(a))),
        _ => return Err("EXPORT SETTINGS.json OUT.json [ARTIFACT]".into()),
    };
    let export = Path::new(export);
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    let (half, count) = (settings.half, settings.sequences);
    if end < first + 2 * count || 2 * half > settings.context || half < 8 || settings.batch == 0 {
        return Err("the held-out rows hold two passages per base sequence of half at least 8 within the context".into());
    }
    let wide = Device::single_precision(GpuPolicy::Auto).map_err(error)?;
    let model = match Device::accelerator(GpuPolicy::Auto).map_err(error)? {
        Some(cuda) => cuda,
        None => wide.clone().unwrap_or_else(Device::host),
    };
    let wide = wide.unwrap_or_else(|| model.clone());
    let imported = import_language_model(export, end, settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { return Err("a token slot".into()) };
    let rows: Vec<Vec<u32>> = tokens.chunks(settings.context).skip(first).map(<[u32]>::to_vec).collect();
    let artifact = match artifact_path {
        Some(path) => Artifact::from_bytes(&std::fs::read(path).map_err(error)?, &native.declarations)?,
        None => library_mdl::explanation(&native, &layers)?.artifact,
    };
    let library = Library::new(&model, &wide, &native, &layers, &artifact, settings.numeric_bytes, settings.tile_rows)?;
    let index = |name: &str| -> Result<usize, String> {
        library.heads().iter().position(|(l, h)| format!("L{l}.H{h}") == name).ok_or_else(|| format!("no head {name}"))
    };
    let (previous, induction, copy) = (index(&settings.previous)?, index(&settings.induction)?, index(&settings.copy)?);
    let layer_of = |h: usize| library.heads()[h].0;

    // Base sequences (a passage twice) and sources (another passage twice); the substitution
    // alphabet is the held-out text's distinct tokens.
    let repeated = |row: &[u32]| -> Vec<u32> { row[..half].iter().chain(&row[..half]).copied().collect() };
    let bases: Vec<Vec<u32>> = (0..count).map(|r| repeated(&rows[r])).collect();
    let sources: Vec<Vec<u32>> = (0..count).map(|r| repeated(&rows[count + r])).collect();
    let mut alphabet: Vec<u32> = rows.iter().flatten().copied().collect();
    alphabet.sort_unstable();
    alphabet.dedup();
    let length = 2 * half;
    let mut rng = StdRng::seed_from_u64(settings.seed);
    // Targets t = half + j with 5 ≤ j ≤ half − 4 (every compared position inside the passage).
    let targets: Vec<(usize, usize)> = (0..count)
        .flat_map(|r| {
            let mut js: Vec<usize> = (5..half - 3).collect();
            for i in 0..settings.targets.min(js.len()) {
                let pick = rng.random_range(i..js.len());
                js.swap(i, pick);
            }
            js.truncate(settings.targets);
            js.into_iter().map(move |j| (r, half + j))
        })
        .collect();
    let base = library.run(&bases, &BTreeMap::new())?;
    let source = library.run(&sources, &BTreeMap::new())?;

    // Substitutions: one token replaced per variant sequence; the written vectors of heads at t.
    let mut substitute = |r: usize, position: usize| -> Vec<u32> {
        let mut sequence = bases[r].clone();
        let original = sequence[position];
        let mut token = original;
        while token == original {
            token = alphabet[rng.random_range(0..alphabet.len())];
        }
        sequence[position] = token;
        sequence
    };
    let mut variants: Vec<(usize, usize, Vec<u32>)> = Vec::new();
    let previous_positions = |t: usize| [t - 1, t - 2, t - 4];
    let induction_positions = |t: usize| {
        let j = t - half;
        [j, j + 1, j - 2, j + 3]
    };
    for (k, &(r, t)) in targets.iter().enumerate() {
        for p in previous_positions(t).into_iter().chain(induction_positions(t)) {
            variants.push((k, p, substitute(r, p)));
        }
    }
    let mut changes: BTreeMap<(usize, usize, usize), f64> = BTreeMap::new();
    for chunk in variants.chunks(settings.batch) {
        let run = library.run(&chunk.iter().map(|(_, _, s)| s.clone()).collect::<Vec<_>>(), &BTreeMap::new())?;
        for (i, (k, p, _)) in chunk.iter().enumerate() {
            let (r, t) = targets[*k];
            for h in [previous, induction] {
                let row = |reads: &Array2<f64>, at: usize| library.write(h, &reads.row(at).insert_axis(ndarray::Axis(0)).to_owned()).row(0).to_owned();
                let changed = row(&run.reads[h], i * length + t) - row(&base.reads[h], r * length + t);
                changes.insert((*k, *p, h), norm(changed.view()));
            }
        }
    }
    let mut previous_selects = Tally::default();
    let mut induction_selects = Tally::default();
    for (k, &(_, t)) in targets.iter().enumerate() {
        let [rel, a, b] = previous_positions(t);
        previous_selects.add(changes[&(k, rel, previous)], changes[&(k, a, previous)].max(changes[&(k, b, previous)]));
        let [j, j1, a, b] = induction_positions(t);
        induction_selects.add(changes[&(k, j, induction)].min(changes[&(k, j1, induction)]), changes[&(k, a, induction)].max(changes[&(k, b, induction)]));
    }

    // Path patching: one head's write from the source run replaces its base write in one read of
    // a later head, the direct path only.
    let patched_stream = |writer: usize, reader: usize| -> Array2<f64> {
        let stream = &base.streams[2 * layer_of(reader)];
        stream - &library.write(writer, &base.reads[writer]) + &library.write(writer, &source.reads[writer])
    };
    let reader_stream = |reader: usize| base.streams[2 * layer_of(reader)].clone();
    let induction_key = patched_stream(previous, induction);
    let (_, key_weights) = library.head_on(induction, &reader_stream(induction), &induction_key, &reader_stream(induction), length);
    let (_, query_weights) = library.head_on(induction, &induction_key, &reader_stream(induction), &reader_stream(induction), length);
    let base_weights = &base.weights[induction];
    let mut key_reads_previous = Tally::default();
    for &(r, t) in &targets {
        let s = t - half + 1;
        let drop = |weights: &[Array2<f64>]| base_weights[r][[t, s]] - weights[r][[t, s]];
        key_reads_previous.add(drop(&key_weights), drop(&query_weights));
    }
    let copy_value = patched_stream(induction, copy);
    let (copy_by_value, _) = library.head_on(copy, &reader_stream(copy), &reader_stream(copy), &copy_value, length);
    let (copy_by_query, _) = library.head_on(copy, &copy_value, &reader_stream(copy), &reader_stream(copy), length);
    let mut copy_reads_induction = Tally::default();
    let copy_base = library.write(copy, &base.reads[copy]);
    let (value_writes, query_writes) = (library.write(copy, &copy_by_value), library.write(copy, &copy_by_query));
    for &(r, t) in &targets {
        let row = r * length + t;
        copy_reads_induction.add(norm((&value_writes.row(row) - &copy_base.row(row)).view()), norm((&query_writes.row(row) - &copy_base.row(row)).view()));
    }

    // Writes: a head's read set to zero at one row, the next token's log-probability at t.
    let base_log_p = library.log_probabilities(&base.last)?;
    let mut writes = BTreeMap::new();
    for h in [induction, copy] {
        let mut tally = Tally::default();
        let cases: Vec<(usize, usize, usize)> = targets.iter().enumerate().flat_map(|(k, &(r, t))| [(k, r, t), (k, r, t - 5)]).collect();
        let mut drops: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for chunk in cases.chunks(settings.batch) {
            let sequences: Vec<Vec<u32>> = chunk.iter().map(|(_, r, _)| bases[*r].clone()).collect();
            let mut read = Array2::<f64>::zeros((chunk.len() * length, base.reads[h].ncols()));
            for (i, (_, r, at)) in chunk.iter().enumerate() {
                read.slice_mut(ndarray::s![i * length..(i + 1) * length, ..]).assign(&base.reads[h].slice(ndarray::s![r * length..(r + 1) * length, ..]));
                read.row_mut(i * length + at).fill(0.0);
            }
            let run = library.run(&sequences, &[(h, read)].into())?;
            for (i, (k, r, at)) in chunk.iter().enumerate() {
                let t = targets[*k].1;
                let next = bases[*r][t + 1] as usize;
                let log_p = library.log_probabilities(&run.last.row(i * length + t).insert_axis(ndarray::Axis(0)).to_owned())?;
                drops.insert((*k, *at), base_log_p[[r * length + t, next]] - log_p[[0, next]]);
            }
        }
        for (k, &(_, t)) in targets.iter().enumerate() {
            tally.add(drops[&(k, t)], drops[&(k, t - 5)]);
        }
        writes.insert(h, tally);
    }

    let names = |h: usize| {
        let (l, head) = library.heads()[h];
        format!("L{l}.H{head}")
    };
    let report = json!({
        "export": export.display().to_string(),
        "artifact": artifact_path.map(|p| p.display().to_string()),
        "artifact_sha256": artifact_path.map(sha256).transpose()?,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "model_device": model.name(),
        "heads": {"previous": names(previous), "induction": names(induction), "copy": names(copy)},
        "targets": targets.len(),
        "predictions": [
            previous_selects.report("previous selects t-1", "substituting the token at t-1 changes the previous-token head's write at t more than substituting it at t-2 or t-4", "|change| at t-1", "max |change| at t-2, t-4"),
            induction_selects.report("induction selects j, j+1", "substituting the token at j or j+1 changes the induction head's write at t more than substituting it at j-2 or j+3", "min |change| at j, j+1", "max |change| at j-2, j+3"),
            key_reads_previous.report("induction key reads previous", "the previous-token head's write from another passage in the induction head's key read lowers its attention to j+1 more than in its query read", "attention drop, key", "attention drop, query"),
            writes[&induction].report("induction writes next", "the induction head's read set to zero at t lowers log p(x_{j+1}) at t more than at t-5", "drop, zero at t", "drop, zero at t-5"),
            copy_reads_induction.report("copy value reads induction", "the induction head's write from another passage in the copy head's value read changes its write at t more than in its query read", "|change|, value", "|change|, query"),
            writes[&copy].report("copy writes next", "the copy head's read set to zero at t lowers log p(x_{j+1}) at t more than at t-5", "drop, zero at t", "drop, zero at t-5"),
        ],
    });
    std::fs::write(out, serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    log::info!("contracts: {}", report["predictions"]);
    Ok(())
}
