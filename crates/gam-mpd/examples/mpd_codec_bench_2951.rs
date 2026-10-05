//! Wall time of canonical saved-byte replay (f32 rounding, encode, decode, re-encode) on the native
//! artifact and on a composed candidate graft, uncached and with the native codeword cache.
//! EXPORT SETTINGS.json: the composed-search settings supply the grammar, width, uses, first
//! expression ID and seed of the graft (the e15 workload by default). Bytes are checked identical
//! across every path.
use gam_mpd::{
    artifact::Artifact,
    canonical_artifact::CanonicalArtifactCache,
    composed_rule_search::{self, Grammar, UseSpec},
    import::import_language_model,
    operator_program::OperatorBody,
    run_check::{layer_nodes, split_sites},
};
use serde::Deserialize;
use std::{path::Path, time::Instant};

#[derive(Deserialize)]
struct Settings {
    layers: usize,
    uses: Vec<usize>,
    width: usize,
    grammar: Grammar,
    expression_ids: Vec<usize>,
    seed: u64,
}

fn timed<T>(label: &str, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let started = Instant::now();
    let out = work()?;
    eprintln!("  {label:<28}{:>9.3} s", started.elapsed().as_secs_f64());
    Ok(out)
}

/// One canonical replay: the bytes, and the re-encoded decode's bytes.
fn replay(label: &str, artifact: &Artifact, cache: Option<&CanonicalArtifactCache>) -> Result<Vec<u8>, String> {
    eprintln!("{label}");
    let f32 = timed("f32 literals", || artifact.f32_literals())?;
    let bytes = match cache {
        Some(cache) => {
            let replay = timed("cached canonical", || cache.canonical(&f32))?;
            eprintln!("    {:?}", replay.timings);
            replay.bytes
        }
        None => {
            let bytes = timed("encode", || f32.to_bytes())?;
            let decoded = timed("decode", || Artifact::from_bytes(&bytes, &artifact.program.declarations))?;
            if timed("re-encode", || decoded.to_bytes())? != bytes {
                return Err(format!("{label}: noncanonical replay"));
            }
            bytes
        }
    };
    eprintln!("  {} bytes", bytes.len());
    Ok(bytes)
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("EXPORT SETTINGS.json".into());
    }
    let settings: Settings =
        serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let imported = timed("import", || import_language_model(Path::new(&args[0]), 1, 8))?;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, settings.layers)?;
    let base = Artifact::native(&native)?;
    let reals: Vec<usize> = native
        .operators
        .iter()
        .map(|op| match &op.body {
            OperatorBody::Identity => 0,
            OperatorBody::Diagonal { values, .. } => values.len(),
            OperatorBody::Dense { values, .. } => values.len(),
            OperatorBody::LowRank { left, right, .. } => left.len() + right.len(),
        })
        .collect();
    eprintln!(
        "{} operators, {} reals, largest {}",
        reals.len(),
        reals.iter().sum::<usize>(),
        reals.iter().max().copied().unwrap_or(0)
    );
    let inventory = composed_rule_search::enumerate(&settings.grammar)?;
    let id = *settings.expression_ids.first().ok_or("an expression ID")?;
    let specs = settings
        .uses
        .iter()
        .map(|&i| {
            Ok(UseSpec {
                input_width: native.node_interface(layers[i].normed).map_err(|e| e.to_string())?.width(),
                output_width: native.node_interface(layers[i].mlp).map_err(|e| e.to_string())?.width(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let proposal = composed_rule_search::compile(&inventory.expressions[id], settings.width, &specs, settings.seed)?;
    let mut candidate = base.clone();
    for (slot, &layer) in settings.uses.iter().enumerate() {
        let function = composed_rule_search::function(&proposal, slot)?;
        candidate = candidate.replace_function(&format!("composed-mlp-{layer}"), &function, layers[layer].normed, layers[layer].mlp)?;
    }
    let plain_native = replay("native, uncached", &base, None)?;
    let plain_candidate = replay("candidate, uncached", &candidate, None)?;
    let cache = timed("cache construction", || CanonicalArtifactCache::new(&base, 16 << 30))?;
    eprintln!("  {:?}", cache.stats());
    if replay("native, cached", &base, Some(&cache))? != plain_native {
        return Err("cached native bytes differ".into());
    }
    if replay("candidate, cached", &candidate, Some(&cache))? != plain_candidate {
        return Err("cached candidate bytes differ".into());
    }
    eprintln!("cache usage {:?}", cache.usage());
    let saved = timed("saved candidate decode", || cache.decode_saved(&plain_candidate, &candidate.program.declarations))?;
    if timed("saved candidate re-encode", || saved.to_bytes())? != plain_candidate {
        return Err("saved candidate replay differs".into());
    }
    eprintln!("identical bytes on every path");
    Ok(())
}
