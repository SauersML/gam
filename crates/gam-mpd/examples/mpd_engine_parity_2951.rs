//! Parity of an interchange engine against the reference (#2951): every experiment family scored
//! by a candidate block engine (`interchange::BlockEngine`) and by the reference (`Model`, the
//! resident-value program in float64 on the host), values and gradients, on a language model.
//!
//! `mpd_engine_parity_2951 MODEL SEQUENCES CONTEXT OUT [WINDOWS]`
//!
//! `MODEL` is an engine export (`export.json`; its token rows) or a Hugging Face checkpoint
//! directory (`config.json`; token rows from `WINDOWS`, rows of `CONTEXT` little-endian u32). `P` is
//! the library explanation (`library_mdl::explanation`) with every trainable value moved off `M`
//! by a deterministic relative perturbation, so that divergences and gradients are not zero. The
//! candidates on the accelerator in f32 storage are its program engine (`Model`) and, where the
//! programs are of the decoder family, the fused decoder (`decoder::Decoder`) the experiments run
//! on there.
//!
//! The families, `SEQUENCES` bases each (the first rows; the next rows are the sources): clean
//! with `P` alone; clean under a random block-subset hybrid; a read patch of one of `M`'s read
//! variables; a joint read patch; both with `M`'s directions (the library's start); and a read
//! patch with `P`'s own directions (adaptive). Per family: the largest per-token difference of
//! `KL(M_e ‖ P_e)` in bits and the reference's mean, and the gradient's relative difference
//! `|g − g_ref| / |g_ref|` over all trainable operators, per candidate (a fused family that fails
//! records its error). One JSON object goes to `OUT/engine_parity.json` and stdout.

use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::log_to_stderr,
    import::{hugging_face_language_model, import_language_model},
    interchange::{self, Batch, Experiment, Interchange, Patch},
    library_mdl,
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::path::Path;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A family's experiments: per base one experiment.
fn family(name: &str, rng: &mut StdRng, sequences: usize, blocks: usize, length: usize, variables: &[interchange::ReadVariable]) -> Vec<Experiment> {
    let at_block = |b: usize| -> Vec<usize> { (0..variables.len()).filter(|i| variables[*i].block == b).collect() };
    (0..sequences)
        .map(|n| {
            let (explained, patch) = match name {
                "clean_alone" => (vec![true; blocks], None),
                "clean_hybrid" => (interchange::hybrid(rng, blocks), None),
                _ => {
                    let candidates = at_block(rng.random_range(0..blocks));
                    let patch = if name == "read_joint" {
                        Patch::Reads { variables: interchange::subset(rng, &candidates) }
                    } else {
                        Patch::Read { variable: candidates[rng.random_range(0..candidates.len())] }
                    };
                    (interchange::hybrid(rng, blocks), Some(patch))
                }
            };
            let position = if patch.is_some() { rng.random_range(0..length) } else { 0 };
            Experiment { base: n, source: n, explained, patch, position }
        })
        .collect()
}

/// `KL` per experiment and token, and the gradient per trainable operator (host), of `experiments`
/// on `x` with directions at `values`.
fn scored(x: &Interchange, batch: &Batch, experiments: &[Experiment], values: &[Array2<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), String> {
    let design = x.design_at(x.variables(), experiments, values)?;
    let scored = x.evaluate(batch, experiments, &design, true)?;
    Ok((scored.bits, scored.gradient))
}

/// [`scored`] on the reference engine of `x`'s device (its programs), whether or not `x` runs
/// the fused engine; the gradient in `trainable` order.
fn scored_by_programs(x: &Interchange, trainable: &[usize], batch: &Batch, experiments: &[Experiment], values: &[Array2<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), String> {
    let design = x.design_at(x.variables(), experiments, values)?;
    let (m, p) = x.models();
    let targets = interchange::targets(&m, x.head(), batch, experiments, &design)?;
    let evaluation = interchange::evaluate(&m, &p, x.head(), batch, &targets, experiments, &design, true)?;
    let d = p.program.device();
    let gradient = trainable
        .iter()
        .map(|op| match evaluation.gradient.get(op) {
            Some(g) => d.download(g).map_err(error),
            None => p.program.dense(*op).map(|t| Array2::zeros((t.rows(), t.cols()))),
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((evaluation.bits, gradient))
}

/// The differences of `candidate` from `reference`: the largest per-token difference of the bits,
/// and the gradient's relative difference; with the reference's mean bits per token.
fn compare((bits, gradient): &(Vec<Vec<f64>>, Vec<Array2<f64>>), (candidate_bits, candidate_gradient): &(Vec<Vec<f64>>, Vec<Array2<f64>>)) -> Value {
    let tokens = bits.iter().map(Vec::len).sum::<usize>() as f64;
    let mean = bits.iter().flatten().sum::<f64>() / tokens;
    let largest = bits.iter().flatten().zip(candidate_bits.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let (mut difference, mut norm) = (0.0, 0.0);
    for (g, c) in gradient.iter().zip(candidate_gradient) {
        difference += (g - c).mapv(|v| v * v).sum();
        norm += g.mapv(|v| v * v).sum();
    }
    json!({"mean_bits_per_token": mean, "largest_token_difference_bits": largest, "gradient_relative_difference": (difference / norm).sqrt()})
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "MODEL SEQUENCES CONTEXT OUT [WINDOWS]";
    let [model, sequences, context, out, windows @ ..] = &args[..] else {
        return Err(usage.into());
    };
    let parse = |s: &str| s.parse::<usize>().map_err(|e| format!("{s}: {e}"));
    let (sequences, context) = (parse(sequences)?, parse(context)?);
    if sequences == 0 || context == 0 {
        return Err("positive sequences and context required".into());
    }
    let model = Path::new(model);
    let (program, rows, layer_count) = if model.join("config.json").exists() {
        let [windows] = windows else { return Err(format!("a Hugging Face checkpoint needs WINDOWS: {usage}")) };
        let text = std::fs::read_to_string(model.join("config.json")).map_err(error)?;
        let layers = serde_json::from_str::<Value>(&text).map_err(error)?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
        let bytes = std::fs::read(windows).map_err(|e| format!("{windows}: {e}"))?;
        if bytes.len() < 2 * sequences * context * 4 {
            return Err(format!("{windows}: fewer than {} rows of {context} tokens", 2 * sequences));
        }
        let tokens: Vec<u32> = bytes[..2 * sequences * context * 4].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        (hugging_face_language_model(model, 0..layers)?.0, tokens.chunks(context).map(<[u32]>::to_vec).collect::<Vec<_>>(), layers)
    } else {
        let imported = import_language_model(model, 2 * sequences, context)?;
        let layers = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
            return Err("a token slot".into());
        };
        (imported.program, tokens.chunks(context).map(<[u32]>::to_vec).collect(), layers)
    };
    let native = split_sites(&program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let batch = Batch::new(rows[..sequences].to_vec(), rows[sequences..2 * sequences].to_vec())?;
    let explanation = library_mdl::explanation(&native, &layers)?;
    let variables = interchange::library_reads(&explanation.artifact.program, layer_count)?;
    let trainable = &explanation.trainable;
    let start: Vec<Array2<f64>> = trainable.iter().map(|op| explanation.artifact.program.operators[*op].matrix()).collect();
    // P: every trainable value moved by a relative 10%, deterministically.
    let mut rng = StdRng::seed_from_u64(1);
    let moved: Vec<Array2<f64>> = start.iter().map(|m| m.mapv(|v| v * (1.0 + 0.1 * (2.0 * rng.random::<f64>() - 1.0)))).collect();
    let host = Device::host();
    let accelerator = Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("no accelerator")?;
    let mut reference = Interchange::new(&host, &native, &layers, &explanation.artifact, trainable, variables.clone(), usize::MAX, 4096)?;
    let mut candidate = Interchange::new(&accelerator, &native, &layers, &explanation.artifact, trainable, variables.clone(), usize::MAX, 4096)?;
    reference.load(&moved)?;
    candidate.load(&moved)?;
    let blocks = 2 * layer_count;
    // The families' experiments, and the reference's scores of them.
    let mut families = Vec::new();
    for name in ["clean_alone", "clean_hybrid", "read", "read_joint", "adaptive"] {
        let experiments = family(name, &mut rng, sequences, blocks, context, &variables);
        let directions = if name == "adaptive" { &moved } else { &start };
        let expected = scored(&reference, &batch, &experiments, directions)?;
        families.push((name, experiments, directions, expected));
    }
    // The accelerator's program engine first, then the fused engine when it runs: a fault in the
    // second leaves the first's numbers.
    let mut programs = serde_json::Map::new();
    for (name, experiments, directions, expected) in &families {
        let row = compare(expected, &scored_by_programs(&candidate, trainable, &batch, experiments, directions)?);
        log::info!("parity {name}, programs: {row}");
        programs.insert((*name).into(), row);
    }
    let mut fused = serde_json::Map::new();
    if candidate.fused() {
        for (name, experiments, directions, expected) in &families {
            let row = match scored(&candidate, &batch, experiments, directions) {
                Ok(found) => compare(expected, &found),
                Err(e) => json!({"error": e}),
            };
            log::info!("parity {name}, fused: {row}");
            fused.insert((*name).into(), row);
        }
    }
    let report = json!({
        "model": model.display().to_string(),
        "reference": host.name(),
        "candidate": accelerator.name(),
        "sequences": sequences,
        "context": context,
        "programs": programs,
        "fused": if candidate.fused() { Value::Object(fused) } else { Value::Null },
    });
    println!("{report}");
    std::fs::create_dir_all(out).map_err(error)?;
    std::fs::write(Path::new(out).join("engine_parity.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)
}
