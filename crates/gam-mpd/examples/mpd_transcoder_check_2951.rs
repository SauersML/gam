//! A real transcoder layer as library functions, checked against the transcoder (#2951,
//! `gam_mpd::library_transcoder`): the Hugging Face checkpoint `MODEL` cut after layer `LAYER`,
//! the first `ROWS` rows of `context` tokens of a windows file, all on the host in float64.
//!
//! MODEL TRANSCODER.safetensors LAYER WINDOWS CONTEXT ROWS OUT
//!
//! The features that fire on those tokens at `M`'s MLP input are kept (`OUT/kept.safetensors`);
//! the library explanation with that layer's MLP replaced by them is run, and `OUT/CHECK.json`
//! holds the largest difference between the library block's output and the full transcoder's own
//! reconstruction `Σ_i relu(g_i·x + c_i) u_i + b` at the block's input (the block's sink vector at
//! each sequence's first token, `M`'s MLP output there averaged over the rows), against the reconstruction's
//! largest entry, and the count of kept functions that are off (exactly zero) per token after the
//! first.
use gam_gpu::tensor::Device;
use gam_mpd::{
    engine::log_to_stderr,
    import::hugging_face_language_model_prefix,
    library_mdl, library_transcoder,
    run_check::{layer_nodes, split_sites},
};
use serde_json::json;
use std::{collections::BTreeMap, path::Path};

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, transcoder, layer, windows, context, rows, out] = &args[..] else {
        return Err("MODEL TRANSCODER.safetensors LAYER WINDOWS CONTEXT ROWS OUT".into());
    };
    let parse = |s: &str| s.parse::<usize>().map_err(|e| format!("{s}: {e}"));
    let (layer, context, rows) = (parse(layer)?, parse(context)?, parse(rows)?);
    let out = Path::new(out);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let bytes = std::fs::read(windows).map_err(|e| format!("{windows}: {e}"))?;
    let sequences: Vec<Vec<u32>> = bytes
        .get(..rows * context * 4)
        .ok_or("too few rows")?
        .chunks_exact(context * 4)
        .map(|row| row.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        .collect();
    let (program, _) = hugging_face_language_model_prefix(Path::new(model), layer + 1)?;
    let native = split_sites(&program)?;
    drop(program);
    let layers = layer_nodes(&native, layer + 1)?;
    let mut transcoders = BTreeMap::new();
    transcoders.insert(layer, library_transcoder::Transcoder::open(Path::new(transcoder))?);
    let fired = library_transcoder::firing(&Device::host(), &native, &layers, &transcoders, &sequences, 1)?;
    let full = &transcoders[&layer];
    let kept: Vec<usize> = (0..full.features).filter(|&f| fired[&layer].counts[f] > 0).collect();
    let kept_path = out.join("kept.safetensors");
    full.write_kept(&kept, &fired[&layer].sink, &kept_path)?;
    let explanation = library_mdl::explanation_with(&native, &layers, &BTreeMap::from([(layer, kept_path)]))?;
    explanation.artifact.validate_coverage(&native)?;
    let family = library_mdl::sequence_family(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
    let trace = explanation.artifact.execute(&family)?;
    let place = |n: usize| explanation.artifact.place(n).ok_or_else(|| format!("native node {n} has no place"));
    let x = &trace.values[place(layers[layer].normed)?];
    let mlp = &trace.values[place(layers[layer].mlp)?];
    let mut reference = full.reconstruction(x)?;
    let native_trace = native.execute(&family, false).map_err(|e| e.to_string())?;
    let first: Vec<bool> = family.layout.as_ref().ok_or("a layout")?.position.iter().map(|&p| p == 0).collect();
    // The block's sink vector at first tokens, as stored (float32).
    let sink: Vec<f64> = fired[&layer].sink.iter().map(|&v| f64::from(v as f32)).collect();
    for (row, &first) in first.iter().enumerate() {
        if first {
            reference.row_mut(row).assign(&ndarray::ArrayView1::from(&sink[..]));
        }
    }
    let scale = reference.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let difference = reference.iter().zip(mlp).fold(0.0_f64, |a, (r, m)| a.max((r - m).abs()));
    // The library's input to the block, against M's own (every other block is M's at the start).
    let native_x = native_trace.values[layers[layer].normed].clone();
    let input_difference = native_x.iter().zip(x).fold(0.0_f64, |a, (p, q)| a.max((p - q).abs()));
    let tokens = sequences.len() * (context - 1);
    let fired_tokens: u64 = fired[&layer].counts.iter().sum();
    let record = json!({
        "layer": layer,
        "tokens": tokens,
        "features": full.features,
        "kept": kept.len(),
        "active_per_token": fired_tokens as f64 / tokens as f64,
        "off_per_token": kept.len() as f64 - fired_tokens as f64 / tokens as f64,
        "max_abs_difference": difference,
        "reconstruction_max_abs": scale,
        "relative_difference": difference / scale,
        "input_max_abs_difference_from_m": input_difference,
    });
    log::info!("transcoder check: {record}");
    std::fs::write(out.join("CHECK.json"), serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
