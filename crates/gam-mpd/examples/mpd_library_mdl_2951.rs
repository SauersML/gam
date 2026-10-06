//! The library explanation of a language model, fitted end to end by variational minimum
//! description length on interchange experiments (`gam_mpd::library_mdl`, #2951), on an export's
//! token rows, and scored on held-out rows after every epoch.
//!
//! MODEL SETTINGS.json OUT host|gpu [artifact | edits EDITS.json]
//!
//! `MODEL` is an engine export (`export.json` and its token rows), or a Hugging Face checkpoint
//! directory (`config.json` and its safetensors, one file or sharded) whose token rows come from
//! the settings' `windows` (files of rows of `context` little-endian u32, as
//! `bench/mpd_qwen3_fineweb_2951.py` writes them): its `held_out` file's rows and its `training`
//! file's rows are the two sets. `export_sha256` is `export.json`'s SHA-256, or for a checkpoint
//! the fingerprint of `config.json`'s and every safetensors file's SHA-256 (a mismatch names it).
//!
//! On an export, the sequences `held_out = [start, end)` are never fitted (for VPD-4L's
//! `vpd4l_clean4096`, rows 1024..1056 are `vpd4l_frontier32`); the training sequences are the
//! first `training_sequences` of the others, in order. On a checkpoint, `held_out` is a range of
//! the held-out file's rows and the training sequences are the training file's first rows. `gpu` is the single-precision device (CUDA in f32 storage, or the Apple GPU). The
//! fit is checkpointed in `OUT/checkpoint.bin` after every epoch, with its trajectory readable in
//! `OUT/checkpoint.json`; rerunning the same command resumes it. With `artifact`, nothing is
//! fitted: the explanation the checkpoint holds, at its posterior mean with the device's literals,
//! is written to `OUT/checkpoint.artifact.bin` (a running fit's explanation, read where it is wanted).
//!
//! With `edits EDITS.json`, nothing is fitted: edit faithfulness. The explanation (the posterior mean
//! of the checkpoint `OUT/{checkpoint}`, or with no checkpoint the library as built, the transcoder
//! features as they are) is scored on held-out sequences `sequences` (a range of the held-out rows)
//! under edits of its parts (`interchange::Interchange::sample_edits`: per sequence its clean
//! experiment and `edits_per_sequence` edits, each of a family in `families`, `remove_part`,
//! `amplify_part`, `remove_parts` (a random subset of the parts firing at a row of one block),
//! `swap_part` (a part's activation from the next held-out sequence), `remove_head` or
//! `cut_connection` (its source the next held-out sequence),
//! applied identically to `M` and to `P`). `OUT/EDITS_{name}.json` holds per family
//! `KL(M_e ‖ P_e)` in bits per token: the mean and 99th percentile over every scored token (from the
//! edited token on) and over the edited tokens alone, with the clean experiments' as `clean`; and
//! next to it, over the same tokens, the edit's effect on the model `KL(M_e ‖ M)` (`effect_*`), the
//! size of the change the explanation is asked to predict, and the gaps again in bins of the
//! effect at the edited token (`by_effect`: below 0.01, 0.01–0.1, 0.1–1 and above 1 bits), so a
//! comparison can rest on the edits that change `M`. Each family states its `objects`:
//! `native` for experiments on `M`'s own objects (clean text, head removals), the same for every
//! explanation and the primary comparison between explanations, and `own_parts` for edits of the
//! explanation's own parts.
//!
//! With `blocks` (block `2l` layer `l`'s attention, `2l + 1` its MLP), the explanation is of those
//! blocks alone and `M` everywhere else (`library_mdl::scoped`): the fast loop for comparing method
//! changes, F against N for one block before a whole-model run.
//!
//! With `transcoders` (`{"dir": D, "layers": [l, ...]}`, `D/layer_{l}.safetensors` circuit-tracer
//! transcoder files), those layers' MLPs are the transcoders' features (`library_transcoder`, with
//! one described vector at each sequence's first token, started at `M`'s MLP output there averaged
//! over the counted training sequences; kept files written before it are refused): the features that fire on the training
//! sequences' later tokens at `M`'s MLP inputs are written to
//! `OUT/transcoder_l{l}.safetensors` (kept from an earlier run of the same command), with their
//! counts in `OUT/TRANSCODERS.json`; every other layer keeps `M`'s own MLP functions. With
//! `min_frequency`, only features firing on at least that fraction of the counted tokens are kept,
//! counted on the first `count_sequences` training sequences when given.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::{hugging_face_language_model, hugging_face_language_model_prefix, import_language_model},
    interchange, library_mdl, library_transcoder,
    operator_program::{OperatorProgram, SlotValues},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use gam_runtime::warm_start::Fingerprinter;
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    training_sequences: usize,
    context: usize,
    held_out: [usize; 2],
    /// The blocks the explanation explains, when not all of them.
    #[serde(default)]
    blocks: Option<Vec<usize>>,
    /// A Hugging Face checkpoint's token rows (absent for an engine export, which holds its own).
    #[serde(default)]
    windows: Option<Windows>,
    /// A Hugging Face checkpoint cut to its first `layers` blocks, then its final norm and head
    /// (`import::hugging_face_language_model_prefix`): the same fit on a smaller model of the same
    /// tokens, for measuring per-layer costs. All blocks when absent.
    #[serde(default)]
    layers: Option<usize>,
    /// Layers whose MLPs are transcoder features, and the directory of the transcoder files.
    #[serde(default)]
    transcoders: Option<Transcoders>,
    fit: library_mdl::Settings,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Transcoders {
    dir: PathBuf,
    layers: Vec<usize>,
    /// Keep only features firing on at least this fraction of the training tokens after the
    /// first (every feature that fires, when absent): F's keep rule `N f Δ > |G| / (2 ln 2)` for a
    /// feature of `|G|` parameters saving `Δ` bits per firing at `N` tokens, with `Δ` at most a
    /// few bits, rules out features below about 1e-3 at N = 2^22 (toygate, 10-06).
    #[serde(default)]
    min_frequency: Option<f64>,
    /// Count the firing on the first this many training sequences (all of them when absent): the
    /// count reads every feature of every layer (28 × 163,840 for Qwen3-0.6B), so on a 2^24-token
    /// training set it costs about four times a 2^22-token one, and a frequency of 1e-4 is already
    /// counted to within a few percent from 2^22 tokens (about 420 firings).
    #[serde(default)]
    count_sequences: Option<usize>,
}

/// Per transcoder layer its kept file in `out`: the features that fire on `train` at `M`'s MLP
/// inputs, counted on `device` once and kept for a resumed run, with the counts in
/// `out/TRANSCODERS.json`.
fn transcoder_files(device: &Device, native: &OperatorProgram, layers: &[LayerNodes], settings: &Transcoders, train: &[Vec<u32>], batch: usize, out: &Path) -> Result<BTreeMap<usize, PathBuf>, String> {
    let files: BTreeMap<usize, PathBuf> = settings.layers.iter().map(|&l| (l, out.join(format!("transcoder_l{l}.safetensors")))).collect();
    if files.values().all(|f| f.exists()) {
        return Ok(files);
    }
    let started = Instant::now();
    let transcoders = settings
        .layers
        .iter()
        .map(|&l| Ok((l, library_transcoder::Transcoder::open(&settings.dir.join(format!("layer_{l}.safetensors")))?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let train = &train[..settings.count_sequences.map_or(train.len(), |n| n.min(train.len()))];
    let fired = library_transcoder::firing(device, native, layers, &transcoders, train, batch)?;
    // Tokens after each sequence's first, where the transcoders run (the first takes the block's sink vector).
    let tokens: usize = train.iter().map(|s| s.len() - 1).sum();
    let mut record = Vec::new();
    for (l, transcoder) in &transcoders {
        let least = settings.min_frequency.map_or(1.0, |f| (f * tokens as f64).max(1.0));
        let counts = &fired[l].counts;
        let kept: Vec<usize> = (0..transcoder.features).filter(|&f| counts[f] as f64 >= least).collect();
        transcoder.write_kept(&kept, &fired[l].sink, &files[l])?;
        let fired_tokens: u64 = counts.iter().sum();
        let kept_fired: u64 = kept.iter().map(|&f| counts[f]).sum();
        record.push(json!({
            "layer": l,
            "features": transcoder.features,
            "kept": kept.len(),
            "active_per_token": fired_tokens as f64 / tokens as f64,
            "kept_active_per_token": kept_fired as f64 / tokens as f64,
            "ever_fired": counts.iter().filter(|&&c| c > 0).count(),
            "tokens": tokens,
        }));
        log::info!("transcoder layer {l}: {} of {} features fire on {tokens} training tokens after the first, {:.2} per token", kept.len(), transcoder.features, fired_tokens as f64 / tokens as f64);
    }
    save(&out.join("TRANSCODERS.json"), &json!({"layers": record, "seconds": started.elapsed().as_secs_f64()}))?;
    Ok(files)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Windows {
    training: PathBuf,
    held_out: PathBuf,
}

/// The first `count` rows of `context` tokens of a windows file.
fn rows(path: &Path, count: usize, context: usize) -> Result<Vec<Vec<u32>>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < count * context * 4 {
        return Err(format!("{}: fewer than {count} rows of {context} tokens", path.display()));
    }
    Ok(bytes[..count * context * 4]
        .chunks_exact(context * 4)
        .map(|row| row.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        .collect())
}

/// A Hugging Face checkpoint's fingerprint: `config.json`'s and every safetensors file's SHA-256,
/// in name order.
fn checkpoint_digest(dir: &Path) -> Result<String, String> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .map(|entry| entry.map(|e| e.path()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "config.json") || p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    names.sort();
    let mut fingerprint = Fingerprinter::new();
    for path in &names {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        fingerprint.absorb_str(name.as_bytes(), &sha256(path)?);
    }
    Ok(fingerprint.finalize().to_hex())
}

/// The settings of an edit-faithfulness run (module note).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EditSettings {
    /// The fit checkpoint in `OUT` whose posterior mean is scored; none scores the library as built.
    checkpoint: Option<String>,
    sequences: [usize; 2],
    families: Vec<interchange::Family>,
    edits_per_sequence: usize,
    batch_sequences: usize,
    seed: u64,
    /// The most bytes of operator values each program holds on the device.
    numeric_bytes: usize,
    name: String,
    /// When set, nothing is scored: the explanation's parts (its transcoder features at the
    /// scored posterior mean) are written to `OUT/{functions}` in the oracle's names
    /// (`bench/oracle`): per layer `l` with parts, `h.{l}.mlp.function.U` [C, d] (the writes),
    /// `.V` [d, C] (the reads), `.bias` [C] and `.index` [C] (each part's row in the layer's
    /// operators, its transcoder feature through the kept file), in float32.
    #[serde(default)]
    functions: Option<String>,
}

/// The parts in the oracle's names (`EditSettings::functions`), a safetensors file.
fn write_functions(parts: &[interchange::Part], path: &Path) -> Result<(), String> {
    let mut by_layer: BTreeMap<usize, Vec<&interchange::Part>> = BTreeMap::new();
    for part in parts {
        by_layer.entry((part.block - 1) / 2).or_default().push(part);
    }
    let floats = |values: &mut dyn Iterator<Item = f64>| -> Vec<u8> { values.flat_map(|v| (v as f32).to_le_bytes()).collect() };
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (l, chosen) in &by_layer {
        let (c, d) = (chosen.len(), chosen[0].write.len());
        let name = format!("h.{l}.mlp.function");
        tensors.push((format!("{name}.U"), vec![c, d], floats(&mut chosen.iter().flat_map(|p| p.write.iter().copied()))));
        tensors.push((format!("{name}.V"), vec![d, c], floats(&mut (0..d).flat_map(|j| chosen.iter().map(move |p| p.read[j])))));
        tensors.push((format!("{name}.bias"), vec![c], floats(&mut chosen.iter().map(|p| p.bias))));
        tensors.push((format!("{name}.index"), vec![c], floats(&mut chosen.iter().map(|p| p.index as f64))));
    }
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape, data) in &tensors {
        header.insert(name.clone(), json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + data.len()]}));
        offset += data.len();
    }
    let mut text = Value::Object(header).to_string().into_bytes();
    while text.len() % 8 != 0 {
        text.push(b' ');
    }
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&text);
    tensors.iter().for_each(|(_, _, data)| bytes.extend_from_slice(data));
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Edit faithfulness (module note): per family the bits per token of `KL(M_e ‖ P_e)`.
fn edit_faithfulness(
    device: &Device,
    (native, layers): (&OperatorProgram, &[LayerNodes]),
    explanation: &library_mdl::Explanation,
    identity: &library_mdl::Identity,
    held_out: &[Vec<u32>],
    settings: &EditSettings,
    out: &Path,
) -> Result<(), String> {
    let started = Instant::now();
    let [first, end] = settings.sequences;
    if first >= end || end > held_out.len() || settings.batch_sequences == 0 {
        return Err("edits: a nonempty range of the held-out sequences and a batch size are required".into());
    }
    let artifact = match &settings.checkpoint {
        Some(file) => {
            let path = out.join(file);
            library_mdl::check_checkpoint(&path, identity)?;
            library_mdl::checkpoint_artifact(explanation, &path, library_mdl::Literals::of(device))?
        }
        None => explanation.artifact.clone(),
    };
    let parts = interchange::parts_of(&artifact.program, layers.len())?;
    if let Some(file) = &settings.functions {
        log::info!("edits: {} parts written to {file}", parts.len());
        return write_functions(&parts, &out.join(file));
    }
    let mut experiments = interchange::Interchange::new(device, native, layers, &artifact, &[], explanation.reads.clone(), settings.numeric_bytes, 256)?;
    let count = parts.len();
    experiments.set_parts(parts.clone())?;
    log::info!("edits: {count} parts, {} held-out sequences, {:.0} s to compile", end - first, started.elapsed().as_secs_f64());
    let mut rng = rand::rngs::StdRng::seed_from_u64(settings.seed);
    let family = |e: &interchange::Experiment| match &e.patch {
        None => "clean",
        Some(interchange::Patch::Part { factor: 0, .. }) => "remove_part",
        Some(interchange::Patch::Part { .. }) => "amplify_part",
        Some(interchange::Patch::Head { .. }) => "remove_head",
        Some(interchange::Patch::Cut { .. }) => "cut_connection",
        Some(interchange::Patch::Parts { .. }) => "remove_parts",
        Some(interchange::Patch::Swap { .. }) => "swap_part",
        Some(interchange::Patch::PartFrom { .. }) => "remove_part_from",
        Some(interchange::Patch::HeadFrom { .. }) => "remove_head_from",
        Some(_) => "read",
    };
    // Per family: every scored token's bits, the edited tokens' bits, and the experiments.
    let mut scores: BTreeMap<&str, (Vec<f64>, Vec<f64>, usize)> = BTreeMap::new();
    let mut batches = Vec::new();
    for (b, chunk) in held_out[first..end].chunks(settings.batch_sequences).enumerate() {
        let batch = interchange::Batch::new(chunk.to_vec(), chunk.to_vec())?;
        let drawn = experiments.sample_edits(&mut rng, &batch, &settings.families, settings.edits_per_sequence, false)?;
        let scored = experiments.evaluate(&batch, &drawn, false)?;
        for (e, bits) in drawn.iter().zip(&scored.bits) {
            let entry = scores.entry(family(e)).or_default();
            entry.0.extend_from_slice(bits);
            entry.1.extend(bits.first());
            entry.2 += 1;
        }
        batches.push((b, batch, drawn, scored.bits));
        log::info!("edits: batch {b} scored ({:.0} s)", started.elapsed().as_secs_f64());
    }
    // The edits' effect on M, KL(M_e ‖ M), over the same tokens: the same experiments with P = M
    // applying no edit.
    drop(experiments);
    let mut reference = interchange::Interchange::new(device, native, layers, &gam_mpd::artifact::Artifact::native(native)?, &[], explanation.reads.clone(), settings.numeric_bytes, 256)?;
    reference.set_parts(parts.clone())?;
    reference.unedited_explanation();
    // An edit's evidence grows with how much it moves M: per family, the gaps of the edits whose
    // effect at the edited token KL(M_e ‖ M) falls in each bin (bits).
    const BINS: [f64; 3] = [0.01, 0.1, 1.0];
    let mut effects: BTreeMap<&str, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut binned: BTreeMap<(&str, usize), (Vec<f64>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    // Per experiment (one JSON line each): its held-out sequence, edited position, family, parts
    // as (layer, row in the layer), factor, effect at the edited token and gap there, so another
    // implementation of the same edit (bench/oracle/qwen_labels.py) can be checked against it.
    let part_of = |i: &usize| parts.get(*i).map(|p| json!([(p.block - 1) / 2, p.index]));
    let mut records = String::new();
    for (b, batch, drawn, gaps) in &batches {
        for ((e, bits), gap) in drawn.iter().zip(&reference.evaluate(batch, drawn, false)?.bits).zip(gaps) {
            let (chosen, factor): (Vec<usize>, Option<usize>) = match &e.patch {
                Some(interchange::Patch::Part { part, factor }) => (vec![*part], Some(*factor)),
                Some(interchange::Patch::PartFrom { part, factor }) => (vec![*part], Some(*factor)),
                Some(interchange::Patch::Parts { parts: chosen, factor }) => (chosen.clone(), Some(*factor)),
                Some(interchange::Patch::Swap { part }) => (vec![*part], None),
                Some(interchange::Patch::Cut { from, to }) => (vec![*from, *to], None),
                _ => (Vec::new(), None),
            };
            records.push_str(&json!({
                "sequence": first + b * settings.batch_sequences + e.base, "position": e.position, "family": family(e),
                "parts": chosen.iter().filter_map(part_of).collect::<Vec<_>>(), "factor": factor.and_then(|f| interchange::FACTORS.get(f).copied()),
                "effect_bits_at_edited_token": bits.first(), "gap_bits_at_edited_token": gap.first(),
            }).to_string());
            records.push('\n');
            let entry = effects.entry(family(e)).or_default();
            entry.0.extend_from_slice(bits);
            entry.1.extend(bits.first());
            let at = bits.first().copied().unwrap_or(0.0);
            let bin = BINS.iter().filter(|b| at >= **b).count();
            let entry = binned.entry((family(e), bin)).or_default();
            entry.0.extend_from_slice(gap);
            entry.1.extend(gap.first());
            entry.2.push(at);
        }
    }
    std::fs::write(out.join(format!("EDITS_{}.experiments.jsonl", settings.name)), records).map_err(|e| e.to_string())?;
    let summary = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
        let p99 = values.get(((values.len() as f64 * 0.99).ceil() as usize).saturating_sub(1)).copied().unwrap_or(f64::NAN);
        (mean, p99)
    };
    let mut families = serde_json::Map::new();
    for (family, (mut all, mut edited, count)) in scores {
        let (tokens, (mean, p99), (edited_mean, edited_p99)) = (all.len(), summary(&mut all), summary(&mut edited));
        let (mut effect_all, mut effect_edited) = effects.remove(family).unwrap_or_default();
        let ((effect_mean, effect_p99), (effect_edited_mean, effect_edited_p99)) = (summary(&mut effect_all), summary(&mut effect_edited));
        // Experiments on M's own objects (clean text, heads) ask every explanation the same
        // question; edits of parts ask each explanation about its own parts.
        let objects = if matches!(family, "clean" | "remove_head" | "remove_head_from" | "read") { "native" } else { "own_parts" };
        let bins: Vec<Value> = (0..=BINS.len())
            .filter_map(|bin| {
                let (all, at, effect) = binned.get(&(family, bin))?;
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
                let low = if bin == 0 { 0.0 } else { BINS[bin - 1] };
                let high = BINS.get(bin).copied().unwrap_or(f64::INFINITY);
                Some(json!({"effect_bits_at_edited_token": [low, if high.is_finite() { json!(high) } else { json!("inf") }], "experiments": effect.len(), "mean_bits_per_token": mean(all), "edited_token_mean_bits": mean(at), "effect_edited_token_mean_bits": mean(effect)}))
            })
            .collect();
        families.insert(
            family.into(),
            json!({
                "objects": objects,
                "by_effect": bins,
                "experiments": count, "tokens": tokens,
                "mean_bits_per_token": mean, "p99_bits_per_token": p99, "edited_token_mean_bits": edited_mean, "edited_token_p99_bits": edited_p99,
                "effect_mean_bits_per_token": effect_mean, "effect_p99_bits_per_token": effect_p99, "effect_edited_token_mean_bits": effect_edited_mean, "effect_edited_token_p99_bits": effect_edited_p99,
            }),
        );
    }
    let report = json!({
        "checkpoint": settings.checkpoint,
        "parts": count,
        "sequences": settings.sequences,
        "edits_per_sequence": settings.edits_per_sequence,
        "seed": settings.seed,
        "device": device.name(),
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "families": families,
        "seconds": started.elapsed().as_secs_f64(),
    });
    log::info!("edits: {report}");
    save(&out.join(format!("EDITS_{}.json", settings.name)), &report)
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (export, settings_path, out, mode, read_artifact, edits) = match &args[..] {
        [export, settings, out, mode] => (export, settings, out, mode, false, None),
        [export, settings, out, mode, artifact] if artifact == "artifact" => (export, settings, out, mode, true, None),
        [export, settings, out, mode, edits, file] if edits == "edits" => (export, settings, out, mode, false, Some(Path::new(file))),
        _ => return Err("EXPORT SETTINGS.json OUT host|gpu [artifact | edits EDITS.json]".into()),
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let checkpoint_model = export.join("config.json").exists() && !export.join("export.json").exists();
    let digest = if checkpoint_model { checkpoint_digest(export)? } else { sha256(&export.join("export.json"))? };
    if digest != settings.export_sha256 {
        return Err(format!("model digest mismatch: {} is {digest}", export.display()));
    }
    let [first, end] = settings.held_out;
    if first >= end || settings.training_sequences == 0 {
        return Err("held-out sequences must be a nonempty range, and training sequences nonempty".into());
    }
    let checkpoint = out.join("checkpoint.bin");
    if (out.exists() || read_artifact) && edits.is_none() && !checkpoint.exists() {
        return Err("a fresh output directory, or one holding this fit's checkpoint, required".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("a GPU required")?,
        _ => return Err("host|gpu required".into()),
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let (program, layer_count, train, held_out): (OperatorProgram, usize, Vec<Vec<u32>>, Vec<Vec<u32>>) = if checkpoint_model {
        let windows = settings.windows.as_ref().ok_or("a Hugging Face checkpoint needs the settings' windows")?;
        let text = std::fs::read_to_string(export.join("config.json")).map_err(|e| e.to_string())?;
        let all = serde_json::from_str::<Value>(&text).map_err(|e| e.to_string())?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
        let (program, layers) = match settings.layers {
            Some(k) => (hugging_face_language_model_prefix(export, k)?.0, k),
            None => (hugging_face_language_model(export, 0..all)?.0, all),
        };
        let held_out = rows(&windows.held_out, end, settings.context)?[first..].to_vec();
        (program, layers, rows(&windows.training, settings.training_sequences, settings.context)?, held_out)
    } else {
        if settings.windows.is_some() || settings.layers.is_some() {
            return Err("an engine export holds its own token rows and blocks; windows and layers are for a Hugging Face checkpoint".into());
        }
        // The rows to import: the held-out range and the training sequences around it.
        let count = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
        let imported = import_language_model(export, count, settings.context)?;
        let layers = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
            return Err("a token slot".into());
        };
        let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
        let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).take(settings.training_sequences).cloned().collect();
        (imported.program, layers, train, sequences[first..end].to_vec())
    };
    if train.len() != settings.training_sequences {
        return Err("the model's token rows hold fewer training sequences than asked for".into());
    }
    let held_out = &held_out[..];
    let native = split_sites(&program)?;
    // The imported program's operators the split replaced are not read again.
    drop(program);
    let layers = layer_nodes(&native, layer_count)?;
    let explanation = match &settings.transcoders {
        Some(transcoders) => {
            let files = transcoder_files(&device, &native, &layers, transcoders, &train, settings.fit.batch_sequences, out)?;
            library_mdl::explanation_with(&native, &layers, &files)?
        }
        None => library_mdl::explanation(&native, &layers)?,
    };
    let explanation = match &settings.blocks {
        Some(blocks) => library_mdl::scoped(&explanation, blocks)?,
        None => explanation,
    };
    explanation.artifact.validate_coverage(&native)?;
    // A checkpoint there must be this fit's (export, sequences, program, groups, shared
    // parameters) before anything is written.
    let identity = library_mdl::identity(&settings.export_sha256, &native, &explanation, &train, held_out);
    library_mdl::check_checkpoint(&checkpoint, &identity)?;
    if let Some(file) = edits {
        let settings: EditSettings = serde_json::from_slice(&std::fs::read(file).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        return edit_faithfulness(&device, (&native, &layers), &explanation, &identity, held_out, &settings, out);
    }
    if read_artifact {
        let artifact = library_mdl::checkpoint_artifact(&explanation, &checkpoint, library_mdl::Literals::of(&device))?;
        artifact.validate_coverage(&native)?;
        return std::fs::write(out.join("checkpoint.artifact.bin"), artifact.to_bytes()?).map_err(|e| e.to_string());
    }
    let provenance = json!({
        "export": export.display().to_string(),
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "device": device.name(),
        "training_sequences": train.len(),
        "training_tokens": train.len() * settings.context,
        "held_out_sequences": [first, end],
        "context": settings.context,
        "groups": explanation.groups.len(),
        "blocks": settings.blocks,
        "layers": settings.layers,
        "identity": identity,
    });
    log::info!("library run: {provenance}");
    save(&out.join("START.json"), &provenance)?;
    let fit = library_mdl::fit(&device, &native, &explanation, &train, held_out, &settings.fit, &settings.export_sha256, Some(&checkpoint), None)?;
    save(&out.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
    // The reported artifact: the posterior mean or its rounding, whichever scores better held out,
    // with the literals its evaluation ran with.
    let artifact = fit.artifact(&explanation)?;
    artifact.validate_coverage(&native)?;
    std::fs::write(out.join("artifact.bin"), artifact.to_bytes()?).map_err(|e| e.to_string())?;
    let summary = json!({
        "provenance": provenance,
        "objective_bits": fit.report.objective_bits,
        "active_groups": fit.report.active_groups,
        "epochs": fit.report.epochs.len(),
        "removals": fit.report.removals.iter().map(|r| r.removed).collect::<Vec<_>>(),
        "start": fit.report.start,
        "held_out": fit.report.end,
        "representative": fit.report.representative,
        // The artifact's held-out data term against the samples' (the posterior's expected one).
        "mean_minus_sample_bits_per_token": fit.report.end.mean_bits_per_token - fit.report.end.data_bits_per_token,
        "seconds": started.elapsed().as_secs_f64(),
    });
    log::info!("library summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}
