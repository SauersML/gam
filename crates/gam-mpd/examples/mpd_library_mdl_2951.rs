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
//! under experiments drawn from the seed alone, the same for every explanation
//! (`interchange::Interchange::sample_ops`: per sequence its clean experiment and
//! `edits_per_sequence` experiments, each of a family in `families`: `swap` (sites' values from the
//! next held-out sequence), `zero`, `scale` and `push` (a seeded direction at a site's typical
//! norm), each of one to sixteen operations on sites `M` and every explanation share, at one row,
//! onward or at every row), applied identically to `M` and to `P`. `OUT/EDITS_{name}.json` holds per family
//! `KL(M_e ‖ P_e)` in bits per token: the mean and 99th percentile over every scored token (from the
//! edited token on) and over the edited tokens alone, with the clean experiments' as `clean`; and
//! next to it, over the same tokens, the edit's effect on the model `KL(M_e ‖ M)` (`effect_*`), the
//! size of the change the explanation is asked to predict, the edit-ignoring baseline
//! `KL(M_e ‖ P)` (`ignoring_*`: P's clean prediction against M's edited outcome, which a gap must
//! beat for the explanation to predict the edit at all), the response diagnostic
//! `KL(p_e ‖ p̃_e)`, `p̃_e ∝ p_0 q_e / q_0` (`response_*`: `M`'s clean prediction moved by `P`'s
//! response to the edit, zero where `P` responds as `M` does whatever its clean error; `p` = `M`,
//! `q` = `P`, `0` clean, `e` edited), and the gaps again in bins of the
//! effect at the edited token (`by_effect`: below 0.01, 0.01–0.1, 0.1–1 and above 1 bits), so a
//! comparison can rest on the edits that change `M`. With `weights`, native weight edits of `M`'s
//! maps follow (`weight_faithfulness`), compiled into `M` and into `P` through `P`'s owners
//! (`gam_mpd::weight_edit`), reported under `weights`. The experiments are an immutable manifest
//! (`Manifest`, `EditSettings::manifest`): drawn and written once, then scored as written by any
//! binary, so explanations compare on the manifest file's SHA-256 (`manifest.sha256` in the
//! report).
//!
//! With `blocks` (block `2l` layer `l`'s attention, `2l + 1` its MLP), the explanation is of those
//! blocks alone and `M` everywhere else (`library_mdl::scoped`): the fast loop for comparing method
//! changes, F against N for one block before a whole-model run.
//!
//! With `vpd` (`{"decomposition": D, "start": S, "arm": A}`), every block is VPD's slices with
//! intrinsic gates (`library_vpd`), arm `A` of the start file `S` that `mpd_battery_2951 start`
//! writes (`per_slice_own`, `grouped_own`, `grouped_direction`).
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
    /// VPD's slices with intrinsic gates in every block (`library_vpd`): the decomposition, the
    /// start file `vpd_start` writes and its arm.
    #[serde(default)]
    vpd: Option<VpdStart>,
    fit: library_mdl::Settings,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VpdStart {
    decomposition: PathBuf,
    start: PathBuf,
    arm: String,
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
    /// The experiment manifest (`Manifest`), a path in `OUT` or absolute: scored as it stands when
    /// it exists (never rewritten), else drawn and written there. None draws and writes
    /// `OUT/MANIFEST_{name}.json`, which must not exist yet.
    #[serde(default)]
    manifest: Option<String>,
    /// Native weight edits of `M`'s maps, scored after the operations (`weight_faithfulness`).
    #[serde(default)]
    weights: Option<WeightFamily>,
}

/// Native weight edits (`EditSettings::weights`): `edits` of them, each drawn from the settings'
/// seed and scored on the first `sequences` held-out sequences of the range.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightFamily {
    edits: usize,
    sequences: usize,
}

/// An immutable experiment manifest: what an edits score is a score of, so two explanations are
/// compared on the same experiments by the manifest file's SHA-256 alone. It names the export, the
/// held-out rows (their range and their token ids' digest), the seed, families, experiments per
/// sequence and batch size it was drawn with, the binary that drew it (its source revision), the
/// pushed directions' digest and the sites' typical norms (the units of a push), and every
/// experiment, batch by batch. A driver given it scores exactly those experiments, whatever its own
/// sampler would draw, and refuses it when the export, the rows or the directions differ.
#[derive(serde::Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Manifest {
    export: String,
    sequences: [usize; 2],
    rows: String,
    seed: u64,
    families: Vec<interchange::Family>,
    edits_per_sequence: usize,
    batch_sequences: usize,
    binary: Option<String>,
    directions: String,
    typical: Vec<(interchange::SharedSite, f64)>,
    experiments: Vec<Vec<interchange::Experiment>>,
    /// The native weight edits (`EditSettings::weights`), as drawn.
    #[serde(default)]
    weights: Vec<WeightDraw>,
}

/// One native weight edit of `M`'s operator `operator`, as a manifest lists it
/// (`weight_faithfulness`).
#[derive(Clone, serde::Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum WeightDraw {
    /// Its rows (`rows`), else its columns, `units` scaled by `alpha`.
    Units { operator: String, rows: bool, units: Vec<usize>, alpha: f64 },
    /// `scale · u vᵀ` added.
    RankOne { operator: String, scale: f64, u: Vec<f64>, v: Vec<f64> },
}

impl WeightDraw {
    fn operator(&self) -> &str {
        match self {
            Self::Units { operator, .. } | Self::RankOne { operator, .. } => operator,
        }
    }

    fn kind(&self) -> String {
        match self {
            Self::Units { rows, units, alpha, .. } => format!("{} {} scaled by {alpha}", units.len(), if *rows { "rows" } else { "columns" }),
            Self::RankOne { scale, .. } => format!("rank one of Frobenius size {scale:.4}"),
        }
    }

    /// The edit `ΔW` of `M`'s operator (`native` its program).
    fn delta(&self, native: &OperatorProgram) -> Result<ndarray::Array2<f64>, String> {
        let op = native.operators.iter().find(|op| op.name == self.operator()).ok_or_else(|| format!("edits: M has no operator {}", self.operator()))?;
        let w = op.matrix();
        let (r, c) = w.dim();
        match self {
            Self::Units { rows, units, alpha, .. } => {
                let mut delta = ndarray::Array2::zeros((r, c));
                for &u in units {
                    if *rows && u < r {
                        delta.row_mut(u).assign(&w.row(u).mapv(|v| v * (alpha - 1.0)));
                    } else if !*rows && u < c {
                        delta.column_mut(u).assign(&w.column(u).mapv(|v| v * (alpha - 1.0)));
                    } else {
                        return Err(format!("edits: unit {u} outside {}", self.operator()));
                    }
                }
                Ok(delta)
            }
            Self::RankOne { scale, u, v, .. } if u.len() == r && v.len() == c => Ok(ndarray::Array2::from_shape_fn((r, c), |(a, b)| scale * u[a] * v[b])),
            Self::RankOne { .. } => Err(format!("edits: a rank-one edit of another shape than {}", self.operator())),
        }
    }
}

/// `family.edits` native weight edits drawn from `seed` (`weight_faithfulness`).
fn draw_weights(native: &OperatorProgram, family: &WeightFamily, seed: u64) -> Result<Vec<WeightDraw>, String> {
    use rand::RngExt;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5745_4947_4854);
    let pool: Vec<usize> = native.operators.iter().enumerate().filter(|(_, op)| op.name.starts_with("blocks.") && op.rows.width() > 1 && op.cols.width() > 1 && op.diagonal().is_none()).map(|(i, _)| i).collect();
    // A map stored per head (q0, q1, ...) is one tensor: tensors are drawn uniformly, then one of
    // their blocks, so the many per-head blocks do not crowd out the MLPs' maps.
    let mut tensors: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for &i in &pool {
        tensors.entry(native.operators[i].name.trim_end_matches(|c: char| c.is_ascii_digit())).or_default().push(i);
    }
    let tensors: Vec<Vec<usize>> = tensors.into_values().collect();
    if tensors.is_empty() {
        return Err("edits: no map of M to edit".into());
    }
    let normal = |rng: &mut rand::rngs::StdRng, n: usize| -> Vec<f64> {
        let v: Vec<f64> = (0..n)
            .map(|_| {
                let (a, b): (f64, f64) = (rng.random::<f64>().max(f64::MIN_POSITIVE), rng.random());
                (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    };
    let mut out = Vec::with_capacity(family.edits);
    for _ in 0..family.edits {
        let blocks = &tensors[rng.random_range(0..tensors.len())];
        let op = &native.operators[blocks[rng.random_range(0..blocks.len())]];
        let (r, c) = (op.rows.width(), op.cols.width());
        let operator = op.name.clone();
        if rng.random_range(0..2) == 0 {
            let rows = rng.random_range(0..2) == 0;
            let n = if rows { r } else { c };
            let k = (1usize << rng.random_range(0..=4usize)).min(n);
            let mut units: Vec<usize> = (0..n).collect();
            for j in 0..k {
                let t = rng.random_range(j..n);
                units.swap(j, t);
            }
            units.truncate(k);
            out.push(WeightDraw::Units { operator, rows, units, alpha: interchange::SCALES[rng.random_range(0..interchange::SCALES.len())] });
        } else {
            let size = interchange::SIZES[rng.random_range(0..interchange::SIZES.len())];
            let (u, v) = (normal(&mut rng, r), normal(&mut rng, c));
            let scale = size * op.matrix().iter().map(|x| x * x).sum::<f64>().sqrt() / (r.min(c) as f64).sqrt();
            out.push(WeightDraw::RankOne { operator, scale, u, v });
        }
    }
    Ok(out)
}

/// The 64-bit FNV-1a digest of `words`, in hexadecimal.
fn digest(words: impl IntoIterator<Item = u64>) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words {
        for byte in w.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{h:016x}")
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
    log::info!("edits: {count} parts, {} held-out sequences, {:.0} s to compile", end - first, started.elapsed().as_secs_f64());
    let mut rng = rand::rngs::StdRng::seed_from_u64(settings.seed);
    // Operations on shared sites push seeded directions at each site's typical norm, measured on
    // M's own runs of the first held-out batch: the same for every explanation.
    let typical_batch: Vec<Vec<u32>> = held_out[first..end].iter().take(settings.batch_sequences).cloned().collect();
    let typical_batch = interchange::Batch::new(typical_batch.clone(), typical_batch)?;
    experiments.set_directions(interchange::DIRECTIONS, settings.seed);
    let manifest_path = out.join(settings.manifest.clone().unwrap_or_else(|| format!("MANIFEST_{}.json", settings.name)));
    let rows = digest(held_out[first..end].iter().flat_map(|s| s.iter().map(|t| u64::from(*t)).chain([u64::MAX])));
    let directions = digest(experiments.push_directions().iter().flatten().map(|v| v.to_bits()));
    let loaded: Option<Manifest> = if manifest_path.exists() {
        if settings.manifest.is_none() {
            return Err(format!("edits: {} exists and is immutable: name it in `manifest` to score on it", manifest_path.display()));
        }
        let m: Manifest = serde_json::from_slice(&std::fs::read(&manifest_path).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        let differs = [
            ("export", m.export != identity.export),
            ("held-out rows", m.sequences != settings.sequences || m.rows != rows),
            ("seed", m.seed != settings.seed),
            ("families", m.families != settings.families),
            ("experiments per sequence", m.edits_per_sequence != settings.edits_per_sequence),
            ("batch size", m.batch_sequences != settings.batch_sequences),
            ("pushed directions", m.directions != directions),
        ];
        if let Some((what, _)) = differs.iter().find(|d| d.1) {
            return Err(format!("edits: the manifest {} differs in its {what}", manifest_path.display()));
        }
        if m.weights.len() != settings.weights.as_ref().map_or(0, |w| w.edits) {
            return Err(format!("edits: the manifest {} lists {} weight edits", manifest_path.display(), m.weights.len()));
        }
        experiments.set_typical(m.typical.iter().copied().collect());
        Some(m)
    } else {
        experiments.measure_typical(&typical_batch)?;
        None
    };
    let family = |e: &interchange::Experiment| match &e.patch {
        None => "clean",
        Some(interchange::Patch::Ops { family: interchange::Family::Swap, .. }) => "swap",
        Some(interchange::Patch::Ops { family: interchange::Family::Zero, .. }) => "zero",
        Some(interchange::Patch::Ops { family: interchange::Family::Scale, .. }) => "scale",
        Some(interchange::Patch::Ops { family: interchange::Family::Push, .. }) => "push",
        Some(interchange::Patch::Ops { family: interchange::Family::Cut, .. }) => "cut",
        Some(_) => "read",
    };
    // Per family: every scored token's bits, the edited tokens' bits, and the experiments.
    let mut scores: BTreeMap<&str, (Vec<f64>, Vec<f64>, usize)> = BTreeMap::new();
    let mut batches = Vec::new();
    for (b, chunk) in held_out[first..end].chunks(settings.batch_sequences).enumerate() {
        let batch = interchange::Batch::new(chunk.to_vec(), chunk.to_vec())?;
        // Each base's donor is the next held-out sequence of its batch.
        let donors: Vec<usize> = (0..chunk.len()).map(|n| (n + 1) % chunk.len()).collect();
        let drawn = match &loaded {
            Some(m) => m.experiments.get(b).cloned().ok_or_else(|| format!("edits: the manifest has no batch {b}"))?,
            None => experiments.sample_ops(&mut rng, &batch, &settings.families, settings.edits_per_sequence, &donors, false)?,
        };
        let scored = experiments.evaluate(&batch, &drawn, false)?;
        for (e, bits) in drawn.iter().zip(&scored.bits) {
            let entry = scores.entry(family(e)).or_default();
            entry.0.extend_from_slice(bits);
            entry.1.extend(bits.first());
            entry.2 += 1;
        }
        batches.push((b, batch, drawn, scored.bits));
        let w = scored.work;
        log::info!(
            "edits: batch {b} scored ({:.0} s): {} paths in {} lanes ({} suffix lanes), {} block rows run of {} whole",
            started.elapsed().as_secs_f64(),
            w.paths,
            w.lanes,
            w.suffix_lanes,
            w.rows,
            w.whole_rows
        );
    }
    let manifest_weights = match &loaded {
        Some(m) => m.weights.clone(),
        None => {
            let m = Manifest {
                export: identity.export.clone(),
                sequences: settings.sequences,
                rows: rows.clone(),
                seed: settings.seed,
                families: settings.families.clone(),
                edits_per_sequence: settings.edits_per_sequence,
                batch_sequences: settings.batch_sequences,
                binary: option_env!("GAM_BUILD_GIT_SHA").map(String::from),
                directions: directions.clone(),
                typical: experiments.typical_norms().into_iter().collect(),
                experiments: batches.iter().map(|(_, _, drawn, _)| drawn.clone()).collect(),
                weights: match &settings.weights {
                    Some(family) => draw_weights(native, family, settings.seed)?,
                    None => Vec::new(),
                },
            };
            std::fs::write(&manifest_path, serde_json::to_vec(&m).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            m.weights
        }
    };
    let manifest = json!({"file": manifest_path.display().to_string(), "sha256": sha256(&manifest_path)?});
    // The response diagnostic KL(p_e ‖ p̃_e), p̃_e ∝ p_0 q_e / q_0 (p = M, q = P, 0 clean, e edited;
    // Interchange::response): whether P predicts M's change under the edit, apart from P's clean
    // error, over the same tokens.
    let mut responses = Vec::with_capacity(batches.len());
    for (_, batch, drawn, _) in &batches {
        responses.push(experiments.response(batch, drawn)?);
    }
    // The edits' effect on M, KL(M_e ‖ M), over the same tokens: the same experiments with P = M
    // applying no edit.
    // The edit-ignoring baseline: P's clean prediction against M's edited outcome, KL(M_e ‖ P),
    // over the same tokens (the same experiments with P applying no edit).
    experiments.unedited_explanation();
    let mut ignoring: BTreeMap<&str, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (_, batch, drawn, _) in &batches {
        for (e, bits) in drawn.iter().zip(&experiments.evaluate(batch, drawn, false)?.bits) {
            let entry = ignoring.entry(family(e)).or_default();
            entry.0.extend_from_slice(bits);
            entry.1.extend(bits.first());
        }
    }
    let typical = experiments.typical_norms();
    drop(experiments);
    let mut reference = interchange::Interchange::new(device, native, layers, &gam_mpd::artifact::Artifact::native(native)?, &[], explanation.reads.clone(), settings.numeric_bytes, 256)?;
    reference.set_directions(interchange::DIRECTIONS, settings.seed);
    reference.set_typical(typical);
    reference.unedited_explanation();
    // An edit's evidence grows with how much it moves M: per family, the gaps of the edits whose
    // effect at the edited token KL(M_e ‖ M) falls in each bin (bits).
    const BINS: [f64; 3] = [0.01, 0.1, 1.0];
    let mut effects: BTreeMap<&str, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut binned: BTreeMap<(&str, usize), (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut response: BTreeMap<&str, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    // Per experiment (one JSON line each): its held-out sequence, edited position, family,
    // operations, effect at the edited token and gap there.
    let mut records = String::new();
    for ((b, batch, drawn, gaps), moved) in batches.iter().zip(&responses) {
        for (((e, bits), gap), moved) in drawn.iter().zip(&reference.evaluate(batch, drawn, false)?.bits).zip(gaps).zip(moved) {
            let ops: Vec<Value> = match &e.patch {
                Some(interchange::Patch::Ops { ops, .. }) => ops.iter().map(|o| json!({"site": o.site, "operation": format!("{:?}", o.operation), "onward": o.onward})).collect(),
                _ => Vec::new(),
            };
            records.push_str(&json!({
                "sequence": first + b * settings.batch_sequences + e.base, "position": e.position, "family": family(e), "ops": ops,
                "effect_bits_at_edited_token": bits.first(), "gap_bits_at_edited_token": gap.first(), "response_bits_at_edited_token": moved.first(),
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
            entry.3.extend_from_slice(moved);
            let entry = response.entry(family(e)).or_default();
            entry.0.extend_from_slice(moved);
            entry.1.extend(moved.first());
        }
    }
    std::fs::write(out.join(format!("EDITS_{}.experiments.jsonl", settings.name)), records).map_err(|e| e.to_string())?;
    drop(reference);
    let weights = match &settings.weights {
        Some(family) => {
            let rows = &held_out[first..end.min(first + family.sequences)];
            Some(weight_faithfulness(device, (native, layers), (&artifact, &explanation.reads), rows, &manifest_weights, settings)?)
        }
        None => None,
    };
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
        let (mut ignored_all, mut ignored_edited) = ignoring.remove(family).unwrap_or_default();
        let ((ignored_mean, ignored_p99), (ignored_edited_mean, _)) = (summary(&mut ignored_all), summary(&mut ignored_edited));
        let ((effect_mean, effect_p99), (effect_edited_mean, effect_edited_p99)) = (summary(&mut effect_all), summary(&mut effect_edited));
        let (mut moved_all, mut moved_edited) = response.remove(family).unwrap_or_default();
        let ((moved_mean, moved_p99), (moved_edited_mean, _)) = (summary(&mut moved_all), summary(&mut moved_edited));
        let bins: Vec<Value> = (0..=BINS.len())
            .filter_map(|bin| {
                let (all, at, effect, moved) = binned.get(&(family, bin))?;
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
                let low = if bin == 0 { 0.0 } else { BINS[bin - 1] };
                let high = BINS.get(bin).copied().unwrap_or(f64::INFINITY);
                Some(json!({"effect_bits_at_edited_token": [low, if high.is_finite() { json!(high) } else { json!("inf") }], "experiments": effect.len(), "mean_bits_per_token": mean(all), "edited_token_mean_bits": mean(at), "effect_edited_token_mean_bits": mean(effect), "response_mean_bits_per_token": mean(moved)}))
            })
            .collect();
        families.insert(
            family.into(),
            json!({
                "by_effect": bins,
                "experiments": count, "tokens": tokens,
                "mean_bits_per_token": mean, "p99_bits_per_token": p99, "edited_token_mean_bits": edited_mean, "edited_token_p99_bits": edited_p99,
                "effect_mean_bits_per_token": effect_mean, "effect_p99_bits_per_token": effect_p99, "effect_edited_token_mean_bits": effect_edited_mean, "effect_edited_token_p99_bits": effect_edited_p99,
                "ignoring_mean_bits_per_token": ignored_mean, "ignoring_p99_bits_per_token": ignored_p99, "ignoring_edited_token_mean_bits": ignored_edited_mean,
                "response_mean_bits_per_token": moved_mean, "response_p99_bits_per_token": moved_p99, "response_edited_token_mean_bits": moved_edited_mean,
            }),
        );
    }
    let report = json!({
        "checkpoint": settings.checkpoint,
        "parts": count,
        "sequences": settings.sequences,
        "edits_per_sequence": settings.edits_per_sequence,
        "seed": settings.seed,
        "manifest": manifest,
        "weights": weights,
        "device": device.name(),
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "families": families,
        "seconds": started.elapsed().as_secs_f64(),
    });
    log::info!("edits: {report}");
    save(&out.join(format!("EDITS_{}.json", settings.name)), &report)
}

/// Native weight edits of `M`'s maps (`EditSettings::weights`), the same for every explanation,
/// drawn once into the manifest (`draw_weights`): per edit one map of a layer (an operator of `M`
/// whose name starts with `blocks.`, neither a vector nor a norm's diagonal; a map stored per head
/// is one tensor, of which one head's block is drawn), tensors uniformly, and either
/// `k = 2^u` of its rows or of its columns (`u` uniform in `0..=4`) scaled by a factor of `SCALES`,
/// or a rank-one push `s ‖W‖_F / √min(r, c) · u vᵀ` with `u`, `v` seeded unit directions and `s`
/// of `SIZES`. Each is compiled into `M` and into `P` (`weight_edit::compile`: `M` computes with
/// `W + ΔW`, `P` with its decoded `W` plus `ΔW` through its owners) and scored on `rows` with `P`
/// autonomous: the gap `KL(M_e ‖ P_e)`, the effect `KL(M_e ‖ M)`, the edit-ignoring baseline
/// `KL(M_e ‖ P)` and the response diagnostic, in bits per token, with the share of the edit `P` took
/// (`Compiled::owned`); an explanation holding no copy of the edited map counts it as not
/// applicable.
fn weight_faithfulness(
    device: &Device,
    (native, layers): (&OperatorProgram, &[LayerNodes]),
    (artifact, reads): (&gam_mpd::artifact::Artifact, &[interchange::ReadVariable]),
    rows: &[Vec<u32>],
    draws: &[WeightDraw],
    settings: &EditSettings,
) -> Result<Value, String> {
    let started = Instant::now();
    if rows.is_empty() {
        return Err("edits: no rows to score the weight edits on".into());
    }
    let batch = interchange::Batch::new(rows.to_vec(), rows.to_vec())?;
    let clean: Vec<interchange::Experiment> = (0..rows.len()).map(|base| interchange::Experiment { base, source: base, explained: vec![true; 2 * layers.len()], patch: None, position: 0 }).collect();
    let interchange = |model: &OperatorProgram, explanation: &gam_mpd::artifact::Artifact| interchange::Interchange::new(device, model, layers, explanation, &[], reads.to_vec(), settings.numeric_bytes, 256);
    let unedited = interchange(native, artifact)?;
    let mean = |bits: &[Vec<f64>]| bits.iter().flatten().sum::<f64>() / bits.iter().map(Vec::len).sum::<usize>().max(1) as f64;
    let mut records = Vec::new();
    for (i, draw) in draws.iter().enumerate() {
        let (operator, kind) = (draw.operator(), draw.kind());
        let edit = gam_mpd::weight_edit::WeightEdit { native: operator.to_string(), delta: draw.delta(native)? };
        let Some(compiled) = gam_mpd::weight_edit::compile(native, artifact, &[edit])? else {
            records.push(json!({"operator": operator, "kind": kind, "applicable": false}));
            continue;
        };
        let edited = interchange(&compiled.model, &compiled.explanation)?;
        let gap = edited.evaluate(&batch, &clean, false)?.bits;
        let response = edited.response_from(&unedited, &batch, &clean)?;
        drop(edited);
        let ignoring = interchange(&compiled.model, artifact)?.evaluate(&batch, &clean, false)?.bits;
        let effect = interchange(&compiled.model, &gam_mpd::artifact::Artifact::native(native)?)?.evaluate(&batch, &clean, false)?.bits;
        let record = json!({
            "operator": operator, "kind": kind, "applicable": true, "owned": compiled.owned,
            "mean_bits_per_token": mean(&gap), "effect_mean_bits_per_token": mean(&effect),
            "ignoring_mean_bits_per_token": mean(&ignoring), "response_mean_bits_per_token": mean(&response),
        });
        log::info!("edits: weight edit {i} ({:.0} s): {record}", started.elapsed().as_secs_f64());
        records.push(record);
    }
    let applicable: Vec<&Value> = records.iter().filter(|r| r["applicable"] == json!(true)).collect();
    let average = |key: &str| applicable.iter().filter_map(|r| r[key].as_f64()).sum::<f64>() / applicable.len().max(1) as f64;
    Ok(json!({
        "edits": records.len(), "not_applicable": records.len() - applicable.len(), "sequences": rows.len(),
        "mean_bits_per_token": average("mean_bits_per_token"), "effect_mean_bits_per_token": average("effect_mean_bits_per_token"),
        "ignoring_mean_bits_per_token": average("ignoring_mean_bits_per_token"), "response_mean_bits_per_token": average("response_mean_bits_per_token"),
        "records": records,
    }))
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
    let explanation = match (&settings.transcoders, &settings.vpd) {
        (Some(_), Some(_)) => return Err("transcoders and vpd are two different starts".into()),
        (Some(transcoders), None) => {
            let files = transcoder_files(&device, &native, &layers, transcoders, &train, settings.fit.batch_sequences, out)?;
            library_mdl::explanation_with(&native, &layers, &files)?
        }
        (None, Some(vpd)) => gam_mpd::library_vpd::explanation(&native, &layers, &vpd.decomposition, &vpd.start, &vpd.arm)?,
        (None, None) => library_mdl::explanation(&native, &layers)?,
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
