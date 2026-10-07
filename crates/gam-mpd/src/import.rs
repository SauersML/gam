//! Import a language-model export (`export.json` plus raw little-endian float64 `<name>.f64`
//! tensors, the layout `bench/mpd_engine_export_hf_2951.py` and `bench/vpd_2951/vpd_engine_export.py`
//! write) or a Hugging Face checkpoint into an operator program ([`import_language_model`],
//! [`hugging_face_language_model`]). Every tensor enters as its own operator at the finest lattice
//! that holds it exactly; tied uses reference one operator.

use crate::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance,
    Rotary, Scale, SequenceLayout, Slot, SlotValues, exact_precision,
};
use crate::safetensors::SafetensorsFile;
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, Axis, s};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

/// An imported model: its native program, the token rows it is read on, and the export record.
pub struct Imported {
    pub program: OperatorProgram,
    pub family: FamilyInputs,
    pub record: Value,
}

/// Raw little-endian float64 values as rows of `cols`, as many rows as the file holds.
pub fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if cols == 0 || bytes.len() % (cols * 8) != 0 {
        return Err(format!("{}: {} bytes are not rows of {cols} float64", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    Array2::from_shape_vec((bytes.len() / (cols * 8), cols), values).map_err(|e| e.to_string())
}

/// [`read_f64`] of a file that must hold exactly `rows × cols` values.
pub fn read_f64_shaped(path: &Path, rows: usize, cols: usize) -> Result<Array2<f64>, String> {
    let matrix = read_f64(path, cols)?;
    if matrix.nrows() != rows {
        return Err(format!("{}: {} rows, expected {rows}", path.display(), matrix.nrows()));
    }
    Ok(matrix)
}

fn shape_of(value: &Value) -> Result<(usize, usize), String> {
    let dims: Vec<usize> = value
        .as_array()
        .ok_or("a shape is not a list")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or("a shape entry is not an integer"))
        .collect::<Result<_, _>>()?;
    match dims[..] {
        [n] => Ok((1, n)),
        [r, c] => Ok((r, c)),
        _ => Err(format!("shape {dims:?} is not one or two axes")),
    }
}

/// Where a model's tensors come from, by their export names: an export directory (`<name>.f64`
/// files listed in `export.json`), or a Hugging Face checkpoint read under those names
/// ([`hugging_face_name`]), in one safetensors file or sharded over several.
enum Tensors<'a> {
    Export { dir: &'a Path, record: &'a Value },
    HuggingFace { files: Vec<SafetensorsFile> },
}

/// The file of `files` holding tensor `stored`.
fn holding<'a>(files: &'a [SafetensorsFile], stored: &str) -> Option<&'a SafetensorsFile> {
    files.iter().find(|f| f.tensors().contains_key(stored))
}

impl Tensors<'_> {
    fn has(&self, name: &str) -> bool {
        match self {
            Self::Export { record, .. } => record["files"].get(name).is_some(),
            Self::HuggingFace { files } => hugging_face_name(name).is_some_and(|n| holding(files, &n).is_some()),
        }
    }

    /// A stored matrix, or a stored vector as one row.
    fn get(&self, name: &str) -> Result<Array2<f64>, String> {
        match self {
            Self::Export { dir, record } => {
                let (rows, cols) = shape_of(&record["files"][name]["shape"]).map_err(|e| format!("{name}: {e}"))?;
                read_f64_shaped(&dir.join(format!("{name}.f64")), rows, cols)
            }
            Self::HuggingFace { files } => {
                let stored = hugging_face_name(name).ok_or_else(|| format!("{name}: no Hugging Face tensor"))?;
                let file = holding(files, &stored).ok_or_else(|| format!("{stored}: missing"))?;
                let shape = file.tensors().get(&stored).map(|e| e.shape.clone()).ok_or_else(|| format!("{stored}: missing"))?;
                match shape[..] {
                    [n] => Ok(file.vector(&stored, n).map_err(|e| e.to_string())?.insert_axis(Axis(0))),
                    [rows, cols] => {
                        let mut governed = file.matrix(MemoryGovernor::global(), &stored, rows, cols).map_err(|e| e.to_string())?;
                        Ok(std::mem::take(&mut *governed))
                    }
                    _ => Err(format!("{stored}: shape {shape:?} is not one or two axes")),
                }
            }
        }
    }

    /// A Hugging Face checkpoint's two-axis tensor where the file stores it, its reals not read
    /// ([`Stored`](crate::safetensors::Stored)); none for an export, whose reals are read.
    fn stored(&self, name: &str) -> Result<Option<crate::safetensors::Stored>, String> {
        let Self::HuggingFace { files } = self else { return Ok(None) };
        let stored = hugging_face_name(name).ok_or_else(|| format!("{name}: no Hugging Face tensor"))?;
        let file = holding(files, &stored).ok_or_else(|| format!("{stored}: missing"))?;
        let shape = file.tensors().get(&stored).map(|e| e.shape.clone()).ok_or_else(|| format!("{stored}: missing"))?;
        let [rows, cols] = shape[..] else { return Err(format!("{stored}: shape {shape:?} is not two axes")) };
        file.stored(&stored, rows, cols).map(Some).map_err(|e| e.to_string())
    }

    /// The operator of the two-axis tensor `name` between `rows` and `cols`: where a checkpoint
    /// stores it, or read from an export.
    fn operator(&self, b: &mut Builder, label: &str, rows: &Interface, cols: &Interface, name: &str) -> Result<usize, String> {
        match self.stored(name)? {
            Some(stored) => b.stored(label, rows, cols, stored),
            None => b.operator(label, rows, cols, self.get(name)?),
        }
    }

    /// The rows of the two-axis tensor `name`, without reading its reals.
    fn rows_of(&self, name: &str) -> Result<usize, String> {
        match self.stored(name)? {
            Some(stored) => Ok(stored.dim().0),
            None => Ok(self.get(name)?.nrows()),
        }
    }

    /// A stored row vector as a column.
    fn column(&self, name: &str) -> Result<Array2<f64>, String> {
        Ok(self.get(name)?.t().to_owned())
    }
}

fn config(record: &Value, key: &str) -> Result<usize, String> {
    record["config"][key].as_u64().map(|v| v as usize).ok_or_else(|| format!("config.{key}"))
}

/// Builds a program node by node.
struct Builder {
    operators: Vec<Arc<Operator>>,
    nodes: Vec<Node>,
}

impl Builder {
    fn operator(&mut self, name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Result<usize, String> {
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        let op = Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name))
            .map_err(|e| e.to_string())?;
        self.operators.push(Arc::new(op));
        Ok(self.operators.len() - 1)
    }

    /// The operator `name` whose reals stay where `stored` holds them.
    fn stored(&mut self, name: &str, rows: &Interface, cols: &Interface, stored: crate::safetensors::Stored) -> Result<usize, String> {
        let op = Operator::stored(name, rows.clone(), cols.clone(), stored, Provenance::native(name)).map_err(|e| e.to_string())?;
        self.operators.push(Arc::new(op));
        Ok(self.operators.len() - 1)
    }

    /// A norm gain: the diagonal operator on `interface` holding the stored row `name`.
    fn gain(&mut self, name: &str, interface: &Interface, values: Array1<f64>) -> Result<usize, String> {
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        let op = Operator::diag(name, interface.clone(), values, precision, Provenance::native(name)).map_err(|e| e.to_string())?;
        self.operators.push(Arc::new(op));
        Ok(self.operators.len() - 1)
    }

    fn node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }
}

fn interface(count: usize, width: usize, kind: LabelKind) -> Result<Interface, String> {
    Interface::uniform(count, width, kind, 0).map_err(|e| e.to_string())
}

/// A pre-norm rotary language model export as a per-position program over the first `sequences`
/// token rows at positions `0..context`: every row runs one shared local program, and attention
/// reads the rows of its own sequence up to its position. The readout at each row is the
/// next-token distribution.
///
/// `config`: d_model, n_layers, n_heads, n_kv_heads, head_dim, vocab, rope_theta, rope_pairing,
/// norm_eps, mlp_act, tied_embeddings, and optionally `rotary_dims` (a partial rotary on the first
/// dims of each head; default head_dim), `norm` (`"rms"`, the default, or `"layer"`: the mean is
/// removed first, `x − 1 (1ᵀx)/d`, a rank-one operator shared by every norm), `parallel_residual`
/// (attention and MLP read the same stream, GPT-NeoX), `qk_norm` (an RMS norm with a gain on each
/// head's query and key before the rotary, Qwen3) and `mlp_gated` (`down(act(gate x) ⊙ up x)`).
/// Tensors: `wte` (vocab × d), `lm_head` when untied, `blocks.{l}.attn.{q,k,v,o}_proj`,
/// `blocks.{l}.attn.{q,k}_norm.gain`, `blocks.{l}.mlp.{c_fc,gate_proj,down_proj}`,
/// `blocks.{l}.rms{1,2}.gain`, `final_norm.gain`; any of these with a `.bias` suffix (a
/// projection's or a layer norm's bias) enters as a bias operator; and the token rows `tokens`.
pub fn import_language_model(dir: &Path, sequences: usize, context: usize) -> Result<Imported, String> {
    let text = std::fs::read_to_string(dir.join("export.json")).map_err(|e| e.to_string())?;
    let record: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let layers = config(&record, "n_layers")?;
    let program = language_model(&Tensors::Export { dir, record: &record }, &record, 0..layers)?;
    let (token_rows, token_cols) = shape_of(&record["files"]["tokens"]["shape"])?;
    if sequences > token_rows || context > token_cols {
        return Err(format!("{sequences} sequences of {context} tokens from a {token_rows}×{token_cols} table"));
    }
    let tokens = read_f64_shaped(&dir.join("tokens.f64"), token_rows, token_cols)?;
    let mut ids = Vec::new();
    let (mut sequence, mut position) = (Vec::new(), Vec::new());
    for row in 0..sequences {
        for pos in 0..context {
            ids.push(tokens[[row, pos]] as u32);
            sequence.push(row as u32);
            position.push(pos as u32);
        }
    }
    let rows = ids.len();
    let family = FamilyInputs {
        rows,
        slots: vec![SlotValues::Tokens(ids)],
        layout: Some(SequenceLayout { sequence, position }),
    };
    Ok(Imported { program, family, record })
}

/// The Hugging Face Qwen2, Qwen3 or Llama tensor stored under export name `name` (the names
/// `bench/mpd_engine_export_hf_2951.py` writes), or `None` for a name such a checkpoint never holds.
fn hugging_face_name(name: &str) -> Option<String> {
    let fixed = match name {
        "wte" => Some("model.embed_tokens.weight"),
        "lm_head" => Some("lm_head.weight"),
        "final_norm.gain" => Some("model.norm.weight"),
        _ => None,
    };
    if let Some(fixed) = fixed {
        return Some(fixed.to_string());
    }
    let rest = name.strip_prefix("blocks.")?;
    let (layer, part) = rest.split_once('.')?;
    let (part, suffix) = match part.strip_suffix(".bias") {
        Some(stem) => (stem, "bias"),
        None => (part, "weight"),
    };
    let stored = match part {
        "attn.q_proj" | "attn.k_proj" | "attn.v_proj" | "attn.o_proj" => format!("self_attn.{}", &part[5..]),
        "mlp.c_fc" => "mlp.up_proj".to_string(),
        "mlp.gate_proj" | "mlp.down_proj" => part.to_string(),
        "attn.q_norm.gain" | "attn.k_norm.gain" if suffix == "weight" => format!("self_attn.{}", &part[5..11]),
        "rms1.gain" if suffix == "weight" => "input_layernorm".to_string(),
        "rms2.gain" if suffix == "weight" => "post_attention_layernorm".to_string(),
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{stored}.{suffix}"))
}

/// A Hugging Face checkpoint directory (`config.json` and `model.safetensors`) of a Qwen2 (q, k, v
/// biases), Qwen3 (`qk_norm`) or Llama model as the program of [`import_language_model`], read
/// straight from the stored tensors (each widened exactly), over `blocks`. Past block 0 the
/// program's one slot is the residual stream entering `blocks.start` (raw, `d` per row, positions
/// from the inputs' layout), so the blocks before it can run elsewhere and enter as their output;
/// short of the last block its output is the residual stream entering `blocks.end` (no final norm,
/// no readout), for the blocks after it to run elsewhere. Only the blocks' tensors (and the
/// embedding, when the program reads tokens or the tied readout) are read. Returns the program and
/// its `{config}` record (the export's conventions).
pub fn hugging_face_language_model(dir: &Path, blocks: std::ops::Range<usize>) -> Result<(OperatorProgram, Value), String> {
    hugging_face(dir, blocks, None)
}

/// The Hugging Face checkpoint at `dir` cut to its first `layers` blocks, then its final norm and
/// its head: the model's embedding, those blocks and its readout, the later blocks left out (a
/// smaller language model on the same tokens, for measuring per-layer costs where the whole model
/// does not fit). Its `{config}` record gives `n_layers = layers`.
pub fn hugging_face_language_model_prefix(dir: &Path, layers: usize) -> Result<(OperatorProgram, Value), String> {
    hugging_face(dir, 0..layers, Some(layers))
}

/// [`hugging_face_language_model`] over `blocks`, or with `prefix` its first `prefix` blocks read
/// out ([`hugging_face_language_model_prefix`]).
fn hugging_face(dir: &Path, blocks: std::ops::Range<usize>, prefix: Option<usize>) -> Result<(OperatorProgram, Value), String> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let hf: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    refuse_unsupported(&hf).map_err(|e| format!("config.json: {e}"))?;
    // A sharded checkpoint names its files in its index's weight map.
    let index = dir.join("model.safetensors.index.json");
    let names: Vec<String> = if index.exists() {
        let text = std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
        let map: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", index.display()))?;
        let shards = map["weight_map"].as_object().ok_or_else(|| format!("{}: no weight_map", index.display()))?;
        let mut names = shards
            .values()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| format!("{}: a weight_map entry is not a file name", index.display())))
            .collect::<Result<Vec<_>, _>>()?;
        names.sort_unstable();
        names.dedup();
        names
    } else {
        vec!["model.safetensors".to_string()]
    };
    let files = names.iter().map(|name| SafetensorsFile::open(&dir.join(name)).map_err(|e| format!("{name}: {e}"))).collect::<Result<Vec<_>, String>>()?;
    let integer = |key: &str| hf[key].as_u64().ok_or_else(|| format!("config.json: {key}"));
    let kind = hf["model_type"].as_str().ok_or("config.json: model_type")?;
    if !matches!(kind, "qwen2" | "qwen3" | "llama") {
        return Err(format!("unsupported model_type {kind}"));
    }
    let (d, heads) = (integer("hidden_size")?, integer("num_attention_heads")?);
    let head_dim = hf["head_dim"].as_u64().unwrap_or(d / heads.max(1));
    let vocab = holding(&files, "model.embed_tokens.weight")
        .and_then(|f| f.tensors().get("model.embed_tokens.weight"))
        .map(|e| e.shape[0])
        .ok_or("model.embed_tokens.weight: missing")?;
    let act = match hf["hidden_act"].as_str() {
        Some("silu") => "silu",
        other => return Err(format!("unsupported hidden_act {other:?}")),
    };
    let all = integer("num_hidden_layers")?;
    if prefix.is_some_and(|k| k == 0 || k as u64 > all) {
        return Err(format!("a prefix of {prefix:?} blocks of a {all}-block model"));
    }
    let record = serde_json::json!({
        "source": {"model": dir.display().to_string(), "blocks": [blocks.start, blocks.end], "prefix_of": prefix.map(|_| all)},
        "config": {
            "norm": "rms", "parallel_residual": false, "rotary_dims": head_dim,
            "rope_theta": hf["rope_theta"].as_f64().ok_or("config.json: rope_theta")?,
            "norm_eps": hf["rms_norm_eps"].as_f64().ok_or("config.json: rms_norm_eps")?, "mlp_act": act,
            "tied_embeddings": hf["tie_word_embeddings"].as_bool().unwrap_or(false), "mlp_gated": true,
                        "qk_norm": kind == "qwen3", "d_model": d, "n_layers": prefix.map_or(all, |k| k as u64), "n_heads": heads,
            "n_kv_heads": integer("num_key_value_heads")?, "head_dim": head_dim,
            "d_mlp": integer("intermediate_size")?, "vocab": vocab, "rope_pairing": "rotate_half",
        },
    });
    let program = language_model(&Tensors::HuggingFace { files }, &record, blocks)?;
    Ok((program, record))
}

/// Refuse a configuration option the program does not compute: a rope scaling (any `rope_type`
/// other than the default rotary), a partial rotary factor other than 1, a sliding attention window
/// (`use_sliding_window`, a `sliding_window` without it, or a sliding layer in `layer_types`).
/// The program is the full causal attention with the plain rotary, so such a model is refused
/// rather than imported as a different function.
fn refuse_unsupported(config: &Value) -> Result<(), String> {
    let set = |key: &str| config.get(key).is_some_and(|v| !v.is_null());
    if set("rope_scaling") {
        let rope_type = config["rope_scaling"]["rope_type"].as_str().or(config["rope_scaling"]["type"].as_str());
        if rope_type != Some("default") {
            return Err(format!("rope_scaling {} is not supported", config["rope_scaling"]));
        }
    }
    if config["partial_rotary_factor"].as_f64().is_some_and(|f| f != 1.0) {
        return Err(format!("partial_rotary_factor {} is not supported", config["partial_rotary_factor"]));
    }
    let sliding = match config.get("use_sliding_window").and_then(Value::as_bool) {
        Some(used) => used,
        None => set("sliding_window"),
    };
    let sliding_layers = config["layer_types"].as_array().is_some_and(|types| types.iter().any(|t| t.as_str() != Some("full_attention")));
    if sliding || sliding_layers {
        return Err("a sliding attention window is not supported".to_string());
    }
    Ok(())
}

/// The program of [`import_language_model`] over `blocks` (module note): its input is the tokens
/// at block 0, else the residual stream entering `blocks.start`, raw; its output is the readout
/// after the last block, else the residual stream entering `blocks.end`.
fn language_model(tensors: &Tensors<'_>, record: &Value, blocks: std::ops::Range<usize>) -> Result<OperatorProgram, String> {
    let (layers, heads, kv_heads, hd, d, vocab) = (
        config(record, "n_layers")?,
        config(record, "n_heads")?,
        config(record, "n_kv_heads")?,
        config(record, "head_dim")?,
        config(record, "d_model")?,
        config(record, "vocab")?,
    );
    refuse_unsupported(&record["config"])?;
    let flag = |key: &str| record["config"][key].as_bool().unwrap_or(false);
    let (parallel, qk_norm, gated) = (flag("parallel_residual"), flag("qk_norm"), flag("mlp_gated"));
    // `none`: no norm anywhere (a real-valued toy's residual MLP, `bench/toys_2951`).
    let (layer_norm, no_norm) = match record["config"]["norm"].as_str() {
        None | Some("rms") => (false, false),
        Some("layer") => (true, false),
        Some("none") => (false, true),
        Some(other) => return Err(format!("unsupported norm {other}")),
    };
    // A Gaussian head (`fixed_head_target::GAUSSIAN_HEAD`): the outputs `law(E h + b)` of the
    // last stream, `E` and `b` the tensors `gaussian_head` and `gaussian_head.bias` (already in
    // units of the model's task residual), `law` ReLU when `head.relu`, else the identity.
    let gaussian = match record["config"]["head"]["law"].as_str() {
        None | Some("softmax") => None,
        Some("gaussian") => Some(record["config"]["head"]["relu"].as_bool().unwrap_or(false)),
        Some(other) => return Err(format!("unsupported head law {other}")),
    };
    let tied = record["config"]["tied_embeddings"].as_bool().unwrap_or(true);
    let rotary_dims = record["config"]["rotary_dims"].as_u64().map_or(hd, |v| v as usize);
    let theta = record["config"]["rope_theta"].as_f64().ok_or("config.rope_theta")?;
    let epsilon = record["config"]["norm_eps"].as_f64().ok_or("config.norm_eps")?;
    let half_split = record["config"]["rope_pairing"].as_str() == Some("rotate_half");
    let act = match record["config"]["mlp_act"].as_str() {
        Some("gelu_tanh") => Law::GeluTanh,
        Some("gelu") => Law::Gelu,
        Some("silu") => Law::Silu,
        Some("relu") => Law::Relu,
        Some("identity") => Law::Identity,
        other => return Err(format!("unsupported mlp activation {other:?}")),
    };
    if theta.fract() != 0.0 || theta <= 0.0 || theta > f64::from(u32::MAX) {
        return Err(format!("rope_theta {theta} is not a positive integer"));
    }
    let rotary = Rotary { base: theta as u32, dims: rotary_dims as u32, half_split };
    // The residual stream is in coordinate groups, so a norm gain is a diagonal of present blocks.
    let model = interface(d, 1, LabelKind::Unit)?;
    let head = Interface::native(hd).map_err(|e| e.to_string())?;
    let constant = Interface::constant();
    let tokens_interface = interface(vocab, 1, LabelKind::Token)?;
    let coordinates = model.clone();
    let mut b = Builder { operators: Vec::new(), nodes: Vec::new() };
    b.operators.push(Arc::new(Operator::identity("I", model.clone())));
    let identity = 0;
    if blocks.start > blocks.end || blocks.end > layers {
        return Err(format!("blocks {blocks:?} of {layers}"));
    }
    let reads_out = blocks.end == layers;
    let embedding = if blocks.start == 0 || (reads_out && tied) {
        Some(b.operator("wte", &model, &tokens_interface, tensors.get("wte")?.t().to_owned())?)
    } else {
        None
    };
    // A norm gain is a diagonal operator: d reals, held as the diagonal.
    let gain = |b: &mut Builder, name: &str| -> Result<usize, String> {
        let g = tensors.get(name)?;
        if g.dim() != (1, d) {
            return Err(format!("{name}: shape {:?}, not a row of {d}", g.dim()));
        }
        b.gain(name, &coordinates, g.row(0).to_owned())
    };
    // A stored bias (a row vector, sliced to `range`) as a column operator on `rows`.
    let bias = |b: &mut Builder, name: &str, rows: &Interface, range: Option<(usize, usize)>| -> Result<Option<usize>, String> {
        if !tensors.has(name) {
            return Ok(None);
        }
        let column = tensors.column(name)?;
        let column = match range {
            Some((first, last)) => column.slice(s![first..last, ..]).to_owned(),
            None => column,
        };
        let label = range.map_or(name.to_string(), |(first, _)| format!("{name}[{}]", first / hd));
        Ok(Some(b.operator(&label, rows, &constant, column)?))
    };
    // A layer norm removes the mean first: `x − 1 (1ᵀx)/d`, one rank-one operator for every norm.
    let centring = if layer_norm {
        let precision = exact_precision([-1.0 / d as f64, 1.0]).map_err(|e| e.to_string())?;
        let op = Operator::low_rank(
            "centre",
            model.clone(),
            model.clone(),
            Array2::from_elem((d, 1), -1.0 / d as f64),
            Array2::from_elem((1, d), 1.0),
            precision,
            Provenance::native("layer norm mean"),
        )
        .map_err(|e| e.to_string())?;
        b.operators.push(Arc::new(op));
        Some(b.operators.len() - 1)
    } else {
        None
    };
    let norm = |b: &mut Builder, x: usize, name: &str| -> Result<usize, String> {
        if no_norm {
            return Ok(x);
        }
        let input = match centring {
            Some(centre) => b.node(Node::Affine { terms: vec![(x, identity), (x, centre)], bias: None }),
            None => x,
        };
        let g = gain(b, &format!("{name}.gain"))?;
        let beta = bias(b, &format!("{name}.bias"), &model, None)?;
        let normed = b.node(Node::RmsNorm { input, epsilon });
        Ok(b.node(Node::Affine { terms: vec![(normed, g)], bias: beta }))
    };
    // A head norm (Qwen3's q_norm, k_norm): an RMS norm over the head, then its gain, a diagonal
    // operator on the head.
    let head_norm = |b: &mut Builder, x: usize, name: &str| -> Result<usize, String> {
        let g = tensors.get(name)?;
        if g.dim() != (1, hd) {
            return Err(format!("{name}: shape {:?}, not a row of {hd}", g.dim()));
        }
        let op = b.gain(name, &head, g.row(0).to_owned())?;
        let normed = b.node(Node::RmsNorm { input: x, epsilon });
        Ok(b.node(Node::Affine { terms: vec![(normed, op)], bias: None }))
    };
    let (input_slot, mut x) = if let (0, Some(embedding)) = (blocks.start, embedding) {
        let feature = b.node(Node::Feature { slot: 0, basis: 0 });
        (Slot::Token { domain: 0 }, b.node(Node::Affine { terms: vec![(feature, embedding)], bias: None }))
    } else {
        // The raw stream enters the coordinate groups through the identity.
        let raw = b.node(Node::Raw { slot: 0 });
        let lift = b.operator("residual input", &model, &Interface::native(d).map_err(|e| e.to_string())?, Array2::eye(d))?;
        (Slot::Raw { width: d }, b.node(Node::Affine { terms: vec![(raw, lift)], bias: None }))
    };
    for l in blocks {
        let prefix = format!("blocks.{l}.");
        let h = norm(&mut b, x, &format!("{prefix}rms1"))?;
        // The query, key and value maps' heads are row blocks: where a checkpoint stores the map,
        // each head reads its rows there; an export's map is read. The output map's heads are
        // column blocks, read.
        let (q_name, k_name, v_name) = (format!("{prefix}attn.q_proj"), format!("{prefix}attn.k_proj"), format!("{prefix}attn.v_proj"));
        let (sq, sk, sv) = (tensors.stored(&q_name)?, tensors.stored(&k_name)?, tensors.stored(&v_name)?);
        let read = |name: &str, stored: &Option<crate::safetensors::Stored>| if stored.is_some() { Ok(None) } else { tensors.get(name).map(Some) };
        let (wq, wk, wv, wo) = (read(&q_name, &sq)?, read(&k_name, &sk)?, read(&v_name, &sv)?, tensors.get(&format!("{prefix}attn.o_proj"))?);
        let rows_of = |b: &mut Builder, label: &str, stored: &Option<crate::safetensors::Stored>, read: &Option<Array2<f64>>, (from, to): (usize, usize)| match (stored, read) {
            (Some(stored), _) => b.stored(label, &head, &model, stored.rows(from..to).ok_or_else(|| format!("{label}: rows {from}..{to} of {:?}", stored.dim()))?),
            (None, Some(values)) => b.operator(label, &head, &model, values.slice(s![from..to, ..]).to_owned()),
            (None, None) => Err(format!("{label}: a map neither stored nor read")),
        };
        let group = heads / kv_heads.max(1);
        let mut keys = Vec::new();
        for g in 0..kv_heads {
            let range = (g * hd, (g + 1) * hd);
            let k_op = rows_of(&mut b, &format!("{prefix}k{g}"), &sk, &wk, range)?;
            let v_op = rows_of(&mut b, &format!("{prefix}v{g}"), &sv, &wv, range)?;
            let k_bias = bias(&mut b, &format!("{prefix}attn.k_proj.bias"), &head, Some(range))?;
            let v_bias = bias(&mut b, &format!("{prefix}attn.v_proj.bias"), &head, Some(range))?;
            let mut k = b.node(Node::Affine { terms: vec![(h, k_op)], bias: k_bias });
            if qk_norm {
                k = head_norm(&mut b, k, &format!("{prefix}attn.k_norm.gain"))?;
            }
            let v = b.node(Node::Affine { terms: vec![(h, v_op)], bias: v_bias });
            keys.push((k, v));
        }
        let mut terms = vec![(x, identity)];
        for hh in 0..heads {
            let range = (hh * hd, (hh + 1) * hd);
            let q_op = rows_of(&mut b, &format!("{prefix}q{hh}"), &sq, &wq, range)?;
            let o_op = b.operator(&format!("{prefix}o{hh}"), &model, &head, wo.slice(s![.., range.0..range.1]).to_owned())?;
            let q_bias = bias(&mut b, &format!("{prefix}attn.q_proj.bias"), &head, Some(range))?;
            let mut q = b.node(Node::Affine { terms: vec![(h, q_op)], bias: q_bias });
            if qk_norm {
                q = head_norm(&mut b, q, &format!("{prefix}attn.q_norm.gain"))?;
            }
            let (k, v) = keys[hh / group.max(1)];
            let read = b.node(Node::Attend {
                query: q,
                key: k,
                value: v,
                scale: Scale::InverseSqrt(hd as u32),
                rotary: Some(rotary),
                causal: true,
            });
            terms.push((read, o_op));
        }
        let o_bias = bias(&mut b, &format!("{prefix}attn.o_proj.bias"), &model, None)?;
        let attended = b.node(Node::Affine { terms, bias: o_bias });
        // A parallel block's MLP reads the stream the attention read; a sequential one reads the
        // stream after it.
        let h2 = norm(&mut b, if parallel { x } else { attended }, &format!("{prefix}rms2"))?;
        let hidden = tensors.rows_of(&format!("{prefix}mlp.c_fc"))?;
        let neurons = interface(hidden, 1, LabelKind::Unit)?;
        let up = tensors.operator(&mut b, &format!("{prefix}c_fc"), &neurons, &model, &format!("{prefix}mlp.c_fc"))?;
        let up_bias = bias(&mut b, &format!("{prefix}mlp.c_fc.bias"), &neurons, None)?;
        let down = tensors.operator(&mut b, &format!("{prefix}down_proj"), &model, &neurons, &format!("{prefix}mlp.down_proj"))?;
        let down_bias = bias(&mut b, &format!("{prefix}mlp.down_proj.bias"), &model, None)?;
        let pre = b.node(Node::Affine { terms: vec![(h2, up)], bias: up_bias });
        let active = if gated {
            let gate = tensors.operator(&mut b, &format!("{prefix}gate_proj"), &neurons, &model, &format!("{prefix}mlp.gate_proj"))?;
            let gate_bias = bias(&mut b, &format!("{prefix}mlp.gate_proj.bias"), &neurons, None)?;
            let gate_pre = b.node(Node::Affine { terms: vec![(h2, gate)], bias: gate_bias });
            let gate_active = b.node(Node::Pointwise { input: gate_pre, laws: vec![act; hidden] });
            b.node(Node::Hadamard { left: gate_active, right: pre })
        } else {
            b.node(Node::Pointwise { input: pre, laws: vec![act; hidden] })
        };
        x = b.node(Node::Affine { terms: vec![(attended, identity), (active, down)], bias: down_bias });
    }
    let output = if !reads_out {
        x
    } else {
        let h = norm(&mut b, x, "final_norm")?;
        match gaussian {
            Some(relu) => {
                let name = crate::resident_causal_fit::fixed_head_target::GAUSSIAN_HEAD;
                let outputs = interface(tensors.rows_of(name)?, 1, LabelKind::Unit)?;
                let head_op = tensors.operator(&mut b, name, &outputs, &model, name)?;
                let head_bias = bias(&mut b, &format!("{name}.bias"), &outputs, None)?;
                let y = b.node(Node::Affine { terms: vec![(h, head_op)], bias: head_bias });
                match relu {
                    true => b.node(Node::Pointwise { input: y, laws: vec![Law::Relu; outputs.groups().len()] }),
                    false => y,
                }
            }
            None => {
                let logits = match embedding {
                    Some(embedding) if tied => b.node(Node::Transposed { input: h, operator: embedding }),
                    _ => {
                        let head_op = tensors.operator(&mut b, "lm_head", &tokens_interface, &model, "lm_head")?;
                        b.node(Node::Affine { terms: vec![(h, head_op)], bias: None })
                    }
                };
                b.node(Node::Readout { input: logits, basis: 0 })
            }
        }
    };
    let declarations = Declarations { parameters: 0, domains: vec![Domain { size: vocab }], slots: vec![input_slot] };
    Ok(OperatorProgram {
        rules: Vec::new(),
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: b.operators,
        nodes: b.nodes,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::{hugging_face_language_model, hugging_face_language_model_prefix, refuse_unsupported};
    use serde_json::{Value, json};
    use std::path::Path;

    /// A configuration the program does not compute is refused, never imported as another function.
    #[test]
    fn unsupported_rope_scaling_and_sliding_windows_are_refused() {
        assert!(refuse_unsupported(&json!({})).is_ok());
        assert!(refuse_unsupported(&json!({"rope_scaling": null})).is_ok());
        assert!(refuse_unsupported(&json!({"rope_scaling": {"rope_type": "default"}})).is_ok());
        assert!(refuse_unsupported(&json!({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}})).is_err());
        assert!(refuse_unsupported(&json!({"rope_scaling": {"type": "linear", "factor": 2.0}})).is_err());
        assert!(refuse_unsupported(&json!({"partial_rotary_factor": 0.5})).is_err());
        assert!(refuse_unsupported(&json!({"use_sliding_window": true, "sliding_window": 4096})).is_err());
        assert!(refuse_unsupported(&json!({"use_sliding_window": false, "sliding_window": 4096})).is_ok());
        assert!(refuse_unsupported(&json!({"sliding_window": 4096})).is_err());
        assert!(refuse_unsupported(&json!({"layer_types": ["full_attention", "sliding_attention"]})).is_err());
        assert!(refuse_unsupported(&json!({"layer_types": ["full_attention", "full_attention"]})).is_ok());
    }

    /// A checkpoint sharded over several files imports as the same program as its one-file form.
    #[test]
    fn a_sharded_checkpoint_imports_as_its_merged_file() {
        let (d, heads, kv, hidden, vocab) = (8usize, 2usize, 1usize, 12usize, 11usize);
        let shapes: Vec<(&str, Vec<usize>)> = vec![
            ("model.embed_tokens.weight", vec![vocab, d]),
            ("model.layers.0.input_layernorm.weight", vec![d]),
            ("model.layers.0.self_attn.q_proj.weight", vec![heads * 4, d]),
            ("model.layers.0.self_attn.k_proj.weight", vec![kv * 4, d]),
            ("model.layers.0.self_attn.v_proj.weight", vec![kv * 4, d]),
            ("model.layers.0.self_attn.o_proj.weight", vec![d, heads * 4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![d]),
            ("model.layers.0.mlp.gate_proj.weight", vec![hidden, d]),
            ("model.layers.0.mlp.up_proj.weight", vec![hidden, d]),
            ("model.layers.0.mlp.down_proj.weight", vec![d, hidden]),
            ("model.norm.weight", vec![d]),
            ("lm_head.weight", vec![vocab, d]),
        ];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = shapes
            .into_iter()
            .map(|(name, shape)| {
                let values = (0..shape.iter().product::<usize>())
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect();
                (name, shape, values)
            })
            .collect();
        // One safetensors file of `part`, in the stored F32 layout.
        let write = |path: &Path, part: &[(&str, Vec<usize>, Vec<f32>)]| {
            let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
            for (name, shape, values) in part {
                let start = data.len();
                values.iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
                header.insert(name.to_string(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
            }
            let header = Value::Object(header).to_string();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&data);
            std::fs::write(path, bytes).expect("safetensors file");
        };
        let config = json!({
            "model_type": "llama", "hidden_act": "silu", "hidden_size": d, "num_attention_heads": heads, "num_key_value_heads": kv,
            "intermediate_size": hidden, "num_hidden_layers": 1, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
        });
        let base = std::env::temp_dir().join(format!("gam-mpd-sharded-{}", std::process::id()));
        let (whole, sharded) = (base.join("whole"), base.join("sharded"));
        for dir in [&whole, &sharded] {
            std::fs::create_dir_all(dir).expect("temporary directory");
            std::fs::write(dir.join("config.json"), config.to_string()).expect("config");
        }
        write(&whole.join("model.safetensors"), &tensors);
        let (first, second) = tensors.split_at(5);
        write(&sharded.join("model-00001-of-00002.safetensors"), first);
        write(&sharded.join("model-00002-of-00002.safetensors"), second);
        let weight_map: serde_json::Map<String, Value> = tensors
            .iter()
            .enumerate()
            .map(|(i, (name, _, _))| (name.to_string(), json!(if i < 5 { "model-00001-of-00002.safetensors" } else { "model-00002-of-00002.safetensors" })))
            .collect();
        std::fs::write(sharded.join("model.safetensors.index.json"), json!({"metadata": {}, "weight_map": weight_map}).to_string()).expect("index");
        let (a, _) = hugging_face_language_model(&whole, 0..1).expect("one file");
        let (b, _) = hugging_face_language_model(&sharded, 0..1).expect("two shards");
        // The same tensors as the first block of a two-block model: its one-block prefix is the
        // one-block model, head included; without the prefix the first block ends in the stream.
        let longer = base.join("longer");
        std::fs::create_dir_all(&longer).expect("temporary directory");
        let mut deeper = config.clone();
        deeper["num_hidden_layers"] = json!(2);
        std::fs::write(longer.join("config.json"), deeper.to_string()).expect("config");
        write(&longer.join("model.safetensors"), &tensors);
        let (prefix, record) = hugging_face_language_model_prefix(&longer, 1).expect("a one-block prefix");
        let (cut, _) = hugging_face_language_model(&longer, 0..1).expect("the first block alone");
        assert!(hugging_face_language_model_prefix(&longer, 3).is_err(), "no prefix longer than the model");
        std::fs::remove_dir_all(&base).expect("cleanup");
        assert_eq!((record["config"]["n_layers"].as_u64(), record["source"]["prefix_of"].as_u64()), (Some(1), Some(2)));
        assert_eq!(format!("{:?}", prefix.nodes), format!("{:?}", a.nodes), "the prefix is the one-block model");
        assert!(prefix.operators.iter().zip(&a.operators).all(|(x, y)| x.name == y.name && x.matrix() == y.matrix()));
        assert!(crate::resident_causal_fit::fixed_head_target::Head::of(&prefix).is_ok(), "its head is the compact targets' head");
        assert!(crate::resident_causal_fit::fixed_head_target::Head::of(&cut).is_err(), "a cut program has no head");
        assert_eq!(a.operators.len(), b.operators.len());
        for (x, y) in a.operators.iter().zip(&b.operators) {
            assert_eq!(x.name, y.name);
            assert!(x.matrix().iter().zip(y.matrix().iter()).all(|(u, v)| u.to_bits() == v.to_bits()), "{} differs", x.name);
        }
        assert_eq!(format!("{:?}", a.nodes), format!("{:?}", b.nodes));
        // The MLP maps and the head stay where the checkpoint stores them, and a device program
        // made from them runs exactly as one made from their float64 values, on the host and on a
        // single-precision device.
        use crate::operator_program::{Operator, OperatorBody};
        let stored: Vec<&str> = a.operators.iter().filter(|op| matches!(&op.body, OperatorBody::Dense { values, .. } if values.stored().is_some())).map(|op| op.name.as_str()).collect();
        assert_eq!(stored, ["blocks.0.k0", "blocks.0.v0", "blocks.0.q0", "blocks.0.q1", "blocks.0.c_fc", "blocks.0.down_proj", "blocks.0.gate_proj", "lm_head"]);
        let mut held = a.clone();
        for op in &mut held.operators {
            if matches!(&op.body, OperatorBody::Dense { values, .. } if values.stored().is_some()) {
                let OperatorBody::Dense { present, precision, .. } = &op.body else { continue };
                *op = std::sync::Arc::new(Operator::blocks(op.name.clone(), op.rows.clone(), op.cols.clone(), op.matrix(), present.clone(), *precision, op.provenance.clone()).expect("the same reals"));
            }
        }
        assert!(held.operators.iter().all(|op| !matches!(&op.body, OperatorBody::Dense { values, .. } if values.stored().is_some())));
        let family = crate::library_mdl::sequence_family(&[&[1, 4, 2, 7, 3]]).expect("a family");
        let devices = [Some(gam_gpu::tensor::Device::host()), gam_gpu::tensor::Device::single_precision(gam_gpu::GpuPolicy::Auto).ok().flatten()];
        for device in devices.into_iter().flatten() {
            let run = |program: &crate::operator_program::OperatorProgram| {
                let mut compiled = crate::device_program::DeviceProgram::compile_values(&device, program).expect("a device program");
                if device.storage() == gam_gpu::tensor::Storage::F32 {
                    compiled.set_arithmetic(gam_gpu::tensor::Arithmetic::F32);
                }
                let trace = compiled.forward(&family).expect("a forward pass");
                device.download(trace.value(program.output).expect("the output")).expect("its values")
            };
            let (from_file, from_host) = (run(&a), run(&held));
            assert!(from_file.iter().zip(from_host.iter()).all(|(u, v)| u.to_bits() == v.to_bits()), "{}: stored and held reals run differently", device.name());
        }
    }
}
