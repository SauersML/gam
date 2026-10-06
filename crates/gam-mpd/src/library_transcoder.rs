//! Transcoder features as the library's MLP functions (#2951).
//!
//! A single-layer transcoder (circuit-tracer's, e.g. `mwhanna/qwen3-0.6b-transcoders-lowl0`: one
//! safetensors file per layer holding `W_enc` and `W_dec` of `F × d`, `b_enc` of `F` and `b_dec`
//! of `d`) reads the MLP's input `x` (the normed stream, `mlp.hook_in`) and writes
//! `Σ_i relu(g_i·x + c_i) u_i + b` as the MLP's output (`mlp.hook_out`), with `g_i` row `i` of
//! `W_enc`, `c_i = b_enc[i]`, `u_i` row `i` of `W_dec` and `b = b_dec`. Feature `i` becomes the
//! library function `relu(g_i·x + c_i) u_i` in the grammar of a plain MLP's functions
//! (`library_mdl::explanation`): its gate row with its bias, the ReLU law, its output column, one
//! prior group each. `b` is the block's output bias, held fixed. A function is exactly zero on a
//! token where its pre-activation is not positive, so per token most functions contribute nothing:
//! the sparsity is the functions' own, not a penalty.
//!
//! The first token of a sequence is outside a transcoder's domain: Qwen3's attention sink, where
//! the MLP's input is small (norm 2.4 at layer 14, against 33 elsewhere), M's MLP output is tiny
//! (norm 0.25) and the transcoder fires about 54,000 features with a relative error of about 3,000.
//! So the block runs `M`'s own MLP at position 0 (its operators, fixed and not described) and the
//! transcoder's features at every other position ([`Node::Select`]).
//!
//! Only features that fire on the training sample at positions after the first are kept
//! ([`firing`]): a feature whose pre-activation is not positive on any such token adds exactly zero
//! to the block's output on every clean training input at the starting point, so dropping it
//! leaves the start unchanged there. The kept features are written once in the library's layout ([`write_kept`]): `gate`
//! `k × d`, `gate_bias` `k × 1`, `out` `d × k` (`W_dec`'s kept rows transposed) and `bias` `d × 1`
//! in the transcoder's own storage type (bfloat16 reals copied bit for bit), and `features` (the
//! kept features' indices in the transcoder, as float32, exact below 2^24). The library's operators
//! read them where that file stores them ([`Operator::stored`]), with no float64 copy on the host.
//!
//! A transcoder's gate rows are not maps of `M`, so they carry no read patch: the experiments of a
//! transcoder layer are the clean ones, the hybrids and the other layers' read patches
//! (`interchange::reads_of` leaves out read variables whose owners are not operators of `M`).

use crate::{
    artifact::{Argument, Artifact, Callee, Owner},
    device_program::{DeviceProgram, gelu_tanh_constant, law_of},
    library_mdl::sequence_family,
    operator_program::{Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Rule},
    run_check::LayerNodes,
    safetensors::{SafetensorsFile, Stored, StoredFloat, StoredType},
};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The tensors a plain ReLU transcoder file holds; any other (a JumpReLU `threshold`, a skip
/// connection `W_skip`) is refused rather than dropped.
const TENSORS: [&str; 4] = ["W_enc", "W_dec", "b_enc", "b_dec"];

/// One layer's transcoder file.
pub struct Transcoder {
    path: PathBuf,
    file: SafetensorsFile,
    /// `F` features reading and writing `d`-wide vectors.
    pub features: usize,
    pub width: usize,
    float: StoredFloat,
}

impl Transcoder {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = SafetensorsFile::open(path).map_err(error)?;
        let at = |e: String| format!("{}: {e}", path.display());
        if let Some(other) = file.tensors().keys().find(|k| !TENSORS.contains(&k.as_str())) {
            return Err(at(format!("tensor {other}: only a plain ReLU transcoder ({TENSORS:?}) is read")));
        }
        let entry = |name: &str| file.tensors().get(name).ok_or_else(|| at(format!("no tensor {name}")));
        let (features, width) = match entry("W_enc")?.shape[..] {
            [f, d] => (f, d),
            ref other => return Err(at(format!("W_enc of shape {other:?}"))),
        };
        for (name, shape) in [("W_dec", vec![features, width]), ("b_enc", vec![features]), ("b_dec", vec![width])] {
            if entry(name)?.shape != shape {
                return Err(at(format!("{name} of shape {:?}, not {shape:?}", entry(name)?.shape)));
            }
        }
        let float = match entry("W_enc")?.dtype {
            StoredType::Float(float @ (StoredFloat::F32 | StoredFloat::Bf16)) => float,
            ref other => return Err(at(format!("W_enc stored as {other:?}: float32 or bfloat16 required"))),
        };
        if TENSORS.iter().any(|name| file.tensors()[*name].dtype != StoredType::Float(float)) {
            return Err(at("its tensors are stored in different types".into()));
        }
        Ok(Self { path: path.to_path_buf(), file, features, width, float })
    }

    fn matrix(&self, name: &str) -> Result<Stored, String> {
        self.file.stored(name, self.features, self.width).map_err(error)
    }

    fn vector(&self, name: &str, len: usize) -> Result<Vec<f64>, String> {
        Ok(self.file.vector(name, len).map_err(error)?.to_vec())
    }

    /// The transcoder's MLP output for the rows of `x` (`n × d`), in float64 on the host: the
    /// reference the library's block is checked against.
    pub fn reconstruction(&self, x: &ndarray::Array2<f64>) -> Result<ndarray::Array2<f64>, String> {
        let (encoder, decoder) = (self.matrix("W_enc")?.matrix(), self.matrix("W_dec")?.matrix());
        let (c, b) = (ndarray::Array1::from(self.vector("b_enc", self.features)?), ndarray::Array1::from(self.vector("b_dec", self.width)?));
        let active = (x.dot(&encoder.t()) + &c).mapv(|t| t.max(0.0));
        Ok(active.dot(&decoder) + &b)
    }

    /// Writes features `kept` (indices into the transcoder, increasing) in the library's layout
    /// (module note) to `path`.
    pub fn write_kept(&self, kept: &[usize], path: &Path) -> Result<(), String> {
        if kept.windows(2).any(|w| w[0] >= w[1]) || kept.last().is_some_and(|&f| f >= self.features) {
            return Err(format!("kept features must increase within {}", self.features));
        }
        let (k, d, bytes) = (kept.len(), self.width, if self.float == StoredFloat::F32 { 4 } else { 2 });
        let (encoder, decoder) = (self.matrix("W_enc")?, self.matrix("W_dec")?);
        // The stored reals re-encoded in their own type: exact, since each was widened from it.
        let float = self.float;
        let encode = move |v: f64, out: &mut Vec<u8>| match float {
            StoredFloat::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            _ => out.extend_from_slice(&(((v as f32).to_bits() >> 16) as u16).to_le_bytes()),
        };
        let row = |m: &Stored, f: usize| m.rows(f..f + 1).ok_or_else(|| format!("row {f}"));
        let mut gate = Vec::with_capacity(k * d * bytes);
        for &f in kept {
            row(&encoder, f)?.values().for_each(|v| encode(v, &mut gate));
        }
        let c = self.vector("b_enc", self.features)?;
        let mut gate_bias = Vec::with_capacity(k * bytes);
        kept.iter().for_each(|&f| encode(c[f], &mut gate_bias));
        // `out` is `W_dec`'s kept rows transposed: entry (j, i) is feature `kept[i]`'s coordinate j.
        let mut columns = vec![0.0f64; d * k];
        for (i, &f) in kept.iter().enumerate() {
            for (j, v) in row(&decoder, f)?.values().enumerate() {
                columns[j * k + i] = v;
            }
        }
        let mut out = Vec::with_capacity(k * d * bytes);
        columns.iter().for_each(|&v| encode(v, &mut out));
        drop(columns);
        let mut bias = Vec::with_capacity(d * bytes);
        self.vector("b_dec", d)?.iter().for_each(|&v| encode(v, &mut bias));
        let mut features = Vec::with_capacity(4 * k);
        kept.iter().for_each(|&f| features.extend_from_slice(&(f as f32).to_le_bytes()));
        let dtype = if self.float == StoredFloat::F32 { "F32" } else { "BF16" };
        let parts: [(&str, &str, Vec<usize>, &[u8]); 5] =
            [("gate", dtype, vec![k, d], &gate), ("gate_bias", dtype, vec![k, 1], &gate_bias), ("out", dtype, vec![d, k], &out), ("bias", dtype, vec![d, 1], &bias), ("features", "F32", vec![k], &features)];
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"transcoder": self.path.display().to_string(), "features": self.features.to_string()}));
        let mut offset = 0usize;
        for (name, dtype, shape, data) in &parts {
            header.insert((*name).into(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.len()]}));
            offset += data.len();
        }
        let mut text = serde_json::Value::Object(header).to_string();
        while text.len() % 8 != 0 {
            text.push(' ');
        }
        let partial = path.with_extension("partial");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(|e| format!("{}: {e}", partial.display()))?);
        file.write_all(&(text.len() as u64).to_le_bytes()).map_err(error)?;
        file.write_all(text.as_bytes()).map_err(error)?;
        for (_, _, _, data) in &parts {
            file.write_all(data).map_err(error)?;
        }
        file.into_inner().map_err(error)?.sync_all().map_err(error)?;
        std::fs::rename(&partial, path).map_err(error)
    }
}

/// The transcoder indices of the features a kept file ([`Transcoder::write_kept`]) holds.
pub fn kept_features(path: &Path) -> Result<Vec<usize>, String> {
    let file = SafetensorsFile::open(path).map_err(error)?;
    let k = file.tensors().get("features").map(|e| e.shape.iter().product()).ok_or("no features")?;
    Ok(file.vector("features", k).map_err(error)?.iter().map(|&f| f as usize).collect())
}

/// Per transcoder layer, on how many tokens of `sequences` after each one's first (where the block
/// runs `M`'s own MLP, module note) each feature fires (its pre-activation `g_i·x + c_i` is
/// positive) at `M`'s own MLP input `x`, from `M` run on `device`, `batch` sequences at a time.
/// The counts are float32 sums of zeros and ones, exact below 2^24 tokens.
pub fn firing(device: &Device, native: &OperatorProgram, layers: &[LayerNodes], transcoders: &BTreeMap<usize, Transcoder>, sequences: &[Vec<u32>], batch: usize) -> Result<BTreeMap<usize, Vec<u64>>, String> {
    let mut program = DeviceProgram::compile(device, native)?;
    program.set_arithmetic(if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 });
    let d = program.device();
    let code = law_of(Law::Relu).code();
    // Per layer its encoder (`F × d`, the device's storage), bias row and counts.
    let mut per_layer: Vec<(usize, Tensor, Tensor, Tensor, usize)> = Vec::new();
    for (&l, transcoder) in transcoders {
        let layer = layers.get(l).ok_or_else(|| format!("transcoder layer {l} of {} layers", layers.len()))?;
        let encoder = transcoder.matrix("W_enc")?.f32_values().ok_or("W_enc in f32")?;
        let encoder = d.upload_f32(transcoder.features, transcoder.width, &encoder).map_err(error)?;
        let bias = d.upload_vec(1, transcoder.features, transcoder.vector("b_enc", transcoder.features)?).map_err(error)?;
        per_layer.push((l, encoder, bias, d.zeros(1, transcoder.features).map_err(error)?, layer.normed));
    }
    let end = per_layer.iter().map(|p| p.4).max().ok_or("no transcoder layers")? + 1;
    for chunk in sequences.chunks(batch.max(1)) {
        let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let trace = program.forward_span(&family, None, end, |_, _| Ok(None))?;
        for (_, encoder, bias, counts, normed) in &mut per_layer {
            let x = trace.value(*normed)?;
            let mut z = d.empty(x.rows(), encoder.rows()).map_err(error)?;
            d.gemm(&mut z, 1.0, x, Op::N, encoder, Op::T, 0.0, program.arithmetic()).map_err(error)?;
            d.add_row(&mut z, 1.0, bias).map_err(error)?;
            let codes = d.upload_indices(&vec![code; z.cols()]).map_err(error)?;
            let ones = d.broadcast_rows(&d.upload_vec(1, z.cols(), vec![1.0; z.cols()]).map_err(error)?, z.rows()).map_err(error)?;
            // ReLU's slope: one where the pre-activation is positive, zero elsewhere.
            let fired = d.law_slopes(&ones, &z, &codes, gelu_tanh_constant()).map_err(error)?;
            drop((z, ones));
            let positions = &family.layout.as_ref().ok_or("a sequence layout")?.position;
            let tokens = d.upload_vec(1, fired.rows(), positions.iter().map(|&p| if p == 0 { 0.0 } else { 1.0 }).collect()).map_err(error)?;
            d.gemm(counts, 1.0, &tokens, Op::N, &fired, Op::N, 1.0, program.arithmetic()).map_err(error)?;
        }
    }
    per_layer
        .into_iter()
        .map(|(l, _, _, counts, _)| Ok((l, d.download(&counts).map_err(error)?.iter().map(|&c| c.round() as u64).collect())))
        .collect()
}

/// `M`'s MLP at a layer: its gate map (and bias), its law per unit group, its up map (and bias)
/// when gated, and its down map.
struct NativeMlp {
    gate: Operator,
    gate_bias: Option<Operator>,
    laws: Vec<Law>,
    up: Option<(Operator, Option<Operator>)>,
    down: Operator,
}

fn native_mlp(native: &OperatorProgram, layer: &LayerNodes, l: usize) -> Result<NativeMlp, String> {
    let affine = |node: usize| -> Result<(Operator, Option<Operator>), String> {
        match &native.nodes[node] {
            Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == layer.normed => {
                Ok((native.operators[terms[0].1].as_ref().clone(), bias.map(|b| native.operators[b].as_ref().clone())))
            }
            other => Err(format!("layer {l}: an MLP map is {other:?}")),
        }
    };
    let (pointwise, up) = match &native.nodes[layer.active] {
        Node::Hadamard { left, right } => (*left, Some(affine(*right)?)),
        Node::Pointwise { .. } => (layer.active, None),
        other => return Err(format!("layer {l}: the MLP's activation is {other:?}")),
    };
    let Node::Pointwise { input, laws } = &native.nodes[pointwise] else {
        return Err(format!("layer {l}: the MLP's law is not one pointwise node"));
    };
    let (gate, gate_bias) = affine(*input)?;
    let down = match &native.nodes[layer.mlp] {
        Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == layer.active => native.operators[terms[0].1].as_ref().clone(),
        other => return Err(format!("layer {l}: the MLP's output map is {other:?}")),
    };
    Ok(NativeMlp { gate, gate_bias, laws: laws.clone(), up, down })
}

/// The native operator applied to `input` by the first affine node reading it.
fn reader(native: &OperatorProgram, input: usize) -> Result<&Operator, String> {
    native
        .nodes
        .iter()
        .find_map(|node| match node {
            Node::Affine { terms, .. } if terms.first().is_some_and(|t| t.0 == input) => Some(native.operators[terms[0].1].as_ref()),
            _ => None,
        })
        .ok_or_else(|| format!("no map reads node {input}"))
}

/// `artifact` with layer `l`'s MLP (`layer.normed` to `layer.mlp`) replaced by the library block of
/// the transcoder features a kept file holds ([`Transcoder::write_kept`]), its operators named as a
/// plain MLP's (`library.l{l}.mlp.gate`, `.gate_bias`, `.out`, and the fixed output `.bias`), and
/// per function the owners of its gate row, gate bias and output column: the transcoder's
/// tensors (`transcoder.l{l}.W_enc` and so on) at the feature's row, not operators of `M`.
pub fn mlp(native: &OperatorProgram, artifact: Artifact, layer: &LayerNodes, l: usize, kept: &Path) -> Result<(Artifact, Vec<Owner>), String> {
    let file = SafetensorsFile::open(kept).map_err(error)?;
    let features = kept_features(kept)?;
    let k = features.len();
    let input = reader(native, layer.normed)?.cols.clone();
    let output = match &native.nodes[layer.mlp] {
        Node::Affine { terms, bias: None } if terms.len() == 1 => native.operators[terms[0].1].rows.clone(),
        other => return Err(format!("layer {l}: the MLP's output node is {other:?}")),
    };
    let d = input.width();
    if output.width() != d {
        return Err(format!("layer {l}: the MLP reads {d} and writes {}", output.width()));
    }
    let units = Interface::uniform(k, 1, LabelKind::Unit, 0).map_err(error)?;
    let name = format!("library.l{l}.mlp");
    let sink = native_mlp(native, layer, l)?;
    let source = format!("transcoder.l{l}");
    let provenance = || Provenance::derived(&[&Provenance::native(&source)], "transcoder import".into());
    let stored = |part: &str, rows: usize, cols: usize| file.stored(part, rows, cols).map_err(error);
    let base = artifact.program.operators.len();
    let mut operators = vec![
        Operator::stored(format!("{name}.gate"), units.clone(), input.clone(), stored("gate", k, d)?, provenance()).map_err(error)?,
        Operator::stored(format!("{name}.gate_bias"), units.clone(), Interface::constant(), stored("gate_bias", k, 1)?, provenance()).map_err(error)?,
        Operator::stored(format!("{name}.out"), output.clone(), units, stored("out", d, k)?, provenance()).map_err(error)?,
        Operator::stored(format!("{name}.bias"), output, Interface::constant(), stored("bias", d, 1)?, provenance()).map_err(error)?,
    ];
    let mut nodes = vec![
        Node::Param { index: 0 },
        Node::Affine { terms: vec![(0, base)], bias: Some(base + 1) },
        Node::Pointwise { input: 1, laws: vec![Law::Relu; k] },
        Node::Affine { terms: vec![(2, base + 2)], bias: Some(base + 3) },
    ];
    // M's own MLP, for position 0: its operators copied under `m_` names (no prior group reads
    // them, so they stay fixed), its laws as M has them.
    let mut copy = |op: &Operator, part: &str| -> usize {
        let mut op = op.clone();
        op.name = format!("{name}.m_{part}");
        operators.push(op);
        base + operators.len() - 1
    };
    let gate = copy(&sink.gate, "gate");
    let gate_bias = sink.gate_bias.as_ref().map(|b| copy(b, "gate_bias"));
    nodes.push(Node::Affine { terms: vec![(0, gate)], bias: gate_bias });
    nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: sink.laws.clone() });
    let mut active = nodes.len() - 1;
    if let Some((up, up_bias)) = &sink.up {
        let (up, up_bias) = (copy(up, "up"), up_bias.as_ref().map(|b| copy(b, "up_bias")));
        nodes.push(Node::Affine { terms: vec![(0, up)], bias: up_bias });
        nodes.push(Node::Hadamard { left: active, right: nodes.len() - 1 });
        active = nodes.len() - 1;
    }
    let down = copy(&sink.down, "down");
    nodes.push(Node::Affine { terms: vec![(active, down)], bias: None });
    nodes.push(Node::Select { inside: nodes.len() - 1, outside: 3, positions: vec![0] });
    let rule = Rule { name: name.clone(), inputs: vec![input], output: nodes.len() - 1, nodes };
    let mut owners = Vec::with_capacity(3 * k);
    for (i, &f) in features.iter().enumerate() {
        for (part, tensor, rows, cols, transposed) in [("gate", "W_enc", i..i + 1, 0..d, false), ("gate_bias", "b_enc", i..i + 1, 0..1, false), ("out", "W_dec", 0..d, i..i + 1, true)] {
            owners.push(Owner {
                operator: format!("{name}.{part}"),
                rows,
                cols,
                body: name.clone(),
                site: name.clone(),
                native: format!("{source}.{tensor}"),
                native_rows: f..f + 1,
                native_cols: if part == "gate_bias" { 0..1 } else { 0..d },
                role: part.to_string(),
                transposed,
                ..Owner::default()
            });
        }
    }
    Ok((artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(layer.normed)], layer.mlp, operators)?, owners))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        library_mdl,
        operator_program::SlotValues,
        run_check::{layer_nodes, split_sites},
    };
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// A bfloat16 transcoder file of `features` random features on `d` coordinates.
    fn transcoder_file(path: &Path, features: usize, d: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bf16 = |n: usize, scale: f64| -> Vec<u8> { (0..n).flat_map(|_| ((((rng.random::<f64>() - 0.5) * scale) as f32).to_bits() >> 16).to_le_bytes()[..2].to_vec()).collect() };
        // Every fourth feature's bias is -64, far below any pre-activation: it never fires.
        let dead = |mut bias: Vec<u8>| {
            bias.chunks_exact_mut(2).step_by(4).for_each(|b| b.copy_from_slice(&((-64.0f32).to_bits() >> 16).to_le_bytes()[..2]));
            bias
        };
        let parts = [("W_dec", vec![features, d], bf16(features * d, 1.0)), ("W_enc", vec![features, d], bf16(features * d, 2.0)), ("b_dec", vec![d], bf16(d, 0.2)), ("b_enc", vec![features], dead(bf16(features, 1.0)))];
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        for (name, shape, data) in &parts {
            header.insert((*name).into(), serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + data.len()]}));
            offset += data.len();
        }
        let text = serde_json::Value::Object(header).to_string();
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        parts.iter().for_each(|(_, _, data)| bytes.extend_from_slice(data));
        std::fs::write(path, bytes).unwrap();
    }

    /// On the tiny Qwen3 decoder with layer 1's MLP replaced by a transcoder's firing features: the
    /// library's MLP output is `M`'s own MLP output at each sequence's first token and the
    /// transcoder's reconstruction at every other (to float64 rounding), every dropped feature is
    /// off on every token after the first, and every kept feature's gate, gate bias and output are
    /// the file's reals; the device's firing counts (first tokens left out) are the host's, and the
    /// device's forward and reverse through the block's position select are the host's.
    #[test]
    fn a_transcoder_layer_is_its_reconstruction_with_the_dead_features_dropped() {
        let dir = std::env::temp_dir().join(format!("library_transcoder_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let export = crate::test_support::tiny_qwen3_export("library_transcoder", 2);
        let imported = import_language_model(&export, 6, 12).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let (features, d) = (64, 8);
        let path = dir.join("layer_1.safetensors");
        transcoder_file(&path, features, d, 3);
        let transcoder = Transcoder::open(&path).unwrap();
        // The host's count of each feature's firing tokens at M's MLP input.
        let family = library_mdl::sequence_family(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        let x = native.execute(&family, false).unwrap().values[layers[1].normed].clone();
        let encoder = transcoder.matrix("W_enc").unwrap().matrix();
        let c = ndarray::Array1::from(transcoder.vector("b_enc", features).unwrap());
        let pre = x.dot(&encoder.t()) + &c;
        let first: Vec<bool> = family.layout.as_ref().unwrap().position.iter().map(|&p| p == 0).collect();
        let expected: Vec<u64> = (0..features).map(|f| pre.column(f).iter().zip(&first).filter(|(t, first)| **t > 0.0 && !**first).count() as u64).collect();
        let mut transcoders = BTreeMap::new();
        transcoders.insert(1, transcoder);
        let counts = firing(&Device::host(), &native, &layers, &transcoders, &sequences, 2).unwrap();
        assert_eq!(counts[&1], expected, "the device counts each feature's firing tokens");
        let kept: Vec<usize> = (0..features).filter(|&f| expected[f] > 0).collect();
        assert!(kept.len() < features && !kept.is_empty(), "some features fire and some do not ({} of {features})", kept.len());
        let kept_path = dir.join("kept_1.safetensors");
        transcoders[&1].write_kept(&kept, &kept_path).unwrap();
        assert_eq!(kept_features(&kept_path).unwrap(), kept);
        let mut files = BTreeMap::new();
        files.insert(1, kept_path.clone());
        let explanation = library_mdl::explanation_with(&native, &layers, &files).unwrap();
        explanation.artifact.validate_coverage(&native).unwrap();
        assert_eq!(explanation.layers[1].functions.len(), kept.len());
        assert!(explanation.layers[1].functions.iter().all(|f| f.len() == 2), "a gate and an output group per function");
        let program = &explanation.artifact.program;
        let index = |name: &str| program.operators.iter().position(|op| op.name == name).unwrap();
        let (gate, out) = (program.operators[index("library.l1.mlp.gate")].matrix(), program.operators[index("library.l1.mlp.out")].matrix());
        let decoder = transcoders[&1].matrix("W_dec").unwrap().matrix();
        for (i, &f) in kept.iter().enumerate() {
            assert!(gate.row(i).iter().zip(encoder.row(f)).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(out.column(i).iter().zip(decoder.row(f)).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        // The library's MLP output at M's input is the full transcoder's reconstruction.
        let trace = explanation.artifact.execute(&family).unwrap();
        let place = |n: usize| explanation.artifact.place(n).unwrap();
        let mlp = trace.values[place(layers[1].mlp)].clone();
        let input = trace.values[place(layers[1].normed)].clone();
        assert_eq!(input, x, "layer 0 is M's, so the transcoder reads M's input");
        let mut reference = transcoders[&1].reconstruction(&x).unwrap();
        let native_mlp = native.execute(&family, false).unwrap().values[layers[1].mlp].clone();
        for (row, &first) in first.iter().enumerate() {
            if first {
                reference.row_mut(row).assign(&native_mlp.row(row));
            }
        }
        let scale = reference.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let difference = reference.iter().zip(&mlp).fold(0.0_f64, |a, (r, m)| a.max((r - m).abs()));
        assert!(difference <= 1e-12 * scale, "the library's MLP differs from M's at first tokens and the transcoder elsewhere by {difference} (scale {scale})");
        // The select node survives the artifact's code.
        let decoded = crate::artifact::Artifact::from_bytes(&explanation.artifact.to_bytes().unwrap(), &explanation.artifact.program.declarations).unwrap();
        assert!(decoded.program.nodes == explanation.artifact.program.nodes && decoded.program.rules.iter().zip(&explanation.artifact.program.rules).all(|(a, b)| a.nodes == b.nodes));
        // The device's forward and reverse through the select against the host's, on the flat
        // program: the block's output, and the cotangent at its input of a cotangent seeded there.
        let (flat, roots) = crate::artifact_device::mapped_inlined(&explanation.artifact.program).unwrap();
        let (input_node, output_node) = (roots[place(layers[1].normed)], roots[place(layers[1].mlp)]);
        let host_trace = flat.execute(&family, false).unwrap();
        let mut rng = StdRng::seed_from_u64(9);
        let seed = ndarray::Array2::from_shape_fn(host_trace.values[output_node].dim(), |_| rng.random::<f64>() - 0.5);
        let host_cotangent = crate::derivatives::vjp_seeded(&flat, &family, &host_trace, BTreeMap::from([(output_node, seed.clone())]), None).unwrap()[input_node].clone().unwrap();
        for device in crate::device_program_tests::devices() {
            let tolerance = if device.float64() { 1e-12 } else { 1e-4 };
            let mut program = DeviceProgram::compile(&device, &flat).unwrap();
            program.set_arithmetic(if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 });
            let trace = program.forward(&family).unwrap();
            let forward = device.download(trace.value(output_node).unwrap()).unwrap();
            let difference = forward.iter().zip(&host_trace.values[output_node]).fold(0.0_f64, |a, (p, q)| a.max((p - q).abs()));
            assert!(difference <= tolerance * scale, "{}: the device's block output differs by {difference}", device.name());
            let zero = device.zeros(family.rows, program.widths()[program.hidden()]).unwrap();
            let extra = BTreeMap::from([(output_node, device.upload(seed.view()).unwrap())]);
            let cotangents = program.vjp_seeded(&trace, zero, extra, &[input_node], if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 }).unwrap();
            let device_cotangent = device.download(&cotangents[&input_node]).unwrap();
            let size = host_cotangent.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let difference = device_cotangent.iter().zip(&host_cotangent).fold(0.0_f64, |a, (p, q)| a.max((p - q).abs()));
            assert!(difference <= tolerance * size, "{}: the device's cotangent through the select differs by {difference} (size {size})", device.name());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
