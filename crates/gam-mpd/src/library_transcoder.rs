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
//! the MLP's input is small (norm 2.4 at layer 14, against 33 elsewhere) and the transcoder fires
//! about 54,000 features with a relative error of about 3,000. `M`'s MLP output there is nearly one
//! vector per layer whatever the token: replacing it at every layer by its mean over the training
//! sequences' first tokens raises KL(M ‖ M') by 0.0084 bits per token after the first (Qwen3-0.6B,
//! 32 held-out windows of 512, the mean over 1,024 training windows), although its norm is 0.1–0.4
//! at layers 3–15 and 295–6,694 at layers 2, 26 and 27. So the block's output at position 0 is one
//! described vector, `library.l{l}.mlp.sink` (`d × 1`, its own prior group), started at that mean
//! ([`firing`]), and the transcoder's features give it at every other position ([`Node::Select`]).
//! `M`'s MLP is not run.
//!
//! Only features that fire on the training sample at positions after the first are kept
//! ([`firing`]): a feature whose pre-activation is not positive on any such token adds exactly zero
//! to the block's output on every clean training input at the starting point, so dropping it
//! leaves the start unchanged there. The kept features are written once in the library's layout ([`Transcoder::write_kept`]): `gate`
//! `k × d`, `gate_bias` `k × 1`, `out` `d × k` (`W_dec`'s kept rows transposed) and `bias` `d × 1`
//! in the transcoder's own storage type (bfloat16 reals copied bit for bit), `features` (the
//! kept features' indices in the transcoder, as float32, exact below 2^24) and `sink` (`d × 1`,
//! float32: the start of the block's output at position 0). The library's operators
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

    /// Writes features `kept` (indices into the transcoder, increasing) and the block's output at
    /// position 0, `sink` (`d` reals, stored as float32), in the library's layout (module note) to
    /// `path`.
    pub fn write_kept(&self, kept: &[usize], sink: &[f64], path: &Path) -> Result<(), String> {
        if kept.windows(2).any(|w| w[0] >= w[1]) || kept.last().is_some_and(|&f| f >= self.features) {
            return Err(format!("kept features must increase within {}", self.features));
        }
        if sink.len() != self.width {
            return Err(format!("a sink of {} reals for width {}", sink.len(), self.width));
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
        let sink: Vec<u8> = sink.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect();
        let dtype = if self.float == StoredFloat::F32 { "F32" } else { "BF16" };
        let parts: [(&str, &str, Vec<usize>, &[u8]); 6] = [
            ("gate", dtype, vec![k, d], &gate),
            ("gate_bias", dtype, vec![k, 1], &gate_bias),
            ("out", dtype, vec![d, k], &out),
            ("bias", dtype, vec![d, 1], &bias),
            ("features", "F32", vec![k], &features),
            ("sink", "F32", vec![d, 1], &sink),
        ];
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

/// What [`firing`] counts per transcoder layer.
pub struct Fired {
    /// Per feature, the tokens it fires on.
    pub counts: Vec<u64>,
    /// `M`'s MLP output at the sequences' first tokens, averaged: the start of the block's output
    /// there (module note).
    pub sink: Vec<f64>,
}

/// Per transcoder layer, on how many tokens of `sequences` after each one's first (where the block
/// gives its described vector, module note) each feature fires (its pre-activation `g_i·x + c_i`
/// is positive) at `M`'s own MLP input `x`, and the mean of `M`'s MLP output over the first tokens,
/// from `M` run on `device`, `batch` sequences at a time. The counts are float32 sums of zeros and
/// ones, exact below 2^24 tokens. The encoders are held on the device [`ENCODER_BYTES`] at a time,
/// `M` run once per such group of layers.
pub fn firing(device: &Device, native: &OperatorProgram, layers: &[LayerNodes], transcoders: &BTreeMap<usize, Transcoder>, sequences: &[Vec<u32>], batch: usize) -> Result<BTreeMap<usize, Fired>, String> {
    let mut program = DeviceProgram::compile(device, native)?;
    program.set_arithmetic(if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 });
    let mut groups: Vec<BTreeMap<usize, &Transcoder>> = vec![BTreeMap::new()];
    let mut bytes = 0usize;
    for (&l, transcoder) in transcoders {
        let size = 4 * transcoder.features * transcoder.width;
        if bytes > 0 && bytes + size > ENCODER_BYTES {
            groups.push(BTreeMap::new());
            bytes = 0;
        }
        bytes += size;
        groups.last_mut().expect("a group").insert(l, transcoder);
    }
    let mut out = BTreeMap::new();
    for group in groups {
        out.extend(firing_group(&program, layers, &group, sequences, batch)?);
    }
    Ok(out)
}

/// The encoders' bytes [`firing`] holds on the device at once (each layer's `F × d` in f32): six
/// of the 163,840-feature Qwen3-0.6B transcoders (671 MB each in f32), leaving a 24 GB card room for `M`.
pub const ENCODER_BYTES: usize = 4 << 30;

fn firing_group(program: &DeviceProgram, layers: &[LayerNodes], transcoders: &BTreeMap<usize, &Transcoder>, sequences: &[Vec<u32>], batch: usize) -> Result<BTreeMap<usize, Fired>, String> {
    let d = program.device();
    let code = law_of(Law::Relu).code();
    // Per layer its encoder (`F × d`, the device's storage), bias row, counts, MLP input and output
    // nodes, and the sum of its MLP outputs at first tokens.
    let mut per_layer: Vec<(usize, Tensor, Tensor, Tensor, (usize, usize), Tensor)> = Vec::new();
    for (&l, transcoder) in transcoders {
        let layer = layers.get(l).ok_or_else(|| format!("transcoder layer {l} of {} layers", layers.len()))?;
        let encoder = transcoder.matrix("W_enc")?.f32_values().ok_or("W_enc in f32")?;
        let encoder = d.upload_f32(transcoder.features, transcoder.width, &encoder).map_err(error)?;
        let bias = d.upload_vec(1, transcoder.features, transcoder.vector("b_enc", transcoder.features)?).map_err(error)?;
        per_layer.push((l, encoder, bias, d.zeros(1, transcoder.features).map_err(error)?, (layer.normed, layer.mlp), d.zeros(1, transcoder.width).map_err(error)?));
    }
    let end = per_layer.iter().map(|p| p.4.0.max(p.4.1)).max().ok_or("no transcoder layers")? + 1;
    let mut firsts = 0usize;
    for chunk in sequences.chunks(batch.max(1)) {
        let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let trace = program.forward_span(&family, None, end, |_, _| Ok(None))?;
        let positions = &family.layout.as_ref().ok_or("a sequence layout")?.position;
        let first = d.upload_vec(1, positions.len(), positions.iter().map(|&p| if p == 0 { 1.0 } else { 0.0 }).collect()).map_err(error)?;
        firsts += positions.iter().filter(|&&p| p == 0).count();
        for (_, encoder, bias, counts, (normed, mlp), sink) in &mut per_layer {
            d.gemm(sink, 1.0, &first, Op::N, trace.value(*mlp)?, Op::N, 1.0, program.arithmetic()).map_err(error)?;
            let x = trace.value(*normed)?;
            let mut z = d.empty(x.rows(), encoder.rows()).map_err(error)?;
            d.gemm(&mut z, 1.0, x, Op::N, encoder, Op::T, 0.0, program.arithmetic()).map_err(error)?;
            d.add_row(&mut z, 1.0, bias).map_err(error)?;
            let codes = d.upload_indices(&vec![code; z.cols()]).map_err(error)?;
            let ones = d.broadcast_rows(&d.upload_vec(1, z.cols(), vec![1.0; z.cols()]).map_err(error)?, z.rows()).map_err(error)?;
            // ReLU's slope: one where the pre-activation is positive, zero elsewhere.
            let fired = d.law_slopes(&ones, &z, &codes, gelu_tanh_constant()).map_err(error)?;
            drop((z, ones));
            let tokens = d.upload_vec(1, fired.rows(), positions.iter().map(|&p| if p == 0 { 0.0 } else { 1.0 }).collect()).map_err(error)?;
            d.gemm(counts, 1.0, &tokens, Op::N, &fired, Op::N, 1.0, program.arithmetic()).map_err(error)?;
        }
    }
    if firsts == 0 {
        return Err("no sequences: the first tokens' MLP output has no mean".into());
    }
    per_layer
        .into_iter()
        .map(|(l, _, _, counts, _, sink)| {
            let counts = d.download(&counts).map_err(error)?.iter().map(|&c| c.round() as u64).collect();
            let sink = d.download(&sink).map_err(error)?.iter().map(|&s| s / firsts as f64).collect();
            Ok((l, Fired { counts, sink }))
        })
        .collect()
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
/// plain MLP's (`library.l{l}.mlp.gate`, `.gate_bias`, `.out`, and the fixed output `.bias`) and
/// its output at each sequence's first token `library.l{l}.mlp.sink` (module note), and per
/// function the owners of its gate row, gate bias and output column: the transcoder's tensors
/// (`transcoder.l{l}.W_enc` and so on) at the feature's row, not operators of `M`.
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
    let source = format!("transcoder.l{l}");
    let provenance = || Provenance::derived(&[&Provenance::native(&source)], "transcoder import".into());
    let stored = |part: &str, rows: usize, cols: usize| file.stored(part, rows, cols).map_err(error);
    let base = artifact.program.operators.len();
    let operators = vec![
        Operator::stored(format!("{name}.gate"), units.clone(), input.clone(), stored("gate", k, d)?, provenance()).map_err(error)?,
        Operator::stored(format!("{name}.gate_bias"), units.clone(), Interface::constant(), stored("gate_bias", k, 1)?, provenance()).map_err(error)?,
        Operator::stored(format!("{name}.out"), output.clone(), units, stored("out", d, k)?, provenance()).map_err(error)?,
        Operator::stored(format!("{name}.bias"), output.clone(), Interface::constant(), stored("bias", d, 1)?, provenance()).map_err(error)?,
        Operator::stored(format!("{name}.sink"), output, Interface::constant(), stored("sink", d, 1)?, Provenance::derived(&[&Provenance::native(&format!("{source}.sink"))], "M's MLP output at first tokens, averaged".into())).map_err(error)?,
    ];
    let nodes = vec![
        Node::Param { index: 0 },
        Node::Affine { terms: vec![(0, base)], bias: Some(base + 1) },
        Node::Pointwise { input: 1, laws: vec![Law::Relu; k] },
        Node::Affine { terms: vec![(2, base + 2)], bias: Some(base + 3) },
        Node::Constant { operator: base + 4 },
        Node::Select { inside: 4, outside: 3, positions: vec![0] },
    ];
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

    /// On the tiny Qwen3 decoder with layer 1's MLP replaced by a transcoder's firing features: the
    /// library's MLP output is the sink (`M`'s MLP output at the first tokens, averaged, as the
    /// device's firing pass makes it) at each sequence's first token and the
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
        crate::test_support::transcoder_file(&path, features, d, 3);
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
        let fired = firing(&Device::host(), &native, &layers, &transcoders, &sequences, 2).unwrap();
        assert_eq!(fired[&1].counts, expected, "the device counts each feature's firing tokens");
        // The sink's start: M's MLP output at the first tokens, averaged.
        let native_mlp = native.execute(&family, false).unwrap().values[layers[1].mlp].clone();
        let firsts: Vec<usize> = (0..first.len()).filter(|&r| first[r]).collect();
        let mean: Vec<f64> = (0..d).map(|j| firsts.iter().map(|&r| native_mlp[[r, j]]).sum::<f64>() / firsts.len() as f64).collect();
        let size = mean.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(fired[&1].sink.iter().zip(&mean).all(|(a, b)| (a - b).abs() <= 1e-12 * size), "the device averages M's MLP output at first tokens");
        let kept: Vec<usize> = (0..features).filter(|&f| expected[f] > 0).collect();
        assert!(kept.len() < features && !kept.is_empty(), "some features fire and some do not ({} of {features})", kept.len());
        let kept_path = dir.join("kept_1.safetensors");
        transcoders[&1].write_kept(&kept, &fired[&1].sink, &kept_path).unwrap();
        assert_eq!(kept_features(&kept_path).unwrap(), kept);
        let mut files = BTreeMap::new();
        files.insert(1, kept_path.clone());
        let explanation = library_mdl::explanation_with(&native, &layers, &files).unwrap();
        explanation.artifact.validate_coverage(&native).unwrap();
        assert_eq!(explanation.layers[1].functions.len(), kept.len());
        assert!(explanation.layers[1].functions.iter().all(|f| f.len() == 2), "a gate and an output group per function");
        // The block's thresholds are one group of the layer (75a2712a6c), kept, with the block, by an
        // explanation of that block alone: its gate-bias operator stays trainable.
        let bias = explanation.artifact.program.operators.iter().position(|op| op.name == "library.l1.mlp.gate_bias").unwrap();
        assert_eq!(explanation.layers[1].thresholds.len(), 1, "one threshold group");
        let block = library_mdl::scoped(&explanation, &[3]).unwrap();
        assert!(block.trainable.contains(&bias), "the scoped block's thresholds are trainable");
        assert_eq!(block.layers[1].thresholds.len(), 1);
        assert!(block.groups[block.layers[1].thresholds[0]].cells.iter().all(|c| c.operator == bias));
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
        // The sink as stored (float32).
        let sink: Vec<f64> = mean.iter().map(|&v| f64::from(v as f32)).collect();
        for (row, &first) in first.iter().enumerate() {
            if first {
                reference.row_mut(row).assign(&ndarray::ArrayView1::from(&sink[..]));
            }
        }
        let scale = reference.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let difference = reference.iter().zip(&mlp).fold(0.0_f64, |a, (r, m)| a.max((r - m).abs()));
        assert!(difference <= 1e-12 * scale, "the library's MLP differs from the sink at first tokens and the transcoder elsewhere by {difference} (scale {scale})");
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

    /// A transcoder block's thresholds are one prior group of the layer, apart from the gate rows
    /// (`library_mdl::explanation_with`): on the tiny Qwen3 decoder with layer 1's MLP a
    /// transcoder's features, each function's gate group is its gate row alone, one group holds
    /// every kept threshold, and at the start with the thresholds 40 times their file values (the
    /// scale of Qwen3's thresholds against their gate weights) the description is below the one
    /// each threshold would cost in its gate row's group, by at least the log-sum gap
    /// `½ Σ_i [(d + 1) ln(S_i⁺ / (d + 1)) − d ln(S_i / d)] − ½ k ln(S_c / k)` less the new group's
    /// own precision and scale bits.
    #[test]
    fn a_transcoder_blocks_thresholds_are_one_group_with_their_own_variance() {
        let dir = std::env::temp_dir().join(format!("library_transcoder_thresholds_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let export = crate::test_support::tiny_qwen3_export("library_transcoder_thresholds", 2);
        let imported = import_language_model(&export, 6, 12).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let path = dir.join("layer_1.safetensors");
        crate::test_support::transcoder_file(&path, 64, 8, 3);
        let transcoders = BTreeMap::from([(1, Transcoder::open(&path).unwrap())]);
        let fired = firing(&Device::host(), &native, &layers, &transcoders, &sequences, 2).unwrap();
        let kept: Vec<usize> = (0..64).filter(|&f| fired[&1].counts[f] > 0).collect();
        let kept_path = dir.join("kept_1.safetensors");
        transcoders[&1].write_kept(&kept, &fired[&1].sink, &kept_path).unwrap();
        let explanation = library_mdl::explanation_with(&native, &layers, &BTreeMap::from([(1, kept_path)])).unwrap();
        let program = &explanation.artifact.program;
        let index = |name: &str| program.operators.iter().position(|op| op.name == name).unwrap();
        let (gate, bias) = (index("library.l1.mlp.gate"), index("library.l1.mlp.gate_bias"));
        let k = kept.len();
        let thresholds = explanation.groups.iter().position(|g| g.name == "library.l1.mlp.gate_bias").expect("the thresholds' group");
        let cells = &explanation.groups[thresholds].cells;
        assert!(cells.len() == 1 && cells[0].operator == bias && cells[0].rows == (0..k).collect::<Vec<_>>() && cells[0].cols == (0..1));
        for (i, function) in explanation.layers[1].functions.iter().enumerate() {
            let cells = &explanation.groups[function[0]].cells;
            assert!(cells.len() == 1 && cells[0].operator == gate && cells[0].rows == [i], "function {i}'s gate group is its gate row alone");
        }
        // M's own MLP at layer 0 keeps its grouping, and has no thresholds' group.
        assert!(!explanation.groups.iter().any(|g| g.name == "library.l0.mlp.gate_bias"));
        let at = |op: usize| explanation.trainable.iter().position(|t| *t == op).unwrap();
        let mut posterior = library_mdl::Posterior::new(&explanation, 1 << 20).unwrap();
        posterior.mean[at(bias)].mapv_inplace(|c| 40.0 * c);
        let costs = posterior.costs();
        // The cost each function's gate group would have with its threshold in it, from the same
        // moments (reference: the merged group's starting mean square).
        let (means, log_sd) = (&posterior.mean[at(gate)], &posterior.log_sd[at(gate)]);
        let (c, c_log_sd) = (&posterior.mean[at(bias)], &posterior.log_sd[at(bias)]);
        let start = |op: usize| program.operators[op].matrix();
        let (gate_start, bias_start) = (start(gate), start(bias));
        let d = means.ncols() as f64;
        let mut merged = 0.0;
        let mut gap = 0.0;
        let mut thresholds_second = 0.0;
        for i in 0..k {
            let row_second: f64 = means.row(i).iter().zip(log_sd.row(i)).map(|(m, s)| m * m + (2.0 * s).exp()).sum();
            let c_second = c[[i, 0]] * c[[i, 0]] + (2.0 * c_log_sd[[i, 0]]).exp();
            let log_variance: f64 = log_sd.row(i).iter().map(|s| 2.0 * s).sum::<f64>() + 2.0 * c_log_sd[[i, 0]];
            let reference = (gate_start.row(i).iter().map(|v| v * v).sum::<f64>() + bias_start[[i, 0]] * bias_start[[i, 0]]) / (d + 1.0);
            let (_, divergence, bits) = gam_gpu::tensor::group_prior(d + 1.0, row_second + c_second, log_variance, Some(reference));
            merged += divergence + 0.5 * (d + 1.0).ln() + bits * std::f64::consts::LN_2;
            merged -= costs[explanation.layers[1].functions[i][0]];
            gap += 0.5 * ((d + 1.0) * ((row_second + c_second) / (d + 1.0)).ln() - d * (row_second / d).ln());
            thresholds_second += c_second;
        }
        merged -= costs[thresholds];
        gap -= 0.5 * k as f64 * (thresholds_second / k as f64).ln();
        let overhead = costs[thresholds] - (0.5 * k as f64 * (thresholds_second / k as f64).ln() - 0.5 * c_log_sd.iter().map(|s| 2.0 * s).sum::<f64>());
        assert!(merged > 0.0, "the thresholds' own group costs {merged} nats more than in their rows");
        assert!(merged >= gap - overhead.abs() - 1e-6 * gap.abs(), "the description falls by {merged} nats, the log-sum gap is {gap} less {overhead}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A removal step on the tiny Qwen3 decoder with layer 1's MLP as a transcoder block (fixed
    /// output bias, the sink vector at the first token), from the Laplace start: it runs through
    /// the compensation and the search, and never raises `F` on the training collection.
    #[test]
    fn a_removal_step_runs_on_a_transcoder_block() {
        let dir = std::env::temp_dir().join(format!("library_transcoder_removal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let export = crate::test_support::tiny_qwen3_export("library_transcoder_removal", 2);
        let imported = import_language_model(&export, 6, 12).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let (train, held) = sequences.split_at(sequences.len() - 2);
        let path = dir.join("layer_1.safetensors");
        crate::test_support::transcoder_file(&path, 64, 8, 3);
        let transcoders = BTreeMap::from([(1, Transcoder::open(&path).unwrap())]);
        let fired = firing(&Device::host(), &native, &layers, &transcoders, train, 2).unwrap();
        let kept: Vec<usize> = (0..64).filter(|&f| fired[&1].counts[f] > 0).collect();
        let kept_path = dir.join("kept_1.safetensors");
        transcoders[&1].write_kept(&kept, &fired[&1].sink, &kept_path).unwrap();
        let explanation = library_mdl::explanation_with(&native, &layers, &BTreeMap::from([(1, kept_path)])).unwrap();
        let settings: library_mdl::Settings = serde_json::from_value(serde_json::json!({"batch_sequences": 2, "seed": 3, "numeric_bytes": 1 << 26, "head_tile_rows": 64})).unwrap();
        let device = Device::host();
        let mut posterior = library_mdl::start_posterior(&device, &native, &explanation, train, &settings).unwrap();
        let active = posterior.active.iter().filter(|a| **a).count();
        let step = library_mdl::Step { sequences: train, held, settings: &settings, log: None };
        let (removal, before, after) = library_mdl::removal_step(&device, &native, &explanation, &mut posterior, step).unwrap();
        assert!(removal.after_bits <= removal.before_bits, "the removal raised F from {} to {} bits", removal.before_bits, removal.after_bits);
        assert_eq!(posterior.active.iter().filter(|a| **a).count(), active - removal.removed);
        assert!(before.objective_bits_per_token.is_finite() && after.objective_bits_per_token.is_finite());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
