//! A decoder model's blocks as fixed computations on the device (#2951): the
//! [`interchange::BlockEngine`] the experiments run on, in place of the general program executor.
//!
//! A model of the decoder family (the native model `M`, or a library explanation `P` of it) is a
//! token embedding, then per layer an attention block and an MLP block, then a final norm. Each
//! block reads the residual stream through an RMS norm with a gain (its read, which an experiment
//! may patch), and adds its output to the stream.
//!
//! * Attention: one product of the read with the stacked query, key and value maps (each key and
//!   value map once, however many query heads read it), the queries' and keys' per-head norms and
//!   rotation, causal attention per query head over its key-value head, and the output map.
//! * MLP: one product with the stacked input maps (the gates, and for a gated law the inputs), the
//!   law (GELU in its tanh form, or the gated SiLU), and the output map.
//!
//! The block is recognized in the program by its nodes ([`Decoder::new`]), its operators stacked
//! once, and run with the fused kernels of `gam_gpu` (`Device::rms_gain`, `heads_rope`,
//! `causal_attention`, `swiglu`, `gelu_tanh` and their reverses). The stream, the norms and every
//! cotangent are f32. The products run in one arithmetic ([`Decoder::with_arithmetic`], f32 by
//! default): in [`Arithmetic::Bf16`] each product's operands are rounded to bfloat16, which
//! changes a model's divergence per token by as much as the divergences an experiment measures;
//! in any other, the operands stay f32 and the product rounds them as its arithmetic says
//! ([`Arithmetic::Tf32x3`] keeps f32's accuracy on the tensor cores). Attention reads its heads
//! in bfloat16 in every arithmetic until its kernel takes f32. A program the recognition does not
//! take runs on the reference engine (`interchange::Model`).
//!
//! The trainable operators' values are read from the program they live in ([`Decoder::refresh`]),
//! their gradients returned per operator.

use crate::{
    device_program::DeviceProgram,
    interchange::BlockEngine,
    operator_program::{Law, Node, Operator, OperatorBody, OperatorProgram, Rotary},
};
use gam_gpu::tensor::{Arithmetic, Device, HeadLayout, Indices, Op, Storage, Tensor};
use ndarray::{Array2, Axis, concatenate};
use std::{
    collections::BTreeMap,
    ops::Range,
    sync::{Arc, Mutex},
};

fn error(e: impl std::fmt::Display) -> String {
    format!("decoder: {e}")
}

/// A stacked operator: its row blocks, each an operator of the program and its first row.
#[derive(Clone, Debug)]
struct Stack {
    parts: Vec<(usize, usize, usize)>,
    rows: usize,
    cols: usize,
}

impl Stack {
    fn new(program: &OperatorProgram, ops: &[usize]) -> Result<Self, String> {
        let mut parts = Vec::with_capacity(ops.len());
        let mut rows = 0;
        let cols = ops.first().map(|op| program.operators[*op].cols.width()).ok_or_else(|| error("an empty stack"))?;
        for &op in ops {
            let (r, c) = (program.operators[op].rows.width(), program.operators[op].cols.width());
            if c != cols {
                return Err(error(format!("operator {} has {c} columns in a stack of {cols}", program.operators[op].name)));
            }
            parts.push((op, rows, r));
            rows += r;
        }
        Ok(Self { parts, rows, cols })
    }

    fn values(&self, program: &OperatorProgram) -> Result<Array2<f64>, String> {
        let blocks: Vec<Array2<f64>> = self.parts.iter().map(|(op, _, _)| program.operators[*op].matrix()).collect();
        let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
        concatenate(Axis(0), &views).map_err(error)
    }
}

/// An attention block's operators.
#[derive(Clone, Debug)]
struct Attention {
    gain: usize,
    epsilon: f64,
    /// Queries, then keys, then values, each key and value map once.
    projections: Stack,
    layout: HeadLayout,
    /// The queries' then keys' per-head norm gains (diagonal operators) and their ε.
    norms: Option<(Vec<usize>, f64)>,
    /// The output map: per query head its operator (output width × head width), side by side.
    output: Vec<usize>,
    scale: f64,
    rotary: Option<Rotary>,
}

/// An MLP block's operators.
#[derive(Clone, Debug)]
struct Mlp {
    gain: usize,
    epsilon: f64,
    /// The gates' map, then for a gated law the inputs' map.
    input: Stack,
    bias: Option<usize>,
    gated: bool,
    output: usize,
    /// The final norm after the last block: its gain and ε.
    last: Option<(usize, f64)>,
}

#[derive(Clone, Debug)]
enum Block {
    Attention(Attention),
    Mlp(Mlp),
}

/// A block's weights on the device, in f32.
struct Weights {
    gain: Tensor,
    /// The stacked input product, and its gradient's row blocks per operator.
    input: Tensor,
    norms: Option<Tensor>,
    output: Tensor,
    bias: Option<Tensor>,
    last: Option<Tensor>,
}

/// The decoder engine of one model (module note).
pub struct Decoder {
    device: Device,
    blocks: Vec<Block>,
    weights: Vec<Weights>,
    embedding: Tensor,
    width: usize,
    trainable: Vec<usize>,
    /// The bytes of one gradient of the trainable operators.
    gradient_bytes: usize,
    /// The products' arithmetic, in the blocks and outside them (the head, the patches).
    arithmetic: Arithmetic,
    /// Per rotary configuration, the angles of positions `0..span` (span × planes), grown to the
    /// longest sequence seen.
    angles: Mutex<Vec<(Rotary, usize, Arc<(Tensor, Tensor)>)>>,
}

/// A layer's shape ([`Decoder::layer_shape`]).
#[derive(Clone, Copy, Debug)]
pub struct LayerShape {
    pub width: usize,
    pub heads: HeadLayout,
    pub projections: usize,
    pub mlp_inputs: usize,
    pub gated: bool,
    pub epsilon: f64,
    pub scale: f64,
}

/// What a block's reverse pass reads of its forward pass.
pub struct Tape {
    x: Tensor,
    scale: Tensor,
    read: Tensor,
    inner: Inner,
}

/// The projections and the MLP's input products are recomputed in the reverse pass from the read
/// (one product each) rather than kept: they are the largest values of a block (the MLP's inputs
/// six times the stream's width per row in Qwen3), and keeping them would bound the rows a step can
/// take by memory.
enum Inner {
    Attention { head_scales: Option<Tensor>, heads: Tensor, angles: Option<(Tensor, Tensor)>, attended: (Tensor, Tensor), sequences: Vec<Range<usize>> },
    Mlp { out: Option<(Tensor, Tensor)> },
}

/// Node `n` when it is an affine node of one term with no bias: its input and operator.
fn single(program: &OperatorProgram, n: usize) -> Option<(usize, usize)> {
    match &program.nodes[n] {
        Node::Affine { terms, bias: None } if terms.len() == 1 => Some(terms[0]),
        _ => None,
    }
}

fn identity(op: &Operator) -> bool {
    matches!(op.body, OperatorBody::Identity)
}

fn diagonal(op: &Operator) -> bool {
    op.diagonal().is_some()
}

/// A read: an RMS norm of `stream` scaled by a diagonal gain, its gain operator and ε.
fn read(program: &OperatorProgram, stream: usize, read: usize) -> Result<(usize, f64), String> {
    let (normed, gain) = single(program, read).ok_or_else(|| error(format!("read {read} is not a gain")))?;
    match program.nodes[normed] {
        Node::RmsNorm { input, epsilon } if input == stream && diagonal(&program.operators[gain]) => Ok((gain, epsilon)),
        _ => Err(error(format!("read {read} is not the RMS norm of stream {stream}"))),
    }
}

/// The residual node `end = stream + inner` (two identity terms), and `inner`.
fn residual(program: &OperatorProgram, stream: usize, end: usize) -> Result<usize, String> {
    match &program.nodes[end] {
        Node::Affine { terms, bias: None } if terms.len() == 2 && terms.iter().all(|(_, op)| identity(&program.operators[*op])) => {
            let others: Vec<usize> = terms.iter().map(|(n, _)| *n).filter(|n| *n != stream).collect();
            match others[..] {
                [inner] => Ok(inner),
                _ => Err(error(format!("node {end} is not stream {stream} plus one block output"))),
            }
        }
        _ => Err(error(format!("node {end} is not a residual sum"))),
    }
}

/// `n`, or the node an identity barrier at `n` passes on (an inlined rule's output).
fn through_barrier(program: &OperatorProgram, n: usize) -> usize {
    match single(program, n) {
        Some((input, op)) if identity(&program.operators[op]) => input,
        _ => n,
    }
}

/// A query's or key's projection from `read`, through its per-head norm when it has one: the
/// projection's operator, and the norm's gain operator and ε.
fn projection(program: &OperatorProgram, n: usize, read: usize) -> Result<(usize, Option<(usize, f64)>), String> {
    if let Some((input, op)) = single(program, n)
        && input == read
        && !diagonal(&program.operators[op])
    {
        return Ok((op, None));
    }
    let (normed, gain) = single(program, n).ok_or_else(|| error(format!("node {n} is not a projection")))?;
    let Node::RmsNorm { input, epsilon } = program.nodes[normed] else { return Err(error(format!("node {n} is not a normed projection"))) };
    let (from, op) = single(program, input).ok_or_else(|| error(format!("node {input} is not a projection")))?;
    if from != read || !diagonal(&program.operators[gain]) {
        return Err(error(format!("node {n} is not a normed projection of read {read}")));
    }
    Ok((op, Some((gain, epsilon))))
}

impl Decoder {
    /// The decoder of the flat program `program` (`interchange::sites`, through its hidden node)
    /// with blocks entering at `entries` and reading at `reads`, its operators `trainable`
    /// updated by [`Self::refresh`]; products in f32. Refused when a block is not of the decoder
    /// family, or when a trainable operator is not one of its stacked maps (the projections, an
    /// MLP's input maps, an MLP's output map), whose values it would not refresh.
    pub fn new(device: &Device, program: &OperatorProgram, (entries, reads, hidden): (&[usize], &[usize], usize), trainable: &[usize]) -> Result<Self, String> {
        if !(device.is_host() || device.storage() == Storage::F32) {
            return Err(error("the decoder runs on the host or on CUDA in f32 storage"));
        }
        let mut blocks = Vec::with_capacity(entries.len());
        for (b, (&stream, &read_node)) in entries.iter().zip(reads).enumerate() {
            let end = entries.get(b + 1).copied().unwrap_or(hidden);
            blocks.push(if b % 2 == 0 { Block::Attention(attention(program, stream, read_node, end)?) } else { Block::Mlp(mlp(program, stream, read_node, end, b + 1 == entries.len())?) });
        }
        let maps: Vec<usize> = blocks
            .iter()
            .flat_map(|b| match b {
                Block::Attention(a) => a.projections.parts.iter().map(|(op, _, _)| *op).collect::<Vec<_>>(),
                Block::Mlp(m) => m.input.parts.iter().map(|(op, _, _)| *op).chain([m.output]).collect(),
            })
            .collect();
        if let Some(op) = trainable.iter().find(|op| !maps.contains(op)) {
            return Err(error(format!("trainable operator {} is not a stacked map", program.operators[*op].name)));
        }
        // The embedding: the first stream is a gather of a table's rows by the tokens.
        let table = single(program, entries[0]).filter(|(input, _)| matches!(program.nodes[*input], Node::Feature { .. })).ok_or_else(|| error("the first stream is not an embedding"))?;
        let embedding = device.upload(program.operators[table.1].matrix().t()).map_err(error)?;
        let width = embedding.cols();
        let value = if device.storage() == Storage::F32 { 4 } else { 8 };
        let gradient_bytes = trainable.iter().map(|op| program.operators[*op].rows.width() * program.operators[*op].cols.width() * value).sum();
        let mut out = Self { device: device.clone(), blocks, weights: Vec::new(), embedding, width, trainable: trainable.to_vec(), gradient_bytes, arithmetic: Arithmetic::F32, angles: Mutex::new(Vec::new()) };
        out.weights = out.blocks.iter().map(|b| out.upload(program, b)).collect::<Result<_, _>>()?;
        Ok(out)
    }

    fn upload(&self, program: &OperatorProgram, block: &Block) -> Result<Weights, String> {
        let d = &self.device;
        let up = |m: Array2<f64>| d.upload(m.view()).map_err(error);
        let row = |op: usize| -> Result<Tensor, String> {
            let values = program.operators[op].diagonal().ok_or_else(|| error("a gain that is not diagonal"))?;
            d.upload_vec(1, values.len(), values.to_vec()).map_err(error)
        };
        Ok(match block {
            Block::Attention(a) => {
                let norms = match &a.norms {
                    Some((gains, _)) => {
                        let rows: Vec<Vec<f64>> = gains.iter().map(|g| program.operators[*g].diagonal().map(|d| d.to_vec()).ok_or_else(|| error("a head gain that is not diagonal"))).collect::<Result<_, _>>()?;
                        let flat: Vec<f64> = rows.concat();
                        Some(d.upload_vec(gains.len(), a.layout.width, flat).map_err(error)?)
                    }
                    None => None,
                };
                let blocks: Vec<Array2<f64>> = a.output.iter().map(|op| program.operators[*op].matrix()).collect();
                let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                Weights { gain: row(a.gain)?, input: up(a.projections.values(program)?)?, norms, output: up(concatenate(Axis(1), &views).map_err(error)?)?, bias: None, last: None }
            }
            Block::Mlp(m) => Weights {
                gain: row(m.gain)?,
                input: up(m.input.values(program)?)?,
                norms: None,
                output: up(program.operators[m.output].matrix())?,
                bias: m.bias.map(|b| up(program.operators[b].matrix().t().to_owned())).transpose()?,
                last: m.last.map(|(g, _)| row(g)).transpose()?,
            },
        })
    }

    /// The floating-point operations of every block's products on `rows` rows of sequences of
    /// `length` (forward, reverse): each product `2 m n k`; attention's scores and values over the
    /// causal half; the reverse twice the forward's products (the input's cotangent and the
    /// weights'), its attention weights recomputed.
    #[must_use]
    pub fn product_flops(&self, rows: usize, length: usize) -> (f64, f64) {
        let (rows, length, d) = (rows as f64, length as f64, self.width as f64);
        let (mut forward, mut reverse) = (0.0, 0.0);
        for block in &self.blocks {
            match block {
                Block::Attention(a) => {
                    let (heads, w) = (a.layout.queries as f64, a.layout.width as f64);
                    let products = 2.0 * rows * d * (a.projections.rows as f64 + heads * w);
                    // Scores and values: two products of rows × (length / 2) × w per query head.
                    let attention = 2.0 * 2.0 * rows * (length / 2.0) * w * heads;
                    forward += products + attention;
                    // The scores recomputed, then four products (weights', values', queries', keys').
                    reverse += 2.0 * products + 2.0 * rows * d * a.projections.rows as f64 + 0.5 * attention + 2.0 * attention;
                }
                Block::Mlp(m) => {
                    let inputs = m.input.rows as f64;
                    let products = 2.0 * rows * d * (inputs + if m.gated { inputs / 2.0 } else { inputs });
                    forward += products;
                    reverse += 2.0 * products + 2.0 * rows * d * inputs;
                }
            }
        }
        (forward, reverse)
    }

    /// The first layer's shape: its heads, the stacked projections' and MLP inputs' rows, whether
    /// its MLP is gated, and the stream's width (the decoder's layers share one shape).
    #[must_use]
    pub fn layer_shape(&self) -> Option<LayerShape> {
        match (self.blocks.first(), self.blocks.get(1)) {
            (Some(Block::Attention(a)), Some(Block::Mlp(m))) => {
                Some(LayerShape { width: self.width, heads: a.layout, projections: a.projections.rows, mlp_inputs: m.input.rows, gated: m.gated, epsilon: a.epsilon, scale: a.scale })
            }
            _ => None,
        }
    }

    /// The products' arithmetic (module note), in the blocks and outside them (the head's sweep of
    /// the vocabulary, the patches).
    #[must_use]
    pub fn with_arithmetic(mut self, arithmetic: Arithmetic) -> Self {
        self.arithmetic = arithmetic;
        self
    }

    /// The trainable operators' current values from `program` (the explanation's resident program,
    /// after its weight sample is written): each stack's row blocks copied into place (widened to
    /// f32 where the program holds bfloat16, `DeviceProgram::hold_bf16`).
    pub fn refresh(&mut self, program: &DeviceProgram) -> Result<(), String> {
        let d = self.device.clone();
        let value = |op: usize| -> Result<Tensor, String> {
            let value = program.dense(op)?;
            if value.storage() == Storage::Bf16 { d.convert(value) } else { d.copy(value) }.map_err(error)
        };
        for (block, weights) in self.blocks.iter().zip(&mut self.weights) {
            let (stack, output) = match block {
                Block::Attention(a) => (&a.projections, None),
                Block::Mlp(m) => (&m.input, Some(m.output)),
            };
            for &(op, at, _) in &stack.parts {
                if self.trainable.contains(&op) {
                    d.set_rows(&mut weights.input, at, &value(op)?).map_err(error)?;
                }
            }
            if let Some(op) = output.filter(|op| self.trainable.contains(op)) {
                weights.output = value(op)?;
            }
        }
        Ok(())
    }

    /// The rows `ranges` (each a sequence from position 0) of `t`, stacked in order.
    fn gather(&self, t: &Tensor, ranges: &[Range<usize>]) -> Result<Tensor, String> {
        let d = &self.device;
        if covers(ranges, t.rows()) {
            return d.copy(t).map_err(error);
        }
        let mut out = d.zeros(ranges.iter().map(ExactSizeIterator::len).sum(), t.cols()).map_err(error)?;
        let mut at = 0;
        for r in ranges {
            d.set_rows(&mut out, at, &d.rows_of(t, r.start, r.len()).map_err(error)?).map_err(error)?;
            at += r.len();
        }
        Ok(out)
    }

    fn scatter(&self, t: &mut Tensor, ranges: &[Range<usize>], values: Tensor) -> Result<(), String> {
        let d = &self.device;
        if covers(ranges, t.rows()) {
            *t = values;
            return Ok(());
        }
        let mut at = 0;
        for r in ranges {
            d.set_rows(t, r.start, &d.rows_of(&values, at, r.len()).map_err(error)?).map_err(error)?;
            at += r.len();
        }
        Ok(())
    }

    /// The rotary angles of the call's rows (each range a sequence from position 0), from a table
    /// of each position's angles computed once.
    fn angles(&self, rotary: Rotary, ranges: &[Range<usize>]) -> Result<(Tensor, Tensor), String> {
        let span = ranges.iter().map(ExactSizeIterator::len).max().unwrap_or(0);
        let table = {
            let mut cached = self.angles.lock().map_err(|_| error("poisoned angle tables"))?;
            match cached.iter().find(|(r, covered, _)| *r == rotary && *covered >= span) {
                Some((_, _, t)) => Arc::clone(t),
                None => {
                    let planes = rotary.pairs().len();
                    let (mut cos, mut sin) = (Vec::with_capacity(span * planes), Vec::with_capacity(span * planes));
                    for position in 0..u32::try_from(span).map_err(error)? {
                        for plane in 0..planes {
                            let (c, s) = rotary.turn(plane, position);
                            cos.push(c);
                            sin.push(s);
                        }
                    }
                    let t = Arc::new((self.device.upload_vec(span, planes, cos).map_err(error)?, self.device.upload_vec(span, planes, sin).map_err(error)?));
                    cached.retain(|(r, ..)| *r != rotary);
                    cached.push((rotary, span, Arc::clone(&t)));
                    t
                }
            }
        };
        let positions: Vec<u32> = ranges.iter().flat_map(|r| 0..r.len() as u32).collect();
        let ids: Indices = self.device.upload_indices(&positions).map_err(error)?;
        Ok((self.device.gather_rows(&table.0, &ids).map_err(error)?, self.device.gather_rows(&table.1, &ids).map_err(error)?))
    }

    /// Whether `stack` holds a trainable operator.
    fn trains(&self, stack: &Stack) -> bool {
        stack.parts.iter().any(|(op, _, _)| self.trainable.contains(op))
    }

    /// Adds row blocks of a stacked gradient into the per-operator gradients of the trainable ones.
    fn add_parts(&self, stack: &Stack, stacked: &Tensor, gradient: &mut BTreeMap<usize, Tensor>) -> Result<(), String> {
        let d = &self.device;
        for &(op, at, rows) in &stack.parts {
            if !self.trainable.contains(&op) {
                continue;
            }
            let part = d.rows_of(stacked, at, rows).map_err(error)?;
            add(d, gradient, op, part)?;
        }
        Ok(())
    }
}

/// Whether `ranges` are consecutive and cover rows `0..rows` in order: the call's rows are the
/// buffer's.
fn covers(ranges: &[Range<usize>], rows: usize) -> bool {
    ranges.first().is_some_and(|r| r.start == 0) && ranges.last().is_some_and(|r| r.end == rows) && ranges.windows(2).all(|w| w[0].end == w[1].start)
}

fn add(d: &Device, gradient: &mut BTreeMap<usize, Tensor>, op: usize, value: Tensor) -> Result<(), String> {
    match gradient.get_mut(&op) {
        Some(total) => d.axpy(total, 1.0, &value).map_err(error),
        None => {
            gradient.insert(op, value);
            Ok(())
        }
    }
}

/// Recognizes an attention block: `stream`'s read `read`, heads reading it, and `end = stream +
/// Σ heads' outputs`.
fn attention(program: &OperatorProgram, stream: usize, read_node: usize, end: usize) -> Result<Attention, String> {
    let (gain, epsilon) = read(program, stream, read_node)?;
    let outputs = residual(program, stream, end)?;
    let Node::Affine { terms, bias: None } = &program.nodes[outputs] else { return Err(error(format!("node {outputs} is not the heads' output map"))) };
    let (mut queries, mut keys, mut values) = (Vec::new(), Vec::<usize>::new(), Vec::<usize>::new());
    let (mut q_norms, mut k_norms) = (Vec::new(), Vec::new());
    let (mut output, mut group_of) = (Vec::new(), Vec::new());
    let mut shape = None;
    for &(term, through) in terms {
        let attend = through_barrier(program, term);
        let Node::Attend { query, key, value, scale, rotary, causal } = program.nodes[attend] else { return Err(error(format!("node {attend} is not an attend"))) };
        if !causal {
            return Err(error("an attention that is not causal"));
        }
        let (q, q_norm) = projection(program, query, read_node)?;
        let (k, k_norm) = projection(program, key, read_node)?;
        let (v, None) = projection(program, value, read_node)? else { return Err(error("a normed value")) };
        if *shape.get_or_insert((scale.value().to_bits(), rotary)) != (scale.value().to_bits(), rotary) {
            return Err(error("heads with different scales or rotations"));
        }
        // Key-value maps by operator: each once, however many query heads read it.
        let g = match keys.iter().position(|op| *op == k) {
            Some(g) if values[g] == v => g,
            Some(_) => return Err(error("a key map read with two value maps")),
            None => {
                keys.push(k);
                values.push(v);
                k_norms.push(k_norm);
                keys.len() - 1
            }
        };
        queries.push(q);
        q_norms.push(q_norm);
        group_of.push(g);
        output.push(through);
    }
    let (heads, kv) = (queries.len(), keys.len());
    if heads == 0 || heads % kv != 0 || group_of.iter().enumerate().any(|(h, g)| *g != h / (heads / kv)) {
        return Err(error("query heads that do not read their key-value heads in equal consecutive groups"));
    }
    let norms = match (q_norms.iter().all(Option::is_some), k_norms.iter().all(Option::is_some), q_norms.iter().any(Option::is_some) || k_norms.iter().any(Option::is_some)) {
        (true, true, _) => {
            let all: Vec<(usize, f64)> = q_norms.iter().chain(&k_norms).flatten().copied().collect();
            if all.iter().any(|(_, e)| *e != all[0].1) {
                return Err(error("head norms with different ε"));
            }
            Some((all.iter().map(|(g, _)| *g).collect(), all[0].1))
        }
        (_, _, false) => None,
        _ => return Err(error("some heads normed and others not")),
    };
    let ops: Vec<usize> = queries.iter().chain(&keys).chain(&values).copied().collect();
    let projections = Stack::new(program, &ops)?;
    let width = projections.rows / (heads + 2 * kv);
    if projections.parts.iter().any(|(_, _, r)| *r != width) {
        return Err(error("heads of different widths"));
    }
    let (bits, rotary) = shape.ok_or_else(|| error("no heads"))?;
    if rotary.is_some_and(|r| 2 * r.pairs().len() > width) {
        return Err(error("a rotation wider than a head"));
    }
    Ok(Attention { gain, epsilon, projections, layout: HeadLayout { queries: heads, keys: kv, width }, norms, output, scale: f64::from_bits(bits), rotary })
}

/// Recognizes an MLP block (GELU in its tanh form with an optional bias, or the gated SiLU), and
/// for the last block the final norm `end = rms(residual) gain`.
fn mlp(program: &OperatorProgram, stream: usize, read_node: usize, end: usize, last: bool) -> Result<Mlp, String> {
    let (gain, epsilon) = read(program, stream, read_node)?;
    let (residual_node, last) = if last {
        let (final_gain, final_epsilon) = {
            let (normed, g) = single(program, end).ok_or_else(|| error("the hidden node is not the final norm's gain"))?;
            let Node::RmsNorm { input, epsilon } = program.nodes[normed] else { return Err(error("the hidden node is not a final norm")) };
            if !diagonal(&program.operators[g]) {
                return Err(error("the final norm's gain is not diagonal"));
            }
            ((input, g), epsilon)
        };
        (final_gain.0, Some((final_gain.1, final_epsilon)))
    } else {
        (end, None)
    };
    let out_node = through_barrier(program, residual(program, stream, residual_node)?);
    let (active, output) = single(program, out_node).ok_or_else(|| error("the MLP's output map is not one product"))?;
    let input_of = |n: usize| -> Result<(usize, Option<usize>), String> {
        match &program.nodes[n] {
            Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == read_node => Ok((terms[0].1, *bias)),
            _ => Err(error(format!("node {n} is not a map of read {read_node}"))),
        }
    };
    let (ops, bias, gated) = match &program.nodes[active] {
        Node::Pointwise { input, laws } if laws.iter().all(|l| *l == Law::GeluTanh) => {
            let (op, bias) = input_of(*input)?;
            (vec![op], bias, false)
        }
        Node::Hadamard { left, right } => {
            let (pointwise, linear) = match (&program.nodes[*left], &program.nodes[*right]) {
                (Node::Pointwise { .. }, _) => (*left, *right),
                (_, Node::Pointwise { .. }) => (*right, *left),
                _ => return Err(error("a gated product without a law")),
            };
            let Node::Pointwise { input, laws } = &program.nodes[pointwise] else { return Err(error("a gated product without a law")) };
            if laws.iter().any(|l| *l != Law::Silu) {
                return Err(error("a gated law other than SiLU"));
            }
            let ((gate, gate_bias), (up, up_bias)) = (input_of(*input)?, input_of(linear)?);
            if gate_bias.is_some() || up_bias.is_some() {
                return Err(error("a gated MLP with biases"));
            }
            (vec![gate, up], None, true)
        }
        _ => return Err(error("an MLP law other than GELU (tanh) or gated SiLU")),
    };
    Ok(Mlp { gain, epsilon, input: Stack::new(program, &ops)?, bias, gated, output, last })
}

impl BlockEngine for Decoder {
    type Tape = Tape;

    fn device(&self) -> &Device {
        &self.device
    }

    fn width(&self) -> usize {
        self.width
    }

    fn blocks(&self) -> usize {
        self.blocks.len()
    }

    fn arithmetic(&self) -> Arithmetic {
        self.arithmetic
    }

    fn forward(
        &self,
        block: usize,
        stream: &mut Tensor,
        ranges: &[Range<usize>],
        tokens: &[&[u32]],
        read_edit: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>>,
        keep: bool,
    ) -> Result<Option<Tape>, String> {
        let d = &self.device;
        let x = if block == 0 {
            let ids: Vec<u32> = tokens.iter().flat_map(|t| t.iter().copied()).collect();
            d.gather_rows(&self.embedding, &d.upload_indices(&ids).map_err(error)?).map_err(error)?
        } else {
            self.gather(stream, ranges)?
        };
        let w = &self.weights[block];
        let (epsilon, rotary) = match &self.blocks[block] {
            Block::Attention(a) => (a.epsilon, a.rotary),
            Block::Mlp(m) => (m.epsilon, None),
        };
        let (mut read, scale) = d.rms_gain(&x, &w.gain, epsilon, false).map_err(error)?;
        if let Some(edit) = read_edit {
            edit(&mut read)?;
        }
        // In bfloat16 each operand is rounded once where it is made, not per product reading it.
        let (arithmetic, half) = (self.arithmetic, self.arithmetic == Arithmetic::Bf16);
        let read = if half { d.bf16_copy(&read).map_err(error)? } else { read };
        let mut out = d.copy(&x).map_err(error)?;
        let inner = match &self.blocks[block] {
            Block::Attention(a) => {
                let mut p = d.empty(x.rows(), a.projections.rows).map_err(error)?;
                d.gemm(&mut p, 1.0, &read, Op::N, &w.input, Op::T, 0.0, arithmetic).map_err(error)?;
                let angles = rotary.map(|r| self.angles(r, ranges)).transpose()?;
                let rotation = angles.as_ref().zip(rotary).map(|((c, s), r)| (c, s, r.half_split));
                let norm = w.norms.as_ref().zip(a.norms.as_ref()).map(|(g, (_, e))| (g, *e));
                let (heads, head_scales) = d.heads_rope(&p, a.layout, norm, rotation).map_err(error)?;
                // Each range is one sequence; the gathered rows hold them in order.
                let sequences: Vec<Range<usize>> = ranges.iter().scan(0, |at, r| { *at += r.len(); Some(*at - r.len()..*at) }).collect();
                let attended = d.causal_attention(&heads, a.layout, &sequences, a.scale).map_err(error)?;
                let wide = if half { None } else { Some(d.convert(&attended.0).map_err(error)?) };
                d.gemm(&mut out, 1.0, wide.as_ref().unwrap_or(&attended.0), Op::N, &w.output, Op::T, 1.0, arithmetic).map_err(error)?;
                Inner::Attention { head_scales, heads, angles, attended, sequences }
            }
            Block::Mlp(m) => {
                let mut pre = d.empty(x.rows(), m.input.rows).map_err(error)?;
                d.gemm(&mut pre, 1.0, &read, Op::N, &w.input, Op::T, 0.0, arithmetic).map_err(error)?;
                let active = if m.gated { d.swiglu(&pre) } else { d.gelu_tanh(&pre, w.bias.as_ref()) }.map_err(error)?;
                d.gemm(&mut out, 1.0, &active, Op::N, &w.output, Op::T, 1.0, arithmetic).map_err(error)?;
                let last = match (&w.last, m.last) {
                    (Some(g), Some((_, e))) => {
                        let (hidden, k) = d.rms_gain(&out, g, e, false).map_err(error)?;
                        let residual = std::mem::replace(&mut out, hidden);
                        Some((residual, k))
                    }
                    _ => None,
                };
                Inner::Mlp { out: last }
            }
        };
        self.scatter(stream, ranges, out)?;
        Ok(keep.then_some(Tape { x, scale, read, inner }))
    }

    fn reverse(
        &self,
        block: usize,
        tape: &Tape,
        cotangent: &mut Tensor,
        ranges: &[Range<usize>],
        read_edit: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>>,
        gradient: &mut BTreeMap<usize, Tensor>,
    ) -> Result<(), String> {
        let d = &self.device;
        let w = &self.weights[block];
        let mut g = self.gather(cotangent, ranges)?;
        let rows = g.rows();
        let mut g_read = d.empty(rows, self.width).map_err(error)?;
        let (arithmetic, half) = (self.arithmetic, self.arithmetic == Arithmetic::Bf16);
        // A cotangent that feeds two products, rounded once in bfloat16.
        let operand = |t: Tensor| if half { d.bf16_copy(&t).map_err(error) } else { Ok(t) };
        match (&self.blocks[block], &tape.inner) {
            (Block::Attention(a), Inner::Attention { head_scales, heads, angles, attended, sequences }) => {
                let mut projections = d.empty(rows, a.projections.rows).map_err(error)?;
                d.gemm(&mut projections, 1.0, &tape.read, Op::N, &w.input, Op::T, 0.0, arithmetic).map_err(error)?;
                let mut g_attended = d.empty(rows, a.layout.queries * a.layout.width).map_err(error)?;
                d.gemm(&mut g_attended, 1.0, &g, Op::N, &w.output, Op::N, 0.0, arithmetic).map_err(error)?;
                let g_heads = d.causal_attention_backward(heads, a.layout, sequences, a.scale, (&attended.0, &attended.1), &g_attended).map_err(error)?;
                let rotation = angles.as_ref().zip(a.rotary).map(|((c, s), r)| (c, s, r.half_split));
                let norm = w.norms.as_ref().zip(head_scales.as_ref());
                let g_p = operand(d.heads_rope_backward(&projections, a.layout, norm, rotation, &g_heads).map_err(error)?)?;
                if self.trains(&a.projections) {
                    let mut stacked = d.empty(a.projections.rows, a.projections.cols).map_err(error)?;
                    d.gemm(&mut stacked, 1.0, &g_p, Op::T, &tape.read, Op::N, 0.0, arithmetic).map_err(error)?;
                    self.add_parts(&a.projections, &stacked, gradient)?;
                }
                d.gemm(&mut g_read, 1.0, &g_p, Op::N, &w.input, Op::N, 0.0, arithmetic).map_err(error)?;
            }
            (Block::Mlp(m), Inner::Mlp { out }) => {
                let mut pre = d.empty(rows, m.input.rows).map_err(error)?;
                d.gemm(&mut pre, 1.0, &tape.read, Op::N, &w.input, Op::T, 0.0, arithmetic).map_err(error)?;
                let active = if m.gated { d.swiglu(&pre) } else { d.gelu_tanh(&pre, w.bias.as_ref()) }.map_err(error)?;
                if let (Some((residual, k)), Some(gain)) = (out, &w.last) {
                    let mut g_residual = d.zeros(rows, self.width).map_err(error)?;
                    d.rms_gain_backward((residual, gain, k), &g, &mut g_residual).map_err(error)?;
                    g = g_residual;
                }
                let rounded = if half { Some(d.bf16_copy(&g).map_err(error)?) } else { None };
                let g_out = rounded.as_ref().unwrap_or(&g);
                let mut g_active = d.empty(rows, active.cols()).map_err(error)?;
                d.gemm(&mut g_active, 1.0, g_out, Op::N, &w.output, Op::N, 0.0, arithmetic).map_err(error)?;
                if self.trainable.contains(&m.output) {
                    let mut g_output = d.empty(self.width, active.cols()).map_err(error)?;
                    d.gemm(&mut g_output, 1.0, g_out, Op::T, &active, Op::N, 0.0, arithmetic).map_err(error)?;
                    add(d, gradient, m.output, g_output)?;
                }
                let g_pre = operand(if m.gated { d.swiglu_backward(&pre, &g_active) } else { d.gelu_tanh_backward(&pre, w.bias.as_ref(), &g_active) }.map_err(error)?)?;
                if self.trains(&m.input) {
                    let mut stacked = d.empty(m.input.rows, m.input.cols).map_err(error)?;
                    d.gemm(&mut stacked, 1.0, &g_pre, Op::T, &tape.read, Op::N, 0.0, arithmetic).map_err(error)?;
                    self.add_parts(&m.input, &stacked, gradient)?;
                }
                d.gemm(&mut g_read, 1.0, &g_pre, Op::N, &w.input, Op::N, 0.0, arithmetic).map_err(error)?;
            }
            _ => return Err(error("a tape of another block")),
        }
        if let Some(edit) = read_edit {
            edit(&mut g_read)?;
        }
        // The stream's cotangent: the residual's own plus the read's through the norm.
        d.rms_gain_backward((&tape.x, &w.gain, &tape.scale), &g_read, &mut g).map_err(error)?;
        if block == 0 {
            g = d.zeros(rows, self.width).map_err(error)?;
        }
        self.scatter(cotangent, ranges, g)
    }

    fn tape_bytes(tape: &Tape) -> usize {
        let inner = match &tape.inner {
            Inner::Attention { head_scales, heads, angles, attended, .. } => {
                head_scales.as_ref().map_or(0, Tensor::bytes)
                    + heads.bytes()
                    + angles.as_ref().map_or(0, |(c, s)| c.bytes() + s.bytes())
                    + attended.0.bytes()
                    + attended.1.bytes()
            }
            Inner::Mlp { out } => out.as_ref().map_or(0, |(r, k)| r.bytes() + k.bytes()),
        };
        tape.x.bytes() + tape.scale.bytes() + tape.read.bytes() + inner
    }

    fn gradient_bytes(&self) -> Result<usize, String> {
        Ok(self.gradient_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        interchange::{self, Model},
        library_mdl,
        operator_program::SlotValues,
        run_check::{layer_nodes, split_sites},
        test_support::{tiny_export, tiny_qwen3_export},
    };

    /// Both engines in `arithmetic`, every block forward and then backward on two sequences, with a
    /// read edit at block 1 (half the read) and its transpose: the final streams, every trainable
    /// operator's gradient and the embedding's cotangent agree within the rounding both share. The
    /// reference rounds its products' operands as the decoder does; only the order of summation,
    /// an operand rounded across a tie, and the decoder's attention, which reads bfloat16 heads in
    /// every arithmetic, differ (a few bfloat16 rounding steps, `2⁻⁸` relative, of the magnitude
    /// the value sums).
    fn parity(dir: std::path::PathBuf, arithmetic: Arithmetic) {
        let imported = import_language_model(&dir, 2, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let explanation = library_mdl::explanation(&native, &layers).unwrap();
        let (flat, entries, reads) = interchange::sites(&explanation.artifact, &layers).unwrap();
        let prefix = interchange::prefix(&flat).unwrap();
        let host = Device::host();
        let mut program = DeviceProgram::compile_values_bounded(&host, &prefix, usize::MAX).unwrap();
        program.set_arithmetic(arithmetic);
        program.prepare_dense_parameters(&explanation.trainable).unwrap();
        let reference = Model::new(&program, &flat, entries.clone(), reads.clone(), &explanation.trainable).unwrap();
        let decoder = Decoder::new(&host, &prefix, (&entries, &reads, program.hidden()), &explanation.trainable).unwrap().with_arithmetic(arithmetic);
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<&[u32]> = tokens.chunks(12).collect();
        let ranges = [0..12, 12..24];
        let run = |engine: &dyn Fn(usize, &mut Tensor, bool) -> Result<(), String>| {
            let mut stream = host.zeros(24, 8).unwrap();
            for b in 0..4 {
                engine(b, &mut stream, b == 1).unwrap();
            }
            stream
        };
        let half = |t: &mut Tensor| -> Result<(), String> {
            let mut out = host.zeros(t.rows(), t.cols()).map_err(|e| e.to_string())?;
            host.axpy(&mut out, 0.5, t).map_err(|e| e.to_string())?;
            *t = out;
            Ok(())
        };
        let reference_stream = run(&|b, s, edit| {
            let mut e = half;
            reference.forward(b, s, &ranges, &sequences, if edit { Some(&mut e) } else { None }, false).map(|_| ())
        });
        let decoder_stream = run(&|b, s, edit| {
            let mut e = half;
            decoder.forward(b, s, &ranges, &sequences, if edit { Some(&mut e) } else { None }, false).map(|_| ())
        });
        let (a, b) = (host.download(&reference_stream).unwrap(), host.download(&decoder_stream).unwrap());
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let difference = a.iter().zip(&b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(difference <= 4.0 / 256.0 * scale, "final streams differ by {difference} of {scale}");
        // The reverse: forward with tapes, then every block backward from one cotangent.
        let cotangent = host.upload(Array2::from_shape_fn((24, 8), |(r, c)| ((r * 8 + c) as f64 * 0.37).sin()).view()).unwrap();
        let mut reference_gradient = BTreeMap::new();
        let mut decoder_gradient = BTreeMap::new();
        {
            let mut stream = host.zeros(24, 8).unwrap();
            let mut tapes = Vec::new();
            for b in 0..4 {
                let mut e = half;
                tapes.push(reference.forward(b, &mut stream, &ranges, &sequences, if b == 1 { Some(&mut e) } else { None }, true).unwrap().unwrap());
            }
            let mut g = host.copy(&cotangent).unwrap();
            for (b, tape) in tapes.iter().enumerate().rev() {
                let mut e = half;
                reference.reverse(b, tape, &mut g, &ranges, if b == 1 { Some(&mut e) } else { None }, &mut reference_gradient).unwrap();
            }
        }
        {
            let mut stream = host.zeros(24, 8).unwrap();
            let mut tapes = Vec::new();
            for b in 0..4 {
                let mut e = half;
                tapes.push(decoder.forward(b, &mut stream, &ranges, &sequences, if b == 1 { Some(&mut e) } else { None }, true).unwrap().unwrap());
            }
            let mut g = host.copy(&cotangent).unwrap();
            for (b, tape) in tapes.iter().enumerate().rev() {
                let mut e = half;
                decoder.reverse(b, tape, &mut g, &ranges, if b == 1 { Some(&mut e) } else { None }, &mut decoder_gradient).unwrap();
            }
        }
        assert_eq!(reference_gradient.keys().collect::<Vec<_>>(), decoder_gradient.keys().collect::<Vec<_>>(), "the same operators receive gradients");
        for (op, r) in &reference_gradient {
            let (r, d) = (host.download(r).unwrap(), host.download(&decoder_gradient[op]).unwrap());
            let scale = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let difference = r.iter().zip(&d).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(difference <= 8.0 / 256.0 * scale, "operator {} gradient differs by {difference} of {scale}", explanation.artifact.program.operators[*op].name);
        }
    }

    #[test]
    fn the_decoder_runs_a_gelu_library_as_the_reference_engine() {
        for arithmetic in [Arithmetic::Bf16, Arithmetic::F32, Arithmetic::Tf32x3] {
            parity(tiny_export(&format!("decoder_gelu_{arithmetic:?}"), 2), arithmetic);
        }
    }

    #[test]
    fn the_decoder_runs_a_qwen3_library_as_the_reference_engine() {
        for arithmetic in [Arithmetic::Bf16, Arithmetic::F32, Arithmetic::Tf32x3] {
            parity(tiny_qwen3_export(&format!("decoder_qwen3_{arithmetic:?}"), 2), arithmetic);
        }
    }
}
