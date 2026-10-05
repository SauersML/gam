//! Sibling attention heads executed as one computation on the device (#2951).
//!
//! An attention layer imported head by head ([`crate::import`]) is an affine node per projection
//! (each query head's query, each key head's key and value, all reading one node `h`), an `Attend`
//! per query head, and one affine node reading every head through its own output operator. [`find`]
//! matches that pattern when a program is lowered. The projections' operators are stacked into
//! one `W` (the queries grouped by the key head they read, then the keys, then the values) and the
//! output operators set side by side into one `O`. A forward is then one product `P = h Wᵀ` (plus
//! the stacked biases); the queries, keys and values copied head-major (each sequence's heads, each
//! head's positions), turned by the rotation in the same pass; the scores and softmax of every
//! (sequence, key head) block, whose rows are the positions of that key head's query heads, as one
//! strided-batched product; the values read the same way; the heads merged back side by side into
//! `A`; and the output node's `Σ other terms + A Oᵀ + bias`.
//!
//! The IR is unchanged. Every projection's value is a column block of `P` and every head's read a
//! column block of `A`; the trace copies a block out only when something reads that node. The
//! reverse pass runs the same way backwards from the output node's cotangent.
//!
//! A group matches when every query, key and value is a one-term affine node with a dense operator
//! on the same `h`. Each query is read only by its own attend, and keys and values only by the
//! group's attends. Keys and values pair one to one, every key head is read by equally many query
//! heads, every attend has the same scale, rotation and mask, and each attend is read once, by the
//! output node. Anything else (a head replaced by a rule, a head read through a mask) runs node by
//! node, and so does a group whose operators a training step replaces.

use super::operator_program::{Node, OperatorBody, OperatorProgram, Rotary};
use gam_gpu::gpu_error::GpuError;
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use std::collections::BTreeSet;

/// The most attention weights a fused group forms at once (sequences × heads × positions²); more
/// key heads run in turn.
const SCORES: usize = 32 * 1024 * 1024;

/// A matched group of sibling heads (module note), in node indices and operator indices.
#[derive(Clone, Debug)]
pub(crate) struct Heads {
    /// The node every projection reads.
    pub input: usize,
    /// The affine node reading every head.
    pub output: usize,
    /// The projection nodes in `P`'s column order: queries, keys, values.
    pub projections: Vec<usize>,
    /// Their operators and biases, in the same order.
    pub projection_operators: Vec<(usize, Option<usize>)>,
    /// The attend nodes in the queries' order, and the output operator each is read through.
    pub attends: Vec<(usize, usize)>,
    /// The output node's other terms, in their order.
    pub rest: Vec<(usize, usize)>,
    /// The output node's bias.
    pub bias: Option<usize>,
    /// Query heads, key heads and the width of each.
    pub heads: usize,
    pub keys: usize,
    pub width: usize,
    pub scale: f64,
    pub rotary: Option<Rotary>,
    pub causal: bool,
}

impl Heads {
    /// The first projection: the group computes there.
    pub fn first(&self) -> usize {
        self.projections.iter().copied().min().unwrap_or(self.output)
    }

    /// Every node whose value is a column block of `P` or `A`.
    pub fn members(&self) -> impl Iterator<Item = usize> + '_ {
        self.projections.iter().copied().chain(self.attends.iter().map(|(a, _)| *a))
    }

    /// Query heads per key head.
    pub fn group(&self) -> usize {
        self.heads / self.keys
    }

    /// `P`'s columns.
    pub fn columns(&self) -> usize {
        (self.heads + 2 * self.keys) * self.width
    }

    /// Every operator the stacked copies are made of.
    pub fn operators(&self) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        for (op, bias) in &self.projection_operators {
            out.insert(*op);
            out.extend(bias);
        }
        out.extend(self.attends.iter().map(|(_, op)| *op));
        out
    }

    /// Node `n`'s column block: in `P` (false) or `A` (true), and its first column.
    pub fn block(&self, n: usize) -> Option<(bool, usize)> {
        if let Some(i) = self.projections.iter().position(|&p| p == n) {
            return Some((false, i * self.width));
        }
        self.attends.iter().position(|(a, _)| *a == n).map(|i| (true, i * self.width))
    }
}

/// A dense operator's shape, when it is held as a dense matrix (not a diagonal fast path).
fn dense(program: &OperatorProgram, op: usize) -> Option<(usize, usize)> {
    let operator = &program.operators[op];
    match operator.body {
        OperatorBody::Dense { .. } if operator.diagonal().is_none() => Some((operator.rows.width(), operator.cols.width())),
        _ => None,
    }
}

/// Every group of `program` (module note); `readers[n]` lists the nodes reading node `n`, `widths[n]`
/// its width, and nodes in `excluded` (the streamed head) belong to none.
pub(crate) fn find(program: &OperatorProgram, readers: &[Vec<usize>], widths: &[usize], excluded: &[usize]) -> Vec<Heads> {
    (0..program.nodes.len()).filter(|o| !excluded.contains(o)).filter_map(|o| matched(program, readers, widths, excluded, o)).collect()
}

fn matched(program: &OperatorProgram, readers: &[Vec<usize>], widths: &[usize], excluded: &[usize], output: usize) -> Option<Heads> {
    let Node::Affine { terms, bias } = &program.nodes[output] else { return None };
    // A projection: a one-term affine node with a dense operator, its input, operator and bias.
    let projection = |n: usize| -> Option<(usize, usize, Option<usize>)> {
        let Node::Affine { terms, bias } = &program.nodes[n] else { return None };
        let [(x, op)] = terms[..] else { return None };
        let plain = !excluded.contains(&n) && !matches!(program.nodes[x], Node::Feature { .. }) && dense(program, op) == Some((widths[n], widths[x]));
        plain.then_some((x, op, *bias))
    };
    // Heads: attends read once, only by this node, through a dense operator, on projections of one input.
    let mut first = None;
    let mut keys: Vec<((usize, usize), Vec<(usize, usize, usize)>)> = Vec::new();
    for &(attend, read) in terms {
        let Node::Attend { query, key, value, scale, rotary, causal } = program.nodes[attend] else { continue };
        let inputs: Option<Vec<usize>> = [query, key, value].iter().map(|n| projection(*n).map(|p| p.0)).collect();
        let head = !excluded.contains(&attend)
            && readers[attend] == [output]
            && readers[query] == [attend]
            && dense(program, read) == Some((widths[output], widths[attend]))
            && inputs.is_some_and(|i| i.iter().all(|x| *x == i[0]));
        if !head {
            continue;
        }
        let shape = (projection(query).map(|p| p.0), scale.value().to_bits(), rotary, causal);
        if *first.get_or_insert(shape) != shape {
            continue;
        }
        match keys.iter_mut().find(|(pair, _)| pair.0 == key || pair.1 == value) {
            Some((pair, queries)) if *pair == (key, value) => queries.push((query, attend, read)),
            Some(_) => return None,
            None => keys.push(((key, value), vec![(query, attend, read)])),
        }
    }
    // A key head whose key or value another node reads stays node by node, with its queries.
    let fused: BTreeSet<usize> = keys.iter().flat_map(|(_, q)| q.iter().map(|x| x.1)).collect();
    keys.retain(|((key, value), _)| [*key, *value].iter().all(|n| readers[*n].iter().all(|r| fused.contains(r))));
    let (input, scale, rotary, causal) = first?;
    let group = keys.first()?.1.len();
    let queries: Vec<(usize, usize, usize)> = keys.iter().flat_map(|(_, q)| q.iter().copied()).collect();
    let projections: Vec<usize> = queries.iter().map(|q| q.0).chain(keys.iter().map(|(p, _)| p.0)).chain(keys.iter().map(|(p, _)| p.1)).collect();
    let width = widths[projections[0]];
    // Equally many queries per key head, every projection one node of one width.
    if keys.iter().any(|(_, q)| q.len() != group)
        || projections.iter().collect::<BTreeSet<_>>().len() != projections.len()
        || projections.iter().any(|&p| widths[p] != width)
        || rotary.is_some_and(|r| 2 * r.pairs().len() > width)
    {
        return None;
    }
    let attends: Vec<(usize, usize)> = queries.iter().map(|q| (q.1, q.2)).collect();
    Some(Heads {
        input: input?,
        output,
        projection_operators: projections.iter().map(|&p| projection(p).map(|(_, op, b)| (op, b))).collect::<Option<Vec<_>>>()?,
        projections,
        rest: terms.iter().copied().filter(|(a, _)| !attends.iter().any(|(b, _)| a == b)).collect(),
        attends,
        bias: *bias,
        heads: queries.len(),
        keys: keys.len(),
        width,
        scale: f64::from_bits(scale),
        rotary,
        causal,
    })
}

/// A group's stacked operators on the device: `W` ((heads + 2 keys)·width × the input's width), the
/// projections' biases as one row (zero where a projection has none) when any has one, and `O`
/// (the output's width × heads·width).
pub(crate) struct Stacked {
    pub weights: Tensor,
    pub biases: Option<Tensor>,
    pub reads: Tensor,
}

impl Stacked {
    pub fn upload(device: &Device, program: &OperatorProgram, heads: &Heads) -> Result<Self, GpuError> {
        let matrix = |op: usize| program.operators[op].matrix_cow();
        let blocks: Vec<_> = heads.projection_operators.iter().map(|(op, _)| matrix(*op)).collect();
        let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
        let weights = ndarray::concatenate(ndarray::Axis(0), &views).map_err(|e| GpuError::DriverCallFailed { reason: e.to_string() })?;
        let biases = if heads.projection_operators.iter().any(|(_, b)| b.is_some()) {
            let mut row = Vec::with_capacity(heads.columns());
            for (_, bias) in &heads.projection_operators {
                match bias {
                    Some(b) => row.extend(matrix(*b).iter().copied()),
                    None => row.extend(std::iter::repeat_n(0.0, heads.width)),
                }
            }
            Some(device.upload_vec(1, row.len(), row)?)
        } else {
            None
        };
        let reads: Vec<_> = heads.attends.iter().map(|(_, op)| matrix(*op)).collect();
        let views: Vec<_> = reads.iter().map(|b| b.view()).collect();
        let reads = ndarray::concatenate(ndarray::Axis(1), &views).map_err(|e| GpuError::DriverCallFailed { reason: e.to_string() })?;
        Ok(Self { weights: device.upload(weights.view())?, biases, reads: device.upload(reads.view())? })
    }

    /// Values held.
    pub fn len(&self) -> usize {
        self.weights.len() + self.biases.as_ref().map_or(0, Tensor::len) + self.reads.len()
    }
}

/// The rotation tables of a group's positions (`rows × planes` each) and its pairing.
pub(crate) type Turn<'a> = Option<(&'a Tensor, &'a Tensor, bool)>;

/// Key heads at a time, for `blocks` sequences of `length`: every one when the weights fit.
fn step(heads: &Heads, blocks: usize, length: usize) -> usize {
    let per_key = blocks.saturating_mul(heads.group()).saturating_mul(length).saturating_mul(length).max(1);
    (SCORES / per_key).clamp(1, heads.keys)
}

/// The head-major copies of `n` heads of width `width` from column `start` of `x`: row
/// `(b·n + h)·L + l` is row `b·L + l`'s columns `start + h·width ..`, turned by `turn`.
fn split_heads(d: &Device, x: &Tensor, start: usize, n: usize, width: usize, blocks: usize, turn: Turn<'_>, inverse: bool) -> Result<Tensor, GpuError> {
    let length = x.rows() / blocks;
    let mut out = d.zeros(x.rows() * n, width)?;
    for h in 0..n {
        let mut part = d.columns_of(x, start + h * width..start + (h + 1) * width)?;
        if let Some((cos, sin, half_split)) = turn {
            part = d.rotate(&part, cos, sin, half_split, inverse)?;
        }
        for b in 0..blocks {
            d.set_rows(&mut out, (b * n + h) * length, &d.rows_of(&part, b * length, length)?)?;
        }
    }
    Ok(out)
}

/// [`split_heads`]' inverse: the head-major `x` written into columns `start..` of `out`, turned by
/// `turn` backwards when `inverse`.
fn merge_heads(d: &Device, x: &Tensor, out: &mut Tensor, start: usize, n: usize, blocks: usize, turn: Turn<'_>, inverse: bool) -> Result<(), GpuError> {
    let (rows, width) = (out.rows(), x.cols());
    let length = rows / blocks;
    for h in 0..n {
        let mut part = d.zeros(rows, width)?;
        for b in 0..blocks {
            d.set_rows(&mut part, b * length, &d.rows_of(x, (b * n + h) * length, length)?)?;
        }
        if let Some((cos, sin, half_split)) = turn {
            part = d.rotate(&part, cos, sin, half_split, inverse)?;
        }
        d.set_columns(out, start + h * width, &part)?;
    }
    Ok(())
}

/// Key heads `first..first + n`'s queries, keys and values from `P`, head-major and turned.
fn split(d: &Device, heads: &Heads, p: &Tensor, (first, n): (usize, usize), blocks: usize, turn: Turn<'_>) -> Result<(Tensor, Tensor, Tensor), GpuError> {
    let (w, g) = (heads.width, heads.group());
    Ok((
        split_heads(d, p, first * g * w, n * g, w, blocks, turn, false)?,
        split_heads(d, p, (heads.heads + first) * w, n, w, blocks, turn, false)?,
        split_heads(d, p, (heads.heads + heads.keys + first) * w, n, w, blocks, None, false)?,
    ))
}

/// The weights `softmax(c q kᵀ)` of `batch` (sequence, key head) blocks.
fn weights(d: &Device, heads: &Heads, (q, k): (&Tensor, &Tensor), batch: usize, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    let length = k.rows() / batch;
    let mut scores = d.zeros(q.rows(), length)?;
    d.gemm_batched(batch, &mut scores, heads.scale, q, Op::N, k, Op::T, 0.0, arithmetic)?;
    d.softmax_rows(&mut scores, heads.causal)?;
    Ok(scores)
}

/// `P = x Wᵀ` plus the stacked biases, `x` the input's value (module note).
pub(crate) fn project(d: &Device, heads: &Heads, stacked: &Stacked, x: &Tensor, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    let mut p = d.zeros(x.rows(), heads.columns())?;
    d.gemm(&mut p, 1.0, x, Op::N, &stacked.weights, Op::T, 0.0, arithmetic)?;
    if let Some(b) = &stacked.biases {
        d.add_row(&mut p, 1.0, b)?;
    }
    Ok(p)
}

/// `A`, every head's read, from `P`, `blocks` sequences (module note).
pub(crate) fn attend(d: &Device, heads: &Heads, p: &Tensor, blocks: usize, turn: Turn<'_>, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    attend_by(d, heads, p, blocks, turn, arithmetic, step(heads, blocks, p.rows() / blocks))
}

/// [`attend`], `step` key heads at a time.
fn attend_by(d: &Device, heads: &Heads, p: &Tensor, blocks: usize, turn: Turn<'_>, arithmetic: Arithmetic, step: usize) -> Result<Tensor, GpuError> {
    let rows = p.rows();
    let mut a = d.zeros(rows, heads.heads * heads.width)?;
    for first in (0..heads.keys).step_by(step) {
        let n = step.min(heads.keys - first);
        let (q, k, v) = split(d, heads, p, (first, n), blocks, turn)?;
        let alpha = weights(d, heads, (&q, &k), blocks * n, arithmetic)?;
        let mut out = d.zeros(q.rows(), heads.width)?;
        d.gemm_batched(blocks * n, &mut out, 1.0, &alpha, Op::N, &v, Op::N, 0.0, arithmetic)?;
        merge_heads(d, &out, &mut a, first * heads.group() * heads.width, n * heads.group(), blocks, None, false)?;
    }
    Ok(a)
}

/// The cotangent of `P` given `A`'s, `g_a`; the weights are recomputed in the forward's arithmetic,
/// `forward`, the products run in `arithmetic`.
pub(crate) fn backward(d: &Device, heads: &Heads, p: &Tensor, g_a: &Tensor, blocks: usize, turn: Turn<'_>, arithmetic: (Arithmetic, Arithmetic)) -> Result<Tensor, GpuError> {
    backward_by(d, heads, (p, g_a), blocks, turn, arithmetic, step(heads, blocks, p.rows() / blocks))
}

/// [`backward`], `step` key heads at a time.
fn backward_by(
    d: &Device,
    heads: &Heads,
    (p, g_a): (&Tensor, &Tensor),
    blocks: usize,
    turn: Turn<'_>,
    (forward, arithmetic): (Arithmetic, Arithmetic),
    step: usize,
) -> Result<Tensor, GpuError> {
    let rows = p.rows();
    let (w, g) = (heads.width, heads.group());
    let mut g_p = d.zeros(rows, heads.columns())?;
    for first in (0..heads.keys).step_by(step) {
        let n = step.min(heads.keys - first);
        let batch = blocks * n;
        let (q, k, v) = split(d, heads, p, (first, n), blocks, turn)?;
        let cot = split_heads(d, g_a, first * g * w, n * g, w, blocks, None, false)?;
        let alpha = weights(d, heads, (&q, &k), batch, forward)?;
        let mut dalpha = d.zeros(alpha.rows(), alpha.cols())?;
        d.gemm_batched(batch, &mut dalpha, 1.0, &cot, Op::N, &v, Op::T, 0.0, arithmetic)?;
        let mut gv = d.zeros(v.rows(), w)?;
        d.gemm_batched(batch, &mut gv, 1.0, &alpha, Op::T, &cot, Op::N, 0.0, arithmetic)?;
        let ds = d.softmax_backward(&alpha, &dalpha)?;
        drop((alpha, dalpha));
        let mut gq = d.zeros(q.rows(), w)?;
        d.gemm_batched(batch, &mut gq, heads.scale, &ds, Op::N, &k, Op::N, 0.0, arithmetic)?;
        let mut gk = d.zeros(k.rows(), w)?;
        d.gemm_batched(batch, &mut gk, heads.scale, &ds, Op::T, &q, Op::N, 0.0, arithmetic)?;
        merge_heads(d, &gq, &mut g_p, first * g * w, n * g, blocks, turn, true)?;
        merge_heads(d, &gk, &mut g_p, (heads.heads + first) * w, n, blocks, turn, true)?;
        merge_heads(d, &gv, &mut g_p, (heads.heads + heads.keys + first) * w, n, blocks, None, false)?;
    }
    Ok(g_p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_program::DeviceProgram;
    use crate::device_program_tests::{devices, noise};
    use crate::operator_program::{Declarations, FamilyInputs, Interface, Operator, Scale, SequenceLayout, Slot, SlotValues, exact_precision};
    use ndarray::Array2;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    const D: usize = 6;
    const WIDTH: usize = 4;
    const LENGTH: usize = 5;
    const SEQUENCES: usize = 3;

    fn dense(name: &str, rows: usize, cols: usize, seed: usize) -> Arc<Operator> {
        let values = Array2::from_shape_fn((rows, cols), |(i, j)| 0.6 * noise(seed * 1000 + i * cols + j));
        let rows = Interface::native(rows).expect("rows");
        let cols = if cols == 1 { Interface::constant() } else { Interface::native(cols).expect("cols") };
        Arc::new(Operator::dense(name, rows, cols, values.clone(), exact_precision(values.iter().copied()).expect("exact"), Default::default()).expect("dense"))
    }

    /// One attention layer of `heads` query heads over `keys` key heads (each key head's queries
    /// interleaved in the output node's terms), biased queries and keys, on a raw stream lifted
    /// by one dense map, and a nonlinear read of the output.
    fn layer(heads: usize, keys: usize, rotary: Option<Rotary>, causal: bool) -> (OperatorProgram, FamilyInputs) {
        let mut operators = vec![Arc::new(Operator::identity("I", Interface::native(D).expect("d"))), dense("lift", D, D, 1)];
        let mut nodes = vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 1)], bias: None }];
        let mut op = |o: Arc<Operator>| {
            operators.push(o);
            operators.len() - 1
        };
        let mut pairs = Vec::new();
        for g in 0..keys {
            let (k, kb, v) = (op(dense("k", WIDTH, D, 10 + g)), op(dense("kb", WIDTH, 1, 20 + g)), op(dense("v", WIDTH, D, 30 + g)));
            nodes.push(Node::Affine { terms: vec![(1, k)], bias: Some(kb) });
            nodes.push(Node::Affine { terms: vec![(1, v)], bias: None });
            pairs.push((nodes.len() - 2, nodes.len() - 1));
        }
        let mut terms = vec![(1, 0)];
        for h in 0..heads {
            let (q, qb, o) = (op(dense("q", WIDTH, D, 40 + h)), op(dense("qb", WIDTH, 1, 50 + h)), op(dense("o", D, WIDTH, 60 + h)));
            nodes.push(Node::Affine { terms: vec![(1, q)], bias: Some(qb) });
            let (key, value) = pairs[h % keys];
            nodes.push(Node::Attend { query: nodes.len() - 1, key, value, scale: Scale::InverseSqrt(WIDTH as u32), rotary, causal });
            terms.push((nodes.len() - 1, o));
        }
        let ob = op(dense("ob", D, 1, 70));
        nodes.push(Node::Affine { terms, bias: Some(ob) });
        nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![crate::operator_program::Law::GeluTanh; D] });
        let output = nodes.len() - 1;
        let program = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: D }], parameters: 0 },
            bases: vec![],
            rules: vec![],
            operators,
            nodes,
            output,
        };
        let rows = SEQUENCES * LENGTH;
        let x = Array2::from_shape_fn((rows, D), |(r, c)| noise(9000 + r * D + c));
        let layout = SequenceLayout { sequence: (0..rows).map(|r| (r / LENGTH) as u32).collect(), position: (0..rows).map(|r| (r % LENGTH + 2) as u32).collect() };
        (program, FamilyInputs { rows, slots: vec![SlotValues::Raw(x)], layout: Some(layout) })
    }

    /// [`layer`] with head `masked` read through a raw mask (slot 1) by the output node.
    fn masked(heads: usize, keys: usize, masked: usize) -> (OperatorProgram, FamilyInputs) {
        let (mut program, mut family) = layer(heads, keys, None, true);
        let (output, law) = (program.output - 1, program.nodes.pop().expect("law"));
        let mut node = program.nodes.pop().expect("output");
        program.declarations.slots.push(Slot::Raw { width: WIDTH });
        program.nodes.push(Node::Raw { slot: 1 });
        let Node::Affine { terms, .. } = &mut node else { return (program, family) };
        program.nodes.push(Node::Hadamard { left: terms[masked + 1].0, right: output });
        terms[masked + 1].0 = output + 1;
        program.nodes.push(node);
        let Node::Pointwise { laws, .. } = law else { return (program, family) };
        program.nodes.push(Node::Pointwise { input: output + 2, laws });
        program.output = output + 3;
        family.slots.push(SlotValues::Raw(Array2::from_shape_fn((family.rows, WIDTH), |(r, c)| noise(8000 + r * WIDTH + c))));
        (program, family)
    }

    fn worst(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        assert_eq!(a.dim(), b.dim());
        a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn fused_heads_match_node_by_node_values_cotangents_and_edits() {
        let rotations = [None, Some(Rotary { base: 10_000, dims: 4, half_split: true }), Some(Rotary { base: 500, dims: 2, half_split: false })];
        for (heads, keys) in [(4, 2), (3, 3), (2, 1)] {
            for rotary in rotations {
                for causal in [true, false] {
                    let (program, family) = layer(heads, keys, rotary, causal);
                    let cpu = program.execute(&family, false).expect("cpu");
                    let seed = Array2::from_shape_fn((family.rows, D), |(r, c)| noise(7000 + r * D + c));
                    let reference = crate::derivatives::vjp(&program, &family, &cpu, seed.clone()).expect("cpu vjp");
                    for device in devices() {
                        let fused = DeviceProgram::compile_values(&device, &program).expect("fused");
                        assert_eq!(fused.fused_groups(), 1, "{heads} heads over {keys} keys form one group");
                        let mut plain = DeviceProgram::compile_values_sharing(&fused, &program).expect("plain");
                        plain.unfuse();
                        let (a, b) = (fused.forward(&family).expect("fused forward"), plain.forward(&family).expect("plain forward"));
                        for node in 0..program.nodes.len() {
                            let (x, y) = (device.download(a.value(node).expect("fused value")).expect("x"), device.download(b.value(node).expect("plain value")).expect("y"));
                            assert!(worst(&x, &y) < 1e-12, "{}: node {node} differs by {:e}", device.name(), worst(&x, &y));
                            assert!(worst(&x, &cpu.values[node]) < 1e-12, "{}: node {node} against the CPU", device.name());
                        }
                        // The reverse pass from the output to the raw input: fused, it never reads a member.
                        let seeds = |d: &Device| BTreeMap::from([(program.output, d.upload(seed.view()).expect("seed"))]);
                        let g = fused.vjp_values_seeded(&a, seeds(&device), &[0, 1], Arithmetic::F64).expect("fused vjp");
                        let h = plain.vjp_values_seeded(&b, seeds(&device), &[0, 1], Arithmetic::F64).expect("plain vjp");
                        for node in [0, 1] {
                            let (x, y) = (device.download(&g[&node]).expect("x"), device.download(&h[&node]).expect("y"));
                            assert!(worst(&x, &y) < 1e-12, "{}: cotangent of node {node}", device.name());
                            assert!(worst(&x, reference[node].as_ref().expect("cpu cotangent")) < 1e-12, "{}: cotangent of node {node} against the CPU", device.name());
                        }
                        // Edits at a query and at a head are written back before the attention and the output read them.
                        let query = program.nodes.iter().position(|n| matches!(n, Node::Attend { .. })).map(|a| a - 1).expect("a query");
                        let read = program.nodes.iter().rposition(|n| matches!(n, Node::Attend { .. })).expect("a head");
                        let edit = |node: usize, trace: &crate::device_program::DeviceTrace| -> Result<Option<Tensor>, String> {
                            if node != query && node != read {
                                return Ok(None);
                            }
                            let mut doubled = device.copy(trace.value(node)?).map_err(|e| e.to_string())?;
                            device.axpy(&mut doubled, if node == query { 1.0 } else { -0.5 }, trace.value(node)?).map_err(|e| e.to_string())?;
                            Ok(Some(doubled))
                        };
                        let none = Default::default();
                        let x = fused.forward_edited(&family, BTreeMap::new(), &none, |_, _| Ok(()), edit).expect("fused edited");
                        let y = plain.forward_edited(&family, BTreeMap::new(), &none, |_, _| Ok(()), edit).expect("plain edited");
                        let expected = program
                            .execute_edited(&family, |node, value, _| {
                                if node == query {
                                    *value *= 2.0;
                                } else if node == read {
                                    *value *= 0.5;
                                }
                                Ok(())
                            })
                            .expect("cpu edited");
                        let (x, y) = (device.download(x.value(program.output).expect("x")).expect("x"), device.download(y.value(program.output).expect("y")).expect("y"));
                        assert!(worst(&x, &y) < 1e-12 && worst(&x, &expected.values[program.output]) < 1e-12, "{}: edited output", device.name());
                        assert!(worst(&x, &cpu.values[program.output]) > 1e-3, "the edits change the output");
                    }
                }
            }
        }
    }

    #[test]
    fn key_heads_in_turn_match_all_at_once() {
        let rotary = Some(Rotary { base: 10_000, dims: 4, half_split: true });
        let (program, family) = layer(4, 2, rotary, true);
        let readers = {
            let mut r = vec![Vec::new(); program.nodes.len()];
            for (i, n) in program.nodes.iter().enumerate() {
                for a in n.arguments() {
                    r[a].push(i);
                }
            }
            r
        };
        let widths: Vec<usize> = program.interfaces().expect("interfaces").iter().map(|i| i.width()).collect();
        let [heads] = &find(&program, &readers, &widths, &[])[..] else { panic!("one group") };
        let cpu = program.execute(&family, false).expect("cpu");
        let rows = family.rows;
        let layout = family.layout.as_ref().expect("layout");
        let r = rotary.expect("rotary");
        let planes = r.pairs().len();
        let table = |sine: bool| Array2::from_shape_fn((rows, planes), |(row, plane)| { let (c, s) = r.turn(plane, layout.position[row]); if sine { s } else { c } });
        for d in devices() {
            let stacked = Stacked::upload(&d, &program, heads).expect("stacked");
            let (cos, sin) = (d.upload(table(false).view()).expect("cos"), d.upload(table(true).view()).expect("sin"));
            let turn = Some((&cos, &sin, r.half_split));
            let p = project(&d, heads, &stacked, &d.upload(cpu.values[heads.input].view()).expect("x"), Arithmetic::F64).expect("P");
            let g_a = d.upload(Array2::from_shape_fn((rows, heads.heads * heads.width), |(i, j)| noise(3000 + i * 31 + j)).view()).expect("g_A");
            let once = d.download(&attend_by(&d, heads, &p, SEQUENCES, turn, Arithmetic::F64, heads.keys).expect("once")).expect("once");
            let turns = d.download(&attend_by(&d, heads, &p, SEQUENCES, turn, Arithmetic::F64, 1).expect("in turn")).expect("in turn");
            assert!(worst(&once, &turns) < 1e-13, "{}: reads", d.name());
            let arithmetic = (Arithmetic::F64, Arithmetic::F64);
            let once = d.download(&backward_by(&d, heads, (&p, &g_a), SEQUENCES, turn, arithmetic, heads.keys).expect("once")).expect("once");
            let turns = d.download(&backward_by(&d, heads, (&p, &g_a), SEQUENCES, turn, arithmetic, 1).expect("in turn")).expect("in turn");
            assert!(worst(&once, &turns) < 1e-13, "{}: cotangents", d.name());
        }
    }

    #[test]
    fn a_head_read_elsewhere_or_through_a_mask_runs_node_by_node() {
        let (mut program, _) = layer(2, 1, None, true);
        let output = program.output - 1;
        let Node::Affine { terms, .. } = &program.nodes[output] else { panic!("output affine") };
        let read = terms[1].0;
        let readers = |p: &OperatorProgram| {
            let mut r = vec![Vec::new(); p.nodes.len()];
            for (i, n) in p.nodes.iter().enumerate() {
                for a in n.arguments() {
                    r[a].push(i);
                }
            }
            r
        };
        let widths: Vec<usize> = program.interfaces().expect("interfaces").iter().map(|i| i.width()).collect();
        assert_eq!(find(&program, &readers(&program), &widths, &[]).len(), 1);
        // A second reader of one head: its key head (shared with the other head) runs node by node.
        program.nodes.push(Node::Concat { parts: vec![read] });
        let widths: Vec<usize> = program.interfaces().expect("interfaces").iter().map(|i| i.width()).collect();
        assert!(find(&program, &readers(&program), &widths, &[]).is_empty());
        // Over two key heads, the other key head's queries stay fused.
        let (mut program, _) = layer(4, 2, None, true);
        let Node::Affine { terms, .. } = &program.nodes[program.output - 1] else { panic!("output affine") };
        let read = terms[1].0;
        program.nodes.push(Node::Concat { parts: vec![read] });
        let widths: Vec<usize> = program.interfaces().expect("interfaces").iter().map(|i| i.width()).collect();
        let groups = find(&program, &readers(&program), &widths, &[]);
        assert_eq!((groups.len(), groups[0].heads, groups[0].keys), (1, 2, 1));
        assert!(groups[0].rest.iter().any(|(a, _)| *a == read));
        // A head read through a mask: its key head runs node by node.
        let (program, family) = masked(4, 2, 2);
        let widths: Vec<usize> = program.interfaces().expect("interfaces").iter().map(|i| i.width()).collect();
        let groups = find(&program, &readers(&program), &widths, &[]);
        assert_eq!((groups.len(), groups[0].heads), (1, 2));
        let cpu = program.execute(&family, false).expect("cpu");
        for device in devices() {
            let fused = DeviceProgram::compile_values(&device, &program).expect("fused");
            let trace = fused.forward(&family).expect("forward");
            let value = device.download(trace.value(program.output).expect("output")).expect("download");
            assert!(worst(&value, &cpu.values[program.output]) < 1e-12, "{}: masked head", device.name());
        }
    }
}
