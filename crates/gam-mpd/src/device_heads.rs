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
    let (mut attends, mut rest) = (Vec::new(), Vec::new());
    for &(argument, op) in terms {
        let fused = matches!(program.nodes[argument], Node::Attend { .. })
            && !excluded.contains(&argument)
            && readers[argument] == [output]
            && dense(program, op) == Some((widths[output], widths[argument]));
        if fused { attends.push((argument, op)) } else { rest.push((argument, op)) }
    }
    let Node::Attend { scale, rotary, causal, .. } = program.nodes[attends.first()?.0] else { return None };
    // Each projection: a one-term affine node with a dense operator on the group's input.
    let mut input = None;
    let mut projection = |n: usize| -> Option<(usize, Option<usize>)> {
        let Node::Affine { terms, bias } = &program.nodes[n] else { return None };
        let [(x, op)] = terms[..] else { return None };
        if excluded.contains(&n) || matches!(program.nodes[x], Node::Feature { .. }) || dense(program, op) != Some((widths[n], widths[x])) {
            return None;
        }
        (*input.get_or_insert(x) == x).then_some((op, *bias))
    };
    let members: BTreeSet<usize> = attends.iter().map(|(a, _)| *a).collect();
    // (key, value) pairs and each one's query heads, in order of appearance.
    let mut keys: Vec<((usize, usize), Vec<(usize, usize, usize)>)> = Vec::new();
    for &(attend, read) in &attends {
        let Node::Attend { query, key, value, scale: s, rotary: r, causal: c } = program.nodes[attend] else { return None };
        if s.value() != scale.value() || r != rotary || c != causal || readers[query] != [attend] {
            return None;
        }
        projection(query)?;
        match keys.iter_mut().find(|(pair, _)| pair.0 == key || pair.1 == value) {
            Some((pair, queries)) if *pair == (key, value) => queries.push((query, attend, read)),
            Some(_) => return None,
            None => keys.push(((key, value), vec![(query, attend, read)])),
        }
    }
    let group = keys[0].1.len();
    let mut nodes = BTreeSet::new();
    for ((key, value), queries) in &keys {
        projection(*key)?;
        projection(*value)?;
        if queries.len() != group || [*key, *value].iter().any(|n| readers[*n].iter().any(|r| !members.contains(r))) {
            return None;
        }
        nodes.extend([*key, *value]);
        nodes.extend(queries.iter().map(|q| q.0));
    }
    // A node is one projection only (a key never doubles as a value or a query).
    if nodes.len() != keys.len() * (group + 2) {
        return None;
    }
    let width = widths[keys[0].0.0];
    let queries: Vec<(usize, usize, usize)> = keys.iter().flat_map(|(_, q)| q.iter().copied()).collect();
    let projections: Vec<usize> = queries.iter().map(|q| q.0).chain(keys.iter().map(|(p, _)| p.0)).chain(keys.iter().map(|(p, _)| p.1)).collect();
    if projections.iter().any(|&p| widths[p] != width) {
        return None;
    }
    if rotary.is_some_and(|r| 2 * r.pairs().len() > width) {
        return None;
    }
    let projection_operators = projections.iter().map(|&p| projection(p)).collect::<Option<Vec<_>>>()?;
    Some(Heads {
        input: input?,
        output,
        projections,
        projection_operators,
        attends: queries.iter().map(|q| (q.1, q.2)).collect(),
        rest,
        bias: *bias,
        heads: queries.len(),
        keys: keys.len(),
        width,
        scale: scale.value(),
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
    let rows = p.rows();
    let mut a = d.zeros(rows, heads.heads * heads.width)?;
    let step = step(heads, blocks, rows / blocks);
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
pub(crate) fn backward(
    d: &Device,
    heads: &Heads,
    p: &Tensor,
    g_a: &Tensor,
    blocks: usize,
    turn: Turn<'_>,
    (forward, arithmetic): (Arithmetic, Arithmetic),
) -> Result<Tensor, GpuError> {
    let rows = p.rows();
    let (w, g) = (heads.width, heads.group());
    let mut g_p = d.zeros(rows, heads.columns())?;
    let step = step(heads, blocks, rows / blocks);
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
