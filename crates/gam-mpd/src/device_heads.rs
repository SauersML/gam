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
//! `A`; and the output node's `Σ other terms + A Oᵀ + bias`. When every query and key is normed per
//! head before it is read (Qwen3's q_norm and k_norm: an RMS norm, then a diagonal gain), the norm
//! runs on all of them at once (`N`, each head's columns a row of its own) and the gains as one
//! row of column scales (`G`), whose queries and keys the attention then reads.
//!
//! The IR is unchanged. Every projection's value is a column block of `P`, every norm's of `N`,
//! every gain's of `G` and every head's read a column block of `A`; the trace copies a block out
//! only when something reads that node. The reverse pass runs the same way backwards from the
//! output node's cotangent.
//!
//! A group matches when every query, key and value is a one-term affine node with a dense operator
//! on the same `h` (a query or key possibly normed as above). Each query is read only by its own
//! attend, and keys and values only by the group's attends. Keys and values pair one to one, every
//! key head is read by equally many query heads, every attend has the same scale, rotation and
//! mask, and each attend is read once, by the output node. Anything else (a head replaced by a
//! rule, a head read through a mask) runs node by node. Parameter updates invalidate stacked
//! copies until `DeviceProgram::refresh_fused` restacks the current resident operators.

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
    /// The per-head norms of the queries and keys, when they have them.
    pub norms: Option<Norms>,
    /// The attend nodes in the queries' order, and the output operator each is read through.
    pub attends: Vec<(usize, usize)>,
    /// Per attend, the identity node between it and the output when there is one (an inlined
    /// rule's output barrier, `artifact_device::mapped_inlined`): the same value, the same block.
    pub barriers: Vec<Option<usize>>,
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

/// Every query's and key's RMS norm and gain: per query and key projection (in `P`'s order) the
/// norm node, the node applying the gain and the gain's diagonal operator; the norms' `ε`.
#[derive(Clone, Debug)]
pub(crate) struct Norms {
    pub nodes: Vec<(usize, usize, usize)>,
    pub epsilon: f64,
}

/// The fused buffer a member's value is a column block of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Buffer {
    Projections,
    Normed,
    Gained,
    Reads,
}

impl Heads {
    /// The first projection: the group computes there.
    pub fn first(&self) -> usize {
        self.projections.iter().copied().min().unwrap_or(self.output)
    }

    /// Every node whose value is a column block of a fused buffer.
    pub fn members(&self) -> impl Iterator<Item = usize> + '_ {
        let norms = self.norms.iter().flat_map(|n| n.nodes.iter().flat_map(|(norm, gain, _)| [*norm, *gain]));
        self.projections.iter().copied().chain(norms).chain(self.attends.iter().map(|(a, _)| *a)).chain(self.barriers.iter().flatten().copied())
    }

    /// Query heads per key head.
    pub fn group(&self) -> usize {
        self.heads / self.keys
    }

    /// `P`'s columns.
    pub fn columns(&self) -> usize {
        (self.heads + 2 * self.keys) * self.width
    }

    /// The queries' and keys' columns (`N`'s and `G`'s).
    pub fn normed_columns(&self) -> usize {
        (self.heads + self.keys) * self.width
    }

    /// Every operator the stacked copies are made of.
    pub fn operators(&self) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        for (op, bias) in &self.projection_operators {
            out.insert(*op);
            out.extend(bias);
        }
        out.extend(self.norms.iter().flat_map(|n| n.nodes.iter().map(|(_, _, gain)| *gain)));
        out.extend(self.attends.iter().map(|(_, op)| *op));
        out
    }

    /// The members of buffer `buffer`, in node order.
    pub fn nodes_of(&self, buffer: Buffer) -> Vec<usize> {
        let mut nodes: Vec<usize> = match (buffer, &self.norms) {
            (Buffer::Projections, _) => self.projections.clone(),
            (Buffer::Normed, Some(n)) => n.nodes.iter().map(|x| x.0).collect(),
            (Buffer::Gained, Some(n)) => n.nodes.iter().map(|x| x.1).collect(),
            (Buffer::Reads, _) => self.attends.iter().map(|(a, _)| *a).chain(self.barriers.iter().flatten().copied()).collect(),
            (_, None) => Vec::new(),
        };
        nodes.sort_unstable();
        nodes
    }

    /// Node `n`'s column block: its buffer and first column.
    pub fn block(&self, n: usize) -> Option<(Buffer, usize)> {
        let at = |i: usize| i * self.width;
        if let Some(i) = self.projections.iter().position(|&p| p == n) {
            return Some((Buffer::Projections, at(i)));
        }
        if let Some(norms) = &self.norms {
            if let Some(i) = norms.nodes.iter().position(|x| x.0 == n) {
                return Some((Buffer::Normed, at(i)));
            }
            if let Some(i) = norms.nodes.iter().position(|x| x.1 == n) {
                return Some((Buffer::Gained, at(i)));
            }
        }
        self.attends.iter().zip(&self.barriers).position(|((a, _), b)| *a == n || *b == Some(n)).map(|i| (Buffer::Reads, at(i)))
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

/// A query's or key's norm: its norm node, gain node, gain operator and `ε`'s bits.
type Norm = (usize, usize, usize, u64);

fn matched(program: &OperatorProgram, readers: &[Vec<usize>], widths: &[usize], excluded: &[usize], output: usize) -> Option<Heads> {
    let Node::Affine { terms, bias } = &program.nodes[output] else { return None };
    // A projection: a one-term affine node with a dense operator, its input, operator and bias.
    let projection = |n: usize| -> Option<(usize, usize, Option<usize>)> {
        let Node::Affine { terms, bias } = &program.nodes[n] else { return None };
        let [(x, op)] = terms[..] else { return None };
        let plain = !excluded.contains(&n) && !matches!(program.nodes[x], Node::Feature { .. }) && dense(program, op) == Some((widths[n], widths[x]));
        plain.then_some((x, op, *bias))
    };
    // What an attend reads as a query or key: a projection, or a projection's RMS norm times a
    // diagonal gain, each read only by the next; the projection and the norm.
    let read = |n: usize| -> Option<(usize, Option<Norm>)> {
        if projection(n).is_some() {
            return Some((n, None));
        }
        let Node::Affine { terms, bias: None } = &program.nodes[n] else { return None };
        let [(m, gain)] = terms[..] else { return None };
        let Node::RmsNorm { input: p, epsilon } = program.nodes[m] else { return None };
        let normed = !excluded.contains(&n)
            && !excluded.contains(&m)
            && readers[m] == [n]
            && readers[p] == [m]
            && widths[p] == widths[n]
            && program.operators[gain].diagonal().is_some_and(|d| d.len() == widths[n])
            && projection(p).is_some();
        normed.then_some((p, Some((m, n, gain, epsilon.to_bits()))))
    };
    // An inlined rule's output barrier: an identity node read only by this node, reading an attend
    // read only by it.
    let barrier = |n: usize| -> Option<usize> {
        let Node::Affine { terms, bias: None } = &program.nodes[n] else { return None };
        let [(a, op)] = terms[..] else { return None };
        let identity = matches!(program.operators[op].body, OperatorBody::Identity);
        (identity && !excluded.contains(&n) && readers[n] == [output] && readers[a] == [n] && matches!(program.nodes[a], Node::Attend { .. })).then_some(a)
    };
    // Heads: attends read once, only by this node (or through a barrier), through a dense operator,
    // on projections of one input.
    let mut first = None;
    let mut keys: Vec<((usize, usize), Vec<(usize, usize, usize, Option<usize>)>)> = Vec::new();
    for &(term, through) in terms {
        let (attend, between) = match barrier(term) {
            Some(a) => (a, Some(term)),
            None => (term, None),
        };
        let Node::Attend { query, key, value, scale, rotary, causal } = program.nodes[attend] else { continue };
        let (Some((q, q_norm)), Some((k, k_norm)), Some((v, _))) = (read(query), read(key), projection(value).map(|_| (value, ()))) else { continue };
        let inputs: Option<Vec<usize>> = [q, k, v].iter().map(|n| projection(*n).map(|p| p.0)).collect();
        let head = !excluded.contains(&attend)
            && (between.is_some() || readers[attend] == [output])
            && readers[query] == [attend]
            && dense(program, through) == Some((widths[output], widths[attend]))
            && inputs.as_ref().is_some_and(|i| i.iter().all(|x| *x == i[0]));
        if !head {
            continue;
        }
        let epsilon = |n: Option<Norm>| n.map(|x| x.3);
        let shape = (inputs.and_then(|i| i.first().copied()), scale.value().to_bits(), rotary, causal, epsilon(q_norm), epsilon(k_norm));
        if *first.get_or_insert(shape) != shape {
            continue;
        }
        match keys.iter_mut().find(|(pair, _)| pair.0 == key || pair.1 == value) {
            Some((pair, queries)) if *pair == (key, value) => queries.push((query, attend, through, between)),
            Some(_) => return None,
            None => keys.push(((key, value), vec![(query, attend, through, between)])),
        }
    }
    // A key head whose key or value another node reads stays node by node, with its queries.
    let fused: BTreeSet<usize> = keys.iter().flat_map(|(_, q)| q.iter().map(|x| x.1)).collect();
    keys.retain(|((key, value), _)| [*key, *value].iter().all(|n| readers[*n].iter().all(|r| fused.contains(r))));
    let (input, scale, rotary, causal, q_epsilon, k_epsilon) = first?;
    let group = keys.first()?.1.len();
    let queries: Vec<(usize, usize, usize, Option<usize>)> = keys.iter().flat_map(|(_, q)| q.iter().copied()).collect();
    // The nodes the attends read as queries and keys, then the values, in `P`'s order.
    let reads: Vec<usize> = queries.iter().map(|q| q.0).chain(keys.iter().map(|(p, _)| p.0)).collect();
    let resolved: Vec<(usize, Option<Norm>)> = reads.iter().map(|&n| read(n)).collect::<Option<_>>()?;
    let projections: Vec<usize> = resolved.iter().map(|r| r.0).chain(keys.iter().map(|(p, _)| p.1)).collect();
    let width = widths[projections[0]];
    // Equally many queries per key head, every projection one node of one width, the queries and
    // keys all normed (with one ε) or none.
    let normed = q_epsilon.is_some() && q_epsilon == k_epsilon;
    if keys.iter().any(|(_, q)| q.len() != group)
        || projections.iter().collect::<BTreeSet<_>>().len() != projections.len()
        || projections.iter().any(|&p| widths[p] != width)
        || rotary.is_some_and(|r| 2 * r.pairs().len() > width)
        || (q_epsilon.is_some() || k_epsilon.is_some()) && !normed
    {
        return None;
    }
    let norms = normed.then(|| Norms {
        nodes: resolved.iter().filter_map(|r| r.1.map(|(norm, gain, op, _)| (norm, gain, op))).collect(),
        epsilon: f64::from_bits(q_epsilon.unwrap_or_default()),
    });
    let attends: Vec<(usize, usize)> = queries.iter().map(|q| (q.1, q.2)).collect();
    let barriers: Vec<Option<usize>> = queries.iter().map(|q| q.3).collect();
    let read_terms: Vec<usize> = queries.iter().map(|q| q.3.unwrap_or(q.1)).collect();
    Some(Heads {
        input: input?,
        output,
        projection_operators: projections.iter().map(|&p| projection(p).map(|(_, op, b)| (op, b))).collect::<Option<Vec<_>>>()?,
        projections,
        norms,
        rest: terms.iter().copied().filter(|(a, _)| !read_terms.contains(a)).collect(),
        attends,
        barriers,
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
/// projections' biases as one row (zero where a projection has none) when any has one, the
/// queries' and keys' gains as one row when they are normed, and `O` (the output's width ×
/// heads·width).
pub(crate) struct Stacked {
    pub weights: Tensor,
    pub biases: Option<Tensor>,
    pub gains: Option<Tensor>,
    pub reads: Tensor,
}

impl Stacked {
    pub fn upload(device: &Device, program: &OperatorProgram, heads: &Heads) -> Result<Self, GpuError> {
        let failed = |e: ndarray::ShapeError| GpuError::DriverCallFailed { reason: e.to_string() };
        let matrix = |op: usize| program.operators[op].matrix_cow();
        let blocks: Vec<_> = heads.projection_operators.iter().map(|(op, _)| matrix(*op)).collect();
        let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
        let weights = ndarray::concatenate(ndarray::Axis(0), &views).map_err(failed)?;
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
        let gains = match &heads.norms {
            Some(norms) => {
                let row: Vec<f64> = norms.nodes.iter().flat_map(|(_, _, op)| program.operators[*op].diagonal().map(|d| d.to_vec()).unwrap_or_default()).collect();
                if row.len() != heads.normed_columns() {
                    return Err(GpuError::DriverCallFailed { reason: "a head norm's gain is not its head's diagonal".into() });
                }
                Some(device.upload_vec(1, row.len(), row)?)
            }
            None => None,
        };
        let reads: Vec<_> = heads.attends.iter().map(|(_, op)| matrix(*op)).collect();
        let views: Vec<_> = reads.iter().map(|b| b.view()).collect();
        let reads = ndarray::concatenate(ndarray::Axis(1), &views).map_err(failed)?;
        Ok(Self { weights: device.upload(weights.view())?, biases, gains, reads: device.upload(reads.view())? })
    }

    /// Values held.
    pub fn len(&self) -> usize {
        self.weights.len() + self.biases.as_ref().map_or(0, Tensor::len) + self.gains.as_ref().map_or(0, Tensor::len) + self.reads.len()
    }
}

/// The rotation tables of a group's positions (`rows × planes` each) and its pairing.
pub(crate) type Turn<'a> = Option<(&'a Tensor, &'a Tensor, bool)>;

/// Key heads at a time, for `blocks` sequences of `length`: every one when the weights fit.
fn step(heads: &Heads, blocks: usize, length: usize) -> usize {
    let per_key = blocks.saturating_mul(heads.group()).saturating_mul(length).saturating_mul(length).max(1);
    (SCORES / per_key).clamp(1, heads.keys)
}

/// Key heads `first..first + n`'s queries and keys (from `qk`, queries then keys) and values (from
/// `P`), head-major and turned.
fn split(d: &Device, heads: &Heads, (qk, p): (&Tensor, &Tensor), (first, n): (usize, usize), blocks: usize, turn: Turn<'_>) -> Result<(Tensor, Tensor, Tensor), GpuError> {
    let (w, g) = (heads.width, heads.group());
    Ok((
        d.split_heads(qk, first * g * w, n * g, w, blocks, turn, false)?,
        d.split_heads(qk, (heads.heads + first) * w, n, w, blocks, turn, false)?,
        d.split_heads(p, (heads.heads + heads.keys + first) * w, n, w, blocks, None, false)?,
    ))
}

/// The weights `softmax(c q kᵀ)` of `batch` (sequence, key head) blocks.
fn weights(d: &Device, heads: &Heads, (q, k): (&Tensor, &Tensor), batch: usize, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    let length = k.rows() / batch;
    let mut scores = d.empty(q.rows(), length)?;
    d.gemm_batched(batch, &mut scores, heads.scale, q, Op::N, k, Op::T, 0.0, arithmetic)?;
    d.softmax_rows(&mut scores, heads.causal)?;
    Ok(scores)
}

/// `P = x Wᵀ` plus the stacked biases, `x` the input's value (module note).
pub(crate) fn project(d: &Device, heads: &Heads, stacked: &Stacked, x: &Tensor, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    let mut p = d.empty(x.rows(), heads.columns())?;
    d.gemm(&mut p, 1.0, x, Op::N, &stacked.weights, Op::T, 0.0, arithmetic)?;
    if let Some(b) = &stacked.biases {
        d.add_row(&mut p, 1.0, b)?;
    }
    Ok(p)
}

/// The queries' and keys' columns of `P`, a row per head when `per_head`.
fn queries_and_keys(d: &Device, heads: &Heads, p: &Tensor, per_head: bool) -> Result<Tensor, GpuError> {
    let qk = d.columns_of(p, 0..heads.normed_columns())?;
    if per_head { qk.reshape(p.rows() * (heads.heads + heads.keys), heads.width) } else { Ok(qk) }
}

/// `N`, every query's and key's RMS norm, from `P`.
pub(crate) fn normalize(d: &Device, heads: &Heads, p: &Tensor, epsilon: f64) -> Result<Tensor, GpuError> {
    d.rms_norm(&queries_and_keys(d, heads, p, true)?, epsilon)?.reshape(p.rows(), heads.normed_columns())
}

/// `G`, `N` times every query's and key's gain.
pub(crate) fn gain(d: &Device, stacked: &Stacked, n: &Tensor) -> Result<Tensor, GpuError> {
    let gains = stacked.gains.as_ref().ok_or_else(|| GpuError::DriverCallFailed { reason: "head norms without gains".into() })?;
    let mut g = d.empty(n.rows(), n.cols())?;
    d.scale_columns(&mut g, n, gains, false)?;
    Ok(g)
}

/// `A`, every head's read, from the queries and keys of `qk` (`G` when they are normed, else `P`)
/// and the values of `P`, `blocks` sequences (module note).
pub(crate) fn attend(d: &Device, heads: &Heads, (qk, p): (&Tensor, &Tensor), blocks: usize, turn: Turn<'_>, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    attend_by(d, heads, (qk, p), blocks, turn, arithmetic, step(heads, blocks, p.rows() / blocks))
}

/// [`attend`], `step` key heads at a time.
fn attend_by(d: &Device, heads: &Heads, (qk, p): (&Tensor, &Tensor), blocks: usize, turn: Turn<'_>, arithmetic: Arithmetic, step: usize) -> Result<Tensor, GpuError> {
    let rows = p.rows();
    let mut a = d.zeros(rows, heads.heads * heads.width)?;
    for first in (0..heads.keys).step_by(step) {
        let n = step.min(heads.keys - first);
        let (q, k, v) = split(d, heads, (qk, p), (first, n), blocks, turn)?;
        let alpha = weights(d, heads, (&q, &k), blocks * n, arithmetic)?;
        let mut out = d.empty(q.rows(), heads.width)?;
        d.gemm_batched(blocks * n, &mut out, 1.0, &alpha, Op::N, &v, Op::N, 0.0, arithmetic)?;
        d.merge_heads(&out, &mut a, first * heads.group() * heads.width, n * heads.group(), blocks, None, false)?;
    }
    Ok(a)
}

/// The cotangent of `P` given `A`'s, `g_a`, from `P` and (normed) `G`; the weights are recomputed
/// in the forward's arithmetic, `forward`, the products run in `arithmetic`.
pub(crate) fn backward(
    d: &Device,
    (heads, stacked): (&Heads, &Stacked),
    (p, gained): (&Tensor, Option<&Tensor>),
    g_a: &Tensor,
    blocks: usize,
    turn: Turn<'_>,
    arithmetic: (Arithmetic, Arithmetic),
) -> Result<Tensor, GpuError> {
    backward_by(d, (heads, stacked), (p, gained, g_a), blocks, turn, arithmetic, step(heads, blocks, p.rows() / blocks))
}

/// [`backward`], `step` key heads at a time.
fn backward_by(
    d: &Device,
    (heads, stacked): (&Heads, &Stacked),
    (p, gained, g_a): (&Tensor, Option<&Tensor>, &Tensor),
    blocks: usize,
    turn: Turn<'_>,
    (forward, arithmetic): (Arithmetic, Arithmetic),
    step: usize,
) -> Result<Tensor, GpuError> {
    let rows = p.rows();
    let (w, g) = (heads.width, heads.group());
    let mut g_p = d.zeros(rows, heads.columns())?;
    // The queries' and keys' cotangents go to `G`'s, then back through the gains and norms.
    let mut g_g = match (&heads.norms, gained) {
        (Some(_), Some(_)) => Some(d.zeros(rows, heads.normed_columns())?),
        (None, None) => None,
        _ => return Err(GpuError::DriverCallFailed { reason: "normed heads without their gained values".into() }),
    };
    let qk = gained.unwrap_or(p);
    for first in (0..heads.keys).step_by(step) {
        let n = step.min(heads.keys - first);
        let batch = blocks * n;
        let (q, k, v) = split(d, heads, (qk, p), (first, n), blocks, turn)?;
        let cot = d.split_heads(g_a, first * g * w, n * g, w, blocks, None, false)?;
        let alpha = weights(d, heads, (&q, &k), batch, forward)?;
        let mut dalpha = d.empty(alpha.rows(), alpha.cols())?;
        d.gemm_batched(batch, &mut dalpha, 1.0, &cot, Op::N, &v, Op::T, 0.0, arithmetic)?;
        let mut gv = d.empty(v.rows(), w)?;
        d.gemm_batched(batch, &mut gv, 1.0, &alpha, Op::T, &cot, Op::N, 0.0, arithmetic)?;
        let ds = d.softmax_backward(&alpha, &dalpha)?;
        drop((alpha, dalpha));
        let mut gq = d.empty(q.rows(), w)?;
        d.gemm_batched(batch, &mut gq, heads.scale, &ds, Op::N, &k, Op::N, 0.0, arithmetic)?;
        let mut gk = d.empty(k.rows(), w)?;
        d.gemm_batched(batch, &mut gk, heads.scale, &ds, Op::T, &q, Op::N, 0.0, arithmetic)?;
        let target = g_g.as_mut().unwrap_or(&mut g_p);
        d.merge_heads(&gq, target, first * g * w, n * g, blocks, turn, true)?;
        d.merge_heads(&gk, target, (heads.heads + first) * w, n, blocks, turn, true)?;
        d.merge_heads(&gv, &mut g_p, (heads.heads + heads.keys + first) * w, n, blocks, None, false)?;
    }
    if let (Some(g_g), Some(norms), Some(gains)) = (g_g, &heads.norms, &stacked.gains) {
        let mut g_n = d.empty(rows, heads.normed_columns())?;
        d.scale_columns(&mut g_n, &g_g, gains, false)?;
        let per_head = rows * (heads.heads + heads.keys);
        let g_qk = d.rms_norm_backward(&queries_and_keys(d, heads, p, true)?, &g_n.reshape(per_head, w)?, norms.epsilon)?;
        d.set_columns(&mut g_p, 0, &g_qk.reshape(rows, heads.normed_columns())?)?;
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
        layer_normed(heads, keys, rotary, causal, false)
    }

    /// [`layer`], each query and key normed per head (an RMS norm and its own diagonal gain) when
    /// `normed`, as Qwen3 imports.
    fn layer_normed(heads: usize, keys: usize, rotary: Option<Rotary>, causal: bool, normed: bool) -> (OperatorProgram, FamilyInputs) {
        let mut operators = vec![Arc::new(Operator::identity("I", Interface::native(D).expect("d"))), dense("lift", D, D, 1)];
        let mut nodes = vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 1)], bias: None }];
        let mut op = |o: Arc<Operator>| {
            operators.push(o);
            operators.len() - 1
        };
        let gain = |seed: usize| {
            let values = ndarray::Array1::from_shape_fn(WIDTH, |j| 1.0 + 0.3 * noise(seed * 1000 + j));
            Arc::new(Operator::diag("gain", Interface::native(WIDTH).expect("w"), values.clone(), exact_precision(values.iter().copied()).expect("exact"), Default::default()).expect("diag"))
        };
        // A projection node, normed per head when `normed`: the node an attend reads.
        let normalize = |nodes: &mut Vec<Node>, gain_op: usize| {
            if normed {
                nodes.push(Node::RmsNorm { input: nodes.len() - 1, epsilon: 1e-6 });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, gain_op)], bias: None });
            }
            nodes.len() - 1
        };
        let mut pairs = Vec::new();
        for g in 0..keys {
            let (k, kb, v, kg) = (op(dense("k", WIDTH, D, 10 + g)), op(dense("kb", WIDTH, 1, 20 + g)), op(dense("v", WIDTH, D, 30 + g)), op(gain(80 + g)));
            nodes.push(Node::Affine { terms: vec![(1, k)], bias: Some(kb) });
            let key = normalize(&mut nodes, kg);
            nodes.push(Node::Affine { terms: vec![(1, v)], bias: None });
            pairs.push((key, nodes.len() - 1));
        }
        let mut terms = vec![(1, 0)];
        for h in 0..heads {
            let (q, qb, o, qg) = (op(dense("q", WIDTH, D, 40 + h)), op(dense("qb", WIDTH, 1, 50 + h)), op(dense("o", D, WIDTH, 60 + h)), op(gain(90 + h)));
            nodes.push(Node::Affine { terms: vec![(1, q)], bias: Some(qb) });
            let query = normalize(&mut nodes, qg);
            let (key, value) = pairs[h % keys];
            nodes.push(Node::Attend { query, key, value, scale: Scale::InverseSqrt(WIDTH as u32), rotary, causal });
            terms.push((nodes.len() - 1, o));
        }
        let ob = op(dense("ob", D, 1, 70));
        nodes.push(Node::Affine { terms, bias: Some(ob) });
        nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![crate::operator_program::Law::GeluTanh] });
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
    fn resident_attention_restacks_updated_parameters_without_changing_shared_programs() {
        let mut backends = devices();
        // CUDA exercises actual float32 storage when available; the host supports float64.
        let single: Vec<_> = backends.iter().filter_map(|d| d.with_storage(gam_gpu::tensor::Storage::F32).ok()).collect();
        backends.extend(single);
        // Metal is discovered separately: unlike CUDA, it has no float64 device.
        if let Some(device) = Device::single_precision(gam_gpu::GpuPolicy::Auto).expect("single-precision device") {
            if !backends.iter().any(|d| d.name() == device.name() && d.storage() == device.storage()) {
                backends.push(device);
            }
        }
        for device in backends {
            for normed in [false, true] {
                let (program, family) = layer_normed(4, 2, Some(Rotary { base: 500, dims: 2, half_split: false }), true, normed);
                let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
                let tolerance = if device.float64() { 1e-11 } else { 2e-5 };
                let trainable: Vec<_> = program.operators.iter().enumerate()
                    .filter(|(_, op)| matches!(op.name.as_str(), "q" | "k" | "v" | "qb" | "kb" | "o" | "ob"))
                    .map(|(id, _)| id).collect();
                let mut original = DeviceProgram::compile_values(&device, &program).expect("original");
                original.set_arithmetic(arithmetic);
                let baseline = original.forward(&family).expect("baseline");
                let baseline_value = device.download(baseline.value(program.output).unwrap()).unwrap();
                let mut fused = DeviceProgram::compile_values_sharing(&original, &program).expect("fused");
                let mut plain = DeviceProgram::compile_values_sharing(&original, &program).expect("plain");
                fused.set_arithmetic(arithmetic);
                plain.set_arithmetic(arithmetic);
                plain.unfuse();
                fused.prepare_dense_parameters(&trainable).expect("prepare fused");
                plain.prepare_dense_parameters(&trainable).expect("prepare plain");
                assert_eq!(fused.fused_groups(), 1);
                assert_eq!(plain.fused_groups(), 0);
                for round in 1..=2 {
                    for &op in &trainable {
                        let moved = program.operators[op].matrix().mapv(|x| x * (1.0 + 0.1 * round as f64) + 0.02 * round as f64);
                        fused.replace_dense_parameter(op, device.upload(moved.view()).unwrap()).unwrap();
                        plain.replace_dense_parameter(op, device.upload(moved.view()).unwrap()).unwrap();
                    }
                    assert_eq!(fused.fused_groups(), 0, "stale stacks never execute");
                    fused.refresh_fused().expect("resident restack");
                    plain.refresh_fused().expect("explicit unfuse survives restacking");
                    assert_eq!(fused.fused_groups(), 1);
                    assert_eq!(plain.fused_groups(), 0);
                    let a = fused.forward(&family).unwrap();
                    let b = plain.forward(&family).unwrap();
                    for node in 0..program.nodes.len() {
                        let av = a.value(node).unwrap();
                        assert_eq!(av.storage(), device.storage());
                        let (x, y) = (device.download(av).unwrap(), device.download(b.value(node).unwrap()).unwrap());
                        assert!(worst(&x, &y) < tolerance, "round {round} node {node}: {}", worst(&x, &y));
                    }
                    let updated = device.download(a.value(program.output).unwrap()).unwrap();
                    assert!(worst(&updated, &baseline_value) > 1e-3);
                    let seed = Array2::from_shape_fn((family.rows, D), |(r, c)| noise(7000 + r * D + c));
                    let seeds = || BTreeMap::from([(program.output, device.upload(seed.view()).unwrap())]);
                    let (ag, ap) = fused.vjp_values_dense(&a, seeds(), &[0], &trainable, arithmetic).unwrap();
                    let (bg, bp) = plain.vjp_values_dense(&b, seeds(), &[0], &trainable, arithmetic).unwrap();
                    assert!(worst(&device.download(&ag[&0]).unwrap(), &device.download(&bg[&0]).unwrap()) < tolerance);
                    for &op in &trainable {
                        assert!(worst(&device.download(&ap[&op]).unwrap(), &device.download(&bp[&op]).unwrap()) < tolerance * 10.0, "gradient {op}");
                    }
                    // An edit belongs to one head invocation, not all heads sharing a stack.
                    let reads: Vec<_> = program.nodes.iter().enumerate().filter_map(|(n, node)| matches!(node, Node::Attend { .. }).then_some(n)).collect();
                    let edit = |node: usize, trace: &crate::device_program::DeviceTrace| -> Result<Option<Tensor>, String> {
                        if node != reads[0] { return Ok(None); }
                        let mut value = device.copy(trace.value(node)?).map_err(|e| e.to_string())?;
                        device.axpy(&mut value, 1.0, trace.value(node)?).map_err(|e| e.to_string())?;
                        Ok(Some(value))
                    };
                    let edited = fused.forward_edited(&family, BTreeMap::new(), &Default::default(), |_, _| Ok(()), edit).unwrap();
                    let expected = plain.forward_edited(&family, BTreeMap::new(), &Default::default(), |_, _| Ok(()), edit).unwrap();
                    assert!(worst(&device.download(edited.value(program.output).unwrap()).unwrap(), &device.download(expected.value(program.output).unwrap()).unwrap()) < tolerance);
                    assert!(worst(&device.download(edited.value(reads[0]).unwrap()).unwrap(), &device.download(a.value(reads[0]).unwrap()).unwrap()) > 1e-3);
                    assert_eq!(device.download(edited.value(reads[1]).unwrap()).unwrap(), device.download(a.value(reads[1]).unwrap()).unwrap());
                    assert_eq!(device.download(original.forward(&family).unwrap().value(program.output).unwrap()).unwrap(), baseline_value);
                    // A fresh program using the old host operators cannot inherit the trained stack.
                    let mut sibling = DeviceProgram::compile_values_sharing(&fused, &program).unwrap();
                    sibling.set_arithmetic(arithmetic);
                    assert_eq!(device.download(sibling.forward(&family).unwrap().value(program.output).unwrap()).unwrap(), baseline_value);
                }
            }
        }
    }

    /// A weight sample written straight into a fused group's stacks
    /// (`DevicePosterior::sample_into`) computes what the same sample run node by node computes,
    /// sample after sample, and the program the stacks were shared with keeps its own.
    #[test]
    fn a_sample_written_into_the_stacks_runs_as_its_operators_do() {
        use crate::device_posterior::{DevicePosterior, Parts};
        let mut backends = devices();
        let single: Vec<_> = backends.iter().filter_map(|d| d.with_storage(gam_gpu::tensor::Storage::F32).ok()).collect();
        backends.extend(single);
        if let Some(device) = Device::single_precision(gam_gpu::GpuPolicy::Auto).expect("single-precision device") {
            if !backends.iter().any(|d| d.name() == device.name() && d.storage() == device.storage()) {
                backends.push(device);
            }
        }
        for device in backends {
            let (program, family) = layer_normed(4, 2, Some(Rotary { base: 500, dims: 2, half_split: false }), true, false);
            let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
            let tolerance = if device.float64() { 1e-11 } else { 2e-5 };
            let trainable: Vec<_> = program.operators.iter().enumerate()
                .filter(|(_, op)| matches!(op.name.as_str(), "q" | "k" | "v" | "qb" | "kb" | "o" | "ob"))
                .map(|(id, _)| id).collect();
            let mut original = DeviceProgram::compile_values(&device, &program).expect("original");
            original.set_arithmetic(arithmetic);
            let before = device.download(original.forward(&family).unwrap().value(program.output).unwrap()).unwrap();
            let mut fused = DeviceProgram::compile_values_sharing(&original, &program).expect("fused");
            let mut plain = DeviceProgram::compile_values_sharing(&original, &program).expect("plain");
            fused.set_arithmetic(arithmetic);
            plain.set_arithmetic(arithmetic);
            plain.unfuse();
            fused.prepare_dense_parameters(&trainable).expect("prepare fused");
            plain.prepare_dense_parameters(&trainable).expect("prepare plain");
            let mean: Vec<Array2<f64>> = trainable.iter().map(|&op| program.operators[op].matrix().mapv(|x| 1.2 * x + 0.01)).collect();
            let log_sd: Vec<Array2<f64>> = mean.iter().map(|m| m.mapv(|_| 0.05_f64.ln())).collect();
            let groups: Vec<Vec<u32>> = mean.iter().map(|m| vec![0; m.len()]).collect();
            let parts = Parts { operators: &trainable, mean: &mean, log_sd: &log_sd, groups: &groups, count: 1 };
            let posterior = DevicePosterior::from_parts(&device, &parts, 100.0, None, 0).expect("posterior");
            for key in [3, 4] {
                posterior.sample_into(&mut fused, key).expect("fused sample");
                posterior.sample_into(&mut plain, key).expect("plain sample");
                assert_eq!((fused.fused_groups(), plain.fused_groups()), (1, 0));
                let (a, b) = (fused.forward(&family).unwrap(), plain.forward(&family).unwrap());
                for node in 0..program.nodes.len() {
                    let (x, y) = (device.download(a.value(node).unwrap()).unwrap(), device.download(b.value(node).unwrap()).unwrap());
                    assert!(worst(&x, &y) < tolerance, "{} key {key} node {node}: {}", device.name(), worst(&x, &y));
                }
                assert!(worst(&device.download(a.value(program.output).unwrap()).unwrap(), &before) > 1e-3, "the sample moved the output");
                assert_eq!(device.download(original.forward(&family).unwrap().value(program.output).unwrap()).unwrap(), before, "the shared program keeps its stacks");
            }
        }
    }

    #[test]
    fn resident_attention_repacking_respects_changed_norm_gain_roles() {
        let device = Device::host();
        let (mut program, family) = layer_normed(2, 1, None, true, true);
        let gain = program.operators.iter().position(|op| op.name == "gain").unwrap();
        let old = &program.operators[gain];
        let values = old.matrix();
        let mut fused = DeviceProgram::compile_values(&device, &program).unwrap();
        assert_eq!(fused.fused_groups(), 1);
        let mut plain = DeviceProgram::compile_values_sharing(&fused, &program).unwrap();
        plain.unfuse();
        let mut moved = values.clone();
        moved[(0, 1)] = 0.3;
        let mut changed = program.clone();
        changed.operators[gain] = Arc::new(Operator::dense("gain", old.rows.clone(), old.cols.clone(), moved.clone(), exact_precision(moved.iter().copied()).unwrap(), Default::default()).unwrap());
        fused.refresh(&changed).unwrap();
        plain.refresh(&changed).unwrap();
        fused.refresh_fused().unwrap();
        assert_eq!(fused.fused_groups(), 0, "off-diagonal trained gains are not column scales");
        let a = fused.forward(&family).unwrap();
        let b = plain.forward(&family).unwrap();
        assert_eq!(device.download(a.value(program.output).unwrap()).unwrap(), device.download(b.value(program.output).unwrap()).unwrap());
        // Refreshing original host parameters makes the gain diagonal again and safe to pack.
        fused.refresh(&program).unwrap();
        assert_eq!(fused.fused_groups(), 1);
        let original = DeviceProgram::compile_values(&device, &program).unwrap();
        let original_value = device.download(original.forward(&family).unwrap().value(program.output).unwrap()).unwrap();
        assert_eq!(device.download(fused.forward(&family).unwrap().value(program.output).unwrap()).unwrap(), original_value);
        // Host sharing must include the gains' identities too.
        let replacement = values.diag().mapv(|x| x * 2.0);
        let old = &program.operators[gain];
        program.operators[gain] = Arc::new(Operator::diag("gain", old.rows.clone(), replacement.clone(), exact_precision(replacement.iter().copied()).unwrap(), Default::default()).unwrap());
        let shared = DeviceProgram::compile_values_sharing(&original, &program).unwrap();
        let fresh = DeviceProgram::compile_values(&device, &program).unwrap();
        let shared_value = device.download(shared.forward(&family).unwrap().value(program.output).unwrap()).unwrap();
        assert_eq!(shared_value, device.download(fresh.forward(&family).unwrap().value(program.output).unwrap()).unwrap());
        assert!(worst(&shared_value, &original_value) > 1e-3);
    }

    #[test]
    fn fused_heads_match_node_by_node_values_cotangents_and_edits() {
        let rotations = [None, Some(Rotary { base: 10_000, dims: 4, half_split: true }), Some(Rotary { base: 500, dims: 2, half_split: false })];
        for (heads, keys, normed) in [(4, 2, false), (3, 3, false), (2, 1, false), (4, 2, true), (2, 1, true)] {
            for rotary in rotations {
                for causal in [true, false] {
                    let (program, family) = layer_normed(heads, keys, rotary, causal, normed);
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
                        let query = program.nodes.iter().find_map(|n| match n { Node::Attend { query, .. } => Some(*query), _ => None }).expect("a query");
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
            let once = d.download(&attend_by(&d, heads, (&p, &p), SEQUENCES, turn, Arithmetic::F64, heads.keys).expect("once")).expect("once");
            let turns = d.download(&attend_by(&d, heads, (&p, &p), SEQUENCES, turn, Arithmetic::F64, 1).expect("in turn")).expect("in turn");
            assert!(worst(&once, &turns) < 1e-13, "{}: reads", d.name());
            let arithmetic = (Arithmetic::F64, Arithmetic::F64);
            let once = d.download(&backward_by(&d, (heads, &stacked), (&p, None, &g_a), SEQUENCES, turn, arithmetic, heads.keys).expect("once")).expect("once");
            let turns = d.download(&backward_by(&d, (heads, &stacked), (&p, None, &g_a), SEQUENCES, turn, arithmetic, 1).expect("in turn")).expect("in turn");
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
