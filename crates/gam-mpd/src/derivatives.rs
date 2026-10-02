//! Exact directional derivatives of an operator program, and precisions derived from them (#2951).
//!
//! [`jvp`] is forward-mode differentiation of the executed program in its operators' reals: given a
//! tangent `dA` for some operators, every node's tangent follows from its law (the chain rule node
//! by node, no finite differences), reading the base values from a trace. [`output_curvature`]
//! estimates the trace of the data code's Gauss–Newton curvature in one operator's reals,
//! `tr F = Σ_rows E_v (J v)ᵀ (diag q − q qᵀ) (J v)` over Rademacher probes `v` on its present
//! reals; the estimate is statistical and only ever proposes, never certifies.
//!
//! # The precision a curvature asks for
//!
//! Rounding an operator's `m` reals to a lattice of step `Δ` moves each by a uniform error of
//! variance `Δ²/12`, so the data code `n Σ KL/ln 2` grows by `n Δ² tr F / (24 ln 2)` to second
//! order, while the reals' indices shrink by `m log₂ Δ` bits. The sum is least at
//! `Δ² = 12 m / (n tr F)`. [`CurvaturePrecision`] proposes each operator's lattice at that step's
//! nearest power of two (and its two neighbours); the exact decoded code and the contract decide.

use super::engine::{EngineError, Edit, Exactness, Primitive, Proposal, SearchContext};
use super::fit::ProposalKind;
use super::operator_program::{FamilyInputs, Node, OperatorBody, OperatorProgram, ProgramError, Scale, Trace};
use super::precision::DeclaredPrecision;
use ndarray::{Array2, Axis, s};
use std::collections::BTreeMap;

fn refuse(message: String) -> ProgramError {
    ProgramError::Input(message)
}

/// The output tangent of `program` on `inputs` when each operator in `tangents` moves along its
/// entry (a matrix of the operator's shape), from the base `trace` (unbanded values).
pub fn jvp(
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    trace: &Trace,
    tangents: &BTreeMap<usize, Array2<f64>>,
) -> Result<Array2<f64>, ProgramError> {
    let interfaces = program.interfaces()?;
    let rows = inputs.rows;
    let mut dv: Vec<Option<Array2<f64>>> = vec![None; program.nodes.len()];
    let value = |node: usize| &trace.values[node];
    for (index, node) in program.nodes.iter().enumerate() {
        let width = interfaces[index].width();
        let zero = || Array2::<f64>::zeros((rows, width));
        let tangent_of = |dv: &Vec<Option<Array2<f64>>>, node: usize| dv[node].clone();
        let t: Option<Array2<f64>> = match node {
            Node::Feature { .. } | Node::Raw { .. } => None,
            Node::Constant { operator } => tangents.get(operator).map(|d| {
                let column = d.column(0).to_owned();
                column.broadcast((rows, column.len())).map(|b| b.to_owned()).unwrap_or_else(zero)
            }),
            Node::Affine { terms, bias } => {
                let mut out: Option<Array2<f64>> = None;
                let mut add = |term: Array2<f64>| match out.as_mut() {
                    Some(o) => *o += &term,
                    None => out = Some(term),
                };
                for (argument, operator) in terms {
                    let a = program.operators[*operator].matrix();
                    if let Some(dx) = tangent_of(&dv, *argument) {
                        add(dx.dot(&a.t()));
                    }
                    if let Some(da) = tangents.get(operator) {
                        add(value(*argument).dot(&da.t()));
                    }
                }
                if let Some(db) = bias.and_then(|b| tangents.get(&b)) {
                    let column = db.column(0).to_owned();
                    let mut term = zero();
                    term += &column;
                    add(term);
                }
                out
            }
            Node::Bilinear { left, right, scale } => {
                let (dl, dr) = (tangent_of(&dv, *left), tangent_of(&dv, *right));
                if dl.is_none() && dr.is_none() {
                    None
                } else {
                    let c = scale.value();
                    let mut out = zero();
                    for row in 0..rows {
                        let mut total = 0.0;
                        if let Some(dl) = &dl {
                            total += dl.row(row).dot(&value(*right).row(row));
                        }
                        if let Some(dr) = &dr {
                            total += value(*left).row(row).dot(&dr.row(row));
                        }
                        out[[row, 0]] = c * total;
                    }
                    Some(out)
                }
            }
            Node::Softmax { scores } => {
                let ds: Vec<Option<Array2<f64>>> = scores.iter().map(|s| tangent_of(&dv, *s)).collect();
                if ds.iter().all(Option::is_none) {
                    None
                } else {
                    let alpha = value(index);
                    let mut out = zero();
                    for row in 0..rows {
                        let d: Vec<f64> = ds.iter().map(|t| t.as_ref().map_or(0.0, |t| t[[row, 0]])).collect();
                        let mean: f64 = (0..scores.len()).map(|j| alpha[[row, j]] * d[j]).sum();
                        for j in 0..scores.len() {
                            out[[row, j]] = alpha[[row, j]] * (d[j] - mean);
                        }
                    }
                    Some(out)
                }
            }
            Node::Mix { weights, payloads } => {
                let dw = tangent_of(&dv, *weights);
                let mut out: Option<Array2<f64>> = None;
                for &(column, payload) in payloads {
                    let alpha = value(*weights).column(column).to_owned();
                    let dp = tangent_of(&dv, payload);
                    if dw.is_none() && dp.is_none() {
                        continue;
                    }
                    let mut term = zero();
                    for row in 0..rows {
                        if let Some(dw) = &dw {
                            term.row_mut(row).scaled_add(dw[[row, column]], &value(payload).row(row));
                        }
                        if let Some(dp) = &dp {
                            term.row_mut(row).scaled_add(alpha[row], &dp.row(row));
                        }
                    }
                    match out.as_mut() {
                        Some(o) => *o += &term,
                        None => out = Some(term),
                    }
                }
                out
            }
            Node::Pointwise { input, laws } => tangent_of(&dv, *input).map(|dx| {
                let interface = &interfaces[*input];
                let mut out = dx;
                for (group, law) in laws.iter().enumerate() {
                    for c in interface.range(group) {
                        for row in 0..rows {
                            out[[row, c]] *= law.derivative(value(*input)[[row, c]]);
                        }
                    }
                }
                out
            }),
            Node::Hadamard { left, right } => {
                let (dl, dr) = (tangent_of(&dv, *left), tangent_of(&dv, *right));
                match (dl, dr) {
                    (None, None) => None,
                    (dl, dr) => {
                        let mut out = zero();
                        if let Some(dl) = dl {
                            out += &(&dl * value(*right));
                        }
                        if let Some(dr) = dr {
                            out += &(value(*left) * &dr);
                        }
                        Some(out)
                    }
                }
            }
            Node::Outer { .. } | Node::Call { .. } | Node::Param { .. } => {
                if node.arguments().iter().any(|a| dv[*a].is_some())
                    || program.node_operators(node).iter().any(|op| tangents.contains_key(op))
                {
                    return Err(refuse(format!("node {index}: no tangent rule for this node kind")));
                }
                None
            }
            Node::Concat { parts } => {
                if parts.iter().all(|p| dv[*p].is_none()) {
                    None
                } else {
                    let blocks: Vec<Array2<f64>> = parts
                        .iter()
                        .map(|p| dv[*p].clone().unwrap_or_else(|| Array2::zeros(value(*p).dim())))
                        .collect();
                    let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                    Some(ndarray::concatenate(Axis(1), &views).map_err(|e| refuse(e.to_string()))?)
                }
            }
            Node::Readout { input, basis } => tangent_of(&dv, *input)
                .map(|dy| -> Result<Array2<f64>, ProgramError> {
                    let size = program.declarations.domains[program.bases[*basis].domain()].size;
                    let classes: Vec<u32> = (0..size as u32).collect();
                    let phi = program.bases[*basis].evaluate(&program.declarations, &classes)?.values;
                    Ok(dy.dot(&phi.t()))
                })
                .transpose()?,
            Node::Gain { input, coefficient } => {
                let (c, _, _) = coefficient.evaluate(&vec![1.0; program.declarations.parameters])?;
                tangent_of(&dv, *input).map(|dx| dx * c)
            }
            Node::RmsNorm { input, epsilon } => tangent_of(&dv, *input).map(|dx| {
                let x = value(*input);
                let n = x.ncols() as f64;
                let mut out = zero();
                for row in 0..rows {
                    let xr = x.row(row);
                    let mean = xr.iter().map(|v| v * v).sum::<f64>() / n;
                    let scale = 1.0 / (mean + epsilon).sqrt();
                    let dm = 2.0 * xr.iter().zip(dx.row(row).iter()).map(|(a, b)| a * b).sum::<f64>() / n;
                    let ds = -0.5 * scale * scale * scale * dm;
                    for c in 0..x.ncols() {
                        out[[row, c]] = dx[[row, c]] * scale + xr[c] * ds;
                    }
                }
                out
            }),
            Node::Transposed { input, operator } => {
                let a = program.operators[*operator].matrix();
                let mut out: Option<Array2<f64>> = None;
                if let Some(dx) = tangent_of(&dv, *input) {
                    out = Some(dx.dot(&a));
                }
                if let Some(da) = tangents.get(operator) {
                    let term = value(*input).dot(da);
                    out = Some(match out {
                        Some(o) => o + term,
                        None => term,
                    });
                }
                out
            }
            Node::Attend { query, key, value: v, scale, rotary, causal } => {
                let (dq, dk, dvv) = (tangent_of(&dv, *query), tangent_of(&dv, *key), tangent_of(&dv, *v));
                if dq.is_none() && dk.is_none() && dvv.is_none() {
                    None
                } else {
                    Some(attend_tangent(
                        inputs,
                        (value(*query), value(*key), value(*v)),
                        (dq, dk, dvv),
                        *scale,
                        *rotary,
                        *causal,
                    )?)
                }
            }
        };
        dv[index] = t;
    }
    Ok(dv[program.output].clone().unwrap_or_else(|| Array2::zeros(trace.values[program.output].dim())))
}

type Pair<'a> = (&'a Array2<f64>, &'a Array2<f64>, &'a Array2<f64>);

/// The tangent of causal rotary attention (see `operator_program`'s attend) along query, key and
/// value tangents: rotation is linear, so tangents rotate like values; the softmax and the read
/// follow the product rule.
fn attend_tangent(
    inputs: &FamilyInputs,
    (query, key, value): Pair<'_>,
    (dq, dk, dv): (Option<Array2<f64>>, Option<Array2<f64>>, Option<Array2<f64>>),
    scale: Scale,
    rotary: Option<super::operator_program::Rotary>,
    causal: bool,
) -> Result<Array2<f64>, ProgramError> {
    let layout = inputs.layout.as_ref().ok_or_else(|| refuse("an attend node needs a sequence layout".to_string()))?;
    let rows = inputs.rows;
    let rotate = |m: &Array2<f64>| -> Array2<f64> {
        let mut out = m.clone();
        if let Some(r) = rotary {
            for row in 0..rows {
                let mut v = out.row(row).to_vec();
                r.rotate(&mut v, None, layout.position[row]);
                out.row_mut(row).assign(&ndarray::ArrayView1::from(&v));
            }
        }
        out
    };
    let (q, k) = (rotate(query), rotate(key));
    let dq = dq.map(|t| rotate(&t));
    let dk = dk.map(|t| rotate(&t));
    let c = scale.value();
    let mut by_sequence: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for row in 0..rows {
        by_sequence.entry(layout.sequence[row]).or_default().push(row);
    }
    let mut out = Array2::<f64>::zeros((rows, value.ncols()));
    for members in by_sequence.values() {
        for &row in members {
            let keys: Vec<usize> =
                members.iter().copied().filter(|&o| !causal || layout.position[o] <= layout.position[row]).collect();
            let scores: Vec<f64> = keys.iter().map(|&o| c * q.row(row).dot(&k.row(o))).collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let total: f64 = e.iter().sum();
            let alpha: Vec<f64> = e.iter().map(|v| v / total).collect();
            let ds: Vec<f64> = keys
                .iter()
                .map(|&o| {
                    let mut t = 0.0;
                    if let Some(dq) = &dq {
                        t += dq.row(row).dot(&k.row(o));
                    }
                    if let Some(dk) = &dk {
                        t += q.row(row).dot(&dk.row(o));
                    }
                    c * t
                })
                .collect();
            let mean: f64 = alpha.iter().zip(&ds).map(|(a, d)| a * d).sum();
            for (j, &o) in keys.iter().enumerate() {
                let dalpha = alpha[j] * (ds[j] - mean);
                out.row_mut(row).scaled_add(dalpha, &value.row(o));
                if let Some(dv) = &dv {
                    out.row_mut(row).scaled_add(alpha[j], &dv.row(o));
                }
            }
        }
    }
    Ok(out)
}

/// A Rademacher probe over an operator's present reals, from a fixed-seed generator so the proposal
/// is reproducible.
fn probe(program: &OperatorProgram, operator: usize, seed: u64) -> Option<Array2<f64>> {
    let op = &program.operators[operator];
    let OperatorBody::Dense { present, .. } = &op.body else { return None };
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(operator as u64 + 1);
    let mut out = Array2::<f64>::zeros((op.rows.width(), op.cols.width()));
    for ((r, c), keep) in present.indexed_iter() {
        if !keep {
            continue;
        }
        for value in out.slice_mut(s![op.rows.range(r), op.cols.range(c)]).iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = if state & 1 == 1 { 1.0 } else { -1.0 };
        }
    }
    Some(out)
}

/// The statistical estimate (module note) of `tr F` for `operator` from `probes` Rademacher probes:
/// `F`'s rows are the model's output distributions `q` at the program's logits `z`, per
/// distribution row `(J v)ᵀ (diag q − q qᵀ)(J v)`.
pub fn output_curvature(
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    trace: &Trace,
    readouts: usize,
    operator: usize,
    probes: u64,
) -> Result<Option<f64>, ProgramError> {
    let logits = &trace.values[program.output];
    let mut total = 0.0;
    for seed in 0..probes {
        let Some(v) = probe(program, operator, seed) else { return Ok(None) };
        let tangents: BTreeMap<usize, Array2<f64>> = [(operator, v)].into_iter().collect();
        let jv = jvp(program, inputs, trace, &tangents)?;
        let classes = logits.ncols() / readouts.max(1);
        for row in 0..logits.nrows() {
            for part in 0..readouts.max(1) {
                let range = part * classes..(part + 1) * classes;
                let z = logits.slice(s![row, range.clone()]);
                let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                let dz = jv.slice(s![row, range]);
                let mean: f64 = e.iter().zip(dz.iter()).map(|(p, d)| p / sum * d).sum();
                total += e.iter().zip(dz.iter()).map(|(p, d)| p / sum * (d - mean) * (d - mean)).sum::<f64>();
            }
        }
    }
    Ok(Some(total / probes.max(1) as f64))
}

/// Each operator's lattice at the step its curvature asks for, and at the two neighbouring steps
/// (module note). `probes` is the estimator's sample size, a search setting.
pub struct CurvaturePrecision {
    pub probes: u64,
}

impl Primitive for CurvaturePrecision {
    fn name(&self) -> &'static str {
        "curvature_precision"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, op) in program.operators.iter().enumerate() {
            let OperatorBody::Dense { precision, .. } = &op.body else { continue };
            let m = op.real_count();
            if m == 0 {
                continue;
            }
            let Some(trace_f) = output_curvature(program, &context.contract.family, context.trace, context.contract.readouts, index, self.probes)?
            else {
                continue;
            };
            let n = context.contract.observations as f64;
            if !(trace_f > 0.0) {
                continue;
            }
            let step = (12.0 * m as f64 / (n * trace_f)).sqrt();
            if !(step > 0.0 && step.is_finite()) {
                continue;
            }
            let centre = (-step.log2()).round() as i32;
            for bits in [centre - 1, centre, centre + 1] {
                if bits == precision.fraction_bits() {
                    continue;
                }
                // A step beyond the declarable exponents is no proposal.
                let Ok(target) = DeclaredPrecision::new(bits) else { continue };
                out.push(Proposal {
                    primitive: "curvature_precision",
                    kind: ProposalKind::Reduce,
                    exactness: Exactness::Approximate,
                    description: format!("precision of {} to 2^-{bits} (curvature step 2^-{centre})", op.name),
                    edit: Edit::Precision { operator: index, precision: target },
                });
            }
        }
        Ok(out)
    }
}
