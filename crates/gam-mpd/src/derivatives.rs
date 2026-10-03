//! Exact directional derivatives of an operator program, and precisions derived from them (#2951).
//!
//! [`vjp`] is reverse-mode differentiation: every node's cotangent from the output's, one pass
//! for all nodes. [`jvp`] is forward-mode differentiation of the executed program in its operators' reals: given a
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
use super::operator_program::{FamilyInputs, Node, Operator, OperatorBody, OperatorProgram, ProgramError, Scale, Trace};
use super::precision::DeclaredPrecision;
use gam_gpu::banded::Layout;
use ndarray::{Array2, Axis, s};
use std::collections::BTreeMap;

fn refuse(message: String) -> ProgramError {
    ProgramError::Input(message)
}

/// The output tangent of `program` on `inputs` when each operator in `tangents` moves along its
/// entry (a matrix of the operator's shape, or a diagonal operator's diagonal as one row), from
/// the base `trace` (unbanded values).
pub fn jvp(
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    trace: &Trace,
    tangents: &BTreeMap<usize, Array2<f64>>,
) -> Result<Array2<f64>, ProgramError> {
    jvp_seeded(program, inputs, trace, tangents, &BTreeMap::new())
}

/// [`jvp`] with node seeds: `seeds[node]` (rows × the node's width) is added to that node's
/// tangent after its own rule, so a seed alone gives the output's derivative along a direction of
/// the node's value (the measured rounding band carries every node's local rounding this way).
pub fn jvp_seeded(
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    trace: &Trace,
    tangents: &BTreeMap<usize, Array2<f64>>,
    seeds: &BTreeMap<usize, Array2<f64>>,
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
                    if let Some(dx) = tangent_of(&dv, *argument) {
                        add(program.operators[*operator].apply(&dx));
                    }
                    if let Some(da) = tangents.get(operator) {
                        add(tangent_product(
                            &program.operators[*operator],
                            da,
                            value(*argument),
                            program.gathered_tokens(*argument, inputs),
                        ));
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
                .map(|dy| program.bases[*basis].read(&program.declarations, &dy))
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
                let op = &program.operators[*operator];
                let mut out: Option<Array2<f64>> = None;
                if let Some(dx) = tangent_of(&dv, *input) {
                    // `x A`; a diagonal is its own transpose.
                    out = Some(match op.diagonal() {
                        Some(d) => dx * &d,
                        None => dx.dot(op.matrix_cow().as_ref()),
                    });
                }
                if let Some(da) = tangents.get(operator) {
                    let term = match &op.body {
                        OperatorBody::Diagonal { .. } if da.dim() == (1, op.rows.width()) => value(*input) * &da.row(0),
                        _ => value(*input).dot(da),
                    };
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
        dv[index] = match (t, seeds.get(&index)) {
            (Some(t), Some(seed)) => Some(t + seed),
            (None, Some(seed)) => Some(seed.clone()),
            (t, None) => t,
        };
    }
    Ok(dv[program.output].clone().unwrap_or_else(|| Array2::zeros(trace.values[program.output].dim())))
}

type Pair<'a> = (&'a Array2<f64>, &'a Array2<f64>, &'a Array2<f64>);

/// `x dAᵀ` for an affine term on operator `op` along its tangent `da`: a matrix of the operator's
/// shape or, for a diagonal operator, its diagonal as one row (`1 × width`). A gathered feature
/// argument (`tokens`) reads the tangent's column at each row's token.
fn tangent_product(op: &Operator, da: &Array2<f64>, x: &Array2<f64>, tokens: Option<&[u32]>) -> Array2<f64> {
    let width = op.rows.width();
    let diagonal = matches!(op.body, OperatorBody::Diagonal { .. }) && da.dim() == (1, width);
    match tokens {
        Some(tokens) => {
            let mut out = Array2::<f64>::zeros((tokens.len(), width));
            for (mut row, &token) in out.outer_iter_mut().zip(tokens) {
                let t = token as usize;
                if diagonal {
                    row[t] = da[[0, t]];
                } else {
                    row.assign(&da.column(t));
                }
            }
            out
        }
        None if diagonal => x * &da.row(0),
        None => x.dot(&da.t()),
    }
}

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
    if inputs.rows >= 32 {
        return super::tiled_attention::tangent(inputs, (query, key, value), (dq.as_ref(), dk.as_ref(), dv.as_ref()), scale.value(), rotary, causal);
    }
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

/// The cotangent of every node of `program` on `inputs` when the output's cotangent is
/// `output` (rows × the output's width), from the base `trace` (unbanded values): reverse-mode
/// differentiation, the transpose of [`jvp`] node by node. `None` where no cotangent reaches.
pub fn vjp(
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    trace: &Trace,
    output: Array2<f64>,
) -> Result<Vec<Option<Array2<f64>>>, ProgramError> {
    vjp_from(program, inputs, trace, program.output, output, None)
}

/// Reverse from a node, retaining only the requested cotangents when `keep` is present.
/// Unrequested intermediates are consumed in place and released after their reverse rule.
pub(crate) fn vjp_from(
    program: &OperatorProgram, inputs: &FamilyInputs, trace: &Trace, seed_node: usize,
    output: Array2<f64>, keep: Option<&[usize]>,
) -> Result<Vec<Option<Array2<f64>>>, ProgramError> {
    let interfaces = program.interfaces()?;
    let rows = inputs.rows;
    if seed_node >= interfaces.len() || output.dim() != (rows, interfaces[seed_node].width()) {
        return Err(refuse("reverse seed shape does not match its node".to_string()));
    }
    let mut retained = vec![keep.is_none(); program.nodes.len()];
    if let Some(keep) = keep {
        for &node in keep {
            if node >= retained.len() { return Err(refuse("retained cotangent node out of range".to_string())); }
            retained[node] = true;
        }
    }
    let mut g: Vec<Option<Array2<f64>>> = vec![None; program.nodes.len()];
    g[seed_node] = Some(output);
    let value = |node: usize| &trace.values[node];
    fn add(g: &mut [Option<Array2<f64>>], node: usize, term: Array2<f64>) {
        match g[node].as_mut() {
            Some(existing) => *existing += &term,
            None => g[node] = Some(term),
        }
    }
    let first = keep.and_then(|nodes| nodes.iter().min().copied()).unwrap_or(0);
    for index in (first..program.nodes.len()).rev() {
        let Some(cot) = g[index].take() else { continue };
        if retained[index] { g[index] = Some(cot.clone()); }
        if keep.is_some() && index == first { break; }
        let node = &program.nodes[index];
        match node {
            Node::Feature { .. } | Node::Raw { .. } | Node::Constant { .. } => {}
            Node::Affine { terms, .. } => {
                for (argument, operator) in terms {
                    let op = &program.operators[*operator];
                    // The identity and a norm gain are column scales, not matrix products.
                    let term = match op.diagonal() {
                        Some(d) => &cot * &d,
                        None => super::device::product(op, &cot, Layout::AsStored)?,
                    };
                    add(&mut g, *argument, term);
                }
            }
            Node::Bilinear { left, right, scale } => {
                let c = scale.value();
                let column = cot.column(0).to_owned();
                let mut gl = value(*right).clone();
                let mut gr = value(*left).clone();
                for row in 0..rows {
                    gl.row_mut(row).mapv_inplace(|v| v * c * column[row]);
                    gr.row_mut(row).mapv_inplace(|v| v * c * column[row]);
                }
                add(&mut g, *left, gl);
                add(&mut g, *right, gr);
            }
            Node::Softmax { scores } => {
                let alpha = value(index);
                for (j, score) in scores.iter().enumerate() {
                    let mut ds = Array2::<f64>::zeros((rows, 1));
                    for row in 0..rows {
                        let mean: f64 = (0..scores.len()).map(|k| alpha[[row, k]] * cot[[row, k]]).sum();
                        ds[[row, 0]] = alpha[[row, j]] * (cot[[row, j]] - mean);
                    }
                    add(&mut g, *score, ds);
                }
            }
            Node::Mix { weights, payloads } => {
                let alpha = value(*weights);
                let mut gw = Array2::<f64>::zeros(alpha.dim());
                for &(column, payload) in payloads {
                    let mut gp = cot.clone();
                    for row in 0..rows {
                        gp.row_mut(row).mapv_inplace(|v| v * alpha[[row, column]]);
                        gw[[row, column]] += cot.row(row).dot(&value(payload).row(row));
                    }
                    add(&mut g, payload, gp);
                }
                add(&mut g, *weights, gw);
            }
            Node::Pointwise { input, laws } => {
                let interface = &interfaces[*input];
                let mut out = cot;
                for (group, law) in laws.iter().enumerate() {
                    for c in interface.range(group) {
                        for row in 0..rows {
                            out[[row, c]] *= law.derivative(value(*input)[[row, c]]);
                        }
                    }
                }
                add(&mut g, *input, out);
            }
            Node::Hadamard { left, right } => {
                add(&mut g, *left, &cot * value(*right));
                add(&mut g, *right, &cot * value(*left));
            }
            Node::Readout { input, basis } => add(&mut g, *input, program.bases[*basis].read_transpose(&program.declarations, &cot)?),
            Node::Concat { parts } => {
                let mut offset = 0;
                for part in parts {
                    let width = value(*part).ncols();
                    add(&mut g, *part, cot.slice(s![.., offset..offset + width]).to_owned());
                    offset += width;
                }
            }
            Node::Gain { input, coefficient } => {
                let (c, _, _) = coefficient.evaluate(&vec![1.0; program.declarations.parameters])?;
                add(&mut g, *input, cot * c);
            }
            Node::RmsNorm { input, epsilon } => {
                let x = value(*input);
                let n = x.ncols() as f64;
                let mut out = Array2::<f64>::zeros(x.dim());
                for row in 0..rows {
                    let xr = x.row(row);
                    let mean = xr.iter().map(|v| v * v).sum::<f64>() / n;
                    let scale = 1.0 / (mean + epsilon).sqrt();
                    let inner = xr.dot(&cot.row(row));
                    for c in 0..x.ncols() {
                        out[[row, c]] = scale * cot[[row, c]] - scale * scale * scale / n * xr[c] * inner;
                    }
                }
                add(&mut g, *input, out);
            }
            Node::Transposed { input, operator } => {
                add(&mut g, *input, super::device::product(&program.operators[*operator], &cot, Layout::Transposed)?);
            }
            Node::Attend { query, key, value: v, scale, rotary, causal } => {
                let (gq, gk, gv) =
                    attend_cotangent(inputs, (value(*query), value(*key), value(*v)), &cot, *scale, *rotary, *causal)?;
                add(&mut g, *query, gq);
                add(&mut g, *key, gk);
                add(&mut g, *v, gv);
            }
            Node::Outer { left, right } => {
                // Columns run over group pairs row-major, coordinates row-major within a pair.
                let (l, r) = (value(*left), value(*right));
                let (li, ri) = (&interfaces[*left], &interfaces[*right]);
                let mut gl = Array2::<f64>::zeros(l.dim());
                let mut gr = Array2::<f64>::zeros(r.dim());
                let mut offset = 0;
                for g1 in 0..li.group_count() {
                    for g2 in 0..ri.group_count() {
                        for i in li.range(g1) {
                            for j in ri.range(g2) {
                                for row in 0..rows {
                                    gl[[row, i]] += cot[[row, offset]] * r[[row, j]];
                                    gr[[row, j]] += cot[[row, offset]] * l[[row, i]];
                                }
                                offset += 1;
                            }
                        }
                    }
                }
                add(&mut g, *left, gl);
                add(&mut g, *right, gr);
            }
            Node::Call { .. } | Node::Param { .. } => {
                return Err(refuse(format!("node {index}: no cotangent rule for this node kind")));
            }
        }
    }
    for (node, gradient) in g.iter_mut().enumerate() {
        if !retained[node] { *gradient = None; }
    }
    Ok(g)
}

/// The rotation of `m`'s rows to their positions (`inverse`: back from them), as the attend node
/// applies it; a rotation is orthogonal, so its transpose is the inverse.
fn rotate_rows(m: &Array2<f64>, rotary: Option<super::operator_program::Rotary>, positions: &[u32], inverse: bool) -> Array2<f64> {
    let mut out = m.clone();
    let Some(r) = rotary else { return out };
    for (row, &position) in positions.iter().enumerate() {
        for (plane, (a, b)) in r.pairs().into_iter().enumerate() {
            let (c, s) = r.turn(plane, position);
            let s = if inverse { -s } else { s };
            let (x, y) = (out[[row, a]], out[[row, b]]);
            out[[row, a]] = c * x - s * y;
            out[[row, b]] = s * x + c * y;
        }
    }
    out
}

/// The cotangents of query, key and value of causal rotary attention given the output's
/// cotangent: the transpose of `attend_tangent`.
fn attend_cotangent(
    inputs: &FamilyInputs,
    (query, key, value): Pair<'_>,
    cot: &Array2<f64>,
    scale: Scale,
    rotary: Option<super::operator_program::Rotary>,
    causal: bool,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), ProgramError> {
    if inputs.rows >= 32 {
        return super::tiled_attention::backward(inputs, (query, key, value), cot, scale.value(), rotary, causal);
    }
    let layout = inputs.layout.as_ref().ok_or_else(|| refuse("an attend node needs a sequence layout".to_string()))?;
    let rows = inputs.rows;
    let (q, k) = (rotate_rows(query, rotary, &layout.position, false), rotate_rows(key, rotary, &layout.position, false));
    let c = scale.value();
    let mut by_sequence: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for row in 0..rows {
        by_sequence.entry(layout.sequence[row]).or_default().push(row);
    }
    let (mut gq, mut gk, mut gv) = (Array2::<f64>::zeros(q.dim()), Array2::<f64>::zeros(k.dim()), Array2::<f64>::zeros(value.dim()));
    for members in by_sequence.values() {
        for &row in members {
            let keys: Vec<usize> =
                members.iter().copied().filter(|&o| !causal || layout.position[o] <= layout.position[row]).collect();
            let scores: Vec<f64> = keys.iter().map(|&o| c * q.row(row).dot(&k.row(o))).collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let total: f64 = e.iter().sum();
            let alpha: Vec<f64> = e.iter().map(|v| v / total).collect();
            let dalpha: Vec<f64> = keys.iter().map(|&o| cot.row(row).dot(&value.row(o))).collect();
            let mean: f64 = alpha.iter().zip(&dalpha).map(|(a, d)| a * d).sum();
            for (j, &o) in keys.iter().enumerate() {
                gv.row_mut(o).scaled_add(alpha[j], &cot.row(row));
                let ds = alpha[j] * (dalpha[j] - mean);
                gq.row_mut(row).scaled_add(c * ds, &k.row(o));
                gk.row_mut(o).scaled_add(c * ds, &q.row(row));
            }
        }
    }
    Ok((rotate_rows(&gq, rotary, &layout.position, true), rotate_rows(&gk, rotary, &layout.position, true), gv))
}

/// A Rademacher probe over an operator's present reals, from a fixed-seed generator so the proposal
/// is reproducible.
fn probe(program: &OperatorProgram, operator: usize, seed: u64) -> Option<Array2<f64>> {
    let op = &program.operators[operator];
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(operator as u64 + 1);
    let mut sign = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if state & 1 == 1 { 1.0 } else { -1.0 }
    };
    if let OperatorBody::Diagonal { values, .. } = &op.body {
        // A diagonal's probe is its diagonal, one row.
        return Some(Array2::from_shape_fn((1, values.len()), |_| sign()));
    }
    let OperatorBody::Dense { present, .. } = &op.body else { return None };
    let mut out = Array2::<f64>::zeros((op.rows.width(), op.cols.width()));
    for ((r, c), keep) in present.indexed_iter() {
        if !keep {
            continue;
        }
        for value in out.slice_mut(s![op.rows.range(r), op.cols.range(c)]).iter_mut() {
            *value = sign();
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
            let (OperatorBody::Dense { precision, .. } | OperatorBody::Diagonal { precision, .. }) = &op.body else { continue };
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
