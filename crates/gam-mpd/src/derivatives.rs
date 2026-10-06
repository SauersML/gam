//! Exact directional derivatives of an operator program (#2951).
//!
//! [`vjp`] is reverse-mode differentiation: every node's cotangent from the output's, one pass
//! for all nodes. [`jvp`] is forward-mode differentiation of the executed program in its operators' reals: given a
//! tangent `dA` for some operators, every node's tangent follows from its law (the chain rule node
//! by node, no finite differences), reading the base values from a trace.

use super::operator_program::{FamilyInputs, Node, Operator, OperatorBody, OperatorProgram, ProgramError, Scale, Trace};
use gam_linalg::faer_ndarray::fast_ab;
use ndarray::{Array2, Axis, s};
use std::collections::BTreeMap;

fn refuse(message: String) -> ProgramError {
    ProgramError::Input(message)
}

/// `x A`, the cotangent an affine term passes back through its operator `A` (whose own product is
/// `x Aᵀ`, `Operator::apply`).
fn transpose_apply(op: &Operator, x: &Array2<f64>) -> Array2<f64> {
    match &op.body {
        OperatorBody::Identity => x.clone(),
        OperatorBody::Diagonal { values, .. } => x * values,
        OperatorBody::Dense { values, .. } => fast_ab(x, values),
        OperatorBody::LowRank { left, right, .. } => fast_ab(&fast_ab(x, left), right),
    }
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
            Node::Select { inside, outside, positions } => {
                let selected = crate::operator_program::selected_rows(inputs, positions)?;
                match (tangent_of(&dv, *inside), tangent_of(&dv, *outside)) {
                    (None, None) => None,
                    (di, dout) => {
                        let mut out = dout.unwrap_or_else(zero);
                        let di = di.unwrap_or_else(zero);
                        for &row in &selected {
                            out.row_mut(row).assign(&di.row(row));
                        }
                        Some(out)
                    }
                }
            }
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
                    let scale = super::operator_program::rms_scale(xr, *epsilon);
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
    super::tiled_attention::tangent(inputs, (query, key, value), (dq.as_ref(), dk.as_ref(), dv.as_ref()), scale.value(), rotary, causal)
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
    vjp_seeded(program, inputs, trace, std::collections::BTreeMap::from([(seed_node, output)]), keep)
}

/// Reverse the sum of scalar terms reading several nodes, in one pass. A seed at an internal
/// node is added to the cotangents arriving from its consumers before its reverse rule runs.
pub(crate) fn vjp_seeded(
    program: &OperatorProgram, inputs: &FamilyInputs, trace: &Trace,
    seeds: std::collections::BTreeMap<usize, Array2<f64>>, keep: Option<&[usize]>,
) -> Result<Vec<Option<Array2<f64>>>, ProgramError> {
    let interfaces = program.interfaces()?;
    let rows = inputs.rows;
    for (&node, seed) in &seeds {
        if node >= interfaces.len() || seed.dim() != (rows, interfaces[node].width()) {
            return Err(refuse("reverse seed shape does not match its node".to_string()));
        }
    }
    let mut retained = vec![keep.is_none(); program.nodes.len()];
    if let Some(keep) = keep {
        for &node in keep {
            if node >= retained.len() { return Err(refuse("retained cotangent node out of range".to_string())); }
            retained[node] = true;
        }
    }
    let mut g: Vec<Option<Array2<f64>>> = vec![None; program.nodes.len()];
    for (node, seed) in seeds { g[node] = Some(seed); }
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
        if keep.is_some() && index == first {
            g[index] = Some(cot);
            break;
        }
        if retained[index] { g[index] = Some(cot.clone()); }
        let node = &program.nodes[index];
        match node {
            Node::Feature { .. } | Node::Raw { .. } | Node::Constant { .. } => {}
            Node::Affine { terms, .. } => {
                for (argument, operator) in terms {
                    let op = &program.operators[*operator];
                    // The identity and a norm gain are column scales, not matrix products.
                    let term = match op.diagonal() {
                        Some(d) => &cot * &d,
                        None => transpose_apply(op, &cot),
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
            Node::Select { inside, outside, positions } => {
                let selected = crate::operator_program::selected_rows(inputs, positions)?;
                let (mut to_inside, mut to_outside) = (Array2::<f64>::zeros(cot.dim()), cot);
                for &row in &selected {
                    to_inside.row_mut(row).assign(&to_outside.row(row));
                    to_outside.row_mut(row).fill(0.0);
                }
                add(&mut g, *inside, to_inside);
                add(&mut g, *outside, to_outside);
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
                    let scale = super::operator_program::rms_scale(xr, *epsilon);
                    let inner = xr.dot(&cot.row(row));
                    for c in 0..x.ncols() {
                        out[[row, c]] = scale * cot[[row, c]] - scale * scale * scale / n * xr[c] * inner;
                    }
                }
                add(&mut g, *input, out);
            }
            Node::Transposed { input, operator } => {
                add(&mut g, *input, program.operators[*operator].apply(&cot));
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
    super::tiled_attention::backward(inputs, (query, key, value), cot, scale.value(), rotary, causal)
}
