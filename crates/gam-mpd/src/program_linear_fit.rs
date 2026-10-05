//! Conservative fixed-feature local coefficient proposals. Acceptance belongs to the caller.
use crate::{
    linear_coefficient_fit,
    operator_program::{
        exact_precision, FamilyInputs, Node, OperatorBody, OperatorProgram, Slot, SlotValues,
    },
};
use ndarray::{s, Array2};
use serde::Serialize;
use std::{collections::BTreeSet, sync::Arc};

pub struct Fit {
    pub program: OperatorProgram,
    pub report: Report,
}
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub blocks: Vec<Block>,
    pub refusals: Vec<String>,
    pub planned_numeric_bytes: usize,
    pub budget_scope: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct Block {
    pub node: usize,
    pub operators: Vec<usize>,
    pub diagnostics: linear_coefficient_fit::Diagnostics,
}

/// Solve only unique dense owners at terminal affine output leaves. All upstream
/// features and all unselected terms stay fixed. Duplicate output constraints are
/// refused, rather than silently selecting one occurrence. No oracle is consulted.
pub fn prefit(
    source: &OperatorProgram,
    inputs: &[Array2<f64>],
    targets: &Array2<f64>,
    trainable: &[usize],
    settings: linear_coefficient_fit::Settings,
    numeric_bytes: usize,
) -> Result<Fit, String> {
    if source.declarations.parameters != 0
        || !source.declarations.domains.is_empty()
        || source.declarations.slots.len() != inputs.len()
        || inputs.is_empty()
    {
        return Err("linear prefit requires parameter-free Raw local panels".into());
    }
    if !source.rules.is_empty()
        || source.nodes.iter().any(|n| {
            !matches!(
                n,
                Node::Raw { .. }
                    | Node::Constant { .. }
                    | Node::Affine { .. }
                    | Node::Pointwise { .. }
                    | Node::Hadamard { .. }
                    | Node::Concat { .. }
                    | Node::Transposed { .. }
                    | Node::Gain { .. }
            )
        })
    {
        return Err("linear prefit supports only Raw, constant, affine, pointwise, Hadamard, Concat, transpose and fixed gain nodes without rules".into());
    }
    if !settings.ridge.is_finite()
        || settings.ridge < 0.
        || settings
            .relative_rank_cutoff
            .is_some_and(|r| !r.is_finite() || !(0. ..1.).contains(&r))
    {
        return Err("invalid linear coefficient settings".into());
    }
    let n = targets.nrows();
    if n == 0 || targets.ncols() == 0 || !targets.iter().all(|v| v.is_finite()) {
        return Err("invalid target panel".into());
    }
    for (slot, x) in source.declarations.slots.iter().zip(inputs) {
        if !matches!(slot, Slot::Raw {width} if *width == x.ncols())
            || x.nrows() != n
            || !x.iter().all(|v| v.is_finite())
        {
            return Err("invalid aligned Raw input panel".into());
        }
    }
    let interfaces = source.interfaces().map_err(|e| e.to_string())?;
    if interfaces[source.output].width() != targets.ncols() {
        return Err("target width differs from program output".into());
    }
    let selected: BTreeSet<_> = trainable.iter().copied().collect();
    if selected.iter().any(|id| *id >= source.operators.len()) {
        return Err("trainable operator out of range".into());
    }
    fn leaves(
        p: &OperatorProgram,
        id: usize,
        offset: &mut usize,
        out: &mut Vec<(usize, usize)>,
        widths: &[crate::operator_program::Interface],
    ) {
        if let Node::Concat { parts } = &p.nodes[id] {
            for &part in parts {
                leaves(p, part, offset, out, widths);
            }
        } else {
            out.push((id, *offset));
            *offset += widths[id].width();
        }
    }
    let mut output_leaves = Vec::new();
    leaves(
        source,
        source.output,
        &mut 0,
        &mut output_leaves,
        &interfaces,
    );
    let mut counts = vec![0usize; source.operators.len()];
    for node in &source.nodes {
        for op in node.operators() {
            counts[op] += 1;
        }
    }
    let mut ancestors = vec![BTreeSet::new(); source.nodes.len()];
    for (id, node) in source.nodes.iter().enumerate() {
        for arg in node.arguments() {
            let inherited = ancestors[arg].clone();
            ancestors[id].insert(arg);
            ancestors[id].extend(inherited);
        }
    }
    let mut plans = Vec::new();
    let mut refusals = Vec::new();
    for &(id, offset) in &output_leaves {
        if output_leaves
            .iter()
            .filter(|(other, _)| *other == id)
            .count()
            > 1
        {
            refusals.push(format!("node {id}: duplicate output constraint"));
            continue;
        }
        if output_leaves
            .iter()
            .any(|(other, _)| *other != id && ancestors[*other].contains(&id))
        {
            refusals.push(format!("node {id}: upstream of another output leaf"));
            continue;
        }
        let Node::Affine { terms, bias } = &source.nodes[id] else {
            refusals.push(format!("node {id}: output leaf is not affine"));
            continue;
        };
        let mut pieces = Vec::new();
        for (input, op) in terms
            .iter()
            .map(|(input, op)| (Some(*input), *op))
            .chain(bias.iter().map(|op| (None, *op)))
        {
            if !selected.contains(&op) {
                continue;
            }
            let OperatorBody::Dense { present, .. } = &source.operators[op].body else {
                refusals.push(format!("operator {op}: not dense"));
                continue;
            };
            let owner_uses: usize = source
                .operators
                .iter()
                .enumerate()
                .filter(|(_, other)| Arc::ptr_eq(other, &source.operators[op]))
                .map(|(id, _)| counts[id])
                .sum();
            if owner_uses != 1 || present.iter().any(|v| !*v) {
                refusals.push(format!(
                    "operator {op}: reused owner or incomplete dense blocks"
                ));
                continue;
            }
            pieces.push((input, op));
        }
        if !pieces.is_empty() {
            plans.push((id, offset, pieces));
        }
    }
    for op in &selected {
        if !plans
            .iter()
            .any(|(_, _, pieces)| pieces.iter().any(|(_, id)| id == op))
            && !refusals
                .iter()
                .any(|reason| reason.starts_with(&format!("operator {op}:")))
        {
            refusals.push(format!(
                "operator {op}: no eligible terminal affine output use"
            ));
        }
    }
    // Explicit retained arrays, conservative thin-SVD products and program copies.
    // Evaluator/library allocator workspace is not a hard process-memory bound.
    let trace_cells = interfaces
        .iter()
        .try_fold(0usize, |sum, i| sum.checked_add(n.checked_mul(i.width())?))
        .ok_or("numeric plan overflow")?;
    let mut cells = trace_cells
        .checked_add(inputs.iter().map(|x| x.len()).sum())
        .ok_or("numeric plan overflow")?;
    for (id, _, pieces) in &plans {
        let p = pieces
            .iter()
            .map(|(input, _)| input.map_or(1, |v| interfaces[v].width()))
            .sum::<usize>();
        let k = interfaces[*id].width();
        let r = n.min(p);
        let block = n
            .checked_mul(p)
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| v.checked_add(n.checked_mul(k)?.checked_mul(6)?))
            .and_then(|v| v.checked_add(p.checked_mul(k)?.checked_mul(4)?))
            .and_then(|v| v.checked_add((n.checked_add(p)?).checked_mul(r)?.checked_mul(3)?))
            .ok_or("numeric plan overflow")?;
        cells = cells.checked_add(block).ok_or("numeric plan overflow")?;
    }
    cells = cells
        .checked_add(
            source
                .operators
                .iter()
                .map(|op| match &op.body {
                    OperatorBody::Dense { values, .. } => values.len(),
                    _ => 0,
                })
                .sum::<usize>(),
        )
        .ok_or("numeric plan overflow")?;
    let planned_numeric_bytes = cells.checked_mul(8).ok_or("numeric plan overflow")?;
    if planned_numeric_bytes > numeric_bytes {
        return Err(format!(
            "linear prefit numeric plan {planned_numeric_bytes} exceeds budget {numeric_bytes}"
        ));
    }
    let mut program = source.clone();
    let mut blocks = Vec::new();
    if !plans.is_empty() {
        let family = FamilyInputs {
            rows: n,
            slots: inputs.iter().cloned().map(SlotValues::Raw).collect(),
            layout: None,
        };
        let trace = source.execute(&family, false).map_err(|e| e.to_string())?;
        for (id, offset, pieces) in plans {
            let k = interfaces[id].width();
            let p = pieces
                .iter()
                .map(|(input, _)| input.map_or(1, |v| interfaces[v].width()))
                .sum();
            let mut x = Array2::ones((n, p));
            let mut b0 = Array2::zeros((p, k));
            let mut column = 0;
            for &(input, op) in &pieces {
                let OperatorBody::Dense { values, .. } = &source.operators[op].body else {
                    return Err("selected operator changed kind".into());
                };
                let width = values.ncols();
                if let Some(input) = input {
                    x.slice_mut(s![.., column..column + width])
                        .assign(&trace.values[input]);
                }
                b0.slice_mut(s![column..column + width, ..])
                    .assign(&values.t());
                column += width;
            }
            let adjusted = targets.slice(s![.., offset..offset + k]).to_owned() - &trace.values[id]
                + x.dot(&b0);
            let fit = linear_coefficient_fit::fit(&x, &adjusted, &b0, settings.clone())?;
            column = 0;
            for &(_, op) in &pieces {
                let OperatorBody::Dense {
                    values, precision, ..
                } = &mut Arc::make_mut(&mut program.operators[op]).body
                else {
                    return Err("selected operator changed kind".into());
                };
                let width = values.ncols();
                *values = fit
                    .coefficients
                    .slice(s![column..column + width, ..])
                    .t()
                    .to_owned();
                *precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
                column += width;
            }
            blocks.push(Block {
                node: id,
                operators: pieces.iter().map(|(_, op)| *op).collect(),
                diagnostics: fit.diagnostics,
            });
        }
    }
    Ok(Fit {program,report:Report {blocks,refusals,planned_numeric_bytes,budget_scope:"explicit f64 input/trace/design/SVD-product/coefficient arrays; excludes evaluator and linear algebra allocator workspace, metadata, and existing source/target storage"}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Interface, Law, Operator, Provenance};
    use ndarray::array;
    fn model() -> OperatorProgram {
        let scalar = Interface::native(1).unwrap();
        let dense = |v, bias| {
            Arc::new(
                Operator::dense(
                    "coefficient",
                    scalar.clone(),
                    if bias {
                        Interface::constant()
                    } else {
                        scalar.clone()
                    },
                    array![[v]],
                    exact_precision([v]).unwrap(),
                    Provenance::default(),
                )
                .unwrap(),
            )
        };
        OperatorProgram {
            declarations: Declarations {
                parameters: 0,
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
            },
            bases: vec![],
            rules: vec![],
            operators: vec![dense(2., false), dense(0.25, false), dense(0.5, true)],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Silu],
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: Some(2),
                },
            ],
            output: 3,
        }
    }
    fn values(p: &OperatorProgram, x: &Array2<f64>) -> Array2<f64> {
        p.execute(
            &FamilyInputs {
                rows: x.nrows(),
                slots: vec![SlotValues::Raw(x.clone())],
                layout: None,
            },
            false,
        )
        .unwrap()
        .values[p.output]
            .clone()
    }
    fn run(p: &OperatorProgram, x: &Array2<f64>, y: &Array2<f64>, ids: &[usize]) -> Fit {
        prefit(p, &[x.clone()], y, ids, Default::default(), 1 << 24).unwrap()
    }
    #[test]
    fn nonlinear_fixed_features_recover_terminal_terms_bias_and_unseen_inputs() {
        let p = model();
        let mut teacher = p.clone();
        for (id, v) in [(1, 3.), (2, -0.75)] {
            let OperatorBody::Dense {
                values, precision, ..
            } = &mut Arc::make_mut(&mut teacher.operators[id]).body
            else {
                panic!("test fixture must be dense")
            };
            values[(0, 0)] = v;
            *precision = exact_precision([v]).unwrap();
        }
        let x = array![[-2.], [-1.], [0.], [0.5], [1.], [2.]];
        let fit = run(&p, &x, &values(&teacher, &x), &[0, 1, 2]);
        assert_eq!(fit.report.blocks.len(), 1);
        assert_eq!(fit.report.blocks[0].operators, vec![1, 2]);
        assert!(Arc::ptr_eq(&p.operators[0], &fit.program.operators[0]));
        let unseen = array![[-1.3], [0.7], [3.1]];
        assert!(
            (&values(&teacher, &unseen) - &values(&fit.program, &unseen))
                .iter()
                .all(|v| v.abs() < 1e-10)
        );
    }
    #[test]
    fn reused_owner_and_self_dependence_are_frozen() {
        let mut p = model();
        p.nodes[3] = Node::Affine {
            terms: vec![(2, 0)],
            bias: None,
        };
        let x = array![[-1.], [1.], [2.]];
        let fit = run(&p, &x, &values(&p, &x), &[0]);
        assert!(fit.report.blocks.is_empty());
        assert!(fit
            .report
            .refusals
            .iter()
            .any(|v| v.contains("reused owner")));
        assert!(Arc::ptr_eq(&p.operators[0], &fit.program.operators[0]));
    }
    #[test]
    fn upstream_output_leaf_and_duplicate_constraints_are_refused() {
        let mut p = model();
        p.nodes.push(Node::Concat { parts: vec![1, 3] });
        p.output = 4;
        let x = array![[-1.], [0.], [1.]];
        let fit = run(&p, &x, &values(&p, &x), &[0, 1, 2]);
        assert_eq!(fit.report.blocks.len(), 1);
        assert_eq!(fit.report.blocks[0].node, 3);
        assert!(fit.report.refusals.iter().any(|v| v.contains("upstream")));
        p.nodes[4] = Node::Concat { parts: vec![3, 3] };
        let fit = run(&p, &x, &values(&p, &x), &[1, 2]);
        assert!(fit.report.blocks.is_empty());
        assert!(fit.report.refusals.iter().any(|v| v.contains("duplicate")));
    }
    #[test]
    fn transpose_and_pointer_alias_owner_uses_are_frozen() {
        let x = array![[-1.], [0.], [1.]];
        let mut p = model();
        p.nodes.push(Node::Transposed {
            input: 0,
            operator: 1,
        });
        let fit = run(&p, &x, &values(&p, &x), &[1]);
        assert!(fit.report.blocks.is_empty());
        let mut p = model();
        p.operators.push(p.operators[1].clone());
        p.nodes.push(Node::Affine {
            terms: vec![(0, 3)],
            bias: None,
        });
        let fit = run(&p, &x, &values(&p, &x), &[1]);
        assert!(fit.report.blocks.is_empty());
        assert!(fit
            .report
            .refusals
            .iter()
            .any(|v| v.contains("reused owner")));
    }
    #[test]
    fn fixed_terms_are_subtracted_and_budget_refuses_before_trace() {
        let mut p = model();
        p.nodes[3] = Node::Affine {
            terms: vec![(2, 1), (0, 0)],
            bias: Some(2),
        };
        let x = array![[-2.], [-1.], [0.], [1.], [2.]];
        let y = values(&p, &x) + 1.;
        let fit = run(&p, &x, &y, &[2]);
        assert_eq!(fit.report.blocks[0].operators, vec![2]);
        assert!((&values(&fit.program, &x) - &y)
            .iter()
            .all(|v| v.abs() < 1e-10));
        assert!(prefit(&p, &[x], &y, &[2], Default::default(), 0)
            .unwrap_err_string()
            .contains("budget"));
    }
    trait ErrorString {
        fn unwrap_err_string(self) -> String;
    }
    impl ErrorString for Result<Fit, String> {
        fn unwrap_err_string(self) -> String {
            match self {
                Err(e) => e,
                Ok(_) => panic!("expected refusal"),
            }
        }
    }
}
