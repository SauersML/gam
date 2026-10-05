//! Conservative fixed-feature local coefficient proposals. Acceptance belongs to the caller.
use crate::{
    linear_coefficient_fit,
    operator_program::{
        FamilyInputs, Node, OperatorBody, OperatorProgram, Slot, SlotValues, exact_precision,
    },
};
use ndarray::{Array2, s};
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
    pub row_tile_rows: usize,
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
    // Include resident panels and source/fitted coefficient storage. Blocks are
    // solved sequentially; peak workspace is a maximum, not the sum of all plans.
    // QR reflectors/SVD operate only on a tile plus at most p retained rows.
    let overflow = || "linear prefit numeric plan overflow".to_string();
    let input_width = inputs
        .iter()
        .try_fold(0usize, |sum, x| sum.checked_add(x.ncols()))
        .ok_or_else(overflow)?;
    let trace_width = interfaces
        .iter()
        .try_fold(0usize, |sum, i| sum.checked_add(i.width()))
        .ok_or_else(overflow)?;
    let source_cells = source
        .operators
        .iter()
        .try_fold(0usize, |sum, op| {
            let cells = match &op.body {
                OperatorBody::Dense { values, .. } => values.len(),
                OperatorBody::LowRank { left, right, .. } => left.len().checked_add(right.len())?,
                OperatorBody::Diagonal { values, .. } => values.len(),
                OperatorBody::Identity => 0,
            };
            sum.checked_add(cells)
        })
        .ok_or_else(overflow)?;
    let base_cells = inputs
        .iter()
        .try_fold(targets.len(), |sum, x| sum.checked_add(x.len()))
        .and_then(|v| v.checked_add(source_cells.checked_mul(2)?))
        .ok_or_else(overflow)?;
    let mut tile_plans = Vec::new();
    let mut peak_cells = base_cells;
    for (id, _, pieces) in &plans {
        let p = pieces
            .iter()
            .try_fold(0usize, |sum, (input, _)| {
                sum.checked_add(input.map_or(1, |v| interfaces[v].width()))
            })
            .ok_or_else(overflow)?;
        let k = interfaces[*id].width();
        // Conservative explicit QR copies/reflectors, small SVD vectors/copies,
        // coefficient/RHS products, and per-tile evaluator values/products.
        let fixed = p
            .checked_mul(n.min(p))
            .and_then(|v| v.checked_mul(20))
            .and_then(|v| v.checked_add(p.checked_mul(k)?.checked_mul(12)?))
            .and_then(|v| v.checked_add(base_cells))
            .ok_or_else(overflow)?;
        let per_row = trace_width
            .checked_mul(2)
            .and_then(|v| v.checked_add(input_width))
            .and_then(|v| v.checked_add(p.checked_mul(8)?))
            .and_then(|v| v.checked_add(k.checked_mul(8)?))
            .ok_or_else(overflow)?;
        let available = (numeric_bytes / 8).checked_sub(fixed).ok_or(
            "linear prefit budget cannot hold reduced QR/SVD workspace and resident panels",
        )?;
        let tile_rows = (available / per_row).min(4096).min(n);
        if tile_rows == 0 {
            return Err("linear prefit budget cannot hold one execution/QR row tile".into());
        }
        peak_cells = peak_cells.max(
            fixed
                .checked_add(per_row.checked_mul(tile_rows).ok_or_else(overflow)?)
                .ok_or_else(overflow)?,
        );
        tile_plans.push(tile_rows);
    }
    let planned_numeric_bytes = peak_cells.checked_mul(8).ok_or_else(overflow)?;
    if planned_numeric_bytes > numeric_bytes {
        return Err(format!(
            "linear prefit numeric plan {planned_numeric_bytes} exceeds budget {numeric_bytes}"
        ));
    }
    let mut program = source.clone();
    let mut blocks = Vec::new();
    for ((id, offset, pieces), tile_rows) in plans.into_iter().zip(tile_plans) {
        let k = interfaces[id].width();
        let p = pieces
            .iter()
            .map(|(input, _)| input.map_or(1, |v| interfaces[v].width()))
            .sum();
        let mut b0 = Array2::zeros((p, k));
        let mut column = 0;
        for &(_, op) in &pieces {
            let OperatorBody::Dense { values, .. } = &source.operators[op].body else {
                return Err("selected operator changed kind".into());
            };
            let width = values.ncols();
            b0.slice_mut(s![column..column + width, ..])
                .assign(&values.t());
            column += width;
        }
        let fit = linear_coefficient_fit::fit_residual_streaming(
            n,
            &b0,
            settings.clone(),
            tile_rows,
            |rows| {
                let count = rows.end - rows.start;
                let family = FamilyInputs {
                    rows: count,
                    slots: inputs
                        .iter()
                        .map(|x| SlotValues::Raw(x.slice(s![rows.clone(), ..]).to_owned()))
                        .collect(),
                    layout: None,
                };
                let trace = source.execute(&family, false).map_err(|e| e.to_string())?;
                let mut x = Array2::ones((count, p));
                let mut column = 0;
                for &(input, op) in &pieces {
                    let width = source.operators[op].cols.width();
                    if let Some(input) = input {
                        x.slice_mut(s![.., column..column + width])
                            .assign(&trace.values[input]);
                    }
                    column += width;
                }
                // Work directly with Y - current output. Frozen terms cancel
                // without constructing (Y - output + X B0) and subtracting X B0.
                let residual =
                    targets.slice(s![rows, offset..offset + k]).to_owned() - &trace.values[id];
                Ok((x, residual))
            },
        )?;
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
            row_tile_rows: tile_rows,
        });
    }
    Ok(Fit {
        program,
        report: Report {
            blocks,
            refusals,
            planned_numeric_bytes,
            budget_scope: "resident input/target panels, source/fitted coefficients, bounded row-tile traces and conservative explicit Householder/reduced-SVD arrays; blocks solved sequentially; excludes allocator/evaluator/library scratch and metadata, not a hard process-RSS guarantee; every row used in reduction and residual replay",
        },
    })
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
        assert!(
            fit.report
                .refusals
                .iter()
                .any(|v| v.contains("reused owner"))
        );
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
        assert!(
            fit.report
                .refusals
                .iter()
                .any(|v| v.contains("reused owner"))
        );
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
        assert!(
            (&values(&fit.program, &x) - &y)
                .iter()
                .all(|v| v.abs() < 1e-10)
        );
        assert!(
            prefit(&p, &[x], &y, &[2], Default::default(), 0)
                .unwrap_err_string()
                .contains("budget")
        );
    }
    #[test]
    fn tall_program_prefit_executes_bounded_tiles_and_preserves_frozen_features() {
        let source = model();
        let x = Array2::from_shape_fn((20001, 1), |(r, _)| r as f64 / 5000. - 2.);
        let mut teacher = source.clone();
        if let OperatorBody::Dense { values, .. } =
            &mut Arc::make_mut(&mut teacher.operators[1]).body
        {
            values[[0, 0]] = 3.;
        }
        if let OperatorBody::Dense { values, .. } =
            &mut Arc::make_mut(&mut teacher.operators[2]).body
        {
            values[[0, 0]] = -0.75;
        }
        let y = values(&teacher, &x);
        // Resident panels fit, but a full-height trace/design/SVD plan does not.
        let budget = 700_000;
        let solved = prefit(
            &source,
            &[x.clone()],
            &y,
            &[1, 2],
            Default::default(),
            budget,
        )
        .unwrap();
        assert!(solved.report.planned_numeric_bytes <= budget);
        assert_eq!(solved.report.blocks.len(), 1);
        assert!(solved.report.blocks[0].row_tile_rows < x.nrows());
        assert!(solved.report.blocks[0].diagnostics.row_tiles > 1);
        assert_eq!(
            solved.report.blocks[0].diagnostics.method,
            "streaming_householder_qr_reduced_svd"
        );
        assert!(Arc::ptr_eq(
            &source.operators[0],
            &solved.program.operators[0]
        ));
        assert!(
            (&values(&solved.program, &x) - &y)
                .iter()
                .all(|v| v.abs() < 1e-10)
        );
        assert!(prefit(&source, &[x], &y, &[1, 2], Default::default(), 1).is_err());
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
