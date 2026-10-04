//! Exact native primitive-unary initialization: a capacity control, not rule discovery.
//! Coefficients are copied without fitting or projection; every proposal map remains paid.
use crate::composed_rule_search::{Expr, Proposal, Unary};
use crate::operator_program::{Law, Node, Operator, OperatorProgram, exact_precision};
use ndarray::Array2;
use std::sync::Arc;

/// Explicit original graph boundaries; no layer or unit correspondence is inferred.
#[derive(Clone, Copy, Debug)]
pub struct NativeUse {
    pub input: usize,
    pub read: usize,
    pub active: usize,
    pub write: usize,
}

fn affine(
    program: &OperatorProgram,
    node: usize,
    input: usize,
) -> Result<(usize, Option<usize>), String> {
    match program.nodes.get(node) {
        Some(Node::Affine { terms, bias }) if terms.len() == 1 && terms[0].0 == input => {
            Ok((terms[0].1, *bias))
        }
        _ => Err("native initialization requires a direct single-term affine boundary".into()),
    }
}
fn values(program: &OperatorProgram, index: usize) -> Result<Array2<f64>, String> {
    let matrix = program
        .operators
        .get(index)
        .ok_or("native operator index")?
        .matrix();
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite native initialization".into());
    }
    Ok(matrix)
}
fn replace(program: &mut OperatorProgram, index: usize, values: Array2<f64>) -> Result<(), String> {
    let old = program
        .operators
        .get(index)
        .ok_or("proposal operator index")?;
    if values.dim() != (old.rows.width(), old.cols.width()) {
        return Err("native/proposal map width mismatch".into());
    }
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    program.operators[index] = Arc::new(
        Operator::dense(
            old.name.clone(),
            old.rows.clone(),
            old.cols.clone(),
            values,
            precision,
            old.provenance.clone(),
        )
        .map_err(|e| e.to_string())?,
    );
    Ok(())
}

/// Initialize only `Unary(Argument(0))`, with native hidden width and exact law.
/// Atomic on error. Native source may be a clean standalone graph or full native graph.
pub fn initialize(
    proposal: &Proposal,
    expression: &Expr,
    native: &OperatorProgram,
    uses: &[NativeUse],
) -> Result<Proposal, String> {
    let law = match expression {
        Expr::Unary(unary, input) if **input == Expr::Argument(0) => match unary {
            Unary::Relu => Law::Relu,
            Unary::Silu => Law::Silu,
            Unary::Gelu => Law::Gelu,
            Unary::GeluTanh => Law::GeluTanh,
        },
        _ => {
            return Err(
                "native initialization supports primitive unary argument-zero controls only".into(),
            );
        }
    };
    native.interfaces().map_err(|e| e.to_string())?;
    let mut out = Proposal {
        program: proposal.program.clone(),
        trainable: proposal.trainable.clone(),
        groups: proposal.groups.clone(),
    };
    let outputs = match &out.program.nodes[out.program.output] {
        Node::Concat { parts } if parts.len() == uses.len() && !uses.is_empty() => parts.clone(),
        _ => return Err("proposal/native use count mismatch".into()),
    };
    for (slot, (boundary, output)) in uses.iter().zip(outputs).enumerate() {
        let (reader, reader_bias) = affine(native, boundary.read, boundary.input)?;
        match native.nodes.get(boundary.active) {
            Some(Node::Pointwise { input, laws })
                if *input == boundary.read
                    && !laws.is_empty()
                    && laws.iter().all(|v| *v == law) => {}
            _ => return Err("native activation law or boundary mismatch".into()),
        }
        let (writer, writer_bias) = affine(native, boundary.write, boundary.active)?;
        let (call, dst_writer, dst_bias) = match &out.program.nodes[output] {
            Node::Affine {
                terms,
                bias: Some(bias),
            } if terms.len() == 1 => (terms[0].0, terms[0].1, *bias),
            _ => return Err("proposal writer boundary mismatch".into()),
        };
        let (rule, argument) = match &out.program.nodes[call] {
            Node::Call { rule, arguments } if arguments.len() == 1 => (*rule, arguments[0]),
            _ => return Err("proposal unary call mismatch".into()),
        };
        let body = out.program.rules.get(rule).ok_or("proposal rule index")?;
        if body.nodes
            != vec![
                Node::Param { index: 0 },
                Node::Pointwise {
                    input: 0,
                    laws: vec![law.clone()],
                },
            ]
            || body.output != 1
        {
            return Err("proposal body is not the declared primitive unary rule".into());
        }
        let (raw, dst_reader, dst_reader_bias) = match &out.program.nodes[argument] {
            Node::Affine {
                terms,
                bias: Some(bias),
            } if terms.len() == 1 => (terms[0].0, terms[0].1, *bias),
            _ => return Err("proposal reader boundary mismatch".into()),
        };
        if out.program.nodes[raw] != (Node::Raw { slot }) {
            return Err("proposal raw slot mismatch".into());
        }
        replace(&mut out.program, dst_reader, values(native, reader)?)?;
        replace(&mut out.program, dst_writer, values(native, writer)?)?;
        for (source, destination) in [(reader_bias, dst_reader_bias), (writer_bias, dst_bias)] {
            let shape = out.program.operators[destination].matrix().dim();
            let matrix = match source {
                Some(index) => values(native, index)?,
                None => Array2::zeros(shape),
            };
            replace(&mut out.program, destination, matrix)?;
        }
    }
    out.program.interfaces().map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composed_rule_search::{UseSpec, compile};
    use crate::operator_program::{FamilyInputs, SlotValues};
    fn fixture() -> (Proposal, Expr, OperatorProgram, Vec<NativeUse>) {
        let expression = Expr::Unary(Unary::GeluTanh, Box::new(Expr::Argument(0)));
        let proposal = compile(
            &expression,
            3,
            &[UseSpec {
                input_width: 2,
                output_width: 2,
            }; 2],
            73,
        )
        .expect("compile fixture");
        let mut native = proposal.program.clone();
        // Flatten the primitive calls into native pointwise nodes, preserving both uses.
        for node in &mut native.nodes {
            if let Node::Call { arguments, .. } = node {
                *node = Node::Pointwise {
                    input: arguments[0],
                    laws: vec![Law::GeluTanh],
                };
            }
        }
        for (index, operator) in native.operators.iter_mut().enumerate() {
            let old = operator.clone();
            let matrix = Array2::from_shape_fn(old.matrix().dim(), |(i, j)| {
                (index + i + j + 1) as f64 / 13.
            });
            *operator = Arc::new(
                Operator::dense(
                    old.name.clone(),
                    old.rows.clone(),
                    old.cols.clone(),
                    matrix.clone(),
                    exact_precision(matrix.iter().copied()).expect("precision"),
                    Default::default(),
                )
                .expect("native map"),
            );
        }
        (
            proposal,
            expression,
            native,
            vec![
                NativeUse {
                    input: 0,
                    read: 1,
                    active: 2,
                    write: 3,
                },
                NativeUse {
                    input: 4,
                    read: 5,
                    active: 6,
                    write: 7,
                },
            ],
        )
    }
    #[test]
    fn multiple_native_uses_and_biases_agree() {
        let (proposal, expression, native, uses) = fixture();
        let initialized =
            initialize(&proposal, &expression, &native, &uses).expect("native initialization");
        let inputs = FamilyInputs {
            rows: 4,
            slots: (0..2)
                .map(|slot| {
                    SlotValues::Raw(Array2::from_shape_fn((4, 2), |(i, j)| {
                        (i + j + slot) as f64 / 5. - 1.
                    }))
                })
                .collect(),
            layout: None,
        };
        let a = native.execute(&inputs, false).expect("native forward");
        let b = initialized
            .program
            .execute(&inputs, false)
            .expect("initialized forward");
        assert_eq!(
            a.values[native.output],
            b.values[initialized.program.output]
        );
    }
    #[test]
    fn missing_bias_zeroed_and_width_or_law_rejected() {
        let (proposal, expression, mut native, uses) = fixture();
        for boundary in &uses {
            for index in [boundary.read, boundary.write] {
                if let Node::Affine { bias, .. } = &mut native.nodes[index] {
                    *bias = None;
                }
            }
        }
        let initialized =
            initialize(&proposal, &expression, &native, &uses).expect("zero missing biases");
        for index in [1, 3, 5, 7] {
            assert!(
                initialized.program.operators[index]
                    .matrix()
                    .iter()
                    .all(|v| *v == 0.)
            );
        }
        let wrong = Expr::Unary(Unary::Relu, Box::new(Expr::Argument(0)));
        assert!(initialize(&proposal, &wrong, &native, &uses).is_err());
        let small = compile(
            &expression,
            2,
            &[UseSpec {
                input_width: 2,
                output_width: 2,
            }; 2],
            3,
        )
        .expect("wrong width fixture");
        assert!(initialize(&small, &expression, &native, &uses).is_err());
    }
}
