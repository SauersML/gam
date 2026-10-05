//! Finite additive native matrix edits lowered before downstream nonlinearities.
//! This is an executable condition-on-edit graph, not an affine response model.
//! All Affine uses of one shared stored matrix (including rule invocations) are
//! edited together. Other operator uses are explicitly rejected. Real-algebra
//! equivalence does not imply floating-point bit parity with literal matrix edits.
use crate::{
    intervention_program::{self, Compiled, Control, ControlValue},
    operator_program::{
        FamilyInputs, Node, Operator, OperatorBody, OperatorProgram, exact_precision,
    },
};
use ndarray::Array2;
use std::sync::Arc;

pub struct Family {
    pub compiled: Compiled,
    /// Root node identities in the executable controlled program.
    pub node_mapping: Vec<usize>,
    /// Fixed dense edit coordinates, stored in the ordinary program.
    pub direction_operators: Vec<usize>,
    original: OperatorProgram,
    target_operator: usize,
}
impl Family {
    pub fn inputs(&self, base: &FamilyInputs, amplitudes: &[f64]) -> Result<FamilyInputs, String> {
        self.compiled.family(
            base,
            &amplitudes
                .iter()
                .copied()
                .map(ControlValue::GlobalScale)
                .collect::<Vec<_>>(),
        )
    }
    /// Independent teacher: edit the original stored matrix and execute original syntax.
    pub fn literal_native(&self, amplitudes: &[f64]) -> Result<OperatorProgram, String> {
        if amplitudes.len() != self.direction_operators.len()
            || amplitudes.iter().any(|a| !a.is_finite())
        {
            return Err("invalid native edit amplitudes".into());
        }
        let mut p = self.original.clone();
        let mut op = (*p.operators[self.target_operator]).clone();
        let OperatorBody::Dense {
            values, present, ..
        } = &op.body
        else {
            return Err("dense target required".into());
        };
        let mut edited = values.clone();
        for (&a, &id) in amplitudes.iter().zip(&self.direction_operators) {
            let OperatorBody::Dense {
                values: direction, ..
            } = &self.compiled.program.operators[id].body
            else {
                return Err("dense direction required".into());
            };
            for (x, d) in edited.iter_mut().zip(direction) {
                *x += a * d;
            }
        }
        let precision = exact_precision(edited.iter().copied()).map_err(|e| e.to_string())?;
        op.body = OperatorBody::Dense {
            values: edited,
            present: present.clone(),
            precision,
        };
        p.operators[self.target_operator] = Arc::new(op);
        p.interfaces().map_err(|e| e.to_string())?;
        Ok(p)
    }
}

/// Declare W(a) = W + sum_j a_j D_j, with arbitrary finite dense directions.
/// Each amplitude is one global scalar, shared across rows and all matrix uses.
/// Candidate programs may process these controls inside their nonlinearities;
/// this API imposes no affine constraint on their final edit response.
pub fn build(
    native: &OperatorProgram,
    target: usize,
    directions: &[Array2<f64>],
) -> Result<Family, String> {
    native.interfaces().map_err(|e| e.to_string())?;
    let op = native
        .operators
        .get(target)
        .ok_or("native edit target out of range")?;
    if directions.is_empty()
        || !matches!(&op.body, OperatorBody::Dense { present, .. } if present.iter().all(|p| *p))
    {
        return Err("nonempty edit coordinates and all-present dense target required".into());
    }
    let mut uses = 0;
    for node in native
        .nodes
        .iter()
        .chain(native.rules.iter().flat_map(|r| &r.nodes))
    {
        if let Node::Affine { terms, bias } = node {
            if *bias == Some(target) {
                return Err("native parameter edit bias use unsupported".into());
            }
            uses += terms.iter().filter(|(_, id)| *id == target).count();
        } else if node.operators().contains(&target) {
            return Err(
                "native parameter edit supports all Affine terms only; unsupported shared use"
                    .into(),
            );
        }
    }
    if uses == 0 {
        return Err("native edit target has no Affine use".into());
    }
    let mut augmented = native.clone();
    let mut ids = vec![];
    for values in directions {
        if values.dim() != (op.rows.width(), op.cols.width())
            || values.iter().any(|x| !x.is_finite())
            || !values.iter().any(|x| *x != 0.)
        {
            return Err("invalid native edit direction shape/values".into());
        }
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        let direction = Operator::dense(
            "fixed native edit coordinate",
            op.rows.clone(),
            op.cols.clone(),
            values.clone(),
            precision,
            Default::default(),
        )
        .map_err(|e| e.to_string())?;
        let OperatorBody::Dense { values: stored, .. } = &direction.body else {
            return Err("native edit direction construction was not dense".into());
        };
        if !stored
            .iter()
            .zip(values)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err("native edit direction not exactly stored".into());
        }
        ids.push(augmented.operators.len());
        augmented.operators.push(Arc::new(direction));
    }
    for node in augmented
        .nodes
        .iter_mut()
        .chain(augmented.rules.iter_mut().flat_map(|r| &mut r.nodes))
    {
        if let Node::Affine { terms, .. } = node {
            let additions: Vec<_> = terms
                .iter()
                .filter(|(_, id)| *id == target)
                .flat_map(|(input, _)| ids.iter().map(move |id| (*input, *id)))
                .collect();
            terms.extend(additions);
        }
    }
    let controls = ids
        .iter()
        .map(|id| Control::GlobalOperatorScale { operator: *id })
        .collect::<Vec<_>>();
    let compiled = intervention_program::compile(&augmented, &controls)?;
    let node_mapping = compiled.root_mapping.clone();
    Ok(Family {
        compiled,
        node_mapping,
        direction_operators: ids,
        original: native.clone(),
        target_operator: target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Interface, Law, Rule, Slot};
    use ndarray::array;
    fn source() -> OperatorProgram {
        let i = Interface::native(1).unwrap();
        let op = Operator::dense(
            "upstream",
            i.clone(),
            i.clone(),
            array![[0.25]],
            exact_precision([0.25]).unwrap(),
            Default::default(),
        )
        .unwrap();
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![Arc::new(op)],
            rules: vec![Rule {
                name: "shared nonlinear".into(),
                inputs: vec![i],
                nodes: vec![
                    Node::Param { index: 0 },
                    Node::Affine {
                        terms: vec![(0, 0)],
                        bias: None,
                    },
                    Node::Pointwise {
                        input: 1,
                        laws: vec![Law::Relu],
                    },
                ],
                output: 2,
            }],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Relu],
                },
                Node::Call {
                    rule: 0,
                    arguments: vec![0],
                },
                Node::Concat { parts: vec![2, 3] },
            ],
            output: 4,
        }
    }
    #[test]
    fn rejects_unsupported_shared_use() {
        let mut p = source();
        p.nodes.push(Node::Transposed {
            input: 0,
            operator: 0,
        });
        assert!(
            build(&p, 0, &[array![[1.]]])
                .err()
                .unwrap()
                .contains("unsupported")
        );
    }
}
