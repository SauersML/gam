//! Adapter-free equation proposals on actual current-artifact boundary coordinates.
//! This finite coordinatewise grammar cannot express general linear mixing, learned
//! representations or persistent memory. A successful replacement is not by itself
//! recovery of computational organization.
use crate::{
    artifact::{Argument, Artifact, Callee},
    composed_rule_search::{self, Binary, Expr, Grammar, Unary},
    operator_program::{Coefficient, Interface, Law, Node, Operator, OperatorBody, Rule},
    program_regions::{self, Region},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rejection {
    pub expression: Expr,
    pub reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    pub expressions: Vec<Expr>,
    pub rejections: Vec<Rejection>,
    pub truncated: bool,
    pub intermediate_expressions: usize,
}
fn boundaries(
    artifact: &Artifact,
    region: &Region,
    args: &[usize],
) -> Result<(Vec<Interface>, Interface), String> {
    // Reuse the established closed-cut and intervention-coverage validation.
    program_regions::extract(artifact, region)?;
    if args.len() != region.native_reads.len()
        || args.iter().copied().collect::<BTreeSet<_>>()
            != region.native_reads.iter().copied().collect()
    {
        return Err("expression arguments must be a bijection of complete region boundary".into());
    }
    let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let inputs = args
        .iter()
        .map(|native| {
            artifact
                .place(*native)
                .map(|current| interfaces[current].clone())
                .ok_or("expression native argument absent".into())
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((inputs, interfaces[region.current_write].clone()))
}
struct Builder<'a> {
    artifact: &'a Artifact,
    inputs: Vec<Interface>,
    nodes: Vec<Node>,
    types: Vec<Interface>,
    operators: Vec<Operator>,
    identities: Vec<(Interface, usize)>,
    cache: BTreeMap<Expr, usize>,
}
impl Builder<'_> {
    fn identity(&mut self, interface: &Interface) -> usize {
        if let Some((_, id)) = self.identities.iter().find(|(ty, _)| ty == interface) {
            return *id;
        }
        let id = if let Some(id) = self.artifact.program.operators.iter().position(|op| {
            op.rows == *interface
                && op.cols == *interface
                && matches!(op.body, OperatorBody::Identity)
        }) {
            id
        } else {
            let id = self.artifact.program.operators.len() + self.operators.len();
            self.operators.push(Operator::identity(
                "expression fixed identity",
                interface.clone(),
            ));
            id
        };
        self.identities.push((interface.clone(), id));
        id
    }
    fn expr(&mut self, expr: &Expr) -> Result<usize, String> {
        if let Some(id) = self.cache.get(expr) {
            return Ok(*id);
        }
        let (node, ty) = match expr {
            Expr::Argument(index) => {
                let ty = self
                    .inputs
                    .get(*index)
                    .ok_or("expression references argument outside complete boundary")?
                    .clone();
                (Node::Param { index: *index }, ty)
            }
            Expr::Affine(_) => {
                return Err("learned Affine is forbidden in adapter-free synthesis".into());
            }
            Expr::Unary(op, input) => {
                let input = self.expr(input)?;
                let ty = self.types[input].clone();
                let law = match op {
                    Unary::Relu => Law::Relu,
                    Unary::Silu => Law::Silu,
                    Unary::Gelu => Law::Gelu,
                    Unary::GeluTanh => Law::GeluTanh,
                };
                (
                    Node::Pointwise {
                        input,
                        laws: vec![law; ty.group_count()],
                    },
                    ty,
                )
            }
            Expr::Binary(op, left, right) => {
                let left = self.expr(left)?;
                let mut right = self.expr(right)?;
                let ty = self.types[left].clone();
                if ty != self.types[right] {
                    return Err(
                        "coordinatewise binary operands require exactly equal interfaces".into(),
                    );
                }
                let node = match op {
                    Binary::Multiply => Node::Hadamard { left, right },
                    Binary::Add | Binary::Subtract => {
                        if *op == Binary::Subtract {
                            let id = self.nodes.len();
                            self.nodes.push(Node::Gain {
                                input: right,
                                coefficient: Coefficient::Number(-1.),
                            });
                            self.types.push(ty.clone());
                            right = id;
                        }
                        let identity = self.identity(&ty);
                        Node::Affine {
                            terms: vec![(left, identity), (right, identity)],
                            bias: None,
                        }
                    }
                };
                (node, ty)
            }
        };
        let id = self.nodes.len();
        self.nodes.push(node);
        self.types.push(ty);
        self.cache.insert(expr.clone(), id);
        Ok(id)
    }
}
fn compile(
    artifact: &Artifact,
    target: &Interface,
    expr: &Expr,
    inputs: Vec<Interface>,
) -> Result<(Rule, Vec<Operator>), String> {
    let mut builder = Builder {
        artifact,
        inputs: inputs.clone(),
        nodes: vec![],
        types: vec![],
        operators: vec![],
        identities: vec![],
        cache: BTreeMap::new(),
    };
    let output = builder.expr(expr)?;
    if &builder.types[output] != target {
        return Err("expression output interface differs from native write".into());
    }
    Ok((
        Rule {
            name: "adapter-free typed expression".into(),
            inputs,
            nodes: builder.nodes,
            output,
        },
        builder.operators,
    ))
}
pub fn enumerate(
    artifact: &Artifact,
    region: &Region,
    grammar: &Grammar,
    native_arguments: &[usize],
) -> Result<Inventory, String> {
    if grammar.affine {
        return Err("adapter-free grammar requires affine=false".into());
    }
    if grammar.arguments != region.native_reads.len() {
        return Err("grammar arguments must declare full region boundary arity".into());
    }
    let (inputs, target) = boundaries(artifact, region, native_arguments)?;
    let raw = composed_rule_search::enumerate(grammar)?;
    let mut result = Inventory {
        expressions: vec![],
        rejections: vec![],
        truncated: raw.truncated,
        intermediate_expressions: raw.intermediate_expressions,
    };
    for expression in raw.expressions {
        match compile(artifact, &target, &expression, inputs.clone()) {
            Ok(_) => result.expressions.push(expression),
            Err(reason) => result.rejections.push(Rejection { expression, reason }),
        }
    }
    Ok(result)
}
/// All cut inputs remain declared, including unused inputs (e.g. exact cancellation).
/// No donor/output/ambient state, coordinate adapter or learned matrix is supplied.
pub fn apply(
    artifact: &Artifact,
    region: &Region,
    expression: &Expr,
    native_arguments: &[usize],
) -> Result<Artifact, String> {
    let (inputs, target) = boundaries(artifact, region, native_arguments)?;
    let (rule, operators) = compile(artifact, &target, expression, inputs)?;
    let same = |stored: &Rule| {
        stored.inputs == rule.inputs && stored.nodes == rule.nodes && stored.output == rule.output
    };
    let callee = if operators.is_empty() {
        artifact
            .program
            .rules
            .iter()
            .position(same)
            .map_or(Callee::New(rule), Callee::Existing)
    } else {
        Callee::New(rule)
    };
    let result = artifact.replace_block(
        "adapter-free-expression",
        callee,
        native_arguments
            .iter()
            .copied()
            .map(Argument::Native)
            .collect(),
        region.native_write,
        operators,
    )?;
    if result.controls.len() != artifact.controls.len()
        || result.exceptions.len() != artifact.exceptions.len()
    {
        return Err("expression removes declared control/exception coverage".into());
    }
    crate::native_control::validate_shape(&result)?;
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{
        Declarations, FamilyInputs, OperatorProgram, SequenceLayout, Slot, SlotValues,
    };
    use std::sync::Arc;
    fn base(nodes: Vec<Node>, slots: usize) -> Artifact {
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }; slots],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![Arc::new(Operator::identity(
                "native add",
                Interface::native(2).expect("interface"),
            ))],
            rules: vec![],
            output: nodes.len() - 1,
            nodes,
        };
        Artifact::native(&p).expect("native artifact")
    }
    fn cut(a: &Artifact, internal: &[usize]) -> Region {
        program_regions::propose_regions(
            a,
            program_regions::Limits {
                max_internal_nodes: 16,
                max_inputs: 8,
                max_regions: 1000,
                max_states: 1000,
            },
        )
        .expect("inventory")
        .regions
        .into_iter()
        .find(|r| r.native_write == a.program.output && r.current_internal_nodes == internal)
        .expect("requested cut")
    }
    fn family(inputs: Vec<ndarray::Array2<f64>>) -> FamilyInputs {
        FamilyInputs {
            rows: 2,
            slots: inputs.into_iter().map(SlotValues::Raw).collect(),
            layout: Some(SequenceLayout {
                sequence: vec![0; 2],
                position: vec![0, 1],
            }),
        }
    }
    fn output(a: &Artifact, x: &FamilyInputs) -> ndarray::Array2<f64> {
        a.program.execute(x, false).expect("execute").values[a.program.output].clone()
    }
    #[test]
    fn alternate_factorization_not_literal_native_dag_with_controls_and_decode() {
        let a = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::Hadamard { left: 0, right: 1 },
                Node::Hadamard { left: 0, right: 2 },
                Node::Affine {
                    terms: vec![(3, 0), (4, 0)],
                    bias: None,
                },
            ],
            3,
        );
        let r = cut(&a, &[3, 4, 5]);
        let expression = Expr::Binary(
            Binary::Multiply,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Binary(
                Binary::Add,
                Box::new(Expr::Argument(1)),
                Box::new(Expr::Argument(2)),
            )),
        );
        let grammar = Grammar {
            arguments: 3,
            max_operations: 2,
            max_expressions: 1000,
            unary: vec![],
            binary: vec![Binary::Add, Binary::Multiply],
            affine: false,
        };
        assert!(
            enumerate(&a, &r, &grammar, &r.native_reads)
                .expect("equation inventory")
                .expressions
                .contains(&expression)
        );
        let b = apply(&a, &r, &expression, &r.native_reads).expect("new equation");
        assert!(
            b.program
                .operators
                .iter()
                .all(|o| matches!(o.body, OperatorBody::Identity))
        );
        let body = b.program.rules.last().expect("synthesized rule");
        assert!(body.nodes.iter().any(|n|matches!(n,Node::Hadamard{left:_,right} if matches!(body.nodes[*right],Node::Affine{..}))));
        assert!(a.program.nodes.iter().all(|n|!matches!(n,Node::Hadamard{left:_,right} if matches!(a.program.nodes[*right],Node::Affine{..}))));
        for gains in [
            [1., 1., 1.],
            [-1., 1., 1.],
            [1., 2., 1.],
            [1., 1., -2.],
            [2., -1., 0.5],
        ] {
            let x = family(vec![
                ndarray::array![[1., 2.], [3., 4.]] * gains[0],
                ndarray::array![[2., -1.], [1., 0.5]] * gains[1],
                ndarray::array![[1., 3.], [-2., 2.]] * gains[2],
            ]);
            assert_eq!(output(&a, &x), output(&b, &x));
            let bytes = b.to_bytes().expect("save");
            let replay = Artifact::from_bytes(&bytes, &a.program.declarations).expect("decode");
            assert_eq!(bytes, replay.to_bytes().expect("canonical"));
            assert_eq!(output(&a, &x), output(&replay, &x));
        }
    }
    #[test]
    fn unused_boundary_cancellation_and_independent_control_negative() {
        let a = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Gain {
                    input: 1,
                    coefficient: Coefficient::Number(-1.),
                },
                Node::Affine {
                    terms: vec![(0, 0), (1, 0), (2, 0)],
                    bias: None,
                },
            ],
            2,
        );
        let r = cut(&a, &[2, 3]);
        let b = apply(&a, &r, &Expr::Argument(0), &r.native_reads).expect("cancelled unused input");
        assert_eq!(b.program.rules.last().expect("rule").inputs.len(), 2);
        for gain in [-2., 0., 1., 3.] {
            let x = family(vec![
                ndarray::array![[1., 2.], [3., 4.]],
                ndarray::array![[2., 1.], [4., 8.]] * gain,
            ]);
            assert_eq!(output(&a, &x), output(&b, &x));
        }
        let negative = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::Gain {
                    input: 2,
                    coefficient: Coefficient::Number(-1.),
                },
                Node::Affine {
                    terms: vec![(0, 0), (1, 0), (3, 0)],
                    bias: None,
                },
            ],
            3,
        );
        let r = cut(&negative, &[3, 4]);
        let wrong = apply(&negative, &r, &Expr::Argument(0), &r.native_reads)
            .expect("clean shortcut proposal");
        let branch = ndarray::array![[2., 1.], [4., 8.]];
        let clean = family(vec![
            ndarray::array![[1., 2.], [3., 4.]],
            branch.clone(),
            branch.clone(),
        ]);
        assert_eq!(output(&negative, &clean), output(&wrong, &clean));
        for gains in [[0.5, 1.], [1., -1.], [2., 0.5]] {
            let edited = family(vec![
                ndarray::array![[1., 2.], [3., 4.]],
                &branch * gains[0],
                &branch * gains[1],
            ]);
            assert_ne!(output(&negative, &edited), output(&wrong, &edited));
        }
    }
    #[test]
    fn adapters_and_hidden_arguments_refused() {
        let a = base(
            vec![Node::Raw { slot: 0 }, Node::Hadamard { left: 0, right: 0 }],
            1,
        );
        let r = cut(&a, &[1]);
        assert!(
            apply(
                &a,
                &r,
                &Expr::Affine(Box::new(Expr::Argument(0))),
                &r.native_reads
            )
            .is_err()
        );
        assert!(apply(&a, &r, &Expr::Argument(1), &r.native_reads).is_err());
        assert!(apply(&a, &r, &Expr::Argument(0), &[]).is_err());
        let grammar = Grammar {
            arguments: 1,
            max_operations: 1,
            max_expressions: 2,
            unary: vec![Unary::Relu],
            binary: vec![Binary::Add],
            affine: false,
        };
        assert!(
            enumerate(&a, &r, &grammar, &r.native_reads)
                .expect("bounded inventory")
                .truncated
        );
    }
}
