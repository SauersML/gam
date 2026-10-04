//! Native-grounded executable MLP proposal rules. Fitting and proposal generation
//! do not decide acceptance: the finite bank pays its complete serialized C32 and
//! is measured on decoded programs by Local and RunCheck.
use crate::artifact::{Argument, Artifact, Callee};
use crate::operator_program::{Interface, Law, Node, Operator, Provenance, Rule, exact_precision};
use crate::run_check::LayerNodes;
use ndarray::{Array2, Axis};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Feature {
    Linear(usize),
    Gelu(usize),
    Product(usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Family {
    Affine,
    Gelu,
    Products,
}

/// Complete declared feature families; pair products enter together, independent
/// of marginal correlation or the acceptance of any smaller feature set.
pub fn features(reads: usize, family: Family) -> Vec<Feature> {
    let mut out: Vec<_> = (0..reads).map(Feature::Linear).collect();
    if family != Family::Affine {
        out.extend((0..reads).map(Feature::Gelu));
    }
    if family == Family::Products {
        for a in 0..reads {
            for b in a..reads {
                out.push(Feature::Product(a, b));
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct NativeRule {
    /// Rows are explicitly transmitted input directions (m x input width).
    pub reads: Array2<f64>,
    /// Rows are explicitly transmitted output directions (K x output width).
    pub writes: Array2<f64>,
    pub features: Vec<Feature>,
    /// Intercept first, then one row per feature; columns are output amplitudes.
    pub coefficients: Array2<f64>,
}

/// Evaluate the exact declared proposal feature table, including its intercept.
pub fn design(amplitudes: &Array2<f64>, selected: &[Feature]) -> Result<Array2<f64>, String> {
    let mut out = Array2::ones((amplitudes.nrows(), selected.len() + 1));
    for (column, feature) in selected.iter().enumerate() {
        for row in 0..amplitudes.nrows() {
            let get = |i: usize| {
                amplitudes
                    .get((row, i))
                    .copied()
                    .ok_or("feature read outside amplitude table")
            };
            out[[row, column + 1]] = match *feature {
                Feature::Linear(a) => get(a)?,
                Feature::Gelu(a) => Law::GeluTanh.apply(get(a)?),
                Feature::Product(a, b) => get(a)? * get(b)?,
            };
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite native rule feature".into());
    }
    Ok(out)
}

fn dense(
    name: &str,
    rows: Interface,
    cols: Interface,
    values: Array2<f64>,
) -> Result<Operator, String> {
    // Transposed writer/response arrays may own Fortran storage. The literal
    // wire codec transmits canonical row order; retain every logical value.
    let values = values.as_standard_layout().to_owned();
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    Operator::dense(name, rows, cols, values, precision, Provenance::default())
        .map_err(|e| e.to_string())
}
fn primitive(
    program: &mut crate::operator_program::OperatorProgram,
    product: bool,
) -> Result<usize, String> {
    let scalar = Interface::native(1).map_err(|e| e.to_string())?;
    let body = if product {
        Rule {
            name: "product(a,b)=a*b".into(),
            inputs: vec![scalar.clone(), scalar],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Param { index: 1 },
                Node::Hadamard { left: 0, right: 1 },
            ],
            output: 2,
        }
    } else {
        Rule {
            name: "tanh GELU(a)".into(),
            inputs: vec![scalar],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Pointwise {
                    input: 0,
                    laws: vec![Law::GeluTanh],
                },
            ],
            output: 1,
        }
    };
    if let Some(index) = program
        .rules
        .iter()
        .position(|r| r.inputs == body.inputs && r.nodes == body.nodes && r.output == body.output)
    {
        return Ok(index);
    }
    let index = program.rules.len();
    program.rules.push(body);
    Ok(index)
}

/// Compile every reader, coefficient, writer, literal and reusable law into the
/// artifact. No fitted evaluator closure or unpriced library survives execution.
pub fn with_native_rule(
    artifact: &Artifact,
    name: &str,
    layer: &LayerNodes,
    fit: &NativeRule,
) -> Result<Artifact, String> {
    let mut base = artifact.clone();
    let interfaces = base.program.interfaces().map_err(|e| e.to_string())?;
    let input = interfaces[base
        .place(layer.normed)
        .ok_or("native rule input is not held")?]
    .clone();
    let output = interfaces[base
        .place(layer.mlp)
        .ok_or("native rule output is not held")?]
    .clone();
    let (m, k) = (fit.reads.nrows(), fit.writes.nrows());
    if m == 0
        || k == 0
        || fit.reads.ncols() != input.width()
        || fit.writes.ncols() != output.width()
        || fit.coefficients.dim() != (fit.features.len() + 1, k)
    {
        return Err("native rule dimensions disagree with held MLP interfaces".into());
    }
    if fit
        .reads
        .iter()
        .chain(fit.writes.iter())
        .chain(fit.coefficients.iter())
        .any(|x| !x.is_finite())
    {
        return Err("nonfinite native rule literal".into());
    }
    let gelu = fit
        .features
        .iter()
        .any(|f| matches!(f, Feature::Gelu(_)))
        .then(|| primitive(&mut base.program, false))
        .transpose()?;
    let product = fit
        .features
        .iter()
        .any(|f| matches!(f, Feature::Product(_, _)))
        .then(|| primitive(&mut base.program, true))
        .transpose()?;
    let scalar = Interface::native(1).map_err(|e| e.to_string())?;
    let amplitudes = Interface::native(k).map_err(|e| e.to_string())?;
    let first = base.program.operators.len();
    let mut operators = Vec::new();
    let mut nodes = vec![Node::Param { index: 0 }];
    let mut read_nodes = Vec::new();
    for row in fit.reads.outer_iter() {
        let index = first + operators.len();
        operators.push(dense(
            "native input direction",
            scalar.clone(),
            input.clone(),
            row.to_owned().insert_axis(Axis(0)),
        )?);
        nodes.push(Node::Affine {
            terms: vec![(0, index)],
            bias: None,
        });
        read_nodes.push(nodes.len() - 1);
    }
    let bias = first + operators.len();
    operators.push(dense(
        "native rule intercept",
        amplitudes.clone(),
        Interface::constant(),
        fit.coefficients.row(0).to_owned().insert_axis(Axis(1)),
    )?);
    let mut terms = Vec::new();
    for (index, feature) in fit.features.iter().enumerate() {
        let get = |i: usize| {
            read_nodes
                .get(i)
                .copied()
                .ok_or("native rule feature outside readers")
        };
        let node = match *feature {
            Feature::Linear(a) => get(a)?,
            Feature::Gelu(a) => {
                nodes.push(Node::Call {
                    rule: gelu.ok_or("GELU body missing")?,
                    arguments: vec![get(a)?],
                });
                nodes.len() - 1
            }
            Feature::Product(a, b) => {
                nodes.push(Node::Call {
                    rule: product.ok_or("product body missing")?,
                    arguments: vec![get(a)?, get(b)?],
                });
                nodes.len() - 1
            }
        };
        let op = first + operators.len();
        operators.push(dense(
            "native feature coefficients",
            amplitudes.clone(),
            scalar.clone(),
            fit.coefficients
                .row(index + 1)
                .to_owned()
                .insert_axis(Axis(1)),
        )?);
        terms.push((node, op));
    }
    nodes.push(Node::Affine {
        terms,
        bias: Some(bias),
    });
    let response = nodes.len() - 1;
    let writer = first + operators.len();
    operators.push(dense(
        "native output directions",
        output,
        amplitudes,
        fit.writes.t().to_owned(),
    )?);
    nodes.push(Node::Affine {
        terms: vec![(response, writer)],
        bias: None,
    });
    let rule = Rule {
        name: name.into(),
        inputs: vec![input],
        output: nodes.len() - 1,
        nodes,
    };
    base.replace_block(
        name,
        Callee::New(rule),
        vec![Argument::Native(layer.normed)],
        layer.mlp,
        operators,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{
        Change, CostCache, Edit, Episode, FamilyRun, RunCheck, structural_cost,
    };
    use crate::operator_program::{Declarations, FamilyInputs, OperatorProgram, Slot, SlotValues};
    use ndarray::array;
    use std::sync::Arc;
    #[test]
    fn multiresponse_rules_have_canonical_literals_and_decode_execution_parity() {
        let a = array![[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]];
        for k in [2, 4, 8] {
            let input = Interface::native(2).unwrap();
            let response = Interface::native(k).unwrap();
            let native = OperatorProgram {
                declarations: Declarations {
                    domains: vec![],
                    slots: vec![Slot::Raw { width: 2 }],
                    parameters: 0,
                },
                bases: vec![],
                rules: vec![],
                operators: vec![
                    Arc::new(
                        dense(
                            "native placeholder",
                            response.clone(),
                            input,
                            Array2::zeros((k, 2)),
                        )
                        .unwrap(),
                    ),
                    Arc::new(dense("head", response.clone(), response, Array2::eye(k)).unwrap()),
                ],
                nodes: vec![
                    Node::Raw { slot: 0 },
                    Node::Affine {
                        terms: vec![(0, 0)],
                        bias: None,
                    },
                    Node::Affine {
                        terms: vec![(1, 1)],
                        bias: None,
                    },
                ],
                output: 2,
            };
            let fit = NativeRule {
                reads: Array2::eye(2),
                writes: Array2::eye(k),
                features: vec![Feature::Linear(0), Feature::Gelu(1), Feature::Product(0, 1)],
                coefficients: Array2::from_shape_fn((4, k), |(r, c)| (r + c + 1) as f64 / 8.0),
            };
            // The writer transpose is Fortran-layout for K>1; the executable
            // literal codec requires canonical row storage, not a K=1 accident.
            assert!(!fit.writes.t().to_owned().is_standard_layout());
            let candidate = with_native_rule(
                &Artifact::native(&native).unwrap(),
                "multiresponse",
                &LayerNodes {
                    normed: 0,
                    mlp: 1,
                    ..LayerNodes::default()
                },
                &fit,
            )
            .unwrap()
            .f32_literals()
            .unwrap();
            assert!(
                candidate
                    .program
                    .operators
                    .iter()
                    .all(|op| op.matrix_cow().is_standard_layout())
            );
            let decoded =
                Artifact::from_bytes(&candidate.to_bytes().unwrap(), &native.declarations).unwrap();
            let family = FamilyInputs {
                rows: 3,
                slots: vec![SlotValues::Raw(a.clone())],
                layout: None,
            };
            let got = decoded.execute(&family).unwrap();
            let expected = design(&a, &fit.features).unwrap().dot(&fit.coefficients);
            for (x, y) in got.values[decoded.program.output]
                .iter()
                .zip(expected.iter())
            {
                assert!((x - y).abs() < 1e-13, "K={k}: {x} versus {y}");
            }
            let cost = structural_cost(&decoded, &mut CostCache::default()).unwrap();
            assert_eq!(cost.literals, (2 * k * k + 4 * k + 4) as u64);
        }
    }
    #[test]
    fn fit_features_use_the_executable_law_and_complete_joint_family() {
        let a = array![[-2.13, 0.12], [0.72, 3.17]];
        let table = design(&a, &[Feature::Gelu(0)]).unwrap();
        for row in 0..a.nrows() {
            assert_eq!(
                table[[row, 1]].to_bits(),
                Law::GeluTanh.apply(a[[row, 0]]).to_bits()
            );
        }
        assert_eq!(features(2, Family::Affine).len(), 2);
        assert_eq!(features(2, Family::Gelu).len(), 4);
        assert_eq!(features(2, Family::Products).len(), 7);
        assert!(features(2, Family::Products).contains(&Feature::Product(0, 1)));
    }
    #[test]
    fn joint_product_reaches_a_target_with_zero_marginal_correlations_and_decodes() {
        let a = array![[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]];
        let target = array![1.0, -1.0, -1.0, 1.0];
        assert_eq!(a.column(0).dot(&target), 0.0);
        assert_eq!(a.column(1).dot(&target), 0.0);
        let input = Interface::native(2).unwrap();
        let output = Interface::native(1).unwrap();
        let native = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                Arc::new(
                    dense("read a", output.clone(), input.clone(), array![[1.0, 0.0]]).unwrap(),
                ),
                Arc::new(dense("read b", output.clone(), input, array![[0.0, 1.0]]).unwrap()),
                Arc::new(
                    dense(
                        "head",
                        Interface::native(2).unwrap(),
                        output,
                        array![[1.0], [-1.0]],
                    )
                    .unwrap(),
                ),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(0, 1)],
                    bias: None,
                },
                Node::Hadamard { left: 1, right: 2 },
                Node::Affine {
                    terms: vec![(3, 2)],
                    bias: None,
                },
            ],
            output: 4,
        };
        let layer = LayerNodes {
            normed: 0,
            mlp: 3,
            ..LayerNodes::default()
        };
        let fit = NativeRule {
            reads: Array2::eye(2),
            writes: array![[1.0]],
            features: vec![Feature::Product(0, 1)],
            coefficients: array![[0.0], [1.0]],
        };
        let candidate =
            with_native_rule(&Artifact::native(&native).unwrap(), "joint", &layer, &fit)
                .unwrap()
                .f32_literals()
                .unwrap();
        let decoded =
            Artifact::from_bytes(&candidate.to_bytes().unwrap(), &native.declarations).unwrap();
        let family = FamilyInputs {
            rows: 4,
            slots: vec![SlotValues::Raw(a.clone())],
            layout: None,
        };
        let got = decoded.execute(&family).unwrap();
        assert_eq!(got.values[decoded.program.output].column(0), target.view());
        assert_eq!(design(&a, &fit.features).unwrap().column(1), target.view());
        assert_eq!(
            decoded
                .program
                .rules
                .iter()
                .filter(|r| r.nodes
                    == vec![
                        Node::Param { index: 0 },
                        Node::Param { index: 1 },
                        Node::Hadamard { left: 0, right: 1 }
                    ])
                .count(),
            1
        );
        let cost = structural_cost(&decoded, &mut CostCache::default()).unwrap();
        assert_eq!(cost.literals, 9); // readers4 + writer1 + coefficients2 + native head2.
        let run = FamilyRun {
            model: &native,
            family,
            readouts: 1,
            episodes: vec![
                Episode {
                    id: "represented input".into(),
                    group: "input".into(),
                    edits: vec![Edit {
                        node: 0,
                        rows: None,
                        columns: 0..1,
                        change: Change::Scale(0.5),
                    }],
                },
                Episode {
                    id: "unheld internal".into(),
                    group: "internal".into(),
                    edits: vec![Edit {
                        node: 1,
                        rows: None,
                        columns: 0..1,
                        change: Change::Scale(0.5),
                    }],
                },
            ],
        };
        let scores = run.episodes(&decoded).unwrap();
        assert!(scores[0].kl.abs() < 1e-14 && scores[0].native_effect > 0.01);
        assert_eq!(scores[0].unheld, 0);
        assert!(scores[1].kl > 0.01 && scores[1].native_effect > 0.01);
        assert_eq!(scores[1].unheld, 1);
    }
}
