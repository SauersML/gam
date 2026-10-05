//! Full finite-search capability calibration; no native-mechanism recovery claim.
use super::*;
use crate::{
    composed_rule_search::Unary,
    operator_program::{
        Declarations, FamilyInputs, Interface, Law, Node, Operator, OperatorProgram, Slot,
        SlotValues, exact_precision,
    },
    resident_causal_fit::{self, Episode, NativeResponseTarget, NativeResponses},
};
use gam_gpu::tensor::Device;
use ndarray::{Array2, array};
use std::{collections::BTreeMap, sync::Arc};

fn dense(values: Array2<f64>) -> Arc<Operator> {
    Arc::new(
        Operator::dense(
            "calibration teacher",
            Interface::native(values.nrows())
                .expect("valid calibration construction or declared search operation"),
            Interface::native(values.ncols())
                .expect("valid calibration construction or declared search operation"),
            values.clone(),
            exact_precision(values.iter().copied())
                .expect("valid calibration construction or declared search operation"),
            Default::default(),
        )
        .expect("valid calibration construction or declared search operation"),
    )
}
fn teacher(width: usize, independent: bool) -> (OperatorProgram, Vec<usize>) {
    let inverse_basis = if width == 1 {
        array![[-1.]]
    } else {
        array![[0.6, 0.8], [-0.8, 0.6]]
    };
    let matrices = if width == 1 {
        vec![array![[1.1]], array![[-0.7]], array![[0.45]]]
    } else {
        vec![
            array![[1.2, -0.7], [0.3, 0.9]],
            array![[-0.5, 0.8], [1.1, 0.4]],
            array![[0.6, 1.2], [-0.8, 0.5]],
        ]
    };
    let mut p = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw { width }],
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators: vec![dense(inverse_basis)],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
        ],
        output: 0,
    };
    let mut latents = Vec::new();
    for (index, matrix) in matrices.into_iter().enumerate() {
        let operator = p.operators.len();
        p.operators.push(dense(matrix));
        let bias = p.operators.len();
        let mut bias_operator =
            (*dense(Array2::from_elem((width, 1), 0.12 * (index + 1) as f64))).clone();
        bias_operator.cols = Interface::constant();
        p.operators.push(Arc::new(bias_operator));
        latents.push(p.nodes.len());
        p.nodes.push(Node::Affine {
            terms: vec![(1, operator)],
            bias: Some(bias),
        });
    }
    // This assignment belongs only to the hidden teacher, never to enumeration settings.
    let assignment = if independent { [0, 1, 2] } else { [1, 0, 1] };
    let mut outputs = Vec::new();
    for (index, law) in [Law::Silu, Law::Gelu, Law::Gelu].into_iter().enumerate() {
        outputs.push(p.nodes.len());
        p.nodes.push(Node::Pointwise {
            input: latents[assignment[index]],
            laws: vec![law],
        });
    }
    p.output = p.nodes.len();
    p.nodes.push(Node::Concat {
        parts: outputs.clone(),
    });
    p.interfaces()
        .expect("valid calibration construction or declared search operation");
    (p, outputs)
}
fn inputs(width: usize, heldout: bool) -> FamilyInputs {
    let grid = if heldout {
        vec![-1.8, -0.9, 0.1, 0.8, 1.7]
    } else {
        vec![-2., -1.4, -0.6, 0.3, 1.2, 2.]
    };
    let gains = if heldout {
        vec![-0.9, 1.15]
    } else {
        vec![-0.65, 1., 1.3]
    };
    let mut values = Vec::new();
    for gain in gains {
        for &a in &grid {
            if width == 1 {
                values.push(-gain * a);
            } else {
                for &b in &grid {
                    // Native boundary intervention followed by a hidden change of input coordinates.
                    values.extend([gain * (0.6 * a - 0.8 * b), gain * (0.8 * a + 0.6 * b)]);
                }
            }
        }
    }
    let x = Array2::from_shape_vec((values.len() / width, width), values)
        .expect("valid calibration construction or declared search operation");
    FamilyInputs {
        rows: x.nrows(),
        slots: vec![SlotValues::Raw(x)],
        layout: None,
    }
}
fn topology(expressions: &[Expr]) -> Option<(Vec<usize>, Vec<u8>)> {
    let mut owners = BTreeMap::new();
    let mut partition = Vec::new();
    let mut laws = Vec::new();
    for expression in expressions {
        let Expr::Unary(law, input) = expression else {
            return None;
        };
        let law = match law {
            Unary::Silu => 0,
            Unary::Gelu => 1,
            _ => return None,
        };
        let Expr::Affine {
            parameter,
            input,
            bias: true,
            ..
        } = input.as_ref()
        else {
            return None;
        };
        if !matches!(input.as_ref(), Expr::Argument(0)) {
            return None;
        }
        let next = owners.len();
        let owner = *owners.entry(*parameter).or_insert(next);
        partition.push(owner);
        laws.push(law);
    }
    Some((partition, laws))
}
fn targets(
    p: &OperatorProgram,
    output_nodes: &[usize],
    input: &FamilyInputs,
    native: &OperatorProgram,
    native_outputs: &[usize],
    scales: Option<&[f64]>,
) -> (Vec<Episode>, NativeResponses, Vec<f64>) {
    let trace = native
        .execute(input, false)
        .expect("valid calibration construction or declared search operation");
    let scales = scales.map(Vec::from).unwrap_or_else(|| {
        native_outputs
            .iter()
            .map(|&node| {
                let rms = (trace.values[node].iter().map(|v| v * v).sum::<f64>()
                    / input.rows as f64)
                    .sqrt();
                if rms == 0. { 1. } else { rms }
            })
            .collect()
    });
    let episode = Episode {
        label: "native responses".into(),
        group: "all".into(),
        inputs: input.clone(),
        target_logits: trace.values[native.output].clone(),
        scored: None,
    };
    assert_eq!(
        p.node_interface(p.output)
            .expect("valid calibration construction or declared search operation")
            .width(),
        episode.target_logits.ncols()
    );
    let responses = output_nodes
        .iter()
        .zip(native_outputs)
        .enumerate()
        .map(|(index, (&node, &teacher_node))| NativeResponseTarget {
            label: format!("exit{index}"),
            source_node: node,
            values: trace.values[teacher_node].clone(),
            scored: None,
            scale: scales[index],
            weight: 1. / native_outputs.len() as f64,
        })
        .collect();
    (
        vec![episode],
        BTreeMap::from([("native responses".into(), responses)]),
        scales,
    )
}
fn worst_response(measurement: &resident_causal_fit::Measurement) -> f64 {
    measurement
        .episodes
        .iter()
        .flat_map(|e| &e.responses)
        .map(|r| r.mean_normalized_squared_error)
        .fold(0., f64::max)
}

#[test]
fn learned_projection_restricted_family_search_hidden_basis_and_negative_sharing_control() {
    let d = Device::host();
    let tolerance = 2e-5;
    let evidence_path = std::env::temp_dir().join("gam-learned-dag-calibration.json");
    let mut evidence = Vec::new();
    for (width, independent) in [(1, false), (2, false), (2, true)] {
        let interface = Interface::native(width)
            .expect("valid calibration construction or declared search operation");
        let exits = vec![interface.clone(); 3];
        let settings = Settings {
            latent_widths: vec![width],
            unary: vec![Unary::Silu, Unary::Gelu],
            binary: vec![],
            affine_bias: true,
            require_shared: false,
            max_operations: 2,
            max_affine_parameters: 3,
            max_parameter_elements: 256,
            max_expression_states: 4096,
            max_tuple_checks: 1_000_000,
            max_tuples: 100_000,
            max_body_nodes: 8,
            seed: 17,
        };
        let inventory = enumerate_interfaces(&[interface.clone()], &exits, &settings)
            .expect("valid calibration construction or declared search operation");
        let mut classes = BTreeMap::new();
        for proposal in &inventory.proposals {
            if let Some(key) = topology(&proposal.expressions) {
                classes
                    .entry(key)
                    .or_insert_with(|| proposal.expressions.clone());
            }
        }
        eprintln!(
            "inventory states={} tuples={} proposals={} truncated={} classes={}",
            inventory.explored_states,
            inventory.checked_tuples,
            inventory.proposals.len(),
            inventory.truncated,
            classes.len()
        );
        // All five partitions of three named outputs crossed with both laws on every output.
        assert_eq!(
            classes.len(),
            40,
            "enumeration did not cover the declared finite hypothesis family"
        );
        let (native, native_outputs) = teacher(width, independent);
        let training = inputs(width, false);
        let heldout = inputs(width, true);
        let mut winners = Vec::new();
        let mut best_shared = f64::INFINITY;
        let mut candidates: Vec<_> = classes.into_iter().collect();
        candidates.sort_by_key(|((partition, laws), _)| {
            (
                partition
                    .iter()
                    .max()
                    .expect("valid calibration construction or declared search operation")
                    + 1,
                partition.clone(),
                laws.clone(),
            )
        });
        let inventory_proposals = inventory.proposals.len();
        let mut attempts = Vec::new();
        for ((partition, laws), expressions) in candidates {
            let compiled = compile_program(&[interface.clone()], &exits, &expressions, &settings)
                .expect("valid calibration construction or declared search operation");
            let (episodes, responses, scales) = targets(
                &compiled.program,
                &compiled.output_nodes,
                &training,
                &native,
                &native_outputs,
                None,
            );
            let mut best_fit: Option<resident_causal_fit::Fit> = None;
            let mut starts = Vec::new();
            for seed in [17, 41, 113, 607] {
                let start_settings = Settings {
                    seed,
                    ..settings.clone()
                };
                let start =
                    compile_program(&[interface.clone()], &exits, &expressions, &start_settings)
                        .expect("each fixed multistart seed compiles the same candidate topology");
                assert_eq!(start.output_nodes, compiled.output_nodes);
                let fitted = resident_causal_fit::fit_with_native(
                    &d,
                    &start.program,
                    &episodes,
                    &responses,
                    &start.trainable_operator_ids,
                    resident_causal_fit::Settings {
                        iterations: 1200,
                        learning_rate: 0.03,
                        beta1: 0.9,
                        beta2: 0.999,
                        epsilon: 1e-8,
                        numeric_bytes: 1 << 25,
                        schedule: None,
                        arithmetic: resident_causal_fit::FitArithmetic::F64,
                        exact_scan_every: 1,
                    },
                )
                .expect("declared TRAIN-only multistart fits ordinary candidate parameters");
                starts.push(
                    serde_json::json!({"seed":seed,"objective":fitted.report.best.objective,
                    "worst_response":worst_response(&fitted.report.best)}),
                );
                if best_fit
                    .as_ref()
                    .is_none_or(|best| fitted.report.best.objective < best.report.best.objective)
                {
                    best_fit = Some(fitted);
                }
            }
            let fit = best_fit.expect("the fixed nonempty multistart inventory yields a TRAIN fit");
            let error = worst_response(&fit.report.best);
            let owners = partition
                .iter()
                .max()
                .expect("valid calibration construction or declared search operation")
                + 1;
            attempts.push(serde_json::json!({"partition":partition,"laws":laws,"expressions":expressions,
                "affine_owners":owners,"worst_normalized_response_squared_error":error,
                "mean_kl":fit.report.best.episodes[0].mean_kl,"iterations":fit.report.settings.iterations,"starts":starts}));
            if owners < 3 {
                best_shared = best_shared.min(error);
            }
            eprintln!(
                "learned-DAG calibration width={width} independent={independent} partition={partition:?} laws={laws:?} response_error={error}"
            );
            if error < tolerance && fit.report.best.episodes[0].mean_kl < 1e-4 {
                winners.push((
                    owners,
                    error,
                    partition,
                    laws,
                    fit.program,
                    compiled.output_nodes,
                    scales,
                ));
            }
        }
        std::fs::write(&evidence_path, serde_json::to_vec_pretty(&serde_json::json!({
            "completed_fixtures":evidence,"current_fixture":{"width":width,"independent":independent,
            "settings":settings,"attempts":attempts,"admitted_candidates":winners.len(),
            "stage":"TRAIN fits complete; selection and heldout measurement pending"}
        })).expect("training attempt evidence serializes without altering selection"))
            .expect("archive all successful and failed TRAIN attempts outside repository");
        winners.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        let (owners, _, partition, laws, saved, outputs, scales) = winners
            .into_iter()
            .next()
            .expect("full search admitted no native-response law");
        assert_eq!(
            laws,
            vec![0, 1, 1],
            "selected law differs from the independently generated native fixture"
        );
        if independent {
            assert_eq!(owners, 3);
            assert_eq!(partition, vec![0, 1, 2]);
            assert!(best_shared >= tolerance);
        } else {
            assert_eq!(owners, 2);
            assert_eq!(partition, vec![0, 1, 0]);
        }
        let artifact = crate::artifact::Artifact::native(&saved)
            .expect("selected ordinary program has native artifact bindings")
            .f32_literals()
            .expect("selected coefficients have an ordinary f32 literal encoding");
        let bytes = artifact
            .to_bytes()
            .expect("selected artifact encodes ordinarily");
        let decoded = crate::artifact::Artifact::from_bytes(&bytes, &saved.declarations)
            .expect("selected artifact decodes ordinarily");
        assert_eq!(decoded.to_bytes().expect("decoded canonical replay"), bytes);
        assert_eq!(
            decoded.program.nodes, saved.nodes,
            "observation IDs survive ordinary replay"
        );
        let artifact_path =
            std::env::temp_dir().join(format!("gam-learned-dag-{width}-{independent}.artifact"));
        std::fs::write(&artifact_path, &bytes)
            .expect("archive selected ordinary artifact outside repository");
        let saved = decoded.program;
        // Selection above uses TRAIN only. No heldout targets or scales enter fitting/ranking.
        let (episodes, responses, _) = targets(
            &saved,
            &outputs,
            &heldout,
            &native,
            &native_outputs,
            Some(&scales),
        );
        let measured =
            resident_causal_fit::measure_with_native(&d, &saved, &episodes, &responses, 1 << 25)
                .expect("valid calibration construction or declared search operation");
        assert!(
            worst_response(&measured) < tolerance * 3.,
            "selected law failed input/control recombinations"
        );
        assert!(measured.episodes[0].mean_kl < 3e-4);
        evidence.push(serde_json::json!({"width":width,"independent":independent,
            "inventory_proposals":inventory_proposals,"inventory_states":inventory.explored_states,
            "inventory_tuple_checks":inventory.checked_tuples,
            "inventory_skeleton_tuples":inventory.checked_skeleton_tuples,
            "inventory_completed_parameter_bindings":inventory.completed_parameter_bindings,
            "inventory_truncated":inventory.truncated,
            "settings":settings,"fitted_classes":attempts.len(),"attempts":attempts,
            "ordinary_artifact_path":artifact_path,"ordinary_artifact_bytes":bytes.len(),
            "selected_partition":partition,"selected_laws":laws,"heldout":measured,
            "scope":"All candidates originate in enumerate_interfaces. Full search over the disclosed 40 unary-after-named-affine classes (five three-exit sharing partitions times eight SiLU/GELU law assignments); other generated shapes excluded. Hidden input basis and native operators never initialize candidates. Selection uses TRAIN only; heldout response scales frozen from TRAIN. Capability calibration, not general algorithm recovery."}));
        std::fs::write(
            &evidence_path,
            serde_json::to_vec_pretty(&evidence)
                .expect("valid calibration construction or declared search operation"),
        )
        .expect("valid calibration construction or declared search operation");
        // Every learned width-two representation keeps both coordinates.
        if width == 2 {
            let projections: Vec<_> = saved
                .operators
                .iter()
                .filter_map(|op| match &op.body {
                    crate::operator_program::OperatorBody::Dense { values, .. }
                        if values.dim() == (2, 2) =>
                    {
                        Some(values)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(
                projections.len(),
                owners,
                "one full-width matrix per named latent owner"
            );
            assert!(
                projections
                    .iter()
                    .all(|m| (m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]]).abs() > 0.05),
                "every selected vector projection is numerically full rank"
            );
        }
    }
}
