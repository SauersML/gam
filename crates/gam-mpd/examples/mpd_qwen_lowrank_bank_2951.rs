//! Whole-model Qwen SwiGLU parameter baseline; every native node/place remains.
//! EXPORT OUT layer=N ranks=N,... max_bank=N context=N deltas=X,... epsilons=X,...
//! max_message_bytes=N host_estimate_limit=N. Layer/ranks are declared scope knobs.
use gam_linalg::decompose::svd;
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    coder_capture::sha256,
    import::import_language_model,
    operator_program::{
        Node, Operator, OperatorBody, OperatorProgram, Provenance, exact_precision,
    },
};
use ndarray::{Array1, Axis};
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};
#[derive(Debug, Serialize)]
struct MlpMap {
    up_operator: usize,
    gate_operator: usize,
    down_operator: usize,
    normed: usize,
    skip: usize,
    up_write: usize,
    gate_write: usize,
    active: usize,
    post_mlp_residual: usize,
    final_norm: usize,
}
fn operator(p: &OperatorProgram, name: &str) -> Result<usize, String> {
    let found: Vec<_> = p
        .operators
        .iter()
        .enumerate()
        .filter(|(_, o)| o.name == name)
        .map(|(i, _)| i)
        .collect();
    if found.len() != 1 {
        return Err(format!("expected one operator {name}, got{}", found.len()));
    }
    Ok(found[0])
}
fn affine(p: &OperatorProgram, op: usize) -> Result<usize, String> {
    let found: Vec<_> = p
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n,Node::Affine{terms,..}if terms.iter().any(|(_,o)|*o==op)))
        .map(|(i, _)| i)
        .collect();
    if found.len() != 1 {
        return Err(format!(
            "operator{} needs exactly one native affine use",
            p.operators[op].name
        ));
    }
    Ok(found[0])
}
fn map(p: &OperatorProgram, layer: usize) -> Result<MlpMap, String> {
    let up_operator = operator(p, &format!("blocks.{layer}.c_fc"))?;
    let gate_operator = operator(p, &format!("blocks.{layer}.gate_proj"))?;
    let down_operator = operator(p, &format!("blocks.{layer}.down_proj"))?;
    let (up_write, gate_write, post_mlp_residual) = (
        affine(p, up_operator)?,
        affine(p, gate_operator)?,
        affine(p, down_operator)?,
    );
    let one = |node| match &p.nodes[node] {
        Node::Affine { terms, bias: None } if terms.len() == 1 => Ok(terms[0].0),
        _ => Err("up/gate needs one unbiased affine input".to_string()),
    };
    let normed = one(up_write)?;
    if one(gate_write)? != normed {
        return Err("gate/up do not read the same native normalized input".into());
    }
    let (skip, active) = match &p.nodes[post_mlp_residual] {
        Node::Affine { terms, bias: None } if terms.len() == 2 => {
            let mut down = None;
            let mut skip = None;
            for &(arg, op) in terms {
                if op == down_operator {
                    down = Some(arg)
                } else if matches!(p.operators[op].body, OperatorBody::Identity) {
                    skip = Some(arg)
                }
            }
            (
                skip.ok_or("missing native identity residual skip")?,
                down.ok_or("missing native down input")?,
            )
        }
        _ => return Err("expected native two-term unbiased post-MLP residual".into()),
    };
    match &p.nodes[active] {
        Node::Hadamard { left, right } if *right == up_write => match &p.nodes[*left] {
            Node::Pointwise { input, laws }
                if *input == gate_write
                    && laws
                        .iter()
                        .all(|l| *l == gam_mpd::operator_program::Law::Silu) => {}
            _ => return Err("native gate is not elementwise SiLU".into()),
        },
        _ => return Err("native activation is not SiLU(gate)*up".into()),
    }
    let final_norm = affine(p, operator(p, "final_norm.gain")?)?;
    Ok(MlpMap {
        up_operator,
        gate_operator,
        down_operator,
        normed,
        skip,
        up_write,
        gate_write,
        active,
        post_mlp_residual,
        final_norm,
    })
}
fn changed(base: &Artifact, m: &MlpMap, factors: &[Arc<Operator>]) -> Result<Artifact, String> {
    if factors.len() != 3 {
        return Err("three parameter factors required".into());
    }
    let mut a = base.clone();
    for (index, op) in [m.up_operator, m.gate_operator, m.down_operator]
        .into_iter()
        .zip(factors)
    {
        a.program.operators[index] = op.clone();
    }
    a = a
        .bind(
            "Qwen gated MLP post-residual boundary",
            &[m.skip, m.normed],
            m.post_mlp_residual,
        )?
        .f32_literals()?;
    if a.program.nodes != base.program.nodes
        || a.places != base.places
        || a.program.output != base.program.output
    {
        return Err("parameter baseline changed native nodes/places/output".into());
    }
    a.validate_coverage(&base.program)?;
    Ok(a)
}
fn floats(s: &str) -> Result<Vec<f64>, String> {
    let v: Vec<f64> = s
        .split(',')
        .map(|x| x.parse().map_err(|e| format!("{e}")))
        .collect::<Result<_, _>>()?;
    if v.is_empty() || v.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("nonempty finite nonnegative tolerance list required".into());
    }
    Ok(v)
}
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 10 {
        return Err("EXPORT OUT layer=N ranks=N,... max_bank=N context=N deltas=X,... epsilons=X,... max_message_bytes=N host_estimate_limit=N".into());
    }
    let mut keys = BTreeMap::new();
    for x in &a[2..] {
        let (k, v) = x.split_once('=').ok_or("expected key=value")?;
        if ![
            "layer",
            "ranks",
            "max_bank",
            "context",
            "deltas",
            "epsilons",
            "max_message_bytes",
            "host_estimate_limit",
        ]
        .contains(&k)
            || keys.insert(k, v).is_some()
        {
            return Err(format!("unknown/duplicate option{k}"));
        }
    }
    let get = |k| keys.get(k).copied().ok_or_else(|| format!("declare{k}"));
    let number = |k| -> Result<usize, String> { get(k)?.parse().map_err(|e| format!("{e}")) };
    let (layer, max_bank, context, message_limit, host_limit) = (
        number("layer")?,
        number("max_bank")?,
        number("context")?,
        number("max_message_bytes")?,
        number("host_estimate_limit")?,
    );
    let ranks: Vec<usize> = get("ranks")?
        .split(',')
        .map(|x| x.parse().map_err(|e| format!("{e}")))
        .collect::<Result<_, _>>()?;
    if ranks.is_empty()
        || ranks.contains(&0)
        || ranks
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != ranks.len()
        || ranks.len() + 1 > max_bank
        || context == 0
        || message_limit == 0
        || host_limit == 0
    {
        return Err(
            "invalid declared rank/bank/context/resource bounds; complete bank cannot be truncated"
                .into(),
        );
    }
    let deltas = floats(get("deltas")?)?;
    let epsilons = floats(get("epsilons")?)?;
    let (export, out) = (Path::new(&a[0]), Path::new(&a[1]));
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    let start = Instant::now();
    let imported = import_language_model(export, 1, context)?;
    let record = imported.record;
    let p = imported.program;
    if record["config"]["mlp_gated"].as_bool() != Some(true)
        || record["config"]["mlp_act"] != "silu"
        || record["config"]["norm"] != "rms"
        || record["config"]["qk_norm"].as_bool() != Some(true)
        || record["source"]["layers_kept"] != record["config"]["n_layers"]
    {
        return Err("whole Qwen3 gated-SiLU RMS/qk-norm export required".into());
    }
    let m = map(&p, layer)?;
    let base = Artifact::native(&p)?.f32_literals()?;
    let source_bytes: u64 = p.operators.iter().map(|o| o.real_count() as u64 * 8).sum();
    let estimate = source_bytes
        .checked_add(
            (message_limit as u64)
                .checked_mul(2)
                .ok_or("byte estimate overflow")?,
        )
        .and_then(|x| x.checked_add(1 << 30))
        .ok_or("byte estimate overflow")?;
    if estimate > host_limit as u64 {
        return Err(format!(
            "known generation buffer estimate{estimate} exceeds declared{host_limit}; estimate excludes allocator/runtime overhead"
        ));
    }
    let mut variants: BTreeMap<usize, Vec<Arc<Operator>>> =
        ranks.iter().map(|&r| (r, Vec::new())).collect();
    let mut spectral = Vec::new();
    for index in [m.up_operator, m.gate_operator, m.down_operator] {
        let original = &p.operators[index];
        let matrix = original.matrix_cow();
        let decomposition = svd(matrix.view(), false).map_err(|e| e.to_string())?;
        for &rank in &ranks {
            if rank > decomposition.singular_values.len() {
                return Err(format!("rank{rank} exceeds{}", original.name));
            }
            let kept: Vec<_> = (0..rank).collect();
            let roots = Array1::from_iter(
                decomposition
                    .singular_values
                    .iter()
                    .take(rank)
                    .map(|v| v.sqrt()),
            );
            let left = (decomposition.u.select(Axis(1), &kept) * &roots)
                .as_standard_layout()
                .into_owned();
            let right = (decomposition.vt.select(Axis(0), &kept) * &roots.insert_axis(Axis(1)))
                .as_standard_layout()
                .into_owned();
            let precision = exact_precision(left.iter().chain(right.iter()).copied())
                .map_err(|e| e.to_string())?;
            let op=Operator::low_rank(original.name.clone(),original.rows.clone(),original.cols.clone(),left,right,precision,Provenance::derived(&[&original.provenance],format!("declared layer{layer} balanced truncated SVD rank{rank}; parameter baseline"))).map_err(|e|e.to_string())?;
            variants
                .get_mut(&rank)
                .ok_or("rank absent")?
                .push(Arc::new(op));
        }
        spectral.push(json!({"operator":original.name,"shape":matrix.dim(),"singular_values":decomposition.singular_values.to_vec(),"numerical_band":decomposition.band}));
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut costs = CostCache::default();
    let native_cost = structural_cost(&base, &mut costs)?;
    let mut candidates = Vec::new();
    let mut records = Vec::new();
    for &rank in &ranks {
        let artifact = changed(&base, &m, &variants[&rank])?;
        let cost = structural_cost(&artifact, &mut costs)?;
        let bytes = artifact.to_bytes()?;
        if bytes.len() > message_limit {
            return Err(format!(
                "candidate message{} exceeds declared{message_limit}",
                bytes.len()
            ));
        }
        let file = format!("qwen-L{layer}-joint-r{rank}.mpd");
        std::fs::write(out.join(&file), &bytes).map_err(|e| e.to_string())?;
        let size = bytes.len();
        drop(bytes);
        let label = format!("qwen-declared-L{layer}-joint-SVD-r{rank}");
        candidates.push(json!({"label":label,"path":file}));
        records.push(json!({"label":label,"rank":rank,"file":file,"file_bytes":size,"sha256":sha256(&out.join(&file))?,"C32":cost}));
    }
    let width = p
        .node_interface(m.up_write)
        .map_err(|e| e.to_string())?
        .width();
    let norm_width = p
        .node_interface(m.final_norm)
        .map_err(|e| e.to_string())?
        .width();
    let edit = |node, width, columns: [usize; 2], scale: f64| json!({"node":node,"width":width,"rows":null,"columns":columns,"scale":scale,"add":null});
    let gate = edit(m.gate_write, width, [0, width], 0.0);
    let up = edit(m.up_write, width, [0, width], 0.5);
    let final_norm = edit(m.final_norm, norm_width, [0, 1], 0.0);
    let episodes = vec![
        json!({"id":"clean","group":"clean","edits":[]}),
        json!({"id":"last-layer-gate-removal","group":"gate-removal","edits":[gate]}),
        json!({"id":"last-layer-up-half","group":"up-scale","edits":[up]}),
        json!({"id":"final-norm-column0-control","group":"final-norm-control","edits":[final_norm]}),
        json!({"id":"gate-removal-plus-up-half","group":"gate-up-combination","edits":[gate,up]}),
    ];
    let constraints: Vec<Value> = deltas
        .iter()
        .flat_map(|&local| {
            epsilons
                .iter()
                .map(move |&run| json!({"local":local,"run":run}))
        })
        .collect();
    let spec = json!({"checkpoint_sha256":record["source"]["weights_sha256"],"sequences":1,"context":context,"nodes":p.nodes.len(),"batch_rows":context,"budget":ranks.len()+1,"max_bank":max_bank,"constraints":constraints,"episodes":episodes,"candidates":candidates,"cuda":true,"intermediate_bytes_limit":1073741824_u64,"compare_cpu_metrics":false});
    std::fs::write(
        out.join("spec.json"),
        serde_json::to_vec_pretty(&spec).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let report = json!({"method":"truncated-SVD parameter compression baseline, not mechanism discovery","source":record["source"],"config":record["config"],"export_json_sha256":sha256(&export.join("export.json"))?,"declared_layer_scope":layer,"ranks":ranks,"bank_including_native":ranks.len()+1,"scope":"each candidate jointly factors up, gate and down at one declared layer; complete independent bank; no fidelity pruning","node_map":m,"every_native_node_place_and_readout_preserved":true,"local_boundary":{"reads":[m.skip,m.normed],"write":m.post_mlp_residual,"denominator":"RMS native post-MLP-residual row L2 norm on same declared family; not contribution-only scale and delta not comparable to 4L contribution-normalized Local","absolute_error":"BlockError.worst * BlockError.scale; row L2 write discrepancy"},"generation_memory_estimate_incomplete":estimate,"host_estimate_limit":host_limit,"max_message_bytes":message_limit,"native_C32":native_cost,"candidates":records,"spectral":spectral,"seconds":start.elapsed().as_secs_f64(),"acceptance":"none during generation; CPU native-parent Local and autonomous required-CUDA counterfactual Run evaluate serialized decoded artifacts"});
    std::fs::write(
        out.join("GENERATION.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", out.join("spec.json").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_mpd::operator_program::{Declarations, Interface, Law, Slot};
    use ndarray::array;

    fn gated() -> OperatorProgram {
        let face = Interface::native(2).unwrap();
        let dense = |name: &str| {
            Arc::new(
                Operator::dense(
                    name,
                    face.clone(),
                    face.clone(),
                    array![[2., 0.], [0., 1.]],
                    exact_precision([2., 0., 1.]).unwrap(),
                    Provenance::native(name),
                )
                .unwrap(),
            )
        };
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense("blocks.27.c_fc"),
                dense("blocks.27.gate_proj"),
                dense("blocks.27.down_proj"),
                Arc::new(Operator::identity("skip", face.clone())),
                dense("final_norm.gain"),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 3)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(1, 0)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(1, 1)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 3,
                    laws: vec![Law::Silu],
                },
                Node::Hadamard { left: 4, right: 2 },
                Node::Affine {
                    terms: vec![(0, 3), (5, 2)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(6, 4)],
                    bias: None,
                },
            ],
            output: 7,
        }
    }

    #[test]
    fn lowrank_changes_preserve_all_native_places_and_joint_boundary() {
        let p = gated();
        let m = map(&p, 27).unwrap();
        assert_eq!((m.skip, m.normed, m.post_mlp_residual), (0, 1, 6));
        let base = Artifact::native(&p).unwrap();
        let factors: Vec<_> = [m.up_operator, m.gate_operator, m.down_operator]
            .into_iter()
            .map(|i| {
                let old = &p.operators[i];
                Arc::new(
                    Operator::low_rank(
                        old.name.clone(),
                        old.rows.clone(),
                        old.cols.clone(),
                        array![[1.], [0.]],
                        array![[1., 0.]],
                        exact_precision([1., 0.]).unwrap(),
                        Provenance::derived(&[&old.provenance], "test declared rank1".into()),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let candidate = changed(&base, &m, &factors).unwrap();
        assert_eq!(candidate.program.nodes, p.nodes);
        assert_eq!(candidate.places, base.places);
        assert_eq!(candidate.program.output, p.output);
        candidate.validate_coverage(&p).unwrap();
        for i in [m.up_operator, m.gate_operator, m.down_operator] {
            assert!(matches!(
                candidate.program.operators[i].body,
                OperatorBody::LowRank { .. }
            ));
        }
        let decoded =
            Artifact::from_bytes(&candidate.to_bytes().unwrap(), &p.declarations).unwrap();
        decoded.validate_coverage(&p).unwrap();
        assert_eq!(decoded.program.nodes, p.nodes);
    }

    #[test]
    fn gate_removal_is_held_and_up_scaling_remains_executable() {
        use gam_mpd::operator_program::{FamilyInputs, SlotValues};
        let p = gated();
        let m = map(&p, 27).unwrap();
        let base = Artifact::native(&p).unwrap();
        let factors: Vec<_> = [m.up_operator, m.gate_operator, m.down_operator]
            .into_iter()
            .map(|i| {
                let old = &p.operators[i];
                Arc::new(
                    Operator::low_rank(
                        old.name.clone(),
                        old.rows.clone(),
                        old.cols.clone(),
                        array![[1.], [0.]],
                        array![[1., 0.]],
                        exact_precision([1., 0.]).unwrap(),
                        Provenance::derived(&[&old.provenance], "declared test factor".into()),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let candidate = changed(&base, &m, &factors).unwrap();
        let candidate =
            Artifact::from_bytes(&candidate.to_bytes().unwrap(), &p.declarations).unwrap();
        let inputs = FamilyInputs {
            rows: 1,
            slots: vec![SlotValues::Raw(array![[1., 2.]])],
            layout: None,
        };
        let clean = p.execute(&inputs, false).unwrap();
        let altered = candidate.program.execute(&inputs, false).unwrap();
        assert_ne!(clean.values[p.output], altered.values[p.output]);
        for up_half in [false, true] {
            let run = |program: &OperatorProgram| {
                program
                    .execute_edited(&inputs, |node, value, _| {
                        if node == m.gate_write {
                            value.fill(0.);
                        }
                        if up_half && node == m.up_write {
                            *value *= 0.5;
                        }
                        Ok(())
                    })
                    .unwrap()
            };
            assert_eq!(
                run(&p).values[p.output],
                run(&candidate.program).values[p.output]
            );
        }
        let half = p
            .execute_edited(&inputs, |node, value, _| {
                if node == m.up_write {
                    *value *= 0.5;
                }
                Ok(())
            })
            .unwrap();
        assert_ne!(half.values[p.output], clean.values[p.output]);
    }

    #[test]
    fn ungated_or_wrong_gate_input_is_rejected() {
        let mut p = gated();
        p.nodes[4] = Node::Pointwise {
            input: 3,
            laws: vec![Law::Identity],
        };
        assert!(map(&p, 27).is_err());
        p.nodes[4] = Node::Pointwise {
            input: 1,
            laws: vec![Law::Silu],
        };
        assert!(map(&p, 27).is_err());
    }
}
