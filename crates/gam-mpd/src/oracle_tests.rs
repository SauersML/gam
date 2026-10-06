#![cfg(test)]
//! Tests of the oracle's measured interventions (#2951) on tiny decoders.

use crate::operator_program::{FamilyInputs, OperatorBody, SequenceLayout, SlotValues, exact_precision};
use crate::oracle::{Component, Edit, Intervention, Native, OptionItem, Pair, Patch, PatchValue, Request, Session, Site};
use crate::test_support::{tiny_export, tiny_qwen3_export};
use ndarray::Array2;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

fn sequences() -> Vec<Vec<u32>> {
    vec![vec![1, 4, 2, 7, 4, 2, 9], vec![3, 3, 8, 1, 0], vec![5, 6, 10, 6]]
}

/// The full program's logits (its own readout, every node evaluated) at every row.
fn full_logits(native: &Native, sequences: &[Vec<u32>]) -> Array2<f64> {
    let (mut ids, mut sequence, mut position) = (Vec::new(), Vec::new(), Vec::new());
    for (s, tokens) in sequences.iter().enumerate() {
        for (p, t) in tokens.iter().enumerate() {
            ids.push(*t);
            sequence.push(s as u32);
            position.push(p as u32);
        }
    }
    let inputs = FamilyInputs { rows: ids.len(), slots: vec![SlotValues::Tokens(ids)], layout: Some(SequenceLayout { sequence, position }) };
    let trace = native.program.execute(&inputs, false).expect("full run");
    trace.values[native.program.output].clone()
}

/// Every position's top-`vocab` log-probabilities from a run request, as (sequence, position, token) -> value.
fn run(session: &mut Session, model: &str, intervention: Value) -> BTreeMap<(usize, usize, usize), f64> {
    let positions: Vec<i64> = (0..4).collect();
    let request: Request = serde_json::from_value(serde_json::json!({
        "op": "run", "model": model, "sequences": sequences(), "positions": positions, "top": 11, "intervention": intervention,
    }))
    .expect("request");
    let reply = session.handle(&request).expect("run");
    let mut out = BTreeMap::new();
    for (s, r) in reply["runs"].as_array().expect("runs").iter().enumerate() {
        for e in r["positions"].as_array().expect("positions") {
            let p = e["position"].as_u64().expect("position") as usize;
            for pair in e["top"].as_array().expect("top") {
                out.insert((s, p, pair[0].as_u64().expect("token") as usize), pair[1].as_f64().expect("log-probability"));
            }
        }
    }
    out
}

fn largest_difference(a: &BTreeMap<(usize, usize, usize), f64>, b: &BTreeMap<(usize, usize, usize), f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().map(|(k, v)| (v - b[k]).abs()).fold(0.0, f64::max)
}

fn session(dir: &std::path::Path, work_bytes: usize) -> Session {
    let mut models = BTreeMap::new();
    models.insert("m".to_string(), Native::load(dir).expect("load"));
    Session::new(models, work_bytes)
}

/// A run that stops at the final hidden state and forms the logits only at the read rows gives the
/// full program's logits, in one group or in groups of one sequence.
#[test]
fn stopping_at_the_hidden_state_keeps_the_logits() {
    for dir in [tiny_export("oracle_hidden", 2), tiny_qwen3_export("oracle_hidden_q", 2)] {
        let native = Native::load(&dir).expect("load");
        let full = full_logits(&native, &sequences());
        let row_bytes = native.row_bytes;
        let mut whole = session(&dir, 1 << 30);
        let mut grouped = session(&dir, row_bytes);
        for s in [&mut whole, &mut grouped] {
            let reply = s
                .handle(&serde_json::from_value(serde_json::json!({"op": "run", "model": "m", "sequences": sequences(), "positions": [0, 1, 2, 3], "top": 11})).expect("request"))
                .expect("run");
            let mut offset = 0;
            for (k, r) in reply["runs"].as_array().expect("runs").iter().enumerate() {
                for e in r["positions"].as_array().expect("positions") {
                    let p = e["position"].as_u64().expect("position") as usize;
                    let row = full.row(offset + p).to_vec();
                    let lp = gam_math::categorical::log_softmax(&row).expect("log softmax");
                    for pair in e["top"].as_array().expect("top") {
                        let t = pair[0].as_u64().expect("token") as usize;
                        assert!((pair[1].as_f64().expect("value") - lp[t]).abs() < 1e-12, "sequence {k} position {p} token {t}");
                    }
                }
                offset += sequences()[k].len();
            }
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}

/// alpha = 1 is the model exactly; a head's output map at alpha = 0 is its attention read set to
/// zero; a patch from the model's own run on the same sequence changes nothing.
#[test]
fn native_edits_and_patches_are_exact_where_they_must_be() {
    for dir in [tiny_export("oracle_exact", 2), tiny_qwen3_export("oracle_exact_q", 2)] {
        let mut s = session(&dir, 1 << 30);
        let clean = run(&mut s, "m", serde_json::json!({}));
        let unit = run(&mut s, "m", serde_json::json!({"edits": [{"component": {"kind": "operator", "name": "blocks.1.c_fc"}, "alpha": 1.0}]}));
        assert_eq!(largest_difference(&clean, &unit), 0.0);
        let edited = run(&mut s, "m", serde_json::json!({"edits": [{"component": {"kind": "head", "layer": 0, "head": 1}, "alpha": 0.0}]}));
        let patched = run(&mut s, "m", serde_json::json!({"patches": [{"site": {"kind": "head", "layer": 0, "head": 1}, "value": {"kind": "zero"}}]}));
        assert!(largest_difference(&edited, &patched) < 1e-12, "{}", largest_difference(&edited, &patched));
        assert!(largest_difference(&clean, &edited) > 1e-6);
        let same = run(&mut s, "m", serde_json::json!({"patches": [{"site": {"kind": "mlp", "layer": 0}, "value": {"kind": "source"}}]}));
        assert_eq!(largest_difference(&clean, &same), 0.0);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}

/// Setting a difference to alpha = 0 makes the operator the reference's: the model then runs as
/// the reference where only that operator differs; activation swaps of every site at once do too.
#[test]
fn a_difference_at_alpha_zero_is_the_reference() {
    let dir = tiny_export("oracle_difference", 2);
    let base = Native::load(&dir).expect("load");
    let mut program = base.program.clone();
    let op = base.operator("blocks.1.down_proj").expect("operator");
    let mut operator = (*program.operators[op]).clone();
    let OperatorBody::Dense { values, present, .. } = &operator.body else { panic!("dense") };
    let values = values.mapv(|v| v * 1.5 + 0.01);
    let precision = exact_precision(values.iter().copied()).expect("precision");
    operator.body = OperatorBody::Dense { values: values.into(), present: present.clone(), precision };
    program.operators[op] = Arc::new(operator);
    let updated = Native::new(program, 2).expect("updated");
    let mut models = BTreeMap::new();
    models.insert("base".to_string(), base);
    models.insert("updated".to_string(), updated);
    let mut s = Session::new(models, 1 << 30);
    let reference = run(&mut s, "base", serde_json::json!({}));
    let model = run(&mut s, "updated", serde_json::json!({}));
    assert!(largest_difference(&reference, &model) > 1e-6);
    let reverted = run(&mut s, "updated", serde_json::json!({"edits": [{"component": {"kind": "difference", "name": "blocks.1.down_proj", "reference": "base"}, "alpha": 0.0}]}));
    assert!(largest_difference(&reference, &reverted) < 1e-12);
    let svd_all = run(&mut s, "updated", serde_json::json!({"edits": [{"component": {"kind": "difference", "name": "blocks.1.down_proj", "reference": "base", "components": [0, 1, 2, 3, 4, 5, 6, 7]}, "alpha": 0.0}]}));
    assert!(largest_difference(&reference, &svd_all) < 1e-10, "{}", largest_difference(&reference, &svd_all));
    let swapped = run(&mut s, "updated", serde_json::json!({"patches": [{"site": {"kind": "mlp", "layer": 1}, "value": {"kind": "source", "model": "base"}}]}));
    assert!(largest_difference(&reference, &swapped) < 1e-12);
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// Crossed interventions: gamma is zero when the two interventions are equal, and equals the
/// difference of the two input effects by construction.
#[test]
fn crossed_gamma_is_the_change_of_the_input_effect() {
    let dir = tiny_export("oracle_crossed", 2);
    let mut s = session(&dir, 1 << 30);
    let pairs = vec![Pair { x0: vec![1, 4, 2, 7], x1: vec![1, 5, 2, 7], target: 3, versus: Some(6), positions: None }];
    let ablate = Intervention { edits: vec![Edit { component: Component::Head { layer: 1, head: 0 }, alpha: 0.0 }], patches: vec![] };
    let same = s.handle(&Request::Crossed { model: "m".into(), pairs: pairs.clone(), a0: ablate.clone(), a1: ablate.clone() }).expect("crossed");
    assert_eq!(same["gamma"]["mean"].as_f64(), Some(0.0));
    let reply = s.handle(&Request::Crossed { model: "m".into(), pairs, a0: Intervention::default(), a1: ablate }).expect("crossed");
    let p = &reply["pairs"][0];
    let f = |k: &str| p[k].as_f64().expect(k);
    assert!((f("gamma") - (f("input_effect_a1") - f("input_effect_a0"))).abs() < 1e-12);
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// An option's log-probability is the sum over its tokens of the full run's next-token
/// log-probabilities after the prompt.
#[test]
fn option_log_probabilities_sum_the_continuation() {
    let dir = tiny_qwen3_export("oracle_options", 2);
    let mut s = session(&dir, 1 << 30);
    let items = vec![OptionItem { prompt: vec![2, 7, 1], options: vec![vec![4, 9], vec![3, 3, 10]] }];
    let lp = s.option_log_probabilities("m", &items, &Intervention::default()).expect("options");
    let native = Native::load(&dir).expect("load");
    for (k, option) in items[0].options.iter().enumerate() {
        let mut tokens = items[0].prompt.clone();
        tokens.extend_from_slice(option);
        let full = full_logits(&native, std::slice::from_ref(&tokens));
        let expected: f64 = (items[0].prompt.len() - 1..tokens.len() - 1)
            .map(|p| gam_math::categorical::log_softmax(&full.row(p).to_vec()).expect("log softmax")[tokens[p + 1] as usize])
            .sum();
        assert!((lp[0][k] - expected).abs() < 1e-12, "option {k}: {} against {expected}", lp[0][k]);
    }
    // A patch naming one sequence of a request reaches it alone, also across groups.
    let patch = Patch { site: Site::Mlp { layer: 1 }, sequences: Some(vec![1]), positions: None, coordinates: None, direction: None, value: PatchValue::Zero };
    let grouped_session = &mut session(&dir, native.row_bytes);
    let a = grouped_session.option_log_probabilities("m", &items, &Intervention { edits: vec![], patches: vec![patch.clone()] }).expect("options");
    let b = s.option_log_probabilities("m", &items, &Intervention { edits: vec![], patches: vec![patch] }).expect("options");
    assert!((a[0][0] - lp[0][0]).abs() < 1e-12 && (a[0][1] - lp[0][1]).abs() > 1e-9);
    assert!((a[0][1] - b[0][1]).abs() < 1e-12);
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// `rows` as the request's list of rows.
fn rows_of(m: &Array2<f64>) -> Vec<Vec<f64>> {
    m.outer_iter().map(|r| r.to_vec()).collect()
}

/// `native`'s program with `W + scale · P` on each listed operator: the literal edit.
fn literally_edited(native: &Native, edits: &[(&str, Array2<f64>)], scale: f64) -> Native {
    let mut program = native.program.clone();
    for (name, p) in edits {
        let op = native.operator(name).expect("operator");
        let mut operator = (*program.operators[op]).clone();
        let OperatorBody::Dense { values, present, .. } = &operator.body else { panic!("dense") };
        let values = &*values.matrix() + &(p * scale);
        let precision = exact_precision(values.iter().copied()).expect("precision");
        operator.body = OperatorBody::Dense { values: values.into(), present: present.clone(), precision };
        program.operators[op] = Arc::new(operator);
    }
    Native::new(program, native.layers.len()).expect("edited model")
}

/// A registered component is a literal edit of the stored matrices: at alpha = 1 the model
/// exactly; at other alphas, with uses on two operators of different layers (one shared
/// component), the runs of a model whose operators hold W + (alpha - 1) L R; and a component
/// registered as one neuron's down column runs as the native neuron component.
#[test]
fn registered_components_are_literal_edits() {
    for dir in [tiny_export("oracle_registered", 2), tiny_qwen3_export("oracle_registered_q", 2)] {
        let native = Native::load(&dir).expect("load");
        let factor = |rows: usize, cols: usize, seed: f64| Array2::from_shape_fn((rows, cols), |(r, c)| ((r * 7 + c * 3) as f64 * 0.37 + seed).sin() * 0.2);
        let shape = |name: &str| native.program.operators[native.operator(name).expect("operator")].matrix_cow().dim();
        let ((fc_rows, fc_cols), (o_rows, o_cols)) = (shape("blocks.1.c_fc"), shape("blocks.0.o1"));
        let uses = [("blocks.1.c_fc", factor(fc_rows, 2, 0.1), factor(2, fc_cols, 0.7)), ("blocks.0.o1", factor(o_rows, 1, 1.3), factor(1, o_cols, 2.9))];
        let mut s = session(&dir, 1 << 30);
        let components = serde_json::json!({"shared": uses.iter().map(|(n, l, r)| serde_json::json!({"operator": n, "left": rows_of(l), "right": rows_of(r)})).collect::<Vec<_>>()});
        let reply = s.handle(&serde_json::from_value(serde_json::json!({"op": "register", "model": "m", "components": components})).expect("request")).expect("register");
        assert_eq!(reply["uses"].as_u64(), Some(2));
        let clean = run(&mut s, "m", serde_json::json!({}));
        let edit = |alpha: f64| serde_json::json!({"edits": [{"component": {"kind": "registered", "id": "shared"}, "alpha": alpha}]});
        assert_eq!(largest_difference(&clean, &run(&mut s, "m", edit(1.0))), 0.0);
        let products: Vec<(&str, Array2<f64>)> = uses.iter().map(|(n, l, r)| (*n, l.dot(r))).collect();
        for alpha in [0.0, 2.5, -1.0] {
            s.models.insert("literal".into(), literally_edited(&native, &products, alpha - 1.0));
            let (registered, literal) = (run(&mut s, "m", edit(alpha)), run(&mut s, "literal", serde_json::json!({})));
            assert!(largest_difference(&registered, &literal) < 1e-12, "alpha {alpha}: {}", largest_difference(&registered, &literal));
            assert!(largest_difference(&clean, &registered) > 1e-9, "alpha {alpha} changes nothing");
        }
        // A neuron's write registered as its down column times the unit row selecting it.
        let down = native.program.operators[native.operator("blocks.1.down_proj").expect("down")].matrix_cow().into_owned();
        let mut select = Array2::zeros((1, down.ncols()));
        select[[0, 3]] = 1.0;
        let column = down.column(3).to_owned().insert_axis(ndarray::Axis(1));
        s.register("m", "neuron", &[crate::oracle::ComponentUse { operator: "blocks.1.down_proj".into(), left: rows_of(&column), right: rows_of(&select) }]).expect("register neuron");
        let registered = run(&mut s, "m", serde_json::json!({"edits": [{"component": {"kind": "registered", "id": "neuron"}, "alpha": 0.0}]}));
        let neuron = run(&mut s, "m", serde_json::json!({"edits": [{"component": {"kind": "neuron", "layer": 1, "index": 3}, "alpha": 0.0}]}));
        assert!(largest_difference(&registered, &neuron) < 1e-12);
        // Factors that do not fit their operator are refused.
        assert!(s.register("m", "bad", &[crate::oracle::ComponentUse { operator: "blocks.1.c_fc".into(), left: rows_of(&factor(fc_rows + 1, 1, 0.0)), right: rows_of(&factor(1, fc_cols, 0.0)) }]).is_err());
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}

/// VPD's subcomponents registered from an exported decomposition: a query subcomponent is held by
/// every head's query operator (its output rows split by head) and an output-map subcomponent by
/// every head's output operator (its input columns split by head), and each edit runs as the
/// literal edit of those slices of `u_c v_cᵀ`; at alpha = 1 the model exactly.
#[test]
fn vpd_subcomponents_register_on_every_split_operator() {
    let dir = tiny_export("oracle_vpd_register", 2);
    let native = Native::load(&dir).expect("load");
    let decomposition = std::env::temp_dir().join(format!("gam_mpd_oracle_vpd_decomposition_{}", std::process::id()));
    std::fs::create_dir_all(&decomposition).expect("decomposition dir");
    let (d, hd, heads, kv) = (native.width, native.head_width, native.heads, native.kv_heads);
    let mlp = native.mlp_width;
    let count = 3;
    let mut sites = Vec::new();
    let mut files = serde_json::Map::new();
    for l in 0..2 {
        for (kind, (d_in, d_out)) in [("attn.q_proj", (d, heads * hd)), ("attn.k_proj", (d, kv * hd)), ("attn.v_proj", (d, kv * hd)), ("attn.o_proj", (heads * hd, d)), ("mlp.c_fc", (d, mlp)), ("mlp.down_proj", (mlp, d))] {
            let name = format!("h.{l}.{kind}");
            for (part, (rows, cols)) in [("U", (count, d_out)), ("V", (d_in, count))] {
                let values: Vec<u8> = (0..rows * cols).flat_map(|i| (((i * 13 + l * 5) as f64 * 0.31).cos() * 0.3).to_le_bytes()).collect();
                std::fs::write(decomposition.join(format!("{name}.{part}.f64")), values).expect("write factor");
                files.insert(format!("{name}.{part}"), serde_json::json!({"shape": [rows, cols]}));
            }
            sites.push(name);
        }
    }
    std::fs::write(decomposition.join("export.json"), serde_json::to_vec(&serde_json::json!({"config": {"sites": sites}, "files": files})).expect("json")).expect("write export");
    let mut s = session(&dir, 1 << 30);
    let reply = s.handle(&serde_json::from_value(serde_json::json!({"op": "register_vpd", "model": "m", "decomposition": decomposition.display().to_string()})).expect("request")).expect("register");
    assert_eq!(reply["components"].as_u64(), Some((2 * 6 * count) as u64));
    let factors = crate::explanation_battery::load_factors(&decomposition).expect("factors");
    let clean = run(&mut s, "m", serde_json::json!({}));
    for (site, c, letter, by_rows) in [(0usize, 1usize, 'q', true), (6 + 3, 2, 'o', false), (6 + 1, 0, 'k', true)] {
        let f = &factors[site];
        let p = f.u.row(c).to_owned().insert_axis(ndarray::Axis(1)).dot(&f.v.column(c).to_owned().insert_axis(ndarray::Axis(0)));
        let parts = if letter == 'q' || letter == 'o' { heads } else { kv };
        let slices: Vec<(String, Array2<f64>)> = (0..parts)
            .map(|h| {
                let range = h * hd..(h + 1) * hd;
                let slice = if by_rows { p.slice(ndarray::s![range, ..]).to_owned() } else { p.slice(ndarray::s![.., range]).to_owned() };
                (format!("blocks.{}.{letter}{h}", f.layer), slice)
            })
            .collect();
        let named: Vec<(&str, Array2<f64>)> = slices.iter().map(|(n, m)| (n.as_str(), m.clone())).collect();
        let id = format!("{}:{c}", f.name);
        assert_eq!(largest_difference(&clean, &run(&mut s, "m", serde_json::json!({"edits": [{"component": {"kind": "registered", "id": id}, "alpha": 1.0}]}))), 0.0);
        s.models.insert("literal".into(), literally_edited(&native, &named, 2.0));
        let registered = run(&mut s, "m", serde_json::json!({"edits": [{"component": {"kind": "registered", "id": id}, "alpha": 3.0}]}));
        let literal = run(&mut s, "literal", serde_json::json!({}));
        assert!(largest_difference(&registered, &literal) < 1e-12, "{id}: {}", largest_difference(&registered, &literal));
        assert!(largest_difference(&clean, &registered) > 1e-9, "{id} changes nothing");
    }
    std::fs::remove_dir_all(&dir).expect("cleanup");
    std::fs::remove_dir_all(&decomposition).expect("cleanup");
}
