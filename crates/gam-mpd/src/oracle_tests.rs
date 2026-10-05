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
    operator.body = OperatorBody::Dense { values, present: present.clone(), precision };
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
