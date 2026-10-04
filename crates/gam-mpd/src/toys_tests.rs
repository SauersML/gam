#![cfg(test)]
//! The toys' interventions on a small random transformer export: a scaled head or unit at every
//! position is the same weight edit of its output map, a scale of 1 or a patch from the prompt
//! itself changes nothing, and an explanation's edited run without edits is its own autonomous
//! execution.

use crate::explanation::{Explanation, Fitted, Replacement};
use crate::masked::{Library, matrix, sites};
use crate::operator_program::OperatorBody;
use crate::toys::{Case, Edit, Explained, Kind, Native, Place, Question, Runner};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

const VOCAB: usize = 4;
const POSITIONS: usize = 3;
const D: usize = 6;
const HEADS: usize = 2;
const D_HEAD: usize = 3;
const HIDDEN: usize = 5;

fn write(dir: &std::path::Path, name: &str, a: &Array2<f64>) -> serde_json::Value {
    let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f64")), bytes).expect("a tensor file");
    json!({"shape": [a.nrows(), a.ncols()]})
}

/// A random two-layer transformer export with every prompt as its samples, and one question.
fn case(seed: u64) -> Case {
    let dir: PathBuf = std::env::temp_dir().join(format!("gam_mpd_toys_{}_{seed}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary case directory");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut random = |r: usize, c: usize| Array2::from_shape_fn((r, c), |_| rng.random_range(-1.0..1.0));
    let mut files = serde_json::Map::new();
    for (name, (r, c)) in [("W_E", (VOCAB, D)), ("W_pos", (POSITIONS, D)), ("W_U", (VOCAB, D))] {
        files.insert(name.to_string(), write(&dir, name, &random(r, c)));
    }
    for l in 0..2 {
        for (name, (r, c)) in [("W_Q", (HEADS * D_HEAD, D)), ("W_K", (HEADS * D_HEAD, D)), ("W_V", (HEADS * D_HEAD, D)), ("W_O", (D, HEADS * D_HEAD)), ("W_in", (HIDDEN, D)), ("W_out", (D, HIDDEN))] {
            files.insert(format!("blocks.{l}.{name}"), write(&dir, &format!("blocks.{l}.{name}"), &random(r, c)));
        }
    }
    let prompts: Vec<Vec<f64>> = (0..VOCAB.pow(POSITIONS as u32)).map(|i| (0..POSITIONS).map(|j| ((i / VOCAB.pow(j as u32)) % VOCAB) as f64).collect()).collect();
    let samples = Array2::from_shape_fn((prompts.len(), POSITIONS), |(r, c)| prompts[r][c]);
    write(&dir, "inputs", &samples);
    write(&dir, "transcript", &samples);
    let record = json!({
        "model": "toy", "kind": "transformer",
        "config": {"n_layers": 2, "n_heads": HEADS, "d_model": D, "d_head": D_HEAD, "d_mlp": HIDDEN, "n_ctx": POSITIONS, "causal": true},
        "input": {"vocab_size": VOCAB, "seq_len": POSITIONS},
        "output": {"n_classes": VOCAB, "readout_positions": [1, 2]},
        "samples": {"file": "inputs.f64", "shape": [prompts.len(), POSITIONS]},
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).expect("export.json");
    let clean = vec![vec![0.0; VOCAB]; 2];
    let question = json!({"id": "toy/000", "tag": "t", "split": "dev", "input": [1.0, 2.0, 3.0], "edits": [], "clean": clean, "outcome": clean});
    std::fs::write(dir.join("questions.json"), json!({"questions": [question]}).to_string()).expect("questions.json");
    std::fs::write(dir.join("truth.json"), json!({"mechanism": "", "components": []}).to_string()).expect("truth.json");
    Case::load(&dir).expect("the toy case loads")
}

fn question(edits: Vec<Edit>) -> Question {
    Question { id: "q".to_string(), tag: String::new(), held_out: false, input: vec![1.0, 2.0, 3.0], edits, clean: Array2::zeros((2, VOCAB)), outcome: None }
}

fn place(kind: Kind, layer: i64, head: Option<usize>, units: Option<Vec<usize>>) -> Place {
    Place { kind, layer, head, position: None, units }
}

/// `program` with operator `name`'s columns `columns` (all when none) times `factor`.
fn scaled(case: &Case, name: &str, columns: Option<&[usize]>, factor: f64) -> crate::operator_program::OperatorProgram {
    let mut program = case.imported.program.clone();
    let op = program.operators.iter().position(|o| o.name == name).expect("the operator");
    let operator = Arc::make_mut(&mut program.operators[op]);
    let OperatorBody::Dense { values, .. } = &mut operator.body else { panic!("{name} is not dense") };
    for c in 0..values.ncols() {
        if columns.is_none_or(|cs| cs.contains(&c)) {
            values.column_mut(c).mapv_inplace(|v| v * factor);
        }
    }
    program
}

fn close(a: &Array2<f64>, b: &Array2<f64>, scale: f64) -> bool {
    a.dim() == b.dim() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12 * scale.max(1.0))
}

/// A head or unit scaled at every position is its output map's columns scaled.
#[test]
fn native_scales_are_weight_edits() {
    let case = case(1);
    let native = Native(&case.imported.program);
    for (edit, name, columns, factor) in [
        (place(Kind::Head, 1, Some(1), None), "blocks.1.W_O1", None, 0.0),
        (place(Kind::Head, 0, Some(0), None), "blocks.0.W_O0", None, 2.5),
        (place(Kind::Mlp, 0, None, Some(vec![1, 3])), "blocks.0.W_out", Some(vec![1, 3]), -0.5),
    ] {
        let predicted = case.predict(&native, &question(vec![Edit::Scale { place: edit, factor }])).expect("an edited forward");
        let program = scaled(&case, name, columns.as_deref(), factor);
        let expected = case.predict(&Native(&program), &question(Vec::new())).expect("the weight edit");
        assert!(close(&predicted, &expected, 1.0), "{name}: {predicted} vs {expected}");
    }
}

/// A scale of 1 and a patch from the prompt itself change nothing; a patch of the last residual
/// from a donor gives the donor's logits; the explanation's run without edits is its autonomous
/// execution, selections and logits.
#[test]
fn identities_hold_in_both_programs() {
    let case = case(2);
    let model = &case.imported.program;
    let chosen = crate::explanation::in_execution_order(sites(model));
    let fitted: Vec<Fitted> = chosen
        .iter()
        .map(|site| {
            let w = matrix(model, site).expect("the site's map");
            let decomposition = gam_linalg::decompose::svd(w.view(), false).expect("an SVD");
            let rank = decomposition.singular_values.len();
            let u = Array2::from_shape_fn((rank, w.nrows()), |(c, i)| decomposition.u[[i, c]] * decomposition.singular_values[c]);
            let library = Library { v: decomposition.vt.clone(), u, mean: Array1::zeros(w.ncols()) };
            let (d_out, d_in) = w.dim();
            Fitted::new(site.clone(), w, (library, vec![1; rank], vec![4.0; rank]), (Array2::eye(d_out), Array2::eye(d_in)), 1e3).expect("a fitted site")
        })
        .collect();
    let explanation = Explanation { observations: 1e3, sites: fitted };
    let explained = Explained::new(model, &explanation).expect("the explanation's program");
    let prompt = [1.0, 2.0, 3.0];
    let family = case.family(&[&prompt]).expect("a family");
    let members: Vec<usize> = (0..explanation.sites.len()).collect();
    let masked = explanation.masked(model, &members).expect("masked");
    let (reference, sets) = explanation.execute(&masked, &members, &family).expect("autonomous execution");
    let (ours, our_sets) = explained.run_with_sets(&family, &[]).expect("the edited run");
    assert_eq!(sets, our_sets);
    assert_eq!(reference.values[masked.program.output], ours.values[explained.output()]);
    let runners: [&dyn Runner; 2] = [&Native(model), &explained];
    for runner in runners {
        let clean = case.predict(runner, &question(Vec::new())).expect("clean");
        for edit in [
            Edit::Scale { place: place(Kind::Head, 1, Some(0), None), factor: 1.0 },
            Edit::Patch { place: place(Kind::Mlp, 0, None, None), donor: prompt.to_vec() },
            Edit::Patch { place: place(Kind::Resid, 0, None, None), donor: prompt.to_vec() },
        ] {
            assert!(close(&case.predict(runner, &question(vec![edit.clone()])).expect("an identity edit"), &clean, 1.0), "{edit:?}");
        }
        let donor = vec![3.0, 0.0, 1.0];
        let patched = case.predict(runner, &question(vec![Edit::Patch { place: place(Kind::Resid, 1, None, None), donor: donor.clone() }])).expect("a patch");
        let moved = case.predict(runner, &Question { input: donor, ..question(Vec::new()) }).expect("the donor");
        assert!(close(&patched, &moved, 1.0));
    }
}
