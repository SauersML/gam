use super::counterfactual::{Action, Decoder, Donor, InputChange, OutputChange, Program, Rows, passages, score, site_index};
use super::import::import_language_model;
use super::test_support::tiny_export;
use ndarray::Array2;
use std::sync::Arc;

fn max_gap(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// The donor states `actions` read, from the native program on `tokens`.
fn donor(decoder: &Decoder, tokens: &[u32], actions: &[Action]) -> Donor {
    let mut program = Program::new(decoder, &[], None);
    program.record = actions.iter().filter_map(Action::donor_state).map(|k| (k, None)).collect();
    decoder.forward(tokens, &mut program, &[]);
    Donor { states: program.record.into_iter().map(|(k, v)| (k, v.expect("donor state reached"))).collect() }
}

#[test]
fn every_native_action_has_an_effect_and_a_program_scores_zero_against_itself() {
    let dir = tiny_export("counterfactual_actions", 2);
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let tokens: Vec<u32> = (0..9).map(|t| (t * 7 % 11) as u32).collect();
    let donor_tokens: Vec<u32> = (0..9).map(|t| (t * 3 % 11) as u32).collect();
    let edit_left = Arc::new(Array2::from_shape_fn((8, 2), |(i, j)| ((i + 3 * j) as f64 * 0.37).sin()));
    let edit_right = Arc::new(Array2::from_shape_fn((16, 2), |(i, j)| ((2 * i + j) as f64 * 0.21).cos() * 0.2));
    let cases: Vec<Vec<Action>> = vec![
        vec![],
        // A neuron at one row, and a head at every row.
        vec![Action::Input { site: site_index(0, 5), change: InputChange::Scale { rows: Rows::One(4), cols: (3, 4), scale: 0.0 } }],
        vec![Action::Input { site: site_index(1, 3), change: InputChange::Scale { rows: Rows::All, cols: (4, 8), scale: 2.0 } }],
        // Resampling a site's input and another site's output, together.
        vec![
            Action::Input { site: site_index(0, 1), change: InputChange::Mix { row: 5, alpha: 0.5 } },
            Action::Output { site: site_index(1, 4), change: OutputChange::Mix { row: 6, alpha: 1.0 } },
        ],
        vec![Action::Output { site: site_index(1, 5), change: OutputChange::Add { left: edit_left, right: edit_right } }],
    ];
    let rows: Vec<usize> = (0..tokens.len()).collect();
    let mut clean_native = Program::new(&decoder, &[], None);
    let clean = decoder.forward(&tokens, &mut clean_native, &rows);
    for actions in &cases {
        let native_donor = donor(&decoder, &donor_tokens, actions);
        let interface = [5usize, 7];
        let mut first = Program::new(&decoder, actions, Some(&native_donor));
        let a = decoder.forward(&tokens, &mut first, &interface);
        let mut second = Program::new(&decoder, actions, Some(&native_donor));
        let b = decoder.forward(&tokens, &mut second, &interface);
        let from = actions.iter().map(Action::first_row).min().unwrap_or(0);
        let scores = score(&decoder, &a, &b, &clean, &interface, from, 3);
        assert!(scores.kl.abs() < 1e-12 && scores.top1_agree == 1.0 && scores.interface_kl.iter().all(|k| k.abs() < 1e-12), "{actions:?}: {scores:?}");
        assert_eq!(scores.native_effect > 1e-9, !actions.is_empty(), "{actions:?}: {scores:?}");
    }
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}

#[test]
fn the_decoder_agrees_with_the_imported_program() {
    let dir = tiny_export("counterfactual_import", 2);
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let context = 10;
    let imported = import_language_model(&dir, 2, context).expect("import");
    let model = imported.program;
    let tokens = passages(&dir, context).expect("passages");
    for (passage, tokens) in tokens.iter().take(2).enumerate() {
        let rows: Vec<usize> = (passage * context..(passage + 1) * context).collect();
        let logits = model.execute(&imported.family.select(&rows), false).expect("native program").values.swap_remove(model.output);
        let mut native = Program::new(&decoder, &[], None);
        let ours = decoder.log_probs(&decoder.forward(tokens, &mut native, &[]).residual);
        let mut theirs = logits;
        for mut row in theirs.outer_iter_mut() {
            let max = row.fold(f64::NEG_INFINITY, |m, v| m.max(*v));
            let total = row.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
            row.mapv_inplace(|v| v - total);
        }
        assert!(max_gap(&ours, &theirs) < 1e-9, "passage {passage}: native gap {}", max_gap(&ours, &theirs));
    }
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}
