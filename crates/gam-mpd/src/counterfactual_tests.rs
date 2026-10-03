use super::counterfactual::{Action, Decoder, Donor, Library, Maps, Program, Rows, Selection, Selector, score, site_index, site_name};
use ndarray::{Array2, Axis};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A small random decoder export (2 layers, 2 heads of 4, MLP 16, vocabulary 11).
fn tiny_export(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gam_counterfactual_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut draw = |n: usize, scale: f64| -> Vec<f64> {
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * scale
            })
            .collect()
    };
    let mut files = serde_json::Map::new();
    let mut write = |dir: &Path, name: &str, shape: [usize; 2], values: Vec<f64>| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{name}.f64")), bytes).expect("write tensor");
        files.insert(name.to_string(), serde_json::json!({"shape": shape}));
    };
    let (d, mlp, vocab) = (8, 16, 11);
    write(&dir, "wte", [vocab, d], draw(vocab * d, 2.0));
    write(&dir, "final_norm.gain", [1, d], draw(d, 0.4).iter().map(|g| 1.0 + g).collect());
    for l in 0..2 {
        for (name, shape) in [("attn.q_proj", [d, d]), ("attn.k_proj", [d, d]), ("attn.v_proj", [d, d]), ("attn.o_proj", [d, d]), ("mlp.c_fc", [mlp, d]), ("mlp.down_proj", [d, mlp])] {
            write(&dir, &format!("blocks.{l}.{name}"), shape, draw(shape[0] * shape[1], 1.0));
        }
        for g in ["rms1", "rms2"] {
            write(&dir, &format!("blocks.{l}.{g}.gain"), [1, d], draw(d, 0.4).iter().map(|v| 1.0 + v).collect());
        }
    }
    let record = serde_json::json!({
        "config": {"d_model": d, "n_layers": 2, "n_heads": 2, "head_dim": 4, "d_mlp": mlp, "vocab": vocab, "rope_theta": 10000.0,
                   "rope_pairing": "rotate_half", "norm_eps": 1e-6, "mlp_act": "gelu_tanh"},
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).expect("export.json");
    dir
}

/// Each site's exact library of input-coordinate units: `v_c = e_c`, `u_c = W e_c`.
fn coordinate_libraries(decoder: &Decoder) -> Vec<Library> {
    (0..decoder.sites())
        .map(|site| {
            let w = decoder.native(site);
            Library { v: Array2::eye(w.ncols()), u: w.t().to_owned() }
        })
        .collect()
}

fn all_on(libraries: &[Library], rows: usize) -> Selection {
    let row: Vec<(u32, u32)> = libraries.iter().enumerate().flat_map(|(k, l)| (0..l.v.nrows()).map(move |c| (k as u32, c as u32))).collect();
    Selection { rows: vec![row; rows] }
}

fn max_gap(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// The donor states `actions` read, from one program on `tokens`.
fn donor(decoder: &Decoder, maps: Maps<'_>, tokens: &[u32], actions: &[Action]) -> Donor {
    let record = actions.iter().filter_map(Action::donor_state).map(|k| (k, None)).collect();
    let mut program = Program { maps, actions: &[], donor: None, record };
    decoder.forward(tokens, &mut program, &[]);
    Donor { states: program.record.into_iter().map(|(k, v)| (k, v.expect("donor state reached"))).collect() }
}

#[test]
fn an_exact_library_all_on_reproduces_every_native_action() {
    let dir = tiny_export("exact");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let libraries = coordinate_libraries(&decoder);
    let tokens: Vec<u32> = (0..9).map(|t| (t * 7 % 11) as u32).collect();
    let donor_tokens: Vec<u32> = (0..9).map(|t| (t * 3 % 11) as u32).collect();
    let edit_left = Arc::new(Array2::from_shape_fn((8, 2), |(i, j)| ((i + 3 * j) as f64 * 0.37).sin()));
    let edit_right = Arc::new(Array2::from_shape_fn((16, 2), |(i, j)| ((2 * i + j) as f64 * 0.21).cos() * 0.2));
    let cases: Vec<Vec<Action>> = vec![
        vec![],
        // A neuron at one row, and a head at every row.
        vec![Action::ScaleInput { site: site_index(0, 5), rows: Rows::One(4), cols: (3, 4), scale: 0.0 }],
        vec![Action::ScaleInput { site: site_index(1, 3), rows: Rows::All, cols: (4, 8), scale: 2.0 }],
        // Resampling a site's input and another site's output, together.
        vec![
            Action::MixInput { site: site_index(0, 1), row: 5, alpha: 0.5 },
            Action::MixOutput { site: site_index(1, 4), row: 6, alpha: 1.0 },
        ],
        vec![Action::AddMap { site: site_index(1, 5), left: edit_left, right: edit_right }],
    ];
    let rows: Vec<usize> = (0..tokens.len()).collect();
    let mut clean_native = Program { maps: Maps::Native(&decoder), actions: &[], donor: None, record: Vec::new() };
    let clean = decoder.forward(&tokens, &mut clean_native, &rows);
    for actions in &cases {
        let native_donor = donor(&decoder, Maps::Native(&decoder), &donor_tokens, actions);
        let mut donor_selection = all_on(&libraries, donor_tokens.len());
        let own_donor = donor(&decoder, Maps::Units { libraries: &libraries, selector: &mut donor_selection }, &donor_tokens, actions);
        let interface = [5usize, 7];
        let mut native = Program { maps: Maps::Native(&decoder), actions, donor: Some(&native_donor), record: Vec::new() };
        let a = decoder.forward(&tokens, &mut native, &interface);
        let mut selection = all_on(&libraries, tokens.len());
        let selector: &mut dyn Selector = &mut selection;
        let mut explained = Program { maps: Maps::Units { libraries: &libraries, selector }, actions, donor: Some(&own_donor), record: Vec::new() };
        let b = decoder.forward(&tokens, &mut explained, &interface);
        assert!(max_gap(&a.residual, &b.residual) < 1e-10, "{actions:?}: {}", max_gap(&a.residual, &b.residual));
        let from = actions.iter().map(Action::first_row).min().unwrap_or(0);
        let scores = score(&decoder, &a, &b, &clean, &interface, from, 3);
        assert!(scores.kl.abs() < 1e-12 && scores.top1_agree == 1.0 && scores.interface_kl.iter().all(|k| k.abs() < 1e-12), "{actions:?}: {scores:?}");
        if !actions.is_empty() {
            assert!(scores.native_effect > 1e-9, "{actions:?} changed nothing: {scores:?}");
        }
    }
    assert_eq!(site_name(11), "blocks.1.down_proj");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_partial_explanation_is_scored_by_its_disagreement() {
    let dir = tiny_export("partial");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let libraries = coordinate_libraries(&decoder);
    let tokens: Vec<u32> = (0..6).map(|t| (t * 5 % 11) as u32).collect();
    let rows: Vec<usize> = (0..tokens.len()).collect();
    let mut clean_native = Program { maps: Maps::Native(&decoder), actions: &[], donor: None, record: Vec::new() };
    let clean = decoder.forward(&tokens, &mut clean_native, &rows);
    // Half of every site's units at every row.
    let mut selection = all_on(&libraries, tokens.len());
    for row in &mut selection.rows {
        row.retain(|(_, c)| c % 2 == 0);
    }
    let selector: &mut dyn Selector = &mut selection;
    let mut explained = Program { maps: Maps::Units { libraries: &libraries, selector }, actions: &[], donor: None, record: Vec::new() };
    let b = decoder.forward(&tokens, &mut explained, &[2]);
    let mut native = Program { maps: Maps::Native(&decoder), actions: &[], donor: None, record: Vec::new() };
    let a = decoder.forward(&tokens, &mut native, &[2]);
    let scores = score(&decoder, &a, &b, &clean, &[2], 0, 4);
    assert!(scores.kl > 1e-6 && scores.native_effect.abs() < 1e-12, "{scores:?}");
    assert_eq!(clean.layers[0].len_of(Axis(0)), tokens.len());
    std::fs::remove_dir_all(&dir).ok();
}
