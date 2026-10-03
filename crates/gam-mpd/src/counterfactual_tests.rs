use super::counterfactual::{Decoder, Explained, ExplanationChange, Library, Native, NativeChange, Selection, score, site_name};
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};

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

fn all_on(decoder: &Decoder, libraries: &[Library], rows: usize) -> Selection {
    let row: Vec<(u32, u32, f64)> = (0..decoder.sites()).flat_map(|k| (0..libraries[k].v.nrows()).map(move |c| (k as u32, c as u32, 1.0))).collect();
    Selection { rows: vec![row; rows] }
}

fn max_gap(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

#[test]
fn an_exact_library_all_on_reproduces_every_native_intervention() {
    let dir = tiny_export("exact");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let libraries = coordinate_libraries(&decoder);
    let tokens: Vec<u32> = (0..9).map(|t| (t * 7 % 11) as u32).collect();
    let donor: Vec<u32> = (0..9).map(|t| (t * 3 % 11) as u32).collect();
    let selection = all_on(&decoder, &libraries, tokens.len());
    let run = |native_change: NativeChange, explanation_change: ExplanationChange| {
        let mut native = Native { decoder: &decoder, record: Vec::new(), intervention: native_change };
        let a = decoder.residual(&tokens, &mut native);
        let mut explained = Explained { libraries: &libraries, selection: &selection, record: Vec::new(), intervention: explanation_change };
        let b = decoder.residual(&tokens, &mut explained);
        (a, b)
    };
    let (a, b) = run(NativeChange::None, ExplanationChange::None);
    assert!(max_gap(&a, &b) < 1e-10, "clean: {}", max_gap(&a, &b));
    // A unit scaled at one row: the native rank-one weight change is the explanation's mask.
    let site = 4;
    let (u, v) = (libraries[site].u.row(3).to_owned(), libraries[site].v.row(3).to_owned());
    let (a, b) = run(NativeChange::Unit { site, row: 5, u, v, scale: 0.5 }, ExplanationChange::Unit { site, row: 5, unit: 3, scale: 0.5 });
    assert!(max_gap(&a, &b) < 1e-10, "unit: {}", max_gap(&a, &b));
    let (clean, _) = run(NativeChange::None, ExplanationChange::None);
    assert!(max_gap(&a, &clean) > 1e-6, "the unit change did nothing");
    // A site's input patched from a donor.
    let mut native = Native { decoder: &decoder, record: vec![(7, None)], intervention: NativeChange::None };
    decoder.residual(&donor, &mut native);
    let patch: Array1<f64> = native.record[0].1.as_ref().expect("recorded").row(4).to_owned();
    let (a, b) = run(NativeChange::Input { site: 7, row: 4, input: patch.clone() }, ExplanationChange::Input { site: 7, row: 4, input: patch });
    assert!(max_gap(&a, &b) < 1e-10, "input: {}", max_gap(&a, &b));
    // A weight edit equal to one unit's write replaced.
    let site = 11;
    let unit = 2;
    let write: Array1<f64> = libraries[site].u.row(unit).mapv(|x| -2.0 * x + 0.1);
    let left = (&write - &libraries[site].u.row(unit)).insert_axis(ndarray::Axis(1));
    let right = libraries[site].v.row(unit).to_owned().insert_axis(ndarray::Axis(1));
    let (a, b) = run(NativeChange::Edit { site, left, right }, ExplanationChange::Write { site, unit, write });
    assert!(max_gap(&a, &b) < 1e-10, "edit: {}", max_gap(&a, &b));
    let (kl, effect, agree) = score(&decoder, &a, &b, &clean, 2, 3);
    assert!(kl.abs() < 1e-12 && effect > 0.0 && agree == 1.0, "score {kl} {effect} {agree}");
    assert_eq!(site_name(11), "blocks.1.down_proj");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_unselected_unit_is_predicted_to_do_nothing() {
    let dir = tiny_export("unselected");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let libraries = coordinate_libraries(&decoder);
    let tokens: Vec<u32> = (0..6).map(|t| (t * 5 % 11) as u32).collect();
    let mut selection = all_on(&decoder, &libraries, tokens.len());
    // Unit 1 of site 2 is left out at row 3.
    selection.rows[3].retain(|(k, c, _)| !(*k == 2 && *c == 1));
    assert!(!selection.has(3, 2, 1) && selection.has(2, 2, 1));
    let mut before = Explained { libraries: &libraries, selection: &selection, record: Vec::new(), intervention: ExplanationChange::None };
    let a = decoder.residual(&tokens, &mut before);
    let mut removed = Explained { libraries: &libraries, selection: &selection, record: Vec::new(), intervention: ExplanationChange::Unit { site: 2, row: 3, unit: 1, scale: 0.0 } };
    let b = decoder.residual(&tokens, &mut removed);
    assert!(max_gap(&a, &b) == 0.0);
    std::fs::remove_dir_all(&dir).ok();
}
