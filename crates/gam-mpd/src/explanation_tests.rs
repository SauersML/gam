//! The explanation pipeline end to end on a small rotary language model: sites fitted in
//! execution order on hybrid inputs, run on their own states, scored by every stage.

use super::explanation::{Explanation, Given, Passage, Replacement, Settings, bits_per_word, fit, replaced, site_switch};
use super::import::import_language_model;
use super::masked::{kl_score_only, sites};
use super::operator_program::{FamilyInputs, SlotValues};
use std::path::{Path, PathBuf};

/// A small random decoder export (`layers` layers, 2 heads of 4, MLP 16, vocabulary 11) with six
/// token rows of 12.
fn tiny_export(tag: &str, layers: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gam_explanation_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut state = 0x2545_F491_4F6C_DD1Du64;
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
    let (d, mlp, vocab, rows, context) = (8, 16, 11, 6, 12);
    write(&dir, "wte", [vocab, d], draw(vocab * d, 2.0));
    write(&dir, "final_norm.gain", [1, d], draw(d, 0.4).iter().map(|g| 1.0 + g).collect());
    for l in 0..layers {
        for (name, shape) in [("attn.q_proj", [d, d]), ("attn.k_proj", [d, d]), ("attn.v_proj", [d, d]), ("attn.o_proj", [d, d]), ("mlp.c_fc", [mlp, d]), ("mlp.down_proj", [d, mlp])] {
            write(&dir, &format!("blocks.{l}.{name}"), shape, draw(shape[0] * shape[1], 1.0));
        }
        for g in ["rms1", "rms2"] {
            write(&dir, &format!("blocks.{l}.{g}.gain"), [1, d], draw(d, 0.4).iter().map(|v| 1.0 + v).collect());
        }
    }
    let tokens: Vec<f64> = draw(rows * context, 1.0).iter().map(|x| ((x + 0.5) * vocab as f64).floor().min(vocab as f64 - 1.0)).collect();
    write(&dir, "tokens", [rows, context], tokens);
    let record = serde_json::json!({
        "config": {"d_model": d, "n_layers": layers, "n_heads": 2, "n_kv_heads": 2, "head_dim": 4, "d_mlp": mlp, "vocab": vocab, "rope_theta": 10000.0,
                   "rope_pairing": "rotate_half", "norm_eps": 1e-6, "mlp_act": "gelu_tanh", "tied_embeddings": true},
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).expect("export.json");
    dir
}

const CONTEXT: usize = 12;

/// The export's model, its first two sequences as training batches and its last four as passages,
/// and the explanation of its sites named with `prefix` fitted on the batches.
fn fitted(tag: &str, layers: usize, prefix: &str) -> (super::operator_program::OperatorProgram, Vec<Passage>, Explanation) {
    let dir = tiny_export(tag, layers);
    let imported = import_language_model(&dir, 6, CONTEXT).expect("import");
    let model = imported.program;
    let family = imported.contract.family;
    let sequence = |s: usize| family.select(&(s * CONTEXT..(s + 1) * CONTEXT).collect::<Vec<_>>());
    let batches: Vec<FamilyInputs> = (0..2).map(sequence).collect();
    let passages: Vec<Passage> = (2..6).map(|s| Passage::new(&model, sequence(s)).expect("passage")).collect();
    let chosen = sites(&model).into_iter().filter(|s| s.name.starts_with(prefix)).collect();
    let settings = Settings { observations: 1e4, rounds: 3, evidence_rounds: 3, draws: 2, seed: 7 };
    let mut seen = Vec::new();
    let explanation = fit(&model, chosen, &batches, &settings, |_| Ok(None), |f| {
        seen.push(f.site.name.clone());
        Ok(())
    })
    .expect("fit");
    assert_eq!(seen, explanation.sites.iter().map(|f| f.site.name.clone()).collect::<Vec<_>>(), "every fitted site reported, in order");
    std::fs::remove_dir_all(&dir).expect("remove the export");
    (model, passages, explanation)
}

#[test]
fn a_layer_is_fitted_in_execution_order_and_runs_on_its_own_states() {
    let (model, passages, explanation) = fitted("layer", 1, "blocks.0.");
    let names: Vec<String> = explanation.sites.iter().map(|f| f.site.name.clone()).collect();
    assert_eq!(names.len(), 6, "{names:?}");
    let at = |n: &str| names.iter().position(|m| m.ends_with(n)).expect("site");
    // Attention's maps before its output's, the MLP's input before its output.
    assert!(at(".q").max(at(".k")).max(at(".v")) < at(".o") && at(".o") < at(".c_fc") && at(".c_fc") < at(".down_proj"), "{names:?}");
    for f in &explanation.sites {
        assert_eq!(f.ranks.iter().sum::<usize>(), f.library.v.nrows(), "{}", f.site.name);
        assert!(f.bits.iter().all(|b| b.is_finite() && *b >= 0.0), "{}: {:?}", f.site.name, f.bits);
    }
    let members: Vec<usize> = (0..6).collect();
    let masked = explanation.masked(&model, &members).expect("masked");
    for passage in &passages {
        let (trace, masks) = explanation.execute(&masked, &members, &passage.base).expect("autonomous");
        // The decisions it made, held fixed, replay its forward exactly.
        let replay = masked.program.execute(&masked.family(&passage.base, &masks), false).expect("replay");
        assert_eq!(trace.values, replay.values);
        let kl = kl_score_only(&passage.target, &trace.values[masked.program.output]);
        assert!(kl.iter().all(|k| k.is_finite() && *k >= -1e-12), "{kl:?}");
        // A row's decisions read only its own and earlier rows: a later token changes none of them.
        let mut later = passage.base.clone();
        if let SlotValues::Tokens(ids) = &mut later.slots[0] {
            ids[CONTEXT - 1] = (ids[CONTEXT - 1] + 1) % 11;
        }
        let (_, changed) = explanation.execute(&masked, &members, &later).expect("autonomous");
        for (a, b) in masks.iter().zip(&changed) {
            assert_eq!(a.slice(ndarray::s![..CONTEXT - 1, ..]), b.slice(ndarray::s![..CONTEXT - 1, ..]));
        }
    }
    // The explanation's own decisions, given back as fixed sets, score the same.
    let ours = replaced(&model, &explanation, &members, &passages).expect("replaced");
    let given = Given {
        sites: explanation.sites(),
        libraries: explanation.sites.iter().map(|f| f.library.clone()).collect(),
        ranks: explanation.sites.iter().map(|f| f.ranks.clone()).collect(),
        masks: ours.iter().map(|(_, m)| m.clone()).collect(),
    };
    let fixed = replaced(&model, &given, &members, &passages).expect("given");
    for ((a, _), (b, _)) in ours.iter().zip(&fixed) {
        assert_eq!(a, b);
    }
    // Every block that ran is paid its bits on its row.
    let bits: Vec<Vec<f64>> = explanation.sites.iter().map(|f| f.bits.clone()).collect();
    let per_word = bits_per_word(&ours[0].1, &bits).expect("bits");
    for r in 0..CONTEXT {
        let expected: f64 = ours[0].1.iter().zip(&bits).map(|(m, b)| (0..b.len()).filter(|c| m[[r, *c]] == 1.0).map(|c| b[c]).sum::<f64>()).sum();
        assert!((per_word[r] - expected).abs() <= 1e-9 * expected.abs().max(1.0));
    }
}

#[test]
fn the_site_switch_attack_covers_every_subset_of_a_layer_and_searches_past_it() {
    let (model, passages, explanation) = fitted("switch", 2, "blocks.");
    assert_eq!(explanation.sites.len(), 12);
    // One layer: every subset of its six sites.
    let layer = Explanation { observations: explanation.observations, sites: explanation.sites.iter().filter(|f| f.site.name.starts_with("blocks.0.")).cloned().collect() };
    let attack = site_switch(&model, &layer, &passages, 0, 1).expect("exhaustive");
    assert!(attack.exhaustive);
    assert_eq!(attack.forwards, 63 * passages.len());
    let all: Vec<usize> = (0..6).collect();
    let together = replaced(&model, &layer, &all, &passages).expect("replaced");
    for p in 0..passages.len() {
        assert!(attack.worst[p] >= attack.all_replaced[p] && attack.alone.iter().all(|a| attack.worst[p] >= a[p]));
        assert!((attack.all_replaced[p] - together[p].0.mean().expect("rows")).abs() <= 1e-12 * attack.all_replaced[p].abs().max(1.0));
        assert!(together[p].0.iter().zip(&attack.word_worst[p]).all(|(k, w)| w >= k));
        assert!(!attack.worst_subset[p].is_empty());
    }
    // Both layers: past the exhaustive limit, the search's lower bound is at least its starts.
    let searched = site_switch(&model, &explanation, &passages, 8, 3).expect("search");
    assert!(!searched.exhaustive);
    for p in 0..passages.len() {
        assert!(searched.worst[p] >= searched.all_replaced[p] && searched.alone.iter().all(|a| searched.worst[p] >= a[p]));
    }
}
