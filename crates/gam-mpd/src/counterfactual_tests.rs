use super::counterfactual::{
    Action, Decoder, Donor, Episode, Explanation as Explained, FittedRule, InputChange, Library, Maps, OutputChange, Program, Rows, Selection, Selector, Spec, evaluate,
    fitted_libraries, fitted_sites, rule_price, score, site_index, site_name,
};
use super::explanation::{Explanation, Fitted, Replacement, in_execution_order};
use super::import::import_language_model;
use super::masked::{self, matrix, sites};
use ndarray::{Array1, Array2, Axis, s};
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
    // Two passages of 10 tokens (and their next tokens), for the imported program.
    write(&dir, "tokens", [2, 11], (0..22).map(|t| ((t * 7 + t / 11) % vocab) as f64).collect());
    let record = serde_json::json!({
        "config": {"d_model": d, "n_layers": 2, "n_heads": 2, "n_kv_heads": 2, "head_dim": 4, "d_mlp": mlp, "vocab": vocab, "rope_theta": 10000.0,
                   "rope_pairing": "rotate_half", "norm_eps": 1e-6, "mlp_act": "gelu_tanh"},
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).expect("export.json");
    dir
}

/// Each site's exact library of input-coordinate units: `v_c = e_c`, `u_c = W e_c`.
fn coordinate_libraries(decoder: &Decoder) -> Vec<Option<Library>> {
    (0..decoder.sites())
        .map(|site| {
            let w = decoder.native(site);
            Some(Library { v: Array2::eye(w.ncols()), u: w.t().to_owned() })
        })
        .collect()
}

fn all_on(libraries: &[Option<Library>], rows: usize) -> Selection {
    let row: Vec<(u32, u32)> =
        libraries.iter().enumerate().flat_map(|(k, l)| (0..l.as_ref().map_or(0, |l| l.v.nrows())).map(move |c| (k as u32, c as u32))).collect();
    Selection { rows: vec![row; rows] }
}

fn max_gap(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// The donor states `actions` read, from one program on `tokens`.
fn donor(decoder: &Decoder, maps: Maps<'_>, tokens: &[u32], actions: &[Action]) -> Donor {
    let record = actions.iter().filter_map(Action::donor_state).map(|k| (k, None)).collect();
    let mut program = Program::new(maps, &[], None);
    program.record = record;
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
    let mut clean_native = Program::new(Maps::Native(&decoder), &[], None);
    let clean = decoder.forward(&tokens, &mut clean_native, &rows);
    for actions in &cases {
        let native_donor = donor(&decoder, Maps::Native(&decoder), &donor_tokens, actions);
        let mut donor_selection = all_on(&libraries, donor_tokens.len());
        let own_donor = donor(&decoder, Maps::Units { decoder: &decoder, libraries: &libraries, selector: &mut donor_selection }, &donor_tokens, actions);
        let interface = [5usize, 7];
        let mut native = Program::new(Maps::Native(&decoder), actions, Some(&native_donor));
        let a = decoder.forward(&tokens, &mut native, &interface);
        let mut selection = all_on(&libraries, tokens.len());
        let selector: &mut dyn Selector = &mut selection;
        let mut explained = Program::new(Maps::Units { decoder: &decoder, libraries: &libraries, selector }, actions, Some(&own_donor));
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
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}

#[test]
fn a_partial_explanation_is_scored_by_its_disagreement() {
    let dir = tiny_export("partial");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let libraries = coordinate_libraries(&decoder);
    let tokens: Vec<u32> = (0..6).map(|t| (t * 5 % 11) as u32).collect();
    let rows: Vec<usize> = (0..tokens.len()).collect();
    let mut clean_native = Program::new(Maps::Native(&decoder), &[], None);
    let clean = decoder.forward(&tokens, &mut clean_native, &rows);
    // Half of every site's units at every row.
    let mut selection = all_on(&libraries, tokens.len());
    for row in &mut selection.rows {
        row.retain(|(_, c)| c % 2 == 0);
    }
    let selector: &mut dyn Selector = &mut selection;
    let mut explained = Program::new(Maps::Units { decoder: &decoder, libraries: &libraries, selector }, &[], None);
    let b = decoder.forward(&tokens, &mut explained, &[2]);
    let mut native = Program::new(Maps::Native(&decoder), &[], None);
    let a = decoder.forward(&tokens, &mut native, &[2]);
    let scores = score(&decoder, &a, &b, &clean, &[2], 0, 4);
    assert!(scores.kl > 1e-6 && scores.native_effect.abs() < 1e-12, "{scores:?}");
    assert_eq!(clean.layers[0].len_of(Axis(0)), tokens.len());
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}

#[test]
fn a_site_without_a_library_runs_its_native_map() {
    let dir = tiny_export("scope");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let mut libraries = coordinate_libraries(&decoder);
    // Only block 0's sites are explained; block 1 runs native.
    for site in 6..decoder.sites() {
        libraries[site] = None;
    }
    let tokens: Vec<u32> = (0..7).map(|t| (t * 4 % 11) as u32).collect();
    let mut selection = all_on(&libraries, tokens.len());
    let selector: &mut dyn Selector = &mut selection;
    let mut explained = Program::new(Maps::Units { decoder: &decoder, libraries: &libraries, selector }, &[], None);
    let b = decoder.forward(&tokens, &mut explained, &[]);
    let mut native = Program::new(Maps::Native(&decoder), &[], None);
    let a = decoder.forward(&tokens, &mut native, &[]);
    assert!(max_gap(&a.residual, &b.residual) < 1e-10);
    // Block 0's units: five maps reading 8 coordinates and the down-projection reading 16.
    assert_eq!(explained.selected, (5 * 8 + 16) * tokens.len());
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}

/// Every site of the imported tiny model fitted with its coordinate units, each its own block
/// priced at `bits` in a unit metric, in execution order.
fn coordinate_explanation(dir: &Path, bits: f64) -> (super::operator_program::OperatorProgram, super::operator_program::FamilyInputs, Explanation) {
    let imported = import_language_model(dir, 2, 10).expect("import");
    let model = imported.program;
    let observations = 50.0;
    let fitted = in_execution_order(sites(&model))
        .into_iter()
        .map(|site| {
            let w = matrix(&model, &site).expect("site map");
            let (d_out, d_in) = w.dim();
            let library = masked::Library { v: Array2::eye(d_in), u: w.t().to_owned(), mean: Array1::zeros(d_in) };
            // Block prices that differ, so some units are worth dropping and some are not.
            let prices = (0..d_in).map(|c| bits * (1.0 + (c % 3) as f64)).collect();
            Fitted::new(site, w, (library, vec![1; d_in], prices), (Array2::eye(d_out), Array2::eye(d_in)), observations).expect("fitted")
        })
        .collect();
    (model, imported.contract.family, Explanation { observations, sites: fitted })
}

/// Units selected per site and row by a counterfactual program, the rule's own record.
struct Seen<'a> {
    rule: FittedRule<'a>,
    chosen: Vec<Vec<Vec<u32>>>,
}

impl Selector for Seen<'_> {
    fn select(&mut self, site: usize, input: &Array2<f64>) -> Vec<Vec<(u32, f64)>> {
        let chosen = self.rule.select(site, input);
        self.chosen[site] = chosen.iter().map(|row| row.iter().map(|(c, _)| *c).collect()).collect();
        chosen
    }
}

#[test]
fn the_decoder_and_the_explanations_rule_agree_with_the_imported_program() {
    let dir = tiny_export("fitted");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let (model, family, explanation) = coordinate_explanation(&dir, 2.0);
    let members: Vec<usize> = (0..explanation.sites.len()).collect();
    let masked = explanation.masked(&model, &members).expect("masked");
    let by_site = fitted_sites(&decoder, &explanation.sites).expect("every site is the decoder's");
    let libraries = fitted_libraries(&by_site);
    for passage in 0..2 {
        let rows: Vec<usize> = (passage * 10..(passage + 1) * 10).collect();
        let base = family.select(&rows);
        let tokens: Vec<u32> = (0..10).map(|t| ((passage * 11 + t) * 7 + (passage * 11 + t) / 11) as u32 % 11).collect();
        // The native forwards.
        let logits = model.execute(&base, false).expect("native program").values.swap_remove(model.output);
        let mut native = Program::new(Maps::Native(&decoder), &[], None);
        let ours = decoder.log_probs(&decoder.forward(&tokens, &mut native, &[]).residual);
        let theirs = decoder_log_softmax(&logits);
        assert!(max_gap(&ours, &theirs) < 1e-9, "passage {passage}: native gap {}", max_gap(&ours, &theirs));
        // The explanation run by its own rule, in the imported program and in the decoder.
        let (trace, chosen) = explanation.execute(&masked, &members, &base).expect("explanation");
        let mut seen = Seen { rule: FittedRule { sites: &by_site }, chosen: vec![Vec::new(); decoder.sites()] };
        let mut program = Program::new(Maps::Units { decoder: &decoder, libraries: &libraries, selector: &mut seen }, &[], None);
        let explained = decoder.log_probs(&decoder.forward(&tokens, &mut program, &[]).residual);
        let theirs = decoder_log_softmax(&trace.values[masked.program.output]);
        assert!(max_gap(&explained, &theirs) < 1e-9, "passage {passage}: explained gap {}", max_gap(&explained, &theirs));
        let mut dropped = 0;
        for (k, f) in explanation.sites.iter().enumerate() {
            let site = (0..decoder.sites()).find(|j| site_name(*j) == f.site.name).expect("decoder site");
            let theirs: Vec<Vec<u32>> = chosen[k].outer_iter().map(|on| (0..on.len() as u32).filter(|c| on[*c as usize] == 1.0).collect()).collect();
            assert_eq!(seen.chosen[site], theirs, "passage {passage}, {}", f.site.name);
            dropped += theirs.iter().map(|r| f.ranks.len() - r.len()).sum::<usize>();
        }
        assert!(dropped > 0, "the rule dropped no unit, so the test compares nothing");
    }
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}

fn decoder_log_softmax(logits: &Array2<f64>) -> Array2<f64> {
    let mut out = logits.clone();
    for mut row in out.outer_iter_mut() {
        let max = row.fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        let total = row.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
        row.mapv_inplace(|v| v - total);
    }
    out
}

#[test]
fn the_rules_price_is_its_lowest_selection_keeping_precision() {
    let dir = tiny_export("rule");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let (_, _, explanation) = coordinate_explanation(&dir, 2.0);
    let by_site = fitted_sites(&decoder, &explanation.sites).expect("every site is the decoder's");
    let libraries = fitted_libraries(&by_site);
    let tokens: Vec<u32> = (0..10).map(|t| (t * 7 % 11) as u32).collect();
    struct Reads<'a>(FittedRule<'a>, Vec<Array2<f64>>);
    impl Selector for Reads<'_> {
        fn select(&mut self, site: usize, input: &Array2<f64>) -> Vec<Vec<(u32, f64)>> {
            self.1[site] = input.clone();
            self.0.select(site, input)
        }
    }
    let mut reads = Reads(FittedRule { sites: &by_site }, vec![Array2::zeros((0, 0)); decoder.sites()]);
    let mut program = Program::new(Maps::Units { decoder: &decoder, libraries: &libraries, selector: &mut reads }, &[], None);
    decoder.forward(&tokens, &mut program, &[]);
    let chosen: Vec<&Fitted> = by_site.iter().flatten().copied().collect();
    let tables: Vec<Array2<f64>> = (0..decoder.sites()).map(|k| reads.1[k].clone()).collect();
    let price = rule_price(&chosen, &tables, explanation.observations).expect("price");
    let reals: usize = chosen.iter().map(|f| f.w.len() + f.bits.len() + f.fisher.nrows() * (f.fisher.nrows() + 1) / 2).sum();
    assert_eq!(price.reals, reals);
    assert!(price.bits > price.reals as f64, "{price:?}");
    // At the price's precision every site selects as it did; one bit coarser, some site does not.
    let rounded = |m: &Array2<f64>, p: i32| m.mapv(|w| (w * 2f64.powi(p)).round() / 2f64.powi(p));
    let same = |p: i32| {
        chosen.iter().zip(&tables).all(|(f, x)| {
            let bits: Vec<f64> = f.bits.iter().map(|b| (b * 2f64.powi(p)).round() / 2f64.powi(p)).collect();
            let rule = super::site_fit::Selector::new(&rounded(&f.w, p), &rounded(&f.fisher, p), &f.library, &f.ranks, &bits, explanation.observations).expect("rule");
            rule.select(x) == f.select(x)
        })
    };
    assert!(same(price.precision));
    assert!(price.precision == 0 || !same(price.precision - 1), "{price:?}");
    assert_eq!(price.binding.is_some(), price.precision > 0, "{price:?}");
    assert_eq!(reads.1[0].slice(s![.., 0]).len(), tokens.len());
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}

/// One selection for every episode.
fn every_episode<'a>(selection: Selection) -> impl Fn(&str) -> Result<Box<dyn Selector + 'a>, String> + Sync + 'a {
    move |_: &str| Ok(Box::new(selection.clone()))
}

#[test]
fn every_episode_of_an_exact_explanation_scores_zero_and_the_native_effect_is_measured() {
    let dir = tiny_export("evaluate");
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let libraries = coordinate_libraries(&decoder);
    let rows = 9;
    let passages: Vec<Vec<u32>> = (0..3).map(|p| (0..rows).map(|t| ((t * (p + 3) + p) % 11) as u32).collect()).collect();
    let episode = |id: &str, passage: usize, donor: Option<usize>, actions: Vec<Action>| Episode {
        id: id.to_string(),
        group: id.split('/').next().unwrap_or(id).to_string(),
        passage,
        donor,
        interface_rows: vec![4, 7],
        actions,
    };
    let spec = Spec {
        rows,
        episodes: vec![
            episode("clean/0", 0, None, vec![]),
            episode("clean/1", 1, None, vec![]),
            episode("clean/2", 2, None, vec![]),
            // Two episodes mixing toward one donor at different sites: the donor runs once with both.
            episode("mix/0", 0, Some(1), vec![Action::Input { site: site_index(0, 2), change: InputChange::Mix { row: 4, alpha: 1.0 } }]),
            episode("mix/2", 2, Some(1), vec![Action::Output { site: site_index(1, 4), change: OutputChange::Mix { row: 5, alpha: 0.5 } }]),
            episode("scale/1", 1, None, vec![Action::Input { site: site_index(1, 5), change: InputChange::Scale { rows: Rows::All, cols: (2, 5), scale: 4.0 } }]),
        ],
    };
    let native = evaluate(&decoder, &spec, &passages, None, 4).expect("native");
    let all = all_on(&libraries, rows);
    let exact = evaluate(&decoder, &spec, &passages, Some(&Explained { libraries: &libraries, selector: Box::new(every_episode(all)) }), 4).expect("exact explanation");
    for (a, b) in native.iter().zip(&exact) {
        assert_eq!(a.id, b.id);
        for s in [&a.scores, &b.scores] {
            assert!(s.kl.abs() < 1e-12 && s.interface_kl.iter().all(|k| k.abs() < 1e-12), "{}: {s:?}", a.id);
        }
        assert!((a.scores.native_effect - b.scores.native_effect).abs() < 1e-15, "{}", a.id);
        assert_eq!(a.scores.native_effect > 1e-9, !a.id.starts_with("clean"), "{}: {:?}", a.id, a.scores);
    }
    assert_eq!(exact[3].from_row, 4);
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}
