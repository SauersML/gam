//! CPU diagnostic of necessary scope floors for the existing fixed native episode family.
//! EXPORT SPEC.json OUT.json. No proposals, pruning, fitting or benchmark changes.
use gam_mpd::{
    counterfactual::{Action, Decoder, Spec, passages},
    import::import_language_model,
    missing_interface_bound::omitted_intervention_bound,
    run_check::{LanguageRun, split_sites},
};
use serde_json::json;
use std::{path::Path, time::Instant};
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        return Err("EXPORT SPEC.json OUT.json".into());
    }
    let out = Path::new(&a[3]);
    if out.exists() {
        return Err("report exists".into());
    }
    let start = Instant::now();
    let export = Path::new(&a[1]);
    let decoder = Decoder::from_export(export)?;
    let spec = Spec::load(Path::new(&a[2]), &decoder)?;
    let passages = passages(export, spec.rows)?;
    let imported = import_language_model(export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let run = LanguageRun::new(&decoder, &native, &spec, &passages, 8)?;
    let mut pairs = Vec::new();
    for (edited_index, edited) in spec
        .episodes
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.actions.is_empty())
    {
        let Some((clean_index, clean)) = spec
            .episodes
            .iter()
            .enumerate()
            .find(|(_, e)| e.actions.is_empty() && e.passage == edited.passage)
        else {
            return Err(format!("{} has no same-input clean teacher", edited.id));
        };
        let from = edited
            .actions
            .iter()
            .map(Action::first_row)
            .min()
            .ok_or("edited episode has no first row")?;
        let clean_group = spec
            .episodes
            .iter()
            .filter(|e| e.group == clean.group)
            .count();
        let edited_group = spec
            .episodes
            .iter()
            .filter(|e| e.group == edited.group)
            .count();
        let p = run.native_episode_log_probs(clean_index, from..spec.rows)?;
        let r = run.native_episode_log_probs(edited_index, from..spec.rows)?;
        let bound = omitted_intervention_bound(&p, &r, spec.rows, clean_group, edited_group)?;
        pairs.push(json!({"clean":clean.id,"edited":edited.id,"clean_group":clean.group,"edited_group":edited.group,"passage":edited.passage,"clean_score_rows":[0,spec.rows],"matched_suffix_rows":[from,spec.rows],"bound":bound,"applicability":"only explanations that predict identical output Q on this same input for clean and edited episodes (all effective edited interfaces omitted); no pruning"}));
    }
    let report = json!({"source":imported.record["source"],"spec":a[2],"pairs":pairs,"seconds":start.elapsed().as_secs_f64(),"native_teacher_timing":run.timing(),"scope":"exact JS theorem on fixed teacher distributions; numerical interval conditional on existing KL comparison assumptions, not a libm/full-network certificate"});
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", out.display());
    Ok(())
}
