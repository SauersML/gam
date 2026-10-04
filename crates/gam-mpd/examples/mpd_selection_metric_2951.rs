//! How well a site's selection charge predicts the KL of what it leaves (#2951).
//!
//! `mpd_selection_metric_2951 FRONTIER LIBRARY_DIR SITE OUT.json [PASSAGES] [DRAWS]`
//!
//! `LIBRARY_DIR` holds a site fitted by `mpd_e2e_2951` (`{SITE}.{v,u,fisher,moment}.f64`,
//! `{SITE}.json` with its blocks, bits and the `n` it was fitted at; rank-one blocks). On each of
//! the frontier export's first `PASSAGES` passages (default 4, 512 positions) with only `SITE`
//! replaced, the run-time selection (`gam_mpd::explanation::Fitted::select`) leaves the residual
//! `r_t = W x_t − Σ_{c on} u_c (v_c · x_t)`, priced at second order three ways against the KL it
//! really costs (`KL(model ‖ model with SITE's output replaced)`, every row):
//!
//! * `charged`: `½ Σ_t r_tᵀ F̄ r_t`, the site's code (`F̄` its mean written Fisher);
//! * `own_row`: `½ Σ_t r_tᵀ F_t r_t`, each row's own Fisher (`DRAWS` sampled-label gradients of the
//!   native output, default 16);
//! * `field`: `½ E (Σ_t g_t · r_t)²`, every pair of rows (the whole passage's second-order KL).
//!
//! It also selects every row's subcomponents by the same code under each row's own Fisher (an
//! oracle: those gradients need the model's backward pass) and under the mean one, both by
//! `gam_mpd::sparse_code::code_site`, and reports each selection's real KL and subcomponents per
//! row. All values are per row; `OUT.json` gets every passage and their means.

use gam_mpd::counterfactual::read_f64_matrix;
use gam_mpd::derivatives::vjp;
use gam_mpd::explanation::Fitted;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Masked, Target, kl_score_only, matrix, read_values, sampled_label_cotangent, sites};
use gam_mpd::sparse_code::{Metric, Problem, code_site};
use ndarray::{Array1, Array2, Axis};
use serde_json::{Value, json};
use std::path::PathBuf;

const CONTEXT: usize = 512;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_selection_metric_2951 FRONTIER LIBRARY_DIR SITE OUT.json [PASSAGES] [DRAWS]";
    if args.len() < 5 {
        return Err(usage.to_string());
    }
    let (frontier, dir, name, out) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]), &args[3], PathBuf::from(&args[4]));
    let count = |i: usize, default: usize| args.get(i).map_or(Ok(default), |v| v.parse::<usize>().map_err(|e| format!("{usage}: {e}")));
    let (passages, draws) = (count(5, 4)?, count(6, 16)?);
    let imported = import_language_model(&frontier, passages, CONTEXT)?;
    let model = &imported.program;
    let site = sites(model).into_iter().find(|s| &s.name == name).ok_or_else(|| format!("{name}: not a site"))?;
    let w = matrix(model, &site)?;
    let (d_out, d_in) = w.dim();
    let read = |side: &str, cols: usize| read_f64_matrix(&dir.join(format!("{name}.{side}.f64")), cols);
    let library = Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) };
    let record: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).map_err(|e| format!("{name}.json: {e}"))?)
        .map_err(|e| format!("{name}.json: {e}"))?;
    let ranks: Vec<usize> = serde_json::from_value(record["ranks"].clone()).map_err(|e| format!("{name} ranks: {e}"))?;
    let bits: Vec<f64> = serde_json::from_value(record["bits"].clone()).map_err(|e| format!("{name} bits: {e}"))?;
    let n = record["fitted_with"]["n"].as_f64().ok_or_else(|| format!("{name}.json: no fitted_with.n"))?;
    if ranks.iter().any(|r| *r != 1) {
        return Err(format!("{name}: blocks {ranks:?}; the oracle selection takes rank-one subcomponents"));
    }
    let fisher = read("fisher", d_out)?;
    let fitted = Fitted::new(site.clone(), w.clone(), (library.clone(), ranks.clone(), bits.clone()), (fisher.clone(), read("moment", d_in)?), n)?;
    let masked = Masked::build_blocks(model, vec![site.clone()], vec![library.clone()], vec![ranks.clone()])?;
    let no_rows = Target { logits: Array2::zeros((0, 0)), scored: None };
    let mut per_passage = Vec::new();
    for p in 0..passages {
        let base = imported.contract.family.select(&(p * CONTEXT..(p + 1) * CONTEXT).collect::<Vec<_>>());
        let trace = model.execute(&base, false).map_err(|e| e.to_string())?;
        let logits = &trace.values[model.output];
        let target = Target::every_row(logits.clone());
        let x = read_values(&trace, &site)?;
        let targets = x.dot(&w.t());
        let rows = x.nrows() as f64;
        // A selection's residual, its real KL and its subcomponents on, per row.
        let run = |on: &Array2<f64>| -> Result<(Array2<f64>, f64, f64), String> {
            let residual = &targets - &(x.dot(&library.v.t()) * on).dot(&library.u);
            let replaced = masked.program.execute(&masked.family(&base, std::slice::from_ref(on)), false).map_err(|e| e.to_string())?;
            let kl = kl_score_only(&target, &replaced.values[masked.program.output]).sum() / rows;
            Ok((residual, kl, on.sum() / rows))
        };
        let (residual, actual, on) = run(&fitted.select(&x))?;
        let charged = 0.5 * (residual.dot(&fisher) * &residual).sum() / rows;
        let (mut own_row, mut field) = (0.0, 0.0);
        let mut gradients = Vec::with_capacity(draws);
        for k in 0..draws {
            let cotangent = sampled_label_cotangent(logits, &no_rows, 0x5EED ^ ((p * draws + k) as u64).wrapping_mul(0x9E37_79B9));
            let back = vjp(model, &base, &trace, cotangent).map_err(|e| e.to_string())?;
            let parts: Vec<Array2<f64>> = site.writes.iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
            let views: Vec<_> = parts.iter().map(|g| g.view()).collect();
            let g = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
            let along = (&g * &residual).sum_axis(Axis(1));
            own_row += 0.5 * along.mapv(|a| a * a).sum() / (draws as f64 * rows);
            field += 0.5 * along.sum().powi(2) / (draws as f64 * rows);
            gradients.push(g);
        }
        let views: Vec<_> = gradients.iter().map(|g| g.view()).collect();
        let mut selections = serde_json::Map::new();
        for (label, metric) in [("own_row_oracle", Metric::PerRow(&views)), ("mean_relaxed", Metric::Mean(fisher.view()))] {
            let problem = Problem { reads: x.view(), targets: targets.view(), v: library.v.view(), u: library.u.view(), ranks: &ranks, bits: &bits, metric, observations: n, nodes: 0 };
            let coding = code_site(&problem, None)?;
            let mut chosen = Array2::<f64>::zeros((x.nrows(), ranks.len()));
            for (t, set) in coding.sets.iter().enumerate() {
                for c in set {
                    chosen[[t, *c as usize]] = 1.0;
                }
            }
            let (_, kl, on) = run(&chosen)?;
            selections.insert(label.to_string(), json!({"kl": kl, "on": on}));
        }
        let entry = json!({"charged": charged, "own_row": own_row, "field": field, "actual": actual, "on": on, "selections": selections});
        eprintln!("{name} passage {p}: {entry}");
        per_passage.push(entry);
    }
    let mean = |path: &[&str]| -> f64 {
        per_passage.iter().map(|e| path.iter().fold(e, |v, k| &v[*k]).as_f64().unwrap_or(f64::NAN)).sum::<f64>() / per_passage.len().max(1) as f64
    };
    let summary = json!({
        "site": name, "n": n, "passages": passages, "draws": draws,
        "mean": {"charged": mean(&["charged"]), "own_row": mean(&["own_row"]), "field": mean(&["field"]), "actual": mean(&["actual"]), "on": mean(&["on"]),
                 "own_row_oracle": {"kl": mean(&["selections", "own_row_oracle", "kl"]), "on": mean(&["selections", "own_row_oracle", "on"])},
                 "mean_relaxed": {"kl": mean(&["selections", "mean_relaxed", "kl"]), "on": mean(&["selections", "mean_relaxed", "on"])}},
        "per_passage": per_passage,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!("{}", summary["mean"]);
    Ok(())
}
