//! Each MLP-output subcomponent's activation as a small function of the MLP-input subcomponents'
//! amplitudes at the same position, priced in the one total and checked by reinsertion (#2951,
//! `gam_mpd::gates::best_for` on `Targets::Values`).
//!
//! `mpd_mlp_functions_2951 EXPORT_DIR LIBRARY_DIR OUT.json OBSERVATIONS TRAIN EVAL [LAYERS] [DRAWS] [CONTEXT]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`): its first
//! `TRAIN` sequences of `CONTEXT` positions (default 512) fit the functions and the next `EVAL`
//! score them. `LIBRARY_DIR` holds each site's library (`{site}.v.f64`, `{site}.u.f64`, raw float64,
//! as `mpd_site_fit_2951` writes them). `LAYERS` (comma-separated, default every one) picks the
//! blocks whose `blocks.L.c_fc` (MLP input) and `blocks.L.down_proj` (MLP output) sites both have a
//! library.
//!
//! An output subcomponent `j` reads `a_j = v_j · GELU(W_in x)`, and `W_in x = Σ_c u_c b_c` sums
//! the input subcomponents' contributions, so `a_j` is exactly a function of their amplitudes
//! `b_c = v_c · x`. Its function here is the one total's choice among `β + ℓᵀb_S + Σ c GELU(wᵀb_S
//! + d)` over subsets `S` of the input amplitudes (each subset costing `log₂ C(C_in, |S|)`): grown
//! one amplitude at a time, the one whose linear term the residual favours most (the score
//! `(Σ_t r_t b_tc)² / Σ_t b_tc²`), kept while the function's bits plus its error fall. The error of
//! a predicted amplitude costs what the one total charges it at the written side, `n/2 ‖u_j‖²_F`
//! nats per unit² (`F` the output site's mean written Fisher on the training inputs,
//! `n = OBSERVATIONS`).
//!
//! On the eval inputs, `OUT.json` gets per layer: the functions' sizes (features, units) and own
//! bits, their error bits per input beside the constant functions', and the end-to-end check: the
//! output site replaced by its library alone (`Masked`, every mask 1) and by the library with every
//! amplitude its function's prediction (each mask `â_tj / a_tj`), the rest of the model native,
//! with the mean KL(model ‖ program) per token of each. `OUT.functions.json` holds the functions.

use gam_mpd::gates::{self, Feature, Switch, Targets};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Masked, Target, matrix, read_values, score_only, sites};
use gam_mpd::site_fit::samples;
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde_json::json;
use statrs::function::gamma::ln_gamma;
use std::path::PathBuf;

fn read_library(dir: &std::path::Path, name: &str, d_in: usize, d_out: usize) -> Result<Library, String> {
    let read = |side: &str, cols: usize| -> Result<Array2<f64>, String> {
        let path = dir.join(format!("{name}.{side}.f64"));
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
        Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
    };
    Ok(Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) })
}

fn log2_binomial(n: usize, k: usize) -> f64 {
    (ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)) / std::f64::consts::LN_2
}

/// Output subcomponent `j`'s function of the input amplitudes `b` (inputs × C_in), grown one
/// amplitude at a time while the total falls (module note).
fn grow(b: &Array2<f64>, norms: &Array1<f64>, target: &[f64], weight: f64, layer: usize, j: usize) -> Switch {
    let y = Targets::Values { values: target, weight };
    let pool = b.ncols();
    let mut chosen: Vec<usize> = Vec::new();
    let mut current = gates::constant(y);
    loop {
        let x = b.select(Axis(1), &chosen);
        let residual: Array1<f64> = (0..b.nrows()).map(|t| target[t] - current.logit(&x.row(t).to_vec())).collect();
        let score = b.t().dot(&residual);
        let Some(next) = (0..pool)
            .filter(|c| !chosen.contains(c) && norms[*c] > 0.0)
            .max_by(|p, q| (score[*p] * score[*p] / norms[*p]).total_cmp(&(score[*q] * score[*q] / norms[*q])))
        else {
            return current;
        };
        let mut trial_set = chosen.clone();
        trial_set.push(next);
        let features: Vec<Feature> = trial_set.iter().map(|c| Feature { site: layer, piece: *c, lag: 0, magnitude: false }).collect();
        let start = (!chosen.is_empty()).then_some(&current);
        let trial = gates::fit_to(b.select(Axis(1), &trial_set).view(), y, &features, log2_binomial(pool, trial_set.len()), start);
        if trial.total_bits() >= current.total_bits() {
            return current;
        }
        log::debug!("output {j}: {} amplitudes, {:.1} bits", trial_set.len(), trial.total_bits());
        (chosen, current) = (trial_set, trial);
    }
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_mlp_functions_2951 EXPORT_DIR LIBRARY_DIR OUT.json OBSERVATIONS TRAIN EVAL [LAYERS] [DRAWS] [CONTEXT]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let library_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let train: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let eval: usize = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("EVAL: {e}"))?;
    let wanted: Option<Vec<String>> = args.get(7).filter(|s| *s != "all").map(|s| s.split(',').map(str::to_string).collect());
    let draws: usize = args.get(8).map_or(Ok(2), |v| v.parse()).map_err(|e| format!("DRAWS: {e}"))?;
    let context: usize = args.get(9).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let all = sites(model);
    let has = |name: &str| library_dir.join(format!("{name}.v.f64")).exists();
    // The blocks with both MLP sites given.
    let layers: Vec<(String, usize, usize)> = all
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let layer = s.name.strip_prefix("blocks.")?.strip_suffix(".c_fc")?.to_string();
            let j = all.iter().position(|o| o.name == format!("blocks.{layer}.down_proj"))?;
            (wanted.as_ref().is_none_or(|w| w.contains(&layer)) && has(&s.name) && has(&all[j].name)).then_some((layer, i, j))
        })
        .collect();
    if layers.is_empty() {
        return Err(format!("no block of {wanted:?} has both MLP libraries in {}", library_dir.display()));
    }
    let chosen: Vec<_> = layers.iter().flat_map(|(_, i, j)| [all[*i].clone(), all[*j].clone()]).collect();
    let started = std::time::Instant::now();
    let sequence = |s: usize| family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>());
    let fitted_on = samples(model, &chosen, (0..train).map(sequence), draws, 0x317E)?;
    let scored_on = samples(model, &chosen, (train..train + eval).map(sequence), 1, 0x5C0E)?;
    eprintln!("samples of {} MLPs on {train} + {eval} sequences, {:.0}s", layers.len(), started.elapsed().as_secs_f64());
    let mut report = Vec::new();
    let mut functions = Vec::new();
    for (k, (layer, _, _)) in layers.iter().enumerate() {
        let (input_site, output_site) = (&chosen[2 * k], &chosen[2 * k + 1]);
        let (w_in, w_out) = (matrix(model, input_site)?, matrix(model, output_site)?);
        let input = read_library(&library_dir, &input_site.name, w_in.ncols(), w_in.nrows())?;
        let output = read_library(&library_dir, &output_site.name, w_out.ncols(), w_out.nrows())?;
        let amplitudes = |reads: &Array2<f32>, library: &Library| reads.mapv(f64::from).dot(&library.v.t());
        let (b_train, a_train) = (amplitudes(&fitted_on[2 * k].reads, &input), amplitudes(&fitted_on[2 * k + 1].reads, &output));
        let (b_eval, a_eval) = (amplitudes(&scored_on[2 * k].reads, &input), amplitudes(&scored_on[2 * k + 1].reads, &output));
        // Each output subcomponent's error price, `n/2 ‖u_j‖²_F` nats per unit² of amplitude.
        let fisher = &fitted_on[2 * k + 1].fisher;
        let weights: Vec<f64> = output.u.outer_iter().map(|u| 0.5 * observations * u.dot(&fisher.dot(&u))).collect();
        let norms = b_train.map_axis(Axis(0), |c| c.dot(&c));
        let fitted: Vec<Switch> = (0..output.v.nrows())
            .into_par_iter()
            .map(|j| grow(&b_train, &norms, &a_train.column(j).to_vec(), weights[j], k, j))
            .collect();
        // Predictions on the eval inputs, their error bits, and the constants'.
        let rows = b_eval.nrows();
        let mut predicted = Array2::<f64>::zeros(a_eval.dim());
        let (mut error, mut constant_error) = (0.0, 0.0);
        for (j, f) in fitted.iter().enumerate() {
            let x = b_eval.select(Axis(1), &f.features.iter().map(|g| g.piece).collect::<Vec<_>>());
            let mean = a_train.column(j).mean().unwrap_or(0.0);
            for t in 0..rows {
                let p = f.logit(&x.row(t).to_vec());
                predicted[[t, j]] = p;
                let (r, r0) = (a_eval[[t, j]] - p, a_eval[[t, j]] - mean);
                error += weights[j] * r * r / std::f64::consts::LN_2;
                constant_error += weights[j] * r0 * r0 / std::f64::consts::LN_2;
            }
        }
        // Reinsertion: the output site as its library alone, then with every amplitude predicted.
        let masked = Masked::build(model, vec![output_site.clone()], vec![output.clone()])?;
        let (mut kl_library, mut kl_functions) = (0.0, 0.0);
        for e in 0..eval {
            let inputs = sequence(train + e);
            let native = model.execute(&inputs, false).map_err(|e| e.to_string())?;
            let target = Target::every_row(native.values[model.output].clone());
            let a = read_values(&native, output_site)?.dot(&output.v.t());
            let ones = Array2::<f64>::ones(a.dim());
            let ratio = Array2::from_shape_fn(a.dim(), |(t, j)| {
                let real = a[[t, j]];
                if real.abs() > f64::MIN_POSITIVE { predicted[[e * context + t, j]] / real } else { 1.0 }
            });
            kl_library += score_only(&masked, &masked.family(&inputs, &[ones]), &target)?.sum();
            kl_functions += score_only(&masked, &masked.family(&inputs, &[ratio]), &target)?.sum();
        }
        let tokens = (eval * context) as f64;
        let own: f64 = fitted.iter().map(|f| f.function_bits).sum();
        let mut sizes = std::collections::BTreeMap::<String, usize>::new();
        for f in &fitted {
            *sizes.entry(format!("{} amplitudes, {} units", f.features.len(), f.units.len())).or_default() += 1;
        }
        eprintln!(
            "block {layer}: {} outputs {sizes:?}, own bits {own:.0}; eval error {:.1} bits per input (constants {:.1}); KL per token: library {:.4}, functions {:.4}; {:.0}s",
            fitted.len(),
            error / rows as f64,
            constant_error / rows as f64,
            kl_library / tokens,
            kl_functions / tokens,
            started.elapsed().as_secs_f64()
        );
        report.push(json!({
            "block": layer, "outputs": fitted.len(), "inputs": input.v.nrows(), "sizes": sizes, "function_bits": own,
            "eval": {"error_bits_per_input": error / rows as f64, "constant_error_bits_per_input": constant_error / rows as f64,
                "kl_library": kl_library / tokens, "kl_functions": kl_functions / tokens},
        }));
        functions.push(json!({"block": layer, "functions": fitted}));
        let written = json!({"observations": observations, "train": train, "eval": eval, "blocks": report});
        std::fs::write(&out, serde_json::to_string_pretty(&written).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        std::fs::write(out.with_extension("functions.json"), serde_json::to_string(&functions).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    Ok(())
}
