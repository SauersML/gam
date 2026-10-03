//! Diagnose how fixed per-token matrix replacements compose on real model executions.
//!
//! `mpd_composition_2951 EXPORT LIBRARY SETS OUT.json [SEQUENCES] [CONTEXT] [joint|isolated] [SWITCHES.json] [BIASES]`
//!
//! Reads the standard rank-one library and CSR sets, without fitting or selecting anything.
//! For each site the signed identity separates native propagation, clean-input omission,
//! omission acting on the incoming error, and all-on reconstruction error. These are measured
//! values, not certificates. Norms use each site's native coordinates and are not comparable
//! across different sites. `isolated` additionally measures each replacement with every other
//! site kept native. The full model, all-on library and joint replacements share the same text.
//! Optional learned switches are compared on clean-model features and their own execution's
//! features; the returned autonomous masks are replayed to check execution consistency.
//! Optional comma-separated BIASES evaluates fixed shared logit offsets, exposing the fidelity
//! versus activity tradeoff. This is a diagnostic curve, not a fitted or held-out optimum.

use gam_mpd::composition::linear_site;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Masked, Target, kl_score_only, matrix, previous_inputs, read_values, score_only, sites};
use gam_linalg::faer_ndarray::fast_abt;
use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn read_raw<T>(path: &Path, decode: impl Fn([u8; 8]) -> T) -> Result<Vec<T>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{}: not a whole number of 8-byte values", path.display()));
    }
    Ok(bytes.chunks_exact(8).map(|c| decode([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

fn summary(values: &[f64]) -> Value {
    json!({"mean": values.iter().sum::<f64>() / values.len() as f64,
           "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max), "rows": values})
}

fn row_norms(values: &Array2<f64>) -> Vec<f64> {
    values.outer_iter().map(|row| row.iter().fold(0.0_f64, |s, x| s.hypot(*x))).collect()
}

fn frobenius(values: &Array2<f64>) -> f64 {
    values.iter().fold(0.0_f64, |s, x| s.hypot(*x))
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| numerator / denominator)
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_composition_2951 EXPORT LIBRARY SETS OUT.json [SEQUENCES] [CONTEXT] [joint|isolated] [SWITCHES.json] [BIASES]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let library_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let sets_dir = PathBuf::from(args.get(3).ok_or(usage)?);
    let out = PathBuf::from(args.get(4).ok_or(usage)?);
    let sequences: usize = args.get(5).map_or(Ok(1), |s| s.parse()).map_err(|e| format!("SEQUENCES: {e}"))?;
    let context: usize = args.get(6).map_or(Ok(32), |s| s.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let isolated = match args.get(7).map_or("joint", String::as_str) {
        "joint" => false,
        "isolated" => true,
        other => return Err(format!("mode {other}: expected joint or isolated")),
    };
    if sequences == 0 || context == 0 { return Err("positive sequence and context counts required".into()); }
    let started = std::time::Instant::now();
    let imported = import_language_model(&export, sequences, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let length = imported.record["config"]["n_ctx"].as_u64().ok_or("config.n_ctx missing")? as usize;
    let listed = std::fs::read_to_string(sets_dir.join("sites.txt")).map_err(|e| e.to_string())?;
    let named: Vec<(String, usize)> = listed.lines().map(|line| {
        let (name, count) = line.rsplit_once(' ').ok_or("sites.txt: name and component count required")?;
        let count = count.parse::<usize>().map_err(|e| e.to_string())?;
        if count == 0 { return Err("empty site in sites.txt".to_string()); }
        Ok((name.to_string(), count))
    }).collect::<Result<_, String>>()?;
    if named.is_empty() { return Err("no sites listed".into()); }
    let all = sites(model);
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    for (name, count) in &named {
        if chosen.iter().any(|s: &gam_mpd::masked::Site| s.name == *name) {
            return Err(format!("duplicate site {name}"));
        }
        let site = all.iter().find(|s| s.name == *name).ok_or_else(|| format!("{name}: not a model site"))?;
        let w = matrix(model, site)?;
        let v = read_raw(&library_dir.join(format!("{name}.v.f64")), f64::from_le_bytes)?;
        let u = read_raw(&library_dir.join(format!("{name}.u.f64")), f64::from_le_bytes)?;
        libraries.push(Library {
            v: Array2::from_shape_vec((*count, w.ncols()), v).map_err(|e| format!("{name} V: {e}"))?,
            u: Array2::from_shape_vec((*count, w.nrows()), u).map_err(|e| format!("{name} U: {e}"))?,
            mean: Array1::zeros(w.ncols()),
        });
        chosen.push(site.clone());
    }
    let offsets: Vec<usize> = std::iter::once(0).chain(named.iter().scan(0, |total, (_, count)| {
        *total += count;
        Some(*total)
    })).collect();
    let ptr = read_raw(&sets_dir.join("indptr.i64"), i64::from_le_bytes)?;
    let indices = read_raw(&sets_dir.join("indices.i64"), i64::from_le_bytes)?;
    if ptr.first() != Some(&0) || ptr.last() != Some(&(indices.len() as i64))
        || ptr.windows(2).any(|p| p[0] < 0 || p[0] > p[1])
        || ptr.len() < sequences * length + 1
        || indices.iter().any(|i| *i < 0 || *i >= offsets[named.len()] as i64) {
        return Err("sets are not valid CSR over the export's full-length sequences".into());
    }
    let mut masks: Vec<Array2<f64>> = named.iter().map(|(_, count)| Array2::zeros((family.rows, *count))).collect();
    for s in 0..sequences {
        for t in 0..context {
            let row = s * length + t;
            for &index in &indices[ptr[row] as usize..ptr[row + 1] as usize] {
                let index = index as usize;
                let k = offsets.partition_point(|o| *o <= index) - 1;
                let entry = &mut masks[k][[s * context + t, index - offsets[k]]];
                if *entry != 0.0 { return Err(format!("duplicate component {index} in sets row {row}")); }
                *entry = 1.0;
            }
        }
    }
    let clock = std::time::Instant::now();
    let clean = model.execute(family, false).map_err(|e| e.to_string())?;
    let native_seconds = clock.elapsed().as_secs_f64();
    let target = Target::every_row(clean.values[model.output].clone());
    let clock = std::time::Instant::now();
    let masked = Masked::build(model, chosen.clone(), libraries)?;
    let build_seconds = clock.elapsed().as_secs_f64();
    let clock = std::time::Instant::now();
    let candidate = masked.program.execute(&masked.family(family, &masks), false).map_err(|e| e.to_string())?;
    let fixed_mask_seconds = clock.elapsed().as_secs_f64();
    let joint_kl = kl_score_only(&target, &candidate.values[masked.program.output]).to_vec();
    let all_on: Vec<Array2<f64>> = masks.iter().map(|m| Array2::ones(m.dim())).collect();
    let all_on_kl = score_only(&masked, &masked.family(family, &all_on), &target)?.to_vec();
    // Standard input format is rank one. This is retained factor size, not the dense
    // executor's actual work, and does not assign a quantized description length.
    let factor_sizes = chosen.iter().map(|site| {
        let w = matrix(model, site)?;
        Ok(w.nrows() + w.ncols())
    }).collect::<Result<Vec<_>, String>>()?;
    let retained_entries = |selected: &[Array2<f64>]| selected.iter().zip(&factor_sizes)
        .map(|(m, size)| m.sum() * *size as f64).sum::<f64>() / family.rows as f64;
    let switch_execution = if let Some(path) = args.get(8) {
        #[derive(serde::Deserialize)]
        struct SiteSwitches { name: String, switches: Vec<gam_mpd::gates::Switch> }
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let records: Vec<SiteSwitches> = serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))?;
        if records.len() != chosen.len() || records.iter().zip(&chosen).any(|(r, s)| r.name != s.name) {
            return Err("switch site order differs from the library".into());
        }
        let switches: Vec<_> = records.into_iter().map(|r| r.switches).collect();
        // Validate and execute before the older masks helper, whose inputs are assumed valid.
        let clock = std::time::Instant::now();
        let (autonomous, autonomous_masks) = gam_mpd::switched::execute(&masked, family, &switches)?;
        let seconds = clock.elapsed().as_secs_f64();
        let autonomous_kl = kl_score_only(&target, &autonomous.values[masked.program.output]).to_vec();
        let amplitudes = chosen.iter().enumerate().map(|(k, site)| {
            Ok(fast_abt(&read_values(&clean, site)?, &masked.library(k)?.v))
        }).collect::<Result<Vec<_>, String>>()?;
        let teacher_masks = gam_mpd::gates::masks(&switches, &amplitudes, &previous_inputs(family));
        let teacher_kl = score_only(&masked, &masked.family(family, &teacher_masks), &target)?.to_vec();
        let replay = masked.program.execute(&masked.family(family, &autonomous_masks), false).map_err(|e| e.to_string())?;
        let replay_max_abs = autonomous.values.iter().zip(&replay.values).flat_map(|(a, b)| a.iter().zip(b.iter()))
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let changes: Vec<_> = chosen.iter().enumerate().map(|(k, site)| json!({
            "site": site.name,
            "changed_decisions": autonomous_masks[k].iter().zip(&teacher_masks[k]).filter(|(a, b)| a != b).count(),
            "decisions": autonomous_masks[k].len(),
            "autonomous_active_per_token": autonomous_masks[k].sum() / family.rows as f64,
            "teacher_active_per_token": teacher_masks[k].sum() / family.rows as f64
        })).collect();
        let clock = std::time::Instant::now();
        let compiled = gam_mpd::switched_compile::compile(model, chosen.clone(),
            (0..chosen.len()).map(|k| masked.library(k)).collect::<Result<Vec<_>, _>>()?,
            (0..chosen.len()).map(|k| masked.ranks(k).to_vec()).collect(), switches.clone())?;
        let compile_seconds = clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let (compiled_trace, compiled_masks) = gam_mpd::switched::execute(&compiled.masked, family, &compiled.switches)?;
        let compiled_seconds = clock.elapsed().as_secs_f64();
        let compiled_logits = &compiled_trace.values[compiled.masked.program.output];
        let compiled_kl = kl_score_only(&target, compiled_logits);
        let compiled_logit_difference = compiled_logits.iter().zip(&autonomous.values[masked.program.output])
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let compiled_decision_differences: usize = compiled_masks.iter().enumerate().map(|(k, m)| {
            m.indexed_iter().filter(|((r, c), value)| **value != autonomous_masks[k][[*r, compiled.original_blocks[k][*c]]]).count()
        }).sum();
        let compiled_execution = json!({"compile_seconds": compile_seconds, "forward_seconds": compiled_seconds,
            "kl": summary(&compiled_kl.to_vec()), "max_logit_difference": compiled_logit_difference,
            "changed_retained_decisions": compiled_decision_differences,
            "reads_per_token": compiled.counts.iter().map(|c| c.kept_reads).sum::<usize>(),
            "original_blocks": compiled.original_blocks,
            "scope": "fixed-policy dead-code elimination; algebraic equivalence, floating execution measured here; no all-on reconstruction claim"});
        drop(compiled_trace);
        drop(compiled);
        let mut bias_curve = Vec::new();
        if let Some(spec) = args.get(9) {
            for item in spec.split(',') {
                let bias: f64 = item.parse().map_err(|e| format!("switch bias {item}: {e}"))?;
                if !bias.is_finite() { return Err("finite switch biases required".into()); }
                let mut adjusted = switches.clone();
                for switch in adjusted.iter_mut().flatten() { switch.beta += bias; }
                let clock = std::time::Instant::now();
                let (trace, selected) = gam_mpd::switched::execute(&masked, family, &adjusted)?;
                let seconds = clock.elapsed().as_secs_f64();
                let kl = kl_score_only(&target, &trace.values[masked.program.output]);
                let active = selected.iter().map(|m| m.sum()).sum::<f64>() / family.rows as f64;
                eprintln!("switch bias {bias}: KL {:.6}, active {active:.2}, {seconds:.3}s", kl.mean().unwrap_or(0.0));
                bias_curve.push(json!({"logit_bias": bias, "kl": summary(&kl.to_vec()),
                                       "active_per_token": active, "seconds": seconds,
                                       "retained_factor_entries_per_token": retained_entries(&selected)}));
            }
        }
        Some(json!({"switches": path, "autonomous_seconds": seconds, "autonomous_kl": summary(&autonomous_kl),
                    "teacher_features_kl": summary(&teacher_kl), "fixed_mask_replay_max_abs_all_nodes": replay_max_abs,
                    "amplitudes_computed_per_token": masked.z.iter().enumerate().map(|(k, _)| masked.pieces(k)).sum::<usize>(),
                    "execution_cost_note": "dense library reads include off components; active counts omit this work",
                    "retained_factor_entries_per_token": {"autonomous": retained_entries(&autonomous_masks),
                        "teacher_features": retained_entries(&teacher_masks)},
                    "bias_curve": bias_curve,
                    "compiled_execution": compiled_execution,
                    "bias_curve_scope": "fixed policy variants on these inputs; no held-out optimum or description-length comparison claimed",
                    "sites": changes}))
    } else { None };
    let mut records = Vec::new();
    for (k, site) in chosen.iter().enumerate() {
        let w = matrix(model, site)?;
        let library = masked.library(k)?;
        let original = read_values(&clean, site)?;
        let actual = read_values(&candidate, &masked.sites[k])?;
        let error = linear_site(w.view(), library.v.view(), library.u.view(), original.view(), actual.view(), masks[k].view())?;
        let native_output = fast_abt(&original, &w);
        let incoming = frobenius(&error.incoming_omission);
        let clean_error = frobenius(&error.clean_omission);
        let reconstruction = frobenius(&error.reconstruction);
        let observed = frobenius(&error.observed);
        let closure = error.closure.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let isolation = if isolated {
            let one = Masked::build(model, vec![site.clone()], vec![library])?;
            Some(score_only(&one, &one.family(family, &[masks[k].clone()]), &target)?.to_vec())
        } else { None };
        eprintln!("{}: incoming/clean omission {:?}, reconstruction/observed {:?}, closure {closure:.3e}",
                  site.name, ratio(incoming, clean_error), ratio(reconstruction, observed));
        records.push(json!({
            "site": site.name, "active_per_token": masks[k].sum() / family.rows as f64,
            "incoming_over_clean_omission_frobenius": ratio(incoming, clean_error),
            "reconstruction_over_observed_frobenius": ratio(reconstruction, observed),
            "closure_max_abs": closure,
            "native_input_norm": summary(&row_norms(&original)),
            "native_output_norm": summary(&row_norms(&native_output)),
            "incoming_norm": summary(&row_norms(&error.incoming)),
            "native_propagation_norm": summary(&row_norms(&error.propagated)),
            "clean_omission_norm": summary(&row_norms(&error.clean_omission)),
            "incoming_omission_norm": summary(&row_norms(&error.incoming_omission)),
            "reconstruction_norm": summary(&row_norms(&error.reconstruction)),
            "observed_norm": summary(&row_norms(&error.observed)),
            "isolated_kl": isolation.as_ref().map(|v| summary(v))
        }));
    }
    let result = json!({
        "schema": 1, "evidence": "measured fixed-mask executions; not a uniform bound",
        "identity": "observed = native_propagation - clean_omission - incoming_omission + reconstruction",
        "norms": "Euclidean in native site coordinates; compare terms within a site only",
        "export": export, "library": library_dir, "sets": sets_dir, "source": imported.record["source"],
        "sequences": sequences, "context": context, "sets_context": length,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "timing": {"native_forward_seconds": native_seconds, "masked_build_seconds": build_seconds,
                   "fixed_mask_forward_seconds": fixed_mask_seconds, "scope": "single wall-clock observations, not a speedup benchmark"},
        "joint_kl": summary(&joint_kl), "all_on_kl": summary(&all_on_kl), "sites": records,
        "retained_factor_entries_per_token": retained_entries(&masks),
        "switch_execution": switch_execution
    });
    std::fs::write(&out, serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!("joint KL {:.6}, all-on KL {:.6}, {:.1}s; {}", joint_kl.iter().sum::<f64>() / family.rows as f64,
              all_on_kl.iter().sum::<f64>() / family.rows as f64, started.elapsed().as_secs_f64(), out.display());
    Ok(())
}
