//! Per-input pieces of a language model trained through its own masked forward (#2951).
//!
//! `mpd_pieces_masked_2951 EXPORT_DIR PIECES_DIR OUT.json OBSERVATIONS SEQUENCES {fit|wsvd} [CONTEXT]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`);
//! `PIECES_DIR` holds `bench/mpd_pieces_2951.py dump`'s site statistics and, for `fit`, the
//! starting libraries of `mpd_pieces_2951` (VPD naming, `h.{l}.attn.q_proj` and so on); `wsvd`
//! starts from each site's exact Fisher-whitened singular pieces (`gam_mpd::pieces::fisher_svd`).
//! Each input starts with the pieces whose second-order KL bits alone, in the global Fisher, exceed
//! the uniform listing cost. The first `SEQUENCES` token rows are the fit inputs and the next `SEQUENCES` the eval
//! inputs, `CONTEXT` positions each (default 512). From no piece on, the fit alternates exact
//! selection and preconditioned steps of the pieces (`gam_mpd::masked`) at `OBSERVATIONS` per input,
//! until an iteration saves less than one bit per input; after each iteration the eval inputs are
//! selected with the fit's listing costs and the masked forward reports their mean active pieces
//! (L0), KL and bits per token in the per-token frontier's code (per site `ω(k + 1) + log₂ C(C, k)`).
//! Written to `OUT.json` after every iteration.

use gam_mpd::codec::prefix_integer_len_bits;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Masked, listing_bits, listing_costs, matrix, select, sites, step_pieces};
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2};
use serde_json::json;
use statrs::function::gamma::ln_gamma;
use std::path::{Path, PathBuf};

fn read_f64(path: &Path, rows: usize, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|e| e.to_string())
}

/// The VPD name of a site of the imported program (`blocks.{l}.q` → `h.{l}.attn.q_proj`).
fn vpd_name(site: &str) -> Option<String> {
    let rest = site.strip_prefix("blocks.")?;
    let (layer, kind) = rest.split_once('.')?;
    let full = match kind {
        "q" => "attn.q_proj",
        "k" => "attn.k_proj",
        "v" => "attn.v_proj",
        "o" => "attn.o_proj",
        "c_fc" => "mlp.c_fc",
        "down_proj" => "mlp.down_proj",
        _ => return None,
    };
    Some(format!("h.{layer}.{full}"))
}

/// Mean active pieces, KL and frontier bits per input.
fn report(masks: &[Array2<f64>], kl: &Array1<f64>) -> serde_json::Value {
    let rows = kl.len() as f64;
    let mut l0 = 0.0;
    let mut bits = 0.0;
    for m in masks {
        let c = m.ncols() as f64;
        for row in m.outer_iter() {
            let k = row.iter().filter(|x| **x > 0.0).count();
            l0 += k as f64;
            let omega = prefix_integer_len_bits(k as u64 + 1).map_or(0.0, |b| b as f64);
            let binomial = (ln_gamma(c + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma(c - k as f64 + 1.0)) / std::f64::consts::LN_2;
            bits += omega + binomial;
        }
    }
    json!({"l0": l0 / rows, "kl": kl.sum() / rows, "bits": bits / rows})
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_pieces_masked_2951 EXPORT_DIR PIECES_DIR OUT.json OBSERVATIONS SEQUENCES {fit|wsvd} [CONTEXT]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let pieces_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let sequences: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("SEQUENCES: {e}"))?;
    let start = args.get(6).ok_or(usage)?.clone();
    let context: usize = args.get(7).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let imported = import_language_model(&export, 2 * sequences, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let fit_rows: Vec<usize> = (0..sequences * context).collect();
    let eval_rows: Vec<usize> = (sequences * context..2 * sequences * context).collect();
    let (fit, eval) = (family.select(&fit_rows), family.select(&eval_rows));
    let target_of = |inputs: &gam_mpd::operator_program::FamilyInputs| -> Result<Array2<f64>, String> {
        Ok(model.execute(inputs, false).map_err(|e| e.to_string())?.values[model.output].clone())
    };
    let (fit_target, eval_target) = (target_of(&fit)?, target_of(&eval)?);
    let all = sites(model);
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    let mut weights: Vec<Array1<f64>> = Vec::new();
    for site in all {
        let Some(name) = vpd_name(&site.name) else { continue };
        let w = matrix(model, &site)?;
        let (d_out, d_in) = w.dim();
        let mean = Array1::from_vec(read_f64(&pieces_dir.join(format!("{name}.mu.f64")), 1, d_in)?.into_raw_vec_and_offset().0);
        let fisher = read_f64(&pieces_dir.join(format!("{name}.B.f64")), d_out, d_out)?;
        let (v, u) = match start.as_str() {
            "wsvd" => {
                let second_moment = read_f64(&pieces_dir.join(format!("{name}.A.f64")), d_in, d_in)?;
                let library = fisher_svd(&gam_mpd::pieces::Site { w: w.clone(), second_moment, mean: mean.clone(), fisher: fisher.clone() })?;
                (library.v.t().to_owned(), library.u)
            }
            "fit" => {
                let count = std::fs::read(pieces_dir.join(format!("{name}.V.f64"))).map_err(|e| format!("{name}: {e}"))?.len() / 8 / d_in;
                (read_f64(&pieces_dir.join(format!("{name}.V.f64")), d_in, count)?.t().to_owned(), read_f64(&pieces_dir.join(format!("{name}.U.f64")), count, d_out)?)
            }
            other => return Err(format!("unknown start {other}; {usage}")),
        };
        let pieces = v.nrows();
        let error = (&v.t().dot(&u).t() - &w).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let largest = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if error > 1e-6 * largest {
            return Err(format!("{name}: the starting library is not the site ({error:e} against {largest:e})"));
        }
        // Each piece's second-order weight `uᵀ B u` in the global Fisher.
        weights.push((&u.dot(&fisher) * &u).sum_axis(ndarray::Axis(1)));
        eprintln!("{} = {name}: {d_out}×{d_in}, {pieces} pieces", site.name);
        chosen.push(site);
        libraries.push(Library { v, u, mean });
    }
    let total_pieces: usize = libraries.iter().map(|l| l.v.nrows()).sum();
    let original_sites = chosen.clone();
    let mut masked = Masked::build(model, chosen, libraries)?;
    let samples = 2;
    // Every input starts with the pieces whose own second-order KL bits, `n a² uᵀBu / (2 ln 2)`
    // with `a = v · (x − μ)` on the model's clean forward, exceed the uniform listing cost.
    let uniform = (total_pieces as f64).log2();
    let start_masks = |inputs: &gam_mpd::operator_program::FamilyInputs, libraries: &[Library]| -> Result<Vec<Array2<f64>>, String> {
        let trace = model.execute(inputs, false).map_err(|e| e.to_string())?;
        let mut masks = Vec::new();
        for (k, library) in libraries.iter().enumerate() {
            let original = &original_sites[k];
            let x = gam_mpd::masked::read_values(&trace, original)? - &library.mean;
            let a = x.dot(&library.v.t());
            let scale = observations / (2.0 * std::f64::consts::LN_2);
            masks.push(Array2::from_shape_fn(a.dim(), |(r, c)| if scale * a[[r, c]] * a[[r, c]] * weights[k][c] > uniform { 1.0 } else { 0.0 }));
        }
        Ok(masks)
    };
    let mut fit_masks = start_masks(&fit, &masked.libraries)?;
    let mut eval_masks = start_masks(&eval, &masked.libraries)?;
    let mut costs = listing_costs(&fit_masks);
    let mut history = Vec::new();
    let mut previous = f64::INFINITY;
    for iteration in 0.. {
        let started = std::time::Instant::now();
        let (masks, selected_kl) = select(&masked, &fit, &fit_target, fit_masks, &costs, observations, samples)?;
        log::info!("selection: KL {:.4} per input, {}", selected_kl.sum() / fit.rows as f64, report(&masks, &selected_kl));
        fit_masks = masks;
        costs = listing_costs(&fit_masks);
        // One step of the pieces, then the selection again from where it stands: alternating
        // minimisation, stopped when an iteration saves less than a bit per input.
        let steps = usize::from(
            step_pieces(&mut masked, &fit, &fit_target, &fit_masks, samples, 0xF00D + iteration as u64)?
                .inspect(|(before, after)| log::info!("pieces step: KL {:.4} -> {:.4} per input", before / fit.rows as f64, after / fit.rows as f64))
                .is_some(),
        );
        let fit_kl = {
            let family = masked.family(&fit, &fit_masks);
            gam_mpd::masked::forward(&masked, &family, &fit_target)?.0
        };
        let total = (listing_bits(&fit_masks, &costs).sum() + fit_kl.sum() * observations / std::f64::consts::LN_2) / fit.rows as f64;
        // The eval inputs, selected with the fit's costs from where their last selection stands.
        let (masks, eval_kl) = select(&masked, &eval, &eval_target, eval_masks, &costs, observations, samples)?;
        eval_masks = masks;
        let point = json!({
            "iteration": iteration,
            "fit": report(&fit_masks, &fit_kl),
            "eval": report(&eval_masks, &eval_kl),
            "fit_code_bits_per_input": total,
            "pieces_steps": steps,
            "seconds": started.elapsed().as_secs_f64(),
        });
        eprintln!("{point}");
        history.push(point);
        std::fs::write(&out, serde_json::to_string_pretty(&json!({"observations": observations, "iterations": history})).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if previous - total < 1.0 {
            break;
        }
        previous = total;
    }
    Ok(())
}
