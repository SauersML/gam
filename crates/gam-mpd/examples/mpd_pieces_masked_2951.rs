//! Per-input pieces of a language model trained through its own masked forward, on streamed
//! sequences (#2951).
//!
//! `mpd_pieces_masked_2951 EXPORT_DIR PIECES_DIR OUT.json OBSERVATIONS {fit|wsvd} TRAIN EVAL [CONTEXT]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`) whose first
//! `TRAIN` token rows train the pieces and whose next `EVAL` rows evaluate them, `CONTEXT`
//! positions each (default 512). `PIECES_DIR` holds `bench/mpd_pieces_2951.py dump`'s site
//! statistics and, for `fit`, the starting libraries of `mpd_pieces_2951` (VPD naming,
//! `h.{l}.attn.q_proj` and so on); `wsvd` starts from each site's exact Fisher-whitened singular
//! pieces (`gam_mpd::pieces::fisher_svd`).
//!
//! The training sequences stream one at a time: each starts with the pieces whose own second-order
//! KL bits in the global Fisher, on its clean forward, exceed their listing cost; its sets are
//! selected exactly in the masked forward (`gam_mpd::masked::select`) with the listing costs of
//! every set selected so far; then the pieces take one exact-gradient step on it, preconditioned by
//! the running read covariances and written Fishers of every sequence seen
//! (`gam_mpd::masked::step_pieces`). Four times per pass the first four eval sequences, and at each
//! pass's end all of them, are selected one at a time with the current costs, and their mean active
//! pieces (L0), KL and bits per token in the
//! per-token frontier's code (per site `ω(k + 1) + log₂ C(C, k)`) are appended to `OUT.json` as
//! `{points: [{l0, bits, kl, …}]}`. Passes repeat until one saves less than a bit per token.

use gam_mpd::codec::prefix_integer_len_bits;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Masked, Running, costs_from_counts, listing_bits, matrix, read_values, select, sites, step_pieces};
use gam_mpd::operator_program::FamilyInputs;
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2, Axis};
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

/// Sums over a sequence's tokens: active pieces, KL, frontier bits.
fn sums(masks: &[Array2<f64>], kl: &Array1<f64>) -> (f64, f64, f64) {
    let mut l0 = 0.0;
    let mut bits = 0.0;
    for m in masks {
        let c = m.ncols() as f64;
        for row in m.outer_iter() {
            let k = row.iter().filter(|x| **x > 0.0).count();
            l0 += k as f64;
            let omega = prefix_integer_len_bits(k as u64 + 1).map_or(0.0, |b| b as f64);
            bits += omega + (ln_gamma(c + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma(c - k as f64 + 1.0)) / std::f64::consts::LN_2;
        }
    }
    (l0, kl.sum(), bits)
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_pieces_masked_2951 EXPORT_DIR PIECES_DIR OUT.json OBSERVATIONS {fit|wsvd} TRAIN EVAL [CONTEXT]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let pieces_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let start = args.get(5).ok_or(usage)?.clone();
    let train: usize = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let eval: usize = args.get(7).ok_or(usage)?.parse().map_err(|e| format!("EVAL: {e}"))?;
    let context: usize = args.get(8).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let sequence = |s: usize| -> FamilyInputs { family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>()) };
    let target_of = |inputs: &FamilyInputs| -> Result<Array2<f64>, String> {
        Ok(model.execute(inputs, false).map_err(|e| e.to_string())?.values[model.output].clone())
    };
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    let mut weights: Vec<Array1<f64>> = Vec::new();
    for site in sites(model) {
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
        let error = (&v.t().dot(&u).t() - &w).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let largest = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if error > 1e-6 * largest {
            return Err(format!("{name}: the starting library is not the site ({error:e} against {largest:e})"));
        }
        // Each piece's second-order weight `uᵀ B u` in the global Fisher.
        weights.push((&u.dot(&fisher) * &u).sum_axis(Axis(1)));
        eprintln!("{} = {name}: {d_out}×{d_in}, {} pieces", site.name, v.nrows());
        chosen.push(site);
        libraries.push(Library { v, u, mean });
    }
    let original_sites = chosen.clone();
    let mut masked = Masked::build(model, chosen, libraries)?;
    let samples = 2;
    // The firing counts of every set selected so far, and the costs they give.
    let mut counts: Vec<Array1<f64>> = masked.libraries.iter().map(|l| Array1::zeros(l.v.nrows())).collect();
    // A sequence's start: the pieces whose own second-order KL bits, `n a² uᵀBu / (2 ln 2)` with
    // `a = v · (x − μ)` on the clean forward, exceed their current listing cost.
    let start_masks = |inputs: &FamilyInputs, libraries: &[Library], costs: &[Array1<f64>]| -> Result<Vec<Array2<f64>>, String> {
        let trace = model.execute(inputs, false).map_err(|e| e.to_string())?;
        let scale = observations / (2.0 * std::f64::consts::LN_2);
        let mut masks = Vec::new();
        for (k, library) in libraries.iter().enumerate() {
            let x = read_values(&trace, &original_sites[k])? - &library.mean;
            let a = x.dot(&library.v.t());
            masks.push(Array2::from_shape_fn(a.dim(), |(r, c)| if scale * a[[r, c]] * a[[r, c]] * weights[k][c] > costs[k][c] { 1.0 } else { 0.0 }));
        }
        Ok(masks)
    };
    let mut running = Running::default();
    let mut points = Vec::new();
    let mut previous = f64::INFINITY;
    let report_every = (train / 4).max(1);
    for pass in 0.. {
        let mut pass_code = 0.0;
        for s in 0..train {
            let started = std::time::Instant::now();
            let inputs = sequence(s);
            let target = target_of(&inputs)?;
            let costs = costs_from_counts(&counts);
            let begin = start_masks(&inputs, &masked.libraries, &costs)?;
            let (masks, kl) = select(&masked, &inputs, &target, begin, &costs, observations, samples)?;
            pass_code += (listing_bits(&masks, &costs).sum() + kl.sum() * observations / std::f64::consts::LN_2) / inputs.rows as f64;
            for (count, m) in counts.iter_mut().zip(&masks) {
                *count += &m.sum_axis(Axis(0));
            }
            let step = step_pieces(&mut masked, &inputs, &target, &masks, samples, 0xF00D + (pass * train + s) as u64, &mut running)?;
            let (l0, kl_sum, _) = sums(&masks, &kl);
            log::info!(
                "pass {pass} sequence {s}: L0 {:.1}, KL {:.4} per token; step {:?}; {:.0}s",
                l0 / inputs.rows as f64,
                kl_sum / inputs.rows as f64,
                step.map(|(b, a)| (b / inputs.rows as f64, a / inputs.rows as f64)),
                started.elapsed().as_secs_f64()
            );
            // Four times a pass a progress point on the first four eval sequences; at the pass's
            // end the full eval set.
            let last = s + 1 == train;
            if last || (s + 1) % report_every == 0 {
                let costs = costs_from_counts(&counts);
                let evaluated = if last { eval } else { eval.min(4) };
                let (mut l0, mut kl, mut bits, mut tokens) = (0.0, 0.0, 0.0, 0.0);
                for e in 0..evaluated {
                    let inputs = sequence(train + e);
                    let target = target_of(&inputs)?;
                    let begin = start_masks(&inputs, &masked.libraries, &costs)?;
                    let (masks, values) = select(&masked, &inputs, &target, begin, &costs, observations, samples)?;
                    let (a, b, c) = sums(&masks, &values);
                    l0 += a;
                    kl += b;
                    bits += c;
                    tokens += inputs.rows as f64;
                }
                let point = json!({
                    "l0": l0 / tokens, "kl": kl / tokens, "bits": bits / tokens,
                    "pass": pass, "sequences_trained": pass * train + s + 1, "observations": observations,
                    "eval_sequences": evaluated,
                });
                eprintln!("eval {point}");
                points.push(point);
                std::fs::write(&out, serde_json::to_string_pretty(&json!({"points": points})).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            }
        }
        let code = pass_code / train as f64;
        log::info!("pass {pass}: {code:.1} bits per token");
        if previous - code < 1.0 {
            break;
        }
        previous = code;
    }
    Ok(())
}
