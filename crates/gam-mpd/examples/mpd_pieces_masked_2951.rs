//! Per-input pieces of a language model trained through its own masked forward, on streamed
//! sequences (#2951).
//!
//! `mpd_pieces_masked_2951 EXPORT_DIR PIECES_DIR OUT.json OBSERVATIONS {fit|wsvd|wsvd2} TRAIN EVAL [CONTEXT] [GPU]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`) whose first
//! `TRAIN` token rows train the pieces and whose next `EVAL` rows evaluate them, `CONTEXT`
//! positions each (default 512). `GPU` (`off`, `auto` (default) or `required`) is the policy for the
//! proposal products (`gam_mpd::device`); every acceptance runs in float64 on the CPU. `PIECES_DIR` holds `bench/mpd_pieces_2951.py dump`'s site
//! statistics and, for `fit`, the starting libraries of `mpd_pieces_2951` (VPD naming,
//! `h.{l}.attn.q_proj` and so on); `wsvd` starts from each site's exact Fisher-whitened singular
//! pieces (`gam_mpd::pieces::fisher_svd`), and `wsvd2` grows those to twice as many on the first
//! training sequence (`gam_mpd::masked::split`).
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
use gam_mpd::masked::{Context, Library, Masked, Running, matrix, previous_inputs, read_values, select, sites, split, step_pieces};
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
/// A one-dimensional little-endian int64 `.npy` file.
fn write_npy(path: &Path, values: &[i64]) -> Result<(), String> {
    let mut header = format!("{{'descr': '<i8', 'fortran_order': False, 'shape': ({},), }}", values.len());
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

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
    let usage = "mpd_pieces_masked_2951 EXPORT_DIR PIECES_DIR OUT.json OBSERVATIONS {fit|wsvd|wsvd2} TRAIN EVAL [CONTEXT] [GPU]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let pieces_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let start = args.get(5).ok_or(usage)?.clone();
    let train: usize = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let eval: usize = args.get(7).ok_or(usage)?.parse().map_err(|e| format!("EVAL: {e}"))?;
    let context: usize = args.get(8).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let gpu = args.get(9).map_or("auto", String::as_str);
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::parse(gpu).ok_or_else(|| format!("GPU {gpu}: expected off, auto or required"))?);
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let context_rows = context;
    let sequence = |s: usize| -> FamilyInputs { family.select(&(s * context_rows..(s + 1) * context_rows).collect::<Vec<_>>()) };
    let target_of = |inputs: &FamilyInputs| -> Result<Array2<f64>, String> {
        Ok(model.execute(inputs, false).map_err(|e| e.to_string())?.values[model.output].clone())
    };
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    let mut fishers: Vec<Array2<f64>> = Vec::new();
    for site in sites(model) {
        let Some(name) = vpd_name(&site.name) else { continue };
        let w = matrix(model, &site)?;
        let (d_out, d_in) = w.dim();
        let mean = Array1::from_vec(read_f64(&pieces_dir.join(format!("{name}.mu.f64")), 1, d_in)?.into_raw_vec_and_offset().0);
        let fisher = read_f64(&pieces_dir.join(format!("{name}.B.f64")), d_out, d_out)?;
        let (v, u) = match start.as_str() {
            "wsvd" | "wsvd2" => {
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
        fishers.push(fisher);
        eprintln!("{} = {name}: {d_out}×{d_in}, {} pieces", site.name, v.nrows());
        chosen.push(site);
        libraries.push(Library { v, u, mean });
    }
    let original_sites = chosen.clone();
    let mut masked = Masked::build(model, chosen, libraries)?;
    let samples = 2;
    // The firing counts of every set selected so far, and the costs they give.
    let mut context = Context::new(&masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>());
    // A sequence's start: the pieces whose own second-order KL bits, `n a² uᵀBu / (2 ln 2)` with
    // `a = v · (x − μ)` on the clean forward, exceed their current listing cost.
    let start_masks = |inputs: &FamilyInputs, libraries: &[Library], costs: &[Array1<f64>]| -> Result<Vec<Array2<f64>>, String> {
        let trace = model.execute(inputs, false).map_err(|e| e.to_string())?;
        let scale = observations / (2.0 * std::f64::consts::LN_2);
        let mut masks = Vec::new();
        for (k, library) in libraries.iter().enumerate() {
            let x = read_values(&trace, &original_sites[k])? - &library.mean;
            let a = x.dot(&library.v.t());
            // Each piece's second-order weight `uᵀ B u` in the global Fisher.
            let weights = (&library.u.dot(&fishers[k]) * &library.u).sum_axis(Axis(1));
            masks.push(Array2::from_shape_fn(a.dim(), |(r, c)| if scale * a[[r, c]] * a[[r, c]] * weights[c] > costs[k][c] { 1.0 } else { 0.0 }));
        }
        Ok(masks)
    };
    // `wsvd2`: the Fisher-SVD library grown to twice its pieces on the first training sequence,
    // every piece its listing inputs use in two ways split in two (`gam_mpd::masked::split`).
    if start == "wsvd2" {
        let inputs = sequence(0);
        let target = target_of(&inputs)?;
        let coder = context.coder(previous_inputs(&inputs));
        let begin = start_masks(&inputs, &masked.libraries, &coder.costs)?;
        let (masks, _) = select(&masked, &inputs, &target, begin, &coder, observations, samples)?;
        let trace = model.execute(&inputs, false).map_err(|e| e.to_string())?;
        let mut grown = Vec::new();
        for (k, library) in masked.libraries.iter().enumerate() {
            let x = read_values(&trace, &original_sites[k])?;
            grown.push(split(library, &x, &masks[k]).0);
        }
        eprintln!("grown to {} pieces", grown.iter().map(|l| l.v.nrows()).sum::<usize>());
        masked = Masked::build(model, original_sites.clone(), grown)?;
        context = Context::new(&masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>());
    }
    // The bits of one real of a library piece: they are sent in single precision.
    const BITS_PER_REAL: f64 = 32.0;
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
            let previous_rows = previous_inputs(&inputs);
            let coder = context.coder(previous_rows.clone());
            let begin = start_masks(&inputs, &masked.libraries, &coder.costs)?;
            let (masks, kl) = select(&masked, &inputs, &target, begin, &coder, observations, samples)?;
            let sequence_code = (coder.bits(&masks).sum() + kl.sum() * observations / std::f64::consts::LN_2) / inputs.rows as f64;
            pass_code += sequence_code;
            context.absorb(&masks, &previous_rows);
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
                // Growth, tested by the code: every piece this sequence lists two ways is split, the
                // sequence is selected again, and the split stays when its explanation and KL bits
                // per token fall by more than the added pieces' library bits spread over every
                // token trained so far.
                let trace = model.execute(&inputs, false).map_err(|e| e.to_string())?;
                let mut grown = Vec::new();
                let mut origins = Vec::new();
                let mut grown_masks = Vec::new();
                let mut added_reals = 0.0;
                for (k, library) in masked.libraries.iter().enumerate() {
                    let x = read_values(&trace, &original_sites[k])?;
                    let (bigger, m, origin) = split(library, &x, &masks[k]);
                    added_reals += ((bigger.v.nrows() - library.v.nrows()) * (library.v.ncols() + library.u.ncols())) as f64;
                    grown.push(bigger);
                    grown_masks.push(m);
                    origins.push(origin);
                }
                let candidate = Masked::build(model, original_sites.clone(), grown)?;
                let candidate_context = context.grown(&origins);
                let candidate_coder = candidate_context.coder(previous_rows.clone());
                let (candidate_masks, candidate_kl) = select(&candidate, &inputs, &target, grown_masks, &candidate_coder, observations, samples)?;
                let candidate_code = (candidate_coder.bits(&candidate_masks).sum() + candidate_kl.sum() * observations / std::f64::consts::LN_2) / inputs.rows as f64;
                // The library as it stands after this sequence's step, on the same sets.
                let now_kl = gam_mpd::masked::forward(&masked, &masked.family(&inputs, &masks), &target)?.0;
                let sequence_code = (coder.bits(&masks).sum() + now_kl.sum() * observations / std::f64::consts::LN_2) / inputs.rows as f64;
                let tokens_trained = ((pass * train + s + 1) * context_rows) as f64;
                let library_bits = added_reals * BITS_PER_REAL / tokens_trained;
                let kept = candidate_code + library_bits < sequence_code;
                log::info!(
                    "split test: {:.1} -> {:.1} bits per token, library {:.1} bits per token; {}",
                    sequence_code,
                    candidate_code,
                    library_bits,
                    if kept { "kept" } else { "refused" }
                );
                let masks = if kept {
                    masked = candidate;
                    context = candidate_context;
                    candidate_masks
                } else {
                    masks
                };
                // Growth from what selection leaves out, tested the same way: per site, the leading
                // regression pieces of the left-out map that would recover more KL bits on this
                // sequence than their library bits (`gam_mpd::masked::dropped_atoms`), appended off.
                let family = masked.family(&inputs, &masks);
                let (base_kl, masked_trace, _) = gam_mpd::masked::forward(&masked, &family, &target)?;
                let base_code = (context.coder(previous_rows.clone()).bits(&masks).sum() + base_kl.sum() * observations / std::f64::consts::LN_2) / inputs.rows as f64;
                let mut grown = Vec::new();
                let mut grown_masks = Vec::new();
                let mut added = Vec::new();
                let mut added_reals = 0.0;
                for (k, library) in masked.libraries.iter().enumerate() {
                    let per_piece = (library.v.ncols() + library.u.ncols()) as f64 * BITS_PER_REAL * inputs.rows as f64 / tokens_trained;
                    let (v, u) = gam_mpd::masked::dropped_atoms(&masked, k, &masked_trace, &masks[k], &running, observations, per_piece)?;
                    added_reals += (v.nrows() * (v.ncols() + u.ncols())) as f64;
                    added.push(v.nrows());
                    grown_masks.push(ndarray::concatenate(Axis(1), &[masks[k].view(), Array2::<f64>::zeros((inputs.rows, v.nrows())).view()]).map_err(|e| e.to_string())?);
                    grown.push(gam_mpd::masked::with_pieces(library, &v, &u)?);
                }
                if added.iter().any(|a| *a > 0) {
                    let candidate = Masked::build(model, original_sites.clone(), grown)?;
                    let candidate_context = context.extended(&added);
                    let candidate_coder = candidate_context.coder(previous_rows.clone());
                    let (candidate_masks, candidate_kl) = select(&candidate, &inputs, &target, grown_masks, &candidate_coder, observations, samples)?;
                    let candidate_code = (candidate_coder.bits(&candidate_masks).sum() + candidate_kl.sum() * observations / std::f64::consts::LN_2) / inputs.rows as f64;
                    let library_bits = added_reals * BITS_PER_REAL / tokens_trained;
                    let kept = candidate_code + library_bits < base_code;
                    log::info!(
                        "dropped-atoms test: {} pieces, {base_code:.1} -> {candidate_code:.1} bits per token, library {library_bits:.1}; {}",
                        added.iter().sum::<usize>(),
                        if kept { "kept" } else { "refused" }
                    );
                    if kept {
                        masked = candidate;
                        context = candidate_context;
                    }
                }
                let evaluated = if last { eval } else { eval.min(4) };
                let (mut l0, mut kl, mut bits, mut tokens, mut explanation) = (0.0, 0.0, 0.0, 0.0, 0.0);
                // At a full eval the selected sets are written as CSR over all pieces (sites in
                // order), so other context codes can score exactly these sets.
                let mut indptr: Vec<i64> = vec![0];
                let mut indices: Vec<i64> = Vec::new();
                for e in 0..evaluated {
                    let inputs = sequence(train + e);
                    let target = target_of(&inputs)?;
                    let coder = context.coder(previous_inputs(&inputs));
                    let begin = start_masks(&inputs, &masked.libraries, &coder.costs)?;
                    let (masks, values) = select(&masked, &inputs, &target, begin, &coder, observations, samples)?;
                    explanation += coder.bits(&masks).sum();
                    if last {
                        for r in 0..inputs.rows {
                            let mut offset = 0;
                            for m in &masks {
                                indices.extend((0..m.ncols()).filter(|&c| m[[r, c]] > 0.0).map(|c| (offset + c) as i64));
                                offset += m.ncols();
                            }
                            indptr.push(indices.len() as i64);
                        }
                    }
                    let (a, b, c) = sums(&masks, &values);
                    l0 += a;
                    kl += b;
                    bits += c;
                    tokens += inputs.rows as f64;
                }
                if last {
                    let mut offsets = vec![0i64];
                    for l in &masked.libraries {
                        offsets.push(offsets[offsets.len() - 1] + l.v.nrows() as i64);
                    }
                    let stem = out.with_extension("");
                    for (name, values) in [("indptr", &indptr), ("indices", &indices), ("offsets", &offsets)] {
                        write_npy(&PathBuf::from(format!("{}.pass{pass}.{name}.npy", stem.display())), values)?;
                    }
                }
                let point = json!({
                    "l0": l0 / tokens, "kl": kl / tokens, "bits": bits / tokens,
                    "context_bits": explanation / tokens,
                    "pieces": masked.libraries.iter().map(|l| l.v.nrows()).sum::<usize>(),
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
