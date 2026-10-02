//! Per-input pieces of a language model trained through its own masked forward, on streamed
//! sequences (#2951).
//!
//! `mpd_pieces_masked_2951 EXPORT_DIR OUT.json OBSERVATIONS {wsvd|wsvd2|library:DIR} TRAIN EVAL [CONTEXT] [GPU]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`) whose first
//! `TRAIN` token rows train the pieces and whose next `EVAL` rows evaluate them, `CONTEXT`
//! positions each (default 512). `GPU` (`off`, `auto` (default) or `required`) is the policy for the
//! proposal products (`gam_mpd::device`); every acceptance runs in float64 on the CPU. The sites are
//! the model's hidden-to-hidden maps (`gam_mpd::masked::sites`); their read moments and output
//! Fishers are measured on the training sequences (`gam_mpd::masked::site_statistics`), so any
//! imported model runs as it is. `wsvd` starts from each site's exact Fisher-whitened singular
//! pieces (`gam_mpd::pieces::fisher_svd`), and `wsvd2` grows those to twice as many on the first
//! training sequence (`gam_mpd::masked::split`). `library:DIR` starts from a given library: per
//! site `DIR/{site}.v.f64` (pieces × d_in) and `DIR/{site}.u.f64` (pieces × d_out), raw float64,
//! on the uncentred read with nothing beyond the pieces (a site without files stays native).
//!
//! With `TRAIN` 0 nothing is trained: the eval sequences are selected pass after pass, each pass
//! with the counts of the sets the pass before selected, until a pass saves less than a bit per
//! token; every pass is a full eval. A running point after each eval sequence goes to
//! `OUT.progress.json`.
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
//! `{points: [{l0, bits, kl, …}]}`; a full eval also writes its selected sets as CSR
//! (`OUT.pass{P}.{indptr,indices,offsets}.npy`, pieces numbered site after site) and each eval
//! token's KL (`OUT.pass{P}.kl.npy`, float64, rows in order) and whether its argmax is the model's
//! (`OUT.pass{P}.agree.npy`, int64). Passes repeat until one saves less than a bit per token.

use gam_mpd::codec::prefix_integer_len_bits;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Context, Library, Masked, Running, Target, previous_inputs, read_values, select, site_statistics, sites, split, step_pieces};
use gam_mpd::operator_program::FamilyInputs;
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2, Axis};
use serde_json::json;
use statrs::function::gamma::ln_gamma;
use std::path::{Path, PathBuf};

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % (cols * 8) != 0 {
        return Err(format!("{}: {} bytes are not rows of {cols} float64", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((bytes.len() / (cols * 8), cols), values).map_err(|e| e.to_string())
}

/// A one-dimensional little-endian `.npy` file of 8-byte values of type `descr`.
fn write_npy(path: &Path, descr: &str, values: impl ExactSizeIterator<Item = [u8; 8]>) -> Result<(), String> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({},), }}", values.len());
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in values {
        bytes.extend_from_slice(&v);
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())
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
    let usage = "mpd_pieces_masked_2951 EXPORT_DIR OUT.json OBSERVATIONS {wsvd|wsvd2|library:DIR} TRAIN EVAL [CONTEXT] [GPU]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let observations: f64 = args.get(3).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let start = args.get(4).ok_or(usage)?.clone();
    let given = start.strip_prefix("library:").map(PathBuf::from);
    if start != "wsvd" && start != "wsvd2" && given.is_none() {
        return Err(format!("unknown start {start}; {usage}"));
    }
    let train: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let eval: usize = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("EVAL: {e}"))?;
    let context: usize = args.get(7).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let gpu = args.get(8).map_or("auto", String::as_str);
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::parse(gpu).ok_or_else(|| format!("GPU {gpu}: expected off, auto or required"))?);
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let context_rows = context;
    let sequence = |s: usize| -> FamilyInputs { family.select(&(s * context_rows..(s + 1) * context_rows).collect::<Vec<_>>()) };
    let target_of = |inputs: &FamilyInputs| -> Result<Target, String> {
        Ok(Target::every_row(model.execute(inputs, false).map_err(|e| e.to_string())?.values[model.output].clone()))
    };
    let all_sites = sites(model);
    // The model's statistics on the training sequences (on the eval sequences when nothing trains).
    let measured_on = if train > 0 { 0..train } else { train..train + eval };
    let statistics = site_statistics(model, &all_sites, measured_on.map(sequence), 2, 0x5EED)?;
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    let mut fishers: Vec<Array2<f64>> = Vec::new();
    for (site, measured) in all_sites.iter().zip(statistics) {
        if let Some(dir) = &given {
            let v_path = dir.join(format!("{}.v.f64", site.name));
            if !v_path.exists() {
                eprintln!("{}: no given library, native", site.name);
                continue;
            }
            let (d_out, d_in) = measured.w.dim();
            let v = read_f64(&v_path, d_in)?;
            let u = read_f64(&dir.join(format!("{}.u.f64", site.name)), d_out)?;
            if u.nrows() != v.nrows() {
                return Err(format!("{}: {} v pieces and {} u pieces", site.name, v.nrows(), u.nrows()));
            }
            let left = &measured.w - &v.t().dot(&u).t();
            let norm = |m: &Array2<f64>| m.iter().map(|x| x * x).sum::<f64>().sqrt();
            eprintln!("{}: {d_out}×{d_in}, {} given pieces, ‖W − Σ u vᵀ‖/‖W‖ = {:.2e}", site.name, v.nrows(), norm(&left) / norm(&measured.w));
            fishers.push(measured.fisher);
            chosen.push(site.clone());
            libraries.push(Library { v, u, mean: Array1::zeros(d_in) });
            continue;
        }
        let library = fisher_svd(&measured)?;
        let (v, u) = (library.v.t().to_owned(), library.u);
        let error = (&v.t().dot(&u).t() - &measured.w).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let largest = measured.w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if error > 1e-6 * largest {
            return Err(format!("{}: the starting library is not the site ({error:e} against {largest:e})", site.name));
        }
        eprintln!("{}: {}×{}, {} pieces", site.name, measured.w.nrows(), measured.w.ncols(), v.nrows());
        fishers.push(measured.fisher);
        chosen.push(site.clone());
        libraries.push(Library { v, u, mean: measured.mean });
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
    let stem = out.with_extension("");
    // Select the first `evaluated` eval sequences with `context`'s counts; a full eval also writes
    // the sets, each token's KL and argmax agreement. Returns the point, the code per token, and
    // the counts of the sets selected.
    let evaluate = |masked: &Masked, context: &Context, pass: usize, trained: usize, evaluated: usize, full: bool| -> Result<(serde_json::Value, f64, Context), String> {
        let mut seen = Context::new(&masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>());
        let (mut l0, mut kl, mut bits, mut tokens, mut explanation) = (0.0, 0.0, 0.0, 0.0, 0.0);
        // Sets as CSR over all pieces (sites in order), so other context codes can score them.
        let mut indptr: Vec<i64> = vec![0];
        let mut indices: Vec<i64> = Vec::new();
        let mut token_kl: Vec<f64> = Vec::new();
        let mut agree: Vec<i64> = Vec::new();
        let point_of = |l0: f64, kl: f64, bits: f64, explanation: f64, tokens: f64, evaluated: usize| {
            json!({
                "l0": l0 / tokens, "kl": kl / tokens, "bits": bits / tokens,
                "context_bits": explanation / tokens,
                "code": (explanation + kl * observations / std::f64::consts::LN_2) / tokens,
                "pieces": masked.libraries.iter().map(|l| l.v.nrows()).sum::<usize>(),
                "pass": pass, "sequences_trained": trained, "observations": observations,
                "eval_sequences": evaluated,
            })
        };
        for e in 0..evaluated {
            let inputs = sequence(train + e);
            let target = target_of(&inputs)?;
            let previous_rows = previous_inputs(&inputs);
            let coder = context.coder(previous_rows.clone());
            let begin = start_masks(&inputs, &masked.libraries, &coder.costs)?;
            let (masks, values) = select(masked, &inputs, &target, begin, &coder, observations, samples)?;
            explanation += coder.bits(&masks).sum();
            seen.absorb(&masks, &previous_rows);
            if full {
                token_kl.extend(values.iter().copied());
                for r in 0..inputs.rows {
                    let mut offset = 0;
                    for m in &masks {
                        indices.extend((0..m.ncols()).filter(|&c| m[[r, c]] > 0.0).map(|c| (offset + c) as i64));
                        offset += m.ncols();
                    }
                    indptr.push(indices.len() as i64);
                }
                let trace = gam_mpd::masked::forward(masked, &masked.family(&inputs, &masks), &target)?.1;
                let argmax = |row: ndarray::ArrayView1<f64>| row.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
                let logits = &trace.values[masked.program.output];
                agree.extend((0..inputs.rows).map(|r| i64::from(argmax(logits.row(r)) == argmax(target.logits.row(r)))));
            }
            let (a, b, c) = sums(&masks, &values);
            l0 += a;
            kl += b;
            bits += c;
            tokens += inputs.rows as f64;
            let progress = point_of(l0, kl, bits, explanation, tokens, e + 1);
            std::fs::write(format!("{}.progress.json", stem.display()), progress.to_string()).map_err(|e| e.to_string())?;
        }
        if full {
            let mut offsets = vec![0i64];
            for l in &masked.libraries {
                offsets.push(offsets[offsets.len() - 1] + l.v.nrows() as i64);
            }
            for (name, values) in [("indptr", &indptr), ("indices", &indices), ("offsets", &offsets), ("agree", &agree)] {
                write_npy(&PathBuf::from(format!("{}.pass{pass}.{name}.npy", stem.display())), "<i8", values.iter().map(|v| v.to_le_bytes()))?;
            }
            write_npy(&PathBuf::from(format!("{}.pass{pass}.kl.npy", stem.display())), "<f8", token_kl.iter().map(|v| v.to_le_bytes()))?;
        }
        let point = point_of(l0, kl, bits, explanation, tokens, evaluated);
        let code = point["code"].as_f64().unwrap_or(f64::INFINITY);
        Ok((point, code, seen))
    };
    let mut points = Vec::new();
    let write_points = |points: &[serde_json::Value]| -> Result<(), String> {
        std::fs::write(&out, serde_json::to_string_pretty(&json!({"points": points})).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    };
    if train == 0 {
        let mut previous = f64::INFINITY;
        for pass in 0.. {
            let (point, code, seen) = evaluate(&masked, &context, pass, 0, eval, true)?;
            eprintln!("eval {point}");
            points.push(point);
            write_points(&points)?;
            if previous - code < 1.0 {
                break;
            }
            previous = code;
            context = seen;
        }
        return Ok(());
    }
    let mut running = Running::default();
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
                let (point, _, _) = evaluate(&masked, &context, pass, pass * train + s + 1, evaluated, last)?;
                eprintln!("eval {point}");
                points.push(point);
                write_points(&points)?;
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
