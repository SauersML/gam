//! Refusal and over-refusal pieces of a chat model, from a behaviour-scoped decomposition (#2951).
//!
//! `mpd_refusal_pieces_2951 validate MODEL_DIR DIR [GPU]`
//! `mpd_refusal_pieces_2951 fit MODEL_DIR DATA_DIR OUT_DIR LAST OBSERVATIONS STATS [BATCH] [GPU]`
//!
//! `MODEL_DIR` is a Hugging Face Qwen2/Qwen3/Llama checkpoint, imported from block `FIRST` on
//! (`gam_mpd::import::hugging_face_language_model`): the blocks before it run elsewhere, and their
//! output, the residual stream entering `FIRST`, is the program's input.
//!
//! `validate` checks the import: `DIR/meta.json` (`ids`, `first`, `rows`, `scored`, `vocab`),
//! `DIR/resid.f64` (the stream entering `first`, `rows × d`) and `DIR/logits.f64` (the reference
//! logits of the last `scored` rows); it prints the largest logit difference and KL.
//!
//! `fit` decomposes blocks `FIRST..LAST` against the behaviour. `DATA_DIR/prompts.json` holds
//! `first`, `d`, `refusal_tokens` (the first tokens of the model's refusals) and per prompt its
//! `ids` (chat-formatted prompt and the first reply tokens), `scored_from` (its rows from there on
//! are the behaviour's: the last user token, the template tokens after it and the reply tokens)
//! and `decision` (the row whose next token is the reply's first); `DATA_DIR/resid.f32` holds every
//! prompt's stream entering `FIRST`, prompts in order. The run:
//!
//! 1. measures each site's narrow-side statistics on the first `STATS` prompts' scored rows (the
//!    reads' mean and covariance, and the output Fisher of the scored rows' KL, `gam_mpd::pieces::
//!    Narrow`) and builds its exact Fisher-SVD library (`gam_mpd::pieces::fisher_svd_narrow`);
//! 2. selects every prompt's pieces in the masked program on its scored rows only, every piece on
//!    elsewhere (`gam_mpd::masked::Target`), with the context code of the sets selected so far;
//! 3. at the selected sets, takes the exact gradient of the refusal score `log p(R) − log(1 −
//!    p(R))` at the decision row (`R` the refusal tokens) in every mask entry: per scored row and
//!    active piece, the piece's first-order share of the score; and per prompt and piece, summed
//!    over all of the prompt's rows where it is on, the first-order change of the score were the
//!    piece removed from the weights.
//!
//! Writes `OUT_DIR`: `library/<site>.{v,u}.f32` (`C × d_in`, `C × d_out`) and `.mean.f64`;
//! `pieces.json` (sites, offsets, weights `s_c`); `rows.json` (per scored row its prompt and
//! position, KL and the selection's code); the scored rows' sets as CSR over all pieces
//! (`sets.indptr.i64`, `sets.indices.i64`) with each entry's score gradient (`sets.share.f32`);
//! `effects.f32` (prompts × pieces); and per prompt the native and masked refusal scores.

use gam_mpd::derivatives::vjp;
use gam_mpd::device::proposing;
use gam_mpd::import::hugging_face_language_model;
use gam_mpd::masked::{
    Context, Library, Masked, Site, Target, forward, mask_gradients, matrix, read_values, sampled_label_cotangent, select, sites,
};
use gam_mpd::operator_program::{FamilyInputs, SequenceLayout, SlotValues};
use gam_mpd::pieces::{Narrow, fisher_svd_narrow};
use ndarray::{Array1, Array2, Axis, concatenate};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};

fn read_le<const N: usize, T>(path: &Path, convert: fn([u8; N]) -> T) -> Result<Vec<T>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % N != 0 {
        return Err(format!("{}: {} bytes", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(N).map(|c| convert(c.try_into().expect("a chunk"))).collect())
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| e.to_string())
}

fn integer(value: &Value, key: &str) -> Result<usize, String> {
    value[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("{key}: not an integer"))
}

fn write_raw<T: Copy, const N: usize>(file: &mut std::fs::File, values: impl IntoIterator<Item = T>, bytes: fn(T) -> [u8; N]) -> Result<(), String> {
    let mut out = Vec::new();
    for v in values {
        out.extend_from_slice(&bytes(v));
    }
    file.write_all(&out).map_err(|e| e.to_string())
}

fn create(path: &Path) -> Result<std::fs::File, String> {
    std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// One prompt of the behaviour.
struct Prompt {
    ids: Vec<u32>,
    scored_from: usize,
    decision: usize,
}

/// A batch of prompts as program inputs: the streams entering `FIRST`, each prompt one sequence.
struct Batch {
    inputs: FamilyInputs,
    scored: Vec<bool>,
    /// Per prompt: its first row and its decision row.
    starts: Vec<usize>,
    decisions: Vec<usize>,
}

fn batch(prompts: &[Prompt], offsets: &[usize], resid: &[f32], d: usize, range: std::ops::Range<usize>) -> Batch {
    let rows: usize = range.clone().map(|p| prompts[p].ids.len()).sum();
    let mut values = Array2::<f64>::zeros((rows, d));
    let (mut sequence, mut position, mut scored, mut starts, mut decisions) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut row = 0;
    for (s, p) in range.enumerate() {
        let prompt = &prompts[p];
        starts.push(row);
        decisions.push(row + prompt.decision);
        for t in 0..prompt.ids.len() {
            let source = &resid[(offsets[p] + t) * d..(offsets[p] + t + 1) * d];
            values.row_mut(row).iter_mut().zip(source).for_each(|(v, x)| *v = f64::from(*x));
            sequence.push(s as u32);
            position.push(t as u32);
            scored.push(t >= prompt.scored_from);
            row += 1;
        }
    }
    let inputs = FamilyInputs { rows, slots: vec![SlotValues::Raw(values)], layout: Some(SequenceLayout { sequence, position }) };
    Batch { inputs, scored, starts, decisions }
}

fn softmax(z: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e = z.mapv(|v| (v - m).exp());
    let total = e.sum();
    e / total
}

/// The refusal score `log p(R) − log(1 − p(R))` of a row's logits.
fn refusal_score(logits: ndarray::ArrayView1<'_, f64>, refusal: &[usize]) -> f64 {
    let q = softmax(logits);
    let p: f64 = refusal.iter().map(|&i| q[i]).sum();
    p.ln() - (1.0 - p).ln()
}

fn validate(model: &Path, dir: &Path) -> Result<(), String> {
    let meta = read_json(&dir.join("meta.json"))?;
    let (first, rows, scored, vocab) = (integer(&meta, "first")?, integer(&meta, "rows")?, integer(&meta, "scored")?, integer(&meta, "vocab")?);
    let (program, record) = hugging_face_language_model(model, first)?;
    let d = integer(&record["config"], "d_model")?;
    let resid = read_le::<8, f64>(&dir.join("resid.f64"), f64::from_le_bytes)?;
    let reference = Array2::from_shape_vec((scored, vocab), read_le::<8, f64>(&dir.join("logits.f64"), f64::from_le_bytes)?).map_err(|e| e.to_string())?;
    let inputs = FamilyInputs {
        rows,
        slots: vec![SlotValues::Raw(Array2::from_shape_vec((rows, d), resid).map_err(|e| e.to_string())?)],
        layout: Some(SequenceLayout { sequence: vec![0; rows], position: (0..rows as u32).collect() }),
    };
    let started = std::time::Instant::now();
    let logits = program.execute(&inputs, false).map_err(|e| e.to_string())?.values[program.output].slice(ndarray::s![rows - scored.., ..]).to_owned();
    let largest = reference.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let difference = (&logits - &reference).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let kl = Target::every_row(reference.clone());
    let kls = gam_mpd::masked::kl(&kl, &logits).0;
    let report = json!({
        "first": first, "rows": rows, "blocks": integer(&record["config"], "n_layers")? - first,
        "max_abs_logit_difference": difference, "largest_logit": largest,
        "max_kl_reference_to_engine": kls.iter().fold(0.0_f64, |m, x| m.max(*x)),
        "argmax_agree": (0..scored).all(|r| {
            let arg = |a: ndarray::ArrayView1<'_, f64>| a.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
            arg(logits.row(r)) == arg(reference.row(r))
        }),
        "seconds": started.elapsed().as_secs_f64(),
    });
    println!("{report}");
    Ok(())
}

/// A site's narrow-side running sums over scored rows (`gam_mpd::pieces::Narrow`).
enum Sums {
    Reads { x: Array1<f64>, xx: Array2<f64>, pulled: Array2<f64> },
    Writes { x: Array1<f64>, y: Array1<f64>, yy: Array2<f64>, gg: Array2<f64> },
}

fn layer_of(site: &Site) -> Option<usize> {
    site.name.strip_prefix("blocks.")?.split('.').next()?.parse().ok()
}

fn fit(
    model: &Path,
    data: &Path,
    out: &Path,
    last: usize,
    observations: f64,
    stats: usize,
    batch_size: usize,
) -> Result<(), String> {
    let meta = read_json(&data.join("prompts.json"))?;
    let (first, d) = (integer(&meta, "first")?, integer(&meta, "d")?);
    let refusal: Vec<usize> = meta["refusal_tokens"].as_array().ok_or("refusal_tokens")?.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect();
    let mut prompts = Vec::new();
    for p in meta["prompts"].as_array().ok_or("prompts")? {
        let ids: Vec<u32> = p["ids"].as_array().ok_or("ids")?.iter().filter_map(|v| v.as_u64().map(|v| v as u32)).collect();
        prompts.push(Prompt { ids, scored_from: integer(p, "scored_from")?, decision: integer(p, "decision")? });
    }
    let mut offsets = vec![0];
    for p in &prompts {
        offsets.push(offsets[offsets.len() - 1] + p.ids.len());
    }
    let resid = read_le::<4, f32>(&data.join("resid.f32"), f32::from_le_bytes)?;
    if resid.len() != offsets[prompts.len()] * d {
        return Err(format!("resid.f32 holds {} values for {} rows of {d}", resid.len(), offsets[prompts.len()]));
    }
    std::fs::create_dir_all(out.join("library")).map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let (program, _) = hugging_face_language_model(model, first)?;
    let chosen: Vec<Site> = sites(&program).into_iter().filter(|s| layer_of(s).is_some_and(|l| (first..last).contains(&l))).collect();
    let maps: Vec<Array2<f64>> = chosen.iter().map(|s| matrix(&program, s)).collect::<Result<_, _>>()?;
    log::info!("imported blocks {first}.. ({:.0}s); {} sites in {first}..{last}", started.elapsed().as_secs_f64(), chosen.len());
    let batches: Vec<std::ops::Range<usize>> = (0..prompts.len()).step_by(batch_size).map(|s| s..(s + batch_size).min(prompts.len())).collect();

    // 1. Narrow-side statistics on the scored rows of the first `stats` prompts.
    let samples = 2;
    let mut sums: Vec<Sums> = maps
        .iter()
        .map(|w| {
            let (d_out, d_in) = w.dim();
            if d_in <= d_out {
                Sums::Reads { x: Array1::zeros(d_in), xx: Array2::zeros((d_in, d_in)), pulled: Array2::zeros((d_in, d_in)) }
            } else {
                Sums::Writes { x: Array1::zeros(d_in), y: Array1::zeros(d_out), yy: Array2::zeros((d_out, d_out)), gg: Array2::zeros((d_out, d_out)) }
            }
        })
        .collect();
    let (mut rows, mut draws) = (0.0, 0.0);
    for range in batches.iter().filter(|r| r.start < stats) {
        let b = batch(&prompts, &offsets, &resid, d, range.start..range.end.min(stats));
        let keep: Vec<usize> = (0..b.inputs.rows).filter(|r| b.scored[*r]).collect();
        let trace = program.execute(&b.inputs, false).map_err(|e| e.to_string())?;
        let target = Target { logits: trace.values[program.output].clone(), scored: Some(b.scored.clone()) };
        for ((site, w), sum) in chosen.iter().zip(&maps).zip(sums.iter_mut()) {
            let x = read_values(&trace, site)?.select(Axis(0), &keep);
            match sum {
                Sums::Reads { x: sx, xx, .. } => {
                    *sx += &x.sum_axis(Axis(0));
                    *xx += &x.t().dot(&x);
                }
                Sums::Writes { x: sx, y: sy, yy, .. } => {
                    let y = x.dot(&w.t());
                    *sx += &x.sum_axis(Axis(0));
                    *sy += &y.sum_axis(Axis(0));
                    *yy += &y.t().dot(&y);
                }
            }
        }
        rows += keep.len() as f64;
        for draw in 0..samples {
            let cotangent = sampled_label_cotangent(&target.logits, &target, 0x5EED ^ (((range.start * samples + draw) as u64) << 8));
            let back = proposing(|| vjp(&program, &b.inputs, &trace, cotangent)).map_err(|e| e.to_string())?;
            for ((site, w), sum) in chosen.iter().zip(&maps).zip(sums.iter_mut()) {
                let written: Vec<Array2<f64>> =
                    site.writes.iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
                let views: Vec<_> = written.iter().map(|w| w.view()).collect();
                let g = concatenate(Axis(1), &views).map_err(|e| e.to_string())?.select(Axis(0), &keep);
                match sum {
                    Sums::Reads { pulled, .. } => {
                        let p = g.dot(w);
                        *pulled += &p.t().dot(&p);
                    }
                    Sums::Writes { gg, .. } => *gg += &g.t().dot(&g),
                }
            }
            draws += keep.len() as f64;
        }
        log::info!("statistics: prompts {}..{} ({:.0}s)", range.start, range.end.min(stats), started.elapsed().as_secs_f64());
    }
    if rows == 0.0 {
        return Err("no scored rows for the statistics".to_string());
    }
    let outer = |a: &Array1<f64>| a.view().insert_axis(Axis(1)).dot(&a.view().insert_axis(Axis(0)));
    let mut libraries = Vec::new();
    let mut weights = Vec::new();
    let mut piece_sites = Vec::new();
    let mut offset = 0;
    for ((site, w), sum) in chosen.iter().zip(&maps).zip(sums) {
        let (narrow, mean) = match sum {
            Sums::Reads { x, xx, pulled } => {
                let mean = x / rows;
                (Narrow::Reads { covariance: xx / rows - outer(&mean), pulled_fisher: pulled / draws }, mean)
            }
            Sums::Writes { x, y, yy, gg } => {
                let y_mean = y / rows;
                (Narrow::Writes { fisher: gg / draws, written_covariance: yy / rows - outer(&y_mean) }, x / rows)
            }
        };
        let (library, weight) = fisher_svd_narrow(w, &narrow)?;
        let exactness = library.exactness(w);
        log::info!("{}: {}×{}, {} pieces ({} for exactness), exactness {exactness:.1e}", site.name, w.nrows(), w.ncols(), library.u.nrows(), library.exactness_pieces);
        let v = library.v.t().to_owned();
        let file = |suffix: &str| out.join("library").join(format!("{}.{suffix}", site.name));
        write_raw(&mut create(&file("v.f32"))?, v.iter().map(|x| *x as f32), f32::to_le_bytes)?;
        write_raw(&mut create(&file("u.f32"))?, library.u.iter().map(|x| *x as f32), f32::to_le_bytes)?;
        write_raw(&mut create(&file("mean.f64"))?, mean.iter().copied(), f64::to_le_bytes)?;
        piece_sites.push(json!({
            "name": site.name, "layer": layer_of(site), "d_out": w.nrows(), "d_in": w.ncols(), "pieces": v.nrows(),
            "offset": offset, "exactness_pieces": library.exactness_pieces, "exactness": exactness, "weights": weight,
        }));
        offset += v.nrows();
        weights.push(weight);
        libraries.push(Library { v, u: library.u, mean });
    }
    let total_pieces = offset;
    std::fs::write(out.join("pieces.json"), serde_json::to_string(&json!({"first": first, "last": last, "observations": observations, "sites": piece_sites})).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let masked = Masked::build(&program, chosen.clone(), libraries)?;
    log::info!("libraries built: {total_pieces} pieces ({:.0}s)", started.elapsed().as_secs_f64());

    // 2-3. Selection and the refusal score's gradient, prompt batch by batch.
    let mut context = Context::new(&masked.libraries.iter().map(|l| l.v.nrows()).collect::<Vec<_>>());
    let (mut indptr_file, mut indices_file, mut share_file, mut effects_file) = (
        create(&out.join("sets.indptr.i64"))?,
        create(&out.join("sets.indices.i64"))?,
        create(&out.join("sets.share.f32"))?,
        create(&out.join("effects.f32"))?,
    );
    let mut entries: i64 = 0;
    write_raw(&mut indptr_file, [entries], i64::to_le_bytes)?;
    let (mut row_records, mut prompt_records) = (Vec::new(), Vec::new());
    let scale = observations / std::f64::consts::LN_2;
    for range in &batches {
        let tick = std::time::Instant::now();
        let b = batch(&prompts, &offsets, &resid, d, range.clone());
        let native = program.execute(&b.inputs, false).map_err(|e| e.to_string())?;
        let target = Target { logits: native.values[program.output].clone(), scored: Some(b.scored.clone()) };
        let previous: Vec<Option<usize>> = (0..b.inputs.rows).map(|r| (r > 0 && b.scored[r] && b.scored[r - 1] && !b.starts.contains(&r)).then(|| r - 1)).collect();
        let coder = context.coder(previous.clone());
        // The start: every piece on off the behaviour's rows; on its rows the pieces whose own
        // second-order KL bits, `n s_c a² / (2 ln 2)` with `a = v · (x − μ)`, exceed their cost.
        let mut start = Vec::new();
        for (k, (site, library)) in chosen.iter().zip(&masked.libraries).enumerate() {
            let a = (read_values(&native, site)? - &library.mean).dot(&library.v.t());
            start.push(Array2::from_shape_fn(a.dim(), |(r, c)| {
                let on = !b.scored[r] || 0.5 * scale * a[[r, c]] * a[[r, c]] * weights[k][c] > coder.costs[k][c];
                if on { 1.0 } else { 0.0 }
            }));
        }
        let (masks, _) = select(&masked, &b.inputs, &target, start, &coder, observations, samples)?;
        // The code over the behaviour's rows alone, and its counts.
        let keep: Vec<usize> = (0..b.inputs.rows).filter(|r| b.scored[*r]).collect();
        let kept_masks: Vec<Array2<f64>> = masks.iter().map(|m| m.select(Axis(0), &keep)).collect();
        let kept_previous: Vec<Option<usize>> = keep.iter().map(|r| previous[*r].and_then(|p| keep.iter().position(|q| *q == p))).collect();
        let listing = context.coder(kept_previous.clone()).bits(&kept_masks);
        context.absorb(&kept_masks, &kept_previous);
        // The refusal score's gradient in every mask entry at the selected sets.
        let family = masked.family(&b.inputs, &masks);
        let (kl, trace, _) = forward(&masked, &family, &target)?;
        let logits = &trace.values[masked.program.output];
        let mut cotangent = Array2::<f64>::zeros(logits.dim());
        for &r in &b.decisions {
            let q = softmax(logits.row(r));
            let p: f64 = refusal.iter().map(|&i| q[i]).sum();
            for (j, value) in cotangent.row_mut(r).iter_mut().enumerate() {
                *value = -q[j] / (1.0 - p);
            }
            for &i in &refusal {
                cotangent[[r, i]] = q[i] / p;
            }
        }
        let shares = mask_gradients(&masked, &family, &trace, cotangent)?;
        for (s, p) in range.clone().enumerate() {
            let (from, to) = (b.starts[s], b.starts[s] + prompts[p].ids.len());
            let mut effects = Vec::with_capacity(total_pieces);
            for (m, g) in masks.iter().zip(&shares) {
                let on = &m.slice(ndarray::s![from..to, ..]) * &g.slice(ndarray::s![from..to, ..]);
                effects.extend(on.sum_axis(Axis(0)).iter().map(|x| *x as f32));
            }
            write_raw(&mut effects_file, effects, f32::to_le_bytes)?;
            prompt_records.push(json!({
                "native_score": refusal_score(native.values[program.output].row(b.decisions[s]), &refusal),
                "masked_score": refusal_score(logits.row(b.decisions[s]), &refusal),
            }));
        }
        let mut active = 0usize;
        for (i, &r) in keep.iter().enumerate() {
            let mut base = 0;
            let (mut indices, mut values) = (Vec::new(), Vec::new());
            for (m, g) in masks.iter().zip(&shares) {
                for c in 0..m.ncols() {
                    if m[[r, c]] > 0.0 {
                        indices.push((base + c) as i64);
                        values.push(g[[r, c]] as f32);
                    }
                }
                base += m.ncols();
            }
            active += indices.len();
            entries += indices.len() as i64;
            write_raw(&mut indices_file, indices, i64::to_le_bytes)?;
            write_raw(&mut share_file, values, f32::to_le_bytes)?;
            write_raw(&mut indptr_file, [entries], i64::to_le_bytes)?;
            let s = b.starts.iter().rposition(|start| *start <= r).unwrap_or(0);
            row_records.push(json!({"prompt": range.start + s, "position": r - b.starts[s], "kl": kl[r], "listing_bits": listing[i]}));
        }
        let kl_mean = keep.iter().map(|r| kl[*r]).sum::<f64>() / keep.len().max(1) as f64;
        log::info!(
            "prompts {}..{}: L0 {:.1} of {total_pieces}, KL {kl_mean:.4} per scored row, listing {:.1} bits; {:.0}s",
            range.start,
            range.end,
            active as f64 / keep.len().max(1) as f64,
            listing.sum() / keep.len().max(1) as f64,
            tick.elapsed().as_secs_f64()
        );
        std::fs::write(out.join("rows.json"), serde_json::to_string(&json!({"rows": row_records, "prompts": prompt_records})).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    log::info!("done ({:.0}s)", started.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_refusal_pieces_2951 {validate MODEL_DIR DIR [GPU] | fit MODEL_DIR DATA_DIR OUT_DIR LAST OBSERVATIONS STATS [BATCH] [GPU]}";
    let arg = |i: usize| args.get(i).ok_or_else(|| usage.to_string());
    let gpu = |i: usize| -> Result<(), String> {
        let raw = args.get(i).map_or("off", String::as_str);
        gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::parse(raw).ok_or_else(|| format!("GPU {raw}: expected off, auto or required"))?);
        Ok(())
    };
    match arg(1)?.as_str() {
        "validate" => {
            gpu(4)?;
            validate(&PathBuf::from(arg(2)?), &PathBuf::from(arg(3)?))
        }
        "fit" => {
            let number = |i: usize| -> Result<f64, String> { arg(i)?.parse().map_err(|e| format!("{}: {e}", args[i])) };
            let batch_size = args.get(8).map_or(Ok(4), |v| v.parse()).map_err(|e| format!("BATCH: {e}"))?;
            gpu(9)?;
            fit(
                &PathBuf::from(arg(2)?),
                &PathBuf::from(arg(3)?),
                &PathBuf::from(arg(4)?),
                number(5)? as usize,
                number(6)?,
                number(7)? as usize,
                batch_size,
            )
        }
        _ => Err(usage.to_string()),
    }
}
