//! How often a keep/refuse decision of the masked selection could be taken in low precision and
//! still be exact (#2951): the rate that decides the trainer's hardware.
//!
//! `mpd_decision_margins_2951 EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json [SEQUENCES] [CONTEXT] [OBSERVATIONS] [GPU] [time]`
//!
//! With `GPU` `auto` the selection's trials score through screened heads (`gam_mpd::masked`'s
//! module note); with a trailing `time` the rounds are only timed, no margins measured.
//!
//! A selection round decides each sequence by the sign of its code change `Σ_r (before_r −
//! after_r)` (`gam_mpd::masked::select`), and each input's code is its listing bits (exact
//! integers and logarithms of counts) plus `n KL_r / ln 2`. A forward in a lower precision with unit
//! roundoff `u` moves each logit by at most its rounding band, which the CPU's banded execution
//! bounds for float64 (`u = 2⁻⁵³`); every term of that band is a rounding of some operation, so to
//! first order in `u` the band of the same forward at `u'` is the float64 band times `u'/2⁻⁵³`. A
//! logit change `Δz` moves `KL(p ‖ q)` by at most `2 max_c |Δz_c|`, and the KL's own sum over `V`
//! classes rounds by at most `γ_{V+8}(u') Σ_c p_c (|ln p_c| + |ln q_c|)`. A decision is certified
//! at `u'` when its float64 margin exceeds `n/ln 2` times the sum of its inputs' KL bounds under both
//! masks; otherwise it needs the float64 forward.
//!
//! The library and the given per-token sets (`library:DIR` and `SETS` of
//! `mpd_pieces_masked_2951`) start the selection of the first `SEQUENCES` export sequences (default
//! 4), their first `CONTEXT` tokens (default 128), all in one family; every round of
//! `select_observed` is scored at f32 (`u' = 2⁻²⁴`) and at TF32 operands (`u' = 2⁻¹¹`, an upper
//! bound on the tensor-core path's per-operation rounding). Per sequence decision and per input
//! (each flipping input's own code change) it records the margin and both bounds; `OUT.json`
//! summarises the fractions certified, `OUT.decisions.json` lists every decision.

use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Context, Library, Masked, Round, Target, previous_inputs, select_observed, sites};
use ndarray::{Array1, Array2};
use serde_json::json;
use std::path::{Path, PathBuf};

fn read_raw<T>(path: &Path, width: usize, decode: impl Fn([u8; 8]) -> T) -> Result<Vec<T>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % (8 * width.max(1)) != 0 {
        return Err(format!("{}: not rows of {width} 8-byte values", path.display()));
    }
    Ok(bytes.chunks_exact(8).map(|c| decode([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

fn softmax(z: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e = z.mapv(|v| (v - m).exp());
    let total = e.sum();
    e / total
}

fn gamma(n: usize, u: f64) -> f64 {
    n as f64 * u / (1.0 - n as f64 * u)
}

/// Per input, the float64 band of its logits (largest entry's box plus its ball) and the
/// magnitude `Σ_c p_c (|ln p_c| + |ln q_c|)` the KL sums.
fn row_bands(masked: &Masked, family: &gam_mpd::operator_program::FamilyInputs, target: &Target) -> Result<(Vec<f64>, Vec<f64>), String> {
    let trace = masked.program.execute(family, true).map_err(|e| e.to_string())?;
    let output = masked.program.output;
    let (bands, balls) = (trace.bands.as_ref().ok_or("no bands")?, trace.balls.as_ref().ok_or("no balls")?);
    let logits = &trace.values[output];
    let mut band = Vec::new();
    let mut magnitude = Vec::new();
    for r in 0..logits.nrows() {
        band.push(bands[output].row(r).iter().fold(0.0_f64, |m, v| m.max(*v)) + balls[output][r]);
        let (p, q) = (softmax(target.logits.row(r)), softmax(logits.row(r)));
        magnitude.push(p.iter().zip(q.iter()).map(|(a, b)| a * (a.max(f64::MIN_POSITIVE).ln().abs() + b.max(f64::MIN_POSITIVE).ln().abs())).sum());
    }
    Ok((band, magnitude))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_decision_margins_2951 EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json [SEQUENCES] [CONTEXT] [OBSERVATIONS] [GPU] [time]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let library_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let sets_dir = PathBuf::from(args.get(3).ok_or(usage)?);
    let out = PathBuf::from(args.get(4).ok_or(usage)?);
    let sequences: usize = args.get(5).map_or(Ok(4), |v| v.parse()).map_err(|e| format!("SEQUENCES: {e}"))?;
    let context: usize = args.get(6).map_or(Ok(128), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let observations: f64 = args.get(7).map_or(Ok(1024.0), |v| v.parse()).map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let gpu = args.get(8).map_or("off", String::as_str);
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::parse(gpu).ok_or_else(|| format!("GPU {gpu}: expected off, auto or required"))?);

    let imported = import_language_model(&export, sequences, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    // The library, and the sets' numbering: `sites.txt` names each site and its pieces, and the
    // sets' rows run over the export's sequences at the length they were made for.
    let listed = std::fs::read_to_string(sets_dir.join("sites.txt")).map_err(|e| e.to_string())?;
    let named: Vec<(String, usize)> = listed
        .lines()
        .map(|l| {
            let (name, n) = l.rsplit_once(' ').ok_or("sites.txt: name and pieces")?;
            Ok((name.to_string(), n.parse::<usize>().map_err(|e| e.to_string())?))
        })
        .collect::<Result<_, String>>()?;
    let all = sites(model);
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    for (name, pieces) in &named {
        let site = all.iter().find(|s| &s.name == name).ok_or_else(|| format!("{name}: not a site"))?;
        let w = gam_mpd::masked::matrix(model, site)?;
        let (d_out, d_in) = w.dim();
        let v = read_raw(&library_dir.join(format!("{name}.v.f64")), d_in, f64::from_le_bytes)?;
        let u = read_raw(&library_dir.join(format!("{name}.u.f64")), d_out, f64::from_le_bytes)?;
        if v.len() != pieces * d_in || u.len() != pieces * d_out {
            return Err(format!("{name}: the library does not have {pieces} pieces"));
        }
        chosen.push(site.clone());
        libraries.push(Library {
            v: Array2::from_shape_vec((*pieces, d_in), v).map_err(|e| e.to_string())?,
            u: Array2::from_shape_vec((*pieces, d_out), u).map_err(|e| e.to_string())?,
            mean: Array1::zeros(d_in),
        });
    }
    let offsets: Vec<usize> = std::iter::once(0).chain(named.iter().scan(0, |a, (_, p)| {
        *a += p;
        Some(*a)
    })).collect();
    let masked = Masked::build(model, chosen, libraries)?;
    let indptr = read_raw(&sets_dir.join("indptr.i64"), 1, i64::from_le_bytes)?;
    let indices = read_raw(&sets_dir.join("indices.i64"), 1, i64::from_le_bytes)?;
    // The sets were made at the model's context (`config.n_ctx`) per sequence.
    let length = imported.record["config"]["n_ctx"].as_u64().ok_or("config.n_ctx")? as usize;
    if (indptr.len() - 1) < sequences * length {
        return Err(format!("{}: sets of fewer than {sequences} sequences", sets_dir.display()));
    }
    let mut masks: Vec<Array2<f64>> = named.iter().map(|(_, p)| Array2::zeros((family.rows, *p))).collect();
    for s in 0..sequences {
        for t in 0..context {
            let position = s * length + t;
            for &i in &indices[indptr[position] as usize..indptr[position + 1] as usize] {
                let i = i as usize;
                let k = offsets.partition_point(|o| *o <= i) - 1;
                masks[k][[s * context + t, i - offsets[k]]] = 1.0;
            }
        }
    }
    eprintln!("{} sites, {} pieces, sets of {length}-token sequences, {} rows", named.len(), offsets[named.len()], family.rows);

    let target = Target::every_row(model.execute(family, false).map_err(|e| e.to_string())?.values[model.output].clone());
    let coder = Context::new(&masked.all_pieces()).coder(previous_inputs(family));
    let scale = observations / std::f64::consts::LN_2;
    let classes = target.logits.ncols();
    // Unit roundoffs of the low-precision paths, and float64's.
    let paths = [("f32", 2f64.powi(-24)), ("tf32", 2f64.powi(-11))];
    let u64_ = 2f64.powi(-53);
    let mut decisions = Vec::new();
    let mut inputs = Vec::new();
    let started = std::time::Instant::now();
    let mut banding = 0.0;
    let margins = args.get(9).is_none_or(|v| v != "time");
    // The forward a selection's trials take, whole and with each site's `z` at its mask's nonzeros.
    let start_family = masked.family(family, &masks);
    let timed = |gated: bool| -> Result<f64, String> {
        let mut seconds = Vec::new();
        for _ in 0..3 {
            let clock = std::time::Instant::now();
            if gated {
                masked.program.execute_gated(&start_family, &masked.gates()).map_err(|e| e.to_string())?;
            } else {
                masked.program.execute(&start_family, false).map_err(|e| e.to_string())?;
            }
            seconds.push(clock.elapsed().as_secs_f64());
        }
        seconds.sort_by(f64::total_cmp);
        Ok(seconds[1])
    };
    let density = masks.iter().map(|m| m.iter().filter(|v| **v != 0.0).count()).sum::<usize>() as f64 / masks.iter().map(|m| m.len()).sum::<usize>() as f64;
    eprintln!("forward {:.3}s, gated {:.3}s, mask density {density:.4}", timed(false)?, timed(true)?);
    drop(start_family);
    let mut observe = |round: &Round<'_>| -> Result<(), String> {
        if !margins {
            return Ok(());
        }
        let clock = std::time::Instant::now();
        let (band_now, magnitude_now) = row_bands(&masked, round.current, &target)?;
        let (band_new, magnitude_new) = row_bands(&masked, round.proposed, &target)?;
        banding += clock.elapsed().as_secs_f64();
        // Per input and path, the bound on its code change's error.
        let bound = |r: usize, u: f64| -> f64 {
            let kl = |band: f64, magnitude: f64| 2.0 * band * (u / u64_) + gamma(classes + 8, u) * magnitude;
            scale * (kl(band_now[r], magnitude_now[r]) + kl(band_new[r], magnitude_new[r]))
        };
        let count = round.sequence_of.iter().copied().max().map_or(0, |m| m + 1);
        for q in 0..count {
            let rows: Vec<usize> = (0..round.flipped.len()).filter(|r| round.sequence_of[*r] == q).collect();
            if rows.iter().all(|r| round.flipped[*r] == 0) {
                continue;
            }
            let margin: f64 = rows.iter().map(|r| round.before[*r] - round.after[*r]).sum();
            let mut record = json!({"sequence": q, "margin": margin});
            for (name, u) in paths {
                record[name] = json!(rows.iter().map(|r| bound(*r, u)).sum::<f64>());
            }
            decisions.push(record);
        }
        for r in (0..round.flipped.len()).filter(|r| round.flipped[*r] > 0) {
            let mut record = json!({"margin": round.before[r] - round.after[r], "flips": round.flipped[r]});
            for (name, u) in paths {
                record[name] = json!(bound(r, u));
            }
            inputs.push(record);
        }
        Ok(())
    };
    select_observed(&masked, family, &target, masks, &coder, observations, 2, None, &mut observe)?;
    let total = started.elapsed().as_secs_f64();
    let summary = |records: &[serde_json::Value]| -> serde_json::Value {
        let mut out = json!({"count": records.len()});
        for (name, _) in paths {
            let certified = records.iter().filter(|d| d["margin"].as_f64().unwrap_or(0.0).abs() > d[name].as_f64().unwrap_or(f64::INFINITY)).count();
            out[name] = json!({
                "certified": certified,
                "fallback_fraction": if records.is_empty() { 0.0 } else { 1.0 - certified as f64 / records.len() as f64 },
            });
        }
        let mut ratios: Vec<f64> = records.iter().map(|d| d["margin"].as_f64().unwrap_or(0.0).abs() / d["f32"].as_f64().unwrap_or(1.0)).collect();
        ratios.sort_by(f64::total_cmp);
        if !ratios.is_empty() {
            out["margin_over_f32_bound_quantiles"] = json!([0.01, 0.1, 0.5, 0.9].map(|q| ratios[((ratios.len() - 1) as f64 * q) as usize]));
        }
        out
    };
    let report = json!({
        "sequences": sequences, "context": context, "observations": observations,
        "pieces": offsets[named.len()],
        "sequence_decisions": summary(&decisions),
        "input_decisions": summary(&inputs),
        "seconds": total, "banding_seconds": banding,
    });
    eprintln!("{report}");
    std::fs::write(&out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let listing = out.with_extension("decisions.json");
    std::fs::write(&listing, json!({"sequences": decisions, "inputs": inputs}).to_string()).map_err(|e| e.to_string())?;
    Ok(())
}
