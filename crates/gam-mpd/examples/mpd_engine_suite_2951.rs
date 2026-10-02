//! The decomposition engine on one exported model with the suite's defaults (#2951): one row of the
//! evaluation table.
//!
//! `mpd_engine_suite_2951 MODEL_DIR OUT_DIR [SEQUENCES CONTEXT [SCREENINGS CERTIFICATIONS [RUNGS]]]`
//!
//! `MODEL_DIR` is any export `gam_mpd::import` reads: a transformer, residual MLP or RNN (`import`),
//! or a rotary language model (`import_language_model`, its first `SEQUENCES` token rows at
//! positions `0..CONTEXT`, default 1 × 32). Every model gets the same library, the same budget and
//! the contract its export declares; nothing is tuned per model. The amount of behaviour explained
//! is a ladder, `n = 10^RUNGS … 10^0` observations of every row (default `RUNGS = 6`), each rung's
//! search starting from the previous (larger-n) rung's program, so the row is a frontier of program bits
//! against behaviour, as in the blind benchmark.
//!
//! The row (`OUT_DIR/report.json`, and one summary line on stdout):
//! * native: the imported program's bits (algorithm and lookup-table data), reals, explanation
//!   size, and for a language model the largest centred difference between its logits and the
//!   export's own forward (`logits_row0`);
//! * `rounded`: the model with every operator on the lattice of `b` bits under its RMS,
//!   `p = b − round(log₂ rms)` (the dense-target baseline of the VPD scoreboard), for
//!   `b = 16, 12, 8, 6, 5, 4, 3, 2`: program bits, data bits at one observation, maximal row KL,
//!   argmax agreement, and per rung the best of them, `min_b bits_b + n Σ KL_b / ln 2`;
//! * per rung, the engine's program: program bits, data bits, maximal row KL, argmax agreement, the stop
//!   reason, the time, its explanation size (mean acting pieces per input and their bits given
//!   the program, the score's `Explanation`) and its largest components (what the top rules say).

use gam_mpd::contract::{Contract, ProgramScore};
use gam_mpd::derivatives::CurvaturePrecision;
use gam_mpd::engine::{
    Budget, Coarsen, DeadUnits, DropBlocks, Edit, LawSubstitution, LowRank, Primitive, apply_edit, decompose_with_reference,
};
use gam_mpd::factors::SharedFactors;
use gam_mpd::import::{import, import_language_model, is_language_model};
use gam_mpd::operator_program::{OperatorBody, OperatorProgram};
use gam_mpd::operator_rewrites::{
    BilinearConstantSide, CenterLogits, ComposeAffine, DropKeyBias, FoldConstants, PlaneBasis, PushThroughMix, StackTerms,
};
use gam_mpd::precision::DeclaredPrecision;
use gam_mpd::refit::RefitSearch;
use gam_mpd::secant::BandedMatrix;
use gam_mpd::view::{ProgramView, view};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn library() -> Vec<Box<dyn Primitive>> {
    vec![
        Box::new(FoldConstants),
        Box::new(BilinearConstantSide),
        Box::new(ComposeAffine),
        Box::new(PushThroughMix),
        Box::new(PlaneBasis),
        Box::new(DropBlocks),
        Box::new(Coarsen),
        Box::new(DeadUnits),
        Box::new(LowRank),
        Box::new(StackTerms),
        Box::new(CenterLogits),
        Box::new(DropKeyBias),
        Box::new(SharedFactors),
        Box::new(LawSubstitution),
        Box::new(CurvaturePrecision { probes: 4 }),
    ]
}

/// The model with every operator's reals on the lattice of `bits` bits under their RMS.
fn rounded(model: &OperatorProgram, bits: i32) -> Result<OperatorProgram, String> {
    let mut program = model.clone();
    for index in 0..program.operators.len() {
        let op = &program.operators[index];
        let squares: f64 = match &op.body {
            OperatorBody::Identity => continue,
            OperatorBody::Dense { values, .. } => values.iter().map(|v| v * v).sum(),
            OperatorBody::LowRank { left, right, .. } => left.iter().chain(right.iter()).map(|v| v * v).sum(),
        };
        let count = op.real_count();
        if count == 0 || squares == 0.0 {
            continue;
        }
        let rms = (squares / count as f64).sqrt();
        let precision = DeclaredPrecision::new(bits - rms.log2().round() as i32)?;
        apply_edit(&mut program, &Edit::Precision { operator: index, precision }).map_err(|e| e.to_string())?;
    }
    Ok(program)
}

fn scored(score: &ProgramScore, rows: usize, observations: u64) -> Value {
    let evaluation = &score.evaluation;
    json!({
        "program_bits": score.program_bits,
        "structure_bits": score.structure_bits,
        "precision_bits": score.precision_bits,
        "explanation_bits": score.explanation.bits,
        "instances": score.explanation.instances,
        "active_per_input": score.explanation.mean_active(),
        "explanation_bits_per_input": score.explanation.bits_per_input(),
        "data_bits": score.data_bits,
        "max_kl": evaluation.max_kl.upper_bound(),
        "mean_kl": score.data_bits * std::f64::consts::LN_2 / (rows as f64 * observations as f64),
        "max_tv": evaluation.max_tv_upper,
        "argmax_agreement": 1.0 - evaluation.argmax_disagreements as f64 / rows as f64,
        "argmax_uncertified": evaluation.argmax_uncertified,
    })
}

fn components(program_view: &ProgramView, top: usize) -> Vec<Value> {
    let mut sorted: Vec<_> = program_view.components.iter().filter(|c| c.bits > 0).collect();
    sorted.sort_by(|a, b| b.bits.cmp(&a.bits));
    sorted
        .into_iter()
        .take(top)
        .map(|c| {
            json!({
                "name": c.name, "reads": c.reads, "writes": c.writes, "laws": c.laws, "uses": c.uses,
                "reals": c.reals, "bits": c.bits, "native": c.unresolved, "data": c.data,
            })
        })
        .collect()
}

/// The largest centred difference between the program's logits and the export's own forward on
/// token row 0, over the first `context` positions.
fn export_check(dir: &Path, record: &Value, logits: &BandedMatrix, context: usize) -> Result<Option<f64>, String> {
    let shape = &record["files"]["logits_row0"]["shape"];
    let (Some(rows), Some(cols)) = (shape[0].as_u64(), shape[1].as_u64()) else { return Ok(None) };
    let bytes = std::fs::read(dir.join("logits_row0.f64")).map_err(|e| e.to_string())?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    let reference = ndarray::Array2::from_shape_vec((rows as usize, cols as usize), values).map_err(|e| e.to_string())?;
    let mut worst = 0.0_f64;
    for row in 0..context.min(rows as usize).min(logits.values.nrows()) {
        let (a, b) = (logits.values.row(row), reference.row(row));
        let (ma, mb) = (a.mean().unwrap_or(0.0), b.mean().unwrap_or(0.0));
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max(((x - ma) - (y - mb)).abs());
        }
    }
    Ok(Some(worst))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_engine_suite_2951 MODEL_DIR OUT_DIR [SEQUENCES CONTEXT [SCREENINGS CERTIFICATIONS [RUNGS]]]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let number = |i: usize, default: u64| -> Result<u64, String> {
        args.get(i).map_or(Ok(default), |v| v.parse().map_err(|e| format!("argument {i}: {e}")))
    };
    let (sequences, context) = (number(3, 1)? as usize, number(4, 32)? as usize);
    let budget = Budget {
        screenings: number(5, 1 << 20)?,
        certifications: number(6, 1 << 12)?,
        refit: Some(RefitSearch { newton_steps: 8, conjugate_gradient_steps: 16 }),
    };
    let rungs = number(7, 6)? as u32;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let language_model = is_language_model(&dir)?;
    let imported = if language_model { import_language_model(&dir, sequences, context)? } else { import(&dir)? };
    let model = &imported.program;
    let contract: &Contract = &imported.contract;
    let rows = contract.family.rows * contract.readouts;
    let import_seconds = started.elapsed().as_secs_f64();
    let reference = contract.logits(model).map_err(|e| e.to_string())?;
    let reference_seconds = started.elapsed().as_secs_f64() - import_seconds;
    // The reference's widest band (a non-finite band counts as infinite): how much the banded
    // forward of the model itself resolves.
    let widest_band = reference.bands.iter().fold(0.0_f64, |w, b| if b.is_finite() { w.max(*b) } else { f64::INFINITY });
    let check = if language_model { export_check(&dir, &imported.record, &reference, context)? } else { None };
    eprintln!(
        "{}: {} rows ({} inputs × {} readouts) imported in {import_seconds:.1}s, executed with bands in {reference_seconds:.1}s (widest band {widest_band:e}), export check {check:?}",
        dir.display(),
        rows,
        contract.family.rows,
        contract.readouts
    );
    let native_view = view(model).map_err(|e| e.to_string())?;
    let native_score = contract.score(model, &reference).map_err(|e| e.to_string())?;
    let mut once = contract.clone();
    once.observations = 1;
    let mut baselines = Vec::new();
    for bits in [16, 12, 8, 6, 5, 4, 3, 2] {
        let program = rounded(model, bits)?;
        let score = once.score(&program, &reference).map_err(|e| e.to_string())?;
        let mut row = scored(&score, rows, 1);
        row["b"] = json!(bits);
        row["algorithm_bits"] = json!(view(&program).map_err(|e| e.to_string())?.algorithm_bits);
        eprintln!("  rounded b={bits}: {}", row);
        baselines.push(row);
    }
    // The ladder over the amount of behaviour explained: n = 10^k observations of every row.
    let mut start = model.clone();
    let mut ladder: Vec<Value> = Vec::new();
    let mut search_seconds = 0.0;
    // Descending: the search only removes and coarsens, so each rung starts from the program of
    // the rung with more behaviour (an ascending ladder would start n = 10 from n = 1's program,
    // which explains almost nothing and cannot regrow).
    for exponent in (0..=rungs).rev() {
        let n = 10u64.pow(exponent);
        let mut contract = contract.clone();
        contract.observations = n;
        let search = Instant::now();
        let result = decompose_with_reference(&reference, &[model, &start], &contract, &library(), &budget).map_err(|e| e.to_string())?;
        let seconds = search.elapsed().as_secs_f64();
        search_seconds += seconds;
        let result_view = view(&result.program).map_err(|e| e.to_string())?;
        // The best rounded model at this n: its bits plus n Σ KL / ln 2.
        let best_rounded = baselines
            .iter()
            .map(|row| {
                let total = row["program_bits"].as_f64().unwrap_or(f64::INFINITY)
                    + row["explanation_bits"].as_f64().unwrap_or(f64::INFINITY)
                    + n as f64 * row["data_bits"].as_f64().unwrap_or(f64::INFINITY);
                (total, row["b"].as_i64().unwrap_or(0))
            })
            .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
        let mut rung = scored(&result.score, rows, n);
        rung["observations"] = json!(n);
        rung["total_bits"] = json!(result.score.total());
        rung["best_rounded_total_bits"] = json!(best_rounded.0);
        rung["best_rounded_b"] = json!(best_rounded.1);
        rung["algorithm_bits"] = json!(result_view.algorithm_bits);
        rung["table_bits"] = json!(result_view.data_bits);
        rung["unresolved_fraction"] = json!(result_view.unresolved_fraction());
        rung["reals"] = json!(result.program.real_count());
        rung["stop"] = json!(format!("{:?}", result.stop));
        rung["seconds"] = json!(seconds);
        rung["top_components"] = json!(components(&result_view, 8));
        println!(
            "ROW {} n=1e{exponent} | native {} | ours {} + {:.1} expl + {:.1} data (rounded best {:.0} at b={}) | max KL {:.3e} | argmax {:.4} | {:?} | {:.0}s | active {:.1}/{} ({:.1} bits/input)",
            dir.file_name().map_or(String::new(), |n| n.to_string_lossy().to_string()),
            native_view.bits,
            result.score.program_bits,
            result.score.explanation.bits,
            result.score.data_bits,
            best_rounded.0,
            best_rounded.1,
            result.score.evaluation.max_kl.upper_bound().unwrap_or(f64::INFINITY),
            1.0 - result.score.evaluation.argmax_disagreements as f64 / rows as f64,
            result.stop,
            seconds,
            result.score.explanation.mean_active(),
            result.score.explanation.instances,
            result.score.explanation.bits_per_input(),
        );
        ladder.push(rung);
        start = result.program;
        let report = json!({
            "model": dir.display().to_string(),
            "name": imported.name,
            "kind": imported.kind,
            "rows": rows,
            "inputs": contract.family.rows,
            "readouts": contract.readouts,
            "sequences_context": language_model.then_some((sequences, context)),
            "export_logits_max_centred_difference": check,
            "reference_widest_band": widest_band,
            "native": {
                "program_bits": native_view.bits,
                "algorithm_bits": native_view.algorithm_bits,
                "table_bits": native_view.data_bits,
                "reals": model.real_count(),
                "score": scored(&native_score, rows, 1),
            },
            "rounded_at_one_observation": baselines,
            "ladder": ladder,
            "seconds": {"import": import_seconds, "reference": reference_seconds, "search": search_seconds, "total": started.elapsed().as_secs_f64()},
        });
        std::fs::write(out.join("report.json"), serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
