//! The decomposition engine on exported models, with default settings (#2951 blind benchmark).
//!
//! `mpd_engine_blind_2951 MODEL_DIR OUT_DIR [SCREENINGS CERTIFICATIONS]`
//!
//! `MODEL_DIR` holds an `export.json` and raw float64 tensors (`gam_mpd::import`
//! reads transformers, residual MLPs and RNNs). The contract is the export's samples at its declared
//! readouts. No tolerance is declared: each program is chosen by its two-part code, over a ladder of
//! observations per sample `n = 1, 10, …, 10⁶`, each search starting from the previous rung's
//! program, so the report is the frontier of program bits against the behaviour explained.
//!
//! Written to `OUT_DIR`: `report.json` (per rung: program bits, data bits, maximal row KL, argmax
//! disagreements, population bounds for a sampled family, and the component view) and, per rung,
//! the program's decoded message `program_n{n}.bits` (raw bytes; its length in bits is in the
//! report).

use gam_mpd::import::{import, import_language_model, is_language_model};
use gam_mpd::engine::{Budget, Coarsen, DeadUnits, DropBlocks, LawSubstitution, LowRank, Primitive, decompose_from};
use gam_mpd::operator_rewrites::{
    BilinearConstantSide, CenterLogits, ComposeAffine, DropKeyBias, FoldConstants, PlaneBasis, PushThroughMix, StackTerms,
};
use gam_mpd::derivatives::CurvaturePrecision;
use gam_mpd::factors::SharedFactors;
use gam_mpd::refit::RefitSearch;
use gam_mpd::view::view;
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_engine_blind_2951 MODEL_DIR OUT_DIR [SCREENINGS CERTIFICATIONS]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let screenings: u64 = args.get(3).map_or(Ok(1 << 20), |v| v.parse()).map_err(|e| format!("SCREENINGS: {e}"))?;
    let certifications: u64 = args.get(4).map_or(Ok(1 << 12), |v| v.parse()).map_err(|e| format!("CERTIFICATIONS: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    // A language model's family is its first 4 token rows at 64 positions.
    let imported = if is_language_model(&dir)? { import_language_model(&dir, 4, 64)? } else { import(&dir)? };
    let library: Vec<Box<dyn Primitive>> = vec![
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
    ];
    let budget = Budget { screenings, certifications, refit: Some(RefitSearch { newton_steps: 8, conjugate_gradient_steps: 16 }) };
    let model = &imported.program;
    let native_bits = model.code_bits().map_err(|e| e.to_string())?;
    let mut start = model.clone();
    let mut frontier = Vec::new();
    for exponent in 0..=6u32 {
        let n = 10u64.pow(exponent);
        let mut contract = imported.contract.clone();
        contract.observations = n;
        let started = std::time::Instant::now();
        let result = decompose_from(model, &start, &contract, &library, &budget).map_err(|e| e.to_string())?;
        let evaluation = &result.score.evaluation;
        let program_view = view(&result.program).map_err(|e| e.to_string())?;
        let message = result.program.encode().map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity((message.len_bits() as usize).div_ceil(8));
        let mut reader = message.reader();
        let mut byte = 0u8;
        for index in 0..message.len_bits() {
            byte = (byte << 1) | u8::from(reader.read_bit().map_err(|e| format!("{e:?}"))?);
            if index % 8 == 7 {
                bytes.push(byte);
                byte = 0;
            }
        }
        if message.len_bits() % 8 != 0 {
            bytes.push(byte << (8 - message.len_bits() % 8));
        }
        std::fs::write(out.join(format!("program_n{n}.bits")), &bytes).map_err(|e| e.to_string())?;
        frontier.push(json!({
            "observations": n,
            "program_bits": result.score.program_bits,
            "structure_bits": result.score.structure_bits,
            "precision_bits": result.score.precision_bits,
            "explanation_bits": result.score.explanation.bits,
            "explanation_bits_per_input": result.score.explanation.bits_per_input(),
            "active_per_input": result.score.explanation.mean_active(),
            "data_bits": result.score.data_bits,
            "reals": result.program.real_count(),
            "max_kl": evaluation.max_kl.upper_bound(),
            "argmax_disagreements": evaluation.argmax_disagreements,
            "argmax_uncertified": evaluation.argmax_uncertified,
            "population": result.score.population.as_ref().map(|p| json!({
                "units": p.units, "disagreeing": p.disagreeing,
                "fixed_program_upper": p.fixed_program_upper, "selected_program_upper": p.selected_program_upper,
            })),
            "unresolved_bits_fraction": program_view.unresolved_fraction(),
            "components": program_view.components.iter().filter(|c| c.bits > 0).map(|c| json!({
                "name": c.name, "reads": c.reads, "writes": c.writes, "laws": c.laws, "uses": c.uses,
                "reals": c.reals, "bits": c.bits, "sources": c.sources, "unresolved": c.unresolved,
            })).collect::<Vec<_>>(),
            "stop": format!("{:?}", result.stop),
            "seconds": started.elapsed().as_secs_f64(),
        }));
        eprintln!(
            "{} n={n}: {} + {:.1} bits (native {native_bits}), max KL <= {:e}, {} argmax disagreements, {:.1}s",
            imported.name,
            result.score.program_bits,
            result.score.data_bits,
            evaluation.max_kl.upper_bound().unwrap_or(f64::INFINITY),
            evaluation.argmax_disagreements,
            started.elapsed().as_secs_f64()
        );
        start = result.program;
    }
    let report = json!({
        "model": imported.name,
        "kind": imported.kind,
        "config": imported.record["config"],
        "native_bits": native_bits,
        "native_reals": model.real_count(),
        "rows": imported.contract.family.rows,
        "readouts": imported.contract.readouts,
        "frontier": frontier,
    });
    std::fs::write(out.join("report.json"), serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    Ok(())
}
