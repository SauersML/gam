//! The decomposition engine on a pre-norm rotary language model (#2951), e.g. the 4-layer Pile
//! model exported by `~/mpd-data/vpd/vpd_engine_export.py`.
//!
//! `mpd_engine_llm_2951 EXPORT_DIR SEQUENCES CONTEXT [SCREENINGS CERTIFICATIONS]`
//!
//! The model is imported as one per-position program (`gam_mpd::import`,
//! `import_language_model`): a shared local program at every row, causal rotary attention reading
//! the rows of the same sequence. The contract is the next-token distribution at every position of
//! the first `SEQUENCES` token rows, up to `CONTEXT`. The report: the imported program's agreement
//! with the export's own logits (`logits_row0`, a measured comparison with the reference forward),
//! the native program's bits split into algorithm and data, and the engine's program with its
//! two-part code, maximal row KL and argmax agreement.

use gam_mpd::import::{import, import_language_model, is_language_model};
use gam_mpd::engine::{Budget, Coarsen, DropBlocks, LowRank, Primitive, decompose};
use gam_mpd::operator_rewrites::{CenterLogits, ComposeAffine, FoldConstants};
use gam_mpd::view::{render, view};
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_engine_llm_2951 EXPORT_DIR SEQUENCES CONTEXT [SCREENINGS CERTIFICATIONS]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let sequences: usize = args.get(2).ok_or(usage)?.parse().map_err(|e| format!("SEQUENCES: {e}"))?;
    let context: usize = args.get(3).ok_or(usage)?.parse().map_err(|e| format!("CONTEXT: {e}"))?;
    let screenings: u64 = args.get(4).map_or(Ok(1 << 12), |v| v.parse()).map_err(|e| format!("SCREENINGS: {e}"))?;
    let certifications: u64 = args.get(5).map_or(Ok(1 << 8), |v| v.parse()).map_err(|e| format!("CERTIFICATIONS: {e}"))?;
    let imported = if is_language_model(&dir)? { import_language_model(&dir, sequences, context)? } else { import(&dir)? };
    let program = &imported.program;
    let started = std::time::Instant::now();
    let logits = imported.contract.logits(program).map_err(|e| e.to_string())?;
    eprintln!(
        "{} ({}) imported and executed on {} rows in {:.1}s",
        imported.name,
        imported.kind,
        imported.contract.family.rows,
        started.elapsed().as_secs_f64()
    );
    // The export's own forward on sequence 0, positions 0..512: compare its first CONTEXT rows,
    // centred per row (logits are defined up to a shift).
    let shape = &imported.record["files"]["logits_row0"]["shape"];
    let check = match (shape[0].as_u64(), shape[1].as_u64()) {
        (Some(rows), Some(cols)) => {
            let bytes = std::fs::read(dir.join("logits_row0.f64")).map_err(|e| e.to_string())?;
            let values: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            let reference = ndarray::Array2::from_shape_vec((rows as usize, cols as usize), values).map_err(|e| e.to_string())?;
            let mut worst = 0.0_f64;
            for row in 0..context.min(rows as usize) {
                let (a, b) = (logits.values.row(row), reference.row(row));
                let (ma, mb) = (a.mean().unwrap_or(0.0), b.mean().unwrap_or(0.0));
                for (x, y) in a.iter().zip(b.iter()) {
                    worst = worst.max(((x - ma) - (y - mb)).abs());
                }
            }
            Some(worst)
        }
        _ => None,
    };
    let native_view = view(program).map_err(|e| e.to_string())?;
    eprintln!("{}", render(&native_view));
    let library: Vec<Box<dyn Primitive>> = vec![
        Box::new(FoldConstants),
        Box::new(ComposeAffine),
        Box::new(CenterLogits),
        Box::new(DropBlocks),
        Box::new(Coarsen),
        Box::new(LowRank),
    ];
    let result = decompose(program, &imported.contract, &library, &Budget { screenings, certifications, refit: None })
        .map_err(|e| e.to_string())?;
    let result_view = view(&result.program).map_err(|e| e.to_string())?;
    eprintln!("{}", render(&result_view));
    let evaluation = &result.score.evaluation;
    let report = json!({
        "rows": imported.contract.family.rows,
        "export_logits_max_centred_difference": check,
        "native": {"bits": native_view.bits, "algorithm_bits": native_view.algorithm_bits, "data_bits": native_view.data_bits, "reals": program.real_count()},
        "result": {
            "bits": result.score.program_bits,
            "structure_bits": result.score.structure_bits,
            "precision_bits": result.score.precision_bits,
            "explanation_bits": result.score.explanation.bits,
            "explanation_bits_per_input": result.score.explanation.bits_per_input(),
            "active_per_input": result.score.explanation.mean_active(),
            "algorithm_bits": result_view.algorithm_bits,
            "data_bits": result_view.data_bits,
            "behaviour_bits": result.score.data_bits,
            "reals": result.program.real_count(),
            "max_kl": evaluation.max_kl.upper_bound(),
            "max_tv": evaluation.max_tv_upper,
            "argmax_disagreements": evaluation.argmax_disagreements,
            "stop": format!("{:?}", result.stop),
        },
        "seconds": started.elapsed().as_secs_f64(),
    });
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}
