//! Exact invariance quotients of imported models (#2951).
//!
//! `mpd_quotient_2951 [SEQUENCES,CONTEXT] EXPORT_DIR...`
//!
//! Each `EXPORT_DIR` is an export `gam_mpd::import` reads; a rotary language model is read over
//! its first `SEQUENCES` token rows up to `CONTEXT` (default `1,16`). The native program is mapped to
//! its canonical representative (`quotient::quotient`); the report (JSON lines on stdout) holds the
//! message length before and after, the bits each invariance removed, the free generators of the
//! power-of-two gauge, and the representative's certified data bits against the native model, which
//! an exact quotient keeps at zero up to the banded executions' radii.

use gam_mpd::import::{import, import_language_model, is_language_model};
use gam_mpd::quotient::quotient;
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let (mut sequences, mut context) = (1, 16);
    if let Some((s, c)) = args.first().and_then(|a| a.split_once(',')) {
        sequences = s.parse().map_err(|e| format!("SEQUENCES: {e}"))?;
        context = c.parse().map_err(|e| format!("CONTEXT: {e}"))?;
        args.remove(0);
    }
    if args.is_empty() {
        return Err("mpd_quotient_2951 [SEQUENCES,CONTEXT] EXPORT_DIR...".to_string());
    }
    for dir in args.iter().map(PathBuf::from) {
        let imported = if is_language_model(&dir)? { import_language_model(&dir, sequences, context)? } else { import(&dir)? };
        let (program, contract) = (&imported.program, &imported.contract);
        let started = std::time::Instant::now();
        let q = quotient(program, contract.readouts).map_err(|e| e.to_string())?;
        let seconds = started.elapsed().as_secs_f64();
        let reference = contract.logits(program).map_err(|e| e.to_string())?;
        let score = contract.score(&q.program, &reference).map_err(|e| e.to_string())?;
        let saved: serde_json::Map<String, serde_json::Value> =
            q.saved.iter().map(|(invariance, bits)| (format!("{invariance:?}"), json!(bits))).collect();
        let line = json!({
            "model": imported.name,
            "kind": imported.kind,
            "reals_before": program.real_count(),
            "reals_after": q.program.real_count(),
            "bits_before": q.bits_before,
            "bits_after": q.bits_after,
            "saved": saved,
            "scale_generators": q.scale_generators,
            "data_bits": score.data_bits,
            "data_bits_error": score.data_bits_error,
            "argmax_disagreements": score.evaluation.argmax_disagreements,
            "seconds": seconds,
        });
        println!("{line}");
    }
    Ok(())
}
