//! The decomposition engine on an exported model, read out by the printer (#2951).
//!
//! `mpd_printer_2951 MODEL_DIR OUT_DIR [LAST_EXPONENT [SCREENINGS CERTIFICATIONS]]`
//!
//! `MODEL_DIR` is an export `gam_mpd::import` reads. The engine runs over the ladder of observations
//! per input `n = 10^LAST_EXPONENT … 10⁰` (default 6), largest first, each rung starting from the
//! previous rung's program, with the blind benchmark's importer, library and budget. The order matters
//! because the search only removes and rewrites: a program chosen for more behaviour keeps what a
//! smaller `n` may still remove, while one chosen for less cannot regain it. Each rung's program, and
//! the native one, is printed (`printer::print`) with its behaviour: per input
//! `n KL_upper/ln 2` and the argmax agreement. Written to `OUT_DIR`: `printout_native.txt`,
//! `printout_n{n}.txt`, and `printouts.json` (per rung: observations, the certified score and the
//! printout).

use gam_mpd::import::{import, import_language_model, is_language_model};
use gam_mpd::contract::{Contract, ProgramScore};
use gam_mpd::engine::{Budget, decompose_from, library};
use gam_mpd::operator_program::OperatorProgram;
use gam_mpd::printer::{Behaviour, Printout, print};
use serde_json::json;
use std::path::{Path, PathBuf};

fn printout(program: &OperatorProgram, contract: &Contract, score: &ProgramScore) -> Result<Printout, String> {
    let evaluation = &score.evaluation;
    let scale = contract.observations as f64 / std::f64::consts::LN_2;
    let row_bits: Vec<f64> = evaluation.kl_upper.iter().map(|kl| kl * scale).collect();
    let behaviour = Behaviour { inputs: &contract.family, row_bits: &row_bits, argmax_agrees: &evaluation.argmax_agrees };
    print(program, Some(&behaviour)).map_err(|e| e.to_string())
}

fn write(out: &Path, name: &str, text: &str) -> Result<(), String> {
    std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_printer_2951 MODEL_DIR OUT_DIR [LAST_EXPONENT [SCREENINGS CERTIFICATIONS]]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let last: u32 = args.get(3).map_or(Ok(6), |v| v.parse()).map_err(|e| format!("LAST_EXPONENT: {e}"))?;
    let screenings: u64 = args.get(4).map_or(Ok(1 << 20), |v| v.parse()).map_err(|e| format!("SCREENINGS: {e}"))?;
    let certifications: u64 = args.get(5).map_or(Ok(1 << 12), |v| v.parse()).map_err(|e| format!("CERTIFICATIONS: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    // A language model's family is its first 4 token rows at 64 positions, as in the blind benchmark.
    let imported = if is_language_model(&dir)? { import_language_model(&dir, 4, 64)? } else { import(&dir)? };
    let library = library();
    let budget = Budget { screenings, certifications, ..Budget::default() };
    let model = &imported.program;
    let reference = imported.contract.logits(model).map_err(|e| e.to_string())?;
    let native_score = imported.contract.score(model, &reference).map_err(|e| e.to_string())?;
    let native = printout(model, &imported.contract, &native_score)?;
    write(&out, "printout_native.txt", &native.to_string())?;
    let mut rungs = vec![json!({"observations": null, "program_bits": native_score.program_bits, "printout": native})];
    let mut start = model.clone();
    for exponent in (0..=last).rev() {
        let n = 10u64.pow(exponent);
        let mut contract = imported.contract.clone();
        contract.observations = n;
        let started = std::time::Instant::now();
        let result = decompose_from(model, &start, &contract, &library, &budget).map_err(|e| e.to_string())?;
        let reading = printout(&result.program, &contract, &result.score)?;
        let text = reading.to_string();
        eprintln!("{} n={n} ({:.1}s, stop {:?})\n{text}", imported.name, started.elapsed().as_secs_f64(), result.stop);
        write(&out, &format!("printout_n{n}.txt"), &text)?;
        rungs.push(json!({
            "observations": n,
            "program_bits": result.score.program_bits,
            "data_bits": result.score.data_bits,
            "max_kl": result.score.evaluation.max_kl.upper_bound(),
            "argmax_disagreements": result.score.evaluation.argmax_disagreements,
            "stop": format!("{:?}", result.stop),
            "printout": reading,
        }));
        start = result.program;
    }
    let report = json!({"model": imported.name, "kind": imported.kind, "export": imported.record, "rows": imported.contract.family.rows, "rungs": rungs});
    write(&out, "printouts.json", &serde_json::to_string(&report).map_err(|e| e.to_string())?)
}
