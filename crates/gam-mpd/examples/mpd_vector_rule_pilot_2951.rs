//! Prepare an explicitly declared full-vector arithmetic source for the resident fitter.
//! WIDTH SEED OUT_DIR. Data/model lineage is frozen by the caller before fitting.
use gam_mpd::{artifact::Artifact, vector_rule_pilot};
use serde_json::json;
use std::path::Path;
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 { return Err("usage: mpd_vector_rule_pilot_2951 WIDTH SEED OUT_DIR".into()); }
    let width: usize = args[1].parse().map_err(|e| format!("width: {e}"))?;
    let seed: u64 = args[2].parse().map_err(|e| format!("seed: {e}"))?;
    // Explicit allocation guard: this driver is a bounded prototype, not a grammar sweep.
    if width > 4096 { return Err("declared driver width cap 4096 exceeded".into()); }
    let out = Path::new(&args[3]);
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    let (program, trainable) = vector_rule_pilot::product(width, seed)?;
    let source = Artifact::native(&program)?.f32_literals()?;
    let bytes = source.to_bytes()?;
    let decoded = Artifact::from_bytes(&bytes, &program.declarations)?;
    if decoded.to_bytes()? != bytes { return Err("source ordinary roundtrip mismatch".into()); }
    std::fs::write(out.join("SOURCE.bin"), bytes).map_err(|e| e.to_string())?;
    std::fs::write(out.join("TRAINABLE.json"), serde_json::to_vec_pretty(&trainable).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let metadata = json!({"scope":"single explicit proposal/runtime vertical slice, not learned reuse evidence", "width":width,"nonlinear_features":width,"arity":2,"argument_coordinates":2*width,"seed":seed,"rule":"Hadamard(Arg0,Arg1)","operators":"two full dense readers with paid nonzero offsets, full dense writer, explicit output offset", "trainable":trainable,"cost_scope":"standalone function only; full native graft must be priced separately", "native_unit_controls":"not represented by learned body coordinates"});
    std::fs::write(out.join("PROPOSAL.json"), serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(())
}
fn main() -> Result<(), String> { run() }
