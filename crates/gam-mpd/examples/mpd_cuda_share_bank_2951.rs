//! Prepare a diagnostic bank for fresh/shared CUDA parity, never a quality bank.
//! Usage: mpd_cuda_share_bank_2951 EXPORT_DIR SPEC.json CANDIDATE.bin OUT_DIR
//! Native is supplied by the frontier driver; the two bank members are exact saved
//! candidate bytes and the same decoded candidate with permuted operator indices
//! and ordered write exceptions.
use gam_mpd::artifact::{Artifact, Exception, contexts};
use gam_mpd::counterfactual::passages;
use gam_mpd::import::import_language_model;
use gam_mpd::operator_program::{FamilyInputs, SequenceLayout, SlotValues};
use gam_mpd::run_check::split_sites;
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("mpd_cuda_share_bank_2951 EXPORT_DIR SPEC.json CANDIDATE.bin OUT_DIR".into());
    }
    let export = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[4]);
    let spec: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&args[2]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let rows = spec["rows"].as_u64().ok_or("spec needs rows")? as usize;
    let sequences = passages(&export, rows)?;
    let first = sequences.first().ok_or("no passages")?;
    if rows == 0 || first.len() < rows {
        return Err("positive complete passage required".into());
    }
    let imported = import_language_model(&export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let bytes = std::fs::read(&args[3]).map_err(|e| e.to_string())?;
    let mut artifact = Artifact::from_bytes(&bytes, &native.declarations)?;
    artifact.validate_coverage(&native)?;
    if !artifact.derived.is_empty() {
        return Err(
            "diagnostic permutation currently requires materialized operators without derivations"
                .into(),
        );
    }
    let family = FamilyInputs {
        rows,
        slots: vec![SlotValues::Tokens(first[..rows].to_vec())],
        layout: Some(SequenceLayout {
            sequence: vec![0; rows],
            position: (0..rows).map(|r| r as u32).collect(),
        }),
    };
    let original = artifact.execute(&family)?.values[artifact.program.output].clone();
    if original.iter().any(|v| !v.is_finite()) {
        return Err("source candidate has nonfinite diagnostic outputs".into());
    }
    let operators: Vec<_> = (0..artifact.program.operators.len()).rev().collect();
    artifact.program.operators.reverse();
    let nodes: Vec<_> = (0..artifact.program.nodes.len()).collect();
    let bases: Vec<_> = (0..artifact.program.bases.len()).collect();
    let rules: Vec<_> = (0..artifact.program.rules.len()).collect();
    for node in &mut artifact.program.nodes {
        gam_mpd::operator_program::remap_node(node, &nodes, &operators, &bases, &rules);
    }
    for rule in &mut artifact.program.rules {
        let body: Vec<_> = (0..rule.nodes.len()).collect();
        for node in &mut rule.nodes {
            gam_mpd::operator_program::remap_node(node, &body, &operators, &bases, &rules);
        }
    }
    artifact.program.interfaces().map_err(|e| e.to_string())?;
    let permuted = artifact.execute(&family)?.values[artifact.program.output].clone();
    if !original
        .iter()
        .zip(permuted.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        return Err("operator permutation changed candidate output bits".into());
    }
    let block = artifact
        .blocks
        .first()
        .ok_or("candidate must replace at least one block")?;
    let (write, native_write) = (block.write, block.native_write);
    let context = contexts(&family).pop().ok_or("no causal context")?;
    // Large opposite additions deliberately expose order/coalescing bugs. The
    // final small addition remains meaningful after cancellation at column zero.
    for value in [1e20f32, -1e20f32, 0.375f32] {
        artifact.exceptions.push(Exception {
            context: context.clone(),
            node: write,
            column: 0,
            value,
        });
    }
    let exceptional = artifact.to_bytes()?;
    Artifact::from_bytes(&exceptional, &native.declarations)?.validate_coverage(&native)?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("candidate.bin"), bytes).map_err(|e| e.to_string())?;
    std::fs::write(out.join("ordered-exceptions.bin"), exceptional).map_err(|e| e.to_string())?;
    let bank = json!([
        {"label": "candidate", "artifact": "candidate.bin"},
        {"label": "candidate-permuted-ordered-exceptions", "artifact": "ordered-exceptions.bin"}
    ]);
    std::fs::write(
        out.join("BANK.json"),
        serde_json::to_vec_pretty(&bank).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        out.join("diagnostic.json"),
        serde_json::to_vec_pretty(&json!({
            "scope": "CUDA parameter-sharing parity only; not the quality protocol",
            "candidate": args[3], "spec": args[2], "passage": 0, "row": rows-1,
            "native_write": native_write, "candidate_write": write, "column": 0,
            "operator_permutation": "reverse all indices; node and Rule references remapped; literals unchanged",
            "permutation_clean_output_bit_parity": true,
            "ordered_additions": [1e20, -1e20, 0.375]
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
