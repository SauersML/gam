//! Expand saved legacy Copy templates into paid explicit arithmetic bodies.
//! This is an explicit-body template baseline, not automatic mechanism discovery.
//! EXPORT_DIR SOURCE.bin OUT_DIR. Reads export declarations, never checkpoint weights.
use gam_mpd::acceptance::{CostCache, structural_cost};
use gam_mpd::artifact::{Artifact, OperatorLaw};
use gam_mpd::coder_capture::sha256;
use gam_mpd::operator_program::{Declarations, Domain, Slot};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path, sync::Arc, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("EXPORT_DIR SOURCE.bin OUT_DIR".into());
    }
    let (export, source, out) = (
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    );
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    let metadata: Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if metadata["config"].get("rope_theta").is_none() {
        return Err("a full token-only rotary LM artifact is required".into());
    }
    let vocab = usize::try_from(
        metadata["config"]["vocab"]
            .as_u64()
            .ok_or("missing vocabulary declaration")?,
    )
    .map_err(|e| e.to_string())?;
    let declarations = Declarations {
        parameters: 0,
        domains: vec![Domain { size: vocab }],
        slots: vec![Slot::Token { domain: 0 }],
    };
    let started = Instant::now();
    let saved = std::fs::read(source).map_err(|e| e.to_string())?;
    let original = Artifact::from_bytes(&saved, &declarations)?;
    if original.to_bytes()? != saved {
        return Err("source artifact is not canonical".into());
    }
    drop(saved);
    let copies = original
        .derived
        .iter()
        .filter(|d| matches!(d.law, OperatorLaw::Copy { .. }))
        .count();
    if copies == 0 {
        return Err("source artifact contains no legacy Copy templates".into());
    }
    let before = structural_cost(&original, &mut CostCache::default())?;
    let explicit = original.expand_copy_templates()?;
    if explicit.program.nodes != original.program.nodes
        || explicit.blocks != original.blocks
        || explicit.places != original.places
        || explicit.exceptions != original.exceptions
    {
        return Err("conversion changed executable graph or intervention bindings".into());
    }
    let targets: BTreeSet<usize> = original.derived.iter().map(|d| d.operator).collect();
    if original.program.operators.len() != explicit.program.operators.len()
        || original
            .program
            .operators
            .iter()
            .zip(&explicit.program.operators)
            .enumerate()
            .any(|(i, (a, b))| !targets.contains(&i) && !Arc::ptr_eq(a, b))
    {
        return Err("conversion changed an independently stored native operator".into());
    }
    let mut operators = Vec::new();
    let mut expected = Vec::new();
    for &target in &targets {
        let a = original.program.operators[target].matrix();
        let b = explicit.program.operators[target].matrix();
        if a.dim() != b.dim() {
            return Err("conversion changed an operator shape".into());
        }
        let mut bit_differences = 0usize;
        let mut max_absolute_difference = 0.0_f64;
        for (a, b) in a.iter().zip(&b) {
            if !a.is_finite() || !b.is_finite() {
                return Err("conversion produced a nonfinite operator".into());
            }
            bit_differences += usize::from(a.to_bits() != b.to_bits());
            max_absolute_difference = max_absolute_difference.max((a - b).abs());
        }
        if !max_absolute_difference.is_finite() {
            return Err("operator comparison overflow".into());
        }
        operators.push(json!({"index":target,"shape":a.dim(),"compared_entries":a.len(),"bit_differences":bit_differences,"max_absolute_difference":max_absolute_difference,"bit_identical":bit_differences==0}));
        expected.push((target, b));
    }
    let after = structural_cost(&explicit, &mut CostCache::default())?;
    let additional_definition_bits = after
        .total()
        .checked_sub(before.total())
        .ok_or("explicit body reduced its description price")?;
    let mut bodies = Vec::new();
    for d in &explicit.derived {
        if let OperatorLaw::Expression { body, .. } = &d.law {
            let encoded = body.encode()?;
            if !bodies.contains(&encoded) {
                bodies.push(encoded);
            }
        }
    }
    let legacy_remaining = explicit
        .derived
        .iter()
        .filter(|d| matches!(d.law, OperatorLaw::Copy { .. } | OperatorLaw::Match { .. }))
        .count();
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let file = out.join("explicit.artifact");
    std::fs::write(&file, explicit.to_bytes()?).map_err(|e| e.to_string())?;
    drop(explicit);
    drop(original);
    // Independently recover the full program from saved bytes plus declarations.
    let saved = std::fs::read(&file).map_err(|e| e.to_string())?;
    let decoded = Artifact::from_bytes(&saved, &declarations)?;
    if decoded.to_bytes()? != saved
        || structural_cost(&decoded, &mut CostCache::default())? != after
    {
        return Err("independent replay changed bytes or complete C32".into());
    }
    for (target, matrix) in &expected {
        if decoded.program.operators[*target]
            .matrix()
            .iter()
            .zip(matrix)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("independent replay changed computed operator bits".into());
        }
    }
    let report = json!({"scope":"explicit-body Copy template baseline; conversion is not discovery; fidelity must be evaluated on decoded new bytes","source":source,"source_sha256":sha256(source)?,"export_metadata_sha256":sha256(&export.join("export.json"))?,"checkpoint_weights_read":false,"output":file,"output_sha256":sha256(&file)?,"bytes":saved.len(),"converted_copy_calls":copies,"shared_explicit_bodies":bodies.len(),"remaining_legacy_templates":legacy_remaining,"before":before,"after":after,"before_c32":before.total(),"after_c32":after.total(),"additional_definition_bits":additional_definition_bits,"operators":operators,"graph_and_intervention_bindings_preserved":true,"independent_saved_byte_replay":true,"seconds":started.elapsed().as_secs_f64()});
    std::fs::write(
        out.join("CONVERSION.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let bank = json!([{"label":"legacy Copy template baseline","artifact":source},{"label":"explicit-body Copy template baseline","artifact":file}]);
    std::fs::write(
        out.join("BANK.json"),
        serde_json::to_vec_pretty(&bank).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "expanded {copies} Copy calls into {} shared explicit bodies; C32 +{additional_definition_bits} bits; independent replay passed",
        bodies.len()
    );
    Ok(())
}
