//! Complete single-substitution attention Copy bank, with no greedy pruning (#2951).
//!
//! `mpd_rule_bank_2951 EXPORT_DIR OUT_DIR max_bank=N`
//! Writes one self-contained artifact per attention head, relative BANK.json entries,
//! and a provenance/scope manifest. max_bank includes the native candidate added by
//! mpd_candidate_frontier_2951. No combinations, match-plane search, data fitting or
//! feasibility selection occur. Optimality can only be claimed within this explicit bank.

use gam_mpd::acceptance::{CostCache, structural_cost};
use gam_mpd::artifact::Artifact;
use gam_mpd::import::import_language_model;
use gam_mpd::proposals::HeadRules;
use gam_mpd::run_check::{layer_nodes, split_sites};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn digest(path: Option<&Path>, text: Option<&str>) -> Result<String, String> {
    for (program, options) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        let mut command = Command::new(program);
        command.args(options).stdout(Stdio::piped()).stderr(Stdio::piped());
        if let Some(path) = path {
            command.arg("--").arg(path);
        } else {
            command.stdin(Stdio::piped());
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("{program}: {e}")),
        };
        if let Some(text) = text {
            child.stdin.take().ok_or("hash process has no stdin")?.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        }
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(format!("{program}: {}", String::from_utf8_lossy(&output.stderr)));
        }
        let stdout = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
        let hash = stdout.split_whitespace().next().ok_or("empty SHA-256 output")?;
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("invalid SHA-256 output from {program}"));
        }
        return Ok(hash.to_ascii_lowercase());
    }
    Err("provenance needs sha256sum or shasum on PATH".into())
}

fn number(config: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(config[key].as_u64().ok_or_else(|| format!("config needs positive integer {key}"))?).map_err(|e| e.to_string())
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("mpd_rule_bank_2951 EXPORT_DIR OUT_DIR max_bank=N".into());
    }
    let (export, out) = (Path::new(&args[1]), Path::new(&args[2]));
    let max_bank = args[3].strip_prefix("max_bank=").ok_or("expected max_bank=N")?.parse::<usize>().map_err(|e| e.to_string())?;
    let export_json = export.join("export.json");
    let record: Value = serde_json::from_slice(&std::fs::read(&export_json).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let config = &record["config"];
    let (layers, heads, kv_heads) = (number(config, "n_layers")?, number(config, "n_heads")?, number(config, "n_kv_heads")?);
    if layers == 0 || heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
        return Err("positive layers and divisible query/KV head counts required".into());
    }
    let count = layers.checked_mul(heads).ok_or("bank size overflow")?;
    let total = count.checked_add(1).ok_or("bank size overflow")?;
    // This guard precedes tensor import, hashes, output creation and any rule computation.
    if total > max_bank {
        return Err(format!("{count} single Copy substitutions plus native exceeds max_bank={max_bank}"));
    }
    if out.exists() {
        return Err(format!("{} already exists; use a fresh bank directory", out.display()));
    }
    let files = record["files"].as_object().ok_or("export files must be an object")?;
    let mut input_hashes = BTreeMap::new();
    input_hashes.insert("export.json".to_string(), digest(Some(&export_json), None)?);
    for name in files.keys() {
        let filename = format!("{name}.f64");
        input_hashes.insert(filename.clone(), digest(Some(&export.join(&filename)), None)?);
    }
    let mut source_hashes = BTreeMap::new();
    for (name, text) in [
        ("mpd_rule_bank_2951.rs", include_str!("mpd_rule_bank_2951.rs")),
        ("proposals.rs", include_str!("../src/proposals.rs")),
        ("rules.rs", include_str!("../src/rules.rs")),
        ("artifact.rs", include_str!("../src/artifact.rs")),
        ("run_check.rs", include_str!("../src/run_check.rs")),
        ("import.rs", include_str!("../src/import.rs")),
        ("acceptance.rs", include_str!("../src/acceptance.rs")),
    ] {
        source_hashes.insert(name, digest(None, Some(text))?);
    }
    let binary_hash = digest(Some(&std::env::current_exe().map_err(|e| e.to_string())?), None)?;
    let imported = import_language_model(export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let nodes = layer_nodes(&native, layers)?;
    if nodes.iter().any(|n| n.queries.len() != heads || n.values.len() != kv_heads) {
        return Err("imported head inventory differs from declared export configuration".into());
    }
    // All candidates share one rounded base; scaling/derivation uses exactly the
    // literals the decoder receives, and a bank does not retain many base copies.
    let start = Artifact::native(&native)?.f32_literals()?;
    let bank = HeadRules::copy_bank(&start, &nodes, heads / kv_heads, max_bank)?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut cache = CostCache::default();
    let (mut entries, mut candidates) = (Vec::new(), Vec::new());
    for (index, proposal) in bank.enumerate() {
        let (layer, head) = (index / heads, index % heads);
        let filename = format!("copy-l{layer:03}-h{head:03}.bin");
        let artifact = proposal?.candidate;
        artifact.validate_coverage(&native)?;
        let bytes = artifact.to_bytes()?;
        let decoded = Artifact::from_bytes(&bytes, &native.declarations)?;
        decoded.validate_coverage(&native)?;
        if decoded.derived.len() != 1 || decoded.blocks.len() != 1 {
            return Err(format!("{filename}: expected one derived operator and one complete attention binding"));
        }
        if decoded.to_bytes()? != bytes {
            return Err(format!("{filename}: decode/re-encode differs"));
        }
        // Decode/re-encode equality pins this price to the transmitted artifact.
        // Reuse the shared base operator cache instead of retaining decoded base
        // operators separately for every bank entry.
        let cost = structural_cost(&artifact, &mut cache)?;
        let path = out.join(&filename);
        std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
        let label = format!("copy blocks.{layer}.o{head}");
        entries.push(json!({"label":label,"artifact":filename}));
        candidates.push(json!({"layer":layer,"head":head,"replaced_operator":format!("blocks.{layer}.o{head}"),
            "artifact":filename,"sha256":digest(Some(&path),None)?,"serialized_bytes":bytes.len(),
            "structural_cost":cost,"structural_total_bits":cost.total(),"derived_count":decoded.derived.len(),
            "binding_count":decoded.blocks.len(),"scale":decoded.derived[0].scale}));
    }
    if entries.len() != count {
        return Err("copy bank generation omitted a declared attention head".into());
    }
    write_json(&out.join("MANIFEST.json"), &json!({
        "schema":"attention-single-copy-bank-v1","bank":"BANK.json","export":export.display().to_string(),
        "scope":{"law":"Copy","layers":"all","heads":"all","n_layers":layers,"query_heads_per_layer":heads,
            "kv_heads_per_layer":kv_heads,"substitutions_per_candidate":1,"combinations_included":false,
            "match_laws_included":false,"feasibility_pruning":false,"data_fitting":false,"max_bank":max_bank,
            "emitted_candidates":count,"evaluator_candidates_including_native":total,"native":"added by evaluator; no native file"},
        "literal_precision":"f32; one rounded base shared; Copy scale fitted only to each head's own native weights",
        "input_sha256":input_hashes,"compiled_source_sha256":source_hashes,"binary_sha256":binary_hash,
        "bank_candidates":candidates,"claim":"finite single-Copy bank only; no global optimum or accepted mechanism claim"
    }))?;
    // BANK is the final completion marker; the evaluator resolves paths relative to it.
    write_json(&out.join("BANK.json"), &Value::Array(entries))?;
    eprintln!("emitted {count} single Copy artifacts; evaluator adds native for {total} candidates");
    Ok(())
}
