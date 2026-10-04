//! Verified minimum description cost within an explicit finite candidate bank (#2951).
//!
//! `mpd_candidate_frontier_2951 EXPORT_DIR SPEC.json BANK.json OUT_DIR budget=N max_bank=N [KEY=VALUE ...]`
//!
//! BANK.json is a list of {"label":"name", "artifact":"relative/or/absolute.bin"}.
//! Paths are relative to BANK.json. The native artifact is always included. Optional
//! accounts=DIR adds the complete Cartesian bank of MLP accounts, with the native
//! computation as an alternative at every layer. max_bank bounds generation before
//! evaluation; budget bounds distinct messages assessed in description-cost order.
//! A zero gap proves optimality within this bank and these tested inputs only.
//!
//! Other keys: local_export=EXPORT_DIR, local=4, context=512, ascent=0,
//! deltas=0.05,0.1,0.2, epsilons=0.01,0.03,0.1, parallel=8, batch=1024,
//! groups= (all declared episode groups), cuda_share_native=0 (opt-in exact GPU parameter reuse),
//! cuda_local=0 (opt-in CUDA native-parent graft; requires backend=cuda, ascent=0).
//! SHA-256 input manifests require sha256sum
//! or shasum on PATH. No greedy feasibility pruning is used.

use gam_mpd::acceptance::{Ascent, Assessment, Constraint, CostCache, Local, SlotDomain, assess_once};
use gam_mpd::artifact::Artifact;
use gam_mpd::candidate_frontier::{Candidate, account_bank, frontier};
use gam_mpd::counterfactual::{Decoder, Spec, passages};
use gam_mpd::import::import_language_model;
use gam_mpd::decoded_intern::DecodedOperatorInterner;
use gam_mpd::operator_program::{FamilyInputs, SequenceLayout, SlotValues};
use gam_mpd::proposals::AccountProposer;
use gam_mpd::run_check::{LanguageRun, layer_nodes, split_sites};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BankEntry {
    label: String,
    artifact: PathBuf,
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn floats(text: &str) -> Result<Vec<f64>, String> {
    let values = text.split(',').map(|s| s.parse::<f64>().map_err(|e| format!("{s}: {e}"))).collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("tolerance lists must be nonempty, finite and nonnegative".into());
    }
    Ok(values)
}

fn family(sequences: &[Vec<u32>], count: usize, context: usize) -> Result<FamilyInputs, String> {
    if count == 0 || context == 0 || sequences.len() < count || sequences.iter().take(count).any(|r| r.len() < context) {
        return Err(format!("local family needs {count} nonempty sequences of {context} tokens"));
    }
    let (mut tokens, mut sequence, mut position) = (Vec::new(), Vec::new(), Vec::new());
    for (s, row) in sequences.iter().take(count).enumerate() {
        for (p, t) in row.iter().take(context).enumerate() {
            tokens.push(*t);
            sequence.push(u32::try_from(s).map_err(|e| e.to_string())?);
            position.push(u32::try_from(p).map_err(|e| e.to_string())?);
        }
    }
    Ok(FamilyInputs { rows: tokens.len(), slots: vec![SlotValues::Tokens(tokens)], layout: Some(SequenceLayout { sequence, position }) })
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { base.join(path) }
}

fn sha256(path: &Path) -> Result<String, String> {
    for (program, options) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        match Command::new(program).args(options).arg("--").arg(path).output() {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
                let hash = stdout.split_whitespace().next().ok_or("empty SHA-256 output")?;
                if hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Ok(hash.to_ascii_lowercase());
                }
                return Err(format!("{program}: invalid SHA-256 output"));
            }
            Ok(output) => {
                return Err(format!("{}: {}", path.display(), String::from_utf8_lossy(&output.stderr)));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("{program}: {e}")),
        }
    }
    Err("input provenance needs sha256sum or shasum on PATH".into())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_candidate_frontier_2951 EXPORT_DIR SPEC.json BANK.json OUT_DIR budget=N max_bank=N [KEY=VALUE ...]";
    if args.len() < 5 {
        return Err(usage.into());
    }
    let (export, spec_path, bank_path, out) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]), PathBuf::from(&args[3]), PathBuf::from(&args[4]));
    let mut keys = BTreeMap::new();
    for argument in &args[5..] {
        let (k, v) = argument.split_once('=').ok_or_else(|| format!("expected KEY=VALUE: {argument}"))?;
        if keys.insert(k.to_string(), v.to_string()).is_some() {
            return Err(format!("duplicate option {k}"));
        }
    }
    let allowed = [
        "budget",
        "max_bank",
        "accounts",
        "local_export",
        "local",
        "context",
        "ascent",
        "deltas",
        "epsilons",
        "parallel",
        "batch",
        "groups",
        "backend",
        "cuda_trace_bytes",
        "cuda_share_native",
        "cuda_local",
        "local_kl",
    ];
    if let Some(k) = keys.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(format!("unknown option {k}"));
    }
    let key = |k: &str, default: &str| keys.get(k).cloned().unwrap_or_else(|| default.to_string());
    let number = |k: &str, default: usize| -> Result<usize, String> { key(k, &default.to_string()).parse::<usize>().map_err(|e| format!("{k}: {e}")) };
    let required = |k: &str| -> Result<usize, String> {
        keys.get(k).ok_or_else(|| format!("declare {k}=N explicitly"))?.parse::<usize>().map_err(|e| format!("{k}: {e}"))
    };
    let (budget, max_bank) = (required("budget")?, required("max_bank")?);
    let (context, count, batch, parallel) = (number("context", 512)?, number("local", 4)?, number("batch", 1024)?, number("parallel", 8)?);
    if max_bank == 0 || batch == 0 || parallel == 0 {
        return Err("max_bank, batch and parallel must be positive".into());
    }
    let (deltas, epsilons) = (floats(&key("deltas", "0.05,0.1,0.2"))?, floats(&key("epsilons", "0.01,0.03,0.1"))?);
    let constraints: Vec<Constraint> = deltas.iter().flat_map(|&local| epsilons.iter().map(move |&run| Constraint { local, run })).collect();
    let entries: Vec<BankEntry> = serde_json::from_value(read_json(&bank_path)?).map_err(|e| format!("{}: {e}", bank_path.display()))?;
    if entries.len().checked_add(1).ok_or("bank size overflow")? > max_bank {
        return Err("explicit bank plus native exceeds max_bank".into());
    }
    let started = Instant::now();
    let mut input_paths = BTreeSet::from([bank_path.clone(), spec_path.clone(), export.join("export.json")]);
    let export_record = read_json(&export.join("export.json"))?;
    for name in export_record["files"].as_object().ok_or("export has no files")?.keys() {
        input_paths.insert(export.join(format!("{name}.f64")));
    }
    let spec_record = read_json(&spec_path)?;
    let spec_dir = spec_path.parent().unwrap_or(Path::new("."));
    for edit in spec_record["edits"].as_object().into_iter().flat_map(|m| m.values()) {
        for name in ["left", "right"] {
            input_paths.insert(resolve(spec_dir, Path::new(edit[name].as_str().ok_or("edit lacks a factor file")?)));
        }
    }
    let decoder = Decoder::from_export(&export)?;
    let mut spec = Spec::load(&spec_path, &decoder)?;
    let groups: Vec<String> = key("groups", "").split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
    for group in &groups {
        if !spec.episodes.iter().any(|e| &e.group == group) {
            return Err(format!("unknown episode group {group}"));
        }
    }
    if !groups.is_empty() {
        spec.episodes.retain(|e| groups.contains(&e.group));
    }
    if spec.episodes.is_empty() || spec.rows == 0 {
        return Err("declare at least one run episode with positive run rows".into());
    }
    let run_passages = passages(&export, spec.rows)?;
    let imported = import_language_model(&export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let run = LanguageRun::new(&decoder, &native, &spec, &run_passages, parallel)?;
    let backend = key("backend", "cpu");
    let cuda_local = number("cuda_local", 0)?;
    if cuda_local > 1 || (cuda_local == 1 && backend != "cuda") {
        return Err("cuda_local must be 0 or 1; 1 requires backend=cuda".into());
    }
    let mut local_device = None;
    let run = match backend.as_str() {
        "cpu" => run,
        "cuda" => {
            let limit = keys.get("cuda_trace_bytes").ok_or("backend=cuda requires cuda_trace_bytes=N")?.parse::<usize>().map_err(|e| e.to_string())?;
            let device =
                gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("required CUDA device unavailable")?;
            if cuda_local == 1 { local_device = Some((device.clone(), limit)); }
            run.with_cuda(device, limit)?
        }
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let local_export = PathBuf::from(key("local_export", &export.display().to_string()));
    input_paths.extend([local_export.join("export.json"), local_export.join("tokens.f64")]);
    let local_family = family(&passages(&local_export, context)?, count, context)?;
    let ascent_evaluations = number("ascent", 0)?;
    let ascent = if ascent_evaluations == 0 {
        None
    } else {
        let vocabulary = native.declarations.domains.first().ok_or("native has no token domain")?.size;
        let vocab = u32::try_from(vocabulary).map_err(|e| e.to_string())?;
        Some(Ascent { domain: vec![SlotDomain::Tokens((0..vocab).collect())], pool: Vec::new(), evaluations: ascent_evaluations })
    };
    let local = Local::new(&native, local_family.clone(), ascent, batch);
    let local = match local_device { Some((device, limit)) => local.with_cuda(device, limit)?, None => local };
    // Decode the native source once: exact interface/body interning then shares
    // unchanged tensors across index compaction, ignoring disposable decoder labels.
    let native_artifact = Artifact::native(&native)?.f32_literals()?;
    let start = Artifact::from_bytes(&native_artifact.to_bytes()?, &native.declarations)?;
    drop(native_artifact);
    let share_native = number("cuda_share_native", 0)?;
    if share_native > 1 || (share_native == 1 && backend != "cuda") {
        return Err("cuda_share_native must be 0 or 1; 1 requires backend=cuda".into());
    }
    let run = if share_native == 1 { run.with_cuda_native_source(&start)? } else { run };
    let interner = DecodedOperatorInterner::new(&start)?;
    let mut candidates = vec![Candidate { label: "native".into(), artifact: start.clone() }];
    let mut labels = BTreeSet::from(["native".to_string()]);
    let bank_dir = bank_path.parent().unwrap_or(Path::new("."));
    for entry in entries {
        if entry.label.is_empty() || !labels.insert(entry.label.clone()) {
            return Err(format!("empty or duplicate candidate label {}", entry.label));
        }
        let path = resolve(bank_dir, &entry.artifact);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut artifact = Artifact::from_bytes(&bytes, &native.declarations)?;
        drop(bytes);
        let shared = interner.intern(&mut artifact);
        eprintln!("loaded {}: {shared}/{} operators shared with decoded native", entry.label, artifact.program.operators.len());
        candidates.push(Candidate { label: entry.label, artifact });
        input_paths.insert(path);
    }
    let mut account_count = 0;
    if let Some(directory) = keys.get("accounts") {
        let dir = Path::new(directory);
        let layers = layer_nodes(&native, decoder.layers())?;
        let first = layers.first().ok_or("accounts need a model with layers")?;
        let d_in = native.node_interface(first.normed).map_err(|e| e.to_string())?.width();
        let d_out = native.node_interface(first.mlp).map_err(|e| e.to_string())?.width();
        let proposer = AccountProposer::load(dir, layers, d_in, d_out)?;
        account_count = proposer.accounts.len();
        for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let Some(stem) = path.file_name().and_then(|s| s.to_str()).and_then(|s| s.strip_suffix(".rules.json")) else {
                continue;
            };
            if !stem.starts_with('L') {
                continue;
            }
            input_paths.insert(path.clone());
            for suffix in ["reads.f64", "writes.f64", "offset.f64"] {
                let sibling = dir.join(format!("{stem}.{suffix}"));
                if sibling.exists() {
                    input_paths.insert(sibling);
                }
            }
        }
        let available = max_bank - (candidates.len() - 1);
        for candidate in account_bank(&start, &proposer, available)?.into_iter().skip(1) {
            if !labels.insert(candidate.label.clone()) {
                return Err(format!("duplicate generated label {}", candidate.label));
            }
            candidates.push(candidate);
        }
    }
    let supplied_candidates = candidates.len();
    let loaded_seconds = started.elapsed().as_secs_f64();
    let timer = Instant::now();
    let mut inputs = Vec::new();
    for path in input_paths {
        let canonical = std::fs::canonicalize(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        inputs.push(json!({"path": canonical.display().to_string(), "bytes": std::fs::metadata(&canonical).map_err(|e| e.to_string())?.len(), "sha256": sha256(&canonical)?}));
    }
    let hashing_seconds = timer.elapsed().as_secs_f64();
    eprintln!("{} candidate messages supplied, {} episodes, {} local rows", supplied_candidates, spec.episodes.len(), local_family.rows);
    let timer = Instant::now();
    let evaluated = frontier(&local, &run, candidates, &constraints, budget)?;
    let frontier_seconds = timer.elapsed().as_secs_f64();
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let local_kl_enabled = match key("local_kl", "0").as_str() {
        "0" => false, "1" => true, _ => return Err("local_kl must be 0 or 1".into()),
    };
    let mut isolated_replays = BTreeMap::new();
    let mut details = Vec::new();
    let mut replay_cache = CostCache::default();
    let mut replayed: BTreeMap<usize, (PathBuf, usize, String, Assessment)> = BTreeMap::new();
    let timer = Instant::now();
    for point in &evaluated.points {
        let selected = if let Some(index) = point.selected {
            let candidate = &evaluated.bank[index];
            if let std::collections::btree_map::Entry::Vacant(entry) = replayed.entry(index) {
                let bytes = candidate.artifact.to_bytes()?;
                let decoded = Artifact::from_bytes(&bytes, &native.declarations)?;
                let measured = assess_once(&local, &run, &decoded, point.constraint, &mut replay_cache)?;
                if local_kl_enabled {
                    isolated_replays.insert(index, gam_mpd::local_kl::isolated_downstream_kl(&native, &decoded, &local_family, batch)?);
                }
                let path = out.join(format!("artifact.{index}.bin"));
                std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
                let hash = sha256(&path)?;
                entry.insert((path, bytes.len(), hash, measured));
            }
            let (path, byte_count, hash, measured) = replayed.get(&index).ok_or("missing replay evidence")?;
            let local_fidelity = measured.local.with_tolerance(point.constraint.local)?;
            let run_fidelity = measured.run.with_tolerance(point.constraint.run)?;
            if local_fidelity.verdict() != gam_mpd::precision::FidelityVerdict::Meets
                || run_fidelity.verdict() != gam_mpd::precision::FidelityVerdict::Meets
                || Some(measured.cost.total()) != point.upper_cost
            {
                return Err(format!("selected candidate {} failed saved-byte verification", candidate.label));
            }
            Some(json!({"index": index, "label": candidate.label, "artifact": path.display().to_string(), "bytes": byte_count, "sha256": hash,
                "cost_bits": measured.cost.total(), "replay_index": index,
                "local_verdict": format!("{:?}", local_fidelity.verdict()), "run_verdict": format!("{:?}", run_fidelity.verdict())}))
        } else {
            None
        };
        details.push(json!({"point": point, "selected": selected}));
    }
    let replay_seconds = timer.elapsed().as_secs_f64();
    let mut report = json!({
        "execution": {"backend": run.backend_name(), "local": local.backend_name(), "local_numerical_scope": "comparison-rounding intervals on executed values; not CPU/CUDA execution-equivalence certificates", "teacher": "immutable cached CPU residuals and native effects", "readout_and_KL": "CPU f64", "cuda_episode_parallelism": 1, "teacher_parallelism": parallel, "teacher_cache_memory": "one final residual matrix per episode plus scalar native effects; initialization also retains clean native passage traces and requested donor rows",
            "cuda_trace_limit_scope": "intermediate activation estimate only; excludes operators, attention workspaces, edit masks and allocator overhead",
            "cuda_native_source_enabled": run.cuda_native_sharing(), "cuda_native_source_memory": "when enabled: one immutable native parameter resident plus current candidate changed parameters; no native source trace or donor cache",
            "cuda_upload_scope": if run.cuda_native_sharing() { "one explicit decoded native source per Run; exact re-interning shares parameters across candidates; candidate bases and episode/donor forks are independent; requested donor rows downloaded then reuploaded for mixes" } else { "one base per candidate; episode/donor forks share unchanged operator tensors by Arc identity and role; requested donor rows downloaded then reuploaded for mixes" }},
        "isolated_kl_diagnostic": {"enabled": local_kl_enabled, "backend": "CPU f64", "scope": "selected saved-byte artifacts only; one block on native parents, then native downstream; comparison error excludes neural-forward arithmetic; no acceptance threshold"},
        "local_definition": "maximum tested per-row Euclidean write error divided by RMS native-write row norm over declared family",
        "scope": "explicit finite candidate bank; tested local inputs and declared counterfactual episodes only",
        "objective": "min C(P) subject to D_local <= delta and D_run <= epsilon",
        "optimality": "gap zero proves optimum within this bank only; unresolved, failed and unevaluated candidates remain in the lower bound",
        "export": export.display().to_string(), "spec": spec_path.display().to_string(), "bank_manifest": bank_path.display().to_string(),
        "options": keys, "budget": budget, "max_bank": max_bank, "supplied_candidates": supplied_candidates,
        "distinct_candidates": evaluated.bank.len(), "measured_candidates": evaluated.measured_candidates, "account_count": account_count,
        "frontier_stage_seconds": evaluated.timing,
        "bank": evaluated.bank.iter().enumerate().map(|(index, c)| json!({"index": index, "label": c.label})).collect::<Vec<_>>(),
        "assessments": evaluated.assessments.iter().enumerate().map(|(index, result)| match result {
            None => json!({"index": index, "state": "unevaluated"}),
            Some(Err(error)) => json!({"index": index, "state": "failed", "error": error}),
            Some(Ok(measured)) => json!({"index": index, "state": "measured", "cost": measured.cost,
                "local_measure": measured.local_measure, "run_measure": measured.run_measure,
                "local_bounds": {"lower": measured.local.status().lower_bound(), "upper": measured.local.status().upper_bound()},
                "run_bounds": {"lower": measured.run.status().lower_bound(), "upper": measured.run.status().upper_bound()}}),
        }).collect::<Vec<_>>(),
        "local_export": local_export.display().to_string(), "local_sequences": count, "context": context, "local_rows": local_family.rows,
        "ascent_evaluations_per_step": ascent_evaluations, "run_rows": spec.rows, "episodes": spec.episodes.len(), "groups": spec.episodes.iter().map(|e| e.group.clone()).collect::<BTreeSet<_>>(),
        "selected_replays": replayed.iter().map(|(index, (path, byte_count, hash, measured))| json!({
            "index": index, "artifact": path.display().to_string(), "bytes": byte_count, "sha256": hash, "cost": measured.cost,
            "isolated_downstream_kl": isolated_replays.get(index),
            "local_measure": measured.local_measure, "run_measure": measured.run_measure,
            "local_bounds": {"lower": measured.local.status().lower_bound(), "upper": measured.local.status().upper_bound()},
            "run_bounds": {"lower": measured.run.status().lower_bound(), "upper": measured.run.status().upper_bound()}
        })).collect::<Vec<_>>(),
        "input_hash_algorithm": "SHA-256", "input_files": inputs, "details": details,
        "seconds": {"load_and_generation": loaded_seconds, "input_hashing": hashing_seconds, "frontier": frontier_seconds, "selected_replay": replay_seconds, "total": started.elapsed().as_secs_f64()}
    });
    report["run_stage_seconds"] = serde_json::to_value(run.timing()).map_err(|e| e.to_string())?;
    report["run_stage_timing_scope"] = json!("cumulative across all assessments and saved-byte replays; parallel donor/episode durations summed, not additive wall time; CUDA includes transfer to host, not kernel-only timing");
    let path = out.join("report.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("{} distinct candidates, {} measured; {}", evaluated.bank.len(), evaluated.measured_candidates, path.display());
    Ok(())
}
