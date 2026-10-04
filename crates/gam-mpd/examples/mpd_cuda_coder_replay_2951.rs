//! Required-CUDA actual fitter capture/replay. Cache remains opt-in.
//! capture TRAIN OUT SITE; replay FIXTURE rows=8 scratch_MiB=512 slots=4 pairs=6
use gam_mpd::coder_capture::{Fixture, sha256};
use gam_gpu::{GpuPolicy, configure_global_policy};
use gam_gpu::tensor::{CodeRowsWorkspace, Device};
use ndarray::{Array1, Array2, s};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

fn source_hash(text: &str) -> Result<String, String> {
    let mut command = Command::new("sha256sum");
    command.stdin(Stdio::piped()).stdout(Stdio::piped());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Command::new("shasum").args(["-a", "256"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|e| e.to_string())?,
        Err(e) => return Err(e.to_string()),
    };
    child.stdin.take().ok_or("missing hash stdin")?.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() { return Err("compiled-source hashing failed".into()); }
    let stdout = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let hash = stdout.split_whitespace().next().ok_or("missing hash")?;
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) { return Err("invalid source hash".into()); }
    Ok(hash.into())
}

// CPU-only summary of interrupted runs: only complete frozen AB/BA pairs enter
// aggregate timing statistics; every unpaired measurement remains explicit.
fn summarize(path: &Path, expected_pairs: usize) -> Result<(), String> {
    let mut trials = BTreeMap::<u64, [Option<f64>; 2]>::new();
    for line in std::fs::read_to_string(path).map_err(|e| e.to_string())?.lines() {
        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if value["kind"] != "measured" { continue; }
        if value["bitwise_reference_equal"] != true { return Err("non-equivalent measured output".into()); }
        let trial = value["trial"].as_u64().ok_or("missing trial")?;
        let cached = value["cached"].as_bool().ok_or("missing cached flag")?;
        let seconds = value["seconds"].as_f64().ok_or("missing time")?;
        if !seconds.is_finite() || seconds <= 0.0 { return Err("invalid time".into()); }
        if trials.entry(trial).or_insert([None, None])[usize::from(cached)].replace(seconds).is_some() { return Err("duplicate measured variant".into()); }
    }
    let mut complete = [Vec::new(), Vec::new()];
    let mut incomplete = Vec::new();
    for (trial, pair) in &trials {
        match pair { [Some(a), Some(b)] => { complete[0].push(*a); complete[1].push(*b); }, _ => incomplete.push(json!({"trial":trial,"seconds":pair})) }
    }
    let count = complete[0].len();
    if count > expected_pairs { return Err("more pairs than frozen protocol".into()); }
    let samples: Vec<_> = complete.iter_mut().enumerate().map(|(i, s)| {
        s.sort_by(f64::total_cmp);
        let median = if s.is_empty() { None } else if s.len()%2==0 { Some((s[s.len()/2-1]+s[s.len()/2])/2.0) } else { Some(s[s.len()/2]) };
        json!({"cached":i==1,"samples_sorted":s,"median":median,"min":s.first(),"max":s.last()})
    }).collect();
    println!("{}", json!({"kind":"paired_partial_summary","expected_pairs":expected_pairs,"complete_pairs":count,"incomplete_pairs":incomplete,"samples":samples,"scope":"original cache comparison; complete measured pairs only; no full256 throughput claim"}));
    Ok(())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).ok_or("capture TRAIN OUT SITE; replay FIXTURE [KEY=VALUE]")?;
    if mode == "summarize" {
        if args.len() != 4 { return Err("summarize JSONL FROZEN_PAIR_COUNT".into()); }
        return summarize(Path::new(&args[2]), args[3].parse().map_err(|e| format!("pair count: {e}"))?);
    }
    let offset = match mode.as_str() { "capture" => 5, "replay" => 3, _ => return Err("expected capture/replay".into()) };
    if args.len() < offset { return Err("missing positional arguments".into()); }
    let mut keys = BTreeMap::new();
    let allowed = if mode == "capture" { vec!["sequences", "context", "draws", "n", "seed"] } else { vec!["rows", "scratch_MiB", "slots", "pairs", "comparison", "cache"] };
    for argument in &args[offset..] {
        let (k, v) = argument.split_once('=').ok_or("expected KEY=VALUE")?;
        if !allowed.contains(&k) || keys.insert(k, v).is_some() { return Err(format!("unknown/duplicate {k}")); }
    }
    let count = |k: &str, default: usize| -> Result<usize, String> { keys.get(k).map_or(Ok(default), |v| v.parse().map_err(|e| format!("{k}: {e}"))) };
    configure_global_policy(GpuPolicy::Required);
    gam_mpd::core_device::choose(gam_mpd::core_device::Choice::Float64);
    let device = Device::accelerator(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("required CUDA device absent")?;
    if !device.float64() { return Err("required CUDA f64 device absent".into()); }
    if mode == "capture" {
        use gam_mpd::blocks::Generic;
        use gam_mpd::describe::{Geometry, Metric, Structured, Tiered, declared_charts};
        use gam_mpd::masked::{Library, matrix, sites};
        let (train, out, name) = (Path::new(&args[2]), Path::new(&args[3]), &args[4]);
        if out.exists() { return Err("capture output must be fresh".into()); }
        let (sequences, context, draws, seed) = (count("sequences", 2)?, count("context", 128)?, count("draws", 1)?, count("seed", 0x517E)? as u64);
        let observations = keys.get("n").map_or(Ok(1e5), |v| v.parse::<f64>().map_err(|e| e.to_string()))?;
        if sequences == 0 || context == 0 || draws == 0 || !observations.is_finite() || observations <= 0.0 { return Err("invalid capture settings".into()); }
        let mut source_hashes = BTreeMap::new();
        for (name, source) in [
            ("driver", include_str!("mpd_cuda_coder_replay_2951.rs")), ("site_fit", include_str!("../src/site_fit.rs")),
            ("core_device", include_str!("../src/core_device.rs")), ("sparse_code", include_str!("../src/sparse_code.rs")),
            ("tensor", include_str!("../../gam-gpu/src/tensor.rs")), ("fixture", include_str!("../src/coder_capture.rs")),
            ("describe", include_str!("../src/describe.rs")), ("pieces", include_str!("../src/pieces.rs")),
        ] { source_hashes.insert(name, source_hash(source)?); }
        let record: Value = serde_json::from_slice(&std::fs::read(train.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut inputs = BTreeMap::new();
        inputs.insert("export.json".to_string(), sha256(&train.join("export.json"))?);
        for name in record["files"].as_object().ok_or("missing export files")?.keys() { let file = format!("{name}.f64"); inputs.insert(file.clone(), sha256(&train.join(file))?); }
        let started = Instant::now();
        let imported = gam_mpd::import::import_language_model(train, sequences, context)?;
        let model = &imported.program;
        let site = sites(model).into_iter().find(|site| &site.name == name).ok_or("unknown site")?;
        let w = matrix(model, &site)?;
        let batches: Vec<_> = (0..sequences).map(|s| imported.contract.family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>())).collect();
        eprintln!("capture {name}: sample real native-parent fit reads/Fisher on {sequences}×{context} tokens");
        let explanation = gam_mpd::explanation::Explanation { observations, sites: Vec::new() };
        let sample = gam_mpd::core_device::samples(&device, model, (None, &explanation), std::slice::from_ref(&site), &batches, draws, seed)?.swap_remove(0);
        let statistics = gam_mpd::pieces::Site { w: w.clone(), second_moment: sample.second_moment.clone(), mean: Array1::zeros(w.ncols()), fisher: sample.fisher.clone() };
        let (writers, readers) = declared_charts(model, &site)?;
        let describe = Tiered { cheap: Generic::new(std::slice::from_ref(&statistics), observations), exact: Structured::new(vec![Geometry::new(Metric::of(&statistics, observations), writers, readers)?]) };
        let units = site.reads.len() == 1 && matches!(model.nodes[site.reads[0]], gam_mpd::operator_program::Node::Pointwise { .. });
        let exact = if units { gam_mpd::pieces::unit_pieces(&w, gam_mpd::pieces::Units::Read) } else { gam_mpd::pieces::fisher_svd(&statistics)? };
        let start = Library { v: exact.v.t().to_owned(), u: exact.u, mean: Array1::zeros(w.ncols()) };
        let settings = gam_mpd::site_fit::Settings { observations, pieces: (w.nrows() + w.ncols()).max(start.v.nrows()), rounds: 2, seed: seed ^ 0xF17 };
        eprintln!("capture {name}: initialize real fit C={} B={} before coding; {:.3}s", settings.pieces, settings.pieces, started.elapsed().as_secs_f64());
        let (products, coder, library) = gam_mpd::site_fit::capture_initial_selection(0, &w, &sample, &describe, settings, Some(&start))?;
        let fixture = Fixture::from_coder(products, &coder)?;
        let provenance = json!({"kind":"fresh actual own-fit initial selection","not_replay_of_job":15676,"site":name,"phase":"initial selection before first coder call","selection_ordinal":0,
            "native_parents":true,"upstream_fitted_replacements":false,"device":device.name(),"GEMM_arithmetic":"F32","coder_arithmetic":"F64","Gram":"f64 fitter ops.k",
            "sequences":sequences,"context":context,"draws":draws,"sample_seed":seed,"fit_seed":settings.seed,"observations_bits":observations.to_bits(),"intended_rounds":settings.rounds,
            "pieces":settings.pieces,"d_in":w.ncols(),"d_out":w.nrows(),"compiled_sources":source_hashes,"input_sha256":inputs,"binary_sha256":sha256(&std::env::current_exe().map_err(|e| e.to_string())?)?,
            "start":"same own-fit Fisher SVD/read-unit policy","capture_seconds":started.elapsed().as_secs_f64()});
        fixture.write(out, provenance, &[("U.f64", &library.u), ("V.f64", &library.v), ("Fisher.f64", &sample.fisher)])?;
        println!("{}", json!({"captured":out,"site":name,"rows":fixture.products.z.nrows(),"columns":fixture.products.z.ncols(),"blocks":fixture.bits.len(),"nodes":fixture.nodes,"seconds":started.elapsed().as_secs_f64()}));
        return Ok(());
    }
    let (fixture, manifest) = Fixture::read(Path::new(&args[2]))?;
    let (rows, slots, pairs) = (count("rows", 8)?, count("slots", 4)?, count("pairs", 6)?);
    let bytes = count("scratch_MiB", 512)?.checked_mul(1 << 20).ok_or("scratch overflow")?;
    if rows == 0 || rows > fixture.products.z.nrows() || slots == 0 || pairs == 0 { return Err("invalid replay bounds".into()); }
    let comparison = keys.get("comparison").copied().unwrap_or("cache");
    if comparison != "cache" && comparison != "fusion" { return Err("comparison must be cache or fusion".into()); }
    let fixed_cache = count("cache", 0)?;
    if fixed_cache > 1 || (comparison == "cache" && keys.contains_key("cache")) { return Err("cache=0|1 applies only to fusion comparison".into()); }
    let variants = if comparison == "fusion" { [(fixed_cache == 1, false), (fixed_cache == 1, true)] } else { [(false, false), (true, false)] };
    let columns = fixture.products.z.ncols(); let blocks = fixture.bits.len();
    for (cached, _) in variants {
        let plan = CodeRowsWorkspace { bytes, max_rows: slots, cache_columns: cached }.plan(rows, columns, blocks, fixture.nodes).map_err(|e| e.to_string())?;
        if plan.slots != slots.min(rows) { return Err("workspace budget cannot provide equal requested slots for both variants".into()); }
        eprintln!("replay fixed plan cached={cached}: {} slots, {} scratch bytes", plan.slots, plan.bytes);
    }
    let up = |matrix: &Array2<f64>| device.upload(matrix.view()).map_err(|e| e.to_string());
    let z = up(&fixture.products.z.slice(s![..rows, ..]).to_owned())?;
    let weights = up(&fixture.products.weights.slice(s![..rows, ..]).to_owned())?;
    let yfy = device.upload_vec(rows, 1, fixture.products.yfy[..rows].to_vec()).map_err(|e| e.to_string())?;
    let gram = up(&fixture.gram)?; let starts = device.upload_indices(&fixture.starts).map_err(|e| e.to_string())?;
    let bits = device.upload_vec(1, blocks, fixture.bits.clone()).map_err(|e| e.to_string())?;
    let warm = fixture.warm.as_ref().map(|w| up(&w.slice(s![..rows, ..]).to_owned())).transpose()?;
    let settings = (fixture.kappa, fixture.nodes, fixture.tolerance);
    eprintln!("replay actual operands: default reference/NVRTC warmup, {rows} rows C={columns} B={blocks} nodes={}", fixture.nodes);
    let mut reference_on = device.zeros(rows, blocks).map_err(|e| e.to_string())?;
    let (reference_upper, reference_lower) = device.code_rows((&z, &weights, &yfy), (&gram, &starts, &bits), warm.as_ref(), settings, &mut reference_on).map_err(|e| e.to_string())?;
    let reference = device.download(&reference_on).map_err(|e| e.to_string())?;
    let mut samples = [Vec::new(), Vec::new()];
    for trial in 0..(2 + pairs) {
        for variant in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
            let (cached, fused) = variants[variant];
            let mut on = device.zeros(rows, blocks).map_err(|e| e.to_string())?;
            eprintln!("replay {} trial={trial} cached={cached} fused={fused} start", if trial < 2 { "warmup" } else { "measured" });
            let workspace = CodeRowsWorkspace { bytes, max_rows: slots, cache_columns: cached };
            let result = if fused {
                device.code_rows_profiled_fused((&z, &weights, &yfy), (&gram, &starts, &bits), warm.as_ref(), settings, &mut on, workspace)
            } else {
                device.code_rows_profiled((&z, &weights, &yfy), (&gram, &starts, &bits), warm.as_ref(), settings, &mut on, workspace)
            };
            let (upper, lower, diagnostics) = result.map_err(|e| e.to_string())?;
            let actual = device.download(&on).map_err(|e| e.to_string())?;
            if actual.iter().map(|v| v.to_bits()).ne(reference.iter().map(|v| v.to_bits())) || upper.iter().map(|v| v.to_bits()).ne(reference_upper.iter().map(|v| v.to_bits()))
                || lower.iter().map(|v| v.to_bits()).ne(reference_lower.iter().map(|v| v.to_bits())) { return Err(format!("bitwise parity failed trial={trial} cached={cached}")); }
            if upper.iter().chain(&lower).any(|v| !v.is_finite()) || upper.iter().zip(&lower).any(|(u,l)| l > u) { return Err("invalid output bounds".into()); }
            let counts = diagnostics.rows.iter().fold([0u64;7], |mut sum,c| { for(i,v) in [c.relaxations,c.sweeps,c.column_computations,c.column_cache_hits,c.rounding_flips,c.explored_nodes,c.sweep_limit_hits].iter().enumerate() { sum[i] += v; } sum });
            let seconds = diagnostics.elapsed.as_secs_f64();
            if trial >= 2 { samples[variant].push(seconds); }
            let gaps: Vec<f64> = upper.iter().zip(&lower).map(|(u,l)|u-l).collect();
            println!("{}", json!({"kind":if trial<2 {"warmup"} else {"measured"},"trial":trial,"cached":cached,"fused":fused,"seconds":seconds,"bitwise_reference_equal":true,
                "rows":rows,"columns":columns,"blocks":blocks,"nodes":fixture.nodes,"kappa_bits":fixture.kappa.to_bits(),"tolerance_bits":fixture.tolerance.to_bits(),"warm":fixture.warm.is_some(),
                "workspace_bytes":diagnostics.workspace_bytes,"slots":diagnostics.concurrent_rows,"counts":counts,"absolute_gaps":gaps}));
        }
    }
    let summaries: Vec<_> = samples.iter_mut().enumerate().map(|(i,s)| { s.sort_by(f64::total_cmp); json!({"cached":variants[i].0,"fused":variants[i].1,"samples_sorted":s,"median":if s.len()%2==0 {(s[s.len()/2-1]+s[s.len()/2])/2.0} else {s[s.len()/2]},"min":s[0],"max":s[s.len()-1]}) }).collect();
    println!("{}",json!({"kind":"summary","device":device.name(),"fixture_manifest_sha256":sha256(&Path::new(&args[2]).join("MANIFEST.json"))?,"fixture_provenance":manifest["provenance"],
        "rows_scope":"explicit prefix of actual full captured rows; no full256 throughput claim","rows":rows,"pairs":pairs,"comparison":comparison,"scratch_budget_bytes":bytes,"slots":slots,
        "timing_scope":"synchronous call incl scratch allocation, launch, result download; excludes extra mask-download parity check; not kernel time","samples":summaries}));
    Ok(())
}
