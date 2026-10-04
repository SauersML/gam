//! Assess all prepared native-input head factors without fitting or changing the frozen panel.
//! FIT_EXPORT EVAL_EXPORT SPEC PREPARED_REPORT OUT head_start=N head_count=N
//! train_export=DIR max_bank=385 codec_bytes=N local_source_bytes=N trace_bytes=N
//! readout_resident_bytes=N readout_workspace_bytes=N parallel=N preflight=0|1
//! bank=single_head (default, max_bank385) or uniform_joint (all24 heads, max_bank17).
//! A shard preserves the complete 384-proposal scope with other heads explicitly unmeasured.
use gam_mpd::{
    acceptance::{
        Assessment, Constraint, CostCache, EpisodeScore, Local, RunCheck, StagedAssessment,
        assess_once, assess_once_local_first_with_native_codec, structural_cost,
    },
    artifact::{Artifact, EncodedArtifact},
    coder_capture::sha256,
    counterfactual::{Decoder, Spec, passages},
    import::import_language_model,
    native_readout::Budget,
    operator_program::{FamilyInputs, NativeOperatorCodec, Operator, SequenceLayout, SlotValues},
    precision::{DecodableArtifact, FidelityVerdict},
    proposals::{CopyResidualBank, CopyResidualChoice, HeadApproximation, LowRankProposal},
    run_check::{LanguageRun, layer_nodes, split_sites},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::Path,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::Instant,
};
const RANKS: [usize; 4] = [8, 32, 64, 96];
const FITS: [&str; 2] = ["weight_Frobenius", "training_native_inputs"];
const DELTAS: [f64; 5] = [0.05, 0.1, 0.2, 0.5, 1.0];
const EPSILONS: [f64; 6] = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0];
const SPEC_SHA: &str = "3c05b66324dfb02e33da7eff98784a7ec432123e24fea78018b892efb4256436";
const PANEL_TOKENS_SHA: &str = "9938c3b6995c4c1b9f74b7bf26b7a17a991941952de9a2e6598fec95003ca8cf";
const COMPLETE: usize = 385;
const HEADS: usize = 24;
const CONTEXT: usize = 512;
#[derive(Clone, Copy, PartialEq, Eq)]
enum BankMode {
    SingleHead,
    UniformJoint,
}
impl BankMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "single_head" => Ok(Self::SingleHead),
            "uniform_joint" => Ok(Self::UniformJoint),
            _ => Err("bank must be single_head or uniform_joint".into()),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::SingleHead => "single_head",
            Self::UniformJoint => "uniform_joint",
        }
    }
    fn groups(self) -> Vec<Vec<usize>> {
        let mut groups = vec![Vec::new()]; // Native, not an inferred joint outcome.
        match self {
            Self::SingleHead => groups.extend((0..COMPLETE - 1).map(|i| vec![i])),
            Self::UniformJoint => groups.extend(
                (0..16).map(|variant| (0..HEADS).map(|head| head * 16 + variant).collect()),
            ),
        }
        groups
    }
    fn validate_scope(self, start: usize, count: usize, max_bank: usize) -> Result<(), String> {
        if count == 0
            || start.checked_add(count).is_none_or(|end| end > HEADS)
            || max_bank != self.groups().len()
            || (self == Self::UniformJoint && (start != 0 || count != HEADS))
        {
            return Err(
                "invalid declared bank budget/head range; joint programs require all24 heads"
                    .into(),
            );
        }
        Ok(())
    }
}
#[derive(Clone)]
struct Entry {
    index: usize,
    choice: CopyResidualChoice,
    fit: String,
    filename: String,
    hash: String,
    bytes: u64,
    conditional_cost: Option<u64>,
    diagnostics: Value,
}
fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())
}
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn integer(value: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(
        value[key]
            .as_u64()
            .ok_or_else(|| format!("missing integer {key}"))?,
    )
    .map_err(|e| e.to_string())
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
fn inventory(report: &Value) -> Result<Vec<Entry>, String> {
    if report["ranks"] != json!(RANKS) || report["used_training_rows"] != CONTEXT * 512 {
        return Err("prepared bank requires declared ranks and all 512 training passages".into());
    }
    let heads = report["heads"]
        .as_array()
        .ok_or("prepared heads array required")?;
    if heads.len() != HEADS {
        return Err("prepared inventory must include all 24 heads".into());
    }
    let mut entries = BTreeMap::new();
    let mut seen_heads = BTreeSet::new();
    for head in heads {
        let (layer, h) = (integer(head, "layer")?, integer(head, "head")?);
        if layer >= 4 || h >= 6 || !seen_heads.insert((layer, h)) {
            return Err("duplicate or invalid prepared head".into());
        }
        let points = head["points"]
            .as_array()
            .ok_or("prepared points required")?;
        if points.len() != 16 {
            return Err("each prepared head requires all sixteen variants".into());
        }
        for point in points {
            let rank = integer(point, "rank")?;
            let ri = RANKS
                .iter()
                .position(|r| *r == rank)
                .ok_or("rank outside declared bank")?;
            let family_text = point["family"].as_str().ok_or("family required")?;
            let (fi, family) = match family_text {
                "NativeSvd" => (0, HeadApproximation::NativeSvd),
                "CopyResidual" => (1, HeadApproximation::CopyResidual),
                _ => return Err("unknown prepared family".into()),
            };
            let fit = point["fit"].as_str().ok_or("fit required")?;
            let ti = FITS
                .iter()
                .position(|f| *f == fit)
                .ok_or("unknown prepared fit")?;
            let index = 1 + (layer * 6 + h) * 16 + ri * 4 + fi * 2 + ti;
            let factors = &point["factors"];
            let filename = format!(
                "L{layer}.H{h}.{family_text}.{}.rank{rank}.json",
                if ti == 0 { "weight" } else { "input" }
            );
            let hash = factors["sha256"]
                .as_str()
                .ok_or("every variant requires saved factor hashes")?;
            let bytes = factors["bytes"].as_u64().ok_or("factor bytes required")?;
            if factors["file"] != filename
                || !digest(hash)
                || bytes == 0
                || factors["independent_factor_replay"].as_bool() != Some(true)
                || factors["final_artifact_acceptance"].as_bool() != Some(false)
            {
                return Err("prepared factor filename/hash/replay contract mismatch".into());
            }
            let entry = Entry {
                index,
                choice: CopyResidualChoice {
                    layer,
                    head: h,
                    rank,
                    family,
                },
                fit: fit.into(),
                filename,
                hash: hash.into(),
                bytes,
                conditional_cost: point["C32_bits"].as_u64(),
                diagnostics: point.clone(),
            };
            if entries.insert(index, entry).is_some() {
                return Err("duplicate prepared variant".into());
            }
        }
    }
    if !entries.keys().copied().eq(1..COMPLETE) {
        return Err("incomplete prepared bank".into());
    }
    Ok(entries.into_values().collect())
}
fn load_factor(dir: &Path, entry: &Entry, native: &Operator) -> Result<LowRankProposal, String> {
    let path = dir.join(&entry.filename);
    let maximum = entry
        .choice
        .rank
        .checked_mul(native.rows.width() + native.cols.width())
        .and_then(|v| v.checked_mul(11))
        .and_then(|v| v.checked_add(1024))
        .ok_or("factor size overflow")?;
    if entry.bytes > maximum as u64
        || std::fs::metadata(&path).map_err(|e| e.to_string())?.len() != entry.bytes
        || sha256(&path)? != entry.hash
    {
        return Err("prepared factor size or digest mismatch".into());
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let header: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let root = format!("blocks.{}.o{}", entry.choice.layer, entry.choice.head);
    let name = if entry.fit == FITS[1] {
        format!("{root}.input_fit")
    } else if entry.choice.family == HeadApproximation::CopyResidual {
        format!("{root}.copy_residual")
    } else {
        root
    };
    if integer(&header, "rank")? != entry.choice.rank || header["name"] != name {
        return Err("prepared factor rank/head/fit name mismatch".into());
    }
    let payload: LowRankProposal = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let reconstructed = payload.operator(native)?;
    if serde_json::to_value(LowRankProposal::of(&reconstructed)?).map_err(|e| e.to_string())?
        != header
    {
        return Err("independent prepared factor bits changed".into());
    }
    Ok(payload)
}
fn weight_lineage(train: &Path, fit: &Path, eval: &Path, report: &Value) -> Result<Value, String> {
    let f = read_json(&fit.join("export.json"))?;
    let e = read_json(&eval.join("export.json"))?;
    let t = read_json(&train.join("export.json"))?;
    let manifest = &report["manifest"];
    let checkpoint = manifest["checkpoint_sha256"]
        .as_str()
        .ok_or("checkpoint lineage required")?;
    if !digest(checkpoint)
        || manifest["config"] != f["config"]
        || f["config"] != e["config"]
        || t["config"] != f["config"]
        || t["source"]["checkpoint_sha256"] != checkpoint
        || f["source"]["checkpoint_sha256"] != checkpoint
        || e["source"]["checkpoint_sha256"] != checkpoint
        || manifest["families"].as_array().map(Vec::len) != Some(2)
        || manifest["families"][0]["name"] != "train"
        || manifest["families"][1]["name"] != "eval"
        || manifest["families"][0]["rows"] != CONTEXT * 512
        || manifest["context"] != CONTEXT
        || manifest["identical_train_eval_token_sequences"] != 0
        || manifest["families"][0]["export_sha256"] != sha256(&train.join("export.json"))?
        || manifest["families"][1]["export_sha256"] != sha256(&fit.join("export.json"))?
    {
        return Err("prepared training/checkpoint/config/export lineage mismatch".into());
    }
    for family in manifest["families"].as_array().ok_or("families")? {
        if family["source"]["checkpoint_sha256"] != checkpoint
            || !family["tokens_file_sha256"].as_str().is_some_and(digest)
        {
            return Err("training/reference family hashes or checkpoint mismatch".into());
        }
    }
    let tf = t["files"].as_object().ok_or("train files")?;
    let ff = f["files"].as_object().ok_or("fit files")?;
    let ef = e["files"].as_object().ok_or("eval files")?;
    let names = |files: &serde_json::Map<String, Value>| {
        files
            .keys()
            .filter(|k| {
                ![
                    "tokens",
                    "row_ids",
                    "logits_row0",
                    "logits_logsumexp",
                    "logits_topk_indices",
                    "logits_topk_values",
                ]
                .contains(&k.as_str())
            })
            .cloned()
            .collect::<BTreeSet<_>>()
    };
    let model_files = names(ff);
    if model_files.is_empty() || model_files != names(ef) || model_files != names(tf) {
        return Err("native weight inventory differs".into());
    }
    let mut records = Vec::new();
    for name in model_files {
        if name.contains('/')
            || name.contains('\\')
            || name.contains("..")
            || ff[&name]["shape"] != ef[&name]["shape"]
            || tf[&name]["shape"] != ff[&name]["shape"]
        {
            return Err("native weight name/shape differs".into());
        }
        let shape = ff[&name]["shape"].as_array().ok_or("weight shape")?;
        let count = shape.iter().try_fold(1u64, |a, b| {
            a.checked_mul(
                b.as_u64()
                    .filter(|v| *v > 0)
                    .ok_or("positive weight dimension")?,
            )
            .ok_or("weight size overflow")
        })?;
        let size = count.checked_mul(8).ok_or("weight byte overflow")?;
        let fp = fit.join(format!("{name}.f64"));
        let ep = eval.join(format!("{name}.f64"));
        let tp = train.join(format!("{name}.f64"));
        let fh = sha256(&fp)?;
        let eh = sha256(&ep)?;
        let th = sha256(&tp)?;
        if fh != eh
            || th != fh
            || std::fs::metadata(&tp).map_err(|e| e.to_string())?.len() != size
            || std::fs::metadata(&fp).map_err(|e| e.to_string())?.len() != size
            || std::fs::metadata(&ep).map_err(|e| e.to_string())?.len() != size
            || tf[&name].get("sha256").is_some_and(|v| v != &json!(th))
            || ff[&name].get("sha256").is_some_and(|v| v != &json!(fh))
            || ef[&name].get("sha256").is_some_and(|v| v != &json!(eh))
        {
            return Err(format!("actual native weight mismatch: {name}"));
        }
        records.push(json!({"file":format!("{name}.f64"),"shape":shape,"bytes":size,"train_sha256":th,"fit_sha256":fh,"evaluation_sha256":eh}));
    }
    if manifest["families"][0]["tokens_file_sha256"] != sha256(&train.join("tokens.f64"))?
        || manifest["families"][1]["tokens_file_sha256"] != sha256(&fit.join("tokens.f64"))?
    {
        return Err("reference tokens hash mismatch".into());
    }
    let training = token_prefixes(train, 512)?;
    let evaluation = token_prefixes(eval, 2)?;
    let seen: BTreeSet<_> = training.iter().collect();
    if evaluation.iter().any(|row| seen.contains(row)) {
        return Err("training and scored evaluation share a causal token sequence".into());
    }
    Ok(
        json!({"checkpoint_sha256":checkpoint,"actual_train_scored_eval_shared_sequences":0,"training_export_sha256":sha256(&train.join("export.json"))?,"actual_native_weights_equal":true,"weights":records,
        "fit_export_sha256":sha256(&fit.join("export.json"))?,"evaluation_export_sha256":sha256(&eval.join("export.json"))?,
        "evaluation_tokens_sha256":sha256(&eval.join("tokens.f64"))?,"training_manifest":manifest,
        "excluded_from_weight_comparison":["tokens","row_ids","logits_row0","logits_logsumexp","logits_topk_indices","logits_topk_values"],"extraction_arrays_rehashed":false}),
    )
}
fn token_prefixes(dir: &Path, count: usize) -> Result<Vec<Vec<u32>>, String> {
    let metadata = read_json(&dir.join("export.json"))?;
    let shape = metadata["files"]["tokens"]["shape"]
        .as_array()
        .ok_or("token shape")?;
    if shape.len() != 2 {
        return Err("token matrix required".into());
    }
    let rows =
        usize::try_from(shape[0].as_u64().ok_or("token rows")?).map_err(|e| e.to_string())?;
    let width =
        usize::try_from(shape[1].as_u64().ok_or("token width")?).map_err(|e| e.to_string())?;
    let vocab = integer(&metadata["config"], "vocab")?;
    if count > rows || width < CONTEXT {
        return Err("declared token prefix unavailable".into());
    }
    let bytes = std::fs::read(dir.join("tokens.f64")).map_err(|e| e.to_string())?;
    if rows.checked_mul(width).and_then(|n| n.checked_mul(8)) != Some(bytes.len()) {
        return Err("token byte shape mismatch".into());
    }
    let mut result = Vec::new();
    for row in bytes.chunks_exact(width * 8).take(count) {
        let mut tokens = Vec::new();
        for value in row.chunks_exact(8).take(CONTEXT) {
            let v = f64::from_le_bytes(
                value
                    .try_into()
                    .map_err(|e: std::array::TryFromSliceError| e.to_string())?,
            );
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v >= vocab as f64 {
                return Err("invalid scored/training token".into());
            }
            tokens.push(v as u32);
        }
        result.push(tokens);
    }
    Ok(result)
}
fn validate_activation_manifest(
    report: &Value,
    base: &Artifact,
    layers: &[gam_mpd::run_check::LayerNodes],
) -> Result<(), String> {
    for family in report["manifest"]["families"]
        .as_array()
        .ok_or("activation families")?
    {
        let name = family["name"].as_str().ok_or("activation family name")?;
        let rows = integer(family, "rows")?;
        if rows == 0 || rows % CONTEXT != 0 {
            return Err("invalid native activation family rows".into());
        }
        let heads = family["heads"]
            .as_array()
            .ok_or("activation head inventory")?;
        if heads.len() != HEADS {
            return Err("activation family must identify all native heads".into());
        }
        let mut seen = BTreeSet::new();
        for head in heads {
            let (l, h) = (integer(head, "layer")?, integer(head, "head")?);
            if l >= 4 || h >= 6 || !seen.insert((l, h)) {
                return Err("duplicate native activation head".into());
            }
            let operator = format!("blocks.{l}.o{h}");
            let width = base
                .program
                .operators
                .iter()
                .find(|o| o.name == operator)
                .ok_or("native activation operator")?
                .cols
                .width();
            if head["operator"] != operator
                || head["node"] != layers[l].reads[h]
                || head["shape"] != json!([rows, width])
                || head["file"] != format!("{name}.{l}.{h}.f64")
                || !head["sha256"].as_str().is_some_and(digest)
            {
                return Err("activation read-node/operator/shape/hash mismatch".into());
            }
        }
    }
    Ok(())
}
fn local_family(rows: &[Vec<u32>]) -> Result<FamilyInputs, String> {
    if rows.len() < 2 || rows[..2].iter().any(|r| r.len() < CONTEXT) {
        return Err("full two-passage Local family required".into());
    }
    Ok(FamilyInputs {
        rows: 2 * CONTEXT,
        slots: vec![SlotValues::Tokens(
            rows[..2]
                .iter()
                .flat_map(|r| r[..CONTEXT].iter().copied())
                .collect(),
        )],
        layout: Some(SequenceLayout {
            sequence: (0..2).flat_map(|s| vec![s; CONTEXT]).collect(),
            position: (0..2).flat_map(|_| 0..CONTEXT as u32).collect(),
        }),
    })
}
fn grid() -> Vec<Constraint> {
    DELTAS
        .iter()
        .flat_map(|&local| EPSILONS.iter().map(move |&run| Constraint { local, run }))
        .collect()
}
fn verdicts(a: &StagedAssessment, grid: &[Constraint]) -> Result<Vec<&'static str>, String> {
    grid.iter()
        .map(|c| {
            Ok(match a.verdict(*c)? {
                FidelityVerdict::Meets => "Meets",
                FidelityVerdict::Violates => "Violates",
                FidelityVerdict::Unresolved => "Unresolved",
            })
        })
        .collect()
}
fn measures(a: &StagedAssessment) -> Value {
    match a {
        StagedAssessment::Complete(a) => {
            json!({"stage":"Complete","local":a.local_measure,"run":a.run_measure,
            "local_bounds":{"lower":a.local.status().lower_bound(),"upper":a.local.status().upper_bound()},
            "run_bounds":{"lower":a.run.status().lower_bound(),"upper":a.run.status().upper_bound()}})
        }
        StagedAssessment::LocalRejected {
            local,
            local_measure,
            max_local_tolerance,
            ..
        } => json!({"stage":"LocalRejected","local":local_measure,
            "local_bounds":{"lower":local.status().lower_bound(),"upper":local.status().upper_bound()},
            "max_local_tolerance":max_local_tolerance,"run":null,"run_state":"NotMeasured"}),
    }
}
fn complete_evidence(a: &Assessment) -> Value {
    measures(&StagedAssessment::Complete(a.clone()))
}
fn points(records: &[Value], constraints: &[Constraint]) -> Result<Value, String> {
    let mut result = Vec::new();
    for (g, constraint) in constraints.iter().enumerate() {
        let winner = records
            .iter()
            .filter(|r| r["states"][g] == "Meets")
            .min_by_key(|r| {
                (
                    r["cost_bits"].as_u64().unwrap_or(u64::MAX),
                    r["index"].as_u64().unwrap_or(u64::MAX),
                )
            });
        let upper = winner.and_then(|r| r["cost_bits"].as_u64());
        let lower = records
            .iter()
            .filter(|r| r["states"][g] != "Violates")
            .filter_map(|r| r["cost_lower_bound"].as_u64())
            .min();
        let gap = match (upper, lower) {
            (Some(u), Some(l)) => Some(u.checked_sub(l).ok_or("finite bank lower exceeds upper")?),
            _ => None,
        };
        result.push(json!({"constraint":constraint,"selected":winner.and_then(|r|r["index"].as_u64()),"upper_cost":upper,"lower_cost":lower,"gap":gap}));
    }
    Ok(json!(result))
}
struct TimedRun<'a> {
    inner: &'a dyn RunCheck,
    nanos: AtomicU64,
    calls: AtomicUsize,
}
impl RunCheck for TimedRun<'_> {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        let start = Instant::now();
        let result = self.inner.episodes(artifact);
        self.nanos.fetch_add(
            start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        result
    }
}
fn peak_rss() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|s| {
            s.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()?
                .checked_mul(1024)
        })
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err(
            "FIT_EXPORT EVAL_EXPORT SPEC PREPARED_REPORT OUT plus explicit resource/shard keys"
                .into(),
        );
    }
    let (fit_export, export, spec_path, prepared_path, out) = (
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
    );
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    let mut keys = BTreeMap::new();
    for arg in &args[5..] {
        let (k, v) = arg.split_once('=').ok_or("expected KEY=VALUE")?;
        if ![
            "train_export",
            "head_start",
            "head_count",
            "max_bank",
            "codec_bytes",
            "local_source_bytes",
            "trace_bytes",
            "readout_resident_bytes",
            "readout_workspace_bytes",
            "parallel",
            "preflight",
            "bank",
        ]
        .contains(&k)
            || keys.insert(k, v).is_some()
        {
            return Err("unknown or duplicate option".into());
        }
    }
    let mode = BankMode::parse(keys.get("bank").copied().unwrap_or("single_head"))?;
    let groups = mode.groups();
    let complete_count = groups.len();
    let preflight = match keys.get("preflight").copied().unwrap_or("0") {
        "0" => false,
        "1" => true,
        _ => return Err("preflight must be0 or1".into()),
    };
    let number = |key| {
        keys.get(key)
            .ok_or_else(|| format!("declare {key}"))?
            .parse::<usize>()
            .map_err(|e| e.to_string())
    };
    let (start, count, max_bank) = (
        number("head_start")?,
        number("head_count")?,
        number("max_bank")?,
    );
    let (codec_bytes, source_bytes, trace_bytes, parallel) = (
        number("codec_bytes")?,
        number("local_source_bytes")?,
        number("trace_bytes")?,
        number("parallel")?,
    );
    let readout = Budget {
        resident_bytes: number("readout_resident_bytes")?,
        workspace_bytes: number("readout_workspace_bytes")?,
    };
    mode.validate_scope(start, count, max_bank)?;
    if [
        codec_bytes,
        source_bytes,
        trace_bytes,
        parallel,
        readout.resident_bytes,
        readout.workspace_bytes,
    ]
    .contains(&0)
    {
        return Err("invalid contiguous head shard or explicit resource budgets".into());
    }
    if sha256(spec_path)? != SPEC_SHA || sha256(&export.join("tokens.f64"))? != PANEL_TOKENS_SHA {
        return Err("frozen strong-panel Spec or scored token hash mismatch".into());
    }
    let train_export = Path::new(*keys.get("train_export").ok_or("declare train_export")?);
    let begun = Instant::now();
    let prepared = read_json(prepared_path)?;
    let entries = inventory(&prepared)?;
    let lineage = weight_lineage(train_export, fit_export, export, &prepared)?;
    let hashes_seconds = begun.elapsed().as_secs_f64();
    let imported = import_language_model(export, 1, 1)?;
    if integer(&imported.record["config"], "n_layers")? != 4
        || integer(&imported.record["config"], "n_heads")? != 6
        || integer(&imported.record["config"], "n_kv_heads")? != 6
    {
        return Err("prepared panel requires native 4L/6-head model".into());
    }
    let native = split_sites(&imported.program)?;
    let base = Artifact::native(&native)?.f32_literals()?;
    let layers = layer_nodes(&native, 4)?;
    validate_activation_manifest(&prepared, &base, &layers)?;
    let bank = CopyResidualBank::new(&base, &layers, 1, &RANKS, 193)?;
    let factors_dir = prepared_path
        .parent()
        .ok_or("report directory required")?
        .join("factors");
    let listed: BTreeSet<_> = std::fs::read_dir(&factors_dir)
        .map_err(|e| e.to_string())?
        .map(|e| {
            e.map(|e| e.file_name().to_string_lossy().into_owned())
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    if listed != entries.iter().map(|e| e.filename.clone()).collect() {
        return Err("factor directory differs from complete declared inventory".into());
    }
    let validation_start = Instant::now();
    for entry in &entries {
        let op = base
            .program
            .operators
            .iter()
            .find(|o| o.name == format!("blocks.{}.o{}", entry.choice.layer, entry.choice.head))
            .ok_or("native head missing")?;
        let validated = load_factor(&factors_dir, entry, op)?;
        drop(validated);
    }
    let factor_validation_seconds = validation_start.elapsed().as_secs_f64();
    let decoder = Decoder::from_export(export)?;
    let spec = Spec::load(spec_path, &decoder)?;
    if spec.rows != CONTEXT || spec.episodes.len() != 80 {
        return Err("frozen eighty-episode full-context panel required".into());
    }
    let rows = passages(export, CONTEXT)?;
    let family = local_family(&rows)?;
    if preflight {
        std::fs::create_dir(out).map_err(|e| e.to_string())?;
        write_json(
            &out.join("PREFLIGHT.json"),
            &json!({
                "status":"all lineage/factor/spec/schema checks passed; no GPU initialized and no fidelity measurements",
                "complete_proposal_inventory":384,"factors_validated":entries.len(),"bank":mode.name(),"declared_program_count":complete_count,"lineage":lineage,
                "prepared_report_sha256":sha256(prepared_path)?,"spec_sha256":SPEC_SHA,
                "binary_sha256":sha256(&std::env::current_exe().map_err(|e| e.to_string())?)?,
                "seconds":{"lineage":hashes_seconds,"factor_validation":factor_validation_seconds,"total":begun.elapsed().as_secs_f64()}
            }),
        )?;
        return Ok(());
    }
    let init = Instant::now();
    let codec = NativeOperatorCodec::new(&base.program, codec_bytes).map_err(|e| e.to_string())?;
    let codec_initialization_seconds = init.elapsed().as_secs_f64();
    let decoded_source = EncodedArtifact::of_with_native_codec(&base, &codec)?
        .using_native_codec(&codec)
        .decode()?;
    let device = gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required)
        .map_err(|e| e.to_string())?
        .ok_or("CUDA required")?;
    let init = Instant::now();
    let local = Local::new(&native, family.clone(), None, CONTEXT)
        .with_cuda(device.clone(), trace_bytes)?
        .with_cuda_native_sharing(source_bytes)?;
    let local_source_initialization_seconds = init.elapsed().as_secs_f64();
    let run = LanguageRun::new(&decoder, &native, &spec, &rows, parallel)?
        .with_cuda(device, trace_bytes)?
        .with_cuda_native_source(&decoded_source)?
        .with_cuda_readout(readout)?;
    let timed_run = TimedRun {
        inner: &run,
        nanos: AtomicU64::new(0),
        calls: AtomicUsize::new(0),
    };
    let constraints = grid();
    let scope = json!({"method":"prepared explicit linear factors plus fully paid arithmetic Copy body; template baseline, not discovery",
        "bank":mode.name(),"complete_count_including_native":complete_count,"prepared_source_factor_count":384,"heads":HEADS,"ranks":RANKS,"families":["NativeSvd","CopyResidual"],"fits":FITS,
        "shard":{"head_start":start,"head_count":count,"candidate_count_plus_native":if mode==BankMode::UniformJoint{17}else{1+count*16}},"grid":constraints,
        "local":{"sequences":2,"context":CONTEXT,"rows":2*CONTEXT,"batch":CONTEXT,"denominator":"RMS of native split attention-contribution row norms over the full family; maximum normalized row Euclidean error","ascent":null},
        "run":{"spec_sha256":SPEC_SHA,"episodes":80,"context":CONTEXT,"parallel":parallel,"backend":run.backend_name(),"metrics":"CPU normalization/KL oracle on CUDA f64 head logits; uncertified GPU metric proposals never used"},
        "local_first":"only decoded full Local proof Violates at maximum declared delta1 omits Run; no singleton pruning",
        "unknown_cost_lower_bound":0,"failed_and_unmeasured_retained":true,"optimality":"only the explicitly declared finite bank; unmeasured/failed candidates remain unresolved, and a singleton head shard is incomplete",
        "joint_semantics":"uniform_joint executes all24 replacements together and measures the full composed Local/Run; no singleton outcomes reused; exact shared Copy body pool priced once",
        "budgets":{"codec_bytes":codec_bytes,"local_source_numeric_bytes":source_bytes,"trace_bytes":trace_bytes,"readout_resident_bytes":readout.resident_bytes,"readout_workspace_bytes":readout.workspace_bytes},
        "budget_exclusions":"codec excludes caller source and transient constructor; source excludes host/indices/activations/workspaces/allocator; trace excludes operators/workspaces; head numeric budgets exclude allocator/context/library workspaces",
        "comparison_scope":"comparison-rounding intervals on executed values; no CPU/CUDA full-network equivalence certificate",
        "numerical_model":"conditional arithmetic bounds: existing CPU KL assumes exp within2ULP and ln within1ULP; these are not Rust-guaranteed or unconditional certificates",
        "lineage":lineage,"prepared_report_sha256":sha256(prepared_path)?,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,
        "conditional_preparation_costs":"proposal diagnostics only; final artifacts repriced with explicit Copy definitions"});
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    write_json(&out.join("SCOPE.json"), &scope)?;
    let mut records = vec![
        json!({"index":0,"label":"native","cost_bits":null,"cost_lower_bound":0,"states":vec!["Unevaluated";constraints.len()],"run_state":"NotMeasured"}),
    ];
    for (index, group) in groups.iter().enumerate().skip(1) {
        let first = &entries[group[0]];
        let sources: Vec<_> = group.iter().map(|i| { let e=&entries[*i];
            json!({"source_index":e.index,"choice":e.choice,"fit":e.fit,"factors":{"file":e.filename,"sha256":e.hash,"bytes":e.bytes},
                "prepared_conditional_single_head_cost_bits":e.conditional_cost,"fit_diagnostics":e.diagnostics})
        }).collect();
        records.push(json!({"index":index,"bank":mode.name(),"rank":first.choice.rank,"family":first.choice.family,"fit":first.fit,
            "replaced_heads":group.len(),"sources":sources,"cost_bits":null,"cost_lower_bound":0,
            "states":vec!["Unevaluated";constraints.len()],"run_state":"NotMeasured"}));
    }
    let mut cache = CostCache::default();
    let mut journal = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("ASSESSMENTS.jsonl"))
        .map_err(|e| e.to_string())?;
    let build = |index: usize| -> Result<Artifact, String> {
        if index == 0 {
            return Ok(base.clone());
        }
        let group = &groups[index];
        let payloads: Vec<_> = group
            .iter()
            .map(|i| {
                let e = &entries[*i];
                let op = base
                    .program
                    .operators
                    .iter()
                    .find(|o| o.name == format!("blocks.{}.o{}", e.choice.layer, e.choice.head))
                    .ok_or("native head missing")?;
                load_factor(&factors_dir, e, op)
            })
            .collect::<Result<_, String>>()?;
        if mode == BankMode::SingleHead {
            bank.candidate_with_prepared_factors(entries[group[0]].choice, &payloads[0])?
                .f32_literals()
        } else {
            let replacements: Vec<_> = group
                .iter()
                .zip(&payloads)
                .map(|(i, payload)| (entries[*i].choice, payload))
                .collect();
            bank.compose_with_prepared_factors(&replacements)?
                .f32_literals()
        }
    };
    let chosen = groups
        .iter()
        .enumerate()
        .filter(|(index, group)| {
            *index == 0 || mode == BankMode::UniformJoint || {
                let e = &entries[group[0]];
                (start..start + count).contains(&(e.choice.layer * 6 + e.choice.head))
            }
        })
        .map(|(index, _)| index);
    for index in chosen {
        eprintln!(
            "prepared {} candidate {index}/{}: start",
            mode.name(),
            complete_count - 1
        );
        let start_time = Instant::now();
        let rn = timed_run.nanos.load(Ordering::Relaxed);
        let rc = timed_run.calls.load(Ordering::Relaxed);
        let result = (|| -> Result<(), String> {
            let compose = Instant::now();
            let artifact = build(index)?;
            records[index]["compose_seconds"] = json!(compose.elapsed().as_secs_f64());
            if artifact.places != base.places {
                return Err("native intervention places changed".into());
            }
            let cost = structural_cost(&artifact, &mut cache)?;
            records[index]["cost_bits"] = json!(cost.total());
            records[index]["cost_lower_bound"] = json!(cost.total());
            let assess = Instant::now();
            let measured = assess_once_local_first_with_native_codec(
                &local,
                &timed_run,
                &artifact,
                &constraints,
                &mut cache,
                &codec,
            )?;
            records[index]["assessment_seconds"] = json!(assess.elapsed().as_secs_f64());
            if measured.cost() != cost {
                return Err("decoded complete cost changed".into());
            }
            records[index]["states"] = json!(verdicts(&measured, &constraints)?);
            records[index]["evidence"] = measures(&measured);
            records[index]["run_state"] = json!(if measured.run_measure().is_some() {
                "Measured"
            } else {
                "NotMeasured"
            });
            Ok(())
        })();
        if let Err(error) = result {
            records[index]["states"] = json!(vec!["Failed"; constraints.len()]);
            records[index]["error"] = json!(error);
            records[index]["run_state"] = json!(if timed_run.calls.load(Ordering::Relaxed) > rc {
                "Failed"
            } else {
                "NotMeasured"
            });
        }
        records[index]["seconds"] = json!(start_time.elapsed().as_secs_f64());
        records[index]["run_calls"] = json!(timed_run.calls.load(Ordering::Relaxed) - rc);
        records[index]["run_wall_seconds"] =
            json!((timed_run.nanos.load(Ordering::Relaxed) - rn) as f64 * 1e-9);
        records[index]["codec_usage"] = json!(codec.usage());
        records[index]["peak_rss_bytes"] = json!(peak_rss());
        writeln!(journal, "{}", records[index]).map_err(|e| e.to_string())?;
        journal.flush().map_err(|e| e.to_string())?;
        write_json(
            &out.join("PARTIAL.json"),
            &json!({"scope":scope,"records":records,"points":points(&records,&constraints)?,"run_stage_seconds":run.timing(),"complete":false}),
        )?;
        cache.clear_measurements();
        eprintln!(
            "prepared candidate {index}: {}s",
            start_time.elapsed().as_secs_f64()
        );
    }
    let frontier_points = points(&records, &constraints)?;
    let selected: BTreeSet<_> = frontier_points
        .as_array()
        .ok_or("points")?
        .iter()
        .filter_map(|p| p["selected"].as_u64().map(|i| i as usize))
        .collect();
    let replay_start = Instant::now();
    let mut replays = Vec::new();
    for index in selected {
        let artifact = build(index)?;
        let path = out.join(format!("selected.{index}.bin"));
        std::fs::write(&path, artifact.to_bytes()?).map_err(|e| e.to_string())?;
        drop(artifact);
        let saved = std::fs::read(&path).map_err(|e| e.to_string())?;
        let decoded = Artifact::from_bytes(&saved, &native.declarations)?;
        let byte_count = saved.len();
        if decoded.to_bytes()? != saved || decoded.places != base.places {
            return Err("selected ordinary saved-byte replay mismatch".into());
        }
        drop(saved);
        decoded.validate_coverage(&native)?;
        let measured = assess_once(&local, &timed_run, &decoded, constraints[0], &mut cache)?;
        let staged = StagedAssessment::Complete(measured.clone());
        if Some(measured.cost.total()) != records[index]["cost_bits"].as_u64()
            || json!(verdicts(&staged, &constraints)?) != records[index]["states"]
            || complete_evidence(&measured) != records[index]["evidence"]
        {
            return Err("selected replay changed complete cost, evidence or verdicts".into());
        }
        let isolated =
            gam_mpd::local_kl::isolated_downstream_kl(&native, &decoded, &family, CONTEXT)?;
        replays.push(json!({"index":index,"sha256":sha256(&path)?,"file":path.file_name().and_then(|p|p.to_str()),"bytes":byte_count,"cost":measured.cost,"evidence":complete_evidence(&measured),"isolated_downstream_patch_kl":isolated,"diagnostic_only":true}));
        cache.clear_measurements();
    }
    write_json(
        &out.join("REPORT.json"),
        &json!({"scope":scope,"records":records,"points":frontier_points,"selected_saved_byte_replays":replays,
        "seconds":{"lineage_and_manifest":hashes_seconds,"factor_validation":factor_validation_seconds,"codec_initialization":codec_initialization_seconds,"local_source_initialization":local_source_initialization_seconds,"selected_replay":replay_start.elapsed().as_secs_f64(),"total":begun.elapsed().as_secs_f64()},
        "local_source_retained_numeric_bytes":local.cuda_native_source_numeric_bytes()?,"codec":{"stats":codec.stats(),"usage":codec.usage(),"selected_replay":"ordinary uncached independent byte decoding"},"run_stage_seconds":run.timing(),"peak_rss_bytes":peak_rss(),"complete_shard":true,"complete_bank":mode==BankMode::UniformJoint || count==HEADS}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_mpd::operator_program::{Interface, OperatorBody, exact_precision};
    use ndarray::Array2;
    #[test]
    fn uniform_joint_bank_covers_every_prepared_factor_in_canonical_full_head_programs() {
        let entries = inventory(&fixture_report()).expect("complete inventory");
        let groups = BankMode::UniformJoint.groups();
        assert_eq!(groups.len(), 17);
        assert!(groups[0].is_empty());
        let used: BTreeSet<_> = groups.iter().flatten().copied().collect();
        assert!(used.iter().copied().eq(0..384));
        for group in groups.iter().skip(1) {
            assert_eq!(group.len(), HEADS);
            let first = &entries[group[0]];
            for (head, source) in group.iter().enumerate() {
                let entry = &entries[*source];
                assert_eq!(entry.choice.layer * 6 + entry.choice.head, head);
                assert_eq!(entry.choice.rank, first.choice.rank);
                assert_eq!(entry.choice.family, first.choice.family);
                assert_eq!(entry.fit, first.fit);
            }
        }
        assert!(BankMode::UniformJoint.validate_scope(0, 24, 17).is_ok());
        assert!(BankMode::UniformJoint.validate_scope(0, 1, 17).is_err());
        assert!(BankMode::UniformJoint.validate_scope(0, 24, 385).is_err());
        assert!(BankMode::SingleHead.validate_scope(0, 1, 385).is_ok());
        assert_eq!(BankMode::SingleHead.groups().len(), 385);
        assert!(
            BankMode::SingleHead
                .validate_scope(usize::MAX, 2, 385)
                .is_err()
        );
        assert!(BankMode::parse("joint_best_singletons").is_err());
    }
    fn fixture_report() -> Value {
        let mut heads = Vec::new();
        for layer in 0..4 {
            for head in 0..6 {
                let mut points = Vec::new();
                for rank in RANKS {
                    for family in ["NativeSvd", "CopyResidual"] {
                        for fit in FITS {
                            let filename = format!(
                                "L{layer}.H{head}.{family}.{}.rank{rank}.json",
                                if fit == FITS[0] { "weight" } else { "input" }
                            );
                            points.push(json!({"family":family,"fit":fit,"rank":rank,"C32_bits":1,
                    "factors":{"file":filename,"sha256":"0".repeat(64),"bytes":1,"independent_factor_replay":true,"final_artifact_acceptance":false}}));
                        }
                    }
                }
                heads.push(json!({"layer":layer,"head":head,"points":points}));
            }
        }
        json!({"ranks":RANKS,"used_training_rows":CONTEXT*512,"heads":heads})
    }
    fn temporary(label: &str) -> std::path::PathBuf {
        let tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mpd-prepared-{label}-{}-{tick}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).expect("directory");
        dir
    }
    #[test]
    fn prepared_inventory_is_complete_and_canonical_without_cost_trust() {
        let report = fixture_report();
        let entries = inventory(&report).expect("384 entries");
        assert_eq!(entries.len(), 384);
        assert_eq!(entries[0].choice.layer, 0);
        assert_eq!(entries[0].choice.head, 0);
        assert_eq!(entries[15].choice.rank, 96);
        assert_eq!(entries[16].choice.head, 1);
        assert_eq!(entries[383].choice.layer, 3);
        assert_eq!(entries[383].choice.head, 5);
        let mut duplicate = report.clone();
        duplicate["heads"][0]["points"][1] = duplicate["heads"][0]["points"][0].clone();
        assert!(inventory(&duplicate).is_err());
        let mut missing = report.clone();
        missing["heads"].as_array_mut().expect("heads").pop();
        assert!(inventory(&missing).is_err());
        let mut malformed = report.clone();
        malformed["heads"][0]["points"][0]["factors"]["file"] = json!("../outside.json");
        assert!(inventory(&malformed).is_err());
        let mut costs = report.clone();
        costs["heads"][0]["points"][0]["C32_bits"] = Value::Null;
        assert!(
            inventory(&costs).expect("cost not required")[0]
                .conditional_cost
                .is_none()
        );
    }
    #[test]
    fn finite_bank_points_retain_failed_and_unmeasured_candidates() {
        let grid = [
            Constraint {
                local: 0.1,
                run: 0.01,
            },
            Constraint {
                local: 1.0,
                run: 0.01,
            },
        ];
        let mut records = vec![
            json!({"index":0,"cost_bits":10,"cost_lower_bound":10,"states":["Meets","Meets"]}),
            json!({"index":1,"cost_bits":2,"cost_lower_bound":2,"states":["Violates","Meets"]}),
            json!({"index":2,"cost_bits":null,"cost_lower_bound":0,"states":["Unevaluated","Unevaluated"]}),
            json!({"index":3,"cost_bits":1,"cost_lower_bound":1,"states":["Failed","Failed"]}),
        ];
        let p = points(&records, &grid).expect("frontier");
        assert_eq!(p[0]["selected"], 0);
        assert_eq!(p[1]["selected"], 1);
        assert_eq!(p[0]["lower_cost"], 0);
        assert_eq!(p[1]["gap"], 2);
        records[2]["states"] = json!(["Violates", "Violates"]);
        let p = points(&records, &grid).expect("failed retained");
        assert_eq!(p[0]["lower_cost"], 1);
        records[3]["states"] = json!(["Violates", "Violates"]);
        let p = points(&records, &grid).expect("complete proof");
        assert_eq!(p[0]["gap"], 0);
        assert_eq!(p[1]["gap"], 0);
    }
    #[test]
    fn factor_validation_checks_hash_rank_head_and_signed_zero_bits() {
        let dir = temporary("factor");
        let rows = Interface::native(768).expect("rows");
        let cols = Interface::native(128).expect("cols");
        let native = Operator::dense(
            "blocks.0.o0",
            rows.clone(),
            cols.clone(),
            Array2::zeros((768, 128)),
            exact_precision([0.0]).expect("precision"),
            Default::default(),
        )
        .expect("native");
        let mut left = Array2::zeros((768, 8));
        left[[0, 0]] = -0.0;
        let right = Array2::zeros((8, 128));
        let precision = exact_precision([0.0]).expect("precision");
        let mut fitted = Operator::low_rank(
            "blocks.0.o0",
            rows,
            cols,
            left.clone(),
            right.clone(),
            precision,
            Default::default(),
        )
        .expect("operator");
        fitted.body = OperatorBody::LowRank {
            left,
            right,
            precision,
        };
        let payload = LowRankProposal::of(&fitted).expect("transport");
        let mut entry = inventory(&fixture_report()).expect("inventory").remove(0);
        let path = dir.join(&entry.filename);
        let bytes = serde_json::to_vec(&payload).expect("JSON");
        std::fs::write(&path, &bytes).expect("write");
        entry.bytes = bytes.len() as u64;
        entry.hash = sha256(&path).expect("hash");
        let loaded = load_factor(&dir, &entry, &native).expect("replay");
        assert_eq!(
            serde_json::to_value(loaded).expect("value"),
            serde_json::to_value(payload).expect("value")
        );
        let mut bad = entry.clone();
        bad.choice.rank = 32;
        assert!(load_factor(&dir, &bad, &native).is_err());
        bad = entry.clone();
        bad.choice.head = 1;
        assert!(load_factor(&dir, &bad, &native).is_err());
        let mut altered = bytes;
        let last = altered.len() - 1;
        altered[last] = b' ';
        std::fs::write(&path, &altered).expect("alter");
        assert!(load_factor(&dir, &entry, &native).is_err());
        std::fs::remove_dir_all(dir).expect("cleanup");
    }
    fn metadata(dir: &Path, rows: usize, token: f64) -> Value {
        let tokens = vec![token; rows * CONTEXT];
        std::fs::write(
            dir.join("tokens.f64"),
            tokens
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .expect("tokens");
        std::fs::write(dir.join("wte.f64"), 1.0_f64.to_le_bytes()).expect("weight");
        std::fs::write(dir.join("logits_row0.f64"), token.to_le_bytes()).expect("reference");
        let value = json!({"config":{"vocab":8},"source":{"checkpoint_sha256":"0".repeat(64)},
            "files":{"wte":{"shape":[1,1]},"tokens":{"shape":[rows,CONTEXT]},"logits_row0":{"shape":[1,1]}}});
        write_json(&dir.join("export.json"), &value).expect("metadata");
        value
    }
    #[test]
    fn lineage_checks_actual_weights_and_strong_panel_training_disjointness() {
        let root = temporary("lineage");
        let train = root.join("train");
        let fit = root.join("fit");
        let eval = root.join("eval");
        for path in [&train, &fit, &eval] {
            std::fs::create_dir(path).expect("directory");
        }
        let mut t = metadata(&train, 512, 1.0);
        // Training exports carry row provenance and sparse reference logits, not weights.
        for key in [
            "row_ids",
            "logits_logsumexp",
            "logits_topk_indices",
            "logits_topk_values",
        ] {
            t["files"][key] = json!({"shape":[1,1]});
        }
        write_json(&train.join("export.json"), &t).expect("training sidecars");
        metadata(&fit, 2, 3.0);
        metadata(&eval, 2, 2.0);
        let report = json!({"manifest":{"checkpoint_sha256":"0".repeat(64),"config":t["config"],"context":CONTEXT,"identical_train_eval_token_sequences":0,
            "families":[{"name":"train","rows":512*CONTEXT,"source":t["source"],"export_sha256":sha256(&train.join("export.json")).expect("SHA"),"tokens_file_sha256":sha256(&train.join("tokens.f64")).expect("SHA")},
                {"name":"eval","rows":2*CONTEXT,"source":t["source"],"export_sha256":sha256(&fit.join("export.json")).expect("SHA"),"tokens_file_sha256":sha256(&fit.join("tokens.f64")).expect("SHA")} ]}});
        assert!(weight_lineage(&train, &fit, &eval, &report).is_ok());
        // Matching checkpoint labels cannot hide a changed actual native tensor.
        std::fs::write(eval.join("wte.f64"), 2.0_f64.to_le_bytes()).expect("changed weight");
        assert!(weight_lineage(&train, &fit, &eval, &report).is_err());
        metadata(&eval, 2, 1.0);
        assert!(
            weight_lineage(&train, &fit, &eval, &report).is_err(),
            "strong evaluation shares training token sequences"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
