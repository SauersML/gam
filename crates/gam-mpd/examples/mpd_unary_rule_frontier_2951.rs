//! Complete generic unary finite bank with all-head zero controls and Local-first acceptance.
//! EXPORT OUT max_bank=5137 max_assess=25 stop_when_optimal=0 codec_bytes=2147483648 trace_bytes=1073741824 max_seconds=600
use gam_mpd::{
    acceptance::{
        Change, Constraint, CostCache, Edit, Episode, EpisodeScore, FamilyRun, Local, RunCheck,
        StagedAssessment, assess_once_local_first_with_native_codec, structural_cost,
    },
    artifact::Artifact,
    attention_map::AttentionLayerMap,
    engine::sha256,
    device_family_run::{DeviceFamilyRun, Timing},
    import::import_language_model,
    operator_program::{NativeOperatorCodec, OperatorProgram, SlotValues},
    precision::FidelityVerdict,
    unary_rule_bank::{Choice, Limits, UnaryRuleBank},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};
const DELTAS: [f64; 3] = [0.01, 0.05, 0.1];
const EPSILONS: [f64; 3] = [0.001, 0.01, 0.1];
#[derive(Clone, Copy, Debug)]
enum Candidate {
    Native,
    Zero(usize),
    Rule(Choice),
}
struct Runs<'a> {
    runs: Vec<DeviceFamilyRun<'a>>,
}
impl RunCheck for Runs<'_> {
    fn episodes(&self, a: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        let mut out = vec![];
        for r in &self.runs {
            out.extend(r.episodes(a)?);
        }
        Ok(out)
    }
}
impl Runs<'_> {
    fn timing(&self) -> Timing {
        let mut sum = Timing::default();
        for r in &self.runs {
            let t = r.timing();
            sum.native_teacher_seconds += t.native_teacher_seconds;
            sum.candidate_construction_seconds += t.candidate_construction_seconds;
            sum.cuda_forward_hooks_download_seconds += t.cuda_forward_hooks_download_seconds;
            sum.cpu_metric_seconds += t.cpu_metric_seconds;
        }
        sum
    }
}
fn in_candidate_scope(index: usize, requested: Option<usize>) -> bool {
    index == 0 || requested.is_none_or(|wanted| wanted == index)
}
fn in_head_range(target: usize, start: u64, count: u64) -> bool {
    (start as usize..(start + count) as usize).contains(&target)
}
fn write_json(p: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(p, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn rss() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|l| {
            l.strip_prefix("VmHWM:")
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<u64>().ok())
                .and_then(|n| n.checked_mul(1024))
        })
}
fn statuses(a: &StagedAssessment, grid: &[Constraint]) -> Result<Vec<&'static str>, String> {
    grid.iter()
        .map(|&p| {
            Ok(match a.verdict(p)? {
                FidelityVerdict::Meets => "Verified",
                FidelityVerdict::Violates => "Violates",
                FidelityVerdict::Unresolved => "Unresolved",
            })
        })
        .collect()
}
fn points(records: &[Value], grid: &[Constraint]) -> Vec<Value> {
    grid.iter().enumerate().map(|(g,c)|{
    let lower=records.iter().filter(|r|r["states"][g].as_str()!=Some("Violates")).map(|r|r["cost_bits"].as_u64().unwrap_or(0)).min();
    let upper=records.iter().enumerate().filter(|(_,r)|r["states"][g].as_str()==Some("Verified")).filter_map(|(i,r)|r["cost_bits"].as_u64().map(|cost|(cost,i))).min();
    json!({"constraint":c,"lower_cost":lower,"upper_cost":upper.map(|p|p.0),"selected":upper.map(|p|p.1),"gap":upper.and_then(|u|lower.map(|l|u.0-l))})
}).collect()
}
fn optimal(points: &[Value]) -> bool {
    !points.is_empty() && points.iter().all(|p| p["gap"].as_u64() == Some(0))
}
fn local_measure(a: &StagedAssessment) -> &gam_mpd::acceptance::LocalMeasure {
    match a {
        StagedAssessment::Complete(a) => &a.local_measure,
        StagedAssessment::LocalRejected { local_measure, .. } => local_measure,
    }
}
fn episodes(maps: &[AttentionLayerMap], program: &OperatorProgram, passage: usize) -> Vec<Episode> {
    let mut out = vec![Episode {
        id: format!("clean/{passage}"),
        group: "clean".into(),
        edits: vec![],
    }];
    for map in maps {
        for h in &map.heads {
            out.push(Episode {
                id: format!("remove-head/L{}H{}/{passage}", map.native_layer, h.head),
                group: format!("remove-head/L{}H{}", map.native_layer, h.head),
                edits: vec![Edit {
                    node: h.read,
                    rows: None,
                    columns: 0..program.operators[h.output_operator].cols.width(),
                    change: Change::Scale(0.0),
                }],
            });
        }
    }
    out
}
fn main() -> Result<(), String> {
    let begun = Instant::now();
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 8 {
        return Err("EXPORT OUT max_bank=N max_assess=N stop_when_optimal=0|1 codec_bytes=N trace_bytes=N max_seconds=N".into());
    }
    let mut opts = BTreeMap::new();
    for arg in &args[2..] {
        let (k, v) = arg.split_once('=').ok_or("KEY=VALUE required")?;
        if ![
            "max_bank",
            "max_assess",
            "stop_when_optimal",
            "codec_bytes",
            "trace_bytes",
            "max_seconds",
            "head_start",
            "head_count",
            "local_source_bytes",
            "candidate_index",
        ]
        .contains(&k)
            || opts.insert(k, v).is_some()
        {
            return Err("unknown/duplicate option".into());
        }
    }
    let n = |k| -> Result<u64, String> {
        opts.get(k)
            .ok_or("missing option")?
            .parse::<u64>()
            .map_err(|e| e.to_string())
    };
    let (max_bank, max_assess, stop, codec_bytes, trace_bytes, max_seconds) = (
        n("max_bank")?,
        n("max_assess")?,
        n("stop_when_optimal")?,
        n("codec_bytes")?,
        n("trace_bytes")?,
        n("max_seconds")?,
    );
    let optional = |k, default| -> Result<u64, String> {
        opts.get(k)
            .map(|v| v.parse::<u64>().map_err(|e| e.to_string()))
            .unwrap_or(Ok(default))
    };
    let candidate_index = opts.get("candidate_index").map(|v|v.parse::<usize>().map_err(|e|e.to_string())).transpose()?;
    if candidate_index.is_some_and(|i| i>=5137) { return Err("candidate_index outside declared complete bank".into()); }
    let head_start = optional("head_start", 0)?;
    let head_count = optional("head_count", 24)?;
    let local_source_bytes = optional("local_source_bytes", 0)?;
    if head_count == 0
        || head_start
            .checked_add(head_count)
            .is_none_or(|end| end > 24)
    {
        return Err("invalid explicit native head range".into());
    }
    if max_assess == 0 || stop > 1 || codec_bytes == 0 || trace_bytes == 0 || max_seconds == 0 {
        return Err("invalid explicit resource/stop budget".into());
    }
    let (export, out) = (Path::new(&args[0]), Path::new(&args[1]));
    if out.exists() {
        return Err("fresh scientific output required".into());
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let record: Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if record["config"]["n_layers"].as_u64() != Some(4)
        || record["config"]["n_heads"].as_u64() != Some(6)
    {
        return Err(
            "this frozen protocol requires4L/all6heads; other models need new declaredscope".into(),
        );
    }
    let grid: Vec<_> = DELTAS
        .iter()
        .flat_map(|&local| EPSILONS.iter().map(move |&run| Constraint { local, run }))
        .collect();
    let scope = json!({"grammar":{"max_inputs":1,"max_nodes":2,"internal_scale_coefficients":[]},"coefficients":"one deterministic native-weight least-squares f32 amplitude perbody/source/target; NOTcontinuous-coefficient-family optimum","residual_ranks":[0],"native_layer_ids":[0,1,2,3],"all_heads":24,"complete_count":5137,"zero_controls":24,"expression_relations":5112,"source_scope":"allnativelearnedDense/LowRank/Diagonaloperators+globals excludingtarget/dependencies; Identity fixedprimitive excluded","local":{"sequences":2,"context":16,"tokens":32,"batch":16,"ascent":0,"boundary":"fixednativepost-attention-residual","denominator":gam_mpd::attention_map::LOCAL_DENOMINATOR,"normalizer_comparison":"different fromoldsplit contribution-only; do notcompareequaldeltaasidenticalconstraints"},"run":{"passages":2,"context":16,"episodes":50,"groups":25,"edits":"everyheadreadallrows/allcolumns removed separatelyperpassage; twoindependentcleans","backend":"CUDA f64 candidates; fixedCPU native reference episodes cachedonceperpassage"},"grid":grid,"codec_bytes":codec_bytes,"trace_bytes":trace_bytes,"cuda_native_weight_sharing":local_source_bytes>0,"local_source_bytes":local_source_bytes,"assessment_candidate_index":candidate_index,"assessment_head_start":head_start,"assessment_head_count":head_count,"cuda_Local_compile":"freshResident perartifact; devicehandle andCPUdenominators shared","max_assess":max_assess,"stop_when_all_cost_gaps_zero":stop==1,"max_seconds":max_seconds,"resource_cases":"failed/unmeasured kept in complete bank; unknowncost lower0","optimality_scope":"complete5137declaredfittedsingle-substitutionbank only; unmeasuredfidelity remainsexplicit; no global programor continuousamplitudeclaim"});
    write_json(&out.join("SCOPE.json"), &scope)?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "export.json".to_string(),
        sha256(&export.join("export.json"))?,
    );
    for name in record["files"]
        .as_object()
        .ok_or("exportfiles absent")?
        .keys()
    {
        let f = format!("{name}.f64");
        inputs.insert(f.clone(), sha256(&export.join(f))?);
    }
    let mut sources = BTreeMap::new();
    for (name, text) in [
        ("driver", include_str!("mpd_unary_rule_frontier_2951.rs")),
        ("bank", include_str!("../src/unary_rule_bank.rs")),
        (
            "enumerator",
            include_str!("../src/matrix_rule_enumeration.rs"),
        ),
        ("matrix_rule", include_str!("../src/matrix_rule.rs")),
        ("acceptance", include_str!("../src/acceptance.rs")),
        (
            "device_family_run",
            include_str!("../src/device_family_run.rs"),
        ),
        ("artifact", include_str!("../src/artifact.rs")),
    ] {
        let p = out.join(format!("{name}.source"));
        std::fs::write(&p, text).map_err(|e| e.to_string())?;
        sources.insert(name, sha256(&p)?);
        std::fs::remove_file(p).map_err(|e| e.to_string())?;
    }
    write_json(
        &out.join("PROVENANCE.json"),
        &json!({"inputs":inputs,"sources":sources,"binary":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"export_source":record["source"]}),
    )?;
    let imported = import_language_model(export, 2, 16)?;
    let native = Artifact::native(&imported.program)?.f32_literals()?;
    let maps: Vec<_> = (0..4)
        .map(|layer| AttentionLayerMap::of(&native.program, layer))
        .collect::<Result<_, _>>()?;
    let mut bank = UnaryRuleBank::all_attention(
        &native,
        &[0, 1, 2, 3],
        Limits {
            max_prefixes: 1_000_000,
            max_bodies: 100_000,
            max_candidates_including_native: max_bank,
            cache_bytes: 268435456,
            matrix_workspace_bytes: 67108864,
        },
    )?;
    let count = bank.cardinality_with_zero_controls()?;
    if count != 5137 || count > max_bank {
        return Err(format!(
            "completeactualbank{count}inclzeros differs/exceeds declared maxbank{max_bank}"
        ));
    }
    let families:Vec<_>=bank.inventory.families.iter().enumerate().map(|(i,f)|json!({"family":i,"target":f.target,"source_pools":f.source_pools,"body":format!("{:?}",f.body.rule),"count":f.binding_count})).collect();
    let sources_meta:Vec<_>=bank.inventory.sources.iter().map(|s|json!({"id":s.id,"name":native.program.operators[s.id].name,"type":format!("{:?}",s.ty),"dependencies":s.dependencies})).collect();
    write_json(
        &out.join("INVENTORY.json"),
        &json!({"complete":true,"families":families,"sources":sources_meta,"prefixes":bank.inventory.visited_prefixes,"count":count,"native":1,"allzeroheads":24,"bindings":5112}),
    )?;
    let token_data = match &imported.contract.family.slots[0] {
        SlotValues::Tokens(t) => t,
        SlotValues::Raw(_) => return Err("expected native tokenfamily".into()),
    };
    write_json(
        &out.join("TOKEN_FAMILY.json"),
        &json!({"tokens":token_data,"sequence":imported.contract.family.layout.as_ref().map(|l|&l.sequence),"position":imported.contract.family.layout.as_ref().map(|l|&l.position),"context":16,"source":"firsttwoexporttokenrows; samefrozenlocal/runfamily"}),
    )?;
    let mut candidates = vec![Candidate::Native];
    candidates.extend(bank.zero_controls().map(Candidate::Zero));
    candidates.extend(bank.choices().map(Candidate::Rule));
    let mut costs = CostCache::default();
    let mut records = vec![];
    let mut price_journal = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("PRICES.jsonl"))
        .map_err(|e| e.to_string())?;
    for (index, &c) in candidates.iter().enumerate() {
        let artifact = match c {
            Candidate::Native => native.clone(),
            Candidate::Zero(t) => bank.zero_candidate(t)?,
            Candidate::Rule(c) => bank.priced_skeleton(c)?,
        };
        let cost = structural_cost(&artifact, &mut costs);
        let mut r = json!({"index":index,"candidate":format!("{c:?}"),"states":vec!["Unevaluated";grid.len()],"quality_measured":false});
        match cost {
            Ok(cost) => {
                r["cost_bits"] = json!(cost.total());
                r["C32"] = json!(cost);
            }
            Err(e) => {
                r["cost_bits"] = Value::Null;
                r["cost_lower_bound"] = json!(0);
                r["cost_error"] = json!(e);
            }
        }
        writeln!(price_journal, "{r}").map_err(|e| e.to_string())?;
        records.push(r);
    }
    price_journal.flush().map_err(|e| e.to_string())?;
    let mut order: Vec<_> = (1..candidates.len())
        .filter(|&i| {
            if !in_candidate_scope(i, candidate_index) {return false;}
            let target = match candidates[i] {
                Candidate::Native => return true,
                Candidate::Zero(t) => t,
                Candidate::Rule(c) => bank
                    .targets
                    .iter()
                    .position(|t| t.operator == bank.inventory.families[c.family].target)
                    .expect("inventory target belongs to bank"),
            };
            in_head_range(target, head_start, head_count)
        })
        .collect();
    order.sort_by_key(|&i| (records[i]["cost_bits"].as_u64().unwrap_or(0), i));
    order.insert(0, 0);
    if candidate_index.is_some_and(|i|!order.contains(&i)) { return Err("candidate_index excluded by explicit head range".into()); }
    let codec_start = Instant::now();
    let codec = NativeOperatorCodec::new(
        &native.program,
        usize::try_from(codec_bytes).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let codec_seconds = codec_start.elapsed().as_secs_f64();
    let device = gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required)
        .map_err(|e| e.to_string())?
        .ok_or("CUDA required")?;
    if !device.float64() {
        return Err("f64CUDA required".into());
    }
    let limit = usize::try_from(trace_bytes).map_err(|e| e.to_string())?;
    let mut local = Local::new(&native.program, imported.contract.family.clone(), None, 16)
        .with_cuda(device.clone(), limit)?;
    if local_source_bytes > 0 {
        local = local.with_cuda_native_sharing(
            usize::try_from(local_source_bytes).map_err(|e| e.to_string())?,
        )?;
    }
    if local_source_bytes > 0 {
        let fresh = Local::new(&native.program, imported.contract.family.clone(), None, 16)
            .with_cuda(device.clone(), limit)?;
        let choice = candidates
            .iter()
            .find_map(|c| match c {
                Candidate::Rule(choice) => Some(*choice),
                _ => None,
            })
            .ok_or("expression parity control absent")?;
        let expression = bank.candidate(choice)?.0.f32_literals()?;
        let decoded = Artifact::from_bytes(&expression.to_bytes()?, &native.program.declarations)?;
        let mut parity = vec![];
        for (label, artifact) in [("native", &native), ("expression", &decoded)] {
            let a = fresh.measure(artifact)?;
            let b = local.measure(artifact)?;
            if serde_json::to_value(&a).map_err(|e| e.to_string())?
                != serde_json::to_value(&b).map_err(|e| e.to_string())?
            {
                return Err("fresh/shared Local exact evidence mismatch".into());
            }
            parity.push(json!({"label":label,"fresh":a,"shared":b,"exact_equal":true}));
        }
        write_json(
            &out.join("LOCAL_SHARING_PARITY.json"),
            &json!({"source_bytes":local.cuda_native_source_numeric_bytes(),"controls":parity}),
        )?;
    }
    let mut native_prefix = native.program.clone();
    let last = bank
        .targets
        .iter()
        .map(|t| t.write)
        .max()
        .ok_or("emptytargetscope")?;
    native_prefix.nodes.truncate(last + 1);
    native_prefix.output = last;
    let trace = native_prefix
        .execute(&imported.contract.family, false)
        .map_err(|e| e.to_string())?;
    let mut diagnostics = vec![];
    for map in &maps {
        for h in &map.heads {
            let y =
                trace.values[h.read].dot(&native.program.operators[h.output_operator].matrix().t());
            let norm = (y.iter().map(|v| v * v).sum::<f64>() / 32.0).sqrt();
            diagnostics.push(json!({"layer":map.native_layer,"head":h.head,"read":h.read,"operator":h.output_operator,"native_write":map.output,"native_contribution_RMS_rowL2":norm,"Local_postresidual_denominator":local.scale(map.output)?}));
        }
    }
    write_json(
        &out.join("NATIVE_HEAD_DIAGNOSTICS.json"),
        &json!({"domain":"same32declaredLocaltokenrows; diagnosticonly; no thresholdchange","heads":diagnostics}),
    )?;
    drop(trace);
    let mut specs = vec![];
    let mut runs_cpu = vec![];
    for passage in 0..2 {
        let ep = episodes(&maps, &native.program, passage);
        for e in &ep {
            specs.push(json!({"id":e.id,"group":e.group,"passage":passage,"edits":e.edits.iter().map(|d|json!({"native_node":d.node,"rows":"all16","columns":[d.columns.start,d.columns.end],"scale":0.0})).collect::<Vec<_>>()}));
        }
        runs_cpu.push(FamilyRun {
            model: &native.program,
            family: imported
                .contract
                .family
                .select(&(passage * 16..(passage + 1) * 16).collect::<Vec<_>>()),
            readouts: 1,
            episodes: ep,
        });
    }
    write_json(
        &out.join("RUN_SPEC.json"),
        &json!({"episodes":specs,"episode_count":50,"group_count":25,"context":16,"passages":2,"scope":scope}),
    )?;
    let run = Runs {
        runs: runs_cpu
            .iter()
            .map(|r| DeviceFamilyRun::new(r, device.clone(), limit))
            .collect::<Result<_, _>>()?,
    };
    let mut journal = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("ASSESSMENTS.jsonl"))
        .map_err(|e| e.to_string())?;
    let mut measured = 0u64;
    let mut local_rejected = 0u64;
    let mut stopped = "inventory exhausted";
    for index in order {
        if measured >= max_assess {
            stopped = "declared assessment budget";
            break;
        }
        if begun.elapsed().as_secs() >= max_seconds {
            stopped = "declared wall budget";
            break;
        }
        if stop == 1 && optimal(&points(&records, &grid)) {
            stopped = "all declared objective cost gaps zero";
            break;
        }
        let t = Instant::now();
        let candidate_start = Instant::now();
        let built = match candidates[index] {
            Candidate::Native => Ok((native.clone(), None)),
            Candidate::Zero(t) => bank.zero_candidate(t).map(|a| (a, None)),
            Candidate::Rule(c) => bank.candidate(c).map(|(a, d)| (a, Some(d))),
        };
        let build_seconds = candidate_start.elapsed().as_secs_f64();
        let result = (|| -> Result<(), String> {
            let (artifact, fit) = built?;
            if artifact.places != native.places {
                return Err("native intervention places changed".into());
            }
            let actual = structural_cost(&artifact, &mut costs)?;
            if records[index]["cost_bits"].as_u64() != Some(actual.total()) {
                return Err("actualcandidate differsfromprepricedC32".into());
            }
            if candidate_index==Some(index) {
                let bytes=artifact.to_bytes()?;
                let decoded=Artifact::from_bytes(&bytes,&native.program.declarations)?;
                decoded.validate_coverage(&native.program)?;
                if decoded.to_bytes()?!=bytes || structural_cost(&decoded,&mut CostCache::default())?!=actual {
                    return Err("requested candidate ordinary saved-byte parity failed".into());
                }
                let file=format!("requested.{index}.bin");
                std::fs::write(out.join(&file),bytes).map_err(|e|e.to_string())?;
                write_json(&out.join("REQUESTED_CANDIDATE.json"),&json!({"index":index,"candidate":format!("{:?}",candidates[index]),"file":file,"sha256":sha256(&out.join(&file))?,"cost_bits":actual.total(),"C32":actual,"ordinary_decode_canonical_coverage_cost":true}))?;
            }
            let assessment = assess_once_local_first_with_native_codec(
                &local, &run, &artifact, &grid, &mut costs, &codec,
            )?;
            if assessment.cost() != actual {
                return Err("decodedC32 pricechanged".into());
            }
            records[index]["states"] = json!(statuses(&assessment, &grid)?);
            records[index]["local"] = json!(local_measure(&assessment));
            records[index]["run"] = json!(assessment.run_measure());
            records[index]["quality_measured"] = json!(true);
            records[index]["run_measured"] = json!(assessment.run_measure().is_some());
            if matches!(assessment, StagedAssessment::LocalRejected { .. }) {
                local_rejected += 1;
            }
            if let Some(d) = fit {
                records[index]["fit"] = json!({"amplitude":d.amplitude,"amplitude_bits":d.amplitude.to_bits(),"relative_weight_frobenius":d.relative_weight_frobenius});
            }
            Ok(())
        })();
        if let Err(e) = result {
            records[index]["error"] = json!(e);
            records[index]["states"] = json!(vec!["Unresolved"; grid.len()]);
        }
        measured += 1;
        records[index]["seconds"] = json!(t.elapsed().as_secs_f64());
        records[index]["construction_fit_seconds"] = json!(build_seconds);
        records[index]["run_cumulative_timing"] = json!(run.timing());
        records[index]["peak_rss_bytes"] = json!(rss());
        records[index]["codec_usage"] = json!(codec.usage());
        records[index]["free_total_device_bytes"] =
            json!(device.memory().map_err(|e| e.to_string())?);
        writeln!(journal, "{}", records[index]).map_err(|e| e.to_string())?;
        journal.flush().map_err(|e| e.to_string())?;
        eprintln!(
            "assessedindex={index} ordinal={measured}/{max_assess} local_rejected={local_rejected} elapsed={:.1}s",
            begun.elapsed().as_secs_f64()
        );
    }
    let final_points = points(&records, &grid);
    let selected: std::collections::BTreeSet<_> = final_points
        .iter()
        .filter_map(|p| p["selected"].as_u64().map(|i| i as usize))
        .collect();
    let mut replays = vec![];
    for index in selected {
        if begun.elapsed().as_secs() >= max_seconds {
            replays.push(
                json!({"index":index,"unmeasured":"replaywallbudget; unresolvedsavedbytefollowup"}),
            );
            continue;
        }
        let artifact = match candidates[index] {
            Candidate::Native => native.clone(),
            Candidate::Zero(t) => bank.zero_candidate(t)?,
            Candidate::Rule(c) => bank.candidate(c)?.0,
        };
        let bytes = artifact.to_bytes()?;
        let p = out.join(format!("selected.{index}.bin"));
        std::fs::write(&p, &bytes).map_err(|e| e.to_string())?;
        let saved = std::fs::read(&p).map_err(|e| e.to_string())?;
        let decoded = Artifact::from_bytes(&saved, &native.program.declarations)?;
        decoded.validate_coverage(&native.program)?;
        if decoded.to_bytes()? != saved {
            return Err("selectedsavedbytes noncanonical".into());
        }
        let assessment = gam_mpd::acceptance::assess_once_local_first(
            &local, &run, &decoded, &grid, &mut costs,
        )?;
        if statuses(&assessment, &grid)?
            != records[index]["states"]
                .as_array()
                .ok_or("savedstates absent")?
                .iter()
                .map(|v| v.as_str().ok_or("state invalid"))
                .collect::<Result<Vec<_>, _>>()?
            || Some(assessment.cost().total()) != records[index]["cost_bits"].as_u64()
        {
            return Err("ordinarysavedreplay changedcost/verdict".into());
        }
        replays.push(json!({"index":index,"file":p.file_name().and_then(|s|s.to_str()),"bytes":saved.len(),"sha256":sha256(&p)?,"ordinary_saved_byte_replay":true,"local":local_measure(&assessment),"run":assessment.run_measure(),"isolated_downstream_patch_KL":gam_mpd::local_kl::isolated_downstream_kl(&native.program,&decoded,&imported.contract.family,16)?}));
    }
    write_json(
        &out.join("REPORT.json"),
        &json!({"scope":scope,"records":records,"points":final_points,"selected_saved_byte_replays":replays,"assessments_attempted":measured,"Local_rejections_Run_not_measured":local_rejected,"unattempted_count":count-measured,"fidelity_unmeasured":records.iter().filter(|r|r["quality_measured"].as_bool()!=Some(true)).count(),"failed_assessments":records.iter().filter(|r|r.get("error").is_some()).count(),"Run_measured":records.iter().filter(|r|r["run_measured"].as_bool()==Some(true)).count(),"stop_reason":stopped,"run_timing":run.timing(),"codec_initialization_seconds":codec_seconds,"codec":{"stats":codec.stats(),"usage":codec.usage()},"driver_seconds":begun.elapsed().as_secs_f64(),"peak_rss_bytes":rss(),"Local_timing":"not separated: currentAPI includesfreshgraftcompile/nativeexecution; assessmentwall minusRun timing alsoincludescodec/coverage/cost","selected_followup":"expanded512contexts/strongerinterventions neededbeforebroaderclaim"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_candidate_scope_keeps_native_and_only_requested() {
        let selected: Vec<_> = (0..5137).filter(|&i| in_candidate_scope(i, Some(3789))).collect();
        assert_eq!(selected, vec![0,3789]);
        assert_eq!((0..5137).filter(|&i| in_candidate_scope(i, None)).count(),5137);
        assert_eq!((0..5137).filter(|&i| in_candidate_scope(i, Some(0))).collect::<Vec<_>>(),vec![0]);
    }
    #[test]
    fn four_shards_partition_all_heads_without_overlap() {
        for target in 0..24 {
            assert_eq!(
                (0..4)
                    .filter(|shard| in_head_range(target, shard * 6, 6))
                    .count(),
                1
            );
        }
        assert!(!in_head_range(24, 18, 6));
    }
    #[test]
    fn objective_bounds_keep_unknown_and_failed() {
        let grid = vec![Constraint {
            local: 0.01,
            run: 0.001,
        }];
        let mut r = vec![
            json!({"cost_bits":10,"states":["Verified"]}),
            json!({"cost_bits":3,"states":["Unevaluated"]}),
            json!({"cost_bits":null,"states":["Unresolved"]}),
        ];
        assert!(!optimal(&points(&r, &grid)));
        r[2]["cost_bits"] = json!(12);
        r[1]["states"] = json!(["Violates"]);
        assert!(optimal(&points(&r, &grid)));
        r[2]["cost_bits"] = json!(9);
        assert!(!optimal(&points(&r, &grid)));
    }
}
