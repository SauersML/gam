//! Generic complete imported LM frontier; no specialized Decoder or proposal translation.
//! EXPORT SPEC.json OUT.json. Candidates use Artifact::to_bytes format.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::decoded_intern::DecodedOperatorInterner;
use gam_mpd::device_family_run::DeviceFamilyRun;
use gam_mpd::{
    acceptance::{Change, Constraint, Edit, Episode, FamilyRun, Local, RunCheck},
    artifact::Artifact,
    candidate_frontier::{Candidate, frontier},
    import::import_language_model,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    checkpoint_sha256: String,
    sequences: usize,
    context: usize,
    nodes: usize,
    batch_rows: usize,
    budget: usize,
    max_bank: usize,
    constraints: Vec<ConstraintSpec>,
    episodes: Vec<EpisodeSpec>,
    candidates: Vec<FileCandidate>,
    #[serde(default)]
    cuda: bool,
    #[serde(default)]
    intermediate_bytes_limit: usize,
    #[serde(default)]
    compare_cpu_metrics: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstraintSpec {
    local: f64,
    run: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileCandidate {
    label: String,
    path: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EpisodeSpec {
    id: String,
    group: String,
    edits: Vec<EditSpec>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EditSpec {
    node: usize,
    width: usize,
    rows: Option<Vec<usize>>,
    columns: [usize; 2],
    scale: Option<f64>,
    add: Option<f64>,
}
fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        return Err("EXPORT SPEC.json OUT.json".into());
    }
    let out = Path::new(&a[3]);
    if out.exists() {
        return Err("report exists".into());
    }
    let spec_path = Path::new(&a[2]);
    let spec: Spec = serde_json::from_slice(&std::fs::read(spec_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if spec.sequences == 0
        || spec.context == 0
        || spec.batch_rows == 0
        || spec.episodes.is_empty()
        || spec.constraints.is_empty()
    {
        return Err("positive family/batch and nonempty episodes/constraints required".into());
    }
    if spec.max_bank == 0
        || spec
            .candidates
            .len()
            .checked_add(1)
            .ok_or("bank size overflow")?
            > spec.max_bank
    {
        return Err("declared complete bank exceeds explicit max_bank; no truncation".into());
    }
    if spec.cuda && spec.intermediate_bytes_limit == 0 {
        return Err("CUDA requires explicit positive intermediate_bytes_limit".into());
    }
    if spec.compare_cpu_metrics && !spec.cuda {
        return Err("CPU/CUDA metric comparison requires cuda=true".into());
    }
    let device = if spec.cuda {
        Some(
            Device::accelerator(GpuPolicy::Required)
                .map_err(|e| e.to_string())?
                .ok_or("CUDA required")?,
        )
    } else {
        None
    };
    let start = Instant::now();
    let imported = import_language_model(Path::new(&a[1]), spec.sequences, spec.context)?;
    if imported.record["source"]["weights_sha256"].as_str() != Some(spec.checkpoint_sha256.as_str())
        || imported.program.nodes.len() != spec.nodes
    {
        return Err("checkpoint/node-map identity mismatch".into());
    }
    let model = imported.program;
    let family = imported.contract.family;
    let interfaces = model.interfaces().map_err(|e| e.to_string())?;
    let mut episodes = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for e in spec.episodes {
        if e.id.is_empty() || e.group.is_empty() || !ids.insert(e.id.clone()) {
            return Err("empty or duplicate episode identity".into());
        }
        let mut edits = Vec::new();
        for x in e.edits {
            let width = interfaces.get(x.node).ok_or("edit node absent")?.width();
            if width != x.width
                || x.columns[0] >= x.columns[1]
                || x.columns[1] > width
                || x.rows
                    .as_ref()
                    .is_some_and(|r| r.is_empty() || r.iter().any(|&i| i >= family.rows))
            {
                return Err("edit width/range/rows invalid".into());
            }
            let change = match (x.scale, x.add) {
                (Some(v), None) if v.is_finite() => Change::Scale(v),
                (None, Some(v)) if v.is_finite() => Change::Add(v),
                _ => return Err("exactly one finite scale or add required".into()),
            };
            edits.push(Edit {
                node: x.node,
                rows: x.rows,
                columns: x.columns[0]..x.columns[1],
                change,
            });
        }
        episodes.push(Episode {
            id: e.id,
            group: e.group,
            edits,
        });
    }
    let local = Local::new(&model, family.clone(), None, spec.batch_rows);
    let run = FamilyRun {
        model: &model,
        family,
        readouts: 1,
        episodes,
    };
    let native_artifact = Artifact::native(&model)?.f32_literals()?;
    let decoded_native = Artifact::from_bytes(&native_artifact.to_bytes()?, &model.declarations)?;
    drop(native_artifact);
    let interner = DecodedOperatorInterner::new(&decoded_native)?;
    let mut bank = vec![Candidate {
        label: "native".into(),
        artifact: decoded_native,
    }];
    let mut labels = std::collections::BTreeSet::from(["native".to_string()]);
    for c in spec.candidates {
        if c.label.is_empty() || !labels.insert(c.label.clone()) {
            return Err("empty or duplicate candidate label".into());
        }
        let path = if c.path.is_absolute() {
            c.path
        } else {
            spec_path.parent().unwrap_or(Path::new(".")).join(c.path)
        };
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut artifact = Artifact::from_bytes(&bytes, &model.declarations)?;
        drop(bytes);
        artifact.validate_coverage(&model)?;
        let shared = interner.intern(&mut artifact);
        eprintln!(
            "loaded {}: {shared}/{} operators shared with decoded native",
            c.label,
            artifact.program.operators.len()
        );
        bank.push(Candidate {
            label: c.label,
            artifact,
        });
    }
    let constraints: Vec<Constraint> = spec
        .constraints
        .iter()
        .map(|c| Constraint {
            local: c.local,
            run: c.run,
        })
        .collect();
    let device_run = if let Some(device) = device {
        Some(DeviceFamilyRun::new(
            &run,
            device,
            spec.intermediate_bytes_limit,
        )?)
    } else {
        None
    };
    let parity = if spec.compare_cpu_metrics {
        let device = device_run
            .as_ref()
            .ok_or("CPU/CUDA metric comparison requires cuda=true")?;
        Some(
            json!({"cpu":run.episodes(&bank[0].artifact)?,"cuda":device.episodes(&bank[0].artifact)?,"scope":"native bank member only, same declared episodes; measured discrepancy, no acceptance threshold"}),
        )
    } else {
        None
    };
    let runner: &dyn RunCheck = device_run
        .as_ref()
        .map_or(&run as &dyn RunCheck, |r| r as &dyn RunCheck);
    let result = frontier(&local, runner, bank, &constraints, spec.budget)?;
    let measures:Vec<_>=result.assessments.iter().map(|a|match a {Some(Ok(v))=>json!({"local":v.local_measure(),"run_episodes":v.run_measure().map(|r|&r.episodes),"run_groups":v.run_measure().map(|r|&r.groups),"state":if v.run_measure().is_some(){"measured"}else{"local_rejected_run_not_measured"}}),Some(Err(e))=>json!({"error":e}),None=>serde_json::Value::Null}).collect();
    let report = json!({"native_episode_cpu_cuda_comparison":parity,"backend":device_run.as_ref().map_or("CPU",|r|r.backend_name()),"source":imported.record["source"],"config":imported.record["config"],"node_map":"exact imported full program indices; validated widths; fixed native output","sequences":spec.sequences,"context":spec.context,"points":result.points,"measured_candidates":result.measured_candidates,"measures":measures,"seconds":start.elapsed().as_secs_f64(),"scope":"declared finite bank and fixed input/intervention family only"});
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", out.display());
    Ok(())
}
