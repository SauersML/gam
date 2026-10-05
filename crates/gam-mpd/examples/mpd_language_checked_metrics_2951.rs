//! EXPORT SPEC OUT_JSON ARTIFACT...; explicit CUDA checked metrics, default acceptance unchanged.
use gam_gpu::{
    GpuPolicy,
    tensor::{CheckedInterval, CheckedScalar, Device, checked_interval_output_bytes},
};
use gam_mpd::{
    artifact::Artifact,
    counterfactual::{Decoder, Spec, passages},
    fixed_logit_interval::*,
    fixed_metric_device::{Budget, Outcome},
    import::import_language_model,
    run_check::{LanguageRun, split_sites},
};
use serde_json::json;
use std::path::Path;
#[path = "../src/fixed_logit_interval_tests.rs"]
mod rational_reference;

fn bounded(value: CheckedInterval) -> Result<(f64, f64), String> {
    match value {
        CheckedInterval::Bounded { lower, upper } => Ok((lower, upper)),
        CheckedInterval::Unresolved(r) => Err(format!("rational gate unresolved {r:?}")),
    }
}
fn rational_gate(device: &Device) -> Result<serde_json::Value, String> {
    let z = vec![0.0, -1.0, -8.0];
    let w = vec![0.2, -0.8, -7.5];
    let p = device
        .upload_vec(1, 3, z.clone())
        .map_err(|e| e.to_string())?;
    let q = device
        .upload_vec(1, 3, w.clone())
        .map_err(|e| e.to_string())?;
    let (lo, hi) = bounded(
        device
            .checked_kl_intervals(
                &p,
                &q,
                checked_interval_output_bytes(1).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?[0],
    )?;
    rational_reference::assert_contains_reference(
        Enclosure::Bounded(Interval::new(lo, hi)),
        rational_reference::kl_reference(&z, &w),
    );
    for (op, values) in [
        (CheckedScalar::Exp, vec![-745., -1., 0., 1., 709.]),
        (
            CheckedScalar::Log,
            vec![f64::from_bits(1), 0.5, 1., 2., f64::MAX],
        ),
    ] {
        let input = device
            .upload_vec(1, values.len(), values.clone())
            .map_err(|e| e.to_string())?;
        let results = device
            .checked_scalar_intervals(
                &input,
                op,
                checked_interval_output_bytes(values.len()).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        for (x, result) in values.into_iter().zip(results) {
            let (lo, hi) = bounded(result)?;
            let reference = match op {
                CheckedScalar::Exp => rational_reference::exp_reference(x),
                CheckedScalar::Log => rational_reference::log_reference(x),
            };
            rational_reference::assert_contains_reference(
                Enclosure::Bounded(Interval::new(lo, hi)),
                reference,
            );
        }
    }
    // Exercise production packing, partial last batch, exact row domain and budget.
    let resident = gam_mpd::fixed_metric_device::Resident::new(
        device.clone(),
        3,
        3,
        Budget {
            batch_rows: 2,
            workspace_bytes: 1 << 20,
        },
        true,
    )?;
    let p = gam_mpd::native_readout::normalize_logits(ndarray::arr2(&[
        [0., -1., -8.],
        [1., 2., 3.],
        [-2., 0., 4.],
    ]))?;
    let q = gam_mpd::native_readout::normalize_logits(ndarray::arr2(&[
        [0.2, -0.8, -7.5],
        [1.2, 1.8, 3.1],
        [-1., 0.5, 3.],
    ]))?;
    let mut stream = resident.stream();
    stream.append(&p, &q, 0)?;
    if stream.append(&p, &q, 4).is_ok() {
        return Err("row-domain gap ignored".into());
    }
    let episode = stream.finish("fixture".into(), "fixture".into(), 0, 3, 12.0, 0)?;
    let Outcome::Bounded { lower, upper } = episode.kl else {
        return Err("production fixture unresolved".into());
    };
    let mut reference = (
        num_rational::BigRational::from_integer(0.into()),
        num_rational::BigRational::from_integer(0.into()),
    );
    for row in 0..3 {
        let r = rational_reference::kl_reference(
            p.row(row).as_slice().ok_or("p fixture layout")?,
            q.row(row).as_slice().ok_or("q fixture layout")?,
        );
        reference.0 += r.0;
        reference.1 += r.1;
    }
    let three = num_rational::BigRational::from_integer(3.into());
    reference.0 /= &three;
    reference.1 /= &three;
    rational_reference::assert_contains_reference(
        Enclosure::Bounded(Interval::new(lower, upper)),
        reference,
    );
    if gam_mpd::fixed_metric_device::Resident::new(
        device.clone(),
        3,
        3,
        Budget {
            batch_rows: 2,
            workspace_bytes: 1,
        },
        false,
    )
    .is_ok()
    {
        return Err("metric workspace budget ignored".into());
    }
    Ok(
        json!({"status":"PASS","independent_exact_rational_scalar_and_KL":true,"production_normalized_fixture":episode,"row_domain_gap_rejected":true,"workspace_budget_rejected":true}),
    )
}
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        return Err("EXPORT SPEC OUT_JSON ARTIFACT...".into());
    }
    let started = std::time::Instant::now();
    let export = Path::new(&a[1]);
    let spec_path = Path::new(&a[2]);
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut export_hashes = serde_json::Map::new();
    for (name, file) in record["files"]
        .as_object()
        .ok_or("export manifest absent")?
    {
        let path = export.join(format!("{name}.f64"));
        let actual = gam_mpd::engine::sha256(&path)?;
        if file["sha256"]
            .as_str()
            .is_some_and(|declared| declared != actual)
        {
            return Err(format!("export hash mismatch {name}"));
        }
        export_hashes.insert(name.clone(), json!(actual));
    }
    let decoder = Decoder::from_export(export)?;
    let spec = Spec::load(spec_path, &decoder)?;
    let passages = passages(export, spec.rows)?;
    let native = split_sites(&import_language_model(export, 1, 1)?.program)?;
    let mut artifacts = Vec::new();
    let mut artifact_hashes = Vec::new();
    for path in &a[4..] {
        let hash = gam_mpd::engine::sha256(Path::new(path))?;
        let artifact = Artifact::from_bytes(
            &std::fs::read(path).map_err(|e| e.to_string())?,
            &native.declarations,
        )?;
        artifact.validate_coverage(&native)?;
        artifact_hashes.push(json!({"file":path,"sha256":hash}));
        artifacts.push(artifact);
    }
    let source = artifacts.first().ok_or("native artifact first required")?;
    let interner = gam_mpd::decoded_intern::DecodedOperatorInterner::new(source)?;
    for artifact in &mut artifacts[1..] {
        interner.intern(artifact);
    }
    let device = Device::accelerator(GpuPolicy::Required)
        .map_err(|e| e.to_string())?
        .ok_or("CUDA required")?;
    let gate = rational_gate(&device)?;
    let compiler = device
        .checked_interval_compiler_info()
        .map_err(|e| e.to_string())?;
    let runner = LanguageRun::new(&decoder, &native, &spec, &passages, 1)?
        .with_cuda(device, 2 << 30)?
        .with_cuda_native_source(&artifacts[0])?
        .with_cuda_readout(gam_mpd::native_readout::Budget {
            resident_bytes: 512 << 20,
            workspace_bytes: 256 << 20,
        })?;
    let budget = Budget {
        batch_rows: 128,
        workspace_bytes: 1 << 30,
    };
    let setup_seconds = started.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let mut passes = true;
    for (index, artifact) in artifacts.iter().enumerate() {
        let timer = std::time::Instant::now();
        let paired = runner.checked_metric_episodes(artifact, budget, true)?;
        let paired_wall = timer.elapsed().as_secs_f64();
        if paired.episodes.len() != spec.episodes.len() {
            return Err("episode count changed".into());
        }
        for episode in &paired.episodes {
            passes &= matches!(episode.kl, Outcome::Bounded { .. });
            passes &= episode
                .cpu_comparison
                .as_ref()
                .is_some_and(|c| c.conditional_intervals_disjoint == 0);
        }
        let mut classifications = Vec::new();
        for group in &paired.groups {
            let es: Vec<_> = paired
                .episodes
                .iter()
                .filter(|e| e.group == group.name)
                .collect();
            let count = es.len() as f64;
            let cpu = es
                .iter()
                .map(|e| e.cpu_comparison.as_ref().map_or(f64::NAN, |c| c.mean))
                .sum::<f64>()
                / count;
            let error = es
                .iter()
                .map(|e| {
                    e.cpu_comparison
                        .as_ref()
                        .map_or(f64::NAN, |c| c.conditional_error)
                })
                .sum::<f64>()
                / count;
            if let Outcome::Bounded { lower, upper } = group.mean_kl {
                for epsilon in [0.001, 0.01, 0.1, 1.0, 10.0] {
                    let status = |lo: f64, hi: f64| {
                        if hi <= epsilon {
                            "Feasible"
                        } else if lo > epsilon {
                            "Infeasible"
                        } else {
                            "Unresolved"
                        }
                    };
                    let cpu_lower = if error == 0.0 {
                        cpu.max(0.0)
                    } else {
                        (cpu - error).next_down().max(0.0)
                    };
                    let cpu_upper = if error == 0.0 {
                        cpu
                    } else {
                        (cpu + error).next_up()
                    };
                    let cpu_status = status(cpu_lower, cpu_upper);
                    let gpu_status = status(lower, upper);
                    passes &= cpu_status == gpu_status;
                    classifications.push(json!({"group":group.name,"epsilon":epsilon,"CPU_conditional":cpu_status,"checked_fixed_input":gpu_status,"CPU_mean":cpu,"CPU_conditional_error":error,"checked_lower":lower,"checked_upper":upper}));
                }
            }
        }
        let after_paired = runner.timing();
        let timer = std::time::Instant::now();
        let fast = runner.checked_metric_episodes(artifact, budget, false)?;
        let fast_wall = timer.elapsed().as_secs_f64();
        for (left, right) in paired.episodes.iter().zip(&fast.episodes) {
            passes &= left.id == right.id
                && left.group == right.group
                && left.scored_from == right.scored_from
                && left.scored_until == right.scored_until
                && left.top1_agree == right.top1_agree
                && left.unheld == right.unheld
                && left.native_effect_cpu_metric == right.native_effect_cpu_metric;
            if let (
                Outcome::Bounded { lower: a, upper: b },
                Outcome::Bounded { lower: c, upper: d },
            ) = (&left.kl, &right.kl)
            {
                passes &= *a <= *d && *c <= *b;
            } else {
                passes = false;
            }
        }
        results.push(json!({"artifact":artifact_hashes[index],"paired_exact_same_array_CPU_reference_wall_seconds":paired_wall,"checked_metric_warm_teacher_CPU_reference_disabled_wall_seconds":fast_wall,"paired":paired,"fast":fast,"classifications":classifications,"cumulative_timing_after_paired":after_paired,"cumulative_timing_after_fast":runner.timing()}));
    }
    let report = json!({"passes":passes,"episode_count":spec.episodes.len(),"context":spec.rows,"source":record["source"],"export":export,"export_json_sha256":gam_mpd::engine::sha256(&export.join("export.json"))?,"verified_export_files_sha256":export_hashes,"spec":spec_path,"spec_sha256":gam_mpd::engine::sha256(spec_path)?,"artifact_files":artifact_hashes,"setup_seconds":setup_seconds,"gate":gate,"results":results,"nvrtc":{"major":compiler.nvrtc_major,"minor":compiler.nvrtc_minor,"actual_flags":compiler.flags,"fastmath_policy":compiler.fastmath_policy},"scope":"Typed optional production metric endpoint only; unchanged readout/CPU normalization/teacher/native effects/top1; exact fixed normalized binary64 arrays softmax-renormalized by both CPU helper and CUDA checked metric; no default acceptance integration or upstream real arithmetic guarantee. CPU reference retains conditional exp/log ULP assumptions. Paired validation CPU reference time is not fast-path timing. Fast arm reuses immutable teachers and executes candidates again; no matched end-to-end speedup claimed."});
    std::fs::write(
        &a[3],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !passes {
        return Err("checked production parity gate failed; inspect saved report".into());
    }
    Ok(())
}
