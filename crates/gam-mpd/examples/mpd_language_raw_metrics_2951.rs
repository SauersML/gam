//! EXPORT SPEC OUT_JSON NORMALIZED_REPORT SPOTCHECKS ARTIFACT...; optional raw CUDA fixed-logit metrics.
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
    let malformed = device
        .upload_vec(1, 3, vec![f64::NAN, 0.0, 1.0])
        .map_err(|e| e.to_string())?;
    if !matches!(
        device
            .checked_kl_intervals(
                &malformed,
                &q,
                checked_interval_output_bytes(1).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?[0],
        CheckedInterval::Unresolved(_)
    ) {
        return Err("nonfinite fixed input fabricated a bounded metric".into());
    }
    if device.checked_kl_intervals(&p, &q, 0).is_ok() {
        return Err("checked output budget ignored".into());
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
    if a.len() < 7 {
        return Err("EXPORT SPEC OUT_JSON NORMALIZED_REPORT SPOTCHECKS ARTIFACT...".into());
    }
    let started = std::time::Instant::now();
    let export = Path::new(&a[1]);
    let spec_path = Path::new(&a[2]);
    let baseline_bytes = std::fs::read(&a[4]).map_err(|e| e.to_string())?;
    let baseline: serde_json::Value =
        serde_json::from_slice(&baseline_bytes).map_err(|e| e.to_string())?;
    if baseline["passes"].as_bool() != Some(true) {
        return Err("normalized-array baseline not PASS".into());
    }
    let spotchecks = match a[5].as_str() {
        "yes" => true,
        "no" => false,
        _ => return Err("SPOTCHECKS must yes|no".into()),
    };
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut export_hashes = serde_json::Map::new();
    for (name, file) in record["files"]
        .as_object()
        .ok_or("export manifest absent")?
    {
        let actual = gam_mpd::coder_capture::sha256(&export.join(format!("{name}.f64")))?;
        if file["sha256"]
            .as_str()
            .is_some_and(|declared| declared != actual)
        {
            return Err(format!("export hash mismatch {name}"));
        }
        if baseline["verified_export_files_sha256"][name].as_str() != Some(actual.as_str()) {
            return Err(format!("baseline export mismatch {name}"));
        }
        export_hashes.insert(name.clone(), json!(actual));
    }
    let spec_sha = gam_mpd::coder_capture::sha256(spec_path)?;
    if baseline["spec_sha256"].as_str() != Some(spec_sha.as_str()) {
        return Err("baseline panel mismatch".into());
    }
    let decoder = Decoder::from_export(export)?;
    let spec = Spec::load(spec_path, &decoder)?;
    let passages = passages(export, spec.rows)?;
    let native = split_sites(&import_language_model(export, 1, 1)?.program)?;
    let mut artifacts = Vec::new();
    let mut hashes = Vec::new();
    for (index, path) in a[6..].iter().enumerate() {
        let hash = gam_mpd::coder_capture::sha256(Path::new(path))?;
        if baseline["artifact_files"][index]["sha256"].as_str() != Some(hash.as_str()) {
            return Err("baseline artifact mismatch".into());
        }
        let artifact = Artifact::from_bytes(
            &std::fs::read(path).map_err(|e| e.to_string())?,
            &native.declarations,
        )?;
        artifact.validate_coverage(&native)?;
        artifacts.push(artifact);
        hashes.push(json!({"file":path,"sha256":hash}));
    }
    let interner = gam_mpd::decoded_intern::DecodedOperatorInterner::new(
        artifacts.first().ok_or("native first required")?,
    )?;
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
    if runner
        .checked_raw_metric_episodes(
            &artifacts[0],
            Budget {
                batch_rows: 128,
                workspace_bytes: 1,
            },
            false,
        )
        .is_ok()
    {
        return Err("raw numeric budget ignored".into());
    }
    let setup_seconds = started.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let mut passes = true;
    for (index, artifact) in artifacts.iter().enumerate() {
        let before = runner.timing();
        let timer = std::time::Instant::now();
        let raw = runner.checked_raw_metric_episodes(artifact, budget, spotchecks)?;
        let wall = timer.elapsed().as_secs_f64();
        let after = runner.timing();
        let original = &baseline["results"][index]["fast"];
        if original["episodes"]
            .as_array()
            .ok_or("baseline episodes absent")?
            .len()
            != raw.episodes.len()
        {
            return Err("baseline episode count mismatch".into());
        }
        let mut maximum_native_effect_difference = 0.0_f64;
        let mut maximum_interval_separation = 0.0_f64;
        for (episode, old) in raw
            .episodes
            .iter()
            .zip(original["episodes"].as_array().ok_or("baseline episodes")?)
        {
            passes &= matches!(episode.kl, Outcome::Bounded { .. }) && episode.top1_defined;
            passes &= old["id"].as_str() == Some(&episode.id)
                && old["group"].as_str() == Some(&episode.group)
                && old["scored_from"].as_u64() == Some(episode.scored_from as u64)
                && old["scored_until"].as_u64() == Some(episode.scored_until as u64)
                && old["unheld"].as_u64() == Some(episode.unheld as u64)
                && old["top1_agree"].as_f64() == Some(episode.top1_agree);
            let old_effect = old["native_effect_cpu_metric"]
                .as_f64()
                .ok_or("baseline effect absent")?;
            maximum_native_effect_difference = maximum_native_effect_difference
                .max((old_effect - episode.native_effect_cpu_metric).abs());
            passes &= old_effect == episode.native_effect_cpu_metric;
            if spotchecks {
                passes &= episode.host_analytic_spotcheck.as_ref().is_some_and(|c| {
                    c.rows > 0 && c.disjoint == 0 && c.unresolved == 0 && c.top1_mismatches == 0
                });
            }
            if let Outcome::Bounded { lower, upper } = episode.kl {
                let lo = old["kl"]["Bounded"]["lower"]
                    .as_f64()
                    .ok_or("baseline interval lower")?;
                let hi = old["kl"]["Bounded"]["upper"]
                    .as_f64()
                    .ok_or("baseline interval upper")?;
                maximum_interval_separation =
                    maximum_interval_separation.max((lower - hi).max(lo - upper).max(0.0));
            }
        }
        let status = |lo: f64, hi: f64, epsilon: f64| {
            if hi <= epsilon {
                "Feasible"
            } else if lo > epsilon {
                "Infeasible"
            } else {
                "Unresolved"
            }
        };
        let mut classifications = Vec::new();
        for group in &raw.groups {
            let old = original["groups"]
                .as_array()
                .ok_or("baseline groups absent")?
                .iter()
                .find(|g| g["name"].as_str() == Some(group.name.as_str()))
                .ok_or("baseline group unmatched")?;
            let lo = old["mean_kl"]["Bounded"]["lower"]
                .as_f64()
                .ok_or("baseline group lower")?;
            let hi = old["mean_kl"]["Bounded"]["upper"]
                .as_f64()
                .ok_or("baseline group upper")?;
            if let Outcome::Bounded { lower, upper } = group.mean_kl {
                for epsilon in [0.001, 0.01, 0.1, 1.0, 10.0] {
                    let prior = status(lo, hi, epsilon);
                    let new = status(lower, upper, epsilon);
                    passes &= prior == new;
                    classifications.push(json!({"group":group.name,"epsilon":epsilon,"normalized_fixed_input":prior,"raw_fixed_input":new,"raw_lower":lower,"raw_upper":upper,"normalized_lower":lo,"normalized_upper":hi}));
                }
            }
        }
        let timer = std::time::Instant::now();
        let warm = runner.checked_raw_metric_episodes(artifact, budget, false)?;
        let warm_wall = timer.elapsed().as_secs_f64();
        for (left, right) in raw.episodes.iter().zip(&warm.episodes) {
            passes &=
                left.id == right.id && left.top1_agree == right.top1_agree && right.top1_defined;
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
        results.push(json!({"artifact":hashes[index],"first_arm_host_spotchecks":spotchecks,"first_arm_wall_seconds":wall,"warm_production_wall_seconds":warm_wall,"raw":raw,"warm":warm,"classifications":classifications,"maximum_native_effect_difference":maximum_native_effect_difference,"maximum_raw_vs_normalized_interval_separation":maximum_interval_separation,"timing_before":before,"timing_after_first":after,"timing_after_warm":runner.timing()}));
    }
    let report = json!({"passes":passes,"episode_count":spec.episodes.len(),"context":spec.rows,"export":export,"verified_export_files_sha256":export_hashes,"spec_sha256":spec_sha,"baseline_report_sha256":gam_mpd::coder_capture::sha256(Path::new(&a[4]))?,"artifact_files":hashes,"setup_seconds":setup_seconds,"gate":gate,"results":results,"nvrtc":{"major":compiler.nvrtc_major,"minor":compiler.nvrtc_minor,"actual_flags":compiler.flags,"fastmath_policy":compiler.fastmath_policy},"scope":"Optional exact fixed raw CUDA binary64 head-logit intervals. Scored raw metric tiles with oracle disabled download only row bounds and argmax; no CPU normalization/reupload in that phase. Cold teacher/nativeeffect preparation retains existing CPU normalization/full-vocabulary transfers, reused from immutable caches in warm calls. Independent host analytic first/last-row spotchecks use identical downloaded raw values only when enabled. Upstream RMS/gain/GEMM/network rounding excluded; raw inputs differ from former CPU-normalized arrays. Frozen classification/top1/nativeeffect compatibility is measured, not bit-identical KL. Default acceptance unchanged. Warm wall repeats candidate execution with immutable teacher cache; no cold/warm speed comparison."});
    std::fs::write(
        &a[3],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !passes {
        return Err("raw head metric parity failed; inspect saved report".into());
    }
    Ok(())
}
