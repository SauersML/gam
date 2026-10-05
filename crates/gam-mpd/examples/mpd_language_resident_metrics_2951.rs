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
    run_check::{LanguageRun, ResidentBudget, ResidentMeasure, split_sites},
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
    if a.len() < 8 {
        return Err(
            "EXPORT SPEC OUT_JSON NORMALIZED_REPORT RAW_REPORT SPOTCHECKS ARTIFACT...".into(),
        );
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
    let spotchecks = match a[6].as_str() {
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
        let actual = gam_mpd::engine::sha256(&export.join(format!("{name}.f64")))?;
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
    let spec_sha = gam_mpd::engine::sha256(spec_path)?;
    if baseline["spec_sha256"].as_str() != Some(spec_sha.as_str()) {
        return Err("baseline panel mismatch".into());
    }
    let decoder = Decoder::from_export(export)?;
    let spec = Spec::load(spec_path, &decoder)?;
    let passages = passages(export, spec.rows)?;
    let native = split_sites(&import_language_model(export, 1, 1)?.program)?;
    let mut artifacts = Vec::new();
    let mut hashes = Vec::new();
    for (index, path) in a[7..].iter().enumerate() {
        let hash = gam_mpd::engine::sha256(Path::new(path))?;
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
    let raw_baseline: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&a[5]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if raw_baseline["passes"].as_bool() != Some(true)
        || raw_baseline["spec_sha256"] != baseline["spec_sha256"]
        || raw_baseline["artifact_files"] != baseline["artifact_files"]
        || raw_baseline["verified_export_files_sha256"] != baseline["verified_export_files_sha256"]
    {
        return Err("raw baseline provenance differs".into());
    }
    let budget = ResidentBudget {
        aggregate_numeric_bytes: 12 << 30,
        teacher_bytes: 512 << 20,
        operator_bytes: 2 << 30,
        edit_bytes: 512 << 20,
        metric: Budget {
            batch_rows: 128,
            workspace_bytes: 1 << 30,
        },
    };

    let setup_seconds = started.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let mut passes = true;
    for (index, artifact) in artifacts.iter().enumerate() {
        let prepare_timer = std::time::Instant::now();
        let prepared = runner.prepare_resident_candidate(artifact, budget)?;
        let preparation_seconds = prepare_timer.elapsed().as_secs_f64();
        let cold = prepared.measure_resident(spotchecks)?;
        let warm = prepared.measure_resident(false)?;
        let normalized = &baseline["results"][index]["fast"];
        let raw = &raw_baseline["results"][index]["warm"];
        let cold_comparison = compare(&cold, normalized, raw, spotchecks)?;
        let warm_comparison = compare(&warm, normalized, raw, false)?;
        passes &= cold_comparison["passes"].as_bool() == Some(true)
            && warm_comparison["passes"].as_bool() == Some(true);
        passes &= warm.transfers.residual_upload_bytes == 0
            && warm.transfers.raw_oracle_download_bytes == 0
            && warm.transfers.edit_constant_upload_bytes == 0
            && warm.transfers.final_residual_download_bytes == 0
            && warm.transfers.donor_download_bytes == 0;
        results.push(json!({"artifact":hashes[index],"preparation_seconds":preparation_seconds,"cold":cold,"warm":warm,"cold_comparison":cold_comparison,"warm_comparison":warm_comparison}));
    }
    let report = json!({"passes":passes,"episode_count":spec.episodes.len(),"context":spec.rows,"export":export,"verified_export_files_sha256":export_hashes,"spec_sha256":spec_sha,"normalized_baseline_sha256":gam_mpd::engine::sha256(Path::new(&a[4]))?,"raw_baseline_sha256":gam_mpd::engine::sha256(Path::new(&a[5]))?,"artifact_files":hashes,"setup_seconds":setup_seconds,"budget":budget,"gate":gate,"results":results,"nvrtc":{"major":compiler.nvrtc_major,"minor":compiler.nvrtc_minor,"actual_flags":compiler.flags,"fastmath_policy":compiler.fastmath_policy},"scope":"Opt-in actual-f64 CUDA native teacher, candidate and head with analytic checked fixed-raw-logit KL. All original mapped interventions, clean donor semantics, scored suffixes and group identities are retained. Upstream neural/RMS/GEMM rounding is excluded from fixed-input intervals. Native effects now have independently labelled raw intervals rather than CPU metrics; observed differences are reported. Cold and warm classifications against frozen normalized and raw baselines are checked independently. No default acceptance switch. Recorded residual/donor/vocabulary transfers cover these evaluator paths; source weight, edit constant and token/layout initialization are separate. Numeric memory counts exclude library/context/allocator/register overhead."});
    std::fs::write(
        &a[3],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !passes {
        return Err("resident evaluator parity failed; inspect saved report".into());
    }
    Ok(())
}
fn bounds(value: &serde_json::Value) -> Result<(f64, f64), String> {
    Ok((
        value["Bounded"]["lower"]
            .as_f64()
            .ok_or("baseline lower absent")?,
        value["Bounded"]["upper"]
            .as_f64()
            .ok_or("baseline upper absent")?,
    ))
}
fn status(lo: f64, hi: f64, epsilon: f64) -> &'static str {
    if hi <= epsilon {
        "Feasible"
    } else if lo > epsilon {
        "Infeasible"
    } else {
        "Unresolved"
    }
}
fn compare(
    measure: &ResidentMeasure,
    normalized: &serde_json::Value,
    raw: &serde_json::Value,
    spotchecks: bool,
) -> Result<serde_json::Value, String> {
    let mut passes = true;
    let mut max_effect_difference = 0.0_f64;
    let mut max_separation = 0.0_f64;
    for old in [normalized, raw] {
        let episodes = old["episodes"]
            .as_array()
            .ok_or("baseline episodes absent")?;
        let groups = old["groups"].as_array().ok_or("baseline groups absent")?;
        if episodes.len() != measure.episodes.len() || groups.len() != measure.groups.len() {
            return Err("resident baseline episode/group cardinality differs".into());
        }
        for (episode, prior) in measure.episodes.iter().zip(episodes) {
            passes &= episode.top1_defined
                && prior["id"].as_str() == Some(&episode.id)
                && prior["group"].as_str() == Some(&episode.group)
                && prior["scored_from"].as_u64() == Some(episode.scored_from as u64)
                && prior["scored_until"].as_u64() == Some(episode.scored_until as u64)
                && prior["unheld"].as_u64() == Some(episode.unheld as u64)
                && prior["top1_agree"].as_f64() == Some(episode.top1_agree);
            let (lo, hi) = bounds(&prior["kl"])?;
            if let Outcome::Bounded { lower, upper } = episode.kl {
                max_separation = max_separation.max((lower - hi).max(lo - upper).max(0.));
            } else {
                passes = false;
            }
            let prior_effect = prior["native_effect_cpu_metric"]
                .as_f64()
                .ok_or("baseline native effect absent")?;
            if let Outcome::Bounded { lower, upper } = episode.native_effect_raw {
                max_effect_difference =
                    max_effect_difference.max(((lower + upper) / 2. - prior_effect).abs());
            } else {
                passes = false;
            }
            if spotchecks {
                passes &= episode.host_analytic_spotcheck.as_ref().is_some_and(|s| {
                    s.rows > 0 && s.disjoint == 0 && s.unresolved == 0 && s.top1_mismatches == 0
                });
            }
        }
    }
    let mut classifications = Vec::new();
    for group in &measure.groups {
        let n = normalized["groups"]
            .as_array()
            .ok_or("normalized groups")?
            .iter()
            .find(|g| g["name"].as_str() == Some(&group.name))
            .ok_or("normalized group unmatched")?;
        let r = raw["groups"]
            .as_array()
            .ok_or("raw groups")?
            .iter()
            .find(|g| g["name"].as_str() == Some(&group.name))
            .ok_or("raw group unmatched")?;
        passes &= n["episodes"].as_u64() == Some(group.episodes as u64)
            && r["episodes"].as_u64() == Some(group.episodes as u64);
        let (nl, nu) = bounds(&n["mean_kl"])?;
        let (rl, ru) = bounds(&r["mean_kl"])?;
        if let Outcome::Bounded { lower, upper } = group.mean_kl {
            for epsilon in [0.001, 0.01, 0.1, 1., 10.] {
                let current = status(lower, upper, epsilon);
                let normalized = status(nl, nu, epsilon);
                let raw = status(rl, ru, epsilon);
                passes &= current == normalized && current == raw;
                classifications.push(json!({"group":group.name,"epsilon":epsilon,"resident":current,"normalized":normalized,"raw":raw,"lower":lower,"upper":upper}));
            }
        } else {
            passes = false;
        }
    }
    Ok(
        json!({"passes":passes,"classifications":classifications,"maximum_native_effect_difference":max_effect_difference,"maximum_metric_interval_separation":max_separation,"numerical_scope":"Effects and KL fixed input arrays differ when native CPU forward is replaced by CUDA f64; observed midpoint differences/interval separations are diagnostic and do not certify neural forward."}),
    )
}
