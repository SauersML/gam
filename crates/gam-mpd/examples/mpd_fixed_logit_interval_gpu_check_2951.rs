//! OUT_JSON [PUBLIC_EXPORT_DIR] [NUMERIC_BUDGET_BYTES=1073741824]
//! Required-CUDA fixed-input interval checks; no acceptance integration.
use gam_gpu::{
    GpuPolicy,
    tensor::{CheckedInterval, CheckedScalar, Device, checked_interval_output_bytes},
};
use gam_mpd::fixed_logit_interval::*;
use serde_json::json;
#[path = "../src/fixed_logit_interval_tests.rs"]
mod rational_reference;

fn bounds(result: CheckedInterval) -> Result<(f64, f64), String> {
    match result {
        CheckedInterval::Bounded { lower, upper } => Ok((lower, upper)),
        CheckedInterval::Unresolved(reason) => Err(format!("unexpected unresolved: {reason:?}")),
    }
}

fn stored_batches(
    device: &Device,
    export: &std::path::Path,
    limit: usize,
) -> Result<serde_json::Value, String> {
    use std::io::Read;
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let shape = record["files"]["logits_row0"]["shape"]
        .as_array()
        .ok_or("stored logits shape absent")?;
    if shape.len() != 2 {
        return Err("stored logits must be matrix".into());
    }
    let rows =
        usize::try_from(shape[0].as_u64().ok_or("row count absent")?).map_err(|e| e.to_string())?;
    let classes = usize::try_from(shape[1].as_u64().ok_or("class count absent")?)
        .map_err(|e| e.to_string())?;
    if rows < 512 || classes == 0 {
        return Err("512 positive-width stored rows required".into());
    }
    let path = export.join("logits_row0.f64");
    let sha = gam_mpd::engine::sha256(&path)?;
    if Some(sha.as_str()) != record["files"]["logits_row0"]["sha256"].as_str() {
        return Err("stored logits SHA mismatch".into());
    }
    let file_bytes = rows
        .checked_mul(classes)
        .and_then(|n| n.checked_mul(8))
        .ok_or("file size overflow")?;
    if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() != file_bytes as u64 {
        return Err("stored logits byte length mismatch".into());
    }
    let mut batches = Vec::new();
    for batch in [1usize, 8, 32, 128, 512] {
        let count = batch.checked_mul(classes).ok_or("input count overflow")?;
        let bytes = count.checked_mul(8).ok_or("input bytes overflow")?;
        let output = checked_interval_output_bytes(batch).map_err(|e| e.to_string())?;
        // Peak loading/uploading: three complete numeric buffers. Four spot
        // rows retained for host checks, plus checked API output allocations.
        let numeric = bytes
            .checked_mul(3)
            .and_then(|n| n.checked_add(classes * 8 * 4))
            .and_then(|n| n.checked_add(output))
            .ok_or("numeric budget overflow")?;
        if numeric > limit {
            return Err(format!(
                "batch {batch} requires {numeric} numeric bytes beyond {limit}"
            ));
        }
        let mut raw = vec![0u8; bytes];
        std::fs::File::open(&path)
            .map_err(|e| e.to_string())?
            .read_exact(&mut raw)
            .map_err(|e| e.to_string())?;
        let z: Vec<f64> = raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("eight-byte chunk")))
            .collect();
        drop(raw);
        let w: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, v)| v + if (i % classes) % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        if z.iter().chain(&w).any(|v| !v.is_finite()) {
            return Err("stored or perturbed nonfinite logit".into());
        }
        let mut checks = Vec::new();
        for row in [0, batch - 1] {
            let begin = row * classes;
            let timer = std::time::Instant::now();
            let host = match kl_logits(&z[begin..begin + classes], &w[begin..begin + classes]) {
                Enclosure::Bounded(b) => b,
                Enclosure::Unresolved(r) => return Err(format!("host spot unresolved {r:?}")),
            };
            checks.push((row, host, timer.elapsed().as_secs_f64()));
        }
        let teacher = device
            .upload_vec(batch, classes, z)
            .map_err(|e| e.to_string())?;
        let candidate = device
            .upload_vec(batch, classes, w)
            .map_err(|e| e.to_string())?;
        let warm = device
            .checked_kl_intervals(&teacher, &candidate, output)
            .map_err(|e| e.to_string())?;
        if warm
            .iter()
            .any(|r| !matches!(r, CheckedInterval::Bounded { .. }))
        {
            return Err("public fixed-input batch unresolved".into());
        }
        drop(warm);
        let mut samples = Vec::new();
        let mut intervals = Vec::new();
        for repetition in 0..3 {
            let timer = std::time::Instant::now();
            let result = device
                .checked_kl_intervals(&teacher, &candidate, output)
                .map_err(|e| e.to_string())?;
            let seconds = timer.elapsed().as_secs_f64();
            let mut gap = 0.0_f64;
            for (row, value) in result.iter().copied().enumerate() {
                let (lo, hi) = bounds(value)?;
                gap = gap.max(hi - lo);
                if repetition == 0 {
                    intervals.push(json!({"row":row,"lower":lo,"upper":hi}));
                }
            }
            for (row, host, _) in &checks {
                let (lo, hi) = bounds(result[*row])?;
                if lo > host.hi || hi < host.lo {
                    return Err(format!(
                        "public batch {batch} row {row} disjoint host interval"
                    ));
                }
            }
            samples.push(json!({"repetition":repetition,"seconds":seconds,"rows_per_second":batch as f64/seconds,"seconds_per_row":seconds/batch as f64,"maximum_absolute_interval_gap":gap}));
        }
        let host_checks: Vec<_> = checks
            .iter()
            .map(|(row, h, t)| json!({"row":row,"lower":h.lo,"upper":h.hi,"seconds":t}))
            .collect();
        batches.push(json!({"rows":batch,"classes":classes,"input_resident_numeric_bytes":2*bytes,"peak_numeric_buffer_budget_estimate":numeric,"numeric_budget_limit":limit,"output_numeric_bytes":output,"warmup_calls":1,"samples":samples,"host_spotchecks":host_checks,"intervals":intervals}));
    }
    Ok(
        json!({"export":export,"export_json_sha256":gam_mpd::engine::sha256(&export.join("export.json"))?,"source":record["source"],"logits_file_sha256":sha,"stored_shape":[rows,classes],"perturbation":"comparison[row,column] = stored[row,column] + (column even ? 0.25 : -0.25), rounded binary64 addition; fixed declared numerical stress, not intervention or candidate fitting","batches":batches,"memory_scope":"explicit numeric buffer bound only; excludes report JSON, allocator metadata, CUDA context/module and compiler registers/spills, 2KiB on-chip shared per block","scientific_scope":"stored public-model FP32 reference widened to binary64 according to export provenance; exact fixed-input KL only; not native f64 forward parity, no intervention/fidelity/acceptance claim"}),
    )
}

fn main() -> Result<(), String> {
    let output = std::env::args().nth(1).ok_or("OUT_JSON required")?;
    let device = Device::accelerator(GpuPolicy::Required)
        .map_err(|e| e.to_string())?
        .ok_or("CUDA required")?;
    if !device.float64() {
        return Err("CUDA f64 required".into());
    }
    let compiler = device
        .checked_interval_compiler_info()
        .map_err(|e| e.to_string())?;
    let mut fixtures = Vec::new();
    for (z, w) in [
        (vec![0., 1., -2.], vec![0., 1., -2.]),
        (vec![0., 1., -2.], vec![4., 5., 2.]),
        (vec![0., -1., -8.], vec![0.2, -0.8, -7.5]),
        (vec![0., -745., -2000.], vec![-1., -744., -1999.]),
    ] {
        let teacher = device
            .upload_vec(1, z.len(), z.clone())
            .map_err(|e| e.to_string())?;
        let candidate = device
            .upload_vec(1, w.len(), w.clone())
            .map_err(|e| e.to_string())?;
        let result = device
            .checked_kl_intervals(
                &teacher,
                &candidate,
                checked_interval_output_bytes(1).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        let (lo, hi) = bounds(result[0])?;
        if fixtures.len() < 2 {
            if lo != 0.0 || hi != 0.0 {
                return Err("exact common shift not zero".into());
            }
        } else {
            let reference = rational_reference::kl_reference(&z, &w);
            rational_reference::assert_contains_reference(
                Enclosure::Bounded(Interval { lo, hi }),
                reference,
            );
        }
        fixtures.push(json!({"teacher":z,"candidate":w,"lower":lo,"upper":hi}));
    }
    let mut scalars = Vec::new();
    for (operation, values) in [
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
                operation,
                checked_interval_output_bytes(values.len()).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        for (value, result) in values.into_iter().zip(results) {
            let (lo, hi) = bounds(result)?;
            let reference = match operation {
                CheckedScalar::Exp => rational_reference::exp_reference(value),
                CheckedScalar::Log => rational_reference::log_reference(value),
            };
            if rational_reference::exact(lo) > reference.0
                || rational_reference::exact(hi) < reference.1
            {
                return Err(format!(
                    "rational scalar not enclosed {operation:?} {value}"
                ));
            }
            scalars.push(
                json!({"operation":format!("{operation:?}"),"input":value,"lower":lo,"upper":hi}),
            );
        }
    }
    for (operation, values) in [
        (CheckedScalar::Exp, vec![f64::NAN, f64::INFINITY, f64::MAX]),
        (CheckedScalar::Log, vec![0., -1., f64::NAN]),
    ] {
        let input = device
            .upload_vec(1, values.len(), values)
            .map_err(|e| e.to_string())?;
        let results = device
            .checked_scalar_intervals(
                &input,
                operation,
                checked_interval_output_bytes(3).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        if results
            .iter()
            .any(|r| matches!(r, CheckedInterval::Bounded { .. }))
        {
            return Err("invalid or unbounded scalar fabricated a finite bound".into());
        }
    }
    let invalid = device
        .upload_vec(1, 2, vec![f64::MAX, -f64::MAX])
        .map_err(|e| e.to_string())?;
    let opposite = device
        .upload_vec(1, 2, vec![-f64::MAX, f64::MAX])
        .map_err(|e| e.to_string())?;
    if matches!(
        device
            .checked_kl_intervals(
                &invalid,
                &opposite,
                checked_interval_output_bytes(1).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?[0],
        CheckedInterval::Bounded { .. }
    ) {
        return Err("overflow KL fabricated a finite interval".into());
    }
    if let Some(export) = std::env::args().nth(2) {
        let limit = std::env::args()
            .nth(3)
            .map_or(Ok(1_073_741_824usize), |s| s.parse::<usize>())
            .map_err(|e| e.to_string())?;
        let stored = stored_batches(&device, std::path::Path::new(&export), limit)?;
        let report = json!({"status":"PASS","scope":"optional fixed binary64 output distribution interval; no upstream head/GEMM/network certificate or acceptance integration","rational_fixture_gate":fixtures,"scalar_spots":scalars,"malformed_input_gate":"PASS","stored_public_logits":stored,"nvrtc":{"major":compiler.nvrtc_major,"minor":compiler.nvrtc_minor,"actual_flags":compiler.flags,"fastmath_policy":compiler.fastmath_policy}});
        return std::fs::write(
            output,
            serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string());
    }
    let classes = 50_257;
    let z: Vec<f64> = (0..classes)
        .map(|i| ((i * 17) % 128) as f64 * 0.125 - 16.)
        .collect();
    let w: Vec<f64> = z
        .iter()
        .enumerate()
        .map(|(i, v)| v + (((i * 37) % 13) as f64 - 6.) * 0.05)
        .collect();
    let host_start = std::time::Instant::now();
    let host = match kl_logits(&z, &w) {
        Enclosure::Bounded(b) => b,
        Enclosure::Unresolved(r) => return Err(format!("host unresolved {r:?}")),
    };
    let host_seconds = host_start.elapsed().as_secs_f64();
    let teacher = device
        .upload_vec(1, classes, z)
        .map_err(|e| e.to_string())?;
    let candidate = device
        .upload_vec(1, classes, w)
        .map_err(|e| e.to_string())?;
    let budget = checked_interval_output_bytes(1).map_err(|e| e.to_string())?;
    device
        .checked_kl_intervals(&teacher, &candidate, budget)
        .map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    for repetition in 0..3 {
        let start = std::time::Instant::now();
        let result = device
            .checked_kl_intervals(&teacher, &candidate, budget)
            .map_err(|e| e.to_string())?;
        let elapsed = start.elapsed().as_secs_f64();
        let (lo, hi) = bounds(result[0])?;
        if lo > host.hi || hi < host.lo {
            return Err("50k host and CUDA enclosures disjoint".into());
        }
        samples.push(json!({"repetition":repetition,"seconds":elapsed,"lower":lo,"upper":hi}));
    }
    if device.checked_kl_intervals(&teacher, &candidate, 0).is_ok() {
        return Err("output budget ignored".into());
    }
    let report = json!({"scope":"exact fixed binary64 logits only; directed CUDA basic operations, analytic tails, no forward/GEMM certificate, no acceptance integration","status":"PASS","classes":classes,"fixtures":fixtures,"scalar_spots":scalars,"host_lower":host.lo,"host_upper":host.hi,"host_seconds":host_seconds,"warmup_calls":1,"samples":samples,"nvrtc":{"major":compiler.nvrtc_major,"minor":compiler.nvrtc_minor,"actual_flags":compiler.flags,"fastmath_policy":compiler.fastmath_policy},"output_numeric_bytes":budget,"excluded_memory":"caller tensors, CUDA context/allocator, compiler registers/spills, 2KiB shared per block"});
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
