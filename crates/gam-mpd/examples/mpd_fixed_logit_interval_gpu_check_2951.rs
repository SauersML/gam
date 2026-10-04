//! Required-CUDA synthetic fixed-input interval check; no acceptance integration.
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
