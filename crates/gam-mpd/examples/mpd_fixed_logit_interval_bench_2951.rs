//! OUT_JSON [CLASSES=50257] [REPETITIONS=3] [FIXTURE_JSON]
//! Optional fixture is {"teacher":[...],"candidate":[...]}; no model execution.
use gam_mpd::fixed_logit_interval::{Enclosure, kl_logits};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, Serialize)]
struct Fixture {
    teacher: Vec<f64>,
    candidate: Vec<f64>,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let output = args
        .get(1)
        .ok_or("OUT_JSON [CLASSES] [REPETITIONS] [FIXTURE_JSON]")?;
    let classes = args
        .get(2)
        .map_or(Ok(50_257), |s| s.parse::<usize>())
        .map_err(|e| e.to_string())?;
    let repetitions = args
        .get(3)
        .map_or(Ok(3), |s| s.parse::<usize>())
        .map_err(|e| e.to_string())?;
    if classes == 0 || repetitions == 0 {
        return Err("positive classes and repetitions required".into());
    }
    let (fixture, provenance) = if let Some(path) = args.get(4) {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        let f: Fixture = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
        if f.teacher.len() != classes || f.candidate.len() != classes {
            return Err("fixture class count mismatch".into());
        }
        (
            f,
            json!({"file":path,"scope":"caller-supplied fixed logits; no neural-forward claim"}),
        )
    } else {
        let teacher: Vec<f64> = (0..classes)
            .map(|i| ((i * 17) % 128) as f64 * 0.125 - 16.0)
            .collect();
        let candidate: Vec<f64> = teacher
            .iter()
            .enumerate()
            .map(|(i, v)| v + (((i * 37) % 13) as f64 - 6.0) * 0.05)
            .collect();
        (
            Fixture { teacher, candidate },
            json!({"scope":"deterministic synthetic vocabulary-sized fixed logits; not native model data or scientific fidelity evidence","teacher_formula":"0.125*((i*17)%128)-16","candidate_offset":"0.05*(((i*37)%13)-6)"}),
        )
    };
    if fixture
        .teacher
        .iter()
        .chain(&fixture.candidate)
        .any(|x| !x.is_finite())
    {
        return Err("finite fixture required".into());
    }
    let warmup = kl_logits(&fixture.teacher, &fixture.candidate);
    let mut samples = Vec::new();
    for repetition in 0..repetitions {
        let start = std::time::Instant::now();
        let result = kl_logits(&fixture.teacher, &fixture.candidate);
        let elapsed = start.elapsed().as_secs_f64();
        if result != warmup {
            return Err("deterministic interval changed on repeated fixed inputs".into());
        }
        match result {
            Enclosure::Bounded(bounds)=>samples.push(json!({"repetition":repetition,"seconds":elapsed,"lower":bounds.lo,"upper":bounds.hi,"absolute_gap":bounds.hi-bounds.lo})),
            Enclosure::Unresolved(reason)=>return Err(format!("fixed-input interval unresolved: {reason:?}")),
        }
    }
    // Independent operational comparison, not a transcendental proof oracle.
    let cpu_start = std::time::Instant::now();
    let (cpu, conditional) = gam_mpd::acceptance::kl_logits(
        ndarray::ArrayView1::from(&fixture.teacher),
        ndarray::ArrayView1::from(&fixture.candidate),
    );
    let cpu_seconds = cpu_start.elapsed().as_secs_f64();
    if !cpu.is_finite() || !conditional.is_finite() {
        return Err("nonfinite operational CPU comparison".into());
    }
    let report = json!({"classes":classes,"repetitions":repetitions,"warmup_calls":1,"provenance":provenance,"samples":samples,"operational_CPU_seconds":cpu_seconds,"operational_CPU_KL":cpu,"operational_CPU_conditional_comparison_error":conditional,"CPU_reference_scope":"Rust exp/log precision unspecified; existing ULP assumptions are conditional","interval_scope":"Exact fixed binary64 logits under IEEE basic arithmetic with gradual underflow; analytic scalar remainders; no upstream head/GEMM/network guarantee; no acceptance integration","auxiliary_storage":"O(1) interval state; two input vectors only; no vocabulary probability cache"});
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
