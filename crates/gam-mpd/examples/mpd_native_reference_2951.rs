//! Lightweight native f64 forward versus archived HF logits, no artifact roundtrip.
use gam_mpd::{
    coder_capture::sha256, counterfactual::read_f64_matrix, import::import_language_model,
};
use serde_json::json;
use std::{path::Path, time::Instant};
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("EXPORT OUT.json CONTEXT".into());
    }
    let p = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    if out.exists() {
        return Err("output exists".into());
    }
    let n: usize = a[3].parse().map_err(|e| format!("context:{e}"))?;
    if n == 0 {
        return Err("positive context required".into());
    }
    let t = Instant::now();
    let imported = import_language_model(p, 1, n)?;
    let vocab = imported.record["config"]["vocab"].as_u64().ok_or("vocab")? as usize;
    let reference = read_f64_matrix(&p.join("logits_row0.f64"), vocab)?;
    if reference.nrows() < n {
        return Err("reference shorter than requested context".into());
    }
    let trace = imported
        .program
        .execute(&imported.contract.family, false)
        .map_err(|e| e.to_string())?;
    let y = &trace.values[imported.program.output];
    let mut rows = Vec::new();
    let mut max = 0f64;
    for i in 0..n {
        let mut err = 0f64;
        let mut square = 0f64;
        for j in 0..vocab {
            let x = y[[i, j]];
            let z = reference[[i, j]];
            if !x.is_finite() || !z.is_finite() {
                return Err("nonfinite logits".into());
            }
            let d = (x - z).abs();
            err = err.max(d);
            square += d * d;
        }
        let native_row = y.row(i);
        let reference_row = reference.row(i);
        let top = |r: ndarray::ArrayView1<'_, f64>| -> Result<usize, String> {
            r.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|v| v.0)
                .ok_or("empty row".into())
        };
        let log_z = |r: ndarray::ArrayView1<'_, f64>| {
            let peak = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            peak + r.iter().map(|x| (x - peak).exp()).sum::<f64>().ln()
        };
        let zn = log_z(native_row);
        let zr = log_z(reference_row);
        let kl = reference_row
            .iter()
            .zip(native_row.iter())
            .map(|(&r, &q)| (r - zr).exp() * ((r - zr) - (q - zn)))
            .sum::<f64>();
        let top1_agreement = top(native_row)? == top(reference_row)?;
        max = max.max(err);
        rows.push(json!({"position":i,"max_absolute_logit_difference":err,"rms_logit_difference":(square/vocab as f64).sqrt(),"hf_to_rust_kl":kl,"top1_agreement":top1_agreement}));
    }
    let report = json!({"scope":"Native original f64 weights vs archived Hugging Face fp32 forward widened to f64. Numerical comparison, no full neural certificate or serialized candidate check.","source":imported.record,"export_sha256":sha256(&p.join("export.json"))?,"reference_sha256":sha256(&p.join("logits_row0.f64"))?,"positions":n,"vocabulary":vocab,"max_absolute_logit_difference":max,"rows":rows,"seconds":t.elapsed().as_secs_f64()});
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
