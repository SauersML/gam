//! Fit real harvested Qwen attention heads with the attention-kernel module.
//!
//! Usage:
//!   attn_real_fit <observations.json> <out-dir> [max_harmonic]

use std::fs;
use std::path::{Path, PathBuf};

use gam_sae::attention_kernel::{fit_attention_kernel, fit_ov_coordinate_map};
use ndarray::Array2;
use serde_json::{Value, json};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        return Err(format!(
            "usage: {} <observations.json> <out-dir> [max_harmonic]",
            args.first()
                .cloned()
                .unwrap_or_else(|| "attn_real_fit".to_string())
        ));
    }
    let observations_path = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let max_harmonic = if args.len() == 4 {
        args[3]
            .parse::<usize>()
            .map_err(|err| format!("max_harmonic must be an integer: {err}"))?
    } else {
        4
    };

    let raw = fs::read_to_string(&observations_path)
        .map_err(|err| format!("failed to read {}: {err}", observations_path.display()))?;
    let payload: Value = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse {}: {err}", observations_path.display()))?;
    let observations = payload
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "observations.json must contain an observations array".to_string())?;
    if observations.is_empty() {
        return Err("observations array is empty".to_string());
    }

    fs::create_dir_all(&out_dir)
        .map_err(|err| format!("failed to create {}: {err}", out_dir.display()))?;
    let mut rows = Vec::with_capacity(observations.len());
    for observation in observations {
        rows.push(fit_one_head(observation, max_harmonic)?);
    }
    rows.sort_by(|left, right| {
        right["legibility_score"]
            .as_f64()
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&left["legibility_score"].as_f64().unwrap_or(f64::NEG_INFINITY))
    });
    let best = rows
        .first()
        .cloned()
        .ok_or_else(|| "no fitted heads were produced".to_string())?;
    let numbers = json!({
        "experiment": payload.get("experiment").cloned().unwrap_or_else(|| json!("attn_real")),
        "model": payload.get("model").cloned().unwrap_or(Value::Null),
        "layer": payload.get("layer").cloned().unwrap_or(Value::Null),
        "period": payload.get("period").cloned().unwrap_or(Value::Null),
        "seq_len": payload.get("seq_len").cloned().unwrap_or(Value::Null),
        "phase_probe_r2": payload.get("phase_probe_r2").cloned().unwrap_or(Value::Null),
        "max_harmonic": max_harmonic,
        "heads": rows,
        "most_legible_head": best,
    });
    write_json(&out_dir.join("numbers.json"), &numbers)?;
    write_report(&out_dir.join("results.md"), &numbers)?;
    Ok(())
}

fn fit_one_head(observation: &Value, max_harmonic: usize) -> Result<Value, String> {
    let layer = required_i64(observation, "layer")?;
    let head = required_i64(observation, "head")?;
    let period = required_i64(observation, "period")?;
    let query_t = required_vec_f64(observation, "query_t")?;
    let key_t = required_vec_f64(observation, "key_t")?;
    let scores = required_matrix(observation, "scores")?;
    let ov_key_t = required_vec_f64(observation, "ov_key_t")?;
    let ov_delta_t = required_vec_f64(observation, "ov_delta_t")?;
    let qk = fit_attention_kernel(&query_t, &key_t, scores.view(), max_harmonic)
        .map_err(|err| format!("layer {layer} head {head}: QK fit failed: {err}"))?;
    let ov = fit_ov_coordinate_map(&ov_key_t, &ov_delta_t, max_harmonic)
        .map_err(|err| format!("layer {layer} head {head}: OV map fit failed: {err}"))?;
    let report = qk.report();
    let dominant = report.dominant_stationary_harmonic;
    let dominant_harmonic = dominant.as_ref().map(|coefficient| coefficient.harmonic);
    let dominant_amplitude = dominant
        .as_ref()
        .map(|coefficient| coefficient.amplitude)
        .unwrap_or(0.0);
    let dominant_phase_turns = dominant
        .as_ref()
        .map(|coefficient| coefficient.sin.atan2(coefficient.cos) / (2.0 * std::f64::consts::PI));
    let ov_shift = ov.intercept;
    let ov_dominant = ov.dominant_harmonic().cloned();
    let formula = formula_label(
        dominant_harmonic,
        dominant_phase_turns,
        ov_shift,
        report.is_stationary,
        period,
    );
    let stationarity_credit = if report.is_stationary { 0.05 } else { 0.0 };
    let legibility_score = report.stationary_r2.max(0.0)
        + stationarity_credit
        + dominant_amplitude.abs().ln_1p() * 0.01
        + ov.r2.max(0.0) * 0.05;
    Ok(json!({
        "layer": layer,
        "head": head,
        "period": period,
        "stationary_r2": report.stationary_r2,
        "separable_r2": report.separable_r2,
        "stationary_r2_gap": report.stationary_r2_gap,
        "is_stationary": report.is_stationary,
        "dominant_stationary_harmonic": dominant_harmonic,
        "dominant_stationary_amplitude": dominant_amplitude,
        "dominant_stationary_phase_turns": dominant_phase_turns,
        "stationary_harmonic_content": report.stationary_harmonic_content.iter().map(|content| json!({
            "harmonic": content.harmonic,
            "cos": content.cos,
            "sin": content.sin,
            "amplitude": content.amplitude,
            "amplitude_fraction": content.amplitude_fraction,
        })).collect::<Vec<_>>(),
        "ov_shift_turns": ov_shift,
        "ov_shift_bins": ov_shift * period as f64,
        "ov_r2": ov.r2,
        "ov_dominant_harmonic": ov_dominant.as_ref().map(|coefficient| coefficient.harmonic),
        "ov_dominant_amplitude": ov_dominant.as_ref().map(|coefficient| coefficient.amplitude).unwrap_or(0.0),
        "phase_probe_r2": optional_f64(observation, "phase_probe_r2"),
        "qk_observations": optional_i64(observation, "qk_observations"),
        "formula": formula,
        "legibility_score": legibility_score,
    }))
}

fn formula_label(
    harmonic: Option<usize>,
    phase_turns: Option<f64>,
    ov_shift: f64,
    stationary: bool,
    period: i64,
) -> String {
    let harmonic_text = match harmonic {
        Some(1) => "same-phase",
        Some(h) => return format!("harmonic-{h} positional kernel"),
        None => return "no dominant harmonic".to_string(),
    };
    let phase_bins = phase_turns.unwrap_or(0.0) * period as f64;
    let qk_shift = phase_bins.round();
    let qk_text = if qk_shift == 0.0 {
        harmonic_text.to_string()
    } else if qk_shift > 0.0 {
        format!("{harmonic_text} +{qk_shift:.0}-bin")
    } else {
        format!("{harmonic_text} {qk_shift:.0}-bin")
    };
    let ov_bins = ov_shift * period as f64;
    let stationarity_text = if stationary { "stationary" } else { "nonstationary" };
    format!("{stationarity_text} {qk_text}; OV shift {ov_bins:.2} bins")
}

fn required_i64(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing integer field {key}"))
}

fn optional_i64(value: &Value, key: &str) -> Option<i64> {
    value.get(key).and_then(Value::as_i64)
}

fn optional_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn required_vec_f64(value: &Value, key: &str) -> Result<Vec<f64>, String> {
    let raw = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array field {key}"))?;
    raw.iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_f64()
                .ok_or_else(|| format!("{key}[{index}] must be numeric"))
        })
        .collect()
}

fn required_matrix(value: &Value, key: &str) -> Result<Array2<f64>, String> {
    let rows = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing matrix field {key}"))?;
    if rows.is_empty() {
        return Err(format!("{key} matrix is empty"));
    }
    let width = rows[0]
        .as_array()
        .ok_or_else(|| format!("{key}[0] must be an array"))?
        .len();
    if width == 0 {
        return Err(format!("{key} matrix has empty rows"));
    }
    let mut data = Vec::with_capacity(rows.len() * width);
    for (row_index, row) in rows.iter().enumerate() {
        let row_values = row
            .as_array()
            .ok_or_else(|| format!("{key}[{row_index}] must be an array"))?;
        if row_values.len() != width {
            return Err(format!(
                "{key}[{row_index}] width {} differs from first row width {width}",
                row_values.len()
            ));
        }
        for (col_index, item) in row_values.iter().enumerate() {
            data.push(
                item.as_f64()
                    .ok_or_else(|| format!("{key}[{row_index}][{col_index}] must be numeric"))?,
            );
        }
    }
    Array2::from_shape_vec((rows.len(), width), data)
        .map_err(|err| format!("failed to build {key} matrix: {err}"))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|err| format!("failed to serialize {}: {err}", path.display()))?;
    fs::write(path, format!("{text}\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn write_report(path: &Path, numbers: &Value) -> Result<(), String> {
    let heads = numbers
        .get("heads")
        .and_then(Value::as_array)
        .ok_or_else(|| "numbers must contain heads array".to_string())?;
    let best = numbers
        .get("most_legible_head")
        .ok_or_else(|| "numbers must contain most_legible_head".to_string())?;
    let mut lines = vec![
        "# Real Qwen Attention Kernel Fits".to_string(),
        String::new(),
        "This experiment harvested real Qwen3-8B attention-layer QK/OV observations on repeated-token positional prompts, then fit the landed `gam_sae::attention_kernel` harmonic models.".to_string(),
        String::new(),
        "## Summary".to_string(),
        String::new(),
        format!(
            "- model: `{}`",
            numbers
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ),
        format!(
            "- layer: {}",
            numbers
                .get("layer")
                .map(Value::to_string)
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "- period bins: {}",
            numbers
                .get("period")
                .map(Value::to_string)
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "- max harmonic: {}",
            numbers
                .get("max_harmonic")
                .map(Value::to_string)
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "- phase-probe R2: {:.6}",
            numbers
                .get("phase_probe_r2")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN)
        ),
        String::new(),
        "## Most Legible Head".to_string(),
        String::new(),
        format!(
            "- layer/head: L{} H{}",
            best.get("layer").and_then(Value::as_i64).unwrap_or(-1),
            best.get("head").and_then(Value::as_i64).unwrap_or(-1)
        ),
        format!(
            "- formula read: {}",
            best.get("formula").and_then(Value::as_str).unwrap_or("unknown")
        ),
        format!(
            "- dominant QK harmonic: {}",
            best.get("dominant_stationary_harmonic")
                .map(Value::to_string)
                .unwrap_or_else(|| "null".to_string())
        ),
        format!(
            "- stationary: {}",
            best.get("is_stationary")
                .and_then(Value::as_bool)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "- QK stationary R2: {:.6}",
            best.get("stationary_r2")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN)
        ),
        format!(
            "- QK separable R2: {:.6}",
            best.get("separable_r2")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN)
        ),
        format!(
            "- OV shift: {:.6} turns ({:.3} bins), OV R2 {:.6}",
            best.get("ov_shift_turns")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
            best.get("ov_shift_bins")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
            best.get("ov_r2").and_then(Value::as_f64).unwrap_or(f64::NAN)
        ),
        String::new(),
        "## Head Table".to_string(),
        String::new(),
        "| head | stationary | dominant harmonic | stationary R2 | separable R2 | gap | OV shift bins | OV R2 | formula |".to_string(),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |".to_string(),
    ];
    for head in heads {
        lines.push(format!(
            "| H{} | {} | {} | {:.6} | {:.6} | {:.6} | {:.3} | {:.6} | {} |",
            head.get("head").and_then(Value::as_i64).unwrap_or(-1),
            head.get("is_stationary")
                .and_then(Value::as_bool)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            head.get("dominant_stationary_harmonic")
                .map(Value::to_string)
                .unwrap_or_else(|| "null".to_string()),
            head.get("stationary_r2")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
            head.get("separable_r2")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
            head.get("stationary_r2_gap")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
            head.get("ov_shift_bins")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
            head.get("ov_r2").and_then(Value::as_f64).unwrap_or(f64::NAN),
            head.get("formula").and_then(Value::as_str).unwrap_or("unknown")
        ));
    }
    lines.push(String::new());
    lines.push("## Method".to_string());
    lines.push(String::new());
    lines.push("QK scores are pre-softmax `q_h(t_q) k_h(t_k)^T / sqrt(d_head)` values averaged into a 16x16 positional chart. Stationarity is the landed module's comparison between a delta-only harmonic kernel and the separable low-harmonic surface. The OV shift maps each head's `O_h V_h(t)` contribution through a layer-input phase probe and fits `delta_t` with the same harmonic coordinate-map API.".to_string());
    lines.push(String::new());
    fs::write(path, lines.join("\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}
