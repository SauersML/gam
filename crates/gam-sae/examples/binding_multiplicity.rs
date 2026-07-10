//! App C — binding / multiplicity via harmonic super-resolution.
//!
//! The *signature* capability a linear SAE cannot represent: one circle-feature
//! firing **m times** in a single context (two dates, two colors, two weekdays)
//! is a sum of `m` point masses on one harmonic-circle atom. This example
//! validates, end-to-end and against known ground truth, that the matrix-pencil
//! recovery ([`gam_sae::super_resolution::recover_spikes`]) plus the production
//! gated readout ([`gam_sae::sparse_dict::recover_measure_from_code`], the same
//! decision [`harmonic_measure_coordinates`] applies per firing) recover the
//! *number* of spikes — the multiplicity — from the superposed Fourier
//! coefficients the block stores.
//!
//! # Part A — planted-fixture spike-count vs ground truth
//!
//! Random `m`-spike measures (`m ∈ {1, 2, 3}`) are planted on an `H`-harmonic
//! circle at separations that bracket the Candès–Fernández-Granda limit `2/H`,
//! with additive coefficient noise and optional f32 quantisation (the real codes
//! are f32). Each planted code is passed through both the raw Prony order and the
//! gated production readout; the predicted count is scored against the true `m`
//! into a confusion matrix, and position/amplitude error is reported when the
//! count is exact.
//!
//! # Part B — real-code superposition (binding demo)
//!
//! When a `--weekday-codes <csv>` of *real* single-instance harmonic codes is
//! supplied (one row per single-weekday activation projected onto a fitted
//! weekday-circle `2H`-frame, produced by `weekday_binding.py`), the example
//! synthesises genuine two-instance codes by summing pairs of distinct-weekday
//! rows and recovers them: two spikes at the two weekdays' circle positions. This
//! is a controlled 2-spike fixture built from *real activations* — the same
//! superposition the model forms when a prompt names two weekdays — and the
//! baseline single-coordinate readout provably cannot represent it.
//!
//! # Part C — fully-real two-instance recovery
//!
//! When a `--two-instance-codes <csv>` of real *two-weekday-prompt* activations
//! (each projected onto the fitted `2H`-frame, with the two ground-truth circle
//! positions `t1,t2` in the first two columns) is supplied, the example recovers
//! each directly and scores the recovered count (should be 2) and positions
//! against the ground truth. This is the model actually superposing two instances
//! of the same circle feature in one context — the capability a linear SAE cannot
//! represent — read back out by super-resolution.
//!
//! Usage:
//!   binding_multiplicity <out_dir> [--weekday-codes <csv>] [--two-instance-codes <csv>]

use gam_sae::sparse_dict::recover_measure_from_code;
use gam_sae::super_resolution::{recover_spikes, separation_limit};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "binding_multiplicity".to_string());
    let Some(out_dir_raw) = args.next() else {
        return Err(format!(
            "usage: {program} <out_dir> [--weekday-codes <csv>]"
        ));
    };
    let out_dir = PathBuf::from(out_dir_raw);
    std::fs::create_dir_all(&out_dir)
        .map_err(|err| format!("create {}: {err}", out_dir.display()))?;

    let mut weekday_codes: Option<PathBuf> = None;
    let mut two_instance_codes: Option<PathBuf> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--weekday-codes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--weekday-codes needs a path".to_string())?;
                weekday_codes = Some(PathBuf::from(value));
            }
            "--two-instance-codes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--two-instance-codes needs a path".to_string())?;
                two_instance_codes = Some(PathBuf::from(value));
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }

    let fixture = part_a_fixture_sweep();
    let fixture_summary = summarize_fixture(&fixture);
    let diagnostic = part_a_diagnostic();
    write_json(
        &out_dir.join("fixture_spike_counts.json"),
        &Value::Array(fixture.clone()),
    )?;

    let mut report = json!({
        "app": "C_binding_multiplicity",
        "part_a_fixture": fixture_summary,
        "part_a_m2_diagnostic": diagnostic,
    });

    if let Some(csv) = weekday_codes {
        let binding = part_b_real_superposition(&csv)?;
        report["part_b_real_superposition"] = binding;
    } else {
        report["part_b_real_superposition"] =
            json!({"status": "skipped", "reason": "no --weekday-codes csv supplied"});
    }

    if let Some(csv) = two_instance_codes {
        let real = part_c_two_instance(&csv)?;
        report["part_c_two_instance"] = real;
    } else {
        report["part_c_two_instance"] =
            json!({"status": "skipped", "reason": "no --two-instance-codes csv supplied"});
    }

    write_json(&out_dir.join("binding_multiplicity_report.json"), &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

// --------------------------------------------------------------------------- //
// Part A: planted-fixture spike-count sweep.
// --------------------------------------------------------------------------- //

/// One planted-fixture trial outcome.
#[derive(Clone)]
struct TrialOutcome {
    h: usize,
    sigma: f64,
    quantize_f32: bool,
    sep_factor: f64,
    m_true: usize,
    /// Raw matrix-pencil model order (order selection alone).
    prony_order: usize,
    /// Gated production count (the per-firing readout's returned support size).
    gated_count: usize,
    /// Whether the gated path escalated to super-resolution (multi-modal trigger).
    used_super_resolution: bool,
    /// Single-spike BLASSO dual birth ratio (multiplicity trigger; > 1 = birth).
    dual_eta: f64,
    /// Max wrap-around position error when gated_count == m_true (else NaN).
    position_error: f64,
}

fn part_a_fixture_sweep() -> Vec<Value> {
    // f32 codes are the real dtype; the sweep runs both to show the numerical
    // shelf handling. Noise levels span clean to moderate. Planted spikes are
    // separated at `sep_factor × (2/H)` — `sep_factor = 1` sits exactly at the
    // Candès–Fernández-Granda guarantee, above it is the comfortably-resolvable
    // regime. The separation sweep is the honest characterisation: the raw
    // matrix-pencil order recovers the count even at the limit, while the gated
    // production readout (which filters spikes closer than 2/H, buying zero
    // false-binding) needs separation above the limit to keep both spikes.
    let harmonics = [8usize, 16];
    let noises = [0.0f64, 1e-3, 1e-2, 5e-2];
    let quant = [false, true];
    let sep_factors = [1.0f64, 1.5, 2.0, 3.0];
    let trials_per_cell = 200usize;

    let mut rows = Vec::new();
    for &h in &harmonics {
        let limit = separation_limit(h); // 2/H
        for &sigma in &noises {
            for &quantize in &quant {
                for &sep_factor in &sep_factors {
                    for m_true in 1..=3usize {
                        // Skip m the atom cannot host: Prony resolves at most ⌊H/2⌋.
                        if m_true > h / 2 {
                            continue;
                        }
                        // Feasibility: m spikes need pairwise sep ≤ 1/m on the
                        // unit circle. Cap the target so the sampler never stalls.
                        let target = (sep_factor * limit).min(0.95 / m_true as f64);
                        let mut outcomes = Vec::with_capacity(trials_per_cell);
                        for trial in 0..trials_per_cell {
                            let seed = mix_seed(h, sigma, quantize, m_true, trial)
                                ^ ((sep_factor * 1000.0) as u64).wrapping_mul(0x1000193);
                            outcomes.push(run_trial(
                                h, sigma, quantize, sep_factor, m_true, target, seed,
                            ));
                        }
                        rows.push(cell_report(&outcomes));
                    }
                }
            }
        }
    }
    rows
}

/// Focused m=2 diagnostic: for a handful of clean, well-separated H=8 two-spike
/// fixtures, dump the planted measure, the raw matrix-pencil spikes, and the
/// gated readout's spikes — so the gate's accept/reject decision is auditable.
fn part_a_diagnostic() -> Value {
    let h = 8usize;
    let sigma = 0.0;
    let min_sep = 3.0 * separation_limit(h); // comfortably resolvable
    let min_sep = min_sep.min(0.45);
    let mut cases = Vec::new();
    for trial in 0..6usize {
        let seed = mix_seed(h, sigma, false, 2, trial) ^ 0xABCDEF;
        let mut rng = StdRng::seed_from_u64(seed);
        let planted = sample_separated_spikes(&mut rng, 2, min_sep);
        let coeffs = coeffs_from_spikes(&planted, h);
        let raw = recover_spikes(&coeffs, sigma);
        let mut z = Vec::with_capacity(2 * h);
        for &(c, s) in &coeffs {
            z.push(c);
            z.push(s);
        }
        let (gated, eta, used_sr) = recover_measure_from_code(&z, sigma);
        let (raw_order, raw_spikes): (usize, Vec<(f64, f64)>) = match &raw {
            Ok(r) => (
                r.model_order,
                r.spikes.iter().map(|s| (s.t, s.amplitude)).collect(),
            ),
            Err(_) => (0, Vec::new()),
        };
        cases.push(json!({
            "planted": planted,
            "raw_model_order": raw_order,
            "raw_spikes_t_amp": raw_spikes,
            "gated_count": gated.len(),
            "gated_spikes_t_amp": gated.iter().map(|s| (s.coordinate, s.amplitude)).collect::<Vec<_>>(),
            "dual_eta": eta,
            "used_super_resolution": used_sr,
        }));
    }
    json!({ "h": h, "min_sep": min_sep, "cases": cases })
}

fn run_trial(
    h: usize,
    sigma: f64,
    quantize_f32: bool,
    sep_factor: f64,
    m_true: usize,
    min_sep: f64,
    seed: u64,
) -> TrialOutcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let planted = sample_separated_spikes(&mut rng, m_true, min_sep);
    let mut coeffs = coeffs_from_spikes(&planted, h);
    if sigma > 0.0 {
        add_noise(&mut coeffs, sigma, &mut rng);
    }
    if quantize_f32 {
        for coeff in coeffs.iter_mut() {
            coeff.0 = coeff.0 as f32 as f64;
            coeff.1 = coeff.1 as f32 as f64;
        }
    }

    let prony_order = recover_spikes(&coeffs, sigma)
        .map(|r| r.model_order)
        .unwrap_or(0);

    // Gated production readout works off the flattened within-block code
    // z = [c_1, s_1, c_2, s_2, ...].
    let mut z = Vec::with_capacity(2 * h);
    for &(c, s) in &coeffs {
        z.push(c);
        z.push(s);
    }
    let (spikes, dual_eta, used_sr) = recover_measure_from_code(&z, sigma);
    let gated_count = spikes.len();

    let position_error = if gated_count == m_true {
        let recovered: Vec<(f64, f64)> =
            spikes.iter().map(|s| (s.coordinate, s.amplitude)).collect();
        matched_position_error(&recovered, &planted)
    } else {
        f64::NAN
    };

    TrialOutcome {
        h,
        sigma,
        quantize_f32,
        sep_factor,
        m_true,
        prony_order,
        gated_count,
        used_super_resolution: used_sr,
        dual_eta,
        position_error,
    }
}

fn cell_report(outcomes: &[TrialOutcome]) -> Value {
    let n = outcomes.len() as f64;
    let first = &outcomes[0];
    let m_true = first.m_true;

    let gated_correct = outcomes.iter().filter(|o| o.gated_count == m_true).count();
    let prony_correct = outcomes.iter().filter(|o| o.prony_order == m_true).count();
    // Confusion over predicted gated counts 0..=4.
    let mut confusion = [0usize; 5];
    for o in outcomes {
        let idx = o.gated_count.min(4);
        confusion[idx] += 1;
    }
    // For multi-spike cells, the multi-modal trigger should fire; for m=1 it
    // should stay off (no spurious binding).
    let used_sr = outcomes.iter().filter(|o| o.used_super_resolution).count();

    let pos_errors: Vec<f64> = outcomes
        .iter()
        .filter_map(|o| {
            if o.position_error.is_finite() {
                Some(o.position_error)
            } else {
                None
            }
        })
        .collect();
    let max_pos_error = pos_errors.iter().cloned().fold(0.0f64, f64::max);
    let mean_eta = outcomes.iter().map(|o| o.dual_eta).sum::<f64>() / n;

    json!({
        "h": first.h,
        "sigma": first.sigma,
        "quantize_f32": first.quantize_f32,
        "sep_factor": first.sep_factor,
        "m_true": m_true,
        "trials": outcomes.len(),
        "gated_count_accuracy": gated_correct as f64 / n,
        "prony_order_accuracy": prony_correct as f64 / n,
        "gated_confusion_over_0_to_4plus": confusion,
        "used_super_resolution_fraction": used_sr as f64 / n,
        "mean_dual_eta": mean_eta,
        "max_position_error_when_count_exact": max_pos_error,
    })
}

fn summarize_fixture(cells: &[Value]) -> Value {
    // Two headline stories:
    //  (1) RAW matrix-pencil order recovers the count even at the 2/H limit —
    //      the representational capability (a linear SAE has no such readout).
    //  (2) GATED production readout: exact-count accuracy as a function of
    //      separation above the limit, at clean and moderate noise, with the
    //      m=1 false-binding rate (its precision guarantee).
    // Accuracy is aggregated over H and f32/f64 within each (sep_factor, m).

    // Raw prony order accuracy at sep_factor == 1 (the limit), by m.
    let mut raw_limit = [0.0f64; 4];
    let mut raw_limit_n = [0usize; 4];
    // Gated accuracy keyed by (sep_factor bucket, m) at clean and at sigma=1e-2.
    let sep_keys = [1.0f64, 1.5, 2.0, 3.0];
    let mut gated_clean = [[0.0f64; 4]; 4]; // [sep_idx][m]
    let mut gated_clean_n = [[0usize; 4]; 4];
    let mut gated_noisy = [[0.0f64; 4]; 4];
    let mut gated_noisy_n = [[0usize; 4]; 4];
    let mut false_binding_num = 0.0f64;
    let mut false_binding_den = 0usize;

    for cell in cells {
        let m = cell["m_true"].as_u64().unwrap_or(0) as usize;
        let sigma = cell["sigma"].as_f64().unwrap_or(-1.0);
        let sep = cell["sep_factor"].as_f64().unwrap_or(-1.0);
        let gated = cell["gated_count_accuracy"].as_f64().unwrap_or(0.0);
        let raw = cell["prony_order_accuracy"].as_f64().unwrap_or(0.0);
        if m >= 4 {
            continue;
        }
        let sep_idx = sep_keys.iter().position(|&s| (s - sep).abs() < 1e-9);
        if (sep - 1.0).abs() < 1e-9 {
            raw_limit[m] += raw;
            raw_limit_n[m] += 1;
        }
        if let Some(si) = sep_idx {
            if sigma == 0.0 {
                gated_clean[si][m] += gated;
                gated_clean_n[si][m] += 1;
            }
            if (sigma - 1e-2).abs() < 1e-12 {
                gated_noisy[si][m] += gated;
                gated_noisy_n[si][m] += 1;
            }
        }
        if m == 1 {
            false_binding_num += 1.0 - gated;
            false_binding_den += 1;
        }
    }
    let avg1 = |num: &[f64; 4], den: &[usize; 4], m: usize| -> Value {
        if den[m] == 0 {
            Value::Null
        } else {
            json!(num[m] / den[m] as f64)
        }
    };
    let by_sep = |grid: &[[f64; 4]; 4], gridn: &[[usize; 4]; 4], m: usize| -> Value {
        let mut obj = serde_json::Map::new();
        for (si, &s) in sep_keys.iter().enumerate() {
            let v = if gridn[si][m] == 0 {
                Value::Null
            } else {
                json!(grid[si][m] / gridn[si][m] as f64)
            };
            obj.insert(format!("{s:.1}xlimit"), v);
        }
        Value::Object(obj)
    };
    json!({
        "raw_matrix_pencil_order_accuracy_at_2overH_limit": {
            "1": avg1(&raw_limit, &raw_limit_n, 1),
            "2": avg1(&raw_limit, &raw_limit_n, 2),
            "3": avg1(&raw_limit, &raw_limit_n, 3),
        },
        "gated_exact_count_accuracy_clean_by_separation": {
            "m1": by_sep(&gated_clean, &gated_clean_n, 1),
            "m2": by_sep(&gated_clean, &gated_clean_n, 2),
            "m3": by_sep(&gated_clean, &gated_clean_n, 3),
        },
        "gated_exact_count_accuracy_sigma1e-2_by_separation": {
            "m1": by_sep(&gated_noisy, &gated_noisy_n, 1),
            "m2": by_sep(&gated_noisy, &gated_noisy_n, 2),
            "m3": by_sep(&gated_noisy, &gated_noisy_n, 3),
        },
        "m1_false_binding_rate_avg": if false_binding_den == 0 {
            Value::Null
        } else {
            json!(false_binding_num / false_binding_den as f64)
        },
        "n_cells": cells.len(),
    })
}

// --------------------------------------------------------------------------- //
// Part B: real-code two-instance superposition.
// --------------------------------------------------------------------------- //

fn part_b_real_superposition(csv: &Path) -> Result<Value, String> {
    // CSV: header `weekday,label,<2H code columns>`; one row per single-weekday
    // activation projected onto the fitted weekday-circle 2H-frame. We take one
    // representative (mean) code per weekday, then form every distinct pair sum
    // and recover — the ground-truth multiplicity of a pair is 2, at the two
    // weekdays' circle positions.
    let (labels, codes) = read_weekday_codes(csv)?;
    if codes.is_empty() {
        return Err("weekday-codes csv has no data rows".to_string());
    }
    let b = codes[0].len();
    if b < 4 || b % 2 != 0 {
        return Err(format!(
            "weekday code width {b} must be even and >= 4 (b = 2H, H >= 2)"
        ));
    }

    // Per-weekday mean code and its single-spike recovered position.
    let mut per_weekday: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();
    for &label in labels.iter() {
        if !seen.contains(&label) {
            seen.push(label);
            // mean over all rows with this label
            let mut acc = vec![0.0f64; b];
            let mut count = 0usize;
            for (r2, &l2) in labels.iter().enumerate() {
                if l2 == label {
                    for k in 0..b {
                        acc[k] += codes[r2][k];
                    }
                    count += 1;
                }
            }
            for v in acc.iter_mut() {
                *v /= count as f64;
            }
            per_weekday.push((label, acc));
        }
    }
    per_weekday.sort_by_key(|(label, _)| *label);

    // Estimate a global noise sigma from the residual scatter of the
    // single-spike fits across the mean codes (radial-style), so the gated
    // readout uses a realistic, data-derived noise level.
    let sigma = estimate_sigma(&per_weekday);

    // Singles: each mean weekday code should recover exactly one spike.
    let mut singles = Vec::new();
    let mut singles_correct = 0usize;
    for (label, code) in &per_weekday {
        let (spikes, eta, used_sr) = recover_measure_from_code(code, sigma);
        if spikes.len() == 1 {
            singles_correct += 1;
        }
        singles.push(json!({
            "weekday_label": label,
            "recovered_count": spikes.len(),
            "positions": spikes.iter().map(|s| s.coordinate).collect::<Vec<_>>(),
            "dual_eta": eta,
            "used_super_resolution": used_sr,
        }));
    }

    // Pairs: sum two distinct weekday mean codes → a genuine 2-instance code.
    let mut pairs = Vec::new();
    let mut pairs_count_correct = 0usize;
    let mut pairs_total = 0usize;
    let mut pair_pos_err_sum = 0.0f64;
    let mut pair_pos_err_n = 0usize;
    for i in 0..per_weekday.len() {
        for j in (i + 1)..per_weekday.len() {
            let (label_i, code_i) = &per_weekday[i];
            let (label_j, code_j) = &per_weekday[j];
            // The single-spike positions are the ground-truth spike locations.
            let (spikes_i, _, _) = recover_measure_from_code(code_i, sigma);
            let (spikes_j, _, _) = recover_measure_from_code(code_j, sigma);
            if spikes_i.len() != 1 || spikes_j.len() != 1 {
                continue; // only pair up clean singles
            }
            let ti = spikes_i[0].coordinate;
            let tj = spikes_j[0].coordinate;
            let mut summed = vec![0.0f64; code_i.len()];
            for k in 0..summed.len() {
                summed[k] = code_i[k] + code_j[k];
            }
            let (spikes, eta, used_sr) = recover_measure_from_code(&summed, sigma);
            pairs_total += 1;
            let count_ok = spikes.len() == 2;
            if count_ok {
                pairs_count_correct += 1;
                let recovered: Vec<(f64, f64)> =
                    spikes.iter().map(|s| (s.coordinate, s.amplitude)).collect();
                let truth = [(ti, 1.0), (tj, 1.0)];
                let err = matched_position_error(&recovered, &truth);
                pair_pos_err_sum += err;
                pair_pos_err_n += 1;
            }
            if pairs.len() < 30 {
                pairs.push(json!({
                    "weekday_i": label_i,
                    "weekday_j": label_j,
                    "truth_positions": [ti, tj],
                    "recovered_count": spikes.len(),
                    "recovered_positions": spikes.iter().map(|s| s.coordinate).collect::<Vec<_>>(),
                    "dual_eta": eta,
                    "used_super_resolution": used_sr,
                }));
            }
        }
    }

    Ok(json!({
        "status": "ok",
        "csv": csv.display().to_string(),
        "n_weekdays": per_weekday.len(),
        "code_width_2h": b,
        "sigma_hat": sigma,
        "singles": {
            "count_exactly_one_spike": singles_correct,
            "total": per_weekday.len(),
            "detail": singles,
        },
        "pairs": {
            "count_exactly_two_spikes": pairs_count_correct,
            "total": pairs_total,
            "two_spike_accuracy": if pairs_total == 0 { 0.0 } else { pairs_count_correct as f64 / pairs_total as f64 },
            "mean_position_error_when_two": if pair_pos_err_n == 0 { f64::NAN } else { pair_pos_err_sum / pair_pos_err_n as f64 },
            "examples": pairs,
        },
    }))
}

// --------------------------------------------------------------------------- //
// Part C: fully-real two-instance recovery from two-weekday-prompt codes.
// --------------------------------------------------------------------------- //

fn part_c_two_instance(csv: &Path) -> Result<Value, String> {
    // CSV: header `t1,t2,<2H code columns>`; one row per two-weekday prompt,
    // its last-token activation projected onto the fitted 2H-frame. t1,t2 are the
    // ground-truth circle positions of the two named weekdays.
    let text =
        std::fs::read_to_string(csv).map_err(|err| format!("read {}: {err}", csv.display()))?;
    let mut truths: Vec<(f64, f64)> = Vec::new();
    let mut codes: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 4 {
            return Err(format!(
                "{}:{} needs t1,t2,<code...>",
                csv.display(),
                lineno + 1
            ));
        }
        if lineno == 0 && fields[0].trim().parse::<f64>().is_err() {
            continue; // header
        }
        let t1 = fields[0]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("t1: {e}"))?;
        let t2 = fields[1]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("t2: {e}"))?;
        let mut code = Vec::with_capacity(fields.len() - 2);
        for f in &fields[2..] {
            code.push(f.trim().parse::<f64>().map_err(|e| format!("code: {e}"))?);
        }
        truths.push((t1, t2));
        codes.push(code);
    }
    if codes.is_empty() {
        return Err("two-instance csv has no data rows".to_string());
    }

    // Data-derived noise from the residual of a single-spike fit across all rows'
    // norms would conflate the two spikes; instead use the per-row inter-code
    // radial scatter as a conservative sigma proxy.
    let norms: Vec<f64> = codes
        .iter()
        .map(|z| z.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect();
    let mean_norm = norms.iter().sum::<f64>() / norms.len() as f64;
    let sigma = if norms.len() < 2 {
        0.0
    } else {
        (norms
            .iter()
            .map(|&r| (r - mean_norm) * (r - mean_norm))
            .sum::<f64>()
            / (norms.len() - 1) as f64)
            .sqrt()
            * 0.5
    };

    let mut count_two = 0usize;
    let mut count_one = 0usize;
    let mut count_hist = [0usize; 5];
    let mut raw_two = 0usize; // raw matrix-pencil order == 2
    let mut pos_err_sum = 0.0f64;
    let mut pos_err_n = 0usize;
    let mut used_sr = 0usize;
    // Resolvability by true separation: how the recovered count tracks the
    // Part-A separation law on the real circle. Bucketed by |t1 - t2| wrap dist.
    let mut sep_buckets: Vec<(f64, usize)> = Vec::new(); // (true_sep, gated_two_flag)
    let mut examples = Vec::new();
    for (row, (code, &(t1, t2))) in codes.iter().zip(truths.iter()).enumerate() {
        let (spikes, eta, sr) = recover_measure_from_code(code, sigma);
        // Raw matrix-pencil order (the representational capability, at the 2/H
        // limit) alongside the conservative gated count.
        let coeffs: Vec<(f64, f64)> = code.chunks_exact(2).map(|p| (p[0], p[1])).collect();
        let raw_order = recover_spikes(&coeffs, sigma)
            .map(|r| r.model_order)
            .unwrap_or(0);
        if raw_order == 2 {
            raw_two += 1;
        }
        let true_sep = circle_dist(t1, t2);
        sep_buckets.push((true_sep, usize::from(spikes.len() == 2)));
        let c = spikes.len();
        count_hist[c.min(4)] += 1;
        if sr {
            used_sr += 1;
        }
        if c == 2 {
            count_two += 1;
            let recovered: Vec<(f64, f64)> =
                spikes.iter().map(|s| (s.coordinate, s.amplitude)).collect();
            let truth = [(t1, 1.0), (t2, 1.0)];
            let err = matched_position_error(&recovered, &truth);
            if err.is_finite() {
                pos_err_sum += err;
                pos_err_n += 1;
            }
        } else if c == 1 {
            count_one += 1;
        }
        if row < 40 {
            examples.push(json!({
                "truth_positions": [t1, t2],
                "recovered_count": c,
                "recovered_positions": spikes.iter().map(|s| s.coordinate).collect::<Vec<_>>(),
                "dual_eta": eta,
                "used_super_resolution": sr,
            }));
        }
    }
    let total = codes.len();
    // Separation law on the real circle: gated 2-spike rate for pairs whose true
    // separation is ABOVE vs BELOW the super-resolution limit 2/H (= 4/b). This
    // is the real-data echo of the Part-A separation sweep.
    let h_count = codes[0].len() / 2;
    let limit = separation_limit(h_count);
    let mut above_n = 0usize;
    let mut above_two = 0usize;
    let mut below_n = 0usize;
    let mut below_two = 0usize;
    for &(sep, got_two) in &sep_buckets {
        if sep >= limit {
            above_n += 1;
            above_two += got_two;
        } else {
            below_n += 1;
            below_two += got_two;
        }
    }
    Ok(json!({
        "status": "ok",
        "csv": csv.display().to_string(),
        "n_two_instance_prompts": total,
        "sigma_hat": sigma,
        "n_harmonics": h_count,
        "super_resolution_limit_2overH": limit,
        "recovered_count_histogram_0_to_4plus": count_hist,
        "recovered_exactly_two_gated": count_two,
        "recovered_exactly_two_raw_matrix_pencil": raw_two,
        "recovered_exactly_one": count_one,
        "two_spike_accuracy_gated": count_two as f64 / total as f64,
        "two_spike_accuracy_raw_matrix_pencil": raw_two as f64 / total as f64,
        "separation_law": {
            "pairs_above_limit": above_n,
            "gated_two_rate_above_limit": if above_n == 0 { Value::Null } else { json!(above_two as f64 / above_n as f64) },
            "pairs_below_limit": below_n,
            "gated_two_rate_below_limit": if below_n == 0 { Value::Null } else { json!(below_two as f64 / below_n as f64) },
        },
        "used_super_resolution_fraction": used_sr as f64 / total as f64,
        "mean_position_error_when_two": if pos_err_n == 0 { f64::NAN } else { pos_err_sum / pos_err_n as f64 },
        "examples": examples,
    }))
}

fn estimate_sigma(per_weekday: &[(usize, Vec<f64>)]) -> f64 {
    // Radial scatter of the mean-code norms about their mean (matches
    // coordinate.rs radius_and_sigma convention), a conservative noise proxy.
    let norms: Vec<f64> = per_weekday
        .iter()
        .map(|(_, z)| z.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect();
    let n = norms.len();
    if n < 2 {
        return 0.0;
    }
    let mean = norms.iter().sum::<f64>() / n as f64;
    let ss: f64 = norms.iter().map(|&r| (r - mean) * (r - mean)).sum();
    (ss / (n - 1) as f64).sqrt()
}

// --------------------------------------------------------------------------- //
// Shared helpers.
// --------------------------------------------------------------------------- //

/// Build exact Fourier coefficients `y_h = Σ_j a_j e^{2πi h t_j}`, `h = 1..H`.
fn coeffs_from_spikes(spikes: &[(f64, f64)], n_harmonics: usize) -> Vec<(f64, f64)> {
    (1..=n_harmonics)
        .map(|h| {
            let mut c = 0.0;
            let mut s = 0.0;
            for &(t, a) in spikes {
                let phase = std::f64::consts::TAU * (h as f64) * t;
                c += a * phase.cos();
                s += a * phase.sin();
            }
            (c, s)
        })
        .collect()
}

fn add_noise(coeffs: &mut [(f64, f64)], sigma: f64, rng: &mut StdRng) {
    for coeff in coeffs.iter_mut() {
        coeff.0 += sigma * gaussian(rng);
        coeff.1 += sigma * gaussian(rng);
    }
}

fn gaussian(rng: &mut StdRng) -> f64 {
    use rand::RngExt as _;
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Sample `m` spikes with pairwise wrap-around separation ≥ `min_sep`, unit-ish
/// amplitudes in `[0.6, 1.4]`. Rejection sampling with a bounded retry budget.
fn sample_separated_spikes(rng: &mut StdRng, m: usize, min_sep: f64) -> Vec<(f64, f64)> {
    use rand::RngExt as _;
    let mut positions: Vec<f64> = Vec::with_capacity(m);
    let mut guard = 0;
    while positions.len() < m {
        let t = rng.random::<f64>();
        if positions.iter().all(|&p| circle_dist(t, p) >= min_sep) {
            positions.push(t);
        }
        guard += 1;
        if guard > 100_000 {
            // Fall back to an even spacing if rejection stalls (min_sep too tight).
            positions.clear();
            for j in 0..m {
                positions.push(j as f64 / m as f64);
            }
            break;
        }
    }
    positions
        .into_iter()
        .map(|t| {
            let a = 0.6 + 0.8 * rng.random::<f64>();
            (t, a)
        })
        .collect()
}

fn circle_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

/// Greedy nearest-position match of recovered spikes to planted, returning the
/// max wrap-around position error. Requires equal counts.
fn matched_position_error(recovered: &[(f64, f64)], planted: &[(f64, f64)]) -> f64 {
    if recovered.len() != planted.len() {
        return f64::NAN;
    }
    let mut rec: Vec<f64> = recovered.iter().map(|(t, _)| *t).collect();
    rec.sort_by(|a, b| a.total_cmp(b));
    let mut plt: Vec<f64> = planted.iter().map(|(t, _)| *t).collect();
    plt.sort_by(|a, b| a.total_cmp(b));
    // Both sorted by position; the wrap-around optimal match of equal-count
    // circular point sets is a cyclic rotation — try all rotations.
    let n = rec.len();
    let mut best = f64::INFINITY;
    for shift in 0..n {
        let mut worst = 0.0f64;
        for k in 0..n {
            worst = worst.max(circle_dist(rec[k], plt[(k + shift) % n]));
        }
        best = best.min(worst);
    }
    best
}

fn mix_seed(h: usize, sigma: f64, quant: bool, m: usize, trial: usize) -> u64 {
    let sigma_bits = (sigma * 1e9) as u64;
    let mut x = 0x9E3779B97F4A7C15u64;
    for v in [h as u64, sigma_bits, quant as u64, m as u64, trial as u64] {
        x ^= v
            .wrapping_add(0x9E3779B97F4A7C15)
            .wrapping_add(x << 6)
            .wrapping_add(x >> 2);
    }
    x
}

fn read_weekday_codes(path: &Path) -> Result<(Vec<usize>, Vec<Vec<f64>>), String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut labels = Vec::new();
    let mut codes = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip a header row (non-numeric second field).
        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 3 {
            return Err(format!(
                "{}:{} needs weekday,label,<code...>",
                path.display(),
                lineno + 1
            ));
        }
        if lineno == 0 && fields[1].trim().parse::<i64>().is_err() {
            continue; // header
        }
        let label = fields[1]
            .trim()
            .parse::<usize>()
            .map_err(|err| format!("{}:{} parse label: {err}", path.display(), lineno + 1))?;
        let mut code = Vec::with_capacity(fields.len() - 2);
        for f in &fields[2..] {
            code.push(
                f.trim().parse::<f64>().map_err(|err| {
                    format!("{}:{} parse code: {err}", path.display(), lineno + 1)
                })?,
            );
        }
        labels.push(label);
        codes.push(code);
    }
    Ok((labels, codes))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut file =
        std::fs::File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    file.write_all(
        serde_json::to_string_pretty(value)
            .map_err(|e| e.to_string())?
            .as_bytes(),
    )
    .map_err(|err| format!("write {}: {err}", path.display()))?;
    Ok(())
}
