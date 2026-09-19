//! mpd-modadd's Schur cross-check on the modular-addition transformer's shift operator (#2951 S2).
//!
//! `mpd_modadd_schur_2951 --run RUN --settings SETTINGS --out REPORT`
//!
//! `RUN` is what `bench/mpd_modadd_2951.py execute` writes: `export.json`, and for each exported
//! checkpoint `name` the `(p+1) x d` embedding table `W_E.<name>.npy` in its trained `<f4`, widened
//! here to binary64, and the `d x d` least-squares operator `T1.<name>.npy` (`<f8`) with
//! `E T1ᵀ = E_shift` over the `p` cycled rows.
//!
//! Two routes to the planes of one operator:
//! * `cyclic_action` reads the closed-form planes `[u_kc, u_ks]` of `W_E` under the declared cycle
//!   `a -> a + 1 mod p`, with row `p` (the `=` token) fixed;
//! * `schur` recovers certified invariant blocks from the operator alone.
//!
//! For each checkpoint the example certifies every closed-form plane as an invariant subspace of
//! `T1`, and of `Vᵀ T1 V` as computed, with `V` the right singular vectors of the cycled rows. The
//! second operator leaves out the `d - p` directions the minimum-norm solve maps to zero. A
//! certified rotation-scaling is compared with the cycle's angle `ω_k = 2πk/p` by the distance of
//! `cos ω_k` from its certified cosine interval, and of 1 from its modulus interval. Then
//! `recover_invariant_blocks` runs on both operators, and each recovered rotation-scaling is read as
//! the frequency `α p / 2π`.
//!
//! Controls:
//! * positive: the `p x p` cyclic permutation with the identity table, whose planes are invariant
//!   with angle `ω_k` and modulus 1. The example refuses unless every one certifies as a
//!   rotation-scaling;
//! * negative: the mixed plane `[u_jc + u_kc, u_js + u_ks]` of the two declared frequencies, which
//!   the operator turns by two different angles, so it is not invariant. The example refuses when
//!   it certifies on any operator.
//!
//! The restricted operator `Vᵀ T1 V` carries none of `T1`'s `d - p` zero eigenvalues, which the full
//! recovery must certify as one repeated block. So every checkpoint's plane certificates and
//! restricted recovery run first, and the full-`T1` recoveries run last, one checkpoint at a time.
//! Each phase prints a `[phase]` line when it starts and its measured seconds when it ends, so a
//! stalled phase is named by the last line written.
//!
//! The report is rewritten after each phase, and the refusals are returned after the last write.
//!
//! This is a route receipt, not a mechanism test. `T1` is the cyclic shift `P` conjugated through
//! the embedding, so for any table of full row rank with `p <= d` its spectrum is `P`'s: `1`, the
//! pairs `e^{±2πik/p}`, and the `d - p` null block. Every such checkpoint (trained, random init or
//! random labels) therefore certifies every plane at its cycle frequency. The random-label and
//! random-init checkpoints are recorded for route agreement only: they cannot fail here by
//! construction, so they are not controls of this receipt, whose negative control is the mixed
//! plane. The receipt measures whether the two routes agree and where certification stops as the
//! table's conditioning grows.

use gam_linalg::faer_ndarray::FaerSvd;
use gam_linalg::roundoff::factor_singular_band;
use gam_sae::parameter_decomposition::cyclic_action::{CyclicPlanes, RowCycle, cyclic_planes};
use gam_sae::parameter_decomposition::schur::{
    InvariantBlockKind, InvariantSubspaceCertificate, SubspaceVerdict, certify_invariant_subspace,
    recover_invariant_blocks,
};
use ndarray::{Array2, ArrayView2, s};
use serde::Deserialize;
use serde_json::{Value, json};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "support/npy_header.rs"]
mod npy_header;
use npy_header::{NpyFloat, parse_npy_float_header, parse_npy_header};

const USAGE: &str = "usage: mpd_modadd_schur_2951 --run RUN --settings SETTINGS --out REPORT";

/// The fields of the settings this example reads; the driver reads `runs`.
#[derive(Deserialize)]
struct Settings {
    stage: String,
    negative_control_frequencies: [usize; 2],
}

#[derive(Deserialize)]
struct Export {
    stage: String,
    exports: Vec<Checkpoint>,
}

/// One exported checkpoint, with the driver's float64 solve diagnostics.
#[derive(Deserialize)]
struct Checkpoint {
    name: String,
    labels: String,
    step: u64,
    p: usize,
    d_model: usize,
    sigma_max: f64,
    sigma_min: f64,
    relative_residual: f64,
}

fn flag(args: &[String], name: &str) -> Result<PathBuf, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {name}; {USAGE}"))
}

/// A two-axis `<f4` array in its trained dtype, widened to binary64 (exact for every `f32`).
fn read_f4_widened(path: &Path) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let (rows, cols, width, is_f4, data_off) = parse_npy_header(&bytes, path)?;
    if !is_f4 {
        return Err(format!("{}: expected the trained <f4 values", path.display()));
    }
    let end = rows
        .checked_mul(cols)
        .and_then(|count| count.checked_mul(width))
        .and_then(|size| size.checked_add(data_off))
        .ok_or_else(|| format!("{}: size overflow", path.display()))?;
    if end != bytes.len() {
        return Err(format!(
            "{}: {} bytes, expected {end}",
            path.display(),
            bytes.len()
        ));
    }
    let values = bytes[data_off..end]
        .chunks_exact(4)
        .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect();
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// A two-axis `<f8` array.
fn read_f8(path: &Path) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let header = parse_npy_float_header(&bytes, path)?;
    if header.float != NpyFloat::F8 {
        return Err(format!("{}: expected <f8", path.display()));
    }
    let [rows, cols] = header.shape[..] else {
        return Err(format!(
            "{}: expected two axes, got {:?}",
            path.display(),
            header.shape
        ));
    };
    let end = rows
        .checked_mul(cols)
        .and_then(|count| count.checked_mul(header.float.bytes()))
        .and_then(|size| size.checked_add(header.data_off))
        .ok_or_else(|| format!("{}: size overflow", path.display()))?;
    if end != bytes.len() {
        return Err(format!(
            "{}: {} bytes, expected {end}",
            path.display(),
            bytes.len()
        ));
    }
    let values = bytes[header.data_off..end]
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn write_report(out: &Path, report: &Value) -> Result<(), String> {
    let partial = out.with_extension("json.partial");
    let text = serde_json::to_string_pretty(report).map_err(|error| format!("report: {error}"))?;
    std::fs::write(&partial, text)
        .map_err(|error| format!("write {}: {error}", partial.display()))?;
    std::fs::rename(&partial, out).map_err(|error| format!("rename to {}: {error}", out.display()))
}

fn cycle_angle(frequency: usize, length: usize) -> f64 {
    2.0 * PI * frequency as f64 / length as f64
}

fn inside(value: f64, interval: (f64, f64)) -> bool {
    interval.0 <= value && value <= interval.1
}

/// The distance of `value` from `interval`, zero inside it.
fn outside(value: f64, interval: (f64, f64)) -> f64 {
    (interval.0 - value).max(value - interval.1).max(0.0)
}

/// `omega` is the cycle angle a certified rotation-scaling is compared with, when there is one.
fn kind_report(kind: &InvariantBlockKind, omega: Option<f64>) -> Value {
    match kind {
        InvariantBlockKind::RotationScaling {
            modulus_interval,
            cosine_interval,
            angle,
            ..
        } => json!({
            "kind": "rotation_scaling",
            "modulus_interval": [modulus_interval.0, modulus_interval.1],
            "cosine_interval": [cosine_interval.0, cosine_interval.1],
            "angle": angle,
            "cycle_cosine_outside_interval": omega.map(|value| outside(value.cos(), *cosine_interval)),
            "unit_modulus_outside_interval": outside(1.0, *modulus_interval),
        }),
        InvariantBlockKind::Real {
            eigenvalue_interval,
        } => json!({
            "kind": "real",
            "eigenvalue_interval": [eigenvalue_interval.0, eigenvalue_interval.1],
        }),
        InvariantBlockKind::RealPair { eigenvalues } => json!({
            "kind": "real_pair",
            "eigenvalues": [eigenvalues.0, eigenvalues.1],
        }),
        InvariantBlockKind::Unresolved => json!({ "kind": "unresolved" }),
        InvariantBlockKind::Repeated { dimension } => {
            json!({ "kind": "repeated", "dimension": dimension })
        }
    }
}

fn certificate_report(
    certificate: &InvariantSubspaceCertificate,
    kind: Option<&InvariantBlockKind>,
    omega: Option<f64>,
) -> Value {
    let verdict = match certificate.verdict {
        SubspaceVerdict::Certified {
            projector_bar,
            restriction_error,
        } => json!({
            "certified": true,
            "projector_bar": projector_bar,
            "restriction_error": restriction_error,
        }),
        SubspaceVerdict::NotSeparated {
            required_separation,
        } => json!({ "certified": false, "required_separation": required_separation }),
    };
    json!({
        "dimension": certificate.basis.ncols(),
        "residual": certificate.residual,
        "coupling": certificate.coupling,
        "separation": certificate.separation,
        "frame_defect": certificate.frame_defect,
        "similarity_error": certificate.similarity_error,
        "verdict": verdict,
        "kind": kind.map(|value| kind_report(value, omega)),
    })
}

/// Certify every frequency's plane, columns `2(k - 1)` and `2k - 1` of `planes`, against
/// `operator`. Returns one report per frequency and the counts certified, certified as a
/// rotation-scaling, and certified as a rotation-scaling whose cosine interval holds `cos ω_k`.
fn certify_planes(
    operator: ArrayView2<'_, f64>,
    planes: ArrayView2<'_, f64>,
    power: &[f64],
    length: usize,
) -> (Vec<Value>, [usize; 3]) {
    let total: f64 = power.iter().sum();
    let mut reports = Vec::with_capacity(power.len());
    let mut counts = [0; 3];
    for (plane, &share) in power.iter().enumerate() {
        let frequency = plane + 1;
        let omega = cycle_angle(frequency, length);
        let basis = planes.slice(s![.., 2 * plane..2 * plane + 2]);
        let certificate = match certify_invariant_subspace(operator, basis) {
            Ok(certificate) => {
                let kind = certificate.kind();
                if matches!(certificate.verdict, SubspaceVerdict::Certified { .. }) {
                    counts[0] += 1;
                }
                if let Some(InvariantBlockKind::RotationScaling {
                    cosine_interval, ..
                }) = &kind
                {
                    counts[1] += 1;
                    if inside(omega.cos(), *cosine_interval) {
                        counts[2] += 1;
                    }
                }
                certificate_report(&certificate, kind.as_ref(), Some(omega))
            }
            Err(error) => json!({ "error": error.to_string() }),
        };
        reports.push(json!({
            "frequency": frequency,
            "power_share": share / total,
            "certificate": certificate,
        }));
    }
    (reports, counts)
}

/// `[u_jc + u_kc, u_js + u_ks]` for the declared pair `(j, k)`, refused by `cyclic_action` unless
/// `1 <= j < k <= m`.
fn mixed_plane(planes: &CyclicPlanes, pair: [usize; 2]) -> Result<Array2<f64>, String> {
    let basis = planes.basis(&pair).map_err(|error| error.to_string())?;
    let mut mixed = Array2::<f64>::zeros((basis.nrows(), 2));
    for column in 0..2 {
        let mut target = mixed.column_mut(column);
        target.assign(&basis.column(column));
        target += &basis.column(column + 2);
    }
    Ok(mixed)
}

/// The negative control on one operator: whether the mixed plane certified, and its report.
fn mixed_control(operator: ArrayView2<'_, f64>, basis: ArrayView2<'_, f64>) -> (bool, Value) {
    match certify_invariant_subspace(operator, basis) {
        Ok(certificate) => {
            let certified = matches!(certificate.verdict, SubspaceVerdict::Certified { .. });
            let kind = certificate.kind();
            (certified, certificate_report(&certificate, kind.as_ref(), None))
        }
        Err(error) => (false, json!({ "error": error.to_string() })),
    }
}

/// Recover the certified invariant blocks of `operator`, with a one-line summary: the block count,
/// the rotation-scaling planes, the distinct nearest frequencies among them, the largest offset of
/// `α p / 2π` from its nearest integer, and the dimensions of every other block.
fn recovery(operator: ArrayView2<'_, f64>, length: usize) -> (Value, String) {
    let started = Instant::now();
    let recovered = recover_invariant_blocks(operator);
    let seconds = started.elapsed().as_secs_f64();
    let recovery = match recovered {
        Ok(recovery) => recovery,
        Err(error) => {
            let message = error.to_string();
            let summary = format!("refused ({message}) seconds={seconds:.1}");
            return (json!({ "seconds": seconds, "error": message }), summary);
        }
    };
    let mut blocks = Vec::with_capacity(recovery.blocks.len());
    let mut frequencies = Vec::new();
    let mut offset = 0.0_f64;
    let mut other_dimensions = Vec::new();
    for block in &recovery.blocks {
        let estimate = match &block.kind {
            InvariantBlockKind::RotationScaling { angle, .. } => {
                Some(angle * length as f64 / (2.0 * PI))
            }
            _ => None,
        };
        let omega = match estimate {
            Some(value) => {
                frequencies.push(value.round() as usize);
                offset = offset.max((value - value.round()).abs());
                Some(cycle_angle(value.round() as usize, length))
            }
            None => {
                other_dimensions.push(block.certificate.basis.ncols());
                None
            }
        };
        blocks.push(json!({
            "eigenvalue_estimates": block.eigenvalue_estimates.len(),
            "frequency_estimate": estimate,
            "certificate": certificate_report(&block.certificate, Some(&block.kind), omega),
        }));
    }
    let planes = frequencies.len();
    frequencies.sort_unstable();
    frequencies.dedup();
    let ambiguities = format!("{:?}", recovery.ambiguities());
    let summary = format!(
        "blocks={} rotation_planes={planes} distinct_frequencies={} max_frequency_offset={offset:.3e} \
         other_block_dimensions={other_dimensions:?} ambiguities={ambiguities} seconds={seconds:.1}",
        recovery.blocks.len(),
        frequencies.len()
    );
    let report = json!({
        "seconds": seconds,
        "blocks": blocks,
        "rotation_planes": planes,
        "distinct_frequencies": frequencies,
        "max_frequency_offset": offset,
        "other_block_dimensions": other_dimensions,
        "ambiguities": ambiguities,
    });
    (report, summary)
}

fn counts_report(counts: [usize; 3]) -> Value {
    json!({ "certified": counts[0], "rotation_scaling": counts[1], "cosine_inside": counts[2] })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(USAGE.to_string());
    }
    let run = flag(&args, "--run")?;
    let settings_path = flag(&args, "--settings")?;
    let out = flag(&args, "--out")?;
    let text = std::fs::read_to_string(&settings_path)
        .map_err(|error| format!("read {}: {error}", settings_path.display()))?;
    let settings: Settings =
        serde_json::from_str(&text).map_err(|error| format!("settings: {error}"))?;
    if settings.stage != "schur_cross_check" {
        return Err(format!(
            "settings are stage {:?}, not schur_cross_check",
            settings.stage
        ));
    }
    let pair = settings.negative_control_frequencies;
    let text = std::fs::read_to_string(run.join("export.json"))
        .map_err(|error| format!("read export.json in {}: {error}", run.display()))?;
    let export: Export =
        serde_json::from_str(&text).map_err(|error| format!("export.json: {error}"))?;
    if export.stage != "execute" {
        return Err(format!("export.json is stage {:?}, not execute", export.stage));
    }
    let length = export
        .exports
        .first()
        .map(|checkpoint| checkpoint.p)
        .ok_or_else(|| "export.json lists no checkpoint".to_string())?;
    if let Some(checkpoint) = export.exports.iter().find(|checkpoint| checkpoint.p != length) {
        return Err(format!(
            "checkpoint {} has p = {}, the first has p = {length}",
            checkpoint.name, checkpoint.p
        ));
    }
    println!(
        "[load] run={} checkpoints={} p={length}",
        run.display(),
        export.exports.len()
    );
    println!("[setting] negative_control_frequencies={pair:?}");
    let mut refusals = Vec::new();

    let successor: Vec<usize> = (0..length).map(|row| (row + 1) % length).collect();
    let planted_cycle = RowCycle::from_successor(&successor).map_err(|error| error.to_string())?;
    let plane_count = planted_cycle.plane_count();
    let planted_planes = cyclic_planes(Array2::<f64>::eye(length).view(), &planted_cycle)
        .map_err(|error| error.to_string())?;
    let permutation = Array2::from_shape_fn((length, length), |(row, col)| {
        if row == (col + 1) % length { 1.0 } else { 0.0 }
    });
    let started = Instant::now();
    let (planted_reports, planted_counts) = certify_planes(
        permutation.view(),
        planted_planes.planes.view(),
        &planted_planes.power,
        length,
    );
    let planted_mixed = mixed_plane(&planted_planes, pair)?;
    let (planted_mixed_certified, planted_mixed_report) =
        mixed_control(permutation.view(), planted_mixed.view());
    let certify_seconds = started.elapsed().as_secs_f64();
    let (planted_recovery, planted_summary) = recovery(permutation.view(), length);
    println!(
        "[control] planted permutation: planes certified={}/{plane_count} rotation_scaling={} \
         cosine_inside={} seconds={certify_seconds:.1}; mixed plane certified={planted_mixed_certified}; \
         recovery {planted_summary}",
        planted_counts[0], planted_counts[1], planted_counts[2]
    );
    if planted_counts[1] != plane_count {
        refusals.push(format!(
            "positive control: {} of {plane_count} planted planes certified as rotation-scalings",
            planted_counts[1]
        ));
    }
    if planted_mixed_certified {
        refusals.push("negative control: the planted mixed plane certified".to_string());
    }
    let controls = json!({
        "planted_permutation": {
            "planes": planted_reports,
            "counts": counts_report(planted_counts),
            "mixed_plane": planted_mixed_report,
            "recovery": planted_recovery,
        },
    });
    let mut checkpoints: Vec<Value> = Vec::new();
    let report = |checkpoints: &[Value]| {
        json!({
            "negative_control_frequencies": pair,
            "controls": controls,
            "checkpoints": checkpoints,
        })
    };
    write_report(&out, &report(&checkpoints))?;

    let run_started = Instant::now();
    let phase = |checkpoint: &str, name: &str| {
        println!(
            "[phase] {checkpoint} {name} started at={:.1}s",
            run_started.elapsed().as_secs_f64()
        );
    };
    let mut operators = Vec::with_capacity(export.exports.len());
    for checkpoint in &export.exports {
        let table = read_f4_widened(&run.join(format!("W_E.{}.npy", checkpoint.name)))?;
        let operator = read_f8(&run.join(format!("T1.{}.npy", checkpoint.name)))?;
        let width = checkpoint.d_model;
        if table.dim() != (length + 1, width) || operator.dim() != (width, width) {
            return Err(format!(
                "checkpoint {}: W_E {:?} and T1 {:?}, expected ({}, {width}) and ({width}, {width})",
                checkpoint.name,
                table.dim(),
                operator.dim(),
                length + 1
            ));
        }
        let successor: Vec<usize> = (0..=length)
            .map(|row| if row < length { (row + 1) % length } else { row })
            .collect();
        let cycle = RowCycle::from_successor(&successor).map_err(|error| error.to_string())?;
        let planes = cyclic_planes(table.view(), &cycle).map_err(|error| error.to_string())?;
        let mixed = mixed_plane(&planes, pair)?;

        let (_, singular_values, right) = table
            .slice(s![..length, ..])
            .svd(false, true)
            .map_err(|error| format!("checkpoint {}: {error}", checkpoint.name))?;
        let right = right.ok_or_else(|| {
            format!(
                "checkpoint {}: the SVD returned no right singular vectors",
                checkpoint.name
            )
        })?;
        let sigma_max = singular_values.iter().fold(0.0_f64, |acc, &value| acc.max(value));
        let sigma_min = singular_values
            .iter()
            .fold(f64::INFINITY, |acc, &value| acc.min(value));
        let band = factor_singular_band(length, width, sigma_max);
        let resolved = singular_values.iter().filter(|&&value| value > band).count();
        if resolved < length || right.nrows() < length || right.ncols() != width {
            return Err(format!(
                "checkpoint {}: {resolved} of {length} cycled rows resolved above the SVD band {band:.3e}, \
                 right vectors {:?}",
                checkpoint.name,
                right.dim()
            ));
        }
        let span = right.slice(s![..length, ..]).t().to_owned();
        let restricted = span.t().dot(&operator.dot(&span));
        let restricted_planes = span.t().dot(&planes.planes);
        let restricted_mixed = span.t().dot(&mixed);

        phase(&checkpoint.name, "T1 plane certificates");
        let started = Instant::now();
        let (full_reports, full_counts) =
            certify_planes(operator.view(), planes.planes.view(), &planes.power, length);
        let (full_mixed_certified, full_mixed) = mixed_control(operator.view(), mixed.view());
        let full_seconds = started.elapsed().as_secs_f64();
        phase(&checkpoint.name, "VtT1V plane certificates");
        let started = Instant::now();
        let (restricted_reports, restricted_counts) = certify_planes(
            restricted.view(),
            restricted_planes.view(),
            &planes.power,
            length,
        );
        let (restricted_mixed_certified, restricted_mixed_report) =
            mixed_control(restricted.view(), restricted_mixed.view());
        let restricted_seconds = started.elapsed().as_secs_f64();
        println!(
            "[receipt] {} labels={} step={} kappa={:.4e} (driver {:.4e}) relative_residual={:.3e}: \
             T1 planes certified={}/{plane_count} rotation_scaling={} cosine_inside={} mixed_certified={full_mixed_certified} \
             seconds={full_seconds:.1}; VtT1V planes certified={}/{plane_count} rotation_scaling={} cosine_inside={} \
             mixed_certified={restricted_mixed_certified} seconds={restricted_seconds:.1}",
            checkpoint.name,
            checkpoint.labels,
            checkpoint.step,
            sigma_max / sigma_min,
            checkpoint.sigma_max / checkpoint.sigma_min,
            checkpoint.relative_residual,
            full_counts[0],
            full_counts[1],
            full_counts[2],
            restricted_counts[0],
            restricted_counts[1],
            restricted_counts[2]
        );
        if full_mixed_certified || restricted_mixed_certified {
            refusals.push(format!(
                "negative control: the mixed plane certified at checkpoint {} (T1 {full_mixed_certified}, \
                 VtT1V {restricted_mixed_certified})",
                checkpoint.name
            ));
        }
        checkpoints.push(json!({
            "name": checkpoint.name,
            "labels": checkpoint.labels,
            "step": checkpoint.step,
            "driver_sigma_max": checkpoint.sigma_max,
            "driver_sigma_min": checkpoint.sigma_min,
            "relative_residual": checkpoint.relative_residual,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "singular_band": band,
            "power_share": planes.power.iter().map(|value| value / planes.power.iter().sum::<f64>()).collect::<Vec<f64>>(),
            "T1": {
                "planes": full_reports,
                "counts": counts_report(full_counts),
                "mixed_plane": full_mixed,
                "certify_seconds": full_seconds,
                "recovery": Value::Null,
            },
            "VtT1V": {
                "planes": restricted_reports,
                "counts": counts_report(restricted_counts),
                "mixed_plane": restricted_mixed_report,
                "certify_seconds": restricted_seconds,
                "recovery": Value::Null,
            },
        }));
        write_report(&out, &report(&checkpoints))?;

        phase(&checkpoint.name, "VtT1V recovery");
        let (restricted_recovery, restricted_summary) = recovery(restricted.view(), length);
        println!(
            "[receipt] {} VtT1V recovery {restricted_summary}",
            checkpoint.name
        );
        if let Some(entry) = checkpoints.last_mut() {
            entry["VtT1V"]["recovery"] = restricted_recovery;
        }
        write_report(&out, &report(&checkpoints))?;
        operators.push(operator);
    }

    for (index, (checkpoint, operator)) in export.exports.iter().zip(&operators).enumerate() {
        phase(&checkpoint.name, "T1 recovery");
        let (full_recovery, full_summary) = recovery(operator.view(), length);
        println!("[receipt] {} T1 recovery {full_summary}", checkpoint.name);
        checkpoints[index]["T1"]["recovery"] = full_recovery;
        write_report(&out, &report(&checkpoints))?;
    }

    if !refusals.is_empty() {
        return Err(refusals.join("; "));
    }
    println!("[receipt] wrote {}", out.display());
    Ok(())
}
