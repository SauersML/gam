//! Behavioral curvature map — at which layers does the network COMPUTE on a
//! feature versus merely CARRY it?
//!
//! For a feature that has an independently fitted circular chart at several
//! layers (e.g. the weekday circle), this runner fits every pairwise inter-layer
//! transport (an angle map `h_{l→l'}` with an `O(2)` winding/phase certificate),
//! then, for each layer triangle `(a < b < c)`, closes the composition loop
//! `h_ab , h_bc , h_ac⁻¹` and measures its holonomy with the crate's
//! [`contracts::loop_holonomy`] primitive. Nontrivial holonomy is a
//! gauge-invariant, layer-resolved signature of computation happening on the
//! feature between those layers; flat (near-identity, translation-like)
//! stretches are the feature being carried.
//!
//! Two instruments are combined, both already implemented in `gam-sae`:
//!   * per-edge Fourier-rigidity classification
//!     ([`transport_class::classify_circle_transport_fit`]) — the `O(2)` element
//!     `(winding, phase)` and its departure `defect`;
//!   * per-triangle loop holonomy ([`contracts::loop_holonomy`] via
//!     [`contracts::invert_o2_edge`]) whose trivial/nontrivial tolerance is
//!     DERIVED from the loop's own composed defect (sum of the three edge
//!     defects), never a magic constant; plus the analytic delta-method
//!     composition-law test ([`layer_transport::composition_defect`]) as an
//!     independent calibrated cross-check.
//!
//! # Per-interval computation score (attribution rule)
//!
//! Holonomy is a property of a triangle, not of a single adjacent interval. To
//! localize it, each triangle `(a, b, c)` distributes its EXCESS holonomy
//! `max(0, |net_angle| − angle_tolerance)` — the part of the composition defect
//! that exceeds the loop's own noise floor, in radians — uniformly across every
//! adjacent selected-layer interval its direct edge `(a, c)` spans. A triangle
//! cannot localize its defect finer than the intervals it covers, so uniform
//! spreading is the honest, information-preserving attribution; summing over all
//! triangles then concentrates load on the intervals that repeatedly participate
//! in nontrivial loops. The per-interval `computation_score` is that summed,
//! null-calibrated holonomy load (radians). The adjacent edge's own `O(2)`
//! departure and isometry defect are reported alongside as complementary LOCAL
//! instruments (see `attention_kernel.rs`'s circulant fit for the complementary
//! per-head phase-difference view).
//!
//! # Input formats (sniffed by filename/shape)
//!
//! 1. Banked multi-layer angle JSON (e.g.
//!    `crates/gam-sae/tests/data/qwen3_l11_l17_l23_theta.json`): a single object
//!    with `layer_keys` and `theta: { "acts_L11": [radians…], … }`, all arrays
//!    row-aligned across layers.
//! 2. `experiments/binding_multiplicity/weekday_binding.py` per-layer output: a
//!    directory of per-layer subdirectories (or CSVs), each a `weekday_codes.csv`
//!    with header `weekday,label,z0,z1,…` and a sibling `weekday_frame_meta.json`
//!    carrying the integer `layer`. The circle angle of row `i` is recovered from
//!    the fundamental harmonic code as `atan2(z1, z0)`; the code amplitude
//!    `hypot(z0, z1)` is the per-row on-circle gate.
//! 3. Per-layer single-angle JSON: a file with `theta: [radians…]` and a scalar
//!    `layer` (the natural per-layer shape if the harvest writes angles directly).
//!
//! When per-row gates are present, rows whose amplitude is numerically
//! degenerate in ANY layer (an undefined angle) are dropped so the paired-row
//! structure is preserved; the transport fit itself is unweighted (the public
//! `fit_transport_map` takes no weights), and the honest per-edge fit-quality
//! uncertainty is propagated into the holonomy tolerance through each edge's
//! `O(2)` defect. The mean gate per edge is reported as a quality annotation.
//!
//! Usage:
//!
//! ```text
//! cargo run -p gam-sae --example behavioral_curvature_map -- \
//!   <input_dir_or_json> [out.json] [max_rows] [--grid G]
//! ```

use gam_sae::inference::contracts::{HolonomyReport, invert_o2_edge, loop_holonomy};
use gam_sae::inference::layer_transport::{
    ChartTopology, CompositionDefectReport, FittedTransport, composition_defect, fit_transport_map,
};
use gam_sae::inference::transport_class::{CircleTransportReport, classify_circle_transport_fit};
use ndarray::Array1;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Cap on rows fed to each transport fit (deterministic strided subsample).
const DEFAULT_MAX_ROWS: usize = 20_000;
/// Grid for the O(2) classification and the analytic composition-law test.
const DEFAULT_GRID: usize = 256;
/// Fundamental-harmonic amplitude below this fraction of the per-layer maximum
/// makes the recovered `atan2(z1, z0)` angle numerically undefined; such rows
/// are dropped (see module docs).
const GATE_REL_FLOOR: f64 = 1e-9;

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(path) => {
            // Per house rule, the only stdout line is the final JSON path summary.
            println!("behavioral_curvature_map_json={}", path.display());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[behavioral_curvature_map] error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    input: PathBuf,
    out: PathBuf,
    max_rows: usize,
    grid: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut raw = std::env::args();
        let program = raw
            .next()
            .unwrap_or_else(|| "behavioral_curvature_map".to_string());
        let usage = format!(
            "usage: {program} <input_dir_or_json> [out.json] [max_rows] [--grid G]"
        );
        let mut input: Option<PathBuf> = None;
        let mut positional: Vec<String> = Vec::new();
        let mut grid = DEFAULT_GRID;
        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "--grid" => {
                    let Some(v) = raw.next() else {
                        return Err(format!("--grid requires an integer\n{usage}"));
                    };
                    grid = v
                        .parse::<usize>()
                        .map_err(|e| format!("--grid must be a positive integer: {e}\n{usage}"))?;
                    if grid < 16 {
                        return Err(format!("--grid must be at least 16, got {grid}\n{usage}"));
                    }
                }
                flag if flag.starts_with("--") => {
                    return Err(format!("unknown argument {flag}\n{usage}"));
                }
                value if input.is_none() => input = Some(PathBuf::from(value)),
                value => positional.push(value.to_string()),
            }
        }
        let Some(input) = input else {
            return Err(usage);
        };
        let out = positional
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("behavioral_curvature_map.json"));
        let max_rows = match positional.get(1) {
            Some(text) => text
                .parse::<usize>()
                .map_err(|e| format!("max_rows must be a positive integer: {e}\n{usage}"))
                .and_then(|v| {
                    if v == 0 {
                        Err(format!("max_rows must be positive\n{usage}"))
                    } else {
                        Ok(v)
                    }
                })?,
            None => DEFAULT_MAX_ROWS,
        };
        if positional.len() > 2 {
            return Err(usage);
        }
        Ok(Self {
            input,
            out,
            max_rows,
            grid,
        })
    }
}

/// One layer's row-aligned circular coordinates (radians) plus optional per-row
/// on-circle gate and a provenance label.
#[derive(Clone, Debug)]
struct LayerSource {
    layer: i64,
    theta: Vec<f64>,
    gate: Option<Vec<f64>>,
    provenance: String,
}

/// The whole loaded atlas: the sniffed format plus every layer's coordinates.
struct LoadedAtlas {
    format: String,
    model: Option<String>,
    coordinate: Option<String>,
    layers: Vec<LayerSource>,
}

fn run(args: &Args) -> Result<PathBuf, String> {
    let atlas = load_atlas(&args.input)?;
    if atlas.layers.len() < 2 {
        return Err(format!(
            "need at least 2 layers to fit a transport, found {}",
            atlas.layers.len()
        ));
    }

    // Row-align every layer, apply the gate mask (dropping numerically degenerate
    // rows in ANY layer), and strided-subsample to max_rows.
    let n_full = atlas.layers[0].theta.len();
    for l in &atlas.layers {
        if l.theta.len() != n_full {
            return Err(format!(
                "layer {} has {} rows but layer {} has {n_full}; every layer must coordinatize \
                 the same rows",
                l.layer,
                l.theta.len(),
                atlas.layers[0].layer
            ));
        }
    }
    let has_gates = atlas.layers.iter().any(|l| l.gate.is_some());
    let keep_mask = build_keep_mask(&atlas.layers, n_full);
    let kept: Vec<usize> = (0..n_full).filter(|&i| keep_mask[i]).collect();
    let n_kept = kept.len();
    let take = args.max_rows.min(n_kept);
    if take < 16 {
        return Err(format!(
            "only {take} usable paired rows after gating (need at least 16); \
             n_full={n_full}, kept={n_kept}"
        ));
    }
    let stride = (n_kept / take).max(1);
    let sampled: Vec<usize> = (0..take).map(|i| kept[i * stride]).collect();
    let n_used = sampled.len();

    // Materialize per-layer subsampled coordinate vectors (radians).
    let mut layers: Vec<i64> = atlas.layers.iter().map(|l| l.layer).collect();
    // Sort ascending; carry the coordinate/gate vectors with the order.
    let mut order: Vec<usize> = (0..atlas.layers.len()).collect();
    order.sort_by_key(|&i| atlas.layers[i].layer);
    layers.sort();
    for w in layers.windows(2) {
        if w[0] == w[1] {
            return Err(format!("duplicate layer index {} across sources", w[0]));
        }
    }
    let coords: Vec<Array1<f64>> = order
        .iter()
        .map(|&i| Array1::from_iter(sampled.iter().map(|&r| atlas.layers[i].theta[r])))
        .collect();
    let gates: Vec<Option<Array1<f64>>> = order
        .iter()
        .map(|&i| {
            atlas.layers[i]
                .gate
                .as_ref()
                .map(|g| Array1::from_iter(sampled.iter().map(|&r| g[r])))
        })
        .collect();
    let provenance: Vec<String> = order
        .iter()
        .map(|&i| atlas.layers[i].provenance.clone())
        .collect();

    let depth = layers.len();

    // --- fit every pairwise transport (all circle→circle) -------------------
    // Store fits keyed by (from_idx, to_idx) with from_idx < to_idx.
    let mut fits: BTreeMap<(usize, usize), FittedTransport> = BTreeMap::new();
    let mut classes: BTreeMap<(usize, usize), CircleTransportReport> = BTreeMap::new();
    for a in 0..depth {
        for b in (a + 1)..depth {
            let fit = fit_transport_map(
                coords[a].view(),
                coords[b].view(),
                ChartTopology::Circle,
                ChartTopology::Circle,
            )
            .map_err(|e| {
                format!("transport {}→{} failed: {e}", layers[a], layers[b])
            })?;
            let class = classify_circle_transport_fit(
                &fit,
                ChartTopology::Circle,
                ChartTopology::Circle,
                layers[a] as usize,
                layers[b] as usize,
                args.grid,
            )
            .ok_or_else(|| {
                format!(
                    "O(2) classification of {}→{} returned nothing (non-circle endpoint?)",
                    layers[a], layers[b]
                )
            })?;
            classes.insert((a, b), class);
            fits.insert((a, b), fit);
        }
    }

    // --- per-edge JSON (every pair) -----------------------------------------
    let mut edges_json: Vec<Value> = Vec::new();
    for a in 0..depth {
        for b in (a + 1)..depth {
            let fit = &fits[&(a, b)];
            let class = &classes[&(a, b)];
            let mean_gate = edge_mean_gate(&gates[a], &gates[b]);
            edges_json.push(json!({
                "from": layers[a],
                "to": layers[b],
                "adjacent": b == a + 1,
                "winding": class.winding,
                "class": format!("{:?}", class.class),
                "phase_rad": jf(class.phase),
                "phase_deg": jf(class.phase_degrees()),
                "o2_defect": jf(class.defect),
                "resultant_shift": jf(class.resultant_shift),
                "resultant_reflect": jf(class.resultant_reflect),
                "degree": fit.degree,
                "degree_concentration": fit.degree_concentration.map(jf).unwrap_or(Value::Null),
                "isometry_defect": jf(fit.isometry_defect),
                "isometry_defect_se": jf(fit.isometry_defect_se),
                "topology_preserved": fit.topology_preserved,
                "min_directional_derivative": jf(fit.min_directional_derivative),
                "residual_rms": jf(fit.residual_rms),
                "transport_edf": jf(fit.edf),
                "smoothing_lambda": jf(fit.smoothing_lambda),
                "noise_variance": jf(fit.noise_variance),
                "carried_fraction": jf((1.0 - class.defect).clamp(0.0, 1.0)),
                "mean_gate": mean_gate.map(jf).unwrap_or(Value::Null),
            }));
        }
    }

    // --- per-triangle holonomy + analytic composition cross-check -----------
    // Accumulate per-adjacent-interval holonomy load along the way.
    let mut interval_load = vec![0.0_f64; depth.saturating_sub(1)];
    let mut triangles_json: Vec<Value> = Vec::new();
    for a in 0..depth {
        for b in (a + 1)..depth {
            for c in (b + 1)..depth {
                let e_ab = &classes[&(a, b)];
                let e_bc = &classes[&(b, c)];
                let e_ac = &classes[&(a, c)];
                // Close the loop a→b→c→a: forward hops then the inverse of the
                // direct edge. loop_holonomy measures exactly the failure of the
                // composition law h_ac = h_bc ∘ h_ab as an O(2) element.
                let edges = [
                    (e_ab.winding, e_ab.phase),
                    (e_bc.winding, e_bc.phase),
                    invert_o2_edge((e_ac.winding, e_ac.phase)),
                ];
                let defects = [e_ab.defect, e_bc.defect, e_ac.defect];
                let holo: HolonomyReport = loop_holonomy(&edges, &defects);
                let net_abs = holo.net_angle.abs();
                let excess = (net_abs - holo.angle_tolerance).max(0.0);
                let significance = if holo.angle_tolerance > 0.0 {
                    Some(net_abs / holo.angle_tolerance)
                } else if net_abs > 0.0 {
                    None // undefined (zero-noise loop with nonzero angle)
                } else {
                    Some(0.0)
                };

                // Analytic delta-method composition-law test (independent,
                // calibrated). Best-effort: record its error rather than aborting
                // the whole map if a single triple degenerates.
                let comp: Result<CompositionDefectReport, String> = composition_defect(
                    &fits[&(a, b)],
                    &fits[&(b, c)],
                    &fits[&(a, c)],
                    args.grid,
                );

                // Attribute the excess holonomy uniformly across the adjacent
                // intervals spanned by the direct edge (a..c).
                let span = c - a; // number of adjacent intervals from a to c
                if span > 0 {
                    let share = excess / span as f64;
                    for iv in a..c {
                        interval_load[iv] += share;
                    }
                }

                triangles_json.push(json!({
                    "layers": [layers[a], layers[b], layers[c]],
                    "edge_ab": [layers[a], layers[b]],
                    "edge_bc": [layers[b], layers[c]],
                    "edge_ac": [layers[a], layers[c]],
                    "holonomy": {
                        "loop_len": holo.loop_len,
                        "net_sign": holo.net_sign,
                        "net_angle_rad": jf(holo.net_angle),
                        "net_angle_deg": jf(holo.net_angle.to_degrees()),
                        "angle_tolerance_rad": jf(holo.angle_tolerance),
                        "is_trivial": holo.is_trivial,
                        "significance": significance.map(jf).unwrap_or(Value::Null),
                        "excess_holonomy_rad": jf(excess),
                        "verdict": if holo.is_trivial { "carried" } else { "computes" },
                    },
                    "composition_defect": match &comp {
                        Ok(r) => json!({
                            "rms_defect": jf(r.rms_defect),
                            "mean_abs_defect": jf(r.mean_abs_defect),
                            "max_abs_defect": jf(r.max_abs_defect),
                            "max_studentized_defect": jf(r.max_studentized_defect),
                            "max_studentized_p_value": jf(r.max_studentized_p_value),
                            "p_value": jf(r.p_value),
                            "gauge_reflected": r.gauge_reflected,
                            "n_grid": r.n_grid,
                        }),
                        Err(e) => json!({ "error": e }),
                    },
                }));
            }
        }
    }

    // --- per adjacent-interval computation score ----------------------------
    let mut intervals_json: Vec<Value> = Vec::new();
    for iv in 0..depth.saturating_sub(1) {
        let class = &classes[&(iv, iv + 1)];
        let fit = &fits[&(iv, iv + 1)];
        let score = interval_load[iv];
        intervals_json.push(json!({
            "from": layers[iv],
            "to": layers[iv + 1],
            "local_o2_defect": jf(class.defect),
            "local_isometry_defect": jf(fit.isometry_defect),
            "holonomy_load_rad": jf(score),
            "computation_score": jf(score),
            "verdict": if score > class.defect.max(1e-6) {
                "computes"
            } else {
                "carried"
            },
        }));
    }

    // --- headline summary ---------------------------------------------------
    let (max_iv, max_score) = interval_load
        .iter()
        .enumerate()
        .fold((None, 0.0_f64), |(mi, ms), (i, &v)| {
            if v > ms { (Some(i), v) } else { (mi, ms) }
        });
    let any_nontrivial = triangles_json.iter().any(|t| {
        t.get("holonomy")
            .and_then(|h| h.get("is_trivial"))
            .and_then(Value::as_bool)
            == Some(false)
    });
    let summary = json!({
        "n_layers": depth,
        "n_edges": edges_json.len(),
        "n_triangles": triangles_json.len(),
        "any_nontrivial_triangle": any_nontrivial,
        "max_computation_interval": max_iv.map(|i| json!([layers[i], layers[i + 1]])).unwrap_or(Value::Null),
        "max_computation_score_rad": jf(max_score),
    });

    let layer_provenance: Vec<Value> = layers
        .iter()
        .zip(provenance.iter())
        .map(|(l, p)| json!({ "layer": l, "source": p }))
        .collect();
    let report = json!({
        "runner": "behavioral_curvature_map",
        "input_format": atlas.format,
        "model": atlas.model.clone().map(Value::String).unwrap_or(Value::Null),
        "coordinate": atlas.coordinate.clone().map(Value::String).unwrap_or(Value::Null),
        "topology": "circle",
        "has_per_row_gates": has_gates,
        "n_rows_total": n_full,
        "n_rows_kept_after_gating": n_kept,
        "n_rows_used": n_used,
        "composition_grid": args.grid,
        "layers": layers,
        "layer_provenance": layer_provenance,
        "attribution_rule": "per-interval computation_score = summed EXCESS holonomy \
            max(0, |net_angle| - angle_tolerance) of every triangle whose direct edge spans the \
            interval, split uniformly across the adjacent intervals that edge covers; \
            angle_tolerance is the loop's own composed defect (sum of the three edge O(2) defects), \
            so the score is null-calibrated in radians.",
        "summary": summary,
        "edges": edges_json,
        "triangles": triangles_json,
        "intervals": intervals_json,
    });

    let text = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("serialize report: {e}"))?;
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(&args.out, format!("{text}\n"))
        .map_err(|e| format!("write {}: {e}", args.out.display()))?;
    Ok(args.out.clone())
}

/// Non-finite-safe f64 → JSON (maps NaN/±inf to null; serde_json would otherwise
/// emit them as null anyway, but this is explicit).
fn jf(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        Value::Null
    }
}

/// Keep row `i` iff its angle is finite in every layer and — where gates exist —
/// its on-circle amplitude clears the per-layer numerical floor in every layer.
fn build_keep_mask(layers: &[LayerSource], n: usize) -> Vec<bool> {
    // Per-layer amplitude floors (relative to the layer's max gate).
    let floors: Vec<Option<f64>> = layers
        .iter()
        .map(|l| {
            l.gate.as_ref().map(|g| {
                let gmax = g.iter().copied().filter(|v| v.is_finite()).fold(0.0_f64, f64::max);
                gmax * GATE_REL_FLOOR
            })
        })
        .collect();
    let mut mask = vec![true; n];
    for (li, l) in layers.iter().enumerate() {
        for i in 0..n {
            if !l.theta[i].is_finite() {
                mask[i] = false;
                continue;
            }
            if let (Some(g), Some(floor)) = (l.gate.as_ref(), floors[li]) {
                let gi = g[i];
                if !gi.is_finite() || gi <= floor {
                    mask[i] = false;
                }
            }
        }
    }
    mask
}

/// Mean of the two endpoints' per-row gates over the subsampled rows (already
/// aligned), or `None` when neither endpoint carries gates.
fn edge_mean_gate(ga: &Option<Array1<f64>>, gb: &Option<Array1<f64>>) -> Option<f64> {
    match (ga, gb) {
        (None, None) => None,
        _ => {
            let mut sum = 0.0_f64;
            let mut cnt = 0usize;
            for g in [ga, gb].into_iter().flatten() {
                for &v in g.iter() {
                    if v.is_finite() {
                        sum += v;
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 { None } else { Some(sum / cnt as f64) }
        }
    }
}

// ============================ input loading =================================

/// Sniff the input (file or directory) and load every layer's circular
/// coordinates.
fn load_atlas(input: &Path) -> Result<LoadedAtlas, String> {
    let meta = std::fs::metadata(input)
        .map_err(|e| format!("stat {}: {e}", input.display()))?;

    // A single JSON file: either the banked multi-layer form or a per-layer form.
    if meta.is_file() {
        let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("json") {
            let value = read_json(input)?;
            if value.get("theta").map(Value::is_object).unwrap_or(false) {
                return load_banked_theta(&value, input);
            }
            if value.get("theta").map(Value::is_array).unwrap_or(false) {
                let layer = load_single_layer_json(&value, input)?;
                return Ok(LoadedAtlas {
                    format: "per_layer_theta_json".to_string(),
                    model: value.get("model").and_then(Value::as_str).map(str::to_string),
                    coordinate: value
                        .get("coordinate")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    layers: vec![layer],
                });
            }
        }
        if input
            .file_name()
            .and_then(|f| f.to_str())
            .map(|f| f.eq_ignore_ascii_case("weekday_codes.csv"))
            .unwrap_or(false)
        {
            let dir = input.parent().unwrap_or(Path::new("."));
            let layer = load_weekday_codes(input, dir)?;
            return Ok(LoadedAtlas {
                format: "weekday_binding_codes".to_string(),
                model: None,
                coordinate: Some("weekday-circle fundamental-harmonic angle".to_string()),
                layers: vec![layer],
            });
        }
        return Err(format!(
            "{}: unrecognized input file (expected a *.json with a `theta` field or a \
             weekday_codes.csv)",
            input.display()
        ));
    }

    // A directory: collect banked JSONs, per-layer JSONs, and weekday CSVs.
    let mut layers: Vec<LayerSource> = Vec::new();
    let mut formats: Vec<&str> = Vec::new();
    let mut model: Option<String> = None;
    let mut coordinate: Option<String> = None;

    let mut candidates: Vec<PathBuf> = Vec::new();
    collect_candidates(input, &mut candidates, 0)?;
    candidates.sort();

    for path in &candidates {
        let name = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if name.eq_ignore_ascii_case("weekday_codes.csv") {
            let dir = path.parent().unwrap_or(input);
            layers.push(load_weekday_codes(path, dir)?);
            formats.push("weekday_binding_codes");
            if coordinate.is_none() {
                coordinate = Some("weekday-circle fundamental-harmonic angle".to_string());
            }
        } else if name.eq_ignore_ascii_case("weekday_frame_meta.json") {
            // Consumed as sibling metadata by load_weekday_codes; skip standalone.
            continue;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("json"))
            .unwrap_or(false)
        {
            let value = read_json(path)?;
            if value.get("theta").map(Value::is_object).unwrap_or(false) {
                let banked = load_banked_theta(&value, path)?;
                model = model.or(banked.model);
                coordinate = coordinate.or(banked.coordinate);
                layers.extend(banked.layers);
                formats.push("banked_theta_json");
            } else if value.get("theta").map(Value::is_array).unwrap_or(false) {
                layers.push(load_single_layer_json(&value, path)?);
                formats.push("per_layer_theta_json");
                model = model.or_else(|| {
                    value.get("model").and_then(Value::as_str).map(str::to_string)
                });
            }
        }
    }

    if layers.is_empty() {
        return Err(format!(
            "{}: no layer sources found (expected banked *_theta.json, per-layer theta json, \
             or weekday_codes.csv files)",
            input.display()
        ));
    }
    formats.sort();
    formats.dedup();
    let format = if formats.len() == 1 {
        formats[0].to_string()
    } else {
        "mixed".to_string()
    };
    Ok(LoadedAtlas {
        format,
        model,
        coordinate,
        layers,
    })
}

/// Collect layer-bearing files up to one directory level below `dir` (the
/// natural weekday_binding.py per-layer-subdir layout).
fn collect_candidates(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("read dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read entry in {}: {e}", dir.display()))?;
        let path = entry.path();
        let ty = entry
            .file_type()
            .map_err(|e| format!("file type {}: {e}", path.display()))?;
        if ty.is_dir() {
            if depth < 1 {
                collect_candidates(&path, out, depth + 1)?;
            }
        } else if ty.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Banked multi-layer angle JSON: `layer_keys` + `theta: { key: [rad…] }`.
fn load_banked_theta(value: &Value, path: &Path) -> Result<LoadedAtlas, String> {
    let theta = value
        .get("theta")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{}: `theta` is not an object", path.display()))?;
    // Preserve `layer_keys` order when present; else the object's own order.
    let keys: Vec<String> = match value.get("layer_keys").and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        None => theta.keys().cloned().collect(),
    };
    let mut layers = Vec::new();
    for key in &keys {
        let arr = theta
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{}: theta[{key}] missing or not an array", path.display()))?;
        let angles: Vec<f64> = arr
            .iter()
            .map(|v| v.as_f64().ok_or_else(|| format!("{}: non-numeric angle in {key}", path.display())))
            .collect::<Result<_, _>>()?;
        layers.push(LayerSource {
            layer: parse_layer_key(key),
            theta: angles,
            gate: None,
            provenance: format!("{}::{key}", path.display()),
        });
    }
    Ok(LoadedAtlas {
        format: "banked_theta_json".to_string(),
        model: value.get("model").and_then(Value::as_str).map(str::to_string),
        coordinate: value
            .get("coordinate")
            .and_then(Value::as_str)
            .map(str::to_string),
        layers,
    })
}

/// Per-layer angle JSON: `theta: [rad…]` + a scalar `layer` (or `layer_key`).
fn load_single_layer_json(value: &Value, path: &Path) -> Result<LayerSource, String> {
    let arr = value
        .get("theta")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{}: `theta` is not an array", path.display()))?;
    let angles: Vec<f64> = arr
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| format!("{}: non-numeric theta value", path.display())))
        .collect::<Result<_, _>>()?;
    let layer = value
        .get("layer")
        .and_then(Value::as_i64)
        .or_else(|| value.get("layer_key").and_then(Value::as_str).map(parse_layer_key))
        .ok_or_else(|| format!("{}: per-layer theta json needs a `layer` integer", path.display()))?;
    // Optional per-row gate array under a few natural names.
    let gate = ["gate", "weight", "amplitude"]
        .iter()
        .find_map(|k| value.get(*k).and_then(Value::as_array))
        .map(|g| {
            g.iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<f64>>()
        });
    Ok(LayerSource {
        layer,
        theta: angles,
        gate,
        provenance: path.display().to_string(),
    })
}

/// weekday_binding.py `weekday_codes.csv` (header `weekday,label,z0,z1,…`): the
/// circle angle of each row is the fundamental-harmonic phase `atan2(z1, z0)` and
/// its gate is the amplitude `hypot(z0, z1)`. The layer index is read from the
/// sibling `weekday_frame_meta.json`, falling back to the parent dir name.
fn load_weekday_codes(csv: &Path, dir: &Path) -> Result<LayerSource, String> {
    let text = std::fs::read_to_string(csv)
        .map_err(|e| format!("read {}: {e}", csv.display()))?;
    let mut theta = Vec::new();
    let mut gate = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip the header row (`weekday,label,z0,…`); data rows start with a
        // weekday name (`Monday,…`), never `weekday,`.
        if trimmed.starts_with("weekday,") {
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').collect();
        // Columns: weekday(str), label(int), z0, z1, ...
        if fields.len() < 4 {
            return Err(format!(
                "{}:{}: expected at least 4 columns (weekday,label,z0,z1), got {}",
                csv.display(),
                lineno + 1,
                fields.len()
            ));
        }
        let z0 = fields[2]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("{}:{}: parse z0: {e}", csv.display(), lineno + 1))?;
        let z1 = fields[3]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("{}:{}: parse z1: {e}", csv.display(), lineno + 1))?;
        theta.push(z1.atan2(z0));
        gate.push(z0.hypot(z1));
    }
    if theta.is_empty() {
        return Err(format!("{}: no data rows", csv.display()));
    }
    let layer = weekday_layer_index(dir).unwrap_or_else(|| parse_layer_key(
        dir.file_name().and_then(|f| f.to_str()).unwrap_or(""),
    ));
    Ok(LayerSource {
        layer,
        theta,
        gate: Some(gate),
        provenance: csv.display().to_string(),
    })
}

/// Read the integer `layer` from a sibling `weekday_frame_meta.json`, if present.
fn weekday_layer_index(dir: &Path) -> Option<i64> {
    let meta_path = dir.join("weekday_frame_meta.json");
    let text = std::fs::read_to_string(&meta_path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    value.get("layer").and_then(Value::as_i64)
}

/// Extract the integer layer index from a key like `acts_L11`, `L17`, `layer_23`,
/// or a bare number; falls back to the first run of digits, else 0.
fn parse_layer_key(key: &str) -> i64 {
    // Prefer digits immediately following an 'L'/'l' marker.
    let bytes = key.as_bytes();
    for i in 0..bytes.len() {
        if (bytes[i] == b'L' || bytes[i] == b'l')
            && i + 1 < bytes.len()
            && bytes[i + 1].is_ascii_digit()
        {
            let digits: String = key[i + 1..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(v) = digits.parse::<i64>() {
                return v;
            }
        }
    }
    // Otherwise the first run of digits anywhere.
    let digits: String = key
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<i64>().unwrap_or(0)
}
