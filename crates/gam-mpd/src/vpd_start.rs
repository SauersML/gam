//! VPD's slices with intrinsic gates (#2951): the start of our main line on vpd4l, before any fit.
//!
//! VPD's masks come from a 539M-parameter network that reads every site's input at every position,
//! later ones included (`explanation_battery::vpd_lookahead`). Here each gate reads only its own
//! model's input at its own row, so the replacement is causal and autonomous by construction: no
//! gating network and no all-on pass. Three arms, each run on its own activations layer by layer
//! with the remainder dropped:
//!
//! * `per_slice_own`: every slice is its own component, on at a row iff `|v_iᵀx| > τ_i`, `x` its
//!   site's input there.
//! * `grouped_own`: components span a block's maps. Each read-side slice (q, k, v in an attention
//!   block; c_fc in an MLP block) seeds one, and each write-side slice (o; down_proj) joins the seed
//!   of its block whose mask it co-fires with most under VPD's masks on the fitting rows (the
//!   cosine of the two masks over the rows). A component is on iff `|v_seedᵀx| > τ`, `x` the
//!   block's input (the normed stream entering the attention or the MLP), and then all its slices
//!   run, so its rank counts against the per-token budget.
//! * `grouped_direction`: the same components, on iff `gᵀx + c > τ`, with `g` and `c` the least
//!   squares regression of VPD's mask of the seed on `x` over the fitting rows.
//!
//! A slice whose mask is zero on every fitting row is dropped: VPD never runs it there. Each
//! threshold is the one at which the gate disagrees with VPD's on/off (mask above zero) on the
//! fewest fitting rows. The fit reads `M`'s activations and VPD's masks of them, VPD's own setting.

use crate::{
    device_program::DeviceTrace,
    explanation_battery::{self as battery, KINDS, Kind, Vpd, streams},
    library_mdl::sequence_family,
    operator_program::FamilyInputs,
};
use gam_gpu::tensor::{Op, Tensor};
use gam_linalg::{decompose::eigh, faer_ndarray::fast_atb, roundoff::SymmetricAssembly};
use ndarray::{Array2, Axis, s};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

fn error(e: impl std::fmt::Display) -> String {
    format!("vpd start: {e}")
}

/// Where a site's gates read their input within a layer, in the order a layer computes them: the
/// attention's input (q, k, v), the heads' outputs (o), the MLP's input (c_fc), its activations
/// (down_proj).
fn stage(kind: Kind) -> usize {
    match kind {
        Kind::Query | Kind::Key | Kind::Value => 0,
        Kind::Output => 1,
        Kind::Up => 2,
        Kind::Down => 3,
    }
}

fn kind_of(site: usize) -> Kind {
    KINDS[site % KINDS.len()]
}

/// What a gate reads at its row.
#[derive(Clone, Debug)]
enum Read {
    /// `|v_iᵀx|` of slice `index` of `site`, `x` that site's input.
    Own { site: usize, index: usize },
    /// `gᵀx + c` of the input of `site` (`coefficients`: `g` then `c`).
    Direction { site: usize, coefficients: Vec<f64> },
}

impl Read {
    fn site(&self) -> usize {
        match self {
            Self::Own { site, .. } | Self::Direction { site, .. } => *site,
        }
    }
}

/// A component: its gate's read, threshold and width (the standard deviation of the read over the
/// fitting rows, `library_vpd`'s starting gate width), and its slices (site, index).
#[derive(Clone, Debug)]
struct Component {
    read: Read,
    tau: f64,
    width: f64,
    slices: Vec<(usize, usize)>,
}

/// The standard deviation of `values`.
fn deviation(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = values.clone().count().max(1) as f64;
    let mean = values.clone().sum::<f64>() / n;
    (values.map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt()
}

/// An arm: its components and, per site and slice, the component holding it (none: dropped).
struct Arm {
    name: &'static str,
    components: Vec<Component>,
    holder: Vec<Vec<Option<usize>>>,
}

/// The threshold at which `value > τ` disagrees with `on` on the fewest rows, and that count.
/// The cuts are between distinct consecutive values, below all and at the largest.
fn best_threshold(values: &[f32], on: &[bool]) -> (f64, usize) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut errors = on.iter().filter(|o| !**o).count() as i64;
    let below = values.get(*order.first().unwrap_or(&0)).map_or(0.0, |v| f64::from(*v) - 1.0);
    let mut best = (errors, below);
    for (j, &k) in order.iter().enumerate() {
        errors += if on[k] { 1 } else { -1 };
        let next = order.get(j + 1).map(|&n| values[n]);
        if next == Some(values[k]) {
            continue;
        }
        let tau = match next {
            Some(n) => 0.5 * (f64::from(values[k]) + f64::from(n)),
            None => f64::from(values[k]),
        };
        if errors < best.0 {
            best = (errors, tau);
        }
    }
    (best.1, usize::try_from(best.0).unwrap_or(0))
}

/// Per site, on the fitting rows: `|v_iᵀx|` (rows × C, float32), VPD's masks (float32), and per
/// gate-input site its input with a column of ones (float64), from `M`'s run.
struct Fit {
    reads: Vec<Array2<f32>>,
    masks: Vec<Array2<f32>>,
    inputs: BTreeMap<usize, Array2<f64>>,
}

fn fit_data(vpd: &Vpd, sequences: &[Vec<u32>], batch: usize) -> Result<Fit, String> {
    let d = vpd.e.program.device().clone();
    let sites = vpd.sites.len();
    let v: Vec<Tensor> = vpd.factors.iter().map(|f| d.upload(f.v.view()).map_err(error)).collect::<Result<_, _>>()?;
    let mut reads: Vec<Vec<Array2<f32>>> = vec![Vec::new(); sites];
    let mut masks: Vec<Vec<Array2<f32>>> = vec![Vec::new(); sites];
    let mut inputs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
    for chunk in sequences.chunks(batch) {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = sequence_family(&views)?;
        for (s, g) in vpd.importances(&family)?.into_iter().enumerate() {
            masks[s].push(g.mapv(|x| x as f32));
        }
        let (_, entering) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
        for l in 0..vpd.layers() {
            let entry = if l == 0 { None } else { Some(&entering[l]) };
            let trace = vpd.m.layer_trace(&family, l, entry, BTreeMap::new(), |_, _| Ok(None))?;
            for s in (0..sites).filter(|&s| vpd.sites[s].0 == l) {
                let x = trace.value(vpd.m_layout.inputs[s])?;
                let mut a = d.zeros(family.rows, vpd.sites[s].1).map_err(error)?;
                d.gemm(&mut a, 1.0, x, Op::N, &v[s], Op::N, 0.0, vpd.m.program.arithmetic()).map_err(error)?;
                reads[s].push(d.download(&a).map_err(error)?.mapv(|x| x.abs() as f32));
                if matches!(kind_of(s), Kind::Query | Kind::Up) {
                    let x = d.download(x).map_err(error)?;
                    let mut with_one = Array2::ones((x.nrows(), x.ncols() + 1));
                    with_one.slice_mut(s![.., ..x.ncols()]).assign(&x);
                    inputs.entry(s).or_default().push(with_one);
                }
            }
        }
    }
    let stack32 = |parts: Vec<Array2<f32>>| -> Result<Array2<f32>, String> {
        let views: Vec<_> = parts.iter().map(|a| a.view()).collect();
        ndarray::concatenate(Axis(0), &views).map_err(error)
    };
    let stack64 = |parts: Vec<Array2<f64>>| -> Result<Array2<f64>, String> {
        let views: Vec<_> = parts.iter().map(|a| a.view()).collect();
        ndarray::concatenate(Axis(0), &views).map_err(error)
    };
    Ok(Fit {
        reads: reads.into_iter().map(stack32).collect::<Result<_, _>>()?,
        masks: masks.into_iter().map(stack32).collect::<Result<_, _>>()?,
        inputs: inputs.into_iter().map(|(s, parts)| Ok((s, stack64(parts)?))).collect::<Result<_, String>>()?,
    })
}

/// The three arms (module note) from the fitting rows.
fn arms(vpd: &Vpd, fit: &Fit) -> Result<Vec<Arm>, String> {
    let sites = vpd.sites.len();
    let on = |s: usize| fit.masks[s].mapv(|m| m > 0.0);
    // Per site and slice: alive on the fitting rows, and its own threshold.
    let mut alive: Vec<Vec<bool>> = Vec::with_capacity(sites);
    let mut own_tau: Vec<Vec<f64>> = Vec::with_capacity(sites);
    let mut own_width: Vec<Vec<f64>> = Vec::with_capacity(sites);
    for s in 0..sites {
        let on = on(s);
        let c = vpd.sites[s].1;
        // A slice whose read is constant on every fitting row (zero there: `|v_iᵀx|` with
        // `v_iᵀx = 0`) gates nothing and is dropped with the slices VPD never turns on.
        let fitted: Vec<(bool, f64, f64)> = (0..c)
            .into_par_iter()
            .map(|i| {
                let labels: Vec<bool> = on.column(i).to_vec();
                let values: Vec<f32> = fit.reads[s].column(i).to_vec();
                let width = deviation(values.iter().map(|v| f64::from(*v)));
                if !labels.iter().any(|b| *b) || !(width > 0.0) {
                    return (false, f64::INFINITY, 0.0);
                }
                (true, best_threshold(&values, &labels).0, width)
            })
            .collect();
        alive.push(fitted.iter().map(|f| f.0).collect());
        own_tau.push(fitted.iter().map(|f| f.1).collect());
        own_width.push(fitted.iter().map(|f| f.2).collect());
    }
    let empty_holder = || -> Vec<Vec<Option<usize>>> { vpd.sites.iter().map(|&(_, c, _)| vec![None; c]).collect() };
    // per_slice_own.
    let mut per_slice = Arm { name: "per_slice_own", components: Vec::new(), holder: empty_holder() };
    for s in 0..sites {
        for i in (0..vpd.sites[s].1).filter(|&i| alive[s][i]) {
            per_slice.holder[s][i] = Some(per_slice.components.len());
            per_slice.components.push(Component { read: Read::Own { site: s, index: i }, tau: own_tau[s][i], width: own_width[s][i], slices: vec![(s, i)] });
        }
    }
    // The grouped components: seeds and the write-side slices joining them.
    let mut own = Arm { name: "grouped_own", components: Vec::new(), holder: empty_holder() };
    let mut direction = Arm { name: "grouped_direction", components: Vec::new(), holder: empty_holder() };
    for l in 0..vpd.layers() {
        let site = |kind: Kind| KINDS.len() * l + KINDS.iter().position(|k| *k == kind).unwrap_or(0);
        for (seeds, writes, input) in [(vec![site(Kind::Query), site(Kind::Key), site(Kind::Value)], site(Kind::Output), site(Kind::Query)), (vec![site(Kind::Up)], site(Kind::Down), site(Kind::Up))] {
            // The seeds, in order, and their masks as float64 columns.
            let alive = &alive;
            let seed_list: Vec<(usize, usize)> = seeds.iter().flat_map(|&s| (0..vpd.sites[s].1).filter(move |&i| alive[s][i]).map(move |i| (s, i))).collect();
            let seed_masks = Array2::from_shape_fn((fit.masks[seeds[0]].nrows(), seed_list.len()), |(t, k)| f64::from(fit.masks[seed_list[k].0][[t, seed_list[k].1]]));
            let write_masks = fit.masks[writes].mapv(f64::from);
            let co = fast_atb(&write_masks, &seed_masks);
            let seed_norm: Vec<f64> = seed_masks.columns().into_iter().map(|c| c.dot(&c).sqrt()).collect();
            // The regression of each seed's mask on its block's input (with a column of ones),
            // through the pseudo-inverse of the input's Gram (unresolved directions dropped).
            let x = &fit.inputs[&input];
            let gram = gam_linalg::faer_ndarray::fast_ata(x);
            let inverse = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?.psd_map(0.0, |v| 1.0 / v).map_err(|e| error(format!("{e:?}")))?;
            let coefficients = inverse.dot(&fast_atb(x, &seed_masks));
            let scores = x.dot(&coefficients);
            // Each seed's own read as a direction (`v_i` and no constant), and its values.
            let d_in = x.ncols() - 1;
            let own_coefficients = Array2::from_shape_fn((d_in + 1, seed_list.len()), |(j, k)| if j < d_in { vpd.factors[seed_list[k].0].v[[j, seed_list[k].1]] } else { 0.0 });
            let own_scores = x.dot(&own_coefficients);
            let first = own.components.len();
            for (k, &(s, i)) in seed_list.iter().enumerate() {
                let labels: Vec<bool> = fit.masks[s].column(i).iter().map(|m| *m > 0.0).collect();
                // The direction gate reads the regression's direction or the slice's own read
                // `v_iᵀx` (signed), whichever disagrees with VPD's mask on fewer fitting rows at its
                // best threshold (the regression on ties): a mask on at nearly every row regresses
                // onto the constant alone, a direction of no slope (toys saw such a gate, g = 0 and
                // c = 1, saturate and λ climb to 5.6e8), and the start's widths for this arm went
                // down to 3e-9.
                let candidate = |read: Vec<f64>, values: ndarray::ArrayView1<f64>| {
                    let values32: Vec<f32> = values.iter().map(|v| *v as f32).collect();
                    let (tau, errors) = best_threshold(&values32, &labels);
                    (read, tau, errors, deviation(values.iter().copied()))
                };
                let regression = candidate(coefficients.column(k).to_vec(), scores.column(k));
                let own_read = candidate(own_coefficients.column(k).to_vec(), own_scores.column(k));
                let (read, tau, _, width) = if own_read.2 < regression.2 || !(regression.3 > 0.0) { own_read } else { regression };
                // In units of its read's spread on the fitting rows: `g`, the constant and `τ` over
                // that spread and the width 1, the same gate with its parameters at the data's scale
                // (a direction of spread 3e-9 would otherwise take curvature near 1/w² in `g`).
                let (read, tau, width): (Vec<f64>, f64, f64) = (read.iter().map(|g| g / width).collect(), tau / width, 1.0);
                for arm in [&mut own, &mut direction] {
                    arm.holder[s][i] = Some(first + k);
                }
                own.components.push(Component { read: Read::Own { site: s, index: i }, tau: own_tau[s][i], width: own_width[s][i], slices: vec![(s, i)] });
                direction.components.push(Component { read: Read::Direction { site: input, coefficients: read }, tau, width, slices: vec![(s, i)] });
            }
            for w in (0..vpd.sites[writes].1).filter(|&w| alive[writes][w]) {
                let norm = write_masks.column(w).dot(&write_masks.column(w)).sqrt();
                let best = (0..seed_list.len()).max_by(|&a, &b| {
                    let score = |k: usize| if seed_norm[k] > 0.0 { co[[w, k]] / (norm * seed_norm[k]) } else { 0.0 };
                    score(a).total_cmp(&score(b))
                });
                if let Some(k) = best {
                    for arm in [&mut own, &mut direction] {
                        arm.holder[writes][w] = Some(first + k);
                        arm.components[first + k].slices.push((writes, w));
                    }
                }
            }
        }
    }
    Ok(vec![per_slice, own, direction])
}

/// One arm on `family` run on its own activations, layer by layer (module note): the final
/// residual and, per layer, the slices executed summed over the rows.
fn run_arm(vpd: &Vpd, arm: &Arm, family: &FamilyInputs) -> Result<(Tensor, Vec<f64>), String> {
    let d = vpd.e.program.device().clone();
    let rows = family.rows;
    let mut entry: Option<Tensor> = None;
    let mut executed = vec![0.0; vpd.layers()];
    let mut v: BTreeMap<usize, Tensor> = BTreeMap::new();
    for l in 0..vpd.layers() {
        let layer_sites: Vec<usize> = (0..vpd.sites.len()).filter(|&s| vpd.sites[s].0 == l).collect();
        let mut masks: BTreeMap<usize, Array2<f64>> = layer_sites.iter().map(|&s| (s, Array2::zeros((rows, vpd.sites[s].1)))).collect();
        let run = |masks: &BTreeMap<usize, Array2<f64>>| -> Result<DeviceTrace, String> {
            let mut given = BTreeMap::new();
            for (&s, mask) in masks {
                given.insert(vpd.layout.masks[s], d.upload(mask.view()).map_err(error)?);
                given.insert(vpd.layout.deltas[s], d.zeros(rows, vpd.sites[s].2).map_err(error)?);
            }
            vpd.e.layer_trace(family, l, entry.as_ref(), given, |_, _| Ok(None))
        };
        for stage_index in 0..4 {
            let gated: Vec<usize> = (0..arm.components.len()).filter(|&c| vpd.sites[arm.components[c].read.site()].0 == l && stage(kind_of(arm.components[c].read.site())) == stage_index).collect();
            if gated.is_empty() {
                continue;
            }
            let trace = run(&masks)?;
            // Each gate's value at every row, from its site's input in this run.
            let mut on: BTreeMap<usize, Vec<bool>> = BTreeMap::new();
            let mut by_site: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for &c in &gated {
                by_site.entry(arm.components[c].read.site()).or_default().push(c);
            }
            for (site, components) in by_site {
                let x = trace.value(vpd.layout.inputs[site])?;
                let own: Vec<usize> = components.iter().copied().filter(|&c| matches!(arm.components[c].read, Read::Own { .. })).collect();
                if !own.is_empty() {
                    if !v.contains_key(&site) {
                        v.insert(site, d.upload(vpd.factors[site].v.view()).map_err(error)?);
                    }
                    let mut a = d.zeros(rows, vpd.sites[site].1).map_err(error)?;
                    d.gemm(&mut a, 1.0, x, Op::N, &v[&site], Op::N, 0.0, vpd.e.program.arithmetic()).map_err(error)?;
                    let a = d.download(&a).map_err(error)?;
                    for c in own {
                        let Read::Own { index, .. } = arm.components[c].read else { continue };
                        on.insert(c, a.column(index).iter().map(|x| x.abs() > arm.components[c].tau).collect());
                    }
                }
                let directed: Vec<usize> = components.iter().copied().filter(|&c| matches!(arm.components[c].read, Read::Direction { .. })).collect();
                if !directed.is_empty() {
                    let x = d.download(x).map_err(error)?;
                    let width = x.ncols();
                    let g = Array2::from_shape_fn((width + 1, directed.len()), |(j, k)| match &arm.components[directed[k]].read {
                        Read::Direction { coefficients, .. } => coefficients[j],
                        Read::Own { .. } => 0.0,
                    });
                    let scores = x.dot(&g.slice(s![..width, ..])) + &g.row(width);
                    for (k, &c) in directed.iter().enumerate() {
                        on.insert(c, scores.column(k).iter().map(|z| *z > arm.components[c].tau).collect());
                    }
                }
            }
            for (c, rows_on) in on {
                for &(s, i) in &arm.components[c].slices {
                    let mask = masks.get_mut(&s).ok_or_else(|| error("a component's slice outside its layer"))?;
                    for (t, &o) in rows_on.iter().enumerate() {
                        mask[[t, i]] = if o { 1.0 } else { 0.0 };
                    }
                }
            }
        }
        let trace = run(&masks)?;
        executed[l] = masks.values().map(|m| m.sum()).sum();
        entry = Some(d.copy(trace.value(vpd.e.leaving(l))?).map_err(error)?);
    }
    Ok((entry.ok_or_else(|| error("no layers"))?, executed))
}

/// The arms fitted on `fit_rows` and scored on `held_out` (module note): per arm, held-out
/// `KL(M ‖ P)` in bits per token, the slices executed per token (rank-one equivalents) per layer and
/// in all, its components and the slices it keeps. Every arm's components are written to
/// `components` (JSON: per component its gate read, threshold and slices).
pub fn vpd_start(vpd: &Vpd, fit_rows: &[Vec<u32>], held_out: &[Vec<u32>], batch: usize, components: &Path) -> Result<Value, String> {
    let d = vpd.e.program.device().clone();
    let fit = fit_data(vpd, fit_rows, batch)?;
    let arms = arms(vpd, &fit)?;
    drop(fit);
    let records: Vec<Value> = arms
        .iter()
        .map(|arm| {
            json!({
                "arm": arm.name,
                "components": arm.components.iter().map(|c| {
                    let read = match &c.read {
                        Read::Own { site, index } => json!({"own": [site, index]}),
                        Read::Direction { site, coefficients } => json!({"direction": {"site": site, "coefficients": coefficients}}),
                    };
                    json!({"read": read, "tau": c.tau, "width": c.width, "slices": c.slices})
                }).collect::<Vec<Value>>(),
            })
        })
        .collect();
    std::fs::write(components, serde_json::to_vec(&records).map_err(error)?).map_err(error)?;
    let tokens: usize = held_out.iter().map(Vec::len).sum();
    let mut out = serde_json::Map::new();
    for arm in &arms {
        let (mut kl, mut executed) = (0.0, vec![0.0; vpd.layers()]);
        for chunk in held_out.chunks(batch) {
            let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
            let family = sequence_family(&views)?;
            let length = views[0].len();
            let (_, m_streams) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
            let m_hidden = vpd.m.hidden_of(&family, &m_streams[vpd.layers()])?;
            let (residual, counts) = run_arm(vpd, arm, &family)?;
            let p_hidden = vpd.e.hidden_of(&family, &residual)?;
            kl += battery::divergence_bits(&d, &m_hidden, &p_hidden, &vpd.e, length)?;
            for (a, c) in executed.iter_mut().zip(counts) {
                *a += c;
            }
        }
        let kept: usize = arm.holder.iter().flatten().filter(|h| h.is_some()).count();
        out.insert(
            arm.name.to_string(),
            json!({
                "kl_bits_per_token": kl / tokens as f64,
                "executed_per_token": executed.iter().sum::<f64>() / tokens as f64,
                "executed_per_token_per_layer": executed.iter().map(|e| e / tokens as f64).collect::<Vec<f64>>(),
                "components": arm.components.len(),
                "slices_kept": kept,
            }),
        );
        log::info!("vpd start {}: {}", arm.name, out[arm.name]);
    }
    Ok(Value::Object(out))
}
