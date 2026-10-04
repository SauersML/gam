//! An adversary inside a box claim (#2951): sign ascent over a masked program's off gates, each
//! anywhere in its interval, from the masks, every gate's top and middle, and uniform draws. Every
//! point it visits is a gate setting the claim allows, so the largest KL it finds per input is a
//! lower bound on the box's worst case: it can refute a claim, never certify one.

use gam_linalg::utils::splitmix64;
use ndarray::{Array1, Array2, Axis, Zip, s};

use super::masked::{HeadScreen, Masked, Target, exact_rows, forward, mask_gradients, screened_point};
use super::operator_program::FamilyInputs;

/// Per site, each block's gate interval (rows × blocks).
#[derive(Clone, Debug)]
pub struct Gates {
    pub lower: Vec<Array2<f64>>,
    pub upper: Vec<Array2<f64>>,
}

impl Gates {
    /// The box claim at `masks`: an on gate (positive) fixed at its value, an off gate anywhere in `[0, 1]`.
    pub fn claim(masks: &[Array2<f64>]) -> Self {
        Self {
            lower: masks.iter().map(|m| m.mapv(|x| if x > 0.0 { x } else { 0.0 })).collect(),
            upper: masks.iter().map(|m| m.mapv(|x| if x > 0.0 { x } else { 1.0 })).collect(),
        }
    }
}

/// The next uniform draw on `[0, 1)` of the deterministic generator for the adversary's starts.
fn uniform(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// The largest KL found per input inside `gates`: every point visited by `restarts` sign-ascent runs
/// of `steps` steps on the free gates (those whose interval is wider than a point). The runs start
/// from every gate at its lower end (for the box claim, the masks themselves), at its upper end, at
/// its middle, then at uniform draws; the step shrinks linearly from half of each gate's width to
/// `1/(2 steps)` of it. The ascent climbs the KL of input `focus` alone when given, the total
/// otherwise. A lower bound on the box's worst case.
pub fn adversary(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    gates: &Gates,
    focus: Option<usize>,
    steps: usize,
    restarts: usize,
    seed: u64,
) -> Result<Array1<f64>, String> {
    adversary_screened(masked, base, target, gates, focus, (steps, restarts, seed), HeadScreen::Device)
}

/// [`adversary`] in each box of `boxes` (seeded by `seeds`), every input's KL at the points found
/// per box. On the program's device twin every box's every start climbs at once, as one batch of
/// sequences per step (the same points as one at a time); elsewhere the boxes run in turn.
pub fn adversary_batch(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    boxes: &[Gates],
    steps: usize,
    restarts: usize,
    seeds: &[u64],
) -> Result<Vec<Array1<f64>>, String> {
    if boxes.len() != seeds.len() {
        return Err("adversary batch: one seed per box".to_string());
    }
    let rows = base.rows;
    let copies = boxes.len() * restarts.max(1);
    let lowered = masked.on_device(|_| Ok(()))?.is_some();
    if !lowered || copies <= 1 {
        return boxes.iter().zip(seeds).map(|(gates, &seed)| adversary(masked, base, target, gates, None, steps, restarts, seed)).collect();
    }
    // Every start of every box, in the order a box's own run draws them.
    let mut points: Vec<(usize, Vec<Array2<f64>>)> = Vec::with_capacity(copies);
    for (b, gates) in boxes.iter().enumerate() {
        let mut rng = seeds[b];
        for restart in 0..restarts.max(1) {
            points.push((b, start_point(gates, restart, &mut rng)));
        }
    }
    let mut family = base.clone();
    for _ in 1..copies {
        family = family.append(base).map_err(|e| e.to_string())?;
    }
    let views: Vec<_> = (0..copies).map(|_| target.logits.view()).collect();
    let batch_target = Target {
        logits: ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())?,
        scored: target.scored.as_ref().map(|s| (0..copies).flat_map(|_| s.iter().copied()).collect()),
    };
    let on_device = masked.on_device(|accelerated| accelerated.target(&batch_target))?.ok_or("adversary: the masked program left its device")?;
    drop(batch_target);
    let mut best: Vec<Array1<f64>> = boxes.iter().map(|_| Array1::from_elem(rows, f64::NEG_INFINITY)).collect();
    for step in 0..=steps {
        let sites = points[0].1.len();
        let masks: Vec<Array2<f64>> = (0..sites)
            .map(|k| ndarray::concatenate(Axis(0), &points.iter().map(|(_, p)| p[k].view()).collect::<Vec<_>>()).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        let batch = masked.family(&family, &masks);
        let last = step == steps;
        let (kl, ascent) = masked
            .on_device(|accelerated| {
                let state = accelerated.forward(&batch, &on_device)?;
                let ascent = if last { None } else { Some(accelerated.mask_gradients(masked, &state)?) };
                Ok((state.kl, ascent))
            })?
            .ok_or("adversary: the masked program left its device")?;
        for (c, (b, _)) in points.iter().enumerate() {
            Zip::from(&mut best[*b]).and(&kl.slice(s![c * rows..(c + 1) * rows])).for_each(|m, &v| *m = m.max(v));
        }
        let Some(ascent) = ascent else { break };
        for (c, (b, point)) in points.iter_mut().enumerate() {
            let own: Vec<Array2<f64>> = ascent.iter().map(|a| a.slice(s![c * rows..(c + 1) * rows, ..]).to_owned()).collect();
            climb(point, &boxes[*b], &own, step, steps);
        }
    }
    Ok(best)
}

/// Start `restart` of [`adversary`]: every gate at its lower end, its upper end, its middle, then
/// uniform draws.
fn start_point(gates: &Gates, restart: usize, rng: &mut u64) -> Vec<Array2<f64>> {
    gates
        .lower
        .iter()
        .zip(&gates.upper)
        .map(|(l, u)| {
            Zip::from(l).and(u).map_collect(|&l, &u| match restart {
                0 => l,
                1 => u,
                2 => 0.5 * (l + u),
                _ => l + (u - l) * uniform(rng),
            })
        })
        .collect()
}

/// One sign-ascent step of [`adversary`] on the free gates, its rate shrinking linearly with `step`.
fn climb(point: &mut [Array2<f64>], gates: &Gates, ascent: &[Array2<f64>], step: usize, steps: usize) {
    let rate = 0.5 - (0.5 - 0.5 / steps as f64) * step as f64 / steps.max(2).saturating_sub(1) as f64;
    for (((g, l), u), a) in point.iter_mut().zip(&gates.lower).zip(&gates.upper).zip(ascent) {
        Zip::from(g).and(l).and(u).and(a).for_each(|g, &l, &u, &a| {
            if u > l {
                *g = (*g + rate * (u - l) * a.signum()).clamp(l, u);
            }
        });
    }
}

/// [`adversary`], each point's head run as `screen` says ([`masked::ScreenedPoint`](super::masked::ScreenedPoint)): a point's KL is then
/// known within a band per row, and at the end each row's float64 KL is computed only at the
/// points whose upper end reaches the largest lower end, so the returned maximum is the float64
/// maximum over the same points. The ascent steers from the screened logits.
pub(crate) fn adversary_screened(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    gates: &Gates,
    focus: Option<usize>,
    (steps, restarts, seed): (usize, usize, u64),
    screen: HeadScreen,
) -> Result<Array1<f64>, String> {
    let rows = base.rows;
    let mut rng = seed;
    let mut best = Array1::<f64>::from_elem(rows, f64::NEG_INFINITY);
    // The screened points: per point its rows' KL, band and hidden values.
    let mut screened: Vec<(Array1<f64>, Array1<f64>, Array2<f64>)> = Vec::new();
    // On the program's device twin (masked, module note, "Devices") every point's forward, its
    // float64 KL and its ascent run there; a focused ascent stays on the CPU.
    let on_device = match focus {
        None => masked.on_device(|accelerated| accelerated.target(target))?,
        Some(_) => None,
    };
    for restart in 0..restarts.max(1) {
        let mut point = start_point(gates, restart, &mut rng);
        for step in 0..=steps {
            let family = masked.family(base, &point);
            if let Some(on_device) = &on_device {
                let last = step == steps;
                let (kl, ascent) = masked
                    .on_device(|accelerated| {
                        let state = accelerated.forward(&family, on_device)?;
                        let ascent = if last { None } else { Some(accelerated.mask_gradients(masked, &state)?) };
                        Ok((state.kl, ascent))
                    })?
                    .ok_or("adversary: the masked program left its device")?;
                Zip::from(&mut best).and(&kl).for_each(|b, &v| *b = b.max(v));
                let Some(ascent) = ascent else { break };
                climb(&mut point, gates, &ascent, step, steps);
                continue;
            }
            let ascent = match screened_point(masked, &family, target, screen)? {
                Some(point) => {
                    let ascent = if step == steps { None } else { Some(point.mask_gradients(masked, &family, target, focus)?) };
                    screened.push((point.kl.clone(), point.band.clone(), point.hidden().clone()));
                    ascent
                }
                None => {
                    let (kl, trace, cotangent) = forward(masked, &family, target)?;
                    Zip::from(&mut best).and(&kl).for_each(|b, &v| *b = b.max(v));
                    if step == steps {
                        None
                    } else {
                        let cotangent = match focus {
                            Some(row) => {
                                let mut focused = Array2::<f64>::zeros(cotangent.dim());
                                focused.row_mut(row).assign(&cotangent.row(row));
                                focused
                            }
                            None => cotangent,
                        };
                        Some(mask_gradients(masked, &family, &trace, cotangent)?)
                    }
                }
            };
            let Some(ascent) = ascent else { break };
            climb(&mut point, gates, &ascent, step, steps);
        }
    }
    // Each row settles to float64 at every screened point that could hold its maximum: one whose
    // upper end reaches the largest lower end (or the float64 maximum already known).
    let mut floor = best.clone();
    for (kl, band, _) in &screened {
        Zip::from(&mut floor).and(kl).and(band).for_each(|f, &v, &b| *f = f.max(v - b));
    }
    for (kl, band, hidden) in &screened {
        // A row without a band (unscored) is exact as it stands.
        for r in (0..rows).filter(|r| band[*r] == 0.0) {
            best[r] = best[r].max(kl[r]);
        }
        let open: Vec<usize> = (0..rows).filter(|r| band[*r] > 0.0 && kl[*r] + band[*r] >= floor[*r] && kl[*r] + band[*r] > best[*r]).collect();
        if open.is_empty() {
            continue;
        }
        let exact = exact_rows(masked, target, &hidden.select(Axis(0), &open), &open)?;
        for (i, &r) in open.iter().enumerate() {
            best[r] = best[r].max(exact[i]);
        }
    }
    Ok(best)
}
