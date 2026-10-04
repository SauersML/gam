#![cfg(test)]
//! The masked fit's device path against its CPU twins in `masked`, on the small rotary language
//! model of `device_program_tests` with every site replaced by a random three-piece library and
//! random masks. The bands are that module's: the KL within twice the logits' band plus each
//! evaluation's rounding; every proposal (mask gradients, pieces' gradients, Fishers, covariances,
//! the step's curvature) within the f32 proposal band of the reference's largest entry.

use super::derivatives::jvp;
use super::device_program_tests::{assert_proposal, devices, fixture, gamma, noise, softmax};
use super::masked::{self, Library, Masked, Target, matrix, read_values, sites};
use super::masked_device::Accelerated;
use super::operator_program::FamilyInputs;
use gam_gpu::tensor::Arithmetic;
use gam_linalg::faer_ndarray::fast_atb;
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;

const PIECES: usize = 3;

fn masked_fixture() -> (Masked, FamilyInputs, Vec<Array2<f64>>, Array2<f64>) {
    let (program, base) = fixture();
    let all = sites(&program);
    let mut libraries = Vec::new();
    for (k, site) in all.iter().enumerate() {
        let (d_out, d_in) = matrix(&program, site).expect("map").dim();
        let at = |salt: usize| move |(i, j): (usize, usize)| noise(10_000 * k + 1000 * salt + 37 * i + j);
        libraries.push(Library {
            v: Array2::from_shape_fn((PIECES, d_in), at(1)) * 0.5,
            u: Array2::from_shape_fn((PIECES, d_out), at(2)) * 0.5,
            mean: Array1::from_shape_fn(d_in, |i| 0.1 * noise(10_000 * k + 7 * i + 3)),
        });
    }
    let masks: Vec<Array2<f64>> = (0..all.len())
        .map(|k| Array2::from_shape_fn((base.rows, PIECES), |(r, c)| if noise(500 * k + 11 * r + c + 1) > -0.2 { 1.0 } else { 0.0 }))
        .collect();
    let clean = program.execute(&base, false).expect("clean").values[program.output].clone();
    let masked = Masked::build(&program, all, libraries).expect("masked");
    (masked, base, masks, clean)
}

fn targets(clean: &Array2<f64>) -> [Target; 2] {
    let half: Vec<bool> = (0..clean.nrows()).map(|r| r % 3 != 1).collect();
    [Target::every_row(clean.clone()), Target { logits: clean.clone(), scored: Some(half) }]
}

#[test]
fn the_masked_device_path_matches_the_cpu_within_bands() {
    let (masked, base, masks, clean) = masked_fixture();
    let family = masked.family(&base, &masks);
    let banded = masked.program.execute(&family, true).expect("banded");
    let output_band = &banded.bands.as_ref().expect("bands")[masked.program.output];
    let output_ball = &banded.balls.as_ref().expect("balls")[masked.program.output];
    let widest = 16;
    for target in targets(&clean) {
        let (kl, trace, cotangent) = masked::forward(&masked, &family, &target).expect("cpu forward");
        let mask_gradients = masked::mask_gradients(&masked, &family, &trace, cotangent.clone()).expect("cpu mask gradients");
        let gradients = masked::gradients(&masked, &family, &trace, &masks, cotangent).expect("cpu gradients");
        let fisher = masked::fisher(&masked, &family, &trace, &target, 2, 0x5EED, true).expect("cpu fisher");
        let covariances: Vec<Array2<f64>> = masked
            .sites
            .iter()
            .map(|site| {
                let centred = read_values(&trace, site).expect("reads");
                fast_atb(&centred, &centred) / family.rows as f64
            })
            .collect();
        // A direction on the first site's pieces, by operator name, and the CPU's curvature on it.
        let named = |suffix: &str| {
            let name = format!("{}·{suffix}", masked.sites[0].name);
            masked.program.operators.iter().position(|op| op.name == name).expect("a library operator")
        };
        let mut tangents = BTreeMap::new();
        for (salt, suffix) in ["V0", "U0"].into_iter().enumerate() {
            let op = named(suffix);
            let (rows, cols) = masked.program.operators[op].matrix().dim();
            tangents.insert(op, Array2::from_shape_fn((rows, cols), |(i, j)| noise(90_000 + 1000 * salt + 31 * i + j)));
        }
        let output = jvp(&masked.program, &family, &trace, &tangents).expect("cpu jvp");
        let logits = &trace.values[masked.program.output];
        let mut quadratic = 0.0;
        for r in (0..family.rows).filter(|r| target.scores(*r)) {
            let q = softmax(logits.row(r));
            let t = output.row(r);
            let mean: f64 = q.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
            quadratic += q.iter().zip(t.iter()).map(|(a, b)| a * (b - mean) * (b - mean)).sum::<f64>();
        }
        for device in devices() {
            let name = device.name();
            let accelerated = Accelerated::new(&device, &masked, Arithmetic::F64).expect("lowered");
            let on_device = accelerated.target(&target).expect("target");
            let state = accelerated.forward(&family, &on_device).expect("device forward");
            let mut deferred = accelerated.score_state(&family, &on_device).expect("score state");
            assert_eq!(deferred.kl, state.kl);
            accelerated.prepare_gradient(&mut deferred, &on_device).expect("deferred gradient");
            assert_eq!(accelerated.mask_gradients(&masked, &deferred).expect("deferred masks"), accelerated.mask_gradients(&masked, &state).expect("masks"));
            let full = accelerated.gradients(&masked, &state, &masks).expect("full gradients");
            for moves_u in [false, true] {
                let partial = accelerated.piece_gradients(&masked, &state, moves_u).expect("active side");
                for (full, partial) in full.iter().zip(partial) {
                    assert!(partial.0.is_empty());
                    if moves_u { assert!(partial.1.is_empty()); assert_eq!(partial.2, full.2); }
                    else { assert!(partial.2.is_empty()); assert_eq!(partial.1, full.1); }
                }
            }
            let full_fisher = accelerated.fisher(&masked, &state, &on_device, 2, 0x5EED, true).expect("full Fisher");
            let step_fisher = accelerated.step_fisher(&masked, &state, &on_device, 2, 0x5EED).expect("step Fisher");
            for (full, partial) in full_fisher.iter().zip(step_fisher) { assert!(partial.0.is_empty()); assert_eq!(partial.1, full.1); }
            assert_eq!(accelerated.score_only(&family, &on_device).expect("score only"), state.kl);
            for r in 0..family.rows {
                if !target.scores(r) {
                    assert_eq!(state.kl[r], 0.0, "{name}: an unscored row has no KL");
                    continue;
                }
                let (p, q) = (softmax(target.logits.row(r)), softmax(logits.row(r)));
                let magnitude: f64 = p.iter().zip(q.iter()).map(|(a, b)| a * (a.ln().abs() + b.ln().abs())).sum();
                let moved = 2.0 * (output_band.row(r).iter().fold(0.0_f64, |m, v| m.max(*v)) + output_ball[r]);
                let band = 2.0 * moved + 2.0 * gamma(clean.ncols() + 8) * magnitude;
                assert!((state.kl[r] - kl[r]).abs() <= band, "{name}: KL of row {r} differs by {:e}, band {band:e}", (state.kl[r] - kl[r]).abs());
            }
            for (k, g) in accelerated.mask_gradients(&masked, &state).expect("device mask gradients").iter().enumerate() {
                assert_proposal(&format!("{name}: mask gradient of site {k}"), g, &mask_gradients[k], widest);
            }
            for (k, (m, v, u)) in accelerated.gradients(&masked, &state, &masks).expect("device gradients").iter().enumerate() {
                assert_proposal(&format!("{name}: mask gradient of site {k}"), m, &gradients[k].0, widest);
                assert_proposal(&format!("{name}: V gradient of site {k}"), v, &gradients[k].1, family.rows);
                assert_proposal(&format!("{name}: U gradient of site {k}"), u, &gradients[k].2, family.rows);
            }
            for (k, (h, f)) in accelerated.fisher(&masked, &state, &on_device, 2, 0x5EED, true).expect("device fisher").iter().enumerate() {
                assert_proposal(&format!("{name}: mask Fisher of site {k}"), h, &fisher[k].0, widest);
                let (f, reference) = (f.as_ref().expect("written"), fisher[k].1.as_ref().expect("written"));
                assert_proposal(&format!("{name}: written Fisher of site {k}"), f, reference, family.rows);
            }
            for (k, c) in accelerated.covariances(&masked, &state).expect("device covariances").iter().enumerate() {
                assert_proposal(&format!("{name}: read covariance of site {k}"), c, &covariances[k], family.rows);
            }
            let device_quadratic = accelerated.quadratic(&state, &on_device, &tangents).expect("device quadratic");
            let band = (widest + 2) as f64 * 2f64.powi(-24) * quadratic.abs();
            assert!((device_quadratic - quadratic).abs() <= band, "{name}: curvature {device_quadratic} against {quadratic}");
        }
    }
}

/// Selection on a lowered program keeps or refuses by the same float64 KL: lowered onto the host
/// reference device it selects the masks the program selects unlowered, and the KL it returns is
/// the CPU forward's on them.
#[test]
fn selection_on_a_lowered_program_selects_the_unlowered_masks() {
    let (unlowered, base, masks, clean) = masked_fixture();
    let (lowered, _, _, _) = masked_fixture();
    lowered.lower_on(&gam_gpu::tensor::Device::host(), Arithmetic::F64).expect("lowered");
    for target in targets(&clean) {
        let coder = masked::Coder::ran(masked::listing_costs(&masks), base.rows);
        let (reference, _) = masked::select(&unlowered, &base, &target, masks.clone(), &coder, 1.0, 2).expect("unlowered selection");
        let (selected, kl) = masked::select(&lowered, &base, &target, masks.clone(), &coder, 1.0, 2).expect("lowered selection");
        assert_eq!(selected, reference, "the lowered selection's masks");
        assert_ne!(selected, masks, "the selection moved");
        let (cpu, _, _) = masked::forward(&unlowered, &unlowered.family(&base, &selected), &target).expect("cpu forward");
        for r in 0..base.rows {
            assert!((kl[r] - cpu[r]).abs() <= 1e-9 * (1.0 + cpu[r].abs()), "row {r}: {} against {}", kl[r], cpu[r]);
        }
    }
}

/// Every keep/refuse a selection takes with screened heads (module note of `masked`, "Screened
/// heads": f32 head products on the Apple GPU where it resolves, float64 elsewhere) is the one the
/// float64 codes of the same masks take: a 2048-class head over 256 rows clears the device's size
/// floors, and each round's decision per sequence is checked against float64 forwards of both
/// masks.
#[test]
fn screened_selection_decisions_are_the_float64_ones() {
    use super::device_program_tests::fixture_sized;
    use super::masked::{Context, Round, code, previous_inputs, score_only, select_observed};
    use super::operator_program::SlotValues;
    let (program, family) = fixture_sized(128, 2048, 64, 4);
    let all = sites(&program);
    let libraries: Vec<Library> = all
        .iter()
        .enumerate()
        .map(|(k, site)| {
            let (d_out, d_in) = matrix(&program, site).expect("map").dim();
            Library {
                v: Array2::from_shape_fn((PIECES, d_in), |(i, j)| 0.3 * noise(10_000 * k + 37 * i + j)),
                u: Array2::from_shape_fn((PIECES, d_out), |(i, j)| 0.3 * noise(10_000 * k + 5000 + 37 * i + j)),
                mean: Array1::zeros(d_in),
            }
        })
        .collect();
    let clean = program.execute(&family, false).expect("clean").values[program.output].clone();
    let masked = Masked::build(&program, all, libraries).expect("masked");
    let target = Target::every_row(clean);
    let masks: Vec<Array2<f64>> = masked.all_pieces().iter().map(|p| Array2::ones((family.rows, *p))).collect();
    let coder = Context::new(&masked.all_pieces()).coder(previous_inputs(&family));
    let observations = 64.0;
    let masks_of = |inputs: &FamilyInputs| -> Vec<Array2<f64>> {
        masked.slots.iter().map(|&slot| match &inputs.slots[slot] {
            SlotValues::Raw(m) => m.clone(),
            SlotValues::Tokens(_) => Array2::zeros((0, 0)),
        }).collect()
    };
    let mut decisions = 0;
    let mut observe = |round: &Round<'_>| -> Result<(), String> {
        let exact = |inputs: &FamilyInputs| -> Result<Array1<f64>, String> {
            Ok(code(&score_only(&masked, inputs, &target)?, &coder.bits(&masks_of(inputs)), observations))
        };
        let (before, after) = (exact(round.current)?, exact(round.proposed)?);
        let sequences = round.sequence_of.iter().copied().max().map_or(0, |m| m + 1);
        for q in 0..sequences {
            let members: Vec<usize> = (0..round.flipped.len()).filter(|r| round.sequence_of[*r] == q).collect();
            if members.iter().all(|r| round.flipped[*r] == 0) {
                continue;
            }
            let used: f64 = members.iter().map(|r| round.before[*r] - round.after[*r]).sum();
            let truth: f64 = members.iter().map(|r| before[*r] - after[*r]).sum();
            // Ties within float64 rounding of the codes are no decision.
            if truth.abs() > 1e-9 * before.sum().abs() {
                assert_eq!(used > 0.0, truth > 0.0, "sequence {q}: screened saving {used}, float64 {truth}");
                decisions += 1;
            }
        }
        Ok(())
    };
    select_observed(&masked, &family, &target, masks, &coder, observations, 2, &mut observe).expect("selection");
    assert!(decisions > 0, "the selection took decisions");
}

/// A selection whose trials score through screened heads returns the float64 KL of the masks it
/// returns, bit for bit what one forward of those masks alone gives: never a screened KL (whose
/// logits carry an f32 band), nor rows settled to float64 on a subset of the batch. Selection ends
/// both when every sequence is done and when no single flip nor group move pays; each start here
/// reaches one of them.
#[test]
fn a_screened_selection_returns_the_float64_kl_of_its_masks() {
    use super::device_program_tests::fixture_sized;
    use super::masked::{Coder, score_only, select};
    let (program, family) = fixture_sized(128, 2048, 64, 4);
    let all = sites(&program);
    let libraries: Vec<Library> = all
        .iter()
        .enumerate()
        .map(|(k, site)| {
            let (d_out, d_in) = matrix(&program, site).expect("map").dim();
            Library {
                v: Array2::from_shape_fn((PIECES, d_in), |(i, j)| 0.3 * noise(20_000 * k + 37 * i + j)),
                u: Array2::from_shape_fn((PIECES, d_out), |(i, j)| 0.3 * noise(20_000 * k + 7000 + 37 * i + j)),
                mean: Array1::zeros(d_in),
            }
        })
        .collect();
    let masked = Masked::build(&program, all, libraries).expect("masked");
    // The target is the masked program itself at a planted set: every piece on but the first of
    // each site, so a selection from everything on has something to find and then nothing.
    let planted: Vec<Array2<f64>> = masked.all_pieces().iter().map(|p| Array2::from_shape_fn((family.rows, *p), |(_, c)| if c == 0 { 0.0 } else { 1.0 })).collect();
    let target = Target::every_row(masked.program.execute(&masked.family(&family, &planted), false).expect("planted").values[masked.program.output].clone());
    let costs: Vec<Array1<f64>> = masked.all_pieces().iter().map(|p| Array1::from_elem(*p, 0.5)).collect();
    let coder = Coder::ran(costs, family.rows);
    let starts: [Vec<Array2<f64>>; 2] = [
        masked.all_pieces().iter().map(|p| Array2::ones((family.rows, *p))).collect(),
        planted.clone(),
    ];
    for begin in starts {
        let (masks, kl) = select(&masked, &family, &target, begin, &coder, 64.0, 2).expect("selection");
        let exact = score_only(&masked, &masked.family(&family, &masks), &target).expect("float64");
        assert_eq!(kl, exact, "the returned KL is not the float64 KL of the returned masks");
    }
}

/// A site whose gates are all on computes the same through its library's sum as through its
/// pieces (`Masked::dense_program`), and a suffix evaluated alone (`execute_suffix_node`) is the
/// whole suffix's value at that node.
#[test]
fn sites_all_on_run_through_their_library_sums() {
    let (masked, base, masks, _) = masked_fixture();
    let dense: Vec<bool> = (0..masked.sites.len()).map(|k| k % 2 == 0).collect();
    let gated: Vec<Array2<f64>> = masks.iter().enumerate().map(|(k, m)| if dense[k] { Array2::ones(m.dim()) } else { m.clone() }).collect();
    let family = masked.family(&base, &gated);
    let reference = masked.program.execute(&family, false).expect("pieces");
    let program = masked.dense_program(&dense).expect("sums");
    let summed = program.execute(&family, false).expect("sums execute");
    let output = masked.program.output;
    let scale = reference.values[output].iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let gap = (&summed.values[output] - &reference.values[output]).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(gap <= 1e-10 * scale, "library sums moved the output by {gap:e} of {scale:e}");
    let from = masked.z[masked.sites.len() - 1] - 1;
    let alone = masked.program.execute_suffix_node(&family, &reference, from, output).expect("suffix node");
    let whole = masked.program.execute_suffix(&family, &reference, from).expect("suffix");
    assert_eq!(alone, whole[output - from]);
}

/// A selection killed between any two rounds and resumed from the state it showed its checkpoint
/// there ends with bit-identical masks and KL, and passes through the same states, as the
/// uninterrupted one: here with screened heads, whose kept proposals' f32-banded forwards are
/// never carried into the next round.
#[test]
fn a_resumed_screened_selection_is_the_uninterrupted_one() {
    use super::device_program_tests::fixture_sized;
    use super::masked::{Coder, Progress, Resume, Round, select_resumable};
    let (program, family) = fixture_sized(128, 2048, 64, 4);
    let all = sites(&program);
    let libraries: Vec<Library> = all
        .iter()
        .enumerate()
        .map(|(k, site)| {
            let (d_out, d_in) = matrix(&program, site).expect("map").dim();
            Library {
                v: Array2::from_shape_fn((PIECES, d_in), |(i, j)| 0.3 * noise(30_000 * k + 37 * i + j)),
                u: Array2::from_shape_fn((PIECES, d_out), |(i, j)| 0.3 * noise(30_000 * k + 9000 + 37 * i + j)),
                mean: Array1::zeros(d_in),
            }
        })
        .collect();
    let masked = Masked::build(&program, all, libraries).expect("masked");
    let target = Target::every_row(program.execute(&family, false).expect("clean").values[program.output].clone());
    let costs: Vec<Array1<f64>> = masked.all_pieces().iter().map(|p| Array1::from_elem(*p, 2.0)).collect();
    let coder = Coder::ran(costs, family.rows);
    let start: Vec<Array2<f64>> = masked.all_pieces().iter().map(|p| Array2::ones((family.rows, *p))).collect();
    let run = |resume: Option<Resume>| -> (Vec<Array2<f64>>, Array1<f64>, Vec<Resume>) {
        let mut states = Vec::new();
        let mut checkpoint = |progress: &Progress<'_>| -> Result<(), String> {
            states.push(progress.to_resume());
            Ok(())
        };
        let (masks, kl) = select_resumable(&masked, &family, &target, start.clone(), resume, &coder, 64.0, 2, &mut |_: &Round<'_>| Ok(()), &mut checkpoint)
            .expect("selection");
        (masks, kl, states)
    };
    let (masks, kl, states) = run(None);
    assert!(states.len() >= 3, "the selection took too few rounds to resume mid-way: {}", states.len());
    for k in [1, states.len() / 2, states.len() - 1] {
        let (resumed_masks, resumed_kl, resumed_states) = run(Some(states[k].clone()));
        assert_eq!(resumed_masks, masks, "resumed at round {k}: other masks");
        assert_eq!(resumed_kl, kl, "resumed at round {k}: another KL");
        assert_eq!(resumed_states, states[k..], "resumed at round {k}: other states on the way");
    }
}

/// On a lowered program the adversary climbs every box's every start as one batch of sequences
/// and finds what each box's own run finds on the CPU.
#[test]
fn a_batched_adversary_on_a_device_finds_each_boxs_points() {
    use super::adversary::{Gates, adversary, adversary_batch};
    let (unlowered, base, masks, clean) = masked_fixture();
    let (lowered, _, _, _) = masked_fixture();
    lowered.lower_on(&gam_gpu::tensor::Device::host(), Arithmetic::F64).expect("lowered");
    let claim = Gates::claim(&masks);
    let mut first = claim.clone();
    for k in 1..masks.len() {
        first.upper[k] = first.lower[k].clone();
    }
    let boxes = [claim, first];
    let seeds = [5u64, 9];
    for target in targets(&clean) {
        let batched = adversary_batch(&lowered, &base, &target, &boxes, 3, 4, &seeds).expect("batched");
        for (b, gates) in boxes.iter().enumerate() {
            // One box at a time on the same device: the same points, the same values.
            assert_eq!(batched[b], adversary(&lowered, &base, &target, gates, None, 3, 4, seeds[b]).expect("on the device"), "box {b}");
            let alone = adversary(&unlowered, &base, &target, gates, None, 3, 4, seeds[b]).expect("alone");
            for r in 0..base.rows {
                assert!((batched[b][r] - alone[r]).abs() <= 1e-9 * (1.0 + alone[r].abs()), "box {b} row {r}: {} against {}", batched[b][r], alone[r]);
            }
        }
    }
}

/// A forward whose sites' `z` are evaluated at their masks' nonzeros alone gives the whole
/// forward's logits within its float64 bands, and every value no `z` feeds unchanged.
#[test]
fn a_gated_forward_scores_as_the_whole_forward() {
    let (masked, base, _, _) = masked_fixture();
    let masks: Vec<Array2<f64>> = masked
        .all_pieces()
        .iter()
        .enumerate()
        .map(|(k, p)| Array2::from_shape_fn((base.rows, *p), |(r, c)| if noise(900 * k + 13 * r + c) > 0.5 { 1.0 } else { 0.0 }))
        .collect();
    let family = masked.family(&base, &masks);
    let whole = masked.program.execute(&family, true).expect("whole");
    let gated = masked.program.execute_gated(&family, &masked.gates()).expect("gated");
    assert_eq!(masked.gates().len(), masked.sites.len());
    let output = masked.program.output;
    let (bands, balls) = (whole.bands.as_ref().expect("bands"), whole.balls.as_ref().expect("balls"));
    for ((r, c), v) in gated.values[output].indexed_iter() {
        let band = bands[output][[r, c]] + balls[output][r];
        assert!((v - whole.values[output][[r, c]]).abs() <= 2.0 * band, "logit ({r},{c})");
    }
    for (z, mask) in masked.gates() {
        for ((r, c), m) in gated.values[mask].indexed_iter() {
            if *m == 0.0 {
                assert_eq!(gated.values[z][[r, c]], 0.0);
            }
        }
    }
}

/// On the Apple GPU (`masked_device`'s module note: f32 throughout, its twin never decides) the
/// masked forward and its reverse pass agree with the CPU's within the f32 proposal band, `(w + 2)
/// 2⁻²⁴` of the largest entry (`device_program_tests`), the one the CUDA path's f32 proposals are
/// held to; the logits per row of their largest entry, and the KL within twice the logits' band
/// (its gradient in the logits has `ℓ₁` norm at most 2) plus the f32 evaluation's own rounding. The
/// CPU's float64 bands scaled to f32 (first order in the unit roundoff, `2⁻⁵³` there and `2⁻²⁴`
/// here, with each product's operand rounding at most as many units again: `2 · 2²⁹` times band
/// plus ball) are a rigorous but much wider bound on the logits, checked as well.
#[test]
fn the_masked_program_on_the_apple_gpu_matches_the_cpu_in_f32_bands() {
    use gam_gpu::tensor::Device;
    let Some(metal) = Device::single_precision(gam_gpu::GpuPolicy::Auto).expect("a probe that does not fault").filter(|d| !d.float64()) else { return };
    const SCALE: f64 = 2.0 * (1u64 << 29) as f64;
    const UNIT: f64 = 1.0 / 16_777_216.0;
    let (masked, base, masks, clean) = masked_fixture();
    let family = masked.family(&base, &masks);
    let banded = masked.program.execute(&family, true).expect("banded");
    let output = masked.program.output;
    let (bands, balls) = (&banded.bands.as_ref().expect("bands")[output], &banded.balls.as_ref().expect("balls")[output]);
    assert!(Accelerated::new(&metal, &masked, Arithmetic::F64).is_err(), "float64 proposals are refused");
    let accelerated = Accelerated::new(&metal, &masked, Arithmetic::F32).expect("lowered");
    assert!(!accelerated.decides());
    let widest = 16;
    let proposal = (widest + 2) as f64 * UNIT;
    for target in targets(&clean) {
        let (kl, trace, cotangent) = masked::forward(&masked, &family, &target).expect("cpu forward");
        let mask_gradients = masked::mask_gradients(&masked, &family, &trace, cotangent.clone()).expect("cpu mask gradients");
        let gradients = masked::gradients(&masked, &family, &trace, &masks, cotangent).expect("cpu gradients");
        let on_device = accelerated.target(&target).expect("target");
        let state = accelerated.forward(&family, &on_device).expect("metal forward");
        let logits = accelerated.program().logits(&state.trace, 0, family.rows).expect("logits");
        let cpu = &trace.values[output];
        for r in 0..family.rows {
            let size = cpu.row(r).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for c in 0..cpu.ncols() {
                let (v, w) = (logits[[r, c]], cpu[[r, c]]);
                assert!((v - w).abs() <= SCALE * (bands[[r, c]] + balls[r]), "logit ({r},{c}) outside the scaled float64 band");
                assert!((v - w).abs() <= proposal * size, "logit ({r},{c}): {v} against {w} (band {:e})", proposal * size);
            }
            if !target.scores(r) {
                assert_eq!(state.kl[r], 0.0, "an unscored row has no KL");
                continue;
            }
            let (p, q) = (softmax(target.logits.row(r)), softmax(cpu.row(r)));
            let zmax = target.logits.row(r).iter().fold(size, |m, v| m.max(v.abs()));
            let terms: f64 = p.iter().zip(q.iter()).map(|(a, b)| a * (a.ln().abs() + b.ln().abs() + 2.0 * zmax + 1.0)).sum();
            let n = (clean.ncols() + 16) as f64;
            let band = 2.0 * proposal * size + (4.0 * UNIT * zmax + n * UNIT / (1.0 - n * UNIT)) * terms;
            assert!((state.kl[r] - kl[r]).abs() <= band, "KL of row {r}: {} against {} (band {band:e})", state.kl[r], kl[r]);
        }
        for (k, g) in accelerated.mask_gradients(&masked, &state).expect("metal mask gradients").iter().enumerate() {
            assert_proposal(&format!("Metal mask gradient of site {k}"), g, &mask_gradients[k], widest);
        }
        for (k, (m, v, u)) in accelerated.gradients(&masked, &state, &masks).expect("metal gradients").iter().enumerate() {
            assert_proposal(&format!("Metal mask gradient of site {k}"), m, &gradients[k].0, widest);
            assert_proposal(&format!("Metal V gradient of site {k}"), v, &gradients[k].1, family.rows);
            assert_proposal(&format!("Metal U gradient of site {k}"), u, &gradients[k].2, family.rows);
        }
    }
}
