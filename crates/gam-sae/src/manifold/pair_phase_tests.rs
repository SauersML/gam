// Tests for the F4 phase-coupling screen + App D phase circuits. Included from
// `pair_phase.rs` via `include!` so the helpers below share its private items.

use super::*;
use ndarray::Array2;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-12);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Axis-aligned 2-plane candidate on ambient dims `(d0, d1)`, active on `active`.
fn axis_candidate(p: usize, d0: usize, d1: usize, active: &[bool]) -> IsaPlaneCandidate {
    let n = active.len();
    let mut basis = Array2::<f64>::zeros((p, 2));
    basis[[d0, 0]] = 1.0;
    basis[[d1, 1]] = 1.0;
    let gate_logits: Vec<f64> = active
        .iter()
        .map(|&a| if a { 0.0 } else { f64::NEG_INFINITY })
        .collect();
    IsaPlaneCandidate {
        basis,
        amplitudes: [1.0, 1.0],
        phases_turns: Array2::<f64>::zeros((n, 1)),
        gate_logits,
        kappa: 1.0,
        q_hat: active.iter().filter(|&&a| a).count() as f64 / n as f64,
    }
}

const NOISE: f64 = 0.02;

/// Two DENSE circles with a rotation phase law `θ_B = θ_A + φ`. Energy `ρ ≈ 1`
/// (the pair_kappa screen is blind); the phase screen's Difference1 channel must
/// FIRE. This is report F4 case 3 at zero reconstruction cost.
#[test]
fn torus_phase_law_fires_difference1() {
    let mut s = 0x5EED_u64;
    let n = 2000usize;
    let p = 8usize;
    let phi = 0.8_f64;
    let mut data = Array2::<f64>::zeros((n, p));
    let active = vec![true; n];
    for i in 0..n {
        let ta = std::f64::consts::TAU * lcg(&mut s);
        let tb = ta + phi + 0.06 * lcg_normal(&mut s);
        data[[i, 0]] += ta.cos();
        data[[i, 1]] += ta.sin();
        data[[i, 2]] += tb.cos();
        data[[i, 3]] += tb.sin();
        for j in 0..p {
            data[[i, j]] += NOISE * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    let ca = axis_candidate(p, 0, 1, &active);
    let cb = axis_candidate(p, 2, 3, &active);
    let v = screen_pair_phase(data.view(), &mean, 0, 1, &ca, &cb, 60, 0xA1).unwrap();
    let d1 = &v.channels[0];
    eprintln!(
        "[torus phase-law] T1={:.4} null_mean={:.4} z={:.2} p={:.4} e={:.2} best={:?}",
        d1.resultant, d1.null_mean, d1.z, d1.p_value, d1.e_value, v.best_channel
    );
    assert_eq!(d1.channel, PhaseChannel::Difference1);
    assert!(
        d1.resultant > 0.8,
        "rigid θ_B=θ_A+φ must give T1≈1; got {:.4}",
        d1.resultant
    );
    assert!(
        d1.p_value <= 2.0 / 61.0,
        "phase law must fire near the exact-null floor; p={:.4}",
        d1.p_value
    );
    assert_eq!(
        v.best_channel,
        PhaseChannel::Difference1,
        "the rotation channel must be the strongest"
    );
}

/// Two INDEPENDENT dense circles: no phase law. The screen must be SILENT — the
/// exact Monte-Carlo p-value on the Difference1 channel is calibrated (uniform)
/// under the phase-randomised null, so the 0.05-level rejection rate over many
/// independent draws sits near 0.05, not inflated.
#[test]
fn independent_circles_silent_and_calibrated() {
    let trials = 40usize;
    let b = 40usize;
    let n = 1200usize;
    let p = 8usize;
    let mut reject_at_05 = 0usize;
    let mut p_sum = 0.0_f64;
    for t in 0..trials {
        let mut s = 0x1000_u64 + t as u64 * 0x9E37;
        let mut data = Array2::<f64>::zeros((n, p));
        let active = vec![true; n];
        for i in 0..n {
            let ta = std::f64::consts::TAU * lcg(&mut s);
            let tb = std::f64::consts::TAU * lcg(&mut s); // independent
            data[[i, 0]] += ta.cos();
            data[[i, 1]] += ta.sin();
            data[[i, 2]] += tb.cos();
            data[[i, 3]] += tb.sin();
            for j in 0..p {
                data[[i, j]] += NOISE * lcg_normal(&mut s);
            }
        }
        let mean = Array1::<f64>::zeros(p);
        let ca = axis_candidate(p, 0, 1, &active);
        let cb = axis_candidate(p, 2, 3, &active);
        let v = screen_pair_phase(data.view(), &mean, 0, 1, &ca, &cb, b, 0x50 + t as u64).unwrap();
        let p1 = v.channels[0].p_value;
        p_sum += p1;
        if p1 < 0.05 {
            reject_at_05 += 1;
        }
    }
    let fpr = reject_at_05 as f64 / trials as f64;
    let mean_p = p_sum / trials as f64;
    eprintln!("[independent circles] FPR@0.05={fpr:.3} mean_p={mean_p:.3} ({trials} trials)");
    // Calibrated null: the empirical FPR must not be materially inflated above the
    // nominal 0.05. With 40 trials the estimate is noisy (SE≈0.03), so bound it
    // generously while still catching gross inflation, and require the mean p-value
    // to sit near the uniform-null 0.5.
    assert!(fpr <= 0.20, "independent-circle FPR@0.05 inflated: {fpr:.3}");
    assert!(
        (0.30..=0.70).contains(&mean_p),
        "Difference1 p under the null must be ~uniform (mean≈0.5); got {mean_p:.3}"
    );
}

/// ANTIPODAL / diameter coupling: `θ_B = θ_A + Bernoulli·π` (B is A or its
/// antipode). The difference is `{0, −π}`; the h=1 channel cancels but the h=2
/// (diameter) channel is invariant to the π flip and FIRES.
#[test]
fn antipodal_coupling_fires_difference2() {
    let mut s = 0xD1A_u64;
    let n = 2000usize;
    let p = 8usize;
    let mut data = Array2::<f64>::zeros((n, p));
    let active = vec![true; n];
    for i in 0..n {
        let ta = std::f64::consts::TAU * lcg(&mut s);
        let flip = if lcg(&mut s) < 0.5 { std::f64::consts::PI } else { 0.0 };
        let tb = ta + flip + 0.05 * lcg_normal(&mut s);
        data[[i, 0]] += ta.cos();
        data[[i, 1]] += ta.sin();
        data[[i, 2]] += tb.cos();
        data[[i, 3]] += tb.sin();
        for j in 0..p {
            data[[i, j]] += NOISE * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    let ca = axis_candidate(p, 0, 1, &active);
    let cb = axis_candidate(p, 2, 3, &active);
    let v = screen_pair_phase(data.view(), &mean, 0, 1, &ca, &cb, 60, 0xB2).unwrap();
    let d1 = &v.channels[0];
    let d2 = &v.channels[1];
    eprintln!(
        "[antipodal] T1={:.4}(p={:.4}) T2={:.4}(p={:.4})",
        d1.resultant, d1.p_value, d2.resultant, d2.p_value
    );
    assert!(d1.resultant < 0.3, "h=1 must cancel on antipodal; T1={:.4}", d1.resultant);
    assert!(d2.resultant > 0.7, "h=2 diameter channel must lock; T2={:.4}", d2.resultant);
    assert!(d2.p_value <= 2.0 / 61.0, "diameter coupling must fire; p={:.4}", d2.p_value);
}

/// ORIENTATION-REVERSING coupling: `θ_B = −θ_A + φ`. The difference `2θ_A − φ`
/// winds (h=1,2 silent); the SUM `θ_A + θ_B = φ` is constant ⇒ the Sum1 channel
/// FIRES. Only the phase-sum channel can see a mirror law.
#[test]
fn reversal_coupling_fires_sum1() {
    let mut s = 0x3EE_u64;
    let n = 2000usize;
    let p = 8usize;
    let phi = 1.2_f64;
    let mut data = Array2::<f64>::zeros((n, p));
    let active = vec![true; n];
    for i in 0..n {
        let ta = std::f64::consts::TAU * lcg(&mut s);
        let tb = -ta + phi + 0.06 * lcg_normal(&mut s);
        data[[i, 0]] += ta.cos();
        data[[i, 1]] += ta.sin();
        data[[i, 2]] += tb.cos();
        data[[i, 3]] += tb.sin();
        for j in 0..p {
            data[[i, j]] += NOISE * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    let ca = axis_candidate(p, 0, 1, &active);
    let cb = axis_candidate(p, 2, 3, &active);
    let v = screen_pair_phase(data.view(), &mean, 0, 1, &ca, &cb, 60, 0xC3).unwrap();
    let d1 = &v.channels[0];
    let sum = &v.channels[2];
    eprintln!(
        "[reversal] T1={:.4}(p={:.4}) Tsum={:.4}(p={:.4}) best={:?}",
        d1.resultant, d1.p_value, sum.resultant, sum.p_value, v.best_channel
    );
    assert_eq!(sum.channel, PhaseChannel::Sum1);
    assert!(d1.resultant < 0.3, "difference channel must be blind to a mirror; T1={:.4}", d1.resultant);
    assert!(sum.resultant > 0.8, "phase-sum channel must lock on reversal; Tsum={:.4}", sum.resultant);
    assert!(sum.p_value <= 2.0 / 61.0, "reversal must fire the sum channel; p={:.4}", sum.p_value);
    assert_eq!(v.best_channel, PhaseChannel::Sum1);
}

/// The e-BH ledger over three atoms: an independent circle A plus a phase-locked
/// B–C torus. Only the {B,C} pair may be `torus_proposed`.
#[test]
fn screen_all_pairs_ebh_selects_only_locked_pair() {
    let mut s = 0xF00D_u64;
    let n = 1500usize;
    let p = 12usize;
    let phi = 0.5_f64;
    let mut data = Array2::<f64>::zeros((n, p));
    let active = vec![true; n];
    for i in 0..n {
        let ta = std::f64::consts::TAU * lcg(&mut s); // A: independent
        let tbc = std::f64::consts::TAU * lcg(&mut s); // B/C base angle
        data[[i, 0]] += ta.cos();
        data[[i, 1]] += ta.sin();
        data[[i, 2]] += tbc.cos();
        data[[i, 3]] += tbc.sin();
        let tc = tbc + phi + 0.06 * lcg_normal(&mut s); // C phase-locked to B
        data[[i, 4]] += tc.cos();
        data[[i, 5]] += tc.sin();
        for j in 0..p {
            data[[i, j]] += NOISE * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    let cands = vec![
        axis_candidate(p, 0, 1, &active), // 0: independent A
        axis_candidate(p, 2, 3, &active), // 1: torus factor B
        axis_candidate(p, 4, 5, &active), // 2: torus factor C
    ];
    let verdicts = screen_all_pairs_phase(data.view(), &mean, &cands, 50, 0xEB, 0.05).unwrap();
    let proposed: Vec<(usize, usize)> = verdicts
        .iter()
        .filter(|v| v.torus_proposed)
        .map(|v| (v.atom_a, v.atom_b))
        .collect();
    eprintln!("[e-BH ledger] torus_proposed pairs = {proposed:?}");
    assert_eq!(proposed, vec![(1, 2)], "only the phase-locked B–C pair may be proposed");

    // The proposal producer emits exactly the {1,2} binding Fusion move.
    let moves = phase_fusion_moves(data.view(), &mean, &cands, 50, 0xEB, 0.05).unwrap();
    eprintln!("[e-BH ledger] fusion moves = {moves:?}");
    assert!(
        moves
            .iter()
            .any(|m| matches!(m, gam_solve::structure_search::StructureMove::Fusion { a: 1, b: 2 })),
        "the phase-locked pair must produce a binding Fusion move"
    );
}

/// FUSE-RACE (case 1): a single circle split across two dense frames. Frame A owns
/// dim 0, frame B owns dim 1 — complementary diameters `r_A²+r_B²≈1`. The phase
/// screen flags `fuse_race_proposed`, and the fused union 2-plane captures nearly
/// all the energy (the circle IS 2-D).
#[test]
fn fuse_race_flags_split_single_circle() {
    let mut s = 0x5F0_u64;
    let n = 2000usize;
    let p = 8usize;
    let mut data = Array2::<f64>::zeros((n, p));
    let active = vec![true; n];
    for i in 0..n {
        let th = std::f64::consts::TAU * lcg(&mut s);
        data[[i, 0]] += th.cos();
        data[[i, 1]] += th.sin();
        for j in 0..p {
            data[[i, j]] += NOISE * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    // Frame A = (dim0, noise dim2); frame B = (dim1, noise dim3): each sees one
    // diameter of the single circle in dims (0,1).
    let ca = axis_candidate(p, 0, 2, &active);
    let cb = axis_candidate(p, 1, 3, &active);
    let v = screen_pair_phase(data.view(), &mean, 0, 1, &ca, &cb, 30, 0xF5).unwrap();
    eprintln!(
        "[fuse-race] rho={:.3} total_cv={:.3} fuse_race={}",
        v.energy_rho, v.total_energy_cv, v.fuse_race_proposed
    );
    assert!(v.fuse_race_proposed, "split single circle must trigger the fuse-race");
    let fused = fuse_race_candidate(data.view(), &mean, 0, 1, &ca, &cb).unwrap();
    eprintln!(
        "[fuse-race] support={:?} captured_energy={:.3}",
        fused.support_columns, fused.captured_energy_fraction
    );
    assert!(
        fused.captured_energy_fraction > 0.9,
        "the fused 2-plane must capture the whole circle; got {:.3}",
        fused.captured_energy_fraction
    );
}

/// A PHASE CIRCUIT (App D): with the generative law `θ_B = θ_A + φ`, the fitted
/// transfer operator recovers a rotation by φ, and an intervention shard (steer
/// `θ_A += Δ`) produces an observed `Δθ_B` that tracks the prediction with slope
/// ≈ 1 ⇒ CERTIFIED. Independent circles yield no faithful transfer ⇒ NOT certified.
#[test]
fn phase_circuit_certifies_rotation_and_rejects_independence() {
    let mut s = 0xC112_u64;
    let n = 1500usize;
    let phi = 0.7_f64;
    let (mut ta, mut tb, mut tb_indep) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let a = std::f64::consts::TAU * lcg(&mut s);
        ta.push(a);
        tb.push(a + phi + 0.05 * lcg_normal(&mut s));
        tb_indep.push(std::f64::consts::TAU * lcg(&mut s));
    }
    let w = vec![1.0_f64; n];

    // Transfer operator + polar angle recovers the rotation φ.
    let op = phase_transfer_operator(&ta, &tb, &w).unwrap();
    let ang = so2_polar_angle(op.view()).unwrap();
    eprintln!("[phase circuit] transfer angle={ang:.4} (want φ={phi:.4})");
    assert!((wrap_pi(ang - phi)).abs() < 0.1, "transfer angle must recover φ; got {ang:.4}");

    // Intervention shard: steer θ_A by a dose Δ, predict Δθ_B through the operator,
    // observe Δθ_B through the true law (steering A physically moves B by Δ).
    let deltas = [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0];
    let (mut pred, mut obs) = (Vec::new(), Vec::new());
    for i in 0..n {
        let d = deltas[i % deltas.len()];
        let base = predicted_theta_b(op.view(), ta[i]);
        let steered = predicted_theta_b(op.view(), ta[i] + d);
        pred.push(wrap_pi(steered - base));
        obs.push(d); // ground-truth response of θ_B to steering θ_A under θ_B=θ_A+φ
    }
    let cert = certify_phase_circuit(&ta, &tb, &w, &pred, &obs).unwrap();
    eprintln!(
        "[phase circuit] orient={} transport_defect={:.3} dose_slope={:.3} r2={:.3} certified={}",
        cert.orientation, cert.transport_defect, cert.dose_slope, cert.dose_r2, cert.certified
    );
    assert!(cert.certified, "faithful rotation circuit must certify");
    assert_eq!(cert.orientation, 1, "rotation is orientation-preserving");
    assert!((cert.dose_slope - 1.0).abs() < 0.2, "dose slope must be ≈1; got {:.3}", cert.dose_slope);

    // Independent B: operator collapses, no faithful transfer ⇒ not certified.
    let op0 = phase_transfer_operator(&ta, &tb_indep, &w);
    let (mut pred0, mut obs0) = (Vec::new(), Vec::new());
    if let Ok(ref opm) = op0 {
        for i in 0..n {
            let d = deltas[i % deltas.len()];
            let base = predicted_theta_b(opm.view(), ta[i]);
            let steered = predicted_theta_b(opm.view(), ta[i] + d);
            pred0.push(wrap_pi(steered - base));
            obs0.push(0.0); // steering A does not move an independent B
        }
    }
    let cert0 = certify_phase_circuit(&ta, &tb_indep, &w, &pred0, &obs0).unwrap();
    eprintln!(
        "[phase circuit indep] orient={} transport_defect={:.3} dose_slope={:.3} r2={:.3} certified={}",
        cert0.orientation, cert0.transport_defect, cert0.dose_slope, cert0.dose_r2, cert0.certified
    );
    assert!(!cert0.certified, "independent circles must NOT certify a phase circuit");
}

/// e-BH unit check: with a single huge e-value in a family of nulls (e≈1), only the
/// large one is rejected at α=0.05.
#[test]
fn ebh_rejects_dominant_e_value() {
    let mut es = vec![1.0_f64; 20];
    es[7] = 500.0;
    let rej = ebh_reject(&es, 0.05);
    assert_eq!(rej, vec![7], "only the dominant e-value clears m/(αk)");
    // No discoveries when nothing dominates.
    let flat = vec![1.0_f64; 20];
    assert!(ebh_reject(&flat, 0.05).is_empty());
}
