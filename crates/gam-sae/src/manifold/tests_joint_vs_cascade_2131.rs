//! JOINT-vs-CASCADE: what the cheap pairwise-κ ENERGY screen catches, and what
//! it cannot — the statistical half of the "blocks are linear objects; in-block
//! refinement is suboptimal under joint dependencies" investigation (#2131).
//!
//! The pairwise screen ([`super::pair_kappa::screen_pair`]) adjudicates a pair of
//! accepted atoms on the NORMALISED ENERGY CROSS-MOMENT
//! `ρ = E[r_A²·r_B²] / (E[r_A²]·E[r_B²])`, firing a MERGE only on POSITIVE binding
//! evidence `ρ > 1` (shared presence gate). That is a deliberate, sharp design:
//! `ρ = 1` is the independence null. These tests pin down, on the REAL shipped
//! screen and at frontier ambient width, the THREE distinct joint dependencies a
//! cascade can split across frames and which tail each lands in:
//!
//!   1. ONE circle whose 2-plane is SPLIT across two dense frames (each atom sees
//!      one diameter). The per-row energies are COMPLEMENTARY (`r_A²+r_B² = 1`),
//!      an ANTI-correlation ⇒ `ρ ≈ 1/2 < 1`. The presence-binding screen does NOT
//!      fire (it fires only on `ρ > 1`): the fragmentation of a single curved set
//!      into two linear frames lives in the LOWER tail, which the merge screen —
//!      by design — does not adjudicate. A DOCUMENTED GAP the terminal joint fit,
//!      not the screen, must close.
//!   2. Two circles, co-gated SHARED presence (a gated torus), independent angles.
//!      `ρ = 1/q > 1` ⇒ the screen FIRES. The screen's home tail.
//!   3. Two DENSE circles (`q = 1`) with CORRELATED phases (a torus density
//!      concentrated on the diagonal). Presence is constant, so each `r² ≡ 1`;
//!      the energy cross-moment is blind to the phase law ⇒ `ρ ≈ 1`, NO fire. The
//!      joint DENSITY (the interpretation) is invisible to a second-order ENERGY
//!      screen even though it is a genuine inter-atom dependence — recoverable
//!      only by a joint 2-D coordinate, at ZERO reconstruction cost (marginals
//!      already give full EV). The second documented gap.
//!
//! Together: the cheap screen catches exactly ONE of the three joint-dependence
//! regimes (shared-presence binding). The other two — energy complementarity of a
//! split single chart, and a phase law at dense presence — are structurally
//! outside an energy-cross-moment screen and are the province of the terminal
//! joint fit. Scale (`p ∈ {512, 2048}`) confirms the ρ anchors are ambient-width
//! invariant, so the claim is not a small-p artifact.

use super::isa_seed::IsaPlaneCandidate;
use super::pair_kappa::screen_pair;
use ndarray::{Array1, Array2};

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

/// A 2-plane candidate spanning ambient dims `(d0, d1)`, active on `active`.
/// Only `basis` and `gate_logits` are read by the screen; the rest are anchors.
fn plane_candidate(p: usize, d0: usize, d1: usize, active: &[bool]) -> IsaPlaneCandidate {
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

/// EXPERIMENT 1 — a single circle whose 2-plane is SPLIT across two dense frames.
/// The circle lives in ambient dims (0,1); frame A owns dim 0 (+ a noise dim 2),
/// frame B owns dim 1 (+ a noise dim 3). Each atom captures one diameter of the
/// circle, so `r_A² = cos²θ`, `r_B² = sin²θ` — COMPLEMENTARY, always co-present.
/// Anchor: `ρ = E[cos²θ sin²θ]/(E[cos²θ]E[sin²θ]) = (1/8)/(1/4) = 1/2`. The
/// merge screen (ρ>1 only) does NOT fire — the split lives in the lower tail.
fn split_single_circle_rho(p: usize, seed: u64) -> (f64, f64, bool) {
    let mut s = seed;
    let n = 6000usize;
    let mut data = Array2::<f64>::zeros((n, p));
    let active = vec![true; n]; // dense: the single circle is always present
    for i in 0..n {
        let th = std::f64::consts::TAU * lcg(&mut s);
        data[[i, 0]] += th.cos();
        data[[i, 1]] += th.sin();
        for j in 0..p {
            data[[i, j]] += 0.02 * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    // Frame A = span(dim0, noise dim2); frame B = span(dim1, noise dim3).
    let ca = plane_candidate(p, 0, 2, &active);
    let cb = plane_candidate(p, 1, 3, &active);
    let v = screen_pair(data.view(), &mean, 0, 1, &ca, &cb);
    (v.rho, v.z, v.merge_proposed)
}

/// EXPERIMENT 3 (screen half) — a gated torus split into two atoms: SHARED
/// presence gate `q`, independent angles. Anchor `ρ = 1/q`. The screen FIRES.
fn gated_torus_rho(p: usize, q: f64, seed: u64) -> (f64, f64, bool) {
    let mut s = seed;
    let n = 6000usize;
    let mut data = Array2::<f64>::zeros((n, p));
    let mut act = vec![false; n];
    for i in 0..n {
        if lcg(&mut s) < q {
            act[i] = true;
            let ta = std::f64::consts::TAU * lcg(&mut s);
            let tb = std::f64::consts::TAU * lcg(&mut s);
            data[[i, 0]] += ta.cos();
            data[[i, 1]] += ta.sin();
            data[[i, 2]] += tb.cos();
            data[[i, 3]] += tb.sin();
        }
        for j in 0..p {
            data[[i, j]] += 0.02 * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    let ca = plane_candidate(p, 0, 1, &act);
    let cb = plane_candidate(p, 2, 3, &act);
    let v = screen_pair(data.view(), &mean, 0, 1, &ca, &cb);
    (v.rho, v.z, v.merge_proposed)
}

/// EXPERIMENT 2 (screen half) — two DENSE circles (`q = 1`) with CORRELATED
/// phases: `θ_B = θ_A + small jitter` (a torus density on the diagonal). Presence
/// is constant, so `r_A² ≡ r_B² ≡ 1`; the energy cross-moment cannot see the
/// phase law ⇒ `ρ ≈ 1`, NO fire. The dependence is real but invisible to an
/// energy screen — it is a JOINT-DENSITY fact, recoverable only by a 2-D
/// coordinate, and at zero EV cost (each marginal circle is already captured).
fn phase_correlated_dense_rho(p: usize, seed: u64) -> (f64, f64, bool) {
    let mut s = seed;
    let n = 6000usize;
    let mut data = Array2::<f64>::zeros((n, p));
    let active = vec![true; n];
    for i in 0..n {
        let ta = std::f64::consts::TAU * lcg(&mut s);
        let tb = ta + 0.10 * lcg_normal(&mut s); // tightly phase-locked
        data[[i, 0]] += ta.cos();
        data[[i, 1]] += ta.sin();
        data[[i, 2]] += tb.cos();
        data[[i, 3]] += tb.sin();
        for j in 0..p {
            data[[i, j]] += 0.02 * lcg_normal(&mut s);
        }
    }
    let mean = Array1::<f64>::zeros(p);
    let ca = plane_candidate(p, 0, 1, &active);
    let cb = plane_candidate(p, 2, 3, &active);
    let v = screen_pair(data.view(), &mean, 0, 1, &ca, &cb);
    (v.rho, v.z, v.merge_proposed)
}

#[test]
fn split_single_circle_is_a_lower_tail_gap() {
    for &p in &[512usize, 2048] {
        let (rho, z, merge) = split_single_circle_rho(p, 0xA11CE ^ p as u64);
        eprintln!("[exp1 split-circle] p={p} ρ={rho:.4} z={z:.3} merge={merge}");
        // Energy complementarity of a split single chart lands at ρ≈1/2, the
        // LOWER tail: a genuine one-structure signal the ρ>1 merge screen misses.
        assert!(
            (rho - 0.5).abs() < 0.08,
            "split single circle must give ρ≈1/2 (complementary energies); p={p} got {rho:.4}"
        );
        assert!(
            !merge,
            "the ρ>1 presence screen must NOT fire on the lower-tail split; p={p} z={z:.3}"
        );
    }
}

#[test]
fn gated_torus_fires_scale_invariant() {
    let q = 0.4;
    for &p in &[512usize, 2048] {
        let (rho, z, merge) = gated_torus_rho(p, q, 0x7013 ^ p as u64);
        eprintln!("[exp3 gated-torus] p={p} q={q} ρ={rho:.4} z={z:.3} merge={merge}");
        assert!(
            merge,
            "co-gated torus MUST fire the merge screen; p={p} ρ={rho:.4} z={z:.3}"
        );
        // Anchor 1/q = 2.5, ambient-width invariant.
        assert!(
            (rho - 1.0 / q).abs() < 0.5,
            "co-gated ρ must sit near 1/q=2.5; p={p} got {rho:.4}"
        );
    }
}

#[test]
fn phase_correlation_is_invisible_to_energy_screen() {
    for &p in &[512usize, 2048] {
        let (rho, z, merge) = phase_correlated_dense_rho(p, 0xB0BA ^ p as u64);
        eprintln!("[exp2 phase-corr dense] p={p} ρ={rho:.4} z={z:.3} merge={merge}");
        // Dense presence pins each r²≡1, so the cross-moment sees independence
        // regardless of the (real) phase lock: ρ≈1, NO fire. Joint-density gap.
        assert!(
            (rho - 1.0).abs() < 0.05,
            "dense phase-locked circles must read ρ≈1 to the energy screen; p={p} got {rho:.4}"
        );
        assert!(
            !merge,
            "energy screen must NOT fire on a pure phase law at dense presence; p={p}"
        );
    }
}
