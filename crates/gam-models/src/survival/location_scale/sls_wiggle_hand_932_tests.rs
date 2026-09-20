//! #932 / #3319: parity and speed of the production link-wiggle row schedule
//! ([`sls_wiggle_row_order2`], [`sls_wiggle_row_third`], [`sls_wiggle_row_fourth`])
//! against the packed dynamic jets over [`sls_row_nll_wiggle`], which are the oracle.
//!
//! Production reads, per row at runtime width `KW = SLS_ROW_K + pw`, the gradient,
//! the `KW × KW` Hessian, the directional third contraction and the
//! second-directional fourth contraction (`SurvivalLsWiggleRowKernel`). It used to
//! read them off the packed jets, which ran 7–30× slower than the hand schedule,
//! so SPEC rule 1 ("exact forward-mode AD that is verified to match or surpass
//! hand-derived speed") did not admit them (#3319). The hand schedule is now
//! production and the jets are the test oracle: this module pins the schedule's
//! parity with the jets and races the two in release.
#![cfg(test)]

use super::*;
use gam_math::jet_scalar::{
    DynamicJetArena, DynamicOneSeed, DynamicOrder2, DynamicTwoSeed, RuntimeJetScalar,
};
use gam_math::paired_timing::{SpeedGate, batched, paired_interleaved};

/// The jet oracle's per-row order-two lowering: a reset arena, `DynamicOrder2`
/// seeds over the base primaries and `βw`, and `sls_row_nll_wiggle`. Returns the
/// gradient and the row-major Hessian.
fn jet_row_order2(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    arena: &mut DynamicJetArena,
) -> (Vec<f64>, Vec<f64>) {
    arena.reset();
    let arena: &DynamicJetArena = arena;
    let kw = SLS_ROW_K + betaw.len();
    let vars = arena.alloc_slice_fill_with(kw, |a| {
        let x = if a < SLS_ROW_K {
            p[a]
        } else {
            betaw[a - SLS_ROW_K]
        };
        DynamicOrder2::variable(x, a, kw, arena)
    });
    let out = sls_row_nll_wiggle(vars, kernel, betaw.len(), basis);
    (out.g().to_vec(), out.h().to_vec())
}

/// Base primaries and outer stacks from the patterned SLS order-two fixture, with
/// a nonzero third and fourth log-density slot. `event` sets the outcome, and an
/// untruncated row carries an all-zero entry stack.
fn row_fixture(event: f64, entry_truncated: bool) -> ([f64; SLS_ROW_K], SurvivalExactRowKernel) {
    let mut kernel = SurvivalExactRowKernel {
        w: 1.3,
        d: event,
        log_s0: -0.8,
        r0: 0.7,
        dr0: -0.3,
        ddr0: 0.12,
        dddr0: -0.05,
        log_s1: -1.1,
        r1: 0.9,
        dr1: -0.4,
        ddr1: 0.18,
        dddr1: -0.08,
        logphi1: -1.4,
        dlogphi1: -0.6,
        d2logphi1: -1.0,
        d3logphi1: 0.35,
        d4logphi1: -0.2,
        log_pdf1_minus_log_s0: -1.4 - (-0.8),
        log_s1_minus_log_s0: -1.1 - (-0.8),
        log_g: -0.2,
        d_log_g: 1.4,
        d2_log_g: -1.96,
        d3_log_g: 5.488,
        d4_log_g: -23.0496,
    };
    if !entry_truncated {
        kernel.log_s0 = 0.0;
        kernel.r0 = 0.0;
        kernel.dr0 = 0.0;
        kernel.ddr0 = 0.0;
        kernel.dddr0 = 0.0;
    }
    (
        [0.4, -0.7, 0.2, 0.8, -0.35, 0.11, -0.25, 0.31, -0.17],
        kernel,
    )
}

/// Per-row warp basis stacks at the entry (`B, …, B⁗`) and exit (`B, …, B⁽⁵⁾`)
/// indices. Every slot holds distinct values, so a lowering reading the wrong slot
/// cannot agree by coincidence.
struct BasisRows {
    entry: [Vec<f64>; 5],
    exit: [Vec<f64>; 6],
}

impl BasisRows {
    fn new(pw: usize) -> Self {
        Self {
            entry: std::array::from_fn(|order| {
                (0..pw)
                    .map(|j| (((j * 7 + order * 3 + 1) % 11) as f64 / 11.0 - 0.45) * 0.4)
                    .collect()
            }),
            exit: std::array::from_fn(|order| {
                (0..pw)
                    .map(|j| (((j * 5 + order * 7 + 2) % 13) as f64 / 13.0 - 0.5) * 0.4)
                    .collect()
            }),
        }
    }

    fn view(&self) -> SlsWiggleRowBasis<'_> {
        SlsWiggleRowBasis {
            b_u0: std::array::from_fn(|order| self.entry[order].as_slice()),
            b_u1: std::array::from_fn(|order| self.exit[order].as_slice()),
        }
    }
}

fn wiggle_amplitudes(pw: usize) -> Vec<f64> {
    (0..pw)
        .map(|j| (((j * 3 + 2) % 7) as f64 / 7.0 - 0.4) * 0.5)
        .collect()
}

/// The schedule's gradient and Hessian equal the jet oracle's on every entry, at
/// runtime widths 3 and 7, on an event row, a censored row and an untruncated event
/// row whose entry stack is all zero. The band is the `1e-11·max(1, |a|, |b|)`
/// of the dynamic-versus-padded-static parity oracle in `row_kernel.rs`. Every
/// entry is measured and the worst printed before any assertion, and a one-ppm
/// corruption of the exit stack's second slot must leave the band.
#[test]
fn sls_wiggle_row_order2_matches_jet_oracle_932() {
    let band = |a: f64, b: f64| 1e-11 * a.abs().max(b.abs()).max(1.0);
    let mut arena = DynamicJetArena::new();
    let mut scratch = SlsWiggleOrder2Scratch::new();
    let mut worst_over_band = 0.0_f64;
    let mut failures = Vec::new();
    let mut corrupted_trip = 0.0_f64;
    for pw in [3usize, 7] {
        let rows = BasisRows::new(pw);
        let basis = rows.view();
        let betaw = wiggle_amplitudes(pw);
        let kw = SLS_ROW_K + pw;
        for (label, event, entry_truncated) in [
            ("event", 1.0, true),
            ("censored", 0.0, true),
            ("event_untruncated", 1.0, false),
        ] {
            let (p, kernel) = row_fixture(event, entry_truncated);
            let (oracle_gradient, oracle) = jet_row_order2(&p, &betaw, &kernel, &basis, &mut arena);
            assert_eq!(oracle.len(), kw * kw);
            sls_wiggle_row_order2(&p, &betaw, &kernel, &basis, &mut scratch);
            for a in 0..kw {
                let want = oracle_gradient[a];
                let got = scratch.gradient[a];
                let over = (want - got).abs() / band(want, got);
                if !(over <= worst_over_band) {
                    worst_over_band = over;
                }
                if !(over <= 1.0) {
                    failures.push(format!(
                        "pw={pw} {label} g[{a}]: jet {want:+.15e} schedule {got:+.15e}"
                    ));
                }
            }
            for a in 0..kw {
                for b in 0..kw {
                    let want = oracle[a * kw + b];
                    let got = scratch.hessian[a * kw + b];
                    let over = (want - got).abs() / band(want, got);
                    if !(over <= worst_over_band) {
                        worst_over_band = over;
                    }
                    if !(over <= 1.0) {
                        failures.push(format!(
                            "pw={pw} {label} H[{a}][{b}]: jet {want:+.15e} schedule {got:+.15e}"
                        ));
                    }
                }
            }
            if label == "censored" {
                let mut corrupted = kernel;
                corrupted.dr1 *= 1.0 + 1e-6;
                sls_wiggle_row_order2(&p, &betaw, &corrupted, &basis, &mut scratch);
                for a in 0..kw {
                    for b in 0..kw {
                        let want = oracle[a * kw + b];
                        let got = scratch.hessian[a * kw + b];
                        let trip = (want - got).abs() / band(want, got);
                        if !(trip <= corrupted_trip) {
                            corrupted_trip = trip;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "SLS-WIGGLE-HAND-932 order=2 worst_over_band={worst_over_band:.3e} \
         corrupted_dr1_trip_over_band={corrupted_trip:.3e}"
    );
    assert!(
        failures.is_empty(),
        "{} gradient or Hessian entries miss the band:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        corrupted_trip > 1.0,
        "a one-ppm corruption of the exit stack's second slot stayed inside the band \
         ({corrupted_trip:.3e} of it)"
    );
}

#[inline(never)]
fn jet_order2_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    arena: &mut DynamicJetArena,
) -> f64 {
    arena.reset();
    let arena: &DynamicJetArena = arena;
    let kw = SLS_ROW_K + betaw.len();
    let vars = arena.alloc_slice_fill_with(kw, |a| {
        let x = if a < SLS_ROW_K {
            p[a]
        } else {
            betaw[a - SLS_ROW_K]
        };
        DynamicOrder2::variable(x, a, kw, arena)
    });
    sls_row_nll_wiggle(vars, kernel, betaw.len(), basis)
        .h()
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}

#[inline(never)]
fn schedule_order2_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    scratch: &mut SlsWiggleOrder2Scratch,
) -> f64 {
    sls_wiggle_row_order2(p, betaw, kernel, basis, scratch);
    scratch
        .hessian
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}

/// #932 / #3319 release cells: the production schedule's Hessian, directional third
/// contraction and second-directional fourth contraction must not be measurably
/// slower than the packed-jet oracle at runtime widths 3 and 12, so production
/// never again runs a lowering the jets would beat. Both arms
/// consume every entry, and both reuse their buffers across rows as production
/// does. Release profile only (`SpeedGate::open` documents why); parity is pinned
/// by `sls_wiggle_row_order2_matches_jet_oracle_932`,
/// `sls_wiggle_row_third_matches_jet_oracle_932` and
/// `sls_wiggle_row_fourth_matches_jet_oracle_932`.
#[test]
fn release_measure_sls_wiggle_schedule_vs_jet_oracle_932() {
    if cfg!(debug_assertions) {
        return;
    }
    const ROWS: usize = 8;
    let mut gate = SpeedGate::open("SLS-WIGGLE-HAND-932");
    for pw in [3usize, 12] {
        let rows = BasisRows::new(pw);
        let basis = rows.view();
        let betaw = wiggle_amplitudes(pw);
        let (p, kernel) = row_fixture(1.0, true);
        let mut arena = DynamicJetArena::new();
        let mut scratch = SlsWiggleOrder2Scratch::new();
        let timing = paired_interleaved(
            15,
            2_000,
            0x9320_5802 ^ pw as u64,
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                schedule_order2_checksum(&shifted, &betaw, &kernel, &basis, &mut scratch)
            }),
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                jet_order2_checksum(&shifted, &betaw, &kernel, &basis, &mut arena)
            }),
        );
        gate.not_slower(
            &format!("order=2 pw={pw}"),
            &timing,
            "production",
            "jet_oracle",
        );
        let dir = wiggle_direction(SLS_ROW_K + pw);
        let mut third_scratch = SlsWiggleThirdScratch::new();
        let timing_third = paired_interleaved(
            15,
            2_000,
            0x9320_5803 ^ pw as u64,
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                schedule_order3_checksum(&shifted, &betaw, &kernel, &basis, &dir, &mut third_scratch)
            }),
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                jet_order3_checksum(&shifted, &betaw, &kernel, &basis, &dir, &mut arena)
            }),
        );
        gate.not_slower(
            &format!("order=3 pw={pw}"),
            &timing_third,
            "production",
            "jet_oracle",
        );
        let second_dir = wiggle_second_direction(SLS_ROW_K + pw);
        let mut fourth_scratch = SlsWiggleFourthScratch::new();
        let timing_fourth = paired_interleaved(
            15,
            2_000,
            0x9320_5804 ^ pw as u64,
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                schedule_order4_checksum(
                    &shifted,
                    &betaw,
                    &kernel,
                    &basis,
                    &dir,
                    &second_dir,
                    &mut fourth_scratch,
                )
            }),
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                jet_order4_checksum(
                    &shifted,
                    &betaw,
                    &kernel,
                    &basis,
                    &dir,
                    &second_dir,
                    &mut arena,
                )
            }),
        );
        gate.not_slower(
            &format!("order=4 pw={pw}"),
            &timing_fourth,
            "production",
            "jet_oracle",
        );
    }
    gate.finish();
}

/// The jet oracle's per-row directional third lowering (`DynamicOneSeed`).
fn jet_row_third(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    dir: &[f64],
    arena: &mut DynamicJetArena,
) -> Vec<f64> {
    arena.reset();
    let arena: &DynamicJetArena = arena;
    let kw = SLS_ROW_K + betaw.len();
    let vars = arena.alloc_slice_fill_with(kw, |a| {
        let x = if a < SLS_ROW_K {
            p[a]
        } else {
            betaw[a - SLS_ROW_K]
        };
        DynamicOneSeed::seed_direction(x, a, dir[a], kw, arena)
    });
    sls_row_nll_wiggle(vars, kernel, betaw.len(), basis)
        .contracted_third()
        .to_vec()
}

/// A direction with every primary live, distinct across axes.
fn wiggle_direction(kw: usize) -> Vec<f64> {
    (0..kw)
        .map(|a| (((a * 5 + 3) % 11) as f64 / 11.0 - 0.45) * 1.2)
        .collect()
}

/// #932 row 58, stage 2: the schedule's directional third contraction equals the
/// jet oracle's `DynamicOneSeed` lowering on every entry, at runtime widths 3 and 7, on an event
/// row, a censored row and an untruncated event row, along a direction with every
/// primary live. It uses the band and the measure-then-assert structure of the
/// Hessian test. The control corrupts the exit stack's third slot, which no
/// order-two quantity reads.
#[test]
fn sls_wiggle_row_third_matches_jet_oracle_932() {
    let band = |a: f64, b: f64| 1e-11 * a.abs().max(b.abs()).max(1.0);
    let mut arena = DynamicJetArena::new();
    let mut scratch = SlsWiggleThirdScratch::new();
    let mut worst_over_band = 0.0_f64;
    let mut failures = Vec::new();
    let mut corrupted_trip = 0.0_f64;
    for pw in [3usize, 7] {
        let rows = BasisRows::new(pw);
        let basis = rows.view();
        let betaw = wiggle_amplitudes(pw);
        let kw = SLS_ROW_K + pw;
        let dir = wiggle_direction(kw);
        for (label, event, entry_truncated) in [
            ("event", 1.0, true),
            ("censored", 0.0, true),
            ("event_untruncated", 1.0, false),
        ] {
            let (p, kernel) = row_fixture(event, entry_truncated);
            let oracle = jet_row_third(&p, &betaw, &kernel, &basis, &dir, &mut arena);
            assert_eq!(oracle.len(), kw * kw);
            sls_wiggle_row_third(&p, &betaw, &kernel, &basis, &dir, &mut scratch);
            for a in 0..kw {
                for b in 0..kw {
                    let want = oracle[a * kw + b];
                    let got = scratch.third[a * kw + b];
                    let over = (want - got).abs() / band(want, got);
                    if !(over <= worst_over_band) {
                        worst_over_band = over;
                    }
                    if !(over <= 1.0) {
                        failures.push(format!(
                            "pw={pw} {label} T[{a}][{b}]: jet {want:+.15e} schedule {got:+.15e}"
                        ));
                    }
                }
            }
            if label == "censored" {
                let mut corrupted = kernel;
                corrupted.ddr1 *= 1.0 + 1e-6;
                sls_wiggle_row_third(&p, &betaw, &corrupted, &basis, &dir, &mut scratch);
                for a in 0..kw {
                    for b in 0..kw {
                        let want = oracle[a * kw + b];
                        let got = scratch.third[a * kw + b];
                        let trip = (want - got).abs() / band(want, got);
                        if !(trip <= corrupted_trip) {
                            corrupted_trip = trip;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "SLS-WIGGLE-HAND-932 order=3 worst_over_band={worst_over_band:.3e} \
         corrupted_ddr1_trip_over_band={corrupted_trip:.3e}"
    );
    assert!(
        failures.is_empty(),
        "{} third-contraction entries miss the band:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        corrupted_trip > 1.0,
        "a one-ppm corruption of the exit stack's third slot stayed inside the band \
         ({corrupted_trip:.3e} of it)"
    );
}

#[inline(never)]
fn jet_order3_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    dir: &[f64],
    arena: &mut DynamicJetArena,
) -> f64 {
    arena.reset();
    let arena: &DynamicJetArena = arena;
    let kw = SLS_ROW_K + betaw.len();
    let vars = arena.alloc_slice_fill_with(kw, |a| {
        let x = if a < SLS_ROW_K {
            p[a]
        } else {
            betaw[a - SLS_ROW_K]
        };
        DynamicOneSeed::seed_direction(x, a, dir[a], kw, arena)
    });
    sls_row_nll_wiggle(vars, kernel, betaw.len(), basis)
        .contracted_third()
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}

#[inline(never)]
fn schedule_order3_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    dir: &[f64],
    scratch: &mut SlsWiggleThirdScratch,
) -> f64 {
    sls_wiggle_row_third(p, betaw, kernel, basis, dir, scratch);
    scratch
        .third
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}

/// The jet oracle's per-row second-directional fourth lowering (`DynamicTwoSeed`).
fn jet_row_fourth(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    u: &[f64],
    v: &[f64],
    arena: &mut DynamicJetArena,
) -> Vec<f64> {
    arena.reset();
    let arena: &DynamicJetArena = arena;
    let kw = SLS_ROW_K + betaw.len();
    let vars = arena.alloc_slice_fill_with(kw, |a| {
        let x = if a < SLS_ROW_K {
            p[a]
        } else {
            betaw[a - SLS_ROW_K]
        };
        DynamicTwoSeed::seed(x, a, u[a], v[a], kw, arena)
    });
    sls_row_nll_wiggle(vars, kernel, betaw.len(), basis)
        .contracted_fourth()
        .to_vec()
}

/// A second direction with every primary live, distinct from `wiggle_direction`.
fn wiggle_second_direction(kw: usize) -> Vec<f64> {
    (0..kw)
        .map(|a| (((a * 7 + 2) % 13) as f64 / 13.0 - 0.5) * 1.1)
        .collect()
}

/// #932 row 58, stage 3: the schedule's second-directional fourth contraction
/// equals the jet oracle's `DynamicTwoSeed` lowering on every entry, at runtime widths 3 and 7,
/// on an event row, a censored row and an untruncated event row, along two directions
/// with every primary live. It uses the band and the measure-then-assert structure of
/// the lower orders. The control corrupts the exit stack's fourth slot, which no
/// lower-order quantity reads.
#[test]
fn sls_wiggle_row_fourth_matches_jet_oracle_932() {
    let band = |a: f64, b: f64| 1e-11 * a.abs().max(b.abs()).max(1.0);
    let mut arena = DynamicJetArena::new();
    let mut scratch = SlsWiggleFourthScratch::new();
    let mut worst_over_band = 0.0_f64;
    let mut failures = Vec::new();
    let mut corrupted_trip = 0.0_f64;
    for pw in [3usize, 7] {
        let rows = BasisRows::new(pw);
        let basis = rows.view();
        let betaw = wiggle_amplitudes(pw);
        let kw = SLS_ROW_K + pw;
        let u = wiggle_direction(kw);
        let v = wiggle_second_direction(kw);
        for (label, event, entry_truncated) in [
            ("event", 1.0, true),
            ("censored", 0.0, true),
            ("event_untruncated", 1.0, false),
        ] {
            let (p, kernel) = row_fixture(event, entry_truncated);
            let oracle = jet_row_fourth(&p, &betaw, &kernel, &basis, &u, &v, &mut arena);
            assert_eq!(oracle.len(), kw * kw);
            sls_wiggle_row_fourth(&p, &betaw, &kernel, &basis, &u, &v, &mut scratch);
            for a in 0..kw {
                for b in 0..kw {
                    let want = oracle[a * kw + b];
                    let got = scratch.fourth[a * kw + b];
                    let over = (want - got).abs() / band(want, got);
                    if !(over <= worst_over_band) {
                        worst_over_band = over;
                    }
                    if !(over <= 1.0) {
                        failures.push(format!(
                            "pw={pw} {label} F[{a}][{b}]: jet {want:+.15e} schedule {got:+.15e}"
                        ));
                    }
                }
            }
            if label == "censored" {
                let mut corrupted = kernel;
                corrupted.dddr1 *= 1.0 + 1e-6;
                sls_wiggle_row_fourth(&p, &betaw, &corrupted, &basis, &u, &v, &mut scratch);
                for a in 0..kw {
                    for b in 0..kw {
                        let want = oracle[a * kw + b];
                        let got = scratch.fourth[a * kw + b];
                        let trip = (want - got).abs() / band(want, got);
                        if !(trip <= corrupted_trip) {
                            corrupted_trip = trip;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "SLS-WIGGLE-HAND-932 order=4 worst_over_band={worst_over_band:.3e} \
         corrupted_dddr1_trip_over_band={corrupted_trip:.3e}"
    );
    assert!(
        failures.is_empty(),
        "{} fourth-contraction entries miss the band:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        corrupted_trip > 1.0,
        "a one-ppm corruption of the exit stack's fourth slot stayed inside the band \
         ({corrupted_trip:.3e} of it)"
    );
}

#[inline(never)]
fn jet_order4_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    u: &[f64],
    v: &[f64],
    arena: &mut DynamicJetArena,
) -> f64 {
    arena.reset();
    let arena: &DynamicJetArena = arena;
    let kw = SLS_ROW_K + betaw.len();
    let vars = arena.alloc_slice_fill_with(kw, |a| {
        let x = if a < SLS_ROW_K {
            p[a]
        } else {
            betaw[a - SLS_ROW_K]
        };
        DynamicTwoSeed::seed(x, a, u[a], v[a], kw, arena)
    });
    sls_row_nll_wiggle(vars, kernel, betaw.len(), basis)
        .contracted_fourth()
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}

#[inline(never)]
fn schedule_order4_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    u: &[f64],
    v: &[f64],
    scratch: &mut SlsWiggleFourthScratch,
) -> f64 {
    sls_wiggle_row_fourth(p, betaw, kernel, basis, u, v, scratch);
    scratch
        .fourth
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}
