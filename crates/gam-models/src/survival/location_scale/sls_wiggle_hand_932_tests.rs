//! #932 row 58: the strongest hand opponent for the runtime-width survival
//! location-scale link-wiggle row.
//!
//! Production evaluates `sls_row_nll_wiggle` at runtime width
//! `KW = SLS_ROW_K + pw` with packed dynamic jets and reads, per row, the
//! `KW × KW` Hessian, the directional third contraction and the second-directional
//! fourth contraction (`SurvivalLsWiggleRowKernel`). SPEC rule 1 admits a
//! forward-mode lowering only where it is verified to match or surpass
//! hand-derived speed. Until now the only gate, `SLS-WIGGLE-DYN-932`, raced a
//! reused arena against a fresh one, which measures the arena, not the derivative
//! schedule. This module holds the hand schedule, its parity with production, and
//! the release cell that races the two. It holds the Hessian, the directional third
//! contraction and the second-directional fourth contraction.
//!
//! The row NLL is a sum of three scalar compositions, `F0(u0w) + F1(u1w) + F2(g)`,
//! plus a term linear in `x6`:
//! - `u0w = x0·e^{−x7} + q0 + Σ βw_j·B0_j(q0)` with `q0 = −x4·e^{−x7}`;
//! - `u1w = x1·e^{−x6} + q1 + Σ βw_j·B1_j(q1)` with `q1 = −x3·e^{−x6}`;
//! - `g = (x2 − x1·x8) + m1·qdot0` with `m1 = 1 + Σ βw_j·B1′_j(q1)` and
//!   `qdot0 = x3·x8 − x5`.
//!
//! The scale divides the time channels too (#2695): `t0 = x0·e^{−x7}` and
//! `t1 = x1·e^{−x6}` are the time transform's share of `u0w` and `u1w`, and
//! `tg = x2 − x1·x8` its share of `g`. The event Jacobian is `e^{−x6}·g`, whose
//! `−x6` enters the event log-density linearly and so has no curvature here.
//!
//! Each intermediate reads a few primaries and is linear in `βw`. So the hand
//! schedule forms the Hessian as three rank-one outer products plus the sparse
//! intermediate Hessians, while the packed jet carries all `KW²` channels through
//! every operation.
#![cfg(test)]

use super::*;
use gam_math::jet_scalar::{
    DynamicJetArena, DynamicOneSeed, DynamicOrder2, DynamicTwoSeed, RuntimeJetScalar,
};
use gam_math::paired_timing::{SpeedGate, batched, paired_interleaved};

/// Reusable buffers for the hand schedule, the counterpart of production's
/// per-fold `DynamicJetArena`.
struct HandWiggleScratch {
    grad_entry: Vec<f64>,
    grad_exit: Vec<f64>,
    grad_rate: Vec<f64>,
    hessian: Vec<f64>,
}

impl HandWiggleScratch {
    fn new() -> Self {
        Self {
            grad_entry: Vec::new(),
            grad_exit: Vec::new(),
            grad_rate: Vec::new(),
            hessian: Vec::new(),
        }
    }
}

fn stack_is_zero(stack: &[f64; 5]) -> bool {
    stack.iter().all(|value| *value == 0.0)
}

/// Add `scale·v·vᵀ` to the row-major `kw × kw` buffer, visiting only the nonzero
/// components of `v`.
fn add_outer(out: &mut [f64], kw: usize, scale: f64, v: &[f64]) {
    for a in 0..kw {
        let left = scale * v[a];
        if left != 0.0 {
            let row = &mut out[a * kw..(a + 1) * kw];
            for b in 0..kw {
                row[b] += left * v[b];
            }
        }
    }
}

/// Add one entry of a symmetric matrix: `value` at `(i, j)` and, when `i ≠ j`,
/// at `(j, i)`.
fn add_symmetric(out: &mut [f64], kw: usize, i: usize, j: usize, value: f64) {
    out[i * kw + j] += value;
    if i != j {
        out[j * kw + i] += value;
    }
}

/// Add one term of `m·qᵀ + q·mᵀ`: `value = m_i·q_j` at both `(i, j)` and `(j, i)`,
/// which doubles on the diagonal as that sum does.
fn add_pair(out: &mut [f64], kw: usize, i: usize, j: usize, value: f64) {
    out[i * kw + j] += value;
    out[j * kw + i] += value;
}

/// The per-row `KW × KW` Hessian of `sls_row_nll_wiggle`, row-major into
/// `scratch.hessian`, by the structure the module doc describes. Production skips
/// an exactly-zero outer stack instead of composing it (the #2342 far-tail `0·∞`
/// guard), so an inactive term contributes nothing here either.
#[inline(never)]
fn hand_sls_wiggle_row_hessian(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    scratch: &mut HandWiggleScratch,
) {
    let pw = betaw.len();
    let kw = SLS_ROW_K + pw;
    let entry = [
        kernel.log_s0,
        -kernel.r0,
        -kernel.dr0,
        -kernel.ddr0,
        -kernel.dddr0,
    ];
    let exit = [
        kernel.log_s1,
        -kernel.r1,
        -kernel.dr1,
        -kernel.ddr1,
        -kernel.dddr1,
    ];
    let pdf = [
        kernel.logphi1,
        kernel.dlogphi1,
        kernel.d2logphi1,
        kernel.d3logphi1,
        kernel.d4logphi1,
    ];
    let rate = [
        kernel.log_g,
        kernel.d_log_g,
        kernel.d2_log_g,
        kernel.d3_log_g,
        kernel.d4_log_g,
    ];
    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let entry_active = !stack_is_zero(&entry);
    let censored_active = censored_weight != 0.0 && !stack_is_zero(&exit);
    let pdf_active = event_weight != 0.0 && !stack_is_zero(&pdf);
    let rate_active = event_weight != 0.0 && !stack_is_zero(&rate);
    let (f0_first, f0_second) = (kernel.w * entry[1], kernel.w * entry[2]);
    let mut f1_first = 0.0;
    let mut f1_second = 0.0;
    if censored_active {
        f1_first -= censored_weight * exit[1];
        f1_second -= censored_weight * exit[2];
    }
    if pdf_active {
        f1_first -= event_weight * pdf[1];
        f1_second -= event_weight * pdf[2];
    }
    let (f2_first, f2_second) = (-event_weight * rate[1], -event_weight * rate[2]);

    let s7 = (-p[7]).exp();
    let q0 = -p[4] * s7;
    let s6 = (-p[6]).exp();
    let q1 = -p[3] * s6;
    let qdot0 = p[3] * p[8] - p[5];
    let t0 = p[0] * s7;
    let t1 = p[1] * s6;
    let (b0, b0d1, b0d2) = (basis.b_u0[0], basis.b_u0[1], basis.b_u0[2]);
    let (b1, b1d1, b1d2, b1d3) = (basis.b_u1[0], basis.b_u1[1], basis.b_u1[2], basis.b_u1[3]);
    // a0 = ∂u0w/∂q0, a0p = ∂²u0w/∂q0², a1 = ∂u1w/∂q1 = m1, a1p = ∂m1/∂q1,
    // a1pp = ∂²m1/∂q1².
    let mut a0 = 1.0;
    let mut a0p = 0.0;
    let mut a1 = 1.0;
    let mut a1p = 0.0;
    let mut a1pp = 0.0;
    for j in 0..pw {
        a0 += betaw[j] * b0d1[j];
        a0p += betaw[j] * b0d2[j];
        a1 += betaw[j] * b1d1[j];
        a1p += betaw[j] * b1d2[j];
        a1pp += betaw[j] * b1d3[j];
    }

    let HandWiggleScratch {
        grad_entry,
        grad_exit,
        grad_rate,
        hessian,
    } = scratch;
    for buffer in [&mut *grad_entry, &mut *grad_exit, &mut *grad_rate] {
        buffer.clear();
        buffer.resize(kw, 0.0);
    }
    hessian.clear();
    hessian.resize(kw * kw, 0.0);

    // ∇q0 = (−s7 at x4, −q0 at x7) and ∇q1 = (−s6 at x3, −q1 at x6).
    // ∇t0 = (s7 at x0, −t0 at x7) and ∇t1 = (s6 at x1, −t1 at x6).
    grad_entry[0] = s7;
    grad_entry[4] = -a0 * s7;
    grad_entry[7] = -a0 * q0 - t0;
    grad_exit[1] = s6;
    grad_exit[3] = -a1 * s6;
    grad_exit[6] = -a1 * q1 - t1;
    // ∇g = ∇tg + qdot0·∇m1 + m1·∇qdot0, with ∇m1 = a1p·∇q1 + Σ_j B1′_j·e_j,
    // ∇qdot0 = (x8 at x3, −1 at x5, x3 at x8) and ∇tg = (−x8 at x1, 1 at x2,
    // −x1 at x8).
    grad_rate[1] = -p[8];
    grad_rate[2] = 1.0;
    grad_rate[3] = -qdot0 * a1p * s6 + a1 * p[8];
    grad_rate[5] = -a1;
    grad_rate[6] = -qdot0 * a1p * q1;
    grad_rate[8] = a1 * p[3] - p[1];
    for j in 0..pw {
        grad_entry[SLS_ROW_K + j] = b0[j];
        grad_exit[SLS_ROW_K + j] = b1[j];
        grad_rate[SLS_ROW_K + j] = qdot0 * b1d1[j];
    }

    if entry_active {
        add_outer(hessian, kw, f0_second, grad_entry);
        // F0′·∇²u0w, ∇²u0w = a0p·∇q0∇q0ᵀ + a0·∇²q0 + Σ_j B0′_j·(e_j∇q0ᵀ + ∇q0e_jᵀ),
        // where ∇²q0 = (s7 at (x4, x7), q0 at (x7, x7)).
        add_symmetric(hessian, kw, 4, 4, f0_first * a0p * s7 * s7);
        add_symmetric(hessian, kw, 4, 7, f0_first * (a0p * s7 * q0 + a0 * s7));
        add_symmetric(hessian, kw, 7, 7, f0_first * (a0p * q0 * q0 + a0 * q0));
        // F0′·∇²t0, ∇²t0 = (−s7 at (x0, x7), t0 at (x7, x7)).
        add_symmetric(hessian, kw, 0, 7, -f0_first * s7);
        add_symmetric(hessian, kw, 7, 7, f0_first * t0);
        for j in 0..pw {
            add_symmetric(hessian, kw, SLS_ROW_K + j, 4, -f0_first * b0d1[j] * s7);
            add_symmetric(hessian, kw, SLS_ROW_K + j, 7, -f0_first * b0d1[j] * q0);
        }
    }
    if censored_active || pdf_active {
        add_outer(hessian, kw, f1_second, grad_exit);
        // F1′·∇²u1w, the same form over (x3, x6).
        add_symmetric(hessian, kw, 3, 3, f1_first * a1p * s6 * s6);
        add_symmetric(hessian, kw, 3, 6, f1_first * (a1p * s6 * q1 + a1 * s6));
        add_symmetric(hessian, kw, 6, 6, f1_first * (a1p * q1 * q1 + a1 * q1));
        // F1′·∇²t1, ∇²t1 = (−s6 at (x1, x6), t1 at (x6, x6)).
        add_symmetric(hessian, kw, 1, 6, -f1_first * s6);
        add_symmetric(hessian, kw, 6, 6, f1_first * t1);
        for j in 0..pw {
            add_symmetric(hessian, kw, SLS_ROW_K + j, 3, -f1_first * b1d1[j] * s6);
            add_symmetric(hessian, kw, SLS_ROW_K + j, 6, -f1_first * b1d1[j] * q1);
        }
    }
    if rate_active {
        add_outer(hessian, kw, f2_second, grad_rate);
        // F2′·∇²g, ∇²g = qdot0·∇²m1 + (∇m1∇qdot0ᵀ + ∇qdot0∇m1ᵀ) + m1·∇²qdot0.
        // First qdot0·∇²m1, ∇²m1 = a1pp·∇q1∇q1ᵀ + a1p·∇²q1 + Σ_j B1″_j·(e_j∇q1ᵀ + ∇q1e_jᵀ).
        let scale_m1 = f2_first * qdot0;
        add_symmetric(hessian, kw, 3, 3, scale_m1 * a1pp * s6 * s6);
        add_symmetric(hessian, kw, 3, 6, scale_m1 * (a1pp * s6 * q1 + a1p * s6));
        add_symmetric(hessian, kw, 6, 6, scale_m1 * (a1pp * q1 * q1 + a1p * q1));
        for j in 0..pw {
            add_symmetric(hessian, kw, SLS_ROW_K + j, 3, -scale_m1 * b1d2[j] * s6);
            add_symmetric(hessian, kw, SLS_ROW_K + j, 6, -scale_m1 * b1d2[j] * q1);
        }
        // Then ∇m1∇qdot0ᵀ + ∇qdot0∇m1ᵀ.
        let qdot0_grad = [(3, p[8]), (5, -1.0), (8, p[3])];
        for (index, value) in qdot0_grad {
            add_pair(hessian, kw, 3, index, f2_first * -a1p * s6 * value);
            add_pair(hessian, kw, 6, index, f2_first * -a1p * q1 * value);
            for j in 0..pw {
                add_pair(hessian, kw, SLS_ROW_K + j, index, f2_first * b1d1[j] * value);
            }
        }
        // Then m1·∇²qdot0, ∇²qdot0 = (1 at (x3, x8)), and F2′·∇²tg,
        // ∇²tg = (−1 at (x1, x8)).
        add_symmetric(hessian, kw, 3, 8, f2_first * a1);
        add_symmetric(hessian, kw, 1, 8, -f2_first);
    }
}

/// Production's per-row order-two lowering, as `SurvivalLsWiggleRowKernel::row_order2`
/// runs it: a reset arena, `DynamicOrder2` seeds over the base primaries and `βw`,
/// and `sls_row_nll_wiggle`.
fn production_row_hessian(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
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
        DynamicOrder2::variable(x, a, kw, arena)
    });
    sls_row_nll_wiggle(vars, kernel, betaw.len(), basis)
        .h()
        .to_vec()
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

/// The hand Hessian equals production's order-two lowering on every entry, at
/// runtime widths 3 and 7, on an event row, a censored row and an untruncated event
/// row whose entry stack is all zero. The band is the `1e-11·max(1, |a|, |b|)`
/// of the dynamic-versus-padded-static parity oracle in `row_kernel.rs`. Every
/// entry is measured and the worst printed before any assertion, and a one-ppm
/// corruption of the exit stack's second slot must leave the band.
#[test]
fn hand_sls_wiggle_row_hessian_matches_production_jet_932() {
    let band = |a: f64, b: f64| 1e-11 * a.abs().max(b.abs()).max(1.0);
    let mut arena = DynamicJetArena::new();
    let mut scratch = HandWiggleScratch::new();
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
            let production = production_row_hessian(&p, &betaw, &kernel, &basis, &mut arena);
            assert_eq!(production.len(), kw * kw);
            hand_sls_wiggle_row_hessian(&p, &betaw, &kernel, &basis, &mut scratch);
            for a in 0..kw {
                for b in 0..kw {
                    let want = production[a * kw + b];
                    let got = scratch.hessian[a * kw + b];
                    let over = (want - got).abs() / band(want, got);
                    if !(over <= worst_over_band) {
                        worst_over_band = over;
                    }
                    if !(over <= 1.0) {
                        failures.push(format!(
                            "pw={pw} {label} H[{a}][{b}]: production {want:+.15e} hand {got:+.15e}"
                        ));
                    }
                }
            }
            if label == "censored" {
                let mut corrupted = kernel;
                corrupted.dr1 *= 1.0 + 1e-6;
                hand_sls_wiggle_row_hessian(&p, &betaw, &corrupted, &basis, &mut scratch);
                for a in 0..kw {
                    for b in 0..kw {
                        let want = production[a * kw + b];
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
        "{} Hessian entries miss the band:\n{}",
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
fn production_order2_checksum(
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
fn hand_order2_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    scratch: &mut HandWiggleScratch,
) -> f64 {
    hand_sls_wiggle_row_hessian(p, betaw, kernel, basis, scratch);
    scratch
        .hessian
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}

/// #932 release cells: production's packed-jet Hessian, directional third contraction
/// and second-directional fourth contraction must not be measurably slower than the
/// hand schedules at runtime
/// widths 3 and 12 (SPEC rule 1's "match or surpass hand-derived speed"). Both arms
/// consume every entry, and both reuse their buffers across rows as production
/// does. Release profile only (`SpeedGate::open` documents why); parity is pinned
/// by `hand_sls_wiggle_row_hessian_matches_production_jet_932`,
/// `hand_sls_wiggle_row_third_matches_production_jet_932` and
/// `hand_sls_wiggle_row_fourth_matches_production_jet_932`.
#[test]
fn release_measure_sls_wiggle_production_jet_vs_hand_932() {
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
        let mut scratch = HandWiggleScratch::new();
        let timing = paired_interleaved(
            15,
            2_000,
            0x9320_5802 ^ pw as u64,
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                production_order2_checksum(&shifted, &betaw, &kernel, &basis, &mut arena)
            }),
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                hand_order2_checksum(&shifted, &betaw, &kernel, &basis, &mut scratch)
            }),
        );
        gate.not_slower(
            &format!("order=2 pw={pw}"),
            &timing,
            "production_jet",
            "strongest_hand",
        );
        let dir = wiggle_direction(SLS_ROW_K + pw);
        let mut third_scratch = HandWiggleThirdScratch::new();
        let timing_third = paired_interleaved(
            15,
            2_000,
            0x9320_5803 ^ pw as u64,
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                production_order3_checksum(&shifted, &betaw, &kernel, &basis, &dir, &mut arena)
            }),
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                hand_order3_checksum(&shifted, &betaw, &kernel, &basis, &dir, &mut third_scratch)
            }),
        );
        gate.not_slower(
            &format!("order=3 pw={pw}"),
            &timing_third,
            "production_jet",
            "strongest_hand",
        );
        let second_dir = wiggle_second_direction(SLS_ROW_K + pw);
        let mut fourth_scratch = HandWiggleFourthScratch::new();
        let timing_fourth = paired_interleaved(
            15,
            2_000,
            0x9320_5804 ^ pw as u64,
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                production_order4_checksum(
                    &shifted,
                    &betaw,
                    &kernel,
                    &basis,
                    &dir,
                    &second_dir,
                    &mut arena,
                )
            }),
            batched(ROWS, |nudge| {
                let mut shifted = p;
                shifted[0] += nudge;
                hand_order4_checksum(
                    &shifted,
                    &betaw,
                    &kernel,
                    &basis,
                    &dir,
                    &second_dir,
                    &mut fourth_scratch,
                )
            }),
        );
        gate.not_slower(
            &format!("order=4 pw={pw}"),
            &timing_fourth,
            "production_jet",
            "strongest_hand",
        );
    }
    gate.finish();
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// `out = M·v` for a row-major `kw × kw` matrix `M`.
fn matvec(matrix: &[f64], kw: usize, v: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.extend((0..kw).map(|a| dot(&matrix[a * kw..(a + 1) * kw], v)));
}

/// One intermediate's derivative tower over the `KW` primaries: its gradient, its
/// Hessian and its third derivative contracted with the direction `d`.
struct WiggleTower {
    grad: Vec<f64>,
    hess: Vec<f64>,
    third: Vec<f64>,
}

impl WiggleTower {
    fn new() -> Self {
        Self {
            grad: Vec::new(),
            hess: Vec::new(),
            third: Vec::new(),
        }
    }

    fn reset(&mut self, kw: usize) {
        self.grad.clear();
        self.grad.resize(kw, 0.0);
        self.hess.clear();
        self.hess.resize(kw * kw, 0.0);
        self.third.clear();
        self.third.resize(kw * kw, 0.0);
    }
}

/// Reusable buffers for the hand third contraction.
struct HandWiggleThirdScratch {
    entry: WiggleTower,
    exit: WiggleTower,
    multiplier: WiggleTower,
    rate_index: WiggleTower,
    time_rate: WiggleTower,
    rate: WiggleTower,
    hess_dir: Vec<f64>,
    multiplier_hess_dir: Vec<f64>,
    rate_index_hess_dir: Vec<f64>,
    third: Vec<f64>,
}

impl HandWiggleThirdScratch {
    fn new() -> Self {
        Self {
            entry: WiggleTower::new(),
            exit: WiggleTower::new(),
            multiplier: WiggleTower::new(),
            rate_index: WiggleTower::new(),
            time_rate: WiggleTower::new(),
            rate: WiggleTower::new(),
            hess_dir: Vec::new(),
            multiplier_hess_dir: Vec::new(),
            rate_index_hess_dir: Vec::new(),
            third: Vec::new(),
        }
    }
}

/// The tower of `U = t + G(q, βw)` with `q = −x_numerator·e^{−x_log_scale}` and, when
/// `linear = Some((index, t))`, the time share `t = x_index·e^{−x_log_scale}` (#2695), else
/// `t = 0`. `derivs` holds `∂G/∂q`, `∂²G/∂q²` and `∂³G/∂q³`. `slots` holds, per wiggle
/// coefficient, `∂G/∂βw_j`, `∂²G/∂q∂βw_j` and `∂³G/∂q²∂βw_j`. `G` is linear in `βw`,
/// so no second `βw` derivative exists. With `∇q = (−s at numerator, −q at
/// log_scale)` and `∇²q = (s at (numerator, log_scale), q at (log_scale, log_scale))`:
/// - `∇U = e_linear + G_q·∇q + Σ_j G_βj·e_j`;
/// - `∇²U = G_qq·∇q∇qᵀ + G_q·∇²q + Σ_j G_qβj·(e_j∇qᵀ + ∇qe_jᵀ)`;
/// - `∇³U[d] = (G_qqq·dq + Σ_j G_qqβj·dβ_j)·∇q∇qᵀ + (G_qq·dq + Σ_j G_qβj·dβ_j)·∇²q
///   + G_qq·(h∇qᵀ + ∇qhᵀ) + G_q·∇³q[d] + Σ_j [G_qqβj·dq·(e_j∇qᵀ + ∇qe_jᵀ) + G_qβj·(e_jhᵀ + he_jᵀ)]`,
///
/// where `dq = ∇q·d`, `h = ∇²q·d`, and `∇³q[d] = (−s·d_log_scale at (numerator,
/// log_scale), dq at (log_scale, log_scale))`. The time share adds `∇t = (s at index, −t
/// at log_scale)`, `∇²t = (−s at (index, log_scale), t at (log_scale, log_scale))` and
/// `∇³t[d] = (s·d_log_scale at (index, log_scale), s·d_index − t·d_log_scale at
/// (log_scale, log_scale))`.
#[inline(never)]
fn warp_tower(
    kw: usize,
    linear: Option<(usize, f64)>,
    numerator: usize,
    log_scale: usize,
    scale: f64,
    q: f64,
    derivs: [f64; 3],
    slots: [&[f64]; 3],
    dir: &[f64],
    tower: &mut WiggleTower,
) {
    tower.reset(kw);
    let pw = kw - SLS_ROW_K;
    let [g_q, g_qq, g_qqq] = derivs;
    let dq = -scale * dir[numerator] - q * dir[log_scale];
    let grad_q = [(numerator, -scale), (log_scale, -q)];
    let hess_q_dir = [
        (numerator, scale * dir[log_scale]),
        (log_scale, scale * dir[numerator] + q * dir[log_scale]),
    ];
    let mut slot1_dir = 0.0;
    let mut slot2_dir = 0.0;
    for j in 0..pw {
        slot1_dir += slots[1][j] * dir[SLS_ROW_K + j];
        slot2_dir += slots[2][j] * dir[SLS_ROW_K + j];
    }
    let WiggleTower { grad, hess, third } = tower;
    if let Some((index, time)) = linear {
        grad[index] = scale;
        grad[log_scale] = -time;
        add_symmetric(hess, kw, index, log_scale, -scale);
        add_symmetric(hess, kw, log_scale, log_scale, time);
        add_symmetric(third, kw, index, log_scale, scale * dir[log_scale]);
        add_symmetric(
            third,
            kw,
            log_scale,
            log_scale,
            scale * dir[index] - time * dir[log_scale],
        );
    }
    for (index, value) in grad_q {
        grad[index] += g_q * value;
    }
    for j in 0..pw {
        grad[SLS_ROW_K + j] = slots[0][j];
    }
    for (a, left) in grad_q {
        for (b, right) in grad_q {
            hess[a * kw + b] += g_qq * left * right;
            third[a * kw + b] += (g_qqq * dq + slot2_dir) * left * right;
        }
    }
    add_symmetric(hess, kw, numerator, log_scale, g_q * scale);
    add_symmetric(hess, kw, log_scale, log_scale, g_q * q);
    let curvature_scale = g_qq * dq + slot1_dir;
    add_symmetric(third, kw, numerator, log_scale, curvature_scale * scale);
    add_symmetric(third, kw, log_scale, log_scale, curvature_scale * q);
    for (a, left) in hess_q_dir {
        for (b, right) in grad_q {
            add_pair(third, kw, a, b, g_qq * left * right);
        }
    }
    add_symmetric(third, kw, numerator, log_scale, -g_q * scale * dir[log_scale]);
    add_symmetric(third, kw, log_scale, log_scale, g_q * dq);
    for j in 0..pw {
        for (index, value) in grad_q {
            add_symmetric(hess, kw, SLS_ROW_K + j, index, slots[1][j] * value);
            add_symmetric(third, kw, SLS_ROW_K + j, index, slots[2][j] * dq * value);
        }
        for (index, value) in hess_q_dir {
            add_symmetric(third, kw, SLS_ROW_K + j, index, slots[1][j] * value);
        }
    }
}

/// The tower of the bilinear `value = sign·(x_product·x8 − x_offset)`: `qdot0 =
/// x3·x8 − x5` is `(product, offset, sign) = (3, 5, 1)` and the time share
/// `tg = x2 − x1·x8` is `(1, 2, −1)`. Its one Hessian entry is `(x_product, x8) =
/// sign`, and every third and higher partial is zero.
#[inline(never)]
fn rate_index_tower(
    kw: usize,
    product: usize,
    offset: usize,
    p: &[f64; SLS_ROW_K],
    sign: f64,
    tower: &mut WiggleTower,
) {
    tower.reset(kw);
    let WiggleTower { grad, hess, .. } = tower;
    grad[product] = sign * p[8];
    grad[offset] = -sign;
    grad[8] = sign * p[product];
    add_symmetric(hess, kw, product, 8, sign);
}

/// The tower of `g = t + m·r` from the towers of the time share `t = tg`, `m = m1` and
/// `r = qdot0`, by the product rule through third order:
/// `∇³g[d] = ∇³t[d] + r·∇³m[d] + (∇r·d)·∇²m + (∇²m·d)∇rᵀ + ∇r(∇²m·d)ᵀ + (∇²r·d)∇mᵀ
/// + ∇m(∇²r·d)ᵀ + (∇m·d)·∇²r + m·∇³r[d]`.
#[inline(never)]
fn rate_tower(
    kw: usize,
    m: f64,
    r: f64,
    time_rate: &WiggleTower,
    multiplier: &WiggleTower,
    rate_index: &WiggleTower,
    dir: &[f64],
    multiplier_hess_dir: &mut Vec<f64>,
    rate_index_hess_dir: &mut Vec<f64>,
    tower: &mut WiggleTower,
) {
    tower.reset(kw);
    let dm = dot(&multiplier.grad, dir);
    let dr = dot(&rate_index.grad, dir);
    matvec(&multiplier.hess, kw, dir, multiplier_hess_dir);
    matvec(&rate_index.hess, kw, dir, rate_index_hess_dir);
    let WiggleTower { grad, hess, third } = tower;
    for a in 0..kw {
        let (ma, ra) = (multiplier.grad[a], rate_index.grad[a]);
        grad[a] = time_rate.grad[a] + r * ma + m * ra;
        for b in 0..kw {
            let index = a * kw + b;
            let (mb, rb) = (multiplier.grad[b], rate_index.grad[b]);
            hess[index] = time_rate.hess[index]
                + r * multiplier.hess[index]
                + ma * rb
                + ra * mb
                + m * rate_index.hess[index];
            third[index] = time_rate.third[index]
                + r * multiplier.third[index]
                + dr * multiplier.hess[index]
                + multiplier_hess_dir[a] * rb
                + ra * multiplier_hess_dir[b]
                + rate_index_hess_dir[a] * mb
                + ma * rate_index_hess_dir[b]
                + dm * rate_index.hess[index]
                + m * rate_index.third[index];
        }
    }
}

/// Add the direction derivative of `F″·∇u∇uᵀ + F′·∇²u` along `d`:
/// `F‴·du·∇u∇uᵀ + F″·(w∇uᵀ + ∇uwᵀ) + F″·du·∇²u + F′·∇³u[d]`, with `du = ∇u·d`,
/// `w = ∇²u·d` and `outer = (F′, F″, F‴)`.
#[inline(never)]
fn add_composition_third(
    out: &mut [f64],
    kw: usize,
    outer: [f64; 3],
    tower: &WiggleTower,
    dir: &[f64],
    hess_dir: &mut Vec<f64>,
) {
    let [first, second, third] = outer;
    let du = dot(&tower.grad, dir);
    matvec(&tower.hess, kw, dir, hess_dir);
    for a in 0..kw {
        for b in 0..kw {
            let index = a * kw + b;
            out[index] += third * du * tower.grad[a] * tower.grad[b]
                + second * (hess_dir[a] * tower.grad[b] + tower.grad[a] * hess_dir[b])
                + second * du * tower.hess[index]
                + first * tower.third[index];
        }
    }
}

/// The per-row directional third contraction `T[a][b] = Σ_c ∂³ℓ/∂a∂b∂c·d_c` of
/// `sls_row_nll_wiggle`, row-major into `scratch.third`, from the towers of `u0w`,
/// `u1w` and `g`. It skips inactive terms exactly as `hand_sls_wiggle_row_hessian`
/// does.
#[inline(never)]
fn hand_sls_wiggle_row_third(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    dir: &[f64],
    scratch: &mut HandWiggleThirdScratch,
) {
    let pw = betaw.len();
    let kw = SLS_ROW_K + pw;
    let entry_stack = [
        kernel.log_s0,
        -kernel.r0,
        -kernel.dr0,
        -kernel.ddr0,
        -kernel.dddr0,
    ];
    let exit_stack = [
        kernel.log_s1,
        -kernel.r1,
        -kernel.dr1,
        -kernel.ddr1,
        -kernel.dddr1,
    ];
    let pdf_stack = [
        kernel.logphi1,
        kernel.dlogphi1,
        kernel.d2logphi1,
        kernel.d3logphi1,
        kernel.d4logphi1,
    ];
    let rate_stack = [
        kernel.log_g,
        kernel.d_log_g,
        kernel.d2_log_g,
        kernel.d3_log_g,
        kernel.d4_log_g,
    ];
    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let entry_active = !stack_is_zero(&entry_stack);
    let censored_active = censored_weight != 0.0 && !stack_is_zero(&exit_stack);
    let pdf_active = event_weight != 0.0 && !stack_is_zero(&pdf_stack);
    let rate_active = event_weight != 0.0 && !stack_is_zero(&rate_stack);
    let entry_outer = [
        kernel.w * entry_stack[1],
        kernel.w * entry_stack[2],
        kernel.w * entry_stack[3],
    ];
    let mut exit_outer = [0.0; 3];
    for k in 0..3 {
        if censored_active {
            exit_outer[k] -= censored_weight * exit_stack[k + 1];
        }
        if pdf_active {
            exit_outer[k] -= event_weight * pdf_stack[k + 1];
        }
    }
    let rate_outer = [
        -event_weight * rate_stack[1],
        -event_weight * rate_stack[2],
        -event_weight * rate_stack[3],
    ];

    let s7 = (-p[7]).exp();
    let q0 = -p[4] * s7;
    let s6 = (-p[6]).exp();
    let q1 = -p[3] * s6;
    let qdot0 = p[3] * p[8] - p[5];
    let t0 = p[0] * s7;
    let t1 = p[1] * s6;
    let (b0, b1) = (basis.b_u0, basis.b_u1);
    // `∂u0w/∂q0` through `∂³u0w/∂q0³`, the same for `u1w` over `q1`, and
    // `∂m1/∂q1` through `∂³m1/∂q1³`; the value of `m1` equals `∂u1w/∂q1`.
    let mut entry_derivs = [1.0, 0.0, 0.0];
    let mut exit_derivs = [1.0, 0.0, 0.0];
    let mut multiplier_derivs = [0.0; 3];
    for j in 0..pw {
        for k in 0..3 {
            entry_derivs[k] += betaw[j] * b0[k + 1][j];
            exit_derivs[k] += betaw[j] * b1[k + 1][j];
            multiplier_derivs[k] += betaw[j] * b1[k + 2][j];
        }
    }

    let HandWiggleThirdScratch {
        entry,
        exit,
        multiplier,
        rate_index,
        time_rate,
        rate,
        hess_dir,
        multiplier_hess_dir,
        rate_index_hess_dir,
        third,
    } = scratch;
    third.clear();
    third.resize(kw * kw, 0.0);
    if entry_active {
        warp_tower(
            kw,
            Some((0, t0)),
            4,
            7,
            s7,
            q0,
            entry_derivs,
            [b0[0], b0[1], b0[2]],
            dir,
            entry,
        );
        add_composition_third(third, kw, entry_outer, entry, dir, hess_dir);
    }
    if censored_active || pdf_active {
        warp_tower(
            kw,
            Some((1, t1)),
            3,
            6,
            s6,
            q1,
            exit_derivs,
            [b1[0], b1[1], b1[2]],
            dir,
            exit,
        );
        add_composition_third(third, kw, exit_outer, exit, dir, hess_dir);
    }
    if rate_active {
        warp_tower(
            kw,
            None,
            3,
            6,
            s6,
            q1,
            multiplier_derivs,
            [b1[1], b1[2], b1[3]],
            dir,
            multiplier,
        );
        rate_index_tower(kw, 3, 5, p, 1.0, rate_index);
        rate_index_tower(kw, 1, 2, p, -1.0, time_rate);
        rate_tower(
            kw,
            exit_derivs[0],
            qdot0,
            time_rate,
            multiplier,
            rate_index,
            dir,
            multiplier_hess_dir,
            rate_index_hess_dir,
            rate,
        );
        add_composition_third(third, kw, rate_outer, rate, dir, hess_dir);
    }
}

/// Production's per-row directional third lowering, as
/// `SurvivalLsWiggleRowKernel::row_third_contracted` runs it.
fn production_row_third(
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

/// #932 row 58, stage 2: the hand directional third contraction equals production's
/// `DynamicOneSeed` lowering on every entry, at runtime widths 3 and 7, on an event
/// row, a censored row and an untruncated event row, along a direction with every
/// primary live. It uses the band and the measure-then-assert structure of the
/// Hessian test. The control corrupts the exit stack's third slot, which no
/// order-two quantity reads.
#[test]
fn hand_sls_wiggle_row_third_matches_production_jet_932() {
    let band = |a: f64, b: f64| 1e-11 * a.abs().max(b.abs()).max(1.0);
    let mut arena = DynamicJetArena::new();
    let mut scratch = HandWiggleThirdScratch::new();
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
            let production = production_row_third(&p, &betaw, &kernel, &basis, &dir, &mut arena);
            assert_eq!(production.len(), kw * kw);
            hand_sls_wiggle_row_third(&p, &betaw, &kernel, &basis, &dir, &mut scratch);
            for a in 0..kw {
                for b in 0..kw {
                    let want = production[a * kw + b];
                    let got = scratch.third[a * kw + b];
                    let over = (want - got).abs() / band(want, got);
                    if !(over <= worst_over_band) {
                        worst_over_band = over;
                    }
                    if !(over <= 1.0) {
                        failures.push(format!(
                            "pw={pw} {label} T[{a}][{b}]: production {want:+.15e} hand {got:+.15e}"
                        ));
                    }
                }
            }
            if label == "censored" {
                let mut corrupted = kernel;
                corrupted.ddr1 *= 1.0 + 1e-6;
                hand_sls_wiggle_row_third(&p, &betaw, &corrupted, &basis, &dir, &mut scratch);
                for a in 0..kw {
                    for b in 0..kw {
                        let want = production[a * kw + b];
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
fn production_order3_checksum(
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
fn hand_order3_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    dir: &[f64],
    scratch: &mut HandWiggleThirdScratch,
) -> f64 {
    hand_sls_wiggle_row_third(p, betaw, kernel, basis, dir, scratch);
    scratch
        .third
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}

/// One intermediate's towers along both directions, plus its fourth derivative
/// contracted with `u` and `v`. The two order-3 towers repeat the gradient and
/// Hessian work, which a tighter hand schedule would share, so the order-4 race
/// is against this schedule, not a proven optimum.
struct TowerPair {
    along_u: WiggleTower,
    along_v: WiggleTower,
    fourth: Vec<f64>,
}

impl TowerPair {
    fn new() -> Self {
        Self {
            along_u: WiggleTower::new(),
            along_v: WiggleTower::new(),
            fourth: Vec::new(),
        }
    }
}

/// Work vectors for the order-4 assembly.
struct FourthWork {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    e: Vec<f64>,
    f: Vec<f64>,
}

impl FourthWork {
    fn new() -> Self {
        Self {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            e: Vec::new(),
            f: Vec::new(),
        }
    }
}

/// Reusable buffers for the hand second-directional fourth contraction.
struct HandWiggleFourthScratch {
    entry: TowerPair,
    exit: TowerPair,
    multiplier: TowerPair,
    rate_index: TowerPair,
    time_rate: TowerPair,
    rate: TowerPair,
    work: FourthWork,
    fourth: Vec<f64>,
}

impl HandWiggleFourthScratch {
    fn new() -> Self {
        Self {
            entry: TowerPair::new(),
            exit: TowerPair::new(),
            multiplier: TowerPair::new(),
            rate_index: TowerPair::new(),
            time_rate: TowerPair::new(),
            rate: TowerPair::new(),
            work: FourthWork::new(),
            fourth: Vec::new(),
        }
    }
}

/// `∇⁴U[u, v]` for `U = x_linear + G(q, βw)`, `q = −x_numerator·e^{−x_log_scale}`, with
/// `derivs = (G_q, G_qq, G_qqq, G_qqqq)` and `slots[k][j] = ∂^{k+1}G/∂q^k∂βw_j`. The
/// partials of `q` are `q_n = −s`, `q_l = −q`, `q_nl = s`, `q_ll = q`, `q_nll = −s`,
/// `q_lll = −q`, `q_nlll = s`, `q_llll = q`. Any partial with two `n` derivatives is zero,
/// and so is any with two `βw` derivatives. Over the primaries `a, b`, with
/// `du = ∇q·u`, `dv = ∇q·v`, `h_u = ∇²q·u`, `h_v = ∇²q·v`, `c = uᵀ∇²q v` and
/// `r = ∇³q[u]·v`, the Faà di Bruno terms are:
/// - `(G_qqqq·du·dv + G_qqq·c + G_qqqβ·v·du + G_qqqβ·u·dv)·∇q∇qᵀ`;
/// - `(G_qqq·du·dv + G_qq·c + G_qqβ·v·du + G_qqβ·u·dv)·∇²q`;
/// - `(G_qqq·dv + G_qqβ·v)·(h_u∇qᵀ + ∇qh_uᵀ) + (G_qqq·du + G_qqβ·u)·(h_v∇qᵀ + ∇qh_vᵀ)`;
/// - `G_qq·(h_uh_vᵀ + h_vh_uᵀ + r∇qᵀ + ∇qrᵀ)`;
/// - `(G_qq·dv + G_qβ·v)·∇³q[u] + (G_qq·du + G_qβ·u)·∇³q[v]`;
/// - `G_q·∇⁴q[u, v]`;
/// - the `(x, βw_j)` blocks `G_qqqβj·du·dv·∇q + G_qqβj·(dv·h_u + du·h_v + c·∇q) + G_qβj·r`.
#[inline(never)]
fn warp_fourth(
    kw: usize,
    numerator: usize,
    log_scale: usize,
    scale: f64,
    q: f64,
    derivs: [f64; 4],
    slots: [&[f64]; 4],
    u: &[f64],
    v: &[f64],
    fourth: &mut Vec<f64>,
) {
    fourth.clear();
    fourth.resize(kw * kw, 0.0);
    let pw = kw - SLS_ROW_K;
    let [g1, g2, g3, g4] = derivs;
    let (n, l) = (numerator, log_scale);
    let grad_q = [(n, -scale), (l, -q)];
    let du = -scale * u[n] - q * u[l];
    let dv = -scale * v[n] - q * v[l];
    let hess_q_u = [(n, scale * u[l]), (l, scale * u[n] + q * u[l])];
    let hess_q_v = [(n, scale * v[l]), (l, scale * v[n] + q * v[l])];
    let cross = scale * (u[n] * v[l] + u[l] * v[n]) + q * u[l] * v[l];
    let third_q_uv = [(n, -scale * u[l] * v[l]), (l, -cross)];
    let (mut s1u, mut s2u, mut s3u) = (0.0, 0.0, 0.0);
    let (mut s1v, mut s2v, mut s3v) = (0.0, 0.0, 0.0);
    for j in 0..pw {
        let (along_u, along_v) = (u[SLS_ROW_K + j], v[SLS_ROW_K + j]);
        s1u += slots[1][j] * along_u;
        s2u += slots[2][j] * along_u;
        s3u += slots[3][j] * along_u;
        s1v += slots[1][j] * along_v;
        s2v += slots[2][j] * along_v;
        s3v += slots[3][j] * along_v;
    }
    let outer = g4 * du * dv + g3 * cross + s3v * du + s3u * dv;
    for (a, left) in grad_q {
        for (b, right) in grad_q {
            fourth[a * kw + b] += outer * left * right;
        }
    }
    let curvature = g3 * du * dv + g2 * cross + s2v * du + s2u * dv;
    add_symmetric(fourth, kw, n, l, curvature * scale);
    add_symmetric(fourth, kw, l, l, curvature * q);
    for (a, left) in hess_q_u {
        for (b, right) in grad_q {
            add_pair(fourth, kw, a, b, (g3 * dv + s2v) * left * right);
        }
    }
    for (a, left) in hess_q_v {
        for (b, right) in grad_q {
            add_pair(fourth, kw, a, b, (g3 * du + s2u) * left * right);
        }
    }
    for (a, left) in hess_q_u {
        for (b, right) in hess_q_v {
            add_pair(fourth, kw, a, b, g2 * left * right);
        }
    }
    for (a, left) in third_q_uv {
        for (b, right) in grad_q {
            add_pair(fourth, kw, a, b, g2 * left * right);
        }
    }
    // ∇³q[u] = (−s·u_l at (n, l), du at (l, l)), and ∇³q[v] alike.
    let (weight_u, weight_v) = (g2 * dv + s1v, g2 * du + s1u);
    add_symmetric(fourth, kw, n, l, -scale * (weight_u * u[l] + weight_v * v[l]));
    add_symmetric(fourth, kw, l, l, weight_u * du + weight_v * dv);
    // ∇⁴q[u, v] = (s·u_l·v_l at (n, l), c at (l, l)).
    add_symmetric(fourth, kw, n, l, g1 * scale * u[l] * v[l]);
    add_symmetric(fourth, kw, l, l, g1 * cross);
    for j in 0..pw {
        let column = SLS_ROW_K + j;
        for (a, value) in grad_q {
            add_symmetric(
                fourth,
                kw,
                column,
                a,
                (slots[3][j] * du * dv + slots[2][j] * cross) * value,
            );
        }
        for (a, value) in hess_q_u {
            add_symmetric(fourth, kw, column, a, slots[2][j] * dv * value);
        }
        for (a, value) in hess_q_v {
            add_symmetric(fourth, kw, column, a, slots[2][j] * du * value);
        }
        for (a, value) in third_q_uv {
            add_symmetric(fourth, kw, column, a, slots[1][j] * value);
        }
    }
}

/// The warp towers along `u` and `v` and their fourth contraction. The time share
/// `t = x_index·s` adds `∇⁴t[u, v] = (−s·u_l·v_l at (index, l), t·u_l·v_l −
/// s·(u_index·v_l + u_l·v_index) at (l, l))` with `l = log_scale`.
#[inline(never)]
fn warp_pair(
    kw: usize,
    linear: Option<(usize, f64)>,
    numerator: usize,
    log_scale: usize,
    scale: f64,
    q: f64,
    derivs: [f64; 4],
    slots: [&[f64]; 4],
    u: &[f64],
    v: &[f64],
    pair: &mut TowerPair,
) {
    let lower_derivs = [derivs[0], derivs[1], derivs[2]];
    let lower_slots = [slots[0], slots[1], slots[2]];
    warp_tower(kw, linear, numerator, log_scale, scale, q, lower_derivs, lower_slots, u, &mut pair.along_u);
    warp_tower(kw, linear, numerator, log_scale, scale, q, lower_derivs, lower_slots, v, &mut pair.along_v);
    warp_fourth(kw, numerator, log_scale, scale, q, derivs, slots, u, v, &mut pair.fourth);
    if let Some((index, time)) = linear {
        let l = log_scale;
        add_symmetric(&mut pair.fourth, kw, index, l, -scale * u[l] * v[l]);
        add_symmetric(
            &mut pair.fourth,
            kw,
            l,
            l,
            time * u[l] * v[l] - scale * (u[index] * v[l] + u[l] * v[index]),
        );
    }
}

/// The towers of the bilinear `value = sign·(x_product·x8 − x_offset)` along `u` and
/// `v` (`rate_index_tower` names the two uses), whose fourth contraction is zero.
#[inline(never)]
fn rate_index_pair(
    kw: usize,
    product: usize,
    offset: usize,
    p: &[f64; SLS_ROW_K],
    sign: f64,
    pair: &mut TowerPair,
) {
    rate_index_tower(kw, product, offset, p, sign, &mut pair.along_u);
    rate_index_tower(kw, product, offset, p, sign, &mut pair.along_v);
    pair.fourth.clear();
    pair.fourth.resize(kw * kw, 0.0);
}

/// The towers of `g = t + m·r` along `u` and `v`, and its fourth contraction: `∇⁴t[u, v]`
/// plus the sixteen-term product rule, where each of the four indices `a, b, u, v` falls
/// on `m` or `r`:
/// `r·∇⁴m + (∇r·v)·∇³m[u] + (∇r·u)·∇³m[v] + (uᵀ∇²r v)·∇²m + (∇³m[u]v)∇rᵀ + ∇r(∇³m[u]v)ᵀ
/// + (∇²m u)(∇²r v)ᵀ + (∇²r v)(∇²m u)ᵀ + (∇²m v)(∇²r u)ᵀ + (∇²r u)(∇²m v)ᵀ
/// + ∇m(∇³r[u]v)ᵀ + (∇³r[u]v)∇mᵀ + (uᵀ∇²m v)·∇²r + (∇m·u)·∇³r[v] + (∇m·v)·∇³r[u] + m·∇⁴r`.
#[inline(never)]
fn rate_pair(
    kw: usize,
    m: f64,
    r: f64,
    time_rate: &TowerPair,
    multiplier: &TowerPair,
    rate_index: &TowerPair,
    u: &[f64],
    v: &[f64],
    work: &mut FourthWork,
    pair: &mut TowerPair,
) {
    rate_tower(
        kw,
        m,
        r,
        &time_rate.along_u,
        &multiplier.along_u,
        &rate_index.along_u,
        u,
        &mut work.a,
        &mut work.b,
        &mut pair.along_u,
    );
    rate_tower(
        kw,
        m,
        r,
        &time_rate.along_v,
        &multiplier.along_v,
        &rate_index.along_v,
        v,
        &mut work.a,
        &mut work.b,
        &mut pair.along_v,
    );
    let (m_grad, m_hess) = (&multiplier.along_u.grad, &multiplier.along_u.hess);
    let (r_grad, r_hess) = (&rate_index.along_u.grad, &rate_index.along_u.hess);
    let (m_third_u, m_third_v) = (&multiplier.along_u.third, &multiplier.along_v.third);
    let (r_third_u, r_third_v) = (&rate_index.along_u.third, &rate_index.along_v.third);
    let (m_fourth, r_fourth) = (&multiplier.fourth, &rate_index.fourth);
    let (m_u, m_v) = (dot(m_grad, u), dot(m_grad, v));
    let (r_u, r_v) = (dot(r_grad, u), dot(r_grad, v));
    matvec(m_hess, kw, u, &mut work.a);
    matvec(m_hess, kw, v, &mut work.b);
    matvec(r_hess, kw, u, &mut work.c);
    matvec(r_hess, kw, v, &mut work.d);
    matvec(m_third_u, kw, v, &mut work.e);
    matvec(r_third_u, kw, v, &mut work.f);
    let m_uv = dot(&work.a, v);
    let r_uv = dot(&work.c, v);
    let fourth = &mut pair.fourth;
    fourth.clear();
    fourth.resize(kw * kw, 0.0);
    for a in 0..kw {
        for b in 0..kw {
            let index = a * kw + b;
            fourth[index] = time_rate.fourth[index]
                + r * m_fourth[index]
                + r_v * m_third_u[index]
                + r_u * m_third_v[index]
                + r_uv * m_hess[index]
                + work.e[a] * r_grad[b]
                + r_grad[a] * work.e[b]
                + work.a[a] * work.d[b]
                + work.d[a] * work.a[b]
                + work.b[a] * work.c[b]
                + work.c[a] * work.b[b]
                + m_grad[a] * work.f[b]
                + work.f[a] * m_grad[b]
                + m_uv * r_hess[index]
                + m_u * r_third_v[index]
                + m_v * r_third_u[index]
                + m * r_fourth[index];
        }
    }
}

/// Add `∂/∂v` of the composition's order-3 contribution along `u`. With `a = ∇U·u`,
/// `b = ∇U·v`, `c = uᵀ∇²U v`, `p = ∇²U·u`, `q = ∇²U·v`, `r = ∇³U[u]·v` and
/// `outer = (F′, F″, F‴, F⁗)`, it adds
/// `(F⁗·a·b + F‴·c)·∇U∇Uᵀ + F‴·a·(q∇Uᵀ + ∇Uqᵀ) + F‴·b·(p∇Uᵀ + ∇Upᵀ) + F″·(r∇Uᵀ + ∇Urᵀ)
/// + F″·(pqᵀ + qpᵀ) + (F‴·a·b + F″·c)·∇²U + F″·a·∇³U[v] + F″·b·∇³U[u] + F′·∇⁴U[u, v]`.
#[inline(never)]
fn add_composition_fourth(
    out: &mut [f64],
    kw: usize,
    outer: [f64; 4],
    pair: &TowerPair,
    u: &[f64],
    v: &[f64],
    work: &mut FourthWork,
) {
    let [first, second, third, fourth] = outer;
    let (grad, hess) = (&pair.along_u.grad, &pair.along_u.hess);
    let (third_u, third_v) = (&pair.along_u.third, &pair.along_v.third);
    let along_u = dot(grad, u);
    let along_v = dot(grad, v);
    matvec(hess, kw, u, &mut work.a);
    matvec(hess, kw, v, &mut work.b);
    matvec(third_u, kw, v, &mut work.c);
    let cross = dot(&work.a, v);
    for a in 0..kw {
        for b in 0..kw {
            let index = a * kw + b;
            let (grad_a, grad_b) = (grad[a], grad[b]);
            out[index] += (fourth * along_u * along_v + third * cross) * grad_a * grad_b
                + third * along_u * (work.b[a] * grad_b + grad_a * work.b[b])
                + third * along_v * (work.a[a] * grad_b + grad_a * work.a[b])
                + second * (work.c[a] * grad_b + grad_a * work.c[b])
                + second * (work.a[a] * work.b[b] + work.b[a] * work.a[b])
                + (third * along_u * along_v + second * cross) * hess[index]
                + second * along_u * third_v[index]
                + second * along_v * third_u[index]
                + first * pair.fourth[index];
        }
    }
}

/// The per-row second-directional fourth contraction
/// `T[a][b] = Σ_{c,d} ∂⁴ℓ/∂a∂b∂c∂d·u_c·v_d` of `sls_row_nll_wiggle`, row-major into
/// `scratch.fourth`. It skips inactive terms exactly as `hand_sls_wiggle_row_hessian` does.
#[inline(never)]
fn hand_sls_wiggle_row_fourth(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    u: &[f64],
    v: &[f64],
    scratch: &mut HandWiggleFourthScratch,
) {
    let pw = betaw.len();
    let kw = SLS_ROW_K + pw;
    let entry_stack = [
        kernel.log_s0,
        -kernel.r0,
        -kernel.dr0,
        -kernel.ddr0,
        -kernel.dddr0,
    ];
    let exit_stack = [
        kernel.log_s1,
        -kernel.r1,
        -kernel.dr1,
        -kernel.ddr1,
        -kernel.dddr1,
    ];
    let pdf_stack = [
        kernel.logphi1,
        kernel.dlogphi1,
        kernel.d2logphi1,
        kernel.d3logphi1,
        kernel.d4logphi1,
    ];
    let rate_stack = [
        kernel.log_g,
        kernel.d_log_g,
        kernel.d2_log_g,
        kernel.d3_log_g,
        kernel.d4_log_g,
    ];
    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let entry_active = !stack_is_zero(&entry_stack);
    let censored_active = censored_weight != 0.0 && !stack_is_zero(&exit_stack);
    let pdf_active = event_weight != 0.0 && !stack_is_zero(&pdf_stack);
    let rate_active = event_weight != 0.0 && !stack_is_zero(&rate_stack);
    let entry_outer: [f64; 4] = std::array::from_fn(|k| kernel.w * entry_stack[k + 1]);
    let mut exit_outer = [0.0; 4];
    for k in 0..4 {
        if censored_active {
            exit_outer[k] -= censored_weight * exit_stack[k + 1];
        }
        if pdf_active {
            exit_outer[k] -= event_weight * pdf_stack[k + 1];
        }
    }
    let rate_outer: [f64; 4] = std::array::from_fn(|k| -event_weight * rate_stack[k + 1]);

    let s7 = (-p[7]).exp();
    let q0 = -p[4] * s7;
    let s6 = (-p[6]).exp();
    let q1 = -p[3] * s6;
    let qdot0 = p[3] * p[8] - p[5];
    let t0 = p[0] * s7;
    let t1 = p[1] * s6;
    let (b0, b1) = (basis.b_u0, basis.b_u1);
    let mut entry_derivs = [1.0, 0.0, 0.0, 0.0];
    let mut exit_derivs = [1.0, 0.0, 0.0, 0.0];
    let mut multiplier_derivs = [0.0; 4];
    for j in 0..pw {
        for k in 0..4 {
            entry_derivs[k] += betaw[j] * b0[k + 1][j];
            exit_derivs[k] += betaw[j] * b1[k + 1][j];
            multiplier_derivs[k] += betaw[j] * b1[k + 2][j];
        }
    }

    let HandWiggleFourthScratch {
        entry,
        exit,
        multiplier,
        rate_index,
        time_rate,
        rate,
        work,
        fourth,
    } = scratch;
    fourth.clear();
    fourth.resize(kw * kw, 0.0);
    if entry_active {
        warp_pair(
            kw,
            Some((0, t0)),
            4,
            7,
            s7,
            q0,
            entry_derivs,
            [b0[0], b0[1], b0[2], b0[3]],
            u,
            v,
            entry,
        );
        add_composition_fourth(fourth, kw, entry_outer, entry, u, v, work);
    }
    if censored_active || pdf_active {
        warp_pair(
            kw,
            Some((1, t1)),
            3,
            6,
            s6,
            q1,
            exit_derivs,
            [b1[0], b1[1], b1[2], b1[3]],
            u,
            v,
            exit,
        );
        add_composition_fourth(fourth, kw, exit_outer, exit, u, v, work);
    }
    if rate_active {
        warp_pair(
            kw,
            None,
            3,
            6,
            s6,
            q1,
            multiplier_derivs,
            [b1[1], b1[2], b1[3], b1[4]],
            u,
            v,
            multiplier,
        );
        rate_index_pair(kw, 3, 5, p, 1.0, rate_index);
        rate_index_pair(kw, 1, 2, p, -1.0, time_rate);
        rate_pair(
            kw,
            exit_derivs[0],
            qdot0,
            time_rate,
            multiplier,
            rate_index,
            u,
            v,
            work,
            rate,
        );
        add_composition_fourth(fourth, kw, rate_outer, rate, u, v, work);
    }
}

/// Production's per-row second-directional fourth lowering, as
/// `SurvivalLsWiggleRowKernel::row_fourth_contracted` runs it.
fn production_row_fourth(
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

/// #932 row 58, stage 3: the hand second-directional fourth contraction equals
/// production's `DynamicTwoSeed` lowering on every entry, at runtime widths 3 and 7,
/// on an event row, a censored row and an untruncated event row, along two directions
/// with every primary live. It uses the band and the measure-then-assert structure of
/// the lower orders. The control corrupts the exit stack's fourth slot, which no
/// lower-order quantity reads.
#[test]
fn hand_sls_wiggle_row_fourth_matches_production_jet_932() {
    let band = |a: f64, b: f64| 1e-11 * a.abs().max(b.abs()).max(1.0);
    let mut arena = DynamicJetArena::new();
    let mut scratch = HandWiggleFourthScratch::new();
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
            let production = production_row_fourth(&p, &betaw, &kernel, &basis, &u, &v, &mut arena);
            assert_eq!(production.len(), kw * kw);
            hand_sls_wiggle_row_fourth(&p, &betaw, &kernel, &basis, &u, &v, &mut scratch);
            for a in 0..kw {
                for b in 0..kw {
                    let want = production[a * kw + b];
                    let got = scratch.fourth[a * kw + b];
                    let over = (want - got).abs() / band(want, got);
                    if !(over <= worst_over_band) {
                        worst_over_band = over;
                    }
                    if !(over <= 1.0) {
                        failures.push(format!(
                            "pw={pw} {label} F[{a}][{b}]: production {want:+.15e} hand {got:+.15e}"
                        ));
                    }
                }
            }
            if label == "censored" {
                let mut corrupted = kernel;
                corrupted.dddr1 *= 1.0 + 1e-6;
                hand_sls_wiggle_row_fourth(&p, &betaw, &corrupted, &basis, &u, &v, &mut scratch);
                for a in 0..kw {
                    for b in 0..kw {
                        let want = production[a * kw + b];
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
fn production_order4_checksum(
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
fn hand_order4_checksum(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    u: &[f64],
    v: &[f64],
    scratch: &mut HandWiggleFourthScratch,
) -> f64 {
    hand_sls_wiggle_row_fourth(p, betaw, kernel, basis, u, v, scratch);
    scratch
        .fourth
        .iter()
        .enumerate()
        .fold(0.0, |acc, (index, value)| acc + value * (1.0 + index as f64 * 1e-3))
}
