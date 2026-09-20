//! The runtime-width survival location-scale link-wiggle row derivatives (#932,
//! #3319): the per-row gradient and Hessian, the directional third contraction and
//! the second-directional fourth contraction of [`sls_row_nll_wiggle`] at width
//! `KW = SLS_ROW_K + pw`, written out by hand. [`SurvivalLsWiggleRowKernel`] folds
//! these into the joint Hessian, its directional derivatives and the ψ terms.
//!
//! SPEC rule 1 admits a forward-mode lowering in production only where it is
//! verified to match or surpass hand-derived speed. The packed dynamic jets over
//! [`sls_row_nll_wiggle`] carry all `KW²` channels through every operation and ran
//! 7–30× slower than this schedule (#3319), so the jet program is the test oracle
//! (`sls_wiggle_hand_932_tests` holds the parity and the release race) and this
//! schedule is production.
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
//! `−x6` enters the event log-density linearly and so has no curvature.
//!
//! Each intermediate reads a few primaries and is linear in `βw`, so the Hessian is
//! three rank-one outer products plus the sparse intermediate Hessians, and the
//! third and fourth contractions follow by Faà di Bruno over the same towers.

use super::*;

/// Reusable per-fold buffers for [`sls_wiggle_row_order2`]. `gradient` (length
/// `KW`) and `hessian` (`KW × KW`, row-major) hold the last row's output; the
/// three intermediate gradients are working storage.
pub(crate) struct SlsWiggleOrder2Scratch {
    grad_entry: Vec<f64>,
    grad_exit: Vec<f64>,
    grad_rate: Vec<f64>,
    pub(crate) gradient: Vec<f64>,
    pub(crate) hessian: Vec<f64>,
}

impl SlsWiggleOrder2Scratch {
    pub(crate) fn new() -> Self {
        Self {
            grad_entry: Vec::new(),
            grad_exit: Vec::new(),
            grad_rate: Vec::new(),
            gradient: Vec::new(),
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

/// `out += scale·v`.
fn add_scaled(out: &mut [f64], scale: f64, v: &[f64]) {
    for (target, value) in out.iter_mut().zip(v) {
        *target += scale * value;
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

/// The per-row gradient (`scratch.gradient`) and `KW × KW` Hessian
/// (`scratch.hessian`, row-major) of [`sls_row_nll_wiggle`], by the structure the
/// module doc describes. An exactly-zero outer stack is skipped instead of composed
/// (the #2342 far-tail `0·∞` guard), exactly as the jet program skips it, so an
/// inactive term contributes nothing.
pub(crate) fn sls_wiggle_row_order2(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    scratch: &mut SlsWiggleOrder2Scratch,
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

    let SlsWiggleOrder2Scratch {
        grad_entry,
        grad_exit,
        grad_rate,
        gradient,
        hessian,
    } = scratch;
    for buffer in [
        &mut *grad_entry,
        &mut *grad_exit,
        &mut *grad_rate,
        &mut *gradient,
    ] {
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
        add_scaled(gradient, f0_first, grad_entry);
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
        add_scaled(gradient, f1_first, grad_exit);
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
        add_scaled(gradient, f2_first, grad_rate);
        // `log(du1/dt) = log g − x6` puts `event_weight·x6` in the NLL (#2695).
        gradient[6] += event_weight;
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

/// Reusable per-fold buffers for [`sls_wiggle_row_third`]; `third` (`KW × KW`,
/// row-major) holds the last row's output.
pub(crate) struct SlsWiggleThirdScratch {
    entry: WiggleTower,
    exit: WiggleTower,
    multiplier: WiggleTower,
    rate_index: WiggleTower,
    time_rate: WiggleTower,
    rate: WiggleTower,
    hess_dir: Vec<f64>,
    multiplier_hess_dir: Vec<f64>,
    rate_index_hess_dir: Vec<f64>,
    pub(crate) third: Vec<f64>,
}

impl SlsWiggleThirdScratch {
    pub(crate) fn new() -> Self {
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
/// `u1w` and `g`. It skips inactive terms exactly as `sls_wiggle_row_order2`
/// does.
pub(crate) fn sls_wiggle_row_third(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    dir: &[f64],
    scratch: &mut SlsWiggleThirdScratch,
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

    let SlsWiggleThirdScratch {
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

/// One intermediate's towers along both directions, plus its fourth derivative
/// contracted with `u` and `v`. The two order-3 towers repeat the gradient and
/// Hessian work; a schedule that shared it would be cheaper still.
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

/// Reusable per-fold buffers for [`sls_wiggle_row_fourth`]; `fourth` (`KW × KW`,
/// row-major) holds the last row's output.
pub(crate) struct SlsWiggleFourthScratch {
    entry: TowerPair,
    exit: TowerPair,
    multiplier: TowerPair,
    rate_index: TowerPair,
    time_rate: TowerPair,
    rate: TowerPair,
    work: FourthWork,
    pub(crate) fourth: Vec<f64>,
}

impl SlsWiggleFourthScratch {
    pub(crate) fn new() -> Self {
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
/// `scratch.fourth`. It skips inactive terms exactly as `sls_wiggle_row_order2` does.
pub(crate) fn sls_wiggle_row_fourth(
    p: &[f64; SLS_ROW_K],
    betaw: &[f64],
    kernel: &SurvivalExactRowKernel,
    basis: &SlsWiggleRowBasis<'_>,
    u: &[f64],
    v: &[f64],
    scratch: &mut SlsWiggleFourthScratch,
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

    let SlsWiggleFourthScratch {
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
