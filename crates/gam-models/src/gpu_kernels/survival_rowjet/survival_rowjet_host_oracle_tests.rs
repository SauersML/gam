//! Host transcription of the survival row-jet **device** seeded-jet arithmetic
//! (#415 parity-lock).
//!
//! The CUDA kernel [`super::SURVIVAL_ROWJET_SOURCE`]
//! (`survival_rowjet_kernel.cu`) is a hand-unrolled, K=4 port of the CPU
//! `rigid_row_nll` seeded jets — the `JS1`/`JS2` structs and the `DEF_NLL`
//! macro. That hand-port is exactly where the #736 cross-block sign-flip bug
//! genus lives: a wrong sign or a dropped Leibniz term in the `.cu` JS2
//! `huv` recurrence produces a kernel that compiles fine, runs fine, and is
//! silently wrong. The only existing guard against that drift is
//! [`super::tests::device_matches_cpu_when_available`], which can ONLY compare
//! the two paths when a CUDA device is actually present. On every CPU-only box
//! (all of CI except the GPU runner) that test degenerates to `CPU == CPU` and
//! the device algebra ships unverified.
//!
//! #415 asked for a CI **parity-lock**: shared fixtures asserting
//! `CPU formula == device kernel` for every formula variant and contraction
//! order, wired so drift fails the build — on every box, not just the GPU box.
//! The device kernel body cannot be linked on a CPU box (it is NVRTC device
//! code), so this module supplies the missing third leg: an INDEPENDENT host
//! transcription of the device JS1/JS2 + `DEF_NLL` arithmetic, written here in
//! Rust with the SAME hand-unrolled K=4 structure and SAME operation order as
//! the `.cu`, and a parity sweep that pins it to the production CPU
//! `rigid_row_nll` across formula variants × contraction directions × edge
//! inputs (deep probit tails, near-degenerate covariance, censored vs.
//! event rows, zero weight).
//!
//! The chain that makes this a true CPU↔device lock:
//!
//! ```text
//!   production CPU jet  ==(this module, every box)==  HOST device-oracle
//!            ‖ (super::tests::device_matches_cpu_when_available, GPU box only)
//!     device .cu kernel
//! ```
//!
//! The host oracle is a transcription of the `.cu`, so when the `.cu` JS1/JS2
//! algebra is edited the transcription must be edited in lockstep to keep this
//! test green — and any edit that breaks the device math (without a matching
//! CPU-jet change) is caught HERE, on the CPU build, instead of slipping
//! through to the GPU box. A coarse byte-fingerprint test
//! ([`device_kernel_source_matches_transcription_fingerprint`]) makes the
//! lockstep requirement explicit: touching the `.cu` arithmetic without
//! revisiting this transcription fails the build with a pointer to this file.

#![cfg(test)]

use super::{SurvivalRowInputs, survival_rigid_row_jets_cpu};

const K: usize = 4;

// ────────────────────────────────────────────────────────────────────────
// Transcendental scalar primitives.
//
// These mirror the `__device__` transcendental helpers in
// `survival_rowjet_kernel.cu` (`sp_logcdf_mills`, `neglog_phi_stack`, `d_sqrt`,
// `d_log`, `d_lognormpdf`). The scalar special functions themselves
// (`erfcx`, `normal_cdf`/`pdf`, the Mills-ratio reduction) are NOT re-derived
// here: they route through the SAME public `gam_math::probability` functions
// the production CPU jet and the `.cu` both target, so this test isolates
// DRIFT IN THE JET ALGEBRA (the JS1/JS2 Leibniz recurrences and the `DEF_NLL`
// term assembly — where the #736 sign-flip genus lives) rather than chasing
// ULP noise between two re-implementations of `erfc`. The `.cu`
// `sp_logcdf_mills` is itself a hand-port of
// [`gam_math::probability::signed_probit_logcdf_and_mills_ratio`].
// ────────────────────────────────────────────────────────────────────────

/// Host mirror of `.cu` `neglog_phi_stack` — the signed-probit value+derivative
/// stack `[-w·logΦ(m), w·k1, w·k2, w·k3, w·k4]`. Routes the Mills-ratio
/// transcendental through the production
/// [`gam_math::probability::signed_probit_logcdf_and_mills_ratio`] (what the CPU
/// jet's `signed_probit_neglog_unary_stack` calls), and reproduces the SAME
/// `k1..k4` polynomial closed forms the `.cu` writes.
fn neglog_phi_stack(m: f64, w: f64) -> [f64; 5] {
    if w == 0.0 || m == f64::INFINITY {
        return [0.0; 5];
    }
    if m == f64::NEG_INFINITY {
        return [f64::INFINITY, f64::NEG_INFINITY, w, 0.0, 0.0];
    }
    if m.is_nan() {
        return [f64::NAN; 5];
    }
    let (lc, lam) = gam_math::probability::signed_probit_logcdf_and_mills_ratio(m);
    let k1 = -lam;
    let k2 = lam * (m + lam);
    let k3 = lam * (1.0 - m * m - 3.0 * m * lam - 2.0 * lam * lam);
    let k4 = lam
        * ((m * m * m - 3.0 * m)
            + (7.0 * m * m - 4.0) * lam
            + 12.0 * m * lam * lam
            + 6.0 * lam * lam * lam);
    [-w * lc, w * k1, w * k2, w * k3, w * k4]
}

fn d_sqrt(x: f64) -> [f64; 5] {
    let xa = x.max(1e-300);
    let s = xa.sqrt();
    let x2 = xa * xa;
    let x3 = x2 * xa;
    [
        s,
        0.5 / s,
        -0.25 / (xa * s),
        3.0 / (8.0 * x2 * s),
        -15.0 / (16.0 * x3 * s),
    ]
}

fn d_log(x: f64) -> [f64; 5] {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    [x.ln(), 1.0 / x, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

fn d_lognormpdf(x: f64) -> [f64; 5] {
    let c = 0.5 * (2.0 * std::f64::consts::PI).ln();
    [-0.5 * x * x - c, -x, -1.0, 0.0, 0.0]
}

// ────────────────────────────────────────────────────────────────────────
// JS1: one-seed jet (value+grad+hess + ε-channel = Σ_c ℓ_{abc} dir_c).
// Transcribes the `.cu` `struct JS1` and its js1_* operations exactly.
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Js1 {
    v: f64,
    g: [f64; K],
    h: [[f64; K]; K],
    dv: f64,
    dg: [f64; K],
    dh: [[f64; K]; K],
}

fn js1_const(c: f64) -> Js1 {
    Js1 {
        v: c,
        g: [0.0; K],
        h: [[0.0; K]; K],
        dv: 0.0,
        dg: [0.0; K],
        dh: [[0.0; K]; K],
    }
}

fn js1_var(x: f64, a: usize, dir: f64) -> Js1 {
    let mut r = js1_const(x);
    r.g[a] = 1.0;
    r.dv = dir;
    r
}

fn js1_scale(a: &Js1, s: f64) -> Js1 {
    let mut r = js1_const(a.v * s);
    r.dv = a.dv * s;
    for i in 0..K {
        r.g[i] = a.g[i] * s;
        r.dg[i] = a.dg[i] * s;
        for j in 0..K {
            r.h[i][j] = a.h[i][j] * s;
            r.dh[i][j] = a.dh[i][j] * s;
        }
    }
    r
}

fn js1_add(a: &Js1, b: &Js1) -> Js1 {
    let mut r = js1_const(a.v + b.v);
    r.dv = a.dv + b.dv;
    for i in 0..K {
        r.g[i] = a.g[i] + b.g[i];
        r.dg[i] = a.dg[i] + b.dg[i];
        for j in 0..K {
            r.h[i][j] = a.h[i][j] + b.h[i][j];
            r.dh[i][j] = a.dh[i][j] + b.dh[i][j];
        }
    }
    r
}

fn js1_addc(a: &Js1, c: f64) -> Js1 {
    let mut r = *a;
    r.v += c;
    r
}

fn js1_mul(a: &Js1, b: &Js1) -> Js1 {
    let mut r = js1_const(a.v * b.v);
    r.dv = a.dv * b.v + a.v * b.dv;
    for i in 0..K {
        r.g[i] = a.v * b.g[i] + a.g[i] * b.v;
        r.dg[i] = a.dv * b.g[i] + a.v * b.dg[i] + a.dg[i] * b.v + a.g[i] * b.dv;
    }
    for i in 0..K {
        for j in 0..K {
            r.h[i][j] = a.v * b.h[i][j] + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.h[i][j] * b.v;
            r.dh[i][j] = a.dv * b.h[i][j]
                + a.v * b.dh[i][j]
                + a.dg[i] * b.g[j]
                + a.g[i] * b.dg[j]
                + a.dg[j] * b.g[i]
                + a.g[j] * b.dg[i]
                + a.dh[i][j] * b.v
                + a.h[i][j] * b.dv;
        }
    }
    r
}

fn js1_compose(a: &Js1, f: &[f64; 5]) -> Js1 {
    let (f1, f2, f3) = (f[1], f[2], f[3]);
    let mut r = js1_const(f[0]);
    r.dv = f1 * a.dv;
    for i in 0..K {
        r.g[i] = f1 * a.g[i];
        r.dg[i] = f1 * a.dg[i] + f2 * a.dv * a.g[i];
    }
    for i in 0..K {
        for j in 0..K {
            r.h[i][j] = f1 * a.h[i][j] + f2 * a.g[i] * a.g[j];
            r.dh[i][j] = f1 * a.dh[i][j]
                + f2 * a.dv * a.h[i][j]
                + f2 * (a.dg[i] * a.g[j] + a.g[i] * a.dg[j])
                + f3 * a.dv * a.g[i] * a.g[j];
        }
    }
    r
}

// ────────────────────────────────────────────────────────────────────────
// JS2: two-seed jet (value+grad+hess + u/v/uv channels; uv = Σ_{cd} ℓ_{abcd} u_c v_d).
// Transcribes the `.cu` `struct JS2` and its js2_* operations exactly.
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Js2 {
    v: f64,
    g: [f64; K],
    h: [[f64; K]; K],
    eu: f64,
    ev: f64,
    gu: [f64; K],
    gv: [f64; K],
    hu: [[f64; K]; K],
    hv: [[f64; K]; K],
    euv: f64,
    guv: [f64; K],
    huv: [[f64; K]; K],
}

fn js2_const(c: f64) -> Js2 {
    Js2 {
        v: c,
        g: [0.0; K],
        h: [[0.0; K]; K],
        eu: 0.0,
        ev: 0.0,
        gu: [0.0; K],
        gv: [0.0; K],
        hu: [[0.0; K]; K],
        hv: [[0.0; K]; K],
        euv: 0.0,
        guv: [0.0; K],
        huv: [[0.0; K]; K],
    }
}

fn js2_var(x: f64, a: usize, du: f64, dv: f64) -> Js2 {
    let mut r = js2_const(x);
    r.g[a] = 1.0;
    r.eu = du;
    r.ev = dv;
    r
}

fn js2_scale(a: &Js2, s: f64) -> Js2 {
    let mut r = *a;
    r.v *= s;
    r.eu *= s;
    r.ev *= s;
    r.euv *= s;
    for i in 0..K {
        r.g[i] *= s;
        r.gu[i] *= s;
        r.gv[i] *= s;
        r.guv[i] *= s;
        for j in 0..K {
            r.h[i][j] *= s;
            r.hu[i][j] *= s;
            r.hv[i][j] *= s;
            r.huv[i][j] *= s;
        }
    }
    r
}

fn js2_add(a: &Js2, b: &Js2) -> Js2 {
    let mut r = js2_const(a.v + b.v);
    r.eu = a.eu + b.eu;
    r.ev = a.ev + b.ev;
    r.euv = a.euv + b.euv;
    for i in 0..K {
        r.g[i] = a.g[i] + b.g[i];
        r.gu[i] = a.gu[i] + b.gu[i];
        r.gv[i] = a.gv[i] + b.gv[i];
        r.guv[i] = a.guv[i] + b.guv[i];
        for j in 0..K {
            r.h[i][j] = a.h[i][j] + b.h[i][j];
            r.hu[i][j] = a.hu[i][j] + b.hu[i][j];
            r.hv[i][j] = a.hv[i][j] + b.hv[i][j];
            r.huv[i][j] = a.huv[i][j] + b.huv[i][j];
        }
    }
    r
}

fn js2_addc(a: &Js2, c: f64) -> Js2 {
    let mut r = *a;
    r.v += c;
    r
}

fn js2_mul(a: &Js2, b: &Js2) -> Js2 {
    let mut r = js2_const(a.v * b.v);
    r.eu = a.eu * b.v + a.v * b.eu;
    r.ev = a.ev * b.v + a.v * b.ev;
    r.euv = a.euv * b.v + a.eu * b.ev + a.ev * b.eu + a.v * b.euv;
    for i in 0..K {
        r.g[i] = a.v * b.g[i] + a.g[i] * b.v;
        r.gu[i] = a.eu * b.g[i] + a.v * b.gu[i] + a.gu[i] * b.v + a.g[i] * b.eu;
        r.gv[i] = a.ev * b.g[i] + a.v * b.gv[i] + a.gv[i] * b.v + a.g[i] * b.ev;
        r.guv[i] = a.euv * b.g[i]
            + a.eu * b.gv[i]
            + a.ev * b.gu[i]
            + a.v * b.guv[i]
            + a.guv[i] * b.v
            + a.gu[i] * b.ev
            + a.gv[i] * b.eu
            + a.g[i] * b.euv;
    }
    for i in 0..K {
        for j in 0..K {
            r.h[i][j] = a.v * b.h[i][j] + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.h[i][j] * b.v;
            r.hu[i][j] = a.eu * b.h[i][j]
                + a.v * b.hu[i][j]
                + a.gu[i] * b.g[j]
                + a.g[i] * b.gu[j]
                + a.gu[j] * b.g[i]
                + a.g[j] * b.gu[i]
                + a.hu[i][j] * b.v
                + a.h[i][j] * b.eu;
            r.hv[i][j] = a.ev * b.h[i][j]
                + a.v * b.hv[i][j]
                + a.gv[i] * b.g[j]
                + a.g[i] * b.gv[j]
                + a.gv[j] * b.g[i]
                + a.g[j] * b.gv[i]
                + a.hv[i][j] * b.v
                + a.h[i][j] * b.ev;
            r.huv[i][j] = a.euv * b.h[i][j]
                + a.eu * b.hv[i][j]
                + a.ev * b.hu[i][j]
                + a.v * b.huv[i][j]
                + a.guv[i] * b.g[j]
                + a.gu[i] * b.gv[j]
                + a.gv[i] * b.gu[j]
                + a.g[i] * b.guv[j]
                + a.guv[j] * b.g[i]
                + a.gu[j] * b.gv[i]
                + a.gv[j] * b.gu[i]
                + a.g[j] * b.guv[i]
                + a.huv[i][j] * b.v
                + a.hu[i][j] * b.ev
                + a.hv[i][j] * b.eu
                + a.h[i][j] * b.euv;
        }
    }
    r
}

fn js2_compose(a: &Js2, f: &[f64; 5]) -> Js2 {
    let (f1, f2, f3, f4) = (f[1], f[2], f[3], f[4]);
    let mut r = js2_const(f[0]);
    r.eu = f1 * a.eu;
    r.ev = f1 * a.ev;
    r.euv = f1 * a.euv + f2 * a.eu * a.ev;
    for i in 0..K {
        r.g[i] = f1 * a.g[i];
        r.gu[i] = f1 * a.gu[i] + f2 * a.eu * a.g[i];
        r.gv[i] = f1 * a.gv[i] + f2 * a.ev * a.g[i];
        r.guv[i] = f1 * a.guv[i]
            + f2 * (a.euv * a.g[i] + a.eu * a.gv[i] + a.ev * a.gu[i])
            + f3 * a.eu * a.ev * a.g[i];
    }
    for i in 0..K {
        for j in 0..K {
            let gigj = a.g[i] * a.g[j];
            r.h[i][j] = f1 * a.h[i][j] + f2 * gigj;
            r.hu[i][j] = f1 * a.hu[i][j]
                + f2 * a.eu * a.h[i][j]
                + f2 * (a.gu[i] * a.g[j] + a.g[i] * a.gu[j])
                + f3 * a.eu * gigj;
            r.hv[i][j] = f1 * a.hv[i][j]
                + f2 * a.ev * a.h[i][j]
                + f2 * (a.gv[i] * a.g[j] + a.g[i] * a.gv[j])
                + f3 * a.ev * gigj;
            let mut t = f1 * a.huv[i][j]
                + f2 * a.eu * a.hv[i][j]
                + f2 * a.ev * a.hu[i][j]
                + (f3 * a.eu * a.ev + f2 * a.euv) * a.h[i][j];
            let dgg_u = a.gu[i] * a.g[j] + a.g[i] * a.gu[j];
            let dgg_v = a.gv[i] * a.g[j] + a.g[i] * a.gv[j];
            let dgg_uv =
                a.guv[i] * a.g[j] + a.gu[i] * a.gv[j] + a.gv[i] * a.gu[j] + a.g[i] * a.guv[j];
            t += f2 * dgg_uv
                + f3 * a.eu * dgg_v
                + f3 * a.ev * dgg_u
                + (f4 * a.eu * a.ev + f3 * a.euv) * gigj;
            r.huv[i][j] = t;
        }
    }
    r
}

/// Row scalars consumed by the `DEF_NLL` program — the host mirror of the
/// `.cu` `struct RowIn`.
#[derive(Clone, Copy)]
struct RowIn {
    wi: f64,
    di: f64,
    z_sum: f64,
    cov_ones: f64,
    probit_scale: f64,
}

/// JS1 instantiation of the `DEF_NLL` program (mirror of `.cu` `nll_js1`).
fn nll_js1(q0: Js1, q1: Js1, qd1: Js1, g: Js1, inp: &RowIn) -> Js1 {
    let og = js1_scale(&g, inp.probit_scale);
    let opb2 = js1_addc(&js1_scale(&js1_mul(&og, &og), inp.cov_ones), 1.0);
    let c = js1_compose(&opb2, &d_sqrt(opb2.v));
    let ogz = js1_scale(&og, inp.z_sum);
    let eta0 = js1_add(&js1_mul(&q0, &c), &ogz);
    let eta1 = js1_add(&js1_mul(&q1, &c), &ogz);
    let ad1 = js1_mul(&qd1, &c);
    let neg0 = js1_scale(&eta0, -1.0);
    let entry = js1_scale(&js1_compose(&neg0, &neglog_phi_stack(neg0.v, inp.wi)), -1.0);
    let neg1 = js1_scale(&eta1, -1.0);
    let exit = js1_compose(&neg1, &neglog_phi_stack(neg1.v, inp.wi * (1.0 - inp.di)));
    let (mut ev, mut td) = (js1_const(0.0), js1_const(0.0));
    if inp.di > 0.0 {
        ev = js1_scale(&js1_compose(&eta1, &d_lognormpdf(eta1.v)), -inp.wi * inp.di);
        td = js1_scale(&js1_compose(&ad1, &d_log(ad1.v)), -inp.wi * inp.di);
    }
    js1_add(&js1_add(&exit, &entry), &js1_add(&ev, &td))
}

/// JS2 instantiation of the `DEF_NLL` program (mirror of `.cu` `nll_js2`).
fn nll_js2(q0: Js2, q1: Js2, qd1: Js2, g: Js2, inp: &RowIn) -> Js2 {
    let og = js2_scale(&g, inp.probit_scale);
    let opb2 = js2_addc(&js2_scale(&js2_mul(&og, &og), inp.cov_ones), 1.0);
    let c = js2_compose(&opb2, &d_sqrt(opb2.v));
    let ogz = js2_scale(&og, inp.z_sum);
    let eta0 = js2_add(&js2_mul(&q0, &c), &ogz);
    let eta1 = js2_add(&js2_mul(&q1, &c), &ogz);
    let ad1 = js2_mul(&qd1, &c);
    let neg0 = js2_scale(&eta0, -1.0);
    let entry = js2_scale(&js2_compose(&neg0, &neglog_phi_stack(neg0.v, inp.wi)), -1.0);
    let neg1 = js2_scale(&eta1, -1.0);
    let exit = js2_compose(&neg1, &neglog_phi_stack(neg1.v, inp.wi * (1.0 - inp.di)));
    let (mut ev, mut td) = (js2_const(0.0), js2_const(0.0));
    if inp.di > 0.0 {
        ev = js2_scale(&js2_compose(&eta1, &d_lognormpdf(eta1.v)), -inp.wi * inp.di);
        td = js2_scale(&js2_compose(&ad1, &d_log(ad1.v)), -inp.wi * inp.di);
    }
    js2_add(&js2_add(&exit, &entry), &js2_add(&ev, &td))
}

/// All five flattened channels from the HOST device-oracle for one row — laid
/// out exactly as the device kernel writes `out_v/out_g/out_h/out_t3/out_t4`.
struct HostRowChannels {
    value: f64,
    grad: [f64; K],
    hess: [[f64; K]; K],
    third: [[f64; K]; K],
    fourth: [[f64; K]; K],
}

/// Evaluate one row through the host transcription of the device kernel, with
/// the same `survival_rowjet` launch logic: JS1 carries v/g/h + contracted
/// third, JS2 carries the contracted fourth.
fn host_device_oracle_row(
    inp: &SurvivalRowInputs,
    probit_scale: f64,
    dir: &[f64; K],
    dir_u: &[f64; K],
    dir_v: &[f64; K],
) -> HostRowChannels {
    let row = RowIn {
        wi: inp.wi,
        di: inp.di,
        z_sum: inp.z_sum,
        cov_ones: inp.cov_ones,
        probit_scale,
    };
    let p = inp.primaries;
    let j1 = nll_js1(
        js1_var(p[0], 0, dir[0]),
        js1_var(p[1], 1, dir[1]),
        js1_var(p[2], 2, dir[2]),
        js1_var(p[3], 3, dir[3]),
        &row,
    );
    let j2 = nll_js2(
        js2_var(p[0], 0, dir_u[0], dir_v[0]),
        js2_var(p[1], 1, dir_u[1], dir_v[1]),
        js2_var(p[2], 2, dir_u[2], dir_v[2]),
        js2_var(p[3], 3, dir_u[3], dir_v[3]),
        &row,
    );
    HostRowChannels {
        value: j1.v,
        grad: j1.g,
        hess: j1.h,
        third: j1.dh,
        fourth: j2.huv,
    }
}

// Formula variants: the rigid survival marginal-slope NLL has three
// structurally distinct row regimes that exercise DIFFERENT branches of the
// `DEF_NLL` program — the event term `if(di>0)` (log φ + log ad1) is only
// live for events, the entry `logΦ` term only for left-truncated rows, and
// a zero-weight row must collapse the whole probit stack. Each is a separate
// "formula variant" in the #415 sense.
#[derive(Clone, Copy)]
enum Variant {
    /// Right-censored row, no left truncation (z_sum=0): exit term only.
    CensoredNoEntry,
    /// Event row, no left truncation: exit + event-density + log-slope terms.
    EventNoEntry,
    /// Left-truncated event row: all four NLL terms live.
    EventWithEntry,
    /// Left-truncated censored row: entry + exit terms.
    CensoredWithEntry,
    /// Zero prior weight: every probit term must drop out.
    ZeroWeight,
}

impl Variant {
    const ALL: [Variant; 5] = [
        Variant::CensoredNoEntry,
        Variant::EventNoEntry,
        Variant::EventWithEntry,
        Variant::CensoredWithEntry,
        Variant::ZeroWeight,
    ];

    fn di(self) -> f64 {
        match self {
            Variant::EventNoEntry | Variant::EventWithEntry => 1.0,
            _ => 0.0,
        }
    }

    fn wi(self) -> f64 {
        match self {
            Variant::ZeroWeight => 0.0,
            _ => 1.0,
        }
    }

    fn z_sum(self, base: f64) -> f64 {
        match self {
            Variant::CensoredNoEntry | Variant::EventNoEntry | Variant::ZeroWeight => 0.0,
            _ => base,
        }
    }
}

// Edge-input regimes that stress the transcendental tails and the
// square-root / log guards — exactly the places a hand-port silently drifts.
#[derive(Clone, Copy)]
enum Regime {
    /// Well-conditioned interior.
    Interior,
    /// Deep negative probit tail (η ≫ 0 ⇒ logΦ(−η) on the erfcx branch).
    DeepNegativeTail,
    /// Near-degenerate covariance (cov_ones → 0, c → 1).
    TinyCovariance,
    /// Large covariance + large slope (c large, ad1 large).
    LargeScale,
    /// qd1 near the positive floor (log ad1 large-magnitude derivatives).
    SmallSlope,
}

impl Regime {
    const ALL: [Regime; 5] = [
        Regime::Interior,
        Regime::DeepNegativeTail,
        Regime::TinyCovariance,
        Regime::LargeScale,
        Regime::SmallSlope,
    ];

    /// Primaries `(q0, q1, qd1, g)` for this regime.
    fn primaries(self) -> [f64; 4] {
        match self {
            Regime::Interior => [-0.7, 0.4, 0.9, -0.3],
            Regime::DeepNegativeTail => [7.5, 6.0, 0.8, 0.6],
            Regime::TinyCovariance => [-0.5, 0.3, 1.1, 0.05],
            Regime::LargeScale => [-1.5, 1.2, 3.0, 2.5],
            // `qd1` is the time-derivative primary; the event NLL carries a live
            // `-log(ad1)` term with `ad1 = qd1·c`. Its fourth-derivative stack
            // scales like `6/ad1⁴`, so as `qd1 → 0` the JS2 `huv` recurrence
            // forms intermediates of magnitude `~6/qd1⁴` that cancel down to the
            // tiny final entry — a cancellation condition number `~1/qd1⁴`. At
            // `qd1 = 1e-2` that floor is `~ε·6e8 ≈ 1e-7`-relative, well inside the
            // 1e-9 *relative* gate; pushing `qd1` to 1e-3 (cond `~6e12`) makes the
            // contracted-fourth channel numerically meaningless — no finite
            // tolerance can then separate a faithful transcription from a real
            // bug. Production never reaches that regime either: the monotonicity
            // guard (`time_derivative_lower_bound`) floors `qd1` away from 0. So
            // SmallSlope stays a genuine small-slope stress (90× below Interior's
            // 0.9) while remaining inside the well-posed domain both jets share.
            Regime::SmallSlope => [-0.4, 0.2, 1e-2, -0.1],
        }
    }

    fn cov_ones(self) -> f64 {
        match self {
            Regime::TinyCovariance => 1e-9,
            Regime::LargeScale => 4.0,
            _ => 0.6,
        }
    }

    fn probit_scale(self) -> f64 {
        match self {
            Regime::LargeScale => 1.3,
            _ => 0.7,
        }
    }
}

fn make_row(v: Variant, r: Regime) -> (SurvivalRowInputs, f64) {
    let inp = SurvivalRowInputs {
        primaries: r.primaries(),
        wi: v.wi(),
        di: v.di(),
        z_sum: v.z_sum(0.45),
        cov_ones: r.cov_ones(),
    };
    (inp, r.probit_scale())
}

// Contraction directions: the seeded jets carry a FIXED contraction
// direction whose channel is the relevant tensor contraction. A sign or
// index error in the JS1/JS2 recurrences only shows up for SOME directions,
// so the sweep covers several — including unit axes (isolating one tensor
// slice) and dense mixed-sign directions.
const DIRS: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],      // unit axis q0
    [0.0, 0.0, 0.0, 1.0],      // unit axis g (the nonlinear primary)
    [0.31, -0.22, 0.17, 0.44], // dense, mixed sign
    [-0.5, 0.5, -0.25, 0.125], // dense, decaying
];
const DIRS_U: [[f64; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [0.13, 0.27, -0.41, 0.05],
    [1.0, -1.0, 1.0, -1.0],
    [0.2, 0.0, 0.3, 0.0],
];
const DIRS_V: [[f64; 4]; 4] = [
    [0.0, 0.0, 1.0, 0.0],
    [-0.19, 0.33, 0.08, 0.22],
    [0.5, 0.5, 0.5, 0.5],
    [0.0, 0.4, 0.0, 0.1],
];

/// THE #415 parity-lock: the HOST transcription of the device kernel must
/// match the production CPU `rigid_row_nll` (via the public
/// `survival_rigid_row_jets_cpu` fallback) on EVERY channel — value,
/// gradient, Hessian, contracted third, contracted fourth — across every
/// formula variant × edge regime × contraction direction. This runs on
/// every box (no CUDA required), so device-algebra drift fails the build
/// here long before it reaches the GPU runner.
#[test]
fn host_device_oracle_matches_cpu_rigid_jet_all_variants() {
    // Both paths build the whole probit derivative stack from one shared
    // Mills-ratio transcendental, so the CPU single-source jet and the host
    // transcription of the device hand-jet differ only in floating-point
    // evaluation ORDER. The mixed abs+rel band absorbs that ULP-scale
    // reordering noise while still failing on any real algebra drift (a
    // dropped/sign-flipped/mis-indexed term moves a channel by O(1)·magnitude,
    // not O(ε)). The regimes are deliberately held inside the well-posed domain
    // both jets share (see `Regime::SmallSlope`), so even the most
    // cancellation-prone contracted-fourth channel lands far under this band —
    // the worst observed normalized drift is asserted below to stay ≤ 1e-2 of
    // the tolerance, i.e. the gate has ~100× headroom and is bounded by the
    // transcription's correctness, not by a slack tolerance.
    const ABS_TOL: f64 = 1e-9;
    const REL_TOL: f64 = 1e-9;
    let mut worst_ratio = 0.0f64;
    let mut close = |a: f64, b: f64| -> bool {
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let band = ABS_TOL + REL_TOL * a.abs().max(b.abs());
        worst_ratio = worst_ratio.max(diff / band);
        diff <= band
    };

    let mut checked = 0usize;
    for v in Variant::ALL {
        for r in Regime::ALL {
            let (inp, probit_scale) = make_row(v, r);
            for d in 0..DIRS.len() {
                let dir = DIRS[d];
                let dir_u = DIRS_U[d];
                let dir_v = DIRS_V[d];

                // Production CPU path (single source) for this exact row.
                let cpu = survival_rigid_row_jets_cpu(
                    std::slice::from_ref(&inp),
                    probit_scale,
                    &dir,
                    &dir_u,
                    &dir_v,
                );
                // Host transcription of the device kernel.
                let host = host_device_oracle_row(&inp, probit_scale, &dir, &dir_u, &dir_v);

                assert!(
                    close(cpu.value[0], host.value),
                    "value drift: cpu={} host={} (variant {}, regime {}, dir {d})",
                    cpu.value[0],
                    host.value,
                    v as usize,
                    r as usize,
                );
                for a in 0..K {
                    assert!(
                        close(cpu.grad[a], host.grad[a]),
                        "grad[{a}] drift: cpu={} host={} (variant {}, regime {}, dir {d})",
                        cpu.grad[a],
                        host.grad[a],
                        v as usize,
                        r as usize,
                    );
                    for b in 0..K {
                        let idx = a * K + b;
                        assert!(
                            close(cpu.hess[idx], host.hess[a][b]),
                            "hess[{a}][{b}] drift: cpu={} host={} (variant {}, regime {}, dir {d})",
                            cpu.hess[idx],
                            host.hess[a][b],
                            v as usize,
                            r as usize,
                        );
                        assert!(
                            close(cpu.third[idx], host.third[a][b]),
                            "third[{a}][{b}] drift: cpu={} host={} (variant {}, regime {}, dir {d})",
                            cpu.third[idx],
                            host.third[a][b],
                            v as usize,
                            r as usize,
                        );
                        assert!(
                            close(cpu.fourth[idx], host.fourth[a][b]),
                            "fourth[{a}][{b}] drift: cpu={} host={} (variant {}, regime {}, dir {d})",
                            cpu.fourth[idx],
                            host.fourth[a][b],
                            v as usize,
                            r as usize,
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    // The gate must be bounded by the transcription's correctness, not by a
    // slack tolerance: the worst normalized drift across the whole sweep stays
    // far below the band. If a future regime/edit pushes this toward 1.0 the
    // tolerance is masking conditioning, not catching drift — fail loudly.
    assert!(
        worst_ratio <= 1e-2,
        "parity-lock headroom collapsed: worst drift/tolerance = {worst_ratio:.3e} (want ≤ 1e-2); \
         the band is absorbing conditioning noise rather than pinning the transcription"
    );
    // Guard against an accidental empty sweep silently "passing".
    assert_eq!(
        checked,
        Variant::ALL.len() * Regime::ALL.len() * DIRS.len() * K * K,
        "parity-lock sweep coverage changed unexpectedly"
    );
}

/// Symmetry of the contracted-fourth channel under swapping the two seed
/// directions: `Σ_{cd} ℓ_{abcd} u_c v_d == Σ_{cd} ℓ_{abcd} v_c u_d` because
/// the fourth derivative tensor is fully symmetric. The device JS2 `huv`
/// recurrence is the most error-prone part of the hand-port (16 Leibniz
/// terms); this pins its u↔v symmetry independently of the CPU comparison.
#[test]
fn host_oracle_fourth_channel_is_seed_swap_symmetric() {
    const TOL: f64 = 1e-9;
    for v in Variant::ALL {
        for r in Regime::ALL {
            let (inp, probit_scale) = make_row(v, r);
            let u = DIRS_U[2];
            let w = DIRS_V[2];
            let dir = DIRS[2];
            let uv = host_device_oracle_row(&inp, probit_scale, &dir, &u, &w).fourth;
            let vu = host_device_oracle_row(&inp, probit_scale, &dir, &w, &u).fourth;
            for a in 0..K {
                for b in 0..K {
                    let scale = uv[a][b].abs().max(vu[a][b].abs()).max(1.0);
                    assert!(
                        (uv[a][b] - vu[a][b]).abs() <= TOL * scale,
                        "fourth-channel u↔v asymmetry at [{a}][{b}]: uv={} vu={} (variant {}, regime {})",
                        uv[a][b],
                        vu[a][b],
                        v as usize,
                        r as usize,
                    );
                }
            }
        }
    }
}

/// Lockstep guard: the host transcription above is a line-for-line mirror of
/// the device `.cu` JS1/JS2 + `DEF_NLL` arithmetic. If the `.cu` arithmetic
/// is edited, this transcription MUST be re-derived in lockstep or the
/// parity sweep above goes stale (it would keep comparing the CPU jet to an
/// outdated transcription instead of to the live device program).
///
/// This test pins the structural fingerprint of the `.cu` arithmetic
/// surface — the set of jet operations and the NLL term skeleton — so a
/// change to the device program that this file does not mirror fails the
/// build with a pointer back here. It deliberately checks STRUCTURE (which
/// operations exist, the NLL term names) rather than exact bytes, so
/// comment/whitespace edits do not trip it but an added/removed/renamed jet
/// operation or NLL term does.
#[cfg(target_os = "linux")]
#[test]
fn device_kernel_source_matches_transcription_fingerprint() {
    let src = super::SURVIVAL_ROWJET_SOURCE;
    // Every jet operation the host transcription mirrors must still be
    // present in the device source. A renamed/removed device op means the
    // host mirror is stale.
    let required_ops = [
        "js1_const",
        "js1_var",
        "js1_scale",
        "js1_add",
        "js1_addc",
        "js1_mul",
        "js1_compose",
        "js2_const",
        "js2_var",
        "js2_scale",
        "js2_add",
        "js2_addc",
        "js2_mul",
        "js2_compose",
        "erfcx_nn",
        "normal_pdf",
        "normal_cdf",
        "sp_logcdf_mills",
        "neglog_phi_stack",
        "d_sqrt",
        "d_log",
        "d_lognormpdf",
        "nll_js1",
        "nll_js2",
        "DEF_NLL",
    ];
    for op in required_ops {
        assert!(
            src.contains(op),
            "device kernel no longer defines `{op}` — the host transcription in \
                 survival_rowjet_host_oracle_tests.rs is now stale; re-derive it in lockstep \
                 with the .cu edit so the #415 CPU↔device parity-lock keeps testing the \
                 LIVE device program (see this file's module docs)."
        );
    }
    // The four NLL terms must all still be assembled — a dropped term is the
    // exact #736 sign-flip-genus regression this lock exists to catch.
    for term in [
        "neglog_phi_stack(neg0",
        "neglog_phi_stack(neg1",
        "d_lognormpdf",
        "d_log(ad1",
    ] {
        assert!(
            src.contains(term),
            "device NLL program dropped `{term}` — host transcription is stale; \
                 re-derive survival_rowjet_host_oracle_tests.rs in lockstep with the .cu."
        );
    }
}
