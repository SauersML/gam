//! Multinomial-logit (softmax) Taylor-jet oracle (#932, FD-free exactness).
//!
//! Issue #932 single-sources every family's derivative tower onto ONE row-NLL
//! expression evaluated through the generic jet algebra, instead of a
//! hand-written gradient/Hessian/third/fourth cascade. This module wires the
//! **multinomial logit** family — the last of the survey-flagged production
//! families whose derivative tower is still assembled by hand
//! (`crates/gam-models/src/multinomial_reml.rs`: `joint_loglik_and_gradient_from_probs`,
//! `hessian_matvec_into_with_probs`, and the third/fourth "directional Fisher
//! jet" closed forms `assemble_all_axis_directional_derivatives` /
//! `assemble_all_axis_second_directional_derivatives`) — to the universal jet
//! oracle.
//!
//! # Why the multinomial hand tower is exactly the softmax cumulant sequence
//!
//! With `M = K − 1` active classes (the reference class `K−1` is pinned at
//! `η_{K-1} ≡ 0`), the weighted row negative log-likelihood over the active
//! primaries `η = (η_0, …, η_{M-1})` is
//!
//! ```text
//!   Z(η)   = 1 + Σ_{a<M} e^{η_a}
//!   ℓ(η)   = w·ln Z(η) − w·η_obs        (η_obs ≡ 0 when the row is the reference)
//! ```
//!
//! whose derivative channels are the softmax cumulants (`p_a = e^{η_a}/Z`):
//!
//! ```text
//!   ∂_a ℓ        = w (p_a − y_a)                      [residual — the hand gradient]
//!   ∂_a∂_b ℓ     = w (δ_ab p_a − p_a p_b)             [Fisher block W_{a,b}]
//!   ∂_a∂_b∂_c ℓ  = the per-row `Ĵ` "directional Fisher jet" closed form
//!   ∂⁴ ℓ         = the per-row `Ĵ²` second-directional Fisher jet closed form
//! ```
//!
//! Because multinomial logit is a **canonical** GLM, the observed Hessian equals
//! the expected Fisher information — so the jet-derived *observed* tower here is
//! bit-for-bit the same object the production path assembles from Fisher blocks,
//! with **no** observed-vs-expected discrepancy to reconcile. That makes this the
//! cleanest possible #932 cutover witness: the two hand third/fourth closed forms
//! (`Ĵ`, `Ĵ²`) that the near-separable Jeffreys/Firth solve (#1082) runs in its
//! per-inner-cycle hot path are pinned, term for term, against the mechanically
//! jet-differentiated softmax NLL.
//!
//! # What this module asserts
//!
//! * [`multinomial_softmax_jet_value_grad_hessian_matches_closed_form`] — the
//!   jet value / ∇ / H against an INDEPENDENT softmax closed form, through the
//!   same [`verify_kernel_channels`] universal oracle every other family uses.
//! * [`multinomial_softmax_jet_third_fourth_match_production_directional_jets`] —
//!   the jet's contracted third `∂_{a0}H` and fourth `∂_δ∂_{a0}H` against the
//!   EXACT `Ĵ` / `Ĵ²` closed forms replicated verbatim from the production
//!   `assemble_all_axis_*` sweeps. A dropped/sign-flipped term in either hand
//!   closed form is loud here.
//!
//! Both hold at `rel_tol = 1e-11`, tighter than the issue's ~1e-10 ask.

use crate::jet_scalar::JetScalar;
use crate::jet_tower::{
    KernelChannels, RowProgram, Tower4, program_full_tower, verify_kernel_channels,
};

/// One multinomial-logit fixture over `M` active primaries: the active-class
/// log-odds `η` (the reference class `M` is pinned at `0`), the observed class
/// `obs ∈ 0..=M` (`obs == M` is the reference row), and the prior weight `w`.
#[derive(Clone, Copy, Debug)]
struct MultRow<const M: usize> {
    eta: [f64; M],
    obs: usize,
    w: f64,
}

/// The multinomial-logit family written ONCE as a generic
/// [`RowProgram<M>`] over the jet scalar `S`. The row NLL body uses
/// ONLY [`JetScalar`] ops (`exp`, `add`, `ln`, `scale`, `sub`); the per-row data
/// (observed class, weight) enters as plain constants.
struct MultinomialSoftmaxRow<const M: usize> {
    rows: Vec<MultRow<M>>,
}

impl<const M: usize> RowProgram<M> for MultinomialSoftmaxRow<M> {
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; M], String> {
        let r = self
            .rows
            .get(row)
            .ok_or_else(|| format!("MultinomialSoftmaxRow: row {row} out of range"))?;
        Ok(r.eta)
    }

    fn eval<S: JetScalar<M>>(&self, row: usize, p: &[S; M]) -> Result<S, String> {
        let data = self
            .rows
            .get(row)
            .ok_or_else(|| format!("MultinomialSoftmaxRow: row {row} out of range"))?;
        // Z = 1 + Σ_a e^{η_a}  (the reference class contributes e^0 = 1).
        let mut z = S::constant(1.0);
        for a in 0..M {
            z = z.add(&p[a].exp());
        }
        // ℓ = w·ln Z − w·η_obs.  The linear −w·η_obs term (zero when the row is
        // the reference class, obs == M) shifts only value + gradient; every
        // second-and-higher channel comes entirely from w·ln Z.
        let mut ell = z.ln().scale(data.w);
        if data.obs < M {
            ell = ell.sub(&p[data.obs].scale(data.w));
        }
        Ok(ell)
    }
}

/// Softmax probabilities `p_a = e^{η_a}/Z` over the `M` active classes at `eta`.
fn softmax_active<const M: usize>(eta: &[f64; M]) -> [f64; M] {
    let mut z = 1.0_f64;
    let mut ex = [0.0_f64; M];
    for a in 0..M {
        ex[a] = eta[a].exp();
        z += ex[a];
    }
    let mut p = [0.0_f64; M];
    for a in 0..M {
        p[a] = ex[a] / z;
    }
    p
}

/// INDEPENDENT softmax closed-form value / gradient / Hessian at `row` (NO jet),
/// packaged as the [`KernelChannels`] a hand kernel would claim. Third/fourth are
/// left empty — they are pinned separately against the production `Ĵ`/`Ĵ²` forms.
fn multinomial_closed_form_vgh<const M: usize>(row: &MultRow<M>) -> KernelChannels<M> {
    let p = softmax_active(&row.eta);
    let z = {
        let mut s = 1.0_f64;
        for a in 0..M {
            s += row.eta[a].exp();
        }
        s
    };
    let obs_eta = if row.obs < M { row.eta[row.obs] } else { 0.0 };
    let value = row.w * z.ln() - row.w * obs_eta;

    let mut gradient = [0.0_f64; M];
    for a in 0..M {
        let y_a = if row.obs == a { 1.0 } else { 0.0 };
        gradient[a] = row.w * (p[a] - y_a);
    }

    let mut hessian = [[0.0_f64; M]; M];
    for a in 0..M {
        for b in 0..M {
            let delta = if a == b { 1.0 } else { 0.0 };
            hessian[a][b] = row.w * (delta * p[a] - p[a] * p[b]);
        }
    }

    KernelChannels {
        value,
        gradient,
        hessian,
        third: Vec::new(),
        fourth: Vec::new(),
    }
}

/// The production per-row third "directional Fisher jet" `Ĵ_{a0}` — the
/// derivative of the Fisher block `W_{c,d} = w(δ_cd p_c − p_c p_d)` along the
/// canonical class axis `a0` — replicated VERBATIM from
/// `assemble_all_axis_directional_derivatives`
/// (`crates/gam-models/src/multinomial_reml.rs`).
fn production_jhat_third<const M: usize>(p: &[f64; M], w: f64, a0: usize) -> [[f64; M]; M] {
    let pa0 = p[a0];
    let mut dphat = [0.0_f64; M];
    for c in 0..M {
        dphat[c] = p[c] * ((if c == a0 { 1.0 } else { 0.0 }) - pa0);
    }
    let mut jhat = [[0.0_f64; M]; M];
    for c in 0..M {
        jhat[c][c] = w * (dphat[c] - 2.0 * dphat[c] * p[c]);
        for d in (c + 1)..M {
            let off = w * (-(dphat[c] * p[d] + p[c] * dphat[d]));
            jhat[c][d] = off;
            jhat[d][c] = off;
        }
    }
    jhat
}

/// The production per-row fourth "second-directional Fisher jet" `Ĵ²_{a0,δ}` —
/// the derivative of `Ĵ_{a0}` along an arbitrary first direction `δ` (in
/// η-space) — replicated VERBATIM from
/// `assemble_all_axis_second_directional_derivatives`
/// (`crates/gam-models/src/multinomial_reml.rs`).
fn production_jhat_fourth<const M: usize>(
    p: &[f64; M],
    w: f64,
    a0: usize,
    delta: &[f64; M],
) -> [[f64; M]; M] {
    let mut s_u = 0.0_f64;
    for c in 0..M {
        s_u += p[c] * delta[c];
    }
    let mut dp_u = [0.0_f64; M];
    for c in 0..M {
        dp_u[c] = p[c] * (delta[c] - s_u);
    }
    let pa0 = p[a0];
    let mut dp_v_hat = [0.0_f64; M];
    let mut ds_u_dv = 0.0_f64;
    for c in 0..M {
        let v = p[c] * ((if c == a0 { 1.0 } else { 0.0 }) - pa0);
        dp_v_hat[c] = v;
        ds_u_dv += v * delta[c];
    }
    let mut ddp_hat = [0.0_f64; M];
    for c in 0..M {
        ddp_hat[c] = dp_v_hat[c] * (delta[c] - s_u) - p[c] * ds_u_dv;
    }
    let mut jhat = [[0.0_f64; M]; M];
    for a in 0..M {
        let pa = p[a];
        jhat[a][a] = w * (ddp_hat[a] * (1.0 - 2.0 * pa) - 2.0 * dp_u[a] * dp_v_hat[a]);
        for b in (a + 1)..M {
            let off = w
                * (-(ddp_hat[a] * p[b]
                    + dp_u[a] * dp_v_hat[b]
                    + dp_v_hat[a] * dp_u[b]
                    + pa * ddp_hat[b]));
            jhat[a][b] = off;
            jhat[b][a] = off;
        }
    }
    jhat
}

/// A tiny deterministic LCG so the fixtures are pseudo-random yet fixed across
/// runs (NO `rand`, NO date/clock seeding — per the #932 rules).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Build `count` deterministic fixtures over `M` active classes; `obs` cycles
/// through every active class AND the reference class (`obs == M`).
fn make_rows<const M: usize>(seed: u64, count: usize) -> Vec<MultRow<M>> {
    let mut rng = Lcg(seed);
    let mut rows = Vec::with_capacity(count);
    for i in 0..count {
        let mut eta = [0.0_f64; M];
        for a in 0..M {
            eta[a] = rng.uniform(-2.0, 2.0);
        }
        rows.push(MultRow {
            eta,
            obs: i % (M + 1),
            w: rng.uniform(0.25, 2.5),
        });
    }
    rows
}

const REL_TOL: f64 = 1e-11;

fn assert_vgh<const M: usize>(seed: u64) {
    let rows = make_rows::<M>(seed, 24);
    let program = MultinomialSoftmaxRow { rows: rows.clone() };
    for (row, fixture) in rows.iter().enumerate() {
        let tower: Tower4<M> = program_full_tower(&program, row).expect("multinomial jet tower");
        let claims = multinomial_closed_form_vgh(fixture);
        verify_kernel_channels(&tower, &claims, REL_TOL).unwrap_or_else(|e| {
            panic!("M={M} row {row}: softmax closed form disagrees with #932 jet tower: {e}")
        });
    }
}

/// The mechanically jet-derived multinomial value / ∇ / H equals the INDEPENDENT
/// softmax closed form (residual gradient + Fisher Hessian), channel by channel,
/// through the SAME universal [`verify_kernel_channels`] oracle every other #932
/// family uses — for `M = 2` (K=3 classes) and `M = 3` (K=4), observed rows
/// spanning every active class and the reference.
#[test]
fn multinomial_softmax_jet_value_grad_hessian_matches_closed_form() {
    assert_vgh::<2>(0x9322_2020_1109_face);
    assert_vgh::<3>(0x0bad_c0de_2020_1109);
}

fn assert_third_fourth<const M: usize>(seed: u64) {
    let rows = make_rows::<M>(seed, 16);
    let program = MultinomialSoftmaxRow { rows: rows.clone() };
    // Deterministic first-direction δ in η-space for the fourth contraction.
    let mut rng = Lcg(seed ^ 0xf0f0_f0f0_f0f0_f0f0);
    let close = |a: f64, b: f64, label: &str| {
        let band = REL_TOL + REL_TOL * a.abs().max(b.abs());
        assert!(
            (a - b).abs() <= band,
            "{label}: jet {a:+.15e} vs production {b:+.15e} (band {band:.3e})"
        );
    };
    for (row, fixture) in rows.iter().enumerate() {
        let tower: Tower4<M> = program_full_tower(&program, row).expect("multinomial jet tower");
        let p = softmax_active(&fixture.eta);
        let w = fixture.w;
        let mut delta = [0.0_f64; M];
        for a in 0..M {
            delta[a] = rng.uniform(-1.5, 1.5);
        }
        for a0 in 0..M {
            // Third: ∂_{a0} of the Fisher block == jet third contracted along e_{a0}.
            let mut e_a0 = [0.0_f64; M];
            e_a0[a0] = 1.0;
            let jet_third = tower.third_contracted(&e_a0);
            let prod_third = production_jhat_third(&p, w, a0);
            for c in 0..M {
                for d in 0..M {
                    close(
                        jet_third[c][d],
                        prod_third[c][d],
                        &format!("M={M} row {row} a0 {a0} third[{c}][{d}]"),
                    );
                }
            }
            // Fourth: ∂_δ∂_{a0} of the Fisher block == jet fourth contracted (δ, e_{a0}).
            let jet_fourth = tower.fourth_contracted(&delta, &e_a0);
            let prod_fourth = production_jhat_fourth(&p, w, a0, &delta);
            for c in 0..M {
                for d in 0..M {
                    close(
                        jet_fourth[c][d],
                        prod_fourth[c][d],
                        &format!("M={M} row {row} a0 {a0} fourth[{c}][{d}]"),
                    );
                }
            }
        }
    }
}

/// The jet's contracted third `∂_{a0}H` and fourth `∂_δ∂_{a0}H` reproduce the
/// EXACT production `Ĵ` / `Ĵ²` directional-Fisher-jet closed forms (replicated
/// verbatim from the `assemble_all_axis_*` sweeps) at machine precision — for
/// `M = 2` and `M = 3`. This pins the two hand third/fourth closed forms that the
/// #1082 Jeffreys/Firth hot path runs against the mechanically differentiated
/// softmax NLL: any dropped or sign-flipped term is loud.
#[test]
fn multinomial_softmax_jet_third_fourth_match_production_directional_jets() {
    assert_third_fourth::<2>(0x1109_2020_9322_c0de);
    assert_third_fourth::<3>(0xface_1109_2020_9322);
}
