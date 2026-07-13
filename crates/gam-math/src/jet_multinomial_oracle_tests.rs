//! Multinomial-logit (softmax) Taylor-jet oracle (#932, FD-free exactness).
//!
//! The test writes the multinomial row NLL once through generic jet algebra and
//! compares its value, gradient, and Hessian with an independent softmax
//! closed form. Production's higher derivatives are now generated from the
//! canonical row program and are tested at that production seam; duplicating
//! those formulas here would only compare a kernel with a frozen copy of itself.
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
//! The comparison holds at `rel_tol = 1e-11`, tighter than the issue's ~1e-10 ask.

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
        let tower: Box<Tower4<M>> =
            program_full_tower(&program, row).expect("multinomial jet tower");
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
