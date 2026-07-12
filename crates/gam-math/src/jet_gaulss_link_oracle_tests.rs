//! Gaussian location-scale (`gaulss`) Taylor-jet oracle in the PRODUCTION
//! `log-b` σ-link parameterization (#932, FD-free exactness).
//!
//! The existing [`super::jet_gaussian_oracle_tests`] pins the Gaussian
//! location-scale OBSERVED tower in the *direct* `s = log σ` parameterization.
//! Production `gaulss` (`gam-models/src/gamlss/gaussian/joint_psi.rs`) does NOT
//! use that parameterization: it fits the scale channel through the **log-b**
//! σ-link `σ(η) = b + e^{η}` (with the `LOGB_SIGMA_FLOOR` `b = 0.01`), so its
//! score / Hessian are a hand-written `κ`-chain-rule cascade over
//! `κ = e^{η}/σ = dlogσ/dη`, `κ' = κ(1−κ)`, `κ'' = κ'(1−2κ)`. That hand cascade
//! (`gaussian_joint_psi_firstweights`: `scoremu`, `score_ls`, `dscoremu`,
//! `dscore_ls`, `hmumu`) is the #932 target this module single-sources.
//!
//! # Scope: the OBSERVED tower only (by design)
//!
//! Production deliberately assembles a **hybrid** Newton Hessian, NOT the raw
//! observed Hessian:
//! * the `(μ, ls)` cross block is forced to its Fisher expectation **0** (μ ⊥ σ;
//!   #684) instead of the observed `2mκ`, and
//! * the `(ls, ls)` block uses the expected information `2κ²a` (#566) instead of
//!   the observed `2κ²n + κ'(a−n)`.
//!
//! Those two substitutions are a documented *modeling* layer (they stabilize the
//! REML determinant / EDF), NOT hand-derivative-tower work — so they are OUT OF
//! SCOPE for jet single-sourcing and are NOT asserted here. What #932 owns is the
//! OBSERVED `κ`-chain-rule tower, which the production **score** and its
//! directional derivatives (`dscoremu`, `dscore_ls`) carry exactly. This oracle
//! pins that observed tower — value / ∇ / observed H — against the mechanically
//! jet-differentiated row NLL.
//!
//! # The model
//!
//! Two primaries `p = (η_μ, η_ls)`; `μ = η_μ` (identity mean link),
//! `σ = b + e^{η_ls}`, prior weight `a`, residual `r = y − μ`. The weighted row
//! NLL (dropping the data-only `½ log 2π` constant) is
//!
//! ```text
//!   ℓ(η_μ, η_ls) = a·log σ + ½·a·r²/σ².
//! ```
//!
//! Writing `w = a/σ²`, `m = r w`, `n = r² w`, `κ = e^{η_ls}/σ`, `κ' = κ(1−κ)`,
//! the OBSERVED tower (matched term-for-term to the production hand scalars) is
//!
//! ```text
//!   ∇      = [ −m,  κ(a − n) ]                       (scoremu, score_ls)
//!   H_obs  = [[ w,            2 m κ                 ],
//!             [ 2 m κ,   2 κ² n + κ'(a − n)         ]]
//! ```
//!
//! where `H_obs[μ][μ] = w`, `H_obs[μ][ls] = 2mκ` is exactly what
//! `dscoremu = w·μ̇ + 2m·κ·η̇` contracts, and
//! `H_obs[ls][ls] = 2κ²n + κ'(a−n)` is exactly what
//! `dscore_ls = κ(2m·μ̇ + 2n·κ·η̇) + κ'(a−n)·η̇` contracts.

use crate::jet_scalar::JetScalar;
use crate::jet_tower::{
    KernelChannels, RowProgram, Tower4, program_full_tower, verify_kernel_channels,
};

/// The production `log-b` σ-link floor `b` (`gam-model-kernels::sigma_link::LOGB_SIGMA_FLOOR`).
/// Replicated as a local constant so this oracle stays a leaf test in `gam-math`
/// (which does not depend on `gam-model-kernels`); a drift in the production
/// floor is caught by the `gam-models` gaulss parity suite, not here — here the
/// contract is that the jet tower reproduces the OBSERVED κ-chain-rule for the
/// σ-link `σ = b + e^{η}` whatever finite `b` is used.
const LOGB_SIGMA_FLOOR: f64 = 0.01;

/// One `gaulss` fixture: response `y`, mean primary `η_μ`, scale primary `η_ls`,
/// and prior weight `a`.
#[derive(Clone, Copy, Debug)]
struct GaulssRow {
    y: f64,
    eta_mu: f64,
    eta_ls: f64,
    a: f64,
}

/// The `gaulss` family written ONCE as a generic [`RowProgram<2>`] over
/// the jet scalar `S`, in the production log-b σ-link parameterization. The body
/// uses ONLY [`JetScalar`] ops; the per-row data (`y`, `a`) enters as constants.
struct GaulssLinkRow {
    rows: Vec<GaulssRow>,
}

impl RowProgram<2> for GaulssLinkRow {
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        let r = self
            .rows
            .get(row)
            .ok_or_else(|| format!("GaulssLinkRow: row {row} out of range"))?;
        // Index 0 = η_μ, 1 = η_ls, matching the closed form / channels below.
        Ok([r.eta_mu, r.eta_ls])
    }

    fn eval<S: JetScalar<2>>(&self, row: usize, p: &[S; 2]) -> Result<S, String> {
        let data = self
            .rows
            .get(row)
            .ok_or_else(|| format!("GaulssLinkRow: row {row} out of range"))?;
        let eta_mu = &p[0];
        let eta_ls = &p[1];
        // σ = b + e^{η_ls}  (the production log-b link).
        let sigma = eta_ls.exp().add(&S::constant(LOGB_SIGMA_FLOOR));
        // r = y − η_μ.
        let r = S::constant(data.y).sub(eta_mu);
        // ℓ = a·ln σ + ½·a·(r/σ)².
        let log_term = sigma.ln().scale(data.a);
        let r_over_sigma = r.mul(&sigma.recip());
        let quad = r_over_sigma.mul(&r_over_sigma).scale(0.5 * data.a);
        Ok(log_term.add(&quad))
    }
}

/// INDEPENDENT observed closed-form channels at `row` (NO jet), replicating the
/// production `gaussian_joint_psi_firstweights` OBSERVED scalars verbatim: the
/// score `(−m, κ(a−n))` and the observed Hessian
/// `[[w, 2mκ], [2mκ, 2κ²n + κ'(a−n)]]`. Third/fourth are left empty.
fn gaulss_observed_closed_form(row: &GaulssRow) -> KernelChannels<2> {
    let s_exp = row.eta_ls.exp();
    let sigma = LOGB_SIGMA_FLOOR + s_exp;
    let kappa = s_exp / sigma; // κ = e^{η}/σ  (= dlogσ/dη)
    let kappa_prime = kappa * (1.0 - kappa); // κ' = κ(1 − κ)
    let r = row.y - row.eta_mu;
    let w = row.a / (sigma * sigma); // w = a/σ²
    let m = r * w; // m = r w
    let n = r * r * w; // n = r² w
    let a = row.a;

    // ℓ = a·ln σ + ½·a·r²/σ².
    let value = a * sigma.ln() + 0.5 * a * r * r / (sigma * sigma);

    // ∇ = (−m, κ(a − n))  ==  (scoremu, score_ls).
    let gradient = [-m, kappa * (a - n)];

    // Observed H: [[w, 2mκ], [2mκ, 2κ²n + κ'(a − n)]].
    let h_mu_mu = w;
    let h_mu_ls = 2.0 * m * kappa;
    let h_ls_ls = 2.0 * kappa * kappa * n + kappa_prime * (a - n);
    let hessian = [[h_mu_mu, h_mu_ls], [h_mu_ls, h_ls_ls]];

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

/// The mechanically jet-derived `gaulss` OBSERVED tower (value / ∇ / observed H)
/// in the production log-b σ-link parameterization must equal the INDEPENDENT
/// closed form that replicates the production `gaussian_joint_psi_firstweights`
/// observed scalars, channel by channel, through the SAME universal
/// [`verify_kernel_channels`] oracle every #932 family uses. This pins the hand
/// `κ`-chain-rule score/observed-Hessian cascade (σ = b + e^{η}, κ, κ') against
/// the single-expression row NLL. (The production Newton Hessian's deliberate
/// Fisher substitutions — `(μ,ls)→0`, `(ls,ls)→2κ²a` — are a modeling layer, not
/// jet-tower work, and are intentionally not asserted here.)
#[test]
fn gaulss_link_jet_tower_matches_production_observed_score_and_hessian() {
    let mut rng = Lcg(0x9322_2020_1109_ca75);
    // Moderate ranges keep σ = 0.01 + e^{η_ls} and w = a/σ² finite and well-scaled.
    let mut rows = Vec::new();
    for _ in 0..24 {
        rows.push(GaulssRow {
            y: rng.uniform(-3.0, 3.0),
            eta_mu: rng.uniform(-2.0, 2.0),
            eta_ls: rng.uniform(-1.5, 1.5),
            a: rng.uniform(0.5, 2.5),
        });
    }
    let program = GaulssLinkRow { rows: rows.clone() };

    const REL_TOL: f64 = 1e-11;
    for (row, fixture) in rows.iter().enumerate() {
        let tower: Tower4<2> = program_full_tower(&program, row).expect("gaulss jet tower");
        let claims = gaulss_observed_closed_form(fixture);
        verify_kernel_channels(&tower, &claims, REL_TOL).unwrap_or_else(|e| {
            panic!(
                "row {row}: gaulss production observed κ-chain-rule tower disagrees with \
                 #932 jet-tower truth: {e}"
            )
        });
    }
}
