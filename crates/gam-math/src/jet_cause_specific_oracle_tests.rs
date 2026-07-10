//! Cause-specific Royston-Parmar survival Taylor-jet oracle (#932, FD-free).
//!
//! The cause-specific competing-risks survival block
//! (`gam-models/src/survival/base.rs`) is the last hand-coded derivative tower in
//! the survival family: `cause_specific_hessian_directional_derivative` (third
//! derivative) and `cause_specific_hessian_second_directional_derivative`
//! (fourth) assemble the β-space Hessian directional derivatives from per-row
//! scalar weights built by explicit chain rules. This module single-sources
//! those closed forms onto the jet.
//!
//! # The model
//!
//! Royston-Parmar on the log-cumulative-hazard scale (`η = log H(t)`). Each row
//! has three β-linear predictors — exit index `η1 = a1ᵀβ + o_X`, entry index
//! `η0 = a0ᵀβ + o_E`, and the spline derivative `s = dᵀβ + o_D > 0` — plus a
//! prior weight `w`, an event indicator `δ ∈ {0,1}`, and a left-truncation flag
//! `1{entry}`. The per-row negative log-likelihood (base.rs:2161) is
//!
//! ```text
//!   ℓ = w·[ exp(η1) − 1{entry}·exp(η0) − δ·(η1 + log s) ].
//! ```
//!
//! Because the three predictors are additively separable in `ℓ`, the derivative
//! tower is DIAGONAL in `(η1, η0, s)` space, and the production per-row weights
//! are exactly its diagonal derivatives:
//!
//! ```text
//!   ∇      = [ w(exp η1 − δ),  −w·1{entry}·exp η0,  −wδ/s ]     (offset residuals)
//!   ∂²     diag = [ w exp η1,  −w·1{entry} exp η0,   wδ/s²  ]
//!   ∂³     diag = [ w exp η1,  −w·1{entry} exp η0,  −2wδ/s³ ]   ← directional weights
//!   ∂⁴     diag = [ w exp η1,  −w·1{entry} exp η0,   6wδ/s⁴ ]   ← 2nd-directional weights
//! ```
//!
//! The production `cause_specific_hessian_directional_derivative` builds
//! `w_exit = w·exp(η1)·η̇1`, `w_entry = w·exp(η0)·η̇0` (assembled with a MINUS),
//! and `w_derivative = −2w·ṡ/s³` (event rows), then contracts each through its
//! design block — i.e. `xᵀ diag(∂³·direction) x`. The `_second_directional_`
//! variant uses `w·exp·u̇·v̇` and `6w·u̇·v̇/s⁴`. This oracle writes `ℓ` ONCE as a
//! `RowNllProgramGeneric<3>` and pins the jet's contracted third/fourth against
//! those EXACT production per-row weight formulas at `rel_tol = 1e-11`.

use crate::jet_scalar::JetScalar;
use crate::jet_tower::{
    KernelChannels, RowNllProgramGeneric, Tower4, generic_full_tower, verify_kernel_channels,
};

/// One cause-specific Royston-Parmar fixture. `s` (the spline derivative) is kept
/// strictly positive, the finite regime the production formulas require.
#[derive(Clone, Copy, Debug)]
struct CauseRow {
    eta1: f64,
    eta0: f64,
    s: f64,
    w: f64,
    /// Event indicator δ ∈ {0.0, 1.0}.
    delta: f64,
    /// Left-truncation: whether the entry (exp η0) term is present.
    has_entry: bool,
}

/// The cause-specific row NLL written ONCE as a generic [`RowNllProgramGeneric<3>`]
/// over the jet scalar `S`. Primary order: 0 = η1 (exit), 1 = η0 (entry), 2 = s
/// (spline derivative). The `1{entry}` / `δ` indicators fold the entry and event
/// terms in or out — a dropped term contributes exact-zero derivative channels,
/// exactly as the production per-row gates (`has_entry`, `event_target > 0`) do.
struct CauseSpecificRow {
    rows: Vec<CauseRow>,
}

impl RowNllProgramGeneric<3> for CauseSpecificRow {
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
        let r = self
            .rows
            .get(row)
            .ok_or_else(|| format!("CauseSpecificRow: row {row} out of range"))?;
        Ok([r.eta1, r.eta0, r.s])
    }

    fn row_nll_generic<S: JetScalar<3>>(&self, row: usize, p: &[S; 3]) -> Result<S, String> {
        let data = self
            .rows
            .get(row)
            .ok_or_else(|| format!("CauseSpecificRow: row {row} out of range"))?;
        let eta1 = &p[0];
        let eta0 = &p[1];
        let s = &p[2];
        // exp(η1) exit term.
        let mut ell = eta1.exp();
        // − 1{entry}·exp(η0) entry (left-truncation) term.
        if data.has_entry {
            ell = ell.sub(&eta0.exp());
        }
        // − δ·(η1 + log s) event term.
        if data.delta != 0.0 {
            ell = ell.sub(&eta1.add(&s.ln()).scale(data.delta));
        }
        // Prior weight.
        Ok(ell.scale(data.w))
    }
}

/// INDEPENDENT closed-form channels replicating the production per-row weight
/// formulas verbatim: gradient = the offset-channel residuals, and the third /
/// fourth contracted tensors = `cause_specific_hessian_(second_)directional_derivative`
/// per-row weights, contracted with the supplied directions. The tower is
/// diagonal in `(η1, η0, s)`, so all off-diagonal channel entries are exact 0.
fn cause_specific_closed_form(
    row: &CauseRow,
    third_dirs: &[[f64; 3]],
    fourth_pairs: &[([f64; 3], [f64; 3])],
) -> KernelChannels<3> {
    let w = row.w;
    let e1 = row.eta1.exp();
    let e0 = row.eta0.exp();
    let entry = if row.has_entry { 1.0 } else { 0.0 };
    let s = row.s;
    let d = row.delta;

    let value = w * (e1 - entry * e0 - d * (row.eta1 + s.ln()));

    // ∇ = [ w(exp η1 − δ), −w·1{entry}·exp η0, −wδ/s ].
    let gradient = [w * (e1 - d), -w * entry * e0, -w * d / s];

    // Diagonal 2nd/3rd/4th derivatives per predictor.
    // η1: exp(η1) at every order.  η0: −1{entry}·exp(η0).  s: −wδ log s ⇒
    //   ∂² = wδ/s², ∂³ = −2wδ/s³, ∂⁴ = 6wδ/s⁴.
    let h_diag = [w * e1, -w * entry * e0, w * d / (s * s)];
    let t3_diag = [w * e1, -w * entry * e0, -2.0 * w * d / (s * s * s)];
    let t4_diag = [w * e1, -w * entry * e0, 6.0 * w * d / (s * s * s * s)];

    let mut hessian = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        hessian[a][a] = h_diag[a];
    }

    // Third contracted: out[a][b] = Σ_c t3[a][b][c]·dir[c]; diagonal-only tower ⇒
    // nonzero only when a == b == c, giving out[a][a] = t3_diag[a]·dir[a].
    let third = third_dirs
        .iter()
        .map(|dir| {
            let mut m = [[0.0_f64; 3]; 3];
            for a in 0..3 {
                m[a][a] = t3_diag[a] * dir[a];
            }
            (*dir, m)
        })
        .collect();

    // Fourth contracted: out[a][a] = t4_diag[a]·u[a]·v[a].
    let fourth = fourth_pairs
        .iter()
        .map(|(u, v)| {
            let mut m = [[0.0_f64; 3]; 3];
            for a in 0..3 {
                m[a][a] = t4_diag[a] * u[a] * v[a];
            }
            (*u, *v, m)
        })
        .collect();

    KernelChannels {
        value,
        gradient,
        hessian,
        third,
        fourth,
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

/// The mechanically jet-derived cause-specific Royston-Parmar tower (value / ∇ /
/// H / contracted-third / contracted-fourth) equals the INDEPENDENT closed form
/// that replicates the production directional-derivative per-row weights, channel
/// by channel, through the universal [`verify_kernel_channels`] oracle. Fixtures
/// span all four (event × entry) corners so the `δ` / `1{entry}` gates — which
/// zero out whole predictor channels exactly as the production `event_target > 0`
/// / `has_entry` gates do — are each exercised.
#[test]
fn cause_specific_royston_parmar_jet_tower_matches_production_directional_weights() {
    let mut rng = Lcg(0x9322_2020_1109_5171);
    let third_dirs: [[f64; 3]; 3] = [[0.7, -1.3, 0.5], [-0.4, 0.6, -0.9], [1.2, 0.2, 0.3]];
    let fourth_pairs: [([f64; 3], [f64; 3]); 3] = [
        ([0.7, -1.3, 0.5], [-0.4, 0.6, -0.9]),
        ([-0.4, 0.6, -0.9], [1.2, 0.2, 0.3]),
        ([1.2, 0.2, 0.3], [0.7, -1.3, 0.5]),
    ];

    let mut rows = Vec::new();
    for i in 0..24 {
        rows.push(CauseRow {
            // Moderate η keep exp(η) well-scaled; s strictly positive.
            eta1: rng.uniform(-1.5, 1.5),
            eta0: rng.uniform(-1.5, 1.5),
            s: rng.uniform(0.2, 3.0),
            w: rng.uniform(0.4, 2.5),
            delta: if i % 2 == 0 { 1.0 } else { 0.0 },
            has_entry: i % 3 != 0,
        });
    }
    let program = CauseSpecificRow { rows: rows.clone() };

    const REL_TOL: f64 = 1e-11;
    for (row, fixture) in rows.iter().enumerate() {
        let tower: Tower4<3> = generic_full_tower(&program, row).expect("cause-specific jet tower");
        let claims = cause_specific_closed_form(fixture, &third_dirs, &fourth_pairs);
        verify_kernel_channels(&tower, &claims, REL_TOL).unwrap_or_else(|e| {
            panic!(
                "row {row}: cause-specific Royston-Parmar production directional-derivative \
                 weights disagree with #932 jet-tower truth: {e}"
            )
        });
    }
}
