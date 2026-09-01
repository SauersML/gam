//! Atomic linear-community → curved REPLACEMENT proposals (audit §5 / §34).
//!
//! # The move class the residual-birth path structurally cannot make
//!
//! Every other curved producer in this crate discovers curvature from a
//! *residual*: the structured-residual birth path
//! ([`crate::structure_harvest`]) mines the reconstruction residual `R = x − x̂`
//! for factor directions, and the compose/co-fit lane
//! (`crate::sparse_dict::cofit`) fits charts to the linear tier's
//! least-squares residual. Both share a sufficient statistic that is a function
//! ONLY of the linear residual.
//!
//! There is an exact impossibility result for that statistic. Two linear atoms
//! `u, v` reconstruct a centered ring `x = r·cosθ·u + r·sinθ·v` **exactly** — a
//! circle's mean-zero cone IS its 2-plane, so a nonnegative or signed linear
//! dictionary lawfully parks two directions on it and splits every firing into a
//! co-active `(α, β)`. The linear residual is then identically zero. A discovery
//! rule whose inputs are functions of that residual sees nothing to birth, yet a
//! single curved chart `(θ, r)` encodes the same rows at strictly lower code
//! dimension (one phase, one amplitude vs. two amplitudes). The circle is
//! invisible to residual mining not because the fit is poor but because it is
//! *perfect*.
//!
//! # Compression promotion, not residual birth
//!
//! The missing move is a **compression promotion**: take an already-active group
//! of linear atoms `B` (a "linear community"), read the block's OWN contribution
//! `y_B = Σ_{j∈B} c_j w_j` (the other model components are a fixed offset — NOT
//! the residual after `B`), and adjudicate whether ONE curved chart describes the
//! same rows in fewer description-length bits than the whole linear community.
//! The comparison is ATOMIC: `M_old = {linear atoms of B}` vs.
//! `M_new = {one curved chart replacing B}`, priced directly in bits. It never
//! asks the curved atom to first be added on top of the exact linear
//! reconstruction and then monotonically lower the SSE — that energy barrier
//! (the residual is already zero, so no on-top addition can reduce it) is exactly
//! the bug this move routes around.
//!
//! # Pricing (the DL currency, bits)
//!
//! With `s = |B|` linear atoms, `P` ambient channels, a curved topology of
//! intrinsic dim `d` and basis size `m` matched to the block's ambient span `ŝ`,
//! a per-coordinate distortion floor `δ`, `N` tokens and `f = ρ·N` active
//! firings, the two models cost
//!
//! ```text
//!   DL_old = f·( c_flat + s·log₂(G/L0) )              + s·P·½log₂N
//!   DL_new = f·( c_curved + 1·log₂(G/L0) )            + m·P·½log₂N
//!   c_flat   = Σ_{k∈{α,β}} scalar_rate_bits(varₖ, δ²)   (two amplitude coords)
//!   c_curved = max(0, c_flat − circle_coding_gain_bits(R̂, δ))
//! ```
//!
//! so the atomic saving is
//!
//! ```text
//!   DL_old − DL_new = f·[ circle_coding_gain_bits(R̂, δ) + (s−1)·log₂(G/L0) ]
//!                     − (m−s)·P·½log₂N.
//! ```
//!
//! This is precisely the #2233 crossover ledger
//! ([`crate::description_length::predicted_birth_dl_bits`]) with the exact
//! Theorem-3 circle coding gain in place of the crossover's coarse
//! `(ŝ−d−1)·½log₂(1+λ/δ)` code term (which vanishes for a circle, `ŝ=2, d=1`).
//! The circle therefore pays through the **support** dividend `(s−1)·log₂(G/L0)`
//! — the active slots the single curved atom no longer spends — against the
//! **dictionary surcharge** `(m−s)·P·½log₂N` of the wider harmonic basis. Both
//! scale so that a HIGH firing rate `f` is what tips the atomic ledger positive:
//! the promotion is a genuine compression win exactly when the community fires
//! often enough to amortise the harmonic decoder columns. The zero-residual
//! circle is discovered without any residual energy ever being present.
//!
//! # Pure proposal producer
//!
//! [`propose_curve_promotion`] mutates nothing in the live fit loop. It reads a
//! [`LinearCommunity`] (the block's atoms and code cloud), performs the local-PCA
//! chart geometry, consumes the #2233 crossover as a pre-screen, and emits a
//! typed [`CurvePromotionProposal`]. The structural controller consumes the
//! proposal later; whether `accept` is set is a pure function of the DL ledger,
//! the crossover pre-screen, and the ring geometry screens.

use ndarray::{Array1, Array2, ArrayView2};

use super::curl::{CircleSeed, CurlVerdict, curl_seed, curl_verdict};
use super::geometry_plan::SaeAtomGeometryPlan;
use crate::description_length::{
    BirthMdlPrescreen, circle_coding_gain_bits, predicted_birth_dl_bits, scalar_rate_bits,
};

/// An active Tier-1 linear community `B`: the block's linear atoms and the
/// per-row code cloud on them. The block's OWN contribution is `y_B = C · W`
/// (`row i = Σ_j codes[i,j] · atoms[j]`) — reconstructed here, never the residual
/// after `B`.
#[derive(Clone, Copy, Debug)]
pub struct LinearCommunity<'a> {
    /// Stable identifier of the block the controller would replace.
    pub block_id: usize,
    /// The `s` linear atoms of the block, one ambient direction per row
    /// (`s × P`). Need not be orthonormal — the chart geometry orthonormalises.
    pub atoms: ArrayView2<'a, f64>,
    /// The per-row code coefficients on those atoms over the block's ACTIVE rows
    /// (`f × s`, `f` firings). Row `i` reconstructs `y_B[i] = Σ_j codes[i,j]·wⱼ`.
    pub codes: ArrayView2<'a, f64>,
}

/// Static context the atomic DL ledger prices against.
#[derive(Clone, Copy, Debug)]
pub struct PromotionContext {
    /// Total token count `N` (used for `ρ = f/N` and the `½log₂N` BIC charge).
    pub n_tokens: f64,
    /// Current dictionary size `G` (for the `log₂(G/L0)` support dividend).
    pub g_dict: usize,
    /// Mean active atoms per token `L0` (the support-budget denominator).
    pub l0: f64,
    /// Per-coordinate distortion floor `δ` (a reconstruction-tolerance SCALE, the
    /// RD reference and the quantisation cell the coordinate is coded to). Both
    /// the ring RD screen (`sigma = δ`) and the code bits (`δ²`) read it.
    pub tolerance: f64,
}

/// The atomic linear-community → curved replacement proposal. A PURE product: it
/// mutates no fit state. The controller consumes `curved_candidate` (a race-ready
/// circle seed) iff it decides to act on `accept`.
#[derive(Clone, Debug)]
pub struct CurvePromotionProposal {
    /// The block the curved chart would replace.
    pub block: usize,
    /// Number of linear atoms in the community (`s`).
    pub n_linear_atoms: usize,
    /// The candidate curved chart in the engine's periodic-harmonic layout.
    pub curved_candidate: CircleSeed,
    /// The ring geometry verdict on the block's code cloud (κ, resultants, RD).
    pub verdict: CurlVerdict,
    /// Ambient span `ŝ` (participation ratio of the block's energy spectrum).
    pub span: f64,
    /// Firing rate `ρ = f/N` of the community.
    pub firing_rate: f64,
    /// Total description length of `M_old = {linear atoms of B}`, in bits.
    pub dl_old: f64,
    /// Total description length of `M_new = {curved chart replacing B}`, in bits.
    pub dl_new: f64,
    /// The #2233 crossover pre-screen: predicted net DL saving of the curved
    /// birth over the flat span from spectra alone
    /// ([`crate::description_length::predicted_birth_dl_bits`]). A positive value
    /// is the necessary pre-screen; the atomic `dl_new < dl_old` is the decision.
    pub crossover_prescreen_bits: f64,
    /// `true` iff the crossover pre-screen is positive, the ring geometry screens
    /// pass, AND the atomic ledger strictly prefers the curved chart
    /// (`dl_new < dl_old`). Never depends on residual explained variance.
    pub accept: bool,
}

/// The participation ratio `(Σλ)² / Σλ²` of a non-negative energy spectrum — the
/// effective number of significant ambient directions the cloud occupies
/// (circle ≈ 2). Zero on a degenerate spectrum.
fn participation_ratio(spectrum: &[f64]) -> f64 {
    let sum: f64 = spectrum.iter().map(|&e| e.max(0.0)).sum();
    let sum_sq: f64 = spectrum.iter().map(|&e| e.max(0.0) * e.max(0.0)).sum();
    if sum_sq > 0.0 {
        (sum * sum) / sum_sq
    } else {
        0.0
    }
}

/// The curved topology `(intrinsic_dim d, basis_size m)` matched to an ambient
/// span. A 2-plane span promotes to a circle (`d=1`, `m=2·d+1=3` harmonic rows);
/// higher spans to the sphere/torus atoms.
///
/// #2749: this was a SECOND transcription of the structured-birth path's
/// span→topology map, and both copies priced the sphere at the basis width of
/// the `(lat, lon)` chart deleted in `1dfa70140`. There is now one definition —
/// [`SaeAtomGeometryPlan::curved_prescreen_atom_for_span`] — and both call sites
/// read `d` and `m` off the plan it builds, so this pre-screen and the one in
/// `structure_harvest` cannot drift apart, or away from the birth race, again.
fn curved_topology_for_span(span: f64) -> Result<(usize, usize), String> {
    let plan = SaeAtomGeometryPlan::curved_prescreen_atom_for_span(span)?;
    Ok((plan.intrinsic_dim(), plan.basis_size()?))
}

#[cfg(test)]
mod curve_promotion_tests {
    use super::*;
    use std::f64::consts::TAU;

    /// Build a community whose block is TWO orthonormal linear atoms `e0, e1` in
    /// `P` ambient dims, whose code cloud is a clean radius-`R` ring at `n`
    /// evenly-spaced phases: `codes[i] = (R·cosθ_i, R·sinθ_i)`. The reconstructed
    /// `y_B` is a PERFECT circle exactly represented by the two linear atoms, so
    /// the linear residual of the block is identically zero.
    fn ring_community(n: usize, radius: f64, p: usize) -> (Array2<f64>, Array2<f64>) {
        let mut atoms = Array2::<f64>::zeros((2, p));
        atoms[[0, 0]] = 1.0;
        atoms[[1, 1]] = 1.0;
        let mut codes = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let theta = TAU * (i as f64) / (n as f64);
            codes[[i, 0]] = radius * theta.cos();
            codes[[i, 1]] = radius * theta.sin();
        }
        (atoms, codes)
    }

    #[test]
    fn zero_residual_circle_is_proposed_and_accepted_by_dl() {
        // A perfect circle exactly reconstructed by two linear atoms: the block's
        // linear residual is ZERO. The promotion must still be PROPOSED and, at a
        // high firing rate, ACCEPTED by the atomic DL ledger — proving the move
        // does not depend on any residual explained variance.
        let n = 512;
        let radius = 1.0;
        let p = 16;
        let (atoms, codes) = ring_community(n, radius, p);

        // Confirm the planted premise by reconstructing y_B in FULL ambient space
        // from the atoms: a clean ring lies exactly in the {e0,e1} plane (zero
        // energy in every other channel) at constant radius. There is no residual
        // structure anywhere for a residual-birth path to mine.
        let mut max_offplane = 0.0_f64;
        let mut min_r = f64::INFINITY;
        let mut max_r = 0.0_f64;
        for i in 0..n {
            let mut y = Array1::<f64>::zeros(p);
            for j in 0..2 {
                let wj = atoms.row(j);
                for out in 0..p {
                    y[out] += codes[[i, j]] * wj[out];
                }
            }
            for out in 2..p {
                max_offplane = max_offplane.max(y[out].abs());
            }
            let ri = (y[0] * y[0] + y[1] * y[1]).sqrt();
            min_r = min_r.min(ri);
            max_r = max_r.max(ri);
        }
        assert!(
            max_offplane < 1.0e-12,
            "planted ring must live exactly in the 2-plane (off-plane max {max_offplane})"
        );
        assert!(
            (max_r - min_r).abs() < 1.0e-9 && (max_r - radius).abs() < 1.0e-9,
            "planted ring must have constant radius R (min {min_r}, max {max_r})"
        );

        // High firing rate: the whole community fires (f = N), so the support
        // dividend amortises the harmonic decoder surcharge.
        let community = LinearCommunity {
            block_id: 7,
            atoms: atoms.view(),
            codes: codes.view(),
        };
        let ctx = PromotionContext {
            n_tokens: n as f64,
            g_dict: 4096,
            l0: 8.0,
            tolerance: 0.05,
        };
        let proposal = propose_curve_promotion(community, &ctx)
            .expect("proposal producer runs")
            .expect("a 2-plane ring must yield a proposal");

        // (1) It is PROPOSED for the right block, as a 2-atom → circle replacement.
        assert_eq!(proposal.block, 7);
        assert_eq!(proposal.n_linear_atoms, 2);

        // (2) The ring geometry is recognised: span ≈ 2, κ ≈ 1 (ring, sub-Gaussian).
        assert!(
            (proposal.span - 2.0).abs() < 0.05,
            "clean ring spans a 2-plane (span={})",
            proposal.span
        );
        assert!(
            proposal.verdict.kappa < 1.5,
            "clean ring radius law is sub-Gaussian (κ={})",
            proposal.verdict.kappa
        );

        // (3) The #2233 crossover pre-screen is positive at this firing rate.
        assert!(
            proposal.crossover_prescreen_bits > 0.0,
            "crossover pre-screen must pay at high firing rate (bits={})",
            proposal.crossover_prescreen_bits
        );

        // (4) The ATOMIC ledger strictly prefers the curved chart, and the
        //     proposal is ACCEPTED — with zero residual anywhere in the pipeline.
        assert!(
            proposal.dl_new < proposal.dl_old,
            "curved chart must cost fewer bits (dl_new={}, dl_old={})",
            proposal.dl_new,
            proposal.dl_old
        );
        assert!(
            proposal.accept,
            "zero-residual circle must be accepted by DL (prescreen={}, dl_old={}, dl_new={}, recommend={})",
            proposal.crossover_prescreen_bits,
            proposal.dl_old,
            proposal.dl_new,
            proposal.verdict.recommend_curl
        );
    }

    #[test]
    fn low_firing_rate_defers_the_same_ring() {
        // The identical ring geometry, but fired by only a tiny fraction of a huge
        // token budget, must NOT be accepted: the support dividend f·(s−1)·log₂(G/L0)
        // can no longer amortise the (m−s)·P·½log₂N harmonic decoder surcharge. This
        // shows acceptance is driven by the compression ledger, not by geometry alone.
        let n = 32;
        let radius = 1.0;
        let p = 256;
        let (atoms, codes) = ring_community(n, radius, p);
        let community = LinearCommunity {
            block_id: 3,
            atoms: atoms.view(),
            codes: codes.view(),
        };
        let ctx = PromotionContext {
            n_tokens: 5.0e6, // f = 64 firings out of 5M tokens ⇒ ρ ≈ 1.3e-5
            g_dict: 4096,
            l0: 8.0,
            tolerance: 0.05,
        };
        let proposal = propose_curve_promotion(community, &ctx)
            .expect("runs")
            .expect("still yields a proposal");

        // The ring is still recognised geometrically (same cloud) ...
        assert!(proposal.verdict.recommend_curl || proposal.span > 1.9);
        // ... but the atomic ledger refuses to pay for the decoder columns.
        assert!(
            proposal.dl_new > proposal.dl_old,
            "at ρ≈1e-5 the curved decoder surcharge is not amortised (dl_new={}, dl_old={})",
            proposal.dl_new,
            proposal.dl_old
        );
        assert!(
            !proposal.accept,
            "a rarely-firing ring must be deferred by the compression ledger"
        );
    }

    #[test]
    fn collinear_community_yields_no_proposal() {
        // A community whose code cloud is a LINE (only one active plane direction)
        // has no 2-plane to host a ring; the producer returns None rather than a
        // spurious circle proposal.
        let n = 128;
        let p = 8;
        let mut atoms = Array2::<f64>::zeros((2, p));
        atoms[[0, 0]] = 1.0;
        atoms[[1, 1]] = 1.0;
        let mut codes = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = (i as f64) / (n as f64) - 0.5;
            // Both codes proportional ⇒ y_B rides a single ambient line.
            codes[[i, 0]] = t;
            codes[[i, 1]] = 2.0 * t;
        }
        let community = LinearCommunity {
            block_id: 1,
            atoms: atoms.view(),
            codes: codes.view(),
        };
        let ctx = PromotionContext {
            n_tokens: n as f64,
            g_dict: 4096,
            l0: 8.0,
            tolerance: 0.05,
        };
        let out = propose_curve_promotion(community, &ctx).expect("runs");
        assert!(
            out.is_none(),
            "a collinear (rank-1) community must not yield a ring proposal"
        );
    }

    #[test]
    fn eigendecomposition_matches_known_symmetric_matrix() {
        // Sanity on the Jacobi solver: a 2×2 with known spectrum. Eigenvalues of
        // [[2,1],[1,2]] are 3 and 1.
        let m = ndarray::arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let (vals, vecs) = jacobi_symmetric_eig(&m);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted[0] - 3.0).abs() < 1.0e-10, "top eig {}", sorted[0]);
        assert!((sorted[1] - 1.0).abs() < 1.0e-10, "low eig {}", sorted[1]);
        // Eigenvectors orthonormal.
        let c0 = vecs.column(0).to_owned();
        let c1 = vecs.column(1).to_owned();
        assert!((c0.dot(&c0) - 1.0).abs() < 1.0e-10);
        assert!(c0.dot(&c1).abs() < 1.0e-10);
    }
}
