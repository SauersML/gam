//! Atomic linear-community → curved REPLACEMENT proposals (audit §5 / §34).
//!
//! # The move class the residual-birth path structurally cannot make
//!
//! Every other curved producer in this crate discovers curvature from a
//! *residual*: the structured-residual birth path
//! ([`crate::structure_harvest`]) mines the reconstruction residual `R = x − x̂`
//! for factor directions, and the tiered curved refinement
//! ([`crate::tiered::fit_tiered`]) fits charts to the Tier-1 linear peel's
//! residual. Both share a sufficient statistic that is a function
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
//!   c_curved = log₂ M                                  (one phase index)
//! ```
//!
//! `c_flat` is the Gaussian surrogate rate of the ring plane's two amplitudes,
//! which spends the in-plane distortion `D = Σₖ min(varₖ, δ²)`. `c_curved` is the
//! rate of the least circle phase codebook whose exact distortion on the cloud's
//! own radial moments fits that same `D`
//! ([`crate::description_length::circle_phase_code`], #2933 F23). The ring's
//! radial spread is fitting error, so it and the quantization error share the one
//! budget; when the spread alone exceeds `D` no phase code matches the flat arm's
//! fidelity and `DL_new = +∞`.
//!
//! The ledger sets a Gaussian surrogate against a finite codebook and prices
//! support and decoder storage with per-slot and BIC-style charges, so it is a
//! declared HEURISTIC comparison
//! ([`crate::description_length::ScoreComparison::ExplicitHeuristic`]), not a
//! certified message-length difference. The circle pays through the one-phase
//! versus two-amplitude code saving and the **support** dividend
//! `(s−1)·log₂(G/L0)`, against the **dictionary surcharge** `(m−s)·P·½log₂N` of
//! the wider harmonic basis; a HIGH firing rate `f` is what amortises the harmonic
//! decoder columns. The zero-residual circle is discovered without any residual
//! energy ever being present.
//!
//! # Pure proposal producer
//!
//! [`propose_curve_promotion`] mutates nothing in the live fit loop. It reads a
//! [`LinearCommunity`] (the block's atoms and code cloud), performs the local-PCA
//! chart geometry, records the #2233 birth proposal priority, and emits a typed
//! [`CurvePromotionProposal`]. The structural controller consumes the proposal
//! later; whether `accept` is set is a pure function of the DL ledger and the ring
//! recognition (κ, coverage, no diameter). The priority is a spectra-only heuristic
//! and never vetoes it (#2933 F22), and neither does the small-cell circle screen
//! of [`super::curl::curl_verdict`], which prices no support dividend (#2933 F23).

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView2};

use super::curl::{CircleSeed, RingRecognition, curl_seed, ring_recognition};
use crate::description_length::{
    BirthMdlPrescreen, BirthProposalPriority, CirclePhaseCode, DescriptionLengthScoreKind,
    ScoreComparison, ScoredBits, birth_proposal_priority, circle_phase_code,
    description_length_delta, scalar_rate_bits,
};
// One span estimate and one span→topology map for the #2233 pre-screen, shared
// with the residual-birth path so the two producers cannot price the same span
// differently (#2749).
use crate::structure_harvest::{curved_topology_for_span, participation_ratio};

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
    /// Per-coordinate distortion floor `δ` (a reconstruction-tolerance SCALE and the
    /// quantisation cell the coordinate is coded to). The code bits (`δ²`) and the
    /// phase code's in-plane budget read it. It is not a noise annulus, so the ring
    /// recognition never reads it.
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
    /// The ring recognition on the block's code cloud (κ, resultants, coverage).
    pub verdict: RingRecognition,
    /// Ambient span `ŝ` (participation ratio of the block's energy spectrum).
    pub span: f64,
    /// Firing rate `ρ = f/N` of the community.
    pub firing_rate: f64,
    /// Total description length of `M_old = {linear atoms of B}`, in bits.
    pub dl_old: f64,
    /// Total description length of `M_new = {curved chart replacing B}`, in bits
    /// (`+∞` when no circle phase code meets the flat arm's in-plane distortion).
    pub dl_new: f64,
    /// The #2233 birth proposal priority of the curved birth over the flat span,
    /// from spectra alone ([`crate::description_length::birth_proposal_priority`]).
    /// An ordering and calibration heuristic: reported, never a veto on `accept`.
    pub crossover_prescreen: BirthProposalPriority,
    /// The least circle phase codebook meeting the flat arm's in-plane distortion,
    /// or `None` when the ring's radial spread alone exceeds that distortion.
    pub curved_phase_code: Option<CirclePhaseCode>,
    /// `true` iff the ring is recognized AND the atomic ledger strictly prefers the
    /// curved chart (`dl_new < dl_old`). Never depends on residual explained
    /// variance, on the birth proposal priority, or on the small-cell circle screen.
    pub accept: bool,
}

/// Adjudicate the atomic replacement of a linear community by a single curved
/// chart. Returns `Ok(None)` when the community's own contribution has fewer than
/// two effective ambient dimensions (no 2-plane to host a ring); `Ok(Some(_))`
/// otherwise, with `accept` decided purely by the DL ledger and ring recognition.
pub fn propose_curve_promotion(
    community: LinearCommunity<'_>,
    ctx: &PromotionContext,
) -> Result<Option<CurvePromotionProposal>, String> {
    let (f, s) = community.codes.dim();
    let (s_atoms, p) = community.atoms.dim();
    if s_atoms != s {
        return Err(format!(
            "curve_promotion: codes have s={s} columns but atoms have s={s_atoms} rows"
        ));
    }
    if s < 2 {
        return Ok(None);
    }
    if f < 2 {
        return Ok(None);
    }
    if !(ctx.tolerance > 0.0 && ctx.tolerance.is_finite()) {
        return Err(format!(
            "curve_promotion: tolerance must be finite and > 0, got {}",
            ctx.tolerance
        ));
    }
    if !(ctx.n_tokens >= f as f64) {
        return Err(format!(
            "curve_promotion: n_tokens {} must be >= firings {f}",
            ctx.n_tokens
        ));
    }

    // ---- Local PCA of the block's OWN contribution in an orthonormal block
    //      basis. Gram–Schmidt the atoms to an orthonormal ambient frame Q, read
    //      the ambient coords Z = Y_B·Qᵀ = codes·(W·Qᵀ) (no P-sized intermediate),
    //      and eigendecompose the small r×r coord covariance.
    let q = gram_schmidt(community.atoms);
    let r = q.len();
    if r < 2 {
        // The block span collapses to a line (or point): no plane for a ring.
        return Ok(None);
    }
    // B[j][k] = wⱼ · q_k, so Z = codes · B (f × r) without materialising Y_B.
    let mut bmat = Array2::<f64>::zeros((s, r));
    for j in 0..s {
        let wj = community.atoms.row(j);
        for (k, qk) in q.iter().enumerate() {
            bmat[[j, k]] = wj.dot(qk);
        }
    }
    let mut z = Array2::<f64>::zeros((f, r));
    for i in 0..f {
        for k in 0..r {
            let mut acc = 0.0;
            for j in 0..s {
                acc += community.codes[[i, j]] * bmat[[j, k]];
            }
            z[[i, k]] = acc;
        }
    }
    // Column means (the ambient center in the Q frame) and centered coords.
    let mut zmean = Array1::<f64>::zeros(r);
    for k in 0..r {
        zmean[k] = z.column(k).sum() / f as f64;
    }
    for i in 0..f {
        for k in 0..r {
            z[[i, k]] -= zmean[k];
        }
    }
    // Coord covariance (r × r) and its symmetric eigendecomposition.
    let mut cov = Array2::<f64>::zeros((r, r));
    for a in 0..r {
        for b in 0..r {
            let mut acc = 0.0;
            for i in 0..f {
                acc += z[[i, a]] * z[[i, b]];
            }
            cov[[a, b]] = acc / f as f64;
        }
    }
    let (eigvals, eigvecs) = cov.eigh(Side::Lower).map_err(|error| {
        format!("curve_promotion: coordinate covariance eigendecomposition: {error}")
    })?;
    // Descending eigenvalue order.
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| eigvals[b].total_cmp(&eigvals[a]));
    let sorted_vals: Vec<f64> = order.iter().map(|&i| eigvals[i].max(0.0)).collect();
    let span = participation_ratio(&sorted_vals);
    if sorted_vals[1] <= 0.0 {
        // The second principal energy is zero — a line, not a ring.
        return Ok(None);
    }

    // Ring plane coords (α, β): projection of Z onto the top-2 principal axes.
    let psi1 = eigvecs.column(order[0]).to_owned();
    let psi2 = eigvecs.column(order[1]).to_owned();
    let mut alpha = Array1::<f64>::zeros(f);
    let mut beta = Array1::<f64>::zeros(f);
    for i in 0..f {
        let zi = z.row(i);
        alpha[i] = zi.dot(&psi1);
        beta[i] = zi.dot(&psi2);
    }

    // ---- Ring recognition on the code cloud (κ, resultants). Whether the ring pays is
    //      the atomic ledger's question below: the small-cell circle screen prices no
    //      support dividend, and read at `sigma = δ` it would debias a noiseless ring
    //      by a noise annulus the codes do not carry (#2933 F23).
    let verdict = ring_recognition(alpha.view(), beta.view())?;

    // Curved topology matched to the ambient span (circle ŝ≈2 ⇒ (d,m)=(1,3)).
    let (d, m) = curved_topology_for_span(span)?;
    let harmonics = (m.max(1) - 1) / 2;

    // ---- Ambient plane frame e1,e2 (orthonormal — Q is orthonormal and ψ are
    //      orthonormal) and the ambient center, for the race-ready circle seed.
    let mut e1 = Array1::<f64>::zeros(p);
    let mut e2 = Array1::<f64>::zeros(p);
    let mut center = Array1::<f64>::zeros(p);
    for k in 0..r {
        let qk = &q[k];
        for out in 0..p {
            e1[out] += psi1[k] * qk[out];
            e2[out] += psi2[k] * qk[out];
            center[out] += zmean[k] * qk[out];
        }
    }
    let curved_candidate = curl_seed(
        e1.view(),
        e2.view(),
        alpha.view(),
        beta.view(),
        harmonics.max(1),
        center.view(),
    )?;

    // ---- Atomic DL ledger (bits). δ is the distortion SCALE: each flat amplitude
    //      is coded at variance δ², and the curved phase code must meet the same
    //      in-plane distortion the flat arm spends.
    let delta = ctx.tolerance;
    let delta2 = delta * delta;
    let var_alpha = alpha.iter().map(|&a| a * a).sum::<f64>() / f as f64;
    let var_beta = beta.iter().map(|&b| b * b).sum::<f64>() / f as f64;
    let c_flat = scalar_rate_bits(var_alpha, delta2) + scalar_rate_bits(var_beta, delta2);
    let flat_distortion = var_alpha.min(delta2) + var_beta.min(delta2);
    // The cloud's radial moments. The spread about the mean radius is summed
    // directly, not formed as E[r²] − E[r]², which cancels at the resolution floor.
    let radii: Vec<f64> = alpha
        .iter()
        .zip(beta.iter())
        .map(|(&a, &b)| a.hypot(b))
        .collect();
    let mean_radius = radii.iter().sum::<f64>() / f as f64;
    let radial_variance = radii
        .iter()
        .map(|&radius| (radius - mean_radius) * (radius - mean_radius))
        .sum::<f64>()
        / f as f64;
    let curved_phase_code = circle_phase_code(mean_radius, radial_variance, flat_distortion)?;
    let c_curved = curved_phase_code.map_or(f64::INFINITY, |code| code.rate_bits);

    let unit_sel = if ctx.g_dict > 0 && ctx.l0 > 0.0 {
        (ctx.g_dict as f64 / ctx.l0).log2().max(0.0)
    } else {
        0.0
    };
    let l_param = if ctx.n_tokens >= 2.0 {
        0.5 * ctx.n_tokens.log2()
    } else {
        0.0
    };
    let s_f = s as f64;
    let m_f = m as f64;
    let p_f = p as f64;
    let dl_old = f as f64 * (c_flat + s_f * unit_sel) + s_f * p_f * l_param;
    let dl_new = f as f64 * (c_curved + unit_sel) + m_f * p_f * l_param;

    // ---- #2233 birth proposal priority (spectra only), reported for ordering and
    //      calibration. A heuristic, so it may not veto the ledger (#2933 F22).
    let firing_rate = f as f64 / ctx.n_tokens;
    let crossover_prescreen = birth_proposal_priority(&BirthMdlPrescreen {
        rho: firing_rate,
        span,
        intrinsic_dim: d,
        basis_size: m,
        signal_var: sorted_vals[0],
        noise_floor: delta2,
        n_tokens: ctx.n_tokens,
        p_out: p,
        g_dict: ctx.g_dict,
        l0: ctx.l0,
    });

    // The flat arm is a Gaussian surrogate and the curved arm a finite codebook:
    // a declared heuristic comparison, not a certified length difference.
    let ledger_saving = description_length_delta(
        ScoredBits {
            bits: dl_old,
            kind: DescriptionLengthScoreKind::GaussianSurrogate,
        },
        ScoredBits {
            bits: dl_new,
            kind: DescriptionLengthScoreKind::FiniteQuantizer,
        },
        ScoreComparison::ExplicitHeuristic,
    )?;
    let accept = verdict.recognized && ledger_saving > 0.0;

    Ok(Some(CurvePromotionProposal {
        block: community.block_id,
        n_linear_atoms: s,
        curved_candidate,
        verdict,
        span,
        firing_rate,
        dl_old,
        dl_new,
        crossover_prescreen,
        curved_phase_code,
        accept,
    }))
}

/// Gram–Schmidt an `s × P` set of ambient atom directions into an orthonormal
/// ambient frame, dropping directions that are (numerically) already in the span.
/// The retained count `r ≤ s` is the effective rank of the block span.
fn gram_schmidt(atoms: ArrayView2<'_, f64>) -> Vec<Array1<f64>> {
    let (s, p) = atoms.dim();
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(s);
    // A direction lying in the span of the stored bases must come out of modified
    // Gram–Schmidt as rounding, and nothing more. Each projection forms one
    // length-`P` inner product and updates every entry with a product and a
    // subtraction, leaking at most `γ_{P+4}·‖w‖`; it also leaks what the stored bases'
    // own loss of orthogonality leaves behind, at most `Σ ω·‖w‖`. After `k` projections
    // a dependent residual stays inside `k·(γ_{P+4} + Σ ω)·‖w‖`, and a basis stored
    // from residual `r` of row `w` carries `ω = band·‖w‖/‖r‖`.
    let projection_growth = gam_linalg::roundoff::accumulation_growth(p + 4);
    let mut orthogonality_defect = 0.0_f64;
    for j in 0..s {
        let mut v = atoms.row(j).to_owned();
        for q in &basis {
            let proj = v.dot(q);
            v.scaled_add(-proj, q);
        }
        let norm = v.dot(&v).sqrt();
        let row_norm = atoms.row(j).dot(&atoms.row(j)).sqrt();
        let band = basis.len() as f64 * (projection_growth + orthogonality_defect);
        if norm > band * row_norm {
            orthogonality_defect += band * row_norm / norm;
            v.mapv_inplace(|x| x / norm);
            basis.push(v);
        }
    }
    basis
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

        // (3) The #2233 priority is finite and positive at this firing rate (a
        //     reported ordering key; it does not decide acceptance).
        assert!(
            proposal.crossover_prescreen.bits().is_some_and(|bits| bits > 0.0),
            "crossover priority must pay at high firing rate: {:?}",
            proposal.crossover_prescreen
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
            "zero-residual circle must be accepted by DL (prescreen={:?}, dl_old={}, dl_new={}, recognized={})",
            proposal.crossover_prescreen,
            proposal.dl_old,
            proposal.dl_new,
            proposal.verdict.recognized
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
        assert!(proposal.verdict.recognized || proposal.span > 1.9);
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
        // Sanity on the eigendecomposition the ring verdict reads: a 2×2 with known
        // spectrum. Eigenvalues of
        // [[2,1],[1,2]] are 3 and 1.
        let m = ndarray::arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let (vals, vecs) = m.eigh(Side::Lower).expect("2×2 symmetric eigendecomposition");
        let mut sorted = vals.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted[0] - 3.0).abs() < 1.0e-10, "top eig {}", sorted[0]);
        assert!((sorted[1] - 1.0).abs() < 1.0e-10, "low eig {}", sorted[1]);
        // Eigenvectors orthonormal.
        let c0 = vecs.column(0).to_owned();
        let c1 = vecs.column(1).to_owned();
        assert!((c0.dot(&c0) - 1.0).abs() < 1.0e-10);
        assert!(c0.dot(&c1).abs() < 1.0e-10);
    }

    /// A ring community whose phases sit at the centres of `n` equal arcs.
    fn centred_ring_community(n: usize, radius: f64, p: usize) -> (Array2<f64>, Array2<f64>) {
        let mut atoms = Array2::<f64>::zeros((2, p));
        atoms[[0, 0]] = 1.0;
        atoms[[1, 1]] = 1.0;
        let mut codes = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let theta = TAU * (i as f64 + 0.5) / n as f64;
            codes[[i, 0]] = radius * theta.cos();
            codes[[i, 1]] = radius * theta.sin();
        }
        (atoms, codes)
    }

    /// Encode each in-plane point to the index of its angular cell
    /// `[2πk/M, 2π(k+1)/M)`, decode it to `ρ·(cos φ_k, sin φ_k)` at the cell centre
    /// `φ_k`, and return the mean squared reconstruction error. `ρ` is the
    /// least-squares decoder radius of this codebook, `mean ⟨x, e_k⟩`; with one
    /// cell the centre direction averages to zero and the decoder is the origin.
    fn measured_phase_codebook_distortion(codes: &Array2<f64>, m: usize) -> f64 {
        let cell = TAU / m as f64;
        let n = codes.nrows() as f64;
        let mut directions = Vec::with_capacity(codes.nrows());
        let mut projection = 0.0_f64;
        for row in codes.rows() {
            let theta = row[1].atan2(row[0]).rem_euclid(TAU);
            let k = ((theta / cell).floor() as usize).min(m - 1);
            let phi = (k as f64 + 0.5) * cell;
            projection += row[0] * phi.cos() + row[1] * phi.sin();
            directions.push(phi);
        }
        let rho = projection / n;
        codes
            .rows()
            .into_iter()
            .zip(&directions)
            .map(|(row, &phi)| {
                let dx = row[0] - rho * phi.cos();
                let dy = row[1] - rho * phi.sin();
                dx * dx + dy * dy
            })
            .sum::<f64>()
            / n
    }

    /// #2933 F23: the curved arm of the atomic ledger must be priced at the rate of
    /// an actual codebook that meets the flat arm's in-plane distortion, not at
    /// the small-cell expansion `½log₂(3a²/(π²δ²))`. The test measures quantized
    /// reconstructions of the community's own points for M = 1, 2, … and takes
    /// the least M whose measured error fits the flat code's budget
    /// `min(var_α, δ²) + min(var_β, δ²)`; `dl_new` must charge `log₂ M` per firing.
    /// A coarse codebook (M = 4) and a fine one (M = 65) are both checked, and
    /// each budget sits more than 1% from both neighbouring codebooks' errors, far
    /// beyond the finite-grid effect of the planted points.
    #[test]
    fn curved_arm_is_priced_at_a_measured_codebook_2933() {
        let p = 4;
        let radius = 1.0;
        for &(tolerance, n, expected_m) in &[(0.35_f64, 4096usize, 4usize), (0.01988, 1 << 16, 65)] {
            let (atoms, codes) = centred_ring_community(n, radius, p);
            let ctx = PromotionContext {
                n_tokens: n as f64,
                g_dict: 64,
                l0: 2.0,
                tolerance,
            };
            let proposal = propose_curve_promotion(
                LinearCommunity {
                    block_id: 0,
                    atoms: atoms.view(),
                    codes: codes.view(),
                },
                &ctx,
            )
            .expect("proposal producer runs")
            .expect("a 2-plane ring yields a proposal");
            let delta2 = tolerance * tolerance;
            let per_axis_variance = radius * radius / 2.0;
            let budget = 2.0 * per_axis_variance.min(delta2);
            let measured_m = (1..=4 * expected_m)
                .find(|&m| measured_phase_codebook_distortion(&codes, m) <= budget)
                .expect("a fine enough codebook meets the budget");
            assert_eq!(
                measured_m, expected_m,
                "planted premise: the least measured codebook at δ={tolerance}"
            );
            let below = measured_phase_codebook_distortion(&codes, measured_m - 1);
            let at = measured_phase_codebook_distortion(&codes, measured_m);
            assert!(
                below > 1.01 * budget && at < 0.99 * budget,
                "the budget must separate neighbouring codebooks: D(M-1)={below}, \
                 D(M)={at}, budget={budget}"
            );
            let unit_sel = (ctx.g_dict as f64 / ctx.l0).log2();
            let l_param = 0.5 * (n as f64).log2();
            let expected_dl_new =
                n as f64 * ((measured_m as f64).log2() + unit_sel) + 3.0 * p as f64 * l_param;
            assert!(
                (proposal.dl_new - expected_dl_new).abs() <= 1.0e-9 * expected_dl_new,
                "δ={tolerance}: dl_new {} must charge the measured {measured_m}-cell codebook \
                 ({expected_dl_new})",
                proposal.dl_new
            );
            let code = proposal
                .curved_phase_code
                .expect("the planted ring has a phase code at this budget");
            assert!(
                (code.distortion - at).abs() <= 1.0e-4 * budget,
                "the code's exact distortion {} must match the measured {at}",
                code.distortion
            );
        }
    }

    /// #2933 F22: the spectra-only prescreen is not a certificate, so it may not
    /// veto a promotion. A clean ring at G = L0 has no support dividend, which
    /// sends the prescreen negative; the atomic ledger still prefers one phase to
    /// two amplitudes by far more than the one extra harmonic decoder column.
    #[test]
    fn a_negative_prescreen_does_not_veto_the_atomic_ledger_2933() {
        let n = 512;
        let p = 4;
        let (atoms, codes) = centred_ring_community(n, 1.0, p);
        let ctx = PromotionContext {
            n_tokens: n as f64,
            g_dict: 2,
            l0: 2.0,
            tolerance: 0.05,
        };
        let proposal = propose_curve_promotion(
            LinearCommunity {
                block_id: 0,
                atoms: atoms.view(),
                codes: codes.view(),
            },
            &ctx,
        )
        .expect("proposal producer runs")
        .expect("a 2-plane ring yields a proposal");
        let surcharge = p as f64 * 0.5 * (n as f64).log2();
        let saving = proposal.dl_old - proposal.dl_new;
        assert!(proposal.verdict.recognized, "planted premise: a clean ring");
        assert!(
            proposal.crossover_prescreen.bits().is_some_and(|bits| bits < 0.0),
            "planted premise: with G = L0 the prescreen is negative, got {:?}",
            proposal.crossover_prescreen
        );
        assert!(
            saving > 10.0 * surcharge,
            "planted premise: the ledger's saving {saving} dwarfs the surcharge {surcharge}"
        );
        assert!(
            proposal.accept,
            "a ring the atomic ledger and geometry both prefer must not be vetoed by the \
             spectra-only prescreen"
        );
    }

    /// #2933 F23: the small-cell circle screen prices neither the support nor the
    /// decoder, so it may not veto a ring the atomic ledger buys. A noiseless full
    /// ring of radius 1.5 at tolerance δ = 1: read at `sigma = δ`, the screen debiases
    /// the radius to `√(1.5² − 2) = 0.5 < π/√3` and refuses. The ledger codes two
    /// amplitudes of variance 1.125 at `log₂ 1.125` bits against the least phase
    /// codebook meeting the in-plane budget `2·min(1.125, 1) = 2` (`M = 1` leaves
    /// `2.25`, `M = 2` leaves `2.25·(1 − 4/π²) = 1.338`, so one bit), plus
    /// `log₂(G/L0) = 4` bits per atom slot per firing. Over 1024 firings the one-slot
    /// support pays for the extra decoder row `4·½log₂4096 = 24` bits many times.
    #[test]
    fn a_small_cell_screen_does_not_veto_a_ring_the_ledger_buys_2933() {
        let (n, p, radius, tolerance) = (1024usize, 4usize, 1.5_f64, 1.0_f64);
        let (atoms, codes) = ring_community(n, radius, p);
        let ctx = PromotionContext {
            n_tokens: 4096.0,
            g_dict: 64,
            l0: 4.0,
            tolerance,
        };
        let proposal = propose_curve_promotion(
            LinearCommunity {
                block_id: 0,
                atoms: atoms.view(),
                codes: codes.view(),
            },
            &ctx,
        )
        .expect("proposal producer runs")
        .expect("a 2-plane ring yields a proposal");
        let screen = crate::manifold::curl_verdict(
            codes.column(0),
            codes.column(1),
            tolerance,
            n as f64,
            0.0,
        )
        .expect("the planted plane has in-plane energy");
        let l_param = 0.5 * ctx.n_tokens.log2();
        let unit_sel = (ctx.g_dict as f64 / ctx.l0).log2();
        let expected_old =
            n as f64 * (1.125_f64.log2() + 2.0 * unit_sel) + 2.0 * p as f64 * l_param;
        let expected_new = n as f64 * (1.0 + unit_sel) + 3.0 * p as f64 * l_param;
        println!(
            "PROBE_F23_SCREEN_VETO screen_recommend={} screen_radius={} z={} r1={} r2={} \
             dl_old={} dl_new={} accept={}",
            screen.recommend_curl,
            screen.radius,
            screen.z_below_gaussian,
            screen.resultant1,
            screen.resultant2,
            proposal.dl_old,
            proposal.dl_new,
            proposal.accept
        );
        assert!(
            screen.z_below_gaussian > 2.0 && screen.resultant1 < 0.05 && screen.resultant2 < 0.05,
            "planted premise: a covered ring with κ far below the Gaussian fill"
        );
        assert!(
            !screen.recommend_curl && (screen.radius - 0.5).abs() < 1.0e-9,
            "planted premise: the small-cell screen refuses at the debiased radius 0.5, got {}",
            screen.radius
        );
        assert!(
            (proposal.dl_old - expected_old).abs() <= 1.0e-9 * expected_old
                && (proposal.dl_new - expected_new).abs() <= 1.0e-9 * expected_new,
            "the ledger must price the flat arm at {expected_old} and the 2-cell phase code at \
             {expected_new}, got dl_old={} dl_new={}",
            proposal.dl_old,
            proposal.dl_new
        );
        assert!(
            proposal.accept,
            "a recognized ring the atomic ledger buys by {} bits must not be vetoed by the \
             small-cell screen",
            proposal.dl_old - proposal.dl_new
        );
    }

    /// The negative control for the test above: the same ring at `G = L0` earns no
    /// support dividend, so one phase bit per firing against `log₂ 1.125` for the
    /// amplitudes, plus the extra decoder row, loses and the ring is refused.
    /// Recognition alone never accepts.
    #[test]
    fn a_recognized_ring_the_ledger_refuses_is_not_accepted_2933() {
        let (n, p, radius, tolerance) = (1024usize, 4usize, 1.5_f64, 1.0_f64);
        let (atoms, codes) = ring_community(n, radius, p);
        let ctx = PromotionContext {
            n_tokens: 4096.0,
            g_dict: 4,
            l0: 4.0,
            tolerance,
        };
        let proposal = propose_curve_promotion(
            LinearCommunity {
                block_id: 0,
                atoms: atoms.view(),
                codes: codes.view(),
            },
            &ctx,
        )
        .expect("proposal producer runs")
        .expect("a 2-plane ring yields a proposal");
        let l_param = 0.5 * ctx.n_tokens.log2();
        let expected_old = n as f64 * 1.125_f64.log2() + 2.0 * p as f64 * l_param;
        let expected_new = n as f64 + 3.0 * p as f64 * l_param;
        assert!(
            (proposal.dl_old - expected_old).abs() <= 1.0e-9 * expected_old
                && (proposal.dl_new - expected_new).abs() <= 1.0e-9 * expected_new,
            "the ledger must price the flat arm at {expected_old} and the 2-cell phase code at \
             {expected_new}, got dl_old={} dl_new={}",
            proposal.dl_old,
            proposal.dl_new
        );
        assert!(
            !proposal.accept,
            "a ring the atomic ledger refuses by {} bits must not be accepted",
            proposal.dl_new - proposal.dl_old
        );
    }
}
