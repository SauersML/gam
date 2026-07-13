//! Atomic linear-community → curved REPLACEMENT proposals (audit §5 / §34).
//!
//! # The move class the residual-birth path structurally cannot make
//!
//! Every other curved producer in this crate discovers curvature from a
//! *residual*: the structured-residual birth path
//! ([`crate::structure_harvest`]) mines the reconstruction residual `R = x − x̂`
//! for factor directions, and the compose/co-fit lane
//! ([`crate::sparse_dict::cofit`]) fits charts to the linear tier's
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

/// Adjudicate the atomic replacement of a linear community by a single curved
/// chart. Returns `Ok(None)` when the community's own contribution has fewer than
/// two effective ambient dimensions (no 2-plane to host a ring); `Ok(Some(_))`
/// otherwise, with `accept` decided purely by the DL ledger and geometry screens.
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
    // Coord covariance (r × r) and its symmetric eigendecomposition (Jacobi).
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
    let (eigvals, eigvecs) = jacobi_symmetric_eig(&cov);
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

    // ---- Ring geometry verdict on the code cloud (κ, resultants, RD screen).
    //      n_eff = f (each active firing contributes one occupancy count);
    //      delta_charge = 0 keeps curl's `recommend` a pure geometry gate — the
    //      DL charge accounting lives in the atomic ledger below (no double count).
    let verdict = curl_verdict(alpha.view(), beta.view(), ctx.tolerance, f as f64, 0.0)?;

    // Curved topology matched to the ambient span (circle ŝ≈2 ⇒ (d,m)=(1,3)).
    let (d, m) = curved_topology_for_span(span);
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

    // ---- Atomic DL ledger (bits). δ is the distortion SCALE; the amplitude code
    //      bits are charged at variance δ².
    let delta = ctx.tolerance;
    let delta2 = delta * delta;
    let var_alpha = alpha.iter().map(|&a| a * a).sum::<f64>() / f as f64;
    let var_beta = beta.iter().map(|&b| b * b).sum::<f64>() / f as f64;
    let c_flat = scalar_rate_bits(var_alpha, delta2) + scalar_rate_bits(var_beta, delta2);
    let circle_gain = circle_coding_gain_bits(verdict.radius, delta);
    let c_curved = (c_flat - circle_gain).max(0.0);

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

    // ---- #2233 crossover pre-screen (spectra-only predicted saving). Consumed as
    //      the necessary pre-screen; the atomic ledger is the sufficient decision.
    let firing_rate = f as f64 / ctx.n_tokens;
    let crossover_prescreen_bits = predicted_birth_dl_bits(&BirthMdlPrescreen {
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

    let accept =
        crossover_prescreen_bits > 0.0 && verdict.recommend_curl && dl_new < dl_old;

    Ok(Some(CurvePromotionProposal {
        block: community.block_id,
        n_linear_atoms: s,
        curved_candidate,
        verdict,
        span,
        firing_rate,
        dl_old,
        dl_new,
        crossover_prescreen_bits,
        accept,
    }))
}

/// Gram–Schmidt an `s × P` set of ambient atom directions into an orthonormal
/// ambient frame, dropping directions that are (numerically) already in the span.
/// The retained count `r ≤ s` is the effective rank of the block span.
fn gram_schmidt(atoms: ArrayView2<'_, f64>) -> Vec<Array1<f64>> {
    let (s, _p) = atoms.dim();
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(s);
    for j in 0..s {
        let mut v = atoms.row(j).to_owned();
        for q in &basis {
            let proj = v.dot(q);
            v.scaled_add(-proj, q);
        }
        let norm = v.dot(&v).sqrt();
        // Relative drop threshold: a residual whose norm collapses under the
        // atom's own scale carries no new direction. Uses the row norm as the
        // reference so the test is scale-free, not a hand-set absolute floor.
        let row_norm = atoms.row(j).dot(&atoms.row(j)).sqrt();
        if norm > row_norm * f64::EPSILON.sqrt() {
            v.mapv_inplace(|x| x / norm);
            basis.push(v);
        }
    }
    basis
}

/// The participation ratio `(Σλ)² / Σλ²` of a non-negative energy spectrum — the
/// effective number of significant ambient directions the cloud occupies
/// (circle ≈ 2). Zero on a degenerate spectrum.
fn participation_ratio(spectrum: &[f64]) -> f64 {
    let sum: f64 = spectrum.iter().map(|&e| e.max(0.0)).sum();
    let sum_sq: f64 = spectrum.iter().map(|&e| e.max(0.0) * e.max(0.0)).sum();
    if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 }
}

/// The curved topology `(intrinsic_dim d, basis_size m)` matched to an ambient
/// span. A 2-plane span promotes to a circle (`d=1`, `m=2·d+1=3` harmonic rows);
/// higher spans to the sphere/torus charts. Mirrors the structured-birth path's
/// span→topology map so the pre-screen prices the atom that would actually race.
fn curved_topology_for_span(span: f64) -> (usize, usize) {
    match span.round().max(1.0) as usize {
        0 | 1 | 2 => (1, 3), // circle (PeriodicHarmonicEvaluator, 2·d+1 rows)
        3 => (2, 7),         // sphere chart
        _ => (2, 25),        // torus (H=2)
    }
}

/// Cyclic Jacobi eigendecomposition of a small symmetric `r × r` matrix. Returns
/// `(eigenvalues, eigenvectors)` with eigenvectors as COLUMNS of the returned
/// matrix. Deterministic; converges quadratically for the small covariances here.
fn jacobi_symmetric_eig(sym: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let r = sym.nrows();
    let mut a = sym.clone();
    let mut v = Array2::<f64>::eye(r);
    if r == 1 {
        return (vec![a[[0, 0]]], v);
    }
    // Sweep until the off-diagonal is negligible relative to the diagonal scale.
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..r {
            for q in (p + 1)..r {
                off += a[[p, q]] * a[[p, q]];
            }
        }
        let diag_scale: f64 = (0..r).map(|i| a[[i, i]] * a[[i, i]]).sum();
        if off <= diag_scale * f64::EPSILON * f64::EPSILON || off == 0.0 {
            break;
        }
        for p in 0..r {
            for q in (p + 1)..r {
                let apq = a[[p, q]];
                if apq == 0.0 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                // Jacobi rotation angle that zeros the (p,q) entry.
                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let t = if tau == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * c;
                // Apply the rotation to rows/cols p, q of A.
                for k in 0..r {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - sn * akq;
                    a[[k, q]] = sn * akp + c * akq;
                }
                for k in 0..r {
                    let apk = a[[p, k]];
                    let aqk = a[[q, k]];
                    a[[p, k]] = c * apk - sn * aqk;
                    a[[q, k]] = sn * apk + c * aqk;
                }
                // Accumulate the eigenvector rotation.
                for k in 0..r {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - sn * vkq;
                    v[[k, q]] = sn * vkp + c * vkq;
                }
            }
        }
    }
    let eigvals: Vec<f64> = (0..r).map(|i| a[[i, i]]).collect();
    (eigvals, v)
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
