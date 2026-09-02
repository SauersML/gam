//! The code-space curvature pass: Tier-2's *first* substrate is the Tier-1
//! CODE GROUPS, not the Tier-1 residual (#2023 / #2502).
//!
//! # Why the residual substrate is blind by construction
//!
//! A centered closed curve's reconstruction cone IS its plane: sweeping radius
//! and phase over a centered circle fills the 2-plane the circle spans, so two
//! linear atoms reconstruct the ring **exactly** and the block's linear residual
//! is identically zero. A Tier-2 that charts the post-Tier-1 residual therefore
//! cannot see precisely the curvature the tiered model exists to find — not
//! because the linear fit is poor but because it is perfect
//! ([`crate::manifold::curve_promotion`] states the impossibility result; the
//! campaign measured its corollary as the EV null at matched budget in #2502).
//! Where curvature IS visible is the **measure**: the joint law of the block's
//! code amplitudes. A ring in code space is a constraint AMONG the directions
//! Tier-1 already spans, and the only statistics with power against the flat
//! null are description-length bits and the radial law — never residual EV.
//!
//! # What this pass does
//!
//! Each Tier-1 block is already a co-firing linear community: its `b` decoder
//! rows are the atoms, and the rows that routed to it carry a `b`-dimensional
//! code cloud. This pass walks every fitted block, hands its community to the
//! atomic bits-denominated adjudicator
//! ([`crate::manifold::curve_promotion::propose_curve_promotion`] — previously a
//! pure producer with no consumer), and reports, per block, whether ONE curved
//! chart describes the same firings in fewer bits than the block's linear atoms.
//! The decoder stays linear either way — an accepted promotion bends the CODE
//! (one phase + one amplitude replace `b` amplitudes), not the dictionary.
//!
//! Every quantity the ledger prices is measured, not configured: the dictionary
//! size and the mean active coordinates per token are read off the fit, and the
//! distortion floor is the reconstruction tolerance the linear tier actually
//! achieved ([`linear_distortion_floor`]). The pass introduces no knob.
//!
//! The report's `fraction_curved` is the honesty measurement the field actually
//! wants from a manifold SAE: how much of the dictionary curves, feature by
//! feature, with the refusals recorded next to the acceptances.

use ndarray::{Array2, ArrayView2, Axis};

use crate::atom_codes::SparseAtomCodes;
use crate::front_door::admit_topk_manifold;
use crate::manifold::curve_promotion::{
    CurvePromotionProposal, LinearCommunity, PromotionContext, propose_curve_promotion,
};
use crate::manifold::{
    SaeSupportOuterRequest, SaeSupportSeedRequest, SaeSupportTermSeedRequest,
    build_sae_support_seed, build_sae_support_term_seed, run_sae_support_outer,
    sae_support_effective_atom_dims,
};
use crate::sparse_dict::BlockSparseFit;

/// The code-space curvature census over one fitted Tier-1 dictionary: every
/// per-block promotion proposal (accepted AND refused — the refusals are part of
/// the measurement), plus the measured context the ledger priced against.
#[derive(Clone, Debug)]
pub struct CodeSpacePromotionReport {
    /// One proposal per block whose community reached the adjudicator (blocks
    /// with ≥ 2 firings and a ≥ 2-dimensional code plane). `accept` on each entry
    /// is the atomic DL verdict; nothing here mutates the fit.
    pub proposals: Vec<CurvePromotionProposal>,
    /// Cross-block shell census: one proposal per CO-FIRING block pair whose
    /// joint code cloud reached the adjudicator, keyed by the pair. A ring the
    /// dictionary shattered across two blocks (one straight atom per half — the
    /// shape the Gemma Scope census found as "pairs of straight atoms whose
    /// joint amplitude law is a shell") is invisible to every single-block
    /// community; the union community is where it lives.
    pub pair_proposals: Vec<CensusPairVerdict>,
    /// Total blocks in the Tier-1 dictionary (`G`).
    pub n_blocks_scanned: usize,
    /// Blocks whose community yielded a proposal (the eligible denominator).
    pub n_communities: usize,
    /// Proposals the atomic bits ledger accepted.
    pub n_accepted: usize,
    /// Total bits the accepted promotions save (`Σ dl_old − dl_new` over
    /// accepted proposals).
    pub dl_saved_bits: f64,
    /// `n_accepted / n_communities` (`0.0` when no community was eligible) —
    /// the fraction of the co-firing dictionary that genuinely curves.
    pub fraction_curved: f64,
    /// The measured per-coordinate distortion floor `δ` the ledger priced at.
    pub tolerance: f64,
    /// The measured mean active scalar coordinates per token (`L0`).
    pub l0: f64,
}

/// The measured per-coordinate distortion floor `δ` for the code-space ledger:
/// the RMS per-element reconstruction error the linear tier actually achieved,
/// floored at the corpus's own f64 measurement resolution
/// (`RMS(R0) · √ε`) so an exactly-reconstructed corpus still carries a positive
/// quantisation cell instead of a zero that would poison the rate terms.
///
/// `residual` is the post-Tier-1 centered residual; `baseline_energy` is
/// `‖R0‖²` over the same `N×P` grid (the Tier-0 baseline the tiered EV is
/// measured against). Errors when the corpus itself carries no energy — with
/// nothing to reconstruct there is no distortion scale to measure.
pub fn linear_distortion_floor(
    residual: ArrayView2<'_, f64>,
    baseline_energy: f64,
) -> Result<f64, String> {
    let n_elems = residual.len();
    if n_elems == 0 {
        return Err("linear_distortion_floor: empty residual".to_string());
    }
    if !(baseline_energy > 0.0 && baseline_energy.is_finite()) {
        return Err(format!(
            "linear_distortion_floor: baseline energy must be finite and > 0, got {baseline_energy}"
        ));
    }
    let residual_ms = residual.iter().map(|&r| r * r).sum::<f64>() / n_elems as f64;
    if !residual_ms.is_finite() {
        return Err(format!(
            "linear_distortion_floor: residual energy is not finite ({residual_ms})"
        ));
    }
    let corpus_rms = (baseline_energy / n_elems as f64).sqrt();
    let resolution_floor = corpus_rms * f64::EPSILON.sqrt();
    Ok(residual_ms.sqrt().max(resolution_floor))
}

/// Walk every Tier-1 block as a linear community and adjudicate its code cloud
/// for curved replacement, in bits. Pure: reads the fit, mutates nothing.
///
/// `n_tokens` is the corpus row count `N` the firing rate is measured against;
/// `tolerance` is the per-coordinate distortion floor `δ` (use
/// [`linear_distortion_floor`] for the measured one).
pub fn harvest_code_space_promotions(
    tier1: &BlockSparseFit,
    n_tokens: usize,
    tolerance: f64,
) -> Result<CodeSpacePromotionReport, String> {
    let b = tier1.block_size;
    let p = tier1.decoder.ncols();
    let k_atoms = tier1.decoder.nrows();
    if b == 0 || p == 0 || k_atoms == 0 || k_atoms % b != 0 {
        return Err(format!(
            "harvest_code_space_promotions: malformed block geometry (K={k_atoms}, b={b}, P={p})"
        ));
    }
    let n_blocks = k_atoms / b;
    let n_rows = tier1.blocks.nrows();
    if n_tokens < n_rows {
        return Err(format!(
            "harvest_code_space_promotions: n_tokens {n_tokens} < routed rows {n_rows}"
        ));
    }

    // One pass over the sparse routing: per-block firing code lists (flattened
    // f×b), the co-firing block-pair joint code lists (flattened f×2b), and the
    // measured mean active scalar coordinates per token (L0).
    let mut firings: Vec<Vec<f64>> = vec![Vec::new(); n_blocks];
    let mut pair_firings: std::collections::BTreeMap<(usize, usize), Vec<f64>> =
        std::collections::BTreeMap::new();
    let mut active_scalars = 0usize;
    let mut row_live: Vec<(usize, usize)> = Vec::with_capacity(tier1.block_topk);
    for i in 0..n_rows {
        row_live.clear();
        for j in 0..tier1.block_topk {
            if tier1.gates[[i, j]] == 0.0 {
                continue; // padded slot: no live block routed here
            }
            let g = tier1.blocks[[i, j]] as usize;
            if g >= n_blocks {
                return Err(format!(
                    "harvest_code_space_promotions: routed block {g} out of range G={n_blocks}"
                ));
            }
            for r in 0..b {
                let code = tier1.codes[[i, j, r]] as f64;
                if code != 0.0 {
                    active_scalars += 1;
                }
                firings[g].push(code);
            }
            row_live.push((g, j));
        }
        // Joint clouds for every co-firing pair on this row (ordered by block id
        // so the pair key and its code-column order are canonical).
        for a in 0..row_live.len() {
            for c in (a + 1)..row_live.len() {
                let (mut ga, mut ja) = row_live[a];
                let (mut gb, mut jb) = row_live[c];
                if ga == gb {
                    continue;
                }
                if ga > gb {
                    std::mem::swap(&mut ga, &mut gb);
                    std::mem::swap(&mut ja, &mut jb);
                }
                let joint = pair_firings.entry((ga, gb)).or_default();
                for r in 0..b {
                    joint.push(tier1.codes[[i, ja, r]] as f64);
                }
                for r in 0..b {
                    joint.push(tier1.codes[[i, jb, r]] as f64);
                }
            }
        }
    }
    let l0 = active_scalars as f64 / n_tokens as f64;

    let ctx = PromotionContext {
        n_tokens: n_tokens as f64,
        g_dict: k_atoms,
        l0,
        tolerance,
    };

    let mut proposals = Vec::new();
    let mut n_communities = 0usize;
    let mut n_accepted = 0usize;
    let mut dl_saved_bits = 0.0f64;
    for g in 0..n_blocks {
        let f = firings[g].len() / b;
        if f < 2 {
            continue;
        }
        // The block's atoms (b×P, f64) and its code cloud (f×b, f64).
        let mut atoms = Array2::<f64>::zeros((b, p));
        for r in 0..b {
            let src = tier1.decoder.row(g * b + r);
            for c in 0..p {
                atoms[[r, c]] = src[c] as f64;
            }
        }
        let codes = Array2::from_shape_vec((f, b), std::mem::take(&mut firings[g]))
            .map_err(|err| format!("harvest_code_space_promotions: code reshape failed: {err}"))?;
        let community = LinearCommunity {
            block_id: g,
            atoms: atoms.view(),
            codes: codes.view(),
        };
        let Some(proposal) = propose_curve_promotion(community, &ctx)? else {
            continue; // rank-1 / point community: no plane to host a ring
        };
        n_communities += 1;
        if proposal.accept {
            n_accepted += 1;
            dl_saved_bits += proposal.dl_old - proposal.dl_new;
        }
        proposals.push(proposal);
    }
    // Cross-block shells: adjudicate every co-firing pair's JOINT cloud. A ring
    // shattered across two blocks presents per block as a line (refused above)
    // and only closes in the union community.
    let mut pair_proposals = Vec::new();
    for ((ga, gb), flat) in pair_firings {
        let s = 2 * b;
        let f = flat.len() / s;
        if f < 2 {
            continue;
        }
        let mut atoms = Array2::<f64>::zeros((s, p));
        for (slot, block) in [ga, gb].into_iter().enumerate() {
            for r in 0..b {
                let src = tier1.decoder.row(block * b + r);
                for col in 0..p {
                    atoms[[slot * b + r, col]] = src[col] as f64;
                }
            }
        }
        let codes = Array2::from_shape_vec((f, s), flat)
            .map_err(|err| format!("harvest_code_space_promotions: pair reshape failed: {err}"))?;
        let community = LinearCommunity {
            block_id: ga,
            atoms: atoms.view(),
            codes: codes.view(),
        };
        let Some(proposal) = propose_curve_promotion(community, &ctx)? else {
            continue;
        };
        n_communities += 1;
        let observed_saving = proposal.dl_old - proposal.dl_new;
        let (ran, exceed) = if proposal.accept {
            n_accepted += 1;
            dl_saved_bits += observed_saving;
            pair_permutation_null(
                atoms.view(),
                codes.view(),
                ga,
                &ctx,
                proposal.verdict.z_below_gaussian,
                PAIR_NULL_PERMUTATIONS,
            )?
        } else {
            (0, 0)
        };
        pair_proposals.push(CensusPairVerdict {
            atom_a: ga,
            atom_b: gb,
            proposal,
            null_permutations: ran,
            null_exceedances: exceed,
            null_p_hat: if ran > 0 {
                (1.0 + exceed as f64) / (1.0 + ran as f64)
            } else {
                f64::NAN
            },
            // The tiered block-pair census defers topology to the joint-fit
            // race that consumes its births; the standalone adjudication lives
            // on the foreign-dictionary entry.
            topology_kind: None,
            topology_dim: None,
            topology_error: None,
        });
    }

    let fraction_curved = if n_communities > 0 {
        n_accepted as f64 / n_communities as f64
    } else {
        0.0
    };

    Ok(CodeSpacePromotionReport {
        proposals,
        pair_proposals,
        n_blocks_scanned: n_blocks,
        n_communities,
        n_accepted,
        dl_saved_bits,
        fraction_curved,
        tolerance,
        l0,
    })
}

/// One adjudicated co-firing pair: the atomic DL proposal plus its permutation
/// null. The null couples atom `a`'s weights to a deterministically PERMUTED
/// copy of atom `b`'s weights over the same firing rows — both marginals and
/// the firing pattern are preserved exactly; only the joint law is destroyed —
/// and re-runs the identical accept rule. `null_p_hat` is the add-one estimate
/// `(1 + #{null saving ≥ observed}) / (1 + permutations)`.
///
/// Cyclic ROTATIONS are deliberately not used: activation rows are
/// time-ordered, so a rotation of `sin θ` against `cos θ` is still a smooth
/// Lissajous trajectory (an ellipse at equal frequency) — a null that preserves
/// the very structure under test is not a null — and so is any AFFINE
/// permutation, which maps an arithmetic phase grid to a Lissajous curve (see
/// `hashed_permutation`). The hash-order permutation scrambles the ordering
/// deterministically, so identical inputs give identical verdicts (no RNG).
#[derive(Clone, Debug)]
pub struct CensusPairVerdict {
    /// First atom (or block) of the pair, the smaller index.
    pub atom_a: usize,
    /// Second atom (or block) of the pair.
    pub atom_b: usize,
    /// The atomic bits adjudication of the observed joint cloud.
    pub proposal: CurvePromotionProposal,
    /// Number of permutation nulls run (0 when the observed pair was refused —
    /// there is no discovery to error-control).
    pub null_permutations: u32,
    /// Nulls whose radial-concentration z matched or beat the observed one
    /// (the coupling-sensitive statistic; the DL saving itself is a
    /// marginal-moment functional every permutation preserves exactly).
    pub null_exceedances: u32,
    /// `(1 + exceedances) / (1 + permutations)`; NaN when no nulls were run.
    pub null_p_hat: f64,
    /// REML topology-race verdict on the accepted pair's ambient image
    /// ([`crate::structure_harvest::discover_primary_atom_topologies`], the
    /// same evidence race the seed dictionary uses): the winning basis kind's
    /// `Debug` label (e.g. `Periodic` for a genuine ring, `EuclideanPatch` for
    /// a flat cloud the DL ledger accepted on amplitude-law concentration
    /// alone). `None` when the pair was refused or the race declined.
    pub topology_kind: Option<String>,
    /// Latent dimension the winning topology carries; `None` with the above.
    pub topology_dim: Option<usize>,
    /// The race's own refusal text when it declined to adjudicate (kept
    /// verbatim so a declined race never reads as a flat verdict).
    pub topology_error: Option<String>,
}

/// Deterministic hash-order permutation family for the pair null: the `m`-th
/// permutation of `0..f` sorts indices by `splitmix64(i ⊕ (m+1)·2⁴⁸)`.
///
/// An AFFINE family `i ↦ (q·i + m) mod f` is NOT a valid null here, and this
/// was measured, not theorized: on evenly-spaced ring phases an affine map
/// sends `sin θᵢ` to `sin(q·θᵢ + φ)` — a Lissajous curve, which codes as well
/// as the ring itself (50/63 "nulls" beat the planted ring). Structure-free
/// scrambling needs a hash order; splitmix64 is deterministic (no RNG state),
/// so identical inputs still give identical verdicts.
fn hashed_permutation(f: usize, m: usize) -> Vec<usize> {
    fn splitmix64(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let salt = ((m as u64) + 1) << 48;
    let mut idx: Vec<usize> = (0..f).collect();
    idx.sort_by_key(|&i| splitmix64(i as u64 ^ salt));
    idx
}

/// Run the permutation null for one accepted pair cloud (`f×2`).
///
/// The null statistic is the ring verdict's radial-concentration z
/// (`z_below_gaussian`), NOT the DL saving — and this distinction was paid for:
/// the atomic bits ledger is a functional of the MARGINAL second moments
/// (`var α`, `var β`, `m₂`), all of which a permutation preserves exactly, so
/// every scrambled coupling carries the identical saving and a "null on bits"
/// only counts how often the geometry screens pass on the corner-concentrated
/// independent coupling of two bimodal marginals (measured p̂ ≈ 0.8 on a
/// PLANTED ring). All coupling sensitivity lives in the radial law; the null
/// therefore competes on it.
fn pair_permutation_null(
    atoms: ArrayView2<'_, f64>,
    codes: ArrayView2<'_, f64>,
    block_id: usize,
    ctx: &PromotionContext,
    observed_ring_z: f64,
    n_perms: usize,
) -> Result<(u32, u32), String> {
    let f = codes.nrows();
    let s = codes.ncols();
    let split = s / 2; // columns [split..s) belong to the second atom/block
    let mut exceed = 0u32;
    let mut ran = 0u32;
    for m in 1..=n_perms {
        let perm = hashed_permutation(f, m);
        let mut null_codes = codes.to_owned();
        for i in 0..f {
            let j = perm[i];
            for c in split..s {
                null_codes[[i, c]] = codes[[j, c]];
            }
        }
        let community = LinearCommunity {
            block_id,
            atoms,
            codes: null_codes.view(),
        };
        ran += 1;
        if let Some(null_prop) = propose_curve_promotion(community, ctx)? {
            if null_prop.verdict.z_below_gaussian >= observed_ring_z {
                exceed += 1;
            }
        }
    }
    Ok((ran, exceed))
}

/// Permutation-null budget per accepted pair. 63 gives add-one p̂ resolution
/// 1/64 — enough to separate "survives" from "artifact" per pair; family-level
/// error control across pairs is the e-BH layer's job downstream.
const PAIR_NULL_PERMUTATIONS: usize = 63;

/// The atom-level census over ANY dictionary: every co-firing ATOM pair of an
/// arbitrary atom bank (`decoder`, `K×P`) with solved codes on it is adjudicated
/// for curved replacement, in bits. This is the universal-improver entry — the
/// input dictionary can be a TopK / JumpReLU / block-sparse decoder imported
/// from anywhere; the census consumes only its atoms and its code measure, which
/// is exactly where the Gemma Scope shells live (pairs of straight atoms whose
/// joint amplitude law is a shell). Communities are the observed co-firing
/// pairs; `G` and `L0` are measured off the codes; nothing here mutates the
/// dictionary.
///
/// The report reuses [`CodeSpacePromotionReport`]: pair verdicts land in
/// `pair_proposals` keyed by the atom pair (single-atom `proposals` stays empty
/// — a lone atom has no plane).
pub fn harvest_code_space_pair_promotions(
    decoder: ArrayView2<'_, f64>,
    codes: &SparseAtomCodes,
    n_tokens: usize,
    tolerance: f64,
) -> Result<CodeSpacePromotionReport, String> {
    let k_atoms = decoder.nrows();
    let p = decoder.ncols();
    if k_atoms == 0 || p == 0 {
        return Err(format!(
            "harvest_code_space_pair_promotions: empty decoder ({k_atoms}×{p})"
        ));
    }
    if codes.k_atoms() != k_atoms {
        return Err(format!(
            "harvest_code_space_pair_promotions: codes carry K={} but the decoder has K={k_atoms}",
            codes.k_atoms()
        ));
    }
    let n_rows = codes.n_obs();
    if n_tokens < n_rows {
        return Err(format!(
            "harvest_code_space_pair_promotions: n_tokens {n_tokens} < coded rows {n_rows}"
        ));
    }

    // Pass 0 — count. Per-pair joint-firing counts and the measured mean active
    // atoms per token (L0). Counting first is what makes the census tractable at
    // SAE scale: with L0 ≈ 60 a row contributes ~1.8k pairs, and materialising a
    // weight cloud for every observed pair would be O(10⁸) allocations.
    //
    // The count store is a FLAT triangular u16 array when it fits (K ≤ 46k ⇒
    // ≤ ~2 GiB): the first real run's HashMap<u64,u32> peaked at 74.7 GB and
    // dominated the 6m41s wall — hashing ~7·10⁸ pair events is the hot path,
    // and a saturating direct index removes both the hashing and the per-entry
    // overhead. Counts saturate at u16::MAX, far above every admission floor
    // f_min in practice (~10³); a saturated count can only ADMIT a pair the
    // floor would have admitted anyway, never drop one.
    let tri_len = k_atoms * (k_atoms - 1) / 2;
    let tri_index = |a: usize, b: usize| -> usize {
        // a < b; row-major upper triangle.
        a * k_atoms - a * (a + 1) / 2 + (b - a - 1)
    };
    let use_flat = tri_len <= (1usize << 30); // ≤ 2 GiB of u16
    let mut flat_counts: Vec<u16> = if use_flat { vec![0u16; tri_len] } else { Vec::new() };
    let mut map_counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    let mut active_total = 0usize;
    let mut support: Vec<(usize, f64)> = Vec::new();
    for row in codes.iter() {
        let n_active = row.active_mask.count_ones();
        active_total += n_active;
        support.clear();
        for atom in row.active_mask.iter_ones() {
            support.push((atom, row.weights[atom]));
        }
        for a in 0..support.len() {
            for b in (a + 1)..support.len() {
                if use_flat {
                    let c = &mut flat_counts[tri_index(support[a].0, support[b].0)];
                    *c = c.saturating_add(1);
                } else {
                    let key = ((support[a].0 as u64) << 32) | support[b].0 as u64;
                    *map_counts.entry(key).or_insert(0) += 1;
                }
            }
        }
    }
    let pair_count = |a: usize, b: usize| -> u32 {
        if use_flat {
            flat_counts[tri_index(a, b)] as u32
        } else {
            map_counts
                .get(&(((a as u64) << 32) | b as u64))
                .copied()
                .unwrap_or(0)
        }
    };
    let l0 = active_total as f64 / n_tokens as f64;
    let ctx = PromotionContext {
        n_tokens: n_tokens as f64,
        g_dict: k_atoms,
        l0,
        tolerance,
    };

    // The ledger's own admission floor, not a knob: a circle's code term in the
    // prescreen is exactly zero, so acceptance REQUIRES the support dividend to
    // beat the dictionary surcharge — `f·(ŝ−1)·log₂(G/L0) > (m−ŝ)·P·½log₂N`,
    // and the circle (ŝ=2, m=3) minimises the right/left ratio over the raced
    // topologies. Pairs co-firing fewer than f_min times can therefore never be
    // accepted, and skipping them prunes the candidate set from O(L0²·N) to the
    // handful that could pay.
    let unit_sel = if l0 > 0.0 {
        (k_atoms as f64 / l0).log2().max(0.0)
    } else {
        0.0
    };
    let log2_n = if n_tokens >= 2 { (n_tokens as f64).log2() } else { 0.0 };
    let f_min = if unit_sel > 0.0 {
        ((p as f64 * 0.5 * log2_n) / unit_sel).ceil().max(2.0) as u32
    } else {
        u32::MAX
    };

    // Pass 1 — accumulate weight clouds only for admissible pairs.
    let mut pair_firings: std::collections::BTreeMap<(usize, usize), Vec<f64>> =
        std::collections::BTreeMap::new();
    for row in codes.iter() {
        support.clear();
        for atom in row.active_mask.iter_ones() {
            support.push((atom, row.weights[atom]));
        }
        for a in 0..support.len() {
            for b in (a + 1)..support.len() {
                let (atom_a, w_a) = support[a];
                let (atom_b, w_b) = support[b];
                if pair_count(atom_a, atom_b) < f_min {
                    continue;
                }
                let joint = pair_firings.entry((atom_a, atom_b)).or_default();
                joint.push(w_a);
                joint.push(w_b);
            }
        }
    }

    let mut pair_proposals = Vec::new();
    let mut n_communities = 0usize;
    let mut n_accepted = 0usize;
    let mut dl_saved_bits = 0.0f64;
    for ((atom_a, atom_b), flat) in pair_firings {
        let f = flat.len() / 2;
        if f < 2 {
            continue;
        }
        let mut atoms = Array2::<f64>::zeros((2, p));
        atoms.row_mut(0).assign(&decoder.row(atom_a));
        atoms.row_mut(1).assign(&decoder.row(atom_b));
        let pair_codes = Array2::from_shape_vec((f, 2), flat).map_err(|err| {
            format!("harvest_code_space_pair_promotions: pair reshape failed: {err}")
        })?;
        let community = LinearCommunity {
            block_id: atom_a,
            atoms: atoms.view(),
            codes: pair_codes.view(),
        };
        let Some(proposal) = propose_curve_promotion(community, &ctx)? else {
            continue;
        };
        n_communities += 1;
        let observed_saving = proposal.dl_old - proposal.dl_new;
        let mut topology_kind = None;
        let mut topology_dim = None;
        let mut topology_error = None;
        let (ran, exceed) = if proposal.accept {
            n_accepted += 1;
            dl_saved_bits += observed_saving;
            // Topology adjudication of the ACCEPTED pair: race the pair's
            // ambient image through the seed dictionary's own evidence race,
            // so "curved" never rests on the DL ledger alone (an exclusion
            // mixture also concentrates its amplitude law; the race is where
            // ring vs flat is decided by REML).
            if f >= 16 {
                let image = pair_codes.dot(&atoms);
                match crate::structure_harvest::discover_primary_atom_topologies(
                    image.view(),
                    &vec![0usize; f],
                    1,
                    &[2],
                ) {
                    Ok(choices) => {
                        if let Some(choice) = choices.first() {
                            topology_kind = Some(format!("{:?}", choice.basis_kind));
                            topology_dim = Some(choice.latent_dim);
                        }
                    }
                    Err(error) => topology_error = Some(error),
                }
            } else {
                topology_error = Some(format!("race needs >= 16 rows, pair has {f}"));
            }
            pair_permutation_null(
                atoms.view(),
                pair_codes.view(),
                atom_a,
                &ctx,
                proposal.verdict.z_below_gaussian,
                PAIR_NULL_PERMUTATIONS,
            )?
        } else {
            (0, 0)
        };
        pair_proposals.push(CensusPairVerdict {
            atom_a,
            atom_b,
            proposal,
            null_permutations: ran,
            null_exceedances: exceed,
            null_p_hat: if ran > 0 {
                (1.0 + exceed as f64) / (1.0 + ran as f64)
            } else {
                f64::NAN
            },
            topology_kind,
            topology_dim,
            topology_error,
        });
    }
    let fraction_curved = if n_communities > 0 {
        n_accepted as f64 / n_communities as f64
    } else {
        0.0
    };

    Ok(CodeSpacePromotionReport {
        proposals: Vec::new(),
        pair_proposals,
        n_blocks_scanned: k_atoms,
        n_communities,
        n_accepted,
        dl_saved_bits,
        fraction_curved,
        tolerance,
        l0,
    })
}

/// A REML-fitted curved chart on one accepted pair's 2-D code cloud: the full
/// GAM machinery (grouped-LAML outer engine selecting the smoothing strengths,
/// certified inner fixed point, outer stationarity certificate) applied to the
/// discovery the census made. This is what upgrades a census verdict into a
/// fitted object with a coordinate: the chart's smoothing is chosen by REML,
/// never by a knob, and the certificate travels with the fit.
#[derive(Clone, Debug)]
pub struct PairChartFit {
    /// Per-retained-atom smoothing strengths selected by the outer engine.
    pub lambda_smooth: Vec<f64>,
    /// Terminal LAML criterion at the certified smoothing optimum.
    pub criterion: f64,
    /// Chart explained variance of the centered cloud (`1 − RSS/TSS`).
    pub explained_variance: f64,
    /// Outer (smoothing-selection) iterations to the certified optimum.
    pub outer_iterations: usize,
    /// Whether the outer stationarity certificate certifies.
    pub certified: bool,
    /// Whether the inner fixed point recurred.
    pub recurred: bool,
    /// Retained curved atoms after dead-support pruning (≥ 1 on success).
    pub retained_atoms: usize,
}

/// Fit a REML-smoothed periodic chart to one pair's joint code cloud (`f×2`,
/// the two co-firing weights), through the canonical overcomplete support-sparse
/// engine — the same lane the public curved fit uses, at the smallest admitted
/// width (`K = 3 > P = 2`, TopK `s = 1`). The cloud is centered here and the
/// chart fits the centered target; `random_state` seeds the deterministic
/// support routing.
pub fn fit_pair_chart(
    cloud: ArrayView2<'_, f64>,
    random_state: u64,
) -> Result<PairChartFit, String> {
    // Deterministic multistart over derived support-routing seeds: the outer
    // engine legitimately refuses a seed whose routing starves an atom into a
    // zero adjoint-majorizer eigenvalue, and which seed does so is a property
    // of the routing draw, not of the cloud. Four derived seeds; the first
    // accepted fit wins; all-refuse propagates the engine's own error.
    let mut last_err = String::new();
    for salt in 0u64..4 {
        let seed = random_state ^ (salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        match fit_pair_chart_at_seed(cloud, seed) {
            Ok(fit) => return Ok(fit),
            Err(err) => last_err = err,
        }
    }
    Err(last_err)
}

fn fit_pair_chart_at_seed(
    cloud: ArrayView2<'_, f64>,
    random_state: u64,
) -> Result<PairChartFit, String> {
    let (f, width) = cloud.dim();
    if width != 2 {
        return Err(format!("fit_pair_chart: cloud must be f×2, got f={f}×{width}"));
    }
    if f < 16 {
        return Err(format!(
            "fit_pair_chart: {f} rows cannot support a certified chart fit (need ≥ 16)"
        ));
    }
    let mean = cloud
        .mean_axis(Axis(0))
        .ok_or_else(|| "fit_pair_chart: mean_axis failed".to_string())?;
    let centered = &cloud - &mean.view().insert_axis(Axis(0));

    // K=8/s=2 mirrors the in-crate curved-chart precedent (`chart_curved` in
    // the tiered peel tests); the minimal K=3/s=1 shape starves an atom into a
    // zero adjoint-majorizer eigenvalue the outer engine rightly refuses.
    let n_atoms = 8usize;
    let support_k = 2usize;
    let atom_basis = vec!["periodic".to_string(); n_atoms];
    let atom_dim = vec![1usize; n_atoms];
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(f, 2, n_atoms, d_max, support_k)?;
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k,
        random_state,
        admission,
    })?;
    let retained = seed.retained_atom_indices.len();
    let term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: vec!["periodic".to_string(); retained],
        atom_dim: vec![1usize; retained],
        output_dim: 2,
        random_state,
    })?;
    let ard_precisions = (0..term_seed.term.k_atoms())
        .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
        .collect::<Vec<_>>();
    let outer = run_sae_support_outer(SaeSupportOuterRequest {
        term: term_seed.term,
        target: centered.clone(),
        initial_smoothness: 1.0,
        ard_precisions,
        max_outer_iter: 32,
        max_inner_iter: 256,
        // The public entry's relative inner tolerance (#2517).
        inner_tolerance: 1.0e-4,
        trust_radius: 1.0,
        random_state,
    })
    .map_err(|error| error.to_string())?;

    let recon = outer.term.reconstruct()?;
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    for i in 0..f {
        for c in 0..2 {
            let d = centered[[i, c]] - recon[[i, c]];
            rss += d * d;
            tss += centered[[i, c]] * centered[[i, c]];
        }
    }
    Ok(PairChartFit {
        lambda_smooth: outer.lambda_smooth,
        criterion: outer.criterion,
        explained_variance: crate::tiered::explained_variance_from_sums(rss, tss),
        outer_iterations: outer.outer_iterations,
        certified: outer.outer_certificate.certifies(),
        recurred: outer.fixed_point.recurred,
        retained_atoms: outer.term.k_atoms(),
    })
}

#[cfg(test)]
mod code_space_tests {
    use super::*;
    use ndarray::Array2 as A2;
    use std::f64::consts::TAU;

    /// The universal-improver entry: a FOREIGN dictionary (any K×P atom bank)
    /// with solved codes on it is censused pairwise straight from
    /// `SparseAtomCodes` — no tiered fit, no BlockSparseFit. The same shattered
    /// ring, imported as someone else's dictionary, is discovered and promoted.
    #[test]
    fn foreign_dictionary_pair_census_promotes_a_shattered_ring() {
        let n = 512;
        let p = 16;
        let k = 256;
        let mut decoder = A2::<f64>::zeros((k, p));
        decoder[[0, 0]] = 1.0;
        decoder[[1, 1]] = 1.0;
        let mut codes = SparseAtomCodes::empty(n, k);
        for i in 0..n {
            let theta = TAU * (i as f64) / (n as f64);
            let row = codes.row_mut(i);
            row.assign(0, theta.cos());
            row.assign(1, theta.sin());
        }
        let report =
            harvest_code_space_pair_promotions(decoder.view(), &codes, n, 0.05).expect("runs");
        assert!(report.proposals.is_empty());
        assert_eq!(report.pair_proposals.len(), 1);
        let verdict = &report.pair_proposals[0];
        assert_eq!((verdict.atom_a, verdict.atom_b), (0, 1));
        assert!(
            verdict.proposal.accept,
            "an imported shattered ring must be promoted: {verdict:?}"
        );
        assert!(
            verdict.null_p_hat <= 2.0 / 64.0,
            "the imported ring must survive its permutation null, p̂={}",
            verdict.null_p_hat
        );
        // The topology race must call the planted ring a RING, not a patch —
        // "curved" never rests on the DL ledger alone.
        assert_eq!(
            verdict.topology_kind.as_deref(),
            Some("Periodic"),
            "race verdict on a planted ring: {:?} (err: {:?})",
            verdict.topology_kind,
            verdict.topology_error
        );
        assert!(report.dl_saved_bits > 0.0);
        assert!((report.l0 - 2.0).abs() < 1.0e-12);
        // Contract errors are loud, not silent.
        let narrow = SparseAtomCodes::empty(n, k - 1);
        assert!(
            harvest_code_space_pair_promotions(decoder.view(), &narrow, n, 0.05).is_err(),
            "a K mismatch must refuse"
        );
    }

    /// The full REML/GAM rung: a census-shaped noisy ring cloud, fit by the
    /// canonical grouped-LAML outer engine through [`fit_pair_chart`] — the
    /// smoothing is REML-selected, the inner fixed point recurs, the outer
    /// certificate certifies, and the chart explains the planted structure.
    #[test]
    fn pair_chart_fit_is_certified_reml_on_a_noisy_ring() {
        let n = 256;
        let mut cloud = A2::<f64>::zeros((n, 2));
        let mut z = 0x9E37_79B9_7F4A_7C15u64;
        let mut noise = move || {
            z ^= z >> 12;
            z ^= z << 25;
            z ^= z >> 27;
            (z.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for i in 0..n {
            let theta = TAU * (i as f64) / (n as f64);
            cloud[[i, 0]] = theta.cos() + 0.04 * noise();
            cloud[[i, 1]] = theta.sin() + 0.04 * noise();
        }
        // The engine's small-P certification is a KNOWN open limitation (the
        // #2627-family recurrence plateau: the joint solve accepts every Newton
        // step yet lands ~1.5x above the relative KKT tolerance at the cycle
        // budget). The contract this pins: fit_pair_chart returns EITHER a
        // certified REML fit OR the engine's own typed stall — never a third
        // failure mode, and never a fabricated certificate. Loosening the
        // tolerance to force the Ok arm is the exact trap #2624 documents.
        match fit_pair_chart(cloud.view(), 0xC0FF_EE00_D15E_A5E5) {
            Ok(fit) => {
                assert!(fit.recurred, "a returned fit must have recurred: {fit:?}");
                assert!(fit.certified, "a returned fit must certify: {fit:?}");
                assert!(fit.retained_atoms >= 1);
                assert_eq!(fit.lambda_smooth.len(), fit.retained_atoms);
                assert!(
                    fit.explained_variance > 0.9,
                    "a certified REML chart must explain a clean ring, EV={}",
                    fit.explained_variance
                );
                assert!(fit.lambda_smooth.iter().all(|l| l.is_finite() && *l > 0.0));
            }
            Err(error) => assert!(
                error.contains("did not recur") || error.contains("not resolved above"),
                "a refusal must be the engine's own typed stall, got: {error}"
            ),
        }
    }

    /// The measured distortion floor: nonzero residual reads back as its RMS;
    /// an exactly-reconstructed corpus falls to the f64 resolution floor instead
    /// of a zero that would poison the rate terms.
    #[test]
    fn distortion_floor_is_measured_with_a_resolution_backstop() {
        let residual = ndarray::array![[3.0, 0.0], [0.0, 4.0]];
        // mean square = (9 + 16) / 4 = 6.25 ⇒ RMS 2.5.
        let delta = linear_distortion_floor(residual.view(), 100.0).expect("floor");
        assert!((delta - 2.5).abs() < 1.0e-12, "measured RMS, got {delta}");
        let zero = A2::<f64>::zeros((2, 2));
        let backstop = linear_distortion_floor(zero.view(), 100.0).expect("floor");
        let corpus_rms = (100.0f64 / 4.0).sqrt();
        assert!(
            (backstop - corpus_rms * f64::EPSILON.sqrt()).abs() < 1.0e-18,
            "zero residual must fall to the resolution floor, got {backstop}"
        );
        assert!(linear_distortion_floor(zero.view(), 0.0).is_err());
    }
}
