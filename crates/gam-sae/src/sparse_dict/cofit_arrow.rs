//! #2023 Increment 5a (Stage 1, dense) — route the co-fit's linear tier through the
//! unified arrow-Schur inner solver instead of the hand-rolled block coordinate
//! descent.
//!
//! The thesis (#2232): there is ONE joint solver. The block-sparse linear tier a
//! `BlockSparseFit` produces is a set of `d = b` Euclidean (linear) atoms of that
//! solver — a linear atom is the degree-1 / `b₂ = 0` special case of the curved
//! atom. This module builds those linear atoms + a frozen-support assignment from a
//! block routing `(decoder, blocks, codes, γ)` and runs
//! [`SaeManifoldTerm::run_joint_fit_arrow_schur_for_quasi_laplace`] on them, then
//! reads the composed reconstruction + the fixed-point certificate back out.
//!
//! **Stage 1 scope (this file).** DENSE assignment, moderate `K`: the parity
//! evidence the fold needs does not require the massive-`K` support-sparse state
//! (risk #1 of the design applies to PRODUCTION routing, not the parity fixtures).
//! The behaviour-parity claim is that the arrow-routed linear fit **matches or
//! beats** the direct block-sparse linear reconstruction in explained variance —
//! the joint solve descends the same linear model, warm-started from the same
//! routing, so it cannot do worse. The support-sparse in-core seam (an engine
//! entry consuming `SaeAssignmentState::from_topk_support`) is Stage 2, specced to
//! the engine lane on #2023 — this module never touches the driver internals, it
//! only CALLS `SaeManifoldTerm`.
//!
//! **#2275 certificate reconciliation.** The joint solver's
//! `EvidenceJointFitOutcome.fixed_point` is a construction gate, not report
//! telemetry: `false` returns a non-convergence error, so every
//! [`ArrowCofitReport`] necessarily came from an idempotent re-entry.

use std::collections::HashSet;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

use gam_terms::latent::LatentManifold;

use super::block_chart::{
    BlockChartComposeConfig, BlockChartComposeResult, compose_block_coordinate_charts,
};
use super::coordinate::explained_variance_from_reconstruction;
use crate::assignment::{AssignmentMode, SaeAssignment};
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};

/// Result of the arrow-Schur-routed co-fit linear tier (Stage 1).
#[derive(Clone, Debug)]
pub struct ArrowCofitReport {
    /// Composed reconstruction `N×P` read back from the fitted term.
    pub reconstructed: Array2<f32>,
    /// Explained variance of `reconstructed` against the target (mean-baseline via
    /// the shared `explained_variance_from_reconstruction` helper cofit uses).
    pub explained_variance: f64,
    /// Number of curved (periodic) atoms folded into the joint solve — the count
    /// of blocks whose BIC-gated chart discovery ([`compose_block_coordinate_charts`])
    /// promoted them from a flat linear atom to a curved chart. `0` for the
    /// linear-only path ([`cofit_linear_via_arrow`]). This is the curved-birth
    /// count the migration ledger banks.
    pub n_curved_atoms: usize,
    /// Total BIC complexity charge (`Σ ½·d_eff·ln n_eff`, nats) of the curved
    /// charts folded in — the description-length currency the ledger records as
    /// `dl_bits`. `0.0` for the linear-only path.
    pub curved_charge: f64,
}

/// Tuning for the arrow-routed linear fit. `max_iter` must be generous enough for
/// the evidence policy to settle: a too-small iteration budget is a
/// non-convergence error and can never produce an [`ArrowCofitReport`].
#[derive(Clone, Debug)]
pub struct ArrowCofitConfig {
    pub log_lambda_sparse: f64,
    pub log_lambda_smooth: f64,
    pub max_iter: usize,
    pub step_size: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    /// Number of periodic-harmonic basis columns `M = 2·h + 1` for a folded
    /// curved atom (must be odd, `>= 3`). `3` is one harmonic — an exact circle,
    /// the ring the chart-discovery lane certifies. Used only by
    /// [`cofit_composed_via_arrow`].
    pub curved_num_basis: usize,
    /// Chart-discovery configuration passed to [`compose_block_coordinate_charts`]
    /// to decide WHICH blocks fold in as curved atoms. Its `block_size` /
    /// `block_topk` / `gamma` are overwritten from the passed routing so the tiers
    /// always agree on geometry. Used only by [`cofit_composed_via_arrow`].
    pub chart: BlockChartComposeConfig,
}

impl Default for ArrowCofitConfig {
    fn default() -> Self {
        Self {
            log_lambda_sparse: (1.0e-4f64).ln(),
            log_lambda_smooth: (1.0e-4f64).ln(),
            max_iter: 128,
            step_size: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            curved_num_basis: 3,
            chart: BlockChartComposeConfig::default(),
        }
    }
}

/// Build the linear atoms + frozen-support assignment from a block routing and run
/// the unified arrow-Schur joint solve on them (Stage 1, dense).
///
/// `decoder` is `K×P` (`K = G·b`), `blocks` is `N×k` (which of the `G` blocks fired
/// per row), `codes` is `N×k×b` (the signed within-block coordinates). `γ` scales
/// the stored codes into latent coordinates (`t = γ·code`), matching the block
/// lane's tied-scalar convention.
pub fn cofit_linear_via_arrow(
    target: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    gamma: f32,
    config: &ArrowCofitConfig,
) -> Result<ArrowCofitReport, String> {
    require_fitting_iteration("cofit_linear_via_arrow", config.max_iter)?;
    let (n, k_active) = blocks.dim();
    let b = codes.shape()[2];
    if b == 0 {
        return Err("cofit_linear_via_arrow: block_size (codes.shape[2]) must be >= 1".to_string());
    }
    if decoder.nrows() == 0 || decoder.nrows() % b != 0 {
        return Err(format!(
            "cofit_linear_via_arrow: decoder rows {} must be a positive multiple of block_size {b}",
            decoder.nrows()
        ));
    }
    let g = decoder.nrows() / b;
    let p = decoder.ncols();
    if target.nrows() != n || target.ncols() != p {
        return Err(format!(
            "cofit_linear_via_arrow: target {:?} incompatible with N={n}, P={p}",
            target.dim()
        ));
    }
    if codes.shape()[0] != n || codes.shape()[1] != k_active {
        return Err(format!(
            "cofit_linear_via_arrow: codes shape {:?} incompatible with blocks {:?}",
            codes.shape(),
            blocks.dim()
        ));
    }

    // Per-block latent coordinates `T_g` (N×b): row i carries the block's signed
    // codes (scaled by γ) if block g fired in row i, else 0. A block that does not
    // fire in a row contributes zero to that row's reconstruction, so the extra
    // atoms a `top_k_support` pick may include are inert.
    let mut coord_blocks: Vec<Array2<f64>> = (0..g).map(|_| Array2::<f64>::zeros((n, b))).collect();
    for i in 0..n {
        for j in 0..k_active {
            let atom = blocks[[i, j]] as usize;
            if atom >= g {
                return Err(format!(
                    "cofit_linear_via_arrow: routed block {atom} out of range (G={g})"
                ));
            }
            for r in 0..b {
                coord_blocks[atom][[i, r]] = (gamma * codes[[i, j, r]]) as f64;
            }
        }
    }

    // One linear (degree-1 monomial) atom per block — a flat atom is the degree-1
    // special case of the curved atom the composed lane also builds.
    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(g);
    for gi in 0..g {
        atoms.push(build_linear_atom(gi, &coord_blocks[gi], decoder, b, p)?);
    }

    // Frozen routing: dense logits large on fired atoms, small elsewhere; the
    // `top_k_support(k)` mode reads them read-only and keeps the top-k support per
    // row (== the block routing, up to inert zero-coord fillers).
    const ON: f64 = 1.0;
    const OFF: f64 = -1.0e3;
    let mut logits = Array2::<f64>::from_elem((n, g), OFF);
    for i in 0..n {
        for j in 0..k_active {
            logits[[i, blocks[[i, j]] as usize]] = ON;
        }
    }
    let k_support = k_active.min(g).max(1);
    let assignment = SaeAssignment::from_blocks_with_mode(
        logits,
        coord_blocks,
        AssignmentMode::top_k_support(k_support),
    )?;

    let mut term = SaeManifoldTerm::new(atoms, assignment)?;
    term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(
        config.log_lambda_sparse,
        config.log_lambda_smooth,
        (0..g).map(|_| Array1::<f64>::zeros(b)).collect(),
    );

    let target_f64 = target.mapv(|v| v as f64);
    let outcome = term.run_joint_fit_arrow_schur_for_quasi_laplace(
        target_f64.view(),
        &mut rho,
        None,
        config.max_iter,
        config.step_size,
        config.ridge_ext_coord,
        config.ridge_beta,
    )?;
    require_idempotent_fixed_point(
        outcome.fixed_point,
        "cofit_linear_via_arrow",
        config.max_iter,
    )?;

    let recon_f64 = term.try_fitted_for_rho(&rho)?;
    let reconstructed = recon_f64.mapv(|v| v as f32);
    let explained_variance = explained_variance_from_reconstruction(target, reconstructed.view())?;

    Ok(ArrowCofitReport {
        reconstructed,
        explained_variance,
        n_curved_atoms: 0,
        curved_charge: 0.0,
    })
}

/// Build one linear (degree-1 monomial) atom for block `gi`: basis `Φ = [1, t₁,…,t_b]`
/// (`M = b+1`), jet `∂Φ/∂t` (row 0 the intercept → 0; the identity block for the
/// linear columns), decoder = `[0; block's b decoder rows]`, roughness Gram `0`
/// (a linear atom is flat). Shared by the linear-only and composed paths.
fn build_linear_atom(
    gi: usize,
    coord_block: &Array2<f64>,
    decoder: ArrayView2<'_, f32>,
    b: usize,
    p: usize,
) -> Result<SaeManifoldAtom, String> {
    let n = coord_block.nrows();
    let mut phi = Array2::<f64>::zeros((n, b + 1));
    let mut jet = Array3::<f64>::zeros((n, b + 1, b));
    for i in 0..n {
        phi[[i, 0]] = 1.0;
        for r in 0..b {
            phi[[i, r + 1]] = coord_block[[i, r]];
            jet[[i, r + 1, r]] = 1.0;
        }
    }
    let mut atom_decoder = Array2::<f64>::zeros((b + 1, p));
    for r in 0..b {
        for c in 0..p {
            atom_decoder[[r + 1, c]] = decoder[[gi * b + r, c]] as f64;
        }
    }
    let gram = Array2::<f64>::zeros((b + 1, b + 1));
    SaeManifoldAtom::new_with_provided_function_gram(
        format!("t1_block_{gi}"),
        SaeAtomBasisKind::Linear,
        b,
        phi,
        jet,
        atom_decoder,
        gram,
    )
}

/// Build one curved (periodic-harmonic) atom for block `gi` over a per-row angle
/// coordinate `sᵢ = θᵢ / 2π ∈ [0,1)`, `θᵢ = atan2(t₂, t₁)` of the block's first two
/// latent coords. The fundamental sin/cos decoder rows are seeded from the block's
/// two decoder directions scaled by the mean firing radius `r̄` so the seed already
/// traces the block's circle (`Φ·β ≈ r̄(cos θ · d₀ + sin θ · d₁)`); the joint solve
/// refines `β` and the angle. Returns the atom and its `N×1` angle coordinate block.
fn build_curved_atom(
    gi: usize,
    coord_block: &Array2<f64>,
    decoder: ArrayView2<'_, f32>,
    b: usize,
    p: usize,
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    m: usize,
) -> Result<(SaeManifoldAtom, Array2<f64>), String> {
    let n = coord_block.nrows();
    let inv_two_pi = 1.0 / (2.0 * std::f64::consts::PI);
    let mut angle = Array2::<f64>::zeros((n, 1));
    let mut radius_sum = 0.0;
    let mut radius_n = 0.0;
    for i in 0..n {
        let t0 = coord_block[[i, 0]];
        let t1 = coord_block[[i, 1]];
        if t0 != 0.0 || t1 != 0.0 {
            let theta = t1.atan2(t0);
            // Wrap θ/2π into [0,1); atan2 ∈ (-π,π] ⇒ raw ∈ (-0.5,0.5].
            let mut s = theta * inv_two_pi;
            if s < 0.0 {
                s += 1.0;
            }
            angle[[i, 0]] = s;
            radius_sum += (t0 * t0 + t1 * t1).sqrt();
            radius_n += 1.0;
        }
    }
    let r_bar = if radius_n > 0.0 {
        radius_sum / radius_n
    } else {
        1.0
    };

    let (phi, jet) = evaluator.evaluate(angle.view())?;
    // Seed the fundamental harmonic decoder from the block's two directions so the
    // atom starts on the block's circle. Column layout (PeriodicHarmonicEvaluator):
    // 0 = constant, 1 = sin(2π·t), 2 = cos(2π·t), … .
    let mut atom_decoder = Array2::<f64>::zeros((m, p));
    if m >= 3 {
        for c in 0..p {
            atom_decoder[[2, c]] = r_bar * decoder[[gi * b, c]] as f64;
            atom_decoder[[1, c]] = r_bar * decoder[[gi * b + 1, c]] as f64;
        }
    }
    let gram = Array2::<f64>::eye(m);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        format!("circle_block_{gi}"),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        atom_decoder,
        gram,
    )?
    .with_basis_second_jet(evaluator.clone());
    Ok((atom, angle))
}

/// Fold BOTH the flat linear tier AND the BIC-discovered curved tier into ONE
/// unified arrow-Schur joint solve (#2023 Increment 5, the composed cutover).
///
/// This is the composed analogue of [`cofit_linear_via_arrow`] and the match-or-beat
/// replacement for the hand-rolled A/B coordinate descent of
/// [`super::cofit_block_and_curved`]: instead of alternating a linear-code refit with
/// a guarded curved-chart commit, it builds a mixed atom set — a flat linear atom for
/// every block the chart-discovery lane leaves flat, a curved periodic atom for every
/// block it promotes to a ring — and descends the SINGLE joint objective the unified
/// engine minimises. A linear atom is the degree-1 special case of the curved atom
/// (#2232), so the two tiers live in one assembly and one solve; the residual-
/// orthogonality trap the alternation closed is closed here by construction, because
/// the joint solve never freezes one tier against the other's stale residual.
///
/// **Chart discovery, not chart fitting.** [`compose_block_coordinate_charts`] is
/// called ONCE — only to decide WHICH blocks are curved (its BIC-gated
/// `selected_chart_blocks` / `selected_chart_pairs`) and to price them
/// (`curved_charge`). The actual reconstruction is the arrow-Schur joint fit over the
/// mixed atoms, warm-started from the routing, NOT the compose lane's own radial
/// charts. Blocks with `b < 2` cannot carry an angle and stay linear.
///
/// **Stage 1 scope (dense).** DENSE assignment, moderate `K`, same as
/// [`cofit_linear_via_arrow`]: the massive-`K` support-sparse engine seam is Stage 2
/// (specced to the engine lane on #2023). This module only CALLS `SaeManifoldTerm`.
pub fn cofit_composed_via_arrow(
    target: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    gamma: f32,
    config: &ArrowCofitConfig,
) -> Result<ArrowCofitReport, String> {
    require_fitting_iteration("cofit_composed_via_arrow", config.max_iter)?;
    let (n, k_active) = blocks.dim();
    let b = codes.shape()[2];
    if b == 0 {
        return Err(
            "cofit_composed_via_arrow: block_size (codes.shape[2]) must be >= 1".to_string(),
        );
    }
    if decoder.nrows() == 0 || decoder.nrows() % b != 0 {
        return Err(format!(
            "cofit_composed_via_arrow: decoder rows {} must be a positive multiple of block_size {b}",
            decoder.nrows()
        ));
    }
    let g = decoder.nrows() / b;
    let p = decoder.ncols();
    if target.nrows() != n || target.ncols() != p {
        return Err(format!(
            "cofit_composed_via_arrow: target {:?} incompatible with N={n}, P={p}",
            target.dim()
        ));
    }
    if codes.shape()[0] != n || codes.shape()[1] != k_active {
        return Err(format!(
            "cofit_composed_via_arrow: codes shape {:?} incompatible with blocks {:?}",
            codes.shape(),
            blocks.dim()
        ));
    }
    let m = config.curved_num_basis;
    if m < 3 || m % 2 == 0 {
        return Err(format!(
            "cofit_composed_via_arrow: curved_num_basis must be odd and >= 3, got {m}"
        ));
    }

    // --- Chart discovery: which blocks fold in as curved atoms, and their charge. ---
    let mut chart_cfg = config.chart.clone();
    chart_cfg.block_size = b;
    chart_cfg.block_topk = k_active;
    chart_cfg.gamma = gamma;
    chart_cfg.residual_target = true;
    let discovery = compose_block_coordinate_charts(target, decoder, blocks, codes, &chart_cfg)?;
    let curved_blocks = accepted_curved_blocks(&discovery, g, b);
    let curved_charge = accepted_curved_charge(&discovery);

    // --- Per-block latent coordinates T_g (N×b): the block's γ-scaled signed codes
    //     on the rows it fired, 0 elsewhere. Shared by both atom kinds. ---
    let mut coord_blocks: Vec<Array2<f64>> = (0..g).map(|_| Array2::<f64>::zeros((n, b))).collect();
    for i in 0..n {
        for j in 0..k_active {
            let atom = blocks[[i, j]] as usize;
            if atom >= g {
                return Err(format!(
                    "cofit_composed_via_arrow: routed block {atom} out of range (G={g})"
                ));
            }
            for r in 0..b {
                coord_blocks[atom][[i, r]] = (gamma * codes[[i, j, r]]) as f64;
            }
        }
    }

    // --- Build the mixed atom set: curved for discovered ring blocks, linear else. ---
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m)?);
    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(g);
    let mut assignment_coords: Vec<Array2<f64>> = Vec::with_capacity(g);
    let mut manifolds: Vec<LatentManifold> = Vec::with_capacity(g);
    let mut n_curved_atoms = 0usize;
    for gi in 0..g {
        if curved_blocks.contains(&gi) {
            let (atom, angle) =
                build_curved_atom(gi, &coord_blocks[gi], decoder, b, p, &evaluator, m)?;
            atoms.push(atom);
            assignment_coords.push(angle);
            manifolds.push(LatentManifold::Circle { period: 1.0 });
            n_curved_atoms += 1;
        } else {
            let atom = build_linear_atom(gi, &coord_blocks[gi], decoder, b, p)?;
            atoms.push(atom);
            assignment_coords.push(coord_blocks[gi].clone());
            manifolds.push(LatentManifold::Euclidean);
        }
    }

    // --- Frozen routing: dense logits large on fired atoms, small elsewhere. ---
    const ON: f64 = 1.0;
    const OFF: f64 = -1.0e3;
    let mut logits = Array2::<f64>::from_elem((n, g), OFF);
    for i in 0..n {
        for j in 0..k_active {
            logits[[i, blocks[[i, j]] as usize]] = ON;
        }
    }
    let k_support = k_active.min(g).max(1);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        assignment_coords,
        manifolds,
        AssignmentMode::top_k_support(k_support),
    )?;

    let mut term = SaeManifoldTerm::new(atoms, assignment)?;
    term.set_guards_enabled(false);
    // Per-atom ARD log-precision vectors sized to each atom's latent dim (b for a
    // linear atom, 1 for a curved atom).
    let log_ard: Vec<Array1<f64>> = (0..g)
        .map(|gi| {
            if curved_blocks.contains(&gi) {
                Array1::<f64>::zeros(1)
            } else {
                Array1::<f64>::zeros(b)
            }
        })
        .collect();
    let mut rho = SaeManifoldRho::new(config.log_lambda_sparse, config.log_lambda_smooth, log_ard);

    let target_f64 = target.mapv(|v| v as f64);
    let outcome = term.run_joint_fit_arrow_schur_for_quasi_laplace(
        target_f64.view(),
        &mut rho,
        None,
        config.max_iter,
        config.step_size,
        config.ridge_ext_coord,
        config.ridge_beta,
    )?;
    require_idempotent_fixed_point(
        outcome.fixed_point,
        "cofit_composed_via_arrow",
        config.max_iter,
    )?;

    let recon_f64 = term.try_fitted_for_rho(&rho)?;
    let reconstructed = recon_f64.mapv(|v| v as f32);
    let explained_variance = explained_variance_from_reconstruction(target, reconstructed.view())?;

    Ok(ArrowCofitReport {
        reconstructed,
        explained_variance,
        n_curved_atoms,
        curved_charge,
    })
}

fn require_idempotent_fixed_point(
    fixed_point: bool,
    entry: &str,
    max_iter: usize,
) -> Result<(), String> {
    if fixed_point {
        Ok(())
    } else {
        Err(format!(
            "{entry}: deterministic joint-solver re-entry did not reach an idempotent fixed point within {max_iter} iterations"
        ))
    }
}

fn require_fitting_iteration(entry: &str, max_iter: usize) -> Result<(), String> {
    if max_iter > 0 {
        Ok(())
    } else {
        Err(format!(
            "{entry}: zero iterations is a checkpoint freeze, not an idempotent fitted fixed point"
        ))
    }
}

/// The BIC-selected chart-owned block set (single blocks + both members of each
/// selected pair), restricted to blocks that can carry an angle (`b >= 2`).
fn accepted_curved_blocks(result: &BlockChartComposeResult, g: usize, b: usize) -> HashSet<usize> {
    let mut s = HashSet::new();
    if b < 2 {
        return s;
    }
    for &gi in &result.selected_chart_blocks {
        if gi < g {
            s.insert(gi);
        }
    }
    for &(g0, g1) in &result.selected_chart_pairs {
        if g0 < g {
            s.insert(g0);
        }
        if g1 < g {
            s.insert(g1);
        }
    }
    s
}

/// Total BIC complexity charge (nats) of the selected charts — the ledger's
/// description-length currency for the curved births.
fn accepted_curved_charge(result: &BlockChartComposeResult) -> f64 {
    let mut charge = 0.0;
    for rec in &result.block_records {
        if rec.evidence.selected_by_bic {
            charge += rec.evidence.charge;
        }
    }
    for rec in &result.pair_records {
        if rec.evidence.selected_by_bic {
            charge += rec.evidence.charge;
        }
    }
    charge
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_dict::{
        BlockSparseConfig, fit_block_sparse_dictionary, reconstruct_block_sparse_rows,
    };
    use ndarray::Array2;

    /// #2023 Inc 5a Stage 1: the arrow-Schur-routed linear tier reproduces the
    /// block-sparse linear reconstruction — its EV must MATCH-OR-BEAT the direct
    /// block reconstruction (same linear model, warm-started from the same routing,
    /// descended by the unified joint solver). Moderate K, dense lane.
    #[test]
    fn arrow_routed_linear_tier_matches_or_beats_block_reconstruction_2023() {
        // Planted linear structure: 3 directions in P=8, K = 3 blocks of b=2.
        let (p, b, n_blocks) = (8usize, 2usize, 3usize);
        let n = 120usize;
        let mut x = Array2::<f32>::zeros((n, p));
        let mut s = 0x2023_5a01u64;
        for i in 0..n {
            for d in 0..n_blocks {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let amp = ((s >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
                x[[i, 2 * d]] += amp;
                x[[i, 2 * d + 1]] += 0.5 * amp;
            }
        }

        let mut config = BlockSparseConfig::new(n_blocks, b);
        config.block_topk = n_blocks;
        config.max_epochs = 60;
        config.aux_k = 2;
        let fit = fit_block_sparse_dictionary(x.view(), &config)
            .expect("block-sparse linear fit must converge on planted structure");

        let block_recon = reconstruct_block_sparse_rows(
            fit.decoder.view(),
            fit.blocks.view(),
            fit.codes.view(),
            b,
        )
        .expect("block reconstruction");
        let block_ev =
            explained_variance_from_reconstruction(x.view(), block_recon.view()).expect("block EV");

        let arrow = cofit_linear_via_arrow(
            x.view(),
            fit.decoder.view(),
            fit.blocks.view(),
            fit.codes.view(),
            fit.gamma,
            &ArrowCofitConfig::default(),
        )
        .expect("arrow-routed linear cofit must run end to end");

        eprintln!(
            "[#2023 5a] block_ev={:.6} arrow_ev={:.6}",
            block_ev, arrow.explained_variance
        );
        assert!(
            arrow.explained_variance.is_finite(),
            "arrow EV must be finite, got {}",
            arrow.explained_variance
        );
        let tol = 1.0e-3 * (1.0 + block_ev.abs());
        assert!(
            arrow.explained_variance >= block_ev - tol,
            "#2023 5a: arrow-routed linear EV {} must match-or-beat block EV {} (tol {})",
            arrow.explained_variance,
            block_ev,
            tol
        );
        assert_eq!(arrow.reconstructed.dim(), (n, p));
    }

    /// Orthonormal-per-block decoder with a planted overlap, mirroring the fixture
    /// [`super::super::cofit`] is tested on: 3 blocks of size b=2 in P=5, block 1
    /// overlapping block 0 on e1 (so the tied projection double-counts e1), block 2
    /// the plane holding a planted circle.
    fn planted_decoder() -> Array2<f32> {
        let s = 1.0f32 / 2.0f32.sqrt();
        Array2::from_shape_vec(
            (6, 5),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, // e0
                0.0, 1.0, 0.0, 0.0, 0.0, // e1
                0.0, s, s, 0.0, 0.0, // (e1+e2)/√2
                0.0, s, -s, 0.0, 0.0, // (e1−e2)/√2
                0.0, 0.0, 0.0, 1.0, 0.0, // e3
                0.0, 0.0, 0.0, 0.0, 1.0, // e4
            ],
        )
        .unwrap()
    }

    /// Planted linear part in span{e0,e1,e2} plus a unit circle in the {e3,e4}
    /// plane, tiny noise — the trap fixture (a genuine curved chart in block 2).
    fn planted_data(n: usize) -> Array2<f32> {
        let mut x = Array2::<f32>::zeros((n, 5));
        for i in 0..n {
            let a = ((i * 7 + 1) % 17) as f32 / 17.0 - 0.5;
            let bb = ((i * 13 + 5) % 19) as f32 / 19.0 - 0.5;
            let cc = ((i * 5 + 3) % 23) as f32 / 23.0 - 0.5;
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let noise = 0.002 * (((i * 3) % 11) as f32 / 11.0 - 0.5);
            x[[i, 0]] = a;
            x[[i, 1]] = bb;
            x[[i, 2]] = cc;
            x[[i, 3]] = t.cos() as f32 + noise;
            x[[i, 4]] = t.sin() as f32 - noise;
        }
        x
    }

    /// Tied per-block routing (every row fires all three blocks; γ = 1): exactly
    /// what a converged block-sparse fit stores.
    fn tied_routing(
        x: &Array2<f32>,
        decoder: &Array2<f32>,
        b: usize,
    ) -> (Array2<u32>, Array3<f32>) {
        let n = x.nrows();
        let g = decoder.nrows() / b;
        let mut blocks = Array2::<u32>::zeros((n, g));
        let mut codes = Array3::<f32>::zeros((n, g, b));
        for i in 0..n {
            for gg in 0..g {
                blocks[[i, gg]] = gg as u32;
                for r in 0..b {
                    let atom = decoder.row(gg * b + r);
                    let mut dot = 0.0f32;
                    for c in 0..decoder.ncols() {
                        dot += x[[i, c]] * atom[c];
                    }
                    codes[[i, gg, r]] = dot;
                }
            }
        }
        (blocks, codes)
    }

    fn parity_chart_cfg() -> BlockChartComposeConfig {
        BlockChartComposeConfig {
            block_size: 2,
            block_topk: 3,
            min_firings: 8,
            crossfit_folds: 4,
            pair_screen: false,
            ..BlockChartComposeConfig::default()
        }
    }

    #[test]
    fn insufficient_iterations_return_error_instead_of_open_arrow_cofit_2023() {
        let n = 120usize;
        let b = 2usize;
        let decoder = planted_decoder();
        let x = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, b);
        let config = ArrowCofitConfig {
            max_iter: 0,
            chart: parity_chart_cfg(),
            ..ArrowCofitConfig::default()
        };

        let error = cofit_composed_via_arrow(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &config,
        )
        .expect_err("an open joint-solver re-entry must not mint ArrowCofitReport");
        assert!(
            error.contains("not an idempotent fitted fixed point"),
            "unexpected non-convergence error: {error}"
        );
    }

    /// #2023 Increment 5 (the composed cutover): the unified arrow-Schur joint solve
    /// over a MIXED linear + curved atom set must MATCH-OR-BEAT the hand-rolled A/B
    /// coordinate-descent co-fit ([`super::super::cofit_block_and_curved`]) in
    /// explained variance on the same planted routing, and it must actually fold in
    /// a curved atom (the discovered circle in block 2). This is the parity evidence
    /// that licenses deleting the alternation and repointing `fit_tiered`.
    #[test]
    fn composed_arrow_matches_or_beats_block_cofit_2023() {
        use crate::sparse_dict::{CofitConfig, cofit_block_and_curved};

        let n = 240usize;
        let b = 2usize;
        let decoder = planted_decoder();
        let x = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, b);

        // Reference: the hand-rolled A/B co-fit (the deletion target).
        let cofit_cfg = CofitConfig {
            code_ridge: 1.0e-6,
            chart: parity_chart_cfg(),
            ..CofitConfig::default()
        };
        let cofit = cofit_block_and_curved(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &cofit_cfg,
        )
        .expect("block A/B co-fit runs");

        // Candidate: the unified arrow-Schur composed joint solve.
        let arrow_cfg = ArrowCofitConfig {
            max_iter: 256,
            chart: parity_chart_cfg(),
            ..ArrowCofitConfig::default()
        };
        let arrow = cofit_composed_via_arrow(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &arrow_cfg,
        )
        .expect("composed arrow-Schur co-fit runs end to end");

        eprintln!(
            "[#2023 5] cofit_ev={:.6} arrow_ev={:.6} n_curved={} charge={:.4}",
            cofit.explained_variance,
            arrow.explained_variance,
            arrow.n_curved_atoms,
            arrow.curved_charge
        );

        assert!(
            arrow.explained_variance.is_finite(),
            "composed arrow EV must be finite, got {}",
            arrow.explained_variance
        );
        assert!(
            arrow.n_curved_atoms >= 1,
            "the composed fold must promote at least one curved atom (the circle in \
             block 2); got {}",
            arrow.n_curved_atoms
        );
        let tol = 1.0e-2 * (1.0 + cofit.explained_variance.abs());
        assert!(
            arrow.explained_variance >= cofit.explained_variance - tol,
            "#2023 5: composed arrow EV {} must match-or-beat block A/B co-fit EV {} (tol {})",
            arrow.explained_variance,
            cofit.explained_variance,
            tol
        );
        assert_eq!(arrow.reconstructed.dim(), (n, 5));
    }
}
