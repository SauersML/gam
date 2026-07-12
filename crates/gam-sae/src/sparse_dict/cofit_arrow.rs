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
//! **#2275 certificate reconciliation.** The block lane's `certified` flag maps to
//! the joint solver's `EvidenceJointFitOutcome.fixed_point`: the arrow-routed fit
//! is `certified` iff the evidence policy certified an idempotent fixed point.

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

use crate::assignment::{AssignmentMode, SaeAssignment};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};
use super::coordinate::explained_variance_from_reconstruction;

/// Result of the arrow-Schur-routed co-fit linear tier (Stage 1).
#[derive(Clone, Debug)]
pub struct ArrowCofitReport {
    /// Composed reconstruction `N×P` read back from the fitted term.
    pub reconstructed: Array2<f32>,
    /// Explained variance of `reconstructed` against the target (mean-baseline via
    /// the shared `explained_variance_from_reconstruction` helper cofit uses).
    pub explained_variance: f64,
    /// #2275 certificate: `true` iff the joint solver certified an idempotent
    /// fixed point (`EvidenceJointFitOutcome.fixed_point`); `false` is the typed
    /// open certificate at `K ≫ rank`.
    pub certified: bool,
}

/// Tuning for the arrow-routed linear fit. `max_iter` must be generous enough for
/// the evidence policy to settle: `run_joint_fit_arrow_schur_for_quasi_laplace`
/// rejects a heuristic termination, so a too-small budget errors rather than
/// returning an open certificate.
#[derive(Clone, Copy, Debug)]
pub struct ArrowCofitConfig {
    pub log_lambda_sparse: f64,
    pub log_lambda_smooth: f64,
    pub max_iter: usize,
    pub step_size: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
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

    // One linear (degree-1 monomial) atom per block. Basis Φ = [1, t₁,…,t_b]
    // (M = b+1), jet = ∂Φ/∂t (row 0 the intercept → 0; the identity block for the
    // linear columns), decoder = [0; block's b decoder rows] (intercept seeded 0,
    // the target is the de-meaned Tier-1 residual). Roughness gram is 0 — a linear
    // atom is flat.
    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(g);
    for gi in 0..g {
        let t = &coord_blocks[gi];
        let mut phi = Array2::<f64>::zeros((n, b + 1));
        let mut jet = Array3::<f64>::zeros((n, b + 1, b));
        for i in 0..n {
            phi[[i, 0]] = 1.0;
            for r in 0..b {
                phi[[i, r + 1]] = t[[i, r]];
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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("t1_block_{gi}"),
            SaeAtomBasisKind::Linear,
            b,
            phi,
            jet,
            atom_decoder,
            gram,
        )?;
        atoms.push(atom);
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

    let recon_f64 = term.try_fitted_for_rho(&rho)?;
    let reconstructed = recon_f64.mapv(|v| v as f32);
    let explained_variance =
        explained_variance_from_reconstruction(target, reconstructed.view())?;

    Ok(ArrowCofitReport {
        reconstructed,
        explained_variance,
        certified: outcome.fixed_point,
    })
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
        let block_ev = explained_variance_from_reconstruction(x.view(), block_recon.view())
            .expect("block EV");

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
            "[#2023 5a] block_ev={:.6} arrow_ev={:.6} certified={}",
            block_ev, arrow.explained_variance, arrow.certified
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
}
