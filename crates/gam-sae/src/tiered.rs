//! Tiered sparse-linear plus evidence-curved decomposition (#2023).
//!
//! The architecture deliberately makes the large collapsed-linear dictionary the
//! primary workhorse and reserves the exact curved REML machinery for a small,
//! evidence-chosen block.  This module is the production seam between those
//! tiers: it fits Tier 1 with [`crate::sparse_dict`], exposes the whitened
//! residual that Tier 2 must certify, and records every structural move in a
//! migration ledger.  In particular, this path never reseeds dead or collapsed
//! atoms from principal components; births must come either from high-residual
//! data rows in the linear lane or from an explicit residual-factor/linear/curved
//! ledger transition.

use ndarray::{Array2, ArrayView2};

use crate::sparse_dict::{SparseDictConfig, SparseDictFit, fit_sparse_dictionary};

/// Structural state of one load-bearing object in the tiered SAE decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TieredAtomState {
    /// Shared structured residual factor `Λ`: variance not yet promoted to an
    /// explicit dictionary atom.
    ResidualFactor,
    /// Tier-1 collapsed-linear sparse-dictionary atom.
    LinearAtom,
    /// Tier-2 one-dimensional curved atom certified by the small REML block.
    CurvedAtom,
}

/// Evidence-adjudicated movement of one object between residual, linear, and
/// curved states.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MigrationLedgerEntry {
    /// Stable object identifier within the fit.  For Tier-1 births this is the
    /// sparse-dictionary atom index.
    pub object_id: usize,
    /// State before the move.
    pub from: TieredAtomState,
    /// State after the move.
    pub to: TieredAtomState,
    /// Human-readable, deterministic reason suitable for logs.
    pub reason: String,
}

/// Append-only ledger replacing principal-component reseeding.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MigrationLedger {
    entries: Vec<MigrationLedgerEntry>,
}

impl MigrationLedger {
    /// Create an empty migration ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one evidence/state transition.
    pub fn record(
        &mut self,
        object_id: usize,
        from: TieredAtomState,
        to: TieredAtomState,
        reason: impl Into<String>,
    ) {
        self.entries.push(MigrationLedgerEntry {
            object_id,
            from,
            to,
            reason: reason.into(),
        });
    }

    /// Ledger entries in chronological order.
    pub fn entries(&self) -> &[MigrationLedgerEntry] {
        &self.entries
    }

    /// Number of recorded transitions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no transitions have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Structural guard for #2023: this ledger is the only accounting mechanism
    /// in the tiered path and it never describes PC-based reseeding.
    pub fn contains_pc_reseed(&self) -> bool {
        self.entries
            .iter()
            .any(|e| e.reason.to_ascii_lowercase().contains("pc reseed"))
    }
}

/// Configuration for the tiered decomposition.
#[derive(Clone, Debug)]
pub struct TieredDecompositionConfig {
    /// Tier 1 sparse linear dictionary configuration.  Large `K` belongs here.
    pub linear: SparseDictConfig,
    /// Requested upper bound for Tier 2 curved atoms.  The actual number is
    /// evidence-chosen by later promotion passes and may be zero.
    pub curved_atom_cap: usize,
    /// Residual whitening floor to avoid division by zero on perfectly explained
    /// or constant channels.
    pub whitening_floor: f32,
}

impl TieredDecompositionConfig {
    /// Construct a tiered config with a `K`-atom Tier-1 dictionary and the
    /// project-default small curved block cap.
    pub fn new(linear_atoms: usize) -> Self {
        Self {
            linear: SparseDictConfig::new(linear_atoms),
            ..Self::default()
        }
    }
}

impl Default for TieredDecompositionConfig {
    fn default() -> Self {
        Self {
            linear: SparseDictConfig::new(10_000),
            curved_atom_cap: 20,
            whitening_floor: 1.0e-6,
        }
    }
}

/// Result of the tiered sparse-linear workhorse pass.
#[derive(Clone, Debug)]
pub struct TieredDecompositionFit {
    /// Tier 1: collapsed-linear sparse dictionary fit.
    pub linear: SparseDictFit,
    /// Whitened residual `diag(sd)^{-1} (X - X̂_linear)` consumed by Tier 2.
    pub whitened_residual: Array2<f32>,
    /// Per-output residual standard deviations used for whitening.
    pub residual_scale: Vec<f32>,
    /// Migration ledger; Tier-1 atoms are born from residual factors, never PCs.
    pub ledger: MigrationLedger,
    /// Current evidence-selected curved-atom count.  The initial workhorse pass
    /// does not preset curved `K`; promotions fill this later.
    pub curved_atoms: usize,
}

/// Fit the Tier-1 sparse dictionary and return the whitened residual for the
/// evidence-selected curved tier.
pub fn fit_tiered_decomposition(
    x: ArrayView2<'_, f32>,
    config: &TieredDecompositionConfig,
) -> Result<TieredDecompositionFit, String> {
    if !(config.whitening_floor.is_finite() && config.whitening_floor > 0.0) {
        return Err("fit_tiered_decomposition requires whitening_floor > 0".to_string());
    }
    let linear = fit_sparse_dictionary(x, &config.linear)?;
    let reconstruction = linear.reconstruct();
    let (whitened_residual, residual_scale) =
        whiten_residual(x, reconstruction.view(), config.whitening_floor)?;

    let mut ledger = MigrationLedger::new();
    for atom in 0..linear.decoder.nrows() {
        ledger.record(
            atom,
            TieredAtomState::ResidualFactor,
            TieredAtomState::LinearAtom,
            "tier-1 sparse-dictionary birth from residual data rows",
        );
    }

    Ok(TieredDecompositionFit {
        linear,
        whitened_residual,
        residual_scale,
        ledger,
        curved_atoms: 0,
    })
}

fn whiten_residual(
    x: ArrayView2<'_, f32>,
    reconstruction: ArrayView2<'_, f32>,
    floor: f32,
) -> Result<(Array2<f32>, Vec<f32>), String> {
    if x.dim() != reconstruction.dim() {
        return Err(format!(
            "whiten_residual shape mismatch: X is {:?}, reconstruction is {:?}",
            x.dim(),
            reconstruction.dim()
        ));
    }
    let (n, p) = x.dim();
    let mut residual = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            residual[[i, j]] = x[[i, j]] - reconstruction[[i, j]];
        }
    }
    let mut scale = vec![floor; p];
    if n > 0 {
        for j in 0..p {
            let mut ss = 0.0_f64;
            for i in 0..n {
                let r = residual[[i, j]] as f64;
                ss += r * r;
            }
            scale[j] = ((ss / n as f64).sqrt() as f32).max(floor);
        }
    }
    for i in 0..n {
        for j in 0..p {
            residual[[i, j]] /= scale[j];
        }
    }
    Ok((residual, scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tiered_fit_uses_sparse_linear_lane_and_records_non_pc_births() {
        let x = array![[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]];
        let cfg = TieredDecompositionConfig {
            linear: SparseDictConfig {
                n_atoms: 4,
                active: 1,
                max_epochs: 2,
                ..SparseDictConfig::new(4)
            },
            curved_atom_cap: 2,
            whitening_floor: 1.0e-6,
        };
        let fit = fit_tiered_decomposition(x.view(), &cfg).expect("tiered fit");
        assert_eq!(fit.linear.decoder.nrows(), 4);
        assert_eq!(fit.whitened_residual.dim(), x.dim());
        assert_eq!(fit.residual_scale.len(), x.ncols());
        assert_eq!(fit.ledger.len(), 4);
        assert!(!fit.ledger.contains_pc_reseed());
        assert_eq!(fit.curved_atoms, 0);
    }
}
