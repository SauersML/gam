//! Joint Godambe / composite-likelihood reporting types.
//!
//! The implementation lives on [`super::SaeManifoldTerm`] because it must use
//! the fitted arrow layout and exact stationarity solver.  In particular, the
//! sandwich is formed from complete per-row score vectors before any atom block
//! is extracted; a collection of atomwise sandwiches is not equivalent when
//! atoms, routing coordinates, and nuisance coordinates are coupled.

/// Joint composite-likelihood information charge.
///
/// `joint_clic_dof = tr(J A^-1)` uses the complete fitted parameter vector
/// (routing, coordinates, every atom, and decoder border).  It is intentionally
/// not paired with a blockwise "model based" number: diagonal atom subblocks
/// cannot recover `tr(F A^-1)` in the presence of nuisance and cross-atom
/// coupling.
#[derive(Debug, Clone, Copy)]
pub struct CompositeLikelihoodCharge {
    /// `tr(J A^-1)` — joint composite-likelihood (CLIC / Takeuchi) effective dof.
    pub joint_clic_dof: f64,
}
