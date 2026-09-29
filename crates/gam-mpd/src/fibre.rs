//! The function-level fibre of a parameterization, bounded from its parameter Jacobian
//! (#2951): the oracle a declared gauge census is checked against.
//!
//! # The bound
//!
//! Let `θ ∈ ℝⁿ` parameterize a block `F_θ`, and let a declared finite input family
//! `x_1..x_N` give the map `Φ(θ) = (F_θ(x_1), …, F_θ(x_N)) ∈ ℝᵐ` with Jacobian
//! `J = ∂Φ/∂θ` (`m × n`). Every curve of parameters along which `F_θ` does not move (a
//! gauge orbit, an exact null coordinate, or a redundancy no family declares) has its
//! tangent in `ker J`. So the local fibre of `θ ↦ F_θ` through `θ` has dimension at most
//! `dim ker J = n − rank J`. More inputs only add rows, so the bound holds for the
//! function itself, not only for the sampled family.
//!
//! A computed `Ĵ` within `formation` (spectral norm) of the exact `J` has singular values
//! within `factor_singular_band` `+ formation` (`gam_linalg::roundoff`) of the exact ones (a backward-stable SVD,
//! then Weyl). A computed singular value above that band proves a nonzero exact one, so
//! the resolved count is a certified lower bound on `rank J` and `n − resolved` a
//! certified upper bound on the nullity. That is the only direction the arithmetic
//! certifies: a singular value inside the band is not proven zero. The rank is exact
//! only where the resolved count reaches `min(m, n)`.
//!
//! # What it is checked against
//!
//! A lower bound on a fibre comes from algebra, never from the SVD: the orbit dimension
//! and null coordinates of the families [`super::gauge`] declares, or a hand-derived
//! family. Where a lower bound meets this upper bound the fibre dimension is identified.
//! Where the declared census falls short of the upper bound, the difference is either a
//! redundancy no family declares or a rank the band did not resolve, and the oracle alone
//! does not say which. The planted toys of #2951 are the first kind: a cancelling pair of
//! GELU branches carries `GL(d)` and a bias shift, and two heads with one routing law
//! carry a cross-head `GL` of their value/output transports. Neither is a declared
//! family, which is why [`super::gauge`] charges its count as a bound on what a code must
//! carry and never as the whole fibre.
//!
//! The Jacobian and its formation bound are the caller's: an analytic Jacobian whose
//! rounding the caller derives. Nothing here differentiates.

use gam_runtime::resource::MemoryGovernor;
use ndarray::Array2;

use super::state::{StateError, resolve_stacked_factor};
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};

/// The Jacobian a nullity is stated for: `outputs` rows over `parameters` columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterJacobian {
    pub outputs: usize,
    pub parameters: usize,
}

/// The resolved rank of a parameter Jacobian and the nullity bounds it certifies.
#[derive(Clone, Debug)]
pub struct ParameterFibre {
    pub jacobian: ParameterJacobian,
    /// Singular values of the computed Jacobian, descending.
    pub singular_values: Vec<f64>,
    /// `gam_linalg::roundoff::factor_singular_band` plus the caller's formation bound.
    pub band: f64,
    /// Singular values above `band`: a certified lower bound on `rank J`.
    pub resolved_rank: usize,
}

impl ParameterFibre {
    /// `n − resolved_rank`: a certified upper bound on `dim ker J`, and so on the local
    /// fibre dimension.
    pub fn nullity_at_most(&self) -> usize {
        self.jacobian.parameters - self.resolved_rank
    }

    /// `n − min(m, n)`: the nullity the shape forces.
    pub fn nullity_at_least(&self) -> usize {
        self.jacobian.parameters - self.jacobian.parameters.min(self.jacobian.outputs)
    }

    /// `dim ker J`: exact when the resolved rank reaches `min(m, n)`, otherwise unresolved
    /// between the shape's nullity and [`ParameterFibre::nullity_at_most`].
    pub fn nullity(&self) -> Result<EvidenceStatus<(), ParameterJacobian>, EvidenceStatusError> {
        let (lower, upper) = (self.nullity_at_least(), self.nullity_at_most());
        if lower == upper {
            EvidenceStatus::exact(upper as f64, 0.0, ExactBasis::Algebraic, None, self.jacobian)
        } else {
            EvidenceStatus::unresolved(lower as f64, upper as f64, Extremum::Supremum, None, self.jacobian)
        }
    }

    /// The smallest resolved singular value and the largest unresolved one, when they
    /// exist: the measured gap the band falls in.
    pub fn gap(&self) -> (Option<f64>, Option<f64>) {
        let resolved = self.resolved_rank.checked_sub(1).map(|index| self.singular_values[index]);
        (resolved, self.singular_values.get(self.resolved_rank).copied())
    }

    /// The most fibre directions a census with `declared` proven orbit and null
    /// coordinates can leave uncounted: `nullity_at_most − declared`, or zero.
    pub fn undeclared_at_most(&self, declared: usize) -> usize {
        self.nullity_at_most().saturating_sub(declared)
    }
}

/// The fibre bound of the computed parameter Jacobian `jacobian` (`m × n`), within
/// `formation` of the exact one in spectral norm (a Frobenius bound qualifies). The SVD is
/// the state owner's rank-revealing read, [`resolve_stacked_factor`], reserved on
/// `governor`.
pub fn parameter_fibre(
    governor: &MemoryGovernor,
    jacobian: &Array2<f64>,
    formation: f64,
) -> Result<ParameterFibre, StateError> {
    let (outputs, parameters) = jacobian.dim();
    let local = resolve_stacked_factor(governor, jacobian, formation)?;
    Ok(ParameterFibre {
        jacobian: ParameterJacobian { outputs, parameters },
        singular_values: local.singular_values,
        band: local.band,
        resolved_rank: local.resolved_rank,
    })
}

#[cfg(test)]
#[path = "fibre_tests.rs"]
mod tests;
