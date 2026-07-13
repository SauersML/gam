//! Atom resolution policy and analytic-evaluator rebuild (issue #2236).
//!
//! Widths are derived only from constructor-validated [`SaeAtomGeometryPlan`]s.
//! There is intentionally no evaluator reconstruction from realized widths.

use std::sync::Arc;

use super::{SaeAtomBasisKind, SaeAtomGeometryPlan, SaeBasisSecondJet};

/// Default per-axis harmonic order for a torus atom (Φ has `(2H+1)^d`
/// columns). Three harmonics per axis gives a 7-column 1-D factor and a
/// 49-column tensor basis at `d=2`, which is the smallest expansion that
/// reliably resolves a non-trivial signal on `T^2` without exploding the
/// design.
pub const SAE_DEFAULT_TORUS_HARMONICS: usize = 3;
/// Sphere chart basis size (lat/lon ⇒ `[1, x, y, z, xy, yz, xz]`).
pub const SAE_SPHERE_BASIS_SIZE: usize = 7;

/// Duchon nullspace knob `m` for a SAE-manifold atom of latent dimension
/// `dim`, sized so the native reproducing-norm Gram (`PenaltySource::Primary`)
/// on the scale-free polyharmonic basis is well-posed in every dimension.
///
/// `m` maps to the polynomial-nullspace order via `duchon_nullspace_from_m`
/// (`m = 1 → Zero/p=1`, `m = 2 → Linear/p=2`, `m = k+1 → Degree(k)/p=k+1`), so
/// `p_order == m`. The pure-polyharmonic basis the `DuchonCoordinateEvaluator`
/// evaluates uses `s = 0`; sizing the null space by `2·m > dim + 2` keeps the
/// kernel CPD/null-space adequacy comfortable across dimensions, whose smallest
/// integer solution is `m = ⌊dim / 2⌋ + 2`.
///
/// For `dim == 1` this reproduces the historical `m = 2` (constant + linear
/// null space). For `dim ≥ 2` it grows the null space with the latent
/// dimension — the previous hard-coded `m = 2` left the resolved power/order
/// inconsistent with the power-0 evaluator. The seed build and every
/// [`DuchonCoordinateEvaluator`] refresh read this same derived `m`, so the
/// design `Φ`, its jet, and the penalty stay column-consistent (the issue-247
/// invariant).
pub fn sae_duchon_atom_m(dim: usize) -> usize {
    dim / 2 + 2
}
/// Maximum total monomial degree for a Euclidean tangent-patch SAE atom.
///
/// IMPORTANT (#1201): this is **degree 2**, so the `"euclidean"` atom basis is a
/// QUADRATIC patch `{1, t, t²}` at `d_atom = 1` (and `{1, t_a, t_b, t_a², t_a t_b,
/// t_b²}` at `d_atom = 2`), NOT a single straight decoder direction `γ(t) = t·b`.
/// Any comparison that calls the `"euclidean"` atom the "linear" baseline is
/// therefore comparing curved-vs-QUADRATIC, not curved-vs-linear — label such
/// comparisons honestly. (A genuinely linear secant baseline is available as the
/// per-atom hybrid-split LINEAR candidate, `crate::terms::sae::hybrid_split`,
/// which fits `b₀ + (t − t̄)·b₁` exactly; the `"euclidean"` OUTER fit path is the
/// quadratic patch.)
pub const SAE_EUCLIDEAN_PATCH_MAX_DEGREE: usize = 2;

/// Largest explicitly selectable Euclidean-patch degree in the topology race.
/// Seed patches use degree 2 ([`SAE_EUCLIDEAN_PATCH_MAX_DEGREE`]); a structure
/// birth may explicitly persist the degree-3 line candidate. The degree lives
/// in [`SaeBasisResolution::Polynomial`], never inferred from decoder width.
pub const SAE_EUCLIDEAN_PATCH_RACE_MAX_DEGREE: usize = 3;

/// Flat-line polynomial degree of a Cylinder `S¹ × ℝ` atom's line axis (axis 1).
/// Mirrors the Euclidean-patch degree so the cylinder's flat factor matches the
/// patch candidate it races against; `Ml = SAE_CYLINDER_LINE_DEGREE + 1`.
pub const SAE_CYLINDER_LINE_DEGREE: usize = 2;

/// Möbius production convention (#2240): circle harmonics on the DOUBLE-COVER
/// angle and the width monomial degree of the deck-invariant basis. Must match
/// the seed builder and the topology-race candidate (`MobiusHarmonicEvaluator::
/// new(3, 2)`, deck-invariant width 10).
pub const SAE_MOBIUS_CIRCLE_HARMONICS: usize = 3;
pub const SAE_MOBIUS_WIDTH_DEGREE: usize = 2;

pub const SAE_MAX_PERIODIC_HARMONICS: usize = 4096;

pub fn sae_periodic_basis_size(n_harmonics: usize) -> Result<usize, String> {
    if n_harmonics > SAE_MAX_PERIODIC_HARMONICS {
        return Err(format!(
            "sae_build_periodic_atom: n_harmonics={n_harmonics} exceeds dense limit {SAE_MAX_PERIODIC_HARMONICS}"
        ));
    }
    n_harmonics
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(1))
        .ok_or_else(|| {
            format!("sae_build_periodic_atom: basis size overflows for n_harmonics={n_harmonics}")
        })
}

/// Build per-atom Rust basis evaluators so the Newton loop can refresh
/// `Phi_k` and `dPhi_k/dt` between steps without bouncing back to Python.
///
/// Every evaluator is rebuilt directly from its tagged resolution. No
/// harmonic order, polynomial degree, or Duchon centers are inferred from a
/// realized width or coordinate snapshot.
pub fn build_sae_basis_evaluators(
    geometry_plans: &[SaeAtomGeometryPlan],
) -> Result<Vec<Arc<dyn SaeBasisSecondJet>>, String> {
    geometry_plans
        .iter()
        .map(SaeAtomGeometryPlan::build_evaluator)
        .collect()
}

/// Native [`SaeAtomBasisKind`] for an exact canonical basis token. Public and
/// Rust fit front doors validate seed tokens before calling this converter;
/// `Precomputed` remains an internal representation for typed native artifacts.
pub fn sae_atom_basis_kind_from_str(value: &str) -> SaeAtomBasisKind {
    match value {
        "duchon" => SaeAtomBasisKind::Duchon,
        "periodic" => SaeAtomBasisKind::Periodic,
        "sphere" => SaeAtomBasisKind::Sphere,
        "torus" => SaeAtomBasisKind::Torus,
        "projective_plane" => SaeAtomBasisKind::ProjectivePlane,
        "klein_bottle" => SaeAtomBasisKind::KleinBottle,
        // The genuinely-linear atom is the degree-1 monomial patch, distinct
        // from the degree-2 canonical `euclidean` patch.
        "linear" => SaeAtomBasisKind::Linear,
        // #BSF — `"linear_block"` is a BSF block expressed AS a manifold-SAE atom:
        // γ_g(t) = t·D_g with an orthonormal decoder frame D_g and block-level
        // (norm-selection or separate-gate) gating. Mathematically it IS the
        // `Linear` degree-1 patch (the honest encoding of "BSF ⊂ ManifoldSAE" is a
        // frame + gating CONFIG on the linear atom, not a new basis type), so it
        // maps to `SaeAtomBasisKind::Linear` for construction/evidence; the
        // exact `linear_block` public label is retained by the fit artifact.
        "linear_block" => SaeAtomBasisKind::Linear,
        "euclidean" => SaeAtomBasisKind::EuclideanPatch,
        "poincare" => SaeAtomBasisKind::Poincare,
        "cylinder" => SaeAtomBasisKind::Cylinder,
        "mobius" => SaeAtomBasisKind::Mobius,
        "finite_set" => SaeAtomBasisKind::FiniteSet,
        other => SaeAtomBasisKind::Precomputed(other.to_string()),
    }
}

/// The canonical lowercase string name of an atom basis kind — the inverse of
/// [`sae_atom_basis_kind_from_str`] for the round-trippable kinds, and the
/// string the python `from_payload` boundary reads under each atom's
/// `"basis_kind"` key and each plan's `"kind"` key. Derived from the FITTED
/// atom (not the user's input metadata) so a structure-search-grown / shrunk
/// dictionary serializes its DISCOVERED per-atom topology, not the input one.
pub fn sae_atom_basis_kind_name(kind: &SaeAtomBasisKind) -> String {
    match kind {
        SaeAtomBasisKind::Periodic => "periodic".to_string(),
        SaeAtomBasisKind::Duchon => "duchon".to_string(),
        SaeAtomBasisKind::Sphere => "sphere".to_string(),
        SaeAtomBasisKind::Torus => "torus".to_string(),
        SaeAtomBasisKind::ProjectivePlane => "projective_plane".to_string(),
        SaeAtomBasisKind::KleinBottle => "klein_bottle".to_string(),
        // The degree-1 and degree-2 patch families have distinct canonical names.
        SaeAtomBasisKind::Linear => "linear".to_string(),
        SaeAtomBasisKind::EuclideanPatch => "euclidean".to_string(),
        SaeAtomBasisKind::Poincare => "poincare".to_string(),
        SaeAtomBasisKind::Cylinder => "cylinder".to_string(),
        SaeAtomBasisKind::Mobius => "mobius".to_string(),
        // The finite-set (discrete-anchor) candidate is inert scaffolding that is
        // not enrolled in the topology race by default, so a discovered dictionary
        // never actually carries it (see
        // `structure_harvest::finite_set_race_enrolled`). Round-trip it under the
        // same `"finite_set"` token the gam-sae inference/harvest paths already
        // emit, so serialization stays consistent the moment it is enrolled.
        SaeAtomBasisKind::FiniteSet => "finite_set".to_string(),
        SaeAtomBasisKind::Precomputed(name) => name.clone(),
    }
}
