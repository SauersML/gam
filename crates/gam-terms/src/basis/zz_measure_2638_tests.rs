//! #2638 follow-up: gates on [`duchon_resolve_chart`], the seam that lets the
//! ψ-derivative entry point ANSWER for a cold spec instead of refusing it.
//!
//! `1f9171850` closed #2638 by routing the three ψ-derivative builders through
//! `duchon_frozen_radial_chart`, which refuses a spec carrying no frozen `V`
//! while the constrained kernel block is non-empty. That is the right answer for
//! those three builders — they receive `centers`, not `data`, so they *cannot*
//! solve the `Ω_c v = μ G_c v` eigenproblem that decides `V`.
//!
//! `build_duchon_basis_log_kappa_derivatives(data, spec)` does receive the data,
//! and its documented job is to return the ψ-jet of `build_duchon_basis(data,
//! spec)`. For a cold spec that forward build is perfectly well defined — it
//! adopts a fresh `V` — so the derivative entry point has no reason to refuse:
//! it resolves the same chart and differentiates in it.
//!
//! What is gated here:
//!
//!  1. `duchon_resolve_chart` is behaviour-preserving — `build_duchon_basis` on
//!     the RESOLVED spec reproduces the cold build bit-for-bit. That pins all
//!     five resolved decisions at once (centers, effective null-space order,
//!     seeded anisotropy, adopted `V`, identifiability transform `T`).
//!  2. `T` is read off the `V`-ROTATED design. The pre-#2638 derivative context
//!     derived it from the un-rotated one, which constrains a different function
//!     space.
//!  3. The cold-spec ψ-jet matches a finite difference of the forward taken at
//!     the resolved chart, on EVERY penalty block. Before the entry point
//!     resolved, this same call returned the raw-`Z`-chart jet: 32× the true
//!     Primary jet and 242× too small on OperatorMass (measured; see the table
//!     in the commit that added this file).
//!  4. The chart-motion decomposition itself, printed rather than asserted —
//!     the evidence that the residual #2638 reported is `|FD_cold − FD_frozen|`
//!     and not a dropped term.

#![cfg(test)]

use ndarray::Array2;

use super::*;

/// The `_no_ident` fixture from
/// `test_duchon_log_kappa_derivative_matchesfd_dim1_power1_linear_no_ident`:
/// 1-D, `power=1`, `Linear` null space, 8 farthest-point centers, no outer
/// identifiability constraint. Four penalties survive (Primary,
/// DoublePenaltyNullspace, OperatorMass, OperatorTension).
fn no_ident_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    (data, spec)
}

/// Same geometry, but with the default outer identifiability constraint, so the
/// realized `T` is non-trivial and pin 2 has something to discriminate.
fn constrained_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let (data, mut spec) = no_ident_fixture();
    spec.identifiability = SpatialIdentifiability::default();
    (data, spec)
}

/// A 2-D fixture whose auto-seeded anisotropy contrasts are non-trivial, so the
/// resolver's `auto_seed_aniso_contrasts` step is exercised rather than skipped.
fn aniso_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 60usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        // Deliberately anisotropic support: axis 1 spans ~8× axis 0.
        data[[i, 0]] = t;
        data[[i, 1]] = 8.0 * (t * 7.0).sin();
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        // All-zero is the auto-seed sentinel: the forward replaces it with
        // geometry-derived contrasts, and a consumer that passes it through raw
        // builds a different kernel metric.
        aniso_log_scales: Some(vec![0.0, 0.0]),
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    (data, spec)
}

fn fro(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn duchon_metadata_chart(
    result: &BasisBuildResult,
) -> (Array2<f64>, Option<Array2<f64>>, Option<Array2<f64>>) {
    match &result.metadata {
        BasisMetadata::Duchon {
            centers,
            identifiability_transform,
            radial_reparam,
            ..
        } => (
            centers.clone(),
            radial_reparam.clone(),
            identifiability_transform.clone(),
        ),
        other => panic!(
            "expected Duchon metadata, got {:?}",
            std::mem::discriminant(other)
        ),
    }
}

