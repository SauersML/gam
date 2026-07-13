//! Typed automatic seed construction for the minimal SAE fit surface (#2236).
//!
//! This module owns topology discovery, PCA seeding, atom plans, padded basis
//! stacks, cold routing policy, deterministic jitter, and decoder LSQ init.

use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3};

use super::*;

pub struct SaeMinimalSeedRequest<'a> {
    pub target: ArrayView2<'a, f64>,
    pub atom_basis: Vec<String>,
    pub atom_dim: Vec<usize>,
    pub assignment_kind: SaeFitAssignmentKind,
    pub alpha: f64,
    pub tau: f64,
    pub threshold: f64,
    pub top_k: Option<usize>,
    pub random_state: u64,
    pub initial_logits: Option<ArrayView2<'a, f64>>,
    pub initial_coords: Option<ArrayView3<'a, f64>>,
}

pub struct SaeMinimalSeedReport {
    /// One immutable authority for every atom kind, chart dimension,
    /// resolution, center set, basis width, and reference metric.
    pub geometry_plans: Vec<SaeAtomGeometryPlan>,
    pub basis_values: Array3<f64>,
    pub basis_jacobian: Array4<f64>,
    pub decoder_coefficients: Array3<f64>,
    pub smooth_penalties: Array3<f64>,
    pub initial_logits: Array2<f64>,
    pub initial_coords: Array3<f64>,
    pub refine_routing: bool,
}

pub fn build_sae_minimal_seed(
    mut request: SaeMinimalSeedRequest<'_>,
) -> Result<SaeMinimalSeedReport, String> {
    let (n_obs, p_out) = request.target.dim();
    let k_atoms = request.atom_basis.len();
    if n_obs == 0 || p_out == 0 {
        return Err(format!(
            "sae_manifold_fit_minimal: target must be non-empty; got shape ({n_obs}, {p_out})"
        ));
    }
    if k_atoms == 0 {
        return Err("sae_manifold_fit_minimal: atom_basis must be non-empty".to_string());
    }
    if request.atom_dim.len() != k_atoms {
        return Err(format!(
            "sae_manifold_fit_minimal: atom_dim length {} must equal atom_basis length {k_atoms}",
            request.atom_dim.len()
        ));
    }
    let admission = admit_sae_fit_shape(
        n_obs,
        p_out,
        k_atoms,
        request.atom_dim.iter().copied().max().unwrap_or(1),
        request.assignment_kind,
        request.top_k,
    )?;
    if admission.lane != crate::front_door::SaeFitLane::DenseCertification {
        return Err(
            "build_sae_minimal_seed is the dense-certification constructor; overcomplete hard-TopK requests must use the support-sparse minimal seed entry"
                .to_string(),
        );
    }
    if !request.target.iter().all(|value| value.is_finite()) {
        return Err("sae_manifold_fit_minimal: target contains non-finite values".to_string());
    }
    for basis in &request.atom_basis {
        crate::atom_schema::validate_seed_basis_kind(basis)?;
    }

    let auto_labels = if request.atom_basis.iter().any(|basis| basis == "auto") {
        Some(sae_output_energy_cluster_labels(request.target, k_atoms))
    } else {
        None
    };
    let (overrides, coord_overrides) = if let Some(labels) = auto_labels.as_ref() {
        crate::structure_harvest::resolve_auto_primary_atoms(
            request.target,
            labels,
            &mut request.atom_basis,
            &mut request.atom_dim,
        )?
    } else {
        (vec![None; k_atoms], vec![None; k_atoms])
    };
    let basis_kinds: Vec<SaeAtomBasisKind> = request
        .atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect();
    let mut seed_coords =
        sae_pca_seed_initial_coords(request.target, &basis_kinds, &request.atom_dim)?;
    if basis_kinds
        .iter()
        .any(|kind| matches!(kind, SaeAtomBasisKind::Mobius))
    {
        let labels = auto_labels
            .unwrap_or_else(|| sae_output_energy_cluster_labels(request.target, k_atoms));
        sae_refine_mobius_seed_coords_by_cluster(
            request.target,
            &basis_kinds,
            &labels,
            &mut seed_coords,
        )?;
    }
    // Install each auto topology winner's exact coordinate realization LAST,
    // after generic PCA construction and topology-specific refinements.  The
    // topology kind and chart are one evidence candidate; rebuilding or
    // refining the chart after the verdict silently creates a different seed.
    // In particular, this preserves the unfolded geodesic coordinates of an
    // intrinsic sheet winner instead of re-creasing it through PCA.
    for (atom_idx, chart) in coord_overrides.iter().enumerate() {
        if let Some(chart) = chart {
            let d = chart.ncols().min(seed_coords.shape()[2]);
            for row in 0..n_obs.min(chart.nrows()) {
                for col in 0..d {
                    seed_coords[[atom_idx, row, col]] = chart[[row, col]];
                }
            }
        }
    }
    let plans = sae_build_atom_plans(
        request.target,
        &request.atom_basis,
        &request.atom_dim,
        seed_coords.view(),
        request.random_state,
        &overrides,
    )?;
    let effective_atom_dim: Vec<usize> = plans.iter().map(SaeAtomBuildPlan::latent_dim).collect();

    let coords_are_cold = request.initial_coords.is_none();
    let mut start_coords = match request.initial_coords {
        Some(view) => {
            let shape = view.shape();
            if shape[0] != k_atoms || shape[1] != n_obs {
                return Err(format!(
                    "sae_manifold_fit_minimal: initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {shape:?}"
                ));
            }
            for (atom_idx, &d) in effective_atom_dim.iter().enumerate() {
                if d > shape[2] {
                    return Err(format!(
                        "sae_manifold_fit_minimal: initial_coords D_max={} is too small for atom {atom_idx} latent_dim={d}",
                        shape[2]
                    ));
                }
            }
            if !view.iter().all(|value| value.is_finite()) {
                return Err(
                    "sae_manifold_fit_minimal: initial_coords contains non-finite values"
                        .to_string(),
                );
            }
            view.to_owned()
        }
        None => seed_coords,
    };
    if coords_are_cold
        && k_atoms > 1
        && matches!(
            request.assignment_kind,
            SaeFitAssignmentKind::Softmax | SaeFitAssignmentKind::OrderedBetaBernoulli
        )
    {
        let labels = sae_output_energy_cluster_labels(request.target, k_atoms);
        let plan_kinds: Vec<SaeAtomBasisKind> =
            plans.iter().map(|plan| plan.kind().clone()).collect();
        sae_refine_periodic_seed_coords_by_cluster(
            request.target,
            &plan_kinds,
            &labels,
            &mut start_coords,
        )?;
    }
    let (basis_values, basis_jacobian, smooth_penalties, basis_sizes, _) =
        sae_build_padded_basis_stacks(&plans, start_coords.view(), n_obs)?;

    let warm_logits = match request.initial_logits {
        Some(view) => {
            if view.dim() != (n_obs, k_atoms) {
                return Err(format!(
                    "sae_manifold_fit_minimal: initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
                    view.dim()
                ));
            }
            if !view.iter().all(|value| value.is_finite()) {
                return Err(
                    "sae_manifold_fit_minimal: initial_logits contains non-finite values"
                        .to_string(),
                );
            }
            Some(view.to_owned())
        }
        None => None,
    };
    let logits_are_cold = warm_logits.is_none();
    let mut initial_logits = match warm_logits {
        Some(logits) => logits,
        None if request.assignment_kind == SaeFitAssignmentKind::ThresholdGate => {
            const THRESHOLD_GATE_SEED_MARGIN: f64 = 1.0;
            Array2::<f64>::from_elem(
                (n_obs, k_atoms),
                request.threshold + THRESHOLD_GATE_SEED_MARGIN,
            )
        }
        None if k_atoms == 1
            && request.assignment_kind == SaeFitAssignmentKind::OrderedBetaBernoulli =>
        {
            const ORDERED_BETA_BERNOULLI_K1_PRESENT_GATE_LOGIT: f64 = 6.0;
            Array2::<f64>::from_elem(
                (n_obs, k_atoms),
                ORDERED_BETA_BERNOULLI_K1_PRESENT_GATE_LOGIT * request.tau,
            )
        }
        None => Array2::<f64>::zeros((n_obs, k_atoms)),
    };
    if logits_are_cold
        && k_atoms > 1
        && matches!(
            request.assignment_kind,
            SaeFitAssignmentKind::Softmax | SaeFitAssignmentKind::OrderedBetaBernoulli
        )
    {
        const RESIDUAL_SEED_GAIN: f64 = 4.0;
        initial_logits = sae_residual_seed_logits(
            basis_values.view(),
            &basis_sizes,
            request.target,
            RESIDUAL_SEED_GAIN,
        )?;
    }
    if logits_are_cold {
        const RANDOM_STATE_LOGIT_JITTER: f64 = 1.0e-3;
        let mut state = request
            .random_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for row in 0..n_obs {
            for atom_idx in 0..k_atoms {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
                initial_logits[[row, atom_idx]] += RANDOM_STATE_LOGIT_JITTER * (2.0 * unit - 1.0);
            }
        }
    }
    let decoder_coefficients = sae_decoder_lsq_init(
        basis_values.view(),
        &basis_sizes,
        request.target,
        initial_logits.view(),
        request.assignment_kind.tag(),
        request.alpha,
        request.tau,
        request.threshold,
        request.top_k,
    )?;
    let geometry_plans = plans.into_iter().map(|plan| plan.geometry).collect();

    Ok(SaeMinimalSeedReport {
        geometry_plans,
        basis_values,
        basis_jacobian,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords: start_coords,
        refine_routing: logits_are_cold && coords_are_cold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_seed_rejects_empty_target() {
        let target = Array2::<f64>::zeros((0, 2));
        let error = build_sae_minimal_seed(SaeMinimalSeedRequest {
            target: target.view(),
            atom_basis: vec!["periodic".to_string()],
            atom_dim: vec![1],
            assignment_kind: SaeFitAssignmentKind::Softmax,
            alpha: 1.0,
            tau: 1.0,
            threshold: 0.0,
            top_k: None,
            random_state: 0,
            initial_logits: None,
            initial_coords: None,
        })
        .err()
        .expect("empty target must fail");
        assert!(error.contains("non-empty"));
    }
}
