//! Typed automatic seed construction for the minimal SAE fit surface (#2236).
//!
//! This module owns topology discovery, PCA seeding, atom plans, padded basis
//! stacks, cold routing policy, and decoder LSQ init.

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
    /// Decoder smoothness strength `λ` the fit starts from; the LSQ decoder
    /// seed is the MAP under the same `½ λ tr(BᵀSB)` prior.
    pub smoothness: f64,
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

fn install_discovered_geometry_overrides(
    plans: &mut [SaeAtomBuildPlan],
    basis_kinds: &[SaeAtomBasisKind],
    geometry_overrides: Vec<Option<SaeAtomGeometryPlan>>,
) -> Result<(), String> {
    if plans.len() != basis_kinds.len() || plans.len() != geometry_overrides.len() {
        return Err(format!(
            "install_discovered_geometry_overrides: plans={}, kinds={}, overrides={} must align",
            plans.len(),
            basis_kinds.len(),
            geometry_overrides.len()
        ));
    }
    for (atom_idx, geometry) in geometry_overrides.into_iter().enumerate() {
        if let Some(geometry) = geometry {
            if geometry.kind() != &basis_kinds[atom_idx] {
                return Err(format!(
                    "install_discovered_geometry_overrides: discovered geometry kind {:?} disagrees with resolved atom {atom_idx} kind {:?}",
                    geometry.kind(),
                    basis_kinds[atom_idx]
                ));
            }
            plans[atom_idx] = SaeAtomBuildPlan { geometry };
        }
    }
    Ok(())
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
    let (overrides, coord_overrides, geometry_overrides) =
        if let Some(labels) = auto_labels.as_ref() {
            crate::structure_harvest::resolve_auto_primary_atoms(
                request.target,
                labels,
                &mut request.atom_basis,
                &mut request.atom_dim,
            )?
        } else {
            (
                vec![None; k_atoms],
                vec![None; k_atoms],
                vec![None; k_atoms],
            )
        };
    let basis_kinds: Vec<SaeAtomBasisKind> = request
        .atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect::<Result<Vec<_>, String>>()?;
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
    let mut plans = sae_build_atom_plans(
        request.target,
        &request.atom_basis,
        &request.atom_dim,
        seed_coords.view(),
        request.random_state,
        &overrides,
    )?;
    install_discovered_geometry_overrides(&mut plans, &basis_kinds, geometry_overrides)?;
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
    // Every routing map that picks atoms per row needs the atoms' cold charts
    // separated by the data, or each atom sees the same shared PCA chart and no
    // row prefers any atom. Hard TopK is such a map too (#4519).
    let cold_routing = k_atoms > 1
        && matches!(
            request.assignment_kind,
            SaeFitAssignmentKind::Softmax
                | SaeFitAssignmentKind::OrderedBetaBernoulli
                | SaeFitAssignmentKind::TopK
        );
    if coords_are_cold && cold_routing {
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
    // Neutral cold logits are a tie on every row. Hard TopK breaks ties toward
    // the lower atom index, so it would route every row to the first `top_k`
    // atoms and leave the rest with an identically zero decoder (#4519). The
    // residual seed is the data's own per-row preference; TopK reads only its
    // order, and rows the data leave tied stay tied.
    if logits_are_cold && cold_routing {
        const RESIDUAL_SEED_GAIN: f64 = 4.0;
        initial_logits = sae_residual_seed_logits(
            basis_values.view(),
            &basis_sizes,
            request.target,
            RESIDUAL_SEED_GAIN,
        )?;
    }
    let decoder_coefficients = sae_decoder_lsq_init(
        basis_values.view(),
        &basis_sizes,
        smooth_penalties.view(),
        request.smoothness,
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
            smoothness: 1.0,
            initial_logits: None,
            initial_coords: None,
        })
        .err()
        .expect("empty target must fail");
        assert!(error.contains("non-empty"));
    }

    /// The cold routing seed is the data's own residual preference and
    /// nothing else (#3090): no `random_state`-keyed logit perturbation is
    /// added on top of it. A single atom has no routing to seed, so its cold
    /// logits stay exactly neutral; two atoms carry exactly the mean-centred
    /// residual logits of the seed basis stack, bit for bit, for every seed.
    #[test]
    fn cold_routing_seed_is_the_residual_seed_without_perturbation_3090() {
        let n = 64usize;
        let mut target = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let theta = std::f64::consts::TAU * row as f64 / n as f64;
            let radius = if row % 2 == 0 { 1.0 } else { 2.5 };
            target[[row, 0]] = radius * theta.cos();
            target[[row, 1]] = radius * theta.sin();
        }
        let seed = |atoms: usize, random_state: u64| {
            build_sae_minimal_seed(SaeMinimalSeedRequest {
                target: target.view(),
                atom_basis: vec!["periodic".to_string(); atoms],
                atom_dim: vec![1; atoms],
                assignment_kind: SaeFitAssignmentKind::Softmax,
                alpha: 1.0,
                tau: 1.0,
                threshold: 0.0,
                top_k: None,
                random_state,
                smoothness: 1.0,
                initial_logits: None,
                initial_coords: None,
            })
            .expect("a planted two-circle seed must build")
        };

        let single = seed(1, 11);
        assert!(
            single.initial_logits.iter().all(|&logit| logit == 0.0),
            "a single atom's cold logits must be exactly neutral"
        );

        for random_state in [0_u64, 11, 20260920] {
            let report = seed(2, random_state);
            let basis_sizes: Vec<usize> = report
                .geometry_plans
                .iter()
                .map(|plan| plan.basis_size().expect("seed plans carry a basis size"))
                .collect();
            let residual =
                sae_residual_seed_logits(report.basis_values.view(), &basis_sizes, target.view(), 4.0)
                    .expect("the residual seed of the report's own basis must build");
            assert_eq!(
                report.initial_logits, residual,
                "random_state={random_state}: the cold logits must be the residual seed itself"
            );
        }
    }

    /// #4519 — a cold hard-TopK(1) seed over two planted circles routes rows to
    /// both atoms and seeds a nonzero decoder for each. Neutral logits tie every
    /// row, TopK breaks ties toward atom 0, and atom 1 used to get no rows and an
    /// identically zero decoder, which the #2822 entry gate refuses.
    #[test]
    fn cold_topk_seed_routes_rows_to_every_atom_4519() {
        let n = 48usize;
        let mut target = Array2::<f64>::zeros((n, 4));
        for row in 0..n {
            let theta = std::f64::consts::TAU * (row / 2) as f64 / (n / 2) as f64;
            let plane = 2 * (row % 2);
            target[[row, plane]] = 2.0 * theta.cos();
            target[[row, plane + 1]] = 2.0 * theta.sin();
        }
        let report = build_sae_minimal_seed(SaeMinimalSeedRequest {
            target: target.view(),
            atom_basis: vec!["periodic".to_string(); 2],
            atom_dim: vec![1, 1],
            assignment_kind: SaeFitAssignmentKind::TopK,
            alpha: 1.0,
            tau: 1.0,
            threshold: 0.0,
            top_k: Some(1),
            random_state: 45,
            initial_logits: None,
            initial_coords: None,
        })
        .expect("a planted two-circle TopK seed must build");
        for atom_idx in 0..2 {
            let won = (0..n)
                .filter(|&row| {
                    crate::assignment::topk_row(report.initial_logits.row(row), 1)[atom_idx] == 1.0
                })
                .count();
            assert!(won > 0, "atom {atom_idx} wins no row of the cold TopK seed");
            let decoder_energy: f64 = report
                .decoder_coefficients
                .index_axis(ndarray::Axis(0), atom_idx)
                .iter()
                .map(|value| value * value)
                .sum();
            assert!(
                decoder_energy > 0.0,
                "atom {atom_idx} has an identically zero cold decoder"
            );
        }
    }

    #[test]
    fn discovered_torus_metric_plan_replaces_the_builder_default_atomically() {
        let default = SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Torus,
            2,
            SaeBasisResolution::TorusHarmonics { per_axis_order: 3 },
            SaeReferenceMetricPlan::FlatRectangularTorus { tau: 0.0 },
        )
        .unwrap();
        let selected = SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Torus,
            2,
            SaeBasisResolution::TorusHarmonics { per_axis_order: 5 },
            SaeReferenceMetricPlan::EmbeddedDonutTorus { tau: 0.8 },
        )
        .unwrap();
        let mut plans = vec![SaeAtomBuildPlan { geometry: default }];
        install_discovered_geometry_overrides(
            &mut plans,
            &[SaeAtomBasisKind::Torus],
            vec![Some(selected.clone())],
        )
        .unwrap();
        assert_eq!(plans[0].geometry, selected);

        let mismatch = SaeAtomGeometryPlan::projective_plane(1).unwrap();
        assert!(
            install_discovered_geometry_overrides(
                &mut plans,
                &[SaeAtomBasisKind::Torus],
                vec![Some(mismatch)],
            )
            .is_err(),
            "a parallel kind scalar must never override the resolved typed plan"
        );
    }
}
