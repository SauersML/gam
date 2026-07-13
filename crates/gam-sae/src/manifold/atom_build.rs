//! Seed atom planning and analytic padded-stack construction (issue #2236).
//!
//! This module owns every policy decision that turns typed atom topology,
//! dimension, and deterministic seed metadata into basis evaluators, Duchon
//! centers, smoothness penalties, and the padded arrays consumed by a manifold
//! fit. Python bindings only marshal arrays and call these entries.

use super::*;
use gam_terms::basis::duchon_nullspace_dimension;

/// Per-atom basis spec used by [`sae_build_padded_basis_stacks`] to assemble the
/// padded `(K, N, M_max)` design plus jacobian, smoothness penalty stack, and
/// per-atom `basis_sizes`. The Python wrapper passes only `(z, atom_basis,
/// atom_dim)`; this Rust helper picks `n_harmonics` for periodic atoms and
/// samples Duchon centers deterministically from the PCA seed.
#[derive(Debug, Clone)]
pub struct SaeAtomBuildPlan {
    pub geometry: SaeAtomGeometryPlan,
}

impl SaeAtomBuildPlan {
    pub fn kind(&self) -> &SaeAtomBasisKind {
        self.geometry.kind()
    }

    pub fn latent_dim(&self) -> usize {
        self.geometry.latent_dim()
    }

    pub fn basis_size(&self) -> Result<usize, String> {
        self.geometry.basis_size()
    }

    pub fn duchon_centers(&self) -> Option<&Array2<f64>> {
        self.geometry.duchon_centers()
    }
}

/// Deterministically pick Duchon centers from the PCA-seeded coordinates.
/// Uses a Lehmer (LCG) walk over `0..n_obs` keyed by `random_state` so the
/// result is reproducible without a heavy RNG dependency.
pub fn sae_pick_duchon_center_indices(
    n_obs: usize,
    n_centers: usize,
    random_state: u64,
) -> Vec<usize> {
    let want = n_centers.min(n_obs);
    if want == 0 || n_obs == 0 {
        return Vec::new();
    }
    if want >= n_obs {
        return (0..n_obs).collect();
    }
    let mut chosen: Vec<usize> = (0..n_obs).collect();
    let mut state = random_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..n_obs).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        chosen.swap(i, j);
    }
    chosen.truncate(want);
    chosen.sort_unstable();
    chosen
}

/// Build the padded `(phi_stack, jet_stack, penalty_stack)` arrays plus
/// per-atom `basis_sizes` for the given atom plans and seed coords. Returns
/// `(basis_values, basis_jacobian, smooth_penalties, basis_sizes, coord_blocks)`.
pub fn sae_build_padded_basis_stacks(
    plans: &[SaeAtomBuildPlan],
    seed_coords: ArrayView3<'_, f64>,
    n_obs: usize,
) -> Result<
    (
        Array3<f64>,
        Array4<f64>,
        Array3<f64>,
        Vec<usize>,
        Vec<Array2<f64>>,
    ),
    String,
> {
    let k_atoms = plans.len();
    let seed_shape = seed_coords.shape();
    if seed_shape[0] != k_atoms || seed_shape[1] < n_obs {
        return Err(format!(
            "sae_build_padded_basis_stacks: seed_coords must have shape (K, N_seed, D_max) with K={k_atoms} and N_seed >= {n_obs}; got {:?}",
            seed_shape
        ));
    }
    for (atom_idx, plan) in plans.iter().enumerate() {
        if plan.latent_dim() > seed_shape[2] {
            return Err(format!(
                "sae_build_padded_basis_stacks: atom {atom_idx} latent_dim {} exceeds seed_coords D_max={}",
                plan.latent_dim(), seed_shape[2]
            ));
        }
    }
    let basis_sizes: Vec<usize> = plans
        .iter()
        .map(SaeAtomBuildPlan::basis_size)
        .collect::<Result<_, _>>()?;
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    let d_max = plans
        .iter()
        .map(SaeAtomBuildPlan::latent_dim)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut phi_stack = Array3::<f64>::zeros((k_atoms, n_obs, m_max));
    let mut jet_stack = Array4::<f64>::zeros((k_atoms, n_obs, m_max, d_max));
    let mut penalty_stack = Array3::<f64>::zeros((k_atoms, m_max, m_max));
    let mut coord_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    for (atom_idx, plan) in plans.iter().enumerate() {
        let d = plan.latent_dim();
        let coords = seed_coords.slice(s![atom_idx, 0..n_obs, 0..d]).to_owned();
        match plan.kind() {
            SaeAtomBasisKind::Periodic => {
                let t = if d >= 1 {
                    coords.column(0).to_owned()
                } else {
                    Array1::<f64>::zeros(n_obs)
                };
                let SaeBasisResolution::PeriodicHarmonics { order } = plan.geometry.resolution()
                else {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} periodic kind has non-periodic resolution {:?}",
                        plan.geometry.resolution()
                    ));
                };
                let (phi, jet, penalty) = sae_build_periodic_atom(t.view(), *order)?;
                let m = phi.ncols();
                if phi.nrows() != n_obs || m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} periodic basis shape {:?} disagrees with N={n_obs}, declared M={}",
                        phi.dim(),
                        basis_sizes[atom_idx]
                    ));
                }
                if jet.shape() != &[n_obs, m, 1] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} periodic jet shape {:?} disagrees with expected ({n_obs}, {m}, 1)",
                        jet.shape()
                    ));
                }
                if penalty.dim() != (m, m) {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} periodic penalty shape {:?} disagrees with M={m}",
                        penalty.dim()
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Sphere => {
                if d != 2 {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Sphere requires latent_dim == 2, got {d}"
                    ));
                }
                let (phi, jet, penalty) = sae_build_sphere_atom(coords.view())?;
                let m = phi.ncols();
                if m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Sphere basis size {m} disagrees with declared M={}",
                        basis_sizes[atom_idx]
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Torus => {
                let SaeBasisResolution::TorusHarmonics { per_axis_order } =
                    plan.geometry.resolution()
                else {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} torus kind has non-torus resolution {:?}",
                        plan.geometry.resolution()
                    ));
                };
                let h = *per_axis_order;
                let (phi, jet, penalty) = sae_build_torus_atom(coords.view(), d, h)?;
                let m = phi.ncols();
                if m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Torus basis size {m} disagrees with declared M={}",
                        basis_sizes[atom_idx]
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::ProjectivePlane | SaeAtomBasisKind::KleinBottle => {
                let evaluator = plan
                    .geometry
                    .build_evaluator()?;
                let (phi, jet) = evaluator.evaluate(coords.view())?;
                let penalty = plan
                    .geometry
                    .quotient_spectral_penalty(2)?
                    .ok_or_else(|| {
                        format!(
                            "sae_build_padded_basis_stacks: quotient atom {atom_idx} has no exact spectral penalty"
                        )
                    })?;
                let m = plan.basis_size()?;
                if phi.dim() != (n_obs, m)
                    || jet.dim() != (n_obs, m, d)
                    || penalty.dim() != (m, m)
                {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: quotient atom {atom_idx} rebuilt shapes phi={:?}, jet={:?}, penalty={:?}, expected ({n_obs}, {m}, {d})",
                        phi.dim(),
                        jet.dim(),
                        penalty.dim()
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..d])
                    .assign(&jet);
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Duchon => {
                let centers = plan.duchon_centers().ok_or_else(|| {
                    format!(
                        "sae_build_padded_basis_stacks: Duchon atom {atom_idx} has no centers in its resolution plan"
                    )
                })?;
                let (phi, jet, penalty) = sae_build_duchon_atom(coords.view(), centers.view())?;
                let m = phi.ncols();
                if phi.nrows() != n_obs || m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Duchon basis shape {:?} disagrees with N={n_obs}, declared M={}",
                        phi.dim(),
                        basis_sizes[atom_idx]
                    ));
                }
                if jet.shape() != &[n_obs, m, d] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Duchon jet shape {:?} disagrees with expected ({n_obs}, {m}, {d})",
                        jet.shape()
                    ));
                }
                if penalty.dim() != (m, m) {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Duchon penalty shape {:?} disagrees with M={m}",
                        penalty.dim()
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                let SaeBasisResolution::Polynomial { degree } = plan.geometry.resolution() else {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: polynomial atom {atom_idx} has incompatible resolution {:?}",
                        plan.geometry.resolution()
                    ));
                };
                let (phi, jet, penalty) =
                    sae_build_euclidean_atom_with_degree(coords.view(), d, *degree)?;
                let m = phi.ncols();
                if phi.dim() != (n_obs, basis_sizes[atom_idx])
                    || jet.dim() != (n_obs, m, d)
                    || penalty.dim() != (m, m)
                {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: polynomial atom {atom_idx} rebuilt shapes phi={:?}, jet={:?}, penalty={:?}, expected ({n_obs}, {m}, {d})",
                        phi.dim(),
                        jet.dim(),
                        penalty.dim()
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..d])
                    .assign(&jet);
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Mobius => {
                // Möbius band (#2240): the deck-invariant double-cover basis is
                // fully analytic, so the stack is built straight off the
                // evaluator (design + jet) with its closed-form roughness Gram.
                let evaluator = MobiusHarmonicEvaluator::new(
                    SAE_MOBIUS_CIRCLE_HARMONICS,
                    SAE_MOBIUS_WIDTH_DEGREE,
                )?;
                let (phi, jet) = evaluator.evaluate(coords.view())?;
                let penalty = evaluator.roughness_gram();
                let m = phi.ncols();
                if m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Mobius basis size {m} disagrees with declared M={}",
                        basis_sizes[atom_idx]
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Cylinder => {
                // Cylinder is a birth-discovered topology, never built through the
                // seed-plan stack path (`sae_build_atom_plans` rejects it above), so
                // it cannot reach here from a `SaeAtomBuildPlan`. Surfaced loudly if
                // it ever does, rather than mis-built.
                return Err(format!(
                    "sae_build_padded_basis_stacks: atom {atom_idx} 'cylinder' is a birth-discovered \
                     topology, not a seed-plan stack kind; it has no padded seed basis to build"
                ));
            }
            SaeAtomBasisKind::FiniteSet => {
                // The finite-set candidate is inert scaffolding not enrolled in the
                // topology race by default, and its latent is categorical (a
                // discrete anchor index) rather than a continuous seed coordinate,
                // so there is no padded seed basis to lay down here. Rejected above
                // in `sae_build_atom_plans`, so it cannot reach this stack builder;
                // surfaced loudly if it ever does, rather than mis-built.
                return Err(format!(
                    "sae_build_padded_basis_stacks: atom {atom_idx} 'finite_set' is a discrete-anchor \
                     (categorical) candidate, not a continuous seed-plan stack kind; it has no padded \
                     seed basis to build"
                ));
            }
            SaeAtomBasisKind::Precomputed(name) => {
                return Err(format!(
                    "sae_build_padded_basis_stacks: unsupported atom {atom_idx} basis {:?}; precomputed atoms require caller-supplied padded basis arrays",
                    name
                ));
            }
        }
        coord_blocks.push(coords);
    }
    Ok((
        phi_stack,
        jet_stack,
        penalty_stack,
        basis_sizes,
        coord_blocks,
    ))
}

/// Build [`SaeAtomBuildPlan`]s from `(z, atom_basis, atom_dim)` + per-atom
/// PCA seed. Periodic atoms get `n_harmonics = max(1, d_atom)`; Duchon atoms
/// get deterministic center indices from the PCA seed.
///
/// `resolution_overrides` (aligned with `atom_basis`) carries the evidence-
/// selected basis-native resolution for an auto-discovery winner
/// (`resolve_auto_primary_atoms`), interpreted per the atom's resolved basis
/// kind: the thin-plate center count for a #2240 Duchon-sheet winner (clamped to
/// the identifiability floor and to `n_obs`), or the per-axis harmonic order for
/// a #2243 torus winner (clamped to the dense guard). `None` entries keep the
/// fixed economy budget below.
pub fn sae_build_atom_plans(
    z: ArrayView2<'_, f64>,
    atom_basis: &[String],
    atom_dim: &[usize],
    seed_coords: ArrayView3<'_, f64>,
    random_state: u64,
    resolution_overrides: &[Option<usize>],
) -> Result<Vec<SaeAtomBuildPlan>, String> {
    let k_atoms = atom_basis.len();
    let n_obs = z.nrows();
    let seed_shape = seed_coords.shape();
    if atom_dim.len() != k_atoms {
        return Err(format!(
            "sae_build_atom_plans: atom_dim length {} must equal atom_basis length {k_atoms}",
            atom_dim.len()
        ));
    }
    if resolution_overrides.len() != k_atoms {
        return Err(format!(
            "sae_build_atom_plans: resolution_overrides length {} must equal atom_basis length {k_atoms}",
            resolution_overrides.len()
        ));
    }
    if seed_shape[0] != k_atoms || seed_shape[1] != n_obs {
        return Err(format!(
            "sae_build_atom_plans: seed_coords must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            seed_shape
        ));
    }
    let mut plans: Vec<SaeAtomBuildPlan> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let kind = sae_atom_basis_kind_from_str(&atom_basis[atom_idx]);
        let d = atom_dim[atom_idx];
        if d == 0 {
            return Err(format!(
                "sae_build_atom_plans: atom_dim[{atom_idx}] must be positive"
            ));
        }
        if d > seed_shape[2] {
            return Err(format!(
                "sae_build_atom_plans: atom_dim[{atom_idx}]={d} exceeds seed_coords D_max={}",
                seed_shape[2]
            ));
        }
        match &kind {
            SaeAtomBasisKind::Periodic => {
                // A periodic atom parameterises a circle, which is intrinsically
                // 1-dimensional: the latent coordinate is a single phase angle
                // `t ∈ [0, 1)`. The user-facing `atom_dim` (a.k.a. `d_atom`) for
                // a periodic basis selects the *number of harmonics* in the
                // truncated Fourier expansion (basis size `2·n_harmonics + 1`),
                // not a latent-space dimensionality. Setting
                // `latent_dim = atom_dim` would make
                // `build_sae_basis_evaluators` reject the atom (the analytic
                // `PeriodicHarmonicEvaluator` requires `latent_dim == 1`),
                // since there is no longer a frozen-snapshot fallback. Bind the
                // optimizer-visible latent dimension to 1 and route the user's
                // `d_atom` into the harmonic count.
                let n_harmonics = d.max(1);
                plans.push(SaeAtomBuildPlan {
                    geometry: SaeAtomGeometryPlan::new(
                        SaeAtomBasisKind::Periodic,
                        1,
                        SaeBasisResolution::PeriodicHarmonics {
                            order: n_harmonics,
                        },
                        SaeReferenceMetricPlan::UnitCircle,
                    )?,
                });
            }
            SaeAtomBasisKind::Sphere => {
                // The (lat, lon) chart fixes latent_dim = 2 and basis_size = 7
                // regardless of the user-supplied `atom_dim` — the chart
                // already captures the embedded S² geometry. Reject any
                // d_atom other than 2 to keep the contract honest.
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'sphere' requires atom_dim == 2, got {d}"
                    ));
                }
                plans.push(SaeAtomBuildPlan {
                    geometry: SaeAtomGeometryPlan::new(
                        SaeAtomBasisKind::Sphere,
                        2,
                        SaeBasisResolution::SphereChart,
                        SaeReferenceMetricPlan::SphereChart,
                    )?,
                });
            }
            SaeAtomBasisKind::Torus => {
                // Torus of dim `d` uses a tensor-product periodic harmonic
                // basis of size `(2H+1)^d`. The user's `atom_dim` selects
                // the latent dimension; the per-axis order `H` defaults to
                // `SAE_DEFAULT_TORUS_HARMONICS` but a #2243 auto-discovery
                // winner carries its evidence-selected order in the resolution
                // override (which the selector already bounds by the same dense
                // guard checked below). The design grows exponentially in `d`,
                // so reject runaway combinations.
                let h = resolution_overrides[atom_idx]
                    .map(|selected| selected.max(1))
                    .unwrap_or(SAE_DEFAULT_TORUS_HARMONICS);
                let geometry = SaeAtomGeometryPlan::new(
                    SaeAtomBasisKind::Torus,
                    d,
                    SaeBasisResolution::TorusHarmonics { per_axis_order: h },
                    SaeReferenceMetricPlan::FlatRectangularTorus { tau: 0.0 },
                )?;
                let basis_size = geometry.basis_size()?;
                if basis_size > SAE_MAX_PERIODIC_HARMONICS * 4 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} torus basis size {basis_size} = (2*{h}+1)^{d} exceeds the dense limit; reduce atom_dim or harmonics"
                    ));
                }
                plans.push(SaeAtomBuildPlan { geometry });
            }
            SaeAtomBasisKind::ProjectivePlane => {
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'projective_plane' requires atom_dim == 2, got {d}"
                    ));
                }
                let quotient_order = resolution_overrides[atom_idx]
                    .map(|order| order.max(1))
                    .unwrap_or(1);
                plans.push(SaeAtomBuildPlan {
                    geometry: SaeAtomGeometryPlan::projective_plane(quotient_order)?,
                });
            }
            SaeAtomBasisKind::KleinBottle => {
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'klein_bottle' requires atom_dim == 2, got {d}"
                    ));
                }
                let per_axis_order = resolution_overrides[atom_idx]
                    .map(|order| order.max(2))
                    .unwrap_or(2);
                plans.push(SaeAtomBuildPlan {
                    geometry: SaeAtomGeometryPlan::klein_bottle(per_axis_order)?,
                });
            }
            SaeAtomBasisKind::Duchon => {
                // A Duchon atom's curvature penalty degrades (and ultimately
                // fails its D2 collocation) when the center count does not
                // exceed the polynomial nullspace dimension of its resolved
                // order. Pick enough centers to clear that dimension with a
                // margin (so a positive-rank kernel block survives), bounded
                // above by `n_obs` and the dense cap. The Euclidean patch
                // ignores centers, so this lower bound is harmless there.
                let duchon_m = sae_duchon_atom_m(d);
                let poly_nullspace_dim = duchon_nullspace_dimension(d, duchon_m.saturating_sub(1));
                let center_floor = (poly_nullspace_dim + d + 1).max(8);
                let center_ceiling = center_floor.max(32);
                let lo = center_floor.min(n_obs);
                let hi = center_ceiling.min(n_obs);
                // #2240 — a Duchon-sheet discovery winner carries its
                // evidence-selected center count; honor it (clamped to the
                // identifiability floor and the row count) instead of the
                // fixed economy ceiling.
                let n_centers = match resolution_overrides[atom_idx] {
                    Some(selected) => selected.min(n_obs).max(lo),
                    None => n_obs.min(hi).max(lo),
                };
                let idx = sae_pick_duchon_center_indices(
                    n_obs,
                    n_centers,
                    random_state.wrapping_add(atom_idx as u64),
                );
                let mut centers = Array2::<f64>::zeros((idx.len(), d));
                for (out_row, src_row) in idx.iter().copied().enumerate() {
                    for col in 0..d {
                        centers[[out_row, col]] = seed_coords[[atom_idx, src_row, col]];
                    }
                }
                plans.push(SaeAtomBuildPlan {
                    geometry: SaeAtomGeometryPlan::new(
                        SaeAtomBasisKind::Duchon,
                        d,
                        SaeBasisResolution::DuchonCoordinates { centers },
                        SaeReferenceMetricPlan::EuclideanDuchon,
                    )?,
                });
            }
            SaeAtomBasisKind::Linear => plans.push(SaeAtomBuildPlan {
                geometry: SaeAtomGeometryPlan::new(
                    SaeAtomBasisKind::Linear,
                    d,
                    SaeBasisResolution::Polynomial { degree: 1 },
                    SaeReferenceMetricPlan::EuclideanPolynomial,
                )?,
            }),
            SaeAtomBasisKind::EuclideanPatch => plans.push(SaeAtomBuildPlan {
                geometry: SaeAtomGeometryPlan::new(
                    SaeAtomBasisKind::EuclideanPatch,
                    d,
                    SaeBasisResolution::Polynomial {
                        degree: SAE_EUCLIDEAN_PATCH_MAX_DEGREE,
                    },
                    SaeReferenceMetricPlan::EuclideanPolynomial,
                )?,
            }),
            SaeAtomBasisKind::Poincare => plans.push(SaeAtomBuildPlan {
                geometry: SaeAtomGeometryPlan::new(
                    SaeAtomBasisKind::Poincare,
                    d,
                    SaeBasisResolution::Polynomial {
                        degree: SAE_EUCLIDEAN_PATCH_MAX_DEGREE,
                    },
                    SaeReferenceMetricPlan::UnitPoincareBall,
                )?,
            }),
            SaeAtomBasisKind::Mobius => {
                // Möbius band (#2240) is a first-class SEEDABLE kind: the
                // deck-invariant double-cover layout is fixed by the production
                // convention, so the plan needs no centers — just the width.
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'mobius' requires atom_dim == 2, got {d}"
                    ));
                }
                plans.push(SaeAtomBuildPlan {
                    geometry: SaeAtomGeometryPlan::new(
                        SaeAtomBasisKind::Mobius,
                        2,
                        SaeBasisResolution::MobiusHarmonics {
                            circle_order: SAE_MOBIUS_CIRCLE_HARMONICS,
                            width_degree: SAE_MOBIUS_WIDTH_DEGREE,
                        },
                        SaeReferenceMetricPlan::MobiusQuotient,
                    )?,
                });
            }
            SaeAtomBasisKind::Cylinder => {
                // A cylinder atom is not SEEDED through `sae_manifold_fit_minimal`:
                // it arises only by EVIDENCE, when the #977 birth topology race
                // selects `S¹ × ℝ` for a residual factor (the born atom's evaluator
                // is built directly by `race_birth_topology`, and OOS refresh reads
                // it back through `build_sae_basis_evaluators`). There is no
                // user-facing cylinder seed geometry to derive a plan from here, so
                // a cylinder in the seed dictionary is a caller error, surfaced
                // loudly rather than mis-built as a torus / patch.
                return Err(
                    "sae_build_atom_plans: 'cylinder' is a birth-discovered topology, not a seed \
                     dictionary kind; seed with periodic, duchon, sphere, torus, or \
                     euclidean_patch and let the structure search grow a cylinder by evidence"
                        .to_string(),
                );
            }
            SaeAtomBasisKind::FiniteSet => {
                // The finite-set (discrete-anchor) candidate is inert scaffolding
                // not enrolled in the topology race by default
                // (`structure_harvest::finite_set_race_enrolled` is false), and its
                // latent is CATEGORICAL rather than a continuous seed coordinate, so
                // there is no user-facing finite-set seed geometry to derive a plan
                // from here. First-class integration into the continuous-latent
                // optimizer is the documented follow-up; until then a finite_set in
                // the seed dictionary is a caller error, surfaced loudly rather than
                // mis-built as a patch.
                return Err(
                    "sae_build_atom_plans: 'finite_set' is a discrete-anchor (categorical) \
                     candidate that is not enrolled in the topology race and cannot be seeded \
                     through sae_manifold_fit_minimal; seed with periodic, duchon, sphere, \
                     torus, or euclidean_patch"
                        .to_string(),
                );
            }
            SaeAtomBasisKind::Precomputed(name) => {
                return Err(format!(
                    "sae_build_atom_plans: unsupported atom_basis {:?}; sae_manifold_fit_minimal can build only periodic, duchon, sphere, torus, or euclidean_patch atoms",
                    name
                ));
            }
        }
    }
    Ok(plans)
}

/// Persistable basis metadata derived from one converged fitted atom.
#[derive(Clone, Debug)]
pub struct SaeFittedAtomPlan {
    pub kind: SaeAtomBasisKind,
    pub latent_dim: usize,
    pub n_harmonics: usize,
    pub basis_size: usize,
    pub duchon_centers: Option<Array2<f64>>,
}

/// Derive the complete OOS rebuild metadata from the converged dictionary.
///
/// This is model logic, so it lives beside native atom construction rather
/// than in a language binding. Invalid harmonic widths and missing Duchon
/// centers are errors; no metadata is guessed at the FFI boundary.
pub fn sae_fitted_atom_plans(
    term: &SaeManifoldTerm,
    seed_duchon_centers: &[Option<Array2<f64>>],
    random_state: u64,
) -> Result<Vec<SaeFittedAtomPlan>, String> {
    let mut plans = Vec::with_capacity(term.k_atoms());
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let kind = atom.basis_kind.clone();
        let latent_dim = atom.latent_dim;
        let basis_size = atom.full_basis_size();
        let n_harmonics = match &kind {
            SaeAtomBasisKind::Periodic => {
                if basis_size < 3 || basis_size % 2 == 0 {
                    return Err(format!(
                        "sae_fitted_atom_plans: periodic atom {atom_idx} has invalid odd basis width {basis_size}"
                    ));
                }
                (basis_size - 1) / 2
            }
            SaeAtomBasisKind::Torus => {
                let axis_size = sae_torus_axis_basis_size(basis_size, latent_dim)?;
                (axis_size - 1) / 2
            }
            SaeAtomBasisKind::Cylinder => sae_cylinder_harmonics_degree(basis_size)?.0,
            SaeAtomBasisKind::Mobius => SAE_MOBIUS_CIRCLE_HARMONICS,
            _ => 0,
        };
        let duchon_centers = match &kind {
            SaeAtomBasisKind::Duchon => Some(
                seed_duchon_centers
                    .get(atom_idx)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        format!("sae_fitted_atom_plans: Duchon atom {atom_idx} has no seed centers")
                    })?
                    .clone(),
            ),
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                let coords = term.assignment.coords[atom_idx].as_matrix();
                if coords.nrows() == 0 || coords.ncols() < latent_dim {
                    return Err(format!(
                        "sae_fitted_atom_plans: atom {atom_idx} has coordinate shape {:?}, expected non-empty rows and at least {latent_dim} columns",
                        coords.dim()
                    ));
                }
                let center_count = coords.nrows().min((latent_dim + 2).max(8));
                let indices = sae_pick_duchon_center_indices(
                    coords.nrows(),
                    center_count,
                    random_state.wrapping_add(atom_idx as u64),
                );
                let mut centers = Array2::<f64>::zeros((indices.len(), latent_dim));
                for (out_row, src_row) in indices.into_iter().enumerate() {
                    for col in 0..latent_dim {
                        centers[[out_row, col]] = coords[[src_row, col]];
                    }
                }
                Some(centers)
            }
            _ => None,
        };
        plans.push(SaeFittedAtomPlan {
            kind,
            latent_dim,
            n_harmonics,
            basis_size,
            duchon_centers,
        });
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duchon_center_picker_is_seeded_sorted_and_unique() {
        let first = sae_pick_duchon_center_indices(64, 12, 7);
        let again = sae_pick_duchon_center_indices(64, 12, 7);
        let other = sae_pick_duchon_center_indices(64, 12, 8);

        assert_eq!(first, again);
        assert_ne!(first, other);
        assert_eq!(first.len(), 12);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(first.iter().all(|&row| row < 64));
    }

    #[test]
    fn periodic_plan_and_padded_stack_share_one_typed_entry() {
        let z = Array2::<f64>::zeros((4, 3));
        let mut seed_coords = Array3::<f64>::zeros((1, 4, 2));
        seed_coords[[0, 1, 0]] = 0.25;
        seed_coords[[0, 2, 0]] = 0.5;
        seed_coords[[0, 3, 0]] = 0.75;
        let plans = sae_build_atom_plans(
            z.view(),
            &["periodic".to_string()],
            &[2],
            seed_coords.view(),
            11,
            &[None],
        )
        .expect("periodic plan");

        assert_eq!(plans.len(), 1);
        assert!(matches!(plans[0].kind, SaeAtomBasisKind::Periodic));
        assert_eq!(plans[0].latent_dim, 1);
        assert_eq!(plans[0].n_harmonics, 2);
        assert_eq!(plans[0].basis_size, 5);

        let (phi, jet, penalty, basis_sizes, coords) =
            sae_build_padded_basis_stacks(&plans, seed_coords.view(), 4)
                .expect("periodic padded stack");
        assert_eq!(phi.dim(), (1, 4, 5));
        assert_eq!(jet.dim(), (1, 4, 5, 1));
        assert_eq!(penalty.dim(), (1, 5, 5));
        assert_eq!(basis_sizes, vec![5]);
        assert_eq!(coords[0].dim(), (4, 1));
        assert!(phi.slice(s![0, .., 0]).iter().all(|&value| value == 1.0));
        assert_eq!(penalty[[0, 0, 0]], 1.0e-8);
        assert_eq!(penalty[[0, 3, 3]], 16.0);
        assert_eq!(penalty[[0, 4, 4]], 16.0);
    }
}
