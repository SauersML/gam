//! Seed atom planning and analytic padded-stack construction (issue #2236).
//!
//! This module owns every policy decision that turns typed atom topology,
//! dimension, and deterministic seed metadata into basis evaluators, Duchon
//! centers, smoothness penalties, and the padded arrays consumed by a manifold
//! fit. Python bindings only marshal arrays and call these entries.

use super::*;
use gam_terms::basis::{DuchonNullspaceOrder, duchon_nullspace_dimension, monomial_exponents};

fn duchon_nullspace_from_m(m: usize) -> DuchonNullspaceOrder {
    match m {
        1 => DuchonNullspaceOrder::Zero,
        2 => DuchonNullspaceOrder::Linear,
        other => DuchonNullspaceOrder::Degree(other - 1),
    }
}

fn build_wrapped_periodic_harmonic_basis_with_jet(
    t: ArrayView1<'_, f64>,
    n_harmonics: usize,
    label: &str,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    if t.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} requires finite t values"));
    }

    let n_rows = t.len();
    let n_cols = 1 + 2 * n_harmonics;
    let mut phi = Array2::<f64>::zeros((n_rows, n_cols));
    let mut jet = Array3::<f64>::zeros((n_rows, n_cols, 1));
    let mut penalty = Array2::<f64>::zeros((n_cols, n_cols));

    phi.column_mut(0).fill(1.0);
    penalty[[0, 0]] = 1.0e-8;

    for h in 1..=n_harmonics {
        let h_f = h as f64;
        let frequency = std::f64::consts::TAU * h_f;
        let sin_col = 1 + 2 * (h - 1);
        let cos_col = sin_col + 1;
        let harmonic_penalty = h_f * h_f * h_f * h_f;

        penalty[[sin_col, sin_col]] = harmonic_penalty;
        penalty[[cos_col, cos_col]] = harmonic_penalty;

        for row in 0..n_rows {
            let angle = frequency * t[row].rem_euclid(1.0);
            let sin_value = angle.sin();
            let cos_value = angle.cos();

            phi[[row, sin_col]] = sin_value;
            phi[[row, cos_col]] = cos_value;
            jet[[row, sin_col, 0]] = frequency * cos_value;
            jet[[row, cos_col, 0]] = -frequency * sin_value;
        }
    }

    Ok((phi, jet, penalty))
}

/// Per-atom basis spec used by [`sae_build_padded_basis_stacks`] to assemble the
/// padded `(K, N, M_max)` design plus jacobian, smoothness penalty stack, and
/// per-atom `basis_sizes`. The Python wrapper passes only `(z, atom_basis,
/// atom_dim)`; this Rust helper picks `n_harmonics` for periodic atoms and
/// samples Duchon centers deterministically from the PCA seed.
#[derive(Debug, Clone)]
pub struct SaeAtomBuildPlan {
    pub kind: SaeAtomBasisKind,
    pub latent_dim: usize,
    pub n_harmonics: usize,
    pub duchon_centers: Option<Array2<f64>>,
    pub basis_size: usize,
}

fn sae_atom_basis_size(plan: &SaeAtomBuildPlan) -> usize {
    plan.basis_size
}

/// Build (phi, jet, penalty) for a periodic 1-D atom — same math as
/// `periodic_basis_with_jet`, but plain Rust so the helper can be reused by
/// [`sae_build_padded_basis_stacks`] without Python in the loop.
fn sae_build_periodic_atom(
    t: ArrayView1<'_, f64>,
    n_harmonics: usize,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let (phi, jet, penalty) =
        build_wrapped_periodic_harmonic_basis_with_jet(t, n_harmonics, "sae_build_periodic_atom")?;
    let expected_cols = sae_periodic_basis_size(n_harmonics)?;
    if phi.ncols() != expected_cols {
        return Err(format!(
            "sae_build_periodic_atom: basis width {} disagrees with declared width {expected_cols}",
            phi.ncols()
        ));
    }
    Ok((phi, jet, penalty))
}

/// Build (phi, jet, penalty) for a sphere atom via the (lat, lon) chart
/// evaluator. The penalty is identity on the six non-constant basis
/// functions (the constant column gets a 1e-8 floor so the (M,M) block stays
/// strictly positive on the constant subspace).
fn sae_build_sphere_atom(
    coords: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let (phi, jet) = SphereChartEvaluator.evaluate(coords)?;
    let m = phi.ncols();
    let mut penalty = Array2::<f64>::zeros((m, m));
    penalty[[0, 0]] = 1.0e-8;
    for i in 1..m {
        penalty[[i, i]] = 1.0;
    }
    Ok((phi, jet, penalty))
}

/// Build (phi, jet, penalty) for a torus atom via the tensor-product
/// periodic harmonic evaluator. Penalty diagonal encodes the squared
/// Laplace–Beltrami eigenvalue
/// `((2π)^2 · Σ_a h_a^2)^2` so the smoothness term penalises high-frequency
/// modes — same shape as the 1-D periodic harmonic penalty
/// (`(2π·h)^4` reduces to `h^4` up to a constant), generalised to T^d.
fn sae_build_torus_atom(
    coords: ArrayView2<'_, f64>,
    latent_dim: usize,
    num_harmonics: usize,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let evaluator = TorusHarmonicEvaluator::new(latent_dim, num_harmonics)?;
    let (phi, jet) = evaluator.evaluate(coords)?;
    let axis_m = evaluator.axis_basis_size();
    let m = phi.ncols();
    let mut penalty = Array2::<f64>::zeros((m, m));
    // Decode axis index `idx_axis ∈ {0..axis_m}` → harmonic number `h`:
    // 0 → 0 (constant), 1 → 1 (sin), 2 → 1 (cos), 3 → 2, 4 → 2, …
    let axis_harmonic = |idx_axis: usize| -> usize {
        if idx_axis == 0 {
            0
        } else {
            idx_axis.div_ceil(2)
        }
    };
    let mut idx = vec![0usize; latent_dim];
    for flat in 0..m {
        let mut h_sum_sq: usize = 0;
        for axis in 0..latent_dim {
            let h = axis_harmonic(idx[axis]);
            h_sum_sq += h * h;
        }
        let lambda = if h_sum_sq == 0 {
            1.0e-8
        } else {
            (h_sum_sq as f64).powi(2)
        };
        penalty[[flat, flat]] = lambda;
        for axis in (0..latent_dim).rev() {
            idx[axis] += 1;
            if idx[axis] < axis_m {
                break;
            }
            idx[axis] = 0;
        }
    }
    Ok((phi, jet, penalty))
}

/// Build (phi, jet, penalty) for a Duchon atom. Mirrors the pure-Rust path
/// inside `duchon_basis_with_jet` but accepts in-Rust types and returns
/// owned `(Array2, Array3, Array2)`.
fn sae_build_duchon_atom(
    pts: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    // The smoothness penalty is the native reproducing-norm Gram
    // `ω = α²·Zᵀ K_CC Z`, built directly on the SAME `[ (Φ_radial·α)·Z | P ]`
    // columns the `DuchonCoordinateEvaluator` produces (issue #247: the seed
    // must match the refresh evaluator bit-for-bit) and — critically — at the
    // SAME width `m` as `phi`. It is NOT sourced from `build_duchon_basis`: that
    // design path runs the TPRS generalized-eigen reparameterization / near-null
    // mode dropping (#1347), which on coincident/duplicate seed centers (the
    // over-complete large-K regime) emits a penalty NARROWER than `m`, desyncing
    // it from the evaluator's fixed-`m` basis — the #1026 32K Duchon shape bug.
    // `duchon_sae_atom_penalty` keeps all `m` columns; degenerate directions get
    // ~zero penalty (handled by the inner solve's per-row Tikhonov ridge), the
    // matrix itself is the declared Duchon reference-function seminorm used by
    // the atom; no decoder-dependent metric reweighting is applied.
    let dim = centers.ncols();
    let m: usize = sae_duchon_atom_m(dim);
    let penalty = gam_terms::basis::duchon_sae_atom_penalty(centers, duchon_nullspace_from_m(m))
        .map_err(|err| err.to_string())?;
    let evaluator = DuchonCoordinateEvaluator::new(centers.to_owned(), m)?;
    let (phi, jet) = if pts.nrows() == 0 {
        let probe = Array2::<f64>::zeros((1, pts.ncols()));
        let (probe_phi, _probe_jet) = evaluator.evaluate(probe.view())?;
        let cols = probe_phi.ncols();
        (
            Array2::<f64>::zeros((0, cols)),
            Array3::<f64>::zeros((0, cols, dim)),
        )
    } else {
        evaluator.evaluate(pts)?
    };
    if phi.ncols() != jet.shape()[1] {
        return Err(format!(
            "sae_build_duchon_atom: phi/jet column mismatch {} vs {}",
            phi.ncols(),
            jet.shape()[1]
        ));
    }
    Ok((phi, jet, penalty))
}

/// Build (phi, jet, penalty) for a Euclidean tangent-patch atom.
///
/// A Euclidean atom is a *flat* (zero-curvature) polynomial expansion in the
/// atom's latent coordinates — distinct from the thin-plate Duchon kernel.
/// The basis is the set of monomials of total degree ≤ `EUCLIDEAN_PATCH_MAX_DEGREE`,
/// the jet is the first derivative of those monomials, and the penalty is an
/// identity ridge over the non-constant monomials (the constant term is left
/// unpenalized to preserve the affine-equivariance of the patch).
///
/// `centers` is accepted for API symmetry with the Duchon path; its row count
/// determines the random-state matching seam (issue #246), but it is not
/// otherwise used: a polynomial atom has no center-based locality.
fn sae_build_euclidean_atom_with_degree(
    pts: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    max_degree: usize,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let dim = centers.ncols();
    let exponents = monomial_exponents(dim, max_degree);
    let n_basis = exponents.len();
    // The design `Phi` and its jet come from the same evaluator the inner
    // Newton loop refreshes against, so the seed atom and every refresh share
    // one monomial layout.
    let evaluator = EuclideanPatchEvaluator::new(dim, max_degree)?;
    let (phi, jet) = if pts.nrows() == 0 {
        (
            Array2::<f64>::zeros((0, n_basis)),
            Array3::<f64>::zeros((0, n_basis, dim)),
        )
    } else {
        evaluator.evaluate(pts)?
    };
    if jet.shape()[1] != n_basis {
        return Err(format!(
            "sae_build_euclidean_atom: monomial/jet column mismatch {} vs {}",
            n_basis,
            jet.shape()[1]
        ));
    }
    // Identity ridge with the constant term (alpha == zeros) unpenalized so the
    // patch can absorb a global offset without paying a penalty.
    let mut penalty = Array2::<f64>::zeros((n_basis, n_basis));
    for (col, alpha) in exponents.iter().enumerate() {
        let is_constant = alpha.iter().all(|&e| e == 0);
        if !is_constant {
            penalty[[col, col]] = 1.0;
        }
    }
    Ok((phi, jet, penalty))
}

fn sae_build_euclidean_atom(
    pts: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    sae_build_euclidean_atom_with_degree(pts, centers, SAE_EUCLIDEAN_PATCH_MAX_DEGREE)
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
        if plan.latent_dim == 0 {
            return Err(format!(
                "sae_build_padded_basis_stacks: atom {atom_idx} latent_dim must be positive"
            ));
        }
        if plan.latent_dim > seed_shape[2] {
            return Err(format!(
                "sae_build_padded_basis_stacks: atom {atom_idx} latent_dim {} exceeds seed_coords D_max={}",
                plan.latent_dim, seed_shape[2]
            ));
        }
    }
    let basis_sizes: Vec<usize> = plans.iter().map(sae_atom_basis_size).collect();
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    let d_max = plans.iter().map(|p| p.latent_dim).max().unwrap_or(1).max(1);
    let mut phi_stack = Array3::<f64>::zeros((k_atoms, n_obs, m_max));
    let mut jet_stack = Array4::<f64>::zeros((k_atoms, n_obs, m_max, d_max));
    let mut penalty_stack = Array3::<f64>::zeros((k_atoms, m_max, m_max));
    let mut coord_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    for (atom_idx, plan) in plans.iter().enumerate() {
        let d = plan.latent_dim;
        let coords = seed_coords.slice(s![atom_idx, 0..n_obs, 0..d]).to_owned();
        match &plan.kind {
            SaeAtomBasisKind::Periodic => {
                let t = if d >= 1 {
                    coords.column(0).to_owned()
                } else {
                    Array1::<f64>::zeros(n_obs)
                };
                let (phi, jet, penalty) = sae_build_periodic_atom(t.view(), plan.n_harmonics)?;
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
                let h = plan.n_harmonics.max(1);
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
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                let centers = plan
                    .duchon_centers
                    .as_ref()
                    .ok_or_else(|| {
                        format!(
                            "sae_build_padded_basis_stacks: atom {atom_idx} non-periodic atom requires centers"
                        )
                    })?;
                if centers.ncols() != d {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} centers have dim {} but plan latent_dim is {d}",
                        centers.ncols()
                    ));
                }
                let (phi, jet, penalty) = match plan.kind {
                    // #1221 — the linear atom and the euclidean (quadratic) patch
                    // share the monomial evaluator; the polynomial DEGREE is
                    // recovered from the plan's basis width (`d + 1` ⇒ degree 1
                    // linear, the full monomial count ⇒ degree 2 quadratic), so a
                    // genuinely-linear atom builds `{1, t}` and a euclidean atom
                    // builds `{1, t, t²}`.
                    SaeAtomBasisKind::Linear
                    | SaeAtomBasisKind::EuclideanPatch
                    | SaeAtomBasisKind::Poincare => {
                        let degree = sae_euclidean_degree_for_basis_size(d, basis_sizes[atom_idx])?;
                        sae_build_euclidean_atom_with_degree(coords.view(), centers.view(), degree)?
                    }
                    _ => sae_build_duchon_atom(coords.view(), centers.view())?,
                };
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
/// `duchon_center_overrides` (aligned with `atom_basis`) carries the
/// evidence-selected thin-plate center count for a #2240 Duchon-sheet
/// discovery winner (`resolve_auto_primary_atoms`); `None` entries keep the
/// economy budget below. Overrides are clamped to the same identifiability
/// floor and to `n_obs`.
pub fn sae_build_atom_plans(
    z: ArrayView2<'_, f64>,
    atom_basis: &[String],
    atom_dim: &[usize],
    seed_coords: ArrayView3<'_, f64>,
    random_state: u64,
    duchon_center_overrides: &[Option<usize>],
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
    if duchon_center_overrides.len() != k_atoms {
        return Err(format!(
            "sae_build_atom_plans: duchon_center_overrides length {} must equal atom_basis length {k_atoms}",
            duchon_center_overrides.len()
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
                let basis_size = sae_periodic_basis_size(n_harmonics)?;
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Periodic,
                    latent_dim: 1,
                    n_harmonics,
                    duchon_centers: None,
                    basis_size,
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
                    kind: SaeAtomBasisKind::Sphere,
                    latent_dim: 2,
                    n_harmonics: 0,
                    duchon_centers: None,
                    basis_size: SAE_SPHERE_BASIS_SIZE,
                });
            }
            SaeAtomBasisKind::Torus => {
                // Torus of dim `d` uses a tensor-product periodic harmonic
                // basis of size `(2H+1)^d`. The user's `atom_dim` selects
                // the latent dimension; `n_harmonics` defaults to
                // `SAE_DEFAULT_TORUS_HARMONICS`. The design grows
                // exponentially in `d`, so reject runaway combinations.
                let h = SAE_DEFAULT_TORUS_HARMONICS;
                let evaluator = TorusHarmonicEvaluator::new(d, h)?;
                let basis_size = evaluator.basis_size();
                if basis_size > SAE_MAX_PERIODIC_HARMONICS * 4 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} torus basis size {basis_size} = (2*{h}+1)^{d} exceeds the dense limit; reduce atom_dim or harmonics"
                    ));
                }
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Torus,
                    latent_dim: d,
                    n_harmonics: h,
                    duchon_centers: None,
                    basis_size,
                });
            }
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
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
                let n_centers = match duchon_center_overrides[atom_idx] {
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
                // Probe one build to learn the final basis size. The linear atom
                // builds the degree-1 monomial patch `{1, t}` (width `d + 1`); the
                // euclidean (quadratic) patch builds the degree-2 monomial patch
                // (#1221); everything else (Duchon) uses the thin-plate kernel.
                let probe_pts = Array2::<f64>::zeros((1, d));
                let (phi, _jet, _penalty) = match kind {
                    SaeAtomBasisKind::Linear => {
                        sae_build_euclidean_atom_with_degree(probe_pts.view(), centers.view(), 1)?
                    }
                    SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Poincare => {
                        sae_build_euclidean_atom(probe_pts.view(), centers.view())?
                    }
                    _ => sae_build_duchon_atom(probe_pts.view(), centers.view())?,
                };
                let basis_size = phi.ncols();
                plans.push(SaeAtomBuildPlan {
                    kind,
                    latent_dim: d,
                    n_harmonics: 0,
                    duchon_centers: Some(centers),
                    basis_size,
                });
            }
            SaeAtomBasisKind::Mobius => {
                // Möbius band (#2240) is a first-class SEEDABLE kind: the
                // deck-invariant double-cover layout is fixed by the production
                // convention, so the plan needs no centers — just the width.
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'mobius' requires atom_dim == 2, got {d}"
                    ));
                }
                let evaluator = MobiusHarmonicEvaluator::new(
                    SAE_MOBIUS_CIRCLE_HARMONICS,
                    SAE_MOBIUS_WIDTH_DEGREE,
                )?;
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Mobius,
                    latent_dim: 2,
                    n_harmonics: SAE_MOBIUS_CIRCLE_HARMONICS,
                    duchon_centers: None,
                    basis_size: evaluator.basis_size(),
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
