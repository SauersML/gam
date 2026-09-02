//! Padded-FFI term builder, split out of `construction.rs` to keep that tracked
//! file under the #780 10k-line gate. `term_from_padded_blocks_with_mode` is the
//! single entry point Python callers use to assemble a [`SaeManifoldTerm`] from
//! `(K, N, M_max[, D_max])`-padded arrays; it is re-exported from `mod.rs` via
//! `pub use construction_padded_blocks::*;` so every caller keeps reaching it
//! bare through `use super::*`.

use super::*;

/// Construct a native analytic term from immutable geometry plans.
///
/// Unlike the caller-managed padded-array entry below, every active width,
/// latent dimension, topology, evaluator, and reference-metric identity is
/// derived from `geometry_plans`. The padded arrays are observations of those
/// plans and are validated against them; they are never metadata authorities.
#[must_use = "build error must be handled"]
pub fn term_from_geometry_plans_with_mode(
    n_obs: usize,
    p_out: usize,
    geometry_plans: &[SaeAtomGeometryPlan],
    basis_values: ArrayView3<'_, f64>,
    basis_jacobian: ArrayView4<'_, f64>,
    decoder_coefficients: ArrayView3<'_, f64>,
    smooth_penalties: ArrayView3<'_, f64>,
    logits: ArrayView2<'_, f64>,
    coords: &[Array2<f64>],
    mode: AssignmentMode,
) -> Result<SaeManifoldTerm, String> {
    let k_atoms = geometry_plans.len();
    if k_atoms == 0 || coords.len() != k_atoms {
        return Err(format!(
            "term_from_geometry_plans_with_mode: requires non-empty aligned plans/coords; got plans={k_atoms}, coords={}",
            coords.len()
        ));
    }
    if logits.dim() != (n_obs, k_atoms) {
        return Err(format!(
            "term_from_geometry_plans_with_mode: logits must be ({n_obs}, {k_atoms}); got {:?}",
            logits.dim()
        ));
    }
    let values_shape = basis_values.shape();
    let jacobian_shape = basis_jacobian.shape();
    let decoder_shape = decoder_coefficients.shape();
    let penalty_shape = smooth_penalties.shape();
    if values_shape[0] != k_atoms
        || values_shape[1] != n_obs
        || jacobian_shape[0] != k_atoms
        || jacobian_shape[1] != n_obs
        || decoder_shape[0] != k_atoms
        || decoder_shape[2] != p_out
        || penalty_shape[0] != k_atoms
        || penalty_shape[1] != penalty_shape[2]
    {
        return Err(format!(
            "term_from_geometry_plans_with_mode: padded shapes values={values_shape:?}, jacobian={jacobian_shape:?}, decoder={decoder_shape:?}, penalty={penalty_shape:?} disagree with K={k_atoms}, N={n_obs}, p={p_out}"
        ));
    }

    let mut atoms = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for (atom_index, plan) in geometry_plans.iter().enumerate() {
        let d = plan.latent_dim();
        let m = plan.basis_size()?;
        if m > values_shape[2]
            || m > jacobian_shape[2]
            || d > jacobian_shape[3]
            || m > decoder_shape[1]
            || m > penalty_shape[1]
        {
            return Err(format!(
                "term_from_geometry_plans_with_mode: atom {atom_index} plan (M={m}, d={d}) exceeds padded shapes values={values_shape:?}, jacobian={jacobian_shape:?}, decoder={decoder_shape:?}, penalty={penalty_shape:?}"
            ));
        }
        if coords[atom_index].dim() != (n_obs, d) {
            return Err(format!(
                "term_from_geometry_plans_with_mode: atom {atom_index} coordinates {:?}, expected ({n_obs}, {d})",
                coords[atom_index].dim()
            ));
        }
        let phi = basis_values
            .slice(s![atom_index, 0..n_obs, 0..m])
            .to_owned();
        let jet = basis_jacobian
            .slice(s![atom_index, 0..n_obs, 0..m, 0..d])
            .to_owned();
        let decoder = decoder_coefficients
            .slice(s![atom_index, 0..m, 0..p_out])
            .to_owned();
        let reference_penalty = smooth_penalties
            .slice(s![atom_index, 0..m, 0..m])
            .to_owned();
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("atom_{atom_index}"),
            plan.kind().clone(),
            d,
            phi,
            jet,
            decoder,
            reference_penalty,
        )?
        .with_basis_second_jet(plan.build_evaluator()?)
        .with_geometry_plan(plan.clone())?;
        manifolds.push(plan.kind().latent_manifold(d));
        atoms.push(atom);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.to_owned(),
        coords.to_vec(),
        manifolds,
        mode,
    )?;
    SaeManifoldTerm::new(atoms, assignment)
}

