//! Padded-FFI term builder, split out of `construction.rs` to keep that tracked
//! file under the #780 10k-line gate. `term_from_padded_blocks_with_mode` is the
//! single entry point Python callers use to assemble a [`SaeManifoldTerm`] from
//! `(K, N, M_max[, D_max])`-padded arrays; it is re-exported from `mod.rs` via
//! `pub use construction_padded_blocks::*;` so every caller keeps reaching it
//! bare through `use super::*`.

use super::*;

/// Helper for padded FFI callers. Arrays use `(K, N, M_max)` and
/// `(K, N, M_max, D_max)` storage, with `basis_sizes` and `latent_dims`
/// selecting each atom's active prefix.
///
/// `evaluators`, when non-empty, must have length `K`. Each entry attaches an
/// optional [`SaeBasisSecondJet`] to the matching atom so the Rust Newton
/// loop can refresh `Phi`/`dPhi/dt` between iterations without rebuilding the
/// term from Python. The evaluator is installed through
/// [`SaeManifoldAtom::with_basis_second_jet`], so its closed-form Hessian slot
/// is populated too — this is what lets the #1117 rank-revealing reduction
/// (`reduce_atoms_to_data_supported_rank`) reparametrize a rank-deficient
/// fixed-width decoder (e.g. the periodic circle's 5-column basis whose data
/// Gram comes out rank 3/5 on a near-degenerate checkpoint) onto its
/// data-supported subspace instead of stalling on the flat REML valley. An
/// empty slice leaves every atom in snapshot-only mode.
#[must_use = "build error must be handled"]
pub fn term_from_padded_blocks_with_mode(
    n_obs: usize,
    p_out: usize,
    basis_kinds: &[SaeAtomBasisKind],
    basis_values: ArrayView3<'_, f64>,
    basis_jacobian: ArrayView4<'_, f64>,
    basis_sizes: &[usize],
    latent_dims: &[usize],
    decoder_coefficients: ArrayView3<'_, f64>,
    smooth_penalties: ArrayView3<'_, f64>,
    logits: ArrayView2<'_, f64>,
    coords: &[Array2<f64>],
    mode: AssignmentMode,
    evaluators: &[Option<Arc<dyn SaeBasisSecondJet>>],
) -> Result<SaeManifoldTerm, String> {
    let k_atoms = basis_sizes.len();
    if latent_dims.len() != k_atoms || basis_kinds.len() != k_atoms || coords.len() != k_atoms {
        return Err("term_from_padded_blocks: K-length metadata mismatch".into());
    }
    if !evaluators.is_empty() && evaluators.len() != k_atoms {
        return Err(format!(
            "term_from_padded_blocks: evaluators length {} must equal K={k_atoms} or be empty",
            evaluators.len()
        ));
    }
    if logits.dim() != (n_obs, k_atoms) {
        return Err(format!(
            "term_from_padded_blocks: logits must be ({n_obs}, {k_atoms}); got {:?}",
            logits.dim()
        ));
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let m = basis_sizes[k];
        let d = latent_dims[k];
        let phi = basis_values.slice(s![k, 0..n_obs, 0..m]).to_owned();
        let jet = basis_jacobian.slice(s![k, 0..n_obs, 0..m, 0..d]).to_owned();
        let b = decoder_coefficients.slice(s![k, 0..m, 0..p_out]).to_owned();
        let s = smooth_penalties.slice(s![k, 0..m, 0..m]).to_owned();
        let atom = SaeManifoldAtom::new(
            format!("atom_{k}"),
            basis_kinds[k].clone(),
            d,
            phi,
            jet,
            b,
            s,
        )?;
        let atom = match evaluators.get(k).and_then(|slot| slot.clone()) {
            // Install through the second-jet slot so the analytic Hessian is
            // available: the #1117 rank-revealing reduction needs it to compose
            // the reduced jets when it reparametrizes a rank-deficient atom onto
            // its data-supported subspace. All production SAE evaluators
            // (periodic/sphere/torus/cylinder/Duchon/Euclidean-patch) implement
            // `SaeBasisSecondJet`, so this is the standard install path.
            Some(evaluator) => atom.with_basis_second_jet(evaluator),
            None => atom,
        };
        // #2022 — the SCALE-gauge quotient SEED peel is applied in the FFI
        // (gated by the typed `quotient_scale` kwarg) after the term is built,
        // not here (this builder has no per-fit flag + must stay env-free).
        atoms.push(atom);
    }
    let manifolds = basis_kinds
        .iter()
        .zip(latent_dims.iter().copied())
        .map(|(kind, d)| kind.latent_manifold(d))
        .collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.to_owned(),
        coords.to_vec(),
        manifolds,
        mode,
    )?;
    SaeManifoldTerm::new(atoms, assignment)
}
