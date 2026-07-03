//! Isometry-penalty cache-refresh leaf helpers, split out of `construction.rs`
//! to keep that tracked file under the #780 10k-line gate. These are the two
//! trailing free functions (`refresh_isometry_caches_from_atom` /
//! `refresh_isometry_caches_from_term`); they are re-exported from `mod.rs` via
//! `pub use construction_cache_refresh::*;` so every caller keeps reaching them
//! bare through `use super::*`.

use super::*;

/// Build the per-row Jacobian `J` and Hessian `H` of the decoded output
/// `Z_n = Phi_n B` with respect to the latent coordinates `t_n` of a single
/// SAE atom and install them on the supplied [`IsometryPenalty`].
///
/// Layout follows the convention used by [`IsometryPenalty::grad_target`] and
/// friends:
///
/// * `J ∈ ℝ^{n_obs × (p · d)}`, flattened as `J[n, i*d + a]` —
///   `J[n, i, a] = ∂Z_{n,i} / ∂t_{n,a} = Σ_m dPhi[n, m, a] · B[m, i]`.
/// * `H ∈ ℝ^{n_obs × (p · d · d)}`, flattened as `H[n, (i*d + a)*d + c]` —
///   `H[n, i, a, c] = ∂J[n, i, a] / ∂t_{n, c} = Σ_m d²Phi[n, m, a, c] · B[m, i]`.
/// * `K`, an `Array3` of shape `(n_obs, p, d·d·d)` with last axis packed
///   `((a·d + c)·d + e)` — `K[n, i, a, c, e] = ∂³Z_{n,i} / ∂t_a ∂t_c ∂t_e =
///   Σ_m d³Phi[n, m, a, c, e] · B[m, i]`. Installed via the new third-jet slot
///   whenever the base evaluator's `third_jet_dyn` yields a jet AND the penalty
///   carries no `duchon_radial_source`. This is the residual-curvature source
///   for the exact isometry `hvp`.
///
/// Returns `Ok(true)` when both caches were installed (i.e. the atom was
/// built via [`SaeManifoldAtom::with_basis_second_jet`], so its
/// `basis_second_jet` slot holds a [`SaeBasisSecondJet`] implementation
/// that supplies the analytic Hessian). Returns `Ok(false)` when only the
/// base [`SaeBasisEvaluator`] is installed (no second jet available) — in
/// that case only the first-jet `jacobian_cache` is installed and the
/// penalty's `has_jacobian_second_source` check still has a chance to
/// succeed via a pre-supplied `duchon_radial_source`. Returns `Err` on
/// shape mismatches (which would indicate a buggy evaluator) or when the
/// second-jet implementation itself fails (e.g. wrong latent dimension).
///
/// This entry point takes `&IsometryPenalty` rather than `&mut` because the
/// caches are interior-mutable (see [`IsometryPenalty::refresh_caches`]).
pub fn refresh_isometry_caches_from_atom(
    penalty: &IsometryPenalty,
    atom: &SaeManifoldAtom,
    coords: ArrayView2<'_, f64>,
) -> Result<bool, String> {
    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!(
            "refresh_isometry_caches_from_atom: atom {} has no basis evaluator",
            atom.name
        )
    })?;
    let (_phi, jet) = evaluator.evaluate(coords)?;

    let n_obs = coords.nrows();
    let d = atom.latent_dim;
    let m = atom.basis_size();
    let p = atom.decoder_coefficients.ncols();
    if penalty.p_out != p {
        return Err(format!(
            "refresh_isometry_caches_from_atom: penalty.p_out={} but atom.decoder.cols={p}",
            penalty.p_out
        ));
    }
    if jet.dim() != (n_obs, m, d) {
        return Err(format!(
            "refresh_isometry_caches_from_atom: evaluator first jet has shape {:?}, expected ({n_obs}, {m}, {d})",
            jet.dim()
        ));
    }

    // J[n, i*d + a] = Σ_m dPhi[n, m, a] · B[m, i]. One (n×m)·(m×p) GEMM per
    // latent axis `a` (jet slice × decoder), scattered into the row-major
    // (n, p, d) layout: the m-contraction is a matmul, not a quadruple scalar
    // loop of bounds-checked element reads (the profiled BLOCKER-1 hot leaf).
    let b = &atom.decoder_coefficients;
    let mut jac3d = ndarray::Array3::<f64>::zeros((n_obs, p, d));
    for a in 0..d {
        let slab = jet.slice(ndarray::s![.., .., a]).dot(b);
        jac3d.slice_mut(ndarray::s![.., .., a]).assign(&slab);
    }
    let jac = jac3d
        .into_shape_with_order((n_obs, p * d))
        .map_err(|err| format!("refresh_isometry_caches_from_atom: J reshape failed: {err}"))?;

    // The second jet is sourced from the optional `basis_second_jet`
    // slot. The trait split (`SaeBasisEvaluator` vs `SaeBasisSecondJet`)
    // encodes "no closed-form Hessian" as trait absence: when the atom
    // was built with `with_basis_evaluator` (base trait only) the slot
    // is `None` and the `H` cache is not installed. When the atom was
    // built with `with_basis_second_jet` the slot holds the same Arc
    // upcast to the supertrait, and `second_jet` returns the analytic
    // Hessian here.
    let jac2_opt = if let Some(second_eval) = atom.basis_second_jet.as_ref() {
        let hess = second_eval.second_jet(coords)?;
        if hess.dim() != (n_obs, m, d, d) {
            return Err(format!(
                "refresh_isometry_caches_from_atom: evaluator second jet has shape {:?}, expected ({n_obs}, {m}, {d}, {d})",
                hess.dim()
            ));
        }
        // H[n, (i*d + a)*d + c]: one (n×m)·(m×p) GEMM per (a, c) pair,
        // scattered into the row-major (n, p, d, d) layout (same GEMM-not-
        // scalar-loop rewrite as J above).
        let mut jac2_4d = ndarray::Array4::<f64>::zeros((n_obs, p, d, d));
        for a in 0..d {
            for c in 0..d {
                let slab = hess.slice(ndarray::s![.., .., a, c]).dot(b);
                jac2_4d
                    .slice_mut(ndarray::s![.., .., a, c])
                    .assign(&slab);
            }
        }
        let jac2 = jac2_4d.into_shape_with_order((n_obs, p * d * d)).map_err(|err| {
            format!("refresh_isometry_caches_from_atom: H reshape failed: {err}")
        })?;
        Some(Arc::new(jac2))
    } else {
        None
    };

    // Third jet K[n, i, ((a·d + c)·d + e)] = Σ_m d³Phi[n, m, a, c, e] · B[m, i]
    // feeds the residual-curvature term of the exact isometry Hessian
    //   B_{ab,cd} = K_{a,cd}^T W J_b + H_{a,c}^T W H_{b,d}
    //             + H_{a,d}^T W H_{b,c} + J_a^T W K_{b,cd}.
    // Sourced from the base evaluator's object-safe `third_jet_dyn` forwarder
    // (closed-form analytic override for every basis with an analytic Hessian:
    // sphere/circle/torus/affine/euclidean/duchon; `None` otherwise — no
    // finite-difference fallback). Installed only when the penalty
    // has no `duchon_radial_source` — a Duchon penalty already carries its own
    // analytic third source and `jacobian_third` would shadow it with this
    // cache. Always written (Some or None) so a stale K from a prior outer step
    // never survives a refresh.
    let jac3_opt = if penalty.duchon_radial_source.is_none() {
        match evaluator.third_jet_dyn(coords) {
            Some(third) => {
                let t3 = third?;
                if t3.dim() != (n_obs, m, d, d, d) {
                    return Err(format!(
                        "refresh_isometry_caches_from_atom: evaluator third jet has shape {:?}, expected ({n_obs}, {m}, {d}, {d}, {d})",
                        t3.dim()
                    ));
                }
                // K[n, i, ((a·d + c)·d + e)]: one (n×m)·(m×p) GEMM per
                // (a, c, e) triple into the row-major (n, p, d, d, d) layout,
                // then flattened — the last axis packing ((a·d + c)·d + e) IS
                // the row-major order of (a, c, e).
                let mut jac3_5d = ndarray::Array5::<f64>::zeros((n_obs, p, d, d, d));
                for a in 0..d {
                    for c in 0..d {
                        for e in 0..d {
                            let slab = t3.slice(ndarray::s![.., .., a, c, e]).dot(b);
                            jac3_5d
                                .slice_mut(ndarray::s![.., .., a, c, e])
                                .assign(&slab);
                        }
                    }
                }
                let jac3 = jac3_5d
                    .into_shape_with_order((n_obs, p, d * d * d))
                    .map_err(|err| {
                        format!("refresh_isometry_caches_from_atom: K reshape failed: {err}")
                    })?;
                Some(Arc::new(jac3))
            }
            None => None,
        }
    } else {
        None
    };

    let installed = jac2_opt.is_some();
    penalty.refresh_caches(Some(Arc::new(jac)), jac2_opt);
    penalty.set_third_decoder_derivative(jac3_opt);
    Ok(installed)
}

/// Walk an [`AnalyticPenaltyRegistry`] and refresh every Isometry penalty
/// against the SAE atom it owns. The alignment rule is positional within each
/// `(latent_dim, p_out)` signature: the penalty's `target.latent_dim` must
/// equal the atom's `latent_dim` AND the penalty's `p_out` must equal the
/// atom's decoder column count `p`. Multi-atom configurations install one
/// isometry penalty per atom, so the *k*-th isometry penalty matching a given
/// signature is paired with the *k*-th atom matching that same signature. This
/// reduces to the unambiguous single-atom/single-penalty case wired by
/// `solver/workflow.rs`, and never collapses multiple penalties onto the first
/// matching atom (which would leave every later atom's coords un-refreshed).
///
/// Returns the number of penalties that got both caches populated (i.e. the
/// number of atoms whose `basis_second_jet` slot holds a
/// [`SaeBasisSecondJet`] implementation supplying the analytic Hessian).
pub fn refresh_isometry_caches_from_term(
    registry: &AnalyticPenaltyRegistry,
    term: &SaeManifoldTerm,
    coords_per_atom: &[Array2<f64>],
) -> Result<usize, String> {
    if coords_per_atom.len() != term.atoms.len() {
        return Err(format!(
            "refresh_isometry_caches_from_term: coords_per_atom length {} != number of atoms {}",
            coords_per_atom.len(),
            term.atoms.len()
        ));
    }
    let mut refreshed_with_second = 0usize;
    // Per-signature cursor: how many atoms matching a given (latent_dim, p_out)
    // have already been consumed by earlier isometry penalties. Pairing the
    // k-th penalty of a signature with the k-th atom of that signature gives a
    // stable one-to-one mapping for multi-atom configs.
    let mut consumed_per_signature: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for entry in registry.penalties.iter() {
        let AnalyticPenaltyKind::Isometry(p) = entry else {
            continue;
        };
        let Some(p_latent_dim) = p.target.latent_dim else {
            continue;
        };
        let signature = (p_latent_dim, p.p_out);
        let already_consumed = consumed_per_signature.entry(signature).or_insert(0);
        // Advance to the (already_consumed)-th atom matching this signature.
        let mut seen = 0usize;
        let mut paired: Option<usize> = None;
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let matches = atom.latent_dim == p_latent_dim
                && atom.decoder_coefficients.ncols() == p.p_out
                && atom.basis_evaluator.is_some();
            if !matches {
                continue;
            }
            if seen == *already_consumed {
                paired = Some(atom_idx);
                break;
            }
            seen += 1;
        }
        let Some(atom_idx) = paired else {
            continue;
        };
        *already_consumed += 1;
        let atom = &term.atoms[atom_idx];
        let coords = coords_per_atom[atom_idx].view();
        if refresh_isometry_caches_from_atom(p, atom, coords)? {
            refreshed_with_second += 1;
        }
    }
    Ok(refreshed_with_second)
}
