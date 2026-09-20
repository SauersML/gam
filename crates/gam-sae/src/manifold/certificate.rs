use super::*;

// Superposed-Geometry theory (internal memo, Part VI), curvature/measure side.
//
// The memo's central slogan is "curvature IS identifiability": superposition
// ambiguity is fundamentally a FLATNESS disease. If two atoms' active regions
// co-fire and both are LINEAR (flat) subspaces, any invertible recombination
// (any GL relabeling of the co-active span) reconstructs the data identically
// — the gauge groupoid acting on a flat dictionary is enormous (as large as
// GL itself on the shared span), so a purely linear dictionary is generically
// NON-identifiable under superposition. A CURVED atom is generically RIGID
// instead: by jet transversality, two generic embeddings' second-order
// osculation (agreement of position + tangent + curvature) is an
// infinite-codimension coincidence, so a curved atom's residual gauge
// collapses down to the much smaller Diff × Sym (reparameterize the chart,
// permute/relabel symmetric atoms) rather than a full linear group. Circles —
// showing up everywhere in fitted dictionaries — are not a curiosity; they
// are the optimizer's equilibrium response to superposition pressure: bend
// just enough to buy back identifiability.
//
// This module measures the quantities that claim is stated in: the empirical
// cross-atom frame incoherence `μ̂` and the per-atom second-fundamental-form
// curvature `κ̂`, with the activity floors and reconstruction SNR beside them.
// It publishes them as measurements. No sufficient condition turning them into
// a global-optimality verdict has been derived for curved atoms, so none is
// computed here. The support-side Terracini rank test (`identifiability.rs` /
// `isa_seed.rs`) and the measured dual certificate (`dual_certificate.rs`) are
// separate owners and are not changed by this.

/// Empirical cross-atom incoherence, curvature, activity and SNR of a fitted
/// curved dictionary (#1008). These are measurements; no verdict is derived
/// from them.
#[derive(Clone, Debug)]
pub struct DictionaryIncoherenceReport {
    /// `max_{j != k} sigma_max(U_j^T U_k)` over decoder output subspaces.
    pub mu_hat: f64,
    /// Per-atom maximum empirical second-fundamental-form norm on the fitted
    /// coordinate grid. `None` means the bound is unbounded because a nonzero
    /// normal second derivative was observed at an unresolved tangent frame.
    pub per_atom_kappa_hat: Vec<Option<f64>>,
    /// Mean fitted gate/assignment mass per atom.
    pub per_atom_mean_activity: Vec<f64>,
    /// Largest fitted gate/assignment mass per atom.
    pub per_atom_peak_activity: Vec<f64>,
    /// Conservative dictionary activity floor, `min_k mean_i a_ik`.
    pub mean_activity_floor: f64,
    /// Support floor matching the collapse guard statistic, `min_k max_i a_ik`.
    pub peak_activity_floor: f64,
    /// `mean_i ||sum_k a_ik g_k(t_ik)||^2 / dispersion`.
    pub snr_proxy: f64,
    /// Dispersion used in [`Self::snr_proxy`].
    pub dispersion: f64,
    /// Human-readable summary of the quantities.
    pub note: String,
}

/// The additive post-fit diagnostics for a fitted [`SaeManifoldTerm`]: the
/// two-score per-atom lens, residual-gauge certificate, and empirical
/// incoherence/curvature measurements.
///
/// Built by [`SaeManifoldTerm::fit_diagnostics_report`]. Both reports are pure
/// reads of the fitted term + its single per-row metric; nothing here feeds back
/// into any loss, criterion, penalty, or optimizer state. Under a Euclidean /
/// no-harvest provenance the lens coupling degrades to `None` and the gauge is
/// certified under Euclidean provenance — never an error, never flag-gated.
#[derive(Clone, Debug)]
pub struct SaeManifoldFitDiagnostics {
    /// Per-atom presence / behavioral coupling / discrepancy
    /// ([`crate::inference::atom_lens::atom_two_lens`]).
    pub atom_two_lens: crate::inference::atom_lens::AtomTwoLensReport,
    /// Residual-gauge certificate: which symmetry group the fit is identified up
    /// to (`crate::identifiability::residual_gauge_exact_from_curvature` and
    /// `residual_gauge_exact_from_streamed`).
    pub residual_gauge: crate::identifiability::ResidualGaugeReport,
    /// Empirical curved-dictionary incoherence/curvature measurements (#1008).
    /// Present when the caller supplies the fitted reconstruction dispersion
    /// needed for the SNR proxy; absent for legacy callers that only need the
    /// existing diagnostics.
    pub incoherence_report: Option<DictionaryIncoherenceReport>,
    /// Per-atom Riesz-debiased smooth-functional inference and the any-n-valid
    /// split-LRT smooth-structure e-value (#1097 / #1103), one entry per fitted
    /// atom in atom order.
    /// Each entry's `functionals` / `smooth_significance` are `Some` only when
    /// the atom's inner-decoder smooth was harvested at fit time (the caller ran
    /// [`SaeManifoldTerm::set_atom_inner_fits`] and the inner penalized Hessian
    /// was SPD on a non-empty active set); otherwise they degrade to `None`.
    pub atom_inference: Vec<crate::identifiability::AtomInferenceReport>,
    /// #2081 — per-atom chart coordinate-fidelity certificate: the circular
    /// coordinate-uniformity statistic (Watson `U²` + closed-form p-value)
    /// against the atom's invariant measure, and the arc-length (unit-speed)
    /// defect of the chart parameterization. One entry per fitted atom in atom
    /// order; `None` for atoms without a `d = 1` circle/interval chart. Reports
    /// coordinate quality — which reconstruction EV provably does not certify
    /// (see [`AtomCoordinateFidelity`]).
    pub coordinate_fidelity: Vec<Option<AtomCoordinateFidelity>>,
    /// #2518 — per-atom certificate that the atom's DECODED IMAGE is embedded,
    /// not merely immersed: a rigorous lower bound on the separation function
    /// `‖m(u+s) − m(u)‖² / 4sin²(πs)` over the whole `(center, separation)`
    /// domain ([`crate::manifold::embeddedness::AtomEmbeddednessCertificate`]).
    /// One entry per fitted atom in atom order; `None` for atoms outside the
    /// `d = 1` periodic family. This is the only global statement in this
    /// struct: coordinate fidelity, the residual gauge and the chart guards are
    /// all local or reparameterization-shaped, and a decoder that traverses its
    /// image twice satisfies every one of them while making each encode fiber
    /// two-valued. Read `embedded` as CERTIFIED EMBEDDED — its negation
    /// certifies nothing.
    pub decoder_embeddedness: Vec<Option<crate::manifold::AtomEmbeddednessCertificate>>,
    /// Reviewer-F3 persistent-homology topology audit: for each atom, the
    /// Vietoris–Rips persistence of its assigned-row image points confronted
    /// with the topology the raced type predicts. `Some(..)` carries the
    /// measured components/loops and the first-class
    /// [`AtomTopologyPersistence::contested`] flag (raised when the measured
    /// topology disagrees with the latched race winner — extra components, a
    /// missing predicted loop, or an unpredicted loop), which the probe planner
    /// reads to re-adjudicate rather than trust the winner. `None` for atoms
    /// whose topology is caller-supplied ([`SaeAtomBasisKind::Precomputed`]) or
    /// with too few assigned rows to resolve H₁. One entry per fitted atom in
    /// atom order.
    pub topology_persistence: Vec<Option<AtomTopologyPersistence>>,
}

/// Honest trust-diagnostics payload for the Python `diagnostics` block (#1005).
///
/// This deliberately contains only quantities with exact fitted-state producers:
/// tangent spectrum/condition, assignment support, activation frequency, and the
/// basis-kind untyped flag. No topology margins, level-0 references, coherence,
/// or reconstruction proxy fields are represented here.
#[derive(Clone, Debug)]
pub struct SaeTrustDiagnostics {
    pub atom_trust: Vec<f64>,
    pub atoms: Vec<SaeAtomTrustDiagnostics>,
}

#[derive(Clone, Debug)]
pub struct SaeAtomTrustDiagnostics {
    pub trust_score: f64,
    pub sigma_min_tangent: f64,
    pub sigma_max_tangent: f64,
    pub tangent_condition_score: f64,
    pub coverage: f64,
    pub activation_frequency: f64,
    pub support_mass: f64,
    pub effective_n: f64,
    pub support_ess: f64,
    pub untyped: bool,
    pub active_token_count: usize,
}

/// Measure the curved-dictionary incoherence/curvature quantities of a fitted
/// term under an explicit Gaussian reconstruction dispersion.
///
/// `mu_hat` (via `dictionary_frame_incoherence`) is the empirical cross-atom
/// incoherence — the superposition coupling — and each `per_atom_kappa_hat`
/// entry (via `atom_curvature_bound`) is the empirical second-fundamental-form
/// curvature of that atom's image.
pub(crate) fn dictionary_incoherence_report_with_dispersion(
    term: &SaeManifoldTerm,
    dispersion: f64,
    fitted: ArrayView2<'_, f64>,
) -> Result<DictionaryIncoherenceReport, String> {
    if !dispersion.is_finite() || dispersion <= 0.0 {
        return Err(format!(
            "dictionary_incoherence_report_with_dispersion: dispersion must be finite and positive, got {dispersion}"
        ));
    }
    if fitted.dim() != (term.n_obs(), term.output_dim()) {
        return Err(format!(
            "dictionary_incoherence_report_with_dispersion: fitted {:?} != ({}, {})",
            fitted.dim(),
            term.n_obs(),
            term.output_dim()
        ));
    }
    let mu_hat = dictionary_frame_incoherence(term)?;
    let per_atom_kappa_hat = term
        .atoms
        .iter()
        .enumerate()
        .map(|(atom_idx, _)| atom_curvature_bound(term, atom_idx))
        .collect::<Result<Vec<_>, _>>()?;
    let assignments = term.assignment.assignments();
    let n = assignments.nrows();
    let k_atoms = assignments.ncols();
    let mut per_atom_mean_activity = Vec::with_capacity(k_atoms);
    let mut per_atom_peak_activity = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let support = SupportMeasure::from_assignment_matrix(assignments.view(), atom_idx)?;
        let peak = support.weights().iter().copied().fold(0.0_f64, f64::max);
        per_atom_mean_activity.push(if n > 0 {
            support.mass() / n as f64
        } else {
            0.0
        });
        per_atom_peak_activity.push(peak);
    }
    let mean_activity_floor = per_atom_mean_activity
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let peak_activity_floor = per_atom_peak_activity
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let signal_power = if fitted.is_empty() {
        0.0
    } else {
        fitted.iter().map(|v| v * v).sum::<f64>() / fitted.len() as f64
    };
    let mean_activity_floor = if mean_activity_floor.is_finite() {
        mean_activity_floor
    } else {
        0.0
    };
    let peak_activity_floor = if peak_activity_floor.is_finite() {
        peak_activity_floor
    } else {
        0.0
    };
    let snr_proxy = signal_power / dispersion;
    let kappa_summary = per_atom_kappa_hat
        .iter()
        .copied()
        .try_fold(0.0_f64, |largest, value| {
            value.map(|value| largest.max(value))
        })
        .map(|value| format!("{value:.3e}"))
        .unwrap_or_else(|| "unbounded (unresolved tangent frame)".to_string());
    let note = format!(
        "μ̂={mu_hat:.3e}, κ̂_max={kappa_summary}, a_floor={peak_activity_floor:.3e}, \
         SNR={snr_proxy:.3e}"
    );
    Ok(DictionaryIncoherenceReport {
        mu_hat,
        per_atom_kappa_hat,
        per_atom_mean_activity,
        per_atom_peak_activity,
        mean_activity_floor,
        peak_activity_floor,
        snr_proxy,
        dispersion,
        note,
    })
}

pub(crate) fn dictionary_frame_incoherence(term: &SaeManifoldTerm) -> Result<f64, String> {
    let frames = (0..term.k_atoms())
        .map(|atom_idx| certificate_output_frame(term, atom_idx))
        .collect::<Result<Vec<_>, _>>()?;
    let mut mu = 0.0_f64;
    for j in 0..frames.len() {
        for k in (j + 1)..frames.len() {
            if frames[j].ncols() == 0 || frames[k].ncols() == 0 {
                continue;
            }
            let overlap = fast_atb(&frames[j], &frames[k]);
            let (_u, s, _vt) = overlap.svd(false, false).map_err(|e| {
                format!("dictionary_frame_incoherence: SVD failed for atom pair ({j}, {k}): {e}")
            })?;
            let pair = s.iter().copied().fold(0.0_f64, f64::max);
            mu = mu.max(pair);
        }
    }
    Ok(mu)
}

pub(crate) fn certificate_output_frame(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<Array2<f64>, String> {
    let atom = &term.atoms[atom_idx];
    if atom.decoder_frame.is_some() {
        return Ok(term.frame_output_matrix(atom_idx));
    }
    let p = atom.output_dim();
    let (_u, s, vt_opt) = atom
        .decoder_coefficients()
        .svd(false, true)
        .map_err(|e| format!("certificate_output_frame: SVD failed for atom {atom_idx}: {e}"))?;
    let max_sv = s.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return Ok(Array2::<f64>::zeros((p, 0)));
    }
    let tol = SAE_FRAME_RANK_CUTOFF * max_sv;
    let rank = s.iter().filter(|&&value| value > tol).count();
    let vt = vt_opt.ok_or_else(|| {
        format!("certificate_output_frame: SVD returned no right factor for atom {atom_idx}")
    })?;
    let rank = rank.min(vt.nrows());
    let mut frame = Array2::<f64>::zeros((p, rank));
    for col in 0..rank {
        for row in 0..p {
            frame[[row, col]] = vt[[col, row]];
        }
    }
    Ok(frame)
}

pub(crate) fn atom_curvature_bound(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<Option<f64>, String> {
    let atom = &term.atoms[atom_idx];
    let coords = term.assignment.coords[atom_idx].as_matrix();
    let second = atom
        .basis_evaluator
        .as_ref()
        .and_then(|evaluator| evaluator.second_jet_dyn(coords.view()))
        .ok_or_else(|| {
            format!(
                "atom_curvature_bound: atom {atom_idx} has no analytic second jet; cannot compute kappa_hat"
            )
        })?
        .map_err(|e| format!("atom_curvature_bound: atom {atom_idx} second jet failed: {e}"))?;
    atom_curvature_bound_with_decoder(
        atom,
        atom_idx,
        second.view(),
        atom.decoder_coefficients().view(),
    )
}

/// The sup-norm extrinsic-curvature bound `atom_curvature_bound` as an explicit
/// function of the decoder coefficient matrix `decoder` (shape `(M_k, p)`) and
/// the precomputed second jet, so the #1099 delta-method gradient `∂κ/∂β` can be
/// formed by finite-differencing it in the captured channel's coefficients
/// without mutating the term. With `decoder = atom.decoder_coefficients()` this is
/// exactly `atom_curvature_bound`.
///
/// This is the actual measurement of `κ̂`: at each observation row it forms the
/// tangent frame `J(t) = Φ'(t) B` (the atom's embedded tangent space) and the
/// second jet pushed through the same decoder, projects the second jet
/// orthogonally *off* the tangent frame ([`projected_perp_norm`]), and
/// normalizes by the local tangent scale. That perp-projected, tangent-scaled
/// second derivative is exactly the (extrinsic) second fundamental form of the
/// atom's image manifold — the differential-geometric object whose size *is*
/// the rigidity measure the theory trades on: zero second fundamental form
/// means the atom is locally flat (gauge-vulnerable, per the module's
/// flatness-disease framing). The `max` over rows and axis pairs makes `κ̂` a sup-norm (worst-case, hence
/// conservative) bound. An unresolved tangent frame with a nonzero perpendicular
/// second derivative returns `None`: the bound is unbounded, but absence is kept
/// typed rather than smuggled through report serialization as infinity.
pub(crate) fn atom_curvature_bound_with_decoder(
    atom: &SaeManifoldAtom,
    atom_idx: usize,
    second: ArrayView4<'_, f64>,
    decoder: ArrayView2<'_, f64>,
) -> Result<Option<f64>, String> {
    let n = atom.n_obs();
    let m = atom.basis_size();
    let d = atom.latent_dim();
    let p = atom.output_dim();
    if second.dim() != (n, m, d, d) {
        return Err(format!(
            "atom_curvature_bound: atom {atom_idx} second jet shape {:?} must be ({n}, {m}, {d}, {d})",
            second.dim()
        ));
    }
    if decoder.dim() != (m, p) {
        return Err(format!(
            "atom_curvature_bound: atom {atom_idx} decoder shape {:?} must be ({m}, {p})",
            decoder.dim()
        ));
    }
    if second.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "atom_curvature_bound: atom {atom_idx} second jet contains a non-finite value"
        ));
    }
    if decoder.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "atom_curvature_bound: atom {atom_idx} decoder contains a non-finite value"
        ));
    }
    let mut max_kappa = 0.0_f64;
    let mut tangent = Array2::<f64>::zeros((p, d));
    let mut second_vec = vec![0.0_f64; p];
    for row in 0..n {
        // Tangent J(t) = Φ'(t) B on this row, formed from the explicit decoder.
        tangent.fill(0.0);
        for basis_col in 0..m {
            for axis in 0..d {
                let dphi = atom.basis_jacobian[[row, basis_col, axis]];
                if !dphi.is_finite() {
                    return Err(format!(
                        "atom_curvature_bound: atom {atom_idx} basis Jacobian contains a non-finite value at row {row}, basis {basis_col}, axis {axis}"
                    ));
                }
                if dphi == 0.0 {
                    continue;
                }
                for out in 0..p {
                    tangent[[out, axis]] += dphi * decoder[[basis_col, out]];
                }
            }
        }
        if tangent.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        let tangent_rank = tangent_frame_rank(tangent.view())?;
        let tangent_scale = tangent_rank.0;
        let q = tangent_rank.1;
        for axis_a in 0..d {
            for axis_b in 0..d {
                second_vec.fill(0.0);
                for basis_col in 0..m {
                    let h = second[[row, basis_col, axis_a, axis_b]];
                    if h == 0.0 {
                        continue;
                    }
                    for out in 0..p {
                        second_vec[out] += h * decoder[[basis_col, out]];
                    }
                }
                if second_vec.iter().any(|value| !value.is_finite()) {
                    return Ok(None);
                }
                let perp_norm = projected_perp_norm(&second_vec, q.view());
                if !perp_norm.is_finite() {
                    return Ok(None);
                }
                if tangent_scale > 0.0 {
                    let curvature = perp_norm / tangent_scale;
                    if !curvature.is_finite() {
                        return Ok(None);
                    }
                    max_kappa = max_kappa.max(curvature);
                } else if perp_norm > 0.0 {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(max_kappa))
}

pub(crate) fn tangent_frame_rank(
    tangent: ArrayView2<'_, f64>,
) -> Result<(f64, Array2<f64>), String> {
    let p = tangent.nrows();
    let d = tangent.ncols();
    if p == 0 || d == 0 {
        return Ok((0.0, Array2::<f64>::zeros((p, 0))));
    }
    let (u_opt, s, _vt) = tangent
        .to_owned()
        .svd(true, false)
        .map_err(|e| format!("tangent_frame_rank: SVD failed: {e}"))?;
    let max_sv = s.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return Ok((0.0, Array2::<f64>::zeros((p, 0))));
    }
    let tol = SAE_FRAME_RANK_CUTOFF * max_sv;
    let rank = s.iter().filter(|&&value| value > tol).count();
    let min_positive = s
        .iter()
        .copied()
        .filter(|value| *value > tol)
        .fold(f64::INFINITY, f64::min);
    let u = u_opt.ok_or_else(|| "tangent_frame_rank: SVD returned no U".to_string())?;
    let rank = rank.min(u.ncols());
    let mut q = Array2::<f64>::zeros((p, rank));
    for col in 0..rank {
        for row in 0..p {
            q[[row, col]] = u[[row, col]];
        }
    }
    Ok((min_positive * min_positive, q))
}

pub(crate) fn projected_perp_norm(vector: &[f64], tangent_frame: ArrayView2<'_, f64>) -> f64 {
    let mut residual = vector.to_vec();
    for axis in 0..tangent_frame.ncols() {
        let mut coeff = 0.0_f64;
        for out in 0..tangent_frame.nrows() {
            coeff += tangent_frame[[out, axis]] * vector[out];
        }
        if coeff == 0.0 {
            continue;
        }
        for out in 0..tangent_frame.nrows() {
            residual[out] -= coeff * tangent_frame[[out, axis]];
        }
    }
    residual.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[cfg(test)]
mod certificate_curvature_tests {
    use super::*;

    /// Closed-form check of the extrinsic-curvature bound `κ̂`
    /// ([`atom_curvature_bound_with_decoder`]): a planted radius-`r` circle
    /// `m(θ) = r·(cos θ, sin θ)` has plane-curvature exactly `1/r`, and the
    /// certificate's tangent-scaled second-fundamental-form norm
    /// `κ̂ = ‖P_⊥ m''‖ / ‖m'‖²` must reproduce it to machine precision (and
    /// scale as `1/r`, since a bigger circle bends less). This pins the
    /// normalization — dividing the perp second-derivative by the SQUARED
    /// tangent singular value, not the singular value.
    #[test]
    fn atom_curvature_bound_recovers_circle_reciprocal_radius() {
        use ndarray::{Array2, Array4};
        let thetas = [0.0_f64, 0.3, 1.1, 2.7, 4.9];
        for &r in &[0.5_f64, 1.0, 2.0, 7.5] {
            let n = thetas.len();
            let mut phi = Array2::<f64>::zeros((n, 2));
            let mut jac = ndarray::Array3::<f64>::zeros((n, 2, 1));
            let mut second = Array4::<f64>::zeros((n, 2, 1, 1));
            for (i, &t) in thetas.iter().enumerate() {
                phi[[i, 0]] = t.cos();
                phi[[i, 1]] = t.sin();
                jac[[i, 0, 0]] = -t.sin();
                jac[[i, 1, 0]] = t.cos();
                second[[i, 0, 0, 0]] = -t.cos();
                second[[i, 1, 0, 0]] = -t.sin();
            }
            // Decoder m(θ) = r·(cos θ, sin θ): B = r·I₂ (basis rows → output).
            let mut decoder = Array2::<f64>::zeros((2, 2));
            decoder[[0, 0]] = r;
            decoder[[1, 1]] = r;
            let atom = SaeManifoldAtom::new_with_provided_function_gram(
                "circle",
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jac,
                decoder.clone(),
                Array2::<f64>::eye(2),
            )
            .unwrap();
            let kappa = atom_curvature_bound_with_decoder(
                &atom,
                0,
                second.view(),
                decoder.view(),
            )
            .unwrap()
            .expect("a regular circle has a finite curvature bound");
            let expected = 1.0 / r;
            assert!(
                (kappa - expected).abs() < 1.0e-9,
                "circle radius {r}: κ̂ must be 1/r = {expected}, got {kappa}"
            );
        }
    }

    #[test]
    fn unresolved_tangent_curvature_is_typed_unbounded() {
        use ndarray::{Array2, Array3, Array4};

        let phi = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let jac = Array3::from_shape_vec((1, 2, 1), vec![0.0, 1.0]).unwrap();
        let second = Array4::from_shape_vec((1, 2, 1, 1), vec![-1.0, 0.0]).unwrap();
        let decoder = Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap();
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "degenerate-point",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jac,
            decoder.clone(),
            Array2::<f64>::eye(2),
        )
        .unwrap();

        assert_eq!(
            atom_curvature_bound_with_decoder(&atom, 0, second.view(), decoder.view()).unwrap(),
            None,
            "nonzero normal curvature with no resolved tangent has an unbounded, not infinite-sentinel, bound"
        );
    }
}
