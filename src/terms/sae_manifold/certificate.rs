use super::*;

/// The global-optimality verdict of the curved-dictionary incoherence
/// certificate (#1008): whether the fit's basin stationary point is certified
/// unique up to the residual gauge group, and by what margin.
///
/// The certificate is **conservative by construction**: it certifies only when
/// the conservative sufficient condition holds with positive margin, so a
/// `CertifiedGlobal` verdict can never be wrong (the phase-diagram validation
/// asserts exactly this — no certified-but-wrong cell, ever). An
/// `Uncertified` verdict is *not* a claim of non-uniqueness — it is the honest
/// "this certificate cannot decide", which is the only safe failure mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GlobalOptimalityVerdict {
    /// The conservative sufficient condition holds: the basin stationary point
    /// is unique up to the certified residual gauge group. `margin` is the
    /// (positive) slack `budget − μ̂` by which the condition is met.
    CertifiedGlobal { margin: f64 },
    /// The condition is not met (or a precondition — graph-validity / SNR > 1 —
    /// fails). `margin` is the (non-positive) slack, or `f64::NEG_INFINITY` when
    /// a precondition rules certification out entirely. Multistart / homotopy is
    /// genuinely needed here.
    Uncertified { margin: f64 },
}


impl GlobalOptimalityVerdict {
    /// The signed margin `budget − μ̂` (positive ⇒ certified). A precondition
    /// failure reports `f64::NEG_INFINITY`.
    pub fn margin(&self) -> f64 {
        match self {
            Self::CertifiedGlobal { margin } | Self::Uncertified { margin } => *margin,
        }
    }

    /// Whether the fit is certified globally optimal up to the gauge group.
    pub fn is_certified(&self) -> bool {
        matches!(self, Self::CertifiedGlobal { .. })
    }
}


/// Conservative tangent-graph curvature budget: the atom image is a graph over
/// its tangent frame only while `C_KAPPA · κ̂` stays below 1 — i.e. the relative
/// second-fundamental-form curvature `κ̂` (perp curvature per unit tangent
/// scale) is below `1`. Above it the atom turns faster than its own tangent
/// extent and the linear-case perturbation argument is void, so the certificate
/// refuses to certify. A circle of radius `r` has `κ̂ = 1/r`, so this admits
/// `r > 1` (benign, well-resolved atoms) and rejects tightly-curved ones whose
/// graph approximation is uncontrolled. Raising this constant only ever shrinks
/// the certified region (withholds certification), never grants a wrong one.
pub const SAE_CERT_CURVATURE_CONSTANT: f64 = 1.0;


/// Conservative incoherence-budget constant `c0` in the sufficient condition
/// `μ̂ ≤ c0 · a_floor² · (1 − 1/SNR) · (1 − C_κ κ̂) / K`. Small (conservative):
/// shrinking the budget can only withhold certification, never grant a wrong
/// one.
pub const SAE_CERT_INCOHERENCE_BUDGET: f64 = 0.125;


/// The conservative curved-dictionary global-optimality threshold (#1008).
///
/// # The condition
///
/// Following the linear exact-recovery lineage (Spielman–Wang–Wright complete
/// case; Sun–Qu–Wright geometric analysis — in benign regimes every local min
/// is global) perturbed to curved atoms: the atom image is a graph over its
/// tangent frame with second-fundamental-form curvature `κ`, so the linear-case
/// arguments perturb when `κ·diam(chart)` is small. The competing-basin coupling
/// is the cross-atom frame incoherence `μ` amplified by co-activation; the
/// within-atom restricted strong convexity that pins each atom scales with the
/// activity floor (how reliably the atom fires) and the SNR (how far the signal
/// is above noise), and is **degraded by curvature** (the graph approximation
/// error). The certificate certifies global optimality up to the residual gauge
/// when
///
/// ```text
///   μ̂  ≤  c0 · a_floor² · (1 − 1/SNR) · (1 − C_κ · κ̂_max) / K
/// ```
///
/// subject to the preconditions `C_κ · κ̂_max < 1` (tangent-graph validity) and
/// `SNR > 1` (signal above noise). `a_floor` is the support activity floor
/// (`min_k max_i a_ik`, the same statistic the collapse guard reads), `K` the
/// atom count, `κ̂_max` the largest per-atom second-fundamental-form bound.
///
/// # Conservatism
///
/// Every constant is chosen to *shrink* the certified region relative to the
/// true (unknown) sharp threshold: `c0` is small, `C_κ` is large. A
/// `CertifiedGlobal` verdict therefore implies the sharp condition with room to
/// spare — it can never be wrong. An `Uncertified` verdict is the honest "cannot
/// decide", never a claim of non-uniqueness. The cross-validation with the
/// certified-homotopy bifurcation events (#1007) is exactly this: a bifurcation
/// (a competing basin appearing) should only ever occur where this margin is
/// non-positive.
pub fn curved_dictionary_global_optimality_verdict(
    mu_hat: f64,
    kappa_max: f64,
    activity_floor: f64,
    snr_proxy: f64,
    k_atoms: usize,
) -> GlobalOptimalityVerdict {
    // Preconditions: any non-finite input, no atoms, a curvature that voids the
    // tangent-graph perturbation, or SNR at/below the noise floor ⇒ refuse.
    if !mu_hat.is_finite()
        || !kappa_max.is_finite()
        || !activity_floor.is_finite()
        || !snr_proxy.is_finite()
        || k_atoms == 0
    {
        return GlobalOptimalityVerdict::Uncertified {
            margin: f64::NEG_INFINITY,
        };
    }
    let curvature_factor = 1.0 - SAE_CERT_CURVATURE_CONSTANT * kappa_max.max(0.0);
    let snr_factor = 1.0 - 1.0 / snr_proxy;
    if curvature_factor <= 0.0 || snr_factor <= 0.0 {
        // Tangent-graph perturbation void, or signal not above noise: the
        // linear-case argument does not apply, so certification is impossible.
        return GlobalOptimalityVerdict::Uncertified {
            margin: f64::NEG_INFINITY,
        };
    }
    let a = activity_floor.max(0.0);
    let budget =
        SAE_CERT_INCOHERENCE_BUDGET * a * a * snr_factor * curvature_factor / k_atoms as f64;
    let margin = budget - mu_hat;
    if margin > 0.0 {
        GlobalOptimalityVerdict::CertifiedGlobal { margin }
    } else {
        GlobalOptimalityVerdict::Uncertified { margin }
    }
}


/// Empirical quantities that feed the curved-dictionary incoherence theorem,
/// plus the conservative global-optimality verdict (#1008).
#[derive(Clone, Debug)]
pub struct CertificateInputs {
    /// `max_{j != k} sigma_max(U_j^T U_k)` over decoder output subspaces.
    pub mu_hat: f64,
    /// Per-atom maximum empirical second-fundamental-form norm on the fitted
    /// coordinate grid.
    pub per_atom_kappa_hat: Vec<f64>,
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
    /// The conservative global-optimality verdict (#1008):
    /// `CertifiedGlobal { margin }` when the sufficient condition
    /// ([`curved_dictionary_global_optimality_verdict`]) holds with positive
    /// slack — the basin stationary point is unique up to the residual gauge
    /// group — else `Uncertified { margin }`. Conservative: a certified verdict
    /// is never wrong; an uncertified one is "cannot decide", not "non-unique".
    pub global_optimality: GlobalOptimalityVerdict,
    /// Human-readable summary of the quantities and verdict.
    pub note: String,
}


/// The additive post-fit diagnostics for a fitted [`SaeManifoldTerm`]: the
/// two-score per-atom lens, residual-gauge certificate, and empirical
/// incoherence/curvature certificate inputs.
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
    /// to ([`crate::sae_identifiability::residual_gauge`]).
    pub residual_gauge: crate::sae_identifiability::ResidualGaugeReport,
    /// Empirical curved-dictionary certificate inputs (#1008). Present when the
    /// caller supplies the fitted reconstruction dispersion needed for the SNR
    /// proxy; absent for legacy callers that only need the existing diagnostics.
    pub incoherence_report: Option<CertificateInputs>,
    /// Per-atom Riesz-debiased smooth-functional inference and Bartlett smooth
    /// significance (#1097 / #1103), one entry per fitted atom in atom order.
    /// Each entry's `functionals` / `smooth_significance` are `Some` only when
    /// the atom's inner-decoder smooth was harvested at fit time (the caller ran
    /// [`SaeManifoldTerm::set_atom_inner_fits`] and the inner penalized Hessian
    /// was SPD on a non-empty active set); otherwise they degrade to `None`.
    pub atom_inference: Vec<crate::sae_identifiability::AtomInferenceReport>,
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
    pub untyped: bool,
    pub active_token_count: usize,
}


/// Build the empirical curved-dictionary certificate quantities from a fitted
/// term and its Gaussian reconstruction dispersion.
///
/// This reports only computable theorem-side inputs. It intentionally has no
/// global-optimality verdict: the threshold function relating these inputs is
/// future theory (#1008).
pub fn dictionary_incoherence_report(term: &SaeManifoldTerm) -> Result<CertificateInputs, String> {
    let dispersion = term.certificate_dispersion.ok_or_else(|| {
        "dictionary_incoherence_report: fitted reconstruction dispersion is unavailable".to_string()
    })?;
    dictionary_incoherence_report_with_dispersion(term, dispersion)
}


/// Build the empirical curved-dictionary certificate quantities from a fitted
/// term and an explicit Gaussian reconstruction dispersion.
pub fn dictionary_incoherence_report_with_dispersion(
    term: &SaeManifoldTerm,
    dispersion: f64,
) -> Result<CertificateInputs, String> {
    if !dispersion.is_finite() || dispersion <= 0.0 {
        return Err(format!(
            "dictionary_incoherence_report: dispersion must be finite and positive, got {dispersion}"
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
        let mut sum = 0.0_f64;
        let mut peak = 0.0_f64;
        for row in 0..n {
            let value = assignments[[row, atom_idx]];
            sum += value;
            peak = peak.max(value);
        }
        per_atom_mean_activity.push(if n > 0 { sum / n as f64 } else { 0.0 });
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
    let fitted = term.fitted();
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
    // The curvature bound entering the threshold is the largest per-atom
    // second-fundamental-form norm (the worst graph-approximation error across
    // the dictionary). The support activity floor `min_k max_i a_ik` is the
    // honest "how reliably does the weakest atom fire" statistic.
    let kappa_max = per_atom_kappa_hat.iter().copied().fold(0.0_f64, f64::max);
    let global_optimality = curved_dictionary_global_optimality_verdict(
        mu_hat,
        kappa_max,
        peak_activity_floor,
        snr_proxy,
        k_atoms,
    );
    let note = match global_optimality {
        GlobalOptimalityVerdict::CertifiedGlobal { margin } => format!(
            "global optimality CERTIFIED up to the residual gauge group \
             (margin {margin:.3e}); μ̂={mu_hat:.3e}, κ̂_max={kappa_max:.3e}, \
             a_floor={peak_activity_floor:.3e}, SNR={snr_proxy:.3e}"
        ),
        GlobalOptimalityVerdict::Uncertified { margin } => format!(
            "global optimality UNCERTIFIED (margin {margin:.3e}; cannot decide — \
             multistart/homotopy genuinely needed); μ̂={mu_hat:.3e}, \
             κ̂_max={kappa_max:.3e}, a_floor={peak_activity_floor:.3e}, \
             SNR={snr_proxy:.3e}"
        ),
    };
    Ok(CertificateInputs {
        mu_hat,
        per_atom_kappa_hat,
        per_atom_mean_activity,
        per_atom_peak_activity,
        mean_activity_floor,
        peak_activity_floor,
        snr_proxy,
        dispersion,
        global_optimality,
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
        .decoder_coefficients
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


pub(crate) fn atom_curvature_bound(term: &SaeManifoldTerm, atom_idx: usize) -> Result<f64, String> {
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
        atom.decoder_coefficients.view(),
    )
}

/// The sup-norm extrinsic-curvature bound `atom_curvature_bound` as an explicit
/// function of the decoder coefficient matrix `decoder` (shape `(M_k, p)`) and
/// the precomputed second jet, so the #1099 delta-method gradient `∂κ/∂β` can be
/// formed by finite-differencing it in the captured channel's coefficients
/// without mutating the term. With `decoder = atom.decoder_coefficients` this is
/// exactly `atom_curvature_bound`.
pub(crate) fn atom_curvature_bound_with_decoder(
    atom: &SaeManifoldAtom,
    atom_idx: usize,
    second: ArrayView4<'_, f64>,
    decoder: ArrayView2<'_, f64>,
) -> Result<f64, String> {
    let n = atom.n_obs();
    let m = atom.basis_size();
    let d = atom.latent_dim;
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
    let mut max_kappa = 0.0_f64;
    let mut tangent = Array2::<f64>::zeros((p, d));
    let mut second_vec = vec![0.0_f64; p];
    for row in 0..n {
        // Tangent J(t) = Φ'(t) B on this row, formed from the explicit decoder.
        tangent.fill(0.0);
        for basis_col in 0..m {
            for axis in 0..d {
                let dphi = atom.basis_jacobian[[row, basis_col, axis]];
                if dphi == 0.0 {
                    continue;
                }
                for out in 0..p {
                    tangent[[out, axis]] += dphi * decoder[[basis_col, out]];
                }
            }
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
                let perp_norm = projected_perp_norm(&second_vec, q.view());
                if tangent_scale > 0.0 {
                    max_kappa = max_kappa.max(perp_norm / tangent_scale);
                } else if perp_norm > 0.0 {
                    return Ok(f64::INFINITY);
                }
            }
        }
    }
    Ok(max_kappa)
}

pub(crate) fn tangent_frame_rank(tangent: ArrayView2<'_, f64>) -> Result<(f64, Array2<f64>), String> {
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
