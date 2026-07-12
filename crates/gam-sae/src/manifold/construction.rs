use super::*;
use gam_math::special::bessel_i0_centered_terms_from_log_abs;

// ── Theorem K: the rank charge is a RUNNING COMPLEXITY λ(n) ──────────────────
//
// The birth/death evidence charge on an atom is not an ad-hoc penalty; it is one
// evaluation of the running (marginal-likelihood) complexity
//
//     λ(n) := d(−log Z_n) / d(log n),
//
// the local slope of the log marginal likelihood in log sample size. Watanabe's
// singular-learning theory says −log Z_n = n·L_n(ŵ) + λ·log n + o(log n), so λ IS
// the coefficient of log n in the evidence. Theorem K observes that the THREE
// quantities this code juggles are the SAME object λ evaluated in three regimes:
//
//   • HARD MP reconstruction rank (n → ∞ limit, atom well above the noise edge):
//     every resolved decoder direction is a regular parameter,
//     λ → ½·rank_reconstructed·basis_edf = ½·d_eff.
//   • WBIC SOFT count (finite n, atom NEAR the Marchenko–Pastur edge): the
//     audit-only `wbic_audit` report records the tempered fractional count. It
//     is diagnostic, not an alternative production criterion.
//   • RLCT (SINGULAR truth, a symmetry orbit or a null atom): λ drops below ½·d
//     to the real log-canonical threshold. The null atom (truth B*=0) has λ=½ from
//     the amplitude singularity of a²‖B‖² — see the veto in `penalized_quasi_laplace_criterion`.
//
// Soft → hard away from the edge (every sigmoid → 1) and soft → RLCT at singular
// truths (sigmoids → 0), so the single ledger `λ(n_eff)·ln n_eff` interpolates all
// three regimes continuously. The log-scale is the OCCUPANCY-corrected `ln n_eff`
// (Fisher information actually accumulated by a gated atom), never the global row
// count — see the #2a inert-row axiom in `penalized_quasi_laplace_criterion`.
//
// The production criterion has one charge currency: the chargeable-rank branch.
// It equals the hard MP reconstruction rank when at least one direction clears the edge;
// #2258 promotes an MP-rank-zero but numerically alive decoder to the minimum
// chargeable rank one. Keeping an un-differentiated soft alternative would make
// value and analytic gradient describe different objectives, so the fractional
// WBIC count remains audit-only.

/// #9 streaming rank-charge inputs, accumulated in a SINGLE pass through
/// [`SaeManifoldTerm::streaming_exact_arrow_log_det`]: the coordinate-block
/// log-det `log_det_tt` (= 2·`htt_half`; the part the
/// rank charge replaces), plus the per-atom decoder Grams `G_k =
/// Φ_kᵀdiag(a_k²)Φ_k` and the effective sample sizes `N_eff,k = Σ_row a_k²`.
/// Both are chunk-additive, so accumulating them over the streaming chunks equals
/// the dense `accumulate_decoder_gram` / `Σ a²` exactly — the streaming criterion
/// then prices atoms through the SAME `rank_dof_from_grams` MP hard count as the
/// dense path (the dense-vs-streaming parity guarantee).
#[derive(Default)]
pub struct StreamingRankInputs {
    pub(crate) log_det_tt: f64,
    pub(crate) grams: Vec<Array2<f64>>,
    pub(crate) n_eff: Vec<f64>,
}

/// #16/#2023 — the SINGLE per-atom rank-charge DOF core: `d_eff = rank_eff · basis_edf`
/// for ONE atom from its weighted basis Gram `gram = Φᵀdiag(a²)Φ` (m×m), `decoder`
/// (m×p), effective sample size `n_eff = Σ_row a²`, output dim `p_out`, noise floor
/// `r_floor` (dispersion R, assumed already guarded > 0), and smoothness `(lam_smooth,
/// smooth_penalty)`.
///   * `rank_reconstructed` = Marchenko–Pastur HARD reconstruction count on the per-atom
///     reconstruction Gram
///     `(1/n_eff)·BᵀB`, `B = diag(a)·Φ·D`: eigenvalues = svd(diag(√λ)·Uᵀ·D)²/n_eff with
///     `(λ,U)=eigh(gram)`; count those above `R·(1+√(p/n_eff))²` (a real rank-2 circle
///     → 2). `rank_eff` is the production chargeable rank: it equals
///     `rank_reconstructed` unless the #2258 alive-below-edge rule promotes 0 to 1;
///     only a vanished decoder remains 0. [#1893/#11]
///   * `basis_edf = tr(gram·(gram+λS)⁻¹)`.
/// This is the source of truth the term-level `rank_dof_from_grams` (dense + #9
/// streaming) loops, AND that the #2023 migration gate prices linear/curved candidates
/// through — so PROMOTE (birth) and DEMOTE (hybrid split) adjudicate in ONE currency.
/// #2258 reconstruction-rank-vs-degeneracy threshold: a rank-0 atom whose top
/// reconstruction-Gram eigenvalue exceeds `RANK_VANISHED_REL · R` is ALIVE
/// (below the MP reconstruction-rank edge, not degenerate) and is promoted to the
/// minimum chargeable rank 1; at or below it the decoder has genuinely
/// vanished and the categorical Laplace-validity veto applies. Shared by the
/// value path here, the WBIC diagnostic, and the ρ-derivative through
/// [`classify_reconstruction_rank`], so the three views cannot desync.
pub(crate) const RANK_VANISHED_REL: f64 = 1.0e-9;

/// The two integer rank notions carried by the evidence code.
///
/// `mp_reconstruction_rank` counts how many reconstruction directions clear
/// the Marchenko–Pastur rank edge. It is not an information-theoretic detection
/// limit.
/// `production_chargeable_rank` answers a different degeneracy question: how
/// many directions may the Laplace value price? They differ only when no
/// direction clears the MP edge but the fitted decoder is still numerically
/// alive, in which case #2258 charges the minimum non-degenerate rank one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ReconstructionRankClassification {
    pub mp_reconstruction_rank: usize,
    pub production_chargeable_rank: usize,
    pub top_signal: f64,
}

/// Convert a reconstruction singular value to per-observation energy without
/// forming `singular_value²` first. Squaring first can overflow even when the
/// quotient is representable; normalizing by `√n_eff` before squaring preserves
/// that representable range. A genuinely unrepresentable energy is an error,
/// not an infinite direction that silently wins the MP comparison.
pub(super) fn normalized_reconstruction_energy(
    singular_value: f64,
    n_eff: f64,
) -> Result<f64, String> {
    if !singular_value.is_finite() || singular_value < 0.0 {
        return Err(format!(
            "reconstruction singular value must be finite and non-negative; got {singular_value}"
        ));
    }
    if !n_eff.is_finite() || n_eff <= 0.0 {
        return Err(format!(
            "reconstruction energy needs finite positive n_eff; got {n_eff}"
        ));
    }
    let normalized = singular_value / n_eff.sqrt();
    let energy = normalized * normalized;
    if !energy.is_finite() {
        return Err(format!(
            "per-observation reconstruction energy overflowed for singular value \
             {singular_value} and n_eff={n_eff}"
        ));
    }
    Ok(energy)
}

/// Single source of truth for MP reconstruction rank versus production chargeability.
///
/// Callers supply the per-observation reconstruction-Gram eigenvalues `mu`, the
/// MP reconstruction-rank `edge`, and the reconstruction dispersion `r_floor`. Inputs are
/// validated by [`validate_rank_charge_problem`] before production reaches this
/// helper; [`super::wbic_audit::ReconSpectrum`] stores the same validated values.
pub(super) fn classify_reconstruction_rank(
    mu: &[f64],
    edge: f64,
    r_floor: f64,
) -> ReconstructionRankClassification {
    let mut mp_reconstruction_rank = 0usize;
    let mut top_signal = 0.0_f64;
    for &signal in mu {
        if signal > edge {
            mp_reconstruction_rank += 1;
        }
        top_signal = top_signal.max(signal);
    }
    let production_chargeable_rank = if mp_reconstruction_rank > 0 {
        mp_reconstruction_rank
    } else if top_signal > RANK_VANISHED_REL * r_floor {
        1
    } else {
        0
    };
    ReconstructionRankClassification {
        mp_reconstruction_rank,
        production_chargeable_rank,
        top_signal,
    }
}

/// Validate the shared reconstruction-rank problem before either the production
/// value path or the WBIC audit touches a factorization. Keeping this contract
/// in one place prevents the two mathematically identical paths from drifting,
/// and turns ndarray dimension mismatches into ordinary errors instead of dot-
/// product panics.
pub(super) fn validate_rank_charge_problem(
    gram: &Array2<f64>,
    decoder: &Array2<f64>,
    n_eff: f64,
    p_out: f64,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<(), String> {
    let m = gram.nrows();
    if gram.ncols() != m {
        return Err(format!(
            "rank-charge Gram must be square; got {:?}",
            gram.dim()
        ));
    }
    if decoder.nrows() != m {
        return Err(format!(
            "rank-charge decoder must have {m} rows; got {:?}",
            decoder.dim()
        ));
    }
    if !p_out.is_finite() || p_out < 0.0 || p_out != decoder.ncols() as f64 {
        return Err(format!(
            "rank-charge p_out must equal the decoder width {}; got {p_out}",
            decoder.ncols()
        ));
    }
    if !n_eff.is_finite() || n_eff < 0.0 {
        return Err(format!(
            "rank-charge n_eff must be finite and non-negative; got {n_eff}"
        ));
    }
    if !r_floor.is_finite() || r_floor < 0.0 {
        return Err(format!(
            "rank-charge dispersion must be finite and non-negative; got {r_floor}"
        ));
    }
    if !lam_smooth.is_finite() || lam_smooth < 0.0 {
        return Err(format!(
            "rank-charge smoothing weight must be finite and non-negative; got {lam_smooth}"
        ));
    }
    if gram.iter().any(|value| !value.is_finite()) {
        return Err("rank-charge Gram must be finite".to_string());
    }
    if decoder.iter().any(|value| !value.is_finite()) {
        return Err("rank-charge decoder must be finite".to_string());
    }

    let gram_scale = gram.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let gram_symmetry_tolerance =
        64.0 * m as f64 * f64::EPSILON * gram_scale.max(f64::MIN_POSITIVE);
    for row in 0..m {
        for col in 0..row {
            if (gram[[row, col]] - gram[[col, row]]).abs() > gram_symmetry_tolerance {
                return Err(format!(
                    "rank-charge Gram is materially asymmetric at ({row}, {col})"
                ));
            }
        }
    }

    if let Some(penalty) = smooth_penalty {
        if penalty.dim() != (m, m) {
            return Err(format!(
                "rank-charge smooth penalty shape {:?} does not match Gram shape ({m}, {m})",
                penalty.dim()
            ));
        }
        if penalty.iter().any(|value| !value.is_finite()) {
            return Err("rank-charge smooth penalty must be finite".to_string());
        }
        let penalty_scale = penalty
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let penalty_symmetry_tolerance =
            64.0 * m as f64 * f64::EPSILON * penalty_scale.max(f64::MIN_POSITIVE);
        for row in 0..m {
            for col in 0..row {
                if (penalty[[row, col]] - penalty[[col, row]]).abs() > penalty_symmetry_tolerance {
                    return Err(format!(
                        "rank-charge smooth penalty is materially asymmetric at ({row}, {col})"
                    ));
                }
            }
        }
        let (penalty_eigenvalues, _) = penalty
            .eigh(super::Side::Lower)
            .map_err(|error| format!("rank-charge smooth-penalty eigendecomposition: {error}"))?;
        certified_psd_spectrum(penalty_eigenvalues.view(), "rank-charge smooth penalty")?;
    }
    Ok(())
}

/// Certify the eigenspectrum of a matrix promised to be a Gram matrix. Tiny
/// negative eigenvalues inside the symmetric eigensolver's backward-error
/// envelope are numerical zero; a material negative direction is invalid data,
/// not a direction the rank charge may silently discard.
pub(super) fn certified_psd_spectrum(
    eigenvalues: ArrayView1<'_, f64>,
    matrix_name: &str,
) -> Result<Vec<f64>, String> {
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!("{matrix_name} eigenspectrum is non-finite"));
    }
    let scale = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = 64.0 * eigenvalues.len() as f64 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    let mut certified = Vec::with_capacity(eigenvalues.len());
    for (axis, &eigenvalue) in eigenvalues.iter().enumerate() {
        if eigenvalue < -tolerance {
            return Err(format!(
                "{matrix_name} is materially indefinite: eigenvalue {axis}={eigenvalue:.6e} is below -{tolerance:.6e}"
            ));
        }
        certified.push(eigenvalue.max(0.0));
    }
    Ok(certified)
}

/// Certify the ridge-trace effective dimension
/// `tr((G + lambda S)^-1 G)`.  For positive-semidefinite `G` and `S` and a
/// positive-definite penalized Gram matrix the exact value lies in `[0, m]`.
/// Only roundoff inside the standard dense-solve backward-error envelope may
/// be snapped to that interval; a larger excursion is a failed numerical
/// certificate, not an EDF that may be silently projected to a different
/// model.
pub(super) fn certified_basis_edf(
    raw_basis_edf: f64,
    basis_dim: usize,
    context: &str,
) -> Result<f64, String> {
    if !raw_basis_edf.is_finite() {
        return Err(format!("{context}: basis EDF is non-finite"));
    }
    let upper = basis_dim as f64;
    let tolerance = 64.0
        * upper.max(1.0)
        * f64::EPSILON
        * raw_basis_edf.abs().max(upper).max(f64::MIN_POSITIVE);
    if raw_basis_edf < -tolerance || raw_basis_edf > upper + tolerance {
        return Err(format!(
            "{context}: basis EDF {raw_basis_edf:.6e} is outside the certified [0, {upper}] interval (roundoff tolerance {tolerance:.6e})"
        ));
    }
    Ok(raw_basis_edf.clamp(0.0, upper))
}

#[cfg(test)]
mod basis_edf_certificate_tests {
    use super::certified_basis_edf;

    #[test]
    fn snaps_only_backward_error_scale_boundary_drift() {
        let drift = 64.0 * f64::EPSILON;
        assert_eq!(certified_basis_edf(-drift, 4, "test").unwrap(), 0.0);
        assert_eq!(certified_basis_edf(4.0 + drift, 4, "test").unwrap(), 4.0);
    }

    #[test]
    fn refuses_material_or_nonfinite_edf_excursions() {
        for raw in [-1.0e-8, 4.0 + 1.0e-8, f64::NAN, f64::INFINITY] {
            assert!(certified_basis_edf(raw, 4, "test").is_err());
        }
    }
}

pub(crate) fn realised_rank_charge_dof(
    gram: &Array2<f64>,
    decoder: &Array2<f64>,
    n_eff: f64,
    p_out: f64,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<f64, String> {
    let m = gram.nrows();
    validate_rank_charge_problem(
        gram,
        decoder,
        n_eff,
        p_out,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )?;
    if m == 0 || n_eff == 0.0 {
        return Ok(0.0);
    }
    // MP reconstruction rank on the reconstruction Gram. U orthogonal ⇒ svd of
    // diag(√λ)·Uᵀ·D equals svd of the reconstruction square root G^½·D.
    let (evals, u) = gram
        .eigh(super::Side::Lower)
        .map_err(|e| format!("realised_rank_charge_dof: eigh(G): {e}"))?;
    let evals = certified_psd_spectrum(evals.view(), "rank-charge Gram")?;
    let mut scaled = u.t().dot(decoder);
    let cols = scaled.ncols();
    for i in 0..m {
        let s = evals[i].sqrt();
        for j in 0..cols {
            scaled[[i, j]] *= s;
        }
    }
    let sv = match scaled.svd(false, false) {
        Ok((_, sv, _)) => sv,
        Err(e) => return Err(format!("realised_rank_charge_dof: recon svd: {e}")),
    };
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p_out, r_floor)
        .map_err(|error| format!("realised_rank_charge_dof: {error}"))?;
    let mu = sv
        .iter()
        .map(|&singular_value| normalized_reconstruction_energy(singular_value, n_eff))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("realised_rank_charge_dof: {error}"))?;
    let rank = classify_reconstruction_rank(&mu, edge, r_floor);
    // RECONSTRUCTION RANK vs DEGENERACY (#2258 real-activation class). MP rank zero
    // conflated two regimes with opposite correct handling:
    //   · VANISHED decoder (a²‖B‖² → 0): the β-mode is degenerate, the
    //     β-Schur log-det → −∞ is the Laplace approximation BREAKING DOWN,
    //     and the RLCT argument above makes the categorical +∞ veto the only
    //     valid pricing. This is the regime the veto was built for (births on
    //     featureless residuals).
    //   · BELOW the MP RECONSTRUCTION-RANK EDGE but numerically alive: in a p ≈ n_eff
    //     regime the edge is ≈ 4R (measured on real GPT-2 activations:
    //     R=1.005, n_eff=84, p=64 ⇒ edge=3.53 vs top signal 0.32–0.84), so a
    //     genuine fitted decoder simply isn't separable from noise AT THIS
    //     SAMPLE SIZE. The Laplace value is perfectly valid there; vetoing
    //     turned every weak-signal PRIMARY fit into a hard refusal — the
    //     mechanism behind the 0/39 real-activation grid. Price such an atom
    //     at the MINIMUM non-degenerate rank (1): an alive decoder occupies
    //     at least a ray, the charge is strictly HIGHER than the rank-0 zero
    //     charge (so the birth-gate null-license stays closed — a
    //     featureless-residual birth must now pay ½·basis_edf·ln n it cannot
    //     earn), and the fit minted for the user carries its honest weak
    //     evidence instead of no model at all.
    if rank.mp_reconstruction_rank == 0 && rank.production_chargeable_rank == 1 {
        log::debug!(
            "realised_rank_charge_dof: below-reconstruction-rank-edge atom promoted to rank 1 — \
             top sv²/n_eff={:.6e} vs MP edge={edge:.6e} \
             (R={r_floor:.6e}, n_eff={n_eff:.3e}, p_out={p_out})",
            rank.top_signal
        );
    } else if rank.production_chargeable_rank == 0 {
        log::debug!(
            "realised_rank_charge_dof: VANISHED decoder (categorical veto upstream) — \
             top sv²/n_eff={:.6e} ≤ {RANK_VANISHED_REL:.0e}·R (R={r_floor:.6e}, \
             n_eff={n_eff:.3e}, p_out={p_out})",
            rank.top_signal
        );
    }
    // basis_edf = tr(gram·(gram+λS)⁻¹).
    let mut mmat = gram.clone();
    if let Some(pen) = smooth_penalty {
        for i in 0..m {
            for j in 0..m {
                mmat[[i, j]] += lam_smooth * pen[[i, j]];
            }
        }
    }
    let factor = mmat.cholesky(super::Side::Lower).map_err(|error| {
        format!("realised_rank_charge_dof: G + lambda*S is not positive definite: {error}")
    })?;
    let x = factor.solve_mat(gram); // X = (G+λS)⁻¹ G
    let raw_basis_edf = (0..m).map(|i| x[[i, i]]).sum::<f64>();
    let basis_edf = certified_basis_edf(raw_basis_edf, m, "realised_rank_charge_dof")?;
    Ok(rank.production_chargeable_rank as f64 * basis_edf)
}

/// Coordinate-block log-determinant `log|H_tt|` carried by an exact dense
/// arrow cache. The undamped row factors are the value operator used by the
/// rank-adjusted Laplace criterion, so a non-positive or non-finite diagonal is
/// an invalid factorization, not a term to skip.
pub(crate) fn coordinate_block_log_det(cache: &ArrowFactorCache) -> Result<f64, String> {
    let mut log_det_tt = 0.0_f64;
    for row in 0..cache.undamped_factor_count() {
        let factor = cache.undamped_factor(row);
        for diagonal in 0..factor.nrows() {
            let value = factor[[diagonal, diagonal]];
            if !(value.is_finite() && value > 0.0) {
                return Err(format!(
                    "coordinate_block_log_det: row {row} diagonal {diagonal} is {value}; \
                     the undamped coordinate factor is invalid"
                ));
            }
            log_det_tt += 2.0 * value.ln();
        }
    }
    Ok(log_det_tt)
}

/// The one production Laplace-complexity scalar:
///
/// `0.5 * log|H| - 0.5 * log|H_tt| +
///  sum_k 0.5 * d_eff_k * log(max(N_eff_k, 1))`.
///
/// Dense, streaming, and criterion-as-atoms assembly all call this function so
/// the value cannot retain the full coordinate logdet after the analytic
/// gradient has switched to the realised-rank charge. A zero realised rank is
/// the categorical Laplace-invalid branch and therefore yields positive
/// infinity, matching the production criterion contract.
pub(crate) fn rank_adjusted_quasi_laplace_complexity(
    log_det: f64,
    log_det_tt: f64,
    d_eff: &[f64],
    n_eff: &[f64],
) -> Result<f64, String> {
    if d_eff.len() != n_eff.len() {
        return Err(format!(
            "rank_adjusted_quasi_laplace_complexity: d_eff length {} does not match N_eff length {}",
            d_eff.len(),
            n_eff.len()
        ));
    }
    if d_eff.iter().any(|&value| value == 0.0) {
        return Ok(f64::INFINITY);
    }
    if !(log_det.is_finite() && log_det_tt.is_finite()) {
        return Err(format!(
            "rank_adjusted_quasi_laplace_complexity: non-finite logdet input \
             (joint={log_det}, coordinate={log_det_tt})"
        ));
    }
    let mut rank_charge = 0.0_f64;
    for (atom, (&dof, &occupancy)) in d_eff.iter().zip(n_eff.iter()).enumerate() {
        if !(dof.is_finite() && dof > 0.0) {
            return Err(format!(
                "rank_adjusted_quasi_laplace_complexity: atom {atom} has invalid positive realised DOF {dof}"
            ));
        }
        if !(occupancy.is_finite() && occupancy >= 0.0) {
            return Err(format!(
                "rank_adjusted_quasi_laplace_complexity: atom {atom} has invalid effective sample size {occupancy}"
            ));
        }
        rank_charge += 0.5 * dof * occupancy.max(1.0).ln();
    }
    let value = 0.5 * (log_det - log_det_tt) + rank_charge;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "rank_adjusted_quasi_laplace_complexity: assembled non-finite value {value}"
        ))
    }
}

// [#780] Softmax-entropy Gershgorin majorizer leaf helpers live in a sibling
// cohesive module, inlined here so they share this module scope.
include!("softmax_entropy_majorizer.rs");

// [#780] The exact stationarity-Jacobian correction and exact-Hessian solve
// methods live in a sibling file, inlined here so they share this `impl
// SaeManifoldTerm` / module scope while keeping this file under the line-count
// gate.
include!("construction_exact_hessian.rs");

// [#2253] Exact hard-rank-charge direct and implicit-response derivatives.
include!("construction_rank_charge_derivative.rs");

// [#780] The outer-gradient error taxonomy (`OuterGradientError`), the
// `ForcedRowLayout` override alias, the `COTRAIN_*` co-training weight
// constants, and the `AmortizedEncoderConsistency` report were extracted
// verbatim into the sibling `construction_aux_types` module to keep this file
// under the per-file line-count gate. They re-enter this module's scope via the
// parent's glob re-export (`use super::*;` above).

/// The undamped (ridge-0) deflated criterion factorization at an acceptance
/// iterate, packaged with the factorisation-independent KKT residual norms read
/// off the SAME assembled system. Produced by
/// [`SaeManifoldTerm::factor_deflated_evidence_with_grad_norms`] at the
/// objective-stall diagnostic point; the discarded Newton step
/// `(delta_t, delta_beta)` is retained only to report the affine Newton
/// decrement. A small decrement cannot replace the KKT acceptance gate.
pub(crate) struct DeflatedEvidenceFactor {
    pub(crate) delta_t: Array1<f64>,
    pub(crate) delta_beta: Array1<f64>,
    pub(crate) cache: ArrowFactorCache,
    pub(crate) grad_norm: f64,
    pub(crate) quotient_grad_norm: f64,
}

impl SaeManifoldTerm {
    #[must_use = "build error must be handled"]
    pub fn new(atoms: Vec<SaeManifoldAtom>, assignment: SaeAssignment) -> Result<Self, String> {
        if atoms.is_empty() {
            return Err("SaeManifoldTerm::new: at least one atom required".into());
        }
        let n = atoms[0].n_obs();
        let p = atoms[0].output_dim();
        if assignment.n_obs() != n || assignment.k_atoms() != atoms.len() {
            return Err(format!(
                "SaeManifoldTerm::new: assignment shape ({}, {}) does not match atoms ({n}, {})",
                assignment.n_obs(),
                assignment.k_atoms(),
                atoms.len()
            ));
        }
        for (k, atom) in atoms.iter().enumerate() {
            if atom.n_obs() != n {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} has n_obs={} but atom 0 has {n}",
                    atom.n_obs()
                ));
            }
            if atom.output_dim() != p {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} output_dim={} but atom 0 has {p}",
                    atom.output_dim()
                ));
            }
            if atom.latent_dim != assignment.coords[k].latent_dim() {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} latent_dim={} but assignment coord has {}",
                    atom.latent_dim,
                    assignment.coords[k].latent_dim()
                ));
            }
        }
        Ok(Self {
            atoms,
            assignment,
            chart_atlases: Vec::new(),
            temperature_schedule: None,
            last_row_layout: None,
            row_metric: None,
            data_row_reseed: false,
            // SAC — the collapse-guard stack is armed by default; the stagewise
            // K=1 lane disarms it explicitly (see the field docs on term.rs).
            guards_enabled: true,
            collapse_events: Vec::new(),
            row_loss_weights: None,
            crosscoder_pricing_spans: None,
            last_frames_active: false,
            assembly_chunk_override: None,
            fixed_decoder_assembly: false,
            border_hbb_workspace: Array2::<f64>::zeros((0, 0)),
            arrow_assembly_workspace: SaeArrowAssemblyWorkspace::default(),
            certificate_dispersion: None,
            curvature_walk_report: None,
            expected_criterion_gauge_deflated_directions: None,
            criterion_gauge_deflation_reanchors: 0,
            criterion_gauge_deflation_last_delta_sign: 0,
            dictionary_cocollapse_reseeds: 0,
            best_cocollapse_incumbent: None,
            best_fit_incumbent: None,
            structural_cocollapse_reseeds: 0,
            decoder_repulsion_gate: None,
            barrier_coactivation_gate: None,
            // #1801 — default false: the dense/full-batch assembly refreshes the
            // collapse-prevention gates per assembly (bit-for-bit historical). The
            // streaming fit driver re-arms this to freeze them once globally.
            streaming_gates_frozen: false,
            hybrid_split_report: None,
            atom_inner_fits: None,
            oos_linear_images: None,
            separation_barrier_strength_override: None,
            // Rung-2 behavioral block: default None (ordinary single-block term,
            // bit-for-bit unchanged). Attached via `set_behavior_block`.
            behavior: None,
            // Crosscoder stacked-column layout: default None (no multi-block
            // layout installed; `layer_decoder` errors until a multi-block fit or
            // `set_crosscoder_layout` records one). Bit-for-bit historical path.
            crosscoder_layout: None,
            // #2023 C4 — Tier-0 shared mean: default None (no de-meaning; the
            // historical path is bit-for-bit). Installed via `set_tier0_mean` /
            // `fit_tier0_mean`.
            tier0_mean: None,
            tier0_scale: None,
        })
    }

    /// Apply the FFI-facing [`SaeFitConfig`] as the source of truth for this fit.
    ///
    /// Distributes the config to its two authorities: the barrier strength override
    /// onto the term (read by `separation_barrier_strength`), and the ordered Beta--Bernoulli-α
    /// override onto the assignment (read by
    /// [`SaeAssignment::resolved_ordered_beta_bernoulli_alpha`]). A `None` field selects the canonical
    /// data-derived or assignment-mode default. Call this after building the term
    /// and before fitting; distinct terms remain isolated by construction.
    pub fn set_fit_config(&mut self, config: SaeFitConfig) {
        self.separation_barrier_strength_override = config.separation_barrier_strength_override;
        self.assignment.set_ordered_beta_bernoulli_alpha_override(
            config.ordered_beta_bernoulli_alpha_override,
        );
    }

    /// #1777 — the per-fit configuration currently in force on this term,
    /// reconstructed from its two authorities (the term's barrier override and the
    /// assignment's α override). Round-trips with [`Self::set_fit_config`].
    #[must_use]
    pub fn fit_config(&self) -> SaeFitConfig {
        SaeFitConfig {
            separation_barrier_strength_override: self.separation_barrier_strength_override,
            ordered_beta_bernoulli_alpha_override: self
                .assignment
                .ordered_beta_bernoulli_alpha_override,
        }
    }

    /// #2023 — merge two fitted terms (tier-1 linear bulk `primary` + tier-2
    /// curved `secondary`) into one whose atom set is `primary.atoms ++
    /// secondary.atoms`, for the final joint polish of the two-tier fit-order.
    /// Both must share `n_obs`, `output_dim`, and assignment-mode VARIANT.
    /// Concatenates in (primary, secondary) order: atoms; assignment logits
    /// (column hstack), coords, ungated; rho `log_lambda_smooth` and `log_ard`.
    /// The global sparsity ρ and ALL per-fit config (row_metric, row-loss
    /// weights, fit-config, data-row reseeding, temperature, and softmax
    /// cap, assignment mode) are carried from `primary`; `secondary`'s config is
    /// discarded. This asymmetry is deliberate: in the two-tier fit-order
    /// `primary` is the linear/bulk tier that defines the fit's global regime —
    /// it owns the sparse-penalty scale (`log_lambda_sparse`), the observation
    /// `row_metric` / row-loss weighting (the whitening the curved tier is fit
    /// *against*), and the fit-config (barrier / ordered Beta--Bernoulli-α). The curved `secondary`
    /// tier is fit on the whitened residual under that same regime, so it
    /// contributes only its per-atom parameters (atoms, coords, ungated,
    /// per-atom `log_lambda_smooth` / `log_ard`); its globals are byproducts of
    /// the residual sub-problem and must not overwrite the bulk tier's. K-
    /// dependent / per-assembly transient state (row layout, frame flag, border
    /// workspace, frozen routing, repulsion/coactivation gates, co-collapse /
    /// gauge-deflation bookkeeping) is RESET — it is rebuilt at the next assembly.
    ///
    /// This primitive is intentionally MODE-GENERAL: structural concatenation is
    /// well-defined for any assignment mode, so the only mode check here is
    /// variant-equality between tiers. The restriction that two-tier fit-order
    /// applies only to independent-gate modes lives at the orchestration layer,
    /// not in this merge — see below.
    ///
    /// Fitted-additivity `merged.fitted() == primary.fitted() + secondary.fitted()`
    /// holds EXACTLY for independent-gate modes (ThresholdGate / ordered Beta--Bernoulli, where each atom's
    /// gate is computed independently); under Softmax the gate re-normalizes over
    /// the merged `K`, so the merge is a WARM START into the joint objective (the
    /// two-tier driver's final joint polish reconciles it).
    pub fn merge_tiers(
        mut primary: SaeManifoldTerm,
        primary_rho: &SaeManifoldRho,
        secondary: SaeManifoldTerm,
        secondary_rho: &SaeManifoldRho,
    ) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
        let n = primary.n_obs();
        let p = primary.output_dim();
        let k1 = primary.k_atoms();
        let k2 = secondary.k_atoms();
        if secondary.n_obs() != n {
            return Err(format!(
                "SaeManifoldTerm::merge_tiers: n_obs mismatch: {n} vs {}",
                secondary.n_obs()
            ));
        }
        if secondary.output_dim() != p {
            return Err(format!(
                "SaeManifoldTerm::merge_tiers: output_dim mismatch: {p} vs {}",
                secondary.output_dim()
            ));
        }
        if std::mem::discriminant(&primary.assignment.mode)
            != std::mem::discriminant(&secondary.assignment.mode)
        {
            return Err(
                "SaeManifoldTerm::merge_tiers: assignment-mode variant mismatch between tiers"
                    .to_string(),
            );
        }
        if primary_rho.log_lambda_smooth.len() != k1
            || secondary_rho.log_lambda_smooth.len() != k2
            || primary_rho.log_ard.len() != k1
            || secondary_rho.log_ard.len() != k2
        {
            return Err(format!(
                "SaeManifoldTerm::merge_tiers: rho per-atom lengths (smooth {}/{}, ard {}/{}) \
                 must equal K1/K2 = {k1}/{k2}",
                primary_rho.log_lambda_smooth.len(),
                secondary_rho.log_lambda_smooth.len(),
                primary_rho.log_ard.len(),
                secondary_rho.log_ard.len()
            ));
        }
        // Symmetric per-atom guard on the ASSIGNMENT side: coords / ungated must be
        // one entry per atom in each tier, or the concatenation below silently
        // desynchronizes the atom↔coord↔gate correspondence.
        if primary.assignment.coords.len() != k1
            || secondary.assignment.coords.len() != k2
            || primary.assignment.ungated.len() != k1
            || secondary.assignment.ungated.len() != k2
        {
            return Err(format!(
                "SaeManifoldTerm::merge_tiers: assignment per-atom lengths (coords {}/{}, \
                 ungated {}/{}) must equal K1/K2 = {k1}/{k2}",
                primary.assignment.coords.len(),
                secondary.assignment.coords.len(),
                primary.assignment.ungated.len(),
                secondary.assignment.ungated.len()
            ));
        }
        // Assignment: column-hstack logits (n×K1 | n×K2), append per-atom coords
        // and ungated flags. Carries primary's mode + ordered_beta_bernoulli_alpha_override.
        let mut logits = Array2::<f64>::zeros((n, k1 + k2));
        logits
            .slice_mut(s![.., 0..k1])
            .assign(&primary.assignment.logits);
        logits
            .slice_mut(s![.., k1..k1 + k2])
            .assign(&secondary.assignment.logits);
        primary.assignment.logits = logits;
        primary
            .assignment
            .coords
            .extend(secondary.assignment.coords);
        primary
            .assignment
            .ungated
            .extend(secondary.assignment.ungated);
        primary.assignment.frozen_logits = None;
        // Atoms and first-class chart atlases.  Secondary atlas chart indices
        // are local to its atom vector, so shift them by the primary width
        // before appending; primary indices are unchanged.
        let mut secondary_atlases = secondary.chart_atlases;
        for atlas in &mut secondary_atlases {
            atlas.shift_indices(k1);
        }
        primary.atoms.extend(secondary.atoms);
        primary.chart_atlases.extend(secondary_atlases);
        // Reset K-dependent / per-assembly transient state (rebuilt next assembly).
        primary.last_row_layout = None;
        primary.last_frames_active = false;
        primary.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
        primary.decoder_repulsion_gate = None;
        primary.barrier_coactivation_gate = None;
        // Evidence-gauge / co-collapse cluster — the canonical reset (mirrors
        // outer_objective.rs and the ctor) clears all FIVE fields together: the
        // reanchor count and last-delta sign feed the penalized_quasi_laplace_criterion reversal-
        // budget loop, so carrying `primary`'s stale tier-1 values would either
        // spuriously flag a reversal on the merged term's FIRST deflation step or
        // start the joint polish with a partially-consumed budget (erroring
        // earlier than a fresh fit on an ill-conditioned tier-1).
        primary.expected_criterion_gauge_deflated_directions = None;
        primary.criterion_gauge_deflation_reanchors = 0;
        primary.criterion_gauge_deflation_last_delta_sign = 0;
        primary.dictionary_cocollapse_reseeds = 0;
        primary.best_cocollapse_incumbent = None;
        primary.best_fit_incumbent = None;
        primary.structural_cocollapse_reseeds = 0;
        // Stale tier-1 diagnostics — rebuilt at the next assembly / post-fit pass.
        primary.collapse_events = Vec::new();
        primary.curvature_walk_report = None;
        // Rho: global sparsity from primary; per-atom smoothness + ARD concatenated.
        let mut rho = primary_rho.clone();
        rho.log_lambda_smooth
            .extend_from_slice(&secondary_rho.log_lambda_smooth);
        rho.log_ard.extend(secondary_rho.log_ard.iter().cloned());
        rho = rho.for_assignment(primary.assignment.mode);
        Ok((primary, rho))
    }

    /// Gather a `Vec` into a new order without cloning: `out[new] = items[order[new]]`.
    /// `order` MUST be a permutation of `0..items.len()` (each source index visited
    /// exactly once); the caller [`Self::reorder_atoms`] validates that first.
    fn gather_by_order<T>(items: Vec<T>, order: &[usize]) -> Vec<T> {
        let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
        order
            .iter()
            .map(|&src| {
                slots[src]
                    .take()
                    .expect("reorder_atoms: order must visit each source index exactly once")
            })
            .collect()
    }

    /// #2023 — permute this term's atoms (and the paired `rho`) into a new order:
    /// the atom currently at `order[i]` moves to final position `i`
    /// (`new[i] = old[order[i]]`, a gather). Used by the two-tier fit-order to
    /// restore the CALLER's atom order after [`Self::merge_tiers`] concatenates the
    /// linear (primary) and curved (secondary) tiers — merge yields
    /// linear++curved order, and this scatters each atom back to its original
    /// input index so the entire downstream (joint polish, into_fitted,
    /// shape-uncertainty, structured passes, and every by-original-index
    /// serialization read) sees the caller's order with zero further changes.
    ///
    /// Permutes, in lockstep: atoms; assignment logit COLUMNS; per-atom coords and
    /// ungated flags; and the paired `rho`'s `log_lambda_smooth` / `log_ard`. The
    /// global sparsity ρ and the assignment mode are order-independent and left
    /// untouched. Atom NAMES travel with their atom (the caller renames tiers to
    /// their input indices before merging, so after this the names read
    /// `atom_0..atom_{K-1}` in caller order — identical to a single-tier build).
    /// K-dependent transient state that encodes the OLD column order (row layout,
    /// frame flag, border workspace, frozen routing) is reset — rebuilt at the
    /// next assembly (the joint polish).
    ///
    /// `order` must be a permutation of `0..K` and `rho` must carry `K` per-atom
    /// entries, or this errs without mutating anything observable downstream.
    pub fn reorder_atoms(
        &mut self,
        order: &[usize],
        rho: &mut SaeManifoldRho,
    ) -> Result<(), String> {
        let k = self.k_atoms();
        if order.len() != k {
            return Err(format!(
                "SaeManifoldTerm::reorder_atoms: order length {} must equal K={k}",
                order.len()
            ));
        }
        // Validate `order` is a permutation of 0..K (every index present once).
        let mut seen = vec![false; k];
        for &src in order {
            let slot = seen.get_mut(src).ok_or_else(|| {
                format!("SaeManifoldTerm::reorder_atoms: order index {src} out of range 0..{k}")
            })?;
            if *slot {
                return Err(format!(
                    "SaeManifoldTerm::reorder_atoms: order index {src} repeated (not a permutation)"
                ));
            }
            *slot = true;
        }
        if rho.log_lambda_smooth.len() != k || rho.log_ard.len() != k {
            return Err(format!(
                "SaeManifoldTerm::reorder_atoms: rho per-atom lengths (smooth {}, ard {}) \
                 must equal K={k}",
                rho.log_lambda_smooth.len(),
                rho.log_ard.len()
            ));
        }
        // Assignment logit COLUMNS: new column i is old column order[i].
        let n = self.n_obs();
        let mut new_logits = Array2::<f64>::zeros((n, k));
        for (new_j, &old_j) in order.iter().enumerate() {
            new_logits
                .column_mut(new_j)
                .assign(&self.assignment.logits.column(old_j));
        }
        self.assignment.logits = new_logits;
        // Per-atom Vecs (atoms / coords / ungated) and the paired rho blocks.
        let atoms = std::mem::take(&mut self.atoms);
        self.atoms = Self::gather_by_order(atoms, order);
        // Atlas endpoints are atom indices.  `order[new] = old`, so invert the
        // gather permutation to obtain old -> new and remap every seam.
        let mut old_to_new = vec![None; k];
        for (new, &old) in order.iter().enumerate() {
            old_to_new[old] = Some(new);
        }
        for atlas in &mut self.chart_atlases {
            atlas.remap(&old_to_new)?;
        }
        let coords = std::mem::take(&mut self.assignment.coords);
        self.assignment.coords = Self::gather_by_order(coords, order);
        let ungated = std::mem::take(&mut self.assignment.ungated);
        self.assignment.ungated = Self::gather_by_order(ungated, order);
        let smooth = std::mem::take(&mut rho.log_lambda_smooth);
        rho.log_lambda_smooth = Self::gather_by_order(smooth, order);
        let ard = std::mem::take(&mut rho.log_ard);
        rho.log_ard = Self::gather_by_order(ard, order);
        // Reset K-ordered transient state that encoded the OLD column order.
        self.assignment.frozen_logits = None;
        self.last_row_layout = None;
        self.last_frames_active = false;
        self.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
        Ok(())
    }

    /// Install the fitted reconstruction dispersion used by
    /// [`dictionary_incoherence_report`]. This is a pure diagnostic scalar and
    /// does not feed any loss, criterion, penalty, or optimizer state.
    pub fn set_certificate_dispersion(&mut self, dispersion: f64) -> Result<(), String> {
        if !dispersion.is_finite() || dispersion <= 0.0 {
            return Err(format!(
                "SaeManifoldTerm::set_certificate_dispersion: dispersion must be finite and positive, got {dispersion}"
            ));
        }
        self.certificate_dispersion = Some(dispersion);
        Ok(())
    }

    /// Harvest the per-atom inner-decoder-smooth byproducts (#1097 / #1103) the
    /// residual-gauge certificate's post-PIRLS atom inference reports consume.
    ///
    /// This is the post-fit harness seam: it needs the reconstruction target `Z`
    /// (`target`) and the fitted dispersion `φ` (`dispersion`), both available
    /// only after the joint fit converges and the engine has discarded `Z` from
    /// the objective. For each atom `k` it captures the Gaussian-identity
    /// penalized smooth of the atom's leading decoder output channel `j`
    /// (largest column 2-norm of `B_k`) against its partial residual
    /// `e_{i} = z_i − fitted_i + a_{ik} g_k(t_i)` on channel `j`, holding all
    /// other atoms and the assignment fixed at the fitted optimum — exactly the
    /// fixed snapshot ([`crate::identifiability::AtomInnerFit`]) the Riesz
    /// debiasing and split-LRT smooth-structure e-value read.
    ///
    /// A pure read of the fitted state: it mutates only the diagnostic
    /// `atom_inner_fits` field, never a loss / criterion / penalty / optimizer
    /// state. Atoms with no active rows or a degenerate (rank-deficient,
    /// non-SPD) inner Hessian get a `None` slot — the genuine prerequisite (an
    /// SPD penalized inner Hessian on a non-empty active set) is absent there.
    pub fn set_atom_inner_fits(
        &mut self,
        target: ArrayView2<'_, f64>,
        dispersion: f64,
    ) -> Result<(), String> {
        if !dispersion.is_finite() || dispersion <= 0.0 {
            return Err(format!(
                "SaeManifoldTerm::set_atom_inner_fits: dispersion must be finite and positive, got {dispersion}"
            ));
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::set_atom_inner_fits: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }

        // #1026 — `atom_inner_fits` is a pure diagnostic; skip its dense (N×K×P)
        // tensor (~256 GiB at K=32768,P=32) past a cell ceiling — all-None slots,
        // never OOM. The fit is unaffected; only this audit field is absent.
        if n.saturating_mul(k_atoms).saturating_mul(p) > 64_000_000 {
            self.atom_inner_fits = Some((0..k_atoms).map(|_| None).collect());
            return Ok(());
        }

        // Settled per-row assignments and per-(row, atom) decoded outputs, so the
        // per-atom partial residual is `e_k = (z − fitted) + a_k decoded_k`.
        let mut assignments = Vec::with_capacity(n);
        for row in 0..n {
            assignments.push(self.assignment.try_assignments_row(row)?);
        }
        let mut decoded = Array3::<f64>::zeros((n, k_atoms, p));
        let mut dbuf = vec![0.0_f64; p];
        for row in 0..n {
            for atom_idx in 0..k_atoms {
                self.atoms[atom_idx].fill_decoded_row(row, &mut dbuf);
                for c in 0..p {
                    decoded[[row, atom_idx, c]] = dbuf[c];
                }
            }
        }
        let mut fitted = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for atom_idx in 0..k_atoms {
                let a = assignments[row][atom_idx];
                if a == 0.0 {
                    continue;
                }
                for c in 0..p {
                    fitted[[row, c]] += a * decoded[[row, atom_idx, c]];
                }
            }
        }

        let mut inner_fits: Vec<Option<crate::identifiability::AtomInnerFit>> =
            Vec::with_capacity(k_atoms);
        for atom_idx in 0..k_atoms {
            inner_fits.push(self.build_atom_inner_fit(
                atom_idx,
                target,
                &assignments,
                decoded.view(),
                fitted.view(),
                dispersion,
            )?);
        }
        self.atom_inner_fits = Some(inner_fits);
        Ok(())
    }

    /// Build one atom's fixed inner-smooth snapshot for the post-PIRLS atom
    /// inference reports, or `None` when the atom has no active rows or the
    /// penalized inner Hessian is not SPD. Returns `Err` only on a structural
    /// inconsistency (shape mismatch), never on a benign degenerate atom.
    pub(crate) fn build_atom_inner_fit(
        &self,
        atom_idx: usize,
        target: ArrayView2<'_, f64>,
        assignments: &[Array1<f64>],
        decoded: ArrayView3<'_, f64>,
        fitted: ArrayView2<'_, f64>,
        dispersion: f64,
    ) -> Result<Option<crate::identifiability::AtomInnerFit>, String> {
        let atom = &self.atoms[atom_idx];
        let n = atom.n_obs();
        let m = atom.basis_size();
        let p = atom.output_dim();
        if m == 0 || p == 0 {
            return Ok(None);
        }

        // Leading decoder output channel j = argmax_j ‖B_k[:, j]‖, the channel
        // that carries the atom's signal.
        let mut j_lead = 0usize;
        let mut best_norm = -1.0_f64;
        for col in 0..p {
            let mut norm = 0.0_f64;
            for r in 0..m {
                let v = atom.decoder_coefficients[[r, col]];
                norm += v * v;
            }
            if norm > best_norm {
                best_norm = norm;
                j_lead = col;
            }
        }
        let beta = atom.decoder_coefficients.column(j_lead).to_owned();

        // Active rows: a_{ik} > 0.
        let active: Vec<usize> = (0..n)
            .filter(|&row| assignments[row][atom_idx] > 0.0)
            .collect();
        let n_active = active.len();
        // The penalized smooth needs at least as many active rows as it has
        // basis columns to give a non-degenerate data Gram; below that the inner
        // fit's SPD prerequisite is genuinely unmet.
        if n_active == 0 {
            return Ok(None);
        }

        let mut design = Array2::<f64>::zeros((n_active, m));
        let mut derivative_design = Array2::<f64>::zeros((n_active, m));
        let mut row_scores = Array2::<f64>::zeros((n_active, m));
        let mut weights = Array1::<f64>::zeros(n_active);
        for (slot, &row) in active.iter().enumerate() {
            let a_ik = assignments[row][atom_idx];
            let w_i = a_ik * a_ik;
            weights[slot] = w_i;
            for col in 0..m {
                design[[slot, col]] = atom.basis_values[[row, col]];
                // Leading latent axis (axis 0) is the atom's primary coordinate;
                // it is the one the average-derivative functional integrates.
                derivative_design[[slot, col]] = atom.basis_jacobian[[row, col, 0]];
            }
            // Partial residual on channel j, then the inner-smooth working
            // response z_i = e_i / a_ik so that w_i (z_i − Φᵀβ) = a_ik r_i.
            let e_i = target[[row, j_lead]] - fitted[[row, j_lead]]
                + a_ik * decoded[[row, atom_idx, j_lead]];
            let mu_hat = design.row(slot).dot(&beta);
            let z_i = e_i / a_ik;
            let res_i = z_i - mu_hat;
            // Gaussian-identity score s_i = −w_i res_i Φ_i / φ.
            let scale = -w_i * res_i / dispersion;
            for col in 0..m {
                row_scores[[slot, col]] = scale * design[[slot, col]];
            }
        }

        // Penalized inner Hessian H = ΦᵀWΦ + S̃_k.
        let mut xtwx = Array2::<f64>::zeros((m, m));
        for slot in 0..n_active {
            let w_i = weights[slot];
            for a in 0..m {
                let xa = design[[slot, a]];
                if xa == 0.0 {
                    continue;
                }
                for b in 0..m {
                    xtwx[[a, b]] += w_i * xa * design[[slot, b]];
                }
            }
        }
        let penalty = atom.smooth_penalty.clone();
        if penalty.dim() != (m, m) {
            return Err(format!(
                "build_atom_inner_fit: atom {atom_idx} smooth penalty {:?} != ({m}, {m})",
                penalty.dim()
            ));
        }
        let penalized_hessian = &xtwx + &penalty;

        // SPD prerequisite: the inner penalized Hessian must factor, else the
        // atom's inner-smooth fit is degenerate and no report is producible.
        if penalized_hessian.cholesky(Side::Lower).is_err() {
            return Ok(None);
        }

        // Peak (largest fitted |g_k| on channel j) and mode (largest assignment
        // mass) design rows, over the active set.
        let mut peak_slot = 0usize;
        let mut peak_val = -1.0_f64;
        let mut mode_slot = 0usize;
        let mut mode_mass = -1.0_f64;
        for (slot, &row) in active.iter().enumerate() {
            let g_val = design.row(slot).dot(&beta).abs();
            if g_val > peak_val {
                peak_val = g_val;
                peak_slot = slot;
            }
            let mass = assignments[row][atom_idx];
            if mass > mode_mass {
                mode_mass = mass;
                mode_slot = slot;
            }
        }
        let peak_design_row = design.row(peak_slot).to_owned();
        let mode_design_row = design.row(mode_slot).to_owned();

        Ok(Some(crate::identifiability::AtomInnerFit {
            design,
            derivative_design,
            beta,
            penalty,
            penalized_hessian,
            row_scores,
            weights,
            dispersion,
            peak_design_row,
            mode_design_row,
        }))
    }

    /// Profile the Gaussian reconstruction dispersion at the current seed
    /// state. This is the scale used to make SAE penalty seeds dimensionless
    /// before the outer rho search starts.
    pub fn seed_reconstruction_dispersion(
        &self,
        target: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let fitted = self.try_fitted()?;
        if fitted.dim() != target.dim() {
            return Err(format!(
                "SaeManifoldTerm::seed_reconstruction_dispersion: fitted {:?} != target {:?}",
                fitted.dim(),
                target.dim()
            ));
        }
        let n_scalar = (target.nrows() * target.ncols()).max(1) as f64;
        let mut rss = 0.0_f64;
        for row in 0..target.nrows() {
            for col in 0..target.ncols() {
                let r = target[[row, col]] - fitted[[row, col]];
                rss += r * r;
            }
        }
        if !rss.is_finite() || rss < 0.0 {
            return Err(format!(
                "SaeManifoldTerm::seed_reconstruction_dispersion: non-finite seed RSS {rss}"
            ));
        }
        Ok((rss / n_scalar).max(SAE_SEED_DISPERSION_FLOOR))
    }

    /// Install per-row design honesty weights (#991) — the `1/π` inclusion
    /// corrections of a designed corpus subsample (see the field docs on
    /// `row_loss_weights` for exactly where they enter the objective).
    ///
    /// Weights must be finite and nonnegative, with positive total mass and one
    /// value per term row. Exact zeros represent rows excluded by a designed
    /// estimation split; no numerical epsilon is substituted for zero. They
    /// are self-normalized to mean `1.0` here (only the *relative* design
    /// correction matters at the fitted sample size; the absolute `n/budget`
    /// scale would silently inflate the dispersion estimate against the
    /// sample-sized dof). Weights that are identically equal after
    /// normalization (an exact full pass, or any uniform design) are stored
    /// as `None`, so the unweighted path stays bit-for-bit identical rather
    /// than "multiplied by 1.0".
    pub fn set_row_loss_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
        // The reciprocal of `with_crosscoder_blocks`'s refusal: block pricing
        // snapshots a full-N pristine copy and prices the Jacobian at the full
        // row count, so engaging a row subsample AFTER pricing is installed
        // would desync the two silently (#2231 stage-1 deferral, both ways).
        if self.crosscoder_pricing_spans.is_some() {
            return Err(
                "SaeManifoldTerm::set_row_loss_weights: crosscoder block pricing is installed; \
                 the #991 row-subsample and block pricing are mutually exclusive (stage 1)"
                    .to_string(),
            );
        }
        if weights.len() != self.n_obs() {
            return Err(format!(
                "SaeManifoldTerm::set_row_loss_weights: {} weights for {} rows",
                weights.len(),
                self.n_obs()
            ));
        }
        if weights.is_empty() {
            self.row_loss_weights = None;
            return Ok(());
        }
        if !weights.iter().all(|w| w.is_finite() && *w >= 0.0) || !weights.iter().any(|w| *w > 0.0)
        {
            return Err(
                "SaeManifoldTerm::set_row_loss_weights: weights must be finite, nonnegative, \
                 and contain positive total mass"
                    .to_string(),
            );
        }
        let first = weights[0];
        if weights.iter().all(|w| *w == first) {
            // Uniform design (full pass, or flat measure): the normalized
            // weight is exactly 1 everywhere — take the unweighted path.
            self.row_loss_weights = None;
            return Ok(());
        }
        let mean = weights.iter().sum::<f64>() / weights.len() as f64;
        self.row_loss_weights = Some(weights.into_iter().map(|w| w / mean).collect());
        Ok(())
    }

    /// The installed (mean-1 normalized) design honesty weights, `None` on the
    /// exact unweighted path.
    pub fn row_loss_weights(&self) -> Option<&[f64]> {
        self.row_loss_weights.as_deref()
    }

    /// Drop any installed per-row reconstruction weights, returning the term to
    /// the exact unweighted (full-pass) path. Used by the #997 structure-search
    /// wiring to clear the internal estimation/evaluation mask off the adopted
    /// term before the payload reconstruction is read over all rows.
    pub fn clear_row_loss_weights(&mut self) {
        self.row_loss_weights = None;
    }

    /// Huber-style OUTLIER-ROBUST per-row weights from the target activation
    /// norms — the missing default *policy* for the existing
    /// [`set_row_loss_weights`](Self::set_row_loss_weights) mechanism.
    ///
    /// The SAE fits unweighted least squares, which weights each token by its
    /// squared residual ∝ `‖z_i‖²`. On real LLM residual streams the per-token
    /// norm distribution is heavy-tailed (e.g. an OLMo mixed-layer slice has
    /// `p99/median ≈ 4.7`), so a small **coherent** cluster of high-norm tokens —
    /// typically special / attention-sink tokens, not semantic content —
    /// dominates the objective (measured: the top 5% of tokens carry ~31% of the
    /// total `‖z‖²` budget) and pulls dictionary atoms toward their direction.
    /// Mean-centering does NOT address this (it is per-feature, not per-token).
    ///
    /// This returns Huber weights `w_i = min(1, δ·m / ‖z_i‖)` where `m` is the
    /// MEDIAN token norm: tokens at or below `δ·m` keep full weight, higher-norm
    /// tokens are downweighted so their objective share grows only LINEARLY (not
    /// quadratically) with norm. `δ` is the robustness knob (`δ=1` thresholds at
    /// the median; larger `δ` only touches the extreme tail). The result is
    /// mean-normalized (overall objective scale preserved). OPT-IN: the caller
    /// installs it via `set_row_loss_weights` — the default fit is unchanged.
    pub fn robust_norm_row_weights(
        target: ArrayView2<'_, f64>,
        delta: f64,
    ) -> Result<Vec<f64>, String> {
        if !(delta.is_finite() && delta > 0.0) {
            return Err(format!(
                "robust_norm_row_weights: delta must be finite and positive; got {delta}"
            ));
        }
        let n = target.nrows();
        if n == 0 {
            return Ok(Vec::new());
        }
        let norms: Vec<f64> = (0..n)
            .map(|i| {
                let r = target.row(i);
                r.dot(&r).sqrt()
            })
            .collect();
        let mut sorted = norms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Median token norm (lower-median for even n; floored off zero so an
        // all-zero/degenerate slice yields uniform weights instead of NaN).
        let median = sorted[n / 2].max(f64::MIN_POSITIVE);
        let thresh = delta * median;
        let raw: Vec<f64> = norms
            .iter()
            .map(|&nm| if nm <= thresh { 1.0 } else { thresh / nm })
            .collect();
        let mean = raw.iter().sum::<f64>() / n as f64;
        if !(mean.is_finite() && mean > 0.0) {
            return Err("robust_norm_row_weights: degenerate weight normalizer".to_string());
        }
        Ok(raw.into_iter().map(|w| w / mean).collect())
    }

    /// Install the single per-row [`RowMetric`](gam_problem::RowMetric)
    /// that both the reconstruction likelihood and the isometry gauge read.
    /// Installing per-row output-Fisher factors here flips the provenance to
    /// `OutputFisher` *and* is the only way the gauge acquires a non-identity
    /// weight, so the two inner products cannot diverge. Passing a Euclidean
    /// metric (or never calling this) keeps the bit-identical isotropic path.
    ///
    /// The metric's row count and output dimension must match the term.
    pub fn set_row_metric(&mut self, metric: gam_problem::RowMetric) -> Result<(), String> {
        if metric.n_rows() != self.n_obs() {
            return Err(format!(
                "SaeManifoldTerm::set_row_metric: metric has {} rows but term has {}",
                metric.n_rows(),
                self.n_obs()
            ));
        }
        if metric.p_out() != self.output_dim() {
            return Err(format!(
                "SaeManifoldTerm::set_row_metric: metric output dim {} but term has {}",
                metric.p_out(),
                self.output_dim()
            ));
        }
        self.row_metric = Some(metric);
        Ok(())
    }

    /// #2023 C4 — install a Tier-0 shared mean μ (the manifold analogue of
    /// [`crate::tiered::Tier0Mean`]). Once set, [`Self::try_fitted_with_rho`] adds
    /// μ back to the assembled per-atom reconstruction, so the atoms only ever
    /// need to explain the DE-MEANED target `Z − μ`. Pass a length-`p` vector;
    /// mismatched length is rejected. Passing the column-mean of the fit target
    /// (see [`Self::fit_tier0_mean`]) moves the global DC out of the K per-atom
    /// intercepts into ONE shared mean — structurally removing the
    /// co-collapse-to-mean incentive (a pure DC-constant decoder then reconstructs
    /// a constant that the de-meaned target no longer contains, so it earns zero
    /// EV and is priced at realised rank 0 by the rank charge — unrepresentable as
    /// a survivor by construction).
    pub fn set_tier0_mean(&mut self, mean: Array1<f64>) -> Result<(), String> {
        let p = self.output_dim();
        if mean.len() != p {
            return Err(format!(
                "SaeManifoldTerm::set_tier0_mean: mean length {} must equal output_dim {p}",
                mean.len()
            ));
        }
        if !mean.iter().all(|v| v.is_finite()) {
            return Err("SaeManifoldTerm::set_tier0_mean: mean must be finite".to_string());
        }
        self.tier0_mean = Some(mean);
        Ok(())
    }

    /// #2023 C4 — the installed Tier-0 shared mean, or `None` on the historical
    /// (no-de-meaning) path. Round-trips with [`Self::set_tier0_mean`].
    pub fn tier0_mean(&self) -> Option<&Array1<f64>> {
        self.tier0_mean.as_ref()
    }

    /// Tier-0 per-column scale σ (input standardization). The fit runs on
    /// `(Z − μ)/σ`; every reconstruction lifts back `x̂ = μ + σ ⊙ x̂_internal`.
    /// Standardization is a CONDITIONING fix at the model level: with no column
    /// equilibration anywhere in the fit path, raw activation targets carry
    /// measured column-norm spreads of ~1e4 (joint Hessian κ ≈ 1e8, #2015),
    /// which sets the linear contraction rate of the majorized inner solver —
    /// the direct driver of the "~1e3 iterations then refusal" wall. Each σ_c
    /// must be finite and positive.
    pub fn set_tier0_scale(&mut self, scale: Array1<f64>) -> Result<(), String> {
        let p = self.output_dim();
        if scale.len() != p {
            return Err(format!(
                "SaeManifoldTerm::set_tier0_scale: scale length {} must equal output_dim {p}",
                scale.len()
            ));
        }
        if !scale.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(
                "SaeManifoldTerm::set_tier0_scale: scale must be finite and positive".to_string(),
            );
        }
        self.tier0_scale = Some(scale);
        Ok(())
    }

    /// The installed Tier-0 per-column scale, or `None` on the historical
    /// (unstandardized) path. Round-trips with [`Self::set_tier0_scale`].
    pub fn tier0_scale(&self) -> Option<&Array1<f64>> {
        self.tier0_scale.as_ref()
    }

    /// #2023 C4 — lift an assembled `Σ_k a_k g_k` reconstruction from the
    /// internal (standardized, de-meaned) frame back to raw-target space, in
    /// place: `x̂ ← μ + σ ⊙ x̂`. A strict no-op on the historical path
    /// (`tier0_mean == None`, `tier0_scale == None`), so every reconstruction
    /// entry point can call it unconditionally and stay bit-for-bit unchanged
    /// when Tier-0 is inactive. The scale multiplies BEFORE the mean adds —
    /// the fit frame is `(Z − μ)/σ`, so the inverse is `σ·x̂ + μ`.
    pub(crate) fn add_tier0_mean_inplace(&self, out: &mut Array2<f64>) {
        if let Some(scale) = self.tier0_scale.as_ref() {
            for mut out_row in out.rows_mut() {
                for (out_col, s) in out_row.iter_mut().zip(scale.iter()) {
                    *out_col *= *s;
                }
            }
        }
        if let Some(mean) = self.tier0_mean.as_ref() {
            for mut out_row in out.rows_mut() {
                for (out_col, m) in out_row.iter_mut().zip(mean.iter()) {
                    *out_col += *m;
                }
            }
        }
    }

    /// #2023 C4 — fit the Tier-0 shared mean as the column mean of the fit target
    /// `Z` (`N×P`), install it on the term, and return the DE-MEANED target
    /// `Z − μ` the atoms should be fit against. This is the single seam a driver
    /// calls before the joint fit so the global DC is carried by Tier-0 and the
    /// atoms chase only structure. The mean is the TRAIN-split mean: hold it fixed
    /// and reuse it for out-of-sample de-meaning and the EV baseline so held-out
    /// EV is measured against the same Tier-0 constant (no full-data leak).
    ///
    /// DOUBLE-SUBTRACTION HAZARD: exactly ONE stage may own the mean. If an
    /// upstream data-prep step already centers the target (e.g. the COMPOSE L17
    /// driver's `tier0.json` mean/scale), the term must NOT also de-mean — leave
    /// `tier0_mean` at `None` (the default), which is CORRECT for already-centered
    /// data. Only call this on RAW (un-centered) targets, where the term takes
    /// ownership of the mean.
    pub fn fit_tier0_mean(&mut self, z: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let p = self.output_dim();
        if z.ncols() != p {
            return Err(format!(
                "SaeManifoldTerm::fit_tier0_mean: target has P={} but output_dim is {p}",
                z.ncols()
            ));
        }
        if z.nrows() == 0 {
            return Err("SaeManifoldTerm::fit_tier0_mean: empty target".to_string());
        }
        let mean = z.mean_axis(ndarray::Axis(0)).ok_or_else(|| {
            "SaeManifoldTerm::fit_tier0_mean: mean_axis returned None".to_string()
        })?;
        let demeaned = &z - &mean.view().insert_axis(ndarray::Axis(0));
        self.set_tier0_mean(mean)?;
        Ok(demeaned)
    }

    /// #5/(B) — per-atom realised-rank effective DOF for the rank-charge criterion:
    /// `d_eff_k = rank_eff_k · basis_edf_k`, where
    ///   * `rank_eff_k` = the Marchenko–Pastur HARD count of the atom's realised
    ///     output rank: the number of per-atom reconstruction-Gram eigenvalues
    ///     (`(1/N_eff)·BᵀB`, `B = diag(a_k)·Φ_k·D_k`, `N_eff = Σ_row a_k²`) above
    ///     the DERIVED bulk edge `R·(1+√(p/N_eff))²` (`R = dispersion_r`, the
    ///     residual variance). Exactly 2 for a rank-2 circle; 0 for a decoder
    ///     collapsing to `‖B‖→0` (every eigenvalue → 0 ≪ edge) → charge 0 →
    ///     neutral (the co-collapse fix). The edge is parameter-free (NOT a
    ///     self-relative `ε·max_sv`): pure output noise cannot exceed it, so an
    ///     eigenvalue above it is identified signal. [#1893]
    ///   * `basis_edf_k = tr(G_k · (G_k + λ_k S_k)⁻¹)` on the atom's `m×m`
    ///     decoder data Gram (its identified basis dimension, ~m minus the
    ///     smoothness/DC shrinkage).
    /// The charge `½·d_eff_k·log n` is the honest BIC on the atom's realised
    /// decoder parameters. It is ROTATION-INVARIANT (rank + basis EDF are), so it
    /// does NOT distinguish a clean circle from a blend (both rank-2) — the
    /// producer owns cleanliness.
    pub(crate) fn per_atom_realised_rank_dof(
        &self,
        rho: &SaeManifoldRho,
        dispersion_r: f64,
    ) -> Result<Vec<f64>, String> {
        // Dense path: materialise the per-atom Grams G_k = Φ_kᵀdiag(a_k²)Φ_k and the
        // effective sample sizes N_eff,k = Σ_row a_k² from `self`, then delegate the
        // rank/EDF pricing to the shared `rank_dof_from_grams`. The #9 streaming path
        // ACCUMULATES the same `grams`/`n_eff` chunk-by-chunk (basis_values is not
        // persisted there) and calls the SAME core — so the criterion is identical.
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams);
        let n_eff = self.per_atom_effective_sample_size();
        self.rank_dof_from_grams(&grams, &n_eff, rho, dispersion_r)
    }

    /// Per-atom effective sample size `N_eff,k = Σ_i w_{ik}²` read through the
    /// shared [`SupportMeasure`] — the occupancy-aware Fisher information a gated
    /// atom k actually accumulates. This is the honest BIC/Laplace log-sample-size
    /// for the #2a rank charge (NOT the global row count `n_obs`): a row on which
    /// atom k's support is OFF contributes `w²=0`, so appending such rows leaves
    /// `N_eff,k` — and hence atom k's charge — unchanged (inert-row invariance).
    /// Matches the `ri.n_eff` the #9 streaming log-det pass accumulates.
    pub(crate) fn per_atom_effective_sample_size(&self) -> Vec<f64> {
        (0..self.k_atoms())
            .map(|k| {
                SupportMeasure::from_assignment(&self.assignment, k)
                    .map(|support| support.fisher_n())
                    .expect("term assignment shape must match atom count")
            })
            .collect()
    }

    /// Shared rank-charge DOF core (#11): `d_eff_k = rank_eff_k · basis_edf_k` from the
    /// PRE-ACCUMULATED per-atom Grams `grams[k] = Φ_kᵀdiag(a_k²)Φ_k` and effective sample
    /// sizes `n_eff[k] = Σ_row a_k²`. Split out of `per_atom_realised_rank_dof` so the
    /// dense path (grams from `self`) and the #9 streaming path (grams accumulated over
    /// `materialize_chunk` chunks) price the atom IDENTICALLY — only the Gram source
    /// differs. Reads only the persisted `decoder_coefficients`/`smooth_penalty`, never
    /// `basis_values` (absent under streaming).
    pub(crate) fn rank_dof_from_grams(
        &self,
        grams: &[Array2<f64>],
        n_eff: &[f64],
        rho: &SaeManifoldRho,
        dispersion_r: f64,
    ) -> Result<Vec<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let lam = rho.lambda_smooth_vec()?;
        // Fixed noise floor R = residual variance (dispersion). Guard finite/positive.
        let r_floor = if dispersion_r.is_finite() && dispersion_r > 0.0 {
            dispersion_r
        } else {
            f64::MIN_POSITIVE
        };
        let p_out = self.output_dim() as f64;
        let mut out = Vec::with_capacity(self.k_atoms());
        for k in 0..self.k_atoms() {
            // Each atom is priced through the shared `realised_rank_charge_dof` core
            // (the SAME fn the #2023 migration gate uses), so dense, #9 streaming, and
            // the tier PROMOTE/DEMOTE sites all adjudicate in one currency.
            let n_eff_k = *n_eff.get(k).ok_or_else(|| {
                format!("rank_dof_from_grams: missing effective sample size for atom {k}")
            })?;
            let lam_k = lam[k];
            let d = realised_rank_charge_dof(
                &grams[k],
                &self.atoms[k].decoder_coefficients,
                n_eff_k,
                p_out,
                r_floor,
                lam_k,
                Some(&self.atoms[k].smooth_penalty),
            )
            .map_err(|e| format!("rank_dof_from_grams: atom {k}: {e}"))?;
            out.push(d);
        }
        Ok(out)
    }

    /// #2023 — set the per-fit dead-atom data-row reseed opt-in (typed kwarg, no
    /// env lever). Default false.
    pub fn set_data_row_reseed(&mut self, enabled: bool) {
        self.data_row_reseed = enabled;
    }

    /// SAC — arm (`true`, the default) or disarm (`false`) the #976 Layer-1
    /// collapse-guard stack for this term's inner joint fits. The Sequential Atom
    /// Composition K=1 lane disarms it: a single atom never trips the guards, so
    /// disarming is a no-op on reconstruction while guaranteeing the per-atom and
    /// backfitting refits stay reseed-free (a mid-refit reseed would break the
    /// block-coordinate monotonicity). See [`super::stagewise`].
    pub fn set_guards_enabled(&mut self, enabled: bool) {
        self.guards_enabled = enabled;
    }

    /// SAC — whether the Layer-1 collapse-guard stack is armed on this term.
    pub fn guards_enabled(&self) -> bool {
        self.guards_enabled
    }

    /// Rung-2 — attach the behavioral data block, declaring this an augmented
    /// two-block term. Validates that the block's augmented output width
    /// `p_x + p_y` equals the term's actual `output_dim()` (the caller must have
    /// built the atoms at the augmented width) and that its row count matches, so
    /// the descriptor cannot silently disagree with the decoders it describes.
    pub fn set_behavior_block(
        &mut self,
        block: crate::manifold::BehaviorBlock,
    ) -> Result<(), String> {
        if block.augmented_dim() != self.output_dim() {
            return Err(format!(
                "SaeManifoldTerm::set_behavior_block: block augmented width p_x+p_y = {} but the \
                 term's output_dim is {} (atoms must be built at the augmented width)",
                block.augmented_dim(),
                self.output_dim()
            ));
        }
        if block.target.nrows() != self.n_obs() {
            return Err(format!(
                "SaeManifoldTerm::set_behavior_block: behavior target has {} rows but term has {}",
                block.target.nrows(),
                self.n_obs()
            ));
        }
        self.behavior = Some(block);
        Ok(())
    }

    /// Rung-2 — the behavioral data block, if this is a two-block term.
    pub fn behavior_block(&self) -> Option<&crate::manifold::BehaviorBlock> {
        self.behavior.as_ref()
    }

    /// Rung-2 — the activation output width `p_x` (the split point in the
    /// augmented output). Equals the full `output_dim()` for an ordinary
    /// single-block term (no behavior block installed).
    pub fn activation_output_dim(&self) -> usize {
        match &self.behavior {
            Some(block) => block.activation_dim,
            None => self.output_dim(),
        }
    }

    /// Rung-2 — the half-open behavior output column range `[p_x, p_x + p_y)`, or
    /// `None` for a single-block term.
    pub fn behavior_output_range(&self) -> Option<std::ops::Range<usize>> {
        self.behavior
            .as_ref()
            .map(|block| block.activation_dim..block.augmented_dim())
    }

    /// The installed per-row metric, if any. `None` ⇒ Euclidean / isotropic.
    /// Consumed by the gauge wiring (to build the matching `WeightField`) and by
    /// Object 4 (to read the [`MetricProvenance`](gam_problem::MetricProvenance)).
    pub fn row_metric(&self) -> Option<&gam_problem::RowMetric> {
        self.row_metric.as_ref()
    }

    /// The per-row inner product the additive diagnostics read through: the
    /// installed [`RowMetric`](gam_problem::RowMetric) when one
    /// was set (output-Fisher harvest present), otherwise a freshly-built
    /// Euclidean metric of the term's own `(n_obs, output_dim)` shape. Either way
    /// a metric always exists, so the diagnostics are never gated by a flag — the
    /// Euclidean fallback is the bit-identical isotropic path.
    pub(crate) fn diagnostic_metric(&self) -> Result<gam_problem::RowMetric, String> {
        match self.row_metric() {
            Some(metric) => Ok(metric.clone()),
            None => gam_problem::RowMetric::euclidean(self.n_obs(), self.output_dim()),
        }
    }

    /// Build the additive post-fit diagnostic report for this fitted term: the
    /// two-score per-atom [`AtomTwoLensReport`](crate::inference::atom_lens::AtomTwoLensReport)
    /// (presence / behavioral coupling / discrepancy) and the residual-gauge
    /// [`ResidualGaugeReport`](crate::identifiability::ResidualGaugeReport)
    /// certificate.
    ///
    /// Both reports are read through the same single metric
    /// ([`Self::diagnostic_metric`]): under a Euclidean / no-harvest provenance
    /// the lens coupling is `None` and the gauge is certified under Euclidean
    /// provenance — never an error, never gated by a flag (magic-by-default,
    /// mirroring the metric selection itself).
    ///
    /// `per_atom_ard_variances`, when supplied, is one ARD variance vector per
    /// atom (length = `latent_dim_k`), threaded into the certificate's
    /// equal-ARD-rotation detection. `None` (or a per-atom `None`) ⇒ no ARD prior
    /// on that atom. `isometry_pin_active` records whether an isometry gauge
    /// penalty was installed on the fit: `false` escalates the certificate to the
    /// `diffeomorphism-unpinned` verdict (the honest "no metric pin" statement),
    /// exactly as the certificate's own escalation flag specifies.
    ///
    /// Pure read: it never mutates the term, never touches a loss / criterion /
    /// penalty / optimizer state.
    pub fn fit_diagnostics_report(
        &self,
        per_atom_ard_variances: Option<&[Option<Array1<f64>>]>,
        isometry_pin_active: bool,
        reconstruction_dispersion: Option<f64>,
        fitted: ArrayView2<'_, f64>,
        assignments_override: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeManifoldFitDiagnostics, String> {
        if fitted.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "fit_diagnostics_report: fitted shape {:?} must be ({}, {})",
                fitted.dim(),
                self.n_obs(),
                self.output_dim()
            ));
        }
        if let Some(view) = assignments_override {
            let n = self.n_obs();
            let k = self.k_atoms();
            if view.dim() != (n, k) {
                return Err(format!(
                    "fit_diagnostics_report: assignments_override shape {:?} must be ({n}, {k})",
                    view.dim()
                ));
            }
        }
        let metric = self.diagnostic_metric()?;
        let atom_two_lens =
            crate::inference::atom_lens::atom_two_lens(self, &metric, assignments_override)?;

        let (certificate_model, streamed_curvature) =
            self.to_residual_gauge_model(metric, per_atom_ard_variances, isometry_pin_active)?;
        // #998: within-atom gauge families are certified on their EXACT orbits
        // in the model's own (decoder, coordinate) parameter space — compensated
        // symmetries are data-nulls by construction there, no lowering-error
        // calibration involved. This now holds whether or not an isometry pin is
        // active:
        //   * pin INACTIVE ⇒ the orbit verdict is the data residual alone (no
        //     penalty operator);
        //   * pin ACTIVE ⇒ the orbit verdict adds the isometry pin's orbit-space
        //     curvature through an [`OrbitPenaltyOperator`] lowered from the
        //     atom's second jet `Φ''` (the pullback-metric change along the orbit
        //     differentiates `J = Φ'B` through `t`). A model-class symmetry that
        //     preserves the metric stays a certified freedom; a non-isometric
        //     orbit (a basis not closed under the action) is genuinely pinned.
        // The relative-curvature fraction `cost/stiffness²` is invariant to the
        // pin strength μ (both faces scale with μ), so the operator is built at a
        // canonical unit weight. An atom whose basis exposes no analytic second
        // jet supplies no operator and falls back to the data residual — never an
        // error. Magic-by-default either way: the choice is derived from the fit,
        // never a flag.
        let views = self.atom_parameter_views();
        let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> = if isometry_pin_active
        {
            views
                .iter()
                .map(|view| {
                    view.as_ref().and_then(|v| {
                        crate::identifiability::isometry_orbit_penalty_operator(v, 1.0)
                    })
                })
                .collect()
        } else {
            (0..self.k_atoms()).map(|_| None).collect()
        };
        let residual_gauge = if isometry_pin_active {
            // The pin-active path consumes the per-row Jacobian curvature
            // directly (the certificate_model retains it under a pin), so route
            // through the non-streamed exact entry point.
            crate::identifiability::residual_gauge_exact(&certificate_model, &views, &ops)?
        } else {
            let (curvature_gram, root_rows) = streamed_curvature.ok_or_else(|| {
                "fit_diagnostics_report: missing streamed residual-gauge curvature for unpinned exact path"
                    .to_string()
            })?;
            crate::identifiability::residual_gauge_exact_from_curvature_gram(
                &certificate_model,
                &views,
                &ops,
                curvature_gram,
                root_rows,
            )?
        };

        // #1097 / #1103: per-atom Riesz-debiased functionals and the any-n-valid
        // split-LRT smooth-structure e-value (non-constant vs constant inner
        // decoder), read straight off the certificate model — which carries
        // each atom's `inner_fit` snapshot when the caller harvested it via
        // [`Self::set_atom_inner_fits`] before this report. Atoms without a
        // harvested inner fit degrade their inference fields to `None` inside
        // `atom_inference_reports`, so this is always populated (one entry per
        // atom) and never gated by a flag.
        let atom_inference = crate::identifiability::atom_inference_reports(&certificate_model);

        // #2081 — per-atom coordinate-fidelity certificate (uniformity + arc-length
        // defect). Always populated (one entry per atom, `None` for non-`d = 1`
        // charts), never dispersion-gated: coordinate quality does not depend on the
        // reconstruction dispersion the incoherence report needs.
        let coordinate_fidelity = (0..self.k_atoms())
            .map(|atom_idx| atom_coordinate_fidelity(self, atom_idx))
            .collect::<Result<Vec<_>, _>>()?;

        // Reviewer-F3 persistent-homology topology audit (one entry per atom,
        // `None` for caller-supplied or under-sampled atoms). A pure read of the
        // fitted decoder image and shared soft support measure; never gated by a flag and
        // feeds nothing back into the loss/criterion.
        let topology_persistence = (0..self.k_atoms())
            .map(|atom_idx| atom_topology_persistence(self, atom_idx))
            .collect::<Vec<_>>();

        Ok(SaeManifoldFitDiagnostics {
            atom_two_lens,
            residual_gauge,
            incoherence_report: match reconstruction_dispersion.or(self.certificate_dispersion) {
                Some(dispersion) => Some(dictionary_incoherence_report_with_dispersion(
                    self, dispersion, fitted,
                )?),
                None => None,
            },
            atom_inference,
            coordinate_fidelity,
            topology_persistence,
        })
    }

    /// Build the trust-diagnostics producer for the Python `diagnostics` block.
    ///
    /// `assignments` is the exact matrix used by reconstruction. Each atom's
    /// support is read through [`SupportMeasure`] so the
    /// trust scores use the same occupancy/effective-N convention as coordinate
    /// fidelity, persistence, and rank charge.
    pub fn trust_diagnostics_report(
        &self,
        assignments: ArrayView2<'_, f64>,
    ) -> Result<SaeTrustDiagnostics, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        if assignments.dim() != (n, k_atoms) {
            return Err(format!(
                "trust_diagnostics_report: assignments shape {:?} must be ({n}, {k_atoms})",
                assignments.dim()
            ));
        }
        if !assignments.iter().all(|v| v.is_finite()) {
            return Err("trust_diagnostics_report: assignments must be finite".to_string());
        }
        let metric = self.diagnostic_metric()?;
        let mut atoms = Vec::with_capacity(k_atoms);
        let mut atom_trust = Vec::with_capacity(k_atoms);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let support = SupportMeasure::from_assignment_matrix(assignments, atom_idx)?;
            let active_token_count = support.positive_rows().len();
            let coverage = if n > 0 { support.ess() / n as f64 } else { 0.0 };
            let activation_frequency = if n > 0 {
                support.mass() / n as f64
            } else {
                0.0
            };
            let (sigma_min_tangent, sigma_max_tangent) =
                self.atom_tangent_spectrum_from_assignments(atom_idx, &support, &metric)?;
            let tangent_condition_score = if sigma_max_tangent > 0.0 {
                (sigma_min_tangent / sigma_max_tangent).clamp(0.0, 1.0)
            } else {
                0.0
            };
            // Curvature-certification power scales with the fourth power of
            // observed chart coverage: λ₂ ≈ r²·a⁴/45, hence N* ∝ a⁻⁴. A
            // well-conditioned tangent basis on a thinly covered atom is still
            // not globally trustworthy, so trust must decay quartically rather
            // than linearly (or not at all) with observed extent/coverage.
            let chart_coverage_weight = coverage.powi(4);
            let trust_score = tangent_condition_score * chart_coverage_weight;
            atom_trust.push(trust_score);
            atoms.push(SaeAtomTrustDiagnostics {
                trust_score,
                sigma_min_tangent,
                sigma_max_tangent,
                tangent_condition_score,
                coverage,
                activation_frequency,
                support_mass: support.mass(),
                effective_n: support.fisher_n(),
                support_ess: support.ess(),
                untyped: matches!(atom.basis_kind, SaeAtomBasisKind::Precomputed(_)),
                active_token_count,
            });
        }
        Ok(SaeTrustDiagnostics { atom_trust, atoms })
    }

    pub(crate) fn atom_tangent_spectrum_from_assignments(
        &self,
        atom_idx: usize,
        support: &SupportMeasure,
        metric: &gam_problem::RowMetric,
    ) -> Result<(f64, f64), String> {
        let atom = &self.atoms[atom_idx];
        let d = atom.latent_dim;
        let p = self.output_dim();
        if d == 0 || p == 0 {
            return Ok((0.0, 0.0));
        }
        if support.len() != self.n_obs() || support.atom_idx() != atom_idx {
            return Err(format!(
                "atom_tangent_spectrum_from_assignments: support atom/rows ({}, {}) != ({atom_idx}, {})",
                support.atom_idx(),
                support.len(),
                self.n_obs()
            ));
        }
        let mut gram = Array2::<f64>::zeros((d, d));
        let mut active_mass_sum = 0.0_f64;
        let mut jac_row = vec![0.0_f64; p * d];
        for row in 0..self.n_obs() {
            let mass = support.weight(row);
            if !(mass > 0.0) {
                continue;
            }
            active_mass_sum += mass;
            for axis in 0..d {
                let start = axis;
                let mut tangent = vec![0.0_f64; p];
                atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                for out in 0..p {
                    jac_row[out * d + start] = tangent[out];
                }
            }
            let row_pullback = metric.pullback(row, &jac_row, d);
            for axis_a in 0..d {
                for axis_b in 0..=axis_a {
                    gram[[axis_a, axis_b]] += mass * row_pullback[[axis_a, axis_b]];
                }
            }
            jac_row.fill(0.0);
        }
        if !(active_mass_sum > 0.0) {
            return Ok((0.0, 0.0));
        }
        let inv_mass = 1.0 / active_mass_sum;
        for axis_a in 0..d {
            for axis_b in 0..=axis_a {
                let value = gram[[axis_a, axis_b]] * inv_mass;
                gram[[axis_a, axis_b]] = value;
                gram[[axis_b, axis_a]] = value;
            }
        }
        let (evals, _) = gram.eigh(Side::Lower).map_err(|e| {
            format!(
                "trust_diagnostics_report: atom {atom_idx} tangent eigendecomposition failed: {e}"
            )
        })?;
        let mut sigma_min = f64::INFINITY;
        let mut sigma_max = 0.0_f64;
        for value in evals.iter().copied() {
            let clamped = value.max(0.0);
            let sigma = clamped.sqrt();
            sigma_min = sigma_min.min(sigma);
            sigma_max = sigma_max.max(sigma);
        }
        if sigma_min.is_finite() {
            Ok((sigma_min, sigma_max))
        } else {
            Ok((0.0, 0.0))
        }
    }

    /// Per-atom exact parameter-space views for the #998 certificate path:
    /// the basis values / first-derivative jet, decoder coefficients, latent
    /// coordinates, and assignment mass each atom was actually fitted with.
    /// Sphere atoms get `None` (their chart's group action is nonlinear, so
    /// the exact-orbit realisation does not apply and they stay on the frame
    /// path), as does any atom whose coordinate chart width disagrees with its
    /// latent dimension (a structurally inconsistent atom must not masquerade
    /// as exactly certified).
    pub(crate) fn atom_parameter_views(
        &self,
    ) -> Vec<Option<crate::identifiability::AtomParameterView>> {
        let assignments = self.assignment.assignments();
        let n = self.n_obs();
        self.atoms
            .iter()
            .enumerate()
            .map(|(k, atom)| {
                if matches!(atom.basis_kind, SaeAtomBasisKind::Sphere) {
                    return None;
                }
                let coords = self.assignment.coords[k].as_matrix().to_owned();
                if coords.nrows() != n || coords.ncols() != atom.latent_dim {
                    return None;
                }
                let mut activations = Array1::<f64>::zeros(n);
                for row in 0..n {
                    activations[row] = assignments[[row, k]];
                }
                // Second jet Φ'' (#998): supplied when the atom's evaluator
                // exposes an analytic Hessian, so a pin-active fit can lower its
                // orbit-space isometry penalty operator (the metric-change of the
                // pullback gram differentiates Φ' through t). Absent ⇒ the orbit
                // verdict stays on the data residual / no-pin path, never an
                // error.
                let basis_second_jet = atom
                    .basis_evaluator
                    .as_ref()
                    .and_then(|evaluator| evaluator.second_jet_dyn(coords.view()))
                    .and_then(|res| res.ok());
                Some(crate::identifiability::AtomParameterView {
                    basis_values: atom.basis_values.clone(),
                    basis_jacobian: atom.basis_jacobian.clone(),
                    decoder: atom.decoder_coefficients.clone(),
                    coords,
                    activations,
                    basis_second_jet,
                })
            })
            .collect()
    }

    /// Lower this fitted term into the self-contained
    /// [`FittedSaeManifold`](crate::identifiability::FittedSaeManifold) the
    /// residual-gauge certificate consumes.
    ///
    /// The certificate's parameter space is the per-atom decoder **frame** — the
    /// `(output_dim, latent_dim)` image of the atom's latent axes in output space.
    /// We realise it as the active-mass-weighted mean decoder tangent
    /// `frame_k[:, a] = (Σ_n a_{nk} · ∂g_k/∂t_a(n)) / Σ_n a_{nk}` over the atom's
    /// active rows (the centroid decoder Jacobian columns the certificate docs
    /// name). The per-row pinning Jacobian block `J_n ∈ ℝ^{p × param_dim}` is the
    /// assignment-weighted per-row decoder tangent placed at each atom's frame
    /// slot: column `(k, i, a)` of `J_n` is `a_{nk} · ∂g_k/∂t_a(n)[i]` — exactly
    /// the directions the reconstruction data gives cost to, in the same metric
    /// the fit used (whitened by the certificate through `RowMetric`).
    ///
    /// The flattened frame layout matches the certificate's
    /// `vec(frame_0) ⊕ vec(frame_1) ⊕ …`, row-major within each frame
    /// (`frame_k[i, a]` at offset `atom_offset(k) + i·latent_dim_k + a`).
    pub(crate) fn to_residual_gauge_model(
        &self,
        metric: gam_problem::RowMetric,
        per_atom_ard_variances: Option<&[Option<Array1<f64>>]>,
        isometry_pin_active: bool,
    ) -> Result<
        (
            crate::identifiability::FittedSaeManifold,
            Option<(Array2<f64>, usize)>,
        ),
        String,
    > {
        use crate::identifiability::{AtomTopology, FittedAtom, FittedSaeManifold};

        let n = self.n_obs();
        let p = self.output_dim();
        let k = self.k_atoms();
        let assignments = self.assignment.assignments();

        // Per-atom frame `(p, d)` = active-mass-weighted mean decoder tangent,
        // and the flattened-frame column offset bookkeeping for the joint
        // parameter vector (`vec(frame_0) ⊕ …`, row-major within each frame).
        let mut fitted_atoms: Vec<FittedAtom> = Vec::with_capacity(k);
        let mut atom_offsets: Vec<usize> = Vec::with_capacity(k);
        let mut atom_axis_dim: Vec<usize> = Vec::with_capacity(k);
        let mut cursor = 0usize;
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let d = atom.latent_dim;
            let topology = match (&atom.basis_kind, d) {
                (SaeAtomBasisKind::Periodic, 1) | (SaeAtomBasisKind::Torus, 1) => {
                    AtomTopology::Circle
                }
                (SaeAtomBasisKind::Periodic, _) | (SaeAtomBasisKind::Torus, _) => {
                    AtomTopology::Torus { latent_dim: d }
                }
                (SaeAtomBasisKind::Sphere, _) => AtomTopology::Sphere,
                // `Cylinder` (`S¹ × ℝ`) has exactly one continuous gauge: the
                // rotation (shift) of the periodic axis. The unbounded line axis
                // carries no rotational gauge, and its translation is already
                // pinned by the design's constant column — so the identifiability
                // gauge is that of a single circle. Fixing it as `Torus` would
                // over-impose a second (nonexistent) circle shift; fixing it as
                // `EuclideanPatch { 2 }` would over-impose a frame rotation
                // mixing the periodic and linear axes. `Circle` fixes the one
                // real continuous gauge and leaves the linear axis ungauged.
                (SaeAtomBasisKind::Cylinder, _) => AtomTopology::Circle,
                // The double-cover chart has one continuous angular gauge;
                // the bounded width axis carries no rotational gauge.
                (SaeAtomBasisKind::Mobius, _) => AtomTopology::Circle,
                (
                    SaeAtomBasisKind::Linear
                    | SaeAtomBasisKind::Duchon
                    | SaeAtomBasisKind::EuclideanPatch
                    | SaeAtomBasisKind::Poincare
                    | SaeAtomBasisKind::FiniteSet
                    | SaeAtomBasisKind::Precomputed(_),
                    _,
                ) => AtomTopology::EuclideanPatch { latent_dim: d },
            };

            let mut frame = Array2::<f64>::zeros((p, d));
            let mut active_mass = 0.0_f64;
            let mut tangent = vec![0.0_f64; p];
            for row in 0..n {
                let a_nk = assignments[[row, atom_idx]];
                if !(a_nk > 0.0) {
                    continue;
                }
                active_mass += a_nk;
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        frame[[i, axis]] += a_nk * tangent[i];
                    }
                }
            }
            if active_mass > 0.0 {
                let inv = 1.0 / active_mass;
                frame.mapv_inplace(|v| v * inv);
            }

            // #995 lowering-error scale: mass-weighted relative dispersion of
            // the per-row tangents around the mean frame just built,
            //   Σ_n a_n Σ_ax ‖t_ax(n) − frame[:,ax]‖² / Σ_n a_n Σ_ax ‖t_ax(n)‖².
            // 0 ⇒ the frame represents every active row exactly (flat
            // decoder); → 1 ⇒ the tangent field disperses so strongly (e.g. a
            // full circle, whose tangents average out) that the mean-frame
            // compression cannot distinguish gauge motion from curvature. The
            // certificate calibrates its per-generator verdict tolerance to
            // this scale so it never claims a pin it cannot resolve.
            let mut disp_num = 0.0_f64;
            let mut disp_den = 0.0_f64;
            for row in 0..n {
                let a_nk = assignments[[row, atom_idx]];
                if !(a_nk > 0.0) {
                    continue;
                }
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        let dev = tangent[i] - frame[[i, axis]];
                        disp_num += a_nk * dev * dev;
                        disp_den += a_nk * tangent[i] * tangent[i];
                    }
                }
            }
            let lowering_error = if disp_den > 0.0 {
                (disp_num / disp_den).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let ard_variances = per_atom_ard_variances
                .and_then(|all| all.get(atom_idx))
                .and_then(|opt| opt.clone())
                .filter(|v| v.len() == d);

            fitted_atoms.push(FittedAtom {
                name: atom.name.clone(),
                topology,
                frame,
                ard_variances,
                lowering_error,
                // #1019: post-fit chart canonicalization (arc length for
                // d = 1, isometry-flow for d = 2 torus, flat-reference
                // isometry-flow for d = 2 free/patch, round-sphere
                // conformal-boost flow for d = 2 sphere atoms) pins the chart;
                // the certificate downgrades this atom's chart freedom to the
                // finite isometry group with PinnedByCanonicalization
                // provenance.
                chart_canonicalized: atom.chart_canonicalized
                    && (d == 1
                        || (d == 2
                            && matches!(
                                atom.basis_kind,
                                SaeAtomBasisKind::Torus
                                    | SaeAtomBasisKind::Linear
                                    | SaeAtomBasisKind::Duchon
                                    | SaeAtomBasisKind::EuclideanPatch
                                    | SaeAtomBasisKind::Sphere
                            ))),
                // #1097 / #1103: the per-atom inner-decoder-smooth snapshot,
                // attached when the post-fit harness has run
                // [`Self::set_atom_inner_fits`] (it needs the reconstruction
                // target Z, dropped from the objective at fit end). `None` on a
                // bare certificate-only model, or for a degenerate atom whose
                // inner Hessian was not SPD.
                inner_fit: self
                    .atom_inner_fits
                    .as_ref()
                    .and_then(|fits| fits.get(atom_idx))
                    .and_then(|slot| slot.clone()),
            });
            atom_offsets.push(cursor);
            atom_axis_dim.push(d);
            cursor += p * d;
        }
        let param_dim = cursor;

        // Per-row pinning Jacobian `J_n ∈ ℝ^{p × param_dim}` flattened row-major
        // (`J_n[i, c] = jacobian_rows[n][i · param_dim + c]`). Column `(k, i', a)`
        // of `J_n` is `a_{nk} · ∂g_k/∂t_a(n)[i']` placed at the atom-k frame slot
        // and read out on output coordinate `i = i'` (a frame perturbation of
        // output `i'` moves only the row's output coordinate `i'`).
        //
        // The pinned certificate still consumes the legacy row-block contract.
        // The unpinned exact path consumes only `RᵀR`, so stream each transient
        // row Jacobian through the metric whitening and discard it immediately.
        let (jacobian_rows, streamed_curvature) = if isometry_pin_active {
            let mut jacobian_rows: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut tangent = vec![0.0_f64; p];
            for row in 0..n {
                let mut j_flat = vec![0.0_f64; p * param_dim];
                for (atom_idx, atom) in self.atoms.iter().enumerate() {
                    let a_nk = assignments[[row, atom_idx]];
                    if !(a_nk > 0.0) {
                        continue;
                    }
                    let d = atom_axis_dim[atom_idx];
                    let base = atom_offsets[atom_idx];
                    for axis in 0..d {
                        atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                        for i in 0..p {
                            // Frame coordinate `(k, i, axis)` sits at column
                            // `base + i·d + axis`; it sources output coordinate `i`.
                            j_flat[i * param_dim + base + i * d + axis] += a_nk * tangent[i];
                        }
                    }
                }
                jacobian_rows.push(j_flat);
            }
            (jacobian_rows, None)
        } else {
            let streamed = self.residual_gauge_streamed_data_curvature(
                &metric,
                &atom_offsets,
                &atom_axis_dim,
                param_dim,
            )?;
            (Vec::new(), Some(streamed))
        };

        // Isometry-penalty curvature root over the frame parameter space. When
        // the isometry gauge pin is active it gives curvature along every fitted
        // frame direction (it resists deviation of the decoder image from its
        // arc-length parameterization), so its row space is the span of the
        // per-atom frame columns: one root row per `(k, axis)` carrying that
        // atom's frame column at the atom's frame slot. Empty (`0 × param_dim`)
        // when the pin is inactive — exactly the certificate's escalation
        // condition to `diffeomorphism-unpinned`.
        let isometry_penalty_root = if isometry_pin_active && param_dim > 0 {
            let mut root_rows: Vec<Array1<f64>> = Vec::new();
            for (atom_idx, fitted) in fitted_atoms.iter().enumerate() {
                let d = atom_axis_dim[atom_idx];
                let base = atom_offsets[atom_idx];
                for axis in 0..d {
                    let mut r = Array1::<f64>::zeros(param_dim);
                    let mut any = false;
                    for i in 0..p {
                        let v = fitted.frame[[i, axis]];
                        if v != 0.0 {
                            any = true;
                        }
                        r[base + i * d + axis] = v;
                    }
                    if any {
                        root_rows.push(r);
                    }
                }
            }
            let mut root = Array2::<f64>::zeros((root_rows.len(), param_dim));
            for (ri, r) in root_rows.iter().enumerate() {
                root.row_mut(ri).assign(r);
            }
            root
        } else {
            Array2::<f64>::zeros((0, param_dim))
        };

        Ok((
            FittedSaeManifold {
                atoms: fitted_atoms,
                jacobian_rows,
                isometry_penalty_root,
                metric,
            },
            streamed_curvature,
        ))
    }

    pub(crate) fn residual_gauge_streamed_data_curvature(
        &self,
        metric: &gam_problem::RowMetric,
        atom_offsets: &[usize],
        atom_axis_dim: &[usize],
        param_dim: usize,
    ) -> Result<(Array2<f64>, usize), String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if metric.p_out() != p {
            return Err(format!(
                "residual_gauge_streamed_data_curvature: metric output dim {} but term has {p}",
                metric.p_out()
            ));
        }
        let rank = metric.metric_rank();
        let mut gram = Array2::<f64>::zeros((param_dim, param_dim));
        if param_dim == 0 || n == 0 || rank == 0 {
            return Ok((gram, n * rank));
        }

        let assignments = self.assignment.assignments();
        let mut tangent = vec![0.0_f64; p];
        let mut j_flat = vec![0.0_f64; p * param_dim];
        let mut root_row = Array1::<f64>::zeros(param_dim);
        for row in 0..n {
            j_flat.fill(0.0);
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let a_nk = assignments[[row, atom_idx]];
                if !(a_nk > 0.0) {
                    continue;
                }
                let d = atom_axis_dim[atom_idx];
                let base = atom_offsets[atom_idx];
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        j_flat[i * param_dim + base + i * d + axis] += a_nk * tangent[i];
                    }
                }
            }

            if metric.drives_gauge() {
                for r in 0..rank {
                    root_row.fill(0.0);
                    for c in 0..param_dim {
                        let mut acc = 0.0_f64;
                        for i in 0..p {
                            acc += metric.factor_entry(row, i, r) * j_flat[i * param_dim + c];
                        }
                        root_row[c] = acc;
                    }
                    let row_slice = root_row.as_slice().ok_or_else(|| {
                        "residual_gauge_streamed_data_curvature: non-contiguous root row"
                            .to_string()
                    })?;
                    Self::accumulate_residual_gauge_gram_row(&mut gram, row_slice);
                }
            } else {
                for i in 0..p {
                    let start = i * param_dim;
                    let end = start + param_dim;
                    Self::accumulate_residual_gauge_gram_row(&mut gram, &j_flat[start..end]);
                }
            }
        }

        for a in 0..param_dim {
            for b in 0..a {
                gram[[b, a]] = gram[[a, b]];
            }
        }
        Ok((gram, n * rank))
    }

    pub(crate) fn accumulate_residual_gauge_gram_row(gram: &mut Array2<f64>, row: &[f64]) {
        for a in 0..row.len() {
            let va = row[a];
            if va == 0.0 {
                continue;
            }
            for b in 0..=a {
                let vb = row[b];
                if vb != 0.0 {
                    gram[[a, b]] += va * vb;
                }
            }
        }
    }

    pub fn set_temperature_schedule(
        &mut self,
        sched: GumbelTemperatureSchedule,
    ) -> Result<(), String> {
        sched.validate()?;
        self.assignment
            .mode
            .set_temperature(sched.current_tau(sched.iter_count))?;
        self.temperature_schedule = Some(sched);
        Ok(())
    }

    pub(crate) fn advance_temperature_schedule(&mut self) -> Result<Option<f64>, String> {
        let Some(schedule) = self.temperature_schedule.as_mut() else {
            return Ok(None);
        };
        schedule.validate()?;
        let tau = schedule.step();
        self.assignment.mode.set_temperature(tau)?;
        Ok(Some(tau))
    }

    pub fn n_obs(&self) -> usize {
        self.assignment.n_obs()
    }

    pub fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Auto-derived in-core vs streaming plan for SAE Arrow-Schur work.
    ///
    /// This is intentionally not user-configurable: the route follows the
    /// retained full-batch working-set estimate and the currently selected GPU
    /// memory budget when CUDA is usable, otherwise a conservative host budget.
    pub fn streaming_plan(&self) -> SaeStreamingPlan {
        let n_obs = self.n_obs();
        let total_basis: usize = self.atoms.iter().map(|atom| atom.basis_size()).sum();
        let d_max = self
            .atoms
            .iter()
            .map(|atom| atom.latent_dim)
            .max()
            .unwrap_or(0);
        let border_dim = if self.any_frame_active() {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        sae_streaming_plan_for_shape(n_obs, total_basis, self.k_atoms(), d_max, border_dim)
    }

    /// Construction-time validation: every Psi-tier analytic penalty in the
    /// registry must be dispatchable into the SAE arrow-Schur row layout.
    ///
    /// Two invariants are enforced upfront so the dispatch loop in
    /// `add_sae_analytic_penalty_contributions` is total (no runtime
    /// "unsupported penalty" fallthrough, no per-call K-gating):
    ///
    /// 1. Every Psi-tier penalty is either in [`sae_penalty_is_row_block_supported`],
    ///    or `NuclearNorm` (which is redirected to the per-atom decoder (β) block
    ///    rather than the coord "t" row block). Assignment sparsity penalties
    ///    (`OrderedBetaBernoulli`, `SoftmaxAssignmentSparsity`) are refused because the SAE
    ///    term already owns them through its built-in assignment path
    ///    (`loss.assignment_sparsity`). Penalty kinds with cross-row structure
    ///    (`TotalVariation`, `Monotonicity`, `BlockSparsity`,
    ///    `IvaeRidgeMeanGauge`, `Orthogonality`, `NestedPrefix`,
    ///    `SheafConsistency`) cannot be expressed in the SAE row-block layout
    ///    and are refused here.
    ///
    /// 2. If any Psi-tier row-block penalty is present, every atom shares
    ///    the same coord latent dim. The current registry model carries one
    ///    `latent_dim` per descriptor (the "t" latent block declares one
    ///    `d` value); per-atom dispatch with heterogeneous `d_k` would
    ///    require per-atom registry entries or per-kind in-place
    ///    reshaping. Mixed-d row-block fits are rejected with an actionable
    ///    error pointing at the configuration mismatch.
    ///
    /// The K=1 case trivially satisfies (2). Beta-tier and rho-tier
    /// penalties are not constrained here.
    pub(crate) fn validate_analytic_penalty_registry(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<(), String> {
        let mut non_composing_row_block: Option<&str> = None;
        for penalty in &registry.penalties {
            if penalty.tier() != PenaltyTier::Psi {
                continue;
            }
            if matches!(
                penalty,
                AnalyticPenaltyKind::OrderedBetaBernoulli(_)
                    | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            ) {
                return Err(format!(
                    "SAE-manifold term refuses analytic penalty {:?}: assignment sparsity \
                     is owned by the built-in SAE assignment path (loss.assignment_sparsity). \
                     Registering it would double-count the objective and gradient",
                    penalty.name()
                ));
            }
            // NuclearNorm is redirected to the per-atom decoder (β) block in
            // `add_sae_beta_penalty` (it penalizes each atom's decoder matrix
            // singular spectrum, i.e. its embedding rank), so it bypasses the
            // coord "t" row-block requirement below.
            if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) {
                continue;
            }
            if !sae_penalty_is_row_block_supported(penalty) {
                return Err(format!(
                    "SAE-manifold term refuses analytic penalty {:?}: this kind \
                     has cross-row structure and cannot be expressed in the \
                     arrow-Schur row layout. Use only row-block-supported \
                     coord penalties (ARD, BlockOrthogonality, \
                     Sparsity/TopK/ThresholdGate, RowPrecisionPrior, \
                     ParametricRowPrecisionPrior, ScadMcp, Isometry) on the \
                     coord latent block, or move the penalty to a non-SAE \
                     term",
                    penalty.name()
                ));
            }
            // A row-block penalty that composes over heterogeneous coord dims
            // (per-atom-additive, dim-adaptive: ScadMcp / Sparsity / native ARD /
            // Isometry) dispatches cleanly on a mixed dictionary, so it never
            // forces a uniform `atom_dim`. Only the fixed-`d` structural
            // penalties (BlockOrthogonality, TopK/ThresholdGate, row-precision) do.
            if !sae_row_block_penalty_composes_over_heterogeneous_coord_dims(penalty) {
                non_composing_row_block = Some(penalty.name());
            }
        }
        if let Some(offender) = non_composing_row_block {
            let mut dims = self.assignment.coords.iter().map(|c| c.latent_dim());
            if let Some(first) = dims.next() {
                if let Some(mismatch) = dims.find(|d| *d != first) {
                    return Err(format!(
                        "SAE-manifold term refuses row-block analytic penalty {offender:?}: \
                         atoms have heterogeneous coord latent dims (saw {first} \
                         and {mismatch}). This penalty carries a fixed per-axis \
                         structure bound to one shared `d` (BlockOrthogonality \
                         reshapes to `(n_eff × d)` and groups axes; TopK/ThresholdGate \
                         hold per-axis thresholds; the row-precision priors hold a \
                         `(n_eff × d × d)` stack), so per-atom dispatch with mixed \
                         `d_k` would silently truncate or expand axes. Configure all \
                         atoms with the same `atom_dim`, or drop this penalty. \
                         (Dim-adaptive row-block penalties — ScadMcp, Sparsity, \
                         native ARD, Isometry — compose on a mixed dictionary and \
                         are admitted.)"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Up-front cross-check (issue #2098, SPEC-8; F6): a heterogeneous-`d_atom`
    /// dictionary is compatible with the *dim-adaptive* row-block "t"-block
    /// penalties (native ARD / SCAD-MCP coord sparsity / sparsity / isometry) but
    /// incompatible with the *fixed-`d` structural* ones (block-orthogonality,
    /// TopK/ThresholdGate, row-precision priors).
    ///
    /// The dim-adaptive penalties are per-atom-additive and read each atom's own
    /// `d_k` (`ScadMcp`/`Sparsity` iterate the flat block element-wise; native
    /// ARD sums per atom over `d_k` axes with a per-atom `log_ard[k]`; isometry
    /// is rebuilt per atom by `corrected_isometry_penalty`), so the arrow-Schur
    /// assembler dispatches them cleanly across mixed dims and the penalized quasi-Laplace
    /// evidence — itself a per-atom sum — stays exact with no padding or
    /// truncation (see
    /// [`sae_row_block_penalty_composes_over_heterogeneous_coord_dims`]). The
    /// structural penalties carry a fixed per-axis shape bound to one shared `d`
    /// (reshape to `(n_eff × d)`, per-axis thresholds, a `(n_eff × d × d)`
    /// precision stack) and cannot dispatch on mixed dims without silently
    /// truncating or padding axes.
    ///
    /// The engine self-protects here so a genuine incompatibility surfaces as a
    /// direct, actionable error at the FFI boundary rather than as a deep
    /// `RemlConvergenceError` mid penalized quasi-Laplace solve (the failure mode
    /// [`Self::validate_analytic_penalty_registry`] otherwise produces during
    /// `assemble_arrow_schur`).
    ///
    /// Native ARD rides the separate `native_ard_enabled` FFI flag rather than a
    /// registry descriptor, but because it composes it is admitted on a mixed
    /// dictionary; only a NON-composing REGISTRY penalty triggers the refusal.
    ///
    /// Homogeneous coord dims (including `K == 1`) always pass, as does a
    /// heterogeneous dictionary that carries only composing penalties.
    pub fn validate_heterogeneous_atom_compatibility(
        &self,
        registry: Option<&AnalyticPenaltyRegistry>,
        // Retained for FFI signature stability and self-documentation. Post-F6 it
        // no longer gates: native ARD composes over heterogeneous coord dims
        // (`ard_value` is a per-atom sum over `d_k`), so it is admitted whether or
        // not it is enabled — only a NON-composing registry penalty refuses.
        native_ard_enabled: bool,
    ) -> Result<(), String> {
        // Per-atom coord latent dims via the same accessor the registry
        // validator uses, so the two cannot disagree on "heterogeneous".
        let mut dims = self.assignment.coords.iter().map(|c| c.latent_dim());
        let Some(first) = dims.next() else {
            return Ok(());
        };
        let Some(mismatch) = dims.find(|d| *d != first) else {
            // Homogeneous coord dims: every row-block penalty dispatches cleanly.
            return Ok(());
        };
        // Native ARD (the `native_ard_enabled` flag) composes over heterogeneous
        // coord dims: `ard_value` sums per atom over `d_k` axes with a per-atom
        // `log_ard[k]` of length `d_k`, so a mixed dictionary is its native shape
        // and it never forces a uniform `atom_dim`. Only the fixed-`d` structural
        // REGISTRY penalties do — detect them via the composability predicate.
        let non_composing = registry.and_then(|reg| {
            reg.penalties.iter().find(|penalty| {
                penalty.tier() == PenaltyTier::Psi
                    && sae_penalty_is_row_block_supported(penalty)
                    && !sae_row_block_penalty_composes_over_heterogeneous_coord_dims(penalty)
            })
        });
        let Some(offender) = non_composing else {
            return Ok(());
        };
        Err(format!(
            "SAE-manifold fit refuses row-block analytic penalty {:?} on heterogeneous \
             atom coordinate dims (saw {first} and {mismatch}): this penalty carries a \
             fixed per-axis structure bound to one shared `d` (BlockOrthogonality reshapes \
             to `(n_eff × d)` and groups axes; TopK/ThresholdGate hold per-axis thresholds; the \
             row-precision priors hold a `(n_eff × d × d)` stack), so mixed per-atom \
             coordinate dims cannot be dispatched (they would silently truncate or pad axes). \
             Either configure a uniform atom_dim for all atoms, or drop this penalty. The \
             dim-adaptive row-block penalties — SCAD-MCP, sparsity, native ARD, isometry — \
             compose on a mixed dictionary and are admitted (native ARD enabled here: {}).",
            offender.name(),
            native_ard_enabled
        ))
    }

    pub fn output_dim(&self) -> usize {
        self.atoms[0].output_dim()
    }

    /// gam#2144 — `true` when the installed row metric whitens the likelihood at
    /// ANY rank. Drives whitening of the log-det row jets so they differentiate
    /// the SAME whitened operator (`JᵀU UᵀJ`) the assembly builds. Independent of
    /// the ordered Beta--Bernoulli PSD majorization, which (#2144/#1038) is UNCONDITIONAL — the
    /// assembly, criterion log-det, ρ-trace, and θ-adjoint all carry the majorized
    /// ordered Beta--Bernoulli curvature on every path, whitened or not, so there is no rank-gated
    /// majorization predicate anymore. `false` for the identity metric or no
    /// metric.
    pub(crate) fn whiten_logdet_row_jets(&self) -> bool {
        self.row_metric
            .as_ref()
            .is_some_and(|m| m.whitens_likelihood())
    }

    pub fn beta_dim(&self) -> usize {
        let p = self.output_dim();
        self.atoms.iter().map(|a| a.basis_size() * p).sum()
    }

    pub(crate) fn take_border_hbb_workspace(&mut self, border_dim: usize) -> Array2<f64> {
        let mut workspace =
            std::mem::replace(&mut self.border_hbb_workspace, Array2::<f64>::zeros((0, 0)));
        if workspace.dim() != (border_dim, border_dim) {
            workspace = Array2::<f64>::zeros((border_dim, border_dim));
        } else {
            workspace.fill(0.0);
        }
        workspace
    }

    pub(crate) fn reclaim_border_hbb_workspace(&mut self, sys: &mut ArrowSchurSystem) {
        let workspace = std::mem::replace(&mut sys.hbb, Array2::<f64>::zeros((0, 0)));
        self.border_hbb_workspace = workspace;
    }

    pub(crate) fn take_arrow_assembly_buffers(&mut self) -> (Vec<ArrowRowBlock>, Array1<f64>) {
        (
            std::mem::take(&mut self.arrow_assembly_workspace.rows),
            std::mem::replace(
                &mut self.arrow_assembly_workspace.gb,
                Array1::<f64>::zeros(0),
            ),
        )
    }

    /// Install a completely refreshed device descriptor while retaining its
    /// allocation identity when the prior iterate returned one to the pool.
    pub(crate) fn install_device_sae_pcg_data(
        &mut self,
        sys: &mut ArrowSchurSystem,
        data: DeviceSaePcgData,
    ) {
        let recycled = self.arrow_assembly_workspace.device_sae_pcg.take();
        sys.set_device_sae_pcg_data_reusing(data, recycled);
    }

    /// Return allocation storage after every consumer of this iteration's
    /// numerical system has finished. No operator or factor cache is retained;
    /// the next assembly zeroes and recomputes all row/shared blocks.
    pub(crate) fn reclaim_arrow_assembly_workspace(&mut self, sys: &mut ArrowSchurSystem) {
        self.arrow_assembly_workspace.rows = std::mem::take(&mut sys.rows);
        self.arrow_assembly_workspace.gb = std::mem::replace(&mut sys.gb, Array1::<f64>::zeros(0));
        if let Some(device) = sys.device_sae_pcg.take() {
            self.arrow_assembly_workspace.device_sae_pcg = Some(device);
        }
        if !sys.hbb.is_empty() {
            self.reclaim_border_hbb_workspace(sys);
        }
    }

    /// Factored arrow-Schur border dimension `Σ_k M_k · r_k` (issue #972): the
    /// number of decoder coordinates the border actually carries once the
    /// low-rank Grassmann frames are profiled out. Atoms with no active frame
    /// contribute their full `M_k · p` (`r_k == p`), so on the all-full-`B` path
    /// this equals [`Self::beta_dim`]. The border Cholesky / criterion log-det
    /// scale with THIS count, not `beta_dim`.
    pub fn factored_border_dim(&self) -> usize {
        self.atoms.iter().map(|a| a.border_coeff_count()).sum()
    }

    /// Total profiled-out Grassmann manifold dimension `Σ_k r_k·(p − r_k)` across
    /// all active frames (issue #972). This is the count of decoder-frame degrees
    /// of freedom estimated OUTSIDE the border by closed-form polar steps, and it
    /// must enter the quasi-Laplace score dimension accounting (evidence honesty):
    /// the profiled frame is a MAP point on `∏_k Gr(r_k, p)`, contributing this
    /// many free dimensions to the model. `0` when every atom is on the full-`B`
    /// path. Counted (unscaled by `log λ`) in the effective decoder-parameter dof
    /// of `reconstruction_dispersion`; it does NOT enter the `log λ`-scaled
    /// smoothing Occam normalizer (the frame orientation is unpenalized by `λ`).
    pub fn grassmann_evidence_dimension(&self) -> usize {
        self.atoms
            .iter()
            .map(|a| a.frame_manifold_dimension())
            .sum()
    }

    /// True iff any atom has an active low-rank Grassmann frame (issue #972).
    pub fn frames_active(&self) -> bool {
        self.atoms.iter().any(|a| a.decoder_frame.is_some())
    }

    /// Alias of [`Self::frames_active`] (issue #972 / #977 T1): the predicate the
    /// assembly / step-lift branch on to decide whether the β-tier is built in
    /// the factored coordinate layout. Named to read as the question
    /// "is the factored path engaged?" at its call sites.
    pub fn any_frame_active(&self) -> bool {
        self.frames_active()
    }

    /// Per-atom column offsets of the *factored* border (issue #972 / #977 T1):
    /// the running prefix sum of `M_k · r_k`, one entry per atom (the same
    /// convention as [`Self::beta_offsets`]). This is the start of each atom's
    /// `C_k` block in the reduced border vector; on the all-full-`B` path it
    /// equals `beta_offsets`. Distinct from [`Self::factored_border_offsets`]
    /// only in name (both compute the identical prefix sum) — this method is the
    /// one the frame transform reads, mirroring `beta_offsets` at the call site.
    pub fn factored_beta_offsets(&self) -> Vec<usize> {
        self.factored_border_offsets()
    }

    /// Frame output matrix `U_k ∈ St(p, r_k)` for atom `k` (issue #972 / #977 T1).
    /// Returns the active frame `U_k` (`p × r_k`) when atom `k` is framed, else
    /// the identity `I_p` (the `r_k == p`, `U_k == I_p` full-`B` special case) so
    /// the projection / lift code is uniform across a mixed dictionary.
    pub fn frame_output_matrix(&self, atom_idx: usize) -> Array2<f64> {
        let atom = &self.atoms[atom_idx];
        match &atom.decoder_frame {
            Some(frame) => frame.frame().to_owned(),
            None => Array2::<f64>::eye(atom.output_dim()),
        }
    }

    /// Per-pair frame factor `W_{ij} = U_iᵀ U_j` (`r_i × r_j`) used as the output
    /// factor of the factored data β-Hessian block `G_{ij} ⊗ W_{ij}` (issue #972
    /// / #977 T1). When both atoms are framed this is the dense principal-angle
    /// cosine matrix between the two frames; for `i == j` with an orthonormal
    /// frame it is exactly `I_{r_i}`; for any un-framed atom the corresponding
    /// `U` is `I_p`, so a same-atom un-framed pair gives `I_p` (the clean full-`B`
    /// `G ⊗ I_p` collapse) and a framed/un-framed cross pair gives the rectangular
    /// `U_iᵀ` / `U_j` overlap.
    pub fn frame_cross_factor(&self, atom_i: usize, atom_j: usize) -> Array2<f64> {
        let ui = self.frame_output_matrix(atom_i);
        let uj = self.frame_output_matrix(atom_j);
        // `U_iᵀ U_j`: `(r_i × p) · (p × r_j)`. `fast_atb` forms `U_iᵀ U_j` directly.
        fast_atb(&ui, &uj)
    }

    /// Per-atom column offsets of the *factored* border (issue #972): the
    /// running prefix sum of `M_k · r_k`. The analogue of [`Self::beta_offsets`]
    /// for the reduced coordinate layout — atom `k`'s `C_k` occupies
    /// `[factored_border_offsets()[k] .. + M_k·r_k)`. On the full-`B` path this
    /// equals `beta_offsets`.
    pub fn factored_border_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            out.push(cursor);
            cursor += atom.border_coeff_count();
        }
        out
    }

    /// Assemble the factored border coordinate vector `C = [vec(C_1); …; vec(C_K)]`
    /// in row-major `C_k[m, j] → C[off_k + m·r_k + j]` layout (issue #972).
    ///
    /// This is the reduced state the arrow-Schur border carries when frames are
    /// active: its length is [`Self::factored_border_dim`] (`Σ M_k·r_k`), the
    /// border-size invariant verified by [`grassmann_assert_border_dim_invariant`].
    /// Atoms
    /// without an active frame contribute their full `vec(B_k)` (their `r_k == p`
    /// coordinates are the decoder itself), so on the all-full-`B` path this
    /// reproduces [`Self::flatten_beta`].
    pub fn flatten_factored_border(&self) -> Result<Array1<f64>, String> {
        let offsets = self.factored_border_offsets();
        let mut out = Array1::<f64>::zeros(self.factored_border_dim());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let off = offsets[atom_idx];
            let r = atom.border_frame_rank();
            let m = atom.basis_size();
            let coords = match atom.factored_coordinates()? {
                Some(c) => c,
                // Full-`B` path: the decoder itself is the coordinate matrix.
                None => atom.decoder_coefficients.clone(),
            };
            for basis_col in 0..m {
                for j in 0..r {
                    out[off + basis_col * r + j] = coords[[basis_col, j]];
                }
            }
        }
        Ok(out)
    }

    /// Scatter a factored border coordinate vector `C` (length
    /// [`Self::factored_border_dim`]) back into the per-atom decoders, refreshing
    /// each `decoder_coefficients = C_k · U_kᵀ` so the full-`B` consumers stay
    /// consistent after a factored border solve (issue #972). The inverse of
    /// [`Self::flatten_factored_border`].
    pub fn scatter_factored_border(&mut self, border: ArrayView1<'_, f64>) -> Result<(), String> {
        let expected = self.factored_border_dim();
        if border.len() != expected {
            return Err(format!(
                "SaeManifoldTerm::scatter_factored_border: border length {} must equal \
                 factored border dim {expected}",
                border.len()
            ));
        }
        let offsets = self.factored_border_offsets();
        for atom_idx in 0..self.atoms.len() {
            let off = offsets[atom_idx];
            let (r, m, has_frame) = {
                let atom = &self.atoms[atom_idx];
                (
                    atom.border_frame_rank(),
                    atom.basis_size(),
                    atom.decoder_frame.is_some(),
                )
            };
            let mut coords = Array2::<f64>::zeros((m, r));
            for basis_col in 0..m {
                for j in 0..r {
                    coords[[basis_col, j]] = border[off + basis_col * r + j];
                }
            }
            if has_frame {
                self.atoms[atom_idx].set_factored_coordinates(coords.view())?;
            } else {
                // Full-`B` path: the coordinates ARE the decoder.
                self.atoms[atom_idx].decoder_coefficients = coords;
            }
        }
        Ok(())
    }

    /// Auto-derive and install low-rank Grassmann decoder frames across all
    /// atoms (issue #972) — magic-by-default, no flag. Each atom independently
    /// activates its frame iff the factorization materially shrinks its border
    /// (see [`SaeManifoldAtom::maybe_activate_decoder_frame`]). Returns the
    /// number of atoms that activated a frame. Idempotent: re-running re-derives
    /// each frame from the current decoder.
    ///
    /// The decision keys on the *frontier* regime the issue targets: at large
    /// ambient `p` the full border `Σ M_k · p` reaches `10^7`–`10^8` and the
    /// border Cholesky dies, while the decoder's effective column rank `r` stays
    /// `≪ p`. Small-`p` atoms (where `r` cannot beat the activation margin)
    /// keep the bit-for-bit full-`B` path, so the small-model evidence is
    /// unchanged (verified by `factored_evidence_matches_full_b_at_small_p`).
    pub fn auto_activate_decoder_frames(&mut self) -> Result<usize, String> {
        let mut activated = 0usize;
        for atom in &mut self.atoms {
            let expected_rank = atom.decoder_frame_activation_rank()?;
            match (
                expected_rank,
                atom.decoder_frame.as_ref().map(GrassmannFrame::rank),
            ) {
                (Some(expected), Some(current)) if expected == current => {
                    continue;
                }
                (None, Some(_)) => {
                    atom.deactivate_decoder_frame();
                    continue;
                }
                (None, None) => {
                    continue;
                }
                (Some(_), _) => {}
            }
            if atom.maybe_activate_decoder_frame()?.is_some() {
                activated += 1;
            }
        }
        Ok(activated)
    }

    /// Reconcile decoder-frame activation before a fit entry point. The
    /// user-facing `auto_activate_decoder_frames` contract returns only newly
    /// installed frames; this helper enforces the stronger invariant the large-p
    /// solver needs: every atom whose current decoder satisfies the activation
    /// predicate has an active frame after the pass.
    pub(crate) fn ensure_decoder_frames_active_for_current_decoder(
        &mut self,
    ) -> Result<(), String> {
        self.auto_activate_decoder_frames()?;
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let expected_rank = atom.decoder_frame_activation_rank()?;
            if let Some(expected_rank) = expected_rank {
                match atom.decoder_frame.as_ref() {
                    Some(frame) if frame.rank() == expected_rank => {}
                    Some(frame) => {
                        return Err(format!(
                            "SaeManifoldTerm::ensure_decoder_frames_active_for_current_decoder: \
                             atom {atom_idx} frame rank {} must equal audited rank {expected_rank}",
                            frame.rank()
                        ));
                    }
                    None => {
                        return Err(format!(
                            "SaeManifoldTerm::ensure_decoder_frames_active_for_current_decoder: \
                             atom {atom_idx} has audited rank {expected_rank} but no active frame"
                        ));
                    }
                }
            } else if atom.decoder_frame.is_some() {
                return Err(format!(
                    "SaeManifoldTerm::ensure_decoder_frames_active_for_current_decoder: \
                     atom {atom_idx} kept a frame after the full-B predicate won"
                ));
            }
        }
        Ok(())
    }

    /// Closed-form streaming POLAR refresh of every ACTIVE decoder frame from the
    /// current data evidence (issue #972 / #977 T1) — the U-block of the
    /// alternating block-coordinate ascent that complements the border's
    /// C-block Newton step.
    ///
    /// For each framed atom `k` we accumulate the `p × r_k` cross-moment
    ///   `A_k = Σ_n a_{n,k} · e_{n,k} · ĉ_{n,k}ᵀ`,
    /// where `e_{n,k} = z_n − Σ_{k'≠k} a_{n,k'}·decoded_{k'}(n)` is the row's
    /// partial reconstruction residual (everything except atom `k`) and
    /// `ĉ_{n,k} = Φ_k(t_n)·C_k ∈ ℝ^{r_k}` is atom `k`'s in-span decoded
    /// coordinate. The polar factor `U_new = polar(A_k)` is the closed-form MAP
    /// frame on `Gr(r_k, p)` given the C-coordinates held fixed — the same
    /// `O(p r²)` thin SVD the issue prescribes, run OUTSIDE the border. The frame
    /// is then re-installed and the decoder re-projected onto it so the
    /// authoritative `B_k = C_k U_newᵀ` and the `(C_k, U_new)` pair stay
    /// consistent (a no-op in span for a truly rank-`r` atom). Un-framed atoms
    /// are skipped. Returns the number of frames refreshed.
    pub(crate) fn refresh_active_frames_from_data(
        &mut self,
        target: ArrayView2<'_, f64>,
    ) -> Result<usize, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if n == 0 {
            return Ok(0);
        }
        // Per-row assignments and per-(row, atom) decoded outputs, computed once.
        // All three builds below are per-row independent (each row reads only
        // immutable `&self`/prior arrays and writes ONLY its own output row), so
        // the row-parallel paths are bit-identical to the serial sweeps
        // (disjoint-writes determinism — no cross-row float reduction).
        let parallel = n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        // Gate map: order-preserving parallel collect == serial push.
        let assignments = self.assignments_all_parallel(n)?;
        let mut decoded = Array3::<f64>::zeros((n, k_atoms, p));
        {
            let atoms = &self.atoms;
            if parallel {
                use rayon::prelude::*;
                decoded
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(row, mut drow)| {
                        // #1557 — pin any faer GEMM reachable via `fill_decoded_row`.
                        with_nested_parallel(|| {
                            for atom_idx in 0..k_atoms {
                                let mut arow = drow.row_mut(atom_idx);
                                let arow = arow.as_slice_mut().expect("contiguous decoded row");
                                atoms[atom_idx].fill_decoded_row(row, arow);
                            }
                        });
                    });
            } else {
                let mut dbuf = vec![0.0_f64; p];
                for row in 0..n {
                    for atom_idx in 0..k_atoms {
                        atoms[atom_idx].fill_decoded_row(row, &mut dbuf);
                        for c in 0..p {
                            decoded[[row, atom_idx, c]] = dbuf[c];
                        }
                    }
                }
            }
        }
        // Full fitted reconstruction `Σ_k a_k decoded_k`, so the per-atom partial
        // residual is `e_k = (z − fitted) + a_k decoded_k` (add atom k back in).
        let mut fitted = Array2::<f64>::zeros((n, p));
        {
            let decoded_ref = &decoded;
            let assignments_ref = &assignments;
            if parallel {
                use rayon::prelude::*;
                fitted
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(row, mut frow)| {
                        for atom_idx in 0..k_atoms {
                            let a = assignments_ref[row][atom_idx];
                            if a == 0.0 {
                                continue;
                            }
                            for c in 0..p {
                                frow[c] += a * decoded_ref[[row, atom_idx, c]];
                            }
                        }
                    });
            } else {
                for row in 0..n {
                    for atom_idx in 0..k_atoms {
                        let a = assignments[row][atom_idx];
                        if a == 0.0 {
                            continue;
                        }
                        for c in 0..p {
                            fitted[[row, c]] += a * decoded[[row, atom_idx, c]];
                        }
                    }
                }
            }
        }
        let mut refreshed = 0usize;
        for atom_idx in 0..k_atoms {
            // Only atoms with an active frame are refreshed.
            let Some(coords_c) = self.atoms[atom_idx].factored_coordinates()? else {
                continue;
            };
            let r = self.atoms[atom_idx].border_frame_rank();
            let m = self.atoms[atom_idx].basis_size();
            // Accumulate `A_k = Σ_n a_k · e_{n,k} · ĉ_{n,k}ᵀ` directly (p × r).
            let mut cross = GrassmannCrossMoment::new(p, r);
            // Build per-row p-target `a_k·e_k` and r-coord `a_k·ĉ` batched, then
            // accumulate as one outer-product sum. `accumulate` forms
            // `targetsᵀ·coords`, so scaling EITHER side by `a_k` once gives the
            // `a_k²` weight on the cross-moment that matches the C-block normal
            // equations (residual leg carries `a_k`, coordinate leg carries
            // `a_k`).
            let mut targets = Array2::<f64>::zeros((n, p));
            let mut rcoords = Array2::<f64>::zeros((n, r));
            // Per-row build of `(a_k·e_k, a_k·ĉ_k)`: each row reads only immutable
            // state and writes ONLY its own `targets`/`rcoords` rows (disjoint), so
            // the row-parallel path is bit-identical to the serial sweep. Pure
            // scalar work (no faer GEMM) — no nested-parallel guard needed.
            let atom = &self.atoms[atom_idx];
            let build_row = |row: usize, trow: &mut [f64], rrow: &mut [f64]| {
                let a = assignments[row][atom_idx];
                // Partial residual e_{n,k} = z_n − (fitted − a_k decoded_k).
                for c in 0..p {
                    let e = target[[row, c]] - fitted[[row, c]] + a * decoded[[row, atom_idx, c]];
                    trow[c] = a * e;
                }
                // In-span coordinate ĉ_{n,k} = Φ_k(t_n)·C_k ∈ ℝ^r.
                for j in 0..r {
                    let mut acc = 0.0_f64;
                    for basis_col in 0..m {
                        acc += atom.basis_values[[row, basis_col]] * coords_c[[basis_col, j]];
                    }
                    rrow[j] = a * acc;
                }
            };
            if parallel {
                use rayon::prelude::*;
                targets
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip(rcoords.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
                    .enumerate()
                    .for_each(|(row, (mut trow, mut rrow))| {
                        let trow = trow.as_slice_mut().expect("contiguous targets row");
                        let rrow = rrow.as_slice_mut().expect("contiguous rcoords row");
                        build_row(row, trow, rrow);
                    });
            } else {
                for row in 0..n {
                    let mut trow = targets.row_mut(row);
                    let trow = trow.as_slice_mut().expect("contiguous targets row");
                    let mut rrow = rcoords.row_mut(row);
                    let rrow = rrow.as_slice_mut().expect("contiguous rcoords row");
                    build_row(row, trow, rrow);
                }
            }
            cross.accumulate(targets.view(), rcoords.view())?;
            // `polar(A_k)` is well-defined only when the moment is non-trivial;
            // a zero moment (e.g. a fully collapsed atom) leaves the frame as-is.
            if cross.moment().iter().all(|&v| v == 0.0) {
                continue;
            }
            self.atoms[atom_idx].refresh_frame_from_cross_moment(cross.moment())?;
            refreshed += 1;
        }
        Ok(refreshed)
    }

    pub fn beta_offsets(&self) -> Vec<usize> {
        let p = self.output_dim();
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            out.push(cursor);
            cursor += atom.basis_size() * p;
        }
        out
    }

    /// Per-atom β column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// Returns one `Range<usize>` per atom, covering that atom's decoder
    /// coefficients in the flat β vector:
    ///   `[beta_offsets[k] .. beta_offsets[k] + basis_size[k] * p_out]`.
    ///
    /// Pass to [`ArrowSchurSystem::set_block_offsets`] so that
    /// [`gam_solve::arrow_schur::JacobiPreconditioner`] builds one dense
    /// Schur sub-block per atom instead of scalar-diagonal inversion.
    pub fn beta_block_offsets(&self) -> Arc<[std::ops::Range<usize>]> {
        let p = self.output_dim();
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            let width = atom.basis_size() * p;
            ranges.push(cursor..cursor + width);
            cursor += width;
        }
        Arc::from(ranges.into_boxed_slice())
    }

    /// Irreducible resident bytes for exact full-support row curvature and the
    /// decoder data Gram. This is a lower bound used for admission before either
    /// allocation; saturating arithmetic turns dimension overflow into refusal.
    pub(crate) fn exact_dense_assignment_bytes(&self) -> usize {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let m_total: usize = self.atoms.iter().map(|atom| atom.basis_size()).sum();
        n.saturating_mul(q)
            .saturating_mul(q)
            .saturating_mul(SAE_BYTES_PER_F64)
            .saturating_add(
                m_total
                    .saturating_mul(m_total)
                    .saturating_mul(SAE_BYTES_PER_F64),
            )
    }

    /// Enforce exact dense-assignment admission. Smooth assignment families are
    /// never converted into an active-set surrogate to fit memory.
    pub(crate) fn require_exact_dense_assignment_budget(
        &self,
        budget_bytes: usize,
    ) -> Result<(), String> {
        let family = match self.assignment.mode {
            AssignmentMode::TopK { .. } => return Ok(()),
            AssignmentMode::Softmax { .. } => "softmax",
            AssignmentMode::OrderedBetaBernoulli { .. } => "ordered_beta_bernoulli",
            AssignmentMode::ThresholdGate { .. } => "threshold_gate",
        };
        let required_bytes = self.exact_dense_assignment_bytes();
        if required_bytes <= budget_bytes {
            return Ok(());
        }
        Err(format!(
            "exact {family} assignment assembly requires at least {required_bytes} bytes for row curvature and decoder Gram, exceeding the in-core budget {budget_bytes} bytes at N={}, K={}. Use assignment='topk' with an explicit support size; smooth assignments are never silently truncated",
            self.n_obs(),
            self.k_atoms(),
        ))
    }

    pub fn flatten_beta(&self) -> Array1<f64> {
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        let mut out = Array1::<f64>::zeros(self.beta_dim());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            for basis_col in 0..m {
                for out_col in 0..p {
                    out[off + basis_col * p + out_col] =
                        atom.decoder_coefficients[[basis_col, out_col]];
                }
            }
        }
        out
    }

    pub fn set_flat_beta(&mut self, beta: ArrayView1<'_, f64>) -> Result<(), String> {
        if beta.len() != self.beta_dim() {
            return Err(format!(
                "set_flat_beta: beta length {} != expected {}",
                beta.len(),
                self.beta_dim()
            ));
        }
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        for (atom_idx, atom) in self.atoms.iter_mut().enumerate() {
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            for basis_col in 0..m {
                for out_col in 0..p {
                    atom.decoder_coefficients[[basis_col, out_col]] =
                        beta[off + basis_col * p + out_col];
                }
            }
        }
        Ok(())
    }

    pub fn refit_decoder_least_squares_at_current_state(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: Option<&SaeManifoldRho>,
    ) -> Result<(), String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::refit_decoder_least_squares_at_current_state: target shape {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        let k_atoms = self.k_atoms();
        let offsets = self.beta_offsets();
        let m_total = self.beta_dim() / p;
        let mut design = Array2::<f64>::zeros((n, m_total));
        for row in 0..n {
            let assignments = match rho {
                Some(_) => self.assignment.try_assignments_row(row)?,
                None => self.assignment.try_assignments_row(row)?,
            };
            for atom_idx in 0..k_atoms {
                let atom = &self.atoms[atom_idx];
                let weight = assignments[atom_idx];
                let m = atom.basis_size();
                let off = offsets[atom_idx] / p;
                for basis_col in 0..m {
                    design[[row, off + basis_col]] = weight * atom.basis_values[[row, basis_col]];
                }
            }
        }
        let beta = solve_design_least_squares(design.view(), target)?;
        if beta.dim() != (m_total, p) {
            return Err(format!(
                "SaeManifoldTerm::refit_decoder_least_squares_at_current_state: beta shape {:?} != ({m_total}, {p})",
                beta.dim()
            ));
        }
        for atom_idx in 0..k_atoms {
            let m = self.atoms[atom_idx].basis_size();
            let off = offsets[atom_idx] / p;
            for basis_col in 0..m {
                for out_col in 0..p {
                    self.atoms[atom_idx].decoder_coefficients[[basis_col, out_col]] =
                        beta[[off + basis_col, out_col]];
                }
            }
        }
        Ok(())
    }

    pub fn fitted(&self) -> Array2<f64> {
        self.try_fitted().expect(
            "fitted reconstruction requires finite assignments and no target-dependent rescue",
        )
    }

    /// The #1026 hybrid-collapse substitution map: `atom_idx → &AtomLinearImage`
    /// for every `d = 1` slot whose post-fit verdict selected its straight
    /// (`Θ → 0`) sub-model. Empty when no report has been computed
    /// (`hybrid_split_report == None`, e.g. mid-fit) or no slot collapsed. The
    /// SINGLE source of the collapse policy — every reconstruction path (the
    /// rho-keyed `try_fitted_with_rho` and the explicit-assignment
    /// [`Self::reconstruct_from_assignments`]) reads it so every reconstruction decodes collapsed slots
    /// identically (#1228, #1233).
    pub(crate) fn hybrid_linear_image_map(
        &self,
    ) -> std::collections::HashMap<usize, &crate::hybrid_split::AtomLinearImage> {
        // A fitted term carries its collapse policy on the post-fit
        // `hybrid_split_report`; an OOS term carries the same trained images on
        // `oos_linear_images` (#1228). At most one is `Some` in practice, but
        // prefer the report when both are present.
        if let Some(report) = self.hybrid_split_report.as_ref() {
            return report
                .verdicts
                .iter()
                .filter_map(|v| v.linear_image.as_ref().map(|img| (img.atom_idx, img)))
                .collect();
        }
        if let Some(images) = self.oos_linear_images.as_ref() {
            return images.iter().map(|img| (img.atom_idx, img)).collect();
        }
        std::collections::HashMap::new()
    }

    /// #1228 — attach the trained dictionary's hybrid-collapsed linear images to
    /// this (typically OOS) term so target-aware reconstruction decodes
    /// verdict-linear `d = 1` slots by the SAME straight sub-model the training
    /// reconstruction used, instead of the original curved decoder. Each image's
    /// `atom_idx` must be unique and index a real slot; an image whose channel
    /// count `p` disagrees with this term's output dim, or whose `atom_idx` is out
    /// of range, is rejected so a stale/mismatched payload cannot silently corrupt
    /// the reconstruction. Pass an empty vector (or never call this) for an
    /// all-curved OOS reconstruction.
    ///
    /// `pub` (not `pub(crate)`): this is part of the FFI surface — the gam-pyffi
    /// crate calls it from `latent_basis_and_sae_ffi.rs` to attach a trained
    /// dictionary's hybrid-linear images to an OOS reconstruction term (#1228).
    /// Downgrading it to `pub(crate)` breaks the gam-pyffi cdylib build with
    /// E0624 (the gam lib still compiles, so the lib build does not catch it).
    pub fn set_hybrid_linear_images(
        &mut self,
        images: Vec<crate::hybrid_split::AtomLinearImage>,
    ) -> Result<(), String> {
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let mut seen = std::collections::HashSet::with_capacity(images.len());
        for img in &images {
            if !seen.insert(img.atom_idx) {
                return Err(format!(
                    "set_hybrid_linear_images: duplicate image for atom {}",
                    img.atom_idx
                ));
            }
            if img.atom_idx >= k_atoms {
                return Err(format!(
                    "set_hybrid_linear_images: atom_idx {} out of range (k_atoms={k_atoms})",
                    img.atom_idx
                ));
            }
            if img.b0.len() != p || img.b1.len() != p {
                return Err(format!(
                    "set_hybrid_linear_images: atom {} linear image has p=({}, {}) != output_dim {p}",
                    img.atom_idx,
                    img.b0.len(),
                    img.b1.len()
                ));
            }
            // #1777 — a collapse-rescued image's projection direction `v` must
            // have one entry per output channel so `coordinate_from_residual` can
            // project a held-out row's `p`-vector residual onto it.
            if let Some(v) = img.v.as_ref() {
                if v.len() != p {
                    return Err(format!(
                        "set_hybrid_linear_images: atom {} projection direction v has len {} != output_dim {p}",
                        img.atom_idx,
                        v.len()
                    ));
                }
            }
            if self.atoms[img.atom_idx].latent_dim != 1 {
                return Err(format!(
                    "set_hybrid_linear_images: atom {} is not d=1; only d=1 slots collapse to a straight image",
                    img.atom_idx
                ));
            }
        }
        self.oos_linear_images = if images.is_empty() {
            None
        } else {
            Some(images)
        };
        Ok(())
    }

    /// Assemble the reconstruction `Σ_k a[i,k]·g_k(t_{ik})` from an EXPLICIT
    /// per-row assignment matrix, honouring the #1026 hybrid collapse when `collapse` is
    /// set: a verdict-linear `d = 1` slot decodes its straight sub-model image
    /// instead of its curved curve, exactly as the production `try_fitted` does.
    /// This shared assembler prevents callers from re-deriving the curved image
    /// by hand and silently bypassing the verdict.
    /// The atom coordinates (`t`) and decoded curves are the term's own fitted
    /// ones; only the assignment masses come from `assignments`. Because this
    /// entry point has no target, it explicitly refuses a collapse-rescued image;
    /// callers with a target must use
    /// [`Self::reconstruct_from_assignments_target_aware`].
    pub fn reconstruct_from_assignments(
        &self,
        assignments: ArrayView2<'_, f64>,
        collapse: bool,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if assignments.dim() != (n, k_atoms) {
            return Err(format!(
                "SaeManifoldTerm::reconstruct_from_assignments: assignments {:?} != ({n}, {k_atoms})",
                assignments.dim()
            ));
        }
        let linear_images = if collapse {
            self.hybrid_linear_image_map()
        } else {
            std::collections::HashMap::new()
        };
        if let Some(image) = linear_images
            .values()
            .find(|image| image.is_collapse_rescued())
        {
            return Err(format!(
                "SaeManifoldTerm::reconstruct_from_assignments: collapse-rescued atom {} requires reconstruct_from_assignments_target_aware",
                image.atom_idx
            ));
        }
        let mut out = Array2::<f64>::zeros((n, p));
        // Per-row reconstruction: each row reads only immutable `&self`/`assignments`
        // state and writes ONLY its own `out` row (a per-row accumulation over atoms,
        // never a cross-row float reduction), so the row-parallel path is bit-identical
        // to the serial sweep (disjoint-writes determinism). Structural twin of the
        // reconstruction in `try_fitted_with_rho`; the only difference is that the
        // per-row mass here is read straight from the `assignments` view (no per-row
        // `?`), so the closure is infallible.
        let fill_out_row = |row: usize, out_row: &mut [f64], g_buf: &mut [f64]| {
            for atom_idx in 0..k_atoms {
                let a_k = assignments[[row, atom_idx]];
                if a_k == 0.0 {
                    continue;
                }
                if let Some(image) = linear_images.get(&atom_idx) {
                    let own_t = self.assignment.coords[atom_idx].as_matrix()[[row, 0]];
                    image.fill_row(own_t, g_buf);
                } else {
                    self.atoms[atom_idx].fill_decoded_row(row, g_buf);
                }
                for out_col in 0..p {
                    out_row[out_col] += a_k * g_buf[out_col];
                }
            }
        };
        let parallel = n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 32;
            // #1557 — pin any faer GEMM reached via `fill_decoded_row` / `image.fill_row`
            // to `Par::Seq` so nested faer does not re-fan the pool (bit-identical).
            out.axis_chunks_iter_mut(ndarray::Axis(0), CHUNK)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk, mut block)| {
                    with_nested_parallel(|| {
                        let start = chunk * CHUNK;
                        let mut g_buf = vec![0.0_f64; p];
                        for local in 0..block.nrows() {
                            let row = start + local;
                            let mut out_row = block.row_mut(local);
                            let out_row = out_row.as_slice_mut().expect("contiguous out row");
                            fill_out_row(row, out_row, &mut g_buf);
                        }
                    });
                });
        } else {
            let mut g_buf = vec![0.0_f64; p];
            for row in 0..n {
                let mut out_row = out.row_mut(row);
                let out_row = out_row.as_slice_mut().expect("contiguous out row");
                fill_out_row(row, out_row, &mut g_buf);
            }
        }
        // #2023 C4 — Tier-0 shared mean add-back (no-op when inactive).
        self.add_tier0_mean_inplace(&mut out);
        Ok(out)
    }

    /// Assemble a hybrid-collapsed reconstruction from explicit assignment
    /// masses and the response being reconstructed. Ordinary straight images use
    /// the atom's realized coordinate. A collapse-rescued image derives every
    /// coordinate from that row's leave-this-atom-out residual projected onto its
    /// persisted direction `v`; no train-row coordinate cache exists.
    pub fn reconstruct_from_assignments_target_aware(
        &self,
        target: ArrayView2<'_, f64>,
        assignments: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if target.dim() != (n, p) || assignments.dim() != (n, k_atoms) {
            return Err(format!(
                "SaeManifoldTerm::reconstruct_from_assignments_target_aware: target={:?}, assignments={:?} disagree with ({n}, {p}) and ({n}, {k_atoms})",
                target.dim(),
                assignments.dim()
            ));
        }
        let linear_images = self.hybrid_linear_image_map();
        let full_curved = self.reconstruct_from_assignments(assignments, false)?;
        if linear_images.is_empty() {
            return Ok(full_curved);
        }

        let mut out = Array2::<f64>::zeros((n, p));
        let mut decoded = vec![0.0_f64; p];
        let mut image_row = vec![0.0_f64; p];
        let mut residual = vec![0.0_f64; p];
        for row in 0..n {
            for atom_idx in 0..k_atoms {
                let mass = assignments[[row, atom_idx]];
                if mass == 0.0 {
                    continue;
                }
                if let Some(image) = linear_images.get(&atom_idx) {
                    let coordinate = if image.is_collapse_rescued() {
                        self.atoms[atom_idx].fill_decoded_row(row, &mut decoded);
                        for output in 0..p {
                            residual[output] = target[[row, output]] - full_curved[[row, output]]
                                + mass * decoded[output];
                        }
                        image.coordinate_from_residual(&residual).ok_or_else(|| {
                            format!(
                                "SaeManifoldTerm::reconstruct_from_assignments_target_aware: collapse-rescued atom {atom_idx} cannot project a {p}-channel residual"
                            )
                        })?
                    } else {
                        self.assignment.coords[atom_idx].as_matrix()[[row, 0]]
                    };
                    image.fill_row(coordinate, &mut image_row);
                } else {
                    self.atoms[atom_idx].fill_decoded_row(row, &mut image_row);
                }
                for output in 0..p {
                    out[[row, output]] += mass * image_row[output];
                }
            }
        }
        self.add_tier0_mean_inplace(&mut out);
        Ok(out)
    }

    /// #1777 — TARGET-AWARE hybrid-collapsed reconstruction: identical to
    /// [`Self::try_fitted`] except that a #1026 COLLAPSE-RESCUED `d = 1` slot
    /// (whose linear image carries a projection direction `v`) recomputes each
    /// row's coordinate from THIS `target` as
    /// `uᵢ = ⟨y_i − Σ_{j≠k} f_j(x_i), v⟩` — its own leave-this-atom-out residual
    /// projected onto `v`. This projection is the only collapse-rescue coordinate
    /// model; there is no train-row cache or own-coordinate substitute.
    ///
    /// This is the SAME math the train split used to fit the image, so train and
    /// held-out rows use one model. Ordinary (non-rescued) straight images and
    /// curved slots are decoded exactly as in [`Self::try_fitted`]; they ignore
    /// `target`.
    ///
    /// `rho` selects the assignment-mass resolution (`Some` uses the ρ-keyed
    /// gates, `None` the persisted gates), mirroring [`Self::try_fitted_with_rho`].
    /// This is the reconstruction path an OOS predict should call once the trained
    /// hybrid-linear images are attached via [`Self::set_hybrid_linear_images`].
    pub fn try_fitted_target_aware(
        &self,
        target: ArrayView2<'_, f64>,
        rho: Option<&SaeManifoldRho>,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::try_fitted_target_aware: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        let linear_images = self.hybrid_linear_image_map();
        // The all-curved reconstruction `full = Σ_j a_j·γ_j`, the same quantity the
        // train split's `target_resid_for` subtracts. A rescued slot `k`'s
        // leave-this-atom-out residual is then `target − full + a_k·γ_k`.
        let full_curved = self.try_fitted_with_rho(rho, false)?;
        let mut out = Array2::<f64>::zeros((n, p));
        let mut g_buf = vec![0.0_f64; p];
        let mut decoded_buf = vec![0.0_f64; p];
        let mut resid_buf = vec![0.0_f64; p];
        for row in 0..n {
            let a = match rho {
                Some(_) => self.assignment.try_assignments_row(row)?,
                None => self.assignment.try_assignments_row(row)?,
            };
            for atom_idx in 0..k_atoms {
                let a_k = a[atom_idx];
                if let Some(image) = linear_images.get(&atom_idx) {
                    if image.is_collapse_rescued() {
                        // Recompute this row's coordinate from its own
                        // leave-this-atom-out residual projected onto `v`.
                        self.atoms[atom_idx].fill_decoded_row(row, &mut decoded_buf);
                        for col in 0..p {
                            resid_buf[col] = target[[row, col]] - full_curved[[row, col]]
                                + a_k * decoded_buf[col];
                        }
                        let coord = image.coordinate_from_residual(&resid_buf).ok_or_else(|| {
                            format!(
                                "SaeManifoldTerm::try_fitted_target_aware: collapse-rescued atom {atom_idx} cannot project a {p}-channel residual"
                            )
                        })?;
                        image.fill_row(coord, &mut g_buf);
                    } else {
                        // Ordinary straight image: decode at the atom's own coord.
                        let own_t = self.assignment.coords[atom_idx].as_matrix()[[row, 0]];
                        image.fill_row(own_t, &mut g_buf);
                    }
                } else {
                    self.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                }
                let mut out_row = out.row_mut(row);
                for out_col in 0..p {
                    out_row[out_col] += a_k * g_buf[out_col];
                }
            }
        }
        // #2023 C4 — Tier-0 shared mean add-back (no-op when inactive).
        self.add_tier0_mean_inplace(&mut out);
        Ok(out)
    }

    pub fn try_fitted(&self) -> Result<Array2<f64>, String> {
        // Production/user-facing reconstruction: honours the #1026 hybrid-split
        // verdict (verdict-linear `d = 1` slots decode their straight sub-model).
        self.try_fitted_with_rho(None, true)
    }

    pub fn try_fitted_for_rho(&self, rho: &SaeManifoldRho) -> Result<Array2<f64>, String> {
        // Fitting reconstruction: the pure CURVED image at a specific `rho` (the
        // joint fit and the #1026 adjudication both require the uncollapsed
        // curve). Exposed for callers that need the rho-specific curved image
        // rather than the collapse-adjudicated production `try_fitted`.
        self.try_fitted_with_rho(Some(rho), false)
    }

    pub(crate) fn try_fitted_with_rho(
        &self,
        rho: Option<&SaeManifoldRho>,
        collapse: bool,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, p));
        // #1026 — the curved/linear hybrid-split verdict is LOAD-BEARING on the
        // production reconstruction, not just a side report. When
        // [`Self::compute_hybrid_split_report`] (run post-fit in
        // `canonicalize_charts_post_fit`) adjudicated a `d = 1` atom's evidence
        // in favour of its straight (Θ→0) sub-model, the model's output
        // reconstruction (`fitted()` / `try_fitted` → predict and the user-facing
        // output) decodes that slot with its fitted linear image instead of its
        // curved decoded curve. The linear images are coordinate-keyed and
        // rho-independent (exact weighted-LS lines realised inside the
        // adjudication — no re-fit, no #1051 outer continuation).
        //
        // The collapse engages only when the caller asks for it (`collapse`):
        // the production `try_fitted` path and the explicit
        // `hybrid_collapsed_reconstruction` entry point. The pure-curved
        // `try_fitted_for_rho` opts out — the joint fit's loss/assembly optimise
        // the curved decoder coefficients and must see the curved image, and the
        // #1026 adjudication itself compares the curved fit against its straight
        // sub-model — both require the uncollapsed curve. (During fitting the
        // report is `None` regardless; it is only computed post-fit.)
        let linear_images = if collapse {
            self.hybrid_linear_image_map()
        } else {
            std::collections::HashMap::new()
        };
        if let Some(image) = linear_images
            .values()
            .find(|image| image.is_collapse_rescued())
        {
            return Err(format!(
                "SaeManifoldTerm::try_fitted: collapse-rescued atom {} requires try_fitted_target_aware",
                image.atom_idx
            ));
        }
        // Reuse a single scratch buffer across all (row, atom) pairs instead of
        // allocating a fresh `Array1<f64>` of length p per call.
        //
        // Per-row reconstruction: each row reads only immutable `&self` state and
        // writes ONLY its own `out` row. Every output cell is written exactly once
        // (a per-row accumulation over atoms — never a cross-row float reduction),
        // so the row-parallel path is bit-identical to the serial sweep
        // (disjoint-writes determinism).
        let fill_out_row =
            |row: usize, out_row: &mut [f64], g_buf: &mut [f64]| -> Result<(), String> {
                let a = match rho {
                    Some(_) => self.assignment.try_assignments_row(row)?,
                    None => self.assignment.try_assignments_row(row)?,
                };
                for atom_idx in 0..k_atoms {
                    let a_k = a[atom_idx];
                    if let Some(image) = linear_images.get(&atom_idx) {
                        // Verdict-linear slot: substitute the straight sub-model
                        // image at this row's fitted on-atom coordinate. Rescued
                        // images were refused above because they require a target.
                        let own_t = self.assignment.coords[atom_idx].as_matrix()[[row, 0]];
                        image.fill_row(own_t, g_buf);
                    } else {
                        self.atoms[atom_idx].fill_decoded_row(row, g_buf);
                    }
                    for out_col in 0..p {
                        out_row[out_col] += a_k * g_buf[out_col];
                    }
                }
                Ok(())
            };
        let parallel = n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 32;
            // Disjoint row-block writes via `axis_chunks_iter_mut`; per-worker
            // `g_buf` scratch. #1557 — wrap the chunk body in `with_nested_parallel`
            // so any faer GEMM reached via `fill_decoded_row` / `image.fill_row`
            // pins to `Par::Seq` rather than re-fanning the pool (bit-identical).
            out.axis_chunks_iter_mut(ndarray::Axis(0), CHUNK)
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk, mut block)| -> Result<(), String> {
                    with_nested_parallel(|| {
                        let start = chunk * CHUNK;
                        let mut g_buf = vec![0.0_f64; p];
                        for local in 0..block.nrows() {
                            let row = start + local;
                            let mut out_row = block.row_mut(local);
                            let out_row = out_row.as_slice_mut().expect("contiguous out row");
                            fill_out_row(row, out_row, &mut g_buf)?;
                        }
                        Ok(())
                    })
                })?;
        } else {
            let mut g_buf = vec![0.0_f64; p];
            for row in 0..n {
                let mut out_row = out.row_mut(row);
                let out_row = out_row.as_slice_mut().expect("contiguous out row");
                fill_out_row(row, out_row, &mut g_buf)?;
            }
        }
        // #2023 C4 — Tier-0 shared mean add-back (no-op when inactive).
        self.add_tier0_mean_inplace(&mut out);
        Ok(out)
    }

    /// Per-atom **leave-one-atom-out (LOAO) explained-variance contribution**
    /// (#1026): for each atom `k`, the drop in reconstruction explained variance
    /// `ΔEV_k = EV(full) − EV(full ⊖ atom_k)` when that atom's contribution
    /// `a[i,k]·g_k(coord[i,k])` is removed from the assembled reconstruction and
    /// nothing else is refit. Because every atom adds linearly into the same
    /// fitted reconstruction (`fitted[i] = Σ_k a[i,k]·g_k`), zeroing one atom is
    /// the exact "this atom withheld" counterfactual, and the EV it was earning
    /// is `EV(full) − EV(without k)`. This is the per-atom held-out EV
    /// attribution the #1026 roadmap pairs with each atom's fitted turning `Θ`:
    /// a `Θ ≈ 0` atom earning a large `ΔEV` is a linear-tail direction; a
    /// high-`Θ` atom earning a large `ΔEV` is a genuine curved family carrying
    /// reconstruction it would otherwise shatter into `N(ε) ≈ Θ/(2√(2ε))` linear
    /// directions. Pure read-only diagnostic — never mutates any atom.
    ///
    /// Returns one `Option<f64>` per atom in atom order; `None` for an atom
    /// whose ⊖-reconstruction EV is undefined (degenerate target variance), and
    /// `None` for the whole vector if the full-reconstruction EV is undefined.
    /// #1026: the load-bearing curved-vs-linear hybrid-split verdict for the
    /// fitted dictionary, or `None` until [`Self::canonicalize_charts_post_fit`]
    /// has run (or when no `d = 1` atom is eligible). Surfaced in the Python model
    /// output so the user sees which atoms genuinely earn their curvature.
    pub fn hybrid_split_report(&self) -> Option<&crate::hybrid_split::SaeHybridSplitReport> {
        self.hybrid_split_report.as_ref()
    }

    /// Build the #1026 curved-vs-linear hybrid-split report by adjudicating each
    /// eligible `d = 1` atom's fitted curved image against its straight (linear
    /// special-case) sub-model on the common rank-aware quasi-Laplace score scale.
    ///
    /// Both candidates are scored against the SAME data — the atom's
    /// leave-this-atom-out response residual `y_resp = target − (full − a_k·γ_k)`
    /// (#1202) — over its assigned rows: the curved candidate predicts its actual
    /// mass-scaled contribution `a_k·γ_k`, the linear candidate the best
    /// mass-weighted straight line fit to `y_resp` (the collapsed linear lane —
    /// closed form, NOT the broken euclidean outer fit path of #1051). Linear is
    /// the curved family's nested `Θ = 0` sub-model on common data, so the
    /// per-slot evidence argmin is a genuine match-or-beat comparison. Eligible
    /// atoms are `d = 1` atoms with an installed evaluator at the full curvature
    /// dial (`homotopy_eta == 1.0`) whose live coordinate dim still matches the
    /// atom's latent dim. Returns `None` when no reconstruction `target` is
    /// supplied (there is no data to adjudicate against).
    pub fn compute_hybrid_split_report(
        &self,
        rho: &SaeManifoldRho,
        target: Option<ArrayView2<'_, f64>>,
    ) -> Result<Option<crate::hybrid_split::SaeHybridSplitReport>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        // Per-atom held-out `ΔEV_k` (leave-one-atom-out explained-variance drop),
        // paired with each atom's fitted turning Θ onto the verdict so the report
        // carries the #1026 `(Θ, ΔEV)` frontier point as structured data. Absent
        // when no reconstruction target is supplied.
        let loao_ev: Vec<Option<f64>> = match target {
            Some(t) => self.per_atom_loao_explained_variance(t, rho)?,
            None => vec![None; self.k_atoms()],
        };
        let delta_ev_for =
            |atom_idx: usize| -> Option<f64> { loao_ev.get(atom_idx).copied().flatten() };
        // The common-evidence comparison (#1202) scores both candidates against
        // the response data the atom is responsible for. That requires a target;
        // with none supplied there is nothing to adjudicate against, so no report.
        let Some(target) = target else {
            return Ok(None);
        };
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::compute_hybrid_split_report: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        // Per-row assignment masses (once), so each atom's weighted straight-line
        // fit uses the same row weighting the joint reconstruction loss does.
        let mut weights: Vec<Array1<f64>> = Vec::with_capacity(n);
        for row in 0..n {
            weights.push(self.assignment.try_assignments_row(row)?);
        }
        // The full assembled reconstruction `Σ_k a[i,k]·γ_k`, computed once. Each
        // atom's leave-this-atom-out response residual is `y_resp = target −
        // (full − a_k·γ_k)`, the data both that atom's candidates fit (#1202).
        let full = self.try_fitted_for_rho(rho)?;
        let eligible: Vec<usize> = (0..self.k_atoms())
            .filter(|&atom_idx| {
                let atom = &self.atoms[atom_idx];
                atom.latent_dim == 1
                    && atom.basis_evaluator.is_some()
                    && atom.homotopy_eta == 1.0
                    && self.assignment.coords[atom_idx].latent_dim() == atom.latent_dim
            })
            .collect();
        // Per-atom fitted decoded image at every row (the curved candidate's
        // realized curve, which the linear candidate must approximate).
        let coords_for = |atom_idx: usize| -> Array1<f64> {
            self.assignment.coords[atom_idx]
                .as_matrix()
                .column(0)
                .to_owned()
        };
        let assign_for = |atom_idx: usize| -> Array1<f64> {
            Array1::from_iter((0..n).map(|row| weights[row][atom_idx]))
        };
        let decoded_for = |atom_idx: usize| -> Array2<f64> {
            let mut decoded = Array2::<f64>::zeros((n, p));
            let mut buf = vec![0.0_f64; p];
            for row in 0..n {
                self.atoms[atom_idx].fill_decoded_row(row, &mut buf);
                for col in 0..p {
                    decoded[[row, col]] = buf[col];
                }
            }
            decoded
        };
        // The atom's leave-this-atom-out response residual `y_resp = target −
        // (full − a_k·γ_k) = (target − full) + a_k·γ_k`. Both the curved and the
        // linear candidate are scored against this on common data (#1202).
        let target_resid_for = |atom_idx: usize| -> Array2<f64> {
            let mut resid = Array2::<f64>::zeros((n, p));
            let mut buf = vec![0.0_f64; p];
            for row in 0..n {
                let a_k = weights[row][atom_idx];
                self.atoms[atom_idx].fill_decoded_row(row, &mut buf);
                for col in 0..p {
                    resid[[row, col]] = target[[row, col]] - full[[row, col]] + a_k * buf[col];
                }
            }
            resid
        };
        let manifold_for = |atom_idx: usize| -> gam_terms::latent::LatentManifold {
            self.assignment.coords[atom_idx].manifold().clone()
        };
        // #1026 EV-preservation gate denominator: the full target's total
        // column-centered variance `SST_full` (the SAME `sst` the reconstruction
        // EV is measured against), so the gate vetoes any collapse that would drop
        // full-reconstruction EV by more than its tolerance.
        let total_centered_variance = {
            let mut tss = 0.0_f64;
            for col in 0..p {
                let mut mean = 0.0_f64;
                for row in 0..n {
                    mean += target[[row, col]];
                }
                mean /= n as f64;
                for row in 0..n {
                    let c = target[[row, col]] - mean;
                    tss += c * c;
                }
            }
            tss
        };
        // #16 DEMOTE rank-charge noise floor: the full-reconstruction residual
        // variance φ̂ = ‖target − full‖² / (n·p). This is tier2's sanctioned fallback
        // for the MP edge when the term's exact reconstruction_dispersion isn't in
        // scope at the hybrid-split site; the MP rank count is R-robust for real
        // (signal ≫ noise) circles, so the demote decision is currency-consistent.
        let dispersion_r = {
            let mut rss = 0.0_f64;
            for row in 0..n {
                for col in 0..p {
                    let r = target[[row, col]] - full[[row, col]];
                    rss += r * r;
                }
            }
            let denom = (n * p).max(1) as f64;
            rss / denom
        };
        crate::hybrid_split::build_hybrid_split_report(
            &self.atoms,
            eligible.into_iter(),
            coords_for,
            assign_for,
            decoded_for,
            target_resid_for,
            manifold_for,
            delta_ev_for,
            total_centered_variance,
            n,
            dispersion_r,
        )
    }

    pub fn per_atom_loao_explained_variance(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Option<f64>>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::per_atom_loao_explained_variance: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        let full = self.try_fitted_for_rho(rho)?;
        let Some(ev_full) = reconstruction_explained_variance(target, full.view()) else {
            return Ok(vec![None; k_atoms]);
        };
        // Cache each row's assignment weights once, then subtract a single
        // atom's decoded contribution per LOAO pass instead of reassembling the
        // whole dictionary k times.
        let mut weights: Vec<Array1<f64>> = Vec::with_capacity(n);
        for row in 0..n {
            weights.push(self.assignment.try_assignments_row(row)?);
        }
        let mut g_buf = vec![0.0_f64; p];
        let mut out = Vec::with_capacity(k_atoms);
        for atom_idx in 0..k_atoms {
            let mut without = full.clone();
            for row in 0..n {
                let a_k = weights[row][atom_idx];
                if a_k == 0.0 {
                    continue;
                }
                self.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                let mut without_row = without.row_mut(row);
                for out_col in 0..p {
                    without_row[out_col] -= a_k * g_buf[out_col];
                }
            }
            out.push(
                reconstruction_explained_variance(target, without.view())
                    .map(|ev_without| ev_full - ev_without),
            );
        }
        Ok(out)
    }

    /// #1026 — the LOAD-BEARING collapsed reconstruction: the assembled
    /// dictionary output `Σ_k a[i,k]·g_k(coord[i,k])` in which every slot whose
    /// hybrid-split verdict selected LINEAR has its curved decoded image replaced
    /// by its fitted straight sub-model `b₀ + (t − t̄)·b₁`. This is what makes the
    /// verdict *change the reconstruction* instead of merely logging a choice:
    /// the linear-collapsed atom no longer pays its `M·p` curved coefficients, it
    /// carries a `2·p` straight image whose decoded curve has zero turning.
    ///
    /// The straight images are the exact weighted-least-squares lines already
    /// realized inside [`Self::compute_hybrid_split_report`] (no re-fit, no outer
    /// continuation, sidestepping #1051). Returns the curved reconstruction
    /// unchanged when no verdict selected linear, or when the report has not been
    /// computed yet (`hybrid_split_report == None`). A collapse-rescued image is
    /// refused because this method has no target from which to derive its
    /// coordinate; use [`Self::try_fitted_target_aware`] instead.
    pub fn hybrid_collapsed_reconstruction(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Array2<f64>, String> {
        // #1026 — the hybrid collapse is realised by the SINGLE reconstruction
        // path ([`Self::try_fitted_with_rho`]) with the collapse flag set: a
        // verdict-linear `d = 1` slot decodes its straight sub-model image
        // instead of its curved curve. This replaces the dedicated re-collapse
        // loop this method used to carry (a parallel layer). The production
        // `try_fitted` shares the identical routine at `rho = None`; this entry
        // point keeps the rho-keyed, target-less collapse for callers whose
        // report contains only ordinary straight images.
        self.try_fitted_with_rho(Some(rho), true)
    }

    /// #1026 — the reconstruction explained variance of the hybrid-collapsed
    /// dictionary (every verdict-linear slot decoded by its straight sub-model)
    /// against `target`. The companion of [`Self::per_atom_loao_explained_variance`]
    /// for the dominance claim: because each linear-collapsed slot is the curved
    /// family's `Θ → 0` sub-model and is only kept when its evidence beats the
    /// curved candidate's parameter price, the collapsed dictionary match-or-beats
    /// the all-curved one on EV-per-parameter — the strict-generalization floor
    /// the #1026 hybrid argument rests on. `None` when EV is undefined (degenerate
    /// target variance).
    pub fn hybrid_collapsed_explained_variance(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<Option<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::hybrid_collapsed_explained_variance: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        let collapsed = self.try_fitted_target_aware(target, Some(rho))?;
        Ok(reconstruction_explained_variance(target, collapsed.view()))
    }

    /// #1026 ladder item 2/3 — the AMORTIZED ENCODER, wired from the fitted
    /// dictionary. Builds the offline certified [`EncodeAtlas`] over this term's
    /// frozen atoms and encodes a target corpus `targets` (`n × p`) through the
    /// per-chart distilled Jacobian predictor, with the Kantorovich certificate
    /// supplying chart-aware starts. Those starts are then refined together by
    /// the frozen dictionary's shared-residual objective. The returned
    /// [`JointEncodeResult`] carries one coordinate block per atom plus a
    /// numerical joint-stationarity mask; it does not mislabel a composition of
    /// per-atom certificates as a certificate for the multi-atom problem.
    ///
    /// The distilled map and per-atom atlas are initializer machinery only. A
    /// row whose cheap start is not certifiable tries the atlas's colder start;
    /// either way, the final coordinates come from the joint residual solve and
    /// its explicit first-order convergence verdict.
    ///
    /// Magic by default: the atlas's worst-case bounds are auto-derived from the
    /// fit — `amplitude_bound[k]` is the largest fitted assignment mass `a[i,k]`
    /// the encode can produce for atom `k` (the encode recovers `t` from
    /// `x ≈ z·γ_k(t)` at amplitude `z = a[i,k]`), and `target_norm_bound` is the
    /// largest target row norm — so no caller supplies a knob. Per-row amplitudes
    /// are the fitted assignment masses for the same target the dictionary was fit
    /// against; an external corpus reuses the per-row masses the assignment
    /// produces for it upstream (passed in `amplitudes`, one column per atom).
    pub fn amortized_encode_target(
        &self,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView2<'_, f64>,
    ) -> Result<crate::encode::JointEncodeResult, String> {
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let n = targets.nrows();
        if targets.ncols() != p {
            return Err(format!(
                "SaeManifoldTerm::amortized_encode_target: targets have {} cols but output_dim is {p}",
                targets.ncols()
            ));
        }
        if amplitudes.dim() != (n, k_atoms) {
            return Err(format!(
                "SaeManifoldTerm::amortized_encode_target: amplitudes {:?} must be (n={n}, K={k_atoms})",
                amplitudes.dim()
            ));
        }

        // Magic-by-default offline bounds, auto-derived from the fit so no caller
        // supplies a knob. `target_norm_bound` is the largest target row L2 norm
        // (bounds `‖x‖` over the corpus); `amplitude_bound[k]` is the largest
        // fitted assignment mass for atom `k` (bounds `|z_k|`), with a strictly
        // positive floor so a near-inactive atom still certifies a finite radius.
        let mut target_norm_bound = 0.0_f64;
        for row in 0..n {
            let norm = targets.row(row).dot(&targets.row(row)).sqrt();
            if norm.is_finite() && norm > target_norm_bound {
                target_norm_bound = norm;
            }
        }
        let mut amplitude_bound = vec![0.0_f64; k_atoms];
        for atom_idx in 0..k_atoms {
            let mut bound = 0.0_f64;
            for row in 0..n {
                let z = amplitudes[[row, atom_idx]].abs();
                if z.is_finite() && z > bound {
                    bound = z;
                }
            }
            // A strictly positive amplitude floor keeps the offline Lipschitz
            // scaling finite for atoms with no active row in this corpus (those
            // rows encode to the chart center via the certificate anyway).
            amplitude_bound[atom_idx] = bound.max(1.0);
        }

        let atlas = crate::encode::EncodeAtlas::build(
            &self.atoms,
            &amplitude_bound,
            target_norm_bound,
            crate::encode::AtlasConfig::default(),
        )?;

        // F3 — certify against the TRUE encode objective whenever it departs from
        // the bare Euclidean, prior-free field the fast path assumes: either the
        // installed per-row metric WHITENS THE LIKELIHOOD (GLS reconstruction loss
        // `½ rᵀ M_n r`, `M_n = U_n U_nᵀ`), or a latent ARD / von-Mises coordinate
        // prior was fitted on `t` (`atom.ard_precisions`). In either case a bare
        // Euclidean encode certifies the root of a DIFFERENT problem, so route every
        // (row, atom) through the metric-and-prior-aware certified encode
        // (`certified_encode_row_with_objective`).
        //
        // Metric gate is `whitens_likelihood()`, NOT merely non-Euclidean: the
        // gauge-only `OutputFisher`/`OutputFisherDownstream` provenances leave the
        // data loss isotropic (whitening by them would be the #980 failure mode —
        // silently replacing the reconstruction loss with a Fisher pullback). Only
        // `WhitenedStructured` (estimated noise model) and `BehavioralFisher`
        // (GLS-in-nats, elected) actually price `½ rᵀ M_n r`. When active the
        // residual and SSE guard are whitened by the row factor `U_n`, and the
        // offline chart Lipschitz is scaled by the global bound
        // `max_n tr(M_n) ≥ max_n ‖M_n‖` (for PSD `M_n`, `‖M_n‖ = λ_max ≤ tr(M_n)`).
        //
        // The ARD precisions `α_a = exp(log_ard[k][a])` were stamped onto each atom
        // from the terminal rho at finalization (`canonicalize_charts_post_fit`), so
        // the encode adds the SAME coordinate prior gradient / Hessian / Lipschitz
        // the fit used. The distilled fast path is PRESERVED — the same
        // amortized-then-certified cascade as the Euclidean branch, but both tiers
        // certify under the objective (`*_with_objective`), so structured/prior fits
        // keep the one-mat-vec encode without a broad slow-path regression.
        // Euclidean-metric, prior-free fits (empty `log_ard`) skip this branch and
        // take the cascade below bit-for-bit unchanged.
        let metric = self.row_metric.as_ref().filter(|m| m.whitens_likelihood());
        let (metric_rank, metric_norm_bound) = match metric {
            Some(m) => {
                if m.p_out() != p || m.n_rows() != n {
                    return Err(format!(
                        "SaeManifoldTerm::amortized_encode_target: row_metric is ({} rows, \
                         p={}) but target is (n={n}, p={p})",
                        m.n_rows(),
                        m.p_out()
                    ));
                }
                let bound = m.row_traces().iter().copied().fold(0.0_f64, f64::max);
                (m.metric_rank(), bound)
            }
            None => (0usize, 1.0_f64),
        };
        let mut coords: Vec<Array2<f64>> = self
            .atoms
            .iter()
            .map(|a| Array2::<f64>::zeros((n, a.latent_dim)))
            .collect();
        let mut converged = vec![false; n];

        for row in 0..n {
            let u_row = metric.map(|m| {
                Array2::<f64>::from_shape_fn((p, metric_rank), |(i, k)| m.factor_entry(row, i, k))
            });
            let mut starts = Vec::with_capacity(k_atoms);
            for atom_idx in 0..k_atoms {
                let atom = &self.atoms[atom_idx];
                let prior_alpha = atom
                    .ard_precisions
                    .as_ref()
                    .filter(|pa| !pa.is_empty())
                    .and_then(|pa| pa.as_slice());
                let objective = crate::encode::EncodeObjective {
                    metric_factor: u_row.as_ref().map(|u| u.view()),
                    prior_alpha,
                    metric_norm_bound,
                };
                let amplitude = amplitudes[[row, atom_idx]];
                let (mut start, start_cert) = atlas.amortized_encode_row_with_objective(
                    atom,
                    atom_idx,
                    targets.row(row),
                    amplitude,
                    &objective,
                )?;
                if !start_cert.certified() {
                    let (cold, cold_cert) = atlas.certified_encode_row_with_objective(
                        atom,
                        atom_idx,
                        targets.row(row),
                        amplitude,
                        &objective,
                    )?;
                    // A valid per-atom certificate improves the initializer. If it
                    // is unavailable, retain the finite amortized chart start; the
                    // joint solver below, not this initializer, decides validity.
                    if cold_cert.certified() {
                        start = cold;
                    }
                }
                starts.push(start);
            }

            let (joint, row_converged) = crate::encode::joint_encode_refine_row(
                &self.atoms,
                &starts,
                targets.row(row),
                amplitudes.row(row),
                u_row.as_ref().map(|u| u.view()),
            )?;
            for atom_idx in 0..k_atoms {
                coords[atom_idx].row_mut(row).assign(&joint[atom_idx]);
            }
            converged[row] = row_converged;
        }
        Ok(crate::encode::JointEncodeResult::new(coords, converged))
    }

    /// #1026 — the fitted per-row assignment masses `a[i,k]` (the activation
    /// amplitudes `z_k` the amortized encode recovers `t` against), as an
    /// `n × K` matrix. These are the posterior assignment intensities `a_{ik}`
    /// that [`Self::try_fitted_with_rho`] multiplies into each atom's decoded row.
    pub fn fitted_assignment_amplitudes(&self) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let mut amplitudes = Array2::<f64>::zeros((n, k_atoms));
        for row in 0..n {
            let a = self.assignment.try_assignments_row(row)?;
            for atom_idx in 0..k_atoms {
                amplitudes[[row, atom_idx]] = a[atom_idx];
            }
        }
        Ok(amplitudes)
    }

    /// #1026 — encode the dictionary's own fit-time target with the amortized
    /// encoder, deriving the per-row amplitudes from the fitted assignment so the
    /// caller supplies neither bounds nor amplitudes (magic by default). The
    /// end-to-end "fit → distilled encoder → certificate-gated encode" path.
    pub fn amortized_encode_fitted(
        &self,
        targets: ArrayView2<'_, f64>,
    ) -> Result<crate::encode::JointEncodeResult, String> {
        let amplitudes = self.fitted_assignment_amplitudes()?;
        self.amortized_encode_target(targets, amplitudes.view())
    }

    /// #1154 — amortized-encoder consistency of the CURRENT dictionary against
    /// its own fit-time target. This is the co-training signal of the joint
    /// amortized-encoder + penalized quasi-Laplace loop (Design A): the amortized (one-mat-vec)
    /// encode is built from the *current* fitted decoder, run on `targets`, and
    /// scored on two principled axes —
    ///
    /// * `recon_consistency` (the bilinear part of the co-training loss): the
    ///   mean per-element squared gap between the **amortized** reconstruction
    ///   `Σ_k z_k · Φ_k(t̂_k) B_k` (decode the amortized coords) and the
    ///   **exact** fitted reconstruction `Σ_k z_k · Φ_k(t_k^*) B_k` the inner
    ///   solve converged to. A dictionary whose encode map is well-approximated
    ///   to first order by the per-chart IFT predictor scores near zero; a
    ///   dictionary the amortized encoder *cannot* invert faithfully (sharp
    ///   curvature, poorly-charted regions) scores high. Minimising this jointly
    ///   with penalized quasi-Laplace steers the fit toward dictionaries that admit a fast,
    ///   faithful amortized encode — the architectural co-adaptation #1154 adds.
    /// * `unconverged_fraction`: the share of rowwise joint shared-residual
    ///   solves that did not meet the first-order stationarity tolerance.
    ///
    /// Shared-residual refinement keeps the reported reconstruction tied to the
    /// fitted multi-atom objective; the convergence fraction records rows that
    /// did not reach its first-order tolerance.
    pub fn amortized_encoder_consistency(
        &self,
        targets: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<AmortizedEncoderConsistency, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if targets.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::amortized_encoder_consistency: targets {:?} must be (n={n}, p={p})",
                targets.dim()
            ));
        }
        let amplitudes = self.fitted_assignment_amplitudes()?;
        let encodes = self.amortized_encode_target(targets, amplitudes.view())?;
        // The EXACT fitted reconstruction the inner solve converged to (pure
        // curved image, rho-keyed) is the supervision target for the amortized
        // reconstruction. Both are n×p ambient, so the comparison is layout-free.
        let exact_recon = self.try_fitted_for_rho(rho)?;

        // Build the amortized reconstruction Σ_k z_k · Φ_k(t̂_k) B_k by decoding
        // each atom's amortized coords through that atom's own basis evaluator.
        let mut amortized_recon = Array2::<f64>::zeros((n, p));
        for atom_idx in 0..k_atoms {
            let atom = &self.atoms[atom_idx];
            let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                format!("amortized_encoder_consistency: atom {atom_idx} has no basis evaluator")
            })?;
            let coord_block = &encodes.coords[atom_idx];
            // Decode the amortized coords: Φ_k(t̂) is (n × M_k); B_k is (M_k × p).
            let (phi, _jac) = evaluator.evaluate(coord_block.view())?;
            // Decode `Φ_k(t̂) · B_k` (n×M · M×p) through the faer GEMM; small
            // shapes fall back to `ndarray::dot` inside `fast_ab` (reduction
            // order may differ, acceptable per the crate convention).
            let decoded = fast_ab(&phi, &atom.decoder_coefficients); // (n × p)
            for row in 0..n {
                let z = amplitudes[[row, atom_idx]];
                if z == 0.0 {
                    continue;
                }
                for col in 0..p {
                    amortized_recon[[row, col]] += z * decoded[[row, col]];
                }
            }
        }

        let mut sse = 0.0_f64;
        for row in 0..n {
            for col in 0..p {
                let gap = amortized_recon[[row, col]] - exact_recon[[row, col]];
                sse += gap * gap;
            }
        }
        let denom = (n.max(1) * p.max(1)) as f64;
        let recon_consistency = sse / denom;
        let total_encodes = n.max(1) as f64;
        let unconverged_fraction = encodes.unconverged_count as f64 / total_encodes;

        Ok(AmortizedEncoderConsistency {
            recon_consistency,
            unconverged_fraction,
            n_unconverged: encodes.unconverged_count,
            n_encodes: n,
        })
    }

    /// #1154 — the co-trained penalized quasi-Laplace criterion: the exact penalized quasi-Laplace criterion at `rho`
    /// PLUS the amortized-encoder consistency penalty, so the outer optimizer
    /// co-adapts the dictionary + smoothing parameters λ toward a dictionary the
    /// fast initializer and joint refinement can faithfully invert.
    ///
    /// This is Design A of #1154. The inner solve still converges the `(t, β)`
    /// system to stationarity at the engine's current ρ (so the implicit-function
    /// penalized quasi-Laplace λ-gradient `dβ̂/dλ = −(H+S_λ)⁻¹(dS_λ/dλ)β̂` stays exact — the encoder
    /// only warm-starts/co-adapts, it never replaces the stationary point). The
    /// added term
    ///
    /// ```text
    ///   J_cotrain(ρ) = penalized_quasi_laplace(ρ) + w · ‖x̂_amortized − x̂_exact‖²/(n·p)
    ///                            +  w_conv · unconverged_fraction
    /// ```
    ///
    /// folds the post-fit amortized-encode quality into the ranked objective. The
    /// weights are auto-scaled to the penalized quasi-Laplace criterion magnitude (magic by default:
    /// no caller knob) so the consistency term is a meaningful but non-dominant
    /// fraction of the objective regardless of problem scale.
    pub fn penalized_quasi_laplace_criterion_cotrained(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, AmortizedEncoderConsistency), String> {
        // #1154: always attempt the amortized warm-start first inside
        // `penalized_quasi_laplace_criterion_cotrained` (the encode/warm path for the cotrained
        // objective). Good warm-starts from the running dictionary land the
        // inner solve closer to the stationary point used for the fold.
        // Advisory only (0 or err falls back to cold); telemetry recorded by
        // outer objective callers when present.
        self.warm_start_latents_from_amortized_encoder(target, rho)
            .unwrap_or(0);
        let (penalized_quasi_laplace, loss) = self
            .penalized_quasi_laplace_criterion_with_refine_policy(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                true,
            )?;
        let consistency = self.amortized_encoder_consistency(target, rho)?;
        // Auto-scale the co-training weights to the penalized quasi-Laplace magnitude so the
        // consistency penalty is a bounded, scale-free fraction of the objective
        // (magic by default: no caller knob). `criterion_scale` floors at 1 so a
        // near-zero criterion still admits a meaningful consistency contribution.
        let cotrained = Self::fold_cotrain_consistency(penalized_quasi_laplace, &consistency);
        Ok((cotrained, loss, consistency))
    }

    /// #1154 — the single source of the co-training fold arithmetic: add the
    /// auto-scaled amortized-encoder consistency penalty to an already-computed
    /// penalized quasi-Laplace criterion at the converged dictionary. Both the public
    /// [`Self::penalized_quasi_laplace_criterion_cotrained`] entry point and the outer-loop value /
    /// gradient lanes (`SaeManifoldOuterObjective::fold_cotrain_consistency`)
    /// route through THIS function, so the folded objective cannot drift between
    /// the criterion and the cascade-ranked cost (the objective↔gradient desync
    /// bug class). The weights are auto-scaled to the penalized quasi-Laplace magnitude
    /// (`max(|penalized_quasi_laplace|,
    /// 1)`) so the penalty is a bounded, scale-free fraction of the objective
    /// regardless of problem scale; the fold carries no analytic gradient (under
    /// Design A the penalized quasi-Laplace λ-gradient stays the exact implicit-function path).
    #[must_use]
    pub fn fold_cotrain_consistency(
        penalized_quasi_laplace_cost: f64,
        consistency: &AmortizedEncoderConsistency,
    ) -> f64 {
        let criterion_scale = penalized_quasi_laplace_cost.abs().max(1.0);
        penalized_quasi_laplace_cost
            + COTRAIN_RECON_WEIGHT * criterion_scale * consistency.recon_consistency
            + COTRAIN_CONVERGENCE_WEIGHT * criterion_scale * consistency.unconverged_fraction
    }

    /// #1154 item 2 — warm-start the inner latent coordinates from the amortized
    /// encoder (Design A). Builds per-chart starts from the current dictionary,
    /// refines all atoms against the shared row residual, and overwrites stored
    /// latent coordinates only on rows whose joint solve reaches first-order
    /// stationarity. Unconverged rows are left at their current coordinates, so the
    /// warm-start can only help. The subsequent inner Newton refines from this seed to
    /// the SAME stationary point (the warm-start changes only the basin entry,
    /// not the root), so the penalized quasi-Laplace λ-gradient stays exactly the implicit-function
    /// path and the criterion is unchanged at convergence — the amortized encoder
    /// only accelerates/co-adapts the inner solve, it never replaces the
    /// stationary point.
    ///
    /// Returns the number of rows actually warm-started — rows whose joint solve
    /// converged and cleared the per-row acceptance guard — for
    /// instrumentation / tests. A first-build dictionary with no usable charts, or
    /// an already-converged one whose seeds are all rejected, simply warm-starts
    /// nothing and returns 0 (the inner state is left byte-for-byte unchanged).
    pub fn warm_start_latents_from_amortized_encoder(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<usize, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        if n == 0 || k_atoms == 0 {
            return Ok(0);
        }
        let amplitudes = self.fitted_assignment_amplitudes()?;
        let encodes = self.amortized_encode_target(target, amplitudes.view())?;
        let p = self.output_dim();
        // Per-row reconstruction squared error BEFORE any seed is applied. The
        // amortized encoder is an approximate inverse: on a not-yet-converged
        // dictionary its converged rows accelerate the inner solve, but against an
        // ALREADY-converged (per-row optimal) dictionary a seed can only move a
        // coord off its optimum. Adopting such a seed would corrupt a good inner
        // state — precisely the regression the warm-start contract forbids ("changes
        // basin entry, not root"). So each converged seed is applied under a per-row
        // acceptance guard: a row keeps the encoder coord only if it does not worsen
        // that row's reconstruction. This makes the warm-start a monotone operation
        // on the reconstruction objective (post-warm per-row SSE ≤ pre-warm), so
        // recovery can never regress, while still adopting every seed that helps.
        let row_sse = |fitted: &Array2<f64>, row: usize| -> f64 {
            let mut acc = 0.0_f64;
            for col in 0..p {
                let r = target[[row, col]] - fitted[[row, col]];
                acc += r * r;
            }
            acc
        };
        let pre_fitted = self.try_fitted_for_rho(rho)?;
        let pre_sse: Vec<f64> = (0..n).map(|row| row_sse(&pre_fitted, row)).collect();

        // Snapshot the pre-warm coords so a rejected row can be reverted exactly.
        let orig_coords: Vec<Array2<f64>> = (0..k_atoms)
            .map(|atom_idx| self.assignment.coords[atom_idx].as_matrix())
            .collect();
        // Tentatively apply every converged joint solution, then accept/reject per row.
        let mut candidate_rows: Vec<bool> = vec![false; n];
        for atom_idx in 0..k_atoms {
            let d = self.atoms[atom_idx].latent_dim;
            if d == 0 {
                continue;
            }
            let coord_block = &encodes.coords[atom_idx];
            let mut coords = orig_coords[atom_idx].clone();
            if coords.dim() != (n, d) {
                return Err(format!(
                    "warm_start_latents_from_amortized_encoder: atom {atom_idx} coords {:?} != (n={n}, d={d})",
                    coords.dim()
                ));
            }
            for row in 0..n {
                if !encodes.converged[row] {
                    continue;
                }
                for axis in 0..d {
                    coords[[row, axis]] = coord_block[[row, axis]];
                }
                candidate_rows[row] = true;
            }
            // `as_matrix` lays coords out row-major (`[[row, axis]]`), exactly the
            // `values[row*d + axis]` order `set_flat` expects, so a plain
            // row-major iterator reconstructs the flat vector.
            let flat = Array1::from_iter(coords.iter().copied());
            self.assignment.coords[atom_idx].set_flat(flat.view());
        }
        // The basis caches must follow the freshly-seeded coords so the fit (and the
        // acceptance check just below) evaluates Φ at the warm-started t̂.
        self.refresh_basis_from_current_coords()?;

        // Reject the seed on any row that got worse, reverting ALL of that row's atom
        // coords to the snapshot. Reconstruction couples atoms within a row, so the
        // accept/reject decision is per row, not per (row, atom).
        let post_fitted = self.try_fitted_for_rho(rho)?;
        let accepted: Vec<bool> = (0..n)
            .map(|row| candidate_rows[row] && row_sse(&post_fitted, row) <= pre_sse[row] + 1.0e-12)
            .collect();
        let mut reverted_any = false;
        for atom_idx in 0..k_atoms {
            let d = self.atoms[atom_idx].latent_dim;
            if d == 0 {
                continue;
            }
            let mut coords = self.assignment.coords[atom_idx].as_matrix();
            let mut changed = false;
            for row in 0..n {
                if candidate_rows[row] && !accepted[row] {
                    for axis in 0..d {
                        coords[[row, axis]] = orig_coords[atom_idx][[row, axis]];
                    }
                    changed = true;
                }
            }
            if changed {
                let flat = Array1::from_iter(coords.iter().copied());
                self.assignment.coords[atom_idx].set_flat(flat.view());
                reverted_any = true;
            }
        }
        if reverted_any {
            self.refresh_basis_from_current_coords()?;
        }

        let warm_started = accepted.iter().filter(|&&a| a).count();
        Ok(warm_started)
    }

    pub fn loss(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<SaeManifoldLoss, String> {
        self.loss_scaled(target, rho, 1.0)
    }

    /// Penalized objective with a `penalty_scale` applied to the β-tier
    /// (decoder smoothness) penalty, mirroring
    /// [`Self::assemble_arrow_schur_scaled`]. The streaming line search sums
    /// per-chunk `loss_scaled(..., n_chunk / N)` so that the global smoothness
    /// penalty is counted exactly once across a pass while the per-row data,
    /// assignment-prior, and ARD terms sum naturally. `penalty_scale == 1.0`
    /// recovers the full-batch objective.
    pub fn loss_scaled(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        penalty_scale: f64,
    ) -> Result<SaeManifoldLoss, String> {
        self.assignment.validate_rho_domain(rho)?;
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::loss_scaled: penalty_scale must be finite and positive; got {penalty_scale}"
            ));
        }
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::loss: Z must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        // The likelihood whitens through the RowMetric **only** when the metric
        // is a genuinely estimated noise model (`metric.whitens_likelihood()`,
        // i.e. `WhitenedStructured` — the #974 residual-covariance seam). For
        // Euclidean (default `None`) and for the OutputFisher *gauge* metric the
        // reconstruction data-fit stays the isotropic `0.5 * Σ r²`: a gauge /
        // output-Fisher inner product must NOT silently replace the
        // reconstruction loss with a Fisher pullback (#980). It only drives the
        // gauge (see `analytic_penalties::corrected_isometry_penalty`). The
        // producer of `WhitenedStructured` is
        // `inference::residual_factor::StructuredResidualModel::row_metric`; the
        // SAME metric whitens the assembled gradient/Hessian in
        // `assemble_arrow_schur` (the single #974 seam), so this value and that
        // gradient cannot desync. Without a whitening metric this path is
        // bit-for-bit the historical isotropic data-fit.
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        // #991 design honesty weights: the reconstruction channel of row `i`
        // is weighted by `w_i` (mean-1 HT inclusion correction). The assembly
        // applies the same `w_i` via a `√w_i` scaling of the row residual /
        // Jacobian / β load at its single seam, so this value and that
        // gradient/Hessian carry the identical per-row factor. `None` ⇒ the
        // historical unweighted sum, bit-for-bit.
        let row_loss_w = self.row_loss_weights.as_deref();
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        // #Bug2: reconstruct over the SAME per-row active support the compact
        // Arrow-Schur assembly used, so this scalar objective value and the
        // assembled Newton gradient/Hessian are derivatives of ONE truncated
        // reconstruction. When a compact layout is engaged (softmax top-k /
        // large-K ordered Beta--Bernoulli), the assembly forms `fitted` from the row's active atoms
        // only; summing all K here would make `loss_scaled` a DIFFERENT objective
        // than the Newton step descends whenever dropped atoms carry mass. `None`
        // (dense layout) ⇒ the historical full-K sum, bit-for-bit. Guarded on the
        // row count so a stale/foreign layout is never mis-indexed.
        let recon_layout = self
            .last_row_layout
            .as_ref()
            .filter(|l| l.active_atoms.len() == n);
        // #1017: the data-fit is the dominant per-line-search-trial cost (it
        // re-runs every Armijo halving × every inner Newton iteration × every
        // outer ρ evaluation). The old path materialised the whole `n × p`
        // fitted matrix (`try_fitted_for_rho`) and then walked it AGAIN to form
        // the residual sum — two sequential `n·p` passes plus an `n·p`
        // allocation per trial. Fuse the reconstruction and the residual reduce
        // into ONE row-parallel pass that never materialises the fitted matrix:
        // each row decodes its atoms into per-worker scratch, differences
        // against the target, and contributes its scalar `0.5·w·‖r‖²` to a
        // deterministic length-only pairwise tree (bit-identical across thread
        // count AND nesting — see the fold below). Per-block scratch keeps the
        // only allocations one `g_buf`/`fitted_row`/`assign_buf` triple per base
        // block rather than per row, and each base block pins its faer GEMMs to
        // `Par::Seq` (the topology race owns the outer pool) to avoid nested
        // oversubscription.
        let parallel = n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        let row_data_fit = |row: usize,
                            g_buf: &mut [f64],
                            fitted_row: &mut [f64],
                            assign_buf: &mut [f64]|
         -> Result<f64, String> {
            // #1557 — fill the per-atom assignment row into reused per-worker
            // scratch via the `_into` twin instead of heap-allocating a fresh
            // `Array1` per row per loss eval. Bit-identical to the allocating
            // `try_assignments_row` (same arithmetic, same order); this
            // loss reruns every Armijo halving × inner Newton iter × outer ρ
            // eval, so the per-row K-sized allocation was a hot-path churn.
            self.assignment.try_assignments_row_into(row, assign_buf)?;
            let a = &*assign_buf;
            for slot in fitted_row.iter_mut() {
                *slot = 0.0;
            }
            match recon_layout {
                // Compact active support: reconstruct only the row's active atoms,
                // exactly as the compact assembly forms `fitted`.
                Some(layout) => {
                    for &atom_idx in &layout.active_atoms[row] {
                        self.atoms[atom_idx].fill_decoded_row(row, g_buf);
                        let a_k = a[atom_idx];
                        for out_col in 0..p {
                            fitted_row[out_col] += a_k * g_buf[out_col];
                        }
                    }
                }
                None => {
                    for atom_idx in 0..k_atoms {
                        self.atoms[atom_idx].fill_decoded_row(row, g_buf);
                        let a_k = a[atom_idx];
                        for out_col in 0..p {
                            fitted_row[out_col] += a_k * g_buf[out_col];
                        }
                    }
                }
            }
            for out_col in 0..p {
                fitted_row[out_col] = target[[row, out_col]] - fitted_row[out_col];
            }
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            let mut acc = 0.0_f64;
            match self.row_metric.as_ref() {
                Some(metric) if whitens => {
                    let resid = ArrayView1::from(&fitted_row[..p]);
                    for w in metric.whiten_residual_row(row, resid) {
                        acc += 0.5 * w_row * w * w;
                    }
                }
                _ => {
                    for &r in fitted_row[..p].iter() {
                        acc += 0.5 * w_row * r * r;
                    }
                }
            }
            Ok(acc)
        };
        // #2228 reduction doctrine: the parallel and sequential branches MUST be
        // bit-identical, so both reduce the per-row scalars through the SAME
        // length-only pairwise tree (`pairwise_sum` over base blocks of
        // `BASE_CHUNK`, combined by `left_split`). The parallel branch drives
        // that tree with `par_deterministic_try_block_fold` (each base block owns
        // its own scratch and folds its rows through `pairwise_sum`); the
        // sequential branch materialises the same per-row scalars and calls
        // `pairwise_sum` directly. Both are pure functions of the ordered row
        // values, so a nested K=1 fit (where `current_thread_index()` is `None`
        // and the parallel branch is taken) matches the top-level serial sweep to
        // the last bit. A per-chunk running sum would associate differently from
        // the whole-slice fold and silently perturb the objective (#2228).
        let data_fit = if parallel {
            use gam_linalg::pairwise_reduce::{pairwise_sum, par_deterministic_try_block_fold};
            par_deterministic_try_block_fold(
                n,
                |range: core::ops::Range<usize>| -> Result<f64, String> {
                    // #1557 — pin any faer GEMM reached from this base block to
                    // `Par::Seq` (no nested Rayon re-fan); the per-row reductions
                    // are tiny, so the result is bit-identical.
                    with_nested_parallel(|| {
                        let mut g_buf = vec![0.0_f64; p];
                        let mut fitted_row = vec![0.0_f64; p];
                        let mut assign_buf = vec![0.0_f64; k_atoms];
                        let mut block = Vec::with_capacity(range.len());
                        for row in range {
                            block.push(row_data_fit(
                                row,
                                &mut g_buf,
                                &mut fitted_row,
                                &mut assign_buf,
                            )?);
                        }
                        Ok(pairwise_sum(&block))
                    })
                },
                |a, b| Ok(a + b),
            )?
            .unwrap_or(0.0)
        } else {
            use gam_linalg::pairwise_reduce::pairwise_sum;
            let mut g_buf = vec![0.0_f64; p];
            let mut fitted_row = vec![0.0_f64; p];
            let mut assign_buf = vec![0.0_f64; k_atoms];
            let mut vals = Vec::with_capacity(n);
            for row in 0..n {
                vals.push(row_data_fit(
                    row,
                    &mut g_buf,
                    &mut fitted_row,
                    &mut assign_buf,
                )?);
            }
            pairwise_sum(&vals)
        };
        let assignment_sparsity = crate::assignment::assignment_prior_value_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?;
        let smoothness = penalty_scale * self.decoder_smoothness_value(&rho.lambda_smooth_vec()?);
        let ard = self.ard_value(rho)?;
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
            criterion_gauge_deflated_directions: 0,
        })
    }

    /// Reconstruction data-fit `0.5·Σ_i w_i·‖whiten(Z_i − R_i)‖²` for an EXPLICIT
    /// reconstruction matrix `R` (e.g. the hard top-k–projected `fitted`), using
    /// the SAME per-row metric and design-honesty weights as [`Self::loss_scaled`]
    /// (the soft-assignment data-fit). The only difference is the residual source:
    /// `loss_scaled` decodes the soft assignments on the fly, this consumes a
    /// reconstruction the caller already assembled (so the projected loss and the
    /// returned projected `fitted` describe one and the same model). The penalty
    /// terms (`assignment_sparsity`/`smoothness`/`ard`) are decoder/ρ properties
    /// the top-k gate does not change, so the caller keeps them from the soft
    /// `loss_scaled` and only swaps this data-fit in — see #1232.
    pub fn data_fit_for_reconstruction(
        &self,
        target: ArrayView2<'_, f64>,
        reconstruction: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::data_fit_for_reconstruction: Z must be ({n}, {p}); got {:?}",
                target.dim()
            ));
        }
        if reconstruction.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::data_fit_for_reconstruction: reconstruction must be ({n}, {p}); got {:?}",
                reconstruction.dim()
            ));
        }
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let row_loss_w = self.row_loss_weights.as_deref();
        let mut resid = vec![0.0_f64; p];
        let mut total = 0.0_f64;
        for row in 0..n {
            for out_col in 0..p {
                resid[out_col] = target[[row, out_col]] - reconstruction[[row, out_col]];
            }
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            match self.row_metric.as_ref() {
                Some(metric) if whitens => {
                    let r = ArrayView1::from(&resid[..p]);
                    for w in metric.whiten_residual_row(row, r) {
                        total += 0.5 * w_row * w * w;
                    }
                }
                _ => {
                    for &r in resid[..p].iter() {
                        total += 0.5 * w_row * r * r;
                    }
                }
            }
        }
        Ok(total)
    }

    pub fn analytic_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
    ) -> Result<f64, ArrowSchurError> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "SaeManifoldTerm::analytic_penalty_value_total: penalty_scale must be finite \
                     and positive; got {penalty_scale}"
                ),
            });
        }
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        registry
            .validate_rho(rho_global.view())
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let layout = registry.rho_layout();
        let beta = self.flatten_beta();
        let mut value = 0.0_f64;
        for (penalty, (rho_slice, tier, name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            // Skip the registry `ARDPenalty` here for the same reason it is
            // skipped in `add_sae_analytic_penalty_contributions`: the coordinate
            // ARD energy is already counted by `loss.ard` (the von-Mises
            // `ard_value`), and the registry penalty's legacy Gaussian `½λt²` is
            // period-discontinuous. Including it would double-count the energy and
            // make this line-search objective jump across the branch cut while the
            // assembled gradient (von-Mises only, after the assembly fix) stays
            // continuous — i.e. a near-zero step would change the objective by a
            // finite amount and Armijo would wrongly reject it.
            if matches!(penalty, AnalyticPenaltyKind::Ard(_)) {
                continue;
            }
            match tier {
                PenaltyTier::Psi => {
                    if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
                        for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                            value += penalty_scale
                                * per_atom.value(beta.slice(s![start..end]), rho_local);
                        }
                    } else {
                        if !sae_penalty_is_row_block_supported(penalty) {
                            return Err(ArrowSchurError::SchurFactorFailed {
                                reason: format!(
                                    "validate_analytic_penalty_registry should have refused \
                                     non-row-block Psi-tier penalty {:?} (registry layout name \
                                     {name:?})",
                                    penalty.name()
                                ),
                            });
                        }
                        for atom_idx in 0..self.k_atoms() {
                            let coord = &self.assignment.coords[atom_idx];
                            if let AnalyticPenaltyKind::Isometry(iso) = penalty {
                                let corrected_kind =
                                    self.corrected_isometry_penalty(iso, atom_idx, coord)?;
                                value += corrected_kind.value(coord.as_flat().view(), rho_local);
                            } else if sae_coord_penalty_is_origin_anchored_magnitude(penalty) {
                                // Origin-anchored magnitude shrinkage (SCAD/MCP) is
                                // restricted to the Euclidean axes; periodic axes have
                                // no chart origin and would make this energy
                                // period-discontinuous (issue #795). This must mirror
                                // the gradient/curvature assembly in
                                // `add_sae_coord_penalty` exactly.
                                match sae_coord_penalty_euclidean_restriction(coord) {
                                    Some((_axes, compacted)) => {
                                        value += penalty.value(compacted.view(), rho_local);
                                    }
                                    None => {
                                        value += penalty.value(coord.as_flat().view(), rho_local);
                                    }
                                }
                            } else {
                                value += penalty.value(coord.as_flat().view(), rho_local);
                            }
                        }
                    }
                }
                PenaltyTier::Beta => {
                    if let AnalyticPenaltyKind::DecoderIncoherence(base) = penalty {
                        if let Some(per_fit) = self.live_decoder_incoherence_penalty(base) {
                            value += penalty_scale * per_fit.value(beta.view(), rho_local);
                        }
                    } else if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
                        for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                            if start < end {
                                value += penalty_scale * per_atom.value(beta.view(), rho_local);
                            }
                        }
                    } else {
                        value += penalty_scale * penalty.value(beta.view(), rho_local);
                    }
                }
                PenaltyTier::Rho => {}
            }
        }
        Ok(value)
    }

    /// Energy of the decoder-block analytic penalties that have no native
    /// `SaeManifoldLoss` counterpart, evaluated at the current decoder `β` and
    /// the converged SAE state. These act on the per-atom decoder coefficient
    /// matrices: cross-atom decoder incoherence (#671), mechanism
    /// (feature-group) sparsity, and nuclear-norm embedding rank (#672). Each
    /// is injected with its live per-atom shape / co-activation before its
    /// value is taken, mirroring the assemble path.
    ///
    /// This is deliberately narrower than [`Self::analytic_penalty_value_total`]:
    /// it excludes the Psi-tier coordinate / assignment penalties (ARD,
    /// Isometry, ScadMcp, BlockOrthogonality, ordered Beta--Bernoulli/softmax assignment sparsity).
    /// The SAE already carries its own ARD (`loss.ard`) and assignment sparsity
    /// (`loss.assignment_sparsity`) energy, so adding the registry ARD /
    /// assignment value on top would double-count, and the gauge-only
    /// coordinate penalties are not part of the penalized deviance the
    /// penalized quasi-Laplace criterion scores. The decoder-block penalties, by contrast,
    /// are real penalized-energy terms with no `loss.*` representative: the
    /// inner solve minimizes them (they enter `gb`/`hbb`) but they were absent
    /// from the criterion scalar `v`. This restores that consistency so the
    /// ρ-sweep ranks the same objective the inner solve descends — the #671
    /// incoherence lever in particular now shapes model selection, not just the
    /// Newton step.
    ///
    /// NOTE: the coordinate-block penalties with no native `loss.*` twin
    /// (`ScadMcp`, `BlockOrthogonality`) carry the same residual inconsistency
    /// (scored in the line search via `penalized_objective_total`, absent from
    /// the penalized quasi-Laplace scalar). They are left out here because they share a registry
    /// dispatch with the always-on `Isometry` gauge, whose inclusion in the
    /// topology-comparison criterion is a separate design question (#673:
    /// topology evidence is gauge-conditional). Folding the coord-tier energy in
    /// is tracked apart from this #671 decoder fix.
    pub fn analytic_decoder_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<f64, ArrowSchurError> {
        // Resolve each penalty's rho slice exactly as `analytic_penalty_value_total`
        // does (registry-local rho at zeros), so a learnable decoder-penalty weight
        // is honoured rather than indexing into an empty view.
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        registry
            .validate_rho(rho_global.view())
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let layout = registry.rho_layout();
        let beta = self.flatten_beta();
        let mut value = 0.0_f64;
        for (penalty, (rho_slice, _tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            match penalty {
                AnalyticPenaltyKind::DecoderIncoherence(base) => {
                    if let Some(per_fit) = self.live_decoder_incoherence_penalty(base) {
                        value += per_fit.value(beta.view(), rho_local);
                    }
                }
                AnalyticPenaltyKind::MechanismSparsity(base) => {
                    for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                        if start < end {
                            value += per_atom.value(beta.view(), rho_local);
                        }
                    }
                }
                AnalyticPenaltyKind::NuclearNorm(base) => {
                    for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                        value += per_atom.value(beta.slice(s![start..end]), rho_local);
                    }
                }
                _ => {}
            }
        }
        Ok(value)
    }

    /// Energy of the COORDINATE-tier isometry penalty(ies) at the converged
    /// SAE state. This is the per-atom `½μ Σ_n ‖J_n^T W_n J_n / gbar − g_ref‖²`
    /// summed over atoms, evaluated through `corrected_isometry_penalty` so the
    /// live decoder/coordinate caches drive the value exactly as the assemble
    /// path does. It has no `SaeManifoldLoss` twin (the loss carries only
    /// data-fit / assignment / smoothness / ARD), so the penalized quasi-Laplace criterion
    /// must add it explicitly to score the same penalized objective the inner
    /// solve descends.
    pub fn isometry_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<f64, ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        registry
            .validate_rho(rho_global.view())
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let layout = registry.rho_layout();
        let mut value = 0.0_f64;
        for (penalty, (rho_slice, _tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            if let AnalyticPenaltyKind::Isometry(iso) = penalty {
                let rho_local = rho_global.slice(s![rho_slice.clone()]);
                for atom_idx in 0..self.k_atoms() {
                    let coord = &self.assignment.coords[atom_idx];
                    let corrected_kind = self.corrected_isometry_penalty(iso, atom_idx, coord)?;
                    value += corrected_kind.value(coord.as_flat().view(), rho_local);
                }
            }
        }
        Ok(value)
    }

    /// Whether assembling `registry` will scatter an isometry Gauss-Newton
    /// cross-block (`H_tβ`) into the per-row dense `htbeta` slabs.
    ///
    /// `add_sae_isometry_metric_gn_blocks` writes the coupled cross-block (and
    /// flips on `activate_dense_htbeta_supplement`) only when (a) the registry
    /// carries an `Isometry` penalty and (b) the atom's chart
    /// `preserves_isometry_cross_block_coherence` (flat charts — `Euclidean`,
    /// `Circle`, and flat products — keep the full `μ AᵀA` coupling; curved /
    /// boundary charts drop it to stay PSD). On the non-frames matrix-free path
    /// the data-fit cross-block is carried by the Kronecker row operator and the
    /// per-row `htbeta` slab is allocated at zero width (#1406/#1407 anti-leak),
    /// so this dense isometry supplement has nowhere to land unless the slab is
    /// widened to the full `beta_dim`. This predicate decides exactly that. The
    /// effective isometry weight `μ` is NOT consulted here: a near-zero `μ`
    /// short-circuits the per-row write, but the slab must still exist so the
    /// solver's `htbeta_dense_supplement` read is well-shaped.
    pub(crate) fn registry_writes_dense_isometry_cross_block(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> bool {
        registry
            .penalties
            .iter()
            .any(|p| matches!(p, AnalyticPenaltyKind::Isometry(_)))
            && self
                .assignment
                .coords
                .iter()
                .any(|coord| coord.manifold().preserves_isometry_cross_block_coherence())
    }

    /// Extra penalized-objective energy that has no native `SaeManifoldLoss`
    /// component but is part of the objective the inner Newton solve descends,
    /// and therefore of the penalized deviance the SAE penalized quasi-Laplace criterion
    /// must rank.
    ///
    /// ENVELOPE CONTRACT: the criterion value `v = loss.total() + extra +
    /// ½log|H| … − occam` is differentiated at the inner KKT root by the
    /// envelope theorem, which cancels the fitted-state response ONLY when the
    /// value's data+prior base is the SAME function whose gradient the KKT gate
    /// certified. That gradient (assembled in `assemble_arrow_schur`) carries
    /// every registry analytic penalty (Isometry, SCAD/MCP, BlockOrthogonality,
    /// DecoderIncoherence, MechanismSparsity, NuclearNorm), the decoder
    /// repulsion conditioner, and the Jeffreys separation barrier. The former
    /// composition here (decoder-block + isometry only) omitted ScadMcp,
    /// BlockOrthogonality, repulsion, and the barrier — a documented "residual
    /// inconsistency" that is inert at K=1 but a live envelope violation
    /// exactly in the K≥2 near-collinear / co-collapse regime where those
    /// terms carry energy. The base is now exactly
    /// `penalized_objective_total − loss.total()`: the full registry value
    /// (ARD skipped inside, `loss.ard` already carries it) plus repulsion plus
    /// the separation barrier. All of these have zero DIRECT ρ-derivative
    /// (their weights are not ρ coordinates), so the analytic outer-gradient
    /// channels are unchanged — this only restores value/gradient consistency.
    pub fn reml_extra_penalty_value_total(
        &self,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<f64, ArrowSchurError> {
        let registry_energy = match registry {
            Some(reg) => self.analytic_penalty_value_total(reg, 1.0)?,
            None => 0.0,
        };
        Ok(
            registry_energy
                + self.decoder_repulsion_value(1.0)
                + self.separation_barrier_value(1.0),
        )
    }

    pub fn penalized_objective_total(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
    ) -> Result<f64, String> {
        let mut total = self.loss_scaled(target, rho, penalty_scale)?.total();
        if let Some(analytic_registry) = registry {
            total += self
                .analytic_penalty_value_total(analytic_registry, penalty_scale)
                .map_err(|err| format!("SaeManifoldTerm::penalized_objective_total: {err}"))?;
        }
        // #1026 — decoder-repulsion value, on the SAME frozen gate the assembly
        // used, so the line search sees the term the Newton step optimizes. 0
        // unless two atoms are near-collinear (the no-op case).
        total += self.decoder_repulsion_value(penalty_scale);
        // #1026/#1522 — interior-point collapse-prevention barriers, on the SAME
        // decoders the assembly's gradient/curvature used, so the line search sees
        // exactly the term the inner Newton step optimises (no value/grad desync).
        total += self.separation_barrier_value(penalty_scale);
        Ok(total)
    }

    pub(crate) fn decoder_smoothness_value(&self, lambda_smooth: &[f64]) -> f64 {
        // Smoothness penalty value is `0.5·λ·Σ_oc B[:,oc]ᵀ S B[:,oc]`. Form the
        // `S·B` matrix product once per atom (O(M²·p)) and reduce against `B`
        // with a single O(M·p) Hadamard sum, instead of the previous
        // four-factor multiply-accumulate inside an `O(M²·p)` triple loop.
        // The quadratic form only sees the symmetric part of `S`, so reusing
        // the raw (un-symmetrised) `smooth_penalty` here is numerically
        // identical to the symmetrised assembly form.
        // Per-atom `S_k · B_k` products are independent across atoms, so they ride
        // the multi-GPU batched smoothness GEMM (uniform-shape groups tiled across
        // every device); `symmetrize = false` because the quadratic form only sees
        // the symmetric part of `S` regardless. Exact CPU fallback per atom.
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, false);
        let mut acc = 0.0;
        for (atom_idx, (atom, sb)) in self.atoms.iter().zip(sb_all.iter()).enumerate() {
            acc += 0.5 * lambda_smooth[atom_idx] * (&atom.decoder_coefficients * sb).sum();
        }
        acc
    }

    /// Per-atom decoder-smoothness values (#1556): entry `k` is
    /// `0.5·λ_smooth[k]·<B_k, S_k B_k>` (sum = [`Self::decoder_smoothness_value`]).
    /// This is the explicit `∂loss.smoothness/∂log λ_smooth[k]` gradient entry.
    pub(crate) fn decoder_smoothness_value_per_atom(&self, lambda_smooth: &[f64]) -> Vec<f64> {
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, false);
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        for (atom_idx, (atom, sb)) in self.atoms.iter().zip(sb_all.iter()).enumerate() {
            per_atom[atom_idx] =
                0.5 * lambda_smooth[atom_idx] * (&atom.decoder_coefficients * sb).sum();
        }
        per_atom
    }

    pub(crate) fn ard_value(&self, rho: &SaeManifoldRho) -> Result<f64, String> {
        self.assignment.validate_rho_domain(rho)?;
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atoms but term has {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let ard_precisions = rho.ard_precisions()?;
        let n = self.n_obs();
        // Design-honesty weights change the relative contribution of rows while
        // preserving total sample mass: `set_row_loss_weights` normalizes them to
        // mean one. The ARD energy therefore uses the per-row weights, while its
        // log-partition normalizer remains the observed row count exactly.
        let row_w = self.row_loss_weights.as_deref();
        let n_eff = n as f64;
        let mut acc = 0.0;
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            if rho.log_ard[atom_idx].is_empty() {
                continue;
            }
            if rho.log_ard[atom_idx].len() != d {
                return Err(format!(
                    "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                    rho.log_ard[atom_idx].len()
                ));
            }
            // Per-axis periodicity selects the smooth von-Mises energy on
            // wrapped (Circle) axes and the Gaussian on Euclidean axes.
            let periods = coord.effective_axis_periods();
            for axis in 0..d {
                let log_alpha = rho.log_ard[atom_idx][axis];
                let alpha = ard_precisions[atom_idx][axis];
                let period = periods[axis];
                let mut energy = 0.0;
                for row in 0..n {
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let v = coord.row(row)[axis];
                    energy += w_row * ArdAxisPrior::eval(alpha, v, period).value;
                }
                // Negative-log prior for precision alpha. The data-dependent
                // energy is the (Gaussian or von-Mises) coordinate prior; the
                // accompanying normaliser is the precision log-partition.
                //
                // Euclidean axes keep the Gaussian normaliser `-0.5 n log α`.
                // Periodic (von-Mises) axes use the EXACT von-Mises precision
                // log-partition `n[-η + log I0(η)]`, η = α/κ², κ = 2π/P, rather
                // than the Gaussian surrogate: the von-Mises partition function
                // is `2π I0(η)` (up to the κ Jacobian), so the per-observation
                // normaliser is `-η + log I0(η)` and is exact across the cut.
                match period {
                    None => {
                        acc += energy - 0.5 * n_eff * log_alpha;
                    }
                    Some(p) => {
                        // Evaluate η = αP²/(2π)² in log space: both η and the
                        // intermediate κ² can leave the float range even when
                        // the centered log-partition remains representable.
                        let log_eta = log_alpha + 2.0 * (p.ln() - std::f64::consts::TAU.ln());
                        let centered_log_i0 = bessel_i0_centered_terms_from_log_abs(log_eta).0;
                        // EXACT von-Mises precision log-partition. The partition over
                        // one period is `Z(α) = ∫₀ᴾ exp[-V] dt = P·e^{-η}·I0(η)` (sub
                        // `u=κt`, `dt = P/(2π) du`), so `log Z = log P − η + log I0(η)`.
                        // The `log P` period-Jacobian was previously dropped: harmless
                        // for unit-period axes (`P=1 ⇒ ln P = 0`, e.g. Circle{period:1}),
                        // but it under-counts non-unit periodic axes (sphere longitude,
                        // `P=2π`) by `n_eff·ln P` in the absolute prior evidence that
                        // cross-topology/K model comparison consumes. `ln P` is
                        // ρ-independent, so no inner gradient / FD channel is affected.
                        acc += energy + n_eff * (p.ln() + centered_log_i0);
                    }
                }
            }
        }
        Ok(acc)
    }

    pub(crate) fn ext_coord_matrix(&self) -> Array2<f64> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let flat = self.assignment.flatten_ext_coords();
        let mut out = Array2::<f64>::zeros((n, q));
        for row in 0..n {
            for col in 0..q {
                out[[row, col]] = flat[row * q + col];
            }
        }
        out
    }

    pub(crate) fn ext_coord_manifold(&self) -> LatentManifold {
        let mut parts = Vec::with_capacity(self.assignment.row_block_dim());
        for _ in 0..self.assignment.assignment_coord_dim() {
            parts.push(LatentManifold::Euclidean);
        }
        let mut any_constrained = false;
        for coord in &self.assignment.coords {
            if coord.manifold().is_euclidean() {
                for _ in 0..coord.latent_dim() {
                    parts.push(LatentManifold::Euclidean);
                }
            } else {
                any_constrained = true;
                parts.push(coord.manifold().clone());
            }
        }
        if any_constrained {
            LatentManifold::Product(parts)
        } else {
            LatentManifold::Euclidean
        }
    }

    pub(crate) fn apply_sae_riemannian_geometry(&self, sys: &mut ArrowSchurSystem) {
        let manifold = self.ext_coord_manifold();
        if manifold.is_euclidean() {
            return;
        }
        let ext = self.ext_coord_matrix();
        let latent =
            LatentCoordValues::from_matrix_with_manifold(ext.view(), LatentIdMode::None, manifold);
        sys.apply_riemannian_latent_geometry(&latent);
    }

    /// Build the compact-layout ext-coord product manifold and point for one row.
    ///
    /// TopK has no free gate coordinates, so a compact row is exactly the
    /// product of its selected atoms' coordinate manifolds in support order.
    /// On that support this is identical to slicing the full product manifold.
    pub(crate) fn compact_row_ext_manifold_and_point(
        &self,
        row: usize,
        layout: &SaeRowLayout,
    ) -> (LatentManifold, Array1<f64>) {
        let active = &layout.active_atoms[row];
        let q_active = layout.row_q_active(row);
        let mut parts: Vec<LatentManifold> = Vec::with_capacity(active.len());
        let mut point = Array1::<f64>::zeros(q_active);
        // Coordinate blocks: each active atom's coordinate manifold + point, at
        // the compact coord start the layout assigned it.
        for (j, &k) in active.iter().enumerate() {
            let coord = &self.assignment.coords[k];
            let d = coord.latent_dim();
            let coord_start = layout.coord_starts[row][j];
            let manifold_k = coord.manifold();
            // A `d`-dim coordinate whose manifold is a product (e.g. a torus =
            // Circle×Circle) already carries its `d` parts; a scalar manifold is
            // one part. Either way the manifold's ambient width must equal `d`,
            // matching the `d` compact columns at `coord_start`.
            parts.push(manifold_k.clone());
            let coord_point = coord.row(row);
            for axis in 0..d {
                point[coord_start + axis] = coord_point[axis];
            }
        }
        (LatentManifold::Product(parts), point)
    }

    /// Numerical rank of a symmetric matrix: the count of eigenvalues
    /// exceeding `tol · max_eig`, with `tol = 1e-9` (the conventional
    /// relative spectral cutoff used elsewhere in the codebase).
    ///
    /// Used to count the penalised dimension of each atom's `smooth_penalty`
    /// `S_k` so the penalized quasi-Laplace criterion's `−½·p·rank(S)·log λ_smooth` Occam term
    /// uses the *effective* penalty rank rather than the ambient basis size
    /// (a thin-plate / B-spline penalty has a non-trivial null space).
    pub(crate) fn symmetric_rank(s: &Array2<f64>) -> Result<usize, String> {
        if s.nrows() != s.ncols() {
            return Err(format!(
                "SaeManifoldTerm::symmetric_rank: matrix must be square, got {}x{}",
                s.nrows(),
                s.ncols()
            ));
        }
        let m = s.ncols();
        if m == 0 {
            return Ok(0);
        }
        // Symmetrize defensively through the shared ndarray helper. The SAE
        // rank cutoff is intentionally local to the SAE evidence contract; only
        // the symmetric cleanup is shared with the other construction modules.
        let mut sym = s.clone();
        gam_linalg::matrix::symmetrize_in_place(&mut sym);
        let (evals, _evecs) = sym
            .eigh(Side::Lower)
            .map_err(|e| format!("SaeManifoldTerm::symmetric_rank: eigh failed: {e}"))?;
        let max_eig = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v));
        if !(max_eig > 0.0) {
            return Ok(0);
        }
        let tol = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * max_eig;
        Ok(evals.iter().filter(|&&v| v > tol).count())
    }
}

// [#780 line-count gate] The quasi-Laplace criterion (`penalized_quasi_laplace_criterion*`)
// and the evidence-pricing machinery around it live in the sibling
// `construction_quasi_laplace.rs` as a second `impl SaeManifoldTerm` block,
// inlined here so it keeps the SAME module scope and private-field access.
include!("construction_quasi_laplace.rs");

// [#780 line-count gate] Per-row jet / reconstruction-channel assembly for the
// streaming-exact arrow log-det lives in a sibling file as a second
// `impl SaeManifoldTerm` block, inlined here so it keeps the SAME module scope
// and private-field access. Keeps this tracked file under the 10k limit.
include!("construction_row_jet_logdet_channels.rs");

// [#780 line-count gate] Massive-K decoder-smoothness effective-dof Hutchinson
// estimator (associated constants + the matrix-free per-atom trace) lives in a
// sibling file as another `impl SaeManifoldTerm` block, inlined here so it keeps
// the SAME module scope and private-field access. The two gated exact/estimator
// entry points above dispatch into it at `K >= MIN_ATOMS`.
include!("construction_smoothness_dof.rs");

// [#780 line-count gate] `term_from_padded_blocks_with_mode` (the padded-FFI
// term builder) was split into the sibling `construction_padded_blocks.rs`
// module (declared and re-exported from `mod.rs`), keeping this tracked file
// under the 10k limit. Callers still reach it bare through `use super::*`.

// [#780 line-count gate] `refresh_isometry_caches_from_atom` and
// `refresh_isometry_caches_from_term` were split into the sibling
// `construction_cache_refresh.rs` module (declared and re-exported from
// `mod.rs`), keeping this tracked file under the 10k limit. Callers still reach
// both functions bare through `use super::*`.

// [#780 line-count gate] The `#[cfg(test)]` modules below the production code
// are mechanically split into a sibling `*_tests` file and inlined via
// `include!` (the sanctioned cohesive-module decomposition — see build.rs
// file_stem_is_exempt_test_module). Keeps this tracked file under the 10k limit.
include!("construction_tests.rs");

/// Solve-invariant operands of `selected_inverse_row_blocks_or_solve` (#932
/// FRONT C): everything fixed across the per-row sweep of one
/// trace/adjoint pass — the deflated solver, the factor cache, the dense
/// `(H⁻¹)_ββ`, the Takahashi-vs-solve route flag, the shared zero β-RHS, and
/// the error-context prefix — bundled so each per-row call carries only the
/// row coordinates and the reusable scratch buffer.
pub(crate) struct SelectedInverseRowSolve<'a> {
    pub(crate) solver: &'a DeflatedArrowSolver<'a>,
    pub(crate) cache: &'a ArrowFactorCache,
    pub(crate) beta_inv: &'a Array2<f64>,
    pub(crate) fast_selected: bool,
    pub(crate) rhs_beta_zero: ArrayView1<'a, f64>,
    pub(crate) context: &'a str,
}
