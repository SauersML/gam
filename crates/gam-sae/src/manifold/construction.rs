use super::*;
use gam_math::jet_scalar::JetScalar;

// [#780] Softmax-entropy Gershgorin majorizer leaf helpers live in a sibling
// cohesive module, inlined here so they share this module scope.
include!("softmax_entropy_majorizer.rs");

/// Typed error from the SAE outer-gradient analytic assembly path (#1436).
///
/// The `eval()` analytic fallback (#1273/#1440: the plain undeflated analytic
/// outer gradient, NOT a finite difference) must fire ONLY for the genuine
/// conditioning/identifiability failure modes it was designed for — a
/// near-singular-but-valid joint Hessian or a gauge-degenerate direction.
/// Shape/indexing bugs, non-finite intermediates, and violated internal
/// invariants are [`OuterGradientError::InternalInvariant`] and MUST propagate
/// as hard errors so regressions surface instead of being silently masked by a
/// degraded descent direction.
#[derive(Clone, Debug)]
pub(crate) enum OuterGradientError {
    /// Expected: near-singular or ill-conditioned joint Hessian at a feasible ρ
    /// (the genuine #1273 flat-valley case). Eligible for the FD fallback.
    IllConditioned { reason: String },
    /// Expected: a non-identifiable / gauge-degenerate direction at this ρ.
    /// Eligible for the FD fallback.
    NonIdentifiable { reason: String },
    /// Unexpected: shape/dimension mismatch, non-finite intermediate, or a
    /// violated internal invariant. MUST propagate — never fall back to FD.
    InternalInvariant { reason: String },
}

impl OuterGradientError {
    /// Whether this error class is recoverable by the #1273/#1440 analytic
    /// plain-solver fallback (i.e. it represents a legitimate
    /// conditioning/identifiability failure, not a programming/invariant defect).
    pub(crate) fn is_conditioning_recoverable(&self) -> bool {
        matches!(
            self,
            Self::IllConditioned { .. } | Self::NonIdentifiable { .. }
        )
    }

    /// Construct an [`OuterGradientError::InternalInvariant`] from any error
    /// displayable — the default classification for unexpected assembly failures
    /// (shape mismatches, non-finite intermediates, violated invariants).
    pub(crate) fn internal<E: std::fmt::Display>(err: E) -> Self {
        Self::InternalInvariant {
            reason: err.to_string(),
        }
    }

    /// #1451 — classify a `String` error surfaced by the deflation linear-algebra
    /// path (`apply_cached_arrow_hessian`, `DeflatedArrowSolver::from_orthonormal_gauges`)
    /// into the correct [`OuterGradientError`] class.
    ///
    /// A genuine rank-deficiency / near-singularity failure (a back-solve or
    /// Cholesky/Woodbury factor that tripped on a finite, correctly-shaped input)
    /// is a legitimate #1273 conditioning failure and keeps `conditioning_err`
    /// (`IllConditioned`), so it stays recoverable by the analytic fallback. A
    /// shape/dimension mismatch or a non-finite intermediate is an
    /// internal-invariant defect and MUST propagate ([`Self::internal`]) instead
    /// of being masked as a plausible-but-wrong descent direction — exactly the
    /// #1436 contract.
    ///
    /// The two solver helpers return `String` (not a typed error), so the
    /// distinction is drawn from the stable markers those helpers emit for their
    /// shape/non-finite guards (`vector shapes`, `gauge length`, `must be finite`,
    /// `non-finite`). Everything else — including the `cholesky`/back-solve
    /// near-singular failures — is treated as a genuine conditioning trip.
    pub(crate) fn classify_arrow_solver_error(message: &str, conditioning_err: Self) -> Self {
        let lower = message.to_ascii_lowercase();
        let is_internal = lower.contains("vector shapes")
            || lower.contains("gauge length")
            || lower.contains("solution length")
            || lower.contains("!= cache")
            || lower.contains("must be finite")
            || lower.contains("non-finite")
            || lower.contains("not finite")
            || lower.contains("nan")
            || lower.contains("inf");
        if is_internal {
            Self::internal(message)
        } else {
            conditioning_err
        }
    }

    /// The exact gate the gradient lane (`SaeManifoldOuterObjective::eval`) uses
    /// to decide whether to descend with the #1273/#1440 analytic plain-solver
    /// fallback instead of propagating the error as a hard failure.
    ///
    /// The fallback is admissible ONLY when BOTH hold:
    /// * the REML cost at this rho is finite (a genuinely feasible point -- the
    ///   plain analytic solver supplies a descent direction for a value the
    ///   analytic path already produced), and
    /// * the error is a legitimate conditioning/identifiability failure
    ///   ([`Self::is_conditioning_recoverable`]) -- the genuine #1273 flat-valley
    ///   case.
    ///
    /// A non-finite cost or an [`OuterGradientError::InternalInvariant`] must
    /// propagate: masking a shape/indexing bug, a non-finite intermediate, or a
    /// violated invariant behind a plausible-but-wrong step is exactly the
    /// regression #1436 closes. Centralising the decision here (rather than
    /// inlining the boolean at the call site) makes the `cost x error-class`
    /// contract a single, directly unit-testable predicate.
    pub(crate) fn admits_plain_solver_fallback(&self, cost: f64) -> bool {
        cost.is_finite() && self.is_conditioning_recoverable()
    }
}

impl std::fmt::Display for OuterGradientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllConditioned { reason } => write!(f, "ill-conditioned: {reason}"),
            Self::NonIdentifiable { reason } => write!(f, "non-identifiable: {reason}"),
            Self::InternalInvariant { reason } => write!(f, "internal invariant: {reason}"),
        }
    }
}

impl From<OuterGradientError> for String {
    fn from(e: OuterGradientError) -> String {
        e.to_string()
    }
}

/// Active-set layout override for [`SaeManifoldTerm::assemble_arrow_schur_inner`].
///
/// `None` is the production path: the layout is derived from the assignment mode
/// and `sparse_active_plan`. `Some(layout_opt)` pins a specific layout — dense
/// (`Some(None)`) or a chosen compact `SaeRowLayout` (`Some(Some(..))`) — so the
/// compact-vs-dense Riemannian-geometry equality regression can drive both code
/// paths on identical data without depending on the host/device memory budget
/// that gates the compact path in production.
pub(crate) type ForcedRowLayout = Option<Option<SaeRowLayout>>;

/// #1154 — base co-training weight for the amortized-encoder reconstruction
/// consistency penalty, as a fraction of the REML criterion magnitude. The
/// effective weight is `COTRAIN_RECON_WEIGHT · max(|REML|, 1)`, so the penalty
/// is a bounded, scale-free share of the objective and needs no caller knob.
pub(crate) const COTRAIN_RECON_WEIGHT: f64 = 0.1;

/// #1154 — base co-training weight for the encoder's certifiable-coverage
/// penalty (the fraction of (row, atom) encodes the Kantorovich certificate
/// rejected). Scaled like [`COTRAIN_RECON_WEIGHT`].
pub(crate) const COTRAIN_CERT_WEIGHT: f64 = 0.05;

/// #1154 — amortized-encoder consistency of a fitted dictionary against its own
/// fit-time target. The co-training signal of the joint amortized-encoder +
/// REML loop: how faithfully (and how certifiably) the cheap one-mat-vec
/// encoder inverts the dictionary the inner solve converged to.
#[derive(Debug, Clone, Copy)]
pub struct AmortizedEncoderConsistency {
    /// Mean per-element squared gap between the amortized reconstruction and the
    /// exact fitted reconstruction (`‖x̂_amortized − x̂_exact‖² / (n·p)`). Zero ⇒
    /// the IFT predictor reproduces the encode map exactly to first order.
    pub recon_consistency: f64,
    /// Fraction of (row, atom) amortized encodes whose Kantorovich certificate
    /// failed (`h > ½`) and fell back to the certified Newton encode.
    pub uncertified_fraction: f64,
    /// Count of uncertified (row, atom) encodes (numerator of the fraction).
    pub n_uncertified: usize,
    /// Total (row, atom) encodes scored (`n · K`).
    pub n_encodes: usize,
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
            temperature_schedule: None,
            last_row_layout: None,
            row_metric: None,
            collapse_events: Vec::new(),
            row_loss_weights: None,
            last_frames_active: false,
            assembly_chunk_override: None,
            fixed_decoder_assembly: false,
            softmax_active_cap: None,
            border_hbb_workspace: Array2::<f64>::zeros((0, 0)),
            certificate_dispersion: None,
            curvature_walk_report: None,
            expected_evidence_gauge_deflated_directions: None,
            evidence_gauge_deflation_reanchors: 0,
            evidence_gauge_deflation_last_delta_sign: 0,
            dictionary_cocollapse_reseeds: 0,
            best_cocollapse_incumbent: None,
            decoder_repulsion_gate: None,
            hybrid_split_report: None,
            atom_inner_fits: None,
            oos_linear_images: None,
        })
    }

    /// #1408/#1409 — install the optional hard per-row active-atom cap for
    /// Softmax mode (threaded from the fit/encode `top_k`). A `Some(k)` with
    /// `1 <= k < K` makes the Softmax assignment optimize on the COMPACT
    /// top-`k` row layout (see [`Self::softmax_active_cap`]); `Some(k) >= K`
    /// and `None` are both no-ops (full support). Non-softmax modes ignore it.
    pub fn set_softmax_active_cap(&mut self, top_k: Option<usize>) {
        self.softmax_active_cap = match top_k {
            Some(k) if k >= 1 && k < self.k_atoms() => Some(k),
            _ => None,
        };
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
        rho: &SaeManifoldRho,
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
            assignments.push(self.assignment.try_assignments_row_for_rho(row, rho)?);
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
    /// Weights must be finite and strictly positive, one per term row. They
    /// are self-normalized to mean `1.0` here (only the *relative* design
    /// correction matters at the fitted sample size; the absolute `n/budget`
    /// scale would silently inflate the dispersion estimate against the
    /// sample-sized dof). Weights that are identically equal after
    /// normalization (an exact full pass, or any uniform design) are stored
    /// as `None`, so the unweighted path stays bit-for-bit identical rather
    /// than "multiplied by 1.0".
    pub fn set_row_loss_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
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
        if !weights.iter().all(|w| w.is_finite() && *w > 0.0) {
            return Err(
                "SaeManifoldTerm::set_row_loss_weights: weights must be finite and strictly \
                 positive"
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
    pub fn set_row_metric(
        &mut self,
        metric: gam_problem::RowMetric,
    ) -> Result<(), String> {
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
    pub(crate) fn diagnostic_metric(
        &self,
    ) -> Result<gam_problem::RowMetric, String> {
        match self.row_metric() {
            Some(metric) => Ok(metric.clone()),
            None => {
                gam_problem::RowMetric::euclidean(self.n_obs(), self.output_dim())
            }
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
        assignments_override: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeManifoldFitDiagnostics, String> {
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
            crate::inference::atom_lens::atom_two_lens(self, &metric, assignments_override);

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
        let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
            if isometry_pin_active {
                views
                    .iter()
                    .map(|view| {
                        view.as_ref().and_then(|v| {
                            crate::identifiability::isometry_orbit_penalty_operator(
                                v, 1.0,
                            )
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
            crate::identifiability::residual_gauge_exact(
                &certificate_model,
                &views,
                &ops,
            )?
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
        let atom_inference =
            crate::identifiability::atom_inference_reports(&certificate_model);

        Ok(SaeManifoldFitDiagnostics {
            atom_two_lens,
            residual_gauge,
            incoherence_report: match reconstruction_dispersion.or(self.certificate_dispersion) {
                Some(dispersion) => Some(dictionary_incoherence_report_with_dispersion(
                    self, dispersion,
                )?),
                None => None,
            },
            atom_inference,
        })
    }

    /// Build the trust-diagnostics producer for the Python `diagnostics` block.
    ///
    /// `assignments` is supplied by the payload assembly site so top-k projection,
    /// when requested, is reflected in coverage/frequency and in the tangent
    /// spectra. The active threshold is shared with the atom lens so all
    /// assignment-support diagnostics agree on what "active" means.
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
        let active_threshold = crate::inference::atom_lens::SAE_TRUST_ACTIVE_MASS_FLOOR;
        let mut atoms = Vec::with_capacity(k_atoms);
        let mut atom_trust = Vec::with_capacity(k_atoms);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let mut active_token_count = 0usize;
            let mut activation_sum = 0.0_f64;
            for row in 0..n {
                let mass = assignments[[row, atom_idx]];
                activation_sum += mass;
                if mass > active_threshold {
                    active_token_count += 1;
                }
            }
            let coverage = if n > 0 {
                active_token_count as f64 / n as f64
            } else {
                0.0
            };
            let activation_frequency = if n > 0 {
                activation_sum / n as f64
            } else {
                0.0
            };
            let (sigma_min_tangent, sigma_max_tangent) = self
                .atom_tangent_spectrum_from_assignments(
                    atom_idx,
                    assignments,
                    &metric,
                    active_threshold,
                )?;
            let tangent_condition_score = if sigma_max_tangent > 0.0 {
                (sigma_min_tangent / sigma_max_tangent).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let trust_score = tangent_condition_score;
            atom_trust.push(trust_score);
            atoms.push(SaeAtomTrustDiagnostics {
                trust_score,
                sigma_min_tangent,
                sigma_max_tangent,
                tangent_condition_score,
                coverage,
                activation_frequency,
                untyped: matches!(atom.basis_kind, SaeAtomBasisKind::Precomputed(_)),
                active_token_count,
            });
        }
        Ok(SaeTrustDiagnostics { atom_trust, atoms })
    }

    pub(crate) fn atom_tangent_spectrum_from_assignments(
        &self,
        atom_idx: usize,
        assignments: ArrayView2<'_, f64>,
        metric: &gam_problem::RowMetric,
        active_threshold: f64,
    ) -> Result<(f64, f64), String> {
        let atom = &self.atoms[atom_idx];
        let d = atom.latent_dim;
        let p = self.output_dim();
        if d == 0 || p == 0 {
            return Ok((0.0, 0.0));
        }
        let mut gram = Array2::<f64>::zeros((d, d));
        let mut active_mass_sum = 0.0_f64;
        let mut jac_row = vec![0.0_f64; p * d];
        for row in 0..self.n_obs() {
            let mass = assignments[[row, atom_idx]];
            if !(mass > active_threshold) {
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
                (
                    SaeAtomBasisKind::Linear
                    | SaeAtomBasisKind::Duchon
                    | SaeAtomBasisKind::EuclideanPatch
                    | SaeAtomBasisKind::Poincare
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
    ///    (`IBPAssignment`, `SoftmaxAssignmentSparsity`) are refused because the SAE
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
        let mut row_block_penalty_present = false;
        for penalty in &registry.penalties {
            if penalty.tier() != PenaltyTier::Psi {
                continue;
            }
            if matches!(
                penalty,
                AnalyticPenaltyKind::IBPAssignment(_)
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
                     Sparsity/TopK/JumpReLU, RowPrecisionPrior, \
                     ParametricRowPrecisionPrior, ScadMcp, Isometry) on the \
                     coord latent block, or move the penalty to a non-SAE \
                     term",
                    penalty.name()
                ));
            }
            row_block_penalty_present = true;
        }
        if row_block_penalty_present {
            let mut dims = self.assignment.coords.iter().map(|c| c.latent_dim());
            if let Some(first) = dims.next() {
                if let Some(mismatch) = dims.find(|d| *d != first) {
                    return Err(format!(
                        "SAE-manifold term refuses row-block analytic penalty: \
                         atoms have heterogeneous coord latent dims (saw {first} \
                         and {mismatch}). Row-block penalties (ARD, \
                         BlockOrthogonality, ...) target the unified \"t\" \
                         latent block whose declared `d` matches one shape; \
                         per-atom dispatch with mixed `d_k` would silently \
                         truncate or expand axes. Configure all atoms with the \
                         same `atom_dim`, or split the row-block penalty into \
                         per-atom descriptors keyed to per-atom latent blocks"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn output_dim(&self) -> usize {
        self.atoms[0].output_dim()
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

    /// Factored arrow-Schur border dimension `Σ_k M_k · r_k` (issue #972): the
    /// number of decoder coordinates the border actually carries once the
    /// low-rank Grassmann frames are profiled out. Atoms with no active frame
    /// contribute their full `M_k · p` (`r_k == p`), so on the all-full-`B` path
    /// this equals [`Self::beta_dim`]. The border Cholesky / evidence log-det
    /// scale with THIS count, not `beta_dim`.
    pub fn factored_border_dim(&self) -> usize {
        self.atoms.iter().map(|a| a.border_coeff_count()).sum()
    }

    /// Total profiled-out Grassmann manifold dimension `Σ_k r_k·(p − r_k)` across
    /// all active frames (issue #972). This is the count of decoder-frame degrees
    /// of freedom estimated OUTSIDE the border by closed-form polar steps, and it
    /// must enter the Laplace evidence dimension accounting (evidence honesty):
    /// the profiled frame is a MAP point on `∏_k Gr(r_k, p)`, contributing this
    /// many free dimensions to the model. `0` when every atom is on the full-`B`
    /// path. Threaded into [`Self::reml_occam_term`].
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
        rho: &SaeManifoldRho,
    ) -> Result<usize, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if n == 0 {
            return Ok(0);
        }
        // Per-row assignments and per-(row, atom) decoded outputs, computed once.
        let mut assignments = Vec::with_capacity(n);
        for row in 0..n {
            assignments.push(self.assignment.try_assignments_row_for_rho(row, rho)?);
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
        // Full fitted reconstruction `Σ_k a_k decoded_k`, so the per-atom partial
        // residual is `e_k = (z − fitted) + a_k decoded_k` (add atom k back in).
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
            for row in 0..n {
                let a = assignments[row][atom_idx];
                // Partial residual e_{n,k} = z_n − (fitted − a_k decoded_k).
                for c in 0..p {
                    let e = target[[row, c]] - fitted[[row, c]] + a * decoded[[row, atom_idx, c]];
                    targets[[row, c]] = a * e;
                }
                // In-span coordinate ĉ_{n,k} = Φ_k(t_n)·C_k ∈ ℝ^r.
                for j in 0..r {
                    let mut acc = 0.0_f64;
                    for basis_col in 0..m {
                        acc += self.atoms[atom_idx].basis_values[[row, basis_col]]
                            * coords_c[[basis_col, j]];
                    }
                    rcoords[[row, j]] = a * acc;
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

    /// Decide whether the sparse per-row active-set layout is engaged for a
    /// dense-weight assignment mode, and if so derive the per-row active-atom
    /// cap and magnitude cutoff.
    ///
    /// #1408: this plan is mode-agnostic. `assemble_arrow_schur` consults it
    /// directly for IBP-MAP, and for `AssignmentMode::Softmax` via
    /// [`Self::softmax_active_plan`], which tightens it with an explicit `top_k`
    /// (`softmax_active_cap`). Softmax therefore engages the compact active-set
    /// layout whenever `top_k` or the budget bounds the active set (the
    /// active-sub-block Gershgorin majorizer + coherent logdet/θ-adjoint are
    /// landed — see `SaeRowLayout`'s doc); it keeps the full `K`-atom layout only
    /// when neither lever engages. The decision is auto-derived from
    /// the problem size and the device/host working-set budget — never a CLI flag
    /// or kwarg. JumpReLU is not handled here (it always uses its structural gate
    /// via [`SaeRowLayout::from_jumprelu`]). The dense Gauss-Newton data Gram `G`
    /// is `(m_total × m_total)` f64; if its dense form fits the budget we keep
    /// the exact full-support solve (every atom active per row), so small-`K`
    /// problems are bit-for-bit unchanged. Above that, we cap each row to the
    /// `k_active` atoms that make the *sparse* Gram fit the same budget, with a
    /// relative magnitude cutoff that drops assignment mass contributing
    /// negligible `O(a²)` curvature.
    ///
    /// Returns `Some((k_active_cap, cutoff))` to engage sparsity, or `None` to
    /// keep the dense full-support layout.
    pub(crate) fn sparse_active_plan(&self) -> Option<(usize, f64)> {
        // The per-row Riemannian tangent projection for non-Euclidean atom
        // latents is now applied directly on the compact active-set rows (see
        // the `Some(layout)` arm in `assemble_arrow_schur`, via
        // `compact_row_ext_manifold_and_point`), which rebuilds each row's
        // product manifold in its compact column order and applies the SAME
        // gt/htt/htbeta + Kronecker-Jacobian projections the dense path uses. So
        // the sparse plan may engage on curved ext-coord manifolds (circle /
        // torus / sphere atoms) — the affordability lever for manifold-SAE at
        // large `K`, where the dense `K²` co-assignment Gram is the cost. (The
        // former `is_euclidean()`-only restriction punted every curved atom to
        // the dense layout; it is lifted.) The host/device in-core budget is the
        // single gate now; it is parameterised in `sparse_active_plan_for_budget`
        // so the engagement regression can pin a small budget without allocating
        // a multi-GB dense Gram.
        let budget = match crate::gpu::device_runtime::GpuRuntime::global() {
            // Allow up to one quarter of the AGGREGATE device budget for the dense
            // Gram, matching the streaming dispatcher's in-core fraction. The
            // per-atom-pair Gram blocks fan out across the whole device pool, so
            // the in-core fraction sums every ordinal's budget, not just the
            // primary's.
            Some(rt) => {
                let aggregate: usize = rt
                    .device_ordinals()
                    .iter()
                    .map(|&ord| rt.memory_budget_for(ord))
                    .sum();
                aggregate / 4
            }
            None => sae_host_in_core_budget_bytes().0,
        };
        self.sparse_active_plan_for_budget(budget)
    }

    /// Budget-parameterised core of [`Self::sparse_active_plan`]. The dense data
    /// Gram footprint `(m_total · m_total) f64` is compared against `budget`; a
    /// term whose dense Gram exceeds the budget engages the compact active-set
    /// plan (returns `Some((k_active_cap, cutoff))`), regardless of whether any
    /// atom latent is curved. Pulled out so the curved-atom engagement
    /// regression can pin a small budget deterministically.
    pub(crate) fn sparse_active_plan_for_budget(&self, budget: usize) -> Option<(usize, f64)> {
        // Relative magnitude cutoff: assignment mass below this fraction of the
        // row's peak `|a_k|` enters the Gram only as `O(a²)` curvature and is
        // dropped. Chosen so dropped terms are ~1e-6 of the peak self-coupling.
        const RELATIVE_CUTOFF: f64 = 1.0e-3;

        let k_atoms = self.k_atoms();
        if k_atoms <= 1 {
            return None;
        }
        let p = self.output_dim();
        let m_total: usize = self.atoms.iter().map(|a| a.basis_size()).sum();
        // Dense data Gram footprint: (m_total · m_total) f64.
        let dense_gram_bytes = m_total
            .saturating_mul(m_total)
            .saturating_mul(SAE_BYTES_PER_F64);
        if dense_gram_bytes <= budget {
            return None;
        }

        // Sparse Gram footprint scales with the per-row active basis count
        // `k_active · m_atom`. Solve for the largest `k_active` whose sparse
        // Gram `(k_active · m_atom)²` still fits the budget.
        let m_atom = (m_total as f64 / k_atoms as f64).max(1.0);
        let max_active_basis = ((budget as f64 / SAE_BYTES_PER_F64 as f64).sqrt() / m_atom).floor();
        let k_active_cap = (max_active_basis as usize).clamp(1, k_atoms);
        // p does not enter the Gram dimension (it is carried by the `⊗ I_p`
        // structure), but a degenerate `p == 0` term has no decoder columns.
        if p == 0 {
            return None;
        }
        Some((k_active_cap, RELATIVE_CUTOFF))
    }

    /// #1408/#1409 — per-row active-set plan for the Softmax assignment.
    ///
    /// Engages the compact top-`k` row layout when EITHER the user supplied a
    /// hard `top_k` cap ([`Self::softmax_active_cap`], `1 <= k < K`) OR the
    /// dense data Gram exceeds the in-core budget (the same memory lever the
    /// IBP path uses via [`Self::sparse_active_plan`]). The returned
    /// `k_active_cap` is the tighter of the two, so an explicit `top_k`
    /// genuinely bounds the optimization even below the memory threshold and a
    /// large-K budget breach still bounds it when no `top_k` is set. Returns
    /// `None` (keep the exact full-`K` dense softmax layout) when neither lever
    /// engages.
    ///
    /// The cutoff is the same relative magnitude floor as the budget plan
    /// (`1e-3` of the row peak); under an explicit `top_k` cap alone (no budget
    /// breach) it is `0.0` so exactly the top-`k` atoms are retained.
    pub(crate) fn softmax_active_plan(&self) -> Option<(usize, f64)> {
        if self.k_atoms() <= 1 {
            return None;
        }
        let budget_plan = self.sparse_active_plan();
        match (self.softmax_active_cap, budget_plan) {
            (Some(cap), Some((budget_cap, cutoff))) => Some((cap.min(budget_cap), cutoff)),
            // Explicit cap only: retain exactly the top-`cap` atoms (no extra
            // magnitude cutoff beyond the cap).
            (Some(cap), None) => Some((cap, 0.0)),
            (None, plan) => plan,
        }
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
                Some(rho) => self.assignment.try_assignments_row_for_rho(row, rho)?,
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
            self.atoms[atom_idx].refresh_intrinsic_smooth_penalty();
        }
        Ok(())
    }

    pub fn fitted(&self) -> Array2<f64> {
        self.try_fitted().expect("assignment logits must be finite")
    }

    /// The #1026 hybrid-collapse substitution map: `atom_idx → &AtomLinearImage`
    /// for every `d = 1` slot whose post-fit verdict selected its straight
    /// (`Θ → 0`) sub-model. Empty when no report has been computed
    /// (`hybrid_split_report == None`, e.g. mid-fit) or no slot collapsed. The
    /// SINGLE source of the collapse policy — every reconstruction path (the
    /// rho-keyed `try_fitted_with_rho`, the explicit-assignment
    /// [`Self::reconstruct_from_assignments`] used by the top-k projection)
    /// reads it so train, OOS, and top-k reconstructions decode collapsed slots
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
    /// this (typically OOS) term so its reconstruction (`fitted` / the top-k
    /// assembler) decodes verdict-linear `d = 1` slots by the SAME straight
    /// sub-model the training reconstruction used, instead of the original
    /// curved decoder. Each image's `atom_idx` must index a real slot; an image
    /// whose channel count `p` disagrees with this term's output dim, or whose
    /// `atom_idx` is out of range, is rejected so a stale/mismatched payload
    /// cannot silently corrupt the reconstruction. Pass an empty slice (or never
    /// call this) for an all-curved OOS reconstruction.
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
        for img in &images {
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
    /// per-row assignment matrix (e.g. a hard top-k projection of the fitted
    /// soft assignments), honouring the #1026 hybrid collapse when `collapse` is
    /// set: a verdict-linear `d = 1` slot decodes its straight sub-model image
    /// instead of its curved curve, exactly as the production `try_fitted` does.
    /// This is the shared assembler the FFI top-k path uses so the projected
    /// reconstruction composes with hybrid collapse (#1233) instead of
    /// re-deriving the curved image by hand and silently bypassing the verdict.
    /// The atom coordinates (`t`) and decoded curves are the term's own fitted
    /// ones; only the assignment masses come from `assignments`.
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
        let mut out = Array2::<f64>::zeros((n, p));
        let mut g_buf = vec![0.0_f64; p];
        for row in 0..n {
            for atom_idx in 0..k_atoms {
                let a_k = assignments[[row, atom_idx]];
                if a_k == 0.0 {
                    continue;
                }
                if let Some(image) = linear_images.get(&atom_idx) {
                    let own_t = self.assignment.coords[atom_idx].as_matrix()[[row, 0]];
                    image.fill_row(image.coordinate_for_row(row, own_t), &mut g_buf);
                } else {
                    self.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                }
                let mut out_row = out.row_mut(row);
                for out_col in 0..p {
                    out_row[out_col] += a_k * g_buf[out_col];
                }
            }
        }
        Ok(out)
    }

    pub fn try_fitted(&self) -> Result<Array2<f64>, String> {
        // Production/user-facing reconstruction: honours the #1026 hybrid-split
        // verdict (verdict-linear `d = 1` slots decode their straight sub-model).
        self.try_fitted_with_rho(None, true)
    }

    pub(crate) fn try_fitted_for_rho(&self, rho: &SaeManifoldRho) -> Result<Array2<f64>, String> {
        // Internal/fitting reconstruction: the pure CURVED image (the joint fit
        // and the #1026 adjudication both require the uncollapsed curve).
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
        // Reuse a single scratch buffer across all (row, atom) pairs instead of
        // allocating a fresh `Array1<f64>` of length p per call.
        let mut g_buf = vec![0.0_f64; p];
        for row in 0..n {
            let a = match rho {
                Some(rho) => self.assignment.try_assignments_row_for_rho(row, rho)?,
                None => self.assignment.try_assignments_row(row)?,
            };
            for atom_idx in 0..k_atoms {
                let a_k = a[atom_idx];
                if let Some(image) = linear_images.get(&atom_idx) {
                    // Verdict-linear slot: substitute the straight sub-model image
                    // at this row's fitted on-atom coordinate — or, for a #1026
                    // collapse-rescued slot, at its fresh per-row code.
                    let own_t = self.assignment.coords[atom_idx].as_matrix()[[row, 0]];
                    image.fill_row(image.coordinate_for_row(row, own_t), &mut g_buf);
                } else {
                    self.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                }
                let mut out_row = out.row_mut(row);
                for out_col in 0..p {
                    out_row[out_col] += a_k * g_buf[out_col];
                }
            }
        }
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
    pub fn hybrid_split_report(
        &self,
    ) -> Option<&crate::hybrid_split::SaeHybridSplitReport> {
        self.hybrid_split_report.as_ref()
    }

    /// Build the #1026 curved-vs-linear hybrid-split report by adjudicating each
    /// eligible `d = 1` atom's fitted curved image against its straight (linear
    /// special-case) sub-model on the common rank-aware Laplace evidence scale.
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
            weights.push(self.assignment.try_assignments_row_for_rho(row, rho)?);
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
            weights.push(self.assignment.try_assignments_row_for_rho(row, rho)?);
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
    /// computed yet (`hybrid_split_report == None`).
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
        // point keeps the rho-keyed collapse for the #1026 EV-dominance reporting
        // (`hybrid_collapsed_explained_variance`) and the regression battery.
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
        let collapsed = self.hybrid_collapsed_reconstruction(rho)?;
        Ok(reconstruction_explained_variance(target, collapsed.view()))
    }

    /// #1026 ladder item 2/3 — the AMORTIZED ENCODER, wired from the fitted
    /// dictionary. Builds the offline certified [`EncodeAtlas`] over this term's
    /// frozen atoms and encodes a target corpus `targets` (`n × p`) through the
    /// per-chart distilled Jacobian predictor, with the Kantorovich certificate
    /// gating each row and an exact-solve fallback for the rows the amortized
    /// predictor cannot certify. Returns one [`EncodeResult`] per atom (the
    /// per-atom encoded coordinates + per-row certificate mask), in dictionary
    /// order.
    ///
    /// This is the thread's "encoder + certificate-gated exact fallback"
    /// deployment made reachable from a fit: the distilled map approximates
    /// inference at one mat-vec/row, and any row whose amortized prediction fails
    /// `h ≤ ½` falls back to the certified IFT-warm-start Newton encode
    /// ([`EncodeAtlas::certified_encode_row`]); rows that still cannot be
    /// certified ride the [`EncodeResult::encode_uncertified_count`] flag for the
    /// upstream exact multi-start solve (honesty, never a silent wrong encode).
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
    ) -> Result<Vec<crate::encode::EncodeResult>, String> {
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

        // Per-atom amortized encode with a certificate-gated exact-solve fallback:
        // a row whose distilled prediction fails `h ≤ ½` is retried through the
        // certified IFT-warm-start Newton path; a row that still cannot be
        // certified stays flagged for the upstream multi-start solve.
        // (The atlas is rho-free; the per-row amplitudes already carry the
        // rho-resolved assignment masses the caller produced upstream.)
        let mut results = Vec::with_capacity(k_atoms);
        for atom_idx in 0..k_atoms {
            let atom = &self.atoms[atom_idx];
            let amp_col = amplitudes.column(atom_idx).to_owned();
            let amortized =
                atlas.amortized_encode_batch(atom, atom_idx, targets, amp_col.view())?;
            let mut coords = amortized.coords;
            let mut certified = amortized.certified;
            for row in 0..n {
                if certified[row] {
                    continue;
                }
                let (t, cert) =
                    atlas.certified_encode_row(atom, atom_idx, targets.row(row), amp_col[row])?;
                if cert.certified() {
                    coords.row_mut(row).assign(&t);
                    certified[row] = true;
                }
            }
            results.push(crate::encode::EncodeResult::from_rows(
                coords, certified,
            ));
        }
        Ok(results)
    }

    /// #1026 — the fitted per-row assignment masses `a[i,k]` (the activation
    /// amplitudes `z_k` the amortized encode recovers `t` against), as an
    /// `n × K` matrix. These are exactly the masses
    /// [`Self::try_fitted_with_rho`] assembles the reconstruction from, so
    /// feeding them to [`Self::amortized_encode_target`] re-encodes the SAME
    /// inference the dictionary was fit against — the self-consistency the
    /// distilled encoder is supervised to approximate.
    pub fn fitted_assignment_amplitudes(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let mut amplitudes = Array2::<f64>::zeros((n, k_atoms));
        for row in 0..n {
            let a = self.assignment.try_assignments_row_for_rho(row, rho)?;
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
        rho: &SaeManifoldRho,
    ) -> Result<Vec<crate::encode::EncodeResult>, String> {
        let amplitudes = self.fitted_assignment_amplitudes(rho)?;
        self.amortized_encode_target(targets, amplitudes.view())
    }

    /// #1154 — amortized-encoder consistency of the CURRENT dictionary against
    /// its own fit-time target. This is the co-training signal of the joint
    /// amortized-encoder + REML loop (Design A): the amortized (one-mat-vec)
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
    ///   with REML steers the fit toward dictionaries that admit a fast,
    ///   faithful amortized encode — the architectural co-adaptation #1154 adds.
    /// * `uncertified_fraction`: the share of (row, atom) encodes whose
    ///   Kantorovich certificate failed (`h > ½`), i.e. that fell back to the
    ///   certified IFT-warm-start Newton. This is the encoder's *certifiable coverage*
    ///   of the dictionary; co-training rewards dictionaries the cheap encode
    ///   certifies, not just ones it happens to land.
    ///
    /// The certificate keeps every accepted amortized coord honest (uncertified
    /// rows already ride the exact fallback inside `amortized_encode_target`), so
    /// this metric never silently trusts a wrong encode — it MEASURES how much of
    /// the dictionary the cheap encoder can faithfully and certifiably invert.
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
        let amplitudes = self.fitted_assignment_amplitudes(rho)?;
        let encodes = self.amortized_encode_target(targets, amplitudes.view())?;
        // The EXACT fitted reconstruction the inner solve converged to (pure
        // curved image, rho-keyed) is the supervision target for the amortized
        // reconstruction. Both are n×p ambient, so the comparison is layout-free.
        let exact_recon = self.try_fitted_for_rho(rho)?;

        // Build the amortized reconstruction Σ_k z_k · Φ_k(t̂_k) B_k by decoding
        // each atom's amortized coords through that atom's own basis evaluator.
        let mut amortized_recon = Array2::<f64>::zeros((n, p));
        let mut uncertified = 0usize;
        for atom_idx in 0..k_atoms {
            let atom = &self.atoms[atom_idx];
            let result = &encodes[atom_idx];
            // An atom with no basis evaluator cannot decode an amortized
            // reconstruction; every one of its rows is necessarily uncertified
            // (the encode flagged them all), so it contributes nothing to the
            // amortized recon and its full row-count to the uncertified tally.
            // Count it and skip the decode rather than erroring — the consistency
            // fold stays a bounded penalty, never a hard abort of the criterion.
            let Some(evaluator) = atom.basis_evaluator.as_ref() else {
                uncertified += n;
                continue;
            };
            uncertified += result.encode_uncertified_count;
            // Decode the amortized coords: Φ_k(t̂) is (n × M_k); B_k is (M_k × p).
            let (phi, _jac) = evaluator.evaluate(result.coords.view())?;
            let decoded = phi.dot(&atom.decoder_coefficients); // (n × p)
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
        let total_encodes = (n * k_atoms).max(1) as f64;
        let uncertified_fraction = uncertified as f64 / total_encodes;

        Ok(AmortizedEncoderConsistency {
            recon_consistency,
            uncertified_fraction,
            n_uncertified: uncertified,
            n_encodes: n * k_atoms,
        })
    }

    /// #1154 — the co-trained REML criterion: the exact REML criterion at `rho`
    /// PLUS the amortized-encoder consistency penalty, so the outer optimizer
    /// co-adapts the dictionary + smoothing parameters λ TOWARD a dictionary the
    /// fast amortized encoder can faithfully and certifiably invert.
    ///
    /// This is Design A of #1154. The inner solve still converges the `(t, β)`
    /// system to stationarity at the engine's current ρ (so the implicit-function
    /// REML λ-gradient `dβ̂/dλ = −(H+S_λ)⁻¹(dS_λ/dλ)β̂` stays EXACT — the encoder
    /// only warm-starts/co-adapts, it never replaces the stationary point). The
    /// added term
    ///
    /// ```text
    ///   J_cotrain(ρ) = REML(ρ)  +  w · ‖x̂_amortized − x̂_exact‖²/(n·p)
    ///                            +  w_cert · uncertified_fraction
    /// ```
    ///
    /// folds the post-fit amortized-encode quality into the ranked objective. The
    /// weights are auto-scaled to the REML criterion magnitude (magic by default:
    /// no caller knob) so the consistency term is a meaningful but non-dominant
    /// fraction of the objective regardless of problem scale.
    pub fn reml_criterion_cotrained(
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
        // `reml_criterion_cotrained` (the encode/warm path for the cotrained
        // objective). Good warm-starts from the running dictionary land the
        // inner solve closer to the stationary point used for the fold.
        // Advisory only (0 or err falls back to cold); telemetry recorded by
        // outer objective callers when present.
        self.warm_start_latents_from_amortized_encoder(target, rho)
            .unwrap_or(0);
        let (reml, loss) = self.reml_criterion_with_refine_policy(
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
        // Auto-scale the co-training weights to the REML magnitude so the
        // consistency penalty is a bounded, scale-free fraction of the objective
        // (magic by default: no caller knob). `reml_scale` floors at 1 so a
        // near-zero criterion still admits a meaningful consistency contribution.
        let cotrained = Self::fold_cotrain_consistency(reml, &consistency);
        Ok((cotrained, loss, consistency))
    }

    /// #1154 — the single source of the co-training fold arithmetic: add the
    /// auto-scaled amortized-encoder consistency penalty to an already-computed
    /// REML criterion at the converged dictionary. Both the public
    /// [`Self::reml_criterion_cotrained`] entry point and the outer-loop value /
    /// gradient lanes (`SaeManifoldOuterObjective::fold_cotrain_consistency`)
    /// route through THIS function, so the folded objective cannot drift between
    /// the criterion and the cascade-ranked cost (the objective↔gradient desync
    /// bug class). The weights are auto-scaled to the REML magnitude (`max(|REML|,
    /// 1)`) so the penalty is a bounded, scale-free fraction of the objective
    /// regardless of problem scale; the fold carries no analytic gradient (under
    /// Design A the REML λ-gradient stays the exact implicit-function path).
    #[must_use]
    pub fn fold_cotrain_consistency(
        reml_cost: f64,
        consistency: &AmortizedEncoderConsistency,
    ) -> f64 {
        let reml_scale = reml_cost.abs().max(1.0);
        reml_cost
            + COTRAIN_RECON_WEIGHT * reml_scale * consistency.recon_consistency
            + COTRAIN_CERT_WEIGHT * reml_scale * consistency.uncertified_fraction
    }

    /// #1154 item 2 — warm-start the inner latent coordinates from the amortized
    /// encoder (Design A). Builds the per-chart IFT-Jacobian atlas from the
    /// CURRENT dictionary, runs the one-mat-vec amortized encode of `target`
    /// against each atom at the rho-resolved assignment masses, and overwrites
    /// each atom's stored latent coords with the predicted `t̂` ON THE ROWS THE
    /// KANTOROVICH CERTIFICATE ACCEPTS. Uncertified rows are left at their
    /// current coords (the previous-iterate start), so the
    /// warm-start can only HELP — a row the cheap predictor cannot certify never
    /// corrupts the seed. The subsequent inner Newton refines from this seed to
    /// the SAME stationary point (the warm-start changes only the basin entry,
    /// not the root), so the REML λ-gradient stays exactly the implicit-function
    /// path and the criterion is unchanged at convergence — the amortized encoder
    /// only accelerates/co-adapts the inner solve, it never replaces the
    /// stationary point.
    ///
    /// Returns the number of (row, atom) coords actually warm-started (the
    /// certified-prediction count), for instrumentation / tests. A first-build
    /// dictionary with no usable charts simply warm-starts nothing and returns 0
    /// (the cold path is byte-for-byte unchanged).
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
        let amplitudes = self.fitted_assignment_amplitudes(rho)?;
        let encodes = self.amortized_encode_target(target, amplitudes.view())?;
        let mut warm_started = 0usize;
        for atom_idx in 0..k_atoms {
            let d = self.atoms[atom_idx].latent_dim;
            if d == 0 {
                continue;
            }
            let result = &encodes[atom_idx];
            // Start from the atom's CURRENT coords so uncertified rows are left
            // exactly as they were; overwrite only the certified predictions.
            let mut coords = self.assignment.coords[atom_idx].as_matrix();
            if coords.dim() != (n, d) {
                return Err(format!(
                    "warm_start_latents_from_amortized_encoder: atom {atom_idx} coords {:?} != (n={n}, d={d})",
                    coords.dim()
                ));
            }
            for row in 0..n {
                if !result.certified[row] {
                    continue;
                }
                for axis in 0..d {
                    coords[[row, axis]] = result.coords[[row, axis]];
                }
                warm_started += 1;
            }
            // `as_matrix` lays coords out row-major (`[[row, axis]]`), exactly the
            // `values[row*d + axis]` order `set_flat` expects, so a plain
            // row-major iterator reconstructs the flat vector.
            let flat = Array1::from_iter(coords.iter().copied());
            self.assignment.coords[atom_idx].set_flat(flat.view());
        }
        // The basis caches must follow the freshly-seeded coords so the next
        // inner solve evaluates Φ at the warm-started t̂, not the stale coords.
        self.refresh_basis_from_current_coords()?;
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
        // #1017: the data-fit is the dominant per-line-search-trial cost (it
        // re-runs every Armijo halving × every inner Newton iteration × every
        // outer ρ evaluation). The old path materialised the whole `n × p`
        // fitted matrix (`try_fitted_for_rho`) and then walked it AGAIN to form
        // the residual sum — two sequential `n·p` passes plus an `n·p`
        // allocation per trial. Fuse the reconstruction and the residual reduce
        // into ONE row-parallel pass that never materialises the fitted matrix:
        // each row decodes its atoms into per-worker scratch, differences
        // against the target, and contributes its scalar `0.5·w·‖r‖²` to a
        // chunk-ordered fold (bit-identical run-to-run). Per-worker scratch
        // (`map_init`) keeps the only allocations one `g_buf`/`fitted_row` pair
        // per rayon thread rather than per row. Stay sequential inside a worker
        // (the topology race owns the outer pool) to avoid nested
        // oversubscription.
        let parallel = n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        let row_data_fit =
            |row: usize,
             g_buf: &mut [f64],
             fitted_row: &mut [f64],
             assign_buf: &mut [f64]|
             -> Result<f64, String> {
                // #1557 — fill the per-atom assignment row into reused per-worker
                // scratch via the `_into` twin instead of heap-allocating a fresh
                // `Array1` per row per loss eval. Bit-identical to the allocating
                // `try_assignments_row_for_rho` (same arithmetic, same order); this
                // loss reruns every Armijo halving × inner Newton iter × outer ρ
                // eval, so the per-row K-sized allocation was a hot-path churn.
                self.assignment
                    .try_assignments_row_for_rho_into(row, rho, assign_buf)?;
                let a = &*assign_buf;
                for slot in fitted_row.iter_mut() {
                    *slot = 0.0;
                }
                for atom_idx in 0..k_atoms {
                    self.atoms[atom_idx].fill_decoded_row(row, g_buf);
                    let a_k = a[atom_idx];
                    for out_col in 0..p {
                        fitted_row[out_col] += a_k * g_buf[out_col];
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
        let data_fit = if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 32;
            let partials: Vec<Result<f64, String>> = (0..n)
                .into_par_iter()
                .chunks(CHUNK)
                .map_init(
                    || (vec![0.0_f64; p], vec![0.0_f64; p], vec![0.0_f64; k_atoms]),
                    |(g_buf, fitted_row, assign_buf), idxs| {
                        // #1557 — pin any faer GEMM reached from this row-parallel
                        // data-fit chunk to `Par::Seq` (no nested Rayon re-fan); the
                        // per-row reductions are tiny, so the result is bit-identical.
                        with_nested_parallel(|| {
                            let mut acc = 0.0_f64;
                            for row in idxs {
                                acc += row_data_fit(row, g_buf, fitted_row, assign_buf)?;
                            }
                            Ok(acc)
                        })
                    },
                )
                .collect();
            let mut total = 0.0_f64;
            for partial in partials {
                total += partial?;
            }
            total
        } else {
            let mut g_buf = vec![0.0_f64; p];
            let mut fitted_row = vec![0.0_f64; p];
            let mut assign_buf = vec![0.0_f64; k_atoms];
            let mut total = 0.0_f64;
            for row in 0..n {
                total += row_data_fit(row, &mut g_buf, &mut fitted_row, &mut assign_buf)?;
            }
            total
        };
        let assignment_sparsity = assignment_prior_value(&self.assignment, rho);
        let smoothness = penalty_scale * self.decoder_smoothness_value(&rho.lambda_smooth_vec());
        let ard = self.ard_value(rho)?;
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
            evidence_gauge_deflated_directions: 0,
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
    /// Isometry, ScadMcp, BlockOrthogonality, IBP/softmax assignment sparsity).
    /// The SAE already carries its own ARD (`loss.ard`) and assignment sparsity
    /// (`loss.assignment_sparsity`) energy, so adding the registry ARD /
    /// assignment value on top would double-count, and the gauge-only
    /// coordinate penalties are not part of the penalized deviance the
    /// REML/Laplace criterion scores. The decoder-block penalties, by contrast,
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
    /// the REML scalar). They are left out here because they share a registry
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
    /// data-fit / assignment / smoothness / ARD), so the Laplace/REML criterion
    /// must add it explicitly to score the same penalized objective the inner
    /// solve descends.
    pub fn isometry_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<f64, ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
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

    /// Extra analytic-penalty energy that has no native `SaeManifoldLoss`
    /// component but is part of the penalized objective ranked by the SAE
    /// Laplace/REML criterion.
    pub fn reml_extra_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<f64, ArrowSchurError> {
        Ok(self.analytic_decoder_penalty_value_total(registry)?
            + self.isometry_penalty_value_total(registry)?)
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
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atoms but term has {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs();
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
                // Clamp the log-precision before exponentiating: a raw
                // `exp(log_ard)` overflows to `inf` for `log_ard ≳ 709`, and the
                // `inf` precision then poisons the ARD energy / curvature with
                // `inf · 0.0 = NaN` (#742, Issue 4).
                let alpha = SaeManifoldRho::stable_exp_strength(log_alpha);
                let period = periods[axis];
                let mut energy = 0.0;
                for row in 0..n {
                    let v = coord.row(row)[axis];
                    energy += ArdAxisPrior::eval(alpha, v, period).value;
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
                        acc += energy - 0.5 * (n as f64) * log_alpha;
                    }
                    Some(p) => {
                        let kappa = std::f64::consts::TAU / p;
                        let eta = alpha / (kappa * kappa);
                        // Overflow-free `log I0(η)`; `bessel_i0(η).ln()` would be
                        // `+inf` for `η ≳ 709` (#1113).
                        let log_i0 = bessel_i0_log_and_ratio(eta).0;
                        acc += energy + (n as f64) * (-eta + log_i0);
                    }
                }
            }
        }
        Ok(acc)
    }

    /// Assemble the enlarged `(logits, t)` row-local Arrow-Schur system.
    ///
    /// Full-batch entry point: a single chunk covering all rows, with the
    /// β-tier penalties (decoder smoothness, ARD, analytic β penalties) carrying
    /// their full strength. The streaming driver calls
    /// [`Self::assemble_arrow_schur_scaled`] directly with a `penalty_scale`
    /// equal to the minibatch fraction `n_chunk / N`, so that the sum of the
    /// per-chunk β-tier contributions over a full pass reconstructs exactly the
    /// single global β penalty (the smoothness/ARD/β terms are functions of `B`
    /// and the global coordinates, not of the chunk's rows).
    pub fn assemble_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_scaled(target, rho, analytic_penalties, 1.0)
    }

    /// Assemble the row-local Arrow-Schur system with a `penalty_scale` applied
    /// to the β-tier (decoder smoothness, ARD prior, analytic β penalties).
    ///
    /// `penalty_scale == 1.0` recovers the full-batch assembly. The streaming
    /// driver passes the minibatch fraction `n_chunk / N` so that the β-tier
    /// reduced-Schur and gradient contributions of the chunks sum to exactly one
    /// global copy across a full pass (data-fit, assignment-prior, and per-row
    /// coord/logit analytic terms are *not* scaled — they are genuine per-row
    /// sums).
    pub fn assemble_arrow_schur_scaled(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
            target,
            rho,
            analytic_penalties,
            penalty_scale,
            SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM,
        )
    }

    pub(crate) fn assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
        dense_beta_penalty_probe_max_dim: usize,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_inner(
            target,
            rho,
            analytic_penalties,
            penalty_scale,
            dense_beta_penalty_probe_max_dim,
            None,
        )
    }

    /// Innermost assembly entry. `forced_layout` overrides the budget-derived
    /// active-set layout so a caller can pin the dense (`Forced(None)`) or a
    /// specific compact (`Forced(Some(layout))`) path — used by the
    /// compact-vs-dense Riemannian-geometry equality regression test to drive
    /// both layouts on identical data. `Computed` is the production path:
    /// the layout is derived from the assignment mode + `sparse_active_plan`.
    pub(crate) fn assemble_arrow_schur_inner(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
        dense_beta_penalty_probe_max_dim: usize,
        forced_layout: ForcedRowLayout,
    ) -> Result<ArrowSchurSystem, String> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur_scaled: penalty_scale must be finite and positive; got {penalty_scale}"
            ));
        }
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: Z must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: log_ard length {} != K {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let ard_len = rho.log_ard[atom_idx].len();
            let d = coord.latent_dim();
            if ard_len != 0 && ard_len != d {
                return Err(format!(
                    "SaeManifoldTerm::assemble_arrow_schur: log_ard atom {atom_idx} \
                     has len {ard_len}; expected 0 (disabled) or atom dim {d}"
                ));
            }
        }
        // Reparameterize each atom's roughness Gram into arc length at the
        // current decoder/coordinates (issue #673). This is the single
        // chokepoint for both the inner Newton assembly and the undamped
        // evidence factorization, so freezing the pullback-metric weight here
        // (lagged-diffusivity) keeps the smoothness value, gradient, Kronecker
        // Hessian, and REML log-det mutually consistent within each assembly
        // and makes the converged penalty — hence the topology evidence —
        // gauge-invariant. Constant-speed (periodic) atoms are unaffected.
        for atom in &mut self.atoms {
            atom.refresh_intrinsic_smooth_penalty();
        }
        // #1026 — freeze the decoder-repulsion collinearity gate at the SAME
        // assembly chokepoint as the smoothness Gram, so the repulsion's
        // gradient/curvature (assembled below) and its value (read by the
        // line-search `penalized_objective_total`) share one frozen gate.
        self.refresh_decoder_repulsion_gate();
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        let frame_projection = FrameProjection::new(self);
        let beta_offsets = frame_projection.beta_offsets.clone();
        let coord_offsets = self.assignment.coord_offsets();
        // β-tier decoder smoothness is a global (B-only) penalty; under a
        // minibatch pass it is scaled by the chunk fraction so the per-chunk
        // contributions sum to one global copy.
        // Per-atom decoder-smoothness strengths (#1556): atom k's penalty `S_k`
        // is scaled by `λ_smooth[k]·penalty_scale`. The minibatch `penalty_scale`
        // multiplies every atom uniformly.
        let lambda_smooth: Vec<f64> = rho
            .lambda_smooth_vec()
            .iter()
            .map(|&l| l * penalty_scale)
            .collect();
        let (assignment_grad, assignment_hdiag) =
            assignment_prior_grad_hdiag(&self.assignment, rho)?;

        // #1038 softmax entropy: the exact per-row Hessian in logits is dense
        // (`H_kj = (λ/τ²) a_k[δ_kj(m−L_k−1)+a_j(L_k+L_j+1−2m)]`), not just the
        // `assignment_hdiag` diagonal. Build the shared penalty + `scale = λ/τ²`
        // once here so the dense row block written into `block.htt` below, the
        // criterion's `log|H|`, and the #1006 θ-adjoint all differentiate the
        // SAME operator. JumpReLU / IBP keep their (separately exact) diagonal /
        // cross-row channels and leave this `None`. The block is gauge-null in
        // isolation (`H·𝟙 = 0`); it is only ever summed onto the gauge-breaking
        // data-fit row block before the Cholesky factor, never factored alone.
        let softmax_dense: Option<(
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                Some((
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                    scale,
                ))
            }
            _ => None,
        };

        // Decoder smoothness penalty: build one KroneckerPenaltyOp per atom
        // (structure = λ·S_k ⊗ I_p, offset = beta_offsets[k]) instead of
        // materialising the dense K×K block.  The gradient is a dense K-vector
        // accumulated into `smooth_grad_gb` and written into sys.gb after sys
        // is constructed (#296).
        let mut smooth_ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(self.atoms.len());
        // #972 / #977 T1: retain each atom's symmetrised `λ S_k` (`M_k × M_k`) so
        // the frame transform can rebuild the smooth penalty in the factored
        // coordinate space as `λ S_k ⊗ I_{r_k}` (the `tr(C_kᵀ S_k C_k)` form,
        // using `U_kᵀU_k = I`). Unused — and not even read — on the full-`B`
        // path, so this is a zero-cost capture there.
        let mut smooth_scaled_s: Vec<Array2<f64>> = Vec::with_capacity(self.atoms.len());
        let mut smooth_grad_gb = vec![0.0_f64; beta_dim];
        // #1117 — rank deficiency is handled at the basis layer: any
        // rank-deficient atom was reparametrized onto its data-supported subspace
        // at fit entry (`reduce_atoms_to_data_supported_rank`), so the β-tier here
        // always sees a full-rank design and needs no step-time data-null
        // deflation operator. The well-conditioned (full-rank) path is unchanged.
        // Per-atom smoothness-gradient GEMMs `½(S_k+S_kᵀ)·B_k` are independent
        // across atoms; batch them across ALL GPUs (uniform-shape tiles) and
        // scale by `lambda_smooth` below. `symmetrize = true` reproduces the
        // per-atom symmetrised `scaled_s/λ` used by the Kronecker op. Exact CPU
        // fallback per atom keeps the result bit-for-bit with the all-CPU path.
        let sym_sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sym_sb_all = batched_smooth_sb(&sym_sb_inputs, true);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = beta_offsets[atom_idx];
            // Symmetrise and scale the smoothness penalty matrix.
            let mut scaled_s = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    let s_ij = 0.5 * (atom.smooth_penalty[[i, j]] + atom.smooth_penalty[[j, i]]);
                    scaled_s[[i, j]] = lambda_smooth[atom_idx] * s_ij;
                }
            }
            // Gradient: g[beta_i] += (λ_k S_k B_k)[i, out_col]. The (m×m)·(m×p)
            // GEMM `½(S+Sᵀ)·B_k` was computed in the multi-GPU batch above; here
            // we only apply atom k's `lambda_smooth[atom_idx]`.
            let sb = &sym_sb_all[atom_idx] * lambda_smooth[atom_idx];
            for out_col in 0..p {
                for i in 0..m {
                    let beta_i = off + i * p + out_col;
                    smooth_grad_gb[beta_i] += sb[[i, out_col]];
                }
            }
            // IdentityRightKroneckerPenaltyOp: factor_a = λ·S_k (m×m), factor_b = I_p.
            smooth_ops.push(Arc::new(IdentityRightKroneckerPenaltyOp {
                factor_a: scaled_s.clone(),
                p,
                global_offset: off,
                k: beta_dim,
            }));
            // Retain `λ S_k` for the factored rebuild (no-op cost on full-`B`).
            smooth_scaled_s.push(scaled_s);
        }

        // Per-row active-set layout. Engaged for two regimes:
        //   * JumpReLU — structural gate plus the smooth prior's
        //     machine-precision support: atoms with
        //     `(logit - threshold)/tau > -36` enter the compact solve
        //     ([`jumprelu_in_optimization_band`]). Strictly gated-off atoms
        //     (logit ≤ threshold) carry zero assignment mass so their data-fit
        //     reconstruction contribution and data-fit logit JVP are zero, but
        //     supported atoms keep value-consistent prior gradient in the row block.
        //   * IBP-MAP at large `K` — the dense `(m_total · p)²` data
        //     Gram is infeasible, so each row is truncated to its
        //     top-`k_active` atoms above a relative magnitude cutoff
        //     ([`Self::sparse_active_plan`]). Small-`K` problems return `None`
        //     and keep the exact full-support layout.
        // The compact row block is sized `q_active = |active| + Σ_{k∈active}
        // d_k` instead of the full `q`.
        let coord_dims: Vec<usize> = self
            .assignment
            .coords
            .iter()
            .map(|c| c.latent_dim())
            .collect();
        let row_layout: Option<SaeRowLayout> = match forced_layout {
            Some(layout) => layout,
            None => match self.assignment.mode {
                AssignmentMode::JumpReLU {
                    threshold,
                    temperature,
                } => Some(SaeRowLayout::from_jumprelu(
                    n,
                    k_atoms,
                    threshold,
                    temperature,
                    &self.assignment.logits,
                    coord_dims.clone(),
                    self.assignment.coord_offsets(),
                )),
                // #1408/#1409 — Softmax engages the COMPACT top-`k` row layout
                // inside the optimization (no longer a post-fit projection).
                // The active set is each row's top-`k_active_cap` softmax atoms
                // above the relative cutoff; the cap comes from the user's
                // `top_k` (`softmax_active_cap`) and/or the in-core memory budget
                // ([`Self::softmax_active_plan`]). The full-`K` softmax
                // normalization still forms `a` (the gate map); only the dropped
                // tail logits, carrying negligible `O(a)` reconstruction mass and
                // `O(a²)` curvature, leave the per-row block.
                //
                // Coherence (the load-bearing correctness invariant): the
                // assembly's softmax curvature branch writes the ACTIVE×ACTIVE
                // principal sub-block of the Gershgorin Loewner majorizer
                // `D = diag(Σ_j|H_kj|)` (#1419; PSD and `D ⪰ H_entropy`) on the
                // compact logit slots — NOT the indefinite `assignment_hdiag`
                // diagonal. The logdet ρ-trace
                // (`assignment_log_strength_hessian_trace`) iterates the row's
                // active logit slots and indexes that SAME majorizer by global
                // atom, and the θ-adjoint reads its derivative via `jets.vars`
                // (global-atom indexed), so value, log|H|, and Γ differentiate
                // ONE operator on the compact support. The FFI's after-the-fit
                // top-`k` projection is then a no-op at the optimum.
                AssignmentMode::Softmax { .. } => match self.softmax_active_plan() {
                    Some((k_active_cap, relative_cutoff)) => {
                        let mut assignments_all = Vec::with_capacity(n);
                        for row in 0..n {
                            assignments_all
                                .push(self.assignment.try_assignments_row_for_rho(row, rho)?);
                        }
                        Some(SaeRowLayout::from_dense_weights(
                            &assignments_all,
                            k_active_cap,
                            relative_cutoff,
                            coord_dims.clone(),
                            self.assignment.coord_offsets(),
                        ))
                    }
                    None => None,
                },
                AssignmentMode::IBPMap { .. } => {
                    match self.sparse_active_plan() {
                        Some((k_active_cap, relative_cutoff)) => {
                            // Build per-row dense assignments once to derive the
                            // active set; the row loop re-derives `assignments`
                            // (cheap gate map at the same rho) and reuses these
                            // active sets.
                            let mut assignments_all = Vec::with_capacity(n);
                            for row in 0..n {
                                assignments_all
                                    .push(self.assignment.try_assignments_row_for_rho(row, rho)?);
                            }
                            // #1414: pass the RELATIVE cutoff through;
                            // `from_dense_weights` applies it per row against that
                            // row's own peak `max_k |a_{n,k}|`, matching the
                            // documented `sparse_active_plan` contract. A single
                            // global threshold (relative_cutoff · whole-dataset
                            // peak) wrongly drops every atom of a uniformly-small
                            // row when another row peaks high.
                            Some(SaeRowLayout::from_dense_weights(
                                &assignments_all,
                                k_active_cap,
                                relative_cutoff,
                                coord_dims.clone(),
                                self.assignment.coord_offsets(),
                            ))
                        }
                        None => None,
                    }
                }
            },
        };
        // #974 likelihood-whitening seam. The single per-row decision: when the
        // installed `RowMetric` is a genuinely estimated noise model
        // (`whitens_likelihood()` — only `WhitenedStructured`), the
        // reconstruction data-fit, its t-block Gauss-Newton row block, AND the
        // β-tier data-fit gradient are all assembled through the SAME per-row
        // metric `M_n = U_n U_nᵀ = Σ_n^{-1}`. There is exactly ONE construction
        // site (the `whiten_rows` closure below), so the value the line-search
        // sums and the gradient/Hessian the Newton step solves cannot drift apart
        // (the objective↔gradient-desync cure). For Euclidean / OutputFisher /
        // no-metric the closure is the identity and every downstream loop is
        // byte-identical to the historical isotropic path.
        let whitens_likelihood = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        // #972 / #977 T1: engage the FACTORED Grassmann-coordinate β-tier when
        // any atom has an active decoder frame. The closed-form factorization
        // `Φᵀ(G ⊗ I_p)Φ = G ⊗ (U_iᵀU_j)` is EXACT only for the isotropic
        // likelihood; under an active whitening metric (`whitens_likelihood()`,
        // only `WhitenedStructured`) the per-row output factor would be
        // `U_iᵀ M_n U_j` and does NOT factor out of the basis Gram, so we fall
        // back to the full-`B` path there (frames + whitening is out of scope —
        // see #974). The common Euclidean / OutputFisher / no-metric case factors
        // cleanly. When `frames_engaged` is false, EVERY β-tier object below is
        // assembled bit-for-bit as the historical full-`B` path.
        let frames_engaged = self.any_frame_active() && !whitens_likelihood;
        // #1407: fixed-decoder mode skips the entire β decoder tier (G/gb/htbeta
        // operator/hbb/β-penalties); only per-row htt/gt are produced.
        let fixed_decoder = self.fixed_decoder_assembly;
        let admission_plan = self
            .streaming_plan()
            .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
            .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        // #1407: fixed-decoder builds NO dense β-Hessian (hbb) — force the
        // empty-hbb system constructor so no `beta_dim × beta_dim` workspace is
        // taken (the early return skips `reclaim_border_hbb_workspace`).
        let dense_beta_curvature = !fixed_decoder
            && admission_plan.direct_admitted
            && !(frames_engaged && beta_dim > dense_beta_penalty_probe_max_dim);
        // #1406: the dense per-row cross-block slab `block.htbeta` is only WRITTEN
        // (line ~4243) and READ by the solver when `frames_engaged` (the factored
        // full-B path, which installs NO matrix-free row operator → the solver's
        // `sys_htbeta_apply_row` falls back to the dense slab). On the
        // `!frames_engaged` path the cross block is carried entirely by the
        // matrix-free Kronecker operator (`set_row_htbeta_operator`, ~line 4491);
        // `activate_dense_htbeta_supplement` is never called, so the solver never
        // touches `block.htbeta`. Allocating it at `beta_dim = K·M·p` there is the
        // ~6 TiB high-K leak (#1405/#1406): allocate ZERO columns instead. Frames
        // still use the (much smaller) factored border width.
        let row_htbeta_dim = if frames_engaged && !fixed_decoder {
            self.factored_border_dim()
        } else {
            // #1406/#1407: matrix-free path and fixed-decoder mode hold no dense
            // cross-block slab (fixed-decoder skips the β tier entirely).
            0
        };
        // Build the Arrow-Schur system: heterogeneous row dims when a compact
        // layout is active, uniform `q` otherwise.
        let mut sys = if let Some(ref layout) = row_layout {
            let per_row_dims: Vec<usize> = (0..n).map(|row| layout.row_q_active(row)).collect();
            if dense_beta_curvature {
                let hbb_workspace = self.take_border_hbb_workspace(beta_dim);
                ArrowSchurSystem::new_with_per_row_dims_and_hbb_and_htbeta_cols(
                    per_row_dims,
                    beta_dim,
                    hbb_workspace,
                    row_htbeta_dim,
                )
            } else {
                self.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
                ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(
                    per_row_dims,
                    beta_dim,
                    row_htbeta_dim,
                )
            }
        } else if dense_beta_curvature {
            let hbb_workspace = self.take_border_hbb_workspace(beta_dim);
            ArrowSchurSystem::new_with_hbb_and_htbeta_cols(
                n,
                q,
                beta_dim,
                hbb_workspace,
                row_htbeta_dim,
            )
        } else {
            self.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
            ArrowSchurSystem::new_with_empty_hbb_and_htbeta_cols(n, q, beta_dim, row_htbeta_dim)
        };
        // Apply accumulated smoothness-penalty gradients into sys.gb.
        for (i, g) in smooth_grad_gb.iter().enumerate() {
            sys.gb[i] += g;
        }
        // `w_dim` is the whitened output dimension: `rank` of the metric factor
        // when whitening, else `p` (identity). `error_white` is the whitened
        // residual `U_nᵀ r_n ∈ ℝ^{w_dim}` whose squared norm is `r_nᵀ M_n r_n`,
        // shared by the value path, the t-block GN, and (lifted back to p-space)
        // the β-tier gradient.
        let w_dim = match self.row_metric.as_ref() {
            Some(metric) if whitens_likelihood => metric.metric_rank(),
            _ => p,
        };
        // Data-fit Gauss-Newton β-Hessian is block-diagonal across the `p`
        // output channels and identical in each: with the flat β layout
        // `β[μ·p + oc] = B[μ, oc]` (μ enumerating (atom, basis_col)) the GN
        // outer product `Jβᵀ Jβ` couples only equal `oc`, with the same
        // `(M_total × M_total)` block `G[μ, μ'] = Σ_rows (a_k φ_k[m])(a_{k'} φ_{k'}[m'])`
        // for every channel. So `H_data = G ⊗ I_p`. The `μ` index of an `a_phi`
        // entry whose global β base is `beta_base` is `beta_base / p` (every
        // `beta_offset` and the `basis_col·p` stride are multiples of `p`).
        //
        // `G` is only non-zero on `(atom_i, atom_j)` pairs that co-occur in
        // some row's active set, so we accumulate it as a sparse map of dense
        // per-atom-pair `(m_i × m_j)` blocks keyed by `(atom_i, atom_j)` rather
        // than as a dense `(m_total × m_total)` matrix. At `K = 100K` with
        // per-row active sets of size `k_active ≪ K`, only `O(N · k_active²)`
        // pairs are ever touched, so the data Gram (and every matvec /
        // diagonal pass over it via `SparseBlockKroneckerPenaltyOp`) tracks the
        // active atoms instead of `K²`. In the dense full-support layout the
        // map degenerates to every co-occurring pair, reproducing the dense
        // Gram exactly. A `BTreeMap` key order keeps the installed op's
        // fingerprint deterministic. The `μ`-space offset of atom `k` is
        // `beta_offsets[k] / p`.
        type SaeGBlocks = std::collections::BTreeMap<(usize, usize), Array2<f64>>;
        let m_total: usize = self.atoms.iter().map(|a| a.basis_size()).sum();
        let mu_offsets: Vec<usize> = beta_offsets.iter().map(|&off| off / p).collect();
        // Stick-breaking prior for IBP-MAP depends only on (k_atoms, alpha_eff)
        // which are constant across rows for the current rho; precompute once.
        let ibp_prior_vec = match self.assignment.mode {
            AssignmentMode::IBPMap { .. } => {
                let alpha = self
                    .assignment
                    .mode
                    .resolved_ibp_alpha(rho)
                    .ok_or_else(|| "IBP assignment alpha resolution failed".to_string())?;
                Some(ordered_geometric_shrinkage_prior(k_atoms, alpha).to_vec())
            }
            _ => None,
        };
        let ibp_prior_slice = ibp_prior_vec.as_deref();
        // #991 design honesty weights (mean-1 HT inclusion corrections); see
        // the seam comment at the per-row residual below.
        let row_loss_w = self.row_loss_weights.as_deref();
        // Dense full-support index `[0, k_atoms)`, used by the row loop when no
        // compact layout is engaged so the active-atom iteration is uniform.
        let all_atoms_index: Vec<usize> = (0..k_atoms).collect();
        // Per-atom per-axis periodicity, hoisted out of the row loop. Selects
        // the smooth von-Mises coordinate prior on wrapped (Circle) axes and
        // the Gaussian prior on Euclidean axes; see `ArdAxisPrior`.
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        struct SaeAssemblyRow {
            pub(crate) row: usize,
            pub(crate) block: ArrowRowBlock,
            pub(crate) gb_delta: Vec<(usize, f64)>,
            pub(crate) g_blocks: SaeGBlocks,
            pub(crate) kron_a_phi: Option<Vec<(usize, f64)>>,
            pub(crate) kron_jac: Option<Vec<f64>>,
        }

        // Per-row scratch reused across all rows a rayon worker processes
        // (#1017). The assembly closure is re-run every inner Newton iteration ×
        // every outer ρ evaluation; allocating these eight loop-invariant-sized
        // buffers (`k_atoms·p`, several `p`, one `q·max(w_dim,p)`) once per
        // worker via `map_init` — rather than once per (row × assembly) inside
        // the closure — removes the dominant small-allocation traffic the
        // eu-stack profile attributed to allocator/barrier spin at the SAE LLM
        // shape (p≈5120). Every buffer is fully filled (or `.fill(0.0)`'d) before
        // it is read each row, so reuse is bit-identical to the fresh-alloc path;
        // `gb_delta`/`g_blocks` are NOT scratch (they move into the returned
        // `SaeAssemblyRow`) and stay allocated per row.
        struct RowScratch {
            pub(crate) decoded: Array2<f64>,
            pub(crate) dg_buf: Vec<f64>,
            pub(crate) fitted: Array1<f64>,
            pub(crate) error: Array1<f64>,
            pub(crate) error_white: Vec<f64>,
            pub(crate) error_metric: Array1<f64>,
            pub(crate) jac_white: Vec<f64>,
            pub(crate) decoded_scratch: Vec<f64>,
            // #1557 — per-worker scratch for the row assignment vector (filled via
            // `_into`, not allocated per row); full `k_atoms`, global-atom indexed.
            pub(crate) assignments: Array1<f64>,
        }
        // #1410: size the per-worker scratch by the COMPACT row dimensions, not
        // full `K`/`q`. With a compact layout the assembly only ever touches each
        // row's active atoms (≤ `max_active`) and its compact tangent block
        // (≤ `max_q_row`); allocating `decoded` at `k_atoms·p` and `jac_white` at
        // `q·max(w_dim,p)` was the per-worker `O(K)` blow-up (≈11 GiB/worker at
        // K=100k, p=5120 — and `map_init` gives every Rayon worker its own copy).
        // Without a layout the dense path needs full `k_atoms`/`q`. `decoded` rows
        // are addressed by COMPACT SLOT in the compact branch below (the dense
        // branch keeps global-atom rows), so the row count is the max active set.
        //
        // #1410/#1408/#1409: SOFTMAX now ALSO takes the `Some(layout)` branch
        // whenever a `top_k` cap (`set_softmax_active_cap`) or an in-core memory
        // breach engages `softmax_active_plan` → `from_dense_weights`, so its
        // per-worker `decoded`/`jac_white` scratch is the COMPACT
        // `max_active`/`max_q_row` size too — no longer the full `(k_atoms·p)` /
        // `(q·max(w_dim,p))` blow-up. JumpReLU / IBP-MAP likewise pay only
        // `max_active`. The remaining `None` (full-`K`) branch is the UNCAPPED
        // softmax / no-budget-breach case, which genuinely assembles the dense
        // entropy block over all `K`; capping it (the compact contract) removes
        // the per-worker `O(K)` footprint entirely. (#1410: the residual per-row
        // `O(K)` softmax-majorizer scratch — a `row_logits` copy and the full-`K`
        // `d`/`H_entropy` blocks — is removed separately; see the active-only
        // `active_softmax_gershgorin_majorizer_entry` /
        // `softmax_dense_entropy_hessian_entry` helpers below.)
        let (decoded_rows, scratch_q) = match row_layout.as_ref() {
            Some(layout) => {
                let max_active = (0..n)
                    .map(|r| layout.active_atoms[r].len())
                    .max()
                    .unwrap_or(0)
                    .max(1);
                let max_q_row = (0..n)
                    .map(|r| layout.row_q_active(r))
                    .max()
                    .unwrap_or(q)
                    .max(1);
                (max_active, max_q_row)
            }
            None => (k_atoms, q),
        };
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        // #1033 large-n: fold the per-row assembly results in row-ordered CHUNKS
        // rather than collecting all `n` `SaeAssemblyRow`s at once. The previous
        // path materialized the FULL `Vec<SaeAssemblyRow>` (every row's htt/gt
        // block + per-row `g_blocks` + `kron_a_phi`/`kron_jac`) AND the fold
        // destinations simultaneously — a ~2× transient peak over the resident
        // system during the fold, the assembly-side OOM cliff at large `n`. By
        // collecting one chunk, folding it into `sys.rows`/`g_blocks`/`kron_*`,
        // and dropping the chunk's `Vec` before the next chunk, the transient
        // intermediate is bounded to `O(chunk_size)` while the resident output is
        // unchanged. The fold stays STRICTLY row-ascending (chunk `[c0..c1)` then
        // `[c1..c2)`, rows in order within each chunk), so every `+=` into
        // `sys.gb`, the `g_blocks` BTreeMap, and the `kron_*` pushes lands in the
        // identical order as the single-pass fold — bit-for-bit the same system.
        // Chunk width is the admission plan's `chunk_size` (the same value
        // `streaming_plan` sizes for the matrix-free window), floored so a tiny
        // plan still makes forward progress.
        let assembly_chunk_rows = self
            .assembly_chunk_override
            .unwrap_or(admission_plan.chunk_size)
            .clamp(1, n.max(1));
        let mut g_blocks: SaeGBlocks = std::collections::BTreeMap::new();
        let mut kron_a_phi: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut kron_jac: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let chunk_end = (chunk_start + assembly_chunk_rows).min(n);
            let mut fold_offset_in_chunk = 0usize;
            let row_results: Vec<SaeAssemblyRow> = (chunk_start..chunk_end)
                .into_par_iter()
                .map_init(
                    || RowScratch {
                        decoded: Array2::<f64>::zeros((decoded_rows, p)),
                        dg_buf: vec![0.0_f64; p],
                        fitted: Array1::<f64>::zeros(p),
                        error: Array1::<f64>::zeros(p),
                        error_white: vec![0.0_f64; w_dim],
                        error_metric: Array1::<f64>::zeros(p),
                        jac_white: vec![0.0_f64; scratch_q * w_dim.max(p)],
                        decoded_scratch: vec![0.0_f64; p],
                        assignments: Array1::<f64>::zeros(k_atoms),
                    },
                    |scratch, row| -> Result<SaeAssemblyRow, String> {
                        // #1557 — mark this rayon row worker as a nested data-parallel
                        // region so any faer GEMM reached transitively from the per-row
                        // assembly (frame `Uᵀ` products, the per-row cross-block /
                        // Schur-accumulation matmuls, the Riemannian projections) pins to
                        // `Par::Seq` via `effective_global_parallelism` instead of
                        // re-fanning the global Rayon pool against this outer fan-out
                        // (the `spindle` barrier-spin). Serial vs parallel over these tiny
                        // per-row blocks is a single small product, so the result is
                        // bit-identical. The guard is held for the whole closure body
                        // including its `?`/`return` paths.
                        with_nested_parallel(|| {
                        let RowScratch {
                            decoded,
                            dg_buf,
                            fitted,
                            error,
                            error_white,
                            error_metric,
                            jac_white,
                            decoded_scratch,
                            assignments,
                        } = scratch;
                        let mut gb_delta: Vec<(usize, f64)> = Vec::new();
                        let mut g_blocks: SaeGBlocks = std::collections::BTreeMap::new();
                        // #1557 — fill per-worker scratch (bit-identical to alloc path).
                        let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
                        self.assignment
                            .try_assignments_row_for_rho_into(row, rho, a_scratch)?;
                        // Reconstruction uses the row's active support: for the dense
                        // full-support layout this is all atoms (exact); for a compact
                        // layout the dropped atoms carry negligible `O(a)` reconstruction
                        // mass and zero curvature, so excluding them keeps `fitted`,
                        // `error`, and the logit-JVP cross term `(decoded[k] − fitted)`
                        // mutually consistent with the curvature actually assembled.
                        fitted.fill(0.0);
                        let row_active_owned: Option<&[usize]> =
                            row_layout.as_ref().map(|l| l.active_atoms[row].as_slice());
                        match row_active_owned {
                            Some(active) => {
                                // #1410: `decoded` is a compact (max_active × p) buffer
                                // here; index it by the active-set SLOT `j` (the same
                                // index the compact tangent block / `coord_starts` use),
                                // NOT the global `atom_idx`.
                                for (j, &atom_idx) in active.iter().enumerate() {
                                    let a_k = assignments[atom_idx];
                                    self.atoms[atom_idx]
                                        .fill_decoded_row(row, decoded_scratch.as_mut_slice());
                                    for out_col in 0..p {
                                        decoded[[j, out_col]] = decoded_scratch[out_col];
                                        fitted[out_col] += a_k * decoded_scratch[out_col];
                                    }
                                }
                            }
                            None => {
                                for atom_idx in 0..k_atoms {
                                    let a_k = assignments[atom_idx];
                                    self.atoms[atom_idx]
                                        .fill_decoded_row(row, decoded_scratch.as_mut_slice());
                                    for out_col in 0..p {
                                        decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
                                        fitted[out_col] += a_k * decoded_scratch[out_col];
                                    }
                                }
                            }
                        }
                        for out_col in 0..p {
                            error[out_col] = fitted[out_col] - target[[row, out_col]];
                        }
                        // #991 design-honesty seam: a per-row scalar weight `w_row` on the
                        // reconstruction channel is exactly the metric `w_row · I_p`, so it
                        // is realized as a `√w_row` scaling of the THREE row-local data
                        // quantities at their construction sites — this residual, the
                        // latent Jacobian (below), and the β basis load `a·φ` (below).
                        // Every downstream data object then carries exactly one factor of
                        // `w_row` (gt, htt, htbeta, the β Gram `G`, and the β gradient),
                        // matching the `w_row`-weighted value `loss_scaled` sums; the
                        // per-row latent priors (assignment / ARD, added to `gt`/`htt`
                        // further down) are deliberately unweighted — see the
                        // `row_loss_weights` field docs. `None` ⇒ `sqrt_row_w == 1.0` and
                        // no multiply is applied (bit-identical unweighted path).
                        let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
                        if sqrt_row_w != 1.0 {
                            for out_col in 0..p {
                                error[out_col] *= sqrt_row_w;
                            }
                        }
                        // #974 seam (step 1/2): whiten the per-row residual ONCE.
                        //   * not whitening ⇒ `error_white == error` (length p) and
                        //     `error_metric == error`; every downstream loop is the
                        //     historical isotropic path bit-for-bit.
                        //   * whitening ⇒ `error_white = U_nᵀ r_n ∈ ℝ^{w_dim}` (its squared
                        //     norm is `r_nᵀ M_n r_n`, the value the data-fit sums) and
                        //     `error_metric = U_n (U_nᵀ r_n) = M_n r_n ∈ ℝ^p` (the p-space
                        //     metric-applied residual the β-tier gradient contracts).
                        match self.row_metric.as_ref() {
                            Some(metric) if whitens_likelihood => {
                                let wr = metric.whiten_residual_row(row, error.view());
                                for (slot, &v) in error_white.iter_mut().zip(wr.iter()) {
                                    *slot = v;
                                }
                                let mr = metric.apply_metric_row(row, error.view());
                                for (slot, &v) in error_metric.iter_mut().zip(mr.iter()) {
                                    *slot = v;
                                }
                            }
                            _ => {
                                for out_col in 0..p {
                                    error_white[out_col] = error[out_col];
                                    error_metric[out_col] = error[out_col];
                                }
                            }
                        }

                        // Determine whether this row uses the compact active-set layout.
                        //   * JumpReLU: gated atoms plus the smooth prior's
                        //     machine-precision support enter.
                        //   * IBP-MAP at large K: only the top-`k_active` atoms.
                        //   * Otherwise (small K): the dense uniform-q layout.
                        let (q_row, mut local_jac_row) = if let Some(layout) = row_layout.as_ref() {
                            let active = &layout.active_atoms[row];
                            let starts = &layout.coord_starts[row];
                            let q_active = layout.row_q_active(row);
                            let mut jac_compact = Array2::<f64>::zeros((q_active, p));
                            // Logit JVP rows for active atoms only, using the per-mode
                            // assignment sensitivity `da_k/dl_k` contracted into the
                            // decoded / fitted-corrected output direction.
                            let logits_row = self.assignment.logits.row(row);
                            for (j, &k) in active.iter().enumerate() {
                                fill_active_atom_logit_jvp(
                                    ActiveAtomLogitJvp {
                                        mode: self.assignment.mode,
                                        k,
                                        logit_k: logits_row[k],
                                        a_k: assignments[k],
                                        // #1410: compact slot `j`, not global atom `k`.
                                        decoded_k: decoded.row(j),
                                        fitted: fitted.view(),
                                        ibp_prior: ibp_prior_slice,
                                        compact_index: j,
                                        // #1026/#1033: a FIXED logit (ungated, or every
                                        // atom under frozen routing) has a constant gate
                                        // ⇒ zero logit-JVP.
                                        ungated: self.assignment.logit_is_fixed(k),
                                    },
                                    &mut jac_compact,
                                );
                            }
                            // Coordinate JVP rows for active atoms only.
                            for (j, &k) in active.iter().enumerate() {
                                let d = self.atoms[k].latent_dim;
                                let a_k = assignments[k];
                                let coord_start = starts[j];
                                for axis in 0..d {
                                    self.atoms[k].fill_decoded_derivative_row(
                                        row,
                                        axis,
                                        dg_buf.as_mut_slice(),
                                    );
                                    for out_col in 0..p {
                                        jac_compact[[coord_start + axis, out_col]] =
                                            a_k * dg_buf[out_col];
                                    }
                                }
                            }
                            (q_active, jac_compact)
                        } else {
                            // Fresh per-row Jacobian, structurally identical to the
                            // JumpReLU branch: every (q × p) element is unconditionally
                            // overwritten below (assignment-chart JVP rows + coordinate rows), so the
                            // `Array2::zeros` allocation needs no separate `fill(0.0)` and
                            // the populated buffer is returned by move without a clone.
                            let mut jac_row = Array2::<f64>::zeros((q, p));
                            fill_assignment_logit_jvp_rows(
                                self.assignment.mode,
                                self.assignment.logits.row(row),
                                assignments.view(),
                                decoded.view(),
                                fitted.view(),
                                ibp_prior_slice,
                                // #1026/#1033: zero logit-JVP rows for FIXED-logit atoms
                                // (ungated, and all atoms under frozen routing).
                                &self.assignment.fixed_logit_mask(),
                                &mut jac_row,
                            );
                            // Coordinate columns for all atoms.
                            for atom_idx in 0..k_atoms {
                                let d = self.atoms[atom_idx].latent_dim;
                                let off = coord_offsets[atom_idx];
                                let a_k = assignments[atom_idx];
                                for axis in 0..d {
                                    self.atoms[atom_idx].fill_decoded_derivative_row(
                                        row,
                                        axis,
                                        dg_buf.as_mut_slice(),
                                    );
                                    for out_col in 0..p {
                                        jac_row[[off + axis, out_col]] = a_k * dg_buf[out_col];
                                    }
                                }
                            }
                            (q, jac_row)
                        };

                        // #991 design-honesty seam, Jacobian leg: scale the row's latent
                        // Jacobian by `√w_row` BEFORE the whitening / Kronecker capture so
                        // htt (= J̃J̃ᵀ), the data part of gt (= J̃ẽ, the residual already
                        // carries its own √w_row), and the htbeta cross block (J paired
                        // with the √w_row-scaled β load below) each carry exactly one
                        // factor of `w_row`. No-op on the unweighted path.
                        if sqrt_row_w != 1.0 {
                            for a in 0..q_row {
                                for out_col in 0..p {
                                    local_jac_row[[a, out_col]] *= sqrt_row_w;
                                }
                            }
                        }

                        // #974 seam (step 2/2): whiten the per-row Jacobian through the SAME
                        // metric the residual was whitened by. `jac_white[a*w_dim + k]` holds
                        // `J̃[a, k] = Σ_out U_n[out, k] · J_n[a, out]` so the t-block
                        // Gauss-Newton row block is `htt = J̃ J̃ᵀ = J_n M_n J_nᵀ` and
                        // `gt = J̃ ẽ = J_nᵀ M_n r_n`. When not whitening, `w_dim == p` and the
                        // whitened jac equals the raw Jacobian, so htt/gt are byte-identical
                        // to the historical isotropic assembly. Because the SAME `error_white`
                        // feeds both the value-path data-fit (Σ½ ẽ²) and this gradient
                        // (J̃ ẽ), the objective and its t-block gradient share one whitening
                        // — they cannot desync.
                        if whitens_likelihood {
                            if let Some(metric) = self.row_metric.as_ref() {
                                for a in 0..q_row {
                                    for k in 0..w_dim {
                                        let mut acc = 0.0;
                                        // U_n[out, k] read through the metric's factor layout.
                                        for out_col in 0..p {
                                            acc += metric.factor_entry(row, out_col, k)
                                                * local_jac_row[[a, out_col]];
                                        }
                                        jac_white[a * w_dim + k] = acc;
                                    }
                                }
                            }
                        } else {
                            for a in 0..q_row {
                                for out_col in 0..p {
                                    jac_white[a * w_dim + out_col] = local_jac_row[[a, out_col]];
                                }
                            }
                        }

                        // Build the per-row Arrow-Schur block at the row's active dim.
                        let mut block = ArrowRowBlock::new(q_row, row_htbeta_dim);
                        for a in 0..q_row {
                            let jac_a = &jac_white[a * w_dim..(a + 1) * w_dim];
                            let g = jac_a
                                .iter()
                                .zip(error_white.iter())
                                .map(|(&j, &e)| j * e)
                                .sum::<f64>();
                            block.gt[a] += g;
                            for b in 0..q_row {
                                let jac_b = &jac_white[b * w_dim..(b + 1) * w_dim];
                                let h = jac_a
                                    .iter()
                                    .zip(jac_b.iter())
                                    .map(|(&ja, &jb)| ja * jb)
                                    .sum::<f64>();
                                block.htt[[a, b]] += h;
                            }
                        }

                        // Assignment prior in logit space.
                        // For compact layout: position `j` = active_atoms index.
                        // For dense layout: position `atom_idx` directly.
                        //
                        // H-consistency note (#1006 audit). This `assignment_hdiag` is the
                        // assignment channel's raw diagonal curvature, added un-majorized. It
                        // is exact for JumpReLU and exact within each IBP row/column diagonal,
                        // but it is a deliberate diagonal approximation for two full-Hessian
                        // structures that the current factorization does not yet carry (#1038):
                        //
                        //   * softmax entropy has dense within-row Hessian
                        //     H_kj = (λ/τ²) a_k[δ_kj(m-L_k-1) + a_j(L_k+L_j+1-2m)];
                        //     this block stores only its diagonal.
                        //   * IBP empirical-π has cross-row rank-one terms per column
                        //     H_(i,k),(j,k) = w score_derivative_k z'_ik z'_jk for i != j;
                        //     this row-local block stores only the diagonal/self-row part.
                        //     The exact scalar `D`-coefficient `d_k = w·s'_k` is now
                        //     surfaced as `IbpHessianDiagThirdChannels::cross_row_d`
                        //     (FD-verified against ∂²value/∂ℓ_ik∂ℓ_jk in
                        //     `ibp_cross_row_woodbury_d_matches_full_off_diagonal_hessian`),
                        //     and `z_jac` carries `u_k`'s entries `z'_ik`. The exact
                        //     determinant-lemma consumer is
                        //     log det(I_K + D UᵀH₀'⁻¹U) on the NO-SELF base
                        //     H₀' = H₀ − Σ_k d_k diag(z'_ik²) — which requires re-factoring
                        //     the per-row logit-slot diagonal (a factorization-side change
                        //     in `solver::arrow_schur`, outside this assembly chokepoint).
                        //
                        // The criterion's log|H| and Γ adjoint differentiate this same
                        // assembled diagonal/quasi-Laplace Hessian, so value and gradient stay
                        // on one branch. A future dense-row softmax or IBP Woodbury correction
                        // must update both assembly and the θ-adjoint together.
                        let assignment_base = row * k_atoms;
                        if let Some(layout) = row_layout.as_ref() {
                            let active = &layout.active_atoms[row];
                            // #1408/#1409 softmax compact curvature: the entropy
                            // Hessian diagonal in `assignment_hdiag` is INDEFINITE,
                            // so on a compact softmax layout write the Gershgorin
                            // Loewner majorizer `D_kk = Σ_j|H_kj|` (#1419) — the same
                            // PSD operator the dense softmax branch writes — at each
                            // active logit slot. `D` is diagonal, so its active
                            // principal sub-block is `diag(D_kk : k ∈ active)`; each
                            // `D_kk` is the FULL-`K` abs-row-sum, so it still
                            // dominates the active principal sub-block of `H_entropy`
                            // (a genuine majorizer on the retained support). The
                            // gradient stays the EXACT entropy gradient (it sets the
                            // fixed point), so majorizing only conditions the Newton
                            // step. JumpReLU/IBP keep their (exact) diagonal.
                            //
                            // #1410: compute only the active `D_kk` directly from this
                            // row's softmax assignments `a` (= `assignments`, already
                            // in hand), via `active_softmax_gershgorin_majorizer_entry`.
                            // The previous `psd_majorizer_abs_row_sums(&row_logits, ..)`
                            // call allocated TWO length-`K` per-row scratch vectors (a
                            // fresh `row_logits` copy and the full-`K` returned `d`)
                            // only to read `d[k]` for the `≤ top_k` active `k` — an
                            // `O(K)` per-row allocation on the path the compact
                            // contract keeps `K`-free. The shared `m = Σ_j a_j l_j` is
                            // the one irreducible `O(K)` pass, computed once per row.
                            let assignments_slice = assignments
                                .as_slice()
                                .expect("softmax assignments row must be contiguous");
                            let majorizer_log_mean: Option<f64> = softmax_dense
                                .as_ref()
                                .map(|_| softmax_majorizer_log_mean(assignments_slice));
                            for (j, &k) in active.iter().enumerate() {
                                block.gt[j] += assignment_grad[assignment_base + k];
                                match (softmax_dense.as_ref(), majorizer_log_mean) {
                                    (Some((_penalty, scale)), Some(m)) => {
                                        block.htt[[j, j]] +=
                                            active_softmax_gershgorin_majorizer_entry(
                                                assignments_slice,
                                                k,
                                                m,
                                                *scale,
                                            );
                                    }
                                    _ => block.htt[[j, j]] += assignment_hdiag[assignment_base + k],
                                }
                            }
                        } else {
                            for free_idx in 0..assignment_dim {
                                block.gt[free_idx] += assignment_grad[assignment_base + free_idx];
                            }
                            if let Some((penalty, scale)) = softmax_dense.as_ref() {
                                // #1419: write the genuine Gershgorin Loewner majorizer
                                // `D = diag(Σ_j|H_kj|)` of the exact entropy Hessian onto the
                                // row's logit block in place of the EXACT entropy Hessian. The
                                // entropy Hessian is INDEFINITE (concave directions on
                                // long-tailed rows), which drove the per-row evidence block
                                // non-PD and forced the downstream Faddeev–Popov deflation to
                                // flatten data-relevant logit directions (under-identifying the
                                // atoms). `D` is a nonnegative diagonal, hence exactly PSD and
                                // PD-preserving like the previous Fisher surrogate, so the block
                                // stays PD and the deflation no longer fires on the entropy
                                // block. Unlike the Fisher metric `G = scale·(diag(a) − a aᵀ)`,
                                // which is PSD but NOT a majorizer (`G − H_entropy` can be
                                // indefinite — K=2, a=(0.95,0.05): G₁₁=0.0475 < H₁₁=0.0784,
                                // #1419), `D` actually satisfies `D ⪰ H_entropy` and `D ⪰ 0`,
                                // so it is a true MM/Loewner curvature majorizer. Because the
                                // entropy penalty is a FIXED prior whose stationary point is set
                                // by its (unchanged) EXACT gradient, replacing its curvature
                                // with the majorizer only conditions the Newton step and the
                                // Laplace normalizer's curvature operator — it does NOT move the
                                // optimum.
                                //
                                // Softmax uses the REDUCED K−1 free-logit chart (the last
                                // reference logit is fixed at 0, `assignment_coord_dim() = K−1`).
                                // Holding z_{K-1} fixed, the reduced curvature over the free
                                // logits 0..K−1 is exactly the top-left (K−1)×(K−1) submatrix of
                                // the full K×K majorizer (the fixed logit contributes no
                                // row/column to the free curvature). The criterion's `log|H|`
                                // and the #1006 θ-adjoint differentiate this SAME `D` (see the
                                // `row_psd_majorizer_logit_derivative` site below), so value and
                                // adjoint stay on one exact branch.
                                let row_logits: Vec<f64> = (0..k_atoms)
                                    .map(|k| self.assignment.logits[[row, k]])
                                    .collect();
                                let h_dense = penalty.row_psd_majorizer(&row_logits, *scale);
                                for ki in 0..assignment_dim {
                                    for kj in 0..assignment_dim {
                                        block.htt[[ki, kj]] += h_dense[[ki, kj]];
                                    }
                                }
                            } else {
                                for free_idx in 0..assignment_dim {
                                    block.htt[[free_idx, free_idx]] +=
                                        assignment_hdiag[assignment_base + free_idx];
                                }
                            }
                        }

                        // ARD on each on-atom coordinate.
                        // For compact layout: only active atoms; coord positions use compact starts.
                        // For dense layout: all atoms; coord positions use coord_offsets.
                        if let Some(layout) = row_layout.as_ref() {
                            let active = &layout.active_atoms[row];
                            let starts = &layout.coord_starts[row];
                            for (j, &k) in active.iter().enumerate() {
                                let coord = &self.assignment.coords[k];
                                let d = coord.latent_dim();
                                if rho.log_ard[k].is_empty() {
                                    continue;
                                }
                                if rho.log_ard[k].len() != d {
                                    return Err(format!(
                                        "ARD rho atom {k} has len {} but atom dim is {d}",
                                        rho.log_ard[k].len()
                                    ));
                                }
                                let row_t = coord.row(row);
                                let periods = &ard_axis_periods[k];
                                for axis in 0..d {
                                    // ARD on coords is a genuine per-row prior (each row
                                    // contributes the per-axis prior energy), so it is NOT
                                    // minibatch-scaled — the per-chunk row sums already
                                    // reconstruct the full coordinate prior across a pass.
                                    // The value (`ard_value`/`loss.ard`) and the gradient
                                    // both come from the SAME `ArdAxisPrior` energy, so they
                                    // stay FD-consistent on periodic axes. The exact
                                    // von-Mises curvature `V'' = α·cos(κt)` is INDEFINITE —
                                    // it goes negative for |t| past a quarter period — so
                                    // writing it raw into the Newton/Schur `htt` diagonal
                                    // makes that PSD curvature block indefinite and the Schur
                                    // Cholesky (used both for the Newton step and the exact
                                    // log-det) fails on a non-PD pivot. Accumulate the PSD
                                    // majorizer `max(V'', 0)` instead, exactly as
                                    // `add_sae_coord_penalty` does for the registry coord
                                    // penalties: the positive part keeps `htt` PSD so the
                                    // factorization succeeds, and majorizing the curvature of
                                    // a fixed prior only damps the Newton step — it does not
                                    // move the stationary point (the gradient, which sets the
                                    // fixed point, stays the exact `V'`).
                                    let alpha =
                                        SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                                    let prior =
                                        ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                                    block.gt[starts[j] + axis] += prior.grad;
                                    block.htt[[starts[j] + axis, starts[j] + axis]] +=
                                        prior.hess.max(0.0);
                                }
                            }
                        } else {
                            for atom_idx in 0..k_atoms {
                                let coord = &self.assignment.coords[atom_idx];
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
                                let off = coord_offsets[atom_idx];
                                let row_t = coord.row(row);
                                let periods = &ard_axis_periods[atom_idx];
                                for axis in 0..d {
                                    // PSD-majorize the (possibly negative) von-Mises curvature
                                    // into the Newton/Schur `htt` block; see the compact-layout
                                    // branch above for why `max(V'', 0)` is required to keep
                                    // `htt` PD (the exact `V'' = α·cos κt` is indefinite past a
                                    // quarter period and breaks the Schur/log-det Cholesky).
                                    let alpha = SaeManifoldRho::stable_exp_strength(
                                        rho.log_ard[atom_idx][axis],
                                    );
                                    let prior =
                                        ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                                    block.gt[off + axis] += prior.grad;
                                    block.htt[[off + axis, off + axis]] += prior.hess.max(0.0);
                                }
                            }
                        }

                        // Beta gradient/Hessian — Kronecker form J_β = φᵀ ⊗ I_p.
                        //
                        // The per-row beta Jacobian is
                        //   J_β[out_col, beta_idx] = a_k · phi_k[basis_col]   if out_col == out_col(beta_idx)
                        //                            0                         otherwise
                        // so the data-fit Gauss-Newton beta-Hessian factors as a rank-`p`
                        // sum of outer products. We pre-compute the per-(atom, basis_col)
                        // scalar `a_k · phi_k` once and reuse it across the `out_col`
                        // and inner `(atom_j, basis_col2)` loops.
                        //
                        // Full-B rows keep the matrix-free Kronecker path below. Factored
                        // rows write the `q_i × Σ M_k r_k` C-space cross slab directly by
                        // folding each output-channel contribution through the atom frame,
                        // so no `q_i × β_dim` slab is ever materialized.
                        //
                        // Only the row's active atoms contribute `a_phi` support and data
                        // curvature: in a compact layout (JumpReLU gate or large-K
                        // top-`k_active` truncation) the inactive atoms carry zero (gated)
                        // or sub-cutoff assignment mass and are excluded — this is what
                        // keeps both the htbeta support and the `G` accumulation
                        // `O(k_active)` rather than `O(K)`. In the dense full-support
                        // layout `row_active` spans all atoms.
                        let row_active: &[usize] = match row_layout.as_ref() {
                            Some(layout) => layout.active_atoms[row].as_slice(),
                            None => &all_atoms_index,
                        };
                        // #1407: in fixed-decoder mode the β tier is not assembled at
                        // all — leave gb_delta/g_blocks empty and kron None. htt/gt
                        // (built above) are the only outputs the frozen-decoder step
                        // consumes.
                        let mut a_phi: Vec<(usize, f64)> = Vec::with_capacity(row_active.len() * 4);
                        // Per-active-atom weighted basis row `a_k · φ_k[·]`, retained so the
                        // data Gram blocks can be accumulated as clean per-atom-pair outer
                        // products `(a_k φ_k) (a_{k'} φ_{k'})ᵀ`.
                        let mut weighted_phi: Vec<(usize, Vec<f64>)> =
                            Vec::with_capacity(row_active.len());
                        if !fixed_decoder {
                            for &atom_idx in row_active {
                                let atom = &self.atoms[atom_idx];
                                let atom_beta_off = beta_offsets[atom_idx];
                                let m = atom.basis_size();
                                let a_k = assignments[atom_idx];
                                let mut wphi = Vec::with_capacity(m);
                                for basis_col in 0..m {
                                    let phi = atom.basis_values[[row, basis_col]];
                                    // #991 design-honesty seam, β leg: the `√w_row` here pairs
                                    // with the `√w_row` on the residual (β gradient =
                                    // `a·φ · M r` ⇒ w_row) and with itself (β Gram `G` and the
                                    // htbeta Kronecker capture ⇒ w_row). `1.0` when unweighted.
                                    let w = a_k * phi * sqrt_row_w;
                                    a_phi.push((atom_beta_off + basis_col * p, w));
                                    wphi.push(w);
                                }
                                weighted_phi.push((atom_idx, wphi));
                            }
                            // β data-fit gradient `gᵦ += J_βᵀ M_n r_n`. The β-Jacobian is
                            // `J_β = φ_nᵀ ⊗ I_p`, so `J_βᵀ M_n r_n = φ_n ⊗ (M_n r_n)` —
                            // contract the basis weight `a·φ` against the p-space metric-applied
                            // residual `error_metric` (= `M_n r_n`), the SAME whitening the value
                            // path and t-block share. When not whitening, `error_metric == error`
                            // and this is byte-identical to the historical `J_βᵀ r`.
                            for &(beta_base_i, j_beta_i) in a_phi.iter() {
                                if j_beta_i == 0.0 {
                                    continue;
                                }
                                for out_col in 0..p {
                                    gb_delta.push((
                                        beta_base_i + out_col,
                                        j_beta_i * error_metric[out_col],
                                    ));
                                    // No dense hbb write — the sparse `G ⊗ I_p` op installed
                                    // after the loop carries the data-fit GN β-Hessian.
                                }
                            }
                            if frames_engaged {
                                for &atom_idx in row_active {
                                    let atom = &self.atoms[atom_idx];
                                    let m = atom.basis_size();
                                    let a_k = assignments[atom_idx];
                                    for basis_col in 0..m {
                                        let phi = atom.basis_values[[row, basis_col]];
                                        let w = a_k * phi * sqrt_row_w;
                                        if w == 0.0 {
                                            continue;
                                        }
                                        let c_base = frame_projection.border_offsets[atom_idx]
                                            + basis_col * frame_projection.ranks[atom_idx];
                                        for c in 0..q_row {
                                            let mut hrow = block.htbeta.row_mut(c);
                                            let hrow_slice = hrow
                                                .as_slice_mut()
                                                .expect("htbeta row is contiguous");
                                            for out_col in 0..p {
                                                let value = local_jac_row[[c, out_col]] * w;
                                                frame_projection.accumulate_output_project(
                                                    atom_idx, c_base, out_col, value, hrow_slice,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            // Data-fit GN β-Hessian: accumulate the channel-independent block
                            // `G[μ_i, μ_j] += (a_k φ_k)[μ_i] (a_{k'} φ_{k'})[μ_j]` into the
                            // sparse per-atom-pair map (the `out_col` dimension is carried by
                            // `I_p`). Only co-occurring `(atom_i, atom_j)` pairs are touched.
                            for ai in 0..weighted_phi.len() {
                                let (atom_i, ref wphi_i) = weighted_phi[ai];
                                let m_i = wphi_i.len();
                                for aj in 0..weighted_phi.len() {
                                    let (atom_j, ref wphi_j) = weighted_phi[aj];
                                    let m_j = wphi_j.len();
                                    let blk = g_blocks
                                        .entry((atom_i, atom_j))
                                        .or_insert_with(|| Array2::<f64>::zeros((m_i, m_j)));
                                    for li in 0..m_i {
                                        let wi = wphi_i[li];
                                        if wi == 0.0 {
                                            continue;
                                        }
                                        for lj in 0..m_j {
                                            blk[[li, lj]] += wi * wphi_j[lj];
                                        }
                                    }
                                }
                            }
                        } // #1407 end `if !fixed_decoder` β-tier accumulation
                        let (kron_a_phi, kron_jac) = if !frames_engaged && !fixed_decoder {
                            // Flatten local_jac_row row-major into a plain Vec<f64> (q_row * p entries).
                            let mut jac_flat = vec![0.0_f64; q_row * p];
                            for c in 0..q_row {
                                for j in 0..p {
                                    jac_flat[c * p + j] = local_jac_row[[c, j]];
                                }
                            }
                            (Some(a_phi), Some(jac_flat))
                        } else {
                            (None, None)
                        };
                        Ok(SaeAssemblyRow {
                            row,
                            block,
                            gb_delta,
                            g_blocks,
                            kron_a_phi,
                            kron_jac,
                        })
                        }) // #1557 with_nested_parallel
                    },
                )
                .collect::<Result<Vec<_>, String>>()?;

            // Fold THIS chunk's rows (ascending) into the global accumulators.
            // The parallel collect preserves index order within the chunk and
            // chunks are visited in ascending `chunk_start` order, so the overall
            // fold order is `0,1,2,…,n-1` — identical to the former single-pass
            // fold. The `row == chunk_start + fold_offset_in_chunk` assert pins
            // that strict sequential arrival (the invariant the `kron_*`
            // row-aligned pushes depend on).
            for row_result in row_results.into_iter() {
                let row = row_result.row;
                assert_eq!(
                    row,
                    chunk_start + fold_offset_in_chunk,
                    "parallel SAE row assembly returned rows out of order"
                );
                fold_offset_in_chunk += 1;
                for (idx, value) in row_result.gb_delta {
                    sys.gb[idx] += value;
                }
                for ((atom_i, atom_j), data) in row_result.g_blocks {
                    let m_i = data.nrows();
                    let m_j = data.ncols();
                    let blk = g_blocks
                        .entry((atom_i, atom_j))
                        .or_insert_with(|| Array2::<f64>::zeros((m_i, m_j)));
                    for li in 0..m_i {
                        for lj in 0..m_j {
                            blk[[li, lj]] += data[[li, lj]];
                        }
                    }
                }
                if !frames_engaged && !fixed_decoder {
                    // Rows arrive in ascending order across chunks, so pushing
                    // here yields `kron_*[row]` aligned to the row index exactly
                    // as the single-pass `push` did.
                    kron_a_phi.push(
                        row_result
                            .kron_a_phi
                            .expect("full-B SAE row assembly must return a_phi rows"),
                    );
                    kron_jac.push(
                        row_result
                            .kron_jac
                            .expect("full-B SAE row assembly must return local Jacobian rows"),
                    );
                }
                sys.rows[row] = row_result.block;
            }
            chunk_start = chunk_end;
        }
        // #1407: fixed-decoder early return. The per-row htt/gt are now fully
        // assembled (data GN + assignment/ARD prior). Apply only the htt/gt
        // Riemannian projection (the decoder/β tier is intentionally absent), then
        // return the block-diagonal system. `fixed_decoder_step_from_rows` reads
        // only `rows[*].htt`/`gt` + `row_offsets`, so no β-tier object is needed.
        if fixed_decoder {
            match row_layout.as_ref() {
                None => {
                    // Dense uniform-q: project htt/gt (and the 0-width htbeta, a
                    // no-op) through the ext-coord manifold.
                    self.apply_sae_riemannian_geometry(&mut sys);
                }
                Some(layout) => {
                    // Compact heterogeneous-q: project each row's htt/gt at its
                    // own ext-coord point, mirroring the full path's compact
                    // Riemannian block (htbeta is 0-width here, so skipped).
                    if !self.ext_coord_manifold().is_euclidean() {
                        for row_idx in 0..n {
                            let (manifold_i, point_i) =
                                self.compact_row_ext_manifold_and_point(row_idx, layout);
                            let t_i = point_i.view();
                            let gt_e = sys.rows[row_idx].gt.clone();
                            let htt_e = sys.rows[row_idx].htt.clone();
                            sys.rows[row_idx].gt =
                                manifold_i.project_gradient_to_tangent(t_i, gt_e.view());
                            sys.rows[row_idx].htt = manifold_i.riemannian_hessian_matrix(
                                t_i,
                                gt_e.view(),
                                htt_e.view(),
                            );
                        }
                    }
                }
            }
            if let Some(deflation) = self.row_gauge_deflation_for_layout(row_layout.as_ref()) {
                sys.set_row_gauge_deflation(deflation);
            }
            self.last_row_layout = row_layout;
            self.last_frames_active = frames_engaged;
            return Ok(sys);
        }
        // Apply Riemannian geometry to the per-row row blocks (htt, gt) and
        // also to the per-row Kronecker local Jacobians stored in kron_jac.
        // When the SAE ext-coord manifold is non-Euclidean (any atom latent
        // on sphere / circle / interval), the local Jacobian rows that map
        // into the t-block tangent space must be projected via the per-row
        // tangent projector P_i.  This mirrors what
        // `apply_riemannian_latent_geometry` does to `row.htbeta`, applied
        // here to the (q × p) kron_jac so the Kronecker htbeta_matvec uses
        // the Riemannian-projected form.
        // Apply Riemannian geometry only for the dense uniform-q layout. Any
        // compact active-set layout (JumpReLU gate or large-K softmax/IBP
        // truncation) has heterogeneous q_i; the Riemannian projector path
        // requires a uniform latent dimension. The sparse plan only engages on
        // Euclidean ext-coord manifolds (see `sparse_active_plan`), so skipping
        // the projector here is correct — there is nothing to project.
        match row_layout.as_ref() {
            None => {
                let raw_gt_rows: Vec<Array1<f64>> =
                    sys.rows.iter().map(|row| row.gt.clone()).collect();
                self.apply_sae_riemannian_geometry(&mut sys);
                let manifold = self.ext_coord_manifold();
                if !frames_engaged && !manifold.is_euclidean() {
                    let ext = self.ext_coord_matrix();
                    // Project the local Jacobian columns onto the tangent space at
                    // each row's ext-coord point. Each column `j` of the row's
                    // (q_row × p) Jacobian is an ambient-space vector of length
                    // `q_row`; the manifold projector acts on one such column at a
                    // time. Working directly on the row-major `jac_flat` storage via
                    // a single reusable `col_buf` avoids the two dense (q × p) copies
                    // (flatten→Array2, project, unflatten→Vec) that previously fired
                    // per row. `t_buf` still holds the row's ext-coord vector.
                    let mut t_buf = vec![0.0_f64; q];
                    let mut col_buf = Array1::<f64>::zeros(q);
                    for row_idx in 0..n {
                        let ext_row = ext.row(row_idx);
                        for (slot, &v) in t_buf.iter_mut().zip(ext_row.iter()) {
                            *slot = v;
                        }
                        let t_i = ArrayView1::from(t_buf.as_slice());
                        let raw_gt = raw_gt_rows[row_idx].view();
                        let jac_flat = &mut kron_jac[row_idx];
                        let q_row = jac_flat.len() / p;
                        for j in 0..p {
                            for c in 0..q_row {
                                col_buf[c] = jac_flat[c * p + j];
                            }
                            let projected_col = manifold.project_vector_to_gradient_tangent(
                                t_i,
                                raw_gt.slice(ndarray::s![..q_row]),
                                col_buf.slice(ndarray::s![..q_row]),
                            );
                            for c in 0..q_row {
                                jac_flat[c * p + j] = projected_col[c];
                            }
                        }
                    }
                }
            }
            Some(layout) => {
                // Compact active-set layout (#1117 follow-up): the dense
                // `ext_coord_manifold()` is keyed to the uniform full-`q` block
                // ordering, so it cannot be applied to the heterogeneous compact
                // rows directly. Instead we rebuild, PER ROW, the product manifold
                // and ext-coord point in that row's compact column order (see
                // `compact_row_ext_manifold_and_point`) and apply the SAME three
                // per-row Riemannian operations the dense
                // `apply_riemannian_latent_geometry` applies — gradient tangent
                // projection of `gt`, the Riemannian Hessian correction of `htt`,
                // and the column tangent projection of `htbeta` — plus the
                // identical Kronecker `kron_jac` column projection. On the shared
                // active support this is byte-identical to slicing the dense
                // product manifold, so engaging the sparse plan on a non-Euclidean
                // ext manifold is now correct (the former
                // `is_euclidean()`-only guard in `sparse_active_plan` is lifted).
                //
                // Euclidean ext manifolds still skip all of this (every
                // per-row manifold is a product of Euclidean parts whose
                // projector is the identity); we early-out so those rows stay
                // byte-for-byte the historical compact path.
                if !self.ext_coord_manifold().is_euclidean() {
                    for row_idx in 0..n {
                        let (manifold_i, point_i) =
                            self.compact_row_ext_manifold_and_point(row_idx, layout);
                        let t_i = point_i.view();
                        // gt / htt / htbeta on the compact ArrowRowBlock, exactly
                        // as `apply_riemannian_latent_geometry` does for dense
                        // uniform-q rows.
                        let gt_e = sys.rows[row_idx].gt.clone();
                        let htt_e = sys.rows[row_idx].htt.clone();
                        sys.rows[row_idx].gt =
                            manifold_i.project_gradient_to_tangent(t_i, gt_e.view());
                        sys.rows[row_idx].htt =
                            manifold_i.riemannian_hessian_matrix(t_i, gt_e.view(), htt_e.view());
                        // #1406: only the frames path holds a real dense `htbeta`
                        // slab; the matrix-free path leaves it 0-width (the
                        // cross-block geometry is applied to `kron_jac` below), so
                        // projecting a zero-column matrix is a no-op we skip.
                        if frames_engaged {
                            let htbeta_e = sys.rows[row_idx].htbeta.clone();
                            sys.rows[row_idx].htbeta = manifold_i
                                .project_matrix_columns_to_gradient_tangent(
                                    t_i,
                                    gt_e.view(),
                                    htbeta_e.view(),
                                );
                        }
                        // Kronecker local-Jacobian column projection (full-B path
                        // only), using the SAME pre-projection gradient `gt_e` so
                        // the cross-block geometry matches the dense branch.
                        if !frames_engaged {
                            let jac_flat = &mut kron_jac[row_idx];
                            let q_row = jac_flat.len() / p;
                            let mut col_buf = Array1::<f64>::zeros(q_row);
                            for j in 0..p {
                                for c in 0..q_row {
                                    col_buf[c] = jac_flat[c * p + j];
                                }
                                let projected_col = manifold_i.project_vector_to_gradient_tangent(
                                    t_i,
                                    gt_e.view(),
                                    col_buf.view(),
                                );
                                for c in 0..q_row {
                                    jac_flat[c * p + j] = projected_col[c];
                                }
                            }
                        }
                    }
                }
            }
        }
        // Build and install the full-B Kronecker htbeta_matvec.
        //
        // `SaeKroneckerRows` holds per-row `(a_phi, local_jac)` and implements
        // the cross-block operator without ever materialising the dense
        // `(q × K·p)` slab.  The cross-block factorises as `H_tβ = L · J_β`,
        // where `J_β = φᵀ ⊗ I_p` projects a length-`K` β vector onto the
        // `p`-dimensional decoded output space (`apply_jbeta`) and `L_i` is
        // the per-row `(q_i × p)` assignment+coordinate Jacobian that lifts
        // that p-vector into the row's `q_i`-dim tangent block (`apply_l`).
        // Both factors are required: the contract of `set_row_htbeta_operator`
        // is `out.len() == d` (= `q_i`), so writing `apply_jbeta`'s p-vector
        // output directly into a length-`q_i` buffer overflows whenever
        // `p > q_i` (the common case once `p` reflects real feature width).
        // Symmetric for the transpose: `H_βt = J_βᵀ · Lᵀ`, so apply `Lᵀ`
        // first to map the q_i-vector back to p-space, then scatter through
        // the support.
        // #1017/#1026: the legacy full-B device PCG assumes `G ⊗ I_p`, while
        // framed systems carry `G_ij ⊗ W_ij` with rank-r atom blocks. Feeding a
        // framed system to that kernel would silently return the wrong Newton
        // step. Framed device PCG therefore needs the dedicated factored kernel.
        // #1033 large-n: the per-row support `kron_a_phi` and local Jacobians
        // `kron_jac` are consumed by BOTH the host matrix-free row operator
        // (`SaeKroneckerRows`) and the solver's `DeviceSaePcgData`. Previously
        // each took its own full `O(n·q·p)` / `O(n·k_active)` clone, so the
        // always-resident footprint of the CPU non-frames path carried TWO copies
        // of the dominant Jacobian slab. Promote each to a single `Arc<[…]>` once
        // and hand both consumers a refcount bump (`O(1)`) — the backing
        // allocation is shared, halving the resident per-row Jacobian memory.
        // Reads are identical (`&arc[row]`, `.len()`), so the assembled system and
        // every matvec are bit-for-bit unchanged.
        let device_rows = if frames_engaged {
            None
        } else {
            let a_phi_shared: Arc<[Vec<(usize, f64)>]> =
                Arc::from(std::mem::take(&mut kron_a_phi).into_boxed_slice());
            let jac_shared: Arc<[Vec<f64>]> =
                Arc::from(std::mem::take(&mut kron_jac).into_boxed_slice());
            Some((a_phi_shared, jac_shared))
        };
        if !frames_engaged {
            let (a_phi_shared, jac_shared) = device_rows
                .clone()
                .expect("non-frames path always populates device_rows");
            let kron = Arc::new(SaeKroneckerRows::new(p, a_phi_shared, jac_shared));
            let kron_t = Arc::clone(&kron);
            let p_dim = p;
            sys.set_row_htbeta_operator(
                move |row_idx, x, out| {
                    // out = L_i · (J_β · x). Allocate a length-p scratch buffer
                    // for the intermediate decoded-output vector; both factors
                    // overwrite their output buffers (`apply_jbeta` zeroes
                    // before accumulating, `apply_l` writes per-row), so no
                    // pre-zeroing of `u_p`/`out` is needed.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(xs) = x.as_slice() {
                        kron.apply_jbeta(row_idx, xs, &mut u_p);
                    } else {
                        let x_vec: Vec<f64> = x.iter().copied().collect();
                        kron.apply_jbeta(row_idx, &x_vec, &mut u_p);
                    }
                    kron.apply_l(row_idx, &u_p, out_slice);
                },
                move |row_idx, v, out| {
                    // out += J_βᵀ · (Lᵀ · v). `apply_l_t` accumulates into a
                    // zero-initialised length-p buffer to produce the p-vector
                    // `Lᵀ v`; `scatter_jbeta_t` then adds φ_i[s] · u_p[j] into
                    // the length-K β accumulator at each active `(s, j)`.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(vs) = v.as_slice() {
                        kron_t.apply_l_t(row_idx, vs, &mut u_p);
                    } else {
                        let v_vec: Vec<f64> = v.iter().copied().collect();
                        kron_t.apply_l_t(row_idx, &v_vec, &mut u_p);
                    }
                    kron_t.scatter_jbeta_t(row_idx, &u_p, out_slice);
                },
            );
        }
        let mut beta_penalty_assembly = SaeBetaPenaltyAssembly::default();
        let factored_row_projection = if frames_engaged && analytic_penalties.is_some() {
            Some(&frame_projection)
        } else {
            None
        };
        if let Some(registry) = analytic_penalties {
            // Upfront validation: refuse penalty kinds the SAE row layout
            // cannot host, and refuse mixed-d row-block configurations.
            // This makes the dispatch loop below total — no runtime
            // "unsupported penalty" fallthrough, no K-gating.
            self.validate_analytic_penalty_registry(registry)
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
            beta_penalty_assembly = self
                .add_sae_analytic_penalty_contributions(
                    &mut sys,
                    registry,
                    penalty_scale,
                    row_layout.as_ref(),
                    dense_beta_curvature,
                    factored_row_projection,
                )
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        }
        // #1026 — decoder repulsion (collinearity-gated, registry-independent):
        // accumulate into the full-`B` β-tier here, BEFORE the frame transform,
        // so a framed system carries it identically to the analytic β penalties.
        // No-op unless two atoms are near-collinear (the frozen gate is `None`).
        if self.add_sae_decoder_repulsion(&mut sys, penalty_scale, dense_beta_curvature) {
            beta_penalty_assembly.record_curvature(dense_beta_curvature);
        }
        // #1026/#1522 — interior-point collapse-prevention barriers. The amplitude
        // barrier supplies the OUTWARD radial force at the zero-decoder collapse
        // point (the principal failure state the threshold repulsion skips), and
        // the separation barrier supplies the alignment-divergent separating
        // curvature on normalized shapes weighted by coactivation. Both accumulate
        // into the full-`B` β-tier here, BEFORE the frame transform, so a framed
        // system carries them identically to the analytic β penalties.
        // #1610 — on the dense path the barrier's Levenberg majorizer scatters
        // onto `sys.hbb`; on the matrix-free / framed production path `sys.hbb` is
        // unused, so the barrier hands back a per-atom scalar ridge which we fold
        // into `smooth_scaled_s` (the single source for the CPU composite penalty
        // op AND the device smooth blocks), restoring the collapse-prevention
        // curvature the operator was silently dropping there.
        let mut sep_atom_curv = vec![0.0_f64; self.atoms.len()];
        if self.add_sae_separation_barrier(
            &mut sys,
            penalty_scale,
            dense_beta_curvature,
            &mut sep_atom_curv,
        ) {
            if dense_beta_curvature {
                beta_penalty_assembly.record_curvature(true);
            } else {
                // Fold the per-atom majorizer `lev_k·I_{M_k}` into the smooth
                // penalty factor `λ S_k`. With `⊗ I_p` (full-`B`) or `⊗ I_{r_k}`
                // (factored, `U_kᵀU_k = I`) this is exactly the `lev_k·I` block
                // diagonal the dense path writes — and it now flows through the
                // structured penalty op and the device smooth blocks. No
                // `deferred_factored` mark: the curvature is in the smooth op, not
                // a deferred dense block, so the device path stays engaged.
                for atom_idx in 0..self.atoms.len() {
                    let c = sep_atom_curv[atom_idx];
                    if c > 0.0 {
                        let m = smooth_scaled_s[atom_idx].nrows();
                        for i in 0..m {
                            smooth_scaled_s[atom_idx][[i, i]] += c;
                        }
                        smooth_ops[atom_idx] = Arc::new(IdentityRightKroneckerPenaltyOp {
                            factor_a: smooth_scaled_s[atom_idx].clone(),
                            p,
                            global_offset: beta_offsets[atom_idx],
                            k: beta_dim,
                        });
                    }
                }
            }
        }
        if frames_engaged {
            // ── #972 / #977 T1 — FACTORED β-tier transform ──────────────────
            //
            // The entire β-tier above was assembled in the full-`B` (p-wide)
            // layout: `sys.gb` is `g_B` (length `beta_dim`), `sys.hbb` carries
            // any analytic Beta-tier penalty, and `g_blocks` is the
            // FRAME-INDEPENDENT basis Gram. We now rebuild the β-tier in the
            // factored coordinate space `C` (width `factored_border_dim`), the
            // full-`B` system sandwiched by `Φ = blkdiag(I_{M_k} ⊗ U_k)`:
            //   * gradient   `g_C = Φᵀ g_B`              (per atom `(g_B U_k)`),
            //   * data H      `Φᵀ(G⊗I_p)Φ = G_{ij}⊗(U_iᵀU_j)`,
            //   * smooth      `λ S_k ⊗ I_{r_k}`          (since `U_kᵀU_k = I`),
            //   * analytic    `Φᵀ hbb Φ`                 (dense, only if written).
            // Un-framed atoms ride the `r_k = p, U_k = I_p` identity special case.
            let off_c = &frame_projection.border_offsets;
            let ranks = &frame_projection.ranks;
            let basis_sizes = &frame_projection.basis_sizes;
            let border_dim = frame_projection.border_dim();
            let gb_c = frame_projection.project_border_vec(sys.gb.view());

            // Data β-Hessian: `G_{ij} ⊗ W_{ij}` with `W_{ij} = U_iᵀU_j`. The
            // basis Gram `g_blocks` is unchanged; only the output factor is the
            // per-pair frame overlap (`I_{r_k}` within a framed atom, `I_p` for
            // un-framed).
            let mut frame_blocks: Vec<FactoredFrameGBlock> = Vec::with_capacity(g_blocks.len());
            for ((atom_i, atom_j), data) in g_blocks.into_iter() {
                if data.iter().all(|&v| v == 0.0) {
                    continue;
                }
                // `W_{ij} = U_iᵀ U_j` from the precomputed per-atom frames.
                let w = self.frame_cross_factor(atom_i, atom_j);
                frame_blocks.push(FactoredFrameGBlock {
                    atom_i,
                    atom_j,
                    g: data,
                    w,
                });
            }
            // #1017/#1026 — snapshot the factored data-fit blocks for the
            // frames-engaged device PCG BEFORE `FactoredFrameKroneckerOp::new`
            // consumes them. Cheap clone (co-occurring blocks only).
            let device_frame_blocks = frame_blocks.clone();
            let data_op =
                FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), frame_blocks)?;

            // Smooth penalty in factored space: `λ S_k ⊗ I_{r_k}` at `off_C[k]`.
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(self.atoms.len() + 2);
            for k in 0..self.atoms.len() {
                let r = ranks[k];
                ops.push(Arc::new(IdentityRightKroneckerPenaltyOp {
                    factor_a: smooth_scaled_s[k].clone(),
                    p: r,
                    global_offset: off_c[k],
                    k: border_dim,
                }));
            }
            ops.push(Arc::new(data_op));
            // Analytic Beta-tier penalty: project the dense full-`B` `hbb` block
            // `Φᵀ hbb Φ` into the factored space. Only present when a Beta-tier
            // penalty actually wrote `hbb` (else `hbb` is all-zero and the dense
            // `(border_dim)²` op is skipped entirely, exactly as full-`B`).
            if beta_penalty_assembly.dense_written {
                let hbb_c =
                    self.project_dense_penalty_to_factored(sys.hbb.view(), &frame_projection);
                ops.push(Arc::new(DensePenaltyOp(hbb_c)));
            } else if beta_penalty_assembly.deferred_factored {
                // Registry Beta-tier curvature deferred to factored-space probing.
                // The registry may be absent when `deferred_factored` was set ONLY
                // by the frozen-gate decoder repulsion (which is
                // registry-independent), so start from a zero factored block in
                // that case instead of unwrapping.
                let mut hbb_c = match analytic_penalties {
                    Some(registry) => self.build_factored_beta_penalty_curvature(
                        registry,
                        penalty_scale,
                        &frame_projection,
                    ),
                    None => Array2::<f64>::zeros((
                        frame_projection.border_dim(),
                        frame_projection.border_dim(),
                    )),
                };
                // #1610 — the frozen-gate decoder repulsion's PSD majorizer was
                // dropped on this matrix-free/framed path (only its gradient was
                // applied). Project it into the factored block via the same
                // `psd_majorizer_hvp` + frame-projection probe pattern the registry
                // DecoderIncoherence uses, so the collapse-prevention curvature
                // reaches the operator here too. No-op when no repulsion is active.
                self.add_factored_repulsion_curvature(
                    &mut hbb_c,
                    penalty_scale,
                    &frame_projection,
                );
                ops.push(Arc::new(DensePenaltyOp(hbb_c)));
            }

            // Re-point the system's β-tier to the factored width. The t-tier
            // (per-row `htt`, `gt`) is frame-independent and untouched; row
            // cross-block slabs were allocated and assembled directly in
            // factored coordinates, so analytic row supplements and data-fit
            // cross terms already share shape `(q_i × factored_border_dim)`.
            sys.k = border_dim;
            sys.gb = gb_c;
            self.reclaim_border_hbb_workspace(&mut sys);
            // Factored per-atom block ranges for the block-Jacobi Schur
            // preconditioner: `[off_C[k] .. off_C[k] + M_k·r_k]`.
            let mut block_ranges: Vec<std::ops::Range<usize>> =
                Vec::with_capacity(self.atoms.len());
            for k in 0..self.atoms.len() {
                let start = off_c[k];
                block_ranges.push(start..start + basis_sizes[k] * ranks[k]);
            }
            sys.set_block_offsets(Arc::from(block_ranges.into_boxed_slice()));
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: border_dim, ops }));
            // #1017/#1026 — install the frames-engaged device SAE PCG data. Skipped
            // (CPU fallback) when a dense analytic Beta-tier penalty fired (the
            // device kernel does not model that extra dense term). Builder:
            // `crate::frames::build_framed_device_sae_data`.
            let has_dense_beta_penalty =
                beta_penalty_assembly.dense_written || beta_penalty_assembly.deferred_factored;
            if !has_dense_beta_penalty {
                let device = crate::frames::build_framed_device_sae_data(
                    crate::frames::FramedDeviceArgs {
                        p,
                        border_dim,
                        border_offsets: off_c.as_slice(),
                        ranks: ranks.as_slice(),
                        basis_sizes: basis_sizes.as_slice(),
                        smooth_scaled_s: &smooth_scaled_s,
                        frame_blocks: device_frame_blocks,
                        rows: &sys.rows,
                    },
                );
                sys.set_device_sae_pcg_data(device);
            }
        } else {
            let (device_a_phi, device_local_jac) =
                device_rows.expect("full-beta SAE PCG rows are cloned before row operator install");
            // Wire per-atom β block ranges so the Jacobi preconditioner builds one
            // dense Schur sub-block per atom (block-Jacobi) instead of scalar-diagonal
            // inversion.  Each atom's decoder coefficients form a natural block:
            // `[beta_offsets[k] .. beta_offsets[k] + basis_size[k] * p_out]`.
            sys.set_block_offsets(self.beta_block_offsets());
            // Install the composite BetaPenaltyOp (#296): smoothness contributions
            // via per-atom KroneckerPenaltyOp (avoid dense K×K materialisation), the
            // data-fit Gauss-Newton β-Hessian as the structured `G ⊗ I_p`
            // SparseBlockKroneckerPenaltyOp (block-sparse over co-occurring
            // `(atom, atom')` pairs, block-diagonal across the `p` output channels,
            // identical per channel), plus — only when a Beta-tier analytic penalty
            // was written — the dense `sys.hbb` residual contribution. When no beta
            // penalty fired, `sys.hbb` is all-zero and the dense `(K·p)²` operator
            // is skipped entirely. The sparse data op tracks only the active-atom
            // couplings, so its storage and matvec cost scale with `k_active`, not
            // `K`, at `K = 100K`.
            // Convert the per-atom-pair coupling map into `SparseGBlock`s keyed
            // by μ-space offsets. Empty blocks (no co-occurrence) are simply
            // absent from the map.
            let g_sparse_blocks: Vec<SparseGBlock> = g_blocks
                .into_iter()
                .filter_map(|((atom_i, atom_j), data)| {
                    if data.iter().all(|&v| v == 0.0) {
                        None
                    } else {
                        Some(SparseGBlock {
                            row_off: mu_offsets[atom_i],
                            col_off: mu_offsets[atom_j],
                            data,
                        })
                    }
                })
                .collect();
            let device_smooth_blocks = smooth_scaled_s
                .iter()
                .enumerate()
                .map(|(atom_idx, factor_a)| {
                    // #1117 — rank deficiency is removed at the basis layer, so the
                    // device PCG smooth block is just `λ S_k ⊗ I_p` (full-rank
                    // design); no data-null deflation is folded in here.
                    DeviceSaeSmoothBlock {
                        global_offset: beta_offsets[atom_idx],
                        factor_a: factor_a.clone(),
                    }
                })
                .collect();
            sys.set_device_sae_pcg_data(DeviceSaePcgData {
                p,
                beta_dim,
                a_phi: device_a_phi,
                local_jac: device_local_jac,
                smooth_blocks: device_smooth_blocks,
                sparse_g_blocks: g_sparse_blocks.clone(),
                frame: None,
            });
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = smooth_ops;
            ops.push(Arc::new(SparseBlockKroneckerPenaltyOp {
                p,
                dim_a: m_total,
                k: beta_dim,
                blocks: g_sparse_blocks,
            }));
            if beta_penalty_assembly.dense_written {
                ops.push(Arc::new(DensePenaltyOp(sys.hbb.clone())));
            }
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: beta_dim, ops }));
            self.reclaim_border_hbb_workspace(&mut sys);
        }
        if let Some(deflation) = self.row_gauge_deflation_for_layout(row_layout.as_ref()) {
            sys.set_row_gauge_deflation(deflation);
        }
        // #1038 IBP cross-row Woodbury source. The exact IBP Hessian has the
        // per-column rank-one cross-row block `H_(i,k),(j,k) = w·s'_k·z'_ik·z'_jk`
        // (for ALL `i,j`, including the `i=j` self term) that couples DISTINCT
        // latent rows through the shared empirical mass `M_k = Σ_i z_ik`. The
        // assembled row-block-diagonal `htt` already carries the `i=j` self term
        // `w·s'_k·z'_ik²` — it is the first summand of `assignment_hdiag`'s
        // `hessian_diag` value `w·(score_derivative·z_jac² + score·c_ik)` written
        // at the logit diagonal above. So the consumer (`solver::arrow_schur`,
        // #1038 `IbpCrossRowSource`/`CrossRowWoodbury`) DOWNDATES exactly
        // `Σ_k d_k·z'_ik²` (`self_term_downdate`) to recover the NO-SELF base
        // `H₀'`, then re-adds the FULL rank-one `U D Uᵀ` via the determinant
        // lemma — so value, the evidence log-determinant, and the θ/ρ-adjoint all
        // differentiate the SAME `H_full = H₀' + U D Uᵀ`.
        //
        // The source is built from the SAME `ibp_assignment_third_channels`
        // operator the #1006 θ-adjoint consumes:
        //   * `d[k] = cross_row_d[k] = w·s'_k = w·score_derivative_k` (the column
        //     `D`-coefficient — NOT sign-definite, hence the consumer's
        //     indefinite-capacitance LU);
        //   * `entries[(i,k)] = (global_t_index, k, z'_ik)` with `z'_ik =
        //     z_jac[i·K + k]`. For the DENSE layout (`assignment_coord_dim() = K`,
        //     `last_row_layout = None`) atom `k`'s logit slot is local position `k`
        //     of row `i`'s block, so `global_t_index = sys.row_offsets[i] + k`. For
        //     the COMPACT layout (#1420) only the row's active atoms are
        //     coordinates and atom `k` lives at local position `pos` of
        //     `active_atoms[row]`, so `global_t_index = sys.row_offsets[i] + pos`.
        //     Both pin the `U`-column convention bit-for-bit to the consumer's
        //     `ibp_logit_sites`/`row_vars_for_cache_row` slot mapping.
        if let Some(channels) = ibp_assignment_third_channels(&self.assignment, rho)? {
            let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(n * k_atoms);
            for row in 0..n {
                let start = row * k_atoms;
                let g_base = sys.row_offsets[row];
                match row_layout.as_ref() {
                    // #1420: compact layout — the local logit slot `pos` (not the
                    // global atom index `k`) is the t-coordinate. Atom `k`'s logit
                    // lives at local position `pos` of `active_atoms[row]`, so emit
                    // `(g_base + pos, atom, z_jac[row·K + atom])` for the active set
                    // only. Using `g_base + k` would attach atom `k`'s derivative to
                    // the wrong slot (and run out of range for compact rows),
                    // violating the `IbpCrossRowSource` contract.
                    Some(layout) => {
                        for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                            let z_prime = channels.z_jac[start + atom];
                            entries.push((g_base + pos, atom, z_prime));
                        }
                    }
                    // Dense layout: atom `k`'s logit slot is local position `k`.
                    None => {
                        for k in 0..k_atoms {
                            let z_prime = channels.z_jac[start + k];
                            entries.push((g_base + k, k, z_prime));
                        }
                    }
                }
            }
            let source = IbpCrossRowSource {
                r: k_atoms,
                d: channels.cross_row_d.clone(),
                entries,
            };
            sys.set_ibp_cross_row_source(source);
        }
        // Store the active-set layout for `apply_newton_step`.
        self.last_row_layout = row_layout;
        // Record whether `delta_beta` from this system is a factored ΔC (needs a
        // frame lift) or a full-`B` ΔB. Read by `apply_newton_step_impl`.
        self.last_frames_active = frames_engaged;
        Ok(sys)
    }

    /// Project a dense full-`B` Beta-tier penalty Hessian `hbb` (`beta_dim ×
    /// beta_dim`, the analytic `∂²P/∂B∂B` block) into the factored coordinate
    /// space `Φᵀ hbb Φ` (`border_dim × border_dim`) for the #972 / #977 T1
    /// frame transform. `Φ = blkdiag(I_{M_k} ⊗ U_k)` maps C-space → B-space, so
    /// the projected block contracts both index legs through the per-atom frames.
    ///
    /// The projection is done in two passes to stay `O(beta_dim · border_dim +
    /// border_dim²)` instead of forming the dense `Φ` explicitly: first
    /// `T = hbb · Φ` (right multiply, columns fold `U`), then `Φᵀ · T` (left
    /// multiply, rows fold `U`). Analytic Beta-tier penalties are rare and small,
    /// so this only fires when one is actually installed.
    pub(crate) fn project_dense_penalty_to_factored(
        &self,
        hbb: ArrayView2<'_, f64>,
        projection: &FrameProjection,
    ) -> Array2<f64> {
        projection.project_block(hbb)
    }

    pub(crate) fn build_factored_beta_penalty_curvature(
        &self,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) -> Array2<f64> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let target_beta = self.flatten_beta();
        let mut hbb_c = Array2::<f64>::zeros((projection.border_dim(), projection.border_dim()));
        for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            if matches!(penalty, AnalyticPenaltyKind::Ard(_)) {
                continue;
            }
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) => {
                    self.add_factored_beta_penalty_curvature_for_penalty(
                        &mut hbb_c,
                        penalty,
                        target_beta.view(),
                        rho_local,
                        penalty_scale,
                        projection,
                    );
                }
                PenaltyTier::Beta => {
                    self.add_factored_beta_penalty_curvature_for_penalty(
                        &mut hbb_c,
                        penalty,
                        target_beta.view(),
                        rho_local,
                        penalty_scale,
                        projection,
                    );
                }
                _ => {}
            }
        }
        hbb_c
    }

    pub(crate) fn add_factored_beta_penalty_curvature_for_penalty(
        &self,
        hbb_c: &mut Array2<f64>,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) {
        let p = self.output_dim();
        if let AnalyticPenaltyKind::DecoderIncoherence(base) = penalty {
            let Some(per_fit) = self.live_decoder_incoherence_penalty(base) else {
                return;
            };
            let beta_dim = self.beta_dim();
            let mut probe = Array1::<f64>::zeros(beta_dim);
            for k in 0..self.atoms.len() {
                for basis_col in 0..projection.basis_sizes[k] {
                    for frame_col in 0..projection.ranks[k] {
                        probe.fill(0.0);
                        projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                        let col = projection.border_offsets[k]
                            + basis_col * projection.ranks[k]
                            + frame_col;
                        let hv = per_fit.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                        projection
                            .project_border_vec(hv.view())
                            .iter()
                            .enumerate()
                            .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                    }
                }
            }
            return;
        }
        if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
            for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                let atom_idx = projection
                    .beta_offsets
                    .iter()
                    .position(|&offset| offset == start)
                    .expect("live mechanism-sparsity offset must match an SAE atom");
                let block_len = end - start;
                let mut local_penalty = per_atom.clone();
                local_penalty.target = PsiSlice {
                    range: 0..block_len,
                    latent_dim: Some(projection.basis_sizes[atom_idx]),
                };
                let block = target_beta.slice(s![start..end]);
                let mut probe = Array1::<f64>::zeros(block_len);
                for basis_col in 0..projection.basis_sizes[atom_idx] {
                    for frame_col in 0..projection.ranks[atom_idx] {
                        probe.fill(0.0);
                        projection.lift_local_axis_into(&mut probe, atom_idx, basis_col, frame_col);
                        let col = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx]
                            + frame_col;
                        let hv = local_penalty.psd_majorizer_hvp(block, rho_local, probe.view());
                        projection.project_local_atom_vec_into(
                            atom_idx,
                            hv.view(),
                            hbb_c.column_mut(col),
                            penalty_scale,
                        );
                    }
                }
            }
            return;
        }
        if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
            for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                let atom_idx = projection
                    .beta_offsets
                    .iter()
                    .position(|&offset| offset == start)
                    .expect("live nuclear-norm offset must match an SAE atom");
                let block = target_beta.slice(s![start..end]);
                let block_len = end - start;
                let mut probe = Array1::<f64>::zeros(block_len);
                for basis_col in 0..projection.basis_sizes[atom_idx] {
                    for frame_col in 0..projection.ranks[atom_idx] {
                        probe.fill(0.0);
                        projection.lift_local_axis_into(&mut probe, atom_idx, basis_col, frame_col);
                        let col = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx]
                            + frame_col;
                        let hv = per_atom.psd_majorizer_hvp(block, rho_local, probe.view());
                        projection.project_local_atom_vec_into(
                            atom_idx,
                            hv.view(),
                            hbb_c.column_mut(col),
                            penalty_scale,
                        );
                    }
                }
            }
            return;
        }
        let beta_dim = self.beta_dim();
        let mut probe = Array1::<f64>::zeros(beta_dim);
        for k in 0..self.atoms.len() {
            for basis_col in 0..projection.basis_sizes[k] {
                for frame_col in 0..projection.ranks[k] {
                    probe.fill(0.0);
                    projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                    let col =
                        projection.border_offsets[k] + basis_col * projection.ranks[k] + frame_col;
                    let hv = penalty.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                    projection
                        .project_border_vec(hv.view())
                        .iter()
                        .enumerate()
                        .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                }
            }
        }
        assert_eq!(p, self.output_dim());
    }

    /// #1610 — project the frozen-gate decoder-repulsion PSD majorizer into the
    /// factored β block `hbb_c`. Mirrors the `DecoderIncoherence` arm of
    /// [`Self::add_factored_beta_penalty_curvature_for_penalty`] but sources the
    /// penalty from [`Self::live_decoder_repulsion_penalty`] (registry-independent,
    /// collinearity-gated), so the repulsion curvature reaches the operator on the
    /// matrix-free/framed path where the dense `sys.hbb` write is unused. No-op
    /// when no repulsion is active.
    pub(crate) fn add_factored_repulsion_curvature(
        &self,
        hbb_c: &mut Array2<f64>,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) {
        let Some(per_fit) = self.live_decoder_repulsion_penalty() else {
            return;
        };
        let beta_dim = self.beta_dim();
        let target_beta = self.flatten_beta();
        // The repulsion penalty is non-learnable; its strength is already folded
        // into the frozen gate (see `live_decoder_repulsion_penalty`), so the rho
        // slice is empty/inert.
        let rho_local = Array1::<f64>::zeros(0);
        let mut probe = Array1::<f64>::zeros(beta_dim);
        for k in 0..self.atoms.len() {
            for basis_col in 0..projection.basis_sizes[k] {
                for frame_col in 0..projection.ranks[k] {
                    probe.fill(0.0);
                    projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                    let col =
                        projection.border_offsets[k] + basis_col * projection.ranks[k] + frame_col;
                    let hv =
                        per_fit.psd_majorizer_hvp(target_beta.view(), rho_local.view(), probe.view());
                    projection
                        .project_border_vec(hv.view())
                        .iter()
                        .enumerate()
                        .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                }
            }
        }
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
    /// The dense `ext_coord_manifold()` is keyed to the full-`q` block ordering
    /// `[assignment parts (all Euclidean for IBP-MAP / JumpReLU), then per-atom
    /// coord blocks in atom order]`. A compact active-set row instead lays its
    /// `q_active` columns out as `[one Euclidean logit slot per active atom,
    /// then each active atom's coord block in `active` order]` (see
    /// [`SaeRowLayout::from_active_atoms`] / `coord_starts`). To reuse the exact
    /// per-row Riemannian projector on the compact block we rebuild a product
    /// manifold and the matching ext-coord point in that compact order: the
    /// `active.len()` logit slots are `Euclidean` (the assignment channel is
    /// always Euclidean for the modes that engage sparsity — `assignment_coord_dim
    /// == k_atoms`), and each active atom contributes its own coordinate
    /// manifold. On the shared active support this is byte-identical to slicing
    /// the dense full-`q` product manifold, so the compact projection matches the
    /// dense path exactly — it only drops the inactive atoms' (negligible-mass)
    /// coordinate blocks the compact layout already excludes from curvature.
    ///
    /// Returns `(manifold, t_compact)` where `t_compact` has length `q_active`.
    /// The logit-slot entries of `t_compact` are filled from the row logits (the
    /// Euclidean projector ignores the point, so any finite value is equivalent;
    /// using the true logits keeps the point well-defined and finite).
    pub(crate) fn compact_row_ext_manifold_and_point(
        &self,
        row: usize,
        layout: &SaeRowLayout,
    ) -> (LatentManifold, Array1<f64>) {
        let active = &layout.active_atoms[row];
        let q_active = layout.row_q_active(row);
        let mut parts: Vec<LatentManifold> = Vec::with_capacity(active.len() + active.len());
        let mut point = Array1::<f64>::zeros(q_active);
        // Logit slots: one Euclidean part per active atom, in `active` order.
        let logits_row = self.assignment.logits.row(row);
        for (j, &k) in active.iter().enumerate() {
            parts.push(LatentManifold::Euclidean);
            point[j] = logits_row[k];
        }
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
    /// `S_k` so the REML criterion's `−½·p·rank(S)·log λ_smooth` Occam term
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

    /// Penalised quasi-Laplace evidence score for the SAE term at a FIXED ρ.
    ///
    /// #1421: this is NOT a true normalized-prior REML/evidence objective. The
    /// assignment priors (softmax entropy, JumpReLU) have NO finite normalizer:
    /// for softmax the reference-logit chart sends `P(ℓ)→0` as a free logit →±∞
    /// so `∫ e^{−λP} dℓ = ∞`, and JumpReLU's bounded penalty `0<P<λ` keeps
    /// `e^{−λP}` bounded below over an unbounded domain, also divergent. There is
    /// therefore no ρ-independent assignment-prior normalizer that can be dropped
    /// as a constant. The smoothing-penalty `−½log|λS|_+` term IS a genuine
    /// (proper-Gaussian) REML normalizer and is kept exactly; the rest is a
    /// penalized quasi-Laplace score (Laplace curvature term `½log|H|` around the
    /// inner optimum), which the engine minimizes over ρ.
    ///
    /// Runs the inner `(t, β)` arrow-Schur Newton solve to convergence at the
    /// supplied ρ (with NO in-loop ARD update — ρ is owned by the engine),
    /// then forms the Laplace/REML cost
    ///
    /// ```text
    /// V(ρ) = ℓ_pen(t̂, β̂; ρ) + ½ log|H(t̂, β̂; ρ)|
    ///        − ½ · p · (Σ_k rank S_k) · log λ_smooth
    /// ```
    ///
    /// where `ℓ_pen = loss.total()` is the penalised objective at the inner
    /// optimum and `½ log|H|` is the Laplace normaliser. `H` is the joint
    /// `(t, β)` Hessian assembled by the arrow-Schur system; its `H_tt` block
    /// carries `α = exp(log_ard)` on its diagonal, so as α grows `½ log|H|`
    /// rises while the `−½·n·log α` already inside `loss.ard` falls — their
    /// balance IS the effective-dof term that the deleted `α = n/‖t‖²` rule
    /// dropped, which is why the criterion needs no clamp to stay finite on a
    /// collapsing axis.
    ///
    /// The final `−½·p·rank(S)·log λ_smooth` term is the smoothing-penalty
    /// normaliser `−½ log|λ S|_+` restricted to its ρ-dependent part: `S_k` is
    /// shared across all `p` decoder output channels (the `⊗ I_p` Kronecker
    /// structure), so `log|λ S|_+ = p·rank(S)·log λ + p·log|S|_+`, and the
    /// `½ p·log|S|_+` piece is ρ-independent. The ρ-independent additive
    /// constants that ARE dropped here (they shift `V` by a constant and do not
    /// affect the ρ-argmin) are the `2π` Laplace constant and the base
    /// `½ p·log|S|_+` penalty logdet. #1421: NO assignment-prior normalizer is
    /// dropped, because none exists (softmax/JumpReLU priors are improper — see
    /// the doc on this function): the quasi-Laplace score simply omits a
    /// normalizer that is not a finite constant.
    ///
    /// Returns `(V, loss)` so the engine can both rank ρ and surface the inner
    /// loss breakdown.
    pub fn reml_criterion(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        self.reml_criterion_with_refine_policy(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            true,
        )
    }

    pub(crate) fn reml_criterion_with_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        let plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if plan.streaming {
            // #1225: streaming and dense MUST optimize the SAME mathematical
            // objective — the full REML criterion `loss.total() + extra_penalty +
            // ½ log|H| − Occam`. The streaming branch previously returned only
            // `loss.total() + extra_penalty_energy`, dropping the Laplace
            // normalizer `½ log|H|` and the Occam term, so large shapes (exactly
            // where streaming is needed) were ranked by penalized loss rather than
            // REML — and dense vs streaming disagreed on the objective. Route
            // through the streaming exact-logdet path, which assembles the same
            // chunk-by-chunk-bit-identical `½ log|H|_stream` and the same
            // `−Occam`/extra-penalty terms as the dense `reml_criterion_with_cache`
            // (different memory strategy, same objective).
            self.reml_criterion_streaming_exact(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
        } else {
            let (v, loss, _cache) = self.reml_criterion_with_cache_refine_policy(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                refine_progress_extension,
            )?;
            Ok((v, loss))
        }
    }

    /// As [`Self::reml_criterion`], but also returns the converged undamped
    /// `ArrowFactorCache` so callers (the EFS fixed-point step) can read the
    /// selected-inverse traces `(H⁻¹)_tt` / `(H⁻¹)_ββ` without re-factoring.
    /// The cache is the single shared O(K³) Direct factor; both the
    /// log-determinant criterion and the Fellner-Schall ρ-step consume it.
    pub fn reml_criterion_with_cache(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        self.reml_criterion_with_cache_refine_policy(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            true,
        )
    }

    pub(crate) fn reml_criterion_with_cache_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        let admission_plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if !admission_plan.direct_logdet_admitted() {
            return Err(format!(
                "SaeManifoldTerm::reml_criterion_with_cache: predicted working set {} bytes exceeds budget {} bytes for dense evidence cache; shape n={},p={},K={}; cost-only streaming route is required",
                admission_plan.estimated_direct_peak_bytes,
                admission_plan.in_core_budget_bytes,
                self.n_obs(),
                self.output_dim(),
                self.k_atoms()
            ));
        }
        // 1. Run the inner (t, β) Newton solve to convergence at FIXED ρ.
        //    `run_joint_fit_arrow_schur` no longer touches ρ.
        let mut rho_fixed = rho.clone();
        let mut loss = self.run_joint_fit_arrow_schur(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;

        // 2. Drive the inner (t, β) solve to the KKT/step-converged optimum and
        //    take one final UNDAMPED factor there to obtain the joint Hessian
        //    log-determinant. We force ridge = 0 and the dense `Direct` Schur
        //    mode so `arrow_log_det_from_cache` returns the exact
        //    `log|H| = Σ_i log|H_tt^(i)| + log|Schur_β|` (it rejects damped
        //    factors and InexactPCG caches, which have no dense Schur factor).
        //    This is the same evidence convention the main GAM REML path uses.
        //    The shared `converge_inner_for_undamped_logdet` driver guarantees
        //    the per-row `H_tt^(i)` blocks are PD at the converged optimum so
        //    the undamped (`ridge = 0`) factorization succeeds — the streaming
        //    log-det path reuses the identical driver so both rank the same
        //    converged Laplace optimum and stay bit-identical.
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let cache = self.converge_inner_for_undamped_logdet(
            target,
            rho,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &options,
            refine_progress_extension,
        )?;
        self.record_evidence_gauge_deflation_count(cache.gauge_deflated_directions)?;
        loss.evidence_gauge_deflated_directions = cache.gauge_deflated_directions;
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            // Distinguish a GENUINE infeasibility — a probed ρ where the joint
            // Hessian is not PD so the Laplace evidence log-det is undefined —
            // from a real factorization defect. The cross-row IBP Woodbury
            // capacitance `C = I_R + D·Uᵀ H₀'⁻¹ U` can have det ≤ 0 at a ρ the
            // outer optimizer line-searches into (the indefinite basin adjacent
            // to the PD region); there the log-det legitimately does not exist.
            // That refusal must be RECOVERABLE (the outer BFGS should get +∞ and
            // steer back into the PD region), exactly like the "non-PD per-row
            // H_tt block" refusal — not a fatal `RemlOptimizationFailed` that
            // aborts the whole fit. See `is_recoverable_value_probe_refusal`.
            // (The old message claimed "no dense Schur factor", which is false
            // here — the Schur factor is present; the Woodbury correction is the
            // non-finite term.)
            if cache.cross_row_woodbury.is_some()
                && !cache.cross_row_woodbury_log_det().is_finite()
            {
                "SaeManifoldTerm::reml_criterion: cross-row IBP joint Hessian is non-PD at \
                 this ρ; evidence Laplace log-det undefined (infeasible ρ probe)"
                    .to_string()
            } else {
                "SaeManifoldTerm::reml_criterion: arrow_log_det_from_cache returned None \
                 (undamped joint Hessian log-det unavailable for the Laplace normaliser)"
                    .to_string()
            }
        })?;

        // 3. Smoothing-penalty Occam term `−½·Σ_k r_k·rank(S_k)·log λ_smooth`
        //    plus the profiled-frame evidence-dimension correction
        //    `+½·Σ_k r_k·(p−r_k)·log λ_smooth` (issue #972). On the full-`B` path
        //    (`r_k == p`, no frames) this is exactly the historical
        //    `½·p·(Σ rank S_k)·log λ_smooth`, so the small-model criterion is
        //    unchanged. The single seam is `reml_occam_term`, shared with the
        //    streaming path so both rank the identical Laplace dimension count.
        let occam = self.reml_occam_term(rho)?;

        // Decoder-block analytic-penalty energy (#671/#672). The inner solve
        // descended this energy (it enters `gb`/`hbb`) but it had no native
        // `loss.*` representative, so the Laplace criterion `v` was scoring a
        // different objective than the one minimized. Add the converged
        // decoder-penalty value so the ρ-sweep ranks the same penalized
        // deviance. Excludes the Psi-tier ARD/assignment penalties already
        // accounted for in `loss.total()` (see
        // `analytic_decoder_penalty_value_total`).
        // Extra analytic-penalty energy (#671/#737). Decoder-block penalties and
        // coordinate-tier isometry enter the inner solve but have no `loss.*`
        // representative, so the Laplace criterion must add them explicitly to
        // rank the same penalized deviance the Newton solve descends.
        let extra_penalty_energy = match registry {
            Some(reg) => self
                .reml_extra_penalty_value_total(reg)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?,
            None => 0.0,
        };

        let v = loss.total() + extra_penalty_energy + 0.5 * log_det - occam;
        Ok((v, loss, cache))
    }

    /// The #1037 quotient-dimension invariant: a Laplace normalizer `½log|H|` is
    /// only comparable across ρ at a COMMON quotient (gauge-deflation) dimension.
    /// The first observation pins the expected count; a later match is a no-op.
    ///
    /// A later observation that DIFFERS is, under the K>1 fit, a LEGITIMATE
    /// quotient-dimension event — an atom born, reseeded (the #976 collapse
    /// guards), or rank-reduced moves the number of gauge-flat rows. Because a
    /// deflated direction is lifted to unit stiffness and contributes the
    /// ρ-independent `log 1 = 0` to the evidence, re-anchoring the comparison to
    /// the new dimension is exactly evidence-preserving and keeps every future
    /// cross-ρ comparison consistent — the principled response, not an abort.
    ///
    /// The genuine pathology the guard still catches is a count that NEVER
    /// STABILIZES: re-anchors are bounded by the per-atom structural-event budget
    /// (`k·(reseed_budget+1)+1`), and a runaway quotient dimension past that
    /// bound refuses loudly. This supersedes the prior strict-constant guard and
    /// its ±1 flicker band (#1117) at root — the band was masking exactly the
    /// legitimate K>1 dimension changes this re-anchoring now handles.
    pub(crate) fn record_evidence_gauge_deflation_count(
        &mut self,
        count: usize,
    ) -> Result<(), String> {
        match self.expected_evidence_gauge_deflated_directions {
            Some(expected) if expected == count => Ok(()),
            Some(expected) => {
                // A change in the gauge-deflation count between two evidence
                // factorizations is a legitimate quotient-dimension event under
                // the K>1 fit: an atom can be born, reseeded (the #976 collapse
                // guards), or rank-reduced across the ρ-walk, and each such event
                // moves the number of gauge-flat rows. The #1037 invariant is
                // NOT "the count never changes" — it is "two Laplace normalizers
                // are only comparable at a COMMON quotient dimension". The
                // principled response to a legitimate change is therefore to
                // RE-ANCHOR the comparison to the new dimension (so every future
                // cross-ρ comparison within the optimization is consistent), not
                // to abort the fit. This is exactly evidence-preserving: each
                // gauge-deflated direction is lifted to unit stiffness and
                // contributes the ρ-independent `log 1 = 0` to `½log|H|`, so the
                // converged criterion value is identical whether a given row is
                // counted as deflated or not — only the BOOKKEEPING dimension
                // must agree across a comparison, and re-anchoring restores that.
                //
                // The genuine pathology the guard must still catch is a count
                // that NEVER STABILIZES — an OSCILLATING quotient dimension that
                // re-anchors without converging, signalling a truly ill-posed
                // evidence surface. But the deflation count is NOT a discrete
                // dictionary-level event count: it is the per-ROW-summed number of
                // near-null evidence directions across all N rows (#1217). On real
                // K≥2 activations it is an O(N) quantity that drifts SMOOTHLY and
                // monotonically as the conditioning improves over the ρ-walk
                // (e.g. 171→156→…→113 as smoothing increases) — a benign,
                // evidence-neutral change (each deflated direction contributes the
                // ρ-independent `log 1 = 0` to `½log|H|`, so re-anchoring never
                // moves the criterion value). Charging such a monotone drift
                // against a `k`-sized "structural event" budget was wrong: it
                // counts threshold crossings of a continuous per-row quantity, not
                // atom births/reseeds, so the budget tripped on a perfectly healthy
                // converging K=2 fit (#1217 regression from the #1189/#1190
                // basin-escape fixes, which shifted which rows sit near the
                // deflation floor).
                //
                // The principled discriminator is DIRECTION REVERSALS: a count
                // that drifts one way and settles is benign; a count that bounces
                // up and down without settling is the oscillating-quotient
                // pathology. We therefore charge the re-anchor budget ONLY on a
                // reversal of the change direction, and size the budget by the
                // number of distinct dictionary structural events (births/reseeds)
                // that can each legitimately flip the drift direction. A monotone
                // drift of any length re-anchors freely (it is consistently
                // re-anchored and evidence-neutral); a genuinely oscillating count
                // exhausts the reversal budget and refuses loudly.
                let delta_sign: i8 = if count > expected { 1 } else { -1 };
                let is_reversal = self.evidence_gauge_deflation_last_delta_sign != 0
                    && delta_sign != self.evidence_gauge_deflation_last_delta_sign;
                self.evidence_gauge_deflation_last_delta_sign = delta_sign;
                if is_reversal {
                    self.evidence_gauge_deflation_reanchors += 1;
                }
                let reversal_budget = self
                    .k_atoms()
                    .saturating_mul(
                        SAE_ATOM_COLLAPSE_RESEED_BUDGET
                            + SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET
                            + 1,
                    )
                    .saturating_add(1);
                if self.evidence_gauge_deflation_reanchors > reversal_budget {
                    return Err(format!(
                        "SaeManifoldTerm::reml_criterion: row-gauge evidence deflation count \
                         oscillated (reversed direction {} times, last {expected}->{count}) within \
                         one optimization, exceeding the {reversal_budget}-reversal budget for {} \
                         atoms; the quotient dimension is not stabilizing, refusing to compare \
                         Laplace normalizers",
                        self.evidence_gauge_deflation_reanchors,
                        self.k_atoms()
                    ));
                }
                log::debug!(
                    "SaeManifoldTerm::reml_criterion: per-row evidence deflation count changed \
                     {expected}->{count} (a benign per-row conditioning drift across the ρ-walk; \
                     reversal {}/{reversal_budget}); re-anchoring the Laplace normalizer comparison \
                     to the new dimension",
                    self.evidence_gauge_deflation_reanchors
                );
                self.expected_evidence_gauge_deflated_directions = Some(count);
                Ok(())
            }
            None => {
                self.expected_evidence_gauge_deflated_directions = Some(count);
                Ok(())
            }
        }
    }

    pub(crate) fn is_undamped_evidence_row_non_pd(err: &ArrowSchurError) -> bool {
        matches!(
            err,
            ArrowSchurError::PerRowFactorFailed { reason, .. }
                if reason.contains("H_tt is non-PD at base ridge")
                    && reason.contains("evidence mode preserves the genuine Cholesky")
        )
    }

    /// Drive the inner `(t, β)` Newton solve to the KKT/step-converged optimum
    /// and return the final UNDAMPED (`ridge = 0`) joint-Hessian factor cache.
    ///
    /// The Laplace normaliser `½log|H|` is only the correct REML criterion at
    /// the inner optimum `(t̂, β̂)`, so the criterion must refine the inner state
    /// until either the KKT gradient or the undamped Newton step meets tolerance
    /// before factoring. Crucially, **at the converged optimum the per-row
    /// `H_tt^(i)` blocks are PD**, so the undamped (`ridge = 0`) factorization
    /// succeeds; an off-optimum iterate (e.g. the initial seed, or a state
    /// stopped after only `inner_max_iter` steps) can have an indefinite /
    /// rank-deficient per-row block (`p_out = 1` → rank-1 `JᵀJ`, softmax
    /// assignment-sparsity negative logit curvature) that surfaces
    /// `PerRowFactorFailed` from the undamped `factor_one_row`. Both the dense
    /// (`reml_criterion_with_cache`) and the streaming
    /// (`reml_criterion_streaming_exact`) evidence paths route through this same
    /// driver, so they converge to the identical inner state and their
    /// `ridge = 0` log-determinants stay bit-identical (#847).
    pub(crate) fn converge_inner_for_undamped_logdet(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        rho_fixed: &mut SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        loss: &mut SaeManifoldLoss,
        options: &ArrowSolveOptions,
        refine_progress_extension: bool,
    ) -> Result<ArrowFactorCache, String> {
        // `inner_max_iter == 0` is a genuine FREEZE of the inner `(t, β)` state
        // — a verbatim warm-start reuse, not a convergence request (gam#577/#579,
        // #850). The convergence/refinement loop below MUST NOT run even one
        // Newton step in that case (the old `inner_max_iter.max(1)` floor moved
        // β off the seed), so we factor exactly once at the frozen iterate and
        // return that undamped cache without invoking the stationarity gate.
        // The caller has already run `run_joint_fit_arrow_schur(..., 0, ...)`,
        // which left the seed untouched, so `self` is at the warm-start β here.
        if inner_max_iter == 0 {
            let sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            let factored = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            // The frozen-state Newton step (factored.0, factored.1) is discarded
            // — only the undamped factor cache (factored.2) is consumed for the
            // log-det / selected-inverse traces; β stays at the warm-start seed.
            return Ok(factored.2);
        }
        let mut total_inner_iter = inner_max_iter;
        let accepted_base_refine_iter = inner_max_iter.max(1).saturating_mul(16).max(64);
        let value_probe_base_refine_iter = inner_max_iter.max(1).saturating_mul(4).max(16);
        let base_refine_iter = if refine_progress_extension {
            accepted_base_refine_iter
        } else {
            value_probe_base_refine_iter
        };
        let progress_refine_iter = if refine_progress_extension {
            inner_max_iter.max(1).saturating_mul(64).max(256)
        } else {
            base_refine_iter
        };
        let mut previous_refine_grad_norm: Option<f64> = None;
        let mut saw_refine_progress = false;
        // #1051 — objective-stagnation convergence. On an ill-conditioned
        // penalised bilinear fit (the euclidean / Duchon decoder × latent
        // coordinate system on a trivial shape), the inner Newton crawls: each
        // refine round lowers the penalised objective by a shrinking amount while
        // the KKT gradient and the undamped step stay above their relative
        // tolerances (the near-singular Schur amplifies the step in the
        // weakly-identified decoder direction). The grad-OR-step gate then never
        // fires and the solve is rejected as "did not converge" — the 1e12
        // sentinel. A Newton/LM iterate whose objective has stopped decreasing
        // to within `√εmach` of its scale IS the numerical inner optimum; ranking
        // the Laplace criterion there is correct. We accept that fixed point
        // instead of grinding the budget.
        let entry_loss_total = loss.total();
        let mut previous_loss_total = entry_loss_total;
        let mut refine_rounds: usize = 0;
        // Consecutive stall rounds: counts how many successive refine rounds
        // ended in a stall AND a failed undamped factor.  Once this reaches
        // `SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS` the iterate is at
        // its numerical fixed point and cannot be improved further; returning
        // `Err` here is the same "did not converge" signal that
        // `is_recoverable_value_probe_refusal` already handles, so the outer
        // BFGS treats it as an INFINITY probe and tries a different ρ instead
        // of looping forever burning the extended progress budget.  Without
        // this counter the stagnation handler fell through when the undamped
        // factor failed and the loop kept extending via `saw_refine_progress`
        // from earlier rounds, accumulating minutes of wasted work (#1094).
        let mut consecutive_stall_factor_fail: usize = 0;
        loop {
            let sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            // Evidence-only factorization: the Newton step (Δt, Δβ) is discarded
            // and only the factor cache is consumed — the exact undamped log-det
            // and the selected-inverse traces. As ρ sweeps to extremes (e.g. a
            // wide ARD-α sweep), H_tt is genuinely PD but can be ill-conditioned;
            // the standard Direct guard rejects that to protect Newton-step
            // accuracy, but the log-det is exact from diag(L) regardless of the
            // condition number and the traces only need the (PD) factor. So
            // tolerate the ill-conditioning rejection here (a genuine non-PD pivot
            // still errors). The cache stays undamped at ridge=0, so
            // `arrow_log_det_from_cache` remains exact.
            // The exact KKT stationarity residual is the joint gradient
            // ‖g‖ = √(Σ_i ‖g_t^(i)‖² + ‖g_β‖²), read straight off the assembled
            // system. Unlike the Newton step Δ = H⁻¹g, the gradient is
            // factorisation-independent: it is NOT amplified by an inverse, so a
            // genuinely stationary but ill-conditioned fit (tiny g, possibly large
            // Δ in a flat direction) is correctly recognised as converged. The
            // `with_ill_conditioning_tolerated` Direct factor below documents that
            // its Δ may be inaccurate in exactly those flat directions, so using Δ
            // alone as the convergence gate would falsely reject healthy fits.
            let grad_norm_sq: f64 = sys
                .rows
                .iter()
                .map(|row| row.gt.iter().map(|&v| v * v).sum::<f64>())
                .sum::<f64>()
                + sys.gb.iter().map(|&v| v * v).sum::<f64>();
            let grad_norm = grad_norm_sq.sqrt();
            // Quotient KKT-gradient (#1117): the raw joint gradient retains a
            // persistent small component in the chart-gauge orbit and the
            // rank-deficient decoder β-null even at a stationary fit, so the raw
            // grad gate never clears on a rank-deficient circle and the inner
            // refine loop crawls until the (large) progress budget dies — the
            // 2-min stall. Measure the gradient on the SAME identified quotient
            // the step gate already uses: a fit whose only remaining gradient
            // lives in those flat directions is stationary on the quotient, so
            // ranking the Laplace criterion there is correct. The dense per-row
            // g_t is laid into the `n·q` coordinate layout the gauge basis spans;
            // non-dense/heterogeneous systems fall back to the raw norm.
            let quotient_grad_norm = {
                let n = self.n_obs();
                let q = self.assignment.row_block_dim();
                let dense_len = n.saturating_mul(q);
                let mut grad_ext_coord = Array1::<f64>::zeros(dense_len);
                let mut dense_layout_ok = sys.rows.len() == n;
                if dense_layout_ok {
                    for (row_idx, row) in sys.rows.iter().enumerate() {
                        let base = sys.row_offsets[row_idx];
                        let di = sys.row_dims[row_idx];
                        if base + di > dense_len || row.gt.len() < di {
                            dense_layout_ok = false;
                            break;
                        }
                        for axis in 0..di {
                            grad_ext_coord[base + axis] = row.gt[axis];
                        }
                    }
                }
                if dense_layout_ok {
                    self.quotient_gradient_norm_sq(
                        grad_ext_coord.view(),
                        sys.gb.view(),
                        grad_norm_sq,
                        &rho_fixed.lambda_smooth_vec(),
                    )
                    .map(|v| v.sqrt())
                    .unwrap_or(grad_norm)
                } else {
                    grad_norm
                }
            };
            let iterate_scale = self.inner_iterate_scale();
            // Relative parameter-step tolerance for Δ (well-conditioned charts)
            // and a scaled KKT-gradient tolerance. Convergence is accepted on
            // EITHER a small KKT gradient OR a small undamped Newton step: SAE
            // manifold fits contain gauge-like coordinate/decoder directions (the
            // circle's rotation gauge, decoder column-space rotations) where the
            // shared-block Hessian is near-singular, so the undamped step can stay
            // large in that flat direction even at a genuine stationary point; the
            // gradient, which is not amplified by the inverse, recognises it. With
            // the isometry Gauss-Newton block now a coherent PSD pullback (no
            // indefinite Schur pivot), the inner solve reaches true stationarity,
            // so the gradient tolerance is a standard relative KKT residual rather
            // than the 0.1.154-regression band-aid (3e-3) that masked the
            // non-convergence the indefinite curvature caused.
            let step_tolerance = SAE_MANIFOLD_INNER_STEP_REL_TOL * iterate_scale;
            let grad_tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_scale;
            if !grad_norm_sq.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: undamped inner KKT residual is non-finite \
                     at the inner optimum (‖g‖²={grad_norm_sq}); the joint Hessian \
                     factorisation is degenerate at this ρ"
                ));
            }
            let (delta_t, delta_beta, cache): (Array1<f64>, Array1<f64>, ArrowFactorCache) =
                match solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options) {
                    Ok(factored) => factored,
                    Err(err) if Self::is_undamped_evidence_row_non_pd(&err) => {
                        if grad_norm <= grad_tolerance || quotient_grad_norm <= grad_tolerance {
                            // K>1: the softmax/IBP logit–coordinate Gauss-Newton
                            // cross-terms (H_zt = J_z^T J_t, assembled row-locally from
                            // the assignment JVP × basis JVP) can make a per-row H_tt
                            // indefinite at the TRUE KKT stationary point — when two
                            // atoms' decoders specialise in opposite directions the
                            // Schur complement of the logit block goes negative even
                            // though the priors and the full-joint GN term are PSD.
                            //
                            // The undamped evidence factor already conditions that
                            // block the PRINCIPLED way: `factor_spectral_deflated_
                            // evidence_row` discovers the negative/flat eigen-direction
                            // and stiffens it to UNIT curvature (eigenvalue → +1), so it
                            // contributes a ρ-INDEPENDENT log 1 = 0 to the evidence —
                            // the same quotient pseudo-determinant convention the gauge
                            // (#1037) and data-null (#1117) deflations use. Reaching
                            // THIS arm at stationarity therefore means even the spectral
                            // deflation declined (a non-finite block or a failed
                            // eigendecomposition): the state is genuinely broken, so we
                            // surface the hard refusal and let the outer BFGS treat this
                            // ρ as an INFINITY probe (`is_recoverable_value_probe_
                            // refusal`). We must NOT ridge-damp here: a `+ridge·I`
                            // fallback injects a ρ-dependent ½·log|I + ridge·H_tt⁻¹|
                            // bias into the VALUE that the analytic ρ-gradient (built
                            // for the undamped Laplace log-det) never sees, desyncing
                            // the outer line-search — the multi-atom non-convergence
                            // this fix (#1117) removes.
                            return Err(format!(
                                "SaeManifoldTerm::reml_criterion: stationary undamped \
                                 evidence factorization has a non-PD per-row H_tt block \
                                 that spectral unit-stiffness deflation could not \
                                 condition (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}); \
                                 {err}"
                            ));
                        }
                        let refine_limit = Self::refine_iteration_limit(
                            total_inner_iter,
                            base_refine_iter,
                            progress_refine_iter,
                            previous_refine_grad_norm,
                            grad_norm,
                            saw_refine_progress,
                        );
                        if total_inner_iter >= refine_limit {
                            // #1117/#1118 — pre-stationarity genuinely-indefinite
                            // non-gauge H_tt under K>1 IBP/softmax row-sharing. The
                            // logit × coordinate Gauss-Newton cross term H_zt = J_zᵀJ_t
                            // can drive a shared row's H_tt Schur complement NEGATIVE off
                            // the gauge orbit; the LM-escalated refinement above cannot
                            // always cross the indefinite basin into the PD region within
                            // the descent-extended budget.
                            //
                            // The undamped (ridge=0) evidence factor already conditions
                            // that block the PRINCIPLED way: `factor_spectral_deflated_
                            // evidence_row` discovers the negative/flat eigen-direction
                            // and stiffens it to UNIT curvature (eigenvalue → +1), a
                            // ρ-INDEPENDENT log 1 = 0 evidence contribution — so the
                            // `Ok(factored)` arm above accepts the indefinite block and
                            // returns a finite, monotone-comparable value to the outer
                            // BFGS WITHOUT a ρ-dependent bias. Reaching THIS arm means
                            // even that spectral deflation declined (a non-finite block
                            // or a failed eigendecomposition): the iterate is genuinely
                            // broken, so we surface the hard refusal and let the outer
                            // BFGS treat this ρ as an INFINITY probe.
                            //
                            // We must NOT ridge-damp here: a `+ridge·I` evidence
                            // fallback injects a ρ-dependent ½·log|I + ridge·H_tt⁻¹|
                            // bias into the VALUE that the analytic ρ-gradient (built
                            // for the undamped Laplace log-det) never sees, desyncing
                            // the outer line-search — the multi-atom non-convergence this
                            // fix removes. K=1 (and any already-PD or spectral-deflatable
                            // K>1 row) never reaches this branch.
                            return Err(format!(
                                "SaeManifoldTerm::reml_criterion: undamped evidence \
                                 factorization hit a non-PD per-row H_tt block before KKT \
                                 stationarity (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}) \
                                 and the refinement budget was exhausted after \
                                 {total_inner_iter} inner iterations; {err}"
                            ));
                        }
                        let remaining = refine_limit - total_inner_iter;
                        let refine_iter = inner_max_iter.max(1).min(remaining);
                        saw_refine_progress |=
                            Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
                        previous_refine_grad_norm = Some(grad_norm);
                        *loss = self.run_joint_fit_arrow_schur(
                            target,
                            rho_fixed,
                            registry,
                            refine_iter,
                            learning_rate,
                            ridge_ext_coord,
                            ridge_beta,
                        )?;
                        total_inner_iter += refine_iter;
                        continue;
                    }
                    Err(err) => {
                        return Err(format!("SaeManifoldTerm::reml_criterion: {err}"));
                    }
                };
            // The Laplace normaliser ½log|H| is only the correct REML criterion at
            // the inner optimum (t̂, β̂). Convergence is judged by EITHER a small
            // gradient (KKT stationarity) OR a small undamped Newton step; the
            // solve is only rejected as non-converged when BOTH are large, i.e.
            // the iterate is neither stationary nor about to move negligibly. That
            // disjunction is what keeps an ill-conditioned-but-stationary fit
            // (small g, large Δ) from being rejected while still refusing to rank
            // an off-optimum Laplace criterion that is genuinely mid-flight.
            let step_norm_sq: f64 = delta_t.iter().map(|&v| v * v).sum::<f64>()
                + delta_beta.iter().map(|&v| v * v).sum::<f64>();
            if !step_norm_sq.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: undamped inner residual is non-finite at \
                     the inner optimum (‖Δ‖²={step_norm_sq}, ‖g‖²={grad_norm_sq}); the joint \
                     Hessian factorisation is degenerate at this ρ"
                ));
            }
            let step_norm = step_norm_sq.sqrt();
            let quotient_step_norm_sq = self.quotient_newton_step_norm_sq(
                delta_t.view(),
                delta_beta.view(),
                step_norm_sq,
                &rho_fixed.lambda_smooth_vec(),
            )?;
            let quotient_step_norm = quotient_step_norm_sq.sqrt();
            // Converge on ANY of: the raw KKT gradient (well-conditioned fit),
            // the QUOTIENT KKT gradient (#1117 — rank-deficient fit whose only
            // residual gradient is gauge/null flat-direction crawl), or the
            // quotient Newton step. The quotient-gradient disjunct is what lets
            // a rank-deficient K=1 circle terminate in budget instead of crawling
            // the weakly-identified valley until the refine budget dies.
            if grad_norm <= grad_tolerance
                || quotient_grad_norm <= grad_tolerance
                || quotient_step_norm <= step_tolerance
            {
                return Ok(cache);
            }
            let refine_limit = Self::refine_iteration_limit(
                total_inner_iter,
                base_refine_iter,
                progress_refine_iter,
                previous_refine_grad_norm,
                grad_norm,
                saw_refine_progress,
            );
            if total_inner_iter >= refine_limit {
                // Inner solve did not converge in reml_criterion; the returned
                // Err below carries the full non-convergence diagnostic
                // (gradient / quotient-step norms and tolerances) to the caller.
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ; \
                     neither the KKT gradient ‖g‖={grad_norm:.6e} (tol {grad_tolerance:.6e}) nor \
                     the quotient Newton step ‖Π⊥gauge Δ‖={quotient_step_norm:.6e} \
                     (raw ‖Δ‖={step_norm:.6e}, tol {step_tolerance:.6e}) met \
                     tolerance after {total_inner_iter} inner iterations. Refusing to rank an \
                     off-optimum Laplace criterion."
                ));
            }
            let remaining = refine_limit - total_inner_iter;
            let refine_iter = inner_max_iter.max(1).min(remaining);
            saw_refine_progress |=
                Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
            previous_refine_grad_norm = Some(grad_norm);
            *loss = self.run_joint_fit_arrow_schur(
                target,
                rho_fixed,
                registry,
                refine_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )?;
            total_inner_iter += refine_iter;
            refine_rounds += 1;
            // #1051 — objective-stagnation fixed point. A whole refine round that
            // failed to lower the penalised objective by a meaningful FRACTION of
            // the total since-entry reduction means the Newton/LM iterate is at
            // its numerical optimum: the remaining KKT residual lives in the
            // weakly-identified decoder / gauge directions the near-singular Schur
            // cannot resolve. Ranking the Laplace criterion at this fixed point is
            // correct (the only further motion is cosmetic flat-valley crawl), so
            // accept the current cache instead of refining until the budget dies.
            // Requires a few completed refine rounds (so the fraction baseline is
            // meaningful) but is NOT gated behind the full refine budget — the
            // whole point is to terminate the crawl long before that.
            let new_loss_total = loss.total();
            // Two stagnation signals, both required: (1) the latest refine round
            // contributed a negligible FRACTION of the total objective reduction
            // achieved since entry — the fit has captured essentially all the
            // achievable improvement and is now crawling cosmetically along the
            // weakly-identified valley; (2) the absolute relative decrease is
            // itself tiny. The fraction test is scale- and rate-free (it fires
            // whether the crawl decays fast or slow), so it recognises the
            // over-smoothed / rank-deficient fixed point the bare relative floor
            // misses, while still never firing on a fit that is materially
            // improving round over round.
            let total_improvement = (entry_loss_total - new_loss_total).max(0.0);
            let round_improvement = (previous_loss_total - new_loss_total).max(0.0);
            let objective_scale = previous_loss_total.abs().max(new_loss_total.abs()) + 1.0;
            let relative_decrease = round_improvement / objective_scale;
            let captured_fraction = if total_improvement > 0.0 {
                round_improvement / total_improvement
            } else {
                0.0
            };
            let stalled = new_loss_total.is_finite()
                && relative_decrease.is_finite()
                && (relative_decrease < SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL
                    || captured_fraction < SAE_MANIFOLD_INNER_OBJECTIVE_STALL_FRACTION);
            previous_loss_total = new_loss_total;
            if stalled && refine_rounds >= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS {
                let stationary_sys = self
                    .assemble_arrow_schur(target, rho_fixed, registry)
                    .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
                if let Ok((_dt, _db, stationary_cache)) =
                    solve_arrow_newton_step_with_options(&stationary_sys, 0.0, 0.0, options)
                {
                    return Ok(stationary_cache);
                }
                // Stagnated AND the undamped factor still fails: this is the
                // numerical fixed point of the inner solve under rank-deficient
                // or ill-conditioned geometry (e.g. multi-atom euclidean with
                // near-zero initial latent coords, #1094).  The iterate cannot
                // be improved further at this ρ.  Treat it as "inner solve did
                // not converge" — the same signal `is_recoverable_value_probe_refusal`
                // already handles, causing the outer BFGS to return INFINITY for
                // this ρ probe and try a different one.  Without this early
                // return the stagnation handler fell through and the loop kept
                // burning the extended `progress_refine_iter` budget indefinitely.
                consecutive_stall_factor_fail += 1;
                if consecutive_stall_factor_fail >= SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS {
                    return Err(format!(
                        "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ; \
                         objective stalled for {consecutive_stall_factor_fail} consecutive refine \
                         rounds (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}) and the undamped \
                         evidence factorization failed at each stall point — the iterate is at the \
                         numerical fixed point under rank-deficient geometry (#{consecutive_stall_factor_fail} \
                         stall-factor-fail rounds; refusing to rank an off-optimum Laplace criterion)"
                    ));
                }
            } else {
                consecutive_stall_factor_fail = 0;
            }
        }
    }

    pub(crate) fn refine_iteration_limit(
        total_inner_iter: usize,
        base_refine_iter: usize,
        progress_refine_iter: usize,
        previous_grad_norm: Option<f64>,
        grad_norm: f64,
        saw_refine_progress: bool,
    ) -> usize {
        // Flat affine-gauge valleys can keep crawling productively after the
        // historical base budget. Extend only when the measured KKT residual has
        // shown a real finite round-to-round drop; true stalls end at the base
        // work budget (#968/#1029). Value-order probes pass the base budget as
        // their progress budget, so this branch cannot make probes expensive.
        if total_inner_iter < base_refine_iter {
            return base_refine_iter;
        }
        let making_progress =
            saw_refine_progress || Self::refine_round_made_progress(previous_grad_norm, grad_norm);
        if making_progress && grad_norm.is_finite() {
            progress_refine_iter
        } else {
            base_refine_iter
        }
    }

    pub(crate) fn refine_round_made_progress(
        previous_grad_norm: Option<f64>,
        grad_norm: f64,
    ) -> bool {
        previous_grad_norm
            .is_some_and(|prev| prev.is_finite() && grad_norm.is_finite() && grad_norm < prev)
    }

    pub(crate) fn outer_gradient_arrow_solver<'a>(
        &'a self,
        cache: &'a ArrowFactorCache,
        penalized_gram_scale: &[f64],
    ) -> Result<DeflatedArrowSolver<'a>, OuterGradientError> {
        let Err(conditioning_err) = Self::outer_gradient_conditioning_error(cache) else {
            return Ok(DeflatedArrowSolver::plain(cache));
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(conditioning_err);
        };
        if !(max_pivot.is_finite() && max_pivot > 0.0) {
            return Err(conditioning_err);
        }

        // The conditioning gate has already flagged a near-singular joint Hessian
        // (`conditioning_err`). Below we attempt to attribute that flatness to the
        // closed-form gauge orbit (chart step gauges) plus the penalty-aware
        // decoder-null directions and deflate it. When NO such deflatable
        // direction can be recovered, the flat subspace is genuinely
        // non-identifiable -- a degenerate direction OUTSIDE the gauge orbit -- a
        // diagnosis distinct from the raw pivot-ratio conditioning trip. Both
        // classes are #1273 FD-eligible, but surfacing the gauge-degenerate case
        // as its own [`OuterGradientError::NonIdentifiable`] keeps the diagnostic
        // distinction the FD-eligibility contract is built around.
        let non_identifiable_err = OuterGradientError::NonIdentifiable {
            reason: format!(
                "near-singular joint Hessian with no deflatable gauge/decoder-null \
                 direction (max pivot {max_pivot:.3e})"
            ),
        };

        let full_len = cache.delta_t_len() + cache.k;
        let mut raw_gauges = Vec::new();
        for gauge in self
            .dense_step_gauge_vectors()
            .map_err(OuterGradientError::internal)?
        {
            if gauge.len() != full_len {
                continue;
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            raw_gauges.push(gauge);
        }
        // #1051/#1273: admit the penalty-aware decoder-β null directions as
        // additional deflation candidates. A rank-deficient decoder design
        // (e.g. a euclidean-1D line in a p=2 ambient: decoder column rank 1 of
        // 3) puts a genuine near-null direction of the joint Hessian in the β
        // block, OUTSIDE the closed-form chart gauge orbit. #1273: probing the
        // RAW unit-β basis `e_j` produced an INCOMPLETE candidate set — the
        // true flat direction is the penalised null of `G_k + λ_smooth·S_k`,
        // not an axis-aligned coordinate, so the outer gate rejected trial ρ
        // with a pivot ratio (5.3e-16 < 1e-12) that the inner gate (which
        // already uses `decoder_beta_null_directions(λ_smooth)`) accepts. Use
        // the SAME penalty-aware null directions here, evaluated at the smooth
        // scale the Schur factor used, so the outer and inner gates agree.
        // These full (n·q + beta_dim)-length vectors drop into the same
        // Gram-Schmidt + Rayleigh + Faddeev-Popov path below; the Rayleigh
        // floor still keeps only genuinely flat (sub-floor) directions, so a
        // well-conditioned decoder is unaffected.
        for dir in self
            .decoder_beta_null_directions(penalized_gram_scale)
            .map_err(OuterGradientError::internal)?
        {
            if dir.len() == full_len {
                raw_gauges.push(dir);
            }
        }
        // #1051/#1273: also admit the decoder COLUMN-SPAN null (an unrealised
        // ambient output channel of a rank-deficient decoder), which the
        // channel-free basis-null above structurally cannot represent. The
        // rank-1-decoder-line geometry (e.g. a 1-D euclidean line in p=2
        // ambient: decoder column rank 1 of 2) puts the joint Hessian's
        // sub-floor pivot entirely in one output channel; without this
        // candidate the outer gate had nothing to deflate it with and rejected
        // the trial ρ. The Rayleigh floor below still prunes any candidate that
        // is not genuinely flat against the cached Hessian.
        for dir in self
            .decoder_channel_null_directions()
            .map_err(OuterGradientError::internal)?
        {
            if dir.len() == full_len {
                raw_gauges.push(dir);
            }
        }
        if raw_gauges.is_empty() {
            return Err(non_identifiable_err);
        }

        let mut gauge_span: Vec<Array1<f64>> = Vec::new();
        for mut gauge in raw_gauges {
            for basis in &gauge_span {
                let coeff = gauge.dot(basis);
                for i in 0..gauge.len() {
                    gauge[i] -= coeff * basis[i];
                }
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in gauge.iter_mut() {
                *value *= inv_norm;
            }
            gauge_span.push(gauge);
        }
        if gauge_span.is_empty() {
            return Err(non_identifiable_err);
        }

        let span_rank = gauge_span.len();
        let mut h_span = Array2::<f64>::zeros((span_rank, span_rank));
        for col in 0..span_rank {
            let h_gauge = match apply_cached_arrow_hessian(
                cache,
                gauge_span[col].slice(s![..cache.delta_t_len()]),
                gauge_span[col].slice(s![cache.delta_t_len()..]),
            ) {
                Ok(value) => value,
                // #1451: a shape/dimension mismatch or non-finite intermediate
                // from the Hessian apply is an internal-invariant defect and MUST
                // propagate; only a genuine numeric failure on a finite,
                // correctly-shaped input keeps the FD-eligible conditioning class.
                Err(err) => {
                    return Err(OuterGradientError::classify_arrow_solver_error(
                        &err,
                        conditioning_err.clone(),
                    ));
                }
            };
            let h_flat = flatten_arrow_parts(h_gauge.t.view(), h_gauge.beta.view());
            for row in 0..span_rank {
                h_span[[row, col]] = gauge_span[row].dot(&h_flat);
            }
        }
        for row in 0..span_rank {
            for col in 0..row {
                let sym = 0.5 * (h_span[[row, col]] + h_span[[col, row]]);
                h_span[[row, col]] = sym;
                h_span[[col, row]] = sym;
            }
        }
        // #1451: a non-finite entry in the projected gauge Hessian is an
        // internal-invariant defect (a NaN/Inf intermediate leaked into the
        // span), not a conditioning failure — it MUST propagate rather than be
        // masked behind an FD descent. Guard finiteness BEFORE the eigh so only a
        // genuine decomposition failure on a finite, correctly-shaped matrix keeps
        // the FD-eligible conditioning class.
        if !h_span.iter().all(|v| v.is_finite()) {
            return Err(OuterGradientError::internal(format!(
                "outer_gradient_arrow_solver: non-finite entry in projected gauge \
                 Hessian (h_span is {span_rank}x{span_rank})"
            )));
        }
        let (evals, evecs) = h_span
            .eigh(Side::Lower)
            .map_err(|_| conditioning_err.clone())?;
        let strict_gauge_floor = SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR * max_pivot;
        let mut orthonormal: Vec<Array1<f64>> = Vec::new();
        for eig_idx in 0..evals.len() {
            let rayleigh = evals[eig_idx];
            if !(rayleigh.is_finite() && rayleigh <= strict_gauge_floor) {
                continue;
            }
            let mut direction = Array1::<f64>::zeros(full_len);
            for basis_idx in 0..span_rank {
                let coeff = evecs[[basis_idx, eig_idx]];
                for row in 0..full_len {
                    direction[row] += coeff * gauge_span[basis_idx][row];
                }
            }
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in direction.iter_mut() {
                *value *= inv_norm;
            }
            orthonormal.push(direction);
        }
        if orthonormal.is_empty() {
            // #1273/#1440: the conditioning gate has ALREADY certified a
            // near-singular joint Hessian (`conditioning_err`), so a genuine flat
            // direction exists inside the assembled gauge/decoder-null span even
            // when no projected-Hessian eigenvector cleared the strict or the
            // `fallback_gauge_floor` Rayleigh band. Rather than declining
            // (which historically routed the outer step to a finite-difference
            // descent direction — the FD instrument #1440 removes), deflate the
            // SMALLEST-Rayleigh eigenvector of the projected gauge Hessian
            // UNCONDITIONALLY. That eigenvector is the least-curvature member of
            // the validated gauge span (a Faddeev-Popov gauge candidate), so the
            // Tikhonov stiffness `max_pivot` in `from_orthonormal_gauges` bounds
            // its contribution at the Hessian scale and the components orthogonal
            // to it are byte-for-byte the plain analytic inverse solve. This keeps
            // the descent direction fully ANALYTIC (a projected/damped gradient),
            // never a differenced value path.
            let mut best_idx = None;
            let mut best_rayleigh = f64::INFINITY;
            for eig_idx in 0..evals.len() {
                let rayleigh = evals[eig_idx];
                if rayleigh.is_finite() && rayleigh < best_rayleigh {
                    best_idx = Some(eig_idx);
                    best_rayleigh = rayleigh;
                }
            }
            if let Some(eig_idx) = best_idx {
                let mut direction = Array1::<f64>::zeros(full_len);
                for basis_idx in 0..span_rank {
                    let coeff = evecs[[basis_idx, eig_idx]];
                    for row in 0..full_len {
                        direction[row] += coeff * gauge_span[basis_idx][row];
                    }
                }
                let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
                if norm_sq.is_finite() && norm_sq > 1.0e-24 {
                    let inv_norm = norm_sq.sqrt().recip();
                    for value in direction.iter_mut() {
                        *value *= inv_norm;
                    }
                    orthonormal.push(direction);
                }
            }
        }
        if orthonormal.is_empty() {
            return Err(non_identifiable_err);
        }

        // Quotient-geometry gauge fixing: add stiffness only along the closed-form
        // gauge orbit (Faddeev-Popov style). Components orthogonal to that orbit
        // are identical to the original inverse solve, while gauge components are
        // bounded at the Hessian scale `max_pivot`.
        // #1451: a shape/length mismatch or non-finite stiffness/intermediate in
        // the deflated-solver assembly is an internal-invariant defect and MUST
        // propagate; only a genuine near-singular gauge Woodbury/back-solve keeps
        // the FD-eligible conditioning class.
        DeflatedArrowSolver::from_orthonormal_gauges(cache, orthonormal, max_pivot)
            .map_err(|err| OuterGradientError::classify_arrow_solver_error(&err, conditioning_err))
    }

    pub(crate) fn outer_gradient_conditioning_error(
        cache: &ArrowFactorCache,
    ) -> Result<(), OuterGradientError> {
        let pivot = arrow_factor_min_pivot(cache);
        let Some(min_pivot) = pivot.min_pivot else {
            return Err(OuterGradientError::IllConditioned {
                reason: "joint Hessian numerically singular (no cached Cholesky pivots)"
                    .to_string(),
            });
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(OuterGradientError::IllConditioned {
                reason: "joint Hessian numerically singular (no cached Cholesky pivot scale)"
                    .to_string(),
            });
        };
        let ratio = min_pivot / max_pivot;
        if min_pivot.is_finite()
            && max_pivot.is_finite()
            && max_pivot > 0.0
            && ratio.is_finite()
            && ratio >= SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR
        {
            return Ok(());
        }
        Err(OuterGradientError::IllConditioned {
            reason: format!(
                "joint Hessian numerically singular (min/max pivot ratio {ratio:.3e} < floor {floor:.3e}; min pivot {min_pivot:.3e}, max pivot {max_pivot:.3e})",
                floor = SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR,
            ),
        })
    }

    /// Smoothing-penalty Occam normalizer `−½ Σ_k r_k·rank(S_k)·log λ_smooth`
    /// PLUS the profiled-frame evidence-dimension term `½ Σ_k r_k·(p−r_k)·log
    /// λ_smooth` (issue #972).
    ///
    /// On the full-`B` path every atom's frame rank `r_k == p`, so the first
    /// piece reduces to the historical `½ p·(Σ rank S_k)·log λ_smooth` and the
    /// Grassmann term is zero — bit-for-bit unchanged. When a frame is active the
    /// decoder coordinates `C_k` carry the `⊗ I_{r_k}` Kronecker structure (the
    /// smoothing penalty `S_k` now acts on `r_k` channels, not `p`), so the
    /// penalty-logdet normalizer uses `r_k·rank(S_k)`; and the `r_k·(p−r_k)`
    /// frame degrees of freedom profiled OUT of the border are counted explicitly
    /// in the Laplace dimension accounting (evidence honesty) so the criterion
    /// cannot buy a free evidence boost by hiding decoder freedom in the frame.
    pub(crate) fn reml_occam_term(&self, rho: &SaeManifoldRho) -> Result<f64, String> {
        // #1556: λ_smooth is per-atom, so the Occam penalty normalizer and the
        // profiled-frame evidence-dimension term are both per-atom sums, each
        // atom `k` weighted by its own `log λ_smooth[k]`. With a uniform
        // (broadcast) vector this is bit-for-bit the historical global form.
        let mut acc = 0.0_f64;
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            // Penalized decoder dimension: `r_k` coordinate channels carry the
            // `S_k` roughness penalty (full-`B` path ⇒ `r_k == p`).
            let penalized_channel_dim = atom.border_frame_rank() * rank_s;
            // Profiled Grassmann dimensions enter the Laplace evidence dimension
            // count with the OPPOSITE sign of the penalty Occam term (they are
            // free, unpenalized-by-`S` profiled directions), so `−occam` adds
            // `+½ r(p−r) log λ_k` to the criterion `V` — the honesty correction.
            let frame_dim = atom.frame_manifold_dimension();
            let log_lambda = rho.log_lambda_smooth[atom_idx];
            acc += 0.5 * ((penalized_channel_dim as f64) - (frame_dim as f64)) * log_lambda;
        }
        // `V = … − occam`, so the net occam SUBTRACTS the penalty normalizer and
        // ADDS the frame-dimension count after the caller's `− occam`.
        Ok(acc)
    }

    /// Per-atom derivative `∂(occam)/∂log λ_smooth[k]` (#1556): atom `k`'s entry
    /// is `½·(r_k·rank(S_k) − frame_dim_k)`, matching the per-atom Occam term in
    /// [`Self::reml_occam_term`]. Returns one entry per atom in atom order.
    pub(crate) fn reml_occam_log_lambda_smooth_derivative(&self) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(self.atoms.len());
        for atom in &self.atoms {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            let penalized_channel_dim = atom.border_frame_rank() * rank_s;
            let frame_dim = atom.frame_manifold_dimension();
            out.push(0.5 * ((penalized_channel_dim as f64) - (frame_dim as f64)));
        }
        Ok(out)
    }

    pub fn reml_criterion_streaming_exact(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        let mut rho_fixed = rho.clone();
        let mut loss = self.run_joint_fit_arrow_schur(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        // Drive the inner (t, β) state to the SAME KKT/step-converged optimum the
        // dense `reml_criterion_with_cache` reaches before factoring. At that
        // optimum the per-row `H_tt^(i)` blocks are PD, so the undamped
        // (`ridge_t = 0`) streaming factorization in `streaming_exact_arrow_log_det`
        // succeeds — without this, a state stopped after only `inner_max_iter`
        // steps can leave a rank-deficient / indefinite row block (`p_out = 1` →
        // rank-1 `JᵀJ`, softmax negative-logit curvature) that surfaces
        // `PerRowFactorFailed` at base ridge 0. Sharing the driver also keeps the
        // streaming and dense log-determinants bit-identical (#847).
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        // The dense factor cache from convergence is surplus here — the streaming
        // path recomputes the (bit-identical) log-determinant chunk-by-chunk in
        // `streaming_exact_arrow_log_det` to bound peak memory — so it is dropped.
        let converged_cache = self.converge_inner_for_undamped_logdet(
            target,
            rho,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &options,
            true,
        )?;
        drop(converged_cache);
        let log_det = self.streaming_exact_arrow_log_det(target, rho, registry)?;
        let occam = self.reml_occam_term(rho)?;
        // Extra analytic-penalty energy (#671/#737), matching the full-batch
        // `reml_criterion_with_cache` path so streaming and dense criteria rank
        // the identical penalized objective.
        let extra_penalty_energy = match registry {
            Some(reg) => self
                .reml_extra_penalty_value_total(reg)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion_streaming_exact: {err}"))?,
            None => 0.0,
        };
        Ok((
            loss.total() + extra_penalty_energy + 0.5 * log_det - occam,
            loss,
        ))
    }

    pub fn streaming_exact_arrow_log_det(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<f64, String> {
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::streaming_exact_arrow_log_det: target must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        let plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if plan.estimated_dense_schur_bytes > plan.in_core_budget_bytes {
            return Err(format!(
                "SaeManifoldTerm::streaming_exact_arrow_log_det: predicted dense reduced Schur {} bytes exceeds budget {} bytes; cost-only matrix-free route is required",
                plan.estimated_dense_schur_bytes, plan.in_core_budget_bytes
            ));
        }
        let n_total = self.n_obs();
        let chunk_size = plan.chunk_size.min(n_total.max(1));
        // #972 / #977 T1: the reduced β-Schur is over the FACTORED border when
        // frames are active (each chunk inherits the frames via
        // `materialize_chunk`, so every `chunk_schur` is `border_dim²`), matching
        // the dense path's factored log-det. Full-`B` ⇒ `border_dim == beta_dim`.
        let border_dim = if self.frames_active() {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        let mut schur_acc = Array2::<f64>::zeros((border_dim, border_dim));
        let mut log_det_tt = 0.0_f64;
        // #1038 cross-row IBP Woodbury accumulators. `M = Uᵀ H₀'⁻¹ U` is
        // chunk-additive in `M0 = Σ Uᵢᵀ Aᵢ⁻¹ Uᵢ` and `W = Σ Bᵢᵀ Aᵢ⁻¹ Uᵢ`
        // (`A = H₀'` block-diagonal, `U` row-supported), closed against the
        // GLOBAL reduced Schur `S = schur_acc` after the loop. `None` for every
        // non-IBP (softmax / JumpReLU) term, where the streaming log-det is
        // exactly the bare `log_det_tt + log_det_schur` as before.
        let mut wood_m0: Option<Array2<f64>> = None;
        let mut wood_w: Option<Array2<f64>> = None;
        let mut wood_d: Option<Array1<f64>> = None;
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let mut start = 0usize;
        while start < n_total {
            let end = (start + chunk_size).min(n_total);
            let penalty_scale = (end - start) as f64 / n_total as f64;
            let chunk_logits = self.assignment.logits.slice(s![start..end, ..]).to_owned();
            let chunk_coords: Vec<Array2<f64>> = self
                .assignment
                .coords
                .iter()
                .map(|coord| coord.as_matrix().slice(s![start..end, ..]).to_owned())
                .collect();
            let mut chunk = self.materialize_chunk(chunk_logits, chunk_coords)?;
            // #1117 — rank deficiency is removed at the basis layer at fit entry
            // (`reduce_atoms_to_data_supported_rank`), so each chunk inherits the
            // already-reduced full-rank atoms via `materialize_chunk`; there are
            // no global deflation projectors to propagate.
            // #991: chunk terms inherit the row's design honesty weight slice
            // (global mean-1 normalization preserved — NOT re-normalized per
            // chunk — so the per-chunk sums reconstruct the global weighted
            // objective exactly).
            if let Some(w) = self.row_loss_weights.as_deref() {
                chunk.row_loss_weights = Some(w[start..end].to_vec());
            }
            let z_chunk = target.slice(s![start..end, ..]);
            let sys = chunk
                .assemble_arrow_schur_scaled(z_chunk, rho, registry, penalty_scale)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len().max(1));
            let (chunk_log_det_tt, chunk_schur, chunk_wood) = streaming
                .reduced_schur_log_det_tt_woodbury(0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            log_det_tt += chunk_log_det_tt;
            for row in 0..border_dim {
                for col in 0..border_dim {
                    schur_acc[[row, col]] += chunk_schur[[row, col]];
                }
            }
            if chunk_wood.is_some() && chunk_size < n_total {
                // The cross-row IBP empirical mass `M_k = Σ_i z_ik` couples ALL
                // rows, so the per-row `H₀'` diagonal (`score_derivative_k(M_k)`)
                // and the column coefficient `d_k = w·s'_k(M_k)` are only exact
                // when every row is assembled together — a SINGLE chunk. Under a
                // genuine multi-chunk pass each chunk would see a partial mass and
                // the Woodbury (and the bare per-row log-det) would be inexact, so
                // refuse loudly and route to the dense resident path rather than
                // return a silently-wrong evidence. The streaming log-det only
                // runs when the dense reduced Schur fits budget, so the single-
                // chunk regime is the common case; this guards the rest.
                return Err(
                    "SaeManifoldTerm::streaming_exact_arrow_log_det: exact cross-row IBP \
                     Woodbury evidence requires a single-chunk pass (the empirical mass \
                     M_k = Σ_i z_ik couples all rows); this shape needs >1 chunk. Route \
                     IBP-active large-n fits through the dense resident \
                     ArrowFactorCache::arrow_log_det."
                        .to_string(),
                );
            }
            if let Some(cw) = chunk_wood {
                wood_m0 = Some(match wood_m0.take() {
                    Some(mut acc) => {
                        acc += &cw.m0;
                        acc
                    }
                    None => cw.m0,
                });
                wood_w = Some(match wood_w.take() {
                    Some(mut acc) => {
                        acc += &cw.w;
                        acc
                    }
                    None => cw.w,
                });
                // `D = diag(d_k)` is per-atom; identical across chunks for a
                // single-chunk evidence pass (the regime the streaming log-det
                // runs in — the dense reduced Schur must fit budget here), where
                // it equals the global mass-derived `cross_row_d`.
                wood_d = Some(cw.d);
            }
            start = end;
        }
        let log_det_schur = StreamingArrowSchur::reduced_schur_log_det(&schur_acc, &options)
            .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
        let mut total = log_det_tt + log_det_schur;
        // #1038/#1225: close the exact cross-row IBP Woodbury correction
        // `log det(I_R + D Uᵀ H₀'⁻¹ U)` so the streaming evidence equals the
        // dense `arrow_log_det_from_cache` (which adds the SAME term). Without
        // it the streaming criterion would silently drop the entire cross-row
        // coupling and disagree with the dense path by exactly `log|C|`.
        if let (Some(m0), Some(w), Some(d)) = (wood_m0, wood_w, wood_d) {
            let correction = streaming_cross_row_woodbury_log_det(&schur_acc, &m0, &w, &d)
                .map_err(|err| {
                    format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}")
                })?
                .ok_or_else(|| {
                    "SaeManifoldTerm::reml_criterion: cross-row IBP joint Hessian is non-PD at \
                     this ρ; evidence Laplace log-det undefined (infeasible ρ probe)"
                        .to_string()
                })?;
            total += correction;
        }
        Ok(total)
    }

    /// Per-atom, per-axis coordinate sum-of-squares `‖t_kj‖² = Σ_i t_{i,k,j}²`.
    ///
    /// This is the data-fit sufficient statistic for the ARD precision update
    /// (the numerator-side `‖t‖²` of the deleted `α = n/‖t‖²` rule). Returned
    /// per atom as an `Array1` of length `d_k`.
    ///
    /// On a *periodic* (Circle) axis the relevant statistic is the von-Mises
    /// energy-equivalent `Σ_i 2/α·V(t_i) = Σ_i (2/κ²)(1−cos κ t_i)` (independent
    /// of α), so that `½·α·sumsq == Σ_i V(t_i)` matches `ard_value`. This keeps
    /// the Mackay/Fellner–Schall fixed point `α ← n / (sumsq + tr H⁻¹)`
    /// consistent with the actual periodic prior energy rather than the
    /// origin-dependent raw `t²`.
    pub(crate) fn ard_coord_sumsq(&self) -> Vec<Array1<f64>> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for coord in &self.assignment.coords {
            let d = coord.latent_dim();
            let periods = coord.effective_axis_periods();
            let mut sq = Array1::<f64>::zeros(d);
            for row in 0..coord.n_obs() {
                let t = coord.row(row);
                for axis in 0..d {
                    // `sq_equiv` is independent of `alpha`; pass 1.0.
                    sq[axis] += ArdAxisPrior::eval(1.0, t[axis], periods[axis]).sq_equiv;
                }
            }
            out.push(sq);
        }
        out
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹) =
    /// Σ_i [(H⁻¹)_tt]_{(i,k,j),(i,k,j)}` from the converged factor cache.
    ///
    /// `cache.latent_block_inverse_diagonal()` returns the diagonal of the
    /// latent block `(H⁻¹)_tt` in the cache's compact per-row `delta_t`
    /// layout (length `row_offsets[N]`); each per-row block is laid out as
    /// `[logit scalars…, then per-active-atom coord axes…]`. This routine
    /// sums those diagonal entries over the coord positions belonging to each
    /// `(atom k, axis j)` across all observation rows where atom `k` is active.
    ///
    /// `self.last_row_layout` must be the layout from the *same* assemble that
    /// produced `cache`:
    /// - `Some(layout)`: compact active-set mode (JumpReLU / large-K
    ///   softmax-IBP truncation). For row `i`, atom `k`'s position in the
    ///   active list gives its compact coord-block start `coord_starts[i][pos]`;
    ///   inactive atoms contribute 0 (the prior dominates there anyway).
    /// - `None`: dense full-support layout, uniform row dim
    ///   `q = assignment_dim + Σ d_k`; atom `k`'s coord block sits at the
    ///   fixed full-row offset `coord_offsets[k]` after the assignment chart.
    ///
    /// This `tr_kj(H⁻¹)` is exactly the posterior-variance term the deleted
    /// `α = n/‖t‖²` rule dropped; the corrected Mackay/Fellner-Schall fixed
    /// point is `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`.
    pub(crate) fn ard_inverse_traces(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let inv_diag = cache.latent_block_inverse_diagonal()?;
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    pub(crate) fn ard_log_precision_explicit_derivatives(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Array1<f64>>, String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atoms but term has {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs() as f64;
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            let mut atom_out = Array1::<f64>::zeros(rho.log_ard[atom_idx].len());
            if rho.log_ard[atom_idx].is_empty() {
                out.push(atom_out);
                continue;
            }
            if rho.log_ard[atom_idx].len() != d {
                return Err(format!(
                    "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                    rho.log_ard[atom_idx].len()
                ));
            }
            let periods = coord.effective_axis_periods();
            for axis in 0..d {
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom_idx][axis]);
                let period = periods[axis];
                let mut energy_deriv = 0.0_f64;
                for row in 0..coord.n_obs() {
                    let t = coord.row(row)[axis];
                    energy_deriv += ArdAxisPrior::eval(alpha, t, period).value;
                }
                let normalizer_deriv = match period {
                    None => -0.5 * n,
                    Some(p) => {
                        let kappa = std::f64::consts::TAU / p;
                        let eta = alpha / (kappa * kappa);
                        // d/d(log α) of `n[-η + log I0(η)]` = `n η (I1/I0 - 1)`.
                        // The ratio is computed without forming `e^{η}`, so it
                        // stays finite for large `η` instead of the `inf/inf =
                        // NaN` that `bessel_i1(η)/bessel_i0(η)` produces (#1113).
                        let ratio = bessel_i0_log_and_ratio(eta).1;
                        n * eta * (-1.0 + ratio)
                    }
                };
                atom_out[axis] = energy_deriv + normalizer_deriv;
            }
            out.push(atom_out);
        }
        Ok(out)
    }

    pub(crate) fn ard_log_precision_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        // RAW selected-inverse diagonal: the per-axis diagonal contraction uses
        // the DEFLATED inverse; the full kept-subspace + rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per (row, axis)
        // afterwards via the Daleckii–Krein helper. Each ARD ρ-component
        // `(atom k, axis)` differentiates a SINGLE coordinate-slot diagonal entry,
        // so its `D` is the rank-one `hess·e_s e_sᵀ` at that local slot `s`.
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            // Per-row selected-inverse t-block, built once (only when deflated).
            let inv_vv = if dirs.is_empty() {
                None
            } else {
                let mut m = Array2::<f64>::zeros((q, q));
                for col in 0..q {
                    let mut rhs_t = Array1::<f64>::zeros(total_t);
                    let rhs_beta = Array1::<f64>::zeros(cache.k);
                    rhs_t[row_base + col] = 1.0;
                    let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                        ArrowSchurError::SchurFactorFailed { reason: err }
                    })?;
                    for r in 0..q {
                        m[[r, col]] = solved.t[row_base + r];
                    }
                }
                Some(m)
            };
            // Correction for one local coordinate slot `s` with curvature `hess`.
            let slot_correction = |s: usize, hess: f64| -> f64 {
                let Some(iv) = inv_vv.as_ref() else {
                    return 0.0;
                };
                if s >= q || hess == 0.0 {
                    return 0.0;
                }
                let mut d = Array2::<f64>::zeros((q, q));
                d[[s, s]] = hess;
                Self::deflation_block_correction(iv, &d, dirs, spectrum)
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            let hess = prior.hess.max(0.0);
                            let s = block_start + axis;
                            traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                            traces[k][axis] -= 0.5 * slot_correction(s, hess);
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            let hess = prior.hess.max(0.0);
                            let s = block_start + axis;
                            traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                            traces[k][axis] -= 0.5 * slot_correction(s, hess);
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Per-atom decoder-smoothness penalty quadratic form (#1556): entry `k` is
    /// the λ-free `<B_k, ½(S_k+S_kᵀ)·B_k> = Σ_oc B_k[:,oc]ᵀ S_k B_k[:,oc]`, the
    /// per-atom denominator of atom `k`'s λ_smooth Fellner-Schall update. The sum
    /// over atoms is `βᵀ(⊕_k S_k ⊗ I_p)β`, the un-scaled total penalty energy.
    /// `S_k` is symmetrised defensively (as the assembler does); the per-atom
    /// `½(S+Sᵀ)·B_k` GEMMs ride the multi-GPU batched smoothness GEMM with an
    /// exact per-atom CPU fallback.
    pub(crate) fn decoder_smoothness_quadratic_form_per_atom(&self) -> Vec<f64> {
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, true);
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        for (atom_idx, (atom, sb)) in self.atoms.iter().zip(sb_all.iter()).enumerate() {
            per_atom[atom_idx] = (&atom.decoder_coefficients * sb).sum();
        }
        per_atom
    }

    /// Per-atom effective penalized dof of the decoder smoothness penalty
    /// (#1556): entry `k` is `tr(S_β⁻¹ · M_k)` with `M_k = (λ_smooth[k]·S_k) ⊗ I`
    /// and `S_β⁻¹ = (H⁻¹)_ββ` the Schur-complement inverse, each atom scaled by
    /// its OWN `lambda_smooth[atom_idx]`. Built on
    /// [`ArrowFactorCache::schur_inverse_apply`]: column `(k,μ,oc)` of `M_k` is
    /// `λ_k·S_k[:,μ] ⊗ e_oc` (sparse), so we apply `S_β⁻¹` to that K-vector and
    /// read back `result[col]`. The total edf is the sum of the returned vector
    /// (a uniform/broadcast λ reproduces the historical global trace).
    pub(crate) fn decoder_smoothness_effective_dof_per_atom(
        &self,
        cache: &ArrowFactorCache,
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, ArrowSchurError> {
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (offsets, out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) = if frames_active {
            let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                self.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (self.beta_offsets(), Box::new(move |_k: usize| p))
        };
        let k = cache.k;
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            let lambda = lambda_smooth[atom_idx];
            let mut trace = 0.0_f64;
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda * s_nu_mu;
                    }
                    let z = cache.schur_inverse_apply(m_col.view())?;
                    trace += z[col];
                }
            }
            per_atom[atom_idx] = trace;
        }
        Ok(per_atom)
    }

    /// Per-atom effective penalized dof via the deflated solver (#1556): entry
    /// `k` is `tr((H⁻¹)_ββ · M_k)` for `M_k = (λ_smooth[k]·S_k) ⊗ I`, each atom
    /// scaled by its OWN `lambda_smooth[atom_idx]`. The total is the sum.
    pub(crate) fn decoder_smoothness_effective_dof_with_solver_per_atom(
        &self,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        lambda_smooth: &[f64],
    ) -> Result<Vec<f64>, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache's β block is the FACTORED border when frames
        // are active (`cache.k == factored_border_dim`), so the smoothness edf
        // trace `tr((H⁻¹)_ββ · M)` is taken over the same factored layout, with
        // `M = ⊕_k (λ_k S_k) ⊗ I_{r_k}` at the factored offsets (the `U_kᵀU_k = I`
        // collapse means the per-coordinate-channel penalty is `λ_k S_k`, exactly
        // as in the full-`B` `⊗ I_p` case but with `r_k` channels). On the
        // full-`B` path `frames_active` is false: `out_dim_k = p`, the offsets
        // are `beta_offsets`, and this is bit-for-bit the historical trace.
        let frames_active = self.frames_active();
        let (offsets, out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) = if frames_active {
            let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                self.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (self.beta_offsets(), Box::new(move |_k: usize| p))
        };
        let k = cache.k;
        let mut per_atom = vec![0.0_f64; self.atoms.len()];
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            let lambda = lambda_smooth[atom_idx];
            let mut trace = 0.0_f64;
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    // M[:,col] = λ_k · S_k[:,mu] ⊗ e_oc (nonzero at off+ν·r+oc).
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda * s_nu_mu;
                    }
                    let zero_t = Array1::<f64>::zeros(cache.delta_t_len());
                    let z = solver.solve(zero_t.view(), m_col.view())?.beta;
                    trace += z[col];
                }
            }
            per_atom[atom_idx] = trace;
        }
        Ok(per_atom)
    }

    pub(crate) fn assignment_log_strength_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<f64, String> {
        let k_atoms = self.k_atoms();
        // #1038 softmax: `H` carries the DENSE entropy block, and since the
        // entropy curvature scales linearly with `λ_sparse = exp(ρ)`,
        // `∂H/∂ρ = H_entropy` (the full dense per-row block, not just its
        // diagonal). The trace `½ tr(H⁻¹ ∂H/∂ρ)` must therefore contract the
        // dense `∂H/∂ρ` against the per-row selected-inverse BLOCK, mirroring the
        // dense `log|H|` and θ-adjoint — a diagonal-only contraction would
        // desync the ρ-gradient from the criterion. The assembled majorizer
        // `D = diag(Σ_j|H_kj|)` is itself DIAGONAL (#1419), so the contraction
        // reduces to `½ Σ_slot (H⁻¹)_{slot,slot}·D_atom`. On the dense `None`
        // layout the logit slot equals the atom position; on the compact
        // softmax top-`k` layout (#1408/#1409) the slots are the row's active
        // atoms — the SAME `D_atom` (full-`K` abs-row-sum) the assembly wrote.
        if let AssignmentMode::Softmax {
            temperature,
            sparsity,
        } = self.assignment.mode
        {
            if k_atoms <= 1 {
                return Ok(0.0);
            }
            let inv_tau = 1.0 / temperature;
            let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
            let penalty = gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                k_atoms,
                temperature,
            );
            // Softmax uses the reduced K−1 free-logit chart on the dense layout
            // (last reference logit fixed); the compact layout carries one slot
            // per active atom. The diagonal selected inverse gives each slot's
            // (H⁻¹)_{slot,slot}.
            let assignment_dim = self.assignment.assignment_coord_dim();
            // Kept-subspace inverse diagonal: the deflated inverse assigns
            // `1/λ̃ = 1` to each per-row UNIT-stiffness direction `vᵢ`, so a raw
            // diagonal `D` contraction would spuriously add `½ Σ_i vᵢᵀ D vᵢ` (a
            // ρ-independent direction must add 0). `latent_inverse_diagonal_kept`
            // removes that per-row deflated diagonal centrally.
            let inv_diag = solver
                .latent_inverse_diagonal_kept()
                .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
            let mut trace = 0.0_f64;
            for row in 0..self.n_obs() {
                let row_base = cache.row_offsets[row];
                // ∂(scale·D)/∂ρ = scale·D (linear in λ_sparse = eᵖ) — the SAME
                // operator the assembly and θ-adjoint differentiate.
                match self.last_row_layout {
                    Some(ref layout) => {
                        // #1410: the compact adjoint reads `D_kk` only for this
                        // row's `≤ top_k` active atoms, so compute those entries
                        // directly from the softmax row `a` via the active-only
                        // Gershgorin helper — no full-`K` `row_logits` copy and no
                        // full-`K` `d` vector. `a` itself is the irreducible `O(K)`
                        // softmax normalisation, computed once per row and shared
                        // across the row's active slots.
                        let a = crate::assignment::softmax_row(
                            self.assignment.logits.row(row),
                            temperature,
                        );
                        let a = a.as_slice().expect("softmax row must be contiguous");
                        let m = softmax_majorizer_log_mean(a);
                        for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                            let d_atom =
                                active_softmax_gershgorin_majorizer_entry(a, atom, m, scale);
                            trace += inv_diag[row_base + pos] * d_atom;
                        }
                    }
                    None => {
                        // Dense layout genuinely contracts every free logit slot's
                        // `D_kk`, so the full-`K` `d` is intrinsic here; keep the
                        // single-source dense majorizer call.
                        let row_logits: Vec<f64> = (0..k_atoms)
                            .map(|k| self.assignment.logits[[row, k]])
                            .collect();
                        let d = penalty.psd_majorizer_abs_row_sums(&row_logits, scale);
                        let q = cache.row_dims[row];
                        let logit_dim = assignment_dim.min(q);
                        for atom in 0..logit_dim {
                            trace += inv_diag[row_base + atom] * d[atom];
                        }
                    }
                }
            }
            return Ok(0.5 * trace);
        }
        let hdiag = assignment_prior_log_strength_hdiag(&self.assignment, rho)?;
        if hdiag.is_empty() {
            return Ok(0.0);
        }
        // RAW selected-inverse diagonal: the per-row diagonal contraction uses the
        // DEFLATED inverse; the full kept-subspace + β-Schur/rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per row afterwards
        // (`deflation_block_correction`), exactly as the data trace does. The
        // cross-row off-diagonal pass below contracts only DISTINCT rows `i ≠ j`,
        // off any single-row `vᵢ`'s support, so it needs no deflation correction.
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
        let assignment_dim = self.assignment.assignment_coord_dim();
        let total_t = cache.delta_t_len();
        // #932 FRONT C: row-local Takahashi selected inverse on the plain arrow
        // for the per-row deflation correction below (the diagonal trace already
        // uses the cheap `latent_inverse_diagonal`); gauge / cross-row Woodbury
        // fall back to the per-row full-system `solve` loop.
        let fast_selected = solver.plain_selected_inverse_available();
        let selected_beta_inv = if fast_selected && cache.k > 0 {
            solver
                .beta_inv()
                .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        // #1416 cross-row IBP source: the per-row block that the deflation
        // factorizes is the NO-SELF base `H₀'` — the rank-one self curvature
        // `d_k·J_ik²` is DOWNDATED from each logit diagonal and re-applied through
        // the Woodbury carrier. The full-`H` diagonal contraction below still uses
        // the full `hdiag` (which carries that self term), but the per-row
        // DEFLATION correction must use `(∂H₀'/∂ρ)_tt`, i.e. `hdiag` MINUS the
        // downdated self term — otherwise the Daleckii–Krein correction
        // mis-attributes the (un-deflated) Woodbury self curvature's derivative to
        // the deflated subspace. For non-IBP modes there is no Woodbury source and
        // the self term is `0` (the deflated block IS the full block).
        let cross_channels = if self.last_row_layout.is_none() {
            ibp_assignment_third_channels(&self.assignment, rho)?
        } else {
            None
        };
        let learnable_alpha = matches!(
            self.assignment.mode,
            AssignmentMode::IBPMap {
                learnable_alpha: true,
                ..
            }
        );
        let self_curv = |row: usize, atom: usize| -> f64 {
            let Some(ch) = cross_channels.as_ref() else {
                return 0.0;
            };
            let d_k = if learnable_alpha {
                ch.cross_row_d_logalpha[atom]
            } else {
                ch.cross_row_d[atom]
            };
            let j = ch.z_jac[row * k_atoms + atom];
            d_k * j * j
        };
        let mut trace = 0.0_f64;
        for row in 0..self.n_obs() {
            let row_base = cache.row_offsets[row];
            let assignment_base = row * k_atoms;
            let q = cache.row_dims[row];
            // Per-row diagonal `(∂H₀'/∂ρ)_tt` for the deflation correction: the
            // assignment prior curves only the logit/assignment slots (coordinate
            // slots are 0 — ARD handles those), MINUS the downdated cross-row self
            // curvature. The full-`H` trace contraction keeps the full `hdiag`.
            let mut d_diag = Array1::<f64>::zeros(q);
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                        let d_slot = hdiag[assignment_base + atom];
                        trace += inv_diag[row_base + pos] * d_slot;
                        if pos < q {
                            d_diag[pos] = d_slot - self_curv(row, atom);
                        }
                    }
                }
                None => {
                    for free_idx in 0..assignment_dim {
                        let d_slot = hdiag[assignment_base + free_idx];
                        trace += inv_diag[row_base + free_idx] * d_slot;
                        if free_idx < q {
                            d_diag[free_idx] = d_slot - self_curv(row, free_idx);
                        }
                    }
                }
            }
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if !dirs.is_empty() {
                let inv_vv = if fast_selected {
                    let (inv_vv, _inv_vbeta) = solver
                        .selected_inverse_row_blocks(row, &selected_beta_inv)
                        .map_err(|err| {
                            format!("assignment_log_strength_hessian_trace: selected inverse: {err}")
                        })?;
                    inv_vv
                } else {
                    let mut inv_vv = Array2::<f64>::zeros((q, q));
                    for col in 0..q {
                        let mut rhs_t = Array1::<f64>::zeros(total_t);
                        let rhs_beta = Array1::<f64>::zeros(cache.k);
                        rhs_t[row_base + col] = 1.0;
                        let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                            format!(
                                "assignment_log_strength_hessian_trace: selected inverse: {err}"
                            )
                        })?;
                        for r in 0..q {
                            inv_vv[[r, col]] = solved.t[row_base + r];
                        }
                    }
                    inv_vv
                };
                let mut d_mat = Array2::<f64>::zeros((q, q));
                for s in 0..q {
                    d_mat[[s, s]] = d_diag[s];
                }
                let spectrum = cache
                    .deflation_row_spectra
                    .get(row)
                    .and_then(Option::as_ref);
                trace -= Self::deflation_block_correction(&inv_vv, &d_mat, dirs, spectrum);
            }
        }
        // #1416: the IBP prior Hessian is `H_p = d·J Jᵀ + diag(s, c)`, where the
        // rank-one `d·J Jᵀ` couples EVERY row pair `(i, j)` in a column `k`
        // through the shared empirical mass `M_k`. The assembled `H` carries the
        // full `H_full = H₀' + U D Uᵀ` (Woodbury, construction.rs:4710-4752), and
        // for fixed alpha the entire IBP prior scales with `λ = eᵖ`, so
        // `∂H_p/∂ρ = H_p`. The diagonal loop above already captures the `i = j`
        // self terms (the `d·J_ik²` summand lives in `hdiag`); this pass adds the
        // omitted off-diagonal `½·d_k·Σ_{i≠j}(H⁻¹)_{ik,jk}·J_ik·J_jk`. Only IBP
        // has the cross-row rank-one source; for other diagonal modes
        // `ibp_assignment_third_channels` returns `None` and the trace stays the
        // pure diagonal contraction. (IBP fixed-alpha uses the dense `None`
        // layout, so atom `k`'s logit slot is local position `k`.)
        if self.last_row_layout.is_none() {
            if let Some(channels) = ibp_assignment_third_channels(&self.assignment, rho)? {
                let n = self.n_obs();
                let total_t = cache.delta_t_len();
                // This trace is ½ ∂log|H|/∂ρ. For FIXED-α IBP the whole prior
                // scales with λ=eᵖ so ∂H_p/∂ρ = H_p and the rank-one coefficient
                // is the VALUE `cross_row_d[k] = w·s'_k`. For LEARNABLE-α this trace
                // is ½ ∂log|H|/∂logα, and the rank-one block's logα-derivative is
                // `∂d_k/∂logα = w·∂s'_k/∂logα` (`cross_row_d_logalpha[k]`) — the same
                // α-derivative the DIAGONAL channel (`hessian_diag_log_alpha_derivative`)
                // already uses. Using the value `s'_k` here (the pre-fix bug) made the
                // off-diagonal inconsistent with the diagonal and the α-gradient wrong.
                let learnable_alpha = matches!(
                    self.assignment.mode,
                    AssignmentMode::IBPMap {
                        learnable_alpha: true,
                        ..
                    }
                );
                let mut cross = 0.0_f64;
                for k in 0..k_atoms {
                    let d_k = if learnable_alpha {
                        channels.cross_row_d_logalpha[k]
                    } else {
                        channels.cross_row_d[k]
                    };
                    if d_k == 0.0 {
                        continue;
                    }
                    for i in 0..n {
                        let j_ik = channels.z_jac[i * k_atoms + k];
                        if j_ik == 0.0 {
                            continue;
                        }
                        // (H⁻¹) column at row `i`'s logit-`k` slot.
                        let mut rhs_t = Array1::<f64>::zeros(total_t);
                        let rhs_beta = Array1::<f64>::zeros(cache.k);
                        rhs_t[cache.row_offsets[i] + k] = 1.0;
                        let solved =
                            solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                                format!("assignment_log_strength_hessian_trace: {err}")
                            })?;
                        for j in 0..n {
                            if j == i {
                                continue;
                            }
                            let j_jk = channels.z_jac[j * k_atoms + k];
                            if j_jk == 0.0 {
                                continue;
                            }
                            let inv_ij = solved.t[cache.row_offsets[j] + k];
                            cross += d_k * inv_ij * j_ik * j_jk;
                        }
                    }
                }
                trace += cross;
            }
        }
        Ok(0.5 * trace)
    }

    pub(crate) fn learnable_ibp_forward_alpha_data_derivative(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let AssignmentMode::IBPMap {
            temperature: _,
            learnable_alpha: true,
            ..
        } = self.assignment.mode
        else {
            return Ok(0.0);
        };
        let alpha = self
            .assignment
            .mode
            .resolved_ibp_alpha(rho)
            .ok_or_else(|| "learnable IBP alpha resolution failed".to_string())?;
        let k_atoms = self.k_atoms();
        let prior = ordered_geometric_shrinkage_prior(k_atoms, alpha);
        let mut dprior = Array1::<f64>::zeros(k_atoms);
        for k in 0..k_atoms {
            // dπ_k/dρ for π_k = (α/(α+1))^(k+1) (#614 consistent stick-breaking
            // prior mean): dπ_k/dα = π_k·(k+1)/(α(α+1)), and with α = α₀·exp(ρ)
            // the log-α chain factor α cancels the 1/α ⇒ dπ_k/dρ = π_k·(k+1)/(α+1).
            dprior[k] = prior[k] * (k + 1) as f64 / (alpha + 1.0);
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let row_loss_w = self.row_loss_weights.as_deref();
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut f_rho = Array1::<f64>::zeros(p);
        let mut residual = Array1::<f64>::zeros(p);
        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = vec![0.0_f64; k_atoms];
        let mut total = 0.0_f64;
        for row in 0..n {
            self.assignment
                .try_assignments_row_for_rho_into(row, rho, &mut assignments)?;
            fitted.fill(0.0);
            f_rho.fill(0.0);
            for k in 0..k_atoms {
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                // Ungated (#1026 background-tier) atoms have a force-fixed unit
                // gate (`has_ungated` override), so their mass `a_k ≡ 1` is
                // α-INDEPENDENT (∂a_k/∂logα = 0). The π_k(α) chain below applies
                // ONLY to gated atoms, whose mass is `a_k = σ(ℓ/τ)·π_k(α)`. (NB:
                // frozen routing is NOT ungated — there the gate is a fixed σ(ℓ/τ)
                // but `a_k` still varies with α through `π_k`, so it must NOT be
                // skipped.)
                let da_rho = if self.assignment.ungated.get(k).copied().unwrap_or(false) {
                    0.0
                } else {
                    (assignments[k] / prior[k]) * dprior[k]
                };
                for out_col in 0..p {
                    fitted[out_col] += assignments[k] * decoded[out_col];
                    f_rho[out_col] += da_rho * decoded[out_col];
                }
            }
            for out_col in 0..p {
                residual[out_col] = fitted[out_col] - target[[row, out_col]];
            }
            let residual_metric = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, residual.view()),
                _ => residual.to_vec(),
            };
            let row_weight = row_loss_w.map_or(1.0, |w| w[row]);
            let mut row_dot = 0.0_f64;
            for out_col in 0..p {
                row_dot += residual_metric[out_col] * f_rho[out_col];
            }
            total += row_weight * row_dot;
        }
        Ok(total)
    }

    /// Per-row spectral-deflation correction `tr((H⁻¹)_tt · (D − DΦ[D]))` for one
    /// evidence ρ-component, to be SUBTRACTED from the raw-derivative trace
    /// `tr((H⁻¹)_tt · D)` the trace otherwise accumulates.
    ///
    /// The criterion VALUE re-deflates each per-row `H_tt` at every ρ, so the
    /// correct evidence gradient contracts `(H⁻¹)_tt` against the deflation-map
    /// derivative `DΦ[D]`, not the raw `D = (∂H_raw/∂ρ)_tt`. By Daleckii–Krein,
    /// in the row's RAW eigenbasis `U`,
    ///   `DΦ[D] = U (F ∘ (Uᵀ D U)) Uᵀ`,  `F_{ml} = (λ̃ₘ − λ̃ₗ)/(λₘ − λₗ)`
    /// (raw `λ` in the denominator, conditioned `λ̃` in the numerator; the
    /// diagonal / degenerate entry is `f'(λₘ) = 1` for an unclamped kept
    /// direction and `0` otherwise). Hence `D − DΦ[D] = U ((1−F) ∘ (Uᵀ D U)) Uᵀ`,
    /// whose kept×kept block is `0`, deflated×deflated block is the full `M`, and
    /// kept(m)×deflated(i) block carries the ROTATION coefficient
    /// `(1−λᵢ)/(λₘ−λᵢ)`. Contracting against the FULL deflated selected-inverse
    /// t-block `inv_vv` (which carries the β-Schur back-substitution) captures
    /// both the within-row kept-subspace term and the deferred β-Schur/rotation
    /// coupling in one pass, matching the re-deflating fixed-state FD oracle.
    ///
    /// `spectrum = Some` (spectral deflation): exact Daleckii–Krein. `None` with a
    /// non-empty `dirs` (gauge-only deflation, ρ-independent structural null):
    /// fall back to the within-row kept-subspace term `Σᵢ vᵢᵀ D vᵢ`.
    /// `inv_vv` is assumed symmetric (selected inverse of a symmetric PD system).
    fn deflation_block_correction(
        inv_vv: &Array2<f64>,
        d_mat: &Array2<f64>,
        dirs: &[Array1<f64>],
        spectrum: Option<&RowDeflationSpectrum>,
    ) -> f64 {
        let q = inv_vv.nrows();
        let Some(spec) = spectrum else {
            // Gauge-only deflation: ρ-independent structural null → within-row term.
            let mut acc = 0.0_f64;
            for v in dirs {
                for a in 0..q {
                    let va = if a < v.len() { v[a] } else { 0.0 };
                    if va == 0.0 {
                        continue;
                    }
                    for b in 0..q {
                        let vb = if b < v.len() { v[b] } else { 0.0 };
                        acc += va * vb * d_mat[[a, b]];
                    }
                }
            }
            return acc;
        };
        let u = &spec.evecs;
        if u.nrows() != q || u.ncols() != q {
            return 0.0;
        }
        let raw = &spec.raw_evals;
        let cond = &spec.cond_evals;
        // M = Uᵀ D U, W = Uᵀ inv_vv U (both q×q, symmetric).
        let m = u.t().dot(d_mat).dot(u);
        let w = u.t().dot(inv_vv).dot(u);
        // correction = Σ_{m,l} W[m,l]·M[m,l]·(1 − F[m,l]).
        let mut acc = 0.0_f64;
        let eps = 1.0e-12;
        for a in 0..q {
            for b in 0..q {
                let denom = raw[a] - raw[b];
                let f1 = if denom.abs() > eps {
                    (cond[a] - cond[b]) / denom
                } else if cond[a] == raw[a] {
                    1.0
                } else {
                    0.0
                };
                acc += w[[a, b]] * m[[a, b]] * (1.0 - f1);
            }
        }
        acc
    }

    /// #1417: exact `½ tr(H⁻¹ ∂H_data/∂logα)` for LEARNABLE IBP alpha.
    ///
    /// The forward assignment is `a_ik = σ(ℓ_ik/τ)·π_k(α)` with the #614
    /// consistent stick-breaking mean `π_k(α) = (α/(α+1))^(k+1)`, so
    /// `∂logπ_k/∂logα = (k+1)/(α+1)`. EVERY data-Jacobian column for atom `k` —
    /// the logit-JVP row (carries one `π_k`), the coordinate rows (carry one
    /// `a_k`), and the β-leg (`a_k·φ`) — carries exactly ONE `a_k`/`π_k` factor
    /// (`σ(ℓ/τ)` is α-independent). Hence each Jacobian column scales as
    /// `∂J_·k/∂logα = ((k+1)/(α+1))·J_·k`, and the data Hessian block for the
    /// atom pair `(k_a, k_b)` scales as
    ///   ∂H_data[a,b]/∂logα = (((k_a+1) + (k_b+1))/(α+1))·H_data[a,b].
    /// Therefore the exact data-block contribution to the α-logdet trace is
    ///   ½ tr(H⁻¹ ∂H_data/∂logα)
    ///     = ½/(α+1) · Σ_{a,b} ((k_a+1) + (k_b+1))·(H⁻¹)_{ba}·H_data[a,b],
    /// over the full joint `(t, β)` index set. `H_data[a,b]` is the data-fit
    /// Gauss-Newton block built from the SAME `row_jets_for_logdet` first-jets the
    /// θ-adjoint uses (`H_tt = ⟨J_a,J_b⟩`, `H_tβ = ⟨J_a,J_β⟩`, `H_ββ = ⟨J_β,J_β'⟩`),
    /// and `(H⁻¹)` is contracted through the same per-row selected-inverse blocks.
    /// This closes the learnable-α gradient: combined with the prior-Hessian
    /// trace (`assignment_log_strength_hessian_trace`) the full
    /// `½ tr(H⁻¹ ∂H/∂logα)` is now assembled. For FIXED alpha (and non-IBP modes)
    /// this is identically zero.
    pub(crate) fn learnable_ibp_data_logdet_alpha_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<f64, String> {
        let AssignmentMode::IBPMap {
            learnable_alpha: true,
            ..
        } = self.assignment.mode
        else {
            return Ok(0.0);
        };
        let alpha = self
            .assignment
            .mode
            .resolved_ibp_alpha(rho)
            .ok_or_else(|| "learnable IBP alpha resolution failed".to_string())?;
        let inv_alpha1 = 1.0 / (alpha + 1.0);
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;

        // β-tier selected inverse `(H⁻¹)_ββ` (shared across rows). #932 FRONT C:
        // on the plain bordered arrow this is the cached dense `S⁻¹` formed once
        // (no `K` full-system solves); when a gauge / #1038 cross-row Woodbury is
        // active the row-local Takahashi blocks are NOT valid, so we fall back to
        // the per-β-coordinate `solve` loop (bit-identical, just O(n) per call).
        let fast_selected = solver.plain_selected_inverse_available();
        let beta_inv = if cache.k == 0 {
            Array2::<f64>::zeros((0, 0))
        } else if fast_selected {
            solver.beta_inv().map_err(|err| {
                format!("learnable_ibp_data_logdet_alpha_trace: beta inverse: {err}")
            })?
        } else {
            let mut beta_inv = Array2::<f64>::zeros((cache.k, cache.k));
            let rhs_t = Array1::<f64>::zeros(total_t);
            for col in 0..cache.k {
                let mut rhs_beta = Array1::<f64>::zeros(cache.k);
                rhs_beta[col] = 1.0;
                let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                    format!("learnable_ibp_data_logdet_alpha_trace: beta inverse: {err}")
                })?;
                for r in 0..cache.k {
                    beta_inv[[r, col]] = solved.beta[r];
                }
            }
            beta_inv
        };
        // Atom index of each β border channel (the `k_b` weight for the β leg).
        let border_atom: Vec<usize> = border.iter().map(|c| c.atom).collect();

        let mut trace = 0.0_f64;
        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        // #932 SIMD: jets are built in aligned 4-row SIMD batches through a
        // bounded (≤4-row) look-ahead window; unaligned / non-softmax / remainder
        // rows fall back to the scalar per-row path (bit-identical either way).
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment
                .try_assignments_row_for_rho_into(row, rho, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    rho,
                    jet_window_next,
                    n,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window.pop_front().expect("jet window must be non-empty");
            // Atom index (k-weight) of each local t-var.
            let var_atom: Vec<usize> = jets
                .vars
                .iter()
                .map(|v| match *v {
                    SaeLocalRowVar::Logit { atom } => atom,
                    SaeLocalRowVar::Coord { atom, .. } => atom,
                })
                .collect();

            // Per-row selected inverse blocks `(H⁻¹)_tt` (q×q) and `(H⁻¹)_tβ`.
            // #932 FRONT C: row-local Takahashi (O(q·(q+K))) on the plain arrow;
            // per-row full-system `solve` loop (O(n·q)) under gauge / cross-row
            // Woodbury where the row-local blocks are not valid.
            let (inv_vv, inv_vbeta) = if fast_selected {
                solver
                    .selected_inverse_row_blocks(row, &beta_inv)
                    .map_err(|err| {
                        format!("learnable_ibp_data_logdet_alpha_trace: selected inverse: {err}")
                    })?
            } else {
                let mut inv_vv = Array2::<f64>::zeros((q, q));
                let mut inv_vbeta = Array2::<f64>::zeros((q, cache.k));
                for col in 0..q {
                    let mut rhs_t = Array1::<f64>::zeros(total_t);
                    let rhs_beta = Array1::<f64>::zeros(cache.k);
                    rhs_t[base + col] = 1.0;
                    let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                        format!("learnable_ibp_data_logdet_alpha_trace: selected inverse: {err}")
                    })?;
                    for r in 0..q {
                        inv_vv[[r, col]] = solved.t[base + r];
                    }
                    for b in 0..cache.k {
                        inv_vbeta[[col, b]] = solved.beta[b];
                    }
                }
                (inv_vv, inv_vbeta)
            };

            // #1026 — UNGATED (background-tier) atoms have a force-fixed unit gate,
            // so their mass `a_k ≡ 1` is α-INDEPENDENT: every data-Jacobian column
            // for an ungated atom carries `a_k = 1`, NOT `π_k(α)`, so its α-exponent
            // is `e_k = 0`, not `k+1`. Gated atoms keep `e_k = k+1`. (The prior trace
            // handles ungated separately by zeroing the fixed-logit `z_jac`.)
            let kfac = |atom: usize| -> f64 {
                if self.assignment.ungated.get(atom).copied().unwrap_or(false) {
                    0.0
                } else {
                    (atom + 1) as f64
                }
            };
            // t–t block: Σ_{a,b} (e_a + e_b)·(H⁻¹)_{ba}·⟨J_a, J_b⟩, where the
            // per-atom log-prior exponent is e_k = k+1 for the #614 consistent
            // stick-breaking mean π_k = (α/(α+1))^(k+1) (dlogπ_k/dlogα = (k+1)·inv_alpha1).
            for a in 0..q {
                for b in 0..q {
                    let h_ab = sae_dot(&jets.first[a], &jets.first[b]);
                    if h_ab == 0.0 {
                        continue;
                    }
                    let kw = kfac(var_atom[a]) + kfac(var_atom[b]);
                    trace += kw * inv_vv[[b, a]] * h_ab;
                }
            }
            // Deflation correction (kept-subspace restriction + β-Schur/rotation).
            // `inv_vv` is the DEFLATED selected inverse, so the t–t contraction
            // above contracts the RAW derivative `D` where the re-deflating
            // criterion uses the deflation-map derivative `DΦ[D]`. Subtract the
            // exact over-count `tr(inv_vv·(D − DΦ[D]))` via the Daleckii–Krein
            // helper, with `D_{ab} = kw_ab·⟨J_a, J_b⟩` the SAME t–t operator the
            // trace contracts. The t–β/β–β blocks are not deflated, so only the
            // t–t contraction is corrected.
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if !dirs.is_empty() {
                let mut d_mat = Array2::<f64>::zeros((q, q));
                for a in 0..q {
                    for b in 0..q {
                        let h_ab = sae_dot(&jets.first[a], &jets.first[b]);
                        if h_ab == 0.0 {
                            continue;
                        }
                        d_mat[[a, b]] = (kfac(var_atom[a]) + kfac(var_atom[b])) * h_ab;
                    }
                }
                let spectrum = cache
                    .deflation_row_spectra
                    .get(row)
                    .and_then(Option::as_ref);
                trace -= Self::deflation_block_correction(&inv_vv, &d_mat, dirs, spectrum);
            }
            // t–β and β–t blocks: appear symmetrically, contract once with the
            // factor 2 (H, H⁻¹ symmetric; `(H⁻¹)_βt = (H⁻¹)_tβᵀ`).
            for a in 0..q {
                for (beta_pos, channel) in border.iter().enumerate() {
                    let h_ab = sae_dot(&jets.first[a], &jets.beta[beta_pos]);
                    if h_ab == 0.0 {
                        continue;
                    }
                    let kw = kfac(var_atom[a]) + kfac(border_atom[beta_pos]);
                    trace += 2.0 * kw * inv_vbeta[[a, channel.index]] * h_ab;
                }
            }
            // β–β block: Σ_{β,β'} (k_β + k_β')·(H⁻¹)_{β'β}·⟨J_β, J_β'⟩.
            for (beta_i, channel_i) in border.iter().enumerate() {
                for (beta_j, channel_j) in border.iter().enumerate() {
                    let h_ab = sae_dot(&jets.beta[beta_i], &jets.beta[beta_j]);
                    if h_ab == 0.0 {
                        continue;
                    }
                    let kw = kfac(border_atom[beta_i]) + kfac(border_atom[beta_j]);
                    trace += kw * beta_inv[[channel_i.index, channel_j.index]] * h_ab;
                }
            }
        }
        Ok(0.5 * inv_alpha1 * trace)
    }

    pub(crate) fn add_learnable_ibp_forward_alpha_data_rhs(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        t: &mut Array1<f64>,
        beta: &mut Array1<f64>,
    ) -> Result<(), String> {
        let AssignmentMode::IBPMap {
            temperature,
            learnable_alpha: true,
            ..
        } = self.assignment.mode
        else {
            return Ok(());
        };
        let alpha = self
            .assignment
            .mode
            .resolved_ibp_alpha(rho)
            .ok_or_else(|| "learnable IBP alpha resolution failed".to_string())?;
        let k_atoms = self.k_atoms();
        let p = self.output_dim();
        let prior = ordered_geometric_shrinkage_prior(k_atoms, alpha);
        let mut dprior = Array1::<f64>::zeros(k_atoms);
        for k in 0..k_atoms {
            // dπ_k/dρ for π_k = (α/(α+1))^(k+1) (#614 consistent stick-breaking
            // prior mean): dπ_k/dα = π_k·(k+1)/(α(α+1)), and with α = α₀·exp(ρ)
            // the log-α chain factor α cancels the 1/α ⇒ dπ_k/dρ = π_k·(k+1)/(α+1).
            dprior[k] = prior[k] * (k + 1) as f64 / (alpha + 1.0);
        }
        let inv_tau = 1.0 / temperature;
        let row_loss_w = self.row_loss_weights.as_deref();
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let border = self.border_channels_for_cache(cache)?;
        let mut decoded_rows = vec![vec![0.0_f64; p]; k_atoms];
        let mut decoded_deriv = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut f_rho = Array1::<f64>::zeros(p);
        let mut residual = Array1::<f64>::zeros(p);
        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = vec![0.0_f64; k_atoms];
        for row in 0..self.n_obs() {
            self.assignment
                .try_assignments_row_for_rho_into(row, rho, &mut assignments)?;
            fitted.fill(0.0);
            f_rho.fill(0.0);
            for k in 0..k_atoms {
                self.atoms[k].fill_decoded_row(row, &mut decoded_rows[k]);
                // Ungated (#1026 background-tier) atoms have a force-fixed unit
                // gate (`has_ungated` override), so their mass `a_k ≡ 1` is
                // α-INDEPENDENT (∂a_k/∂logα = 0). The π_k(α) chain below applies
                // ONLY to gated atoms, whose mass is `a_k = σ(ℓ/τ)·π_k(α)`. (NB:
                // frozen routing is NOT ungated — there the gate is a fixed σ(ℓ/τ)
                // but `a_k` still varies with α through `π_k`, so it must NOT be
                // skipped.)
                let da_rho = if self.assignment.ungated.get(k).copied().unwrap_or(false) {
                    0.0
                } else {
                    (assignments[k] / prior[k]) * dprior[k]
                };
                for out_col in 0..p {
                    fitted[out_col] += assignments[k] * decoded_rows[k][out_col];
                    f_rho[out_col] += da_rho * decoded_rows[k][out_col];
                }
            }
            for out_col in 0..p {
                residual[out_col] = fitted[out_col] - target[[row, out_col]];
            }
            let residual_metric = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, residual.view()),
                _ => residual.to_vec(),
            };
            let f_metric = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, f_rho.view()),
                _ => f_rho.to_vec(),
            };
            let row_weight = row_loss_w.map_or(1.0, |w| w[row]);
            let row_vars = self.row_vars_for_cache_row(row, cache)?;
            let row_base = cache.row_offsets[row];
            for (pos, var) in row_vars.iter().enumerate() {
                let mut contribution = 0.0_f64;
                match *var {
                    SaeLocalRowVar::Logit { atom } => {
                        let sigma = assignments[atom] / prior[atom];
                        let sigma_jac = sigma * (1.0 - sigma) * inv_tau;
                        let da_dl = sigma_jac * prior[atom];
                        let d_da_rho_dl = sigma_jac * dprior[atom];
                        for out_col in 0..p {
                            contribution += da_dl * decoded_rows[atom][out_col] * f_metric[out_col];
                            contribution += d_da_rho_dl
                                * decoded_rows[atom][out_col]
                                * residual_metric[out_col];
                        }
                    }
                    SaeLocalRowVar::Coord { atom, axis } => {
                        let sigma = assignments[atom] / prior[atom];
                        let da_rho = sigma * dprior[atom];
                        self.atoms[atom].fill_decoded_derivative_row(row, axis, &mut decoded_deriv);
                        for out_col in 0..p {
                            contribution +=
                                assignments[atom] * decoded_deriv[out_col] * f_metric[out_col];
                            contribution +=
                                da_rho * decoded_deriv[out_col] * residual_metric[out_col];
                        }
                    }
                }
                t[row_base + pos] += row_weight * contribution;
            }
            for channel in &border {
                let phi = self.atoms[channel.atom].basis_values[[row, channel.basis_col]];
                let sigma = assignments[channel.atom] / prior[channel.atom];
                let da_rho = sigma * dprior[channel.atom];
                let mut contribution = 0.0_f64;
                for out_col in 0..p {
                    let output = channel.output[out_col];
                    contribution += assignments[channel.atom] * phi * output * f_metric[out_col];
                    contribution += da_rho * phi * output * residual_metric[out_col];
                }
                beta[channel.index] += row_weight * contribution;
            }
        }
        Ok(())
    }

    pub(crate) fn border_channels_for_cache(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<SaeBorderChannel>, String> {
        let p = self.output_dim();
        let frames_active = self.last_frames_active && cache.k == self.factored_border_dim();
        let offsets = if frames_active {
            self.factored_beta_offsets()
        } else {
            self.beta_offsets()
        };
        let mut channels = Vec::with_capacity(cache.k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let frame = if frames_active {
                self.frame_output_matrix(atom_idx)
            } else {
                Array2::<f64>::eye(p)
            };
            let r = frame.ncols();
            for basis_col in 0..m {
                for channel in 0..r {
                    let mut output = vec![0.0_f64; p];
                    for out_col in 0..p {
                        output[out_col] = frame[[out_col, channel]];
                    }
                    channels.push(SaeBorderChannel {
                        atom: atom_idx,
                        basis_col,
                        index: offsets[atom_idx] + basis_col * r + channel,
                        output,
                    });
                }
            }
        }
        if channels.len() != cache.k {
            return Err(format!(
                "border channel layout has {} entries but cache border has {}",
                channels.len(),
                cache.k
            ));
        }
        Ok(channels)
    }

    pub(crate) fn row_vars_for_cache_row(
        &self,
        row: usize,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<SaeLocalRowVar>, String> {
        let q_row = cache.row_dims[row];
        let mut vars: Vec<Option<SaeLocalRowVar>> = vec![None; q_row];
        match self.last_row_layout {
            Some(ref layout) => {
                for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                    vars[pos] = Some(SaeLocalRowVar::Logit { atom });
                    let start = layout.coord_starts[row][pos];
                    let d = self.assignment.coords[atom].latent_dim();
                    for axis in 0..d {
                        vars[start + axis] = Some(SaeLocalRowVar::Coord { atom, axis });
                    }
                }
            }
            None => {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let coord_offsets = self.assignment.coord_offsets();
                for atom in 0..assignment_dim {
                    vars[atom] = Some(SaeLocalRowVar::Logit { atom });
                }
                for atom in 0..self.k_atoms() {
                    let start = coord_offsets[atom];
                    let d = self.assignment.coords[atom].latent_dim();
                    for axis in 0..d {
                        vars[start + axis] = Some(SaeLocalRowVar::Coord { atom, axis });
                    }
                }
            }
        }
        vars.into_iter()
            .enumerate()
            .map(|(idx, v)| {
                v.ok_or_else(|| {
                    format!("row_vars_for_cache_row: row {row} position {idx} was not mapped")
                })
            })
            .collect()
    }

    pub(crate) fn atom_second_jets(&self) -> Result<Vec<Array4<f64>>, String> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let coords = self.assignment.coords[atom_idx].as_matrix();
            let jet = if let Some(second) = atom.basis_second_jet.as_ref() {
                second.second_jet(coords.view())?
            } else {
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!(
                        "logdet_theta_adjoint: atom '{}' has no basis evaluator for second jets",
                        atom.name
                    )
                })?;
                evaluator
                    .second_jet_dyn(coords.view())
                    .ok_or_else(|| {
                        format!(
                            "logdet_theta_adjoint: atom '{}' basis does not expose analytic second jets",
                            atom.name
                        )
                    })??
            };
            let expected = (
                atom.n_obs(),
                atom.basis_size(),
                atom.latent_dim,
                atom.latent_dim,
            );
            if jet.dim() != expected {
                return Err(format!(
                    "logdet_theta_adjoint: atom '{}' second jet shape {:?}, expected {:?}",
                    atom.name,
                    jet.dim(),
                    expected
                ));
            }
            out.push(jet);
        }
        Ok(out)
    }

    // [#780 line-count gate] The per-row jet / reconstruction-channel cluster
    // (`reconstruction_row_program_for_logdet`, the const-generic
    // reconstruction / β-border channel fills and their dynamic dispatchers,
    // `row_jets_for_logdet`, `row_jets_for_logdet_batch4`, `batch4_assemble`,
    // and `refill_jet_window`) lives in the sibling
    // `construction_row_jet_logdet_channels.rs` file, inlined via `include!`
    // below at module scope as a second `impl SaeManifoldTerm` block. Splitting
    // it out keeps this tracked file under the 10k limit; `include!` preserves
    // the identical module scope and private-field access.

    pub(crate) fn assignment_prior_hdiag_derivative_entry(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        diag_atom: usize,
        wrt: SaeLocalRowVar,
        ibp_channels: Option<&IbpHessianDiagThirdChannels>,
    ) -> f64 {
        let SaeLocalRowVar::Logit { atom: wrt_atom } = wrt else {
            return 0.0;
        };
        match self.assignment.mode {
            AssignmentMode::Softmax { .. } => {
                // #1038: the softmax entropy Hessian is now stored DENSE in
                // `block.htt` and its full θ-derivative `∂H_{k,j}/∂z_w` (diagonal
                // AND off-diagonal) is added inline in `logdet_theta_adjoint` from
                // the shared `row_dense_hessian_logit_derivative`. Returning the
                // diagonal contribution here too would double-count, so this
                // primitive is silent for softmax — the dense path is the single
                // source for value, logdet, and adjoint.
                0.0
            }
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => {
                if diag_atom != wrt_atom {
                    return 0.0;
                }
                let logit = self.assignment.logits[[row, diag_atom]];
                if !crate::assignment::jumprelu_in_optimization_band(
                    logit,
                    threshold,
                    temperature,
                ) {
                    return 0.0;
                }
                let inv_tau = 1.0 / temperature;
                let activation =
                    gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                // #1415: P(ℓ)=λσ((ℓ−θ)/τ); P''(ℓ)=(λ/τ²)s(1−2a) so the third
                // derivative is P'''(ℓ)=(λ/τ³)·s·(1−6a+6a²), because
                // d/dℓ[s(1−2a)] = (1/τ)s[(1−2a)²−2s] = (1/τ)s(1−6a+6a²).
                rho.lambda_sparse()
                    * slope
                    * (1.0 - 6.0 * activation + 6.0 * activation * activation)
                    * inv_tau
                    * inv_tau
                    * inv_tau
            }
            AssignmentMode::IBPMap { .. } => {
                // The assembled `htt` diagonal consumes
                // `IBPAssignmentPenalty::hessian_diag`, whose logit derivative
                // splits into a row-local direct-`z` channel and a global
                // empirical-`M_k` channel (π_k couples every row in column k).
                // This same-row primitive returns only the LOCAL direct-`z`
                // channel — and only on the matching logit (`diag_atom == w`),
                // since H_ik depends on no other row's z explicitly. The global
                // M_k channel is accumulated column-wise in
                // `logdet_theta_adjoint` (it needs the per-row selected-inverse
                // diagonals), so adding it here would double-count.
                if diag_atom != wrt_atom {
                    return 0.0;
                }
                match ibp_channels {
                    Some(ch) => ch.local_logit_third[row * ch.k_max + diag_atom],
                    None => 0.0,
                }
            }
        }
    }

    pub(crate) fn ard_majorized_hessian_derivative(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        atom: usize,
        axis: usize,
    ) -> f64 {
        if rho.log_ard[atom].is_empty() {
            return 0.0;
        }
        let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
        let periods = self.assignment.coords[atom].effective_axis_periods();
        let t = self.assignment.coords[atom].row(row)[axis];
        let prior = ArdAxisPrior::eval(alpha, t, periods[axis]);
        if prior.hess <= 0.0 {
            return 0.0;
        }
        match periods[axis] {
            None => 0.0,
            Some(period) => {
                let kappa = std::f64::consts::TAU / period;
                -alpha * kappa * (kappa * t).sin()
            }
        }
    }

    pub fn outer_rho_gradient_ift_rhs(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        j: usize,
        cache: &ArrowFactorCache,
    ) -> Result<SaeArrowVector, String> {
        let n_params = rho.to_flat().len();
        if j >= n_params {
            return Err(format!(
                "outer_rho_gradient_ift_rhs: coordinate {j} outside rho dim {n_params}"
            ));
        }
        let mut t = Array1::<f64>::zeros(cache.delta_t_len());
        let mut beta = Array1::<f64>::zeros(cache.k);
        if j == 0 {
            let assignment_grad =
                assignment_prior_log_strength_target_mixed(&self.assignment, rho)?;
            let k_atoms = self.k_atoms();
            let assignment_dim = self.assignment.assignment_coord_dim();
            for row in 0..self.n_obs() {
                let base = cache.row_offsets[row];
                let assignment_base = row * k_atoms;
                match self.last_row_layout {
                    Some(ref layout) => {
                        for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                            t[base + pos] = assignment_grad[assignment_base + atom];
                        }
                    }
                    None => {
                        for free_idx in 0..assignment_dim {
                            t[base + free_idx] = assignment_grad[assignment_base + free_idx];
                        }
                    }
                }
            }
            self.add_learnable_ibp_forward_alpha_data_rhs(rho, target, cache, &mut t, &mut beta)?;
        } else if (1..=rho.log_lambda_smooth.len()).contains(&j) {
            // #1556: coordinate `j ∈ 1..=K` is the per-atom smoothness strength
            // `log λ_smooth[j-1]`. `∂(penalty)/∂log λ_k = λ_k·S_k C_k` touches ONLY
            // atom `k = j-1`'s decoder block; every other atom's RHS is zero.
            let target_atom = j - 1;
            let lambda = rho.lambda_smooth_for(target_atom);
            let frames_active = self.last_frames_active && cache.k == self.factored_border_dim();
            let offsets = if frames_active {
                self.factored_beta_offsets()
            } else {
                self.beta_offsets()
            };
            let atom = &self.atoms[target_atom];
            let m = atom.basis_size();
            let coeffs = if frames_active {
                match &atom.decoder_frame {
                    Some(frame) => frame.project_decoder(atom.decoder_coefficients.view())?,
                    None => atom.decoder_coefficients.clone(),
                }
            } else {
                atom.decoder_coefficients.clone()
            };
            let r = coeffs.ncols();
            let off = offsets[target_atom];
            for mu in 0..m {
                for channel in 0..r {
                    let mut acc = 0.0_f64;
                    for nu in 0..m {
                        let s_sym =
                            0.5 * (atom.smooth_penalty[[mu, nu]] + atom.smooth_penalty[[nu, mu]]);
                        acc += s_sym * coeffs[[nu, channel]];
                    }
                    beta[off + mu * r + channel] = lambda * acc;
                }
            }
        } else {
            let mut cursor = 1 + rho.log_lambda_smooth.len();
            for atom in 0..rho.log_ard.len() {
                for axis in 0..rho.log_ard[atom].len() {
                    if cursor == j {
                        let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
                        let periods = self.assignment.coords[atom].effective_axis_periods();
                        for row in 0..self.n_obs() {
                            let row_t = self.assignment.coords[atom].row(row);
                            let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                            let Some(pos) = sae_coord_penalty_offset(
                                self.last_row_layout.as_ref(),
                                self.assignment.coord_offsets()[atom] + axis,
                                row,
                                atom,
                            ) else {
                                continue;
                            };
                            t[cache.row_offsets[row] + pos] = prior.grad;
                        }
                        return Ok(SaeArrowVector { t, beta });
                    }
                    cursor += 1;
                }
            }
        }
        Ok(SaeArrowVector { t, beta })
    }

    pub(crate) fn logdet_theta_adjoint(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<SaeArrowVector, String> {
        // Γ_a = tr(H⁻¹ ∂H/∂θ_a) over the inner variables θ (#1006). `H` here is
        // the SAME object the evidence factor builds — Gauss-Newton data
        // curvature plus the prior majorizers / `hessian_diag` diagonals the
        // Newton/Schur Cholesky factorizes — so each block's θ-derivative channel
        // is differentiated on the criterion's own branch (no value/gradient
        // desync). The IBP-MAP assignment prior is the one block whose
        // `hessian_diag` couples every row in a column through the plug-in
        // empirical mass `M_k = Σ_i z_ik`; its logit derivative therefore has a
        // row-local channel (handled inline via
        // `assignment_prior_hdiag_derivative_entry`) and a cross-row channel
        // (accumulated column-wise after the row loop, below).
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        // #932 FRONT C: plain-arrow `(H⁻¹)_ββ = S⁻¹` formed once from the cached
        // Schur factor; gauge / #1038 cross-row Woodbury fall back to the per-β
        // `solve` loop where the row-local Takahashi blocks are not valid.
        let fast_selected = solver.plain_selected_inverse_available();
        let beta_inv = if cache.k == 0 {
            Array2::<f64>::zeros((0, 0))
        } else if fast_selected {
            solver
                .beta_inv()
                .map_err(|err| format!("logdet_theta_adjoint: beta selected inverse: {err}"))?
        } else {
            let mut beta_inv = Array2::<f64>::zeros((cache.k, cache.k));
            let rhs_t = Array1::<f64>::zeros(total_t);
            for col in 0..cache.k {
                let mut rhs_beta = Array1::<f64>::zeros(cache.k);
                rhs_beta[col] = 1.0;
                let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                    format!("logdet_theta_adjoint: beta selected inverse solve: {err}")
                })?;
                for row in 0..cache.k {
                    beta_inv[[row, col]] = solved.beta[row];
                }
            }
            beta_inv
        };
        // IBP `hessian_diag` logit third-derivative channels (#1006). The full
        // IBP Hessian also has per-column cross-row rank-one terms
        // `H_(i,k),(j,k) = d_k·J_ik·J_jk`; these ARE carried in `H` via the #1038
        // Woodbury source (`IbpCrossRowSource`, construction.rs:4710-4752), the
        // ρ-trace differentiates them (#1416,
        // `assignment_log_strength_hessian_trace`), AND this θ-adjoint now
        // differentiates them exactly too: the empirical-`M_k` channel below
        // contracts the shared-mass coupling of the DIAGONAL curvature, and the
        // cross-row Woodbury pass (further below, using `cross_row_dd` and
        // `logit_curvature`) contracts the `∂/∂ℓ_w (d_k·J_ik·J_jk)` rank-one
        // derivative — so value, logdet, ρ-trace, and θ-adjoint all differentiate
        // the one operator `H = H₀ + Σ_k d_k u_k u_kᵀ`.
        let ibp_channels = ibp_assignment_third_channels(&self.assignment, rho)?;
        let k_atoms = self.k_atoms();
        // #1038 softmax entropy: the dense per-row entropy Hessian written into
        // `block.htt` has off-diagonal logit terms whose θ-derivative the adjoint
        // must contract too (not just the diagonal). Build the SAME penalty +
        // `scale = λ/τ²` the assembly uses so value/logdet/adjoint differentiate
        // one operator. `None` for non-softmax modes (their diagonal/cross-row
        // channels are handled by `assignment_prior_hdiag_derivative_entry` and
        // the IBP column pass).
        let softmax_dense_adjoint: Option<(
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                Some((
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                    scale,
                ))
            }
            _ => None,
        };
        // Per active logit position: (row i, column k, global t-index,
        // (H⁻¹)_ik,ik) — the inputs to the IBP cross-row empirical-`M_k` channel.
        let mut ibp_logit_sites: Vec<(usize, usize, usize, f64)> = Vec::new();

        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        // #932 SIMD: jets are built in aligned 4-row SIMD batches through a
        // bounded (≤4-row) look-ahead window; unaligned / non-softmax / remainder
        // rows fall back to the scalar per-row path (bit-identical either way).
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment
                .try_assignments_row_for_rho_into(row, rho, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    rho,
                    jet_window_next,
                    n,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window.pop_front().expect("jet window must be non-empty");

            // #932 FRONT C: row-local Takahashi on the plain arrow; per-row
            // full-system `solve` loop under gauge / cross-row Woodbury.
            let (inv_vv, inv_vbeta) = if fast_selected {
                solver
                    .selected_inverse_row_blocks(row, &beta_inv)
                    .map_err(|err| {
                        format!("logdet_theta_adjoint: selected inverse: {err}")
                    })?
            } else {
                let mut inv_vv = Array2::<f64>::zeros((q, q));
                let mut inv_vbeta = Array2::<f64>::zeros((q, cache.k));
                for col in 0..q {
                    let mut rhs_t = Array1::<f64>::zeros(total_t);
                    let rhs_beta = Array1::<f64>::zeros(cache.k);
                    rhs_t[base + col] = 1.0;
                    let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                        format!("logdet_theta_adjoint: selected inverse solve: {err}")
                    })?;
                    for r in 0..q {
                        inv_vv[[r, col]] = solved.t[base + r];
                    }
                    for b in 0..cache.k {
                        inv_vbeta[[col, b]] = solved.beta[b];
                    }
                }
                (inv_vv, inv_vbeta)
            };

            // Record each active logit's column, global t-index, and
            // selected-inverse diagonal (H⁻¹)_ik,ik for the IBP cross-row pass.
            if ibp_channels.is_some() {
                for (pos, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        ibp_logit_sites.push((row, atom, base + pos, inv_vv[[pos, pos]]));
                    }
                }
            }

            // #1419: when `w` is a logit and the assignment is softmax, the per-row
            // Gershgorin majorizer `D = diag(Σ_j|H_kj|)` is what the assembly wrote
            // into `htt` (the genuine Loewner majorizer that replaces the indefinite
            // exact entropy Hessian). Its full θ-derivative `∂D_{k,k}/∂z_w` (diagonal;
            // `∂D_kk/∂z_w = Σ_j sign(H_kj)·∂H_kj/∂z_w`) is the SAME operator the
            // assembly and logdet now differentiate, so value and adjoint stay on ONE
            // exact branch. Compute it once per logit `w` and add it at every logit
            // pair `(a,b)` below. The diagonal softmax case is therefore handled here,
            // NOT in `assignment_prior_hdiag_derivative_entry` (which returns 0 for
            // softmax to avoid double-counting).
            // #1410: the softmax majorizer θ-derivative `∂D_kk/∂z_w` is DIAGONAL
            // (`D` is diagonal), and the compact adjoint reads it only for this
            // row's `≤ top_k` active atoms. Compute the needed diagonal entry
            // directly from the softmax row `a` (= `assignments`, in hand) via
            // `active_softmax_majorizer_logit_derivative_entry`, instead of the old
            // per-(row, logit) full `K×K` `row_psd_majorizer_logit_derivative`
            // allocation. `m = Σ_j a_j l_j` is shared across all `(w, k)` pairs of
            // the row, so compute it once. `inv_tau` carries the softmax `∂a/∂z`
            // convention.
            let softmax_adjoint_row: Option<(&[f64], f64, f64, f64)> =
                match (softmax_dense_adjoint.as_ref(), self.assignment.mode) {
                    (Some((_penalty, scale)), AssignmentMode::Softmax { temperature, .. }) => {
                        let a = assignments
                            .as_slice()
                            .expect("softmax assignments row must be contiguous");
                        let m = softmax_majorizer_log_mean(a);
                        Some((a, m, *scale, 1.0 / temperature))
                    }
                    _ => None,
                };
            // Per-row UNIT-stiffness deflated directions: the selected inverse
            // `inv_vv` is the DEFLATED inverse (it assigns `1/λ̃ = 1` to each
            // `vᵢ`), so every `inv_vv`-weighted t–t contraction of `∂H/∂θ_w`
            // below spuriously contracts the RAW derivative where the re-deflating
            // criterion uses the deflation-map derivative `DΦ`. The kept-subspace Γ
            // subtracts `tr(inv_vv·(D − DΦ[D]))` over the t–t block via the same
            // Daleckii–Krein helper the ρ-traces use (the t–β / β–β blocks are not
            // deflated). `θ` enters only the per-row block (no cross-row Woodbury
            // self-downdate on the θ path), so the raw t–t derivative `D` is used
            // directly.
            let defl_dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let defl_spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            for w in 0..q {
                let mut gamma = 0.0_f64;
                // The active logit `w` differentiates against; `None` unless this
                // slot is a softmax logit on the softmax path.
                let softmax_d_dw: Option<(&[f64], f64, f64, f64, usize)> =
                    match (softmax_adjoint_row, jets.vars[w]) {
                        (Some((a, m, scale, inv_tau)), SaeLocalRowVar::Logit { atom: atom_w }) => {
                            Some((a, m, scale, inv_tau, atom_w))
                        }
                        _ => None,
                    };
                let mut dh_mat = Array2::<f64>::zeros((q, q));
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = sae_dot(&jets.second[a][w], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.second[b][w]);
                        // `∂D/∂z_w` is diagonal, so it contributes only when the two
                        // logit slots are the SAME atom (`atom_a == atom_b`).
                        if let (
                            Some((a_soft, m, scale, inv_tau, _atom_w)),
                            SaeLocalRowVar::Logit { atom: atom_a },
                            SaeLocalRowVar::Logit { atom: atom_b },
                        ) = (softmax_d_dw, jets.vars[a], jets.vars[b])
                        {
                            if atom_a == atom_b {
                                dh += active_softmax_majorizer_logit_derivative_entry(
                                    a_soft, atom_a, _atom_w, m, scale, inv_tau,
                                );
                            }
                        }
                        if a == b {
                            dh += match jets.vars[a] {
                                SaeLocalRowVar::Logit { atom } => self
                                    .assignment_prior_hdiag_derivative_entry(
                                        rho,
                                        row,
                                        atom,
                                        jets.vars[w],
                                        ibp_channels.as_ref(),
                                    ),
                                SaeLocalRowVar::Coord { atom, axis } if a == w => {
                                    self.ard_majorized_hessian_derivative(rho, row, atom, axis)
                                }
                                _ => 0.0,
                            };
                        }
                        dh_mat[[a, b]] = dh;
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv, &dh_mat, defl_dirs, defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.second[a][w], &jets.beta[beta_pos])
                            + sae_dot(&jets.first[a], &jets.beta_deriv[w][beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                for (beta_i, channel_i) in border.iter().enumerate() {
                    for (beta_j, channel_j) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.beta_deriv[w][beta_i], &jets.beta[beta_j])
                            + sae_dot(&jets.beta[beta_i], &jets.beta_deriv[w][beta_j]);
                        gamma += beta_inv[[channel_i.index, channel_j.index]] * dh;
                    }
                }
                gamma_t[base + w] = gamma;
            }

            for (w_beta_pos, w_channel) in border.iter().enumerate() {
                let mut gamma = 0.0_f64;
                let mut dh_mat = Array2::<f64>::zeros((q, q));
                for a in 0..q {
                    for b in 0..q {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.beta_l_deriv[b][w_beta_pos]);
                        dh_mat[[a, b]] = dh;
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv, &dh_mat, defl_dirs, defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.beta[beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                gamma_beta[w_channel.index] += gamma;
            }
        }

        // IBP cross-row empirical-`M_k` channel of Γ (#1006). The assembled
        // diagonal H_ik consumes `hessian_diag`, whose dependence on the column
        // mass M_k = Σ_i z_ik couples every row in a column. Differentiating
        // tr(H⁻¹ ∂H/∂ℓ_wk) on that shared branch:
        //   Γ_wk += [ Σ_i (H⁻¹)_ik,ik · ∂_M H_ik ] · J_wk = C_k · J_wk,
        // where ∂_M H_ik = `m_channel[i*K+k]` and J_wk = `z_jac[w*K+k]`. The
        // row-local direct-`z` channel was already added inline above, so this
        // pass adds only the cross-row remainder (it spans `w ≠ i` and the
        // self-row M_k self-coupling, which the row-local primitive deliberately
        // omits to avoid double-counting).
        if let Some(channels) = ibp_channels.as_ref() {
            let mut col_coeff = vec![0.0_f64; k_atoms];
            for &(row, atom, _t_index, inv_diag) in &ibp_logit_sites {
                col_coeff[atom] += inv_diag * channels.m_channel[row * k_atoms + atom];
            }
            for &(row, atom, t_index, _inv_diag) in &ibp_logit_sites {
                gamma_t[t_index] += col_coeff[atom] * channels.z_jac[row * k_atoms + atom];
            }

            // #1416: the EXACT cross-row Woodbury derivative of Γ. The assembled
            // `H` carries the per-column rank-one block `W_k = d_k·u_k u_kᵀ` with
            // `u_k` the J-weighted column indicator (`u_k[slot(i,k)] = J_ik`) and
            // `d_k = w·s'_k` (`cross_row_d[k]`). Both `d_k` (through `M_k`) and the
            // `u_k` entries (through `ℓ_ik`) depend on the logits, so
            //   ∂W_k/∂ℓ_wk = dd_k·J_wk·u_k u_kᵀ
            //               + d_k·c_wk·(e_w u_kᵀ + u_k e_wᵀ),
            // where `dd_k = ∂d_k/∂M_k = w·s''_k` (`cross_row_dd[k]`),
            // `c_wk = ∂J_wk/∂ℓ_wk` (`logit_curvature`), and `e_w` is the unit
            // vector at row `w`'s logit-`k` slot. Contracting `½ tr(H⁻¹ ∂W_k/∂ℓ_wk)`:
            //   Γ_wk += ½·dd_k·J_wk·(u_kᵀ H⁻¹ u_k)        (term A: e·J_w·(JᵀH⁻¹J))
            //         +    d_k·c_wk·(H⁻¹ u_k)_{slot(w,k)}  (term B: 2d·c_w·(H⁻¹J)_w).
            // Both `u_kᵀ H⁻¹ u_k = (JᵀH⁻¹J)_k` and the vector `(H⁻¹ u_k) = (H⁻¹J)_·k`
            // come from ONE solve per column, `x_k = H⁻¹ u_k` — so the adjoint
            // differentiates the SAME `H = H₀ + Σ_k W_k` the value/logdet use,
            // closing the one-operator contract on the rank-one block too.
            //
            // Group the column sites once (the layout is mode-agnostic: dense or
            // compact, `ibp_logit_sites` already carries each active logit's
            // global t-index), then per column build `u_k`, solve, and distribute.
            let total_t = cache.delta_t_len();
            let mut col_sites: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k_atoms];
            for &(row, atom, t_index, _inv_diag) in &ibp_logit_sites {
                col_sites[atom].push((row, t_index));
            }
            for atom in 0..k_atoms {
                let d_k = channels.cross_row_d[atom];
                let dd_k = channels.cross_row_dd[atom];
                if col_sites[atom].is_empty() || (d_k == 0.0 && dd_k == 0.0) {
                    continue;
                }
                // u_k as a full t-RHS: J at each active logit-k slot.
                let mut rhs_t = Array1::<f64>::zeros(total_t);
                let rhs_beta = Array1::<f64>::zeros(cache.k);
                for &(row, t_index) in &col_sites[atom] {
                    rhs_t[t_index] = channels.z_jac[row * k_atoms + atom];
                }
                let x_k = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                    format!("logdet_theta_adjoint: IBP cross-row Woodbury solve: {err}")
                })?;
                // (JᵀH⁻¹J)_k = u_kᵀ x_k.
                let mut jt_hinv_j = 0.0_f64;
                for &(row, t_index) in &col_sites[atom] {
                    jt_hinv_j += channels.z_jac[row * k_atoms + atom] * x_k.t[t_index];
                }
                for &(row, t_index) in &col_sites[atom] {
                    let j_wk = channels.z_jac[row * k_atoms + atom];
                    let c_wk = channels.logit_curvature[row * k_atoms + atom];
                    // term A + term B.
                    gamma_t[t_index] += 0.5 * dd_k * j_wk * jt_hinv_j + d_k * c_wk * x_k.t[t_index];
                }
            }
        }

        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    /// #1418: apply the EXACT stationarity-Jacobian correction `ΔC·v = (A − B)·v`
    /// to a joint `(t, β)` vector, matrix-free and per row.
    ///
    /// `A = ∇²_θθ L` is the true inner-fit Hessian; `B` is the assembled
    /// evidence/Newton operator the solver factors. They differ ONLY by the three
    /// curvature substitutions the assembly makes for stability:
    ///   1. data: `B` uses Gauss-Newton `J̃J̃ᵀ`, dropping the residual curvature
    ///      `R[a,b] = Σ_out r_out·∂²f_out/∂θ_a∂θ_b` (t–t via `jets.second`, t–β via
    ///      `jets.beta_deriv`; the decoder is linear in β so the β–β block is 0);
    ///   2. softmax: `B` uses the Gershgorin majorizer `D = diag(Σ_j|H_kj|)`,
    ///      dropping `H_entropy − D` (#1419);
    ///   3. periodic ARD: `B` uses `max(V'',0)`, dropping the negative part
    ///      `min(V'',0)` (the indefinite tail past a quarter period).
    /// `ΔC` is the sum of exactly these three deltas, each built from the SAME
    /// jets / penalty curvatures the assembly and the θ-adjoint use, so
    /// `A = B + ΔC` is the one true Hessian. Exact on BOTH the isotropic and the
    /// whitened-metric paths: the data fit is `½ r_nᵀ M_n r_n`, so the residual
    /// curvature is `Σ_out (M_n r_n)_out·∂²f_out/∂θ_a∂θ_b` — contract the
    /// metric-applied √w-scaled residual `error_metric = √w·M_n r_n` (the SAME
    /// quantity the assembly's β-tier gradient uses) against the RAW second jets
    /// `jets.second`/`jets.beta_deriv` (the same raw-jet convention the whole
    /// θ-adjoint and the Gauss-Newton `htt = J̃J̃ᵀ = J M Jᵀ` assembly use). On the
    /// isotropic path `M_n = I` so `error_metric = √w·r` and `J M Jᵀ = JJᵀ`,
    /// recovering the plain case. The softmax / ARD deltas are logit/coord-space
    /// prior curvatures and carry no output metric, so they are path-independent.
    fn apply_exact_hessian_minus_b(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let total_t = cache.delta_t_len();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();

        // Optional softmax exact-entropy-minus-majorizer delta operator (#1419).
        let softmax_delta: Option<(
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                Some((
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                    scale,
                ))
            }
            _ => None,
        };

        let mut out = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(cache.k),
        };
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        // #932 SIMD: jets are built in aligned 4-row SIMD batches through a
        // bounded (≤4-row) look-ahead window; unaligned / non-softmax / remainder
        // rows fall back to the scalar per-row path (bit-identical either way).
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment
                .try_assignments_row_for_rho_into(row, rho, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    rho,
                    jet_window_next,
                    n,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window.pop_front().expect("jet window must be non-empty");
            let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());

            // √w-scaled metric-applied per-row residual `error_metric = √w·M_n r_n`
            // (the SAME object the assembly's β-tier gradient contracts). The
            // data-fit `½ r_nᵀ M_n r_n` has residual curvature `Σ (M_n r_n)·∂²f`,
            // so this is exactly the residual contracted against the raw `∂²f`
            // jets. `M_n = I` on the isotropic path ⇒ `error_metric = √w·r`.
            fitted.fill(0.0);
            for k in 0..k_atoms {
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                let a_k = assignments[k];
                for out_col in 0..p {
                    fitted[out_col] += a_k * decoded[out_col];
                }
            }
            for out_col in 0..p {
                error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
            }
            let error_metric: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                _ => error.to_vec(),
            };

            // Local t-slice of `v` for this row.
            let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();

            // (1a) residual curvature, t–t: ΔC_tt[a,b] = ⟨r, ∂²f_ab⟩.
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    let r_ab = sae_dot(&error_metric, &jets.second[a][b]);
                    acc += r_ab * v_t[b];
                }
                out.t[base + a] += acc;
            }
            // (1b) residual curvature, t–β and β–t: ΔC_tβ[a,β] = ⟨r, ∂²f_aβ⟩.
            //      `jets.beta_deriv[a][β]` = ∂(∂f/∂β_β)/∂θ_a (the mixed second jet).
            for a in 0..q {
                for (beta_pos, channel) in border.iter().enumerate() {
                    let r_ab = sae_dot(&error_metric, &jets.beta_deriv[a][beta_pos]);
                    // t row picks up β leg of v; β row picks up t leg of v.
                    out.t[base + a] += r_ab * v.beta[channel.index];
                    out.beta[channel.index] += r_ab * v_t[a];
                }
            }

            // (2) softmax: ΔC_logit = (H_entropy − D) over the free logits, where
            // `D = diag(Σ_j|H_kj|)` is the Gershgorin majorizer the assembled `B`
            // wrote into the logit block (#1419). Adding `H_entropy − D` recovers the
            // EXACT entropy curvature `A = B + ΔC`, so the solver's exact-Hessian
            // correction differentiates the SAME operator the assembly installed.
            if let Some((_penalty, scale)) = softmax_delta.as_ref() {
                let assignment_dim = self.assignment.assignment_coord_dim();
                // #1410: the correction only contracts the ACTIVE logit slots
                // (`jets.vars` carries the row's `≤ top_k` active atoms on the
                // compact layout), so build only the active sub-block of
                // `ΔC = H_entropy − D` ENTRY-WISE rather than materialising the
                // full `K×K` `row_dense_hessian` / `row_psd_majorizer` matrices per
                // row (an `O(K²)`-per-row allocation that defeated the compact
                // contract at the LLM shape). `D` is diagonal, so it subtracts only
                // on `ka == kb`; the off-diagonal `H_entropy` entries come from the
                // shared `(a, l, m)` algebra. The softmax row `a_soft` is the one
                // irreducible `O(K)` term, computed once per row.
                // #1557 — reuse this iteration's `assignments` (bit-identical).
                let a_soft = assignments
                    .as_slice()
                    .expect("softmax assignments row must be contiguous");
                let m = softmax_majorizer_log_mean(a_soft);
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: ka } = *va else {
                        continue;
                    };
                    if ka >= assignment_dim {
                        continue;
                    }
                    let mut acc = 0.0_f64;
                    for (b, vb) in jets.vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                            continue;
                        };
                        if kb >= assignment_dim {
                            continue;
                        }
                        let h_entropy =
                            softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, *scale);
                        // `D` is the diagonal Gershgorin majorizer (#1419), so it
                        // contributes only on the diagonal `ka == kb`.
                        let delta = if ka == kb {
                            h_entropy
                                - active_softmax_gershgorin_majorizer_entry(a_soft, ka, m, *scale)
                        } else {
                            h_entropy
                        };
                        acc += delta * v_t[b];
                    }
                    out.t[base + a] += acc;
                }
            }

            // (3) periodic ARD: ΔC_coord = (V'' − max(V'',0)) = min(V'',0), diagonal.
            for (a, va) in jets.vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.hess.min(0.0);
                if neg != 0.0 {
                    out.t[base + a] += neg * v_t[a];
                }
            }
        }
        Ok(out)
    }

    /// #1418: matrix-free apply of the EXACT stationarity Jacobian `A = ∇²_θθ L`:
    /// `A v = B v + ΔC v`, the assembled arrow Hessian apply
    /// ([`apply_cached_arrow_hessian`]) plus the matrix-free dropped-curvature
    /// correction `ΔC = A − B` ([`Self::apply_exact_hessian_minus_b`]).
    fn apply_exact_hessian(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let b_v = apply_cached_arrow_hessian(cache, v.t.view(), v.beta.view())?;
        let dc_v = self.apply_exact_hessian_minus_b(rho, target, cache, v)?;
        Ok(SaeArrowVector {
            t: &b_v.t + &dc_v.t,
            beta: &b_v.beta + &dc_v.beta,
        })
    }

    /// #1418: solve `A x = rhs` for the EXACT stationarity Jacobian `A = ∇²_θθ L`
    /// via `B`-preconditioned CG ([`solve_b_preconditioned_cg`]) with the
    /// matrix-free `A v = B v + ΔC v` apply ([`Self::apply_exact_hessian`]). The
    /// IFT step `θ̂_ρ = −A⁻¹ g_ρ` must invert the EXACT `A`, not the surrogate `B`;
    /// CG converges for any `ρ(B⁻¹ΔC)`, where the earlier Neumann series diverged
    /// once the dropped curvature `ΔC = ⟨r, ∂²f⟩` grew (large unmodellable residual).
    fn solve_exact_stationarity(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        solve_b_preconditioned_cg(solver, rhs, |v| {
            self.apply_exact_hessian(rho, target, cache, v)
        })
    }

    /// Analytic SAE REML outer-ρ gradient components at the already converged
    /// inner state represented by `loss` and `cache`.
    ///
    /// The returned gradient is the assembled analytic outer derivative:
    /// explicit penalty terms, direct logdet traces, Occam terms, and the #1006
    /// implicit-state third-order correction.
    pub(crate) fn analytic_outer_rho_gradient_components(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        let n_params = rho.to_flat().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);

        explicit[0] = assignment_prior_log_strength_derivative(&self.assignment, rho)
            + self
                .learnable_ibp_forward_alpha_data_derivative(rho, target)
                .map_err(OuterGradientError::internal)?;
        // #1417: the FULL `½ tr(H⁻¹ ∂H/∂logα)` for the assignment coordinate.
        // For LEARNABLE IBP alpha the forward assignments `a_ik = σ(ℓ/τ)·π_k(α)`
        // carry an explicit α-dependence (`∂logπ_k/∂logα = k/(α+1)`), so BOTH the
        // assignment-prior Hessian AND the data Gauss-Newton blocks
        // `H_ββ`, `H_tβ`, `H_tt` depend on logα. We assemble both traces:
        //   • prior:  `assignment_log_strength_hessian_trace`,
        //   • data:   `learnable_ibp_data_logdet_alpha_trace` (#1417), using the
        //             exact `(k_a+k_b)/(α+1)` block-scaling identity.
        // For FIXED alpha (and non-IBP modes) the data term is identically zero,
        // so the fixed-alpha gradient is unchanged and exact.
        logdet_trace[0] = self
            .assignment_log_strength_hessian_trace(rho, cache, solver)
            .map_err(OuterGradientError::internal)?
            + self
                .learnable_ibp_data_logdet_alpha_trace(rho, cache, solver)
                .map_err(OuterGradientError::internal)?;

        // #1556: λ_smooth is per-atom, so the smoothness gradient block occupies
        // flat indices `1..1+K` (one per atom), not a single index 1. Each atom
        // `k` carries its own explicit penalty-energy derivative, log|H| trace,
        // and Occam-normalizer derivative.
        let k_smooth = rho.log_lambda_smooth.len();
        let lambda_smooth_vec = rho.lambda_smooth_vec();
        // Explicit `∂loss.smoothness/∂log λ_k = 0.5·λ_k·<B_k, S_k B_k>` (the
        // per-atom split). Its sum is the λ-scaled penalty energy; renormalize to
        // `loss.smoothness` so the total matches the criterion's reported energy
        // bit-for-bit (folding in any minibatch `penalty_scale` baked into it).
        let mut smooth_explicit = self.decoder_smoothness_value_per_atom(&lambda_smooth_vec);
        let smooth_explicit_sum: f64 = smooth_explicit.iter().sum();
        if smooth_explicit_sum.abs() > 0.0 {
            let renorm = loss.smoothness / smooth_explicit_sum;
            for v in smooth_explicit.iter_mut() {
                *v *= renorm;
            }
        }
        let smooth_logdet = self
            .decoder_smoothness_effective_dof_with_solver_per_atom(
                cache,
                solver,
                &lambda_smooth_vec,
            )
            .map_err(|err| OuterGradientError::InternalInvariant {
                reason: format!("analytic_outer_rho_gradient_components: {err}"),
            })?;
        let smooth_occam = self
            .reml_occam_log_lambda_smooth_derivative()
            .map_err(OuterGradientError::internal)?;
        for atom_idx in 0..k_smooth {
            explicit[1 + atom_idx] = smooth_explicit[atom_idx];
            logdet_trace[1 + atom_idx] = 0.5 * smooth_logdet[atom_idx];
            occam[1 + atom_idx] = -smooth_occam[atom_idx];
        }

        let ard_explicit = self
            .ard_log_precision_explicit_derivatives(rho)
            .map_err(OuterGradientError::internal)?;
        let ard_trace = self
            .ard_log_precision_hessian_trace(rho, cache, solver)
            .map_err(|err| OuterGradientError::InternalInvariant {
                reason: format!("analytic_outer_rho_gradient_components: {err}"),
            })?;
        let mut cursor = 1 + k_smooth;
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                explicit[cursor] = ard_explicit[k][axis];
                logdet_trace[cursor] = ard_trace[k][axis];
                cursor += 1;
            }
        }

        let gamma = self
            .logdet_theta_adjoint(rho, cache, solver)
            .map_err(OuterGradientError::internal)?;
        // #1418: the implicit-function correction is `−½·Γᵀ·θ̂_ρ` with
        // `θ̂_ρ = −A⁻¹ g_ρ`, where `A = ∇²_θθ L` is the EXACT stationarity
        // Jacobian of the inner fit — data residual curvature, exact softmax
        // entropy Hessian, exact periodic ARD curvature. The matrix the `solver`
        // factors is `B` (Gauss-Newton data curvature, softmax Fisher metric,
        // `max(V'',0)` ARD majorizers): the `½log|B|` Laplace term is consistent
        // with `Γ = ½tr(B⁻¹ ∂B/∂θ)`, but the implicit step is governed by `A`.
        // `solve_exact_stationarity` applies the TRUE `A⁻¹` via a B⁻¹-
        // preconditioned Neumann fixed point (`A = B + ΔC`,
        // `ΔC = apply_exact_hessian_minus_b`), so the correction is no longer
        // biased by `(B⁻¹ − A⁻¹)`.
        for coord in 0..n_params {
            let rhs = self
                .outer_rho_gradient_ift_rhs(rho, target, coord, cache)
                .map_err(OuterGradientError::internal)?;
            let solved = self
                .solve_exact_stationarity(rho, target, cache, solver, &rhs)
                .map_err(OuterGradientError::internal)?;
            let mut dot = 0.0_f64;
            for idx in 0..gamma.t.len() {
                dot += gamma.t[idx] * solved.t[idx];
            }
            for idx in 0..gamma.beta.len() {
                dot += gamma.beta[idx] * solved.beta[idx];
            }
            third_order_correction[coord] = -0.5 * dot;
        }

        Ok(SaeOuterRhoGradientComponents {
            explicit,
            logdet_trace,
            occam,
            third_order_correction,
        })
    }

    /// Public analytic outer-ρ gradient at a converged inner state, constructing
    /// the deflated arrow solver from the supplied cache. Use this seam from
    /// integration tests and external consumers that have a converged
    /// `(loss, cache)` from [`Self::reml_criterion_with_cache`] but no access to
    /// the crate-private `DeflatedArrowSolver`.
    pub fn analytic_outer_rho_gradient_at_converged(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<SaeOuterRhoGradientComponents, String> {
        let solver = self.outer_gradient_arrow_solver(cache, &rho.lambda_smooth_vec())?;
        self.analytic_outer_rho_gradient_components(target, rho, loss, cache, &solver)
            .map_err(|e| e.to_string())
    }

    /// Compose the SAE LAML criterion as a sum of atoms (#931 SAE pilot).
    ///
    /// This is the single seam that establishes value↔gradient coherence for
    /// the SAE objective: it runs the inner solve once via
    /// [`Self::reml_criterion_with_cache`], reads the value decomposition
    /// (`loss.total() + extra_penalty_energy`, `log|H|`, `occam`) and the
    /// matching gradient channels (`SaeOuterRhoGradientComponents`) from the
    /// SAME converged cache, and hands them to [`SaeCriterion::assemble`]. The
    /// returned criterion's [`SaeCriterion::value`] and
    /// [`SaeCriterion::gradient`] are then projections of one factorization —
    /// the outer optimizer can no longer evaluate a value path and a gradient
    /// path that disagree (the #752/#748/#901 desync class). The
    /// implicit-stationarity envelope correction (#1006's Γ term) is its own
    /// named atom, so the channel the desync class keeps dropping is visible
    /// rather than a silent zero.
    pub fn criterion_as_atoms(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeCriterion, String> {
        let (_v, loss, cache) = self.reml_criterion_with_cache(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            "criterion_as_atoms: arrow_log_det_from_cache returned None".to_string()
        })?;
        let occam = self.reml_occam_term(rho)?;
        let extra_penalty_energy = match registry {
            Some(reg) => self
                .reml_extra_penalty_value_total(reg)
                .map_err(|err| format!("SaeManifoldTerm::criterion_as_atoms: {err}"))?,
            None => 0.0,
        };
        let data_fit_priors_value = loss.total() + extra_penalty_energy;

        let solver = self.outer_gradient_arrow_solver(&cache, &rho.lambda_smooth_vec())?;
        let components =
            self.analytic_outer_rho_gradient_components(target, rho, &loss, &cache, &solver)?;
        Ok(SaeCriterion::assemble(
            data_fit_priors_value,
            log_det,
            occam,
            components.explicit,
            components.logdet_trace,
            components.occam,
            components.third_order_correction,
        ))
    }

    // [#780 line-count gate] reconstruction_dispersion + assemble_shape_uncertainty
    // + complete_born_atom_shape_bands + shape_uncertainty_without_decoder_covariance
    // (the contiguous trailing methods of this impl block) were split into the
    // sibling construction_reconstruction.rs (declared in mod.rs); callers reach
    // them bare via use super::*.
}

// [#780 line-count gate] Per-row jet / reconstruction-channel assembly for the
// streaming-exact arrow log-det lives in a sibling file as a second
// `impl SaeManifoldTerm` block, inlined here so it keeps the SAME module scope
// and private-field access. Keeps this tracked file under the 10k limit.
include!("construction_row_jet_logdet_channels.rs");

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
