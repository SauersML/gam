use super::*;

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
    /// failed (`h > ½`) and fell back to the exact chart-center Newton.
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
            border_hbb_workspace: Array2::<f64>::zeros((0, 0)),
            certificate_dispersion: None,
            curvature_walk_report: None,
            expected_evidence_gauge_deflated_directions: None,
            evidence_gauge_deflation_reanchors: 0,
            dictionary_cocollapse_reseeds: 0,
            hybrid_split_report: None,
            atom_inner_fits: None,
        })
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
    /// fixed snapshot ([`crate::terms::sae::identifiability::AtomInnerFit`]) the Riesz
    /// debiasing and Bartlett correction read.
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

        let mut inner_fits: Vec<Option<crate::terms::sae::identifiability::AtomInnerFit>> =
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
    ) -> Result<Option<crate::terms::sae::identifiability::AtomInnerFit>, String> {
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

        Ok(Some(crate::terms::sae::identifiability::AtomInnerFit {
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

    /// Install the single per-row [`RowMetric`](crate::inference::row_metric::RowMetric)
    /// that both the reconstruction likelihood and the isometry gauge read.
    /// Installing per-row output-Fisher factors here flips the provenance to
    /// `OutputFisher` *and* is the only way the gauge acquires a non-identity
    /// weight, so the two inner products cannot diverge. Passing a Euclidean
    /// metric (or never calling this) keeps the bit-identical isotropic path.
    ///
    /// The metric's row count and output dimension must match the term.
    pub fn set_row_metric(
        &mut self,
        metric: crate::inference::row_metric::RowMetric,
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
    /// Object 4 (to read the [`MetricProvenance`](crate::inference::row_metric::MetricProvenance)).
    pub fn row_metric(&self) -> Option<&crate::inference::row_metric::RowMetric> {
        self.row_metric.as_ref()
    }

    /// The per-row inner product the additive diagnostics read through: the
    /// installed [`RowMetric`](crate::inference::row_metric::RowMetric) when one
    /// was set (output-Fisher harvest present), otherwise a freshly-built
    /// Euclidean metric of the term's own `(n_obs, output_dim)` shape. Either way
    /// a metric always exists, so the diagnostics are never gated by a flag — the
    /// Euclidean fallback is the bit-identical isotropic path.
    pub(crate) fn diagnostic_metric(
        &self,
    ) -> Result<crate::inference::row_metric::RowMetric, String> {
        match self.row_metric() {
            Some(metric) => Ok(metric.clone()),
            None => {
                crate::inference::row_metric::RowMetric::euclidean(self.n_obs(), self.output_dim())
            }
        }
    }

    /// Build the additive post-fit diagnostic report for this fitted term: the
    /// two-score per-atom [`AtomTwoLensReport`](crate::inference::atom_lens::AtomTwoLensReport)
    /// (presence / behavioral coupling / discrepancy) and the residual-gauge
    /// [`ResidualGaugeReport`](crate::terms::sae::identifiability::ResidualGaugeReport)
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
    ) -> Result<SaeManifoldFitDiagnostics, String> {
        let metric = self.diagnostic_metric()?;
        let atom_two_lens = crate::inference::atom_lens::atom_two_lens(self, &metric);

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
        let ops: Vec<Option<crate::terms::sae::identifiability::OrbitPenaltyOperator>> =
            if isometry_pin_active {
                views
                    .iter()
                    .map(|view| {
                        view.as_ref().and_then(|v| {
                            crate::terms::sae::identifiability::isometry_orbit_penalty_operator(v, 1.0)
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
            crate::terms::sae::identifiability::residual_gauge_exact(&certificate_model, &views, &ops)?
        } else {
            let (curvature_gram, root_rows) = streamed_curvature.ok_or_else(|| {
                "fit_diagnostics_report: missing streamed residual-gauge curvature for unpinned exact path"
                    .to_string()
            })?;
            crate::terms::sae::identifiability::residual_gauge_exact_from_curvature_gram(
                &certificate_model,
                &views,
                &ops,
                curvature_gram,
                root_rows,
            )?
        };

        // #1097 / #1103: per-atom Riesz-debiased functionals and Bartlett smooth
        // significance, read straight off the certificate model — which carries
        // each atom's `inner_fit` snapshot when the caller harvested it via
        // [`Self::set_atom_inner_fits`] before this report. Atoms without a
        // harvested inner fit degrade their inference fields to `None` inside
        // `atom_inference_reports`, so this is always populated (one entry per
        // atom) and never gated by a flag.
        let atom_inference = crate::terms::sae::identifiability::atom_inference_reports(&certificate_model);

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
        metric: &crate::inference::row_metric::RowMetric,
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
    ) -> Vec<Option<crate::terms::sae::identifiability::AtomParameterView>> {
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
                Some(crate::terms::sae::identifiability::AtomParameterView {
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
    /// [`FittedSaeManifold`](crate::terms::sae::identifiability::FittedSaeManifold) the
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
        metric: crate::inference::row_metric::RowMetric,
        per_atom_ard_variances: Option<&[Option<Array1<f64>>]>,
        isometry_pin_active: bool,
    ) -> Result<
        (
            crate::terms::sae::identifiability::FittedSaeManifold,
            Option<(Array2<f64>, usize)>,
        ),
        String,
    > {
        use crate::terms::sae::identifiability::{AtomTopology, FittedAtom, FittedSaeManifold};

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
                    SaeAtomBasisKind::Duchon
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
        metric: &crate::inference::row_metric::RowMetric,
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
    /// [`crate::solver::arrow_schur::JacobiPreconditioner`] builds one dense
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

    /// Decide whether the sparse per-row active-set layout is engaged for the
    /// dense-weight assignment modes (softmax / IBP-MAP), and if so derive the
    /// per-row active-atom cap and magnitude cutoff.
    ///
    /// The decision is auto-derived from the problem size and the
    /// device/host working-set budget — never a CLI flag or kwarg. JumpReLU is
    /// not handled here (it always uses its structural gate via
    /// [`SaeRowLayout::from_jumprelu`]). The dense Gauss-Newton data Gram `G`
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
        let budget = match crate::gpu::runtime::GpuRuntime::global() {
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
        let linear_images: std::collections::HashMap<
            usize,
            &crate::terms::sae::hybrid_split::AtomLinearImage,
        > = if collapse {
            self.hybrid_split_report
                .as_ref()
                .map(|report| {
                    report
                        .verdicts
                        .iter()
                        .filter_map(|v| v.linear_image.as_ref().map(|img| (img.atom_idx, img)))
                        .collect()
                })
                .unwrap_or_default()
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
                    // at this row's fitted on-atom coordinate.
                    let t = self.assignment.coords[atom_idx].as_matrix()[[row, 0]];
                    image.fill_row(t, &mut g_buf);
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
    ) -> Option<&crate::terms::sae::hybrid_split::SaeHybridSplitReport> {
        self.hybrid_split_report.as_ref()
    }

    /// Build the #1026 curved-vs-linear hybrid-split report by adjudicating each
    /// eligible `d = 1` atom's fitted curved image against its straight (linear
    /// special-case) sub-model on the common rank-aware Laplace evidence scale.
    ///
    /// Both candidates reconstruct the SAME fitted decoded image over the SAME
    /// assigned rows; the linear candidate's deviance is the exact penalized-LS
    /// residual of the best straight line through those points (the collapsed
    /// linear lane — closed form, NOT the broken euclidean outer fit path of
    /// #1051). Eligible atoms are `d = 1` atoms with an installed evaluator at
    /// the full curvature dial (`homotopy_eta == 1.0`) whose live coordinate dim
    /// still matches the atom's latent dim.
    pub fn compute_hybrid_split_report(
        &self,
        rho: &SaeManifoldRho,
        target: Option<ArrayView2<'_, f64>>,
    ) -> Result<Option<crate::terms::sae::hybrid_split::SaeHybridSplitReport>, String> {
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
        // Per-row assignment masses (once), so each atom's weighted straight-line
        // fit uses the same row weighting the joint reconstruction loss does.
        let mut weights: Vec<Array1<f64>> = Vec::with_capacity(n);
        for row in 0..n {
            weights.push(self.assignment.try_assignments_row_for_rho(row, rho)?);
        }
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
        let weights_for = |atom_idx: usize| -> Array1<f64> {
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
        let manifold_for = |atom_idx: usize| -> crate::terms::latent_coord::LatentManifold {
            self.assignment.coords[atom_idx].manifold().clone()
        };
        crate::terms::sae::hybrid_split::build_hybrid_split_report(
            &self.atoms,
            eligible.into_iter(),
            coords_for,
            weights_for,
            decoded_for,
            manifold_for,
            delta_ev_for,
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
    /// `h ≤ ½` falls back to the chart-center-start exact Newton encode
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
    ) -> Result<Vec<crate::terms::sae_encode_atlas::EncodeResult>, String> {
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

        let atlas = crate::terms::sae_encode_atlas::EncodeAtlas::build(
            &self.atoms,
            &amplitude_bound,
            target_norm_bound,
            crate::terms::sae_encode_atlas::AtlasConfig::default(),
        )?;

        // Per-atom amortized encode with a certificate-gated exact-solve fallback:
        // a row whose distilled prediction fails `h ≤ ½` is retried from the
        // chart-center start (the non-amortized exact Newton); a row that still
        // cannot be certified stays flagged for the upstream multi-start solve.
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
            results.push(crate::terms::sae_encode_atlas::EncodeResult::from_rows(
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
    ) -> Result<Vec<crate::terms::sae_encode_atlas::EncodeResult>, String> {
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
    ///   exact chart-center Newton. This is the encoder's *certifiable coverage*
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
            let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                format!(
                    "SaeManifoldTerm::amortized_encoder_consistency: atom {atom_idx} has no basis evaluator"
                )
            })?;
            let result = &encodes[atom_idx];
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
        let reml_scale = reml.abs().max(1.0);
        let w_recon = COTRAIN_RECON_WEIGHT * reml_scale;
        let w_cert = COTRAIN_CERT_WEIGHT * reml_scale;
        let cotrained = reml
            + w_recon * consistency.recon_consistency
            + w_cert * consistency.uncertified_fraction;
        Ok((cotrained, loss, consistency))
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
            |row: usize, g_buf: &mut [f64], fitted_row: &mut [f64]| -> Result<f64, String> {
                let a = self.assignment.try_assignments_row_for_rho(row, rho)?;
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
                    || (vec![0.0_f64; p], vec![0.0_f64; p]),
                    |(g_buf, fitted_row), idxs| {
                        let mut acc = 0.0_f64;
                        for row in idxs {
                            acc += row_data_fit(row, g_buf, fitted_row)?;
                        }
                        Ok(acc)
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
            let mut total = 0.0_f64;
            for row in 0..n {
                total += row_data_fit(row, &mut g_buf, &mut fitted_row)?;
            }
            total
        };
        let assignment_sparsity = assignment_prior_value(&self.assignment, rho);
        let smoothness = penalty_scale * self.decoder_smoothness_value(rho.lambda_smooth());
        let ard = self.ard_value(rho)?;
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
            evidence_gauge_deflated_directions: 0,
        })
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
        Ok(total)
    }

    pub(crate) fn decoder_smoothness_value(&self, lambda_smooth: f64) -> f64 {
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
        for (atom, sb) in self.atoms.iter().zip(sb_all.iter()) {
            acc += 0.5 * lambda_smooth * (&atom.decoder_coefficients * sb).sum();
        }
        acc
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
        let lambda_smooth = rho.lambda_smooth() * penalty_scale;
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
            crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                Some((
                    crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
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
                    scaled_s[[i, j]] = lambda_smooth * s_ij;
                }
            }
            // Gradient: g[beta_i] += (λ S_k B_k)[i, out_col]. The (m×m)·(m×p)
            // GEMM `½(S+Sᵀ)·B_k` was computed in the multi-GPU batch above; here
            // we only apply the scalar `lambda_smooth`.
            let sb = &sym_sb_all[atom_idx] * lambda_smooth;
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
            AssignmentMode::Softmax { .. } => None,
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
                        // Absolute cutoff = relative_cutoff · max row peak, so a
                        // single threshold drops sub-1e-3 mass across all rows.
                        let peak = assignments_all
                            .iter()
                            .flat_map(|a| a.iter())
                            .fold(0.0_f64, |m, &v| m.max(v.abs()));
                        let cutoff = relative_cutoff * peak;
                        Some(SaeRowLayout::from_dense_weights(
                            &assignments_all,
                            k_active_cap,
                            cutoff,
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
        let admission_plan = self
            .streaming_plan()
            .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
            .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        let dense_beta_curvature = admission_plan.direct_admitted
            && !(frames_engaged && beta_dim > dense_beta_penalty_probe_max_dim);
        let row_htbeta_dim = if frames_engaged {
            self.factored_border_dim()
        } else {
            beta_dim
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
                Some(ibp_stick_breaking_prior(k_atoms, alpha).to_vec())
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
        }
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let row_results: Vec<SaeAssemblyRow> = (0..n)
            .into_par_iter()
            .map_init(
                || RowScratch {
                    decoded: Array2::<f64>::zeros((k_atoms, p)),
                    dg_buf: vec![0.0_f64; p],
                    fitted: Array1::<f64>::zeros(p),
                    error: Array1::<f64>::zeros(p),
                    error_white: vec![0.0_f64; w_dim],
                    error_metric: Array1::<f64>::zeros(p),
                    jac_white: vec![0.0_f64; q * w_dim.max(p)],
                    decoded_scratch: vec![0.0_f64; p],
                },
                |scratch, row| -> Result<SaeAssemblyRow, String> {
                    let RowScratch {
                        decoded,
                        dg_buf,
                        fitted,
                        error,
                        error_white,
                        error_metric,
                        jac_white,
                        decoded_scratch,
                    } = scratch;
                    let mut gb_delta: Vec<(usize, f64)> = Vec::new();
                    let mut g_blocks: SaeGBlocks = std::collections::BTreeMap::new();
                    let assignments = self.assignment.try_assignments_row_for_rho(row, rho)?;
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
                            for &atom_idx in active {
                                let a_k = assignments[atom_idx];
                                self.atoms[atom_idx]
                                    .fill_decoded_row(row, decoded_scratch.as_mut_slice());
                                for out_col in 0..p {
                                    decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
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
                                    decoded_k: decoded.row(k),
                                    fitted: fitted.view(),
                                    ibp_prior: ibp_prior_slice,
                                    compact_index: j,
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
                        for (j, &k) in active.iter().enumerate() {
                            block.gt[j] += assignment_grad[assignment_base + k];
                            block.htt[[j, j]] += assignment_hdiag[assignment_base + k];
                        }
                    } else {
                        for free_idx in 0..assignment_dim {
                            block.gt[free_idx] += assignment_grad[assignment_base + free_idx];
                        }
                        if let Some((penalty, scale)) = softmax_dense.as_ref() {
                            // #1038: write the EXACT dense entropy Hessian (diagonal +
                            // off-diagonals) onto the row's logit block. Softmax uses
                            // the REDUCED K−1 free-logit chart (the last reference logit
                            // is fixed at 0, `assignment_coord_dim() = K−1`), which
                            // already removes the shift-gauge null. Holding z_{K-1}
                            // fixed, the reduced Hessian over the free logits 0..K−1 is
                            // exactly the top-left (K−1)×(K−1) submatrix of the full
                            // K×K dense entropy Hessian (the fixed logit contributes no
                            // row/column to the free curvature). Summing it onto the
                            // data-fit `htt` keeps the block PD for the Cholesky factor,
                            // and the dense Cholesky makes `log|H|` carry it exactly (no
                            // separate Woodbury — `htt` is already a small dense per-row
                            // factor). Its diagonal equals `assignment_hdiag` for these
                            // logits, so we skip the diagonal-only add above.
                            let row_logits: Vec<f64> = (0..k_atoms)
                                .map(|k| self.assignment.logits[[row, k]])
                                .collect();
                            let h_dense = penalty.row_dense_hessian(&row_logits, *scale);
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
                                let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
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
                                let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
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
                    let mut a_phi: Vec<(usize, f64)> = Vec::with_capacity(row_active.len() * 4);
                    // Per-active-atom weighted basis row `a_k · φ_k[·]`, retained so the
                    // data Gram blocks can be accumulated as clean per-atom-pair outer
                    // products `(a_k φ_k) (a_{k'} φ_{k'})ᵀ`.
                    let mut weighted_phi: Vec<(usize, Vec<f64>)> =
                        Vec::with_capacity(row_active.len());
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
                            gb_delta
                                .push((beta_base_i + out_col, j_beta_i * error_metric[out_col]));
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
                                    let hrow_slice =
                                        hrow.as_slice_mut().expect("htbeta row is contiguous");
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
                    let (kron_a_phi, kron_jac) = if !frames_engaged {
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
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let mut g_blocks: SaeGBlocks = std::collections::BTreeMap::new();
        let mut kron_a_phi: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut kron_jac: Vec<Vec<f64>> = Vec::with_capacity(n);
        for (row, row_result) in row_results.into_iter().enumerate() {
            assert_eq!(
                row, row_result.row,
                "parallel SAE row assembly returned rows out of order"
            );
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
            if !frames_engaged {
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
                        let htbeta_e = sys.rows[row_idx].htbeta.clone();
                        sys.rows[row_idx].gt =
                            manifold_i.project_gradient_to_tangent(t_i, gt_e.view());
                        sys.rows[row_idx].htt =
                            manifold_i.riemannian_hessian_matrix(t_i, gt_e.view(), htt_e.view());
                        sys.rows[row_idx].htbeta = manifold_i
                            .project_matrix_columns_to_gradient_tangent(
                                t_i,
                                gt_e.view(),
                                htbeta_e.view(),
                            );
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
        let device_rows = if frames_engaged {
            None
        } else {
            Some((kron_a_phi.clone(), kron_jac.clone()))
        };
        if !frames_engaged {
            let kron = Arc::new(SaeKroneckerRows::new(p, kron_a_phi, kron_jac));
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
                let registry =
                    analytic_penalties.expect("deferred beta curvature requires registry");
                let hbb_c = self.build_factored_beta_penalty_curvature(
                    registry,
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
        //     z_jac[i·K + k]` and `global_t_index = sys.row_offsets[i] + k`. IBP
        //     is a DENSE assignment mode (`assignment_coord_dim() = K`,
        //     `last_row_layout = None`), so atom `k`'s logit slot is local
        //     position `k` of row `i`'s block — exactly the `(base + pos)` index
        //     the gradient path records in `ibp_logit_sites`
        //     (`row_vars_for_cache_row` maps `vars[atom] = Logit { atom }`). This
        //     pins the `U`-column convention bit-for-bit to the consumer.
        if let Some(channels) = ibp_assignment_third_channels(&self.assignment, rho)? {
            let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(n * k_atoms);
            for row in 0..n {
                let start = row * k_atoms;
                let g_base = sys.row_offsets[row];
                for k in 0..k_atoms {
                    let z_prime = channels.z_jac[start + k];
                    entries.push((g_base + k, k, z_prime));
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
        let m = s.ncols();
        if m == 0 {
            return Ok(0);
        }
        // Symmetrise defensively — `smooth_penalty` is conceptually symmetric
        // but may be stored with tiny asymmetry from assembly arithmetic.
        let mut sym = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
            }
        }
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

    /// True REML criterion for the SAE term at a FIXED ρ.
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
    /// `½ p·log|S|_+` piece is ρ-independent. ALL ρ-independent additive
    /// constants (the `2π` Laplace constant, the base `½ p·log|S|_+` penalty
    /// logdet, the assignment-prior normaliser) are DROPPED here: they shift
    /// `V` by a constant and do not affect the ρ-argmin the engine seeks.
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
            let mut rho_fixed = rho.clone();
            let loss = self.run_joint_fit_arrow_schur(
                target,
                &mut rho_fixed,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )?;
            let extra_penalty_energy = match registry {
                Some(reg) => self
                    .reml_extra_penalty_value_total(reg)
                    .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?,
                None => 0.0,
            };
            Ok((loss.total() + extra_penalty_energy, loss))
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
            "SaeManifoldTerm::reml_criterion: arrow_log_det_from_cache returned None at \
             ridge=0 Direct mode (no dense Schur factor); the joint Hessian log-det is \
             required for the Laplace normaliser"
                .to_string()
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
                // that NEVER STABILIZES — an oscillating / runaway quotient
                // dimension that re-anchors without converging, signalling a
                // truly ill-posed evidence surface rather than a finite number of
                // structural events. Each atom can contribute at most one
                // birth/rank-reduction plus `SAE_ATOM_COLLAPSE_RESEED_BUDGET`
                // per-atom reseeds, and the whole dictionary can co-collapse and be
                // multi-started at most `SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET`
                // times (each touching all K atoms), so a healthy walk re-anchors a
                // bounded number of times. Exceeding that bound is the structural
                // pathology, and there we refuse to compare.
                self.evidence_gauge_deflation_reanchors += 1;
                let reanchor_budget = self
                    .k_atoms()
                    .saturating_mul(
                        SAE_ATOM_COLLAPSE_RESEED_BUDGET
                            + SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET
                            + 1,
                    )
                    .saturating_add(1);
                if self.evidence_gauge_deflation_reanchors > reanchor_budget {
                    return Err(format!(
                        "SaeManifoldTerm::reml_criterion: row-gauge evidence deflation count \
                         re-anchored {} times (last {expected}->{count}) within one optimization, \
                         exceeding the {reanchor_budget}-event budget for {} atoms; the quotient \
                         dimension is not stabilizing, refusing to compare Laplace normalizers",
                        self.evidence_gauge_deflation_reanchors,
                        self.k_atoms()
                    ));
                }
                log::warn!(
                    "SaeManifoldTerm::reml_criterion: per-row evidence deflation count changed \
                     {expected}->{count} (a legitimate quotient-dimension event — atom \
                     birth/reseed/rank-reduction across the ρ-walk); re-anchoring the Laplace \
                     normalizer comparison to the new dimension (re-anchor {}/{reanchor_budget})",
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
                        rho_fixed.lambda_smooth(),
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
                rho_fixed.lambda_smooth(),
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
    ) -> Result<DeflatedArrowSolver<'a>, String> {
        let Err(conditioning_err) = Self::outer_gradient_conditioning_error(cache) else {
            return Ok(DeflatedArrowSolver::plain(cache));
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(conditioning_err);
        };
        if !(max_pivot.is_finite() && max_pivot > 0.0) {
            return Err(conditioning_err);
        }

        let full_len = cache.delta_t_len() + cache.k;
        let mut raw_gauges = Vec::new();
        for gauge in self.dense_step_gauge_vectors()? {
            if gauge.len() != full_len {
                continue;
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            raw_gauges.push(gauge);
        }
        // #1051: admit the β (decoder) coordinate basis as additional deflation
        // candidates when the block is small enough to eigendecompose cheaply.
        // A rank-deficient decoder design (e.g. a euclidean-1D line in a p=2
        // ambient: decoder column rank 1 of 3) puts a genuine near-null
        // direction of the joint Hessian in the β block, OUTSIDE the closed-form
        // chart gauge orbit. Feeding the β basis into the same Rayleigh
        // eigendecomposition below lets that flat direction be identified and
        // Faddeev-Popov-deflated exactly like a chart gauge, so the analytic
        // outer gradient becomes well-defined instead of rejecting the trial ρ.
        // The Rayleigh floor still keeps only genuinely flat (sub-floor)
        // directions, so a well-conditioned decoder is unaffected.
        let delta_t_len = cache.delta_t_len();
        if cache.k > 0 && cache.k <= SAE_OUTER_GRADIENT_BETA_NULL_PROBE_MAX_DIM {
            for beta_idx in 0..cache.k {
                let mut unit = Array1::<f64>::zeros(full_len);
                unit[delta_t_len + beta_idx] = 1.0;
                raw_gauges.push(unit);
            }
        }
        if raw_gauges.is_empty() {
            return Err(conditioning_err);
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
            return Err(conditioning_err);
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
                Err(_) => return Err(conditioning_err),
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
        let (evals, evecs) = h_span
            .eigh(Side::Lower)
            .map_err(|_| conditioning_err.clone())?;
        let strict_gauge_floor = SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR * max_pivot;
        let fallback_gauge_floor = SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR.sqrt() * max_pivot;
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
            let mut best_idx = None;
            let mut best_rayleigh = f64::INFINITY;
            for eig_idx in 0..evals.len() {
                let rayleigh = evals[eig_idx];
                if rayleigh.is_finite()
                    && rayleigh < best_rayleigh
                    && rayleigh <= fallback_gauge_floor
                {
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
            return Err(conditioning_err);
        }

        // Quotient-geometry gauge fixing: add stiffness only along the closed-form
        // gauge orbit (Faddeev-Popov style). Components orthogonal to that orbit
        // are identical to the original inverse solve, while gauge components are
        // bounded at the Hessian scale `max_pivot`.
        DeflatedArrowSolver::from_orthonormal_gauges(cache, orthonormal, max_pivot)
            .map_err(|_| conditioning_err)
    }

    pub(crate) fn outer_gradient_conditioning_error(
        cache: &ArrowFactorCache,
    ) -> Result<(), String> {
        let pivot = arrow_factor_min_pivot(cache);
        let Some(min_pivot) = pivot.min_pivot else {
            return Err(
                "analytic outer gradient undefined at this rho: joint Hessian numerically \
                 singular (no cached Cholesky pivots)"
                    .to_string(),
            );
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(
                "analytic outer gradient undefined at this rho: joint Hessian numerically \
                 singular (no cached Cholesky pivot scale)"
                    .to_string(),
            );
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
        Err(format!(
            "analytic outer gradient undefined at this rho: joint Hessian numerically singular \
             (min/max pivot ratio {ratio:.3e} < floor {floor:.3e}; min pivot {min_pivot:.3e}, \
             max pivot {max_pivot:.3e})",
            floor = SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR,
        ))
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
        let mut penalized_channel_dim = 0usize;
        for atom in &self.atoms {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            // Penalized decoder dimension: `r_k` coordinate channels carry the
            // `S_k` roughness penalty (full-`B` path ⇒ `r_k == p`).
            penalized_channel_dim += atom.border_frame_rank() * rank_s;
        }
        // Profiled Grassmann dimensions enter the Laplace evidence dimension
        // count with the OPPOSITE sign of the penalty Occam term (they are
        // free, unpenalized-by-`S` profiled directions), so `−occam` adds
        // `+½ Σ r(p−r) log λ` to the criterion `V` — the honesty correction.
        let grassmann_dim = self.grassmann_evidence_dimension();
        let occam_penalty = 0.5 * (penalized_channel_dim as f64) * rho.log_lambda_smooth;
        let frame_dim_term = 0.5 * (grassmann_dim as f64) * rho.log_lambda_smooth;
        // `V = … − occam`, so we want the net occam to SUBTRACT the penalty
        // normalizer and ADD the frame-dimension count. Returning
        // `occam_penalty − frame_dim_term` achieves that after the caller's
        // `− occam`.
        Ok(occam_penalty - frame_dim_term)
    }

    pub(crate) fn reml_occam_log_lambda_smooth_derivative(&self) -> Result<f64, String> {
        let mut penalized_channel_dim = 0usize;
        for atom in &self.atoms {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            penalized_channel_dim += atom.border_frame_rank() * rank_s;
        }
        let grassmann_dim = self.grassmann_evidence_dimension();
        Ok(0.5 * ((penalized_channel_dim as f64) - (grassmann_dim as f64)))
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
            let (chunk_log_det_tt, chunk_schur) = streaming
                .reduced_schur_and_log_det_tt(0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            log_det_tt += chunk_log_det_tt;
            for row in 0..border_dim {
                for col in 0..border_dim {
                    schur_acc[[row, col]] += chunk_schur[[row, col]];
                }
            }
            start = end;
        }
        let log_det_schur = StreamingArrowSchur::reduced_schur_log_det(&schur_acc, &options)
            .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
        Ok(log_det_tt + log_det_schur)
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
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
        let n = self.n_obs();
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
                            traces[k][axis] +=
                                0.5 * inv_diag[row_base + block_start + axis] * prior.hess.max(0.0);
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
                            traces[k][axis] +=
                                0.5 * inv_diag[row_base + block_start + axis] * prior.hess.max(0.0);
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Decoder smoothness penalty quadratic form `Σ_k Σ_oc B_k[:,oc]ᵀ S_k B_k[:,oc]`.
    ///
    /// This is `βᵀ (⊕_k S_k ⊗ I_p) β` — the un-scaled (λ-free) penalty energy
    /// in the flat β layout, the denominator of the λ_smooth Fellner-Schall
    /// update. `S_k` is symmetrised defensively (as the assembler does).
    pub(crate) fn decoder_smoothness_quadratic_form(&self) -> f64 {
        // `Σ_k Σ_oc B_k[:,oc]ᵀ ½(S_k+S_kᵀ) B_k[:,oc]` = `Σ_k <B_k, ½(S_k+S_kᵀ)·B_k>`.
        // The per-atom `½(S+Sᵀ)·B_k` GEMMs are independent, so they ride the
        // multi-GPU batched smoothness GEMM (uniform-shape tiles across every
        // device) with an exact per-atom CPU fallback.
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, true);
        let mut acc = 0.0_f64;
        for (atom, sb) in self.atoms.iter().zip(sb_all.iter()) {
            acc += (&atom.decoder_coefficients * sb).sum();
        }
        acc
    }

    /// Effective penalized dof of the decoder smoothness penalty:
    /// `tr(S_β⁻¹ · M)` with `M = ⊕_k (λ_smooth · S_k) ⊗ I_p` embedded in the
    /// flat β layout, where `S_β⁻¹ = (H⁻¹)_ββ` is the Schur-complement inverse.
    ///
    /// Built per keystone's documented pattern on
    /// [`ArrowFactorCache::schur_inverse_apply`]:
    /// `tr(S_β⁻¹ M) = Σ_col e_colᵀ S_β⁻¹ M e_col`. Column `(k, μ, oc)` of `M`
    /// (global index `off_k + μ·p + oc`) is `λ·S_k[:,μ] ⊗ e_oc` — nonzero only
    /// at `off_k + ν·p + oc` for `ν in 0..M_k` — so we materialise just that
    /// sparse K-vector, apply `S_β⁻¹`, and read back `result[col]`. The
    /// `⊗ I_p` only couples equal `oc`, but `S_β` itself couples channels
    /// through the data-fit block, so all `p` channels are summed (no
    /// channel-block-identity shortcut). Total cost `beta_dim` Schur solves.
    pub(crate) fn decoder_smoothness_effective_dof(
        &self,
        cache: &ArrowFactorCache,
        lambda_smooth: f64,
    ) -> Result<f64, ArrowSchurError> {
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
        let mut trace = 0.0_f64;
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda_smooth * s_nu_mu;
                    }
                    let z = cache.schur_inverse_apply(m_col.view())?;
                    trace += z[col];
                }
            }
        }
        Ok(trace)
    }

    pub(crate) fn decoder_smoothness_effective_dof_with_solver(
        &self,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        lambda_smooth: f64,
    ) -> Result<f64, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache's β block is the FACTORED border when frames
        // are active (`cache.k == factored_border_dim`), so the smoothness edf
        // trace `tr((H⁻¹)_ββ · M)` is taken over the same factored layout, with
        // `M = ⊕_k (λ S_k) ⊗ I_{r_k}` at the factored offsets (the `U_kᵀU_k = I`
        // collapse means the per-coordinate-channel penalty is `λ S_k`, exactly
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
        let mut trace = 0.0_f64;
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    // M[:,col] = λ · S_k[:,mu] ⊗ e_oc (nonzero at off+ν·r+oc).
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda_smooth * s_nu_mu;
                    }
                    let zero_t = Array1::<f64>::zeros(cache.delta_t_len());
                    let z = solver.solve(zero_t.view(), m_col.view())?.beta;
                    trace += z[col];
                }
            }
        }
        Ok(trace)
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
        // desync the ρ-gradient from the criterion. (Softmax uses the dense
        // `None` layout, so logit positions index atoms directly.)
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
            let penalty = crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                k_atoms,
                temperature,
            );
            // Softmax uses the reduced K−1 free-logit chart: only positions
            // 0..K−1 are free logit coordinates (last reference logit fixed), and
            // the reduced `∂H/∂ρ` over the free logits is the top-left
            // (K−1)×(K−1) submatrix of the full dense block. Contract it against
            // the matching per-row selected-inverse block.
            let assignment_dim = self.assignment.assignment_coord_dim();
            let total_t = cache.delta_t_len();
            let mut trace = 0.0_f64;
            for row in 0..self.n_obs() {
                let row_base = cache.row_offsets[row];
                let q = cache.row_dims[row];
                let logit_dim = assignment_dim.min(q);
                let row_logits: Vec<f64> = (0..k_atoms)
                    .map(|k| self.assignment.logits[[row, k]])
                    .collect();
                // ∂H/∂ρ over this row's free-logit block (position j ↔ atom j).
                let dh_rho = penalty.row_dense_hessian(&row_logits, scale);
                for kj in 0..logit_dim {
                    let mut rhs_t = Array1::<f64>::zeros(total_t);
                    let rhs_beta = Array1::<f64>::zeros(cache.k);
                    rhs_t[row_base + kj] = 1.0;
                    let solved = solver
                        .solve(rhs_t.view(), rhs_beta.view())
                        .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
                    for ki in 0..logit_dim {
                        // trace += (H⁻¹)_{ki,kj} · (∂H/∂ρ)_{kj,ki}; dh_rho symmetric.
                        trace += solved.t[row_base + ki] * dh_rho[[kj, ki]];
                    }
                }
            }
            return Ok(0.5 * trace);
        }
        let hdiag = assignment_prior_log_strength_hdiag(&self.assignment, rho)?;
        if hdiag.is_empty() {
            return Ok(0.0);
        }
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
        let assignment_dim = self.assignment.assignment_coord_dim();
        let mut trace = 0.0_f64;
        for row in 0..self.n_obs() {
            let row_base = cache.row_offsets[row];
            let assignment_base = row * k_atoms;
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                        trace += inv_diag[row_base + pos] * hdiag[assignment_base + atom];
                    }
                }
                None => {
                    for free_idx in 0..assignment_dim {
                        trace += inv_diag[row_base + free_idx] * hdiag[assignment_base + free_idx];
                    }
                }
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
        let prior = ibp_stick_breaking_prior(k_atoms, alpha);
        let mut dprior = Array1::<f64>::zeros(k_atoms);
        for k in 0..k_atoms {
            dprior[k] = prior[k] * k as f64 / (alpha + 1.0);
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
        let mut total = 0.0_f64;
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row_for_rho(row, rho)?;
            fitted.fill(0.0);
            f_rho.fill(0.0);
            for k in 0..k_atoms {
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                let sigma = assignments[k] / prior[k];
                let da_rho = sigma * dprior[k];
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
        let prior = ibp_stick_breaking_prior(k_atoms, alpha);
        let mut dprior = Array1::<f64>::zeros(k_atoms);
        for k in 0..k_atoms {
            dprior[k] = prior[k] * k as f64 / (alpha + 1.0);
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
        for row in 0..self.n_obs() {
            let assignments = self.assignment.try_assignments_row_for_rho(row, rho)?;
            fitted.fill(0.0);
            f_rho.fill(0.0);
            for k in 0..k_atoms {
                self.atoms[k].fill_decoded_row(row, &mut decoded_rows[k]);
                let sigma = assignments[k] / prior[k];
                let da_rho = sigma * dprior[k];
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

    pub(crate) fn gate_derivatives_for_row(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        assignments: ArrayView1<'_, f64>,
        vars: &[SaeLocalRowVar],
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>), String> {
        let k_atoms = self.k_atoms();
        let q = vars.len();
        let mut dz = vec![vec![0.0_f64; k_atoms]; q];
        let mut d2z = vec![vec![vec![0.0_f64; k_atoms]; q]; q];
        match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                let inv_tau = 1.0 / temperature;
                for (a_idx, var_a) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: j } = *var_a else {
                        continue;
                    };
                    for k in 0..k_atoms {
                        let indicator = if k == j { 1.0 } else { 0.0 };
                        dz[a_idx][k] = assignments[k] * (indicator - assignments[j]) * inv_tau;
                    }
                }
                for (a_idx, var_a) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: j } = *var_a else {
                        continue;
                    };
                    for (b_idx, var_b) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: l } = *var_b else {
                            continue;
                        };
                        for k in 0..k_atoms {
                            let ikl = if k == l { 1.0 } else { 0.0 };
                            let ikj = if k == j { 1.0 } else { 0.0 };
                            let ijl = if j == l { 1.0 } else { 0.0 };
                            d2z[a_idx][b_idx][k] = assignments[k]
                                * ((ikl - assignments[l]) * (ikj - assignments[j])
                                    - assignments[j] * (ijl - assignments[l]))
                                * inv_tau
                                * inv_tau;
                        }
                    }
                }
            }
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => {
                let effective_alpha = self
                    .assignment
                    .mode
                    .resolved_ibp_alpha(rho)
                    .unwrap_or(alpha);
                let prior = ibp_stick_breaking_prior(k_atoms, effective_alpha);
                let inv_tau = 1.0 / temperature;
                for (idx, var) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *var else {
                        continue;
                    };
                    let (_z, d1, d2) =
                        sae_sigmoid_derivatives_from_value(assignments[atom], inv_tau, prior[atom]);
                    dz[idx][atom] = d1;
                    d2z[idx][idx][atom] = d2;
                }
            }
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => {
                let inv_tau = 1.0 / temperature;
                let logits = self.assignment.logits.row(row);
                for (idx, var) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *var else {
                        continue;
                    };
                    if logits[atom] <= threshold {
                        continue;
                    }
                    let (_z, d1, d2) =
                        sae_sigmoid_derivatives_from_value(assignments[atom], inv_tau, 1.0);
                    dz[idx][atom] = d1;
                    d2z[idx][idx][atom] = d2;
                }
            }
        }
        Ok((dz, d2z))
    }

    pub(crate) fn decoded_second_row(
        atom: &SaeManifoldAtom,
        second_jet: &Array4<f64>,
        row: usize,
        axis_a: usize,
        axis_b: usize,
        out: &mut [f64],
    ) {
        out.fill(0.0);
        for basis_col in 0..atom.basis_size() {
            let d2phi = second_jet[[row, basis_col, axis_a, axis_b]];
            if d2phi == 0.0 {
                continue;
            }
            for out_col in 0..atom.output_dim() {
                out[out_col] += d2phi * atom.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }

    pub(crate) fn row_jets_for_logdet(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        vars: Vec<SaeLocalRowVar>,
        assignments: ArrayView1<'_, f64>,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
    ) -> Result<SaeRowJets, String> {
        let p = self.output_dim();
        let q = vars.len();
        let k_atoms = self.k_atoms();
        let sqrt_row_w = self
            .row_loss_weights
            .as_deref()
            .map_or(1.0, |w| w[row].sqrt());
        let (dz, d2z) = self.gate_derivatives_for_row(rho, row, assignments, &vars)?;

        let mut decoded = vec![vec![0.0_f64; p]; k_atoms];
        let mut d1: Vec<Vec<Vec<f64>>> = self
            .atoms
            .iter()
            .map(|atom| vec![vec![0.0_f64; p]; atom.latent_dim])
            .collect();
        let mut d2: Vec<Vec<Vec<Vec<f64>>>> = self
            .atoms
            .iter()
            .map(|atom| vec![vec![vec![0.0_f64; p]; atom.latent_dim]; atom.latent_dim])
            .collect();
        let mut scratch = vec![0.0_f64; p];
        for k in 0..k_atoms {
            self.atoms[k].fill_decoded_row(row, &mut decoded[k]);
            for axis in 0..self.atoms[k].latent_dim {
                self.atoms[k].fill_decoded_derivative_row(row, axis, &mut d1[k][axis]);
            }
            for axis_a in 0..self.atoms[k].latent_dim {
                for axis_b in 0..self.atoms[k].latent_dim {
                    Self::decoded_second_row(
                        &self.atoms[k],
                        &second_jets[k],
                        row,
                        axis_a,
                        axis_b,
                        &mut scratch,
                    );
                    d2[k][axis_a][axis_b].clone_from_slice(&scratch);
                }
            }
        }

        let mut first = vec![vec![0.0_f64; p]; q];
        for (idx, var) in vars.iter().enumerate() {
            match *var {
                SaeLocalRowVar::Logit { .. } => {
                    for k in 0..k_atoms {
                        let coeff = dz[idx][k] * sqrt_row_w;
                        if coeff == 0.0 {
                            continue;
                        }
                        for out_col in 0..p {
                            first[idx][out_col] += coeff * decoded[k][out_col];
                        }
                    }
                }
                SaeLocalRowVar::Coord { atom, axis } => {
                    let coeff = assignments[atom] * sqrt_row_w;
                    for out_col in 0..p {
                        first[idx][out_col] = coeff * d1[atom][axis][out_col];
                    }
                }
            }
        }

        let mut second = vec![vec![vec![0.0_f64; p]; q]; q];
        for a in 0..q {
            for b in 0..q {
                match (vars[a], vars[b]) {
                    (SaeLocalRowVar::Logit { .. }, SaeLocalRowVar::Logit { .. }) => {
                        for k in 0..k_atoms {
                            let coeff = d2z[a][b][k] * sqrt_row_w;
                            if coeff == 0.0 {
                                continue;
                            }
                            for out_col in 0..p {
                                second[a][b][out_col] += coeff * decoded[k][out_col];
                            }
                        }
                    }
                    (SaeLocalRowVar::Logit { .. }, SaeLocalRowVar::Coord { atom, axis }) => {
                        let coeff = dz[a][atom] * sqrt_row_w;
                        for out_col in 0..p {
                            second[a][b][out_col] = coeff * d1[atom][axis][out_col];
                        }
                    }
                    (SaeLocalRowVar::Coord { atom, axis }, SaeLocalRowVar::Logit { .. }) => {
                        let coeff = dz[b][atom] * sqrt_row_w;
                        for out_col in 0..p {
                            second[a][b][out_col] = coeff * d1[atom][axis][out_col];
                        }
                    }
                    (
                        SaeLocalRowVar::Coord {
                            atom: atom_a,
                            axis: axis_a,
                        },
                        SaeLocalRowVar::Coord {
                            atom: atom_b,
                            axis: axis_b,
                        },
                    ) if atom_a == atom_b => {
                        let coeff = assignments[atom_a] * sqrt_row_w;
                        for out_col in 0..p {
                            second[a][b][out_col] = coeff * d2[atom_a][axis_a][axis_b][out_col];
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut beta = vec![vec![0.0_f64; p]; border.len()];
        let mut beta_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
        let mut beta_l_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
        for (beta_pos, channel) in border.iter().enumerate() {
            let atom = channel.atom;
            let phi = self.atoms[atom].basis_values[[row, channel.basis_col]];
            let base = assignments[atom] * phi * sqrt_row_w;
            for out_col in 0..p {
                beta[beta_pos][out_col] = base * channel.output[out_col];
            }
            for (var_idx, var) in vars.iter().enumerate() {
                let scalar = match *var {
                    SaeLocalRowVar::Logit { .. } => dz[var_idx][atom] * phi * sqrt_row_w,
                    SaeLocalRowVar::Coord {
                        atom: coord_atom,
                        axis,
                    } if coord_atom == atom => {
                        assignments[atom]
                            * self.atoms[atom].basis_jacobian[[row, channel.basis_col, axis]]
                            * sqrt_row_w
                    }
                    _ => 0.0,
                };
                if scalar != 0.0 {
                    for out_col in 0..p {
                        beta_deriv[var_idx][beta_pos][out_col] = scalar * channel.output[out_col];
                    }
                }
                let scalar_l = match *var {
                    SaeLocalRowVar::Logit { .. } => {
                        dz[var_idx][atom]
                            * self.atoms[atom].basis_values[[row, channel.basis_col]]
                            * sqrt_row_w
                    }
                    SaeLocalRowVar::Coord {
                        atom: coord_atom,
                        axis,
                    } if coord_atom == atom => {
                        assignments[atom]
                            * self.atoms[atom].basis_jacobian[[row, channel.basis_col, axis]]
                            * sqrt_row_w
                    }
                    _ => 0.0,
                };
                if scalar_l != 0.0 {
                    for out_col in 0..p {
                        beta_l_deriv[var_idx][beta_pos][out_col] =
                            scalar_l * channel.output[out_col];
                    }
                }
            }
        }

        Ok(SaeRowJets {
            vars,
            first,
            second,
            beta,
            beta_deriv,
            beta_l_deriv,
        })
    }

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
                if !crate::terms::sae::assignment::jumprelu_in_optimization_band(
                    logit,
                    threshold,
                    temperature,
                ) {
                    return 0.0;
                }
                let inv_tau = 1.0 / temperature;
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                2.0 * rho.lambda_sparse()
                    * slope
                    * slope
                    * (1.0 - 2.0 * activation)
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
        } else if j == 1 {
            let lambda = rho.lambda_smooth();
            let frames_active = self.last_frames_active && cache.k == self.factored_border_dim();
            let offsets = if frames_active {
                self.factored_beta_offsets()
            } else {
                self.beta_offsets()
            };
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
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
                let off = offsets[atom_idx];
                for mu in 0..m {
                    for channel in 0..r {
                        let mut acc = 0.0_f64;
                        for nu in 0..m {
                            let s_sym = 0.5
                                * (atom.smooth_penalty[[mu, nu]] + atom.smooth_penalty[[nu, mu]]);
                            acc += s_sym * coeffs[[nu, channel]];
                        }
                        beta[off + mu * r + channel] = lambda * acc;
                    }
                }
            }
        } else {
            let mut cursor = 2usize;
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
        let mut beta_inv = Array2::<f64>::zeros((cache.k, cache.k));
        if cache.k > 0 {
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
        }
        // IBP `hessian_diag` logit third-derivative channels (#1006), exact for
        // the diagonal/quasi-Laplace assignment curvature this assembly actually
        // factors. The full IBP Hessian also has per-column cross-row rank-one
        // terms; those are omitted from H and therefore from this adjoint until
        // the evidence factor grows the matching Woodbury correction.
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
            crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                Some((
                    crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
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

        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let assignments = self.assignment.try_assignments_row_for_rho(row, rho)?;
            let jets = self.row_jets_for_logdet(
                rho,
                row,
                vars,
                assignments.view(),
                &second_jets,
                &border,
            )?;

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

            // Record each active logit's column, global t-index, and
            // selected-inverse diagonal (H⁻¹)_ik,ik for the IBP cross-row pass.
            if ibp_channels.is_some() {
                for (pos, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        ibp_logit_sites.push((row, atom, base + pos, inv_vv[[pos, pos]]));
                    }
                }
            }

            // #1038: when `w` is a logit and the assignment is softmax, the dense
            // entropy Hessian's full θ-derivative `∂H_{k,j}/∂z_w` (diagonal AND
            // off-diagonal) is the SAME `(a,L,m)`-derived tensor the assembly and
            // logdet use. Compute it once per logit `w` and add it at every logit
            // pair `(a,b)` below. The diagonal softmax case is therefore handled
            // here, NOT in `assignment_prior_hdiag_derivative_entry` (which returns
            // 0 for softmax to avoid double-counting).
            let row_logits_softmax: Option<Vec<f64>> = softmax_dense_adjoint.as_ref().map(|_| {
                (0..k_atoms)
                    .map(|k| self.assignment.logits[[row, k]])
                    .collect()
            });
            for w in 0..q {
                let mut gamma = 0.0_f64;
                let softmax_dh_w: Option<Array2<f64>> = match (
                    softmax_dense_adjoint.as_ref(),
                    row_logits_softmax.as_ref(),
                    jets.vars[w],
                ) {
                    (Some((penalty, scale)), Some(row_logits), SaeLocalRowVar::Logit { atom }) => {
                        Some(penalty.row_dense_hessian_logit_derivative(row_logits, *scale, atom))
                    }
                    _ => None,
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = sae_dot(&jets.second[a][w], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.second[b][w]);
                        if let (
                            Some(dh_w),
                            SaeLocalRowVar::Logit { atom: atom_a },
                            SaeLocalRowVar::Logit { atom: atom_b },
                        ) = (softmax_dh_w.as_ref(), jets.vars[a], jets.vars[b])
                        {
                            dh += dh_w[[atom_a, atom_b]];
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
                        gamma += inv_vv[[b, a]] * dh;
                    }
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
                for a in 0..q {
                    for b in 0..q {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.beta_l_deriv[b][w_beta_pos]);
                        gamma += inv_vv[[b, a]] * dh;
                    }
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
        }

        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
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
    ) -> Result<SaeOuterRhoGradientComponents, String> {
        let n_params = rho.to_flat().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);

        explicit[0] = assignment_prior_log_strength_derivative(&self.assignment, rho)
            + self.learnable_ibp_forward_alpha_data_derivative(rho, target)?;
        logdet_trace[0] = self.assignment_log_strength_hessian_trace(rho, cache, solver)?;

        explicit[1] = loss.smoothness;
        logdet_trace[1] = 0.5
            * self
                .decoder_smoothness_effective_dof_with_solver(cache, solver, rho.lambda_smooth())
                .map_err(|err| format!("analytic_outer_rho_gradient_components: {err}"))?;
        occam[1] = -self.reml_occam_log_lambda_smooth_derivative()?;

        let ard_explicit = self.ard_log_precision_explicit_derivatives(rho)?;
        let ard_trace = self
            .ard_log_precision_hessian_trace(rho, cache, solver)
            .map_err(|err| format!("analytic_outer_rho_gradient_components: {err}"))?;
        let mut cursor = 2usize;
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                explicit[cursor] = ard_explicit[k][axis];
                logdet_trace[cursor] = ard_trace[k][axis];
                cursor += 1;
            }
        }

        let gamma = self.logdet_theta_adjoint(rho, cache, solver)?;
        for coord in 0..n_params {
            let rhs = self.outer_rho_gradient_ift_rhs(rho, target, coord, cache)?;
            let solved = solver.solve(rhs.t.view(), rhs.beta.view()).map_err(|err| {
                format!("analytic_outer_rho_gradient_components: full_inverse_apply: {err}")
            })?;
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
            third_order_correction_available: true,
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
        let solver = self.outer_gradient_arrow_solver(cache)?;
        self.analytic_outer_rho_gradient_components(target, rho, loss, cache, &solver)
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

        let solver = self.outer_gradient_arrow_solver(&cache)?;
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

    /// Gaussian reconstruction dispersion `φ̂`, the scale that turns the
    /// unscaled inverse-Hessian β-block `S_β⁻¹` into a posterior covariance
    /// `Cov(β) = φ̂·S_β⁻¹` — the same `Vb = φ·H⁻¹` convention the main GAM
    /// inference path uses.
    ///
    /// `RSS = Σ_{i,c} (z_{ic} − ẑ_{ic})² = 2·data_fit` (the loss stores the
    /// half-sum `½Σr²`). The residual degrees of freedom subtract the effective
    /// parameter count from the `N·p` scalar observations:
    ///   * decoder β: `beta_dim − tr(λ_smooth · S_β⁻¹ · ⊕_k S_k⊗I_p)`, the
    ///     smoothness effective-dof already assembled for the Fellner-Schall
    ///     step (penalty-shrunk directions do not cost a full parameter);
    ///   * latent coordinates: enabled ARD axes use the exact ARD-shrunk trace
    ///     `Σ_k Σ_j (n_active_k − α_{kj}·tr_{kj}(H⁻¹))`; atoms with disabled
    ///     native ARD charge the full active coordinate count because those
    ///     latent variables are estimated without an ARD precision.
    ///
    /// The coordinate term is the **exact** ARD-shrunk effective dof of the
    /// latent block: along axis `(k,j)` the MacKay/Fellner-Schall edf is
    /// `n_active_k − α_{kj}·tr_{kj}(H⁻¹)`, the well-determined-direction count
    /// after the ARD prior `α_{kj}` shrinks each coordinate. `tr_{kj}(H⁻¹)` is
    /// the same posterior-variance trace [`Self::ard_inverse_traces`] assembles
    /// for the EFS ARD step (reused here, not recomputed), so the dispersion is
    /// consistent with the precision update `α_new = n/(‖t‖²+tr(H⁻¹))`. The
    /// per-axis scalar count `n_active_k` must match the support the trace sums
    /// over: `n` for the dense full-support layout, or the number of rows where
    /// atom `k` is active for the compact active-set layout (inactive
    /// prior-dominated coordinates contribute 0 to both the trace and the
    /// count, hence 0 edf). The residual dof is floored at 1 so `φ̂` stays
    /// finite and positive.
    pub(crate) fn reconstruction_dispersion(
        &self,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let n_scalar = (n * p) as f64;
        let rss = 2.0 * loss.data_fit;
        let smooth_edf = self
            .decoder_smoothness_effective_dof(cache, rho.lambda_smooth())
            .map_err(|e| format!("reconstruction_dispersion: smooth edf: {e}"))?;
        // #972 / #977 T1: the raw decoder-parameter count is `beta_dim` on the
        // full-`B` path, but when frames are active the estimated decoder freedom
        // is the factored border `Σ M_k·r_k` PLUS the `Σ r_k·(p−r_k)` Grassmann
        // frame degrees profiled out (both are genuinely estimated), which the
        // smoothness shrinkage `smooth_edf` (taken over the factored border) then
        // discounts. On the full-`B` path `factored_border_dim == beta_dim` and
        // `grassmann_evidence_dimension == 0`, so this is exactly `beta_dim`.
        let raw_decoder_dof = if self.frames_active() {
            (self.factored_border_dim() + self.grassmann_evidence_dimension()) as f64
        } else {
            self.beta_dim() as f64
        };
        let beta_edf = (raw_decoder_dof - smooth_edf).max(0.0);
        // Exact ARD-shrunk latent-coordinate edf, reusing the EFS trace cache.
        let traces = self
            .ard_inverse_traces(cache)
            .map_err(|e| format!("reconstruction_dispersion: ARD traces: {e}"))?;
        if rho.log_ard.len() != self.atoms.len() {
            return Err(format!(
                "reconstruction_dispersion: ρ has {} ARD atoms but term has {}",
                rho.log_ard.len(),
                self.atoms.len()
            ));
        }
        let mut coord_edf = 0.0_f64;
        for (k, atom) in self.atoms.iter().enumerate() {
            let d_k = atom.latent_dim;
            if traces[k].len() != d_k {
                return Err(format!(
                    "reconstruction_dispersion: trace shape mismatch at atom {k} \
                     (traces={}, d_k={d_k})",
                    traces[k].len()
                ));
            }
            let ard_len = rho.log_ard[k].len();
            if ard_len != 0 && ard_len != d_k {
                return Err(format!(
                    "reconstruction_dispersion: ARD shape mismatch at atom {k} \
                     (log_ard={ard_len}, d_k={d_k})"
                ));
            }
            // Scalar count matched to the trace support (see fn doc).
            let n_active_k = match self.last_row_layout {
                Some(ref layout) => layout
                    .active_atoms
                    .iter()
                    .filter(|active| active.contains(&k))
                    .count() as f64,
                None => n as f64,
            };
            if ard_len == 0 {
                coord_edf += n_active_k * d_k as f64;
                continue;
            }
            for j in 0..d_k {
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][j]);
                // edf_kj ∈ [0, n_active_k]; clamp against numerical drift.
                let edf_kj = (n_active_k - alpha * traces[k][j]).clamp(0.0, n_active_k);
                coord_edf += edf_kj;
            }
        }
        let resid_dof = (n_scalar - beta_edf - coord_edf).max(1.0);
        let phi = rss / resid_dof;
        if !phi.is_finite() || phi < 0.0 {
            return Err(format!(
                "reconstruction_dispersion: non-finite/negative φ̂={phi} \
                 (RSS={rss}, resid_dof={resid_dof}, beta_edf={beta_edf}, coord_edf={coord_edf})"
            ));
        }
        Ok(phi.max(f64::MIN_POSITIVE))
    }

    /// Posterior covariance and ambient shape band for every atom — the
    /// user-facing uncertainty of the fitted manifold shapes.
    ///
    /// For atom `k` with decoder-block range `r_k` (see
    /// [`Self::beta_block_offsets`]), `Cov(β_k) = φ·S_β⁻¹[r_k, r_k]` is the
    /// φ-scaled posterior covariance of its decoder coefficients with the
    /// latent coordinates marginalized out. The ambient point at a coordinate
    /// `t` is `m_k(t) = Φ_k(t)·B_k`, *linear* in `β_k`, so its per-channel
    /// posterior variance is the closed form
    /// `Var_c(t) = Σ_{b1,b2} Φ_k(t)[b1] Φ_k(t)[b2] · Cov(β_k)[(b1,c),(b2,c)]`
    /// — no sampling. The band is evaluated at up to [`SHAPE_BAND_MAX_POINTS`]
    /// evenly-strided of the atom's own on-atom coordinates, reusing the basis
    /// values already stored on the atom, so it reports uncertainty exactly
    /// where the data lives and needs no basis-kind-specific grid.
    ///
    /// A near-degenerate atom has a near-singular Schur block, so `Cov(β_k)` —
    /// and the band — fans out automatically: the band width is a
    /// per-coordinate visual of how well each atom is identified.
    pub fn assemble_shape_uncertainty(
        &self,
        cache: &ArrowFactorCache,
        dispersion: f64,
    ) -> Result<SaeShapeUncertainty, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache β block is the FACTORED border when frames
        // are active, so each atom's Schur inverse block is the `(M_k·r_k)`
        // coordinate covariance `Cov(vec C_k)`. We LIFT it to the full
        // `(M_k·p)` decoder covariance `Cov(vec B_k) = (I_{M_k} ⊗ U_k) Cov(vec
        // C_k)(I_{M_k} ⊗ U_k)ᵀ` (since `B_k = C_k U_kᵀ`) so the downstream band
        // code — which reads the `b·p + c` flat layout — is unchanged. On the
        // full-`B` path the block is already `(M_k·p)` and the lift is skipped.
        let frames_active = self.frames_active();
        let frame_projection = FrameProjection::new(self);
        let block_ranges = if frames_active {
            (0..self.k_atoms())
                .map(|k| frame_projection.atom_border_range(k))
                .collect::<Vec<_>>()
        } else {
            self.beta_block_offsets().to_vec()
        };
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let cov_block = cache
                .schur_inverse_block(block_ranges[k].clone())
                .map_err(|e| format!("assemble_shape_uncertainty: atom {k}: {e}"))?;
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            // Evenly-strided evaluation rows bound the band cost.
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let mut band_sd = Array2::<f64>::zeros((g, p));
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }

            let framed = frames_active && atom.decoder_frame.is_some();
            let dense_entries = (m * p).saturating_mul(m * p);
            let cov = if framed && dense_entries > SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES {
                // LLM-scale ambient `p`: the dense `(M_k·p)²` lift would be
                // gigabytes per atom and exists only to export the full
                // covariance. Compute the band variance EXACTLY from the
                // factored frame covariance instead: with `B_k = C_k·U_kᵀ`,
                //   Var_c(t) = (φ ⊗ u_c)ᵀ Cov(vec C_k) (φ ⊗ u_c)
                // which is the r×r quadratic form `u_cᵀ Y u_c` with
                //   Y = Σ_{b1,b2} φ[b1] φ[b2] Cov(C)[(b1,·),(b2,·)].
                let mut cov_c = cov_block;
                cov_c.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    let basis = atom.basis_values.row(row);
                    for c in 0..p {
                        let var = frame_projection.output_variance(k, cov_c.view(), basis, c);
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                None
            } else {
                // Lift the factored `(M_k·r_k)` coordinate covariance to the
                // full `(M_k·p)` decoder covariance through this atom's frame;
                // identity (a plain scaled copy) on the un-framed full-`B` path.
                let mut cov = if framed {
                    frame_projection.lift_block(k, cov_block.view())
                } else {
                    cov_block
                };
                cov.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    // Var_c = Σ_{b1,b2} Φ[b1]Φ[b2] Cov[(b1,c),(b2,c)]; the flat
                    // decoder index is basis·p + channel (row-major (M_k, p)).
                    for c in 0..p {
                        let var = frame_projection.full_output_variance(
                            k,
                            cov.view(),
                            atom.basis_values.row(row),
                            c,
                        );
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                Some(cov)
            };
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: cov,
                band_coords,
                band_mean,
                band_sd,
            });
        }
        Ok(SaeShapeUncertainty { dispersion, atoms })
    }

    /// #977 — complete the per-atom shape band for any atom the pre-search
    /// Schur factor could not cover (a structure-search-BORN atom, whose index
    /// is ≥ the seed `K` the Schur cache was assembled at), from that atom's OWN
    /// fitted penalized inner Hessian.
    ///
    /// The Schur path ([`Self::assemble_shape_uncertainty`]) reads the joint
    /// inverse-Hessian β-block per atom, but that factor is assembled ONCE before
    /// the structure search runs, so it is indexed by the SEED dictionary. A born
    /// atom therefore has no Schur block and would otherwise be reported with NO
    /// uncertainty band — a silent gap. This method closes it: every atom carries
    /// a band, none is reported without one.
    ///
    /// The principled per-atom band is the Laplace posterior of the atom's inner
    /// reconstruction smooth, which [`Self::set_atom_inner_fits`] already fits at
    /// the settled state for EVERY atom (born included). With the Gaussian-identity
    /// inner smooth, each output channel `c`'s decoder posterior is
    /// `Cov(β_{k,c}) = φ · H_k⁻¹`, where `H_k = Φ_kᵀ W_k Φ_k + S̃_k` is the atom's
    /// fitted penalized inner Hessian (`AtomInnerFit::penalized_hessian`). The
    /// ambient point `m_k(t) = Φ_k(t)·B_k` is linear in `B_k`, so its per-channel
    /// posterior variance is the closed form
    ///   `Var_c(t) = φ · Φ_k(t)ᵀ H_k⁻¹ Φ_k(t)`,
    /// which is the SAME for every channel `c` (the inner Hessian is shared across
    /// channels; the decoder differs only in the mean). The band is evaluated at
    /// the same evenly-strided on-atom coordinate subset the Schur path uses, so a
    /// born atom's band is reported exactly where its data lives.
    ///
    /// This is a strict completion: an atom whose band the Schur path already
    /// filled (a finite `band_sd`) is left untouched; only atoms with a missing
    /// entry (index past the assembled set) or an all-NaN band (the
    /// no-decoder-covariance fallback) are filled. An atom whose inner fit is
    /// degenerate (`None` — no active rows / non-SPD inner Hessian) is left with
    /// its NaN band, faithfully reporting "unidentified" rather than fabricating a
    /// number. Requires [`Self::set_atom_inner_fits`] to have run; without it the
    /// completion is a no-op (the band stays as the Schur path left it).
    pub fn complete_born_atom_shape_bands(
        &self,
        unc: &mut SaeShapeUncertainty,
    ) -> Result<(), String> {
        let inner_fits = match &self.atom_inner_fits {
            Some(fits) => fits,
            // No inner fits harvested: nothing to complete from. Leave the bands
            // as the Schur path produced them.
            None => return Ok(()),
        };
        let p = self.output_dim();
        let dispersion = unc.dispersion;
        // Grow the per-atom band list to the post-search atom count so a born
        // atom (index past the Schur-assembled set) has a slot. New slots start
        // as NaN bands and are filled below from the inner fit.
        while unc.atoms.len() < self.k_atoms() {
            let k = unc.atoms.len();
            let atom = &self.atoms[k];
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let band_sd = Array2::<f64>::from_elem((g, p), f64::NAN);
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }
            unc.atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords,
                band_mean,
                band_sd,
            });
        }

        for (k, atom) in self.atoms.iter().enumerate() {
            let band = &mut unc.atoms[k];
            // Only complete a MISSING band: an atom the Schur path already filled
            // (a finite sd anywhere) keeps its joint-Hessian band untouched.
            let already_filled = band.band_sd.iter().any(|v| v.is_finite());
            if already_filled {
                continue;
            }
            let inner = match inner_fits.get(k).and_then(|f| f.as_ref()) {
                Some(f) => f,
                // Degenerate atom (no active rows / non-SPD inner Hessian): leave
                // the NaN band — honestly "unidentified", never a fabricated band.
                None => continue,
            };
            let m = atom.basis_size();
            if inner.penalized_hessian.dim() != (m, m) {
                return Err(format!(
                    "complete_born_atom_shape_bands: atom {k} inner Hessian {:?} != ({m}, {m})",
                    inner.penalized_hessian.dim()
                ));
            }
            // Factor the atom's own penalized inner Hessian H_k = ΦᵀWΦ + S̃_k. It
            // was checked SPD when the inner fit was built; re-factor here to solve
            // H_k⁻¹ Φ(t). A factorization failure (numerical drift since the inner
            // fit) leaves the NaN band rather than a fabricated number.
            let chol = match inner.penalized_hessian.cholesky(Side::Lower) {
                Ok(c) => c,
                Err(_) => continue,
            };
            // Evenly-strided on-atom rows, matched to the band the Schur path uses.
            let n_rows = atom.n_obs();
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            for (gi, &row) in eval_rows.iter().enumerate() {
                // Φ_k(t) at this on-atom row.
                let phi_t = atom.basis_values.row(row).to_owned();
                // H_k⁻¹ Φ(t), then the quadratic form Φ(t)ᵀ H_k⁻¹ Φ(t).
                let solved = chol.solvevec(&phi_t);
                let quad = phi_t.dot(&solved).max(0.0);
                // Var_c(t) = φ · Φ(t)ᵀ H_k⁻¹ Φ(t) — identical across channels (the
                // inner Hessian is shared; the decoder differs only in the mean).
                let sd = (dispersion * quad).sqrt();
                for c in 0..p {
                    band.band_sd[[gi, c]] = sd;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn shape_uncertainty_without_decoder_covariance(
        &self,
        dispersion: f64,
    ) -> SaeShapeUncertainty {
        let p = self.output_dim();
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let band_sd = Array2::<f64>::from_elem((g, p), f64::NAN);
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords,
                band_mean,
                band_sd,
            });
        }
        SaeShapeUncertainty { dispersion, atoms }
    }
}

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

    // J[n, i*d + a] = Σ_m dPhi[n, m, a] · B[m, i].
    let b = &atom.decoder_coefficients;
    let mut jac = Array2::<f64>::zeros((n_obs, p * d));
    for n in 0..n_obs {
        for i in 0..p {
            for a in 0..d {
                let mut acc = 0.0;
                for mm in 0..m {
                    acc += jet[[n, mm, a]] * b[[mm, i]];
                }
                jac[[n, i * d + a]] = acc;
            }
        }
    }

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
        let mut jac2 = Array2::<f64>::zeros((n_obs, p * d * d));
        for n in 0..n_obs {
            for i in 0..p {
                for a in 0..d {
                    for c in 0..d {
                        let mut acc = 0.0;
                        for mm in 0..m {
                            acc += hess[[n, mm, a, c]] * b[[mm, i]];
                        }
                        jac2[[n, (i * d + a) * d + c]] = acc;
                    }
                }
            }
        }
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
                let mut jac3 = Array3::<f64>::zeros((n_obs, p, d * d * d));
                for n in 0..n_obs {
                    for i in 0..p {
                        for a in 0..d {
                            for c in 0..d {
                                for e in 0..d {
                                    let mut acc = 0.0;
                                    for mm in 0..m {
                                        acc += t3[[n, mm, a, c, e]] * b[[mm, i]];
                                    }
                                    jac3[[n, i, ((a * d) + c) * d + e]] = acc;
                                }
                            }
                        }
                    }
                }
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

#[cfg(test)]
mod amortized_encoder_tests {
    use crate::terms::sae::manifold::tests::small_two_atom_periodic_term;

    /// #1026 ladder item 2/3 — the amortized encoder is reachable end-to-end
    /// from a fitted term and is certificate-honest: it encodes the dictionary's
    /// own fit-time target, returns one result per atom with the right shape, and
    /// every row is either certified or counted in
    /// `encode_uncertified_count` (never silently miscounted), with the exact
    /// fallback strictly reducing the uncertified count it inherits.
    #[test]
    fn amortized_encode_fitted_is_reachable_and_certificate_honest() {
        let (term, target, rho) = small_two_atom_periodic_term();
        let n = term.n_obs();
        let k = term.k_atoms();

        let results = term
            .amortized_encode_fitted(target.view(), &rho)
            .expect("amortized encode of the fit-time target runs end-to-end");
        assert_eq!(
            results.len(),
            k,
            "one encode result per atom in dictionary order"
        );

        for (atom_idx, result) in results.iter().enumerate() {
            assert_eq!(
                result.coords.nrows(),
                n,
                "atom {atom_idx} encode must produce one coordinate per row"
            );
            assert_eq!(
                result.coords.ncols(),
                term.atoms[atom_idx].latent_dim,
                "atom {atom_idx} encode coords must match its latent dim"
            );
            // The uncertified count is the honest tally of rows the certificate
            // could not gate — it must equal the false entries of the mask.
            let uncertified = result.certified.iter().filter(|c| !**c).count();
            assert_eq!(
                result.encode_uncertified_count, uncertified,
                "atom {atom_idx} uncertified count must match the certificate mask"
            );
            assert_eq!(
                result.certified.len(),
                n,
                "atom {atom_idx} certificate mask must cover every row"
            );
        }
    }

    /// The fitted amplitudes the encoder derives are exactly the assignment
    /// masses the reconstruction is assembled from — feeding them back is the
    /// self-consistency the distilled map is supervised against.
    #[test]
    fn fitted_assignment_amplitudes_match_the_assignment_masses() {
        let (term, _target, rho) = small_two_atom_periodic_term();
        let n = term.n_obs();
        let k = term.k_atoms();
        let amplitudes = term
            .fitted_assignment_amplitudes(&rho)
            .expect("fitted amplitudes derive from the assignment");
        assert_eq!(amplitudes.dim(), (n, k));
        for row in 0..n {
            let a = term
                .assignment
                .try_assignments_row_for_rho(row, &rho)
                .expect("assignment row resolves");
            for atom_idx in 0..k {
                assert_eq!(
                    amplitudes[[row, atom_idx]],
                    a[atom_idx],
                    "amplitude[{row},{atom_idx}] must equal the assignment mass"
                );
            }
        }
    }
}
